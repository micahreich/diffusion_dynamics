import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler  # use the DDPM noise scheduler from diffusers
from tqdm import tqdm
from diffusers.models.resnet import ResidualTemporalBlock1D, Conv1dBlock
from diffusers.models.unets.unet_1d_blocks import DownResnetBlock1D, UpResnetBlock1D, DownBlock1D, UpBlock1D
from diffusers.models.unets.unet_1d import UNet1DModel
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.downsampling import Downsample1D
from diffusers.models.upsampling import Upsample1D
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import LayerNorm
from datetime import datetime
import pytz
import os
import pickle
import einops
from typing import Optional, Callable
from diffusion_dynamics.models.helpers import Residual, PreNorm, LinearAttention
from diffusion_dynamics.models.utils import NumpyDataset1D
from einops.layers.torch import Rearrange


class UNet1D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_channels=64,
                 dim_mults=[1, 2, 4],
                 attention=True):
        super().__init__()
        
        dims = [in_channels, *map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
                
        self.time_embed_dim = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_embed_dim * 4, self.time_embed_dim),
        )
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock1D(dim_in, dim_out, embed_dim=self.time_embed_dim, kernel_size=5),
                ResidualTemporalBlock1D(dim_out, dim_out, embed_dim=self.time_embed_dim, kernel_size=5),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, heads=4, dim_head=32))) if attention else nn.Identity(),
                Downsample1D(dim_out, use_conv=True) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock1D(mid_dim, mid_dim, embed_dim=self.time_embed_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, heads=4, dim_head=32))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock1D(mid_dim, mid_dim, embed_dim=self.time_embed_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock1D(dim_out * 2, dim_in, embed_dim=self.time_embed_dim, kernel_size=5),
                ResidualTemporalBlock1D(dim_in, dim_in, embed_dim=self.time_embed_dim, kernel_size=5),
                Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=4, dim_head=32))) if attention else nn.Identity(),
                Upsample1D(dim_in, use_conv_transpose=True) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, 5, padding=5//2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(8, base_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Conv1d(base_channels, out_channels, 1),
        )

    def forward(self, x, t):
        t_embedded = get_timestep_embedding(t, self.time_embed_dim)
        t = self.time_mlp(t_embedded)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)        
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        return x


class UNet1DModel:
    model = None
    scheduler = None
    n_channels = None
    loaded_model_name = None

    @classmethod
    def train(cls,
              dataset: NumpyDataset1D,
              n_epochs=100,
              batch_size=64,
              learning_rate=1e-4,
              save_model_params=None):
        assert cls.model is not None, "model must be instantiated before training"
        assert cls.scheduler is not None, "noise scheduler must be instantiated before training"
        assert cls.n_channels is not None, "number of channels must be set before training"
        
        if save_model_params is not None:
            assert "save_fpath" in save_model_params, "model save filepath must be provided"
        
        trainable_params = sum(p.numel() for p in cls.model.parameters() if p.requires_grad)
        print(f"Training {cls.model.__class__.__name__} with {trainable_params} trainable parameters")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ddpm_scheduler_params = {
            "num_train_timesteps": cls.scheduler.config.num_train_timesteps
        }
        
        dataset_params = {
            "n_samples": dataset.n_samples,
            "seq_length": dataset.seq_len,
            "n_channels": dataset.n_channels,
            "normalize_data": dataset.normalize_data,
            "data_mean": dataset.data_mean,
            "data_std": dataset.data_std
        }
    
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Instantiate our 1D UNet diffusion model
        cls.model.to(device)
        
        optimizer = torch.optim.Adam(cls.model.parameters(), lr=learning_rate)

        cls.model.train()
        
        try:
            for epoch in range(n_epochs):
                pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
                
                for step, batch in enumerate(pbar):
                    batch = batch.to(device)  # shape (B, 1, seq_length)
                    # Sample a random timestep for each example in the batch.
                    t = torch.randint(0, cls.scheduler.config.num_train_timesteps, (batch.shape[0],), device=device).long()
                    noise = torch.randn_like(batch)
                    # Add noise to the clean batch at the given timesteps.
                    noisy_batch = cls.scheduler.add_noise(batch, noise, t)
                    # Predict the noise that was added.
                    noise_pred = cls.model(noisy_batch, t)
                                
                    loss = F.mse_loss(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update the tqdm progress bar with the current loss.
                    pbar.set_postfix(loss=loss.item())
        except KeyboardInterrupt:
            print("\nTraining interrupted. Do you want to save the model? (y/n): ", end="")
            response = input().strip().lower()
            if response == 'n':
                return
        
        cls._save_model(save_model_params, ddpm_scheduler_params, dataset_params)
    
    @classmethod
    def _save_model(cls, save_model_params, ddpm_scheduler_params, dataset_params):
        assert save_model_params is not None, "save_model_params must be provided"
        
        # Save the trained model weights
        if "save_model_name" not in save_model_params:
            nyc_tz = pytz.timezone('America/New_York')
            time_str = datetime.now(nyc_tz).strftime("%Y-%m-%d__%H-%M-%S")
            save_model_params["save_model_name"] = f"unet1d_{time_str}"
        
        save_fpath_full = os.path.join(save_model_params["save_fpath"], save_model_params["save_model_name"])
        print("Saving model to", save_fpath_full)        
        os.makedirs(save_fpath_full, exist_ok=True)

        torch.save(cls.model.state_dict(), os.path.join(save_fpath_full, "model.pt"))
        
        with open(os.path.join(save_fpath_full, "params.pkl"), "wb") as f:
            combined_dict = {
                "ddpm_scheduler_params": ddpm_scheduler_params,
                "dataset_params": dataset_params
            }
            pickle.dump(combined_dict, f)
                
    @classmethod
    def sample(cls, n_samples, seq_length, g: Optional[Callable] = None) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        cls.model.to(device)
        cls.model.eval()
    
        with torch.no_grad():
            # Start from random Gaussian noise
            sample = torch.randn((n_samples, cls.n_channels, seq_length), device=device)
            # scheduler.timesteps is an iterable of timesteps in descending order
            for t in cls.scheduler.timesteps:
                # For each diffusion step, create a batch of the current timestep
                t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
                # Predict the noise residual
                noise_pred = cls.model(sample, t_batch)
                # Compute the previous sample (one denoising step)
                sample = cls.scheduler.step(noise_pred, t, sample)["prev_sample"]
                
                if g is not None:
                    sample = g(sample, t)
                    
            return sample
    
    @classmethod
    def load_trained_model(cls, saved_model_fpath):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(os.path.join(saved_model_fpath, "model.pt"), map_location=device)
        cls.model.load_state_dict(state_dict)
        cls.loaded_model_name = os.path.basename(saved_model_fpath)
        
        with open(os.path.join(saved_model_fpath, "params.pkl"), "rb") as f:
            combined_dict = pickle.load(f)
            
            dataset_params = combined_dict["dataset_params"]
            return dataset_params

