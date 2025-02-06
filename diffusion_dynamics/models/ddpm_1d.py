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
from diffusion_dynamics.models.utils import TensorDataset1D, TensorDataset1DStats
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
    def __init__(self, unet=None, scheduler=None, n_channels=None):
        self.unet = unet
        self.scheduler = scheduler
        self.n_channels = n_channels
        
        self.train_data_stats = None
    
    def train(self,
              dataset: TensorDataset1D,
              n_epochs=100,
              batch_size=64,
              learning_rate=1e-4,
              save_model_params=None,
              initial_conditioning_channel_idx=[],
              condition_controls_idx=[]):
        assert self.unet is not None, "model must be instantiated before training"
        assert self.scheduler is not None, "noise scheduler must be instantiated before training"
        assert self.n_channels is not None, "number of channels must be set before training"
        
        if save_model_params is not None:
            assert "save_fpath" in save_model_params, "model save filepath must be provided"
        
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f"Training {self.unet.__class__.__name__} with {trainable_params} trainable parameters")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Instantiate our 1D UNet diffusion model
        self.unet.to(device)
        self.unet.train()

        optimizer = torch.optim.Adam(self.unet.parameters(), lr=learning_rate)
        
        try:
            for epoch in range(n_epochs):
                pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
                
                for step, batch in enumerate(pbar):
                    batch = batch.to(device)  # shape (batch_size, n_channels, seq_length)

                    # Randomly sample timestep, add noise to batch
                    t = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch.shape[0],), device=device).long()
                    noise = torch.randn_like(batch)
                    noisy_batch = self.scheduler.add_noise(batch, noise, t)
                    
                    noisy_batch[:, initial_conditioning_channel_idx, 0] = batch[:, initial_conditioning_channel_idx, 0]
                    noisy_batch[:, condition_controls_idx, :] = batch[:, condition_controls_idx, :]
                    
                    # Predict added noise and perform backward pass
                    noise_pred = self.unet(noisy_batch, t)            
                    loss = F.mse_loss(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=loss.item())
        except KeyboardInterrupt:
            print("\nTraining interrupted. Do you want to save the model? (y/n): ", end="")
            response = input().strip().lower()
            if response == 'n':
                return
        
        self._save_model(save_model_params, dataset)
    
    def _save_model(self, save_model_params, train_dataset: TensorDataset1D):
        assert save_model_params is not None, "save_model_params must be provided"
        
        # Save the trained model weights
        if "save_model_name" not in save_model_params:
            nyc_tz = pytz.timezone('America/New_York')
            time_str = datetime.now(nyc_tz).strftime("%Y-%m-%d__%H-%M-%S")
            save_model_params["save_model_name"] = f"unet1d_{time_str}"
        
        save_fpath_full = os.path.join(save_model_params["save_fpath"], save_model_params["save_model_name"])
        print("Saving model to", save_fpath_full)        
        os.makedirs(save_fpath_full, exist_ok=True)

        torch.save(self.unet.state_dict(), os.path.join(save_fpath_full, "model.pt"))
        train_dataset.stats.save(os.path.join(save_fpath_full, "dataset_stats.pt"))
    
    # def sample(self, n_samples, seq_length, g: Optional[Callable] = None) -> torch.Tensor:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #     self.unet.to(device)
    #     self.unet.eval()
    
    #     with torch.no_grad():
    #         # Start from random Gaussian noise
    #         sample = torch.randn((n_samples, self.n_channels, seq_length), device=device)
    #         # scheduler.timesteps is an iterable of timesteps in descending order
    #         for t in self.scheduler.timesteps:
    #             # For each diffusion step, create a batch of the current timestep
    #             t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
    #             noise_pred = self.unet(sample, t_batch)
    #             sample = self.scheduler.step(noise_pred, t, sample)["prev_sample"]
                
    #             if g is not None:
    #                 sample = g(sample, t)
                    
    #         return sample
    
    @classmethod
    def load_trained_model(cls, saved_model_fpath):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(os.path.join(saved_model_fpath, "model.pt"), map_location=device)
        
        m = cls()
        assert m.unet is not None, "model must be instantiated before loading a trained model"
        
        m.unet.load_state_dict(state_dict)
        m.train_data_stats = TensorDataset1DStats.load(os.path.join(saved_model_fpath, "dataset_stats.pt"))
        
        return m        

