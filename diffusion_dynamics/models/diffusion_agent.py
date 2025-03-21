import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from diffusion_dynamics.models.utils import SaveModelParams, TensorDataset1D, TensorDataset1DStats
from diffusion_dynamics.utils import np_sigmoid, np_logit
from dataclasses import dataclass, asdict
from typing import Tuple
import pytz
from datetime import datetime
import os
from tqdm import tqdm
from typing import Optional


class ConditionalDiffusionAgent:
    def __init__(self, model=None, scheduler=None):
        self.model = model
        self.scheduler = scheduler
        self.stats: Optional[TensorDataset1DStats] = None
        
    def to(self, device) -> "ConditionalDiffusionAgent":
        if self.model is not None:
            self.model = self.model.to(device)

        return self
        
    def train(
        self,
        dataset: Dataset,
        n_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        accumulation_steps=2,
        save_model_params: Optional[SaveModelParams]=None,
    ):
        assert self.model is not None, "model must be instantiated before training"
        assert self.scheduler is not None, "noise scheduler must be instantiated before training"
        self.stats = dataset.stats
                
        print(f"Testing {self.model.__class__.__name__} forward pass...")
        
        self.model.eval()
        
        with torch.no_grad():
            u_noisy = torch.randn(batch_size, dataset.stats.u_pred_len, dataset.stats.nu)
            
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,)).long()
            cond = torch.randn(batch_size, dataset.stats.obs_history_len * dataset.stats.nx)

            out = self.model(u_noisy, t, cond)
    
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Training {self.model.__class__.__name__} with {trainable_params} trainable parameters")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        _, _u = dataset[0]
        u_pred_len, nu = _u.shape
        
        assert u_pred_len == dataset.stats.u_pred_len, "dataset u_pred_len must match the model's expected u_pred_len"
        
        # Instantiate our 1D UNet diffusion model
        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)
        
        try:
            for epoch in range(n_epochs):
                pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
                epoch_loss = 0.0
                # current_lr = optimizer.param_groups[0]['lr']

                for step, batch in enumerate(pbar):
                    # Randomly sample timestep, add noise to batch
                    obs_hist, u_hist = batch
                    obs_hist = obs_hist.to(device)
                    u_hist = u_hist.to(device)
                    
                    t = torch.randint(
                        0, self.scheduler.config.num_train_timesteps, (u_hist.shape[0],), device=device
                    ).long()
                    
                    noise = torch.randn_like(u_hist)
                    noisy_u_hist = self.scheduler.add_noise(u_hist, noise, t)

                    # Predict added noise and perform backward pass
                    model_out = self.model(noisy_u_hist, t, obs_hist)

                    # if self.scheduler.config.prediction_type == "epsilon":
                    loss = F.mse_loss(model_out, noise)
                    loss.backward()
                    
                    if (step + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=f"{round(loss.item(), 4):.4f}", lr=learning_rate)                
                
                epoch_loss /= len(dataloader)
                # lr_scheduler.step(epoch_loss)

        except KeyboardInterrupt:
            if save_model_params is not None:
                print("\nTraining interrupted. Do you want to save the model? (y/n): ", end="")
                response = input().strip().lower()
                if response == 'n':
                    return

        if save_model_params is not None:
            self._save_model(save_model_params)

    def _save_model(self, save_model_params: SaveModelParams):
        # Save the trained model weights
        if save_model_params.save_model_name is None:
            nyc_tz = pytz.timezone('America/New_York')
            time_str = datetime.now(nyc_tz).strftime("%Y-%m-%d__%H-%M-%S")
            save_model_params.save_model_name = f"diffusion1d_{time_str}"

        save_fpath_full = os.path.join(save_model_params.save_full_fpath, save_model_params.save_model_name)
        root, ext = os.path.splitext(save_fpath_full)
        
        if ext and ext == ".pt":
            save_fpath_full = root
        elif ext and ext != ".pt":
            save_fpath_full = f"{root}.pt"
        else:
            save_fpath_full = f"{save_fpath_full}.pt"
        
        print(f"Saving model to {save_fpath_full}...")
        torch.save(self, save_fpath_full)

    @classmethod
    def load_trained_model(cls, saved_model_path: str) -> "ConditionalDiffusionAgent":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.load(saved_model_path, map_location=device)
    
    def sample(self, cond, device, num_inference_steps=None):
        n_samples = cond.shape[0]
        sample = torch.randn((n_samples, self.stats.u_pred_len, self.stats.nu), device=device)
        cond = cond.to(device)
        
        assert cond.shape == (n_samples, self.stats.nx * self.stats.obs_history_len), \
            f"cond shape {cond.shape} must be (n_samples, nx * obs_history_len) = ({n_samples}, {self.stats.nx * self.stats.obs_history_len})"
        
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        
        # scheduler.timesteps is an iterable of timesteps in descending order
        for t in self.scheduler.timesteps:    
            with torch.no_grad():
                # For each diffusion step, create a batch of the current timestep
                t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
                
                # Predict the noise residual
                model_out = self.model(sample, t_batch, cond)
                
                # Compute the previous sample (one denoising step)
                sample = self.scheduler.step(model_out, t, sample)["prev_sample"]
        
        if self.stats.u_mean and self.stats.u_std:
            sample = TensorDataset1D.unnormalize(sample, self.stats.u_min, self.stats.u_max)
        
        return sample