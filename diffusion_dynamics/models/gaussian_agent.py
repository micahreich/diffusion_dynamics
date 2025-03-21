import einops
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


class ConditionalGaussianAgent:
    def __init__(self, model=None):
        self.model = model
    
    def to(self, device) -> "ConditionalGaussianAgent":
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
        self.stats = dataset.stats

        print(f"Testing {self.model.__class__.__name__} forward pass...")
        
        self.model.eval()
        
        with torch.no_grad():
            cond = torch.randn(batch_size, dataset.stats.obs_history_len * dataset.stats.nx)
            out = self.model(cond)
    
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

                for step, batch in enumerate(pbar):
                    # Randomly sample timestep, add noise to batch
                    obs_hist, u_hist = batch
                    obs_hist = obs_hist.to(device)
                    u_hist = u_hist.to(device)
                    
                    out_dist = self.model(obs_hist)
                    out_actions = out_dist.rsample()
                    out_actions = einops.rearrange(out_actions, 'b (h c) -> b h c', c=self.stats.nu)

                    loss = F.mse_loss(out_actions, u_hist)
                    loss.backward()
                    
                    if (step + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=f"{round(loss.item(), 4):.4f}", lr=learning_rate)                
                
                epoch_loss /= len(dataloader)

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
    def load_trained_model(cls, saved_model_path: str) -> "ConditionalGaussianAgent":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.load(saved_model_path, map_location=device)

    def sample(self, cond, device):
        cond = cond.to(device)
        normal_dist = self.model(cond)
        sample = normal_dist.sample()
        out = einops.rearrange(sample, 'b (h c) -> b h c', c=self.stats.nu)
        
        if self.stats.normalized:
            out = TensorDataset1D.unnormalize(out, self.stats.u_min, self.stats.u_max)

        return out