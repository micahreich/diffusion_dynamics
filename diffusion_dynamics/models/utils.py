import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from diffusion_dynamics.utils import np_sigmoid, np_logit
from dataclasses import dataclass, asdict
from typing import Tuple
import pytz
from datetime import datetime
import os
from tqdm import tqdm
from typing import Optional


# @dataclass
# class TensorDataset1DStats:
#     mean: torch.Tensor
#     std: torch.Tensor
#     n_samples: int
#     normalized: bool
#     u_dim: int
#     cond_dim: int

#     @staticmethod
#     def load(fpath) -> "TensorDataset1DStats":
#         stats = torch.load(fpath)
#         return TensorDataset1DStats(**stats)

#     def save(self, fpath):
#         torch.save(asdict(self), fpath)

#     def normalize_data(self, data: torch.Tensor) -> torch.Tensor:
#         stats_device = self.mean.device
#         data_device = data.device
        
#         if stats_device != data_device:
#             self.mean = self.mean.to(data_device)
#             self.std = self.std.to(data_device)
        
#         return (data - self.mean) / self.std

#     def unnormalize_data(self, data: torch.Tensor) -> torch.Tensor:
#         stats_device = self.mean.device
#         data_device = data.device
        
#         if stats_device != data_device:
#             self.mean = self.mean.to(data_device)
#             self.std = self.std.to(data_device)
        
#         if len(data.shape) == 3:
#             return data * self.std + self.mean
#         elif len(data.shape) == 2:
#             return data * self.std.squeeze(0) + self.mean.squeeze(0)
        
#         raise ValueError(f"Data must be 2D or 3D, got {len(data.shape)}D")
    
#     def apply_conditioning(self, noisy_sample, x, noise=None, conditioning_indices=[]) -> Tuple[torch.Tensor, torch.Tensor]:
#         if len(self.conditioning_indices) > 0:
#             conditioning_indices = self.conditioning_indices

#         noisy_sample[:, conditioning_indices, 0] = x[..., conditioning_indices, 0]
        
#         if noise is not None:
#             noise[:, conditioning_indices, 0] = 0.0
#             return noisy_sample, noise
#         else:
#             return noisy_sample

@dataclass
class TensorDataset1DStats:
    nu: int
    nx: int
    u_pred_len: int
    obs_history_len: int
    N_samples: int
    normalized: bool


class TensorDataset1D(Dataset):
    def __init__(self, x_hist: torch.Tensor, u_hist: torch.Tensor,
                 obs_history_len: Optional[int] = None,
                 u_pred_len: Optional[int] = None,
                 normalize=False, verbose=False):
        super().__init__()

        N1, T1, nx = x_hist.shape
        N2, T2, nu = u_hist.shape
        
        assert N1 == N2, "Observation and control histories must have the same number of samples"
        assert T1 == T2 + 1, "Observation and control histories must have the correct time dimensions"
        
        self.x_hist = x_hist
        self.u_hist = u_hist
        
        self.obs_history_len = obs_history_len
        self.u_pred_len = u_pred_len
        self.nx, self.nu = nx, nu
        
        self.normalized = normalize
        self.N = N2
        self.T = T2
        
        # self.T_adjusted = self._calculate_T_adjusted(self.T, self.obs_history_len, self.u_pred_len)
        self.stats = TensorDataset1DStats(nu, nx, u_pred_len, obs_history_len, N2, normalize)
     
    # def _calculate_T_adjusted(self, T: int, obs_hist_len: int, u_pred_len: int) -> int:
    #     n_before = obs_hist_len - 1
    #     n_after = u_pred_len - 1
        
    #     return T - n_before - n_after
    
    def set_obs_history_len(self, obs_history_len: int) -> None:
        self.obs_history_len = obs_history_len
    
    def set_u_pred_len(self, u_pred_len: int) -> None:
        self.u_pred_len = u_pred_len
     
    def __len__(self):
        return self.N * self.T
        
    # def __getitem__(self, idx):
    #     sys_idx = idx // self.T_adjusted
    #     t0_idx = idx % self.T_adjusted
        
    #     x_hist_idx_lo = t0_idx
    #     x_hist_idx_hi = x_hist_idx_lo + self.obs_history_len
        
    #     u_hist_idx_lo = x_hist_idx_hi - 1
    #     u_hist_idx_hi = u_hist_idx_lo + self.u_pred_len
        
    #     # Returns the concatenated observation history ([batch_size, nx * obs_hist_len])
    #     # and future controls ([batch_size, u_pred_len, nu])
        
    #     return self.x_hist[sys_idx, x_hist_idx_lo:x_hist_idx_hi, :].flatten(), \
    #            self.u_hist[sys_idx, u_hist_idx_lo:u_hist_idx_hi, :]

    @staticmethod
    def get_state_window(x, i, obs_history_len):
        indices = torch.arange(i - obs_history_len + 1, i + 1)
        indices = torch.clamp(indices, min=0)
        return x[indices]
    
    @staticmethod
    def get_control_window(u, i, u_pred_len):
        indices = torch.arange(i, i + u_pred_len)
        indices = torch.clamp(indices, max=u.shape[0] - 1)
        return u[indices]
    
    def __getitem__(self, idx):
        assert self.obs_history_len is not None and self.obs_history_len > 0, "obs_history_len must be set and > 0"
        assert self.u_pred_len is not None and self.u_pred_len > 0, "u_pred_len must be set and > 0"
        
        sys_idx = idx // self.T
        t0_idx = idx % self.T
        
        # [X, X, X, X, X, X]
        # [U, U, U, U, U]
        
        # x_hist_idx_lo = t0_idx
        # x_hist_idx_hi = x_hist_idx_lo + self.obs_history_len
        
        # u_hist_idx_lo = x_hist_idx_hi - 1
        # u_hist_idx_hi = u_hist_idx_lo + self.u_pred_len
        
        # Returns the concatenated observation history ([batch_size, nx * obs_hist_len])
        # and future controls ([batch_size, u_pred_len, nu])
        
        x_hist = self.get_state_window(self.x_hist[sys_idx], t0_idx, self.obs_history_len)
        u_hist = self.get_control_window(self.u_hist[sys_idx], t0_idx, self.u_pred_len)
        
        return x_hist.flatten(), u_hist


@dataclass
class SaveModelParams:
    save_full_fpath: str
    save_model_name: Optional[str]

class ConditionalDiffusionModel:
    def __init__(self, model=None, scheduler=None):
        self.model = model
        self.scheduler = scheduler
        
    def to(self, device):
        if self.model is not None:
            self.model = self.model.to(device)

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
        
        print(f"Testing {self.model.__class__.__name__} forward pass...")
        
        self.model.eval()
        with torch.no_grad():
            u_noisy = torch.randn(batch_size, dataset.u_pred_len, dataset.nu)
            
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,)).long()
            cond = torch.randn(batch_size, dataset.obs_history_len * dataset.nx)

            out = self.model(u_noisy, t, cond)
    
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Training {self.model.__class__.__name__} with {trainable_params} trainable parameters")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        _, _u = dataset[0]
        u_pred_len, nu = _u.shape
        
        assert u_pred_len == dataset.u_pred_len, "dataset u_pred_len must match the model's expected u_pred_len"
        
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
                    pbar.set_postfix(loss=loss.item(), lr=learning_rate)                
                
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
        print(f"Saving model to {save_fpath_full}...")
        
        os.makedirs(save_fpath_full, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_fpath_full, "model.pt"))

    @classmethod
    def load_trained_model(cls, saved_model_dir_fpath: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(os.path.join(saved_model_dir_fpath, "model.pt"), map_location=device)

        m = cls()
        assert m.model is not None, "model must be instantiated before loading a trained model"

        m.model.load_state_dict(state_dict)

        return m
    
    def sample(self, n_samples, cond, data_stats: TensorDataset1DStats, device, num_inference_steps=None):
        sample = torch.randn((n_samples, data_stats.u_pred_len, data_stats.nu), device=device)
        cond = cond.to(device)
        
        assert cond.shape == (n_samples, data_stats.nx * data_stats.obs_history_len)
        
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
        
        return sample


if __name__ == "__main__":
    xhist = torch.tile(torch.arange(0, 10).unsqueeze(1), (1, 7)).unsqueeze(0)
    uhist = torch.tile(torch.arange(0, 9).unsqueeze(1), (1, 5)).unsqueeze(0)
    
    print(xhist.shape)
    print(uhist.shape)
    
    dataset = TensorDataset1D(xhist, uhist, obs_history_len=3, u_pred_len=2, normalize=False)
        
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for batch in dataloader:
        print(batch)
        print("0:", batch[0].shape)  # Each 'batch' is a tensor containing batch_size items
        print("1:", batch[1].shape)
