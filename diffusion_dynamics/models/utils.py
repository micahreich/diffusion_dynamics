import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pytz
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_dynamics.utils import np_logit, np_sigmoid

activation_dict = nn.ModuleDict({
    "ReLU": nn.ReLU(),
    "ELU": nn.ELU(),
    "GELU": nn.GELU(),
    "Tanh": nn.Tanh(),
    "Mish": nn.Mish(),
    "Identity": nn.Identity(),
    "Softplus": nn.Softplus(),
})


@dataclass
class SaveModelParams:
    save_full_fpath: str
    save_model_name: Optional[str]


@dataclass
class TensorDataset1DStats:
    nu: int
    nx: int
    u_pred_len: int
    obs_history_len: int
    N_samples: int
    normalized: bool
    u_mean: Optional[torch.Tensor] = None
    u_std: Optional[torch.Tensor] = None
    u_min: Optional[torch.Tensor] = None
    u_max: Optional[torch.Tensor] = None


class TensorDataset1D(Dataset):
    def __init__(self,
                 x_hist: torch.Tensor,
                 u_hist: torch.Tensor,
                 obs_history_len: Optional[int] = None,
                 u_pred_len: Optional[int] = None,
                 normalize=False):
        super().__init__()

        N1, T1, nx = x_hist.shape
        N2, T2, nu = u_hist.shape

        assert N1 == N2, "Observation and control histories must have the same number of samples"
        assert T1 == T2 + 1, "Observation and control histories must have the correct time dimensions"

        # Check for any NaNs in the data
        assert not torch.isnan(x_hist).any(), "x_hist contains NaNs"
        assert not torch.isnan(u_hist).any(), "u_hist contains NaNs"

        self.x_hist = x_hist
        self.u_hist = u_hist

        self.normalized = normalize
        self.N = N2
        self.T = T2

        self.stats = TensorDataset1DStats(nu, nx, u_pred_len, obs_history_len, N2, normalize)

        if normalize:
            self.stats.u_max = u_hist.amax(dim=(0, 1))
            self.stats.u_min = u_hist.amin(dim=(0, 1))

            print(f"max_u: {self.stats.u_max}")
            print(f"min_u: {self.stats.u_min}")

            self.stats.u_mean = u_hist.mean(dim=(0, 1))
            self.stats.u_std = u_hist.std(dim=(0, 1))

            print(f"mean_u: {self.stats.u_mean}")
            print(f"std_u: {self.stats.u_std}")

    def set_obs_history_len(self, obs_history_len: int) -> None:
        self.stats.obs_history_len = obs_history_len

    def set_u_pred_len(self, u_pred_len: int) -> None:
        self.stats.u_pred_len = u_pred_len

    def __len__(self):
        return self.N * self.T

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

    @staticmethod
    def normalize(x, xmin, xmax):
        x = (x - xmin) / (xmax - xmin)
        return x * 2. - 1.

    @staticmethod
    def unnormalize(x, xmin, xmax):
        x = (x + 1.) / 2.
        return x * (xmax - xmin) + xmin

    def __getitem__(self, idx):
        assert self.stats.obs_history_len is not None and self.stats.obs_history_len > 0, "obs_history_len must be set and > 0"
        assert self.stats.u_pred_len is not None and self.stats.u_pred_len > 0, "u_pred_len must be set and > 0"

        sys_idx = idx // self.T
        t0_idx = idx % self.T

        x_hist = self.get_state_window(self.x_hist[sys_idx], t0_idx, self.stats.obs_history_len)
        u_hist = self.get_control_window(self.u_hist[sys_idx], t0_idx, self.stats.u_pred_len)

        if self.normalized:
            u_hist = self.normalize(u_hist, self.stats.u_min, self.stats.u_max)

        return x_hist.flatten(), u_hist
        # return torch.randn(self.stats.nx * self.stats.obs_history_len), torch.randn(self.stats.u_pred_len, self.stats.nu)


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
