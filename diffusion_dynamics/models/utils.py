import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from diffusion_dynamics.utils import np_sigmoid, np_logit
from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass
class TensorDataset1DStats:
    mean: torch.Tensor
    std: torch.Tensor
    n_samples: int
    n_channels: int
    seq_len: int
    normalized: bool

    @staticmethod
    def load(fpath) -> "TensorDataset1DStats":
        stats = torch.load(fpath)
        return TensorDataset1DStats(**stats)

    def save(self, fpath):
        torch.save(asdict(self), fpath)

    def normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / self.std

    def unnormalize_data(self, data: torch.Tensor) -> torch.Tensor:
        if len(data.shape) == 3:
            return data * self.std + self.mean
        elif len(data.shape) == 2:
            return data * self.std.squeeze(0) + self.mean.squeeze(0)
        
        raise ValueError(f"Data must be 2D or 3D, got {len(data.shape)}D")


class TensorDataset1D(Dataset):
    def __init__(self, data, normalize, verbose=False, conditioning_indices=[]):
        super().__init__()

        assert len(data.shape) == 3, "Data must have shape (n_samples, n_channels, seq_len)"

        self.data = data
        self.normalized = normalize
        self.stats = self.get_data_stats(self.data, verbose)

        if normalize:
            self.data = (self.data - self.stats.mean) / self.stats.std
        
        self.conditioning_indices = conditioning_indices

    def get_data_stats(self, data, verbose=False):
        assert len(data.shape) == 3, "Data must have shape (n_samples, n_channels, seq_len)"
        
        n_samples, n_channels, seq_len = data.shape

        mu = torch.mean(data, dim=(0, -1), keepdim=True)
        std = torch.maximum(torch.tensor(1e-8), torch.std(data, dim=(0, -1), keepdim=True))
        
        if verbose:
            print(f"Dataset mean: {mu}, Data std: {std}")

        return TensorDataset1DStats(mu, std, n_samples, n_channels, seq_len, self.normalized)

    def apply_conditioning(self, noise, noisy_sample, x, conditioning_indices=[]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.conditioning_indices:
            conditioning_indices = self.conditioning_indices
            
        noisy_sample[..., conditioning_indices, 0] = x[..., conditioning_indices, 0]
        noise[..., conditioning_indices, 0] = 0.0
        
        return noisy_sample, noise

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class NumpyDataset1D(Dataset):
    def __init__(self, np_data=None, normalize_data=True):
        assert np_data is not None, "Data must be provided"
        assert np_data.ndim == 3, "Data must be 3D with shape (n_samples, n_channels, seq_len)"

        self.np_data = np_data
        self.data_mean, self.data_std = None, None

        self.n_samples, self.n_channels, self.seq_len = np_data.shape
        self.normalize_data = normalize_data

        if normalize_data:
            self.np_data, self.data_mean, self.data_std = self.normalize(self.np_data)

        self.data = torch.from_numpy(self.np_data).float()

    def normalize(self, np_data):
        # Normalize data to have zero mean and unit variance
        mean, std = np.mean(np_data, axis=(0, -1), keepdims=True), np.std(np_data, axis=(0, -1), keepdims=True)
        data = (np_data - mean) / std
        data = np.tanh(data)

        # Perform sigmoid normalization to get values in [0, 1]
        return data, mean, std

    @staticmethod
    def unnormalize(data, mean, std):
        data = np.arctanh(data)
        return data * std + mean

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
