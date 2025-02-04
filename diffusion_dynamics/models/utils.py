import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from diffusion_dynamics.utils import np_sigmoid, np_logit


class NumpyDataset1D(Dataset):
    def __init__(self, np_data=None, normalize_data=True):
        assert np_data is not None, "Data must be provided"
        assert np_data.ndim == 3, "Data must be 3D with shape (n_samples, n_channels, seq_len)"
        
        self.np_data = np_data
        
        self.n_samples, self.n_channels, self.seq_len = np_data.shape
        self.normalize_data = normalize_data
        
        if normalize_data:
            self.np_data, self.data_mean, self.data_std = self.normalize(self.np_data)
        
        self.data = torch.from_numpy(self.np_data).float()
        
    def normalize(self, np_data):
        # Normalize data to have zero mean and unit variance
        mean, std = np.mean(np_data, axis=(0, -1), keepdims=True), np.std(np_data, axis=(0, -1), keepdims=True)
        data = (np_data - mean) / std
        
        # Perform sigmoid normalization to get values in [0, 1]
        return data, mean, std
    
    @staticmethod
    def unnormalize(data, mean, std):
        return data * std + mean
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]