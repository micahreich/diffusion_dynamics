import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D, TensorDataset1D
import torch
import os


class PendulumModel(UNet1DModel):
    n_channels = 3
    model = UNet1D(
        in_channels=n_channels,
        out_channels=n_channels,
        base_channels=32,
        dim_mults=[1, 2, 4, 8],
    )
    scheduler = DDPMScheduler(num_train_timesteps=1000,
                              clip_sample=False,
                              variance_type="fixed_small_log")

if __name__ == "__main__":
    # Test DDPM training, saving, and loading
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    saved_model_params = {
        "save_fpath": "/workspace/diffusion_dynamics/experiments/pendulum1/saved_models",
        "save_model_name": "pendulum_model1_nocontrols3",
    }
    
    # Train the model
    data_fpath = "/workspace/diffusion_dynamics/experiments/pendulum1/data/pendulum1_nocontrols3_N=10000_2025-02-05__02-22-34.pt"
    data = torch.load(data_fpath)
    dataset = TensorDataset1D(data=data)
    
    print(f"Training model (dataset: {os.path.basename(data_fpath)})...")
    
    PendulumModel.train(dataset,
                       n_epochs=500,
                       batch_size=128,
                       learning_rate=1e-4,
                       save_model_params=saved_model_params)