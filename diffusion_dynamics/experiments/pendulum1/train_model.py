import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D
import torch


if __name__ == "__main__":
    # Test DDPM training, saving, and loading
    
    class PendulumModel(UNet1DModel):
        n_channels = 3
        model = UNet1D(
            in_channels=n_channels,
            out_channels=n_channels,
            base_channels=64,
            dim_mults=[1, 2, 4],
        )
        scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    saved_model_params = {
        "save_fpath": "/workspace/diffusion_dynamics/experiments/pendulum1/saved_models",
        "save_model_name": "pendulum_model1",
    }
    
    # Train the model
    data = np.load("/workspace/diffusion_dynamics/experiments/pendulum1/data/pendulum1_N=10000_2025-02-03__03-23-14.npy")

    dataset = NumpyDataset1D(np_data=data, normalize_data=True)
    PendulumModel.train(dataset,
                       n_epochs=200,
                       batch_size=128,
                       learning_rate=1e-4,
                       save_model_params=saved_model_params)