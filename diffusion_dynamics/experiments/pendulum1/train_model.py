import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D
import torch


class PendulumModel(UNet1DModel):
    n_channels = 3
    model = UNet1D(
        in_channels=n_channels,
        out_channels=n_channels,
        base_channels=32,
        dim_mults=[1, 2, 4, 8],
    )
    scheduler = DDPMScheduler(num_train_timesteps=1000)

if __name__ == "__main__":
    # Test DDPM training, saving, and loading
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    saved_model_params = {
        "save_fpath": "/workspace/diffusion_dynamics/experiments/pendulum1/saved_models",
        "save_model_name": "pendulum_model1_nocontrols2",
    }
    
    # Train the model
    data = np.load("/workspace/diffusion_dynamics/experiments/pendulum1/data/pendulum1_nocontrols2_N=12000_2025-02-04__03-34-09.npy")

    dataset = NumpyDataset1D(np_data=data, normalize_data=False)
    PendulumModel.train(dataset,
                       n_epochs=500,
                       batch_size=128,
                       learning_rate=1e-4,
                       save_model_params=saved_model_params)