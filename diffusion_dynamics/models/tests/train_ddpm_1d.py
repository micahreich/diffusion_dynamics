import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import TensorDataset1D
import torch
from torch.distributions import Normal, Uniform

class ExampleModel(UNet1DModel):
    n_channels = 2
    model = UNet1D(
        in_channels=n_channels,
        out_channels=n_channels,
        base_channels=32,
        dim_mults=[1, 2, 4],
    )
    scheduler = DDPMScheduler(num_train_timesteps=1000,
                              clip_sample=False,
                              variance_type="fixed_small_log")
    
if __name__ == '__main__':
    # Test DDPM training, saving, and loading    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    saved_model_params = {
        "save_fpath": "/workspace/diffusion_dynamics/models/tests",
        "save_model_name": "unet1d_test_sinusoidal_step"
    }
   
    # Generate some sinusoidal/step data
    n_samples = 5_000
    seq_length = 128
    n_channels = 2
    
    x = torch.linspace(0, 2 * np.pi, seq_length)
    phase = Uniform(0, 2*torch.pi)
    freq = Uniform(0.5, 5.0)
    
    sin_data = 7.0 * torch.sin(
        x.unsqueeze(0) * freq.sample((n_samples, 1)) + freq.sample((n_samples, 1))
    )
    
    step_data = (torch.arange(seq_length).unsqueeze(0).expand(n_samples, seq_length) >= \
                torch.randint(0, seq_length, (n_samples,)).unsqueeze(1)).float()
    
    data = torch.stack([sin_data, step_data], dim=1)

    # Train the model
    dataset = TensorDataset1D(data=data)
    ExampleModel.train(dataset,
                       n_epochs=250,
                       batch_size=128,
                       learning_rate=2e-4,
                       save_model_params=saved_model_params,
                       initial_conditioning_channel_idx=[0, 1])
