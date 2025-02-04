import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D
import torch

class ExampleModel(UNet1DModel):
    n_channels = 2
    model = UNet1D(
        in_channels=n_channels,
        out_channels=n_channels,
        base_channels=32,
        dim_mults=[1, 2, 4],
    )
    scheduler = DDPMScheduler(num_train_timesteps=100,
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
    
    x = np.linspace(0, 2 * np.pi, seq_length)
    data = np.empty(shape=(n_samples, n_channels, seq_length))
    
    for i in range(n_samples):
        freq = np.random.uniform(0.5, 5.0)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = 7.0
        
        sinusoidal_component = amplitude * np.sin(freq * x + phase)
        
        step_idx = np.random.randint(0, seq_length)
        step_component = np.where(np.arange(seq_length) < step_idx, 0.0, 1.0)
        
        data[i, 0, :] = sinusoidal_component
        data[i, 1, :] = step_component

    # Train the model
    dataset = NumpyDataset1D(np_data=data, normalize_data=False)
    ExampleModel.train(dataset,
                       n_epochs=100,
                       batch_size=32,
                       learning_rate=4e-5,
                       save_model_params=saved_model_params)
