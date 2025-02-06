import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D
import torch
import os
import matplotlib.pyplot as plt
import time
from diffusion_dynamics.models.tests.train_ddpm_1d import ExampleModel

if __name__ == '__main__':
    # Test DDPM training, saving, and loading    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model_params = {
        "save_fpath": "/workspace/diffusion_dynamics/models/tests",
        "save_model_name": "unet1d_test_sinusoidal_step"
    }
    
    # Load the trained model
    start_time = time.perf_counter()
    example_model = ExampleModel().load_trained_model(os.path.join(saved_model_params["save_fpath"], saved_model_params["save_model_name"]))
    print(f"Model loading took {time.perf_counter() - start_time :.3f} s")
    
    # Generate some sequences via inference
    n_samples = 5
    seq_length = 128
        
    example_model.unet.to(device)
    example_model.unet.eval()
    
    example_model.scheduler.set_timesteps(num_inference_steps=50)
    
    # ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon", clip_sample=False, timestep_spacing="trailing")
    # ddim_scheduler.set_timesteps(num_inference_steps=50)

    start_time = time.perf_counter()    
    with torch.no_grad():
        # Start from random Gaussian noise
        sample = torch.randn((n_samples, example_model.n_channels, seq_length), device=device)
        # scheduler.timesteps is an iterable of timesteps in descending order
        for t in example_model.scheduler.timesteps:
        # for t in ddim_scheduler.timesteps:
            sample[:, 0, 0] = -1.0
            sample[:, 1, 0] = 0.0
            
            # For each diffusion step, create a batch of the current timestep
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            # Predict the noise residual
            noise_pred = example_model.unet(sample, t_batch)
            # Compute the previous sample (one denoising step)
            sample = example_model.scheduler.step(noise_pred, t, sample)["prev_sample"]
            # sample = ddim_scheduler.step(noise_pred, t, sample)["prev_sample"]
    
    print(f"Inference took {time.perf_counter() - start_time :.3f} s")
    
    sample = sample.cpu().numpy()
    
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4), sharey=True)

    for i in range(n_samples):
        ax_top = axes[i].inset_axes([0, 0.55, 1, 0.4])  # Top subplot
        ax_bottom = axes[i].inset_axes([0, 0.1, 1, 0.4])  # Bottom subplot
        
        # Plot data
        ax_top.plot(sample[i, 0, :])
        ax_top.set_xticks([])  # Hide x-ticks on top subplot
        
        ax_bottom.plot(sample[i, 1, :])
        
        # Hide outer subplot borders
        axes[i].axis("off")
        axes[i].set_title(f"Sample {i + 1}")

    plt.tight_layout()
    plt.show()