import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D
import torch
import os
import matplotlib.pyplot as plt

from diffusion_dynamics.models.tests.train_ddpm_1d import ExampleModel

if __name__ == '__main__':
    # Test DDPM training, saving, and loading    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model_params = {
        "save_fpath": "/workspace/diffusion_dynamics/models/tests",
        "save_model_name": "unet1d_test_sinusoidal_step"
    }
    
    # Load the trained model
    trained_dataset_params = ExampleModel.load_trained_model(os.path.join(saved_model_params["save_fpath"], saved_model_params["save_model_name"]))
    
    mu_np = trained_dataset_params["data_mean"]
    sigma_np = trained_dataset_params["data_std"]
    
    if mu_np is not None and sigma_np is not None:
        mu_pt = torch.squeeze(torch.tensor(mu_np, device=device))
        sigma_pt = torch.squeeze(torch.tensor(sigma_np, device=device))
    
    # Generate some sequences via inference
    n_samples = 5
    seq_length = 128
        
    ExampleModel.model.to(device)
    ExampleModel.model.eval()

    with torch.no_grad():
        # Start from random Gaussian noise
        sample = torch.randn((n_samples, ExampleModel.n_channels, seq_length), device=device)
        # scheduler.timesteps is an iterable of timesteps in descending order
        for t in ExampleModel.scheduler.timesteps:
            # For each diffusion step, create a batch of the current timestep
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            # Predict the noise residual
            noise_pred = ExampleModel.model(sample, t_batch)
            # Compute the previous sample (one denoising step)
            sample = ExampleModel.scheduler.step(noise_pred, t, sample)["prev_sample"]
            
            # sample[:, 0, -1] = 0.0
            # sample[:, 0, 0] = 0.0
    
    sample = sample.cpu().numpy()
    # sample = NumpyDataset1D.unnormalize(sample, mean=mu_np, std=sigma_np)
    
    # generated = ExampleModel.sample(n_inference_samples, 128)
    # generated = generated.cpu().numpy()
    # generated = NumpyDataset1D.unnormalize(generated, mean=trained_dataset_params["data_mean"], std=trained_dataset_params["data_std"])
        
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4), sharey=True)

    for i in range(n_samples):
        ax_top = axes[i].inset_axes([0, 0.55, 1, 0.4])  # Top subplot
        ax_bottom = axes[i].inset_axes([0, 0.1, 1, 0.4])  # Bottom subplot
        
        # Plot data
        ax_top.plot(sample[i, 0, :])

        # ax_top.set_yticks()
        ax_top.set_xticks([])  # Hide x-ticks on top subplot
        
        ax_bottom.plot(sample[i, 1, :])
        
        # Hide outer subplot borders
        axes[i].axis("off")
        axes[i].set_title(f"Sample {i + 1}")

    plt.tight_layout()
    plt.show()