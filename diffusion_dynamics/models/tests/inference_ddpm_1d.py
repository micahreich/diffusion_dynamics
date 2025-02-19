import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D
import torch
import os
import matplotlib.pyplot as plt
import time
from diffusion_dynamics.models.tests.train_ddpm_1d import ExampleModel

def dJ_dx(x):
    _, _, seq_len = x.shape
    
    switch_idx = seq_len // 2
    
    # Encourage low values in the first half of the sequence and high values in the second half
    mid_switch = torch.norm(x[:, 1, :switch_idx]) + torch.norm(torch.exp(-x[:, 1, switch_idx:]))
    mid_switch.backward()
    
    return x.grad

if __name__ == '__main__':
    # Test DDPM training, saving, and loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model_params = {
        "save_fpath": "/workspace/diffusion_dynamics/models/tests",
        "save_model_name": "unet1d_test_sinusoidal_step",
    }

    # Load the trained model
    start_time = time.perf_counter()
    example_model = ExampleModel().load_trained_model(
        os.path.join(saved_model_params["save_fpath"], saved_model_params["save_model_name"])
    )
    print(f"Model loading took {time.perf_counter() - start_time :.3f} s")

    # Generate some sequences via inference
    n_samples = 5
    seq_length = 128

    example_model.to(device)
    example_model.unet.eval()

    example_model.scheduler.set_timesteps(num_inference_steps=50)

    # ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type="epsilon", clip_sample=False, timestep_spacing="trailing")
    # ddim_scheduler.set_timesteps(num_inference_steps=50)

    alpha = 2.0
    
    start_time = time.perf_counter()
    
    # Start from random Gaussian noise
    sample = torch.randn((n_samples, example_model.n_channels, seq_length), device=device, requires_grad=True)
    
    # scheduler.timesteps is an iterable of timesteps in descending order
    for t in example_model.scheduler.timesteps:
        # Compute gradient of cost w.r.t sample and add to sample to encourage low cost
        grad_J = dJ_dx(sample.detach().clone().requires_grad_(True))
        sample = sample - alpha * grad_J
    
        with torch.no_grad():
            conditioning_normalized = example_model.train_data_stats.normalize_data(
                torch.tensor([2.0, 0.0], device=device).view(1, example_model.n_channels, 1)
            )
            
            sample = example_model.train_data_stats.apply_conditioning(
                sample,
                conditioning_normalized
            )
                    
            # For each diffusion step, create a batch of the current timestep
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # Predict the noise residual
            model_out = example_model.unet(sample, t_batch)
            
            # Compute the previous sample (one denoising step)
            sample = example_model.scheduler.step(model_out, t, sample)["prev_sample"]

    print(f"Inference took {time.perf_counter() - start_time :.3f} s")

    sample = example_model.train_data_stats.unnormalize_data(sample).cpu().numpy()
    # sample = sample.cpu().numpy()

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
