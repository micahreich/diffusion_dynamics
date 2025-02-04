import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D
import torch
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Test DDPM training, saving, and loading    
    class ExampleModel(UNet1DModel):
        model = UNet1D(
            in_channels=2,
            out_channels=2,
            base_channels=64,
            dim_mults=[1, 2, 4],
        )
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        n_channels = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    saved_model_params = {
        "save_fpath": "/workspace/diffusion_dynamics/models/tests",
        "save_model_name": "unet1d_test_sinusoidal_step"
    }
   
    # Load the trained model
    trained_dataset_params = ExampleModel.load_trained_model(os.path.join(saved_model_params["save_fpath"], saved_model_params["save_model_name"]))
    
    # Generate some sequences via inference
    n_inference_samples = 6
    
    generated = ExampleModel.sample(n_inference_samples, 128)
    generated = generated.cpu().numpy()
    generated = NumpyDataset1D.unnormalize(generated, mean=trained_dataset_params["data_mean"], std=trained_dataset_params["data_std"])
        
    fig, axes = plt.subplots(1, n_inference_samples, figsize=(4 * n_inference_samples, 4), sharey=True)

    for i in range(n_inference_samples):
        ax_top = axes[i].inset_axes([0, 0.55, 1, 0.4])  # Top subplot
        ax_bottom = axes[i].inset_axes([0, 0.1, 1, 0.4])  # Bottom subplot
        
        # Plot data
        ax_top.plot(generated[i, 0, :])

        # ax_top.set_yticks()
        ax_top.set_xticks([])  # Hide x-ticks on top subplot
        
        ax_bottom.plot(generated[i, 1, :])
        
        # Hide outer subplot borders
        axes[i].axis("off")
        axes[i].set_title(f"Sample {i + 1}")

    plt.tight_layout()
    plt.show()