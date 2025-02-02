import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler  # use the DDPM noise scheduler from diffusers
from tqdm import tqdm
from diffusers.models.resnet import ResidualTemporalBlock1D
from diffusers.models.unets.unet_1d_blocks import DownResnetBlock1D, UpResnetBlock1D, DownBlock1D, UpBlock1D
from diffusers.models.unets.unet_1d import UNet1DModel
from datetime import datetime
import pytz
import os
import pathlib
import pickle

# ---------------------------
# Helper: Sinusoidal timestep embedding
# ---------------------------
def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def np_logit(x):
    return np.log(x / (1 - x))

def get_timestep_embedding(timesteps, embedding_dim):
    """
    From https://github.com/openai/improved-diffusion.
    Create sinusoidal embeddings.
    
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
    
    Returns:
        A tensor of shape (N, embedding_dim) with the timestep embeddings.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# # ---------------------------
# # Model Components: A simple 1D residual block and UNet
# # ---------------------------
# class ResidualBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, time_embed_dim):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.time_mlp = nn.Linear(time_embed_dim, out_channels)
#         # Use a 1x1 convolution if the number of channels changes
#         self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
#         self.activation = nn.ReLU()

#     def forward(self, x, t_emb):
#         # x: (B, C, L)
#         # t_emb: (B, time_embed_dim, 1)
#         h = self.activation(self.conv1(x))
#         # Process time embedding and add it (broadcast along the sequence length)
#         t_emb_proj = self.time_mlp(t_emb.squeeze(-1)).unsqueeze(-1)  # (B, out_channels, 1)
#         h = h + t_emb_proj
#         h = self.activation(self.conv2(h))
#         return h + self.res_conv(x)


class UNet1D(nn.Module):
    """
    A flexible 1D UNet for diffusion over sequences.
    This version downsamples twice and upsamples twice, with skip connections at each scale.
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 base_channels=64,
                 block_out_num_channels_multipliers=[1, 2, 4],
                 time_embed_dim=128):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        
        assert len(block_out_num_channels_multipliers) > 0

        # --- Encoder ---
        # Initial convolution and residual block (no downsampling yet)
        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        # self.resblock1 = ResidualBlock1D(base_channels, base_channels, time_embed_dim)
        self.init_resblock = ResidualTemporalBlock1D(base_channels, base_channels, embed_dim=time_embed_dim, kernel_size=3)
        
        self.encoder_convs = nn.ModuleList()
        self.encoder_resblocks = nn.ModuleList()
        
        for i in range(1, len(block_out_num_channels_multipliers)):
            layer_in_channels = base_channels * block_out_num_channels_multipliers[i-1]
            layer_out_channels = base_channels * block_out_num_channels_multipliers[i]
                        
            self.encoder_convs.append(
                nn.Conv1d(layer_in_channels, layer_out_channels, kernel_size=4, stride=2, padding=1)
            )
            
            self.encoder_resblocks.append(
                ResidualTemporalBlock1D(layer_out_channels, layer_out_channels, embed_dim=time_embed_dim, kernel_size=3)
            )
        
        bottleneck_channels = base_channels * block_out_num_channels_multipliers[-1]
        self.bottleneck = ResidualTemporalBlock1D(bottleneck_channels, bottleneck_channels, embed_dim=time_embed_dim, kernel_size=3)
        
        self.decoder_ups = nn.ModuleList()
        self.decoder_resblocks = nn.ModuleList()
        
        for i in range(len(block_out_num_channels_multipliers) - 1, 0, -1):
            layer_in_channels = base_channels * block_out_num_channels_multipliers[i]
            layer_out_channels = base_channels * block_out_num_channels_multipliers[i-1]
                        
            self.decoder_ups.append(
                nn.ConvTranspose1d(layer_in_channels, layer_out_channels, kernel_size=4, stride=2, padding=1)
            )
            
            self.decoder_resblocks.append(
                ResidualTemporalBlock1D(layer_out_channels, layer_out_channels, embed_dim=time_embed_dim, kernel_size=3)
            )
        
        # Final convolution to bring features back to in_channels
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x, t):
        """
        x: Tensor of shape (B, 1, L) — our 1D sequence.
        t: Tensor of shape (B,) — diffusion timesteps.
        """
        # Create timestep embeddings (B, time_embed_dim, 1)
        t_emb = get_timestep_embedding(t, self.time_embed_dim)
        
        h = self.init_conv(x)
        h = self.init_resblock(h, t_emb)
        
        # down
        skip_connections = []
        for conv, resblock in zip(self.encoder_convs, self.encoder_resblocks):
            skip_connections.append(h)
            
            h = conv(h)
            h = resblock(h, t_emb)
                    
        # bottleneck
        h = self.bottleneck(h, t_emb)
                
        # up
        for up, resblock in zip(self.decoder_ups, self.decoder_resblocks):
            h = up(h)
            skip = skip_connections.pop()
                        
            h = h + skip
            h = resblock(h, t_emb)
        
        h = self.activation(h)
        out = self.final_conv(h)
        
        return out

# ---------------------------
# Dataset: Create sine-wave sequences
# ---------------------------
class SinusoidDataset(Dataset):
    def __init__(self, n_samples, seq_length, normalize=True):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_channels = 2
        self.normalize = normalize
        
        self.np_data, self.data_mean, self.data_std = self.generate_data(self.n_samples, self.seq_length, normalize=normalize)
        self.data = torch.from_numpy(self.np_data).float()
        
    def generate_data(self, n_samples, seq_length, normalize=True):
        assert self.n_channels == 2
        
        x = np.linspace(0, 2 * math.pi, seq_length)
        data = np.empty(shape=(n_samples, 2, seq_length))
        
        for i in range(n_samples):
            freq = np.random.uniform(0.5, 3.5)
            phase = np.random.uniform(0, 2 * math.pi)
            # amplitude = np.random.uniform(0.5, 10)
            amplitude = 1.0
            
            sinusoidal_component = amplitude * np.sin(freq * x + phase)
            
            step_idx = np.random.randint(0, seq_length)
            step_component = np.where(np.arange(seq_length) < step_idx, 0.0, 1.0)
            
            data[i, 0, :] = sinusoidal_component
            data[i, 1, :] = step_component

        if normalize:
            # first z-score normalize all sequences
            mean, std = np.mean(data, axis=(0, -1), keepdims=True), np.std(data, axis=(0, -1), keepdims=True)
            data = (data - mean) / std
            
            # then perform sigmoid normalization to get values in [0, 1]
            return np_sigmoid(data), mean, std
        
        return data, None, None
    
    @staticmethod
    def denormalize(data, mean, std):
        assert len(data.shape) == len(mean.shape) == len(std.shape)
        assert data.shape[1] == mean.shape[1] == std.shape[1]
        
        logits = np_logit(data)
        
        return logits * std + mean
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------
# Training function
# ---------------------------
def train_model(dataset: Dataset,
                n_epochs=100,
                batch_size=64,
                learning_rate=1e-4,
                save=True,
                save_fpath="/workspace/saved_models/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_samples, n_channels, seq_length = dataset.np_data.shape
    ddpm_scheduler_params = {
        "num_train_timesteps": 1000
    }
    dataset_params = {
        "n_samples": n_samples,
        "seq_length": seq_length,
        "n_channels": n_channels,
        "normalize": True,
        "data_mean": dataset.data_mean,
        "data_std": dataset.data_std
    }
 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate our 1D UNet diffusion model
    model = UNet1D(
        in_channels=n_channels,
        out_channels=n_channels,
        base_channels=64,
        block_out_num_channels_multipliers=[1, 2, 4],
        time_embed_dim=128
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create a DDPM scheduler with (for example) 1000 diffusion steps.
    scheduler = DDPMScheduler(num_train_timesteps=ddpm_scheduler_params["num_train_timesteps"])

    model.train()
    for epoch in range(n_epochs):
        # Wrap the dataloader with tqdm for a progress bar.
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
        for step, batch in enumerate(pbar):
            batch = batch.to(device)  # shape (B, 1, seq_length)
            # Sample a random timestep for each example in the batch.
            t = torch.randint(0, scheduler.config.num_train_timesteps, (batch.shape[0],), device=device).long()
            noise = torch.randn_like(batch)
            # Add noise to the clean batch at the given timesteps.
            noisy_batch = scheduler.add_noise(batch, noise, t)
            # Predict the noise that was added.
            noise_pred = model(noisy_batch, t)
                        
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the tqdm progress bar with the current loss.
            pbar.set_postfix(loss=loss.item())

    # Save the trained model weights
    if save:
        nyc_tz = pytz.timezone('America/New_York')
        time_str = datetime.now(nyc_tz).strftime("%Y-%m-%d__%H-%M-%S")
        save_fpath_full = os.path.join(save_fpath, f"unet1d_{time_str}")
        os.makedirs(save_fpath_full, exist_ok=True)
        
        torch.save(model.state_dict(), os.path.join(save_fpath_full, "model.pt"))
        
        with open(os.path.join(save_fpath_full, "params.pkl"), "wb") as f:
            combined_dict = {
                "ddpm_scheduler_params": ddpm_scheduler_params,
                "dataset_params": dataset_params
            }
            pickle.dump(combined_dict, f)
        
        return model, scheduler


def load_model(device, model_prototype, saved_model_fpath):
    state_dict = torch.load(os.path.join(saved_model_fpath, "model.pt"), map_location=device)
    model_prototype.load_state_dict(state_dict)
    
    with open(os.path.join(saved_model_fpath, "params.pkl"), "rb") as f:
        combined_dict = pickle.load(f)
        
        ddpm_scheduler_params = combined_dict["ddpm_scheduler_params"]
        dataset_params = combined_dict["dataset_params"]
        
    scheduler = DDPMScheduler(num_train_timesteps=ddpm_scheduler_params["num_train_timesteps"])
    
    return model_prototype.to(device), scheduler, dataset_params

# ---------------------------
# Inference / Sampling function
# ---------------------------
def sample_sequences(model, scheduler, n_samples, n_channels, seq_length, device, y0=None):
    """
    Generates new sequences by running the reverse diffusion process.
    """
    model.eval()
    with torch.no_grad():
        # Start from random Gaussian noise
        sample = torch.randn((n_samples, n_channels, seq_length), device=device)
        # scheduler.timesteps is an iterable of timesteps in descending order
        for t in scheduler.timesteps:
            # For each diffusion step, create a batch of the current timestep
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            # Predict the noise residual
            noise_pred = model(sample, t_batch)
            # Compute the previous sample (one denoising step)
            sample = scheduler.step(noise_pred, t, sample)["prev_sample"]
            
            if y0 is not None:
                sample[:, :, 0] = y0
                
        return sample


# # ---------------------------
# # Main script: Train, then sample and plot generated sequences.
# # ---------------------------
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dataset = SinusoidDataset(n_samples=1000, seq_length=128, normalize=True)
    
#     # Train the model (or load a pre-trained model if available)
#     model, scheduler = train_model(dataset, n_epochs=100, batch_size=64, learning_rate=1e-4)

#     # Generate some sequences via inference
#     num_gen = 6
    
#     generated = sample_sequences(model, scheduler, num_gen, dataset.seq_length, device)
#     generated = generated.cpu().numpy()
#     generated = dataset.denormalize(generated)
    

#     # Plot the generated sequences
#     x_axis = np.linspace(0, 2 * math.pi, dataset.seq_length)
#     fig, ax = plt.subplots(nrows=)
    
#     for i in range(num_gen):
#         ax.plot(x_axis, generated[i, 0], label=f"Seq {i}")
#         ax.set_xlabel("x")
#         ax.set_ylabel("Amplitude")
    
#     ax.legend()
#     plt.savefig(f"generated_chatgpt.png")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    save_fpath = "/workspace/saved_models"
    saved_model_name = "unet1d_2025-02-02__01-52-58"

    # # Train the model (or load a pre-trained model if available)
    # dataset = SinusoidDataset(n_samples=10_000, seq_length=128, normalize=False)

    # model, scheduler = train_model(dataset, n_epochs=50, batch_size=64, learning_rate=1e-4,
    #                                save=True,
    #                                save_fpath=save_fpath)
    
    model_prototype = UNet1D(
        in_channels=2,
        out_channels=2,
        base_channels=64,
        block_out_num_channels_multipliers=[1, 2, 4],
        time_embed_dim=128
    )
    
    model, scheduler, dataset_params = load_model(device,
                                                  model_prototype,
                                                  os.path.join(save_fpath, saved_model_name))
    
    # Generate some sequences via inference
    num_gen = 6
    
    generated = sample_sequences(model, scheduler, num_gen,
                                 dataset_params["n_channels"],
                                 dataset_params["seq_length"],
                                 device)
    generated = generated.cpu().numpy()
    # generated = SinusoidDataset.denormalize(generated, dataset_params["data_mean"], dataset_params["data_std"])
    
    fig, axes = plt.subplots(1, num_gen, figsize=(4 * num_gen, 4), sharey=True)

    for i in range(num_gen):
        ax_top = axes[i].inset_axes([0, 0.55, 1, 0.4])  # Top subplot
        ax_bottom = axes[i].inset_axes([0, 0.1, 1, 0.4])  # Bottom subplot
        
        # Plot data
        ax_top.plot(generated[i, 0, :])
        ax_top.set_xticks([])  # Hide x-ticks on top subplot
        
        ax_bottom.plot(generated[i, 1, :])
        
        # Hide outer subplot borders
        axes[i].axis("off")
        axes[i].set_title(f"Sample {i + 1}")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{saved_model_name}-eval.png")

