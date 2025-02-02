import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler  # use the DDPM noise scheduler from diffusers
from tqdm import tqdm

# ---------------------------
# Helper: Sinusoidal timestep embedding
# ---------------------------
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


# ---------------------------
# Model Components: A simple 1D residual block and UNet
# ---------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        # Use a 1x1 convolution if the number of channels changes
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x, t_emb):
        # x: (B, C, L)
        # t_emb: (B, time_embed_dim, 1)
        h = self.activation(self.conv1(x))
        # Process time embedding and add it (broadcast along the sequence length)
        t_emb_proj = self.time_mlp(t_emb.squeeze(-1)).unsqueeze(-1)  # (B, out_channels, 1)
        h = h + t_emb_proj
        h = self.activation(self.conv2(h))
        return h + self.res_conv(x)


class UNet1D(nn.Module):
    """
    A flexible 1D UNet for diffusion over sequences.
    This version downsamples twice and upsamples twice, with skip connections at each scale.
    """
    def __init__(self, in_channels=1, base_channels=64, time_embed_dim=128):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # --- Encoder ---
        # Initial convolution and residual block (no downsampling yet)
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.resblock1 = ResidualBlock1D(base_channels, base_channels, time_embed_dim)
        # First downsampling block: downsample from base_channels to base_channels*2
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.resblock2 = ResidualBlock1D(base_channels * 2, base_channels * 2, time_embed_dim)
        # Second downsampling block: downsample from base_channels*2 to base_channels*4
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        # Bottleneck residual block
        self.bottleneck = ResidualBlock1D(base_channels * 4, base_channels * 4, time_embed_dim)

        # --- Decoder ---
        # First upsampling block: upsample from base_channels*4 to base_channels*2
        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.resblock3 = ResidualBlock1D(base_channels * 2, base_channels * 2, time_embed_dim)
        # Second upsampling block: upsample from base_channels*2 to base_channels
        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.resblock4 = ResidualBlock1D(base_channels, base_channels, time_embed_dim)
        # Final convolution to bring features back to in_channels
        self.conv2 = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x, t):
        """
        x: Tensor of shape (B, 1, L) — our 1D sequence.
        t: Tensor of shape (B,) — diffusion timesteps.
        """
        # Create timestep embeddings (B, time_embed_dim, 1)
        t_emb = get_timestep_embedding(t, self.time_embed_dim).unsqueeze(-1)
        
        # --- Encoder ---
        h1 = self.conv1(x)               # (B, base_channels, L)
        h1 = self.resblock1(h1, t_emb)     # (B, base_channels, L)
        # Save h1 for skip connection later.
        h2 = self.down1(h1)              # (B, base_channels*2, L/2)
        h2 = self.resblock2(h2, t_emb)     # (B, base_channels*2, L/2)
        # Save h2 for skip connection.
        h3 = self.down2(h2)              # (B, base_channels*4, L/2/2 = L/4)
        h3 = self.bottleneck(h3, t_emb)    # (B, base_channels*4, L/4)

        # --- Decoder ---
        h4 = self.up2(h3)                # (B, base_channels*2, L/2)
        # Add skip connection from h2
        h4 = h4 + h2                   # (B, base_channels*2, L/2)
        h4 = self.resblock3(h4, t_emb)   # (B, base_channels*2, L/2)
        h5 = self.up1(h4)                # (B, base_channels, L)
        # Add skip connection from h1
        h5 = h5 + h1                    # (B, base_channels, L)
        h5 = self.resblock4(h5, t_emb)    # (B, base_channels, L)
        out = self.conv2(self.activation(h5))  # (B, in_channels, L)
        return out


# ---------------------------
# Dataset: Create sine-wave sequences
# ---------------------------
class SinusoidDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        """
        Each sample is a sine wave with random amplitude, frequency and phase.
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        # Precompute a linspace over one period (0 to 2pi)
        self.x = torch.linspace(0, 2 * math.pi, seq_length)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        amplitude = torch.rand(1) * 10.0            # amplitude in [0, 2]
        frequency = torch.rand(1) * 3.0 + 0.5        # frequency in [0.5, 3.5]
        phase = torch.rand(1) * 2 * math.pi          # phase in [0, 2pi]
        y = amplitude * torch.sin(frequency * self.x + phase)
        y = y.unsqueeze(0)  # shape becomes (1, seq_length)
        return y.float()


# ---------------------------
# Training function
# ---------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 64
    seq_length = 128
    learning_rate = 1e-4
    num_samples = 10000

    dataset = SinusoidDataset(num_samples, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate our 1D UNet diffusion model
    model = UNet1D(in_channels=1, base_channels=64, time_embed_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create a DDPM scheduler with (for example) 1000 diffusion steps.
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    model.train()
    for epoch in range(num_epochs):
        # Wrap the dataloader with tqdm for a progress bar.
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
        for step, batch in enumerate(pbar):
            batch = batch.to(device)  # shape (B, 1, seq_length)
            # Sample a random timestep for each example in the batch.
            t = torch.randint(0, scheduler.num_train_timesteps, (batch.shape[0],), device=device).long()
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
    torch.save(model.state_dict(), "unet1d_sinusoid.pt")
    return model, scheduler


def load_model(device):
    model = UNet1D(in_channels=1, base_channels=64, time_embed_dim=128)
    state_dict = torch.load("unet1d_sinusoid.pt", map_location=device)
    model.load_state_dict(state_dict)
    
    return model.to(device)

# ---------------------------
# Inference / Sampling function
# ---------------------------
def sample_sequences(model, scheduler, num_samples, seq_length, device, y0=None):
    """
    Generates new sequences by running the reverse diffusion process.
    """
    model.eval()
    with torch.no_grad():
        # Start from random Gaussian noise
        sample = torch.randn((num_samples, 1, seq_length), device=device)
        # scheduler.timesteps is an iterable of timesteps in descending order
        for t in scheduler.timesteps:
            # For each diffusion step, create a batch of the current timestep
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            # Predict the noise residual
            noise_pred = model(sample, t_batch)
            # Compute the previous sample (one denoising step)
            sample = scheduler.step(noise_pred, t, sample)["prev_sample"]
            
            if y0 is not None:
                sample[:, :, 0] = y0
                
        return sample


# ---------------------------
# Main script: Train, then sample and plot generated sequences.
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train the model (or load a pre-trained model if available)
    model, scheduler = train_model()

    # Generate some sequences via inference
    num_gen = 5
    seq_length = 128
    generated = sample_sequences(model, scheduler, num_gen, seq_length, device)
    generated = generated.cpu().numpy()

    # Plot the generated sequences
    x_axis = np.linspace(0, 2 * math.pi, seq_length)
    fig, ax = plt.subplots()
    
    for i in range(num_gen):
        ax.plot(x_axis, generated[i, 0], label=f"Seq {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("Amplitude")
    
    ax.legend()
    plt.savefig(f"generated_chatgpt.png")

if __name__ == '__main__':
    # main()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    seq_length = 128
    num_gen = 5
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    generated = sample_sequences(model, scheduler, num_gen, seq_length, device, y0=0.0)
    generated = generated.cpu().numpy()

    # Plot the generated sequences
    x_axis = np.linspace(0, 2 * math.pi, seq_length)
    fig, ax = plt.subplots()
    
    for i in range(num_gen):
        ax.plot(x_axis, generated[i, 0], label=f"Seq {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("Amplitude")
    
    ax.legend()
    plt.savefig(f"generated_chatgpt.png")

