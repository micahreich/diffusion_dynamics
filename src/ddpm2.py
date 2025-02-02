import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, UNet1DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# =============== HYPERPARAMETERS =============== #
NUM_TRAIN_SAMPLES = 100
SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 10_000
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS = 1

# =============== PREPARE DATASET & DATALOADER =============== #
dataset = 0.777 * torch.ones((NUM_TRAIN_SAMPLES, CHANNELS, SEQ_LENGTH), dtype=torch.float32)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============== DEFINE 1D UNET MODEL =============== #
model = UNet1DModel(
    sample_size=SEQ_LENGTH,  # The length of the sequence
    in_channels=CHANNELS,  # 1D input (single channel)
    out_channels=CHANNELS,  # Predict noise in 1D space
    layers_per_block=2,  # Depth of UNet
    block_out_channels=(32, 64, 128),  # Channels at each UNet level
    down_block_types=("DownBlock1D", "AttnDownBlock1D", "DownBlock1D"),  # Down blocks
    up_block_types=("UpBlock1D", "AttnUpBlock1D", "UpBlock1D"),  # Up blocks
    # mid_block_type="MidResTemporalBlock1D",
    # extra_in_hannels=7,
    # norm_num_groups=1,
).to(DEVICE)

# =============== DEFINE DIFFUSION NOISE SCHEDULER =============== #
scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="epsilon")

# =============== DEFINE LOSS FUNCTION & OPTIMIZER =============== #
optimizer = optim.AdamW(model.parameters(), lr=LR)

# =============== TRAINING LOOP =============== #
print("Starting training...")

PRINT_EVERY = 100

for epoch in range(EPOCHS):
    model.train()
    
    # Randomly sample a batch of data
    indices = torch.randint(0, len(dataset), (BATCH_SIZE,))
    
    batch = dataset[indices]
        
    # for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
    batch = batch.to(DEVICE)  # Move data to device
            
    noise = torch.randn_like(batch)  # Generate Gaussian noise
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch.shape[0],), device=DEVICE).long()

    # Add noise using the scheduler
    noisy_data = scheduler.add_noise(batch, noise, timesteps)

    # Predict the noise using UNet
    model_output = model(noisy_data, timesteps).sample

    # Compute loss (MSE between predicted and actual noise)
    loss = F.mse_loss(model_output.float(), noise.float())
    epoch_loss = loss.item()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch + 1) % PRINT_EVERY == 0:
        print(f"Iter [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.6f}")

print("Training complete!")

# # =============== GENERATE NEW SAMPLES =============== #
# print("Generating new sinusoidal sequences...")

# model.eval()
# num_samples = 10
# sample_shape = (num_samples, 1, SEQ_LENGTH)  # (batch, channels, sequence_length)
# generated_samples = torch.randn(sample_shape, device=DEVICE)  # Start from Gaussian noise

# for t in tqdm(reversed(range(scheduler.config.num_train_timesteps)), desc="Sampling"):
#     timesteps = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
#     with torch.no_grad():
#         noise_pred = model(generated_samples, timesteps).sample
#     generated_samples = scheduler.step(noise_pred, t, generated_samples).prev_sample

# generated_samples = generated_samples.cpu().squeeze(1).numpy()  # Remove channel dim

# # =============== PLOT GENERATED SAMPLES =============== #
# plt.figure(figsize=(10, 5))
# for i in range(num_samples):
#     plt.plot(generated_samples[i], label=f"Sample {i+1}")

# plt.title("Generated Sinusoidal Sequences")
# plt.xlabel("Time Step")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.savefig('generated_samples.png')
