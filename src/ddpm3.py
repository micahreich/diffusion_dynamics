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

import numpy as np

if __name__ == "__main__":
    NUM_TRAIN_SAMPLES = 10_000
    SEQ_LENGTH = 128
    BATCH_SIZE = 128
    N_ITERS = 10_000
    LR = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHANNELS = 1
    DDPM_NUM_STEPS = 1000
    
    N_EPOCHS = 100
    
    dataset = torch.rand((NUM_TRAIN_SAMPLES, 1)) * torch.ones((NUM_TRAIN_SAMPLES, SEQ_LENGTH), dtype=torch.float16)
    dataset = dataset.unsqueeze(1)
    
    print(dataset[0])
    
    # 0.5 * torch.ones((NUM_TRAIN_SAMPLES, CHANNELS, SEQ_LENGTH), dtype=torch.float32)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    print("Training data shape:", len(train_dataloader))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision='fp16',
        split_batches=True,
    )
    
    model = UNet1DModel(
        sample_size=SEQ_LENGTH,  # The length of the sequence
        in_channels=CHANNELS,  # 1D input (single channel)
        out_channels=CHANNELS,  # Predict noise in 1D space
        layers_per_block=2,  # Depth of UNet
        block_out_channels=(32, 64, 128),  # Channels at each UNet level
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),  # Down blocks
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D"),  # Up blocks
        # mid_block_type="MidResTemporalBlock1D",
        # extra_in_hannels=7,
        # norm_num_groups=1,
    ).to(DEVICE)
    
    ema_model = EMAModel(
        model.parameters(),
        decay=0.995,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75,
        model_cls=UNet1DModel,
        model_config=model.config,
    )
    
    weight_dtype = torch.float16
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=DDPM_NUM_STEPS, beta_schedule="linear")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-8,
    )
    
    # lr_scheduler = get_scheduler(
    #     "cosine",
    #     optimizer=optimizer,
    #     num_warmup_steps=500 * 2,
    #     num_training_steps=(len(train_dataloader) * N_EPOCHS),
    # )
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    torch.cuda.empty_cache()
    
    ema_model.to(accelerator.device)
    
    total_batch_size = BATCH_SIZE * accelerator.num_processes * 2
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / 2)
    max_train_steps = N_EPOCHS * num_update_steps_per_epoch

    print("***** Running training *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Num Epochs = {N_EPOCHS}")
    print(f"  Instantaneous batch size per device = {BATCH_SIZE}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {2}")
    print(f"  Total optimization steps = {max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)
    
    for epoch in range(first_epoch, N_EPOCHS):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_samples = batch.to(weight_dtype)
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_samples.shape, dtype=weight_dtype, device=clean_samples.device)
            bsz = clean_samples.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_samples.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_samples = noise_scheduler.add_noise(clean_samples, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_samples, timesteps).sample

                loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "step": global_step}
            logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
        progress_bar.close()
        accelerator.wait_for_everyone()
    
    print("Training complete!")
    
    accelerator.end_training()
    model.eval()
    
    unet = accelerator.unwrap_model(model)
    ema_model.store(unet.parameters())
    ema_model.copy_to(unet.parameters())

    with torch.no_grad():
        generator = torch.Generator(device=model.device).manual_seed(0)
        image = torch.randn((1, 1, SEQ_LENGTH), dtype=weight_dtype, device=unet.device)
        
        for t in noise_scheduler.timesteps:
            # 1. predict noise model_output
            model_output = unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = noise_scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = np.squeeze(image.cpu().numpy())
        
        print(image)
