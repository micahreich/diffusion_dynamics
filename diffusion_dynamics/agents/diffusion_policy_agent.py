import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pytz
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from diffusion_dynamics import utils
from diffusion_dynamics.agents.agent import BehaviorCloningAgent
from diffusion_dynamics.models.utils import (
    SaveModelParams,
    TensorDataset1D,
    TensorDataset1DStats,
)
from diffusion_dynamics.utils import np_logit, np_sigmoid


class ConditionalDiffusionAgent(BehaviorCloningAgent):
    def __init__(self, model: nn.Module, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.stats: Optional[TensorDataset1DStats] = None

    def train(
        self,
        dataset: Dataset,
        n_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        accumulation_steps=2,
        save_model_params: Optional[SaveModelParams] = None,
    ):
        self.stats = dataset.stats

        # Test forward pass of the model
        print(f"Testing {self.model.__class__.__name__} forward pass...")

        self.model.eval()

        with torch.no_grad():
            u_noisy = torch.randn(batch_size, dataset.stats.u_pred_len, dataset.stats.nu)

            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size, )).long()
            cond = torch.randn(batch_size, dataset.stats.obs_history_len * dataset.stats.nx)

            _ = self.model(u_noisy, t, cond)

        # Move the model to correct device (gpu/cpu)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Training {self.model.__class__.__name__} with {trainable_params} trainable parameters")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.to(device)
        ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Start training
        self.model.train()
        try:
            for epoch in range(n_epochs):
                pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
                epoch_loss = 0.0

                for step, batch in enumerate(pbar):
                    # Randomly sample timestep, add noise to batch
                    obs_hist, u_hist = batch
                    obs_hist = obs_hist.to(device)
                    u_hist = u_hist.to(device)

                    t = torch.randint(0, self.scheduler.config.num_train_timesteps, (u_hist.shape[0], ),
                                      device=device).long()

                    noise = torch.randn_like(u_hist)
                    noisy_u_hist = self.scheduler.add_noise(u_hist, noise, t)

                    # Predict added noise and perform backward pass
                    model_out = self.model(noisy_u_hist, t, obs_hist)

                    loss = F.mse_loss(model_out, noise)
                    loss.backward()

                    if (step + 1) % accumulation_steps == 0:
                        optimizer.step()
                        ema.update()

                        optimizer.zero_grad()

                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=f"{round(loss.item(), 4):.4f}", lr=learning_rate)

                epoch_loss /= len(dataloader)

        except KeyboardInterrupt:
            if save_model_params is not None:
                print("\nTraining interrupted. Do you want to save the model? (y/n): ", end="")
                response = input().strip().lower()
                if response == 'n':
                    return

        if save_model_params is not None:
            self._save_model(save_model_params)

    def sample(self, cond, num_inference_steps=None):
        original_device = cond.device
        device = utils.get_torch_device()

        n_samples = cond.shape[0]
        sample = torch.randn((n_samples, self.stats.u_pred_len, self.stats.nu), device=device)
        cond = cond.to(device)

        assert cond.shape == (n_samples, self.stats.nx * self.stats.obs_history_len), \
            f"cond shape {cond.shape} must be (n_samples, nx * obs_history_len) = ({n_samples}, {self.stats.nx * self.stats.obs_history_len})"

        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # scheduler.timesteps is an iterable of timesteps in descending order
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                # For each diffusion step, create a batch of the current timestep
                t_batch = torch.full((n_samples, ), t, device=device, dtype=torch.long)

                # Predict the noise residual
                model_out = self.model(sample, t_batch, cond)

                # Compute the previous sample (one denoising step)
                sample = self.scheduler.step(model_out, t, sample)["prev_sample"]

        if self.stats.normalized:
            sample = TensorDataset1D.unnormalize(sample, self.stats.u_min, self.stats.u_max)

        return sample.to(original_device)
