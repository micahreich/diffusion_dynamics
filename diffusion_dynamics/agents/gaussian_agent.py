import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Tuple

import einops
import numpy as np
import pytz
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_dynamics import utils
from diffusion_dynamics.agents.agent import BehaviorCloningAgent
from diffusion_dynamics.models.utils import (
    SaveModelParams,
    TensorDataset1D,
    TensorDataset1DStats,
)
from diffusion_dynamics.utils import np_logit, np_sigmoid


class ConditionalGaussianAgent(BehaviorCloningAgent):
    def __init__(self, model: nn.Module, ent_coeff: float = 0.0):
        self.model = model
        self.ent_coeff = ent_coeff

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

        # Test forward pass
        print(f"Testing {self.model.__class__.__name__} forward pass...")

        self.model.eval()

        with torch.no_grad():
            cond = torch.randn(batch_size, dataset.stats.obs_history_len * dataset.stats.nx)
            _ = self.model(cond)

        # Move the model to correct device (gpu/cpu)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Training {self.model.__class__.__name__} with {trainable_params} trainable parameters")

        device = utils.get_torch_device()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        try:
            for epoch in range(n_epochs):
                pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
                epoch_loss = 0.0

                for step, batch in enumerate(pbar):
                    # Randomly sample timestep, add noise to batch
                    obs_hist, u_hist = batch
                    obs_hist = obs_hist.to(device)

                    u_hist = u_hist.to(device)
                    u_hist = einops.rearrange(u_hist, 'b h c -> b (h c)')

                    out_dist = self.model(obs_hist)
                    out_log_prob = out_dist.log_prob(u_hist)

                    out_actions = out_dist.rsample()
                    out_actions = einops.rearrange(out_actions, 'b (h c) -> b h c', c=self.stats.nu)

                    # loss = F.mse_loss(out_actions, u_hist)
                    # loss.backward()
                    loss = -out_log_prob.mean()
                    # loss += -self.ent_coeff * out_dist.entropy().mean()
                    loss.backward()

                    if (step + 1) % accumulation_steps == 0:
                        optimizer.step()
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

    def sample(self, cond, deterministic=False):
        original_device = cond.device
        device = utils.get_torch_device()

        cond = cond.to(device)
        out_dist = self.model(cond)

        if deterministic:
            sample = out_dist.mean.detach()
        else:
            sample = out_dist.sample()

        out = einops.rearrange(sample, 'b (h c) -> b h c', c=self.stats.nu)

        if self.stats.normalized:
            out = TensorDataset1D.unnormalize(out, self.stats.u_min, self.stats.u_max)

        return out.to(original_device)
