import os
from datetime import datetime

import einops
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.downsampling import Downsample1D
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.resnet import Conv1dBlock, ResidualTemporalBlock1D
from diffusers.models.upsampling import Upsample1D
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_dynamics.models.utils import TensorDataset1D


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups if in_channels >= n_groups else in_channels),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2

        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''

        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)  # [ batch_size x cond_channels x 1 ]

        if self.cond_predict_scale:
            # scale, bias = torch.chunk(embed, 2, dim=1)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            # scale, bias = torch.chunk(embed, 2, dim=1)

            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return get_timestep_embedding(x, self.dim)


class ConditionalUnet1D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 cond_dim=1,
                 base_channels=64,
                 dim_mults=[1, 2, 4],
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=False):
        super().__init__()

        self.in_channels = in_channels
        self.cond_dim = cond_dim

        dims = [in_channels, *map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        self.time_embed_dim = base_channels
        cond_dim += self.time_embed_dim

        # Embed the time step
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_embed_dim * 4, self.time_embed_dim),
        )

        # UNet1D Blocks
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(dim_in,
                                               dim_out,
                                               cond_dim,
                                               kernel_size=kernel_size,
                                               n_groups=n_groups,
                                               cond_predict_scale=cond_predict_scale),
                    ConditionalResidualBlock1D(dim_out,
                                               dim_out,
                                               cond_dim,
                                               kernel_size=kernel_size,
                                               n_groups=n_groups,
                                               cond_predict_scale=cond_predict_scale),
                    Downsample1D(dim_out, use_conv=True) if not is_last else nn.Identity(),
                ]))

        mid_dim = dims[-1]
        self.mids = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim,
                                       mid_dim,
                                       cond_dim,
                                       kernel_size=kernel_size,
                                       n_groups=n_groups,
                                       cond_predict_scale=cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim,
                                       mid_dim,
                                       cond_dim,
                                       kernel_size=kernel_size,
                                       n_groups=n_groups,
                                       cond_predict_scale=cond_predict_scale)
        ])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(dim_out * 2,
                                               dim_in,
                                               cond_dim,
                                               kernel_size=kernel_size,
                                               n_groups=n_groups,
                                               cond_predict_scale=cond_predict_scale),
                    ConditionalResidualBlock1D(dim_in,
                                               dim_in,
                                               cond_dim,
                                               kernel_size=kernel_size,
                                               n_groups=n_groups,
                                               cond_predict_scale=cond_predict_scale),
                    Upsample1D(dim_in, use_conv_transpose=True) if not is_last else nn.Identity(),
                ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(dims[1], dims[1], kernel_size=kernel_size),
            nn.Conv1d(dims[1], in_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x, cond have shape (batch_size, horizon_length, dimension)
        x = einops.rearrange(x, 'b h d -> b d h')

        # Embed the timestep to dimension `self.time_embed_dim`
        t_emb = self.time_mlp(t)
        cond_emb = torch.cat([t_emb, cond], dim=-1)

        # UNet1D forward pass
        h = []

        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, cond_emb)
            x = resnet2(x, cond_emb)
            h.append(x)
            x = downsample(x)

        for i, mid_module in enumerate(self.mids):
            x = mid_module(x, cond_emb)

        for i, (resnet, resnet2, upsample) in enumerate(self.ups):
            v = h.pop()
            x = torch.cat((x, v), dim=1)
            x = resnet(x, cond_emb)
            x = resnet2(x, cond_emb)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b d h -> b h d')
        return x


class ConditionalUnet1DModel:
    def __init__(self, unet=None, scheduler=None):
        self.unet = unet
        self.scheduler = scheduler

    def to(self, device):
        if self.unet is not None:
            self.unet = self.unet.to(device)

    def train(
        self,
        dataset: TensorDataset1D,
        n_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        accumulation_steps=2,
        save_model_params=None,
    ):
        assert self.unet is not None, "model must be instantiated before training"
        assert self.scheduler is not None, "noise scheduler must be instantiated before training"

        if save_model_params is not None:
            assert "save_fpath" in save_model_params, "model save filepath must be provided"

        print(f"Testing {self.unet.__class__.__name__} forward pass...")

        self.unet.eval()
        with torch.no_grad():
            u_noisy = torch.randn(batch_size, dataset.stats.u_pred_len, dataset.stats.nu)

            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size, )).long()
            cond = torch.randn(batch_size, dataset.stats.obs_history_len * dataset.stats.nx)

            out = self.unet(u_noisy, t, cond)

        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f"Training {self.unet.__class__.__name__} with {trainable_params} trainable parameters")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        _, _u = dataset[0]
        u_pred_len, nu = _u.shape

        assert u_pred_len == dataset.stats.u_pred_len, "dataset u_pred_len must match the model's expected u_pred_len"
        assert nu == dataset.stats.nu == self.unet.in_channels, "dataset nu must match the model's expected nu"

        # Instantiate our 1D UNet diffusion model
        self.unet.to(device)
        self.unet.train()

        optimizer = torch.optim.Adam(self.unet.parameters(), lr=learning_rate)
        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

        try:
            for epoch in range(n_epochs):
                pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
                epoch_loss = 0.0
                # current_lr = optimizer.param_groups[0]['lr']

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
                    model_out = self.unet(noisy_u_hist, t, obs_hist)

                    # if self.scheduler.config.prediction_type == "epsilon":
                    loss = F.mse_loss(model_out, noise)

                    loss.backward()

                    if (step + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=loss.item(), lr=learning_rate)

                epoch_loss /= len(dataloader)
                # lr_scheduler.step(epoch_loss)

        except KeyboardInterrupt:
            if save_model_params is not None:
                print("\nTraining interrupted. Do you want to save the model? (y/n): ", end="")
                response = input().strip().lower()
                if response == 'n':
                    return

        if save_model_params is not None:
            self._save_model(save_model_params, dataset)

    def _save_model(self, save_model_params, train_dataset: TensorDataset1D):
        assert save_model_params is not None, "save_model_params must be provided"

        # Save the trained model weights
        if "save_model_name" not in save_model_params:
            nyc_tz = pytz.timezone('America/New_York')
            time_str = datetime.now(nyc_tz).strftime("%Y-%m-%d__%H-%M-%S")
            save_model_params["save_model_name"] = f"unet1d_{time_str}"

        save_fpath_full = os.path.join(save_model_params["save_fpath"], save_model_params["save_model_name"])
        print(f"Saving model to {save_fpath_full}...")

        os.makedirs(save_fpath_full, exist_ok=True)
        torch.save(self.unet.state_dict(), os.path.join(save_fpath_full, "model.pt"))

    @classmethod
    def load_trained_model(cls, saved_model_fpath):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(os.path.join(saved_model_fpath, "model.pt"), map_location=device)

        m = cls()
        assert m.unet is not None, "model must be instantiated before loading a trained model"

        m.unet.load_state_dict(state_dict)

        return m
