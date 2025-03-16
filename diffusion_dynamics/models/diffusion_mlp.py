import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.resnet import ResidualTemporalBlock1D, Conv1dBlock
from diffusers.models.downsampling import Downsample1D
from diffusers.models.upsampling import Upsample1D
from diffusion_dynamics.models.utils import TensorDataset1D, ConditionalDiffusionModel, TensorDataset1DStats
from diffusion_dynamics.utils import torch_to_numpy
import einops
from einops.layers.torch import Rearrange
from tqdm import tqdm
import pytz
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_timestep_embedding


# Sinusoidal positional embedding (same as your original code)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x is assumed to be a tensor of shape (batch,)
        return get_timestep_embedding(x, self.dim)


class ConditionalResidualBlockMLP(nn.Module):
    def __init__(self, in_features, out_features, cond_dim, cond_predict_scale=False):
        """
        Args:
            in_features: Dimension of the input features.
            out_features: Dimension of the output features.
            cond_dim: Dimension of the conditioning vector.
            cond_predict_scale: Whether the condition predicts both scale and bias.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.cond_predict_scale = cond_predict_scale

        # If cond_predict_scale, predict both scale and bias for each channel
        cond_out_dim = out_features * 2 if cond_predict_scale else out_features
        self.cond_fc = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_out_dim)
        )

        # If dimensions do not match, use a shortcut projection
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, cond):
        """
        Args:
            x: Tensor of shape (batch, in_features)
            cond: Conditioning tensor of shape (batch, cond_dim)
        Returns:
            Tensor of shape (batch, out_features)
        """
        h = self.fc1(x)
        cond_out = self.cond_fc(cond)  # (batch, out_features) or (batch, 2*out_features)
        if self.cond_predict_scale:
            # FiLM modulation: scale and bias
            scale, bias = torch.chunk(cond_out, 2, dim=-1)
            h = scale * h + bias
        else:
            h = h + cond_out

        h = F.mish(h)
        h = self.fc2(h)
        return h + self.shortcut(x)


class ResidualBlockMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

        # If dimensions do not match, use a shortcut projection
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.fc1(x)
        h = F.mish(h)
        h = self.fc2(h)
        return h + self.shortcut(x)


class ConditionalMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 cond_dim=1,
                 hidden_dim=128,
                 n_blocks=4,
                 cond_predict_scale=False,
                 use_film_conditioning=True):
        """
        Args:
            input_dim: Dimension of the input vector.
            cond_dim: Dimension of the condition vector.
            hidden_dim: Hidden dimension used in the network.
            n_blocks: Number of residual blocks.
            cond_predict_scale: Whether to use FiLM conditioning that predicts scale and bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        # Time embedding module: embed the diffusion timestep into a vector
        self.time_embed_dim = hidden_dim
        # self.cond_embed_dim = hidden_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_embed_dim * 4, self.time_embed_dim)
        )
        
        # The conditioning will be the concatenation of the time embedding and the external condition.        
        if not use_film_conditioning:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, self.cond_embed_dim),
                nn.Mish(),
                nn.Linear(self.cond_embed_dim, self.cond_embed_dim)
            )
            
            input_total_dim = input_dim + self.cond_embed_dim
        else:
            cond_total_dim = cond_dim + self.time_embed_dim
            input_total_dim = input_dim

            
        # Project input into the hidden dimension.
        self.input_fc = nn.Linear(input_total_dim, hidden_dim)

        # Create a stack of conditional residual blocks.
        self.blocks = nn.ModuleList([
            ConditionalResidualBlockMLP(
                in_features=hidden_dim,
                out_features=hidden_dim,
                cond_dim=cond_total_dim,
                cond_predict_scale=cond_predict_scale,
            ) if use_film_conditioning else \
            ResidualBlockMLP(
                in_features=hidden_dim,
                out_features=hidden_dim
            )
            for _ in range(n_blocks)
        ])

        # Final projection back to input dimension.
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, cond):
        """
        Args:
            x: Tensor of shape (batch, u_pred_len, nu) – the noisy sample.
            t: Tensor of shape (batch,) – the diffusion timesteps.
            cond: Tensor of shape (batch, cond_dim) – external condition.
        Returns:
            Tensor of shape (batch, input_dim) – predicted noise.
        """
        _, T, nu = x.shape
        assert T * nu == self.input_dim, f"Input dimension mismatch: {T * nu} != {self.input_dim}"
        
        # x has shape (batch, T, nu), we need to flatten it to (batch, T*nu) in case of multiple u predictions
        x = einops.rearrange(x, 'b h c -> b (h c)')
        
        # Embed the timestep.
        t_emb = self.time_mlp(t)  # (batch, time_embed_dim)
        # Combine time embedding with the external condition.
        
        cond_emb = torch.cat([t_emb, cond], dim=-1)

        # Project input to hidden dimension.
        h = self.input_fc(x)
        h = F.mish(h)
        
        # Pass through the residual blocks.
        for block in self.blocks:
            h = block(h, cond_emb)
            
        # Final projection to get back to the original input dimension.
        out = self.output_fc(h)
        
        out = einops.rearrange(out, 'b (h c) -> b h c', c=nu)
        return out


# Example usage:
if __name__ == '__main__':
    from diffusers.schedulers import DDPMScheduler
    import matplotlib.pyplot as plt

    batch_size = 128
    input_dim = 1
    cond_dim = 1
    n_blocks = 4

    mlp = ConditionalMLP(input_dim=input_dim,
                           cond_dim=cond_dim,
                           hidden_dim=128,
                           n_blocks=n_blocks,
                           cond_predict_scale=True)
    
    scheduler = DDPMScheduler(num_train_timesteps=1000,
                              clip_sample=False,
                            #   variance_type="fixed_small_log",
                              prediction_type="epsilon")
    
    # Dummy dataset
    modes = torch.tensor([10.0, -10.0])
    N = 2048
    u_pred_len = 1
    
    class MultiModalDataset(Dataset):
        def __init__(self, modes):
            self.modes = modes
            
            self.seeds = modes[torch.randint(0, len(modes), (N, 1))]
            self.dists = torch.randn(N, u_pred_len) + self.seeds
            
            self.u_pred_len = u_pred_len
            self.obs_history_len = 1
            self.nu = 1
            self.nx = 1
            
            self.stats = TensorDataset1DStats(self.nu, self.nx, self.u_pred_len, self.obs_history_len,
                                              N_samples=N, normalized=False)
            
        def __len__(self):
            return len(self.seeds)
        
        def __getitem__(self, idx):
            return self.seeds[idx], torch.reshape(self.dists[idx], shape=(u_pred_len, 1))
    
    # Dummy inputs:
    dataset = MultiModalDataset(modes)
    rand_idxs = torch.randint(0, len(dataset), (5,))
    
    for i in rand_idxs:
        print(f"Sample index {i}:", dataset[i])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_model = ConditionalDiffusionModel(
        mlp,
        scheduler
    )
    
    diffusion_model.train(
        dataset,
        n_epochs=100,
        batch_size=batch_size,
        learning_rate=1e-3,
        accumulation_steps=1
    )
    
    sample_cond = torch.cat([
        10 * torch.ones(128).reshape(-1, 1),
        -10 * torch.ones(128).reshape(-1, 1)
    ], dim=0)
    
    sample_dist = diffusion_model.sample(n_samples=sample_cond.shape[0],
                                         cond=sample_cond,
                                         data_stats=dataset.stats,
                                         device=device,
                                         num_inference_steps=200)

    sample_dist = torch_to_numpy(sample_dist.reshape(-1, 1))
    
    plt.hist(sample_dist.flatten(), bins=30, color='blue', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')
    plt.show()
    
    # for (cond, sample) in zip(sample_cond, sample_dist):
    #     print("Condition:", cond)
    #     print("Sample:", sample.flatten())
        



