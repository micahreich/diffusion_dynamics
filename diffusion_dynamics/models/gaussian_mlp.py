import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from diffusion_dynamics.models.mlp import MLP


class GaussianMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32, n_blocks=4, logstd_min=-20, logstd_max=2):
        super().__init__()

        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

        # layers = []
        # layers.append(nn.Linear(state_dim, hidden_dim))
        # layers.append(nn.Mish())

        # for _ in range(n_blocks - 1):
        #     layers.append(nn.Linear(hidden_dim, hidden_dim))
        #     layers.append(nn.Mish())

        # layers.append(nn.Linear(hidden_dim, action_dim * 2))

        # print(f"Creating GaussianMLP with {n_blocks} blocks, {state_dim} state dim, {action_dim} action dim, {hidden_dim} hidden dim")

        # self.network = nn.Sequential(*layers)
        dim_list = [state_dim] + [hidden_dim] * n_blocks + [action_dim * 2]
        self.network = MLP(
            dim_list=dim_list,
            activation_type="Tanh",
            out_activation_type="Identity",
            use_layernorm=True,
        )

    def forward(self, x) -> TransformedDistribution:
        if torch.isnan(x).any():
            print("Input contains NaNs!")

        out = self.network(x)
        mean, log_std = out.chunk(2, dim=-1)

        # Debug: check for NaNs right after chunking
        if torch.isnan(mean).any():
            print("NaN detected in mean before transformation")
            print(x.amin(), x.amax())
        if torch.isnan(log_std).any():
            print("NaN detected in log_std before transformation")
            print(x.amin(), x.amax())

        log_std = F.tanh(log_std)
        log_std = (1 / 2 * log_std + 1 / 2) * (self.logstd_max - self.logstd_min) + self.logstd_min

        dist = Normal(mean, torch.exp(log_std))
        tf_dist = TransformedDistribution(dist, [TanhTransform()])

        return tf_dist
