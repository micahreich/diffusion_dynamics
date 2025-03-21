import einops
import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32, n_blocks=4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.Mish())
        
        for _ in range(n_blocks - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Mish())
        
        layers.append(nn.Linear(hidden_dim, action_dim * 2))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x) -> Normal:
        out = self.network(x)
        mean, log_var = out.chunk(2, dim=-1)
        
        return Normal(mean, torch.exp(log_var))