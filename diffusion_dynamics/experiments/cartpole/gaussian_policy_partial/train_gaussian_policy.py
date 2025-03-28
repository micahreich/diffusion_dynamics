import os
from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.schedulers import DDPMScheduler
from scipy.linalg import solve_continuous_are
from torch.distributions import Uniform

from diffusion_dynamics.agents.gaussian_agent import ConditionalGaussianAgent
from diffusion_dynamics.models.gaussian_mlp import GaussianMLP
from diffusion_dynamics.models.utils import SaveModelParams, TensorDataset1D
from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
from diffusion_dynamics.simulation.simulator import simulate_batch
from diffusion_dynamics.simulation.systems import CartPole
from diffusion_dynamics.utils import numpy_to_torch, torch_to_numpy

if __name__ == "__main__":
    x_hist_len = 8
    u_pred_len = 1
    batch_size = 64

    # Load in dataset
    dataset_dict = torch.load("/home/dev/workspace/diffusion_dynamics/experiments/cartpole/cartpole_lqr_data.pt")
    x_hist_full = dataset_dict["x_hist"]
    x_hist_partial = x_hist_full[:, :, :2]

    dataset = TensorDataset1D(
        x_hist=x_hist_partial,
        u_hist=dataset_dict["u_hist"],
        obs_history_len=x_hist_len,
        u_pred_len=u_pred_len,
        normalize=True,
    )

    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp = GaussianMLP(state_dim=x_hist_len * dataset.stats.nx,
                      action_dim=u_pred_len * dataset.stats.nu,
                      hidden_dim=32,
                      n_blocks=4)

    agent = ConditionalGaussianAgent(mlp, ent_coeff=0.1)
    agent.train(
        dataset,
        n_epochs=10,
        batch_size=batch_size,
        learning_rate=1e-4,
        accumulation_steps=1,
        save_model_params=SaveModelParams(
            save_full_fpath="/home/dev/workspace/diffusion_dynamics/experiments/cartpole/gaussian_policy_partial",
            save_model_name="cartpole_gaussian_policy_partial1",
        ))
