import torch
from torch.distributions import Uniform
import numpy as np
from typing import Any, Tuple, Callable
from diffusion_dynamics.simulation.systems import CartPole
from diffusion_dynamics.simulation.simulator import simulate_batch
from diffusion_dynamics.utils import torch_to_numpy, numpy_to_torch
from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
from diffusion_dynamics.models.utils import TensorDataset1D, ConditionalDiffusionModel, SaveModelParams
from diffusion_dynamics.models.diffusion_mlp import ConditionalMLP
from diffusers.schedulers import DDPMScheduler
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import time
import os

if __name__ == "__main__":
    x_hist_len = 2
    u_pred_len = 4
    batch_size = 128

    # Load in dataset
    dirname = os.path.dirname(os.path.abspath(__file__))
    dataset: TensorDataset1D = torch.load(f"{dirname}/cartpole_lqr_data.pt")
    dataset.set_u_pred_len(u_pred_len)
    dataset.set_obs_history_len(x_hist_len)
    
    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mlp = ConditionalMLP(input_dim=CartPole.nu * u_pred_len,
                         cond_dim=CartPole.nx * x_hist_len,
                         hidden_dim=128,
                         n_blocks=4,
                         cond_predict_scale=True)
    
    scheduler = DDPMScheduler(num_train_timesteps=1000,
                              clip_sample=False,
                            #   variance_type="fixed_small_log",
                              prediction_type="epsilon")
    
    diffusion_model = ConditionalDiffusionModel(
        mlp,
        scheduler
    )
    
    diffusion_model.train(
        dataset,
        n_epochs=100,
        batch_size=batch_size,
        learning_rate=1e-3,
        accumulation_steps=1,
        save_model_params=SaveModelParams(
            save_full_fpath="/workspace/diffusion_dynamics/experiments/cartpole",
            save_model_name="cartpole_diffusion_policy",
        )
    )
    