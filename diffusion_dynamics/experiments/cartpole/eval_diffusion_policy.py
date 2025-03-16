import torch
import numpy as np
from typing import Any, Tuple, Callable
from diffusion_dynamics.simulation.systems import DynamicalSystem
from diffusion_dynamics.utils import torch_to_numpy, numpy_to_torch
from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
from diffusion_dynamics.simulation.systems import CartPole
from diffusion_dynamics.simulation.simulator import simulate
from diffusion_dynamics.models.utils import TensorDataset1D, ConditionalDiffusionModel, SaveModelParams
from scipy.linalg import solve_continuous_are
import time
import copy
import os
from matplotlib import pyplot as plt

from diffusion_dynamics.experiments.cartpole.generate_lqr_data import cart_pole, K, xbar, ubar
from diffusion_dynamics.experiments.cartpole.train_diffusion_policy import CartpoleDiffusionPolicy

if __name__ == "__main__":
    x0 = torch.tensor([0, torch.pi - 0.4, 0.0, 1.0], dtype=torch.float32)

    # Simulate nominal cartpole + LQR for 5 seconds
    
    def u_lqr(_t, x):
        return ubar - K @ (x - xbar)
    
    cart_pole_lqr = copy.deepcopy(cart_pole)
    ts_lqr, x_hist_lqr, u_hist_lqr = simulate(cart_pole_lqr, 5.0, 0.02, u_lqr, x0, log=True)
    
    # # Simulate diffusion policy controller for 5 seconds
    # dirname = os.path.dirname(os.path.abspath(__file__))
    # dataset: TensorDataset1D = torch.load(f"{dirname}/cartpole_lqr_data.pt")
    # dataset.set_u_pred_len(4)
    # dataset.set_obs_history_len(2)
    
    # print(dataset.stats)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_policy = CartpoleDiffusionPolicy.load_trained_model(
        "/workspace/diffusion_dynamics/experiments/cartpole/cartpole_diffusion_policy.pt"
    ).to(device)
    
    xs = torch.tile(torch.arange(10).reshape((-1, 1)), (1, 4))
    
    def obs_fn_diffusion_policy(i, xhist):
        return TensorDataset1D.get_state_window(xhist, i, 2).flatten()
    
    def u_diffusion_policy(_t, observation):
        action_sample = diffusion_policy.sample(1, observation.unsqueeze(0), device, 50)
        return action_sample[0, 0].to(observation.device)
    
    test_i = 5
    obs = obs_fn_diffusion_policy(test_i, x_hist_lqr)
    u = u_diffusion_policy(0, obs)
    
    print("u_diffusion_policy", u)
    print("u_lqr", u_lqr(0, x_hist_lqr[test_i]))

    # xs = torch.tile(torch.arange(10, dtype=torch.float32).reshape((-1, 1)), (1, 4))
    # obs = obs_fn_diffusion_policy(0, xs)
    # u = u_diffusion_policy(0, obs)

    
    cart_pole_dp = copy.deepcopy(cart_pole)
    ts_dp, x_hist_dp, u_hist_dp = torch_to_numpy(*simulate(cart_pole_dp, 5.0, 0.02, u_diffusion_policy, x0, True, obs_fn_diffusion_policy))
    
    # Plot states and control
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(ts_dp, x_hist_dp[:, 0], label=r"$x$", color="blue")
    ax[0].plot(ts_dp, x_hist_dp[:, 2], label=r"$\dot{x}$", color="blue", alpha=0.4)
    ax[0].plot(ts_dp, x_hist_dp[:, 1], label=r"$\theta$", color="red")
    ax[0].plot(ts_dp, x_hist_dp[:, 3], label=r"$\dot{\theta}$", color="red", alpha=0.4)
    
    
    ax[0].set_ylabel("States")
    ax[0].legend()
    
    ax[1].plot(ts_dp[:-1], u_hist_dp, label="u", color="purple")
    ax[1].set_ylabel("Control")
    ax[1].legend()
    
    plt.show()
    
    # Render the simulation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_aspect('equal')

    env = PlotEnvironment(fig, ax)
    env.add_element(CartPole.PlotElement(env, cart_pole_dp, 'blue'))
    env.add_element(CartPole.PlotElement(env, cart_pole_lqr, 'green'))
    _ = env.render(
        t_range=(0, 5),
        fps=30
    )
    
    plt.show()
