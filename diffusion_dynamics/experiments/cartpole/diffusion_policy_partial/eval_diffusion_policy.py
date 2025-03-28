import copy
from typing import Any, Callable, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

from diffusion_dynamics.agents.diffusion_policy_agent import ConditionalDiffusionAgent
from diffusion_dynamics.experiments.cartpole.generate_lqr_data import (
    K,
    cart_pole,
    ubar,
    xbar,
)
from diffusion_dynamics.models.utils import TensorDataset1D
from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
from diffusion_dynamics.simulation.simulator import simulate
from diffusion_dynamics.simulation.systems import CartPole, DynamicalSystem
from diffusion_dynamics.utils import numpy_to_torch, torch_to_numpy

if __name__ == "__main__":
    x0 = torch.tensor([0.0, torch.pi - 0.5, 0.0, 0.0], dtype=torch.float32)

    # Simulate nominal cartpole + LQR for 5 seconds
    def u_lqr(_t, x):
        return ubar - K @ (x - xbar)

    cart_pole_lqr = copy.deepcopy(cart_pole)
    ts_lqr, x_hist_lqr, u_hist_lqr = simulate(cart_pole_lqr, 5.0, 0.02, u_lqr, x0, log=True)

    # Simulate the behavior cloning policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ConditionalDiffusionAgent.load_trained_model(
        '/home/dev/workspace/diffusion_dynamics/experiments/cartpole/diffusion_policy_partial/cartpole_diffusion_policy_partial1.pt'
    ).to(device)

    def obs_fn_policy(i, xhist):
        x_window = TensorDataset1D.get_state_window(xhist, i, policy.stats.obs_history_len)
        x_window_partial = x_window[:, :2]
        return x_window_partial.flatten()

    def u_policy(_t, observation):
        action_sample = policy.sample(observation.unsqueeze(0), num_inference_steps=10)
        return action_sample[0, 0].to(observation.device)

    test_i = 4
    obs = obs_fn_policy(test_i, x_hist_lqr)
    print(obs)
    upolicy = policy.sample(obs.unsqueeze(0), num_inference_steps=50)

    print("u_lqr", u_lqr(0, x_hist_lqr[test_i]))
    print("diffusion policy", upolicy)

    cart_pole_bc = copy.deepcopy(cart_pole)
    ts_bc, x_hist_bc, u_hist_bc = torch_to_numpy(*simulate(cart_pole_bc, 10.0, 0.05, u_policy, x0, True, obs_fn_policy))

    # Plot states and control
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(ts_bc, x_hist_bc[:, 0], label=r"$x$", color="blue")
    ax[0].plot(ts_bc, x_hist_bc[:, 2], label=r"$\dot{x}$", color="blue", alpha=0.4)
    ax[0].plot(ts_bc, x_hist_bc[:, 1], label=r"$\theta$", color="red")
    ax[0].plot(ts_bc, x_hist_bc[:, 3], label=r"$\dot{\theta}$", color="red", alpha=0.4)
    ax[0].grid(True)
    ax[0].set_ylabel("States")
    ax[0].legend()

    ax[1].grid(True)
    ax[1].plot(ts_bc[:-1], u_hist_bc, label="u", color="purple")
    ax[1].set_ylabel("Control")
    ax[1].legend()

    plt.show()

    # Render the simulation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_aspect('equal')

    env = PlotEnvironment(fig, ax)
    env.add_element(CartPole.PlotElement(env, cart_pole_bc, 'blue'))
    env.add_element(CartPole.PlotElement(env, cart_pole_lqr, 'green'))
    _ = env.render(t_range=(0, 10.), fps=30)

    plt.show()
