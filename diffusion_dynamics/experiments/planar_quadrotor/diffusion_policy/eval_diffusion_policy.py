import copy
from typing import Any, Callable, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

from diffusion_dynamics.agents.diffusion_policy_agent import ConditionalDiffusionAgent
from diffusion_dynamics.experiments.planar_quadrotor.generate_lqr_data import (
    K,
    planar_quadrotor,
    ubar,
    xbar,
)
from diffusion_dynamics.models.utils import TensorDataset1D
from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
from diffusion_dynamics.simulation.simulator import simulate
from diffusion_dynamics.simulation.systems import PlanarQuadrotor
from diffusion_dynamics.utils import numpy_to_torch, torch_to_numpy

if __name__ == "__main__":
    x0 = torch.tensor([-4.0, 2.0, 0.3, 1.0, 1.0, 1.0], dtype=torch.float32)
    tf = 5.0
    dt = 0.02

    # Simulate nominal cartpole + LQR for 5 seconds
    def u_lqr(_t, x):
        return ubar - K @ planar_quadrotor.project_state(x - xbar)

    planar_quadrotor_lqr = copy.deepcopy(planar_quadrotor)
    ts_lqr, x_hist_lqr, u_hist_lqr = simulate(planar_quadrotor_lqr, tf, dt, u_lqr, x0, log=True)

    # Simulate the behavior cloning policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ConditionalDiffusionAgent.load_trained_model(
        '/home/dev/workspace/diffusion_dynamics/experiments/planar_quadrotor/diffusion_policy/planar_quadrotor_diffusion_policy1.pt'
    ).to(device)

    def obs_fn_policy(i, xhist):
        return TensorDataset1D.get_state_window(xhist, i, policy.stats.obs_history_len).flatten()

    def u_policy(_t, observation):
        action_sample = policy.sample(observation.unsqueeze(0), num_inference_steps=10)
        return action_sample[0, 0].to(observation.device)

    test_i = 0
    obs = obs_fn_policy(test_i, x_hist_lqr)
    upolicy = policy.sample(obs.unsqueeze(0), num_inference_steps=50)

    print("u_lqr", u_lqr(0, x_hist_lqr[test_i]))
    print("diffusion policy", upolicy)

    planar_quadrotor_bc = copy.deepcopy(planar_quadrotor)
    ts_bc, x_hist_bc, u_hist_bc = torch_to_numpy(
        *simulate(planar_quadrotor_bc, tf, dt, u_policy, x0, True, obs_fn_policy))

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(ts_bc, x_hist_bc[:, 0], label=r"$x$", color="blue")
    ax[0].plot(ts_bc, x_hist_bc[:, 1], label=r"$y$", color="red")
    ax[0].plot(ts_bc, x_hist_bc[:, 2], label=r"$\theta$", color="green")
    ax[0].plot(ts_bc, x_hist_bc[:, 3], label=r"$\dot{x}$", color="blue", alpha=0.4)
    ax[0].plot(ts_bc, x_hist_bc[:, 4], label=r"$\dot{y}$", color="red", alpha=0.4)
    ax[0].plot(ts_bc, x_hist_bc[:, 5], label=r"$\dot{\theta}$", color="green", alpha=0.4)

    ax[0].set_ylabel("States")
    ax[0].legend()

    ax[1].plot(ts_bc[:-1], u_hist_bc[:, 0], label=r"$u_1$", color="purple")
    ax[1].plot(ts_bc[:-1], u_hist_bc[:, 1], label=r"$u_2$", color="purple")

    ax[1].set_ylabel("Control")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # Render the simulation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_aspect('equal')

    env = PlotEnvironment(fig, ax)
    env.add_element(PlanarQuadrotor.PlotElement(env, planar_quadrotor_lqr, 'gray', alpha=0.4))
    env.add_element(PlanarQuadrotor.PlotElement(env, planar_quadrotor_bc, 'blue'))
    _ = env.render(t_range=(0, tf), fps=30)

    plt.show()
