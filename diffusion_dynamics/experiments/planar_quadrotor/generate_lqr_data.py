import os
import time
from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_continuous_are
from torch.distributions import Normal, Uniform

from diffusion_dynamics.models.utils import TensorDataset1D
from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
from diffusion_dynamics.simulation.simulator import simulate_batch
from diffusion_dynamics.simulation.systems import CartPole, PlanarQuadrotor
from diffusion_dynamics.utils import numpy_to_torch, torch_to_numpy

planar_quadrotor = PlanarQuadrotor(params=PlanarQuadrotor.Params(m=1.0, I=1.0, r=0.5, g=9.81))
# Compute LQR gain matrix K
xbar = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
ubar = torch.tensor([
    planar_quadrotor.params.m * planar_quadrotor.params.g / 2, planar_quadrotor.params.m * planar_quadrotor.params.g / 2
],
                    dtype=torch.float32)

A = torch_to_numpy(torch.autograd.functional.jacobian(lambda _x: planar_quadrotor.dynamics(_x, ubar), xbar))
B = torch_to_numpy(torch.autograd.functional.jacobian(lambda _u: planar_quadrotor.dynamics(xbar, _u), ubar))
Q = np.diag([10., 10., 100., 1., 1., 1.])
R = np.eye(2)
P = solve_continuous_are(A, B, Q, R)

K = numpy_to_torch(np.linalg.inv(R) @ B.T @ P)

if __name__ == "__main__":
    N = 10_000
    print(f"Simulating {N} {planar_quadrotor.name} with LQR stabilization...")

    start = time.perf_counter()

    r = torch.tensor([1, 1, 0.2, 1, 1, 1], dtype=torch.float32)
    x0_dist = Uniform(low=xbar - r, high=xbar + r)
    u_noise = Normal(loc=torch.zeros(N, planar_quadrotor.nu), scale=0.2 * torch.ones(N, planar_quadrotor.nu))

    x0 = x0_dist.sample((N, ))
    u = lambda _t, x: ubar - torch.einsum('i j, n j -> n i', K, x - xbar) + u_noise.sample()

    ts_batch, x_hist_batch, u_hist_batch = simulate_batch(planar_quadrotor, 6.0, 0.05, u, x0)

    print(f"Time taken for {N} {planar_quadrotor.name}: {time.perf_counter() - start : .3f}s")
    print(f"\tX shape: {x_hist_batch.shape}")
    print(f"\tU shape: {u_hist_batch.shape}")

    dirname = os.path.dirname(os.path.abspath(__file__))
    torch.save({'x_hist': x_hist_batch, 'u_hist': u_hist_batch}, f"{dirname}/planar_quadrotor_lqr_data.pt")

    # Look at one of the trajectories
    idx = torch.randint(0, N, (1, )).item()
    xhist_idx = x_hist_batch[idx, :, :]
    uhist_idx = u_hist_batch[idx, :]

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(ts_batch, xhist_idx[:, 0], label=r"$x$", color="blue")
    ax[0].plot(ts_batch, xhist_idx[:, 1], label=r"$y$", color="red")
    ax[0].plot(ts_batch, xhist_idx[:, 2], label=r"$\theta$", color="green")
    ax[0].plot(ts_batch, xhist_idx[:, 3], label=r"$\dot{x}$", color="blue", alpha=0.4)
    ax[0].plot(ts_batch, xhist_idx[:, 4], label=r"$\dot{y}$", color="red", alpha=0.4)
    ax[0].plot(ts_batch, xhist_idx[:, 5], label=r"$\dot{\theta}$", color="green", alpha=0.4)

    ax[0].set_ylabel("States")
    ax[0].legend()

    ax[1].plot(ts_batch[:-1], uhist_idx[:, 0], label=r"$u_1$", color="purple")
    ax[1].plot(ts_batch[:-1], uhist_idx[:, 1], label=r"$u_2$", color="purple")

    ax[1].set_ylabel("Control")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
