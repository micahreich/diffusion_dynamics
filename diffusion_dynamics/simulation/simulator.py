import numpy as np
from typing import Any, Tuple, Callable
from scipy.linalg import block_diag
from dataclasses import dataclass
from diffusion_dynamics.simulation.systems import DynamicalSystem
import torch


def rk4_step(f: Callable, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_dynamical_system(
    sys: DynamicalSystem,
    tf: float,
    x0: torch.Tensor,
    u: Callable,
    dt: float = 1e-2,
    log_data=True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x0.shape[0] == sys.nx, "Initial states must have shape (nx,)"
    assert u(torch.tensor(tf), x0).shape == (sys.nu,), "Control function must return a 1D tensor of shape (nu,)"

    if log_data:
        sys.clear_history()

    ts = torch.arange(0, tf, dt)
    tf = torch.tensor([tf], dtype=ts.dtype)

    if tf - ts[-1] > 10 * torch.finfo(ts.dtype).eps:
        ts = torch.cat([ts, tf])

    X_history = torch.empty((len(ts), sys.nx))
    U_history = torch.empty((len(ts) - 1, sys.nu))

    # Set initial state
    X_history[0] = x0

    for i, t in enumerate(ts[:-1]):
        dt = ts[i + 1] - t

        U_history[i] = u(t, X_history[i])
        X_history[i + 1] = rk4_step(sys.continuous_dynamics, X_history[i], U_history[i], dt)

    if log_data:
        sys.set_history(ts, X_history, U_history)

    return ts, X_history, U_history


if __name__ == "__main__":
    from systems import Pendulum, PendulumParams, PendulumRenderElement
    from animation import RenderEnvironment
    import matplotlib.pyplot as plt

    x0 = torch.tensor([np.pi / 2, 10])
    u = lambda t, x: torch.tensor([0])

    sys = Pendulum(PendulumParams(m=1, l=1, b=1.0))

    ts, xs, us = simulate_dynamical_system(sys=sys, tf=10.0, x0=x0, u=u, dt=0.03, log_data=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    env = RenderEnvironment(fig, ax)
    env.add_element(PendulumRenderElement, sys)
    env.render(
        t_range=(0, 10),
        fps=30,
        save_fpath="/workspace/diffusion_dynamics/simulation/pendulum.mp4",
    )
