import torch
import numpy as np
from typing import Any, Tuple, Callable
from diffusion_dynamics.simulation.systems import DynamicalSystem
from diffusion_dynamics.utils import torch_to_numpy, numpy_to_torch
from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
from scipy.linalg import solve_continuous_are
import time
from tqdm import tqdm


def rk4_step(f: Callable, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_batch(sys: DynamicalSystem,
                   tf: float,
                   dt: float,
                   u: Callable,
                   x0: torch.Tensor,
                   obs_fn: Callable = lambda i, x_hist: x_hist[:, i, :]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N, nx = x0.shape
    
    # assert type(term_cond(0.0, x0) == bool)
    
    ts = torch.arange(0, tf + dt, dt)
        
    x_hist = torch.zeros(N, len(ts), nx)
    u_hist = torch.zeros(N, len(ts) - 1, sys.nu)
    
    assert nx == sys.nx, "Initial states must have shape (N, nx)"
    assert u(0.0, obs_fn(0, x_hist)).shape == (N, sys.nu), "Control function must return a tensor of shape (N, nu)"
    
    # term_idx_hist = torch.zeros(N, len(ts))
    
    # term_cond(0.0, x0)
    
    x_hist[:, 0, :] = x0
    
    for i, t in enumerate(tqdm(ts[1:], desc="Simulation progress", total=len(ts)-1)):
        observation = obs_fn(i, x_hist)
        u_hist[:, i, :] = u(t, observation)
                
        # Run one RK4 integration step
        x_hist[:, i + 1, :] = rk4_step(sys.batch_dynamics, x_hist[:, i, :], u_hist[:, i, :], dt)
        
        # Project state if necesarry to keep it in the manifold, if necesarry
        x_hist[:, i + 1, :] = sys.project_state(x_hist[:, i + 1, :])
    
    return ts, x_hist, u_hist

def simulate(sys: DynamicalSystem,
             tf: float,
             dt: float,
             u: Callable,
             x0: torch.Tensor,
             log: bool = True,
             obs_fn: Callable = lambda i, x_hist: x_hist[i]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u_modified = lambda t, x: u(t, x.squeeze(0)).unsqueeze(0)
    obs_fn_modified = lambda i, x_hist: obs_fn(i, x_hist.squeeze(0)).unsqueeze(0)
    
    ts, x_hist, u_hist = simulate_batch(sys, tf, dt, u_modified, x0.unsqueeze(0), obs_fn_modified)
    
    if log:
        sys.set_history(ts, x_hist.squeeze(0), u_hist.squeeze(0))
    
    return ts, x_hist.squeeze(0), u_hist.squeeze(0)

if __name__ == "__main__":
    from diffusion_dynamics.simulation.systems import CartPole
    import matplotlib.pyplot as plt
    
    print("Simulating 1 CartPole with LQR control")
    
    cart_pole = CartPole(params=CartPole.Params(1, 1, 1, 9.81))
    
    # Compute LQR gain matrix K
    xbar = torch.tensor([1.0, torch.pi, 0, 0], dtype=torch.float32)
    ubar = torch.tensor([0.], dtype=torch.float32)
    
    A = torch_to_numpy(torch.autograd.functional.jacobian(lambda _x : cart_pole.dynamics(_x, ubar), xbar))
    B = torch_to_numpy(torch.autograd.functional.jacobian(lambda _u : cart_pole.dynamics(xbar, _u), ubar))
    Q = np.eye(4)
    R = 0.1 * np.eye(1)
    P = solve_continuous_are(A, B, Q, R)
    
    K = numpy_to_torch(np.linalg.inv(R) @ B.T @ P)
    
    # Simulate cartpole for 5 seconds
    x0 = torch.tensor([0, torch.pi - 0.1, 0.0, 1.0], dtype=torch.float32)
    u = lambda t, x: ubar - K @ (x - xbar)
    
    ts, x_hist, u_hist = torch_to_numpy(*simulate(cart_pole, 5.0, 0.02, u, x0, log=True)) 
    
    # Plot states and control
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(ts, x_hist[:, 0], label=r"$x$", color="blue")
    ax[0].plot(ts, x_hist[:, 2], label=r"$\dot{x}$", color="blue", alpha=0.4)
    ax[0].plot(ts, x_hist[:, 1], label=r"$\theta$", color="red")
    ax[0].plot(ts, x_hist[:, 3], label=r"$\dot{\theta}$", color="red", alpha=0.4)
    
    
    ax[0].set_ylabel("States")
    ax[0].legend()
    
    ax[1].plot(ts[:-1], u_hist, label="u", color="purple")
    ax[1].set_ylabel("Control")
    ax[1].legend()
    
    plt.show()
    
    # Render the simulation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_aspect('equal')

    env = PlotEnvironment(fig, ax)
    env.add_element(CartPole.PlotElement(env, cart_pole))
    _ = env.render(
        t_range=(0, 5),
        fps=30
    )
    
    plt.show()
    
    N = 10_000
    print(f"Simulating {N} CartPoles with LQR control...")
    
    start = time.perf_counter()
    x0 = torch.tile(x0, dims=(N, 1))
    u = lambda t, x: ubar - (K @ (x - xbar).T).T
        
    ts_batch, x_hist_batch, u_hist_batch = simulate_batch(cart_pole, 5.0, 0.02, u, x0)
    
    print(f"Time taken for {N} CartPoles: {time.perf_counter() - start : .3f}s",)
    
    # Verify each trajectory is correct
    i = torch.randint(0, N, (1,))

    assert np.allclose(x_hist, torch_to_numpy(x_hist_batch[i].squeeze(1)))
    assert np.allclose(u_hist, torch_to_numpy(u_hist_batch[i].squeeze(1)))