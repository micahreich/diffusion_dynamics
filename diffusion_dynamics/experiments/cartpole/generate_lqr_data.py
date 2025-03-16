import torch
from torch.distributions import Uniform
import numpy as np
from typing import Any, Tuple, Callable
from diffusion_dynamics.simulation.systems import CartPole
from diffusion_dynamics.simulation.simulator import simulate_batch
from diffusion_dynamics.utils import torch_to_numpy, numpy_to_torch
from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
from diffusion_dynamics.models.utils import TensorDataset1D
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import time
import os


if __name__ == "__main__":        
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
    
    N = 10_000
    print(f"Simulating {N} CartPoles with LQR stabilization...")
    
    start = time.perf_counter()
    
    x0_dist = (-1, 1)
    theta0_range = (torch.pi-0.55, torch.pi+0.55)
    v0_range = (-1, 1)
    omega0_range = (-1, 1)
    
    x0_dist = Uniform(
        low=torch.tensor([x0_dist[0], theta0_range[0], v0_range[0], omega0_range[0]], dtype=torch.float32),
        high=torch.tensor([x0_dist[1], theta0_range[1], v0_range[1], omega0_range[1]], dtype=torch.float32)
    )
    
    x0 = x0_dist.sample((N,))
    u = lambda _t, x: ubar - (K @ (x - xbar).T).T
    
    ts_batch, x_hist_batch, u_hist_batch = simulate_batch(cart_pole, 5.0, 0.02, u, x0)
    
    print(f"Time taken for {N} CartPoles: {time.perf_counter() - start : .3f}s")
    
    dataset = TensorDataset1D(x_hist=x_hist_batch,
                              u_hist=u_hist_batch)
    
    dirname = os.path.dirname(os.path.abspath(__file__))
    torch.save(dataset, f"{dirname}/cartpole_lqr_data.pt")
    