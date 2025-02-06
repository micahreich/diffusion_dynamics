from diffusion_dynamics.simulation.systems import Pendulum, PendulumParams, PendulumRenderElement
from diffusion_dynamics.simulation.simulator import simulate_dynamical_system
import torch
from scipy.linalg import solve_continuous_are
from diffusion_dynamics.simulation.animation import RenderEnvironment
import matplotlib.pyplot as plt
import multiprocessing
import time
from tqdm import tqdm
from datetime import datetime
import pytz
import os
from torch.distributions import Uniform, Normal
import numpy as np
import torch.multiprocessing as mp
import torch.nn.functional as F

system = Pendulum(
    PendulumParams(m=1, l=1, b=1.0)
)

def gaussian_kernel(size, sigma):
    assert size % 2 == 1, "Kernel size must be odd"
    
    x = torch.arange(size) - size // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.view(1, 1, -1)  # Shape (out_channels, in_channels, kernel_size)

def generate_brownian_motion_paths(N, T, dt,
                                   u0_range=(-5, 5), dB_scale=1.0,
                                   smooth=False, kernel_sigma=False, kernel_size=None):    
    dB = dB_scale * torch.sqrt(torch.tensor(dt)) * torch.randn(N, T-1)
    B = torch.cat((torch.zeros(N, 1), torch.cumsum(dB, dim=1)), dim=1)
    
    if smooth:
        kernel = gaussian_kernel(kernel_size, kernel_sigma)
        B = F.conv1d(B.unsqueeze(1), kernel, padding=kernel_size // 2, groups=1).squeeze(1)
    
    d = torch.distributions.Uniform(*u0_range)
    
    return B + d.sample((N, 1))

N_training_trajectories = 10_000

seq_len = 128
dt = 0.05 # s
tf = dt * (seq_len-1) # s

us_brownian_motion_tensor = generate_brownian_motion_paths(N_training_trajectories, seq_len, dt,
                                                           u0_range=(-6, 6), dB_scale=3.0,
                                                           smooth=True, kernel_sigma=2.5, kernel_size=21)

theta_dist = Uniform(-torch.pi, torch.pi)
omega_dist = Normal(0, 1)

def worker(worker_i, n_proc, n_traj, queue):
    training_data = torch.zeros(size=(n_traj, system.nx + system.nu, seq_len))

    for i in range(n_traj): #, f"Worker {i}", position=i, leave=True, nrows=n_proc+1, mininterval=0.5)
        x0 = torch.cat([theta_dist.sample((1,)), omega_dist.sample((1,))])
        
        j = worker_i * n_traj + i
        
        _, xs, us = simulate_dynamical_system(
            sys=system,
            tf=tf,
            x0=x0,
            u=lambda t, x: us_brownian_motion_tensor[j, int(t/dt)].unsqueeze(0),
            dt=dt,
            log_data=False
        )

        # Pad the control sequence with zeros at the beginning
        padded_us = torch.nn.functional.pad(us, (0,0,1,0), mode='constant', value=0.0)
        Y = torch.cat([xs, padded_us], dim=1).T
        
        assert Y.shape == (system.nx + system.nu, seq_len)
        
        training_data[i] = Y
    
    queue.put(training_data)


if __name__ == "__main__":    
    n_proc = 10
    procs = []
    queue = mp.Queue()
    
    N_traj_per_proc = N_training_trajectories // n_proc
    
    print(f"Collecting data ({N_training_trajectories} trajectories) with {n_proc} processes...")
    
    with mp.Manager() as manager:
        queue = manager.Queue()

        procs = []
        for i in range(n_proc):
            if i == n_proc - 1:
                N_traj_per_proc += N_training_trajectories % (N_traj_per_proc * n_proc)
                
            p = mp.Process(target=worker, args=(i, n_proc, N_traj_per_proc, queue))
            p.start()
            procs.append(p)

        res = [queue.get() for _ in range(n_proc)]

        for p in procs:
            p.join()
    
    training_data = torch.cat(res, dim=0)
    
    print("Training data shape:", training_data.shape)
    
    system.set_history(torch.arange(0, seq_len*dt, dt), training_data[0, :system.nx, :].T, training_data[0, system.nx:, :].T)
    
    fig, ax = plt.subplots(figsize=(8, 10))
       
    ax.grid(True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    
    env = RenderEnvironment(fig, ax)
    env.add_element(
        PendulumRenderElement, system
    )
    
    _ = env.render(t_range=(0, (seq_len-1)*dt), fps=30, repeat=True)
    plt.show()
    
    nyc_tz = pytz.timezone('America/New_York')
    time_str = datetime.now(nyc_tz).strftime("%Y-%m-%d__%H-%M-%S")
    save_fpath = "/workspace/diffusion_dynamics/experiments/pendulum2/data"
    save_fpath_full = os.path.join(save_fpath, f"pendulum2_brownianU1_N={N_training_trajectories}_{time_str}.pt")        
    
    os.makedirs(save_fpath, exist_ok=True)
    torch.save(training_data, save_fpath_full)