from diffusion_dynamics.simulation.systems import Pendulum, PendulumParams, PendulumRenderElement
from diffusion_dynamics.simulation.simulator import simulate_dynamical_system
import jax.numpy as jnp
import jax
import numpy as np
from scipy.linalg import solve_continuous_are
from diffusion_dynamics.simulation.animation import RenderEnvironment
import matplotlib.pyplot as plt
import multiprocessing
import time
from tqdm import tqdm
from datetime import datetime
import pytz
import os


if __name__ == "__main__":
    N_training_trajectories = 12_000
    seq_len = 128
    dt = 0.05 # s
    tf = dt * (seq_len - 1) # s
    
    system = Pendulum(
        PendulumParams(m=1, l=1, b=0.5)
    )
    
    # df_dx = jax.jacfwd(system.continuous_dynamics, argnums=0)
    # df_du = jax.jacfwd(system.continuous_dynamics, argnums=1)
    # x0 = jnp.array([jnp.pi, 0.0])
    # u0 = jnp.array([0.0])
    
    # A = df_dx(x0, u0)
    # B = df_du(x0, u0)
    # Q = jnp.eye(system.nx)
    # R = 0.1 * jnp.eye(system.nu)
    # P = solve_continuous_are(A, B, Q, R)
    # K = jnp.linalg.inv(R) @ B.T @ P
    
    training_data = np.empty(shape=(N_training_trajectories, system.nx + system.nu, seq_len))

    for i in tqdm(range(N_training_trajectories)):
        theta0 = np.random.normal(loc=np.pi, scale=0.2)
        omega0 = np.random.normal(loc=0.0, scale=1.0)
        
        ts, xs, us = simulate_dynamical_system(
            sys=system,
            tf=tf,
            x0=np.array([theta0, omega0]),
            u=lambda t, x: np.zeros(system.nu),
            dt=dt
        )
        
        padded_us = np.pad(us, ((0, 1), (0, 0)), mode='constant', constant_values=0.0)
        Y = np.hstack([xs, padded_us]).T
        training_data[i] = Y

    indices = np.random.choice(N_training_trajectories, size=5, replace=False)

    for i in indices:
        xs, us = training_data[i, :system.nx, :].T, training_data[i, system.nx:, :].T
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.grid(True)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        
        env = RenderEnvironment(fig, ax)
        env.add_element(
            PendulumRenderElement
        )
        fig.suptitle(f"Trajectory {i}")
        
        
        env.render(t_range=(0, tf), t_history=ts, X_history=xs, U_history=us, fps=30, repeat=False)

    nyc_tz = pytz.timezone('America/New_York')
    time_str = datetime.now(nyc_tz).strftime("%Y-%m-%d__%H-%M-%S")
    save_fpath = "/workspace/diffusion_dynamics/experiments/pendulum1/data"
    os.makedirs(save_fpath, exist_ok=True)
    save_fpath_full = os.path.join(save_fpath, f"pendulum1_nocontrols2_N={N_training_trajectories}_{time_str}.npy")        
    np.save(save_fpath_full, training_data)