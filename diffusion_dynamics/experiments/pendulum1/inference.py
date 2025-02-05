import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D
import torch
from diffusion_dynamics.experiments.pendulum1.train_model import PendulumModel
from diffusion_dynamics.simulation.systems import Pendulum, PendulumParams, PendulumRenderElement
import os
import matplotlib.pyplot as plt
# from spatialmath.base import angle_wrap
from diffusion_dynamics.simulation.animation import RenderEnvironment
from diffusion_dynamics.simulation.simulator import simulate_dynamical_system

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    trained_dataset_params = PendulumModel.load_trained_model(
        "/workspace/diffusion_dynamics/experiments/pendulum1/saved_models/pendulum_model1_nocontrols2"
    )
    
    mu_np = trained_dataset_params["data_mean"]
    sigma_np = trained_dataset_params["data_std"]
    
    if mu_np is not None and sigma_np is not None:
        mu_pt = torch.squeeze(torch.tensor(mu_np, device=device))
        sigma_pt = torch.squeeze(torch.tensor(sigma_np, device=device))
        
    PendulumModel.model.to(device)
    PendulumModel.model.eval()
    
    n_samples = 1
    seq_len = 128
    dt = 0.05

    with torch.no_grad():
        sample = torch.randn((n_samples, PendulumModel.n_channels, seq_len), device=device)
        
        for t in PendulumModel.scheduler.timesteps:
            sample[0, 0, 0] = torch.tanh((torch.pi - mu_pt[0]) / sigma_pt[0])
            sample[0, 1, 0] = torch.tanh((0.01 - mu_pt[1]) / sigma_pt[1])
        
            # sample[0, 0, -1] = torch.tanh(torch.pi/2 * sigma_pt[0] + mu_pt[0])
            # sample[0, 2, :] = torch.tanh(0.0 * sigma_pt[2] + mu_pt[2])
            
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            noise_pred = PendulumModel.model(sample, t_batch)
            sample = PendulumModel.scheduler.step(noise_pred, t, sample)["prev_sample"]
    
    sample = sample.cpu().numpy()
    sample = NumpyDataset1D.unnormalize(sample, mu_np, sigma_np)
    
    xs = sample[0, :2, :].T
    us = sample[0, 2:, :].T
    
    # Simulate real system
    system = Pendulum(
        PendulumParams(m=1, l=1, b=0.5)
    )
    
    ts_sim, xs_sim, us_sim = simulate_dynamical_system(system, tf=seq_len*dt, x0=xs[0], u=lambda t, x: us[int(t/dt)], dt=dt)
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 5))
    
    ax[0].plot(ts_sim[:-1], angle_wrap(xs_sim[:-1, 0]), label=r"$\theta$", c="r", linestyle="--")
    ax[0].plot(ts_sim[:-1], xs_sim[:-1, 1], label=r"$\dot{\theta}$", c="b", linestyle="--")
    
    ax[0].plot(ts_sim[:-1], angle_wrap(xs[:, 0]), label=r"$\theta_{pred}$", c="r")
    ax[0].plot(ts_sim[:-1], xs[:, 1], label=r"$\dot{\theta}_{pred}$", c="b")
    
    ax[1].plot(ts_sim[:-1], np.zeros_like(ts_sim[:-1]), label=r"$u$", c="g", linestyle="--")
    ax[1].plot(ts_sim[:-1], us[:, 0], label=r"$u_{pred}$", c="g")
    
    ax[0].legend()
    ax[1].legend()
    
    fig.suptitle("Damped pendulum system (No control input)")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ts = dt * np.arange(0, seq_len, 1)
   
    ax.grid(True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    
    env = RenderEnvironment(fig, ax)
    env.add_element(
        PendulumRenderElement
    )
    
    env.render(t_range=(0, seq_len*dt), t_history=ts, X_history=xs, U_history=us, fps=30, repeat=True)