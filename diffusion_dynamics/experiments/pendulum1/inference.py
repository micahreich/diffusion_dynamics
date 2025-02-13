import numpy as np
from diffusion_dynamics.models.ddpm_1d import UNet1DModel, UNet1D
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import NumpyDataset1D
import torch
from diffusion_dynamics.experiments.pendulum1.train_model import PendulumModel
from diffusion_dynamics.simulation.systems import (
    Pendulum,
    PendulumParams,
    PendulumRenderElement,
)
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from diffusion_dynamics.simulation.animation import RenderEnvironment
from diffusion_dynamics.simulation.simulator import simulate_dynamical_system
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def angle_wrap(theta):
#     return (theta + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    saved_model_params = {
        "save_fpath": "/workspace/diffusion_dynamics/experiments/pendulum1/saved_models",
        "save_model_name": "pendulum_model1_nocontrols3",
    }

    # Load the trained model
    pendulum_model = PendulumModel().load_trained_model(
        os.path.join(saved_model_params["save_fpath"], saved_model_params["save_model_name"])
    )

    pendulum_model.unet.to(device)
    pendulum_model.unet.eval()

    n_samples = 10
    seq_len = 64
    dt = 0.05

    pendulum_model.scheduler.set_timesteps(num_inference_steps=20)

    start_time = time.perf_counter()
    with torch.no_grad():
        sample = torch.randn((n_samples, pendulum_model.n_channels, seq_len), device=device)

        for t in pendulum_model.scheduler.timesteps:
            theta0, omega0 = 2.5, 0.7

            sample[:, 0, 0] = theta0
            sample[:, 1, 0] = omega0

            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            noise_pred = pendulum_model.unet(sample, t_batch)
            sample = pendulum_model.scheduler.step(noise_pred, t, sample)["prev_sample"]

    print(f"Inference ({n_samples} samples) took {time.perf_counter() - start_time :.3f} s")

    from data_collection import system

    sample = sample.cpu()
    xs = sample[0, : system.nx, :].T
    us = sample[0, system.nx :, :].T

    ts_sim, xs_sim, us_sim = simulate_dynamical_system(
        system,
        tf=(seq_len - 1) * dt,
        x0=xs[0],
        u=lambda t, x: us[int(t / dt)],
        dt=dt,
        log_data=False,
    )

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 5), sharex=True)
    ax01 = ax[0].twinx()

    print(ts_sim[:5], ts_sim[-5:])

    ax[0].plot(ts_sim, xs_sim[:, 0], label=r"$\theta$", c="r", linestyle="--")
    ax[0].plot(ts_sim, xs[:, 0], label=r"$\theta_{dm}$", c="r")
    ax[0].set_ylabel(r"$\theta$ [rad]")

    ax01.plot(ts_sim, xs_sim[:, 1], label=r"$\dot{\theta}$", c="b", linestyle="--")
    ax01.plot(ts_sim, xs[:, 1], label=r"$\dot{\theta}_{dm}$", c="b")
    ax01.set_ylabel(r"$\dot{\theta}$ [rad/s]")

    ax[1].plot(ts_sim, torch.zeros_like(ts_sim), label=r"$u$", c="g", linestyle="--")
    ax[1].plot(ts_sim, us[:, 0], label=r"$u_{pred}$", c="g")
    ax[1].set_ylabel(r"$\tau$ [Nm]")

    ax[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker='o',
                color='w',
                markerfacecolor='red',
                markersize=10,
                label=r"$\theta$",
            ),
            Line2D(
                [0],
                [0],
                marker='o',
                color='w',
                markerfacecolor='blue',
                markersize=10,
                label=r"$\dot{\theta}$",
            ),
        ],
        loc="upper right",
    )

    ax[1].legend()
    ax[1].set_xlabel("t [s]")

    fig.suptitle("Damped pendulum system (No control input)")
    plt.savefig("pendulum1_nocontrols3.png")
    plt.show()

    # Simulate the diffusion model trajectory
    system.set_history(ts_sim, xs, us[1:])

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.grid(True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    env = RenderEnvironment(fig, ax)
    env.add_element(PendulumRenderElement, system)

    _ = env.render(
        t_range=(0, (seq_len - 1) * dt), fps=30, repeat=True
    )  # , save_fpath="/workspace/diffusion_dynamics/experiments/pendulum1/rollout.mp4")
    plt.show()
