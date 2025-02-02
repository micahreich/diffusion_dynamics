import numpy as np
from typing import Any, Tuple, Callable
from scipy.linalg import block_diag
from dataclasses import dataclass
from diffusion_dynamics.simulation.systems import DynamicalSystem

def rk4_step(f: Callable, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:    
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3, u)
    
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def simulate_dynamical_system(sys: DynamicalSystem, tf: float,
                              x0: np.ndarray, u: Callable,
                              dt: float=1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.arange(0, tf, dt)
    if not np.allclose(ts[-1], tf):
        ts = np.append(ts, tf)
    
    X_history = np.empty((len(ts), sys.nx))
    U_history = np.empty((len(ts) - 1, sys.nu))
    
    X_history[0] = x0
    
    for i, t in enumerate(ts[:-1]):
        u0 = u(X_history[i])
        x_next = rk4_step(sys.continuous_dynamics, X_history[i], u0, dt)
        
        X_history[i+1] = x_next
        U_history[i] = u0
    
    return ts, X_history, U_history

if __name__ == "__main__":
    from systems import Pendulum, PendulumParams, PendulumRenderElement
    from animation import RenderEnvironment
    import matplotlib.pyplot as plt
    
    x0 = np.array([np.pi/2, 0])
    u = lambda x : np.array([0])
    
    sys = Pendulum(PendulumParams(m=1, l=1, b=1.0))

    ts, xs, us = simulate_dynamical_system(
        sys=sys,
        tf=10.0,
        x0=x0,
        u=u,
        dt=0.01
    )
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    env = RenderEnvironment(fig, ax)
    env.add_element(
        PendulumRenderElement
    )
    env.render(t_range=(0, 10), t_history=ts, X_history=xs, U_history=us, fps=30)