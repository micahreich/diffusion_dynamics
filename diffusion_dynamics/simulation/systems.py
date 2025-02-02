import numpy as np
from typing import Any, Tuple
from scipy.linalg import block_diag
from dataclasses import dataclass
from diffusion_dynamics.utils import unpack_state
from diffusion_dynamics.simulation.animation import RenderEnvironment, RenderElement

# Base dynamical system class

class DynamicalSystem:
    def __init__(self, name: str, params: Any, nx: int, nu: int) -> None:
        self.name = name
        self.params = params
        self.nx = nx
        self.nu = nu
        
        self.x = np.zeros((1, nx))
            
    def M(self, q) -> np.ndarray: raise NotImplementedError
    def C(self, q, q_dot) -> np.ndarray: raise NotImplementedError
    def G(self, q) -> np.ndarray: raise NotImplementedError
    def B(self) -> np.ndarray: raise NotImplementedError

    def continuous_dynamics(self, x, u) -> np.ndarray:
        nxx = self.nx//2
    
        q, q_dot = unpack_state(x, self.nx)
        
        M = self.M(q)
        C = self.C(q, q_dot)
        G = self.G(q)
        B = self.B()
        
        q_ddot = np.linalg.solve(
            M.reshape((nxx, nxx)),
            (B @ u - C @ q_dot + G).reshape((-1,))
        )
    
        x_dot = np.concatenate((q_dot, q_ddot))
        return x_dot

# Pendulum system

@dataclass
class PendulumParams:
    m: float
    l: float
    b: float

class Pendulum(DynamicalSystem):
    def __init__(self, params: Any) -> None:
        self.g = 9.81
        super().__init__("Pendulum", params, nx=2, nu=1)
        
    def M(self, q) -> np.ndarray:
        m, l = self.params.m, self.params.l
        return np.array([m*l**2])
    
    def C(self, q, q_dot) -> np.ndarray:        
        b = self.params.b
        return np.array([b])
    
    def G(self, q) -> np.ndarray:
        m, l, g = self.params.m, self.params.l, self.g
        return -np.array([m*g*l*np.sin(q[0])])
    
    def B(self) -> np.ndarray:
        return np.array([1])

class PendulumRenderElement(RenderElement):
    def __init__(self, env: RenderEnvironment) -> None:
        super().__init__(env)
        
        self.rod, = self.env.ax.plot([], [], 'o-', lw=2, markersize=5, c='black', markerfacecolor='gray')

    def update(self, t, x, u):
        xx, yy = np.cos(x[0] - np.pi/2), np.sin(x[0] - np.pi/2)
        
        self.rod.set_data([0, xx], [0, yy])  # Update rod


if __name__ == "__main__":
    x0 = np.array([np.pi/2, 0.0])
    u0 = np.array([0.0])
    
    pend = Pendulum(params=PendulumParams(m=1.0, l=1.0))
    xdot = pend.continuous_dynamics(
        x=x0,
        u=u0
    )
    
    print(x0, u0, xdot)
    print(x0.shape, u0.shape, xdot.shape)