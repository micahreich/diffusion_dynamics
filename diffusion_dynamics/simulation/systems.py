import numpy as np
import torch
from typing import Optional, Any, Tuple
from dataclasses import dataclass
from scipy.linalg import solve_continuous_are
from diffusion_dynamics.utils import torch_to_numpy
from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
import matplotlib.pyplot as plt


class DynamicalSystem:
    def __init__(self, nx: int, nu: int, name: Optional[str] = None, params: Optional[Any] = None) -> None:
        self.nx = nx
        self.nu = nu
        self.name = name
        self.params = params
        
        self.t_history = self.x_history = self.u_history = None
    
    def batch_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.batch_dynamics(x.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
    
    def clear_history(self) -> None:
        self.t_history = self.x_history = self.u_history = None

    def set_history(self, ts, xs, us) -> None:
        self.t_history = ts
        self.x_history = xs
        self.u_history = us

    def get_history(self) -> Any:
        return (
            torch.as_tensor(self.t_history),
            torch.as_tensor(self.x_history),
            torch.as_tensor(self.u_history),
        )

    def query_history(self, t: float) -> Tuple:
        assert self.t_history is not None, "No time history to query"

        # Clamp t to the range of the time history
        t = torch.maximum(torch.tensor(0), torch.minimum(self.t_history[-1], torch.tensor(t)))
        idx_hi = torch.searchsorted(self.t_history, t)
        idx_lo = torch.maximum(torch.tensor(0), idx_hi - 1)

        # Look up indices based on time to do linear interpolation of states, controls
        t_lo, t_hi = self.t_history[idx_lo], self.t_history[idx_hi]

        if idx_lo == idx_hi:
            alpha_lo, alpha_hi = 0.0, 1.0
        else:
            alpha_hi = (t - t_lo) / (t_hi - t_lo)
            alpha_lo = 1.0 - alpha_hi

        x_interp = alpha_lo * self.x_history[idx_lo] + alpha_hi * self.x_history[idx_hi]

        if idx_hi >= len(self.u_history):
            idx_lo = idx_hi = len(self.u_history) - 1
            alpha_hi, alpha_lo = 1.0, 0.0

        u_interp = alpha_lo * self.u_history[idx_lo] + alpha_hi * self.u_history[idx_hi]

        return torch.as_tensor(x_interp), torch.as_tensor(u_interp)

class CartPole(DynamicalSystem):
    @dataclass
    class Params:
        m_c: float
        m_p: float
        l: float
        g: float
    
    class PlotElement(PlotElement):
        def __init__(self, env: PlotEnvironment, sys: "CartPole") -> None:
            super().__init__(env)

            self.sys = sys
            
            self.cart_width, self.cart_height = 0.4, 0.2
            
            self.cart = self.env.ax.add_patch(plt.Rectangle((-self.cart_width/2, -self.cart_height/2), self.cart_width, self.cart_height, fc="blue"))  # Cart
            (self.rod,) = self.env.ax.plot([], [], 'o-', lw=2, markersize=5, c='black', markerfacecolor='gray')  # Pole
            
            x_lo, x_hi = sys.x_history[:, 0].min(), sys.x_history[:, 0].max()
            new_range = (x_hi - x_lo + self.cart_width) * 1.2
            
            x_lo = (x_lo + x_hi) / 2 - new_range / 2
            x_hi = (x_lo + x_hi) / 2 + new_range / 2
            
            self.env.ax.set_xlim(x_lo, x_hi)
            self.env.ax.set_ylim(-sys.params.l * 1.2, sys.params.l * 1.2)

        def update(self, t):
            state, _ = self.sys.query_history(t)  # Get the current state of the cartpole
            x, theta = state[0], state[1]  # Extract cart position and pole angle
            
            # Compute the pole's end position
            l = self.sys.params.l
            pole_x = x + l * np.sin(theta)
            pole_y = -l * np.cos(theta)

            # Update cart position
            self.cart.set_xy((x - 0.2, -0.1))  # Adjusted for cart size
            
            # Update pole position
            self.rod.set_data([x, pole_x], [0, pole_y])
    
    def __init__(self, params: Params) -> None:
        super().__init__(4, 1, "CartPole", params)
    
    def batch_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        N1, nx = x.shape
        N2, nu = u.shape
        
        assert nx == self.nx
        assert nu == self.nu
        assert N1 == N2
        
        _, theta, v, theta_dot = x.T.unsqueeze(-1)
                
        x_ddot = 1/(self.params.m_c + self.params.m_p * torch.sin(theta)**2) * (
            u + self.params.m_p * torch.sin(theta) * (self.params.l * theta_dot**2 + self.params.g * torch.cos(theta))
        )
        
        theta_ddot = 1/(self.params.l * (self.params.m_c + self.params.m_p * torch.sin(theta)**2)) * (
            -u * torch.cos(theta) - self.params.m_p * self.params.l * theta_dot**2 * torch.cos(theta) * torch.sin(theta) - (self.params.m_c + self.params.m_p) * self.params.g * torch.sin(theta)
        )
        
        return torch.column_stack([v, theta_dot, x_ddot, theta_ddot])

if __name__ == "__main__":
    cart_pole = CartPole(params=CartPole.Params(1, 1, 1, 9.81))
    
    xbar = torch.tensor([0,torch.pi - 0.4,0,0], dtype=torch.float32)
    ubar = torch.tensor([24.24184], dtype=torch.float32)
    
    print(cart_pole.dynamics(xbar, ubar))
    
    xbar = torch.tile(xbar, dims=(5, 1))
    ubar = torch.tile(ubar, dims=(5, 1))
    
    print(cart_pole.batch_dynamics(xbar, ubar))
    
    # ubar = torch.tensor([0.], dtype=torch.float32)
    # ubar = torch.tile(ubar, dims=(100, 1))
    
    # print(cart_pole.batch_dynamics(xbar, ubar))
    # print(cart_pole.dynamics(xbar, ubar))
    
    # A = torch_to_numpy(torch.autograd.functional.jacobian(lambda _x : cart_pole.dynamics(_x, ubar), xbar))
    # B = torch_to_numpy(torch.autograd.functional.jacobian(lambda _u : cart_pole.dynamics(xbar, _u), ubar))
    # Q = np.eye(4)
    # R = np.eye(1)
    # P = solve_continuous_are(A, B, Q, R)
    # K = np.linalg.inv(R) @ B.T @ P
    
    # print(A.shape, B.shape, K.shape)