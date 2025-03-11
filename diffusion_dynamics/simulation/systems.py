import numpy as np

# import jax.numpy as np
import jax
from typing import Any, Tuple
from scipy.linalg import block_diag
from dataclasses import dataclass
from diffusion_dynamics.utils import unpack_state
from diffusion_dynamics.simulation.animation import RenderEnvironment, RenderElement
import torch

# Base dynamical system class


class DynamicalSystem:
    def __init__(self, name: str, params: Any, nx: int, nu: int) -> None:
        self.name = name
        self.params = params
        self.nx = nx
        self.nxx = self.nx // 2
        self.nu = nu

        self.t_history = self.x_history = self.u_history = None

    def clear_history(self) -> None:
        self.t_history = self.x_history = self.u_history = None

    def set_history(self, ts, xs, us) -> None:
        self.t_history = ts
        self.x_history = xs
        self.u_history = us

    def get_history(self, dtype="torch") -> Any:
        assert dtype in ["torch", "numpy"], "Type must be 'torch' or 'numpy'"

        if dtype == "torch":
            return (
                torch.tensor(self.t_history),
                torch.tensor(self.x_history),
                torch.tensor(self.u_history),
            )
        else:
            return (
                np.array(self.t_history),
                np.array(self.x_history),
                np.array(self.u_history),
            )

    def query_history(self, t: float, dtype="torch") -> Tuple:
        assert dtype in ["torch", "numpy"], "Type must be 'torch' or 'numpy'"
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

        if dtype == "torch":
            return torch.tensor(x_interp), torch.tensor(u_interp)
        else:
            return np.array(x_interp), np.array(u_interp)

    def M(self, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def C(self, q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def G(self, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def B(self) -> torch.Tensor:
        raise NotImplementedError

    def continuous_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 1
        assert u.ndim == 1

        out = self.batch_continuous_dynamics(x.unsqueeze(0), u.unsqueeze(0))
        return out.view(-1)
    
    def batch_continuous_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        assert u.ndim == 2
        
        Bx, nx = x.shape
        Bu, nu = u.shape
        
        assert Bx == Bu
        assert nx == self.nx
        assert nu == self.nu
        
        q = x[:, :self.nxx]
        qdot = x[:, self.nxx:]
        
        M = self.M(q)
        C = self.C(q, qdot)
        G = self.G(q)
        B = self.B()
        batched_B = B.unsqueeze(0).expand(Bx, -1, -1)

        assert M.shape == (Bx, self.nxx, self.nxx)
        assert C.shape == (Bx, self.nxx, self.nxx)
        assert G.shape == (Bx, self.nxx)
        assert B.shape == (self.nxx, self.nu)
        assert batched_B.shape == (Bx, self.nxx, self.nu)
        
        qddot = torch.linalg.solve(
            M,
            (batched_B @ u.unsqueeze(-1) - C @ qdot.unsqueeze(-1) + G.unsqueeze(-1))
        ).squeeze(-1)
        
        return torch.hstack([qdot, qddot])

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

    def M(self, q: torch.Tensor) -> torch.Tensor:
        N, _ = q.shape
        
        m, l = self.params.m, self.params.l
        return torch.Tensor([m * l**2]).view(self.nxx, self.nxx).repeat(N, 1, 1)

    def C(self, q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
        N, _ = q.shape
        
        b = self.params.b
        return torch.Tensor([b]).view(self.nxx, self.nxx).repeat(N, 1, 1)

    def G(self, q: torch.Tensor) -> torch.Tensor:
        N, _ = q.shape
        
        m, l, g = self.params.m, self.params.l, self.g
        return -(m * g * l * torch.sin(q[:, 0])).view(N, self.nxx)

    def B(self) -> torch.Tensor:
        return torch.Tensor([1.0]).view(self.nxx, self.nu)


class PendulumRenderElement(RenderElement):
    def __init__(self, env: RenderEnvironment, sys: Pendulum) -> None:
        super().__init__(env)

        self.sys = sys
        (self.rod,) = self.env.ax.plot([], [], 'o-', lw=2, markersize=5, c='black', markerfacecolor='gray')

    def update(self, t):
        x, _ = self.sys.query_history(t, dtype='numpy')

        xx, yy = np.cos(x[0] - np.pi / 2), np.sin(x[0] - np.pi / 2)

        self.rod.set_data([0, xx], [0, yy])  # Update rod


if __name__ == "__main__":
    # Bx, nx, nu = 10, 2, 1
    # nxx = nx//2  
    
    # M = torch.randn(Bx, nxx, nxx)
    # C = torch.randn(Bx, nxx, nxx)
    # G = torch.randn(Bx, nxx)
    # B = torch.randn(Bx, nxx, nu)
    
    # q = torch.randn(Bx, nxx)
    # qdot = torch.randn(Bx, nxx)
    # u = torch.randn(Bx, nu)
    
    # qddot = torch.linalg.solve(
    #     M,
    #     (B @ u.unsqueeze(-1) - C @ qdot.unsqueeze(-1) + G.unsqueeze(-1))
    # ).squeeze(-1)
    
    # xdot = torch.hstack([qdot, qddot])
    
    # print("qddot", qddot.shape)
    # print("xdot", xdot.shape)
    
    batch_size = 10_000
    x0 = torch.Tensor([torch.pi, 0.0]).view(1, -1).repeat(batch_size, 1)
    u0 = torch.Tensor([0.0]).view(1, -1).repeat(batch_size, 1)

    pend = Pendulum(params=PendulumParams(m=1.0, l=1.0, b=0.5))
    xdot = pend.batch_continuous_dynamics(x=x0, u=u0)

    print(xdot[:4])
    print(x0.shape, u0.shape, xdot.shape)

    # df_dx, df_du = torch.autograd.functional.jacobian(
    #     pend.continuous_dynamics,
    #     (x0, u0),
    # )

    # print(f"df_dx ({df_dx.shape}): {df_dx}")
    # print(f"df_du ({df_du.shape}): {df_du}")
