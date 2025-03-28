from dataclasses import dataclass
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_continuous_are

from diffusion_dynamics.simulation.animation import PlotElement, PlotEnvironment
from diffusion_dynamics.utils import torch_to_numpy


def angle_wrap(theta, mode='0:2pi'):
    if mode == '0:2pi':
        return torch.remainder(theta, 2 * np.pi)
    elif mode == '-pi:pi':
        return torch.remainder(theta + np.pi, 2 * np.pi) - np.pi


class DynamicalSystem:
    nx = None
    nu = None

    def __init__(self, name: Optional[str] = None, params: Optional[Any] = None) -> None:
        self.name = name
        self.params = params

        self.t_history = self.x_history = self.u_history = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Check if the subclass has its own definition of 'a'
        if cls.nx is DynamicalSystem.nx or cls.nu is DynamicalSystem.nu:
            raise NotImplementedError(f"Class variable 'nx' and 'nu' must be overridden in {cls.__name__}")

    def project_state(self, x: torch.Tensor) -> torch.Tensor:
        return x

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

    nx = 4
    nu = 1

    def __init__(self, params: Params) -> None:
        super().__init__("CartPole", params)

    def project_state(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            theta = x[:, 1]
            x[:, 1] = angle_wrap(theta, mode='-pi:pi')
        else:
            theta = x[1]
            x[1] = angle_wrap(theta, mode='-pi:pi')

        return x

    def batch_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1 and len(u.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        N1, nx = x.shape
        N2, nu = u.shape

        assert nx == self.nx
        assert nu == self.nu
        assert N1 == N2

        _, theta, v, theta_dot = x.T.unsqueeze(-1)

        x_ddot = 1 / (self.params.m_c + self.params.m_p * torch.sin(theta) ** 2) * (
            u + self.params.m_p * torch.sin(theta) *
            (self.params.l * theta_dot ** 2 + self.params.g * torch.cos(theta)))

        theta_ddot = 1 / (self.params.l * (self.params.m_c + self.params.m_p * torch.sin(theta) ** 2)) * (
            -u * torch.cos(theta) -
            self.params.m_p * self.params.l * theta_dot ** 2 * torch.cos(theta) * torch.sin(theta) -
            (self.params.m_c + self.params.m_p) * self.params.g * torch.sin(theta))

        y_dot = torch.column_stack([v, theta_dot, x_ddot, theta_ddot])

        return torch.squeeze(y_dot)

    class PlotElement(PlotElement):
        def __init__(self, env: PlotEnvironment, sys: "CartPole", cart_color='blue') -> None:
            super().__init__(env)

            self.sys = sys

            self.cart_width, self.cart_height = 0.4, 0.2

            self.cart = self.env.ax.add_patch(
                plt.Rectangle(
                    (-self.cart_width / 2, -self.cart_height / 2),
                    self.cart_width,
                    self.cart_height,
                    fc=cart_color,  # face color
                    ec='black',  # edge color
                    lw=1  # line width
                ))

            (self.rod, ) = self.env.ax.plot([], [], 'o-', lw=2, markersize=5, c='black', markerfacecolor='gray')  # Pole

            x_lo, x_hi = sys.x_history[:, 0].min(), sys.x_history[:, 0].max()
            new_range = (x_hi - x_lo + self.cart_width) * 1.2

            x_lo = (x_lo + x_hi) / 2 - new_range / 2
            x_hi = (x_lo + x_hi) / 2 + new_range / 2

            self.env.set_xlim(x_lo, x_hi)
            self.env.set_ylim(-sys.params.l * 1.2, sys.params.l * 1.2)

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


class PlanarQuadrotor(DynamicalSystem):
    @dataclass
    class Params:
        m: float
        I: float
        r: float
        g: float

    nx = 6
    nu = 2

    def __init__(self, params: Params) -> None:
        super().__init__("PlanarQuadrotor", params)

    def project_state(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            theta = x[:, 2]
            x[:, 2] = angle_wrap(theta, mode='-pi:pi')
        else:
            theta = x[2]
            x[2] = angle_wrap(theta, mode='-pi:pi')

        return x

    def batch_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1 and len(u.shape) == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        N1, nx = x.shape
        N2, nu = u.shape

        assert nx == self.nx
        assert nu == self.nu
        assert N1 == N2

        rx, ry, theta, x_dot, y_dot, theta_dot = x.T.unsqueeze(-1)
        u1, u2 = u.T.unsqueeze(-1)

        x_ddot = 1 / self.params.m * -((u1 + u2) * torch.sin(theta))
        y_ddot = 1 / self.params.m * ((u1 + u2) * torch.cos(theta) - self.params.g)
        theta_ddot = 1 / self.params.I * self.params.r * (u1 - u2)

        y_dot = torch.column_stack([x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot])

        return torch.squeeze(y_dot)

    class PlotElement(PlotElement):
        def __init__(self,
                     env: PlotEnvironment,
                     sys: "PlanarQuadrotor",
                     body_color='gray',
                     thrust_color='red',
                     arrow_scale=0.001,
                     alpha=1.0) -> None:
            super().__init__(env)
            self.sys = sys
            self.arrow_scale = arrow_scale  # Factor to scale control inputs for display

            # Body dimensions and drawing
            self.body_width = 0.6
            self.body_height = 0.2
            self.body_patch = self.env.ax.add_patch(
                plt.Rectangle(
                    (-self.body_width / 2, -self.body_height / 2),
                    self.body_width,
                    self.body_height,
                    angle=0.0,
                    alpha=alpha,
                    fc=body_color,  # face color
                    ec='black',  # edge color
                    lw=1  # line width
                ))

            # Initial rotor positions in the body frame (assumed to be at ±r along x-axis)
            rotor_left = np.array([-self.body_width / 2, 0])
            rotor_right = np.array([self.body_width / 2, 0])
            initial_positions = np.array([rotor_left, rotor_right])
            # Initially, no thrust is applied so arrows have zero length
            U_init = np.array([0, 0])
            V_init = np.array([0, 0])
            # Create a quiver plot for the thrust arrows.
            # Note: set angles='xy', scale_units='xy', and scale=1 so that the arrow lengths
            # correspond directly to the vector components.
            self.thrust_quiver = self.env.ax.quiver(initial_positions[0, :],
                                                    initial_positions[1, :],
                                                    U_init,
                                                    V_init,
                                                    angles='xy',
                                                    scale_units='xy',
                                                    scale=1,
                                                    color=thrust_color)

            # Optionally, set plot limits based on state history or a default
            rx_vals = sys.x_history[:, 0]
            ry_vals = sys.x_history[:, 1]
            margin = 1.0

            self.env.set_xlim(rx_vals.min() - margin, rx_vals.max() + margin)
            self.env.set_ylim(ry_vals.min() - margin, ry_vals.max() + margin)

            self.max_thrust = torch.max(torch.abs(sys.u_history))

        def update(self, t):
            # Get current state and control input from the system's history
            state, control = self.sys.query_history(t)
            rx, ry, theta = state[0], state[1], state[2]
            u1, u2 = control[0], control[1]

            # # Build rotation matrix for current orientation
            c, s = torch.cos(theta), torch.sin(theta)
            R = torch.tensor([[c, -s], [s, c]])

            thrust_origins_body = torch.tensor([[self.body_width / 2 - 0.1, self.body_height / 2],
                                                [-self.body_width / 2 + 0.1, self.body_height / 2]])
            thrust_origins_world = thrust_origins_body @ R.T + torch.tensor([rx, ry])

            thrust_directions_body = torch.tensor([[0, u1 / self.max_thrust], [0, u2 / self.max_thrust]])
            thrust_directions_world = thrust_directions_body @ R.T

            self.body_patch.set_xy([rx - self.body_width / 2, ry - self.body_height / 2])
            self.body_patch.set_angle(torch.rad2deg(theta))

            self.thrust_quiver.set_offsets(thrust_origins_world)
            self.thrust_quiver.set_UVC(thrust_directions_world[:, 0], thrust_directions_world[:, 1])


if __name__ == "__main__":
    cart_pole = CartPole(params=CartPole.Params(1, 1, 1, 9.81))

    xbar = torch.tensor([0, torch.pi - 0.4, 0, 0], dtype=torch.float32)
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
