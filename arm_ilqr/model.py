"""Differentiable two-joint, six-muscle arm and recurrent controller.

The original nine scripts contained three subtly different copies of this
model.  Those differences are represented explicitly by ``ArmControllerConfig``.
The flat state is retained for numerical compatibility, while ``StateLayout``
gives every block a readable name and prevents fragile negative indexing.

The arm model definition and the corresponding parameters are based on original MATLAB implementation.
See: https://www.nature.com/articles/s41598-021-96084-2
See: https://github.com/yuki-ueyama/Muscle-Activation-Pattern

The implementations remains the same across the all the unconstrained and constrained controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


GoalMode = Literal["stabilization", "fixation", "movement"]
VisualTarget = Literal["initial", "final"]
ReadoutMode = Literal["legacy", "balanced"]
NumericsMode = Literal["legacy_exact", "corrected"]


@dataclass(frozen=True)
class StateLayout:
    """Named slices for the flat optimization state.

    State order::

        plant(4), activations(6), controller(Nx), go cue(1),
        initial angles(2), final angles(2), initial goal(2),
        final goal(2), controller-stationarity goal(Nx)

    Thus ``state_dim = 19 + 2*Nx`` and, for the default ``Nx=22``, the state
    has 63 entries. The control vector has ``Nx*Nx + Nx*12 + Nx*2`` entries.
    """

    nx: int

    @property
    def plant(self) -> slice:
        return slice(0, 4)

    @property
    def activations(self) -> slice:
        return slice(4, 10)

    @property
    def controller(self) -> slice:
        return slice(10, 10 + self.nx)

    @property
    def go_cue(self) -> slice:
        return slice(10 + self.nx, 11 + self.nx)

    @property
    def theta_initial(self) -> slice:
        return slice(11 + self.nx, 13 + self.nx)

    @property
    def theta_final(self) -> slice:
        return slice(13 + self.nx, 15 + self.nx)

    @property
    def goal_initial(self) -> slice:
        return slice(15 + self.nx, 17 + self.nx)

    @property
    def goal_final(self) -> slice:
        return slice(17 + self.nx, 19 + self.nx)

    @property
    def goal_stationary(self) -> slice:
        return slice(19 + self.nx, 19 + 2 * self.nx)

    @property
    def state_dim(self) -> int:
        return 19 + 2 * self.nx

    @property
    def control_dim(self) -> int:
        return self.nx * (self.nx + 14)


@dataclass(frozen=True)
class ArmControllerConfig:
    """Parameters that change model behavior across experimental phases.

    ``nx`` is the recurrent controller width and ``dt`` is the simulation step
    in seconds. ``goal_mode`` chooses the augmented goal equations appropriate
    for stabilization, fixation, or movement. ``visual_target`` determines
    whether visual error points to the initial or final posture.

    ``numerics="legacy_exact"`` preserves CPU float32 operations and historical
    gradient behavior. ``"corrected"`` enables safer float64/device-aware
    operations. ``readout_mode="legacy"`` preserves the original controller-to-
    muscle mapping; ``"balanced"`` uses every nominally decoupled unit.
    """

    nx: int = 22
    dt: float = 0.005
    goal_mode: GoalMode = "fixation"
    visual_target: VisualTarget = "final"
    go_cue: float = 0.0
    muscle_length_min: float = 0.7
    movement_initial_hand_error: bool = False
    controller_floor: float = 1e-8
    readout_mode: ReadoutMode = "legacy"
    numerics: NumericsMode = "legacy_exact"

    def __post_init__(self) -> None:
        if self.nx < 2 or self.nx % 2:
            raise ValueError("nx must be an even integer of at least 2")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if not 0 < self.muscle_length_min < 1.5:
            raise ValueError("muscle_length_min must lie in (0, 1.5)")


class ArmControllerModel(nn.Module):
    """Advance the physical arm, muscles, and controller by one time step."""

    def __init__(self, config: ArmControllerConfig):
        super().__init__()
        self.config = config
        self.layout = StateLayout(config.nx)

        # These are fixed biomechanical constants, registered as buffers so
        # corrected mode can move them with the model while legacy mode remains
        # CPU float32-compatible.
        dtype = torch.float32 if config.numerics == "legacy_exact" else torch.float64
        self.register_buffer("moment_arm", torch.tensor(
            [[1.5, -1.5, 0.0, 0.0, 1.5, -1.5],
             [0.0, 0.0, 1.5, -1.5, 1.5, -1.5]],
            dtype=dtype,
        ) / 100.0)
        self.register_buffer("pcsa", torch.full((6, 1), 10.0, dtype=dtype))
        self.register_buffer("optimal_length", torch.full((6, 1), 0.08, dtype=dtype))
        self.register_buffer("equilibrium_angle", torch.tensor(
            [[15.0, 5.0, 0.0, 0.0, 15.0, 5.0],
             [0.0, 0.0, 90.0, 110.0, 100.0, 100.0]],
            dtype=dtype,
        ) * torch.pi / 180.0)
        self.register_buffer("readout", self._make_readout(config))

    @staticmethod
    def _make_readout(config: ArmControllerConfig) -> Tensor:
        """Map controller activity to six muscle-excitation commands.

        Half the controller is muscle-specific and half projects weakly to all
        muscles. With Nx=22, legacy integer division assigns only 6 of the 11
        specific units directly; this historical behavior is preserved unless
        ``readout_mode="balanced"`` is requested.
        """
        split = config.nx // 2
        coupled = config.nx - split
        dtype = torch.float32 if config.numerics == "legacy_exact" else torch.float64
        matrix = torch.zeros((6, config.nx), dtype=dtype)
        if config.readout_mode == "legacy":
            width = split // 6
            for muscle in range(6):
                matrix[muscle, muscle * width:(muscle + 1) * width] = 1.0
        else:
            # Use every decoupled unit even when nx/2 is not divisible by six.
            for muscle, indices in enumerate(torch.tensor_split(torch.arange(split), 6)):
                matrix[muscle, indices] = 1.0
        if coupled:
            matrix[:, -coupled:] = 0.1
        return matrix

    def _constants_like(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return fixed model tensors on the input state's dtype and device."""
        return tuple(
            value.to(device=x.device, dtype=x.dtype)
            for value in (
                self.moment_arm,
                self.pcsa,
                self.optimal_length,
                self.equilibrium_angle,
                self.readout,
            )
        )

    @staticmethod
    def _hand_position(theta: Tensor) -> Tensor:
        """Convert shoulder/elbow angles into planar hand x/y position."""
        l1, l2 = 0.15, 0.21
        return torch.stack((
            l1 * torch.cos(theta[0]) + l2 * torch.cos(theta[0] + theta[1]),
            l1 * torch.sin(theta[0]) + l2 * torch.sin(theta[0] + theta[1]),
        ))

    @staticmethod
    def _scaled_error(error: Tensor) -> Tensor:
        """Apply the steep, capped error scaling used by the source scripts."""
        scale = torch.pow(error.new_tensor(1e15), torch.abs(error)).clamp(max=2.0)
        return scale * error

    def _goal_states(
        self,
        theta: Tensor,
        theta_initial: Tensor,
        theta_final: Tensor,
        dx_controller: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Construct augmented states that let quadratic costs express goals."""
        mode = self.config.goal_mode
        if mode == "stabilization":
            # Stabilization uses direct joint-angle errors.
            goal_initial = theta - theta_initial
            goal_final = theta - theta_final
        elif mode == "fixation":
            # Fixation strongly magnifies departures from the held posture.
            goal_initial = self._scaled_error(theta - theta_initial)
            goal_final = self._scaled_error(theta - theta_final)
        else:
            # Movement combines joint-space and endpoint-space errors. The full
            # condition included hand error in goal_initial; ablations did not.
            hand_error = self._hand_position(theta_final) - self._hand_position(theta)
            if self.config.movement_initial_hand_error:  # legacy full-feedback movement
                initial_error = theta - theta_initial + hand_error
            else:
                initial_error = theta - theta_initial
            final_error = 0.5 * (theta - theta_final) + 0.5 * hand_error
            goal_initial = self._scaled_error(initial_error)
            goal_final = self._scaled_error(final_error)
            goal_final = goal_final + 10.0 / torch.pow(10.0, 5.0 * torch.abs(final_error)) * final_error
        return goal_initial, goal_final, torch.abs(dx_controller) - 1.0

    def forward(self, x: Tensor, u: Tensor, dt: float | None = None) -> Tensor:
        """Return the next flat state for current state ``x`` and parameters ``u``."""
        layout = self.layout
        if x.ndim != 1 or x.numel() != layout.state_dim:
            raise ValueError(f"x must have shape ({layout.state_dim},), got {tuple(x.shape)}")
        if u.ndim != 1 or u.numel() != layout.control_dim:
            raise ValueError(f"u must have shape ({layout.control_dim},), got {tuple(u.shape)}")
        legacy = self.config.numerics == "legacy_exact"
        if legacy and (x.device.type != "cpu" or x.dtype != torch.float32):
            raise ValueError("legacy_exact dynamics require CPU float32 tensors")

        # ------------------------------------------------------------------
        # 1. Muscle geometry and force generation
        # ------------------------------------------------------------------
        step = self.config.dt if dt is None else dt
        j, pcsa, l0, eq_theta, readout = self._constants_like(x)
        theta, velocity = x[:2], x[2:4]
        activations = x[layout.activations].reshape(6, 1)

        muscle_length = (
            j[0] * (eq_theta[0] - theta[0])
            + j[1] * (eq_theta[1] - theta[1])
        ).reshape(6, 1)
        muscle_velocity = (-j.T @ velocity.reshape(2, 1))
        muscle_length = (1.0 + muscle_length / l0).clamp(self.config.muscle_length_min, 1.5)
        muscle_velocity = (muscle_velocity / l0).clamp(-2.0, 2.0)

        nf = 2.11 + 3.31 * (1.0 / muscle_length - 1.0)
        af_base = activations / (0.56 * nf)
        if legacy:
            af_base = 0.5 * (torch.abs(af_base) + af_base) + 1e-8
        else:
            af_base = torch.relu(af_base) + 1e-8
        active_fraction = 1.0 - torch.exp(-(af_base ** nf))
        force_length = torch.exp(-torch.abs((muscle_length ** 1.55 - 1.0) / 0.81) ** 2.12)

        eccentric = (-7.39 - muscle_velocity) / (
            -7.39 + (-3.21 + 4.17 * muscle_length) * muscle_velocity
        )
        concentric = (1.05 - (-1.53) * muscle_velocity) / (1.05 + muscle_velocity)
        if legacy:
            force_velocity = concentric.clone()
            force_velocity[muscle_velocity < 0] = eccentric[muscle_velocity < 0]
            passive1 = 25.6 * 0.059 * torch.log(
                torch.exp((muscle_length - 1.54) / 0.059) + 1
            )
        else:
            force_velocity = torch.where(muscle_velocity < 0, eccentric, concentric)
            passive1 = 25.6 * 0.059 * torch.logaddexp(
                (muscle_length - 1.54) / 0.059,
                torch.zeros_like(muscle_length),
            )
        passive2 = -0.020 * (torch.exp(-18.7 * (muscle_length - 0.79)) - 1.0)
        tension = 32.0 * pcsa * (active_fraction * (force_length * force_velocity + passive2) + passive1)
        torque = j @ tension

        # ------------------------------------------------------------------
        # 2. Two-link rigid-body dynamics
        # ------------------------------------------------------------------
        gravity_terms = (
            -0.5 * 0.3 * 9.8 * 0.15 * torch.cos(theta[0]),
            -0.5 * 0.3 * 9.8 * 0.21 * torch.cos(theta[0] + theta[1]),
        )
        if legacy:
            # Preserve the detached gravity Jacobian in the original scripts.
            gravity = torch.FloatTensor(gravity_terms).reshape(2, 1)
        else:
            gravity = torch.stack(gravity_terms).reshape(2, 1)
        torque = torque + gravity

        a1 = 5e-3 + 9e-3 + 0.3 * 0.15 ** 2
        a2 = 0.3 * 0.15 * 0.12
        a3 = 9e-3
        inertia = torch.stack((
            a1 + 2 * a2 * torch.cos(theta[1]),
            a3 + a2 * torch.cos(theta[1]),
            a3 + a2 * torch.cos(theta[1]),
            theta.new_tensor(a3),
        )).reshape(2, 2)
        coriolis = torch.stack((
            -velocity[1] * (2 * velocity[0] + velocity[1]) * a2 * torch.sin(theta[1]),
            velocity[0] ** 2 * a2 * torch.sin(theta[1]),
        )).reshape(2, 1)
        damping = theta.new_tensor([[5e-3, 2.5e-3], [2.5e-3, 5e-3]])
        rhs = torque - coriolis - damping @ velocity.reshape(2, 1)
        acceleration = torch.inverse(inertia) @ rhs if legacy else torch.linalg.solve(inertia, rhs)
        plant_rate = torch.cat((velocity, acceleration.flatten()))
        next_plant = x[:4] + step * 1000.0 * torch.tanh(plant_rate / 1000.0)

        # ------------------------------------------------------------------
        # 3. Activation dynamics and controller-to-muscle readout
        # ------------------------------------------------------------------
        controller = x[layout.controller].reshape(self.config.nx, 1)
        muscle_input = (readout @ controller).clamp(0.0, 1.0)
        time_constant = 0.066 + muscle_input * (0.050 - 0.066) * (muscle_input > activations)
        next_activations = (activations + step * (muscle_input - activations) / time_constant).clamp(0.0, 1.0)

        # ------------------------------------------------------------------
        # 4. Recurrent controller update from proprioception and vision
        # ------------------------------------------------------------------
        target = x[layout.theta_initial] if self.config.visual_target == "initial" else x[layout.theta_final]
        visual_error = self._hand_position(target) - self._hand_position(theta)
        proprioception = torch.cat((muscle_length, muscle_velocity), dim=0)
        nx = self.config.nx
        recurrent = u[:nx * nx].reshape(nx, nx)
        proprio_gain = u[nx * nx:nx * nx + nx * 12].reshape(nx, 12)
        visual_gain = u[nx * nx + nx * 12:].reshape(nx, 2)
        controller_rate = (
            proprio_gain @ proprioception
            + recurrent @ controller
            + visual_gain @ visual_error.reshape(2, 1)
        ).flatten()
        next_controller = x[layout.controller] + step * controller_rate
        if legacy:
            next_controller = 0.5 * (torch.abs(next_controller) + next_controller) + self.config.controller_floor
        else:
            next_controller = torch.relu(next_controller) + self.config.controller_floor

        # ------------------------------------------------------------------
        # 5. Carry task variables forward and refresh augmented goal states
        # ------------------------------------------------------------------
        theta_initial = x[layout.theta_initial]
        theta_final = x[layout.theta_final]
        goal_initial, goal_final, goal_stationary = self._goal_states(
            theta, theta_initial, theta_final, controller_rate
        )
        return torch.cat((
            next_plant,
            next_activations.flatten(),
            next_controller,
            x.new_tensor([self.config.go_cue]),
            theta_initial,
            theta_final,
            goal_initial,
            goal_final,
            goal_stationary,
        ))
