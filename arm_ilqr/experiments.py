"""Experiment definitions for every stabilization, fixation, and movement run.

Here, one :class:`Experiment` object contains the parameters that actually
change between conditions, while the model and solver implementations stay
shared. This makes differences between conditions visible and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import ArmControllerConfig, StateLayout
from .solver import QuadraticCost


@dataclass(frozen=True)
class Experiment:
    """Complete configuration for one legacy-compatible experiment.

    Parameters
    ----------
    name:
        Short command-line preset, such as ``"s"`` or ``"m_p"``.
    phase:
        Behavioral phase: stabilization, fixation (preparation), or movement.
    feedback:
        Feedback condition. ``full`` uses all control blocks; ``hidden``,
        ``proprio``, and ``visual`` isolate one block through the R penalties.
    horizon:
        Number of states in a trajectory. The solver therefore optimizes
        ``horizon - 1`` control vectors.
    fixed_step:
        Fraction of the computed iLQR control update applied per iteration.
        These values are kept from the original scripts for reproducibility.
    max_iterations:
        Phase-specific iteration limit. ``None`` intentionally leaves legacy
        movement runs uncapped until their hand-position criterion is met.
    q_* / r_*:
        Diagonal weights for state goals and the three controller-input blocks.
    input_stage / output_stage:
        Subdirectories that connect stabilization -> fixation -> movement.
    """

    name: str
    phase: str
    feedback: str
    horizon: int
    fixed_step: float
    max_iterations: int | None
    model: ArmControllerConfig
    q_controller: float
    q_goal_initial: float
    q_goal_final: float
    r_recurrent: float
    r_proprio: float
    r_visual: float
    input_stage: str | None
    output_stage: str

    @property
    def layout(self) -> StateLayout:
        return StateLayout(self.model.nx)

    def build_cost(self, theta_desired: np.ndarray) -> QuadraticCost:
        """Build the state, control, and terminal cost matrices.

        Q penalizes errors throughout the trajectory, Qf applies at the final
        state, and R controls which recurrent/sensory parameters may change.
        """
        layout = self.layout

        # State cost: muscle activation and controller activity receive small
        # regularizers, while phase-specific goal states receive large weights.
        q = np.zeros((layout.state_dim, layout.state_dim))
        q[layout.activations, layout.activations] = np.eye(6) * 1e-8
        q[layout.controller, layout.controller] = np.eye(self.model.nx) * self.q_controller
        q[layout.goal_initial, layout.goal_initial] = np.eye(2) * self.q_goal_initial
        q[layout.goal_final, layout.goal_final] = np.eye(2) * self.q_goal_final
        qf = np.zeros_like(q)
        qf[layout.goal_initial, layout.goal_initial] = np.eye(2) * self.q_goal_initial
        qf[layout.goal_final, layout.goal_final] = np.eye(2) * self.q_goal_final

        # Control vector layout:
        #   [recurrent Nx*Nx | proprioceptive Nx*12 | visual Nx*2]
        # A weight of 1e21 is the legacy way of effectively freezing a block.
        r = np.zeros((layout.control_dim, layout.control_dim))
        nx = self.model.nx
        r[:nx * nx, :nx * nx] = np.eye(nx * nx) * self.r_recurrent
        p0, p1 = nx * nx, nx * nx + nx * 12
        r[p0:p1, p0:p1] = np.eye(nx * 12) * self.r_proprio
        r[p1:, p1:] = np.eye(nx * 2) * self.r_visual

        # Most desired entries are zero because the augmented goal states
        # already encode deviations. The stored angles document the task and
        # are used by the terminal expression exactly as in the source scripts.
        desired = np.zeros(layout.state_dim)
        desired[:4] = [0.1, 1.57, 0.0, 0.0]
        desired[layout.theta_initial] = [0.1, 1.57]
        desired[layout.theta_final] = theta_desired
        return QuadraticCost(q, r, qf, desired)

    def initial_state(self, theta_desired: np.ndarray, job_id: int, data_root: Path) -> np.ndarray:
        """Create or load the state that begins this phase.

        Stabilization starts from a fixed posture. Fixation consumes the final
        stabilization state, and movement consumes the matching fixation state.
        The tiny 1e-16 values reproduce the original scripts' nonzero padding.
        """
        layout = self.layout
        if self.input_stage is None:
            # First phase: build the complete state without loading a checkpoint.
            state = np.zeros(layout.state_dim)
            state[:4] = [0.1, 1.57, 0.0, 0.0]
            state[layout.theta_initial] = [0.1, 1.57]
            state[layout.theta_final] = theta_desired
            return state + 1e-16

        # Later phases retain the physical/controller/task prefix from the
        # previous phase, then reset all goal-state entries for a new solve.
        if self.phase == "fixation":
            source = data_root / self.input_stage / f"xk1_init_fixed_{job_id}.npy"
        else:
            source = data_root / self.input_stage / f"xk1_init_move_{job_id}.npy"
        checkpoint = np.load(source)
        if checkpoint.shape != (layout.state_dim,):
            raise ValueError(
                f"incompatible checkpoint {source}: expected shape "
                f"({layout.state_dim},) for Nx={self.model.nx}, got {checkpoint.shape}"
            )
        prefix = checkpoint[:15 + self.model.nx]
        return np.concatenate((prefix, np.full(4 + self.model.nx, 1e-16)))


def _preset(name: str, phase: str, feedback: str) -> Experiment:
    """Translate a legacy filename suffix into one explicit configuration."""
    suffix = "" if feedback == "full" else f"_{feedback[0]}"
    if phase == "stabilization":
        # Hold the initial posture and learn a stable controller state.
        return Experiment(name, phase, feedback, 150, 0.1, 200,
            ArmControllerConfig(goal_mode="stabilization", visual_target="initial"),
            1e-3, 3000 * 5, 0.0, 10.0, 10.0, 10.0, None, "stabilization")
    if phase == "fixation":
        # Fixation lasts 75 states and strongly penalizes leaving the initial
        # posture. Feedback-only conditions freeze the other control blocks.
        # 1e21 should be clear as it isolates on block
        penalties = {
            "full": (5.0, 5.0, 5.0),
            "hidden": (5.0, 1e21, 1e21),
            "proprio": (1e21, 5.0, 1e21),
            "visual": (1e21, 1e21, 5.0),
        }[feedback]
        return Experiment(name, phase, feedback, 75, 0.015 / 4, 500,
            ArmControllerConfig(goal_mode="fixation", visual_target="final", go_cue=0.0),
            1e-4, 3000 * 3e5, 3000 * 1e3, *penalties,
            "stabilization", f"fixation{suffix}")
    # Movement begins at the saved fixation state and targets the final angle.
    # The full-feedback source used a slightly wider muscle-length range.
    # 1e21 should be obvious as it isolates one block
    # for full: proprioception is given preference as the central hypothesis is that it leads to orthogonal subspaces.
    # further justified through constrained controllers.
    penalties = {
        "full": (5.0, 5e-2, 5.0),
        "hidden": (5e-2, 1e21, 1e21),
        "proprio": (1e21, 5e-2, 1e21),
        "visual": (1e21, 1e21, 5e-2),
    }[feedback]
    lower = 0.5 if feedback == "full" else 0.7
    return Experiment(name, phase, feedback, 40, 0.0025 / 2, None,
        ArmControllerConfig(
            goal_mode="movement",
            visual_target="final",
            go_cue=1.0,
            muscle_length_min=lower,
            movement_initial_hand_error=feedback == "full",
        ),
        1e-4, 0.0, 3000.0, *penalties,
        f"fixation{suffix}", f"move{suffix}")


# Public lookup used by both the compatibility wrappers and the CLI runner.
PRESETS = {
    "s": _preset("s", "stabilization", "full"),
    "f": _preset("f", "fixation", "full"),
    "f_h": _preset("f_h", "fixation", "hidden"),
    "f_p": _preset("f_p", "fixation", "proprio"),
    "f_v": _preset("f_v", "fixation", "visual"),
    "m": _preset("m", "movement", "full"),
    "m_h": _preset("m_h", "movement", "hidden"),
    "m_p": _preset("m_p", "movement", "proprio"),
    "m_v": _preset("m_v", "movement", "visual"),
}
