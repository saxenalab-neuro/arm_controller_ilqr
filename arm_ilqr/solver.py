"""Reusable finite-horizon iterative LQR solver.

Two numerical paths intentionally coexist. ``legacy_exact`` mirrors the source
scripts for reproducibility. ``corrected`` uses a conventional iLQR update with
regularization, stable linear solves, convergence checks, and optional line
search. Keeping them separate prevents silent changes to historical results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
import itertools

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.autograd.functional import jacobian

Array = NDArray[np.float64]


@dataclass(frozen=True)
class QuadraticCost:
    """Quadratic objective shared by both solver modes.

    ``Q`` penalizes every nonterminal state, ``R`` penalizes controller
    parameters, and ``Qf`` penalizes the final state. ``x_desired`` is the
    reference state used when reporting and optimizing the corrected objective.
    """

    Q: Array
    R: Array
    Qf: Array
    x_desired: Array

    def total(self, states: Array, controls: Array) -> float:
        """Evaluate the complete finite-horizon objective for diagnostics."""
        running_error = states[:-1] - self.x_desired
        final_error = states[-1] - self.x_desired
        state_cost = np.einsum("ti,ij,tj->", running_error, self.Q, running_error)
        control_cost = np.einsum("ti,ij,tj->", controls, self.R, controls)
        final_cost = final_error @ self.Qf @ final_error
        return 0.5 * float(state_cost + control_cost + final_cost)


@dataclass(frozen=True)
class ILQRConfig:
    """Numerical and stopping choices for an iLQR run.

    ``fixed_step`` applies a predetermined fraction of each update. When it is
    ``None`` in corrected mode, ``line_search_steps`` are tried from largest to
    smallest. ``regularization`` stabilizes the corrected control Hessian;
    legacy mode intentionally ignores it and preserves explicit inversions.
    """

    max_iterations: int | None = 500
    tolerance: float = 1e-6
    regularization: float = 1e-8
    fixed_step: float | None = None
    line_search_steps: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)
    numerics: Literal["legacy_exact", "corrected"] = "legacy_exact"


@dataclass
class ILQRResult:
    """Trajectories and progress metrics returned at a stop or checkpoint."""

    states: Array
    controls: Array
    costs: list[float]
    update_norms: list[float]
    iterations: int
    converged: bool


class ILQRSolver:
    """Optimize a time-varying control sequence for differentiable dynamics."""

    def __init__(
        self,
        dynamics: Callable[[Tensor, Tensor], Tensor],
        cost: QuadraticCost,
        config: ILQRConfig | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.dynamics = dynamics
        self.cost = cost
        self.config = config or ILQRConfig()
        self.dtype = dtype
        self.device = torch.device(device)

    def rollout(self, x0: Array, controls: Array) -> Array:
        """Simulate all controls from ``x0`` and return ``T+1`` states."""
        states = [np.asarray(x0, dtype=np.float64)]
        with torch.no_grad():
            x = torch.as_tensor(x0, dtype=self.dtype, device=self.device)
            for control in controls:
                u = torch.as_tensor(control, dtype=self.dtype, device=self.device)
                x = self.dynamics(x, u)
                if not torch.isfinite(x).all():
                    raise FloatingPointError("non-finite state encountered during rollout")
                states.append(x.detach().cpu().numpy())
        return np.stack(states)

    def linearize(self, states: Array, controls: Array) -> tuple[Array, Array]:
        """Compute local state and control Jacobians A[t] and B[t].

        Legacy mode uses the original non-vectorized autograd call. Corrected
        mode vectorizes Jacobian evaluation where PyTorch supports it.
        """
        a_values, b_values = [], []
        for state, control in zip(states[:-1], controls):
            x = torch.as_tensor(state, dtype=self.dtype, device=self.device)
            u = torch.as_tensor(control, dtype=self.dtype, device=self.device)
            a, b = jacobian(
                self.dynamics,
                (x, u),
                vectorize=self.config.numerics != "legacy_exact",
            )
            a_values.append(a.detach().cpu().numpy())
            b_values.append(b.detach().cpu().numpy())
        return np.stack(a_values), np.stack(b_values)

    def _backward_pass(self, states: Array, controls: Array, a: Array, b: Array) -> tuple[Array, Array]:
        """Compute corrected-mode feedback and feedforward gains backward in time."""
        horizon, control_dim = controls.shape
        state_dim = states.shape[1]
        feedback = np.empty((horizon, control_dim, state_dim))
        feedforward = np.empty((horizon, control_dim))
        value_hessian = self.cost.Qf.copy()
        value_gradient = self.cost.Qf @ (states[-1] - self.cost.x_desired)
        eye_u = np.eye(control_dim)

        # Start from terminal value derivatives and propagate toward time zero.
        for t in range(horizon - 1, -1, -1):
            q_x = self.cost.Q @ (states[t] - self.cost.x_desired) + a[t].T @ value_gradient
            q_u = self.cost.R @ controls[t] + b[t].T @ value_gradient
            q_xx = self.cost.Q + a[t].T @ value_hessian @ a[t]
            q_ux = b[t].T @ value_hessian @ a[t]
            q_uu = self.cost.R + b[t].T @ value_hessian @ b[t]
            q_uu = 0.5 * (q_uu + q_uu.T) + self.config.regularization * eye_u

            # Factor/solve the regularized control Hessian once for both terms.
            solution = np.linalg.solve(q_uu, np.column_stack((q_ux, q_u)))
            feedback[t] = -solution[:, :state_dim]
            feedforward[t] = -solution[:, state_dim]
            k, d = feedback[t], feedforward[t]
            value_gradient = q_x + k.T @ q_uu @ d + k.T @ q_u + q_ux.T @ d
            value_hessian = q_xx + k.T @ q_uu @ k + k.T @ q_ux + q_ux.T @ k
            value_hessian = 0.5 * (value_hessian + value_hessian.T)
        return feedback, feedforward

    def _candidate(self, x0: Array, nominal_states: Array, controls: Array, feedback: Array, feedforward: Array, alpha: float) -> tuple[Array, Array]:
        """Roll out one nonlinear candidate using gains and step size ``alpha``."""
        candidate_controls = np.empty_like(controls)
        candidate_states = [np.asarray(x0, dtype=np.float64)]
        with torch.no_grad():
            x = torch.as_tensor(x0, dtype=self.dtype, device=self.device)
            for t in range(len(controls)):
                delta_x = x.detach().cpu().numpy() - nominal_states[t]
                candidate_controls[t] = controls[t] + alpha * feedforward[t] + feedback[t] @ delta_x
                u = torch.as_tensor(candidate_controls[t], dtype=self.dtype, device=self.device)
                x = self.dynamics(x, u)
                if not torch.isfinite(x).all():
                    raise FloatingPointError("non-finite state encountered during forward pass")
                candidate_states.append(x.detach().cpu().numpy())
        return np.stack(candidate_states), candidate_controls

    def solve(
        self,
        x0: Array,
        initial_controls: Array,
        callback: Callable[[int, ILQRResult], bool | None] | None = None,
    ) -> ILQRResult:
        """Run the selected numerical mode until its stop condition is met."""
        if self.config.numerics == "legacy_exact":
            return self._solve_legacy(x0, initial_controls, callback)

        controls = np.asarray(initial_controls, dtype=np.float64).copy()
        states = self.rollout(x0, controls)
        costs = [self.cost.total(states, controls)]
        update_norms: list[float] = []
        converged = False

        if self.config.max_iterations is None:
            raise ValueError("corrected mode requires a finite max_iterations")
        # Corrected path: linearize, solve locally, then choose a nonlinear
        # forward rollout using either a fixed step or line search.
        for iteration in range(1, self.config.max_iterations + 1):
            a, b = self.linearize(states, controls)
            feedback, feedforward = self._backward_pass(states, controls, a, b)
            previous_controls = controls
            alphas = (self.config.fixed_step,) if self.config.fixed_step is not None else self.config.line_search_steps
            best = None
            for alpha in alphas:
                candidate_states, candidate_controls = self._candidate(
                    x0, states, controls, feedback, feedforward, float(alpha)
                )
                candidate_cost = self.cost.total(candidate_states, candidate_controls)
                if best is None or candidate_cost < best[0]:
                    best = (candidate_cost, candidate_states, candidate_controls)
                if candidate_cost < costs[-1]:
                    break
            assert best is not None
            candidate_cost, states, controls = best
            costs.append(candidate_cost)
            update_norms.append(float(np.linalg.norm(controls - previous_controls)))
            converged = update_norms[-1] <= self.config.tolerance * (1.0 + np.linalg.norm(previous_controls))
            result = ILQRResult(states, controls, costs, update_norms, iteration, converged)
            if callback is not None and callback(iteration, result):
                return result
            if converged:
                return result
        return ILQRResult(states, controls, costs, update_norms, self.config.max_iterations, converged)

    def _solve_legacy(
        self,
        x0: Array,
        initial_controls: Array,
        callback: Callable[[int, ILQRResult], bool | None] | None,
    ) -> ILQRResult:
        """Reproduce the rollout, Riccati recursion, and update in the source files.

        This method deliberately retains explicit inversions, float32 dynamics,
        the historical value-gradient expression, and the L1 update metric.
        Those choices are not recommendations for new solvers; they are the
        compatibility contract for regenerating the original experiments.
        """
        if self.dtype != torch.float32 or self.device.type != "cpu":
            raise ValueError("legacy_exact solver requires CPU float32 dynamics")
        if self.config.fixed_step is None:
            raise ValueError("legacy_exact solver requires the original fixed step")

        controls = np.asarray(initial_controls, dtype=np.float64).copy()
        costs: list[float] = []
        update_norms: list[float] = []
        iterator = (
            range(1, self.config.max_iterations + 1)
            if self.config.max_iterations is not None
            else itertools.count(1)
        )
        states = self.rollout(x0, controls)

        for iteration in iterator:
            # The source scripts rerolled at the start of every iteration.
            states = self.rollout(x0, controls)
            a, b = self.linearize(states, controls)
            # Backward recursion. Lists are prepended exactly as in the original
            # scripts so index t aligns with the forward perturbation pass.
            value_hessians = [self.cost.Qf]
            value_gradients = [
                self.cost.Qf @ (states[-1] - self.cost.x_desired).reshape(-1, 1)
            ]
            feedback: list[Array] = []
            gradient_gain: list[Array] = []
            control_gain: list[Array] = []

            for t in range(len(controls) - 1, -1, -1):
                hessian = b[t].T @ value_hessians[0] @ b[t] + self.cost.R
                # Preserve the three explicit inversions used by every source script.
                k = np.linalg.inv(hessian) @ b[t].T @ value_hessians[0] @ a[t]
                kv = np.linalg.inv(hessian) @ b[t].T
                ku = np.linalg.inv(hessian) @ self.cost.R
                s = a[t].T @ value_hessians[0] @ (a[t] - b[t] @ k) + self.cost.Q
                v = (
                    (a[t] - b[t] @ k).T @ value_gradients[0]
                    - k.T @ self.cost.R @ controls[t].reshape(-1, 1)
                    + self.cost.Q @ states[t].reshape(-1, 1)
                )
                feedback.insert(0, k)
                gradient_gain.insert(0, kv)
                control_gain.insert(0, ku)
                value_hessians.insert(0, s)
                value_gradients.insert(0, v)

            # Forward linear perturbation pass: turn gains into delta_u, then
            # apply the phase-specific fixed step to the full control sequence.
            delta_x = [np.zeros((states.shape[1], 1))]
            delta_u: list[Array] = []
            for t in range(len(controls)):
                du = (
                    -feedback[t] @ delta_x[t]
                    - gradient_gain[t] @ value_gradients[t + 1]
                    - control_gain[t] @ controls[t].reshape(-1, 1)
                )
                delta_x.append(a[t] @ delta_x[t] + b[t] @ du)
                delta_u.append(du)
            delta_u_array = np.asarray(delta_u)
            controls = controls + self.config.fixed_step * delta_u_array.reshape(controls.shape)
            update_norms.append(float(np.sum(np.abs(delta_u_array))))
            states = self.rollout(x0, controls)
            costs.append(self.cost.total(states, controls))
            result = ILQRResult(states, controls, costs, update_norms, iteration, False)
            if callback is not None and callback(iteration, result):
                return result

        return ILQRResult(states, controls, costs, update_norms, iteration, False)
