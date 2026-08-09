"""Command-line runner for all consolidated experiment presets."""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .experiments import PRESETS, Experiment
from .model import ArmControllerModel
from .solver import ILQRConfig, ILQRResult, ILQRSolver


def _job_id(value: int | None) -> int:
    if value is None:
        raw = os.getenv("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("pass --job-id or set SLURM_ARRAY_TASK_ID")
        value = int(raw)
    if not 1 <= value <= 8:
        raise ValueError(f"job id must be in [1, 8], got {value}")
    return value


def _hand_position(theta: np.ndarray) -> np.ndarray:
    return np.array([
        0.15 * np.cos(theta[0]) + 0.21 * np.cos(theta[0] + theta[1]),
        0.15 * np.sin(theta[0]) + 0.21 * np.sin(theta[0] + theta[1]),
    ])


def _output_directory(experiment: Experiment, data_root: Path) -> Path:
    output = data_root / experiment.output_stage
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save_optimizer_state(
    experiment: Experiment,
    result: ILQRResult,
    job_id: int,
    data_root: Path,
) -> None:
    output = _output_directory(experiment, data_root)
    np.save(output / f"delu_traj_{job_id}.npy", np.asarray(result.update_norms))
    np.save(output / f"u_k_traj_{job_id}.npy", result.controls)
    metadata = {
        "preset": experiment.name,
        "numerics": experiment.model.numerics,
        "job_id": job_id,
        "iterations": result.iterations,
        "converged": result.converged,
        "cost": result.costs[-1],
        "update_metric": result.update_norms[-1] if result.update_norms else None,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "nx": experiment.model.nx,
        "dt": experiment.model.dt,
        "horizon": experiment.horizon,
        "fixed_step": experiment.fixed_step,
    }
    (output / f"run_{job_id}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _save_trajectory(
    experiment: Experiment,
    result: ILQRResult,
    job_id: int,
    data_root: Path,
) -> None:
    output = _output_directory(experiment, data_root)
    layout = experiment.layout
    states = result.states
    np.save(output / f"theta1_{job_id}.npy", states[1:, 0])
    np.save(output / f"theta2_{job_id}.npy", states[1:, 1])
    np.save(output / f"net_traj_{job_id}.npy", states[1:, layout.controller])
    np.save(output / f"activation_traj_{job_id}.npy", states[1:, layout.activations])
    if experiment.phase == "stabilization":
        np.save(output / f"xk1_init_fixed_{job_id}.npy", states[-1])
    elif experiment.phase == "fixation":
        np.save(output / f"xk1_init_move_{job_id}.npy", states[-1])


def run(args: argparse.Namespace) -> ILQRResult:
    if args.max_iterations is not None and args.max_iterations < 1:
        raise ValueError("--max-iterations must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    experiment = PRESETS[args.preset]
    if not args.corrected and (
        args.balanced_readout
        or args.line_search
        or args.device != "cpu"
    ):
        raise ValueError(
            "legacy_exact requires Nx=22, dt=0.005, CPU, legacy readout, and the fixed step; "
            "pass --corrected only for the supported numerical alternatives"
        )
    if args.corrected or args.balanced_readout:
        model_config = replace(
            experiment.model,
            readout_mode="balanced" if args.balanced_readout else experiment.model.readout_mode,
            numerics="corrected" if args.corrected else "legacy_exact",
        )
        experiment = replace(experiment, model=model_config)
    job_id = _job_id(args.job_id)
    # The original target table is an object-dtype .npy file. Preserve its
    # loading behavior; only use a trusted, locally generated target file.
    theta_table = np.load(args.final_theta, allow_pickle=True)
    if theta_table.ndim != 2 or theta_table.shape[1] != 2 or theta_table.shape[0] < job_id:
        raise ValueError(f"{args.final_theta} must have shape (at least {job_id}, 2)")
    theta_desired = np.asarray(theta_table[job_id - 1], dtype=np.float64)
    data_root = Path(args.data_root)
    x0 = experiment.initial_state(theta_desired, job_id, data_root)
    cost = experiment.build_cost(theta_desired)
    controls = np.zeros((experiment.horizon - 1, experiment.layout.control_dim))
    model = ArmControllerModel(experiment.model)
    if not args.corrected:
        torch.autograd.set_detect_anomaly(True)
    max_iterations = args.max_iterations
    if max_iterations is None:
        max_iterations = experiment.max_iterations if not args.corrected else (experiment.max_iterations or 500)
    solver = ILQRSolver(
        model,
        cost,
        ILQRConfig(
            max_iterations=max_iterations,
            tolerance=args.tolerance,
            regularization=args.regularization,
            fixed_step=None if args.line_search else experiment.fixed_step,
            numerics="corrected" if args.corrected else "legacy_exact",
        ),
        dtype=torch.float64 if args.corrected else torch.float32,
        device=args.device,
    )

    def checkpoint(iteration: int, result: ILQRResult) -> bool:
        # The source scripts saved u and the update history every iteration.
        _save_optimizer_state(experiment, result, job_id, data_root)
        if iteration % args.checkpoint_every:
            return False
        _save_trajectory(experiment, result, job_id, data_root)
        hand = _hand_position(result.states[-1, :2])
        initial = _hand_position(result.states[-1, experiment.layout.theta_initial])
        target = _hand_position(result.states[-1, experiment.layout.theta_final])
        print(json.dumps({
            "iteration": iteration,
            "cost": result.costs[-1],
            "update_norm": result.update_norms[-1],
            "initial_error_m": float(np.max(np.abs(hand - initial))),
            "target_error_m": float(np.max(np.abs(hand - target))),
        }))
        if experiment.phase == "fixation":
            return bool(np.any(np.abs(hand - initial) > 0.025 / 3.0))
        if experiment.phase == "movement":
            return bool(np.all(np.abs(hand - target) < 0.005))
        return False

    result = solver.solve(x0, controls, callback=checkpoint)
    _save_optimizer_state(experiment, result, job_id, data_root)
    _save_trajectory(experiment, result, job_id, data_root)
    return result


def build_parser(default_preset: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=default_preset, required=default_preset is None)
    parser.add_argument("--job-id", type=int)
    parser.add_argument("--final-theta", type=Path, default=Path("final_theta_traj.npy"))
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("ARM_ILQR_DATA_ROOT", "/gpfs/radev/pi/saxena/ma2493/opt_data_thesis")))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--corrected", action="store_true", help="use corrected float64 iLQR instead of legacy-exact numerics")
    parser.add_argument("--line-search", action="store_true", help="accept only cost-reducing step sizes")
    parser.add_argument("--balanced-readout", action="store_true", help="assign every decoupled controller unit to a muscle")
    return parser


def main(default_preset: str | None = None) -> None:
    args = build_parser(default_preset).parse_args()
    run(args)


if __name__ == "__main__":
    main()
