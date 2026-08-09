"""Compare a consolidated preset with an unmodified source script.

The original file is parsed, but none of its top-level data loading or iLQR loop
is executed. Only its ``arm_controller_model`` function is compiled.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import torch

from arm_ilqr.experiments import PRESETS
from arm_ilqr.model import ArmControllerModel


class _NoDebugger:
    @staticmethod
    def set_trace() -> None:
        raise FloatingPointError("original model entered ipdb.set_trace()")


def _load_original(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "arm_controller_model"
    )
    namespace = {"np": np, "torch": torch, "Nx": 22, "ipdb": _NoDebugger()}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["arm_controller_model"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    torch.manual_seed(0)
    experiment = PRESETS[args.preset]
    layout = experiment.layout
    x = torch.full((layout.state_dim,), 1e-16, dtype=torch.float32)
    x[:4] = torch.tensor([0.1, 1.57, 0.0, 0.0])
    x[layout.activations] = 0.05
    x[layout.controller] = 0.02
    x[layout.theta_initial] = torch.tensor([0.1, 1.57])
    x[layout.theta_final] = torch.tensor([0.25, 1.35])
    u = torch.randn(layout.control_dim, dtype=torch.float32) * 1e-3

    original = _load_original(args.original)
    consolidated = ArmControllerModel(experiment.model)
    y_original = original(x.clone(), u.clone())
    y_consolidated = consolidated(x.clone(), u.clone())
    a_original, b_original = torch.autograd.functional.jacobian(original, (x, u))
    a_consolidated, b_consolidated = torch.autograd.functional.jacobian(consolidated, (x, u))

    differences = {
        "state_step": float(torch.max(torch.abs(y_original - y_consolidated))),
        "state_jacobian": float(torch.max(torch.abs(a_original - a_consolidated))),
        "control_jacobian": float(torch.max(torch.abs(b_original - b_consolidated))),
    }
    for name, difference in differences.items():
        print(f"{name}: max_abs_difference={difference:.9g}")
    if any(value > args.atol for value in differences.values()):
        raise SystemExit("legacy parity check failed")
    print("legacy parity check passed")


if __name__ == "__main__":
    main()
