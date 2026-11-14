"""Guardrails for canonical numerical-stability owners."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

import relic.computation.solvers as solver_package
from relic.computation.operators.functional import (
    compute_cfl_number,
    compute_diffusion_stability,
)
from relic.computation.solvers import SolverBase, SolverFactory, SolverRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHADOW_STABILITY_MODULE = (
    PROJECT_ROOT / "relic" / "computation" / "solvers" / "stability_config.py"
)


def test_shadow_stability_module_is_absent() -> None:
    assert not SHADOW_STABILITY_MODULE.exists()
    assert (
        importlib.util.find_spec("relic.computation.solvers.stability_config")
        is None
    )


def test_solver_package_does_not_export_shadow_stability_manager() -> None:
    forbidden = {
        "StabilityConfiguration",
        "StabilityLevel",
        "NumericalStabilityManager",
        "IterativeSolverConfig",
        "TimeSteppingConfig",
        "LinearSolverConfig",
        "get_stability_manager",
        "set_stability_level",
        "get_solver_stability_config",
    }

    assert not forbidden.intersection(solver_package.__all__)
    assert not [name for name in forbidden if hasattr(solver_package, name)]


def test_canonical_solver_owners_remain_public() -> None:
    assert solver_package.SolverBase is SolverBase
    assert solver_package.SolverFactory is SolverFactory
    assert solver_package.SolverRegistry is SolverRegistry
    assert SolverRegistry.is_registered("backward_euler")
    assert SolverRegistry.is_registered("implicit_midpoint")


def test_functional_stability_operators_match_independent_oracles() -> None:
    velocity = torch.tensor([2.0], dtype=torch.float64)
    diffusivity = torch.tensor([0.1], dtype=torch.float64)

    cfl = compute_cfl_number(
        velocity,
        grid_spacing=4.0,
        time_step=0.5,
    )
    diffusion_number = compute_diffusion_stability(
        diffusivity,
        grid_spacing=2.0,
        time_step=1.0,
    )

    assert torch.equal(cfl, torch.tensor([0.25], dtype=torch.float64))
    assert torch.equal(
        diffusion_number,
        torch.tensor([0.025], dtype=torch.float64),
    )
