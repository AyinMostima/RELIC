"""Guardrails against a shadow computation parameter-schema registry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from relic.computation.operators.registry import OperatorRegistry
from relic.computation.solvers.registry import SolverRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = PROJECT_ROOT / "relic" / "computation" / "config" / "parameter_schemas.py"


def test_shadow_parameter_schema_module_is_absent() -> None:
    assert not SCHEMA_FILE.exists()
    assert (
        importlib.util.find_spec("relic.computation.config.parameter_schemas")
        is None
    )


def test_runtime_registries_are_the_only_provider_identity_owners() -> None:
    assert OperatorRegistry.is_registered("gradient.central_difference_1d")
    assert OperatorRegistry.is_registered("transport.diffusion")
    assert not OperatorRegistry.is_registered("gradient.central_difference")
    assert not OperatorRegistry.is_registered("diffusion.finite_difference")
    assert SolverRegistry.is_registered("backward_euler")
    assert SolverRegistry.is_registered("thomas_algorithm")
    assert not SolverRegistry.is_registered("unknown_solver")
