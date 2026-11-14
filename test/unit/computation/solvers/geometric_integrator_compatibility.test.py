"""Tests for geometric integrator compatibility helpers."""

from __future__ import annotations

import torch

from relic.computation.solvers import SolverStatus
from relic.computation.solvers.geometric import (
    get_implicit_midpoint_compatibility_spec,
    transactional_implicit_midpoint_step,
)
from relic.core.state import StateManager


def test_implicit_midpoint_compatibility_spec_uses_existing_solver():
    spec = get_implicit_midpoint_compatibility_spec()

    assert spec.solver_name == "implicit_midpoint"
    assert spec.uses_existing_solver is True
    assert "checkpoint_restore" in spec.transaction_controls


def test_transactional_implicit_midpoint_step_restores_on_failed_validation():
    manager = StateManager()
    manager.initialize_state({"oscillator": torch.tensor([1.0])})

    def rhs(y_mid: torch.Tensor, _time: float) -> torch.Tensor:
        return -y_mid

    def commit_result(result) -> None:
        manager.set_state("oscillator", result.solution, validate=False)

    result = transactional_implicit_midpoint_step(
        rhs,
        torch.tensor([1.0]),
        t_current=0.0,
        dt=0.1,
        state_manager=manager,
        commit_result=commit_result,
        validator=lambda _result: False,
    )

    restored = manager.get_current_state()
    assert result.status == SolverStatus.INVALID_INPUT
    assert result.metadata["transaction_restored"] is True
    assert torch.allclose(
        restored["oscillator"],
        torch.tensor([1.0], device=restored["oscillator"].device),
    )
