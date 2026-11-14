"""Unit tests for Jacobian-based identifiability diagnostics."""

from __future__ import annotations

import pytest
import torch

from relic.coupling.core.audit import AuditRecord
from relic.optimization.identifiability import (
    compute_identifiability,
    get_identifiability_summary,
)
from relic.parameters import get_global_manager


def test_full_rank_jacobian() -> None:
    jacobian = {
        "obs_1": {"x_1": torch.tensor(1.0), "x_2": torch.tensor(0.0)},
        "obs_2": {"x_1": torch.tensor(0.0), "x_2": torch.tensor(1.0)},
    }
    summary = get_identifiability_summary(jacobian, tol=1e-8)
    assert summary["rank"] == 2
    assert summary["nullity"] == 0
    assert summary["condition_number"] == pytest.approx(1.0, rel=1e-8, abs=1e-8)


def test_rank_deficient_jacobian() -> None:
    jacobian = {
        "obs_1": {"x_1": torch.tensor(1.0), "x_2": torch.tensor(1.0)},
        "obs_2": {"x_1": torch.tensor(2.0), "x_2": torch.tensor(2.0)},
    }
    summary = get_identifiability_summary(jacobian, tol=1e-8)
    assert summary["rank"] == 1
    assert summary["nullity"] == 1


def test_condition_number() -> None:
    well_conditioned = {
        "obs_1": {"x_1": torch.tensor(1.0), "x_2": torch.tensor(0.0)},
        "obs_2": {"x_1": torch.tensor(0.0), "x_2": torch.tensor(1.0)},
    }
    ill_conditioned = {
        "obs_1": {"x_1": torch.tensor(1.0), "x_2": torch.tensor(0.0)},
        "obs_2": {"x_1": torch.tensor(0.0), "x_2": torch.tensor(1.0e-9)},
    }
    cond_well = get_identifiability_summary(well_conditioned, tol=1e-12)["condition_number"]
    cond_ill = get_identifiability_summary(ill_conditioned, tol=1e-12)["condition_number"]
    assert cond_ill > cond_well


def test_wide_jacobian_returns_complete_right_nullspace() -> None:
    jacobian = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    details = compute_identifiability(jacobian, tol=1e-12)
    nullspace = torch.tensor(
        details["unidentifiable_dirs"],
        dtype=torch.float64,
    )

    assert details["rank"] == 2
    assert details["nullity"] == 1
    assert details["condition_number"] == float("inf")
    assert nullspace.shape == (1, 3)
    torch.testing.assert_close(
        nullspace @ nullspace.T,
        torch.eye(1, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        jacobian @ nullspace.T,
        torch.zeros((2, 1), dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_rank_deficient_condition_number_is_infinite() -> None:
    jacobian = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=torch.float64,
    )

    details = compute_identifiability(jacobian, tol=1e-12)

    assert details["rank"] == 1
    assert details["nullity"] == 1
    assert details["condition_number"] == float("inf")


def test_relative_rank_decision_is_invariant_to_global_scaling() -> None:
    jacobian = torch.diag(torch.tensor([1.0, 1.0e-8], dtype=torch.float64))

    original = compute_identifiability(jacobian, tol=1e-6)
    scaled = compute_identifiability(jacobian * 1.0e-12, tol=1e-6)

    assert original["rank"] == 1
    assert scaled["rank"] == original["rank"]
    assert scaled["nullity"] == original["nullity"]
    assert scaled["condition_number"] == original["condition_number"]
    assert scaled["rank_tolerance"] == pytest.approx(
        original["rank_tolerance"] * 1.0e-12,
        rel=1e-12,
    )


def test_zero_jacobian_returns_full_nullspace() -> None:
    jacobian = torch.zeros((2, 3), dtype=torch.float64)

    details = compute_identifiability(jacobian, tol=1e-6)
    nullspace = torch.tensor(
        details["unidentifiable_dirs"],
        dtype=torch.float64,
    )

    assert details["rank"] == 0
    assert details["nullity"] == 3
    assert details["condition_number"] == float("inf")
    assert details["rank_tolerance"] == 0.0
    assert nullspace.shape == (3, 3)
    torch.testing.assert_close(
        nullspace @ nullspace.T,
        torch.eye(3, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        jacobian @ nullspace.T,
        torch.zeros((2, 3), dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_empty_dictionary_rows_preserve_observation_labels_and_shape() -> None:
    details = compute_identifiability({"obs_2": {}, "obs_1": {}}, tol=1e-6)

    assert details["shape"] == [2, 0]
    assert details["obs_vars"] == ["obs_1", "obs_2"]
    assert details["state_vars"] == []
    assert details["rank"] == 0
    assert details["nullity"] == 0


@pytest.mark.parametrize("tol", [-1.0, float("nan"), float("inf")])
def test_invalid_relative_tolerance_is_rejected(tol: float) -> None:
    with pytest.raises(ValueError, match="tol must be finite and non-negative"):
        compute_identifiability(torch.eye(2, dtype=torch.float64), tol=tol)


def test_non_matrix_and_non_finite_jacobians_are_rejected() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        compute_identifiability(torch.ones(2, dtype=torch.float64))

    non_finite = torch.tensor(
        [[1.0, float("nan")]],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="finite"):
        compute_identifiability(non_finite)


def test_audit_record_identifiability_diagnostics() -> None:
    record = AuditRecord(operation="identifiability-test", parameter_manager=get_global_manager())
    record.add_identifiability_diagnostics(
        jacobian_rank=3,
        jacobian_nullity=1,
        condition_number=42.0,
    )
    assert "identifiability" in record.custom_data
    payload = record.custom_data["identifiability"]
    assert payload["jacobian_rank"] == 3
    assert payload["jacobian_nullity"] == 1
    assert payload["condition_number"] == pytest.approx(42.0)


def test_missing_jacobian_entry_is_not_read_as_zero_sensitivity() -> None:
    """A row without an entry for a state variable is incomplete, not a zero.

    Filling the gap with 0.0 would add a false null-space direction.
    """
    jacobian = {
        "obs_1": {"x_1": torch.tensor(1.0), "x_2": torch.tensor(0.5)},
        "obs_2": {"x_1": torch.tensor(2.0)},
    }
    with pytest.raises(KeyError, match="obs_2.*x_2"):
        compute_identifiability(jacobian, tol=1e-12)
