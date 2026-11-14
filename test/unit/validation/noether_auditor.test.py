"""Unit tests for Noether-style invariant drift diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from relic.coupling.core.audit import AuditRecord
from relic.parameters import get_global_manager
from relic.validation.conservation_guardian import BudgetCheckResult, ConservationGuardian


class DummyRegistry:
    def __init__(self, domains):
        self._domains = domains

    def get_domain_members(self, domain):
        return list(self._domains.get(domain, []))


def _guardian() -> ConservationGuardian:
    return ConservationGuardian(DummyRegistry({"carbon.total": ["pool_a", "pool_b"]}))


def test_invariant_drift_zero_for_balanced_fluxes() -> None:
    guardian = _guardian()
    incidence_row = np.array([1.0, -1.0], dtype=np.float64)
    flux_vector = {
        "f1": torch.tensor(2.5),
        "f2": torch.tensor(2.5),
    }
    drift = guardian.compute_invariant_drift(incidence_row, flux_vector, dt=3600.0)
    assert drift == pytest.approx(0.0, abs=1e-12)


def test_invariant_drift_nonzero_for_imbalanced() -> None:
    guardian = _guardian()
    incidence_row = np.array([1.0, -1.0], dtype=np.float64)
    flux_vector = {
        "f1": torch.tensor(3.0),
        "f2": torch.tensor(2.0),
    }
    drift = guardian.compute_invariant_drift(incidence_row, flux_vector, dt=10.0)
    assert drift == pytest.approx(10.0, abs=1e-12)


def test_budget_check_result_has_drift_field() -> None:
    result = BudgetCheckResult(
        domain="carbon.total",
        members=["pool_a"],
        sum_flux_in=1.0,
        sum_flux_out=1.0,
        delta_storage=0.0,
        residual=0.0,
        passed=True,
        tolerance_used=1e-10,
        invariant_drift=0.0,
    )
    assert hasattr(result, "invariant_drift")
    assert result.invariant_drift == pytest.approx(0.0)


def test_audit_record_topology_diagnostics() -> None:
    record = AuditRecord(operation="noether-test", parameter_manager=get_global_manager())
    record.add_topology_diagnostics(
        invariant_drift=1.2e-6,
        betti_number=1,
        graph_rank=3,
        graph_nullity=1,
    )
    assert "topology" in record.custom_data
    payload = record.custom_data["topology"]
    assert payload["invariant_drift"] == pytest.approx(1.2e-6)
    assert payload["betti_number"] == 1
    assert payload["graph_rank"] == 3
    assert payload["graph_nullity"] == 1
