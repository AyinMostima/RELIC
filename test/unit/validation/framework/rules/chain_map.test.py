"""Tests for chain-map structural validation rules."""


import pytest
import torch

from relic.validation.framework.models import ValidationContext, ValidationLevel
from relic.validation.framework.registry import get_global_registry
from relic.validation.framework.rules import chain_map as rules


def test_chain_map_rule_registers_with_structural_tags():
    registry = get_global_registry()

    assert "chain_map_structure" in registry.list_rules(tags=["chain_map"])


def test_chain_map_rule_passes_commuting_square():
    D = torch.tensor([[-1.0, 1.0]])
    identity_0 = torch.eye(2)
    identity_1 = torch.eye(1)
    context = ValidationContext(
        metadata={
            "chain_maps": [
                {
                    "name": "identity_regrid",
                    "source_D": D,
                    "target_D": D,
                    "R_k": identity_0,
                    "R_next": identity_1,
                    "tol": 0.0,
                }
            ]
        }
    )

    results = rules.validate_chain_map_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.INFO
    assert results[0].metrics["residual_norm"] == 0.0


def test_chain_map_rule_fails_noncommuting_square():
    context = ValidationContext(
        metadata={
            "chain_maps": [
                {
                    "name": "bad_regrid",
                    "source_D": torch.tensor([[-1.0, 1.0]]),
                    "target_D": torch.tensor([[1.0, 1.0]]),
                    "R_k": torch.eye(2),
                    "R_next": torch.eye(1),
                    "tol": 0.0,
                }
            ]
        }
    )

    results = rules.validate_chain_map_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "chain map" in results[0].message.lower()


def _bihss_payload():
    derivative = torch.tensor([[-1.0, 1.0]])
    return {
        "name": "identity_bihss_morphism",
        "source_D": derivative,
        "target_D": derivative,
        "R_k": torch.eye(2),
        "R_next": torch.eye(1),
        "tol": 0.0,
        "bihss_morphism": True,
        "source_ledger": torch.tensor([[1.0, 1.0]]),
        "target_ledger": torch.tensor([[1.0, 1.0]]),
        "source_state": torch.tensor([1.0, 2.0]),
        "target_state": torch.tensor([1.0, 2.0]),
        "source_energy": torch.tensor(2.5),
        "target_energy": torch.tensor(2.5),
        "source_entropy": torch.tensor(3.0),
        "target_entropy": torch.tensor(3.0),
        "storage_tol": 0.0,
        "source_input": torch.tensor([2.0]),
        "target_input": torch.tensor([2.0]),
        "source_effort": torch.tensor([3.0]),
        "target_effort": torch.tensor([3.0]),
        "port_map": torch.eye(1),
        "power_tol": 0.0,
        "source_entropy_production": torch.tensor(1.0),
        "target_entropy_production": torch.tensor(1.0),
        "entropy_tol": 0.0,
    }


def test_chain_map_rule_accepts_complete_bihss_morphism():

    result = rules.validate_chain_map_payload(_bihss_payload())

    assert result.passed is True
    assert result.metrics["ledger_residual_norm"] == 0.0
    assert result.metrics["port_power_residual"] == 0.0


def test_chain_map_rule_rejects_incomplete_bihss_morphism_claim():
    payload = _bihss_payload()
    del payload["source_ledger"]

    result = rules.validate_chain_map_payload(payload)

    assert result.passed is False
    assert "source_ledger" in result.message


def test_chain_map_rule_rejects_bihss_ledger_violation():
    payload = _bihss_payload()
    payload["target_ledger"] = torch.tensor([[1.0, 0.0]])

    result = rules.validate_chain_map_payload(payload)

    assert result.passed is False
    assert "ledger" in result.message.lower()


def test_chain_map_rule_accepts_explicit_open_port_ledger_compensation():
    payload = _bihss_payload()
    payload["target_ledger"] = torch.tensor([[1.0, 0.0]])
    payload["ledger_port_compensation"] = torch.tensor([[0.0, 1.0]])
    payload["open_port_route"] = "boundary_export"

    result = rules.validate_chain_map_payload(payload)

    assert result.passed is True


def test_chain_map_rule_rejects_bihss_storage_error_above_bound():
    payload = _bihss_payload()
    payload["target_energy"] = torch.tensor(3.0)
    payload["storage_tol"] = 0.1

    result = rules.validate_chain_map_payload(payload)

    assert result.passed is False
    assert "energy storage" in result.message.lower()


def test_chain_map_rule_rejects_bihss_port_power_violation():
    payload = _bihss_payload()
    payload["target_effort"] = torch.tensor([4.0])

    result = rules.validate_chain_map_payload(payload)

    assert result.passed is False
    assert "port power" in result.message.lower()


def test_chain_map_rule_rejects_systematic_entropy_production_reduction():
    payload = _bihss_payload()
    payload["target_entropy_production"] = torch.tensor(0.5)
    payload["entropy_tol"] = 0.1

    result = rules.validate_chain_map_payload(payload)

    assert result.passed is False
    assert "entropy production" in result.message.lower()


def test_chain_map_rule_propagates_payload_accessor_defects():
    class _DefectivePayload(dict):
        def get(self, key, default=None):
            if key == "source_D":
                raise AttributeError("payload accessor defect")
            return super().get(key, default)

    with pytest.raises(AttributeError, match="payload accessor defect"):
        rules.validate_chain_map_payload(_DefectivePayload(name="defective_payload"))
