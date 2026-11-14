"""Tests for DEC complex validation rules."""


import pytest
import torch

from relic.computation.operators.dec import DECComplexBundle
from relic.validation.framework.models import ValidationContext, ValidationLevel
from relic.validation.framework.registry import get_global_registry
from relic.validation.framework.rules import dec_complex as rules


def _valid_bundle():
    d0 = torch.tensor(
        [
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    d1 = torch.tensor([[1.0, 1.0, 1.0]])
    return DECComplexBundle(
        complex_id="soil_column",
        dimension=2,
        n_cells={0: 3, 1: 3, 2: 1},
        D={0: d0, 1: d1},
        boundary_masks={},
        orientation={},
        cell_measures={},
        dual_measures={},
        orientation_convention="outward_positive",
    )


def test_dec_complex_rule_registers_with_structural_tags():
    registry = get_global_registry()

    assert "dec_complex_structure" in registry.list_rules(tags=["structural"])


def test_dec_complex_rule_passes_valid_complex():
    context = ValidationContext(metadata={"dec_complex_bundles": [_valid_bundle()]})

    results = rules.validate_dec_complex_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.INFO
    assert results[0].passed is True
    assert results[0].metrics["max_residual"] == 0.0


def test_dec_complex_rule_fails_nonzero_composition():
    bad_bundle = DECComplexBundle(
        complex_id="bad_complex",
        dimension=2,
        n_cells={0: 2, 1: 2, 2: 1},
        D={
            0: torch.tensor([[-1.0, 1.0], [0.0, -1.0]]),
            1: torch.tensor([[1.0, 1.0]]),
        },
        boundary_masks={},
        orientation={},
        cell_measures={},
        dual_measures={},
        orientation_convention="outward_positive",
    )
    context = ValidationContext(metadata={"dec_complex_bundles": [bad_bundle]})

    results = rules.validate_dec_complex_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "D[1] @ D[0]" in results[0].message


def test_dec_complex_rule_propagates_strict_completeness_requirement():
    context = ValidationContext(
        metadata={
            "dec_complex_bundles": [_valid_bundle()],
            "strict": True,
        }
    )

    results = rules.validate_dec_complex_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "boundary_masks" in results[0].message


def test_dec_complex_rule_propagates_bundle_defects():
    class _DefectiveBundle:
        complex_id = "defective_complex"

        def validate_complex(self, *, tol, strict):
            raise KeyError("internal degree index defect")

    with pytest.raises(KeyError, match="internal degree index defect"):
        rules.validate_dec_complex_bundle(_DefectiveBundle(), tol=0.0)


def test_dec_complex_rule_reports_a_missing_payload_as_failed():
    result = rules.validate_dec_complex_bundle(None, tol=0.0)

    assert result.passed is False
    assert "requires a DECComplexBundle payload" in result.message


class _RecordingBundle:
    complex_id = "recording_complex"

    def __init__(self):
        self.tolerances = []

    def validate_complex(self, *, tol, strict):
        self.tolerances.append(tol)
        return {"D[1] @ D[0]": 0.0}


def test_dec_complex_bundle_check_requires_an_explicit_tolerance():
    with pytest.raises(TypeError, match="tol"):
        rules.validate_dec_complex_bundle(_RecordingBundle())


class _Manager:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def get(self, key, **kwargs):
        self.calls.append((key, kwargs))
        return self.value


def test_dec_complex_rule_reads_the_tolerance_from_the_parameter_database(monkeypatch):
    manager = _Manager(1.0e-9)
    monkeypatch.setattr(rules, "get_global_manager", lambda: manager)
    bundle = _RecordingBundle()

    results = rules.validate_dec_complex_structure(ValidationContext(metadata={"dec_complex_bundles": [bundle]}))

    assert results[0].passed is True
    assert bundle.tolerances == [1.0e-9]
    assert manager.calls[0][0] == "validation.dec_complex.cochain_residual_tolerance"
    assert manager.calls[0][1]["required"] is True


@pytest.mark.parametrize("key", ["tol", "dec_tol"])
def test_an_explicit_payload_tolerance_overrides_the_database(monkeypatch, key):
    manager = _Manager(1.0e-9)
    monkeypatch.setattr(rules, "get_global_manager", lambda: manager)

    assert rules.resolve_dec_complex_tolerance({key: 1.0e-6}) == 1.0e-6
    assert manager.calls == []


def test_the_database_tolerance_entry_exists():
    from relic.parameters import get_global_manager

    value = get_global_manager().get(
        "validation.dec_complex.cochain_residual_tolerance", units="dimensionless", required=True
    )
    assert float(value) == 0.0


def test_orchestration_structural_check_uses_the_same_tolerance_source(monkeypatch):
    from relic.orchestration import validator as orchestration_validator

    manager = _Manager(2.0e-9)
    monkeypatch.setattr(rules, "get_global_manager", lambda: manager)
    bundle = _RecordingBundle()

    result = orchestration_validator._run_structural_assertion("cochain_complex_check", {"bundle": bundle})

    assert result.passed is True
    assert bundle.tolerances == [2.0e-9]
