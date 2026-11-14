"""Tests for Hodge validation rules."""


import pytest
import torch

from relic.computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from relic.validation.framework.models import ValidationContext, ValidationLevel
from relic.validation.framework.registry import get_global_registry
from relic.validation.framework.rules import hodge as rules


def _spaces():
    domain = CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="primal",
        n_dofs=2,
        units="m",
        extensivity="integral",
        physical_domain="hydrosphere",
        semantic_name="hydraulic_head_difference",
        orientation_convention="outward_positive",
    )
    codomain = CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="dual",
        n_dofs=2,
        units="kg s-1",
        extensivity="integral",
        physical_domain="hydrosphere",
        semantic_name="water_flux",
        orientation_convention="outward_positive",
    )
    return domain, codomain


def _hodge(matrix):
    domain, codomain = _spaces()
    return HodgeStarSpec(
        name="soil_water_hodge",
        domain=domain,
        codomain=codomain,
        matrix=matrix,
        parameter_keys=("hydrosphere.soil.transport.hodge.k",),
        metric_keys=("hydrosphere.soil.transport.metric.area",),
        material_law="darcy",
    )


def _strict_hodge(matrix):
    domain, codomain = _spaces()
    return HodgeStarSpec(
        name="strict_soil_water_hodge",
        domain=domain,
        codomain=codomain,
        matrix=matrix,
        parameter_keys=("hydrosphere.soil.transport.hodge.k",),
        metric_keys=("hydrosphere.soil.transport.metric.area",),
        material_law="darcy",
        complex_dimension=2,
        operator_units="kg m-1 s-1",
        role="metric_hodge",
        definiteness="positive_definite",
    )


def test_hodge_rule_registers_with_structural_tags():
    registry = get_global_registry()

    assert "hodge_passivity_structure" in registry.list_rules(tags=["hodge"])


def test_hodge_rule_passes_positive_diagonal_hodge():
    context = ValidationContext(metadata={"hodge_specs": [_hodge(torch.eye(2))]})

    results = rules.validate_hodge_passivity_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.INFO
    assert results[0].passed is True
    assert results[0].metrics["min_eigenvalue"] == 1.0


def test_hodge_rule_fails_negative_diagonal_hodge():
    context = ValidationContext(
        metadata={"hodge_specs": [_hodge(torch.diag(torch.tensor([1.0, -0.1])))]}
    )

    results = rules.validate_hodge_passivity_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "positive semidefinite" in results[0].message


def test_hodge_rule_propagates_strict_structure_and_parameter_binding():
    context = ValidationContext(
        metadata={
            "hodge_specs": [_strict_hodge(torch.eye(2))],
            "strict": True,
            "parameter_resolver": lambda key: {"key": key},
        }
    )

    results = rules.validate_hodge_passivity_structure(context)

    assert len(results) == 1
    assert results[0].passed is True


def test_hodge_rule_strict_mode_rejects_missing_parameter_binding():
    context = ValidationContext(
        metadata={
            "hodge_specs": [_strict_hodge(torch.eye(2))],
            "strict": True,
            "parameter_resolver": lambda key: None,
        }
    )

    results = rules.validate_hodge_passivity_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "ParameterManager key" in results[0].message


def test_hodge_rule_strict_mode_rejects_unspecified_claim():
    context = ValidationContext(
        metadata={
            "hodge_specs": [_hodge(torch.eye(2))],
            "strict": True,
            "parameter_resolver": lambda key: {"key": key},
        }
    )

    results = rules.validate_hodge_passivity_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "explicit role" in results[0].message


def test_hodge_rule_propagates_spec_defects():
    class _DefectiveSpec:
        name = "defective_hodge"

        def validate(self, *, passive, strict, parameter_resolver):
            raise AttributeError("hodge spec defect")

    with pytest.raises(AttributeError, match="hodge spec defect"):
        rules.validate_hodge_spec(_DefectiveSpec())


def test_hodge_rule_reports_a_missing_payload_as_failed():
    result = rules.validate_hodge_spec(None)

    assert result.passed is False
    assert "requires a HodgeStarSpec payload" in result.message


def test_hodge_rule_does_not_report_unevaluated_passivity():
    result = rules.validate_hodge_spec(
        _hodge(torch.tensor([[2.0, 0.0], [0.0, 3.0]])),
        passive=False,
    )

    assert result.passed is True
    assert "not evaluated" in result.message
    assert "min_eigenvalue" not in result.metrics


def test_hodge_rule_requires_the_min_eigenvalue_of_an_evaluated_audit():
    class _IncompleteAuditSpec:
        name = "incomplete_audit"

        def validate(self, *, passive, strict, parameter_resolver):
            return {"passivity_evaluated": True}

    with pytest.raises(KeyError, match="min_eigenvalue"):
        rules.validate_hodge_spec(_IncompleteAuditSpec())
