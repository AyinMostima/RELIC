"""Tests for IPHS structural validation rules."""

import pytest
import torch

from relic.computation.operators.iphs import IPHSStructureSpec
from relic.validation.framework.models import ValidationContext, ValidationLevel
from relic.validation.framework.registry import get_global_registry
from relic.validation.framework.rules import iphs as rules


_EXECUTABLE_IPHS_SOURCE = (
    "relic.computation.operators.iphs.IPHSStructureSpec.canonical_drift"
)


def _spec(
    M=None,
    J=None,
    *,
    R=None,
    formulation="unspecified",
    energy_gradient=None,
    contribution_count=0,
    contribution_sources=(),
):
    return IPHSStructureSpec(
        name="damped_mode",
        state_vars=("x", "p"),
        energy_functional="E",
        entropy_functional="S",
        J=J if J is not None else torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        M=M if M is not None else torch.eye(2),
        G=None,
        stoichiometric_N=None,
        ports=(),
        R=R,
        formulation=formulation,
        energy_gradient=energy_gradient,
        contribution_count=contribution_count,
        contribution_sources=contribution_sources, poisson_operator="matrix_snapshot", reaction_representation="independent",
    )


def test_iphs_rule_registers_with_structural_tags():
    registry = get_global_registry()

    assert "iphs_structure" in registry.list_rules(tags=["iphs"])


def test_iphs_rule_passes_valid_structure():
    context = ValidationContext(metadata={"iphs_specs": [_spec()]})

    results = rules.validate_iphs_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.INFO
    assert results[0].passed is True


def test_zero_field_rule_preserves_scope_in_serialized_detail_and_keeps_runtime_gate():
    from relic.validation.framework.models import ValidationResultModel

    spec = IPHSStructureSpec(
        name="zero_field_rule", state_vars=("energy", "coordinate"),
        energy_functional="E", entropy_functional="S", formulation="generic",
        J=torch.zeros((2, 2)), M=torch.diag(torch.tensor([0.0, 2.0])),
        G=None, stoichiometric_N=None, ports=(), poisson_operator="zero_field",
        energy_gradient=torch.tensor([1.0, 0.0]), entropy_gradient=torch.tensor([0.0, 3.0]),
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,), reaction_representation="independent",
    )
    result = rules.validate_iphs_spec(spec)
    assert result.passed is True
    restored = ValidationResultModel.model_validate_json(result.model_dump_json())
    audit = restored.details[0].metadata["iphs_audit"]
    assert audit["poisson_operator"] == "zero_field"
    assert audit["jacobi_identity_verified"] is True
    assert audit["jacobi_identity_basis"] == "canonical_zero_field"
    assert audit["evidence_scope"] == "zero_field_structural_pre_gate"
    assert audit["process_level_status"] == "[N] runtime_unbound"
    assert "runtime_verified_level" not in audit
    strict = rules.validate_iphs_spec(spec, required_level=5)
    assert strict.passed is False
    assert "process runtime H/E/S effort linkage" in strict.message
    spec.J[0, 1] = 1.0e-30
    mutated = rules.validate_iphs_spec(spec)
    assert mutated.passed is False
    assert "zero_field" in mutated.message


def test_iphs_rule_fails_negative_dissipation():
    context = ValidationContext(
        metadata={"iphs_specs": [_spec(M=torch.diag(torch.tensor([1.0, -0.2])))]}
    )

    results = rules.validate_iphs_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "positive semidefinite" in results[0].message


def test_iphs_rule_propagates_strict_floor_and_fails_closed_without_runtime_effort():
    spec = _spec(
        M=torch.zeros((2, 2)),
        R=torch.eye(2),
        formulation="dissipative_ph",
        energy_gradient=torch.tensor([1.0, 0.0]),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
    )
    context = ValidationContext(
        metadata={
            "iphs_specs": [spec],
            "required_level": 1,
            "strict": True,
        }
    )

    results = rules.validate_iphs_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "explicit runtime energy_effort" in results[0].message
    assert "process runtime linkage remains [N]" in results[0].message


def test_iphs_rule_level5_rejects_text_only_constant_jacobi_claim():
    spec = IPHSStructureSpec(
        name="generic_mode",
        state_vars=("x", "s"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        M=torch.diag(torch.tensor([0.0, 1.0])),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.tensor([1.0, 0.0]),
        entropy_gradient=torch.tensor([0.0, 0.0]),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="generic",
        reversible_operator_state_dependence="constant", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )
    context = ValidationContext(
        metadata={"iphs_specs": [spec], "required_level": 5}
    )

    results = rules.validate_iphs_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "Level 5 executable Jacobi identity verification remains [N]" in (
        results[0].message
    )


def test_iphs_rule_strict_mode_rejects_empty_payload():
    context = ValidationContext(metadata={"strict": True})

    results = rules.validate_iphs_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert results[0].passed is False
    assert "non-empty" in results[0].message


def test_iphs_rule_reports_contract_while_level4_runtime_is_unbound():
    spec = _spec(
        M=torch.zeros((2, 2)),
        R=torch.eye(2),
        formulation="dissipative_ph",
        energy_gradient=torch.tensor([1.0, 0.0]),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
    )

    result = rules.validate_iphs_spec(spec, required_level=4)

    assert result.passed is False
    assert result.level == ValidationLevel.ERROR
    assert "process runtime linkage remains [N]" in result.message
    assert result.details[0].expected == (
        "dPH: J + J^T = 0, R = R^T >= 0, dH/dt <= 0; "
        "GENERIC: L + L^T = 0, M = M^T >= 0, degeneracy and Jacobi evidence"
    )


@pytest.mark.parametrize("declaration", [
    {"iphs_level": level} for level in range(4)
] + [{"required_level": 0}, {"iphs_level": 3, "required_level": 0}])
def test_descriptive_level_does_not_request_structural_runtime_floor(declaration):
    result, = rules.validate_iphs_structure(
        ValidationContext(metadata={"iphs_specs": [_spec()], **declaration})
    )

    assert result.passed is True
    assert result.metrics["required_level"] == 0
    assert "runtime_verified_level" not in result.metrics


@pytest.mark.parametrize("level", [1, 2, 3])
def test_local_structural_check_does_not_certify_process_runtime(level):
    result, = rules.validate_iphs_structure(
        ValidationContext(metadata={"iphs_specs": [_spec()], "required_level": level})
    )

    assert result.passed is True
    assert result.scope == "iphs_structure"
    assert result.metrics["required_level"] == level
    assert "runtime_verified_level" not in result.metrics


@pytest.mark.parametrize("declaration", [
    {"required_level": level} for level in range(1, 6)
] + [{"iphs_level": 4}, {"iphs_level": 5}, {"strict": True, "required_level": 0}])
def test_requested_structure_requires_nonempty_payload(declaration):
    result, = rules.validate_iphs_structure(ValidationContext(metadata=declaration))

    assert result.passed is False
    assert "non-empty" in result.message


@pytest.mark.parametrize("key", ["required_level", "iphs_level"])
@pytest.mark.parametrize("value", [True, 1.0, "3", "level_4", None, -1, 6, [], {}])
def test_invalid_level_is_rejected_even_without_payload(key, value):
    result, = rules.validate_iphs_structure(
        ValidationContext(metadata={key: value})
    )

    assert result.passed is False
    assert key in result.message
    assert "integer" in result.message


@pytest.mark.parametrize("value", [1, "false", None, [], {}])
def test_strict_flag_requires_boolean(value):
    result, = rules.validate_iphs_structure(
        ValidationContext(metadata={"strict": value, "iphs_specs": [_spec()]})
    )

    assert result.passed is False
    assert "strict must be bool" in result.message


def test_structure_wrapper_delegates_monotone_level_resolution(monkeypatch):
    from relic.computation.operators.iphs_certificate import resolve_iphs_level_requirement

    calls = []

    def resolve(declarations, **kwargs):
        calls.append((declarations, kwargs))
        return resolve_iphs_level_requirement(declarations, **kwargs)

    monkeypatch.setattr(rules, "resolve_iphs_level_requirement", resolve)
    metadata = {"iphs_specs": [_spec()], "required_level": 0, "iphs_level": 5}
    result, = rules.validate_iphs_structure(ValidationContext(metadata=metadata))

    assert calls == [({"metadata": metadata}, {"strict_iphs": False})]
    assert result.passed is False


def test_empty_unrequested_structure_keeps_optional_rule_contract():
    assert rules.validate_iphs_structure(ValidationContext(metadata={})) == []


def test_iphs_rule_propagates_spec_defects():
    class _DefectiveSpec:
        name = "defective_iphs"

        def validate(self, *, required_level=None):
            raise KeyError("iphs spec defect")

    with pytest.raises(KeyError, match="iphs spec defect"):
        rules.validate_iphs_spec(_DefectiveSpec())


def test_iphs_rule_reports_a_missing_payload_as_failed():
    result = rules.validate_iphs_spec(None)

    assert result.passed is False
    assert "requires an IPHSStructureSpec payload" in result.message
