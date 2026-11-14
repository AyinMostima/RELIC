"""Tests for IPHS equivalence certificate validation rules."""

import pytest

from relic.validation.framework.models import ValidationContext
from relic.validation.framework.rules.equivalence_certificate import (
    validate_equivalence_certificate_payload,
    validate_equivalence_certificate_structure,
)


def _payload():
    return {
        "process_id": "relic.scientific.pedosphere.decomposition.first_order.FirstOrderDecomposition",
        "module_name": "carbon_reaction",
        "canonical_role": "reaction",
        "iphs_level": 2,
        "iphs_level_evidence": {
            "classification": "IPHS-wrapped reaction certificate",
            "compatible_basis": "Reaction process is expressed through a carbon ledger",
            "runtime_replacement": "forward uses canonical reaction-network runtime",
            "canonical_operator": "N_h r_h reaction-network state advance",
        },
        "theory_document": "docs\\source\\theory\\decomposition.md",
        "theory_document_status": "already_current",
        "legacy_formula": "dC/dt = -k C",
        "literature_formula": "First-order soil organic matter pool loss",
        "iphs_formula": "x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h",
        "legacy_expression": "dC/dt = -k C",
        "iphs_expression": "x_dot = J_h e_E + M_h e_S + N_h r_h",
        "input_variables": ["soil_carbon"],
        "output_variables": ["delta_soil_carbon"],
        "parameter_keys": ["pedosphere.carbon.decay_rate"],
        "composition_ports": ["carbon_exchange"],
        "references": ["Grmela and Ottinger (1997)"],
        "validation_tests": [
            "test\\unit\\validation\\framework\\rules\\equivalence_certificate.test.py"
        ],
        "runtime_replacement_map": {
            "runtime_uses_iphs": False,
            "legacy_removed": False,
        },
        "state_map": {"soil_carbon": "x_h.carbon_pool"},
        "parameter_map": {"k": "pedosphere.carbon.decay_rate"},
        "parameter_management_map": {
            "pedosphere.carbon.decay_rate": {"source": "ParameterManager"}
        },
        "operator_map": {"first_order_loss": "N_h r_h"},
        "state_advancement_map": {
            "potential_basis": "G_C(x)=0.5*x_C.T*H_C*x_C",
            "effort_variable": "mu_C=dG_C/dx_C=H_C*x_C",
            "state_advancer": "x_dot=N_h*r_h(mu_C)",
            "runtime_usage": "forward evaluates the reaction-network state advancer directly",
        },
        "unit_map": {"soil_carbon": "kg C m-2", "delta_soil_carbon": "kg C m-2"},
        "input_output_map": {
            "soil_carbon": "x_h.carbon_pool",
            "delta_soil_carbon": "state_update.carbon_pool",
        },
        "interface_map": {
            "forward_signature": "unchanged",
            "bridge_class": "unchanged",
            "dependency_semantics": "unchanged",
            "state_write_mode": "existing ContractRegistry and StateManager semantics",
        },
        "composition_management_map": {
            "functor_metadata": "existing FunctorMetadata fields",
            "port_spec": "existing PortSpec or PortSpecIPHS assembly view",
            "state_contract": "existing ContractRegistry and StateManager",
            "validation_rule": "existing ValidationRegistry rule",
        },
        "dependency_interconnection_map": {
            "dependency_type": "causal_state",
            "declared_dependency": "not_applicable",
        },
        "overflow_routing_map": {
            "residual_name": "negative_pool_clip_residual",
            "destination": "rejected_step",
        },
        "conservation_map": {"carbon": "pool plus respiration port ledger"},
        "entropy_audit": {"sigma_expression": "A_h.T r_h >= 0"},
        "equivalence_type": "thermodynamic_embedding",
        "baseline_tests": [
            "test\\unit\\validation\\framework\\rules\\equivalence_certificate.test.py"
        ],
    }


def test_equivalence_certificate_rule_accepts_complete_certificate():
    result = validate_equivalence_certificate_payload(_payload())

    assert result.passed is True
    assert result.metrics["parameter_count"] == 1


def test_equivalence_certificate_rule_rejects_missing_parameter_management():
    payload = _payload()
    payload["parameter_management_map"] = {}

    result = validate_equivalence_certificate_payload(payload)

    assert result.passed is False
    assert "incomplete" in result.message


def test_equivalence_certificate_rule_rejects_missing_state_advancement_map():
    payload = _payload()
    payload["state_advancement_map"] = {}

    result = validate_equivalence_certificate_payload(payload)

    assert result.passed is False
    assert "incomplete" in result.message


def test_equivalence_certificate_rule_rejects_missing_theory_document_status():
    payload = _payload()
    payload["theory_document_status"] = "missing"

    result = validate_equivalence_certificate_payload(payload)

    assert result.passed is False
    assert "incomplete" in result.message


def test_equivalence_certificate_structure_reads_context_metadata():
    results = validate_equivalence_certificate_structure(
        ValidationContext(metadata={"equivalence_certificates": [_payload()]})
    )

    assert len(results) == 1
    assert results[0].passed is True


def test_equivalence_certificate_rule_separates_declared_and_runtime_verified_level():
    payload = _payload()
    payload["iphs_level"] = 4
    payload["iphs_level_evidence"].update(
        {
            "structural_contribution": "declared reaction structure",
            "assembly_participation": "declared assembly participation",
            "storage_functional": "H_h=0.5*x.T*x",
            "effort_definition": "e_h=dH_h/dx",
            "dissipation_audit": "e_h.T*M_h*e_h>=0",
            "port_power_audit": "dH/dt=<y,u>",
        }
    )

    result = validate_equivalence_certificate_payload(payload)

    assert result.passed is False
    assert result.metrics["iphs_level"] == 4
    assert result.metrics["runtime_verified_level"] == 0
    assert result.metrics["runtime_evidence_issue_count"] > 0


def test_equivalence_certificate_rule_enforces_caller_required_runtime_level():
    payload = _payload()
    payload["required_level"] = 0

    result = validate_equivalence_certificate_payload(
        payload,
        strict=True,
        required_level=4,
    )

    assert result.passed is False
    assert result.metrics["iphs_level"] == 2
    assert result.metrics["runtime_verified_level"] == 0
    assert "required runtime-verified IPHS level 4" in result.details[0].value


def test_equivalence_certificate_caller_strict_cannot_be_weakened_by_payload():
    payload = _payload()
    payload["strict"] = False
    payload["state_map"] = {}

    result = validate_equivalence_certificate_payload(payload, strict=True)

    assert result.passed is False
    assert "incomplete" in result.message


def test_equivalence_certificate_allows_explicit_descriptive_validation():
    payload = _payload()
    payload["strict"] = False
    payload["state_map"] = {}

    result = validate_equivalence_certificate_payload(payload, strict=False)

    assert result.passed is True


def test_equivalence_certificate_structure_propagates_context_strict():
    payload = _payload()
    payload["strict"] = False
    payload["state_map"] = {}

    results = validate_equivalence_certificate_structure(
        ValidationContext(
            metadata={"strict": True, "equivalence_certificates": [payload]}
        )
    )

    assert len(results) == 1
    assert results[0].passed is False


def test_equivalence_certificate_structure_rejects_empty_strict_payload():
    results = validate_equivalence_certificate_structure(
        ValidationContext(metadata={"strict": True})
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert "certificate" in results[0].message.lower()


@pytest.mark.parametrize("strict_value", ["false", 0])
def test_equivalence_certificate_payload_requires_exact_bool_strict(strict_value):
    result = validate_equivalence_certificate_payload(
        _payload(),
        strict=strict_value,
    )

    assert result.passed is False
    assert "strict must be bool" in result.message


@pytest.mark.parametrize("payload_strict", ["false", 0])
def test_equivalence_certificate_rejects_invalid_payload_strict_type(payload_strict):
    payload = _payload()
    payload["strict"] = payload_strict

    result = validate_equivalence_certificate_payload(payload, strict=False)

    assert result.passed is False
    assert "payload strict must be bool" in result.message


def test_equivalence_certificate_result_exposes_structural_contract_identity():
    result = validate_equivalence_certificate_payload(_payload())

    assert result.passed is True
    identities = [
        detail.metadata.get("contract_identity")
        for detail in result.details
        if isinstance(detail.metadata.get("contract_identity"), dict)
    ]
    assert identities == [
        {
            "process_id": _payload()["process_id"],
            "canonical_role": "reaction",
            "formulation": "unspecified",
            "declared_iphs_level": 2,
            "module_name": "carbon_reaction",
        }
    ]


def test_equivalence_certificate_failure_retains_structural_contract_identity():
    payload = _payload()
    payload["parameter_management_map"] = {}

    result = validate_equivalence_certificate_payload(payload)

    assert result.passed is False
    assert result.details[0].metadata["contract_identity"]["process_id"] == payload[
        "process_id"
    ]


def test_equivalence_wrapper_merges_required_and_iphs_levels_monotonically():
    results = validate_equivalence_certificate_structure(
        ValidationContext(
            metadata={
                "strict": False,
                "required_level": 4,
                "iphs_level": 5,
                "equivalence_certificates": [_payload()],
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].metrics["required_runtime_level"] == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_level", True),
        ("iphs_level", "5"),
    ],
)
def test_equivalence_wrapper_rejects_invalid_level_type(field, value):
    results = validate_equivalence_certificate_structure(
        ValidationContext(
            metadata={
                field: value,
                "equivalence_certificates": [_payload()],
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert f"{field} must be an integer" in results[0].message


@pytest.mark.parametrize(
    "alias",
    [
        "equivalence_certificate",
        "iphs_equivalence_certificates",
        "iphs_equivalence_certificate",
    ],
)
def test_equivalence_wrapper_rejects_noncanonical_alias(alias):
    results = validate_equivalence_certificate_structure(
        ValidationContext(metadata={alias: _payload()})
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert "canonical equivalence_certificates" in results[0].message


def test_equivalence_wrapper_rejects_plural_and_alias_conflict():
    results = validate_equivalence_certificate_structure(
        ValidationContext(
            metadata={
                "equivalence_certificates": [_payload()],
                "equivalence_certificate": _payload(),
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is False


@pytest.mark.parametrize(
    "items",
    [
        42,
        [_payload(), 42],
    ],
)
def test_equivalence_wrapper_rejects_scalar_or_mixed_certificate_items(items):
    results = validate_equivalence_certificate_structure(
        ValidationContext(metadata={"equivalence_certificates": items})
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert "list or tuple of certificate" in results[0].message


def test_equivalence_wrapper_nonstrict_empty_is_not_applicable():
    results = validate_equivalence_certificate_structure(
        ValidationContext(
            metadata={
                "strict": False,
                "equivalence_certificates": [],
            }
        )
    )

    assert results == []


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_equivalence_wrapper_descriptive_levels_do_not_request_runtime(strict, level):
    results = validate_equivalence_certificate_structure(ValidationContext(metadata={
        "strict": strict, "iphs_level": level,
        "equivalence_certificates": [_payload()],
    }))

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].metrics["runtime_verified_level"] == 0
    assert results[0].metrics["required_runtime_level"] == -1


@pytest.mark.parametrize("requirement", [
    {"required_level": level} for level in range(1, 6)
] + [{"iphs_level": 4}, {"iphs_level": 5}])
def test_equivalence_wrapper_runtime_request_cannot_pass_without_payload(requirement):
    results = validate_equivalence_certificate_structure(ValidationContext(metadata={
        "strict": False, "equivalence_certificates": [], **requirement,
    }))

    assert len(results) == 1
    assert results[0].passed is False


@pytest.mark.parametrize("level", [1, 2, 3])
def test_equivalence_wrapper_positive_floor_is_not_satisfied_by_description(level):
    results = validate_equivalence_certificate_structure(ValidationContext(metadata={
        "strict": True, "required_level": level, "iphs_level": 0,
        "equivalence_certificates": [_payload()],
    }))

    assert results[0].passed is False
    assert results[0].metrics["required_runtime_level"] == level
    assert results[0].metrics["runtime_verified_level"] == 0
    assert "required runtime-verified" in str(results[0].details[0].value)


def test_equivalence_wrapper_zero_floor_does_not_require_an_unused_certificate():
    assert validate_equivalence_certificate_structure(ValidationContext(metadata={
        "strict": False, "required_level": 0,
    })) == []


def test_equivalence_certificate_payload_passes_runtime_point_audit_to_the_owner():
    payload = _payload()
    payload["runtime_point_audit"] = {"runtime_verified_level": 5}

    result = validate_equivalence_certificate_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        "runtime_point_audit is accepted only for Level 4 or 5" in str(detail.value)
        for detail in result.details
    )


def test_equivalence_certificate_payload_rejects_non_mapping_runtime_point_audit():
    payload = _payload()
    payload["runtime_point_audit"] = "verified"

    result = validate_equivalence_certificate_payload(payload, strict=True)

    assert result.passed is False
    assert "runtime_point_audit" in result.message
    assert any(detail.variable_name == "runtime_point_audit" for detail in result.details)


def test_equivalence_certificate_rule_propagates_loader_defects(monkeypatch):
    import relic.validation.framework.rules.equivalence_certificate as certificate_rules

    def _loader_defect(payload):
        raise KeyError("certificate loader defect")

    monkeypatch.setattr(certificate_rules, "load_iphs_equivalence_certificate", _loader_defect)

    with pytest.raises(KeyError, match="certificate loader defect"):
        validate_equivalence_certificate_payload(_payload())
