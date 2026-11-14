"""Tests for state-domain validation rules."""

import warnings

import numpy as np
import pytest
import torch

from relic.validation.framework.models import ValidationContext, ValidationLevel
from relic.validation.framework.registry import get_global_registry
from relic.validation.framework.rules import state_domain as rules


def _quantized_state_value():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.quantize_per_tensor(
            torch.tensor([1.0], dtype=torch.float32),
            scale=0.1,
            zero_point=0,
            dtype=torch.quint8,
        )


def test_state_domain_rule_registers_with_structural_tags():
    registry = get_global_registry()

    assert "state_domain_structure" in registry.list_rules(tags=["state_domain"])


def test_state_domain_rule_passes_finite_bounded_state():
    context = ValidationContext(
        state={
            "soil_water_storage": torch.tensor([1.0, 2.0]),
            "soil_saturation": torch.tensor([0.2, 0.8]),
        },
        metadata={
            "state_domain_constraints": {
                "soil_water_storage": {"nonnegative": True},
                "soil_saturation": {"bounds": (0.0, 1.0)},
            }
        },
    )

    results = rules.validate_state_domain_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.INFO
    assert results[0].passed is True


def test_state_domain_rule_fails_negative_storage():
    context = ValidationContext(
        state={"soil_water_storage": torch.tensor([1.0, -0.1])},
        metadata={"state_domain_constraints": {"soil_water_storage": {"nonnegative": True}}},
    )

    results = rules.validate_state_domain_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert "nonnegative" in results[0].message


def test_state_domain_rule_fails_nonfinite_value():
    context = ValidationContext(state={"soil_water_storage": torch.tensor([float("nan")])})

    results = rules.validate_state_domain_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert "finite" in results[0].message


def test_strict_state_domain_rejects_empty_state():

    result = rules.validate_state_domain_payload({}, {}, strict=True)

    assert result.passed is False
    assert "non-empty" in result.message


def test_strict_state_domain_requires_constraint_coverage():
    state = {
        "soil_water_storage": torch.tensor([1.0]),
        "soil_temperature": torch.tensor([280.0]),
    }
    constraints = {
        "soil_water_storage": {"nonnegative": True, "units": "kg m-2"},
    }

    result = rules.validate_state_domain_payload(
        state,
        constraints,
        strict=True,
    )

    assert result.passed is False
    assert "soil_temperature" in result.message


def test_strict_state_domain_requires_graph_bound_state_variables():

    result = rules.validate_state_domain_payload(
        {"decoy": torch.tensor([1.0])},
        {"decoy": {"domain": "real", "units": "1"}},
        strict=True,
        required_state_vars=["soil_temperature"],
    )

    assert result.passed is False
    assert "soil_temperature" in result.message


def test_strict_state_domain_rejects_empty_required_state_contract():

    result = rules.validate_state_domain_payload(
        {"soil_temperature": torch.tensor([280.0])},
        {"soil_temperature": {"positive": True, "units": "K"}},
        strict=True,
        required_state_vars=[],
    )

    assert result.passed is False
    assert "required_state_vars" in result.message


def test_strict_state_domain_rejects_noncanonical_required_state_identity():

    result = rules.validate_state_domain_payload(
        {"soil_temperature": torch.tensor([280.0])},
        {"soil_temperature": {"positive": True, "units": "K"}},
        strict=True,
        required_state_vars=[" soil_temperature"],
    )

    assert result.passed is False
    assert "exact non-empty strings" in result.message


def test_strict_state_domain_rejects_duplicate_required_state_identity():

    result = rules.validate_state_domain_payload(
        {"soil_temperature": torch.tensor([280.0])},
        {"soil_temperature": {"positive": True, "units": "K"}},
        strict=True,
        required_state_vars=["soil_temperature", "soil_temperature"],
    )

    assert result.passed is False
    assert "unique" in result.message


def test_strict_state_domain_accepts_explicit_real_domain():
    state = {
        "soil_water_storage": torch.tensor([1.0]),
        "signed_flux": torch.tensor([-0.1, 0.2]),
    }
    constraints = {
        "soil_water_storage": {"nonnegative": True, "units": "kg m-2"},
        "signed_flux": {"domain": "real", "units": "kg m-2 s-1"},
    }

    result = rules.validate_state_domain_payload(
        state,
        constraints,
        strict=True,
    )

    assert result.passed is True
    assert result.metrics["covered_count"] == 2


def test_strict_state_domain_rejects_malformed_bounds():

    result = rules.validate_state_domain_payload(
        {"soil_saturation": torch.tensor([0.5])},
        {"soil_saturation": {"bounds": (1.0, 0.0), "units": "1"}},
        strict=True,
    )

    assert result.passed is False
    assert "ordered" in result.message


def test_strict_state_domain_rejects_empty_state_tensor():

    result = rules.validate_state_domain_payload(
        {"soil_water_storage": torch.tensor([])},
        {
            "soil_water_storage": {
                "bounds": (0.0, None),
                "units": "kg m-2",
            }
        },
        strict=True,
    )

    assert result.passed is False
    assert "non-empty" in result.message


def test_strict_state_domain_rejects_complex_value_for_real_domain():

    result = rules.validate_state_domain_payload(
        {"signed_flux": torch.tensor([1.0 + 2.0j])},
        {"signed_flux": {"domain": "real", "units": "kg m-2 s-1"}},
        strict=True,
    )

    assert result.passed is False
    assert "real-valued" in result.message


def test_strict_state_domain_rejects_vacuous_bounds():

    result = rules.validate_state_domain_payload(
        {"soil_saturation": torch.tensor([0.5])},
        {"soil_saturation": {"bounds": (None, None), "units": "1"}},
        strict=True,
    )

    assert result.passed is False
    assert "at least one" in result.message


def test_strict_state_domain_rejects_unparseable_units():

    result = rules.validate_state_domain_payload(
        {"soil_saturation": torch.tensor([0.5])},
        {"soil_saturation": {"bounds": (0.0, 1.0), "units": "banana"}},
        strict=True,
    )

    assert result.passed is False
    assert "parseable" in result.message


def test_state_domain_rule_propagates_strict_context():
    context = ValidationContext(state={}, metadata={"strict": True})

    results = rules.validate_state_domain_structure(context)

    assert len(results) == 1
    assert results[0].passed is False


def test_state_domain_rule_propagates_required_state_contract():
    context = ValidationContext(
        state={"decoy": torch.tensor([1.0])},
        metadata={
            "strict": True,
            "required_state_vars": ["soil_temperature"],
            "state_domain_constraints": {
                "decoy": {"domain": "real", "units": "1"},
            },
        },
    )

    results = rules.validate_state_domain_structure(context)

    assert len(results) == 1
    assert results[0].passed is False
    assert "soil_temperature" in results[0].message


@pytest.mark.parametrize("invalid_key", [1, "", " state"])
def test_state_domain_rejects_noncanonical_state_identity(invalid_key):

    result = rules.validate_state_domain_payload(
        {invalid_key: torch.tensor([1.0])},
        {invalid_key: {"domain": "real", "units": "1"}},
        strict=True,
    )

    assert result.passed is False
    assert "exact non-empty strings" in result.message


@pytest.mark.parametrize(
    "value",
    [
        True,
        [[True]],
        1.0 + 2.0j,
        np.array([1.0 + 2.0j]),
        torch.empty(1, device="meta"),
        torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([1.0]),
            (1,),
            check_invariants=False,
        ),
        _quantized_state_value(),
    ],
)
def test_state_domain_rejects_non_real_dense_state_before_cast(value):

    result = rules.validate_state_domain_payload(
        {"state_value": value},
        {"state_value": {"domain": "real", "units": "1"}},
        strict=True,
    )

    assert result.passed is False
    assert "real-valued dense tensor" in result.message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nonnegative", "false"),
        ("positive", 1),
    ],
)
def test_state_domain_requires_exact_bool_constraint_flags(field, value):
    rule = {"domain": "real", "units": "1", field: value}

    result = rules.validate_state_domain_payload(
        {"state_value": torch.tensor([1.0])},
        {"state_value": rule},
        strict=True,
    )

    assert result.passed is False
    assert f"{field} must be bool" in result.message


def test_state_domain_returns_failure_for_scalar_state():

    result = rules.validate_state_domain_payload(42, {}, strict=True)

    assert result.passed is False
    assert "state must be a Mapping" in result.message


def test_state_domain_returns_failure_for_scalar_constraints():

    result = rules.validate_state_domain_payload(
        {"state_value": torch.tensor([1.0])},
        42,
        strict=True,
    )

    assert result.passed is False
    assert "constraints must be a Mapping" in result.message


@pytest.mark.parametrize("field", ["minimum", "shape", "support"])
def test_strict_state_domain_rejects_unknown_constraint_schema(field):
    rule = {"domain": "real", "units": "1", field: "self_certified"}

    result = rules.validate_state_domain_payload(
        {"state_value": torch.tensor([1.0])},
        {"state_value": rule},
        strict=True,
    )

    assert result.passed is False
    assert "canonical constraint fields" in result.message


def test_required_state_subset_does_not_claim_full_prognostic_coverage():
    state = {
        "required_state": torch.tensor([1.0]),
        "extra_assertion_state": torch.tensor([2.0]),
    }
    constraints = {
        name: {"domain": "real", "units": "1"}
        for name in state
    }

    result = rules.validate_state_domain_payload(
        state,
        constraints,
        strict=True,
        required_state_vars=["required_state"],
    )

    assert result.passed is True
    assert result.metrics["required_count"] == 1
    assert result.metrics["extra_count"] == 1


def test_state_domain_rejects_static_positive_bounds_contradiction():

    result = rules.validate_state_domain_payload(
        {"temperature": torch.tensor([-1.0])},
        {
            "temperature": {
                "positive": True,
                "bounds": (-2.0, 0.0),
                "units": "K",
            }
        },
        strict=True,
    )

    assert result.passed is False
    assert "positive constraint conflicts with upper bound" in result.message


def test_state_domain_wrapper_preserves_scalar_constraint_failure():
    context = ValidationContext(
        state={"state_value": torch.tensor([1.0])},
        metadata={"strict": True, "state_domain_constraints": 42},
    )

    results = rules.validate_state_domain_structure(context)

    assert len(results) == 1
    assert results[0].passed is False
    assert "constraints must be a Mapping" in results[0].message


def test_state_domain_wrapper_string_false_does_not_enable_strict():
    context = ValidationContext(
        state={"state_value": torch.tensor([1.0])},
        metadata={"strict": "false", "state_domain_constraints": {}},
    )

    results = rules.validate_state_domain_structure(context)

    assert len(results) == 1
    assert results[0].passed is True
