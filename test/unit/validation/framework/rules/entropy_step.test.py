"""Tests for entropy production validation rules."""

from __future__ import annotations

import pytest
import torch

from relic.validation.framework.models import ValidationContext
from relic.validation.framework.rules.entropy_step import (
    validate_entropy_step_payload,
    validate_entropy_step_structure,
)


def _complete_tolerance_spec(key, *, value=1.0e-8):
    return {
        "key": key,
        "value": value,
        "category": "model_parameter",
        "units": "1",
        "dtype": "float64",
        "bounds": (0.0, 1.0e-6),
        "references": ("unit-test",),
        "provenance": {"source": "unit_test"},
    }


def test_entropy_step_accepts_nonnegative_entropy_delta():
    result = validate_entropy_step_payload(
        {
            "name": "closed_damped_mode",
            "entropy_before": 1.0,
            "entropy_after": 1.2,
            "tolerance": 1e-12,
        }
    )

    assert result.passed is True
    assert result.metrics["entropy_delta"] == pytest.approx(0.2)


def test_entropy_step_rejects_negative_entropy_production():
    result = validate_entropy_step_payload(
        {
            "name": "invalid_damped_mode",
            "entropy_production": -1e-4,
            "tolerance": 1e-8,
        }
    )

    assert result.passed is False
    assert "nonnegative" in result.message


@pytest.mark.parametrize("tolerance", [-1.0e-8, True, float("nan"), float("inf")])
def test_entropy_step_rejects_invalid_tolerance_before_comparison(tolerance):
    result = validate_entropy_step_payload(
        {
            "name": "invalid_tolerance",
            "entropy_production": -0.5,
            "tolerance": tolerance,
        }
    )

    assert result.passed is False
    assert "tolerance must be a finite non-negative scalar" in result.message


@pytest.mark.parametrize(
    "entropy_production",
    [True, float("nan"), float("inf"), torch.tensor([0.1, 0.2])],
)
def test_entropy_step_rejects_nonfinite_or_nonscalar_evidence(entropy_production):
    result = validate_entropy_step_payload(
        {
            "name": "invalid_evidence",
            "entropy_production": entropy_production,
            "tolerance": 1.0e-12,
        }
    )

    assert result.passed is False
    assert "entropy_production must be a finite scalar" in result.message


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("entropy_before", float("nan")),
        ("entropy_before", True),
        ("entropy_after", float("inf")),
        ("entropy_after", torch.tensor([1.0, 1.1])),
    ],
)
def test_entropy_step_rejects_invalid_before_after_evidence(field_name, field_value):
    payload = {
        "name": "invalid_state_evidence",
        "entropy_before": 1.0,
        "entropy_after": 1.1,
        "tolerance": 1.0e-12,
    }
    payload[field_name] = field_value

    result = validate_entropy_step_payload(payload)

    assert result.passed is False
    assert f"{field_name} must be a finite scalar" in result.message


def test_strict_entropy_step_requires_parameter_manager_tolerance_key():
    result = validate_entropy_step_payload(
        {
            "name": "missing_tolerance_binding",
            "entropy_production": 0.0,
        },
        strict=True,
        parameter_resolver=lambda key: _complete_tolerance_spec(key),
    )

    assert result.passed is False
    assert "tolerance_parameter_key" in result.message


def test_strict_entropy_step_rejects_unresolved_or_mismatched_parameter():
    unresolved = validate_entropy_step_payload(
        {
            "name": "unresolved_tolerance",
            "entropy_production": 0.0,
            "tolerance_parameter_key": "test.iphs.tolerance",
        },
        strict=True,
        parameter_resolver=lambda key: None,
    )
    mismatched = validate_entropy_step_payload(
        {
            "name": "mismatched_tolerance",
            "entropy_production": 0.0,
            "tolerance_parameter_key": "test.iphs.tolerance",
        },
        strict=True,
        parameter_resolver=lambda key: _complete_tolerance_spec(
            "test.other.tolerance"
        ),
    )

    assert unresolved.passed is False
    assert "complete ParameterManager metadata and runtime value" in unresolved.message
    assert mismatched.passed is False
    assert "resolved parameter key" in mismatched.message


def test_strict_entropy_step_rejects_literal_tolerance_override():
    result = validate_entropy_step_payload(
        {
            "name": "literal_override",
            "entropy_production": -1.0e9,
            "tolerance_parameter_key": "test.iphs.tolerance",
            "tolerance": 1.0e12,
            "strict": False,
        },
        strict=True,
        parameter_resolver=lambda key: _complete_tolerance_spec(key),
    )

    assert result.passed is False
    assert "literal tolerance" in result.message


def test_strict_entropy_step_uses_bounded_resolved_tolerance_value():
    result = validate_entropy_step_payload(
        {
            "name": "resolved_tolerance",
            "entropy_production": -5.0e-9,
            "tolerance_parameter_key": "test.iphs.tolerance",
        },
        strict=True,
        parameter_resolver=lambda key: _complete_tolerance_spec(key),
    )
    out_of_bounds = validate_entropy_step_payload(
        {
            "name": "out_of_bounds_tolerance",
            "entropy_production": 0.0,
            "tolerance_parameter_key": "test.iphs.tolerance",
        },
        strict=True,
        parameter_resolver=lambda key: _complete_tolerance_spec(
            key,
            value=1.0,
        ),
    )

    assert result.passed is True
    assert result.metrics["tolerance"] == pytest.approx(1.0e-8)
    assert out_of_bounds.passed is False
    assert "within finite non-negative bounds" in out_of_bounds.message


def test_entropy_step_structure_reads_context_metadata():
    results = validate_entropy_step_structure(
        ValidationContext(
            metadata={
                "entropy_steps": [
                    {
                        "name": "step_0",
                        "entropy_before": 2.0,
                        "entropy_after": 2.0,
                    }
                ]
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is True


def _raising_resolver(error):
    def _resolver(key):
        raise error

    return _resolver


def test_strict_entropy_step_reports_an_undeclared_tolerance_key():
    result = validate_entropy_step_payload(
        {
            "name": "undeclared_tolerance",
            "entropy_production": 0.0,
            "tolerance_parameter_key": "test.iphs.undeclared_tolerance",
        },
        strict=True,
        parameter_resolver=_raising_resolver(KeyError("test.iphs.undeclared_tolerance")),
    )

    assert result.passed is False
    assert "test.iphs.undeclared_tolerance" in result.message


def test_strict_entropy_step_propagates_resolver_defects():
    with pytest.raises(RuntimeError, match="parameter database offline"):
        validate_entropy_step_payload(
            {
                "name": "resolver_defect",
                "entropy_production": 0.0,
                "tolerance_parameter_key": "test.iphs.tolerance",
            },
            strict=True,
            parameter_resolver=_raising_resolver(RuntimeError("parameter database offline")),
        )
