"""Tests for overflow and residual routing validation."""

import numpy as np
import pytest
import torch

from relic.validation.framework.models import ValidationContext
from relic.validation.framework.rules.overflow_routing import (
    validate_overflow_routing_payload,
    validate_overflow_routing_structure,
)


def _canonical_route(*, amount=1.0, route_type="diagnostic_residual"):
    target = (
        "diagnostics.test_residual"
        if route_type == "diagnostic_residual"
        else "test.route_target"
    )
    route = {
        "name": "test_residual",
        "amount": amount,
        "units": "kg",
        "route_type": route_type,
        "target": target,
    }
    if route_type == "rejected_step":
        route["step_rejected"] = True
    return route


def test_overflow_routing_accepts_diagnostic_residual_route():
    result = validate_overflow_routing_payload(
        {
            "routes": [
                {
                    "name": "carbon_balance_residual",
                    "amount": 0.2,
                    "units": "kg",
                    "route_type": "diagnostic_residual",
                    "target": "diagnostics.carbon_balance_residual",
                }
            ]
        }
    )

    assert result.passed is True
    assert result.metrics["active_route_count"] == 1


def test_overflow_routing_accepts_rejected_step_route():
    result = validate_overflow_routing_payload(
        {
            "routes": [
                {
                    "name": "negative_pool_clip",
                    "amount": 1.0,
                    "units": "kg",
                    "route_type": "rejected_step",
                    "target": "simulator.step_transaction",
                    "step_rejected": True,
                }
            ]
        }
    )

    assert result.passed is True


def test_overflow_routing_rejects_missing_route_type_for_active_residual():
    result = validate_overflow_routing_payload(
        {"routes": [{"name": "unconsumed_flux", "amount": 0.5}]}
    )

    assert result.passed is False
    assert "unrouteable" in result.message


def test_overflow_routing_rejects_diagnostic_target_outside_diagnostics_namespace():
    result = validate_overflow_routing_payload(
        {
            "routes": [
                {
                    "name": "bad_diagnostic",
                    "amount": 0.5,
                    "units": "kg",
                    "route_type": "diagnostic_residual",
                    "target": "carbon_pool",
                }
            ]
        }
    )

    assert result.passed is False
    assert "diagnostics" in str(result.details[0].expected)


def test_overflow_routing_structure_reads_context_metadata():
    results = validate_overflow_routing_structure(
        ValidationContext(
            metadata={
                "overflow_routing": {
                    "routes": [
                        {
                            "name": "runoff_export",
                            "amount": 1.0,
                            "units": "kg",
                            "route_type": "boundary_outflow",
                            "target": "watershed_outlet",
                        }
                    ]
                }
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is True


def test_strict_overflow_routing_rejects_empty_payload():
    result = validate_overflow_routing_payload({}, strict=True)

    assert result.passed is False
    assert any("no_residuals" in detail.expected for detail in result.details)


def test_strict_overflow_routing_rejects_self_certified_no_residuals_claim():
    result = validate_overflow_routing_payload(
        {
            "no_residuals": True,
            "no_residuals_evidence": {
                "runtime_owner": (
                    "relic.validation.framework.rules.overflow_routing."
                    "validate_overflow_routing_payload"
                ),
                "audit_key": "active_residual_count",
                "expected_value": 0.0,
                "validation_test": (
                    "test\\unit\\validation\\framework\\rules\\"
                    "overflow_routing.test.py"
                ),
            },
        },
        strict=True,
    )

    assert result.passed is False
    assert result.metrics["no_residuals_declared"] == 1
    assert any(
        "[N]" in detail.expected
        and "executable runtime residual audit" in detail.expected
        for detail in result.details
    )


def test_strict_overflow_routing_rejects_prose_only_no_residuals_evidence():
    result = validate_overflow_routing_payload(
        {
            "no_residuals": True,
            "no_residuals_evidence": "trust me",
        },
        strict=True,
    )

    assert result.passed is False


def test_overflow_routing_rejects_non_numeric_residual_amount():
    result = validate_overflow_routing_payload(
        {
            "routes": [
                {
                    "name": "invalid_residual",
                    "amount": "not-a-number",
                    "units": "kg",
                    "route_type": "diagnostic_residual",
                    "target": "diagnostics.invalid_residual",
                }
            ]
        }
    )

    assert result.passed is False
    assert any("numeric" in detail.expected for detail in result.details)


def test_overflow_routing_rejects_nonfinite_routed_amount():
    result = validate_overflow_routing_payload(
        {
            "routes": [
                {
                    "name": "invalid_routed_residual",
                    "amount": 1.0,
                    "routed_amount": float("nan"),
                    "units": "kg",
                    "route_type": "diagnostic_residual",
                    "target": "diagnostics.invalid_routed_residual",
                }
            ]
        }
    )

    assert result.passed is False
    assert any("finite" in detail.expected for detail in result.details)


def test_overflow_routing_rejects_tensor_route_that_only_matches_in_aggregate():
    result = validate_overflow_routing_payload(
        {
            "routes": [
                {
                    "name": "shape_preserving_residual",
                    "amount": [1.0, 1.0],
                    "routed_amount": [2.0, 0.0],
                    "units": "kg",
                    "route_type": "diagnostic_residual",
                    "target": "diagnostics.shape_preserving_residual",
                }
            ]
        }
    )

    assert result.passed is False


def test_strict_overflow_routing_requires_residual_units():
    result = validate_overflow_routing_payload(
        {
            "routes": [
                {
                    "name": "carbon_balance_residual",
                    "amount": 0.2,
                    "route_type": "diagnostic_residual",
                    "target": "diagnostics.carbon_balance_residual",
                }
            ]
        },
        strict=True,
    )

    assert result.passed is False
    assert any("units" in detail.expected for detail in result.details)


def test_strict_overflow_routing_rejects_unknown_residual_units():
    result = validate_overflow_routing_payload(
        {
            "routes": [
                {
                    "name": "carbon_balance_residual",
                    "amount": 0.2,
                    "units": "banana",
                    "route_type": "diagnostic_residual",
                    "target": "diagnostics.carbon_balance_residual",
                }
            ]
        },
        strict=True,
    )

    assert result.passed is False
    assert any("parseable residual units" in detail.expected for detail in result.details)


def test_overflow_routing_structure_propagates_strict_context():
    results = validate_overflow_routing_structure(
        ValidationContext(metadata={"strict": True, "overflow_routing": {}})
    )

    assert len(results) == 1
    assert results[0].passed is False


def test_strict_zero_route_is_structurally_routed_but_not_active():
    result = validate_overflow_routing_payload(
        {"routes": [_canonical_route(amount=0.0)]},
        strict=True,
    )

    assert result.passed is True
    assert result.metrics["route_count"] == 1
    assert result.metrics["active_route_count"] == 0
    assert result.metrics["routed_structural_count"] == 1
    assert result.metrics["invalid_route_count"] == 0


@pytest.mark.parametrize("missing_field", ["route_type", "target"])
def test_strict_zero_route_still_requires_complete_structure(missing_field):
    route = _canonical_route(amount=0.0)
    route.pop(missing_field)

    result = validate_overflow_routing_payload(
        {"routes": [route]},
        strict=True,
    )

    assert result.passed is False
    assert result.metrics["invalid_route_count"] == 1


def test_overflow_routing_rejects_mixed_non_mapping_route_item():
    result = validate_overflow_routing_payload(
        {"routes": [_canonical_route(), 42]},
        strict=True,
    )

    assert result.passed is False
    assert result.metrics["invalid_route_count"] == 1
    assert any("route item is a mapping" in detail.expected for detail in result.details)


def test_overflow_routing_rejects_scalar_routes_payload():
    result = validate_overflow_routing_payload(
        {"routes": 42},
        strict=True,
    )

    assert result.passed is False
    assert result.metrics["invalid_route_count"] == 1


@pytest.mark.parametrize(
    "alias",
    ["overflow_routes", "residual_routes", "residuals"],
)
def test_overflow_routing_rejects_top_level_route_alias(alias):
    result = validate_overflow_routing_payload(
        {alias: [_canonical_route()]},
        strict=False,
    )

    assert result.passed is False
    assert any("canonical routes field" in detail.expected for detail in result.details)


def test_overflow_routing_rejects_route_field_aliases():
    route = {
        "name": "legacy_residual",
        "residual": 1.0,
        "units": "kg",
        "route": "diagnostic_residual",
        "target_variable": "diagnostics.legacy_residual",
    }

    result = validate_overflow_routing_payload({"routes": [route]})

    assert result.passed is False
    assert any("canonical route fields" in detail.expected for detail in result.details)


def test_overflow_routing_rejects_canonical_plus_alias_even_when_equal():
    route = _canonical_route()
    route["residual"] = route["amount"]

    result = validate_overflow_routing_payload({"routes": [route]})

    assert result.passed is False
    assert any("canonical route fields" in detail.expected for detail in result.details)


def test_overflow_routing_rejects_duplicate_route_names():
    result = validate_overflow_routing_payload(
        {"routes": [_canonical_route(), _canonical_route(amount=2.0)]},
        strict=True,
    )

    assert result.passed is False
    assert any("unique route name" in detail.expected for detail in result.details)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amount", torch.tensor([1.0 + 2.0j])),
        ("routed_amount", np.array([1.0 + 2.0j])),
        ("amount", [[True]]),
        ("amount", torch.empty(1, device="meta")),
        (
            "amount",
            torch.sparse_coo_tensor(
                torch.tensor([[0]]),
                torch.tensor([1.0]),
                (1,),
                check_invariants=False,
            ),
        ),
    ],
)
def test_overflow_routing_rejects_bool_or_complex_before_real_cast(field, value):
    route = _canonical_route(amount=[1.0])
    route["routed_amount"] = [1.0]
    route[field] = value

    result = validate_overflow_routing_payload({"routes": [route]}, strict=True)

    assert result.passed is False
    assert any("real numeric" in detail.expected for detail in result.details)


def test_overflow_routing_requires_exact_bool_flags():
    route = _canonical_route(amount=0.0)
    route["required"] = "false"

    result = validate_overflow_routing_payload({"routes": [route]}, strict=True)

    assert result.passed is False
    assert any("required is bool" in detail.expected for detail in result.details)


def test_rejected_step_requires_exact_true_flag():
    route = _canonical_route(route_type="rejected_step")
    route["step_rejected"] = "false"

    result = validate_overflow_routing_payload({"routes": [route]}, strict=True)

    assert result.passed is False
    assert any("step_rejected is bool" in detail.expected for detail in result.details)


def test_non_rejected_route_cannot_claim_step_rejected():
    route = _canonical_route()
    route["step_rejected"] = True

    result = validate_overflow_routing_payload({"routes": [route]}, strict=True)

    assert result.passed is False
    assert any("only rejected_step" in detail.expected for detail in result.details)


def test_payload_cannot_self_certify_required_residual_coverage():
    result = validate_overflow_routing_payload(
        {
            "routes": [_canonical_route()],
            "required_residual_names": ["test_residual"],
        },
        strict=True,
    )

    assert result.passed is False
    assert any("runtime residual coverage remains [N]" in detail.expected for detail in result.details)


def test_overflow_wrapper_returns_explicit_failure_for_scalar_payload():
    results = validate_overflow_routing_structure(
        ValidationContext(metadata={"strict": True, "overflow_routing": 42})
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert any("route item is a mapping" in detail.expected for detail in results[0].details)
