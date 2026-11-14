"""Tests for reaction-network structural validation."""

import math

import numpy as np
import pytest
import torch

from relic.validation.framework.models import ValidationContext
from relic.validation.framework.rules.reaction_network import (
    validate_reaction_network_payload,
    validate_reaction_network_structure,
)


def _strict_closed_payload():
    return {
        "name": "strict_closed_network",
        "stoichiometric_N": [[-1.0, 1.0], [1.0, -1.0]],
        "conservation_matrix": [[1.0, 1.0]],
        "rates": [-0.25, 0.1],
        "affinities": [-2.0, 1.0],
        "species": ["A", "B"],
        "reactions": ["A_to_B", "B_to_A"],
        "conservation_basis": ["total_amount"],
        "rate_reaction_order": ["A_to_B", "B_to_A"],
        "affinity_reaction_order": ["A_to_B", "B_to_A"],
        "affinity_sign_convention": "affinity_times_rate_nonnegative",
        "state_domain": "test_reactor",
        "reaction_domain": "test_reactor",
        "conservation_domain": "test_reactor",
        "species_units": ["mol", "mol"],
        "rate_units": "mol s-1",
        "affinity_units": "J mol-1",
        "affinity_rate_power_units": "J s-1",
    }


def _strict_open_payload(route="boundary_export"):
    payload = _strict_closed_payload()
    payload.update(
        {
            "open_system_route": route,
            "open_system_exchange": [-0.2, 0.0],
            "open_system_ledger_rate": [-0.2],
            "open_system_exchange_units": "mol s-1",
            "open_system_ledger_rate_units": "mol s-1",
            "conservation_ledger_profile": "dimensionless_homogeneous",
            "open_exchange_domain": "test_reactor",
            "open_ledger_domain": "test_reactor",
            "open_exchange_species_order": ["A", "B"],
            "open_ledger_basis": ["total_amount"],
            "open_system_route_binding": {
                "route": route,
                "owner": "test.reaction_route_owner",
                "target": "test.external_reservoir",
                "exchange_variable": "q_open",
                "ledger_variable": "total_amount_export_rate",
            },
        }
    )
    return payload


def test_reaction_network_accepts_element_conserving_stoichiometry():
    result = validate_reaction_network_payload(
        {
            "name": "carbon_conversion",
            "stoichiometric_N": [[-1.0], [1.0]],
            "conservation_matrix": [[1.0, 1.0]],
            "rates": [0.2],
            "affinities": [1.5],
        }
    )

    assert result.passed is True
    assert result.metrics["stoichiometric_conservation_residual"] == 0.0


def test_reaction_network_rejects_element_loss():
    result = validate_reaction_network_payload(
        {
            "name": "carbon_loss",
            "stoichiometric_N": [[-1.0], [0.0]],
            "conservation_matrix": [[1.0, 1.0]],
        }
    )

    assert result.passed is False
    assert any(detail.expected == "C_h N_h = 0" for detail in result.details)


def test_reaction_network_rejects_negative_affinity_direction():
    result = validate_reaction_network_payload(
        {
            "name": "bad_affinity",
            "stoichiometric_N": [[-1.0], [1.0]],
            "conservation_matrix": [[1.0, 1.0]],
            "rates": [0.2],
            "affinities": [-1.0],
        }
    )

    assert result.passed is False
    assert any(detail.expected == "A^T r >= 0" for detail in result.details)


def test_reaction_network_structure_reads_context_metadata():
    results = validate_reaction_network_structure(
        ValidationContext(
            metadata={
                "reaction_network": {
                    "name": "closed_network",
                    "stoichiometric_N": [[-1.0], [1.0]],
                    "conservation_matrix": [[1.0, 1.0]],
                }
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is True


def test_reaction_network_rejects_mismatched_subsystem_matrix_shapes():
    result = validate_reaction_network_payload(
        {
            "name": "bad_subsystems",
            "stoichiometric_N": [[-1.0], [1.0]],
            "conservation_matrix": [[1.0, 1.0]],
            "subsystem_stoichiometric_matrices": [
                [[-1.0], [1.0]],
                [[1.0, -1.0]],
            ],
        }
    )

    assert result.passed is False
    assert any("subsystem" in str(detail.value) for detail in result.details)


def test_strict_reaction_network_rejects_stoichiometry_only_payload():
    result = validate_reaction_network_payload(
        {"stoichiometric_N": [[1.0]]},
        strict=True,
    )

    assert result.passed is False
    expected = {detail.expected for detail in result.details}
    assert "conservation_matrix C_h for every strict internal ledger" in expected
    assert "both rates r and affinities A" in expected


def test_strict_reaction_network_rejects_zero_stoichiometric_reaction_column():
    payload = _strict_closed_payload()
    payload["stoichiometric_N"] = [[-1.0, 0.0], [1.0, 0.0]]

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        detail.expected == "every reaction column changes at least one species"
        for detail in result.details
    )


def test_strict_reaction_network_rejects_non_numeric_optional_charge_vector():
    payload = _strict_closed_payload()
    payload["charge_vector"] = "not-numeric"

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        detail.expected == "numeric charge_vector"
        for detail in result.details
    )


def test_strict_reaction_network_rejects_malformed_subsystem_declaration():
    payload = _strict_closed_payload()
    payload["subsystem_stoichiometric_matrices"] = "not-a-sequence"

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        "non-empty sequence" in detail.expected
        for detail in result.details
    )


def test_strict_reaction_network_requires_route_for_declared_open_ledger():
    payload = _strict_closed_payload()
    payload.update(
        {
            "open_system_exchange": [-0.2, 0.0],
            "open_system_ledger_rate": [-0.2],
        }
    )

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        detail.expected == "open_system_route for every declared open-system ledger"
        for detail in result.details
    )


def test_strict_reaction_network_returns_failure_for_empty_reaction_axis():
    payload = _strict_closed_payload()
    payload.update(
        {
            "stoichiometric_N": [[], []],
            "rates": [],
            "affinities": [],
        }
    )

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        detail.expected == "non-empty species and reaction axes"
        for detail in result.details
    )


def test_strict_reaction_network_accepts_signed_reversible_net_rates():
    result = validate_reaction_network_payload(
        _strict_closed_payload(),
        strict=True,
    )

    assert result.passed is True
    assert result.metrics["affinity_rate_power_native"] == 0.6


def test_strict_reaction_network_rejects_missing_affinity_evidence():
    payload = _strict_closed_payload()
    payload.pop("affinities")

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        detail.expected == "both rates r and affinities A"
        for detail in result.details
    )


def test_strict_open_reaction_requires_numerical_ledger():
    payload = _strict_closed_payload()
    payload["open_system_route"] = "port"

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("open_system_exchange" in detail.expected for detail in result.details)


def test_strict_open_reaction_accepts_reconciled_ledger():
    payload = _strict_open_payload("boundary_export")

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is True
    assert result.metrics["open_system_ledger_residual"] == 0.0


def test_strict_open_reaction_rejects_inconsistent_ledger():
    payload = _strict_open_payload("managed_export")
    payload["open_system_ledger_rate"] = [0.0]

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        detail.expected == "C_h q_open = declared open-system ledger rate"
        for detail in result.details
    )


def test_reaction_network_structure_propagates_strict_context():
    results = validate_reaction_network_structure(
        ValidationContext(
            metadata={
                "strict": True,
                "reaction_network": {"stoichiometric_N": [[1.0]]},
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is False


def test_reaction_network_structure_rejects_non_mapping_strict_entries():
    results = validate_reaction_network_structure(
        ValidationContext(
            metadata={
                "strict": True,
                "reaction_networks": [123],
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is False


def test_strict_reaction_network_rejects_zero_conservation_rows():
    payload = _strict_closed_payload()
    payload["stoichiometric_N"] = [[1.0]]
    payload["conservation_matrix"] = [[0.0]]
    payload["rates"] = [1.0]
    payload["affinities"] = [1.0]
    payload["species_units"] = ["mol"]

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("nonzero" in detail.expected for detail in result.details)


def test_strict_reaction_network_rejects_nonfinite_affinity_rate_power():
    payload = _strict_closed_payload()
    payload["rates"] = [1.0e308, 1.0e308]
    payload["affinities"] = [1.0e308, 1.0e308]

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert not math.isfinite(
        result.metrics.get("affinity_rate_power_native", float("inf"))
    )
    assert any("finite affinity-rate power" in detail.expected for detail in result.details)


@pytest.mark.parametrize(
    ("field", "value", "expected_fragment"),
    [
        ("charge_vector", [float("nan"), float("nan")], "finite charge"),
        (
            "subsystem_stoichiometric_matrices",
            [[[float("nan"), 0.0], [0.0, float("nan")]]],
            "subsystem",
        ),
    ],
)
def test_reaction_network_rejects_nonfinite_derived_ledgers(
    field,
    value,
    expected_fragment,
):
    payload = _strict_closed_payload()
    payload[field] = value

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        expected_fragment in detail.expected or expected_fragment in str(detail.value)
        for detail in result.details
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("species_units", ["banana", "mol"]),
        ("rate_units", "banana"),
        ("affinity_units", "banana"),
        ("affinity_rate_power_units", "banana"),
    ],
)
def test_strict_reaction_network_rejects_unparseable_units(field, value):
    payload = _strict_closed_payload()
    payload[field] = value

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("parseable" in detail.expected for detail in result.details)


def test_strict_reaction_network_rejects_rate_dimension_without_inverse_time():
    payload = _strict_closed_payload()
    payload["rate_units"] = "mol"
    payload["affinity_rate_power_units"] = "J"

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("species unit per time" in detail.expected for detail in result.details)


def test_strict_reaction_network_rejects_entropy_power_unit_mismatch():
    payload = _strict_closed_payload()
    payload["affinity_rate_power_units"] = "kg s-1"

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("rate_units * affinity_units" in detail.expected for detail in result.details)


def test_strict_reaction_network_rejects_non_power_affinity_pair():
    payload = _strict_closed_payload()
    payload.update(
        {
            "species_units": ["kg", "kg"],
            "rate_units": "kg s-1",
            "affinity_units": "m s-1",
            "affinity_rate_power_units": "N",
        }
    )

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("energy per mass or amount" in detail.expected for detail in result.details)


def test_strict_reaction_network_rejects_retired_entropy_rate_field():
    payload = _strict_closed_payload()
    payload["entropy_production_units"] = payload.pop("affinity_rate_power_units")

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        detail.expected == "non-empty affinity_rate_power_units"
        for detail in result.details
    )


def test_strict_open_reaction_rejects_exchange_unit_mismatch():
    payload = _strict_open_payload("port")
    payload.update(
        {
            "open_system_exchange": [-0.1, 0.0],
            "open_system_ledger_rate": [-0.1],
            "open_system_exchange_units": "kg s-1",
            "open_system_ledger_rate_units": "kg s-1",
        }
    )

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        "open_system_exchange_units compatible with rate_units" in detail.expected
        for detail in result.details
    )


def test_strict_open_reaction_rejects_unbound_ledger_unit_profile():
    payload = _strict_open_payload("port")
    payload.update(
        {
            "open_system_exchange": [-0.1, 0.0],
            "open_system_ledger_rate": [-0.1],
        }
    )
    payload.pop("conservation_ledger_profile")
    payload.pop("open_system_ledger_rate_units")

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("dimensionless_homogeneous" in detail.expected for detail in result.details)


@pytest.mark.parametrize(
    "species_units",
    [
        ["kg", "g"],
        ["mol", "mmol"],
    ],
)
def test_strict_reaction_network_rejects_mixed_species_unit_scales(species_units):
    payload = _strict_closed_payload()
    payload["species_units"] = species_units
    payload["rate_units"] = f"{species_units[0]} s-1"
    payload["affinity_units"] = f"J {species_units[0]}-1"

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("identical canonical species units" in detail.expected for detail in result.details)


def test_strict_reaction_network_rejects_nonsecond_rate_basis():
    payload = _strict_closed_payload()
    payload["rate_units"] = "mol day-1"
    payload["affinity_rate_power_units"] = "J day-1"

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("canonical species_units s-1" in detail.expected for detail in result.details)


def test_strict_reaction_network_accepts_support_aware_power_density_units():
    payload = _strict_closed_payload()
    payload.update(
        {
            "species_units": ["mol m-2", "mol m-2"],
            "rate_units": "mol m-2 s-1",
            "affinity_units": "J mol-1",
            "affinity_rate_power_units": "J m-2 s-1",
        }
    )

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "stoichiometric_N",
            torch.tensor(
                [[-1.0 + 1.0j, 1.0], [1.0 + 1.0j, -1.0]],
                dtype=torch.complex128,
            ),
        ),
        (
            "conservation_matrix",
            np.array([[1.0 + 1.0j, 1.0 - 1.0j]], dtype=np.complex128),
        ),
        ("rates", [[-0.25 + 1.0j], [0.1 + 0.0j]]),
    ],
)
def test_strict_reaction_network_rejects_complex_before_real_cast(field, value):
    payload = _strict_closed_payload()
    payload[field] = value

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("real" in detail.expected for detail in result.details)


def test_strict_reaction_network_rejects_bool_affinity_vector():
    payload = _strict_closed_payload()
    payload["rates"] = [1.0, 1.0]
    payload["affinities"] = torch.tensor([True, True])

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("real" in detail.expected for detail in result.details)


@pytest.mark.parametrize(
    ("field", "paired_field", "paired_value"),
    [
        ("open_system_exchange", "open_system_ledger_rate", [1.0]),
        ("open_system_ledger_rate", "open_system_exchange", [1.0, 0.0]),
    ],
)
def test_strict_open_reaction_rejects_bool_ledger_data(
    field,
    paired_field,
    paired_value,
):
    payload = _strict_open_payload("managed_export")
    payload[field] = np.array([True, False]) if field == "open_system_exchange" else [True]
    payload[paired_field] = paired_value

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("real" in detail.expected for detail in result.details)


@pytest.mark.parametrize(
    ("canonical", "alias"),
    [
        ("stoichiometric_N", "stoichiometric_matrix"),
        ("conservation_matrix", "element_matrix"),
        ("rates", "reaction_rates"),
        ("affinities", "reaction_affinities"),
    ],
)
def test_reaction_network_rejects_legacy_alias_only(canonical, alias):
    payload = _strict_closed_payload()
    payload[alias] = payload.pop(canonical)

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("canonical" in detail.expected for detail in result.details)


def test_reaction_network_rejects_canonical_plus_conflicting_alias():
    payload = _strict_closed_payload()
    payload["N"] = [[-2.0, 2.0], [2.0, -2.0]]

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("canonical" in detail.expected for detail in result.details)


@pytest.mark.parametrize(
    "convention",
    [None, "negative_affinity_times_rate_nonnegative"],
)
def test_strict_reaction_network_requires_canonical_affinity_sign(convention):
    payload = _strict_closed_payload()
    if convention is None:
        payload.pop("affinity_sign_convention")
    else:
        payload["affinity_sign_convention"] = convention

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("affinity sign convention" in detail.expected for detail in result.details)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("species", ["A", "A"]),
        ("reactions", ["A_to_B"]),
        ("rate_reaction_order", ["B_to_A", "A_to_B"]),
        ("affinity_reaction_order", ["B_to_A", "A_to_B"]),
        ("conservation_basis", []),
        ("state_domain", ""),
        ("reaction_domain", ""),
        ("conservation_domain", ""),
    ],
)
def test_strict_reaction_network_rejects_unbound_basis_or_domain(field, value):
    payload = _strict_closed_payload()
    payload[field] = value

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        "basis" in detail.expected
        or "order" in detail.expected
        or "domain" in detail.expected
        or "unique" in detail.expected
        for detail in result.details
    )


def test_strict_reaction_network_uses_aggregate_not_componentwise_dissipation():
    payload = _strict_closed_payload()
    payload["rates"] = [2.0, 1.0]
    payload["affinities"] = [-1.0, 3.0]

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is True
    assert result.metrics["min_rate_affinity_product"] == pytest.approx(-2.0)
    assert result.metrics["affinity_rate_power_native"] == pytest.approx(1.0)


def test_reaction_validation_is_independent_of_torch_default_dtype():
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        result = validate_reaction_network_payload(
            _strict_closed_payload(),
            strict=True,
        )
    finally:
        torch.set_default_dtype(original_dtype)

    assert result.passed is True
    assert result.metrics["affinity_rate_power_native"] == 0.6


def test_strict_reaction_network_rejects_conserved_wrong_subsystem_sum():
    payload = _strict_closed_payload()
    payload["subsystem_stoichiometric_matrices"] = [
        [[-2.0, 2.0], [2.0, -2.0]],
    ]

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("sum_i N_i = N_h" in detail.expected for detail in result.details)


@pytest.mark.parametrize(
    ("route", "declared_route"),
    [
        ("port", "boundary_export"),
        ("boundary_export", "managed_export"),
        ("managed_export", "port"),
    ],
)
def test_strict_open_reaction_rejects_route_binding_mismatch(route, declared_route):
    payload = _strict_open_payload(route)
    payload["open_system_route_binding"]["route"] = declared_route

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("route declaration" in detail.expected for detail in result.details)


@pytest.mark.parametrize("route", ["diagnostic_residual", "observation"])
def test_strict_open_reaction_rejects_nonphysical_exchange_route(route):
    payload = _strict_open_payload(route)

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("physical open-system route" in detail.expected for detail in result.details)


def test_strict_open_reaction_requires_complete_route_declaration():
    payload = _strict_open_payload("port")
    payload["open_system_route_binding"].pop("target")

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any("route declaration" in detail.expected for detail in result.details)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open_exchange_domain", "other_domain"),
        ("open_ledger_domain", "other_domain"),
        ("open_exchange_species_order", ["B", "A"]),
        ("open_ledger_basis", ["other_ledger"]),
    ],
)
def test_strict_open_reaction_rejects_domain_or_basis_mismatch(field, value):
    payload = _strict_open_payload("managed_export")
    payload[field] = value

    result = validate_reaction_network_payload(payload, strict=True)

    assert result.passed is False
    assert any(
        "domain" in detail.expected or "basis" in detail.expected
        for detail in result.details
    )


def test_reaction_network_structure_returns_failure_for_scalar_payload():
    results = validate_reaction_network_structure(
        ValidationContext(
            metadata={
                "strict": True,
                "reaction_network": 42,
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is False


def _ordinary_reaction_payload(*, open_route=None):
    payload = (
        _strict_closed_payload()
        if open_route is None
        else _strict_open_payload(open_route)
    )
    for key in (
        "affinities",
        "affinity_sign_convention",
        "affinity_reaction_order",
        "affinity_units",
        "affinity_rate_power_units",
    ):
        payload.pop(key)
    return payload


@pytest.mark.parametrize("route", [None, "port", "boundary_export", "managed_export"])
def test_ordinary_strict_reaction_accepts_real_ledger_without_affinity(route):
    payload = _ordinary_reaction_payload(open_route=route)

    result = validate_reaction_network_payload(
        payload, strict=True, require_affinity=False,
    )

    assert result.passed is True
    assert result.metrics["stoichiometric_conservation_residual"] == 0.0
    assert "affinity_rate_power_native" not in result.metrics
    if route is not None:
        assert result.metrics["open_system_ledger_residual"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stoichiometric_N", [[-1.0, 1.0], [0.0, -1.0]]),
        ("conservation_matrix", None),
        ("rates", None),
        ("rates", 0.2),
        ("rates", []),
        ("rates", [float("nan"), 0.1]),
        ("rates", [0.1]),
        ("species", ["A", "A"]),
        ("rate_reaction_order", ["B_to_A", "A_to_B"]),
        ("species_units", ["mol", "kg"]),
        ("rate_units", "mol"),
        ("state_domain", ""),
        ("open_system_ledger_rate", [0.0]),
        ("open_system_exchange_units", "kg s-1"),
    ],
)
def test_ordinary_strict_reaction_keeps_basic_contracts(field, value):
    payload = _ordinary_reaction_payload(open_route="port")
    payload[field] = value

    result = validate_reaction_network_payload(
        payload, strict=True, require_affinity=False,
    )

    assert result.passed is False
    assert result.details


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("affinities", [2.0, -1.0]),
        ("affinities", None),
        ("affinity_sign_convention", "opposite"),
        ("affinity_reaction_order", ["B_to_A", "A_to_B"]),
        ("affinity_units", "banana"),
        ("affinity_rate_power_units", "kg s-1"),
    ],
)
def test_ordinary_request_cannot_hide_declared_invalid_affinity(field, value, strict):
    payload = _strict_closed_payload()
    payload[field] = value

    result = validate_reaction_network_payload(
        payload, strict=strict, require_affinity=False,
    )

    assert result.passed is False
    assert result.details


@pytest.mark.parametrize(
    "field",
    [
        "affinities", "affinity_sign_convention", "affinity_reaction_order",
        "affinity_units", "affinity_rate_power_units",
    ],
)
def test_partial_affinity_declaration_is_not_an_ordinary_payload(field):
    payload = _ordinary_reaction_payload()
    payload[field] = _strict_closed_payload()[field]

    result = validate_reaction_network_payload(
        payload, strict=True, require_affinity=False,
    )

    assert result.passed is False


def test_outer_affinity_requirement_cannot_be_disabled_in_payload():
    payload = _ordinary_reaction_payload()
    payload["require_affinity"] = False
    payload["strict"] = False

    result = validate_reaction_network_payload(
        payload, strict=True, require_affinity=True,
    )

    assert result.passed is False
    assert any("affinities" in item.expected for item in result.details)


def test_framework_reaction_wrapper_keeps_outer_affinity_policy():
    results = validate_reaction_network_structure(
        ValidationContext(metadata={
            "strict": True,
            "require_affinity": False,
            "reaction_networks": [_ordinary_reaction_payload()],
        })
    )

    assert len(results) == 1
    assert results[0].passed is True


def test_nonstrict_numerical_affinity_audit_does_not_invent_unit_declarations():
    result = validate_reaction_network_payload(
        {
            "stoichiometric_N": [[-1.0], [1.0]],
            "conservation_matrix": [[1.0, 1.0]],
            "rates": [0.2],
            "affinities": [1.5],
        },
        strict=False,
        require_affinity=False,
    )

    assert result.passed is True
    assert result.metrics["affinity_rate_power_native"] == pytest.approx(0.3)


@pytest.mark.parametrize("declared", [
    {"rate_units": "banana"},
    {"reactions": ["transfer"], "rate_reaction_order": ["wrong"]},
    {"species": ["A", "A"]},
    {"state_domain": ""},
])
def test_nonstrict_rate_only_audit_checks_metadata_that_is_actually_declared(declared):
    payload = {
        "stoichiometric_N": [[-1.0], [1.0]],
        "conservation_matrix": [[1.0, 1.0]], "rates": [0.2],
        **declared,
    }

    result = validate_reaction_network_payload(payload, strict=False, require_affinity=False)

    assert result.passed is False


def test_nonstrict_rate_only_audit_accepts_valid_partial_metadata():
    result = validate_reaction_network_payload({
        "stoichiometric_N": [[-1.0], [1.0]],
        "conservation_matrix": [[1.0, 1.0]], "rates": [0.2],
        "rate_units": "mol s-1", "reactions": ["transfer"],
        "rate_reaction_order": ["transfer"],
    }, strict=False, require_affinity=False)

    assert result.passed is True
    assert "affinity_rate_power_native" not in result.metrics
