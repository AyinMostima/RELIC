"""Tests for IPHS port matching validation."""

from relic.validation.framework.models import ValidationContext
from relic.validation.framework.rules.port_matching import (
    validate_port_matching_payload,
    validate_port_matching_structure,
)
from relic.scientific.hydrosphere.soil_moisture.richards import (
    RichardsSoilMoisture,
)
from relic.scientific.hydrosphere.transport.capillary_rise import (
    CapillaryRiseTransport,
)


def _port(name, convention, **extra):
    payload = {
        "name": name,
        "connection_id": "carbon_exchange",
        "positive_power_convention": convention,
        "port_kind": "matter",
        "power_units": "J s-1",
        "element": "carbon",
        "effort_space": {
            "complex_id": "grid",
            "degree": 0,
            "primality": "dual",
            "extensivity": "intensive",
            "physical_domain": "biosphere",
            "orientation_convention": "positive_into_system",
            "boundary_role": "port",
            "units": "J mol-1",
            "n_dofs": 1,
        },
        "flow_space": {
            "complex_id": "grid",
            "degree": 2,
            "primality": "primal",
            "extensivity": "integral",
            "physical_domain": "biosphere",
            "orientation_convention": "positive_into_system",
            "boundary_role": "port",
            "units": "mol s-1",
            "n_dofs": 1,
        },
    }
    payload.update(extra)
    return payload


def test_port_matching_accepts_closed_power_pair():
    result = validate_port_matching_payload(
        {
            "ports": [
                _port("atmosphere_out", "out_of_system"),
                _port("biomass_in", "into_system"),
            ]
        }
    )

    assert result.passed is True
    assert result.metrics["port_count"] == 2


def test_port_matching_accepts_explicit_open_boundary():
    result = validate_port_matching_payload(
        {
            "ports": [
                _port(
                    "runoff_export",
                    "out_of_system",
                    connection_id="water_runoff",
                    open_boundary=True,
                    boundary_role="managed_export",
                )
            ]
        }
    )

    assert result.passed is True
    assert result.metrics["open_boundary_count"] == 1


def test_port_matching_rejects_unmatched_closed_port():
    result = validate_port_matching_payload(
        {"ports": [_port("soil_respiration_out", "out_of_system")]}
    )

    assert result.passed is False
    assert "unmatched" in result.message


def test_port_matching_rejects_incompatible_power_units():
    result = validate_port_matching_payload(
        {
            "ports": [
                _port("source", "out_of_system", power_units="J s-1"),
                _port("sink", "into_system", power_units="kg s-1"),
            ]
        }
    )

    assert result.passed is False
    assert any(detail.expected == "one power_units value per matched group" for detail in result.details)


def test_port_matching_structure_reads_context_metadata():
    results = validate_port_matching_structure(
        ValidationContext(
            metadata={
                "port_matching": {
                    "ports": [
                        _port("donor", "out_of_system"),
                        _port("receiver", "into_system"),
                    ]
                }
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is True


def test_port_matching_strict_mode_rejects_empty_undeclared_payload():
    result = validate_port_matching_payload({"ports": [], "strict": True})

    assert result.passed is False
    assert any("closed_system" in detail.expected for detail in result.details)


def test_port_matching_strict_mode_accepts_explicit_zero_port_closed_system():
    result = validate_port_matching_payload(
        {"ports": [], "strict": True, "closed_system": True}
    )

    assert result.passed is True
    assert result.metrics["port_count"] == 0


def test_port_matching_strict_mode_rejects_incomplete_typed_space_metadata():
    source = _port("source", "out_of_system")
    target = _port("target", "into_system")
    del source["flow_space"]["complex_id"]
    del target["flow_space"]["complex_id"]

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any(
        detail.expected == "complete typed flow cochain metadata"
        for detail in result.details
    )


def test_port_matching_strict_mode_rejects_incomplete_effort_space_metadata():
    source = _port("source", "out_of_system")
    target = _port("target", "into_system")
    del source["effort_space"]["units"]
    del target["effort_space"]["units"]

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any(
        detail.expected == "complete typed effort cochain metadata"
        for detail in result.details
    )


def test_port_matching_strict_mode_rejects_effort_flow_dof_mismatch():
    source = _port("source", "out_of_system")
    target = _port("target", "into_system")
    source["effort_space"]["n_dofs"] = 2

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any(
        detail.expected == "equal positive effort_space and flow_space n_dofs"
        for detail in result.details
    )


def test_port_matching_strict_mode_rejects_matched_effort_space_mismatch():
    source = _port("source", "out_of_system")
    target = _port("target", "into_system")
    target["effort_space"]["physical_domain"] = "atmosphere"

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any(
        detail.expected == "compatible effort cochain space metadata"
        for detail in result.details
    )


def test_port_matching_strict_mode_rejects_matched_port_dof_mismatch():
    source = _port("source", "out_of_system")
    target = _port("target", "into_system")
    target["effort_space"]["n_dofs"] = 2
    target["flow_space"]["n_dofs"] = 2

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any(
        detail.expected == "one port n_dofs value per matched group"
        for detail in result.details
    )


def test_port_matching_strict_mode_rejects_nonconjugate_power_units():
    source = _port("source", "out_of_system", power_units="kg s-1")
    target = _port("target", "into_system", power_units="kg s-1")

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any(
        detail.expected
        == "power_units dimensionality equals effort units times flow units"
        for detail in result.details
    )


def test_port_matching_strict_mode_preserves_zero_form_degree():
    source = _port("source", "out_of_system")
    target = _port("target", "into_system")
    source["flow_space"]["degree"] = 0
    target["flow_space"]["degree"] = 0

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is True


def test_port_matching_strict_mode_rejects_unknown_flow_unit_labels():
    source = _port("source", "out_of_system")
    target = _port("target", "into_system")
    source["flow_space"]["units"] = "banana"
    target["flow_space"]["units"] = "banana"

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any("parseable flow units" in detail.expected for detail in result.details)


def test_port_matching_strict_mode_rejects_unknown_power_unit_labels():
    source = _port("source", "out_of_system", power_units="banana")
    target = _port("target", "into_system", power_units="banana")

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any("parseable power_units" in detail.expected for detail in result.details)


def test_port_matching_strict_mode_rejects_orientation_mismatch():
    source = _port("source", "out_of_system")
    target = _port("target", "into_system")
    target["flow_space"]["orientation_convention"] = "positive_downward"

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any(
        detail.expected == "compatible flow cochain space metadata"
        for detail in result.details
    )


def test_port_matching_strict_mode_rejects_invalid_mapping_enums():
    source = _port("source", "out_of_system", port_kind="unknown")
    target = _port("target", "into_system", port_kind="unknown")
    for port in (source, target):
        port["effort_space"]["degree"] = 99
        port["effort_space"]["primality"] = "sideways"
        port["flow_space"]["extensivity"] = "mystery"
        port["flow_space"]["boundary_role"] = "nowhere"

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    expected_contracts = {detail.expected for detail in result.details}
    assert "degree is one of 0, 1, 2, or 3" in expected_contracts
    assert "primality is primal or dual" in expected_contracts
    assert "recognized cochain extensivity" in expected_contracts
    assert "recognized cochain boundary_role" in expected_contracts
    assert "recognized port_kind" in expected_contracts


def test_port_matching_strict_mode_rejects_noncanonical_direction_aliases():
    source = _port("source", "out")
    target = _port("target", "in")

    result = validate_port_matching_payload(
        {"ports": [source, target], "strict": True}
    )

    assert result.passed is False
    assert any(
        detail.expected == "canonical into_system or out_of_system"
        for detail in result.details
    )


def test_port_matching_external_role_requires_explicit_allowed_open_route():
    missing_route = _port(
        "external_missing_route",
        "out_of_system",
        role="external",
    )
    invalid_route = _port(
        "external_invalid_route",
        "out_of_system",
        role="external",
        open_system_route="hidden_sink",
    )
    valid_route = _port(
        "external_valid_route",
        "out_of_system",
        role="external",
        open_system_route="port",
    )

    missing = validate_port_matching_payload(
        {"ports": [missing_route], "strict": True}
    )
    invalid = validate_port_matching_payload(
        {"ports": [invalid_route], "strict": True}
    )
    valid = validate_port_matching_payload(
        {"ports": [valid_route], "strict": True}
    )

    assert missing.passed is False
    assert any(
        detail.expected == "explicit allowed open_system_route for external role"
        for detail in missing.details
    )
    assert invalid.passed is False
    assert any(
        detail.expected == "allowed open_system_route"
        for detail in invalid.details
    )
    assert valid.passed is True
    assert valid.metrics["open_boundary_count"] == 1


def test_port_matching_success_records_deterministic_contract_identities():
    result = validate_port_matching_payload(
        {
            "ports": [
                _port("source", "out_of_system"),
                _port("target", "into_system"),
            ],
            "strict": True,
        }
    )

    assert result.passed is True
    assert result.metrics["identity_complete_count"] == 2
    identities = [
        detail.metadata["contract_identity"]
        for detail in result.details
        if "contract_identity" in detail.metadata
    ]
    assert {identity["name"] for identity in identities} == {"source", "target"}
    assert all(identity["matching_group"] == "carbon_exchange" for identity in identities)
    assert all(len(identity["effort_space_signature"]) == 8 for identity in identities)
    assert all(len(identity["flow_space_signature"]) == 8 for identity in identities)


def test_capillary_rise_and_richards_ports_form_one_strict_pair():
    """Validate the real reciprocal groundwater-to-soil water port.

    References: Richards (1931); van der Schaft and Maschke (1995).
    Mathematical expression: s_out F_cap + s_in F_cap = -F_cap + F_cap = 0.
    Parameter meanings: F_cap is upward capillary mass flux and s is the port
        incidence sign defined by positive_power_convention.
    """

    capillary = CapillaryRiseTransport(name="capillary_rise_transport")
    richards = RichardsSoilMoisture(name="richards_soil_moisture")
    producer = next(
        port for port in capillary.ports() if port.flow_var == "capillary_rise_flux"
    )
    consumer = next(
        port for port in richards.ports() if port.flow_var == "capillary_rise_flux"
    )

    result = validate_port_matching_payload(
        {"ports": [producer, consumer], "strict": True}
    )

    assert result.passed is True
    assert result.metrics["port_count"] == 2
