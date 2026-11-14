"""Tests for IPHS port power unit validation."""

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.validation.framework.models import ValidationContext
from relic.validation.framework.rules.port_power import (
    validate_port_power_payload,
    validate_port_power_structure,
)


def _object_port(name="energy_port", power_units="W"):
    effort_space = CochainSpaceSpec(
        complex_id="power_test",
        degree=0,
        primality="dual",
        n_dofs=1,
        units="J",
        extensivity="intensive",
        physical_domain="test",
        semantic_name=f"{name}_effort",
        orientation_convention="positive_into_system",
        boundary_role="port",
    )
    flow_space = CochainSpaceSpec(
        complex_id="power_test",
        degree=0,
        primality="primal",
        n_dofs=1,
        units="s-1",
        extensivity="integral",
        physical_domain="test",
        semantic_name=f"{name}_flow",
        orientation_convention="positive_into_system",
        boundary_role="port",
    )
    return PortSpecIPHS(
        name=name,
        effort_var=f"{name}_effort",
        flow_var=f"{name}_flow",
        effort_space=effort_space,
        flow_space=flow_space,
        power_units=power_units,
        positive_power_convention="into_system",
        port_kind="energy",
        role="external",
    )


def test_port_power_payload_accepts_matching_units():
    result = validate_port_power_payload(
        {
            "name": "heat_boundary",
            "power_units": "W m-2",
            "expected_power_units": "W m-2",
        }
    )

    assert result.passed is True


def test_port_power_payload_rejects_missing_units():
    result = validate_port_power_payload({"name": "missing_units"})

    assert result.passed is False
    assert "power_units" in result.message


def test_port_power_payload_rejects_mismatched_units():
    result = validate_port_power_payload(
        {
            "name": "bad_boundary",
            "power_units": "kg s-1",
            "expected_power_units": "W m-2",
        }
    )

    assert result.passed is False
    assert result.details[0].expected == "W m-2"
    assert result.details[0].value == "kg s-1"


def test_port_power_payload_rejects_unknown_matching_unit_labels():
    result = validate_port_power_payload(
        {
            "name": "unknown_boundary",
            "power_units": "banana",
            "expected_power_units": "banana",
        }
    )

    assert result.passed is False
    assert any("parseable" in detail.expected for detail in result.details)


def test_port_power_payload_rejects_unknown_expected_unit_label():
    result = validate_port_power_payload(
        {
            "name": "unknown_expected_boundary",
            "power_units": "W m-2",
            "expected_power_units": "banana",
        }
    )

    assert result.passed is False
    assert any("expected_power_units" in detail.variable_name for detail in result.details)


def test_port_power_payload_accepts_portspec_object_with_identity():
    result = validate_port_power_payload(_object_port())

    assert result.passed is True
    assert result.metrics["port_count"] == 1
    identity = result.details[0].metadata["contract_identity"]
    assert identity == {
        "name": "energy_port",
        "effort_var": "energy_port_effort",
        "flow_var": "energy_port_flow",
        "power_units": "W",
        "positive_power_convention": "into_system",
        "port_kind": "energy",
        "role": "external",
    }


def test_port_power_structure_aggregates_portspec_object_bundle():
    context = ValidationContext(
        metadata={
            "ports": [
                _object_port("heat_input", "W"),
                _object_port("matter_input", "kg s-1"),
            ]
        }
    )

    results = validate_port_power_structure(context)

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].metrics == {
        "port_count": 2,
        "passed_port_count": 2,
        "failed_port_count": 0,
        "identity_complete_count": 2,
    }
    assert {detail.variable_name for detail in results[0].details} == {
        "heat_input",
        "matter_input",
    }


def test_port_power_payload_aggregates_failure_without_dropping_valid_port():
    result = validate_port_power_payload(
        {
            "ports": [
                _object_port("valid_power", "W"),
                _object_port("invalid_power", "banana"),
            ]
        }
    )

    assert result.passed is False
    assert result.metrics["port_count"] == 2
    assert result.metrics["passed_port_count"] == 1
    assert result.metrics["failed_port_count"] == 1
    assert {detail.variable_name.split(".")[0] for detail in result.details} == {
        "valid_power",
        "invalid_power",
    }


def test_port_power_payload_rejects_invalid_bundle_item_instead_of_filtering_it():
    result = validate_port_power_payload(
        {"ports": [_object_port("valid_power"), object()]}
    )

    assert result.passed is False
    assert result.metrics["port_count"] == 2
    assert result.metrics["failed_port_count"] == 1
    assert "unsupported port contract object" in result.message
