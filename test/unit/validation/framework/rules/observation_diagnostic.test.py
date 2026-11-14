"""Tests for observation and diagnostic structural validation rules."""


import pytest

from relic.computation.operators.dec import DiagnosticSpec, ObservationOperatorSpec
from relic.validation.framework.models import ValidationContext, ValidationLevel
from relic.validation.framework.registry import get_global_registry
from relic.validation.framework.rules import observation_diagnostic as rules


def _observation_mapping(
    *,
    name="tower_et",
    output_name="observations.latent_heat_flux",
):
    return {
        "name": name,
        "input_state_vars": ("soil_water_storage",),
        "output_name": output_name,
        "output_units": "W m-2",
        "support": "tower_footprint",
        "averaging_window": "30 min",
        "footprint_description": "eddy covariance footprint",
        "error_model": "gaussian",
        "conservative_projection_required": False,
        "metadata": {},
    }


def _binding(
    *,
    node_id="tower_operator",
    outputs=("observations.latent_heat_flux",),
):
    return {
        "node_id": node_id,
        "input_ports": ("soil_water_storage",),
        "output_ports": outputs,
    }


def test_observation_diagnostic_rule_registers_with_structural_tags():
    registry = get_global_registry()

    assert "observation_diagnostic_structure" in registry.list_rules(tags=["observation"])


def test_observation_diagnostic_rule_passes_declared_projection_and_readonly_diagnostic():
    observation = ObservationOperatorSpec(
        name="tower_et",
        input_state_vars=("soil_water_storage",),
        output_name="latent_heat_flux_observed",
        output_units="W m-2",
        support="tower_footprint",
        averaging_window="30 min",
        footprint_description="eddy covariance footprint",
        error_model="gaussian",
        conservative_projection_required=True,
        metadata={"assimilation_mode": "conservative_projection"},
    )
    diagnostic = DiagnosticSpec(
        name="water_use_efficiency",
        input_state_vars=("gross_primary_productivity", "evapotranspiration_flux"),
        output_units="g C kg-1 H2O",
        diagnostic_kind="ratio",
    )
    context = ValidationContext(
        metadata={"observation_specs": [observation], "diagnostic_specs": [diagnostic]}
    )

    results = rules.validate_observation_diagnostic_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.INFO
    assert results[0].passed is True


def test_observation_diagnostic_rule_fails_missing_conservative_accounting():
    observation = ObservationOperatorSpec(
        name="tower_et",
        input_state_vars=("soil_water_storage",),
        output_name="latent_heat_flux_observed",
        output_units="W m-2",
        support="tower_footprint",
        averaging_window="30 min",
        footprint_description="eddy covariance footprint",
        error_model="gaussian",
        conservative_projection_required=True,
    )
    context = ValidationContext(metadata={"observation_specs": [observation]})

    results = rules.validate_observation_diagnostic_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert "conservative" in results[0].message


def test_observation_diagnostic_rule_fails_state_writing_diagnostic_without_port():
    context = ValidationContext(
        metadata={
            "diagnostic_specs": [
                {
                    "name": "hidden_source",
                    "input_state_vars": ("x",),
                    "output_units": "kg",
                    "diagnostic_kind": "mass",
                    "writes_state": True,
                    "feedback_port": None,
                }
            ]
        }
    )

    results = rules.validate_observation_diagnostic_structure(context)

    assert len(results) == 1
    assert results[0].level == ValidationLevel.ERROR
    assert "feedback port" in results[0].message


def test_strict_observation_diagnostic_rejects_empty_payload():

    result = rules.validate_observation_diagnostic_payload((), (), strict=True)

    assert result.passed is False
    assert "non-empty" in result.message


def test_strict_observation_cannot_be_replaced_by_diagnostic():
    diagnostic = {
        "name": "carbon_total",
        "input_state_vars": ("soil_carbon",),
        "output_name": "diagnostics.carbon_total",
        "output_units": "kg C m-2",
        "diagnostic_kind": "mass",
        "writes_state": False,
    }

    result = rules.validate_observation_diagnostic_payload(
        (),
        (diagnostic,),
        strict=True,
        required_kinds=("observation",),
        available_state_vars=("soil_carbon",),
    )

    assert result.passed is False
    assert "observation declaration count" in result.message


def test_strict_observation_requires_state_coverage():
    observation = ObservationOperatorSpec(
        name="tower_et",
        input_state_vars=("soil_water_storage",),
        output_name="latent_heat_flux_observed",
        output_units="W m-2",
        support="tower_footprint",
        averaging_window="30 min",
        footprint_description="eddy covariance footprint",
        error_model="gaussian",
    )

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        required_kinds=("observation",),
        available_state_vars=("soil_water_storage", "canopy_temperature"),
        required_observation_state_vars=(
            "soil_water_storage",
            "canopy_temperature",
        ),
        spec_node_bindings={
            "tower_et": {
                "node_id": "tower_operator",
                "input_ports": ("soil_water_storage", "canopy_temperature"),
                "output_ports": ("latent_heat_flux_observed",),
            }
        },
    )

    assert result.passed is False
    assert "canopy_temperature" in result.message


def test_strict_observation_accepts_complete_coverage():
    observation = ObservationOperatorSpec(
        name="tower_et",
        input_state_vars=("soil_water_storage", "canopy_temperature"),
        output_name="latent_heat_flux_observed",
        output_units="W m-2",
        support="tower_footprint",
        averaging_window="30 min",
        footprint_description="eddy covariance footprint",
        error_model="gaussian",
    )

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        required_kinds=("observation",),
        available_state_vars=("soil_water_storage", "canopy_temperature"),
        required_observation_state_vars=(
            "soil_water_storage",
            "canopy_temperature",
        ),
        spec_node_bindings={
            "tower_et": {
                "node_id": "tower_operator",
                "input_ports": ("soil_water_storage", "canopy_temperature"),
                "output_ports": ("latent_heat_flux_observed",),
            }
        },
    )

    assert result.passed is True
    assert result.metrics["covered_observation_state_vars"] == 2


def test_strict_observation_accepts_exact_scoped_node_binding():
    observation = {
        "name": "tower_et",
        "input_state_vars": ("soil_water_storage",),
        "output_name": "observations.latent_heat_flux",
        "output_units": "W m-2",
        "support": "tower_footprint",
        "averaging_window": "30 min",
        "footprint_description": "eddy covariance footprint",
        "error_model": "gaussian",
    }

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        required_kinds=("observation",),
        available_state_vars=("soil_water_storage",),
        spec_node_bindings={
            "tower_et": {
                "node_id": "tower_operator",
                "input_ports": ("soil_water_storage",),
                "output_ports": ("observations.latent_heat_flux",),
            }
        },
    )

    assert result.passed is True
    assert result.metrics["graph_binding_verified"] == 1.0


def test_strict_observation_rejects_cross_node_input_substitution():
    observation = {
        "name": "tower_et",
        "input_state_vars": ("soil_water_storage",),
        "output_name": "observations.latent_heat_flux",
        "output_units": "W m-2",
        "support": "tower_footprint",
        "averaging_window": "30 min",
        "footprint_description": "eddy covariance footprint",
        "error_model": "gaussian",
    }

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        required_kinds=("observation",),
        available_state_vars=("soil_water_storage", "canopy_temperature"),
        spec_node_bindings={
            "tower_et": {
                "node_id": "tower_operator",
                "input_ports": ("canopy_temperature",),
                "output_ports": ("observations.latent_heat_flux",),
            }
        },
    )

    assert result.passed is False
    assert "input ports" in result.message


def test_strict_observation_rejects_output_outside_scoped_node():
    observation = {
        "name": "tower_et",
        "input_state_vars": ("soil_water_storage",),
        "output_name": "observations.decoy",
        "output_units": "W m-2",
        "support": "tower_footprint",
        "averaging_window": "30 min",
        "footprint_description": "eddy covariance footprint",
        "error_model": "gaussian",
    }

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        required_kinds=("observation",),
        available_state_vars=("soil_water_storage",),
        spec_node_bindings={
            "tower_et": {
                "node_id": "tower_operator",
                "input_ports": ("soil_water_storage",),
                "output_ports": ("observations.latent_heat_flux",),
            }
        },
    )

    assert result.passed is False
    assert "output ports" in result.message


def test_strict_diagnostic_accepts_typed_metadata_output_binding():
    diagnostic = DiagnosticSpec(
        name="mass_closure",
        input_state_vars=("diagnostics.mass",),
        output_units="kg",
        diagnostic_kind="closure",
        metadata={"outputs": ("diagnostics.mass_closure",)},
    )

    result = rules.validate_observation_diagnostic_payload(
        (),
        (diagnostic,),
        strict=True,
        required_kinds=("diagnostic",),
        available_state_vars=("diagnostics.mass",),
        spec_node_bindings={
            "mass_closure": {
                "node_id": "mass_diagnostic",
                "input_ports": ("diagnostics.mass",),
                "output_ports": ("diagnostics.mass_closure",),
            }
        },
    )

    assert result.passed is True


def test_strict_observation_rejects_self_referential_output_as_state_input():
    observation = {
        "name": "self_observation",
        "input_state_vars": ("diagnostics.self",),
        "output_name": "diagnostics.self",
        "output_units": "1",
        "support": "point",
        "averaging_window": "instantaneous",
        "footprint_description": "single state location",
        "error_model": "gaussian",
    }

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        required_kinds=("observation",),
        available_state_vars=("diagnostics.self",),
        required_observation_state_vars=("diagnostics.self",),
    )

    assert result.passed is False
    assert "own output" in result.message


def test_strict_observation_rejects_unparseable_output_units():
    observation = {
        "name": "bad_units_observation",
        "input_state_vars": ("soil_carbon",),
        "output_name": "observations.soil_carbon",
        "output_units": "banana",
        "support": "plot",
        "averaging_window": "instantaneous",
        "footprint_description": "plot support",
        "error_model": "gaussian",
    }

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        required_kinds=("observation",),
        available_state_vars=("soil_carbon",),
    )

    assert result.passed is False
    assert "parseable" in result.message


def test_strict_diagnostic_rejects_unparseable_output_units():
    diagnostic = {
        "name": "bad_units_diagnostic",
        "input_state_vars": ("soil_carbon",),
        "output_name": "diagnostics.soil_carbon",
        "output_units": "banana",
        "diagnostic_kind": "mass",
        "writes_state": False,
    }

    result = rules.validate_observation_diagnostic_payload(
        (),
        (diagnostic,),
        strict=True,
        required_kinds=("diagnostic",),
        available_state_vars=("soil_carbon",),
    )

    assert result.passed is False
    assert "parseable" in result.message


def test_observation_diagnostic_rule_propagates_strict_requirements():
    context = ValidationContext(
        metadata={
            "strict": True,
            "required_observation_kinds": ["observation"],
        }
    )

    results = rules.validate_observation_diagnostic_structure(context)

    assert len(results) == 1
    assert results[0].passed is False


def test_observation_payload_rejects_scalar_spec_container():

    result = rules.validate_observation_diagnostic_payload(
        42,
        (),
        strict=True,
    )

    assert result.passed is False
    assert "list or tuple" in result.message


def test_observation_payload_rejects_mixed_invalid_spec_item():

    result = rules.validate_observation_diagnostic_payload(
        (_observation_mapping(), 42),
        (),
        strict=True,
        available_state_vars=("soil_water_storage",),
        spec_node_bindings={"tower_et": _binding()},
    )

    assert result.passed is False
    assert "ObservationOperatorSpec or Mapping" in result.message


@pytest.mark.parametrize("alias", ["observation_spec", "diagnostic_spec"])
def test_observation_wrapper_rejects_singular_alias(alias):
    metadata = {alias: _observation_mapping()}

    results = rules.validate_observation_diagnostic_structure(
        ValidationContext(metadata=metadata)
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert "plural canonical" in results[0].message


def test_observation_wrapper_rejects_plural_and_singular_conflict():
    metadata = {
        "observation_specs": [_observation_mapping()],
        "observation_spec": _observation_mapping(name="decoy"),
    }

    results = rules.validate_observation_diagnostic_structure(
        ValidationContext(metadata=metadata)
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert "plural canonical" in results[0].message


def test_strict_observation_requires_graph_binding_after_spec_validation():

    result = rules.validate_observation_diagnostic_payload(
        (_observation_mapping(),),
        (),
        strict=True,
        available_state_vars=("soil_water_storage",),
    )

    assert result.passed is False
    assert "spec_node_bindings" in result.message
    assert "[N]" in result.message


def test_strict_binding_rejects_unclaimed_node_output():

    result = rules.validate_observation_diagnostic_payload(
        (_observation_mapping(),),
        (),
        strict=True,
        available_state_vars=("soil_water_storage",),
        spec_node_bindings={
            "tower_et": _binding(
                outputs=("observations.latent_heat_flux", "observations.unclaimed"),
            )
        },
    )

    assert result.passed is False
    assert "declared output union" in result.message


def test_strict_multi_spec_node_accepts_exact_output_union():
    first = _observation_mapping(name="tower_et")
    second = _observation_mapping(
        name="tower_h",
        output_name="observations.sensible_heat_flux",
    )
    outputs = (
        "observations.latent_heat_flux",
        "observations.sensible_heat_flux",
    )

    result = rules.validate_observation_diagnostic_payload(
        (first, second),
        (),
        strict=True,
        available_state_vars=("soil_water_storage",),
        spec_node_bindings={
            "tower_et": _binding(outputs=outputs),
            "tower_h": _binding(outputs=outputs),
        },
    )

    assert result.passed is True
    assert result.metrics["graph_binding_verified"] == 1


def test_strict_multi_spec_node_rejects_inconsistent_binding_views():
    first = _observation_mapping(name="tower_et")
    second = _observation_mapping(
        name="tower_h",
        output_name="observations.sensible_heat_flux",
    )

    result = rules.validate_observation_diagnostic_payload(
        (first, second),
        (),
        strict=True,
        available_state_vars=("soil_water_storage",),
        spec_node_bindings={
            "tower_et": _binding(
                outputs=(
                    "observations.latent_heat_flux",
                    "observations.sensible_heat_flux",
                )
            ),
            "tower_h": _binding(outputs=("observations.sensible_heat_flux",)),
        },
    )

    assert result.passed is False
    assert "consistent input/output port sets" in result.message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_name", 2),
        ("support", True),
        ("averaging_window", 30),
        ("footprint_description", False),
        ("error_model", True),
        ("output_units", 1),
    ],
)
def test_strict_observation_rejects_non_string_fields(field, value):
    observation = _observation_mapping()
    observation[field] = value

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        available_state_vars=("soil_water_storage",),
    )

    assert result.passed is False
    assert f"{field} must be a non-empty exact string" in result.message


def test_strict_observation_rejects_non_string_input_identity():
    observation = _observation_mapping()
    observation["input_state_vars"] = (True,)

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        available_state_vars=("soil_water_storage",),
    )

    assert result.passed is False
    assert "input_state_vars must contain non-empty exact string identities" in result.message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conservative_projection_required", "false"),
        ("metadata", "not-a-mapping"),
    ],
)
def test_strict_observation_rejects_invalid_bool_or_metadata(field, value):
    observation = _observation_mapping()
    observation[field] = value

    result = rules.validate_observation_diagnostic_payload(
        (observation,),
        (),
        strict=True,
        available_state_vars=("soil_water_storage",),
    )

    assert result.passed is False
    assert field in result.message


def test_strict_diagnostic_remains_read_only_even_with_bound_feedback_port():
    diagnostic = {
        "name": "state_feedback_diagnostic",
        "input_state_vars": ("soil_water_storage",),
        "output_name": "diagnostics.feedback",
        "output_units": "kg",
        "diagnostic_kind": "mass",
        "writes_state": True,
        "feedback_port": "state_feedback",
        "metadata": {},
    }

    result = rules.validate_observation_diagnostic_payload(
        (),
        (diagnostic,),
        strict=True,
        required_kinds=("diagnostic",),
        available_state_vars=("soil_water_storage",),
        spec_node_bindings={
            "state_feedback_diagnostic": {
                "node_id": "diagnostic_node",
                "input_ports": ("soil_water_storage", "state_feedback"),
                "output_ports": ("diagnostics.feedback",),
            }
        },
    )

    assert result.passed is False
    assert "strict diagnostics are read-only" in result.message


def test_strict_diagnostic_rejects_unbound_feedback_port_reference():
    diagnostic = {
        "name": "diagnostic_with_port_reference",
        "input_state_vars": ("soil_water_storage",),
        "output_name": "diagnostics.mass",
        "output_units": "kg",
        "diagnostic_kind": "mass",
        "writes_state": False,
        "feedback_port": "missing_port",
        "metadata": {},
    }

    result = rules.validate_observation_diagnostic_payload(
        (),
        (diagnostic,),
        strict=True,
        required_kinds=("diagnostic",),
        available_state_vars=("soil_water_storage",),
        spec_node_bindings={
            "diagnostic_with_port_reference": {
                "node_id": "diagnostic_node",
                "input_ports": ("soil_water_storage",),
                "output_ports": ("diagnostics.mass",),
            }
        },
    )

    assert result.passed is False
    assert "feedback_port must belong" in result.message
