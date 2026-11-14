from pathlib import Path
import warnings

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.land_use.mapping import (
    DIAGNOSTIC_PREFIX,
    RUNTIME_DIAGNOSTIC,
    LandUseParameterMapping,
)
from relic.scientific.anthroposphere.land_use.transitions import (
    LandUseTransitions,
    LandUseType,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _state() -> dict[str, torch.Tensor]:
    fractions = torch.zeros((len(LandUseType), 2, 2), dtype=torch.float32)
    fractions[0, :, :] = 0.2
    fractions[1, :, :] = 0.1
    fractions[2, :, :] = 0.3
    fractions[3, :, :] = 0.1
    fractions[4, :, :] = 0.2
    fractions[6, :, :] = 0.1
    return {"land_use_fraction": fractions}


def test_land_use_parameter_mapping_declares_honest_diagnostic_structure():
    mapper = LandUseParameterMapping()

    assert mapper.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert mapper.runtime_path == "canonical_land_use_parameter_diagnostic_observation"

    forms = mapper.declared_forms()
    assert [form.semantic_name for form in forms] == [
        "land_use_fraction_observation"
    ]
    assert mapper.build_hodge_terms() == ()
    assert mapper.ports() == ()
    assert mapper.reversible_structure({}) == {"state_vars": (), "ports": ()}
    assert mapper.dissipative_structure({}) == {"state_vars": (), "ports": ()}


def test_land_use_parameter_mapping_forward_reports_canonical_runtime():
    mapper = LandUseParameterMapping()

    output = mapper.forward(_state(), time=0.0, dt=86400.0)

    assert output[RUNTIME_DIAGNOSTIC].value.item() == 1.0
    for name in (
        f"{DIAGNOSTIC_PREFIX}.vegetation.lai_max",
        f"{DIAGNOSTIC_PREFIX}.soil.soil_porosity",
        f"{DIAGNOSTIC_PREFIX}.hydrology.albedo",
    ):
        update = output[name]
        assert isinstance(update, UpdateItem)
        assert update.mode == UpdateMode.REPLACE
        assert update.per_timestep is False
        assert torch.isfinite(update.value).all()


def test_land_use_parameter_mapping_rejects_zero_support():
    mapper = LandUseParameterMapping()
    state = {"land_use_fraction": torch.zeros((len(LandUseType), 1, 1), dtype=torch.float32)}

    with pytest.raises(ValueError, match="sum to one"):
        mapper.forward(state, time=0.0, dt=86400.0)


def test_land_use_parameter_mapping_rejects_missing_fraction_state():
    mapper = LandUseParameterMapping()

    with pytest.raises(KeyError, match="land_use_fraction"):
        mapper.forward({}, time=0.0, dt=86400.0)


def test_land_use_parameter_mapping_spatial_queries_fail_without_declared_support():
    mapper = LandUseParameterMapping()
    coordinates = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="current spatial field"):
        mapper.get_value_at(coordinates)
    with pytest.raises(RuntimeError, match="current spatial field"):
        mapper.compute_gradient(coordinates)


def test_land_use_parameter_mapping_requires_explicit_spatial_chain_maps():
    mapper = LandUseParameterMapping()
    source_coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    fractions = torch.zeros((len(LandUseType), 3, 1), dtype=torch.float32)
    fractions[0, :, 0] = torch.tensor([1.0, 0.0, 0.0])
    fractions[2, :, 0] = torch.tensor([0.0, 1.0, 0.0])
    fractions[6, :, 0] = torch.tensor([0.0, 0.0, 1.0])
    mapper.update_field(
        time_step=86400.0,
        source_terms={
            "land_use_fraction": fractions,
            "coordinates": source_coordinates,
        },
    )

    with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
        mapper.get_value_at(source_coordinates)
    with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
        mapper.compute_gradient(source_coordinates)


def test_land_use_parameter_mapping_uses_bound_spatial_chain_maps():
    calls: list[str] = []

    def query_bridge(source_coordinates, source_values, target_coordinates):
        calls.append("query")
        torch.testing.assert_close(source_coordinates, target_coordinates)
        return source_values

    def gradient_bridge(source_coordinates, source_values, target_coordinates):
        calls.append("gradient")
        torch.testing.assert_close(source_coordinates, target_coordinates)
        return torch.stack((source_values, -source_values), dim=-1)

    mapper = LandUseParameterMapping(
        field_query_bridge=query_bridge,
        field_gradient_bridge=gradient_bridge,
    )
    source_coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    fractions = torch.zeros((len(LandUseType), 3, 1), dtype=torch.float32)
    fractions[0, :, 0] = torch.tensor([1.0, 0.0, 0.0])
    fractions[2, :, 0] = torch.tensor([0.0, 1.0, 0.0])
    fractions[6, :, 0] = torch.tensor([0.0, 0.0, 1.0])
    mapper.update_field(
        time_step=86400.0,
        source_terms={
            "land_use_fraction": fractions,
            "coordinates": source_coordinates,
        },
    )

    values = mapper.get_value_at(source_coordinates)
    gradients = mapper.compute_gradient(source_coordinates)

    assert calls == ["query", "gradient"]
    assert values.shape == (3,)
    assert gradients.shape == (3, 2)
    assert torch.any(values[1:] != values[:-1])
    assert torch.any(gradients != 0.0)


def test_land_use_parameter_mapping_phase8_certificate_is_valid():
    certificate = load_iphs_equivalence_certificate(
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_land_use_mapping_land_use_parameter_mapping.yaml"
    )

    audit = certificate.validate()
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert certificate.iphs_level == 1
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False
    assert (
        certificate.dependency_interconnection_map["direct_functor_calls"]
        == "none_no_direct_functor_calls"
    )


def test_land_use_parameter_mapping_real_strict_simulator_is_diagnostic_only():
    fractions = torch.tensor(
        [0.25, 0.15, 0.20, 0.05, 0.15, 0.10, 0.05, 0.03, 0.02],
        dtype=torch.float64,
    )
    state = {
        "land_use_fraction": fractions.clone(),
        "time_index": torch.tensor(0, dtype=torch.int64),
        "gdp_per_capita": torch.tensor(250.0, dtype=torch.float64),
        "population_density": torch.tensor(150.0, dtype=torch.float64),
        "market_accessibility": torch.tensor(0.4, dtype=torch.float64),
        "policy_intensity": torch.tensor(0.2, dtype=torch.float64),
    }
    dt = 86400.0
    simulator = RELICSimulator(
        {
            "max_time": 2.0 * dt,
            "time_step": dt,
            "conservation_checks": True,
            "strict_dependencies": True,
            "device": "cpu",
        }
    )
    simulator.initialize_state(state)
    transition = LandUseTransitions()
    mapper = LandUseParameterMapping()
    simulator.register_functor(transition, execution_group="fast")
    simulator.register_functor(mapper, execution_group="fast")
    key = f"{DIAGNOSTIC_PREFIX}.vegetation.lai_max"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first_outputs = simulator.step(dt=dt)
        fraction_after_first = simulator.state_manager.get_state()[
            "land_use_fraction"
        ].clone()
        second_outputs = simulator.step(dt=dt)
        final_state = simulator.state_manager.get_state()
        final_fraction = final_state["land_use_fraction"].clone()

    expected_first_update = mapper.forward(
        {"land_use_fraction": fraction_after_first}, time=dt, dt=dt
    )[key]
    expected_second_update = mapper.forward(
        {"land_use_fraction": final_fraction}, time=2.0 * dt, dt=dt
    )[key]
    first_value = first_outputs[key]
    second_value = second_outputs[key]

    assert isinstance(first_value, torch.Tensor)
    assert isinstance(second_value, torch.Tensor)
    assert isinstance(expected_first_update, UpdateItem)
    assert isinstance(expected_second_update, UpdateItem)
    assert expected_second_update.mode is UpdateMode.REPLACE
    assert expected_second_update.per_timestep is False
    assert expected_second_update.unit == "dimensionless"
    assert expected_second_update.source_functor == mapper.name
    torch.testing.assert_close(
        first_value, expected_first_update.value, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        second_value, expected_second_update.value, rtol=0.0, atol=0.0
    )
    assert key not in final_state
    assert simulator.get_variable_provenance(key) == []
    fraction_provenance = simulator.get_variable_provenance("land_use_fraction")
    assert {entry["functor"] for entry in fraction_provenance} == {transition.name}
    assert {entry["role"] for entry in fraction_provenance} == {
        "writer",
        "final_writer",
    }
    torch.testing.assert_close(
        final_fraction.sum(),
        torch.tensor(1.0, dtype=final_fraction.dtype),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert torch.max(torch.abs(final_fraction - fractions)) > 0.0
    assert simulator.get_effective_device(transition.name).type == "cpu"
    assert simulator.get_effective_device(mapper.name).type == "cpu"
    assert caught == []


def test_land_use_parameter_mapping_forward_has_no_environment_parameter():
    """land_use_fraction, the declared input, comes from the state; forward takes
    (state, time, dt) only, so an environment mapping is rejected (item 113)."""
    import inspect

    import yaml

    parameters = inspect.signature(LandUseParameterMapping.forward).parameters
    assert list(parameters) == ["self", "state", "time", "dt"]
    certificate = yaml.safe_load(
        (
            PROJECT_ROOT
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "anthroposphere_land_use_mapping_land_use_parameter_mapping.yaml"
        ).read_text(encoding="utf-8")
    )
    assert certificate["interface_map"]["forward_signature"] == (
        "forward(self, state, time, dt)"
    )


def test_land_use_parameter_mapping_declares_spatial_dims_as_a_class_attribute():
    """Item 68.2b: a PhysicalFieldFunctor takes its field identity from the declared
    outputs and their contracts and declares SPATIAL_DIMS as a class attribute. The
    mapping queries coordinates of shape [n, 2], so SPATIAL_DIMS = 2, and the
    constructor passes none of the removed PhysicalField keywords."""
    import inspect

    assert LandUseParameterMapping.SPATIAL_DIMS == 2
    source = inspect.getsource(LandUseParameterMapping.__init__)
    for keyword in (
        "field_name=",
        "units=",
        "default_value=",
        "spatial_dims=",
        "temporal_dependence=",
    ):
        assert keyword not in source, keyword


def test_land_use_parameter_mapping_interpolate_has_no_method_argument():
    """The injected chain-map bridge owns the interpolation algorithm; the former
    ``method`` argument accepted only "bound_chain_map" and is removed."""
    import inspect

    parameters = inspect.signature(LandUseParameterMapping.interpolate).parameters
    assert list(parameters) == [
        "self",
        "source_coordinates",
        "target_coordinates",
        "source_values",
    ]
    assert parameters["source_values"].kind is inspect.Parameter.KEYWORD_ONLY
