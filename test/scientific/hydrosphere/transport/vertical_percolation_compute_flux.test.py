from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.hydrosphere.transport.vertical_percolation import (
    VerticalPercolationTransport,
)


def _scalar(value: torch.Tensor) -> float:
    if hasattr(value, "value"):
        value = value.value
    tensor = value.detach().to(dtype=torch.float32)
    if tensor.numel() == 1:
        return float(tensor.item())
    return float(tensor.mean().item())


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


EXPECTED_OUTPUTS = {
    "drainage_flux",
    "diagnostics.vertical_percolation.effective_saturation",
    "diagnostics.vertical_percolation.hydraulic_conductivity",
    "diagnostics.vertical_percolation.relative_hydraulic_conductivity",
    "diagnostics.vertical_percolation.field_capacity_excess",
    "diagnostics.vertical_percolation.water_balance_residual",
}


def _state(theta: float, temperature: float = 298.15) -> dict[str, torch.Tensor]:
    return {
        "soil_moisture": torch.tensor(theta, dtype=torch.float32),
        "soil_temperature": torch.tensor(temperature, dtype=torch.float32),
    }


def test_vertical_percolation_forward_returns_exact_canonical_outputs():
    model = VerticalPercolationTransport(name="vp_flux_finite")

    outputs = model.forward(_state(0.40), time=0.0, dt=3600.0)

    assert set(outputs) == EXPECTED_OUTPUTS
    assert set(model.metadata.output_variables) == EXPECTED_OUTPUTS
    assert set(model.metadata.conservation_laws) == {
        "vertical_drainage_boundary"
    }
    assert {
        "percolation_drainage_flux",
        "soil_drainage",
        "subsurface_runoff_flux",
        "volume_fraction_of_water_in_soil",
        "diagnostics.percolation",
    }.isdisjoint(outputs)
    for name, update in outputs.items():
        assert isinstance(update, UpdateItem), name
        assert update.mode == UpdateMode.REPLACE, name
        assert update.per_timestep is False, name
        assert torch.isfinite(update.value).all(), name

    assert outputs["drainage_flux"].unit == "kg m-2 s-1"
    assert outputs["diagnostics.vertical_percolation.hydraulic_conductivity"].unit == "m s-1"
    assert outputs["diagnostics.vertical_percolation.water_balance_residual"].unit == "kg m-2 s-1"


def test_vertical_percolation_forward_drainage_increases_above_field_capacity():
    model = VerticalPercolationTransport(name="vp_flux_storage_response")
    low = model.forward(_state(0.20), time=0.0, dt=3600.0)["drainage_flux"]
    high = model.forward(_state(0.40), time=0.0, dt=3600.0)["drainage_flux"]

    assert _scalar(low) == pytest.approx(0.0, abs=0.0)
    assert _scalar(high) > _scalar(low)


def test_vertical_percolation_matches_reduced_order_vgm_formula():
    model = VerticalPercolationTransport(name="vp_flux_numerics")
    theta = torch.tensor(0.40, dtype=torch.float32)
    outputs = model.forward(_state(float(theta)), time=0.0, dt=60.0)

    theta_r = theta.new_tensor(model.theta_r_value)
    theta_s = theta.new_tensor(model.theta_s_value)
    theta_fc = theta.new_tensor(model.theta_fc_value)
    n_vg = theta.new_tensor(model.vg_n_value)
    l_vg = theta.new_tensor(model.vg_l_value)
    m_vg = 1.0 - 1.0 / n_vg
    effective_saturation = (theta - theta_r) / (theta_s - theta_r)
    relative_conductivity = effective_saturation.pow(l_vg) * (
        1.0 - (1.0 - effective_saturation.pow(1.0 / m_vg)).pow(m_vg)
    ).pow(2.0)
    expected = (
        torch.clamp(theta - theta_fc, min=0.0)
        / theta.new_tensor(model.drainage_time_constant)
        * relative_conductivity
        * theta.new_tensor(model.water_density)
        * theta.new_tensor(model.soil_layer_thickness)
    )

    assert torch.allclose(outputs["drainage_flux"].value, expected, atol=1.0e-10, rtol=1.0e-6)
    assert torch.allclose(
        outputs["diagnostics.vertical_percolation.water_balance_residual"].value,
        torch.zeros_like(expected),
        atol=0.0,
        rtol=0.0,
    )


def test_vertical_percolation_boundary_checker_fails_closed():
    model = VerticalPercolationTransport(name="vp_boundary_checker")
    state = _state(0.40)
    outputs = model.forward(state, time=0.0, dt=60.0)

    assert model.check_conservation_laws(state, outputs) == {
        "vertical_drainage_boundary": pytest.approx(0.0, abs=0.0)
    }

    missing = dict(outputs)
    missing.pop("diagnostics.vertical_percolation.water_balance_residual")
    with pytest.raises(KeyError, match="water_balance_residual"):
        model.check_conservation_laws(state, missing)

    nonfinite = dict(outputs)
    item = outputs["diagnostics.vertical_percolation.water_balance_residual"]
    nonfinite["diagnostics.vertical_percolation.water_balance_residual"] = UpdateItem(
        value=torch.tensor(float("nan")),
        mode=item.mode,
        per_timestep=item.per_timestep,
        unit=item.unit,
        source_functor=item.source_functor,
    )
    with pytest.raises(ValueError, match="finite"):
        model.check_conservation_laws(state, nonfinite)

    tampered = dict(outputs)
    drainage = outputs["drainage_flux"]
    tampered["drainage_flux"] = UpdateItem(
        value=drainage.value + 1.0e-3,
        mode=drainage.mode,
        per_timestep=drainage.per_timestep,
        unit=drainage.unit,
        source_functor=drainage.source_functor,
    )
    assert model.check_conservation_laws(
        state,
        tampered,
    )["vertical_drainage_boundary"] == pytest.approx(1.0e-3)


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ({"soil_temperature": torch.tensor(298.15)}, "soil_moisture"),
        ({"soil_moisture": torch.tensor(0.30)}, "soil_temperature"),
        (_state(float("nan")), "finite"),
        (_state(0.40, float("inf")), "finite"),
        # Above theta_s = hydrosphere.soil_moisture.porosity (0.45 m3 m-3).
        (_state(0.46), "physical range"),
    ],
)
def test_vertical_percolation_rejects_missing_nonfinite_or_out_of_range_inputs(
    state: dict[str, torch.Tensor], match: str
):
    model = VerticalPercolationTransport(name="vp_input_validation")

    with pytest.raises((ValueError, RuntimeError), match=match):
        model.forward(state, time=0.0, dt=60.0)


@pytest.mark.parametrize(
    ("key", "value", "error_type"),
    [
        ("soil_moisture", 0.30, TypeError),
        ("soil_temperature", 298.15, TypeError),
        ("soil_moisture", torch.tensor([]), ValueError),
        ("soil_temperature", torch.tensor([]), ValueError),
        ("soil_moisture", torch.tensor(0), TypeError),
        ("soil_temperature", torch.tensor(298), TypeError),
    ],
)
def test_vertical_percolation_requires_nonempty_floating_state_tensors(
    key: str,
    value: object,
    error_type: type[Exception],
):
    model = VerticalPercolationTransport(name="vp_tensor_contract")
    state = _state(0.30)
    state[key] = value

    with pytest.raises(error_type, match=key):
        model.forward(state, time=0.0, dt=60.0)


def test_vertical_percolation_satisfies_phase8_iphs_protocol():
    model = VerticalPercolationTransport(name="vp_phase8_protocol")

    forms = model.declared_forms()
    hodge_terms = model.build_hodge_terms()
    dissipative = model.dissipative_structure({})
    ports = model.ports()

    assert isinstance(model, IPHSDECFunctor)
    assert model.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert "hydrosphere.transport.vertical_percolation.hydraulic_model" in model.required_parameters()
    assert not any(key.startswith("relic.physical.constants") for key in model.required_parameters())
    assert {form.semantic_name for form in forms} == {
        "soil_moisture",
        "vertical_hydraulic_gradient",
        "drainage_flux",
    }
    assert len(hodge_terms) == 1
    assert hodge_terms[0].metric_keys == (
        "hydrosphere.transport.soil_layer_thickness",
    )
    assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
    assert {port.name for port in ports} == {"vertical_drainage_boundary_flux"}
    assert {port.flow_var for port in ports} == {"drainage_flux"}
    assert model.get_dependencies() == {"required": [], "optional": []}


def test_vertical_percolation_phase8_certificate_is_valid():
    certificate = load_iphs_equivalence_certificate(
        _certificate_path(
            "hydrosphere_transport_vertical_percolation_vertical_percolation_transport.yaml"
        )
    )
    audit = certificate.validate()

    assert audit["passed"] is True
    assert (
        certificate.process_id
        == "relic.scientific.hydrosphere.transport.vertical_percolation.VerticalPercolationTransport"
    )
    assert certificate.canonical_role == "transport"


def test_vertical_percolation_dt_is_positive_and_does_not_rescale_flux():
    model = VerticalPercolationTransport(name="vp_dt_contract")
    short = model.forward(_state(0.40), time=0.0, dt=60.0)["drainage_flux"].value
    long = model.forward(_state(0.40), time=0.0, dt=3600.0)["drainage_flux"].value

    assert torch.equal(short, long)
    with pytest.raises((ValueError, RuntimeError), match="dt"):
        model.forward(_state(0.40), time=0.0, dt=0.0)


def test_vertical_percolation_real_simulator_boundary_flux_provenance():
    model = VerticalPercolationTransport(name="vertical_percolation_transport")
    state = _state(0.40)
    initial_moisture = state["soil_moisture"].clone()
    direct = model.forward(state, time=0.0, dt=3600.0)
    simulator = RELICSimulator(
        {
            "max_time": 3600.0,
            "time_step": 3600.0,
            "conservation_checks": False,
            "strict_dependencies": False,
            "device": "cpu",
        }
    )
    simulator.initialize_state(state)
    simulator.register_functor(model, execution_group="fast")

    simulator.step(dt=3600.0)

    final_state = simulator.state_manager.get_state()
    actual = final_state["drainage_flux"]
    expected = direct["drainage_flux"].value.to(
        device=actual.device,
        dtype=actual.dtype,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(final_state["soil_moisture"], initial_moisture)
    assert simulator.get_effective_device(model.name).type == "cpu"
    provenance = simulator.get_variable_provenance("drainage_flux")
    assert {entry["functor"] for entry in provenance} == {model.name}
    assert {entry["role"] for entry in provenance} == {"writer", "final_writer"}
