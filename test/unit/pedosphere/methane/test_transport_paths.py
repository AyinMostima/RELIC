import torch

from pathlib import Path

import pytest
import yaml

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor, VariableKind
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.methane import MethaneTransportFunctor


CERTIFICATE_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "pedosphere_methane_transport_methane_transport_functor.yaml"
)

CANONICAL_INPUTS = {
    "ch4_soil_storage",
    "water_filled_pore_space",
    "porosity",
    "atmospheric_ch4_concentration",
}

CANONICAL_OUTPUTS = {
    "ch4_transport_loss_rate",
    "ch4_storage_tendency_trans",
    "diagnostics.methane_transport.diffusion_rate",
    "diagnostics.methane_transport.ebullition_rate",
    "diagnostics.methane_transport.plant_flux_rate",
}


def canonical_state(**overrides):
    state = {
        "ch4_soil_storage": torch.tensor(3.0, dtype=torch.float64),
        "water_filled_pore_space": torch.tensor(0.4, dtype=torch.float64),
        "porosity": torch.tensor(0.45, dtype=torch.float64),
        "atmospheric_ch4_concentration": torch.tensor(0.0, dtype=torch.float64),
    }
    state.update(overrides)
    return state


def apply_updates(state, updates):
    for key, value in updates.items():
        mode = UpdateMode.ACCUMULATE
        raw_value = value
        if isinstance(value, UpdateItem):
            raw_value = value.value
            mode = value.mode
        tensor = raw_value if isinstance(raw_value, torch.Tensor) else torch.tensor(float(raw_value))
        tensor = tensor.to(torch.float32)
        if key in state and mode != UpdateMode.REPLACE:
            state[key] = state[key] + tensor
        else:
            state[key] = tensor.clone()
    return state


def update_value(item):
    if isinstance(item, UpdateItem):
        return item.value
    return item


def run_functor(functor, state, dt=1.0):
    updates = functor(state, time=0.0, dt=torch.tensor(float(dt)))
    return apply_updates(state, updates)


def test_diffusion_increases_with_air_space():
    functor = MethaneTransportFunctor()

    porous_state = {
        "ch4_soil_storage": torch.tensor(3.0),
        "water_filled_pore_space": torch.tensor(0.4),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
    }
    saturated_state = {
        "ch4_soil_storage": torch.tensor(3.0),
        "water_filled_pore_space": torch.tensor(0.95),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
    }

    run_functor(functor, porous_state, dt=3600.0)
    run_functor(functor, saturated_state, dt=3600.0)

    diff_porous = porous_state[
        "diagnostics.methane_transport.diffusion_rate"
    ].item()
    diff_saturated = saturated_state[
        "diagnostics.methane_transport.diffusion_rate"
    ].item()

    assert diff_porous > diff_saturated


def test_diffusion_uses_declared_millington_quirk_porosity_terms():
    functor = MethaneTransportFunctor()
    state = canonical_state()

    updates = functor.forward(state, time=0.0, dt=torch.tensor(3600.0))

    porosity = state["porosity"]
    wfps = state["water_filled_pore_space"]
    to_state_dtype = lambda value: value.to(
        device=porosity.device,
        dtype=porosity.dtype,
    )
    air_filled_porosity = porosity * (torch.ones_like(wfps) - wfps)
    air_component = (
        to_state_dtype(functor.diffusivity_air)
        * torch.pow(
            air_filled_porosity,
            to_state_dtype(functor.millington_quirk_exponent_numerator),
        )
        / torch.pow(
            porosity,
            to_state_dtype(functor.millington_quirk_exponent_denominator),
        )
    )
    gas_diffusivity = to_state_dtype(functor.tortuosity_factor) * air_component
    water_diffusivity = to_state_dtype(functor.diffusivity_water) * wfps
    layer_depth = to_state_dtype(functor.layer_depth)
    concentration = state["ch4_soil_storage"] / layer_depth
    expected = (
        (gas_diffusivity + water_diffusivity)
        * (concentration - state["atmospheric_ch4_concentration"])
        / layer_depth
    )
    actual = updates["diagnostics.methane_transport.diffusion_rate"].value

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-18)


def test_ebullition_triggers_above_threshold():
    functor = MethaneTransportFunctor()

    below_threshold = {
        "ch4_soil_storage": torch.tensor(0.5),
        "water_filled_pore_space": torch.tensor(0.9),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
    }
    above_threshold = {
        "ch4_soil_storage": torch.tensor(5.0),
        "water_filled_pore_space": torch.tensor(0.95),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
    }

    run_functor(functor, below_threshold, dt=3600.0)
    run_functor(functor, above_threshold, dt=3600.0)

    ebb_low = below_threshold[
        "diagnostics.methane_transport.ebullition_rate"
    ].item()
    ebb_high = above_threshold[
        "diagnostics.methane_transport.ebullition_rate"
    ].item()

    assert ebb_high > ebb_low


def test_phase8_iphs_structure_hooks_are_declared():
    functor = MethaneTransportFunctor()
    state = {
        "ch4_soil_storage": torch.tensor(3.0),
        "water_filled_pore_space": torch.tensor(0.75),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
    }

    forms = functor.declared_forms()
    hodge_terms = functor.build_hodge_terms()
    dissipative = functor.dissipative_structure(state)
    ports = functor.ports()

    assert isinstance(functor, IPHSDECFunctor)
    assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert "pedosphere.methane.transport.diffusivity_air" in functor.required_parameters()
    assert "pedosphere.methane.transport.porosity_shape_factor" not in (
        functor.required_parameters()
    )
    parameter_database = yaml.safe_load(
        (
            Path(__file__).resolve().parents[4]
            / "relic"
            / "parameters"
            / "database"
            / "relic"
            / "pedosphere"
            / "methane.yaml"
        ).read_text(encoding="utf-8")
    )
    assert "pedosphere.methane.transport.porosity_shape_factor" not in (
        parameter_database["parameters"]
    )
    assert not any(
        key.startswith("relic.physical.constants")
        for key in functor.required_parameters()
    )
    assert {form.semantic_name for form in forms} == {
        "ch4_soil_storage",
        "ch4_transport_gradient",
        "ch4_transport_flux",
    }
    assert len(hodge_terms) == 1
    assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
    assert {port.name for port in ports} == {"methane_transport_request"}
    assert ports[0].flow_var == "ch4_transport_loss_rate"


def test_transport_reports_raw_request_for_shared_budget_limiter():
    functor = MethaneTransportFunctor()
    state = canonical_state()

    updates = functor(state, time=0.0, dt=torch.tensor(3600.0))

    assert "diagnostics.methane_transport.clipping_residual" not in updates
    requested = update_value(updates["ch4_transport_loss_rate"])
    assert requested.item() > state["ch4_soil_storage"].item() / 3600.0


def test_phase8_certificate_level3_declaration_is_valid():
    certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
    audit = certificate.validate()

    assert audit["passed"] is True
    assert certificate.process_id == (
        "relic.scientific.pedosphere.methane.transport.MethaneTransportFunctor"
    )
    assert certificate.canonical_role == "transport"
    assert certificate.iphs_level == 3
    assert certificate.iphs_level_evidence["level_3_limit"]
    assert audit["runtime_evidence_status"] == "descriptive_only"


def test_metadata_and_runtime_use_only_canonical_transport_ports():
    functor = MethaneTransportFunctor()
    updates = functor.forward(
        canonical_state(), time=0.0, dt=torch.tensor(3600.0)
    )

    assert set(functor.metadata.input_variables) == CANONICAL_INPUTS
    assert set(functor.metadata.output_variables) == CANONICAL_OUTPUTS
    assert functor.metadata.variable_kind is VariableKind.FLUX
    assert functor.get_dependencies() == []
    assert set(updates) == CANONICAL_OUTPUTS
    for removed_name in (
        "ch4_diffusion_rate",
        "ch4_ebullition_rate",
        "ch4_plant_flux_rate",
    ):
        assert removed_name not in updates


def test_transport_outputs_have_explicit_update_semantics_and_local_closure():
    functor = MethaneTransportFunctor()
    updates = functor.forward(
        canonical_state(),
        time=0.0,
        dt=torch.tensor(3600.0, dtype=torch.float64),
    )

    for name, item in updates.items():
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == "g C m-2 s-1"
        assert item.source_functor == "pedosphere_methane_transport"
        assert item.element is None

    component_sum = (
        updates["diagnostics.methane_transport.diffusion_rate"].value
        + updates["diagnostics.methane_transport.ebullition_rate"].value
        + updates["diagnostics.methane_transport.plant_flux_rate"].value
    )
    torch.testing.assert_close(
        updates["ch4_transport_loss_rate"].value,
        component_sum,
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        updates["ch4_storage_tendency_trans"].value,
        -updates["ch4_transport_loss_rate"].value,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("missing_name", sorted(CANONICAL_INPUTS))
def test_required_transport_inputs_fail_fast_when_missing(missing_name):
    functor = MethaneTransportFunctor()
    state = canonical_state()
    state.pop(missing_name)

    with pytest.raises(KeyError, match=missing_name):
        functor.forward(state, time=0.0, dt=torch.tensor(60.0))


@pytest.mark.parametrize("input_name", sorted(CANONICAL_INPUTS))
def test_required_transport_inputs_must_be_tensors(input_name):
    functor = MethaneTransportFunctor()
    state = canonical_state(**{input_name: 0.5})

    with pytest.raises(TypeError, match=input_name):
        functor.forward(state, time=0.0, dt=torch.tensor(60.0))


@pytest.mark.parametrize(
    ("input_name", "invalid_value"),
    [
        ("ch4_soil_storage", -1.0),
        ("ch4_soil_storage", float("nan")),
        ("water_filled_pore_space", -0.1),
        ("water_filled_pore_space", 1.1),
        ("water_filled_pore_space", float("nan")),
        ("porosity", 0.0),
        ("porosity", 1.1),
        ("porosity", float("nan")),
        ("atmospheric_ch4_concentration", -1.0),
        ("atmospheric_ch4_concentration", float("nan")),
    ],
)
def test_transport_input_domains_fail_fast(input_name, invalid_value):
    functor = MethaneTransportFunctor()
    state = canonical_state(
        **{input_name: torch.tensor(invalid_value, dtype=torch.float64)}
    )

    with pytest.raises(ValueError, match=input_name):
        functor.forward(state, time=0.0, dt=torch.tensor(60.0))


@pytest.mark.parametrize(
    "invalid_dt",
    [torch.tensor(0.0), torch.tensor(float("nan")), torch.tensor([1.0, 2.0])],
)
def test_transport_dt_must_be_one_positive_finite_scalar(invalid_dt):
    functor = MethaneTransportFunctor()

    with pytest.raises(ValueError, match="dt"):
        functor.forward(canonical_state(), time=0.0, dt=invalid_dt)


def test_transport_forward_does_not_mutate_input_state():
    functor = MethaneTransportFunctor()
    state = canonical_state()
    baseline = {key: value.clone() for key, value in state.items()}

    functor.forward(state, time=0.0, dt=torch.tensor(60.0))

    assert set(state) == set(baseline)
    for key, value in baseline.items():
        assert torch.equal(state[key], value)


def test_hodge_degrees_of_freedom_come_from_the_declared_layout():
    """Round 47: no literal None fallback; the nominal Hodge acts on the declared
    storage cochain."""
    import inspect

    parameter = inspect.signature(MethaneTransportFunctor.build_hodge_contract).parameters["n_dofs"]
    assert parameter.default is inspect.Parameter.empty
    functor = MethaneTransportFunctor()
    n_state = functor.declared_forms()[0].n_dofs
    (hodge,) = functor.build_hodge_terms()
    assert tuple(hodge.matrix.shape) == (n_state, n_state)
    assert tuple(functor.dissipative_structure({})["M"].shape) == (n_state, n_state)
