"""Boundary-layer emission source: law, ledger, contract and governed execution.

References: Stull (1988), mixed-layer budget; CMAQ User's Guide (2019) emissions conversion.
Mathematical expression: Delta chi = F dt R T / (P h), with n = P / (R T).
Units: F [mol m-2 s-1], h [m], T [K], P [Pa], dt [s], chi [mol mol-1].
"""

import pytest
import torch

from relic.contracts.contract_loader import load_contract_registry
from relic.core.dependency_graph import normalize_dependency_declaration
from relic.core.state import UpdateItem, UpdateMode
from relic.core.state.manager import StateManager
from relic.orchestration import (
    GraphExecutor,
    GraphIR,
    NodeRegistry,
    NodeSpec,
    SSOTCatalog,
    compile_graph,
)
from relic.scientific.anthroposphere.emissions.species_mapping import SpeciesMapper
from relic.scientific.atmosphere.transport.boundary_layer_emission_source import (
    BALANCE_RESIDUAL,
    BoundaryLayerEmissionSource,
)
from relic.utils.physical_constants import get_physical_constant

DT = 20.0  # s


def _state() -> dict[str, torch.Tensor]:
    return {
        "isoprene_emission": torch.tensor([0.0, 2.0e-9, 1.0e-8], dtype=torch.float64),
        "pbl_height": torch.tensor([300.0, 1000.0, 1500.0], dtype=torch.float64),
        "air_temperature": torch.tensor([285.0, 295.0, 300.0], dtype=torch.float64),
        "atmospheric_pressure": torch.tensor([9.8e4, 1.0e5, 1.01e5], dtype=torch.float64),
    }


def test_increment_is_flux_over_column_air_burden():
    state = _state()
    output = BoundaryLayerEmissionSource()(state, time=0.0, dt=DT)
    item = output["isoprene_mixing_ratio"]
    assert isinstance(item, UpdateItem)
    assert item.mode == UpdateMode.ACCUMULATE and item.per_timestep is True
    assert item.unit == "mol mol-1" and item.source_functor == "boundary_layer_emission_source"

    gas_constant = get_physical_constant("gas_constant")
    molar_density = state["atmospheric_pressure"] / (gas_constant * state["air_temperature"])
    expected = state["isoprene_emission"] * DT / (state["pbl_height"] * molar_density)
    torch.testing.assert_close(item.value, expected, rtol=1.0e-14, atol=0.0)
    # 1e-8 mol m-2 s-1 into 1500 m of air (h n = 6.07e4 mol m-2) adds 3.3e-12 mol mol-1 in 20 s.
    assert 3.2e-12 < float(item.value[2]) < 3.4e-12


def test_the_law_is_the_existing_species_mapper_helper():
    state = _state()
    output = BoundaryLayerEmissionSource().forward(state, 0.0, DT)
    direct = SpeciesMapper.flux_to_mixing_ratio(
        state["isoprene_emission"],
        state["atmospheric_pressure"],
        state["air_temperature"],
        state["pbl_height"],
        DT,
    )
    assert torch.equal(output["isoprene_mixing_ratio"].value, direct)


def test_column_ledger_closes_and_a_broken_increment_is_reported():
    source = BoundaryLayerEmissionSource()
    state = _state()
    output = source.forward(state, 0.0, DT)
    assert source.check_conservation_laws(state, output)["mass"] < 1.0e-14
    assert float(output[BALANCE_RESIDUAL][0]) == 0.0  # zero flux, no division by zero

    broken = dict(output)
    broken[BALANCE_RESIDUAL] = output[BALANCE_RESIDUAL] + 0.1
    assert source.check_conservation_laws(state, broken)["mass"] == pytest.approx(0.1)
    with pytest.raises(KeyError):
        source.check_conservation_laws(state, {})


def test_mixed_precision_inputs_keep_the_widest_dtype():
    state = _state()
    state["isoprene_emission"] = state["isoprene_emission"].to(torch.float32)
    increment = BoundaryLayerEmissionSource().forward(state, 0.0, DT)["isoprene_mixing_ratio"].value
    assert increment.dtype == torch.float64


def test_gradient_reaches_the_flux():
    state = _state()
    state["isoprene_emission"] = state["isoprene_emission"].clone().requires_grad_(True)
    increment = BoundaryLayerEmissionSource().forward(state, 0.0, DT)["isoprene_mixing_ratio"].value
    increment.sum().backward()
    assert bool(torch.all(state["isoprene_emission"].grad > 0.0))


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("isoprene_emission", -1.0e-10, ValueError),
        ("isoprene_emission", float("nan"), ValueError),
        ("pbl_height", 0.0, ValueError),
        ("air_temperature", 0.0, ValueError),
        ("atmospheric_pressure", float("inf"), ValueError),
    ],
)
def test_invalid_inputs_raise(name, value, error):
    state = _state()
    state[name] = torch.full_like(state[name], value)
    with pytest.raises(error):
        BoundaryLayerEmissionSource().forward(state, 0.0, DT)


@pytest.mark.parametrize("name", list(BoundaryLayerEmissionSource._REQUIRED_INPUTS))
def test_missing_input_raises_without_a_default(name):
    state = _state()
    del state[name]
    with pytest.raises(KeyError):
        BoundaryLayerEmissionSource().forward(state, 0.0, DT)


def test_non_positive_step_is_refused():
    with pytest.raises(ValueError):
        BoundaryLayerEmissionSource().forward(_state(), 0.0, 0.0)


def test_emission_providers_are_optional_same_step_dependencies():
    declared = normalize_dependency_declaration(
        BoundaryLayerEmissionSource().get_dependencies(),
        owner="boundary_layer_emission_source",
    )
    assert set(declared.optional) == {"megan_bvoc_model", "mechanistic_isoprene_synthesis"}
    assert declared.required == () and declared.lagged == ()


def test_contract_declares_the_writer_as_the_only_input_route():
    registry = load_contract_registry()
    assert registry.can_increment("isoprene_mixing_ratio", "boundary_layer_emission_source")
    assert not registry.can_increment("isoprene_mixing_ratio", "megan_bvoc_model")
    routes = registry.get_flux_whitelist("isoprene_mixing_ratio")
    assert routes["flux_in"] == ["boundary_layer_emission_source"]
    assert "boundary_layer_emission_source" not in routes["flux_out"]


def _compiled_executor() -> tuple[GraphExecutor, list[str]]:
    node_type = (
        f"Functor:{BoundaryLayerEmissionSource.__module__}."
        f"{BoundaryLayerEmissionSource.__name__}"
    )
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    registry.register_functor_instance(BoundaryLayerEmissionSource(), node_type_id=node_type)
    inputs = [*BoundaryLayerEmissionSource._REQUIRED_INPUTS]
    forcing = [*inputs, "isoprene_mixing_ratio"]
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "forcing": NodeSpec(class_type="State:ExternalForcing", outputs=forcing),
            "boundary_layer_emission_source": NodeSpec(
                class_type=node_type,
                config={},
                inputs={name: {"state": name} for name in inputs},
                outputs=["isoprene_mixing_ratio", BALANCE_RESIDUAL],
            ),
        },
    )
    graph.links = graph._infer_links_from_inputs()
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    executor = GraphExecutor(
        plan,
        state_manager=StateManager(max_history_length=8, device=torch.device("cpu")),
        registry=registry,
        simulator_config={"device": "cpu", "require_gpu": False},
    )
    return executor, forcing


def test_strict_graph_compiles_and_two_governed_steps_advance_the_state_once_each():
    executor, forcing = _compiled_executor()
    state = _state()
    chi0 = torch.tensor([1.0e-10, 2.0e-10, 3.0e-10], dtype=torch.float64)
    initial = {name: state[name] for name in forcing if name in state}
    initial["isoprene_mixing_ratio"] = chi0

    gas_constant = get_physical_constant("gas_constant")
    molar_density = state["atmospheric_pressure"] / (gas_constant * state["air_temperature"])
    per_step = state["isoprene_emission"] * DT / (state["pbl_height"] * molar_density)
    for n_steps, multiple in ((1, 1.0), (2, 2.0)):
        result = executor.execute(dict(initial), n_steps=n_steps, dt=DT)
        torch.testing.assert_close(
            result["isoprene_mixing_ratio"], chi0 + multiple * per_step, rtol=1.0e-13, atol=0.0
        )


def test_an_unlisted_writer_is_still_refused_by_the_contract():
    """The new route opens one name only; the source cannot be renamed around the contract."""
    registry = load_contract_registry()
    assert not registry.can_increment("isoprene_mixing_ratio", "boundary_layer_emission_source_2")
