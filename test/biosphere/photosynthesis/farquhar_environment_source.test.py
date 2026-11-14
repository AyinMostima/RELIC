"""Environment source selection of FarquharPhotosynthesis.

The default keeps environmental_forcing and atmospheric_radiation as required providers. The
selection external_forcing removes only those two declarations: vapor_pressure_deficit and ppfd then
arrive through State:ExternalForcing, for example as observed tower series, and the law is unchanged.
Units: T K, CO2 ppm, PPFD umol m-2 s-1, VPD Pa, assimilation umol CO2 m-2 s-1, dt s.
"""

import pytest
import torch

from relic.core.state import UpdateItem
from relic.core.state.manager import StateManager
from relic.orchestration import (
    GraphExecutor,
    GraphIR,
    NodeRegistry,
    NodeSpec,
    SSOTCatalog,
    compile_graph,
)
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

DT = 1800.0  # s
FORCING = "external_forcing"
INPUTS = (
    "air_temperature",
    "atmospheric_co2_ppm",
    "ppfd",
    "vapor_pressure_deficit",
    "leaf_area_index",
)
NODE_TYPE = f"Functor:{FarquharPhotosynthesis.__module__}.{FarquharPhotosynthesis.__name__}"


def _state() -> dict[str, torch.Tensor]:
    return {
        "air_temperature": torch.tensor([290.0, 299.0], dtype=torch.float64),
        "atmospheric_co2_ppm": torch.tensor([415.0, 420.0], dtype=torch.float64),
        "ppfd": torch.tensor([500.0, 1400.0], dtype=torch.float64),
        "vapor_pressure_deficit": torch.tensor([700.0, 1500.0], dtype=torch.float64),
        "leaf_area_index": torch.tensor([3.5, 5.0], dtype=torch.float64),
    }


def _graph(config, forcing_keys=INPUTS):
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    functor = FarquharPhotosynthesis(name="farquhar_photosynthesis", config={}, **config)
    registry.register_functor_instance(functor, node_type_id=NODE_TYPE)
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "forcing": NodeSpec(class_type="State:ExternalForcing", outputs=list(forcing_keys)),
            "farquhar_photosynthesis": NodeSpec(
                class_type=NODE_TYPE,
                config=dict(config),
                inputs={name: {"state": name} for name in INPUTS},
                outputs=["net_photosynthesis"],
            ),
        },
    )
    graph.links = graph._infer_links_from_inputs()
    return graph, registry


def _executor(plan, registry):
    return GraphExecutor(
        plan,
        state_manager=StateManager(max_history_length=8, device=torch.device("cpu")),
        registry=registry,
        simulator_config={"device": "cpu", "require_gpu": False, "auto_report": False},
    )


def _value(item):
    return item.value if isinstance(item, UpdateItem) else item


def test_default_keeps_both_canonical_providers():
    functor = FarquharPhotosynthesis()
    assert functor.environment_source == "canonical_providers"
    assert functor.get_dependencies() == ["environmental_forcing", "atmospheric_radiation"]


def test_forcing_selection_removes_only_the_two_providers():
    assert FarquharPhotosynthesis(environment_source=FORCING).get_dependencies() == []
    kept = FarquharPhotosynthesis(
        config={"dependencies": ["another_provider"]}, environment_source=FORCING
    )
    assert kept.get_dependencies() == ["another_provider"]


def test_unknown_selection_is_refused():
    with pytest.raises(ValueError, match="environment_source"):
        FarquharPhotosynthesis(environment_source="tower")


def test_strict_compile_still_refuses_the_default_without_its_providers():
    graph, registry = _graph({})
    with pytest.raises(RuntimeError, match="farquhar_photosynthesis -> environmental_forcing"):
        compile_graph(graph, registry, strict=True, auto_adapt=False)


def test_forcing_graph_equals_the_default_law_for_two_governed_steps():
    state = _state()
    graph, registry = _graph({"environment_source": FORCING})
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    executor = _executor(plan, registry)
    with torch.no_grad():
        result = executor.execute(dict(state), n_steps=2, dt=DT)
        expected = FarquharPhotosynthesis().forward(dict(state), 0.0, DT)
    assert executor.execution_summary["step_count"] == 2
    for name in INPUTS:
        assert torch.equal(result[name], state[name]), name
    torch.testing.assert_close(
        result["net_photosynthesis"].to(torch.float64),
        _value(expected["net_photosynthesis"]).to(torch.float64),
        rtol=1.0e-12,
        atol=0.0,
    )
    assert bool(torch.all(result["net_photosynthesis"] > 0.0))


def test_missing_forcing_is_refused_before_any_step_without_a_default():
    state = _state()
    keys = [name for name in INPUTS if name != "vapor_pressure_deficit"]
    graph, registry = _graph({"environment_source": FORCING}, forcing_keys=keys)
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    with pytest.raises(ValueError, match="Missing required initial state keys: vapor_pressure_deficit"):
        _executor(plan, registry).execute({name: state[name] for name in keys}, n_steps=1, dt=DT)
