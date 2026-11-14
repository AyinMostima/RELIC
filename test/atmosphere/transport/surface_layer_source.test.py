"""Surface-layer source selection of the boundary-layer transport processes.

The default keeps atmospheric_turbulence as a required provider. The selection external_forcing
removes only that dependency: friction velocity, Obukhov length and aerodynamic resistance then
arrive through State:ExternalForcing, for example as observed tower series, and the law is unchanged.
Units: mixing ratios mol mol-1, r_a s m-1, h m, dt s.
"""

import pytest
import torch

from relic.core.dependency_graph import normalize_dependency_declaration
from relic.core.state.manager import StateManager
from relic.orchestration import (
    GraphExecutor,
    GraphIR,
    NodeRegistry,
    NodeSpec,
    SSOTCatalog,
    compile_graph,
)
from relic.parameters import get_global_manager
from relic.scientific.atmosphere.transport.dry_deposition_aggregator import (
    DryDepositionAggregator,
)
from relic.scientific.atmosphere.transport.eddy_diffusion_blm import (
    EddyDiffusionBLMFunctor,
)

DT = 20.0  # s
FORCING = "external_forcing"


def _dry_deposition_state() -> dict[str, torch.Tensor]:
    return {
        "no_mixing_ratio": torch.tensor([2.0e-10, 4.0e-10], dtype=torch.float64),
        "no2_mixing_ratio": torch.tensor([6.0e-10, 1.0e-9], dtype=torch.float64),
        "o3_mixing_ratio": torch.tensor([3.0e-8, 5.0e-8], dtype=torch.float64),
        "so2_mixing_ratio": torch.tensor([5.0e-10, 5.0e-10], dtype=torch.float64),
        "pbl_height": torch.tensor([800.0, 1200.0], dtype=torch.float64),
        "aerodynamic_resistance": torch.tensor([20.0, 45.0], dtype=torch.float64),
    }


def _eddy_diffusion_state() -> dict[str, torch.Tensor]:
    return {
        "no_mixing_ratio": torch.tensor([2.0e-10, 4.0e-10], dtype=torch.float64),
        "no2_mixing_ratio": torch.tensor([6.0e-10, 1.0e-9], dtype=torch.float64),
        "o3_mixing_ratio": torch.tensor([3.0e-8, 5.0e-8], dtype=torch.float64),
        "isoprene_mixing_ratio": torch.tensor([2.0e-10, 8.0e-10], dtype=torch.float64),
        "pbl_height": torch.tensor([800.0, 1200.0], dtype=torch.float64),
        "friction_velocity": torch.tensor([0.3, 0.6], dtype=torch.float64),
        "monin_obukhov_length": torch.tensor([-120.0, 80.0], dtype=torch.float64),
    }


CASES = {
    "dry_deposition_aggregator": (DryDepositionAggregator, _dry_deposition_state),
    "eddy_diffusion_blm": (EddyDiffusionBLMFunctor, _eddy_diffusion_state),
}


@pytest.fixture
def transport_enabled():
    """The eddy process is opt-in: atmosphere.transport.enabled is false in the database."""
    manager = get_global_manager()
    original = manager.get("atmosphere.transport.enabled")
    manager.set_value("atmosphere.transport.enabled", True)
    yield
    manager.set_value("atmosphere.transport.enabled", original)


def _graph(node_id, cls, state, config, forcing_keys=None):
    node_type = f"Functor:{cls.__module__}.{cls.__name__}"
    functor = cls(**config) if config else cls(config={})
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    registry.register_functor_instance(functor, node_type_id=node_type)
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "forcing": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=list(state if forcing_keys is None else forcing_keys),
            ),
            node_id: NodeSpec(
                class_type=node_type,
                config=dict(config),
                inputs={name: {"state": name} for name in functor.metadata.input_variables},
                outputs=[
                    name
                    for name in functor.metadata.output_variables
                    if not name.startswith("diagnostics.")
                ],
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


@pytest.mark.parametrize("node_id", list(CASES))
def test_default_keeps_the_required_turbulence_provider(node_id):
    cls, _ = CASES[node_id]
    assert cls().surface_layer_source == "atmospheric_turbulence"
    assert cls().get_dependencies() == ["atmospheric_turbulence"]
    assert cls(config={}).get_dependencies() == ["atmospheric_turbulence"]


@pytest.mark.parametrize("node_id", list(CASES))
def test_forcing_selection_removes_only_the_turbulence_dependency(node_id):
    cls, _ = CASES[node_id]
    declared = normalize_dependency_declaration(
        cls(surface_layer_source=FORCING).get_dependencies(), owner=node_id
    )
    assert declared.required == () and declared.optional == () and declared.lagged == ()
    kept = cls(config={"dependencies": ["another_provider"]}, surface_layer_source=FORCING)
    assert kept.get_dependencies() == ["another_provider"]


@pytest.mark.parametrize("node_id", list(CASES))
def test_unknown_selection_is_refused(node_id):
    cls, _ = CASES[node_id]
    with pytest.raises(ValueError, match="surface_layer_source"):
        cls(surface_layer_source="tower")


@pytest.mark.parametrize("node_id", list(CASES))
def test_strict_compile_still_refuses_the_default_without_a_turbulence_provider(node_id):
    cls, make_state = CASES[node_id]
    graph, registry = _graph(node_id, cls, make_state(), {})
    with pytest.raises(RuntimeError, match=f"{node_id} -> atmospheric_turbulence"):
        compile_graph(graph, registry, strict=True, auto_adapt=False)


def test_dry_deposition_forcing_graph_equals_the_default_law_for_two_governed_steps():
    state = _dry_deposition_state()
    graph, registry = _graph(
        "dry_deposition_aggregator",
        DryDepositionAggregator,
        state,
        {"surface_layer_source": FORCING},
    )
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    governed = _executor(plan, registry).execute(dict(state), n_steps=2, dt=DT)

    kernel = DryDepositionAggregator()
    expected = dict(state)
    for _ in range(2):
        output = kernel.forward(expected, 0.0, DT)
        for name in ("no_mixing_ratio", "no2_mixing_ratio", "o3_mixing_ratio", "so2_mixing_ratio"):
            expected[name] = expected[name] + output[name].value
    for name in ("no_mixing_ratio", "no2_mixing_ratio", "o3_mixing_ratio", "so2_mixing_ratio"):
        torch.testing.assert_close(governed[name], expected[name], rtol=1.0e-14, atol=0.0)
        assert bool(torch.all(governed[name] < state[name]))


def test_dry_deposition_missing_forcing_is_refused_before_any_step_without_a_default():
    state = _dry_deposition_state()
    keys = [name for name in state if name != "aerodynamic_resistance"]
    graph, registry = _graph(
        "dry_deposition_aggregator",
        DryDepositionAggregator,
        state,
        {"surface_layer_source": FORCING},
        forcing_keys=keys,
    )
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    with pytest.raises(ValueError, match="Missing required initial state keys: aerodynamic_resistance"):
        _executor(plan, registry).execute({name: state[name] for name in keys}, n_steps=1, dt=DT)


@pytest.mark.parametrize("node_id", list(CASES))
def test_selection_is_reported_in_the_transport_metadata(node_id):
    cls, _ = CASES[node_id]
    assert cls(surface_layer_source=FORCING).get_transport_metadata()["surface_layer_source"] == FORCING
    assert cls().get_transport_metadata()["surface_layer_source"] == "atmospheric_turbulence"


def test_eddy_forcing_graph_equals_the_default_law_for_two_governed_steps(transport_enabled):
    state = _eddy_diffusion_state()
    graph, registry = _graph(
        "eddy_diffusion_blm",
        EddyDiffusionBLMFunctor,
        state,
        {"surface_layer_source": FORCING},
    )
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    governed = _executor(plan, registry).execute(dict(state), n_steps=2, dt=DT)

    kernel = EddyDiffusionBLMFunctor()
    species = ("no_mixing_ratio", "no2_mixing_ratio", "o3_mixing_ratio", "isoprene_mixing_ratio")
    expected = dict(state)
    for _ in range(2):
        output = kernel.forward(expected, 0.0, DT)
        for name in species:
            expected[name] = expected[name] + output[name].value
    for name in species:
        torch.testing.assert_close(governed[name], expected[name], rtol=1.0e-14, atol=0.0)
        assert not torch.equal(governed[name], state[name])


def test_eddy_missing_forcing_is_refused_before_any_step_without_a_default(transport_enabled):
    state = _eddy_diffusion_state()
    keys = [name for name in state if name != "friction_velocity"]
    graph, registry = _graph(
        "eddy_diffusion_blm",
        EddyDiffusionBLMFunctor,
        state,
        {"surface_layer_source": FORCING},
        forcing_keys=keys,
    )
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    with pytest.raises(ValueError, match="Missing required initial state keys: friction_velocity"):
        _executor(plan, registry).execute({name: state[name] for name in keys}, n_steps=1, dt=DT)
