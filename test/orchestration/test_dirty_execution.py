"""Unit tests for dirty execution strategy in GraphExecutor."""

import torch

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import ExecutionStrategy, GraphExecutor
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog

CLASS_TYPE = (
    "Functor:test.fixtures.dummy_functor.DummyPureFieldIncrementFunctor"
)


def _build_registry() -> NodeRegistry:
    registry = NodeRegistry(SSOTCatalog())
    definition = registry.build_from_node_spec(CLASS_TYPE, ["x"], ["x"])
    definition.inputs["x"] = PortSpec(
        name="x",
        standard_name="x",
        units="1",
    )
    definition.outputs["x"] = PortSpec(
        name="x",
        standard_name="x",
        units="1",
    )
    registry.register_definition(CLASS_TYPE, definition)
    return registry


def _build_graph() -> GraphIR:
    nodes = {
        "n1": NodeSpec(
            class_type=CLASS_TYPE,
            inputs={"x": {"state": "x"}},
            outputs=["x"],
            config={
                "input_name": "x",
                "cache_policy": "deterministic_pure",
            },
        )
    }
    return GraphIR(schema_version="1.0.0", nodes=nodes, links=[], metadata={})


def test_dirty_tracking_updates_hashes() -> None:
    registry = _build_registry()
    graph = _build_graph()
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(plan, registry=registry, strategy=ExecutionStrategy.DIRTY)

    executor.execute({"x": torch.tensor(0.0)}, n_steps=1, dt=1.0)
    assert len(executor._last_input_hashes) == 1


def test_is_node_dirty_respects_state_and_time() -> None:
    registry = _build_registry()
    graph = _build_graph()
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(plan, registry=registry, strategy=ExecutionStrategy.DIRTY)
    step = plan.steps[0]

    state = {"x": torch.tensor(0.0)}
    dt = torch.tensor(1.0)
    executor.execute(state, n_steps=0, dt=1.0)

    executor._update_dirty_tracking("n1", step, state, time=0.0, dt=dt)
    assert executor._is_node_dirty("n1", step, state, time=0.0, dt=dt) is False

    changed_state = {"x": torch.tensor(1.0)}
    assert executor._is_node_dirty("n1", step, changed_state, time=0.0, dt=dt) is True

    assert executor._is_node_dirty("n1", step, state, time=1.0, dt=dt) is True


def test_mark_dirty_forces_execution() -> None:
    registry = _build_registry()
    graph = _build_graph()
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(plan, registry=registry, strategy=ExecutionStrategy.DIRTY)

    executor.mark_dirty("n1")
    assert "n1" in executor._dirty_nodes


def test_set_strategy_marks_all_dirty() -> None:
    registry = _build_registry()
    graph = _build_graph()
    plan = compile_graph(graph, registry)
    executor = GraphExecutor(plan, registry=registry, strategy=ExecutionStrategy.PUSH)

    executor.set_strategy(ExecutionStrategy.DIRTY)
    assert len(executor._dirty_nodes) == len(plan.steps)
