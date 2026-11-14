"""Integration test for dirty execution with caching."""

import torch

from relic.orchestration.cache import NodeCache
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
    input_port = PortSpec(
        name="x",
        standard_name="x",
        units="1",
        semantics="state",
        spatial_kind="grid_cell",
        area_basis="ground_area",
        temporal_basis="instantaneous",
        vertical_basis="none",
    )
    output_port = PortSpec(
        name="x",
        standard_name="x",
        units="1",
        semantics="state",
        spatial_kind="grid_cell",
        area_basis="ground_area",
        temporal_basis="instantaneous",
        vertical_basis="none",
    )
    definition.inputs["x"] = input_port
    definition.outputs["x"] = output_port
    registry.register_definition(CLASS_TYPE, definition)
    return registry


def _build_graph() -> GraphIR:
    nodes = {
        "n1": NodeSpec(
            class_type=CLASS_TYPE,
            inputs={"x": {"state": "x"}},
            outputs=["x"],
            config={
                "cache_policy": "deterministic_pure",
                "input_name": "x",
            },
        )
    }
    return GraphIR(schema_version="1.0.0", nodes=nodes, links=[], metadata={})


def test_dirty_execution_with_cache(tmp_path) -> None:
    registry = _build_registry()
    graph = _build_graph()
    plan = compile_graph(graph, registry)

    cache = NodeCache(tmp_path / "cache")
    executor = GraphExecutor(
        plan,
        cache=cache,
        registry=registry,
        strategy=ExecutionStrategy.DIRTY,
    )

    result = executor.execute({"x": torch.tensor(0.0)}, n_steps=2, dt=1.0)
    assert float(result["x"].item()) == 2.0
    assert any((tmp_path / "cache").iterdir())
