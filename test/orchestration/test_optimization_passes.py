"""Unit tests for orchestration optimization passes."""

import pytest

from relic.orchestration.compiler import (
    ConstantFoldingPass,
    IdentityEliminationPass,
    compile_graph,
    plan_to_dict,
)
from relic.orchestration.graph_ir import GraphIR, LinkEndpoint, LinkSpec, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog


class _InjectUnknownFunctorPass:
    def __init__(self, class_type: str) -> None:
        self.class_type = class_type

    def apply(self, graph: GraphIR, registry: NodeRegistry) -> GraphIR:
        nodes = dict(graph.nodes)
        nodes["injected"] = NodeSpec(class_type=self.class_type)
        return GraphIR(
            schema_version=graph.schema_version,
            nodes=nodes,
            links=list(graph.links),
            metadata=dict(graph.metadata),
            assertion=dict(graph.assertion),
        )


class _StatefulNoOpPass:
    def __init__(self, revision: str) -> None:
        self.revision = revision

    def apply(self, graph: GraphIR, registry: NodeRegistry) -> GraphIR:
        return graph


class _AlternateNoOpPass:
    def __init__(self, revision: str) -> None:
        self.revision = revision

    def apply(self, graph: GraphIR, registry: NodeRegistry) -> GraphIR:
        return graph


class _RewriteExternalStatePass:
    def apply(self, graph: GraphIR, registry: NodeRegistry) -> GraphIR:
        nodes = dict(graph.nodes)
        source = nodes["source"]
        nodes["source"] = NodeSpec(
            class_type=source.class_type,
            inputs=dict(source.inputs),
            outputs=["optimized_x"],
            config=dict(source.config),
        )
        return GraphIR(
            schema_version=graph.schema_version,
            nodes=nodes,
            links=list(graph.links),
            metadata=dict(graph.metadata),
            assertion=dict(graph.assertion),
        )


def test_identity_elimination_updates_links_and_inputs() -> None:
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "source": NodeSpec(class_type="State:ExternalForcing", inputs={}, outputs=["x"]),
            "identity": NodeSpec(
                class_type="State:Identity",
                inputs={"x": {"link": ["source", "x"]}},
                outputs=["x"],
            ),
            "sink": NodeSpec(
                class_type="Functor:consumer",
                inputs={"x": {"link": ["identity", "x"]}},
                outputs=["y"],
            ),
        },
        links=[
            LinkSpec(
                source=LinkEndpoint(node="source", port="x"),
                target=LinkEndpoint(node="identity", port="x"),
            ),
            LinkSpec(
                source=LinkEndpoint(node="identity", port="x"),
                target=LinkEndpoint(node="sink", port="x"),
            ),
        ],
        metadata={},
    )

    opt_pass = IdentityEliminationPass()
    registry = NodeRegistry(SSOTCatalog())
    optimized = opt_pass.apply(graph, registry)

    assert "identity" not in optimized.nodes
    sink_inputs = optimized.nodes["sink"].inputs
    assert sink_inputs["x"]["link"][0] == "source"
    assert "identity_elimination" in optimized.metadata.get("optimizations_applied", [])


def test_identity_elimination_preserves_original_source_port() -> None:
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "source": NodeSpec(
                class_type="State:ExternalForcing",
                inputs={},
                outputs=["source_flux"],
            ),
            "identity": NodeSpec(
                class_type="State:Identity",
                inputs={"incoming": {"link": ["source", "source_flux"]}},
                outputs=["forwarded_flux"],
            ),
            "sink": NodeSpec(
                class_type="Functor:consumer",
                inputs={
                    "forcing": {"link": ["identity", "forwarded_flux"]}
                },
                outputs=["result"],
            ),
        },
        links=[],
        metadata={},
    )

    optimized = IdentityEliminationPass().apply(
        graph,
        NodeRegistry(SSOTCatalog()),
    )

    assert optimized.nodes["sink"].inputs["forcing"]["link"] == [
        "source",
        "source_flux",
    ]
    assert len(optimized.links) == 1
    assert optimized.links[0].source == LinkEndpoint(
        node="source",
        port="source_flux",
    )


def test_constant_folding_inlines_values() -> None:
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "const": NodeSpec(
                class_type="State:Constant",
                inputs={},
                outputs=["x"],
                config={"value": 42.0},
            ),
            "consumer": NodeSpec(
                class_type="Functor:adder",
                inputs={"x": {"link": ["const", "x"]}},
                outputs=["y"],
            ),
        },
        links=[
            LinkSpec(
                source=LinkEndpoint(node="const", port="x"),
                target=LinkEndpoint(node="consumer", port="x"),
            )
        ],
        metadata={},
    )

    opt_pass = ConstantFoldingPass()
    registry = NodeRegistry(SSOTCatalog())
    optimized = opt_pass.apply(graph, registry)

    assert "const" not in optimized.nodes
    assert optimized.nodes["consumer"].inputs["x"]["value"] == 42.0
    assert "constant_folding" in optimized.metadata.get("optimizations_applied", [])


def test_compile_graph_with_optimization() -> None:
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "source": NodeSpec(class_type="State:ExternalForcing", inputs={}, outputs=["x"]),
        },
        links=[],
        metadata={},
    )
    registry = NodeRegistry(SSOTCatalog())
    plan = compile_graph(graph, registry, enable_optimization=True)

    assert plan.steps == []
    assert plan.report.metadata["state_boundary_lowering"][
        "required_initial_state_keys"
    ] == ["x"]
    assert plan.report.metadata["state_boundary_lowering"][
        "executable_node_ids"
    ] == []


def test_state_identity_metadata_uses_transformed_graph() -> None:
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "source": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=["x"],
            ),
        },
    )

    plan = compile_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        enable_optimization=True,
        optimization_passes=[_RewriteExternalStatePass()],
    )

    assert plan.report.metadata["state_boundary_lowering"][
        "required_initial_state_keys"
    ] == ["optimized_x"]


def test_compile_graph_rechecks_functor_resolution_after_optimization() -> None:
    class_type = "Functor:not_a_real_module.InjectedFunctor"
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "source": NodeSpec(
                class_type="State:ExternalForcing",
                inputs={},
                outputs=["x"],
            ),
        },
    )
    registry = NodeRegistry(SSOTCatalog())
    registry.register_definition(
        class_type,
        registry.build_from_node_spec(
            class_type,
            input_names=[],
            output_names=[],
        ),
    )

    with pytest.raises(ValueError, match="could not be resolved"):
        compile_graph(
            graph,
            registry,
            enable_optimization=True,
            optimization_passes=[_InjectUnknownFunctorPass(class_type)],
        )


def test_compiler_seal_binds_optimization_pass_type_state_and_order() -> None:
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "source": NodeSpec(
                class_type="State:ExternalForcing",
                inputs={},
                outputs=["x"],
            ),
        },
    )
    registry = NodeRegistry(SSOTCatalog())

    def _seal(passes):
        plan = compile_graph(
            graph,
            registry,
            strict=True,
            enable_optimization=True,
            optimization_passes=passes,
        )
        return plan_to_dict(plan)["report"]["compiler_seal"]

    baseline = _seal([_StatefulNoOpPass("v1")])
    repeated = _seal([_StatefulNoOpPass("v1")])
    changed_state = _seal([_StatefulNoOpPass("v2")])
    changed_type = _seal([_AlternateNoOpPass("v1")])
    first_order = _seal(
        [_StatefulNoOpPass("v1"), _AlternateNoOpPass("v1")]
    )
    reversed_order = _seal(
        [_AlternateNoOpPass("v1"), _StatefulNoOpPass("v1")]
    )

    assert baseline == repeated
    for changed in (changed_state, changed_type, first_order, reversed_order):
        assert changed["source_graph_sha256"] == baseline["source_graph_sha256"]
        assert changed["transformed_graph_sha256"] == baseline[
            "transformed_graph_sha256"
        ]
        assert changed["executable_graph_sha256"] == baseline[
            "executable_graph_sha256"
        ]
        assert changed["plan_sha256"] == baseline["plan_sha256"]
        assert changed["compiler_options_sha256"] != baseline[
            "compiler_options_sha256"
        ]
        assert changed["overall_sha256"] != baseline["overall_sha256"]
    assert first_order["compiler_options_sha256"] != reversed_order[
        "compiler_options_sha256"
    ]
