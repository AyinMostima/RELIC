from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from relic.orchestration.adapters import AdapterRegistry
from relic.orchestration.compiler import (
    compile_graph,
    synthesize_bounded_graphs,
)
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog
from test.fixtures.dummy_functor import (
    DummyConfiguredOutputFunctor,
    DummyIncrementFunctor,
    DummySinkFunctor,
)

SOURCE_TYPE = "Functor:test.fixtures.dummy_functor.DummyIncrementFunctor.ncs"
SINK_TYPE = "Functor:test.fixtures.dummy_functor.DummySinkFunctor.ncs"
DEPENDENCY_TYPE = (
    "Functor:test.fixtures.dummy_functor."
    "DummyConfiguredOutputFunctor.ncs_dependency"
)


class _FailingOptimizationPass:
    name = "test_failure_after_lowering"

    def apply(self, graph: GraphIR, registry: NodeRegistry) -> GraphIR:
        _ = graph, registry
        raise ValueError("intentional post-lowering failure")


def _port(
    *,
    name: str,
    units: str,
    direction: str,
    standard_name: str = "air_temperature",
    semantics: str = "state",
) -> PortSpec:
    return PortSpec(
        name=name,
        standard_name=standard_name,
        units=units,
        semantics=semantics,
        direction=direction,
        spatial_kind="grid_cell",
        area_basis="grid_area",
        temporal_basis="instantaneous",
        vertical_basis="none",
        shape={"dimensions": []},
    )


def _unit_cast_case() -> tuple[GraphIR, NodeRegistry, AdapterRegistry]:
    registry = NodeRegistry(SSOTCatalog())
    source_definition = registry.register_functor_instance(
        DummyIncrementFunctor(),
        node_type_id=SOURCE_TYPE,
    )
    source_definition.outputs = {
        "x": _port(name="x", units="K", direction="out")
    }
    sink_definition = registry.register_functor_instance(
        DummySinkFunctor(config={"input_name": "x_celsius"}),
        node_type_id=SINK_TYPE,
    )
    sink_definition.inputs = {
        "x_celsius": _port(
            name="x_celsius",
            units="degC",
            direction="in",
        )
    }
    sink_definition.outputs = {}
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": SOURCE_TYPE,
                    "inputs": {},
                    "outputs": ["x"],
                    "config": {"execution_group": "fast"},
                },
                "sink": {
                    "class_type": SINK_TYPE,
                    "inputs": {
                        "x_celsius": {
                            "link": ["source", "x"],
                        }
                    },
                    "outputs": [],
                    "config": {
                        "input_name": "x_celsius",
                        "execution_group": "fast",
                    },
                },
            },
            "metadata": {
                "profile_id": "PROFILE-005",
                "adapters": "auto",
                "execution_policy": "all_fast",
                "adaptive_dt_enabled": False,
            },
        }
    )
    return graph, registry, AdapterRegistry()


def _bounded_synthesis_case() -> tuple[
    GraphIR,
    dict[str, object],
    NodeRegistry,
    AdapterRegistry,
]:
    graph, registry, adapter_registry = _unit_cast_case()
    source_kelvin_type = f"{SOURCE_TYPE}.kelvin"
    source_celsius_type = f"{SOURCE_TYPE}.celsius"
    sink_grid_type = f"{SINK_TYPE}.grid"
    sink_footprint_type = f"{SINK_TYPE}.footprint"

    for class_type, units in (
        (source_kelvin_type, "K"),
        (source_celsius_type, "degC"),
    ):
        definition = registry.register_functor_instance(
            DummyIncrementFunctor(),
            node_type_id=class_type,
        )
        definition.outputs = {
            "x": _port(name="x", units=units, direction="out")
        }
    for class_type, spatial_kind in (
        (sink_grid_type, "grid_cell"),
        (sink_footprint_type, "footprint"),
    ):
        definition = registry.register_functor_instance(
            DummySinkFunctor(config={"input_name": "x_celsius"}),
            node_type_id=class_type,
        )
        definition.inputs = {
            "x_celsius": PortSpec(
                name="x_celsius",
                standard_name="air_temperature",
                units="K",
                semantics="state",
                direction="in",
                spatial_kind=spatial_kind,
                area_basis="grid_area",
                temporal_basis="instantaneous",
                vertical_basis="none",
                shape={"dimensions": []},
            )
        }
        definition.outputs = {}

    base_payload = graph.to_dict()
    base_payload["nodes"]["source"]["class_type"] = source_kelvin_type
    base_payload["nodes"]["sink"]["class_type"] = sink_grid_type
    base_graph = GraphIR.from_dict(base_payload)
    request: dict[str, object] = {
        "schema_version": "1.0.0",
        "request_id": "NCS-ORACLE-UNIT-CAST-001",
        "max_combinations": 4,
        "auto_adapt": True,
        "required_outputs": ["x"],
        "choice_groups": [
            {
                "group_id": "source_choice",
                "node_id": "source",
                "candidates": [
                    {
                        "candidate_id": "source_kelvin",
                        "node": {
                            "class_type": source_kelvin_type,
                            "inputs": {},
                            "outputs": ["x"],
                            "config": {"execution_group": "fast"},
                        },
                    },
                    {
                        "candidate_id": "source_celsius",
                        "node": {
                            "class_type": source_celsius_type,
                            "inputs": {},
                            "outputs": ["x"],
                            "config": {"execution_group": "fast"},
                        },
                    },
                ],
            },
            {
                "group_id": "sink_choice",
                "node_id": "sink",
                "candidates": [
                    {
                        "candidate_id": "sink_grid",
                        "node": {
                            "class_type": sink_grid_type,
                            "inputs": {
                                "x_celsius": {
                                    "link": ["source", "x"],
                                }
                            },
                            "outputs": [],
                            "config": {
                                "input_name": "x_celsius",
                                "execution_group": "fast",
                            },
                        },
                    },
                    {
                        "candidate_id": "sink_footprint",
                        "node": {
                            "class_type": sink_footprint_type,
                            "inputs": {
                                "x_celsius": {
                                    "link": ["source", "x"],
                                }
                            },
                            "outputs": [],
                            "config": {
                                "input_name": "x_celsius",
                                "execution_group": "fast",
                            },
                        },
                    },
                ],
            },
        ],
        "constraints": {
            "strict_unique_writer": True,
        },
    }
    return base_graph, request, registry, adapter_registry


def test_auto_adapt_lowers_unit_cast_to_executable_registered_node() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    registry.ssot_catalog = SSOTCatalog.from_parameter_manager()
    original_payload = deepcopy(graph.to_dict())

    plan = compile_graph(
        graph,
        registry,
        adapter_registry=adapter_registry,
        auto_adapt=True,
        strict=True,
    )

    assert graph.to_dict() == original_payload
    assert [step.node_id for step in plan.steps] == [
        "source",
        plan.report.adapter_insertions[0].details["adapter_node_id"],
        "sink",
    ]
    assert len(plan.report.adapter_insertions) == 1
    insertion = plan.report.adapter_insertions[0]
    assert insertion.adapter_type == "unit_cast"
    assert insertion.details["lowered"] is True
    assert insertion.details["source_units"] == "K"
    assert insertion.details["target_units"] == "degC"
    assert insertion.details["adapter_node_type"] in registry.definitions
    assert plan.report.metadata["lowered_graph"]["metadata"]["adapters"] == (
        "explicit"
    )

    lowered_output = insertion.details["lowered_output_port"]
    final_state = GraphExecutor(plan, registry=registry).execute(
        {
            "x": torch.tensor(0.0),
            lowered_output: torch.tensor(0.0),
        },
        n_steps=1,
        dt=1.0,
    )
    assert torch.allclose(
        final_state[lowered_output],
        torch.tensor(-272.15),
        rtol=0.0,
        atol=1.0e-5,
    )
    assert plan.report.validation.errors == []
    assert plan.report.validation.warnings == []


def test_auto_adapt_does_not_hide_unbound_unlinked_ssot_port() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    registry.ssot_catalog = SSOTCatalog.from_parameter_manager()
    unbound_type = f"{SOURCE_TYPE}.unbound"
    definition = registry.register_functor_instance(
        DummyIncrementFunctor(),
        node_type_id=unbound_type,
    )
    definition.outputs = {
        "unbound_temperature": _port(
            name="unbound_temperature",
            units="degC",
            direction="out",
        )
    }
    payload = graph.to_dict()
    payload["nodes"]["unbound"] = {
        "class_type": unbound_type,
        "inputs": {},
        "outputs": ["unbound_temperature"],
        "config": {"execution_group": "fast"},
    }

    with pytest.raises(ValueError, match="exact SSOT unit"):
        compile_graph(
            GraphIR.from_dict(payload),
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )


def test_auto_adapt_is_deterministic_and_registry_idempotent() -> None:
    graph, registry, adapter_registry = _unit_cast_case()

    first = compile_graph(
        graph,
        registry,
        adapter_registry=adapter_registry,
        auto_adapt=True,
        strict=True,
    )
    definition_count = len(registry.definitions)
    second = compile_graph(
        graph,
        registry,
        adapter_registry=adapter_registry,
        auto_adapt=True,
        strict=True,
    )

    assert len(registry.definitions) == definition_count
    assert first.report.metadata["lowered_graph"] == second.report.metadata[
        "lowered_graph"
    ]
    assert first.report.adapter_insertions == second.report.adapter_insertions


def test_strict_compile_rejects_unlowered_adapter_requirement() -> None:
    graph, registry, adapter_registry = _unit_cast_case()

    with pytest.raises(ValueError, match="executable adapter"):
        compile_graph(
            graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=False,
            strict=True,
        )


def test_auto_adapt_rejects_context_dependent_unit_conversion() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    source_definition = registry.get_definition(SOURCE_TYPE)
    sink_definition = registry.get_definition(SINK_TYPE)
    assert source_definition is not None
    assert sink_definition is not None
    source_definition.outputs["x"] = _port(
        name="x",
        units="mol m-2 s-1",
        direction="out",
        standard_name="stomatal_conductance",
    )
    sink_definition.inputs["x_celsius"] = _port(
        name="x_celsius",
        units="m s-1",
        direction="in",
        standard_name="stomatal_conductance",
    )

    with pytest.raises(ValueError, match="Conversion requires context"):
        compile_graph(
            graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )


def test_auto_adapt_requires_explicit_endpoint_execution_groups() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    payload = graph.to_dict()
    payload["nodes"]["source"]["config"] = {}
    missing_group_graph = GraphIR.from_dict(payload)

    with pytest.raises(ValueError, match="explicit execution_group"):
        compile_graph(
            missing_group_graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )


def test_auto_adapt_rejects_cross_group_lowering() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    payload = graph.to_dict()
    payload["nodes"]["sink"]["config"]["execution_group"] = "slow"
    cross_group_graph = GraphIR.from_dict(payload)

    with pytest.raises(ValueError, match="execution groups"):
        compile_graph(
            cross_group_graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )


def test_auto_adapt_rejects_same_named_state_writer() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    sink_definition = registry.get_definition(SINK_TYPE)
    assert sink_definition is not None
    sink_definition.inputs = {
        "x": _port(name="x", units="degC", direction="in")
    }
    payload = graph.to_dict()
    payload["nodes"]["sink"]["inputs"] = {
        "x": {"link": ["source", "x"]}
    }
    payload["nodes"]["sink"]["config"] = {
        "input_name": "x",
        "execution_group": "fast",
    }
    payload["links"] = [
        {
            "from": {"node": "source", "port": "x"},
            "to": {"node": "sink", "port": "x"},
        }
    ]
    same_name_graph = GraphIR.from_dict(payload)
    original_definition_count = len(registry.definitions)

    with pytest.raises(ValueError, match="distinct source and target"):
        compile_graph(
            same_name_graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )

    assert len(registry.definitions) == original_definition_count


def test_auto_adapt_rejects_external_state_boundary_source() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    external_type = "State:ExternalForcing"
    external_definition = registry.build_from_node_spec(
        external_type,
        input_names=[],
        output_names=["x"],
    )
    external_definition.outputs = {
        "x": _port(name="x", units="K", direction="out")
    }
    registry.register_definition(external_type, external_definition)
    payload = graph.to_dict()
    payload["nodes"]["source"] = {
        "class_type": external_type,
        "inputs": {},
        "outputs": ["x"],
        "config": {},
    }
    external_graph = GraphIR.from_dict(payload)
    original_definition_count = len(registry.definitions)

    with pytest.raises(ValueError, match="declarative state boundary"):
        compile_graph(
            external_graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )

    assert len(registry.definitions) == original_definition_count


def test_auto_adapt_rejects_non_unit_semantic_transformation() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    sink_definition = registry.get_definition(SINK_TYPE)
    assert sink_definition is not None
    sink_definition.inputs["x_celsius"].semantics = "diagnostic"
    original_definition_count = len(registry.definitions)

    with pytest.raises(ValueError, match="non-unit port contract"):
        compile_graph(
            graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )

    assert len(registry.definitions) == original_definition_count


def test_failed_multi_link_lowering_rolls_back_registered_aliases() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    second_sink_type = f"{SINK_TYPE}.second"
    second_sink_definition = registry.register_functor_instance(
        DummySinkFunctor(config={"input_name": "forcing_celsius"}),
        node_type_id=second_sink_type,
    )
    second_sink_definition.inputs = {
        "forcing_celsius": _port(
            name="forcing_celsius",
            units="degC",
            direction="in",
        )
    }
    second_sink_definition.outputs = {}
    external_type = "State:ExternalForcing"
    external_definition = registry.build_from_node_spec(
        external_type,
        input_names=[],
        output_names=["forcing_kelvin"],
    )
    external_definition.outputs = {
        "forcing_kelvin": _port(
            name="forcing_kelvin",
            units="K",
            direction="out",
        )
    }
    registry.register_definition(external_type, external_definition)
    payload = graph.to_dict()
    payload["nodes"]["external"] = {
        "class_type": external_type,
        "inputs": {},
        "outputs": ["forcing_kelvin"],
        "config": {},
    }
    payload["nodes"]["second_sink"] = {
        "class_type": second_sink_type,
        "inputs": {
            "forcing_celsius": {
                "link": ["external", "forcing_kelvin"],
            }
        },
        "outputs": [],
        "config": {"input_name": "forcing_celsius"},
    }
    payload["links"] = [
        {
            "from": {"node": "source", "port": "x"},
            "to": {"node": "sink", "port": "x_celsius"},
        },
        {
            "from": {
                "node": "external",
                "port": "forcing_kelvin",
            },
            "to": {
                "node": "second_sink",
                "port": "forcing_celsius",
            },
        },
    ]
    multi_link_graph = GraphIR.from_dict(payload)
    original_definition_count = len(registry.definitions)

    with pytest.raises(ValueError, match="declarative state boundary"):
        compile_graph(
            multi_link_graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )

    assert len(registry.definitions) == original_definition_count


def test_post_lowering_failure_rolls_back_registered_alias() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    original_definition_count = len(registry.definitions)

    with pytest.raises(ValueError, match="post-lowering failure"):
        compile_graph(
            graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
            enable_optimization=True,
            optimization_passes=[_FailingOptimizationPass()],
        )

    assert len(registry.definitions) == original_definition_count


def test_existing_auto_adapter_alias_rejects_full_port_contract_drift() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    first = compile_graph(
        graph,
        registry,
        adapter_registry=adapter_registry,
        auto_adapt=True,
        strict=True,
    )
    insertion = first.report.adapter_insertions[0]
    definition = registry.get_definition(
        insertion.details["adapter_node_type"]
    )
    assert definition is not None
    output_port = definition.outputs[
        insertion.details["lowered_output_port"]
    ]
    output_port.quantity = "intentional_contract_drift"
    definition_count = len(registry.definitions)

    with pytest.raises(ValueError, match="target contract drift"):
        compile_graph(
            graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )

    assert len(registry.definitions) == definition_count


def test_strict_compile_records_non_executing_runtime_preflight() -> None:
    graph, registry, adapter_registry = _unit_cast_case()

    plan = compile_graph(
        graph,
        registry,
        adapter_registry=adapter_registry,
        auto_adapt=True,
        strict=True,
    )

    witness = plan.report.metadata["compile_run_witness"]
    assert witness["schema_version"] == "1.0.0"
    assert witness["passed"] is True
    assert witness["execution_performed"] is False
    assert witness["node_count"] == len(plan.steps)
    assert witness["checks"] == {
        "class_resolution": True,
        "constructor_validation": True,
        "functor_base": True,
        "port_validation": True,
        "writer_validation": True,
    }
    assert [node["node_id"] for node in witness["nodes"]] == [
        step.node_id for step in plan.steps
    ]


def test_strict_compile_preflight_rejects_protected_constructor_config() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    payload = graph.to_dict()
    payload["nodes"]["source"]["config"]["device"] = "cpu"
    protected_config_graph = GraphIR.from_dict(payload)

    with pytest.raises(ValueError, match="protected constructor field 'device'"):
        compile_graph(
            protected_config_graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )


def test_strict_compile_preflight_rejects_runtime_input_drift() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    payload = graph.to_dict()
    payload["nodes"]["sink"]["config"]["input_name"] = "runtime_only_input"
    runtime_drift_graph = GraphIR.from_dict(payload)

    with pytest.raises(ValueError, match="runtime input contract"):
        compile_graph(
            runtime_drift_graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )


def test_strict_compile_preflight_rejects_runtime_output_drift() -> None:
    graph, registry, adapter_registry = _unit_cast_case()
    source_definition = registry.get_definition(SOURCE_TYPE)
    assert source_definition is not None
    source_definition.outputs = {
        "registry_only_output": _port(
            name="registry_only_output",
            units="K",
            direction="out",
        )
    }
    payload = graph.to_dict()
    payload["nodes"]["source"]["outputs"] = ["registry_only_output"]
    payload["nodes"]["sink"]["inputs"] = {
        "x_celsius": {
            "link": ["source", "registry_only_output"],
        }
    }
    payload["links"] = [
        {
            "from": {
                "node": "source",
                "port": "registry_only_output",
            },
            "to": {"node": "sink", "port": "x_celsius"},
        }
    ]
    runtime_drift_graph = GraphIR.from_dict(payload)

    with pytest.raises(ValueError, match="runtime output contract"):
        compile_graph(
            runtime_drift_graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )


def test_bounded_synthesis_matches_frozen_finite_oracle() -> None:
    base_graph, request, registry, adapter_registry = (
        _bounded_synthesis_case()
    )

    result = synthesize_bounded_graphs(
        base_graph,
        request,
        registry=registry,
        adapter_registry=adapter_registry,
        strict=True,
    )

    expected = {
        "sink_choice=sink_grid|source_choice=source_celsius",
        "sink_choice=sink_grid|source_choice=source_kelvin",
    }
    assert result.enumerated_count == 4
    assert result.valid_count == 2
    assert result.invalid_count == 2
    assert set(result.valid_graphs) == expected
    assert all(
        "sink_choice=sink_footprint" in selection_id
        for selection_id in result.invalid_reasons
    )
    assert result.to_dict()["complete_enumeration"] is True
    assert "pareto_selection_ids" not in result.to_dict()
    assert "candidate_metrics" not in result.to_dict()
    invalid_issues = result.to_dict()["invalid_issues"]
    assert set(invalid_issues) == set(result.invalid_reasons)
    assert all(
        issues
        and all(
            isinstance(issue["code"], str) and issue["code"]
            for issue in issues
        )
        for issues in invalid_issues.values()
    )
    for graph in result.valid_graphs.values():
        plan = compile_graph(
            graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )
        assert len(plan.steps) >= 2


def test_bounded_synthesis_is_independent_of_request_order() -> None:
    base_graph, request, registry, adapter_registry = (
        _bounded_synthesis_case()
    )
    reversed_request = deepcopy(request)
    groups = reversed_request["choice_groups"]
    assert isinstance(groups, list)
    groups.reverse()
    for group in groups:
        assert isinstance(group, dict)
        candidates = group["candidates"]
        assert isinstance(candidates, list)
        candidates.reverse()

    first = synthesize_bounded_graphs(
        base_graph,
        request,
        registry=registry,
        adapter_registry=adapter_registry,
        strict=True,
    )
    second = synthesize_bounded_graphs(
        base_graph,
        reversed_request,
        registry=registry,
        adapter_registry=adapter_registry,
        strict=True,
    )

    assert first.to_dict() == second.to_dict()
    assert {
        key: graph.to_dict() for key, graph in first.valid_graphs.items()
    } == {
        key: graph.to_dict() for key, graph in second.valid_graphs.items()
    }


def test_bounded_synthesis_adds_candidate_dependencies_conditionally() -> None:
    base_graph, request, registry, adapter_registry = (
        _bounded_synthesis_case()
    )
    registry.ssot_catalog = SSOTCatalog.from_parameter_manager()
    base_payload = base_graph.to_dict()
    base_payload["nodes"]["forcing"] = {
        "class_type": "State:ExternalForcing",
        "inputs": {},
        "outputs": ["air_temperature"],
        "config": {},
    }
    base_graph = GraphIR.from_dict(base_payload)
    request["external_state_node_id"] = "forcing"
    dependency_definition = registry.register_functor_instance(
        DummyConfiguredOutputFunctor(
            config={"output_name": "dependency_signal"}
        ),
        node_type_id=DEPENDENCY_TYPE,
    )
    dependency_definition.outputs = {
        "dependency_signal": _port(
            name="dependency_signal",
            units="K",
            direction="out",
        )
    }
    groups = request["choice_groups"]
    assert isinstance(groups, list)
    source_group = groups[0]
    assert isinstance(source_group, dict)
    candidates = source_group["candidates"]
    assert isinstance(candidates, list)
    celsius_candidate = candidates[1]
    assert isinstance(celsius_candidate, dict)
    celsius_candidate["required_external_inputs"] = [
        "vapor_pressure_deficit"
    ]
    celsius_candidate["dependency_nodes"] = {
        "candidate_dependency": {
            "class_type": DEPENDENCY_TYPE,
            "inputs": {},
            "outputs": ["dependency_signal"],
            "config": {
                "output_name": "dependency_signal",
                "execution_group": "fast",
            },
        }
    }

    result = synthesize_bounded_graphs(
        base_graph,
        request,
        registry=registry,
        adapter_registry=adapter_registry,
        strict=True,
    )

    assert result.valid_count == 2
    assert any(
        "source_choice=source_celsius" in selection_id
        for selection_id in result.valid_graphs
    )
    for selection_id, graph in result.valid_graphs.items():
        if "source_choice=source_celsius" in selection_id:
            assert "candidate_dependency" in graph.nodes
            assert "vapor_pressure_deficit" in graph.nodes[
                "forcing"
            ].outputs
        else:
            assert "candidate_dependency" not in graph.nodes
            assert graph.nodes["forcing"].outputs == ["air_temperature"]


def test_bounded_synthesis_rejects_dependency_node_collision() -> None:
    base_graph, request, registry, adapter_registry = (
        _bounded_synthesis_case()
    )
    groups = request["choice_groups"]
    assert isinstance(groups, list)
    source_group = groups[0]
    assert isinstance(source_group, dict)
    candidates = source_group["candidates"]
    assert isinstance(candidates, list)
    first_candidate = candidates[0]
    assert isinstance(first_candidate, dict)
    first_candidate["dependency_nodes"] = {
        "sink": {
            "class_type": first_candidate["node"]["class_type"],
            "inputs": {},
            "outputs": ["x"],
            "config": {"execution_group": "fast"},
        }
    }

    with pytest.raises(ValueError, match="collides with base graph node"):
        synthesize_bounded_graphs(
            base_graph,
            request,
            registry=registry,
            adapter_registry=adapter_registry,
            strict=True,
        )


def test_bounded_synthesis_rejects_deferred_pareto_configuration() -> None:
    base_graph, request, registry, adapter_registry = (
        _bounded_synthesis_case()
    )
    request["preferences"] = {
        "minimize": ["validation_loss", "compute_cost"]
    }

    with pytest.raises(ValueError, match="Pareto search is deferred"):
        synthesize_bounded_graphs(
            base_graph,
            request,
            registry=registry,
            adapter_registry=adapter_registry,
            strict=True,
        )


def test_bounded_synthesis_rejects_unbounded_or_unregistered_request() -> None:
    base_graph, request, registry, adapter_registry = (
        _bounded_synthesis_case()
    )
    over_limit = deepcopy(request)
    over_limit["max_combinations"] = 3
    with pytest.raises(ValueError, match="exceeds max_combinations"):
        synthesize_bounded_graphs(
            base_graph,
            over_limit,
            registry=registry,
            adapter_registry=adapter_registry,
            strict=True,
        )

    unregistered = deepcopy(request)
    groups = unregistered["choice_groups"]
    assert isinstance(groups, list)
    first_group = groups[0]
    assert isinstance(first_group, dict)
    candidates = first_group["candidates"]
    assert isinstance(candidates, list)
    first_candidate = candidates[0]
    assert isinstance(first_candidate, dict)
    node = first_candidate["node"]
    assert isinstance(node, dict)
    node["class_type"] = "Functor:unregistered.Candidate"
    with pytest.raises(ValueError, match="not registered"):
        synthesize_bounded_graphs(
            base_graph,
            unregistered,
            registry=registry,
            adapter_registry=adapter_registry,
            strict=True,
        )


def test_bounded_synthesis_propagates_compiler_defects(monkeypatch) -> None:
    """Only a compile validation rejection (ValueError) marks a candidate invalid."""
    import relic.orchestration.compiler as compiler_module

    base_graph, request, registry, adapter_registry = (
        _bounded_synthesis_case()
    )

    def _compiler_defect(*_args, **_kwargs):
        raise KeyError("compiler defect: missing plan slot")

    monkeypatch.setattr(compiler_module, "compile_graph", _compiler_defect)

    with pytest.raises(KeyError, match="compiler defect"):
        synthesize_bounded_graphs(
            base_graph,
            request,
            registry=registry,
            adapter_registry=adapter_registry,
            strict=True,
        )


def test_functor_class_resolution_issue_keeps_the_resolution_error() -> None:
    """An unimportable Functor class is reported with its import error."""
    import relic.orchestration.compiler as compiler_module

    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "n1": {
                    "class_type": "Functor:relic_test_missing_package.MissingFunctor",
                    "outputs": [],
                }
            },
            "links": [],
        }
    )

    issues = compiler_module._functor_class_resolution_issues(
        graph, NodeRegistry(SSOTCatalog())
    )

    assert [issue.code for issue in issues] == ["unresolved_functor_class"]
    assert "relic_test_missing_package" in issues[0].message


def test_functor_class_resolution_propagates_resolver_defects(monkeypatch) -> None:
    """A resolver error outside its resolution contract is not an issue record."""
    import relic.orchestration.compiler as compiler_module

    registry = NodeRegistry(SSOTCatalog())
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "n1": {
                    "class_type": "Functor:relic_test_package.SomeFunctor",
                    "outputs": [],
                }
            },
            "links": [],
        }
    )

    def _resolver_defect(class_type):
        raise AttributeError(f"resolver defect for {class_type}")

    monkeypatch.setattr(registry, "resolve_class_type", _resolver_defect)

    with pytest.raises(AttributeError, match="resolver defect"):
        compiler_module._functor_class_resolution_issues(graph, registry)


def test_auto_adapter_lowering_propagates_converter_infrastructure_failure(monkeypatch) -> None:
    """Only a context rejection (ValueError) is reported as missing runtime context."""
    import relic.orchestration.compiler as compiler_module

    graph, registry, adapter_registry = _unit_cast_case()
    lowering = {"active": False}
    original_contract_check = compiler_module._require_pure_unit_contract
    original_convert = adapter_registry.unit_converter.convert

    def _enter_lowering(*args, **kwargs):
        lowering["active"] = True
        return original_contract_check(*args, **kwargs)

    def _convert(*args, **kwargs):
        if lowering["active"]:
            raise RuntimeError("unit rule table could not be loaded")
        return original_convert(*args, **kwargs)

    monkeypatch.setattr(compiler_module, "_require_pure_unit_contract", _enter_lowering)
    monkeypatch.setattr(adapter_registry.unit_converter, "convert", _convert)

    with pytest.raises(RuntimeError, match="unit rule table could not be loaded"):
        compile_graph(
            graph,
            registry,
            adapter_registry=adapter_registry,
            auto_adapt=True,
            strict=True,
        )
