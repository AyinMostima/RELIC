"""GraphIR and executor integration of the surface ponding topology transition.

References:
    - Horton (1933); Beven (2012) ch. 6; Dooge (1973).
    - Goebel, Sanfelice and Teel (2012), Hybrid Dynamical Systems.
"""

from __future__ import annotations

import pytest
import torch

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR, LinkEndpoint, LinkSpec, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.scientific.atmosphere.stochastic_rainfall import MarkedPoissonRainfallFunctor
from relic.scientific.hydrosphere.runoff.surface_ponding import (
    ACTIVE_VARIABLE,
    EXCESS_VARIABLE,
    PONDED_DEPTH_VARIABLE,
    RAINFALL_DEPTH_VARIABLE,
    TRANSITION_CESSATION,
    TRANSITION_ONSET,
    HortonianPondingEventFunctor,
    SurfacePondingFunctor,
    ponding_cessation_transition,
    ponding_onset_transition,
)

COUNT = "stochastic_rainfall_event_count"
RAIN_CLASS = "relic.scientific.atmosphere.stochastic_rainfall.MarkedPoissonRainfallFunctor"
EVENT_CLASS = "relic.scientific.hydrosphere.runoff.surface_ponding.HortonianPondingEventFunctor"
RAIN_TYPE = f"Functor:{RAIN_CLASS}"
EVENT_TYPE = f"Functor:{EVENT_CLASS}"
SEED = 11
DT = 3600.0
N_CELLS = 2

pytestmark = pytest.mark.filterwarnings("ignore:Conservation law 'none' violated")


def _topology_payload():
    return {
        "required_semantics": ["topology_change"],
        "runtime_owner": "relic.core.simulator.registration",
        "hybrid_topology_transitions": [
            {
                "name": TRANSITION_ONSET,
                "event_name": TRANSITION_ONSET,
                "source_complex": {"complex_id": "K_disconnected", "semantic_name": "rainfall_and_detector"},
                "target_complex": {"complex_id": "K_connected", "semantic_name": "rainfall_detector_and_store"},
                "state_remap": [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "ledger_map": [[1.0, 1.0, 0.0]],
                "ledger_change": [[0.0]],
                "dimension_change": True,
                "reset_route": "relic.core.simulator.registration.apply_topology_transition",
                "parameter_keys": [
                    "hydrosphere.runoff.ponding_infiltration_capacity",
                    "hydrosphere.runoff.ponding_onset_depth",
                ],
                "audit_route": "diagnostics.topology_transition.ledger_residual",
                "state_vars": [PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE, ACTIVE_VARIABLE],
            },
            {
                "name": TRANSITION_CESSATION,
                "event_name": TRANSITION_CESSATION,
                "source_complex": {"complex_id": "K_connected", "semantic_name": "rainfall_detector_and_store"},
                "target_complex": {"complex_id": "K_disconnected", "semantic_name": "rainfall_and_detector"},
                "state_remap": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                "ledger_map": [[1.0, 1.0, 0.0]],
                "ledger_change": [[-1.0, 0.0, 0.0]],
                "dimension_change": True,
                "reset_route": "relic.core.simulator.registration.apply_topology_transition",
                "parameter_keys": ["hydrosphere.runoff.ponding_cessation_depth"],
                "audit_route": "diagnostics.topology_transition.ledger_residual",
                "state_vars": [PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE, ACTIVE_VARIABLE],
            },
        ],
        "topology_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "graph_epoch_owner": "relic.core.simulator.registration",
            "event_queue_state_key": "relic.core.simulator.execution_loop.retry_external_deliveries",
        },
        "state_units": {PONDED_DEPTH_VARIABLE: "mm", EXCESS_VARIABLE: "mm", ACTIVE_VARIABLE: "1"},
        "execution_bindings": {
            "topology_change": {
                "functor_class": EVENT_CLASS,
                "forward_owner": EVENT_CLASS + ".forward",
                "state_vars": [PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE, ACTIVE_VARIABLE],
                "topology_state_scope": "state_manager",
                "validation_test": "test/core/simulator_topology_transition.test.py",
            }
        },
    }


def _graph():
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    rain = MarkedPoissonRainfallFunctor(name="stochastic_rainfall", config={"seed": SEED})
    event = HortonianPondingEventFunctor()
    rain_definition = registry.register_functor_instance(rain, node_type_id=RAIN_TYPE)
    rain_definition.metadata["config_contract"] = {"seed": SEED}
    event_definition = registry.register_functor_instance(event, node_type_id=EVENT_TYPE)
    event_definition.metadata["config_contract"] = {}
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "stochastic_rainfall": NodeSpec(
                class_type=RAIN_TYPE,
                inputs={name: {"state": name} for name in rain.STATE_VARIABLES},
                outputs=list(rain.metadata.output_variables),
                config={"seed": SEED},
            ),
            "ponding_event": NodeSpec(
                class_type=EVENT_TYPE,
                inputs={
                    RAINFALL_DEPTH_VARIABLE: {"link": ["stochastic_rainfall", RAINFALL_DEPTH_VARIABLE]},
                    ACTIVE_VARIABLE: {"state": ACTIVE_VARIABLE},
                    PONDED_DEPTH_VARIABLE: {"state": PONDED_DEPTH_VARIABLE},
                },
                outputs=list(event.metadata.output_variables),
                config={},
            ),
        },
        links=[
            LinkSpec(
                LinkEndpoint("stochastic_rainfall", RAINFALL_DEPTH_VARIABLE),
                LinkEndpoint("ponding_event", RAINFALL_DEPTH_VARIABLE),
            )
        ],
        metadata={"topology_change_extension": True},
        assertion={
            "phi": "the ponding store is connected and retired by the committed topology transition",
            "op": "stochastic_sdae_check",
            "scope": "graph",
            "context": _topology_payload(),
            "audit_key": "stochastic_sdae_audit",
        },
    )
    return registry, graph


def _initial_state():
    z = torch.zeros(N_CELLS, dtype=torch.float64)
    return {
        COUNT: z.clone(), RAINFALL_DEPTH_VARIABLE: z.clone(), PONDED_DEPTH_VARIABLE: z.clone(),
        EXCESS_VARIABLE: z.clone(), ACTIVE_VARIABLE: z.clone(),
    }


def _prepared_executor(plan, registry):
    executor = GraphExecutor(plan, registry=registry)
    executor.execute(_initial_state(), n_steps=0, dt=DT)
    simulator = executor._simulator
    simulator.declare_topology_transition(ponding_onset_transition(SurfacePondingFunctor(), execution_group="fast"))
    simulator.declare_topology_transition(ponding_cessation_transition())
    return executor, simulator


def _run_until(simulator, predicate, max_steps=400):
    for _ in range(max_steps):
        simulator.step(dt=DT)
        if predicate(simulator):
            return simulator.step_count
    raise AssertionError("condition not reached")


class TestTopologyGraph:
    def test_strict_graph_compiles_with_the_executable_topology_binding(self):
        registry, graph = _graph()
        plan = compile_graph(graph, registry, strict=True)
        assert plan.report.validation.errors == []
        audit = plan.report.validation.assertion_audit
        assert audit["present"] is True and audit["valid"] is True and audit["op_resolved"] is True
        assert audit["structural_result"]["passed"] is True

    def test_missing_binding_fails_closed_under_the_topology_extension(self):
        from relic.validation.framework.rules.stochastic_sdae import validate_stochastic_sdae_payload

        registry, graph = _graph()
        graph.assertion["context"].pop("execution_bindings")
        with pytest.raises(ValueError, match="Stochastic IPHS/SDAE structure is incomplete"):
            compile_graph(graph, registry, strict=True)
        result = validate_stochastic_sdae_payload(graph.assertion["context"], required_semantics=("topology_change",))
        assert result.passed is False
        assert any("topology-transition binding" in detail.expected for detail in result.details)

    def test_executor_commits_the_transition_and_records_the_delivery(self, tmp_path):
        registry, graph = _graph()
        plan = compile_graph(graph, registry, strict=True)
        executor, simulator = _prepared_executor(plan, registry)
        epoch = simulator.provider_epoch
        onset_step = _run_until(simulator, lambda s: "surface_ponding" in s.functors)
        assert simulator.provider_epoch == epoch + 1
        assert executor.execution_summary["execution_order"] == ["stochastic_rainfall", "ponding_event", "surface_ponding"]
        delivery = simulator._last_external_delivery_receipts[-1]
        assert delivery["operation"] == "topology" and delivery["accepted_step"] == onset_step
        assert delivery["receipt"]["transition"] == TRANSITION_ONSET
        state = simulator.get_state()
        assert state[ACTIVE_VARIABLE].tolist() == [1.0, 1.0]
        assert float(state[PONDED_DEPTH_VARIABLE].max()) > 0.0
        history = simulator.get_topology_history()
        assert history[-1]["ledger_residual_max"] == 0.0

        # The plan-only executor refuses the post-onset checkpoint fail-closed:
        # execution plans are static, the restart applies the same transition.
        checkpoint = tmp_path / "post_onset.pt"
        receipt = executor.create_checkpoint(checkpoint)
        assert receipt["payload_sha256"] and "provider_epoch" in receipt
        with pytest.raises(ValueError, match="does not match the active simulator"):
            GraphExecutor(plan, registry=registry).execute_from_checkpoint(_initial_state(), checkpoint, n_steps=1, dt=DT)

        for _ in range(3):
            simulator.step(dt=DT)
        restored_executor, restored = _prepared_executor(plan, registry)
        restored.apply_topology_transition(TRANSITION_ONSET)
        restored.restore_checkpoint(checkpoint)
        assert restored.step_count == onset_step
        for _ in range(3):
            restored.step(dt=DT)
        for name in (RAINFALL_DEPTH_VARIABLE, PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE, ACTIVE_VARIABLE):
            assert torch.equal(restored.get_state()[name], simulator.get_state()[name]), name
        assert restored_executor.execution_summary["execution_order"] == executor.execution_summary["execution_order"]

    def test_cessation_retires_the_store_under_the_executor(self):
        registry, graph = _graph()
        plan = compile_graph(graph, registry, strict=True)
        _executor, simulator = _prepared_executor(plan, registry)
        _run_until(simulator, lambda s: "surface_ponding" in s.functors)
        _run_until(simulator, lambda s: "surface_ponding" not in s.functors)
        assert [item["transition"] for item in simulator.get_topology_history()] == [TRANSITION_ONSET, TRANSITION_CESSATION]
        assert simulator.get_state()[ACTIVE_VARIABLE].tolist() == [0.0, 0.0]
        assert simulator.get_state()[PONDED_DEPTH_VARIABLE].tolist() == [0.0, 0.0]
        assert simulator.state_manager.get_diagnostics()["current"]["topology_transition.ledger_change"] <= 0.0
