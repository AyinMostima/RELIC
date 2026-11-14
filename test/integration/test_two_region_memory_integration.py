"""GraphIR and StateManager integration of the two-region memory exchange.

References:
    - van Genuchten and Wierenga (1976), Soil Sci. Soc. Am. J. 40, 473-480.
    - Revised RELIC foundations, memory/Volterra semantics.
"""

import math

import pytest
import torch

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.scientific.pedosphere.transport.two_region_mass_transfer import (
    TwoRegionMassTransferFunctor,
)

MOBILE = "mobile_solute_concentration"
IMMOBILE = "immobile_solute_concentration"
PROCESS = "two_region_mass_transfer"
FUNCTOR_CLASS = (
    "relic.scientific.pedosphere.transport.two_region_mass_transfer.TwoRegionMassTransferFunctor"
)
NODE_TYPE = f"Functor:{FUNCTOR_CLASS}"


def _memory_payload(template):
    alpha = float(template.get_parameter_value("two_region_mass_transfer_coefficient", required=True).detach())
    theta_im = float(template.get_parameter_value("two_region_immobile_water_content", required=True).detach())
    omega = alpha / theta_im
    return {
        "required_semantics": ["memory"],
        "runtime_owner": "relic.scientific.pedosphere.transport.two_region_mass_transfer",
        "memory_kernels": [
            {
                "name": "two_region_exponential_memory",
                # One-term exponential realization K(t) = a exp(-b t), a = alpha omega, b = omega.
                "kernel_operator": [[alpha * omega]],
                "hidden_storage_space": {"complex_id": "K", "degree": 0, "semantic_name": IMMOBILE},
                "effort_variable": MOBILE,
                "passivity_evidence": "storage_function_certificate",
                "parameter_keys": [
                    "pedosphere.transport.two_region_mass_transfer_coefficient",
                    "pedosphere.transport.two_region_immobile_water_content",
                ],
                "residual_route": "diagnostics.two_region_exchange.memory_residual",
                "state_vars": [MOBILE, IMMOBILE],
            }
        ],
        "memory_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "history_state_keys": [IMMOBILE],
            "history_cursor": "state_manager_current_step",
        },
        "state_units": {MOBILE: "kg m-3", IMMOBILE: "kg m-3"},
        "execution_bindings": {
            "memory": {
                "functor_class": FUNCTOR_CLASS,
                "forward_owner": FUNCTOR_CLASS + ".forward",
                "state_vars": [MOBILE, IMMOBILE],
                "history_state_scope": "state_manager",
                "validation_test": "test/unit/pedosphere/test_two_region_mass_transfer.py",
            }
        },
    }


def _graph():
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    template = TwoRegionMassTransferFunctor(name=PROCESS)
    definition = registry.register_functor_instance(template, node_type_id=NODE_TYPE)
    definition.metadata["config_contract"] = {}
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            PROCESS: NodeSpec(
                class_type=NODE_TYPE,
                inputs={state: {"state": state} for state in (MOBILE, IMMOBILE)},
                outputs=list(template.metadata.output_variables),
                config={},
            )
        },
        links=[],
        metadata={"stochastic_extension": True},
        assertion={
            "phi": "the immobile concentration is the hidden memory realization advanced by the bound Functor",
            "op": "stochastic_sdae_check",
            "scope": "graph",
            "context": _memory_payload(template),
            "audit_key": "stochastic_sdae_audit",
        },
    )
    return registry, graph, template


class TestTwoRegionMemoryGraph:
    def test_strict_graph_compiles_with_the_executable_memory_binding(self):
        registry, graph, _ = _graph()
        plan = compile_graph(graph, registry, strict=True)
        assert plan.report.validation.errors == []
        audit = plan.report.validation.assertion_audit
        assert audit["present"] is True and audit["valid"] is True and audit["op_resolved"] is True
        assert audit["structural_result"]["passed"] is True

    def test_missing_execution_binding_fails_closed_under_stochastic_extension(self):
        from relic.validation.framework.rules.stochastic_sdae import validate_stochastic_sdae_payload

        registry, graph, _ = _graph()
        graph.assertion["context"].pop("execution_bindings")
        with pytest.raises(ValueError, match="Stochastic IPHS/SDAE structure is incomplete"):
            compile_graph(graph, registry, strict=True)
        result = validate_stochastic_sdae_payload(
            graph.assertion["context"], required_semantics=("memory",)
        )
        assert result.passed is False
        assert any("passive-memory execution binding [N]" in detail.expected for detail in result.details)

    def test_executor_advances_the_governed_memory_exchange(self):
        registry, graph, template = _graph()
        alpha = float(template.get_parameter_value("two_region_mass_transfer_coefficient", required=True).detach())
        theta_m = float(template.get_parameter_value("two_region_mobile_water_content", required=True).detach())
        theta_im = float(template.get_parameter_value("two_region_immobile_water_content", required=True).detach())
        plan = compile_graph(graph, registry, strict=True)
        executor = GraphExecutor(plan, registry=registry)
        dt = 1800.0
        final = executor.execute(
            {
                MOBILE: torch.tensor([2.0, 1.0], dtype=torch.float64),
                IMMOBILE: torch.tensor([0.0, 0.5], dtype=torch.float64),
            },
            n_steps=4,
            dt=dt,
        )
        mobile = final[MOBILE].detach().cpu()
        immobile = final[IMMOBILE].detach().cpu()
        # theta-weighted total conserved; difference decays exactly.
        torch.testing.assert_close(
            theta_m * mobile + theta_im * immobile,
            torch.tensor([theta_m * 2.0, theta_m * 1.0 + theta_im * 0.5], dtype=torch.float64),
            rtol=1e-12,
            atol=0.0,
        )
        kappa = alpha * (theta_m + theta_im) / (theta_m * theta_im)
        torch.testing.assert_close(
            mobile - immobile,
            torch.tensor([2.0, 0.5], dtype=torch.float64) * math.exp(-kappa * 4 * dt),
            rtol=1e-10,
            atol=0.0,
        )
        runtime = executor._simulator
        provenance = runtime.get_variable_provenance(IMMOBILE)
        assert provenance[-1]["functor"] == PROCESS
        assert provenance[-1]["role"] == "final_writer"
        assert executor.execution_summary["step_count"] == 4
