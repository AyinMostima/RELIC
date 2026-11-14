"""GraphIR and StateManager integration of the equilibrium sorption DAE.

References:
    - Hairer and Wanner (1996), Solving ODEs II, ch. VI.
    - Jury and Horton (2004), Soil Physics, ch. 7.
"""

import pytest
import torch

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.scientific.pedosphere.transport.equilibrium_sorption import EquilibriumSorptionFunctor

TOTAL = "equilibrium_total_solute_mass"
DISSOLVED = "equilibrium_dissolved_concentration"
SORBED = "equilibrium_sorbed_concentration"
THETA = "soil_moisture"
PROCESS = "equilibrium_sorption"
FUNCTOR_CLASS = "relic.scientific.pedosphere.transport.equilibrium_sorption.EquilibriumSorptionFunctor"
NODE_TYPE = f"Functor:{FUNCTOR_CLASS}"


# Soil dry bulk density of the test column: the bulk_density forcing input [kg m-3].
BULK_DENSITY = 1300.0


def _dae_payload(template, theta):
    return {
        "required_semantics": ["descriptor_dae"],
        "runtime_owner": "relic.scientific.pedosphere.transport.equilibrium_sorption",
        "descriptor_constraints": [
            {
                "name": "mass_partition",
                # Linear partition constraint on (y, C, S): -y + theta C + rho_b S = 0.
                "A_h": [[-1.0, theta, BULK_DENSITY]],
                "constrained_space": {"complex_id": "K", "degree": 0, "semantic_name": TOTAL},
                "units": "kg m-3",
                "projection_method": "algebraic_elimination",
                "residual_route": "diagnostics.equilibrium_sorption.constraint_residual",
                "enforces_state_constraint": True,
                "enforces_tangent_constraint": True,
                "constraint_force_neutral": True,
                "state_vars": [TOTAL, DISSOLVED, SORBED],
            }
        ],
        "descriptor_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "algebraic_state_keys": [DISSOLVED, SORBED],
            "projection_on_restore": True,
        },
        "state_units": {TOTAL: "kg m-3", DISSOLVED: "kg m-3", SORBED: "kg kg-1"},
        "execution_bindings": {
            "descriptor_dae": {
                "functor_class": FUNCTOR_CLASS,
                "forward_owner": FUNCTOR_CLASS + ".forward",
                "state_vars": [TOTAL, DISSOLVED, SORBED],
                "algebraic_state_scope": "state_manager",
                "validation_test": "test/unit/pedosphere/test_equilibrium_sorption.py",
            }
        },
    }


def _graph(theta=0.3):
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    template = EquilibriumSorptionFunctor(name=PROCESS)
    definition = registry.register_functor_instance(template, node_type_id=NODE_TYPE)
    definition.metadata["config_contract"] = {}
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            PROCESS: NodeSpec(
                class_type=NODE_TYPE,
                inputs={name: {"state": name} for name in (TOTAL, DISSOLVED, SORBED, THETA, "bulk_density")},
                outputs=list(template.metadata.output_variables),
                config={},
            )
        },
        links=[],
        metadata={"stochastic_extension": True},
        assertion={
            "phi": "the algebraic sorption states are eliminated each step by the bound index-1 DAE Functor",
            "op": "stochastic_sdae_check",
            "scope": "graph",
            "context": _dae_payload(template, theta),
            "audit_key": "stochastic_sdae_audit",
        },
    )
    return registry, graph, template


def _consistent(template, dissolved, theta):
    kf = float(template.get_parameter_value("equilibrium_sorption_freundlich_coefficient", required=True).detach())
    n = float(template.get_parameter_value("equilibrium_sorption_freundlich_exponent", required=True).detach())
    c = torch.tensor(dissolved, dtype=torch.float64)
    s = kf * c**n
    return {
        TOTAL: theta * c + BULK_DENSITY * s,
        DISSOLVED: c,
        SORBED: s,
        THETA: torch.full_like(c, theta),
        "bulk_density": torch.full_like(c, BULK_DENSITY),
    }


class TestEquilibriumSorptionGraph:
    def test_strict_graph_compiles_with_the_executable_dae_binding(self):
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
        result = validate_stochastic_sdae_payload(graph.assertion["context"], required_semantics=("descriptor_dae",))
        assert result.passed is False
        assert any("DAE projection binding [N]" in detail.expected for detail in result.details)

    def test_executor_keeps_the_constraint_satisfied_over_governed_steps(self):
        registry, graph, template = _graph()
        plan = compile_graph(graph, registry, strict=True)
        executor = GraphExecutor(plan, registry=registry)
        final = executor.execute(_consistent(template, [2.0, 0.5], 0.3), n_steps=4, dt=3600.0)
        y = final[TOTAL].detach().cpu()
        c = final[DISSOLVED].detach().cpu()
        s = final[SORBED].detach().cpu()
        kf = float(template.get_parameter_value("equilibrium_sorption_freundlich_coefficient", required=True).detach())
        n = float(template.get_parameter_value("equilibrium_sorption_freundlich_exponent", required=True).detach())
        assert float((0.3 * c + BULK_DENSITY * s - y).abs().max()) < 1e-9 * float(y.max())
        torch.testing.assert_close(s, kf * c**n, rtol=1e-12, atol=0.0)
        assert bool((c < torch.tensor([2.0, 0.5], dtype=torch.float64)).all())
        runtime = executor._simulator
        provenance = runtime.get_variable_provenance(DISSOLVED)
        assert provenance[-1]["functor"] == PROCESS
        assert provenance[-1]["role"] == "final_writer"
        assert executor.execution_summary["step_count"] == 4
