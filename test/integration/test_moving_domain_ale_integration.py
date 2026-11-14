"""GraphIR and StateManager integration of the moving-domain (ALE) diffusion.

References:
    - Thomas and Lombard (1979), AIAA J. 17, 1030-1037.
    - Donea et al. (2004), ALE methods, Encyclopedia of Computational Mechanics.
"""

import pytest
import torch

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.scientific.pedosphere.transport.moving_domain_diffusion import (
    MovingDomainDiffusionFunctor,
)

CONC = "moving_domain_solute_concentration"
LENGTH = "moving_domain_length"
VELOCITY = "moving_domain_boundary_velocity"
BOUNDARY_CONC = "moving_boundary_solute_concentration"
PROCESS = "moving_domain_diffusion"
FUNCTOR_CLASS = (
    "relic.scientific.pedosphere.transport.moving_domain_diffusion.MovingDomainDiffusionFunctor"
)
NODE_TYPE = f"Functor:{FUNCTOR_CLASS}"
N_CELLS = 4


def _ale_payload(n_cells):
    return {
        "required_semantics": ["ale"],
        "runtime_owner": "relic.scientific.pedosphere.transport.moving_domain_diffusion",
        "ale_geometry_corrections": [
            {
                "name": "uniform_stretching_metric",
                # dot(star) = diag(w_b / N) per unit boundary velocity.
                "hodge_time_derivative": (torch.eye(n_cells, dtype=torch.float64) / n_cells).tolist(),
                "state_space": {"complex_id": "K", "degree": 0, "semantic_name": CONC},
                "energy_effort": CONC,
                "correction_route": "diagnostics.moving_domain.geometry_power",
                "port_accounting": "diagnostics.moving_domain.boundary_mass_flux",
                "parameter_keys": [
                    "pedosphere.transport.moving_domain_diffusivity",
                    "pedosphere.transport.moving_domain_mesh_courant_max",
                ],
                "state_vars": [CONC, LENGTH],
            }
        ],
        "ale_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "geometry_state_keys": [LENGTH],
            "geometry_cursor": "state_manager_current_step",
        },
        "state_units": {CONC: "kg m-3", LENGTH: "m"},
        "execution_bindings": {
            "ale": {
                "functor_class": FUNCTOR_CLASS,
                "forward_owner": FUNCTOR_CLASS + ".forward",
                "state_vars": [CONC, LENGTH],
                "geometry_state_scope": "state_manager",
                "validation_test": "test/unit/pedosphere/test_moving_domain_diffusion.py",
            }
        },
    }


def _graph():
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    template = MovingDomainDiffusionFunctor(name=PROCESS)
    definition = registry.register_functor_instance(template, node_type_id=NODE_TYPE)
    definition.metadata["config_contract"] = {}
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            PROCESS: NodeSpec(
                class_type=NODE_TYPE,
                inputs={name: {"state": name} for name in (CONC, LENGTH, VELOCITY, BOUNDARY_CONC)},
                outputs=list(template.metadata.output_variables),
                config={},
            )
        },
        links=[],
        metadata={"stochastic_extension": True},
        assertion={
            "phi": "the column concentration and length are advanced by the bound ALE Functor with relative fluxes and the discrete GCL",
            "op": "stochastic_sdae_check",
            "scope": "graph",
            "context": _ale_payload(N_CELLS),
            "audit_key": "stochastic_sdae_audit",
        },
    )
    return registry, graph


class TestMovingDomainGraph:
    def test_strict_graph_compiles_with_the_executable_ale_binding(self):
        registry, graph = _graph()
        plan = compile_graph(graph, registry, strict=True)
        assert plan.report.validation.errors == []
        audit = plan.report.validation.assertion_audit
        assert audit["present"] is True and audit["valid"] is True and audit["op_resolved"] is True
        assert audit["structural_result"]["passed"] is True

    def test_missing_execution_binding_fails_closed_under_stochastic_extension(self):
        from relic.validation.framework.rules.stochastic_sdae import validate_stochastic_sdae_payload

        registry, graph = _graph()
        graph.assertion["context"].pop("execution_bindings")
        with pytest.raises(ValueError, match="Stochastic IPHS/SDAE structure is incomplete"):
            compile_graph(graph, registry, strict=True)
        result = validate_stochastic_sdae_payload(graph.assertion["context"], required_semantics=("ale",))
        assert result.passed is False
        assert any("ALE geometry-power binding [N]" in detail.expected for detail in result.details)

    def test_executor_preserves_a_constant_field_and_the_column_ledger(self):
        registry, graph = _graph()
        plan = compile_graph(graph, registry, strict=True)
        executor = GraphExecutor(plan, registry=registry)
        final = executor.execute(
            {
                CONC: torch.full((N_CELLS,), 3.0, dtype=torch.float64),
                LENGTH: torch.tensor(2.0, dtype=torch.float64),
                VELOCITY: torch.tensor(5.0e-3, dtype=torch.float64),
                BOUNDARY_CONC: torch.tensor(3.0, dtype=torch.float64),
            },
            n_steps=5,
            dt=60.0,
        )
        torch.testing.assert_close(
            final[CONC].detach().cpu(), torch.full((N_CELLS,), 3.0, dtype=torch.float64), rtol=0.0, atol=1e-13
        )
        assert float(final[LENGTH].detach().cpu()) == pytest.approx(2.0 + 5 * 60.0 * 5.0e-3)
        runtime = executor._simulator
        provenance = runtime.get_variable_provenance(LENGTH)
        assert provenance[-1]["functor"] == PROCESS
        assert provenance[-1]["role"] == "final_writer"
        assert executor.execution_summary["step_count"] == 5
