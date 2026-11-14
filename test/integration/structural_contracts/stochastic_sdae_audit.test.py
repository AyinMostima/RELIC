"""Integration test for stochastic-SDAE compiler audit routing."""

from relic.orchestration.compiler import compile_graph
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog


def test_compile_graph_emits_stochastic_sdae_audit_from_assertion():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={"noise": NodeSpec(class_type="State:ExternalForcing", outputs=["forcing"])},
        links=[],
        metadata={},
        assertion={
            "phi": "stochastic_sdae_check",
            "op": "stochastic_sdae_check",
            "scope": "structural_contracts",
            "audit_key": "stochastic_sdae_audit",
            "context": {
                "runtime_owner": "relic.stochastic",
                "stochastic_terms": [
                    {
                        "name": "flux_noise",
                        "classification": "external_forcing",
                        "runtime_owner": "relic.stochastic.sde_solver",
                        "noise_type": "white",
                        "solver_name": "euler_maruyama",
                    }
                ],
                "stochastic_reproducibility": {
                    "random_seed": 42,
                    "noise_type": "white",
                    "solver_method": "euler_maruyama",
                },
            },
        },
    )

    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()))
    audit = plan.report.metadata["audit_summary"]["stochastic_sdae_audit"]

    assert audit["enabled"] is True
    assert audit["passed"] is True
    assert audit["source"] == "validation.assertion"
