"""Tests for Phase 8 gate enforcement assertions in GraphIR."""

from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.orchestration.validator import validate_graph


def _graph(op, context, audit_key):
    return GraphIR(
        schema_version="1.0.0",
        nodes={"n0": NodeSpec(class_type="State:ExternalForcing", outputs=["forcing"])},
        links=[],
        metadata={},
        assertion={
            "phi": op,
            "op": op,
            "scope": "phase8_gate",
            "context": context,
            "audit_key": audit_key,
        },
    )


def test_graph_assertion_accepts_port_matching_check():
    graph = _graph(
        "port_matching_check",
        {
            "ports": [
                {
                    "name": "source",
                    "connection_id": "carbon",
                    "positive_power_convention": "out_of_system",
                    "port_kind": "matter",
                    "power_units": "J s-1",
                },
                {
                    "name": "sink",
                    "connection_id": "carbon",
                    "positive_power_convention": "into_system",
                    "port_kind": "matter",
                    "power_units": "J s-1",
                },
            ]
        },
        "port_matching_audit",
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert report.errors == []
    assert report.assertion_audit["structural_result"]["passed"] is True


def test_graph_assertion_accepts_overflow_routing_check():
    graph = _graph(
        "overflow_routing_check",
        {
            "routes": [
                    {
                        "name": "runoff",
                        "amount": 1.0,
                        "units": "kg s-1",
                        "route_type": "boundary_outflow",
                        "target": "outlet",
                    }
            ]
        },
        "overflow_routing_audit",
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert report.errors == []
    assert report.assertion_audit["structural_result"]["passed"] is True


def test_graph_assertion_accepts_equivalence_certificate_check():
    graph = _graph(
        "equivalence_certificate_check",
        {
            "certificate": {
                "process_id": "relic.scientific.transport.transport",
                "module_name": "transport",
                "canonical_role": "transport",
                "iphs_level": 3,
                "iphs_level_evidence": {
                    "classification": "IPHS-structured transport certificate",
                    "compatible_basis": "DEC/IPHS transport structure",
                    "runtime_replacement": "forward uses canonical IPHS operator",
                    "canonical_operator": "x_dot=J_h*e_E+M_h*e_S+G_h*u_h",
                    "structural_contribution": "operator_map declares D_h and M_h",
                    "assembly_participation": "GraphIR assertion consumes certificate",
                },
                "theory_document": "docs\\source\\theory\\transport.md",
                "theory_document_status": "already_current",
                "legacy_formula": "q = -K grad(h)",
                "literature_formula": "Darcy or Fick flux driven by a potential gradient",
                "iphs_formula": "x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h",
                "legacy_expression": "q = -K grad(h)",
                "iphs_expression": "x_dot = J_h e_E + M_h e_S + G_h u_h",
                "input_variables": ["h"],
                "output_variables": ["q"],
                "parameter_keys": ["k"],
                "composition_ports": ["water"],
                "references": ["van der Schaft and Maschke (1995)"],
                "validation_tests": [
                    "test\\unit\\orchestration\\graph_phase8_enforcement_assertions.test.py"
                ],
                    "runtime_replacement_map": {
                        "runtime_uses_iphs": False,
                        "legacy_removed": False,
                    },
                "state_map": {"h": "x_h.potential"},
                "parameter_map": {"k": "k"},
                "parameter_management_map": {"k": {"source": "ParameterManager"}},
                "operator_map": {"gradient": "D_h", "mobility": "M_h"},
                "state_advancement_map": {
                    "potential_basis": "G(x)=0.5*x.T*H*x",
                    "effort_variable": "mu=dG/dx",
                    "state_advancer": "x_dot=J_h*e_E+M_h*e_S+G_h*u_h",
                    "runtime_usage": "forward uses canonical IPHS operator",
                },
                "unit_map": {"h": "m", "q": "m s-1"},
                "input_output_map": {"h": "x_h.potential", "q": "port.flow"},
                "interface_map": {
                    "forward_signature": "unchanged",
                    "bridge_class": "unchanged",
                    "dependency_semantics": "unchanged",
                    "state_write_mode": "existing ContractRegistry and StateManager semantics",
                },
                "composition_management_map": {
                    "functor_metadata": "existing FunctorMetadata fields",
                    "port_spec": "existing PortSpec or PortSpecIPHS assembly view",
                    "state_contract": "existing ContractRegistry and StateManager",
                    "validation_rule": "existing ValidationRegistry rule",
                },
                "dependency_interconnection_map": {
                    "dependency_type": "causal_state",
                    "declared_dependency": "not_applicable",
                },
                "overflow_routing_map": {
                    "residual_name": "flux_residual",
                    "destination": "diagnostic_residual",
                },
                "conservation_map": {"mass": "storage plus boundary flux"},
                "entropy_audit": {"sigma_expression": "X_h.T L_h X_h >= 0"},
                "equivalence_type": "port_equivalent",
                "baseline_tests": [
                    "test\\unit\\orchestration\\graph_phase8_enforcement_assertions.test.py"
                ],
            }
        },
        "equivalence_certificate_audit",
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert report.errors == []
    assert report.assertion_audit["structural_result"]["passed"] is True


def test_graph_assertion_accepts_reaction_network_check():
    graph = _graph(
        "reaction_network_check",
        {
            "name": "closed_reaction",
            "stoichiometric_N": [[-1.0], [1.0]],
            "conservation_matrix": [[1.0, 1.0]],
            "rates": [0.1],
            "affinities": [1.0],
        },
        "reaction_network_audit",
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert report.errors == []
    assert report.assertion_audit["structural_result"]["passed"] is True


def test_graph_assertion_accepts_stochastic_sdae_check():
    graph = _graph(
        "stochastic_sdae_check",
        {
            "runtime_owner": "relic.stochastic",
            "stochastic_terms": [
                {
                    "name": "flux_noise",
                    "classification": "fdt_bound",
                    "runtime_owner": "relic.stochastic.sde_solver",
                    "noise_type": "white",
                    "solver_name": "euler_maruyama",
                }
            ],
            "noise_covariances": [
                {
                    "name": "fdt_flux",
                    "relation": "fdt",
                    "sigma": [[1.4142135623730951]],
                    "dissipative_operator": [[1.0]],
                    "covariance_scale": 2.0,
                    "scale_parameter_key": "transport.noise_scale",
                    "physical_constant_keys": ["boltzmann_constant"],
                    "tolerance": 1.0e-9,
                }
            ],
            "stochastic_reproducibility": {
                "random_seed": 7,
                "noise_type": "white",
                "solver_method": "euler_maruyama",
            },
        },
        "stochastic_sdae_audit",
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert report.errors == []
    assert report.assertion_audit["structural_result"]["passed"] is True
