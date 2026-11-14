"""Tests for GraphIR structural assertions using existing assertion field."""

from dataclasses import replace

import pytest
import torch

from relic.computation.operators.dec import CochainSpaceSpec, DECComplexBundle, HodgeStarSpec
from relic.computation.operators.iphs import IPHSStructureSpec
from relic.computation.operators.iphs_certificate import IPHSEquivalenceCertificate
from relic.orchestration.compiler import compile_graph, plan_to_dict
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import NodeDefinition
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.orchestration.validator import build_audit_summary, validate_graph


def test_graph_level_adapter_uses_certificate_owner(monkeypatch):
    import relic.orchestration.validator as validator_module
    from relic.computation.operators.iphs_certificate import resolve_iphs_level_requirement

    observed = []

    def resolve(declarations, **kwargs):
        observed.append((declarations, kwargs))
        return resolve_iphs_level_requirement(declarations, **kwargs)

    monkeypatch.setattr(validator_module, "resolve_iphs_level_requirement", resolve, raising=False)
    assembly = {"strict": True}
    metadata = {"iphs_assembly": assembly}
    result = validator_module._required_iphs_level(
        {"iphs_level": 2}, strict=False,
        canonical_role="observation", graph_metadata=metadata,
    )

    assert result == (4, [])
    assert observed == [(
        {"context": {"iphs_level": 2}, "graph.metadata": metadata,
         "graph.metadata.iphs_assembly": assembly},
        {"strict_iphs": True, "canonical_role": ""},
    )]


def _graph(assertion):
    return GraphIR(
        schema_version="1.0.0",
        nodes={"n0": NodeSpec(class_type="State:ExternalForcing", outputs=["soil_water_storage"])},
        links=[],
        metadata={},
        assertion=assertion,
    )


def _strict_graph(role, op, context, audit_key, **metadata):
    graph_metadata = {"phase8_strict": True, "canonical_role": role}
    graph_metadata.update(metadata)
    return GraphIR(
        schema_version="1.0.0",
        nodes={
            "n0": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=["soil_water_storage"],
            )
        },
        links=[],
        metadata=graph_metadata,
        assertion={
            "phi": op,
            "op": op,
            "scope": "phase8_strict",
            "context": context,
            "audit_key": audit_key,
        },
    )


def _typed_assertion_port(name, convention, connection_id="energy_exchange"):
    return {
        "name": name,
        "connection_id": connection_id,
        "effort_var": f"{name}_effort",
        "flow_var": "energy_flow",
        "positive_power_convention": convention,
        "port_kind": "energy",
        "power_units": "J s-1",
        "effort_space": {
            "complex_id": "test_complex",
            "degree": 0,
            "primality": "dual",
            "extensivity": "intensive",
            "physical_domain": "test_domain",
            "orientation_convention": "positive_into_system",
            "boundary_role": "port",
            "units": "J mol-1",
            "n_dofs": 1,
        },
        "flow_space": {
            "complex_id": "test_complex",
            "degree": 0,
            "primality": "primal",
            "extensivity": "integral",
            "physical_domain": "test_domain",
            "orientation_convention": "positive_into_system",
            "boundary_role": "port",
            "units": "mol s-1",
            "n_dofs": 1,
        },
    }


def _valid_complex():
    d0 = torch.tensor(
        [
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    d1 = torch.tensor([[1.0, 1.0, 1.0]])
    return DECComplexBundle(
        complex_id="soil_column",
        dimension=2,
        n_cells={0: 3, 1: 3, 2: 1},
        D={0: d0, 1: d1},
        boundary_masks={},
        orientation={},
        cell_measures={},
        dual_measures={},
        orientation_convention="outward_positive",
    )


def _bad_hodge():
    domain = CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="primal",
        n_dofs=2,
        units="m",
        extensivity="integral",
        physical_domain="hydrosphere",
        semantic_name="hydraulic_head_difference",
        orientation_convention="outward_positive",
    )
    codomain = CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="dual",
        n_dofs=2,
        units="kg s-1",
        extensivity="integral",
        physical_domain="hydrosphere",
        semantic_name="water_flux",
        orientation_convention="outward_positive",
    )
    return HodgeStarSpec(
        name="bad_hodge",
        domain=domain,
        codomain=codomain,
        matrix=torch.diag(torch.tensor([1.0, -0.1])),
        parameter_keys=("k",),
        metric_keys=("area",),
        material_law="darcy",
    )


def _descriptive_level_two_certificate():
    return IPHSEquivalenceCertificate(
        module_name="descriptive_transport",
        process_id="relic.scientific.descriptive_transport",
        canonical_role="transport",
        iphs_level=2,
        iphs_level_evidence={
            "classification": "IPHS-compatible transport",
            "compatible_basis": "Declared state and port basis",
            "runtime_replacement": "Descriptive compatibility only",
            "canonical_operator": "Declared canonical transport operator",
        },
        theory_document="docs/source/architecture_overview.md",
        theory_document_status="already_current",
        legacy_expression="x_dot = f(x)",
        literature_formula="Conservative transport balance",
        iphs_expression="x_dot = J_h e_H + G_h u_h",
        input_variables=("soil_water_storage",),
        output_variables=("delta_soil_water_storage",),
        parameter_keys=(),
        parameter_free=True,
        state_map={"soil_water_storage": "x_h.water"},
        operator_map={"transport": "canonical declared operator"},
        state_advancement_map={
            "potential_basis": "H_h(x_h)",
            "effort_variable": "e_H=dH_h/dx_h",
            "state_advancer": "x_dot=J_h*e_H+G_h*u_h",
            "runtime_usage": "descriptive compatibility declaration",
        },
        unit_map={"soil_water_storage": "kg m-2"},
        input_output_map={"soil_water_storage": "delta_soil_water_storage"},
        composition_ports=("water_boundary",),
        references=("van der Schaft and Maschke (1995)",),
        validation_tests=(
            "test/unit/orchestration/graph_structural_assertions.test.py",
        ),
        runtime_replacement_map={
            "runtime_uses_iphs": False,
            "legacy_removed": False,
        },
        dependency_interconnection_map={"dependency_type": "causal_state"},
        overflow_routing_map={"residual_route": "rejected_step"},
        conservation_map={"water": "state plus boundary port"},
        entropy_audit={"status": "not a strict thermodynamic claim"},
        equivalence_type="port_equivalent",
        interface_map={"forward_signature": "unchanged"},
    )


_REGISTERED_TRANSPORT_PROCESS = (
    "relic.scientific.pedosphere.transport.SoilWaterTransportFunctor."
    "SoilWaterTransportFunctor"
)


def _level_four_transport_certificate(process_id=_REGISTERED_TRANSPORT_PROCESS):
    return replace(
        _descriptive_level_two_certificate(),
        process_id=process_id,
        formulation="dissipative_ph",
        iphs_level=4,
        iphs_level_evidence={
            "dissipation_audit": {
                "tolerance_parameter_key": "test.iphs.tolerance",
            },
            "port_power_audit": {
                "tolerance_parameter_key": "test.iphs.tolerance",
            },
        },
    )


def _certificate_process_graph(
    certificate,
    *,
    scope="process_node",
    graph_level=4,
    graph_formulation="dissipative_ph",
    runtime_class=_REGISTERED_TRANSPORT_PROCESS,
    register_definition=True,
    class_type=None,
    strict=True,
    context=None,
):
    node_class_type = class_type or f"Functor:{runtime_class}"
    registry = NodeRegistry(SSOTCatalog())
    if register_definition:
        registry.register_definition(
            node_class_type,
            NodeDefinition(
                node_type_id=node_class_type,
                metadata={"runtime_class": runtime_class},
            ),
        )
    metadata = {
        "canonical_role": "transport",
        "iphs_level": graph_level,
        "formulation": graph_formulation,
    }
    if strict:
        metadata["phase8_strict"] = True
    assertion_context = {"certificate": certificate}
    if context:
        assertion_context.update(context)
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={"process_node": NodeSpec(class_type=node_class_type)},
        links=[],
        metadata=metadata,
        assertion={
            "phi": "certificate",
            "op": "equivalence_certificate_check",
            "scope": scope,
            "context": assertion_context,
            "audit_key": "equivalence_certificate_audit",
        },
    )
    return graph, registry


def _complete_runtime_tolerance_spec(key):
    return {
        "key": key,
        "value": 1.0e-8,
        "category": "model_parameter",
        "units": "1",
        "dtype": "float64",
        "bounds": (0.0, 1.0e-6),
        "references": ("unit-test",),
        "provenance": {"source": "unit_test"},
    }


def test_graph_structural_assertion_accepts_valid_dec_complex():
    graph = _graph(
        {
            "phi": "D_next_D_k_zero",
            "op": "cochain_complex_check",
            "scope": "soil_column",
            "context": {"dec_complex_bundle": _valid_complex()},
            "audit_key": "dec_topology_audit",
        }
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())
    audit = build_audit_summary(report)

    assert report.errors == []
    assert report.assertion_audit["op_resolved"] is True
    assert report.assertion_audit["structural_result"]["passed"] is True
    assert audit["dec_topology_audit"]["passed"] is True


def test_graph_structural_assertion_reports_hodge_failure():
    graph = _graph(
        {
            "phi": "hodge_passive",
            "op": "hodge_psd_check",
            "scope": "soil_column",
            "context": {"hodge_spec": _bad_hodge()},
            "audit_key": "hodge_audit",
        }
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert any(issue.code == "structural_assertion_failed" for issue in report.errors)
    assert report.assertion_audit["valid"] is False
    assert report.assertion_audit["structural_result"]["passed"] is False


def test_compile_graph_routes_structural_audit_through_report_audit():
    graph = _graph(
        {
            "phi": "D_next_D_k_zero",
            "op": "cochain_complex_check",
            "scope": "soil_column",
            "context": {"dec_complex_bundle": _valid_complex()},
            "audit_key": "dec_topology_audit",
        }
    )

    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()))
    report = plan_to_dict(plan)["report"]

    assert report["audit"]["dec_topology_audit"]["passed"] is True


def test_graph_structural_assertion_accepts_entropy_step_check():
    graph = _graph(
        {
            "phi": "entropy_nonnegative",
            "op": "entropy_step_check",
            "scope": "iphs_step",
            "context": {
                "name": "closed_step",
                "entropy_before": 1.0,
                "entropy_after": 1.01,
                "tolerance": 1.0e-12,
            },
            "audit_key": "entropy_audit",
        }
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert report.errors == []
    assert report.assertion_audit["op_resolved"] is True
    assert report.assertion_audit["structural_result"]["passed"] is True


def test_graph_metadata_structural_assertions_are_enforced():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={"n0": NodeSpec(class_type="State:ExternalForcing", outputs=["soil_water_storage"])},
        links=[],
        metadata={
            "structural_assertions": [
                {
                    "phi": "D_next_D_k_zero",
                    "op": "cochain_complex_check",
                    "scope": "soil_column",
                    "context": {"dec_complex_bundle": _valid_complex()},
                    "audit_key": "dec_topology_audit",
                },
                {
                    "phi": "entropy_nonnegative",
                    "op": "entropy_step_check",
                    "scope": "iphs_step",
                    "context": {
                        "name": "closed_step",
                        "entropy_before": 1.0,
                        "entropy_after": 1.01,
                    },
                    "audit_key": "entropy_audit",
                },
            ]
        },
        assertion={},
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert report.errors == []
    assert report.assertion_audit["valid"] is True
    assert len(report.assertion_audit["results"]) == 2


def test_phase8_strict_graph_requires_structural_assertions():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={"n0": NodeSpec(class_type="State:ExternalForcing", outputs=["soil_water_storage"])},
        links=[],
        metadata={"phase8_strict": True},
        assertion={},
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert any(issue.code == "structural_assertions_required" for issue in report.errors)
    assert report.assertion_audit["valid"] is False


def test_phase8_strict_graph_requires_explicit_canonical_role():
    graph = _strict_graph(
        "transport",
        "entropy_step_check",
        {"entropy_before": 1.0, "entropy_after": 1.0},
        "entropy_audit",
    )
    del graph.metadata["canonical_role"]

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert any(issue.code == "strict_canonical_role_required" for issue in report.errors)


def test_phase8_strict_graph_rejects_extension_semantic_as_canonical_role():
    graph = _strict_graph(
        "jump",
        "entropy_step_check",
        {"entropy_before": 1.0, "entropy_after": 1.0},
        "entropy_audit",
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert any(issue.code == "strict_canonical_role_invalid" for issue in report.errors)


def test_graph_structural_assertion_reports_port_power_unit_failure():
    graph = _graph(
        {
            "phi": "port_power_units",
            "op": "port_power_units_check",
            "scope": "boundary_port",
            "context": {
                "name": "bad_heat_port",
                "power_units": "kg s-1",
                "expected_power_units": "W m-2",
            },
            "audit_key": "port_power_audit",
        }
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert any(issue.code == "structural_assertion_failed" for issue in report.errors)
    assert report.assertion_audit["valid"] is False
    assert report.assertion_audit["structural_result"]["passed"] is False


def test_graph_port_power_assertion_serializes_contract_identities():
    ports = [
        _typed_assertion_port("source", "out_of_system"),
        _typed_assertion_port("sink", "into_system"),
    ]
    graph = _graph(
        {
            "phi": "port_power_identity",
            "op": "port_power_units_check",
            "scope": "energy_exchange",
            "context": {"ports": ports},
            "audit_key": "port_power_audit",
        }
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )
    structural_result = report.assertion_audit["structural_result"]

    assert structural_result["passed"] is True
    assert {item["name"] for item in structural_result["contract_identities"]} == {
        "source",
        "sink",
    }


@pytest.mark.parametrize(
    ("op", "audit_key"),
    [
        ("port_power_units_check", "port_power_audit"),
        ("port_matching_check", "port_matching_audit"),
    ],
)
def test_graph_port_assertion_rejects_decoy_iphs_assembly_contract(
    op,
    audit_key,
):
    assembly_ports = [
        _typed_assertion_port("assembly_source", "out_of_system"),
        _typed_assertion_port("assembly_sink", "into_system"),
    ]
    decoy_ports = [
        _typed_assertion_port("decoy_source", "out_of_system"),
        _typed_assertion_port("decoy_sink", "into_system"),
    ]
    graph = _graph(
        {
            "phi": "decoy_port_contract",
            "op": op,
            "scope": "energy_exchange",
            "context": {"ports": decoy_ports},
            "audit_key": audit_key,
        }
    )
    graph.metadata["iphs_assembly"] = {
        "state_vars": ("x",),
        "contributions": [{"ports": assembly_ports}],
    }

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        strict=False,
        ssot_catalog=SSOTCatalog(),
    )

    assert any(
        issue.code == "port_assertion_iphs_binding_mismatch"
        for issue in report.errors
    )
    assert report.assertion_audit["valid"] is False
    assert (
        report.assertion_audit["structural_result"][
            "iphs_assembly_contract_bound"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("op", "audit_key"),
    [
        ("port_power_units_check", "port_power_audit"),
        ("port_matching_check", "port_matching_audit"),
    ],
)
def test_graph_port_assertion_binds_same_iphs_assembly_contract(op, audit_key):
    ports = [
        _typed_assertion_port("source", "out_of_system"),
        _typed_assertion_port("sink", "into_system"),
    ]
    graph = _graph(
        {
            "phi": "bound_port_contract",
            "op": op,
            "scope": "energy_exchange",
            "context": {"ports": ports},
            "audit_key": audit_key,
        }
    )
    graph.metadata["iphs_assembly"] = {
        "state_vars": ("x",),
        "contributions": [{"ports": ports}],
    }

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        strict=False,
        ssot_catalog=SSOTCatalog(),
    )

    assert not any(
        issue.code == "port_assertion_iphs_binding_mismatch"
        for issue in report.errors
    )
    assert report.assertion_audit["valid"] is True
    assert (
        report.assertion_audit["structural_result"][
            "iphs_assembly_contract_bound"
        ]
        is True
    )


def test_phase8_strict_graph_propagates_strict_to_dec_complex_check():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={"n0": NodeSpec(class_type="State:ExternalForcing", outputs=["forcing"])},
        links=[],
        metadata={"phase8_strict": True},
        assertion={
            "phi": "D_next_D_k_zero",
            "op": "cochain_complex_check",
            "scope": "soil_column",
            "context": {"dec_complex_bundle": _valid_complex(), "strict": False},
            "audit_key": "dec_topology_audit",
        },
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert any(issue.code == "structural_assertion_failed" for issue in report.errors)
    assert "boundary_masks" in report.assertion_audit["structural_result"]["message"]


def test_phase8_strict_graph_rejects_only_one_unrelated_assertion():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={"n0": NodeSpec(class_type="State:ExternalForcing", outputs=["forcing"])},
        links=[],
        metadata={"phase8_strict": True},
        assertion={
            "phi": "entropy_nonnegative",
            "op": "entropy_step_check",
            "scope": "step",
            "context": {"entropy_before": 1.0, "entropy_after": 1.0},
            "audit_key": "entropy_audit",
        },
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert any(
        issue.code == "missing_required_structural_assertion" for issue in report.errors
    )


def test_phase8_strict_graph_routes_entropy_tolerance_through_parameter_manager():
    graph = _strict_graph(
        "transport",
        "entropy_step_check",
        {
            "entropy_production": 0.0,
            "tolerance_parameter_key": "test.iphs.tolerance",
            "parameter_resolver": lambda key: _complete_runtime_tolerance_spec(key),
            "strict": False,
        },
        "entropy_audit",
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is True


def test_phase8_strict_graph_binds_entropy_and_certificate_to_one_tolerance_key():
    certificate = replace(
        _descriptive_level_two_certificate(),
        formulation="dissipative_ph",
        iphs_level=4,
        iphs_level_evidence={
            "dissipation_audit": {
                "tolerance_parameter_key": "test.iphs.tolerance",
            },
            "port_power_audit": {
                "tolerance_parameter_key": "test.iphs.tolerance",
            },
        },
    )
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "n0": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=["soil_water_storage"],
            )
        },
        links=[],
        metadata={"phase8_strict": True, "canonical_role": "transport"},
        assertion={
            "assertions": [
                {
                    "phi": "certificate",
                    "op": "equivalence_certificate_check",
                    "scope": "phase8_strict",
                    "context": {"certificate": certificate},
                    "audit_key": "equivalence_certificate_audit",
                },
                {
                    "phi": "entropy_nonnegative",
                    "op": "entropy_step_check",
                    "scope": "phase8_strict",
                    "context": {
                        "entropy_production": 0.0,
                        "tolerance_parameter_key": "test.iphs.other_tolerance",
                        "parameter_resolver": (
                            lambda key: _complete_runtime_tolerance_spec(key)
                        ),
                    },
                    "audit_key": "entropy_audit",
                },
            ]
        },
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    mismatch = next(
        issue
        for issue in report.errors
        if issue.code == "strict_entropy_tolerance_binding_mismatch"
    )
    assert mismatch.expected == "test.iphs.tolerance"
    assert mismatch.got == "test.iphs.other_tolerance"


def test_phase8_strict_stochastic_assertion_rejects_empty_declaration():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={"n0": NodeSpec(class_type="State:ExternalForcing", outputs=["forcing"])},
        links=[],
        metadata={
            "phase8_strict": True,
            "canonical_role": "field",
            "stochastic_extension": True,
        },
        assertion={
            "phi": "stochastic_structure",
            "op": "stochastic_sdae_check",
            "scope": "stochastic",
            "context": {},
            "audit_key": "stochastic_sdae_audit",
        },
    )

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()), ssot_catalog=SSOTCatalog())

    assert any(issue.code == "structural_assertion_failed" for issue in report.errors)
    assert report.assertion_audit["structural_result"]["passed"] is False


def test_phase8_strict_iphs_level_cannot_be_downgraded_by_context():
    spec = IPHSStructureSpec(
        name="legacy_unspecified",
        state_vars=("x",),
        energy_functional="H",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        M=torch.ones((1, 1)),
        G=None,
        stoichiometric_N=None,
        ports=(), poisson_operator="matrix_snapshot", reaction_representation="independent",
    )
    graph = _strict_graph(
        "transport",
        "iphs_structure_check",
        {"iphs_spec": spec, "required_level": 1, "strict": False},
        "iphs_audit",
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is False
    assert "strict formulation" in report.assertion_audit["structural_result"]["message"]


def test_phase8_strict_iphs_level_explicit_none_is_invalid():
    spec = IPHSStructureSpec(
        name="legacy_unspecified",
        state_vars=("x",),
        energy_functional="H",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        M=torch.ones((1, 1)),
        G=None,
        stoichiometric_N=None,
        ports=(), poisson_operator="matrix_snapshot", reaction_representation="independent",
    )
    graph = _strict_graph(
        "transport",
        "iphs_structure_check",
        {"iphs_spec": spec, "required_level": None, "strict": False},
        "iphs_audit",
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert any(
        issue.code == "invalid_iphs_level_threshold" for issue in report.errors
    )


def test_phase8_strict_equivalence_certificate_enforces_level_four_minimum():
    graph = _strict_graph(
        "transport",
        "equivalence_certificate_check",
        {
            "certificate": _descriptive_level_two_certificate(),
            "required_level": 0,
            "strict": False,
        },
        "equivalence_certificate_audit",
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is False
    assert (
        report.assertion_audit["structural_result"]["metrics"][
            "required_runtime_level"
        ]
        == 4
    )
    assert not any(
        issue.code == "equivalence_certificate_role_mismatch"
        for issue in report.errors
    )


def test_phase8_graph_level_five_cannot_be_downgraded_by_assertion_context():
    graph = _strict_graph(
        "transport",
        "equivalence_certificate_check",
        {
            "certificate": _descriptive_level_two_certificate(),
            "required_level": 4,
        },
        "equivalence_certificate_audit",
        iphs_level=5,
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert (
        report.assertion_audit["structural_result"]["metrics"][
            "required_runtime_level"
        ]
        == 5
    )


@pytest.mark.parametrize(
    ("source", "value"),
    (
        ("context.required_level", True),
        ("context.required_level", "5"),
        ("context.required_level", 5.0),
        ("context.iphs_level", None),
        ("graph.metadata.iphs_level", False),
        ("graph.metadata.iphs_level", -1),
        ("graph.metadata.iphs_level", 6),
    ),
)
def test_graph_iphs_level_thresholds_reject_invalid_explicit_values(source, value):
    context = {"certificate": _descriptive_level_two_certificate()}
    metadata = {"iphs_level": 4}
    if source == "context.required_level":
        context["required_level"] = value
    elif source == "context.iphs_level":
        context["iphs_level"] = value
    else:
        metadata["iphs_level"] = value
    graph = _strict_graph(
        "transport",
        "equivalence_certificate_check",
        context,
        "equivalence_certificate_audit",
        **metadata,
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    issue = next(
        issue
        for issue in report.errors
        if issue.code == "invalid_iphs_level_threshold"
    )
    assert issue.got == f"{source}={value!r}"


def test_observation_graph_explicit_level_five_is_not_role_capped():
    certificate = replace(
        _descriptive_level_two_certificate(),
        canonical_role="observation",
        iphs_level=1,
        equivalence_type="diagnostic",
    )
    graph = _strict_graph(
        "observation",
        "equivalence_certificate_check",
        {"certificate": certificate},
        "equivalence_certificate_audit",
        iphs_level=5,
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert (
        report.assertion_audit["structural_result"]["metrics"][
            "required_runtime_level"
        ]
        == 5
    )


@pytest.mark.parametrize(
    "scope",
    (
        "phase8_strict",
        {"spec_nodes": {"certificate": "process_node"}},
        {"node_id": "process_node", "role": "transport"},
        {"node_id": ""},
    ),
)
def test_level_four_certificate_scope_must_identify_one_graph_node(scope):
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate(),
        scope=scope,
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    assert any(
        issue.code == "equivalence_certificate_scope_unbound"
        for issue in report.errors
    )


def test_nonstrict_explicit_level_four_still_requires_graph_node_scope():
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate(),
        scope="descriptive_label",
        strict=False,
        context={"required_level": 4},
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    assert any(
        issue.code == "equivalence_certificate_scope_unbound"
        for issue in report.errors
    )


def test_level_four_certificate_rejects_unknown_graph_node_scope():
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate(),
        scope={"node_id": "missing_node"},
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    assert any(
        issue.code == "equivalence_certificate_node_unresolved"
        for issue in report.errors
    )


def test_level_four_certificate_rejects_state_boundary_scope():
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate(),
        class_type="State:ExternalForcing",
        runtime_class=_REGISTERED_TRANSPORT_PROCESS,
        register_definition=False,
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    assert any(
        issue.code == "equivalence_certificate_node_not_process"
        for issue in report.errors
    )


def test_level_four_certificate_rejects_missing_node_definition():
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate(),
        register_definition=False,
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    assert any(
        issue.code == "equivalence_certificate_node_unresolved"
        for issue in report.errors
    )


def test_level_four_certificate_rejects_invalid_registered_runtime_class():
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate(),
        runtime_class="invalid_runtime_class",
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    assert any(
        issue.code == "equivalence_certificate_node_unresolved"
        for issue in report.errors
    )


def test_same_role_certificate_cannot_substitute_another_process():
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate("relic.scientific.unrelated.Process"),
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    mismatch = next(
        issue
        for issue in report.errors
        if issue.code == "equivalence_certificate_process_identity_mismatch"
    )
    assert mismatch.expected == _REGISTERED_TRANSPORT_PROCESS
    assert mismatch.got == "relic.scientific.unrelated.Process"


def test_exact_certificate_graph_identity_remains_compile_time_only():
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate(),
        scope={"node_id": "process_node"},
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    structural_result = report.assertion_audit["structural_result"]
    assert structural_result["certificate_graph_identity_bound"] is True
    assert structural_result["resolved_node_id"] == "process_node"
    assert (
        structural_result["resolved_runtime_class"]
        == _REGISTERED_TRANSPORT_PROCESS
    )
    assert structural_result["identity_evidence_scope"] == (
        "compile_time_identity_only"
    )
    assert structural_result["passed"] is False
    assert structural_result["metrics"]["runtime_verified_level"] == 0


def test_certificate_graph_formulation_must_match_explicit_graph_declaration():
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate(),
        graph_formulation="generic",
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    assert any(
        issue.code == "equivalence_certificate_formulation_mismatch"
        for issue in report.errors
    )


def test_certificate_declared_level_must_match_explicit_graph_level():
    graph, registry = _certificate_process_graph(
        _level_four_transport_certificate(),
        graph_level=5,
    )

    report = validate_graph(graph, registry, ssot_catalog=SSOTCatalog())

    assert any(
        issue.code == "equivalence_certificate_declared_level_mismatch"
        for issue in report.errors
    )


def test_phase8_strict_observation_certificate_does_not_make_level_four_claim():
    observation_certificate = replace(
        _descriptive_level_two_certificate(),
        canonical_role="observation",
        iphs_level=1,
        equivalence_type="diagnostic",
    )
    graph = _strict_graph(
        "observation",
        "equivalence_certificate_check",
        {
            "certificate": observation_certificate,
            "strict": False,
        },
        "equivalence_certificate_audit",
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    structural_result = report.assertion_audit["structural_result"]
    assert structural_result["passed"] is True
    assert structural_result["metrics"]["required_runtime_level"] == -1
    assert not any(
        issue.code == "equivalence_certificate_role_mismatch"
        for issue in report.errors
    )


def test_phase8_strict_graph_role_must_match_equivalence_certificate_role():
    reaction_certificate = replace(
        _descriptive_level_two_certificate(),
        canonical_role="reaction",
    )
    graph = _strict_graph(
        "transport",
        "equivalence_certificate_check",
        {"certificate": reaction_certificate},
        "equivalence_certificate_audit",
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    mismatch = next(
        issue
        for issue in report.errors
        if issue.code == "equivalence_certificate_role_mismatch"
    )
    assert mismatch.expected == "transport"
    assert mismatch.got == "reaction"


def test_phase8_strict_reaction_rejects_stoichiometry_only_context():
    graph = _strict_graph(
        "reaction",
        "reaction_network_check",
        {"stoichiometric_N": [[1.0]], "strict": False},
        "reaction_network_audit",
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is False


def test_phase8_strict_reaction_rejects_unbound_iphs_and_reaction_evidence():
    iphs_spec = IPHSStructureSpec(
        name="unbound_reaction_iphs",
        state_vars=("reactant", "product"),
        energy_functional="H",
        entropy_functional="S",
        J=torch.zeros((2, 2)),
        M=torch.zeros((2, 2)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        R=torch.eye(2),
        energy_gradient=torch.ones(2),
        contribution_count=1,
        contribution_sources=("test.strict_reaction_runtime",),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "n0": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=["reactant", "product"],
            )
        },
        links=[],
        metadata={"phase8_strict": True, "canonical_role": "reaction"},
        assertion={
            "assertions": [
                {
                    "phi": "strict_iphs",
                    "op": "iphs_structure_check",
                    "scope": "reaction_process",
                    "context": {"iphs_spec": iphs_spec},
                    "audit_key": "iphs_audit",
                },
                {
                    "phi": "closed_reaction_ledger",
                    "op": "reaction_network_check",
                    "scope": "reaction_process",
                    "context": {
                        "stoichiometric_N": [[-1.0], [1.0]],
                        "conservation_matrix": [[1.0, 1.0]],
                        "rates": [0.25],
                        "affinities": [2.0],
                        "species_units": ["mol", "mol"],
                        "rate_units": "mol s-1",
                        "affinity_units": "J mol-1",
                        "affinity_rate_power_units": "J s-1",
                    },
                    "audit_key": "reaction_network_audit",
                },
            ]
        },
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert any(
        issue.code == "strict_reaction_iphs_closure_not_implemented"
        for issue in report.errors
    )


def test_phase8_strict_jump_cannot_use_memory_declaration_as_substitute():
    graph = _strict_graph(
        "field",
        "stochastic_sdae_check",
        {
            "strict": False,
            "required_semantics": ["memory"],
            "memory_kernels": [
                {
                    "name": "passive_memory",
                    "kernel_operator": [[0.25]],
                    "hidden_storage_space": {
                        "complex_id": "K",
                        "semantic_name": "memory_state",
                    },
                    "effort_variable": "e_h",
                    "passivity_evidence": "positive_real_realization",
                    "parameter_keys": ["memory.relaxation_time"],
                    "residual_route": "diagnostics.memory_residual",
                }
            ],
            "memory_restart": {
                "checkpoint_owner": "relic.core.simulator.checkpoint",
                "history_state_keys": ["memory.hidden_state"],
                "history_cursor": "memory.history_cursor",
            },
        },
        "stochastic_sdae_audit",
        jump_extension=True,
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is False
    assert "marked_jump" in report.assertion_audit["structural_result"]["message"] or any(
        issue.code == "structural_assertion_failed" for issue in report.errors
    )


def test_phase8_strict_empty_overflow_state_and_observation_contexts_fail():
    cases = (
        ("transport", "overflow_routing_check", {}, "overflow_routing_audit"),
        ("transport", "state_domain_check", {}, "state_domain_audit"),
        ("observation", "observation_diagnostic_check", {}, "observation_audit"),
    )
    for role, op, context, audit_key in cases:
        report = validate_graph(
            _strict_graph(role, op, context, audit_key),
            NodeRegistry(SSOTCatalog()),
            ssot_catalog=SSOTCatalog(),
        )
        assert report.assertion_audit["structural_result"]["passed"] is False, op


def test_phase8_strict_state_domain_cannot_replace_graph_state_with_decoy():
    graph = _strict_graph(
        "transport",
        "state_domain_check",
        {
            "state": {"decoy": torch.tensor([1.0])},
            "state_domain_constraints": {
                "decoy": {"domain": "real", "units": "1"},
            },
            "required_state_vars": [],
        },
        "state_domain_audit",
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    result = report.assertion_audit["structural_result"]
    assert result["passed"] is False
    assert "soil_water_storage" in result["message"]


def test_phase8_strict_state_domain_uses_graph_state_identity_only():
    graph = _strict_graph(
        "transport",
        "state_domain_check",
        {
            "state": {"soil_water_storage": torch.tensor([1.0])},
            "state_domain_constraints": {
                "soil_water_storage": {
                    "nonnegative": True,
                    "units": "kg m-2",
                },
            },
            "required_state_vars": ["decoy"],
        },
        "state_domain_audit",
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    result = report.assertion_audit["structural_result"]
    assert result["passed"] is True
    assert result["metrics"]["checked_count"] == 1
    assert result["metrics"]["required_count"] == 1
    assert result["metrics"]["extra_count"] == 0


def test_phase8_strict_observation_cannot_self_provide_required_state():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "self_observation": NodeSpec(
                class_type="Observation:Diagnostic",
                inputs={"diagnostics.self": {}},
                outputs=["diagnostics.self"],
            )
        },
        links=[],
        metadata={"phase8_strict": True, "canonical_role": "observation"},
        assertion={
            "phi": "observation_diagnostic_check",
            "op": "observation_diagnostic_check",
            "scope": "phase8_strict",
            "context": {
                "required_observation_kinds": ["observation"],
                "required_observation_state_vars": ["diagnostics.self"],
                "observation_specs": [
                    {
                        "name": "self_observation",
                        "input_state_vars": ["diagnostics.self"],
                        "output_name": "diagnostics.self",
                        "output_units": "1",
                        "support": "point",
                        "averaging_window": "instantaneous",
                        "footprint_description": "single state location",
                        "error_model": "gaussian",
                    }
                ],
            },
            "audit_key": "observation_audit",
        },
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is False
    assert any(issue.code == "structural_assertion_failed" for issue in report.errors)


def test_phase8_strict_observation_cannot_consume_diagnostic_node_output_as_state():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "diagnostic_provider": NodeSpec(
                class_type="Observation:Diagnostic",
                outputs=["diagnostics.other"],
            )
        },
        links=[],
        metadata={"phase8_strict": True, "canonical_role": "observation"},
        assertion={
            "phi": "observation_diagnostic_check",
            "op": "observation_diagnostic_check",
            "scope": "phase8_strict",
            "context": {
                "required_observation_kinds": ["observation"],
                "required_observation_state_vars": ["diagnostics.other"],
                "observation_specs": [
                    {
                        "name": "derived_observation",
                        "input_state_vars": ["diagnostics.other"],
                        "output_name": "observations.derived",
                        "output_units": "1",
                        "support": "point",
                        "averaging_window": "instantaneous",
                        "footprint_description": "single state location",
                        "error_model": "gaussian",
                    }
                ],
            },
            "audit_key": "observation_audit",
        },
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is False


def test_phase8_strict_observation_accepts_scientific_state_node_output():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "soil_state": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=["soil_carbon"],
            ),
            "soil_observation_operator": NodeSpec(
                class_type="Observation:Operator",
                inputs={"soil_carbon": {"state": "soil_carbon"}},
                outputs=["observations.soil_carbon"],
            ),
        },
        links=[],
        metadata={"phase8_strict": True, "canonical_role": "observation"},
        assertion={
            "phi": "observation_diagnostic_check",
            "op": "observation_diagnostic_check",
            "scope": "soil_observation_operator",
            "context": {
                "required_observation_kinds": ["observation"],
                "required_observation_state_vars": ["soil_carbon"],
                "observation_specs": [
                    {
                        "name": "soil_carbon_observation",
                        "input_state_vars": ["soil_carbon"],
                        "output_name": "observations.soil_carbon",
                        "output_units": "kg C m-2",
                        "support": "plot",
                        "averaging_window": "instantaneous",
                        "footprint_description": "plot support",
                        "error_model": "gaussian",
                    }
                ],
            },
            "audit_key": "observation_audit",
        },
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is True


def test_phase8_strict_observation_rejects_cross_node_state_substitution():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "soil_state": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=["soil_carbon"],
            ),
            "tower_operator": NodeSpec(
                class_type="Observation:Operator",
                inputs={"canopy_temperature": {}},
                outputs=["observations.soil_carbon"],
            ),
        },
        links=[],
        metadata={"phase8_strict": True, "canonical_role": "observation"},
        assertion={
            "phi": "observation_diagnostic_check",
            "op": "observation_diagnostic_check",
            "scope": "tower_operator",
            "context": {
                "required_observation_kinds": ["observation"],
                "required_observation_state_vars": ["soil_carbon"],
                "available_state_vars": ["soil_carbon"],
                "observation_specs": [
                    {
                        "name": "soil_carbon_observation",
                        "input_state_vars": ["soil_carbon"],
                        "output_name": "observations.soil_carbon",
                        "output_units": "kg C m-2",
                        "support": "plot",
                        "averaging_window": "instantaneous",
                        "footprint_description": "plot support",
                        "error_model": "gaussian",
                    }
                ],
            },
            "audit_key": "observation_audit",
        },
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is False
    assert "unavailable state variables" in report.assertion_audit[
        "structural_result"
    ]["message"]


def test_phase8_strict_observation_accepts_exact_operator_node_binding():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "soil_state": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=["soil_carbon"],
            ),
            "tower_operator": NodeSpec(
                class_type="Observation:Operator",
                inputs={"soil_carbon": {"state": "soil_carbon"}},
                outputs=["observations.soil_carbon"],
            ),
        },
        links=[],
        metadata={"phase8_strict": True, "canonical_role": "observation"},
        assertion={
            "phi": "observation_diagnostic_check",
            "op": "observation_diagnostic_check",
            "scope": "tower_operator",
            "context": {
                "required_observation_kinds": ["observation"],
                "required_observation_state_vars": ["soil_carbon"],
                "available_state_vars": ["decoy"],
                "observation_specs": [
                    {
                        "name": "soil_carbon_observation",
                        "input_state_vars": ["soil_carbon"],
                        "output_name": "observations.soil_carbon",
                        "output_units": "kg C m-2",
                        "support": "plot",
                        "averaging_window": "instantaneous",
                        "footprint_description": "plot support",
                        "error_model": "gaussian",
                    }
                ],
            },
            "audit_key": "observation_audit",
        },
    )

    report = validate_graph(
        graph,
        NodeRegistry(SSOTCatalog()),
        ssot_catalog=SSOTCatalog(),
    )

    assert report.assertion_audit["structural_result"]["passed"] is True
    assert (
        report.assertion_audit["structural_result"]["metrics"][
            "graph_binding_verified"
        ]
        == 1.0
    )


def _ordinary_structural_graph(*, value=1.0, units="kg m-2", role="reaction"):
    graph = _graph({
        "phi": "declared storage domain",
        "op": "state_domain_check",
        "scope": "state",
        "context": {
            "state": {"soil_water_storage": torch.tensor([value])},
            "state_domain_constraints": {
                "soil_water_storage": {"nonnegative": True, "units": units},
            },
            "strict": False,
        },
    })
    graph.metadata = {"strict_structural_assertions": True, "canonical_role": role}
    return graph


def test_ordinary_structural_role_does_not_require_unrepresented_iphs_objects():
    report = validate_graph(
        _ordinary_structural_graph(), NodeRegistry(SSOTCatalog()),
    )

    assert report.errors == []
    assert report.assertion_audit["structural_result"]["passed"] is True
    assert "required_runtime_level" not in report.assertion_audit["structural_result"]["metrics"]


@pytest.mark.parametrize(("value", "units"), [(-1.0, "kg m-2"), (1.0, "banana")])
def test_ordinary_structural_strict_keeps_state_and_unit_rejection(value, units):
    report = validate_graph(
        _ordinary_structural_graph(value=value, units=units),
        NodeRegistry(SSOTCatalog()),
    )

    assert report.assertion_audit["structural_result"]["passed"] is False
    assert not any(item.code == "missing_required_structural_assertion" for item in report.errors)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_descriptive_graph_level_without_advanced_claim_needs_no_iphs_objects(level):
    graph = _graph({})
    graph.metadata = {"strict_structural_assertions": True, "iphs_level": level}

    assert validate_graph(graph, NodeRegistry(SSOTCatalog())).errors == []


def test_ordinary_certificate_level_is_descriptive_not_runtime_minimum():
    graph = _graph({
        "phi": "declared compatibility",
        "op": "equivalence_certificate_check",
        "scope": "state",
        "context": {"certificate": _descriptive_level_two_certificate(), "iphs_level": 2},
    })
    graph.metadata = {
        "strict_structural_assertions": True, "canonical_role": "transport", "iphs_level": 2,
    }

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()))

    assert report.errors == []
    metrics = report.assertion_audit["structural_result"]["metrics"]
    assert metrics["runtime_verified_level"] == 0
    assert metrics["required_runtime_level"] == -1


@pytest.mark.parametrize("level", [1, 3, 4, 5])
def test_explicit_runtime_minimum_remains_binding_for_ordinary_graph(level):
    graph = _graph({
        "phi": "runtime capability requested",
        "op": "equivalence_certificate_check",
        "scope": "state",
        "context": {"certificate": _descriptive_level_two_certificate(), "required_level": 0},
    })
    graph.metadata = {"required_level": level, "canonical_role": "transport"}

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()))

    assert report.assertion_audit["structural_result"]["passed"] is False
    assert report.assertion_audit["structural_result"]["metrics"]["required_runtime_level"] == level


@pytest.mark.parametrize("metadata", [
    {"iphs_level": 4}, {"iphs_level": 5}, {"required_level": 1},
    {"required_level": 5}, {"iphs_level": True}, {"required_level": None},
])
def test_explicit_scientific_requirement_cannot_disappear_without_assertions(metadata):
    graph = _graph({})
    graph.metadata = metadata

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()))

    assert report.errors


def _ordinary_reaction_graph():
    graph = _graph({
        "phi": "closed amount ledger",
        "op": "reaction_network_check",
        "scope": "state",
        "context": {
            "stoichiometric_N": [[-1.0], [1.0]],
            "conservation_matrix": [[1.0, 1.0]],
            "rates": [0.2],
            "species": ["A", "B"], "reactions": ["transfer"],
            "conservation_basis": ["amount"], "rate_reaction_order": ["transfer"],
            "species_units": ["mol", "mol"], "rate_units": "mol s-1",
            "state_domain": "reactor", "reaction_domain": "reactor",
            "conservation_domain": "reactor", "strict": False,
        },
    })
    graph.metadata = {"strict_structural_assertions": True, "canonical_role": "reaction"}
    return graph


def test_ordinary_structural_reaction_requires_budget_but_not_affinity():
    graph = _ordinary_reaction_graph()
    report = validate_graph(graph, NodeRegistry(SSOTCatalog()))
    assert report.errors == []
    assert report.assertion_audit["structural_result"]["metrics"]["stoichiometric_conservation_residual"] == 0.0

    graph.assertion["context"]["conservation_matrix"] = [[1.0, 2.0]]
    invalid = validate_graph(graph, NodeRegistry(SSOTCatalog()))
    assert invalid.assertion_audit["structural_result"]["passed"] is False


@pytest.mark.parametrize("field", ["iphs_level", "required_level"])
@pytest.mark.parametrize("value", [None, True, "5", 5.0, -1, 6])
def test_non_iphs_assertion_cannot_hide_invalid_graph_level(field, value):
    graph = _ordinary_structural_graph()
    graph.metadata[field] = value

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()))

    assert any(item.code == "invalid_iphs_level_threshold" for item in report.errors)
    assert report.assertion_audit["valid"] is False


@pytest.mark.parametrize("missing_map", ["state_map", "unit_map", "interface_map"])
def test_ordinary_strict_still_checks_declared_certificate_maps(missing_map):
    graph = _ordinary_structural_graph(role="transport")
    graph.assertion = {
        "phi": "declared compatibility", "op": "equivalence_certificate_check",
        "scope": "state",
        "context": {"certificate": replace(
            _descriptive_level_two_certificate(), **{missing_map: {}},
        )},
    }

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()))

    assert report.assertion_audit["structural_result"]["passed"] is False
    assert report.assertion_audit["structural_result"]["metrics"]["required_runtime_level"] == -1


def test_ordinary_graph_honors_explicit_affinity_requirement():
    graph = _ordinary_reaction_graph()
    assert validate_graph(graph, NodeRegistry(SSOTCatalog())).errors == []
    graph.assertion["context"]["require_affinity"] = True

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()))

    assert report.assertion_audit["structural_result"]["passed"] is False


@pytest.mark.parametrize("value", [None, "false", 0])
def test_graph_rejects_invalid_explicit_affinity_requirement(value):
    graph = _ordinary_reaction_graph()
    graph.assertion["context"]["require_affinity"] = value

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()))

    assert any(item.code == "invalid_reaction_affinity_requirement" for item in report.errors)


@pytest.mark.parametrize("metadata", [
    {"required_level": 0},
    {"strict_structural_assertions": True, "required_level": 0},
    {"iphs_assembly": {"required_level": 0}},
    {"iphs_assembly": {"iphs_level": 2}},
])
def test_zero_runtime_floor_and_descriptive_assembly_do_not_invent_certificates(metadata):
    graph = _graph({})
    graph.metadata = metadata

    assert validate_graph(graph, NodeRegistry(SSOTCatalog())).errors == []


@pytest.mark.parametrize("assembly", [
    {"required_level": 1}, {"required_level": 2}, {"required_level": 3},
    {"iphs_level": 4}, {"iphs_level": 5}, {"strict": True},
    {"required_level": None}, {"iphs_level": True},
])
def test_assembly_requirement_is_validated_even_without_graph_assertions(assembly):
    graph = _graph({})
    graph.metadata = {"iphs_assembly": assembly}

    assert validate_graph(graph, NodeRegistry(SSOTCatalog())).errors


@pytest.mark.parametrize("level", [1, 2, 3])
def test_assembly_runtime_minimum_uses_existing_certificate_runtime_evidence(level):
    graph = _graph({
        "phi": "requested assembly runtime", "op": "equivalence_certificate_check",
        "scope": "state", "context": {"certificate": _descriptive_level_two_certificate()},
    })
    graph.metadata = {"iphs_assembly": {"required_level": level}}

    report = validate_graph(graph, NodeRegistry(SSOTCatalog()))

    result = report.assertion_audit["structural_result"]
    assert result["passed"] is False
    assert result["metrics"]["required_runtime_level"] == level
    assert result["metrics"]["runtime_verified_level"] == 0


def test_certificate_node_resolution_propagates_resolver_defects(monkeypatch):
    """Only the resolver's contract errors become an unresolved-node issue."""
    graph, registry = _certificate_process_graph(_level_four_transport_certificate())

    def _resolver_defect(class_type):
        raise AttributeError(f"resolver defect for {class_type}")

    monkeypatch.setattr(registry, "resolve_class_type", _resolver_defect)

    with pytest.raises(AttributeError, match="resolver defect"):
        validate_graph(graph, registry, ssot_catalog=SSOTCatalog())


def test_strict_entropy_binding_reports_an_unloadable_certificate(tmp_path):
    """A certificate that cannot be loaded never lets the strict binding pass silently."""
    import relic.orchestration.validator as validator_module

    missing_certificate = tmp_path / "missing_certificate.json"
    items = [
        {"op": "entropy_step_check", "context": {"tolerance_parameter_key": "validation.entropy.tol"}},
        {"op": "equivalence_certificate_check", "context": {"certificate": str(missing_certificate)}},
    ]

    issues = validator_module._strict_entropy_tolerance_binding_issues(
        items, {"phase8_strict": True}
    )

    assert [issue.code for issue in issues] == ["strict_entropy_tolerance_binding_unverifiable"]
    assert "missing_certificate.json" in issues[0].got


@pytest.mark.parametrize(
    "evidence",
    [
        {
            "dissipation_audit": {"tolerance_parameter_key": "test.iphs.dissipation_tol"},
            "port_power_audit": {"tolerance_parameter_key": "test.iphs.port_power_tol"},
        },
        {},
    ],
    ids=["two_keys", "no_key"],
)
def test_strict_entropy_binding_reports_a_certificate_without_one_tolerance_key(evidence):
    """The binding is unverifiable unless the certificate owns exactly one strict key."""
    import relic.orchestration.validator as validator_module

    certificate = replace(_level_four_transport_certificate(), iphs_level_evidence=evidence)
    items = [
        {"op": "entropy_step_check", "context": {"tolerance_parameter_key": "test.iphs.tolerance"}},
        {"op": "equivalence_certificate_check", "context": {"certificate": certificate}},
    ]

    issues = validator_module._strict_entropy_tolerance_binding_issues(
        items, {"phase8_strict": True}
    )

    assert [issue.code for issue in issues] == ["strict_entropy_tolerance_binding_unverifiable"]
