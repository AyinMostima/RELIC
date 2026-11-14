"""Tests for global IPHS-DEC compiler audit facade.

Graft static zero in-degree is not a dead-code signal for this public facade;
runtime callers enter through ``RELICModelCompiler.compile`` and GraphIR data.
"""

import inspect
import json
import re
import sys

import pytest
import torch

import relic.orchestration.compiler as compiler_module
from relic.computation.solvers import (
    BracketedBisectionSolver,
    IntegrationPlan,
    IntegrationStage,
)
from relic.orchestration.compiler import (
    RELICModelCompiler,
    build_global_structural_audit,
    compile_graph,
    plan_to_dict,
)
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog, SSOTEntry
from relic.orchestration.validator import ValidationReport
from relic.validation.framework.registry import get_global_registry
from relic.validation.framework.rules.hodge import (
    validate_hodge_passivity_structure,
    validate_hodge_spec,
)
from relic.validation.framework.rules.iphs import (
    validate_iphs_spec,
    validate_iphs_structure,
)
from relic.validation.framework.rules.reaction_network import (
    validate_reaction_network_payload,
    validate_reaction_network_structure,
)
from test.fixtures.dummy_functor import (
    DummyConfiguredOutputFunctor,
    DummyIPHSFunctor,
    DummyIncrementFunctor,
)


def _dimensionless_catalog(*names: str) -> SSOTCatalog:
    catalog = SSOTCatalog()
    for name in names:
        catalog.add_entry(
            SSOTEntry(
                name=name,
                standard_name=name,
                units="1",
            )
        )
    return catalog


def _graph(metadata=None, assertion=None):
    return GraphIR(
        schema_version="1.0.0",
        nodes={"n0": NodeSpec(class_type="State:ExternalForcing", outputs=["forcing"])},
        links=[],
        metadata=metadata or {},
        assertion=assertion or {},
    )


def _nonlinear_plan_payload(runtime_node_id="n0"):
    return IntegrationPlan(
        plan_id="farquhar_atomic_closure",
        scheme="single",
        stages=(
            IntegrationStage(
                stage_id="farquhar_ci_closure",
                operator_kind="nonlinear_algebraic_closure",
                solver_name="bracketed_bisection",
                structure_kind="algebraic_constraint",
                dt_fraction=0.0,
                advances_time=False,
                runtime_owner="solver_registry",
                reads=("atmospheric_co2_ppm", "ppfd"),
                writes=("intercellular_co2",),
                validation_hooks=("state_domain_check",),
                stage_metadata={
                    "runtime_node_id": runtime_node_id,
                    "unknown_variables": ["intercellular_co2"],
                    "residual_id": "farquhar_medlyn_ci_residual",
                    "bracket_source": "parameter_manager",
                    "requires_sign_change": True,
                    "convergence_required": True,
                    "failure_policy": "raise",
                    "execution_binding": "functor_forward",
                    "solver_parameter_keys": [
                        "computation.solvers.bracketed_bisection.max_iterations",
                        "computation.solvers.bracketed_bisection.rtol",
                        "computation.solvers.bracketed_bisection.atol",
                    ],
                    "bracket_parameter_keys": [
                        "biosphere.photosynthesis.farquhar.ci_lower_fraction",
                        "biosphere.photosynthesis.farquhar.ci_upper_fraction",
                    ],
                }, coupling_mode="direct",
            ),
        ),
    ).to_dict()


def _executable_graph(metadata=None):
    registry = NodeRegistry(_dimensionless_catalog("x"))
    definition = registry.register_functor_instance(DummyIncrementFunctor())
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "n1": NodeSpec(
                class_type=definition.node_type_id,
                outputs=["x"],
            )
        },
        links=[],
        metadata=metadata or {},
    )
    return graph, registry


def _executable_iphs_graph(metadata=None):
    registry = NodeRegistry(_dimensionless_catalog("x"))
    definition = registry.register_functor_instance(DummyIPHSFunctor())
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "n1": NodeSpec(
                class_type=definition.node_type_id,
                outputs=["x"],
            )
        },
        links=[],
        metadata=metadata or {},
    )
    return graph, registry


def _audit_rejected_assembly(graph, registry):
    """Inspect local identity evidence without compiling the rejected claim."""
    with pytest.raises(ValueError, match="structural assertions"):
        compile_graph(graph, registry, strict=False)
    ordinary_graph = GraphIR(
        schema_version=graph.schema_version,
        nodes=graph.nodes,
        links=graph.links,
        metadata={},
    )
    ordinary_plan = compile_graph(ordinary_graph, registry, strict=False)
    validation = compiler_module.validate_graph(graph, registry)
    assert validation.errors
    audit = build_global_structural_audit(
        graph, validation, metadata=ordinary_plan.report.metadata,
        registry=registry, execution_steps=ordinary_plan.steps,
    )
    assert audit["global_compile_audit"]["passed"] is False
    return audit["iphs_audit"]


def test_compile_graph_does_not_treat_metadata_pass_flags_as_executed_evidence():
    integration_plan = IntegrationPlan(
        plan_id="phase8_gate",
        stages=(IntegrationStage(stage_id="iphs_step", operator_kind="iphs", solver_name="implicit_midpoint", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct"),),
    )
    graph = _graph(
        metadata={
            "structural_audits": {
                "port_matching_audit": {"passed": True},
                "overflow_routing_audit": {"passed": True},
            },
            "integration_plan": integration_plan.to_dict(),
        }
    )

    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()))
    report = plan_to_dict(plan)["report"]

    assert report["audit"]["port_matching_audit"]["passed"] is None
    assert report["audit"]["port_matching_audit"]["declared_passed"] is True
    assert report["audit"]["port_matching_audit"]["verified"] is False
    assert report["audit"]["overflow_routing_audit"]["passed"] is None
    assert report["audit"]["overflow_routing_audit"]["declared_passed"] is True
    assert report["audit"]["integration_plan_audit"]["passed"] is True
    assert report["audit"]["integration_plan_audit"]["structurally_validated"] is True
    assert report["audit"]["integration_plan_audit"]["verified"] is False
    assert report["audit"]["integration_plan_audit"]["runtime_ready"] is False
    assert (
        report["audit"]["integration_plan_audit"]["execution_semantics"]
        == "metadata_only"
    )
    assert report["audit"]["integration_plan_audit"]["solver_execution_count"] == 0
    assert report["audit"]["global_compile_audit"]["passed"] is True


def test_compile_graph_revalidates_json_nonlinear_plan_without_executing_solver(
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("compile_graph must not execute nonlinear solvers")

    monkeypatch.setattr(BracketedBisectionSolver, "solve", fail_if_called)
    graph, registry = _executable_graph(
        metadata={"integration_plan": _nonlinear_plan_payload("n1")}
    )

    plan = compile_graph(graph, registry)
    report = plan_to_dict(plan)["report"]

    assert [step.node_id for step in plan.steps] == ["n1"]
    assert report["audit"]["integration_plan_audit"]["passed"] is True
    assert report["audit"]["integration_plan_audit"]["structurally_validated"] is True
    assert report["audit"]["integration_plan_audit"]["verified"] is False
    assert report["audit"]["integration_plan_audit"]["runtime_ready"] is False
    assert report["audit"]["integration_plan_audit"]["solver_execution_count"] == 0


def test_compile_graph_rejects_forged_or_wrong_family_plan_mapping():
    payload = _nonlinear_plan_payload()
    payload["audit"] = {"passed": True}
    payload["stages"][0]["solver_name"] = "implicit_midpoint"

    with pytest.raises(ValueError, match="integration_plan_audit"):
        compile_graph(
            _graph(metadata={"integration_plan": payload}),
            NodeRegistry(SSOTCatalog()),
        )


def test_compile_graph_rejects_missing_nonlinear_runtime_node():
    payload = _nonlinear_plan_payload(runtime_node_id="missing_functor")

    with pytest.raises(ValueError, match="integration_plan_audit"):
        compile_graph(
            _graph(metadata={"integration_plan": payload}),
            NodeRegistry(SSOTCatalog()),
        )


def test_compile_graph_rejects_boundary_as_nonlinear_runtime_node():
    payload = _nonlinear_plan_payload(runtime_node_id="n0")

    with pytest.raises(ValueError, match="integration_plan_audit"):
        compile_graph(
            _graph(metadata={"integration_plan": payload}),
            NodeRegistry(SSOTCatalog()),
        )


def test_compile_graph_rejects_python_object_integration_plan_representation():
    object_plan = IntegrationPlan(
        plan_id="object_representation",
        stages=(IntegrationStage(stage_id="iphs_step", operator_kind="iphs", solver_name="implicit_midpoint", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct"),),
    )

    with pytest.raises(ValueError, match="integration_plan_audit"):
        compile_graph(
            _graph(metadata={"integration_plan": object_plan}),
            NodeRegistry(SSOTCatalog()),
        )


def test_compile_graph_assembles_declared_iphs_metadata():
    graph = _graph(
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "compile_time_iphs",
                "state_vars": ("x", "s"),
                "energy_functional": "E",
                "entropy_functional": "S",
                "contributions": [
                    {"J": torch.tensor([[0.0, 1.0], [-1.0, 0.0]])},
                    {"M": torch.diag(torch.tensor([0.0, 0.2]))},
                ],
            }
        }
    )

    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()))
    report = plan_to_dict(plan)["report"]

    assert report["audit"]["iphs_audit"]["enabled"] is True
    assert report["audit"]["iphs_audit"]["passed"] is True
    assert report["audit"]["iphs_audit"]["state_var_count"] == 2


@pytest.mark.parametrize("operator", ["matrix_snapshot", "zero_field"])
def test_compile_graph_preserves_poisson_policy_through_json_and_audit(operator):
    assembly = {
        "state_vars": ["energy", "coordinate"], "formulation": "generic",
        "poisson_operator": operator, "reaction_representation": "independent",
        "contributions": [{"J": [[0.0, 0.0], [0.0, 0.0]],
                           "M": [[0.0, 0.0], [0.0, 2.0]],
                           "energy_gradient": [1.0, 0.0], "entropy_gradient": [0.0, 3.0]}],
    }
    graph = _graph(metadata={"iphs_assembly": assembly})
    graph = GraphIR.from_dict(json.loads(json.dumps(graph.to_dict())))
    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()))
    validation = plan_to_dict(plan)["report"]["audit"]["iphs_audit"]["validation"]
    assert validation["poisson_operator"] == operator
    if operator == "zero_field":
        assert validation["jacobi_identity_basis"] == "canonical_zero_field"
        assert validation["evidence_scope"] == "zero_field_structural_pre_gate"
        assert validation["process_level_status"] == "[N] runtime_unbound"
    assert "runtime_verified_level" not in validation


@pytest.mark.parametrize("operator,magnitude", [("zero_field", 1.0), ("zero_field", 1.0e-50),
                                                ("invalid_poisson_kind", 0.0)])
def test_compile_graph_rejects_invalid_poisson_policy_instead_of_using_default(operator, magnitude):
    assembly = {
        "reaction_representation": "independent",
        "state_vars": ["x", "y"], "poisson_operator": operator,
        "contributions": [{"J": [[0.0, magnitude], [-magnitude, 0.0]]}],
    }
    graph = _graph(metadata={"iphs_assembly": assembly})
    audit = compiler_module._run_iphs_assembly_audit(assembly, graph)
    assert audit["passed"] is False
    assert "zero_field" in audit["message"] or "poisson_operator" in audit["message"]
    with pytest.raises(ValueError, match="Structural audit failed:.*iphs_audit"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()))


# MA-21: GraphIR metadata forwards the executable Poisson-field declaration.
# References: Grmela and Ottinger (1997); Marsden and Ratiu (1999) Ch. 10.
# Mathematical expression: J(x)_ij = eps_ijk x_k (so(3) Lie-Poisson tensor); the
#   compiled spec checks J(x)^T = -J(x) and the Jacobi residual at each sample.
# Parameter meanings: poisson_field is an importable 'module:attribute' reference to
#   the callable J(x); poisson_domain is a list of state vectors; jacobi_tol is the
#   residual tolerance.
# Units: normalized dimensionless coordinates (dyadic sample values keep float32 exact).
# Domain/constraints: JSON-safe metadata only; anything else is rejected.
# Sign convention: not applicable to an algebraic identity residual.
_POISSON_FIXTURE_MODULE = "relic_ma21_poisson_field_fixture"
_SO3_REFERENCE_J = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
_SO3_DOMAIN = [[0.0, 0.0, 1.0], [0.5, -0.5, 2.0], [-1.0, 2.0, 0.5]]


def _fixture_so3_field(x):
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.stack(
        [
            torch.stack([zero, x[2], -x[1]]),
            torch.stack([-x[2], zero, x[0]]),
            torch.stack([x[1], -x[0], zero]),
        ]
    )


@pytest.fixture
def poisson_field_module(monkeypatch):
    import sys
    import types

    module = types.ModuleType(_POISSON_FIXTURE_MODULE)
    module.so3_field = _fixture_so3_field
    module.fields = types.SimpleNamespace(so3=_fixture_so3_field)
    module.constant_field = lambda x: torch.tensor(_SO3_REFERENCE_J, dtype=x.dtype, device=x.device)
    module.wrong_shape_field = lambda x: torch.zeros((2, 2), dtype=x.dtype, device=x.device)
    module.NOT_CALLABLE = 3.0
    monkeypatch.setitem(sys.modules, _POISSON_FIXTURE_MODULE, module)
    return module


def _poisson_field_assembly(**overrides):
    assembly = {
        "reaction_representation": "independent",
        "name": "state_field_compile", "state_vars": ["x0", "x1", "x2"],
        "formulation": "generic", "poisson_operator": "state_field",
        "reversible_operator_state_dependence": "state_dependent",
        "poisson_field": f"{_POISSON_FIXTURE_MODULE}:so3_field",
        "poisson_domain": [list(row) for row in _SO3_DOMAIN],
        "jacobi_tol": 1.0e-6,
        "contributions": [{"J": [list(row) for row in _SO3_REFERENCE_J]}],
    }
    assembly.update(overrides)
    return assembly


@pytest.mark.parametrize("reference", ["so3_field", "fields.so3"])
def test_compile_graph_resolves_a_state_field_declaration_from_graph_metadata(poisson_field_module, reference):
    graph = _graph(metadata={"iphs_assembly": _poisson_field_assembly(
        poisson_field=f"{_POISSON_FIXTURE_MODULE}:{reference}",
    )})
    graph = GraphIR.from_dict(json.loads(json.dumps(graph.to_dict())))
    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()))
    audit = plan_to_dict(plan)["report"]["audit"]["iphs_audit"]
    assert audit["passed"] is True
    validation = audit["validation"]
    assert validation["poisson_operator"] == "state_field"
    assert validation["jacobi_identity_basis"] == "sampled_domain_jacobi_residual"
    assert validation["jacobi_sample_count"] == 3
    assert validation["jacobi_tolerance"] == pytest.approx(1.0e-6)
    assert validation["jacobi_max_residual"] <= 1.0e-6
    assert "runtime_verified_level" not in validation


def test_compile_graph_without_jacobi_tol_uses_the_parameter_database_tolerance(poisson_field_module):
    from relic.parameters import get_global_manager

    assembly = _poisson_field_assembly()
    del assembly["jacobi_tol"]
    plan = compile_graph(_graph(metadata={"iphs_assembly": assembly}), NodeRegistry(SSOTCatalog()))
    validation = plan_to_dict(plan)["report"]["audit"]["iphs_audit"]["validation"]
    expected = get_global_manager().get("validation.iphs.jacobi_tolerance", required=True)
    assert validation["jacobi_tolerance"] == pytest.approx(expected)


_NEARLY_SKEW_J = [[0.0, 1.0], [-0.99999, 0.0]]


def test_compile_graph_forwards_a_declared_structure_tolerance():
    # J + J^T has norm about 1.4e-5: refused at the database default, accepted
    # when the assembly explicitly declares skew_tol = 1e-4.
    default = {"poisson_operator": "matrix_snapshot", "reaction_representation": "independent", "state_vars": ["x", "y"], "contributions": [{"J": _NEARLY_SKEW_J}]}
    graph = _graph(metadata={"iphs_assembly": default})
    assert compiler_module._run_iphs_assembly_audit(default, graph)["passed"] is False
    declared = dict(default, skew_tol=1.0e-4)
    plan = compile_graph(_graph(metadata={"iphs_assembly": declared}), NodeRegistry(SSOTCatalog()))
    assert plan_to_dict(plan)["report"]["audit"]["iphs_audit"]["passed"] is True


@pytest.mark.parametrize("field_name", ["skew_tol", "psd_tol", "degeneracy_tol"])
@pytest.mark.parametrize("value", [None, "1e-10", True, [1.0e-10]])
def test_compile_graph_rejects_a_declared_null_or_non_numeric_structure_tolerance(field_name, value):
    assembly = {"poisson_operator": "matrix_snapshot", "reaction_representation": "independent", "state_vars": ["x", "y"], "contributions": [{"J": [[0.0, 1.0], [-1.0, 0.0]]}],
                field_name: value}
    graph = _graph(metadata={"iphs_assembly": assembly})
    audit = compiler_module._run_iphs_assembly_audit(assembly, graph)
    assert audit["passed"] is False
    assert f"{field_name} must be a finite non-negative number" in audit["message"], audit["message"]
    with pytest.raises(ValueError, match="Structural audit failed:.*iphs_audit"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()))


def test_compile_graph_checks_a_constant_field_callable_on_the_declared_samples(poisson_field_module):
    assembly = _poisson_field_assembly(
        poisson_operator="constant_field", reversible_operator_state_dependence="constant",
        poisson_field=f"{_POISSON_FIXTURE_MODULE}:constant_field",
    )
    plan = compile_graph(_graph(metadata={"iphs_assembly": assembly}), NodeRegistry(SSOTCatalog()))
    validation = plan_to_dict(plan)["report"]["audit"]["iphs_audit"]["validation"]
    assert validation["jacobi_identity_basis"] == "constant_field_exact"
    assert validation["jacobi_sample_count"] == 3


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"poisson_field": 42}, "poisson_field must be an importable reference"),
        ({"poisson_field": ["relic", "so3_field"]}, "poisson_field must be an importable reference"),
        ({"poisson_field": f"{_POISSON_FIXTURE_MODULE}.so3_field"}, "poisson_field must be an importable reference"),
        ({"poisson_field": f" {_POISSON_FIXTURE_MODULE}:so3_field"}, "poisson_field must be an importable reference"),
        ({"poisson_field": f"{_POISSON_FIXTURE_MODULE}:"}, "poisson_field must be an importable reference"),
        ({"poisson_field": "relic_ma21_missing_module:field"}, "cannot be imported"),
        ({"poisson_field": f"{_POISSON_FIXTURE_MODULE}:missing_field"}, "does not resolve"),
        ({"poisson_field": f"{_POISSON_FIXTURE_MODULE}:NOT_CALLABLE"}, "does not resolve to a callable"),
        ({"poisson_field": f"{_POISSON_FIXTURE_MODULE}:wrong_shape_field"}, r"must return an \(n, n\) tensor"),
        ({"poisson_domain": "[[0, 0, 1]]"}, "poisson_domain must be a non-empty list of numeric state vectors"),
        ({"poisson_domain": []}, "poisson_domain must be a non-empty list of numeric state vectors"),
        ({"poisson_domain": [0.0, 0.0, 1.0]}, "poisson_domain must be a non-empty list of numeric state vectors"),
        ({"poisson_domain": [[0.0, 0.0, 1.0], [0.5, 1.0]]}, "poisson_domain must be a non-empty list of numeric state vectors"),
        ({"poisson_domain": [[True, False, True]]}, "poisson_domain must be a non-empty list of numeric state vectors"),
        ({"poisson_domain": [["0", "0", "1"]]}, "poisson_domain must be a non-empty list of numeric state vectors"),
        ({"poisson_domain": {"x0": [0.0]}}, "poisson_domain must be a non-empty list of numeric state vectors"),
        ({"poisson_domain": [[0.0, 0.0, 1.0, 2.0]]}, r"poisson_domain must be an \(m, 3\) tensor"),
        ({"poisson_domain": [[0.0, float("nan"), 1.0]]}, "poisson_domain must be finite"),
        ({"jacobi_tol": "1e-8"}, "jacobi_tol must be a finite positive number"),
        ({"jacobi_tol": True}, "jacobi_tol must be a finite positive number"),
        ({"jacobi_tol": float("inf")}, "jacobi_tol must be a finite positive number"),
        ({"jacobi_tol": 0.0}, "jacobi_tol must be a finite positive number"),
        ({"jacobi_tol": None}, "jacobi_tol must be a finite positive number"),
        ({"poisson_operator": "zero_field", "reversible_operator_state_dependence": "constant",
          "contributions": [{"J": [[0.0] * 3] * 3}]}, "apply only to poisson_operator constant_field or state_field"),
        ({"poisson_operator": "matrix_snapshot", "reversible_operator_state_dependence": "unspecified"},
         "apply only to poisson_operator constant_field or state_field"),
    ],
)
def test_compile_graph_rejects_a_malformed_poisson_field_declaration(poisson_field_module, overrides, message):
    assembly = _poisson_field_assembly(**overrides)
    graph = _graph(metadata={"iphs_assembly": assembly})
    audit = compiler_module._run_iphs_assembly_audit(assembly, graph)
    assert audit["passed"] is False
    assert re.search(message, audit["message"]), audit["message"]
    # Non-finite metadata is already refused when GraphIR is sealed for compilation.
    with pytest.raises(
        ValueError,
        match=r"Structural audit failed:.*iphs_audit|Non-finite float is not sealable",
    ):
        compile_graph(graph, NodeRegistry(SSOTCatalog()))


def test_compile_graph_preserves_absorbed_reaction_policy_after_json_roundtrip():
    assembly = {
        "state_vars": ["energy", "coordinate"], "formulation": "generic",
        "poisson_operator": "zero_field", "reaction_representation": "absorbed",
        "contributions": [{"M": [[0.0, 0.0], [0.0, 2.0]],
                           "stoichiometric_N": [[0.0], [1.0]],
                           "energy_gradient": [1.0, 0.0], "entropy_gradient": [0.0, 3.0]}],
    }
    graph = _graph(metadata={"iphs_assembly": assembly})
    graph = GraphIR.from_dict(json.loads(json.dumps(graph.to_dict())))
    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()))
    audit = plan_to_dict(plan)["report"]["audit"]["iphs_audit"]
    assert audit["has_stoichiometric_N"] is True
    assert audit["validation"]["reaction_representation"] == "absorbed"
    assert "runtime_verified_level" not in audit["validation"]


def test_compiler_rejects_invalid_reaction_policy_without_defaulting_to_independent():
    assembly = {
        "poisson_operator": "matrix_snapshot",
        "state_vars": ["x"], "reaction_representation": "invalid_reaction_policy",
        "contributions": [{"M": [[1.0]]}],
    }
    audit = compiler_module._run_iphs_assembly_audit(assembly)
    assert audit["passed"] is False
    assert "reaction_representation" in audit["message"]


def _internal_port(name):
    """A typed internal scalar port: C and D couple internal ports only."""
    from relic.computation.operators.dec.forms import CochainSpaceSpec
    from relic.computation.operators.iphs import PortSpecIPHS

    def _space(kind, primality, units, extensivity):
        return CochainSpaceSpec(
            complex_id="test_complex", degree=0, primality=primality, n_dofs=1,
            units=units, extensivity=extensivity, physical_domain="test_domain",
            semantic_name=f"{name}_{kind}", orientation_convention="positive_into_system",
            boundary_role="port",
        )

    return PortSpecIPHS(
        name=name, effort_var=f"{name}_effort", flow_var=f"{name}_flow",
        effort_space=_space("effort", "dual", "J", "intensive"),
        flow_space=_space("flow", "primal", "s-1", "integral"),
        power_units="W", positive_power_convention="into_system", port_kind="energy",
        role="internal",
    )


def test_compile_graph_assembles_local_iphs_contributions_and_port_coupling():
    graph = _graph(
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "local_port_coupled_iphs",
                "state_vars": ("x", "z"),
                "energy_functional": "E",
                "entropy_functional": "S",
                "contributions": [
                    {"state_vars": ("x",), "G": torch.tensor([[1.0]]), "ports": (_internal_port("x_port"),)},
                    {"state_vars": ("z",), "G": torch.tensor([[1.0]]), "ports": (_internal_port("z_port"),)},
                ],
                "port_interconnection_C": torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
                "port_dissipation_D": torch.diag(torch.tensor([0.1, 0.2])),
            }
        }
    )

    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()))
    report = plan_to_dict(plan)["report"]

    assert report["audit"]["iphs_audit"]["enabled"] is True
    assert report["audit"]["iphs_audit"]["passed"] is True
    assert report["audit"]["iphs_audit"]["state_var_count"] == 2


def test_compile_graph_collects_node_iphs_contributions_from_config():
    storage_class = (
        "Functor:test.fixtures.dummy_functor."
        "DummyConfiguredOutputFunctor.iphs_storage"
    )
    boundary_class = (
        "Functor:test.fixtures.dummy_functor."
        "DummyConfiguredOutputFunctor.iphs_boundary"
    )
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "storage": NodeSpec(
                class_type=storage_class,
                outputs=["x"],
                config={
                    "output_name": "x",
                    "iphs_contribution": {
                        "state_vars": ("x",),
                        "M": torch.tensor([[0.1]]),
                    }
                },
            ),
            "boundary": NodeSpec(
                class_type=boundary_class,
                outputs=["z"],
                config={
                    "output_name": "z",
                    "iphs_contribution": {
                        "state_vars": ("z",),
                        "G": torch.tensor([[1.0]]),
                    }
                },
            ),
        },
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "node_declared_iphs",
                "state_vars": ("x", "z"),
                "energy_functional": "E",
                "entropy_functional": "S",
                "collect_node_contributions": True,
            }
        },
    )

    registry = NodeRegistry(_dimensionless_catalog("x", "z"))
    for class_type, output_name in (
        (storage_class, "x"),
        (boundary_class, "z"),
    ):
        registry.register_functor_instance(
            DummyConfiguredOutputFunctor(
                config={"output_name": output_name}
            ),
            node_type_id=class_type,
        )

    plan = compile_graph(graph, registry)
    report = plan_to_dict(plan)["report"]

    assert report["audit"]["iphs_audit"]["enabled"] is True
    assert report["audit"]["iphs_audit"]["passed"] is True
    assert report["audit"]["iphs_audit"]["state_var_count"] == 2
    assert report["audit"]["iphs_audit"]["collected_node_contribution_count"] == 2
    assert report["audit"]["iphs_audit"]["has_port_matrix"] is True


def test_compile_graph_rejects_invalid_declared_iphs_metadata_in_strict_mode():
    graph = _graph(
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "bad_compile_time_iphs",
                "state_vars": ("x", "s"),
                "energy_functional": "E",
                "entropy_functional": "S",
                "contributions": [
                    {"J": torch.ones(2, 2)},
                    {"M": torch.eye(2)},
                ],
            }
        }
    )

    with pytest.raises(ValueError, match="iphs_audit"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()))


def test_relic_model_compiler_facade_uses_existing_compile_graph():
    graph, registry = _executable_graph()
    compiler = RELICModelCompiler(registry)

    plan = compiler.compile(graph)

    assert len(plan.steps) == 1
    assert plan.steps[0].node_id == "n1"
    assert plan.report.metadata["audit_summary"]["global_compile_audit"]["passed"] is True


def test_relic_model_compiler_facade_reaches_canonical_validate_graph(
    monkeypatch,
):
    """Facade compilation must execute the existing GraphIR validator."""
    graph, registry = _executable_graph()
    calls = []
    canonical_validate_graph = compiler_module.validate_graph

    def tracked_validate_graph(*args, **kwargs):
        calls.append((args, kwargs))
        return canonical_validate_graph(*args, **kwargs)

    monkeypatch.setattr(
        compiler_module,
        "validate_graph",
        tracked_validate_graph,
    )

    plan = RELICModelCompiler(registry).compile(graph)

    assert plan.steps[0].node_id == "n1"
    assert calls
    first_args, first_kwargs = calls[0]
    assert not first_args
    assert first_kwargs["graph"] is graph
    assert first_kwargs["registry"] is registry


def test_compiler_structural_rules_are_global_validation_registry_owners():
    """Registry wrappers must delegate to the shared GraphIR rule helpers."""
    registry = get_global_registry()
    expected = {
        "hodge_passivity_structure": (
            validate_hodge_passivity_structure,
            validate_hodge_spec,
        ),
        "iphs_structure": (
            validate_iphs_structure,
            validate_iphs_spec,
        ),
        "reaction_network_structure": (
            validate_reaction_network_structure,
            validate_reaction_network_payload,
        ),
    }

    for rule_id, (registry_wrapper, shared_helper) in expected.items():
        assert registry.get_rule(rule_id) is registry_wrapper
        original_function = registry_wrapper._original_function
        assert shared_helper.__name__ in inspect.getsource(original_function)


def test_compile_graph_revalidates_after_optimization_passes():
    class InvalidAssertionPass:
        name = "invalid_assertion"

        def apply(self, graph, registry):
            _ = registry
            return GraphIR(
                schema_version=graph.schema_version,
                nodes=graph.nodes,
                links=graph.links,
                metadata=graph.metadata,
                assertion={"phi": "invalid_after_optimization", "scope": "graph"},
            )

    with pytest.raises(ValueError, match="Graph validation failed after optimization"):
        compile_graph(
            _graph(),
            NodeRegistry(SSOTCatalog()),
            enable_optimization=True,
            optimization_passes=[InvalidAssertionPass()],
        )


def test_compile_graph_rejects_empty_strict_iphs_assembly():
    graph = _graph(
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "empty_strict_iphs",
                "state_vars": ("x",),
                "energy_functional": "E",
                "entropy_functional": "S",
                "contributions": [],
                "strict": True,
                "required_level": 4,
            }
        }
    )

    with pytest.raises(ValueError, match="structural assertions"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()))


def test_compile_graph_rejects_mapping_only_strict_iphs_runtime_claim():
    graph = _graph(
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "runtime_bound_strict_iphs",
                "state_vars": ("x",),
                "energy_functional": "E",
                "entropy_functional": "S",
                "formulation": "dissipative_ph",
                "contributions": [
                    {
                        "runtime_owner": "test.compiler.runtime_bound",
                        "R": torch.ones((1, 1)),
                        "energy_gradient": torch.ones(1),
                    }
                ],
                "strict": True,
                "required_level": 4,
            }
        }
    )

    # A runtime_owner string, static gradient, and self-issued mapping cannot
    # prove that the lowered execution plan calls canonical_drift.
    with pytest.raises(ValueError, match="structural assertions"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()))


def test_iphs_audit_reports_runtime_owner_text_is_not_compiler_binding():
    graph = _graph(
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "mapping_only_runtime_claim",
                "state_vars": ("x",),
                "energy_functional": "E",
                "entropy_functional": "S",
                "formulation": "dissipative_ph",
                "contributions": [
                    {
                        "runtime_owner": "test.compiler.runtime_bound",
                        "R": torch.ones((1, 1)),
                        "energy_gradient": torch.ones(1),
                    }
                ],
                "strict": True,
                "required_level": 4,
            }
        }
    )

    audit = _audit_rejected_assembly(graph, NodeRegistry(SSOTCatalog()))

    assert audit["passed"] is False
    assert audit["compile_identity_binding"]["compile_identity_bound"] is False
    assert audit["compile_identity_binding"]["binding_count"] == 0
    assert audit["compile_identity_binding"]["reason"] == "source_node_id_required"


def test_iphs_audit_binds_registry_resolved_lowered_execution_step_only():
    graph, registry = _executable_iphs_graph()
    graph.nodes["n1"].config["iphs_contribution"] = {
        "R": torch.ones((1, 1)),
        "energy_gradient": torch.ones(1),
    }
    graph.metadata["iphs_assembly"] = {
        "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
        "name": "registry_and_plan_bound",
        "state_vars": ("x",),
        "energy_functional": "E",
        "entropy_functional": "S",
        "formulation": "dissipative_ph",
        "contribution_node_ids": ["n1"],
        "strict": True,
        "required_level": 4,
    }

    audit = _audit_rejected_assembly(graph, registry)
    binding = audit["compile_identity_binding"]

    assert audit["passed"] is False
    assert "explicit runtime energy_effort" in audit["message"]
    assert binding["compile_identity_bound"] is True
    assert binding["process_runtime_replacement_verified"] is False
    assert binding["binding_count"] == 1
    assert binding["source_node_ids"] == ["n1"]
    assert binding["bindings"][0]["node_type_id"] == graph.nodes["n1"].class_type
    assert binding["bindings"][0]["runtime_class"].endswith(".DummyIPHSFunctor")
    assert binding["bindings"][0]["runtime_owner"].endswith(
        ".DummyIPHSFunctor.forward"
    )
    assert (
        binding["bindings"][0]["binding_kind"]
        == "compile_local_execution_step_identity"
    )
    assert len(binding["bindings"][0]["node_definition_sha256"]) == 64
    assert len(binding["bindings"][0]["execution_step_sha256"]) == 64


def test_iphs_audit_rejects_contribution_node_lowered_out_of_execution_plan():
    graph = _graph(
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "lowered_boundary_is_not_runtime",
                "state_vars": ("x",),
                "energy_functional": "E",
                "entropy_functional": "S",
                "formulation": "dissipative_ph",
                "contributions": [
                    {
                        "source_node_id": "n0",
                        "R": torch.ones((1, 1)),
                        "energy_gradient": torch.ones(1),
                    }
                ],
                "strict": True,
                "required_level": 4,
            }
        }
    )

    audit = _audit_rejected_assembly(graph, NodeRegistry(SSOTCatalog()))

    assert audit["passed"] is False
    assert audit["compile_identity_binding"]["compile_identity_bound"] is False
    assert (
        audit["compile_identity_binding"]["reason"]
        == "inline_metadata_not_compiler_bound"
    )
    assert audit["compile_identity_binding"]["source_node_ids"] == ["n0"]


def test_iphs_audit_rejects_mapping_forged_runtime_binding_flag():
    graph, registry = _executable_iphs_graph()
    graph.nodes["n1"].config["iphs_contribution"] = {
        "R": torch.ones((1, 1)),
        "energy_gradient": torch.ones(1),
        "runtime_bound": True,
    }
    graph.metadata["iphs_assembly"] = {
        "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
        "name": "forged_runtime_binding",
        "state_vars": ("x",),
        "energy_functional": "E",
        "entropy_functional": "S",
        "formulation": "dissipative_ph",
        "contribution_node_ids": ["n1"],
        "strict": True,
        "required_level": 4,
    }

    audit = _audit_rejected_assembly(graph, registry)

    assert audit["passed"] is False
    assert audit["compile_identity_binding"]["compile_identity_bound"] is False
    assert audit["compile_identity_binding"]["reason"] == "forged_evidence_field"


def test_iphs_audit_rejects_executable_non_iphs_functor_identity():
    graph, registry = _executable_graph()
    graph.nodes["n1"].config["iphs_contribution"] = {
        "R": torch.ones((1, 1)),
        "energy_gradient": torch.ones(1),
    }
    graph.metadata["iphs_assembly"] = {
        "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
        "name": "non_iphs_runtime_class",
        "state_vars": ("x",),
        "energy_functional": "E",
        "entropy_functional": "S",
        "formulation": "dissipative_ph",
        "contribution_node_ids": ["n1"],
        "strict": True,
        "required_level": 4,
    }

    audit = _audit_rejected_assembly(graph, registry)

    assert audit["passed"] is False
    assert audit["compile_identity_binding"]["compile_identity_bound"] is False
    assert (
        audit["compile_identity_binding"]["reason"]
        == "runtime_class_not_iphs_functor"
    )


def test_iphs_assertion_cannot_overwrite_failed_assembly_audit():
    graph = _graph(
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "invalid_empty_strict_assembly",
                "state_vars": ("x",),
                "energy_functional": "E",
                "entropy_functional": "S",
                "formulation": "dissipative_ph",
                "contributions": [],
                "strict": True,
                "required_level": 4,
            }
        }
    )
    validation = ValidationReport(
        assertion_audit={
            "audit_key": "iphs_audit",
            "structural_result": {
                "enabled": True,
                "passed": True,
                "source": "unrelated_assertion",
            },
        }
    )

    audit = build_global_structural_audit(graph, validation)

    assert audit["iphs_audit"]["passed"] is False
    assert audit["iphs_audit"]["assembly_result"]["passed"] is False
    assert audit["iphs_audit"]["assertion_result"]["passed"] is True
    assert audit["global_compile_audit"]["passed"] is False


def test_compile_graph_rejects_text_only_constant_operator_level5_generic_assembly():
    # References: Edwards and Oettinger (1997), Phys. Rev. E 56, 4097.
    # Mathematical expression: Jacobi(L)=0 must hold over the declared state domain.
    # Parameter meanings: L is J below; the constant string is descriptive metadata only.
    graph = _graph(
        metadata={
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "constant_operator_generic",
                "state_vars": ("x", "s"),
                "energy_functional": "E",
                "entropy_functional": "S",
                "formulation": "generic",
                "reversible_operator_state_dependence": "constant",
                "contributions": [
                    {
                        "runtime_owner": "test.compiler.constant_generic",
                        "J": torch.zeros((2, 2)),
                        "M": torch.diag(torch.tensor([0.0, 1.0])),
                        "energy_gradient": torch.tensor([1.0, 0.0]),
                        "entropy_gradient": torch.tensor([0.0, 1.0]),
                    }
                ],
                "strict": True,
                "required_level": 5,
            }
        }
    )

    with pytest.raises(ValueError, match="structural assertions"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()))


def test_compile_graph_cannot_lower_graph_level5_with_payload_level4():
    graph = _graph(
        metadata={
            "iphs_level": 5,
            "iphs_assembly": {
                "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
                "name": "attempted_level_downgrade",
                "state_vars": ("x",),
                "energy_functional": "E",
                "entropy_functional": "S",
                "formulation": "dissipative_ph",
                "contributions": [
                    {
                        "runtime_owner": "test.compiler.attempted_downgrade",
                        "R": torch.ones((1, 1)),
                        "energy_gradient": torch.ones(1),
                    }
                ],
                "strict": True,
                "required_level": 4,
            },
        }
    )

    with pytest.raises(ValueError, match="structural assertions"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()))


def _ordinary_assembly_metadata(level=None, *, required_level=None):
    assembly = {
        "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
        "name": "declared_local_structure",
        "state_vars": ("x",),
        "contributions": [{"M": torch.ones((1, 1))}],
    }
    if level is not None:
        assembly["iphs_level"] = level
    if required_level is not None:
        assembly["required_level"] = required_level
    return {"strict_structural_assertions": True, "iphs_assembly": assembly}


@pytest.mark.parametrize("level", [None, 0, 1, 2, 3])
def test_ordinary_strict_structure_assembly_does_not_request_runtime_iphs(level):
    graph = _graph(metadata=_ordinary_assembly_metadata(level))

    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()), strict=True)

    audit = plan.report.metadata["audit_summary"]["iphs_audit"]
    assert audit["passed"] is True
    assert audit["required_level"] is None
    assert audit["compile_identity_binding"]["compile_identity_bound"] is False


def test_zero_assembly_runtime_floor_is_not_a_strict_iphs_request():
    graph = _graph(metadata=_ordinary_assembly_metadata(required_level=0))

    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()), strict=True)

    assert plan.report.metadata["audit_summary"]["iphs_audit"]["passed"] is True


@pytest.mark.parametrize("level", [1, 2, 3])
def test_positive_assembly_runtime_floor_requires_certificate_evidence(level):
    graph = _graph(metadata=_ordinary_assembly_metadata(required_level=level))

    with pytest.raises(ValueError, match="structural assertions"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()), strict=True)


def test_compiler_consumes_validator_level_resolver(monkeypatch):
    from relic.orchestration import validator as validator_module

    assert compiler_module._required_iphs_level is validator_module._required_iphs_level
    calls = []
    resolver = validator_module._required_iphs_level

    def tracked_resolver(*args, **kwargs):
        calls.append((args, kwargs))
        return resolver(*args, **kwargs)

    monkeypatch.setattr(compiler_module, "_required_iphs_level", tracked_resolver)
    graph = _graph(metadata=_ordinary_assembly_metadata(2))
    compile_graph(graph, NodeRegistry(SSOTCatalog()))

    assert calls
    assert calls[0][0][0]["iphs_level"] == 2


def test_ordinary_strict_graph_executes_existing_process_without_iphs_objects():
    from relic.orchestration.executor import GraphExecutor

    graph, registry = _executable_graph({"strict_structural_assertions": True})
    plan = compile_graph(graph, registry, strict=True)
    executor = GraphExecutor(plan, registry=registry)

    state = executor.execute({"x": torch.tensor(0.0)}, n_steps=2, dt=1.0)

    assert state["x"].item() == 2.0
    assert any(
        item["functor"] == "n1"
        for item in executor._simulator.get_variable_provenance("x")
    )


def _strict_binding_audit(monkeypatch, resolver_error):
    """Build the structural audit of a strict IPHS claim whose runtime class resolution fails."""
    graph, registry = _executable_iphs_graph()
    graph.nodes["n1"].config["iphs_contribution"] = {
        "R": torch.ones((1, 1)),
        "energy_gradient": torch.ones(1),
    }
    graph.metadata["iphs_assembly"] = {
        "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
        "name": "runtime_class_resolution",
        "state_vars": ("x",),
        "energy_functional": "E",
        "entropy_functional": "S",
        "formulation": "dissipative_ph",
        "contribution_node_ids": ["n1"],
        "strict": True,
        "required_level": 4,
    }
    ordinary_graph = GraphIR(
        schema_version=graph.schema_version,
        nodes=graph.nodes,
        links=graph.links,
        metadata={},
    )
    ordinary_plan = compile_graph(ordinary_graph, registry, strict=False)
    validation = compiler_module.validate_graph(graph, registry)

    def _failing_resolver(class_type):
        raise resolver_error

    monkeypatch.setattr(registry, "resolve_class_type", _failing_resolver)
    return build_global_structural_audit(
        graph, validation, metadata=ordinary_plan.report.metadata,
        registry=registry, execution_steps=ordinary_plan.steps,
    )


def test_iphs_binding_failure_keeps_the_runtime_class_resolution_error(monkeypatch):
    audit = _strict_binding_audit(
        monkeypatch, ImportError("No module named 'relic_test_removed_iphs_runtime'")
    )
    iphs_audit = audit["iphs_audit"]
    binding = iphs_audit["compile_identity_binding"]

    assert iphs_audit["passed"] is False
    assert binding["reason"] == "runtime_class_unresolved"
    assert "relic_test_removed_iphs_runtime" in binding["detail"]
    assert "relic_test_removed_iphs_runtime" in iphs_audit["message"]


def test_iphs_binding_propagates_resolver_defects(monkeypatch):
    with pytest.raises(AttributeError, match="resolver defect"):
        _strict_binding_audit(monkeypatch, AttributeError("resolver defect"))


def test_iphs_assembly_audit_propagates_operator_import_failure(monkeypatch):
    """The IPHS operator is RELIC's own module; its import failure is not an audit result."""
    monkeypatch.setitem(sys.modules, "relic.computation.operators.iphs", None)

    with pytest.raises(ImportError, match="relic.computation.operators.iphs"):
        compiler_module._run_iphs_assembly_audit({"name": "import_probe"}, _graph())


@pytest.mark.parametrize("missing", ["poisson_operator", "reaction_representation"])
def test_iphs_assembly_without_a_required_policy_fails_naming_declaration_and_field(missing):
    """The compiler reads both policies directly; nothing is filled in for a missing key."""
    assembly = {
        "name": "soil_column_assembly",
        "state_vars": ["x"],
        "poisson_operator": "matrix_snapshot",
        "reaction_representation": "independent",
        "contributions": [{"M": [[1.0]]}],
    }
    assembly.pop(missing)
    graph = _graph(metadata={"iphs_assembly": assembly})

    audit = compiler_module._run_iphs_assembly_audit(assembly, graph)

    assert audit["passed"] is False
    assert f"'{missing}'" in audit["message"]
    assert "graph.metadata.iphs_assembly" in audit["message"]
    assert "soil_column_assembly" in audit["message"]
    with pytest.raises(ValueError, match=missing):
        compile_graph(graph, NodeRegistry(SSOTCatalog()))


def test_poisson_field_keys_without_poisson_operator_name_the_missing_field():
    with pytest.raises(ValueError, match="'poisson_operator'"):
        compiler_module._resolve_iphs_poisson_field_arguments(
            {"name": "field_assembly", "jacobi_tol": 1.0e-8}
        )
