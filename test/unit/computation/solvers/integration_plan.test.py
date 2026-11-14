"""Tests for structure-aware IntegrationPlan validation."""

from relic.computation.solvers import (
    IntegrationPlan,
    IntegrationStage,
    build_stochastic_sdae_plan,
    build_strang_iphs_plan,
)


def test_integration_plan_accepts_direct_acyclic_dependencies():
    plan = IntegrationPlan(
        plan_id="iphs_transport_reaction",
        scheme="lie",
        stages=(
            IntegrationStage(stage_id="transport", operator_kind="transport", solver_name="implicit_midpoint", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct"),
            IntegrationStage(
                stage_id="reaction",
                operator_kind="reaction",
                depends_on=("transport",), solver_name="implicit_midpoint", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct",
            ),
        ),
    )

    audit = plan.validate()

    assert audit["passed"] is True
    assert audit["topological_order"] == ["transport", "reaction"]


def test_integration_plan_allows_lagged_cycle_when_declared():
    plan = IntegrationPlan(
        plan_id="lagged_exchange",
        stages=(
            IntegrationStage(
                stage_id="soil",
                operator_kind="iphs_soil",
                depends_on=("atmosphere",),
                lagged_dependencies=("atmosphere",), solver_name="implicit_midpoint", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct",
            ),
            IntegrationStage(
                stage_id="atmosphere",
                operator_kind="iphs_atmosphere",
                depends_on=("soil",),
                lagged_dependencies=("soil",), solver_name="implicit_midpoint", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct",
            ),
        ),
    )

    audit = plan.validate()

    assert audit["passed"] is True
    assert audit["lagged_edge_count"] == 2


def test_integration_plan_rejects_direct_cycle():
    plan = IntegrationPlan(
        plan_id="bad_cycle",
        stages=(
            IntegrationStage(stage_id="a", operator_kind="iphs_a", depends_on=("b",), solver_name="implicit_midpoint", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct"),
            IntegrationStage(stage_id="b", operator_kind="iphs_b", depends_on=("a",), solver_name="implicit_midpoint", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct"),
        ),
        allow_lagged_cycles=False,
    )

    audit = plan.validate()

    assert audit["passed"] is False
    assert any("cycle" in issue for issue in audit["issues"])


def test_build_strang_iphs_plan_has_symmetric_endpoints():
    plan = build_strang_iphs_plan(
        plan_id="damped_mode",
        reversible_operator="reversible_J",
        dissipative_operator="dissipative_M", solver_name="implicit_midpoint",
    )

    audit = plan.validate()

    assert audit["passed"] is True
    assert plan.stages[0].operator_kind == plan.stages[-1].operator_kind
    assert plan.stages[0].dt_fraction == plan.stages[-1].dt_fraction


def test_integration_plan_accepts_relic_stochastic_stage_owner():
    plan = IntegrationPlan(
        plan_id="stochastic_flux",
        stages=(
            IntegrationStage(
                stage_id="noise",
                operator_kind="conservative_noise_flux",
                solver_name="euler_maruyama",
                runtime_owner="relic.stochastic", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct",
            ),
        ),
    )

    audit = plan.validate()

    assert audit["passed"] is True


def test_integration_plan_rejects_unimplemented_stochastic_solver_names():
    for solver_name in ("milstein", "predictor_corrector"):
        audit = IntegrationPlan(
            plan_id=f"unsupported_{solver_name}",
            stages=(
                IntegrationStage(
                    stage_id="noise",
                    operator_kind="conservative_noise_flux",
                    solver_name=solver_name,
                    runtime_owner="relic.stochastic", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct",
                ),
            ),
        ).validate()

        assert audit["passed"] is False
        assert any("invalid relic.stochastic" in issue for issue in audit["issues"])


def test_build_stochastic_sdae_plan_records_required_stage_metadata():
    plan = build_stochastic_sdae_plan(
        plan_id="sdae_transport",
        include_constraint_projection=True,
        include_jump_reaction=True,
        include_stabilized_advection=True,
    )

    audit = plan.validate()

    assert audit["passed"] is True
    assert any(stage.operator_kind == "conservative_noise_flux" for stage in plan.stages)
    assert any(stage.operator_kind == "constraint_projection" for stage in plan.stages)


def _nonlinear_closure_stage(**overrides):
    values = {
        "stage_id": "farquhar_ci_closure",
        "operator_kind": "nonlinear_algebraic_closure",
        "solver_name": "bracketed_bisection",
        "structure_kind": "algebraic_constraint",
        "dt_fraction": 0.0,
        "advances_time": False,
        "coupling_mode": "direct",
        "runtime_owner": "solver_registry",
        "reads": ("atmospheric_co2_ppm", "ppfd", "vapor_pressure_deficit"),
        "writes": ("intercellular_co2",),
        "validation_hooks": ("state_domain_check",),
        "stage_metadata": {
            "runtime_node_id": "farquhar_photosynthesis",
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
        },
    }
    values.update(overrides)
    return IntegrationStage(**values)


def test_integration_plan_accepts_nonlinear_algebraic_closure():
    plan = IntegrationPlan(
        plan_id="farquhar_atomic_closure",
        scheme="single",
        stages=(_nonlinear_closure_stage(),),
    )

    audit = plan.validate(runtime_node_ids={"farquhar_photosynthesis"})

    assert audit["passed"] is True
    assert audit["structurally_validated"] is True
    assert audit["verified"] is False
    assert audit["runtime_ready"] is False
    assert audit["execution_semantics"] == "metadata_only"
    assert audit["solver_execution_count"] == 0


def test_nonlinear_closure_rejects_wrong_solver_family_and_owner():
    wrong_family = IntegrationPlan(
        plan_id="wrong_family",
        stages=(_nonlinear_closure_stage(solver_name="implicit_midpoint"),),
    ).validate(runtime_node_ids={"farquhar_photosynthesis"})
    wrong_owner = IntegrationPlan(
        plan_id="wrong_owner",
        stages=(_nonlinear_closure_stage(runtime_owner="simulator"),),
    ).validate(runtime_node_ids={"farquhar_photosynthesis"})

    assert wrong_family["passed"] is False
    assert any("NONLINEAR" in issue for issue in wrong_family["issues"])
    assert wrong_owner["passed"] is False
    assert any("runtime_owner" in issue for issue in wrong_owner["issues"])


def test_nonlinear_closure_rejects_time_advance_and_incomplete_metadata():
    time_advance = IntegrationPlan(
        plan_id="time_advance",
        stages=(
            _nonlinear_closure_stage(advances_time=True, dt_fraction=1.0),
        ),
    ).validate(runtime_node_ids={"farquhar_photosynthesis"})
    missing_metadata = IntegrationPlan(
        plan_id="missing_metadata",
        stages=(_nonlinear_closure_stage(stage_metadata={}),),
    ).validate(runtime_node_ids={"farquhar_photosynthesis"})

    assert time_advance["passed"] is False
    assert any("advance time" in issue for issue in time_advance["issues"])
    assert missing_metadata["passed"] is False
    assert any("missing metadata" in issue for issue in missing_metadata["issues"])


def test_nonlinear_closure_rejects_unknown_write_and_runtime_node_mismatch():
    plan = IntegrationPlan(
        plan_id="bad_bindings",
        stages=(_nonlinear_closure_stage(writes=("gross_photosynthesis",)),),
    )

    audit = plan.validate(runtime_node_ids={"other_node"})

    assert audit["passed"] is False
    assert any("unknown_variables" in issue for issue in audit["issues"])
    assert any("runtime node" in issue for issue in audit["issues"])


def test_integration_plan_mapping_roundtrip_revalidates_and_ignores_forged_audit():
    plan = IntegrationPlan(
        plan_id="farquhar_roundtrip",
        scheme="single",
        stages=(_nonlinear_closure_stage(),),
    )
    payload = plan.to_dict()
    assert "audit" not in payload
    rebuilt = IntegrationPlan.from_dict(payload)
    assert rebuilt.to_dict() == payload
    assert rebuilt.validate(
        runtime_node_ids={"farquhar_photosynthesis"}
    )["passed"] is True

    forged = dict(payload)
    forged["audit"] = {"passed": True}
    forged["stages"] = [dict(payload["stages"][0])]
    forged["stages"][0]["solver_name"] = "implicit_midpoint"
    audit = IntegrationPlan.from_dict(forged).validate(
        runtime_node_ids={"farquhar_photosynthesis"}
    )
    assert audit["passed"] is False


def test_integration_plan_mapping_rejects_unknown_schema_fields():
    payload = IntegrationPlan(
        plan_id="strict_schema",
        stages=(_nonlinear_closure_stage(),),
    ).to_dict()
    payload["second_runtime"] = True

    try:
        IntegrationPlan.from_dict(payload)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("Unknown IntegrationPlan fields must fail fast")


# ---------------------------------------------------------------------------
# Round 5: solver, structure kind, time fraction and coupling mode of a stage
# are declared (no implicit_midpoint / iphs / 1.0 / direct choice is made).
# ---------------------------------------------------------------------------

_DECLARED_STAGE = {
    "stage_id": "reaction",
    "operator_kind": "reaction",
    "solver_name": "implicit_midpoint",
    "structure_kind": "iphs",
    "dt_fraction": 1.0,
    "coupling_mode": "direct",
}


def test_integration_stage_declares_solver_structure_fraction_and_coupling():
    import pytest

    for missing in ("solver_name", "structure_kind", "dt_fraction", "coupling_mode"):
        values = dict(_DECLARED_STAGE)
        values.pop(missing)
        with pytest.raises(TypeError, match=missing):
            IntegrationStage(**values)
        with pytest.raises(ValueError, match=missing):
            IntegrationStage.from_dict(values)
    assert IntegrationStage.from_dict(_DECLARED_STAGE) == IntegrationStage(**_DECLARED_STAGE)


def test_strang_plan_builder_needs_the_stage_solver():
    import pytest

    with pytest.raises(TypeError, match="solver_name"):
        build_strang_iphs_plan(
            plan_id="undeclared_solver",
            reversible_operator="iphs_reversible",
            dissipative_operator="iphs_dissipative",
        )
