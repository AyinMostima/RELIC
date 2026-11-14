"""CR-A-04 T10: the IntegrationPlan of the executed composition and its runtime witness.

Design 20261019_cr_a04_design_01, sections 3 ("Executor") and 4 (T10), step 8:

- RELICSimulator.get_composition_plan() is a pure export of the committed provider
  view onto IntegrationPlan stages (computation/solvers/geometric.py): the fast
  group in its published order (Lie: one stage per member, all in the one group
  call, role lie, commit stage 0; Strang with n >= 2 members: the 2n - 1 stages of
  the palindrome with roles first_half / middle / second_half and commit stages
  0..2n-2; one member: a single Lie call), then the medium and the slow group (one
  stage per member, role group, over the elapsed group interval). Sub-steps come
  from the binding payload (substep_overrides) or FunctorMetadata.substep_count;
  multirate exchanges are recorded on the consumer stages that read their variable.
  Stage-view restatements, composition-mean finalizations and exchange commits are
  commits, not stages. Every stage has runtime_owner simulator.
- The stage metadata predicts the commit context of every executed group commit
  (role, commit stage, interval start and interval), which the writer records
  carry since step 5 (get_variable_provenance(include_commit_context=True)).
- GraphExecutor compares a declared graph.metadata.integration_plan whose stages
  are owned by the simulator with the simulator plan of the prepared (or restored)
  run: a mismatch is refused before any step (every executable plan is a strict
  compile); a match is recorded as the witness, and after the steps the executed
  commit contexts are checked against the declared stages; a plan without
  simulator stages is not compared; mixed owners are refused.
- Restart decision: the plan is a pure function of state the exact-restart
  contract already binds (topology incl. groups, execution order and simulator
  configuration; the splitting controls as runtime fields; sub-step overrides in
  the binding digests), so no checkpoint field is added (format_version 3); an
  exact restore reproduces the plan, restores that would change it are refused,
  and the executor re-verifies the declared plan after the restore.

References: Strang (1968) SIAM J. Numer. Anal. 5(3):506-517; Hairer, Lubich and
Wanner (2006) Geometric Numerical Integration, 2nd ed., Sect. II.5; Gear and Wells
(1984) BIT 24(4):484-502 (multirate); W3C PROV-DM (2013) (generation records).
Mathematical expression: Lie Phi_h = Phi_n^h o ... o Phi_1^h; Strang
Phi_h = Phi_1^{h/2} o ... o Phi_{n-1}^{h/2} o Phi_n^h o Phi_{n-1}^{h/2} o ... o Phi_1^{h/2};
a stage with dt fraction a and start fraction b covers [t + b H, t + b H + a H]
of its composition [t, t + H].
Parameter meanings: h the fast step; H the composition interval (h for the fast
group, the elapsed interval for medium and slow); n the fast members.
Units: t, h, H [s]; fractions dimensionless; fixtures dimensionless.
Domain/constraints: CPU, float64 test default dtype (test/conftest.py).
Sign convention: none (no fluxes in the fixtures).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pytest
import torch

from relic.computation.solvers.geometric import IntegrationPlan
from relic.core.simulator import RELICSimulator
from relic.core.simulator import checkpoint as checkpoint_module
from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from test.fixtures.dummy_functor import DummyConfiguredOutputFunctor

_TEST_ROOT = Path(__file__).resolve().parents[1]


def _load_test_module(relative: str, alias: str):
    """Load an existing test module by path to reuse its fixtures (no copies)."""
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, _TEST_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_T1 = _load_test_module("core/splitting_admissibility.test.py", "_cr_a04_t1_admissibility")
_T7 = _load_test_module("core/splitting_multirate_exchanges.test.py", "_cr_a04_t7_multirate_exchanges")
_GE = _load_test_module("orchestration/graph_executor.test.py", "_cr_a04_t10_graph_executor")

_FAST_ROLES = ("lie", "first_half", "middle", "second_half")
_NOT_STAGES = ("stage_view", "composition_mean", "exchange")


def _stages(plan: IntegrationPlan) -> List[Tuple[str, str, str, int, float, float, str]]:
    """(stage_id, functor, role, commit stage, dt fraction, start fraction, coupling) per stage."""
    return [
        (
            stage.stage_id,
            stage.operator_kind,
            stage.stage_metadata["role"],
            stage.stage_metadata["commit_stage"],
            stage.dt_fraction,
            stage.stage_metadata["start_fraction"],
            stage.coupling_mode,
        )
        for stage in plan.stages
    ]


def _relaxations(names: Sequence[str], **config) -> RELICSimulator:
    simulator = _T1._fixture_simulator(**config)
    for name in names:
        simulator.register_functor(_T1._Relaxation(name, name))
    return simulator


# --------------------------------------------------------------------------
# get_composition_plan: the export (main.py).
# --------------------------------------------------------------------------


def test_lie_plan_has_one_stage_per_fast_member_in_the_published_order():
    simulator = _relaxations(("relax_a", "relax_b"))
    plan = simulator.get_composition_plan()
    assert isinstance(plan, IntegrationPlan)
    assert plan.scheme == "lie"
    assert _stages(plan) == [
        ("fast.0.relax_a", "relax_a", "lie", 0, 1.0, 0.0, "lie"),
        ("fast.1.relax_b", "relax_b", "lie", 0, 1.0, 0.0, "lie"),
    ]
    first, second = plan.stages
    assert first.runtime_owner == second.runtime_owner == "simulator"
    assert first.depends_on == () and second.depends_on == ("fast.0.relax_a",)
    assert first.reads == ("relax_a",) and first.writes == ("relax_a",)
    assert first.stage_metadata == {
        "execution_group": "fast",
        "role": "lie",
        "commit_stage": 0,
        "start_fraction": 0.0,
        "interval_basis": "fast_step",
        "group_trigger": {"kind": "every_step"},
        "substeps": {"source": "functor_metadata", "count": 1},
        "sub_flow_accuracy": "undeclared",
        "exchanges": [],
        "splitting_mode": "lie",
        "fast_order_mode": "forward",
    }


def test_strang_plan_is_the_palindrome_of_2n_minus_1_stages():
    simulator = _relaxations(("relax_a", "relax_b", "relax_c"), splitting_mode="strang")
    plan = simulator.get_composition_plan()
    assert plan.scheme == "strang"
    assert _stages(plan) == [
        ("fast.0.relax_a", "relax_a", "first_half", 0, 0.5, 0.0, "strang_half"),
        ("fast.1.relax_b", "relax_b", "first_half", 1, 0.5, 0.0, "strang_half"),
        ("fast.2.relax_c", "relax_c", "middle", 2, 1.0, 0.0, "strang_full"),
        ("fast.3.relax_b", "relax_b", "second_half", 3, 0.5, 0.5, "strang_half"),
        ("fast.4.relax_a", "relax_a", "second_half", 4, 0.5, 0.5, "strang_half"),
    ]
    assert all(stage.stage_metadata["splitting_mode"] == "strang" for stage in plan.stages)
    audit = plan.validate()
    assert audit["passed"] is True and audit["verified"] is False
    assert audit["topological_order"] == [stage.stage_id for stage in plan.stages]


def test_strang_with_one_fast_member_is_a_single_lie_call():
    simulator = _relaxations(("relax_a",), splitting_mode="strang")
    plan = simulator.get_composition_plan()
    assert plan.scheme == "single"
    assert _stages(plan) == [("fast.0.relax_a", "relax_a", "lie", 0, 1.0, 0.0, "lie")]
    assert plan.stages[0].stage_metadata["splitting_mode"] == "strang"


def test_reverse_fast_order_plan_follows_the_published_order():
    simulator = _relaxations(("relax_a", "relax_b", "relax_c"), splitting_mode="strang", fast_order="reverse")
    plan = simulator.get_composition_plan()
    assert [stage.operator_kind for stage in plan.stages] == ["relax_c", "relax_b", "relax_a", "relax_b", "relax_c"]
    assert tuple(stage.operator_kind for stage in plan.stages[:3]) == simulator.get_committed_provider_view().order_fast
    assert {stage.stage_metadata["fast_order_mode"] for stage in plan.stages} == {"reverse"}


def test_substeps_come_from_the_binding_payload_or_the_functor_metadata():
    simulator = _T1._fixture_simulator(substep_overrides={"relax_b": {"max_seconds": 25.0}, "relax_c": {"count": 3}})
    relax_a = _T1._Relaxation("relax_a", "relax_a")
    relax_a.metadata.substep_count = 4
    simulator.register_functor(relax_a)
    simulator.register_functor(_T1._Relaxation("relax_b", "relax_b"))
    simulator.register_functor(_T1._Relaxation("relax_c", "relax_c"))
    substeps = {stage.operator_kind: stage.stage_metadata["substeps"] for stage in simulator.get_composition_plan().stages}
    assert substeps == {
        "relax_a": {"source": "functor_metadata", "count": 4},
        "relax_b": {"source": "substep_override", "max_seconds": 25.0},
        "relax_c": {"source": "substep_override", "count": 3},
    }


@pytest.mark.parametrize("mode, scheme", [("lie", "lie"), ("strang", "custom")])
def test_medium_and_slow_groups_follow_the_fast_composition(mode, scheme):
    simulator = _T7._ramp_simulator([_T7._MEDIUM_MEAN], splitting_mode=mode)
    plan = simulator.get_composition_plan()
    assert plan.scheme == scheme
    slower = [stage for stage in plan.stages if stage.stage_metadata["execution_group"] != "fast"]
    assert [(stage.stage_id, stage.stage_metadata["role"], stage.dt_fraction, stage.coupling_mode) for stage in slower] == [
        ("medium.0.medium_reader", "group", 1.0, "lie"),
        ("medium.1.medium_source", "group", 1.0, "lie"),
        ("slow.0.slow_reader", "group", 1.0, "lie"),
    ]
    reader, source, slow = slower
    assert reader.stage_metadata["group_trigger"] == {"kind": "interval", "seconds": _T7._MEDIUM}
    assert slow.stage_metadata["group_trigger"] == {"kind": "interval", "seconds": _T7._SLOW}
    assert {stage.stage_metadata["interval_basis"] for stage in slower} == {"group_elapsed_interval"}
    assert {stage.stage_metadata["commit_stage"] for stage in slower} == {0}
    assert "splitting_mode" not in reader.stage_metadata
    # The exchange is an overlay of the consumer stage that reads its variable.
    assert reader.stage_metadata["exchanges"] == [
        {"id": "ramp_medium", "variable": "ramp", "mode": "interval_mean", "mean_variable": "exchange_mean__ramp_medium"}
    ]
    assert source.stage_metadata["exchanges"] == [] and slow.stage_metadata["exchanges"] == []
    # The medium group follows the last fast stage.
    fast_last = [stage for stage in plan.stages if stage.stage_metadata["execution_group"] == "fast"][-1]
    assert reader.depends_on == (fast_last.stage_id,)
    assert plan.validate()["passed"] is True


def test_calendar_slow_group_and_hold_exchange_are_recorded():
    hold = {"id": "relax_hold", "variable": "relax_a", "consumer_group": "slow", "mode": "hold"}
    simulator = _T1._fixture_simulator(multirate_exchanges=[hold])
    simulator.register_functor(_T1._Relaxation("relax_a", "relax_a"))
    reader = _T1._Relaxation("relax_b", "relax_b")
    reader.metadata.input_variables.append("relax_a")
    reader.execution_group = "slow"
    simulator.register_functor(reader, execution_group="slow")
    stage = simulator.get_composition_plan().stages[-1]
    assert stage.stage_id == "slow.0.relax_b"
    assert stage.stage_metadata["group_trigger"] == {"kind": "calendar_month", "calendar": simulator._calendar_type}
    assert stage.stage_metadata["exchanges"] == [
        {"id": "relax_hold", "variable": "relax_a", "mode": "hold", "mean_variable": None}
    ]


def test_commits_that_are_not_stages_add_no_stage():
    with_exchange = _T7._ramp_simulator([_T7._MEDIUM_MEAN], splitting_mode="strang")
    without = _T7._ramp_simulator(splitting_mode="strang")
    assert len(with_exchange.get_composition_plan().stages) == len(without.get_composition_plan().stages) == 6
    roles = {stage.stage_metadata["role"] for stage in with_exchange.get_composition_plan().stages}
    assert roles <= set(_FAST_ROLES) | {"group"}
    assert not roles & set(_NOT_STAGES)


def test_the_export_is_pure_and_returns_fresh_objects():
    simulator = _relaxations(("relax_a", "relax_b"), splitting_mode="strang")
    epoch = simulator.provider_epoch
    view = simulator.get_committed_provider_view()
    state = {key: value.clone() for key, value in simulator.state_manager.get_state().items()}
    first = simulator.get_composition_plan()
    first.stages[0].stage_metadata["exchanges"].append({"id": "forged"})
    first.stages[0].stage_metadata["substeps"]["count"] = 99
    second = simulator.get_composition_plan()
    assert second.stages[0].stage_metadata["exchanges"] == []
    assert second.stages[0].stage_metadata["substeps"] == {"source": "functor_metadata", "count": 1}
    assert second.to_dict() == simulator.get_composition_plan().to_dict()
    assert simulator.provider_epoch == epoch and simulator.get_committed_provider_view() is view
    for key, value in simulator.state_manager.get_state().items():
        assert torch.equal(value, state[key]), key


def test_the_plan_round_trips_and_is_json_safe():
    import json

    plan = _T7._ramp_simulator([_T7._MEDIUM_MEAN], splitting_mode="strang").get_composition_plan()
    payload = plan.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert IntegrationPlan.from_dict(payload).to_dict() == payload
    assert {stage["runtime_owner"] for stage in payload["stages"]} == {"simulator"}


def test_an_empty_registry_exports_a_plan_without_stages():
    plan = RELICSimulator({"device": "cpu"}).get_composition_plan()
    assert plan.stages == () and plan.scheme == "custom"


def _writer_calls(simulator: RELICSimulator) -> set:
    calls = set()
    for records in simulator.get_variable_provenance(include_commit_context=True).values():
        for record in records:
            context = record.get("commit_context")
            if record["role"] == "writer" and context is not None:
                calls.add((record["functor"], tuple(sorted(context.items()))))
    return calls


@pytest.mark.parametrize("mode", ["lie", "strang"])
def test_the_plan_predicts_the_commit_context_of_every_executed_call(mode):
    simulator = _T7._ramp_simulator([_T7._MEDIUM_MEAN], splitting_mode=mode)
    # Eight fast steps of 60 s reach the first slow boundary (480 s); the medium
    # group runs at 240 s and 480 s.
    for _ in range(8):
        simulator.step()
    plan = simulator.get_composition_plan()
    observed = set()
    for functor, items in _writer_calls(simulator):
        context = dict(items)
        assert context["role"] not in _NOT_STAGES
        matches = [
            stage for stage in plan.stages
            if stage.operator_kind == functor
            and stage.stage_metadata["role"] == context["role"]
            and stage.stage_metadata["commit_stage"] == context["stage"]
        ]
        assert len(matches) == 1, (functor, context)
        stage = matches[0]
        interval = context["composition_interval"]
        assert context["dt"] == interval * stage.dt_fraction
        assert context["interval_start"] == context["composition_start"] + interval * stage.stage_metadata["start_fraction"]
        observed.add(stage.stage_id)
    assert observed == {stage.stage_id for stage in plan.stages}


def test_variable_provenance_commit_context_is_opt_in():
    simulator = _relaxations(("relax_a", "relax_b", "relax_c"), splitting_mode="strang")
    simulator.step()
    simulator.step()
    plain = simulator.get_variable_provenance("relax_a")
    assert [set(record) for record in plain] == [
        {"functor", "mode", "step", "role", "provider_uid", "provider_epoch"},
        {"functor", "mode", "step", "role"},
    ]
    detailed = simulator.get_variable_provenance("relax_a", include_commit_context=True)
    writers = [record for record in detailed if record["role"] == "writer"]
    # Two steps, two half-step calls each: four distinct generation records.
    assert [(record["commit_context"]["step"], record["commit_context"]["role"]) for record in writers] == [
        (0, "first_half"), (0, "second_half"), (1, "first_half"), (1, "second_half")
    ]
    assert len({record["commit_index"] for record in writers}) == 4
    writers[0]["commit_context"]["role"] = "forged"
    again = simulator.get_variable_provenance("relax_a", include_commit_context=True)
    assert again[0]["commit_context"]["role"] == "first_half"
    assert simulator.get_variable_provenance("relax_a") == plain


# --------------------------------------------------------------------------
# GraphExecutor: declared plan comparison and runtime witness (executor.py).
# --------------------------------------------------------------------------

_CONFIGURED = "Functor:test.fixtures.dummy_functor.DummyConfiguredOutputFunctor"
_FAST_NODES = (("node_a", "wa", "fast"), ("node_b", "wb", "fast"))
_MEDIUM_NODES = _FAST_NODES + (("node_m", "wm", "medium"),)
_DT = 60.0


def _node_type(variable: str) -> str:
    return f"{_CONFIGURED}.t10_{variable}"


def _registry(nodes) -> NodeRegistry:
    registry = NodeRegistry(SSOTCatalog())
    for _, variable, _ in nodes:
        definition = registry.register_functor_instance(
            DummyConfiguredOutputFunctor(config={"output_name": variable}), node_type_id=_node_type(variable)
        )
        definition.outputs[variable] = _GE._dimensionless_state_port(variable)
    return registry


def _graph(nodes, integration_plan=None) -> GraphIR:
    return GraphIR.from_dict({
        "schema_version": "1.0.0",
        "nodes": {
            node_id: {
                "class_type": _node_type(variable),
                "inputs": {},
                "outputs": [variable],
                "config": {"output_name": variable, "execution_group": group},
            }
            for node_id, variable, group in nodes
        },
        "metadata": {} if integration_plan is None else {"integration_plan": integration_plan},
    })


def _config(**settings) -> Dict[str, Any]:
    config = {"device": "cpu", "time_step": _DT, "medium_interval_seconds": 120.0}
    config.update(settings)
    return config


def _initial(nodes) -> Dict[str, torch.Tensor]:
    return {variable: torch.tensor(0.0) for _, variable, _ in nodes}


def _executor(nodes, config, integration_plan=None) -> GraphExecutor:
    registry = _registry(nodes)
    plan = compile_graph(_graph(nodes, integration_plan), registry)
    return GraphExecutor(plan, registry=registry, simulator_config=config)


def _exported(nodes, config) -> Dict[str, Any]:
    """The simulator plan of a prepared run of the same graph without a declaration."""
    probe = _executor(nodes, config)
    probe.execute(_initial(nodes), n_steps=0, dt=_DT)
    return probe._simulator.get_composition_plan().to_dict()


def test_executor_without_a_declared_plan_records_not_declared():
    executor = _executor(_FAST_NODES, _config(splitting_mode="strang"))
    state = executor.execute(_initial(_FAST_NODES), n_steps=2, dt=_DT)
    # Pin of the default path: two Strang steps, node_a runs two half-steps per step.
    assert float(state["wa"]) == 4.0 and float(state["wb"]) == 2.0
    witness = executor.integration_plan_witness
    assert witness["status"] == "not_declared"
    assert witness["executed_stage_check"] is None


@pytest.mark.parametrize("nodes, mode", [(_FAST_NODES, "strang"), (_MEDIUM_NODES, "strang"), (_MEDIUM_NODES, "lie")])
def test_executor_records_a_matching_declared_plan_as_the_witness(nodes, mode):
    config = _config(splitting_mode=mode)
    declared = _exported(nodes, config)
    executor = _executor(nodes, config, declared)
    audit = executor.plan.report.metadata["audit_summary"]["integration_plan_audit"]
    assert audit["passed"] is True and audit["execution_semantics"] == "metadata_only"
    executor.execute(_initial(nodes), n_steps=3, dt=_DT)
    witness = executor.integration_plan_witness
    assert witness["document_kind"] == "integration_plan_runtime_witness"
    assert witness["status"] == "matched"
    assert witness["declared_schedule_sha256"] == witness["executed_schedule_sha256"]
    assert len(witness["declared_schedule_sha256"]) == 64
    assert witness["stage_ids"] == [stage["stage_id"] for stage in declared["stages"]]
    assert [entry["phase"] for entry in witness["verifications"]] == ["prepared"]
    check = witness["executed_stage_check"]
    assert check["passed"] is True and check["violations"] == []
    assert check["checked_steps"] == [0, 1, 2]
    assert check["stages_not_observed"] == []
    fast_stages = [stage for stage in declared["stages"] if stage["stage_metadata"]["execution_group"] == "fast"]
    assert check["checked_calls"] >= 3 * len(fast_stages)


def test_executor_refuses_a_mismatching_declared_plan_before_any_step():
    declared = _exported(_FAST_NODES, _config(splitting_mode="lie"))
    executor = _executor(_FAST_NODES, _config(splitting_mode="strang"), declared)
    with pytest.raises(ValueError, match="integration_plan"):
        executor.execute(_initial(_FAST_NODES), n_steps=2, dt=_DT)
    assert executor.execution_summary["step_count"] == 0
    assert executor.integration_plan_witness is None


@pytest.mark.parametrize("mutation", ["order", "substeps", "scheme", "group"])
def test_executor_refuses_a_plan_that_differs_in_any_schedule_field(mutation):
    config = _config(splitting_mode="strang")
    declared = _exported(_MEDIUM_NODES, config)
    stages = declared["stages"]
    if mutation == "order":
        stages[0]["operator_kind"], stages[1]["operator_kind"] = stages[1]["operator_kind"], stages[0]["operator_kind"]
        stages[2]["operator_kind"] = stages[0]["operator_kind"]
    elif mutation == "substeps":
        stages[1]["stage_metadata"]["substeps"] = {"source": "substep_override", "count": 2}
    elif mutation == "scheme":
        declared["scheme"] = "lie"
    else:
        stages[-1]["stage_metadata"]["group_trigger"] = {"kind": "interval", "seconds": 240.0}
    executor = _executor(_MEDIUM_NODES, config, declared)
    with pytest.raises(ValueError, match="integration_plan"):
        executor.execute(_initial(_MEDIUM_NODES), n_steps=1, dt=_DT)
    assert executor.execution_summary["step_count"] == 0


def test_numeric_spelling_of_the_declared_plan_is_canonical():
    config = _config(splitting_mode="strang")
    declared = _exported(_FAST_NODES, config)
    middle = declared["stages"][1]
    middle["dt_fraction"] = 1
    middle["stage_metadata"]["start_fraction"] = 0
    executor = _executor(_FAST_NODES, config, declared)
    executor.execute(_initial(_FAST_NODES), n_steps=1, dt=_DT)
    assert executor.integration_plan_witness["status"] == "matched"


def test_a_plan_without_simulator_stages_is_not_compared():
    declared = {
        "plan_id": "metadata_only_transport",
        "scheme": "single",
        "stages": [{
            "stage_id": "transport", "operator_kind": "transport", "solver_name": "implicit_midpoint",
            "structure_kind": "iphs", "dt_fraction": 1.0, "coupling_mode": "direct",
        }],
    }
    executor = _executor(_FAST_NODES, _config(), declared)
    executor.execute(_initial(_FAST_NODES), n_steps=1, dt=_DT)
    witness = executor.integration_plan_witness
    assert witness["status"] == "not_simulator_owned"
    assert witness["executed_stage_check"] is None


def test_mixed_runtime_owners_are_refused():
    config = _config(splitting_mode="lie")
    declared = _exported(_FAST_NODES, config)
    declared["stages"].append({
        "stage_id": "transport", "operator_kind": "transport", "solver_name": "implicit_midpoint",
        "structure_kind": "iphs", "dt_fraction": 1.0, "coupling_mode": "direct",
    })
    executor = _executor(_FAST_NODES, config, declared)
    with pytest.raises(ValueError, match="runtime_owner"):
        executor.execute(_initial(_FAST_NODES), n_steps=1, dt=_DT)
    assert executor.execution_summary["step_count"] == 0


def test_the_witness_is_a_detached_copy():
    config = _config(splitting_mode="strang")
    executor = _executor(_FAST_NODES, config, _exported(_FAST_NODES, config))
    executor.execute(_initial(_FAST_NODES), n_steps=1, dt=_DT)
    witness = executor.integration_plan_witness
    witness["status"] = "forged"
    witness["executed_stage_check"]["violations"].append("forged")
    again = executor.integration_plan_witness
    assert again["status"] == "matched" and again["executed_stage_check"]["violations"] == []


def test_an_executed_call_outside_the_declared_plan_is_refused(monkeypatch):
    config = _config(splitting_mode="strang")
    executor = _executor(_FAST_NODES, config, _exported(_FAST_NODES, config))
    original = RELICSimulator.get_variable_provenance

    def with_forged_record(self, var_name=None, *, include_retention=False, include_commit_context=False):
        records = original(
            self, var_name, include_retention=include_retention, include_commit_context=include_commit_context
        )
        if include_commit_context and var_name is None:
            records = {name: list(items) for name, items in records.items()}
            records["wa"].append({
                "functor": "node_a", "mode": "UpdateMode.ACCUMULATE", "step": None, "role": "writer",
                "provider_uid": None, "provider_epoch": None, "commit_index": 10_000,
                "commit_context": {
                    "step": 0, "composition_start": 0.0, "composition_interval": _DT, "stage": 1,
                    "role": "middle", "interval_start": 0.0, "dt": _DT,
                },
            })
        return records

    monkeypatch.setattr(RELICSimulator, "get_variable_provenance", with_forged_record)
    with pytest.raises(RuntimeError, match="executed"):
        executor.execute(_initial(_FAST_NODES), n_steps=1, dt=_DT)
    check = executor.integration_plan_witness["executed_stage_check"]
    assert check["passed"] is False and len(check["violations"]) == 1


def test_a_new_provider_epoch_with_another_plan_stops_the_run(monkeypatch):
    config = _config(splitting_mode="strang")
    executor = _executor(_FAST_NODES, config, _exported(_FAST_NODES, config))
    original_step = RELICSimulator.step
    calls = {"count": 0}

    def step_then_register(self, dt=None):
        result = original_step(self, dt)
        calls["count"] += 1
        if calls["count"] == 1:
            self.register_functor(
                DummyConfiguredOutputFunctor(name="late", config={"output_name": "wc"}), execution_group="fast"
            )
        return result

    monkeypatch.setattr(RELICSimulator, "step", step_then_register)
    initial = dict(_initial(_FAST_NODES), wc=torch.tensor(0.0))
    with pytest.raises(RuntimeError, match="provider epoch"):
        executor.execute(initial, n_steps=3, dt=_DT)
    assert executor.execution_summary["step_count"] == 1


def test_the_executor_reverifies_the_declared_plan_after_a_restore(tmp_path):
    config = _config(splitting_mode="strang")
    declared = _exported(_MEDIUM_NODES, config)
    writer = _executor(_MEDIUM_NODES, config, declared)
    writer.execute(_initial(_MEDIUM_NODES), n_steps=2, dt=_DT)
    path = tmp_path / "t10_witness.ckpt"
    writer.create_checkpoint(path)
    resumed = _executor(_MEDIUM_NODES, config, declared)
    resumed.execute_from_checkpoint(_initial(_MEDIUM_NODES), path, n_steps=2, dt=_DT)
    witness = resumed.integration_plan_witness
    assert witness["status"] == "matched"
    assert [entry["phase"] for entry in witness["verifications"]] == ["prepared", "restored"]
    assert witness["verifications"][-1]["step_count"] == 2
    assert witness["executed_stage_check"]["checked_steps"] == [2, 3]
    assert witness["executed_schedule_sha256"] == writer.integration_plan_witness["executed_schedule_sha256"]


# --------------------------------------------------------------------------
# Restart decision: the plan identity is bound by the existing exact-restart
# contract; no checkpoint field is added.
# --------------------------------------------------------------------------

_RESTART_CONFIG = {"splitting_mode": "strang", "substep_overrides": {"fast_hold_reader": {"count": 2}}}
_PRE_STEP8_RUNTIME_FIELDS = (
    "_active_composition", "_adaptive_atol", "_adaptive_dt_enabled", "_adaptive_dt_max", "_adaptive_dt_min",
    "_adaptive_monitor_vars", "_adaptive_next_dt", "_adaptive_rtol", "_baseline_config_reference",
    "_baseline_snapshot_path", "_baseline_snapshot_reference", "_bootstrap_groups", "_fast_order_mode",
    "_forcing_cursor", "_implicit_midpoint_max_iter", "_implicit_midpoint_tolerance", "_initial_state",
    "_last_medium_execution_time", "_last_slow_execution_time", "_last_state_reset_counter",
    "_last_step_had_violation", "_multirate_exchanges", "_previous_day_index", "_previous_medium_index",
    "_previous_month_index", "_previous_slow_index", "_provenance_evicted_counts", "_resolve_cross_group_deps",
    "_scale_check_history", "_splitting_mode", "_splitting_policy", "_step_functor_increments",
    "_step_provenance_detail", "_step_update_sources", "_substep_overrides", "_time_integration_method",
    "checkpoint_frequency", "current_time", "enable_auto_scale", "enable_conservation_checks",
    "last_checkpoint_time", "max_time", "output_frequency", "run_id", "state", "step_count",
    "strict_conservation_enabled", "strict_scale_detection", "time_series", "time_step", "total_execution_time",
)


def _restart_writer(tmp_path: Path, **config) -> Tuple[RELICSimulator, Path]:
    writer = _T7._ramp_simulator([_T7._MEDIUM_MEAN], **config)
    _T7._run(writer, 3)
    path = tmp_path / "t10_restart.ckpt"
    writer.create_checkpoint(path)
    return writer, path


def test_an_exact_restore_reproduces_the_composition_plan(tmp_path):
    writer, path = _restart_writer(tmp_path, **_RESTART_CONFIG)
    resumed = _T7._ramp_simulator([_T7._MEDIUM_MEAN], **_RESTART_CONFIG)
    resumed.restore_checkpoint(path)
    assert resumed.get_composition_plan().to_dict() == writer.get_composition_plan().to_dict()
    _T7._assert_rows_equal(_T7._run(writer, 4), _T7._run(resumed, 4))


@pytest.mark.parametrize(
    "changed",
    [
        {"splitting_mode": "lie", "substep_overrides": {"fast_hold_reader": {"count": 2}}},
        {"splitting_mode": "strang", "substep_overrides": {"fast_hold_reader": {"count": 3}}},
        {**_RESTART_CONFIG, "fast_order": "reverse"},
        {**_RESTART_CONFIG, "medium_interval_seconds": 480.0},
    ],
    ids=["splitting_mode", "substep_override", "fast_order", "medium_interval"],
)
def test_restores_whose_plan_would_differ_are_refused_by_the_existing_contract(tmp_path, changed):
    writer, path = _restart_writer(tmp_path, **_RESTART_CONFIG)
    target = _T7._ramp_simulator([_T7._MEDIUM_MEAN], **changed)
    before = target.get_composition_plan().to_dict()
    assert before != writer.get_composition_plan().to_dict()
    epoch = target.provider_epoch
    with pytest.raises(ValueError):
        target.restore_checkpoint(path)
    assert target.get_composition_plan().to_dict() == before
    assert target.provider_epoch == epoch


def test_the_checkpoint_carries_no_composition_plan_field(tmp_path):
    _, path = _restart_writer(tmp_path, **_RESTART_CONFIG)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == checkpoint_module.CHECKPOINT_FORMAT_VERSION == 3
    # The runtime field names of this checkpoint, recorded on 2026-10-20 before
    # CR-A-04 step 8 (logs/06_runtime_fields_probe_pre_edit.log of
    # 20261020_cr_a04_impl_06): step 8 adds none.
    assert sorted(payload["runtime"]) == sorted(_PRE_STEP8_RUNTIME_FIELDS)
    assert set(checkpoint_module._CORE_RUNTIME_FIELDS) | set(checkpoint_module._ADDITIONAL_RUNTIME_FIELDS) <= set(
        payload["runtime"]
    )
    flat = repr(sorted(payload) + sorted(payload["runtime"]))
    assert "composition_plan" not in flat and "integration_plan" not in flat
