"""CR-A-04 D7 and T3: auditable half-steps through the commit context of group calls.

Design 20261019_cr_a04_design_01, D7, section 2 (provenance) and T3. Before step 5
the writer records of the two Strang half-step calls of one member were identical
(functor, mode, step index None, provider uid and epoch) and StateManager events
carry only the commit index, so a half-step could not be audited. Step 5
(execution_group.py) builds the commit context where the group owner knows it:

    commit_context = {step, composition_start, composition_interval, stage, role,
                      interval_start, dt}

with role lie (one call over the step), first_half / middle / second_half (the
Strang palindrome) or group (a group call that is a composition of its own: medium
and slow groups), and records it with the StateManager commit index of the group
commit in the provider dict of every writer record of that call. Joining the writer
records with the StateManager events by commit index gives per-half-step ledgers
(T3, part testable in step 5). Since step 7 the group commit passes the context to
StateManager.apply_updates (relic/core/state/manager.py), so every event of the
commit carries a copy with the commit's element-ledger net change (T3b), and the
commits that are not group commits carry their own roles: stage_view (the
restatement of a multi-writer boundary-rate stage view after a call),
composition_mean (the finalization commit of the composition) and exchange (one
multirate exchange mean). Per-half-step ledgers then follow from the events alone.

References: Strang (1968) SIAM J. Numer. Anal. 5(3):506-517; Hairer, Lubich and
Wanner (2006) Geometric Numerical Integration, 2nd ed., Sect. II.5; W3C PROV-DM
(2013), generation and derivation records; LeVeque (2002), finite-volume balance
per commit.
Mathematical expression: palindrome of n members over [t, t + H]: stages k = 0..n-2
first_half over [t, t + H/2], k = n-1 middle over [t, t + H], k = n..2n-2
second_half over [t + H/2, t + H]; per-commit nitrogen ledger
r_c = sum_j (N_j,post - N_j,pre) - I_N dt_c + sum_v q_v,c dt_c = 0.
Parameter meanings: k the stage (index of the group call in its composition); H the
step; N_j soil nitrogen pools; I_N the held organic nitrogen input of mineralization;
q_v,c the boundary rate v (N2O, N2, nitrate leaching) committed by commit c; dt_c the
interval of the call.
Units: t, H, dt_c [s]; N g N m-2; I_N and q g N m-2 s-1.
Domain/constraints: CPU, float64; per-commit residual within 1e-12 g N m-2.
Sign convention: exports positive.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.core.simulator.execution_group import GroupComposition

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
_S2 = _load_test_module("core/splitting_strang_replace_rates.test.py", "_cr_a04_s2_replace_rates")

_PER_COMMIT_TOL = 1.0e-12  # g N m-2
_BOUNDARY_RATES = ("n2o_flux", "n2_production", "nitrogen_leaching_flux")
_STRANG_ROLES = [
    ("mineralization", "first_half"),
    ("nitrification", "first_half"),
    ("denitrification", "first_half"),
    ("nitrogen_leaching", "middle"),
    ("denitrification", "second_half"),
    ("nitrification", "second_half"),
    ("mineralization", "second_half"),
]


def _call_records(simulator: RELICSimulator) -> List[Dict[str, Any]]:
    """One entry per group-call writer (provider dict shared by the call's records), in commit order."""
    calls: Dict[int, Dict[str, Any]] = {}
    for records in simulator._step_provenance_detail.values():
        for entry in records:
            provider = entry[4] if len(entry) > 4 else None
            assert isinstance(provider, dict) and "commit_index" in provider and "commit_context" in provider, entry
            calls[id(provider)] = {"functor": entry[0], **provider}
    return sorted(calls.values(), key=lambda call: (call["commit_index"], call["functor"]))


def test_strang_writer_records_carry_stage_role_interval_and_commit_index():
    simulator = _S2._nitrogen_simulator("strang")
    first_commit = _S2._commit_count(simulator)
    simulator.step(dt=600.0)
    calls = _call_records(simulator)
    assert [(call["functor"], call["commit_context"]["role"]) for call in calls] == _STRANG_ROLES
    for stage, call in enumerate(calls):
        context = call["commit_context"]
        assert context["stage"] == stage
        assert context["step"] == 0
        assert context["composition_start"] == 0.0 and context["composition_interval"] == 600.0
        if context["role"] == "first_half":
            assert (context["interval_start"], context["dt"]) == (0.0, 300.0)
        elif context["role"] == "middle":
            assert (context["interval_start"], context["dt"]) == (0.0, 600.0)
        else:
            assert (context["interval_start"], context["dt"]) == (300.0, 300.0)
    commits = [call["commit_index"] for call in calls]
    assert commits == sorted(commits) and len(set(commits)) == len(commits)
    assert commits[0] == first_commit
    # The two half-step records of one member are now distinguishable.
    denitrification = [call for call in calls if call["functor"] == "denitrification"]
    assert denitrification[0]["commit_context"] != denitrification[1]["commit_context"]


def test_lie_and_group_calls_carry_their_roles():
    simulator = _S2._nitrogen_simulator("lie")
    simulator.step(dt=600.0)
    calls = _call_records(simulator)
    assert {call["commit_context"]["role"] for call in calls} == {"lie"}
    assert {call["commit_context"]["stage"] for call in calls} == {0}
    assert len({call["commit_index"] for call in calls}) == 1
    assert {(call["commit_context"]["interval_start"], call["commit_context"]["dt"]) for call in calls} == {
        (0.0, 600.0)
    }

    grouped = RELICSimulator({
        "time_step": 60.0, "device": "cpu", "medium_interval_seconds": 60.0,
        "state_units": {"relax_a": "dimensionless", "relax_b": "dimensionless"},
    })
    grouped.initialize_state({"relax_a": 1.0, "relax_b": 2.0})
    grouped.register_functor(_T1._Relaxation("relax_a", "relax_a"), execution_group="fast")
    grouped.register_functor(_T1._Relaxation("relax_b", "relax_b"), execution_group="medium")
    for _ in range(3):
        grouped.step(dt=60.0)
    medium = [entry[4]["commit_context"] for entry in grouped._step_provenance_detail["relax_b"]]
    fast = [entry[4]["commit_context"] for entry in grouped._step_provenance_detail["relax_a"]]
    assert medium and {context["role"] for context in medium} == {"group"}
    assert {context["role"] for context in fast} == {"lie"}
    assert [context["step"] for context in fast] == [0, 1, 2]


def test_t3_half_step_nitrogen_ledgers_close_from_writer_contexts_joined_to_commit_events():
    simulator = _S2._nitrogen_simulator("strang")
    recorder = _S2._Recorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    simulator.step(dt=3600.0)
    events: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    for event in recorder.events:
        events.setdefault(event["commit"], {})[event["variable"]] = event
    input_rate = _S2._scalar(simulator, "organic_nitrogen_input")
    calls = _call_records(simulator)
    assert [(call["functor"], call["commit_context"]["role"]) for call in calls] == _STRANG_ROLES
    for call in calls:
        commit = events[call["commit_index"]]
        dt_c = call["commit_context"]["dt"]
        change = sum(
            float(commit[key]["post"].sum() - commit[key]["pre"].sum()) for key in _S2._SOIL_N if key in commit
        )
        exports = sum(float(commit[key]["post"].sum()) for key in _BOUNDARY_RATES if key in commit) * dt_c
        inputs = input_rate * dt_c if call["functor"] == "mineralization" else 0.0
        assert change - inputs + exports == pytest.approx(0.0, abs=_PER_COMMIT_TOL), call


def test_t3_commit_events_carry_the_commit_context_themselves():
    simulator = _S2._nitrogen_simulator("strang")
    recorder = _S2._Recorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    simulator.step(dt=600.0)
    no3_events = [event for event in recorder.events if event["variable"] == "no3_nitrogen"]
    assert no3_events
    for event in no3_events:
        context = event["options"].get("commit_context")
        assert isinstance(context, dict) and context["role"] in {"first_half", "middle", "second_half"}


# --------------------------------------------------------------------------
# Step 7: the events alone (no join with the writer records).
# --------------------------------------------------------------------------

_T7 = _load_test_module("core/splitting_multirate_exchanges.test.py", "_cr_a04_t7_multirate")
_CALL_ROLES = {"first_half", "middle", "second_half"}


def _commit_events(recorder: Any) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Events by commit index, then by variable."""
    commits: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for event in recorder.events:
        commits.setdefault(event["commit"], {})[event["variable"]] = event
    return commits


def _context_of(events: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The context of one commit: every event of the commit carries an equal copy."""
    contexts = [event["options"].get("commit_context") for event in events.values()]
    assert contexts and isinstance(contexts[0], dict), contexts
    assert all(context == contexts[0] for context in contexts), contexts
    return contexts[0]


def test_t3_half_step_nitrogen_ledgers_close_from_the_commit_context_events():
    simulator = _S2._nitrogen_simulator("strang")
    recorder = _S2._Recorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    simulator.step(dt=3600.0)
    input_rate = _S2._scalar(simulator, "organic_nitrogen_input")
    commits = _commit_events(recorder)
    contexts = {commit: _context_of(events) for commit, events in commits.items()}
    calls = [commit for commit in sorted(commits) if contexts[commit]["role"] in _CALL_ROLES]
    assert [contexts[commit]["stage"] for commit in calls] == list(range(len(_STRANG_ROLES)))
    assert [(_STRANG_ROLES[contexts[c]["stage"]][0], contexts[c]["role"]) for c in calls] == _STRANG_ROLES
    for commit in calls:
        context, events = contexts[commit], commits[commit]
        functor = _STRANG_ROLES[context["stage"]][0]
        dt_c = context["dt"]
        change = sum(float(events[key]["post"].sum() - events[key]["pre"].sum()) for key in _S2._SOIL_N if key in events)
        exports = sum(float(events[key]["post"].sum()) for key in _BOUNDARY_RATES if key in events) * dt_c
        inputs = input_rate * dt_c if functor == "mineralization" else 0.0
        assert change - inputs + exports == pytest.approx(0.0, abs=_PER_COMMIT_TOL), context
        # The StateManager nitrogen ledger of the commit counts the soil pools; the
        # three boundary rates are not nitrogen-ledger members of the SCM, so its
        # net change is the storage change (the fixture runs without the check).
        assert context["element_net_change"]["nitrogen"] == pytest.approx(change, abs=_PER_COMMIT_TOL), context
        assert context["conservation_checked"] is False
    others = [commit for commit in sorted(commits) if commit not in calls]
    assert [contexts[commit]["role"] for commit in others] == ["stage_view"] * 3 + ["composition_mean"]
    for commit in others[:3]:
        # A stage view restates n2o_flux (two writers) right after the call it follows.
        call = max(c for c in calls if c < commit)
        view = {key: value for key, value in contexts[commit].items() if key not in ("role", "element_net_change")}
        assert view == {key: value for key, value in contexts[call].items() if key not in ("role", "element_net_change")}
        assert set(commits[commit]) == {"n2o_flux"} and contexts[commit]["element_net_change"] == {}


def test_t3_composition_mean_commit_carries_its_own_role():
    simulator = _S2._nitrogen_simulator("strang")
    recorder = _S2._Recorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    for _ in range(2):
        simulator.step(dt=600.0)
    commits = _commit_events(recorder)
    means = sorted(commit for commit, events in commits.items() if _context_of(events)["role"] == "composition_mean")
    assert len(means) == 2
    for step, commit in enumerate(means):
        assert _context_of(commits[commit]) == {
            "step": step, "composition_start": 600.0 * step, "composition_interval": 600.0,
            "stage": len(_STRANG_ROLES), "role": "composition_mean", "interval_start": 600.0 * step, "dt": 600.0,
            "conservation_checked": False, "element_net_change": {},
        }
        assert {"n2o_flux", "n2_production"} <= set(commits[commit])
        # The finalization commit is the last commit of its step.
        assert commit == max(c for c, events in commits.items() if _context_of(events)["step"] == step)


def test_t3_lie_and_group_commit_events_carry_their_roles():
    simulator = _S2._nitrogen_simulator("lie")
    recorder = _S2._Recorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    simulator.step(dt=600.0)
    commits = _commit_events(recorder)
    assert len(commits) == 1
    context = _context_of(next(iter(commits.values())))
    assert {key: value for key, value in context.items() if key != "element_net_change"} == {
        "step": 0, "composition_start": 0.0, "composition_interval": 600.0, "stage": 0, "role": "lie",
        "interval_start": 0.0, "dt": 600.0, "conservation_checked": False,
    }

    grouped = RELICSimulator({
        "time_step": 60.0, "device": "cpu", "medium_interval_seconds": 60.0,
        "state_units": {"relax_a": "dimensionless", "relax_b": "dimensionless"},
    })
    grouped.initialize_state({"relax_a": 1.0, "relax_b": 2.0})
    grouped.register_functor(_T1._Relaxation("relax_a", "relax_a"), execution_group="fast")
    grouped.register_functor(_T1._Relaxation("relax_b", "relax_b"), execution_group="medium")
    recorder = _S2._Recorder()
    grouped.state_manager.set_provenance_recorder(recorder)
    for _ in range(3):
        grouped.step(dt=60.0)
    roles = {
        event["variable"]: set() for event in recorder.events
    }
    for event in recorder.events:
        roles[event["variable"]].add(event["options"]["commit_context"]["role"])
    assert roles == {"relax_a": {"lie"}, "relax_b": {"group"}}
    medium = [event["options"]["commit_context"] for event in recorder.events if event["variable"] == "relax_b"]
    assert [(context["step"], context["stage"], context["dt"]) for context in medium] == [(0, 0, 60.0), (1, 0, 60.0), (2, 0, 60.0)]


def test_t3_exchange_commit_events_carry_the_exchange_role():
    simulator = _T7._ramp_simulator([_T7._MEDIUM_MEAN])
    recorder = _S2._Recorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    for _ in range(6):
        simulator.step(dt=_T7._DT)
    name = _T7._mean_name("ramp_medium")
    contexts = [event["options"]["commit_context"] for event in recorder.events if event["variable"] == name]
    # The mean of step k covers [interval_start, t_k + h]; the medium blocks are [0, 240] and [240, 480].
    assert contexts == [
        {
            "step": step, "composition_start": _T7._DT * step, "composition_interval": _T7._DT, "stage": 0,
            "role": "exchange", "interval_start": _T7._MEDIUM * ((_T7._DT * step) // _T7._MEDIUM), "dt": _T7._DT,
            "conservation_checked": False, "element_net_change": {},
        }
        for step in range(6)
    ]


def test_a_group_commit_whose_calls_have_different_contexts_is_refused():
    simulator = RELICSimulator({
        "time_step": 60.0, "device": "cpu",
        "state_units": {"relax_a": "dimensionless", "relax_b": "dimensionless"},
    })
    simulator.initialize_state({"relax_a": 1.0, "relax_b": 2.0})
    simulator.register_functor(_T1._Relaxation("relax_a", "relax_a"), execution_group="fast")
    simulator.register_functor(_T1._Relaxation("relax_b", "relax_b"), execution_group="fast")
    before = simulator.state_manager.get_state()
    counter = _S2._commit_count(simulator)
    # A half-step call of two Functors, one of which already ran in this composition:
    # first_half for relax_b, second_half for relax_a (no builder makes such a call).
    composition = GroupComposition(interval=60.0, member_calls={"relax_a": [0.5]}, stage_count=1, start_time=0.0)
    with pytest.raises(RuntimeError, match="D7"):
        simulator._execute_group(["relax_a", "relax_b"], 30.0, start_time=30.0, composition=composition)
    after = simulator.state_manager.get_state()
    assert _S2._commit_count(simulator) == counter
    assert all(torch.equal(before[key], after[key]) for key in before)
