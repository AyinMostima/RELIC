"""CR-A-04 step 7 (D7): the commit context of StateManager.apply_updates and its events.

Design 20261019_cr_a04_design_01, D7 and section 2 ("Provenance"): StateManager
events carried only the commit index (step = update counter), and the per-commit
element ledger was computed but not recorded. Step 7 adds an optional
``commit_context`` argument to StateManager.apply_updates (relic/core/state/manager.py):

- default None: every event and every recorder delivery is byte-identical to the
  pre-change behaviour (pinned below with digests computed before the change);
- a context is a mapping of plain values (None, bool, int, float, str; floats
  finite) with non-empty string keys; it is validated before any state work, and a
  copy is stored in every provenance event of the commit under "commit_context",
  together with "conservation_checked" (whether the commit's ledger check ran) and
  "element_net_change" (the signed net change of the StateManager element ledger of
  this commit, by element; the routine of check_element_conservation);
- a recorder receives it as the keyword ``commit_context`` only when its
  record_state_update declares that keyword or **kwargs (six-argument recorders keep
  receiving exactly six arguments).

The original simulator pins (section B) fixed all pre-context fields. The
2026-10-20 unit metadata repair additionally resolves each accepted target's
canonical schema/SCM unit. These comparisons assert every real event unit first,
then project only the unit field to the historical schema-only view for the
unchanged old SHA witnesses. The projection never enters production or real logs;
it checks that order, tensors, source attribution and the six-argument protocol
remain unchanged apart from the explicit unit repair.

References: W3C PROV-DM (2013), generation and derivation records; LeVeque (2002),
finite-volume balance per commit; Strang (1968) SIAM J. Numer. Anal. 5(3):506-517.
Mathematical expression: element ledger of one commit
R_E = sum_i Delta S_E,i + dt (sum F_E,out - sum F_E,in) over the members of element E;
digest = SHA-256 over a canonical byte encoding of the recorded values.
Parameter meanings: Delta S_E,i realized storage changes of the commit; F_E boundary
rates of the commit; dt its interval.
Units: g E m-2 for C, N, P; kg m-2 for water; dt [s].
Domain/constraints: CPU, float64 test default dtype (test/conftest.py).
Sign convention: positive R_E is a net gain of element storage not explained by the
committed boundary rates.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

import relic.core.state.conservation as state_conservation
from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode

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
_T7 = _load_test_module("core/splitting_multirate_exchanges.test.py", "_cr_a04_t7_multirate")

_CONTEXT = {
    "step": 3,
    "composition_start": 1800.0,
    "composition_interval": 600.0,
    "stage": 1,
    "role": "middle",
    "interval_start": 1800.0,
    "dt": 600.0,
}


# --------------------------------------------------------------------------
# Canonical encoding and recorders.
# --------------------------------------------------------------------------


def _canonical(value: Any) -> bytes:
    """Byte encoding of a recorded value: type tag, dtype, shape and raw bytes of tensors."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        return b"T" + str(tensor.dtype).encode("ascii") + repr(tuple(tensor.shape)).encode("ascii") + tensor.numpy().tobytes()
    if isinstance(value, dict):
        return b"D{" + b"".join(_canonical(key) + b":" + _canonical(value[key]) for key in sorted(value)) + b"}"
    if isinstance(value, (list, tuple)):
        opening = b"L[" if isinstance(value, list) else b"U("
        return opening + b",".join(_canonical(item) for item in value) + b"]"
    if value is None or isinstance(value, (bool, int, float, str)):
        return type(value).__name__.encode("ascii") + b":" + repr(value).encode("ascii")
    raise TypeError(f"no canonical encoding for {type(value).__name__}")


def _digest(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(_canonical(value))
    return digest.hexdigest()


class _SixArgumentRecorder:
    """Recorder with the six-argument protocol (no keywords)."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def record_state_update(self, step, variable, pre_value, post_value, functor_increments, unit):
        self.calls.append((step, variable, pre_value.clone(), post_value.clone(), dict(functor_increments), unit))


class _KeywordRecorder:
    """Recorder accepting any keyword (the existing test-recorder pattern)."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def record_state_update(self, *args, **kwargs):
        self.calls.append((tuple(args), dict(kwargs)))


class _NamedKeywordRecorder:
    """Recorder declaring the commit_context keyword explicitly."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def record_state_update(self, step, variable, pre_value, post_value, functor_increments, unit, *,
                            commit_context=None):
        self.calls.append((step, variable, commit_context))


class _ContextRecorder(_KeywordRecorder):
    """Keyword recorder that also exposes a notification context (persistent destination)."""

    def get_notification_context(self) -> Dict[str, str]:
        return {"run_id": "pin-run", "output_dir": "pin-output"}


def _stripped(calls: List[tuple]) -> List[tuple]:
    """Keyword-recorder calls without the commit_context keyword (the pre-change payload)."""
    return [(args, {key: value for key, value in kwargs.items() if key != "commit_context"}) for args, kwargs in calls]


# --------------------------------------------------------------------------
# A. Default path of StateManager.apply_updates (no commit_context): pinned.
# --------------------------------------------------------------------------


def _pin_manager() -> StateManager:
    schema = {
        "probe_temperature": StateVariableSpec(name="probe_temperature", units="K", bounds=(0.0, 400.0), required=False),
        "nh4_nitrogen": StateVariableSpec(name="nh4_nitrogen", units="g N m-2", conserved=True, required=False),
        "no3_nitrogen": StateVariableSpec(name="no3_nitrogen", units="g N m-2", conserved=True, required=False),
        "probe_water_store": StateVariableSpec(name="probe_water_store", units="kg m-2", bounds=(0.0, 10.0), required=False),
    }
    manager = StateManager(schema=schema)
    manager.set_state({
        "probe_temperature": torch.tensor(300.0, dtype=torch.float64),
        "nh4_nitrogen": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "no3_nitrogen": torch.tensor([0.5, 0.25], dtype=torch.float64),
        "probe_water_store": torch.tensor(4.0, dtype=torch.float64),
    })
    return manager


def _item(value, mode=UpdateMode.ACCUMULATE, unit=None, per_timestep=True, source=None) -> UpdateItem:
    return UpdateItem(
        value=torch.as_tensor(value, dtype=torch.float64), mode=mode, unit=unit,
        basis="per_step" if per_timestep else "per_second", per_timestep=per_timestep, source_functor=source,
    )


def _default_commits(manager: StateManager, **extra: Any) -> None:
    """Representative commits: REPLACE, ACCUMULATE transfer, delta_, FLUX, clipping, increments."""
    manager.apply_updates(
        {"probe_temperature": _item(301.25, mode=UpdateMode.REPLACE, unit="K", source="pin_writer")},
        check_conservation=False, update_sources={"probe_temperature": "pin_writer"}, **extra,
    )
    manager.apply_updates(
        {"nh4_nitrogen": _item([-0.25, -0.5], unit="g N m-2"), "no3_nitrogen": _item([0.25, 0.5], unit="g N m-2")},
        dt=600.0, **extra,
    )
    manager.apply_updates({"delta_probe_temperature": torch.tensor(0.125, dtype=torch.float64)}, dt=2.0,
                          check_conservation=False, **extra)
    manager.apply_updates({"probe_water_store": _item(0.1, mode=UpdateMode.FLUX, unit="kg m-2", per_timestep=False)},
                          dt=3.0, check_conservation=False, **extra)
    manager.apply_updates({"probe_water_store": _item(12.0, mode=UpdateMode.REPLACE, unit="kg m-2")},
                          check_conservation=False, **extra)
    manager.apply_updates(
        {"probe_temperature": _item(0.5, unit="K")}, check_conservation=False,
        update_increments={"probe_temperature": {"writer_a": torch.tensor(0.2, dtype=torch.float64),
                                               "writer_b": torch.tensor(0.3, dtype=torch.float64)}},
        **extra,
    )


def _default_path_records() -> Dict[str, Any]:
    six, keyword, contextual = _SixArgumentRecorder(), _KeywordRecorder(), _ContextRecorder()
    records: Dict[str, Any] = {}
    for label, recorder in (("six", six), ("keyword", keyword), ("contextual", contextual)):
        manager = _pin_manager()
        manager.set_provenance_recorder(recorder)
        _default_commits(manager)
        manager.begin_provenance_transaction()
        _default_commits(manager)
        records[f"{label}_pending"] = manager.checkpoint()["pending_provenance_events"]
        manager.commit_provenance_transaction()
        records[f"{label}_calls"] = recorder.calls
        records[f"{label}_state"] = manager.get_state()
        records[f"{label}_counter"] = manager.get_diagnostics()["update_counter"]
    return records


# Computed on 2026-10-20 before the step-7 edit of relic/core/state/manager.py
# (manager.py SHA-256 364a20df...); see 20261020_cr_a04_impl_05/logs.
_DEFAULT_PATH_DIGEST = "5d6642b5ddbeaa4e3fd7e67cce2ebf0f7e3a21744253de8164385e60f7ccf6c3"


def test_default_path_events_and_deliveries_are_pinned_byte_identical():
    records = _default_path_records()
    for label in ("six", "keyword", "contextual"):
        assert all("commit_context" not in event for event in records[f"{label}_pending"])
    assert all(len(call) == 6 for call in records["six_calls"])
    assert all(set(kwargs) <= {"notification_context"} for _, kwargs in records["keyword_calls"])
    assert _digest(records) == _DEFAULT_PATH_DIGEST


def test_existing_apply_updates_parameters_are_unchanged():
    parameters = list(inspect.signature(StateManager.apply_updates).parameters.values())
    assert [(p.name, p.default) for p in parameters[:7]] == [
        ("self", inspect.Parameter.empty),
        ("updates", inspect.Parameter.empty),
        ("dt", None),
        ("check_conservation", True),
        ("rollback_on_failure", False),
        ("update_sources", None),
        ("update_increments", None),
    ]


# --------------------------------------------------------------------------
# B. Simulator events: existing fields pinned (contexts may only be added).
# --------------------------------------------------------------------------


def _run_recorded(simulator, steps: int, dt: float, recorder) -> Dict[str, Any]:
    simulator.state_manager.set_provenance_recorder(recorder)
    for _ in range(steps):
        simulator.step(dt=dt)
    return {"state": simulator.state_manager.get_state()}


def _grouped_simulator():
    from relic.core.simulator import RELICSimulator

    grouped = RELICSimulator({
        "time_step": 60.0, "device": "cpu", "medium_interval_seconds": 60.0,
        "state_units": {"relax_a": "dimensionless", "relax_b": "dimensionless"},
    })
    grouped.initialize_state({"relax_a": 1.0, "relax_b": 2.0})
    grouped.register_functor(_T1._Relaxation("relax_a", "relax_a"), execution_group="fast")
    grouped.register_functor(_T1._Relaxation("relax_b", "relax_b"), execution_group="medium")
    return grouped


_SIMULATOR_CASES = {
    "nitrogen_lie": (lambda: _S2._nitrogen_simulator("lie"), 2, 600.0),
    "nitrogen_strang": (lambda: _S2._nitrogen_simulator("strang"), 2, 600.0),
    "grouped_medium": (_grouped_simulator, 3, 60.0),
    "exchange_medium": (lambda: _T7._ramp_simulator([_T7._MEDIUM_MEAN]), 5, 60.0),
}

# Computed on 2026-10-20 before any step-7 production edit (keyword recorder: the
# positional payload and the keywords other than commit_context).
_SIMULATOR_DIGESTS = {
    "nitrogen_lie": "2349401384f617e14c9ab9e179df56da37498da7b25526ae473fe33efb3142d6",
    "nitrogen_strang": "275f8feeabff328d63e1c3f22f2c13bceb7ff873e70df9839cb8240e259d4b75",
    "grouped_medium": "a9474add1ea5d29051b2127877d8e634a57fc42d69945d81210105181c66ca79",
    "exchange_medium": "c08cdd53e2a53e0d9121fc76ecbd4a854573d4c41dc92e0770779e6a1c26c385",
}
_SIX_ARGUMENT_STRANG_DIGEST = "03ac8825f43c8099983aaef56b4146353e0ca93d8fb5ba45d1375fc6a56ef8a4"


def _simulator_record(case: str) -> Dict[str, Any]:
    build, steps, dt = _SIMULATOR_CASES[case]
    recorder = _KeywordRecorder()
    simulator = build()
    record = _run_recorded(simulator, steps, dt, recorder)
    record["calls"] = _canonical_units_and_historical_view(
        _stripped(recorder.calls), simulator.state_manager, with_keywords=True,
    )
    return record


@pytest.mark.parametrize("case", sorted(_SIMULATOR_CASES))
def test_simulator_event_payloads_are_pinned(case):
    assert _digest(_simulator_record(case)) == _SIMULATOR_DIGESTS[case]


def test_six_argument_recorder_deliveries_of_a_strang_run_are_pinned():
    recorder = _SixArgumentRecorder()
    simulator = _S2._nitrogen_simulator("strang")
    record = _run_recorded(simulator, 2, 600.0, recorder)
    record["calls"] = _canonical_units_and_historical_view(
        recorder.calls, simulator.state_manager, with_keywords=False,
    )
    assert all(len(call) == 6 for call in recorder.calls)
    assert _digest(record) == _SIX_ARGUMENT_STRANG_DIGEST


# --------------------------------------------------------------------------
# C. commit_context behaviour (RED before the manager.py edit).
# --------------------------------------------------------------------------


def _nitrogen_manager() -> StateManager:
    schema = {
        "nh4_nitrogen": StateVariableSpec(name="nh4_nitrogen", units="g N m-2", conserved=True, required=False),
        "no3_nitrogen": StateVariableSpec(name="no3_nitrogen", units="g N m-2", conserved=True, required=False),
        "probe_temperature": StateVariableSpec(name="probe_temperature", units="K", required=False),
    }
    manager = StateManager(schema=schema)
    manager.set_state({
        "nh4_nitrogen": torch.tensor(2.0, dtype=torch.float64),
        "no3_nitrogen": torch.tensor(1.0, dtype=torch.float64),
        "probe_temperature": torch.tensor(290.0, dtype=torch.float64),
    })
    return manager


def _transfer() -> Dict[str, UpdateItem]:
    return {"nh4_nitrogen": _item(-0.25, unit="g N m-2"), "no3_nitrogen": _item(0.25, unit="g N m-2")}


def test_commit_context_is_copied_into_every_event_with_the_element_net_change():
    manager = _nitrogen_manager()
    recorder = _KeywordRecorder()
    manager.set_provenance_recorder(recorder)
    context = dict(_CONTEXT)
    manager.apply_updates(_transfer(), dt=600.0, check_conservation=False, commit_context=context)
    manager.apply_updates(
        {"nh4_nitrogen": _item(-0.125, unit="g N m-2"), "no3_nitrogen": _item(0.125, unit="g N m-2")},
        dt=600.0, check_conservation=True, commit_context=dict(_CONTEXT, stage=2, role="second_half"),
    )
    context["role"] = "mutated after the commit"
    assert [args[:2] for args, _ in recorder.calls] == [
        (0, "nh4_nitrogen"), (0, "no3_nitrogen"), (1, "nh4_nitrogen"), (1, "no3_nitrogen"),
    ]
    first = [kwargs["commit_context"] for args, kwargs in recorder.calls if args[0] == 0]
    assert first[0] == first[1] == {**_CONTEXT, "conservation_checked": False, "element_net_change": {"nitrogen": 0.0}}
    assert first[0] is not first[1]
    second = [kwargs["commit_context"] for args, kwargs in recorder.calls if args[0] == 1]
    assert second[0] == second[1] == {**_CONTEXT, "stage": 2, "role": "second_half", "conservation_checked": True,
                                      "element_net_change": {"nitrogen": 0.0}}


def test_pending_events_carry_the_context_through_checkpoint_and_restore():
    manager = _nitrogen_manager()
    manager.set_provenance_recorder(_KeywordRecorder())
    manager.begin_provenance_transaction()
    manager.apply_updates(_transfer(), dt=600.0, check_conservation=False, commit_context=_CONTEXT)
    snapshot = manager.checkpoint()
    pending = snapshot["pending_provenance_events"]
    assert [event["variable"] for event in pending] == ["nh4_nitrogen", "no3_nitrogen"]
    expected = {**_CONTEXT, "conservation_checked": False, "element_net_change": {"nitrogen": 0.0}}
    assert all(event["commit_context"] == expected for event in pending)
    restored = _nitrogen_manager()
    recorder = _NamedKeywordRecorder()
    restored.set_provenance_recorder(recorder)
    restored.restore(snapshot)
    assert _digest(restored.checkpoint()["pending_provenance_events"]) == _digest(pending)
    restored.commit_provenance_transaction()
    assert recorder.calls == [(0, "nh4_nitrogen", expected), (0, "no3_nitrogen", expected)]


def test_commit_context_reaches_only_recorders_that_accept_it():
    six, named, keyword, contextual = _SixArgumentRecorder(), _NamedKeywordRecorder(), _KeywordRecorder(), _ContextRecorder()
    for recorder in (six, named, keyword, contextual):
        manager = _nitrogen_manager()
        manager.set_provenance_recorder(recorder)
        manager.apply_updates(_transfer(), dt=600.0, check_conservation=False, commit_context=_CONTEXT)
    expected = {**_CONTEXT, "conservation_checked": False, "element_net_change": {"nitrogen": 0.0}}
    assert [len(call) for call in six.calls] == [6, 6]
    assert [call[2] for call in named.calls] == [expected, expected]
    assert [set(kwargs) for _, kwargs in keyword.calls] == [{"commit_context"}, {"commit_context"}]
    assert [set(kwargs) for _, kwargs in contextual.calls] == [{"commit_context", "notification_context"}] * 2
    assert contextual.calls[0][1]["notification_context"] == {"run_id": "pin-run", "output_dir": "pin-output"}


def test_a_recorder_cannot_alter_a_retained_context():
    class _Mutating:
        def __init__(self) -> None:
            self.fail = True

        def record_state_update(self, step, variable, pre_value, post_value, functor_increments, unit, **options):
            options["commit_context"]["role"] = "altered"
            if self.fail:
                raise OSError("delivery not confirmed")

    manager = _nitrogen_manager()
    recorder = _Mutating()
    manager.set_provenance_recorder(recorder)
    with pytest.raises(RuntimeError, match="notification"):
        manager.apply_updates({"nh4_nitrogen": _item(0.5, unit="g N m-2")}, dt=600.0, check_conservation=False,
                              commit_context=_CONTEXT)
    assert manager._pending_provenance_events[0]["commit_context"]["role"] == "middle"


@pytest.mark.parametrize(
    ("context", "error"),
    [
        ([("role", "lie")], TypeError),
        ({1: "lie"}, TypeError),
        ({"": "lie"}, ValueError),
        ({"role": ["lie"]}, TypeError),
        ({"stage": torch.tensor(1)}, TypeError),
        ({"dt": float("nan")}, ValueError),
        ({"dt": float("inf")}, ValueError),
        ({"element_net_change": {}}, ValueError),
        ({"conservation_checked": True}, ValueError),
        ({"element_net_change_error": "x"}, ValueError),
    ],
    ids=["not_a_mapping", "non_string_key", "empty_key", "list_value", "tensor_value", "nan", "inf",
         "reserved_net_change", "reserved_checked", "reserved_error"],
)
def test_an_invalid_commit_context_is_refused_before_any_change(context, error):
    manager = _nitrogen_manager()
    recorder = _KeywordRecorder()
    manager.set_provenance_recorder(recorder)
    before = manager.get_state()
    with pytest.raises(error, match="^commit_context"):
        manager.apply_updates(_transfer(), dt=600.0, check_conservation=False, commit_context=context)
    after = manager.get_state()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert manager.get_diagnostics()["update_counter"] == 0
    assert recorder.calls == [] and manager._pending_provenance_events == []


def test_without_a_recorder_the_context_is_validated_and_no_ledger_is_evaluated(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the element ledger must not be evaluated without a recorder")

    monkeypatch.setattr(state_conservation, "_collect_element_net_change", refuse)
    manager = _nitrogen_manager()
    manager.apply_updates(_transfer(), dt=600.0, check_conservation=False, commit_context=_CONTEXT)
    assert float(manager.get_state("no3_nitrogen")) == 1.25
    with pytest.raises(TypeError, match="^commit_context"):
        manager.apply_updates(_transfer(), dt=600.0, check_conservation=False, commit_context=["role"])
    assert manager.get_diagnostics()["update_counter"] == 1


def test_a_failed_element_ledger_is_recorded_explicitly_and_the_commit_stands(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Conservation unit for 'x' cannot be normalized")

    manager = _nitrogen_manager()
    recorder = _KeywordRecorder()
    manager.set_provenance_recorder(recorder)
    monkeypatch.setattr(state_conservation, "_collect_element_net_change", fail)
    manager.apply_updates(_transfer(), dt=600.0, check_conservation=False, commit_context=_CONTEXT)
    assert float(manager.get_state("no3_nitrogen")) == 1.25
    context = recorder.calls[0][1]["commit_context"]
    assert context["element_net_change"] is None
    assert context["element_net_change_error"] == "ValueError: Conservation unit for 'x' cannot be normalized"


def test_element_net_change_lists_only_touched_elements_and_is_signed():
    manager = _nitrogen_manager()
    recorder = _KeywordRecorder()
    manager.set_provenance_recorder(recorder)
    manager.apply_updates({"probe_temperature": _item(1.0, unit="K")}, check_conservation=False, commit_context=_CONTEXT)
    manager.apply_updates({"no3_nitrogen": _item(-0.75, unit="g N m-2")}, dt=600.0, check_conservation=False,
                          commit_context=_CONTEXT)
    assert recorder.calls[0][1]["commit_context"]["element_net_change"] == {}
    assert recorder.calls[1][1]["commit_context"]["element_net_change"] == {"nitrogen": -0.75}
    assert math.copysign(1.0, recorder.calls[1][1]["commit_context"]["element_net_change"]["nitrogen"]) == -1.0


# --------------------------------------------------------------------------
# D. The native ProvenanceRecorder stores the context in its runtime event.
# --------------------------------------------------------------------------


def _recorder_event() -> Dict[str, Any]:
    return dict(step=4, variable="no3_nitrogen", pre_value=torch.tensor([1.0, 2.0], dtype=torch.float64),
                post_value=torch.tensor([0.5, 2.5], dtype=torch.float64),
                functor_increments={"denitrification": torch.tensor([-0.5, 0.5], dtype=torch.float64)},
                unit="g N m-2")


def _runtime_records(path: Path) -> List[Dict[str, Any]]:
    import json

    return [json.loads(line) for line in (path / "runtime.log.jsonl").read_text(encoding="ascii").splitlines()]


_RECORDED_CONTEXT = {**_CONTEXT, "conservation_checked": False,
                     "element_net_change": {"nitrogen": -0.25, "water": float("nan")}}


def test_native_recorder_stores_the_commit_context_in_its_runtime_event(tmp_path):
    from relic.core.provenance.recorder import ProvenanceRecorder

    plain, contextual = tmp_path / "plain", tmp_path / "contextual"
    ProvenanceRecorder(plain, run_id="run-a").record_state_update(**_recorder_event())
    ProvenanceRecorder(contextual, run_id="run-a").record_state_update(**_recorder_event(), commit_context=_RECORDED_CONTEXT)
    without, with_context = _runtime_records(plain)[0], _runtime_records(contextual)[0]
    assert "commit_context" not in without
    assert with_context.pop("commit_context") == {**_CONTEXT, "conservation_checked": False,
                                                  "element_net_change": {"nitrogen": -0.25, "water": None}}
    without.pop("timestamp_utc"), with_context.pop("timestamp_utc")
    assert with_context == without
    # The PROV document and the content witness are those of the values only.
    assert (plain / "prov.jsonld").read_bytes() == (contextual / "prov.jsonld").read_bytes()


def test_native_recorder_replays_a_context_idempotently_and_refuses_a_changed_one(tmp_path):
    from relic.core.provenance.recorder import ProvenanceRecorder

    recorder = ProvenanceRecorder(tmp_path, run_id="run-a")
    recorder.record_state_update(**_recorder_event(), commit_context=_RECORDED_CONTEXT)
    before = (tmp_path / "runtime.log.jsonl").read_bytes()
    recorder.record_state_update(**_recorder_event(), commit_context=dict(_RECORDED_CONTEXT))
    assert (tmp_path / "runtime.log.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="conflict"):
        recorder.record_state_update(**_recorder_event(), commit_context=dict(_RECORDED_CONTEXT, role="first_half"))
    assert (tmp_path / "runtime.log.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "context",
    [["role"], {"": 1}, {1: 1}, {"role": object()}],
    ids=["not_a_mapping", "empty_key", "non_string_key", "not_json"],
)
def test_native_recorder_refuses_a_malformed_commit_context_without_writing(tmp_path, context):
    from relic.core.provenance.recorder import ProvenanceRecorder

    with pytest.raises(ValueError, match="commit context"):
        ProvenanceRecorder(tmp_path, run_id="run-a").record_state_update(**_recorder_event(), commit_context=context)
    assert not (tmp_path / "runtime.log.jsonl").exists() and not (tmp_path / "prov.jsonld").exists()


def test_state_manager_commit_context_reaches_the_native_recorder(tmp_path):
    from relic.core.provenance.recorder import ProvenanceRecorder

    manager = _nitrogen_manager()
    manager.set_provenance_recorder(ProvenanceRecorder(tmp_path, run_id="run-a"))
    manager.apply_updates(_transfer(), dt=600.0, check_conservation=False, commit_context=_CONTEXT)
    records = _runtime_records(tmp_path)
    assert [record["variable"] for record in records] == ["nh4_nitrogen", "no3_nitrogen"]
    expected = {**_CONTEXT, "conservation_checked": False, "element_net_change": {"nitrogen": 0.0}}
    assert all(record["commit_context"] == expected for record in records)


# --------------------------------------------------------------------------
# E. The checkpoint grammar of pending events accepts a recorded commit context.
# --------------------------------------------------------------------------


def _pending_with_context(monkeypatch=None, fail_ledger: bool = False) -> List[Dict[str, Any]]:
    manager = _nitrogen_manager()
    manager.set_provenance_recorder(_KeywordRecorder())
    if fail_ledger:
        def fail(*args, **kwargs):
            raise ValueError("ledger unavailable")

        monkeypatch.setattr(state_conservation, "_collect_element_net_change", fail)
    manager.begin_provenance_transaction()
    manager.apply_updates(_transfer(), dt=600.0, check_conservation=False, commit_context=_CONTEXT)
    return manager.checkpoint()["pending_provenance_events"]


def test_checkpoint_grammar_accepts_pending_events_with_a_commit_context(monkeypatch):
    from relic.core.simulator import checkpoint as checkpoint_module

    checkpoint_module._validate_provenance_notification_events(_pending_with_context(), require_notification_context=False)
    failed = _pending_with_context(monkeypatch, fail_ledger=True)
    assert failed[0]["commit_context"]["element_net_change"] is None
    checkpoint_module._validate_provenance_notification_events(failed, require_notification_context=False)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda c: ["role"], TypeError),
        (lambda c: {**c, "role": ["lie"]}, TypeError),
        (lambda c: {key: value for key, value in c.items() if key != "conservation_checked"}, ValueError),
        (lambda c: {key: value for key, value in c.items() if key != "element_net_change"}, ValueError),
        (lambda c: {**c, "element_net_change": {"nitrogen": 1}}, ValueError),
        (lambda c: {**c, "element_net_change": None}, ValueError),
        (lambda c: {**c, "element_net_change_error": "x"}, ValueError),
    ],
    ids=["not_a_mapping", "list_value", "no_checked_flag", "no_net_change", "integer_net_change",
         "none_without_error", "error_with_value"],
)
def test_checkpoint_grammar_refuses_a_malformed_recorded_commit_context(change, error):
    from relic.core.simulator import checkpoint as checkpoint_module

    events = _pending_with_context()
    events[0]["commit_context"] = change(events[0]["commit_context"])
    with pytest.raises(error, match="commit.context"):
        checkpoint_module._validate_provenance_notification_events(events, require_notification_context=False)


def test_checkpoint_grammar_still_refuses_unknown_event_fields():
    from relic.core.simulator import checkpoint as checkpoint_module

    events = _pending_with_context()
    events[0]["other"] = 1
    with pytest.raises(ValueError, match="fields are invalid"):
        checkpoint_module._validate_provenance_notification_events(events, require_notification_context=False)
    del events[0]["other"]
    del events[0]["unit"]
    with pytest.raises(ValueError, match="fields are invalid"):
        checkpoint_module._validate_provenance_notification_events(events, require_notification_context=False)


# --------------------------------------------------------------------------
# 2026-10-20 canonical target-unit event metadata, outside numerical ledgers.
# --------------------------------------------------------------------------


def _canonical_units_and_historical_view(calls, manager, *, with_keywords):
    """Assert real units first; project only unit metadata for historical hashes.

    The returned view is for these old comparison tests only. It never changes
    the recorder calls, native log, pending event, production writer or tensors.
    All old SHA witnesses still check order, values, sources and six arguments.
    """
    historical = []
    for call in calls:
        args, options = call if with_keywords else (call, None)
        assert len(args) == 6
        variable = args[1]
        schema = manager._schema.get(variable)
        contract = manager._contract_registry.get_contract(variable)
        expected = (schema.units if schema is not None else contract.udunits if contract is not None else "unknown")
        assert args[5] == expected, (variable, args[5], expected)
        old_unit = schema.units if schema is not None else "unknown"
        projected = (*args[:5], old_unit)
        historical.append((projected, options) if with_keywords else projected)
    return historical


def _mixed_canonical_unit_updates():
    # References: BIPM SI Brochure (unit conversion); W3C PROV-DM unit metadata.
    # Mathematical expression: Delta NO3 + (F_N2+F_N2O)*dt = -0.3+(0.02+0.01)*10 = 0.
    # Parameter meanings: dt=10 s; all values are manufactured StateManager fixtures.
    # Units: mineral storage g N m-2, gas rates g N m-2 s-1, temperature K/degC.
    # Domain/constraints: no new process or unit resolver; production normalization owns conversions.
    # Sign convention: outward gases positive, nitrate storage loss negative.
    return {
        "n2_production": _item(0.02, mode=UpdateMode.REPLACE, unit="g N m-2 s-1", per_timestep=False, source="denitrification"),
        "n2o_flux": _item(0.01, unit="g N m-2 s-1", per_timestep=False, source="denitrification"),
        "no3_nitrogen": _item(-0.3, unit="g N m-2", source="denitrification"),
        "probe_temperature": _item(20.0, mode=UpdateMode.REPLACE, unit="degC"),
        "audit_scalar": _item(1.5, mode=UpdateMode.REPLACE, unit="1"),
        "unknown_target": _item(2.0, mode=UpdateMode.REPLACE, unit="K"),
    }


@pytest.mark.parametrize("reverse", [False, True], ids=["forward_keys", "reverse_keys"])
def test_actual_event_units_use_each_targets_canonical_contract_after_conversion(reverse):
    manager = _nitrogen_manager()
    manager.register_variable(StateVariableSpec(name="audit_scalar", units="dimensionless", required=False))
    recorder = _KeywordRecorder()
    manager.set_provenance_recorder(recorder)
    updates = _mixed_canonical_unit_updates()
    if reverse:
        updates = dict(reversed(list(updates.items())))
    manager.apply_updates(updates, dt=10.0, check_conservation=False,
                          commit_context={**_CONTEXT, "dt": 10.0, "composition_interval": 10.0})
    actual = {args[1]: args[5] for args, _ in recorder.calls}
    assert actual == {
        "n2_production": "g N m-2 s-1", "n2o_flux": "g N m-2 s-1",
        "no3_nitrogen": "g N m-2", "probe_temperature": "K",
        "audit_scalar": "dimensionless", "unknown_target": "unknown",
    }
    assert len(recorder.calls) == len(updates)
    assert all(args[0] == 0 for args, _ in recorder.calls)
    assert "n2_production" not in manager._schema and "n2o_flux" not in manager._schema
    assert float(manager.get_state("probe_temperature")) == pytest.approx(293.15)
    assert float(manager.get_state("audit_scalar")) == 1.5
    assert float(manager.get_state("unknown_target")) == 2.0
    assert float(manager.get_state("no3_nitrogen")) == pytest.approx(0.7)


def test_declared_units_survive_pending_restore_and_native_delivery_exactly_once(tmp_path):
    from relic.core.provenance.recorder import ProvenanceRecorder

    destination = tmp_path / "canonical_unit_events"
    manager = _nitrogen_manager()
    manager.set_provenance_recorder(ProvenanceRecorder(destination, run_id="unit-metadata"))
    manager.begin_provenance_transaction()
    updates = {key: value for key, value in _mixed_canonical_unit_updates().items()
               if key in {"n2_production", "n2o_flux", "no3_nitrogen"}}
    manager.apply_updates(updates, dt=10.0, check_conservation=False,
                          commit_context={**_CONTEXT, "dt": 10.0, "composition_interval": 10.0})
    snapshot = manager.checkpoint()
    expected = {"n2_production": "g N m-2 s-1", "n2o_flux": "g N m-2 s-1", "no3_nitrogen": "g N m-2"}
    assert {event["variable"]: event["unit"] for event in snapshot["pending_provenance_events"]} == expected
    restored = _nitrogen_manager()
    restored.set_provenance_recorder(ProvenanceRecorder(destination, run_id="unit-metadata"))
    restored.restore(snapshot)
    assert _digest(restored.checkpoint()["pending_provenance_events"]) == _digest(snapshot["pending_provenance_events"])
    restored.commit_provenance_transaction()
    records = _runtime_records(destination)
    assert len(records) == 3
    assert {row["variable"]: row["unit"] for row in records} == expected
    before = (destination / "runtime.log.jsonl").read_bytes()
    restored.commit_provenance_transaction()
    assert (destination / "runtime.log.jsonl").read_bytes() == before


def test_scm_variable_alias_remains_refused_before_any_event_or_state_change():
    manager = _nitrogen_manager()
    recorder = _KeywordRecorder()
    manager.set_provenance_recorder(recorder)
    before = _digest(manager.get_state())
    with pytest.raises(PermissionError, match="alias"):
        manager.apply_updates(
            {"daily_maximum_air_temperature": _item(295.0, mode=UpdateMode.REPLACE, unit="K")},
            check_conservation=False, commit_context=_CONTEXT,
        )
    assert _digest(manager.get_state()) == before
    assert recorder.calls == [] and manager.get_diagnostics()["update_counter"] == 0
