"""CR-A-04 T7: multirate exchange between the fast group and a slower consumer group.

Design 20261019_cr_a04_design_01, section 3 ("Exchange interval_mean"), T7:

- hold: the consumer group reads the committed value of v at its call (zero-order
  hold), in both directions (a slower consumer of a fast producer, a fast consumer
  of a slower producer). Pinned without exchanges; a declared hold exchange runs
  bitwise like the undeclared hold.
- interval_mean: after every accepted fast composition (after the boundary-rate
  and S2 finalization) one owner commit per exchange advances the running mean
      m_e <- (m_e tau + vbar h) / (tau + h),  tau = t_k - t_last(consumer group),
  and the consumer group call sees v := m_e (an overlay of its input state only;
  v itself is not changed). m_e is the exact interval mean of the piecewise-constant
  reconstruction of the accepted fast values over [t_last, t_k + h].
- m_e is a StateManager variable (schema and state) written by the owner identity
  exchange:<id>, so a failed step or a rejected adaptive attempt leaves it unchanged.
- Admissibility: an interval_mean exchange of a variable written by its consumer
  group is refused, and so is a Functor writing the mean variable; interpolation is
  refused by the configuration validator (pinned by splitting_admissibility.test.py
  test_interpolating_multirate_exchange_is_refused and the restore case of
  splitting_checkpoint_restore.test.py; not duplicated here).

References: Gear and Wells (1984) BIT 24(4):484-502 (multirate methods); Strang
(1968) SIAM J. Numer. Anal. 5(3):506-517; Hairer, Lubich and Wanner (2006) Geometric
Numerical Integration, 2nd ed., Sect. II.5.
Mathematical expression (fixtures): fast ramp v(t, h) = (t + h) / 60 (the value at the
end of each fast call); medium source w(t, H) = (t + H) / 240; readers copy what they
see. Block means over [t_last, t_last + T]: sum_k v_k h_k / T.
Parameter meanings: t the start and h the length of a call; T the consumer interval.
Units: all fixture variables dimensionless; t, h, T [s]; the probe rate g N m-2 s-1.
Domain/constraints: CPU, float64 test default dtype (test/conftest.py); medium
interval 240 s, slow interval 480 s, fast step 60 s unless stated.
Sign convention: none (no fluxes in the ramp fixture).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from relic.core.functor import BiologicalProcessFunctor, FunctorMetadata, TemporalScale, VariableKind
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode

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
_T4 = _load_test_module("core/splitting_strang_replace_rates.test.py", "_cr_a04_t4_strang_rates")

_DT = 60.0  # s, fast step
_MEDIUM = 240.0  # s, medium interval (4 fast steps)
_SLOW = 480.0  # s, slow interval (8 fast steps)
_VARIABLES = ("ramp", "seen_medium", "seen_slow", "held", "seen_fast")
_UNITS = {name: "dimensionless" for name in _VARIABLES}
_INITIAL = {name: 0.0 for name in _VARIABLES}
_MEDIUM_MEAN = {"id": "ramp_medium", "variable": "ramp", "consumer_group": "medium", "mode": "interval_mean"}
_SLOW_MEAN = {"id": "ramp_slow", "variable": "ramp", "consumer_group": "slow", "mode": "interval_mean"}


def _mean_name(exchange_id: str) -> str:
    """StateManager name of the mean of one exchange (naming contract of the owner)."""
    return f"exchange_mean__{exchange_id}"


def _owner(exchange_id: str) -> str:
    """Owner identity that writes the mean of one exchange."""
    return f"exchange:{exchange_id}"


class _Probe(BiologicalProcessFunctor):
    """Test Functor writing one variable as a REPLACE value; optionally reads others.

    ``value`` maps (state, time, dt) to the written value; every call is logged as
    (name, time, dt, {read variable: value seen}). ``fail_at`` names a call number
    (1-based) that raises, to exercise the step rollback.
    """

    def __init__(
        self,
        name: str,
        output: str,
        value: Callable[[Dict[str, Any], float, float], float],
        *,
        group: str,
        reads: Sequence[str] = (),
        log: Optional[List[tuple]] = None,
        fail_at: Optional[int] = None,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            description=f"Multirate exchange test fixture {name}",
            references=["Gear and Wells (1984) BIT 24(4):484-502"],
            temporal_scale=TemporalScale.SECOND,
            input_variables=list(reads),
            output_variables=[output],
            variable_kind=VariableKind.STATE,
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.execution_group = group
        self.output = output
        self.value = value
        self.reads = tuple(reads)
        self.log = log if log is not None else []
        self.fail_at = fail_at
        self.calls = 0

    def get_dependencies(self) -> Dict[str, List[str]]:
        return {"required": [], "optional": [], "lagged": []}

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, UpdateItem]:
        # References: test fixture (no scientific law).
        # Mathematical expression: y = value(state, t, h); see the module docstring.
        # Parameter meanings: t the start, h the length of this call.
        # Units: y dimensionless; t, h [s].
        # Domain/constraints: h > 0.
        # Sign convention: none.
        self.calls += 1
        seen = {name: torch.as_tensor(state[name]).detach().clone() for name in self.reads}
        self.log.append((self.name, float(time), float(dt.item()), seen))
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError(f"{self.name}: injected failure at call {self.calls}")
        return {
            self.output: UpdateItem(
                value=torch.as_tensor(self.value(state, float(time), float(dt.item())), dtype=torch.float64),
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                source_functor=self.name,
            )
        }


def _ramp(state: Dict[str, Any], time: float, dt: float) -> float:
    return (time + dt) / 60.0


def _held(state: Dict[str, Any], time: float, dt: float) -> float:
    return (time + dt) / 240.0


def _copy(variable: str) -> Callable[[Dict[str, Any], float, float], Any]:
    return lambda state, time, dt: torch.as_tensor(state[variable]).detach().clone()


def _ramp_simulator(
    exchanges: Sequence[Dict[str, str]] = (),
    *,
    log: Optional[List[tuple]] = None,
    register_first: bool = False,
    medium_fail_at: Optional[int] = None,
    **config: Any,
) -> RELICSimulator:
    """Fast ramp and fast hold reader; medium reader and source; slow reader."""
    settings: Dict[str, Any] = {
        "time_step": _DT,
        "device": "cpu",
        "medium_interval_seconds": _MEDIUM,
        "slow_interval_seconds": _SLOW,
        "state_units": dict(_UNITS),
    }
    if exchanges:
        settings["multirate_exchanges"] = [dict(item) for item in exchanges]
    settings.update(config)
    simulator = RELICSimulator(settings)
    log = log if log is not None else []
    functors = (
        (_Probe("fast_ramp", "ramp", _ramp, group="fast", log=log), "fast"),
        (_Probe("fast_hold_reader", "seen_fast", _copy("held"), group="fast", reads=("held",), log=log), "fast"),
        (
            _Probe(
                "medium_reader", "seen_medium", _copy("ramp"), group="medium", reads=("ramp",), log=log,
                fail_at=medium_fail_at,
            ),
            "medium",
        ),
        (_Probe("medium_source", "held", _held, group="medium", log=log), "medium"),
        (_Probe("slow_reader", "seen_slow", _copy("ramp"), group="slow", reads=("ramp",), log=log), "slow"),
    )
    if not register_first:
        simulator.initialize_state(dict(_INITIAL))
    for functor, group in functors:
        simulator.register_functor(functor, execution_group=group)
    if register_first:
        simulator.initialize_state(dict(_INITIAL))
    return simulator


def _seen(log: List[tuple], name: str, variable: str) -> List[Tuple[float, float, float]]:
    return [(time, dt, float(seen[variable])) for who, time, dt, seen in log if who == name]


def _row(simulator: RELICSimulator, extra: Sequence[str] = ()) -> Dict[str, torch.Tensor]:
    state = simulator.state_manager.get_state()
    return {name: state[name].clone() for name in tuple(_VARIABLES) + tuple(extra)}


def _run(simulator: RELICSimulator, steps: int, dt: float = _DT, extra: Sequence[str] = ()) -> List[Dict[str, torch.Tensor]]:
    rows = []
    for _ in range(steps):
        simulator.step(dt=dt)
        rows.append(_row(simulator, extra))
    return rows


def _assert_rows_equal(left: List[Dict[str, torch.Tensor]], right: List[Dict[str, torch.Tensor]]) -> None:
    assert len(left) == len(right)
    for index, (a, b) in enumerate(zip(left, right)):
        assert set(a) == set(b), index
        for key in a:
            assert torch.equal(a[key], b[key]), (index, key, a[key], b[key])


def _assert_logs_equal(left: List[tuple], right: List[tuple]) -> None:
    assert [(who, time, dt) for who, time, dt, _ in left] == [(who, time, dt) for who, time, dt, _ in right]
    for (_, _, _, a), (_, _, _, b) in zip(left, right):
        assert set(a) == set(b)
        for key in a:
            assert torch.equal(a[key], b[key]), key


# --------------------------------------------------------------------------
# hold, both directions (pinned without exchanges; declared hold is the same).
# --------------------------------------------------------------------------


def test_t7_hold_in_both_directions_without_exchanges_is_pinned():
    """Slower consumers see the last committed fast value; the fast reader sees the held medium value."""
    log: List[tuple] = []
    simulator = _ramp_simulator(log=log)
    rows = _run(simulator, 9)
    # Medium calls over [0, 240] and [240, 480] see the ramp of the last fast step.
    assert _seen(log, "medium_reader", "ramp") == [(0.0, 240.0, 4.0), (240.0, 240.0, 8.0)]
    # The slow call over [0, 480] sees the ramp of step 8.
    assert _seen(log, "slow_reader", "ramp") == [(0.0, 480.0, 8.0)]
    # The fast reader holds the medium value between medium calls (Lie across groups:
    # fast before medium inside a step).
    assert [value for _, _, value in _seen(log, "fast_hold_reader", "held")] == [0.0] * 4 + [1.0] * 4 + [2.0]
    assert [float(row["ramp"]) for row in rows] == [float(k) for k in range(1, 10)]
    assert [float(row["held"]) for row in rows] == [0.0] * 3 + [1.0] * 4 + [2.0] * 2


def test_t7_declared_hold_exchanges_run_bitwise_like_the_undeclared_hold():
    hold = [
        dict(_MEDIUM_MEAN, id="ramp_medium_hold", mode="hold"),
        dict(_SLOW_MEAN, id="ramp_slow_hold", mode="hold"),
    ]
    undeclared_log: List[tuple] = []
    declared_log: List[tuple] = []
    undeclared = _ramp_simulator(log=undeclared_log)
    declared = _ramp_simulator(hold, log=declared_log)
    _assert_rows_equal(_run(undeclared, 10), _run(declared, 10))
    _assert_logs_equal(undeclared_log, declared_log)
    # A hold exchange keeps no mean.
    assert not any(name.startswith("exchange_mean__") for name in declared.state_manager.get_state())
    assert not any(name.startswith("exchange_mean__") for name in declared.state_manager.schema)


# --------------------------------------------------------------------------
# The overlay of the consumer group's input state (group execution owner).
# --------------------------------------------------------------------------


def test_t7_group_input_overlay_is_seen_by_one_group_call_only():
    log: List[tuple] = []
    simulator = _ramp_simulator(log=log)
    simulator.step(dt=_DT)
    committed = simulator.state_manager.get_state("ramp").clone()
    overlay_value = torch.tensor(42.0, dtype=torch.float64)
    simulator._active_exchange_overlay = {"ramp": overlay_value}
    try:
        simulator._execute_group(["medium_reader"], _MEDIUM, start_time=0.0)
    finally:
        simulator._active_exchange_overlay = None
    assert _seen(log, "medium_reader", "ramp")[-1] == (0.0, _MEDIUM, 42.0)
    state = simulator.state_manager.get_state()
    assert torch.equal(state["ramp"], committed)
    assert float(state["seen_medium"]) == 42.0
    # Without an overlay the next call reads the committed value again.
    simulator._execute_group(["medium_reader"], _MEDIUM, start_time=0.0)
    assert _seen(log, "medium_reader", "ramp")[-1] == (0.0, _MEDIUM, float(committed))
    # An overlay of a variable that is not part of the committed state is refused.
    simulator._active_exchange_overlay = {"not_a_state_variable": overlay_value}
    try:
        with pytest.raises(RuntimeError, match="not_a_state_variable"):
            simulator._execute_group(["medium_reader"], _MEDIUM, start_time=0.0)
    finally:
        simulator._active_exchange_overlay = None


# --------------------------------------------------------------------------
# interval_mean: the exact block mean of the accepted fast values.
# --------------------------------------------------------------------------


def test_t7_interval_mean_gives_each_consumer_group_the_exact_block_mean():
    log: List[tuple] = []
    simulator = _ramp_simulator([_MEDIUM_MEAN, _SLOW_MEAN], log=log)
    rows = _run(simulator, 9, extra=(_mean_name("ramp_medium"), _mean_name("ramp_slow")))
    # mean(1, 2, 3, 4) = 2.5 and mean(5, 6, 7, 8) = 6.5 (exact in binary64).
    assert _seen(log, "medium_reader", "ramp") == [(0.0, 240.0, 2.5), (240.0, 240.0, 6.5)]
    # mean(1, ..., 8) = 4.5 over the slow interval.
    assert _seen(log, "slow_reader", "ramp") == [(0.0, 480.0, 4.5)]
    # The overlay never changes the committed ramp; the readers store what they saw.
    assert [float(row["ramp"]) for row in rows] == [float(k) for k in range(1, 10)]
    assert float(rows[3]["seen_medium"]) == 2.5 and float(rows[7]["seen_medium"]) == 6.5
    assert float(rows[7]["seen_slow"]) == 4.5
    # The running means after each accepted step (tau = 0 restarts a block).
    assert [float(row[_mean_name("ramp_medium")]) for row in rows] == [1.0, 1.5, 2.0, 2.5, 5.0, 5.5, 6.0, 6.5, 9.0]
    assert [float(row[_mean_name("ramp_slow")]) for row in rows] == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 9.0]
    # The opposite direction (fast reader of the medium source) keeps its hold.
    assert [value for _, _, value in _seen(log, "fast_hold_reader", "held")] == [0.0] * 4 + [1.0] * 4 + [2.0]


def test_t7_interval_mean_weights_unequal_steps_by_their_length():
    """sum_k vbar_k h_k / T with h = (30, 90, 60, 60) s over T = 240 s."""
    log: List[tuple] = []
    simulator = _ramp_simulator([_MEDIUM_MEAN], log=log)
    steps = (30.0, 90.0, 60.0, 60.0)
    values: List[float] = []
    for dt in steps:
        simulator.step(dt=dt)
        values.append(float(simulator.state_manager.get_state("ramp")))
    assert values == [0.5, 2.0, 3.0, 4.0]
    exact = math.fsum(value * dt for value, dt in zip(values, steps)) / math.fsum(steps)
    ((time, dt, seen),) = _seen(log, "medium_reader", "ramp")
    assert (time, dt) == (0.0, 240.0)
    assert seen == pytest.approx(exact, rel=4.0e-16, abs=0.0)
    assert seen != values[-1]


def test_t7_interval_mean_with_one_step_blocks_is_bitwise_the_hold():
    """K = 1: the consumer runs after every fast step, so the mean is the last value."""
    hold_log: List[tuple] = []
    mean_log: List[tuple] = []
    hold = _ramp_simulator(log=hold_log, medium_interval_seconds=_DT)
    mean = _ramp_simulator([_MEDIUM_MEAN], log=mean_log, medium_interval_seconds=_DT)
    _assert_rows_equal(_run(hold, 6), _run(mean, 6))
    _assert_logs_equal(hold_log, mean_log)


def test_t7_interval_mean_sees_the_strang_step_mean_of_a_replace_rate():
    """The exchange commit follows the boundary-rate and S2 finalization of the fast composition."""
    exchange = {"id": "n2_medium", "variable": "n2_production", "consumer_group": "medium", "mode": "interval_mean"}
    log: List[tuple] = []
    simulator, probe = _T4._probe_simulator(
        "strang",
        medium_interval_seconds=600.0,
        multirate_exchanges=[exchange],
    )
    reader = _Probe(
        "n2_reader", "seen_n2", _copy("n2_production"), group="medium", reads=("n2_production",), log=log,
    )
    simulator.register_functor(reader, execution_group="medium")
    simulator.step(dt=600.0)
    q1 = torch.tensor(probe.calls[0][2], dtype=torch.float64)
    q2 = torch.tensor(probe.calls[1][2], dtype=torch.float64)
    mean = q1 * torch.tensor(0.5, dtype=torch.float64) + q2 * torch.tensor(0.5, dtype=torch.float64)
    ((who, time, dt, seen),) = log
    assert (who, time, dt) == ("n2_reader", 0.0, 600.0)
    assert torch.equal(seen["n2_production"].reshape(()), mean)
    assert not torch.equal(mean, q2)
    assert torch.equal(simulator.state_manager.get_state(_mean_name("n2_medium")).reshape(()), mean)


# --------------------------------------------------------------------------
# The mean is a StateManager variable of its owner; transactions restore it.
# --------------------------------------------------------------------------


class _EventRecorder:
    """Provenance recorder capturing the StateManager commit events."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def record_state_update(self, step, variable, pre_value, post_value, functor_increments, unit, **options):
        self.events.append({
            "commit": int(step),
            "variable": str(variable),
            "post": torch.as_tensor(post_value).detach().clone(),
            "sources": tuple(sorted(functor_increments)),
            "unit": unit,
        })


def test_t7_exchange_mean_is_a_state_manager_variable_written_by_its_owner():
    simulator = _ramp_simulator([_MEDIUM_MEAN])
    name = _mean_name("ramp_medium")
    spec = simulator.state_manager.schema[name]
    assert spec.units == "dimensionless" and spec.conserved is False and spec.bounds is None
    recorder = _EventRecorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    _run(simulator, 4)
    stored = simulator.state_manager.get_state(name)
    assert stored.dtype == torch.float64 and float(stored) == 2.5
    mean_events = [event for event in recorder.events if event["variable"] == name]
    assert [float(event["post"]) for event in mean_events] == [1.0, 1.5, 2.0, 2.5]
    assert {event["sources"] for event in mean_events} == {(_owner("ramp_medium"),)}
    # Commit order in the step that runs the medium group: fast group, then the
    # exchange commit, then the medium group commit.
    last_step = [event for event in recorder.events if event["commit"] >= mean_events[-1]["commit"] - 1]
    fast_commit = max(event["commit"] for event in recorder.events if event["variable"] == "ramp")
    medium_commit = max(event["commit"] for event in recorder.events if event["variable"] == "seen_medium")
    assert fast_commit < mean_events[-1]["commit"] < medium_commit, last_step
    # No overlay outlives the consumer group call.
    assert getattr(simulator, "_active_exchange_overlay", None) is None


def test_t7_failed_step_rolls_back_the_exchange_mean():
    log: List[tuple] = []
    simulator = _ramp_simulator([_MEDIUM_MEAN], log=log, medium_fail_at=2)
    name = _mean_name("ramp_medium")
    _run(simulator, 7)
    before = {key: value.clone() for key, value in simulator.state_manager.get_state().items()}
    last_medium = simulator._last_medium_execution_time
    assert float(before[name]) == 6.0 and last_medium == 240.0
    # The governed call wraps the kernel error (FunctorBase ExecutionError).
    with pytest.raises(Exception, match="injected failure"):
        simulator.step(dt=_DT)
    after = simulator.state_manager.get_state()
    assert set(after) == set(before)
    for key, value in before.items():
        assert torch.equal(after[key], value), key
    assert simulator._last_medium_execution_time == last_medium
    assert getattr(simulator, "_active_exchange_overlay", None) is None
    # The repeated step completes the block with the same mean.
    simulator.functors["medium_reader"].fail_at = None
    simulator.step(dt=_DT)
    assert _seen(log, "medium_reader", "ramp")[-1] == (240.0, 240.0, 6.5)


def test_t7_rejected_adaptive_attempt_leaves_no_contribution_in_the_mean():
    log: List[tuple] = []
    simulator = _ramp_simulator(
        [_MEDIUM_MEAN],
        log=log,
        adaptive_dt_enabled=True,
        adaptive_atol=1.0e-9,
        adaptive_rtol=0.0,
        adaptive_dt_min=1.0,
        adaptive_monitor_vars=["ramp"],
    )
    name = _mean_name("ramp_medium")
    simulator.step(dt=_DT)
    simulator.step(dt=_DT)
    ramp_calls = [(time, dt) for who, time, dt, _ in log if who == "fast_ramp"]
    # Each step: a rejected attempt over 60 s, then the accepted retry over 6 s.
    assert ramp_calls == [(0.0, 60.0), (0.0, 6.0), (6.0, 60.0), (6.0, 6.0)]
    assert simulator.current_time == 12.0
    first, second = 6.0 / 60.0, 12.0 / 60.0
    expected = first + ((second - first) * 6.0) / (6.0 + 6.0)
    assert float(simulator.state_manager.get_state(name)) == pytest.approx(expected, rel=1.0e-15, abs=0.0)


# --------------------------------------------------------------------------
# Admissibility at publication and the report.
# --------------------------------------------------------------------------


def test_t7_interval_mean_of_a_variable_written_by_its_consumer_group_is_refused():
    """medium_source writes held in the medium group, so a medium mean of held is refused."""
    exchange = {"id": "held_medium", "variable": "held", "consumer_group": "medium", "mode": "interval_mean"}
    simulator = RELICSimulator({
        "time_step": _DT, "device": "cpu", "medium_interval_seconds": _MEDIUM,
        "state_units": dict(_UNITS), "multirate_exchanges": [exchange],
    })
    simulator.initialize_state(dict(_INITIAL))
    simulator.register_functor(_Probe("fast_ramp", "ramp", _ramp, group="fast"), execution_group="fast")
    before = _T1._view_record(simulator)
    with pytest.raises(RuntimeError, match=r"held_medium.*written by.*medium_source"):
        simulator.register_functor(_Probe("medium_source", "held", _held, group="medium"), execution_group="medium")
    _T1._assert_publication_rolled_back(simulator, before)
    # A hold exchange of the same variable is only flagged.
    hold = RELICSimulator({
        "time_step": _DT, "device": "cpu", "medium_interval_seconds": _MEDIUM,
        "state_units": dict(_UNITS), "multirate_exchanges": [dict(exchange, mode="hold")],
    })
    hold.initialize_state(dict(_INITIAL))
    hold.register_functor(_Probe("medium_source", "held", _held, group="medium"), execution_group="medium")
    (record,) = hold.get_splitting_order_report()["exchanges"]
    assert record["status"] == "flagged" and record["consumer_writers"] == ("medium_source",)


def test_t7_a_functor_writing_the_mean_variable_is_refused():
    simulator = RELICSimulator({
        "time_step": _DT, "device": "cpu", "medium_interval_seconds": _MEDIUM,
        "state_units": dict(_UNITS), "multirate_exchanges": [dict(_MEDIUM_MEAN)],
    })
    simulator.initialize_state(dict(_INITIAL))
    simulator.register_functor(_Probe("fast_ramp", "ramp", _ramp, group="fast"), execution_group="fast")
    before = _T1._view_record(simulator)
    intruder = _Probe("intruder", _mean_name("ramp_medium"), _ramp, group="fast")
    with pytest.raises(RuntimeError, match=r"ramp_medium.*intruder"):
        simulator.register_functor(intruder, execution_group="fast")
    _T1._assert_publication_rolled_back(simulator, before)


def test_t7_exchange_variable_without_declared_units_is_refused():
    exchange = {"id": "nowhere_medium", "variable": "nowhere", "consumer_group": "medium", "mode": "interval_mean"}
    simulator = RELICSimulator({
        "time_step": _DT, "device": "cpu", "state_units": dict(_UNITS), "multirate_exchanges": [exchange],
    })
    with pytest.raises(ValueError, match=r"nowhere_medium.*units"):
        simulator.initialize_state(dict(_INITIAL))


def test_t7_report_lists_the_exchange_records():
    simulator = _ramp_simulator([_MEDIUM_MEAN, dict(_SLOW_MEAN, mode="hold")])
    records = {record["id"]: record for record in simulator.get_splitting_order_report()["exchanges"]}
    assert set(records) == {"ramp_medium", "ramp_slow"}
    medium, slow = records["ramp_medium"], records["ramp_slow"]
    assert medium["mode"] == "interval_mean" and medium["status"] == "admitted"
    assert medium["mean_variable"] == _mean_name("ramp_medium") and medium["owner"] == _owner("ramp_medium")
    assert medium["readers"] == ("medium_reader",) and medium["fast_writers"] == ("fast_ramp",)
    assert medium["consumer_writers"] == ()
    assert slow["mode"] == "hold" and slow["mean_variable"] is None and slow["readers"] == ("slow_reader",)
