"""Unit tests for time-space architecture enhancements (E1-E6)."""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, Optional

import pytest
import torch

import relic.core.simulator.time_ops as time_ops_module
from relic.core.functor import BiologicalProcessFunctor, FunctorMetadata, TemporalScale
from relic.core.simulator import RELICSimulator
from relic.core.state import ExternalIntent, StateWriteMode, UpdateItem, UpdateMode
from relic.core.state.types import StateVariableSpec
from relic.utils.time_constants import SECONDS_PER_DAY, SECONDS_PER_MINUTE


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'osc_u': 'dimensionless',
    'osc_v': 'dimensionless',
    'shared_x': 'dimensionless',
    'shared_y': 'dimensionless',
    'shared_z': 'dimensionless',
    'test_signal': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class RecordingFunctor(BiologicalProcessFunctor):
    """Minimal functor that records execution timings and dt values."""

    def __init__(
        self,
        name: str,
        execution_group: str,
        update_var: str,
        update_value: float = 0.0,
        input_vars: Optional[List[str]] = None,
        substep_count: int = 1,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            description=f"Recorder functor {name}",
            references=["RELIC simulator execution test fixture"],
            temporal_scale=TemporalScale.SECOND,
            input_variables=list(input_vars or []),
            output_variables=[update_var],
            substep_count=substep_count,
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.execution_group = execution_group
        self.update_var = update_var
        self.update_value = float(update_value)
        self.call_count = 0
        self.dt_calls: List[float] = []
        self.time_calls: List[float] = []

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, UpdateItem]:
        self.call_count += 1
        self.dt_calls.append(float(dt.item()))
        self.time_calls.append(float(time))
        return {
            self.update_var: UpdateItem(
                value=torch.tensor(self.update_value, dtype=torch.float32),
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="dimensionless",
                source_functor=self.name,
            )
        }


class StateUpdateRecorder:
    """In-memory recorder for accepted simulator state updates."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record_state_update(
        self,
        step: int,
        variable: str,
        pre_value: Any,
        post_value: Any,
        functor_increments: Dict[str, Any],
        unit: str,
    ) -> None:
        self.records.append(
            {
                "step": step,
                "variable": variable,
                "pre_value": pre_value,
                "post_value": post_value,
                "functor_increments": functor_increments,
                "unit": unit,
            }
        )


class DiagnosticSubstepFunctor(RecordingFunctor):
    """Exercise read-only diagnostics and control requests across substeps."""

    def __init__(self, *, emit_intents=False, undeclared_output=False):
        super().__init__(
            "diagnostic_substeps", "fast", "shared_y", input_vars=["shared_x"], substep_count=2,
        )
        self.metadata.output_variables.extend(
            ["shared_x", "diagnostics.sample", "diagnostics.accumulated"]
        )
        self.emit_intents = emit_intents
        self.undeclared_output = undeclared_output
        self.seen_x = []
        self.diagnostic_gain = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, state, time, dt):
        # References: RELIC UpdateItem contract; Hairer et al. (1993).
        # Mathematical expression: delta_y = x at each substep; d_x = 100.
        # Parameter meanings: x is a read-only input and y is accumulated.
        # Units: x, y and diagnostic counters are dimensionless; dt is seconds.
        # Domain/constraints: diagnostics never advance the provisional x.
        # Sign convention: positive delta_y increases y.
        result = super().forward(state, time, dt)
        value = state["shared_x"]
        self.seen_x.append(value.item())
        result["shared_y"].value = value
        result["shared_x"] = UpdateItem(
            value.new_tensor(100.0), mode=UpdateMode.REPLACE,
            write_mode=StateWriteMode.DERIVED, unit="dimensionless",
        )
        result["diagnostics.sample"] = self.diagnostic_gain * self.call_count
        result["diagnostics.accumulated"] = UpdateItem(
            value.new_tensor(2.0), mode=UpdateMode.ACCUMULATE,
            write_mode=StateWriteMode.DERIVED, unit="dimensionless", per_timestep=True,
        )
        if self.emit_intents:
            result["export"] = ExternalIntent(
                "write_outputs", "output_file", f"substep-{self.call_count}",
                {"variables": ["shared_y"]},
            )
        if self.undeclared_output:
            result["unscoped_counter"] = value.new_tensor(9.0)
        return result


class DependencyOutputProbe(RecordingFunctor):
    def __init__(self, producer, *, fail=False):
        super().__init__("output_probe", "fast", "shared_z")
        self.producer = producer
        self.fail = fail
        self.received_keys = set()

    def get_dependencies(self):
        return [self.producer]

    def forward(self, state, time, dt, subprocess_outputs):
        if self.fail:
            raise RuntimeError("injected late group failure")
        source = subprocess_outputs[self.producer]
        self.received_keys = set(source)
        result = super().forward(state, time, dt)
        result["shared_z"].value = source["shared_y"].value
        return result


class ExplicitUpdateFunctor(BiologicalProcessFunctor):
    """Functor fixture returning explicit UpdateItem payloads."""

    def __init__(
        self,
        name: str,
        update_var: str,
        mode: UpdateMode,
        update_value: float = 0.0,
        substep_count: int = 1,
        use_state_value: bool = False,
        units_by_call: Optional[List[str]] = None,
        basis: str = "per_step",
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            description=f"Explicit update functor {name}",
            references=["RELIC simulator execution test fixture"],
            temporal_scale=TemporalScale.SECOND,
            input_variables=[update_var] if use_state_value else [],
            output_variables=[update_var],
            substep_count=substep_count,
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.execution_group = "fast"
        self.update_var = update_var
        self.mode = mode
        self.update_value = float(update_value)
        self.use_state_value = use_state_value
        self.units_by_call = list(units_by_call or ["kg m-2"])
        # "per_step": an amount (per_timestep=True) unless the mode is FLUX;
        # "per_second": an unintegrated rate (per_timestep=False).
        self.basis = basis
        self.call_count = 0
        self.time_calls: List[float] = []
        self.seen_values: List[float] = []

    def get_dependencies(self) -> List[str]:
        return []

    def forward(
        self,
        state: Dict[str, Any],
        time: float,
        dt: torch.Tensor,
    ) -> Dict[str, UpdateItem]:
        del dt
        call_index = self.call_count
        self.call_count += 1
        self.time_calls.append(float(time))
        if self.update_var in state:
            self.seen_values.append(float(state[self.update_var]))
        value = state[self.update_var] if self.use_state_value else self.update_value
        unit = self.units_by_call[min(call_index, len(self.units_by_call) - 1)]
        return {
            self.update_var: UpdateItem(
                value=torch.as_tensor(value, dtype=torch.float32),
                mode=self.mode,
                write_mode=StateWriteMode.MUTABLE,
                unit=unit,
                basis=self.basis,
                per_timestep=self.mode != UpdateMode.FLUX and self.basis == "per_step",
                element=None,
                source_functor=self.name,
            )
        }


class StateReaderFunctor(RecordingFunctor):
    """Records one state variable as seen after a named writer of the same group."""

    def __init__(self, name: str, read_var: str, after: str) -> None:
        super().__init__(name, "fast", "shared_y", input_vars=[read_var])
        self.read_var = read_var
        self.after = after
        self.seen: List[float] = []

    def get_dependencies(self) -> List[str]:
        return [self.after]

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, UpdateItem]:
        self.seen.append(float(state[self.read_var]))
        return super().forward(state, time, dt)


def _initialize_shared_x(simulator: RELICSimulator, value: float, units: str = "kg m-2") -> None:
    """Initialize shared_x and declare the unit the explicit-update fixture emits.

    The simulator infers units for ad hoc states from their names (shared_x -> dimensionless);
    the StateManager unit contract then rejects a kg m-2 update, so the fixture must declare
    the storage unit explicitly instead of relying on an unchecked mismatch.
    """
    simulator.initialize_state({"shared_x": value})
    simulator.state_manager.register_variable(
        StateVariableSpec(name="shared_x", units=units, conserved=False, required=False)
    )


def _build_initialized_simulator(config: Dict[str, Any]) -> RELICSimulator:
    simulator = _declared_simulator(config)
    simulator.initialize_state({"test_signal": 300.0, "shared_x": 0.0})
    return simulator


def _execute_group_with_step_tracking(
    simulator: RELICSimulator,
    functor_name: str,
    dt: float,
) -> Dict[str, Any]:
    """Execute one group with the same tracking maps initialized by step()."""
    simulator._step_update_sources = {}
    simulator._step_functor_increments = {}
    simulator._step_provenance_detail = {}
    return simulator._execute_group([functor_name], dt, start_time=simulator.current_time)


def test_configurable_medium_interval() -> None:
    simulator = _build_initialized_simulator(
        {"time_step": 3600.0, "medium_interval_seconds": 21600.0}
    )
    medium = RecordingFunctor(
        name="medium_interval_functor",
        execution_group="medium",
        update_var="test_signal",
        update_value=0.0,
    )
    simulator.register_functor(medium)

    for _ in range(5):
        simulator.step()
    assert medium.call_count == 0

    simulator.step()
    assert medium.call_count == 1


def test_configurable_slow_interval() -> None:
    simulator = _build_initialized_simulator(
        {"time_step": 86400.0, "slow_interval_seconds": 7.0 * 86400.0}
    )
    slow = RecordingFunctor(
        name="slow_interval_functor",
        execution_group="slow",
        update_var="test_signal",
        update_value=0.0,
    )
    simulator.register_functor(slow)

    for _ in range(6):
        simulator.step()
    assert slow.call_count == 0

    simulator.step()
    assert slow.call_count == 1


def test_default_intervals_backward_compat() -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0})

    simulator._previous_day_index = 0
    simulator._previous_medium_index = 0
    assert simulator._should_execute_medium(future_time=86400.0) is True

    simulator._previous_month_index = 0
    simulator._slow_interval_seconds = None
    assert simulator._should_execute_slow(future_time=31.0 * 86400.0 + 60.0) is True


@pytest.mark.parametrize(
    ("origin", "elapsed", "expected"),
    [("2020-01-02T00:00:00", 0.0, 24228),
     ("2020-01-02T00:00:00Z", 0.0, 24228),
     ("2020-01-02T00:00:00+00:00", 0.0, 24228),
     ("2020-01-01T01:00:00+02:00", 0.0, 24227),
     ("2020-01-01T01:00:00+02:00", 3600.0, 24228),
     ("2020-02-29T23:30:00", 1800.0, 24230)],
)
def test_simulator_month_owner_uses_the_original_epoch_instant(origin, elapsed, expected):
    """Timezone conversion must preserve the instant, including a previous month."""
    import cftime

    simulator = _declared_simulator({
        "device": "cpu", "calendar_type": "proleptic_gregorian", "time_origin": origin,
    })
    # References: CF Conventions 1.8, Section 4.4; cftime.num2date reference API.
    # Mathematical expression: d=CF_decode_C(t; seconds since o);
    #   m=12*(year(d)-1)+month(d)-1.
    # Parameter meanings: C is the source calendar, o the original epoch instant,
    #   t elapsed seconds, d the independent decoded date, m its absolute ordinal.
    # Units: t [s]; o,d timestamps; m is dimensionless.
    # Domain/constraints: valid year>=1 civil dates and explicit zone offsets;
    #   retaining the raw input must not change the represented reference instant.
    # Sign convention: positive t advances time; no quantity transfer.
    reference = cftime.num2date(elapsed, f"seconds since {origin}", calendar="proleptic_gregorian")
    assert 12 * (reference.year - 1) + reference.month - 1 == expected
    assert simulator._compute_month_index(elapsed) == expected
    assert simulator._get_setting("time_origin") == origin
    assert simulator._time_origin == origin


def test_simulator_month_owner_delegates_existing_parser_and_period_decoder(monkeypatch):
    simulator = _declared_simulator({
        "device": "cpu", "calendar_type": "gregorian",
        "time_origin": "2020-01-01T01:00:00+02:00",
    })
    calls = []

    def decode(seconds, calendar, time_origin=None):
        calls.append((seconds, calendar, time_origin))
        return 24227

    monkeypatch.setattr(simulator._calendar_period_indexer, "elapsed_month_index", decode)
    assert simulator._compute_month_index(0.0) == 24227
    assert calls == [(0.0, "standard", "2019-12-31T23:00:00")]
    assert simulator._time_origin == "2020-01-01T01:00:00+02:00"


def test_real_origin_initial_month_does_not_force_first_slow_execution():
    half_hour = float(SECONDS_PER_MINUTE) * 30
    simulator = _build_initialized_simulator({
        "device": "cpu", "time_step": half_hour,
        "calendar_type": "proleptic_gregorian", "time_origin": "2020-01-02T00:00:00",
    })
    slow = RecordingFunctor("real_origin_slow", "slow", "test_signal")
    simulator.register_functor(slow)
    assert simulator._previous_month_index == 24228
    simulator.step()
    assert slow.call_count == 0
    assert simulator.current_time == half_hour
    assert simulator._last_slow_execution_time == 0.0
    assert simulator._previous_month_index == 24228


@pytest.mark.parametrize("execution_api", ["step", "run"])
def test_real_leap_month_boundary_uses_original_slow_effective_dt(execution_api):
    half_hour = float(SECONDS_PER_MINUTE) * 30
    simulator = _build_initialized_simulator({
        "device": "cpu", "time_step": half_hour,
        "calendar_type": "proleptic_gregorian", "time_origin": "2020-02-29T23:00:00",
    })
    slow = RecordingFunctor("leap_boundary_slow", "slow", "test_signal")
    simulator.register_functor(slow)
    assert simulator._previous_month_index == 24229
    if execution_api == "step":
        simulator.step()
        assert slow.call_count == 0
        assert not simulator._should_execute_slow(future_time=half_hour)
        assert simulator._should_execute_slow(future_time=2 * half_hour)
        assert simulator._previous_month_index == 24229
        simulator.step()
        simulator.step()
    else:
        simulator.run(start_time=0.0, end_time=3 * half_hour, dt=half_hour)
    assert slow.call_count == 1
    assert slow.time_calls == [0.0]
    assert slow.dt_calls == [2 * half_hour]
    assert simulator._previous_month_index == 24230
    assert simulator._last_slow_execution_time == 2 * half_hour
    assert simulator.current_time == 3 * half_hour


def test_real_origin_bounded_year_domain_is_not_an_epoch_fallback():
    from relic.coupling.scales.temporal.core import ValidationError

    simulator = _declared_simulator({
        "device": "cpu", "calendar_type": "proleptic_gregorian",
        "time_origin": "0000-01-01T00:00:00",
    })
    with pytest.raises(ValidationError, match="Year must be >= 1"):
        simulator._compute_month_index(0.0)


def _leap_boundary_simulator(time_origin="2020-02-27T00:00:00"):
    simulator = _build_initialized_simulator({
        "device": "cpu", "time_step": float(SECONDS_PER_DAY),
        "calendar_type": "proleptic_gregorian", "time_origin": time_origin,
    })
    slow = RecordingFunctor("leap_restart_slow", "slow", "test_signal", update_value=1.0)
    simulator.register_functor(slow)
    return simulator, slow


def test_real_origin_leap_boundary_replays_after_checkpoint_restore(tmp_path):
    """An interrupted run keeps the committed month and the slow interval start."""
    day = float(SECONDS_PER_DAY)
    # References: CF Conventions 1.8, Section 4.4; docs/source/theory/time_scaling.md
    #   Section 4.5; RELIC step transaction contract (execution_loop.step_simulation).
    # Mathematical expression: o = 2020-02-27T00:00; the first month boundary is
    #   t_b = 3 d (27, 28 and 29 February); the slow group runs once over
    #   [t_last, t_b] with effective_dt = t_b - t_last = 3 d, t_last = 0.
    # Parameter meanings: o is the origin, t_b the elapsed time of 2020-03-01T00:00,
    #   t_last the last slow execution time, m the absolute month ordinal
    #   12 * (year - 1) + month - 1 (24229 for February 2020, 24230 for March).
    # Units: times and effective_dt [s]; m is dimensionless.
    # Domain/constraints: proleptic Gregorian calendar, 2020 is a leap year; the
    #   checkpoint is taken at t = 2 d (2020-02-29T00:00), before the boundary.
    # Sign convention: time advances forward; the recorder adds a nonnegative
    #   increment to test_signal.
    reference, reference_slow = _leap_boundary_simulator()
    for _ in range(5):
        reference.step()
    assert reference_slow.time_calls == [0.0]
    assert reference_slow.dt_calls == [3 * day]

    source, source_slow = _leap_boundary_simulator()
    source.step()
    source.step()
    assert source_slow.call_count == 0
    path = tmp_path / "leap_boundary.pt"
    source.create_checkpoint(path)

    restored, restored_slow = _leap_boundary_simulator()
    restored.restore_checkpoint(path)
    assert restored.current_time == 2 * day
    assert restored._previous_month_index == 24229
    assert restored._last_slow_execution_time == 0.0
    for _ in range(3):
        restored.step()

    assert restored_slow.time_calls == [0.0]
    assert restored_slow.dt_calls == [3 * day]
    assert restored._previous_month_index == reference._previous_month_index == 24230
    assert restored._last_slow_execution_time == reference._last_slow_execution_time == 3 * day
    assert restored.current_time == reference.current_time == 5 * day
    assert torch.equal(
        restored.state_manager.get_state()["test_signal"],
        reference.state_manager.get_state()["test_signal"],
    )


def test_checkpoint_of_another_time_origin_is_not_restored(tmp_path):
    """A checkpoint must not continue under a different reference instant."""
    source, _ = _leap_boundary_simulator()
    source.step()
    path = tmp_path / "origin_identity.pt"
    source.create_checkpoint(path)

    other, _ = _leap_boundary_simulator(time_origin="2021-02-27T00:00:00")
    with pytest.raises(ValueError, match="mismatched sections: scientific_contract"):
        other.restore_checkpoint(path)
    assert other.current_time == 0.0
    assert other._previous_month_index == 24241


def test_time_ops_use_day_ssot_and_initialized_medium_interval(monkeypatch) -> None:
    monkeypatch.setattr(
        time_ops_module,
        "SECONDS_PER_DAY",
        float(SECONDS_PER_DAY) / 2.0,
        raising=False,
    )

    assert time_ops_module.compute_day_index(float(SECONDS_PER_DAY)) == 2

    source = inspect.getsource(time_ops_module)
    assert "86400" not in source
    assert "simulator._medium_interval_seconds" in source


def test_adaptive_dt_reject_step() -> None:
    simulator = _build_initialized_simulator(
        {
            "time_step": 60.0,
            "adaptive_dt_enabled": True,
            "adaptive_dt_min": 1.0,
            "adaptive_dt_max": 600.0,
            "adaptive_atol": 1.0e-6,
            "adaptive_rtol": 1.0e-3,
        }
    )
    recorder = StateUpdateRecorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    stiff = RecordingFunctor(
        name="adaptive_reject_functor",
        execution_group="fast",
        update_var="test_signal",
        update_value=1.0e6,
    )
    simulator.register_functor(stiff)

    simulator.step()

    assert stiff.call_count >= 2
    assert stiff.dt_calls[1] < stiff.dt_calls[0]
    assert simulator.current_time < 60.0
    assert simulator.state_manager.get_diagnostics()["update_counter"] == 1
    assert len(recorder.records) == 1
    assert recorder.records[0]["variable"] == "test_signal"


def test_adaptive_rejection_rolls_back_provenance_without_cloning_history() -> None:
    """A rejected attempt leaves no writer record and the earlier records are kept as they are."""
    simulator = _build_initialized_simulator(
        {
            "time_step": 60.0,
            "adaptive_dt_enabled": True,
            "adaptive_dt_min": 1.0,
            "adaptive_dt_max": 600.0,
            "adaptive_atol": 1.0e-6,
            "adaptive_rtol": 1.0e-3,
        }
    )
    stiff = RecordingFunctor(
        name="adaptive_provenance_functor",
        execution_group="fast",
        update_var="test_signal",
        update_value=1.0e6,
    )
    simulator.register_functor(stiff)
    simulator.step()

    detail = simulator._step_provenance_detail
    accepted_records = list(detail["test_signal"])
    assert len(accepted_records) == 1
    calls_before = stiff.call_count

    simulator.step()

    assert stiff.call_count - calls_before == 2
    assert simulator._step_provenance_detail is detail
    records = detail["test_signal"]
    assert len(records) == 2
    assert records[0] is accepted_records[0]


def test_adaptive_dt_accept_and_grow() -> None:
    simulator = _build_initialized_simulator(
        {
            "time_step": 60.0,
            "adaptive_dt_enabled": True,
            "adaptive_dt_min": 1.0,
            "adaptive_dt_max": 600.0,
            "adaptive_atol": 1.0e-6,
            "adaptive_rtol": 1.0e-3,
        }
    )
    smooth = RecordingFunctor(
        name="adaptive_accept_functor",
        execution_group="fast",
        update_var="test_signal",
        update_value=1.0e-8,
    )
    simulator.register_functor(smooth)

    simulator.step()

    assert smooth.call_count == 1
    assert simulator._adaptive_next_dt > 60.0
    assert simulator._adaptive_next_dt <= 600.0


class ReplaceRelaxationFunctor(BiologicalProcessFunctor):
    """REPLACE writer whose new value differs from the old one by rate * dt."""

    def __init__(self, name: str, update_var: str, rate: float) -> None:
        metadata = FunctorMetadata(
            name=name,
            description=f"REPLACE relaxation fixture {name}",
            references=["RELIC simulator adaptive-metric test fixture"],
            temporal_scale=TemporalScale.SECOND,
            input_variables=[update_var],
            output_variables=[update_var],
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.execution_group = "fast"
        self.update_var = update_var
        self.rate = float(rate)
        self.dt_calls: List[float] = []

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, UpdateItem]:
        self.dt_calls.append(float(dt.item()))
        current = state[self.update_var]
        return {
            self.update_var: UpdateItem(
                value=current + self.rate * dt.to(dtype=current.dtype),
                mode=UpdateMode.REPLACE,
                per_timestep=True,
                unit="dimensionless",
                source_functor=self.name,
            )
        }


def test_adaptive_error_norm_measures_the_committed_change_of_each_written_variable() -> None:
    """u_k = x_k(committed) - x_k(before) for every written state variable.

    The payload form only names the written variables: a REPLACE value, an
    integrated amount, a per-second rate and a raw increment that committed the
    same change give the same metric; diagnostics never enter it.
    """
    from types import SimpleNamespace

    from relic.core.simulator.execution_loop import _compute_error_norm

    x = torch.tensor(300.0, dtype=torch.float64)
    tol = 1.0e-6 + 1.0e-3 * 300.0

    def norm(after: Dict[str, Any], updates: Dict[str, Any], monitored=None) -> float:
        controller = SimpleNamespace(
            _adaptive_atol=1.0e-6,
            _adaptive_rtol=1.0e-3,
            _adaptive_monitor_vars=monitored,
            _get_current_state=lambda: after,
        )
        return _compute_error_norm(controller, {"x": x.clone(), "y": x.clone()}, updates)

    committed = {"x": x + 0.36, "y": x.clone()}
    replace = UpdateItem(value=x + 0.36, mode=UpdateMode.REPLACE)
    amount = UpdateItem(
        value=torch.tensor(0.36, dtype=torch.float64), mode=UpdateMode.ACCUMULATE, per_timestep=True,
    )
    rate = UpdateItem(
        value=torch.tensor(0.006, dtype=torch.float64), mode=UpdateMode.ACCUMULATE,
        basis="per_second", per_timestep=False,
    )
    raw = torch.tensor(0.36, dtype=torch.float64)
    for update in (replace, amount, rate, raw):
        assert norm(committed, {"x": update}) == pytest.approx(0.36 / tol, rel=1e-12)

    derived = UpdateItem(value=torch.tensor(99.0), mode=UpdateMode.REPLACE, write_mode=StateWriteMode.DERIVED)
    assert norm(committed, {"x": rate, "y": derived}) == pytest.approx(0.36 / tol, rel=1e-12)
    assert norm(committed, {"x": rate, "diagnostics.sample": torch.tensor(5.0)}) == pytest.approx(
        0.36 / tol, rel=1e-12
    )
    # m counts the written state variables: a written variable left unchanged enters with u = 0.
    assert norm(committed, {"x": rate, "y": UpdateItem(value=torch.tensor(0.0))}) == pytest.approx(
        0.36 / tol / math.sqrt(2.0), rel=1e-12
    )
    assert norm(committed, {"x": rate, "y": UpdateItem(value=torch.tensor(0.0))}, monitored=["y"]) == 0.0
    # A variable without a pre-step value is skipped; m = 0 gives Err = 0.
    assert norm({"z": x.clone()}, {"z": amount}) == 0.0
    with pytest.raises(RuntimeError, match="missing from the committed state"):
        norm({"y": x.clone()}, {"x": rate})
    with pytest.raises(ValueError, match="shape"):
        norm({"x": torch.zeros(2, dtype=torch.float64), "y": x.clone()}, {"x": replace})


@pytest.mark.parametrize("splitting_mode", ["lie", "strang"])
def test_adaptive_replace_updates_are_accepted_and_the_step_grows(splitting_mode) -> None:
    """A small REPLACE change is not rejected as if the whole value were an increment."""
    simulator = _build_initialized_simulator(
        {
            "time_step": 60.0,
            "splitting_mode": splitting_mode,
            "adaptive_dt_enabled": True,
            "adaptive_dt_min": 1.0,
            "adaptive_dt_max": 600.0,
            "adaptive_atol": 1.0e-6,
            "adaptive_rtol": 1.0e-3,
        }
    )
    relaxation = ReplaceRelaxationFunctor("replace_relaxation", "test_signal", rate=1.0e-4)
    simulator.register_functor(relaxation)
    if splitting_mode == "strang":
        simulator.register_functor(
            RecordingFunctor(name="replace_partner", execution_group="fast", update_var="shared_x")
        )

    simulator.step()

    # Err = rate * dt / (atol + rtol * 300) = 0.02 (lie) or sqrt(0.02^2 / 2) (strang);
    # dt_next = 60 * min(5, 0.9 / sqrt(Err)) = 300 s.
    assert simulator.current_time == pytest.approx(60.0)
    assert sum(relaxation.dt_calls) == pytest.approx(60.0)
    assert simulator._adaptive_next_dt == pytest.approx(300.0)
    assert float(simulator.get_state()["test_signal"]) == pytest.approx(300.0 + 6.0e-3, abs=1e-4)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("adaptive_atol", 0.0),
        ("adaptive_atol", -1.0e-6),
        ("adaptive_atol", float("nan")),
        ("adaptive_rtol", -1.0e-3),
        ("adaptive_rtol", float("inf")),
        ("adaptive_dt_min", 0.0),
        ("adaptive_dt_max", 0.5),
    ],
)
def test_invalid_adaptive_controller_setting_is_rejected(setting, value) -> None:
    """With adaptive stepping enabled, atol > 0, rtol >= 0 (finite) and
    0 < dt_min <= dt_max are checked where they are read; without it the
    settings are unused and not checked."""
    config = {"time_step": 60.0, "adaptive_dt_min": 1.0, setting: value}
    with pytest.raises(ValueError, match=setting):
        _declared_simulator({**config, "adaptive_dt_enabled": True})
    _declared_simulator(config)


_ADAPTIVE_KEYS = {
    "atol": "computation.scheduling.adaptive.atol",
    "rtol": "computation.scheduling.adaptive.rtol",
    "dt_min": "computation.scheduling.adaptive.dt_min_seconds",
    "dt_max_factor": "computation.scheduling.adaptive.dt_max_factor",
}


def test_adaptive_controller_defaults_come_from_the_parameter_database(restore_global_parameter_manager) -> None:
    """Without a configured value each setting takes its computation.scheduling.adaptive.*
    value; adaptive_dt_max defaults to dt_max_factor * time_step."""
    restore_global_parameter_manager.set_batch(
        {
            _ADAPTIVE_KEYS["atol"]: 2.0e-6,
            _ADAPTIVE_KEYS["rtol"]: 2.0e-3,
            _ADAPTIVE_KEYS["dt_min"]: 2.0,
            _ADAPTIVE_KEYS["dt_max_factor"]: 7.0,
        },
        validate=True,
        notify=False,
    )
    simulator = _declared_simulator({"time_step": 60.0, "adaptive_dt_enabled": True})
    assert simulator._adaptive_atol == pytest.approx(2.0e-6)
    assert simulator._adaptive_rtol == pytest.approx(2.0e-3)
    assert simulator._adaptive_dt_min == pytest.approx(2.0)
    assert simulator._adaptive_dt_max == pytest.approx(7.0 * 60.0)
    configured = _declared_simulator(
        {"time_step": 60.0, "adaptive_dt_enabled": True, "adaptive_atol": 5.0e-7, "adaptive_dt_max": 900.0}
    )
    assert configured._adaptive_atol == pytest.approx(5.0e-7)
    assert configured._adaptive_dt_max == pytest.approx(900.0)


def test_a_database_adaptive_default_passes_the_same_checks(restore_global_parameter_manager) -> None:
    """A database dt_min above the resulting dt_max violates 0 < dt_min <= dt_max."""
    restore_global_parameter_manager.set_batch(
        {_ADAPTIVE_KEYS["dt_min"]: 1000.0, _ADAPTIVE_KEYS["dt_max_factor"]: 2.0},
        validate=True,
        notify=False,
    )
    with pytest.raises(ValueError, match="adaptive_dt_max"):
        _declared_simulator({"time_step": 60.0, "adaptive_dt_enabled": True})


def test_time_settings_default_to_the_time_constants() -> None:
    """medium_interval_seconds and max_time default to one day, checkpoint_frequency
    to one hour, from relic.utils.time_constants (no literal 86400 / 3600)."""
    import inspect

    from relic.core.simulator import init_ops
    from relic.utils.time_constants import SECONDS_PER_DAY, SECONDS_PER_HOUR

    simulator = _declared_simulator({"time_step": 60.0})
    assert simulator._medium_interval_seconds == SECONDS_PER_DAY
    assert simulator.max_time == SECONDS_PER_DAY
    assert simulator.checkpoint_frequency == SECONDS_PER_HOUR
    import re

    # _get_setting(<quoted key>, 86400 / 3600 ...): the quote is matched by "."
    literal_default = re.compile(r"_get_setting\(\s*.[a-z_]+.,\s*(86400|3600)")
    for source in (inspect.getsource(init_ops.configure_calendar_settings), inspect.getsource(RELICSimulator.__init__)):
        assert not literal_default.search(source)
    assert inspect.getsource(RELICSimulator.__init__).count("'diagnostic_reference_date'") == 1


def test_disabled_adaptive_stepping_reads_no_adaptive_parameters(monkeypatch) -> None:
    """Without adaptive_dt_enabled the controller values are unused: nothing
    under computation.scheduling.adaptive.* is read, and they stay unset."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    original_get = manager.get
    read_keys = []

    def recording_get(key, *args, **kwargs):
        read_keys.append(key)
        return original_get(key, *args, **kwargs)

    monkeypatch.setattr(manager, "get", recording_get)
    simulator = _declared_simulator({"time_step": 60.0})
    assert not [key for key in read_keys if key.startswith("computation.scheduling.adaptive.")]
    for name in ("_adaptive_atol", "_adaptive_rtol", "_adaptive_dt_min", "_adaptive_dt_max"):
        assert getattr(simulator, name) is None, name
    enabled = _declared_simulator({"time_step": 60.0, "adaptive_dt_enabled": True})
    assert enabled._adaptive_atol is not None


def test_a_missing_configuration_is_the_empty_configuration() -> None:
    """RELICSimulator(None) takes the same defaults as RELICSimulator({})."""
    from relic.utils.time_constants import SECONDS_PER_DAY

    none_config = RELICSimulator(None)
    empty_config = RELICSimulator({})
    for name in ("time_step", "max_time", "checkpoint_frequency", "output_frequency", "enable_auto_scale"):
        assert getattr(none_config, name) == getattr(empty_config, name), name
    assert none_config.max_time == SECONDS_PER_DAY
    assert none_config.config == {}


def test_error_norm_reads_the_configured_tolerances_without_defaults() -> None:
    from types import SimpleNamespace

    from relic.core.simulator.execution_loop import _compute_error_norm

    x = torch.tensor(300.0, dtype=torch.float64)
    controller = SimpleNamespace(_adaptive_monitor_vars=None, _get_current_state=lambda: {"x": x + 1.0})
    with pytest.raises(AttributeError, match="_adaptive_atol"):
        _compute_error_norm(controller, {"x": x.clone()}, {"x": UpdateItem(value=x + 1.0, mode=UpdateMode.REPLACE)})


class StorageTendencyFunctor(BiologicalProcessFunctor):
    """One storage tendency dx/dt = q expressed in one supported update form.

    "per_second": ACCUMULATE rate q (StateManager integrates it once);
    "integrated": ACCUMULATE amount q h; "replace": REPLACE x + q h; "flux": FLUX
    rate q (the group owner lowers it to the amount q h). h is the interval of the
    call, so every form commits the same storage change over a step.
    """

    FORMS = ("per_second", "integrated", "replace", "flux")

    def __init__(self, name: str, update_var: str, form: str, rate: float, substep_count: int = 1) -> None:
        if form not in self.FORMS:
            raise ValueError(f"unknown update form {form!r}")
        metadata = FunctorMetadata(
            name=name,
            description=f"Storage tendency fixture {name} ({form})",
            references=["RELIC simulator adaptive-metric test fixture"],
            temporal_scale=TemporalScale.SECOND,
            input_variables=[update_var],
            output_variables=[update_var],
            substep_count=substep_count,
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.execution_group = "fast"
        self.update_var = update_var
        self.form = form
        self.rate = float(rate)
        self.dt_calls: List[float] = []

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, UpdateItem]:
        # References: LeVeque (2002), rate-times-interval integration; RELIC
        #   UpdateItem contract (UpdateMode, per_timestep, basis).
        # Mathematical expression: delta_x = q h in every form; "per_second" and
        #   "flux" return q, "integrated" returns q h, "replace" returns x + q h.
        # Parameter meanings: x is the storage before the call, q the constant
        #   tendency, h the interval of the call (a substep or a Strang half-step).
        # Units: x [kg m-2]; q [kg m-2 s-1]; h [s].
        # Domain/constraints: h > 0; q is constant over the step.
        # Sign convention: positive q increases the storage.
        interval = float(dt.item())
        self.dt_calls.append(interval)
        if self.form in ("per_second", "flux"):
            item = UpdateItem(
                value=torch.tensor(self.rate, dtype=torch.float64),
                mode=UpdateMode.ACCUMULATE if self.form == "per_second" else UpdateMode.FLUX,
                unit="kg m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            )
        elif self.form == "integrated":
            item = UpdateItem(
                value=torch.tensor(self.rate * interval, dtype=torch.float64),
                mode=UpdateMode.ACCUMULATE,
                unit="kg m-2",
                per_timestep=True,
                source_functor=self.name,
            )
        else:
            item = UpdateItem(
                value=state[self.update_var] + self.rate * interval,
                mode=UpdateMode.REPLACE,
                unit="kg m-2",
                per_timestep=True,
                source_functor=self.name,
            )
        return {self.update_var: item}


def _adaptive_step_trace(monkeypatch, simulator: RELICSimulator, variable: str) -> Dict[str, Any]:
    """Run one adaptive step; record the control metrics, time, change and next dt."""
    import relic.core.simulator.execution_loop as execution_loop

    metrics: List[float] = []
    original = execution_loop._compute_error_norm

    def record(sim, state_before, updates):
        value = original(sim, state_before, updates)
        metrics.append(value)
        return value

    before = float(simulator.get_state()[variable])
    with monkeypatch.context() as local:
        local.setattr(execution_loop, "_compute_error_norm", record)
        simulator.step()
    return {
        "metrics": metrics,
        "time": simulator.current_time,
        "change": float(simulator.get_state()[variable]) - before,
        "next_dt": simulator._adaptive_next_dt,
    }


def _adaptive_config(dt: float, splitting_mode: str, monitored: str, **extra: Any) -> Dict[str, Any]:
    config = {
        "time_step": dt,
        "splitting_mode": splitting_mode,
        "adaptive_dt_enabled": True,
        "adaptive_dt_min": 1.0,
        "adaptive_dt_max": 600.0,
        "adaptive_atol": 1.0e-6,
        "adaptive_rtol": 1.0e-3,
        "adaptive_monitor_vars": [monitored],
    }
    config.update(extra)
    return config


@pytest.mark.parametrize("dt", [10.0, 60.0])
@pytest.mark.parametrize("substeps", [1, 2])
@pytest.mark.parametrize("splitting_mode", ["lie", "strang"])
def test_adaptive_metric_is_the_committed_storage_change_of_every_update_form(
    monkeypatch, splitting_mode, substeps, dt,
) -> None:
    """x = 300 kg m-2 and dx/dt = q = 0.006 kg m-2 s-1 in four update forms.

    Every form commits delta_x = q h, so Err = q h / (atol + rtol x) for all of
    them: 0.2 for h = 10 s (accepted; next dt = h * 0.9 / sqrt(Err)) and 1.2 for
    h = 60 s (rejected once; the retry h' = h * 0.9 / sqrt(Err) is accepted
    without a second check and proposed as the next dt).
    """
    tol = 1.0e-6 + 1.0e-3 * 300.0
    err = 0.006 * dt / tol
    factor = 0.9 / math.sqrt(err)
    if err > 1.0:
        interval = max(dt * max(0.1, factor), 1.0)
        next_dt, attempts = interval, 2
    else:
        interval = dt
        next_dt, attempts = min(dt * min(5.0, max(0.1, factor)), 600.0), 1
    calls_per_attempt = substeps * (2 if splitting_mode == "strang" else 1)

    for form in StorageTendencyFunctor.FORMS:
        simulator = _declared_simulator(_adaptive_config(dt, splitting_mode, "shared_x"))
        simulator.initialize_state({"shared_x": torch.tensor(300.0, dtype=torch.float64), "shared_y": 0.0})
        simulator.state_manager.register_variable(
            StateVariableSpec(name="shared_x", units="kg m-2", conserved=False, required=False)
        )
        tendency = StorageTendencyFunctor("storage_tendency", "shared_x", form, 0.006, substep_count=substeps)
        simulator.register_functor(tendency)
        if splitting_mode == "strang":
            simulator.register_functor(
                RecordingFunctor(name="strang_partner", execution_group="fast", update_var="shared_y")
            )
            assert list(simulator._order_fast) == ["storage_tendency", "strang_partner"]

        trace = _adaptive_step_trace(monkeypatch, simulator, "shared_x")

        assert trace["metrics"] == pytest.approx([err], rel=1e-6), form
        assert len(tendency.dt_calls) == attempts * calls_per_attempt, form
        assert trace["time"] == pytest.approx(interval, rel=1e-6), form
        assert trace["change"] == pytest.approx(0.006 * interval, rel=1e-6), form
        assert trace["next_dt"] == pytest.approx(next_dt, rel=1e-6), form


@pytest.mark.parametrize("dt", [10.0, 60.0])
@pytest.mark.parametrize(("stored", "rate"), [(0.5, 0.5), (0.5, 0.5006)])
def test_adaptive_metric_of_a_declared_boundary_rate_is_the_change_of_the_stored_rate(
    monkeypatch, stored, rate, dt,
) -> None:
    """n2o_flux is a ContractRegistry flux: StateManager stores the rate, not q h.

    A per-second or instantaneous rate and a REPLACE of the same value commit the
    same stored rate, so Err = |q - q0| / (atol + rtol |q0|) for all three and
    does not scale with h: a steady rate gives Err = 0 (accepted, next dt = 2 h);
    a changed rate gives Err = 1.198 (rejected once).
    """
    err = abs(float(torch.tensor(rate, dtype=torch.float32)) - stored) / (1.0e-6 + 1.0e-3 * stored)
    if err > 1.0:
        interval = max(dt * max(0.1, 0.9 / math.sqrt(err)), 1.0)
        next_dt, attempts = interval, 2
    else:
        interval, next_dt, attempts = dt, 2.0 * dt, 1

    for mode, basis in (
        (UpdateMode.ACCUMULATE, "per_second"),
        (UpdateMode.ACCUMULATE, "instantaneous"),
        (UpdateMode.REPLACE, "per_step"),
    ):
        simulator = _declared_simulator(
            _adaptive_config(dt, "lie", "n2o_flux", state_units={"n2o_flux": "g N m-2 s-1"})
        )
        simulator.initialize_state({"n2o_flux": stored})
        writer = ExplicitUpdateFunctor(
            name="nitrification",
            update_var="n2o_flux",
            mode=mode,
            update_value=rate,
            units_by_call=["g N m-2 s-1"],
            basis=basis,
        )
        simulator.register_functor(writer)

        trace = _adaptive_step_trace(monkeypatch, simulator, "n2o_flux")

        label = f"{mode.value}/{basis}"
        assert trace["metrics"] == pytest.approx([err], rel=1e-6, abs=1e-12), label
        assert writer.call_count == attempts, label
        assert trace["time"] == pytest.approx(interval, rel=1e-6), label
        assert float(simulator.get_state()["n2o_flux"]) == pytest.approx(rate, rel=1e-6), label
        assert trace["next_dt"] == pytest.approx(next_dt, rel=1e-6), label


def _rejecting_adaptive_simulator(dt_min: float) -> tuple:
    """Adaptive simulator whose fast Functor always gives Err >> 1."""
    simulator = _build_initialized_simulator(
        {
            "time_step": 300.0,
            "adaptive_dt_enabled": True,
            "adaptive_dt_min": dt_min,
            "adaptive_dt_max": 3000.0,
            "adaptive_atol": 1.0e-6,
            "adaptive_rtol": 1.0e-3,
        }
    )
    stiff = RecordingFunctor(
        name="always_rejected", execution_group="fast", update_var="test_signal", update_value=1.0e6
    )
    simulator.register_functor(stiff)
    return simulator, stiff


def test_adaptive_run_ends_exactly_at_end_time_after_a_truncated_rejected_step() -> None:
    """run() never passes end_time: the retry is never longer than the rejected attempt."""
    simulator, stiff = _rejecting_adaptive_simulator(dt_min=300.0)

    simulator.run(start_time=0.0, end_time=500.0)

    assert simulator.current_time == 500.0
    assert simulator.step_count == 2
    # Both attempts are at or below dt_min, so each is accepted without a retry.
    assert stiff.dt_calls == [300.0, 200.0]
    assert simulator._adaptive_next_dt == 300.0


def test_adaptive_run_and_step_keep_their_interval_contract_without_adaptivity() -> None:
    simulator = _build_initialized_simulator({"time_step": 300.0})
    recorder = RecordingFunctor(name="plain_recorder", execution_group="fast", update_var="test_signal")
    simulator.register_functor(recorder)

    simulator.run(start_time=0.0, end_time=500.0)
    assert simulator.current_time == 500.0
    assert recorder.dt_calls == [300.0, 200.0]

    stepped = _build_initialized_simulator({"time_step": 300.0, "adaptive_dt_enabled": True})
    stepped_recorder = RecordingFunctor(name="step_recorder", execution_group="fast", update_var="test_signal")
    stepped.register_functor(stepped_recorder)
    stepped.step(45.0)  # step(dt) runs the given dt; the proposal only feeds run()
    assert stepped.current_time == 45.0
    assert stepped_recorder.dt_calls == [45.0]


def test_rejected_attempt_not_longer_than_dt_min_is_accepted_with_a_warning(caplog) -> None:
    simulator, stiff = _rejecting_adaptive_simulator(dt_min=1.0)

    with caplog.at_level("WARNING", logger=_SCHEDULER_LOGGER):
        simulator.step(0.5)

    assert stiff.call_count == 1
    assert stiff.dt_calls == [0.5]
    assert simulator.current_time == 0.5
    messages = [record.getMessage() for record in caplog.records if "adaptive_dt_min" in record.getMessage()]
    assert len(messages) == 1
    assert "Err=" in messages[0] and "adaptive_dt_min=1 s" in messages[0]
    # The next proposal follows the accepted-step rule, clipped to [dt_min, dt_max].
    assert simulator._adaptive_next_dt == 1.0


def test_retry_after_rejection_is_strictly_shorter_than_the_attempt() -> None:
    simulator, stiff = _rejecting_adaptive_simulator(dt_min=1.0)

    simulator.step(300.0)

    assert stiff.dt_calls[0] == 300.0
    assert len(stiff.dt_calls) == 2
    assert 1.0 <= stiff.dt_calls[1] < 300.0
    assert stiff.time_calls == [0.0, 0.0]
    assert simulator.current_time == stiff.dt_calls[1]


def test_adaptive_dt_disabled_default() -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0})
    functor = RecordingFunctor(
        name="adaptive_disabled_functor",
        execution_group="fast",
        update_var="test_signal",
        update_value=1.0e6,
    )
    simulator.register_functor(functor)

    simulator.step()

    assert functor.call_count == 1
    assert simulator.current_time == pytest.approx(60.0)


def test_strang_splitting_half_steps() -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0, "splitting_mode": "strang"})
    functor_a = RecordingFunctor(
        name="split_a",
        execution_group="fast",
        update_var="test_signal",
        update_value=0.0,
    )
    functor_b = RecordingFunctor(
        name="split_b",
        execution_group="fast",
        update_var="shared_x",
        update_value=0.0,
    )
    simulator.register_functor(functor_a)
    simulator.register_functor(functor_b)

    simulator.step()

    name_to_functor = {functor_a.name: functor_a, functor_b.name: functor_b}
    first_name = simulator._order_fast[0]
    second_name = simulator._order_fast[1]

    assert name_to_functor[first_name].call_count == 2
    assert name_to_functor[first_name].dt_calls == [30.0, 30.0]
    assert name_to_functor[second_name].call_count == 1
    assert name_to_functor[second_name].dt_calls == [60.0]


def test_lie_splitting_default() -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0})
    functor_a = RecordingFunctor(
        name="lie_a",
        execution_group="fast",
        update_var="test_signal",
        update_value=0.0,
    )
    functor_b = RecordingFunctor(
        name="lie_b",
        execution_group="fast",
        update_var="shared_x",
        update_value=0.0,
    )
    simulator.register_functor(functor_a)
    simulator.register_functor(functor_b)

    simulator.step()

    assert functor_a.call_count == 1
    assert functor_b.call_count == 1
    assert functor_a.dt_calls == [60.0]
    assert functor_b.dt_calls == [60.0]


def test_event_trigger_fires() -> None:
    simulator = _build_initialized_simulator(
        {"time_step": 60.0, "medium_interval_seconds": 1.0e9}
    )
    medium = RecordingFunctor(
        name="event_medium_functor",
        execution_group="medium",
        update_var="test_signal",
        update_value=0.0,
    )
    simulator.register_functor(medium)
    simulator.register_event_trigger("medium", lambda state, t: True)

    simulator.step()

    assert medium.call_count == 1


def test_event_trigger_empty_default() -> None:
    simulator = _build_initialized_simulator(
        {"time_step": 60.0, "medium_interval_seconds": 1.0e9}
    )
    medium = RecordingFunctor(
        name="event_medium_default",
        execution_group="medium",
        update_var="test_signal",
        update_value=0.0,
    )
    simulator.register_functor(medium)

    simulator.step()

    assert medium.call_count == 0


def test_cross_group_auto_promote() -> None:
    simulator = _build_initialized_simulator(
        {
            "time_step": 60.0,
            "medium_interval_seconds": 1.0,
            "resolve_cross_group_deps": True,
        }
    )
    fast_producer = RecordingFunctor(
        name="fast_producer",
        execution_group="fast",
        update_var="shared_x",
        update_value=1.0,
        input_vars=[],
    )
    slow_consumer = RecordingFunctor(
        name="slow_consumer",
        execution_group="slow",
        update_var="test_signal",
        update_value=0.0,
        input_vars=["shared_x"],
    )
    simulator.register_functor(fast_producer)
    simulator.register_functor(slow_consumer)

    simulator.step()

    # The promotion is a move: membership and order change once, never duplicated.
    assert simulator._order_medium.count(slow_consumer.name) == 1
    assert slow_consumer.name not in simulator._order_slow
    assert slow_consumer.name in simulator._groups["medium"]
    assert slow_consumer.name not in simulator._groups["slow"]
    assert slow_consumer.call_count == 1


def test_substep_count_divides_dt() -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0})
    substep_functor = RecordingFunctor(
        name="substep_functor",
        execution_group="fast",
        update_var="test_signal",
        update_value=0.0,
        substep_count=4,
    )
    simulator.register_functor(substep_functor)

    simulator.step()

    assert substep_functor.call_count == 4
    assert substep_functor.dt_calls == [15.0, 15.0, 15.0, 15.0]


def test_flux_update_is_integrated_exactly_once() -> None:
    simulator = _declared_simulator({"time_step": 10.0})
    _initialize_shared_x(simulator, 1.0)
    functor = ExplicitUpdateFunctor(
        name="flux_integrator",
        update_var="shared_x",
        mode=UpdateMode.FLUX,
        update_value=2.0,
    )
    simulator.register_functor(functor)

    updates = _execute_group_with_step_tracking(simulator, functor.name, 10.0)

    assert simulator.state_manager.get_state("shared_x").item() == pytest.approx(21.0)
    assert isinstance(updates["shared_x"], UpdateItem)
    assert updates["shared_x"].mode == UpdateMode.ACCUMULATE
    assert updates["shared_x"].per_timestep is True
    assert updates["shared_x"].value.item() == pytest.approx(20.0)


def test_substeps_advance_state_and_time_sequentially() -> None:
    simulator = _declared_simulator({"time_step": 2.0})
    _initialize_shared_x(simulator, 1.0)
    functor = ExplicitUpdateFunctor(
        name="stateful_substeps",
        update_var="shared_x",
        mode=UpdateMode.ACCUMULATE,
        substep_count=2,
        use_state_value=True,
    )
    simulator.register_functor(functor)

    _execute_group_with_step_tracking(simulator, functor.name, 2.0)

    assert simulator.state_manager.get_state("shared_x").item() == pytest.approx(4.0)
    assert functor.time_calls == [0.0, 1.0]


def test_substep_flux_uses_substep_dt_once() -> None:
    simulator = _declared_simulator({"time_step": 10.0})
    _initialize_shared_x(simulator, 1.0)
    functor = ExplicitUpdateFunctor(
        name="substep_flux_integrator",
        update_var="shared_x",
        mode=UpdateMode.FLUX,
        update_value=2.0,
        substep_count=2,
    )
    simulator.register_functor(functor)

    _execute_group_with_step_tracking(simulator, functor.name, 10.0)

    assert simulator.state_manager.get_state("shared_x").item() == pytest.approx(21.0)


@pytest.mark.parametrize("substeps", [1, 2, 4])
def test_substeps_commit_a_per_second_rate_once(substeps: int) -> None:
    """A constant per-second ACCUMULATE rate q commits q * dt for any substep count.

    Each substep returns q over h = dt / n; the merged rate is the time-weighted
    mean sum_k q h / dt = q, which StateManager integrates once over dt.
    """
    simulator = _declared_simulator({"time_step": 10.0})
    _initialize_shared_x(simulator, 0.0)
    functor = ExplicitUpdateFunctor(
        name="per_second_rate",
        update_var="shared_x",
        mode=UpdateMode.ACCUMULATE,
        update_value=0.5,
        substep_count=substeps,
        units_by_call=["kg m-2 s-1"],
        basis="per_second",
    )
    simulator.register_functor(functor)

    updates = simulator.step()

    assert functor.call_count == substeps
    assert simulator.get_state()["shared_x"].item() == pytest.approx(5.0)
    assert float(updates["shared_x"]) == pytest.approx(0.5)


@pytest.mark.parametrize("substeps", [1, 2, 4])
def test_substeps_keep_a_boundary_rate_aggregate_as_the_step_rate(substeps: int) -> None:
    """A per-second rate of a ContractRegistry flux variable is stored as the rate.

    StateManager keeps an unintegrated rate of a flux variable (n2o_flux, written
    by nitrification) as the boundary-rate aggregate; over n substeps the stored
    value is the time-weighted mean q, not n q.
    """
    simulator = _declared_simulator(
        {"time_step": 10.0, "state_units": {"n2o_flux": "g N m-2 s-1"}}
    )
    simulator.initialize_state({"n2o_flux": 0.0})
    functor = ExplicitUpdateFunctor(
        name="nitrification",
        update_var="n2o_flux",
        mode=UpdateMode.ACCUMULATE,
        update_value=0.5,
        substep_count=substeps,
        units_by_call=["g N m-2 s-1"],
        basis="per_second",
    )
    simulator.register_functor(functor)

    simulator.step()

    assert simulator.state_manager._contract_registry.get_contract("n2o_flux").kind == "flux"
    assert simulator.get_state()["n2o_flux"].item() == pytest.approx(0.5)


def test_substeps_advance_a_per_second_rate_by_the_substep_interval() -> None:
    """dx/dt = x (1 s-1) with two substeps of h = 0.5 s (explicit Euler).

    The second substep sees x0 + q1 h = 1.5, not x0 + q1; the committed state
    is x0 (1 + h)^2 = 2.25.
    """
    simulator = _declared_simulator({"time_step": 1.0})
    _initialize_shared_x(simulator, 1.0)
    functor = ExplicitUpdateFunctor(
        name="state_dependent_rate",
        update_var="shared_x",
        mode=UpdateMode.ACCUMULATE,
        substep_count=2,
        use_state_value=True,
        units_by_call=["kg m-2 s-1"],
        basis="per_second",
    )
    simulator.register_functor(functor)

    simulator.step()

    assert functor.seen_values == pytest.approx([1.0, 1.5])
    assert simulator.get_state()["shared_x"].item() == pytest.approx(2.25)


def test_later_functor_of_the_group_sees_the_integrated_per_second_rate() -> None:
    """A reader after a per-second rate writer in one group sees x0 + q dt."""
    simulator = _declared_simulator({"time_step": 10.0})
    simulator.initialize_state({"shared_x": 0.0, "shared_y": 0.0})
    simulator.state_manager.register_variable(
        StateVariableSpec(name="shared_x", units="kg m-2", conserved=False, required=False)
    )
    writer = ExplicitUpdateFunctor(
        name="group_rate_writer",
        update_var="shared_x",
        mode=UpdateMode.ACCUMULATE,
        update_value=0.5,
        units_by_call=["kg m-2 s-1"],
        basis="per_second",
    )
    reader = StateReaderFunctor("group_rate_reader", "shared_x", after="group_rate_writer")
    simulator.register_functor(writer)
    simulator.register_functor(reader)
    assert list(simulator._order_fast) == ["group_rate_writer", "group_rate_reader"]

    simulator.step()

    assert reader.seen == pytest.approx([5.0])
    assert simulator.get_state()["shared_x"].item() == pytest.approx(5.0)


def test_update_item_metadata_survives_execution_group() -> None:
    simulator = _declared_simulator({"time_step": 1.0})
    _initialize_shared_x(simulator, 1.0)
    functor = ExplicitUpdateFunctor(
        name="metadata_preservation",
        update_var="shared_x",
        mode=UpdateMode.ACCUMULATE,
        update_value=2.0,
        substep_count=2,
    )
    simulator.register_functor(functor)

    updates = _execute_group_with_step_tracking(simulator, functor.name, 1.0)

    item = updates["shared_x"]
    assert isinstance(item, UpdateItem)
    assert item.mode == UpdateMode.ACCUMULATE
    assert item.write_mode == StateWriteMode.MUTABLE
    assert item.unit == "kg m-2"
    assert item.basis == "per_step"
    assert item.per_timestep is True
    assert item.element is None
    assert item.source_functor == functor.name


def test_group_commit_appends_provenance_records_in_place() -> None:
    """A group commit adds its writer records to the recorded history without copying it."""
    simulator = _declared_simulator({"time_step": 1.0})
    _initialize_shared_x(simulator, 1.0)
    functor = ExplicitUpdateFunctor(
        name="provenance_appender",
        update_var="shared_x",
        mode=UpdateMode.ACCUMULATE,
        update_value=2.0,
    )
    simulator.register_functor(functor)
    _execute_group_with_step_tracking(simulator, functor.name, 1.0)
    detail = simulator._step_provenance_detail
    records = detail["shared_x"]
    first_record = records[0]
    assert len(records) == 1

    simulator._execute_group([functor.name], 1.0, start_time=simulator.current_time)

    assert simulator._step_provenance_detail is detail
    assert detail["shared_x"] is records
    assert len(records) == 2
    assert records[0] is first_record
    assert records[1][0] == functor.name


def test_substeps_reject_incompatible_update_metadata() -> None:
    simulator = _declared_simulator({"time_step": 2.0})
    simulator.initialize_state({"shared_x": 1.0})
    functor = ExplicitUpdateFunctor(
        name="incompatible_substeps",
        update_var="shared_x",
        mode=UpdateMode.ACCUMULATE,
        update_value=1.0,
        substep_count=2,
        units_by_call=["kg m-2", "mol m-2"],
    )
    simulator.register_functor(functor)

    with pytest.raises(RuntimeError, match="Group execution failed") as exc_info:
        _execute_group_with_step_tracking(simulator, functor.name, 2.0)

    assert "Incompatible UpdateItem metadata" in str(exc_info.value.__cause__)


def test_substep_diagnostics_do_not_advance_state_and_use_declared_reduction():
    simulator = _declared_simulator({"device": "cpu", "conservation_checks": False})
    simulator.initialize_state({"shared_x": 1.0, "shared_y": 0.0})
    functor = DiagnosticSubstepFunctor()
    simulator.register_functor(functor)

    updates = _execute_group_with_step_tracking(simulator, functor.name, 2.0)

    assert functor.seen_x == [1.0, 1.0]
    assert simulator.get_state()["shared_x"].item() == 1.0
    assert simulator.get_state()["shared_y"].item() == 2.0
    assert updates["diagnostics.sample"].item() == 2.0
    assert isinstance(updates["diagnostics.accumulated"], UpdateItem)
    assert updates["diagnostics.accumulated"].value.item() == 4.0
    assert updates["diagnostics.accumulated"].unit == "dimensionless"
    assert updates["shared_x"].value.item() == 100.0
    updates["diagnostics.sample"].backward()
    assert functor.diagnostic_gain.grad.item() == 2.0


def test_substep_intents_are_collected_in_order_without_dependency_leakage():
    simulator = _declared_simulator({"device": "cpu", "conservation_checks": False})
    simulator.initialize_state({"shared_x": 1.0, "shared_y": 0.0, "shared_z": 0.0})
    producer = DiagnosticSubstepFunctor(emit_intents=True)
    consumer = DependencyOutputProbe(producer.name)
    simulator.register_functor(producer)
    simulator.register_functor(consumer)
    simulator._active_step_external_intents = []

    updates = simulator._execute_group([producer.name, consumer.name], 2.0, start_time=0.0)

    records = simulator._active_step_external_intents
    assert [record["intent"].request_id for record in records] == ["substep-1", "substep-2"]
    assert [record["substep"] for record in records] == [0, 1]
    assert [record["time"] for record in records] == [0.0, 1.0]
    assert all(record["functor"] == producer.name for record in records)
    assert "export" not in consumer.received_keys
    assert "diagnostics.sample" in consumer.received_keys
    assert "export" not in updates
    assert simulator.get_state()["shared_z"].item() == 2.0


def test_intents_without_an_accepted_step_owner_fail_before_state_commit():
    simulator = _declared_simulator({"device": "cpu", "conservation_checks": False})
    simulator.initialize_state({"shared_x": 1.0, "shared_y": 0.0})
    functor = DiagnosticSubstepFunctor(emit_intents=True)
    simulator.register_functor(functor)

    with pytest.raises(RuntimeError, match="accepted-step"):
        simulator._execute_group([functor.name], 2.0, start_time=0.0)
    assert simulator.get_state()["shared_y"].item() == 0.0


def test_failed_group_does_not_publish_intents_or_provisional_state():
    from relic.core.functor import ExecutionError

    simulator = _declared_simulator({"device": "cpu", "conservation_checks": False})
    simulator.initialize_state({"shared_x": 1.0, "shared_y": 0.0, "shared_z": 0.0})
    producer = DiagnosticSubstepFunctor(emit_intents=True)
    consumer = DependencyOutputProbe(producer.name, fail=True)
    simulator.register_functor(producer)
    simulator.register_functor(consumer)
    previous = {"previous_group": True}
    simulator._active_step_external_intents = [previous]
    earlier_record = ("earlier_writer", UpdateMode.REPLACE, None, "writer", None)
    recorded = [earlier_record]
    simulator._step_provenance_detail = {"shared_y": recorded}

    with pytest.raises((RuntimeError, ExecutionError)) as error:
        simulator._execute_group([producer.name, consumer.name], 2.0, start_time=0.0)
    assert "injected late group failure" in str(error.value) + str(error.value.__cause__)
    assert simulator._active_step_external_intents == [previous]
    # The producer ran before the failure; its writer record must not be published.
    assert list(simulator._step_provenance_detail) == ["shared_y"]
    assert simulator._step_provenance_detail["shared_y"] is recorded
    assert recorded == [earlier_record]
    assert simulator.get_state()["shared_x"].item() == 1.0
    assert simulator.get_state()["shared_y"].item() == 0.0


@pytest.mark.parametrize("existing", [False, True])
def test_provisional_state_preserves_small_float64_values_and_gradient(existing):
    from relic.core.simulator.execution_group import _build_provisional_state_and_converted_updates

    simulator = _declared_simulator({"device": "cpu", "conservation_checks": False})
    simulator.initialize_state({})
    value = torch.tensor(1.0e-50, dtype=torch.float64, requires_grad=True)
    state = {"precision_state": torch.tensor(0.0, dtype=torch.float64)} if existing else {}
    provisional, updates = _build_provisional_state_and_converted_updates(
        simulator, state,
        {"precision_state": UpdateItem(value=value, mode=UpdateMode.REPLACE)}, {}, dt=1.0,
    )
    result = provisional["precision_state"]
    assert result.dtype == torch.float64
    assert result.item() == value.item()
    assert updates["precision_state"]["item"].value.dtype == torch.float64
    assert torch.autograd.grad(result, value)[0].item() == 1.0


def test_undeclared_numeric_output_is_not_relabelled_as_diagnostic():
    simulator = _declared_simulator({"device": "cpu", "conservation_checks": False})
    simulator.initialize_state({"shared_x": 1.0, "shared_y": 0.0})
    functor = DiagnosticSubstepFunctor(undeclared_output=True)
    simulator.register_functor(functor)

    with pytest.raises(RuntimeError) as error:
        simulator._execute_group([functor.name], 2.0, start_time=0.0)
    assert "Undeclared scientific output" in str(error.value.__cause__)
    assert simulator.get_state()["shared_y"].item() == 0.0


# --------------------------------------------------------------------------- scheduler kernel contracts (D79-D85)


def test_event_trigger_failure_is_not_swallowed() -> None:
    """A trigger that raises must fail the step (fail-fast), not silently skip the group."""
    simulator = _build_initialized_simulator(
        {"time_step": 60.0, "medium_interval_seconds": 1.0e9}
    )
    medium = RecordingFunctor(
        name="event_medium_raising",
        execution_group="medium",
        update_var="test_signal",
        update_value=0.0,
    )
    simulator.register_functor(medium)

    def broken_trigger(state, current_time):
        raise KeyError("missing_driver")

    simulator.register_event_trigger("medium", broken_trigger)

    with pytest.raises(RuntimeError, match=r"Event trigger .*broken_trigger.* group 'medium'"):
        simulator.step()

    assert medium.call_count == 0
    assert simulator.current_time == 0.0


def test_strang_splitting_is_a_palindromic_composition_for_three_functors() -> None:
    """Phi_1(h/2) Phi_2(h/2) Phi_3(h) Phi_2(h/2) Phi_1(h/2): every operator is split symmetrically."""
    simulator = _build_initialized_simulator({"time_step": 60.0, "splitting_mode": "strang"})
    functors = [
        RecordingFunctor(name=f"split_{tag}", execution_group="fast", update_var=var, update_value=0.0)
        for tag, var in (("a", "test_signal"), ("b", "shared_x"), ("c", "shared_y"))
    ]
    simulator.initialize_state({"test_signal": 300.0, "shared_x": 0.0, "shared_y": 0.0})
    for functor in functors:
        simulator.register_functor(functor)

    executed: List[tuple] = []
    original_execute_group = simulator._execute_group

    def recording_execute_group(group_order, dt, *, start_time, **kwargs):
        executed.append((tuple(group_order), float(dt), float(start_time)))
        return original_execute_group(group_order, dt, start_time=start_time, **kwargs)

    simulator._execute_group = recording_execute_group  # type: ignore[method-assign]
    simulator.step()

    by_name = {functor.name: functor for functor in functors}
    order = list(simulator._order_fast)
    assert len(order) == 3
    first, middle, last = (by_name[name] for name in order)
    assert first.dt_calls == [30.0, 30.0]
    assert middle.dt_calls == [30.0, 30.0]
    assert last.dt_calls == [60.0]
    # Palindromic call sequence: first(h/2), middle(h/2), last(h), middle(h/2), first(h/2);
    # the second half-steps cover [t + h/2, t + h].
    assert executed == [
        ((first.name,), 30.0, 0.0),
        ((middle.name,), 30.0, 0.0),
        ((last.name,), 60.0, 0.0),
        ((middle.name,), 30.0, 30.0),
        ((first.name,), 30.0, 30.0),
    ]
    assert simulator.current_time == 60.0


def test_strang_second_half_steps_start_at_t_plus_half_step() -> None:
    """Each (time, dt) pair is the real interval a sub-flow accounts for.

    Palindrome over [t, t + h]: first half-steps and the middle full step start at
    t; the second half-steps cover [t + h/2, t + h] and therefore start at t + h/2.
    """
    simulator = _build_initialized_simulator({"time_step": 60.0, "splitting_mode": "strang"})
    simulator.initialize_state({"test_signal": 300.0, "shared_x": 0.0, "shared_y": 0.0})
    functors = [
        RecordingFunctor(name=f"timed_{tag}", execution_group="fast", update_var=var, update_value=0.0)
        for tag, var in (("a", "test_signal"), ("b", "shared_x"), ("c", "shared_y"))
    ]
    for functor in functors:
        simulator.register_functor(functor)

    simulator.step()
    simulator.step()

    by_name = {functor.name: functor for functor in functors}
    first, middle, last = (by_name[name] for name in simulator._order_fast)
    assert first.time_calls == [0.0, 30.0, 60.0, 90.0]
    assert middle.time_calls == [0.0, 30.0, 60.0, 90.0]
    assert last.time_calls == [0.0, 60.0]
    assert first.dt_calls == [30.0] * 4
    assert last.dt_calls == [60.0, 60.0]


def test_group_call_requires_an_explicit_finite_start_time() -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0})
    functor = RecordingFunctor(name="start_time_guard", execution_group="fast", update_var="test_signal")
    simulator.register_functor(functor)

    with pytest.raises(TypeError):
        simulator._execute_group([functor.name], 60.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="finite"):
        simulator._execute_group([functor.name], 60.0, start_time=float("nan"))
    with pytest.raises(TypeError, match="real scalar"):
        simulator._execute_group([functor.name], 60.0, start_time=True)
    assert functor.call_count == 0


def test_lie_sub_flows_all_start_at_step_start() -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0})
    functor_a = RecordingFunctor(name="lie_time_a", execution_group="fast", update_var="test_signal")
    functor_b = RecordingFunctor(name="lie_time_b", execution_group="fast", update_var="shared_x")
    simulator.register_functor(functor_a)
    simulator.register_functor(functor_b)

    simulator.step()
    simulator.step()

    assert functor_a.time_calls == [0.0, 60.0]
    assert functor_b.time_calls == [0.0, 60.0]


# Rotation test problem for splitting-order regression (scheduler contract, 4.4.1).
# References: Strang (1968) SIAM J. Numer. Anal. 5:506-517; Hairer, Lubich and
#   Wanner (2006) Geometric Numerical Integration, Sect. II.5 and III.3.
# Mathematical expression: y = (u - U0, v - V0), y' = c(t) (M1 + M2) y with
#   M1 = [[0, 1], [0, 0]] (shear of u by v) and M2 = [[0, 0], [-1, 0]] (shear of v
#   by u), c(t) = 1 + A sin(w t); [M1, M2] != 0. Exact solution: rotation by
#   C(t) = int_0^t c(s) ds = t + A (1 - cos(w t)) / w. Each sub-flow is exact over
#   [t0, t1]: u += C(t0, t1) (v - V0) and v -= C(t0, t1) (u - U0); the explicit
#   Euler variant replaces C(t0, t1) by (t1 - t0) c(t0).
# Parameter meanings: U0 = V0 = 5 shift the rotation centre inside the generic
#   state bounds [0, 1000]; A = 0.5 and w = 3 make the flow non-autonomous.
# Units: t, dt [s]; u, v dimensionless; w [s-1].
# Domain/constraints: |y| = 1 keeps u, v in [4, 6]; the error is measured at T.
# Sign convention: positive C rotates (u - U0, v - V0) clockwise.
_ROTATION_CENTRE = 5.0
_ROTATION_AMPLITUDE = 0.5
_ROTATION_FREQUENCY = 3.0


def _rotation_weight_integral(t0: float, t1: float) -> float:
    return (t1 - t0) + _ROTATION_AMPLITUDE * (
        math.cos(_ROTATION_FREQUENCY * t0) - math.cos(_ROTATION_FREQUENCY * t1)
    ) / _ROTATION_FREQUENCY


def _rotation_weight_euler(t0: float, t1: float) -> float:
    return (t1 - t0) * (1.0 + _ROTATION_AMPLITUDE * math.sin(_ROTATION_FREQUENCY * t0))


class ShearSubFlowFunctor(BiologicalProcessFunctor):
    """One linear non-autonomous shear sub-flow over its own [time, time + dt]."""

    def __init__(self, name: str, target: str, source: str, sign: float, exact: bool) -> None:
        metadata = FunctorMetadata(
            name=name,
            description=f"Shear sub-flow {name}",
            references=["Strang (1968) SIAM J. Numer. Anal. 5:506-517"],
            temporal_scale=TemporalScale.SECOND,
            input_variables=[source],
            output_variables=[target],
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.execution_group = "fast"
        self.target = target
        self.source = source
        self.sign = float(sign)
        self.exact = exact

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, UpdateItem]:
        t0 = float(time)
        t1 = t0 + float(dt.item())
        weight = _rotation_weight_integral(t0, t1) if self.exact else _rotation_weight_euler(t0, t1)
        return {
            self.target: UpdateItem(
                value=self.sign * weight * (state[self.source] - _ROTATION_CENTRE),
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="dimensionless",
                source_functor=self.name,
            )
        }


def _rotation_error(splitting_mode: str, exact: bool, n_steps: int, horizon: float = 2.0) -> float:
    step_dt = horizon / n_steps
    simulator = _declared_simulator(
        {"time_step": step_dt, "splitting_mode": splitting_mode, "device": "cpu", "enable_auto_scale": False}
    )
    simulator.initialize_state(
        {
            "osc_u": torch.tensor(_ROTATION_CENTRE + 1.0, dtype=torch.float64),
            "osc_v": torch.tensor(_ROTATION_CENTRE, dtype=torch.float64),
        }
    )
    simulator.register_functor(ShearSubFlowFunctor("shear_u", "osc_u", "osc_v", 1.0, exact))
    simulator.register_functor(ShearSubFlowFunctor("shear_v", "osc_v", "osc_u", -1.0, exact))
    assert list(simulator._order_fast) == ["shear_u", "shear_v"]
    for _ in range(n_steps):
        simulator.step(step_dt)
    angle = _rotation_weight_integral(0.0, simulator.current_time)
    state = simulator.get_state()
    return math.hypot(
        float(state["osc_u"]) - (_ROTATION_CENTRE + math.cos(angle)),
        float(state["osc_v"]) - (_ROTATION_CENTRE - math.sin(angle)),
    )


@pytest.mark.parametrize(
    ("splitting_mode", "exact", "expected_ratio"),
    [("strang", True, 4.0), ("lie", True, 2.0), ("strang", False, 2.0)],
    ids=["strang_exact_sub_flows", "lie_exact_sub_flows", "strang_explicit_euler_sub_flows"],
)
def test_fast_group_splitting_convergence_order(splitting_mode, exact, expected_ratio) -> None:
    """Global error ratio on step halving through the real fast group.

    Exact non-autonomous sub-flows: Strang is second order (ratio 4) only when the
    second half-steps start at t + h/2; Lie is first order (ratio 2). Explicit
    Euler sub-flows keep Strang first order (ratio 2).
    """
    coarse = _rotation_error(splitting_mode, exact, 32)
    fine = _rotation_error(splitting_mode, exact, 64)
    assert abs(coarse / fine - expected_ratio) <= 0.3


_SCHEDULER_LOGGER = "relic.core.simulator.execution_loop"


def _declared_fast_simulator(splitting_mode: str, accuracies: List[str]) -> RELICSimulator:
    """Register one fast Functor per declared sub-flow accuracy (in order)."""
    from relic.core.functor.functor_types import SubFlowAccuracy

    simulator = _build_initialized_simulator({"time_step": 60.0, "splitting_mode": splitting_mode})
    simulator.initialize_state({"test_signal": 300.0, "shared_x": 0.0, "shared_y": 0.0})
    targets = ("test_signal", "shared_x", "shared_y")
    for index, accuracy in enumerate(accuracies):
        functor = RecordingFunctor(
            name=f"declared_{index}", execution_group="fast", update_var=targets[index]
        )
        functor.metadata.sub_flow_accuracy = SubFlowAccuracy(accuracy)
        simulator.register_functor(functor)
    return simulator


@pytest.mark.parametrize(
    ("splitting_mode", "accuracies", "expected_order", "expected_limit"),
    [
        ("strang", ["exact", "exact"], 2, 2),
        ("strang", ["exact", "order_2", "exact"], 2, 2),
        ("strang", ["exact", "order_1"], 1, 2),
        ("strang", ["exact", "undeclared"], "unverified", 2),
        ("lie", ["exact", "exact"], 1, 1),
        ("lie", ["order_1", "undeclared"], "unverified", 1),
        ("strang", ["exact"], "exact", None),
        ("strang", ["order_2"], 2, None),
    ],
)
def test_splitting_order_report_binds_order_to_sub_integrators(
    splitting_mode, accuracies, expected_order, expected_limit
) -> None:
    """lie: min(1, p_i); strang: min(2, p_i) with EXACT = inf; undeclared -> unverified.

    With one fast Functor there is no splitting error and the order is p_1.
    """
    simulator = _declared_fast_simulator(splitting_mode, accuracies)

    report = simulator.get_splitting_order_report()

    assert report["splitting_mode"] == splitting_mode
    assert report["fast_order"] == tuple(simulator._order_fast)
    assert report["splitting_order_limit"] == expected_limit
    assert report["effective_order"] == expected_order
    assert report["sub_flow_accuracy"] == {
        f"declared_{index}": accuracy for index, accuracy in enumerate(accuracies)
    }
    assert report["undeclared"] == tuple(
        f"declared_{index}" for index, accuracy in enumerate(accuracies) if accuracy == "undeclared"
    )
    assert report["epoch"] == simulator.provider_epoch


def test_splitting_order_report_is_read_only_and_empty_without_fast_functors() -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0, "splitting_mode": "strang"})
    report = simulator.get_splitting_order_report()
    assert report["effective_order"] is None
    assert report["fast_order"] == ()

    simulator = _declared_fast_simulator("strang", ["exact", "exact"])
    first = simulator.get_splitting_order_report()
    first["effective_order"] = 1
    first["sub_flow_accuracy"]["declared_0"] = "order_1"
    assert simulator.get_splitting_order_report()["effective_order"] == 2
    assert simulator.get_splitting_order_report()["sub_flow_accuracy"]["declared_0"] == "exact"
    assert not hasattr(simulator, "set_splitting_order_report")


def test_strang_publication_warns_once_when_order_is_below_two_or_unverified(caplog) -> None:
    with caplog.at_level("WARNING", logger=_SCHEDULER_LOGGER):
        _declared_fast_simulator("strang", ["exact", "order_1"])
    below_two = [record for record in caplog.records if "Strang splitting" in record.getMessage()]
    # Publication 1 holds one exact sub-flow (order exact); publication 2 has order 1.
    assert len(below_two) == 1
    assert "effective order 1" in below_two[0].getMessage()
    assert "declared_1" in below_two[0].getMessage()

    caplog.clear()
    with caplog.at_level("WARNING", logger=_SCHEDULER_LOGGER):
        _declared_fast_simulator("strang", ["exact", "undeclared"])
    unverified = [record for record in caplog.records if "Strang splitting" in record.getMessage()]
    assert len(unverified) == 1
    assert "unverified" in unverified[0].getMessage()

    for mode, accuracies in (("strang", ["exact", "order_2"]), ("lie", ["order_1", "undeclared"])):
        caplog.clear()
        with caplog.at_level("WARNING", logger=_SCHEDULER_LOGGER):
            _declared_fast_simulator(mode, accuracies)
        assert not [record for record in caplog.records if "Strang splitting" in record.getMessage()]


def test_strang_order_warning_is_emitted_only_when_the_verdict_changes(caplog) -> None:
    """Many registrations do not flood the log; the report stays the source of truth."""
    from relic.core.functor import SubFlowAccuracy

    simulator = _build_initialized_simulator({"time_step": 60.0, "splitting_mode": "strang"})
    names = [f"flood_{index}" for index in range(6)]
    simulator.initialize_state({"test_signal": 300.0, **{name: 0.0 for name in names}})
    with caplog.at_level("WARNING", logger=_SCHEDULER_LOGGER):
        for name in names:
            simulator.register_functor(RecordingFunctor(name=name, execution_group="fast", update_var=name))
    warnings = [record for record in caplog.records if "Strang splitting" in record.getMessage()]
    assert len(warnings) == 1
    report = simulator.get_splitting_order_report()
    assert report["effective_order"] == "unverified"
    assert report["undeclared"] == tuple(names)

    for name in names:
        simulator.unregister_functor(name)
    caplog.clear()
    with caplog.at_level("WARNING", logger=_SCHEDULER_LOGGER):
        exact = RecordingFunctor(name="flood_exact", execution_group="fast", update_var="flood_0")
        exact.metadata.sub_flow_accuracy = SubFlowAccuracy.EXACT
        simulator.register_functor(exact)  # verdict exact: no warning
        for index in (1, 2, 3):
            first_order = RecordingFunctor(
                name=f"flood_euler_{index}", execution_group="fast", update_var=f"flood_{index}"
            )
            first_order.metadata.sub_flow_accuracy = SubFlowAccuracy.ORDER_1
            simulator.register_functor(first_order)  # verdict 1 once, then unchanged
        simulator.unregister_functor("flood_euler_1")  # verdict still 1
        for index in (2, 3):
            simulator.unregister_functor(f"flood_euler_{index}")  # verdict back to exact
        again = RecordingFunctor(name="flood_euler_4", execution_group="fast", update_var="flood_4")
        again.metadata.sub_flow_accuracy = SubFlowAccuracy.ORDER_1
        simulator.register_functor(again)  # verdict changes to 1 again
    warnings = [record.getMessage() for record in caplog.records if "Strang splitting" in record.getMessage()]
    assert len(warnings) == 2
    assert all("effective order 1" in message for message in warnings)
    assert simulator.get_splitting_order_report()["effective_order"] == 1


@pytest.mark.parametrize(
    ("splitting_mode", "exact", "declared", "expected_order"),
    [("strang", True, "exact", 2), ("lie", True, "exact", 1), ("strang", False, "order_1", 1)],
)
def test_splitting_order_report_matches_measured_rotation_order(
    splitting_mode, exact, declared, expected_order
) -> None:
    """The declared-order report agrees with the measured order of the rotation problem."""
    from relic.core.functor.functor_types import SubFlowAccuracy

    simulator = _declared_simulator({"time_step": 0.1, "splitting_mode": splitting_mode, "device": "cpu"})
    simulator.initialize_state({"osc_u": 6.0, "osc_v": 5.0})
    for functor in (
        ShearSubFlowFunctor("shear_u", "osc_u", "osc_v", 1.0, exact),
        ShearSubFlowFunctor("shear_v", "osc_v", "osc_u", -1.0, exact),
    ):
        functor.metadata.sub_flow_accuracy = SubFlowAccuracy(declared)
        simulator.register_functor(functor)

    assert simulator.get_splitting_order_report()["effective_order"] == expected_order


def test_negative_effective_dt_is_rejected_not_masked() -> None:
    """Inconsistent last-execution bookkeeping must raise instead of falling back to dt."""
    simulator = _build_initialized_simulator(
        {"time_step": 60.0, "medium_interval_seconds": 1.0e9}
    )
    medium = RecordingFunctor(
        name="event_medium_bad_bookkeeping",
        execution_group="medium",
        update_var="test_signal",
        update_value=0.0,
    )
    simulator.register_functor(medium)
    simulator.register_event_trigger("medium", lambda state, t: True)
    simulator._last_medium_execution_time = 1.0e9

    with pytest.raises(RuntimeError, match=r"effective dt"):
        simulator.step()

    assert medium.call_count == 0
    assert simulator.current_time == 0.0


def test_rate_group_predicates_are_pure_until_committed() -> None:
    """Probing a boundary must not consume it; the commit is an explicit separate call."""
    simulator = _build_initialized_simulator(
        {"time_step": 60.0, "medium_interval_seconds": 60.0}
    )
    previous_medium = simulator._previous_medium_index
    assert simulator._should_execute_medium(future_time=60.0) is True
    assert simulator._should_execute_medium(future_time=60.0) is True
    assert simulator._previous_medium_index == previous_medium

    time_ops_module.commit_rate_group_interval(simulator, "medium", 60.0)
    assert simulator._previous_medium_index == previous_medium + 1
    assert simulator._should_execute_medium(future_time=60.0) is False

    simulator._slow_interval_seconds = None
    previous_month = simulator._previous_month_index
    boundary = 31.0 * float(SECONDS_PER_DAY) + 60.0
    assert simulator._should_execute_slow(future_time=boundary) is True
    assert simulator._should_execute_slow(future_time=boundary) is True
    assert simulator._previous_month_index == previous_month

    time_ops_module.commit_rate_group_interval(simulator, "slow", boundary)
    assert simulator._previous_month_index > previous_month
    assert simulator._should_execute_slow(future_time=boundary) is False


def test_single_adaptive_controller_surface() -> None:
    """The simulator's update-magnitude heuristic is the only step controller in the code base."""
    import importlib.util
    from pathlib import Path

    import relic.computation.solvers as solvers_pkg
    import relic.coupling.scales.temporal as temporal_pkg

    assert not hasattr(temporal_pkg, "AdaptiveTimestepController")
    assert not hasattr(temporal_pkg, "CFLChecker")
    assert not hasattr(solvers_pkg, "AdaptiveTimeStepper")
    assert importlib.util.find_spec("relic.coupling.scales.temporal.adaptive_timestepper") is None
    scales_yaml = Path(temporal_pkg.__file__).resolve().parents[3] / "parameters" / "database" / "relic" / "coupling" / "scales_parameters.yaml"
    assert "coupling.scales.temporal.adaptive." not in scales_yaml.read_text(encoding="utf-8")


def test_cross_group_promotion_is_limited_to_slow_consumers() -> None:
    """Only slow consumers are moved to the medium group; medium consumers of fast producers stay put."""
    simulator = _build_initialized_simulator(
        {
            "time_step": 60.0,
            "medium_interval_seconds": 1.0,
            "resolve_cross_group_deps": True,
        }
    )
    fast_producer = RecordingFunctor(
        name="fast_producer_limited",
        execution_group="fast",
        update_var="shared_x",
        update_value=1.0,
        input_vars=[],
    )
    medium_consumer = RecordingFunctor(
        name="medium_consumer_limited",
        execution_group="medium",
        update_var="test_signal",
        update_value=0.0,
        input_vars=["shared_x"],
    )
    simulator.register_functor(fast_producer)
    simulator.register_functor(medium_consumer)

    simulator.step()

    assert medium_consumer.name in simulator._order_medium
    assert medium_consumer.name not in simulator._order_fast
    assert medium_consumer.call_count == 1


class LaggedConsumerFunctor(RecordingFunctor):
    """Recorder that declares cross-group producers in its dependency contract."""

    def __init__(self, *args: Any, lagged: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lagged = list(lagged or [])

    def get_dependencies(self) -> Dict[str, List[str]]:
        return {"required": [], "optional": [], "lagged": list(self.lagged)}


def _cross_group_pair(
    config: Dict[str, Any],
    producer_group: str,
    consumer_group: str,
    *,
    lagged: bool = False,
) -> RELICSimulator:
    simulator = _build_initialized_simulator({"time_step": 60.0, **config})
    producer = RecordingFunctor(
        name="lag_producer", execution_group=producer_group, update_var="shared_x", update_value=1.0
    )
    consumer = LaggedConsumerFunctor(
        name="lag_consumer",
        execution_group=consumer_group,
        update_var="test_signal",
        input_vars=["shared_x"],
        lagged=[producer.name] if lagged else [],
    )
    simulator.register_functor(producer)
    simulator.register_functor(consumer)
    return simulator


def _lag_messages(caplog, prefix: str) -> List[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith(prefix)]


@pytest.mark.parametrize(
    ("producer_group", "consumer_group", "prefix"),
    [
        ("fast", "slow", "Cross-group sampling lag"),
        ("fast", "medium", "Cross-group sampling lag"),
        ("medium", "slow", "Cross-group sampling lag"),
        ("slow", "fast", "Cross-group hold lag"),
        ("medium", "fast", "Cross-group hold lag"),
        ("slow", "medium", "Cross-group hold lag"),
    ],
)
def test_cross_group_lag_is_warned_in_both_directions(
    caplog, producer_group, consumer_group, prefix
) -> None:
    other = "Cross-group hold lag" if prefix.endswith("sampling lag") else "Cross-group sampling lag"
    with caplog.at_level("WARNING", logger=_SCHEDULER_LOGGER):
        _cross_group_pair({}, producer_group, consumer_group)
    messages = _lag_messages(caplog, prefix)
    assert len(messages) == 1
    assert "'lag_consumer'" in messages[0] and "'lag_producer'" in messages[0]
    assert "shared_x" in messages[0]
    assert not _lag_messages(caplog, other)


@pytest.mark.parametrize(("producer_group", "consumer_group"), [("fast", "slow"), ("slow", "fast")])
def test_lagged_declaration_acknowledges_the_cross_group_lag(caplog, producer_group, consumer_group) -> None:
    with caplog.at_level("WARNING", logger=_SCHEDULER_LOGGER):
        simulator = _cross_group_pair({}, producer_group, consumer_group, lagged=True)
    assert not _lag_messages(caplog, "Cross-group")
    assert simulator._groups[consumer_group] == ["lag_consumer"]


@pytest.mark.parametrize(("producer_group", "consumer_group"), [("fast", "slow"), ("slow", "fast")])
def test_strict_cross_group_dependencies_rejects_unacknowledged_reads(
    producer_group, consumer_group
) -> None:
    with pytest.raises(RuntimeError, match=r"lag_consumer.*lag_producer"):
        _cross_group_pair({"strict_cross_group_dependencies": True}, producer_group, consumer_group)

    simulator = _cross_group_pair(
        {"strict_cross_group_dependencies": True}, producer_group, consumer_group, lagged=True
    )
    assert set(simulator.functors) == {"lag_producer", "lag_consumer"}


def test_strict_cross_group_rejection_rolls_back_the_registration() -> None:
    simulator = _build_initialized_simulator(
        {"time_step": 60.0, "strict_cross_group_dependencies": True}
    )
    producer = RecordingFunctor(name="strict_producer", execution_group="fast", update_var="shared_x")
    consumer = RecordingFunctor(
        name="strict_consumer", execution_group="slow", update_var="test_signal", input_vars=["shared_x"]
    )
    simulator.register_functor(producer)
    epoch = simulator.provider_epoch

    with pytest.raises(RuntimeError, match="strict_cross_group_dependencies"):
        simulator.register_functor(consumer)

    assert set(simulator.functors) == {producer.name}
    assert simulator._groups == {"fast": [producer.name], "medium": [], "slow": []}
    assert simulator.provider_epoch == epoch
    assert simulator.get_lifecycle_receipts()[-1].status == "rolled_back"


def test_strict_cross_group_dependencies_requires_a_bool() -> None:
    with pytest.raises(TypeError, match="strict_cross_group_dependencies"):
        _cross_group_pair({"strict_cross_group_dependencies": "false"}, "fast", "slow")


def test_promoted_slow_consumer_is_integrated_once_per_interval() -> None:
    """A promoted accumulating consumer must not be integrated by medium and slow in one step."""
    simulator = _build_initialized_simulator(
        {
            "time_step": 60.0,
            "medium_interval_seconds": 60.0,
            "slow_interval_seconds": 120.0,
            "resolve_cross_group_deps": True,
        }
    )
    producer = RecordingFunctor(name="acc_producer", execution_group="fast", update_var="shared_x")
    consumer = RecordingFunctor(
        name="acc_consumer",
        execution_group="slow",
        update_var="test_signal",
        update_value=1.0,
        input_vars=["shared_x"],
    )
    simulator.register_functor(producer)
    simulator.register_functor(consumer)
    before = float(simulator.get_state()["test_signal"])

    simulator.step()
    simulator.step()  # medium and slow boundaries both fire at t_future = 120 s

    assert consumer.call_count == 2
    assert sum(consumer.dt_calls) == pytest.approx(120.0)
    assert consumer.time_calls == [0.0, 60.0]
    assert float(simulator.get_state()["test_signal"]) == pytest.approx(before + 2.0)


def test_promotion_moves_membership_order_identity_and_bootstrap_mark(caplog) -> None:
    simulator = _build_initialized_simulator(
        {"time_step": 60.0, "medium_interval_seconds": 1.0e9, "resolve_cross_group_deps": True}
    )
    consumer = RecordingFunctor(
        name="late_consumer", execution_group="slow", update_var="test_signal", input_vars=["shared_x"]
    )
    consumer.force_initial_execution = True
    simulator.register_functor(consumer)
    assert simulator._groups["slow"] == [consumer.name]
    assert simulator._bootstrap_groups == {"slow"}

    producer = RecordingFunctor(name="late_producer", execution_group="fast", update_var="shared_x")
    with caplog.at_level("INFO", logger=_SCHEDULER_LOGGER):
        simulator.register_functor(producer)  # the producer's publication promotes the consumer

    promoted = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "INFO" and "late_consumer" in record.getMessage()
        and "medium" in record.getMessage()
    ]
    assert len(promoted) == 1
    view = simulator.get_committed_provider_view()
    assert view.groups["medium"] == (consumer.name,)
    assert view.groups["slow"] == ()
    assert view.order_medium == (consumer.name,)
    assert view.order_slow == ()
    assert view.providers[consumer.name].execution_group == "medium"
    assert simulator._provider_lifecycle.publication_history[-1]["promoted_to_medium"] == (
        consumer.name,
    )
    assert simulator._bootstrap_groups == {"medium"}
    # A sampling lag remains (medium consumer of a fast producer) and is warned.
    assert [
        record for record in caplog.records
        if record.getMessage().startswith("Cross-group sampling lag") and "late_consumer" in record.getMessage()
    ]

    simulator.step()
    assert consumer.call_count == 1


def test_promotion_receipt_matches_the_published_identity() -> None:
    simulator = _cross_group_pair({"resolve_cross_group_deps": True}, "fast", "slow")
    receipt = simulator.get_lifecycle_receipts()[-1]
    identity = simulator.get_committed_provider_view().providers["lag_consumer"]
    assert receipt.provider_name == "lag_consumer"
    assert identity.execution_group == "medium"
    assert receipt.binding_digest == identity.binding_digest
    assert receipt.managed_delta["group"]["name"] == "medium"


def _mid_run_promotion_simulator() -> tuple:
    """Medium member every 60 s, slow consumer every 600 s, auto-promotion on."""
    simulator = _build_initialized_simulator(
        {
            "time_step": 60.0,
            "medium_interval_seconds": 60.0,
            "slow_interval_seconds": 600.0,
            "resolve_cross_group_deps": True,
        }
    )
    simulator.initialize_state({"test_signal": 300.0, "shared_x": 0.0, "shared_y": 0.0})
    medium_member = RecordingFunctor(name="gap_medium", execution_group="medium", update_var="shared_y")
    consumer = RecordingFunctor(
        name="gap_consumer",
        execution_group="slow",
        update_var="test_signal",
        update_value=1.0,
        input_vars=["shared_x"],
    )
    simulator.register_functor(medium_member)
    simulator.register_functor(consumer)
    return simulator, consumer


def _gap_producer() -> RecordingFunctor:
    return RecordingFunctor(name="gap_producer", execution_group="fast", update_var="shared_x")


def test_mid_run_promotion_cannot_leave_an_uncovered_interval() -> None:
    """Moving a scheduled slow member while t_last_slow != t_last_medium fails closed.

    Its slow accounting ends at t_last_slow = 0 s; the medium group would resume at
    t_last_medium = 120 s, so [0 s, 120 s] would never be integrated for it.
    """
    simulator, consumer = _mid_run_promotion_simulator()
    simulator.step()
    simulator.step()
    assert simulator._last_medium_execution_time == 120.0
    assert simulator._last_slow_execution_time == 0.0
    epoch = simulator.provider_epoch
    producer = _gap_producer()

    with pytest.raises(RuntimeError, match=r"gap_consumer.*uncovered interval \[0\.0 s, 120\.0 s\]"):
        simulator.register_functor(producer)

    assert producer.name not in simulator.functors
    assert simulator._groups["slow"] == [consumer.name]
    assert simulator.provider_epoch == epoch
    assert simulator.get_lifecycle_receipts()[-1].status == "rolled_back"
    for _ in range(8):
        simulator.step()
    # The consumer keeps its contiguous slow accounting [0 s, 600 s].
    assert list(zip(consumer.time_calls, consumer.dt_calls)) == [(0.0, 600.0)]


def test_mid_run_promotion_cannot_integrate_an_interval_twice() -> None:
    """Slow ahead of medium: the move would integrate [t_last_medium, t_last_slow] twice."""
    simulator, consumer = _mid_run_promotion_simulator()
    simulator._medium_interval_seconds = 1.0e9
    simulator._slow_interval_seconds = 60.0
    simulator._previous_medium_index = 0
    simulator._previous_slow_index = 0
    simulator.step()
    assert simulator._last_slow_execution_time == 60.0
    assert simulator._last_medium_execution_time == 0.0

    with pytest.raises(RuntimeError, match=r"integrate the interval \[0\.0 s, 60\.0 s\] twice"):
        simulator.register_functor(_gap_producer())
    assert simulator._groups["slow"] == [consumer.name]


def test_promotion_is_allowed_while_both_groups_share_their_last_execution_time() -> None:
    simulator, consumer = _mid_run_promotion_simulator()
    for _ in range(10):
        simulator.step()  # t = 600: medium and slow both last ran at 600 s
    assert simulator._last_medium_execution_time == simulator._last_slow_execution_time == 600.0

    simulator.register_functor(_gap_producer())
    simulator.step()

    assert simulator._groups["medium"] == ["gap_medium", consumer.name]
    assert list(zip(consumer.time_calls, consumer.dt_calls)) == [(0.0, 600.0), (600.0, 60.0)]


def test_new_slow_consumer_registered_mid_run_is_placed_in_medium() -> None:
    """A provider that was never scheduled has no accounting history to leave uncovered."""
    simulator, _ = _mid_run_promotion_simulator()
    simulator.unregister_functor("gap_consumer")
    simulator.register_functor(_gap_producer())
    simulator.step()
    simulator.step()
    newcomer = RecordingFunctor(
        name="gap_newcomer", execution_group="slow", update_var="test_signal", input_vars=["shared_x"]
    )

    simulator.register_functor(newcomer)
    simulator.step()

    assert simulator._groups["medium"] == ["gap_medium", newcomer.name]
    assert list(zip(newcomer.time_calls, newcomer.dt_calls)) == [(120.0, 60.0)]


def test_mid_run_promotion_guard_survives_checkpoint_restore(tmp_path) -> None:
    source, _ = _mid_run_promotion_simulator()
    source.step()
    source.step()
    path = tmp_path / "promotion_gap.pt"
    source.create_checkpoint(path)

    restored, consumer = _mid_run_promotion_simulator()
    restored.restore_checkpoint(path)

    assert restored._last_medium_execution_time == 120.0
    assert restored._last_slow_execution_time == 0.0
    with pytest.raises(RuntimeError, match="uncovered interval"):
        restored.register_functor(_gap_producer())
    assert restored._groups["slow"] == [consumer.name]


# --------------------------------------------------------------------------- fail-fast scheduler owners (MA-21)


def test_group_order_failure_propagates_with_the_group_name(monkeypatch) -> None:
    """A group whose order cannot be computed must fail, never run in registration order."""
    from relic.core.dependency_graph import DependencyGraphResolver

    simulator = _build_initialized_simulator({"time_step": 60.0})
    first = RecordingFunctor(name="order_first", execution_group="fast", update_var="test_signal")
    second = RecordingFunctor(name="order_second", execution_group="fast", update_var="shared_x")
    simulator.register_functor(first)
    simulator.register_functor(second)
    published_order = list(simulator._order_fast)

    def failing_sort(self, handle_optional=True):
        raise ValueError("synthetic sort failure")

    monkeypatch.setattr(DependencyGraphResolver, "topological_sort", failing_sort)
    simulator._groups_dirty = True
    with pytest.raises(ValueError, match="synthetic sort failure") as error:
        simulator._compute_group_execution_orders()
    assert any("'fast'" in note for note in getattr(error.value, "__notes__", []))
    assert simulator._order_fast == published_order
    assert simulator._groups_dirty is True

    third = RecordingFunctor(name="order_third", execution_group="fast", update_var="shared_y")
    epoch = simulator.provider_epoch
    with pytest.raises(ValueError, match="synthetic sort failure"):
        simulator.register_functor(third)
    assert third.name not in simulator.functors
    assert simulator.provider_epoch == epoch


@pytest.mark.parametrize("returned", [None, "yes", 1.0])
def test_event_trigger_type_error_names_the_accepted_types(returned) -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0, "medium_interval_seconds": 1.0e9})
    simulator.register_functor(
        RecordingFunctor(name="typed_trigger_medium", execution_group="medium", update_var="test_signal")
    )
    simulator.register_event_trigger("medium", lambda state, t: returned)

    with pytest.raises(RuntimeError, match=r"returned .*; bool or int is required"):
        simulator.step()


@pytest.mark.parametrize("returned", [True, 1, 0, False])
def test_event_trigger_accepts_bool_and_int(returned) -> None:
    simulator = _build_initialized_simulator({"time_step": 60.0, "medium_interval_seconds": 1.0e9})
    medium = RecordingFunctor(name="int_trigger_medium", execution_group="medium", update_var="test_signal")
    simulator.register_functor(medium)
    simulator.register_event_trigger("medium", lambda state, t: returned)

    simulator.step()

    assert medium.call_count == (1 if returned else 0)


class _TaggedProvider(RecordingFunctor):
    """Provider whose k-th call returns the increment k, identifying its sub-execution."""

    def __init__(self, name: str, update_var: str, execution_group: str = "fast") -> None:
        super().__init__(name, execution_group, update_var)

    def forward(self, state, time, dt):
        result = super().forward(state, time, dt)
        result[self.update_var].value = torch.tensor(float(self.call_count), dtype=torch.float32)
        return result


class _TaggedConsumer(RecordingFunctor):
    """Consumer recording the provider outputs it receives; its k-th call returns scale * k."""

    def __init__(
        self,
        name: str,
        update_var: str,
        providers: List[tuple],
        scale: float,
        *,
        execution_group: str = "fast",
        fail_calls: tuple = (),
    ) -> None:
        super().__init__(name, execution_group, update_var)
        self.providers = list(providers)
        self.scale = float(scale)
        self.fail_calls = set(fail_calls)
        self.seen: List[tuple] = []

    def get_dependencies(self):
        return [provider for provider, _ in self.providers]

    def forward(self, state, time, dt, subprocess_outputs):
        result = super().forward(state, time, dt)
        self.seen.append(
            tuple(float(subprocess_outputs[provider][var].value) for provider, var in self.providers)
        )
        if self.call_count in self.fail_calls:
            raise RuntimeError("injected composition failure")
        result[self.update_var].value = torch.tensor(self.scale * self.call_count, dtype=torch.float32)
        return result


def _composition_simulator(splitting_mode: str, **config: Any) -> RELICSimulator:
    settings = {
        "device": "cpu",
        "conservation_checks": False,
        "time_step": 2.0,
        "splitting_mode": splitting_mode,
    }
    settings.update(config)
    simulator = _declared_simulator(settings)
    simulator.initialize_state({"shared_x": 0.0, "shared_y": 0.0, "shared_z": 0.0})
    return simulator


def _palindrome_chain(simulator: RELICSimulator, *, fail_calls: tuple = ()) -> tuple:
    a = _TaggedProvider("pal_a", "shared_x")
    b = _TaggedConsumer("pal_b", "shared_y", [("pal_a", "shared_x")], 10.0)
    c = _TaggedConsumer(
        "pal_c", "shared_z", [("pal_a", "shared_x"), ("pal_b", "shared_y")], 100.0,
        fail_calls=fail_calls,
    )
    for functor in (a, b, c):
        simulator.register_functor(functor)
    assert list(simulator._order_fast) == ["pal_a", "pal_b", "pal_c"]
    return a, b, c


@pytest.mark.parametrize(
    ("splitting_mode", "expected_y", "expected_z"),
    [("lie", 1.0, 1.0), ("strang", 2.0, 1.0)],
)
def test_dependency_outputs_reach_the_consumer_in_lie_and_strang(splitting_mode, expected_y, expected_z):
    """source_probe -> output_probe: z adds the provider's latest earlier increment.

    Lie: source(h), probe(h) gives y = 1, z = 1. Strang: source(h/2), probe(h),
    source(h/2) gives y = 1 + 1 (a per-call increment applied twice) and the probe
    reads the first half-step's output, z = 1.
    """
    simulator = _composition_simulator(splitting_mode)
    source = RecordingFunctor("source_probe", "fast", "shared_y", update_value=1.0)
    probe = DependencyOutputProbe(source.name)
    simulator.register_functor(source)
    simulator.register_functor(probe)

    simulator.step()

    state = simulator.get_state()
    assert simulator.current_time == 2.0
    assert state["shared_y"].item() == expected_y
    assert state["shared_z"].item() == expected_z
    assert simulator._active_composition is None


@pytest.mark.parametrize("splitting_mode", ["lie", "strang"])
def test_three_functor_composition_sees_the_latest_earlier_provider_outputs(splitting_mode):
    """Strang pal_a(h/2) pal_b(h/2) pal_c(h) pal_b(h/2) pal_a(h/2): every read is pal_a's first half."""
    simulator = _composition_simulator(splitting_mode)
    a, b, c = _palindrome_chain(simulator)

    simulator.step()

    state = simulator.get_state()
    if splitting_mode == "lie":
        assert b.seen == [(1.0,)]
        assert c.seen == [(1.0, 10.0)]
        assert (state["shared_x"].item(), state["shared_y"].item()) == (1.0, 10.0)
    else:
        # pal_b's second half runs before pal_a's second half, so both pal_b calls
        # see pal_a's first half-step (call 1); pal_c sees pal_a call 1 and pal_b call 1.
        assert b.seen == [(1.0,), (1.0,)]
        assert c.seen == [(1.0, 10.0)]
        assert (state["shared_x"].item(), state["shared_y"].item()) == (1.0 + 2.0, 10.0 + 20.0)
    assert state["shared_z"].item() == 100.0
    assert simulator._active_composition is None


def test_adaptive_retry_starts_a_fresh_dependency_output_composition():
    simulator = _composition_simulator(
        "strang",
        time_step=60.0,
        adaptive_dt_enabled=True,
        adaptive_dt_min=1.0,
        adaptive_dt_max=600.0,
        adaptive_atol=1.0e-6,
        adaptive_rtol=1.0e-3,
    )
    a, b, c = _palindrome_chain(simulator)

    simulator.step()

    # Rejected attempt: pal_a calls 1, 2; pal_b calls 1, 2 see pal_a call 1; pal_c call 1.
    # Retry: pal_a calls 3, 4; pal_b calls 3, 4 see pal_a call 3; pal_c call 2 sees (3, 30).
    assert a.dt_calls == [30.0, 30.0, 3.0, 3.0]
    assert b.seen == [(1.0,), (1.0,), (3.0,), (3.0,)]
    assert c.seen == [(1.0, 10.0), (3.0, 30.0)]
    state = simulator.get_state()
    assert simulator.current_time == 6.0
    assert (state["shared_x"].item(), state["shared_y"].item(), state["shared_z"].item()) == (
        3.0 + 4.0, 30.0 + 40.0, 200.0,
    )
    assert simulator._active_composition is None


def test_failed_strang_step_rolls_back_and_the_next_step_starts_empty():
    simulator = _composition_simulator("strang")
    a, b, c = _palindrome_chain(simulator, fail_calls=(1,))

    with pytest.raises(Exception, match="injected composition failure"):
        simulator.step()

    state = simulator.get_state()
    assert simulator.current_time == 0.0
    assert (state["shared_x"].item(), state["shared_y"].item(), state["shared_z"].item()) == (0.0, 0.0, 0.0)
    assert simulator._active_composition is None
    simulator.state = simulator.state.__class__.INITIALIZED

    simulator.step()

    # Second step: pal_a calls 2, 3; pal_b calls 2, 3 see pal_a call 2; pal_c call 2 sees (2, 20).
    assert b.seen == [(1.0,), (2.0,), (2.0,)]
    assert c.seen == [(1.0, 10.0), (2.0, 20.0)]
    state = simulator.get_state()
    assert simulator.current_time == 2.0
    assert (state["shared_x"].item(), state["shared_y"].item(), state["shared_z"].item()) == (
        2.0 + 3.0, 20.0 + 30.0, 200.0,
    )


def test_medium_group_consumer_sees_its_medium_provider_outputs():
    simulator = _composition_simulator("strang", medium_interval_seconds=2.0)
    provider = _TaggedProvider("medium_source", "shared_x", execution_group="medium")
    consumer = _TaggedConsumer(
        "medium_reader", "shared_y", [("medium_source", "shared_x")], 10.0, execution_group="medium"
    )
    simulator.register_functor(provider)
    simulator.register_functor(consumer)

    simulator.step()
    simulator.step()

    assert consumer.seen == [(1.0,), (2.0,)]
    assert simulator._active_composition is None


def test_consumer_of_another_group_provider_is_told_the_composition_scope():
    simulator = _composition_simulator("lie", medium_interval_seconds=2.0)
    provider = _TaggedProvider("fast_source", "shared_x")
    consumer = _TaggedConsumer(
        "medium_reader", "shared_y", [("fast_source", "shared_x")], 10.0, execution_group="medium"
    )
    simulator.register_functor(provider)
    simulator.register_functor(consumer)

    with pytest.raises(Exception) as error:
        simulator.step()

    message = str(error.value) + str(error.value.__cause__)
    assert "fast_source" in message and "group composition" in message


class _ChannelFunctor(RecordingFunctor):
    """One output per merge channel; the ACCUMULATE state increment is 1 per call.

    Optional channels: a REPLACE state value 10 * k on the k-th call, a plain
    diagnostic (an observation), an ACCUMULATE UpdateItem diagnostic (an amount
    per call), a REPLACE UpdateItem diagnostic 10 * k on the k-th call, a
    plain diagnostic emitted on the first calls only, and a per-second
    ACCUMULATE rate (unit s-1) written to a state variable and to a DERIVED
    diagnostic.
    """

    def __init__(
        self,
        name: str,
        *,
        execution_group: str = "fast",
        increment_var: str = "shared_x",
        replace_var: Optional[str] = None,
        plain_name: Optional[str] = None,
        plain_value: float = 0.5,
        amount_name: Optional[str] = None,
        amount_value: float = 0.25,
        early_name: Optional[str] = None,
        early_calls: int = 0,
        tagged_increment: bool = False,
        last_name: Optional[str] = None,
        rate_var: Optional[str] = None,
        rate_name: Optional[str] = None,
        rate_value: float = 0.75,
    ) -> None:
        super().__init__(name, execution_group, increment_var, update_value=1.0)
        self.last_name = last_name
        if replace_var is not None:
            self.metadata.output_variables.append(replace_var)
        if rate_var is not None:
            self.metadata.output_variables.append(rate_var)
        self.replace_var = replace_var
        self.plain_name = plain_name
        self.plain_value = float(plain_value)
        self.amount_name = amount_name
        self.amount_value = float(amount_value)
        self.early_name = early_name
        self.early_calls = int(early_calls)
        self.tagged_increment = tagged_increment
        self.rate_var = rate_var
        self.rate_name = rate_name
        self.rate_value = float(rate_value)

    def forward(self, state, time, dt):
        result = super().forward(state, time, dt)
        if self.tagged_increment:
            result[self.update_var].value = torch.tensor(float(self.call_count), dtype=torch.float32)
        if self.replace_var is not None:
            result[self.replace_var] = UpdateItem(
                value=torch.tensor(10.0 * self.call_count, dtype=torch.float32),
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                source_functor=self.name,
            )
        if self.plain_name is not None:
            result[self.plain_name] = torch.tensor(self.plain_value, dtype=torch.float32)
        if self.amount_name is not None:
            result[self.amount_name] = UpdateItem(
                value=torch.tensor(self.amount_value, dtype=torch.float32),
                mode=UpdateMode.ACCUMULATE,
                write_mode=StateWriteMode.DERIVED,
                unit="dimensionless",
                per_timestep=True,
            )
        if self.last_name is not None:
            result[self.last_name] = UpdateItem(
                value=torch.tensor(10.0 * self.call_count, dtype=torch.float32),
                mode=UpdateMode.REPLACE,
                write_mode=StateWriteMode.DERIVED,
                unit="dimensionless",
            )
        if self.early_name is not None and self.call_count <= self.early_calls:
            result[self.early_name] = torch.tensor(1.0, dtype=torch.float32)
        if self.rate_var is not None:
            result[self.rate_var] = UpdateItem(
                value=torch.tensor(self.rate_value, dtype=torch.float32),
                mode=UpdateMode.ACCUMULATE,
                unit="s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            )
        if self.rate_name is not None:
            result[self.rate_name] = UpdateItem(
                value=torch.tensor(self.rate_value, dtype=torch.float32),
                mode=UpdateMode.ACCUMULATE,
                write_mode=StateWriteMode.DERIVED,
                unit="s-1",
                basis="per_second",
                per_timestep=False,
            )
        return result


class _AfterFunctor(RecordingFunctor):
    """ACCUMULATE writer ordered after a named Functor."""

    def __init__(self, name: str, update_var: str, after: str) -> None:
        super().__init__(name, "fast", update_var, update_value=1.0)
        self.after = after

    def get_dependencies(self):
        return [self.after]


@pytest.mark.parametrize(
    ("splitting_mode", "increment", "replaced", "amount"),
    [("lie", 1.0, 10.0, 0.25), ("strang", 2.0, 20.0, 0.5)],
)
def test_one_merge_rule_for_substeps_and_strang_sub_executions(splitting_mode, increment, replaced, amount):
    """split_a runs outside the middle of the palindrome (twice under strang).

    Plain diagnostic: the last observation, equal to the stored diagnostic.
    ACCUMULATE UpdateItem diagnostic and ACCUMULATE state: the sum of the calls.
    REPLACE state: the last call's value.
    """
    simulator = _composition_simulator(splitting_mode)
    a = _ChannelFunctor(
        "split_a",
        replace_var="shared_y",
        plain_name="diagnostics.split_plain",
        amount_name="diagnostics.split_amount",
    )
    b = _AfterFunctor("split_b", "shared_z", after="split_a")
    simulator.register_functor(a)
    simulator.register_functor(b)
    assert list(simulator._order_fast) == ["split_a", "split_b"]

    updates = simulator.step()

    stored = simulator.state_manager.get_diagnostics()["current"]["split_a.split_plain"]
    assert float(updates["diagnostics.split_plain"]) == 0.5 == stored
    assert float(updates["diagnostics.split_amount"]) == amount
    assert float(updates["shared_x"]) == increment
    assert float(updates["shared_y"]) == replaced
    state = simulator.get_state()
    assert (state["shared_x"].item(), state["shared_y"].item()) == (increment, replaced)


def test_error_norm_reads_the_merged_state_updates_with_their_modes(monkeypatch):
    import relic.core.simulator.execution_loop as execution_loop

    captured = {}
    original = execution_loop._compute_error_norm

    def spy(simulator, state_before, updates):
        captured["state_before"] = dict(state_before)
        captured["updates"] = dict(updates)
        return original(simulator, state_before, updates)

    monkeypatch.setattr(execution_loop, "_compute_error_norm", spy)
    simulator = _composition_simulator(
        "strang", adaptive_dt_enabled=True, adaptive_dt_min=1.0, adaptive_atol=1.0e9, adaptive_rtol=1.0,
    )
    simulator.register_functor(_ChannelFunctor(
        "split_a", replace_var="shared_y", plain_name="diagnostics.split_plain",
        amount_name="diagnostics.split_amount",
    ))
    simulator.register_functor(_AfterFunctor("split_b", "shared_z", after="split_a"))

    simulator.step()

    merged = captured["updates"]
    assert merged["shared_x"].mode == UpdateMode.ACCUMULATE and float(merged["shared_x"].value) == 2.0
    assert merged["shared_y"].mode == UpdateMode.REPLACE and float(merged["shared_y"].value) == 20.0
    assert "diagnostics.split_plain" not in captured["state_before"]
    assert "diagnostics.split_amount" not in captured["state_before"]


def test_rejected_adaptive_attempt_outputs_do_not_reach_the_step_return():
    simulator = _composition_simulator(
        "strang",
        time_step=60.0,
        adaptive_dt_enabled=True,
        adaptive_dt_min=1.0,
        adaptive_dt_max=600.0,
        adaptive_atol=1.0e-6,
        adaptive_rtol=1.0e-3,
    )
    a = _ChannelFunctor(
        "leak_a", early_name="diagnostics.rejected_only", early_calls=2, tagged_increment=True,
    )
    b = _AfterFunctor("leak_b", "shared_z", after="leak_a")
    simulator.register_functor(a)
    simulator.register_functor(b)

    updates = simulator.step()

    # Rejected attempt: leak_a calls 1, 2 (x + 3), leak_b call 1; retry: leak_a calls 3, 4
    # (x + 7), leak_b call 2. Only the accepted retry reaches the return and the state.
    assert a.dt_calls == [30.0, 30.0, 3.0, 3.0]
    assert "diagnostics.rejected_only" not in updates
    assert float(updates["shared_x"]) == 7.0
    assert float(updates["shared_z"]) == 1.0
    state = simulator.get_state()
    assert (state["shared_x"].item(), state["shared_z"].item()) == (7.0, 1.0)


def test_plain_diagnostic_observed_by_two_groups_returns_the_last_observation():
    simulator = _composition_simulator("lie", medium_interval_seconds=2.0)
    simulator.register_functor(_ChannelFunctor(
        "obs_fast", plain_name="diagnostics.shared_obs", plain_value=1.0,
    ))
    simulator.register_functor(_ChannelFunctor(
        "obs_medium", execution_group="medium", increment_var="shared_y",
        plain_name="diagnostics.shared_obs", plain_value=2.0,
    ))

    updates = simulator.step()

    assert float(updates["diagnostics.shared_obs"]) == 2.0


@pytest.mark.parametrize(("splitting_mode", "expected"), [("lie", 0.75), ("strang", 1.0)])
def test_accumulate_diagnostic_of_two_functors_is_summed_by_its_mode(splitting_mode, expected):
    """Lie: 0.25 + 0.5; Strang: 0.25 (first half) + 0.5 (middle) + 0.25 (second half)."""
    simulator = _composition_simulator(splitting_mode)
    simulator.register_functor(_ChannelFunctor(
        "sum_a", amount_name="diagnostics.shared_amount", amount_value=0.25,
    ))
    b = _ChannelFunctor(
        "sum_b", increment_var="shared_z", amount_name="diagnostics.shared_amount", amount_value=0.5,
    )
    b.get_dependencies = lambda: ["sum_a"]
    simulator.register_functor(b)
    assert list(simulator._order_fast) == ["sum_a", "sum_b"]

    updates = simulator.step()

    assert float(updates["diagnostics.shared_amount"]) == expected


@pytest.mark.parametrize(
    ("splitting_mode", "amount", "last"),
    [("lie", 1.0, 10.0), ("strang", 2.0, 20.0)],
)
def test_stored_diagnostics_follow_the_merge_rule_over_sub_executions(splitting_mode, amount, last):
    """store_a runs outside the middle of the palindrome (twice under strang).

    Stored and returned values agree: ACCUMULATE summed, REPLACE and plain last.
    """
    simulator = _composition_simulator(splitting_mode)
    simulator.register_functor(_ChannelFunctor(
        "store_a",
        amount_name="diagnostics.store_amount",
        amount_value=1.0,
        plain_name="diagnostics.store_plain",
        last_name="diagnostics.store_last",
    ))
    simulator.register_functor(_AfterFunctor("store_b", "shared_z", after="store_a"))
    assert list(simulator._order_fast) == ["store_a", "store_b"]

    updates = simulator.step()

    stored = simulator.state_manager.get_diagnostics()["current"]
    assert float(updates["diagnostics.store_amount"]) == amount == stored["store_a.store_amount"]
    assert float(updates["diagnostics.store_last"]) == last == stored["store_a.store_last"]
    assert float(updates["diagnostics.store_plain"]) == 0.5 == stored["store_a.store_plain"]


def test_rejected_adaptive_attempt_leaves_no_stored_diagnostics():
    simulator = _composition_simulator(
        "strang",
        time_step=60.0,
        adaptive_dt_enabled=True,
        adaptive_dt_min=1.0,
        adaptive_dt_max=600.0,
        adaptive_atol=1.0e-6,
        adaptive_rtol=1.0e-3,
    )
    a = _ChannelFunctor(
        "stored_leak_a",
        early_name="diagnostics.rejected_only",
        early_calls=2,
        tagged_increment=True,
        amount_name="diagnostics.leak_amount",
        amount_value=1.0,
    )
    simulator.register_functor(a)
    simulator.register_functor(_AfterFunctor("stored_leak_b", "shared_z", after="stored_leak_a"))

    updates = simulator.step()

    # Rejected attempt: calls 1, 2; accepted retry: calls 3, 4 (two half-steps).
    assert a.dt_calls == [30.0, 30.0, 3.0, 3.0]
    stored = simulator.state_manager.get_diagnostics()["current"]
    assert "stored_leak_a.rejected_only" not in stored
    assert float(updates["diagnostics.leak_amount"]) == 2.0 == stored["stored_leak_a.leak_amount"]


@pytest.mark.parametrize("splitting_mode", ["lie", "strang"])
def test_per_second_rates_merge_time_weighted_over_sub_executions(splitting_mode):
    """rate_a runs outside the middle of the palindrome (twice, h/2 each, under strang).

    Committed: q h. Returned and stored: the rate over the step, q; the sum of
    the two half-step rates would be 2 q.
    """
    simulator = _composition_simulator(splitting_mode)
    simulator.register_functor(_ChannelFunctor(
        "rate_a", rate_var="shared_y", rate_name="diagnostics.rate_signal", rate_value=0.75,
    ))
    simulator.register_functor(_AfterFunctor("rate_b", "shared_z", after="rate_a"))
    assert list(simulator._order_fast) == ["rate_a", "rate_b"]

    updates = simulator.step()

    stored = simulator.state_manager.get_diagnostics()["current"]
    assert simulator.get_state()["shared_y"].item() == pytest.approx(1.5)
    assert float(updates["shared_y"]) == pytest.approx(0.75)
    assert float(updates["diagnostics.rate_signal"]) == pytest.approx(0.75)
    assert stored["rate_a.rate_signal"] == pytest.approx(0.75)


def test_merge_rule_weights_per_second_rates_by_the_covered_fraction():
    """Amounts add; a per-second rate adds w q with w = h / H, so merged * H = sum q h."""
    from relic.core.simulator.execution_group import merge_sequential_updates

    def rate(value: float) -> UpdateItem:
        return UpdateItem(
            torch.tensor(value), UpdateMode.ACCUMULATE, unit="s-1", basis="per_second", per_timestep=False,
        )

    def amount(value: float) -> UpdateItem:
        return UpdateItem(torch.tensor(value), UpdateMode.ACCUMULATE, unit="dimensionless", per_timestep=True)

    merged: Dict[str, Any] = {}
    merge_sequential_updates(merged, {"rate": rate(1.0), "amount": amount(1.0)}, coverage=0.25)
    merge_sequential_updates(merged, {"rate": rate(3.0), "amount": amount(2.0)}, coverage=0.75)

    assert merged["rate"].value.item() == pytest.approx(2.5)
    assert (merged["rate"].basis, merged["rate"].per_timestep) == ("per_second", False)
    assert merged["amount"].value.item() == pytest.approx(3.0)
    for coverage in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="coverage"):
            merge_sequential_updates({}, {"rate": rate(1.0)}, coverage=coverage)

def test_output_specification_failure_fails_the_group() -> None:
    """A Functor whose output specifications cannot be read is not treated as having none."""
    simulator = _build_initialized_simulator({"time_step": 1.0})
    functor = RecordingFunctor("spec_probe", "fast", "shared_x")

    def _broken():
        raise RuntimeError("output specification table unreadable")

    functor.get_output_specifications = _broken
    simulator.register_functor(functor)
    with pytest.raises(RuntimeError) as error:
        simulator.step()
    chain, cause = [], error.value
    while cause is not None:
        chain.append(str(cause))
        cause = cause.__cause__ or cause.__context__
    assert any("unreadable" in message for message in chain), chain


def test_flux_whitelist_check_errors_propagate(monkeypatch) -> None:
    from relic.core.simulator.execution_group import _check_flux_whitelist
    from types import SimpleNamespace

    def _broken(name):
        raise RuntimeError("contract table unreadable")

    registry = SimpleNamespace(
        get_flux_whitelist=lambda name: {"flux_in": ["writer"], "flux_out": []},
        get_contract=_broken,
    )
    owner = SimpleNamespace(state_manager=SimpleNamespace(_contract_registry=registry))
    with pytest.raises(RuntimeError, match="contract table unreadable"):
        _check_flux_whitelist(owner, "writer", {"x": UpdateItem(value=torch.tensor(1.0))})


class _TimeRateWriter(ExplicitUpdateFunctor):
    """Per-second rate writer q(t) = base + slope * t of a boundary-rate aggregate."""

    def __init__(self, name: str, update_var: str, base: float, slope: float, unit: str) -> None:
        super().__init__(
            name=name,
            update_var=update_var,
            mode=UpdateMode.ACCUMULATE,
            units_by_call=[unit],
            basis="per_second",
        )
        self.base = float(base)
        self.slope = float(slope)

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, UpdateItem]:
        self.update_value = self.base + self.slope * float(time)
        return super().forward(state, time, dt)


@pytest.mark.parametrize(
    ("splitting_mode", "expected"),
    [
        # Lie: both writers over [0, h] at t = 0: 0.5 + 0.4.
        ("lie", 0.9),
        # Strang: nitrification over [0, h/2] (0.5) and [h/2, h] (0.5 + 0.1 * 5 = 1.0), weighted
        # 1/2 each (time-weighted mean 0.75), plus denitrification over [0, h] (0.4); the last
        # sub-execution's rate alone would be 1.0.
        ("strang", 1.15),
    ],
)
def test_multi_writer_boundary_rate_aggregate_is_the_sum_of_the_writers_step_rates(
    splitting_mode: str, expected: float
) -> None:
    """A boundary-rate aggregate with several governed writers (n2o_flux: nitrification and
    denitrification, overlap policy 'aggregated') stores after the step the merge-rule value
    sum_f sum_k (h_k / h) q_(f,k) -- the value step() returns -- in both splitting modes.

    Each Strang sub-execution commits its own rate over its own interval (the per-commit
    boundary ledger); the stored aggregate is not the last sub-execution's rate.
    """
    simulator = _declared_simulator(
        {
            "time_step": 10.0,
            "splitting_mode": splitting_mode,
            "state_units": {"n2o_flux": "g N m-2 s-1"},
        }
    )
    simulator.initialize_state({"n2o_flux": 0.0})
    simulator.register_functor(
        _TimeRateWriter("nitrification", "n2o_flux", base=0.5, slope=0.1, unit="g N m-2 s-1")
    )
    simulator.register_functor(
        _TimeRateWriter("denitrification", "n2o_flux", base=0.4, slope=0.0, unit="g N m-2 s-1")
    )
    assert list(simulator._order_fast) == ["nitrification", "denitrification"]
    writes: List[tuple] = []
    commit = simulator.state_manager.apply_updates

    def _recording_commit(updates, dt=None, check_conservation=True, **kwargs):
        if "n2o_flux" in updates:
            item = updates["n2o_flux"]
            writes.append((float(item.value), float(dt), bool(check_conservation)))
        return commit(updates, dt, check_conservation=check_conservation, **kwargs)

    simulator.state_manager.apply_updates = _recording_commit

    updates = simulator.step()

    assert float(updates["n2o_flux"]) == pytest.approx(expected)
    assert simulator.get_state()["n2o_flux"].item() == pytest.approx(expected)
    ledger_commits = [(value, dt) for value, dt, checked in writes if checked]
    stored_merges = [(value, dt) for value, dt, checked in writes if not checked]
    if splitting_mode == "lie":
        # One call over [0, h]: the committed sum is the stored value; nothing is restated.
        assert ledger_commits == [(pytest.approx(0.9), 10.0)]
        assert stored_merges == []
    else:
        # Each physical stage commits its own rate and interval exactly once.
        # Stage observations retain the other writer: .5+.4=.9, then 1+.4=1.4.
        # Successful completion alone publishes the full-step mean .75+.4=1.15.
        # These restatements have no second boundary ledger or material advance.
        assert ledger_commits == [
            (pytest.approx(0.5), 5.0), (pytest.approx(0.4), 10.0), (pytest.approx(1.0), 5.0)
        ]
        assert stored_merges == [
            (pytest.approx(0.9), 10.0), (pytest.approx(1.4), 5.0), (pytest.approx(1.15), 10.0)
        ]

@pytest.mark.parametrize("splitting_mode", ["lie", "strang"])
@pytest.mark.parametrize("substeps", [1, 2])
@pytest.mark.parametrize("dt", [10.0, 60.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adaptive_metric_uses_the_committed_delta_target(
    monkeypatch, splitting_mode, substeps, dt, dtype,
) -> None:
    """The same committed tendency has the same metric for both legal names."""
    # References: StateManager.apply_updates; Hairer and Wanner (1996), weighted RMS.
    # Mathematical expression: Err = abs(x_after - x_before) / (atol + rtol * abs(x_before)).
    # Parameter meanings: q is a constant tendency, h the attempted interval, x the committed store.
    # Units: x [kg m-2], q [kg m-2 s-1], h [s], Err [1].
    # Domain/constraints: Canonical and delta_ update syntax; same owner and supported unit contract.
    # Sign convention: Positive q increases x; rejected attempts leave no extra advance.
    traces = []
    for output_name in ("shared_x", "delta_shared_x"):
        simulator = _declared_simulator(_adaptive_config(dt, splitting_mode, "shared_x"))
        _initialize_shared_x(simulator, torch.tensor(300.0, dtype=dtype))
        simulator.state_manager.register_variable(
            StateVariableSpec(name="shared_y", units="dimensionless", conserved=False, required=False)
        )
        simulator.state_manager.update_state({"shared_y": torch.tensor(0.0)})
        writer = ExplicitUpdateFunctor(
            "same_rate_writer", output_name, UpdateMode.ACCUMULATE,
            update_value=0.006, substep_count=substeps,
            units_by_call=["kg m-2 s-1"], basis="per_second",
        )
        # Two fast Functors make Strang's symmetric half-step composition nontrivial.
        observer = StateReaderFunctor("same_rate_reader", "shared_x", writer.name)
        simulator.register_functor(writer, execution_group="fast")
        simulator.register_functor(observer, execution_group="fast")
        traces.append(_adaptive_step_trace(monkeypatch, simulator, "shared_x"))
    canonical, delta = traces
    assert delta["metrics"] == pytest.approx(canonical["metrics"], rel=0.0, abs=1e-12)
    assert canonical["metrics"][0] == pytest.approx(0.006 * dt / (1e-6 + 0.001 * 300.0), rel=3e-4)
    for key in ("time", "change", "next_dt"):
        assert delta[key] == pytest.approx(canonical[key], rel=0.0, abs=1e-12)
    assert canonical["time"] < dt if dt == 60.0 else canonical["time"] == dt


def test_adaptive_metric_deduplicates_actual_committed_targets() -> None:
    """Multiple update names for one actual target do not change RMS weights."""
    from relic.core.simulator.execution_loop import _compute_error_norm

    simulator = _declared_simulator({
        "adaptive_dt_enabled": True, "adaptive_atol": 1.0, "adaptive_rtol": 0.0,
        "conservation_checks": False,
    })
    simulator.initialize_state({"shared_x": torch.tensor(0.0), "shared_y": torch.tensor(0.0)})
    before = simulator.get_state()
    updates = {
        "shared_x": UpdateItem(torch.tensor(1.0), UpdateMode.ACCUMULATE, unit="dimensionless", per_timestep=True),
        "delta_shared_x": UpdateItem(torch.tensor(1.0), UpdateMode.ACCUMULATE, unit="dimensionless", per_timestep=True),
        "shared_y": UpdateItem(torch.tensor(0.5), UpdateMode.ACCUMULATE, unit="dimensionless", per_timestep=True),
        "diagnostics.ignore": torch.tensor(1e6),
        "derived_ignore": UpdateItem(torch.tensor(1e6), UpdateMode.REPLACE, write_mode=StateWriteMode.DERIVED),
    }
    simulator.state_manager.apply_updates(updates, dt=1.0, check_conservation=False)
    after = simulator.get_state()
    expected = math.sqrt(sum(float((after[key] - before[key]).square()) for key in ("shared_x", "shared_y")) / 2)
    assert _compute_error_norm(simulator, before, updates) == pytest.approx(expected, abs=1e-12)


def test_adaptive_alias_write_keeps_the_existing_scm_rejection() -> None:
    """Read aliases do not become a new writable compatibility path."""
    simulator = _declared_simulator({"conservation_checks": False})
    simulator.initialize_state({"rain_flux": torch.tensor(0.0)})
    before = simulator.get_state()
    with pytest.raises(PermissionError, match="Cannot write to alias"):
        simulator.state_manager.apply_updates({
            "rainfall_flux": UpdateItem(
                torch.tensor(0.001), UpdateMode.REPLACE,
                unit="kg m-2 s-1", basis="instantaneous", per_timestep=False,
            )
        }, dt=1.0, check_conservation=False)
    assert torch.equal(simulator.get_state()["rain_flux"], before["rain_flux"])
    assert "rainfall_flux" not in simulator.get_state()


class _BoundaryStageReader(StateReaderFunctor):
    """Observe both state and declared provider rates at the same real stage."""

    def __init__(self, providers, *, fail_calls=()):
        super().__init__("rate_reader", "n2o_flux", providers[0])
        self.providers = list(providers)
        self.provider_seen = []
        self.fail_calls = set(fail_calls)

    def get_dependencies(self):
        return self.providers

    def forward(self, state, time, dt, subprocess_outputs):
        result = super().forward(state, time, dt)
        self.provider_seen.append(tuple(float(subprocess_outputs[name]["n2o_flux"].value) for name in self.providers))
        if self.call_count in self.fail_calls:
            raise RuntimeError("injected boundary stage failure")
        return result


def _boundary_stage_simulator(mode, initial_rate=0.0, **settings):
    config = {
        "device": "cpu", "time_step": 10.0, "splitting_mode": mode,
        "conservation_checks": False,
        "state_units": {"n2o_flux": "g N m-2 s-1", "shared_x": "kg m-2", "shared_y": "dimensionless"},
    }
    config.update(settings)
    simulator = _declared_simulator(config)
    simulator.initialize_state({
        "n2o_flux": torch.tensor(initial_rate), "shared_x": torch.tensor(300.0), "shared_y": torch.tensor(0.0),
    })
    return simulator


@pytest.mark.parametrize("mode", ["lie", "strang"])
@pytest.mark.parametrize("initial_rate", [0.0, 7.0])
@pytest.mark.parametrize("substeps", [1, 2, 4])
def test_boundary_stage_rate_is_not_a_partial_step_mean_or_previous_step_sum(mode, initial_rate, substeps):
    """A constant rate stays q in both reader channels, including the next step."""
    # References: Strang (1968); the StateManager boundary-rate contract.
    # Mathematical expression: q_stage=2; B=sum_k h_k*q_k/H=2.
    # Parameter meanings: h_k stage interval, H whole-step interval, q a prescribed rate.
    # Units: q and B [g N m-2 s-1], h_k and H [s].
    # Domain/constraints: One declared writer; old stored rates are not new contributions.
    # Sign convention: Positive boundary rate is outward.
    simulator = _boundary_stage_simulator(mode, initial_rate)
    writer = ExplicitUpdateFunctor(
        "nitrification", "n2o_flux", UpdateMode.ACCUMULATE, update_value=2.0,
        substep_count=substeps, units_by_call=["g N m-2 s-1"], basis="per_second",
    )
    reader = _BoundaryStageReader([writer.name])
    simulator.register_functor(writer)
    simulator.register_functor(reader)
    for _ in range(2):
        updates = simulator.step()
        assert float(updates["n2o_flux"]) == pytest.approx(2.0)
        assert float(simulator.get_state()["n2o_flux"]) == pytest.approx(2.0)
        assert simulator._active_composition is None
    assert reader.seen == pytest.approx([2.0, 2.0])
    assert reader.provider_seen == [(2.0,), (2.0,)]
    assert writer.seen_values[0] == initial_rate
    assert writer.seen_values[1:] == pytest.approx([2.0] * (len(writer.seen_values) - 1))
    assert simulator.current_time == 20.0


@pytest.mark.parametrize(
    ("mode", "substeps", "stage_rate", "final_rate"),
    [("lie", 1, 0.5, 0.5), ("lie", 2, 0.75, 0.75),
     ("strang", 1, 0.5, 0.75), ("strang", 2, 0.625, 0.875)],
)
def test_time_varying_boundary_stage_and_final_rates_keep_their_own_intervals(mode, substeps, stage_rate, final_rate):
    """q(t)=.5+.1t is averaged on the provider stage, then weighted on the whole step."""
    # References: Strang (1968); left-endpoint quadrature of a declared rate.
    # Mathematical expression: q(t)=.5+.1*t; B=sum_k q(t_k)*h_k/H.
    # Parameter meanings: t_k is the real substep start; H=10.
    # Units: q [g N m-2 s-1], t_k/H [s], slope [g N m-2 s-2].
    # Domain/constraints: Explicitly sampled linear rate; no order-of-accuracy claim.
    # Sign convention: Positive rates are boundary exports.
    simulator = _boundary_stage_simulator(mode, initial_rate=7.0)
    writer = _TimeRateWriter("nitrification", "n2o_flux", base=0.5, slope=0.1, unit="g N m-2 s-1")
    writer.metadata.substep_count = substeps
    reader = _BoundaryStageReader([writer.name])
    simulator.register_functor(writer)
    simulator.register_functor(reader)
    result = simulator.step()
    assert reader.seen == pytest.approx([stage_rate])
    assert reader.provider_seen[0] == pytest.approx((stage_rate,))
    assert float(result["n2o_flux"]) == pytest.approx(final_rate)
    assert float(simulator.get_state()["n2o_flux"]) == pytest.approx(final_rate)


@pytest.mark.parametrize("mode", ["lie", "strang"])
def test_stage_aggregate_keeps_each_writers_latest_contribution(mode):
    """The second writer replaces its own stage contribution, not another writer's."""
    simulator = _boundary_stage_simulator(mode, initial_rate=7.0)
    first = _TimeRateWriter("nitrification", "n2o_flux", base=0.5, slope=0.1, unit="g N m-2 s-1")
    second = _TimeRateWriter("denitrification", "n2o_flux", base=0.4, slope=0.2, unit="g N m-2 s-1")
    reader = _BoundaryStageReader([first.name, second.name])
    for functor in (first, second, reader):
        simulator.register_functor(functor)
    result = simulator.step()
    assert reader.seen == pytest.approx([0.9])
    assert reader.provider_seen[0] == pytest.approx((0.5, 0.4))
    if mode == "strang":
        assert second.seen_values == pytest.approx([0.5, 0.9])
        assert first.seen_values == pytest.approx([7.0, 1.9])
        expected_final = 1.65
    else:
        assert second.seen_values == pytest.approx([0.5])
        expected_final = 0.9
    assert float(result["n2o_flux"]) == pytest.approx(expected_final)
    assert float(simulator.get_state()["n2o_flux"]) == pytest.approx(expected_final)


@pytest.mark.parametrize("mode", ["lie", "strang"])
def test_stage_and_final_boundary_rates_use_reduced_writer_contributions(mode, monkeypatch):
    """A deterministic reducer oracle stands for two ranks q and 3q: their mean is 2q."""
    # References: PyTorch AllReduce mean; RELIC pre-commit reduction ownership.
    # Mathematical expression: q_bar=(q+3*q)/2=2*q; reduction happens exactly once.
    # Parameter meanings: q is one writer's local rate; q_bar is the shared stage rate.
    # Units: q and q_bar [g N m-2 s-1].
    # Domain/constraints: Public simulator with the existing reduction entry replaced by an exact oracle.
    # Sign convention: Positive rates are outward; this is not a hardware DDP qualification.
    simulator = _boundary_stage_simulator(mode)
    reduced_rates = []
    def reduce(updates):
        if "n2o_flux" in updates:
            item = updates["n2o_flux"]
            reduced_rates.append(float(item.value))
            item.value = item.value * 2.0
    monkeypatch.setattr(simulator, "_reduce_updates_across_ranks", reduce)
    first = _TimeRateWriter("nitrification", "n2o_flux", base=0.5, slope=0.1, unit="g N m-2 s-1")
    second = _TimeRateWriter("denitrification", "n2o_flux", base=0.4, slope=0.0, unit="g N m-2 s-1")
    reader = _BoundaryStageReader([first.name, second.name])
    for functor in (first, second, reader):
        simulator.register_functor(functor)
    result = simulator.step()
    assert reader.seen == pytest.approx([1.8])
    assert reader.provider_seen[0] == pytest.approx((1.0, 0.8))
    expected_final = 2.3 if mode == "strang" else 1.8
    assert float(result["n2o_flux"]) == pytest.approx(expected_final)
    assert float(simulator.get_state()["n2o_flux"]) == pytest.approx(expected_final)
    assert reduced_rates == pytest.approx([0.5, 0.4, 0.4, 1.0] if mode == "strang" else [0.5, 0.4])


class _BoundaryAndStoreWriter(ExplicitUpdateFunctor):
    def __init__(self):
        super().__init__("nitrification", "n2o_flux", UpdateMode.ACCUMULATE,
                         update_value=2.0, units_by_call=["g N m-2 s-1"], basis="per_second")
        self.metadata.output_variables.append("shared_x")

    def forward(self, state, time, dt):
        # References: RELIC rate contract and adaptive committed-state metric.
        # Mathematical expression: delta_x=.006*h; q_boundary=2.
        # Parameter meanings: x is the independent controller probe store; h is its interval.
        # Units: x [kg m-2], its rate [kg m-2 s-1], boundary q [g N m-2 s-1].
        # Domain/constraints: Independent prescribed fixture channels, not one physical N ledger.
        # Sign convention: Positive store rate increases x; boundary rate is outward.
        result = super().forward(state, time, dt)
        result["shared_x"] = UpdateItem(torch.tensor(0.006), UpdateMode.ACCUMULATE,
                                       unit="kg m-2 s-1", basis="per_second", per_timestep=False,
                                       source_functor=self.name)
        return result


def test_adaptive_retry_rebuilds_boundary_stage_and_final_views():
    simulator = _boundary_stage_simulator(
        "strang", initial_rate=7.0, time_step=60.0, adaptive_dt_enabled=True,
        adaptive_dt_min=1.0, adaptive_dt_max=600.0, adaptive_atol=1e-6,
        adaptive_rtol=1e-3, adaptive_monitor_vars=["shared_x"],
    )
    writer = _BoundaryAndStoreWriter()
    reader = _BoundaryStageReader([writer.name])
    simulator.register_functor(writer)
    simulator.register_functor(reader)
    result = simulator.step()
    assert 1.0 < simulator.current_time < 60.0
    assert writer.call_count == 4
    assert reader.seen == pytest.approx([2.0, 2.0])
    assert reader.provider_seen == [(2.0,), (2.0,)]
    assert float(result["n2o_flux"]) == pytest.approx(2.0)
    assert float(simulator.get_state()["n2o_flux"]) == pytest.approx(2.0)
    assert float(simulator.get_state()["shared_x"]) == pytest.approx(300.0 + 0.006 * simulator.current_time, abs=5e-5)
    assert simulator._active_composition is None


def test_failed_boundary_composition_restores_rates_and_next_step_starts_fresh():
    simulator = _boundary_stage_simulator("strang", initial_rate=7.0)
    writer = ExplicitUpdateFunctor("nitrification", "n2o_flux", UpdateMode.ACCUMULATE,
                                   update_value=2.0, units_by_call=["g N m-2 s-1"], basis="per_second")
    reader = _BoundaryStageReader([writer.name], fail_calls=(1,))
    simulator.register_functor(writer)
    simulator.register_functor(reader)
    with pytest.raises(Exception, match="injected boundary stage failure"):
        simulator.step()
    assert float(simulator.get_state()["n2o_flux"]) == 7.0
    assert float(simulator.get_state()["shared_y"]) == 0.0
    assert simulator.current_time == 0.0
    assert simulator._active_composition is None
    simulator.state = simulator.state.__class__.INITIALIZED
    result = simulator.step()
    assert reader.seen == pytest.approx([2.0, 2.0])
    assert float(result["n2o_flux"]) == pytest.approx(2.0)
    assert float(simulator.get_state()["n2o_flux"]) == pytest.approx(2.0)
    assert simulator.current_time == 10.0
    assert simulator._active_composition is None
