"""Unit tests for effective dt passed to medium/slow execution groups."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from relic.core.functor import (
    ExecutionError,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    TransportMechanismFunctor,
    VariableKind,
)
from relic.core.simulator import RELICSimulator
from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


class _RecorderFunctor(TransportMechanismFunctor):
    """Record dt values received by forward() calls."""

    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str, substep_count: int = 1) -> None:
        self.output_name = f"diagnostics.test.effective_dt.{name}"
        metadata = FunctorMetadata(
            name=name,
            description="Effective dt recorder",
            references=["RELIC simulator effective dt policy"],
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            temporal_scale=TemporalScale.SECOND,
            spatial_scale=SpatialScale.POINT,
            variable_kind=VariableKind.DIAGNOSTIC,
            output_variables=[self.output_name],
            substep_count=substep_count,
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self.dt_calls: List[float] = []
        self.time_calls: List[float] = []
        self.calendar_calls: List[str | None] = []

    def get_dependencies(self) -> List[str]:
        return []

    def forward(
        self,
        state: Dict[str, Any],
        time: float,
        dt: torch.Tensor,
        calendar_type: str | None = None,
    ) -> Dict[str, UpdateItem]:
        dt_value = float(dt.item()) if isinstance(dt, torch.Tensor) else float(dt)
        self.dt_calls.append(dt_value)
        self.time_calls.append(float(time))
        self.calendar_calls.append(calendar_type)
        return {
            self.output_name: UpdateItem(
                value=torch.as_tensor(dt_value, dtype=torch.float64),
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="s",
                source_functor=self.name,
            )
        }


@pytest.fixture(params=["state", "owners"])
def simulator_factory(request):
    """Exercise both public initialization paths with the same declared state."""
    def build(config):
        simulator = RELICSimulator(config)
        if request.param == "state":
            simulator.initialize_state({"temperature": 295.0})
        else:
            manager = StateManager({
                "temperature": StateVariableSpec(name="temperature", units="K"),
            })
            manager.initialize({"temperature": torch.tensor(295.0)})
            simulator.initialize(manager, DependencyGraphResolver())
        return simulator

    return build


def test_medium_group_receives_day_interval_effective_dt(simulator_factory) -> None:
    """Medium group should receive one-day effective dt when crossing day boundary."""
    simulator = simulator_factory({"time_step": 43200.0})

    fast_functor = _RecorderFunctor("fast_recorder")
    medium_functor = _RecorderFunctor("medium_recorder")
    simulator.register_functor(fast_functor, execution_group="fast")
    simulator.register_functor(medium_functor, execution_group="medium")

    simulator.step()  # t: 0 -> 43200, no day boundary
    simulator.step()  # t: 43200 -> 86400, day boundary crossed

    assert fast_functor.dt_calls == [43200.0, 43200.0]
    assert len(medium_functor.dt_calls) == 1
    assert medium_functor.dt_calls[0] == 86400.0


def test_slow_group_receives_month_interval_effective_dt_360_day_calendar(simulator_factory) -> None:
    """Slow group should receive one-month effective dt on month boundary."""
    simulator = simulator_factory({"time_step": 86400.0, "calendar_type": "360_day"})

    fast_functor = _RecorderFunctor("fast_month_recorder")
    slow_functor = _RecorderFunctor("slow_recorder")
    simulator.register_functor(fast_functor, execution_group="fast")
    simulator.register_functor(slow_functor, execution_group="slow")

    for _ in range(30):
        simulator.step()

    assert len(slow_functor.dt_calls) == 1
    assert slow_functor.dt_calls[0] == 30.0 * 86400.0
    assert all(dt == 86400.0 for dt in fast_functor.dt_calls)
    assert fast_functor.calendar_calls == ["360_day"] * 30
    assert slow_functor.calendar_calls == ["360_day"]


def test_bootstrap_execution_uses_step_dt_for_medium_and_slow(simulator_factory) -> None:
    """Bootstrap-triggered first execution should use the step dt."""
    simulator = simulator_factory({"time_step": 60.0})

    medium_functor = _RecorderFunctor("medium_bootstrap")
    slow_functor = _RecorderFunctor("slow_bootstrap")
    medium_functor.force_initial_execution = True
    slow_functor.force_initial_execution = True

    simulator.register_functor(medium_functor, execution_group="medium")
    simulator.register_functor(slow_functor, execution_group="slow")
    simulator.step()

    assert medium_functor.dt_calls == [60.0]
    assert slow_functor.dt_calls == [60.0]
    assert medium_functor.time_calls == [0.0]
    assert slow_functor.time_calls == [0.0]


def test_bootstrap_mark_is_consumed_when_a_boundary_already_runs_the_group(simulator_factory) -> None:
    """A group that ran for its first boundary must not run again for the bootstrap mark."""
    simulator = simulator_factory({"time_step": 100.0, "medium_interval_seconds": 100.0})

    medium_functor = _RecorderFunctor("medium_boundary_bootstrap")
    medium_functor.force_initial_execution = True
    simulator.register_functor(medium_functor, execution_group="medium")
    assert simulator._bootstrap_groups == {"medium"}

    simulator.step(100.0)  # boundary floor(100 / 100) = 1 fires; the mark is consumed too
    assert medium_functor.dt_calls == [100.0]
    assert simulator._bootstrap_groups == set()

    simulator.step(50.0)  # t = 150: no boundary and no pending mark
    assert medium_functor.dt_calls == [100.0]
    assert medium_functor.time_calls == [0.0]


class _FailingRecorderFunctor(_RecorderFunctor):
    def forward(self, state, time, dt, calendar_type=None):
        raise RuntimeError("injected medium failure")


def test_bootstrap_mark_is_restored_when_the_step_fails(simulator_factory) -> None:
    simulator = simulator_factory({"time_step": 60.0})
    medium_functor = _FailingRecorderFunctor("medium_failing_bootstrap")
    medium_functor.force_initial_execution = True
    simulator.register_functor(medium_functor, execution_group="medium")

    with pytest.raises((RuntimeError, ExecutionError), match="injected medium failure"):
        simulator.step()

    assert simulator._bootstrap_groups == {"medium"}
    assert simulator.current_time == 0.0


def test_medium_call_time_is_the_start_of_the_accounted_interval(simulator_factory) -> None:
    """(time, dt) of a medium call describes [t_last, t_future], not [t_current, ...]."""
    simulator = simulator_factory({"time_step": 43200.0})

    fast_functor = _RecorderFunctor("fast_interval_recorder")
    medium_functor = _RecorderFunctor("medium_interval_recorder")
    simulator.register_functor(fast_functor, execution_group="fast")
    simulator.register_functor(medium_functor, execution_group="medium")

    for _ in range(4):
        simulator.step()  # boundaries at t_future = 86400 and 172800

    assert fast_functor.time_calls == [0.0, 43200.0, 86400.0, 129600.0]
    assert medium_functor.dt_calls == [86400.0, 86400.0]
    assert medium_functor.time_calls == [0.0, 86400.0]
    for start, length in zip(medium_functor.time_calls, medium_functor.dt_calls):
        assert start + length in (86400.0, 172800.0)


def test_slow_substeps_start_at_last_execution_time(simulator_factory) -> None:
    """Slow sub-step k of an interval [t_last, t_future] starts at t_last + k * h_sub."""
    simulator = simulator_factory(
        {"time_step": 86400.0, "slow_interval_seconds": 3.0 * 86400.0}
    )

    slow_functor = _RecorderFunctor("slow_substep_recorder", substep_count=3)
    simulator.register_functor(slow_functor, execution_group="slow")

    for _ in range(6):
        simulator.step()

    day = 86400.0
    assert slow_functor.dt_calls == [day] * 6
    assert slow_functor.time_calls == [0.0, day, 2.0 * day, 3.0 * day, 4.0 * day, 5.0 * day]


def test_rebinding_preserves_unfinished_rate_group_intervals(simulator_factory) -> None:
    simulator = simulator_factory({
        "time_step": 3600.0,
        "medium_interval_seconds": 10800.0,
        "slow_interval_seconds": 21600.0,
    })
    medium = _RecorderFunctor("rebind_medium")
    slow = _RecorderFunctor("rebind_slow")
    simulator.register_functor(medium, execution_group="medium")
    simulator.register_functor(slow, execution_group="slow")
    for _ in range(4):
        simulator.step()
    simulator.initialize(simulator.state_manager, simulator.dependency_graph)
    for _ in range(2):
        simulator.step()
    assert medium.time_calls == [0.0, 10800.0]
    assert medium.dt_calls == [10800.0, 10800.0]
    assert slow.time_calls == [0.0]
    assert slow.dt_calls == [21600.0]


@pytest.mark.parametrize("calendar", ["gregorian", "360_day"])
def test_nonzero_run_start_keeps_calendar_and_interval_windows(simulator_factory, calendar) -> None:
    simulator = simulator_factory({
        "time_step": 1800.0,
        "medium_interval_seconds": 10800.0,
        "calendar_type": calendar,
        "time_origin": "2024-02-28T23:00:00",
    })
    medium = _RecorderFunctor("offset_medium")
    slow = _RecorderFunctor("offset_slow")
    simulator.register_functor(medium, execution_group="medium")
    simulator.register_functor(slow, execution_group="slow")
    # The civil month has one more day in the 360-day calendar.
    start = 86400.0 if calendar == "gregorian" else 172800.0
    simulator.run(start_time=start, end_time=start + 10800.0, dt=1800.0)
    assert medium.time_calls == [start]
    assert medium.dt_calls == [10800.0]
    assert slow.time_calls == [start]
    assert slow.dt_calls == [3600.0]


@pytest.mark.parametrize("trigger_group", ["medium", "slow"])
def test_event_before_boundary_accounts_only_elapsed_time(simulator_factory, trigger_group) -> None:
    simulator = simulator_factory({
        "time_step": 3600.0,
        "medium_interval_seconds": 10800.0,
        "slow_interval_seconds": 10800.0,
    })
    recorder = _RecorderFunctor("triggered_recorder")
    simulator.register_functor(recorder, execution_group=trigger_group)
    simulator.register_event_trigger(trigger_group, lambda state, time: time == 3600.0)
    simulator.step()
    assert recorder.dt_calls == []
    simulator.step()
    simulator.step()
    assert recorder.time_calls == [0.0, 7200.0]
    assert recorder.dt_calls == [7200.0, 3600.0]


@pytest.mark.writes_artifacts
def test_rebinding_after_checkpoint_restore_keeps_pending_intervals(simulator_factory, tmp_path) -> None:
    config = {
        "time_step": 3600.0,
        "medium_interval_seconds": 10800.0,
        "slow_interval_seconds": 21600.0,
    }
    source = simulator_factory(config)
    for group in ("medium", "slow"):
        source.register_functor(_RecorderFunctor(f"checkpoint_{group}"), execution_group=group)
    for _ in range(4):
        source.step()
    path = tmp_path / "pending_intervals.pt"
    source.create_checkpoint(path)

    restored = simulator_factory(config)
    for group in ("medium", "slow"):
        restored.register_functor(_RecorderFunctor(f"checkpoint_{group}"), execution_group=group)
    restored.restore_checkpoint(path)
    restored.initialize(restored.state_manager, restored.dependency_graph)
    for _ in range(2):
        source.step()
        restored.step()
    assert restored.current_time == source.current_time == 21600.0
    assert restored.functors["checkpoint_medium"].time_calls == [10800.0]
    assert restored.functors["checkpoint_medium"].dt_calls == [10800.0]
    assert restored.functors["checkpoint_slow"].time_calls == [0.0]
    assert restored.functors["checkpoint_slow"].dt_calls == [21600.0]
    for name, value in source.get_state().items():
        assert torch.equal(value, restored.get_state()[name])
