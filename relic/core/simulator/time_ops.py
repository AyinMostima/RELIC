"""Time-index and execution-window helpers for RELICSimulator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from relic.utils.time_constants import SECONDS_PER_DAY


def compute_day_index(seconds: float) -> int:
    """Compute a zero-based day index from elapsed simulation seconds.

    References: CF Conventions time-coordinate units.
    Mathematical expression: d = floor(t / S_d).
    Parameter meanings: t is elapsed seconds; S_d is seconds per day.
    Units: t [s], S_d [s day-1], and d [day index].
    Domain/constraints: S_d comes from the physical-constant SSOT.
    Sign convention: increasing non-negative t advances d.
    """
    return int(seconds // float(SECONDS_PER_DAY))


def compute_interval_index(seconds: float, interval_seconds: float) -> int:
    """Compute interval index from simulation seconds and configurable interval.

    Mathematical Expression:
        index = floor(t / T_interval)
    where:
        t = simulation time [s]
        T_interval = group execution interval [s]

    References:
        Lawrence et al. (2019), CLM5 Technical Note, Section 2.7
    """
    if interval_seconds <= 0:
        return 0
    return int(seconds // interval_seconds)


def should_execute_medium(simulator: Any, future_time: Optional[float] = None) -> bool:
    """Return True when a medium-group interval boundary is crossed at ``future_time``.

    References: Lawrence et al. (2019), CLM5 Technical Note, Section 2.7.
    Mathematical expression: fire iff floor(t / T_m) > i_m, where i_m is the last
        committed medium interval index.
    Parameter meanings: t is the probed time (``future_time`` or the current time);
        T_m is ``medium_interval_seconds``; i_m is ``_previous_medium_index``.
    Units: t, T_m [s]; indices are dimensionless integers.
    Domain/constraints: pure predicate; probing never consumes the boundary. The index
        advances only through ``commit_rate_group_interval`` after the group executed
        inside an accepted step, so a rejected or retried trial cannot lose a trigger.
    Sign convention: boundaries fire when the index strictly increases.

    The interval is initialized from config key ``medium_interval_seconds`` by
    the canonical simulator initialization owner.
    """
    check_time = future_time if future_time is not None else simulator.current_time
    interval = float(simulator._medium_interval_seconds)
    current_idx = compute_interval_index(check_time, interval)
    return current_idx > getattr(simulator, "_previous_medium_index", -1)


def should_execute_slow(simulator: Any, future_time: Optional[float] = None) -> bool:
    """Return True when a slow-group interval boundary is crossed at ``future_time``.

    Uses the configurable interval when ``slow_interval_seconds`` is set and the
    CF calendar month boundary otherwise. Pure predicate: see ``should_execute_medium``
    for the commit contract.
    """
    check_time = future_time if future_time is not None else simulator.current_time
    interval = getattr(simulator, "_slow_interval_seconds", None)
    if interval is not None and float(interval) > 0.0:
        current_idx = compute_interval_index(check_time, float(interval))
        return current_idx > getattr(simulator, "_previous_slow_index", -1)

    current_month = simulator._compute_month_index(check_time)
    return current_month > simulator._previous_month_index


def commit_rate_group_interval(simulator: Any, group_name: str, future_time: float) -> None:
    """Advance the committed interval bookkeeping of ``group_name`` to ``future_time``.

    References: RELIC step transaction contract (execution_loop.step_simulation).
    Mathematical expression: i_g <- max(i_g, floor(t / T_g)) for interval groups;
        the calendar month index for the slow group without a fixed interval.
    Parameter meanings: g is ``medium`` or ``slow``; t is the accepted step end time.
    Units: t, T_g [s]; indices are dimensionless integers.
    Domain/constraints: called by the execution loop after the group ran inside the
        step transaction; the step runtime snapshot restores these fields when the
        step is rejected, so the commit is transactional. Unknown groups raise.
    Sign convention: indices are monotone non-decreasing.
    """
    if group_name == "medium":
        interval = float(simulator._medium_interval_seconds)
        current_idx = compute_interval_index(future_time, interval)
        if current_idx > getattr(simulator, "_previous_medium_index", -1):
            simulator._previous_medium_index = current_idx
            simulator._previous_day_index = compute_day_index(future_time)
        return
    if group_name == "slow":
        interval = getattr(simulator, "_slow_interval_seconds", None)
        if interval is not None and float(interval) > 0.0:
            current_idx = compute_interval_index(future_time, float(interval))
            if current_idx > getattr(simulator, "_previous_slow_index", -1):
                simulator._previous_slow_index = current_idx
            return
        current_month = simulator._compute_month_index(future_time)
        if current_month > simulator._previous_month_index:
            simulator._previous_month_index = current_month
        return
    raise ValueError(f"Unknown rate group '{group_name}'; expected 'medium' or 'slow'.")


def check_event_triggers(
    simulator: Any,
    group: str,
    state: Dict[str, Any],
) -> bool:
    """Check if any registered event trigger fires for the given group.

    Event triggers are callables registered via simulator ``register_event_trigger``.
    Each trigger receives ``(state, current_time)`` and returns a ``bool`` or an
    ``int`` (a nonzero int fires the group). A trigger that raises, or returns any
    other type, fails the step (fail-fast): the error is raised as RuntimeError
    naming the trigger and the group, and the step transaction rolls back. Silently
    treating a broken trigger as "not fired" would hide the defect and skip
    scheduled science.

    References:
        Lawrence et al. (2019), CLM5 Technical Note, Section 2.11
    """
    triggers = getattr(simulator, "_event_triggers", {}).get(group, [])
    current_time = getattr(simulator, "current_time", 0.0)
    for trigger_fn in triggers:
        trigger_name = getattr(trigger_fn, "__qualname__", None) or repr(trigger_fn)
        try:
            fired = trigger_fn(state, current_time)
        except Exception as exc:  # noqa: BLE001 - every failure is re-raised with context
            raise RuntimeError(
                f"Event trigger '{trigger_name}' for group '{group}' failed at "
                f"t={current_time}: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(fired, (bool, int)):
            raise RuntimeError(
                f"Event trigger '{trigger_name}' for group '{group}' returned "
                f"{type(fired).__name__}; bool or int is required."
            )
        if fired:
            return True
    return False


__all__ = [
    "compute_day_index",
    "compute_interval_index",
    "should_execute_medium",
    "should_execute_slow",
    "commit_rate_group_interval",
    "check_event_triggers",
]
