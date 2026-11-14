"""L1 tests for the calibration hook dispatcher error contract.

Module under test: relic/parameters/calibration/hooks.py. A failing callback
(early stopping, bounds monitoring, adaptive strategies) must reach the
calibration run instead of being logged and skipped.
"""

from __future__ import annotations

import inspect

import pytest

from relic.parameters import get_global_manager
from relic.parameters.calibration.hooks import (
    AdaptiveCallback,
    CalibrationEvent,
    CalibrationEventData,
    CalibrationHooks,
    ConvergenceCallback,
)


def _failing_callback(event_data):
    raise ValueError("callback rejected the event")


@pytest.mark.parametrize("async_execution", [False, True], ids=["sync", "async"])
def test_callback_failure_reaches_the_caller(async_execution: bool) -> None:
    hooks = CalibrationHooks(async_execution=async_execution, max_workers=2)
    hooks.register_callback(_failing_callback)
    try:
        with pytest.raises(ValueError, match="callback rejected the event"):
            hooks.trigger_event(CalibrationEvent.ITERATION_COMPLETE, iteration=1)
    finally:
        hooks.cleanup()


@pytest.mark.parametrize("async_execution", [False, True], ids=["sync", "async"])
def test_callback_result_triggers_the_follow_up_event(async_execution: bool) -> None:
    hooks = CalibrationHooks(async_execution=async_execution, max_workers=2)
    received = []
    hooks.register_callback(ConvergenceCallback(patience=1, early_stopping=True))
    hooks.register_callback(
        lambda event_data: received.append(event_data.event) or None,
        events=[CalibrationEvent.OBJECTIVE_IMPROVEMENT],
    )
    try:
        hooks.trigger_event(CalibrationEvent.OBJECTIVE_EVALUATION, objective_value=1.0)
    finally:
        hooks.cleanup()

    assert received == [CalibrationEvent.OBJECTIVE_IMPROVEMENT]


def _key(name: str):
    return get_global_manager().get(name, required=True)


def test_convergence_and_adaptive_settings_are_read_from_the_parameter_database() -> None:
    convergence = ConvergenceCallback()
    assert convergence.patience == _key("optimization.calibration.early_stopping_patience")
    assert convergence.min_improvement == _key("optimization.calibration.min_objective_improvement")
    adaptive = AdaptiveCallback()
    assert adaptive.adaptation_interval == _key("optimization.calibration.adaptive.adaptation_interval")
    for method in (
        ConvergenceCallback.__init__,
        AdaptiveCallback.__init__,
        CalibrationHooks.add_convergence_monitoring,
        CalibrationHooks.add_adaptive_strategies,
    ):
        for name in ("patience", "min_improvement", "adaptation_interval"):
            parameter = inspect.signature(method).parameters.get(name)
            if parameter is not None:
                assert parameter.default is None, (method.__qualname__, name)


def _adaptive_response(previous: float, current: float):
    callback = AdaptiveCallback(adaptation_interval=1, population_adaptation=True, learning_rate_adaptation=True)
    assert callback(CalibrationEventData(
        event=CalibrationEvent.ITERATION_COMPLETE, timestamp=0.0, iteration=1, objective_value=previous,
    )) is None
    return callback(CalibrationEventData(
        event=CalibrationEvent.ITERATION_COMPLETE, timestamp=1.0, iteration=2, objective_value=current,
    ))


def test_adaptive_improvement_rate_uses_the_mixed_scale() -> None:
    # References: Hairer, Norsett and Wanner (1993), Sect. II.4 (mixed absolute/relative scale).
    # rho = (L_prev - L_curr) / max(|L_prev|, s), s = optimization.calibration.adaptive.improvement_scale_floor.
    floor = _key("optimization.calibration.adaptive.improvement_scale_floor")
    response = _adaptive_response(previous=0.0, current=0.5)
    assert response["improvement_rate"] == pytest.approx(-0.5 / floor)
    response = _adaptive_response(previous=2.0, current=1.0)
    assert response["improvement_rate"] == pytest.approx(0.5)
    assert response["suggestions"]["learning_rate_factor"] == _key(
        "optimization.calibration.adaptive.learning_rate_increase_factor")


def test_adaptive_thresholds_are_read_from_the_parameter_database() -> None:
    # A relative improvement between the population rate (0.01) and the increase rate (0.1)
    # triggers neither suggestion; just below the decrease rate the learning-rate factor applies.
    assert _adaptive_response(previous=1.0, current=0.95) is None
    response = _adaptive_response(previous=1.0, current=0.9995)
    assert response["suggestions"]["population_factor"] == _key(
        "optimization.calibration.adaptive.population_growth_factor")
    assert response["suggestions"]["learning_rate_factor"] == _key(
        "optimization.calibration.adaptive.learning_rate_decrease_factor")
