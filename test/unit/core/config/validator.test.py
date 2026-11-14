"""Calendar-aware tests for core configuration validation."""

from __future__ import annotations

import inspect

import pytest

from relic.core.config.models import RELICConfig
from relic.core.config.validator import (
    ConfigValidationStage,
    ConfigValidator,
    IntegrationMethodconsistencyRule,
    ValidationSeverity,
    validate_configuration,
)
from relic.utils.time_constants import SECONDS_PER_DAY


def _configuration(duration_days: float) -> RELICConfig:
    config = RELICConfig()
    config.simulation.start_time = 0.0
    config.simulation.end_time = duration_days * float(SECONDS_PER_DAY)
    return config


def _adaptive_messages(messages):
    """The long-run recommendation names the RELICSimulator runtime key; the
    typed simulation section declares no adaptive-step field."""
    return [
        message
        for message in messages
        if message.location == "runtime.adaptive_dt_enabled"
    ]


def test_long_simulation_boundary_depends_on_calendar() -> None:
    config = _configuration(362.0)

    noleap_messages = IntegrationMethodconsistencyRule(
        calendar_type="noleap"
    ).validate(config)
    day360_messages = IntegrationMethodconsistencyRule(
        calendar_type="360_day"
    ).validate(config)

    assert _adaptive_messages(noleap_messages) == []
    assert len(_adaptive_messages(day360_messages)) == 1
    assert _adaptive_messages(day360_messages)[0].severity is ValidationSeverity.INFO


def test_long_simulation_without_calendar_fails_closed() -> None:
    messages = IntegrationMethodconsistencyRule().validate(_configuration(362.0))
    calendar_messages = [
        message
        for message in messages
        if message.location == "simulation.calendar_type"
    ]

    assert len(calendar_messages) == 1
    assert calendar_messages[0].severity is ValidationSeverity.ERROR
    assert _adaptive_messages(messages) == []


def test_config_validator_rejects_unsupported_calendar() -> None:
    with pytest.raises(ValueError, match="noleap or 360_day"):
        ConfigValidator(calendar_type="gregorian")


def test_config_validator_and_convenience_api_forward_calendar() -> None:
    config = _configuration(362.0)
    levels = [ConfigValidationStage.SEMANTIC]

    direct = ConfigValidator(calendar_type="360_day").validate(
        config,
        levels=levels,
        use_cache=False,
    )
    convenience = validate_configuration(
        config,
        strict=False,
        calendar_type="360_day",
    )

    assert len(_adaptive_messages(direct.messages)) == 1
    assert len(_adaptive_messages(convenience.messages)) == 1


def test_calendar_aware_rule_has_no_fixed_year_or_day_literal() -> None:
    source = inspect.getsource(IntegrationMethodconsistencyRule)

    assert "86400" not in source
    assert "365" not in source
    assert "SECONDS_PER_DAY" in source
    assert "days_per_simulator_year" in source
