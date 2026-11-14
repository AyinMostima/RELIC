"""Unit tests for TimeStagger clock negotiation."""

import pytest

from relic.coupling.protocol.time_stagger import (
    ClockConfig,
    ClockSyncError,
    TimeStagger,
    TimeUnit,
)


def test_register_clock() -> None:
    stagger = TimeStagger()
    clock = ClockConfig(time_step=3600.0, unit=TimeUnit.SECONDS)
    stagger.register_clock("atmosphere", clock)
    assert "atmosphere" in stagger._clocks


def test_clock_to_seconds() -> None:
    clock = ClockConfig(time_step=1.0, unit=TimeUnit.HOURS)
    assert clock.to_seconds() == 3600.0

    clock_days = ClockConfig(time_step=1.0, unit=TimeUnit.DAYS)
    assert clock_days.to_seconds() == 86400.0


def test_negotiate_min_strategy() -> None:
    stagger = TimeStagger()
    stagger.register_clock("fast", ClockConfig(time_step=60.0))
    stagger.register_clock("slow", ClockConfig(time_step=3600.0))
    dt = stagger.negotiate(strategy="min")
    assert dt == 60.0


def test_negotiate_gcd_strategy() -> None:
    stagger = TimeStagger()
    stagger.register_clock("a", ClockConfig(time_step=300.0))
    stagger.register_clock("b", ClockConfig(time_step=600.0))
    dt = stagger.negotiate(strategy="gcd")
    assert dt == 300.0


def test_negotiate_fails_without_clocks() -> None:
    stagger = TimeStagger()
    with pytest.raises(ClockSyncError):
        stagger.negotiate()


def test_get_subcycle_ratio() -> None:
    stagger = TimeStagger()
    stagger.register_clock("fast", ClockConfig(time_step=60.0))
    stagger.register_clock("slow", ClockConfig(time_step=300.0))
    stagger.negotiate(strategy="min")
    assert stagger.get_subcycle_ratio("fast") == 1
    assert stagger.get_subcycle_ratio("slow") == 5


def test_validate_temporal_consistency() -> None:
    stagger = TimeStagger()
    stagger.register_clock("a", ClockConfig(time_step=100.0))
    stagger.register_clock("b", ClockConfig(time_step=300.0))
    stagger.negotiate(strategy="min")
    warnings = stagger.validate_temporal_consistency()
    assert warnings == []


def test_validate_detects_inconsistency() -> None:
    stagger = TimeStagger()
    stagger.register_clock("a", ClockConfig(time_step=100.0))
    stagger.register_clock("b", ClockConfig(time_step=333.0))
    stagger.negotiate(strategy="min")
    warnings = stagger.validate_temporal_consistency()
    assert len(warnings) > 0
