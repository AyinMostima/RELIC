"""Unit tests for relic.utils.time_constants derivation rules (SSOT Section 21.3).

Only an absent SECONDS_PER_HOUR or DAYS_PER_YEAR entry is derived from the
required constants; a present entry that fails validation raises.
"""

from __future__ import annotations

import pytest

import relic.utils.time_constants as time_constants

_DAY_KEY = "relic.physical.constants.time.seconds_per_day"
_YEAR_KEY = "relic.physical.constants.time.seconds_per_year"
_HOUR_KEY = "relic.physical.constants.time.seconds_per_hour"
_CALENDAR_KEY = "coupling.scales.temporal.calendars.gregorian.days_per_year"

# Stub values chosen to exercise the derivation arithmetic only.
_STUB_VALUES = {_DAY_KEY: 240.0, _YEAR_KEY: 2400.0}


class _StubManager:
    """ParameterManager stand-in: absent keys return None, failures raise."""

    def __init__(self, values, failures=None):
        self.values = dict(values)
        self.failures = dict(failures or {})

    def get(self, key, **_kwargs):
        if key in self.failures:
            raise self.failures[key]
        return self.values.get(key)


@pytest.fixture
def install_manager(monkeypatch):
    def _install(manager):
        monkeypatch.setattr(time_constants, "_PARAM_MANAGER", manager)
        monkeypatch.setattr(time_constants, "_TIME_CONSTANT_CACHE", {})

    return _install


def test_absent_seconds_per_hour_is_derived_from_seconds_per_day(install_manager):
    install_manager(_StubManager(_STUB_VALUES))

    assert time_constants._resolve_time_constant("SECONDS_PER_HOUR") == pytest.approx(10.0)


def test_invalid_seconds_per_hour_entry_raises(install_manager):
    install_manager(
        _StubManager(
            _STUB_VALUES,
            failures={_HOUR_KEY: ValueError("seconds_per_hour declares units 'h', expected 's'")},
        )
    )

    with pytest.raises(ValueError, match="declares units"):
        time_constants._resolve_time_constant("SECONDS_PER_HOUR")


def test_absent_calendar_days_per_year_is_derived(install_manager):
    install_manager(_StubManager(_STUB_VALUES))

    assert time_constants._resolve_time_constant("DAYS_PER_YEAR") == pytest.approx(10.0)


def test_invalid_calendar_days_per_year_entry_raises(install_manager):
    install_manager(
        _StubManager(
            _STUB_VALUES,
            failures={_CALENDAR_KEY: TypeError("days_per_year must be a float")},
        )
    )

    with pytest.raises(TypeError, match="days_per_year must be a float"):
        time_constants._resolve_time_constant("DAYS_PER_YEAR")
