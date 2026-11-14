"""
Unified access to model time constants.

This module centralizes retrieval of canonical time-related constants
directly from the ParameterManager database to eliminate hardcoded values.
It provides convenient floats (in seconds) for use across scientific
modules while ensuring that any updates to the parameter database are
propagated automatically.

References
----------
- CF-1.8 / UDUNITS time standards.
- RELIC parameter specification: relic.physical.constants.time.*
"""

from __future__ import annotations

from typing import Dict, Optional

from relic.parameters import get_global_manager

_PARAM_MANAGER: Optional[object] = None
_PARAM_MANAGER_ERROR: Optional[Exception] = None
_TIME_CONSTANT_CACHE: Dict[str, float] = {}


def _get_parameter_manager():
    """Lazily resolve the global parameter manager."""
    global _PARAM_MANAGER, _PARAM_MANAGER_ERROR
    if _PARAM_MANAGER is not None:
        return _PARAM_MANAGER
    try:
        _PARAM_MANAGER = get_global_manager(preload_all=False, warmup_cache=False)
        _PARAM_MANAGER_ERROR = None
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        _PARAM_MANAGER_ERROR = exc
        raise RuntimeError(
            "ParameterManager is required to load time constants; ensure the parameter system is initialized."
        ) from exc
    return _PARAM_MANAGER


def _load_time_constant(key: str, units: str = "s", *, required: bool = True) -> float | None:
    """Retrieve a time constant from the parameter database.

    An absent entry raises KeyError when ``required``; otherwise it returns
    None so the caller can apply its documented derivation. A present entry
    that fails unit or type validation always raises.
    """
    manager = _get_parameter_manager()
    full_key = f"relic.physical.constants.{key}"
    value = manager.get(
        full_key,
        units=units,
        expected_type=float,
    )
    if value is None:
        if not required:
            return None
        raise KeyError(f"Missing required time constant '{full_key}'")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        raise TypeError(f"Time constant '{full_key}' must be convertible to float.") from exc


def _resolve_time_constant(name: str) -> float:
    """Resolve a named constant, caching results for reuse."""
    if name in _TIME_CONSTANT_CACHE:
        return _TIME_CONSTANT_CACHE[name]

    if name == "SECONDS_PER_DAY":
        value = _load_time_constant("time.seconds_per_day")
    elif name == "SECONDS_PER_YEAR":
        value = _load_time_constant("time.seconds_per_year")
    elif name == "SECONDS_PER_HOUR":
        # SSOT Section 21.3: only an absent seconds_per_hour entry is derived.
        # Mathematical expression: SECONDS_PER_HOUR = SECONDS_PER_DAY / 24,
        # where 24 is the number of hours in one day (dimensionless count);
        # units: s. A present entry that fails validation raises.
        loaded = _load_time_constant("time.seconds_per_hour", required=False)
        value = _resolve_time_constant("SECONDS_PER_DAY") / 24.0 if loaded is None else loaded
    elif name == "SECONDS_PER_MINUTE":
        value = _load_time_constant("time.seconds_per_minute")
    elif name == "DAYS_PER_YEAR":
        # SSOT Section 21.3: only an absent Gregorian calendar entry is derived.
        # Mathematical expression: DAYS_PER_YEAR = SECONDS_PER_YEAR /
        # SECONDS_PER_DAY; units: s / s per day = days. A present entry that
        # fails unit or type validation raises.
        manager = _get_parameter_manager()
        calendar_value = manager.get(
            "coupling.scales.temporal.calendars.gregorian.days_per_year",
            units="days",
            expected_type=float,
        )
        if calendar_value is None:
            value = _resolve_time_constant("SECONDS_PER_YEAR") / _resolve_time_constant("SECONDS_PER_DAY")
        else:
            value = float(calendar_value)
    else:
        raise AttributeError(f"Unknown time constant: {name}")

    _TIME_CONSTANT_CACHE[name] = value
    return value


def __getattr__(name: str):
    if name in {
        "SECONDS_PER_DAY",
        "SECONDS_PER_YEAR",
        "SECONDS_PER_HOUR",
        "SECONDS_PER_MINUTE",
        "DAYS_PER_YEAR",
    }:
        return _resolve_time_constant(name)
    raise AttributeError(f"module 'time_constants' has no attribute '{name}'")


def get_time_constants() -> Dict[str, float]:
    """Return a dictionary of loaded time constants."""
    names = ["SECONDS_PER_DAY", "SECONDS_PER_YEAR", "SECONDS_PER_HOUR", "SECONDS_PER_MINUTE", "DAYS_PER_YEAR"]
    return {name: _resolve_time_constant(name) for name in names}


__all__ = [
    "get_time_constants",
]
