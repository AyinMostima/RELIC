"""
Temporal CF metadata utility functions for unified RELIC architecture.

This module provides pure functions consumed by `relic.data.standards.cf_conventions`
so temporal CF metadata synchronization does not depend on legacy fallback paths.

References:
- CF Conventions 1.8, Section 7.3 (Cell methods), Section 4.4 (Time coordinate)
- NetCDF Climate and Forecast Metadata Conventions

Mathematical expression used:
- coverage_fraction = valid_points / total_points
- quality_state = good_data if coverage_fraction >= theta else insufficient_coverage

Parameter meanings:
- valid_points: number of valid temporal samples
- total_points: total number of temporal samples
- theta: coupling.scales.temporal.coverage_threshold (ParameterManager, required)

Coverage is never assumed: callers pass it explicitly and the threshold has no
Python default.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

from ....parameters import get_global_manager
from .core import TimeCoordinate, update_cell_methods as _update_cell_methods

_COVERAGE_THRESHOLD_KEY = "coupling.scales.temporal.coverage_threshold"


def require_text_argument(value: Any, name: str, owner: str) -> str:
    """A required text argument of a temporal operator: a non-empty string, else ValueError."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} requires a non-empty {name}; got {value!r}.")
    return value


def optional_text_argument(value: Any, name: str, owner: str) -> Optional[str]:
    """An optional text argument: None means absent; an empty or non-string value raises.

    Absent identity metadata has a declared meaning (see identity_metadata) and is never
    replaced by an empty string.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{owner}: {name} must be a non-empty string when given (omit it when absent); got {value!r}."
        )
    return value


def identity_metadata(variable_name: Optional[str], standard_name: Optional[str]) -> Dict[str, str]:
    """The identity entries a temporal operator publishes: only the declared ones.

    An absent variable_name is an unnamed field and an absent standard_name a field without a
    CF identity (CF-1.8 section 3.3 makes standard_name optional); neither is published as an
    empty string, so a consumer never mistakes "" for a name or a CF standard name.
    """
    return {
        key: value
        for key, value in (("variable_name", variable_name), ("standard_name", standard_name))
        if value is not None
    }


def create_cf_time_attributes(
    time_coord: TimeCoordinate,
    bounds_name: str = "time_bounds",
) -> Dict[str, Any]:
    """Create the CF attributes of a time coordinate variable (CF-1.8 4.4, 7.1).

    These describe the time axis, not a data variable on it. ``bounds`` names
    the boundary variable and is written only when the coordinate has bounds.
    """
    attrs: Dict[str, Any] = {
        "standard_name": "time",
        "long_name": "time",
        "axis": "T",
        "units": time_coord.units,
        "calendar": time_coord.calendar,
    }
    if time_coord.bounds is not None:
        attrs["bounds"] = bounds_name
    return attrs


def update_cell_methods(cell_methods: str, axis: str, method: str) -> str:
    """Update the CF cell_methods string for a temporal operation."""
    dimension = axis or "time"
    updated = _update_cell_methods(cell_methods or "", method, dimension)
    return str(updated)


def compute_time_coverage(
    coverage_stats: Optional[Mapping[str, Any]] = None,
    valid_points: Optional[int] = None,
    total_points: Optional[int] = None,
) -> Dict[str, Any]:
    """Normalize coverage diagnostics to a canonical dictionary.

    References: CF Conventions 1.8, Section 7.3 (cell methods over
        incomplete samples).
    Mathematical expression: f = n_valid / n_total when f is not given.
    Parameter meanings: n_valid valid samples; n_total samples in the
        period; f coverage fraction.
    Units: counts are sample counts; f is dimensionless.
    Domain/constraints: 0 <= n_valid <= n_total, n_total > 0, 0 <= f <= 1;
        invalid or non-numeric entries raise instead of being dropped.
    Sign convention: not applicable (nonnegative fractions).
    """
    if coverage_stats is not None:
        valid = coverage_stats.get("valid_points")
        total = coverage_stats.get("total_points")
        fraction = coverage_stats.get("coverage_fraction")
    else:
        valid = valid_points
        total = total_points
        fraction = None

    result: Dict[str, Any] = {}
    if valid is not None:
        result["valid_points"] = _sample_count(valid, "valid_points")
    if total is not None:
        result["total_points"] = _sample_count(total, "total_points")
        if result["total_points"] <= 0:
            raise ValueError(
                f"total_points must be positive; received {result['total_points']}"
            )
    if "valid_points" in result and "total_points" in result:
        if result["valid_points"] > result["total_points"]:
            raise ValueError(
                "valid_points must not exceed total_points; received "
                f"{result['valid_points']} > {result['total_points']}"
            )
        if fraction is None:
            fraction = result["valid_points"] / result["total_points"]
    if fraction is not None:
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ValueError(f"coverage_fraction must be a real number; received {fraction!r}")
        fraction_value = float(fraction)
        if not math.isfinite(fraction_value) or not 0.0 <= fraction_value <= 1.0:
            raise ValueError(
                f"coverage_fraction must be in [0, 1]; received {fraction_value}"
            )
        result["coverage_fraction"] = fraction_value
    return result


def _sample_count(value: Any, name: str) -> int:
    """Return one nonnegative integer sample count or raise."""
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise ValueError(f"{name} must be an integer sample count; received {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative; received {value}")
    return value


def _coverage_threshold(parameter_manager: Optional[Any]) -> float:
    """Read the coverage acceptance threshold from its parameter owner."""
    manager = parameter_manager if parameter_manager is not None else get_global_manager()
    if manager is None:
        raise RuntimeError("ParameterManager is required for the coverage threshold")
    value = manager.get(
        _COVERAGE_THRESHOLD_KEY,
        units="dimensionless",
        expected_type=float,
        required=True,
    )
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError(
            f"{_COVERAGE_THRESHOLD_KEY} must be in (0, 1]; received {threshold}"
        )
    return threshold


def create_quality_metadata(
    coverage_stats: Mapping[str, Any],
    parameter_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create quality metadata from explicit coverage diagnostics.

    References: CF Conventions 1.8, Section 7.3; WMO Guide to
        Climatological Practices (2018) on coverage for period statistics.
    Mathematical expression: quality_state = good_data if f >= theta,
        otherwise insufficient_coverage.
    Parameter meanings: f coverage fraction from ``coverage_stats``
        (coverage_fraction, or valid_points / total_points); theta the
        coupling.scales.temporal.coverage_threshold parameter.
    Units: f and theta are dimensionless.
    Domain/constraints: coverage must be supplied (no assumed full
        coverage); 0 <= f <= 1; 0 < theta <= 1; theta has no Python default.
    Sign convention: not applicable (classification).
    """
    coverage = compute_time_coverage(coverage_stats=coverage_stats)
    if "coverage_fraction" not in coverage:
        raise ValueError(
            "coverage_stats must provide coverage_fraction or both valid_points "
            "and total_points; coverage is never assumed"
        )
    coverage_fraction = coverage["coverage_fraction"]
    coverage_threshold = _coverage_threshold(parameter_manager)
    quality_state = (
        "good_data" if coverage_fraction >= coverage_threshold else "insufficient_coverage"
    )
    return {
        "coverage_threshold": coverage_threshold,
        "coverage_fraction": coverage_fraction,
        "quality_state": quality_state,
    }


def validate_cf_temporal_compliance(time_attributes: Mapping[str, Any]) -> List[str]:
    """Validate the minimal CF attributes of a time coordinate variable.

    ``time_attributes`` are the attributes of the time coordinate (as built by
    :func:`create_cf_time_attributes`), never those of a data variable, whose
    standard_name and units keep their own identity.
    """
    issues: List[str] = []
    for key in ("units", "calendar"):
        value = time_attributes.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"Missing or invalid CF time coordinate attribute: {key}")
    if time_attributes.get("standard_name") != "time":
        issues.append("Time coordinate standard_name must be 'time'")
    return issues


def create_temporal_rescaling_metadata(
    time_coord: TimeCoordinate,
    *,
    coverage_stats: Mapping[str, Any],
    parameter_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return the time-axis and coverage metadata of one temporal rescaling.

    The result holds ``time_coordinate`` (the CF attributes of the result time
    coordinate) plus the coverage statistics and quality classification of
    :func:`create_quality_metadata`. It never touches the rescaled variable's
    standard_name, units, cell_methods or history; those are written once by
    ``CFAttributeSynchronizer`` through its shared rescaling path.
    ``coverage_stats`` is required; coverage is never assumed.
    """
    metadata: Dict[str, Any] = {"time_coordinate": create_cf_time_attributes(time_coord)}
    metadata.update(compute_time_coverage(coverage_stats=coverage_stats))
    metadata.update(create_quality_metadata(coverage_stats, parameter_manager=parameter_manager))
    return metadata


__all__ = [
    "identity_metadata",
    "optional_text_argument",
    "require_text_argument",
    "create_temporal_rescaling_metadata",
    "create_cf_time_attributes",
    "update_cell_methods",
    "compute_time_coverage",
    "create_quality_metadata",
    "validate_cf_temporal_compliance",
]
