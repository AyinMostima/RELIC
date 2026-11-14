"""Core data quality validators for RELIC data ingestion pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

try:
    import xarray as _xr
except ImportError:  # pragma: no cover - xarray is an optional runtime dependency (absence only)
    _xr = None

from ..standards.unit_converter import (
    UnitConverter,
    unit_converter as canonical_unit_converter,
)


@dataclass(frozen=True)
class RangeRule:
    """Numeric range rule for a variable."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    inclusive_min: bool = True
    inclusive_max: bool = True


@dataclass(frozen=True)
class QualityValidationIssue:
    """Single quality validation issue."""

    variable: str
    check: str
    message: str
    severity: str = "error"
    details: Optional[Dict[str, Any]] = None


ValidationIssue = QualityValidationIssue


class QualityValidator:
    """
    Validate range, temporal consistency, and units for xarray datasets.

    References:
        - CF Conventions 1.8 metadata and units recommendations.
        - Flux data quality control practices (range + continuity checks).

    Mathematical Expressions:
        1) Range checks:
           valid_min = x >= x_min (or x > x_min when strict lower bound)
           valid_max = x <= x_max (or x < x_max when strict upper bound)
        2) Temporal monotonicity:
           delta_t(i) = t(i+1) - t(i), require delta_t(i) > 0
        3) Time gap threshold:
           max(delta_t) <= delta_t_max

    Parameter Meanings:
        x: observed data value
        x_min, x_max: configured lower and upper bounds
        t(i): ith timestamp
        delta_t_max: maximum allowed interval between consecutive timestamps
    """

    def __init__(
        self,
        *,
        range_rules: Optional[Mapping[str, RangeRule]] = None,
        expected_units: Optional[Mapping[str, str]] = None,
        max_time_gap_seconds: Optional[float] = None,
        allow_missing_units: bool = False,
        unit_converter: Optional[UnitConverter] = None,
    ) -> None:
        self.range_rules: Dict[str, RangeRule] = dict(range_rules or {})
        self.expected_units: Dict[str, str] = dict(expected_units or {})
        self.max_time_gap_seconds = max_time_gap_seconds
        self.allow_missing_units = allow_missing_units
        self._unit_converter = (
            unit_converter if unit_converter is not None else canonical_unit_converter
        )

    def check_data(self, dataset: "xr.Dataset") -> tuple[bool, Dict[str, Any]]:
        """Validate dataset and return `(is_valid, report_dict)`."""
        issues = self.validate(dataset)
        report = {
            "total_issues": len(issues),
            "issues": [asdict(item) for item in issues],
            "checks": sorted({item.check for item in issues}),
        }
        return len(issues) == 0, report

    def validate(self, dataset: "xr.Dataset") -> List[QualityValidationIssue]:
        """Run all configured checks and return all issues."""
        if _xr is None:
            return [
                QualityValidationIssue(
                    variable="__dataset__",
                    check="runtime_dependency",
                    message="xarray is required for quality validation.",
                    severity="error",
                )
            ]
        if not isinstance(dataset, _xr.Dataset):
            return [
                QualityValidationIssue(
                    variable="__dataset__",
                    check="dataset_type",
                    message=f"Expected xarray.Dataset, got {type(dataset)}.",
                    severity="error",
                )
            ]

        issues: List[QualityValidationIssue] = []
        issues.extend(self.validate_ranges(dataset))
        issues.extend(self.validate_temporal_consistency(dataset))
        issues.extend(self.validate_units(dataset))
        return issues

    def validate_ranges(self, dataset: "xr.Dataset") -> List[QualityValidationIssue]:
        """Validate variable ranges against configured rules."""
        issues: List[QualityValidationIssue] = []
        for var_name, rule in self.range_rules.items():
            if var_name not in dataset.data_vars:
                issues.append(
                    QualityValidationIssue(
                        variable=var_name,
                        check="range",
                        message="Variable not found in dataset for range validation.",
                        severity="warning",
                    )
                )
                continue

            values = np.asarray(dataset[var_name].values)
            finite_mask = np.isfinite(values)
            if not np.any(finite_mask):
                issues.append(
                    QualityValidationIssue(
                        variable=var_name,
                        check="range",
                        message="No finite numeric values found for range validation.",
                        severity="warning",
                    )
                )
                continue

            finite_values = values[finite_mask]
            if rule.min_value is not None:
                if rule.inclusive_min:
                    invalid_low = finite_values < rule.min_value
                else:
                    invalid_low = finite_values <= rule.min_value
                if np.any(invalid_low):
                    issues.append(
                        QualityValidationIssue(
                            variable=var_name,
                            check="range",
                            message=f"Values below minimum bound {rule.min_value}.",
                            details={"violations": int(np.sum(invalid_low))},
                        )
                    )

            if rule.max_value is not None:
                if rule.inclusive_max:
                    invalid_high = finite_values > rule.max_value
                else:
                    invalid_high = finite_values >= rule.max_value
                if np.any(invalid_high):
                    issues.append(
                        QualityValidationIssue(
                            variable=var_name,
                            check="range",
                            message=f"Values above maximum bound {rule.max_value}.",
                            details={"violations": int(np.sum(invalid_high))},
                        )
                    )
        return issues

    def validate_temporal_consistency(self, dataset: "xr.Dataset") -> List[QualityValidationIssue]:
        """Validate time coordinate monotonicity and optional gap threshold."""
        issues: List[QualityValidationIssue] = []
        if "time" not in dataset.coords:
            return issues

        time_values = np.asarray(dataset["time"].values)
        if time_values.size < 2:
            return issues

        diffs = np.diff(time_values)
        if np.issubdtype(time_values.dtype, np.datetime64):
            diffs_seconds = diffs.astype("timedelta64[s]").astype(np.int64)
        else:
            diffs_seconds = diffs.astype(np.float64)

        non_monotonic = diffs_seconds <= 0
        if np.any(non_monotonic):
            issues.append(
                QualityValidationIssue(
                    variable="time",
                    check="temporal_consistency",
                    message="Time coordinate is not strictly increasing.",
                    details={"violations": int(np.sum(non_monotonic))},
                )
            )

        if self.max_time_gap_seconds is not None:
            max_gap = float(np.max(diffs_seconds))
            if max_gap > float(self.max_time_gap_seconds):
                issues.append(
                    QualityValidationIssue(
                        variable="time",
                        check="temporal_consistency",
                        message=(
                            f"Maximum time gap {max_gap} exceeds allowed "
                            f"{self.max_time_gap_seconds} seconds."
                        ),
                        details={"max_gap_seconds": max_gap},
                    )
                )
        return issues

    def validate_units(self, dataset: "xr.Dataset") -> List[QualityValidationIssue]:
        """Validate units against expected units with dimension compatibility."""
        issues: List[QualityValidationIssue] = []
        for var_name, expected_unit in self.expected_units.items():
            if var_name not in dataset.data_vars:
                issues.append(
                    QualityValidationIssue(
                        variable=var_name,
                        check="units",
                        message="Variable not found in dataset for unit validation.",
                        severity="warning",
                    )
                )
                continue

            actual_unit = str(dataset[var_name].attrs.get("units", "")).strip()
            if not actual_unit:
                if not self.allow_missing_units:
                    issues.append(
                        QualityValidationIssue(
                            variable=var_name,
                            check="units",
                            message="Units metadata missing.",
                        )
                    )
                continue

            if actual_unit == expected_unit:
                continue

            expected_dim = self._unit_converter.get_dimension(expected_unit)
            actual_dim = self._unit_converter.get_dimension(actual_unit)
            if expected_dim is not None and actual_dim is not None and expected_dim == actual_dim:
                continue

            issues.append(
                QualityValidationIssue(
                    variable=var_name,
                    check="units",
                    message=f"Unit mismatch: expected '{expected_unit}', got '{actual_unit}'.",
                    details={"expected": expected_unit, "actual": actual_unit},
                )
            )
        return issues


__all__ = ["QualityValidator", "RangeRule", "QualityValidationIssue", "ValidationIssue"]

