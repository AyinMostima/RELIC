"""Strict CF time adapter for external-model I/O.

This module is an I/O adapter, not a calendar implementation. Calendar
normalization, coordinate validation, and interval construction are owned by
``relic.coupling.scales.temporal.core``. External datasets are converted to the
canonical Torch ``TimeCoordinate`` at ingress and converted to NumPy only at
the xarray serialization boundary.

References:
    CF Conventions 1.12, Section 4.4, Time Coordinate and Calendar.
Mathematical expressions:
    ``t = u * scale + epoch`` for a declared CF unit/epoch pair, and
    ``I_i = [b_i0, b_i1)`` with ``b_i0 < b_i1`` for time-cell support.
Parameter meanings:
    ``t`` is physical time, ``u`` is the numeric coordinate, ``scale`` is the
    declared unit duration, ``epoch`` is the reference instant, and ``I_i`` is
    the support interval represented by coordinate value ``u_i``.

RELIC strict policy requires an explicit, unambiguous calendar. This is
stronger than the CF default-calendar rule and prevents silent scientific
reinterpretation. Cross-calendar conversion remains non-operational until an
explicit preservation policy is bound to the canonical temporal owner.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
import xarray as xr

from relic.coupling.scales.temporal.core import (
    CFTimeProcessor,
    CalendarError,
    PeriodIndexer,
    TimeCoordinate,
    ValidationError,
)
from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace


class TimeHandler:
    """Adapt xarray CF metadata to the canonical RELIC temporal owner."""

    _CF_UNITS_PATTERN = re.compile(
        r"^(seconds?|minutes?|hours?|days?|months?|years?)\s+since\s+"
        r"(\d{4}-\d{2}-\d{2})"
        r"(?:[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?))?"
        r"(?:\s*(?:UTC|GMT|Z))?$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        parameter_manager: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.parameter_manager = (
            parameter_manager
            if parameter_manager is not None
            else get_global_manager()
        )
        if self.parameter_manager is None:
            raise RuntimeError("ParameterManager is required for TimeHandler")
        self._reference_epoch = self._resolve_reference_epoch()
        self.cf_processor = CFTimeProcessor(
            device=device,
            parameter_manager=self.parameter_manager,
        )
        self.period_indexer = PeriodIndexer(
            device=self.cf_processor.device,
            parameter_manager=self.parameter_manager,
        )

    def _resolve_reference_epoch(self) -> datetime:
        """Read the configured epoch for metadata reporting, never inference."""
        epoch_str = self.parameter_manager.get(
            "reference_epoch",
            namespace=resolve_namespace(ParameterNamespace.COUPLING_EXTERNAL_TIME),
            expected_type=str,
            required=True,
        )
        if not isinstance(epoch_str, str):
            raise ValueError("reference_epoch must be an ISO datetime string")
        normalized = epoch_str.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("reference_epoch must be an ISO datetime string") from exc
        return parsed.replace(tzinfo=None)

    def parse_time_units(self, units_string: str) -> Tuple[str, datetime]:
        """Parse an explicit CF unit and reference epoch without calendar math."""
        # References: CF Conventions 1.12, Section 4.4.1.
        # Expression: numeric_time unit since reference_datetime.
        # Parameters: units_string contains the unit scale and reference epoch.
        if not isinstance(units_string, str):
            raise ValidationError("CF time units must be a string")
        match = self._CF_UNITS_PATTERN.fullmatch(units_string.strip())
        if match is None:
            raise ValidationError(
                "CF time units must declare '<unit> since YYYY-MM-DD'"
            )
        unit = match.group(1).lower()
        reference_text = match.group(2)
        if match.group(3):
            reference_text = f"{reference_text}T{match.group(3)}"
        try:
            reference_time = datetime.fromisoformat(reference_text)
        except ValueError as exc:
            raise ValidationError(f"Invalid CF reference time: {reference_text}") from exc
        return unit, reference_time

    def detect_calendar(
        self,
        data_source: xr.Dataset,
        source_calendar: Optional[str] = None,
    ) -> str:
        """Return one explicit calendar or fail on absence, conflict, or invalidity."""
        if "time" not in data_source.variables:
            raise CalendarError("Dataset has no time coordinate")

        time_variable = data_source["time"]
        declarations: List[Tuple[str, str]] = []
        attr_calendar = time_variable.attrs.get("calendar")
        encoding_calendar = time_variable.encoding.get("calendar")
        if attr_calendar is not None:
            declarations.append(("time.attrs", str(attr_calendar)))
        if encoding_calendar is not None:
            declarations.append(("time.encoding", str(encoding_calendar)))
        if source_calendar is not None:
            declarations.append(("source_profile", str(source_calendar)))
        if not declarations:
            raise CalendarError(
                "RELIC strict external I/O requires an explicit calendar declaration"
            )

        normalized = [
            (source, self.cf_processor._normalize_calendar(calendar))
            for source, calendar in declarations
        ]
        unique_calendars = {calendar for _, calendar in normalized}
        if len(unique_calendars) != 1:
            evidence = ", ".join(
                f"{source}={calendar}" for source, calendar in normalized
            )
            raise CalendarError(f"Conflicting calendar declarations: {evidence}")
        return normalized[0][1]

    def detect_time_units(
        self,
        data_source: xr.Dataset,
        source_units: Optional[str] = None,
    ) -> str:
        """Return one explicit CF time-unit declaration or fail closed."""
        # References: CF Conventions 1.12, Section 4.4.1.
        # Expression: t = u * scale + epoch for one declared unit/epoch pair.
        # Parameters: u is the numeric coordinate; scale and epoch come from units.
        if "time" not in data_source.variables:
            raise ValidationError("Dataset has no time coordinate")

        time_variable = data_source["time"]
        declarations: List[Tuple[str, str]] = []
        attr_units = time_variable.attrs.get("units")
        encoding_units = time_variable.encoding.get("units")
        if attr_units is not None:
            declarations.append(("time.attrs", str(attr_units).strip()))
        if encoding_units is not None:
            declarations.append(("time.encoding", str(encoding_units).strip()))
        if source_units is not None:
            declarations.append(("source_profile", str(source_units).strip()))
        if not declarations:
            raise ValidationError(
                "RELIC strict external I/O requires explicit CF time units"
            )

        for _, units in declarations:
            self.parse_time_units(units)
        unique_units = {units for _, units in declarations}
        if len(unique_units) != 1:
            evidence = ", ".join(
                f"{source}={units}" for source, units in declarations
            )
            raise ValidationError(f"Conflicting CF time units: {evidence}")
        return declarations[0][1]

    def create_time_coordinate(
        self,
        values: Union[Tensor, List[float], Tuple[float, ...], Any],
        units: str,
        calendar: Optional[str] = None,
        compute_bounds: bool = True,
        bounds: Optional[Union[Tensor, List[List[float]], Any]] = None,
        frequency: Optional[str] = None,
    ) -> TimeCoordinate:
        """Create one canonical Torch time coordinate from boundary data."""
        # References: CF Conventions 1.12, Sections 4.4 and 7.1.
        # Expression: I_i = [b_i0, b_i1), with b_i0 < b_i1.
        # Parameters: values are coordinate labels; units/calendar define time;
        # bounds or frequency provide interval-support evidence.
        if calendar is None:
            raise CalendarError("An explicit calendar is required")
        self.parse_time_units(units)
        coordinate = self.cf_processor.parse_cf_time(values, units, calendar)

        coordinate_bounds: Optional[Tensor] = None
        if bounds is not None:
            coordinate_bounds = torch.as_tensor(
                bounds,
                dtype=coordinate.values.dtype,
                device=coordinate.values.device,
            )
        elif compute_bounds:
            if frequency is not None:
                coordinate_bounds = self.cf_processor.create_time_bounds(
                    coordinate,
                    frequency=frequency,
                    irregular_intervals=False,
                )
            elif coordinate.values.numel() == 1:
                raise ValidationError(
                    "Cannot infer singleton time bounds without explicit bounds or frequency"
                )
            else:
                coordinate_bounds = self.cf_processor._create_irregular_bounds(
                    coordinate.values
                )

        result = TimeCoordinate(
            values=coordinate.values,
            units=coordinate.units,
            calendar=coordinate.calendar,
            bounds=coordinate_bounds,
        )
        issues = self.validate_time_coordinate(result)
        if issues:
            raise ValidationError("; ".join(issues))
        return result

    def convert_calendar(
        self,
        time_coord: TimeCoordinate,
        target_calendar: str,
    ) -> TimeCoordinate:
        """Return identity only; reject unbound cross-calendar semantics."""
        normalized_target = self.cf_processor._normalize_calendar(target_calendar)
        if time_coord.calendar == normalized_target:
            return time_coord
        raise CalendarError(
            "Cross-calendar conversion requires an explicit canonical mapping policy"
        )

    def validate_time_coordinate(self, time_coord: TimeCoordinate) -> List[str]:
        """Validate canonical coordinate invariants with Torch operations."""
        # References: CF Conventions 1.12, Sections 4.4 and 7.1.
        # Expression: finite(u), u_i < u_(i+1), b_i0 < b_i1, u_i in I_i.
        # Parameters: u is the numeric coordinate and b contains interval bounds.
        issues: List[str] = []
        if not isinstance(time_coord, TimeCoordinate):
            return ["Time coordinate does not use the canonical temporal owner"]
        try:
            self.parse_time_units(time_coord.units)
            normalized = self.cf_processor._normalize_calendar(time_coord.calendar)
            if normalized != time_coord.calendar:
                issues.append("Time calendar is not normalized")
        except (CalendarError, ValidationError, TypeError, ValueError) as exc:
            issues.append(str(exc))
            return issues

        values = time_coord.values
        if not torch.all(torch.isfinite(values)):
            issues.append("Time coordinate values must be finite")
        if values.numel() > 1 and not torch.all(values[1:] > values[:-1]):
            issues.append("Time coordinate is not strictly increasing")

        bounds = time_coord.bounds
        if bounds is None:
            return issues
        if bounds.ndim != 2 or bounds.shape != (values.shape[0], 2):
            issues.append("Time bounds must have shape [time, 2]")
            return issues
        if not torch.all(torch.isfinite(bounds)):
            issues.append("Time bounds must be finite")
        if not torch.all(bounds[:, 0] < bounds[:, 1]):
            issues.append("Every time bound must have positive duration")
        if bounds.shape[0] > 1 and torch.any(bounds[1:, 0] < bounds[:-1, 1]):
            issues.append("Time bounds must not overlap")
        if torch.any(values < bounds[:, 0]) or torch.any(values > bounds[:, 1]):
            issues.append("Time coordinate values must lie within their bounds")
        return issues

    def to_cf_compliant(self, time_coord: TimeCoordinate) -> Dict[str, Any]:
        """Serialize a validated canonical coordinate at the xarray boundary."""
        issues = self.validate_time_coordinate(time_coord)
        if issues:
            raise ValidationError("; ".join(issues))
        time_attrs: Dict[str, str] = {
            "standard_name": "time",
            "long_name": "time",
            "units": time_coord.units,
            "calendar": time_coord.calendar,
            "axis": "T",
        }
        if time_coord.bounds is not None:
            time_attrs["bounds"] = "time_bounds"

        # NumPy conversion is restricted to this xarray serialization boundary.
        result: Dict[str, Any] = {
            "time": {
                "data": time_coord.values.detach().cpu().numpy(),
                "attrs": time_attrs,
                "dims": ["time"],
            }
        }
        if time_coord.bounds is not None:
            result["time_bounds"] = {
                "data": time_coord.bounds.detach().cpu().numpy(),
                "attrs": {
                    "long_name": "time bounds",
                    "comment": "Half-open support interval for each time coordinate",
                },
                "dims": ["time", "bounds"],
            }
        return result


__all__ = [
    "CalendarError",
    "TimeCoordinate",
    "TimeHandler",
    "ValidationError",
]
