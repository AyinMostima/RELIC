"""
Temporal Scaling Core Module

This module provides the foundational time weight generation and period indexing
functionality for temporal aggregation and disaggregation operations in the
RELIC framework. Implements CF-1.8 compliant time processing with support
for multiple calendar systems.

Key Features:
- CF-aware time coordinate processing with units and calendar validation
- Time weight generation based on exact interval overlaps
- Support for 5 standard calendar types: gregorian, proleptic_gregorian, noleap, all_leap, 360_day
- Period indexing with coverage threshold evaluation
- Vectorized PyTorch operations for performance
- Comprehensive error handling and validation

Mathematical Framework:
    Time weight computation based on interval overlap:
    w_ij = overlap(source_i, target_j) / duration(target_j)
    
    Conservation constraint:
    Sigma_i w_ij <= 1 for each target interval j
    
    Coverage evaluation:
    coverage_j = Sigma_i w_ij
    
    Valid aggregation condition:
    coverage_j >= coverage_threshold

Classes:
    TimeWeightGenerator: Core time weight computation engine
    PeriodIndexer: Period boundary and indexing manager
    CFTimeProcessor: CF-1.8 compliant time coordinate processor

Functions:
    update_cell_methods: CF cell_methods string manipulation

References:
    - CF Conventions Version 1.8: http://cfconventions.org/
    - UDUNITS-2 Library Documentation
    - NetCDF Climate and Forecast Metadata Conventions
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from relic.core import device_policy
from relic.coupling import SCALES_TEMPORAL_NS
from relic.data.standards.dimensionality import (
    Dimensionality,
    dimensionality_from_unit,
)
from relic.data.standards.unit_converter import ConversionContext, unit_converter
from relic.parameters import get_global_manager
from relic.parameters.namespaces import resolve_namespace
from relic.utils.time_constants import (
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
)

from .types import TemporalSemantics

MINUTES_PER_HOUR = int(SECONDS_PER_HOUR / SECONDS_PER_MINUTE)

# Standard frequency mappings
FREQUENCY_HIERARCHY = {
    'H': {'name': 'hourly', 'seconds': int(SECONDS_PER_HOUR)},
    'D': {'name': 'daily', 'seconds': int(SECONDS_PER_DAY)},
    'W': {'name': 'weekly', 'seconds': int(7 * SECONDS_PER_DAY)},
    'M': {'name': 'monthly', 'calendar_dependent': True},
    'Q': {'name': 'quarterly', 'calendar_dependent': True, 'requires_anchor': True},
    'S': {'name': 'seasonal', 'calendar_dependent': True, 'requires_anchor': True},
    'Y': {'name': 'yearly', 'calendar_dependent': True},
}

def _load_calendar_definitions(parameter_manager=None) -> Dict[str, Dict[str, Any]]:
    """
    Load calendar metadata from the centralized parameter system.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of calendar name to metadata fields (description, days_per_year,
        months_per_year, leap_rule).
    """
    manager = (
        parameter_manager
        if parameter_manager is not None
        else get_global_manager()
    )
    if manager is None:
        raise RuntimeError("ParameterManager is required to load calendar definitions.")
    namespace_base = resolve_namespace(SCALES_TEMPORAL_NS)
    calendar_namespace = f"{namespace_base}.calendars"
    params = manager.get_namespace_parameters(
        namespace=calendar_namespace,
        include_subnamespaces=True,
        return_specs=False,
    )
    if not params:
        if hasattr(manager, "_preload_parameters"):
            manager._preload_parameters(warmup_cache=False)
        params = manager.get_namespace_parameters(
            namespace=calendar_namespace,
            include_subnamespaces=True,
            return_specs=False,
        )
    if not params and hasattr(manager, "database"):
        try:
            db_params = manager.database.load_all()
            params = {
                key: (spec.value if hasattr(spec, "value") else spec)
                for key, spec in db_params.items()
                if key.startswith(f"{calendar_namespace}.")
            }
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001
            raise ValueError(
                f"Failed to load calendar definitions from database namespace {calendar_namespace}"
            ) from exc
    if not params:
        raise ValueError(f"No calendar definitions found in namespace {calendar_namespace}")

    definitions: Dict[str, Dict[str, Any]] = {}
    for full_key, value in params.items():
        parts = full_key.split(".")
        try:
            idx = parts.index("calendars")
            calendar_name = parts[idx + 1]
            field = ".".join(parts[idx + 2 :])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid calendar parameter key: {full_key}") from exc
        definitions.setdefault(calendar_name, {})[field] = value
    return definitions


def get_calendar_definitions(parameter_manager=None) -> Dict[str, Dict[str, Any]]:
    """Public accessor for calendar definitions."""
    return _load_calendar_definitions(parameter_manager)


def get_supported_calendar_names(parameter_manager=None) -> List[str]:
    """Return list of supported calendar identifiers."""
    return sorted(get_calendar_definitions(parameter_manager).keys())


class ValidationError(Exception):
    """Raised when input data validation fails."""


class CalendarError(Exception):
    """Raised when calendar operations fail."""


class AggregationError(Exception):
    """Raised when aggregation operations fail."""


class DisaggregationError(Exception):
    """Raised when disaggregation operations fail."""


_CF_TIME_UNITS_PATTERN = re.compile(
    r"^(seconds?|minutes?|hours?|days?|months?|years?) since "
    r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?)$",
    re.IGNORECASE,
)
_TIME_UNIT_NAMES = {
    "second": "second",
    "seconds": "second",
    "minute": "minute",
    "minutes": "minute",
    "hour": "hour",
    "hours": "hour",
    "day": "day",
    "days": "day",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
}
_TIME_UNIT_CONVERTER_LABELS = {
    "second": "s",
    "minute": "minute",
    "hour": "hour",
    "day": "day",
    "month": "month",
    "year": "year",
}
_RATE_TIME_UNIT_ALIASES = {
    "second": ("seconds", "second", "sec", "s"),
    "minute": ("minutes", "minute", "min"),
    "hour": ("hours", "hour", "hr", "h"),
    "day": ("days", "day", "d"),
    "month": ("months", "month"),
    "year": ("years", "year"),
}


def _parse_iso_timestamp(iso_time: str) -> tuple[int, int, int, int, int, float]:
    pattern = (
        r"^(\d{4})-(\d{2})-(\d{2})"
        r"(?:[ T](\d{2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?$"
    )
    match = re.fullmatch(pattern, iso_time.strip())
    if match is None:
        raise ValidationError(f"Invalid ISO timestamp: {iso_time}")
    values = (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)) if match.group(4) is not None else 0,
        int(match.group(5)) if match.group(5) is not None else 0,
        float(match.group(6)) if match.group(6) is not None else 0.0,
    )
    _, month, day, hour, minute, second = values
    if not (
        1 <= month <= 12
        and 1 <= day <= 31
        and 0 <= hour <= 23
        and 0 <= minute <= 59
        and 0.0 <= second < 60.0
    ):
        raise ValidationError(f"Invalid ISO timestamp fields: {iso_time}")
    return values


def _parse_cf_time_units(
    units: str,
) -> tuple[str, tuple[int, int, int, int, int, float]]:
    if not isinstance(units, str) or not units.strip():
        raise ValidationError("CF time units must be a non-empty string")
    match = _CF_TIME_UNITS_PATTERN.fullmatch(units.strip())
    if match is None:
        raise ValidationError(
            f"Invalid CF units string (CF time units required): {units}"
        )
    unit_name = _TIME_UNIT_NAMES[match.group(1).lower()]
    return unit_name, _parse_iso_timestamp(match.group(2))


def _normalize_plain_time_unit(unit_name: str) -> str:
    if not isinstance(unit_name, str) or not unit_name.strip():
        raise ValidationError("integration_time_unit must be a non-empty string")
    normalized = unit_name.strip().lower()
    if normalized == "s":
        normalized = "second"
    canonical_name = _TIME_UNIT_NAMES.get(normalized)
    if canonical_name is None:
        raise ValidationError(f"Unsupported integration time unit: {unit_name}")
    return _TIME_UNIT_CONVERTER_LABELS[canonical_name]


def _normalize_unit_text(units: str) -> str:
    return " ".join(units.strip().split())


def _extract_rate_unit_contract(units: str) -> tuple[str, str]:
    normalized = _normalize_unit_text(units)
    for canonical_name, aliases in _RATE_TIME_UNIT_ALIASES.items():
        for alias in aliases:
            escaped = re.escape(alias)
            inverse_match = re.fullmatch(
                rf"(?P<numerator>.+?)\s+{escaped}(?:\^-1|-1)",
                normalized,
            )
            division_match = re.fullmatch(
                rf"(?P<numerator>.+?)\s*/\s*{escaped}",
                normalized,
            )
            match = inverse_match or division_match
            if match is not None:
                numerator = _normalize_unit_text(match.group("numerator"))
                return numerator, _TIME_UNIT_CONVERTER_LABELS[canonical_name]
    raise ValidationError(
        "rate units must explicitly end in one supported inverse time unit"
    )


def _extract_integrated_mean_contract(units: str) -> tuple[str, str]:
    normalized = _normalize_unit_text(units)
    for canonical_name, aliases in _RATE_TIME_UNIT_ALIASES.items():
        for alias in aliases:
            match = re.fullmatch(
                rf"(?P<numerator>.+?)\s+{re.escape(alias)}",
                normalized,
            )
            if match is not None:
                numerator = _normalize_unit_text(match.group("numerator"))
                return numerator, _TIME_UNIT_CONVERTER_LABELS[canonical_name]
    raise ValidationError(
        "interval_mean measure units must explicitly include the integration time unit"
    )


def _convert_time_tensor(
    values: Tensor,
    source_unit: str,
    target_unit: str,
) -> Tensor:
    """Convert time values through the canonical unit converter."""
    # References:
    #   - CF Conventions 1.13, Section 4.4.
    #   - BIPM SI Brochure, 9th edition, time-unit definitions.
    # Mathematical expression:
    #   t_target = U(t_source; source_unit -> target_unit).
    # Parameter meanings:
    #   U is the canonical UnitConverter transformation.
    # Units:
    #   Input uses source_unit and output uses target_unit.
    # Domain/constraints:
    #   Both labels must be recognized and conversion must be executable.
    # Sign convention:
    #   Conversion factors are positive and preserve temporal orientation.
    try:
        if source_unit == target_unit or unit_converter.can_convert(
            source_unit,
            target_unit,
            "generic",
        ):
            converted = unit_converter.convert(
                values,
                "generic",
                source_unit,
                target_unit,
                ConversionContext(),
            )
        elif (
            unit_converter.can_convert(source_unit, "s", "generic")
            and unit_converter.can_convert("s", target_unit, "generic")
        ):
            seconds = unit_converter.convert(
                values,
                "generic",
                source_unit,
                "s",
                ConversionContext(),
            )
            converted = unit_converter.convert(
                seconds,
                "generic",
                "s",
                target_unit,
                ConversionContext(),
            )
        else:
            raise ValueError(f"No canonical conversion from {source_unit} to {target_unit}")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValidationError(
            f"CF time unit conversion failed: {source_unit} -> {target_unit}"
        ) from exc
    if not isinstance(converted, Tensor):
        raise ValidationError("Canonical time unit conversion did not return a tensor")
    return converted


def _convert_value_tensor(
    values: Tensor,
    source_units: str,
    target_units: str,
) -> Tensor:
    """Convert mapped values through the canonical unit owner."""
    # References:
    #   - CF Conventions 1.13, Sections 3.1 and 4.4.
    #   - BIPM SI Brochure, 9th edition.
    # Mathematical expression:
    #   q_target = U(q_source; source_units -> target_units).
    # Parameter meanings:
    #   q is a temporal value and U is the canonical UnitConverter.
    # Units:
    #   Input and output labels are declared by TemporalUnitContract.
    # Domain/constraints:
    #   Unknown or unsupported same-dimension scale changes fail closed.
    # Sign convention:
    #   Unit conversion preserves the declared physical sign.
    try:
        converted = unit_converter.convert(
            values,
            "generic",
            source_units,
            target_units,
            ConversionContext(),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValidationError(
            f"Value unit conversion failed: {source_units} -> {target_units}"
        ) from exc
    if not isinstance(converted, Tensor):
        raise ValidationError("Canonical value unit conversion did not return a tensor")
    return converted


def _align_time_values_to_source_units(
    source_units: str,
    target_units: str,
    target_values: Tensor,
) -> tuple[Tensor, str]:
    source_name, source_reference = _parse_cf_time_units(source_units)
    target_name, target_reference = _parse_cf_time_units(target_units)
    if source_reference != target_reference:
        raise ValidationError(
            "Source and target CF time reference timestamps must match before mapping"
        )
    source_label = _TIME_UNIT_CONVERTER_LABELS[source_name]
    target_label = _TIME_UNIT_CONVERTER_LABELS[target_name]
    return (
        _convert_time_tensor(target_values, target_label, source_label),
        source_label,
    )


class TemporalBoundsRelation(str, Enum):
    """Actual source-to-target relation derived only from explicit bounds."""

    IDENTITY = "identity"
    COARSENING = "coarsening"
    REFINEMENT = "refinement"
    NON_NESTED = "non_nested"


@dataclass
class TimeCoordinate:
    """
    CF-compliant time coordinate with units and calendar.
    
    Attributes:
        values : Tensor
            Time coordinate values
        units : str
            CF-compliant units string (e.g., "days since 1850-01-01")
        calendar : str
            Calendar type
        bounds : Optional[Tensor]
            Time bounds [N, 2]
    """
    values: Tensor
    units: str
    calendar: str
    bounds: Optional[Tensor] = None
    
    def __post_init__(self):
        """Validate time coordinate after initialization."""
        supported_calendars = get_supported_calendar_names()
        if self.calendar not in supported_calendars:
            raise CalendarError(f"Unsupported calendar: {self.calendar}")
        
        if self.bounds is not None:
            if self.bounds.shape[-1] != 2:
                raise ValidationError("Time bounds must have shape [..., 2]")
            if self.bounds.shape[0] != self.values.shape[0]:
                raise ValidationError("Bounds and values must have same length")


class _DefensiveTensorSnapshot:
    """Return differentiable tensor snapshots from immutable public records."""

    __slots__ = ()

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if isinstance(value, Tensor):
            return value.clone()
        return value


@dataclass(frozen=True, slots=True)
class TemporalOverlapPlan(_DefensiveTensorSnapshot):
    """Immutable structural view of one source/target interval overlap."""

    source_bounds: Tensor
    target_bounds: Tensor
    source_durations: Tensor
    target_durations: Tensor
    overlap: Tensor
    source_fractions: Tensor
    target_fractions: Tensor
    geometric_source_coverage: Tensor
    geometric_target_coverage: Tensor
    units: str
    coordinate_unit: str
    calendar: str
    relation: TemporalBoundsRelation


@dataclass(frozen=True, slots=True)
class TemporalUnitContract:
    """Explicit value, mapped-measure, and integration-time unit contract."""

    source_units: str
    target_units: str
    measure_units: str
    integration_time_unit: str | None = None


@dataclass(frozen=True, slots=True)
class TemporalGapDiagnostic(_DefensiveTensorSnapshot):
    """Coverage gaps kept separate from mapping and physical-ledger residuals."""

    geometric_gap_fraction: Tensor
    missing_data_gap_fraction: Tensor
    total_gap_fraction: Tensor
    observed_duration: Tensor
    target_duration: Tensor
    units: str
    kind: str
    imputation: str


@dataclass(frozen=True, slots=True)
class TemporalKernelResult(_DefensiveTensorSnapshot):
    """Canonical numerical output plus support and coverage diagnostics."""

    values: Tensor
    observed_measure: Tensor
    observed_duration: Tensor
    geometric_coverage: Tensor
    valid_coverage: Tensor
    complete_coverage: Tensor
    semantics: TemporalSemantics
    output_semantics: TemporalSemantics
    unit_contract: TemporalUnitContract
    gap_diagnostic: TemporalGapDiagnostic


@dataclass(frozen=True, slots=True)
class TemporalResidualResult(_DefensiveTensorSnapshot):
    """Mapping residual separated from its signed-cancellation-safe scale."""

    residual: Tensor
    source_activity: Tensor
    relative_residual: Tensor | None
    checked: Tensor
    law_id: str
    kind: str
    units: str
    support: str
    reduction_axes: tuple[int, int]
    sign_convention: str
    geometric_coverage: Tensor
    valid_coverage: Tensor
    accepted_target_mask: Tensor
    imputation: str
    q_ref_provenance: str | None
    source_semantics: TemporalSemantics
    target_semantics: TemporalSemantics


class CFTimeProcessor:
    """
    CF-1.8 compliant time coordinate processor.
    
    Provides parsing, validation, and conversion of CF-compliant time coordinates
    with support for multiple calendar systems and units.
    """
    
    def __init__(self, device: Optional[torch.device] = None, parameter_manager: Optional[Any] = None):
        """
        Initialize CF time processor.
        
        Parameters:
            device : Optional[torch.device]
            Computation device
        """
        self.device = device or (
            device_policy.resolve_device(default="auto")
        )
        self.parameter_manager = (
            parameter_manager
            if parameter_manager is not None
            else get_global_manager()
        )
        if self.parameter_manager is None:
            raise RuntimeError(
                "ParameterManager is required for CFTimeProcessor; ensure it is initialized or pass explicitly."
            )
    
    def parse_cf_time(
        self, 
        time_data: Union[Tensor, List, Tuple],
        units: str,
        calendar: str = 'gregorian'
    ) -> TimeCoordinate:
        """
        Parse CF-compliant time coordinates.
        
        Parameters:
            time_data : Union[Tensor, List, Tuple]
                Time coordinate values
            units : str
                CF units string (e.g., "days since 1850-01-01")
            calendar : str
                Calendar type
                
        Returns:
            TimeCoordinate: Parsed time coordinate
            
        Raises:
            ValidationError: If units or calendar are invalid
            CalendarError: If calendar is unsupported
        """
        # Convert to tensor
        if not isinstance(time_data, Tensor):
            time_values = torch.tensor(time_data, dtype=torch.float64, device=self.device)
        else:
            time_values = time_data.to(device=self.device, dtype=torch.float64)
        
        # Validate and normalize calendar
        calendar_normalized = self._normalize_calendar(calendar)
        
        # Validate units string
        if not self._is_valid_cf_units(units):
            raise ValidationError(f"Invalid CF units string: {units}")
        
        # Validate time ordering
        if len(time_values) > 1:
            if not torch.all(time_values[1:] > time_values[:-1]):
                raise ValidationError(
                    "Time coordinates are not strictly increasing"
                )
        
        return TimeCoordinate(
            values=time_values,
            units=units,
            calendar=calendar_normalized
        )
    
    def _normalize_calendar(self, calendar: str) -> str:
        """Normalize calendar name to standard form."""
        calendar_lower = calendar.lower().replace('-', '_')
        supported_calendars = get_supported_calendar_names(self.parameter_manager)
        
        # Handle aliases
        if calendar_lower == 'standard':
            return 'gregorian'
        if calendar_lower == '365_day':
            return 'noleap'
        if calendar_lower == '366_day':
            return 'all_leap'
        if calendar_lower in supported_calendars:
            return calendar_lower
        raise CalendarError(f"Unsupported calendar: {calendar}")
    
    def _is_valid_cf_units(self, units: str) -> bool:
        """Check if units string follows CF conventions."""
        try:
            _parse_cf_time_units(units)
        except ValidationError:
            return False
        return True

    def _create_irregular_bounds(self, time_values: Tensor) -> Tensor:
        """Create bounds from irregular time coordinate spacing."""
        n_times = len(time_values)
        bounds = torch.zeros(n_times, 2, dtype=time_values.dtype, device=time_values.device)

        if n_times == 1:
            dt = torch.tensor(1.0, dtype=time_values.dtype, device=time_values.device)
            bounds[0, 0] = time_values[0] - dt / 2.0
            bounds[0, 1] = time_values[0] + dt / 2.0
            return bounds

        deltas = time_values[1:] - time_values[:-1]
        bounds[0, 0] = time_values[0] - deltas[0] / 2.0
        bounds[0, 1] = time_values[0] + deltas[0] / 2.0

        for i in range(1, n_times - 1):
            bounds[i, 0] = (time_values[i - 1] + time_values[i]) / 2.0
            bounds[i, 1] = (time_values[i] + time_values[i + 1]) / 2.0

        bounds[-1, 0] = (time_values[-2] + time_values[-1]) / 2.0
        bounds[-1, 1] = time_values[-1] + deltas[-1] / 2.0

        return bounds

    def create_time_bounds(
        self,
        time_coord: TimeCoordinate,
        frequency: str,
        irregular_intervals: bool = False
    ) -> Tensor:
        """
        Create time bounds for given frequency.
        
        Parameters:
            time_coord : TimeCoordinate
                Time coordinate
            frequency : str
                Frequency code ('H', 'D', 'M', 'Y')
            irregular_intervals : bool
                If True, derive bounds from adjacent time values instead of
                assuming fixed-frequency spacing.
                
        Returns:
            Tensor: Time bounds [N, 2]
            
        Mathematical Expression:
            For regular intervals:
            bounds[i] = [time[i] - dt/2, time[i] + dt/2]
            
            For irregular intervals (like months):
            bounds computed based on calendar-specific rules
        """
        time_values = time_coord.values
        indexer = PeriodIndexer(
            device=self.device,
            parameter_manager=self.parameter_manager,
        )
        frequency_code, _step_months, _anchor_month = (
            indexer._parse_period_frequency(frequency)
        )
        if irregular_intervals:
            if frequency_code in {'M', 'Q', 'S', 'Y'}:
                raise ValidationError(
                    "irregular midpoint inference is unsupported for "
                    "calendar-dependent frequencies"
                )
            return self._create_irregular_bounds(time_values)

        if frequency_code in ['H', 'D', 'W']:
            # Regular intervals
            freq_seconds = FREQUENCY_HIERARCHY[frequency_code]['seconds']
            
            # Convert to same units as time coordinate
            if 'seconds' in time_coord.units:
                dt = freq_seconds / 2.0
            elif 'hours' in time_coord.units:
                dt = freq_seconds / (2.0 * float(SECONDS_PER_HOUR))
            elif 'days' in time_coord.units:
                dt = freq_seconds / (2.0 * float(SECONDS_PER_DAY))
            else:
                raise ValidationError(f"Cannot create bounds for units: {time_coord.units}")
            
            dt = torch.tensor(dt, dtype=time_values.dtype, device=time_values.device)
            bounds = torch.stack([
                time_values - dt,
                time_values + dt
            ], dim=1)
            
        elif frequency_code in {'M', 'Q', 'S', 'Y'}:
            bounds = self._create_calendar_period_bounds(time_coord, frequency)
        else:
            raise ValidationError(f"Unsupported frequency for bounds creation: {frequency}")
        
        return bounds
    
    def _create_calendar_period_bounds(
        self,
        time_coord: TimeCoordinate,
        frequency: str,
    ) -> Tensor:
        """
        Build exact period bounds with the canonical PeriodIndexer.

        References:
            CF Conventions 1.13, Section 4.4, time coordinates and calendars.
        Mathematical expression:
            bounds_i = [start_C,f(t_i), advance_C,f(start_C,f(t_i))].
        Parameter meanings:
            t_i is a coordinate value, C is the selected calendar, and f is
            the monthly or yearly frequency. Bounds retain time_coord.units.
        """
        time_values = time_coord.values
        if not torch.all(torch.isfinite(time_values)):
            raise ValidationError("Time coordinates must be finite")

        calendar = self._normalize_calendar(time_coord.calendar)
        indexer = PeriodIndexer(
            device=self.device,
            parameter_manager=self.parameter_manager,
        )
        bounds_values: List[Tuple[float, float]] = []
        for value in time_values.detach().cpu().tolist():
            coordinate_datetime = indexer._units_value_to_datetime(
                float(value),
                time_coord.units,
                calendar,
            )
            period_start = indexer._normalize_period_start(
                coordinate_datetime,
                frequency,
                calendar,
            )
            period_end = indexer._increment_datetime(
                period_start,
                frequency,
                calendar,
            )
            left = indexer._datetime_to_units_value(
                period_start,
                time_coord.units,
                calendar,
            )
            right = indexer._datetime_to_units_value(
                period_end,
                time_coord.units,
                calendar,
            )
            bounds_values.append((left, right))

        return torch.tensor(
            bounds_values,
            dtype=time_values.dtype,
            device=time_values.device,
        )

    def _create_monthly_bounds(self, time_coord: TimeCoordinate) -> Tensor:
        """Create exact calendar-aware monthly bounds."""
        return self._create_calendar_period_bounds(time_coord, "M")

    def _create_yearly_bounds(self, time_coord: TimeCoordinate) -> Tensor:
        """Create exact calendar-aware yearly bounds."""
        return self._create_calendar_period_bounds(time_coord, "Y")


def _materialize_device(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None:
        return torch.empty(0, device=device).device
    return device


def _classify_bounds_relation(
    source_bounds: Tensor,
    target_bounds: Tensor,
    geometric_source_coverage: Tensor,
    geometric_target_coverage: Tensor,
    tolerance: float,
) -> TemporalBoundsRelation:
    """Classify scale direction from the actual converted interval partitions."""
    # References:
    #   - Jones (1999), first-order conservative remapping.
    # Mathematical expression:
    #   target coarsens source iff boundary(T) is a strict subset of boundary(S).
    # Parameter meanings:
    #   S and T are fully mutually covered source and target interval unions.
    # Units:
    #   Both bound sets use the plan coordinate unit.
    # Domain/constraints:
    #   Identity is exact; nested relations require complete mutual coverage.
    # Sign convention:
    #   Coarsening and refinement describe the source-to-target direction.
    if torch.equal(source_bounds, target_bounds):
        return TemporalBoundsRelation.IDENTITY
    mutually_covered = bool(
        torch.all(torch.abs(geometric_source_coverage - 1.0) <= tolerance)
        and torch.all(torch.abs(geometric_target_coverage - 1.0) <= tolerance)
    )
    if not mutually_covered:
        return TemporalBoundsRelation.NON_NESTED
    source_edges = source_bounds.reshape(-1)
    target_edges = target_bounds.reshape(-1)
    target_in_source = bool(
        torch.all(torch.any(target_edges[:, None] == source_edges[None, :], dim=1))
    )
    source_in_target = bool(
        torch.all(torch.any(source_edges[:, None] == target_edges[None, :], dim=1))
    )
    if target_in_source and not source_in_target:
        return TemporalBoundsRelation.COARSENING
    if source_in_target and not target_in_source:
        return TemporalBoundsRelation.REFINEMENT
    if source_in_target and target_in_source:
        return TemporalBoundsRelation.IDENTITY
    return TemporalBoundsRelation.NON_NESTED


class TimeWeightGenerator:
    """
    Time weight generation engine for temporal aggregation/disaggregation.
    
    Computes overlap-based time weights between source and target intervals
    with exact calendar-aware calculations and coverage evaluation.
    """
    
    def __init__(self, device: Optional[torch.device] = None, parameter_manager: Optional[Any] = None):
        """
        Initialize time weight generator.
        
        Parameters:
            device : Optional[torch.device]
                Computation device
        """
        requested_device = torch.device(
            device if device is not None else device_policy.resolve_device(default="auto")
        )
        self.device = _materialize_device(requested_device)
        self.parameter_manager = (
            parameter_manager
            if parameter_manager is not None
            else get_global_manager()
        )
        if self.parameter_manager is None:
            raise RuntimeError(
                "ParameterManager is required for TimeWeightGenerator; ensure it is initialized or passed explicitly."
            )
        self.cf_processor = CFTimeProcessor(
            device=self.device,
            parameter_manager=self.parameter_manager,
        )

    @staticmethod
    def _validate_interval_bounds(bounds: Tensor, name: str) -> Tensor:
        if not isinstance(bounds, Tensor):
            raise TypeError(f"{name} bounds must be a torch.Tensor")
        if bounds.layout != torch.strided or bounds.device.type == "meta":
            raise ValidationError(f"{name} bounds must be materialized strided tensors")
        if bounds.dtype not in {torch.float32, torch.float64}:
            raise ValidationError(f"{name} bounds must use float32 or float64")
        if bounds.dim() != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0:
            raise ValidationError(f"{name} bounds must have non-empty shape [N, 2]")
        if not torch.all(torch.isfinite(bounds)):
            raise ValidationError(f"{name} bounds must contain finite values")

        durations = bounds[:, 1] - bounds[:, 0]
        if torch.any(durations <= 0):
            raise ValidationError(f"{name} bounds must have strictly positive durations")
        if bounds.shape[0] > 1 and torch.any(
            bounds[1:, 0] < bounds[:-1, 1]
        ):
            raise ValidationError(f"{name} bounds overlap or are out of order")
        return durations

    def build_overlap_plan(
        self,
        source_time: TimeCoordinate,
        target_time: TimeCoordinate,
    ) -> TemporalOverlapPlan:
        """Build the sole raw overlap plan for one bounded temporal mapping."""
        if source_time.bounds is None:
            raise ValidationError("Source time must have bounds for weight computation")
        if target_time.bounds is None:
            raise ValidationError("Target time must have bounds for weight computation")
        if source_time.calendar != target_time.calendar:
            raise ValidationError(
                f"Different calendars: source={source_time.calendar}, "
                f"target={target_time.calendar}"
            )
        calendar_indexer = PeriodIndexer(
            device=self.device,
            parameter_manager=self.parameter_manager,
        )
        for units in (source_time.units, target_time.units):
            _, reference_datetime = _parse_cf_time_units(units)
            calendar_indexer._datetime_to_day_index(
                reference_datetime,
                source_time.calendar,
            )
        source_bounds = source_time.bounds.clone()
        target_bounds = target_time.bounds.clone()
        if source_bounds.device != target_bounds.device:
            raise ValidationError("Source and target bounds must use the same device")
        if self.device != source_bounds.device:
            raise ValidationError(
                "TimeWeightGenerator device must match source and target bounds"
            )
        if source_bounds.dtype != target_bounds.dtype:
            raise ValidationError("Source and target bounds must use the same dtype")
        target_bounds, coordinate_unit = _align_time_values_to_source_units(
            source_time.units,
            target_time.units,
            target_bounds,
        )
        target_bounds = target_bounds.clone()

        source_durations = self._validate_interval_bounds(source_bounds, "source")
        target_durations = self._validate_interval_bounds(target_bounds, "target")

        # References:
        #   - Jones (1999), Mon. Weather Rev. 127:2204-2210.
        #   - ESMF Reference Manual, first-order conservative remapping.
        # Mathematical expression:
        #   O_ij = max(0, min(a_i_plus, b_j_plus) - max(a_i_minus, b_j_minus)).
        # Parameter meanings:
        #   O_ij is overlap duration; a and b are source and target bounds.
        # Units:
        #   Bounds, durations, and O_ij use source_time.units.
        # Domain/constraints:
        #   Half-open, finite, positive, ordered, non-overlapping intervals.
        # Sign convention:
        #   Durations and overlap weights are nonnegative.
        overlap_starts = torch.maximum(
            source_bounds[:, None, 0],
            target_bounds[None, :, 0],
        )
        overlap_ends = torch.minimum(
            source_bounds[:, None, 1],
            target_bounds[None, :, 1],
        )
        overlap = torch.clamp(overlap_ends - overlap_starts, min=0.0)
        source_fractions = overlap / source_durations[:, None]
        target_fractions = overlap / target_durations[None, :]
        geometric_source_coverage = torch.sum(source_fractions, dim=1)
        geometric_target_coverage = torch.sum(target_fractions, dim=0)

        coverage_terms = max(source_bounds.shape[0], target_bounds.shape[0], 1)
        tolerance = torch.finfo(overlap.dtype).eps * coverage_terms * 64.0
        if torch.any(geometric_source_coverage > 1.0 + tolerance):
            raise ValidationError("target bounds overlap or duplicate source coverage")
        if torch.any(geometric_target_coverage > 1.0 + tolerance):
            raise ValidationError("source bounds overlap or duplicate target coverage")
        relation = _classify_bounds_relation(
            source_bounds,
            target_bounds,
            geometric_source_coverage,
            geometric_target_coverage,
            tolerance,
        )

        return TemporalOverlapPlan(
            source_bounds=source_bounds,
            target_bounds=target_bounds,
            source_durations=source_durations,
            target_durations=target_durations,
            overlap=overlap,
            source_fractions=source_fractions,
            target_fractions=target_fractions,
            geometric_source_coverage=geometric_source_coverage,
            geometric_target_coverage=geometric_target_coverage,
            units=source_time.units,
            coordinate_unit=coordinate_unit,
            calendar=source_time.calendar,
            relation=relation,
        )


def _coerce_temporal_semantics(
    value: TemporalSemantics | str,
    field_name: str,
) -> TemporalSemantics:
    if isinstance(value, TemporalSemantics):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be TemporalSemantics or string")
    try:
        return TemporalSemantics(value.strip().lower())
    except ValueError as exc:
        raise ValidationError(f"Unsupported {field_name}: {value!r}") from exc


def _prepare_temporal_data(
    data: Tensor,
    *,
    expected_time_size: int,
    time_dim: int,
    name: str,
) -> tuple[Tensor, tuple[int, ...], int]:
    if not isinstance(data, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if data.layout != torch.strided or data.device.type == "meta":
        raise ValidationError(f"{name} must be a materialized strided tensor")
    if data.dtype not in {torch.float32, torch.float64}:
        raise ValidationError(f"{name} must use float32 or float64")
    if data.dim() < 1 or any(size == 0 for size in data.shape):
        raise ValidationError(f"{name} must have non-empty dimensions")
    if type(time_dim) is not int:
        raise TypeError("time_dim must be an integer axis")
    normalized_time_dim = time_dim if time_dim >= 0 else data.dim() + time_dim
    if normalized_time_dim < 0 or normalized_time_dim >= data.dim():
        raise ValidationError(f"time_dim {time_dim} is outside {name} rank")
    if data.shape[normalized_time_dim] != expected_time_size:
        raise ValidationError(
            f"{name} time axis has size {data.shape[normalized_time_dim]}; "
            f"expected {expected_time_size}"
        )
    moved = data.movedim(normalized_time_dim, 0)
    non_time_shape = tuple(moved.shape[1:])
    return moved.reshape(expected_time_size, -1), non_time_shape, normalized_time_dim


def _prepare_valid_mask(
    data: Tensor,
    data_flat: Tensor,
    *,
    valid_mask: Tensor | None,
    time_dim: int,
) -> Tensor:
    if torch.any(torch.isinf(data_flat)):
        raise ValidationError("temporal data contains infinite values")
    finite_mask = ~torch.isnan(data_flat)
    if valid_mask is None:
        return finite_mask
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must be a boolean torch.Tensor")
    if valid_mask.shape != data.shape or valid_mask.device != data.device:
        raise ValidationError("valid_mask must match temporal data shape and device")
    moved = valid_mask.movedim(time_dim, 0).reshape_as(data_flat)
    return moved & finite_mask


def _restore_time_axis(
    flat: Tensor,
    *,
    target_size: int,
    non_time_shape: tuple[int, ...],
    time_dim: int,
) -> Tensor:
    restored = flat.reshape((target_size, *non_time_shape))
    return restored.movedim(0, time_dim)


def _dimension_times_time(dimension: Dimensionality) -> Dimensionality:
    return Dimensionality(
        M=dimension.M,
        L=dimension.L,
        T=dimension.T + 1.0,
        Theta=dimension.Theta,
        Q=dimension.Q,
        N=dimension.N,
        J=dimension.J,
    )


def _unit_dimension(units: str, field_name: str) -> Dimensionality:
    if not isinstance(units, str) or not units.strip():
        raise ValidationError(f"{field_name} must be a non-empty unit string")
    dimension = dimensionality_from_unit(units.strip())
    if dimension is None:
        raise ValidationError(f"Unknown {field_name}: {units}")
    return dimension


def _validate_temporal_unit_contract(
    plan: TemporalOverlapPlan,
    unit_contract: TemporalUnitContract,
    source_semantics: TemporalSemantics,
    target_semantics: TemporalSemantics,
    reference: Tensor,
) -> Tensor:
    if not isinstance(unit_contract, TemporalUnitContract):
        raise TypeError("unit_contract must be a TemporalUnitContract")
    source_dimension = _unit_dimension(
        unit_contract.source_units,
        "source_units",
    )
    target_dimension = _unit_dimension(
        unit_contract.target_units,
        "target_units",
    )
    measure_dimension = _unit_dimension(
        unit_contract.measure_units,
        "measure_units",
    )

    if source_semantics is TemporalSemantics.INTERVAL_ACCUMULATION:
        if target_semantics is not TemporalSemantics.INTERVAL_ACCUMULATION:
            raise ValidationError(
                "interval_accumulation output must remain interval_accumulation"
            )
        if unit_contract.integration_time_unit is not None:
            raise ValidationError(
                "interval_accumulation must not declare integration_time_unit"
            )
        if not source_dimension.is_compatible(measure_dimension) or not (
            target_dimension.is_compatible(measure_dimension)
        ):
            raise ValidationError(
                "interval_accumulation source, target, and measure units are incompatible"
            )
        probe = torch.ones((), dtype=reference.dtype, device=reference.device)
        _convert_value_tensor(
            probe,
            unit_contract.source_units,
            unit_contract.measure_units,
        )
        _convert_value_tensor(
            probe,
            unit_contract.measure_units,
            unit_contract.target_units,
        )
        return probe

    integration_label = _normalize_plain_time_unit(
        unit_contract.integration_time_unit or ""
    )
    integration_dimension = _unit_dimension(
        integration_label,
        "integration_time_unit",
    )
    if not integration_dimension.is_compatible(Dimensionality(T=1.0)):
        raise ValidationError("integration_time_unit must have time dimensionality")

    if source_semantics is TemporalSemantics.INTERVAL_MEAN:
        if target_semantics is not TemporalSemantics.INTERVAL_MEAN:
            raise ValidationError("interval_mean output must remain interval_mean")
        if not source_dimension.is_compatible(target_dimension):
            raise ValidationError("interval_mean source and target units are incompatible")
        expected_measure = _dimension_times_time(target_dimension)
        measure_numerator, measure_time_unit = _extract_integrated_mean_contract(
            unit_contract.measure_units
        )
        if (
            _normalize_unit_text(measure_numerator)
            != _normalize_unit_text(unit_contract.target_units)
            or measure_time_unit != integration_label
        ):
            raise ValidationError(
                "interval_mean measure units do not match target and integration time unit"
            )
        _convert_value_tensor(
            torch.ones((), dtype=reference.dtype, device=reference.device),
            unit_contract.source_units,
            unit_contract.target_units,
        )
    elif source_semantics is TemporalSemantics.RATE:
        if target_semantics not in {
            TemporalSemantics.RATE,
            TemporalSemantics.INTERVAL_ACCUMULATION,
        }:
            raise ValidationError("rate output must be rate or interval_accumulation")
        expected_measure = _dimension_times_time(source_dimension)
        expected_target = (
            source_dimension
            if target_semantics is TemporalSemantics.RATE
            else measure_dimension
        )
        if not target_dimension.is_compatible(expected_target):
            raise ValidationError("rate target units do not match output semantics")
        source_numerator, source_time_unit = _extract_rate_unit_contract(
            unit_contract.source_units
        )
        if source_time_unit != integration_label:
            raise ValidationError(
                "source rate time unit does not match integration_time_unit"
            )
        if _normalize_unit_text(source_numerator) != _normalize_unit_text(
            unit_contract.measure_units
        ):
            raise ValidationError(
                "source rate numerator units must exactly match measure_units"
            )
        if target_semantics is TemporalSemantics.RATE:
            target_numerator, target_time_unit = _extract_rate_unit_contract(
                unit_contract.target_units
            )
            if (
                target_time_unit != integration_label
                or _normalize_unit_text(target_numerator)
                != _normalize_unit_text(unit_contract.measure_units)
            ):
                raise ValidationError(
                    "target rate unit must use the declared measure and integration units"
                )
        else:
            _convert_value_tensor(
                torch.ones((), dtype=reference.dtype, device=reference.device),
                unit_contract.measure_units,
                unit_contract.target_units,
            )
    else:
        raise ValidationError(
            f"{source_semantics.value} requires a dedicated declared mapping"
        )
    if not measure_dimension.is_compatible(expected_measure):
        raise ValidationError(
            "measure_units are dimensionally incompatible with source semantics"
        )
    scale = _convert_time_tensor(
        torch.ones((), dtype=reference.dtype, device=reference.device),
        plan.coordinate_unit,
        integration_label,
    )
    if scale.numel() != 1 or not torch.all(torch.isfinite(scale)) or torch.any(scale <= 0):
        raise ValidationError("canonical integration time conversion is invalid")
    return scale.reshape(())


def apply_temporal_kernel(
    plan: TemporalOverlapPlan,
    data: Tensor,
    *,
    semantics: TemporalSemantics | str,
    unit_contract: TemporalUnitContract,
    output_semantics: TemporalSemantics | str | None = None,
    valid_mask: Tensor | None = None,
    time_dim: int = 0,
) -> TemporalKernelResult:
    """Apply one canonical first-order temporal interval mapping."""
    semantic = _coerce_temporal_semantics(semantics, "temporal semantics")
    output = (
        semantic
        if output_semantics is None
        else _coerce_temporal_semantics(output_semantics, "output temporal semantics")
    )
    scale = _validate_temporal_unit_contract(
        plan,
        unit_contract,
        semantic,
        output,
        data,
    )
    source_size, target_size = plan.overlap.shape
    data_flat, non_time_shape, normalized_time_dim = _prepare_temporal_data(
        data,
        expected_time_size=source_size,
        time_dim=time_dim,
        name="temporal data",
    )
    if data.device != plan.overlap.device:
        raise ValidationError("temporal data and overlap plan must use the same device")
    mask_flat = _prepare_valid_mask(
        data,
        data_flat,
        valid_mask=valid_mask,
        time_dim=normalized_time_dim,
    )
    clean_data = torch.where(mask_flat, data_flat, torch.zeros_like(data_flat))
    if semantic is TemporalSemantics.INTERVAL_MEAN:
        mapping_data = _convert_value_tensor(
            clean_data,
            unit_contract.source_units,
            unit_contract.target_units,
        )
    elif semantic is TemporalSemantics.INTERVAL_ACCUMULATION:
        mapping_data = _convert_value_tensor(
            clean_data,
            unit_contract.source_units,
            unit_contract.measure_units,
        )
    else:
        mapping_data = clean_data
    overlap = plan.overlap.to(dtype=data.dtype)
    valid_overlap = overlap[:, :, None] * mask_flat[:, None, :].to(data.dtype)
    observed_duration_raw = torch.sum(valid_overlap, dim=0)
    target_durations = plan.target_durations.to(dtype=data.dtype)[:, None]
    valid_coverage_flat = observed_duration_raw / target_durations
    tolerance = torch.finfo(data.dtype).eps * 64.0
    complete_flat = (
        torch.abs(valid_coverage_flat - 1.0) <= tolerance
    ) & (
        torch.abs(
            plan.geometric_target_coverage.to(dtype=data.dtype)[:, None] - 1.0
        )
        <= tolerance
    )

    # References:
    #   - CF Conventions 1.13, Sections 4.4, 7.1, and 7.3.
    #   - Jones (1999), first-order conservative remapping.
    # Mathematical expression:
    #   mean_j = sum_i(v_ik O_ij z_ik) / sum_i(v_ik O_ij).
    #   amount_j = sum_i(v_ik O_ij r_ik).
    #   total_j = sum_i(v_ik O_ij / delta_i A_ik).
    # Parameter meanings:
    #   v is validity, O is overlap, delta is source duration, k is support.
    # Units:
    #   Means retain input units; rates times scaled duration yield amounts.
    # Domain/constraints:
    #   First-order piecewise-constant reconstruction on explicit bounds.
    # Sign convention:
    #   Signed data are preserved; only duration weights are nonnegative.
    if semantic is TemporalSemantics.INTERVAL_MEAN:
        scaled_overlap = valid_overlap * scale
        observed_duration_flat = observed_duration_raw * scale
        observed_measure_flat = torch.einsum(
            "ijk,ik->jk",
            scaled_overlap,
            mapping_data,
        )
        has_support = observed_duration_flat > 0
        safe_duration = torch.where(
            has_support,
            observed_duration_flat,
            torch.ones_like(observed_duration_flat),
        )
        values_flat = torch.where(
            has_support,
            observed_measure_flat / safe_duration,
            torch.full_like(observed_measure_flat, float("nan")),
        )
    elif semantic is TemporalSemantics.RATE:
        scaled_overlap = valid_overlap * scale
        observed_duration_flat = observed_duration_raw * scale
        observed_measure_flat = torch.einsum(
            "ijk,ik->jk",
            scaled_overlap,
            mapping_data,
        )
        if output is TemporalSemantics.RATE:
            has_support = observed_duration_flat > 0
            safe_duration = torch.where(
                has_support,
                observed_duration_flat,
                torch.ones_like(observed_duration_flat),
            )
            values_flat = torch.where(
                has_support,
                observed_measure_flat / safe_duration,
                torch.full_like(observed_measure_flat, float("nan")),
            )
        else:
            values_flat = _convert_value_tensor(
                observed_measure_flat,
                unit_contract.measure_units,
                unit_contract.target_units,
            )
    elif semantic is TemporalSemantics.INTERVAL_ACCUMULATION:
        source_fractions = plan.source_fractions.to(dtype=data.dtype)
        valid_fractions = (
            source_fractions[:, :, None]
            * mask_flat[:, None, :].to(data.dtype)
        )
        observed_measure_flat = torch.einsum(
            "ijk,ik->jk",
            valid_fractions,
            mapping_data,
        )
        values_flat = _convert_value_tensor(
            observed_measure_flat,
            unit_contract.measure_units,
            unit_contract.target_units,
        )
        observed_duration_flat = observed_duration_raw
    else:
        raise ValidationError(
            f"{semantic.value} requires a dedicated declared mapping, not overlap aggregation"
        )

    geometric_coverage_flat = plan.geometric_target_coverage.to(
        dtype=data.dtype
    )[:, None].expand_as(valid_coverage_flat)
    geometric_gap_flat = 1.0 - geometric_coverage_flat
    missing_data_gap_flat = geometric_coverage_flat - valid_coverage_flat
    total_gap_flat = 1.0 - valid_coverage_flat

    # References:
    #   - CF Conventions 1.13, Sections 7.1 and 7.3.
    # Mathematical expression:
    #   g_geometry = 1-C_geometry; g_missing = C_geometry-C_valid.
    # Parameter meanings:
    #   C_geometry is bound coverage and C_valid includes the per-support mask.
    # Units:
    #   Gap fractions are dimensionless; durations use the coordinate unit.
    # Domain/constraints:
    #   Gap diagnostics never alter mapped values and declare no imputation.
    # Sign convention:
    #   Positive gaps denote unavailable target support.
    gap_diagnostic = TemporalGapDiagnostic(
        geometric_gap_fraction=_restore_time_axis(
            geometric_gap_flat,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        missing_data_gap_fraction=_restore_time_axis(
            missing_data_gap_flat,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        total_gap_fraction=_restore_time_axis(
            total_gap_flat,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        observed_duration=_restore_time_axis(
            observed_duration_raw,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        target_duration=_restore_time_axis(
            target_durations.expand_as(valid_coverage_flat),
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        units=plan.coordinate_unit,
        kind="gap",
        imputation="none",
    )

    return TemporalKernelResult(
        values=_restore_time_axis(
            values_flat,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        observed_measure=_restore_time_axis(
            observed_measure_flat,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        observed_duration=_restore_time_axis(
            observed_duration_flat,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        geometric_coverage=plan.geometric_target_coverage,
        valid_coverage=_restore_time_axis(
            valid_coverage_flat,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        complete_coverage=_restore_time_axis(
            complete_flat,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        semantics=semantic,
        output_semantics=output,
        unit_contract=unit_contract,
        gap_diagnostic=gap_diagnostic,
    )


def compute_temporal_mapping_residual(
    plan: TemporalOverlapPlan,
    source_data: Tensor,
    target_data: Tensor,
    *,
    semantics: TemporalSemantics | str,
    unit_contract: TemporalUnitContract,
    law_id: str,
    output_semantics: TemporalSemantics | str | None = None,
    source_valid_mask: Tensor | None = None,
    accepted_target_mask: Tensor | None = None,
    excluded_output_mask: Tensor | None = None,
    time_dim: int = 0,
    q_ref: Any | None = None,
    q_ref_provenance: str | None = None,
    imputation: str = "none",
) -> TemporalResidualResult:
    """Compare source and target values on the same accepted temporal support.

    ``accepted_target_mask`` (shape of ``target_data``) selects the target
    cells that enter the residual. Every non-time output element must keep at
    least one accepted target with valid source overlap; a vacuous element
    raises ``ValidationError`` unless the caller lists it explicitly in
    ``excluded_output_mask`` (boolean, non-time shape of ``target_data``).
    Only elements without any accepted target may be excluded. Excluded
    elements are reported with ``checked=False`` and NaN ``residual`` and
    ``relative_residual`` entries; they are never reported as a zero residual.
    """
    semantic = _coerce_temporal_semantics(semantics, "temporal semantics")
    output = (
        semantic
        if output_semantics is None
        else _coerce_temporal_semantics(output_semantics, "output temporal semantics")
    )
    if not isinstance(law_id, str) or not law_id or law_id.strip() != law_id:
        raise ValidationError("law_id must be one exact non-empty identity")
    if imputation != "none":
        raise ValidationError("mapping residual supports only explicit imputation='none'")
    scale = _validate_temporal_unit_contract(
        plan,
        unit_contract,
        semantic,
        output,
        source_data,
    )
    source_size, target_size = plan.overlap.shape
    source_flat, non_time_shape, normalized_time_dim = _prepare_temporal_data(
        source_data,
        expected_time_size=source_size,
        time_dim=time_dim,
        name="source_data",
    )
    target_flat, target_non_time_shape, target_time_dim = _prepare_temporal_data(
        target_data,
        expected_time_size=target_size,
        time_dim=time_dim,
        name="target_data",
    )
    if target_non_time_shape != non_time_shape or target_time_dim != normalized_time_dim:
        raise ValidationError("source and target non-time shapes must match")
    if source_data.dtype != target_data.dtype:
        raise ValidationError("source and target residual data must use the same dtype")
    if source_data.device != plan.overlap.device or target_data.device != plan.overlap.device:
        raise ValidationError("residual data and overlap plan must use the same device")
    source_mask = _prepare_valid_mask(
        source_data,
        source_flat,
        valid_mask=source_valid_mask,
        time_dim=normalized_time_dim,
    )
    if accepted_target_mask is None:
        accepted_flat = torch.ones_like(target_flat, dtype=torch.bool)
    else:
        if (
            not isinstance(accepted_target_mask, Tensor)
            or accepted_target_mask.dtype is not torch.bool
            or accepted_target_mask.shape != target_data.shape
            or accepted_target_mask.device != target_data.device
        ):
            raise ValidationError(
                "accepted_target_mask must be boolean and match target_data"
            )
        accepted_flat = accepted_target_mask.movedim(
            normalized_time_dim,
            0,
        ).reshape_as(target_flat)
    if torch.any(accepted_flat & ~torch.isfinite(target_flat)):
        raise ValidationError("accepted target_data values must be finite")
    if excluded_output_mask is None:
        excluded_flat = torch.zeros(
            accepted_flat.shape[1],
            dtype=torch.bool,
            device=accepted_flat.device,
        )
    else:
        if (
            not isinstance(excluded_output_mask, Tensor)
            or excluded_output_mask.dtype is not torch.bool
            or tuple(excluded_output_mask.shape) != non_time_shape
            or excluded_output_mask.device != target_data.device
        ):
            raise ValidationError(
                "excluded_output_mask must be boolean with the non-time shape "
                "and device of target_data"
            )
        excluded_flat = excluded_output_mask.reshape(-1)
        if torch.any(accepted_flat & excluded_flat[None, :]):
            raise ValidationError(
                "excluded_output_mask may only exclude output elements without "
                "accepted targets"
            )
    clean_target = torch.where(
        accepted_flat,
        target_flat,
        torch.zeros_like(target_flat),
    )

    clean_source = torch.where(source_mask, source_flat, torch.zeros_like(source_flat))
    if semantic is TemporalSemantics.INTERVAL_MEAN:
        mapping_source = _convert_value_tensor(
            clean_source,
            unit_contract.source_units,
            unit_contract.target_units,
        )
    elif semantic is TemporalSemantics.INTERVAL_ACCUMULATION:
        mapping_source = _convert_value_tensor(
            clean_source,
            unit_contract.source_units,
            unit_contract.measure_units,
        )
    else:
        mapping_source = clean_source
    if output is TemporalSemantics.INTERVAL_ACCUMULATION:
        mapping_target = _convert_value_tensor(
            clean_target,
            unit_contract.target_units,
            unit_contract.measure_units,
        )
    else:
        mapping_target = clean_target
    overlap = plan.overlap.to(dtype=source_data.dtype)
    available_overlap = (
        overlap[:, :, None]
        * source_mask[:, None, :].to(source_data.dtype)
    )
    valid_coverage_flat = (
        torch.sum(available_overlap, dim=0)
        / plan.target_durations.to(dtype=source_data.dtype)[:, None]
    )
    valid_overlap = (
        available_overlap
        * accepted_flat[None, :, :].to(source_data.dtype)
    )
    accepted_duration = torch.sum(valid_overlap, dim=0)
    if torch.any(accepted_flat & (accepted_duration <= 0)):
        raise ValidationError(
            "accepted support contains a target without valid source overlap"
        )
    checked = torch.sum(accepted_duration, dim=0) > 0
    if torch.any(~checked & ~excluded_flat):
        raise ValidationError(
            "accepted support is empty for at least one output element that "
            "is not listed in excluded_output_mask"
        )

    tolerance = torch.finfo(source_data.dtype).eps * 64.0
    if output is TemporalSemantics.INTERVAL_ACCUMULATION and torch.any(
        accepted_flat & (torch.abs(valid_coverage_flat - 1.0) > tolerance)
    ):
        raise ValidationError(
            "interval_accumulation residual requires complete accepted support"
        )

    # References:
    #   - Roache (2002), J. Fluids Eng. 124:4-10.
    # Mathematical expression:
    #   R_map = sum_j q_target_j - sum_ij q_source_ij.
    #   activity = sum_ij abs(q_source_ij).
    # Parameter meanings:
    #   q values are contributions on the identical accepted support.
    # Units:
    #   R_map, activity, and q_ref share the mapped measure unit.
    # Domain/constraints:
    #   Masks, accepted targets, semantics, units, and support are explicit.
    #   An output element with no accepted target raises unless the caller
    #   lists it in excluded_output_mask; an excluded element reports
    #   R_map = NaN and checked = False (never a vacuous zero).
    # Sign convention:
    #   Positive residual means target measure exceeds source measure.
    if semantic in {TemporalSemantics.INTERVAL_MEAN, TemporalSemantics.RATE}:
        contributions = valid_overlap * mapping_source[:, None, :] * scale
    elif semantic is TemporalSemantics.INTERVAL_ACCUMULATION:
        fractions = (
            plan.source_fractions.to(dtype=source_data.dtype)[:, :, None]
            * source_mask[:, None, :].to(source_data.dtype)
            * accepted_flat[None, :, :].to(source_data.dtype)
        )
        contributions = fractions * mapping_source[:, None, :]
    else:
        raise ValidationError(
            f"mapping residual is not defined for {semantic.value}"
        )

    if output in {TemporalSemantics.INTERVAL_MEAN, TemporalSemantics.RATE}:
        target_measure_weights = accepted_duration * scale
    elif output is TemporalSemantics.INTERVAL_ACCUMULATION:
        target_measure_weights = accepted_flat.to(source_data.dtype)
    else:
        raise ValidationError(f"mapping residual is not defined for {output.value}")

    source_measure = torch.sum(contributions, dim=(0, 1))
    target_measure = torch.sum(mapping_target * target_measure_weights, dim=0)
    residual = target_measure - source_measure
    source_activity = torch.sum(torch.abs(contributions), dim=(0, 1))

    relative_residual: Tensor | None = None
    if q_ref is not None:
        if (
            not isinstance(q_ref_provenance, str)
            or not q_ref_provenance
            or q_ref_provenance.strip() != q_ref_provenance
        ):
            raise ValidationError(
                "q_ref requires one exact ParameterManager provenance identity"
            )
        reference = torch.as_tensor(
            q_ref,
            dtype=source_data.dtype,
            device=source_data.device,
        )
        if not torch.all(torch.isfinite(reference)) or torch.any(reference <= 0):
            raise ValidationError("q_ref must be finite and strictly positive")
        try:
            reference = torch.broadcast_to(reference, non_time_shape).reshape(-1)
        except RuntimeError as exc:
            raise ValidationError("q_ref is not broadcastable to residual support") from exc
        denominator = torch.maximum(source_activity, reference)
        relative_residual = torch.abs(residual) / denominator
    elif q_ref_provenance is not None:
        raise ValidationError("q_ref_provenance must be absent when q_ref is absent")

    # Explicitly excluded (fully masked) output elements have an empty accepted
    # support: their residual is not evaluated and is reported as NaN with
    # checked=False instead of a vacuous zero.
    unchecked_value = torch.full_like(residual, float("nan"))
    residual = torch.where(checked, residual, unchecked_value)
    if relative_residual is not None:
        relative_residual = torch.where(checked, relative_residual, unchecked_value)

    return TemporalResidualResult(
        residual=residual.reshape(non_time_shape),
        source_activity=source_activity.reshape(non_time_shape),
        relative_residual=(
            None
            if relative_residual is None
            else relative_residual.reshape(non_time_shape)
        ),
        checked=checked.reshape(non_time_shape),
        law_id=law_id,
        kind="mapping",
        units=unit_contract.measure_units,
        support="accepted_target_intervals_x_non_time_support",
        reduction_axes=(0, 1),
        sign_convention="positive_target_minus_source",
        geometric_coverage=plan.geometric_target_coverage,
        valid_coverage=_restore_time_axis(
            valid_coverage_flat,
            target_size=target_size,
            non_time_shape=non_time_shape,
            time_dim=normalized_time_dim,
        ),
        accepted_target_mask=accepted_flat.reshape(
            (target_size, *non_time_shape)
        ).movedim(0, normalized_time_dim),
        imputation=imputation,
        q_ref_provenance=q_ref_provenance,
        source_semantics=semantic,
        target_semantics=output,
    )


def map_state_boundaries(
    source_time: TimeCoordinate,
    target_time: TimeCoordinate,
    data: Tensor,
    *,
    time_dim: int = 0,
) -> Tensor:
    """Map exact state boundary values without inventing interpolation."""
    if source_time.calendar != target_time.calendar:
        raise ValidationError("state boundary calendars must match")
    if source_time.values.device != target_time.values.device:
        raise ValidationError("state boundary coordinates must use the same device")
    if source_time.values.dtype != target_time.values.dtype:
        raise ValidationError("state boundary coordinates must use the same dtype")
    if not torch.all(torch.isfinite(source_time.values)) or not torch.all(
        torch.isfinite(target_time.values)
    ):
        raise ValidationError("state boundary coordinates must be finite")
    if source_time.values.numel() > 1 and torch.any(
        source_time.values[1:] <= source_time.values[:-1]
    ):
        raise ValidationError("source state boundaries must be strictly increasing")
    target_values, _ = _align_time_values_to_source_units(
        source_time.units,
        target_time.units,
        target_time.values,
    )
    if target_values.numel() > 1 and torch.any(
        target_values[1:] <= target_values[:-1]
    ):
        raise ValidationError("target state boundaries must be strictly increasing")

    data_flat, non_time_shape, normalized_time_dim = _prepare_temporal_data(
        data,
        expected_time_size=source_time.values.numel(),
        time_dim=time_dim,
        name="state boundary data",
    )
    if torch.any(~torch.isfinite(data_flat)):
        raise ValidationError("state boundary data must be finite")
    if data.device != source_time.values.device:
        raise ValidationError("state data and coordinates must use the same device")

    # References:
    #   - CF Conventions 1.13, time coordinates and cell boundaries.
    # Mathematical expression:
    #   X_target(t_j) = X_source(t_i) only when t_j equals t_i.
    # Parameter meanings:
    #   X is a boundary state; t_i and t_j are explicit boundary instants.
    # Units:
    #   Output state units equal input state units; coordinate units match.
    # Domain/constraints:
    #   Exact boundary selection only; interpolation remains explicitly unbound.
    # Sign convention:
    #   State signs are preserved without summation or correction.
    source_values = source_time.values
    matches = target_values[:, None] == source_values[None, :]
    match_count = torch.sum(matches, dim=1)
    if torch.any(match_count != 1):
        raise ValidationError(
            "state boundary interpolation is unbound; each target must match one source"
        )
    indices = torch.argmax(matches.to(torch.int64), dim=1)
    selected = data_flat.index_select(0, indices)
    return _restore_time_axis(
        selected,
        target_size=target_values.numel(),
        non_time_shape=non_time_shape,
        time_dim=normalized_time_dim,
    )


class PeriodIndexer:
    """
    Period boundary and indexing manager for temporal operations.
    
    Manages period definitions, boundary calculations, and index mapping
    for different temporal frequencies and calendar systems.
    """

    _MONTH_ABBREVIATIONS = (
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    )
    _SEASON_NAMES = ("DJF", "MAM", "JJA", "SON")
    _SEASON_START_MONTHS = (12, 3, 6, 9)
    
    def __init__(self, device: Optional[torch.device] = None, parameter_manager: Optional[Any] = None):
        """
        Initialize period indexer.
        
        Parameters:
            device : Optional[torch.device]
                Computation device
        """
        self.device = device or (
            device_policy.resolve_device(default="auto")
        )
        self.parameter_manager = (
            parameter_manager
            if parameter_manager is not None
            else get_global_manager()
        )
        if self.parameter_manager is None:
            raise RuntimeError(
                "ParameterManager is required for PeriodIndexer; ensure it is initialized or pass explicitly."
            )
        self.cf_processor = CFTimeProcessor(device=device, parameter_manager=self.parameter_manager)

    @classmethod
    def _parse_period_frequency(
        cls,
        frequency: str,
    ) -> tuple[str, int | None, int | None]:
        """Resolve one controlled frequency into code, month step, and anchor."""
        if not isinstance(frequency, str) or not frequency.strip():
            raise ValidationError("frequency must be a non-empty string")
        normalized = frequency.strip().upper()
        if normalized in {"H", "D", "W", "M", "Y"}:
            step_months = {"M": 1, "Y": 12}.get(normalized)
            anchor_month = 1 if normalized == "Y" else None
            return normalized, step_months, anchor_month
        if normalized in {"Q", "QUARTER", "QUARTERLY", "S", "SEASON", "SEASONAL"}:
            raise ValidationError(
                f"Frequency {frequency!r} requires an explicit quarter or season anchor"
            )
        if normalized.startswith("QS-"):
            anchor_name = normalized[3:]
            if anchor_name not in cls._MONTH_ABBREVIATIONS:
                raise ValidationError(f"Unsupported quarter anchor: {frequency}")
            return "Q", 3, cls._MONTH_ABBREVIATIONS.index(anchor_name) + 1
        if normalized.startswith("SEASON-"):
            season_name = normalized[7:]
            if season_name not in cls._SEASON_NAMES:
                raise ValidationError(f"Unsupported season anchor: {frequency}")
            season_index = cls._SEASON_NAMES.index(season_name)
            return "S", 3, cls._SEASON_START_MONTHS[season_index]
        raise ValidationError(f"Unsupported frequency: {frequency}")

    def _parse_iso_timestamp(self, iso_time: str) -> Tuple[int, int, int, int, int, float]:
        """Parse ISO timestamp into calendar tuple (year, month, day, hour, minute, second)."""
        return _parse_iso_timestamp(iso_time)

    def _parse_units_reference(self, units: str) -> Tuple[str, Tuple[int, int, int, int, int, float]]:
        """Parse CF units and return (time_unit, reference_datetime_tuple)."""
        return _parse_cf_time_units(units)

    def _unit_days_scale(self, unit_name: str) -> float:
        """Return conversion factor from requested unit to days."""
        if unit_name.startswith("second"):
            return 1.0 / float(SECONDS_PER_DAY)
        if unit_name.startswith("minute"):
            return float(SECONDS_PER_MINUTE) / float(SECONDS_PER_DAY)
        if unit_name.startswith("hour"):
            return float(SECONDS_PER_HOUR) / float(SECONDS_PER_DAY)
        if unit_name.startswith("day"):
            return 1.0
        raise ValidationError(
            f"Units '{unit_name}' are not supported for exact calendar alignment; use seconds, minutes, hours, or days."
        )

    def _is_leap_year(self, year: int, calendar: str) -> bool:
        """Return leap-year flag under configured calendar system."""
        if calendar == "all_leap":
            return True
        if calendar == "noleap":
            return False
        if calendar == "360_day":
            return False
        if calendar == "julian" or (calendar == "gregorian" and year <= 1582):
            return year % 4 == 0
        # References: CF Conventions, Calendar section (Gregorian leap rule).
        # Expression: leap = (y mod 4 = 0) and ((y mod 100 != 0) or (y mod 400 = 0)).
        # Parameters: year y and the normalized calendar identifier.
        return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))

    def _days_in_month(self, year: int, month: int, calendar: str) -> int:
        """Return days in month for supported calendar systems."""
        if month < 1 or month > 12:
            raise ValidationError(f"Invalid month: {month}")
        if calendar == "360_day":
            return 30
        if month == 2:
            return 29 if self._is_leap_year(year, calendar) else 28
        if month in (1, 3, 5, 7, 8, 10, 12):
            return 31
        return 30

    def _add_calendar_months(
        self,
        dt_tuple: Tuple[int, int, int, int, int, float],
        months: int,
        calendar: str,
    ) -> Tuple[int, int, int, int, int, float]:
        """Advance a calendar datetime by an exact positive month count."""
        if type(months) is not int or months <= 0:
            raise ValidationError("months must be one positive integer")
        year, month, day, hour, minute, second = dt_tuple

        # References:
        #   - CF Conventions 1.13, Section 4.4.1 calendar definitions.
        # Mathematical expression:
        #   m_abs' = 12(y-1) + (m-1) + n; (y',m') = divmod(m_abs',12).
        # Parameter meanings:
        #   y/m are the source year/month and n is the exact month increment.
        # Units:
        #   n is a calendar-month count; the returned tuple is a civil datetime.
        # Domain/constraints:
        #   n>0 and day is clipped only to the destination calendar month.
        # Sign convention:
        #   Positive n advances time while preserving time-of-day.
        absolute_month = (year - 1) * 12 + (month - 1) + months
        target_year, target_month_zero = divmod(absolute_month, 12)
        target_year += 1
        target_month = target_month_zero + 1
        target_day = min(
            day,
            self._days_in_month(target_year, target_month, calendar),
        )
        return target_year, target_month, target_day, hour, minute, second

    def _datetime_to_day_index(
        self,
        dt_tuple: Tuple[int, int, int, int, int, float],
        calendar: str,
    ) -> float:
        """
        Convert calendar datetime tuple to absolute day index (day 0 = 0001-01-01 00:00:00).
        """
        year, month, day, hour, minute, second = dt_tuple
        if year < 1:
            raise ValidationError(f"Year must be >= 1, got {year}")
        if (
            hour < 0
            or hour > 23
            or minute < 0
            or minute > (MINUTES_PER_HOUR - 1)
            or second < 0.0
            or second >= float(SECONDS_PER_MINUTE)
        ):
            raise ValidationError(f"Invalid time components in {dt_tuple}")
        if calendar == "gregorian" and (year, month, day) >= (1582, 10, 5) and (
            year,
            month,
            day,
        ) <= (1582, 10, 14):
            raise ValidationError(
                f"Date {year:04d}-{month:02d}-{day:02d} is in the Gregorian calendar reform gap"
            )

        max_day = self._days_in_month(year, month, calendar)
        if day < 1 or day > max_day:
            raise ValidationError(f"Invalid day {day} for {year:04d}-{month:02d} ({calendar})")

        fractional_day = (
            hour * float(SECONDS_PER_HOUR)
            + minute * float(SECONDS_PER_MINUTE)
            + second
        ) / float(SECONDS_PER_DAY)

        if calendar == "360_day":
            return float((year - 1) * 360 + (month - 1) * 30 + (day - 1)) + fractional_day
        if calendar == "noleap":
            month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            days_before_year = (year - 1) * 365
            days_before_month = sum(month_lengths[: month - 1])
            return float(days_before_year + days_before_month + (day - 1)) + fractional_day
        if calendar == "all_leap":
            month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            days_before_year = (year - 1) * 366
            days_before_month = sum(month_lengths[: month - 1])
            return float(days_before_year + days_before_month + (day - 1)) + fractional_day

        # References: CF Conventions, Calendar section (Julian/Gregorian rules and 1582 cutover).
        # Expression: D = 365(y-1) + L(y-1) + sum(m_i) + d - 1, with calendar-specific L.
        # Parameters: civil date (year, month, day), fractional day, and normalized calendar.
        if calendar == "julian":
            raw_index = self._raw_calendar_day_index(
                year,
                month,
                day,
                "julian",
            )
            return float(raw_index) + fractional_day
        if calendar == "proleptic_gregorian":
            raw_index = self._raw_calendar_day_index(
                year,
                month,
                day,
                "proleptic_gregorian",
            )
            return float(raw_index) + fractional_day
        if calendar == "gregorian":
            if (year, month, day) <= (1582, 10, 4):
                raw_index = self._raw_calendar_day_index(
                    year,
                    month,
                    day,
                    "julian",
                )
                return float(raw_index) + fractional_day

            last_julian_index = self._raw_calendar_day_index(
                1582,
                10,
                4,
                "julian",
            )
            first_gregorian_index = self._raw_calendar_day_index(
                1582,
                10,
                15,
                "proleptic_gregorian",
            )
            adjusted_index = (
                self._raw_calendar_day_index(
                    year,
                    month,
                    day,
                    "proleptic_gregorian",
                )
                - first_gregorian_index
                + last_julian_index
                + 1
            )
            return float(adjusted_index) + fractional_day
        raise CalendarError(f"Unsupported calendar for conversion: {calendar}")

    def _raw_calendar_day_index(
        self,
        year: int,
        month: int,
        day: int,
        calendar: str,
    ) -> int:
        """Return an unadjusted Julian or proleptic-Gregorian day index."""
        # References: CF Conventions, Calendar section.
        # Expression: D = 365(y-1) + L(y-1) + cumulative_month_days + d - 1.
        # Parameters: civil year, month, day, and the raw calendar leap rule.
        y = year - 1
        if calendar == "julian":
            leap_count = y // 4
        elif calendar == "proleptic_gregorian":
            leap_count = y // 4 - y // 100 + y // 400
        else:
            raise CalendarError(f"Unsupported raw calendar: {calendar}")
        month_lengths = [
            31,
            29 if self._is_leap_year(year, calendar) else 28,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ]
        return y * 365 + leap_count + sum(month_lengths[: month - 1]) + (day - 1)

    def _day_index_to_datetime(
        self,
        day_index: float,
        calendar: str,
    ) -> Tuple[int, int, int, int, int, float]:
        """Inverse of _datetime_to_day_index for supported calendars."""
        if day_index < 0.0:
            raise ValidationError(f"Day index must be non-negative, got {day_index}")

        day_whole = int(day_index)
        fractional = float(day_index - day_whole)
        if fractional < 0.0:
            fractional = 0.0
        if fractional >= 1.0:
            day_whole += int(fractional)
            fractional = fractional % 1.0

        decode_calendar = calendar
        decode_day_whole = day_whole
        if calendar == "gregorian":
            last_julian_index = self._raw_calendar_day_index(
                1582,
                10,
                4,
                "julian",
            )
            if day_whole <= last_julian_index:
                decode_calendar = "julian"
            else:
                first_gregorian_index = self._raw_calendar_day_index(
                    1582,
                    10,
                    15,
                    "proleptic_gregorian",
                )
                decode_calendar = "proleptic_gregorian"
                decode_day_whole = (
                    day_whole
                    - last_julian_index
                    - 1
                    + first_gregorian_index
                )

        if decode_calendar == "360_day":
            year = decode_day_whole // 360 + 1
            rem = decode_day_whole % 360
            month = rem // 30 + 1
            day = rem % 30 + 1
        elif decode_calendar in {
            "noleap",
            "all_leap",
            "julian",
            "proleptic_gregorian",
        }:
            # References: CF Conventions 1.8, Section 4.4.1 (noleap, all_leap,
            #   julian and proleptic_gregorian calendar definitions).
            # Mathematical expression: with D whole days since 0001-01-01,
            #   noleap: y = floor(D / 365) + 1, r = D mod 365;
            #   all_leap: y = floor(D / 366) + 1, r = D mod 366;
            #   julian: (c, s) = divmod(D, 1461), k = min(floor(s / 365), 3),
            #     y = 4 c + k + 1, r = s - 365 k;
            #   proleptic_gregorian: (a, s) = divmod(D, 146097),
            #     b = min(floor(s / 36524), 3), s = s - 36524 b,
            #     (c, s) = divmod(s, 1461), k = min(floor(s / 365), 3),
            #     y = 400 a + 100 b + 4 c + k + 1, r = s - 365 k.
            # Parameter meanings: D is the absolute day index, y the civil year,
            #   r the zero-based day within year y; 1461, 36524 and 146097 are
            #   the day counts of 4, 100 and 400 calendar years, in which the
            #   leap year is the last year of each block counted from year 1.
            # Units: D, r [day]; y and the block counters are dimensionless.
            # Domain/constraints: D >= 0 (checked above), hence y >= 1; the
            #   result equals the year-by-year enumeration of the same leap rule.
            # Sign convention: D increases forward in time; no quantity transfer.
            common_year_days = 365
            four_year_days = 4 * common_year_days + 1
            century_days = 25 * four_year_days - 1
            four_century_days = 4 * century_days + 1
            if decode_calendar == "noleap":
                year_offset, remaining = divmod(decode_day_whole, common_year_days)
            elif decode_calendar == "all_leap":
                year_offset, remaining = divmod(decode_day_whole, common_year_days + 1)
            elif decode_calendar == "julian":
                cycles, remaining = divmod(decode_day_whole, four_year_days)
                years = min(remaining // common_year_days, 3)
                remaining -= common_year_days * years
                year_offset = 4 * cycles + years
            else:
                cycles_400, remaining = divmod(decode_day_whole, four_century_days)
                centuries = min(remaining // century_days, 3)
                remaining -= century_days * centuries
                cycles_4, remaining = divmod(remaining, four_year_days)
                years = min(remaining // common_year_days, 3)
                remaining -= common_year_days * years
                year_offset = 400 * cycles_400 + 100 * centuries + 4 * cycles_4 + years
            year = year_offset + 1

            month = 1
            while True:
                dim = self._days_in_month(year, month, decode_calendar)
                if remaining >= dim:
                    remaining -= dim
                    month += 1
                else:
                    break
            day = remaining + 1
        else:
            raise CalendarError(
                f"Unsupported calendar for inverse conversion: {calendar}"
            )

        seconds_total = fractional * float(SECONDS_PER_DAY)
        hour = int(seconds_total // float(SECONDS_PER_HOUR))
        seconds_total -= hour * float(SECONDS_PER_HOUR)
        minute = int(seconds_total // float(SECONDS_PER_MINUTE))
        second = seconds_total - minute * float(SECONDS_PER_MINUTE)
        if second >= 59.999999:
            second = 0.0
            minute += 1
        if minute >= MINUTES_PER_HOUR:
            minute = 0
            hour += 1
        if hour >= 24:
            return self._day_index_to_datetime(float(day_whole + 1), calendar)

        return year, month, int(day), hour, minute, second

    def _datetime_to_units_value(
        self,
        dt_tuple: Tuple[int, int, int, int, int, float],
        units: str,
        calendar: str,
    ) -> float:
        """Convert datetime tuple to numeric value under CF units."""
        unit_name, ref_dt = self._parse_units_reference(units)
        day_scale = self._unit_days_scale(unit_name)
        dt_day_index = self._datetime_to_day_index(dt_tuple, calendar)
        ref_day_index = self._datetime_to_day_index(ref_dt, calendar)
        delta_days = dt_day_index - ref_day_index
        return delta_days / day_scale

    def _units_value_to_datetime(
        self,
        value: float,
        units: str,
        calendar: str,
    ) -> Tuple[int, int, int, int, int, float]:
        """Convert numeric value under CF units to datetime tuple."""
        unit_name, ref_dt = self._parse_units_reference(units)
        day_scale = self._unit_days_scale(unit_name)
        ref_day_index = self._datetime_to_day_index(ref_dt, calendar)
        day_index = ref_day_index + value * day_scale
        return self._day_index_to_datetime(day_index, calendar)

    def elapsed_month_index(
        self,
        elapsed_seconds: Union[float, Tensor],
        calendar: str,
        time_origin: str | None = None,
    ) -> int:
        """Return the absolute zero-based month containing origin plus elapsed time.

        Mathematical Expression:
            m(t) = 12 * (y(t) - 1) + (q(t) - 1)

        where:
            t = finite elapsed simulation time [s], t >= 0
            y(t) = CF calendar year decoded from t
            q(t) = CF calendar month in {1, ..., 12}
            m(t) = monotone zero-based month index

        References:
            CF Conventions 1.8, Section 4.4, Time Coordinate.

        Parameters:
            elapsed_seconds : float or Tensor
                Scalar elapsed simulation time in seconds from time_origin.
            calendar : str
                CF calendar identifier resolved by ``CFTimeProcessor``.
            time_origin : str or None
                Calendar-valid reference timestamp in the existing bounded
                CF-unit format. None retains the reference 0001-01-01.
        """
        elapsed = torch.as_tensor(
            elapsed_seconds,
            dtype=torch.float64,
            device=self.device,
        )
        if elapsed.numel() != 1:
            raise ValidationError("elapsed_seconds must be a scalar")
        if not bool(torch.isfinite(elapsed).item()) or bool(torch.lt(elapsed, 0.0).item()):
            raise ValidationError("elapsed_seconds must be finite and non-negative")

        if time_origin is not None and (
            not isinstance(time_origin, str) or not time_origin.strip()
        ):
            raise ValidationError("time_origin must be a non-empty ISO timestamp")

        # References: CF Conventions 1.8, Section 4.4; shared mathematical plan
        #   sections 4, 12 and 13 (scientific time and recovery obligations).
        # Mathematical expression: (y,q,...) = decode_C(t; seconds since o);
        #   m_C(t;o) = 12*(y-1) + (q-1).
        # Parameter meanings: t is elapsed time, o the supplied reference instant
        #   or 0001-01-01 when omitted, C the normalized source calendar;
        #   y is the decoded civil year, q its month, m the absolute month ordinal.
        # Units: t [s]; o is a calendar timestamp; y,q,m are dimensionless integers.
        # Domain/constraints: t is finite, scalar and nonnegative; o is a valid
        #   year>=1 timestamp in the existing no-timezone parser; reuse one decoder.
        # Sign convention: time advances forward, m is nondecreasing; this lookup
        #   neither transfers a physical quantity nor advances scientific state.
        normalized_calendar = self.cf_processor._normalize_calendar(calendar)
        year, month, _day, _hour, _minute, _second = self._units_value_to_datetime(
            float(elapsed.item()),
            f"seconds since {time_origin.strip() if time_origin is not None else '0001-01-01'}",
            normalized_calendar,
        )
        return 12 * (year - 1) + (month - 1)

    def _normalize_period_start(
        self,
        dt_tuple: Tuple[int, int, int, int, int, float],
        frequency: str,
        calendar: str,
    ) -> Tuple[int, int, int, int, int, float]:
        """Normalize datetime to period start boundary for target frequency."""
        code, _step_months, anchor_month = self._parse_period_frequency(frequency)
        year, month, day, hour, minute, _second = dt_tuple
        if code == "H":
            return year, month, day, hour, 0, 0.0
        if code in {"D", "W"}:
            return year, month, day, 0, 0, 0.0
        if code == "M":
            return year, month, 1, 0, 0, 0.0
        if code == "Y":
            return year, 1, 1, 0, 0, 0.0
        if code not in {"Q", "S"} or anchor_month is None:
            raise ValidationError(f"Unsupported anchored frequency: {frequency}")

        # References:
        #   - CF Conventions 1.13 calendar definitions.
        # Mathematical expression:
        #   m_start = a + 3 floor((m_abs-a)/3), for anchor month a.
        # Parameter meanings:
        #   m_abs is the absolute civil month and a is the explicit start anchor.
        # Units:
        #   Month indices are dimensionless civil-calendar counts.
        # Domain/constraints:
        #   Quarter/season spans are three calendar months and require an anchor.
        # Sign convention:
        #   The normalized boundary is the latest anchor not after the input.
        absolute_month = (year - 1) * 12 + (month - 1)
        anchor_zero = anchor_month - 1
        period_number = (absolute_month - anchor_zero) // 3
        start_absolute_month = anchor_zero + 3 * period_number
        start_year, start_month_zero = divmod(start_absolute_month, 12)
        normalized_year = start_year + 1
        if normalized_year < 1:
            return 1, 1, 1, 0, 0, 0.0
        return normalized_year, start_month_zero + 1, 1, 0, 0, 0.0

    def _increment_datetime(
        self,
        dt_tuple: Tuple[int, int, int, int, int, float],
        frequency: str,
        calendar: str,
    ) -> Tuple[int, int, int, int, int, float]:
        """Increment datetime by one period under target frequency."""
        code, step_months, anchor_month = self._parse_period_frequency(frequency)
        if code == "H":
            return self._day_index_to_datetime(
                self._datetime_to_day_index(dt_tuple, calendar) + (1.0 / 24.0),
                calendar,
            )
        if code == "D":
            return self._day_index_to_datetime(
                self._datetime_to_day_index(dt_tuple, calendar) + 1.0,
                calendar,
            )
        if code == "W":
            return self._day_index_to_datetime(
                self._datetime_to_day_index(dt_tuple, calendar) + 7.0,
                calendar,
            )
        # References:
        #   - CF Conventions 1.13 calendar definitions.
        # Mathematical expression:
        #   m_first = 1 + ((a-1) mod 3) for quarter anchor month a.
        # Parameter meanings:
        #   a is the declared anchor and m_first is its first year-one phase.
        # Units:
        #   a and m_first are civil month numbers in [1,12].
        # Domain/constraints:
        #   The synthetic epoch boundary replaces only an unrepresentable year-zero start.
        # Sign convention:
        #   The next boundary is the earliest congruent month after the epoch.
        if code in {"Q", "S"} and anchor_month is not None:
            first_representable_month = 1 + ((anchor_month - 1) % 3)
            if (
                first_representable_month > 1
                and dt_tuple == (1, 1, 1, 0, 0, 0.0)
            ):
                return 1, first_representable_month, 1, 0, 0, 0.0
        if step_months is not None:
            return self._add_calendar_months(dt_tuple, step_months, calendar)
        raise ValidationError(f"Unsupported frequency increment: {frequency}")
    
    def create_period_index(
        self,
        start_time: str,
        end_time: str,
        frequency: str,
        calendar: str = 'gregorian',
        units: str = "days since 1850-01-01"
    ) -> TimeCoordinate:
        """
        Create regular period index for given frequency and calendar.
        
        Parameters:
            start_time : str
                Start time in ISO format (e.g., "2000-01-01")
            end_time : str
                End time in ISO format
            frequency : str
                Frequency code ('H', 'D', 'M', 'Y')
            calendar : str
                Calendar type
            units : str
                CF units string
                
        Returns:
            TimeCoordinate: Period index with bounds
        """
        frequency_code, _step_months, _anchor_month = self._parse_period_frequency(
            frequency
        )
        calendar_normalized = self.cf_processor._normalize_calendar(calendar)
        start_dt = self._parse_iso_timestamp(start_time)
        end_dt = self._parse_iso_timestamp(end_time)
        start_day = self._datetime_to_day_index(start_dt, calendar_normalized)
        end_day = self._datetime_to_day_index(end_dt, calendar_normalized)
        if end_day <= start_day:
            raise ValidationError("end_time must be greater than start_time")

        # References:
        #   - CF Conventions 1.13, time coordinates and cell bounds.
        # Mathematical expression:
        #   I_k = [max(t_k,t_0), min(t_{k+1},t_1)).
        # Parameter meanings:
        #   t_0/t_1 bound the domain and t_k are canonical period anchors.
        # Units:
        #   Bounds use the requested CF time-coordinate units.
        # Domain/constraints:
        #   Half-open periods are increasing; partial edge periods are clipped.
        # Sign convention:
        #   Bounds advance from left to right without overlap.
        bounds_list: List[Tuple[float, float]] = []
        current_dt = (
            self._normalize_period_start(
                start_dt,
                frequency,
                calendar_normalized,
            )
            if frequency_code in {"Q", "S"}
            else start_dt
        )
        current_day = self._datetime_to_day_index(current_dt, calendar_normalized)
        while current_day < end_day:
            next_dt = self._increment_datetime(current_dt, frequency, calendar_normalized)
            next_day = self._datetime_to_day_index(next_dt, calendar_normalized)
            if next_day <= current_day:
                raise ValidationError(
                    f"Non-increasing period boundaries for frequency={frequency} at {current_dt}"
                )
            clipped_left_day = max(current_day, start_day)
            clipped_next_day = min(next_day, end_day)
            clipped_left_dt = self._day_index_to_datetime(
                clipped_left_day,
                calendar_normalized,
            )
            clipped_next_dt = self._day_index_to_datetime(
                clipped_next_day,
                calendar_normalized,
            )
            left_value = self._datetime_to_units_value(
                clipped_left_dt,
                units,
                calendar_normalized,
            )
            right_value = self._datetime_to_units_value(
                clipped_next_dt,
                units,
                calendar_normalized,
            )
            if right_value > left_value:
                bounds_list.append((left_value, right_value))
            current_dt = next_dt
            current_day = next_day

        if not bounds_list:
            raise ValidationError("No periods generated; check start_time/end_time/frequency inputs")

        bounds = torch.tensor(bounds_list, dtype=torch.float64, device=self.device)
        values = (bounds[:, 0] + bounds[:, 1]) * 0.5
        return TimeCoordinate(values=values, units=units, calendar=calendar_normalized, bounds=bounds)
    
    def align_periods(
        self,
        source_time: TimeCoordinate,
        target_frequency: str,
        label: str = 'left',
        closed: str = 'left'
    ) -> TimeCoordinate:
        """
        Align time periods to target frequency.
        
        Parameters:
            source_time : TimeCoordinate
                Source time coordinate
            target_frequency : str
                Target frequency code
            label : str
                Period labeling ('left' or 'right')
            closed : str
                Interval closure ('left' or 'right')
                
        Returns:
            TimeCoordinate: Aligned time coordinate
        """
        self._parse_period_frequency(target_frequency)
        if label not in {"left", "right"}:
            raise ValidationError(f"Unsupported label: {label}. Expected 'left' or 'right'.")
        if closed not in {"left", "right"}:
            raise ValidationError(f"Unsupported closed: {closed}. Expected 'left' or 'right'.")

        if source_time.bounds is not None:
            source_bounds = source_time.bounds.to(dtype=torch.float64, device=self.device)
        else:
            source_bounds = self.cf_processor.create_time_bounds(
                source_time,
                target_frequency,
                irregular_intervals=True,
            ).to(dtype=torch.float64, device=self.device)

        start_value = float(source_bounds[0, 0].item())
        end_value = float(source_bounds[-1, 1].item())
        if end_value <= start_value:
            raise ValidationError("Source bounds are invalid for alignment.")

        calendar_normalized = self.cf_processor._normalize_calendar(source_time.calendar)
        raw_start_dt = self._units_value_to_datetime(start_value, source_time.units, calendar_normalized)
        period_start_dt = self._normalize_period_start(raw_start_dt, target_frequency, calendar_normalized)
        current_dt = period_start_dt

        # References: CF Conventions 1.8 and overlap-conservative temporal alignment in climate coupling.
        # Expression: J_j = [max(a_j, t0), min(b_j, t1)], where [a_j, b_j) are target periods.
        # Parameters: source_time.bounds (source span), target_frequency (target cadence), label/closed (coordinate convention).
        aligned_bounds: List[Tuple[float, float]] = []
        for _ in range(200000):
            left = self._datetime_to_units_value(current_dt, source_time.units, calendar_normalized)
            next_dt = self._increment_datetime(current_dt, target_frequency, calendar_normalized)
            right = self._datetime_to_units_value(next_dt, source_time.units, calendar_normalized)
            if right <= left:
                raise ValidationError(
                    f"Non-increasing aligned bounds for frequency={target_frequency} at {current_dt}"
                )
            if right <= start_value:
                current_dt = next_dt
                continue
            if left >= end_value:
                break

            clipped_left = max(left, start_value)
            clipped_right = min(right, end_value)
            if clipped_right > clipped_left:
                aligned_bounds.append((clipped_left, clipped_right))
            if right >= end_value:
                break
            current_dt = next_dt
        else:
            raise ValidationError("Alignment loop exceeded safety bound; check calendar/frequency inputs.")

        if not aligned_bounds:
            aligned_bounds = [(start_value, end_value)]

        bounds = torch.tensor(aligned_bounds, dtype=torch.float64, device=self.device)
        if label == "left":
            values = bounds[:, 0]
        else:
            values = bounds[:, 1]

        return TimeCoordinate(
            values=values,
            units=source_time.units,
            calendar=calendar_normalized,
            bounds=bounds,
        )


def update_cell_methods(
    original: str,
    operation: str,
    dimension: str = 'time'
) -> str:
    """
    Update CF cell_methods string with new operation.
    
    Parameters:
        original : str
            Original cell_methods string
        operation : str
            New operation to add/replace
        dimension : str
            Dimension name
            
    Returns:
        str: Updated cell_methods string
        
    Examples:
        update_cell_methods('', 'mean', 'time') -> 'time: mean'
        update_cell_methods('time: point', 'mean', 'time') -> 'time: mean'
        update_cell_methods('time: point area: mean', 'sum', 'time') -> 'time: sum area: mean'
    """
    if not original:
        return f"{dimension}: {operation}"
    
    # Parse existing cell methods
    parts = original.split()
    
    # Find and replace existing dimension method
    for i, part in enumerate(parts):
        if part == f"{dimension}:":
            if i + 1 < len(parts):
                parts[i + 1] = operation
                return " ".join(parts)
            else:
                parts.append(operation)
                return " ".join(parts)
    
    # Add new dimension method
    return f"{dimension}: {operation} {original}".strip()


# Export main classes and functions
__all__ = [
    # Core classes
    'TimeWeightGenerator',
    'TemporalBoundsRelation',
    'TemporalOverlapPlan',
    'TemporalUnitContract',
    'TemporalGapDiagnostic',
    'TemporalKernelResult',
    'TemporalResidualResult',
    'PeriodIndexer', 
    'CFTimeProcessor',
    'TimeCoordinate',
    
    # Exception classes
    'ValidationError',
    'CalendarError',
    'AggregationError',
    'DisaggregationError',
    
    # Utility functions
    'update_cell_methods',
    'get_calendar_definitions',
    'get_supported_calendar_names',
    'apply_temporal_kernel',
    'compute_temporal_mapping_residual',
    'map_state_boundaries',
    'FREQUENCY_HIERARCHY'
]
