"""
Temporal Calendar and Time Weight Calculation Module

This module provides calendar handling and time weight computation functionality
for temporal aggregation and disaggregation operations in the RELIC framework.
Supports multiple calendar types and CF-1.8 compliant time bounds processing.
CalendarHandler acts as a thin wrapper over the unified
TimeWeightGenerator; prefer using TimeWeightGenerator for new code paths to
avoid duplicated logic.

The CalendarHandler class manages different calendar systems and computes
overlap-based time weights for accurate temporal averaging while preserving
conservation properties and maintaining PyTorch differentiability.

Mathematical Framework:
    Time weights based on interval overlap:
    w_ij = overlap(source_interval_i, target_interval_j) / duration(target_interval_j)
    
    Conservation constraint:
    Sigma_i w_ij = 1 for each target interval j (perfect coverage)
    
    Temporal aggregation:
    phi_target(j) = Sigma_i w_ij * phi_source(i)
    
    Missing data handling:
    w_ij_normalized = w_ij / Sigma_k w_kj (renormalization)
    if Sigma_k w_kj < threshold: phi_target(j) = NaN

Classes:
    CalendarHandler: Main calendar management and conversion class
    
Functions:
    compute_time_weights: Standalone function for time weight calculation

References:
    - CF Conventions Version 1.8: http://cfconventions.org/cf-conventions/v1.8.0/
    - NetCDF Climate and Forecast Metadata Conventions
    - UDUNITS-2 Library Documentation
    - ISO 8601:2004 Data elements and interchange formats
    - Gregorian Calendar Reform and Proleptic Extensions
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from relic.core import device_policy
from relic.parameters import get_global_manager
from relic.utils.time_constants import SECONDS_PER_DAY

from .core import (
    CalendarError,
    CFTimeProcessor,
    PeriodIndexer,
    TimeCoordinate,
    TimeWeightGenerator,
    ValidationError,
    get_calendar_definitions,
)


class CalendarHandler(nn.Module):
    """
    Calendar Handler for Temporal Scale Coupling
    
    Manages different calendar systems and provides time weight computation
    for temporal aggregation and disaggregation operations. Supports CF-1.8
    compliant calendar types and time bounds processing.
    
    This class provides the foundational calendar operations needed for
    accurate temporal upscaling and downscaling while maintaining conservation
    properties and numerical stability in PyTorch computations.
    
    Parameters:
        calendar_type : str
            Calendar type following CF conventions
            Supported: 'gregorian', 'proleptic_gregorian', 'noleap', '365_day',
                      '360_day', 'julian', 'all_leap', '366_day'
        time_units : str
            Time units specification (e.g., "days since 1850-01-01 00:00:00")
        reference_date : Optional[str]
            Reference date for time units (auto-parsed from time_units if not provided)
        device : Optional[torch.device]
            Computation device
            
    Mathematical Framework:
        Calendar conversion between different temporal reference systems:
        
        Time coordinate transformation:
        t_new = (t_old - t_ref_old) * scale_factor + t_ref_new
        
        Leap year handling:
        is_leap(year) = (year % 4 == 0) ? ((year % 100 != 0) ? (year % 400 == 0))
        
        Day-of-year calculation:
        doy = Sigma_{m=1}^{month-1} days_in_month(m, year) + day
        
        Time bounds overlap calculation:
        overlap(I_1, I_2) = max(0, min(I_1.end, I_2.end) - max(I_1.start, I_2.start))
        
        Weight computation:
        w_ij = overlap(source_i, target_j) / duration(target_j)
        
        Conservation check:
        Sigma_i w_ij = coverage_fraction_j <= 1.0
        
    References:
        - Unidata NetCDF CF Conventions: https://cfconventions.org/
        - ISO 8601:2004 Date and time format
        - Dershowitz, N. & Reingold, E.M. (2008). Calendrical Calculations
        - Meeus, J. (1998). Astronomical Algorithms
    """
    
    def __init__(
        self,
        calendar_type: str = 'gregorian',
        time_units: str = 'days since 1850-01-01 00:00:00',
        reference_date: Optional[str] = None,
        device: Optional[torch.device] = None,
        parameter_manager: Optional[Any] = None,
    ) -> None:
        """
        Initialize calendar handler with specified calendar type and units.
        
        Mathematical Expression:
            Calendar initialization: C = {type, units, reference, conversion_rules}
            Time axis definition: t in [t_start, t_end] with units and calendar
        """
        super(CalendarHandler, self).__init__()

        self._param_manager = (
            parameter_manager
            if parameter_manager is not None
            else get_global_manager()
        )
        if self._param_manager is None:
            raise ValueError(
                "ParameterManager is required for CalendarHandler; pass one "
                "explicitly or initialize the canonical global owner."
            )
        self.device = device or (
            device_policy.resolve_device(default="auto")
        )
        self._cf_processor = CFTimeProcessor(
            device=self.device,
            parameter_manager=self._param_manager,
        )
        self._period_indexer = PeriodIndexer(
            device=self.device,
            parameter_manager=self._param_manager,
        )
        self.calendar_type = self._cf_processor._normalize_calendar(calendar_type)
        self.time_units = time_units
        self._calendar_definitions = get_calendar_definitions(self._param_manager)

        # Validate the complete CF units contract through the canonical owner.
        unit_name, _ = self._period_indexer._parse_units_reference(time_units)
        if reference_date is None:
            self.reference_date = self._parse_reference_date(time_units)
        else:
            self.reference_date = self._parse_reference_date(
                f"{unit_name} since {reference_date}"
            )

        # Calendar-specific parameters
        calendar_meta = self._calendar_definitions[self.calendar_type]
        days_per_year_value = calendar_meta.get("days_per_year")
        if days_per_year_value is None:
            raise ValueError(
                f"Calendar {self.calendar_type!r} is missing days_per_year"
            )
        self.days_per_year = float(days_per_year_value)
        self.months_per_year = int(calendar_meta.get("months_per_year", 12))

        # TimeWeightGenerator for unified overlap/coverage logic. Weights and
        # geometric coverage need no threshold; acceptance is applied by
        # TemporalAggregator against coupling.scales.temporal.coverage_threshold.
        self._weight_generator = TimeWeightGenerator(device=self.device, parameter_manager=self._param_manager)
        self._time_coord_cls = TimeCoordinate

    def _parse_reference_date(self, time_units: str) -> str:
        """
        Parse reference date from CF-compliant time units string.
        
        Parameters:
            time_units : str
                CF time units (e.g., "days since 1850-01-01 00:00:00")
                
        Returns:
            str: Reference date in ISO format
            
        Mathematical Expression:
            Parse pattern: "unit since YYYY-MM-DD [HH:MM:SS[.SSS]]"
            Extract: reference_date = "YYYY-MM-DD HH:MM:SS"
        """
        _, reference = self._period_indexer._parse_units_reference(time_units)
        year, month, day, hour, minute, second = reference
        whole_second = int(second)
        if second == float(whole_second):
            second_text = f"{whole_second:02d}"
        else:
            second_text = f"{second:09.6f}".rstrip("0").rstrip(".")
        return (
            f"{year:04d}-{month:02d}-{day:02d} "
            f"{hour:02d}:{minute:02d}:{second_text}"
        )
    
    def validate_calendar_type(self, calendar_type: str) -> bool:
        """
        Validate if calendar type is supported.
        
        Parameters:
            calendar_type : str
                Calendar type to validate
                
        Returns:
            bool: True if supported, False otherwise
        """
        try:
            self._cf_processor._normalize_calendar(calendar_type)
        except (CalendarError, TypeError, ValueError):
            return False
        return True
    
    def is_leap_year(self, year: int) -> bool:
        """
        Determine if a year is a leap year in the current calendar.
        
        Parameters:
            year : int
                Year to check
                
        Returns:
            bool: True if leap year, False otherwise
            
        Mathematical Expression:
            Gregorian rule: leap = (year % 4 == 0) ? ((year % 100 != 0) ? (year % 400 == 0))
            Julian rule: leap = (year % 4 == 0)
            No-leap rule: leap = False
            All-leap rule: leap = True
        """
        return self._period_indexer._is_leap_year(year, self.calendar_type)
    
    def days_in_month(self, month: int, year: int) -> int:
        """
        Get number of days in a specific month and year.
        
        Parameters:
            month : int
                Month number (1-12)
            year : int
                Year
                
        Returns:
            int: Number of days in the month
            
        Scientific contract:
            This adapter delegates the complete calendar rule to PeriodIndexer.
            It owns no month-length table or leap-year branch.
        """
        return self._period_indexer._days_in_month(
            year,
            month,
            self.calendar_type,
        )
    
    def days_in_year(self, year: int) -> int:
        """
        Get total number of days in a year.
        
        Parameters:
            year : int
                Year
                
        Returns:
            int: Number of days in the year
        """
        start = self._period_indexer._datetime_to_day_index(
            (year, 1, 1, 0, 0, 0.0),
            self.calendar_type,
        )
        end = self._period_indexer._datetime_to_day_index(
            (year + 1, 1, 1, 0, 0, 0.0),
            self.calendar_type,
        )
        return int(end - start)

    @staticmethod
    def days_per_simulator_year(calendar_type: str) -> int:
        """Return the exact year length for a supported simulator calendar.

        References:
            CF Conventions 1.8, Section 4.4; revised foundations, time-window
            and calendar contract.
        Mathematical expression:
            Y(C) = 365 for noleap and Y(C) = 360 for 360_day.
        Parameter meanings:
            C is the simulator-owned calendar and Y(C) has units day year-1.
        """
        if calendar_type not in {"noleap", "360_day"}:
            raise ValueError(
                "Simulator calendar_type must be noleap or 360_day."
            )
        definitions = get_calendar_definitions()
        calendar_definition = definitions.get(calendar_type)
        if not isinstance(calendar_definition, dict):
            raise ValueError(
                f"Missing calendar definition for {calendar_type}."
            )
        return int(calendar_definition["days_per_year"])

    @staticmethod
    def resolve_simulator_calendar(
        configured_calendar: Optional[Any],
        runtime_calendar: Optional[Any],
        *,
        owner_name: str,
    ) -> str:
        """Resolve one explicit simulator calendar without silent fallback.

        References:
            CF Conventions 1.8, Section 4.4; revised foundations, calendar
            ownership and runtime-context contract.
        Mathematical expression:
            C = C_runtime if declared, otherwise C_config, with
            C_runtime = C_config whenever both are declared.
        Parameter meanings:
            C_config is the configured calendar, C_runtime is the simulator
            calendar supplied for the step, and owner_name identifies the
            consuming Functor in fail-closed diagnostics.
        """
        configured = (
            None
            if configured_calendar is None
            else str(configured_calendar)
        )
        runtime = (
            None
            if runtime_calendar is None
            else str(runtime_calendar)
        )
        if configured is not None and runtime is not None and configured != runtime:
            raise ValueError(
                "Runtime calendar_type conflicts with the configured "
                f"calendar_type for {owner_name}: {runtime} != {configured}."
            )
        resolved = runtime if runtime is not None else configured
        if resolved is None:
            raise ValueError(
                f"{owner_name} requires explicit calendar_type from the "
                "simulator or node config."
            )
        CalendarHandler.days_per_simulator_year(resolved)
        return resolved

    @staticmethod
    def _finite_real_scalar(value: Any, *, error_message: str) -> Tensor:
        """Return one detached finite real scalar tensor or fail closed."""
        if isinstance(value, bool):
            raise ValueError(error_message)
        try:
            scalar = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(error_message) from exc
        if scalar.ndim != 0 or scalar.dtype == torch.bool or scalar.is_complex():
            raise ValueError(error_message)
        scalar = scalar.detach()
        try:
            finite = bool(torch.isfinite(scalar).item())
        except (TypeError, RuntimeError) as exc:
            raise ValueError(error_message) from exc
        if not finite:
            raise ValueError(error_message)
        return scalar

    @staticmethod
    def validate_simulator_day(day: Any, calendar_type: str) -> int:
        """Validate a one-based integer day in the declared calendar.

        References:
            CF Conventions 1.8, Section 4.4; revised foundations, calendar
            domain contract.
        Mathematical expression:
            day in {1, ..., Y(C)}.
        Parameter meanings:
            day is a one-based calendar index, C is the simulator calendar,
            and Y(C) is its exact length in days. A one-element Tensor is the
            singleton-batch encoding of the same global coordinate.
        """
        days_per_year = CalendarHandler.days_per_simulator_year(calendar_type)
        message = (
            "day_of_year must be a finite in-domain integer numeric scalar "
            f"for calendar_type={calendar_type}."
        )
        normalized_day = (
            day.reshape(())
            if isinstance(day, Tensor) and day.numel() == 1
            else day
        )
        scalar = CalendarHandler._finite_real_scalar(
            normalized_day,
            error_message=message,
        ).to(dtype=torch.float64)
        integral = bool(torch.eq(scalar, torch.trunc(scalar)).item())
        in_domain = bool(
            torch.logical_and(
                scalar >= 1.0,
                scalar <= float(days_per_year),
            ).item()
        )
        if not integral or not in_domain:
            raise ValueError(message)
        return int(scalar.item())

    @staticmethod
    def day_of_year_from_elapsed_seconds(
        time: Any,
        calendar_type: str,
    ) -> int:
        """Resolve a one-based day within a simulator calendar year.

        References:
            CF Conventions 1.8, Section 4.4; revised foundations, time-window
            and calendar contract.
        Mathematical expression:
            doy = floor(t / seconds_per_day) mod days_per_year(C) + 1.
        Parameter meanings:
            t is elapsed simulation time [s] and C is the simulator calendar.
        """
        days_per_year = CalendarHandler.days_per_simulator_year(calendar_type)
        message = "simulation time must be a finite non-negative numeric scalar"
        time_value = CalendarHandler._finite_real_scalar(
            time,
            error_message=message,
        ).to(dtype=torch.float64)
        if bool(torch.lt(time_value, 0.0).item()):
            raise ValueError(message)
        seconds_per_day = torch.as_tensor(
            SECONDS_PER_DAY,
            dtype=time_value.dtype,
            device=time_value.device,
        )
        elapsed_days = torch.div(
            time_value,
            seconds_per_day,
            rounding_mode="floor",
        )
        day_index = torch.remainder(elapsed_days, days_per_year) + 1.0
        return int(day_index.item())
    
    def compute_time_weights(
        self,
        source_time: Tensor,
        source_bounds: Tensor,
        target_time: Tensor,
        target_bounds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute time weights for temporal aggregation based on interval overlaps.

        Returns the target-fraction view and the geometric target coverage of
        the canonical ``TimeWeightGenerator.build_overlap_plan``. Neither
        depends on a coverage threshold; acceptance against
        ``coupling.scales.temporal.coverage_threshold`` is applied by
        ``TemporalAggregator`` to the valid coverage, not here.

        Parameters:
            source_time : Tensor
                Source time coordinates
                Shape: [n_source]
                Units: [time units as specified in time_units]
            source_bounds : Tensor
                Source time bounds (start, end) for each time step
                Shape: [n_source, 2]
                Units: [time units]
            target_time : Tensor
                Target time coordinates
                Shape: [n_target]
                Units: [time units]
            target_bounds : Tensor
                Target time bounds (start, end) for each time step
                Shape: [n_target, 2]
                Units: [time units]

        Returns:
            Tuple[Tensor, Tensor]:
                weights : Weight matrix for aggregation
                    Shape: [n_source, n_target]
                    Units: [dimensionless]
                    Each column sums to <= 1.0 (perfect coverage = 1.0)
                coverage : Coverage fraction for each target interval
                    Shape: [n_target]
                    Units: [dimensionless]
                    Range: [0.0, 1.0]
                    
        Raises:
            ValueError: If input dimensions are inconsistent
            RuntimeError: If weight computation fails
            
        Mathematical Expression:
            Interval overlap calculation:
            overlap(I_source_i, I_target_j) = max(0, min(I_source_i.end, I_target_j.end) - max(I_source_i.start, I_target_j.start))
            
            Weight computation:
            w_ij = overlap(I_source_i, I_target_j) / duration(I_target_j)
            where duration(I_target_j) = I_target_j.end - I_target_j.start
            
            Coverage calculation:
            coverage_j = Sigma_i w_ij = Sigma_i overlap(I_source_i, I_target_j) / duration(I_target_j)
            
            Geometric coverage constraint:
            Sigma_i w_ij <= 1.0 for all j (complete coverage when equal to 1.0)

        References:
            - CF Conventions Section 7.1: Cell Boundaries
            - NetCDF Climate and Forecast Metadata Conventions
            - Temporal Aggregation Methods in Climate Data Processing
        """
        # Validate input dimensions
        n_source = source_time.size(0)
        n_target = target_time.size(0)
        
        if source_bounds.shape != (n_source, 2):
            raise ValueError(f"source_bounds shape {source_bounds.shape} mismatched with source_time {source_time.shape}")
        if target_bounds.shape != (n_target, 2):
            raise ValueError(f"target_bounds shape {target_bounds.shape} mismatched with target_time {target_time.shape}")
        
        # Ensure bounds are properly ordered (start < end)
        source_starts = source_bounds[:, 0]  # [n_source]
        source_ends = source_bounds[:, 1]    # [n_source]
        target_starts = target_bounds[:, 0]  # [n_target]
        target_ends = target_bounds[:, 1]    # [n_target]
        
        # Check for invalid intervals
        if torch.any(source_ends <= source_starts):
            warnings.warn("Some source intervals have invalid bounds (end <= start)")
        if torch.any(target_ends <= target_starts):
            warnings.warn("Some target intervals have invalid bounds (end <= start)")
        
        # Delegate to unified TimeWeightGenerator for consistency
        if self._weight_generator is None or self._time_coord_cls is None:
            raise RuntimeError("TimeWeightGenerator unavailable; temporal core not initialized.")

        source_coord = self._time_coord_cls(
            values=source_time.to(self.device),
            bounds=source_bounds.to(self.device),
            calendar=self.calendar_type,
            units=self.time_units,
        )
        target_coord = self._time_coord_cls(
            values=target_time.to(self.device),
            bounds=target_bounds.to(self.device),
            calendar=self.calendar_type,
            units=self.time_units,
        )

        # References: CF Conventions 1.13, Section 7.1 (cell boundaries);
        #   Jones (1999), Mon. Weather Rev. 127:2204-2210 (overlap weights).
        # Mathematical expression: O_ij = max(0, min(a_i+, b_j+) - max(a_i-, b_j-));
        #   w_ij = O_ij / (b_j+ - b_j-); C_j = sum_i w_ij.
        # Parameter meanings: a_i-, a_i+ source bounds; b_j-, b_j+ target bounds;
        #   w_ij target-fraction weight; C_j geometric target coverage.
        # Units: bounds and O_ij in time_units; w_ij and C_j dimensionless.
        # Domain/constraints: finite, positive-duration, non-overlapping bounds
        #   in one calendar; C_j <= 1 (overlapping sources raise); no threshold.
        # Sign convention: O_ij, w_ij, and C_j are nonnegative.
        plan = self._weight_generator.build_overlap_plan(source_coord, target_coord)
        return plan.target_fractions, plan.geometric_target_coverage
    
    def create_time_bounds(
        self,
        time_coords: Tensor,
        frequency: str = 'daily',
        method: str = 'centered'
    ) -> Tensor:
        """
        Create time bounds for time coordinates based on frequency and method.
        
        Validate a public frequency alias and delegate bound construction to the
        canonical CFTimeProcessor/PeriodIndexer owner.
        
        Parameters:
            time_coords : Tensor
                Time coordinate values
                Shape: [n_time]
                Units: [time units as specified in time_units]
            frequency : str
                Temporal frequency for bound generation
                Simple aliases: hourly, daily, weekly, monthly, yearly, annual.
                Anchored aliases: QS-MON or SEASON-DJF/MAM/JJA/SON.
                Default: 'daily'
            method : str
                Method for bound placement
                Options: 'centered', 'start', 'end'
                Default: 'centered'
                
        Returns:
            Tensor: Time bounds array
                Shape: [n_time, 2]
                Units: [time units]
                
        References:
            CF Conventions 1.13, time coordinates and cell bounds.
        Mathematical expression:
            B = PeriodIndexerBounds(t, normalized_frequency, calendar).
        Parameter meanings:
            t is the coordinate, and B contains half-open period bounds.
        Units:
            Input coordinates and returned bounds use self.time_units.
        Domain/constraints:
            Calendar periods require explicit anchors; only centered wrapper mode
            is accepted because start/end inference needs explicit source bounds.
        Sign convention:
            Bounds are strictly increasing from left to right.
        """
        if method != 'centered':
            raise ValueError(
                "method must be centered; start/end alignment requires "
                "explicit source bounds."
            )
        if not isinstance(time_coords, Tensor):
            raise TypeError("time_coords must be a torch.Tensor")
        if time_coords.layout != torch.strided or time_coords.device.type == "meta":
            raise ValidationError(
                "time_coords must be a materialized strided tensor"
            )
        if time_coords.dtype not in {torch.float32, torch.float64}:
            raise ValidationError("time_coords must use float32 or float64")
        if time_coords.dim() != 1:
            raise ValidationError("time_coords must be one-dimensional")
        if time_coords.numel() == 0:
            raise ValidationError("time_coords must be non-empty")
        if not torch.all(torch.isfinite(time_coords)):
            raise ValidationError("time_coords must contain only finite values")
        public_aliases = {
            'hourly': 'H',
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'yearly': 'Y',
            'annual': 'Y',
        }
        if not isinstance(frequency, str) or not frequency.strip():
            raise ValidationError("frequency must be a non-empty string")
        normalized_frequency = public_aliases.get(
            frequency.strip().lower(),
            frequency.strip(),
        )
        self._period_indexer._parse_period_frequency(normalized_frequency)
        time_coord = self._cf_processor.parse_cf_time(
            time_coords,
            units=self.time_units,
            calendar=self.calendar_type,
        )
        return self._cf_processor.create_time_bounds(
            time_coord,
            normalized_frequency,
        ).to(dtype=time_coords.dtype, device=self.device)
    
    def get_calendar_metadata(self) -> Dict[str, Any]:
        """
        Get serialized CF time-coordinate metadata without dataset claims.
        
        Returns:
            Dict[str, Any]: Calendar metadata dictionary
        """
        calendar_meta = self._calendar_definitions[self.calendar_type]
        leap_rule = calendar_meta.get("leap_rule")
        if not isinstance(leap_rule, str) or not leap_rule.strip():
            raise ValueError(
                f"Calendar {self.calendar_type!r} is missing leap_rule"
            )
        normalized_leap_rule = leap_rule.strip().lower()
        return {
            'calendar': self.calendar_type,
            'units': self.time_units,
            'reference_date': self.reference_date,
            'days_per_year': self.days_per_year,
            'months_per_year': self.months_per_year,
            'leap_rule': normalized_leap_rule,
            'leap_year_support': normalized_leap_rule not in {'none', '360_day'},
            'cf_time_coordinate_metadata_serialized': True,
            'dataset_conformance_checked': False,
            'description': calendar_meta.get("description", "")
        }


def compute_time_weights(
    source_time: Tensor,
    source_bounds: Tensor,
    target_time: Tensor,
    target_bounds: Tensor,
    calendar_type: str = 'gregorian',
    device: Optional[torch.device] = None
) -> Tuple[Tensor, Tensor]:
    """
    Standalone function to compute time weights for temporal aggregation.
    
    Convenience function that creates a temporary CalendarHandler and computes
    time weights. Useful when calendar handling is not needed beyond weight computation.
    
    Parameters:
        source_time : Tensor
            Source time coordinates [n_source]
        source_bounds : Tensor
            Source time bounds [n_source, 2]
        target_time : Tensor
            Target time coordinates [n_target]
        target_bounds : Tensor
            Target time bounds [n_target, 2]
        calendar_type : str
            Calendar type (default: 'gregorian')
        device : Optional[torch.device]
            Computation device
            
    Returns:
        Tuple[Tensor, Tensor]: (weights, coverage)
        
    Mathematical Expression:
        See CalendarHandler.compute_time_weights() for detailed mathematics
    """
    if device is None:
        device = source_time.device
    
    # Create temporary calendar handler
    calendar = CalendarHandler(
        calendar_type=calendar_type,
        device=device
    )
    
    # Compute and return weights
    return calendar.compute_time_weights(
        source_time=source_time,
        source_bounds=source_bounds,
        target_time=target_time,
        target_bounds=target_bounds,
    )


# Export main classes and functions
__all__ = [
    'CalendarHandler',
    'compute_time_weights',
]
