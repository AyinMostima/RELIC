"""
Tillage Management Module

This module implements tillage practices and their direct soil disturbance,
residue-incorporation, porosity, roughness, and carbon-export effects. It does
not calculate decomposition or infiltration process laws; those remain owned
by their registered pedosphere and hydrosphere Functors.

References:
    - Lal, R. (2004). Soil carbon sequestration impacts on global climate change and food security. Science 304, 1623-1627.
    - Paustian, K., et al. (2000). CO2 mitigation by agriculture: An overview. Climatic Change 40, 135-162.
    - Six, J., et al. (2002). Stabilization mechanisms of soil organic matter: Implications for C-saturation of soils. Plant and Soil 241, 155-176.
    - Reicosky, D.C. (1997). Tillage-induced CO2 emission from soil. Nutrient Cycling in Agroecosystems 49, 273-285.

Mathematical Expression:
    f_time = exp(-k_decay * t_since)
    D_tillage = k_disturbance * I * f_theta
    R_incorporation = clamp(k_residue * I * (a_theta + f_moisture), 0, 1)
    F_C = k_C * (1 + a_I * I)
    f_porosity = 1 + a_p * I * f_time

    where all response factors are dimensionless, t_since is in days,
    and F_C is the tillage carbon-export rate [g C m-2 s-1].
"""

import torch
from typing import Any, Dict, List, Mapping, Optional
from enum import Enum
from dataclasses import dataclass

from relic.utils.time_constants import SECONDS_PER_DAY
from relic.utils.physical_constants import get_physical_constant

from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.functor import TransportMechanismFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.coupling.scales.temporal.calendar import CalendarHandler
from .event_scheduler import resolve_day_of_year


class TillageDepth(Enum):
    """Tillage depth categories"""
    SHALLOW = "shallow"     # 0-10 cm depth
    MEDIUM = "medium"       # 10-25 cm depth
    DEEP = "deep"          # 25-40 cm depth
    VERY_DEEP = "very_deep" # >40 cm depth


class TillageIntensity(Enum):
    """Tillage intensity levels"""
    NONE = "none"           # No tillage
    LOW = "low"            # Minimal disturbance
    MODERATE = "moderate"   # Moderate disturbance
    HIGH = "high"          # High disturbance
    VERY_HIGH = "very_high" # Maximum disturbance


@dataclass
class TillageEvent:
    """
    Data class for tillage operation events
    
    Attributes:
        depth: Tillage depth category (selects the depth factor)
        intensity: Tillage intensity level (selects the intensity factor)
        date: Day of year for operation
        soil_moisture_limit: Maximum soil moisture for operation [m^3/m^3]

    Every field enters the schedule or the responses. The date and moisture-limit
    defaults are construction values of a user-specified event, not model
    parameters; the default schedule reads its values from the parameter database.
    """
    depth: TillageDepth
    intensity: TillageIntensity
    date: int = 100  # Day of year
    soil_moisture_limit: float = 0.25  # m^3/m^3


class TillageManagement(TransportMechanismFunctor):
    """
    Tillage Management Functor
    
    Implements tillage operations and their direct effects on soil disturbance,
    porosity, residue incorporation, and carbon export.
    
    Parameters:
        tillage_schedule: List of tillage operation events
        porosity_change: Base porosity change factor [1]
        co2_pulse_factor: tillage carbon-flush factor [1]
        effect_decay_rate: Rate of tillage effect decay [day^-1]
    """

    TRANSPORT_TYPE = "managed_application"
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    EXTRA_STATE_FORMAT = "relic.tillage.runtime_state"
    EXTRA_STATE_VERSION = 1
    
    def __init__(self, name: str = "tillage_management", config: Optional[Dict] = None):
        """
        Initialize tillage management functor
        
        Args:
            name: Functor name
            config: Configuration dictionary
        """
        # Set up metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Tillage management and soil disturbance effects",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "F_disturbance = k_disturbance I f_theta; "
                "F_C = C_soil f_0 f_p I D f_T / S_day in an operation step, 0 otherwise; "
                "f_porosity = 1 + a_p I f_time"
            ),
            references=[
                "Lal (2004) DOI:10.1126/science.1097396",
                "Paustian et al. (2000) DOI:10.1023/A:1005554205788", 
                "Six et al. (2002) DOI:10.1023/A:1016125726789",
                "Reicosky (1997) DOI:10.1023/A:1009766510274",
                "Davidson and Janssens (2006) DOI:10.1038/nature04514"
            ],
            parameters={
                "anthroposphere.management.tillage.disturbance_factor": (
                    "Base soil disturbance scaling for tillage operations [dimensionless]"
                ),
                "anthroposphere.management.tillage.residue_incorporation_rate": (
                    "Fraction of surface residue incorporated by tillage [dimensionless]"
                ),
                "anthroposphere.management.tillage.effect_decay_rate": (
                    "Tillage effect decay rate [day-1]"
                ),
                "anthroposphere.management.tillage.porosity_change": (
                    "Soil porosity change factor [dimensionless]"
                ),
                "anthroposphere.management.tillage.roughness_intensity_coefficient": (
                    "Intensity response coefficient for tillage roughness modifier "
                    "[dimensionless]"
                ),
                "anthroposphere.management.tillage.schedule_window_days": (
                    "Half-width of the calendar window around a scheduled operation day [day]"
                ),
                "anthroposphere.management.tillage.minimum_operation_interval_days": (
                    "Minimum elapsed time since the last operation before the next one [day]"
                ),
                "anthroposphere.management.tillage.default_schedule.*": (
                    "Day of year [day] and soil-moisture upper limit [m3 m-3] of the four "
                    "default operations [units per key]"
                ),
                "anthroposphere.management.tillage.co2_pulse_base_fraction": (
                    "Base fractional scaling for tillage induced CO2 pulse emissions "
                    "[dimensionless]"
                ),
                "anthroposphere.management.tillage.co2_pulse_factor": (
                    "CO2 emission pulse factor [dimensionless]"
                ),
                "anthroposphere.management.tillage.moisture_optimum": (
                    "Optimal soil moisture for tillage disturbance response [m3 m-3]"
                ),
                "anthroposphere.management.tillage.moisture_sigma": (
                    "Gaussian width for moisture response under tillage [m3 m-3]"
                ),
                "anthroposphere.management.tillage.moisture_effect_denominator": (
                    "Soil moisture scaling denominator for tillage moisture effect [m3 m-3]"
                ),
                "anthroposphere.management.tillage.residue_moisture_offset": (
                    "Offset term applied to moisture response in tillage residue incorporation "
                    "[dimensionless]"
                ),
                "anthroposphere.management.tillage.*_intensity_factor": (
                    "Intensity multiplier per tillage intensity class (none, low, moderate, high, "
                    "very high) [dimensionless]"
                ),
                "anthroposphere.management.tillage.*_depth_factor": (
                    "Depth multiplier per tillage depth class (shallow, medium, deep, very deep) "
                    "[dimensionless]"
                ),
                "pedosphere.physical.q10_decomposition": (
                    "Q10 of soil organic matter decomposition in the tillage CO2 pulse "
                    "[dimensionless]"
                ),
                "validation.conservation.conservation_tolerance": (
                    "General relative tolerance of the conservation checks [dimensionless]"
                ),
            },
            input_variables=[
                "soil_moisture", "soil_temperature", "soil_organic_carbon",
                "day_of_year", "days_since_tillage", "residue_cover",
            ],
            output_variables=[
                "porosity_modifier",
                "roughness_modifier",
                "days_since_tillage",
                "tillage_intensity_factor",
                "time_decay_factor",
                "cumulative_operations",
                "soil_disturbance_rate",
                "residue_incorporation_rate",
                "carbon_loss_rate",
                "soil_co2_flux",
                "delta_residue_cover",
                "diagnostics.tillage_mass_balance_error",
            ],
            conservation_laws=["mass"],  # Soil mass balance
            spatial_dimensions=2,
            temporal_scale=TemporalScale.DAY,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            coupling_interfaces=["pedosphere", "hydrosphere"]
        )
        
        # Default configuration
        # Initialize TransportMechanismFunctor base class (includes FunctorBase initialization)
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata
        )
        # residue_cover [1] is read only when an operation executes (the incorporated
        # cover); a step without an operation does not need it.
        self.optional_input_variables = {"residue_cover"}
        configured_calendar = (config or {}).get("calendar_type")
        self._configured_calendar_type = (
            None
            if configured_calendar is None
            else str(configured_calendar)
        )
        if self._configured_calendar_type is not None:
            CalendarHandler.days_per_simulator_year(
                self._configured_calendar_type
            )
        
        # Initialize tillage schedule
        self.tillage_events: List[TillageEvent] = []
        self._initialize_default_schedule()
        if self._configured_calendar_type is not None:
            for event in self.tillage_events:
                self._validate_tillage_event_date(
                    event.date,
                    self._configured_calendar_type,
                )
        
        # Internal tillage state at construction: the untilled soil before any
        # event (construction values of this Functor's own replay state, restored
        # by set_extra_state on rollback; not substitutes of a missing input).
        # Intensity class NONE maps to none_intensity_factor (no disturbance);
        # the depth class is read only when an event sets its own depth; no
        # operation has happened; the porosity and roughness modifiers are the
        # neutral factor 1 (Lal 2004; Six et al. 2002: tillage alters them only
        # after an operation).
        self.current_intensity = TillageIntensity.NONE
        self.current_depth = TillageDepth.SHALLOW
        self.cumulative_tillage_operations = 0
        self.porosity_modifier = torch.ones((), dtype=torch.float32)
        self.roughness_modifier = torch.ones((), dtype=torch.float32)

    def get_extra_state(self) -> Dict[str, Any]:
        """Return replay-critical tillage modifiers for step rollback.

        References:
          - PyTorch Module extra-state contract.
          - Revised RELIC foundations, accepted-step transaction contract.
        Mathematical expression:
          z_hidden' = (I, d, n, f_porosity, f_roughness); rollback restores z_hidden.
        Parameter meanings:
          I is tillage intensity [1], d is depth class, n is accepted event count,
          and f_porosity/f_roughness are dimensionless soil-surface modifiers.
        """

        return {
            "format": self.EXTRA_STATE_FORMAT,
            "version": self.EXTRA_STATE_VERSION,
            "current_intensity": self.current_intensity.value,
            "current_depth": self.current_depth.value,
            "cumulative_tillage_operations": self.cumulative_tillage_operations,
            "porosity_modifier": self.porosity_modifier.detach().clone(),
            "roughness_modifier": self.roughness_modifier.detach().clone(),
        }

    def set_extra_state(self, state: Any) -> None:
        """Restore replay-critical tillage modifiers after strict validation."""

        if not isinstance(state, Mapping):
            raise TypeError("Tillage extra state must be a mapping.")
        expected_keys = {
            "format",
            "version",
            "current_intensity",
            "current_depth",
            "cumulative_tillage_operations",
            "porosity_modifier",
            "roughness_modifier",
        }
        if set(state) != expected_keys:
            raise ValueError("Tillage extra-state keys do not match the required schema.")
        if state["format"] != self.EXTRA_STATE_FORMAT:
            raise ValueError("Tillage extra-state format is unsupported.")
        if state["version"] != self.EXTRA_STATE_VERSION:
            raise ValueError("Tillage extra-state version is unsupported.")
        try:
            intensity = TillageIntensity(state["current_intensity"])
            depth = TillageDepth(state["current_depth"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Tillage extra-state enum value is invalid.") from exc
        count = state["cumulative_tillage_operations"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                "Tillage cumulative_tillage_operations must be a non-negative integer."
            )

        restored_modifiers: Dict[str, torch.Tensor] = {}
        for key in ("porosity_modifier", "roughness_modifier"):
            value = state[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.numel() == 0
                or not torch.is_floating_point(value)
                or not torch.all(torch.isfinite(value))
                or torch.any(value <= 0.0)
            ):
                raise ValueError(
                    f"Tillage {key} must be a non-empty finite positive floating tensor."
                )
            restored_modifiers[key] = value.detach().clone().to(device=self.device)

        self.current_intensity = intensity
        self.current_depth = depth
        self.cumulative_tillage_operations = count
        self.porosity_modifier = restored_modifiers["porosity_modifier"]
        self.roughness_modifier = restored_modifiers["roughness_modifier"]

    def _initialize_default_schedule(self) -> None:
        """Initialize the default tillage schedule from the parameter database.

        The default scenario is a deep high-intensity primary pass, a shallow secondary
        pass, a low-intensity conservation pass and an autumn chisel pass; their days
        of year and soil-moisture upper limits are the scenario conventions
        anthroposphere.management.tillage.default_schedule.*.
        """
        scenario = (
            ("primary", TillageDepth.DEEP, TillageIntensity.HIGH),
            ("secondary", TillageDepth.SHALLOW, TillageIntensity.MODERATE),
            ("conservation", TillageDepth.MEDIUM, TillageIntensity.LOW),
            ("autumn_chisel", TillageDepth.MEDIUM, TillageIntensity.MODERATE),
        )
        for name, depth, intensity in scenario:
            prefix = f"default_schedule.{name}"
            date_value = self.require_scalar(
                f"{prefix}.date",
                namespace="anthroposphere.management.tillage",
            )
            if date_value != int(date_value):
                raise ValueError(f"{prefix}.date must be an integer day, got {date_value}")
            self.tillage_events.append(
                TillageEvent(
                    depth=depth,
                    intensity=intensity,
                    date=int(date_value),
                    soil_moisture_limit=self.require_scalar(
                        f"{prefix}.soil_moisture_limit",
                        namespace="anthroposphere.management.tillage",
                        units="m3 m-3",
                    ),
                )
            )
    
    def get_metadata(self) -> FunctorMetadata:
        """Return the canonical Functor metadata for runtime introspection."""
        return self.metadata

    @staticmethod
    def _scalar_from_value(value: Any) -> float:
        """Convert numeric/tensor-like values to Python float without item/cpu APIs."""
        if isinstance(value, torch.Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    @staticmethod
    def _validate_tillage_event_date(value: Any, calendar_type: str) -> int:
        """Validate an event date through the canonical calendar owner."""
        try:
            return CalendarHandler.validate_simulator_day(value, calendar_type)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Tillage event date must be a finite in-domain integer "
                f"for calendar_type={calendar_type}."
            ) from exc

    def add_tillage_event(
        self,
        event: TillageEvent,
        *,
        calendar_type: Optional[str] = None,
    ) -> None:
        """
        Add a tillage operation event to schedule
        
        Args:
            event: TillageEvent defining operation details
            calendar_type: Existing simulator calendar identifier
        """
        if not isinstance(event, TillageEvent):
            raise TypeError("event must be a TillageEvent instance.")
        resolved_calendar = CalendarHandler.resolve_simulator_calendar(
            configured_calendar=self._configured_calendar_type,
            runtime_calendar=calendar_type,
            owner_name=f"{self.__class__.__name__}.add_tillage_event",
        )
        self._validate_tillage_event_date(event.date, resolved_calendar)
        self.tillage_events.append(event)
    
    def _get_tillage_intensity_factor(self, intensity: TillageIntensity) -> float:
        """
        Get numerical intensity factor for tillage intensity level
        
        Args:
            intensity: Tillage intensity enum
            
        Returns:
            Intensity factor [1]
        """
        # References: Lal (2004); Ogle et al. (2005).
        # Mathematical expression: I = lookup(intensity).
        # Parameter meanings: intensity selects the configured disturbance class.
        # Units: I [1].
        # Domain/constraints: intensity is a declared enum; factors are finite and non-negative.
        # Sign convention: larger positive I denotes stronger soil disturbance.
        # Get intensity factors from ParameterManager
        def _factor(name: str) -> float:
            return self.require_scalar(name, namespace="anthroposphere.management.tillage")

        none_factor = _factor("none_intensity_factor")
        low_factor = _factor("low_intensity_factor")
        moderate_factor = _factor("moderate_intensity_factor")
        high_factor = _factor("high_intensity_factor")
        very_high_factor = _factor("very_high_intensity_factor")
        
        factors = {
            TillageIntensity.NONE: none_factor,
            TillageIntensity.LOW: low_factor,
            TillageIntensity.MODERATE: moderate_factor,
            TillageIntensity.HIGH: high_factor,
            TillageIntensity.VERY_HIGH: very_high_factor
        }
        if not isinstance(intensity, TillageIntensity):
            raise TypeError(
                f"Tillage intensity must be a TillageIntensity, got {type(intensity).__name__}"
            )
        return factors[intensity]
    
    def _get_tillage_depth_factor(self, depth: TillageDepth) -> float:
        """
        Get depth factor for tillage effects
        
        Args:
            depth: Tillage depth enum
            
        Returns:
            Depth factor [1]
        """
        # References: Lal (2004); Reicosky (1997).
        # Mathematical expression: D = lookup(depth).
        # Parameter meanings: depth selects the configured tillage-depth class.
        # Units: D [1].
        # Domain/constraints: depth is a declared enum; factors are finite and non-negative.
        # Sign convention: larger positive D denotes a stronger depth multiplier.
        # Get depth factors from ParameterManager
        def _depth(name: str) -> float:
            return self.require_scalar(name, namespace="anthroposphere.management.tillage")

        shallow_factor = _depth("shallow_depth_factor")
        medium_factor = _depth("medium_depth_factor")
        deep_factor = _depth("deep_depth_factor")
        very_deep_factor = _depth("very_deep_depth_factor")
        
        factors = {
            TillageDepth.SHALLOW: shallow_factor,
            TillageDepth.MEDIUM: medium_factor,
            TillageDepth.DEEP: deep_factor,
            TillageDepth.VERY_DEEP: very_deep_factor
        }
        if not isinstance(depth, TillageDepth):
            raise TypeError(
                f"Tillage depth must be a TillageDepth, got {type(depth).__name__}"
            )
        return factors[depth]
    
    def _calculate_time_decay_factor(self, days_since: torch.Tensor) -> torch.Tensor:
        """
        Calculate exponential decay of tillage effects over time
        
        Args:
            days_since: Days since last tillage operation
            
        Returns:
            Time decay factor [1]
        """
        # References: Lal (2004); Six et al. (2002).
        # Mathematical expression: f_t = exp(-k_decay * t_since).
        # Parameter meanings: k_decay is effect decay and t_since is elapsed time.
        # Units: k_decay [day-1]; t_since [day]; f_t [1].
        # Domain/constraints: inputs are finite, k_decay >= 0, and t_since >= 0.
        # Sign convention: f_t is positive and decreases as elapsed time increases.
        decay_rate_param = self.get_parameter_value(
            "effect_decay_rate",
            namespace="anthroposphere.management.tillage", required=True)
        decay_rate = self._ensure_tensor(decay_rate_param, dtype=days_since.dtype).to(days_since.device)
        return torch.exp(-decay_rate * days_since)
    
    def _calculate_porosity_modifier(self,
                                   intensity_factor: float,
                                   time_factor: torch.Tensor) -> torch.Tensor:
        """
        Calculate soil porosity modifier due to tillage
        
        Args:
            intensity_factor: Tillage intensity factor [1]
            time_factor: Time decay factor [1]
            
        Returns:
            Porosity modifier [1]
        """
        # References: Rawls et al. (1982); Lal (2004).
        # Mathematical expression: f_phi = 1 + a_phi * I * f_t.
        # Parameter meanings: a_phi is base change, I intensity, and f_t decay.
        # Units: a_phi [1]; I [1]; f_t [1]; f_phi [1].
        # Domain/constraints: all factors are finite and non-negative.
        # Sign convention: f_phi > 1 denotes increased porosity relative to baseline.
        base_change_param = self.get_parameter_value(
            "porosity_change",
            namespace="anthroposphere.management.tillage", required=True)
        base_change = self._ensure_tensor(base_change_param, dtype=time_factor.dtype).to(time_factor.device)
        
        # Immediate increase in porosity, then gradual decrease
        porosity_change = base_change * intensity_factor * time_factor
        
        # Modifier (1.0 = no change)
        modifier = 1.0 + porosity_change
        
        return modifier
    
    def _calculate_carbon_flush(self,
                                intensity_factor: float,
                                depth_factor: float,
                                soil_carbon: torch.Tensor,
                                temperature: torch.Tensor) -> torch.Tensor:
        """
        Calculate the tillage-induced soil carbon flush of an operation step
        
        Args:
            intensity_factor: Tillage intensity factor [1]
            depth_factor: Tillage depth factor [1]
            soil_carbon: Soil carbon pool [kg C/m^2]
            temperature: Soil temperature [K]
            
        Returns:
            Carbon flush [kg C m-2 s-1], released in the step of the operation
        """
        # References: Reicosky (1997) DOI:10.1023/A:1009766510274; Lal (2004);
        #   Davidson and Janssens (2006) DOI:10.1038/nature04514 and Lloyd and
        #   Taylor (1994) DOI:10.2307/2389824 for the Q10 of soil organic matter
        #   decomposition.
        # Mathematical expression: F_C = C f_0 f_p I D Q10^((T - T_ref) / 10 K) / S_day.
        # Parameter meanings: C is the soil organic carbon stock, f_0 and f_p the
        #   base and pulse fractions, I intensity, D depth, T the soil temperature
        #   of the tilled layer (the oxidation of the exposed organic matter follows
        #   the soil, not the air), T_ref the physical-constant reference
        #   temperature, Q10 the temperature coefficient of soil organic matter
        #   decomposition shared with the decomposition Functors
        #   (pedosphere.physical.q10_decomposition), 10 K the Q10 interval.
        # Units: C [kg C m-2]; f_0, f_p, I, D, Q10 [1]; T, T_ref [K];
        #   S_day [s day-1]; F_C [kg C m-2 s-1].
        # Domain/constraints: C >= 0, T > 0 K, Q10 within its YAML bounds, and all
        #   multipliers are finite and non-negative. F_C is the one tillage carbon flux
        #   and is emitted only in the step of an executed operation (the short-lived
        #   tillage-induced flush, Reicosky 1997); the enhanced decomposition in the
        #   weeks after tillage is a decomposition-rate multiplier of the decomposition
        #   Functors (CENTURY/DAYCENT cultivation factor, DNDC till_fact) and is not
        #   represented here.
        # Sign convention: positive F_C is carbon emitted outward from the soil pool.
        if intensity_factor == 0.0:
            return torch.zeros_like(soil_carbon)
        
        pulse_factor = self.get_parameter_value(
            "co2_pulse_factor",
            namespace="anthroposphere.management.tillage", required=True)
        pulse_factor_tensor = self._ensure_tensor(pulse_factor, dtype=soil_carbon.dtype).to(soil_carbon.device)
        
        # Temperature effect: the shared SOM decomposition Q10 from the YAML
        # parameter database and the physical-constant SSOT reference temperature.
        standard_temp = get_physical_constant("standard_temperature")
        standard_temp_tensor = self._ensure_tensor(standard_temp, dtype=temperature.dtype).to(temperature.device)
        q10 = self._ensure_tensor(
            self.get_parameter_value(
                "q10_decomposition",
                namespace="pedosphere.physical",
                units="dimensionless",
                required=True,
            ),
            dtype=temperature.dtype,
        ).to(temperature.device)
        temp_factor = q10 ** ((temperature - standard_temp_tensor) / 10.0)
        
        # Flush as a fraction of the soil carbon stock released within the step
        pulse_base_fraction = self.require_scalar(
            "co2_pulse_base_fraction",
            namespace="anthroposphere.management.tillage",
        )
        pulse_fraction = pulse_base_fraction * pulse_factor_tensor * intensity_factor * depth_factor * temp_factor

        # Fraction per day converted to a per-second carbon flux [kg C m-2 s-1]
        carbon_flush = soil_carbon * pulse_fraction / SECONDS_PER_DAY

        return carbon_flush
    
    def _check_tillage_schedule(self,
                              day_of_year: int,
                              soil_moisture: torch.Tensor,
                              days_since_tillage: torch.Tensor,
                              *,
                              calendar_type: str) -> Optional[TillageEvent]:
        """Check if a tillage operation should occur today.

        References:
            CF Conventions 1.8, Section 4.4; revised foundations, time-window
            and calendar contract.
        Mathematical expression:
            d_C(a,b) = min(|a-b|, Y(C)-|a-b|); active iff d_C <= W and
            t_since > T_min.
        Parameter meanings:
            a is current day [day], b is tillage day [day], Y(C) is calendar
            year length [day], W the schedule window and T_min the minimum
            interval between operations [day] (scheduling conventions).

        Args:
            day_of_year: Current day of the configured simulator year
            soil_moisture: Current soil moisture [m^3/m^3]
            days_since_tillage: Days since last tillage operation
            calendar_type: Existing simulator calendar identifier

        Returns:
            TillageEvent if operation should occur, None otherwise
        """
        # References: CF Conventions 1.8; revised RELIC calendar contract.
        # Mathematical expression: d_C = min(|a-b|, Y(C)-|a-b|); active iff d_C <= W,
        #   mean(theta) <= theta_limit and mean(t_since) > T_min.
        # Parameter meanings: a is current day, b event day, Y(C) year length, W the
        #   schedule window (schedule_window_days) and T_min the minimum interval
        #   between operations (minimum_operation_interval_days); both are RELIC
        #   scheduling conventions without a literature value.
        # Units: a, b, d_C, Y(C), W, T_min and elapsed interval [day]; soil moisture
        #   [m3 m-3].
        # Domain/constraints: calendar days are valid; moisture is finite; W and T_min
        #   lie within their YAML bounds.
        # Sign convention: day distances and elapsed time are non-negative; no flux sign is assigned.
        day_of_year = CalendarHandler.validate_simulator_day(
            day_of_year,
            calendar_type,
        )
        days_per_year = CalendarHandler.days_per_simulator_year(calendar_type)
        window_days = self.require_scalar(
            "schedule_window_days",
            namespace="anthroposphere.management.tillage",
        )
        minimum_interval_days = self.require_scalar(
            "minimum_operation_interval_days",
            namespace="anthroposphere.management.tillage",
        )
        for event in self.tillage_events:
            event_day = self._validate_tillage_event_date(
                event.date,
                calendar_type,
            )
            direct_distance = abs(day_of_year - event_day)
            event_distance = min(
                direct_distance,
                days_per_year - direct_distance,
            )
            if event_distance <= window_days:
                # Check soil moisture constraint
                if soil_moisture.mean() <= event.soil_moisture_limit:
                    # Check minimum interval (avoid frequent operations)
                    if days_since_tillage.mean() > minimum_interval_days:
                        return event

        return None
    
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        time: float,
        dt: float,
        calendar_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate tillage effects on soil properties.

        References:
            Lal (2004); Six et al. (2002); Reicosky (1997); CF-1.8.
        Mathematical expression:
            f_time = exp(-k_decay t_since),
            F_disturbance = k_disturbance I f_theta, and event day
            d = floor(t / seconds_per_day) mod Y(C) + 1.
        Parameter meanings:
            t_since is time since tillage [day], I is intensity [1], f_theta
            is the moisture response [1], and C is the explicit simulator
            calendar.
        """
        # References: Lal (2004); Six et al. (2002); Reicosky (1997); CF-1.8.
        # Mathematical expression: f_time = exp(-k_decay * t_since).
        # Parameter meanings: t_since is [day], I and f_theta are [1], and dt is [s].
        # Units: theta [m3 m-3]; T [K]; C [kg C m-2]; dt [s]; carbon flux [kg C m-2 s-1].
        # Domain/constraints: state tensors are finite and in declared ranges; dt > 0; calendar is valid.
        # Sign convention: positive carbon flux leaves soil; negative residue-cover change removes cover.
        # soil_moisture [m3 m-3], soil_temperature [K] and the soil organic
        # carbon stock soil_organic_carbon [kg C m-2] of the tilled layer are
        # required inputs with no ParameterManager fallback. Each is checked
        # against its physical domain (volume fraction in [0, 1]; non-negative
        # absolute temperature and carbon stock, the contract bounds) and a value
        # outside it raises; nothing is clipped.
        soil_moisture = self._get_state_variable(
            state,
            "soil_moisture",
            units="m3 m-3",
            validate_range=(0.0, 1.0)
        )
        # soil_temperature [K] of the tilled layer is a required input with no
        # ParameterManager fallback: a missing value raises.
        if state.get("soil_temperature") is None:
            raise KeyError(
                "TillageManagement needs 'soil_temperature' [K] of the tilled layer "
                "(forcing or initial state); no default is applied."
            )
        temperature = self._get_state_variable(
            state,
            "soil_temperature",
            units="K",
            validate_range=(0.0, float("inf"))
        )
        soil_carbon = self._get_state_variable(
            state,
            "soil_organic_carbon",
            units="kg C m-2",
            validate_range=(0.0, float("inf"))
        )
        calendar_type = CalendarHandler.resolve_simulator_calendar(
            configured_calendar=self._configured_calendar_type,
            runtime_calendar=calendar_type,
            owner_name=self.__class__.__name__,
        )
        state_day = state.get("day_of_year")
        runtime_day = kwargs.get("day_of_year")
        if state_day is not None and runtime_day is not None:
            state_day_value = CalendarHandler.validate_simulator_day(
                state_day,
                calendar_type,
            )
            runtime_day_value = CalendarHandler.validate_simulator_day(
                runtime_day,
                calendar_type,
            )
            if state_day_value != runtime_day_value:
                raise ValueError(
                    "Conflicting day_of_year inputs from state and runtime context."
                )
            day_input: Any = state_day_value
        else:
            day_input = state_day if state_day is not None else runtime_day
        day_of_year = resolve_day_of_year(
            time,
            day_input,
            calendar_type=calendar_type,
        )

        # days_since_tillage [day] is the tillage state itself (tillage_management is
        # its only writer) and a declared input with no ParameterManager fallback:
        # the caller supplies its initial value, and a missing value raises, as in
        # the governed call.
        days_since_tillage = self._get_state_variable(
            state,
            "days_since_tillage",
            units="days",
            validate_range=(0.0, float("inf"))
        )

        # Update days since last tillage
        days_since_tillage = days_since_tillage + dt / SECONDS_PER_DAY  # Convert seconds to days

        # Check for scheduled tillage operations
        tillage_event = self._check_tillage_schedule(
            day_of_year,
            soil_moisture,
            days_since_tillage,
            calendar_type=calendar_type,
        )

        # The tillage carbon flush is nonzero only in the step of an executed operation.
        carbon_flux = torch.zeros_like(soil_carbon)

        # Apply tillage event if scheduled
        if tillage_event is not None:
            # Reset tracking
            days_since_tillage = torch.zeros_like(days_since_tillage)  # Reset to 0 days since tillage
            self.current_intensity = tillage_event.intensity
            self.current_depth = tillage_event.depth
            self.cumulative_tillage_operations += 1

            # Tillage-induced carbon flush of the executing operation
            intensity_factor = self._get_tillage_intensity_factor(tillage_event.intensity)
            depth_factor = self._get_tillage_depth_factor(tillage_event.depth)

            carbon_flux = self._calculate_carbon_flush(
                intensity_factor, depth_factor, soil_carbon, temperature
            )
        else:
            # Use current intensity and depth if no new tillage
            intensity_factor = self._get_tillage_intensity_factor(self.current_intensity)
            depth_factor = self._get_tillage_depth_factor(self.current_depth)

        # Calculate current tillage effects (decay over time)
        time_factor = self._calculate_time_decay_factor(days_since_tillage)

        # Calculate soil property modifiers
        self.porosity_modifier = self._calculate_porosity_modifier(
            intensity_factor, time_factor
        )

        # Calculate surface roughness modifier
        roughness_intensity_coefficient = self.require_scalar(
            "roughness_intensity_coefficient",
            namespace="anthroposphere.management.tillage",
        )
        self.roughness_modifier = 1.0 + roughness_intensity_coefficient * intensity_factor * time_factor

        # The responses follow the executed operations: the event responses and the
        # carbon flush belong to the operation executing in this step. Every input
        # comes from the state; runtime keywords carry no input variables.
        rate_outputs = self._compute_rate_core(
            state,
            event_intensity=intensity_factor if tillage_event is not None else None,
            carbon_flux=carbon_flux,
        )
        soil_disturbance_rate = rate_outputs["soil_disturbance_rate"]
        residue_incorporation_rate = rate_outputs["residue_incorporation_rate"]
        carbon_loss_rate = rate_outputs["carbon_loss_rate"]
        soil_co2_flux = rate_outputs["soil_co2_flux"]
        residue_cover_increment = rate_outputs["delta_residue_cover"]
        dt_tensor = torch.as_tensor(
            dt,
            dtype=residue_cover_increment.dtype,
            device=residue_cover_increment.device,
        )
        if dt_tensor.numel() != 1 or not torch.all(torch.isfinite(dt_tensor)):
            raise ValueError("Tillage dt must be one finite scalar.")
        if torch.any(dt_tensor <= 0.0):
            raise ValueError("Tillage dt must be positive.")
        residue_cover_rate = residue_cover_increment / dt_tensor
        mass_balance_error = self._check_conservation(
            soil_carbon=soil_carbon,
            carbon_flux=carbon_loss_rate,
            dt=dt,
        )

        def _replace(
            value: torch.Tensor,
            unit: str,
            *,
            element: Optional[str] = None,
        ) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit=unit,
                element=element,
                source_functor=self.name,
            )

        return {
            "porosity_modifier": _replace(self.porosity_modifier, "1"),
            "roughness_modifier": _replace(self.roughness_modifier, "1"),
            "days_since_tillage": _replace(days_since_tillage, "day"),
            "tillage_intensity_factor": _replace(
                torch.as_tensor(
                    intensity_factor,
                    dtype=soil_carbon.dtype,
                    device=soil_carbon.device,
                ),
                "1",
            ),
            "time_decay_factor": _replace(time_factor, "1"),
            "cumulative_operations": _replace(
                torch.as_tensor(
                    self.cumulative_tillage_operations,
                    dtype=soil_carbon.dtype,
                    device=soil_carbon.device,
                ),
                "1",
            ),
            "soil_disturbance_rate": _replace(soil_disturbance_rate, "1"),
            "residue_incorporation_rate": _replace(
                residue_incorporation_rate,
                "1",
            ),
            "carbon_loss_rate": _replace(
                carbon_loss_rate,
                "kg C m-2 s-1",
                element="carbon",
            ),
            "soil_co2_flux": _replace(
                soil_co2_flux,
                "kg CO2 m-2 s-1",
            ),
            "delta_residue_cover": UpdateItem(
                value=residue_cover_rate,
                mode=UpdateMode.ACCUMULATE,
                per_timestep=False,
                unit="s-1",
                source_functor=self.name,
            ),
            "diagnostics.tillage_mass_balance_error": _replace(
                mass_balance_error,
                "kg C m-2",
                element="carbon",
            ),
        }

    def _check_conservation(
        self,
        *,
        soil_carbon: torch.Tensor,
        carbon_flux: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Check carbon mass availability for tillage-induced losses.

        References:
            - Lal (2004) DOI:10.1126/science.1097396
        Mathematical expression:
            epsilon = max(0, F_C * dt - C_soil - tol)
        Parameter meanings:
            F_C: tillage carbon flush, the one tillage carbon flux [kg C m-2 s-1]
            C_soil: available soil carbon pool [kg C m-2]
            tol: conservation tolerance [kg C m-2]
        """
        # References: Lal (2004) DOI:10.1126/science.1097396.
        # Mathematical expression: R_C = max(0, F_C dt - C_soil - tol).
        # Parameter meanings: F_C the tillage carbon flush (counted once), C_soil
        #   available carbon, and tol tolerance.
        # Units: F_C [kg C m-2 s-1]; dt [s]; C_soil, tol, and R_C [kg C m-2].
        # Domain/constraints: dt > 0; inputs are finite; available carbon and losses are clamped >= 0.
        # Sign convention: R_C > 0 is excess requested removal; R_C = 0 is available-carbon closure.
        dt_tensor = torch.as_tensor(dt, dtype=soil_carbon.dtype, device=soil_carbon.device)
        tolerance = self._ensure_tensor(
            self.get_parameter_value(
                "conservation_tolerance",
                namespace="validation.conservation", required=True),
            dtype=soil_carbon.dtype,
        ).to(soil_carbon.device)
        available_carbon = torch.clamp(soil_carbon, min=0.0)
        carbon_loss = torch.clamp(carbon_flux, min=0.0)
        removed_carbon = carbon_loss * dt_tensor
        return torch.relu(removed_carbon - available_carbon - tolerance)

    def check_conservation_laws(
        self,
        input_state: Dict[str, torch.Tensor],
        output_updates: Dict[str, Any],
    ) -> Dict[str, float]:
        """Return the executable tillage carbon-availability residual.

        References:
            Lal (2004) DOI:10.1126/science.1097396.
        Mathematical expression:
            epsilon_C = max(0, F_C dt - C_soil - tolerance).
        Parameter meanings:
            F_C is the tillage carbon flush [kg C m-2 s-1], dt is the
            accepted step [s], and C_soil is available carbon [kg C m-2].
        """
        # References: Lal (2004); RELIC local conservation hook contract.
        # Mathematical expression: mass_error = max(abs(R_C)).
        # Parameter meanings: R_C is the emitted tillage carbon availability residual.
        # Units: R_C and mass_error [kg C m-2].
        # Domain/constraints: the diagnostic exists and is finite and non-empty.
        # Sign convention: mass_error is a non-negative magnitude; direction remains in R_C.
        del input_state
        residual_key = "diagnostics.tillage_mass_balance_error"
        if residual_key not in output_updates:
            raise KeyError(f"Missing conservation diagnostic '{residual_key}'")
        residual = output_updates[residual_key]
        if isinstance(residual, UpdateItem):
            residual = residual.value
        residual_tensor = torch.as_tensor(residual)
        if residual_tensor.numel() == 0 or not torch.all(
            torch.isfinite(residual_tensor)
        ):
            raise ValueError("Tillage conservation residual must be finite and non-empty.")
        magnitude = torch.max(torch.abs(residual_tensor))
        return {"mass": float(magnitude.detach().item())}
    
    def get_tillage_statistics(self, days_since_tillage: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Get tillage management statistics

        Args:
            days_since_tillage: Current days since last tillage (optional)

        Returns:
            Dictionary with tillage metrics
        """
        # Use detached values for tracking (non-differentiable statistics)
        with torch.no_grad():
            stats = {
                "cumulative_operations": self.cumulative_tillage_operations,
                "current_porosity_modifier": self._scalar_from_value(self.porosity_modifier),
                "operations_scheduled": len(self.tillage_events)
            }

            if days_since_tillage is not None:
                stats["days_since_last_tillage"] = self._scalar_from_value(days_since_tillage)

            return stats

    def _compute_rate_core(
        self,
        state: Dict[str, torch.Tensor],
        *,
        event_intensity: Optional[float],
        carbon_flux: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the tillage responses of one step from the executed operations.

        event_intensity is the intensity factor of the operation executing in this
        step (None: no operation); carbon_flux is the tillage carbon flush of the
        step (_calculate_carbon_flush in an operation step, zero otherwise).
        """
        # References: Lal (2004); Six et al. (2002); Reicosky (1997) short-lived
        #   post-tillage CO2 flux; molar masses from the physical-constant SSOT
        #   (standard atomic weights, NIST Chemistry WebBook).
        # Mathematical expression: with an operation of intensity I_e in the step,
        #   D = k_D I_e theta/theta_den, R = clamp(k_R I_e (a + g_theta), 0, 1),
        #   delta_cover = -min(cover, R); without one D = R = delta_cover = 0.
        #   F_C is the operation-step carbon flush (zero without an operation) and
        #   F_CO2 = F_C M_CO2 / M_C its CO2-mass form.
        # Parameter meanings: I_e the intensity class factor of the executing operation;
        #   theta/theta_den the linear moisture response and g_theta the Gaussian
        #   moisture response; a the residue moisture offset; k_D, k_R coefficients;
        #   cover the residue cover; M_CO2, M_C molar masses of CO2 and carbon (one CO2
        #   per oxidised C).
        # Units: D, R, I_e, cover [1]; theta, theta_den [m3 m-3]; F_C [kg C m-2 s-1];
        #   F_CO2 [kg CO2 m-2 s-1]; M_CO2, M_C [g mol-1].
        # Domain/constraints: theta in [0, 1] and cover in [0, 1] (contract bounds; a
        #   value outside raises); cover is read only when an operation executes and a
        #   missing value then raises; sigma > 0.
        # Sign convention: F_C and CO2 are outward-positive; delta_cover <= 0 removes cover.
        soil_moisture = self._get_state_variable(
            state,
            "soil_moisture",
            units="m3 m-3",
            validate_range=(0.0, 1.0)
        )
        zeros = torch.zeros_like(soil_moisture)

        def _param(name: str, *, units: Optional[str] = None) -> torch.Tensor:
            param_val = self.get_parameter_value(
                name,
                namespace="anthroposphere.management.tillage",
                units=units,
                required=True,
            )
            return self._ensure_tensor(param_val, dtype=soil_moisture.dtype).to(soil_moisture.device)

        carbon_loss_rate = self._ensure_tensor(
            carbon_flux, dtype=soil_moisture.dtype
        ).to(soil_moisture.device).expand_as(soil_moisture)
        molar_mass_co2 = get_physical_constant("relic.physical.constants.molecular_weights.co2")
        molar_mass_carbon = get_physical_constant("relic.physical.constants.molecular_weights.carbon")
        soil_co2_flux = carbon_loss_rate * (molar_mass_co2 / molar_mass_carbon)

        if event_intensity is None:
            # No operation in this step: the explicit no-tillage case.
            soil_disturbance_rate = zeros
            residue_incorporation_rate = zeros
            residue_cover_change = zeros
        else:
            intensity = torch.full_like(soil_moisture, float(event_intensity))
            disturbance_factor = _param("disturbance_factor")
            residue_incorporation = _param("residue_incorporation_rate")
            moisture_denominator = _param("moisture_effect_denominator")
            moisture_optimum = _param("moisture_optimum")
            moisture_sigma = _param("moisture_sigma")
            residue_moisture_offset = self.require_scalar(
                "residue_moisture_offset",
                namespace="anthroposphere.management.tillage",
            )
            moisture_effect = soil_moisture / moisture_denominator
            moisture_response = torch.exp(
                -((soil_moisture - moisture_optimum) ** 2) / (2 * moisture_sigma ** 2)
            )
            soil_disturbance_rate = disturbance_factor * intensity * moisture_effect
            residue_incorporation_rate = torch.clamp(
                residue_incorporation * intensity * (residue_moisture_offset + moisture_response),
                0.0,
                1.0,
            )
            if state.get("residue_cover") is None:
                raise ValueError(
                    f"{self.name} needs 'residue_cover' [1] when a tillage operation "
                    "executes (the incorporated cover); no default is applied."
                )
            residue_cover = self._get_state_variable(
                state,
                "residue_cover",
                units="1",
                validate_range=(0.0, 1.0),
                dtype=soil_moisture.dtype,
            ).to(soil_moisture.device)
            residue_cover_change = -torch.minimum(residue_cover, residue_incorporation_rate)

        return {
            "soil_disturbance_rate": soil_disturbance_rate,
            "residue_incorporation_rate": residue_incorporation_rate,
            "delta_residue_cover": residue_cover_change,
            "carbon_loss_rate": carbon_loss_rate,
            "soil_co2_flux": soil_co2_flux,
        }
