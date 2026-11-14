"""
Concrete Phenology Models for RELIC Framework

This module provides concrete implementations of phenological models that inherit
from the PhenologyBase abstract class. Each model implements a different approach
to phenological timing based on environmental drivers and scientific literature.

Models Implemented:
    - StaticPhenology: Fixed calendar date phenology
    - GDDPhenology: Growing degree day accumulation model
    - GSIPhenology: Growing season index model (Jolly et al. 2005)
    - CropPhenology: Crop-specific phenological development

All models maintain consistency with the RELIC core architecture and provide
automatic differentiation support through PyTorch integration.

References:
    - Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict 
      foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.
    - White, M.A., et al. (1997). A continental phenology model for monitoring 
      vegetation responses to interannual climatic variability.
    - McMaster, G.S. & Wilhelm, W.W. (1997). Growing degree-days: one equation, 
      two interpretations. Agricultural and Forest Meteorology, 87(4), 291-300.
"""

import logging
import math
from typing import Dict, Any, Optional

import torch
from torch import Tensor

# Import base classes and utilities
from .base import PhenologyBase, PhenologyPhase, PhenologyOutput
from .parameters import PhenologyParameters
from .utils import compute_gdd, compute_gsi

from relic.core.functor import FunctorMetadata, ProcessCategory, ExecutionPhase
from relic.computation.solvers.hybrid_events import (
    event_time_sensitivity,
    locate_guard_crossing,
    sensitivity_jump,
)
from relic.core.abstractions import StateDict
from relic.utils.physical_constants import get_physical_constant
from relic.utils.time_constants import SECONDS_PER_DAY
from relic.core.functor import SpatialScale, VariableKind

T_FREEZE = get_physical_constant("T_FREEZE")

logger = logging.getLogger(__name__)


def _scalar_value(value: Any, *, name: str = "value") -> float:
    """Convert one finite scalar-like value to float without a fallback."""
    if isinstance(value, Tensor):
        detached = value.detach()
        if detached.numel() != 1:
            raise ValueError(f"{name} must contain exactly one scalar value")
        result = float(detached.reshape(-1)[0].tolist())
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be scalar-like") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _int_value(value: Any, *, name: str = "value") -> int:
    """Convert one finite scalar-like value to an integer without a fallback."""
    return int(round(_scalar_value(value, name=name)))


def _require_input(state: StateDict, key: str) -> Any:
    """Return one declared phenology input from the state mapping (no fallback)."""
    if key not in state:
        raise ValueError(f"Missing required phenology input: {key}")
    return state[key]


def _require_input_tensor(
    state: StateDict,
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return one required finite declared phenology input as a tensor."""
    raw = _require_input(state, key)
    try:
        value = torch.as_tensor(raw, device=device, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"Phenology input '{key}' must be tensor-like") from exc
    if not torch.all(torch.isfinite(value)):
        raise ValueError(f"Phenology input '{key}' must be finite")
    return value


def _step_length_tensor(dt: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return the simulator step length dt [s] as a finite tensor of the model precision."""
    try:
        value = torch.as_tensor(dt, device=device, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("dt must be a scalar step length in seconds") from exc
    if not torch.all(torch.isfinite(value)):
        raise ValueError("dt must be finite")
    return value


class StaticPhenology(PhenologyBase):
    """
    Static Date-Based Phenology Model
    
    This model uses fixed calendar dates to determine phenological timing.
    It is the simplest phenology approach and is useful for regions with
    very predictable seasonal patterns or for baseline comparisons.
    
    The model interpolates LAI and growth activity between onset and offset
    dates using smooth transitions to avoid discontinuities in carbon
    allocation and biomass processes.
    
    Mathematical Expression:
        Phenological factor: f(t) = 0.5 * (1 + tanh(k * (t - t_onset)))
        LAI dynamics: LAI(t) = LAI_max * f_growth(t) * f_senescence(t)
        
        Where:
        - t: day of year
        - t_onset, t_offset: phenological transition dates  
        - k: transition steepness parameter
        - f_growth, f_senescence: growth and senescence factors
    
    Advantages:
        - Simple and fast computation
        - Predictable behavior
        - Useful for model validation
        - Minimal parameter requirements
        
    Limitations:
        - No climate sensitivity
        - Fixed timing regardless of weather
        - Limited realism for climate change studies
    
    References:
        - White, M.A., et al. (1997). A continental phenology model
        - Running, S.W. & Coughlan, J.C. (1988). A general model of forest ecosystem processes
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    PHENOLOGY_TYPE = "static"
    
    def __init__(
        self,
        name: str = "static_phenology",
        config: Optional[Dict[str, Any]] = None,
        phenology_parameters: Optional[PhenologyParameters] = None,
        **kwargs
    ) -> None:
        """
        Initialize static phenology model.
        
        Parameters:
            name : str
                Model identifier
            config : Optional[Dict[str, Any]]
                Configuration parameters
            phenology_parameters : Optional[PhenologyParameters]
                Phenology parameter container
            **kwargs
                Additional initialization parameters
        """
        if config is None:
            config = {}

        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Static date-based phenology model with fixed onset/offset timing",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=r"""
            Static phenology equations:
            
            Growth factor: f_{growth}(t) = 
            \begin{cases}
                0 & \text{if } t < t_{onset} \\
                0.5(1 + \tanh(k(t - t_{onset}))) & \text{if } t_{onset} \leq t < t_{full} \\
                1 & \text{if } t_{full} \leq t < t_{senes} \\
                0.5(1 + \tanh(k(t_{offset} - t))) & \text{if } t_{senes} \leq t < t_{offset} \\
                0 & \text{if } t \geq t_{offset}
            \end{cases}
            
            LAI(t) = LAI_{max} \times f_{growth}(t)
            
            Where:
            - t: day of year [days]
            - t_{onset}: leaf onset date [DOY]
            - t_{offset}: leaf offset date [DOY]  
            - t_{full} = t_{onset} + expansion_{days}
            - t_{senes} = t_{offset} - senescence_{days}
            - k = transition_steepness: transition steepness [day-1]
            """,
            references=[
                "White, M.A., et al. (1997). A continental phenology model for monitoring vegetation responses to interannual climatic variability. Global Biogeochemical Cycles, 11(2), 217-234.",
                "Running, S.W. & Coughlan, J.C. (1988). A general model of forest ecosystem processes for regional applications. Ecological Modelling, 42(2), 125-154."
            ],
            coupling_interfaces=["photosynthesis", "allocation", "litterfall"],
            parameters={
                "biosphere.phenology.static.transition_steepness": (
                    "Steepness of the tanh phase transitions [day-1]"
                ),
                "biosphere.phenology.static.sigmoid_midpoint": (
                    "Midpoint of the senescence sigmoid [dimensionless]"
                ),
                "biosphere.phenology.static.sigmoid_steepness": (
                    "Steepness of the senescence sigmoid [dimensionless]"
                ),
                "biosphere.phenology.computational.sigmoid_scale": (
                    "Scale of the tanh-to-sigmoid map [dimensionless]"
                ),
                "biosphere.phenology.computational.clamp_min": (
                    "Lower bound of the phenological factors [dimensionless]"
                ),
                "biosphere.phenology.computational.clamp_max": (
                    "Upper bound of the phenological factors [dimensionless]"
                ),
                "biosphere.phenology.turnover.background_senescence_rate": (
                    "Background senescence rate outside senescence [dimensionless]"
                ),
            },
            input_variables=[
                "day_of_year",
                "air_temperature",
            ],
            output_variables=[
                "leaf_area_index",
                "phenological_factor",
                "phenology_senescence_factor",
                "phenology_phase",
            ],
            conservation_laws=["none"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        
        # Initialize base class
        super().__init__(
            name=name,
            config=config,
            phenology_parameters=phenology_parameters,
            metadata=metadata,
            **kwargs
        )
        
        # Model-specific parameters sourced via ParameterManager
        self.transition_steepness = self.get_parameter_value(
            "biosphere.phenology.static.transition_steepness",
        )
        self.sigmoid_scale = self.get_parameter_value(
            "biosphere.phenology.computational.sigmoid_scale",
        )
        
        logger.info(f"Initialized StaticPhenology model '{name}' with onset DOY {self.phenology_params.leaf_onset_doy}")
    
    def compute_phenological_state(
        self,
        state: StateDict,
        *,
        dt: float,
    ) -> PhenologyOutput:
        """
        Compute phenological state based on calendar dates.

        Parameters:
            state : StateDict
                Declared inputs; the static law reads day_of_year (1-366) only
            dt : float
                Step length of the simulator call [s]; the one time control
                (not used by the calendar law)

        Returns:
            PhenologyOutput: Phenological state with calendar-based timing
        """
        day_of_year = _require_input(state, "day_of_year")
        # Ensure tensor inputs use device precision
        day_of_year_tensor = torch.as_tensor(
            day_of_year,
            device=self.device,
            dtype=torch.float32,
        )

        # StaticPhenology supports vectorized day_of_year inputs for batch runs.
        # Phase diagnostics remain scalar; use the first element in the batch.
        day_scalar = _scalar_value(day_of_year_tensor, name="day_of_year")

        # Extract calendar dates from parameters
        onset_doy = self.phenology_params.leaf_onset_doy
        offset_doy = self.phenology_params.leaf_offset_doy
        expansion_days = self.phenology_params.leaf_expansion_days
        senescence_days = self.phenology_params.leaf_senescence_days
        
        # Calculate key phenological dates
        full_leaf_doy = onset_doy + expansion_days
        senescence_start_doy = offset_doy - senescence_days
        
        # Determine phenological phase
        if day_scalar < onset_doy:
            phase = PhenologyPhase.DORMANCY
        elif day_scalar < full_leaf_doy:
            phase = PhenologyPhase.LEAF_EXPANSION
        elif day_scalar < senescence_start_doy:
            phase = PhenologyPhase.FULL_LEAF
        elif day_scalar < offset_doy:
            phase = PhenologyPhase.SENESCENCE
        else:
            phase = PhenologyPhase.ABSCISSION
        
        # Compute smooth growth factor using sigmoid transitions
        growth_factor = self._compute_smooth_growth_factor(
            day_of_year_tensor,
            onset_doy,
            offset_doy,
            expansion_days,
            senescence_days,
        )
        
        # Compute LAI based on growth factor
        lai = self.phenology_params.max_lai * growth_factor
        lai = torch.clamp(lai, min=self.phenology_params.min_lai, max=self.phenology_params.max_lai)
        
        # Compute senescence rate (higher during leaf drop period)
        # Get senescence parameters from parameter manager
        sigmoid_steepness = self.get_parameter_value("biosphere.phenology.static.sigmoid_steepness")
        sigmoid_midpoint = self.get_parameter_value("biosphere.phenology.static.sigmoid_midpoint")
        background_senescence = self.get_parameter_value("biosphere.phenology.turnover.background_senescence_rate")
            
        doy_tensor = day_of_year_tensor
        within_senescence = (doy_tensor >= senescence_start_doy) & (doy_tensor <= offset_doy)
        senescence_progress = (doy_tensor - senescence_start_doy) / senescence_days
        senescence_progress = torch.clamp(senescence_progress, min=0.0, max=1.0)

        steepness_tensor = torch.as_tensor(
            sigmoid_steepness, device=self.device, dtype=torch.float32
        )
        midpoint_tensor = torch.as_tensor(
            sigmoid_midpoint, device=self.device, dtype=torch.float32
        )
        background_tensor = torch.as_tensor(
            background_senescence, device=self.device, dtype=torch.float32
        )

        senescence_rate_window = torch.sigmoid(
            steepness_tensor * (senescence_progress - midpoint_tensor)
        )
        senescence_rate = torch.where(within_senescence, senescence_rate_window, background_tensor)
        
        # Prepare output
        output = PhenologyOutput(
            phenology_phase=phase,
            growing_season_index=None,  # Not applicable for static model
            thermal_time=None,          # Not applicable for static model
            leaf_area_index=lai,
            growth_factor=growth_factor,
            senescence_rate=senescence_rate,
            diagnostic_variables={
                'day_of_year': day_of_year_tensor,
                'onset_doy': torch.as_tensor(onset_doy, device=self.device, dtype=torch.float32),
                'offset_doy': torch.as_tensor(offset_doy, device=self.device, dtype=torch.float32),
                'full_leaf_doy': torch.as_tensor(full_leaf_doy, device=self.device, dtype=torch.float32),
                'senescence_start_doy': torch.as_tensor(senescence_start_doy, device=self.device, dtype=torch.float32),
                'phenology_phase_enum': phase  # Keep enum for debugging
            }
        )
        
        # Perform conservation checks
        if self.enforce_physical_constraints:
            self._validate_output_bounds(output)
        
        return output
    
    def _compute_smooth_growth_factor(
        self,
        doy: Tensor,
        onset: float,
        offset: float,
        expansion_days: float,
        senescence_days: float
    ) -> Tensor:
        """
        Compute smooth growth factor using sigmoid transitions.
        
        Parameters:
            doy : Tensor
                Day of year
            onset : float
                Onset day of year
            offset : float  
                Offset day of year
            expansion_days : float
                Duration of leaf expansion
            senescence_days : float
                Duration of senescence
                
        Returns:
            Tensor: Smooth growth factor (0-1)
        """
        onset + expansion_days
        offset - senescence_days
        doy = torch.as_tensor(doy, device=self.device, dtype=torch.float32)
        onset_tensor = torch.as_tensor(onset, device=doy.device, dtype=doy.dtype)
        offset_tensor = torch.as_tensor(offset, device=doy.device, dtype=doy.dtype)
        k = torch.as_tensor(self.transition_steepness, device=doy.device, dtype=doy.dtype)
            
        # Growth phase (sigmoid transition from 0 to 1)
        # Using standard sigmoid transformation: 0.5 * (1 + tanh(x)) = sigmoid(2x)
        sigmoid_scale = torch.as_tensor(self.sigmoid_scale, device=doy.device, dtype=doy.dtype)
            
        growth_factor = sigmoid_scale * (1.0 + torch.tanh(k * (doy - onset_tensor)))
        
        # Senescence phase (sigmoid transition from 1 to 0)
        senescence_factor = sigmoid_scale * (1.0 + torch.tanh(k * (offset_tensor - doy)))
        
        # Combine growth and senescence factors
        combined_factor = growth_factor * senescence_factor
        
        # Ensure values are within computational bounds
        clamp_min = torch.as_tensor(
            self.get_parameter_value("biosphere.phenology.computational.clamp_min"),
            device=doy.device,
            dtype=doy.dtype,
        )
        clamp_max = torch.as_tensor(
            self.get_parameter_value("biosphere.phenology.computational.clamp_max"),
            device=doy.device,
            dtype=doy.dtype,
        )
        return torch.minimum(torch.maximum(combined_factor, clamp_min), clamp_max)


class GDDPhenology(PhenologyBase):
    """
    Growing Degree Days (GDD) Phenology Model
    
    This model accumulates thermal time (growing degree days) to determine
    phenological timing. It represents the most widely used temperature-based
    phenology approach in ecosystem and crop models.
    
    The model tracks cumulative GDD from a base temperature and triggers
    phenological events when specific thresholds are reached. It supports
    multiple GDD calculation methods and handles both annual and perennial
    vegetation types.
    
    Mathematical Expression:
        Daily GDD: GDD_i = max(0, T_avg_i - T_base)
        Cumulative GDD: GDD_cum = Sigma GDD_i
        
        Phenological transitions:
        - Onset: GDD_cum > GDD_threshold_start
        - Full leaf: GDD_cum > GDD_threshold_full
        - Senescence: GDD_cum > GDD_threshold_end OR photoperiod trigger
        
        LAI dynamics: LAI(t) = LAI_max * f(GDD_cum, GDD_thresholds)
    
    Advantages:
        - Temperature-responsive phenology
        - Well-established methodology
        - Extensive parameter databases available
        - Suitable for climate change studies
        
    Applications:
        - Crop phenology modeling
        - Forest phenology in temperature-limited regions
        - Climate change impact assessment
    
    References:
        - McMaster, G.S. & Wilhelm, W.W. (1997). Growing degree-days: one equation, 
          two interpretations. Agricultural and Forest Meteorology, 87(4), 291-300.
        - Miller, P., et al. (2001). Using growing degree-days to predict plant stages.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    PHENOLOGY_TYPE = "gdd"
    OUTPUT_UNITS = {
        **PhenologyBase.OUTPUT_UNITS,
        "phenology_gdd_accumulated": "degC day",
        "phenology_gdd_previous_day_of_year": "day",
        "diagnostics.phenology.accumulated_gdd": "degC day",
        "diagnostics.phenology.onset_event_flag": "dimensionless",
        "diagnostics.phenology.onset_event_time_offset": "s",
        "diagnostics.phenology.onset_event_time_sensitivity_gdd_start": "s degC-1 day-1",
        "diagnostics.phenology.lai_sensitivity_jump_gdd_start": "m2 m-2 degC-1 day-1",
    }
    
    def __init__(
        self,
        name: str = "gdd_phenology",
        config: Optional[Dict[str, Any]] = None,
        phenology_parameters: Optional[PhenologyParameters] = None,
        **kwargs
    ) -> None:
        """Initialize GDD phenology model."""
        if config is None:
            config = {}

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Growing degree days (GDD) based phenology model with thermal time accumulation",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=r"""
            GDD phenology equations:
            
            Daily GDD: GDD_{daily} = \max(0, \frac{T_{max} + T_{min}}{2} - T_{base})
            
            Cumulative GDD: GDD_{cum} = \sum_{i=1}^{t} GDD_{daily,i}
            
            Growth factor:
            f_{growth}(GDD_{cum}) = 
            \begin{cases}
                0 & \text{if } GDD_{cum} < GDD_{start} \\
                \frac{GDD_{cum} - GDD_{start}}{GDD_{full} - GDD_{start}} & \text{if } GDD_{start} \leq GDD_{cum} < GDD_{full} \\
                1 & \text{if } GDD_{full} \leq GDD_{cum} < GDD_{end} \\
                1 - \frac{GDD_{cum} - GDD_{end}}{GDD_{senes} - GDD_{end}} & \text{if } GDD_{end} \leq GDD_{cum} < GDD_{senes} \\
                0 & \text{if } GDD_{cum} \geq GDD_{senes}
            \end{cases}
            
            LAI(t) = LAI_{max} \times f_{growth}(GDD_{cum}(t))
            """,
            references=[
                "McMaster, G.S. & Wilhelm, W.W. (1997). Growing degree-days: one equation, two interpretations. Agricultural and Forest Meteorology, 87(4), 291-300.",
                "Miller, P., et al. (2001). Using growing degree-days to predict plant stages. Montana State University Extension Service."
            ],
            parameters={
                "biosphere.phenology.gdd.expansion_phase_gdd": (
                    "Growing degree days of leaf expansion [degC_day]"
                ),
                "biosphere.phenology.gdd.full_growth_offset": (
                    "Growing degree days from onset to full growth [degC_day]"
                ),
                "biosphere.phenology.gdd.reset_day_threshold": (
                    "Day of year that resets the growing degree days [day]"
                ),
                "biosphere.phenology.gdd.senescence_buffer_gdd": (
                    "Growing degree days of the buffer before senescence [degC_day]"
                ),
                "biosphere.phenology.gdd.senescence_start_offset": (
                    "Growing degree days before the end threshold at which senescence starts "
                    "[degC_day]"
                ),
                "biosphere.phenology.computational.unity_factor": (
                    "Unit value of the phase factors [dimensionless]"
                ),
                "biosphere.phenology.computational.zero_initial": (
                    "Initial value of the phase factors [dimensionless]"
                ),
                "biosphere.phenology.computational.clamp_min": (
                    "Lower bound of the phenological factors [dimensionless]"
                ),
                "biosphere.phenology.computational.clamp_max": (
                    "Upper bound of the phenological factors [dimensionless]"
                ),
                "biosphere.phenology.turnover.background_senescence_rate": (
                    "Background senescence rate outside senescence [dimensionless]"
                ),
            },
            input_variables=[
                "day_of_year",
                "air_temperature",
                "air_temperature_max",
                "air_temperature_min",
                "phenology_gdd_accumulated",
                "phenology_gdd_previous_day_of_year",
            ],
            output_variables=[
                "leaf_area_index",
                "phenological_factor",
                "phenology_senescence_factor",
                "phenology_gdd_accumulated",
                "phenology_gdd_previous_day_of_year",
                "diagnostics.phenology.accumulated_gdd",
                "diagnostics.phenology.onset_event_flag",
                "diagnostics.phenology.onset_event_time_offset",
                "diagnostics.phenology.onset_event_time_sensitivity_gdd_start",
                "diagnostics.phenology.lai_sensitivity_jump_gdd_start",
                "phenology_phase",
            ],
            conservation_laws=["none"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        
        super().__init__(
            name=name,
            config=config,
            phenology_parameters=phenology_parameters,
            metadata=metadata,
            **kwargs
        )
        
        logger.info(f"Initialized GDDPhenology model '{name}' with base temp {self.phenology_params.gdd_base_temp}degC")
    
    def compute_phenological_state(
        self,
        state: StateDict,
        *,
        dt: float,
    ) -> PhenologyOutput:
        """
        Compute phenological state based on GDD accumulation.

        Parameters:
            state : StateDict
                Declared inputs: day_of_year [day], air_temperature,
                air_temperature_max and air_temperature_min [K], accumulated GDD
                [degC day] and previous day of year [day]
            dt : float
                Step length of the simulator call [s]; the one time control
                (the GDD increment is GDD_daily * dt / seconds_per_day)

        Returns:
            PhenologyOutput: Phenological state based on thermal time
        """
        day_of_year = _require_input(state, "day_of_year")
        temperature = _require_input(state, "air_temperature")
        temperature = torch.as_tensor(
            temperature,
            device=self.device,
            dtype=torch.float32,
        )
        if not torch.all(torch.isfinite(temperature)):
            raise ValueError("Phenology input 'air_temperature' must be finite")

        # References: McMaster and Wilhelm (1997).
        # Mathematical expression: T_C = T_K - T_freeze.
        # Parameter meanings: T_K is canonical air temperature and T_freeze is
        # the Celsius offset from the physical-constant SSOT.
        # Units: T_K and T_freeze [K], T_C [degC].
        # Open route: affine unit conversion only; no energy is exchanged.
        temp_celsius = temperature - T_FREEZE
        
        # Canonical extrema are required observations; no aliases or estimates.
        temp_max_raw = _require_input_tensor(
            state,
            "air_temperature_max",
            device=temperature.device,
            dtype=temperature.dtype,
        )
        temp_min_raw = _require_input_tensor(
            state,
            "air_temperature_min",
            device=temperature.device,
            dtype=temperature.dtype,
        )

        temp_max_celsius = temp_max_raw - T_FREEZE
        temp_min_celsius = temp_min_raw - T_FREEZE
        
        # Base temperature and max limit are already in Celsius in parameters
        base_temp_celsius = self.phenology_params.gdd_base_temp  # Already in degC
        max_temp_limit_celsius = self.phenology_params.gdd_max_temp_limit  # Already in degC
        
        # References: McMaster and Wilhelm (1997).
        # Mathematical expression:
        # GDD_daily = max(0, (T_max + T_min) / 2 - T_base).
        # Parameter meanings: T_max and T_min are daily extrema, T_base is the
        # ParameterManager-backed biological threshold, and T_limit caps heat.
        # Units: temperatures [degC], GDD_daily [degC day].
        # Open route: thermal time is a prognostic state, not a mass flux.
        daily_gdd = compute_gdd(
            temperature_max=temp_max_celsius,
            temperature_min=temp_min_celsius,
            base_temperature=base_temp_celsius,
            max_temperature=max_temp_limit_celsius,
            method="simple"
        )
        
        accumulated_gdd = _require_input_tensor(
            state,
            "phenology_gdd_accumulated",
            device=temperature.device,
            dtype=temperature.dtype,
        )
        if torch.any(accumulated_gdd < 0.0):
            raise ValueError(
                "Phenology input 'phenology_gdd_accumulated' must be non-negative"
            )
        previous_day = _require_input_tensor(
            state,
            "phenology_gdd_previous_day_of_year",
            device=temperature.device,
            dtype=temperature.dtype,
        )
        time_step = _step_length_tensor(
            dt,
            device=temperature.device,
            dtype=temperature.dtype,
        )
        previous_day_scalar = _scalar_value(
            previous_day,
            name="phenology_gdd_previous_day_of_year",
        )
        time_step_scalar = _scalar_value(time_step, name="dt")
        if time_step_scalar <= 0.0:
            raise ValueError("dt must be positive")

        # Reset accumulation only when the canonical calendar state crosses
        # from the end of one year into the configured early-year window.
        reset_day_threshold = _int_value(
            self.get_parameter_value(
                "biosphere.phenology.gdd.reset_day_threshold",
                required=True,
            ),
            name="gdd.reset_day_threshold",
        )
            
        day_of_year_scalar = _scalar_value(
            torch.as_tensor(day_of_year),
            name="day_of_year",
        )
        crossed_year_boundary = (
            day_of_year_scalar < previous_day_scalar
            and day_of_year_scalar <= reset_day_threshold
        )

        # References: McMaster and Wilhelm (1997); CF-1.8 time units.
        # Mathematical expression:
        # GDD_new = GDD_base + GDD_daily * dt / seconds_per_day.
        # Parameter meanings: GDD_base is zero after a year crossing and is
        # otherwise the StateManager-owned prior state; dt is the current step.
        # Units: GDD [degC day], dt and seconds_per_day [s].
        # Open route: REPLACE commits the complete prognostic state once.
        gdd_base = (
            torch.zeros_like(accumulated_gdd)
            if crossed_year_boundary
            else accumulated_gdd
        )
        seconds_per_day = torch.as_tensor(
            SECONDS_PER_DAY,
            device=temperature.device,
            dtype=temperature.dtype,
        )
        gdd_cum = gdd_base + daily_gdd * (time_step / seconds_per_day)
        current_day_state = torch.as_tensor(
            day_of_year_scalar,
            device=temperature.device,
            dtype=temperature.dtype,
        )
        
        # Determine phenological phase based on GDD thresholds
        gdd_start = self.phenology_params.gdd_threshold_start
        gdd_end = self.phenology_params.gdd_threshold_end
        
        expansion_gdd = _scalar_value(self.get_parameter_value("biosphere.phenology.gdd.expansion_phase_gdd"))
        senescence_buffer_gdd = _scalar_value(self.get_parameter_value("biosphere.phenology.gdd.senescence_buffer_gdd"))
        
        # Use tensor-safe comparisons for phenology phase determination
        gdd_cum_scalar = _scalar_value(torch.mean(gdd_cum))
        
        if gdd_cum_scalar < gdd_start:
            phase = PhenologyPhase.DORMANCY
        elif gdd_cum_scalar < gdd_start + expansion_gdd:  # Expansion phase GDD
            phase = PhenologyPhase.LEAF_EXPANSION
        elif gdd_cum_scalar < gdd_end - senescence_buffer_gdd:    # Full leaf period
            phase = PhenologyPhase.FULL_LEAF
        elif gdd_cum_scalar < gdd_end:
            phase = PhenologyPhase.SENESCENCE
        else:
            phase = PhenologyPhase.ABSCISSION
        
        # Compute growth factor based on GDD
        growth_factor = self._compute_gdd_growth_factor(gdd_cum, gdd_start, gdd_end)
        
        # Compute LAI
        lai = self.phenology_params.max_lai * growth_factor
        lai = torch.clamp(lai, min=self.phenology_params.min_lai, max=self.phenology_params.max_lai)
        
        # Compute senescence rate
        senescence_buffer = senescence_buffer_gdd
        # Use tensor-safe comparison for senescence check
        gdd_cum_val = _scalar_value(torch.mean(gdd_cum))
        if gdd_cum_val > gdd_end - senescence_buffer:  # Senescence period
            senescence_progress = (gdd_cum - (gdd_end - senescence_buffer)) / senescence_buffer
            clamp_min = self.get_parameter_value("biosphere.phenology.computational.clamp_min")
            clamp_max = self.get_parameter_value("biosphere.phenology.computational.clamp_max")
            senescence_rate = torch.clamp(senescence_progress, min=clamp_min, max=clamp_max)
        else:
            senescence_rate = torch.as_tensor(
                self.get_parameter_value("biosphere.phenology.turnover.background_senescence_rate"),
                device=self.device,
                dtype=torch.float32,
            )
        
        # Hybrid onset event (bounded saltation runtime, theory page hybrid_events.md Sec. 4).
        # References: Goebel, Sanfelice and Teel (2012); Kong et al. (2023) arXiv:2306.06862;
        #   McMaster and Wilhelm (1997).
        # Mathematical expression: guard g = GDD - GDD_start; crossing when g_n < 0 <= g_{n+1};
        #   tau - t_n = dt * (GDD_start - GDD_n) / (GDD_{n+1} - GDD_n);
        #   d tau / d GDD_start = -(g_x S_minus + g_theta)/(g_x f_minus + g_t) = t_day / GDD_d
        #   with g_x = 1, g_theta = -1, g_t = 0, S_minus = 0, f_minus = GDD_d / t_day;
        #   LAI jump S_plus - S_minus = (f_minus_LAI - f_plus_LAI) d tau / d GDD_start
        #   = -LAI_max / dGDD_full (R = identity, f_minus_LAI = 0).
        # Parameter meanings: GDD_n = gdd_base, GDD_{n+1} = gdd_cum, GDD_d = daily_gdd,
        #   t_day = seconds_per_day, dGDD_full = full_growth_offset, LAI_max = max_lai.
        # Units: GDD [degC day]; dt, tau offset [s]; tau sensitivity [s degC-1 day-1];
        #   LAI jump [m2 m-2 degC-1 day-1]; flag [1].
        # Domain/constraints: one transversal crossing per step (daily_gdd > 0 on the crossing
        #   step); no crossing -> flag 0 and zero sensitivities; observations only, the GDD
        #   and LAI numerics are unchanged; grazing is impossible here because a crossing
        #   requires gdd_cum > gdd_base.
        # Sign convention: raising GDD_start delays onset (positive tau sensitivity) and
        #   lowers expansion-phase LAI (negative jump).
        gdd_start_tensor = torch.as_tensor(gdd_start, device=gdd_cum.device, dtype=gdd_cum.dtype)
        full_growth_offset_tensor = torch.as_tensor(
            self.get_parameter_value("biosphere.phenology.gdd.full_growth_offset"),
            device=gdd_cum.device,
            dtype=gdd_cum.dtype,
        )
        guard_start = torch.mean(gdd_base) - gdd_start_tensor
        guard_end = torch.mean(gdd_cum) - gdd_start_tensor
        zero_value = torch.zeros((), device=gdd_cum.device, dtype=gdd_cum.dtype)
        if bool((guard_start < 0.0).item()) and bool((guard_end >= 0.0).item()):
            onset_offset = locate_guard_crossing(guard_start, guard_end, zero_value, time_step) - zero_value
            rate_minus = torch.mean(daily_gdd) / seconds_per_day
            onset_tau_theta = event_time_sensitivity(
                torch.ones_like(rate_minus), -torch.ones_like(rate_minus), zero_value, rate_minus, zero_value
            )
            lai_rate_plus = (
                torch.as_tensor(self.phenology_params.max_lai, device=gdd_cum.device, dtype=gdd_cum.dtype)
                * rate_minus
                / full_growth_offset_tensor
            )
            lai_jump = sensitivity_jump(
                zero_value, torch.ones_like(rate_minus), zero_value, zero_value, zero_value, lai_rate_plus, onset_tau_theta
            )
            onset_flag = torch.ones((), device=gdd_cum.device, dtype=gdd_cum.dtype)
        else:
            onset_offset = zero_value
            onset_tau_theta = zero_value
            lai_jump = zero_value
            onset_flag = zero_value
        # Prepare output with diagnostic variables
        diagnostic_vars = {
            'daily_gdd': daily_gdd,
            'cumulative_gdd': gdd_cum,
            'gdd_threshold_start': torch.as_tensor(gdd_start, device=self.device, dtype=torch.float32),
            'gdd_threshold_end': torch.as_tensor(gdd_end, device=self.device, dtype=torch.float32),
            'temperature_celsius': temp_celsius
        }
        
        # Canonical StateManager states and the public accumulated-GDD diagnostic.
        additional_outputs = {
            'phenology_gdd_accumulated': gdd_cum,
            'phenology_gdd_previous_day_of_year': current_day_state,
            'diagnostics.phenology.accumulated_gdd': gdd_cum,
            'diagnostics.phenology.onset_event_flag': onset_flag,
            'diagnostics.phenology.onset_event_time_offset': onset_offset,
            'diagnostics.phenology.onset_event_time_sensitivity_gdd_start': onset_tau_theta,
            'diagnostics.phenology.lai_sensitivity_jump_gdd_start': lai_jump,
        }
        
        output = PhenologyOutput(
            phenology_phase=phase,
            growing_season_index=None,
            thermal_time=gdd_cum,
            leaf_area_index=lai,
            growth_factor=growth_factor,
            senescence_rate=senescence_rate,
            diagnostic_variables=diagnostic_vars,
            additional_outputs=additional_outputs
        )
        
        # Perform conservation checks
        if self.enforce_physical_constraints:
            self._validate_output_bounds(output)
        
        return output
    
    def _compute_gdd_growth_factor(
        self, 
        gdd_cum: Tensor, 
        gdd_start: float, 
        gdd_end: float
    ) -> Tensor:
        """
        Compute growth factor based on cumulative GDD.
        
        Parameters:
            gdd_cum : Tensor
                Cumulative growing degree days
            gdd_start : float
                GDD threshold for growth start
            gdd_end : float
                GDD threshold for growth end
                
        Returns:
            Tensor: Growth factor (0-1)
        """
        # Growth phase parameters from parameter manager
        gdd_start_t = torch.as_tensor(gdd_start, device=gdd_cum.device, dtype=gdd_cum.dtype)
        gdd_end_t = torch.as_tensor(gdd_end, device=gdd_cum.device, dtype=gdd_cum.dtype)
        full_growth_offset = torch.as_tensor(
            self.get_parameter_value("biosphere.phenology.gdd.full_growth_offset"),
            device=gdd_cum.device,
            dtype=gdd_cum.dtype
        )
        senescence_start_offset = torch.as_tensor(
            self.get_parameter_value("biosphere.phenology.gdd.senescence_start_offset"),
            device=gdd_cum.device,
            dtype=gdd_cum.dtype
        )
        gdd_full = gdd_start_t + full_growth_offset  # GDD to reach full growth
        gdd_senescence_start = gdd_end_t - senescence_start_offset  # Start senescence before end

        # Growth factor calculation using parameters from manager
        zero_initial = torch.as_tensor(
            self.get_parameter_value("biosphere.phenology.computational.zero_initial"),
            device=gdd_cum.device,
            dtype=gdd_cum.dtype
        )
        unity_factor = torch.as_tensor(
            self.get_parameter_value("biosphere.phenology.computational.unity_factor"),
            device=gdd_cum.device,
            dtype=gdd_cum.dtype
        )
        clamp_min = torch.as_tensor(
            self.get_parameter_value("biosphere.phenology.computational.clamp_min"),
            device=gdd_cum.device,
            dtype=gdd_cum.dtype
        )
        clamp_max = torch.as_tensor(
            self.get_parameter_value("biosphere.phenology.computational.clamp_max"),
            device=gdd_cum.device,
            dtype=gdd_cum.dtype
        )
        
        growth_factor = torch.where(
            gdd_cum < gdd_start_t,
            zero_initial,
            torch.where(
                gdd_cum < gdd_full,
                (gdd_cum - gdd_start_t) / (gdd_full - gdd_start_t),
                torch.where(
                    gdd_cum < gdd_senescence_start,
                    unity_factor,
                    torch.where(
                        gdd_cum < gdd_end_t,
                        1.0 - (gdd_cum - gdd_senescence_start) / (gdd_end_t - gdd_senescence_start),
                        zero_initial
                    )
                )
            )
        )
        
        return torch.clamp(growth_factor, min=clamp_min, max=clamp_max)
    
class GSIPhenology(PhenologyBase):
    """
    Growing Season Index (GSI) Phenology Model
    
    This model implements the GSI approach developed by Jolly et al. (2005)
    for dynamic phenological timing based on multiple environmental factors.
    GSI integrates minimum temperature, vapor pressure deficit, and
    photoperiod to predict optimal conditions for vegetation activity.
    
    The GSI approach represents one of the most comprehensive environmental
    phenology models and has been extensively tested across global ecosystems.
    This implementation follows the RHESSys computational framework while
    maintaining consistency with the RELIC architecture.
    
    Mathematical Expression:
        iGSI = f_temp(T_min) * f_vpd(VPD) * f_daylen(DL)
        
        Where each factor ranges from 0 (limiting) to 1 (optimal):
        - f_temp: minimum temperature limitation
        - f_vpd: vapor pressure deficit limitation  
        - f_daylen: photoperiod limitation
        
        Phenological transitions occur when GSI crosses thresholds:
        - Onset: GSI > GSI_onset_threshold (typically 0.5)
        - Offset: GSI < GSI_offset_threshold (typically 0.5)
    
    Key Features:
        - Multi-factor environmental integration
        - Dynamic threshold-based transitions
        - Memory effects through running averages
        - Global applicability across biomes
        - Climate change sensitivity
        
    Applications:
        - Global vegetation models (LPJ-GUESS, CLM, etc.)
        - Climate change impact studies
        - Seasonal forecasting systems
        - Ecosystem productivity modeling
        
    References:
        - Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict 
          foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.
        - White, M.A., et al. (1997). A continental phenology model for monitoring 
          vegetation responses to interannual climatic variability.
        - RHESSys Documentation. compute_growingseason_index.c implementation.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    PHENOLOGY_TYPE = "gsi"
    OUTPUT_UNITS = {
        **PhenologyBase.OUTPUT_UNITS,
        "diagnostics.phenology.growing_season_index": "dimensionless",
        "phenology_gsi_history": "dimensionless",
        "phenology_gsi_history_index": "dimensionless",
        "phenology_gsi_history_count": "dimensionless",
        "phenology_gsi_in_growing_season": "dimensionless",
    }
    
    def __init__(
        self,
        name: str = "gsi_phenology",
        config: Optional[Dict[str, Any]] = None,
        phenology_parameters: Optional[PhenologyParameters] = None,
        **kwargs
    ) -> None:
        """Initialize GSI phenology model with Jolly et al. 2005 parameters."""
        
        if config is None:
            config = {}

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Growing Season Index (GSI) phenology model based on Jolly et al. 2005 with multi-factor environmental control",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=r"""
            GSI phenology equations (Jolly et al. 2005):
            
            iGSI = f_{temp}(T_{min}) \times f_{vpd}(VPD) \times f_{day}(DL)
            
            Temperature limitation:
            f_{temp}(T_{min}) = \frac{T_{min} - T_{min}^{thresh}}{T_{min}^{opt} - T_{min}^{thresh}}
            \text{ for } T_{min}^{thresh} \leq T_{min} \leq T_{min}^{opt}
            
            VPD limitation:
            f_{vpd}(VPD) = \frac{VPD_{max} - VPD}{VPD_{max} - VPD_{min}}
            \text{ for } VPD_{min} \leq VPD \leq VPD_{max}
            
            Daylight limitation:  
            f_{day}(DL) = \frac{DL - DL_{min}}{DL_{thresh} - DL_{min}}
            \text{ for } DL_{min} \leq DL \leq DL_{thresh}
            
            Phenological transitions:
            - Onset: \overline{GSI}_{mem} > GSI_{onset}
            - Offset: \overline{GSI}_{mem} < GSI_{offset}
            
            Where \overline{GSI}_{mem} is rolling mean over memory period.
            """,
            references=[
                "Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.",
                "White, M.A., et al. (1997). A continental phenology model for monitoring vegetation responses to interannual climatic variability. Global Biogeochemical Cycles, 11(2), 217-234.",
                "Band, L.E., et al. (1993). Regional Hydro-Ecological Simulation System (RHESSys): An object-oriented approach to spatially distributed modeling of carbon, water, and nutrient cycling. Ecological Modelling, 68(3-4), 129-146."
            ],
            parameters={
                "biosphere.phenology.activity.minimal_dormancy": (
                    "Minimal activity during dormancy [dimensionless]"
                ),
                "biosphere.phenology.gsi.expansion_factor": (
                    "Growth factor of the expansion phase [dimensionless]"
                ),
                "biosphere.phenology.gsi.expansion_min_clamp": (
                    "Lower bound of the expansion growth factor [dimensionless]"
                ),
                "biosphere.phenology.gsi.expansion_max_clamp": (
                    "Upper bound of the expansion growth factor [dimensionless]"
                ),
                "biosphere.phenology.gsi.full_leaf_factor": (
                    "Growth factor of the full-leaf phase [dimensionless]"
                ),
                "biosphere.phenology.gsi.full_leaf_min_clamp": (
                    "Lower bound of the full-leaf growth factor [dimensionless]"
                ),
                "biosphere.phenology.gsi.senescence_factor": (
                    "Growth factor of the senescence phase [dimensionless]"
                ),
                "biosphere.phenology.gsi.senescence_min_clamp": (
                    "Lower bound of the senescence growth factor [dimensionless]"
                ),
                "biosphere.phenology.gsi.senescence_max_clamp": (
                    "Upper bound of the senescence growth factor [dimensionless]"
                ),
                "biosphere.phenology.gsi.memory_multiplier": (
                    "Multiplier of the GSI memory in the senescence rate [dimensionless]"
                ),
                "biosphere.phenology.computational.clamp_min": (
                    "Lower bound of the phenological factors [dimensionless]"
                ),
                "biosphere.phenology.computational.clamp_max": (
                    "Upper bound of the phenological factors [dimensionless]"
                ),
                "biosphere.phenology.turnover.background_senescence_rate": (
                    "Background senescence rate outside senescence [dimensionless]"
                ),
            },
            input_variables=[
                "day_of_year",
                "air_temperature",
                "air_temperature_min",
                "vapor_pressure_deficit",
                "daylength",
                "phenology_gsi_history",
                "phenology_gsi_history_index",
                "phenology_gsi_history_count",
                "phenology_gsi_in_growing_season",
            ],
            output_variables=[
                "leaf_area_index",
                "phenological_factor",
                "diagnostics.phenology.growing_season_index",
                "phenology_gsi_history",
                "phenology_gsi_history_index",
                "phenology_gsi_history_count",
                "phenology_gsi_in_growing_season",
                "phenology_senescence_factor",
                "phenology_phase",
            ],
            coupling_interfaces=["atmosphere", "radiation", "photosynthesis"],
            conservation_laws=["none"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        
        super().__init__(
            name=name,
            config=config,
            phenology_parameters=phenology_parameters,
            metadata=metadata,
            **kwargs
        )
        
        # This is immutable parameter metadata, not prognostic runtime state.
        self.memory_days = int(max(1, round(float(self.phenology_params.gsi_memory_days))))
        logger.info(
            f"Initialized GSIPhenology model '{name}' with "
            f"onset threshold {self.phenology_params.gsi_onset_threshold}, "
            f"memory days {self.memory_days}"
        )
    
    def compute_phenological_state(
        self,
        state: StateDict,
        *,
        dt: float,
    ) -> PhenologyOutput:
        """
        Compute phenological state using GSI approach.

        This method implements the three-factor daily GSI algorithm following
        Jolly et al. (2005), with StateManager-owned rolling memory and
        threshold-based phenological transitions.

        Parameters:
            state : StateDict
                Declared inputs:
                - air_temperature: daily average air temperature [K]
                - air_temperature_min: daily minimum temperature [K]
                - vapor_pressure_deficit: VPD [Pa]
                - daylength: day length [seconds]
                - phenology_gsi_history: daily GSI circular buffer [1]
                - phenology_gsi_history_index: next buffer index [1]
                - phenology_gsi_history_count: valid buffer count [1]
                - phenology_gsi_in_growing_season: hysteresis state [1]
            dt : float
                Step length of the simulator call [s]; the one time control
                (exactly one day)

        Returns:
            PhenologyOutput: Complete GSI-based phenological state
        """
        # References: Jolly et al. (2005), doi:10.1111/j.1365-2486.2005.00930.x.
        # Mathematical expressions: iGSI = f_T * f_VPD * f_day and
        # GSI_bar = mean(history_next[valid samples]).
        # Parameter meanings: climatic thresholds define the three response
        # factors; onset/offset thresholds define phenological hysteresis; and
        # memory_days defines the number of daily samples in the rolling mean.
        # Units: air temperatures [K] at the interface and [degC] internally,
        # VPD [Pa], daylength and dt [s], and all GSI states [1].
        # Open-system route: climate is external forcing; GSI is a control state
        # and carries no carbon, nitrogen, water, or energy flux.
        prototype = torch.as_tensor(
            _require_input(state, "air_temperature"),
            device=self.device,
            dtype=torch.float32,
        )
        if not torch.all(torch.isfinite(prototype)):
            raise ValueError("air_temperature must be finite")
        if torch.any(prototype <= 0.0):
            raise ValueError("air_temperature must use the canonical Kelvin scale")
        temp_min = _require_input_tensor(
            state,
            "air_temperature_min",
            device=prototype.device,
            dtype=prototype.dtype,
        )
        vpd = _require_input_tensor(
            state,
            "vapor_pressure_deficit",
            device=prototype.device,
            dtype=prototype.dtype,
        )
        daylength = _require_input_tensor(
            state,
            "daylength",
            device=prototype.device,
            dtype=prototype.dtype,
        )
        if torch.any(temp_min <= 0.0):
            raise ValueError("air_temperature_min must use the canonical Kelvin scale")
        if torch.any(vpd < 0.0):
            raise ValueError("vapor_pressure_deficit must be non-negative")
        if torch.any(daylength < 0.0) or torch.any(daylength > SECONDS_PER_DAY):
            raise ValueError("daylength must be within one day")

        time_step = _scalar_value(
            _step_length_tensor(dt, device=prototype.device, dtype=prototype.dtype),
            name="dt",
        )
        if not math.isclose(
            time_step,
            float(SECONDS_PER_DAY),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("GSI phenology requires an exact daily time step")

        # The public forcing contract is Kelvin. Jolly response thresholds are
        # stored in degC, so the conversion is unconditional and unit-explicit.
        temp_min_celsius = temp_min - T_FREEZE
        tmin_threshold_celsius = float(self.phenology_params.gsi_tmin_threshold)
        tmin_optimal_celsius = float(self.phenology_params.gsi_tmin_optimal)

        gsi = compute_gsi(
            temperature_min=temp_min_celsius,
            vapor_pressure_deficit=vpd,
            daylength=daylength,
            tmin_threshold=tmin_threshold_celsius,
            tmin_optimal=tmin_optimal_celsius,
            vpd_min=self.phenology_params.gsi_vpd_min,
            vpd_max=self.phenology_params.gsi_vpd_max,
            daylen_threshold=self.phenology_params.gsi_daylen_threshold,
        )
        if gsi.numel() != 1:
            raise ValueError("GSIPhenology requires one grid-cell forcing sample")
        gsi = gsi.reshape(())

        history_key = "phenology_gsi_history"
        history_value = _require_input(state, history_key)
        if not isinstance(history_value, Tensor):
            raise TypeError(f"Phenology input '{history_key}' must be a torch.Tensor")
        if not history_value.is_floating_point():
            raise TypeError(f"Phenology input '{history_key}' must be floating point")
        history = history_value.to(device=prototype.device, dtype=prototype.dtype)
        if history.ndim != 1 or history.numel() != self.memory_days:
            raise ValueError(
                "phenology_gsi_history length must equal gsi_memory_days"
            )
        if not torch.all(torch.isfinite(history)):
            raise ValueError("phenology_gsi_history must be finite")
        if torch.any(history < 0.0) or torch.any(history > 1.0):
            raise ValueError("phenology_gsi_history must be within [0, 1]")

        def require_integer_state(key: str, lower: int, upper: int) -> int:
            value = _require_input(state, key)
            if not isinstance(value, Tensor):
                raise TypeError(f"Phenology input '{key}' must be a torch.Tensor")
            if not value.is_floating_point():
                raise TypeError(f"Phenology input '{key}' must be floating point")
            scalar = _scalar_value(value, name=key)
            integer = int(round(scalar))
            if not math.isclose(scalar, float(integer), rel_tol=0.0, abs_tol=1.0e-6):
                raise ValueError(f"{key} must have integer semantics")
            if integer < lower or integer > upper:
                raise ValueError(f"{key} is outside its canonical bounds")
            return integer

        history_index = require_integer_state(
            "phenology_gsi_history_index",
            0,
            self.memory_days - 1,
        )
        history_count = require_integer_state(
            "phenology_gsi_history_count",
            0,
            self.memory_days,
        )
        if history_count < self.memory_days and history_index != history_count:
            raise ValueError(
                "Partial phenology_gsi_history requires index equal to count"
            )
        season_flag = require_integer_state(
            "phenology_gsi_in_growing_season",
            0,
            1,
        )

        # References: Jolly et al. (2005), daily moving-average GSI.
        # Mathematical expressions: h_next[index] = iGSI,
        # index_next = (index + 1) mod N, count_next = min(count + 1, N),
        # and GSI_bar = sum(h_next_valid) / count_next.
        # Parameter meanings: N is memory_days; index is the next write slot;
        # count is the number of valid daily samples.
        # Units: history, iGSI, and GSI_bar are dimensionless; cadence is day.
        # Open-system route: full-state replacement preserves restart/provenance
        # without adding a conserved-material source or sink.
        history_next = history.clone()
        history_next[history_index] = gsi
        history_index_next = (history_index + 1) % self.memory_days
        history_count_next = min(history_count + 1, self.memory_days)
        if history_count_next < self.memory_days:
            gsi_memory = torch.mean(history_next[:history_count_next])
        else:
            gsi_memory = torch.mean(history_next)

        onset_threshold = float(self.phenology_params.gsi_onset_threshold)
        offset_threshold = float(self.phenology_params.gsi_offset_threshold)
        gsi_memory_scalar = _scalar_value(gsi_memory, name="rolling_gsi")

        # Hysteresis is reconstructed solely from canonical StateManager state.
        if season_flag == 0 and gsi_memory_scalar >= onset_threshold:
            season_flag_next = 1
            phase = PhenologyPhase.LEAF_EXPANSION
        elif season_flag == 1 and gsi_memory_scalar <= offset_threshold:
            season_flag_next = 0
            phase = PhenologyPhase.SENESCENCE
        elif season_flag == 1:
            season_flag_next = 1
            phase = PhenologyPhase.FULL_LEAF
        else:
            season_flag_next = 0
            phase = PhenologyPhase.DORMANCY

        growth_factor = self._compute_gsi_growth_factor(gsi_memory, phase)
        lai = self.phenology_params.max_lai * growth_factor
        lai = torch.clamp(lai, min=self.phenology_params.min_lai, max=self.phenology_params.max_lai)

        if phase == PhenologyPhase.SENESCENCE or phase == PhenologyPhase.ABSCISSION:
            clamp_min = self.get_parameter_value("biosphere.phenology.computational.clamp_min")
            clamp_max = self.get_parameter_value("biosphere.phenology.computational.clamp_max")
            memory_multiplier = self.get_parameter_value("biosphere.phenology.gsi.memory_multiplier")
            senescence_rate = torch.clamp(1.0 - gsi_memory * memory_multiplier, min=clamp_min, max=clamp_max)
        else:
            senescence_rate = self.get_parameter_value(
                "biosphere.phenology.turnover.background_senescence_rate"
            )

        additional_outputs = {
            "diagnostics.phenology.growing_season_index": gsi,
            "phenology_gsi_history": history_next,
            "phenology_gsi_history_index": torch.as_tensor(
                float(history_index_next),
                device=prototype.device,
                dtype=prototype.dtype,
            ),
            "phenology_gsi_history_count": torch.as_tensor(
                float(history_count_next),
                device=prototype.device,
                dtype=prototype.dtype,
            ),
            "phenology_gsi_in_growing_season": torch.as_tensor(
                float(season_flag_next),
                device=prototype.device,
                dtype=prototype.dtype,
            ),
        }

        output = PhenologyOutput(
            phenology_phase=phase,
            growing_season_index=gsi_memory,
            thermal_time=None,
            leaf_area_index=lai,
            growth_factor=growth_factor,
            senescence_rate=senescence_rate,
            diagnostic_variables={},
            additional_outputs=additional_outputs,
        )

        if self.enforce_physical_constraints:
            self._validate_output_bounds(output)

        return output
    
    def _compute_gsi_growth_factor(self, gsi: Tensor, phase: PhenologyPhase) -> Tensor:
        """
        Compute growth factor from GSI value and phenological phase.
        
        Parameters:
            gsi : Tensor
                Current GSI value (0-1)
            phase : PhenologyPhase
                Current phenological phase
                
        Returns:
            Tensor: Growth factor (0-1)
        """
        if phase == PhenologyPhase.DORMANCY:
            minimal_activity = self.get_parameter_value(
                'minimal_dormancy',
                namespace="biosphere.phenology.activity",
            )
            return torch.as_tensor(minimal_activity, device=self.device, dtype=torch.float32)  # Minimal activity
        elif phase == PhenologyPhase.LEAF_EXPANSION:
            # Growth factor increases with GSI during expansion
            expansion_factor = self.get_parameter_value("biosphere.phenology.gsi.expansion_factor")
            expansion_min = self.get_parameter_value("biosphere.phenology.gsi.expansion_min_clamp")
            expansion_max = self.get_parameter_value("biosphere.phenology.gsi.expansion_max_clamp")
            return torch.clamp(gsi * expansion_factor, min=expansion_min, max=expansion_max)
        elif phase == PhenologyPhase.FULL_LEAF:
            # Maximum growth when in full leaf phase
            full_leaf_factor = self.get_parameter_value("biosphere.phenology.gsi.full_leaf_factor")
            full_leaf_min = self.get_parameter_value("biosphere.phenology.gsi.full_leaf_min_clamp")
            full_leaf_max = self.get_parameter_value("biosphere.phenology.computational.clamp_max")
            return torch.clamp(gsi * full_leaf_factor, min=full_leaf_min, max=full_leaf_max)
        elif phase == PhenologyPhase.SENESCENCE:
            # Growth factor decreases during senescence
            senescence_factor = self.get_parameter_value("biosphere.phenology.gsi.senescence_factor")
            senescence_min = self.get_parameter_value("biosphere.phenology.gsi.senescence_min_clamp")
            senescence_max = self.get_parameter_value("biosphere.phenology.gsi.senescence_max_clamp")
            return torch.clamp(gsi * senescence_factor, min=senescence_min, max=senescence_max)
        else:  # ABSCISSION or other
            minimal_activity = self.get_parameter_value(
                'minimal_dormancy',
                namespace="biosphere.phenology.activity",
            )
            return torch.as_tensor(minimal_activity, device=self.device, dtype=torch.float32)
    
class CropPhenology(PhenologyBase):
    """
    Crop-Specific Phenology Model
    
    This model implements phenological development for agricultural crops
    based on thermal time accumulation (Potential Heat Units, PHU) and
    specific crop development stages. It follows the DSSAT/APSIM approach
    commonly used in agricultural modeling systems.
    
    The model tracks crop development through distinct stages from sowing
    to harvest, with stage-specific responses to environmental conditions
    and management practices including sowing date, variety selection,
    and harvest timing optimization.
    
    Mathematical Expression:
        Daily PHU: PHU_daily = f_temp(T_daily - T_base_crop)
        Cumulative PHU: PHU_cum = Sigma PHU_daily  
        Development index: DI = PHU_cum / PHU_total
        
        Harvest index: HI(t) = HI_max * g(DI, DI_start_HI)
        Where:
        - PHU: Potential Heat Units (crop growing degree days)
        - DI: Development index (0 = sowing, 1 = maturity)
        - HI: Harvest index (grain/total biomass ratio)
    
    Key Features:
        - Crop-specific parameterization (wheat, maize, soybean, rice)
        - Development stage tracking (vegetative, reproductive, grain filling)
        - Harvest index dynamics
        - Vernalization requirements (winter crops)
        - Photoperiod sensitivity
        - Water and nitrogen stress responses
        
    Applications:
        - Crop yield forecasting
        - Agricultural system modeling  
        - Climate change impact assessment on crops
        - Irrigation and fertilization optimization
        - Food security analysis
        
    References:
        - Jones, J.W., et al. (2003). The DSSAT cropping system model. 
          European Journal of Agronomy, 18(3-4), 235-265.
        - Keating, B.A., et al. (2003). An overview of APSIM, a model designed 
          for farming systems simulation. European Journal of Agronomy, 18(3-4), 267-288.
        - Ritchie, J.T. (1991). Wheat phasic development. Agricultural Systems, 36(3), 305-330.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    PHENOLOGY_TYPE = "crop"
    OUTPUT_UNITS = {
        **PhenologyBase.OUTPUT_UNITS,
        "harvest_index": "dimensionless",
        "diagnostics.phenology.accumulated_phu": "degC day",
        "phenology_crop_accumulated_phu": "degC day",
        "phenology_crop_is_sown": "dimensionless",
        "phenology_crop_is_harvested": "dimensionless",
        "phenology_crop_vernalization_days": "day",
    }
    
    def __init__(
        self,
        name: str = "crop_phenology",
        config: Optional[Dict[str, Any]] = None,
        phenology_parameters: Optional[PhenologyParameters] = None,
        **kwargs
    ) -> None:
        """Initialize crop phenology model."""
        
        if config is None:
            config = {}

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0", 
            description="Crop-specific phenology model with development stages and harvest index dynamics",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=r"""
            Crop phenology equations:
            
            Daily thermal time: PHU_{daily} = \max(0, T_{avg} - T_{base})
            
            Cumulative PHU: PHU_{cum} = \sum_{i=sowing}^{t} PHU_{daily,i}
            
            Development index: DI = \frac{PHU_{cum}}{PHU_{total}}
            
            Harvest index dynamics:
            HI(t) = HI_{max} \times \frac{DI - DI_{HI-start}}{1 - DI_{HI-start}}
            \text{ for } DI \geq DI_{HI-start}
            
            Development stages:
            - Germination: 0 < DI < 0.15
            - Vegetative: 0.15 < DI < 0.4
            - Reproductive: 0.4 < DI < 0.65
            - Grain filling: 0.65 < DI < 1.0
            - Maturity: DI \geq 1.0
            """,
            references=[
                "Jones, J.W., et al. (2003). The DSSAT cropping system model. European Journal of Agronomy, 18(3-4), 235-265.",
                "Keating, B.A., et al. (2003). An overview of APSIM, a model designed for farming systems simulation. European Journal of Agronomy, 18(3-4), 267-288.",
                "Ritchie, J.T. (1991). Wheat phasic development. Agricultural Systems, 36(3), 305-320."
            ],
            parameters={
                "biosphere.phenology.activity.germination_activity": (
                    "Activity during germination [dimensionless]"
                ),
                "biosphere.phenology.activity.vegetative_activity": (
                    "Maximum activity of the vegetative phase [dimensionless]"
                ),
                "biosphere.phenology.activity.maturity_senescence": (
                    "Maximum senescence rate at maturity [dimensionless]"
                ),
                "biosphere.phenology.crop.di_germination_end": (
                    "Development index at the end of germination [dimensionless]"
                ),
                "biosphere.phenology.crop.di_vegetative_peak": (
                    "Development index of the vegetative peak [dimensionless]"
                ),
                "biosphere.phenology.crop.di_vegetative_end": (
                    "Development index at the end of the vegetative stage [dimensionless]"
                ),
                "biosphere.phenology.crop.di_lai_peak_end": (
                    "Development index at the end of the LAI plateau [dimensionless]"
                ),
                "biosphere.phenology.crop.di_reproductive_end": (
                    "Development index at the end of the reproductive stage [dimensionless]"
                ),
                "biosphere.phenology.crop.initial_lai": "Leaf area index at emergence [m2 m-2]",
                "biosphere.phenology.crop.lai_growth_exponent": (
                    "Exponent of the LAI growth factor [dimensionless]"
                ),
                "biosphere.phenology.crop.grain_filling_factor": (
                    "Senescence multiplier during grain filling [dimensionless]"
                ),
                "biosphere.phenology.crop.senescence_growth_factor": (
                    "Growth multiplier in early senescence [dimensionless]"
                ),
                "biosphere.phenology.crop.senescence_min_factor": (
                    "Minimum growth factor in senescence [dimensionless]"
                ),
                "biosphere.phenology.crop.vernalization_increment": (
                    "Daily increment of the vernalization days [day]"
                ),
                "biosphere.phenology.crop.vernalization_min_factor": (
                    "Lower bound of the vernalization multiplier [dimensionless]"
                ),
                "biosphere.phenology.crop.year_reset_threshold": (
                    "Day of year that resets the crop cycle [day]"
                ),
                "biosphere.phenology.computational.clamp_min": (
                    "Lower bound of the phenological factors [dimensionless]"
                ),
                "biosphere.phenology.computational.clamp_max": (
                    "Upper bound of the phenological factors [dimensionless]"
                ),
            },
            input_variables=[
                "day_of_year",
                "air_temperature",
                "phenology_crop_accumulated_phu",
                "phenology_crop_is_sown",
                "phenology_crop_is_harvested",
                "phenology_crop_vernalization_days",
            ],
            output_variables=[
                "leaf_area_index",
                "phenology_phase",
                "diagnostics.phenology.accumulated_phu",
                "harvest_index",
                "phenological_factor",
                "phenology_crop_accumulated_phu",
                "phenology_crop_is_sown",
                "phenology_crop_is_harvested",
                "phenology_crop_vernalization_days",
            ],
            coupling_interfaces=["crop_management", "harvest"],
            conservation_laws=["none"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        
        super().__init__(
            name=name,
            config=config,
            phenology_parameters=phenology_parameters,
            metadata=metadata,
            **kwargs
        )
        
        logger.info(
            f"Initialized CropPhenology model '{name}' for "
            f"{self.phenology_params.crop_type} with PHU total "
            f"{self.phenology_params.crop_phu_total}"
        )
    
    def compute_phenological_state(
        self,
        state: StateDict,
        *,
        dt: float,
    ) -> PhenologyOutput:
        """
        Compute crop phenology from StateManager-owned temporal state.

        Parameters:
            state : StateDict
                Declared inputs: day_of_year (1-366), air_temperature [K] and the
                canonical Crop temporal state
            dt : float
                Step length of the simulator call [s]; the one time control

        Returns:
            PhenologyOutput: Crop phenological state with development stages
        """
        prototype = torch.as_tensor(
            _require_input(state, "air_temperature"),
            device=self.device,
            dtype=torch.float32,
        )
        if prototype.numel() != 1:
            raise ValueError("CropPhenology requires one grid-cell temperature sample")
        prototype = prototype.reshape(())
        if not torch.isfinite(prototype):
            raise ValueError("air_temperature must be finite")
        if prototype <= 0.0:
            raise ValueError("air_temperature must use the canonical Kelvin scale")

        doy_scalar = _scalar_value(_require_input(state, "day_of_year"), name="day_of_year")
        if doy_scalar < 1.0 or doy_scalar > 366.0:
            raise ValueError("day_of_year must be within [1, 366]")
        time_step = _step_length_tensor(
            dt,
            device=prototype.device,
            dtype=prototype.dtype,
        )
        if time_step.numel() != 1:
            raise ValueError("dt must contain exactly one scalar value")
        time_step = time_step.reshape(())
        if time_step <= 0.0:
            raise ValueError("dt must be positive")

        def require_scalar_state(key: str, *, non_negative: bool) -> Tensor:
            value = _require_input_tensor(
                state,
                key,
                device=prototype.device,
                dtype=prototype.dtype,
            )
            if value.numel() != 1:
                raise ValueError(f"{key} must contain exactly one scalar value")
            value = value.reshape(())
            if non_negative and value < 0.0:
                raise ValueError(f"{key} must be non-negative")
            return value

        def require_binary_state(key: str) -> int:
            value = require_scalar_state(key, non_negative=True)
            scalar = _scalar_value(value, name=key)
            flag = int(round(scalar))
            if flag not in (0, 1) or not math.isclose(
                scalar,
                float(flag),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError(f"{key} must have binary 0 or 1 semantics")
            return flag

        accumulated_phu = require_scalar_state(
            "phenology_crop_accumulated_phu",
            non_negative=True,
        ).clone()
        vernalization_days = require_scalar_state(
            "phenology_crop_vernalization_days",
            non_negative=True,
        ).clone()
        is_sown = require_binary_state("phenology_crop_is_sown")
        is_harvested = require_binary_state("phenology_crop_is_harvested")
        if is_harvested and not is_sown:
            raise ValueError(
                "Invalid crop cycle flags: harvested state requires sown state"
            )

        sowing_doy = int(self.phenology_params.crop_sowing_doy)
        harvest_doy = int(self.phenology_params.crop_harvest_doy)
        reset_threshold = _scalar_value(
            self.get_parameter_value(
                "biosphere.phenology.crop.year_reset_threshold",
                required=True,
            ),
            name="crop.year_reset_threshold",
        )
        sowing_due = abs(doy_scalar - float(sowing_doy)) < 1.0

        # A harvested cycle is reset at the early-year boundary or immediately
        # before the next configured sowing event. StateManager still owns the
        # complete cycle state and commits the returned replacement values.
        if is_harvested and (doy_scalar < reset_threshold or sowing_due):
            accumulated_phu = torch.zeros_like(accumulated_phu)
            vernalization_days = torch.zeros_like(vernalization_days)
            is_sown = 0
            is_harvested = 0
        if not is_sown and sowing_due:
            accumulated_phu = torch.zeros_like(accumulated_phu)
            vernalization_days = torch.zeros_like(vernalization_days)
            is_sown = 1
            is_harvested = 0

        phu_total = float(self.phenology_params.crop_phu_total)
        if not math.isfinite(phu_total) or phu_total <= 0.0:
            raise ValueError("crop_phu_total must be finite and positive")
        prior_development_index = torch.clamp(
            accumulated_phu / phu_total,
            min=0.0,
            max=1.0,
        )
        if sowing_doy <= harvest_doy:
            harvest_date_reached = doy_scalar >= float(harvest_doy)
        else:
            harvest_date_reached = (
                doy_scalar < float(sowing_doy)
                and doy_scalar >= float(harvest_doy)
            )
        maturity_reached = _scalar_value(
            prior_development_index,
            name="crop development index",
        ) >= 1.0
        if is_sown and not is_harvested and (
            harvest_date_reached or maturity_reached
        ):
            is_harvested = 1

        daily_phu = torch.zeros_like(accumulated_phu)
        if is_sown and not is_harvested:
            # References: Jones et al. (2003); McMaster and Wilhelm (1997).
            # Mathematical expressions: T_C = T_K - T_freeze and
            # PHU_step = max(T_C - T_base, 0) * dt / seconds_per_day.
            # Parameter meanings: T_base is the crop base temperature; dt is
            # the explicit integration interval; PHU is prior canonical state.
            # Units: T [K or degC], dt [s], PHU_step and PHU [degC day].
            # Open route: thermal time is a control state with no material flux.
            temperature_celsius = prototype - T_FREEZE
            base_temperature_celsius = float(self.phenology_params.gdd_base_temp)
            day_fraction = time_step / torch.as_tensor(
                SECONDS_PER_DAY,
                device=prototype.device,
                dtype=prototype.dtype,
            )
            raw_phu_step = torch.clamp(
                temperature_celsius - base_temperature_celsius,
                min=0.0,
            ) * day_fraction

            if self.phenology_params.crop_vernalization_required:
                # References: Jones et al. (2003); Ritchie (1991).
                # Mathematical expressions: V_next = V + I(Tmin <= T_C <=
                # Tmax) * v_inc * dt/day and PHU_eff = PHU_raw *
                # clamp(V_next/V_required, f_v_min, f_v_max).
                # Parameter meanings: Tmin/Tmax bound the vernalizing window,
                # v_inc is the daily increment, V_required is the target, and
                # f_v_min is the configurable empirical response floor.
                # Units: temperature [degC], V [day], PHU [degC day].
                # Open route: vernalization is a non-material control state.
                vern_min = float(
                    self.phenology_params.crop_vernalization_temp_min
                )
                vern_max = float(
                    self.phenology_params.crop_vernalization_temp_max
                )
                temperature_celsius_scalar = _scalar_value(
                    temperature_celsius,
                    name="air_temperature_celsius",
                )
                if vern_min <= temperature_celsius_scalar <= vern_max:
                    vernalization_increment = torch.as_tensor(
                        self.get_parameter_value(
                            "biosphere.phenology.crop.vernalization_increment",
                            required=True,
                        ),
                        device=prototype.device,
                        dtype=prototype.dtype,
                    )
                    vernalization_days = (
                        vernalization_days
                        + vernalization_increment * day_fraction
                    )
                vernalization_required = float(
                    self.phenology_params.crop_vernalization_days_required
                )
                if vernalization_required <= 0.0:
                    raise ValueError(
                        "crop_vernalization_days_required must be positive "
                        "when vernalization is enabled"
                    )
                vernalization_min_factor = _scalar_value(
                    self.get_parameter_value(
                        "biosphere.phenology.crop.vernalization_min_factor",
                        required=True,
                    ),
                    name="crop.vernalization_min_factor",
                )
                vernalization_max_factor = _scalar_value(
                    self.get_parameter_value(
                        "biosphere.phenology.computational.clamp_max",
                        required=True,
                    ),
                    name="computational.clamp_max",
                )
                if vernalization_min_factor > vernalization_max_factor:
                    raise ValueError(
                        "crop.vernalization_min_factor must not exceed "
                        "computational.clamp_max"
                    )
                vernalization_factor = torch.clamp(
                    vernalization_days / vernalization_required,
                    min=vernalization_min_factor,
                    max=vernalization_max_factor,
                )
                daily_phu = raw_phu_step * vernalization_factor
            else:
                daily_phu = raw_phu_step
            accumulated_phu = accumulated_phu + daily_phu

        development_index = torch.clamp(
            accumulated_phu / phu_total,
            min=0.0,
            max=1.0,
        )
        harvest_index = self._compute_harvest_index(development_index)
        if not is_sown or is_harvested:
            phase = PhenologyPhase.DORMANCY
            lai = torch.as_tensor(
                self.get_parameter_value(
                    "biosphere.phenology.crop.initial_lai",
                    required=True,
                ),
                device=prototype.device,
                dtype=prototype.dtype,
            )
            growth_factor = torch.zeros_like(accumulated_phu)
            senescence_rate = torch.zeros_like(accumulated_phu)
            allocation_fractions = {}
        else:
            phase = self._get_development_stage(development_index)
            lai = self._compute_crop_lai(development_index)
            growth_factor = self._compute_crop_growth_factor(
                development_index,
                phase,
            )
            senescence_rate = self._compute_crop_senescence_rate(
                development_index,
                phase,
            )
            allocation_fractions = {}

        flag_sown = torch.as_tensor(
            float(is_sown),
            device=prototype.device,
            dtype=prototype.dtype,
        )
        flag_harvested = torch.as_tensor(
            float(is_harvested),
            device=prototype.device,
            dtype=prototype.dtype,
        )
        diagnostic_vars = {
            "development_index": development_index,
            "harvest_index": harvest_index,
            "daily_phu": daily_phu,
            "phu_total": torch.as_tensor(
                phu_total,
                device=prototype.device,
                dtype=prototype.dtype,
            ),
            "is_sown": flag_sown,
            "is_harvested": flag_harvested,
            "vernalization_days": vernalization_days,
        }
        additional_outputs = {
            "diagnostics.phenology.accumulated_phu": accumulated_phu,
            "harvest_index": harvest_index,
            "phenology_crop_accumulated_phu": accumulated_phu,
            "phenology_crop_is_sown": flag_sown,
            "phenology_crop_is_harvested": flag_harvested,
            "phenology_crop_vernalization_days": vernalization_days,
        }
        output = PhenologyOutput(
            phenology_phase=phase,
            growing_season_index=None,
            thermal_time=accumulated_phu,
            leaf_area_index=lai,
            growth_factor=growth_factor,
            senescence_rate=senescence_rate,
            allocation_fractions=allocation_fractions,
            diagnostic_variables=diagnostic_vars,
            additional_outputs=additional_outputs,
        )
        if self.enforce_physical_constraints:
            self._validate_output_bounds(output)
        return output
    
    def _get_development_stage(self, di: Tensor) -> PhenologyPhase:
        """Determine crop development stage from development index."""
        di_germination_end = self.get_parameter_value("biosphere.phenology.crop.di_germination_end")
        di_vegetative_end = self.get_parameter_value("biosphere.phenology.crop.di_vegetative_end")
        di_reproductive_end = self.get_parameter_value("biosphere.phenology.crop.di_reproductive_end")
        
        if di < di_germination_end:
            return PhenologyPhase.BUD_BREAK  # Germination/emergence
        elif di < di_vegetative_end:
            return PhenologyPhase.LEAF_EXPANSION  # Vegetative growth
        elif di < di_reproductive_end:
            return PhenologyPhase.FLOWERING  # Reproductive phase
        elif di < 1.0:
            return PhenologyPhase.SENESCENCE  # Grain filling/maturation
        else:
            return PhenologyPhase.ABSCISSION  # Mature/harvest ready
    
    def _compute_harvest_index(self, di: Tensor) -> Tensor:
        """Compute harvest index based on development index."""
        hi_start = self.phenology_params.crop_hi_start_fraction
        hi_max = self.phenology_params.crop_hi_max
        
        if di < hi_start:
            return torch.zeros(1, device=self.device).squeeze()
        else:
            # Linear increase from HI start to maximum
            hi_progress = (di - hi_start) / (1.0 - hi_start)
            return hi_max * torch.clamp(hi_progress, min=0.0, max=1.0)
    
    def _compute_crop_lai(self, di: Tensor) -> Tensor:
        """Compute LAI based on crop development index."""
        max_lai = self.phenology_params.max_lai        
        di_germination_end = self.get_parameter_value("biosphere.phenology.crop.di_germination_end")
        di_vegetative_peak = self.get_parameter_value("biosphere.phenology.crop.di_vegetative_peak")
        di_lai_peak_end = self.get_parameter_value("biosphere.phenology.crop.di_lai_peak_end")
        lai_growth_exponent = self.get_parameter_value("biosphere.phenology.crop.lai_growth_exponent")
        
        if di < di_germination_end:  # Germination
            initial_lai = self.get_parameter_value(
                'initial_lai',
                namespace="biosphere.phenology.crop",
            )
            return torch.as_tensor(initial_lai, device=self.device, dtype=torch.float32)
        elif di < di_vegetative_peak:  # Vegetative growth - exponential increase
            growth_factor = (di - di_germination_end) / (di_vegetative_peak - di_germination_end)
            return torch.as_tensor(max_lai, device=self.device, dtype=torch.float32) * growth_factor ** lai_growth_exponent
        elif di < di_lai_peak_end:  # Peak LAI maintained
            return torch.as_tensor(max_lai, device=self.device, dtype=torch.float32)
        else:  # Senescence - linear decrease
            senescence_factor = (1.0 - di) / (1.0 - di_lai_peak_end)
            clamp_min = self.get_parameter_value("biosphere.phenology.computational.clamp_min")
            clamp_max = self.get_parameter_value("biosphere.phenology.computational.clamp_max")
            return torch.as_tensor(max_lai, device=self.device, dtype=torch.float32) * torch.clamp(senescence_factor, min=clamp_min, max=clamp_max)
    
    def _compute_crop_growth_factor(self, di: Tensor, phase: PhenologyPhase) -> Tensor:
        """Compute growth activity factor for crops."""
        if phase == PhenologyPhase.DORMANCY:
            return torch.zeros(1, device=self.device).squeeze()
        elif phase == PhenologyPhase.BUD_BREAK:
            germination_activity = self.get_parameter_value(
                'germination_activity',
                namespace="biosphere.phenology.activity",
            )
            return torch.as_tensor(germination_activity, device=self.device, dtype=torch.float32)  # Low activity during germination
        elif phase in [PhenologyPhase.LEAF_EXPANSION, PhenologyPhase.FLOWERING]:
            vegetative_activity = self.get_parameter_value(
                'vegetative_activity',
                namespace="biosphere.phenology.activity",
            )
            return torch.as_tensor(vegetative_activity, device=self.device, dtype=torch.float32)  # Maximum growth activity
        elif phase == PhenologyPhase.SENESCENCE:
            # Decreasing growth as crop matures
            senescence_growth_factor = self.get_parameter_value("biosphere.phenology.crop.senescence_growth_factor")
            senescence_min_factor = self.get_parameter_value("biosphere.phenology.crop.senescence_min_factor")
            clamp_max = self.get_parameter_value("biosphere.phenology.computational.clamp_max")
            return torch.clamp(senescence_growth_factor * (1.0 - di), min=senescence_min_factor, max=clamp_max)
        else:
            initial_lai = self.get_parameter_value(
                'initial_lai',
                namespace="biosphere.phenology.crop",
            )
            return torch.as_tensor(initial_lai, device=self.device, dtype=torch.float32)
    
    def _compute_crop_senescence_rate(self, di: Tensor, phase: PhenologyPhase) -> Tensor:
        """Compute senescence rate for crops."""
        if phase == PhenologyPhase.SENESCENCE:
            # Increasing senescence during grain filling
            di_reproductive_end = self.get_parameter_value("biosphere.phenology.crop.di_reproductive_end")
            grain_filling_factor = self.get_parameter_value("biosphere.phenology.crop.grain_filling_factor")
            clamp_min = self.get_parameter_value("biosphere.phenology.computational.clamp_min")
            clamp_max = self.get_parameter_value("biosphere.phenology.computational.clamp_max")
            return torch.clamp(grain_filling_factor * (di - di_reproductive_end) / (1.0 - di_reproductive_end), min=clamp_min, max=clamp_max)
        elif phase == PhenologyPhase.ABSCISSION:
            maturity_senescence = self.get_parameter_value(
                'maturity_senescence',
                namespace="biosphere.phenology.activity",
            )
            return torch.as_tensor(maturity_senescence, device=self.device, dtype=torch.float32)  # Maximum senescence at maturity
        else:
            initial_lai = self.get_parameter_value(
                'initial_lai',
                namespace="biosphere.phenology.crop",
            )
            return torch.as_tensor(initial_lai, device=self.device, dtype=torch.float32)  # Background senescence
    
# Export all concrete models
__all__ = [
    'StaticPhenology',
    'GDDPhenology', 
    'GSIPhenology',
    'CropPhenology'
]


