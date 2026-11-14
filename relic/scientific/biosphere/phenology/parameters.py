"""
Phenology Parameters for RELIC Framework

This module defines comprehensive parameter models for phenological processes
using Pydantic for type safety and validation. Parameters are organized by
phenology type and follow CF-1.8 conventions for units and naming.

All parameters MUST be loaded from the ParameterManager - no default values
are allowed to ensure consistency with the centralized parameter system.

References:
    - Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict 
      foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.
    - White, M.A., et al. (1997). A continental phenology model for monitoring 
      vegetation responses to interannual climatic variability. 
      Global Biogeochemical Cycles, 11(2), 217-234.
    - Richardson, A.D., et al. (2013). Climate change, phenology, and phenological 
      control of vegetation feedbacks to the climate system. 
      Agricultural and Forest Meteorology, 169, 156-173.
"""

from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import torch

from relic.parameters import get_global_manager
from relic.utils.time_constants import SECONDS_PER_DAY


PHENOLOGY_PARAMETER_FIELD_KEYS = {
    "phenology_type": "biosphere.phenology.common.phenology_type",
    "leaf_onset_doy": "biosphere.phenology.static.leaf_onset_doy",
    "leaf_offset_doy": "biosphere.phenology.static.leaf_offset_doy",
    "leaf_expansion_days": "biosphere.phenology.static.leaf_expansion_days",
    "leaf_senescence_days": "biosphere.phenology.static.leaf_senescence_days",
    "gdd_base_temp": "biosphere.phenology.gdd.base_temp",
    "gdd_threshold_start": "biosphere.phenology.gdd.threshold_start",
    "gdd_threshold_end": "biosphere.phenology.gdd.threshold_end",
    "gdd_max_temp_limit": "biosphere.phenology.gdd.max_temp_limit",
    "gsi_tmin_threshold": "biosphere.phenology.gsi.tmin_threshold",
    "gsi_tmin_optimal": "biosphere.phenology.gsi.tmin_optimal",
    "gsi_vpd_min": "biosphere.phenology.gsi.vpd_min",
    "gsi_vpd_max": "biosphere.phenology.gsi.vpd_max",
    "gsi_daylen_threshold": "biosphere.phenology.gsi.daylen_threshold",
    "gsi_onset_threshold": "biosphere.phenology.gsi.onset_threshold",
    "gsi_offset_threshold": "biosphere.phenology.gsi.offset_threshold",
    "gsi_memory_days": "biosphere.phenology.gsi.memory_days",
    "crop_type": "biosphere.phenology.crop.default_type",
    "crop_phu_total": "biosphere.phenology.crop.phu_total",
    "crop_hi_max": "biosphere.phenology.crop.hi_max",
    "crop_hi_start_fraction": "biosphere.phenology.crop.hi_start_fraction",
    "crop_sowing_doy": "biosphere.phenology.crop.sowing_doy",
    "crop_harvest_doy": "biosphere.phenology.crop.harvest_doy",
    "crop_vernalization_required": (
        "biosphere.phenology.crop.vernalization_required"
    ),
    "crop_vernalization_temp_min": (
        "biosphere.phenology.crop.vernalization_temp_min"
    ),
    "crop_vernalization_temp_max": (
        "biosphere.phenology.crop.vernalization_temp_max"
    ),
    "crop_vernalization_days_required": (
        "biosphere.phenology.crop.vernalization_days_required"
    ),
    "leaf_turnover_rate": "biosphere.phenology.turnover.leaf_turnover_rate",
    "froot_turnover_rate": "biosphere.phenology.turnover.froot_turnover_rate",
    "max_lai": "biosphere.phenology.morphology.max_lai",
    "min_lai": "biosphere.phenology.morphology.min_lai",
    "numerical_tolerance": "biosphere.validation.tolerance.phenology",
}

PHENOLOGY_PARAMETER_FIELD_UNITS = {
    "gdd_base_temp": "degC",
    "gsi_tmin_threshold": "degC",
    "gsi_tmin_optimal": "degC",
    "crop_vernalization_temp_min": "degC",
    "crop_vernalization_temp_max": "degC",
}

# Container fields each phenology model reads. Every model reads the common fields; a model
# reads its own fields and no field of another model (the crop model also reads the GDD base
# temperature of its thermal time). PhenologyParameters holds exactly the set of its
# phenology_type, and PhenologyBase reads and declares only that set.
PHENOLOGY_COMMON_FIELDS: Tuple[str, ...] = (
    "phenology_type",
    "max_lai",
    "min_lai",
    "leaf_turnover_rate",
    "froot_turnover_rate",
    "numerical_tolerance",
)
PHENOLOGY_MODEL_FIELDS: Dict[str, Tuple[str, ...]] = {
    "static": (
        "leaf_onset_doy",
        "leaf_offset_doy",
        "leaf_expansion_days",
        "leaf_senescence_days",
    ),
    "gdd": (
        "gdd_base_temp",
        "gdd_threshold_start",
        "gdd_threshold_end",
        "gdd_max_temp_limit",
    ),
    "gsi": (
        "gsi_tmin_threshold",
        "gsi_tmin_optimal",
        "gsi_vpd_min",
        "gsi_vpd_max",
        "gsi_daylen_threshold",
        "gsi_onset_threshold",
        "gsi_offset_threshold",
        "gsi_memory_days",
    ),
    "crop": (
        "gdd_base_temp",
        "crop_type",
        "crop_phu_total",
        "crop_hi_max",
        "crop_hi_start_fraction",
        "crop_sowing_doy",
        "crop_harvest_doy",
        "crop_vernalization_required",
        "crop_vernalization_temp_min",
        "crop_vernalization_temp_max",
        "crop_vernalization_days_required",
    ),
}


def phenology_field_names(phenology_type: str) -> Tuple[str, ...]:
    """Container fields of one phenology model: the common fields and its own fields.

    Raises
    ------
    ValueError
        If ``phenology_type`` names no phenology model.
    """
    if phenology_type not in PHENOLOGY_MODEL_FIELDS:
        raise ValueError(
            f"Unknown phenology_type '{phenology_type}'; "
            f"expected one of {sorted(PHENOLOGY_MODEL_FIELDS)}"
        )
    return (*PHENOLOGY_COMMON_FIELDS, *PHENOLOGY_MODEL_FIELDS[phenology_type])


# Declared meaning and [database units] of the container field keys (a model reads the keys
# of phenology_field_names(<its type>) when no container is given; the phenology_type key
# only by the generic base, whose type no class fixes) and of the keys the base reads
# itself. The models declare their model-specific keys.
PHENOLOGY_FIELD_PARAMETER_DESCRIPTIONS = {
    "biosphere.phenology.common.phenology_type": (
        "Phenology model type of the generic base (static, gdd, gsi, crop) "
        "[dimensionless]"
    ),
    "biosphere.phenology.static.leaf_onset_doy": "Day of year of leaf onset [day]",
    "biosphere.phenology.static.leaf_offset_doy": "Day of year of leaf offset [day]",
    "biosphere.phenology.static.leaf_expansion_days": "Duration of leaf expansion [day]",
    "biosphere.phenology.static.leaf_senescence_days": "Duration of leaf senescence [day]",
    "biosphere.phenology.gdd.base_temp": "Base temperature of the growing degree days [K]",
    "biosphere.phenology.gdd.threshold_start": "Growing degree days of leaf onset [degC_day]",
    "biosphere.phenology.gdd.threshold_end": "Growing degree days of senescence [degC_day]",
    "biosphere.phenology.gdd.max_temp_limit": (
        "Upper temperature cap of the growing degree days [degC]"
    ),
    "biosphere.phenology.gsi.tmin_threshold": (
        "Minimum temperature at which the GSI temperature factor is zero [K]"
    ),
    "biosphere.phenology.gsi.tmin_optimal": (
        "Minimum temperature at which the GSI temperature factor is one [K]"
    ),
    "biosphere.phenology.gsi.vpd_min": "VPD at which the GSI VPD factor is one [Pa]",
    "biosphere.phenology.gsi.vpd_max": "VPD at which the GSI VPD factor is zero [Pa]",
    "biosphere.phenology.gsi.daylen_threshold": (
        "Day length below which the GSI photoperiod factor is zero [s]"
    ),
    "biosphere.phenology.gsi.onset_threshold": "GSI threshold of leaf onset [dimensionless]",
    "biosphere.phenology.gsi.offset_threshold": "GSI threshold of leaf offset [dimensionless]",
    "biosphere.phenology.gsi.memory_days": (
        "Number of daily samples in the GSI moving average [day]"
    ),
    "biosphere.phenology.crop.default_type": "Default crop type [dimensionless]",
    "biosphere.phenology.crop.phu_total": "Potential heat units to crop maturity [degC day]",
    "biosphere.phenology.crop.hi_max": "Maximum harvest index [dimensionless]",
    "biosphere.phenology.crop.hi_start_fraction": (
        "Development index at which the harvest index starts to accumulate "
        "[dimensionless]"
    ),
    "biosphere.phenology.crop.sowing_doy": "Sowing day of year [day]",
    "biosphere.phenology.crop.harvest_doy": "Harvest day of year [day]",
    "biosphere.phenology.crop.vernalization_required": (
        "Switch of the vernalization requirement [dimensionless]"
    ),
    "biosphere.phenology.crop.vernalization_temp_min": (
        "Lower temperature of effective vernalization [K]"
    ),
    "biosphere.phenology.crop.vernalization_temp_max": (
        "Upper temperature of effective vernalization [K]"
    ),
    "biosphere.phenology.crop.vernalization_days_required": "Vernalization days required [day]",
    "biosphere.phenology.turnover.leaf_turnover_rate": "Leaf turnover rate [s-1]",
    "biosphere.phenology.turnover.froot_turnover_rate": "Fine-root turnover rate [s-1]",
    "biosphere.phenology.morphology.max_lai": "Maximum leaf area index [dimensionless]",
    "biosphere.phenology.morphology.min_lai": "Minimum (dormant) leaf area index [dimensionless]",
    "biosphere.validation.tolerance.phenology": (
        "Numerical tolerance of the phenology checks [dimensionless]"
    ),
}

PHENOLOGY_BASE_PARAMETER_DESCRIPTIONS = {
    "biosphere.phenology.turnover.n_retrans_efficiency": (
        "Leaf nitrogen retranslocation efficiency at senescence [dimensionless]"
    ),
    "biosphere.phenology.constants.max_lai_threshold": (
        "Largest plausible leaf area index of the LAI check [m2 m-2]"
    ),
}

def _read_parameter(
    key: str,
    *,
    units: Optional[str] = None,
    expected_type: Optional[type] = None,
):
    """
    Read a required parameter from UnifiedParameterManager.
    """
    value = get_global_manager().get(
        key,
        units=units,
        required=True,
        expected_type=None,
    )
    if expected_type is not None and value is not None and not isinstance(value, expected_type):
        try:
            value = expected_type(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Parameter {key} cannot be cast to {expected_type}: {exc}") from exc
    return value


# Type definitions
PhenologyType = Literal["static", "gdd", "gsi", "crop"]
CropType = Literal["wheat", "maize", "soybean", "rice"]


class PhenologyParameters(BaseModel):
    """
    Comprehensive phenology parameter model for RELIC framework.
    
    This model defines all parameters needed for different phenology types
    including static date-based, growing degree days (GDD), growing season
    index (GSI), and crop-specific phenology models.
    
    All parameters follow SI units and CF-1.8 conventions where applicable.
    All parameters MUST be loaded from ParameterManager - no hardcoded values.
    
    Mathematical Expressions:
        Static Phenology: Fixed calendar dates
        GDD: Sigma(T_daily - T_base) where T_daily > T_base
        GSI: f_temp * f_vpd * f_daylen * f_swp (Jolly et al. 2005)
        Crop PHU: Sigma(T_daily - T_base_crop) for thermal time accumulation
    
    The container holds exactly the fields of its phenology_type
    (phenology_field_names): the common fields and the fields of that model. A field of
    another model is absent (None) and supplying one is rejected, as is omitting a field
    of the model; no field takes a value the model does not read.

    Attributes:
        phenology_type : str
            Type of phenology model to use
            
    References:
        - CF Convention: http://cfconventions.org/cf-conventions/cf-conventions.html
        - Jolly et al. (2005) Global Change Biology 11:619-632
        - Jones, J.W., et al. (2003). The DSSAT cropping system model. 
          European Journal of Agronomy, 18(3-4), 235-265.
    """
    
    # ========== Core Configuration ==========
    phenology_type: PhenologyType = Field(
        ...,
        description="Phenology model type"
    )
    
    # ========== Static Phenology Parameters ==========
    leaf_onset_doy: Optional[int] = Field(
        None,
        ge=1,
        le=366,
        description="Day of year for leaf onset (static phenology)",
        json_schema_extra={"unit": "day"}
    )
    
    leaf_offset_doy: Optional[int] = Field(
        None,
        ge=1, 
        le=366,
        description="Day of year for leaf offset (static phenology)",
        json_schema_extra={"unit": "day"}
    )
    
    leaf_expansion_days: Optional[int] = Field(
        None,
        ge=1,
        le=180,
        description="Duration of leaf expansion period",
        json_schema_extra={"unit": "day"}
    )
    
    leaf_senescence_days: Optional[int] = Field(
        None,
        ge=1,
        le=180, 
        description="Duration of leaf senescence period",
        json_schema_extra={"unit": "day"}
    )
    
    # ========== GDD Parameters ==========
    gdd_base_temp: Optional[float] = Field(
        None,
        ge=-10.0,
        le=25.0,
        description="Base temperature for GDD calculation",
        json_schema_extra={"unit": "degC"}
    )
    
    gdd_threshold_start: Optional[float] = Field(
        None,
        ge=0.0,
        le=3000.0,
        description="GDD threshold for growth initiation",
        json_schema_extra={"unit": "degC day"}
    )
    
    gdd_threshold_end: Optional[float] = Field(
        None,
        ge=0.0,
        le=5000.0,
        description="GDD threshold for growth cessation", 
        json_schema_extra={"unit": "degC day"}
    )
    
    gdd_max_temp_limit: Optional[float] = Field(
        None,
        ge=25.0,
        le=50.0,
        description="Maximum temperature limit for GDD calculation",
        json_schema_extra={"unit": "degC"}
    )
    
    # ========== GSI Parameters (Jolly et al. 2005) ==========
    gsi_tmin_threshold: Optional[float] = Field(
        None,
        ge=-20.0,
        le=10.0,
        description="Minimum temperature threshold for GSI",
        json_schema_extra={"unit": "degC"}
    )
    
    gsi_tmin_optimal: Optional[float] = Field(
        None,
        ge=0.0,
        le=15.0,
        description="Optimal minimum temperature for GSI",
        json_schema_extra={"unit": "degC"}
    )
    
    gsi_vpd_min: Optional[float] = Field(
        None,
        ge=0.0,
        le=5000.0,
        description="Minimum VPD for optimal conditions",
        json_schema_extra={"unit": "Pa"}
    )
    
    gsi_vpd_max: Optional[float] = Field(
        None,
        ge=1000.0,
        le=8000.0,
        description="Maximum VPD threshold",
        json_schema_extra={"unit": "Pa"}
    )
    
    gsi_daylen_threshold: Optional[float] = Field(
        None,
        ge=28800.0,
        le=SECONDS_PER_DAY,
        description="Daylight length threshold for GSI",
        json_schema_extra={"unit": "s"}
    )
    
    gsi_onset_threshold: Optional[float] = Field(
        None,
        ge=0.1,
        le=1.0,
        description="GSI threshold for leaf onset",
        json_schema_extra={"unit": "dimensionless"}
    )
    
    gsi_offset_threshold: Optional[float] = Field(
        None,
        ge=0.1,
        le=1.0,
        description="GSI threshold for leaf offset",
        json_schema_extra={"unit": "dimensionless"}
    )
    
    gsi_memory_days: Optional[int] = Field(
        None,
        ge=1,
        le=30,
        description="Number of days for GSI moving average",
        json_schema_extra={"unit": "day"}
    )
    
    # ========== Crop-Specific Parameters ==========
    crop_type: Optional[CropType] = Field(
        default=None,
        description="Specific crop type for specialized parameters"
    )
    
    crop_phu_total: Optional[float] = Field(
        None,
        ge=500.0,
        le=3000.0,
        description="Total potential heat units required for crop maturation",
        json_schema_extra={"unit": "degC day"}
    )
    
    crop_hi_max: Optional[float] = Field(
        None,
        ge=0.2,
        le=0.8,
        description="Maximum harvest index",
        json_schema_extra={"unit": "dimensionless"}
    )
    
    crop_hi_start_fraction: Optional[float] = Field(
        None,
        ge=0.2,
        le=0.8,
        description="PHU fraction at which HI accumulation starts",
        json_schema_extra={"unit": "dimensionless"}
    )
    
    crop_sowing_doy: Optional[int] = Field(
        None,
        ge=1,
        le=366,
        description="Day of year for crop sowing",
        json_schema_extra={"unit": "day"}
    )
    
    crop_harvest_doy: Optional[int] = Field(
        None,
        ge=1,
        le=366,
        description="Day of year for crop harvest",
        json_schema_extra={"unit": "day"}
    )
    
    crop_vernalization_required: Optional[bool] = Field(
        None,
        description="Whether crop requires vernalization"
    )
    
    crop_vernalization_temp_min: Optional[float] = Field(
        None,
        ge=-10.0,
        le=5.0,
        description="Minimum temperature for vernalization",
        json_schema_extra={"unit": "degC"}
    )
    
    crop_vernalization_temp_max: Optional[float] = Field(
        None,
        ge=0.0,
        le=15.0,
        description="Maximum temperature for vernalization",
        json_schema_extra={"unit": "degC"}
    )
    
    crop_vernalization_days_required: Optional[int] = Field(
        None,
        ge=0,
        le=120,
        description="Days of vernalization required",
        json_schema_extra={"unit": "day"}
    )
    
    # ========== Biomass Turnover Parameters ==========
    # Conversion: year-1 to s-1 by dividing by 31536000 (seconds per year)
    # Original bounds in year-1: [0.1, 2.0]
    # Converted bounds in s-1: [3.171e-9, 6.342e-8]
    leaf_turnover_rate: float = Field(
        ...,
        ge=3.171e-9,  # 0.1 year-1 converted to s-1
        le=6.342e-8,  # 2.0 year-1 converted to s-1
        description="Annual leaf turnover rate (SI units: s-1)",
        json_schema_extra={"unit": "s-1"}
    )
    
    # Conversion: year-1 to s-1
    # Original bounds in year-1: [0.5, 3.0]
    # Converted bounds in s-1: [1.585e-8, 9.512e-8]
    froot_turnover_rate: float = Field(
        ...,
        ge=1.585e-8,  # 0.5 year-1 converted to s-1
        le=9.512e-8,  # 3.0 year-1 converted to s-1
        description="Annual fine root turnover rate (SI units: s-1)",
        json_schema_extra={"unit": "s-1"}
    )
    
    # ========== Morphological Parameters ==========
    max_lai: float = Field(
        ...,
        ge=0.5,
        le=15.0,
        description="Maximum leaf area index",
        json_schema_extra={"unit": "m2 m-2"}
    )

    min_lai: float = Field(
        ...,
        ge=0.01,
        le=1.0,
        description="Minimum leaf area index",
        json_schema_extra={"unit": "m2 m-2"}
    )

    # ========== Numerical Configuration ==========
    numerical_tolerance: float = Field(
        ...,
        ge=1e-12,
        le=1e-6,
        description="Numerical tolerance for computations",
        json_schema_extra={"unit": "dimensionless"}
    )
    
    # ========== Validators ==========
    @field_validator('leaf_offset_doy')
    @classmethod
    def offset_after_onset(cls, v, info):
        """Ensure leaf offset follows onset within the calendar year"""
        if v is None:
            return v
        if v <= 0:
            raise ValueError('leaf_offset_doy must be positive')
        onset = info.data.get('leaf_onset_doy')
        if onset is not None and v <= onset:
            raise ValueError('leaf_offset_doy must be greater than leaf_onset_doy')
        if v > 366:
            raise ValueError('leaf_offset_doy must be within the calendar year')
        return v
    
    @field_validator('gdd_threshold_end')
    @classmethod
    def gdd_end_after_start(cls, v, info):
        """Ensure GDD end threshold is after start"""
        if v is None:
            return v
        if info.data.get('gdd_threshold_start') and v <= info.data['gdd_threshold_start']:
            raise ValueError('gdd_threshold_end must be greater than gdd_threshold_start')
        return v
    
    @field_validator('gsi_tmin_optimal')
    @classmethod
    def tmin_optimal_above_threshold(cls, v, info):
        """Ensure optimal temperature is above threshold"""
        if v is None:
            return v
        if info.data.get('gsi_tmin_threshold') and v <= info.data['gsi_tmin_threshold']:
            raise ValueError('gsi_tmin_optimal must be greater than gsi_tmin_threshold')
        return v
    
    @field_validator('gsi_vpd_max')
    @classmethod
    def vpd_max_above_min(cls, v, info):
        """Ensure VPD max is above min"""
        if v is None:
            return v
        if info.data.get('gsi_vpd_min') and v <= info.data['gsi_vpd_min']:
            raise ValueError('gsi_vpd_max must be greater than gsi_vpd_min')
        return v
    
    @field_validator('crop_harvest_doy')
    @classmethod
    def harvest_after_sowing(cls, v, info):
        """Ensure harvest occurs after sowing (allows for cross-year crops)"""
        if v is None:
            return v
        sowing = info.data.get('crop_sowing_doy')
        if sowing is not None and v <= sowing:
            allow_cross_year = (
                info.data.get('phenology_type') == 'crop'
                and sowing >= 200
                and v <= 200
            )
            if not allow_cross_year:
                raise ValueError('crop_harvest_doy must be greater than crop_sowing_doy')
        return v
    
    @field_validator('max_lai')
    @classmethod
    def max_lai_above_min(cls, v, info):
        """Ensure max LAI is above min LAI"""
        if info.data.get('min_lai') and v <= info.data['min_lai']:
            raise ValueError('max_lai must be greater than min_lai')
        return v

    @field_validator('min_lai')
    @classmethod
    def min_lai_below_max(cls, v, info):
        """Ensure min LAI remains below max LAI."""
        max_lai = info.data.get('max_lai')
        if max_lai is not None and v >= max_lai:
            raise ValueError('max_lai must be greater than min_lai')
        return v

    @model_validator(mode="after")
    def fields_match_phenology_type(self) -> "PhenologyParameters":
        """Hold exactly the fields of the configured model (phenology_field_names)."""
        expected = set(phenology_field_names(self.phenology_type))
        model_fields = {name for fields in PHENOLOGY_MODEL_FIELDS.values() for name in fields}
        missing = sorted(name for name in expected if getattr(self, name) is None)
        if missing:
            raise ValueError(
                f"{self.phenology_type} phenology requires the fields {missing}"
            )
        foreign = sorted(
            name for name in model_fields - expected if getattr(self, name) is not None
        )
        if foreign:
            raise ValueError(
                f"The fields {foreign} are not parameters of the "
                f"{self.phenology_type} phenology model"
            )
        return self
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True
    )
        
    def to_device(self, device: torch.device) -> 'PhenologyParameters':
        """
        Convert numerical parameters to specified device.
        
        Parameters:
            device : torch.device
                Target device for tensor operations
                
        Returns:
            PhenologyParameters: New instance with parameters on specified device
        """
        # For now, return self since Pydantic models store Python types
        # In the future, could extend to support torch.Tensor parameters
        return self
    
    def validate_for_phenology_type(self) -> None:
        """
        Validate parameter combinations based on phenology type.
        
        Raises:
            ValueError: If parameter combination is invalid for the selected type
        """
        if self.phenology_type == "crop" and self.crop_type is None:
            raise ValueError("crop_type must be specified when using crop phenology")
            
        if self.phenology_type == "gdd" and self.gdd_threshold_start >= self.gdd_threshold_end:
            raise ValueError("Invalid GDD threshold configuration")
            
        if self.phenology_type == "gsi":
            if self.gsi_tmin_optimal <= self.gsi_tmin_threshold:
                raise ValueError("Invalid GSI temperature configuration")
            if self.gsi_vpd_max <= self.gsi_vpd_min:
                raise ValueError("Invalid GSI VPD configuration")


class PhenologyParameterSets:
    """Factory class for predefined phenology parameter sets.
    
    This class provides convenient methods to create commonly-used parameter
    configurations for different vegetation types and phenological models.
    All parameter values are loaded from ParameterManager to ensure consistency
    with the centralized parameter system.
    
    Examples:
        >>> # Get parameters for a deciduous tree using GSI model
        >>> params = PhenologyParameterSets.deciduous_tree_gsi()
        >>> 
        >>> # Get parameters for grasslands using GDD model  
        >>> params = PhenologyParameterSets.grassland_gdd()
        
    References:
        - White, M.A. et al. (1997). A continental phenology model for monitoring 
          vegetation responses to interannual climatic variability. 
          Global Biogeochemical Cycles, 11(2), 217-234.
        - Jolly, W.M. et al. (2005). A generalized, bioclimatic index to predict 
          foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.
        - McMaster, G.S. & Wilhelm, W.W. (1997). Growing degree-days: one equation, 
          two interpretations. Agricultural and Forest Meteorology, 87(4), 291-300.
        - Reich, P.B. (1995). Phenology of tropical forests: patterns, causes, 
          and consequences. Canadian Journal of Botany, 73(2), 164-174.
    """
    
    @staticmethod
    def _load_preset_params(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PhenologyParameters:
        """Load a preset parameter set from ParameterManager.

        The preset namespace ``biosphere.phenology.presets.<preset_name>`` declares the
        phenology_type of the preset and the fields it specializes. Every entry must be a
        field of that model (phenology_field_names); a field of the model that the preset
        does not declare takes the value of its generic phenology key. Preset and generic
        entries are read with the field units, so a read failure raises.

        Args:
            preset_name: Name of the preset to load (e.g., "deciduous_tree_gsi")
            overrides: Optional dictionary of field overrides applied last

        Returns:
            PhenologyParameters instance with loaded values

        Raises:
            ValueError: If the preset declares no phenology_type or declares an entry that
                is not a field of its model; a failing read raises from the manager.
        """
        namespace = f"biosphere.phenology.presets.{preset_name}"
        prefix = f"{namespace}."
        declared = {
            key[len(prefix):]
            for key in get_global_manager().get_namespace_parameters(
                namespace,
                include_subnamespaces=False,
                return_specs=False,
            )
        }
        if "phenology_type" not in declared:
            raise ValueError(
                f"Preset '{preset_name}' not properly configured in parameter database. "
                f"Expected parameters under namespace '{namespace}'"
            )
        phenology_type = _read_parameter(f"{prefix}phenology_type", expected_type=str)
        field_names = phenology_field_names(phenology_type)
        foreign = sorted(declared - set(field_names))
        if foreign:
            raise ValueError(
                f"Preset '{preset_name}' declares {foreign}, which are not fields of the "
                f"{phenology_type} phenology model"
            )

        string_fields = {"phenology_type", "crop_type"}
        params_dict: Dict[str, Any] = {}
        for field_name in field_names:
            key = (
                f"{prefix}{field_name}"
                if field_name in declared
                else PHENOLOGY_PARAMETER_FIELD_KEYS[field_name]
            )
            if field_name in string_fields:
                params_dict[field_name] = _read_parameter(key, expected_type=str)
            else:
                params_dict[field_name] = _read_parameter(
                    key,
                    units=PHENOLOGY_PARAMETER_FIELD_UNITS.get(field_name),
                )

        # Apply any overrides
        if overrides:
            params_dict.update(overrides)

        return PhenologyParameters(**params_dict)
    
    @staticmethod
    def deciduous_tree_gsi() -> PhenologyParameters:
        """Parameter set for deciduous trees using GSI phenology.
        
        Loads parameters from biosphere.phenology.presets.deciduous_tree_gsi namespace.
        Raises ValueError if parameters are not properly configured in ParameterManager.
        """
        return PhenologyParameterSets._load_preset_params("deciduous_tree_gsi")
    
    @staticmethod  
    def evergreen_tree_static() -> PhenologyParameters:
        """Parameter set for evergreen trees using static phenology.
        
        Loads parameters from biosphere.phenology.presets.evergreen_tree_static namespace.
        Raises ValueError if parameters are not properly configured in ParameterManager.
        """
        return PhenologyParameterSets._load_preset_params("evergreen_tree_static")
    
    @staticmethod
    def grassland_gdd() -> PhenologyParameters:
        """Parameter set for grasslands using GDD phenology.
        
        Loads parameters from biosphere.phenology.presets.grassland_gdd namespace.
        Raises ValueError if parameters are not properly configured in ParameterManager.
        """
        return PhenologyParameterSets._load_preset_params("grassland_gdd")
    
    @staticmethod
    def winter_wheat_crop() -> PhenologyParameters:
        """Parameter set for winter wheat using crop phenology.
        
        Loads parameters from biosphere.phenology.presets.winter_wheat_crop namespace.
        Raises ValueError if parameters are not properly configured in ParameterManager.
        """
        return PhenologyParameterSets._load_preset_params("winter_wheat_crop")
    
    @staticmethod
    def maize_crop() -> PhenologyParameters:
        """Parameter set for maize using crop phenology.
        
        Loads parameters from biosphere.phenology.presets.maize_crop namespace.
        Raises ValueError if parameters are not properly configured in ParameterManager.
        """
        return PhenologyParameterSets._load_preset_params("maize_crop")


# Export all classes
__all__ = [
    'PHENOLOGY_COMMON_FIELDS',
    'PHENOLOGY_MODEL_FIELDS',
    'PHENOLOGY_PARAMETER_FIELD_KEYS',
    'PHENOLOGY_PARAMETER_FIELD_UNITS',
    'PhenologyParameters',
    'PhenologyParameterSets', 
    'PhenologyType',
    'CropType',
    'phenology_field_names',
]




