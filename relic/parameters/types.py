"""
Shared types and enums for the parameter management system.

This module contains shared type definitions, enums, and dataclasses
used across the parameter management system to avoid circular imports.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Union, Callable
from torch import Tensor

# Type aliases
ParameterValue = Union[float, int, str, bool, Tensor]
ParameterDict = Dict[str, ParameterValue]
MetadataDict = Dict[str, Any]
ValidationErrors = List[str]
ChangeCallback = Callable[[str, ParameterValue, ParameterValue], None]


class ParameterCategory(Enum):
    """Scientific parameter categories for organization and validation."""
    PHYSICAL = "physical"
    BIOPHYSICAL = "biophysical"
    ENVIRONMENTAL = "environmental"
    COMPUTATIONAL = "computational"
    GENERAL = "general"  # General-purpose parameter category
    CALIBRATION = "calibration"
    LEARNABLE = "learnable"
    
    # Additional category names for parameter classification
    PHYSICAL_CONSTANT = "physical_constant"
    REFERENCE_CONSTANT = "reference_constant"  # Standard reference values
    BIOLOGICAL_PARAMETER = "biological_parameter"
    ENVIRONMENTAL_VARIABLE = "environmental_variable"
    ENVIRONMENTAL_PARAMETER = "environmental_parameter"  # Alias for environmental variables
    PHYSICAL_PARAMETER = "physical_parameter"  # Alias for physical constants
    MODEL_PARAMETER = "model_parameter"
    CALIBRATION_PARAMETER = "calibration_parameter"
    DERIVED_PARAMETER = "derived_parameter"
    
    # Additional categories for hydrosphere and other modules
    default_VALUE = "default_value"
    NUMERICAL_default = "numerical_default"
    DIAGNOSTIC_PARAMETER = "diagnostic_parameter"
    # State variable and default value categories
    STATE_VARIABLE = "state_variable"
    STATE_DEFAULT = "state_default"
    INITIAL_CONDITION = "initial_condition"

    # Process and rate categories
    PROCESS_PARAMETER = "process_parameter"
    PROCESS_RATE = "process_rate"
    PROCESS_COUPLING = "process_coupling"

    # Pool and storage categories
    POOL_SIZE = "pool_size"
    POOL_PARAMETER = "pool_parameter"

    # Constraint and flag categories
    CONSTRAINT_FLAG = "constraint_flag"
    CONSTRAINT_PARAMETER = "constraint_parameter"

    # Configuration categories
    AGGREGATION_PARAMETER = "aggregation_parameter"
    WEIGHTING_PARAMETER = "weighting_parameter"
    PHYSICAL_CONSTRAINT = "physical_constraint"
    TEMPORAL_FLAG = "temporal_flag"
    TEMPORAL_CONSTANT = "temporal_constant"
    SPATIAL_CONFIGURATION = "spatial_configuration"
    NUMERICAL_PARAMETER = "numerical_parameter"

    # Additional missing categories
    TEMPORAL_PARAMETER = "temporal_parameter"
    SPATIAL_PARAMETER = "spatial_parameter"
    SCALING_PARAMETER = "scaling_parameter"

    SOLUTE_TRANSPORT_PARAMETER = "solute_transport_parameter"
    BOUNDARY_CONDITION = "boundary_condition"
    MODEL_CONFIGURATION = "model_configuration"
    GAS_DIFFUSION_PARAMETER = "gas_diffusion_parameter"
    NUMERICAL_STABILITY = "numerical_stability"
    CONVERGENCE_CRITERION = "convergence_criterion"
    ITERATION_LIMIT = "iteration_limit"
    PHYSICAL_BOUNDS = "physical_bounds"
    ERROR_THRESHOLD = "error_threshold"
    VALIDATION_PARAMETER = "validation_parameter"
    SPATIAL_COORDINATE = "spatial_coordinate"
    CONSERVATION_CHECK = "conservation_check"
    NUMERICAL_THRESHOLD = "numerical_threshold"
    NUMERICAL_TOLERANCE = "numerical_tolerance"
    NUMERICAL_CONSTRAINT = "numerical_constraint"
    NUMERICAL_METHOD = "numerical_method"
    NUMERICAL_SCALING = "numerical_scaling"
    EMPIRICAL_CONSTANT = "empirical_constant"
    EMPIRICAL_PARAMETER = "empirical_parameter"
    UNIT_CONVERSION = "unit_conversion"
    REFERENCE_CONDITION = "reference_condition"
    SOIL_PROPERTY = "soil_property"
    WATER_RETENTION = "water_retention"
    HYDRAULIC_PROPERTY = "hydraulic_property"
    TRANSPORT_PROPERTY = "transport_property"
    SOIL_HYDRAULIC_PARAMETER = "soil_hydraulic_parameter"
    HYDRAULIC_PARAMETER = "hydraulic_parameter"

    # Missing categories required by manager.py
    ENVIRONMENTAL_RESPONSE = "environmental_response"
    TRANSPORT_PARAMETER = "transport_parameter"
    STOICHIOMETRY = "stoichiometry"
    CONSTRAINT = "constraint"
    ISOTOPE = "isotope"
    AGGREGATION = "aggregation"
    MANAGEMENT = "management"
    SCALING = "scaling"
    ATMOSPHERIC = "atmospheric"
    NUMERICAL = "numerical"
    GEOCHEMICAL = "geochemical"
    GAS_PROPERTY = "gas_property"
    DATA_QUALITY = "data_quality"
    CHEMICAL_CONSTANT = "chemical_constant"
    ENVIRONMENTAL_DEFAULT = "environmental_default"
    PHYSICAL_LIMIT = "physical_limit"
    PHOSPHORUS_PARTITIONING = "phosphorus_partitioning"
    NITROGEN_PARTITIONING = "nitrogen_partitioning"
    UPTAKE_PARAMETER = "uptake_parameter"
    ROOT_DISTRIBUTION = "root_distribution"
    TEMPERATURE_RESPONSE = "temperature_response"
    MODEL_SWITCH = "model_switch"
    PLANT_STORAGE = "plant_storage"
    POOL_INITIAL_VALUE = "pool_initial_value"
    STOICHIOMETRIC_PARAMETER = "stoichiometric_parameter"
    GEOCHEMICAL_PARAMETER = "geochemical_parameter"
    
    # Hierarchical coupling categories
    HIERARCHY_DEFINITION = "hierarchy_definition"
    HIERARCHY_CONFIGURATION = "hierarchy_configuration"
    LEVEL_MAPPING = "level_mapping"
    
    # Spatial and connectivity categories
    AREA_PARAMETER = "area_parameter"
    CONNECTIVITY_PARAMETER = "connectivity_parameter"
    CONNECTIVITY_FLAG = "connectivity_flag"
    
    # Scaling and strategy categories
    SCALING_STRATEGY = "scaling_strategy"
    TOLERANCE_PARAMETER = "tolerance_parameter"
    
    # Energy and thermodynamic categories
    ACTIVATION_ENERGY = "activation_energy"
    
    # Grid and discretization categories
    GRID_PARAMETER = "grid_parameter"
    BOUNDARY_PARAMETER = "boundary_parameter"
    
    # Numerical scheme categories
    STABILITY_PARAMETER = "stability_parameter"
    SCHEME_PARAMETER = "scheme_parameter"
    SPLITTING_PARAMETER = "splitting_parameter"
    
    # Scenario and performance categories
    SCENARIO_PARAMETER = "scenario_parameter"
    PERFORMANCE_PARAMETER = "performance_parameter"
    
    # Species and composition categories
    SPECIES_PARAMETER = "species_parameter"
    
    # Statistical distribution categories
    DISTRIBUTION_PARAMETER = "distribution_parameter"  # Shape, scale parameters for distributions
    PROBABILITY_PARAMETER = "probability_parameter"    # Probability values and transition matrices
    VOLATILITY_PARAMETER = "volatility_parameter"      # Stochastic process volatility/diffusion parameters
    RUNOFF_PARAMETER = "runoff_parameter"
    DISCRETIZATION_PARAMETER = "discretization_parameter"
    
    # Morphological categories
    MORPHOLOGY_PARAMETER = "morphology_parameter"      # Biological morphology parameters (LAI, height, etc.)
    
    # Meteorological categories
    METEOROLOGICAL_PARAMETER = "meteorological_parameter"  # Meteorological observation/forcing variables
    
    # Hydrological categories
    HYDROLOGICAL_PARAMETER = "hydrological_parameter"  # Hydrological processes and water-related parameters
    
    # Management and modifier categories
    MANAGEMENT_PARAMETER = "management_parameter"  # Management practices and interventions
    MODIFIER_PARAMETER = "modifier_parameter"  # Parameter modifiers for land use effects

    # Stoichiometry and composition categories
    LITTER_STOICHIOMETRY = "litter_stoichiometry"
    SOM_STOICHIOMETRY = "som_stoichiometry"
    MICROBIAL_STOICHIOMETRY = "microbial_stoichiometry"
    LITTER_COMPOSITION = "litter_composition"

    # Ratio and conversion categories
    CONVERSION_FACTOR = "conversion_factor"
    REFERENCE_RATIO = "reference_ratio"

    # Species and configuration categories (to avoid default warnings)
    SPECIES_PROPERTY = "species_property"
    MANAGEMENT_CONFIGURATION = "management_configuration"
    FLAG = "flag"
    DTYPE = "dtype"

    # Threshold and quality indicators
    THRESHOLD = "threshold"
    QUALITY_INDEX = "quality_index"
    
    # Spatial discretization categories
    SPATIAL_DISCRETIZATION = "spatial_discretization"  # Spatial grid and discretization parameters

    # Numerical scheme categories (additional)
    NUMERICAL_SCHEME = "numerical_scheme"  # Numerical integration and solution schemes
    NUMERICAL_EFFICIENCY = "numerical_efficiency"  # Computational efficiency parameters

    # Oceanographic categories
    OCEANOGRAPHIC_PARAMETER = "oceanographic_parameter"  # Ocean and marine system parameters

    # Atmospheric parameter alias
    ATMOSPHERIC_PARAMETER = "atmospheric_parameter"

    # Additional aliases for validation mapping
    STOICHIOMETRY_PARAMETER = "stoichiometry_parameter"
    TRANSFER_COEFFICIENT = "transfer_coefficient"
    KINETIC_PARAMETER = "kinetic_parameter"
    MOISTURE_RESPONSE = "moisture_response"
    IMMOBILIZATION = "immobilization"
    POOL_SPECIFIC_RATE = "pool_specific_rate"
    NUMERICAL_CONSTANT = "numerical_constant"
    LITTER_QUALITY_PARAMETER = "litter_quality_parameter"
    LITTER_PARTITIONING_PARAMETER = "litter_partitioning_parameter"
    ISOTOPE_STANDARD = "isotope_standard"
    FRACTIONATION_PARAMETER = "fractionation_parameter"
    TEMPERATURE_PARAMETER = "temperature_parameter"
    REFERENCE_PARAMETER = "reference_parameter"
    TRACKING_PARAMETER = "tracking_parameter"
    TURNOVER_RATE = "turnover_rate"
    BIOMASS = "biomass"
    ENVIRONMENTAL_MODIFIER = "environmental_modifier"
    MODEL_OPTION = "model_option"
    NORMALIZATION_CONSTANT = "normalization_constant"
    DECOMPOSITION_PARAMETER = "decomposition_parameter"
    DRAINAGE_PARAMETER = "drainage_parameter"
    STATE_DEFAULTS = "state_defaults"
    TURBULENCE_PARAMETER = "turbulence_parameter"
    MIXING_LENGTH_PARAMETER = "mixing_length_parameter"
    SURFACE_PARAMETER = "surface_parameter"
    OPERATIONAL_PARAMETER = "operational_parameter"
    ENVIRONMENTAL_CONSTANT_ALIAS = "environmental_constant"
    CONFIGURATION = "configuration"


class ParameterValidationLevel(Enum):
    """Parameter validation strictness levels."""
    NONE = "none"        # No validation
    BASIC = "basic"      # Type and range checking only
    STANDARD = "standard" # Standard scientific constraints
    STRICT = "strict"    # Full physical and relationship validation
    CUSTOM = "custom"    # User-defined validation level


# Backward-compatible alias. New code should use ParameterValidationLevel to
# avoid confusion with relic.validation.framework.models.ValidationLevel.
ValidationLevel = ParameterValidationLevel


class BoundsPolicy(Enum):
    """Handling strategy for out-of-bounds parameter updates."""
    WARN_AND_CLAMP = "warn_and_clamp"
    ERROR = "error"


class UnitValidationMode(Enum):
    """Unit validation modes for ParameterManager."""
    OFF = "off"
    CHECK_ONLY = "check-only"
    STRICT = "strict"


@dataclass
class ParameterValidationResult:
    """
    Result of parameter validation.
    
    Attributes:
        is_valid : bool
            Whether validation passed
        errors : List[str]
            List of validation errors
        warnings : List[str]
            List of validation warnings
        error_types : List[str]
            Optional machine-readable error categories (e.g., "bounds", "units", "type")
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_types: List[str] = field(default_factory=list)
    
    @classmethod
    def aggregate(cls, results: List['ParameterValidationResult']) -> 'ParameterValidationResult':
        """Aggregate multiple validation results."""
        all_errors = []
        all_warnings = []
        all_error_types = []
        is_valid = True
        
        for result in results:
            if not result.is_valid:
                is_valid = False
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            if hasattr(result, "error_types"):
                all_error_types.extend(result.error_types)
        
        return cls(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            error_types=all_error_types
        )


# Backward-compatible alias. New code should use ParameterValidationResult.
ValidationResult = ParameterValidationResult


@dataclass
class ParameterValidationContext:
    """
    Context for parameter validation.
    
    Provides additional information needed for complex validation
    including related parameters and environmental conditions.
    
    Attributes:
        all_parameters : Dict[str, ParameterValue]
            All current parameter values for relationship checking
        environment : Dict[str, Any]
            Environmental context (temperature, pressure, etc.)
        strict_mode : bool
            Whether to enforce strict validation
        warnings_as_errors : bool
            Treat warnings as errors
    """
    all_parameters: Dict[str, ParameterValue] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    strict_mode: bool = False
    warnings_as_errors: bool = False


# Backward-compatible alias. New code should use ParameterValidationContext.
ValidationContext = ParameterValidationContext

