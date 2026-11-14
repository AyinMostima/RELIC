"""
Configuration Data Models - Pydantic V2-based Configuration Schemas

This module defines the complete configuration schema for RELIC using
Pydantic V2 models for type safety, validation, and automatic documentation.
All configuration parameters are defined with types, defaults, bounds,
and validation rules to ensure scientific correctness.

The configuration models follow a hierarchical structure mirroring the
RELIC architecture, with separate models for physics, simulation,
parameters, outputs, and system settings.

Architecture:
    - Hierarchical configuration models with inheritance
    - Pydantic V2 validators for scientific parameter bounds
    - Automatic type conversion and validation
    - JSON Schema generation for documentation
    - Environment variable integration
    - Configuration composition and merging support
    - Hydra defaults tree alignment

Mathematical Validation:
    - Physical parameter bounds checking
    - Dimensional analysis for unit consistency  
    - Conservation law constraint validation
    - Numerical stability condition checking
    - Cross-domain consistency verification

References:
    - Pydantic V2 documentation: https://docs.pydantic.dev/latest/
    - JSON Schema specification: https://json-schema.org/
    - CF-1.8 Conventions: https://cfconventions.org/
    - Hydra configuration framework: https://hydra.cc/
"""

from typing import Dict, List, Any, Optional, Union, Literal, cast
from pathlib import Path
from dataclasses import dataclass
import importlib.util as importlib_util
from pydantic import BaseModel, Field, field_validator, model_validator, model_serializer, ConfigDict
from enum import Enum
from ...parameters.types import ValidationLevel as ParameterValidationLevel
import warnings
import os

from relic.core import device_policy
from relic.utils import time_constants as _time_constants

if importlib_util.find_spec("torch") is not None:
    import torch
else:  # pragma: no cover - torch is required in production
    torch = None

_STATE_VARIABLE_SPEC_CLS: Optional[type[Any]] = None


@dataclass
class StateConfig:
    """Lightweight wrapper for state schema construction."""
    dimensions: Dict[str, int]
    variables: Dict[str, Dict[str, Any]]

    def to_schema(self) -> Dict[str, Any]:
        global _STATE_VARIABLE_SPEC_CLS
        spec_cls = _STATE_VARIABLE_SPEC_CLS
        if spec_cls is None:
            if importlib_util.find_spec("relic.core.state.types") is None:
                raise ImportError("StateVariableSpec not available; cannot build state schema.")
            from ..state.types import StateVariableSpec as _spec_cls
            spec_cls = _spec_cls
            _STATE_VARIABLE_SPEC_CLS = _spec_cls
        if spec_cls is None:
            raise RuntimeError("StateVariableSpec class resolution failed.")
        schema: Dict[str, Any] = {}
        for name, meta in self.variables.items():
            bounds = meta.get("bounds")
            dims = meta.get("dims", [])
            shape = tuple(self.dimensions.get(dim, 1) for dim in dims)
            default_val = None
            if torch is not None and "initial_value" in meta:
                fill_value = float(meta["initial_value"])
                default_val = torch.full(shape, fill_value, dtype=torch.float32) if shape else torch.tensor(fill_value, dtype=torch.float32)
            schema[name] = spec_cls(
                name=name,
                units=meta.get("units", "dimensionless"),
                bounds=tuple(bounds) if bounds is not None else None,
                conserved=bool(meta.get("conserved", False)),
                spatial_dims=len(dims),
                required=bool(meta.get("required", True)),
                default=default_val,
                description=meta.get("description", ""),
            )
        return schema
from datetime import datetime


# ============================================================================
# Enumerations for Configuration Options
# ============================================================================

class IntegrationMethod(str, Enum):
    """Time integration methods accepted by the runtime.

    simulation.integration_method is the RELICSimulator time_integration_method
    setting; transport operators resolve it against the allowed_values of the
    parameter transport.numerical.time_integration_method (the runtime owner of
    the choice). The members equal those values; test_config_system verifies it.
    """
    EULER = "euler"
    IMPLICIT_MIDPOINT = "implicit_midpoint"
    CRANK_NICOLSON = "crank_nicolson"


class DeviceType(str, Enum):
    """Computation device types."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"
    MPS = "mps"  # Apple Metal Performance Shaders


class PrecisionType(str, Enum):
    """Numerical precision types."""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class ParallelBackend(str, Enum):
    """Parallel execution backends."""
    THREADS = "threads"
    PROCESSES = "processes"
    DASK = "dask"
    RAY = "ray"


class OutputFormat(str, Enum):
    """Output file formats."""
    NETCDF4 = "netcdf4"
    HDF5 = "hdf5"
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# Feature Flag Configuration
# ============================================================================

class FeatureFlagConfig(BaseModel):  # type: ignore[misc]
    """
    Global feature flag configuration for RELIC modules.
    
    Controls enabling/disabling of advanced features to ensure baseline
    behavior consistency and allow incremental feature rollout. All features
    default to False (disabled) to maintain baseline behavior.
    
    Environment Variable Override:
        Each flag can be overridden by environment variable:
        RELIC_ENABLE_<FLAG_NAME>=true/false
    """
    
    model_config = ConfigDict(
        env_prefix="RELIC_ENABLE_",
        case_sensitive=False,
        extra="forbid",
        protected_namespaces=(),
    )
    
    # Richards Equation Solver Features  
    full_richards_solver: bool = Field(
        default=False,
        description="Enable full Newton-Raphson PDE solver for Richards equation"
    )
    
    # Ozone Stress Features
    o3_stress_enabled: bool = Field(
        default=False, 
        description="Enable ozone stress effects on photosynthesis"
    )
    
    # Root Nutrient Uptake Features
    root_nutrient_uptake_enabled: bool = Field(
        default=False,
        description="Enable root nutrient uptake with Michaelis-Menten kinetics"
    )
    
    # Canopy Radiation Features
    canopy_layers_enabled: bool = Field(
        default=False,
        description="Enable multi-layer canopy radiation transfer"
    )
    
    # Conservation Checking Features
    strict_conservation_enabled: bool = Field(
        default=False,
        description="Enable strict element-level (C/N/P) mass conservation checking"
    )
    
    # Radiation Energy Conservation Features (read by RadiationFunctor; the
    # validator also needs its enable_conservation_check config switch)
    radiation_conservation: bool = Field(
        default=False,
        description="Enable the radiation energy conservation validator of RadiationFunctor"
    )

    # Canopy Interception Features
    canopy_interception_diagnostics: bool = Field(
        default=False,
        description="Enable detailed canopy interception diagnostics"
    )
    
    # Snow Albedo Features
    dynamic_snow_albedo: bool = Field(
        default=True,
        description="Enable dynamic snow albedo based on age and temperature"
    )
    
    # Numerical Solver Features
    advanced_numerical_solvers: bool = Field(
        default=False,
        description="Enable advanced numerical solvers"
    )
    
    # Parameter Management Features
    parameter_sensitivity_analysis: bool = Field(
        default=False,
        description="Enable parameter sensitivity analysis"
    )
    
    # Coupling Features  
    external_model_coupling: bool = Field(
        default=False,
        description="Enable coupling with external models"
    )
    
    def model_post_init(self, __context: Any) -> None:
        """Load environment variable overrides after initialization."""
        for field_name in self.__class__.model_fields:
            env_var = f"RELIC_ENABLE_{field_name.upper()}"
            if env_var in os.environ:
                env_value = os.environ[env_var].lower()
                if env_value in ('true', '1', 'yes', 'on'):
                    setattr(self, field_name, True)
                elif env_value in ('false', '0', 'no', 'off'):
                    setattr(self, field_name, False)


# ============================================================================
# Base Configuration Model
# ============================================================================

class ConfigurationModel(BaseModel):  # type: ignore[misc]
    """
    Base configuration model with common functionality.
    
    Provides common validation, serialization, and utility methods
    for all RELIC configuration models.
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,
        str_strip_whitespace=True,
        protected_namespaces=(),
    )

    @model_serializer(mode="wrap")  # type: ignore[misc]
    def serialize_model(self, handler: Any) -> Dict[str, Any]:
        """Ensure JSON dump converts Path/Datetime objects."""
        data = handler(self)
        return cast(Dict[str, Any], _serialize_value(data))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return cast(Dict[str, Any], self.model_dump())
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return str(self.model_dump_json(indent=2))


# ============================================================================
# Physics Configuration
# ============================================================================

class PhysicsConfig(ConfigurationModel):
    """
    Physics process configuration model.
    
    Defines which physical and biological processes are active
    in the simulation and their detailed settings.
    """
    
    # Core biogeochemical cycles
    carbon_cycle: bool = Field(
        True, 
        description="Enable carbon cycle processes"
    )
    water_balance: bool = Field(
        True, 
        description="Enable water balance calculations"
    )  
    nitrogen_cycle: bool = Field(
        False, 
        description="Enable nitrogen cycle processes"
    )
    phosphorus_cycle: bool = Field(
        False, 
        description="Enable phosphorus cycle processes"
    )
    
    # Atmospheric processes
    bvoc_emissions: bool = Field(
        False, 
        description="Enable BVOC emission calculations"
    )
    atmospheric_chemistry: bool = Field(
        False,
        description="Enable atmospheric chemistry reactions"
    )
    aerosol_dynamics: bool = Field(
        False,
        description="Enable aerosol formation and dynamics"
    )
    
    # Biosphere processes
    photosynthesis_model: Literal["farquhar", "simplified", "none"] = Field(
        "farquhar",
        description="Photosynthesis model type"
    )
    respiration_model: Literal["q10", "arrhenius", "none"] = Field(
        "q10",
        description="Respiration model type"
    )
    phenology_model: Literal["static", "dynamic", "satellite"] = Field(
        "static",
        description="Phenology model type"
    )
    allocation_model: Literal["fixed", "dynamic", "optimal"] = Field(
        "fixed",
        description="Carbon allocation model"
    )
    
    # Soil processes
    soil_biogeochemistry: bool = Field(
        True, 
        description="Enable soil biogeochemistry"
    )
    decomposition_model: Literal["first_order", "mimics", "century"] = Field(
        "first_order",
        description="Soil decomposition model"
    )
    
    # Hydrosphere processes
    evapotranspiration_model: Literal["penman_monteith", "priestley_taylor", "simple"] = Field(
        "penman_monteith",
        description="Evapotranspiration model"
    )
    runoff_model: Literal["topmodel", "curve_number", "simple"] = Field(
        "simple",
        description="Surface runoff model"
    )
    soil_moisture_model: Literal["richards", "bucket", "layers"] = Field(
        "layers",
        description="Soil moisture model"
    )
    
    # Transport processes  
    molecular_diffusion: bool = Field(
        True, 
        description="Enable molecular diffusion"
    )
    turbulent_transport: bool = Field(
        False, 
        description="Enable turbulent transport"
    )
    advection: bool = Field(
        False,
        description="Enable advective transport"
    )
    
    @model_validator(mode='after')  # type: ignore[misc]
    def validate_physics_consistency(self) -> 'PhysicsConfig':
        """Validate physics configuration consistency."""
        # BVOC requires carbon cycle
        if self.bvoc_emissions and not self.carbon_cycle:
            raise ValueError("BVOC emissions require carbon cycle to be enabled")
        
        # Atmospheric chemistry requires BVOC
        if self.atmospheric_chemistry and not self.bvoc_emissions:
            warnings.warn("Atmospheric chemistry enabled without BVOC emissions")
        
        # Nitrogen cycle interactions
        if self.nitrogen_cycle and self.decomposition_model == "first_order":
            warnings.warn("Consider using MIMICS or CENTURY for nitrogen cycle")
        
        return self


# ============================================================================
# Simulation Configuration
# ============================================================================

class SimulationConfig(ConfigurationModel):
    """
    Simulation control configuration model.
    
    Defines temporal and spatial discretization, integration methods,
    and simulation control parameters.
    """
    
    # Temporal settings
    start_time: float = Field(
        0.0, 
        ge=0.0, 
        description="Simulation start time [s]"
    )
    # One day from the time-constant SSOT (relic.utils.time_constants).
    end_time: float = Field(
        default_factory=lambda: float(_time_constants.SECONDS_PER_DAY),
        gt=0.0,
        description="Simulation end time [s]"
    )
    time_step: float = Field(
        60.0,
        gt=0.0,
        description="Default time step [s]; at most one hour (SECONDS_PER_HOUR)"
    )
    
    # Spatial settings
    spatial_resolution: float = Field(
        1.0, 
        gt=0.0, 
        description="Spatial resolution [m]"
    )
    grid_size_x: int = Field(
        100, 
        gt=0, 
        le=10000,
        description="Grid size in x-direction"
    )
    grid_size_y: int = Field(
        100,
        gt=0,
        le=10000,
        description="Grid size in y-direction"  
    )
    grid_size_z: int = Field(
        10,
        gt=0,
        le=100,
        description="Grid size in z-direction (layers)"
    )
    
    # Integration settings
    integration_method: IntegrationMethod = Field(
        IntegrationMethod.EULER, 
        description="Time integration method"
    )
    tolerance_relative: float = Field(
        1e-6,
        gt=0,
        le=0.1,
        description="Relative tolerance for adaptive methods"
    )
    tolerance_absolute: float = Field(
        1e-9,
        gt=0,
        le=0.01,
        description="Absolute tolerance for adaptive methods"
    )
    
    # Solver settings
    max_iterations: int = Field(
        100,
        gt=0,
        le=10000,
        description="Maximum solver iterations"
    )
    convergence_criterion: float = Field(
        1e-8,
        gt=0,
        description="Convergence criterion for iterative solvers"
    )
    
    # Conservation checking settings
    strict_conservation: bool = Field(
        False,
        description="Enable strict element-level (C/N/P) mass conservation checking"
    )
    conservation_checks: bool = Field(
        True,
        description="Enable conservation law validation during simulation"
    )
    log_conservation_violations: bool = Field(
        True,
        description="Log conservation violations to console and files"
    )
    
    @model_validator(mode='after')  # type: ignore[misc]
    def validate_temporal_consistency(self) -> 'SimulationConfig':
        """Validate temporal configuration consistency."""
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        
        if self.time_step > (self.end_time - self.start_time):
            raise ValueError("time_step cannot exceed simulation duration")

        seconds_per_hour = float(_time_constants.SECONDS_PER_HOUR)
        if self.time_step > seconds_per_hour:
            raise ValueError(f"time_step must not exceed one hour ({seconds_per_hour} s)")

        return self


# ============================================================================
# Stochastic Configuration
# ============================================================================

class StochasticConfig(ConfigurationModel):
    """
    Stochastic process and uncertainty quantification configuration.
    
    Controls Monte Carlo simulations, weather generation, and
    probabilistic validation methods.
    """
    
    # Monte Carlo settings
    enable_monte_carlo: bool = Field(
        False,
        description="Enable Monte Carlo uncertainty analysis"
    )
    num_realizations: int = Field(
        100,
        ge=1,
        le=10000,
        description="Number of Monte Carlo realizations"
    )
    random_seed: Optional[int] = Field(
        None,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    # Parameter uncertainty
    parameter_sampling: Literal["uniform", "normal", "lognormal", "beta"] = Field(
        "normal",
        description="Parameter sampling distribution"
    )
    correlation_structure: Literal["none", "exponential", "gaussian", "matern"] = Field(
        "none",
        description="Spatial correlation structure"
    )
    correlation_length: float = Field(
        1000.0,
        gt=0,
        description="Spatial correlation length [m]"
    )
    
    # Weather generation
    enable_weather_generator: bool = Field(
        False,
        description="Enable stochastic weather generation"
    )
    weather_model: Literal["richardson", "lars-wg", "markov"] = Field(
        "richardson",
        description="Weather generator model"
    )
    
    # Stochastic differential equations
    enable_sde: bool = Field(
        False,
        description="Enable stochastic differential equations"
    )
    sde_solver: Literal["euler_maruyama", "milstein", "runge_kutta"] = Field(
        "euler_maruyama",
        description="SDE numerical solver"
    )
    noise_type: Literal["white", "colored", "jump"] = Field(
        "white",
        description="Noise process type"
    )
    
    # Validation metrics
    compute_pit: bool = Field(
        True,
        description="Compute probability integral transform"
    )
    compute_crps: bool = Field(
        True,
        description="Compute continuous ranked probability score"
    )
    compute_coverage: bool = Field(
        True,
        description="Compute coverage probabilities"
    )
    confidence_levels: List[float] = Field(
        default=[0.5, 0.9, 0.95],
        description="Confidence levels for interval estimates"
    )


# ============================================================================
# Surrogate Models Configuration
# ============================================================================

class SurrogateConfig(ConfigurationModel):
    """
    Surrogate model configuration.
    
    Controls machine learning surrogate model settings for
    accelerating expensive scientific process calculations.
    """
    
    # General surrogate settings
    class_path: str = Field(
        "relic.models.surrogates.ONNXSurrogate",
        description="Fully qualified class name for surrogate model"
    )
    model_path: Path = Field(
        description="Path to surrogate model file (ONNX, pickle, etc.)"
    )
    
    # Input/Output mapping
    input_variables: List[str] = Field(
        description="Required state variables for inference"
    )
    output_variables: List[str] = Field(
        description="Output flux/rate variables"
    )
    
    # Variable and unit mapping
    input_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Map state variable names to model input names"
    )
    output_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Map model output names to flux variable names"
    )
    units_conversion: Dict[str, tuple[str, str]] = Field(
        default_factory=dict,
        description="Unit conversions: variable -> (from_unit, to_unit)"
    )
    
    # Inference settings
    device: DeviceType = Field(
        DeviceType.AUTO,
        description="Computation device for inference"
    )
    batch_size: int = Field(
        1,
        ge=1,
        le=1024,
        description="Maximum batch size for inference"
    )
    
    # ONNX-specific settings
    providers: List[str] = Field(
        default_factory=list,
        description="ONNX execution providers (auto-detected if empty)"
    )
    session_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="ONNX Runtime session options"
    )
    
    # Conservation and validation
    conservation_check: bool = Field(
        True,
        description="Whether to enforce conservation laws"
    )
    conservation_tolerance: float = Field(
        1e-6,
        gt=0,
        description="Maximum allowed conservation error"
    )
    
    # Dependencies and coupling
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of functor names this surrogate depends on"
    )
    
    # Uncertainty quantification
    uncertainty_quantification: bool = Field(
        False,
        description="Whether to estimate prediction uncertainty"
    )
    
    # Performance and caching
    cache_predictions: bool = Field(
        True,
        description="Cache predictions for identical inputs"
    )
    cache_size: int = Field(
        1000,
        ge=0,
        description="Maximum cache size (0 = unlimited)"
    )
    
    # Metadata
    model_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-loaded model metadata to avoid file parsing"
    )
    notes: str = Field(
        "",
        description="Additional notes about the surrogate model"
    )
    
    @model_validator(mode='after')  # type: ignore[misc]
    def validate_surrogate_config(self) -> 'SurrogateConfig':
        """Validate surrogate configuration consistency."""
        # Check model path exists if it's not a relative path
        if self.model_path.is_absolute() and not self.model_path.exists():
            raise ValueError(f"Surrogate model file not found: {self.model_path}")
        
        # Validate input/output variables are specified
        if not self.input_variables:
            raise ValueError("input_variables cannot be empty")
        if not self.output_variables:
            raise ValueError("output_variables cannot be empty")
        
        # Validate mapping consistency
        for var in self.input_mapping:
            if var not in self.input_variables:
                raise ValueError(f"Input mapping variable '{var}' not in input_variables")
        
        for var in self.output_mapping:
            if var not in self.output_variables:
                raise ValueError(f"Output mapping variable '{var}' not in output_variables")
        
        return self


# ============================================================================
# Optimization Configuration
# ============================================================================

class OptimizationConfig(ConfigurationModel):
    """
    Parameter optimization and calibration configuration.
    
    Defines optimization algorithms, objective functions, and
    calibration strategies.
    """
    
    # Optimization control
    enable_optimization: bool = Field(
        False,
        description="Enable parameter optimization"
    )
    optimization_method: Literal["gradient", "evolutionary", "bayesian", "hybrid"] = Field(
        "gradient",
        description="Optimization algorithm type"
    )
    
    # Gradient-based settings
    learning_rate: float = Field(
        0.01,
        gt=0,
        le=1.0,
        description="Learning rate for gradient methods"
    )
    optimizer: Literal["sgd", "adam", "lbfgs", "rmsprop"] = Field(
        "adam",
        description="Gradient optimizer"
    )
    max_epochs: int = Field(
        100,
        gt=0,
        description="Maximum optimization epochs"
    )
    
    # Evolutionary algorithm settings
    population_size: int = Field(
        50,
        ge=10,
        le=1000,
        description="Population size for evolutionary algorithms"
    )
    mutation_rate: float = Field(
        0.1,
        ge=0,
        le=1.0,
        description="Mutation rate"
    )
    crossover_rate: float = Field(
        0.9,
        ge=0,
        le=1.0,
        description="Crossover rate"
    )
    
    # Bayesian optimization settings
    acquisition_function: Literal["ei", "pi", "ucb"] = Field(
        "ei",
        description="Acquisition function for Bayesian optimization"
    )
    num_initial_points: int = Field(
        10,
        ge=5,
        description="Number of initial sampling points"
    )
    
    # Objective function
    objective: Literal["mse", "mae", "likelihood", "custom"] = Field(
        "mse",
        description="Objective function type"
    )
    regularization: Literal["none", "l1", "l2", "elastic"] = Field(
        "none",
        description="Parameter regularization"
    )
    regularization_strength: float = Field(
        0.01,
        ge=0,
        description="Regularization strength"
    )
    
    # Multi-objective settings
    multi_objective: bool = Field(
        False,
        description="Enable multi-objective optimization"
    )
    pareto_front_size: int = Field(
        50,
        ge=10,
        description="Pareto front population size"
    )


# ============================================================================
# Data Configuration
# ============================================================================

class DataIOConfig(BaseModel):  # type: ignore[misc]
    """Data input/output settings."""
    
    model_config = ConfigDict(
        protected_namespaces=(),
    )
    
    input_path: Path = Field(
        default_factory=lambda: Path("./data/input"),
        description="Input data directory"
    )
    output_path: Path = Field(
        default_factory=lambda: Path("./data/output"),
        description="Output data directory"
    )
    format: OutputFormat = Field(
        OutputFormat.NETCDF4,
        description="Primary output format"
    )
    compression: bool = Field(
        True,
        description="Enable output compression"
    )
    compression_level: int = Field(
        4,
        ge=1,
        le=9,
        description="Compression level (1-9)"
    )
    chunking: bool = Field(
        True,
        description="Enable NetCDF chunking"
    )
    chunk_size: Dict[str, int] = Field(
        default_factory=lambda: {"time": 100, "x": 50, "y": 50},
        description="Chunk dimensions"
    )


class DataQualityConfig(BaseModel):  # type: ignore[misc]
    """Data quality control settings."""
    
    model_config = ConfigDict(
        protected_namespaces=(),
    )
    
    enable_qc: bool = Field(
        True,
        description="Enable quality control checks"
    )
    missing_value_strategy: Literal["interpolate", "forward_fill", "mean", "drop"] = Field(
        "interpolate",
        description="Missing value handling"
    )
    outlier_detection: Literal["zscore", "iqr", "isolation", "none"] = Field(
        "zscore",
        description="Outlier detection method"
    )
    outlier_threshold: float = Field(
        3.0,
        gt=0,
        description="Outlier threshold (method-specific)"
    )
    smooth_input: bool = Field(
        False,
        description="Apply smoothing to input data"
    )
    smoothing_window: int = Field(
        5,
        ge=3,
        description="Smoothing window size"
    )


class DataTransformConfig(BaseModel):  # type: ignore[misc]
    """Data transformation settings."""
    
    model_config = ConfigDict(
        protected_namespaces=(),
    )
    
    standardize: bool = Field(
        False,
        description="Standardize input variables"
    )
    log_transform: List[str] = Field(
        default_factory=list,
        description="Variables to log-transform"
    )
    coordinate_system: Literal["cartesian", "geographic", "projected"] = Field(
        "cartesian",
        description="Coordinate system"
    )
    projection: Optional[str] = Field(
        None,
        description="Map projection (PROJ string)"
    )


class DataConfig(ConfigurationModel):
    """
    Comprehensive data management configuration.
    
    Controls data I/O, quality control, and transformations.
    """
    
    io: DataIOConfig = Field(
        default_factory=DataIOConfig,
        description="Input/output settings"
    )
    quality: DataQualityConfig = Field(
        default_factory=DataQualityConfig,
        description="Quality control settings"
    )
    transforms: DataTransformConfig = Field(
        default_factory=DataTransformConfig,
        description="Data transformation settings"
    )
    
    # Variable selection: canonical ContractRegistry variables that a Functor
    # writes (gross_primary_productivity [kg C m-2 s-1] of the photosynthesis
    # Functors, not the molar SSOT name gpp; net_primary_productivity;
    # evapotranspiration_flux; soil_moisture).
    output_variables: List[str] = Field(
        default_factory=lambda: [
            "gross_primary_productivity",
            "net_primary_productivity",
            "evapotranspiration_flux",
            "soil_moisture",
        ],
        description="Variables to output (canonical variable names)"
    )
    output_frequency: Literal["timestep", "hourly", "daily", "monthly"] = Field(
        "hourly",
        description="Output frequency"
    )
    
    # Metadata
    global_attributes: Dict[str, str] = Field(
        default_factory=lambda: {
            "institution": "Yuchong Long",
            "source": "RELIC v2.0.0"
        },
        description="Caller-owned global metadata; no CF conformance declaration is synthesized"
    )


# ============================================================================
# Parameters Configuration
# ============================================================================

class ParametersConfig(ConfigurationModel):
    """
    Parameter management system configuration.
    
    Controls parameter loading, validation, and database settings.
    """
    
    # Database settings
    database_path: Path = Field(
        default_factory=lambda: Path("relic/parameters/database"),
        description="Parameter database path"
    )
    load_from_database: bool = Field(
        True,
        description="Load parameters from database"
    )
    database_precedence: List[Literal["relic", "calibrated"]] = Field(
        default=["calibrated", "relic"],
        description="Database loading precedence"
    )
    
    # Validation settings
    validation_level: ParameterValidationLevel = Field(
        ParameterValidationLevel.STANDARD,
        description="Parameter validation strictness"
    )
    validate_units: bool = Field(
        True,
        description="Validate parameter units"
    )
    validate_bounds: bool = Field(
        True,
        description="Validate parameter bounds"
    )
    validate_relationships: bool = Field(
        False,
        description="Validate inter-parameter relationships"
    )
    
    # Override settings
    allow_overrides: bool = Field(
        True,
        description="Allow runtime parameter overrides"
    )
    parameter_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Specific parameter overrides"
    )
    
    # Sensitivity analysis
    sensitivity_analysis: bool = Field(
        False,
        description="Enable sensitivity analysis"
    )
    sensitivity_method: Literal["local", "morris", "sobol", "fast"] = Field(
        "sobol",
        description="Sensitivity analysis method"
    )
    sensitivity_samples: int = Field(
        1000,
        ge=100,
        description="Samples for sensitivity analysis"
    )
    
    # Parameter tracking
    track_changes: bool = Field(
        True,
        description="Track parameter changes"
    )
    save_calibrated: bool = Field(
        True,
        description="Save calibrated parameters"
    )
    calibration_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Calibration metadata"
    )


# ============================================================================
# Computation Configuration
# ============================================================================

class ComputationConfig(ConfigurationModel):
    """
    Computation optimization configuration model.
    
    Defines settings for GPU acceleration, JIT compilation, parallel execution,
    and distributed computing for performance optimization.
    """
    
    # Device settings
    device: DeviceType = Field(
        DeviceType.CPU,
        description="Computation device"
    )
    precision: PrecisionType = Field(
        PrecisionType.FLOAT32,
        description="Numerical precision"
    )
    
    # GPU acceleration
    enable_gpu: bool = Field(
        True,
        description="Enable GPU computation if available"
    )
    gpu_id: int = Field(
        0,
        ge=0,
        description="GPU device ID"
    )
    enable_amp: bool = Field(
        False,
        description="Enable Automatic Mixed Precision"
    )
    gpu_memory_fraction: float = Field(
        0.9,
        ge=0.1,
        le=1.0,
        description="GPU memory usage fraction"
    )
    
    # Explicit torch.compile request controls used by computation CLI callers.
    compile_backend: Literal["inductor", "cudagraphs", "onnxrt"] = Field(
        "inductor",
        description="Compilation backend"
    )
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        "default",
        description="Compilation mode"
    )
    
    # Parallel execution
    parallel_backend: ParallelBackend = Field(
        ParallelBackend.THREADS,
        description="Parallel backend"
    )
    num_workers: Optional[int] = Field(
        None,
        ge=1,
        description="Number of parallel workers (None=auto)"
    )
    chunk_size: Literal["auto", "small", "medium", "large", "adaptive"] = Field(
        "auto",
        description="Chunk size strategy"
    )
    
    # Distributed computing
    distributed_backend: Literal["gloo", "nccl", "mpi"] = Field(
        "gloo",
        description="Distributed backend"
    )
    enable_distributed: bool = Field(
        False,
        description="Enable distributed computing"
    )
    world_size: int = Field(
        1,
        ge=1,
        description="Number of distributed processes"
    )
    rank: int = Field(
        0,
        ge=0,
        description="Process rank in distributed setup"
    )
    
    # Memory management
    enable_gradient_checkpointing: bool = Field(
        False,
        description="Enable gradient checkpointing"
    )
    enable_memory_efficient_attention: bool = Field(
        False,
        description="Enable memory-efficient attention"
    )
    clear_cache_frequency: int = Field(
        100,
        ge=0,
        description="Clear GPU cache every N steps (0=disabled)"
    )
    
    # Performance monitoring
    enable_profiling: bool = Field(
        False,
        description="Enable performance profiling"
    )
    profile_memory: bool = Field(
        False,
        description="Include memory profiling"
    )
    profile_wait_steps: int = Field(
        5,
        ge=0,
        description="Warmup steps before profiling"
    )
    profile_warmup_steps: int = Field(
        5,
        ge=0,
        description="Steps to profile after warmup"
    )
    benchmark_mode: bool = Field(
        False,
        description="Enable benchmark mode"
    )


# ============================================================================
# Logging Configuration
# ============================================================================

class LoggingConfig(ConfigurationModel):
    """
    Logging and monitoring configuration.
    
    Controls logging output, metrics collection, and debugging.
    """
    
    # Basic logging
    level: LogLevel = Field(
        LogLevel.INFO,
        description="Global logging level"
    )
    console_output: bool = Field(
        True,
        description="Enable console output"
    )
    file_output: bool = Field(
        True,
        description="Enable file output"
    )
    log_dir: Path = Field(
        default_factory=lambda: Path("./logs"),
        description="Log file directory"
    )
    
    # Log formatting
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    date_format: str = Field(
        "%Y-%m-%d %H:%M:%S",
        description="Timestamp format"
    )
    
    # Module-specific levels
    module_levels: Dict[str, LogLevel] = Field(
        default_factory=lambda: {
            "relic.core": LogLevel.INFO,
            "relic.scientific": LogLevel.INFO,
            "relic.validation": LogLevel.WARNING
        },
        description="Module-specific log levels"
    )
    
    # Performance logging
    log_performance: bool = Field(
        False,
        description="Log performance metrics"
    )
    performance_interval: int = Field(
        100,
        ge=1,
        description="Performance logging interval (steps)"
    )
    
    # Debug options
    debug_mode: bool = Field(
        False,
        description="Enable debug mode"
    )
    trace_calls: bool = Field(
        False,
        description="Trace function calls"
    )
    log_tensors: bool = Field(
        False,
        description="Log tensor statistics"
    )
    
    # Metrics and monitoring
    enable_tensorboard: bool = Field(
        False,
        description="Enable TensorBoard logging"
    )
    tensorboard_dir: Path = Field(
        default_factory=lambda: Path("./runs"),
        description="TensorBoard log directory"
    )
    enable_wandb: bool = Field(
        False,
        description="Enable Weights & Biases logging"
    )
    wandb_project: str = Field(
        "relic",
        description="W&B project name"
    )


# ============================================================================
# Main Configuration Model
# ============================================================================

class RELICConfig(ConfigurationModel):
    """
    Main RELIC configuration model.
    
    Root configuration model that combines all subsystem configurations
    into a single, validated configuration object with comprehensive
    validation and JSON Schema generation support.
    
    Mathematical Validation:
        The configuration ensures:
        - Temporal consistency: dt <= (end_time - start_time)
        - Spatial consistency: grid dimensions match physics requirements
        - Conservation settings alignment across modules
        - Unit consistency following CF-1.8 conventions
        
    Hydra Integration:
        Supports Hydra defaults tree with structured configs and
        command-line overrides for all parameters.
    """
    
    # Metadata
    config_version: str = Field(
        "2.0.0", 
        description="Configuration schema version",
        pattern=r"^\d+\.\d+\.\d+$"
    )
    simulation_name: str = Field(
        "relic_simulation", 
        description="Simulation identifier",
        min_length=1,
        max_length=256
    )
    description: str = Field(
        "", 
        description="Simulation description",
        max_length=1024
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Configuration creation timestamp"
    )
    
    # Subsystem configurations
    physics: PhysicsConfig = Field(
        default_factory=PhysicsConfig,
        description="Physics and biological process settings"
    )
    simulation: SimulationConfig = Field(
        default_factory=SimulationConfig,
        description="Simulation control parameters"
    )
    stochastic: StochasticConfig = Field(
        default_factory=StochasticConfig,
        description="Stochastic process configuration"
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Parameter optimization settings"
    )
    surrogates: Dict[str, SurrogateConfig] = Field(
        default_factory=dict,
        description="Surrogate model configurations by functor name"
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data I/O and quality control"
    )
    parameters: ParametersConfig = Field(
        default_factory=ParametersConfig,
        description="Parameter management configuration"
    )
    computation: ComputationConfig = Field(
        default_factory=ComputationConfig,
        description="Computational optimization settings"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging and monitoring settings"
    )
    feature_flags: FeatureFlagConfig = Field(
        default_factory=FeatureFlagConfig,
        description="Global feature flags"
    )
    
    @field_validator('config_version')  # type: ignore[misc]
    @classmethod
    def validate_config_version(cls, v: str) -> str:
        """Validate configuration version consistency."""
        supported_major = "2"
        major_version = v.split('.')[0]
        
        if major_version != supported_major:
            warnings.warn(
                f"Configuration version {v} may not be consistent "
                f"(expected major version {supported_major})"
            )
        
        return v
    
    @field_validator('simulation_name')  # type: ignore[misc]
    @classmethod
    def validate_simulation_name(cls, v: str) -> str:
        """Ensure simulation name is valid for file systems."""
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValueError(
                "Simulation name must contain only alphanumeric characters, "
                "underscores, and hyphens"
            )
        return v
    
    @model_validator(mode='after')  # type: ignore[misc]
    def validate_cross_config(self) -> 'RELICConfig':
        """Cross-configuration validation."""
        # Temporal consistency
        sim = self.simulation
        if sim.end_time <= sim.start_time:
            raise ValueError("Simulation end_time must be after start_time")
        
        if sim.time_step > (sim.end_time - sim.start_time):
            raise ValueError("Time step cannot exceed total simulation duration")
        
        # Physics-stochastic consistency
        if self.stochastic.enable_weather_generator and not self.physics.carbon_cycle:
            warnings.warn(
                "Weather generator enabled but carbon cycle disabled - "
                "limited functionality"
            )
        
        # Optimization-parameter consistency
        if self.optimization.enable_optimization:
            if self.parameters.validation_level == ParameterValidationLevel.NONE:
                warnings.warn(
                    "Optimization enabled with no parameter validation - "
                    "results may be unreliable"
                )
        
        # Computation-simulation consistency
        if self.computation.enable_distributed:
            if self.computation.world_size > 1 and sim.grid_size_x * sim.grid_size_y < 1000:
                warnings.warn(
                    "Distributed computing enabled for small domain - "
                    "may reduce performance"
                )
        
        # Data output consistency
        if self.data.io.format == OutputFormat.CSV and len(self.data.output_variables) > 10:
            warnings.warn(
                "CSV format with many variables - consider NetCDF4 for efficiency"
            )
        
        return self
    
    def to_schema(self) -> Dict[str, Any]:
        """
        Generate JSON Schema for configuration validation.
        
        Returns:
            Dict[str, Any]: JSON Schema consistent with draft-07
        """
        schema = self.model_json_schema()
        
        # Add schema metadata
        schema['$schema'] = 'http://json-schema.org/draft-07/schema#'
        schema['title'] = 'RELIC Configuration Schema'
        schema['description'] = 'Complete configuration schema for RELIC simulations'
        schema['version'] = self.config_version
        
        return cast(Dict[str, Any], schema)
    
    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Export configuration to YAML format.
        
        Parameters:
            path: Optional file path to save YAML
            
        Returns:
            YAML string if no path provided, None if saved to file
        """
        if importlib_util.find_spec("yaml") is None:
            raise ImportError("PyYAML required for YAML export")
        import yaml
        
        config_dict = self.model_dump(exclude_unset=True, exclude_none=True)
        
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
            return None
        else:
            return yaml.safe_dump(config_dict, default_flow_style=False, indent=2, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'RELICConfig':
        """
        Load configuration from YAML file.

        This is ConfigLoader().load_config(path), the single configuration load
        path: UTF-8 file, RELIC_<SECTION>_<FIELD> environment overrides,
        parameters.parameter_overrides written to the ParameterManager, and
        cross-section validation.

        Parameters:
            path: Path to YAML configuration file

        Returns:
            RELICConfig: Validated configuration object
        """
        # Imported here: loader.py imports this module.
        from .loader import ConfigLoader

        return ConfigLoader().load_config(Path(path))
    
    def merge_with(self, other: Union[Dict[str, Any], 'RELICConfig']) -> 'RELICConfig':
        """
        Merge with another configuration, with other taking precedence.
        
        Parameters:
            other: Configuration to merge with
            
        Returns:
            RELICConfig: New merged configuration
        """
        if isinstance(other, RELICConfig):
            other_dict = other.model_dump(exclude_unset=True)
        else:
            other_dict = other
        
        self_dict = self.model_dump()
        
        # Deep merge dictionaries
        merged = self._deep_merge(self_dict, other_dict)
        
        return RELICConfig(**merged)
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = RELICConfig._deep_merge(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def validate_for_simulation(self) -> List[str]:
        """
        Perform comprehensive validation for simulation readiness.
        
        Returns:
            List[str]: List of validation warnings (empty if fully valid)
        """
        warnings_list = []
        
        # Check required paths exist
        if not self.data.io.input_path.exists():
            warnings_list.append(f"Input path does not exist: {self.data.io.input_path}")
        
        # Check parameter database
        if self.parameters.load_from_database:
            if not self.parameters.database_path.exists():
                warnings_list.append(
                    f"Parameter database not found: {self.parameters.database_path}"
                )
        
        # Check computational resources
        if self.computation.device == DeviceType.CUDA:
            if torch is None:
                warnings_list.append("PyTorch not installed for CUDA check")
            elif not device_policy.is_cuda_available():
                warnings_list.append("CUDA requested but not available")
        
        # Check stochastic settings
        if self.stochastic.enable_monte_carlo:
            if self.stochastic.num_realizations < 10:
                warnings_list.append(
                    "Very few Monte Carlo realizations - results may not be robust"
                )
        
        return warnings_list


# ============================================================================
# Export Configuration Models
# ============================================================================

__all__ = [
    # Main config
    'RELICConfig',
    
    # Sub-configurations
    'PhysicsConfig',
    'SimulationConfig',
    'StochasticConfig',
    'OptimizationConfig',
    'SurrogateConfig',
    'DataConfig',
    'ParametersConfig',
    'ComputationConfig',
    'LoggingConfig',
    'FeatureFlagConfig',
    
    # Data sub-configs
    'DataIOConfig',
    'DataQualityConfig',
    'DataTransformConfig',
    
    # Enums
    'IntegrationMethod',
    'DeviceType',
    'PrecisionType',
    'ParallelBackend',
    'ParameterValidationLevel',
    'OutputFormat',
    'LogLevel',
    
    # Base class
    'ConfigurationModel'
]
# Helper used for JSON serialization (Path/Datetime)
def _serialize_value(value: Any) -> Any:
    """Serialize nested configuration values for JSON output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        container_type = type(value)
        serialized = [_serialize_value(v) for v in value]
        if container_type is tuple:
            return tuple(serialized)
        if container_type is set:
            return set(serialized)
        return serialized
    return value
