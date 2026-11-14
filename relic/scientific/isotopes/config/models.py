"""
Isotope Configuration Models for 15N Tracking Module

This module defines the complete configuration schema for the 15N isotope 
tracking system using Pydantic models. The configuration follows the 
zero-intrusion design pattern and supports flexible process-specific 
fractionation parameters.

The configuration models support:
- Multiple isotope species (focused on 15N)
- Process-specific fractionation models (Q10, Arrhenius)
- Pool mapping to StateManager variables
- NetCDF output specifications with bounded CF metadata
- Initial abundance configuration
- Conservation check parameters

Architecture:
    - IsotopeConfig: Main configuration model
    - ProcessConfig: Process-specific fractionation settings
    - PoolConfig: Pool mapping and initial conditions
    - OutputConfig: NetCDF output specifications
    - FractionationModel: Fractionation parameter models

Mathematical Validation:
    - Fractionation factor bounds (0.5 < alpha < 2.0)
    - Temperature range validation (200K < T < 400K)
    - Initial abundance bounds (0 < r < 1)
    - Conservation tolerance validation

References:
    - Mariotti et al. (1981). Experimental determination of nitrogen kinetic isotope fractionation
    - Casciotti et al. (2003). Linking diversity and stable isotope fractionation
    - CF Conventions and Metadata (2023). NetCDF Climate and Forecast Metadata Conventions
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import warnings
from pathlib import Path
from relic.parameters import get_global_manager


def _build_full_key(key: str, namespace: str) -> str:
    """Build a fully-qualified parameter key."""
    if key.startswith("isotopes."):
        return key
    return f"{namespace}.{key}"


def _require_param(
    key: str,
    *,
    units: str,
    expected_type: type = float,
    namespace: str = "isotopes.n15",
) -> Any:
    """
    Fetch a required parameter from the ParameterManager.

    Raises
    ------
    KeyError
        If the parameter is missing.
    TypeError
        If the parameter cannot be converted to the requested type.
    """
    full_key = _build_full_key(key, namespace)
    manager_expected_type = expected_type if isinstance(expected_type, type) else None
    value = get_global_manager().get(
        full_key,
        units=units,
        namespace=None,
        required=True,
        expected_type=manager_expected_type,
    )
    if expected_type in (float, int):
        try:
            return expected_type(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Isotope parameter '{full_key}' must be convertible to {expected_type.__name__}."
            ) from exc
    if expected_type is str:
        return str(value)
    return value


def _default_pools() -> Dict[str, "PoolConfig"]:
    """Load pool defaults from the parameter system."""
    default_abundance = _require_param("pools.default_initial_abundance", units="dimensionless")
    orgn_abundance = _require_param("pools.orgN.initial_abundance", units="dimensionless")
    nh4_abundance = _require_param("pools.nh4.initial_abundance", units="dimensionless")
    no3_abundance = _require_param("pools.no3.initial_abundance", units="dimensionless")
    return {
        "orgN": PoolConfig(
            pool_name="orgN",
            state_variable="orgN",
            initial_abundance=orgn_abundance if orgn_abundance is not None else default_abundance,
            units="mol m-2",
            description="Organic nitrogen pool",
        ),
        "NH4": PoolConfig(
            pool_name="NH4",
            state_variable="NH4",
            initial_abundance=nh4_abundance if nh4_abundance is not None else default_abundance,
            units="mol m-2",
            description="Ammonium nitrogen pool",
        ),
        "NO3": PoolConfig(
            pool_name="NO3",
            state_variable="NO3",
            initial_abundance=no3_abundance if no3_abundance is not None else default_abundance,
            units="mol m-2",
            description="Nitrate nitrogen pool",
        ),
    }


def _default_processes() -> Dict[str, "ProcessConfig"]:
    """Load process defaults from the parameter system."""
    def frac_params(prefix: str) -> FractionationParameters:
        return FractionationParameters(
            model_type=FractionationModelType.Q10,
            alpha_0=_require_param(f"{prefix}.alpha_0", units="dimensionless"),
            Q10=_require_param(f"{prefix}.q10", units="dimensionless"),
            T_ref=_require_param(f"{prefix}.t_ref", units="K"),
            T_min=_require_param(f"{prefix}.t_min", units="K"),
            T_max=_require_param(f"{prefix}.t_max", units="K"),
        )

    return {
        "mineralization": ProcessConfig(
            process_name="mineralization",
            fractionation=frac_params("process.mineralization"),
            source_pool="orgN",
            target_pool="NH4",
            flux_variable="mineralization_flux",
            references=["Hobbie & Ouimette (2009)", "Templer et al. (2007)"],
        ),
        "nitrification": ProcessConfig(
            process_name="nitrification",
            fractionation=frac_params("process.nitrification"),
            source_pool="NH4",
            target_pool="NO3",
            flux_variable="nitrification_flux",
            references=["Casciotti et al. (2003)", "Mariotti et al. (1981)"],
        ),
        "denitrification": ProcessConfig(
            process_name="denitrification",
            fractionation=frac_params("process.denitrification"),
            source_pool="NO3",
            target_pool="N2O",
            flux_variable="denitrification_flux",
            references=["Granger et al. (2008)", "Ostrom et al. (2007)"],
        ),
    }


def _default_conservation() -> Dict[str, Any]:
    """Load conservation settings from the parameter system."""
    tolerance = _require_param("conservation.tolerance", units="dimensionless")
    return {
        "enabled": True,
        "tolerance": tolerance,
        "check_frequency": "every_step",
        "raise_on_violation": False,
        "log_violations": True,
    }


def _default_output_variables() -> Dict[str, "OutputVariableConfig"]:
    """Load the output variables of the default configuration from the parameter system.

    Output configuration only: output.variables_default lists the variables written by the
    default 15N configuration, and output.variable_compression_default is the compression
    flag of a listed variable that declares none. Neither enters a scientific calculation;
    both are documented output options, not substituted scientific inputs.
    """
    variables_data = _require_param(
        "output.variables_default",
        units="dimensionless",
        expected_type=dict,
        namespace="isotopes.n15",
    )
    variables = {}
    compression_default = _require_param(
        "output.variable_compression_default",
        units="dimensionless",
        expected_type=bool,
        namespace="isotopes",
    )
    fill_value = _require_param(
        "output.netcdf_fill_value",
        units="dimensionless",
        expected_type=float,
        namespace="isotopes",
    )
    for name, stored in variables_data.items():
        # Complete a copy: the stored parameter value is never changed in place.
        data = dict(stored)
        if "variable_name" not in data:
            data["variable_name"] = name
        if "compression" not in data:
            data["compression"] = compression_default
        if "fill_value" not in data:
            data["fill_value"] = fill_value
        variables[name] = OutputVariableConfig(**data)
    return variables


def _default_output_config() -> "OutputConfig":
    """Load output configuration defaults from the parameter system."""
    output_kwargs = {
        "output_file": _require_param("output.file", units="dimensionless", expected_type=str, namespace="isotopes"),
        "create_directory": _require_param("output.create_directory", units="dimensionless", expected_type=bool, namespace="isotopes"),
        "overwrite_existing": _require_param("output.overwrite_existing", units="dimensionless", expected_type=bool, namespace="isotopes"),
        "aggregation_frequency": _require_param("output.aggregation_frequency", units="dimensionless", expected_type=str, namespace="isotopes"),
        "time_units": _require_param("output.time_units", units="dimensionless", expected_type=str, namespace="isotopes"),
        "calendar": _require_param("output.calendar", units="dimensionless", expected_type=str, namespace="isotopes"),
        "netcdf_format": _require_param("output.netcdf_format", units="dimensionless", expected_type=str, namespace="isotopes"),
        "compression_level": _require_param("output.compression_level", units="dimensionless", expected_type=int, namespace="isotopes"),
        "chunking": _require_param("output.chunking", units="dimensionless", expected_type=bool, namespace="isotopes"),
        "shuffle": _require_param("output.shuffle", units="dimensionless", expected_type=bool, namespace="isotopes"),
        "fletcher32": _require_param("output.fletcher32", units="dimensionless", expected_type=bool, namespace="isotopes"),
        "title": _require_param("output.title", units="dimensionless", expected_type=str, namespace="isotopes"),
        "institution": _require_param("output.institution", units="dimensionless", expected_type=str, namespace="isotopes"),
        "source": _require_param("output.source", units="dimensionless", expected_type=str, namespace="isotopes"),
        "history": _require_param("output.history", units="dimensionless", expected_type=str, namespace="isotopes"),
        "references": _require_param("output.references", units="dimensionless", expected_type=str, namespace="isotopes"),
        "comment": _require_param("output.comment", units="dimensionless", expected_type=str, namespace="isotopes"),
        "conventions": _require_param("output.conventions", units="dimensionless", expected_type=str, namespace="isotopes"),
        "variables": _default_output_variables(),
    }
    return OutputConfig(**output_kwargs)


def get_default_15n_config() -> "Nitrogen15Config":
    """Build the canonical default configuration for 15N tracking."""
    return Nitrogen15Config(tracked_isotope="15N")


class FractionationModelType(str, Enum):
    """Supported fractionation model types."""
    Q10 = "q10"
    ARRHENIUS = "arrhenius"
    CONSTANT = "constant"


class AggregationMethod(str, Enum):
    """Supported aggregation methods for output."""
    DAILY_MEAN = "daily_mean"
    DAILY_SUM = "daily_sum"
    END_OF_DAY = "end_of_day"
    INTEGRAL = "integral"


class IsotopeConfigurationModel(BaseModel):
    """
    Base configuration model for isotope module.
    
    Provides common validation, serialization, and utility methods
    for all isotope configuration models.
    """
    
    model_config = {
        "validate_assignment": True,
        "extra": "forbid",  # Prevent extra fields
        "use_enum_values": False,
        "validate_default": True,
        "protected_namespaces": (),
    }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return self.model_dump_json(indent=2)


class FractionationParameters(IsotopeConfigurationModel):
    """
    Fractionation model parameters.
    
    Defines the mathematical parameters for temperature-dependent
    fractionation factor calculations.
    
    Mathematical Expression:
        Q10 model: alpha(T) = alpha_0 * Q_1_0^((T - T_ref)/10)
        Arrhenius: alpha(T) = alpha_0 * exp(-E_a/(R*T) + E_a/(R*T_ref))
        Constant: alpha(T) = alpha_0
    
    Parameters:
        model_type: Type of fractionation model
        alpha_0: Reference fractionation factor at T_ref [dimensionless]
        T_ref: Reference temperature [K]
        Q10: Q10 temperature coefficient [dimensionless] (for Q10 model)
        E_a: Activation energy [J/mol] (for Arrhenius model)
    """
    
    model_type: FractionationModelType = Field(
        ...,
        description="Type of fractionation model"
    )
    
    alpha_0: float = Field(
        ...,
        description="Reference fractionation factor at T_ref [dimensionless]"
    )
    
    T_ref: float = Field(
        ...,
        description="Reference temperature [K]"
    )
    
    # Q10 model parameters
    Q10: Optional[float] = Field(
        None,
        description="Q10 temperature coefficient [dimensionless]"
    )
    
    # Arrhenius model parameters
    E_a: Optional[float] = Field(
        None,
        description="Activation energy [J/mol]"
    )
    
    # Physical bounds
    T_min: float = Field(
        ...,
        description="Minimum temperature for valid fractionation [K]"
    )
    
    T_max: float = Field(
        ...,
        description="Maximum temperature for valid fractionation [K]"
    )

    @model_validator(mode='after')
    def validate_model_consistency(self):
        """Validate cross-field dependencies and model requirements."""
        # Validate Q10 parameters when Q10 model is selected
        if self.model_type == FractionationModelType.Q10 and self.Q10 is None:
            raise ValueError("Q10 parameter required for Q10 model")
        
        # Validate Arrhenius parameters when Arrhenius model is selected
        if self.model_type == FractionationModelType.ARRHENIUS and self.E_a is None:
            raise ValueError("E_a parameter required for Arrhenius model")
        
        # Ensure T_max > T_min
        if self.T_max <= self.T_min:
            raise ValueError("T_max must be greater than T_min")
        
        return self


class ProcessConfig(IsotopeConfigurationModel):
    """
    Process-specific isotope configuration.
    
    Defines fractionation parameters for specific nitrogen transformation
    processes (mineralization, nitrification, denitrification).
    
    Mathematical Expression:
        Process flux: Phi_15N = alpha_p(T) * r_source * Phi_total
        where alpha_p is process-specific fractionation factor
    
    Parameters:
        process_name: Name of the nitrogen transformation process
        fractionation: Fractionation model parameters
        enabled: Whether to track this process
        source_pool: Name of source nitrogen pool
        target_pool: Name of target nitrogen pool
        flux_variable: StateManager variable name for process flux
    """
    
    process_name: str = Field(
        ...,
        min_length=1,
        description="Name of nitrogen transformation process"
    )
    
    fractionation: FractionationParameters = Field(
        ...,
        description="Fractionation model parameters"
    )
    
    enabled: bool = Field(
        True,
        description="Whether to track isotopes for this process"
    )
    
    source_pool: str = Field(
        ...,
        min_length=1,
        description="Name of source nitrogen pool (StateManager variable)"
    )
    
    target_pool: str = Field(
        ...,
        min_length=1,
        description="Name of target nitrogen pool (StateManager variable)"
    )
    
    flux_variable: str = Field(
        ...,
        min_length=1,
        description="StateManager variable name for process flux"
    )
    
    # Scientific metadata
    references: List[str] = Field(
        default_factory=list,
        description="Scientific references for fractionation parameters"
    )
    
    notes: str = Field(
        "",
        description="Additional notes about process implementation"
    )


class PoolConfig(IsotopeConfigurationModel):
    """
    Nitrogen pool configuration for isotope tracking.
    
    Defines mapping between isotope pools and StateManager variables,
    plus initial isotope abundance settings.
    
    Mathematical Expression:
        Initial conditions: r_0 = N_1_5,_0 / N_total,_0
        where r_0 is initial 15N abundance
    
    Parameters:
        pool_name: Name of nitrogen pool
        state_variable: Corresponding StateManager variable
        initial_abundance: Initial 15N abundance ratio
        track_ratio: Whether to output isotope ratio
        track_flux: Whether to track flux contributions
    """
    
    pool_name: str = Field(
        ...,
        min_length=1,
        description="Name of nitrogen pool for isotope tracking"
    )
    
    state_variable: str = Field(
        ...,
        min_length=1,
        description="Corresponding StateManager variable name"
    )
    
    initial_abundance: float = Field(
        ...,
        description="Initial 15N abundance ratio [dimensionless]"
    )
    
    track_ratio: bool = Field(
        True,
        description="Whether to output isotope ratio for this pool"
    )
    
    track_flux: bool = Field(
        True,
        description="Whether to track flux contributions to this pool"
    )
    
    # Units and metadata
    units: str = Field(
        "mol m-2",
        description="Units of the nitrogen pool"
    )
    
    description: str = Field(
        "",
        description="Description of the nitrogen pool"
    )


class OutputVariableConfig(IsotopeConfigurationModel):
    """
    Configuration for individual output variables.
    
    Defines NetCDF variable specifications including CF-1.8 metadata,
    aggregation methods, and compression settings.
    
    Parameters:
        variable_name: NetCDF variable name
        standard_name: CF-1.8 standard name
        units: Variable units
        aggregation_method: How to aggregate data temporally
        compression: Whether to compress this variable
    """
    
    variable_name: str = Field(
        ...,
        min_length=1,
        description="NetCDF variable name"
    )
    
    standard_name: Optional[str] = Field(
        None,
        description="CF-1.8 standard name"
    )
    
    long_name: str = Field(
        "",
        description="Descriptive long name for variable"
    )
    
    units: str = Field(
        ...,
        min_length=1,
        description="Variable units following CF conventions"
    )
    
    aggregation_method: AggregationMethod = Field(
        ...,
        description="Temporal aggregation method"
    )
    
    # NetCDF attributes
    cell_methods: Optional[str] = Field(
        None,
        description="CF cell_methods attribute"
    )
    
    ancillary_variables: Optional[str] = Field(
        None,
        description="Related ancillary variables"
    )
    
    # Isotope-specific metadata
    isotope: str = Field(
        "15N",
        description="Isotope identifier"
    )
    
    parent_variable: Optional[str] = Field(
        None,
        description="Parent (total) variable name"
    )
    
    fractionation_model: Optional[str] = Field(
        None,
        description="Fractionation model used"
    )
    
    # Data handling
    compression: bool = Field(
        ...,
        description="Whether to compress this variable"
    )
    
    fill_value: float = Field(
        ...,
        description="Missing data fill value"
    )


class OutputConfig(IsotopeConfigurationModel):
    """
    Output configuration for isotope tracking data.
    
    Defines NetCDF output specifications including file paths,
    temporal aggregation, bounded CF metadata, and compression settings.
    
    Parameters:
        output_file: Path to output NetCDF file
        aggregation_frequency: Temporal aggregation frequency
        variables: Configuration for each output variable
        compression: Compression settings
    """
    
    # File settings
    output_file: str = Field(
        ...,
        description="Path to output NetCDF file"
    )
    
    create_directory: bool = Field(
        ...,
        description="Create output directory if it doesn't exist"
    )
    
    overwrite_existing: bool = Field(
        ...,
        description="Overwrite existing output file"
    )
    
    # Temporal settings
    aggregation_frequency: str = Field(
        ...,
        description="Temporal aggregation frequency (daily, hourly, etc.)"
    )
    
    time_units: str = Field(
        ...,
        description="CF-compliant time units"
    )
    
    calendar: str = Field(
        ...,
        description="Calendar system"
    )
    
    # Variable configurations
    variables: Dict[str, OutputVariableConfig] = Field(
        ...,
        description="Configuration for each output variable"
    )
    
    # NetCDF settings
    netcdf_format: str = Field(
        ...,
        description="NetCDF file format"
    )
    
    compression_level: int = Field(
        ...,
        description="Compression level (0=none, 9=maximum)"
    )
    
    chunking: bool = Field(
        ...,
        description="Enable chunked storage"
    )
    
    shuffle: bool = Field(
        ...,
        description="Enable shuffle filter for better compression"
    )
    
    fletcher32: bool = Field(
        ...,
        description="Enable Fletcher32 checksum"
    )
    
    # CF-1.8 global attributes
    title: str = Field(
        ...,
        description="Dataset title"
    )
    
    institution: str = Field(
        ...,
        description="Institution responsible for data"
    )
    
    source: str = Field(
        ...,
        description="Source of data"
    )
    
    history: str = Field(
        ...,
        description="Processing history"
    )
    
    references: str = Field(
        ...,
        description="Scientific references"
    )
    
    comment: str = Field(
        ...,
        description="Additional comments"
    )
    
    conventions: str = Field(
        ...,
        description="Conventions followed"
    )
    
    @field_validator('output_file')
    @classmethod
    def validate_output_file(cls, v: Any) -> Any:
        """Validate output file path and extension."""
        path = Path(v)
        if path.suffix not in ['.nc', '.nc4']:
            raise ValueError("Output file must have .nc or .nc4 extension")
        return str(path)
    
    @field_validator('compression_level')
    @classmethod
    def validate_compression_level(cls, v: Any) -> Any:
        """Validate compression level."""
        if v < 0 or v > 9:
            raise ValueError("Compression level must be between 0 and 9")
        return v


class Nitrogen15Config(IsotopeConfigurationModel):
    """
    Main isotope tracking configuration model.
    
    Root configuration model that defines the complete setup for
    15N isotope tracking including processes, pools, and outputs.
    
    Mathematical Expression:
        System configuration: Psi = {pools, processes, outputs, parameters}
        where Psi defines the complete isotope tracking system
    
    Parameters:
        tracked_isotope: Isotope species to track (15N)
        pools: Configuration for tracked nitrogen pools
        processes: Configuration for transformation processes
        outputs: Output file and variable specifications
        conservation_check: Conservation checking parameters
    """
    
    # Metadata
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version"
    )
    
    module_name: str = Field(
        "isotope_tracker_15N",
        description="Module identifier"
    )
    
    description: str = Field(
        "Configuration for 15N isotope tracking in nitrogen cycle processes",
        description="Module description"
    )
    
    # Core settings
    tracked_isotope: str = Field(
        "15N",
        description="Isotope species to track"
    )
    
    enabled: bool = Field(
        True,
        description="Whether isotope tracking is enabled"
    )
    
    # Pool configuration
    pools: Dict[str, PoolConfig] = Field(
        default_factory=_default_pools,
        description="Configuration for tracked nitrogen pools",
    )
    
    # Process configuration
    processes: Dict[str, ProcessConfig] = Field(
        default_factory=_default_processes,
        description="Configuration for transformation processes",
    )
    
    # Output configuration
    outputs: OutputConfig = Field(
        default_factory=_default_output_config,
        description="Output file and variable configuration"
    )
    
    # Conservation settings
    conservation_check: Dict[str, Any] = Field(
        default_factory=_default_conservation,
        description="Conservation checking parameters",
    )
    
    # Advanced settings
    heartbeat_functor: bool = Field(
        True,
        description="Whether to register heartbeat functor for daily aggregation"
    )
    
    debug_mode: bool = Field(
        False,
        description="Enable debug logging and additional checks"
    )
    
    @field_validator('tracked_isotope')
    @classmethod
    def validate_tracked_isotope(cls, v: Any) -> Any:
        """Validate isotope specification."""
        valid_isotopes = ["15N", "13C", "18O"]  # Future expansion
        if v not in valid_isotopes:
            warnings.warn(f"Isotope {v} may not be fully supported")
        return v
    
    @field_validator('pools')
    @classmethod
    def validate_pools(cls, v: Any) -> Any:
        """Validate pool configuration consistency."""
        required_pools = {"orgN", "NH4", "NO3"}
        configured_pools = set(v.keys())
        
        missing_pools = required_pools - configured_pools
        if missing_pools:
            raise ValueError(f"Missing required pools: {missing_pools}")
        
        return v
    

    
    @model_validator(mode='after')
    def validate_configuration_consistency(self):
        """Validate overall configuration consistency."""
        # First, validate process configuration consistency
        if hasattr(self, 'pools') and hasattr(self, 'processes'):
            pool_names = set(self.pools.keys())
            
            for process_name, process_config in self.processes.items():
                # Check source pool exists (except for atmospheric inputs)
                if process_config.source_pool not in pool_names and process_config.source_pool != "atmosphere":
                    raise ValueError(
                        f"Process {process_name} references unknown source pool: {process_config.source_pool}"
                    )
                
                # Check target pool exists (except for losses like N2O)
                if (process_config.target_pool not in pool_names and 
                    process_config.target_pool not in ["N2O", "N2", "atmosphere"]):
                    raise ValueError(
                        f"Process {process_name} references unknown target pool: {process_config.target_pool}"
                    )
        
        # Check that output variables reference valid pools/processes
        if hasattr(self, 'outputs') and hasattr(self, 'pools') and hasattr(self, 'processes'):
            pool_names = set(self.pools.keys())
            set(self.processes.keys())
            
            if hasattr(self.outputs, 'variables'):
                for var_name, var_config in self.outputs.variables.items():
                    parent_var = var_config.parent_variable
                    if parent_var:
                        if parent_var not in pool_names and not parent_var.endswith('_flux'):
                            warnings.warn(
                                f"Output variable {var_name} references unknown parent: {parent_var}"
                            )
        
        return self
    
    def get_enabled_processes(self) -> List[str]:
        """Get list of enabled process names."""
        return [name for name, config in self.processes.items() if config.enabled]
    
    def get_tracked_pools(self) -> List[str]:
        """Get list of tracked pool names."""
        return list(self.pools.keys())
    
    def validate_with_state_manager(self, state_variables: List[str]) -> Dict[str, List[str]]:
        """
        Validate configuration against StateManager variables.
        
        Parameters:
            state_variables: List of available StateManager variables
            
        Returns:
            Dict with 'missing_variables' and 'warnings' lists
        """
        issues = {"missing_variables": [], "warnings": []}
        
        # Check pool variables
        for pool_name, pool_config in self.pools.items():
            if pool_config.state_variable not in state_variables:
                issues["missing_variables"].append(pool_config.state_variable)
        
        # Check process flux variables
        for process_name, process_config in self.processes.items():
            if process_config.enabled and process_config.flux_variable not in state_variables:
                issues["missing_variables"].append(process_config.flux_variable)
        
        return issues


# Export main configuration models
__all__ = [
    'IsotopeConfigurationModel',
    'FractionationModelType',
    'AggregationMethod', 
    'FractionationParameters',
    'ProcessConfig',
    'PoolConfig',
    'OutputVariableConfig',
    'OutputConfig',
    'Nitrogen15Config',
    'get_default_15n_config',
]

