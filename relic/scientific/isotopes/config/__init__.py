"""
Isotope Configuration Module

This module provides the single Pydantic configuration root for coordinated
13C, 15N, and 18O tracking.

Key Models:
    - IsotopeConfig: Main configuration model
    - ProcessConfig: Process-specific fractionation settings
    - PoolConfig: Pool mapping and initial conditions
    - OutputConfig: NetCDF output specifications

Usage:
    from relic.scientific.isotopes.config import IsotopeConfig
    
    # Load default configuration
    config = get_default_isotope_config()
    
    # Customize specific parameters
    config.processes['nitrification'].fractionation.alpha_0 = 0.990
    config.outputs.output_file = "./results/my_isotope_run.nc"
    
    # Validate configuration
    validation_issues = config.validate_with_state_manager(state_variables)
"""

from .models import (
    IsotopeConfigurationModel,
    FractionationModelType,
    AggregationMethod,
    FractionationParameters,
    ProcessConfig,
    PoolConfig, 
    OutputVariableConfig,
    OutputConfig,
)
from .multi_isotope_models import (
    CarbonProcessType,
    IsotopeConfig,
    IsotopeSpecies,
    IsotopeSpeciesConfig,
    OxygenProcessType,
    get_default_isotope_config,
)

__all__ = [
    'IsotopeConfigurationModel',
    'FractionationModelType',
    'AggregationMethod',
    'FractionationParameters', 
    'ProcessConfig',
    'PoolConfig',
    'OutputVariableConfig',
    'OutputConfig',
    'CarbonProcessType',
    'IsotopeConfig',
    'IsotopeSpecies',
    'IsotopeSpeciesConfig',
    'OxygenProcessType',
    'get_default_isotope_config',
]
