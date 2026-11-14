"""
Configuration System Entry Point for RELIC

This module provides the main entry point for the RELIC configuration system,
exposing the key classes and functions needed for configuration management
throughout the ecosystem modeling framework.

The configuration system follows a hierarchical approach with validation,
type checking, and environment-specific overrides. It integrates with
Pydantic for robust data validation and Hydra for flexible configuration
composition.

Architecture:
    - Pydantic models for type-safe configuration definitions
    - Hierarchical configuration loading with overrides
    - Environment-specific configuration management  
    - Runtime configuration validation and updates
    - Integration with scientific parameter databases

References:
    - Pydantic documentation for data validation
    - Hydra configuration framework
    - OmegaConf for hierarchical configuration
"""

from pathlib import Path
from typing import Any, Optional

# Import main configuration components
from .models import (
    RELICConfig,
    PhysicsConfig, 
    SimulationConfig,
    StochasticConfig,
    OptimizationConfig,
    DataConfig,
    ParametersConfig,
    ComputationConfig,
    LoggingConfig,
    FeatureFlagConfig,
    ParameterValidationLevel,
    IntegrationMethod,
    DeviceType
)

from .validator import (
    ConfigValidator,
    validate_configuration,
    ValidationResult,
    ValidationMessage,
    ValidationSeverity
)

from .loader import (
    ConfigLoader,
    load_config,
    save_config,
    merge_configs,
    export_schema,
    configuration_environment,
    resolve_config_source,
)

# Version information
__version__ = "2.0.0"
__author__ = "Yuchong Long"

# The configuration file locations have one owner:
# relic.core.config.loader.DEFAULT_CONFIG_PATHS, read by resolve_config_source.

# Configuration schema version for validation
CONFIG_SCHEMA_VERSION = "2.0.0"

# Cached global configuration for CLI and tooling: (inputs key, configuration).
_GLOBAL_CONFIG_CACHE: Optional[tuple[tuple[Any, ...], RELICConfig]] = None


def get_global_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[list[str] | dict[str, Any]] = None,
    validate: bool = True,
    env_overrides: bool = True,
) -> RELICConfig:
    """Load the global configuration through the one source rule, cached per inputs.

    The source follows ``resolve_config_source`` (a named file that does not
    exist raises FileNotFoundError; without one the first existing
    DEFAULT_CONFIG_PATHS file, else the logged global default configuration).
    The cache key holds the resolved source and its modification time, the
    overrides, both flags and the configuration environment variables, so a
    changed RELIC_CONFIG_PATH, file or override variable is read again.
    """
    global _GLOBAL_CONFIG_CACHE
    source = resolve_config_source(config_path)
    normalized_overrides: Optional[list[str]]
    if overrides is None:
        normalized_overrides = None
    elif isinstance(overrides, dict):
        normalized_overrides = [f"{k}={v}" for k, v in overrides.items()]
    else:
        normalized_overrides = list(overrides)
    key = (
        str(source) if source is not None else None,
        source.stat().st_mtime_ns if source is not None and source.is_file() else None,
        tuple(normalized_overrides) if normalized_overrides is not None else None,
        validate,
        env_overrides,
        configuration_environment() if env_overrides else (),
    )
    if _GLOBAL_CONFIG_CACHE is not None and _GLOBAL_CONFIG_CACHE[0] == key:
        return _GLOBAL_CONFIG_CACHE[1]
    loader = ConfigLoader(validate_on_load=validate)
    config = loader.load_config(
        config_path=source,
        overrides=normalized_overrides,
        env_overrides=env_overrides,
    )
    _GLOBAL_CONFIG_CACHE = (key, config)
    return config

# Export functions for public API
__all__ = [
    # Main configuration classes
    'RELICConfig',
    'PhysicsConfig',
    'SimulationConfig', 
    'StochasticConfig',
    'OptimizationConfig',
    'DataConfig',
    'ParametersConfig',
    'ComputationConfig',
    'LoggingConfig',
    'FeatureFlagConfig',
    
    # Validation components
    'ConfigValidator',
    'validate_configuration', 
    'ValidationResult',
    'ValidationMessage',
    'ValidationSeverity',
    
    # Loading utilities
    'ConfigLoader',
    'load_config',
    'save_config',
    'merge_configs',
    'export_schema',
    
    'get_global_config',
    'resolve_config_source',
    'validate_configuration',
    
    # Enums
    'ParameterValidationLevel',
    'IntegrationMethod',
    'DeviceType',
    
    # Constants
    'CONFIG_SCHEMA_VERSION'
]
