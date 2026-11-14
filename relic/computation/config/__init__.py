"""
Configuration System for RELIC Computation Layer

This package provides comprehensive configuration management for operators
and solvers, integrating Hydra structured configs with ParameterManager
for type-safe, validated parameter management.

Key Components:
    - HydraTemplates: YAML templates for Hydra configuration
    - ConfigLoader: Unified loader integrating multiple config sources
    - Runtime provider validation through OperatorRegistry and SolverRegistry
    - Parameter validation through Factory, Pydantic, and ParameterManager

Usage Example:
    >>> from relic.computation.config import (
    ...     load_computation_config,
    ...     validate_config,
    ... )
    >>>
    >>> # Load configuration
    >>> config = load_computation_config(
    ...     config_path="conf",
    ...     config_name="operators"
    ... )
    >>>
    >>> # Instantiate operator
    >>> operator = config.instantiate_operator("gradient", dx=50.0)
    >>> errors = validate_config(config)

References:
    - Hydra Documentation: https://hydra.cc/
    - CF-1.8 Conventions: http://cfconventions.org/
    - FAIR Data Principles: https://www.go-fair.org/fair-principles/
"""

# Import config loader
from .config_loader import (
    ComputationConfig,
    load_computation_config,
    auto_detect_device,
    validate_config
)

# Version
__version__ = "0.1.0"

# Public API
__all__ = [
    # Config Loader
    'ComputationConfig',
    'load_computation_config',
    'auto_detect_device',
    'validate_config'
]
