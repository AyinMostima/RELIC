"""
Optimization Framework - Parameter Calibration Namespace

This module only re-exports the canonical parameter-calibration algorithm
components owned by relic.parameters.calibration. It owns no model runner,
runtime transaction, optimizer implementation, state, or parameter lifecycle.

Re-exported components:
- ParameterCalibrator: Owner of the five derivative-free methods only
- CalibrationResult: Structured calibration results with metrics
- CalibrationConfig: Configuration management
- CalibrationMethod: Shared method identity; gradient values are rejected here
- CalibrationValidator: Comprehensive validation system

Gradient descent, Adam, and L-BFGS are owned by ``GradientCalibrator`` in
``relic.parameters.calibration``. This compatibility namespace does not
re-export that owner and must not be described as one eight-algorithm facade.
"""

# Re-export core calibration functionality
from relic.parameters.calibration.calibrator import (
    ParameterCalibrator,
    CalibrationConfig, 
    CalibrationMethod,
    ObjectiveType
)
from relic.parameters.calibration.results import (
    CalibrationResult,
    CalibrationReport,
    CalibrationMetrics
)
from relic.parameters.calibration.validator import (
    CalibrationValidator,
    ValidationSeverity,
    ValidationRule
)
from relic.parameters.calibration.hooks import (
    CalibrationHooks,
    CalibrationEvent,
    create_default_hooks
)
from relic.parameters.calibration.versioning import (
    VersionManager,
    ParameterVersion,
    VersionedParameter
)

# Re-export all for convenience
__all__ = [
    # Core calibration (re-exported)
    'ParameterCalibrator',
    'CalibrationConfig',
    'CalibrationMethod', 
    'ObjectiveType',
    'CalibrationResult',
    'CalibrationReport',
    'CalibrationMetrics',
    'CalibrationValidator',
    'ValidationSeverity',
    'ValidationRule',
    'CalibrationHooks',
    'CalibrationEvent',
    'create_default_hooks',
    'VersionManager',
    'ParameterVersion',
    'VersionedParameter',
]
