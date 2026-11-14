"""
Parameter Calibration Framework

This module provides advanced parameter calibration capabilities for the 
RELIC parameter management system, including:

- Parameter calibration hooks and callbacks
- Version control and provenance tracking
- Calibration result validation and rollback
- Integration with optimization algorithms
- Automatic parameter sensitivity analysis

Key Components:
    - ParameterCalibrator: Five derivative-free methods
    - GradientCalibrator: Gradient descent, Adam, and L-BFGS
    - CalibrationHooks: Event-driven calibration system
    - VersionManager: Parameter version control
    - CalibrationResult: Structured calibration results
    - CalibrationValidator: Result validation and QA

Example Usage:
    from relic.parameters.calibration import (
        CalibrationConfig,
        CalibrationMethod,
        ParameterBounds,
        ParameterCalibrator,
    )

    config = CalibrationConfig(
        method=CalibrationMethod.DIFFERENTIAL_EVOLUTION,
        target_parameters=["biosphere.photosynthesis.vcmax25"],
        parameter_bounds={
            "biosphere.photosynthesis.vcmax25": ParameterBounds(50.0, 120.0, scale="linear")
        },
    )
    calibrator = ParameterCalibrator(parameter_manager=manager)
    result = calibrator.calibrate(config, objective, observation_data)
"""

from .gradient_calibrator import (
    GradientCalibrator,
    GradientCalibrationConfig
)

from .calibrator import (
    ParameterCalibrator,
    CalibrationMethod,
    CalibrationConfig,
    ParameterBounds,
    ObjectiveFunction,
    MSEObjective,
    RMSEObjective,
    NashSutcliffeObjective
)

from .hooks import (
    CalibrationHooks,
    CalibrationEvent,
    CalibrationCallback,
    LoggingCallback,
    ProgressCallback,
    ConvergenceCallback,
    ParameterBoundsCallback,
    AdaptiveCallback,
    create_default_hooks
)

from .versioning import (
    VersionManager,
    ParameterVersion,
    VersionMetadata
)

from .results import (
    CalibrationResult,
    CalibrationMetrics,
    CalibrationReport
)

from .validator import (
    CalibrationValidator,
    ValidationRule,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationCategory,
    ConvergenceValidationRule,
    ParameterBoundsValidationRule,
    StatisticalValidationRule,
    PhysicalValidationRule,
    UncertaintyValidationRule,
    ComputationalValidationRule
)

# Main calibration interface
__all__ = [
    # Gradient-based calibration (NEW)
    'GradientCalibrator',
    'GradientCalibrationConfig',
    
    # Core calibration
    'ParameterCalibrator',
    'CalibrationMethod', 
    'CalibrationConfig',
    'ParameterBounds',
    'ObjectiveFunction',
    'MSEObjective',
    'RMSEObjective',
    'NashSutcliffeObjective',
    
    # Event system
    'CalibrationHooks',
    'CalibrationEvent',
    'CalibrationCallback',
    'LoggingCallback',
    'ProgressCallback',
    'ConvergenceCallback',
    'ParameterBoundsCallback',
    'AdaptiveCallback',
    'create_default_hooks',
    
    # Version control
    'VersionManager',
    'ParameterVersion',
    'VersionMetadata',
    
    # Results and reporting
    'CalibrationResult',
    'CalibrationMetrics',
    'CalibrationReport',
    
    # Validation
    'CalibrationValidator',
    'ValidationRule',
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity',
    'ValidationCategory',
    'ConvergenceValidationRule',
    'ParameterBoundsValidationRule',
    'StatisticalValidationRule',
    'PhysicalValidationRule',
    'UncertaintyValidationRule',
    'ComputationalValidationRule'
]
