"""
RELIC Optimization Framework

This module exposes optimization algorithm components for biogeochemical model
analysis, including parameter calibration, data assimilation, sensitivity analysis,
and uncertainty quantification. Component import availability does not imply that a
production model-evaluation runtime is bound.

The optimization framework follows a modular design with four main subsystems:

1. Calibration: Parameter optimization using various algorithms (facade to parameters.calibration)
2. Data Assimilation: State estimation methods (EnKF, variational)
3. Sensitivity Analysis: Global sensitivity analysis (Morris, Sobol)  
4. Uncertainty Quantification: Uncertainty propagation (ensemble, bootstrap)

Production Boundary:
- Algorithm components are imported eagerly; the subpackages have hard
  dependencies only, so an import failure propagates (it is never recorded
  as an unavailable component).
- Public model execution remains [N] runtime_unbound until it is sealed to the
  canonical ParameterManager, GraphIR, compile_graph, ExecutionPlan, GraphExecutor,
  Simulator, observation/unit contracts, and state/RNG transaction owners.
- This package does not own a second simulator, state manager, parameter manager,
  compiler, execution runtime, registry, or cache.

References:
    - Vrugt, J.A. et al. (2008). Treatment of input uncertainty in hydrologic modeling
    - Evensen, G. (2009). Data Assimilation: The Ensemble Kalman Filter
    - Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer
    - Smith, R.C. (2014). Uncertainty Quantification: Theory, Implementation, and Applications
"""

from typing import Any

from .calibration import (
    ParameterCalibrator,
    CalibrationConfig,
    CalibrationMethod,
    CalibrationResult,
    CalibrationReport,
)
from .assimilation import (
    Assimilator,
    AssimilationResult,
    EnsembleKalmanFilter,
    EnKFConfig,
    ThreeDVar,
    VariationalConfig,
)
from .sensitivity import (
    SensitivityAnalyzer,
    SensitivityMethod,
    SensitivityConfig,
    SensitivityResult,
    ParameterRange,
    MorrisAnalyzer,
    MorrisConfig,
    SobolAnalyzer,
    SobolConfig,
)
from .uncertainty import (
    UncertaintyQuantifier,
    UQMethod,
    UQConfig,
    UQResult,
    UncertainParameter,
    DistributionType,
    EnsembleUQ,
    EnsembleConfig,
    BootstrapUQ,
    BootstrapConfig,
)

_ALGORITHM_COMPONENT_RUNTIME_CONTRACT = {
    "runtime_ready": False,
    "status": "[N]",
    "contract": "algorithm_component_only",
    "runtime_blocker": "runtime_unbound",
}

_OPTIMIZER_COMPONENTS = {
    "calibration": ("ParameterCalibrator",),
    "assimilation": ("EnsembleKalmanFilter", "ThreeDVar"),
    "sensitivity": ("MorrisAnalyzer", "SobolAnalyzer"),
    "uncertainty": ("EnsembleUQ", "BootstrapUQ"),
}

_OPTIMIZER_RUNTIME_REQUIREMENTS = {
    "calibration": (
        "ParameterManager parameter, bounds, transform, and unit binding",
        "sealed GraphIR compiled by compile_graph into an ExecutionPlan",
        "GraphExecutor and Simulator model-evaluation binding",
        "declared objective, observation projection, shape, and unit contract",
        "per-evaluation StateManager, ParameterManager, forcing-cursor, and RNG rollback-replay",
    ),
    "assimilation": (
        "StateManager and Simulator forecast-state binding",
        "sealed GraphIR, ExecutionPlan, and GraphExecutor forecast binding",
        "declared observation operator, variable, shape, unit, domain, and time contract",
        "symmetric positive-semidefinite covariance contract with solver-compatible regularity evidence",
        "atomic analysis-state validation, commit, and rollback",
        "restartable ensemble RNG substream and durable checkpoint binding when stochastic",
    ),
    "sensitivity": (
        "ParameterManager parameter-range, transform, bounds, and unit binding",
        "sealed GraphIR, ExecutionPlan, and GraphExecutor model-evaluation binding",
        "declared output statistic, shape, unit, and aggregation contract",
        "per-sample StateManager, ParameterManager, forcing-cursor, and RNG isolation",
        "method-specific sample-design and estimator-validity evidence",
    ),
    "uncertainty": (
        "ParameterManager-backed marginal distribution, bounds, unit, and correlation binding",
        "sealed GraphIR, ExecutionPlan, and GraphExecutor model-evaluation binding",
        "declared output quantity, shape, unit, and aggregation contract",
        "per-member StateManager, ParameterManager, forcing-cursor, and RNG isolation",
        "restartable ensemble substream and durable checkpoint binding",
        "method-specific sampling and interval-estimator validity evidence",
        "declared observation and residual contract when observation-conditioned",
    ),
}

# Version information
__version__ = "1.0.0"
__author__ = "Yuchong Long"

def get_available_optimizers() -> dict[str, dict[str, Any]]:
    """Return component availability separately from runtime readiness.

    The components are imported eagerly with the package (an import failure
    fails the package import), so every listed component is available; the
    runtime contract fields state that no model-evaluation runtime is bound.
    """
    metadata: dict[str, dict[str, Any]] = {}
    for family, component_names in _OPTIMIZER_COMPONENTS.items():
        metadata[family] = {
            "component_available": True,
            "items": list(component_names),
            "error": None,
            "requirements": list(_OPTIMIZER_RUNTIME_REQUIREMENTS[family]),
            **_ALGORITHM_COMPONENT_RUNTIME_CONTRACT,
        }
    return metadata

__all__ = [
    # Calibration
    'ParameterCalibrator',
    'CalibrationConfig',
    'CalibrationMethod', 
    'CalibrationResult',
    'CalibrationReport',
    
    # Data Assimilation
    'Assimilator',
    'AssimilationResult',
    'EnsembleKalmanFilter',
    'EnKFConfig',
    'ThreeDVar',
    'VariationalConfig',
    
    # Sensitivity Analysis
    'SensitivityAnalyzer',
    'SensitivityMethod',
    'SensitivityConfig',
    'SensitivityResult',
    'ParameterRange',
    'MorrisAnalyzer',
    'MorrisConfig',
    'SobolAnalyzer',
    'SobolConfig',
    
    # Uncertainty Quantification
    'UncertaintyQuantifier',
    'UQMethod',
    'UQConfig',
    'UQResult',
    'UncertainParameter',
    'DistributionType',
    'EnsembleUQ',
    'EnsembleConfig',
    'BootstrapUQ',
    'BootstrapConfig',
    'get_available_optimizers'
]
