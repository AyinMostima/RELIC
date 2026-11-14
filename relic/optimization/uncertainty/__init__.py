"""
Uncertainty Quantification Module for RELIC Optimization Framework

This module provides comprehensive uncertainty quantification capabilities
for biogeochemical model analysis including ensemble methods, bootstrap
estimation, and Bayesian approaches.

Available Methods:
- Ensemble uncertainty quantification with Monte Carlo sampling
- Bootstrap methods for non-parametric confidence intervals
- Future: Bayesian inference and polynomial chaos expansion

Scientific Background:
    Uncertainty quantification (UQ) aims to characterize and propagate uncertainties
    in model inputs through to model outputs. This is essential for understanding
    model reliability, supporting decision-making under uncertainty, and providing
    confidence bounds on predictions.

References:
    - Smith, R.C. (2014). Uncertainty Quantification: Theory, Implementation, and Applications
    - Xiu, D. (2010). Numerical Methods for Stochastic Computations
    - Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap
    - Ghanem, R.G. et al. (2017). Handbook of Uncertainty Quantification

Example Usage:
    ```python
    from relic.optimization.uncertainty import BootstrapConfig, BootstrapUQ
    from relic.optimization.uncertainty import EnsembleConfig, EnsembleUQ
    from relic.optimization.uncertainty import DistributionType, UncertainParameter

    # Define uncertain parameters (a NORMAL law declares mean and std;
    # declared units must equal the ParameterSpec units of the key)
    uncertain_params = {
        'vcmax25': UncertainParameter(
            'vcmax25',
            distribution=DistributionType.NORMAL,
            mean=50.0, std=10.0,
            lower_bound=20.0, upper_bound=80.0,
        )
    }

    # Ensemble uncertainty quantification (sizes come from optimization.uncertainty.*)
    ensemble_uq = EnsembleUQ(parameter_manager, EnsembleConfig(sampling_strategy='lhs'))
    ensemble_result = ensemble_uq.quantify(model_runner, uncertain_params)

    # Bootstrap confidence intervals
    bootstrap_config = BootstrapConfig(bootstrap_type='parametric', ci_method='percentile')
    bootstrap_uq = BootstrapUQ(parameter_manager, bootstrap_config)
    bootstrap_result = bootstrap_uq.quantify(model_runner, uncertain_params)
    ```
"""

from .base import (
    UncertaintyQuantifier,
    UQMethod,
    UQConfig, 
    UQResult,
    UncertainParameter,
    DistributionType
)

from .ensemble import (
    EnsembleUQ,
    EnsembleConfig
)

from .bootstrap import (
    BootstrapUQ,
    BootstrapConfig
)

# Version information
__version__ = "1.0.0"
__author__ = "Yuchong Long"

# Convenience aliases
Ensemble = EnsembleUQ
Bootstrap = BootstrapUQ

__all__ = [
    # Base classes and types
    'UncertaintyQuantifier',
    'UQMethod',
    'UQConfig',
    'UQResult',
    'UncertainParameter',
    'DistributionType',
    
    # Ensemble method
    'EnsembleUQ',
    'EnsembleConfig',
    'Ensemble',
    
    # Bootstrap method  
    'BootstrapUQ',
    'BootstrapConfig',
    'Bootstrap',
]
