"""
Sensitivity Analysis Module for RELIC Optimization Framework

This module provides comprehensive global sensitivity analysis capabilities
for biogeochemical model parameter screening and importance quantification.

Available Methods:
- Morris One-At-A-Time (OAT) screening method
- Sobol variance-based global sensitivity analysis

Scientific Background:
    Global Sensitivity Analysis explores the full parameter space to quantify
    the contribution of each parameter to variations in model outputs. Methods
    range from efficient screening (Morris) to comprehensive variance decomposition
    (Sobol), providing complementary insights for model analysis and optimization.

References:
    - Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer
    - Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments
    - Sobol, I.M. (1993). Sensitivity estimates for nonlinear mathematical models

Example Usage:
    ```python
    from relic.optimization.sensitivity import MorrisAnalyzer, SobolAnalyzer
    from relic.optimization.sensitivity import MorrisConfig, ParameterRange, SobolConfig

    # Define parameter ranges (the distribution is declared; declared units
    # must equal the ParameterSpec units of the parameter key)
    param_ranges = {
        'vcmax25': ParameterRange('vcmax25', 20.0, 80.0, distribution='uniform'),
        'jmax25': ParameterRange('jmax25', 50.0, 200.0, distribution='uniform'),
    }

    # Morris screening (r, p and grid jump come from optimization.sensitivity.*)
    morris_analyzer = MorrisAnalyzer(parameter_manager, MorrisConfig(aggregation_method='mean'))
    morris_result = morris_analyzer.analyze(model_runner, param_ranges)

    # Sobol analysis
    sobol_config = SobolConfig(aggregation_method='mean', estimator='saltelli')
    sobol_analyzer = SobolAnalyzer(parameter_manager, sobol_config)
    sobol_result = sobol_analyzer.analyze(model_runner, param_ranges)
    ```
"""

from .base import (
    SensitivityAnalyzer,
    SensitivityMethod,
    SensitivityConfig,
    SensitivityResult,
    ParameterRange
)

from .morris import (
    MorrisAnalyzer,
    MorrisConfig
)

from .sobol import (
    SobolAnalyzer, 
    SobolConfig
)

# Version information
__version__ = "1.0.0"
__author__ = "Yuchong Long"

# Convenience aliases
Morris = MorrisAnalyzer
Sobol = SobolAnalyzer

__all__ = [
    # Base classes and types
    'SensitivityAnalyzer',
    'SensitivityMethod',
    'SensitivityConfig', 
    'SensitivityResult',
    'ParameterRange',
    
    # Morris method
    'MorrisAnalyzer',
    'MorrisConfig',
    'Morris',
    
    # Sobol method
    'SobolAnalyzer',
    'SobolConfig',
    'Sobol',
]
