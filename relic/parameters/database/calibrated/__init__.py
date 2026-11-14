"""
Calibrated Parameter Database

This directory contains parameters that have been calibrated or optimized
using observational data, with complete provenance tracking and performance
metrics documentation.

Organization:
    site_specific/: Parameters calibrated for specific field sites
    regional/: Parameters calibrated for regional/ecosystem-level studies
    global/: Global parameter sets from multi-site calibrations

Calibration Metadata Requirements:
    - calibration_method: Algorithm used (e.g., Differential Evolution, Bayesian)
    - target_observations: Data used for calibration
    - performance_metrics: GOF statistics (R^2, RMSE, Nash-Sutcliffe, etc.)
    - calibration_period: Time period of calibration data
    - validation_period: Independent validation period (if available)
    - calibration_uncertainty: Parameter uncertainty estimates
    - prior_constraints: Prior distributions or bounds used

Quality Assurance:
    - Independent validation against withheld data
    - Cross-validation across different sites/conditions
    - Parameter identifiability analysis
    - Posterior parameter distribution analysis
    - Model structural uncertainty assessment

Data Sources:
    - FluxNet ecosystem monitoring sites
    - Long-term ecological research (LTER) sites
    - Forest inventory and monitoring plots
    - Satellite remote sensing calibration
    - Intensive field measurement campaigns

References:
    - Best practices for model calibration and validation
    - Parameter estimation uncertainty quantification
    - Multi-objective calibration approaches
    - Bayesian parameter estimation methods
"""

from typing import Dict, List, Any

__version__ = "1.0.0"
__tier__ = "calibrated"
__description__ = "Calibrated parameters with performance metrics and provenance"

# Calibration scales and organization
CALIBRATION_SCALES = {
    'site_specific': {
        'description': 'Parameters calibrated for individual field sites',
        'spatial_scale': '< 1 km^2',
        'temporal_scale': 'months to years',
        'data_sources': ['eddy covariance', 'field measurements', 'plot inventories'],
        'typical_parameters': 10-50
    },
    'regional': {
        'description': 'Parameters for ecosystem/biome-level applications',
        'spatial_scale': '10-10000 km^2',
        'temporal_scale': 'years to decades', 
        'data_sources': ['multiple sites', 'remote sensing', 'regional surveys'],
        'typical_parameters': 20-100
    },
    'global': {
        'description': 'Global parameter sets for Earth system models',
        'spatial_scale': 'continental to global',
        'temporal_scale': 'decades to centuries',
        'data_sources': ['global networks', 'synthesis studies', 'meta-analyses'],
        'typical_parameters': 50-200
    }
}

# Standard calibration methods
CALIBRATION_METHODS = {
    'differential_evolution': 'Global optimization using differential evolution',
    'simulated_annealing': 'Stochastic optimization with simulated annealing',
    'genetic_algorithm': 'Evolutionary optimization algorithm',
    'mcmc': 'Markov Chain Monte Carlo Bayesian estimation',
    'particle_swarm': 'Particle swarm optimization',
    'nelder_mead': 'Simplex optimization algorithm',
    'multi_objective': 'Multi-objective optimization (Pareto-optimal solutions)',
    'ensemble_kalman': 'Ensemble Kalman filter parameter estimation'
}

# Performance metrics for calibration quality
PERFORMANCE_METRICS = {
    'r_squared': 'Coefficient of determination',
    'rmse': 'Root mean square error', 
    'mae': 'Mean absolute error',
    'nash_sutcliffe': 'Nash-Sutcliffe efficiency',
    'willmott_index': 'Willmott index of agreement',
    'bias': 'Mean bias',
    'log_likelihood': 'Log-likelihood of observations',
    'aic': 'Akaike Information Criterion',
    'bic': 'Bayesian Information Criterion'
}

def get_scale_info(scale: str) -> Dict[str, Any]:
    """Get information about calibration scale."""
    return CALIBRATION_SCALES.get(scale, {})

def get_method_info(method: str) -> str:
    """Get description of calibration method."""
    return CALIBRATION_METHODS.get(method, "")

def list_scales() -> List[str]:
    """List available calibration scales."""
    return list(CALIBRATION_SCALES.keys())

def list_methods() -> List[str]:
    """List available calibration methods."""
    return list(CALIBRATION_METHODS.keys())

# Export main components
__all__ = [
    'CALIBRATION_SCALES',
    'CALIBRATION_METHODS', 
    'PERFORMANCE_METRICS',
    'get_scale_info',
    'get_method_info',
    'list_scales',
    'list_methods'
]