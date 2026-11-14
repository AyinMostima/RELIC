"""
L3 System Tests - Unified Validation Framework

This package contains L3 system-level tests for the RELIC Unified Validation Framework.
L3 tests focus on system-wide validation, benchmark comparisons, real-world data integration,
and production readiness assessment.

Test Structure:
    - test_benchmark_validation.py: Benchmark dataset validation and model intercomparison
    - test_system_validation.py: Real-world integration and production readiness

Mathematical Framework:
    L3 system validation encompasses:
    - Statistical validation metrics: RMSE, R虏, NSE, MAE, Index of Agreement
    - Multi-scale validation: leaf 鈫?canopy 鈫?ecosystem 鈫?regional scales
    - Temporal validation: diurnal, seasonal, and inter-annual patterns
    - Benchmark comparisons: FluxNet, MODIS, climate datasets
    - Production readiness: performance, reliability, and accuracy metrics

Coverage:
    - FluxNet eddy covariance data integration and validation
    - MODIS satellite data comparison and scaling validation
    - Multi-model ensemble comparison and uncertainty quantification
    - Diurnal and seasonal pattern validation
    - Leaf-to-canopy scaling relationships
    - Ecosystem carbon balance validation
    - System performance and scalability testing
    - Operational readiness assessment

Statistical Metrics:
    - Root Mean Square Error: RMSE = 鈭?危(model - obs)虏 / n)
    - Nash-Sutcliffe Efficiency: NSE = 1 - 危(model - obs)虏 / 危(obs - obs_mean)虏
    - Coefficient of Determination: R虏 = (correlation coefficient)虏
    - Mean Absolute Error: MAE = 危|model - obs| / n
    - Willmott's Index of Agreement: d = 1 - 危(model - obs)虏 / 危(|model - obs_mean| + |obs - obs_mean|)虏

Production Readiness Criteria:
    - Performance: Execution time < 5min, Memory usage < 2GB
    - Data Quality: Completeness > 95%, Quality score > 0.8
    - Validation Coverage: L1 pass rate > 95%, L2 pass rate > 90%
    - Scientific Accuracy: Benchmark correlation > 0.7, Conservation residuals < 1e-4
    - System Reliability: Error rate < 1%, Uptime > 99%

References:
    - Nash & Sutcliffe (1970): River flow forecasting through conceptual models
    - Willmott (1981): On the validation of models
    - Taylor (2001): Summarizing multiple aspects of model performance in a single diagram
    - Pastorello et al. (2020): The FLUXNET2015 dataset and processing pipeline
    - Jung et al. (2020): Scaling carbon fluxes from eddy covariance sites to globe
"""

# L3 System test modules
from . import test_benchmark_validation
from . import test_system_validation

__all__ = [
    'test_benchmark_validation',
    'test_system_validation'
]

# Test configuration constants
L3_TEST_CONFIG = {
    'statistical_thresholds': {
        'excellent': {'rmse': 0.1, 'r2': 0.9, 'nse': 0.8, 'mae': 0.08},
        'good': {'rmse': 0.2, 'r2': 0.7, 'nse': 0.6, 'mae': 0.15},
        'acceptable': {'rmse': 0.4, 'r2': 0.5, 'nse': 0.3, 'mae': 0.3},
        'poor': {'rmse': float('inf'), 'r2': 0.0, 'nse': -float('inf'), 'mae': float('inf')}
    },
    'benchmark_datasets': {
        'fluxnet_sites': ['US-Ha1', 'DE-Hai', 'AU-How', 'BR-Sa1'],  # Example sites
        'modis_products': ['MOD17A2H', 'MCD15A3H'],  # GPP and LAI products
        'climate_datasets': ['ERA5', 'MERRA2', 'NCEP']
    },
    'scaling_validation': {
        'leaf_to_canopy': {'method': 'beer_lambert', 'tolerance': 0.15},
        'canopy_to_ecosystem': {'method': 'tower_footprint', 'tolerance': 0.25},
        'ecosystem_to_regional': {'method': 'upscaling_models', 'tolerance': 0.35}
    },
    'temporal_patterns': {
        'diurnal': {'gpp_peak_hours': (10, 14), 'et_peak_hours': (12, 16)},
        'seasonal': {'gpp_peak_doy': (150, 200), 'lai_peak_doy': (160, 180)},
        'interannual': {'trend_detection_years': 10, 'variability_threshold': 0.2}
    },
    'production_readiness': {
        'performance': {'max_execution_time': 300, 'max_memory_mb': 2000},
        'data_quality': {'min_completeness': 0.95, 'min_quality_score': 0.8},
        'validation_coverage': {'min_l1_pass_rate': 0.95, 'min_l2_pass_rate': 0.90},
        'scientific_accuracy': {'min_benchmark_correlation': 0.7, 'max_conservation_residual': 1e-4},
        'system_reliability': {'max_error_rate': 0.01, 'min_uptime': 0.99}
    }
}

# Common test fixtures and utilities
def get_l3_test_config():
    """Get L3 system test configuration."""
    return L3_TEST_CONFIG.copy()

def calculate_performance_metrics(observed, modeled):
    """
    Calculate comprehensive performance metrics for L3 validation.
    
    Parameters:
        observed : array-like
            Observed/benchmark values
        modeled : array-like
            Modeled/predicted values
            
    Returns:
        dict: Dictionary of performance metrics
    """
    import numpy as np
    
    # Remove any NaN or infinite values
    mask = np.isfinite(observed) & np.isfinite(modeled)
    obs = np.array(observed)[mask]
    mod = np.array(modeled)[mask]
    
    if len(obs) == 0:
        return {'rmse': float('inf'), 'r2': 0.0, 'nse': float('-inf'), 
               'mae': float('inf'), 'd_index': 0.0, 'bias': float('inf')}
    
    # Basic statistics
    obs_mean = np.mean(obs)
    mod_mean = np.mean(mod)
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((mod - obs) ** 2))
    
    # Coefficient of Determination (R虏) - manual calculation to avoid corrcoef issues
    r2 = 0.0
    if np.var(obs) > 1e-10 and np.var(mod) > 1e-10:
        try:
            # Manual correlation calculation
            obs_centered = obs - obs_mean
            mod_centered = mod - mod_mean
            
            numerator = np.sum(obs_centered * mod_centered)
            obs_ss = np.sum(obs_centered ** 2)
            mod_ss = np.sum(mod_centered ** 2)
            
            if obs_ss > 1e-10 and mod_ss > 1e-10:
                correlation = numerator / np.sqrt(obs_ss * mod_ss)
                correlation = np.clip(correlation, -1.0, 1.0)
                if np.isfinite(correlation):
                    r2 = correlation ** 2
        except Exception:
            r2 = 0.0
    
    # Nash-Sutcliffe Efficiency
    ss_res = np.sum((obs - mod) ** 2)
    ss_tot = np.sum((obs - obs_mean) ** 2)
    nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')
    
    # Mean Absolute Error
    mae = np.mean(np.abs(mod - obs))
    
    # Willmott's Index of Agreement
    numerator = np.sum((mod - obs) ** 2)
    denominator = np.sum((np.abs(mod - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    d_index = 1 - (numerator / denominator) if denominator > 0 else 0.0
    
    # Bias
    bias = mod_mean - obs_mean
    
    return {
        'rmse': rmse,
        'r2': r2, 
        'nse': nse,
        'mae': mae,
        'd_index': d_index,
        'bias': bias,
        'n_points': len(obs),
        'correlation': correlation if 'correlation' in locals() and np.isfinite(correlation) else 0.0
    }

def classify_performance(metrics, variable_type='default'):
    """
    Classify model performance based on metrics and variable type.
    
    Parameters:
        metrics : dict
            Performance metrics from calculate_performance_metrics
        variable_type : str
            Type of variable ('gpp', 'nee', 'et', 'lai', etc.)
            
    Returns:
        tuple: (performance_class, validation_level)
    """
    from relic.validation.framework.models import ValidationLevel
    
    config = get_l3_test_config()
    thresholds = config['statistical_thresholds']
    
    # Adjust thresholds based on variable type
    if variable_type == 'nee':
        # NEE is more challenging to predict
        thresholds = {
            'excellent': {'r2': 0.6, 'rmse': 0.3},
            'good': {'r2': 0.4, 'rmse': 0.5},
            'acceptable': {'r2': 0.2, 'rmse': 0.8},
            'poor': {'r2': 0.0, 'rmse': float('inf')}
        }
    elif variable_type in ['lai', 'ndvi']:
        # Remote sensing variables
        thresholds = {
            'excellent': {'r2': 0.8, 'rmse': 0.2},
            'good': {'r2': 0.6, 'rmse': 0.4},
            'acceptable': {'r2': 0.4, 'rmse': 0.6},
            'poor': {'r2': 0.0, 'rmse': float('inf')}
        }
    
    r2 = metrics.get('r2', 0.0)
    rmse = metrics.get('rmse', float('inf'))
    nse = metrics.get('nse', float('-inf'))
    
    if (r2 >= thresholds['excellent']['r2'] and 
        rmse <= thresholds['excellent']['rmse'] and
        nse >= thresholds['excellent'].get('nse', 0.8)):
        return 'excellent', ValidationLevel.INFO
    elif (r2 >= thresholds['good']['r2'] and
          rmse <= thresholds['good']['rmse'] and
          nse >= thresholds['good'].get('nse', 0.6)):
        return 'good', ValidationLevel.INFO
    elif (r2 >= thresholds['acceptable']['r2'] and
          rmse <= thresholds['acceptable']['rmse'] and
          nse >= thresholds['acceptable'].get('nse', 0.3)):
        return 'acceptable', ValidationLevel.WARNING
    else:
        return 'poor', ValidationLevel.ERROR

