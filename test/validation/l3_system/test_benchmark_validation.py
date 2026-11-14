"""
L3 System Tests: Benchmark Validation and Model Comparison

Tests for system-level validation against established benchmarks,
model intercomparison, and performance validation against standard datasets.

Focus Areas:
1. Benchmark Dataset Validation (FluxNet, CMIP, etc.)
2. Model Intercomparison (vs. LPJ-GUESS, CLM, etc.)
3. Statistical Performance Metrics
4. System-Level Integration Testing
5. End-to-End Validation Workflows

Mathematical Framework:
    Statistical validation metrics:
    - Root Mean Square Error: RMSE = 鈭?危(model - obs)虏 / n)
    - Nash-Sutcliffe Efficiency: NSE = 1 - 危(model - obs)虏 / 危(obs - obs_mean)虏
    - Coefficient of Determination: R虏 = (危((model - model_mean)(obs - obs_mean)))虏 / (危(model - model_mean)虏 脳 危(obs - obs_mean)虏)
    - Mean Absolute Error: MAE = 危|model - obs| / n
    - Index of Agreement: d = 1 - 危(model - obs)虏 / 危(|model - obs_mean| + |obs - obs_mean|)虏
    
References:
    - Nash & Sutcliffe (1970): River flow forecasting through conceptual models
    - Willmott (1981): On the validation of models
    - Taylor (2001): Summarizing multiple aspects of model performance in a single diagram
    - FluxNet Data Processing: Pastorello et al. (2020)
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from relic.validation.framework.models import (
    ValidationLevel, ValidationCode, ValidationResultModel, ValidationReportModel, ValidationContext
)
from relic.validation.framework.engine import ValidationEngine
from relic.validation.framework.decorators import validated_rule


class TestBenchmarkDatasetValidation:
    """Test validation against benchmark datasets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
        
        # Create mock benchmark data (simulating FluxNet-like observations)
        self.benchmark_data = self._create_mock_benchmark_data()
        
        # Statistical performance thresholds (adjusted for GPP data scale)
        self.performance_thresholds = {
            'excellent': {'rmse': 3.0, 'r2': 0.9, 'nse': 0.8, 'mae': 2.5},
            'good': {'rmse': 5.0, 'r2': 0.7, 'nse': 0.6, 'mae': 4.0},
            'acceptable': {'rmse': 8.0, 'r2': 0.5, 'nse': 0.3, 'mae': 6.0},
            'poor': {'rmse': float('inf'), 'r2': 0.0, 'nse': -float('inf'), 'mae': float('inf')}
        }
    
    def _create_mock_benchmark_data(self) -> Dict[str, np.ndarray]:
        """Create mock benchmark dataset for testing."""
        np.random.seed(42)  # For reproducible tests
        
        n_points = 100
        
        # Simulate realistic carbon flux data (渭mol CO2 m-2 s-1)
        time_steps = np.arange(n_points)
        
        # Seasonal pattern for GPP
        seasonal_gpp = 15 + 10 * np.sin(2 * np.pi * time_steps / 365 * 12)  # Monthly cycle
        noise_gpp = np.random.normal(0, 2, n_points)
        observed_gpp = np.maximum(0, seasonal_gpp + noise_gpp)
        
        # NEE (net ecosystem exchange) - typically smaller magnitude
        seasonal_nee = -2 + 3 * np.sin(2 * np.pi * time_steps / 365 * 12 + np.pi/4)
        noise_nee = np.random.normal(0, 1, n_points)
        observed_nee = seasonal_nee + noise_nee
        
        # Evapotranspiration (mm day-1)
        seasonal_et = 3 + 2 * np.sin(2 * np.pi * time_steps / 365 * 12)
        noise_et = np.random.normal(0, 0.5, n_points)
        observed_et = np.maximum(0, seasonal_et + noise_et)
        
        return {
            'gpp': observed_gpp,
            'nee': observed_nee,
            'et': observed_et,
            'time_steps': time_steps
        }
    
    def _calculate_performance_metrics(self, observed: np.ndarray, modeled: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        # Convert to numpy arrays and ensure they're 1D
        obs_array = np.atleast_1d(np.asarray(observed, dtype=float))
        mod_array = np.atleast_1d(np.asarray(modeled, dtype=float))
        
        # Ensure same length
        min_len = min(len(obs_array), len(mod_array))
        obs_array = obs_array[:min_len]
        mod_array = mod_array[:min_len]
        
        # Remove any NaN, infinite, or extremely large values
        mask = (np.isfinite(obs_array) & np.isfinite(mod_array) & 
                (np.abs(obs_array) < 1e10) & (np.abs(mod_array) < 1e10))
        obs = obs_array[mask]
        mod = mod_array[mask]
        
        if len(obs) < 2:  # Need at least 2 points for correlation
            return {'rmse': float('inf'), 'r2': 0.0, 'nse': float('-inf'), 
                   'mae': float('inf'), 'd_index': 0.0, 'bias': float('inf')}
        
        # Basic statistics
        obs_mean = np.mean(obs)
        mod_mean = np.mean(mod)
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((mod - obs) ** 2))
        
        # Coefficient of Determination (R虏) - manual calculation to avoid corrcoef issues
        if len(obs) > 1:
            try:
                # Manual correlation calculation
                obs_std = np.std(obs, ddof=1)
                mod_std = np.std(mod, ddof=1)
                
                if obs_std > 1e-15 and mod_std > 1e-15:  # Avoid division by zero
                    obs_centered = obs - np.mean(obs)
                    mod_centered = mod - np.mean(mod)
                    correlation = np.sum(obs_centered * mod_centered) / ((len(obs) - 1) * obs_std * mod_std)
                    r2 = correlation ** 2
                    r2 = np.clip(r2, 0.0, 1.0)  # Ensure R虏 is in valid range
                else:
                    r2 = 0.0
            except (ValueError, FloatingPointError, ZeroDivisionError):
                r2 = 0.0
        else:
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
            'n_points': len(obs)
        }
    
    def _classify_performance(self, metrics: Dict[str, float]) -> Tuple[str, ValidationLevel]:
        """Classify model performance based on metrics."""
        if (metrics['r2'] >= self.performance_thresholds['excellent']['r2'] and 
            metrics['rmse'] <= self.performance_thresholds['excellent']['rmse'] and
            metrics['nse'] >= self.performance_thresholds['excellent']['nse']):
            return 'excellent', ValidationLevel.INFO
        elif (metrics['r2'] >= self.performance_thresholds['good']['r2'] and
              metrics['rmse'] <= self.performance_thresholds['good']['rmse'] and
              metrics['nse'] >= self.performance_thresholds['good']['nse']):
            return 'good', ValidationLevel.INFO
        elif (metrics['r2'] >= self.performance_thresholds['acceptable']['r2'] and
              metrics['rmse'] <= self.performance_thresholds['acceptable']['rmse'] and
              metrics['nse'] >= self.performance_thresholds['acceptable']['nse']):
            return 'acceptable', ValidationLevel.WARNING
        else:
            return 'poor', ValidationLevel.ERROR
    
    def test_fluxnet_gpp_benchmark_validation(self):
        """Test GPP validation against FluxNet-style benchmark data."""
        @validated_rule(
            rule_id="fluxnet_gpp_benchmark",
            layer="L3",
            domains=["biosphere"],
            tags=["benchmark", "fluxnet", "gpp"]
        )
        def validate_gpp_against_fluxnet(context):
            """Validate GPP against FluxNet benchmark data."""
            results = []
            
            # Get model predictions and benchmark data
            benchmark_data = context.config.get('benchmark_data', {})
            model_data = context.config.get('model_data', {})
            
            observed_gpp = benchmark_data.get('gpp', np.array([]))
            modeled_gpp = model_data.get('gpp', np.array([]))
            
            if len(observed_gpp) == 0 or len(modeled_gpp) == 0:
                return [ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_DATA,
                    message="Missing benchmark or model GPP data",
                    layer="L3",
                    scope="fluxnet_gpp_benchmark"
                )]
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(observed_gpp, modeled_gpp)
            performance_class, validation_level = self._classify_performance(metrics)
            
            # Determine validation code based on performance
            if performance_class == 'excellent':
                code = ValidationCode.GENERIC_VALIDATION_FAILED  # TODO: Add BENCHMARK_EXCELLENT
                message = f"GPP model performance excellent: R虏 = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}"
            elif performance_class == 'good':
                code = ValidationCode.GENERIC_VALIDATION_FAILED  # TODO: Add BENCHMARK_GOOD
                message = f"GPP model performance good: R虏 = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}"
            elif performance_class == 'acceptable':
                code = ValidationCode.GENERIC_VALIDATION_FAILED  # TODO: Add BENCHMARK_ACCEPTABLE
                message = f"GPP model performance acceptable: R虏 = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}"
            else:
                code = ValidationCode.GENERIC_VALIDATION_FAILED  # TODO: Add BENCHMARK_POOR
                message = f"GPP model performance poor: R虏 = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}"
            
            result = ValidationResultModel(
                level=validation_level,
                code=code,
                message=message,
                layer="L3",
                scope="fluxnet_gpp_benchmark",
                tags=["benchmark", "gpp", "fluxnet", performance_class],
                metrics=metrics
            )
            
            # Add detailed benchmark comparison
            result.add_detail(
                variable_name="gross_primary_productivity",
                cf_standard_name="gross_primary_productivity_of_carbon",
                units="渭mol CO2 m-2 s-1",
                value=metrics['bias'],  # Model bias
                statistics={
                    "observed_mean": float(np.mean(observed_gpp)),
                    "modeled_mean": float(np.mean(modeled_gpp)),
                    "observed_std": float(np.std(observed_gpp)),
                    "modeled_std": float(np.std(modeled_gpp)),
                    "performance_class": performance_class,
                    **metrics
                }
            )
            
            results.append(result)
            return results
        
        # Create realistic model data (with some systematic bias and noise)
        np.random.seed(123)  # Different seed for model data
        observed_gpp = self.benchmark_data['gpp']
        
        # Simulate model with reasonable performance
        modeled_gpp = observed_gpp * 0.9 + np.random.normal(0, 1.5, len(observed_gpp))
        modeled_gpp = np.maximum(0, modeled_gpp)  # No negative GPP
        
        test_context = ValidationContext(config={
            'benchmark_data': {'gpp': observed_gpp},
            'model_data': {'gpp': modeled_gpp}
        })
        
        results = validate_gpp_against_fluxnet(test_context)
        
        assert len(results) == 1
        result = results[0]
        
        # Should be good or acceptable performance (we designed it that way)
        assert result.level in [ValidationLevel.INFO, ValidationLevel.WARNING]
        assert result.metrics['r2'] > 0.5  # Should have reasonable correlation
        assert result.metrics['rmse'] < 10.0  # Reasonable RMSE for GPP
        assert result.metrics['n_points'] == len(observed_gpp)
        
        # Check details
        assert len(result.details) == 1
        detail = result.details[0]
        assert detail.variable_name == "gross_primary_productivity"
        assert detail.cf_standard_name == "gross_primary_productivity_of_carbon"
        assert "observed_mean" in detail.statistics
        assert "modeled_mean" in detail.statistics
        assert "performance_class" in detail.statistics
    
    def test_poor_model_performance_detection(self):
        """Test detection of poor model performance."""
        @validated_rule(
            rule_id="poor_performance_detection",
            layer="L3",
            domains=["biosphere"],
            tags=["benchmark", "performance"]
        )
        def detect_poor_performance(context):
            """Detect poor model performance against benchmark."""
            results = []
            
            benchmark_data = context.config.get('benchmark_data', {})
            model_data = context.config.get('model_data', {})
            
            observed = benchmark_data.get('test_var', np.array([]))
            modeled = model_data.get('test_var', np.array([]))
            
            metrics = self._calculate_performance_metrics(observed, modeled)
            performance_class, validation_level = self._classify_performance(metrics)
            
            result = ValidationResultModel(
                level=validation_level,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message=f"Model performance: {performance_class}",
                layer="L3",
                scope="poor_performance_detection",
                metrics=metrics
            )
            
            results.append(result)
            return results
        
        # Create poor model performance scenario
        observed_data = self.benchmark_data['gpp']
        # Model that's completely uncorrelated with observations
        poor_model = np.random.uniform(0, 50, len(observed_data))
        
        test_context = ValidationContext(config={
            'benchmark_data': {'test_var': observed_data},
            'model_data': {'test_var': poor_model}
        })
        
        results = detect_poor_performance(test_context)
        
        assert len(results) == 1
        result = results[0]
        
        # Should detect poor performance
        assert result.level == ValidationLevel.ERROR
        assert result.metrics['r2'] < 0.3  # Poor correlation
        assert "poor" in result.message.lower()


class TestModelIntercomparison:
    """Test model intercomparison validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    def test_multi_model_ensemble_validation(self):
        """Test validation against multiple model ensemble."""
        @validated_rule(
            rule_id="multi_model_ensemble",
            layer="L3",
            domains=["biosphere"],
            tags=["intercomparison", "ensemble"]
        )
        def validate_against_ensemble(context):
            """Validate against multi-model ensemble."""
            results = []
            
            ensemble_data = context.config.get('ensemble_models', {})
            target_model = context.config.get('target_model', {})
            
            if not ensemble_data or not target_model:
                return [ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_DATA,
                    message="Missing ensemble or target model data",
                    layer="L3",
                    scope="multi_model_ensemble"
                )]
            
            variable_name = context.config.get('variable', 'test_variable')
            
            # Calculate ensemble statistics
            ensemble_values = list(ensemble_data.values())
            ensemble_mean = np.mean(ensemble_values, axis=0)
            ensemble_std = np.std(ensemble_values, axis=0)
            ensemble_min = np.min(ensemble_values, axis=0)
            ensemble_max = np.max(ensemble_values, axis=0)
            
            target_values = target_model.get(variable_name, np.array([]))
            
            # Calculate how target model compares to ensemble
            within_std = np.mean(np.abs(target_values - ensemble_mean) <= ensemble_std)
            within_range = np.mean((target_values >= ensemble_min) & (target_values <= ensemble_max))
            
            # Calculate correlation with ensemble mean (manual implementation to avoid numpy.corrcoef issues)
            correlation = 0.0
            try:
                if (len(target_values) > 2 and len(ensemble_mean) > 2 and 
                    len(target_values) == len(ensemble_mean)):
                    # Check for valid (non-NaN, finite) data
                    target_valid = np.isfinite(target_values) & ~np.isnan(target_values)
                    ensemble_valid = np.isfinite(ensemble_mean) & ~np.isnan(ensemble_mean)
                    valid_mask = target_valid & ensemble_valid
                    
                    if np.sum(valid_mask) > 2:  # Need at least 3 points for meaningful correlation
                        target_clean = target_values[valid_mask]
                        ensemble_clean = ensemble_mean[valid_mask]
                        
                        # Manual correlation calculation (Pearson correlation coefficient)
                        target_mean = np.mean(target_clean)
                        ensemble_mean_val = np.mean(ensemble_clean)
                        
                        target_centered = target_clean - target_mean
                        ensemble_centered = ensemble_clean - ensemble_mean_val
                        
                        numerator = np.sum(target_centered * ensemble_centered)
                        target_ss = np.sum(target_centered ** 2)
                        ensemble_ss = np.sum(ensemble_centered ** 2)
                        
                        # Check for non-zero sum of squares
                        if target_ss > 1e-10 and ensemble_ss > 1e-10:
                            correlation = numerator / np.sqrt(target_ss * ensemble_ss)
                            # Ensure correlation is within valid range
                            correlation = np.clip(correlation, -1.0, 1.0)
                            if not np.isfinite(correlation):
                                correlation = 0.0
            except Exception:
                correlation = 0.0
            
            # Determine validation level
            if within_std > 0.8 and correlation > 0.8:
                level = ValidationLevel.INFO
                message = f"Model performance consistent with ensemble (within 1蟽: {within_std:.1%}, r = {correlation:.3f})"
            elif within_std > 0.6 and correlation > 0.6:
                level = ValidationLevel.WARNING
                message = f"Model performance moderately consistent with ensemble (within 1蟽: {within_std:.1%}, r = {correlation:.3f})"
            else:
                level = ValidationLevel.ERROR
                message = f"Model performance inconsistent with ensemble (within 1蟽: {within_std:.1%}, r = {correlation:.3f})"
            
            result = ValidationResultModel(
                level=level,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,  # TODO: Add ENSEMBLE_COMPARISON codes
                message=message,
                layer="L3",
                scope="multi_model_ensemble",
                tags=["intercomparison", "ensemble"],
                metrics={
                    "within_std_fraction": within_std,
                    "within_range_fraction": within_range,
                    "ensemble_correlation": correlation,
                    "n_ensemble_models": len(ensemble_values)
                }
            )
            
            # Add ensemble statistics detail
            result.add_detail(
                variable_name=variable_name,
                units="model_units",
                value=correlation,
                statistics={
                    "ensemble_mean": float(np.mean(ensemble_mean)),
                    "ensemble_std": float(np.mean(ensemble_std)),
                    "target_mean": float(np.mean(target_values)),
                    "target_std": float(np.std(target_values)),
                    "ensemble_spread": float(np.mean(ensemble_max - ensemble_min)),
                    "within_1sigma": within_std,
                    "within_minmax": within_range
                }
            )
            
            results.append(result)
            return results
        
        # Create ensemble data (multiple models)
        np.random.seed(42)
        n_points = 50
        base_pattern = np.sin(np.linspace(0, 4*np.pi, n_points)) * 10 + 20
        
        ensemble_models = {
            'model_a': base_pattern + np.random.normal(0, 1, n_points),
            'model_b': base_pattern * 1.1 + np.random.normal(0, 1.5, n_points),
            'model_c': base_pattern * 0.9 + np.random.normal(0, 0.8, n_points),
            'model_d': base_pattern + 2 + np.random.normal(0, 1.2, n_points)
        }
        
        # Target model that's reasonably consistent with ensemble
        target_model = {
            'test_variable': base_pattern + np.random.normal(0, 1.3, n_points)
        }
        
        test_context = ValidationContext(config={
            'ensemble_models': ensemble_models,
            'target_model': target_model,
            'variable': 'test_variable'
        })
        
        results = validate_against_ensemble(test_context)
        
        assert len(results) == 1
        result = results[0]
        
        # Should show reasonable consistency with ensemble
        assert result.level in [ValidationLevel.INFO, ValidationLevel.WARNING]
        assert result.metrics['within_std_fraction'] > 0.3
        assert result.metrics['ensemble_correlation'] > 0.5
        assert result.metrics['n_ensemble_models'] == 4
        
        # Check details
        detail = result.details[0]
        assert "ensemble_mean" in detail.statistics
        assert "target_mean" in detail.statistics
        assert "within_1sigma" in detail.statistics


class TestSystemLevelIntegration:
    """Test system-level integration and end-to-end validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    def test_end_to_end_validation_workflow(self):
        """Test complete end-to-end validation workflow."""
        @validated_rule(
            rule_id="system_integration_test",
            layer="L3",
            domains=["atmosphere", "biosphere", "hydrosphere"],
            tags=["system", "integration", "e2e"]
        )
        def system_integration_validation(context):
            """System-level integration validation."""
            results = []
            
            def _validate_variable_consistency(var_data: Dict[str, Any]) -> bool:
                """Simple variable consistency check."""
                required_fields = ['units', 'cf_standard_name']
                return all(field in var_data for field in required_fields)
            
            system_state = context.config.get('system_state', {})
            validation_config = context.config.get('validation_config', {})
            
            # Check system completeness
            required_domains = validation_config.get('required_domains', [])
            available_domains = list(system_state.keys())
            
            missing_domains = set(required_domains) - set(available_domains)
            
            if missing_domains:
                result = ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.MISSING_REQUIRED_DATA,
                    message=f"System missing required domains: {list(missing_domains)}",
                    layer="L3",
                    scope="system_integration_test"
                )
            else:
                # Check cross-domain consistency
                consistency_issues = 0
                total_checks = 0
                
                for domain, domain_data in system_state.items():
                    variables = domain_data.get('variables', {})
                    for var_name, var_data in variables.items():
                        total_checks += 1
                        if not _validate_variable_consistency(var_data):
                            consistency_issues += 1
                
                consistency_rate = 1.0 - (consistency_issues / total_checks) if total_checks > 0 else 1.0
                
                if consistency_rate >= 0.95:
                    level = ValidationLevel.INFO
                    message = f"System integration excellent: {consistency_rate:.1%} consistency"
                elif consistency_rate >= 0.8:
                    level = ValidationLevel.WARNING 
                    message = f"System integration good: {consistency_rate:.1%} consistency"
                else:
                    level = ValidationLevel.ERROR
                    message = f"System integration poor: {consistency_rate:.1%} consistency"
                
                result = ValidationResultModel(
                    level=level,
                    code=ValidationCode.GENERIC_VALIDATION_FAILED,  # TODO: Add SYSTEM_INTEGRATION codes
                    message=message,
                    layer="L3",
                    scope="system_integration_test",
                    tags=["system", "integration"],
                    metrics={
                        "consistency_rate": consistency_rate,
                        "total_checks": total_checks,
                        "consistency_issues": consistency_issues,
                        "n_domains": len(available_domains)
                    }
                )
                
                # Add system overview detail
                result.add_detail(
                    variable_name="system_consistency",
                    units="fraction",
                    value=consistency_rate,
                    statistics={
                        "available_domains": available_domains,
                        "total_variables": sum(len(d.get('variables', {})) for d in system_state.values()),
                        "total_consistency_checks": total_checks,
                        "failed_checks": consistency_issues
                    }
                )
            
            results.append(result)
            return results
        
        def _validate_variable_consistency(self, var_data: Dict[str, Any]) -> bool:
            """Simple variable consistency check."""
            required_fields = ['units', 'cf_standard_name']
            return all(field in var_data for field in required_fields)
        
        # Create comprehensive system state
        system_state = {
            'atmosphere': {
                'variables': {
                    'temperature': {
                        'units': 'K',
                        'cf_standard_name': 'air_temperature',
                        'values': np.random.normal(285, 10, 100)
                    },
                    'pressure': {
                        'units': 'Pa', 
                        'cf_standard_name': 'air_pressure',
                        'values': np.random.normal(101325, 1000, 100)
                    }
                }
            },
            'biosphere': {
                'variables': {
                    'gpp': {
                        'units': '渭mol CO2 m-2 s-1',
                        'cf_standard_name': 'gross_primary_productivity_of_carbon',
                        'values': np.random.uniform(0, 30, 100)
                    },
                    'lai': {
                        'units': 'm2 m-2',
                        'cf_standard_name': 'leaf_area_index',
                        'values': np.random.uniform(0.1, 8, 100)
                    }
                }
            },
            'hydrosphere': {
                'variables': {
                    'soil_moisture': {
                        'units': 'kg m-2',
                        'cf_standard_name': 'soil_moisture_content',
                        'values': np.random.uniform(50, 400, 100)
                    }
                }
            }
        }
        
        validation_config = {
            'required_domains': ['atmosphere', 'biosphere', 'hydrosphere']
        }
        
        test_context = ValidationContext(config={
            'system_state': system_state,
            'validation_config': validation_config
        })
        
        results = system_integration_validation(test_context)
        
        assert len(results) == 1
        result = results[0]
        
        # Should pass system integration (we designed consistent data)
        assert result.level == ValidationLevel.INFO
        assert result.metrics['consistency_rate'] == 1.0  # All variables have required fields
        assert result.metrics['n_domains'] == 3
        assert result.metrics['total_checks'] == 5  # 5 variables total
        
        # Check system detail
        detail = result.details[0]
        assert detail.variable_name == "system_consistency"
        assert detail.value == 1.0
        assert len(detail.statistics['available_domains']) == 3
    
    def test_performance_scalability_validation(self):
        """Test system performance and scalability validation."""
        @validated_rule(
            rule_id="performance_scalability",
            layer="L3",
            domains=["system"],
            tags=["performance", "scalability"]
        )
        def validate_performance_scalability(context):
            """Validate system performance and scalability characteristics."""
            results = []
            
            performance_data = context.config.get('performance_data', {})
            scalability_thresholds = context.config.get('scalability_thresholds', {})
            
            # Extract performance metrics
            execution_time = performance_data.get('execution_time', 0.0)
            memory_usage = performance_data.get('memory_usage_mb', 0.0)
            rule_count = performance_data.get('rule_count', 0)
            data_points = performance_data.get('data_points', 0)
            
            # Calculate derived metrics
            time_per_rule = execution_time / rule_count if rule_count > 0 else float('inf')
            time_per_datapoint = execution_time / data_points if data_points > 0 else float('inf')
            memory_per_datapoint = memory_usage / data_points if data_points > 0 else float('inf')
            
            # Performance thresholds
            max_execution_time = scalability_thresholds.get('max_execution_time', 60.0)  # seconds
            max_memory_usage = scalability_thresholds.get('max_memory_mb', 1000.0)  # MB
            max_time_per_rule = scalability_thresholds.get('max_time_per_rule', 1.0)  # seconds
            
            # Evaluate performance
            performance_issues = []
            
            if execution_time > max_execution_time:
                performance_issues.append(f"Execution time {execution_time:.1f}s exceeds limit {max_execution_time}s")
            
            if memory_usage > max_memory_usage:
                performance_issues.append(f"Memory usage {memory_usage:.1f}MB exceeds limit {max_memory_usage}MB")
            
            if time_per_rule > max_time_per_rule:
                performance_issues.append(f"Time per rule {time_per_rule:.3f}s exceeds limit {max_time_per_rule}s")
            
            # Determine validation result
            if not performance_issues:
                level = ValidationLevel.INFO
                code = ValidationCode.GENERIC_VALIDATION_FAILED  # TODO: Add PERFORMANCE_EXCELLENT
                message = f"System performance excellent: {execution_time:.1f}s, {memory_usage:.1f}MB"
            else:
                level = ValidationLevel.WARNING if len(performance_issues) <= 2 else ValidationLevel.ERROR
                code = ValidationCode.GENERIC_VALIDATION_FAILED  # TODO: Add PERFORMANCE_POOR
                message = f"Performance issues detected: {'; '.join(performance_issues)}"
            
            result = ValidationResultModel(
                level=level,
                code=code,
                message=message,
                layer="L3",
                scope="performance_scalability",
                tags=["performance", "scalability"],
                metrics={
                    "execution_time_seconds": execution_time,
                    "memory_usage_mb": memory_usage,
                    "time_per_rule": time_per_rule,
                    "time_per_datapoint": time_per_datapoint,
                    "memory_per_datapoint": memory_per_datapoint,
                    "performance_issues_count": len(performance_issues)
                }
            )
            
            # Add performance detail
            result.add_detail(
                variable_name="system_performance",
                units="mixed",
                value=execution_time,
                statistics={
                    "rule_count": rule_count,
                    "data_points": data_points,
                    "execution_time_seconds": execution_time,
                    "memory_usage_mb": memory_usage,
                    "rules_per_second": rule_count / execution_time if execution_time > 0 else 0,
                    "datapoints_per_second": data_points / execution_time if execution_time > 0 else 0,
                    "performance_issues": performance_issues
                }
            )
            
            results.append(result)
            return results
        
        # Test good performance scenario
        good_performance = {
            'performance_data': {
                'execution_time': 5.2,  # seconds
                'memory_usage_mb': 150.0,
                'rule_count': 25,
                'data_points': 10000
            },
            'scalability_thresholds': {
                'max_execution_time': 60.0,
                'max_memory_mb': 1000.0,
                'max_time_per_rule': 1.0
            }
        }
        
        test_context = ValidationContext(config=good_performance)
        results = validate_performance_scalability(test_context)
        
        assert len(results) == 1
        result = results[0]
        
        # Should show excellent performance
        assert result.level == ValidationLevel.INFO
        assert result.metrics['execution_time_seconds'] == 5.2
        assert result.metrics['memory_usage_mb'] == 150.0
        assert result.metrics['time_per_rule'] < 1.0
        assert result.metrics['performance_issues_count'] == 0
        
        # Check performance detail
        detail = result.details[0]
        assert detail.variable_name == "system_performance"
        assert detail.statistics['rule_count'] == 25
        assert detail.statistics['rules_per_second'] > 4.0  # Good throughput
        assert len(detail.statistics['performance_issues']) == 0


class TestL3SystemSuite:
    """Test complete L3 system validation suite."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    def test_complete_l3_system_validation(self):
        """Test execution of complete L3 system validation suite."""
        
        # Register L3 system rules
        @validated_rule(
            rule_id="l3_benchmark_suite",
            layer="L3",
            domains=["biosphere"],
            tags=["system", "benchmark"]
        )
        def l3_benchmark_suite(context):
            """L3 benchmark validation suite."""
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="L3 benchmark suite completed successfully",
                layer="L3",
                scope="l3_benchmark_suite"
            )]
        
        @validated_rule(
            rule_id="l3_integration_suite", 
            layer="L3",
            domains=["system"],
            tags=["system", "integration"]
        )
        def l3_integration_suite(context):
            """L3 integration validation suite."""
            return [ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="L3 integration suite completed with minor issues",
                layer="L3",
                scope="l3_integration_suite"
            )]
        
        @validated_rule(
            rule_id="l3_performance_suite",
            layer="L3",
            domains=["system"], 
            tags=["system", "performance"]
        )
        def l3_performance_suite(context):
            """L3 performance validation suite."""
            return [ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="L3 performance suite detected optimization opportunities",
                layer="L3",
                scope="l3_performance_suite"
            )]
        
        # Execute complete L3 suite
        context = ValidationContext(
            domain_filter=["biosphere", "system"],
            execution_layer="L3",
            config={
                "validation_mode": "comprehensive",
                "benchmark_mode": True,
                "performance_analysis": True
            }
        )
        
        report = self.engine.run(layers=["L3"], context=context)
        
        # Verify comprehensive L3 validation
        assert isinstance(report, ValidationReportModel)
        assert len(report.results) >= 3
        
        # Check L3-specific results
        l3_results = [r for r in report.results if r.layer == "L3"]
        assert len(l3_results) >= 3
        
        # Check validation level distribution
        info_count = len([r for r in l3_results if r.level == ValidationLevel.INFO])
        warning_count = len([r for r in l3_results if r.level == ValidationLevel.WARNING])
        error_count = len([r for r in l3_results if r.level == ValidationLevel.ERROR])
        
        assert info_count >= 1
        assert warning_count >= 1
        assert error_count >= 1
        
        # Verify L3 summary statistics
        assert report.summary["total"] >= 3
        assert "L3" in report.summary["by_layer"]
        assert report.summary["by_layer"]["L3"] >= 3
        
        # Test report status
        assert report.has_warnings() is True
        assert report.has_errors() is True
        
        # Verify execution metadata includes L3 information
        assert "execution_metadata" in report.to_dict()


if __name__ == "__main__":
    pytest.main([__file__])

