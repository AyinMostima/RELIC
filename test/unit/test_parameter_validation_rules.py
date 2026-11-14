"""
Comprehensive L1 Unit Tests for Parameter Validation Rules

This test suite validates the parameter validation rules implemented in 
relic/validation/framework/rules/parameters.py. Tests cover all five 
core validation functions with various edge cases and scenarios.

Test Categories:
1. Parameter Bounds Validation Tests
2. Distribution Parameter Consistency Tests  
3. Log-Scale Parameter Validation Tests
4. Spatial Correlation Validation Tests
5. Parameter Schema Completeness Tests
6. Integration and Error Handling Tests

Mathematical Framework:
    For parameter p with bounds [min, max]: Valid(p) 鉄?min 鈮?p 鈮?max
    For distribution D with parameters 胃: Consistent(胃) 鉄?胃 鈭?Domain(D)
    For log-scale parameter p: Valid(p) 鉄?p > 0 鈭?bounds > 0
    For correlation length 位: Valid(位) 鉄?位 > 0

References:
    - RELIC Parameter Specification
    - Statistical Distribution Theory
    - Spatial Statistics Theory
"""

import pytest
import torch
import sys
import os

# Add relic to Python path
relic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, relic_path)

from relic.validation.framework.rules.parameters import (
    validate_parameter_bounds,
    validate_distribution_parameter_consistency, 
    validate_log_scale_parameters,
    validate_spatial_correlation,
    validate_parameter_schema_completeness
)
from relic.validation.framework.models import (
    ValidationContext, ValidationLevel, ValidationCode
)


class TestParameterBoundsValidation:
    """Test parameter bounds validation rule."""
    
    def test_valid_parameters_within_bounds(self):
        """Test parameters within valid bounds."""
        parameters = {
            'temperature': {
                'value': 25.0,
                'bounds': [0.0, 50.0],
                'units': 'celsius',
                'dtype': 'float32',
                'category': 'model_parameter'
            },
            'humidity': {
                'value': torch.tensor([0.3, 0.5, 0.8]),
                'bounds': [0.0, 1.0],
                'units': 'fraction',
                # A tensor declaration states its dtype (torch.tensor default float32).
                'dtype': 'float32',
                'category': 'model_parameter'
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_parameter_bounds(context)
        
        # Should have summary result
        assert len(results) >= 1
        success_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(success_results) > 0
        assert "within bounds" in success_results[0].message
    
    def test_parameters_violate_bounds(self):
        """Test parameters that violate bounds."""
        parameters = {
            'temperature': {
                'value': -10.0,  # Below minimum
                'bounds': [0.0, 50.0],
                'units': 'celsius',
                'dtype': 'float32',
                'category': 'model_parameter'
            },
            'pressure': {
                'value': 150.0,  # Above maximum
                'bounds': [0.0, 100.0],
                'units': 'kPa',
                'dtype': 'float32',
                'category': 'model_parameter'
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_parameter_bounds(context)
        
        # Should have violation results
        violation_results = [r for r in results if r.code == ValidationCode.PARAM_BOUND_VIOLATION and r.level == ValidationLevel.ERROR]
        assert len(violation_results) >= 2  # One for each violating parameter
        
        # Check specific violations
        temp_violations = [r for r in violation_results if 'temperature' in r.message]
        pressure_violations = [r for r in violation_results if 'pressure' in r.message]
        
        assert len(temp_violations) > 0
        assert len(pressure_violations) > 0
        assert "below minimum" in temp_violations[0].message
        assert "above maximum" in pressure_violations[0].message
    
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(float("nan"), id="nan_float"),
            pytest.param(float("inf"), id="pos_inf_float"),
            pytest.param(-float("inf"), id="neg_inf_float"),
            pytest.param(torch.tensor([0.3, float("nan")]), id="nan_tensor"),
            pytest.param(torch.tensor([0.3, float("inf")]), id="inf_tensor"),
        ],
    )
    def test_non_finite_value_is_a_bound_violation(self, value):
        """A NaN or infinite value is a bound violation, never within bounds.

        NaN fails every order comparison, so a check made of comparisons alone
        passes it; the rule must report a PARAM_BOUND_VIOLATION error naming
        the parameter and must not report PARAM_BOUNDS_OK.
        """
        parameters = {
            'pressure': {
                'value': value,
                'bounds': [0.0, 100.0],
                'units': 'kPa',
                'dtype': 'float32',
                'category': 'model_parameter'
            }
        }

        results = validate_parameter_bounds(ValidationContext(metadata={'parameters': parameters}))

        assert not [r for r in results if r.code == ValidationCode.PARAM_BOUNDS_OK]
        violation_results = [
            r for r in results
            if r.code == ValidationCode.PARAM_BOUND_VIOLATION and r.level == ValidationLevel.ERROR
        ]
        assert [r for r in violation_results if r.message.startswith('pressure:')]
    
    def test_tensor_parameters_mixed_violations(self):
        """Test tensor parameters with mixed valid/invalid values."""
        parameters = {
            'multi_value': {
                'value': torch.tensor([-1.0, 0.5, 2.0]),  # First too low, last too high
                'bounds': [0.0, 1.0],
                'units': 'fraction'
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_parameter_bounds(context)
        
        # Should detect violations
        violation_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(violation_results) >= 1
    
    def test_missing_state_manager(self):
        """Test behavior when state manager is missing."""
        context = ValidationContext(
            state_manager=None,
            metadata={'parameters': {}}
        )
        
        results = validate_parameter_bounds(context)
        
        # Should handle empty parameters gracefully  
        info_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(info_results) > 0
        assert "No parameters found" in info_results[0].message
    
    def test_no_parameters_to_validate(self):
        """Test behavior when no parameters are provided.""" 
        context = ValidationContext(
            metadata={'parameters': {}}
        )
        
        results = validate_parameter_bounds(context)
        
        # Should handle empty parameters gracefully
        assert len(results) >= 1


class TestDistributionParameterConsistency:
    """Test distribution parameter consistency validation."""
    
    def test_valid_normal_distribution(self):
        """Test valid normal distribution parameters."""
        distributions = {
            'temp_dist': {
                'distribution_type': 'normal',
                'parameters': {
                    'loc': 25.0,
                    'scale': 5.0  # Positive scale
                }
            }
        }
        
        context = ValidationContext(
            metadata={'distributions': distributions}
        )
        
        results = validate_distribution_parameter_consistency(context)
        
        # Should pass validation
        success_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(success_results) > 0
        assert "consistent parameters" in success_results[0].message
    
    def test_invalid_normal_distribution_negative_scale(self):
        """Test invalid normal distribution with negative scale."""
        distributions = {
            'invalid_normal': {
                'distribution_type': 'normal',
                'parameters': {
                    'loc': 0.0,
                    'scale': -1.0  # Invalid negative scale
                }
            }
        }
        
        context = ValidationContext(
            metadata={'distributions': distributions}
        )
        
        results = validate_distribution_parameter_consistency(context)
        
        # Should detect scale violation
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) >= 1
        assert "non-positive scale" in error_results[0].message
    
    def test_uniform_distribution_bounds(self):
        """Test uniform distribution parameter validation."""
        # Valid uniform distribution
        valid_distributions = {
            'uniform_valid': {
                'distribution_type': 'uniform', 
                'parameters': {
                    'low': 0.0,
                    'high': 10.0
                }
            }
        }
        
        context = ValidationContext(
            metadata={'distributions': valid_distributions}
        )
        
        results = validate_distribution_parameter_consistency(context)
        success_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(success_results) > 0
        
        # Invalid uniform distribution (low >= high)
        invalid_distributions = {
            'uniform_invalid': {
                'distribution_type': 'uniform',
                'parameters': {
                    'low': 10.0,  # Greater than high
                    'high': 5.0
                }
            }
        }
        
        context = ValidationContext(
            metadata={'distributions': invalid_distributions}
        )
        
        results = validate_distribution_parameter_consistency(context)
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) >= 1
        assert "low >= high" in error_results[0].message
    
    def test_gamma_distribution_parameters(self):
        """Test gamma distribution parameter validation."""
        distributions = {
            'gamma_valid': {
                'distribution_type': 'gamma',
                'parameters': {
                    'alpha': 2.0,  # Positive
                    'beta': 0.5    # Positive
                }
            },
            'gamma_invalid_alpha': {
                'distribution_type': 'gamma',
                'parameters': {
                    'alpha': -1.0,  # Invalid negative
                    'beta': 1.0
                }
            },
            'gamma_invalid_beta': {
                'distribution_type': 'gamma', 
                'parameters': {
                    'alpha': 2.0,
                    'beta': 0.0     # Invalid zero
                }
            }
        }
        
        context = ValidationContext(
            metadata={'distributions': distributions}
        )
        
        results = validate_distribution_parameter_consistency(context)
        
        # Should detect alpha and beta violations
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) >= 2
        
        alpha_errors = [r for r in error_results if 'alpha' in r.message]
        beta_errors = [r for r in error_results if 'beta' in r.message]
        
        assert len(alpha_errors) >= 1
        assert len(beta_errors) >= 1
    
    def test_no_distributions_to_validate(self):
        """Test behavior when no distributions are provided."""
        context = ValidationContext(
            metadata={'distributions': {}}
        )
        
        results = validate_distribution_parameter_consistency(context)
        
        # Should handle gracefully
        info_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(info_results) >= 1
        assert "No distribution specifications" in info_results[0].message


class TestLogScaleParameterValidation:
    """Test log-scale parameter validation."""
    
    def test_valid_log_scale_parameters(self):
        """Test valid log-scale parameters."""
        parameters = {
            'log_param': {
                'log_scale': True,
                'bounds': [0.1, 10.0],  # Positive bounds
                'default_value': 1.0,   # Positive default
                'value': 2.5           # Positive current value
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_log_scale_parameters(context)
        
        # Should pass validation
        success_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(success_results) > 0
        assert "consistent" in success_results[0].message
    
    def test_invalid_log_scale_bounds(self):
        """Test log-scale parameters with invalid bounds."""
        parameters = {
            'invalid_bounds': {
                'log_scale': True,
                'bounds': [-1.0, 10.0],  # Negative minimum bound
                'default_value': 1.0,
                'value': 2.0
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_log_scale_parameters(context)

        # Should detect bound violation
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) >= 1
        assert "minimum bound" in error_results[0].message and "<= 0" in error_results[0].message
    
    def test_invalid_log_scale_values(self):
        """Test log-scale parameters with invalid values."""
        parameters = {
            'negative_default': {
                'log_scale': True,
                'bounds': [0.1, 10.0],
                # default_value is deprecated and should be ignored by validation
                'default_value': -1.0,
                'value': 2.0
            },
            'negative_current': {
                'log_scale': True,
                'bounds': [0.1, 10.0],
                'default_value': 1.0,
                'value': -0.5  # Invalid negative current value
            },
            'tensor_with_negatives': {
                'log_scale': True,
                'bounds': [0.1, 10.0],
                'default_value': 1.0,
                'value': torch.tensor([1.0, -2.0, 3.0])  # Contains negative
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_log_scale_parameters(context)
        
        # Should detect violations
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) >= 2
    
    def test_non_log_scale_parameters_ignored(self):
        """Test that non-log-scale parameters are ignored."""
        parameters = {
            'normal_param': {
                'log_scale': False,
                'bounds': [-5.0, 5.0],  # Negative bounds OK for non-log-scale
                'value': -2.0
            },
            'no_log_scale_field': {
                # Missing log_scale field defaults to False
                'bounds': [-10.0, 10.0],
                'value': -5.0
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_log_scale_parameters(context)
        
        # Should report no log-scale parameters found
        info_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(info_results) >= 1
        assert "No log-scale parameters" in info_results[0].message


class TestSpatialCorrelationValidation:
    """Test spatial correlation validation."""
    
    def test_valid_spatial_correlation(self):
        """Test valid spatial correlation parameters."""
        spatial_params = {
            'spatial_temp': {
                'correlation_length': 100.0,  # Positive correlation length
                'spatial_dimensions': ['x', 'y']
            },
            'anisotropic_param': {
                'correlation_length': [50.0, 200.0],  # Different for each dimension
                'spatial_dimensions': ['x', 'y']
            }
        }
        
        context = ValidationContext(
            metadata={'spatial_parameters': spatial_params}
        )
        
        results = validate_spatial_correlation(context)
        
        # Should pass validation
        success_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(success_results) > 0
        assert "valid correlation lengths" in success_results[0].message
    
    def test_invalid_correlation_lengths(self):
        """Test invalid correlation length parameters."""
        spatial_params = {
            'negative_corr': {
                'correlation_length': -10.0,  # Invalid negative
                'spatial_dimensions': ['x', 'y']
            },
            'zero_corr': {
                'correlation_length': 0.0,    # Invalid zero
                'spatial_dimensions': ['x']
            },
            'mixed_invalid': {
                'correlation_length': [50.0, -20.0, 100.0],  # One invalid
                'spatial_dimensions': ['x', 'y', 'z']
            }
        }
        
        context = ValidationContext(
            metadata={'spatial_parameters': spatial_params}
        )
        
        results = validate_spatial_correlation(context)
        
        # Should detect violations
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) >= 3
        
        # Check specific violation messages
        negative_errors = [r for r in error_results if 'negative_corr' in r.message]
        zero_errors = [r for r in error_results if 'zero_corr' in r.message]
        mixed_errors = [r for r in error_results if 'mixed_invalid' in r.message]
        
        assert len(negative_errors) >= 1
        assert len(zero_errors) >= 1
        assert len(mixed_errors) >= 1
    
    def test_missing_correlation_length(self):
        """Test parameters missing correlation length."""
        spatial_params = {
            'missing_corr': {
                'spatial_dimensions': ['x', 'y']
                # Missing correlation_length
            }
        }
        
        context = ValidationContext(
            metadata={'spatial_parameters': spatial_params}
        )
        
        results = validate_spatial_correlation(context)
        
        # Should generate warning
        warning_results = [r for r in results if r.level == ValidationLevel.WARNING]
        assert len(warning_results) >= 1
        assert "lacks correlation length" in warning_results[0].message
    
    def test_extreme_correlation_lengths(self):
        """Test very large or very small correlation lengths."""
        spatial_params = {
            'very_large': {
                'correlation_length': 1e7,  # Very large
                'spatial_dimensions': ['x', 'y']
            },
            'very_small': {
                'correlation_length': 1e-7,  # Very small
                'spatial_dimensions': ['x']
            }
        }
        
        context = ValidationContext(
            metadata={'spatial_parameters': spatial_params}
        )
        
        results = validate_spatial_correlation(context)
        
        # Should generate warnings for extreme values
        warning_results = [r for r in results if r.level == ValidationLevel.WARNING]
        assert len(warning_results) >= 2
        
        large_warnings = [r for r in warning_results if 'very large' in r.message]
        small_warnings = [r for r in warning_results if 'very small' in r.message]
        
        assert len(large_warnings) >= 1
        assert len(small_warnings) >= 1


class TestParameterSchemaCompleteness:
    """Test parameter schema completeness validation."""
    
    def test_complete_parameter_schema(self):
        """Test parameters with complete schema."""
        parameters = {
            'complete_param': {
                'bounds': [0.0, 10.0],
                'units': 'kg/m3',
                'description': 'A complete parameter description'
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_parameter_schema_completeness(context)
        
        # Should pass validation
        success_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(success_results) > 0
        assert "complete schema" in success_results[0].message
    
    def test_incomplete_parameter_schema(self):
        """Test parameters with incomplete schema."""
        parameters = {
            'missing_bounds': {
                # Missing bounds
                'units': 'kg/m3',
                'description': 'Missing bounds parameter'
            },
            'missing_multiple': {
                'bounds': [0.0, 1.0],
                # Missing units and description
            },
            'completely_empty': {
                # Missing all required fields
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_parameter_schema_completeness(context)
        
        # Should detect missing fields
        warning_results = [r for r in results if r.level == ValidationLevel.WARNING and r.code == ValidationCode.PARAM_SCHEMA_MISSING]
        parameter_warnings = [r for r in warning_results if 'missing required fields' in r.message]
        
        assert len(parameter_warnings) >= 3  # One for each incomplete parameter
        
        # Check specific missing fields
        bounds_warnings = [r for r in parameter_warnings if 'missing_bounds' in r.message]
        assert len(bounds_warnings) >= 1
        assert 'bounds' in bounds_warnings[0].message
        
        multiple_warnings = [r for r in parameter_warnings if 'missing_multiple' in r.message]
        assert len(multiple_warnings) >= 1
    
    def test_completion_rate_calculation(self):
        """Test completion rate calculation in summary."""
        parameters = {
            'complete1': {
                'bounds': [0.0, 1.0],
                'units': 'fraction',
                'description': 'Complete parameter 1'
            },
            'complete2': {
                'bounds': [0.0, 100.0],
                'units': 'percent',
                'description': 'Complete parameter 2'
            },
            'incomplete': {
                'bounds': [0.0, 10.0],
                # Missing units and description
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        results = validate_parameter_schema_completeness(context)
        
        # Should have summary with completion rate
        summary_results = [r for r in results if 'completion rate' in r.message]
        
        if summary_results:
            assert len(summary_results) >= 1
            # Should show 2/3 completion (66.7%)
            assert '66.7%' in summary_results[0].message or '67%' in summary_results[0].message
    
    def test_no_parameters_in_context(self):
        """Test behavior when no parameters are in context."""
        context = ValidationContext(
            metadata={}  # No parameters field
        )
        
        results = validate_parameter_schema_completeness(context)
        
        # Should handle gracefully
        warning_results = [r for r in results if r.level == ValidationLevel.WARNING]
        assert len(warning_results) >= 1
        assert "No parameter schema found" in warning_results[0].message


class TestParameterValidationIntegration:
    """Integration tests for parameter validation rules."""
    
    def test_multiple_rules_execution(self):
        """Test running multiple validation rules together."""
        parameters = {
            'test_param': {
                'value': -1.0,           # Violates bounds
                'bounds': [0.0, 10.0],
                'log_scale': True,       # Violates log-scale (negative value)
                'units': 'kg/m3',
                'description': 'Test parameter',
                'default_value': 5.0
            }
        }
        
        distributions = {
            'test_dist': {
                'distribution_type': 'normal',
                'parameters': {
                    'loc': 0.0,
                    'scale': -2.0  # Invalid negative scale
                }
            }
        }
        
        context = ValidationContext(
            metadata={
                'parameters': parameters,
                'distributions': distributions
            }
        )
        
        # Run all validation rules
        bounds_results = validate_parameter_bounds(context)
        dist_results = validate_distribution_parameter_consistency(context)
        log_scale_results = validate_log_scale_parameters(context)
        schema_results = validate_parameter_schema_completeness(context)
        
        # Should detect multiple violations
        all_results = bounds_results + dist_results + log_scale_results + schema_results
        error_results = [r for r in all_results if r.level == ValidationLevel.ERROR]
        
        # Should have at least 3 errors (bounds, distribution, log-scale)
        assert len(error_results) >= 3
    
    def test_error_handling_robustness(self):
        """Test error handling in validation rules."""
        # Test with malformed context
        context = ValidationContext(
            metadata=None  # This might cause issues
        )
        
        # Should handle gracefully without crashing
        try:
            bounds_results = validate_parameter_bounds(context)
            dist_results = validate_distribution_parameter_consistency(context) 
            log_scale_results = validate_log_scale_parameters(context)
            schema_results = validate_parameter_schema_completeness(context)
            
            # Should return some results, likely warnings about missing data
            all_results = bounds_results + dist_results + log_scale_results + schema_results
            assert len(all_results) > 0
            
        except Exception as e:
            pytest.fail(f"Validation rules should handle malformed input gracefully: {e}")
    
    def test_validation_codes_consistency(self):
        """Test that validation codes are used consistently."""
        # Create scenarios that trigger specific validation codes
        parameters = {
            'bounds_violation': {
                'value': 100.0,
                'bounds': [0.0, 50.0],
                'units': 'test'
            },
            'incomplete_schema': {
                'value': 1.0
                # Missing required fields
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        bounds_results = validate_parameter_bounds(context)
        schema_results = validate_parameter_schema_completeness(context)
        
        # Check that correct validation codes are used
        bounds_errors = [r for r in bounds_results if r.level == ValidationLevel.ERROR]
        if bounds_errors:
            assert any(r.code == ValidationCode.PARAM_BOUND_VIOLATION for r in bounds_errors)
        
        schema_warnings = [r for r in schema_results if r.level == ValidationLevel.WARNING]
        if schema_warnings:
            assert any(r.code == ValidationCode.PARAM_SCHEMA_MISSING for r in schema_warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])




def test_invalid_parameter_declaration_is_a_failed_result_not_a_bounds_comparison():
    """A declaration that cannot form a ParameterSpec is reported as invalid."""
    context = ValidationContext(
        metadata={"parameters": {"inverted": {"value": 5.0, "bounds": [10.0, 0.0], "units": "K",
                                                "dtype": "float32", "category": "model_parameter"}}}
    )

    results = validate_parameter_bounds(context)

    invalid = [r for r in results if "invalid_declaration" in r.tags]
    assert len(invalid) == 1
    assert invalid[0].passed is False
    assert "Invalid bounds" in invalid[0].message
    assert not any(r.code == ValidationCode.PARAM_BOUND_VIOLATION and "inverted" in r.message for r in results)
    assert not any(r.code == ValidationCode.PARAM_BOUNDS_OK for r in results)


def test_tensor_declaration_without_dtype_is_an_invalid_declaration():
    """ParameterSpec requires dtype for a tensor value; the rule reports it, never guesses one."""
    declaration = {"value": torch.tensor([0.3, 0.5, 0.8]), "bounds": [0.0, 1.0], "units": "fraction",
                   "category": "model_parameter"}
    context = ValidationContext(metadata={"parameters": {"humidity": declaration}})

    results = validate_parameter_bounds(context)

    invalid = [r for r in results if "invalid_declaration" in r.tags]
    assert len(invalid) == 1
    assert invalid[0].passed is False
    assert "humidity" in invalid[0].message
    assert "dtype" in invalid[0].message
    assert not any(r.code == ValidationCode.PARAM_BOUNDS_OK for r in results)

    declaration["dtype"] = "float32"
    accepted = validate_parameter_bounds(ValidationContext(metadata={"parameters": {"humidity": declaration}}))
    assert not any("invalid_declaration" in r.tags for r in accepted)


def _correlated_distribution(**overrides):
    spec = {
        "distribution_type": "normal",
        "parameters": {"loc": 0.0, "scale": 1.0},
        "correlation_length": 250.0,
        "kernel": "exponential",
        "correlation_nugget": 0.0,
    }
    spec.update(overrides)
    return {key: value for key, value in spec.items() if value is not _OMIT}


_OMIT = object()


def _consistency_errors(spec):
    context = ValidationContext(metadata={"distributions": {"field": spec}})
    results = validate_distribution_parameter_consistency(context)
    return [r for r in results if r.level == ValidationLevel.ERROR]


def test_correlated_distribution_with_kernel_and_nugget_is_consistent():
    """correlation_length with a CorrelationKernel name and a declared nugget is a valid field."""
    from relic.stochastic.distributions import CorrelationKernel, DistributionSpec
    from relic.validation.framework.rules.parameters import _build_distribution_model

    model = _build_distribution_model(_correlated_distribution(correlation_nugget=0.05))

    assert isinstance(model, DistributionSpec)
    assert model.kernel is CorrelationKernel.EXPONENTIAL
    assert model.correlation_nugget == 0.05
    assert _consistency_errors(_correlated_distribution()) == []


def test_correlated_distribution_with_an_unknown_kernel_is_rejected():
    errors = _consistency_errors(_correlated_distribution(kernel="gaussianish"))

    assert len(errors) == 1
    assert "gaussianish" in errors[0].message
    assert "squared_exponential" in errors[0].message


def test_correlated_distribution_without_a_nugget_is_rejected():
    errors = _consistency_errors(_correlated_distribution(correlation_nugget=_OMIT))

    assert len(errors) == 1
    assert "correlation_nugget" in errors[0].message


def test_uncorrelated_distribution_must_not_declare_kernel_or_nugget():
    errors = _consistency_errors(
        _correlated_distribution(correlation_length=_OMIT, kernel=_OMIT, correlation_nugget=0.1)
    )

    assert len(errors) == 1
    assert "correlation_nugget" in errors[0].message
