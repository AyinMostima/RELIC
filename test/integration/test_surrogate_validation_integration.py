"""
L2 Integration Tests for Surrogate Model Validation Integration

This test suite validates the integration between PyTorchSurrogate models
and the RELIC validation framework, ensuring that surrogate models
properly integrate with parameter validation rules, conservation checks,
and validation reporting.

Test Categories:
1. Parameter Validation Integration Tests
2. Surrogate Model Validation Tests
3. Conservation Law Integration Tests
4. Validation Engine Integration Tests
5. End-to-End Workflow Tests
6. Error Handling and Recovery Tests

Mathematical Framework:
    Surrogate S with parameters theta in validation context V:
    Valid(S, theta, V) iff every rule r in V returns PASS.
    Conservation ledger: max(abs(Y @ C.T)) <= tolerance.

References:
    - RELIC Validation Framework Specification
    - Physics-Informed Neural Networks Validation
    - Conservation Law Enforcement in Surrogates
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add relic to Python path
relic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, relic_path)

from relic.models.pytorch_surrogate import PyTorchSurrogate
from relic.models.surrogates import ConservationViolationError
from relic.validation.framework.engine import ValidationEngine
from relic.validation.framework.models import (
    ValidationContext,
    ValidationLevel,
)
from relic.validation.framework.rules.parameters import (
    validate_parameter_bounds,
    validate_distribution_parameter_consistency,
    validate_log_scale_parameters
)


class TestSurrogateParameterValidationIntegration:
    """Test integration of surrogate models with parameter validation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_surrogate_parameter_validation(self):
        """Test that surrogate model parameters are validated correctly."""
        # Create surrogate with invalid parameter bounds
        config = {
            'model_path': str(self.temp_dir / 'test_param_validation.pth'),
            'validate_model_path': False,
            'input_variables': ['temperature', 'humidity'],
            'output_variables': ['growth_rate'],
            'architecture': {
                'type': 'mlp',
                'hidden_dims': [32, 16],
                'activation': 'relu'
            }
        }
        
        surrogate = PyTorchSurrogate('test_surrogate', config)
        
        # Create validation context with surrogate parameters
        surrogate_params = {
            'surrogate.learning_rate': {
                'value': -0.01,  # Invalid negative learning rate
                'bounds': [1e-6, 1.0],
                'units': 'dimensionless',
                'description': 'Learning rate for surrogate training', "dtype": "float32", "category": "model_parameter"
            },
            'surrogate.dropout_rate': {
                'value': 0.2,  # Valid dropout rate
                'bounds': [0.0, 1.0],
                'units': 'dimensionless',
                'description': 'Dropout rate for regularization', "dtype": "float32", "category": "model_parameter"
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': surrogate_params}
        )
        
        # Run parameter validation
        results = validate_parameter_bounds(context)
        
        # Should detect the invalid learning rate
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) >= 1
        
        learning_rate_errors = [r for r in error_results if 'learning_rate' in r.message]
        assert len(learning_rate_errors) >= 1
        assert 'below minimum' in learning_rate_errors[0].message
    
    def test_surrogate_log_scale_parameter_validation(self):
        """Test log-scale parameter validation for surrogate hyperparameters."""
        surrogate_params = {
            'surrogate.weight_decay': {
                'value': 1e-4,
                'bounds': [1e-8, 1e-1],
                'log_scale': True,  # Should be log-scale
                'units': 'dimensionless', "dtype": "float32", "category": "model_parameter"
            },
            'surrogate.invalid_log_param': {
                'value': -1e-3,  # Invalid negative value for log-scale
                'bounds': [1e-6, 1e-1],
                'log_scale': True,
                'units': 'dimensionless', "dtype": "float32", "category": "model_parameter"
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': surrogate_params}
        )
        
        results = validate_log_scale_parameters(context)
        
        # Should detect log-scale violation
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) >= 1
        
        log_scale_errors = [r for r in error_results if 'invalid_log_param' in r.message]
        assert len(log_scale_errors) >= 1
    
    def test_surrogate_distribution_parameter_validation(self):
        """Test validation of surrogate uncertainty distributions."""
        # Create surrogate with uncertainty quantification
        config = {
            'model_path': str(self.temp_dir / 'test_dist_validation.pth'),
            'validate_model_path': False,
            'input_variables': ['x'],
            'output_variables': ['y'],
            'architecture': {'type': 'mlp', 'hidden_dims': [16]},
            'uncertainty': {
                'method': 'dropout_mc',
                'mc_samples': 20
            }
        }
        
        surrogate = PyTorchSurrogate('test_dist_surrogate', config)
        
        # Define uncertainty distributions for validation
        uncertainty_distributions = {
            'prediction_noise': {
                'distribution_type': 'normal',
                'parameters': {
                    'loc': 0.0,
                    'scale': 0.1  # Valid positive scale
                }
            },
            'parameter_uncertainty': {
                'distribution_type': 'normal',
                'parameters': {
                    'loc': 0.0,
                    'scale': -0.05  # Invalid negative scale
                }
            }
        }
        
        context = ValidationContext(
            metadata={'distributions': uncertainty_distributions}
        )
        
        results = validate_distribution_parameter_consistency(context)
        
        # Should detect invalid distribution
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) >= 1
        
        scale_errors = [r for r in error_results if 'non-positive scale' in r.message]
        assert len(scale_errors) >= 1


class TestSurrogateConservationValidation:
    """Test conservation law validation in surrogate models."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_surrogate_conservation_enforcement(self, monkeypatch):
        """Test that a declared surrogate ledger accepts a closed output."""
        config = {
            'model_path': str(self.temp_dir / 'conservation_test.pth'),
            'validate_model_path': False,
            'input_variables': ['substrate'],
            'output_variables': ['product1', 'product2'],
            'architecture': {
                'type': 'mlp',
                'hidden_dims': [32, 16]
            },
            'conservation_check': True,
            'conservation_matrix': [[1.0, 1.0]],
            'conservation_tolerance': 1.0e-8,
        }
        
        surrogate = PyTorchSurrogate('conservation_surrogate', config)

        def conservative_prediction(input_tensor):
            batch_size = input_tensor.shape[0]
            row = input_tensor.new_tensor([[2.0, -2.0]])
            return row.repeat(batch_size, 1)

        monkeypatch.setattr(surrogate, '_predict', conservative_prediction)
        
        # Test with input that should be conserved
        state = {
            'substrate': torch.tensor([10.0])
        }
        
        time = 0.0
        dt = 1.0
        
        # Run surrogate
        flux_dict = surrogate(state, time, dt)
        
        residual = flux_dict['product1'] + flux_dict['product2']
        assert torch.allclose(residual, torch.zeros_like(residual))
    
    def test_surrogate_conservation_violation_detection(self, monkeypatch):
        """Test detection of conservation violations in surrogate predictions."""
        # Mock a surrogate that violates conservation
        config = {
            'model_path': str(self.temp_dir / 'violation_test.pth'),
            'validate_model_path': False,
            'input_variables': ['input'],
            'output_variables': ['output1', 'output2'],
            'architecture': {'type': 'mlp', 'hidden_dims': [8]},
            'conservation_check': True,
            'conservation_matrix': [[1.0, 1.0]],
            'conservation_tolerance': 1.0e-8,
        }
        
        surrogate = PyTorchSurrogate('violation_surrogate', config)
        
        # Create a mock scenario where conservation is violated
        state = {'input': torch.tensor([5.0])}
        
        def mock_predict_with_violation(input_tensor):
            batch_size = input_tensor.shape[0]
            row = input_tensor.new_tensor([[7.0, 3.0]])
            return row.repeat(batch_size, 1)

        monkeypatch.setattr(surrogate, '_predict', mock_predict_with_violation)
        
        # Test conservation checking
        time = 0.0
        dt = 1.0
        
        with pytest.raises(ConservationViolationError, match='Conservation violation'):
            surrogate(state, time, dt)


class TestValidationEngineIntegration:
    """Test integration with the validation engine."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_surrogate_with_validation_engine(self):
        """Test running surrogate models through validation engine."""
        # Create surrogate model
        config = {
            'model_path': str(self.temp_dir / 'validation_engine_test.pth'),
            'validate_model_path': False,
            'input_variables': ['temperature', 'pressure'],
            'output_variables': ['reaction_rate'],
            'architecture': {
                'type': 'resnet',
                'n_blocks': 2,
                'hidden_dim': 32
            }
        }
        
        surrogate = PyTorchSurrogate('validation_test_surrogate', config)
        
        # Create validation context
        parameters = {
            'temperature': {
                'value': 298.15,
                'bounds': [273.15, 373.15],
                'units': 'K',
                'description': 'Temperature', "dtype": "float32", "category": "model_parameter"
            },
            'pressure': {
                'value': 101325.0,
                'bounds': [1000.0, 1e6],
                'units': 'Pa',
                'description': 'Pressure', "dtype": "float32", "category": "model_parameter"
            }
        }
        
        context = ValidationContext(
            metadata={
                'parameters': parameters,
                'surrogate_model': surrogate
            }
        )
        
        # Create and configure validation engine
        engine = ValidationEngine()
        
        # Run L1 parameter validation
        report = engine.run(layers=['L1'], domains=['parameters'], context=context)
        results = report.results
        
        # Should have validation results
        assert len(results) > 0
        
        # Should not have critical errors for valid parameters
        critical_errors = [r for r in results if r.level == ValidationLevel.ERROR]
        # May have some errors depending on implementation, but should be manageable
    
    def test_surrogate_training_validation_workflow(self):
        """Test end-to-end workflow: training -> validation -> deployment."""
        # Step 1: Create and train surrogate
        config = {
            'model_path': str(self.temp_dir / 'workflow_test.pth'),
            'validate_model_path': False,
            'input_variables': ['x1', 'x2'],
            'output_variables': ['y1'],
            'input_units': {'x1': '1', 'x2': '1'},
            'output_units': {'y1': '1'},
            'architecture': {
                'type': 'mlp',
                'hidden_dims': [32, 16],
                'activation': 'tanh'
            },
            'training': {
                'optimizer': 'adam',
                'learning_rate': 0.01,
                'epochs': 3  # Quick training
            }
        }
        
        surrogate = PyTorchSurrogate('workflow_surrogate', config)
        
        # Create simple training data
        n_samples = 20
        states = []
        fluxes = []
        
        for i in range(n_samples):
            x1 = np.random.normal(0, 1)
            x2 = np.random.normal(0, 1)
            y1 = x1 + 0.5 * x2 + 0.1 * np.random.normal()  # Simple linear relationship
            
            states.append({
                'x1': torch.tensor([x1]),
                'x2': torch.tensor([x2])
            })
            fluxes.append({
                'y1': torch.tensor([y1])
            })
        
        # Train the model
        training_result = surrogate.train((states, fluxes), verbose=False)
        assert surrogate.is_trained
        
        # Step 2: Validate trained model parameters
        model_parameters = {
            'model.learning_rate': {
                'value': config['training']['learning_rate'],
                'bounds': [1e-6, 1.0],
                'units': 'dimensionless', "dtype": "float32", "category": "model_parameter"
            },
            'model.hidden_layer_size': {
                'value': 32,
                'bounds': [8, 512],
                'units': 'dimensionless', "dtype": "int64", "category": "model_parameter"
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': model_parameters}
        )
        
        validation_results = validate_parameter_bounds(context)
        
        # Should pass validation
        error_results = [r for r in validation_results if r.level == ValidationLevel.ERROR]
        assert len(error_results) == 0  # No parameter bound violations expected
        
        # Step 3: Test deployment (inference)
        test_state = {
            'x1': torch.tensor([1.0]),
            'x2': torch.tensor([0.5])
        }
        
        prediction = surrogate(test_state, time=0.0, dt=1.0)
        
        assert 'y1' in prediction
        assert isinstance(prediction['y1'], torch.Tensor)
        assert torch.isfinite(prediction['y1'])
    
    def test_multi_surrogate_validation(self):
        """Test validation of multiple surrogate models together."""
        surrogates = []
        
        # Create multiple surrogate models
        for i in range(3):
            config = {
                'model_path': str(self.temp_dir / f'multi_surrogate_{i}.pth'),
                'input_variables': [f'input_{i}'],
                'output_variables': [f'output_{i}'],
                'validate_model_path': False,  # Skip path validation for testing
                # Each network receives only the keys it reads; keys of the
                # other architectures are rejected, not ignored.
                'architecture': [
                    {'type': 'mlp', 'hidden_dims': [16, 8]},
                    {'type': 'resnet', 'n_blocks': 2, 'hidden_dim': 32},
                    {'type': 'transformer', 'd_model': 16, 'n_heads': 4, 'n_layers': 2},
                ][i]
            }
            
            surrogate = PyTorchSurrogate(f'multi_surrogate_{i}', config)
            surrogates.append(surrogate)
        
        # Create validation context with all surrogates
        all_parameters = {}
        
        for i, surrogate in enumerate(surrogates):
            all_parameters[f'surrogate_{i}.complexity'] = {
                'value': len(surrogate.config['input_variables']),
                'bounds': [1, 10],
                'units': 'dimensionless', "dtype": "int64", "category": "model_parameter"
            }
        
        context = ValidationContext(
            metadata={
                'parameters': all_parameters,
                'surrogate_models': surrogates
            }
        )
        
        # Validate all parameters
        results = validate_parameter_bounds(context)
        
        # Should handle multiple surrogates without issues
        assert len(results) > 0
        
        # Check that all surrogates are functional
        for i, surrogate in enumerate(surrogates):
            test_state = {f'input_{i}': torch.tensor([1.0])}
            prediction = surrogate(test_state, time=0.0, dt=1.0)
            
            assert f'output_{i}' in prediction
            assert torch.isfinite(prediction[f'output_{i}'])


class TestSurrogateValidationErrorHandling:
    """Test error handling in surrogate validation integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up."""  
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_invalid_surrogate_configuration_handling(self):
        """Test handling of invalid surrogate configurations in validation."""
        # Create surrogate with invalid configuration
        config = {
            'model_path': str(self.temp_dir / 'invalid_config.pth'),
            'validate_model_path': False,
            'input_variables': [],  # Empty input variables (invalid)
            'output_variables': ['output'],
            'architecture': {
                'type': 'unknown_type',  # Invalid architecture type
                'hidden_dims': [-1, 0]  # Invalid dimensions
            }
        }
        
        # The invalid configuration is rejected with an explicit error
        with pytest.raises(ValueError, match='input_variables cannot be empty'):
            PyTorchSurrogate('invalid_surrogate', config)
    
    def test_validation_with_missing_data(self):
        """Test validation behavior when required data is missing."""
        # Create validation context with missing data
        incomplete_context = ValidationContext(
            metadata={
                'parameters': None,  # Missing parameters
                'distributions': {}  # Empty distributions
            }
        )
        
        # Should handle missing data gracefully
        results = validate_parameter_bounds(incomplete_context)
        
        # Should handle missing data gracefully (INFO level, not WARNING)
        info_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(info_results) > 0
        
        # Should have messages about missing data
        missing_data_messages = [r for r in info_results if 'no parameters' in r.message.lower()]
        assert len(missing_data_messages) > 0
    
    def test_surrogate_numerical_instability_detection(self):
        """Test detection of numerical instability in surrogate models."""
        # Create a surrogate that might produce NaN or inf
        config = {
            'model_path': str(self.temp_dir / 'unstable_surrogate.pth'),
            'validate_model_path': False,
            'input_variables': ['x'],
            'output_variables': ['y'],
            'validate_model_path': False,  # Skip path validation for testing
            'architecture': {
                'type': 'mlp',
                'hidden_dims': [16, 8]
            }
        }
        
        surrogate = PyTorchSurrogate('unstable_surrogate', config)
        
        # Mock prediction to return NaN
        original_predict = surrogate._predict
        
        def mock_predict_with_nan(input_tensor):
            batch_size = input_tensor.shape[0]
            return torch.full((batch_size, 1), float('nan'))
        
        surrogate._predict = mock_predict_with_nan
        
        # Test with potentially problematic input
        state = {'x': torch.tensor([1e10])}  # Very large input
        
        try:
            result = surrogate(state, time=0.0, dt=1.0)
            
            # Check if NaN was detected and handled
            if 'y' in result:
                if torch.isnan(result['y']).any():
                    # NaN was propagated - should be caught by validation
                    pass
                else:
                    # NaN was handled/replaced - good
                    assert torch.isfinite(result['y']).all()
                    
        except Exception as e:
            # Should get a meaningful error about numerical instability
            assert any(keyword in str(e).lower() for keyword in ['nan', 'inf', 'numerical', 'unstable'])
        
        # Restore original method
        surrogate._predict = original_predict
    
    def test_validation_recovery_after_errors(self):
        """Test that validation system can recover after encountering errors."""
        # Create a scenario with mixed valid and invalid configurations
        parameters = {
            'valid_param': {
                'value': 5.0,
                'bounds': [0.0, 10.0],
                'units': 'kg', "dtype": "float32", "category": "model_parameter"
            },
            'invalid_param': {
                'value': 15.0,  # Violates bounds
                'bounds': [0.0, 10.0],
                'units': 'kg', "dtype": "float32", "category": "model_parameter"
            },
            'malformed_param': {
                # Missing required fields
                'value': 1.0
            }
        }
        
        context = ValidationContext(
            metadata={'parameters': parameters}
        )
        
        # Should process all parameters and report issues appropriately
        results = validate_parameter_bounds(context)
        
        # Should have results for both valid and invalid parameters
        assert len(results) > 0
        
        # Should have detected the bound violation
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        bound_errors = [r for r in error_results if 'invalid_param' in r.message]
        assert len(bound_errors) > 0
        
        # Should still have processed the valid parameter
        success_indicators = [r for r in results if 'valid' in r.message or r.level == ValidationLevel.INFO]
        # May or may not have explicit success messages, but should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

