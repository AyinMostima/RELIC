"""
Test suite for verifying photosynthesis parameter system fixes.

This test ensures that all parameters are properly loaded from the parameter manager
and that no hardcoded values bypass the parameter system.
"""

import pytest
import torch
import sys
import os

# Add relic to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.parameters import get_global_manager
from relic.core.state import UpdateItem


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


class TestPhotosynthesisParameterFix:
    """Test class for photosynthesis parameter system integration."""
    
    @pytest.fixture
    def setup_parameter_manager(self):
        """Setup parameter manager with test configuration."""
        param_manager = get_global_manager()
        # Load numerical validation parameters
        param_manager.load_from_database('relic/validation_numerical.yaml')
        # Optional solver guardrails (kept separate from validation.numerical)
        param_manager.load_from_database('relic/validation/numerical_stability.yaml')
        # Load photosynthesis parameters
        param_manager.load_from_database('relic/biosphere/photosynthesis.yaml')
        return param_manager
    
    @pytest.fixture
    def photosynthesis_model(self, setup_parameter_manager):
        """Create a photosynthesis model instance for testing."""
        model = FarquharPhotosynthesis(
            name="test_photosynthesis",
            device=torch.device('cpu'),
            enable_gpu=False,
            validate_inputs=True
        )
        return model
    
    def test_numerical_stability_parameters_loaded(self, setup_parameter_manager):
        """Test that numerical stability parameters are properly loaded."""
        param_manager = setup_parameter_manager
        
        # Test clamping parameters
        assert param_manager.get("validation.numerical.clamp_min_negative") == -0.1
        assert param_manager.get("validation.numerical.clamp_max_zero") == 0.0
        assert param_manager.get("validation.numerical.clamp_min_zero") == 0.0
        assert param_manager.get("validation.numerical.clamp_min_small") == 0.1
        assert param_manager.get("validation.numerical.clamp_max_one") == 1.0
        
        # Test temperature clamping
        assert param_manager.get("validation.numerical.clamp_min_temperature") == 233.15
        assert param_manager.get("validation.numerical.clamp_max_temperature") == 323.15
        
        # Test CO2 clamping
        assert param_manager.get("validation.numerical.clamp_min_co2") == 10.0
        assert param_manager.get("validation.numerical.clamp_max_co2") == 5000.0
        
        # Test division safety
        assert param_manager.get("validation.numerical.epsilon_divide") == 1e-10
        
        # Test VPD response
        assert param_manager.get("validation.numerical.vpd_response_d0") == 1500.0
        
        # Test C4 factor
        assert param_manager.get("validation.numerical.c4_electron_factor") == 3.0
    
    def test_parameter_access_with_namespace(self, photosynthesis_model):
        """Test that parameters are accessed with proper namespace."""
        # Test that model can retrieve parameters with namespace
        clamp_min_zero = photosynthesis_model.get_parameter_value(
            'clamp_min_zero',
            namespace="validation.numerical"
        )
        assert clamp_min_zero == 0.0
        
        epsilon = photosynthesis_model.get_parameter_value(
            'epsilon_divide',
            namespace="validation.numerical"
        )
        assert float(epsilon) == pytest.approx(1e-10)
        
        vpd_d0 = photosynthesis_model.get_parameter_value(
            'vpd_response_d0',
            namespace="validation.numerical"
        )
        assert vpd_d0 == 1500.0
    
    def test_photosynthesis_computation(self, photosynthesis_model):
        """Test that photosynthesis computation works with fixed parameters."""
        # Create test state
        state = {
            'air_temperature': torch.tensor(298.15, dtype=torch.float32),
            'atmospheric_co2_ppm': torch.tensor(400.0, dtype=torch.float32),
            'ppfd': torch.tensor(1000.0, dtype=torch.float32),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0, dtype=torch.float32),
            'vcmax_scaling': torch.tensor(1.0, dtype=torch.float32),
        }
        
        # Run forward pass (use forward() to get complete output dictionary)
        output = photosynthesis_model.forward(state, time=0.0, dt=1.0)
        
        # Check output structure
        assert set(output) == set(photosynthesis_model.metadata.output_variables)
        assert all(isinstance(item, UpdateItem) for item in output.values())
        assert 'carboxylation_limited' not in output
        assert 'vcmax_actual' not in output
        
        # Check output values are reasonable
        gpp = output['gross_photosynthesis'].value
        assert torch.isfinite(gpp).all()
        assert gpp >= 0  # GPP should be non-negative
        
        an = output['net_photosynthesis'].value
        assert torch.isfinite(an).all()
    
    def test_nan_required_inputs_fail_fast(self, photosynthesis_model):
        """Test that NaN required inputs are never replaced by defaults."""
        # Create state with NaN values
        state = {
            'air_temperature': torch.tensor(float('nan'), dtype=torch.float32),
            'atmospheric_co2_ppm': torch.tensor(400.0, dtype=torch.float32),
            'ppfd': torch.tensor(1000.0, dtype=torch.float32),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0, dtype=torch.float32),
        }

        with pytest.raises(
            ValueError, match="air_temperature must contain only finite values"
        ):
            photosynthesis_model.forward(state, time=0.0, dt=1.0)
    
    def test_extreme_values_fail_fast(self, photosynthesis_model):
        """Test that out-of-domain forcings are not silently clamped."""
        # Test extreme temperature
        state_hot = {
            'air_temperature': torch.tensor(400.0, dtype=torch.float32),
            'atmospheric_co2_ppm': torch.tensor(400.0, dtype=torch.float32),
            'ppfd': torch.tensor(1000.0, dtype=torch.float32),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0, dtype=torch.float32),
        }

        with pytest.raises(ValueError, match="air_temperature outside supported range"):
            photosynthesis_model.forward(state_hot, time=0.0, dt=1.0)
        
        # Test extreme CO2
        state_high_co2 = {
            'air_temperature': torch.tensor(298.15, dtype=torch.float32),
            'atmospheric_co2_ppm': torch.tensor(10000.0, dtype=torch.float32),
            'ppfd': torch.tensor(1000.0, dtype=torch.float32),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0, dtype=torch.float32),
        }

        with pytest.raises(ValueError, match="atmospheric_co2_ppm outside supported range"):
            photosynthesis_model.forward(state_high_co2, time=0.0, dt=1.0)
    
    def test_c4_pathway(self, setup_parameter_manager):
        """Test C4 photosynthesis pathway with parameters."""
        model_c4 = FarquharPhotosynthesis(
            name="test_c4_photosynthesis",
            pathway="C4",
            device=torch.device('cpu'),
            enable_gpu=False
        )
        
        # Create test state
        state = {
            'air_temperature': torch.tensor(303.15, dtype=torch.float32),
            'atmospheric_co2_ppm': torch.tensor(400.0, dtype=torch.float32),
            'ppfd': torch.tensor(1500.0, dtype=torch.float32),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0, dtype=torch.float32),
            'vcmax_scaling': torch.tensor(1.0, dtype=torch.float32),
        }
        
        # Run forward pass (use forward() to get complete output)
        output = model_c4.forward(state, time=0.0, dt=1.0)
        
        assert set(output) == set(model_c4.metadata.output_variables)
        assert 'pep_carboxylation_limited' not in output
        assert 'export_limited' not in output
        assert output['diagnostics.photosynthesis.pathway_type'].value == 1.0
        
        # Check outputs are reasonable
        assert torch.isfinite(output['gross_photosynthesis'].value).all()
        assert output['gross_photosynthesis'].value >= 0
    
    def test_parameter_registration(self, setup_parameter_manager):
        """Test that all required parameters are registered."""
        param_manager = setup_parameter_manager
        
        # Get missing requests (should be minimal after our fixes)
        missing = param_manager.get_missing_requests()
        
        # Check that critical parameters are not in missing
        critical_params = [
            'validation.numerical.clamp_min_zero',
            'validation.numerical.epsilon_divide',
            'validation.numerical.vpd_response_d0',
            'biosphere.photosynthesis.vcmax25',
            'biosphere.photosynthesis.jmax25',
            'biosphere.photosynthesis.rd25'
        ]
        
        for param in critical_params:
            assert param not in missing, f"Critical parameter {param} is missing"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
