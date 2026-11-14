"""
Integration Test for BVOC Parameter System
==========================================
Validates that BVOC responses module correctly uses ParameterManager
without any hardcoded defaults.

Test Coverage:
- Parameter loading from ParameterManager
- No hardcoded defaults in responses.py
- Correct namespace usage
- Required parameter enforcement
"""

import pytest
import torch
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch

from relic.scientific.biosphere.bvoc.responses import (
    BVOCLightResponse,
    BVOCTemperatureResponse,
    BVOCEnvironmentalResponse,
    BVOCResponseIntegrator
)
from relic.parameters.ops.key_normalization import normalize_parameter_key


class TestBVOCParameterIntegration:
    """Test BVOC modules use ParameterManager correctly."""

    @staticmethod
    def assert_required_get(mock_param_manager, key: str, units: str) -> None:
        """Assert the current required ParameterManager get() contract (no validate flag:
        validation is the manager's own, the flag was removed from get())."""
        mock_param_manager.get.assert_any_call(
            key,
            units=units,
            required=True,
            expected_type=None,
        )
    
    @pytest.fixture
    def mock_param_manager(self):
        """Create mock parameter manager with required values."""
        mock = Mock()
        
        # Define all required parameters that should be loaded
        param_values = {
            # Light response parameters
            "biosphere.bvoc.megan.alpha_light": 0.0027,
            "biosphere.bvoc.megan.cl_light": 1.066,
            
            # Temperature response - Isoprene
            "biosphere.bvoc.megan.eopt_isoprene": 1.75,
            "biosphere.bvoc.megan.topt": 313.0,
            "biosphere.bvoc.megan.ct1_isoprene": 95000.0,
            "biosphere.bvoc.megan.ct2": 230000.0,
            
            # Temperature response - Other compounds
            "biosphere.bvoc.megan.beta_monoterpenes": 0.1,
            "biosphere.bvoc.megan.beta_sesquiterpenes": 0.17,
            "biosphere.bvoc.t_standard": 303.15,
            "biosphere.bvoc.megan.ldf_sesquiterpenes": 0.5,
            # Guenther et al. (2012) Table 4 Eq. (8) coefficients of the partly
            # light-dependent classes (Other Monoterpenes, Other Sesquiterpenes).
            "biosphere.bvoc.megan.ldf_monoterpenes": 0.4,
            "biosphere.bvoc.megan.ct1_monoterpenes": 80000.0,
            "biosphere.bvoc.megan.eopt_monoterpenes": 1.83,
            "biosphere.bvoc.megan.ct1_sesquiterpenes": 130000.0,
            "biosphere.bvoc.megan.eopt_sesquiterpenes": 2.37,
            
            # Water stress
            "biosphere.stress.water.wilting_point_potential": -1.5,
            "biosphere.stress.water.field_capacity_potential": -0.1,
        }

        param_units = {
            "biosphere.bvoc.megan.alpha_light": "m2 s umol-1",
            "biosphere.bvoc.megan.cl_light": "dimensionless",
            "biosphere.bvoc.megan.eopt_isoprene": "dimensionless",
            "biosphere.bvoc.megan.topt": "K",
            "biosphere.bvoc.megan.ct1_isoprene": "J mol-1",
            "biosphere.bvoc.megan.ct2": "J mol-1",
            "biosphere.bvoc.megan.beta_monoterpenes": "K-1",
            "biosphere.bvoc.megan.beta_sesquiterpenes": "K-1",
            "biosphere.bvoc.t_standard": "K",
            "biosphere.bvoc.megan.ldf_sesquiterpenes": "dimensionless",
            "biosphere.bvoc.megan.ldf_monoterpenes": "dimensionless",
            "biosphere.bvoc.megan.ct1_monoterpenes": "J mol-1",
            "biosphere.bvoc.megan.eopt_monoterpenes": "dimensionless",
            "biosphere.bvoc.megan.ct1_sesquiterpenes": "J mol-1",
            "biosphere.bvoc.megan.eopt_sesquiterpenes": "dimensionless",
            "biosphere.stress.water.wilting_point_potential": "MPa",
            "biosphere.stress.water.field_capacity_potential": "MPa",
        }
        
        def get_side_effect(key, **kwargs):
            if kwargs.get('required', False) and key not in param_values:
                raise KeyError(f"Required parameter '{key}' not found")
            return param_values.get(key, kwargs.get('default'))
        
        mock.get.side_effect = get_side_effect
        def get_parameter_side_effect(key):
            canonical = normalize_parameter_key(key)
            units = param_units.get(canonical)
            if units is None:
                return None
            return SimpleNamespace(units=units)

        mock.get_parameter.side_effect = get_parameter_side_effect
        return mock
    
    def test_light_response_no_hardcoded_defaults(self, mock_param_manager):
        """Test that light response uses ParameterManager without hardcoded defaults."""
        with patch('relic.scientific.biosphere.bvoc.responses.get_global_manager', 
                   return_value=mock_param_manager):
            
            ppfd = torch.tensor(1000.0)
            
            # This should work with parameters from ParameterManager
            response = BVOCLightResponse.get_compound_light_response(
                'sesquiterpenes', ppfd
            )
            
            # Verify parameters were requested with required=True
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.megan.alpha_light", "m2 s umol-1"
            )
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.megan.cl_light", "dimensionless"
            )
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.megan.ldf_sesquiterpenes", "dimensionless"
            )
            
            # Response should be valid
            assert response is not None
            assert torch.isfinite(response).all()
            assert response >= 0.0
    
    def test_temperature_response_no_hardcoded_defaults(self, mock_param_manager):
        """Test that temperature response uses ParameterManager without hardcoded defaults."""
        with patch('relic.scientific.biosphere.bvoc.responses.get_global_manager',
                   return_value=mock_param_manager):
            
            temperature = torch.tensor(298.15)  # 25掳C
            
            # Test isoprene temperature response
            response = BVOCTemperatureResponse.get_compound_temperature_response(
                'isoprene', temperature
            )
            
            # Verify required parameters were requested
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.megan.eopt_isoprene", "dimensionless"
            )
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.megan.topt", "K"
            )
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.megan.ct1_isoprene", "J mol-1"
            )
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.megan.ct2", "J mol-1"
            )
            
            # Response should be valid
            assert response is not None
            assert torch.isfinite(response).all()
    
    def test_missing_required_parameter_raises_error(self):
        """Test that missing required parameters raise appropriate errors."""
        # Create mock that simulates missing parameters
        mock = Mock()
        def missing_get(key, **kwargs):
            if kwargs.get("required", False):
                raise KeyError(f"Required parameter '{key}' not found")
            return None
        mock.get.side_effect = missing_get
        mock.get_parameter.side_effect = lambda key: None
        
        with patch('relic.scientific.biosphere.bvoc.responses.get_global_manager',
                   return_value=mock):
            
            ppfd = torch.tensor(1000.0)
            
            # This should raise an error due to missing required parameters
            with pytest.raises(KeyError):
                BVOCLightResponse.get_compound_light_response('isoprene', ppfd)
    
    def test_water_stress_parameters(self, mock_param_manager):
        """Test water stress uses correct parameter namespaces."""
        with patch('relic.scientific.biosphere.bvoc.responses.get_global_manager',
                   return_value=mock_param_manager):
            
            water_potential = torch.tensor(-0.5)  # MPa
            
            response = BVOCEnvironmentalResponse.drought_stress_response(
                water_potential=water_potential
            )
            
            # Verify correct namespace was used
            self.assert_required_get(
                mock_param_manager, "biosphere.stress.water.wilting_point_potential", "MPa"
            )
            self.assert_required_get(
                mock_param_manager, "biosphere.stress.water.field_capacity_potential", "MPa"
            )
            
            # Response should be in valid range
            assert 0.0 <= response <= 1.0
    
    def test_compound_specific_responses(self, mock_param_manager):
        """Test compound-specific parameter loading."""
        with patch('relic.scientific.biosphere.bvoc.responses.get_global_manager',
                   return_value=mock_param_manager):
            
            temperature = torch.tensor(298.15)
            
            # Test monoterpenes
            response_mono = BVOCTemperatureResponse.get_compound_temperature_response(
                'monoterpenes', temperature
            )
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.megan.beta_monoterpenes", "K-1"
            )
            
            # Test sesquiterpenes
            response_sesq = BVOCTemperatureResponse.get_compound_temperature_response(
                'sesquiterpenes', temperature
            )
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.megan.beta_sesquiterpenes", "K-1"
            )
            self.assert_required_get(
                mock_param_manager, "biosphere.bvoc.t_standard", "K"
            )
            
            # Both should return valid responses
            assert torch.isfinite(response_mono).all()
            assert torch.isfinite(response_sesq).all()
    
    def test_response_integrator(self, mock_param_manager):
        """Test that response integrator correctly combines all factors."""
        with patch('relic.scientific.biosphere.bvoc.responses.get_global_manager',
                   return_value=mock_param_manager):
            
            integrator = BVOCResponseIntegrator()
            
            state = {'water_stress_factor': torch.tensor(0.8)}

            factors = integrator.calculate_all_activity_factors(
                compound='isoprene',
                temperature=torch.tensor(298.15),
                ppfd=torch.tensor(1000.0),
                state=state,
            )
            
            # Should have all expected factors
            assert 'gamma_light' in factors
            assert 'gamma_temperature' in factors
            assert 'gamma_stress' in factors
            assert 'gamma_co2' not in factors
            assert 'gamma_age' not in factors
            assert 'gamma_total' in factors
            assert factors['gamma_stress'].item() == pytest.approx(0.8)
            
            # Total should be product of individual factors
            expected_total = (
                factors['gamma_light'] * 
                factors['gamma_temperature'] * 
                factors['gamma_stress']
            )
            assert torch.allclose(factors['gamma_total'], expected_total, rtol=1e-5)


class TestParameterConsistency:
    """Test parameter consistency across modules."""
    
    def test_no_default_values_in_get_calls(self):
        """Verify no default values are passed to parameter manager get() calls."""
        import re
        
        # Read the responses.py file to check for default parameters
        with open('relic/scientific/biosphere/bvoc/responses.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to find param_manager.get() calls with default parameter
        # This pattern should NOT match after our fixes
        pattern = r'param_manager\.get\([^)]+,\s*default='
        
        matches = re.findall(pattern, content)
        
        # After our fixes, there should be NO matches
        assert len(matches) == 0, (
            f"Found {len(matches)} param_manager.get() calls with default values. "
            "All parameters should use required=True instead."
        )
    
    def test_no_conditional_parameter_manager(self):
        """Verify parameter manager is always required, not optional."""
        with open('relic/scientific/biosphere/bvoc/responses.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should not have conditional defaults like "if param_manager else default_value"
        pattern = r'if param_manager else \d+'
        matches = re.findall(pattern, content)
        
        assert len(matches) == 0, (
            "Found conditional defaults for parameter manager. "
            "ParameterManager should always be required."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


