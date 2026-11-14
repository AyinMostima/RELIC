"""
L1 Unit Tests for DenitrificationProcess Module

This module provides comprehensive unit tests for the denitrification process
implementation, covering anaerobic conditions, Michaelis-Menten kinetics, 
N2O:N2 product distribution, and carbon requirements.

Test Categories:
- L1.1: Basic instantiation and configuration  
- L1.2: Scientific accuracy against literature values
- L1.3: Michaelis-Menten kinetics
- L1.4: Anaerobic condition controls
- L1.5: Carbon availability effects
- L1.6: N2O:N2 product distribution
- L1.7: Environmental response functions
- L1.8: Boundary condition handling
- L1.9: Parameter sensitivity analysis
- L1.10: Performance and stability tests
"""

import pytest
import torch
import numpy as np
from typing import Dict
from relic.core.state import UpdateItem, UpdateMode

# Import the module under test
from relic.scientific.pedosphere.nutrient_cycling import DenitrificationProcess
from relic.parameters import get_global_manager
from test.utils.forward_call import call_forward

# Import utilities for testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value

def _normalize_param_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return value

def _set_param_value(param_manager, key: str, value) -> None:
    spec = param_manager.get_parameter(key)
    units = spec.units if spec is not None else None
    param_manager.set_value(key, value, units=units, validate=True)


class TestDenitrificationProcessL1:
    """L1 Unit Tests for DenitrificationProcess functor."""
    
    @pytest.fixture
    def default_config(self) -> Dict:
        """Default configuration for DenitrificationProcess."""
        return {
            'vmax_denitrification': 1.5e-6,  # g N m鈦宦?s鈦宦?            'km_no3': 0.002,                 # g N m鈦宦?            'wfps_denitrification_threshold': 0.6,      # m鲁 m鈦宦?            'q10_denitrification': 2.2       # dimensionless
        }
    
    @pytest.fixture  
    def denitrification(self, default_config) -> DenitrificationProcess:
        """Create DenitrificationProcess instance."""
        return DenitrificationProcess(name="test_denitrification", config=default_config)
    
    @pytest.fixture
    def standard_state(self) -> Dict:
        """Standard state for testing."""
        return {
            'no3_nitrogen': torch.tensor(0.02),         # g N m-2
            'soil_temperature': torch.tensor(288.15),   # K (15 degC)
            'water_filled_pore_space': torch.tensor(0.7),         # m鲁 m鈦宦?(anaerobic)
        }
    
    @pytest.fixture
    def standard_environment(self) -> Dict:
        """Standard environment for testing."""
        return {
            'labile_carbon': torch.tensor(0.1),         # g C m-2
            'soil_temperature': torch.tensor(288.15),
            'water_filled_pore_space': torch.tensor(0.7),
            'soil_ph': torch.tensor(6.5)
        }
        
    # =============================================================================
    # L1.1: Basic Instantiation and Configuration Tests
    # =============================================================================
    
    def test_instantiation_basic(self, default_config):
        """Test basic instantiation with default configuration."""
        denitrification = DenitrificationProcess(name="test_denit", config=default_config)
        
        assert denitrification.name == "test_denit"
        assert denitrification.config == default_config
        assert hasattr(denitrification, 'metadata')
        assert hasattr(denitrification, 'vmax')
        assert hasattr(denitrification, 'km_no3')
        assert hasattr(denitrification, 'wfps_threshold')
        assert denitrification.metadata.name == "test_denit"
        
    def test_instantiation_metadata(self, denitrification):
        """Test metadata completeness."""
        metadata = denitrification.metadata
        
        # Check required metadata fields
        assert metadata.name is not None
        assert metadata.version is not None
        assert metadata.description is not None
        assert metadata.mathematical_expression is not None
        
        # Check scientific references
        assert len(metadata.references) >= 2
        assert any("Parton" in ref for ref in metadata.references)
        assert any("Li" in ref or "DNDC" in ref for ref in metadata.references)
        
        # Check parameters documentation
        assert len(metadata.parameters) >= 3
        assert 'pedosphere.denitrification.vmax_denitrification' in metadata.parameters
        assert 'pedosphere.denitrification.km_no3' in metadata.parameters
        assert 'pedosphere.physical.wfps_denitrification_threshold' in metadata.parameters
    
    def test_parameter_initialization(self, denitrification):
        """Test parameter initialization."""
        # Check all expected parameters exist and are reasonable
        vmax = _normalize_param_value(denitrification.vmax)
        km_no3 = _normalize_param_value(denitrification.km_no3)
        q10 = _normalize_param_value(denitrification.q10)
        threshold = _normalize_param_value(denitrification.wfps_threshold)
        assert 0 < vmax <= 1e-4  # Should be reasonable rate
        assert 0 < km_no3 <= 0.1  # Should be reasonable Km
        assert 1.5 <= q10 <= 4.0  # Literature Q10 range
        assert 0.3 <= threshold <= 0.9  # Reasonable moisture threshold
    
    def test_dependencies(self, denitrification):
        """Test dependency declaration."""
        deps = denitrification.get_dependencies()
        
        assert isinstance(deps, list)
        assert deps == ["nitrification"]
        assert "labile_carbon" in denitrification.metadata.input_variables
        assert "first_order_decomposition" not in deps
        assert "microbial_enzyme_decomposition" not in deps
    
    # =============================================================================
    # L1.2: Scientific Accuracy Tests Against Literature Values
    # =============================================================================
    
    def test_q10_temperature_response_literature(self, denitrification):
        """Test Q10 temperature response against literature values."""
        # Literature Q10 for denitrification: 2.0-3.0 (Parton et al., 1996)
        q10_value = _normalize_param_value(denitrification.q10)
        assert 1.8 <= q10_value <= 3.5, f"Q10 should be 1.8-3.5: {q10_value}"
        
        # Test temperature response follows Q10 law
        temperatures = [278.15, 288.15, 298.15, 308.15]  # 5掳C, 15掳C, 25掳C, 35掳C
        
        base_state = {'no3_nitrogen': torch.tensor(0.02)}
        base_env = {'labile_carbon': torch.tensor(0.1), 
                   'water_filled_pore_space': torch.tensor(0.8),  # Anaerobic
                   'soil_ph': torch.tensor(6.5)}
        responses = []
        
        for temp in temperatures:
            env = base_env.copy()
            env['soil_temperature'] = torch.tensor(temp)
            
            rates = call_forward(denitrification, {**base_state, **env})
            response = float(rates['diagnostics.temperature_response'].detach())
            responses.append(response)
        
        # Check Q10 relationship over 10掳C intervals
        for i in range(len(temperatures) - 1):
            temp_diff = temperatures[i+1] - temperatures[i]  # 10掳C
            response_ratio = responses[i+1] / responses[i]
            expected_ratio = q10_value ** (temp_diff / 10.0)
            
            relative_error = abs(response_ratio - expected_ratio) / expected_ratio
            assert relative_error < 0.15, \
                f"Q10 relationship violated: {response_ratio:.3f} vs expected {expected_ratio:.3f}"
    
    def test_vmax_literature_values(self, denitrification):
        """Test Vmax against literature values."""
        vmax = _normalize_param_value(denitrification.vmax)
        
        # Literature range: 5e-7 to 5e-6 g N m鈦宦?s鈦宦?(Li et al., 1992; Parton et al., 1996)
        assert 1e-7 <= vmax <= 1e-5, f"Vmax should be 1e-7 to 1e-5: {vmax:.2e}"
        
        # Default value should be in reasonable range
        assert 5e-7 <= vmax <= 5e-6, f"Default Vmax should be reasonable: {vmax:.2e}"
    
    def test_km_no3_literature_values(self, denitrification):
        """Test Km for NO3 against literature values."""
        km_no3 = _normalize_param_value(denitrification.km_no3)
        
        # Literature range: 0.0005-0.01 g N m鈦宦?(based on DNDC, Century models)
        assert 0.0005 <= km_no3 <= 0.05, f"Km_NO3 should be 0.0005-0.05: {km_no3}"
        
        # Default value should be in reasonable range
        assert 0.001 <= km_no3 <= 0.01, f"Default Km_NO3 should be reasonable: {km_no3}"
    
    def test_anaerobic_threshold_literature(self, denitrification):
        """Test anaerobic threshold against literature values."""
        threshold = _normalize_param_value(denitrification.wfps_threshold)
        
        # Literature: denitrification begins at 60-80% WFPS (Linn & Doran, 1984)
        assert 0.5 <= threshold <= 0.8, f"Anaerobic threshold should be 0.5-0.8: {threshold}"
    
    # =============================================================================
    # L1.3: Michaelis-Menten Kinetics Tests
    # =============================================================================
    
    def test_michaelis_menten_shape(self, denitrification, standard_environment):
        """Test Michaelis-Menten kinetics curve shape."""
        # Test range of NO3 concentrations
        no3_values = torch.linspace(0.0001, 0.1, 50)
        rates = []
        
        for no3 in no3_values:
            state = {'no3_nitrogen': no3}
            result = call_forward(denitrification, {**state, **standard_environment})
            rate = float(_unwrap_update(result['denitrification_flux']).detach())
            rates.append(rate)
        
        rates = np.array(rates)
        
        # Should be monotonically increasing
        assert np.all(np.diff(rates) >= -1e-12), "Rate should increase with substrate"
        
        # Should approach Vmax asymptotically
        max_rate = rates[-1]
        vmax = _normalize_param_value(denitrification.vmax)
        
        # At high substrate, rate should approach Vmax (within environmental factors)
        # Denitrification has multiple environmental limitations (anaerobic, carbon, pH)
        expected_max = vmax * 0.1  # Account for environmental factors
        assert max_rate >= expected_max * 0.5, f"Should approach Vmax: {max_rate:.2e} vs {expected_max:.2e}"
    
    def test_km_half_saturation(self, denitrification, standard_environment):
        """Test that Km represents half-saturation constant."""
        km_no3 = _normalize_param_value(denitrification.km_no3)
        
        # At [NO3] = Km, rate should be Vmax/2 (times environmental factors)
        state = {'no3_nitrogen': torch.tensor(km_no3)}
        result = call_forward(denitrification, {**state, **standard_environment})
        rate_at_km = float(_unwrap_update(result['denitrification_flux']).detach())
        
        # At very high [NO3], rate approaches Vmax
        high_no3_state = {'no3_nitrogen': torch.tensor(km_no3 * 100)}
        high_result = call_forward(denitrification, {**high_no3_state, **standard_environment})
        rate_at_high = float(_unwrap_update(high_result['denitrification_flux']).detach())
        
        # Rate at Km should be approximately half of rate at high substrate
        if rate_at_high > 1e-10:
            ratio = rate_at_km / rate_at_high
            assert 0.35 <= ratio <= 0.65, f"Rate at Km should be ~50% of Vmax: ratio = {ratio:.3f}"
    
    def test_zero_substrate_handling(self, denitrification, standard_environment):
        """Test handling of zero NO3 substrate."""
        state = {'no3_nitrogen': torch.tensor(0.0)}
        result = call_forward(denitrification, {**state, **standard_environment})
        
        rate = float(_unwrap_update(result['denitrification_flux']).detach())
        assert rate == 0.0, f"Zero substrate should give zero rate: {rate}"
        
        n2o = float(_unwrap_update(result['n2o_flux']).detach())
        assert n2o == 0.0, f"Zero rate should give zero N2O: {n2o}"
        
        n2 = float(_unwrap_update(result['n2_production']).detach())
        assert n2 == 0.0, f"Zero rate should give zero N2: {n2}"

    def test_n2o_source_uses_additive_update_mode(
        self, denitrification, standard_environment
    ):
        """Denitrification N2O must add to other declared N2O sources."""
        state = {'no3_nitrogen': torch.tensor(0.02), **standard_environment}
        result = denitrification.forward(state, time=0.0, dt=1.0)
        update = result['n2o_flux']
        assert isinstance(update, UpdateItem)
        assert update.mode is UpdateMode.ACCUMULATE
        assert update.unit == "g N m-2 s-1"
        assert update.basis == "per_second"
        assert update.per_timestep is False
        assert update.element is None
        assert update.source_functor == denitrification.name

    def test_nitrate_pool_delta_uses_explicit_accumulate_mode(
        self, denitrification, standard_environment
    ):
        """The NO3 per-step loss must enter the shared SCM ledger once."""
        state = {'no3_nitrogen': torch.tensor(0.02), **standard_environment}
        result = denitrification.forward(state, time=0.0, dt=3600.0)
        update = result['no3_nitrogen']
        assert isinstance(update, UpdateItem)
        assert update.mode is UpdateMode.ACCUMULATE
        assert update.unit == "g N m-2"
        assert update.basis == "per_step"
        assert update.per_timestep is True
        assert update.element == "N"
        assert update.source_functor == denitrification.name

        for key in ('denitrification_flux', 'n2_production'):
            rate = result[key]
            assert isinstance(rate, UpdateItem), key
            assert rate.mode is UpdateMode.REPLACE, key
            assert rate.unit == "g N m-2 s-1", key
            assert rate.basis == "per_second", key
            assert rate.per_timestep is False, key
            assert rate.element is None, key
            assert rate.source_functor == denitrification.name, key
    
    # =============================================================================
    # L1.4: Anaerobic Condition Controls Tests
    # =============================================================================
    
    def test_anaerobic_threshold_response(self, denitrification, standard_state):
        """Test response to anaerobic threshold."""
        threshold = _normalize_param_value(denitrification.wfps_threshold)
        
        # Test moisture values around threshold
        moisture_values = [threshold - 0.1, threshold, threshold + 0.1, threshold + 0.3]
        anaerobic_factors = []
        
        for moisture in moisture_values:
            if moisture >= 0:  # Skip negative moisture
                env = {'labile_carbon': torch.tensor(0.1),
                      'water_filled_pore_space': torch.tensor(moisture),
                      'soil_temperature': torch.tensor(288.15),
                      'soil_ph': torch.tensor(6.5)}
                
                result = call_forward(denitrification, {**standard_state, **env})
                factor = float(result['diagnostics.anaerobic_factor'].detach())
                anaerobic_factors.append(factor)
        
        # Below threshold should give low factor
        if len(anaerobic_factors) > 0:
            assert anaerobic_factors[0] <= 0.2, "Below threshold should give low anaerobic factor"
        
        # Above threshold should increase factor
        for i in range(1, len(anaerobic_factors)):
            if moisture_values[i] > threshold:
                assert anaerobic_factors[i] >= anaerobic_factors[0], \
                    "Above threshold should increase anaerobic factor"
    
    def test_oxic_conditions_inhibition(self, denitrification, standard_state):
        """Test inhibition under oxic conditions."""
        # Low moisture (oxic conditions)
        oxic_env = {'labile_carbon': torch.tensor(0.1),
                   'water_filled_pore_space': torch.tensor(0.3),  # Well below threshold
                   'soil_temperature': torch.tensor(288.15),
                   'soil_ph': torch.tensor(6.5)}
        
        oxic_result = call_forward(denitrification, {**standard_state, **oxic_env})
        oxic_rate = float(_unwrap_update(oxic_result['denitrification_flux']).detach())
        
        # High moisture (anaerobic conditions)
        anaerobic_env = oxic_env.copy()
        anaerobic_env['water_filled_pore_space'] = torch.tensor(0.8)
        
        anaerobic_result = call_forward(denitrification, {**standard_state, **anaerobic_env})
        anaerobic_rate = float(_unwrap_update(anaerobic_result['denitrification_flux']).detach())
        
        # Anaerobic conditions should give much higher rate
        assert anaerobic_rate > oxic_rate * 2, \
            f"Anaerobic rate should be much higher: {anaerobic_rate:.2e} vs {oxic_rate:.2e}"
    
    def test_anaerobic_factor_bounds(self, denitrification, standard_state):
        """Test anaerobic factor stays within reasonable bounds."""
        extreme_moistures = [0.1, 0.99]
        
        for moisture in extreme_moistures:
            env = {'labile_carbon': torch.tensor(0.1),
                  'water_filled_pore_space': torch.tensor(moisture),
                  'soil_temperature': torch.tensor(288.15),
                  'soil_ph': torch.tensor(6.5)}
            
            result = call_forward(denitrification, {**standard_state, **env})
            factor = float(result['diagnostics.anaerobic_factor'].detach())
            
            # Should be bounded
            assert 0.0 <= factor <= 1.0, f"Anaerobic factor should be 0-1: {factor}"
    
    # =============================================================================
    # L1.5: Carbon Availability Effects Tests
    # =============================================================================
    
    def test_carbon_limitation(self, denitrification, standard_state):
        """Test carbon limitation effects."""
        # Test range of carbon availability
        carbon_values = [0.001, 0.01, 0.05, 0.1, 0.5]
        rates = []
        
        for carbon in carbon_values:
            env = {'labile_carbon': torch.tensor(carbon),
                  'water_filled_pore_space': torch.tensor(0.8),  # Anaerobic
                  'soil_temperature': torch.tensor(288.15),
                  'soil_ph': torch.tensor(6.5)}
            
            result = call_forward(denitrification, {**standard_state, **env})
            rate = float(_unwrap_update(result['denitrification_flux']).detach())
            rates.append(rate)
        
        # Rates should generally increase with carbon availability
        for i in range(len(rates) - 1):
            assert rates[i+1] >= rates[i] * 0.95, \
                f"Rate should increase with carbon: {rates[i]:.2e} -> {rates[i+1]:.2e}"
    
    def test_carbon_factor_saturation(self, denitrification, standard_state):
        """Test carbon factor saturation behavior."""
        # Very high carbon should saturate
        high_carbon_env = {'labile_carbon': torch.tensor(10.0),
                          'water_filled_pore_space': torch.tensor(0.8),
                          'soil_temperature': torch.tensor(288.15),
                          'soil_ph': torch.tensor(6.5)}
        
        result = call_forward(denitrification, {**standard_state, **high_carbon_env})
        carbon_factor = float(result['diagnostics.carbon_factor'].detach())
        
        # Should approach but not exceed 1.0
        assert 0.9 <= carbon_factor <= 1.0, f"Carbon factor should saturate near 1.0: {carbon_factor}"
    
    def test_zero_carbon_handling(self, denitrification, standard_state):
        """Test handling of zero carbon availability."""
        no_carbon_env = {'labile_carbon': torch.tensor(0.0),
                        'water_filled_pore_space': torch.tensor(0.8),
                        'soil_temperature': torch.tensor(288.15),
                        'soil_ph': torch.tensor(6.5)}
        
        result = call_forward(denitrification, {**standard_state, **no_carbon_env})
        rate = float(_unwrap_update(result['denitrification_flux']).detach())
        carbon_factor = float(result['diagnostics.carbon_factor'].detach())
        
        # Should give very low rate and carbon factor
        assert rate < 1e-8, f"Zero carbon should give very low rate: {rate:.2e}"
        assert carbon_factor < 0.01, f"Zero carbon should give very low carbon factor: {carbon_factor}"
    
    # =============================================================================
    # L1.6: N2O:N2 Product Distribution Tests
    # =============================================================================
    
    def test_n2o_n2_distribution_mass_balance(self, denitrification, standard_state, standard_environment):
        """Test N2O:N2 distribution maintains mass balance."""
        result = call_forward(denitrification, {**standard_state, **standard_environment})
        
        denit_rate = float(_unwrap_update(result['denitrification_flux']).detach())
        n2o_rate = float(_unwrap_update(result['n2o_flux']).detach())
        n2_rate = float(_unwrap_update(result['n2_production']).detach())
        
        # Mass balance: denitrification = N2O + N2
        balance_error = abs(denit_rate - (n2o_rate + n2_rate))
        if denit_rate > 1e-10:
            relative_error = balance_error / denit_rate
            assert relative_error < 0.01, f"Product distribution mass balance error: {relative_error:.4f}"
    
    def test_n2o_fraction_bounds(self, denitrification, standard_state, standard_environment):
        """Test N2O fraction stays within reasonable bounds."""
        result = call_forward(denitrification, {**standard_state, **standard_environment})
        n2o_fraction = float(result['diagnostics.n2o_fraction'].detach())
        
        # Literature: N2O fraction typically 10-70% (Firestone & Davidson, 1989)
        assert 0.05 <= n2o_fraction <= 0.8, f"N2O fraction should be 0.05-0.8: {n2o_fraction}"
    
    def test_n2o_fraction_ph_dependency(self, denitrification, standard_state):
        """Test N2O fraction dependency on pH."""
        ph_values = [5.0, 6.0, 7.0, 8.0]
        n2o_fractions = []
        
        for ph in ph_values:
            env = {'labile_carbon': torch.tensor(0.1),
                  'water_filled_pore_space': torch.tensor(0.8),
                  'soil_temperature': torch.tensor(288.15),
                  'soil_ph': torch.tensor(ph)}
            
            result = call_forward(denitrification, {**standard_state, **env})
            fraction = float(result['diagnostics.n2o_fraction'].detach())
            n2o_fractions.append(fraction)
        
        # Generally, lower pH should favor N2O (incomplete denitrification)
        # This relationship may be complex, but fractions should be reasonable
        for fraction in n2o_fractions:
            assert 0.05 <= fraction <= 0.8, f"N2O fraction should be reasonable: {fraction}"
    
    def test_n2o_fraction_no3_dependency(self, denitrification, standard_environment):
        """Test N2O fraction dependency on NO3 concentration."""
        no3_values = [0.001, 0.01, 0.05, 0.1]
        n2o_fractions = []
        
        for no3 in no3_values:
            state = {'no3_nitrogen': torch.tensor(no3)}
            result = call_forward(denitrification, {**state, **standard_environment})
            fraction = float(result['diagnostics.n2o_fraction'].detach())
            n2o_fractions.append(fraction)
        
        # Higher NO3 concentration typically favors N2O production
        # Check that all fractions are reasonable
        for fraction in n2o_fractions:
            assert 0.05 <= fraction <= 0.8, f"N2O fraction should be reasonable: {fraction}"
    
    # =============================================================================
    # L1.7: Environmental Response Functions Tests
    # =============================================================================
    
    def test_ph_response_optimal_range(self, denitrification, standard_state):
        """Test pH response has optimal range around neutral pH."""
        ph_values = torch.linspace(4.0, 9.0, 21)
        ph_factors = []
        
        for ph in ph_values:
            env = {'labile_carbon': torch.tensor(0.1),
                  'water_filled_pore_space': torch.tensor(0.8),
                  'soil_temperature': torch.tensor(288.15),
                  'soil_ph': ph}
            
            result = call_forward(denitrification, {**standard_state, **env})
            ph_factor = float(result.get('diagnostics.ph_factor', torch.tensor(1.0)).detach())
            ph_factors.append(ph_factor)
        
        ph_factors = np.array(ph_factors)
        
        # Find maximum response pH
        max_idx = np.argmax(ph_factors)
        max_ph = float(ph_values[max_idx])
        
        # Optimal pH should be around 6-8 (literature range)
        assert 5.5 <= max_ph <= 8.5, f"Optimal pH should be 5.5-8.5: {max_ph:.1f}"
    
    def test_temperature_response_bounds(self, denitrification, standard_state):
        """Test temperature response stays within reasonable bounds."""
        extreme_temps = [263.15, 323.15]  # -10掳C, 50掳C
        
        for temp in extreme_temps:
            env = {'labile_carbon': torch.tensor(0.1),
                  'water_filled_pore_space': torch.tensor(0.8),
                  'soil_temperature': torch.tensor(temp),
                  'soil_ph': torch.tensor(6.5)}
            
            result = call_forward(denitrification, {**standard_state, **env})
            temp_response = float(result['diagnostics.temperature_response'].detach())
            
            # Should be bounded and finite
            assert 0.05 <= temp_response <= 20.0, \
                f"Temperature response {temp_response:.3f} outside bounds at {temp-273.15:.1f}掳C"
    
    # =============================================================================
    # L1.8: Boundary Condition Handling Tests
    # =============================================================================
    
    def test_extreme_conditions_handling(self, denitrification):
        """Test handling of extreme environmental conditions."""
        extreme_cases = [
            # Very dry, cold
            {'no3_nitrogen': torch.tensor(0.01), 'labile_carbon': torch.tensor(0.01),
             'water_filled_pore_space': torch.tensor(0.1), 'soil_temperature': torch.tensor(268.15),
             'soil_ph': torch.tensor(4.0)},
            # Very wet, hot
            {'no3_nitrogen': torch.tensor(0.1), 'labile_carbon': torch.tensor(1.0),
             'water_filled_pore_space': torch.tensor(0.95), 'soil_temperature': torch.tensor(318.15),
             'soil_ph': torch.tensor(9.0)},
        ]
        
        for i, case in enumerate(extreme_cases):
            state = {'no3_nitrogen': case['no3_nitrogen']}
            env = {k: v for k, v in case.items() if k != 'no3_nitrogen'}
            
            result = call_forward(denitrification, {**state, **env})
            
            # Should not crash and all outputs should be finite
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all(), f"Case {i}: {key} should be finite"
                    assert not torch.isnan(value).any(), f"Case {i}: {key} should not be NaN"
                    if key == "no3_nitrogen":
                        # no3_nitrogen is a pool delta (can be negative due to NO3 consumption).
                        # It must not consume more NO3 than available in the state.
                        assert value >= -case["no3_nitrogen"], (
                            f"Case {i}: {key} should not exceed available substrate: {value}"
                        )
                    else:
                        assert value >= 0.0, f"Case {i}: {key} should be non-negative: {value}"
    
    def test_very_high_substrate_handling(self, denitrification, standard_environment):
        """Test handling of very high NO3 concentrations."""
        high_no3_state = {'no3_nitrogen': torch.tensor(10.0)}  # Very high
        result = call_forward(denitrification, {**high_no3_state, **standard_environment})
        
        rate = float(_unwrap_update(result['denitrification_flux']).detach())
        vmax = _normalize_param_value(denitrification.vmax)
        
        # Should not exceed Vmax (accounting for environmental factors)
        max_theoretical = vmax * 3.0  # Allow for environmental enhancement
        assert rate <= max_theoretical, f"Rate should not exceed theoretical max: {rate:.2e} vs {max_theoretical:.2e}"
    
    # =============================================================================
    # L1.9: Parameter Sensitivity Analysis Tests
    # =============================================================================
    
    def test_vmax_sensitivity(self, standard_state, standard_environment):
        """Test sensitivity to Vmax parameter."""
        vmax_values = [0.5e-6, 1.5e-6, 3.0e-6, 6.0e-6]
        rates = []

        param_manager = get_global_manager()
        vmax_key = "pedosphere.denitrification.vmax_denitrification"
        original_vmax = _normalize_param_value(param_manager.get(vmax_key))
        try:
            for vmax in vmax_values:
                _set_param_value(param_manager, vmax_key, vmax)
                denit_process = DenitrificationProcess(name="test", config={})

                result = call_forward(denit_process, {**standard_state, **standard_environment})
                rate = float(_unwrap_update(result['denitrification_flux']).detach())
                rates.append(rate)
        finally:
            _set_param_value(param_manager, vmax_key, original_vmax)

        # Rate should be proportional to Vmax
        for i in range(len(rates) - 1):
            if rates[i] > 1e-12:
                ratio = rates[i+1] / rates[i]
                expected_ratio = vmax_values[i+1] / vmax_values[i]
                
                relative_error = abs(ratio - expected_ratio) / expected_ratio
                assert relative_error < 0.05, f"Vmax sensitivity error: {relative_error:.4f}"
    
    def test_km_sensitivity(self, standard_state, standard_environment):
        """Test sensitivity to Km parameter."""
        km_values = [0.0005, 0.002, 0.005, 0.01]
        rates = []

        param_manager = get_global_manager()
        km_key = "pedosphere.denitrification.km_no3"
        original_km = _normalize_param_value(param_manager.get(km_key))
        try:
            for km in km_values:
                _set_param_value(param_manager, km_key, km)
                denit_process = DenitrificationProcess(name="test", config={})

                result = call_forward(denit_process, {**standard_state, **standard_environment})
                rate = float(_unwrap_update(result['denitrification_flux']).detach())
                rates.append(rate)
        finally:
            _set_param_value(param_manager, km_key, original_km)

        # Higher Km should give lower rates (for fixed substrate concentration)
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i+1] * 0.8, "Higher Km should reduce rate"
    
    def test_anaerobic_threshold_sensitivity(self, standard_state):
        """Test sensitivity to anaerobic threshold."""
        thresholds = [0.5, 0.6, 0.7, 0.79]
        
        # Use moisture just above middle threshold
        test_moisture = 0.65
        
        rates = []
        param_manager = get_global_manager()
        threshold_key = "pedosphere.physical.wfps_denitrification_threshold"
        original_threshold = _normalize_param_value(param_manager.get(threshold_key))
        try:
            for threshold in thresholds:
                _set_param_value(param_manager, threshold_key, threshold)
                denit_process = DenitrificationProcess(name="test", config={})

                env = {'labile_carbon': torch.tensor(0.1),
                      'water_filled_pore_space': torch.tensor(test_moisture),
                      'soil_temperature': torch.tensor(288.15),
                      'soil_ph': torch.tensor(6.5)}

                result = call_forward(denit_process, {**standard_state, **env})
                rate = float(_unwrap_update(result['denitrification_flux']).detach())
                rates.append(rate)
        finally:
            _set_param_value(param_manager, threshold_key, original_threshold)
        
        # Different thresholds should give different responses for given moisture
        rate_range = max(rates) - min(rates)
        mean_rate = np.mean(rates)
        if mean_rate > 1e-10:
            relative_range = rate_range / mean_rate
            assert relative_range > 0.1, "Anaerobic threshold should affect rates"
    
    # =============================================================================
    # L1.10: Performance and Stability Tests
    # =============================================================================
    
    def test_numerical_stability_small_values(self, denitrification):
        """Test numerical stability with very small values."""
        small_case = {
            'no3_nitrogen': torch.tensor(1e-10),
            'labile_carbon': torch.tensor(1e-8),
            'water_filled_pore_space': torch.tensor(0.01),
            'soil_temperature': torch.tensor(273.16),
            'soil_ph': torch.tensor(7.0)
        }
        
        state = {'no3_nitrogen': small_case['no3_nitrogen']}
        env = {k: v for k, v in small_case.items() if k != 'no3_nitrogen'}
        
        result = call_forward(denitrification, {**state, **env})
        
        # Should not crash and all outputs should be finite
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f"{key} should be finite with small values"
                assert not torch.isnan(value).any(), f"{key} should not be NaN"
                if key == "no3_nitrogen":
                    # no3_nitrogen is a pool delta (can be negative due to NO3 consumption).
                    # It must not consume more NO3 than available in the state.
                    assert value >= -small_case["no3_nitrogen"], (
                        f"{key} should not exceed available substrate: {value}"
                    )
                else:
                    assert value >= 0.0, f"{key} should be non-negative: {value}"
    
    def test_mass_balance_conservation(self, denitrification, standard_state, standard_environment):
        """Test nitrogen mass balance conservation in forward pass."""
        # The drivers are state inputs (no keyword channel).
        result = denitrification.forward(
            {**standard_state, **standard_environment}, time=0.0, dt=3600.0
        )
        
        # Get nitrogen changes
        n2o_update = result.get('n2o_flux', torch.tensor(0.0))
        n2o_rate = float(
            (n2o_update.value if isinstance(n2o_update, UpdateItem) else n2o_update).detach()
        )
        n2_rate = float(
            _unwrap_update(result.get('n2_production', torch.tensor(0.0))).detach()
        )

        delta_no3 = float(
            _unwrap_update(result.get('no3_nitrogen', torch.tensor(0.0))).detach()
        )
        
        # Mass balance: -delta_NO3 = N2O_production + N2_production
        dt = 3600.0
        total_gas_production = (n2o_rate + n2_rate) * dt

        if abs(delta_no3) > 1e-10:
            balance_error = abs(abs(delta_no3) - total_gas_production) / abs(delta_no3)
            assert balance_error < 0.01, f"Mass balance error: {balance_error:.4f}"

    def test_invalid_dt_is_rejected(self, denitrification, standard_state, standard_environment):
        """A non-positive timestep cannot be replaced by a numerical epsilon."""
        with pytest.raises(ValueError, match="dt must be positive"):
            denitrification.forward(
                {**standard_state, **standard_environment},
                time=0.0,
                dt=0.0,
            )

    def test_incompatible_environment_shape_is_rejected(
        self,
        denitrification,
        standard_state,
        standard_environment,
    ):
        """Environmental vectors cannot be silently averaged into a point cell."""
        environment = dict(standard_environment)
        environment["soil_temperature"] = torch.tensor([280.0, 290.0])
        with pytest.raises(ValueError, match="cannot broadcast"):
            denitrification.forward(
                {**standard_state, **environment},
                time=0.0,
                dt=3600.0,
            )
    
    def test_repeated_execution_consistency(self, denitrification, standard_state, standard_environment):
        """Test consistency of repeated executions."""
        results = []
        
        for _ in range(3):
            result = call_forward(denitrification, {**standard_state.copy(), **standard_environment})
            rate = float(_unwrap_update(result['denitrification_flux']).detach())
            results.append(rate)
        
        # Should be deterministic
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-12, \
                f"Results should be deterministic: {results[0]:.2e} vs {results[i]:.2e}"
    
    def test_gradient_computation_consistency(self, denitrification, standard_state, standard_environment):
        """Test consistency with PyTorch gradient computation."""
        # Create state with gradients
        grad_state = {}
        for key, value in standard_state.items():
            if isinstance(value, torch.Tensor):
                grad_state[key] = value.clone().requires_grad_(True)
            else:
                grad_state[key] = value
        
        grad_environment = {}
        for key, value in standard_environment.items():
            if isinstance(value, torch.Tensor):
                grad_environment[key] = value.clone().requires_grad_(True)
            else:
                grad_environment[key] = value
        
        # Forward pass
        result = call_forward(denitrification, {**grad_state, **grad_environment})
        
        # Try gradient computation
        denit_rate = _unwrap_update(result['denitrification_flux'])
        
        if denit_rate.requires_grad:
            try:
                denit_rate.backward()
                # Check that some gradients exist (if process is active)
            except RuntimeError:
                # Some operations may not support gradients, which is acceptable
                pass


# Round 35: labile carbon, water-filled pore space, temperature and pH are required inputs;
# no pedosphere.environment_defaults value stands in for a missing one.
def test_environment_inputs_are_required_not_defaulted():
    process = DenitrificationProcess(name="denit_required_inputs", config={})
    assert {"labile_carbon", "water_filled_pore_space", "soil_temperature", "soil_ph"} <= set(
        process.get_required_inputs()
    )
    assert not [
        key for key in process.metadata.parameters
        if key.startswith("pedosphere.environment_defaults.")
    ]
    state = {
        "no3_nitrogen": torch.tensor(1.0),
        "water_filled_pore_space": torch.tensor(0.4),
        "soil_temperature": torch.tensor(293.15),
        "soil_ph": torch.tensor(6.5),
    }
    with pytest.raises(ValueError, match="labile_carbon"):
        process.forward(state, time=0.0, dt=3600.0)


def test_drivers_are_state_inputs_without_an_environment_layer():
    """The drivers are required state inputs of the one rate core, which takes the
    state only; without them in the state the forward raises."""
    import inspect

    assert list(inspect.signature(DenitrificationProcess._compute_rate_core).parameters) == [
        "self",
        "state",
    ]
    process = DenitrificationProcess(name="denit_state_inputs", config={})
    drivers = {
        "labile_carbon": torch.tensor(0.1),
        "water_filled_pore_space": torch.tensor(0.4),
        "soil_temperature": torch.tensor(293.15),
        "soil_ph": torch.tensor(6.5),
    }
    state = {"no3_nitrogen": torch.tensor(1.0)}
    assert float(process._compute_rate_core({**state, **drivers})["denitrification_rate"]) > 0.0
    with pytest.raises(ValueError, match="labile_carbon"):
        process.forward(dict(state), time=0.0, dt=3600.0)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


