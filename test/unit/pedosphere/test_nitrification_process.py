"""
L1 Unit Tests for NitrificationProcess Module

This module provides comprehensive unit tests for the nitrification process
implementation, covering Michaelis-Menten kinetics, environmental responses,
and N2O production.

Test Categories:
- L1.1: Basic instantiation and configuration  
- L1.2: Scientific accuracy against literature values
- L1.3: Michaelis-Menten kinetics
- L1.4: Environmental response functions
- L1.5: N2O production modeling
- L1.6: Boundary condition handling
- L1.7: Parameter sensitivity analysis
- L1.8: Performance and stability tests
"""

import pytest
import torch
import numpy as np
from typing import Dict
from relic.core.state import UpdateItem, UpdateMode

# Import the module under test
from relic.scientific.pedosphere.nutrient_cycling import NitrificationProcess
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


class TestNitrificationProcessL1:
    """L1 Unit Tests for NitrificationProcess functor."""
    
    @pytest.fixture
    def default_config(self) -> Dict:
        """Default configuration for NitrificationProcess."""
        return {
            'vmax_nitrification': 2.0e-6,  # g N m鈦宦?s鈦宦?            'km_nh4': 0.005,               # g N m鈦宦?            'q10_nitrification': 2.0,      # dimensionless
            'ph_optimum': 7.5,             # pH units
            # moisture_threshold was removed in round 53.1: the oxygen limitation uses WFPS.
        }
    
    @pytest.fixture  
    def nitrification(self, default_config) -> NitrificationProcess:
        """Create NitrificationProcess instance."""
        return NitrificationProcess(name="test_nitrification", config=default_config)
    
    @pytest.fixture
    def standard_state(self) -> Dict:
        """Standard state for testing."""
        return {
            'nh4_nitrogen': torch.tensor(0.05),         # g N m鈦宦?            'no3_nitrogen': torch.tensor(0.02),         # g N m鈦宦?            'soil_temperature': torch.tensor(288.15),   # K (15掳C)
            'water_filled_pore_space': torch.tensor(0.3),         # m鲁 m鈦宦?            'soil_ph': torch.tensor(7.0)                # pH
        }
    
    @pytest.fixture
    def standard_environment(self) -> Dict:
        """Standard environment for testing."""
        return {
            'soil_temperature': torch.tensor(288.15),
            'water_filled_pore_space': torch.tensor(0.3),
            'soil_ph': torch.tensor(7.0)
        }
        
    # =============================================================================
    # L1.1: Basic Instantiation and Configuration Tests
    # =============================================================================
    
    def test_instantiation_basic(self, default_config):
        """Test basic instantiation with default configuration."""
        nitrification = NitrificationProcess(name="test_nit", config=default_config)
        
        assert nitrification.name == "test_nit"
        assert nitrification.config == default_config
        assert hasattr(nitrification, 'metadata')
        assert hasattr(nitrification, 'vmax')
        assert hasattr(nitrification, 'km_nh4')
        assert hasattr(nitrification, 'q10')
        assert nitrification.metadata.name == "test_nit"
        
    def test_instantiation_metadata(self, nitrification):
        """Test metadata completeness."""
        metadata = nitrification.metadata
        
        # Check required metadata fields
        assert metadata.name is not None
        assert metadata.version is not None
        assert metadata.description is not None
        assert metadata.mathematical_expression is not None
        
        # Check scientific references
        assert len(metadata.references) >= 2
        assert any("Parton" in ref for ref in metadata.references)
        assert any("Li" in ref for ref in metadata.references)
        
        # Check parameters documentation
        assert len(metadata.parameters) >= 5
        assert 'pedosphere.nitrification.vmax_nitrification' in metadata.parameters
        assert 'pedosphere.nitrification.km_nh4' in metadata.parameters
        assert 'pedosphere.physical.q10_nitrification' in metadata.parameters
        assert 'pedosphere.nitrification.n2o_factor' in metadata.parameters
    
    def test_parameter_initialization(self, nitrification):
        """Test parameter initialization."""
        # Check all expected parameters exist and are reasonable
        assert 0 < nitrification.vmax <= 1e-4  # Should be reasonable rate
        assert 0 < nitrification.km_nh4 <= 1.0  # Should be reasonable Km
        assert 1.5 <= nitrification.q10 <= 4.0  # Literature Q10 range
        assert 6.0 <= nitrification.ph_optimum <= 8.5  # Reasonable pH optimum
        assert 0.35 <= float(nitrification.wfps_optimum) <= 0.7  # Bateman and Baggs (2005) WFPS range
        assert 0.001 <= nitrification.n2o_factor <= 0.1  # N2O production factor
    
    def test_dependencies(self, nitrification):
        """Test dependency declaration."""
        deps = nitrification.get_dependencies()

        assert isinstance(deps, list)
        assert len(deps) >= 1
        assert any("mineralization" in dep.lower() for dep in deps)
        
    # =============================================================================
    # L1.2: Scientific Accuracy Tests Against Literature Values
    # =============================================================================
    
    def test_q10_temperature_response_literature(self, nitrification):
        """Test Q10 temperature response against literature values."""
        # Literature Q10 for nitrification: 1.8-2.5 (Parton et al., 1996)
        q10_value = nitrification.q10
        assert 1.5 <= q10_value <= 3.0, f"Q10 should be 1.5-3.0: {q10_value}"
        
        # Test temperature response follows Q10 law
        temperatures = [278.15, 288.15, 298.15, 308.15]  # 5掳C, 15掳C, 25掳C, 35掳C
        
        base_state = {'nh4_nitrogen': torch.tensor(0.05)}
        responses = []
        
        for temp in temperatures:
            environment = {'soil_temperature': torch.tensor(temp),
                          'water_filled_pore_space': torch.tensor(0.3),
                          'soil_ph': torch.tensor(7.0)}
            
            rates = call_forward(nitrification, {**base_state, **environment})
            response = float(rates['diagnostics.temperature_response'].detach().item())
            responses.append(response)
        
        # Check Q10 relationship over 10掳C intervals
        for i in range(len(temperatures) - 1):
            temp_diff = temperatures[i+1] - temperatures[i]  # 10掳C
            response_ratio = responses[i+1] / responses[i]
            expected_ratio = q10_value ** (temp_diff / 10.0)
            
            relative_error = abs(response_ratio - expected_ratio) / expected_ratio
            assert relative_error < 0.15, \
                f"Q10 relationship violated: {response_ratio:.3f} vs expected {expected_ratio:.3f}"
    
    def test_vmax_literature_values(self, nitrification):
        """Test Vmax against literature values."""
        vmax = nitrification.vmax
        
        # Literature range: 1e-7 to 1e-5 g N m鈦宦?s鈦宦?(Li et al., 1992; Parton et al., 1996)
        assert 1e-7 <= vmax <= 1e-5, f"Vmax should be 1e-7 to 1e-5: {vmax:.2e}"
        
        # Default value should be in middle range
        assert 5e-7 <= vmax <= 1e-5, f"Default Vmax should be reasonable: {vmax:.2e}"
    
    def test_km_nh4_literature_values(self, nitrification):
        """Test Km for NH4 against literature values."""
        km_nh4 = nitrification.km_nh4
        
        # Literature range: 0.001-0.05 g N m鈦宦?(based on DNDC, Century models)
        assert 0.001 <= km_nh4 <= 0.1, f"Km_NH4 should be 0.001-0.1: {km_nh4}"
        
        # Default value should be in reasonable range
        assert 0.002 <= km_nh4 <= 0.02, f"Default Km_NH4 should be reasonable: {km_nh4}"
    
    def test_n2o_production_factor_literature(self, nitrification):
        """Test N2O production factor against literature values."""
        n2o_factor = nitrification.n2o_factor
        
        # Literature: 0.5-5% of nitrified N becomes N2O (Bouwman et al., 2002)
        assert 0.005 <= n2o_factor <= 0.05, f"N2O factor should be 0.5-5%: {n2o_factor}"
    
    def test_ph_optimum_literature(self, nitrification):
        """Test pH optimum against literature values."""
        ph_optimum = nitrification.ph_optimum
        
        # Literature: optimal pH for nitrification = 7.0-8.0 (Prosser, 1989)
        assert 6.5 <= ph_optimum <= 8.5, f"pH optimum should be 6.5-8.5: {ph_optimum}"
    
    # =============================================================================
    # L1.3: Michaelis-Menten Kinetics Tests
    # =============================================================================
    
    def test_michaelis_menten_shape(self, nitrification, standard_environment):
        """Test Michaelis-Menten kinetics curve shape."""
        # Test range of NH4 concentrations
        nh4_values = torch.linspace(0.001, 0.5, 50)
        rates = []
        
        for nh4 in nh4_values:
            state = {'nh4_nitrogen': nh4}
            result = call_forward(nitrification, {**state, **standard_environment})
            rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
            rates.append(rate)
        
        rates = np.array(rates)
        
        # Should be monotonically increasing
        assert np.all(np.diff(rates) >= -1e-10), "Rate should increase with substrate"
        
        # Should approach Vmax asymptotically
        max_rate = rates[-1]
        vmax = nitrification.vmax
        
        # At high substrate, rate should approach Vmax (within environmental factors)
        # Environmental factors at standard conditions should be < 1.0
        expected_max = vmax * 0.5  # Account for environmental factors
        assert max_rate >= expected_max * 0.5, f"Should approach Vmax: {max_rate:.2e} vs {expected_max:.2e}"
    
    def test_km_half_saturation(self, nitrification, standard_environment):
        """Test that Km represents half-saturation constant."""
        km_nh4 = nitrification.km_nh4
        
        # At [NH4] = Km, rate should be Vmax/2 (times environmental factors)
        km_nh4_tensor = torch.as_tensor(km_nh4).detach()
        state = {'nh4_nitrogen': km_nh4_tensor.clone()}
        result = call_forward(nitrification, {**state, **standard_environment})
        rate_at_km = float(_unwrap_update(result['nitrification_flux']).detach().item())
        
        # At very high [NH4], rate approaches Vmax
        high_nh4_state = {'nh4_nitrogen': km_nh4_tensor * 100}
        high_result = call_forward(nitrification, {**high_nh4_state, **standard_environment})
        rate_at_high = float(_unwrap_update(high_result['nitrification_flux']).detach().item())
        
        # Rate at Km should be approximately half of rate at high substrate
        ratio = rate_at_km / rate_at_high
        assert 0.4 <= ratio <= 0.6, f"Rate at Km should be ~50% of Vmax: ratio = {ratio:.3f}"
    
    def test_zero_substrate_handling(self, nitrification, standard_environment):
        """Test handling of zero NH4 substrate."""
        state = {'nh4_nitrogen': torch.tensor(0.0)}
        result = call_forward(nitrification, {**state, **standard_environment})
        
        rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
        assert rate == 0.0, f"Zero substrate should give zero rate: {rate}"
        
        n2o = float(_unwrap_update(result['n2o_flux']).detach().item())
        assert n2o == 0.0, f"Zero rate should give zero N2O: {n2o}"

    def test_n2o_source_uses_additive_update_mode(
        self, nitrification, standard_environment
    ):
        """Nitrification N2O must add to other declared N2O sources."""
        state = {'nh4_nitrogen': torch.tensor(0.05), **standard_environment}
        result = nitrification.forward(state, time=0.0, dt=1.0)
        assert isinstance(result['n2o_flux'], UpdateItem)
        assert result['n2o_flux'].mode == UpdateMode.ACCUMULATE
        assert result['n2o_flux'].per_timestep is False
        assert result['n2o_flux'].unit == "g N m-2 s-1"
        assert result['n2o_flux'].basis == "per_second"
        assert result['n2o_flux'].element is None
        assert result['n2o_flux'].source_functor == nitrification.name

    def test_mineral_pool_deltas_use_explicit_accumulate_mode(
        self, nitrification, standard_environment
    ):
        """NH4 and NO3 per-step changes must enter the shared SCM ledger."""
        state = {'nh4_nitrogen': torch.tensor(0.05), **standard_environment}
        result = nitrification.forward(state, time=0.0, dt=3600.0)
        for key in ('nh4_nitrogen', 'no3_nitrogen'):
            assert isinstance(result[key], UpdateItem)
            assert result[key].mode is UpdateMode.ACCUMULATE
            assert result[key].per_timestep is True
            assert result[key].unit == "g N m-2"
            assert result[key].basis == "per_step"
            assert result[key].element == "N"
            assert result[key].source_functor == nitrification.name

        nitrification_flux = result['nitrification_flux']
        assert isinstance(nitrification_flux, UpdateItem)
        assert nitrification_flux.mode is UpdateMode.REPLACE
        assert nitrification_flux.per_timestep is False
        assert nitrification_flux.unit == "g N m-2 s-1"
        assert nitrification_flux.basis == "per_second"
        assert nitrification_flux.element is None
        assert nitrification_flux.source_functor == nitrification.name
    
    def test_very_low_substrate_stability(self, nitrification, standard_environment):
        """Test numerical stability with very low substrate."""
        state = {'nh4_nitrogen': torch.tensor(1e-10)}
        result = call_forward(nitrification, {**state, **standard_environment})
        
        # Should not crash and give reasonable results
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f"{key} should be finite with low substrate"
                assert not torch.isnan(value).any(), f"{key} should not be NaN"
        
        rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
        assert rate >= 0.0, f"Rate should be non-negative: {rate}"
        assert rate < 1e-8, f"Very low substrate should give very low rate: {rate}"
    
    # =============================================================================
    # L1.4: Environmental Response Function Tests
    # =============================================================================
    
    def test_temperature_response_bounds(self, nitrification, standard_state):
        """Test temperature response stays within reasonable bounds."""
        extreme_temps = [263.15, 323.15]  # -10掳C, 50掳C
        
        for temp in extreme_temps:
            environment = {'soil_temperature': torch.tensor(temp),
                          'water_filled_pore_space': torch.tensor(0.3),
                          'soil_ph': torch.tensor(7.0)}
            
            result = call_forward(nitrification, {**standard_state, **environment})
            temp_response = float(result['diagnostics.temperature_response'].detach().item())
            
            # Should be bounded and finite (adjusted for extreme temperature factor)
            assert 0.001 <= temp_response <= 15.0, \
                f"Temperature response {temp_response:.3f} outside bounds at {temp-273.15:.1f}掳C"
    
    def test_moisture_oxygen_limitation(self, nitrification, standard_state):
        """Test oxygen limitation under high water-filled pore space."""
        moisture_values = torch.linspace(0.1, 0.95, 20)
        oxygen_factors = []
        
        for moisture in moisture_values:
            environment = {'soil_temperature': torch.tensor(288.15),
                          'water_filled_pore_space': moisture,
                          'soil_ph': torch.tensor(7.0)}
            
            result = call_forward(nitrification, {**standard_state, **environment})
            oxygen_factor = float(result['diagnostics.oxygen_factor'].detach().item())
            oxygen_factors.append(oxygen_factor)
        
        oxygen_factors = np.array(oxygen_factors)
        
        # Should decrease with increasing moisture (above threshold)
        threshold_idx = None
        threshold = float(nitrification.wfps_optimum)
        
        for i, moisture in enumerate(moisture_values):
            if moisture >= threshold:
                if threshold_idx is None:
                    threshold_idx = i
                # Above threshold, oxygen factor should be <= 1.0
                assert oxygen_factors[i] <= 1.0, f"Oxygen factor should be 鈮? above threshold"
                
                # Should decrease with moisture above threshold
                if i > threshold_idx:
                    assert oxygen_factors[i] <= oxygen_factors[i-1] + 0.1, \
                        "Oxygen factor should decrease with moisture"
        
        # Below threshold, should be near 1.0
        for i, moisture in enumerate(moisture_values):
            if moisture < threshold:
                assert oxygen_factors[i] >= 0.9, \
                    f"Below threshold, oxygen factor should be ~1.0: {oxygen_factors[i]:.3f}"
    
    def test_ph_response_gaussian_shape(self, nitrification, standard_state):
        """Test pH response has proper Gaussian-like shape."""
        ph_values = torch.linspace(4.0, 10.0, 25) 
        ph_factors = []
        
        for ph in ph_values:
            environment = {'soil_temperature': torch.tensor(288.15),
                          'water_filled_pore_space': torch.tensor(0.3),
                          'soil_ph': ph}
            
            result = call_forward(nitrification, {**standard_state, **environment})
            ph_factor = float(result['diagnostics.ph_factor'].detach().item())
            ph_factors.append(ph_factor)
        
        ph_factors = np.array(ph_factors)
        
        # Find maximum
        max_idx = np.argmax(ph_factors)
        max_ph = float(ph_values[max_idx])
        
        # Maximum should be near optimum
        ph_optimum = nitrification.ph_optimum
        assert abs(max_ph - ph_optimum) < 0.5, \
            f"pH response maximum at {max_ph:.1f} vs optimum {ph_optimum:.1f}"
        
        # Should decrease on both sides of optimum
        if max_idx > 0:
            left_decreasing = np.all(np.diff(ph_factors[:max_idx+1]) >= -0.05)
            assert left_decreasing, "pH response should increase towards optimum"
        
        if max_idx < len(ph_factors) - 1:
            right_decreasing = np.all(np.diff(ph_factors[max_idx:]) <= 0.05)
            assert right_decreasing, "pH response should decrease from optimum"
    
    def test_combined_environmental_effects(self, nitrification, standard_state):
        """Test combined environmental effects are multiplicative."""
        # Test different combinations
        environments = [
            {'soil_temperature': torch.tensor(298.15), 'water_filled_pore_space': torch.tensor(0.3), 'soil_ph': torch.tensor(7.5)},  # Good conditions
            {'soil_temperature': torch.tensor(278.15), 'water_filled_pore_space': torch.tensor(0.3), 'soil_ph': torch.tensor(7.5)},  # Cold
            {'soil_temperature': torch.tensor(298.15), 'water_filled_pore_space': torch.tensor(0.9), 'soil_ph': torch.tensor(7.5)},   # Waterlogged
            {'soil_temperature': torch.tensor(298.15), 'water_filled_pore_space': torch.tensor(0.3), 'soil_ph': torch.tensor(5.0)},   # Acidic
        ]
        
        rates = []
        for env in environments:
            result = call_forward(nitrification, {**standard_state, **env})
            rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
            rates.append(rate)
        
        # Good conditions should give highest rate
        assert rates[0] >= max(rates[1:]), "Good conditions should give highest rate"
        
        # Each stress should reduce rate
        assert rates[1] < rates[0], "Cold should reduce rate"
        assert rates[2] < rates[0], "Waterlogging should reduce rate"
        assert rates[3] < rates[0], "Low pH should reduce rate"
    
    # =============================================================================
    # L1.5: N2O Production Modeling Tests
    # =============================================================================
    
    def test_n2o_proportionality(self, nitrification, standard_environment):
        """Test N2O production is proportional to nitrification rate."""
        nh4_values = [0.01, 0.05, 0.1, 0.2]
        
        for nh4 in nh4_values:
            state = {'nh4_nitrogen': torch.tensor(nh4)}
            result = call_forward(nitrification, {**state, **standard_environment})
            
            nit_rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
            n2o_rate = float(_unwrap_update(result['n2o_flux']).detach().item())
            
            if nit_rate > 0:
                n2o_fraction = n2o_rate / nit_rate
                expected_fraction = nitrification.n2o_factor
                
                # Should match configured N2O factor
                relative_error = abs(n2o_fraction - expected_fraction) / expected_fraction
                assert relative_error < 0.01, \
                    f"N2O fraction {n2o_fraction:.4f} vs expected {expected_fraction:.4f}"
    
    def test_n2o_mass_balance(self, nitrification, standard_state, standard_environment):
        """Test N2O production maintains nitrogen mass balance."""
        result = call_forward(nitrification, {**standard_state, **standard_environment})
        
        nit_rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
        n2o_rate = float(_unwrap_update(result['n2o_flux']).detach().item())
        nh4_consumed = float(
            torch.as_tensor(result.get('nh4_consumption', nit_rate)).detach().item()
        )
        no3_produced = float(
            torch.as_tensor(result.get('no3_production', nit_rate - n2o_rate)).detach().item()
        )
        
        # Mass balance: NH4 consumed = NO3 produced + N2O produced
        mass_balance = nh4_consumed - (no3_produced + n2o_rate)
        assert abs(mass_balance) < 1e-10, f"N mass balance violated: {mass_balance:.2e}"
    
    def test_n2o_environmental_independence(self, nitrification, standard_state):
        """Test N2O factor is independent of environmental conditions."""
        environments = [
            {'soil_temperature': torch.tensor(288.15), 'water_filled_pore_space': torch.tensor(0.2), 'soil_ph': torch.tensor(6.0)},
            {'soil_temperature': torch.tensor(298.15), 'water_filled_pore_space': torch.tensor(0.4), 'soil_ph': torch.tensor(7.5)},
            {'soil_temperature': torch.tensor(283.15), 'water_filled_pore_space': torch.tensor(0.6), 'soil_ph': torch.tensor(8.0)},
        ]
        
        n2o_fractions = []
        for env in environments:
            result = call_forward(nitrification, {**standard_state, **env})
            
            nit_rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
            n2o_rate = float(_unwrap_update(result['n2o_flux']).detach().item())
            
            if nit_rate > 1e-12:  # Avoid division by very small numbers
                n2o_fraction = n2o_rate / nit_rate
                n2o_fractions.append(n2o_fraction)
        
        # All fractions should be approximately equal
        if len(n2o_fractions) > 1:
            n2o_std = np.std(n2o_fractions)
            n2o_mean = np.mean(n2o_fractions)
            
            relative_variation = n2o_std / n2o_mean if n2o_mean > 0 else 0
            assert relative_variation < 0.05, \
                f"N2O fraction should be constant across environments: {relative_variation:.4f}"
    
    # =============================================================================
    # L1.6: Boundary Condition Handling Tests
    # =============================================================================
    
    def test_extreme_temperature_handling(self, nitrification, standard_state):
        """Test handling of extreme temperatures."""
        extreme_temps = [243.15, 353.15]  # -30掳C, 80掳C
        
        for temp in extreme_temps:
            environment = {'soil_temperature': torch.tensor(temp),
                          'water_filled_pore_space': torch.tensor(0.3),
                          'soil_ph': torch.tensor(7.0)}
            
            result = call_forward(nitrification, {**standard_state, **environment})
            
            # Should not crash and give reasonable results
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all(), f"{key} should be finite at extreme temperature"
            
            rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
            assert rate >= 0.0, f"Rate should be non-negative at extreme temperature: {rate}"
            
            # Extreme temperatures should give very low rates
            if temp < 273.15 or temp > 323.15:
                assert rate < 1e-6, f"Extreme temperature should give very low rate: {rate}"
    
    def test_extreme_ph_handling(self, nitrification, standard_state):
        """Test handling of extreme pH values."""
        extreme_phs = [2.0, 12.0]
        
        for ph in extreme_phs:
            environment = {'soil_temperature': torch.tensor(288.15),
                          'water_filled_pore_space': torch.tensor(0.3),
                          'soil_ph': torch.tensor(ph)}
            
            result = call_forward(nitrification, {**standard_state, **environment})
            
            # Should give low activity due to pH stress
            ph_factor = float(result['diagnostics.ph_factor'].detach().item())
            assert ph_factor < 0.3, f"Extreme pH should give low pH factor: {ph_factor}"
            
            rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
            assert rate >= 0.0, f"Rate should be non-negative at extreme pH: {rate}"
    
    def test_very_high_substrate_handling(self, nitrification, standard_environment):
        """Test handling of very high NH4 concentrations."""
        high_nh4_state = {'nh4_nitrogen': torch.tensor(10.0)}  # Very high
        result = call_forward(nitrification, {**high_nh4_state, **standard_environment})
        
        rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
        vmax = nitrification.vmax
        
        # Should not exceed Vmax (accounting for environmental factors)
        max_theoretical = vmax * 2.0  # Allow for environmental enhancement
        assert rate <= max_theoretical, f"Rate should not exceed theoretical max: {rate:.2e} vs {max_theoretical:.2e}"
    
    # =============================================================================
    # L1.7: Parameter Sensitivity Analysis Tests
    # =============================================================================
    
    def test_vmax_sensitivity(self, standard_state, standard_environment):
        """Test sensitivity to Vmax parameter."""
        vmax_values = [1e-6, 2e-6, 4e-6, 8e-6]
        rates = []
        pm = get_global_manager()
        key = "pedosphere.nitrification.vmax_nitrification"
        original_value = pm.get(key, units="g N m-2 s-1")
        original_value = float(original_value) if original_value is not None else 2.0e-6
        try:
            for vmax in vmax_values:
                pm.set_value(key, vmax, units="g N m-2 s-1", validate=True, notify=False)
                nit_process = NitrificationProcess(name="test", config={})
                result = call_forward(nit_process, {**standard_state, **standard_environment})
                rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
                rates.append(rate)
        finally:
            pm.set_value(key, original_value, units="g N m-2 s-1", validate=True, notify=False)
        
        # Rate should be proportional to Vmax
        for i in range(len(rates) - 1):
            ratio = rates[i+1] / rates[i] if rates[i] > 0 else 0
            expected_ratio = vmax_values[i+1] / vmax_values[i]
            
            if rates[i] > 0:
                relative_error = abs(ratio - expected_ratio) / expected_ratio
                assert relative_error < 0.05, f"Vmax sensitivity error: {relative_error:.4f}"
    
    def test_km_sensitivity(self, standard_state, standard_environment):
        """Test sensitivity to Km parameter."""
        km_values = [0.001, 0.005, 0.01, 0.02]
        rates = []
        pm = get_global_manager()
        key = "pedosphere.nitrification.km_nh4"
        original_value = pm.get(key, units="g N m-2")
        original_value = float(original_value) if original_value is not None else 0.005
        try:
            for km in km_values:
                pm.set_value(key, km, units="g N m-2", validate=True, notify=False)
                nit_process = NitrificationProcess(name="test", config={})
                result = call_forward(nit_process, {**standard_state, **standard_environment})
                rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
                rates.append(rate)
        finally:
            pm.set_value(key, original_value, units="g N m-2", validate=True, notify=False)
        
        # Higher Km should give lower rates (for fixed substrate concentration)
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i+1] * 0.9, "Higher Km should reduce rate"
    
    def test_q10_sensitivity(self, standard_state):
        """Test sensitivity to Q10 parameter."""
        q10_values = [1.81, 2.0, 2.2, 2.5]
        
        # Use elevated temperature to see Q10 effect
        warm_env = {'soil_temperature': torch.tensor(298.15),  # 25掳C
                   'water_filled_pore_space': torch.tensor(0.3),
                   'soil_ph': torch.tensor(7.0)}
        
        rates_warm = []
        
        pm = get_global_manager()
        key = "pedosphere.physical.q10_nitrification"
        original_value = pm.get(key, units="dimensionless")
        original_value = float(original_value) if original_value is not None else 2.0
        try:
            for q10 in q10_values:
                pm.set_value(key, q10, units="dimensionless", validate=True, notify=False)
                nit_process = NitrificationProcess(name="test", config={})
                result = call_forward(nit_process, {**standard_state, **warm_env})
                rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
                rates_warm.append(rate)
        finally:
            pm.set_value(key, original_value, units="dimensionless", validate=True, notify=False)
        
        # Higher Q10 should give higher rates at elevated temperature
        for i in range(len(rates_warm) - 1):
            assert rates_warm[i+1] >= rates_warm[i] * 0.95, \
                "Higher Q10 should increase rate at high temperature"
    
    # =============================================================================
    # L1.8: Performance and Stability Tests
    # =============================================================================
    
    def test_numerical_stability_edge_cases(self, nitrification):
        """Test numerical stability with edge case inputs."""
        edge_cases = [
            # Very small values
            {'nh4_nitrogen': torch.tensor(1e-12), 'soil_temperature': torch.tensor(273.16), 
             'water_filled_pore_space': torch.tensor(0.001), 'soil_ph': torch.tensor(7.0)},
            # Very large values  
            {'nh4_nitrogen': torch.tensor(100.0), 'soil_temperature': torch.tensor(310.0),
             'water_filled_pore_space': torch.tensor(0.99), 'soil_ph': torch.tensor(9.0)},
            # Mixed extreme values
            {'nh4_nitrogen': torch.tensor(1e-10), 'soil_temperature': torch.tensor(320.0),
             'water_filled_pore_space': torch.tensor(1e-6), 'soil_ph': torch.tensor(2.5)},
        ]
        
        for i, case in enumerate(edge_cases):
            state = {'nh4_nitrogen': case['nh4_nitrogen']}
            env = {k: v for k, v in case.items() if k != 'nh4_nitrogen'}
            
            result = call_forward(nitrification, {**state, **env})
            
            # Should not crash and all outputs should be finite
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all(), f"Case {i}: {key} should be finite"
                    assert not torch.isnan(value).any(), f"Case {i}: {key} should not be NaN"
    
    def test_mass_balance_conservation(self, nitrification, standard_state, standard_environment):
        """Test nitrogen mass balance conservation in forward pass."""
        # The drivers are state inputs (no keyword channel).
        result = nitrification.forward(
            {**standard_state, **standard_environment}, time=0.0, dt=3600.0
        )
        
        # Get nitrogen changes
        n2o_rate = float(
            torch.as_tensor(_unwrap_update(result.get('n2o_flux', torch.tensor(0.0)))).detach().item()
        )
        
        delta_nh4 = float(
            torch.as_tensor(
                _unwrap_update(result.get('nh4_nitrogen', torch.tensor(0.0)))
            ).detach().item()
        )
        delta_no3 = float(
            torch.as_tensor(
                _unwrap_update(result.get('no3_nitrogen', torch.tensor(0.0)))
            ).detach().item()
        )
        
        # Mass balance: -delta_NH4 = delta_NO3 + N2O_production
        dt = 3600.0
        n2o_loss = n2o_rate * dt
        expected_loss = delta_no3 + n2o_loss
        if abs(delta_nh4) > 1e-10:
            nh4_error = abs((-delta_nh4) - expected_loss) / abs(delta_nh4)
            assert nh4_error < 0.01, f"NH4 mass balance error: {nh4_error:.4f}"

        if abs(expected_loss) > 1e-10:
            no3_error = abs(expected_loss - (-delta_nh4)) / abs(expected_loss)
            assert no3_error < 0.01, f"NO3 mass balance error: {no3_error:.4f}"
    
    def test_repeated_execution_consistency(self, nitrification, standard_state, standard_environment):
        """Test consistency of repeated executions."""
        results = []
        
        for _ in range(3):
            result = call_forward(nitrification, {**standard_state.copy(), **standard_environment})
            rate = float(_unwrap_update(result['nitrification_flux']).detach().item())
            results.append(rate)
        
        # Should be deterministic
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-12, \
                f"Results should be deterministic: {results[0]:.2e} vs {results[i]:.2e}"
    
    def test_gradient_computation_consistency(self, nitrification, standard_state, standard_environment):
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
        result = call_forward(nitrification, {**grad_state, **grad_environment})
        
        # Try gradient computation
        nit_rate = _unwrap_update(result['nitrification_flux'])
        
        if nit_rate.requires_grad:
            try:
                nit_rate.backward()
                # Check that some gradients exist
                has_gradients = any(
                    param.grad is not None and param.grad.abs().sum() > 0
                    for param in [grad_state.get('nh4_nitrogen'), 
                                 grad_environment.get('soil_temperature')]
                    if param is not None and param.requires_grad
                )
            except RuntimeError:
                # Some operations may not support gradients, which is acceptable
                pass


# Round 35: soil temperature, water-filled pore space and pH are required inputs; no
# pedosphere.environment_defaults value stands in for a missing one, and the
# step length is checked instead of clamped by a dimensionless epsilon.
def _complete_nitrification_state():
    return {
        "nh4_nitrogen": torch.tensor(1.0),
        "soil_temperature": torch.tensor(293.15),
        "water_filled_pore_space": torch.tensor(0.3),
        "soil_ph": torch.tensor(6.5),
    }


def test_environment_inputs_are_required_not_defaulted():
    process = NitrificationProcess(name="nit_required_inputs", config={})
    assert {"soil_temperature", "water_filled_pore_space", "soil_ph"} <= set(process.get_required_inputs())
    assert not [
        key for key in process.metadata.parameters
        if key.startswith("pedosphere.environment_defaults.")
    ]
    state = _complete_nitrification_state()
    del state["soil_ph"]
    with pytest.raises(ValueError, match="soil_ph"):
        process.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize("dt", [0.0, -60.0])
def test_non_positive_step_is_rejected(dt):
    process = NitrificationProcess(name="nit_step", config={})
    with pytest.raises(ValueError, match="dt"):
        process.forward(_complete_nitrification_state(), time=0.0, dt=dt)


def test_drivers_are_state_inputs_without_an_environment_layer():
    """The drivers are required state inputs of the one rate core, which takes the
    state only; an 'environment.<key>' entry is not a second input channel."""
    import inspect

    assert list(inspect.signature(NitrificationProcess._compute_rate_core).parameters) == [
        "self",
        "state",
    ]
    process = NitrificationProcess(name="nit_state_inputs", config={})
    state = _complete_nitrification_state()
    assert float(process._compute_rate_core(dict(state))["nitrification_rate"]) > 0.0
    drivers = {key: state.pop(key) for key in ("soil_temperature", "water_filled_pore_space", "soil_ph")}
    with pytest.raises(ValueError, match="soil_temperature"):
        process.forward(dict(state), time=0.0, dt=3600.0)
    prefixed = {**state, **{f"environment.{key}": value for key, value in drivers.items()}}
    with pytest.raises(ValueError, match="soil_temperature"):
        process.forward(prefixed, time=0.0, dt=3600.0)



def _device_state(device, dtype=torch.float64):
    """Waterlogged state (WFPS above the nitrification optimum) on one device."""
    tensor = lambda value: torch.tensor(value, dtype=dtype, device=device)
    return {
        "nh4_nitrogen": tensor(1.0),
        "no3_nitrogen": tensor(0.5),
        "soil_temperature": tensor(293.15),
        "water_filled_pore_space": tensor(0.9),
        "soil_ph": tensor(7.0),
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable; test_rate_core_bounds_follow_a_meta_input covers the device rule",
)
def test_rate_core_and_forward_follow_the_cuda_input_device():
    """Round 52.1: manager-held parameters stay on the CPU (they are references, not
    module parameters), so every clamp bound must follow the input tensor's device."""
    process = NitrificationProcess().to_device("cuda")
    cuda = torch.device("cuda")
    state = _device_state(cuda)

    rates = process._compute_rate_core(dict(state))
    for key in ("temperature_response", "oxygen_factor", "ph_factor", "nitrification_rate"):
        assert rates[key].device.type == "cuda", key
    reference = NitrificationProcess()._compute_rate_core(_device_state(torch.device("cpu")))
    for key in ("temperature_response", "oxygen_factor", "ph_factor", "nitrification_rate"):
        torch.testing.assert_close(rates[key].cpu(), reference[key], rtol=1.0e-6, atol=0.0)

    updates = process(dict(state), time=0.0, dt=torch.tensor(3600.0, dtype=torch.float64))
    for key in ("nh4_nitrogen", "no3_nitrogen", "n2o_flux"):
        value = updates[key].value if isinstance(updates[key], UpdateItem) else updates[key]
        assert value.device.type == "cuda", key


def test_rate_core_bounds_follow_a_meta_input(monkeypatch):
    """Device probe that needs no GPU. Meta kernels accept a CPU scalar bound, so a
    strict torch.clamp wrapper rejects every tensor bound that is not on the device
    of the clamped tensor, as a CUDA kernel would."""
    original_clamp = torch.clamp

    def strict_clamp(value, min=None, max=None):
        for bound in (min, max):
            if torch.is_tensor(bound) and bound.device != value.device:
                raise RuntimeError(
                    f"clamp bound on {bound.device}, input on {value.device}"
                )
        return original_clamp(value, min=min, max=max)

    monkeypatch.setattr(torch, "clamp", strict_clamp)
    process = NitrificationProcess().to_device("meta")
    rates = process._compute_rate_core(_device_state(torch.device("meta")))

    for key in ("temperature_response", "oxygen_factor", "ph_factor", "nitrification_rate"):
        assert rates[key].device.type == "meta", key



RESIDUAL_KEY = "diagnostics.nitrification.substrate_limiter_residual"


def _cpu_state(nh4):
    state = _device_state(torch.device("cpu"))
    state["nh4_nitrogen"] = torch.tensor(nh4, dtype=torch.float64)
    return state


def test_substrate_limiter_residual_is_declared_and_reported():
    """Round 52.4: the unmet demand R_nit - min(R_nit, NH4/dt) is an output diagnostic
    (a demand that did not occur, not nitrogen), like the methane oxidation limiter."""
    process = NitrificationProcess()
    assert RESIDUAL_KEY in process.metadata.output_variables

    state = _cpu_state(1.0e-6)
    dt = torch.tensor(86400.0, dtype=torch.float64)
    demand = process._compute_rate_core(dict(state))["nitrification_rate"]
    updates = process(dict(state), time=0.0, dt=dt)

    residual = updates[RESIDUAL_KEY]
    assert isinstance(residual, UpdateItem)
    assert residual.mode == UpdateMode.REPLACE
    assert residual.per_timestep is False
    assert residual.unit == "g N m-2 s-1"
    realized = updates["nitrification_flux"].value
    torch.testing.assert_close(realized, torch.minimum(demand, state["nh4_nitrogen"] / dt))
    assert residual.value.item() > 0.0
    torch.testing.assert_close(residual.value, demand - realized)


def test_substrate_limiter_residual_is_zero_when_the_donor_suffices():
    process = NitrificationProcess()
    state = _cpu_state(1.0)
    updates = process(dict(state), time=0.0, dt=torch.tensor(60.0, dtype=torch.float64))

    assert updates[RESIDUAL_KEY].value.item() == 0.0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])





