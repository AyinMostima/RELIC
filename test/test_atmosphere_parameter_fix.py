"""
Test module for verifying atmosphere hardcoded parameter fixes.

This test module validates that:
1. Physical constants are properly loaded from parameter system
2. Calculations remain consistent after parameter system integration
3. No hardcoded values remain in the critical paths

Author: Yuchong Long
Date: 2025-09-17
"""

import pytest
import torch
from unittest.mock import patch

from relic.scientific.atmosphere import turbulence, chemistry
from relic.scientific.atmosphere.most import stability_parameters
from relic.core.state import UpdateItem, UpdateMode


class TestPhysicalConstantsIntegration:
    """Test physical constants parameter system integration."""
    
    def test_turbulence_constants_from_parameter_system(self):
        """Test that turbulence module loads constants from parameter system."""
        # Test that the get_physical_constant function works
        von_karman = turbulence.get_physical_constant("von_karman_constant")
        gravity = turbulence.get_physical_constant("gravity")
        air_density = turbulence.get_physical_constant(
            "relic.physical.constants.air.density_reference"
        )
        specific_heat = turbulence.get_physical_constant("air_specific_heat_cp")
        
        # Verify values are reasonable (within expected ranges)
        assert 0.35 <= von_karman <= 0.45, f"von Karman constant out of range: {von_karman}"
        assert 9.7 <= gravity <= 9.9, f"Gravity out of range: {gravity}"
        assert 1.0 <= air_density <= 1.4, f"Air density out of range: {air_density}"
        assert 1000 <= specific_heat <= 1010, f"Specific heat out of range: {specific_heat}"
        
        # Turbulence avoids duplicate module-level constants; Functor instances
        # cache SSOT values for runtime calculations.
        assert not hasattr(turbulence, 'VON_KARMAN_CONSTANT')
        assert not hasattr(turbulence, 'GRAVITY_ACCELERATION')
        assert not hasattr(turbulence, 'AIR_DENSITY_STANDARD')
        assert not hasattr(turbulence, 'SPECIFIC_HEAT_AIR')

        turb_functor = turbulence.TurbulenceFunctor()
        assert abs(turb_functor.von_karman_constant - von_karman) < 1e-12
        assert abs(turb_functor.gravity_acceleration - gravity) < 1e-12
        # Boussinesq reference density of the structural Hodge mobility only;
        # the runtime fluxes use the moist-air density of the inputs.
        assert abs(turb_functor.reference_air_density - air_density) < 1e-12
        assert not hasattr(turb_functor, 'air_density_standard')
        assert abs(turb_functor.air_specific_heat_cp - specific_heat) < 1e-12
    
    def test_stability_parameters_constants_from_parameter_system(self):
        """Test that stability_parameters module loads constants correctly."""
        von_karman = stability_parameters.get_physical_constant("von_karman_constant")
        gravity = stability_parameters.get_physical_constant("gravity")
        
        assert 0.35 <= von_karman <= 0.45
        assert 9.7 <= gravity <= 9.9
        
        # Test module-level constants
        assert hasattr(stability_parameters, 'VON_KARMAN_CONSTANT')
        assert hasattr(stability_parameters, 'GRAVITY_ACCELERATION')
    
    def test_chemistry_constants_from_parameter_system(self):
        """Test that chemistry module loads constants correctly."""
        avogadro = chemistry.get_physical_constant("avogadro_number")
        boltzmann = chemistry.get_physical_constant("boltzmann_constant")
        ref_pressure = chemistry.get_physical_constant("reference_pressure")
        ref_temp = chemistry.get_physical_constant("reference_temperature")
        
        # Verify values are reasonable
        assert 6.0e23 <= avogadro <= 6.1e23
        assert 1.3e-23 <= boltzmann <= 1.4e-23
        assert 100000 <= ref_pressure <= 102000
        assert 298.0 <= ref_temp <= 299.0
        
        # No module-level copies: the functor loads its constants through the SSOT.
        for name in ('AVOGADRO_NUMBER', 'BOLTZMANN_CONSTANT', 'REFERENCE_PRESSURE',
                     'REFERENCE_TEMPERATURE'):
            assert not hasattr(chemistry, name), name
        constants = chemistry.ChemistryFunctor(physical_boundary="open_mechanism")._constants
        assert constants['avogadro_number'] == pytest.approx(avogadro)
        assert constants['boltzmann_constant'] == pytest.approx(boltzmann)


class TestParameterSystemMissingManager:
    """Test behavior when parameter manager is unavailable."""
    
    def test_get_physical_constant_raises_when_manager_missing(self):
        """get_physical_constant should raise if manager is unavailable."""
        with patch('relic.utils.physical_constants.get_global_manager', return_value=None):
            with pytest.raises(ValueError):
                turbulence.get_physical_constant("von_karman_constant")
            with pytest.raises(ValueError):
                chemistry.get_physical_constant("avogadro_number")


class TestCalculationConsistency:
    """Test that calculations remain consistent after parameter system integration."""
    
    def test_turbulence_functor_basic_calculation(self):
        """Test basic turbulence functor calculation consistency."""
        # Create test state
        state = {
            'wind_speed': torch.tensor(5.0),
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.0),
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
        # Create turbulence functor
        turb_functor = turbulence.TurbulenceFunctor()
        
        # Perform calculation
        fluxes = turb_functor.forward(state, time=0.0, dt=3600.0)
        
        # Check that key outputs are present and reasonable
        assert 'friction_velocity' in fluxes
        assert 'aerodynamic_resistance' in fluxes
        assert 'diagnostics.turbulence.stability_parameter' in fluxes
        
        # Verify values are physically reasonable
        u_star_item = fluxes['friction_velocity']
        ra_item = fluxes['aerodynamic_resistance']
        assert isinstance(u_star_item, UpdateItem)
        assert isinstance(ra_item, UpdateItem)
        assert u_star_item.mode == UpdateMode.REPLACE
        assert ra_item.mode == UpdateMode.REPLACE
        u_star = u_star_item.value
        ra = ra_item.value
        
        assert u_star > 0, f"Friction velocity should be positive: {u_star}"
        assert ra > 0, f"Aerodynamic resistance should be positive: {ra}"
        assert u_star < 5.0, f"Friction velocity unreasonably high: {u_star}"
        assert ra < 1000, f"Aerodynamic resistance unreasonably high: {ra}"
    
    def test_chemistry_functor_basic_calculation(self):
        """Test basic chemistry functor calculation consistency."""
        # Create test state
        state = {
            'air_temperature': torch.tensor(298.15),
            'no_mixing_ratio': torch.tensor(1e-9),   # 1 ppb
            'no2_mixing_ratio': torch.tensor(5e-9),  # 5 ppb
            'o3_mixing_ratio': torch.tensor(50e-9),  # 50 ppb
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0)
        }
        
        # Create chemistry functor
        chem_functor = chemistry.ChemistryFunctor(physical_boundary="open_mechanism")
        
        # The current-state explicit positivity limit is about 280 s (overhead sun).
        fluxes = chem_functor.forward(state, time=0.0, dt=200.0)
        
        state_keys = ('no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio')
        for key in state_keys:
            item = fluxes[key]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == 'mol mol-1'
            assert torch.isfinite(item.value)
            assert state[key] + item.value >= 0.0
        
        # Verify photolysis rate is reasonable
        diagnostic = fluxes['diagnostics.chemistry.photolysis_rate_no2']
        assert isinstance(diagnostic, UpdateItem)
        j_no2 = diagnostic.value
        assert j_no2 > 0, f"Photolysis rate should be positive: {j_no2}"
        assert j_no2 < 0.1, f"Photolysis rate unreasonably high: {j_no2}"
    
    def test_most_stability_functions(self):
        """Test MOST stability functions consistency."""
        # Test stable conditions
        zeta_stable = torch.tensor(1.0)
        psi_m_stable = stability_parameters.psi_m(zeta_stable)
        psi_h_stable = stability_parameters.psi_h(zeta_stable)
        
        # For stable conditions, stability functions should be negative
        assert psi_m_stable < 0, "Momentum stability function should be negative for stable conditions"
        assert psi_h_stable < 0, "Heat stability function should be negative for stable conditions"
        
        # Test unstable conditions
        zeta_unstable = torch.tensor(-1.0)
        psi_m_unstable = stability_parameters.psi_m(zeta_unstable)
        psi_h_unstable = stability_parameters.psi_h(zeta_unstable)
        
        # For unstable conditions, stability functions should be positive
        assert psi_m_unstable > 0, "Momentum stability function should be positive for unstable conditions"
        assert psi_h_unstable > 0, "Heat stability function should be positive for unstable conditions"


class TestParameterConsistency:
    """Test parameter consistency across modules."""
    
    def test_von_karman_constant_consistency(self):
        """Test that von Karman constant is consistent across modules."""
        turb_vk = turbulence.get_physical_constant("von_karman_constant")
        stab_vk = stability_parameters.VON_KARMAN_CONSTANT
        
        # Should be the same value
        assert abs(turb_vk - stab_vk) < 1e-10, f"von Karman constant inconsistent: {turb_vk} vs {stab_vk}"
    
    def test_gravity_constant_consistency(self):
        """Test that gravity constant is consistent across modules."""
        turb_g = turbulence.get_physical_constant("gravity_acceleration")
        stab_g = stability_parameters.GRAVITY_ACCELERATION
        
        # Should be the same value
        assert abs(turb_g - stab_g) < 1e-10, f"Gravity constant inconsistent: {turb_g} vs {stab_g}"


class TestNoHardcodedValues:
    """Test that no hardcoded values remain in critical calculation paths."""
    
    def test_turbulence_no_hardcoded_constants(self):
        """Test that turbulence calculations don't use hardcoded constants."""
        # Check that constants are loaded from parameter system
        assert callable(turbulence.get_physical_constant), "get_physical_constant function should exist"
        
        # Verify constants are not duplicated as module-level values.
        assert not hasattr(turbulence, 'VON_KARMAN_CONSTANT')
        turb_functor = turbulence.TurbulenceFunctor()
        assert isinstance(turb_functor.von_karman_constant, (int, float))
    
    def test_chemistry_no_hardcoded_constants(self):
        """Test that chemistry calculations don't use hardcoded constants."""
        assert callable(chemistry.get_physical_constant), "get_physical_constant function should exist"
        
        # The constants are loaded per functor through the SSOT, not as module values.
        assert not hasattr(chemistry, 'AVOGADRO_NUMBER')
        assert isinstance(chemistry.ChemistryFunctor(physical_boundary="open_mechanism")._constants['avogadro_number'], float)


if __name__ == '__main__':
    # Run basic smoke tests
    test_constants = TestPhysicalConstantsIntegration()
    test_constants.test_turbulence_constants_from_parameter_system()
    test_constants.test_chemistry_constants_from_parameter_system()
    
    test_calc = TestCalculationConsistency()
    test_calc.test_turbulence_functor_basic_calculation()
    test_calc.test_chemistry_functor_basic_calculation()
    
    print("[OK] All basic tests passed - atmosphere parameter fixes working correctly")

