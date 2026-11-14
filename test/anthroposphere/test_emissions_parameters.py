"""
Test suite for emissions module parameter system integration

This test suite verifies:
1. Parameter loading from the refactored emissions_parameters.yaml
2. Correct namespace usage throughout the module
3. Unit conversions for molecular weights
4. Conservation tolerance checks
5. Integration with ParameterManager

Author: Yuchong Long
Date: 2025-09-21
"""

import pytest
import torch
from pathlib import Path
import sys
import logging

# Add relic to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.parameters import get_global_manager
from relic.scientific.anthroposphere.emissions.species_mapping import SpeciesMapper

logger = logging.getLogger(__name__)


class TestParameterLoading:
    """Test parameter loading from the new emissions_parameters.yaml"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup parameter manager for tests"""
        self.param_manager = get_global_manager()
        # Parameter manager automatically loads all available parameters
        
    def test_physical_constants_loaded(self):
        """Test that physical constants are correctly loaded"""
        # Test universal gas constant
        R_gas = self.param_manager.get(
            'relic.physical.constants.universal.gas_constant'
        )
        assert R_gas is not None, "Universal gas constant not found"
        assert abs(R_gas - 8.314462618) < 1e-6, f"Incorrect gas constant: {R_gas}"
        
        # Test Avogadro's number  
        avogadro = self.param_manager.get(
            'relic.physical.constants.universal.avogadro_constant'
        )
        assert avogadro is not None, "Avogadro's number not found"
        assert abs(avogadro - 6.02214076e23) < 1e20, f"Incorrect Avogadro: {avogadro}"
        
    def test_molecular_weights_loaded(self):
        """Test that molecular weights are correctly loaded"""
        species_to_test = {
            'NO': 30.006,
            'NO2': 46.005,
            'CO': 28.010,
            'SO2': 64.066,
            'NH3': 17.031,
            'O3': 47.998
        }
        
        for species, expected_mw in species_to_test.items():
            # The CO molar mass is owned by the physical-constant SSOT.
            key = (
                'relic.physical.constants.molecular_weights.co'
                if species == 'CO'
                else f'anthroposphere.emissions.molecular_weights.{species}'
            )
            mw = self.param_manager.get(key)
            assert mw is not None, f"Molecular weight for {species} not found"
            assert abs(mw - expected_mw) < 0.01, \
                f"Incorrect MW for {species}: {mw} vs {expected_mw}"
            
    def test_cb05_speciation_profiles(self):
        """Test VOC speciation profiles are loaded correctly"""
        profiles = self.param_manager.get(
            'anthroposphere.emissions.speciation.voc_split_fractions'
        )
        
        assert profiles is not None, "VOC speciation profiles not found"
        
        # Check that profiles exist for major sectors
        expected_sectors = ['power_plant', 'industrial', 'transport', 'residential']
        for sector in expected_sectors:
            assert sector in profiles, f"Profile for {sector} not found"
            
            # Check that fractions sum to 1.0
            fractions = profiles[sector]
            total = sum(fractions.values())
            assert abs(total - 1.0) < 0.01, \
                f"VOC fractions for {sector} don't sum to 1.0: {total}"


class TestSpeciesMapperIntegration:
    """Test SpeciesMapper with new molecular weight loading"""
    
    def test_molecular_weight_initialization(self):
        """Test that SpeciesMapper loads molecular weights correctly"""
        mapper = SpeciesMapper()
        
        # Check critical species are loaded
        critical = ['NO', 'NO2', 'CO', 'SO2', 'NH3', 'PAR']
        for species in critical:
            mw = mapper.get_molecular_weight(species)
            assert 0.010 < mw < 0.150, \
                f"MW for {species} out of range: {mw} kg/mol"
                
    def test_unit_conversion_with_parameters(self):
        """Test unit conversions using loaded molecular weights"""
        mapper = SpeciesMapper()
        mass_flux = torch.tensor(1.0)  # kg/s
        molar_flux = mapper.mass_to_molar_flux(mass_flux, 'NO2')
        
        # Expected: 1.0 kg/s / 0.046005 kg/mol 鈮?21.74 mol/s
        expected = 1.0 / mapper.get_molecular_weight('NO2')
        assert abs(molar_flux.item() - expected) < 0.1, \
            f"Incorrect conversion: {molar_flux.item()} vs {expected}"


class TestParameterValidation:
    """Test parameter validation and bounds checking"""
    
    def test_unread_temperature_response_q10_is_retired(self):
        """The emission temperature-response Q10 had no functional reader (only the
        cache warm-up list named it) and is removed from the database."""
        param_manager = get_global_manager()
        assert param_manager.get(
            'anthroposphere.emissions.temperature_response.q10', required=False
        ) is None


class TestQualityControl:
    """Test quality control parameters"""
    
    def test_quality_control_parameters(self):
        """Test QC parameters are loaded"""
        param_manager = get_global_manager()
        
        # Test mass balance check enablement
        mass_check = param_manager.get(
            'anthroposphere.emissions.quality_control.mass_balance_checks.enabled'
        )
        assert mass_check is True, "Mass balance checks should be enabled"
        
        # Test CF standards enforcement
        cf_enforce = param_manager.get(
            'anthroposphere.emissions.quality_control.unit_consistency.enforce_cf_standards'
        )
        assert cf_enforce is True, "CF standards should be enforced"
        
        # Test emission rate bounds
        min_rate = param_manager.get(
            'anthroposphere.emissions.quality_control.validation.min_emission_rate'
        )
        max_rate = param_manager.get(
            'anthroposphere.emissions.quality_control.validation.max_emission_rate'
        )
        
        assert min_rate >= 0.0, "Minimum rate should be non-negative"
        assert max_rate > min_rate, "Maximum rate should exceed minimum"
        

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

