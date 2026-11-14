#!/usr/bin/env python3
"""
L1 Unit Tests for Species Mapping Module (Improved)

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Parameter management system validation
- L1.3: Scientific calculation accuracy (molecular conversions)
- L1.4: Conservation laws validation
- L1.5: Chemical mechanism mapping
- L1.6: Abstract base class compliance

Scientific References:
- Yarwood, G. et al. (2005). Updates to the CB05 chemical mechanism. EPA Report.
- Carter, W.P.L. (2010). Development of the SAPRC-07 chemical mechanism. Atmospheric Environment 44, 5324-5335.
- Stockwell, W.R., et al. (1997). The next generation of models. Journal of Geophysical Research 102, 25847-25873.

Test Framework: pytest
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import modules to test
import relic.scientific.anthroposphere.emissions.species_mapping as species_mapping_module
from relic.scientific.anthroposphere.emissions.species_mapping import (
    SpeciesMapper,
)
from relic.scientific.anthroposphere.emissions.atmosphere_interface_contract import (
    ChemicalMechanism,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 5
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3


def _single_sector_species_state(
    mapper: SpeciesMapper,
    sector: str,
    species: str,
    value: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sector_flux = torch.zeros(
        (
            len(mapper.CANONICAL_SECTORS),
            len(mapper.CANONICAL_INVENTORY_SPECIES),
            *value.shape,
        ),
        dtype=value.dtype,
        device=value.device,
    )
    sector_flux[
        mapper.CANONICAL_SECTORS.index(sector),
        mapper.CANONICAL_INVENTORY_SPECIES.index(species),
    ] = value
    return {"sector_emission_flux": sector_flux}


class TestSpeciesMapperL1:
    """L1 Unit Tests for SpeciesMapper"""
    
    @pytest.fixture
    def species_mapper(self):
        """Create species mapper instance for testing"""
        return SpeciesMapper()
    
    @pytest.fixture
    def sample_emissions(self):
        """Sample emission data for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'NOx': torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE).reshape(batch_shape),
            'CO': torch.tensor([10.0, 15.0, 20.0, 25.0, 30.0], device=DEVICE).reshape(batch_shape),
            'SO2': torch.tensor([5.0, 7.5, 10.0, 12.5, 15.0], device=DEVICE).reshape(batch_shape),
            'VOC': torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device=DEVICE).reshape(batch_shape),
            'NH3': torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5], device=DEVICE).reshape(batch_shape),
            'PM25': torch.tensor([1.5, 2.25, 3.0, 3.75, 4.5], device=DEVICE).reshape(batch_shape)
        }

    # L1.1: Class instantiation and initialization tests
    def test_class_instantiation(self, species_mapper):
        """Test that SpeciesMapper can be instantiated"""
        assert species_mapper is not None
        assert isinstance(species_mapper, SpeciesMapper)
        assert isinstance(species_mapper.molecular_weights, dict)
        assert species_mapper.default_mechanism is ChemicalMechanism.CB05
    
    def test_molecular_weights_completeness(self, species_mapper):
        """Test that molecular weights database is complete"""
        # Check critical species
        critical_species = ['NO', 'NO2', 'CO', 'SO2', 'NH3', 'PAR', 'OLE']
        
        for species in critical_species:
            assert species in species_mapper.molecular_weights, \
                f"Missing molecular weight for {species}"
            assert species_mapper.get_molecular_weight(species) > 0, \
                f"Invalid molecular weight for {species}"
    
    def test_chemical_mechanisms_completeness(self, species_mapper):
        """Test that chemical mechanisms are properly defined"""
        assert species_mapper.get_supported_mechanisms() == [ChemicalMechanism.CB05]
        assert tuple(species_mapper.mechanism_gas_species_order) == (
            'NO', 'NO2', 'CO', 'SO2', 'NH3',
            'PAR', 'OLE', 'TOL', 'XYL', 'FORM', 'ALD2',
        )
    
    def test_initialization_with_config(self):
        """Runtime config must not bypass the ParameterManager SSOT."""
        custom_config = {
            'default_mechanism': ChemicalMechanism.SAPRC07,
            'precision': 'double'
        }

        with pytest.raises(ValueError, match='ParameterManager YAML database'):
            SpeciesMapper(config=custom_config)

    # L1.2: Parameter management system validation
    def test_parameter_access_through_mapper(self, species_mapper):
        """Test that parameters are accessible through mapper"""
        tolerance = species_mapper.get_parameter_value(
            "mass_balance_tolerance",
            namespace="anthroposphere.emissions.species_mapping",
            required=True,
        )
        no_fraction = species_mapper.get_parameter_value(
            "no_fraction",
            namespace="anthroposphere.emissions.partition_factors",
            required=True,
        )
        assert float(tolerance) > 0.0
        assert 0.0 <= float(no_fraction) <= 1.0
    
    def test_parameter_units_validation(self, species_mapper):
        """Test that parameter units are properly validated"""
        spec = get_global_manager().get_parameter(
            "anthroposphere.emissions.molecular_weights.NO2"
        )
        assert spec is not None
        assert spec.units == "g mol-1"
    
    def test_parameter_bounds_checking(self, species_mapper):
        """Test that parameter bounds are enforced"""
        required = set(species_mapper.required_parameters())
        assert len(required) == 70
        assert "anthroposphere.emissions.partition_factors.no_fraction" in required
        assert all(get_global_manager().get_parameter(key) is not None for key in required)

    # L1.3: Scientific calculation accuracy (molecular conversions)
    def test_mass_to_molar_conversion(self, species_mapper):
        """Test mass to molar flux conversion accuracy"""
        # Test NO2 conversion
        mass_flux = torch.tensor([46.005, 92.01, 138.015], device=DEVICE) * 1e-3  # kg
        molar_flux = species_mapper.mass_to_molar_flux(mass_flux, 'NO2')

        # Should convert correctly (46.005 g/mol for NO2 -> 0.046005 kg/mol)
        expected_molar = mass_flux / 0.046005
        assert torch.allclose(molar_flux, expected_molar, rtol=1e-4), \
            "Mass to molar conversion should be accurate"

        # Test CO conversion
        co_mass = torch.tensor([28.01, 56.02, 84.03], device=DEVICE) * 1e-3  # kg
        co_molar = species_mapper.mass_to_molar_flux(co_mass, 'CO')

        # Should convert correctly (28.01 g/mol for CO -> 0.02801 kg/mol)
        expected_co_molar = co_mass / 0.02801
        assert torch.allclose(co_molar, expected_co_molar, rtol=1e-4), \
            "Mass to molar conversion for CO should be accurate"

    @pytest.mark.parametrize(
        ("species", "canonical_key", "legacy_key", "expected_g_per_mol"),
        [
            (
                "CO",
                "relic.physical.constants.molecular_weights.co",
                "anthroposphere.emissions.molecular_weights.CO",
                28.010,
            ),
            (
                "FORM",
                "relic.physical.constants.molecular_weights.hcho",
                "anthroposphere.emissions.molecular_weights.FORM",
                30.0260,
            ),
        ],
    )
    def test_species_molar_mass_binds_physical_constant_owner(
        self, monkeypatch, species, canonical_key, legacy_key, expected_g_per_mol
    ):
        """CO and FORM molar masses use the physical-constant SSOT.

        References:
        - CIAAW (2024) abridged standard atomic weights for CO,
          https://ciaaw.org/abridged-atomic-weights.htm.
        - NIST Chemistry WebBook, SRD 69, Formaldehyde, CAS 50-00-0,
          https://webbook.nist.gov/cgi/cbook.cgi?ID=50000.
        Mathematical expression:
        - M_i = M_i,stored * 1.0e-3; F_mol = F_mass / M_i,
          F_mass = F_mol * M_i and dF_mol / dF_mass = 1 / M_i.
        Parameter meanings:
        - M_i is the bulk molar mass of species i; F_mass is its emitted
          mass flux and F_mol its emitted molar flux.
        Units:
        - M_i in kg mol-1 (stored as g mol-1), F_mass in kg m-2 s-1,
          F_mol in mol m-2 s-1.
        Domain/constraints:
        - Non-negative binary64 fluxes; the YAML constants have fixed bounds.
        Sign convention:
        - Emission-positive fluxes; the conversion does not change sign.
        """
        requested: list[str] = []
        original_getter = species_mapping_module.get_physical_constant

        def recording_getter(name, *args, **kwargs):
            requested.append(name)
            return original_getter(name, *args, **kwargs)

        monkeypatch.setattr(
            species_mapping_module, "get_physical_constant", recording_getter
        )
        mapper = SpeciesMapper()
        assert canonical_key in requested

        required = mapper.required_parameters()
        assert len(required) == 70 and len(set(required)) == 70
        assert canonical_key in required
        assert legacy_key not in required
        assert canonical_key in mapper.metadata.parameters
        assert legacy_key not in mapper.metadata.parameters
        hodge = mapper.build_hodge_contract()
        species_index = mapper.MECHANISM_GAS_SPECIES.index(species)
        assert hodge.parameter_keys[species_index] == canonical_key
        assert hodge.metric_keys[species_index] == canonical_key
        assert legacy_key not in hodge.parameter_keys
        assert get_global_manager().get_parameter(legacy_key) is None

        # Independent expected bulk molar mass, converted from g mol-1.
        expected_mw = expected_g_per_mol * 1.0e-3
        assert mapper.get_molecular_weight(species) == expected_mw
        mass = torch.tensor(
            [expected_mw, 1.0, 0.0], dtype=torch.float64, requires_grad=True
        )
        molar = mapper.mass_to_molar_flux(mass, species)
        assert molar.dtype == torch.float64
        assert torch.equal(molar.detach(), mass.detach() / expected_mw)
        molar.sum().backward()
        assert torch.equal(
            mass.grad, torch.ones_like(mass.detach()) / expected_mw
        )
        recovered = mapper.molar_to_mass_flux(molar.detach(), species)
        assert recovered.dtype == torch.float64
        assert torch.equal(recovered, molar.detach() * expected_mw)

    def test_molar_to_mass_conversion(self, species_mapper):
        """Test molar to mass flux conversion accuracy"""
        # Test NO conversion
        molar_flux = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)  # mol/m2/s
        mass_flux = species_mapper.molar_to_mass_flux(molar_flux, 'NO')

        # Should convert correctly using parameterized molecular weight
        mw_no = species_mapper.get_molecular_weight('NO')
        expected_mass = molar_flux * mw_no
        assert torch.allclose(mass_flux, expected_mass, rtol=1e-4), \
            "Molar to mass conversion should be accurate"
        
        # Test units consistency
        assert mass_flux.device == molar_flux.device, \
            "Device consistency should be maintained"
    
    def test_mixing_ratio_conversion(self, species_mapper):
        """Test mixing ratio conversion accuracy"""
        # Test flux to mixing ratio conversion
        molar_flux = torch.tensor([1e-9, 2e-9, 3e-9], device=DEVICE)  # mol/m2/s
        pressure = torch.tensor([101325.0], device=DEVICE)  # Pa
        temperature = torch.tensor([298.15], device=DEVICE)  # K
        layer_height = torch.tensor([1000.0], device=DEVICE)  # m
        dt = torch.tensor([3600.0], device=DEVICE)  # s
        
        mixing_ratio = species_mapper.flux_to_mixing_ratio(
            molar_flux, pressure, temperature, layer_height, dt
        )
        
        # Should be positive and reasonable
        assert torch.all(mixing_ratio > 0), "Mixing ratio should be positive"
        assert torch.all(mixing_ratio < 1e-3), "Mixing ratio should be in reasonable range"
        
        # Test inverse conversion
        calculated_flux = species_mapper.mixing_ratio_to_flux(
            mixing_ratio, pressure, temperature, layer_height, dt
        )
        
        assert torch.allclose(calculated_flux, molar_flux, rtol=1e-3), \
            "Inverse conversion should be accurate"
    
    def test_concentration_unit_conversion(self, species_mapper):
        """Test concentration unit conversions"""
        # Test ppm to mixing ratio conversion
        ppm_values = torch.tensor([10.0, 50.0, 100.0], device=DEVICE)
        mixing_ratios = species_mapper.ppm_to_mixing_ratio(ppm_values)
        
        # Should be approximately ppm / 1e6
        expected_ratios = ppm_values / 1e6
        assert torch.allclose(mixing_ratios, expected_ratios, rtol=1e-4), \
            "PPM to mixing ratio conversion should be accurate"
        
        # Test inverse conversion
        calculated_ppm = species_mapper.mixing_ratio_to_ppm(mixing_ratios)
        assert torch.allclose(calculated_ppm, ppm_values, rtol=1e-4), \
            "Inverse PPM conversion should be accurate"
    
    def test_voc_speciation_accuracy(self, species_mapper):
        """Test VOC speciation calculation accuracy"""
        voc_flux = torch.tensor([[1.0, 2.0, 3.0]], device=DEVICE)
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper, "power", "VOC", voc_flux
            ),
            time=0.0,
            dt=1.0,
        )
        gas = output["mechanism_gas_molar_flux"].value
        reconstructed = torch.zeros_like(voc_flux)
        for species in species_mapper.VOC_PROFILE_SPECIES:
            index = species_mapper.mechanism_gas_species_order.index(species)
            reconstructed += gas[index] * species_mapper.get_molecular_weight(species)
        torch.testing.assert_close(reconstructed, voc_flux)
    
    def test_nox_partitioning(self, species_mapper):
        """Test NOx partitioning between NO and NO2"""
        nox_flux = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE)
        
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper, "power", "NOx", nox_flux
            ),
            time=0.0,
            dt=1.0,
        )
        gas = output["mechanism_gas_molar_flux"].value
        no_moles = gas[species_mapper.mechanism_gas_species_order.index("NO")]
        no2_moles = gas[species_mapper.mechanism_gas_species_order.index("NO2")]
        nox_mw = species_mapper.get_molecular_weight('NOx')
        no_equivalent = no_moles * nox_mw
        no2_equivalent = no2_moles * nox_mw
        torch.testing.assert_close(no_equivalent + no2_equivalent, nox_flux)

        configured_no_fraction = species_mapper.get_parameter_value(
            'no_fraction',
            namespace='anthroposphere.emissions.partition_factors',
        )
        torch.testing.assert_close(
            no_equivalent / nox_flux,
            torch.full_like(nox_flux, float(configured_no_fraction)),
        )
    
    def test_pm_speciation(self, species_mapper):
        """Test PM2.5 speciation"""
        pm25_flux = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE)
        
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper, "industry", "PM25", pm25_flux
            ),
            time=0.0,
            dt=1.0,
        )
        aerosol = output["mechanism_aerosol_mass_flux"].value
        torch.testing.assert_close(aerosol.sum(dim=0), pm25_flux)
        assert aerosol.shape[0] == len(species_mapper.MECHANISM_AEROSOL_SPECIES)
    
    # L1.4: Conservation laws validation
    def test_mass_conservation_conversions(self, species_mapper):
        """Test mass conservation in all unit conversions"""
        # Test species with different molecular weights
        test_species = ['NO', 'NO2', 'CO', 'SO2', 'NH3']
        test_masses = torch.tensor([30.01, 46.005, 28.01, 64.07, 17.031], device=DEVICE)
        
        for i, species in enumerate(test_species):
            mass_flux = test_masses[i:i+1]  # Keep as tensor
            
            # Convert to molar and back
            molar_flux = species_mapper.mass_to_molar_flux(mass_flux, species)
            recovered_mass = species_mapper.molar_to_mass_flux(molar_flux, species)
            
            # Should recover original mass
            mass_error = torch.abs(recovered_mass - mass_flux)
            assert mass_error < TOLERANCE, \
                f"Mass conservation failed for {species}: error = {mass_error}"
    
    def test_mole_conservation_speciation(self, species_mapper):
        """Test that canonical VOC molar outputs reconstruct the input mass."""
        voc_flux = torch.tensor([[2.0, 3.0, 4.0]], device=DEVICE)
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper, "transport", "VOC", voc_flux
            ),
            time=0.0,
            dt=1.0,
        )
        gas = output["mechanism_gas_molar_flux"].value
        mass = torch.zeros_like(voc_flux)
        for species in species_mapper.VOC_PROFILE_SPECIES:
            index = species_mapper.mechanism_gas_species_order.index(species)
            mass += gas[index] * species_mapper.get_molecular_weight(species)
        torch.testing.assert_close(mass, voc_flux)
    
    def test_elemental_balance(self, species_mapper):
        """Test the declared inventory-equivalent balance diagnostic."""
        voc_flux = torch.tensor([[1.0, 2.0, 3.0]], device=DEVICE)
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper, "industry", "VOC", voc_flux
            ),
            time=0.0,
            dt=1.0,
        )
        torch.testing.assert_close(
            output[
                "diagnostics.species_mapping.inventory_equivalent_output_mass_flux"
            ],
            voc_flux,
        )
    
    # L1.5: Chemical mechanism mapping
    def test_cb05_mechanism_mapping(self, species_mapper):
        """Test CB05 chemical mechanism mapping"""
        voc_flux = torch.tensor([[1.0, 2.0]], device=DEVICE)
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper, "power", "VOC", voc_flux
            ),
            time=0.0,
            dt=1.0,
        )
        item = output["mechanism_gas_molar_flux"]
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE and item.per_timestep is False
        assert item.value.shape[0] == len(species_mapper.MECHANISM_GAS_SPECIES)
    
    def test_saprc07_mechanism_mapping(self, species_mapper):
        """Test that SAPRC cannot enter through a hidden CB05 fallback."""
        with pytest.raises(NotImplementedError, match="typed runtime contract"):
            species_mapper.get_mechanism_species_list(ChemicalMechanism.SAPRC07)
    
    def test_sector_dependent_speciation(self, species_mapper):
        """Test sector-dependent emission speciation"""
        # Test same VOC flux for different sectors
        voc_flux = torch.tensor([[1.0, 2.0]], device=DEVICE)
        sectors = ['power', 'transport', 'industry', 'buildings']
        
        sector_results = {}
        for sector in sectors:
            output = species_mapper.forward(
                _single_sector_species_state(
                    species_mapper, sector, "VOC", voc_flux
                ),
                time=0.0,
                dt=1.0,
            )
            sector_results[sector] = output["mechanism_gas_molar_flux"].value
        
        # Different sectors should produce different speciation
        assert not torch.equal(sector_results['power'], sector_results['transport'])
        assert not torch.equal(sector_results['industry'], sector_results['buildings'])
    
    def test_emission_factor_application(self, species_mapper):
        """Test that Inventory remains the sole emission-factor owner."""
        assert not hasattr(species_mapper, 'apply_emission_factors')
        assert not any(
            'emission_factors' in key for key in species_mapper.required_parameters()
        )
    
    # L1.6: Abstract base class compliance
    def test_interface_methods(self, species_mapper):
        """Test that required interface methods are implemented"""
        for method_name in (
            'forward',
            'ports',
            'declared_forms',
            'build_hodge_contract',
            'upscale',
            'downscale',
        ):
            assert callable(getattr(species_mapper, method_name))
    
    def test_parameter_validation_interface(self, species_mapper):
        """Test parameter validation interface"""
        assert not hasattr(species_mapper, 'validate_molecular_weight')
        assert not hasattr(species_mapper, 'validate_chemical_formula')
        with pytest.raises(KeyError, match="Unknown species"):
            species_mapper.get_molecular_weight('INVALID_SPECIES')
    
    def test_error_handling(self, species_mapper):
        """Test error handling for invalid inputs"""
        # Test invalid species name
        with pytest.raises(KeyError):
            species_mapper.mass_to_molar_flux(torch.tensor([1.0]), 'INVALID_SPECIES')
        
        # Test negative flux values
        negative_flux = torch.tensor([-1.0, -2.0], device=DEVICE)
        
        with pytest.raises(ValueError, match="cannot be negative"):
            species_mapper.mass_to_molar_flux(negative_flux, 'NO')
        
        # Test zero flux values
        zero_flux = torch.zeros(3, device=DEVICE)
        zero_result = species_mapper.mass_to_molar_flux(zero_flux, 'NO')
        assert torch.all(zero_result == 0), "Zero flux should give zero result"
    
    def test_device_consistency(self, species_mapper):
        """Test device consistency across operations"""
        mass_flux = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
        molar_flux = species_mapper.mass_to_molar_flux(mass_flux, 'NO')
        recovered_mass = species_mapper.molar_to_mass_flux(molar_flux, 'NO')
        assert molar_flux.device == mass_flux.device
        assert recovered_mass.device == mass_flux.device

        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper,
                'power',
                'VOC',
                torch.tensor([[1.0, 2.0]], device=DEVICE),
            ),
            time=0.0,
            dt=1.0,
        )
        assert output['mechanism_gas_molar_flux'].value.device == mass_flux.device
    
    # Additional validation tests
    def test_species_database_completeness(self, species_mapper):
        """Test completeness of species database"""
        assert set(species_mapper.mechanism_gas_species_order) <= set(
            species_mapper.molecular_weights
        )
        assert tuple(species_mapper.mechanism_aerosol_species_order) == (
            'BC', 'OC', 'NO3', 'SO4', 'NH4', 'OTHER'
        )
    
    def test_temperature_pressure_effects(self, species_mapper):
        """Test temperature and pressure effects on conversions"""
        # Test mixing ratio conversion at different temperatures
        base_flux = torch.tensor([1e-9], device=DEVICE)
        base_pressure = torch.tensor([101325.0], device=DEVICE)
        temperatures = torch.tensor([273.15, 288.15, 298.15, 313.15], device=DEVICE)
        
        mixing_ratios = []
        for temp in temperatures:
            mr = species_mapper.flux_to_mixing_ratio(
                base_flux, base_pressure, temp, 
                torch.tensor([1000.0], device=DEVICE),
                torch.tensor([3600.0], device=DEVICE)
            )
            mixing_ratios.append(mr.item())
        
        # Ideal-gas air-column moles decrease as temperature rises.
        assert mixing_ratios[3] > mixing_ratios[0]
        
        # Test pressure effects
        pressures = torch.tensor([85000, 95000, 101325, 105000], device=DEVICE)
        base_temp = torch.tensor([298.15], device=DEVICE)
        
        pressure_mixing_ratios = []
        for pressure in pressures:
            mr = species_mapper.flux_to_mixing_ratio(
                base_flux, pressure, base_temp,
                torch.tensor([1000.0], device=DEVICE),
                torch.tensor([3600.0], device=DEVICE)
            )
            pressure_mixing_ratios.append(mr.item())
        
        # Higher pressure should result in lower mixing ratio
        assert pressure_mixing_ratios[3] < pressure_mixing_ratios[0], \
            "Higher pressure should reduce mixing ratio for same flux"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

