#!/usr/bin/env python3
"""
L1 Unit Tests for Species Mapping Module

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Molecular weight database validation
- L1.3: Species conversion calculations
- L1.4: Chemical mechanism speciation
- L1.5: VOC and other species breakdown
- L1.6: Abstract base class compliance

Scientific References:
- Carter, J.A. (2010). Development of the SAPRC-07 chemical mechanism. Atmospheric Environment 44(40), 5324-5335.
- Yarwood, G. et al. (2005). Development and evaluation of the CB05 gas-phase chemical mechanism. Journal of Applied Remote Sensing.
- EPA (2020). SPECIATE v5.0: Speciation database development for the EPA.

Test Framework: pytest
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import modules to test
from relic.scientific.anthroposphere.emissions.species_mapping import (
    SpeciesMapper,
)
from relic.scientific.anthroposphere.emissions.atmosphere_interface_contract import (
    ChemicalMechanism,
)
from relic.core.abstractions import ScaleCoupling
from relic.core.functor import ScaleCouplingFunctor
from relic.core.state import UpdateItem

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 4
TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3


def _sector_state(
    mapper: SpeciesMapper,
    emissions: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    reference = next(iter(emissions.values()))
    sector_flux = torch.zeros(
        (
            len(mapper.CANONICAL_SECTORS),
            len(mapper.CANONICAL_INVENTORY_SPECIES),
            *reference.shape,
        ),
        dtype=reference.dtype,
        device=reference.device,
    )
    for species_index, species in enumerate(mapper.CANONICAL_INVENTORY_SPECIES):
        sector_flux[:, species_index] = (
            emissions[species] / len(mapper.CANONICAL_SECTORS)
        )
    return {"sector_emission_flux": sector_flux}


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
        """Sample emissions dictionary for testing"""
        batch_shape = (TEST_BATCH_SIZE, 1)
        return {
            'NOx': torch.tensor([100.0, 80.0, 120.0, 90.0], device=DEVICE).reshape(batch_shape),
            'CO': torch.tensor([200.0, 160.0, 240.0, 180.0], device=DEVICE).reshape(batch_shape),
            'SO2': torch.tensor([80.0, 64.0, 96.0, 72.0], device=DEVICE).reshape(batch_shape),
            'VOC': torch.tensor([150.0, 120.0, 180.0, 135.0], device=DEVICE).reshape(batch_shape),
            'NH3': torch.tensor([30.0, 24.0, 36.0, 27.0], device=DEVICE).reshape(batch_shape),
            'PM25': torch.tensor([40.0, 32.0, 48.0, 36.0], device=DEVICE).reshape(batch_shape)
        }

    # L1.1: Class instantiation and initialization tests
    def test_class_instantiation(self, species_mapper):
        """Test that SpeciesMapper can be instantiated"""
        assert species_mapper is not None
        assert isinstance(species_mapper, SpeciesMapper)
        assert isinstance(species_mapper, ScaleCouplingFunctor)
        assert isinstance(species_mapper, ScaleCoupling)
        assert hasattr(species_mapper, 'name')
        assert species_mapper.name == "species_mapping"

    def test_molecular_weight_database_initialization(self, species_mapper):
        """Test molecular weight database initialization"""
        assert isinstance(species_mapper.molecular_weights, dict)

        assert set(species_mapper.molecular_weights) == set(
            species_mapper.MOLECULAR_WEIGHT_SPECIES
        )
        for species in species_mapper.MOLECULAR_WEIGHT_SPECIES:
            mw = species_mapper.get_molecular_weight(species)
            assert isinstance(mw, float)
            assert mw > 0

    def test_chemical_mechanism_initialization(self, species_mapper):
        """Test chemical mechanism initialization"""
        assert species_mapper.default_mechanism is ChemicalMechanism.CB05
        assert species_mapper.get_supported_mechanisms() == [ChemicalMechanism.CB05]
        assert species_mapper.get_mechanism_species_list(ChemicalMechanism.CB05) == [
            *species_mapper.mechanism_gas_species_order,
            *species_mapper.mechanism_aerosol_species_order,
        ]

    def test_speciation_factors_initialization(self, species_mapper):
        """Test speciation factors initialization"""
        assert set(species_mapper.voc_speciation_profiles) == set(
            species_mapper.CANONICAL_SECTORS
        )
        for sector in species_mapper.CANONICAL_SECTORS:
            profile = species_mapper.voc_speciation_profiles[sector]
            assert tuple(profile) == species_mapper.VOC_PROFILE_SPECIES
            assert sum(profile.values()) == pytest.approx(1.0, abs=1.0e-12)

    def test_mapper_owns_the_exact_canonical_input_species_axis(self):
        assert SpeciesMapper.CANONICAL_INVENTORY_SPECIES == (
            "NOx",
            "CO",
            "SO2",
            "VOC",
            "NH3",
            "PM25",
        )

    # L1.2: Molecular weight database validation
    def test_common_molecular_weights(self, species_mapper):
        """Test molecular weights of common species"""
        # Test molecular weights with known values
        known_mw = {
            'NO': 30.01,      # g/mol
            'NO2': 46.01,     # g/mol
            'CO': 28.01,      # g/mol
            'SO2': 64.07,     # g/mol
            'NH3': 17.03,     # g/mol
        }

        for species, expected_mw in known_mw.items():
            actual_mw = species_mapper.get_molecular_weight(species)

            expected_mw_kg = expected_mw / 1000.0
            # Should be close to expected value (within 1%) using kg/mol
            assert abs(actual_mw - expected_mw_kg) / expected_mw_kg < 0.01, \
                f"Molecular weight for {species}: expected {expected_mw_kg} kg/mol, got {actual_mw}"

    def test_complex_species_molecular_weights(self, species_mapper):
        """Test representative CB05 molecular weights."""
        complex_species = {
            'PAR': 14.0,
            'OLE': 28.0,
            'TOL': 92.14,
            'XYL': 106.17,
            'FORM': 30.026,
            'ALD2': 44.053,
        }

        for species, expected_mw in complex_species.items():
            actual_mw = species_mapper.get_molecular_weight(species)
            expected_mw_kg = expected_mw / 1000.0
            assert 0.01 < actual_mw < 0.5
            assert abs(actual_mw - expected_mw_kg) / expected_mw_kg < 0.01

    def test_molecular_weight_database_completeness(self, species_mapper):
        """Test that every gas port species has a molecular weight."""
        missing_mw = set(species_mapper.mechanism_gas_species_order) - set(
            species_mapper.molecular_weights
        )
        assert not missing_mw

    # L1.3: Species conversion calculations
    def test_mass_to_molar_conversion(self, species_mapper):
        """Test mass to molar flux conversion"""
        # Test conversion for known species
        test_cases = [
            ('NO2', 46.005, 1.0),      # 46.005 g/s should give 1 mol/s
            ('CO', 28.01, 1.0),         # 28.01 g/s should give 1 mol/s
            ('NH3', 17.031, 1.0),       # 17.031 g/s should give 1 mol/s
            ('SO2', 64.07, 1.0),        # 64.07 g/s should give 1 mol/s
        ]

        for species, mass_flux_g_s, expected_mol_s in test_cases:
            # Convert g to kg for the function
            mass_flux_kg_s = mass_flux_g_s / 1000.0
            mass_tensor = torch.tensor([mass_flux_kg_s], device=DEVICE)

            molar_flux = species_mapper.mass_to_molar_flux(mass_tensor, species)

            assert isinstance(molar_flux, torch.Tensor)
            assert torch.allclose(molar_flux, torch.tensor([expected_mol_s], device=DEVICE), rtol=1e-3)

    def test_molar_to_mass_conversion(self, species_mapper):
        """Test molar to mass flux conversion"""
        # Test conversion for known species
        test_cases = [
            ('NO2', 1.0, 46.005),      # 1 mol/s should give 46.005 g/s
            ('CO', 1.0, 28.01),         # 1 mol/s should give 28.01 g/s
            ('NH3', 1.0, 17.031),       # 1 mol/s should give 17.031 g/s
            ('SO2', 1.0, 64.07),        # 1 mol/s should give 64.07 g/s
        ]

        for species, molar_flux_mol_s, expected_mass_g_s in test_cases:
            molar_tensor = torch.tensor([molar_flux_mol_s], device=DEVICE)

            mass_flux_kg_s = species_mapper.molar_to_mass_flux(molar_tensor, species)

            # Convert to g/s for comparison
            mass_flux_g_s = mass_flux_kg_s * 1000.0

            assert isinstance(mass_flux_kg_s, torch.Tensor)
            assert torch.allclose(mass_flux_g_s, torch.tensor([expected_mass_g_s], device=DEVICE), rtol=1e-3)

    def test_reverse_conversion_consistency(self, species_mapper):
        """Test that mass->molar->mass conversion is consistent"""
        test_species = ['NO2', 'CO', 'NH3', 'SO2']
        original_mass = torch.tensor([100.0, 50.0, 25.0, 75.0], device=DEVICE)  # kg/s

        for i, species in enumerate(test_species):
            # Mass to molar
            molar_flux = species_mapper.mass_to_molar_flux(
                original_mass[i:i+1], species
            )

            # Molar back to mass
            recovered_mass = species_mapper.molar_to_mass_flux(molar_flux, species)

            # Should recover original mass (within tolerance)
            assert torch.allclose(
                original_mass[i:i+1], recovered_mass, rtol=1e-6
            ), f"Inconsistent conversion for {species}"

    def test_batch_conversion_operations(self, species_mapper):
        """Test batch conversion operations"""
        # Create batch of different species
        batch_mass = torch.tensor([
            [46.005],  # NO2
            [28.01],   # CO
            [17.031],  # NH3
            [64.07]    # SO2
        ], device=DEVICE) / 1000.0  # Convert to kg

        species_list = ['NO2', 'CO', 'NH3', 'SO2']

        # Convert all to molar
        molar_fluxes = []
        for i, species in enumerate(species_list):
            molar_flux = species_mapper.mass_to_molar_flux(
                batch_mass[i:i+1], species
            )
            molar_fluxes.append(molar_flux)

        # Should all equal 1 mol/s
        expected_molar = torch.ones((4, 1), device=DEVICE)
        stacked_molar = torch.cat(molar_fluxes, dim=0)

        assert torch.allclose(stacked_molar, expected_molar, rtol=1e-3)

    # L1.4: Chemical mechanism speciation
    def test_cb05_mechanism_speciation(self, species_mapper):
        """Test canonical CB05 VOC speciation."""
        voc_flux = torch.tensor([100.0], device=DEVICE)  # kg/s as carbon
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper,
                "power",
                "VOC",
                voc_flux,
            ),
            time=0.0,
            dt=1.0,
        )
        gas_molar = output["mechanism_gas_molar_flux"].value
        reconstructed = torch.zeros_like(voc_flux)
        for species in species_mapper.VOC_PROFILE_SPECIES:
            index = species_mapper.mechanism_gas_species_order.index(species)
            reconstructed += (
                gas_molar[index] * species_mapper.get_molecular_weight(species)
            )
        torch.testing.assert_close(reconstructed, voc_flux)

    def test_saprc07_mechanism_speciation(self, species_mapper):
        """Test that an undeclared mechanism cannot use the CB05 runtime."""
        with pytest.raises(NotImplementedError, match="typed runtime contract"):
            species_mapper.get_mechanism_species_list(ChemicalMechanism.SAPRC07)

    def test_nox_speciation(self, species_mapper):
        """Test NOx speciation (NO and NO2)"""
        nox_flux = torch.tensor([100.0], device=DEVICE)  # kg/s as NO2 equivalent
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper,
                "power",
                "NOx",
                nox_flux,
            ),
            time=0.0,
            dt=1.0,
        )
        gas = output["mechanism_gas_molar_flux"].value
        no_moles = gas[species_mapper.mechanism_gas_species_order.index("NO")]
        no2_moles = gas[species_mapper.mechanism_gas_species_order.index("NO2")]
        expected_moles = nox_flux / species_mapper.get_molecular_weight("NOx")
        torch.testing.assert_close(no_moles + no2_moles, expected_moles)

    def test_sox_speciation(self, species_mapper):
        """Test that canonical SO2 remains on the SO2 gas axis."""
        so2_flux = torch.tensor([80.0], device=DEVICE)
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper,
                "industry",
                "SO2",
                so2_flux,
            ),
            time=0.0,
            dt=1.0,
        )
        gas = output["mechanism_gas_molar_flux"].value
        index = species_mapper.mechanism_gas_species_order.index("SO2")
        reconstructed = gas[index] * species_mapper.get_molecular_weight("SO2")
        torch.testing.assert_close(reconstructed, so2_flux)

    def test_sector_specific_speciation_patterns(self, species_mapper):
        """Test that different sectors have different speciation patterns"""
        sectors_to_test = ['power', 'transport', 'industry']
        voc_flux = torch.tensor([100.0], device=DEVICE)
        outputs = []
        for sector in sectors_to_test:
            result = species_mapper.forward(
                _single_sector_species_state(
                    species_mapper,
                    sector,
                    "VOC",
                    voc_flux,
                ),
                time=0.0,
                dt=1.0,
            )
            outputs.append(result["mechanism_gas_molar_flux"].value)
        assert not torch.equal(outputs[0], outputs[1])
        assert not torch.equal(outputs[1], outputs[2])

    # L1.5: VOC and other species breakdown
    def test_voc_compound_group_breakdown(self, species_mapper):
        """Test fixed VOC species-axis mass closure."""
        voc_flux = torch.tensor([200.0], device=DEVICE)  # kg/s as carbon
        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper,
                "industry",
                "VOC",
                voc_flux,
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

    def test_pm_speciation(self, species_mapper):
        """Test PM speciation (PM2.5 components)"""
        pm25_flux = torch.tensor([50.0], device=DEVICE)  # kg/s

        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper,
                "industry",
                "PM25",
                pm25_flux,
            ),
            time=0.0,
            dt=1.0,
        )
        aerosol = output["mechanism_aerosol_mass_flux"].value
        assert aerosol.shape[0] == len(species_mapper.MECHANISM_AEROSOL_SPECIES)
        torch.testing.assert_close(aerosol.sum(dim=0), pm25_flux)

    def test_ammonia_speciation(self, species_mapper):
        """Test ammonia speciation"""
        nh3_flux = torch.tensor([30.0], device=DEVICE)  # kg/s

        output = species_mapper.forward(
            _single_sector_species_state(
                species_mapper,
                "agriculture",
                "NH3",
                nh3_flux,
            ),
            time=0.0,
            dt=1.0,
        )
        gas = output["mechanism_gas_molar_flux"].value
        index = species_mapper.mechanism_gas_species_order.index("NH3")
        reconstructed = gas[index] * species_mapper.get_molecular_weight("NH3")
        torch.testing.assert_close(reconstructed, nh3_flux)

    def test_multi_species_batch_speciation(self, species_mapper, sample_emissions):
        """Test speciation of multiple species simultaneously"""
        original = {key: value.clone() for key, value in sample_emissions.items()}
        state = _sector_state(species_mapper, sample_emissions)
        output = species_mapper.forward(state, time=0.0, dt=1.0)
        assert isinstance(output["mechanism_gas_molar_flux"], UpdateItem)
        assert isinstance(output["mechanism_aerosol_mass_flux"], UpdateItem)
        torch.testing.assert_close(
            output[
                "diagnostics.species_mapping.inventory_equivalent_output_mass_flux"
            ],
            state["sector_emission_flux"].sum(dim=(0, 1)),
        )

        for key, value in original.items():
            torch.testing.assert_close(sample_emissions[key], value)
            assert sample_emissions[key].shape == value.shape

    # L1.6: Abstract base class compliance
    def test_scale_coupling_interface(self, species_mapper):
        """Test compliance with ScaleCoupling interface"""
        # Should implement upscale method
        assert hasattr(species_mapper, 'upscale')
        assert callable(species_mapper.upscale)

        # Should implement downscale method
        assert hasattr(species_mapper, 'downscale')
        assert callable(species_mapper.downscale)

        fine_speciation = {'NO': torch.tensor([1.0]), 'NO2': torch.tensor([0.1])}
        with pytest.raises(RuntimeError, match="does not own a spatial upscale"):
            species_mapper.upscale(fine_speciation)

    def test_downscale_regional_species(self, species_mapper):
        """Test downscaling regional species emissions"""
        # Regional species emissions
        regional_emissions = {
            'NOx': torch.tensor(1e6),  # kg/year
            'VOC': torch.tensor(5e5),  # kg/year
            'SO2': torch.tensor(2e5)   # kg/year
        }

        # Spatial distribution weights
        spatial_weights = torch.tensor([
            [0.2, 0.3],
            [0.1, 0.4]
        ])  # 2x2 grid

        with pytest.raises(RuntimeError, match="does not own a spatial downscale"):
            species_mapper.downscale(regional_emissions, spatial_weights)

    def test_forward_method_integration(self, species_mapper, sample_emissions):
        """Test forward method integration"""
        canonical_state = _sector_state(species_mapper, sample_emissions)
        original = {key: value.clone() for key, value in canonical_state.items()}
        outputs = species_mapper.forward(
            state=canonical_state,
            time=7200,  # 2 hours
            dt=3600,    # 1 hour timestep
        )

        assert isinstance(outputs, dict)
        assert set(outputs) == set(species_mapper.metadata.output_variables)
        assert torch.all(
            outputs['diagnostics.species_mapping.relative_error'] < 1.0e-6
        )

        for key, value in outputs.items():
            tensor = value.value if isinstance(value, UpdateItem) else value
            assert isinstance(tensor, torch.Tensor)
            assert not torch.isnan(tensor).any()
            assert not torch.isinf(tensor).any()
            if key != 'diagnostics.species_mapping.inventory_basis_adjustment_flux':
                assert torch.all(tensor >= 0)

        for key, value in original.items():
            torch.testing.assert_close(canonical_state[key], value)
            assert canonical_state[key].shape == value.shape

        missing_state = dict(canonical_state)
        missing_state.pop('sector_emission_flux')
        with pytest.raises(ValueError, match='sector_emission_flux'):
            species_mapper.forward(missing_state, time=7200, dt=3600)

    def test_dependency_declaration(self, species_mapper):
        """sector_allocation is the optional same-step producer of the input.

        Its output is DERIVED (visible only to later Functors of the same group
        call), so the dependency orders it first whenever it is registered;
        without it the caller supplies sector_emission_flux as input state.
        """
        from relic.core.dependency_graph import normalize_dependency_declaration

        declared = normalize_dependency_declaration(
            species_mapper.get_dependencies(), owner=species_mapper.name
        )
        assert list(declared.optional) == ['sector_allocation']
        assert list(declared.required) == []
        assert list(declared.lagged) == []
        assert species_mapper.metadata.input_variables == ['sector_emission_flux']

    # Additional validation tests
    def test_mechanism_conservation_checks(self, species_mapper):
        """Test every declared sector profile through the canonical ledger."""
        original_voc = torch.tensor([100.0], device=DEVICE)
        for sector in species_mapper.CANONICAL_SECTORS:
            output = species_mapper.forward(
                _single_sector_species_state(
                    species_mapper,
                    sector,
                    "VOC",
                    original_voc,
                ),
                time=0.0,
                dt=1.0,
            )
            torch.testing.assert_close(
                output[
                    "diagnostics.species_mapping.inventory_equivalent_output_mass_flux"
                ],
                original_voc,
            )

    def test_unit_consistency_across_conversions(self, species_mapper):
        """Test unit consistency across different conversion types"""
        # Test that all conversions use consistent units
        test_mass = torch.tensor([1000.0], device=DEVICE)  # kg/s
        species_list = ['NO2', 'CO', 'NH3']

        for species in species_list:
            # Mass to molar to mass should return original
            molar = species_mapper.mass_to_molar_flux(test_mass, species)
            recovered_mass = species_mapper.molar_to_mass_flux(molar, species)

            relative_error = torch.abs(recovered_mass - test_mass) / test_mass
            assert relative_error < 1e-5, \
                f"Unit inconsistency for {species}: {relative_error.item():.2e}"

    def test_edge_cases_and_error_handling(self, species_mapper):
        """Test edge cases and error handling"""
        zero = torch.zeros((5, 6, 2), dtype=torch.float32)
        result = species_mapper.forward(
            {"sector_emission_flux": zero},
            time=0.0,
            dt=1.0,
        )
        assert torch.count_nonzero(result["mechanism_gas_molar_flux"].value) == 0
        assert torch.count_nonzero(result["mechanism_aerosol_mass_flux"].value) == 0

        with pytest.raises(ValueError, match="prefix"):
            species_mapper.forward(
                {"sector_emission_flux": torch.zeros((4, 6, 2))},
                time=0.0,
                dt=1.0,
            )

        negative = zero.clone()
        negative[0, 0, 0] = -1.0
        with pytest.raises(ValueError, match="negative"):
            species_mapper.forward(
                {"sector_emission_flux": negative},
                time=0.0,
                dt=1.0,
            )

    def test_performance_with_large_emission_arrays(self, species_mapper):
        """Test performance with large emission arrays"""
        import time

        # Create large emission array
        large_size = 1000
        large_emissions = torch.ones((5, 6, large_size), dtype=torch.float32)

        start_time = time.time()
        result = species_mapper.forward(
            {"sector_emission_flux": large_emissions},
            time=0.0,
            dt=1.0,
        )
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete in reasonable time
        assert processing_time < 1.0, f"Too slow: {processing_time:.3f}s"
        assert isinstance(result, dict)
        assert result["mechanism_gas_molar_flux"].value.shape == (11, large_size)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

