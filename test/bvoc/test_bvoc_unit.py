"""
L1 Unit Tests for BVOC Module - Fixed for Namespace System

This test suite provides comprehensive unit-level testing for the BVOC emission
module with proper namespace-based parameter loading.

Test Coverage:
- Parameter loading with namespace verification
- MEGAN model core calculations
- Response functions
- Numerical stability
"""

import pytest
import torch
from typing import List
from torch import Tensor

# Import BVOC modules
from relic.core.state.types import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.scientific.biosphere.bvoc.megan_model import EmissionPathway, MEGANModel
from relic.scientific.biosphere.bvoc.emission_factors import EmissionFactorDatabase
from relic.scientific.biosphere.bvoc.responses import (
    BVOCLightResponse, BVOCTemperatureResponse
)
from relic.scientific.biosphere.bvoc.synthesis.isoprene import MEPPathwayModel, IsopreneSynthesis
from relic.scientific.biosphere.bvoc.synthesis.monoterpene import MonoterpeneSynthesis
from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import SesquiterpeneSynthesis


def _canonical_megan_state(
    device: torch.device,
    *,
    air_temperature: float = 303.15,
    ppfd: float = 1000.0,
    leaf_area_index: float = 2.0,
    atmospheric_co2_ppm: float = 400.0,
):
    return {
        "air_temperature": torch.tensor(air_temperature, device=device),
        "ppfd": torch.tensor(ppfd, device=device),
        "leaf_area_index": torch.tensor(leaf_area_index, device=device),
        "atmospheric_co2_ppm": torch.tensor(
            atmospheric_co2_ppm, device=device
        ),
    }


def _unwrap_megan(result):
    values = {}
    for key, item in result.items():
        assert isinstance(item, UpdateItem), key
        assert item.mode is UpdateMode.REPLACE, key
        assert item.per_timestep is False, key
        assert item.element is None, key
        values[key] = item.value
    return values


class TestParameterNamespaceLoading:
    """Test proper namespace-based parameter loading."""
    
    def test_megan_parameter_loading(self, param_manager):
        """Test MEGAN model loads parameters from correct namespaces.
        
        Verifies that parameters are loaded from:
        - biosphere.bvoc namespace for BVOC-specific parameters
        - physical_constants namespace for universal constants
        """
        model = MEGANModel(name="test_megan")
        manager = param_manager
        for compound in model.supported_compounds:
            expected = manager.get(
                f"biosphere.bvoc.emission_factors.{compound}"
            )
            assert model.emission_factors[compound].item() == pytest.approx(
                float(expected), rel=1.0e-6
            )

        assert model.alpha_light.item() == pytest.approx(
            float(manager.get("biosphere.bvoc.megan.alpha_light")),
            rel=1.0e-6,
        )
        assert model.cl_light.item() == pytest.approx(
            float(manager.get("biosphere.bvoc.megan.cl_light")),
            rel=1.0e-6,
        )
        assert model.topt.item() == pytest.approx(
            float(manager.get("biosphere.bvoc.megan.topt")),
            rel=1.0e-6,
        )
        assert model.gas_constant.item() == pytest.approx(
            float(manager.get("relic.physical.constants.universal.gas_constant")),
            rel=1.0e-6,
        )
        
    def test_synthesis_parameter_loading(self):
        """Test synthesis models load parameters correctly."""
        # Test isoprene synthesis
        iso_model = IsopreneSynthesis(name="test_iso")
        assert hasattr(iso_model, 'v_max_isoprene_synthase')
        
        # Test monoterpene synthesis  
        mono_model = MonoterpeneSynthesis()
        assert hasattr(mono_model, 'f_carbon_monoterpenes')
        
        # Test sesquiterpene synthesis
        sesq_model = SesquiterpeneSynthesis(name="test_sesq")
        assert hasattr(sesq_model, 'base_emission_factor')


class TestMEGANCalculations:
    """Test MEGAN model calculations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.model = MEGANModel(
            name="test_megan",
            emission_pathway=EmissionPathway.STANDARD,
        )
        self.device = self.model.device
        
    def test_standard_emission_calculation(self):
        """Test standard MEGAN emission calculation.
        
        Mathematical Expression: E = epsilon * gamma_T * gamma_L * rho_f / 3600
        References: Guenther et al. (2012)
        """
        state = _canonical_megan_state(self.device)
        result = _unwrap_megan(
            self.model.forward(state, time=0.0, dt=3600.0)
        )
        
        # Check outputs exist and are reasonable
        assert 'isoprene_emission' in result
        assert 'total_bvoc_emission' in result
        assert 'bvoc_carbon_allocation_flux' in result
        assert result['isoprene_emission'] >= 0.0
        assert result['total_bvoc_emission'] >= 0.0
        assert result['bvoc_carbon_allocation_flux'] >= 0.0
        assert torch.isfinite(result['bvoc_carbon_allocation_flux'])

    def test_carbon_flux_consistency(self):
        """Ensure carbon flux equals sum over species-specific contributions."""
        state = _canonical_megan_state(
            self.device,
            air_temperature=298.15,
            ppfd=1200.0,
            leaf_area_index=1.5,
        )
        result = _unwrap_megan(
            self.model.forward(state, time=0.0, dt=3600.0)
        )
        terms: List[Tensor] = []
        for compound in self.model.enabled_compounds:
            emission_key = f'{compound}_emission'
            if emission_key not in result:
                continue
            carbon_fraction = self.model._get_carbon_fraction(compound)
            terms.append(
                result[emission_key]
                * self.model.molecular_weights[compound]
                * carbon_fraction
            )
        if terms:
            expected_g = torch.stack(terms).sum(dim=0)
        else:
            expected_g = torch.zeros(1, device=self.device)
        expected = unit_converter.convert(
            expected_g,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        ).to(
            device=result['bvoc_carbon_allocation_flux'].device,
            dtype=result['bvoc_carbon_allocation_flux'].dtype,
        )
        assert torch.allclose(
            result['bvoc_carbon_allocation_flux'],
            expected,
            atol=1e-12,
            rtol=1e-6,
        )
        
    def test_temperature_response(self):
        """Test temperature response functions."""
        # Test at different temperatures
        temps = torch.tensor([283.15, 303.15, 318.15], device=self.device)  # 10, 30, 45degC
        
        for temp in temps:
            state = _canonical_megan_state(
                self.device,
                air_temperature=float(temp.item()),
            )
            result = _unwrap_megan(
                self.model.forward(state, time=0.0, dt=3600.0)
            )
            
            # All emissions should be non-negative
            assert result['isoprene_emission'] >= 0.0
            assert result['monoterpenes_emission'] >= 0.0
            
    def test_light_response(self):
        """Test light response for light-dependent compounds."""
        # Test at different PAR levels
        par_levels = torch.tensor([0.0, 500.0, 1500.0, 2500.0], device=self.device)
        
        emissions = []
        for par in par_levels:
            state = _canonical_megan_state(
                self.device,
                ppfd=float(par.item()),
            )
            result = _unwrap_megan(
                self.model.forward(state, time=0.0, dt=3600.0)
            )
            emissions.append(result['isoprene_emission'].item())
        
        # Isoprene should increase with light (but saturate)
        assert emissions[0] < emissions[1]  # Dark < Low light
        assert emissions[1] < emissions[2]  # Low < High light
        # Check saturation behavior
        assert (emissions[3] - emissions[2]) < (emissions[2] - emissions[1])
        
    def test_co2_inhibition(self):
        """Test CO2 inhibition of isoprene emissions."""
        co2_levels = [280.0, 400.0, 600.0, 800.0]
        
        emissions = []
        for co2 in co2_levels:
            state = _canonical_megan_state(
                self.device,
                atmospheric_co2_ppm=co2,
            )
            result = _unwrap_megan(
                self.model.forward(state, time=0.0, dt=3600.0)
            )
            emissions.append(result['isoprene_emission'].item())
        
        # Isoprene should decrease with increasing CO2
        for i in range(len(emissions) - 1):
            assert emissions[i] >= emissions[i+1]


class TestResponseFunctions:
    """Test individual response function components."""
    
    def test_light_response_function(self):
        """Test BVOCLightResponse calculations."""
        light_response = BVOCLightResponse()
        
        # Test at standard conditions
        par = torch.tensor(1000.0)
        factor = light_response.calculate_factor(par)
        
        assert factor > 0.0
        assert factor <= 2.0  # Should not exceed max from literature
        
        # Test saturation
        high_par = torch.tensor(3000.0)
        high_factor = light_response.calculate_factor(high_par)
        
        # Should approach asymptote
        assert high_factor < 2.5
        
    def test_temperature_response_function(self):
        """Test BVOCTemperatureResponse calculations."""
        temp_response = BVOCTemperatureResponse()
        
        # Test isoprene response (peaked)
        temp = torch.tensor(303.15)  # 30degC
        iso_factor = temp_response.calculate_factor(temp, compound='isoprene')
        
        assert iso_factor > 0.0
        assert iso_factor < 3.0
        
        # Test monoterpene response (exponential)
        mono_factor = temp_response.calculate_factor(temp, compound='monoterpenes')
        
        assert mono_factor > 0.0
        assert mono_factor < 10.0


class TestSynthesisModels:
    """Test synthesis pathway models."""
    
    def test_mep_pathway_model(self):
        """Test MEP pathway precursor synthesis.
        
        Mathematical Expression: Michaelis-Menten kinetics
        References: Lichtenthaler (1999)
        """
        mep_model = MEPPathwayModel()
        
        # Test with substrate concentrations
        g3p = torch.tensor(100.0)
        pyruvate = torch.tensor(50.0)
        atp = torch.tensor(10.0)
        nadph = torch.tensor(8.0)
        temperature = torch.tensor(298.15)
        
        result = mep_model(g3p, pyruvate, atp, nadph, temperature)
        
        assert 'dmapp_flux' in result
        assert result['dmapp_flux'] >= 0.0
        assert 'energy_limitation' in result
        assert 0.0 <= result['energy_limitation'] <= 1.0
        
    def test_isoprene_synthesis(self):
        """Test mechanistic isoprene synthesis."""
        model = IsopreneSynthesis()
        
        state = {
            'air_temperature': torch.tensor(303.15),
            'gross_photosynthesis': torch.tensor(20.0),
            'electron_transport_rate': torch.tensor(150.0),
            'atmospheric_co2_ppm': torch.tensor(280.0),
            'leaf_area_index': torch.tensor(1.0),
        }
        
        result = model.forward(state, time=0.0, dt=3600.0)

        assert "isoprene_emission" in result
        assert "isoprene_emission_mechanistic" not in result
        assert "isoprenes_emission" not in result
        emission = result["isoprene_emission"]
        assert isinstance(emission, UpdateItem)
        assert emission.mode is UpdateMode.REPLACE
        assert emission.per_timestep is False
        assert emission.value >= 0.0
        
    def test_monoterpene_synthesis(self):
        """Test monoterpene synthesis with coupling."""
        model = MonoterpeneSynthesis()
        
        state = {
            'air_temperature': torch.tensor(303.15),
            'gross_photosynthesis': torch.tensor(20.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: model.reset_pools(),
        }
        
        result = model.forward(state, time=0.0, dt=3600.0)
        
        # Check the canonical port and explicitly namespaced diagnostics.
        assert 'monoterpenes_emission' in result
        assert 'monoterpene_emission_mechanistic' not in result
        assert 'diagnostics.bvoc.monoterpene.emission_rate_molar' in result
        assert 'diagnostics.bvoc.monoterpene.emission_rate_mass' in result
        
        # All should be non-negative
        assert result['monoterpenes_emission'].value >= 0.0
        assert result[
            'diagnostics.bvoc.monoterpene.emission_rate_mass'
        ].value >= 0.0
        molar_umol = result[
            'diagnostics.bvoc.monoterpene.emission_rate_molar'
        ].value
        assert molar_umol >= 0.0
        
        # Verify unit conversions if emissions are non-zero
        if molar_umol > 0.0:
            # Check umol -> mol conversion
            expected_mol = molar_umol * 1e-6
            assert torch.allclose(
                result['monoterpenes_emission'].value,
                expected_mol,
                rtol=0.01
            ), "Standard output (mol) should equal molar/1e6"


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def setup_method(self):
        """Setup test environment."""
        self.model = MEGANModel(
            name="test_stability",
            validate_inputs=False,
            emission_pathway=EmissionPathway.STANDARD,
        )
        
    def test_extreme_temperatures(self):
        """Test model stability at extreme temperatures."""
        # Very cold
        cold_state = _canonical_megan_state(
            self.model.device,
            air_temperature=263.15,
        )
        result_cold = _unwrap_megan(
            self.model.forward(cold_state, time=0.0, dt=3600.0)
        )
        assert torch.isfinite(result_cold['total_bvoc_emission'])
        assert result_cold['total_bvoc_emission'] >= 0.0
        
        # Very hot
        hot_state = _canonical_megan_state(
            self.model.device,
            air_temperature=333.15,
        )
        result_hot = _unwrap_megan(
            self.model.forward(hot_state, time=0.0, dt=3600.0)
        )
        assert torch.isfinite(result_hot['total_bvoc_emission'])
        
    def test_zero_inputs(self):
        """Test model behavior with zero inputs."""
        zero_state = _canonical_megan_state(
            self.model.device,
            ppfd=0.0,
            leaf_area_index=0.0,
        )
        result = _unwrap_megan(
            self.model.forward(zero_state, time=0.0, dt=3600.0)
        )
        
        # Should produce zero or near-zero emissions
        assert result['total_bvoc_emission'] == 0.0
        
    def test_device_consistency(self):
        """Test tensor device handling."""
        device = self.model.device
        
        # All model parameters should be on correct device
        assert self.model.isoprene_ef.device.type == device.type
        assert self.model.alpha_light.device.type == device.type
        
        # Test with CPU tensors (should auto-convert)
        cpu_state = _canonical_megan_state(torch.device("cpu"))
        result = _unwrap_megan(
            self.model.forward(cpu_state, time=0.0, dt=3600.0)
        )
        
        # Results should be on configured output device
        assert result['total_bvoc_emission'].device == self.model.device


class TestEmissionFactorDatabase:
    """Test emission factor database functionality."""
    
    def test_database_initialization(self):
        """Test proper database initialization."""
        database = EmissionFactorDatabase()
        
        # Should have default entries
        pfts = database.list_available_pfts()
        assert len(pfts) > 0
        
        compounds = database.list_available_compounds()
        assert len(compounds) >= 4  # At least the main 4 compounds
        
    def test_emission_factor_retrieval(self):
        """Test retrieval with proper default."""
        database = EmissionFactorDatabase()
        
        # Test retrieval for known PFT
        from relic.scientific.biosphere.bvoc.emission_factors import PlantFunctionalType, CompoundCategory
        
        factor = database.get_emission_factor(
            PlantFunctionalType.DEFAULT,
            CompoundCategory.ISOPRENE
        )
        
        assert factor is not None
        assert factor.value > 0.0
        assert factor.units == "ug g-1 DW h-1"


# Fixtures for testing
@pytest.fixture
def standard_bvoc_state():
    """Standard BVOC model state for testing."""
    device = torch.device("cpu")
    return {
        "air_temperature": torch.tensor(303.15, device=device),
        "ppfd": torch.tensor(1000.0, device=device),
        "leaf_area_index": torch.tensor(2.0, device=device),
        "atmospheric_co2_ppm": torch.tensor(400.0, device=device),
        "water_stress_factor": torch.tensor(1.0, device=device),
        "temperature_stress_factor": torch.tensor(1.0, device=device),
        "ozone_stress_factor": torch.tensor(1.0, device=device),
    }


@pytest.fixture
def coupled_bvoc_state(standard_bvoc_state):
    """BVOC state with photosynthesis coupling."""
    device = standard_bvoc_state["air_temperature"].device
    coupled = standard_bvoc_state.copy()
    coupled.update({
        'gross_photosynthesis': torch.tensor(25.0, device=device),
        'net_photosynthesis': torch.tensor(20.0, device=device),
        'electron_transport_rate': torch.tensor(150.0, device=device),
    })
    return coupled


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


