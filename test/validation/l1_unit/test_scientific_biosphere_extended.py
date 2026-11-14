"""
L1 Unit Tests - Scientific Module: Biosphere Extended

Tests additional biosphere scientific modules beyond photosynthesis to ensure:
- Correct implementation of FunctorBase abstractions
- Accurate biological process calculations (respiration, growth, phenology, stress, BVOC)
- Temperature dependencies and environmental responses
- Parameter management and validation
- Conservation laws and mass balance
- Tensor differentiability and device consistency

Covers:
- Respiration modules: AutotrophicRespiration, GrowthRespiration, MaintenanceRespiration
- Stress modules: WaterStress, ThermalStress, NutrientStress, OzoneStress
- Growth modules: BiomassAllocation, StructuralGrowth, PhenologyGrowth
- Phenology modules: PhenologyFunctor and related classes
- BVOC modules: MEGANModel, IsopreneSynthesis, MonoterpeneSynthesis, SesquiterpeneSynthesis
"""

import pytest
import torch
import numpy as np
from typing import Dict

# Import biosphere modules
from relic.scientific.biosphere.respiration.autotrophic_respiration import AutotrophicRespiration
from relic.scientific.biosphere.respiration.growth_respiration import GrowthRespiration
from relic.scientific.biosphere.respiration.maintenance_respiration import MaintenanceRespiration
from relic.scientific.biosphere.stress.water_stress import WaterStress
from relic.scientific.biosphere.stress.thermal_stress import ThermalStress
from relic.scientific.biosphere.stress.nutrient_stress import NutrientStress
from relic.scientific.biosphere.stress.ozone_stress import OzoneStress
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.bvoc.megan_model import EmissionPathway, MEGANModel
from relic.scientific.biosphere.bvoc.synthesis.isoprene import IsopreneSynthesis
from relic.scientific.biosphere.bvoc.synthesis.monoterpene import MonoterpeneSynthesis
from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import SesquiterpeneSynthesis
from relic.core.functor import FunctorBase
from relic.core.state import UpdateItem, UpdateMode


def _canonical_respiration_state(
    **overrides: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build an explicit canonical kg-C respiration test state."""
    state = {
        "air_temperature": torch.tensor(298.15),
        "gross_primary_productivity": torch.tensor(1.0e-6),
        "biomass_leaf": torch.tensor(200.0),
        "biomass_stem": torch.tensor(500.0),
        "biomass_root": torch.tensor(150.0),
        "soluble_sugar": torch.tensor(50.0),
        "soil_water_potential": torch.tensor(-0.5),
        "plant_nitrogen": torch.tensor(2.0),
        "plant_phosphorus": torch.tensor(0.2),
    }
    state.update(overrides)
    return state


def _run_scheduled_respiration(
    state: Dict[str, torch.Tensor],
    environment: Dict[str, torch.Tensor],
) -> Dict[str, UpdateItem]:
    """Execute maintenance, growth, and aggregate Functors in graph order."""
    scheduled_state = {**state, **environment}

    maintenance = MaintenanceRespiration()
    maintenance_output = maintenance.forward(scheduled_state, time=0.0, dt=1.0)
    maintenance_item = maintenance_output["maintenance_respiration_total"]
    assert isinstance(maintenance_item, UpdateItem)
    scheduled_state["maintenance_respiration_total"] = maintenance_item.value

    growth = GrowthRespiration()
    growth_output = growth.forward(scheduled_state, time=0.0, dt=1.0)
    growth_item = growth_output["growth_respiration_total"]
    assert isinstance(growth_item, UpdateItem)
    scheduled_state["growth_respiration_total"] = growth_item.value

    aggregate = AutotrophicRespiration()
    return aggregate.forward(scheduled_state, time=0.0, dt=1.0)


class TestRespirationFunctors:
    """Test suite for respiration modules."""
    
    def test_autotrophic_respiration_inheritance(self):
        """Test AutotrophicRespiration inherits from FunctorBase."""
        respiration = AutotrophicRespiration()
        
        assert isinstance(respiration, FunctorBase), "Not inheriting from FunctorBase"
        assert hasattr(respiration, 'forward'), "Missing forward method"
        assert hasattr(respiration, 'get_dependencies'), "Missing get_dependencies method"
    
    def test_autotrophic_respiration_basic(self):
        """Test basic autotrophic respiration calculations."""
        state = _canonical_respiration_state()
        result = _run_scheduled_respiration({}, state)

        respiration = result['total_autotrophic_respiration']
        npp = result['net_primary_productivity']
        assert isinstance(respiration, UpdateItem)
        assert isinstance(npp, UpdateItem)
        assert respiration.mode is UpdateMode.REPLACE
        assert npp.mode is UpdateMode.REPLACE
        assert respiration.unit == 'kg C m-2 s-1'
        assert npp.unit == 'kg C m-2 s-1'
        assert respiration.value > 0
        torch.testing.assert_close(
            state['gross_primary_productivity'],
            respiration.value + npp.value,
            rtol=0.0,
            atol=1.0e-14,
        )
    
    def test_maintenance_respiration_basic(self):
        """Test basic maintenance respiration calculations."""
        respiration = MaintenanceRespiration()
        result = respiration.forward(
            _canonical_respiration_state(),
            time=0.0,
            dt=1.0,
        )

        item = result['maintenance_respiration_total']
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == 'kg C m-2 s-1'
        assert item.value > 0
    
    def test_growth_respiration_basic(self):
        """Test basic growth respiration calculations."""
        respiration = GrowthRespiration()
        state = _canonical_respiration_state(
            maintenance_respiration_total=torch.tensor(2.0e-7)
        )
        result = respiration.forward(state, time=0.0, dt=1.0)

        item = result['growth_respiration_total']
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == 'kg C m-2 s-1'
        assert item.value >= 0
        assert item.value <= (
            state['gross_primary_productivity']
            - state['maintenance_respiration_total']
        )
    
    def test_respiration_temperature_response(self):
        """Test temperature response of respiration processes."""
        respiration = AutotrophicRespiration()
        
        temperatures = torch.tensor([283.15, 298.15, 313.15])  # 10, 25, 40 degC
        rates = []
        
        for temp in temperatures:
            environment = _canonical_respiration_state(air_temperature=temp)
            
            result = _run_scheduled_respiration({}, environment)
            rate = result['total_autotrophic_respiration'].value
            rates.append(rate.item())
        
        rates = np.array(rates)
        
        # Should increase with temperature (Q10 response)
        assert rates[1] > rates[0], "Respiration should increase from 10 degC to 25 degC"
        assert rates[2] > rates[1], "Respiration should increase from 25 degC to 40 degC"


class TestStressFunctors:
    """Test suite for plant stress modules."""
    
    def test_water_stress_inheritance(self):
        """Test WaterStress inherits from FunctorBase."""
        stress = WaterStress()
        
        assert isinstance(stress, FunctorBase), "Not inheriting from FunctorBase"
        assert hasattr(stress, 'forward'), "Missing forward method"
        assert hasattr(stress, 'get_dependencies'), "Missing get_dependencies method"
    
    def test_water_stress_basic(self):
        """Test basic water stress calculations."""
        stress = WaterStress()
        
        # Test different water potentials
        water_potentials = torch.tensor([-0.1, -0.5, -1.0, -2.0])  # MPa
        stress_factors = []
        
        for wp in water_potentials:
            state = {
                'soil_water_potential': wp
            }

            result = stress.forward(state, time=43200.0, dt=3600.0)

            factor = result['water_stress_factor']
            assert isinstance(factor, UpdateItem)
            stress_factors.append(factor.value.item())

        stress_factors = np.array(stress_factors)

        # Stress factor should decrease with more negative water potential.
        assert stress_factors[0] > stress_factors[-1], "Water stress should increase with more negative potential"

        # Stress factor is a bounded instantaneous diagnostic.
        assert np.all(stress_factors >= 0), "Stress factors should be non-negative"
        assert np.all(stress_factors <= 1), "Stress factors should not exceed 1"
    
    def test_thermal_stress_basic(self):
        """Test basic thermal stress calculations."""
        stress = ThermalStress()
        
        # Test different temperatures
        temperatures = torch.tensor([278.15, 298.15, 318.15, 338.15])  # 5 deg C to 65 deg C
        stress_factors = []
        
        for temp in temperatures:
            state = {
                'air_temperature': temp,
                'leaf_temperature': temp + 2.0  # Leaf slightly warmer
            }
            environment = {
                'time_of_day': torch.tensor(12.0)  # Noon
            }
            
            result = stress.forward(state, time=43200.0, dt=3600.0)
            
            # Extract stress factor
            if isinstance(result, dict):
                factor = result.get('thermal_stress_factor', result.get('stress_factor'))
            else:
                factor = result
            
            if factor is not None:
                stress_factors.append(factor.item() if isinstance(factor, torch.Tensor) else factor)
        
        if stress_factors:
            stress_factors = np.array(stress_factors)
            
            # Should have optimal range around 25 deg C
            optimal_idx = 1  # 298.15 K (25 deg C)
            assert stress_factors[optimal_idx] > stress_factors[0], "Should be higher at optimal temperature"
            assert stress_factors[optimal_idx] > stress_factors[-1], "Should decrease at very high temperatures"
    
    def test_nutrient_stress_basic(self):
        """Test basic nutrient stress calculations."""
        stress = NutrientStress()
        
        # Test tissue nitrogen mass fractions with non-limiting phosphorus.
        nitrogen_levels = torch.tensor([0.005, 0.0125, 0.020, 0.025])
        stress_factors = []
        
        for n_level in nitrogen_levels:
            state = {
                'nitrogen_concentration': n_level,
                'phosphorus_concentration': torch.tensor(0.003),
            }
            
            result = stress.forward(state, time=43200.0, dt=3600.0)

            assert set(result) == set(stress.metadata.output_variables)
            factor = result['nutrient_stress_factor']
            assert isinstance(factor, UpdateItem)
            stress_factors.append(factor.value.item())

        assert len(stress_factors) == len(nitrogen_levels)
        stress_factors_array = np.array(stress_factors)

        # The control factor must increase monotonically with tissue nitrogen.
        assert stress_factors_array[-1] > stress_factors_array[0]
        assert np.all(np.diff(stress_factors_array) >= 0.0)
    
    def test_ozone_stress_basic(self):
        """Test basic ozone stress calculations."""
        stress = OzoneStress()
        
        # Test different ozone concentrations
        o3_concentrations = torch.tensor([30e-9, 50e-9, 80e-9, 120e-9])  # ppb
        stress_factors = []
        
        for o3 in o3_concentrations:
            state = {
                'o3_mixing_ratio': o3,
                'stomatal_conductance': torch.tensor(0.2)  # mol m-2 s-1
            }
            environment = {
                'temperature': torch.tensor(298.15),
                'vpd': torch.tensor(1.0)
            }
            
            result = stress.forward(state, time=43200.0, dt=3600.0)
            
            # Extract stress factor
            if isinstance(result, dict):
                factor = result['ozone_stress_factor'].value
            else:
                factor = result
            
            if factor is not None:
                stress_factors.append(factor.item() if isinstance(factor, torch.Tensor) else factor)
        
        if stress_factors:
            stress_factors = np.array(stress_factors)
            
            # Should decrease with higher ozone (more stress)
            assert stress_factors[0] >= stress_factors[-1], "Stress should increase with ozone"


class TestGrowthFunctors:
    """Test suite for plant growth modules."""
    
    def test_biomass_allocation_inheritance(self):
        """Test BiomassAllocation inherits from FunctorBase."""
        growth = BiomassAllocation()
        
        assert isinstance(growth, FunctorBase), "Not inheriting from FunctorBase"
        assert hasattr(growth, 'forward'), "Missing forward method"
        assert hasattr(growth, 'get_dependencies'), "Missing get_dependencies method"
    
    def test_biomass_allocation_basic(self):
        """Test basic biomass allocation calculations."""
        growth = BiomassAllocation()
        dt = 3600.0
        state = {
            'net_primary_productivity': torch.tensor(1.0e-7),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(0.9),
            'leaf_area_index': torch.tensor(4.0),
            'phenology_phase': torch.tensor(2.0),
        }

        result = growth.forward(state, time=43200.0, dt=dt)

        assert set(result) == set(growth.metadata.output_variables)
        assert all(isinstance(value, UpdateItem) for value in result.values())

        rate_names = ('growth_rate_leaf', 'growth_rate_stem', 'growth_rate_root')
        rate_sum = sum((result[name].value for name in rate_names), torch.zeros(()))
        torch.testing.assert_close(rate_sum, state['net_primary_productivity'])

        pool_names = ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c')
        delta_sum = sum((result[name].value for name in pool_names), torch.zeros(()))
        torch.testing.assert_close(delta_sum, state['net_primary_productivity'] * dt)

        fractions = [
            result['diagnostics.biomass_allocation.allocation_fraction_leaf'].value,
            result['diagnostics.biomass_allocation.allocation_fraction_stem'].value,
            result['diagnostics.biomass_allocation.allocation_fraction_root'].value,
        ]
        torch.testing.assert_close(sum(fractions, torch.zeros(())), torch.ones(()))
    
    def test_structural_growth_basic(self):
        """Test basic structural growth calculations."""
        growth = StructuralGrowth()
        
        state = {
            'leaf_area_index': torch.tensor(3.0),
            'growth_rate_leaf': torch.tensor(2.0e-7),
        }
        dt = 3600.0

        result = growth.forward(state, time=43200.0, dt=dt)

        assert set(result) == set(growth.metadata.output_variables)
        assert all(isinstance(value, UpdateItem) for value in result.values())
        assert all(torch.isfinite(value.value) for value in result.values())
        gross = result[
            'diagnostics.structural_growth.gross_leaf_area_growth_tendency'
        ]
        loss = result[
            'diagnostics.structural_growth.leaf_area_senescence_tendency'
        ]
        net = result['diagnostics.structural_growth.net_leaf_area_tendency']
        step = result['diagnostics.structural_growth.step_leaf_area_delta']
        torch.testing.assert_close(net.value, gross.value - loss.value)
        torch.testing.assert_close(step.value, net.value * dt)
    
    def test_phenology_growth_basic(self):
        """Test basic phenology growth calculations."""
        growth = PhenologyGrowth(phenology_model_owner='gsi_phenology')

        state = {
            'phenological_factor': torch.tensor(0.6),
        }

        result = growth.forward(state, time=43200.0, dt=3600.0)

        assert set(result) == set(growth.metadata.output_variables)
        assert 'rate' not in result
        assert 'leaf_area_index' not in result
        assert all(isinstance(item, UpdateItem) for item in result.values())
        assert all(item.mode is UpdateMode.REPLACE for item in result.values())
        assert all(item.per_timestep is False for item in result.values())
        assert all(item.unit == 'dimensionless' for item in result.values())
        assert result['diagnostics.phenology.growth_active'].value.dtype is torch.bool
        assert all(torch.isfinite(item.value).all() for item in result.values())


class TestBVOCFunctors:
    """Test suite for BVOC emission modules."""

    def test_megan_model_inheritance(self):
        megan = MEGANModel()
        assert isinstance(megan, FunctorBase)
        assert hasattr(megan, "forward")
        assert hasattr(megan, "get_dependencies")

    def test_megan_model_basic(self):
        megan = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        state = {
            "leaf_area_index": torch.tensor(4.0),
            "air_temperature": torch.tensor(298.15),
            "ppfd": torch.tensor(1500.0),
            "atmospheric_co2_ppm": torch.tensor(400.0),
        }
        result = megan.forward(state, time=43200.0, dt=3600.0)
        assert isinstance(result, dict)
        assert set(result) == set(megan.metadata.output_variables)
        for key in [
            "isoprene_emission",
            "monoterpenes_emission",
            "sesquiterpenes_emission",
            "ovocs_emission",
            "total_bvoc_emission",
        ]:
            emission = result[key]
            assert isinstance(emission, UpdateItem)
            assert emission.mode is UpdateMode.REPLACE
            assert emission.per_timestep is False
            assert emission.unit == "mol m-2 s-1"
            assert emission.value >= 0

    def test_isoprene_synthesis_basic(self):
        isoprene = IsopreneSynthesis()
        state = {
            "air_temperature": torch.tensor(303.15),
            "gross_photosynthesis": torch.tensor(20.0),
            "electron_transport_rate": torch.tensor(150.0),
            "atmospheric_co2_ppm": torch.tensor(250.0),
            "leaf_area_index": torch.tensor(1.0),
        }
        result = isoprene.forward(state, time=43200.0, dt=3600.0)
        emission = result["isoprene_emission"]
        assert isinstance(emission, UpdateItem)
        assert emission.mode is UpdateMode.REPLACE
        assert emission.per_timestep is False
        assert emission.unit == "mol m-2 s-1"
        assert emission.value >= 0
        assert "isoprenes_emission" not in result
        assert "isoprene_emission_mechanistic" not in result

    def test_monoterpene_synthesis_basic(self):
        monoterpene = MonoterpeneSynthesis()
        state = {
            "air_temperature": torch.tensor(298.15),
            "gross_photosynthesis": torch.tensor(20.0),
            "monoterpene_pool_concentration": torch.tensor(50.0),
        }
        result = monoterpene.forward(state, time=43200.0, dt=3600.0)
        emission = result["monoterpenes_emission"].value
        assert result["monoterpenes_emission"].unit == "mol m-2 s-1"
        if emission is not None:
            assert isinstance(emission, torch.Tensor)
            assert emission >= 0

    def test_monoterpene_species_basic(self):
        monoterpene = MonoterpeneSynthesis()
        state = {
            "air_temperature": torch.tensor(295.15),
            "gross_photosynthesis": torch.tensor(15.0),
            "monoterpene_pool_concentration": torch.tensor(25.0),
        }
        result = monoterpene.forward(state, time=43200.0, dt=3600.0)
        emission = result["monoterpenes_emission"].value
        assert result["monoterpenes_emission"].unit == "mol m-2 s-1"
        if emission is not None:
            assert isinstance(emission, torch.Tensor)
            assert emission >= 0

    def test_sesquiterpene_synthesis_basic(self):
        sesqui = SesquiterpeneSynthesis()
        state = {
            "temperature": torch.tensor(300.15),
            "ppfd": torch.tensor(900.0),
            "leaf_area_index": torch.tensor(1.0),
        }
        result = sesqui.forward(state, time=43200.0, dt=3600.0)
        assert set(result) == set(sesqui.metadata.output_variables)
        assert "sesquiterpene_emission_mechanistic" not in result
        emission = result["sesquiterpenes_emission"]
        assert isinstance(emission, UpdateItem)
        assert emission.mode is UpdateMode.REPLACE
        assert emission.per_timestep is False
        assert emission.unit == "mol m-2 s-1"
        assert emission.value >= 0

    def test_bvoc_temperature_dependence(self):
        isoprene = IsopreneSynthesis()
        temperatures = torch.tensor([288.15, 298.15, 308.15, 318.15])
        emissions = []
        for temp in temperatures:
            state = {
                "air_temperature": temp,
                "gross_photosynthesis": torch.tensor(20.0),
                "electron_transport_rate": torch.tensor(120.0),
                "atmospheric_co2_ppm": torch.tensor(250.0),
                "leaf_area_index": torch.tensor(1.0),
            }
            result = isoprene.forward(state, time=43200.0, dt=3600.0)
            emission = result["isoprene_emission"]
            assert isinstance(emission, UpdateItem)
            emissions.append(emission.value.item())
        if len(emissions) >= 3:
            emissions = np.array(emissions)
            assert emissions[1] > emissions[0]
            assert emissions[2] >= emissions[1]

def test_biosphere_integration():
    """Test integration between different biosphere modules."""
    # Test photosynthesis -> respiration coupling
    from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
    
    photosynthesis = FarquharPhotosynthesis()
    stress = WaterStress()
    
    # Standard conditions
    state = {
        'soil_water_potential': torch.tensor(-0.8)  # MPa
    }
    environment = {
        'air_temperature': torch.tensor(298.15),
        'atmospheric_co2_ppm': torch.tensor(400.0),
        'ppfd': torch.tensor(1500.0),
        'leaf_area_index': torch.tensor(1.0),
        'vapor_pressure_deficit': torch.tensor(1500.0),
        'vcmax_scaling': torch.tensor(1.0),
    }
    
    # Calculate photosynthesis
    photo_result = photosynthesis.forward(
        {**state, **environment},
        time=0.0,
        dt=3600.0,
    )
    gpp_item = photo_result['gross_primary_productivity']
    assert isinstance(gpp_item, UpdateItem)
    gpp = gpp_item.value
    
    # Calculate water stress
    stress_result = stress.forward(state, time=43200.0, dt=3600.0)
    water_stress_item = stress_result['water_stress_factor']
    assert isinstance(water_stress_item, UpdateItem)
    water_stress = water_stress_item.value
    
    # Calculate respiration using GPP
    resp_env = _canonical_respiration_state(
        air_temperature=environment['air_temperature'],
        gross_primary_productivity=gpp,
    )
    
    resp_result = _run_scheduled_respiration({}, resp_env)
    resp_rate = resp_result['total_autotrophic_respiration'].value
    npp = resp_result['net_primary_productivity'].value
    
    # Check coupling makes sense
    assert gpp > 0, "Photosynthesis should be positive"
    assert resp_rate > 0, "Respiration should be positive"
    torch.testing.assert_close(gpp, resp_rate + npp, rtol=0.0, atol=1.0e-14)
    
    # Water stress should be between 0 and 1
    if isinstance(water_stress, torch.Tensor):
        assert 0 <= water_stress <= 1, f"Water stress factor should be 0-1: {water_stress}"


def test_biosphere_batch_processing():
    """Test biosphere modules with batch processing."""
    batch_size = 4
    
    # Create batch data
    state_batch = {
        'leaf_temperature': torch.randn(batch_size) * 3 + 298.15,        # 25 +/- 3 degC
        'air_temperature': torch.randn(batch_size) * 3 + 296.15,         # 23 +/- 3 degC
        'soil_water_potential': torch.rand(batch_size) * -1.5 - 0.2,     # -0.2 to -1.7 MPa
        'leaf_biomass': torch.rand(batch_size) * 200 + 100,              # 100-300 g C m-2
        'leaf_area_index': torch.rand(batch_size) * 4 + 1                # 1-5 m2 m-2
    }
    
    environment_batch = {
        'temperature': state_batch['air_temperature'],
        'air_temperature': state_batch['air_temperature'],
        'photosynthetically_active_radiation': torch.rand(batch_size) * 1000 + 500,  # 500-1500 umol m-2 s-1
        'vpd': torch.rand(batch_size) * 2 + 0.5,                         # 0.5-2.5 kPa
        'co2_concentration': torch.ones(batch_size) * 400e-6,            # 400 ppm
        'leaf_nitrogen': torch.rand(batch_size) * 2 + 1.5,               # 1.5-3.5% N
        'gross_primary_productivity': torch.rand(batch_size) * 1.0e-7 + 5.0e-8,
        'biomass_leaf': state_batch['leaf_biomass'],
        'biomass_stem': torch.rand(batch_size) * 500 + 300,
        'biomass_root': torch.rand(batch_size) * 200 + 100,
        'soluble_sugar': torch.rand(batch_size) * 50 + 10,
        'soil_water_potential': state_batch['soil_water_potential'],
        'plant_nitrogen': torch.rand(batch_size) * 2 + 1.5,
        'plant_phosphorus': torch.rand(batch_size) * 0.2 + 0.1,
    }
    
    # Test water stress with batch
    stress = WaterStress()
    stress_result = stress.forward(state_batch, time=43200.0, dt=3600.0)
    
    stress_item = stress_result['water_stress_factor']
    assert isinstance(stress_item, UpdateItem)
    stress_factor = stress_item.value
    assert stress_factor.shape == (batch_size,), f"Batch shape mismatch: {stress_factor.shape}"
    assert torch.isfinite(stress_factor).all(), "Stress factors should be finite"
    assert (stress_factor >= 0).all() and (stress_factor <= 1).all(), "Stress factors should be 0-1"
    
    # Test respiration with batch
    resp_result = _run_scheduled_respiration({}, environment_batch)
    
    resp_rate = resp_result['total_autotrophic_respiration'].value
    npp = resp_result['net_primary_productivity'].value
    assert resp_rate.shape == (batch_size,), f"Batch shape mismatch: {resp_rate.shape}"
    assert torch.isfinite(resp_rate).all(), "Respiration rates should be finite"
    assert (resp_rate >= 0).all(), "Respiration rates should be non-negative"
    torch.testing.assert_close(
        environment_batch['gross_primary_productivity'],
        resp_rate + npp,
        rtol=0.0,
        atol=1.0e-14,
    )
    
    # Test BVOC emission with batch.
    megan = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
    bvoc_state = {**state_batch, **environment_batch}
    bvoc_state['ppfd'] = environment_batch['photosynthetically_active_radiation']
    bvoc_state['atmospheric_co2_ppm'] = torch.full((batch_size,), 400.0)
    bvoc_result = megan.forward(bvoc_state, time=43200.0, dt=3600.0)

    assert isinstance(bvoc_result, dict)
    emission_outputs = [
        (key, value)
        for key, value in bvoc_result.items()
        if isinstance(value, UpdateItem) and key.endswith('_emission')
    ]
    assert emission_outputs, "MEGAN batch output should include emission tensors"
    for key, item in emission_outputs:
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == 'mol m-2 s-1'
        value = item.value
        assert value.shape == (batch_size,), f"BVOC batch shape mismatch for {key}: {value.shape}"
        assert torch.isfinite(value).all(), f"BVOC {key} should be finite"
        assert (value >= 0).all(), f"BVOC {key} should be non-negative"


def test_conservation_laws():
    """Test conservation laws in biosphere processes."""
    # Test carbon conservation in respiration
    # Set up carbon balance
    gpp = torch.tensor(1.0e-6)
    environment = _canonical_respiration_state(
        gross_primary_productivity=gpp
    )
    
    result = _run_scheduled_respiration({}, environment)
    resp_rate = result['total_autotrophic_respiration'].value
    npp = result['net_primary_productivity'].value
    torch.testing.assert_close(gpp, resp_rate + npp, rtol=0.0, atol=1.0e-14)
    
    # Test biomass allocation conservation
    allocation = BiomassAllocation()
    dt = 3600.0
    state = {
        'net_primary_productivity': npp,
        'air_temperature': torch.tensor(298.15),
        'water_stress_factor': torch.tensor(1.0),
        'leaf_area_index': torch.tensor(3.0),
        'phenology_phase': torch.tensor(2.0),
    }

    result = allocation.forward(state, time=43200.0, dt=dt)
    fractions = [
        result['diagnostics.biomass_allocation.allocation_fraction_leaf'].value,
        result['diagnostics.biomass_allocation.allocation_fraction_stem'].value,
        result['diagnostics.biomass_allocation.allocation_fraction_root'].value,
    ]
    torch.testing.assert_close(sum(fractions, torch.zeros(())), torch.ones(()))

    rate_sum = sum(
        (result[name].value for name in ('growth_rate_leaf', 'growth_rate_stem', 'growth_rate_root')),
        torch.zeros(()),
    )
    delta_sum = sum(
        (result[name].value for name in ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c')),
        torch.zeros(()),
    )
    torch.testing.assert_close(rate_sum, npp)
    torch.testing.assert_close(delta_sum, npp * dt)
    residual = result['diagnostics.biomass_allocation.carbon_balance_residual'].value
    torch.testing.assert_close(residual, torch.zeros_like(residual))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
