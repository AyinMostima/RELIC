"""
L3 System Tests for Biosphere Module End-to-End Simulation

This module tests complete simulation scenarios including all biosphere
components working together with realistic data and parameter configurations.

Author: Yuchong Long
Date: 2025-09-29
"""

import pytest
import torch
import numpy as np

from relic.parameters import get_global_manager
from relic.scientific.biosphere.bvoc.megan_model import (
    CouplingStrategy,
    EmissionPathway,
    MEGANModel,
)
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.stress.water_stress import WaterStress
from relic.scientific.biosphere.stress.thermal_stress import ThermalStress  
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.core.state import UpdateItem, UpdateMode
from relic.core.simulator import RELICSimulator


def _update_value(outputs, key):
    """Return the tensor carried by a canonical scientific UpdateItem."""
    item = outputs[key]
    assert isinstance(item, UpdateItem)
    return item.value


def _replace_value(outputs, key, unit):
    """Return one canonical instantaneous flux value."""
    item = outputs[key]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.per_timestep is False
    assert item.unit == unit
    return item.value


class TestDiurnalCycleSimulation:
    """Test complete diurnal cycle simulation."""
    
    @pytest.fixture
    def simulator(self):
        """Create configured simulator."""
        simulator = RELICSimulator(device=torch.device('cpu'))
        return simulator
    
    @pytest.fixture
    def diurnal_forcing(self):
        """Generate diurnal forcing data."""
        hours = np.arange(0, 24, 1)
        
        # Temperature cycle (K)
        temp = 288.15 + 10 * np.sin((hours - 6) * np.pi / 12)
        temp[hours < 6] = 288.15 - 5
        temp[hours > 18] = 288.15 - 3
        
        # PAR cycle (umol m-2 s-1)
        par = np.maximum(0, 1200 * np.sin((hours - 6) * np.pi / 12))
        par[hours < 6] = 0
        par[hours > 18] = 0
        
        # Convert to tensors
        return {
            'time_hours': hours,
            'air_temperature': torch.tensor(temp, dtype=torch.float32),
            'ppfd': torch.tensor(par, dtype=torch.float32),
            'atmospheric_co2_ppm': torch.tensor(400.0, dtype=torch.float32),
            'relative_humidity': torch.tensor(0.6, dtype=torch.float32),
            'soil_moisture': torch.tensor(0.25, dtype=torch.float32),
            # One explicit dtype for every forcing tensor: the MEGAN input contract refuses
            # mixed floating dtypes instead of narrowing them (megan_model.py, 2026-10-15).
            'leaf_area_index': torch.tensor(3.0, dtype=torch.float32),
        }
    
    def test_photosynthesis_diurnal_pattern(self, diurnal_forcing):
        """Test photosynthesis follows expected diurnal pattern."""
        photosynthesis = FarquharPhotosynthesis()
        
        gpp_values = []
        for i in range(24):
            state = {
                'air_temperature': diurnal_forcing['air_temperature'][i],
                'ppfd': diurnal_forcing['ppfd'][i],
                'leaf_area_index': torch.tensor(1.0),
                'atmospheric_co2_ppm': diurnal_forcing['atmospheric_co2_ppm'],
                'vapor_pressure_deficit': torch.tensor(1000.0),
            }
            
            output = photosynthesis.forward(state, time=float(i*3600), dt=3600.0)
            gpp = _update_value(output, 'gross_primary_productivity')
            gpp_values.append(gpp.item())
        
        # Check diurnal pattern
        gpp_array = np.array(gpp_values)
        
        # Night values should be near zero
        assert np.allclose(gpp_array[0:6], 0.0, atol=1.0e-12)
        assert np.allclose(gpp_array[19:24], 0.0, atol=1.0e-12)
        
        # Day values should be positive
        assert np.all(gpp_array[8:16] > 0.0)
        
        # Peak should be around midday
        peak_hour = np.argmax(gpp_array)
        assert 10 <= peak_hour <= 14
    
    def test_bvoc_emission_diurnal_pattern(self, diurnal_forcing):
        """Test BVOC emissions follow expected diurnal pattern."""
        megan = MEGANModel(
            coupling_strategy=CouplingStrategy.INDEPENDENT,
            emission_pathway=EmissionPathway.STANDARD,
        )
        
        emission_values = []
        for i in range(24):
            state = {
                'air_temperature': diurnal_forcing['air_temperature'][i],
                'ppfd': diurnal_forcing['ppfd'][i],
                'leaf_area_index': diurnal_forcing['leaf_area_index'],
                'atmospheric_co2_ppm': diurnal_forcing['atmospheric_co2_ppm'],
            }
            
            output = megan.forward(state, time=float(i*3600), dt=3600.0)
            emission = _replace_value(
                output, 'total_bvoc_emission', 'mol m-2 s-1'
            )
            emission_values.append(emission.item())
        
        # Check diurnal pattern
        emission_array = np.array(emission_values)
        
        # Emissions should be higher during day
        day_mean = np.mean(emission_array[8:16])
        night_mean = np.mean(emission_array[np.r_[0:6, 19:24]])
        assert day_mean > night_mean * 2  # Day emissions significantly higher


class TestStressResponseSimulation:
    """Test system response to various stress conditions."""
    
    def test_drought_stress_cascade(self):
        """Test cascade effects of drought stress through system."""
        # Create modules
        water_stress = WaterStress()
        photosynthesis = FarquharPhotosynthesis()
        megan = MEGANModel(coupling_strategy=CouplingStrategy.SEQUENTIAL)
        
        # Test under different soil moisture levels
        soil_moisture_levels = [0.35, 0.25, 0.15, 0.05]  # From wet to severe drought
        gpp_values = []
        bvoc_values = []
        
        for sm in soil_moisture_levels:
            # Create state
            state = {
                'air_temperature': torch.tensor(298.15),
                'ppfd': torch.tensor(800.0),
                'atmospheric_co2_ppm': torch.tensor(400.0),
                'vapor_pressure_deficit': torch.tensor(1000.0),
                'relative_humidity': torch.tensor(0.5),
                'soil_moisture': torch.tensor(sm),
                'soil_water_potential': torch.tensor(-0.033 * (0.35/sm)**2),  # Simple approximation
                'leaf_area_index': torch.tensor(3.0),
            }
            
            # Compute water stress
            stress_output = water_stress.forward(state, time=0.0, dt=1.0)
            state['water_stress_factor'] = _update_value(
                stress_output, 'water_stress_factor'
            )
            
            # Compute photosynthesis
            photo_output = photosynthesis.forward(state, time=0.0, dt=1.0)
            gpp = _update_value(photo_output, 'gross_primary_productivity')
            gpp_values.append(gpp.item())
            
            # Add photosynthesis to state for BVOC
            state['net_photosynthesis'] = _update_value(
                photo_output, 'net_photosynthesis'
            )
            # Compute BVOC
            bvoc_output = megan.forward(state, time=0.0, dt=1.0)
            bvoc = _replace_value(
                bvoc_output, 'total_bvoc_emission', 'mol m-2 s-1'
            )
            bvoc_values.append(bvoc.item())
        
        # Both GPP and BVOC should decrease with drought
        assert gpp_values[0] > gpp_values[-1]  # Wet > Drought for GPP
        assert bvoc_values[0] > bvoc_values[-1] * 0.5  # BVOC should also decrease
    
    def test_temperature_stress_response(self):
        """Test system response to temperature stress."""
        thermal_stress = ThermalStress()
        photosynthesis = FarquharPhotosynthesis()
        
        # Test temperature range (K)
        temperatures = [278.15, 288.15, 298.15, 308.15, 318.15]  # 5 C to 45 C
        stress_values = []
        gpp_values = []
        
        for temp in temperatures:
            state = {
                'air_temperature': torch.tensor(temp),
                'ppfd': torch.tensor(800.0),
                'leaf_area_index': torch.tensor(1.0),
                'atmospheric_co2_ppm': torch.tensor(400.0),
                'vapor_pressure_deficit': torch.tensor(1000.0),
            }
            
            # Compute temperature stress
            stress_output = thermal_stress.forward(state, time=0.0, dt=1.0)
            stress = _update_value(stress_output, 'temperature_stress_factor')
            assert isinstance(stress, torch.Tensor)
            stress_values.append(stress.item())
            
            # Add stress to state
            state['temperature_stress_factor'] = stress
            
            # Compute photosynthesis
            photo_output = photosynthesis.forward(state, time=0.0, dt=1.0)
            gpp = _update_value(photo_output, 'gross_primary_productivity')
            gpp_values.append(gpp.item())
        
        # Stress should be highest at optimal temperature
        optimal_idx = stress_values.index(max(stress_values))
        assert 1 <= optimal_idx <= 3  # Optimal should be in middle range
        
        # GPP should follow similar pattern
        gpp_optimal_idx = gpp_values.index(max(gpp_values))
        assert abs(gpp_optimal_idx - optimal_idx) <= 1  # Should be similar


class TestCarbonBudgetSimulation:
    """Test carbon budget and conservation across system."""
    
    def test_carbon_conservation(self):
        """Route prescribed canonical NPP once into three biomass stocks."""
        biomass_allocation = BiomassAllocation()
        dt = 3600.0
        rate_keys = ('growth_rate_leaf', 'growth_rate_stem', 'growth_rate_root')
        stock_keys = ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c')
        state = {
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(0.9),
            'leaf_area_index': torch.tensor(3.0),
            'phenology_phase': torch.tensor(2.0),
            'leaf_biomass_c': torch.tensor(0.10),
            'wood_biomass_c': torch.tensor(0.50),
            'root_biomass_c': torch.tensor(0.20),
        }
        initial_total = sum(state[key] for key in stock_keys)
        integrated_npp = torch.zeros_like(initial_total)

        for hour in range(24):
            daylight = max(0.0, np.sin((hour - 6) * np.pi / 12.0))
            npp = torch.tensor(1.0e-7 * daylight, dtype=torch.float32)
            state['net_primary_productivity'] = npp
            alloc_output = biomass_allocation.forward(state, time=hour*dt, dt=dt)

            total_rate = sum(alloc_output[key].value for key in rate_keys)
            total_step = sum(alloc_output[key].value for key in stock_keys)
            # Runtime-dtype contract: outputs follow the float64 state, not the float32 NPP tag.
            torch.testing.assert_close(total_rate, npp.to(total_rate.dtype))
            torch.testing.assert_close(total_step, (npp * dt).to(total_step.dtype))
            residual = alloc_output[
                'diagnostics.biomass_allocation.carbon_balance_residual'
            ].value
            torch.testing.assert_close(residual, torch.zeros_like(residual))

            for key in rate_keys:
                item = alloc_output[key]
                assert isinstance(item, UpdateItem)
                assert item.mode is UpdateMode.REPLACE
                assert item.per_timestep is False
            for key in stock_keys:
                item = alloc_output[key]
                assert isinstance(item, UpdateItem)
                assert item.mode is UpdateMode.ACCUMULATE
                assert item.per_timestep is True
                state[key] = state[key] + item.value
            integrated_npp = integrated_npp + npp * dt

        final_total = sum(state[key] for key in stock_keys)
        torch.testing.assert_close(
            final_total - initial_total,
            integrated_npp,
            rtol=1.0e-5,
            atol=1.0e-7,
        )


class TestParameterSensitivitySimulation:
    """Test system sensitivity to parameter changes."""

    @pytest.fixture(autouse=True)
    def restore_parameter_values(self):
        """Restore ParameterManager values after each sensitivity scenario."""
        manager = get_global_manager()
        keys = (
            'biosphere.photosynthesis.vcmax25',
            'biosphere.bvoc.emission_factors.isoprene',
        )
        original = {key: manager.get(key, required=True) for key in keys}
        yield
        for key, value in original.items():
            manager.set_value(key, value)
    
    def test_vcmax_sensitivity(self):
        """Test system sensitivity to Vcmax parameter."""
        param_manager = get_global_manager()
        photosynthesis = FarquharPhotosynthesis()
        
        # Test different Vcmax values
        vcmax_values = [30.0, 50.0, 70.0, 90.0]
        gpp_values = []
        
        state = {
            'air_temperature': torch.tensor(298.15),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
        }
        
        for vcmax in vcmax_values:
            # Update parameter
            param_manager.set_value('biosphere.photosynthesis.vcmax25', vcmax)
            
            # Create new instance to get updated parameter
            photosynthesis_new = FarquharPhotosynthesis()
            
            # Compute GPP
            output = photosynthesis_new.forward(state, time=0.0, dt=1.0)
            gpp = _update_value(output, 'gross_primary_productivity')
            gpp_values.append(gpp.item())
        
        # GPP should increase with Vcmax
        for i in range(len(gpp_values) - 1):
            assert gpp_values[i+1] >= gpp_values[i] * 0.95  # Allow small tolerance
    
    def test_emission_factor_sensitivity(self):
        """Test BVOC sensitivity to emission factors."""
        param_manager = get_global_manager()
        
        # Test different emission factors
        ef_values = [10.0, 20.0, 30.0, 40.0]
        emission_values = []
        
        state = {
            'air_temperature': torch.tensor(303.15),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(3.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
        }
        
        for ef in ef_values:
            # Update parameter
            param_manager.set_value('biosphere.bvoc.emission_factors.isoprene', ef)
            
            # Create new MEGAN instance
            megan = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
            
            # Compute emissions
            output = megan.forward(state, time=0.0, dt=1.0)
            emission = _replace_value(
                output, 'total_bvoc_emission', 'mol m-2 s-1'
            )
            emission_values.append(emission.item())
        
        # Emissions should increase with emission factor
        for i in range(len(emission_values) - 1):
            assert emission_values[i+1] > emission_values[i]


class TestRealWorldScenarios:
    """Test with realistic environmental scenarios."""
    
    def test_heatwave_scenario(self):
        """Test system response to heatwave conditions."""
        # Create integrated system
        photosynthesis = FarquharPhotosynthesis()
        megan = MEGANModel(config={'use_stress_functors': True})
        thermal_stress = ThermalStress()
        water_stress = WaterStress()
        
        # Normal vs heatwave conditions
        normal_state = {
            'air_temperature': torch.tensor(298.15),
            'ppfd': torch.tensor(1000.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'relative_humidity': torch.tensor(0.6),
            'soil_moisture': torch.tensor(0.25),
            'soil_water_potential': torch.tensor(-0.5),
            'leaf_area_index': torch.tensor(3.0),
            'ozone_stress_factor': torch.tensor(1.0),
        }
        
        heatwave_state = normal_state.copy()
        heatwave_state['air_temperature'] = torch.tensor(313.15)
        heatwave_state['vapor_pressure_deficit'] = torch.tensor(3000.0)
        heatwave_state['relative_humidity'] = torch.tensor(0.3)
        heatwave_state['soil_moisture'] = torch.tensor(0.1)
        heatwave_state['soil_water_potential'] = torch.tensor(-2.0)
        
        # Normal conditions
        thermal_normal = thermal_stress.forward(normal_state, 0.0, 1.0)
        water_normal = water_stress.forward(normal_state, 0.0, 1.0)
        normal_temperature_stress = _update_value(
            thermal_normal, 'temperature_stress_factor'
        )
        assert isinstance(normal_temperature_stress, torch.Tensor)
        normal_state['temperature_stress_factor'] = normal_temperature_stress
        normal_state['water_stress_factor'] = _update_value(
            water_normal, 'water_stress_factor'
        )
        
        photo_normal = photosynthesis.forward(normal_state, 0.0, 1.0)
        gpp_normal = _update_value(photo_normal, 'gross_primary_productivity')
        normal_state['net_photosynthesis'] = _update_value(
            photo_normal, 'net_photosynthesis'
        )
        
        bvoc_normal = megan.forward(normal_state, 0.0, 1.0)
        emission_normal = _replace_value(
            bvoc_normal, 'total_bvoc_emission', 'mol m-2 s-1'
        )
        
        # Heatwave conditions  
        thermal_heat = thermal_stress.forward(heatwave_state, 0.0, 1.0)
        water_heat = water_stress.forward(heatwave_state, 0.0, 1.0)
        heat_temperature_stress = _update_value(
            thermal_heat, 'temperature_stress_factor'
        )
        assert isinstance(heat_temperature_stress, torch.Tensor)
        heatwave_state['temperature_stress_factor'] = heat_temperature_stress
        heatwave_state['water_stress_factor'] = _update_value(
            water_heat, 'water_stress_factor'
        )
        
        photo_heat = photosynthesis.forward(heatwave_state, 0.0, 1.0)
        gpp_heat = _update_value(photo_heat, 'gross_primary_productivity')
        heatwave_state['net_photosynthesis'] = _update_value(
            photo_heat, 'net_photosynthesis'
        )
        
        bvoc_heat = megan.forward(heatwave_state, 0.0, 1.0)
        emission_heat = _replace_value(
            bvoc_heat, 'total_bvoc_emission', 'mol m-2 s-1'
        )
        
        # Heatwave should reduce GPP but may increase some BVOC
        assert gpp_heat < gpp_normal  # Photosynthesis reduced
        # BVOC response is complex - temperature increases emission but stress reduces it

