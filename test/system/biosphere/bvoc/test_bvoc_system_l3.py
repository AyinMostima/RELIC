"""
L3 System Tests for BVOC Module.

Tests full system integration including:
- Complete simulator integration
- Multi-module coupling
- Real-world scenarios
- Performance and scalability

Author: Yuchong Long  
Date: 2025-09-28
"""

import pytest
import torch
import numpy as np
import time

from relic.scientific.biosphere.bvoc.megan_model import (
    EmissionPathway,
    MEGANModel,
)
from relic.core.state import StateManager
from relic.core.dependency_graph import DependencyGraphResolver
from relic.parameters import get_global_manager


def _output_value(result, key):
    """Return the tensor carried by a canonical MEGAN UpdateItem."""
    item = result[key]
    assert hasattr(item, "value")
    return item.value


def _assert_canonical_emission_ledger(output) -> None:
    """Assert the molar species sum and equivalent carbon-allocation port."""
    species_sum = sum(
        _output_value(output, key)
        for key in (
            'isoprene_emission',
            'monoterpenes_emission',
            'sesquiterpenes_emission',
            'ovocs_emission',
        )
    )
    assert torch.allclose(
        _output_value(output, 'total_bvoc_emission'),
        species_sum,
        rtol=1.0e-6,
    )
    assert output['total_bvoc_emission'].unit == 'mol m-2 s-1'
    assert (
        output['bvoc_carbon_allocation_flux'].unit
        == 'kg C m-2 s-1'
    )


def ensure_bvoc_core_parameters(param_manager) -> None:
    """Ensure required BVOC parameters are present in the parameter system."""
    required_keys = [
        "biosphere.bvoc.default.isoprene",
        "biosphere.bvoc.default.monoterpenes",
        "biosphere.bvoc.default.sesquiterpenes",
        "biosphere.bvoc.default.ovocs",
        "biosphere.bvoc.megan.alpha_light",
        "biosphere.bvoc.megan.cl_light",
        "biosphere.bvoc.megan.ldf_monoterpenes",
        "biosphere.bvoc.megan.ldf_sesquiterpenes",
        "biosphere.bvoc.megan.topt",
        "biosphere.bvoc.megan.ct1_isoprene",
        "biosphere.bvoc.megan.ct2",
        "biosphere.bvoc.megan.eopt_isoprene",
        "biosphere.bvoc.megan.beta_monoterpenes",
        "relic.physical.constants.universal.gas_constant",
    ]
    for key in required_keys:
        param_manager.get(key, required=True)


class TestBVOCSystemIntegration:
    """Test BVOC module in full system context."""
    
    @pytest.fixture
    def setup_simulator(self):
        """Setup a minimal simulator with BVOC module."""
        param_manager = get_global_manager()

        # Ensure parameters are loaded from the central database
        ensure_bvoc_core_parameters(param_manager)
        
        # Create simulator components
        state_manager = StateManager()
        dep_resolver = DependencyGraphResolver()
        
        # Initialize system state
        initial_state = {
            'air_temperature': torch.tensor(298.0),
            'ppfd': torch.tensor(800.0),
            'leaf_area_index': torch.tensor(1.5),
            'atmospheric_co2_ppm': torch.tensor(410.0),
            'net_photosynthesis': torch.tensor(16.0),
            'water_stress_factor': torch.tensor(1.0),
            'temperature_stress_factor': torch.tensor(1.0),
            'ozone_stress_factor': torch.tensor(1.0),
            'soil_moisture': torch.tensor(0.3),
            'wind_speed': torch.tensor(2.0),
            'relative_humidity': torch.tensor(0.6),
            'atmospheric_pressure': torch.tensor(101325.0)
        }
        
        state_manager.initialize_state(initial_state)
        
        # Create BVOC module
        bvoc_model = MEGANModel()
        
        return state_manager, dep_resolver, bvoc_model
    
    def test_diurnal_cycle_simulation(self, setup_simulator):
        """Test BVOC emissions over a full diurnal cycle."""
        state_manager, dep_resolver, bvoc_model = setup_simulator
        
        # Simulate 24-hour cycle
        dt = 3600  # 1 hour timestep
        results = []
        
        for hour in range(24):
            # Update environmental conditions
            time_of_day = hour * 3600
            
            # Simple diurnal patterns
            if 6 <= hour <= 18:
                # Daytime
                par = 800 * np.sin(np.pi * (hour - 6) / 12)
                temp = 288 + 10 * np.sin(np.pi * (hour - 6) / 12)
            else:
                # Nighttime
                par = 0
                temp = 288 + 5 * np.cos(np.pi * (hour - 15) / 12)
            
            # Update state
            current_state = state_manager.get_current_state()
            current_state['ppfd'] = torch.tensor(float(par))
            current_state['air_temperature'] = torch.tensor(float(temp))
            current_state['net_photosynthesis'] = torch.tensor(
                max(0.0, 16.0 * float(par) / 800.0)
            )
            
            # Calculate emissions
            output = bvoc_model.forward(current_state, time=time_of_day, dt=float(dt))
            
            results.append({
                'hour': hour,
                'temperature': temp,
                'par': par,
                'total_emission': _output_value(
                    output, 'total_bvoc_emission'
                ).item(),
                'isoprene': _output_value(
                    output, 'isoprene_emission'
                ).item(),
            })
        
        # Verify diurnal pattern
        results_array = np.array([r['total_emission'] for r in results])
        
        # Daytime emissions should be higher
        day_mean = results_array[6:18].mean()
        night_mean = np.concatenate([results_array[:6], results_array[18:]]).mean()
        assert day_mean > night_mean * 5  # Significantly higher during day
        
        # Peak should be around midday
        peak_hour = np.argmax(results_array)
        assert 10 <= peak_hour <= 15
        
    def test_seasonal_variation(self, setup_simulator):
        """Test BVOC emissions across seasons."""
        state_manager, dep_resolver, bvoc_model = setup_simulator
        
        # Define seasonal conditions
        seasons = {
            'winter': {'temp': 278.0, 'par': 400.0, 'lai': 1.0},
            'spring': {'temp': 288.0, 'par': 600.0, 'lai': 3.0},
            'summer': {'temp': 298.0, 'par': 800.0, 'lai': 5.0},
            'fall': {'temp': 283.0, 'par': 500.0, 'lai': 2.0}
        }
        
        results = {}
        
        for season, conditions in seasons.items():
            # Update state
            current_state = state_manager.get_current_state()
            current_state['air_temperature'] = torch.tensor(conditions['temp'])
            current_state['ppfd'] = torch.tensor(conditions['par'])
            current_state['leaf_area_index'] = torch.tensor(conditions['lai'])
            current_state['net_photosynthesis'] = torch.tensor(
                16.0 * conditions['par'] / 800.0
            )
            
            # Calculate emissions
            output = bvoc_model.forward(current_state, time=0.0, dt=3600.0)
            results[season] = _output_value(
                output, 'total_bvoc_emission'
            ).item()
        
        # Verify seasonal pattern
        assert results['summer'] > results['spring']
        assert results['spring'] > results['fall']
        assert results['fall'] > results['winter']
        assert results['summer'] > results['winter'] * 10  # Much higher in summer
        
    def test_stress_response(self, setup_simulator):
        """Test BVOC response to environmental stress."""
        state_manager, dep_resolver, bvoc_model = setup_simulator
        stressed_model = MEGANModel(
            name="megan_stress_enabled_l3",
            emission_pathway=EmissionPathway.STANDARD,
            config={"use_stress_functors": True},
        )
        
        # Normal conditions
        normal_state = state_manager.get_current_state().copy()
        normal_output = stressed_model.forward(normal_state, time=0.0, dt=3600.0)
        normal_emission = _output_value(
            normal_output, 'total_bvoc_emission'
        ).item()
        
        # Heat stress
        heat_state = normal_state.copy()
        heat_state['air_temperature'] = torch.tensor(313.0)
        heat_output = stressed_model.forward(heat_state, time=0.0, dt=3600.0)
        heat_emission = _output_value(
            heat_output, 'total_bvoc_emission'
        ).item()
        
        # Water stress (if supported)
        drought_state = normal_state.copy()
        drought_state['water_stress_factor'] = torch.tensor(0.5)
        drought_output = stressed_model.forward(
            drought_state, time=0.0, dt=3600.0
        )
        drought_emission = _output_value(
            drought_output, 'total_bvoc_emission'
        ).item()
        
        # Verify stress responses
        # Heat initially increases emissions (up to optimum)
        assert heat_emission != normal_emission  # Should change with temperature
        assert drought_emission == pytest.approx(normal_emission * 0.5)
        
    def test_multi_compound_emissions(self, setup_simulator):
        """Test emissions of multiple BVOC compounds."""
        state_manager, dep_resolver, bvoc_model = setup_simulator
        
        current_state = state_manager.get_current_state()
        output = bvoc_model.forward(current_state, time=0.0, dt=3600.0)
        
        # Check for multiple compounds
        compounds = ['isoprene', 'monoterpenes', 'sesquiterpenes']
        found_compounds = []
        
        for compound in compounds:
            key = f'{compound}_emission'
            if key in output and output[key] is not None:
                found_compounds.append(compound)
                # Check reasonable values
                value = _output_value(output, key)
                assert value >= 0
                assert torch.isfinite(value)
        
        # Should have multiple compounds
        assert len(found_compounds) >= 2
        
        # Isoprene typically dominates
        if 'isoprene_emission' in output and 'monoterpenes_emission' in output:
            assert _output_value(
                output, 'isoprene_emission'
            ) > _output_value(output, 'monoterpenes_emission') * 0.5
            
    def test_conservation_laws(self, setup_simulator):
        """Test mass and energy conservation."""
        state_manager, dep_resolver, bvoc_model = setup_simulator
        
        # Run for multiple timesteps
        dt = 3600  # 1 hour
        total_emitted = 0.0
        carbon_emitted = 0.0
        
        for _ in range(24):  # 24 hours
            current_state = state_manager.get_current_state()
            output = bvoc_model.forward(current_state, time=0.0, dt=float(dt))
            
            species_sum = sum(
                _output_value(output, key)
                for key in (
                    'isoprene_emission',
                    'monoterpenes_emission',
                    'sesquiterpenes_emission',
                    'ovocs_emission',
                )
            )
            total_flux = _output_value(output, 'total_bvoc_emission')
            assert torch.allclose(total_flux, species_sum, rtol=1.0e-6)
            total_emitted += total_flux.item() * dt
            carbon_emitted += _output_value(
                output, 'bvoc_carbon_allocation_flux'
            ).item() * dt

        assert total_emitted > 0.0
        assert carbon_emitted > 0.0
        
    def test_performance_scaling(self, setup_simulator):
        """Test performance with different grid sizes."""
        state_manager, dep_resolver, bvoc_model = setup_simulator
        
        # Test different spatial resolutions
        grid_sizes = [(1, 1), (10, 10), (50, 50)]
        timing_results = []
        
        for grid_size in grid_sizes:
            # Create gridded state
            state = {
                'air_temperature': torch.randn(grid_size) * 5 + 298.0,
                'ppfd': torch.rand(grid_size) * 1000,
                'leaf_area_index': torch.ones(grid_size) * 1.5,
                'atmospheric_co2_ppm': torch.ones(grid_size) * 410.0,
                'net_photosynthesis': torch.ones(grid_size) * 16.0,
            }
            
            # Time the computation
            start = time.time()
            output = bvoc_model.forward(state, time=0.0, dt=3600.0)
            elapsed = time.time() - start
            
            timing_results.append({
                'size': grid_size,
                'pixels': grid_size[0] * grid_size[1],
                'time': elapsed
            })
            
            # Check output shape
            assert _output_value(
                output, 'total_bvoc_emission'
            ).shape == grid_size
        
        # Performance should scale reasonably
        # Time per pixel shouldn't increase dramatically
        time_per_pixel = [r['time'] / r['pixels'] for r in timing_results]
        
        # Later grids shouldn't be much slower per pixel
        if len(time_per_pixel) > 1:
            assert time_per_pixel[-1] < time_per_pixel[0] * 10
            
    def test_coupled_system_behavior(self, setup_simulator):
        """Test BVOC coupling with other modules."""
        state_manager, dep_resolver, bvoc_model = setup_simulator
        
        # Simulate coupling with photosynthesis
        current_state = state_manager.get_current_state()
        
        # The Hybrid closure requires the canonical photosynthesis port.
        output_coupled = bvoc_model.forward(current_state, time=0.0, dt=3600.0)

        # Missing canonical photosynthesis must fail instead of falling back.
        state_no_photo = current_state.copy()
        del state_no_photo['net_photosynthesis']

        assert _output_value(output_coupled, 'total_bvoc_emission') > 0
        with pytest.raises(ValueError, match="net_photosynthesis"):
            bvoc_model.forward(state_no_photo, time=0.0, dt=3600.0)
        
    def test_numerical_stability(self, setup_simulator):
        """Test numerical stability under extreme conditions."""
        state_manager, dep_resolver, bvoc_model = setup_simulator
        
        # Test with extreme values
        extreme_cases = [
            {'air_temperature': torch.tensor(253.0)},
            {'air_temperature': torch.tensor(323.0)},
            {'ppfd': torch.tensor(0.0)},
            {'ppfd': torch.tensor(2500.0)},
            {'leaf_area_index': torch.tensor(0.1)},
            {'leaf_area_index': torch.tensor(5.0)},
        ]
        
        for extreme in extreme_cases:
            state = state_manager.get_current_state().copy()
            state.update(extreme)
            
            # Should not crash or produce NaN/Inf
            output = bvoc_model.forward(state, time=0.0, dt=3600.0)
            
            total = _output_value(output, 'total_bvoc_emission')
            assert not torch.isnan(total).any()
            assert not torch.isinf(total).any()
            assert total >= 0


class TestBVOCRealWorldScenarios:
    """Test BVOC with real-world scenarios."""
    
    @pytest.fixture
    def setup_scenario(self):
        """Setup for scenario tests."""
        param_manager = get_global_manager()
        
        # Ensure required parameters are available in the database
        ensure_bvoc_core_parameters(param_manager)
        
        model = MEGANModel()
        return model
    
    def test_tropical_forest_emissions(self, setup_scenario):
        """Test emissions for tropical forest conditions."""
        model = setup_scenario
        
        # Tropical forest conditions
        state = {
            'air_temperature': torch.tensor(301.0),
            'ppfd': torch.tensor(1200.0),
            'leaf_area_index': torch.tensor(2.5),
            'atmospheric_co2_ppm': torch.tensor(410.0),
            'net_photosynthesis': torch.tensor(22.0),
        }
        
        output = model.forward(state, time=43200.0, dt=3600.0)  # Midday
        
        # Tropical forests are high emitters
        assert _output_value(output, 'total_bvoc_emission') > 0.0
        _assert_canonical_emission_ledger(output)
        
        # Isoprene dominates in many tropical species
        if 'isoprene_emission' in output:
            assert _output_value(output, 'isoprene_emission') > 0
            
    def test_temperate_deciduous_forest(self, setup_scenario):
        """Test emissions for temperate deciduous forest."""
        model = setup_scenario
        
        # Summer conditions in temperate forest
        state = {
            'air_temperature': torch.tensor(295.0),
            'ppfd': torch.tensor(800.0),
            'leaf_area_index': torch.tensor(1.75),
            'atmospheric_co2_ppm': torch.tensor(410.0),
            'net_photosynthesis': torch.tensor(16.0),
        }
        
        output = model.forward(state, time=43200.0, dt=3600.0)
        
        # Moderate emissions expected
        assert _output_value(output, 'total_bvoc_emission') > 0.0
        _assert_canonical_emission_ledger(output)
        
    def test_boreal_forest_emissions(self, setup_scenario):
        """Test emissions for boreal forest conditions."""
        model = setup_scenario
        
        # Summer in boreal forest
        state = {
            'air_temperature': torch.tensor(288.0),
            'ppfd': torch.tensor(600.0),
            'leaf_area_index': torch.tensor(1.25),
            'atmospheric_co2_ppm': torch.tensor(410.0),
            'net_photosynthesis': torch.tensor(12.0),
        }
        
        output = model.forward(state, time=43200.0, dt=3600.0)
        
        # Lower emissions than tropical/temperate
        assert _output_value(output, 'total_bvoc_emission') > 0.0
        _assert_canonical_emission_ledger(output)
        
        # Monoterpenes often important in conifers
        if 'monoterpenes_emission' in output:
            assert _output_value(output, 'monoterpenes_emission') >= 0
            
    def test_agricultural_crop_emissions(self, setup_scenario):
        """Test emissions from agricultural crops."""
        model = setup_scenario
        
        # Crop field in growing season
        state = {
            'air_temperature': torch.tensor(298.0),
            'ppfd': torch.tensor(1500.0),
            'leaf_area_index': torch.tensor(1.0),
            'atmospheric_co2_ppm': torch.tensor(410.0),
            'net_photosynthesis': torch.tensor(18.0),
        }
        
        output = model.forward(state, time=43200.0, dt=3600.0)
        
        # Crops generally emit less than forests
        assert _output_value(output, 'total_bvoc_emission') > 0.0
        _assert_canonical_emission_ledger(output)
        
    def test_urban_vegetation_emissions(self, setup_scenario):
        """Test emissions from urban vegetation."""
        model = setup_scenario
        
        # Urban park conditions
        state = {
            'air_temperature': torch.tensor(303.0),
            'ppfd': torch.tensor(500.0),
            'leaf_area_index': torch.tensor(0.75),
            'atmospheric_co2_ppm': torch.tensor(450.0),
            'net_photosynthesis': torch.tensor(10.0),
        }
        
        output = model.forward(state, time=43200.0, dt=3600.0)
        
        # Urban vegetation emissions
        assert _output_value(output, 'total_bvoc_emission') > 0
        _assert_canonical_emission_ledger(output)
        
        # Heat stress might enhance emissions
        # But low biomass limits total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

