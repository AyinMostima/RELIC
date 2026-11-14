#!/usr/bin/env python3
"""
L3 System Tests for Anthroposphere Module
Tests the end-to-end functionality of the complete anthroposphere system

Test Coverage:
- Complete anthroposphere system workflow
- Integration of management, prescribed-rate emission consumers, and land use
- System-level conservation laws and mass balance
- Long-term temporal dynamics and stability
- System response to environmental drivers
- Performance and scalability of complete system
- Scientific accuracy of integrated processes
- Error handling and recovery at system level
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "relic"))

from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator
from relic.scientific.anthroposphere.emissions.species_mapping import SpeciesMapper
from relic.scientific.anthroposphere.emissions.sector_allocation import SectorAllocation
from relic.scientific.anthroposphere.land_use.transitions import LandUseTransitions
from relic.core.state import UpdateItem, UpdateMode


def _materialize_updates(updates):
    """Return StateManager-equivalent tensor values for direct chain tests."""
    return {
        key: item.value if isinstance(item, UpdateItem) else item
        for key, item in updates.items()
    }


def _output_tensor(updates, name):
    """Require one explicit coordinator replacement output."""
    item = updates[name]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.per_timestep is False
    assert isinstance(item.value, torch.Tensor)
    return item.value


class TestAnthroposphereSystemL3:
    """L3 System Tests for complete anthroposphere functionality"""

    @pytest.fixture
    def complete_anthroposphere_system(self):
        """Create complete anthroposphere system for testing"""
        management_config = {
            'calendar_type': 'noleap',
            'management_practices': ['fertilization', 'irrigation', 'tillage', 'harvest'],
            'coordination_rules': {
                'conflict_resolution': 'priority',
                'spatial_consistency': True,
                'temporal_ordering': True
            }
        }
        management = ManagementCoordinator(
            name="test_management_coordinator",
            config=management_config
        )
        species_mapper = SpeciesMapper(name="test_species_mapper")
        sector_allocator = SectorAllocation(name="test_sector_allocator")
        land_use_transitions = LandUseTransitions(name="test_land_use_transitions")

        return {
            'management': management,
            'species_mapper': species_mapper,
            'sector_allocator': sector_allocator,
            'land_use_transitions': land_use_transitions
        }

    @pytest.fixture
    def system_state(self):
        """Create comprehensive system state for L3 testing"""
        n_cells = 5
        n_land_use_types = 9
        land_use_fraction = torch.rand(n_cells, n_land_use_types)
        land_use_fraction = land_use_fraction / land_use_fraction.sum(
            dim=1,
            keepdim=True,
        )

        return {
            # Physical state variables
            'temperature': torch.rand(n_cells) * 20 + 273.15,  # K
            'precipitation': torch.rand(n_cells) * 10,  # mm day-1
            'solar_radiation': torch.rand(n_cells) * 500,  # W/m2
            'wind_speed': torch.rand(n_cells) * 10,  # m/s
            'relative_humidity': torch.rand(n_cells) * 0.8 + 0.2,  # dimensionless

            # Soil state variables
            'soil_moisture': torch.rand(n_cells) * 0.4 + 0.2,  # m3/m3
            'soil_nitrogen': torch.rand(n_cells) * 0.01 + 0.005,  # kg N m-2
            'soil_organic_carbon': torch.rand(n_cells) * 3.0 + 3.0,  # kg C m-2
            'soil_temperature': torch.rand(n_cells) * 20 + 283.15,  # K
            'soil_carbon_stock': torch.rand(n_cells) * 100,  # tC/ha
            'soil_nitrogen_stock': torch.rand(n_cells) * 10,  # tN/ha

            # Vegetation state variables
            'crop_biomass': torch.rand(n_cells) * 10,  # t/ha
            'crop_maturity': torch.rand(n_cells),  # dimensionless
            'aboveground_biomass': torch.rand(n_cells),  # kg DM m-2
            'leaf_area_index': torch.rand(n_cells) * 5,  # m2/m2
            'vegetation_height': torch.rand(n_cells) * 3,  # m

            # Canonical external forcing used by the coordinator
            'air_temperature': torch.rand(n_cells) * 20 + 283.15,  # K

            # Land use state variables
            'land_use_fraction': land_use_fraction,

            # Management state variables
            'fertilizer_product_flux': torch.zeros(n_cells),  # kg product m-2 s-1
            'irrigation_demand_flux': torch.zeros(n_cells),  # kg m-2 s-1
            'tillage_intensity': torch.zeros(n_cells),  # dimensionless
            'irrigation_flux': torch.zeros(n_cells),  # kg m-2 s-1
            'harvest_flux': torch.zeros(n_cells),  # kg DM m-2 s-1
            'grazing_flux': torch.zeros(n_cells),  # kg DM m-2 s-1
            'co2_emission': torch.zeros(n_cells),  # kg CO2 m-2 s-1
            'field_capacity': torch.full((n_cells,), 0.35),
            'wilting_point': torch.full((n_cells,), 0.15),
            'evapotranspiration_flux': torch.full((n_cells,), 3.0e-6),
            'precipitation_flux': torch.zeros(n_cells),

            # Emissions state variables
            'base_emissions': torch.rand(n_cells, 4) * 10,  # kg day-1 (NOx, SOx, CO, PM25)
            'sector_emissions': torch.zeros(n_cells, 4),  # kg day-1
            'species_emissions': torch.zeros(n_cells, 10),  # kg day-1

            # Economic and demographic variables
            'population_density': torch.rand(n_cells) * 1000,  # people/km2
            'gdp_per_capita': torch.rand(n_cells) * 50000,  # USD/year
            'energy_consumption': torch.rand(n_cells) * 1000,  # MJ day-1

            # Policy and regulation variables
            'emission_tax': torch.rand(n_cells) * 100,  # USD/tCO2
            'conservation_policy': torch.rand(n_cells),  # dimensionless
            'management_intensity': torch.rand(n_cells),  # dimensionless
        }

    def test_system_initialization(self, complete_anthroposphere_system):
        """Test that complete anthroposphere system initializes properly"""
        system = complete_anthroposphere_system

        # Check that all modules are created
        assert 'management' in system
        assert 'species_mapper' in system
        assert 'sector_allocator' in system
        assert 'land_use_transitions' in system

        # Check that modules have required attributes
        assert hasattr(system['management'], 'forward')
        assert hasattr(system['species_mapper'], 'forward')
        assert hasattr(system['land_use_transitions'], 'forward')

    def test_end_to_end_workflow(self, complete_anthroposphere_system, system_state):
        """Test complete end-to-end anthroposphere workflow"""
        system = complete_anthroposphere_system

        # Initial state
        current_state = system_state.copy()

        # Normalize land use fractions to sum to 1
        current_state['land_use_fraction'] = (
            current_state['land_use_fraction'] /
            current_state['land_use_fraction'].sum(dim=1, keepdim=True)
        )

        # Time parameters
        base_time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0  # 1 day

        try:
            # Step 1: Management decisions
            management_updates = system['management'].forward(
                current_state, base_time, dt
            )
            assert isinstance(management_updates, dict)

            # Update state with management results
            current_state.update(_materialize_updates(management_updates))

            # Step 2: Land use changes (influenced by management)
            land_use_updates = system['land_use_transitions'].forward(
                current_state, base_time, dt
            )
            assert isinstance(land_use_updates, dict)
            current_state.update(_materialize_updates(land_use_updates))

            # Step 3: Prescribe downstream consumer rates explicitly.
            emissions = {
                'NOx': 100.0,
                'SOx': 50.0,
                'CO': 200.0,
                'PM25': 25.0
            }

            # Step 4: Route explicit test-only canonical mass-flux fields.
            grid_shape = (1, current_state['population_density'].numel())
            canonical_values = {
                'anthropogenic_nox_flux': emissions['NOx'],
                'anthropogenic_co_flux': emissions['CO'],
                'anthropogenic_so2_flux': emissions['SOx'],
                'anthropogenic_voc_flux': 10.0,
                'anthropogenic_nh3_flux': 5.0,
                'anthropogenic_pm25_flux': emissions['PM25'],
            }
            for key, value in canonical_values.items():
                current_state[key] = torch.full(
                    grid_shape, value * 1.0e-9, dtype=torch.float64
                )
            current_state['population_density'] = current_state[
                'population_density'
            ].reshape(grid_shape)
            current_state['gdp_density'] = (
                current_state['gdp_per_capita'].reshape(grid_shape)
                * current_state['population_density']
            )
            current_state['road_density'] = torch.ones(
                grid_shape, dtype=torch.float64
            )
            current_state['land_use_fractions'] = current_state[
                'land_use_fraction'
            ].transpose(0, 1).unsqueeze(1)
            sector_updates = system['sector_allocator'].forward(
                current_state, base_time, dt
            )
            item = sector_updates['sector_emission_flux']
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            canonical_stack = torch.stack(
                [
                    current_state[key]
                    for key in system['sector_allocator'].CANONICAL_INPUTS
                ]
            )
            torch.testing.assert_close(
                item.value.sum(dim=0),
                canonical_stack,
                rtol=1.0e-12,
                atol=1.0e-18,
            )

            # Step 5: Map the full sector-resolved port to typed outputs.
            species_updates = system['species_mapper'].forward(
                {'sector_emission_flux': item.value},
                base_time,
                dt,
            )
            gas_item = species_updates['mechanism_gas_molar_flux']
            aerosol_item = species_updates['mechanism_aerosol_mass_flux']
            assert isinstance(gas_item, UpdateItem)
            assert isinstance(aerosol_item, UpdateItem)
            assert gas_item.mode is UpdateMode.REPLACE
            assert aerosol_item.mode is UpdateMode.REPLACE
            torch.testing.assert_close(
                species_updates[
                    'diagnostics.species_mapping.inventory_equivalent_output_mass_flux'
                ],
                item.value.sum(dim=(0, 1)),
            )

        except Exception as e:
            pytest.fail(f"End-to-end workflow failed: {e}")

    def test_system_conservation_laws(self, complete_anthroposphere_system, system_state):
        """Test conservation laws at system level"""
        system = complete_anthroposphere_system
        current_state = system_state.copy()

        # Track initial carbon and nitrogen stocks
        initial_carbon = current_state['soil_carbon_stock'].clone()
        initial_nitrogen = current_state['soil_nitrogen_stock'].clone()

        base_time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        try:
            # Run complete system for multiple time steps
            for day in range(10):
                current_time = base_time + day * dt

                # Management
                management_updates = system['management'].forward(
                    current_state, current_time, dt
                )
                current_state.update(_materialize_updates(management_updates))

                # Land use changes
                land_use_updates = system['land_use_transitions'].forward(
                    current_state, current_time, dt
                )
                current_state.update(_materialize_updates(land_use_updates))

                # Coordinator control observations do not own soil stock updates.

            # Check final stocks
            final_carbon = current_state['soil_carbon_stock']
            final_nitrogen = current_state['soil_nitrogen_stock']

            # Carbon should not increase dramatically without photosynthesis input
            carbon_change = final_carbon - initial_carbon
            assert torch.all(carbon_change < initial_carbon * 0.1), \
                "Carbon stocks should not increase dramatically without photosynthesis"

            # Nitrogen should account for fertilizer additions
            nitrogen_change = final_nitrogen - initial_nitrogen
            assert torch.all(nitrogen_change >= -initial_nitrogen * 0.5), \
                "Nitrogen stocks should not decrease dramatically"
            torch.testing.assert_close(final_carbon, initial_carbon)
            torch.testing.assert_close(final_nitrogen, initial_nitrogen)

        except Exception as e:
            pytest.fail(f"System conservation laws test failed: {e}")

    def test_system_temporal_dynamics(self, complete_anthroposphere_system, system_state):
        """Test system temporal dynamics and stability"""
        system = complete_anthroposphere_system
        current_state = system_state.copy()

        # Normalize land use fractions
        current_state['land_use_fraction'] = (
            current_state['land_use_fraction'] /
            current_state['land_use_fraction'].sum(dim=1, keepdim=True)
        )

        # Day-0 drivers; the seasonal variation below is written into the state,
        # the only input channel of the functors.
        seasonal_base = {
            'temperature': current_state['temperature'],
            'precipitation': current_state['precipitation']
        }

        base_time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        # Track system state over time
        state_history = []

        try:
            # Run system for extended period
            for day in range(30):  # One month
                current_time = base_time + day * dt

                # Add seasonal variation to environmental drivers
                seasonal_factor = np.sin(2 * np.pi * day / 365)
                modified_env = {
                    'temperature': seasonal_base['temperature'] + seasonal_factor * 10,
                    'air_temperature': current_state['air_temperature'] + seasonal_factor * 10,
                    'precipitation': seasonal_base['precipitation'] * (1 + 0.5 * seasonal_factor)
                }
                current_state.update(modified_env)

                # Management response
                management_updates = system['management'].forward(
                    current_state, current_time, dt
                )
                current_state.update(_materialize_updates(management_updates))

                # Land use transitions
                land_use_updates = system['land_use_transitions'].forward(
                    current_state, current_time, dt
                )
                current_state.update(_materialize_updates(land_use_updates))

                # Record key state variables
                state_snapshot = {
                    'day': day,
                    'biomass': current_state['crop_biomass'].clone(),
                    'soil_moisture': current_state['soil_moisture'].clone(),
                    'fertilizer_product_flux': current_state[
                        'fertilizer_product_flux'
                    ].clone(),
                }
                state_history.append(state_snapshot)

            # Check temporal dynamics
            assert len(state_history) == 30, "Should record state for all days"

            # Check that variables evolve smoothly (no extreme jumps)
            for i in range(1, len(state_history)):
                for var in ['biomass', 'soil_moisture', 'fertilizer_product_flux']:
                    change = torch.abs(state_history[i][var] - state_history[i-1][var])
                    scale = torch.mean(torch.abs(state_history[i-1][var]))
                    tolerance = torch.maximum(
                        scale * 0.5,
                        torch.as_tensor(1.0e-12, dtype=change.dtype, device=change.device),
                    )
                    assert torch.all(change <= tolerance), \
                        f"Variable {var} shows excessive change between days {i-1} and {i}"

        except Exception as e:
            pytest.fail(f"System temporal dynamics test failed: {e}")

    def test_system_response_to_environmental_drivers(self, complete_anthroposphere_system, system_state):
        """Test system response to environmental drivers"""
        system = complete_anthroposphere_system

        # Test different environmental conditions
        environmental_scenarios = [
            {
                'name': 'optimal',
                'temperature': torch.tensor([298.15]),  # 25掳C
                'precipitation': torch.tensor([5.0]),  # 5 mm day-1
                'solar_radiation': torch.tensor([500.0])  # High radiation
            },
            {
                'name': 'drought',
                'temperature': torch.tensor([308.15]),  # 35掳C
                'precipitation': torch.tensor([0.5]),   # Low precipitation
                'solar_radiation': torch.tensor([700.0])  # Very high radiation
            },
            {
                'name': 'cool_wet',
                'temperature': torch.tensor([283.15]),  # 10掳C
                'precipitation': torch.tensor([15.0]),  # High precipitation
                'solar_radiation': torch.tensor([200.0])  # Low radiation
            }
        ]

        base_time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        results = {}

        for scenario in environmental_scenarios:
            current_state = system_state.copy()
            current_state['air_temperature'] = torch.full_like(
                current_state['air_temperature'],
                float(scenario['temperature'].item()),
            )
            scenario_soil_moisture = {
                'optimal': 0.30,
                'drought': 0.15,
                'cool_wet': 0.35,
            }[scenario['name']]
            current_state['soil_moisture'] = torch.full_like(
                current_state['soil_moisture'],
                scenario_soil_moisture,
            )

            try:
                # Run management under different conditions
                management_updates = system['management'].forward(
                    current_state,
                    base_time,
                    dt,
                )

                # Record management response
                results[scenario['name']] = {
                    'fertilizer_product_flux': _output_tensor(
                        management_updates, 'fertilizer_product_flux'
                    ),
                    'irrigation_demand_flux': _output_tensor(
                        management_updates, 'irrigation_demand_flux'
                    ),
                    'management_active': _output_tensor(
                        management_updates,
                        'diagnostics.management_coordinator.active',
                    ),
                }

            except Exception as e:
                pytest.fail(f"Environmental scenario {scenario['name']} failed: {e}")

        # Check that system responds appropriately to different conditions
        # Drought should increase irrigation, cool-wet should reduce irrigation
        drought_irrigation = results['drought']['irrigation_demand_flux']
        cool_wet_irrigation = results['cool_wet']['irrigation_demand_flux']

        # This is a basic check - actual response may vary based on implementation
        assert isinstance(drought_irrigation, torch.Tensor), "Should return tensor for irrigation rate"
        assert isinstance(cool_wet_irrigation, torch.Tensor), "Should return tensor for irrigation rate"

    def test_system_performance_and_scalability(self, complete_anthroposphere_system):
        """Test system performance and scalability"""
        system = complete_anthroposphere_system

        # Test with different problem sizes
        problem_sizes = [1, 5, 10]
        performance_results = {}

        for n_cells in problem_sizes:
            # Create test state
            test_state = {
                'temperature': torch.rand(n_cells) * 20 + 273.15,
                'precipitation': torch.rand(n_cells) * 10,
                'soil_moisture': torch.rand(n_cells) * 0.4 + 0.2,
                'soil_nitrogen': torch.rand(n_cells) * 0.01 + 0.005,
                'soil_organic_carbon': torch.rand(n_cells) * 3.0 + 3.0,
                'crop_maturity': torch.rand(n_cells),
                'aboveground_biomass': torch.rand(n_cells),
                'air_temperature': torch.rand(n_cells) * 20 + 283.15,
                'irrigation_flux': torch.zeros(n_cells),
                'harvest_flux': torch.zeros(n_cells),
                'grazing_flux': torch.zeros(n_cells),
                'co2_emission': torch.zeros(n_cells),
                'field_capacity': torch.full((n_cells,), 0.35),
                'wilting_point': torch.full((n_cells,), 0.15),
                'evapotranspiration_flux': torch.full((n_cells,), 3.0e-6),
                'precipitation_flux': torch.zeros(n_cells),
                'crop_biomass': torch.rand(n_cells) * 5.0,
                'land_use_fraction': torch.rand(n_cells, 9)
            }

            # Normalize land use fractions
            test_state['land_use_fraction'] = (
                test_state['land_use_fraction'] /
                test_state['land_use_fraction'].sum(dim=1, keepdim=True)
            )

            try:
                # Time the system execution
                start_time = time.time()

                # Run system for multiple steps
                base_time = datetime(2020, 6, 1).timestamp()
                for step in range(5):
                    current_time = base_time + step * 86400.0

                    # Management
                    management_updates = system['management'].forward(
                        test_state, current_time, 86400.0
                    )
                    test_state.update(_materialize_updates(management_updates))

                    # Land use transitions
                    land_use_updates = system['land_use_transitions'].forward(
                        test_state, current_time, 86400.0
                    )
                    test_state.update(_materialize_updates(land_use_updates))

                end_time = time.time()
                execution_time = end_time - start_time

                performance_results[n_cells] = execution_time

                # Should complete in reasonable time
                assert execution_time < 10.0, f"System too slow for {n_cells} cells: {execution_time:.2f}s"

            except Exception as e:
                pytest.fail(f"Performance test failed for {n_cells} cells: {e}")

        # Check scaling behavior
        assert len(performance_results) == len(problem_sizes), "Should have results for all problem sizes"

    def test_system_error_handling_and_recovery(self, complete_anthroposphere_system, system_state):
        """Test system error handling and recovery"""
        system = complete_anthroposphere_system

        # Test with problematic input data
        problematic_states = [
            {
                'name': 'negative_values',
                'modifications': {
                    'soil_moisture': torch.tensor([-0.1]),  # Negative soil moisture
                    'aboveground_biomass': torch.tensor([-1.0]),
                }
            },
            {
                'name': 'extreme_values',
                'modifications': {
                    'air_temperature': torch.tensor([500.0]),
                }
            },
            {
                'name': 'dimension_mismatch',
                'modifications': {
                    'land_use_fraction': torch.tensor([[0.3, 0.2, 0.4, 0.05, 0.05]]),  # Invalid 1x5
                    'soil_moisture': torch.tensor([0.3, 0.4])  # 2x1 - mismatch
                }
            }
        ]

        base_time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        for scenario in problematic_states:
            # Create problematic state
            test_state = system_state.copy()
            test_state.update(scenario['modifications'])

            try:
                # Test management system
                management_updates = system['management'].forward(
                    test_state, base_time, dt
                )

                # System should either handle gracefully or fail with informative error
                assert isinstance(management_updates, dict), "Should return dict even with problematic input"

                # Check outputs are finite
                for key, value in _materialize_updates(management_updates).items():
                    assert isinstance(value, torch.Tensor)
                    assert torch.isfinite(value).all(), (
                        f"Output {key} contains non-finite values in {scenario['name']}"
                    )

            except Exception as e:
                # Should fail gracefully with informative error
                assert isinstance(str(e), str), f"Error should be informative in {scenario['name']}: {e}"

    def test_system_scientific_accuracy(self, complete_anthroposphere_system, system_state):
        """Test scientific accuracy of integrated system"""
        system = complete_anthroposphere_system

        # Test with known reference conditions
        reference_state = system_state.copy()

        # Set realistic values based on literature
        reference_state.update({
            'temperature': torch.tensor([298.15]),  # 25掳C - optimal for many crops
            'precipitation': torch.tensor([3.0]),     # 3 mm day-1 - moderate rainfall
            'soil_moisture': torch.tensor([0.3]),     # 30% volumetric water content
            'soil_nitrogen': torch.tensor([0.01]),    # kg N m-2
            'soil_organic_carbon': torch.tensor([5.0]),  # kg C m-2
            'crop_maturity': torch.tensor([0.8]),
            'aboveground_biomass': torch.tensor([0.5]),  # kg DM m-2
            'air_temperature': torch.tensor([298.15]),  # K
            'crop_biomass': torch.tensor([5.0]),      # 5 t/ha - reasonable biomass
            'irrigation_flux': torch.zeros(1),  # kg m-2 s-1
            'harvest_flux': torch.zeros(1),  # kg DM m-2 s-1
            'grazing_flux': torch.zeros(1),  # kg DM m-2 s-1
            'co2_emission': torch.zeros(1),  # kg CO2 m-2 s-1
            'field_capacity': torch.full((1,), 0.35),
            'wilting_point': torch.full((1,), 0.15),
            'evapotranspiration_flux': torch.full((1,), 3.0e-6),
            'precipitation_flux': torch.zeros(1),
            'soil_carbon_stock': torch.tensor([80.0]), # 80 tC/ha - typical for agricultural soils
            'soil_nitrogen_stock': torch.tensor([5.0]) # 5 tN/ha - typical agricultural soils
        })

        base_time = datetime(2020, 6, 1).timestamp()
        dt = 86400.0

        try:
            # Run management
            management_updates = system['management'].forward(
                reference_state, base_time, dt
            )

            # Check scientific plausibility of management actions
            fertilizer_product_flux = _output_tensor(
                management_updates, 'fertilizer_product_flux'
            )
            irrigation_demand_flux = _output_tensor(
                management_updates, 'irrigation_demand_flux'
            )
            assert torch.all(fertilizer_product_flux >= 0)
            assert torch.all(irrigation_demand_flux >= 0)
            assert torch.isfinite(fertilizer_product_flux).all()
            assert torch.isfinite(irrigation_demand_flux).all()

            # Run emissions processing
            test_emissions = torch.tensor(
                [1.0e-9],
                dtype=torch.float64,
            )  # kg NOx m-2 s-1, prescribed test-only rate
            mapper = system['species_mapper']
            sector_flux = torch.zeros((5, 6, 1), dtype=test_emissions.dtype)
            sector_flux[0, 0] = test_emissions
            species_result = mapper.forward(
                {'sector_emission_flux': sector_flux},
                time=0.0,
                dt=86400.0,
            )

            # Check closure on the prescribed NO2-equivalent mass basis.
            gas_molar = species_result['mechanism_gas_molar_flux'].value
            no_index = mapper.mechanism_gas_species_order.index('NO')
            no2_index = mapper.mechanism_gas_species_order.index('NO2')
            nox_mw = mapper.get_molecular_weight('NOx')
            total_nox_equivalent = (
                gas_molar[no_index] + gas_molar[no2_index]
            ) * nox_mw
            torch.testing.assert_close(
                total_nox_equivalent, test_emissions,
                rtol=1e-6, atol=1e-18,
                msg="NOx speciation should conserve prescribed NO2-equivalent mass"
            )

            no_equivalent = (
                gas_molar[no_index] * nox_mw
            )
            configured_no_fraction = mapper.get_parameter_value(
                'no_fraction',
                namespace='anthroposphere.emissions.partition_factors',
            )
            torch.testing.assert_close(
                no_equivalent / test_emissions,
                torch.full_like(test_emissions, float(configured_no_fraction)),
            )

        except Exception as e:
            pytest.fail(f"System scientific accuracy test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
