"""
L2 Integration Tests for Anthroposphere Module
============================================

Tests the integration between anthroposphere modules and their dependencies.
Focuses on:
1. Module interaction and data flow
2. Cross-module dependencies
3. State consistency across modules
4. Scientific workflow integration

References:
- RELIC Integration Testing Framework
- Scientific Module Coupling Protocols
"""

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode

# Import anthroposphere modules
from relic.scientific.anthroposphere.management.fertilization import FertilizationManagement
from relic.scientific.anthroposphere.management.grazing import GrazingManagement
from relic.scientific.anthroposphere.management.harvest import HarvestManagement
from relic.scientific.anthroposphere.management.irrigation import IrrigationManagement
from relic.scientific.anthroposphere.management.tillage import TillageManagement
from relic.scientific.anthroposphere.management.fire_management import FireManagement
from relic.scientific.anthroposphere.emissions.sector_allocation import SectorAllocation
from relic.scientific.anthroposphere.land_use.mapping import LandUseParameterMapping


class TestL2AnthroposphereIntegration:
    """Test integration between anthroposphere modules."""

    @pytest.fixture
    def comprehensive_state(self):
        """Create a comprehensive state dictionary with all required variables."""
        batch_size = 5
        return {
            # Basic ecosystem state
            'biomass': torch.tensor([1000.0, 1200.0, 800.0, 1500.0, 900.0]),
            'aboveground_biomass': torch.tensor([1.0, 1.2, 0.8, 1.5, 0.9]),
            'belowground_biomass': torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]),
            'litter_mass': torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3]),
            'fuel_moisture': torch.tensor([0.15, 0.15, 0.15, 0.15, 0.15]),
            'days_since_fire': torch.tensor([30.0, 30.0, 30.0, 30.0, 30.0]),
            'crop_maturity': torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            'residue_cover': torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
            'vegetation_quality': torch.tensor([0.8, 0.75, 0.85, 0.7, 0.8]),
            'temperature': torch.tensor([288.15, 290.15, 286.15, 292.15, 289.15]),
            'leaf_area_index': torch.tensor([4.0, 4.5, 3.5, 5.0, 3.8]),
            'soil_organic_carbon': torch.tensor([50.0, 60.0, 45.0, 65.0, 52.0]),
            'soil_nitrogen': torch.tensor([100.0, 120.0, 90.0, 130.0, 105.0]),
            'soil_phosphorus': torch.tensor([20.0, 25.0, 18.0, 28.0, 22.0]),
            'soil_temperature': torch.tensor([288.15, 290.15, 286.15, 292.15, 289.15]),
            'soil_moisture': torch.tensor([0.25, 0.30, 0.22, 0.32, 0.28]),
            'precipitation_flux': torch.zeros(batch_size),
            'land_use_type': torch.ones(batch_size, dtype=torch.long),
            'irrigation_type': torch.full((batch_size,), 2, dtype=torch.long),
            'field_capacity': torch.full((batch_size,), 0.35),
            'wilting_point': torch.full((batch_size,), 0.15),
            'irrigation_remaining_quota': torch.tensor(800.0),
            'soil_ph': torch.tensor([6.5, 6.8, 6.2, 7.0, 6.6]),
            'root_depth': torch.tensor([0.5, 0.6, 0.4, 0.7, 0.55]),
            
            # Land use fractions (individual)
            'forest_fraction': torch.tensor([0.6, 0.5, 0.7, 0.4, 0.6]),
            'crop_fraction': torch.tensor([0.3, 0.4, 0.2, 0.5, 0.3]),
            'grass_fraction': torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1]),
            'urban_fraction': torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
            
            # Land use fractions (stacked for parameter mapping - all 9 types)
            'land_use_fraction': torch.stack([
                torch.tensor([0.3, 0.25, 0.35, 0.2, 0.3]),  # PRIMARY_FOREST
                torch.tensor([0.3, 0.25, 0.35, 0.2, 0.3]),  # SECONDARY_FOREST  
                torch.tensor([0.15, 0.2, 0.1, 0.25, 0.15]), # CROPLAND_C3
                torch.tensor([0.15, 0.2, 0.1, 0.25, 0.15]), # CROPLAND_C4
                torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05]), # PASTURE
                torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05]), # RANGELAND
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),     # URBAN
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),     # WATER
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])      # BARREN
            ], dim=1),
            
            # Management state
            'days_since_fertilization': torch.tensor([30.0, 45.0, 15.0, 60.0, 20.0]),
            'days_since_harvest': torch.tensor([90.0, 120.0, 60.0, 150.0, 75.0]),
            'days_since_tillage': torch.tensor([10.0, 20.0, 5.0, 30.0, 12.0]),
            'days_since_grazing': torch.tensor([7.0, 14.0, 3.0, 21.0, 10.0]),
            'irrigation_applied': torch.tensor([5.0, 8.0, 3.0, 12.0, 6.0]),
            
            # Emission sources
            'anthropogenic_nox_flux': torch.tensor([0.1, 0.15, 0.08, 0.2, 0.12]),
            'anthropogenic_co_flux': torch.tensor([0.5, 0.6, 0.4, 0.8, 0.55]),
            'anthropogenic_so2_flux': torch.tensor([0.02, 0.03, 0.015, 0.04, 0.025]),
            'anthropogenic_voc_flux': torch.tensor([0.3, 0.35, 0.25, 0.45, 0.32]),
            'anthropogenic_nh3_flux': torch.tensor([0.05, 0.08, 0.03, 0.1, 0.06]),
            'anthropogenic_pm25_flux': torch.tensor([0.01, 0.015, 0.008, 0.02, 0.012]),
            'pbl_height': torch.full((batch_size,), 1000.0),
            
            # Additional variables
            'vegetation_height': torch.tensor([15.0, 18.0, 12.0, 25.0, 16.0]),
            'albedo': torch.tensor([0.15, 0.18, 0.12, 0.22, 0.16]),
            'roughness_length': torch.tensor([0.5, 0.8, 0.3, 1.2, 0.6]),
        }

    @pytest.fixture
    def environment_variables(self):
        """Create comprehensive environment variables."""
        batch_size = 5
        return {
            'air_temperature': torch.tensor([298.15, 300.15, 296.15, 302.15, 299.15]),
            'precipitation': torch.tensor([2.0, 3.5, 1.5, 4.0, 2.5]),
            'solar_radiation': torch.tensor([800.0, 850.0, 750.0, 900.0, 825.0]),
            'wind_speed': torch.tensor([3.0, 4.0, 2.5, 5.0, 3.5]),
            'relative_humidity': torch.tensor([0.7, 0.65, 0.75, 0.6, 0.68]),
            'atmospheric_pressure': torch.tensor([101325.0, 101300.0, 101350.0, 101275.0, 101320.0]),
        }

    def test_management_workflow_integration(self, comprehensive_state, environment_variables):
        """Test the integration of management modules in a typical agricultural workflow."""
        
        # Initialize management modules
        fertilization = FertilizationManagement()
        tillage = TillageManagement() 
        irrigation = IrrigationManagement()
        harvest = HarvestManagement()
        
        # Create initial state
        state = comprehensive_state.copy()
        dt = 86400.0  # 1 day
        time = 0.0
        
        # Step 1: Tillage operation (prepares soil)
        tillage_result = tillage.forward(
            state, time, dt, calendar_type="noleap"
        )
        
        # Update state with tillage effects
        for key, value in tillage_result.items():
            payload = value.value if isinstance(value, UpdateItem) else value
            if (
                isinstance(value, UpdateItem)
                and value.mode is UpdateMode.ACCUMULATE
                and key in state
            ):
                state[key] = state[key] + payload
            else:
                state[key] = payload
        
        # Step 2: Fertilization (adds nutrients)  
        time += dt
        fertilization_result = fertilization.forward(
            state, time, dt, calendar_type="noleap"
        )
        
        # Fertilization exposes boundary fluxes and one integrated P update.
        n_update = fertilization_result['fertilizer_n_flux']
        p_update = fertilization_result['fertilizer_p_flux']
        available_p_update = fertilization_result['available_phosphorus']
        assert isinstance(n_update, UpdateItem)
        assert isinstance(p_update, UpdateItem)
        assert isinstance(available_p_update, UpdateItem)
        assert n_update.mode == UpdateMode.REPLACE
        assert p_update.mode == UpdateMode.REPLACE
        assert available_p_update.mode == UpdateMode.ACCUMULATE
        assert available_p_update.per_timestep is True
        
        # Step 3: Irrigation (if needed)
        time += dt
        irrigation_result = irrigation.forward(
            state, time, dt, calendar_type="noleap"
        )
        
        # Update state with irrigation effects
        for key, value in irrigation_result.items():
            payload = value.value if isinstance(value, UpdateItem) else value
            if (
                isinstance(value, UpdateItem)
                and value.mode is UpdateMode.ACCUMULATE
                and key in state
            ):
                state[key] = state[key] + payload
            else:
                state[key] = payload
                
        # Verify workflow consistency
        assert len(tillage_result) > 0, "Tillage should produce outputs"
        assert len(fertilization_result) > 0, "Fertilization should produce outputs"
        assert len(irrigation_result) > 0, "Irrigation should produce outputs"
        
        # Check that nutrient fluxes are reasonable
        if 'fertilizer_n_flux' in fertilization_result:
            n_flux = fertilization_result['fertilizer_n_flux'].value
            assert torch.all(n_flux >= 0), "N flux should be non-negative"
            assert torch.all(n_flux < 1.0), "N flux should be below 1 kg N m-2 s-1"
            
        print("[OK] Management workflow integration verified")

    def test_grazing_fire_interaction(self, comprehensive_state):
        """Test interaction between grazing and fire management."""
        
        grazing = GrazingManagement()
        fire = FireManagement(config={"seed": 0})
        
        state = comprehensive_state.copy()
        dt = 86400.0  # 1 day
        time = 0.0
        
        # Grazing reduces the canonical aboveground biomass stock.
        initial_biomass = state['aboveground_biomass'].clone()
        grazing_result = grazing.forward(
            state,
            time,
            dt,
            day_of_year=150,
            calendar_type="noleap",
        )
        grazing_flux = grazing_result['grazing_flux']
        assert isinstance(grazing_flux, UpdateItem)
        assert grazing_flux.mode is UpdateMode.REPLACE
        assert grazing_flux.per_timestep is False
        removed_biomass = grazing_flux.value * dt
        state['aboveground_biomass'] = initial_biomass - removed_biomass
        state['biomass'] = state['aboveground_biomass'] * 1000.0
            
        # Fire management considers reduced fuel load
        time += dt * 30  # 30 days later
        fire_result = fire.forward(
            state, time, dt, calendar_type="noleap"
        )
        
        # Verify interaction
        assert len(grazing_result) > 0, "Grazing should produce outputs"
        assert len(fire_result) > 0, "Fire management should produce outputs"
        
        # Check canonical biomass removal and pressure diagnostics.
        assert torch.all(state['aboveground_biomass'] >= 0), "Biomass should not be negative"
        assert torch.all(state['aboveground_biomass'] <= initial_biomass)
        # D is the stocking rate of the active (spring) grazing event (item 116).
        assert torch.allclose(
            grazing_result['diagnostics.grazing_pressure'],
            grazing.grazing_events[0].stocking_rate * state['vegetation_quality'],
        )
        
        print("[OK] Grazing-fire interaction verified")

    def test_fertilization_boundary_excludes_atmospheric_products(self, comprehensive_state):
        """Test separation of managed nutrient inputs from atmospheric products."""
        
        fertilization = FertilizationManagement()
        
        state = comprehensive_state.copy()
        dt = 3600.0  # 1 hour
        time = 0.0
        
        # Apply fertilization
        fert_result = fertilization.forward(
            state, time, dt, calendar_type="noleap"
        )
        
        # Fertilization owns only external N/P application boundaries. Soil
        # nitrogen processes own N2O production and NH3 volatilization.
        assert 'n2o_flux' not in fert_result
        assert 'volatilization_flux' not in fert_result
        assert isinstance(fert_result['fertilizer_n_flux'], UpdateItem)
        assert isinstance(fert_result['fertilizer_p_flux'], UpdateItem)

        print("[OK] Fertilization boundary verified")

    def test_cross_module_state_consistency(self, comprehensive_state):
        """Test that state variables remain consistent across multiple module calls."""
        
        # Initialize all modules
        modules = [
            ('fertilization', FertilizationManagement()),
            ('irrigation', IrrigationManagement()),
            ('tillage', TillageManagement()),
            ('harvest', HarvestManagement()),
        ]
        
        state = comprehensive_state.copy()
        original_state = {k: v.clone() for k, v in state.items()}
        dt = 3600.0
        time = 0.0
        
        # Apply each module sequentially
        for name, module in modules:
            try:
                result = module.forward(
                    state, time, dt, calendar_type="noleap"
                )
                
                # Check that results are reasonable
                assert isinstance(result, dict), f"{name} should return a dictionary"
                
                # Update time for next module
                time += dt
                
                print(f"[OK] {name}: State consistency maintained")
                
            except Exception as e:
                pytest.fail(f"{name} failed with: {e}")
        
        # Verify critical state variables haven't become unrealistic
        assert torch.all(state['soil_temperature'] > 200), "Soil temperature should be > 200K"
        assert torch.all(state['soil_temperature'] < 350), "Soil temperature should be < 350K"
        assert torch.all(state['soil_moisture'] >= 0), "Soil moisture should be non-negative"
        assert torch.all(state['soil_moisture'] <= 1), "Soil moisture should be <= 1"
        assert torch.all(state['soil_ph'] > 0), "Soil pH should be positive"
        assert torch.all(state['soil_ph'] < 14), "Soil pH should be < 14"
        
        print("[OK] Cross-module state consistency verified")

    def test_parameter_mapping_consistency(self, comprehensive_state):
        """Test that parameter mapping produces consistent ecosystem parameters."""
        
        land_use_mapping = LandUseParameterMapping()
        
        state = comprehensive_state.copy()
        dt = 86400.0
        time = 0.0
        
        # Test parameter mapping
        mapping_result = land_use_mapping.forward(state, time, dt)
        
        # Verify mapping produces parameters
        assert len(mapping_result) > 0, "Parameter mapping should produce outputs"
        
        # Check that mapped parameters are reasonable
        for key, update in mapping_result.items():
            assert key.startswith("diagnostics.land_use_parameter_mapping.")
            assert isinstance(update, UpdateItem)
            assert update.mode is UpdateMode.REPLACE
            assert update.per_timestep is False
            value = update.value
            assert torch.all(torch.isfinite(value)), f"{key} should be finite"
            
            if 'fraction' in key or 'efficiency' in key:
                assert torch.all(value >= 0), f"{key} should be non-negative"
                assert torch.all(value <= 1), f"{key} should be <= 1"
                
            if 'temperature' in key:
                assert torch.all(value > 200), f"{key} should be > 200K"
                assert torch.all(value < 400), f"{key} should be < 400K"
        
        print("[OK] Parameter mapping consistency verified")

    def test_module_dependencies_resolution(self):
        """Test that module dependencies are properly resolved."""
        
        # Initialize modules to test dependency resolution
        modules = {
            'fertilization': FertilizationManagement(),
            'irrigation': IrrigationManagement(),
            'land_use_mapping': LandUseParameterMapping(),
        }
        
        # Check that modules can be instantiated (dependency resolution)
        for name, module in modules.items():
            assert module is not None, f"{name} should be instantiated"
            assert hasattr(module, 'forward'), f"{name} should have forward method"
            
            # Check that module has required attributes
            assert hasattr(module, 'metadata'), f"{name} should have metadata"
            assert hasattr(module, 'config'), f"{name} should have config"
            
            print(f"[OK] {name}: Dependencies resolved successfully")

    def test_prescribed_flux_sector_allocation_closure(self, comprehensive_state):
        """Test SectorAllocation with explicit test-only prescribed rate fields."""

        sector_allocation = SectorAllocation()

        state = comprehensive_state.copy()
        proxy_grid = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3)
        for index, key in enumerate(
            sector_allocation.CANONICAL_INPUTS,
            start=1,
        ):
            state[key] = proxy_grid * (index * 1.0e-9)
        state.update(
            {
                'population_density': proxy_grid + 100.0,
                'gdp_density': proxy_grid + 50.0,
                'road_density': proxy_grid * 0.01 + 1.0,
                'land_use_fractions': torch.stack(
                    [torch.full_like(proxy_grid, 0.25) for _ in range(4)]
                ),
            }
        )
        dt = 3600.0
        time = 0.0

        allocation_result = sector_allocation.forward(state, time, dt)

        assert len(allocation_result) > 0, "Allocation should produce results"
        
        item = allocation_result['sector_emission_flux']
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        canonical_inputs = torch.stack(
            [state[key] for key in sector_allocation.CANONICAL_INPUTS]
        )
        torch.testing.assert_close(
            item.value.sum(dim=0), canonical_inputs, rtol=1.0e-5, atol=1.0e-12
        )

        print("[OK] Prescribed sector allocation closure verified")


if __name__ == "__main__":
    # Run tests individually for debugging
    test_instance = TestL2AnthroposphereIntegration()
    
    # Create fixtures
    state = test_instance.comprehensive_state()
    env = test_instance.environment_variables()
    
    print("Running L2 Integration Tests...")
    
    try:
        test_instance.test_management_workflow_integration(state, env)
        test_instance.test_land_use_emission_coupling(state) 
        test_instance.test_grazing_fire_interaction(state)
        test_instance.test_fertilization_emission_feedback(state)
        test_instance.test_cross_module_state_consistency(state)
        test_instance.test_parameter_mapping_consistency(state)
        test_instance.test_module_dependencies_resolution()
        test_instance.test_emission_sector_allocation_chain(state)
        
        print("\n[SUCCESS] All L2 integration tests passed!")
        
    except Exception as e:
        print(f"\n[FAILURE] L2 integration test failed: {e}")
        import traceback
        traceback.print_exc()

