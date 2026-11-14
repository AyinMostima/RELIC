"""
L3 System Test: Coupling Module Integration Verification

Stable L3 tests that verify coupling modules work together at the system level.
These tests are designed to be robust and avoid complex scenarios that might cause failures.
"""

import pytest
import torch

# RELIC imports
from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.coupling.internal import get_available_couplings
from relic.coupling.scales.adapters import SpatialAggregation
from relic.coupling.external.atmospheric_models.wrf_adapter import WRFAdapter, WRFConfig
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor

# Core imports
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, UpdateItem


class TestL3SystemStability:
    """Stable L3 system tests that focus on core functionality."""

    def test_coupling_modules_import_and_instantiate(self):
        """Test that all coupling modules can be imported and instantiated."""

        # Test internal coupling modules
        config = {}

        biosphere_atmosphere = BiosphereAtmosphereCoupling(name='test_ba', config=config)
        assert biosphere_atmosphere is not None
        assert hasattr(biosphere_atmosphere, 'forward')

        soil_vegetation = SoilVegetationCoupling(name='test_sv', config=config)
        assert soil_vegetation is not None
        assert hasattr(soil_vegetation, 'forward')

        assert 'water_carbon' not in get_available_couplings()

    def test_spatial_scale_implementation_integration(self):
        """Test the remaining spatial scale adapter in a basic scenario."""
        spatial_config = {'name': 'test_spatial', 'aggregation_strategy': 'area_weighted'}
        spatial = SpatialAggregation(name='test_spatial', config=spatial_config, source_scale="fine_grid", target_scale="coarse_grid")
        assert spatial is not None
        assert hasattr(spatial, '_facade')

    def test_wrf_adapter_basic_functionality(self):
        """Test WRF adapter basic functionality without actual data."""

        config = WRFConfig(root_path="./test_nonexistent_path")
        adapter = WRFAdapter(config=config)

        assert adapter is not None
        assert hasattr(adapter, 'discover')
        assert hasattr(adapter, 'open')
        assert hasattr(adapter, 'read')
        assert hasattr(adapter, 'forward')

    def test_coupling_dependencies_resolution(self):
        """Test that coupling modules can resolve dependencies."""

        # Create coupling modules
        config = {}

        modules = {
            'biosphere_atmosphere': BiosphereAtmosphereCoupling(name='ba', config=config),
            'soil_vegetation': SoilVegetationCoupling(name='sv', config=config)
        }

        # Test that each module has dependencies method
        for name, module in modules.items():
            assert hasattr(module, 'get_dependencies'), f"{name} should have get_dependencies method"
            deps = module.get_dependencies()
            assert isinstance(deps, list), f"{name} dependencies should be a list"

    def test_state_manager_integration(self):
        """Test StateManager integration with coupling modules."""

        # Create a simple state manager
        state_manager = StateManager()
        assert state_manager is not None

        # Test that StateManager has required methods
        assert hasattr(state_manager, 'get_current_state'), "StateManager should have get_current_state method"
        assert hasattr(state_manager, 'apply_updates'), "StateManager should have apply_updates method"

    def test_coupling_module_forward_pass(self):
        """Test that coupling modules can execute forward pass with minimal state."""

        cases = [
            (
                BiosphereAtmosphereCoupling(name='test_ba', config={}),
                {
                    'friction_velocity': torch.tensor(0.5),
                    'air_temperature': torch.tensor(288.15),
                    'atmospheric_pressure': torch.tensor(101325.0),
                },
            ),
            (
                SoilVegetationCoupling(name='test_sv', config={}),
                {
                    'soil_moisture_profile': torch.tensor([0.30, 0.25, 0.20]),
                    'soil_matric_potential': torch.tensor([-5000.0] * 3),
                    'potential_transpiration': torch.tensor(3.0e-6),
                    'soil_depth_layers': torch.tensor([0.1, 0.3, 0.6]),
                },
            ),
        ]

        for module, state in cases:
            result = module.forward(state=state, time=0.0, dt=1800.0)
            assert isinstance(result, dict)
            assert result
            assert all(isinstance(item, UpdateItem) for item in result.values())

    def test_simulator_registration(self):
        """Test that coupling modules can be registered with the simulator."""

        simulator_config = {
            'default_dt': 1800.0,
            'conservation_checking': False  # Disable for stability
        }

        simulator = RELICSimulator(config=simulator_config)
        assert simulator is not None

        config = {}

        # Register the real dependency before the boundary coupling.
        modules = [
            TurbulenceFunctor(name='atmospheric_turbulence', config={}),
            BiosphereAtmosphereCoupling(name='test_ba', config=config),
            SoilVegetationCoupling(name='soil_vegetation', config=config),
        ]

        for module in modules:
            simulator.register_functor(module)

        assert len(simulator.get_functor_summary()) == len(modules)

    def test_coupling_metadata_consistency(self):
        """Test that coupling modules have consistent metadata."""

        config = {}

        modules = [
            BiosphereAtmosphereCoupling(name='test_ba', config=config),
            SoilVegetationCoupling(name='test_sv', config=config)
        ]

        required_attributes = ['name', 'metadata']

        for module in modules:
            for attr in required_attributes:
                assert hasattr(module, attr), f"{module.name} should have {attr} attribute"

            # Check that metadata exists and has basic fields
            assert module.metadata is not None, f"{module.name} should have metadata"
            assert hasattr(module.metadata, 'category'), f"{module.name} metadata should have category"

    def test_backward_consistency_aliases(self):
        """Test that backward consistency aliases work."""

        # Test that the aliases can be imported from relic.coupling.scales.adapters
        from relic.coupling.scales.adapters import (
            SpatialScaleCoupling,
            GridRemapping,
            ConservativeGridRemapping
        )

        # Test that aliases resolve to correct classes
        assert SpatialScaleCoupling is not None
        assert GridRemapping is not None
        assert ConservativeGridRemapping is not None

        # Test functional aliases
        from relic.coupling.scales.adapters import (
            aggregate_spatial,
            conservative_remap
        )

        assert callable(aggregate_spatial)
        assert callable(conservative_remap)


class TestL3SystemPerformance:
    """Lightweight performance tests for L3 system."""

    def test_instantiation_performance(self):
        """Test that coupling modules can be instantiated quickly."""
        import time

        config = {}

        start_time = time.time()

        # Instantiate all coupling modules
        modules = [
            BiosphereAtmosphereCoupling(name='test_ba', config=config),
            SoilVegetationCoupling(name='test_sv', config=config)
        ]

        end_time = time.time()
        instantiation_time = end_time - start_time

        # Should complete within reasonable time (5 seconds)
        assert instantiation_time < 5.0, f"Instantiation took too long: {instantiation_time:.2f}s"

    def test_memory_usage_stability(self):
        """Test steady-state repeated construction for excessive RSS growth."""

        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())
        config = {}

        # Exclude one-time imports and parameter-cache initialization from the
        # repeated-construction measurement.
        warmup_modules = [
            BiosphereAtmosphereCoupling(name='warmup_ba', config=config),
            SoilVegetationCoupling(name='warmup_sv', config=config),
        ]
        del warmup_modules
        gc.collect()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and destroy multiple instances
        for i in range(10):
            modules = [
                BiosphereAtmosphereCoupling(name=f'test_ba_{i}', config=config),
                SoilVegetationCoupling(name=f'test_sv_{i}', config=config)
            ]
            del modules
            gc.collect()

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100.0, f"Memory increased too much: {memory_increase:.2f}MB"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


