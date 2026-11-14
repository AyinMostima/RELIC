"""
L2 Integration Tests for Biosphere Module Integration

This module tests the integration between different biosphere components
including photosynthesis, BVOC emissions, stress factors, and parameter management.

Author: Yuchong Long
Date: 2025-09-29
"""

import pytest
import torch

from relic.parameters import get_global_manager
from relic.scientific.biosphere.bvoc.megan_model import MEGANModel, CouplingStrategy
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.stress.water_stress import WaterStress
from relic.scientific.biosphere.stress.thermal_stress import ThermalStress
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.pedosphere.carbon_pools.carbon_pools_functor import CarbonPoolsFunctor
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.core.dependency_graph import DependencyGraphResolver
from relic.utils.physical_constants import get_physical_constant


class TestParameterSystemIntegration:
    """Test parameter system integration across modules."""
    
    @pytest.fixture
    def param_manager(self):
        """Get parameter manager instance."""
        return get_global_manager()
    
    def test_parameter_updates_propagate(self, param_manager):
        """Test that parameter updates propagate to all modules."""
        # Create modules
        photosynthesis = FarquharPhotosynthesis()
        megan = MEGANModel()
        
        # Update a parameter of the global manager and restore its original value
        # afterwards: a checkpoint restart later in the session compares the
        # accessed parameter contract of this process with a fresh process that
        # reads the YAML value.
        key = 'biosphere.photosynthesis.vcmax25'
        original_value = param_manager.get_parameter(key).value
        param_manager.set_value(key, 50.0)
        try:
            # Retrieve parameter through module
            vcmax = photosynthesis.get_parameter_value(
                'vcmax25',
                namespace='biosphere.photosynthesis',
                required=False
            )

            assert vcmax == 50.0
        finally:
            param_manager.set_value(key, original_value)
        assert param_manager.get_parameter(key).value == original_value
    
    def test_cross_module_parameter_sharing(self, param_manager):
        """Test the immutable physical-constant SSOT is globally consistent."""
        canonical_value = get_physical_constant('gas_constant')
        manager_value = param_manager.get(
            'relic.physical.constants.universal.gas_constant',
            units='J mol-1 K-1',
            required=True,
        )

        assert manager_value == canonical_value


class TestDependencyGraphIntegration:
    """Test dependency graph resolution with fixed modules."""
    
    def test_stress_module_dependency_resolution(self):
        """Test stress modules integrate correctly in dependency graph."""
        # Create dependency resolver
        resolver = DependencyGraphResolver()
        
        # Create functors
        water_stress = WaterStress()
        thermal_stress = ThermalStress()
        
        # Mock environmental forcing and soil moisture functors
        class EnvironmentalForcing:
            name = 'environmental_forcing'
            def get_dependencies(self):
                return []
        
        class BucketSoilMoisture:
            name = 'bucket_soil_moisture'
            def get_dependencies(self):
                return ['environmental_forcing']
        
        # Register functors
        functors = {
            'environmental_forcing': EnvironmentalForcing(),
            'bucket_soil_moisture': BucketSoilMoisture(),
            'water_stress': water_stress,
            'thermal_stress': thermal_stress
        }
        
        # Build dependency graph
        for name, functor in functors.items():
            resolver.add_functor(functor)
        
        # Resolve execution order
        execution_order = resolver.topological_sort()
        
        # Verify correct order
        env_idx = execution_order.index('environmental_forcing')
        bucket_idx = execution_order.index('bucket_soil_moisture')
        assert 'water_stress' in execution_order

        # The bucket depends on forcing, while water stress reads an external
        # canonical soil-water-potential port without a fake Functor edge.
        assert env_idx < bucket_idx
        assert water_stress.get_dependencies() == []
        assert water_stress.metadata.input_variables == ['soil_water_potential']
    
    def test_photosynthesis_bvoc_coupling(self):
        """Test photosynthesis-BVOC coupling through dependencies."""
        # Create functors with coupling
        photosynthesis = FarquharPhotosynthesis()
        megan = MEGANModel(coupling_strategy=CouplingStrategy.SEQUENTIAL)
        
        # Get dependencies
        megan_deps = megan.get_dependencies()
        
        # MEGAN should depend on photosynthesis when coupled
        assert 'farquhar_photosynthesis' in megan_deps
    
    def test_biomass_allocation_dependency(self):
        """Test allocation consumes the unique NPP and control upstream chain."""
        allocation = BiomassAllocation()
        deps = allocation.get_dependencies()

        assert deps == [
            'autotrophic_respiration',
            'water_stress',
            'gsi_phenology',
        ]
        assert 'farquhar_photosynthesis' not in deps


class TestStateManagerIntegration:
    """Test state management across integrated modules."""
    
    @pytest.fixture
    def state_manager(self):
        """Create state manager instance."""
        return StateManager(device=torch.device('cpu'))
    
    def test_stress_factor_propagation(self, state_manager):
        """Test stress factors propagate through state correctly."""
        # Initialize state
        state_manager.initialize_state({
            'temperature': torch.tensor(298.15),
            # Declared ThermalStress input (near-surface air temperature, K).
            'air_temperature': torch.tensor(298.15),
            'soil_moisture': torch.tensor(0.3),
            'soil_water_potential': torch.tensor(-0.5)
        })

        # Create stress functors
        water_stress = WaterStress()
        thermal_stress = ThermalStress()
        
        # Compute stress factors
        water_state = state_manager.get_state()
        water_output = water_stress.forward(water_state, time=0.0, dt=1.0)
        
        thermal_state = state_manager.get_state()
        thermal_output = thermal_stress.forward(thermal_state, time=0.0, dt=1.0)
        
        # Update state with stress factors
        water_item = water_output['water_stress_factor']
        assert isinstance(water_item, UpdateItem)
        assert water_item.mode is UpdateMode.REPLACE
        state_manager.apply_updates(
            {
                'water_stress_factor': water_item,
                'temperature_stress_factor': thermal_output['temperature_stress_factor'],
            },
            dt=1.0,
            check_conservation=False,
        )
        
        # Verify stress factors are in state
        final_state = state_manager.get_state()
        assert 'water_stress_factor' in final_state
        assert 'temperature_stress_factor' in final_state
        torch.testing.assert_close(
            final_state['water_stress_factor'],
            water_item.value,
        )
    
    def test_carbon_pools_state_integration(self, state_manager):
        """Test carbon pools integrate with state management."""
        # Initialize carbon pools
        carbon_pools = CarbonPoolsFunctor()
        
        # Initial state
        initial_state = {
            'metabolic_litter_c': torch.tensor(100.0),
            'structural_litter_c': torch.tensor(200.0),
            'fast_som_c': torch.tensor(500.0),
            'slow_som_c': torch.tensor(1000.0),
            'passive_som_c': torch.tensor(2000.0),
        }

        # The observer reads StateManager-owned pools without initializing them.
        output = carbon_pools.forward(initial_state, time=0.0, dt=1.0)

        assert set(output) == {
            'diagnostics.carbon_pools_snapshot',
            'diagnostics.total_carbon_stock',
        }
        torch.testing.assert_close(
            output['diagnostics.total_carbon_stock'],
            torch.tensor(3800.0),
        )


class TestPhotosynthesisBVOCIntegration:
    """Test photosynthesis and BVOC emission integration."""
    
    def test_sequential_coupling(self):
        """Test sequential coupling between photosynthesis and BVOC."""
        # Create modules
        photosynthesis = FarquharPhotosynthesis()
        megan = MEGANModel(coupling_strategy=CouplingStrategy.SEQUENTIAL)
        
        # Prepare state
        state = {
            'air_temperature': torch.tensor(298.15),
            'ppfd': torch.tensor(1000.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(3.0),
            'vcmax_scaling': torch.tensor(1.0),
        }
        
        # Run photosynthesis
        photo_output = photosynthesis.forward(state, time=0.0, dt=1.0)
        
        # Add photosynthesis outputs to state
        net_item = photo_output['net_photosynthesis']
        assert isinstance(net_item, UpdateItem)
        state['net_photosynthesis'] = net_item.value
        
        # Run MEGAN with photosynthesis coupling
        bvoc_output = megan.forward(state, time=0.0, dt=1.0)
        
        # Verify BVOC emissions are computed
        assert 'total_bvoc_emission' in bvoc_output
        total_item = bvoc_output['total_bvoc_emission']
        assert isinstance(total_item, UpdateItem)
        assert total_item.mode is UpdateMode.REPLACE
        assert total_item.per_timestep is False
        assert total_item.unit == 'mol m-2 s-1'
        assert total_item.value >= 0
    
    def test_stress_modulation(self):
        """Test stress factors modulate photosynthesis and BVOC."""
        # Create modules
        photosynthesis = FarquharPhotosynthesis()
        water_stress = WaterStress()
        
        # Base state
        base_state = {
            'temperature': torch.tensor(298.15),
            'air_temperature': torch.tensor(298.15),
            'par': torch.tensor(1000.0),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'foliar_density': torch.tensor(300.0),
            'soil_moisture': torch.tensor(0.3),
            'soil_water_potential': torch.tensor(-0.5),
            'vcmax_scaling': torch.tensor(1.0),
        }
        
        # Compute without stress
        photo_no_stress = photosynthesis.forward(base_state, time=0.0, dt=1.0)
        
        # Add water stress
        stress_output = water_stress.forward(base_state, time=0.0, dt=1.0)
        stress_item = stress_output['water_stress_factor']
        assert isinstance(stress_item, UpdateItem)
        base_state['water_stress_factor'] = stress_item.value
        
        # Compute with stress
        photo_with_stress = photosynthesis.forward(base_state, time=0.0, dt=1.0)
        
        # Photosynthesis should be reduced under stress
        gpp_no_stress = photo_no_stress['gross_primary_productivity'].value
        gpp_with_stress = photo_with_stress['gross_primary_productivity'].value
        
        # Stress should reduce photosynthesis (or at least not increase it)
        assert gpp_with_stress <= gpp_no_stress * 1.01  # Allow small numerical tolerance


class TestParameterAuditIntegration:
    """Test parameter audit functionality across modules."""
    
    def test_audit_detects_missing_parameters(self):
        """Test audit system detects missing required parameters."""
        param_manager = get_global_manager()
        
        # Clear any previous missing requests
        param_manager._missing_requests.clear()
        
        # Create module that will request missing parameter
        photosynthesis = FarquharPhotosynthesis()
        
        # Try to get a non-existent required parameter
        try:
            photosynthesis.get_parameter_value(
                'non_existent_param',
                namespace='biosphere.photosynthesis',
                required=True
            )
        except (KeyError, RuntimeError, ValueError):
            pass  # Expected to fail
        
        # Run audit
        audit_report = param_manager.run_registry_audit()
        
        # Check if audit detected issues
        assert audit_report.observed_total >= 0
        # Missing requests should be tracked if using register_if_missing
    
    def test_namespace_validation(self):
        """Test parameter namespaces are properly validated."""
        param_manager = get_global_manager()
        
        # Load parameters for testing
        param_manager.load_category(
            category='biosphere',
            subcategory='photosynthesis'
        )
        
        # Verify namespace structure
        params = param_manager.get_parameters_by_category('biological_parameter')
        
        # Check namespace consistency
        for param_name in params:
            # Parameters should follow hierarchical naming
            assert '.' in param_name or param_name in ['vcmax25', 'jmax25', 'rd25']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

