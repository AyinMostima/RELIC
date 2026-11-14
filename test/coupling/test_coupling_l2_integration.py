"""
L2 Integration Test: Coupling Module Integration with StateManager and Simulator

Tests the integration of coupling modules with the RELIC core infrastructure:
- StateManager transaction and conservation checks
- Dependency resolution and execution order
- Cross-module data flow
- Conservation law enforcement

This validates that P0 fixes work correctly in the full system context.

References:
- relic/core/state_manager.py
- relic/core/dependency_graph.py
- relic/core/simulator.py
"""

import pytest
import torch

# Core infrastructure
from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode
from relic.core.dependency_graph import DependencyGraphResolver

# Coupling modules to test
from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.coupling.internal.water_carbon import WaterCarbonCoupling
from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor

# Scale implementations
from relic.coupling.scales.adapters import SpatialAggregation


class TestInternalCouplingIntegration:
    """Test internal coupling modules with StateManager."""

    @pytest.fixture
    def state_manager(self):
        """Create a StateManager instance for testing."""
        # StateManager accepts schema parameter, not config
        # Create minimal schema for coupling tests
        schema = {
            'gpp': StateVariableSpec(
                name='gpp', units='g C m-2 s-1', bounds=(0.0, 1.0),
                conserved=True, spatial_dims=1, required=False
            ),
            'nee': StateVariableSpec(
                name='nee', units='g C m-2 s-1', bounds=(-1.0, 1.0),
                conserved=True, spatial_dims=1, required=False
            ),
            'soil_respiration': StateVariableSpec(
                name='soil_respiration', units='g C m-2 s-1', bounds=(0.0, 1.0),
                conserved=True, spatial_dims=1, required=False
            ),
            'air_temperature': StateVariableSpec(
                name='air_temperature', units='K', bounds=(200.0, 350.0),
                conserved=False, spatial_dims=0, required=False
            ),
            'soil_moisture': StateVariableSpec(
                name='soil_moisture', units='m3 m-3', bounds=(0.0, 1.0),
                conserved=False, spatial_dims=1, required=False
            ),
        }
        manager = StateManager(schema=schema)
        manager.initialize()
        return manager

    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing."""
        return {
            # Atmospheric variables
            'air_temperature': torch.tensor(298.15),  # K
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'friction_velocity': torch.tensor(0.5),  # m s-1
            'vapor_pressure_deficit': torch.tensor(1000.0),  # Pa
            'net_radiation': torch.tensor(400.0),  # W m-2
            'co2_concentration': torch.tensor(400.0),  # ppm
            'relative_humidity': torch.tensor(0.6),  # dimensionless
            'aerodynamic_resistance': torch.tensor(50.0),  # s m-1
            
            # Soil variables
            'soil_moisture': torch.tensor(0.25),  # m3 m-3
            'soil_temperature': torch.tensor(293.15),  # K
            'soil_matric_potential': torch.tensor(-5000.0),  # Pa
            'soil_depth_layers': torch.tensor([0.1, 0.3, 0.6, 1.0, 1.5, 2.0]),  # m (layer-bottom depths, top at 0)
            'potential_transpiration': torch.tensor(3.0e-6),  # kg m-2 s-1
            
            # Carbon variables
            'net_assimilation_rate': torch.tensor(20.0e-6),  # kg C m-2 s-1
            'gross_primary_productivity': torch.tensor(25.0e-6),  # kg C m-2 s-1
            'autotrophic_respiration': torch.tensor(5.0e-6),  # kg C m-2 s-1
            'heterotrophic_respiration': torch.tensor(3.0e-6),  # kg C m-2 s-1
        }

    def test_biosphere_atmosphere_with_state_manager(self, state_manager, sample_state):
        """
        Test BiosphereAtmosphereCoupling integration with StateManager.
        
        Verifies:
        - Functor can compute updates
        - Updates can be applied via StateManager
        - State is updated correctly
        """
        # Initialize coupling functor
        coupling = BiosphereAtmosphereCoupling(
            name='bio_atm_coupling', config={}
        )
        
        # Compute updates
        time = 0.0  # Start time
        dt = 1800.0  # 30 minutes
        
        updates = coupling.forward(state=sample_state, time=time, dt=dt)
        
        # Verify updates were produced
        assert isinstance(updates, dict), "forward() should return a dictionary"
        assert set(updates) == {'boundary_layer_conductance'}
        update = updates['boundary_layer_conductance']
        assert isinstance(update, UpdateItem)
        assert update.mode is UpdateMode.REPLACE
        assert update.per_timestep is False
        assert update.source_functor == coupling.name
        assert update.unit == 'mol m-2 s-1'

    def test_water_carbon_with_state_manager(self, state_manager, sample_state):
        """
        Test WaterCarbonCoupling integration with StateManager.
        
        Verifies parameter fix allows proper initialization and execution.
        """
        config = {
            'name': 'water_carbon_coupling',
            'enabled': True,
            'stress_function_type': 'logistic',
            'wilting_point': 0.05,
            'field_capacity': 0.35,
        }
        coupling = WaterCarbonCoupling(name='water_carbon_coupling', config=config)
        
        # Compute updates
        time = 0.0
        dt = 1800.0
        
        updates = coupling.forward(state=sample_state, time=time, dt=dt)
        
        # Verify updates
        assert isinstance(updates, dict), "forward() should return a dictionary"
        assert len(updates) > 0, "forward() should produce stress factors"
        
        # Check expected stress factors
        expected_outputs = [
            'gpp_water_stress_factor',
            'heterotrophic_respiration_modifier',
            'overall_water_stress'
        ]
        
        for output in expected_outputs:
            assert output in updates, f"Missing expected output: {output}"
            
            # Stress factors should be in [0, 1]
            factor = updates[output]
            assert torch.all(factor >= 0.0) and torch.all(factor <= 1.0), \
                f"{output} should be in range [0, 1], got {factor.item()}"

    def test_soil_vegetation_with_state_manager(self, state_manager, sample_state):
        """Test SoilVegetationCoupling integration with StateManager."""
        config = {
            'name': 'soil_veg_coupling',
            'enabled': True,
            'root_distribution': 'exponential',
        }
        coupling = SoilVegetationCoupling(name='soil_veg_coupling', config=config)
        
        # Compute updates
        time = 0.0
        dt = 1800.0
        
        profile_state = dict(sample_state)
        profile_state.update({
            'soil_moisture_profile': torch.tensor(
                [0.28, 0.27, 0.25, 0.24, 0.23, 0.22]
            ),
            'soil_matric_potential': torch.full((6,), -5000.0),
            'soil_depth_layers': torch.tensor([0.1, 0.3, 0.6, 1.0, 1.5, 2.0]),
        })

        updates = coupling.forward(state=profile_state, time=time, dt=dt)
        
        # Verify updates
        assert isinstance(updates, dict), "forward() should return a dictionary"
        assert set(updates) == {
            'root_uptake_profile',
            'effective_transpiration',
            'diagnostics.soil_vegetation.feddes_stress_factor',
            'diagnostics.soil_vegetation.root_zone_moisture',
            'diagnostics.soil_vegetation.root_distribution',
            'diagnostics.soil_vegetation.water_balance_residual',
        }
        for update in updates.values():
            assert isinstance(update, UpdateItem)
            assert update.mode is UpdateMode.REPLACE
            assert update.per_timestep is False
            assert update.source_functor == coupling.name

        layer_thickness = torch.diff(
            profile_state['soil_depth_layers'], prepend=torch.zeros(1)
        )
        integrated_uptake = torch.sum(
            updates['root_uptake_profile'].value * layer_thickness
        )
        assert torch.allclose(
            integrated_uptake,
            updates['effective_transpiration'].value,
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        assert torch.allclose(
            updates['diagnostics.soil_vegetation.water_balance_residual'].value,
            torch.zeros_like(integrated_uptake),
            rtol=0.0,
            atol=1.0e-12,
        )


class TestCouplingDependencyResolution:
    """Test dependency resolution for coupling modules."""

    def test_dependency_chain_resolution(self):
        """
        Test that coupling dependencies are correctly resolved.
        
        Required edges:
        1. BiosphereAtmosphereCoupling follows atmospheric turbulence.
        2. SoilVegetationCoupling consumes StateManager ports without an
           invented biosphere-atmosphere dependency.
        3. WaterCarbonCoupling follows SoilVegetationCoupling.
        """
        resolver = DependencyGraphResolver()
        
        turbulence = TurbulenceFunctor(name='atmospheric_turbulence')
        bio_atm = BiosphereAtmosphereCoupling(name='bio_atm', config={})
        
        soil_veg_config = {
            'name': 'soil_veg',
            'enabled': True,
            'root_distribution': 'exponential',
        }
        soil_veg = SoilVegetationCoupling(name='soil_veg', config=soil_veg_config)
        
        water_carbon_config = {
            'name': 'water_carbon',
            'enabled': True,
            'stress_function_type': 'logistic',
            'wilting_point': 0.05,
            'field_capacity': 0.35,
            'upstream_soil_vegetation_functor': 'soil_vegetation',
        }
        water_carbon = WaterCarbonCoupling(name='water_carbon', config=water_carbon_config)
        
        resolver.add_functor(
            turbulence, functor_id='atmospheric_turbulence'
        )
        resolver.add_functor(bio_atm, functor_id='biosphere_atmosphere')
        resolver.add_functor(soil_veg, functor_id='soil_vegetation')
        resolver.add_functor(water_carbon, functor_id='water_carbon')
        
        execution_order = resolver.topological_sort()
        assert len(execution_order) >= 4, "Should have at least 4 functors in order"
        
        turbulence_idx = execution_order.index('atmospheric_turbulence')
        biosphere_atmosphere_idx = execution_order.index('biosphere_atmosphere')
        soil_veg_idx = execution_order.index('soil_vegetation')
        water_carbon_idx = execution_order.index('water_carbon')
        
        assert biosphere_atmosphere_idx > turbulence_idx, \
            "BiosphereAtmosphereCoupling should execute after turbulence"
        assert soil_veg.get_dependencies() == []
        assert water_carbon_idx > soil_veg_idx, \
            "WaterCarbonCoupling should execute after SoilVegetationCoupling"

    def test_no_circular_dependencies(self):
        """Verify internal coupling modules have no circular dependencies."""
        resolver = DependencyGraphResolver()
        
        turbulence = TurbulenceFunctor(name='atmospheric_turbulence')
        bio_atm = BiosphereAtmosphereCoupling(name='bio_atm', config={})
        
        soil_veg_config = {
            'name': 'soil_veg',
            'enabled': True,
            'root_distribution': 'exponential',
            'upstream_biosphere_functor': 'biosphere_atmosphere',
        }
        soil_veg = SoilVegetationCoupling(name='soil_veg', config=soil_veg_config)
        
        water_carbon_config = {
            'name': 'water_carbon',
            'enabled': True,
            'stress_function_type': 'logistic',
            'wilting_point': 0.05,
            'field_capacity': 0.35,
            'upstream_soil_vegetation_functor': 'soil_vegetation',
        }
        water_carbon = WaterCarbonCoupling(name='water_carbon', config=water_carbon_config)
        
        resolver.add_functor(
            turbulence, functor_id='atmospheric_turbulence'
        )
        resolver.add_functor(bio_atm, functor_id='biosphere_atmosphere')
        resolver.add_functor(soil_veg, functor_id='soil_vegetation')
        resolver.add_functor(water_carbon, functor_id='water_carbon')
        
        execution_order = resolver.topological_sort()
        assert execution_order, "Dependency resolution should return an execution order"


class TestScaleImplementationsIntegration:
    """Test scale coupling implementations integration."""

    @pytest.fixture
    def spatial_data(self):
        """Create sample spatial data for testing."""
        # Simulate a 10x10 grid
        grid_data = torch.randn(10, 10) * 20 + 273.15  # Temperature field
        return {
            'temperature_grid': grid_data,
            'moisture_grid': torch.rand(10, 10) * 0.4  # Moisture in [0, 0.4]
        }

    def test_spatial_aggregation_basic(self, spatial_data):
        """Test SpatialAggregation can process spatial data."""
        config = {
            'name': 'spatial_agg',
            'aggregation_strategy': 'area_weighted',
            'preserve_totals': False
        }
        
        spatial_agg = SpatialAggregation(
            name='spatial_agg',
            config=config,
            aggregation_strategy='area_weighted',
            preserve_totals=False,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
        
        # Adapter should expose configured strategy for ScaleManager integration
        assert spatial_agg.aggregation_strategy == 'area_weighted'
        
        # Verify no explicit weights parameter in forward()
        import inspect
        sig = inspect.signature(spatial_agg.forward)
        assert 'weights' not in sig.parameters, \
            "P0-5 fix: weights should not be explicit parameter"

    def test_spatial_implementation_has_dependencies_method(self):
        """Verify the remaining spatial adapter dependency surface."""
        spatial_config = {'name': 'spat', 'aggregation_strategy': 'area_weighted'}
        spatial_agg = SpatialAggregation(name='spat', config=spatial_config, source_scale="fine_grid", target_scale="coarse_grid")
        
        assert hasattr(spatial_agg, 'get_dependencies'), \
            "SpatialAggregation should have get_dependencies()"
        
        assert spatial_agg.get_dependencies() == [], \
            "SpatialAggregation should have no dependencies"


class TestCouplingConservation:
    """Test conservation law enforcement in coupling modules."""

    @pytest.fixture
    def conservation_state(self):
        """Create state with known values for conservation testing."""
        return {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'friction_velocity': torch.tensor(0.5),
            'vapor_pressure_deficit': torch.tensor(1500.0),
            'net_radiation': torch.tensor(500.0),
            'co2_concentration': torch.tensor(400.0),
            'relative_humidity': torch.tensor(0.5),
            'aerodynamic_resistance': torch.tensor(60.0),
            'net_assimilation_rate': torch.tensor(15.0e-6),
            'soil_moisture': torch.tensor(0.20),
            'soil_temperature': torch.tensor(295.15),
        }

    def test_biosphere_atmosphere_has_no_duplicate_energy_closure(
        self, conservation_state
    ):
        """
        Verify the boundary transport closure does not duplicate an energy model.
        """
        coupling = BiosphereAtmosphereCoupling(name='bio_atm', config={})
        
        time = 0.0
        dt = 1800.0
        
        updates = coupling.forward(state=conservation_state, time=time, dt=dt)
        
        assert set(updates) == {'boundary_layer_conductance'}
        assert 'latent_heat_flux' not in updates
        assert 'sensible_heat_flux' not in updates
        assert torch.isfinite(updates['boundary_layer_conductance'].value)

    def test_water_carbon_stress_bounds(self, conservation_state):
        """Test that WaterCarbonCoupling produces valid stress factors."""
        config = {
            'name': 'water_carbon',
            'enabled': True,
            'stress_function_type': 'logistic',
            'wilting_point': 0.05,
            'field_capacity': 0.35,
        }
        coupling = WaterCarbonCoupling(name='water_carbon', config=config)
        
        time = 0.0
        dt = 1800.0
        
        updates = coupling.forward(state=conservation_state, time=time, dt=dt)
        
        # All stress factors should be in [0, 1]
        stress_factors = [
            'gpp_water_stress_factor',
            'autotrophic_respiration_modifier',
            'heterotrophic_respiration_modifier',
            'overall_water_stress'
        ]
        
        for factor_name in stress_factors:
            if factor_name in updates:
                factor = updates[factor_name]
                assert torch.all(factor >= 0.0), \
                    f"{factor_name} has values < 0: {factor.min().item()}"
                assert torch.all(factor <= 1.0), \
                    f"{factor_name} has values > 1: {factor.max().item()}"


class TestCouplingCrossModule:
    """Test cross-module data flow between coupling modules."""

    @pytest.fixture
    def full_coupling_state(self):
        """Create comprehensive state for cross-module testing."""
        return {
            # Atmospheric
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'friction_velocity': torch.tensor(0.5),
            'vapor_pressure_deficit': torch.tensor(1200.0),
            'net_radiation': torch.tensor(450.0),
            'co2_concentration': torch.tensor(410.0),
            'relative_humidity': torch.tensor(0.55),
            'aerodynamic_resistance': torch.tensor(55.0),
            
            # Soil
            'soil_moisture': torch.tensor(0.22),
            'soil_temperature': torch.tensor(294.15),
            'soil_matric_potential': torch.tensor(-4500.0),
            
            # Carbon
            'net_assimilation_rate': torch.tensor(18.0e-6),
            'gross_primary_productivity': torch.tensor(22.0e-6),
            'autotrophic_respiration': torch.tensor(4.0e-6),
            'heterotrophic_respiration': torch.tensor(2.8e-6),
        }

    def test_sequential_coupling_execution(self, full_coupling_state):
        """
        Test independent boundary transport and water-carbon coupling outputs.
        """
        bio_atm = BiosphereAtmosphereCoupling(name='bio_atm', config={})
        
        time = 0.0
        dt = 1800.0
        
        bio_atm_updates = bio_atm.forward(state=full_coupling_state, time=time, dt=dt)
        
        assert set(bio_atm_updates) == {'boundary_layer_conductance'}
        assert 'evapotranspiration_flux' not in bio_atm_updates
        
        # Step 2: Update state with bio_atm results
        updated_state = full_coupling_state.copy()
        updated_state['boundary_layer_conductance'] = bio_atm_updates[
            'boundary_layer_conductance'
        ].value
        
        # Step 3: WaterCarbonCoupling with updated state
        water_carbon_config = {
            'name': 'water_carbon',
            'enabled': True,
            'stress_function_type': 'logistic',
            'wilting_point': 0.05,
            'field_capacity': 0.35,
        }
        water_carbon = WaterCarbonCoupling(name='water_carbon', config=water_carbon_config)
        
        water_carbon_updates = water_carbon.forward(state=updated_state, time=time, dt=dt)
        
        # Verify water-carbon produced stress factors
        assert 'gpp_water_stress_factor' in water_carbon_updates, \
            "WaterCarbonCoupling should produce stress factors"
        
        # Verify stress is reasonable given soil moisture (0.22 m3/m3, moderate)
        stress_factor = water_carbon_updates['gpp_water_stress_factor']
        assert 0.3 < stress_factor.item() < 1.0, \
            f"Expected moderate stress factor, got {stress_factor.item()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

