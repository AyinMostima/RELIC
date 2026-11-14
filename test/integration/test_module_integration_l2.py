"""
L2 Integration Tests - Module Integration Verification
Tests that modules work correctly together after refactoring
"""

import unittest
import pytest
import torch

from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.core.dependency_graph import DependencyGraphResolver
from relic.parameters import get_global_manager


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


class TestModuleIntegration(unittest.TestCase):
    """Test module integration and coupling"""

    def setUp(self):
        """Set up test environment"""
        self.state_manager = StateManager()
        self.resolver = DependencyGraphResolver()
        
    def test_photosynthesis_respiration_coupling(self):
        """Test canonical photosynthesis and respiration carbon flux ports."""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
        from relic.scientific.biosphere.respiration import MaintenanceRespiration
        
        # Initialize modules
        photo = FarquharPhotosynthesis()
        resp = MaintenanceRespiration()
        
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'biomass_leaf': torch.tensor(50.0),
            'biomass_stem': torch.tensor(30.0),
            'biomass_root': torch.tensor(20.0),
            'soluble_sugar': torch.tensor(5.0),
            'soil_water_potential': torch.tensor(-0.5),
            'growing_temperature': torch.tensor(298.15),
        }
        
        # Run photosynthesis
        photo_flux = photo.forward(state, time=0.0, dt=3600.0)
        self.assertIn('gross_primary_productivity', photo_flux)
        gpp_update = photo_flux['gross_primary_productivity']
        self.assertIsInstance(gpp_update, UpdateItem)
        self.assertEqual(gpp_update.unit, 'kg C m-2 s-1')
        state['gross_primary_productivity'] = gpp_update.value

        resp_flux = resp.forward(state, time=0.0, dt=3600.0)
        self.assertIn('maintenance_respiration_total', resp_flux)
        respiration_update = resp_flux['maintenance_respiration_total']
        self.assertIsInstance(respiration_update, UpdateItem)
        self.assertEqual(respiration_update.unit, 'kg C m-2 s-1')
        
    def test_state_manager_update_application(self):
        """Test StateManager correctly applies updates"""
        # Initialize state
        initial_state = {
            'carbon_pool': torch.tensor(100.0),
            'water_content': torch.tensor(0.3),
            'nitrogen_pool': torch.tensor(10.0)
        }
        
        self.state_manager.initialize_state(initial_state)
        
        # Create updates
        updates = {
            'carbon_pool': torch.tensor(5.0),  # Add 5 units
            'water_content': torch.tensor(-0.05),  # Remove 0.05
            'nitrogen_pool': torch.tensor(0.5)  # Add 0.5
        }
        
        # Apply updates
        self.state_manager.apply_updates(updates, dt=3600.0, check_conservation=False)
        
        # Check state was updated correctly
        state = self.state_manager.get_state()
        self.assertAlmostEqual(state['carbon_pool'].item(), 105.0, places=5)
        self.assertAlmostEqual(state['water_content'].item(), 0.25, places=5)
        self.assertAlmostEqual(state['nitrogen_pool'].item(), 10.5, places=5)
        
    def test_dependency_resolution(self):
        """Test DependencyGraphResolver correctly orders functors"""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
        from relic.scientific.biosphere.respiration import MaintenanceRespiration
        from relic.scientific.biosphere.growth import BiomassAllocation
        from relic.core.functor import FunctorBase, FunctorMetadata

        class DummyCarbonPools(FunctorBase):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="pedosphere_carbon_pools",
                    description="Dummy carbon pools functor for dependency test",
                    references=["unit-test"],
                    input_variables=["dependency_fixture_input"],
                )
                super().__init__(
                    name="pedosphere_carbon_pools",
                    config={},
                    metadata=metadata,
                )

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {}

        class DummyMeteorology(FunctorBase):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="atmosphere_meteorology",
                    description="Dummy meteorology functor for dependency test",
                    references=["unit-test"],
                    input_variables=["dependency_fixture_input"],
                )
                super().__init__(
                    name="atmosphere_meteorology",
                    config={},
                    metadata=metadata,
                )

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {}

        class DummyDependencySource(FunctorBase):
            def __init__(self, name):
                metadata = FunctorMetadata(
                    name=name,
                    description="Required upstream source for dependency test",
                    references=["unit-test"],
                    input_variables=["dependency_fixture_input"],
                )
                super().__init__(name=name, config={}, metadata=metadata)

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {}

        # Create functors
        functors = [
            FarquharPhotosynthesis(),
            DummyCarbonPools(),
            DummyMeteorology(),
            DummyDependencySource("environmental_forcing"),
            DummyDependencySource("atmospheric_radiation"),
            DummyDependencySource("leaf_temperature_calculator"),
            DummyDependencySource("autotrophic_respiration"),
            DummyDependencySource("water_stress"),
            DummyDependencySource("gsi_phenology"),
            MaintenanceRespiration(),
            BiomassAllocation()
        ]
        
        # Resolve dependencies
        for functor in functors:
            if isinstance(functor, DummyCarbonPools):
                self.resolver.add_functor(functor, functor_id="pedosphere.carbon_pools")
            elif isinstance(functor, DummyMeteorology):
                self.resolver.add_functor(functor, functor_id="atmosphere.meteorology")
            else:
                self.resolver.add_functor(functor)
        groups = self.resolver.topological_sort()
        
        # Should have at least one group
        self.assertTrue(len(groups) > 0)
        
        # Verify no cycles detected
        self.assertFalse(self.resolver.get_execution_stats()["has_cycles"])
        
    def test_parameter_flow_from_yaml(self):
        """Test parameters flow correctly from YAML to modules"""
        from relic.scientific.atmosphere.chemistry import ChemistryFunctor
        
        # Create module
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # A parameter declared in atmosphere/chemistry.yaml reaches the module.
        rate = chem.get_parameter_value(
            'k_no_o3_298',
            units='cm3 molecule-1 s-1',
            namespace='atmosphere.chemistry',
            required=True,
        )
        self.assertIsInstance(rate, (float, int, torch.Tensor))
        # A key the YAML does not declare is None: there is no caller default.
        self.assertIsNone(
            chem.get_parameter_value('oh_reaction_rate', namespace='atmosphere.chemistry')
        )
        
    def test_conservation_check_integration(self):
        """Test conservation checks work across modules"""
        from relic.scientific.pedosphere.decomposition import FirstOrderDecomposition
        
        # Initialize
        decomp = FirstOrderDecomposition()
        
        # Create state with carbon pools
        state = {
            'metabolic_litter_c': torch.tensor(50.0),
            'structural_litter_c': torch.tensor(50.0),
            'fast_som_c': torch.tensor(500.0),
            'slow_som_c': torch.tensor(400.0),
            'passive_som_c': torch.tensor(300.0),
            'soil_temperature': torch.tensor(298.15),
            'soil_moisture': torch.tensor(0.3),
            'metabolic_litter_input': torch.tensor(8.5e-7),
            'structural_litter_input': torch.tensor(1.5e-7),
            'lignin_n_ratio': torch.tensor(0.3),
            'clay_content': torch.tensor(0.2),
        }
        
        # Run decomposition
        flux = decomp.forward(state, time=0.0, dt=3600.0)
        
        # Check carbon is conserved (flux out of litter into soil + CO2)
        if 'heterotrophic_respiration' in flux:
            total_flux = flux['heterotrophic_respiration']
            self.assertIsInstance(total_flux, UpdateItem)
            self.assertIs(total_flux.mode, UpdateMode.REPLACE)
            self.assertFalse(total_flux.per_timestep)
            self.assertEqual(total_flux.unit, 'g C m-2 s-1')
            self.assertGreaterEqual(total_flux.value.item(), 0)
        violations = decomp.check_conservation_laws(
            state,
            flux,
            dt=3600.0,
        )
        self.assertLessEqual(violations['carbon'], 2.0e-9)
            

class TestCouplingLayerIntegration(unittest.TestCase):
    """Test coupling layer integration"""
    
    def test_biosphere_atmosphere_coupling(self):
        """Test canonical boundary conductance through the coupling facade."""
        from relic.coupling.internal import (
            BiosphereAtmosphereCoupling,
            get_default_configs,
        )

        param_manager = get_global_manager()
        param_manager.load_from_database("relic/parameters/database")
        config = get_default_configs(parameter_manager=param_manager).get('biosphere_atmosphere', {})
        self.assertEqual(config, {})

        coupling = BiosphereAtmosphereCoupling(
            name="biosphere_atmosphere",
            config=config,
            parameter_manager=param_manager,
        )
        result = coupling.forward(
            {
                "friction_velocity": torch.tensor(0.5),
                "air_temperature": torch.tensor(298.15),
                "atmospheric_pressure": torch.tensor(101325.0),
            },
            time=0.0,
            dt=60.0,
        )
        self.assertEqual(set(result), {"boundary_layer_conductance"})
        self.assertIsInstance(result["boundary_layer_conductance"], UpdateItem)
        
    def test_soil_vegetation_coupling(self):
        """Test soil-vegetation coupling configuration"""
        from relic.coupling.internal import get_default_configs

        param_manager = get_global_manager()
        param_manager.load_from_database("relic/parameters/database")
        config = get_default_configs(parameter_manager=param_manager).get('soil_vegetation', {})
        root_distribution = config.get('root_distribution', {})
        water_stress = config.get('water_stress', {})

        self.assertEqual(
            set(root_distribution),
            {'distribution_type', 'root_depth_max', 'shape_parameter'},
        )
        self.assertEqual(
            set(water_stress),
            {'wilting_point_potential', 'optimal_range', 'anaerobic_point'},
        )
        optimal_dry, optimal_wet = water_stress['optimal_range']
        self.assertLess(water_stress['wilting_point_potential'], optimal_dry)
        self.assertLess(optimal_dry, optimal_wet)
        self.assertLess(optimal_wet, water_stress['anaerobic_point'])
        

if __name__ == '__main__':
    unittest.main()

