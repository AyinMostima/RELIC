#!/usr/bin/env python
"""
L2 Integration Tests for RELIC
Tests integration between modules and cross-module functionality
"""

import unittest
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode
from relic.core.dependency_graph import DependencyGraphResolver
from relic.parameters import get_global_manager
from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.coupling.internal.water_carbon import WaterCarbonCoupling
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.atmosphere.environmental_forcing import EnvironmentalForcingFunctor
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.utils.physical_constants import get_physical_constant


class TestCouplingIntegration(unittest.TestCase):
    """Test coupling between different system components"""

    def setUp(self):
        """Set up test environment"""
        self.param_manager = get_global_manager()
        self.device = torch.device('cpu')

    def test_biosphere_atmosphere_coupling(self):
        """Test biosphere-atmosphere coupling integration"""
        # Create coupling instance
        coupling = BiosphereAtmosphereCoupling(
            name="test_biosphere_atmosphere",
            config={}
        )

        # Create realistic state
        state = {
            'friction_velocity': torch.tensor(0.5),  # m s-1
            'air_temperature': torch.tensor(293.15),  # K
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
        }

        # Run coupling
        fluxes = coupling.forward(state, time=0.0, dt=3600.0)

        # Check outputs exist and are reasonable
        self.assertEqual(set(fluxes), {'boundary_layer_conductance'})
        update = fluxes['boundary_layer_conductance']
        self.assertIsInstance(update, UpdateItem)
        self.assertEqual(update.unit, 'mol m-2 s-1')
        self.assertTrue(torch.all(update.value > 0))
        self.assertTrue(torch.all(torch.isfinite(update.value)))

    def test_soil_vegetation_coupling(self):
        """Test soil-vegetation coupling integration"""
        # Create coupling instance
        coupling = SoilVegetationCoupling(
            name="test_soil_vegetation",
            config={}
        )

        # Create the canonical layer-resolved water state and demand port.
        state = {
            'soil_moisture_profile': torch.tensor([0.28, 0.25, 0.22]),
            'soil_matric_potential': torch.tensor([-5000.0, -6000.0, -7000.0]),
            'potential_transpiration': torch.tensor(3.0e-6),
            'soil_depth_layers': torch.tensor([0.1, 0.3, 0.6]),
        }

        # Run coupling
        fluxes = coupling.forward(state, time=0.0, dt=3600.0)

        self.assertEqual(set(fluxes), {
            'root_uptake_profile',
            'effective_transpiration',
            'diagnostics.soil_vegetation.feddes_stress_factor',
            'diagnostics.soil_vegetation.root_zone_moisture',
            'diagnostics.soil_vegetation.root_distribution',
            'diagnostics.soil_vegetation.water_balance_residual',
        })
        self.assertTrue(all(isinstance(value, UpdateItem) for value in fluxes.values()))
        layer_thickness = torch.diff(
            state['soil_depth_layers'], prepend=torch.zeros(1)
        )
        integrated_uptake = torch.sum(
            fluxes['root_uptake_profile'].value * layer_thickness
        )
        self.assertTrue(torch.allclose(
            integrated_uptake,
            fluxes['effective_transpiration'].value,
            rtol=1.0e-6,
            atol=1.0e-12,
        ))

    def test_water_carbon_coupling(self):
        """Test water-carbon coupling integration"""
        # Create coupling instance
        coupling = WaterCarbonCoupling(
            name="test_water_carbon",
            config={}
        )

        # Create state
        state = {
            'soil_moisture': torch.tensor(0.3),
            'soil_temperature': torch.tensor(293.15),
            'vapor_pressure_deficit': torch.tensor(800.0),  # Pa, required stress input
            'soil_carbon': torch.tensor(10.0),  # kg C m-2
            'microbial_biomass': torch.tensor(0.1),  # kg C m-2
            'litter_carbon': torch.tensor(1.0),  # kg C m-2
        }

        # Run coupling
        fluxes = coupling.forward(state, time=0.0, dt=3600.0)

        # Check outputs
        self.assertIn('decomposition_rate_modifier', fluxes)
        # The coupling emits dimensionless modifiers; the soil respiration flux is the
        # pedosphere's heterotrophic_respiration (no second respiration flux).
        self.assertNotIn('respiration_flux', fluxes)
        self.assertIn('heterotrophic_respiration_modifier', fluxes)

        # Check constraints
        self.assertTrue(torch.all(fluxes['decomposition_rate_modifier'] >= 0))
        self.assertTrue(torch.all(fluxes['heterotrophic_respiration_modifier'] >= 0))


class TestModuleChaining(unittest.TestCase):
    """Test chaining of multiple scientific modules"""

    def setUp(self):
        """Set up test environment"""
        self.param_manager = get_global_manager()
        self.state_manager = StateManager()

    def test_radiation_photosynthesis_chain(self):
        """Test radiation -> photosynthesis module chain"""
        # Create modules
        radiation = RadiationFunctor(name="radiation", config={})

        photosynthesis = FarquharPhotosynthesis(
            name="photosynthesis",
            config={}
        )

        # Initial state
        state = {
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(297.15),
            'incoming_shortwave_potential': torch.tensor(1000.0),
            'cos_zenith': torch.tensor(0.8660254),
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'relative_humidity': torch.tensor(0.7),
            'solar_zenith_angle': torch.tensor(30.0),
            'day_of_year': torch.tensor(180.0),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'leaf_area_index': torch.tensor(4.0),
        }

        # Run radiation first
        radiation_fluxes = radiation.forward(state, time=43200.0, dt=3600.0)  # Noon

        # Update state with radiation outputs
        state.update({
            'ppfd': radiation_fluxes['ppfd'].value,
            'net_radiation': radiation_fluxes['net_radiation'].value,
        })

        # Run photosynthesis
        photosynthesis_fluxes = photosynthesis.forward(state, time=43200.0, dt=3600.0)

        # Check chain results
        self.assertIn('net_photosynthesis', photosynthesis_fluxes)
        self.assertTrue(torch.all(
            photosynthesis_fluxes['net_photosynthesis'].value >= 0
        ))

    def test_evapotranspiration_soil_moisture_chain(self):
        """Test ET -> soil moisture update chain"""
        # Create modules
        et_module = PenmanMonteithET(
            name="penman_monteith",
            config={"canopy_resistance_source": "state"}
        )

        soil_module = RichardsSoilMoisture(name="richards")

        # Initial state
        state = {
            'net_radiation': torch.tensor(400.0),
            'air_temperature': torch.tensor(293.15),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'wind_speed': torch.tensor(2.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'surface_resistance': torch.tensor(100.0),
            'canopy_height': torch.tensor(2.0),
            'leaf_area_index': torch.tensor(3.0),
            'soil_moisture': torch.full(
                (soil_module.soil_layers,), 0.3, dtype=torch.float32
            ),
            'precipitation_flux': torch.tensor(0.0),
        }

        # Run ET
        et_fluxes = et_module.forward(state, time=0.0, dt=3600.0)

        # Update state with ET output
        evapotranspiration = et_fluxes['evapotranspiration_flux']
        if isinstance(evapotranspiration, UpdateItem):
            evapotranspiration = evapotranspiration.value
        state['infiltration_flux'] = state['precipitation_flux']
        state['evapotranspiration_flux'] = evapotranspiration

        # Run soil moisture
        soil_fluxes = soil_module.forward(state, time=0.0, dt=3600.0)

        # Check water balance
        change_key = (
            'diagnostics.soil_moisture_richards.soil_moisture_change_rate'
        )
        self.assertIn(change_key, soil_fluxes)
        # ET should reduce soil moisture
        soil_moisture_change = soil_fluxes[change_key]
        if isinstance(soil_moisture_change, UpdateItem):
            soil_moisture_change = soil_moisture_change.value
        self.assertTrue(torch.any(soil_moisture_change <= 0))


class TestParameterSystemIntegration(unittest.TestCase):
    """Test parameter system integration across modules"""

    def setUp(self):
        """Set up test environment"""
        self.param_manager = get_global_manager()

    def test_parameter_consistency_across_modules(self):
        """Test that shared parameters are consistent across modules"""
        # Physical constants should be the same everywhere
        modules = [
            BiosphereAtmosphereCoupling("test1", {}),
            SoilVegetationCoupling("test2", {}),
            WaterCarbonCoupling("test3", {}),
        ]

        expected_gravity = torch.as_tensor(
            get_physical_constant("gravity_acceleration"),
            dtype=torch.float32,
        )

        # Check that all modules see the same SSOT physical constants.
        for module in modules:
            gravity = module.get_parameter_value(
                "relic.physical.constants.earth.gravity_acceleration",
                required=True,
            )
            gravity = torch.as_tensor(gravity, dtype=torch.float32)
            self.assertTrue(
                torch.allclose(gravity, expected_gravity, atol=1e-6, rtol=0.0),
                "Gravity constant should be resolved from the SSOT parameter database",
            )

    def test_yaml_parameter_retrieval_across_modules(self):
        """Test canonical YAML parameter retrieval across bridge classes."""
        parameter_key = "hydrosphere.soil_moisture.soil_depth"
        expected = self.param_manager.get(parameter_key, required=True)
        spec = self.param_manager.get_parameter(parameter_key)

        self.assertIsNotNone(spec)
        self.assertTrue(spec.references)
        self.assertTrue(spec.dimension)
        self.assertIsNotNone(spec.bounds)

        module1 = BiosphereAtmosphereCoupling("test1", {})
        module2 = SoilVegetationCoupling("test2", {})

        val1 = module1.get_parameter_value(parameter_key, required=True)
        val2 = module2.get_parameter_value(parameter_key, required=True)

        self.assertEqual(val1, val2, "Parameter should be consistent across modules")
        self.assertEqual(val1, expected, "Parameter value should match YAML SSOT")


class TestStateManagement(unittest.TestCase):
    """Test state management across module integration"""

    def setUp(self):
        """Set up test environment"""
        self.state_manager = StateManager()

    def test_state_update_propagation(self):
        """Test that state updates propagate correctly"""
        # Initialize state
        initial_state = {
            'air_temperature': torch.tensor(293.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'friction_velocity': torch.tensor(0.5),
            'soil_moisture': torch.tensor(0.25),
            'leaf_area_index': torch.tensor(3.0),
            'aerodynamic_resistance': torch.tensor(50.0),
            'net_assimilation_rate': torch.tensor(1.0e-6),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'net_radiation': torch.tensor(400.0),
            'co2_concentration': torch.tensor(400.0),
            'soil_moisture_profile': torch.tensor([0.28, 0.25, 0.22]),
            'soil_matric_potential': torch.tensor([-5000.0, -6000.0, -7000.0]),
            'potential_transpiration': torch.tensor(3.0e-6),
            'soil_depth_layers': torch.tensor([0.1, 0.3, 0.6]),
        }

        self.state_manager.initialize_state(initial_state)

        # Create modules that modify state
        module1 = BiosphereAtmosphereCoupling("test1", {})
        module2 = SoilVegetationCoupling("test2", {})

        # Get current state
        state = self.state_manager.get_state()

        # Run modules and collect updates
        fluxes1 = module1.forward(state, time=0.0, dt=3600.0)
        fluxes2 = module2.forward(state, time=0.0, dt=3600.0)

        # Apply updates
        combined_updates = {**fluxes1, **fluxes2}
        self.state_manager.apply_updates(combined_updates, dt=3600.0)

        # Check state was updated
        new_state = self.state_manager.get_state()
        # At least some values should have changed
        self.assertTrue(
            any(k in new_state for k in combined_updates.keys()),
            "State should contain some updates"
        )

    def test_conservation_checking_integration(self):
        """Test conservation checking across module boundaries"""
        schema = {
            name: StateVariableSpec(
                name=name,
                units='kg C m-2',
                conserved=True,
                required=False,
            )
            for name in ('vegetation_carbon', 'soil_carbon')
        }
        state_manager = StateManager(schema=schema)
        initial_state = {
            'vegetation_carbon': torch.tensor(40.0),
            'soil_carbon': torch.tensor(60.0),
        }

        state_manager.initialize_state(initial_state)
        state_manager.set_conservation_ledger(
            {'carbon': ['vegetation_carbon', 'soil_carbon']}
        )

        updates = {
            'vegetation_carbon': UpdateItem(
                value=torch.tensor(5.0),
                mode=UpdateMode.ACCUMULATE,
                unit='kg C m-2',
                per_timestep=True,
            ),
            'soil_carbon': UpdateItem(
                value=torch.tensor(-5.0),
                mode=UpdateMode.ACCUMULATE,
                unit='kg C m-2',
                per_timestep=True,
            ),
        }

        state_manager.apply_updates(
            updates,
            dt=3600.0,
            check_conservation=True
        )

        report = state_manager.generate_conservation_report()
        self.assertIsNotNone(report)


class TestDependencyResolution(unittest.TestCase):
    """Test dependency resolution for module integration"""

    def setUp(self):
        """Set up test environment"""
        self.resolver = DependencyGraphResolver()

    def test_coupling_dependency_resolution(self):
        """Test that coupling modules resolve dependencies correctly"""
        # Create modules with dependencies
        modules = {
            'environmental_forcing': EnvironmentalForcingFunctor("environmental_forcing", {}),
            'atmospheric_radiation': RadiationFunctor("atmospheric_radiation", {}),
            'atmospheric_turbulence': TurbulenceFunctor("atmospheric_turbulence", {}),
            'farquhar_photosynthesis': FarquharPhotosynthesis("farquhar_photosynthesis", {}),
            'biosphere_atmosphere': BiosphereAtmosphereCoupling("biosphere_atmosphere", {}),
        }

        # Build dependency graph
        for name, module in modules.items():
            self.resolver.add_functor(module)

        # Resolve execution order
        execution_groups = self.resolver.topological_sort()
        self.assertIsNotNone(execution_groups)
        for group in execution_groups:
            self.assertTrue(len(group) > 0)

    def test_parallel_execution_identification(self):
        """Test identification of modules that can run in parallel"""
        # Create independent modules
        modules = [
            RadiationFunctor("radiation", {}),
            TurbulenceFunctor(),
        ]

        for module in modules:
            self.resolver.add_functor(module)

        execution_groups = self.resolver.topological_sort()

        # These should be in the same group (can run in parallel)
        # since they don't depend on each other
        parallel_modules = []
        for group in execution_groups:
            if len(group) > 1:
                parallel_modules.extend(group)

        # We should identify some parallel execution opportunities
        self.assertTrue(
            len(parallel_modules) > 0 or len(execution_groups[0]) > 1,
            "Should identify parallel execution opportunities"
        )


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling across module integration"""

    def test_invalid_state_handling(self):
        """Test handling of invalid state across modules"""
        coupling = BiosphereAtmosphereCoupling("test", {})

        # Create state with NaN values
        invalid_state = {
            'friction_velocity': torch.tensor(0.5),
            'air_temperature': torch.tensor(float('nan')),
            'atmospheric_pressure': torch.tensor(101325.0),
        }

        with self.assertRaisesRegex(ValueError, "air_temperature"):
            coupling.forward(invalid_state, time=0.0, dt=3600.0)

    def test_missing_required_state_handling(self):
        """Test handling of missing required state variables"""
        coupling = BiosphereAtmosphereCoupling("test", {})

        # Create incomplete state
        incomplete_state = {
            'air_temperature': torch.tensor(293.15),
            # Missing other required variables
        }

        # Should raise informative error
        with self.assertRaises((ValueError, KeyError)) as context:
            coupling.forward(incomplete_state, time=0.0, dt=3600.0)

        # Error message should be informative
        self.assertTrue(
            'required' in str(context.exception).lower() or
            'missing' in str(context.exception).lower()
        )


if __name__ == '__main__':
    # Set up test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCouplingIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleChaining))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterSystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestStateManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestDependencyResolution))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandlingIntegration))

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("L2 INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

