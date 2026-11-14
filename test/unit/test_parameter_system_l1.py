"""
L1 Unit Tests for RELIC Parameter System Refactoring

This test suite validates the parameter management system refactoring:
- P1: PFT parameter integration  
- P2: Hydrosphere parameters
- P3: Numerical solver parameters
"""

import unittest
import torch
import sys
from pathlib import Path

# Add relic to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



class TestP1PFTParameters(unittest.TestCase):
    """Test P1: PFT parameter integration."""
    
    def test_photosynthesis_pft_loading(self):
        """Test that photosynthesis parameters can be loaded via parameter manager."""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
        
        # Create photosynthesis module with PFT
        config = {
            'name': 'test_photosynthesis',
            'pft_name': 'broadleaf_deciduous',
            'pathway': 'C3',  # Add pathway parameter
            'vcmax25': 60.0,
            'jmax25': 120.0,
            'rd25': 1.2
        }
        
        # Initialize photosynthesis model
        photosynthesis = FarquharPhotosynthesis(config=config)
        
        # Check that parameters were loaded
        self.assertIsNotNone(photosynthesis.vcmax25)
        self.assertIsNotNone(photosynthesis.jmax25)
        self.assertIsNotNone(photosynthesis.rd25)
        
        # Check parameter types: values given through config are persistent module
        # buffers (a manager-owned learnable value is held by reference instead).
        buffers = dict(photosynthesis.named_buffers())
        for name in ("vcmax25", "jmax25", "rd25"):
            self.assertIsInstance(getattr(photosynthesis, name), torch.Tensor)
            self.assertIs(buffers[name], getattr(photosynthesis, name))
    
    def test_respiration_pft_loading(self):
        """Test respiration PFT parameter loading."""
        from relic.scientific.biosphere.respiration.maintenance_respiration import MaintenanceRespiration
        
        config = {
            'name': 'test_respiration',
            'pft_name': 'needleleaf_evergreen',
            'rm_leaf_25': 0.015,
            'rm_stem_25': 0.003,
            'rm_root_25': 0.010
        }
        
        # Initialize respiration model
        respiration = MaintenanceRespiration(config=config)
        
        # Check that parameters were loaded
        self.assertIsNotNone(respiration.rm_leaf_25)
        self.assertIsNotNone(respiration.rm_stem_25)
        self.assertIsNotNone(respiration.rm_root_25)


class TestP2HydrosphereParameters(unittest.TestCase):
    """Test P2: Hydrosphere parameter loading."""
    
    def test_vertical_percolation_parameters(self):
        """Test vertical percolation parameter loading from database."""
        from relic.scientific.hydrosphere.transport.vertical_percolation import VerticalPercolationTransport
        from relic.utils.physical_constants import get_physical_constant

        # Initialize percolation model
        percolation = VerticalPercolationTransport()
        
        # Check that physical constants were loaded
        self.assertIsNotNone(percolation.water_density)
        self.assertIsNotNone(percolation.reference_temperature)
        self.assertFalse(hasattr(percolation, "gravity"))

        expected_density = get_physical_constant("water_density")
        expected_temperature = get_physical_constant("reference_temperature")
        self.assertAlmostEqual(percolation.water_density, float(expected_density), delta=10.0)
        self.assertAlmostEqual(
            percolation.reference_temperature,
            float(expected_temperature),
            delta=1.0e-6,
        )
        
        # Check soil parameters
        self.assertIsNotNone(percolation.Ks_value)
        self.assertIsNotNone(percolation.theta_s_value)
        self.assertIsNotNone(percolation.theta_r_value)
    
    def test_numerical_parameters(self):
        """Test numerical parameter loading for hydrosphere."""
        from relic.scientific.hydrosphere.transport.vertical_percolation import VerticalPercolationTransport

        # Initialize model
        model = VerticalPercolationTransport()
        
        # Check numerical parameters
        self.assertIsNotNone(model.temperature_bounds)
        self.assertIsNotNone(model.saturation_epsilon)
        self.assertIsNotNone(model.effective_saturation_bounds)

        self.assertLess(model.temperature_bounds[0], model.temperature_bounds[1])
        self.assertGreater(model.saturation_epsilon, 0.0)
        self.assertGreaterEqual(model.effective_saturation_bounds[0], 0.0)
        self.assertLessEqual(model.effective_saturation_bounds[1], 1.0)

    def test_database_list_parameters_satisfy_their_componentwise_bounds(self):
        """Every list value that declares one [min, max] pair per component lies inside its pairs."""
        from relic.parameters import get_global_manager
        from relic.parameters.spec import componentwise_bounds

        pm = get_global_manager()
        checked = []
        for key in pm.list():
            spec = pm.get_parameter(key)
            if isinstance(spec.value, (list, tuple)) and componentwise_bounds(spec.bounds) is not None:
                checked.append(key)
                self.assertEqual(spec.validate_value(spec.value), [], key)

        self.assertIn("hydrosphere.numerical.temperature_bounds", checked)
        self.assertEqual(pm.validate_parameters(checked), [])

    def test_net_radiation_limits_declare_numeric_component_pairs(self):
        """The [min, max] net-radiation limits declare one numeric pair per component, as the other limit parameters do."""
        from relic.parameters import get_global_manager
        from relic.parameters.spec import componentwise_bounds

        pm = get_global_manager()
        key = "hydrosphere.numerical.net_radiation_bounds"
        spec = pm.get_parameter(key)
        # Declared intervals [W m-2]: minimum in [-2000, 0], maximum in [1000, 2500].
        self.assertEqual(componentwise_bounds(spec.bounds), ((-2000.0, 0.0), (1000.0, 2500.0)))
        self.assertEqual(spec.validate_value(spec.value), [])
        original = list(pm.get(key))
        self.assertEqual(original, [-500.0, 1500.0])
        rejected = {
            "component 0 above maximum: 1000000000.0 > 0.0": [1.0e9, -1.0e9],
            "component 1 above maximum: 5000.0 > 2500.0": [-500.0, 5000.0],
            "component 1 below minimum: 10.0 < 1000.0": [-500.0, 10.0],
        }
        for message, candidate in rejected.items():
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(ValueError, message):
                    pm.set_value(key, candidate, units="W m-2", validate=True)
                self.assertEqual(list(pm.get(key)), original)

    def test_set_value_enforces_componentwise_bounds_of_a_list_parameter(self):
        """The manager rejects a list value outside its per-component bounds and keeps the stored value."""
        from relic.parameters import get_global_manager

        pm = get_global_manager()
        key = "hydrosphere.numerical.temperature_bounds"
        original = list(pm.get(key))
        rejected = {
            "component 0 below minimum: 180.0 < 183.15": [180.0, 323.15],
            "component 1 above maximum: 400.0 > 373.15": [233.15, 400.0],
            "has length 1 but its bounds declare 2 components": [273.15],
            "component 0 must be a real number": ["273.15", 323.15],
            "component 0 is not finite": [float("nan"), 323.15],
        }
        for message, candidate in rejected.items():
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(ValueError, message):
                    pm.set_value(key, candidate, units="K", validate=True)
                self.assertEqual(list(pm.get(key)), original)

        try:
            # The declared limits themselves are admitted (closed intervals).
            pm.set_value(key, [223.15, 373.15], units="K", validate=True)
            self.assertEqual(list(pm.get(key)), [223.15, 373.15])
        finally:
            pm.set_value(key, original, units="K", validate=True)
        self.assertEqual(list(pm.get(key)), original)

    def test_set_value_rejects_a_non_sequence_value_of_a_componentwise_list_parameter(self):
        """A scalar or Tensor candidate is a validation error (ValueError), not a TypeError."""
        from relic.parameters import get_global_manager

        pm = get_global_manager()
        key = "hydrosphere.numerical.temperature_bounds"
        original = list(pm.get(key))
        for candidate in (300.0, float("nan"), torch.tensor(300.0), torch.tensor([273.15, 323.15])):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(ValueError, "takes a list or tuple of 2 numbers"):
                    pm.set_value(key, candidate, units="K", validate=True)
                self.assertEqual(list(pm.get(key)), original)


class TestP3NumericalStability(unittest.TestCase):
    """Test P3: Canonical numerical parameter loading."""
    
    def test_solver_tolerances_load_through_solver_base(self):
        """SolverBase is the runtime owner of solver tolerance bindings."""
        from relic.computation.solvers import BackwardEulerSolver

        solver = BackwardEulerSolver()

        self.assertGreater(solver.atol, 0.0)
        self.assertGreaterEqual(solver.rtol, 0.0)
        self.assertGreater(solver.max_iterations, 0)
    
    def test_cfl_limit_loads_from_parameter_manager(self):
        """CFL policy values must come from the canonical parameter owner."""
        from relic.parameters import get_global_manager

        cfl_limit = get_global_manager().get(
            "transport.numerical.cfl_advection_max",
            expected_type=float,
            required=True,
        )

        self.assertGreater(cfl_limit, 0.0)
        self.assertLessEqual(cfl_limit, 1.0)


class TestParameterIntegration(unittest.TestCase):
    """Integration tests for parameter system."""
    
    def test_parameter_manager_singleton(self):
        """Test that parameter manager is properly initialized."""
        from relic.parameters import get_global_manager
        
        # Get parameter manager
        pm = get_global_manager()
        
        # Check that it exists
        self.assertIsNotNone(pm)
        
        # Check that it's the same instance
        pm2 = get_global_manager()
        self.assertIs(pm, pm2)
    
    def test_parameter_namespace_access(self):
        """Test parameter access through namespaces."""
        from relic.parameters import get_global_manager
        
        pm = get_global_manager()
        
        # Try to access physical constants
        gravity = pm.get('relic.physical.constants.earth.gravity_acceleration', default=9.81)
        self.assertIsNotNone(gravity)
        
        # Access a runtime-consumed hydrosphere parameter without a hidden default.
        minimum_timestep_tolerance = pm.get(
            'hydrosphere.numerical.minimum_timestep_tolerance',
            required=True,
        )
        self.assertIsNotNone(minimum_timestep_tolerance)

    def test_trainable_parameters_registered(self):
        """Verify trainable parameters are registered with nn.Parameter."""
        from relic.parameters import get_global_manager

        pm = get_global_manager()
        trainable = pm.get_trainable_parameters()
        self.assertGreater(len(trainable), 0)
        self.assertIn("hydrosphere.soil_moisture.field_capacity", trainable)


def run_tests():
    """Run all L1 unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestP1PFTParameters))
    suite.addTests(loader.loadTestsFromTestCase(TestP2HydrosphereParameters))
    suite.addTests(loader.loadTestsFromTestCase(TestP3NumericalStability))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests
    success = run_tests()
    
    # Print summary
    print("\n" + "="*60)
    if success:
        print("[OK] All L1 unit tests passed!")
    else:
        print("[ERROR] Some L1 unit tests failed!")
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

