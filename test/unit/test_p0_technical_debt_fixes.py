"""
Unit tests for P0 Technical Debt Fixes

This test suite validates that all P0-level technical debt items have been
properly addressed according to TECHNICAL_DEBT.md.

Test Coverage:
    TD-001: Simulator-StateManager Transaction Integration
    TD-002: Validation Framework Success Codes
    TD-003: Irregular Grid Interpolation Error Standardization
    TD-007: Parameter Direct Access Pattern
"""

import unittest
from unittest.mock import Mock
import torch
import sys
import os
import importlib

# Add relic to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.validation.framework.models import ValidationCode, ValidationResultModel, ValidationLevel
from relic.coupling.capability_registry import CouplingErrorCode, create_not_implemented_error
from relic.core.functor import FunctorBase
from test.fixtures.dummy_functor import DummyConfiguredOutputFunctor


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'x': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class TestSimulatorStateManagerIntegration(unittest.TestCase):
    """Test TD-001: Simulator-StateManager Transaction Integration"""

    def setUp(self):
        """Set up test fixtures."""
        self.state_manager = Mock(spec=StateManager)
        # Attaching a manager synchronises its provenance identity through the
        # private recorder slot; a spec mock must expose it explicitly.
        self.state_manager._provenance_recorder = None
        config = {'max_time': 100.0, 'time_step': 1.0}
        self.simulator = _declared_simulator(config)
        self.simulator.state_manager = self.state_manager

    def test_apply_updates_uses_state_manager(self):
        """Test that _apply_updates delegates to StateManager.apply_updates()"""
        # Setup
        updates = {'carbon': torch.tensor(1.0), 'water': torch.tensor(2.0)}
        self.state_manager.apply_updates = Mock()
        self.state_manager.get_state = Mock(return_value={'carbon': torch.tensor(10.0)})

        # Execute
        self.simulator._apply_updates({}, updates)

        # Verify
        self.state_manager.apply_updates.assert_called_once_with(
            updates,
            dt=self.simulator.time_step,
            check_conservation=True,
            rollback_on_failure=True,
        )

    def test_apply_updates_handles_conservation_error(self):
        """Test that conservation violations are properly handled"""
        # Setup
        updates = {'carbon': torch.tensor(-1000.0)}  # Violation
        self.state_manager.apply_updates = Mock(side_effect=ValueError("Conservation violation"))

        # Execute & Verify
        with self.assertRaises(RuntimeError) as context:
            self.simulator._apply_updates({}, updates)
        self.assertIn("State update violated constraints", str(context.exception))

    def test_get_current_state_from_state_manager(self):
        """Test that _get_current_state uses StateManager as single source of truth"""
        # Setup
        expected_state = {
            'carbon': torch.tensor(100.0),
            'water': torch.tensor(50.0)
        }
        self.state_manager.get_state = Mock(return_value=expected_state.copy())

        # Execute
        state = self.simulator._get_current_state()

        # Verify
        self.state_manager.get_state.assert_called_once()
        self.assertIn('carbon', state)
        self.assertIn('water', state)
        self.assertIn('time', state)  # Added metadata
        self.assertIn('step', state)  # Added metadata


class TestValidationFrameworkSuccessCodes(unittest.TestCase):
    """Test TD-002: Validation Framework Success Codes"""

    def test_success_codes_exist(self):
        """Test that all required success codes are defined"""
        success_codes = [
            'CF_COMPLIANCE_OK',
            'CF_UNIT_OK',
            'CF_NAME_OK',
            'MASS_BALANCE_OK_CARBON',
            'MASS_BALANCE_OK_NITROGEN',
            'MASS_BALANCE_OK_WATER',
            'ENERGY_BALANCE_OK',
            'PARAM_BOUNDS_OK',
            'PARAM_TYPE_OK',
            'COUPLING_CONTRACT_OK',
            'STRESS_FACTOR_OK'
        ]

        for code_name in success_codes:
            self.assertTrue(hasattr(ValidationCode, code_name),
                          f"Success code {code_name} not found in ValidationCode enum")

    def test_validation_result_with_success_code(self):
        """Test that ValidationResultModel correctly handles success codes"""
        result = ValidationResultModel(
            level=ValidationLevel.INFO,
            code=ValidationCode.MASS_BALANCE_OK_CARBON,
            message="Carbon conservation satisfied",
            layer="L2",
            scope="carbon_mass_conservation"
        )

        # Verify passed flag is auto-set to True for _OK codes
        self.assertTrue(result.passed)
        self.assertTrue(result.is_passing())
        self.assertFalse(result.is_failing())

    def test_validation_result_with_failure_code(self):
        """Test that ValidationResultModel correctly handles failure codes"""
        result = ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.MASS_BALANCE_FAIL_CARBON,
            message="Carbon conservation violated",
            layer="L2",
            scope="carbon_mass_conservation"
        )

        # Verify passed flag is auto-set to False for ERROR level
        self.assertFalse(result.passed)
        self.assertFalse(result.is_passing())
        self.assertTrue(result.is_failing())


class TestCouplingErrorStandardization(unittest.TestCase):
    """Test TD-003: Irregular Grid Interpolation Error Standardization"""

    def test_error_codes_defined(self):
        """Test that coupling error codes are properly defined"""
        error_codes = [
            'IRREGULAR_GRID_NOT_SUPPORTED',
            'UNSTRUCTURED_MESH_NOT_SUPPORTED',
            'ADAPTIVE_GRID_NOT_SUPPORTED'
        ]

        for code_name in error_codes:
            self.assertTrue(hasattr(CouplingErrorCode, code_name),
                          f"Error code {code_name} not found in CouplingErrorCode enum")

    def test_create_not_implemented_error(self):
        """Test standardized error creation function"""
        error = create_not_implemented_error(
            "conservative_regridding_irregular",
            "Grid shape: (100, 100)"
        )

        self.assertIsInstance(error, NotImplementedError)
        # Check that the error message contains key information
        error_str = str(error)
        self.assertTrue(
            "bilinear" in error_str.lower() or "not implemented" in error_str.lower(),
            f"Error message doesn't contain expected content: {error_str}"
        )
        # The error should contain the additional info
        self.assertIn("Grid shape: (100, 100)", str(error))

    def test_capability_registry_alternatives(self):
        """Test that capability registry provides alternatives"""
        from relic.coupling.capability_registry import CouplingCapabilityRegistry

        capability = CouplingCapabilityRegistry.CAPABILITIES.get("conservative_regridding_irregular")
        self.assertIsNotNone(capability)
        self.assertEqual(capability.status, "not_implemented")
        self.assertGreater(len(capability.alternatives), 0)


class TestParameterAccessPattern(unittest.TestCase):
    """Test TD-007: Parameter Direct Access Pattern"""

    def test_functor_base_has_parameter_manager_method(self):
        """Test that FunctorBase provides _get_parameter_manager method"""
        # Check if it's a module-level function or method
        import relic.core.functor.functor_base as functor_base
        has_method = hasattr(FunctorBase, '_get_parameter_manager')
        has_function = hasattr(functor_base, '_get_parameter_manager')
        self.assertTrue(has_method or has_function,
                       "_get_parameter_manager not found as method or function")

    def test_functor_uses_parameter_manager_method(self):
        """Test that Functors have access to parameter manager through proper channels"""
        # Test that FunctorBase provides get_parameter_value method
        from relic.core.functor import FunctorBase
        
        # Check that FunctorBase has the get_parameter_value method
        self.assertTrue(hasattr(FunctorBase, 'get_parameter_value'),
                       "FunctorBase should have get_parameter_value method")
        
        # This method should be the primary way functors access parameters
        # Direct access to global manager is discouraged

    def test_cli_manager_usage_is_acceptable(self):
        """Test that CLI has been properly removed or restricted"""
        # Legacy CLI namespace must be absent.
        with self.assertRaises(ImportError):
            importlib.import_module("relic.cli")

        # New CLI namespace must be available.
        new_cli = importlib.import_module("relic.interfaces.cli")
        self.assertIsNotNone(new_cli)


class TestIntegrationSuite(unittest.TestCase):
    """Integration tests for P0 technical debt fixes working together"""

    def test_full_simulation_step_with_conservation(self):
        """Test complete simulation step with all P0 fixes integrated"""
        config = {'max_time': 10.0, 'time_step': 1.0}
        simulator = _declared_simulator(config)
        simulator.initialize_state({
            'x': torch.tensor(0.0),
        })
        state_manager = simulator.state_manager
        original_apply_updates = state_manager.apply_updates
        state_manager.apply_updates = Mock(wraps=original_apply_updates)

        # Use a protocol-complete bridge functor so registration exercises the
        # canonical Abstraction -> Bridge -> FunctorBase lifecycle.
        mock_functor = DummyConfiguredOutputFunctor(
            name="test_functor",
            config={"output_name": "x"},
        )

        # Register functor
        simulator.register_functor(mock_functor)
        
        updates = simulator.step()

        # Verify the protocol-complete functor produced a real governed update.
        self.assertIn("x", updates)
        self.assertEqual(float(updates["x"]), 1.0)
        state_manager.apply_updates.assert_called_once()
        self.assertTrue(
            state_manager.apply_updates.call_args.kwargs["check_conservation"]
        )
        self.assertEqual(
            float(state_manager.get_state()["x"]),
            1.0,
        )

    def test_validation_with_success_codes(self):
        """Test validation returns proper success codes"""
        from relic.validation.framework.rules.conservation import validate_water_mass_conservation
        from relic.validation.framework.models import ValidationContext

        # Setup context with good conservation
        context = ValidationContext()
        context.metadata = {
            'water_fluxes': {
                'precipitation': torch.tensor(10.0),
                'evaporation': torch.tensor(5.0),
                'transpiration': torch.tensor(3.0),
                'runoff': torch.tensor(2.0),
                'storage_change': torch.tensor(0.0)  # Perfect balance
            }
        }
        context.tolerance_overrides = {}

        # Execute validation
        results = validate_water_mass_conservation(context)

        # Check for success code
        success_results = [r for r in results if r.code == ValidationCode.MASS_BALANCE_OK_WATER]
        self.assertGreater(len(success_results), 0, "Should have at least one success result")
        self.assertTrue(success_results[0].passed)


def run_tests():
    """Run all P0 technical debt fix tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSimulatorStateManagerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationFrameworkSuccessCodes))
    suite.addTests(loader.loadTestsFromTestCase(TestCouplingErrorStandardization))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterAccessPattern))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationSuite))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("P0 TECHNICAL DEBT FIX VALIDATION SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    if result.wasSuccessful():
        print("\n[OK] All P0 technical debt fixes have been validated!")
        print("TD-001: Simulator-StateManager integration - VERIFIED")
        print("TD-002: Validation success codes - VERIFIED")
        print("TD-003: Coupling error standardization - VERIFIED")
        print("TD-007: Parameter access pattern - VERIFIED")
    else:
        print("\n[ERROR] Some P0 technical debt fixes need attention")
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed: {test}")
            print(traceback[:500])  # Print first 500 chars of traceback

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

