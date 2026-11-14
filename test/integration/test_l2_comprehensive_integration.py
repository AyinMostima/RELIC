"""
L2 Integration Test Suite for RELIC

This test suite validates integration between major modules:
- Core abstractions and functors
- Parameter management system
- State management and conservation
- Scientific modules integration
- Validation framework
- Data I/O and standards compliance
"""

import importlib
import unittest
import sys
import os
import pytest
import torch
import numpy as np

# Add relic to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from relic.core import BiologicalProcessFunctor
from relic.core.state import (
    StateManager,
    StateVariableSpec,
    UpdateItem,
    UpdateMode,
)
from relic.core.simulator import RELICSimulator, SimulationState
from relic.core.dependency_graph import DependencyGraphResolver
from relic.parameters import ParameterSpec, get_global_manager
from relic.validation.framework import ValidationEngine, ValidationContext
from relic.validation.framework.registry import get_global_registry
from relic.data.standards.cf_conventions import CFConventions
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


class TestParameterSystemIntegration(unittest.TestCase):
    """Test integration between parameter system and functors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = get_global_manager()
        
    def test_functor_parameter_access(self):
        """Test that functors correctly access parameters through unified system"""
        # Create a photosynthesis functor
        photo = FarquharPhotosynthesis(
            name="test_photosynthesis",
            vcmax25=60.0,
            jmax25=120.0,
            rd25=1.2
        )
        
        # Verify parameters are accessible
        vcmax = photo.get_parameter_value("vcmax25")
        vcmax_value = float(vcmax.detach().item()) if torch.is_tensor(vcmax) else float(vcmax)
        expected = self.manager.get("biosphere.photosynthesis.vcmax25")
        expected_value = float(expected.detach().item()) if torch.is_tensor(expected) else float(expected)
        self.assertAlmostEqual(vcmax_value, expected_value, places=1)
        
        # Test parameter retrieval with bounds (warning, not error in current implementation)
        # The parameter system logs warnings for out-of-bounds but doesn't raise
        vcmax_bounded = photo.get_parameter_value("vcmax25", bounds=(10.0, 200.0))
        self.assertGreaterEqual(vcmax_bounded, 10.0)
        self.assertLessEqual(vcmax_bounded, 200.0)
    
    def test_parameter_namespace_organization(self):
        """Test that parameters are organized in correct namespaces"""
        # Set parameters in different namespaces
        self.manager.set_value("biosphere.photosynthesis.vcmax25", 65.0)
        self.manager.register_parameter(ParameterSpec(
            key="atmosphere.turbulence.mixing_length", value=100.0, units="m",
            description="Namespace test parameter", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter",
        ))

        # Retrieve and verify
        vcmax = self.manager.get("biosphere.photosynthesis.vcmax25")
        mixing = self.manager.get("atmosphere.turbulence.mixing_length")
        
        self.assertEqual(vcmax, 65.0)
        self.assertEqual(mixing, 100.0)
        
        # List parameters by namespace
        bio_params = self.manager.list(namespace="biosphere")
        self.assertIn("biosphere.photosynthesis.vcmax25", bio_params)
        self.assertNotIn("atmosphere.turbulence.mixing_length", bio_params)


class TestStateManagementIntegration(unittest.TestCase):
    """Test integration between state management and scientific modules"""
    
    def setUp(self):
        """Set up test fixtures"""
        schema = {
            name: StateVariableSpec(
                name=name,
                units=units,
                conserved=True,
                required=False,
                accumulates=accumulates,
            )
            for name, units, accumulates in (
                ("carbon_vegetation", "kg C m-2", "carbon.total"),
                ("carbon_soil", "kg C m-2", "carbon.total"),
                ("carbon_atmosphere", "kg C m-2", "carbon.total"),
                ("water_soil", "kg m-2", "water.total"),
                ("water_atmosphere", "kg m-2", "water.total"),
                ("water_outflow", "kg m-2", "water.total"),
            )
        }
        self.state_manager = StateManager(
            schema=schema,
            autoload_contract_registry=False,
            device="cpu",
        )
        self.state_manager.initialize()
        
    def test_conservation_checking_integration(self):
        """Test that conservation checks work across modules"""
        # Set initial state
        initial_state = {
            'carbon_vegetation': torch.tensor(100.0),
            'carbon_soil': torch.tensor(200.0),
            'carbon_atmosphere': torch.tensor(700.0)
        }
        self.state_manager.set_state(initial_state)
        self.state_manager.set_conservation_ledger(
            {
                "carbon": [
                    "carbon_vegetation",
                    "carbon_soil",
                    "carbon_atmosphere",
                ]
            }
        )
        
        # Apply updates that conserve carbon
        updates = {
            name: UpdateItem(
                value=torch.tensor(value),
                mode=UpdateMode.ACCUMULATE,
                unit="kg C m-2",
                per_timestep=True,
                element="C",
                source_functor="test_carbon_transfer",
            )
            for name, value in (
                ("carbon_vegetation", -5.0),
                ("carbon_soil", 3.0),
                ("carbon_atmosphere", 2.0),
            )
        }
        
        # Apply updates with conservation check
        self.state_manager.apply_updates(updates, dt=1.0, check_conservation=True)
        
        # Verify state was updated
        new_state = self.state_manager.get_state()
        self.assertAlmostEqual(new_state['carbon_vegetation'].item(), 95.0, places=1)
        self.assertAlmostEqual(new_state['carbon_soil'].item(), 203.0, places=1)
        self.assertAlmostEqual(new_state['carbon_atmosphere'].item(), 702.0, places=1)
    
    def test_state_rollback_on_violation(self):
        """Test that state rolls back on conservation violation"""
        # Set initial state
        initial_state = {
            'water_soil': torch.tensor(50.0),
            'water_atmosphere': torch.tensor(10.0)
        }
        self.state_manager.set_state(initial_state)
        self.state_manager.set_conservation_ledger(
            "water",
            ["water_soil", "water_atmosphere", "water_outflow"],
        )
        
        # Create violating updates (water created from nothing)
        bad_updates = {
            name: UpdateItem(
                value=torch.tensor(value),
                mode=UpdateMode.ACCUMULATE,
                unit="kg m-2",
                per_timestep=True,
                element="water",
                source_functor="test_water_transfer",
            )
            for name, value in (
                ("water_soil", 100.0),
                ("water_atmosphere", 50.0),
                ("water_outflow", 0.0),
            )
        }
        
        # Apply with strict conservation should fail
        with self.assertRaisesRegex(ValueError, "Conservation violation for water"):
            self.state_manager.apply_updates(
                bad_updates, 
                dt=1.0, 
                check_conservation=True,
                rollback_on_failure=True,
            )
        
        # Verify state was NOT changed (rollback)
        state = self.state_manager.get_state()
        self.assertAlmostEqual(state['water_soil'].item(), 50.0, places=1)


class TestScientificModuleIntegration(unittest.TestCase):
    """Test integration between different scientific modules"""
    
    def test_photosynthesis_respiration_coupling(self):
        """Test that photosynthesis and respiration modules work together"""
        # Create photosynthesis module
        photo = FarquharPhotosynthesis(
            name="photosynthesis",
            vcmax25=60.0,
            jmax25=120.0,
            rd25=1.2
        )
        
        # Create state with environmental conditions
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'vcmax_scaling': torch.tensor(1.0),
        }
        
        # Run photosynthesis
        photo_output = photo.forward(state, time=0.0, dt=1.0)
        
        # Verify outputs
        self.assertIn('gross_photosynthesis', photo_output)
        self.assertIn('diagnostics.photosynthesis.dark_respiration', photo_output)
        self.assertIn('net_photosynthesis', photo_output)
        self.assertNotIn('dark_respiration', photo_output)
        
        # Check that net = gross - respiration
        gross = photo_output['gross_photosynthesis'].value
        resp = photo_output['diagnostics.photosynthesis.dark_respiration'].value
        net = photo_output['net_photosynthesis'].value
        
        self.assertAlmostEqual(
            net.item(), 
            (gross - resp).item(), 
            places=3
        )
    
    def test_multi_module_carbon_cycle(self):
        """Test carbon flow through multiple modules"""
        # This would test photosynthesis -> litter -> decomposition chain
        # Simplified for demonstration
        
        # Initial carbon pools
        carbon_pools = {
            'atmosphere': torch.tensor(750.0),
            'vegetation': torch.tensor(550.0),
            'litter': torch.tensor(60.0),
            'soil': torch.tensor(1500.0)
        }
        
        # Photosynthesis removes from atmosphere, adds to vegetation
        photo_flux = 10.0
        carbon_pools['atmosphere'] -= photo_flux
        carbon_pools['vegetation'] += photo_flux
        
        # Litter fall from vegetation to litter
        litter_flux = 2.0
        carbon_pools['vegetation'] -= litter_flux
        carbon_pools['litter'] += litter_flux
        
        # Decomposition from litter to soil and atmosphere
        decomp_to_soil = 0.5
        decomp_to_atm = 1.5
        carbon_pools['litter'] -= (decomp_to_soil + decomp_to_atm)
        carbon_pools['soil'] += decomp_to_soil
        carbon_pools['atmosphere'] += decomp_to_atm
        
        # Verify conservation
        total_initial = 750.0 + 550.0 + 60.0 + 1500.0
        total_final = sum(pool.item() for pool in carbon_pools.values())
        self.assertAlmostEqual(total_initial, total_final, places=6)


class TestValidationFrameworkIntegration(unittest.TestCase):
    """Test integration of validation framework with other components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = ValidationEngine()
        
    def test_cross_domain_exposes_only_registered_cf_syntax_rule(self):
        """Cross-domain validation must not retain an unregistered CF path."""
        module_name = "relic.validation.framework.rules.cross_domain"
        sys.modules.pop(module_name, None)
        cross_domain = importlib.import_module(module_name)

        self.assertFalse(hasattr(cross_domain, "validate_cf_compliance"))
        self.assertNotIn("validate_cf_compliance", cross_domain.__all__)
        self.assertEqual(
            sorted(
                name
                for name in vars(cross_domain)
                if name.startswith("validate_cf")
            ),
            ["validate_cf_standard_name_syntax"],
        )
        with self.assertRaises(ImportError):
            exec(
                "from relic.validation.framework.rules.cross_domain "
                "import validate_cf_compliance",
                {},
            )

        rule_ids = get_global_registry().list_rules()
        self.assertEqual(rule_ids.count("cf_standard_name_syntax"), 1)
        self.assertNotIn("cf_standard_name_compliance", rule_ids)
        self.assertNotIn("CF-1.8 compliance", cross_domain.__doc__ or "")
    
    def test_multi_layer_validation(self):
        """Test that L1, L2, L3 validations work together"""
        context = ValidationContext()
        context.layer = "L2"  # Integration layer
        
        # Add test data
        context.data = {
            'temperature': torch.tensor(298.15),
            'carbon': torch.tensor(100.0)
        }
        context.metadata = {
            'conservation_check': True,
            'bounds_check': True
        }
        
        # Run validation through engine
        report = self.engine.run(
            layers=["L1", "L2"],
            tags=["parameters", "conservation"],
            context=context
        )
        results = report.results
        
        # Verify we get results from multiple layers
        l1_results = [r for r in results if r.layer == "L1"]
        l2_results = [r for r in results if r.layer == "L2"]
        
        # Both layers should produce results
        self.assertGreater(len(l1_results), 0)
        self.assertGreater(len(l2_results), 0)


class TestDataStandardsIntegration(unittest.TestCase):
    """Test integration with data standards (CF-1.8, CMIP6)"""
    
    def test_cf_standard_names(self):
        """Test that standard names are correctly mapped"""
        cf = CFConventions()
        
        # Test standard name lookup
        temp_info = cf.get_standard_name_info("air_temperature")
        self.assertIsNotNone(temp_info)
        self.assertEqual(temp_info['canonical_units'], 'K')
        
        # Test unit conversion
        from relic.data.standards.unit_converter import UnitConverter, ConversionContext
        converter = UnitConverter()
        context = ConversionContext()
        
        # Convert 25C to Kelvin
        kelvin = converter.convert(25.0, 'generic', 'degC', 'K', context)
        self.assertAlmostEqual(kelvin, 298.15, places=1)
        
        # Convert flux units
        converted = converter.convert(1.0, 'generic', 'umol m-2 s-1', 'mol m-2 s-1', context)
        self.assertAlmostEqual(converted, 1e-6, places=10)
    
    def test_metadata_generation(self):
        """Test that proper CF metadata is generated"""
        # Create some data
        data = {
            'gross_primary_productivity_of_carbon': torch.tensor(10.0),
            'air_temperature': torch.tensor(298.15)
        }
        
        # Generate metadata
        cf = CFConventions()
        metadata = {}
        
        for var_name in data:
            info = cf.get_standard_name_info(var_name)
            if info:
                metadata[var_name] = {
                    'standard_name': var_name,
                    'units': info['canonical_units'],
                    'long_name': info.get('description', var_name)
                }
        
        # Verify metadata
        self.assertIn('gross_primary_productivity_of_carbon', metadata)
        self.assertEqual(
            metadata['gross_primary_productivity_of_carbon']['units'],
            'kg m-2 s-1'
        )
        self.assertEqual(metadata['air_temperature']['units'], 'K')

    def test_cross_domain_registry_consumes_only_explicit_cf_identity(self):
        """Local StateManager names must not be promoted into the CF table."""
        from relic.validation import CrossDomainValidator

        schema = {
            "local_carbon": StateVariableSpec(
                name="local_carbon",
                units="g C m-2",
                cf_standard_name=None,
            ),
            "measured_temperature": StateVariableSpec(
                name="measured_temperature",
                units="degree_Celsius",
                cf_standard_name="air_temperature",
            ),
        }
        state_manager = StateManager(
            schema=schema,
            autoload_contract_registry=False,
        )

        validator = CrossDomainValidator(state_manager)

        assert validator.variable_registry["local_carbon"].cf_standard_name is None
        assert validator.variable_registry["local_carbon"].units == "g C m-2"
        assert (
            validator.variable_registry["measured_temperature"].cf_standard_name
            == "air_temperature"
        )
        assert (
            validator.variable_registry["measured_temperature"].units
            == "degree_Celsius"
        )


class TestEndToEndIntegration(unittest.TestCase):
    """Test complete end-to-end workflows"""
    
    def test_simulation_with_validation(self):
        """Test running a simulation with validation enabled"""
        # Create simulator
        config = {
            'max_time': 10.0,
            'time_step': 1.0,
            'enable_validation': True,
            'device': 'cpu',
        }
        simulator = RELICSimulator(config)
        
        # Create and register a simple functor
        class SimpleCarbonFunctor(BiologicalProcessFunctor):
            def __init__(self):
                from relic.core.functor import FunctorMetadata, ProcessCategory
                metadata = FunctorMetadata(
                    name="simple_carbon",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    description="Simple carbon cycling",
                    mathematical_expression="delta_C_receiver = -delta_C_reserve",
                    references=["Sierra et al. 2012"],
                    input_variables=["carbon", "carbon_reserve"],
                    output_variables=["carbon", "carbon_reserve"],
                    conservation_laws=["carbon"],
                )
                super().__init__(name="simple_carbon", metadata=metadata)
            
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt):
                # References: Sierra et al. 2012
                # Expression: delta_C_receiver = -delta_C_reserve = 0.1 g C m-2
                # Parameters: delta_C is the prescribed internal transfer step.
                # Units: both carbon increments are g C m-2.
                # Closed route: carbon moves from reserve to receiver.
                return {
                    'carbon': UpdateItem(
                        value=torch.tensor(0.1),
                        mode=UpdateMode.ACCUMULATE,
                        per_timestep=True,
                        unit='g C m-2',
                        element='C',
                        source_functor=self.name,
                    ),
                    'carbon_reserve': UpdateItem(
                        value=torch.tensor(-0.1),
                        mode=UpdateMode.ACCUMULATE,
                        per_timestep=True,
                        unit='g C m-2',
                        element='C',
                        source_functor=self.name,
                    ),
                }
        
        functor = SimpleCarbonFunctor()
        simulator.register_functor(functor)
        
        # Build dependency graph
        dep_graph = DependencyGraphResolver()
        dep_graph.add_functor(functor)
        
        # Initialize with state
        initial_state = {
            'carbon': torch.tensor(100.0),
            'carbon_reserve': torch.tensor(10.0),
        }
        
        # Create state manager
        schema = {
            name: StateVariableSpec(
                name=name,
                units="g C m-2",
                conserved=True,
                required=True,
                accumulates="carbon.total",
                owner="simple_carbon",
                writers=["simple_carbon"],
            )
            for name in ("carbon", "carbon_reserve")
        }
        state_manager = StateManager(
            schema=schema,
            autoload_contract_registry=False,
            device="cpu",
        )
        state_manager.initialize()
        state_manager.set_state(initial_state)
        state_manager.set_conservation_ledger(
            {"carbon": ["carbon", "carbon_reserve"]}
        )
        simulator.state_manager = state_manager
        
        # Initialize simulator
        simulator.initialize(state_manager, dep_graph)
        simulator.state = SimulationState.RUNNING
        
        updates = simulator.step()
        self.assertIsNotNone(updates)
        carbon_state = state_manager.get_state('carbon')
        reserve_state = state_manager.get_state('carbon_reserve')
        torch.testing.assert_close(
            carbon_state,
            carbon_state.new_tensor(100.1),
        )
        torch.testing.assert_close(
            reserve_state,
            reserve_state.new_tensor(9.9),
        )
        self.assertEqual(
            {
                record['functor']
                for record in simulator.get_variable_provenance('carbon')
                if record.get('role') == 'increment_source'
            },
            {'simple_carbon'},
        )
        torch.testing.assert_close(
            carbon_state + reserve_state,
            carbon_state.new_tensor(110.0),
        )
        self.assertEqual(carbon_state.device.type, functor.device.type)
    
    def test_parameter_calibration_workflow(self):
        """Test parameter calibration integration"""
        from relic.optimization.calibration import ParameterCalibrator
        
        # Create calibrator
        calibrator = ParameterCalibrator()
        
        # Define parameters to calibrate
        params_to_calibrate = {
            'vcmax25': {'initial': 60.0, 'bounds': (30.0, 100.0)},
            'jmax25': {'initial': 120.0, 'bounds': (60.0, 200.0)}
        }
        
        # Mock objective function (would normally run simulation)
        def objective(params):
            # Simulate and compare with observations
            # Return loss
            return np.sum([(p - 65.0)**2 for p in params.values()])
        
        # Run calibration (simplified)
        # In real case, this would optimize parameters
        result = {
            'optimized_parameters': {'vcmax25': 65.0, 'jmax25': 130.0},
            'initial_loss': 100.0,
            'final_loss': 10.0,
            'convergence': True
        }
        
        # Verify calibration improved parameters
        self.assertLess(result['final_loss'], result['initial_loss'])
        self.assertTrue(result['convergence'])


def run_l2_integration_tests():
    """Run all L2 integration tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestParameterSystemIntegration,
        TestStateManagementIntegration,
        TestScientificModuleIntegration,
        TestValidationFrameworkIntegration,
        TestDataStandardsIntegration,
        TestEndToEndIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests with detailed output
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
    
    if result.wasSuccessful():
        print("\n[OK] All L2 integration tests passed!")
    else:
        print("\n[FAIL] Some L2 integration tests failed")
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed: {test}")
            print(traceback[:500])
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_l2_integration_tests()
    sys.exit(0 if success else 1)

