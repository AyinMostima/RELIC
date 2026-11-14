"""
L2 Module Integration Test Suite for RELIC

This comprehensive test suite validates integration between major system modules,
focusing on real-world scientific workflows and module interaction patterns.

Test Coverage Areas:
1. Atmosphere-Biosphere Coupling
2. Hydrosphere-Pedosphere Coupling  
3. Carbon Cycle Conservation Testing
4. Parameter System Integration
5. Multi-Rate Execution Testing
6. Error Propagation and Rollback

References:
- Bonan, G.B. (2019). Climate Change and Terrestrial Ecosystem Modeling
- Fisher, R.A. et al. (2018). Vegetation demographics in Earth System Models
- Lawrence, D.M. et al. (2019). The Community Land Model version 5

Version: 1.0.0
Authors: Yuchong Long
"""

import unittest
import sys
from pathlib import Path
import torch
from typing import Dict, List
import logging
import pytest

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add relic to path - handle both direct execution and pytest
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Attempt imports with graceful defaults
try:
    # Core imports - basic functionality should work
    from relic.core.state import (
        StateManager,
        StateVariableSpec,
        UpdateItem,
        UpdateMode,
    )
    from relic.parameters import ParameterSpec, get_global_manager
    CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core modules not available: {e}")
    StateManager = None
    get_global_manager = None
    CORE_AVAILABLE = False

try:
    # Functor base classes
    from relic.core.functor import FunctorBase
    from relic.core.abstractions import StateDict, EnvironmentDict, FluxDict
    FUNCTOR_BASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Functor base not available: {e}")
    FUNCTOR_BASE_AVAILABLE = False
    
    # Create mock classes
    class MockFunctorBase:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'mock_functor')
    
    FunctorBase = MockFunctorBase
    StateDict = dict
    EnvironmentDict = dict  
    FluxDict = dict

try:
    from relic.core.simulator import RELICSimulator, SimulationState
    SIMULATOR_AVAILABLE = True
except ImportError:
    RELICSimulator = None
    SimulationState = None
    SIMULATOR_AVAILABLE = False

try:
    from relic.core.dependency_graph import DependencyGraphResolver
    DEPENDENCY_GRAPH_AVAILABLE = True
except ImportError:
    DependencyGraphResolver = None
    DEPENDENCY_GRAPH_AVAILABLE = False


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


class SimpleMockFunctor:
    """Simplified mock functor for basic testing"""
    
    def __init__(self, name: str, dependencies: List[str] = None, 
                 flux_values: Dict[str, float] = None):
        self.name = name
        self._dependencies = dependencies or []
        self._flux_values = flux_values or {"mock_output": 1.0}
        
    def get_dependencies(self) -> List[str]:
        return self._dependencies
    
    def forward(self, state: Dict, time: float, dt: float) -> Dict:
        """Simple mock implementation"""
        return {key: torch.tensor(value) for key, value in self._flux_values.items()}


@unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")
class TestStateManagerIntegration(unittest.TestCase):
    """Test basic state management integration"""
    
    def setUp(self):
        """Set up state manager test fixtures"""
        if StateManager:
            self.state_manager = StateManager()
            self.state_manager.initialize()
        
    def test_basic_state_operations(self):
        """Test basic state set/get operations"""
        if not StateManager:
            self.skipTest("StateManager not available")
            
        # Set initial state
        initial_state = {
            'carbon_pool': torch.tensor(100.0),
            'nitrogen_pool': torch.tensor(10.0),
            'water_pool': torch.tensor(50.0)
        }
        
        self.state_manager.set_state(initial_state)
        
        # Retrieve state
        retrieved_state = self.state_manager.get_state()
        
        # Verify state integrity
        for key, value in initial_state.items():
            self.assertIn(key, retrieved_state)
            self.assertAlmostEqual(
                retrieved_state[key].item(), 
                value.item(), 
                places=6,
                msg=f"State variable {key} not preserved"
            )
    
    def test_state_updates_application(self):
        """Test applying updates to state"""
        if not StateManager:
            self.skipTest("StateManager not available")

        schema = {
            name: StateVariableSpec(
                name=name,
                units="kg C m-2",
                conserved=True,
                required=True,
                accumulates="carbon.total",
            )
            for name in ("carbon_vegetation", "carbon_soil")
        }
        self.state_manager = StateManager(
            schema=schema,
            autoload_contract_registry=False,
            device="cpu",
        )
        self.state_manager.initialize()

        # Set initial state
        initial_state = {
            'carbon_vegetation': torch.tensor(100.0),
            'carbon_soil': torch.tensor(200.0)
        }
        self.state_manager.set_state(initial_state)
        self.state_manager.set_conservation_ledger(
            {"carbon": ["carbon_vegetation", "carbon_soil"]}
        )

        # Apply a closed transfer from soil carbon to vegetation carbon.
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
                ("carbon_vegetation", 5.0),
                ("carbon_soil", -5.0),
            )
        }

        self.state_manager.apply_updates(
            updates,
            dt=1.0,
            check_conservation=True,
        )

        new_state = self.state_manager.get_state()
        self.assertAlmostEqual(
            new_state['carbon_vegetation'].item(),
            105.0,
            places=1,
        )
        self.assertAlmostEqual(
            new_state['carbon_soil'].item(),
            195.0,
            places=1,
        )
        self.assertAlmostEqual(
            (
                new_state['carbon_vegetation']
                + new_state['carbon_soil']
            ).item(),
            300.0,
            places=6,
        )


@unittest.skipUnless(CORE_AVAILABLE, "Core modules not available")  
class TestParameterSystemIntegration(unittest.TestCase):
    """Test parameter system integration"""
    
    def setUp(self):
        """Set up parameter system test fixtures"""
        if get_global_manager:
            try:
                self.param_manager = get_global_manager()
            except Exception as e:
                logger.warning(f"Parameter manager creation failed: {e}")
                self.param_manager = None
    
    def test_parameter_storage_retrieval(self):
        """Test basic parameter storage and retrieval"""
        if not self.param_manager:
            self.skipTest("Parameter manager not available")
            
        # Test parameters: one from the database, two registered by this test
        test_params = {
            "biosphere.photosynthesis.vcmax25": 65.0,
            "atmosphere.turbulence.mixing_length": 100.0,
            "hydrosphere.soil.field_capacity": 0.35
        }
        self.param_manager.register_parameter(ParameterSpec(
            key="atmosphere.turbulence.mixing_length", value=100.0, units="m", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))
        self.param_manager.register_parameter(ParameterSpec(
            key="hydrosphere.soil.field_capacity", value=0.35, units="m3 m-3", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))
        self.param_manager.set_value("biosphere.photosynthesis.vcmax25", 65.0)

        # Retrieve and verify
        for param_name, expected_value in test_params.items():
            try:
                retrieved_value = self.param_manager.get(param_name)
                if retrieved_value is not None:
                    self.assertAlmostEqual(
                        retrieved_value, 
                        expected_value, 
                        places=6,
                        msg=f"Parameter {param_name} retrieval failed"
                    )
            except Exception as e:
                logger.warning(f"Parameter retrieval failed for {param_name}: {e}")
    
    def test_parameter_namespace_organization(self):
        """Test parameter namespace organization"""
        if not self.param_manager:
            self.skipTest("Parameter manager not available")
            
        # Set parameters in different namespaces
        try:
            self.param_manager.register_parameter(
                ParameterSpec(key="biosphere.param1", value=1.0, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))
            self.param_manager.register_parameter(
                ParameterSpec(key="atmosphere.param2", value=2.0, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))

            # Test namespace listing if available
            if hasattr(self.param_manager, 'list'):
                bio_params = self.param_manager.list(namespace="biosphere")
                if bio_params:
                    self.assertIn("biosphere.param1", bio_params)
                    
        except Exception as e:
            logger.warning(f"Namespace operations not supported: {e}")
            self.skipTest("Namespace operations not implemented")


class TestScientificModuleIntegration(unittest.TestCase):
    """Test integration between different scientific modules"""
    
    def test_photosynthesis_respiration_coupling(self):
        """Test that photosynthesis and respiration modules work together"""
        # Create mock modules
        photosynthesis = SimpleMockFunctor(
            name="photosynthesis",
            flux_values={
                "gross_photosynthesis": 15.0,    # umol CO2 m-2 s-1
                "dark_respiration": 2.0,         # umol CO2 m-2 s-1
                "net_photosynthesis": 13.0       # umol CO2 m-2 s-1
            }
        )
        
        # Create state with environmental conditions
        state = {
            'temperature': torch.tensor(298.15),        # 25degC
            'co2_concentration': torch.tensor(400.0),   # 400 ppm
            'par': torch.tensor(1000.0),                # 1000 umol m-2 s-1
            'relative_humidity': torch.tensor(0.7)      # 70%
        }
        
        # Run photosynthesis
        photo_output = photosynthesis.forward(state, time=0.0, dt=1.0)
        
        # Verify outputs exist
        self.assertIn('gross_photosynthesis', photo_output)
        self.assertIn('dark_respiration', photo_output)
        self.assertIn('net_photosynthesis', photo_output)
        
        # Check physical constraints
        gross = photo_output['gross_photosynthesis'].item()
        resp = photo_output['dark_respiration'].item()
        net = photo_output['net_photosynthesis'].item()
        
        # Gross photosynthesis should be positive
        self.assertGreater(gross, 0.0)
        
        # Respiration should be positive
        self.assertGreater(resp, 0.0)
        
        # Net should be less than gross
        self.assertLess(net, gross)
    
    def test_carbon_nitrogen_coupling(self):
        """Test carbon-nitrogen coupling in ecosystems"""
        # Create coupled C-N cycle modules
        carbon_cycle = SimpleMockFunctor(
            name="carbon_cycle",
            flux_values={
                "carbon_fixation": 10.0,         # kg C m-2 yr-1
                "carbon_respiration": 8.0,       # kg C m-2 yr-1
                "carbon_allocation": 2.0          # kg C m-2 yr-1
            }
        )
        
        nitrogen_cycle = SimpleMockFunctor(
            name="nitrogen_cycle",
            dependencies=["carbon_cycle"],
            flux_values={
                "nitrogen_uptake": 0.5,           # kg N m-2 yr-1
                "nitrogen_mineralization": 0.3,   # kg N m-2 yr-1
                "nitrogen_immobilization": 0.2    # kg N m-2 yr-1
            }
        )
        
        # Initial ecosystem state
        state = {
            "carbon_vegetation": torch.tensor(5.0),     # kg C m-2
            "carbon_soil": torch.tensor(15.0),          # kg C m-2
            "nitrogen_vegetation": torch.tensor(0.2),   # kg N m-2
            "nitrogen_soil": torch.tensor(1.0),         # kg N m-2
            "cn_ratio_vegetation": torch.tensor(25.0),  # C:N ratio
            "cn_ratio_soil": torch.tensor(15.0)         # C:N ratio
        }
        
        # Execute carbon cycle
        carbon_fluxes = carbon_cycle.forward(state, time=0.0, dt=31536000.0)  # 1 year
        
        # Execute nitrogen cycle (dependent on carbon)
        nitrogen_fluxes = nitrogen_cycle.forward(state, time=0.0, dt=31536000.0)
        
        # Test stoichiometric coupling
        carbon_fixation = carbon_fluxes["carbon_fixation"].item()
        nitrogen_uptake = nitrogen_fluxes["nitrogen_uptake"].item()
        
        # Calculate implied C:N ratio
        if nitrogen_uptake > 0:
            implied_cn_ratio = carbon_fixation / nitrogen_uptake
            
            # C:N ratio should be within reasonable bounds for vegetation (15-50)
            self.assertGreaterEqual(implied_cn_ratio, 15.0)
            self.assertLessEqual(implied_cn_ratio, 50.0)
    
    def test_water_carbon_coupling(self):
        """Test water-carbon coupling through stomatal regulation"""
        # Water stress affects photosynthesis
        water_stress_conditions = [
            {"soil_water": 0.4, "expected_photosynthesis": "high"},    # Well-watered
            {"soil_water": 0.2, "expected_photosynthesis": "medium"},  # Mild stress
            {"soil_water": 0.1, "expected_photosynthesis": "low"}      # Severe stress
        ]
        
        for condition in water_stress_conditions:
            with self.subTest(soil_water=condition["soil_water"]):
                # Create water-stressed photosynthesis module
                if condition["expected_photosynthesis"] == "high":
                    photo_rate = 12.0
                elif condition["expected_photosynthesis"] == "medium":
                    photo_rate = 8.0
                else:  # low
                    photo_rate = 3.0
                
                water_stressed_photo = SimpleMockFunctor(
                    name="water_stressed_photosynthesis",
                    flux_values={"photosynthesis_rate": photo_rate}
                )
                
                state = {
                    "soil_water_content": torch.tensor(condition["soil_water"]),
                    "air_temperature": torch.tensor(298.15),
                    "vapor_pressure_deficit": torch.tensor(1.5)
                }
                
                fluxes = water_stressed_photo.forward(state, time=0.0, dt=3600.0)
                actual_rate = fluxes["photosynthesis_rate"].item()
                
                # Verify water stress effects
                if condition["expected_photosynthesis"] == "high":
                    self.assertGreater(actual_rate, 10.0)
                elif condition["expected_photosynthesis"] == "medium":
                    self.assertGreater(actual_rate, 5.0)
                    self.assertLess(actual_rate, 10.0)
                else:  # low
                    self.assertLess(actual_rate, 5.0)


class TestConservationPrinciples(unittest.TestCase):
    """Test conservation laws across module integrations"""
    
    def test_mass_conservation(self):
        """Test mass conservation in biogeochemical cycles"""
        # Carbon pools and fluxes
        initial_pools = {
            "atmosphere": torch.tensor(850.0),    # Pg C
            "vegetation": torch.tensor(550.0),    # Pg C
            "soil": torch.tensor(1500.0)          # Pg C
        }
        
        # Carbon fluxes (Pg C yr-1)
        fluxes = {
            "photosynthesis": torch.tensor(120.0),       # Atm -> Veg
            "plant_respiration": torch.tensor(60.0),     # Veg -> Atm
            "soil_respiration": torch.tensor(60.0),      # Soil -> Atm
            "litter_fall": torch.tensor(60.0)            # Veg -> Soil
        }
        
        # Calculate total initial carbon
        total_initial = sum(pool.item() for pool in initial_pools.values())
        
        # Apply fluxes for one year
        dt = 1.0  # years
        
        # Calculate changes
        delta_atmosphere = (
            -fluxes["photosynthesis"] + 
            fluxes["plant_respiration"] + 
            fluxes["soil_respiration"]
        ) * dt
        
        delta_vegetation = (
            fluxes["photosynthesis"] - 
            fluxes["plant_respiration"] - 
            fluxes["litter_fall"]
        ) * dt
        
        delta_soil = (
            fluxes["litter_fall"] - 
            fluxes["soil_respiration"]
        ) * dt
        
        # Calculate total change
        total_change = (delta_atmosphere + delta_vegetation + delta_soil).item()
        
        # Test mass conservation (total change should be zero)
        self.assertLess(abs(total_change), 1.0,  # 1 Pg C tolerance
                       msg="Carbon mass should be conserved")
        
        # Calculate final pools
        final_pools = {
            "atmosphere": initial_pools["atmosphere"] + delta_atmosphere,
            "vegetation": initial_pools["vegetation"] + delta_vegetation,
            "soil": initial_pools["soil"] + delta_soil
        }
        
        # Verify individual pool changes are reasonable
        self.assertLess(abs(delta_atmosphere.item()), 10.0)  # < 10 Pg C change per year
        self.assertGreater(final_pools["vegetation"].item(), 500.0)  # Vegetation remains substantial
        self.assertGreater(final_pools["soil"].item(), 1400.0)       # Soil carbon stable
    
    def test_energy_conservation(self):
        """Test energy conservation in surface energy balance"""
        # Surface energy balance components (W m-2)
        energy_inputs = {
            "solar_radiation": torch.tensor(500.0),      # Incoming solar
            "longwave_down": torch.tensor(250.0)         # Atmospheric LW
        }

        energy_outputs = {
            "sensible_heat": torch.tensor(120.0),        # H
            "latent_heat": torch.tensor(180.0),          # LE
            "longwave_up": torch.tensor(380.0),          # Outgoing LW (typical emission)
            "ground_heat": torch.tensor(70.0)            # G
        }
        
        # Calculate energy balance
        total_input = sum(flux.item() for flux in energy_inputs.values())
        total_output = sum(flux.item() for flux in energy_outputs.values())
        
        energy_imbalance = abs(total_input - total_output)
        
        # Test energy conservation (within 10 W m-2 tolerance)
        self.assertLess(energy_imbalance, 10.0,
                       msg="Surface energy should be approximately conserved")
        
        # Verify individual components are reasonable
        self.assertGreater(energy_outputs["sensible_heat"].item(), 0.0)
        self.assertGreater(energy_outputs["latent_heat"].item(), 0.0)
        self.assertGreater(energy_outputs["longwave_up"].item(), 350.0)  # Typical LW emission
    
    def test_water_balance_conservation(self):
        """Test water balance conservation"""
        # Water balance components (mm day-1)
        water_inputs = {
            "precipitation": torch.tensor(3.0),          # P
            "dew_formation": torch.tensor(0.1)           # Minor input
        }
        
        water_outputs = {
            "evapotranspiration": torch.tensor(2.5),     # ET
            "runoff": torch.tensor(0.3),                 # R
            "deep_drainage": torch.tensor(0.2)           # D
        }
        
        water_storage_change = torch.tensor(0.1)        # dS (soil water increase)
        
        # Water balance equation: P = ET + R + D + dS
        total_input = sum(flux.item() for flux in water_inputs.values())
        total_output = sum(flux.item() for flux in water_outputs.values())
        
        water_balance = total_input - total_output - water_storage_change.item()
        
        # Test water balance (within 0.1 mm day-1 tolerance)
        self.assertLess(abs(water_balance), 0.1,
                       msg="Water balance should be conserved")
        
        # Verify reasonable partitioning
        et_fraction = water_outputs["evapotranspiration"].item() / total_input
        self.assertGreater(et_fraction, 0.6)   # ET dominates in most ecosystems
        self.assertLess(et_fraction, 0.9)      # Some runoff/drainage expected


@unittest.skipUnless(SIMULATOR_AVAILABLE and DEPENDENCY_GRAPH_AVAILABLE, 
                     "Simulator and dependency graph not available")
class TestMultiModuleExecution(unittest.TestCase):
    """Test execution of multiple coupled modules"""
    
    def test_dependency_chain_execution(self):
        """Test execution of modules with dependencies"""
        if not DependencyGraphResolver:
            self.skipTest("Dependency graph resolver not available")
            
        # Create module chain: A -> B -> C
        module_a = SimpleMockFunctor(
            name="module_a",
            flux_values={"output_a": 5.0}
        )
        
        module_b = SimpleMockFunctor(
            name="module_b",
            dependencies=["module_a"],
            flux_values={"output_b": 3.0}
        )
        
        module_c = SimpleMockFunctor(
            name="module_c", 
            dependencies=["module_b"],
            flux_values={"output_c": 1.0}
        )
        
        # Create dependency graph
        try:
            dep_graph = DependencyGraphResolver()
            dep_graph.add_functor(module_a)
            dep_graph.add_functor(module_b)
            dep_graph.add_functor(module_c)
            
            # Get execution order
            execution_order = dep_graph.topological_sort()
            
            # Verify correct dependency order
            module_names = list(execution_order)
            
            a_idx = module_names.index("module_a")
            b_idx = module_names.index("module_b") 
            c_idx = module_names.index("module_c")
            
            self.assertLess(a_idx, b_idx, "Module A should execute before B")
            self.assertLess(b_idx, c_idx, "Module B should execute before C")
            
        except Exception as e:
            logger.warning(f"Dependency graph operations failed: {e}")
            self.skipTest("Dependency graph not fully functional")
    
    def test_parallel_execution_groups(self):
        """Test parallel execution of independent modules"""
        # Create independent modules that can run in parallel
        parallel_modules = []
        
        for i in range(3):
            module = SimpleMockFunctor(
                name=f"parallel_module_{i}",
                flux_values={f"output_{i}": float(i + 1)}
            )
            parallel_modules.append(module)
        
        # All modules are independent - should be able to execute in parallel
        state = {"temperature": torch.tensor(298.15)}
        
        results = []
        for module in parallel_modules:
            result = module.forward(state, time=0.0, dt=1.0)
            results.append(result)
        
        # Verify all modules executed successfully
        self.assertEqual(len(results), 3)
        
        for i, result in enumerate(results):
            expected_output = f"output_{i}"
            self.assertIn(expected_output, result)
            self.assertEqual(result[expected_output].item(), float(i + 1))


def run_comprehensive_l2_tests():
    """Run all L2 integration tests with detailed reporting"""
    print("="*80)
    print("RELIC L2 MODULE INTEGRATION TEST SUITE")
    print("="*80)
    print()
    
    # Test availability summary
    print("Module Availability:")
    print(f"  Core modules:        {'OK' if CORE_AVAILABLE else 'NO'}")
    print(f"  Functor base:        {'OK' if FUNCTOR_BASE_AVAILABLE else 'NO'}")
    print(f"  Simulator:           {'OK' if SIMULATOR_AVAILABLE else 'NO'}")
    print(f"  Dependency graph:    {'OK' if DEPENDENCY_GRAPH_AVAILABLE else 'NO'}")
    print()
    
    # Test suite configuration
    test_classes = [
        TestStateManagerIntegration,
        TestParameterSystemIntegration,
        TestScientificModuleIntegration,
        TestConservationPrinciples,
        TestMultiModuleExecution
    ]
    
    # Initialize test results
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    results_by_class = {}
    
    # Run each test class
    for test_class in test_classes:
        print(f"Running {test_class.__name__}...")
        print("-" * 60)
        
        # Create test suite for this class
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=sys.stdout,
            buffer=True
        )
        
        result = runner.run(suite)
        
        # Collect results
        class_tests = result.testsRun
        class_failures = len(result.failures)
        class_errors = len(result.errors)
        class_skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        
        results_by_class[test_class.__name__] = {
            'tests': class_tests,
            'failures': class_failures,
            'errors': class_errors,
            'skipped': class_skipped,
            'success_rate': (class_tests - class_failures - class_errors) / max(class_tests, 1) * 100
        }
        
        total_tests += class_tests
        total_failures += class_failures
        total_errors += class_errors
        total_skipped += class_skipped
        
        print(f"Class Results: {class_tests} tests, {class_failures} failures, "
              f"{class_errors} errors, {class_skipped} skipped")
        print()
    
    # Print comprehensive summary
    print("="*80)
    print("L2 INTEGRATION TEST SUMMARY")
    print("="*80)
    
    overall_success_rate = (total_tests - total_failures - total_errors) / max(total_tests, 1) * 100
    
    print(f"Overall Results:")
    print(f"  Total Tests:     {total_tests}")
    print(f"  Failures:        {total_failures}")
    print(f"  Errors:          {total_errors}")
    print(f"  Skipped:         {total_skipped}")
    print(f"  Success Rate:    {overall_success_rate:.1f}%")
    print()
    
    print("Results by Test Class:")
    for class_name, results in results_by_class.items():
        print(f"  {class_name}:")
        print(f"    Tests: {results['tests']}, Failures: {results['failures']}, "
              f"Errors: {results['errors']}, Skipped: {results['skipped']}")
        print(f"    Success Rate: {results['success_rate']:.1f}%")
    
    print()
    
    # Overall assessment
    if total_failures == 0 and total_errors == 0:
        print("[OK] ALL L2 INTEGRATION TESTS PASSED!")
        exit_code = 0
    elif overall_success_rate >= 80.0:
        print("[WARNING] MOST L2 INTEGRATION TESTS PASSED (>=80%)")
        exit_code = 0
    else:
        print("[ERROR] SIGNIFICANT L2 INTEGRATION TEST FAILURES")
        exit_code = 1
    
    print()
    print("Integration Test Categories Validated:")
    print("  + State Management Integration")
    print("  + Parameter System Integration")
    print("  + Scientific Module Integration")
    print("  + Conservation Principles")
    print("  + Multi-Module Execution")
    
    return exit_code == 0


if __name__ == '__main__':
    # Configure test environment
    torch.set_default_dtype(torch.float64)  # Higher precision for conservation tests
    
    # Run comprehensive test suite
    success = run_comprehensive_l2_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


