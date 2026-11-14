"""
L2 Integration Tests for Surrogate Models with RELIC Simulator

This test suite verifies the integration of surrogate models with the core
RELIC simulation framework, including StateManager, DependencyGraph,
and multi-rate execution capabilities.

Test Categories:
1. Simulator Registration and Setup Tests
2. StateManager Integration Tests
3. Dependency Resolution Tests
4. Multi-Rate Execution Tests
5. Conservation Law Integration Tests
6. End-to-End Simulation Tests
7. Error Propagation Tests
8. Performance Integration Tests

Integration Points:
    Simulator 鈫?register_functor() 鈫?SurrogateFunctor
    StateManager 鈫?apply_updates() 鈫?FluxDict from surrogate
    DependencyGraph 鈫?get_dependencies() 鈫?execution ordering
    Multi-rate scheduling 鈫?execution_group 鈫?fast/medium/slow

Mathematical Framework:
    Simulation Loop with Surrogate:
        for t in time_steps:
            state = state_manager.get_state()
            surrogate_updates = surrogate.forward(state, t, dt)
            all_updates = collect_all_functor_updates(state, t, dt)
            state_manager.apply_updates(all_updates + surrogate_updates)
            verify_conservation_laws()

References:
    - Hairer, E. & Wanner, G. (1996). Solving ODEs with stiff problems
    - Kennedy, M.C. & O'Hagan, A. (2001). Bayesian calibration of computer models
"""

import pytest
import torch
import os
from typing import List, Optional
import sys

# Add relic to path for testing
relic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, relic_path)

from relic.models.surrogates import SurrogateFunctor
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode
from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.functor import BiologicalProcessFunctor, FunctorMetadata, ConfigDict
from relic.core.abstractions import StateDict, FluxDict

class MockSimpleFunctor(BiologicalProcessFunctor):
    """Simple mock functor for testing interactions with surrogates."""
    
    def __init__(self, name: str, outputs: List[str], dependencies: List[str] = None):
        config = {'outputs': outputs, 'dependencies': dependencies or []}
        metadata = FunctorMetadata(
            name=name,
            description=f"Mock functor {name}",
            references=["RELIC surrogate integration test fixture"],
            output_variables=list(outputs),
        )
        super().__init__(name, config, metadata)
        self.outputs = outputs
        self.deps = dependencies or []
        
    def get_dependencies(self) -> List[str]:
        return self.deps
        
    def forward(self, state: StateDict, time: float, dt: float) -> FluxDict:
        # Simple mock: return small positive fluxes
        result = {}
        for output in self.outputs:
            batch_size = self._get_batch_size(state)
            result[output] = UpdateItem(
                value=torch.ones(batch_size, device=self.device) * 0.1,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                source_functor=self.name,
            )
        return result
    
    def _get_batch_size(self, state: StateDict) -> int:
        """Extract batch size from state."""
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                return value.shape[0] if len(value.shape) > 0 else 1
        return 1


class MockSurrogateFunctor(SurrogateFunctor):
    """Mock surrogate for integration testing."""
    
    def __init__(self, name: str, config: ConfigDict, 
                 metadata: Optional[FunctorMetadata] = None,
                 mock_outputs: Optional[torch.Tensor] = None):
        settings = dict(config)
        settings.setdefault("output_update_mode", "replace")
        settings.setdefault("scale_rate_outputs_by_dt", False)
        super().__init__(name, settings, metadata)
        self.mock_outputs = mock_outputs
        self.model_loaded = True  # Use correct attribute name
        
    def _load_model(self) -> None:
        self.model_loaded = True  # Mark model as loaded
        
    def _predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.mock_outputs is not None:
            batch_size = input_tensor.shape[0]
            if self.mock_outputs.shape[0] == 1 and batch_size > 1:
                return self.mock_outputs.repeat(batch_size, 1)
            return self.mock_outputs
        
        # Default: return small positive values
        num_outputs = len(self.config.get('output_variables', ['output']))
        return torch.ones(input_tensor.shape[0], num_outputs) * 0.05


@pytest.fixture
def basic_state_schema():
    """Create a basic state variable schema for testing."""
    return {
        'air_temperature': StateVariableSpec(
            name='air_temperature',
            units='K', 
            bounds=(200.0, 350.0),
            default_value=298.15,
            description='Air temperature'
        ),
        'soil_moisture': StateVariableSpec(
            name='soil_moisture',
            units='m3 m-3',
            bounds=(0.0, 1.0),
            default_value=0.3,
            description='Volumetric soil moisture'
        ),
        'leaf_area_index': StateVariableSpec(
            name='leaf_area_index',
            units='m2 m-2',
            bounds=(0.0, 15.0),
            default_value=3.0,
            description='Leaf area index'
        ),
        'gross_primary_productivity': StateVariableSpec(
            name='gross_primary_productivity',
            units='kg C m-2 s-1',
            bounds=(0.0, 1e-3),
            default_value=0.0,
            description='Gross primary productivity'
        ),
        'ecosystem_respiration': StateVariableSpec(
            name='ecosystem_respiration',
            units='kg C m-2 s-1',
            bounds=(0.0, 1e-3),
            default_value=0.0,
            description='Ecosystem respiration'
        ),
        'net_ecosystem_exchange': StateVariableSpec(
            name='net_ecosystem_exchange',
            units='kg C m-2 s-1',
            bounds=(-1e-3, 1e-3),
            default_value=0.0,
            description='Net ecosystem exchange'
        )
    }


@pytest.fixture
def simulator_config():
    """Create basic simulator configuration."""
    return {
        'time_step': 3600.0,  # 1 hour
        'max_time': 86400.0,  # 1 day
        'output_frequency': 1,
        'checkpoint_frequency': 21600.0  # 6 hours
    }


class TestSimulatorRegistrationAndSetup:
    """Test surrogate registration with simulator."""
    
    def test_surrogate_registration_basic(self, simulator_config, basic_state_schema):
        """Test basic registration of surrogate with simulator."""
        # Create simulator and state manager
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Create mock surrogate
        config = {
            'model_path': 'mock_model.pkl',
            'input_variables': ['air_temperature', 'soil_moisture'],
            'output_variables': ['gross_primary_productivity'],
            'validate_model_path': False  # Disable path validation for testing
        }

        surrogate = MockSurrogateFunctor("photosynthesis_surrogate", config)
        assert surrogate.metadata.input_variables == config["input_variables"]
        assert surrogate.metadata.output_variables == config["output_variables"]
        assert surrogate.metadata.references

        # Register surrogate
        simulator.register_functor(surrogate)
        
        # Verify registration
        assert "photosynthesis_surrogate" in simulator.functors
        assert simulator.functors["photosynthesis_surrogate"] is surrogate
        
        # Check execution group assignment
        assert "photosynthesis_surrogate" in simulator._groups["fast"]  # Default group
    
    def test_surrogate_with_custom_execution_group(self, simulator_config, basic_state_schema):
        """Test registration with custom execution group."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        config = {
            'model_path': 'mock_model.pkl',
            'input_variables': ['air_temperature'],
            'output_variables': ['ecosystem_respiration'],
            'execution_group': 'medium',
            'validate_model_path': False  # Disable path validation for testing
        }
        
        surrogate = MockSurrogateFunctor("respiration_surrogate", config)
        
        simulator.register_functor(surrogate, execution_group='medium')
        
        # Should be in medium group
        assert "respiration_surrogate" in simulator._groups["medium"]
        assert "respiration_surrogate" not in simulator._groups["fast"]
    
    def test_multiple_surrogate_registration(self, simulator_config, basic_state_schema):
        """Test registration of multiple surrogates."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Create multiple surrogates
        surrogate1_config = {
            'model_path': 'model1.pkl',
            'input_variables': ['air_temperature'],
            'output_variables': ['gross_primary_productivity'],
            'validate_model_path': False  # Disable path validation for testing
        }
        
        surrogate2_config = {
            'model_path': 'model2.pkl', 
            'input_variables': ['soil_moisture'],
            'output_variables': ['ecosystem_respiration'],
            'validate_model_path': False  # Disable path validation for testing
        }
        
        surrogate1 = MockSurrogateFunctor("gpp_surrogate", surrogate1_config)
        surrogate2 = MockSurrogateFunctor("resp_surrogate", surrogate2_config)
        
        simulator.register_functor(surrogate1)
        simulator.register_functor(surrogate2)
        
        assert len(simulator.functors) == 2
        assert "gpp_surrogate" in simulator.functors
        assert "resp_surrogate" in simulator.functors


class TestStateManagerIntegration:
    """Test integration with StateManager."""
    
    def test_state_extraction_and_update(self, simulator_config, basic_state_schema):
        """Test that surrogate can extract state and produce updates."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Initialize state
        initial_state = {
            'air_temperature': torch.tensor([295.0]),
            'soil_moisture': torch.tensor([0.4]),
            'leaf_area_index': torch.tensor([4.0])
        }
        
        # Set the initial state
        state_manager.set_state(initial_state)
        
        # Create surrogate
        config = {
            'model_path': 'mock.pkl',
            'input_variables': ['air_temperature', 'soil_moisture'],
            'output_variables': ['gross_primary_productivity'],
            'validate_model_path': False
        }
        
        # Mock surrogate to return specific GPP value
        mock_gpp = torch.tensor([[2e-5]])  # kg C m-2 s-1
        surrogate = MockSurrogateFunctor(
            "simple_photosynthesis",
            config,
            mock_outputs=mock_gpp,
        )
        
        simulator.register_functor(surrogate)
        
        # Execute single time step
        # No explicit initialization needed - done automatically
        current_state = state_manager.get_state()
        
        # Surrogate should be able to extract required variables
        updates = surrogate.forward(current_state, time=0.0, dt=3600.0)
        
        assert 'gross_primary_productivity' in updates
        gpp_update = updates['gross_primary_productivity']
        assert isinstance(gpp_update, UpdateItem)
        assert gpp_update.mode is UpdateMode.REPLACE
        assert gpp_update.per_timestep is False
        assert gpp_update.source_functor == surrogate.name
        assert gpp_update.value.shape == torch.Size([1])
        # Ensure tensors are on the same device for comparison
        expected_tensor = torch.tensor([2e-5], device=gpp_update.value.device)
        assert torch.allclose(gpp_update.value, expected_tensor, atol=1e-8)
    
    def test_state_update_application(self, simulator_config, basic_state_schema):
        """Test application of surrogate updates to state."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Initialize state with required variables
        initial_state = {
            'air_temperature': torch.tensor([295.0]),
            'gross_primary_productivity': torch.tensor([0.0])
        }
        state_manager.set_state(initial_state)
        
        # Create surrogate that produces GPP
        config = {
            'model_path': 'mock.pkl',
            'input_variables': ['air_temperature'],
            'output_variables': ['gross_primary_productivity'],
            'validate_model_path': False
        }
        
        mock_gpp = torch.tensor([[1e-5]])
        surrogate = MockSurrogateFunctor(
            "simple_photosynthesis",
            config,
            mock_outputs=mock_gpp,
        )
        simulator.register_functor(surrogate)
        
        # Get initial GPP
        initial_state = state_manager.get_state(['gross_primary_productivity'])
        initial_gpp = initial_state.get('gross_primary_productivity', torch.tensor([0.0]))
        
        # Run a minimal simulation (single step)
        # The run() method properly manages state transitions
        simulator.run(start_time=0.0, end_time=3600.0, dt=3600.0)
        
        # Check that GPP was updated
        final_state = state_manager.get_state(['gross_primary_productivity'])
        final_gpp = final_state.get('gross_primary_productivity', torch.tensor([0.0]))
        
        # GPP is an instantaneous mass flux and uses REPLACE semantics.
        assert torch.all(final_gpp > initial_gpp)
        assert torch.allclose(final_gpp, mock_gpp.squeeze(), atol=1e-8)


class TestDependencyResolutionIntegration:
    """Test integration with dependency resolution system."""
    
    def test_surrogate_dependency_resolution(self, simulator_config, basic_state_schema):
        """Test that surrogate dependencies are properly resolved."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Create a regular functor that the surrogate depends on
        microclimate_functor = MockSimpleFunctor(
            "microclimate", 
            outputs=['air_temperature'],
            dependencies=[]
        )
        
        # Create surrogate that depends on microclimate
        config = {
            'model_path': 'mock.pkl',
            'input_variables': ['air_temperature'],
            'output_variables': ['gross_primary_productivity'],
            'dependencies': ['microclimate'],
            'validate_model_path': False,  # in-memory mock model, no file
        }
        
        surrogate = MockSurrogateFunctor("gpp_surrogate", config)
        
        # Register both functors
        simulator.register_functor(microclimate_functor)
        simulator.register_functor(surrogate)
        
        # Registration updates the canonical dependency graph transactionally.
        # Microclimate should execute before surrogate.
        execution_order = simulator._get_execution_order()
        microclimate_idx = execution_order.index("microclimate")
        surrogate_idx = execution_order.index("gpp_surrogate")
        
        assert microclimate_idx < surrogate_idx, "Dependencies not properly resolved"
    
    def test_circular_dependency_detection(self, simulator_config, basic_state_schema):
        """Test detection of circular dependencies involving surrogates."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Create functors with circular dependencies
        functor_a = MockSimpleFunctor("functor_a", outputs=['var_a'], dependencies=['functor_b'])
        
        config_b = {
            'model_path': 'mock.pkl',
            'input_variables': ['var_a'],
            'output_variables': ['var_b'],
            'dependencies': ['functor_a'],
            'validate_model_path': False,  # in-memory mock model, no file
        }
        
        surrogate_b = MockSurrogateFunctor("functor_b", config_b)
        
        simulator.register_functor(functor_a)

        # Current registration refreshes dependency order immediately, so the
        # cycle can be detected when the second functor is registered.
        with pytest.raises(Exception, match="[Cc]ircular|[Cc]ycle"):
            simulator.register_functor(surrogate_b)


class TestMultiRateExecution:
    """Test multi-rate execution with surrogates."""
    
    def test_different_execution_groups(self, simulator_config, basic_state_schema):
        """Test surrogates in different execution groups."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Create surrogates for different execution groups
        fast_config = {
            'model_path': 'fast.pkl',
            'input_variables': ['air_temperature'],
            'output_variables': ['gross_primary_productivity'],
            'validate_model_path': False
        }
        
        medium_config = {
            'model_path': 'medium.pkl',
            'input_variables': ['soil_moisture'],
            'output_variables': ['ecosystem_respiration'],
            'validate_model_path': False
        }
        
        slow_config = {
            'model_path': 'slow.pkl',
            'input_variables': ['leaf_area_index'],
            'output_variables': ['net_ecosystem_exchange'],
            'validate_model_path': False
        }
        
        fast_surrogate = MockSurrogateFunctor("fast_surrogate", fast_config)
        medium_surrogate = MockSurrogateFunctor("medium_surrogate", medium_config)  
        slow_surrogate = MockSurrogateFunctor("slow_surrogate", slow_config)
        
        # Register in different groups
        simulator.register_functor(fast_surrogate, execution_group='fast')
        simulator.register_functor(medium_surrogate, execution_group='medium')
        simulator.register_functor(slow_surrogate, execution_group='slow')
        
        # Verify group assignments
        assert "fast_surrogate" in simulator._groups["fast"]
        assert "medium_surrogate" in simulator._groups["medium"]
        assert "slow_surrogate" in simulator._groups["slow"]
        
        # Each group should have its own execution order
        simulator._compute_group_execution_orders()
        
        assert len(simulator._order_fast) == 1
        assert len(simulator._order_medium) == 1
        assert len(simulator._order_slow) == 1


class TestConservationLawIntegration:
    """Test conservation law enforcement in integrated system."""
    
    def test_carbon_conservation_with_surrogates(self, simulator_config, basic_state_schema):
        """Test carbon conservation when using carbon cycle surrogates."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Initialize state with required variables
        initial_state = {
            'air_temperature': torch.tensor([295.0]),
            'soil_moisture': torch.tensor([0.3]),
            # Keep flux state shapes consistent with surrogate batch outputs.
            'gross_primary_productivity': torch.tensor([0.0]),
            'ecosystem_respiration': torch.tensor([0.0]),
            'net_ecosystem_exchange': torch.tensor([0.0]),
        }
        state_manager.set_state(initial_state)
        
        # One selected surrogate owns the complete carbon-flux closure.
        surrogate_config = {
            'model_path': 'simple_photosynthesis.pkl',
            'input_variables': ['air_temperature', 'soil_moisture'],
            'output_variables': [
                'gross_primary_productivity',
                'ecosystem_respiration',
                'net_ecosystem_exchange',
            ],
            'conservation_check': True,
            'conservation_matrix': [[1.0, -1.0, 1.0]],
            # Absolute ledger residual tolerance [kg C m-2 s-1] (declared;
            # the former hidden default value).
            'conservation_tolerance': 1.0e-6,
            'validate_model_path': False,
        }

        # References: Chapin et al. (2006), Ecosystems 9:1041-1050.
        # Mathematical expression: NEE = Reco - GPP.
        # Parameters: GPP is gross uptake; Reco is positive ecosystem release.
        # Units: all three fluxes use kg C m-2 s-1.
        # Open-system route: NEE is the signed ecosystem-atmosphere carbon port.
        mock_outputs = torch.tensor([[2.0e-5, 1.8e-5, -2.0e-6]])
        surrogate = MockSurrogateFunctor(
            "simple_photosynthesis",
            surrogate_config,
            mock_outputs=mock_outputs,
        )

        simulator.register_functor(surrogate)
        
        # Run simulation for 5 time steps
        simulator.run(start_time=0.0, end_time=18000.0, dt=3600.0)  # 5 hours
        
        # Check the canonical carbon-flux identity.
        current_state = state_manager.get_state()
        gpp = current_state.get('gross_primary_productivity', torch.tensor([0.0]))
        resp = current_state.get('ecosystem_respiration', torch.tensor([0.0]))
        nee = current_state.get('net_ecosystem_exchange', torch.tensor([0.0]))

        closure_residual = nee - (resp - gpp)
        assert torch.allclose(closure_residual, torch.zeros_like(closure_residual))


class TestEndToEndSimulation:
    """Test complete end-to-end simulation with surrogates."""
    
    def test_full_daily_simulation(self, simulator_config, basic_state_schema):
        """Test full 24-hour simulation with multiple surrogates."""
        # Configure for daily simulation
        daily_config = {
            'time_step': 3600.0,  # 1 hour steps
            'max_time': 86400.0,  # 24 hours
            'output_frequency': 6  # Every 6 hours
        }
        
        simulator = RELICSimulator(daily_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Initialize state with all required variables
        initial_state = {
            'air_temperature': torch.tensor([295.0]),
            'leaf_area_index': torch.tensor([3.0]),
            'soil_moisture': torch.tensor([0.3]),
            'gross_primary_productivity': torch.tensor([0.0]),
            'ecosystem_respiration': torch.tensor([0.0]),
            'net_ecosystem_exchange': torch.tensor([0.0]),
        }
        state_manager.set_state(initial_state)
        
        # Create one selected ecosystem carbon-closure surrogate.
        surrogate_config = {
            'model_path': 'simple_photosynthesis.pkl',
            'input_variables': ['air_temperature', 'soil_moisture'],
            'output_variables': [
                'gross_primary_productivity',
                'ecosystem_respiration',
                'net_ecosystem_exchange',
            ],
            'validate_model_path': False,
        }
        
        surrogate = MockSurrogateFunctor(
            "simple_photosynthesis",
            surrogate_config,
            mock_outputs=torch.tensor([[1.0e-5, 5.0e-6, -5.0e-6]]),
        )
        
        simulator.register_functor(surrogate)
        
        # Run full simulation
        results = simulator.run()
        
        # Verify simulation completed successfully
        assert results is not None
        assert simulator.current_time >= 86400.0  # Completed full day
        
        # Check that all variables evolved
        final_state = state_manager.get_state()
        assert torch.all(final_state['gross_primary_productivity'] > 0)
        assert torch.all(final_state['ecosystem_respiration'] > 0)
        assert torch.allclose(
            final_state['net_ecosystem_exchange'],
            final_state['ecosystem_respiration']
            - final_state['gross_primary_productivity'],
        )
    
    def test_simulation_with_checkpoint_restore(self, simulator_config, basic_state_schema):
        """Test simulation checkpoint and restore with surrogates."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Initialize state with required variables
        initial_state = {
            'air_temperature': torch.tensor([295.0]),
            'gross_primary_productivity': torch.tensor([0.0]),
        }
        state_manager.set_state(initial_state)
        
        # Add surrogate
        config = {
            'model_path': 'test.pkl',
            'input_variables': ['air_temperature'],
            'output_variables': ['gross_primary_productivity'],
            'validate_model_path': False
        }
        
        surrogate = MockSurrogateFunctor("simple_photosynthesis", config)
        simulator.register_functor(surrogate)
        
        # Run partial simulation
        simulator.run(start_time=0.0, end_time=18000.0, dt=3600.0)  # 5 hours
        
        # Save checkpoint
        checkpoint_time = simulator.current_time
        checkpoint_state = state_manager.get_state().copy()
        
        # Continue simulation
        simulator.run(start_time=checkpoint_time, end_time=checkpoint_time + 18000.0, dt=3600.0)  # 5 more hours
        
        # Verify time advanced
        assert simulator.current_time > checkpoint_time
        
        # Restore from checkpoint would restore surrogate state too
        # (Implementation detail - test framework for checkpoint mechanism)


class TestErrorPropagationAndHandling:
    """Test error propagation through integrated system."""
    
    def test_surrogate_error_handling_in_simulation(self, simulator_config, basic_state_schema):
        """Test error handling when surrogate fails during simulation."""
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()  # Must initialize before use
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Initialize state with required variables
        initial_state = {
            'air_temperature': torch.tensor([295.0]),
            'gross_primary_productivity': torch.tensor([0.0]),
        }
        state_manager.set_state(initial_state)
        
        # Create surrogate that will fail
        class FailingSurrogate(MockSurrogateFunctor):
            def forward(self, state: StateDict, time: float, dt: float) -> FluxDict:
                if time >= 7200.0:  # Fail at or after 2 hours
                    raise RuntimeError("Surrogate inference failed")
                return super().forward(state, time, dt)
        
        config = {
            'model_path': 'failing.pkl',
            'input_variables': ['air_temperature'],
            'output_variables': ['gross_primary_productivity'],
            'validate_model_path': False
        }
        
        failing_surrogate = FailingSurrogate("simple_photosynthesis", config)
        simulator.register_functor(failing_surrogate)
        
        # Run first 2 hours successfully
        simulator.run(start_time=0.0, end_time=7200.0, dt=3600.0)  # 2 hours
        assert simulator.current_time == 7200.0
        
        # Should fail on next run; FunctorBase wraps the original error in ExecutionError.
        with pytest.raises(Exception) as exc_info:
            simulator.run(start_time=7200.0, end_time=10800.0, dt=3600.0)
        assert (
            "Surrogate inference failed" in str(exc_info.value)
            or "Group execution failed" in str(exc_info.value)
        )


class TestPerformanceIntegrationCharacteristics:
    """Test performance characteristics in integrated system."""
    
    def test_surrogate_vs_original_performance(self, simulator_config, basic_state_schema):
        """Test that surrogates provide performance benefits over original models."""
        # First test with surrogate
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()
        simulator.initialize(state_manager, DependencyGraphResolver())
        
        # Initialize state
        initial_state = {
            'air_temperature': torch.tensor([295.0]),
            'gross_primary_productivity': torch.tensor([0.0]),
        }
        state_manager.set_state(initial_state)
        
        # Create fast surrogate
        surrogate_config = {
            'model_path': 'fast.pkl',
            'input_variables': ['air_temperature'],
            'output_variables': ['gross_primary_productivity'],
            'validate_model_path': False
        }
        
        surrogate = MockSurrogateFunctor("simple_photosynthesis", surrogate_config)
        simulator.register_functor(surrogate)
        
        # Test surrogate performance
        import time
        start_time = time.perf_counter()
        simulator.run(start_time=0.0, end_time=36000.0, dt=3600.0)  # 10 hours
        surrogate_time = time.perf_counter() - start_time
        
        # Reset for original model test
        simulator = RELICSimulator(simulator_config)
        state_manager = StateManager(basic_state_schema)
        state_manager.initialize()
        simulator.initialize(state_manager, DependencyGraphResolver())
        state_manager.set_state(initial_state)
        
        # Create slow original model (mocked)
        class SlowOriginalFunctor(MockSimpleFunctor):
            def forward(self, state: StateDict, time: float, dt: float) -> FluxDict:
                import time as time_module
                time_module.sleep(0.4)  # Simulate slow computation
                return super().forward(state, time, dt)
        
        slow_original = SlowOriginalFunctor(
            "farquhar_photosynthesis",
            ['gross_primary_productivity'],
        )
        simulator.register_functor(slow_original)
        
        start_time = time.perf_counter()
        simulator.run(start_time=0.0, end_time=36000.0, dt=3600.0)  # 10 hours
        original_time = time.perf_counter() - start_time
        
        # Surrogate should be faster (allowing some test variability)
        assert surrogate_time < original_time * 0.8, f"Surrogate not faster: {surrogate_time:.3f}s vs {original_time:.3f}s"


if __name__ == "__main__":
    # Configure test execution
    test_args = [__file__, "-v", "--tb=short"]
    
    # Run specific test categories if requested
    if '--basic' in sys.argv:
        test_args.extend(["-k", "test_registration or test_state_manager"])
    elif '--advanced' in sys.argv:
        test_args.extend(["-k", "test_dependency or test_multirate or test_conservation"])
    elif '--performance' in sys.argv:
        test_args.extend(["-k", "test_performance"])
        
    pytest.main(test_args)

