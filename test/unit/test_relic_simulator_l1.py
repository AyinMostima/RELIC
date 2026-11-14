"""
RELICSimulator L1 Unit Tests

Tests for the core simulation engine with multi-rate execution,
functor orchestration, and multi-scale coupling capabilities.

Mathematical Expressions:
- Time integration: y(t+dt) = y(t) + integral_{t,t+dt} F(y(tau), tau) d tau
- Multi-rate execution: groups = {fast, medium, slow} with different time scales
- Scale coupling: y_scaled = R(y) where R preserves C(y) for conserved quantities
- Conservation: sum updates_i = 0 for conserved quantities

References:
- RELIC Architecture Documentation
- Multi-rate time integration schemes
- Conservative regridding algorithms
"""

import pytest
import torch
import tempfile
import os
from types import SimpleNamespace
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Mock imports to avoid complex dependencies
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from relic.coupling.scales.temporal.core import PeriodIndexer
    import relic.core.simulator.execution_loop as simulator_execution_loop
    import relic.core.simulator.time_ops as simulator_time_ops
    from relic.core.simulator import (
        RELICSimulator, SimulationState, create_simulator
    )
    from relic.core.state import StateManager
    from relic.core.dependency_graph import DependencyGraphResolver
    from relic.core.functor import (
        BiologicalProcessFunctor,
        ExecutionPhase,
        FunctorMetadata,
        ProcessCategory,
    )
    SIMULATOR_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Cannot import simulator components: {e}")
    SIMULATOR_AVAILABLE = False

# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'carbon_pool': 'kg C m-2',
    'export_source_output': 'dimensionless',
    'moisture': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class MockFunctor(BiologicalProcessFunctor):
    """Mock functor for testing."""

    def __init__(self, name: str, execution_group: str = None,
                 characteristic_time_days: float = None,
                 return_value: float = 1.0):
        metadata = FunctorMetadata(
            name=name,
            description="Simulator scheduling test functor",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["RELIC simulator unit-test contract"],
            output_variables=[f"{name}_output"],
        )
        super().__init__(name=name, config={}, metadata=metadata)
        # Only set execution_group if explicitly provided
        if execution_group is not None:
            self.execution_group = execution_group
        # Only set characteristic_time_days if explicitly provided
        if characteristic_time_days is not None:
            self.characteristic_time_days = characteristic_time_days
        self.return_value = return_value
        self.call_count = 0
        self.last_state = None
        self.last_time = None
        self.last_dt = None

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.call_count += 1
        self.last_state = state
        self.last_time = time
        self.last_dt = dt
        
        # Return simple update
        update_value = torch.tensor(self.return_value, dtype=torch.float32)
        return {f"{self.name}_output": update_value}


class MockStateFunctor(BiologicalProcessFunctor):
    """Mock functor reporting explicitly declared state-derived diagnostics."""

    def __init__(self, name: str, execution_group: str = None):
        metadata = FunctorMetadata(
            name=name,
            description="Simulator state-update test functor",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["RELIC simulator unit-test contract"],
            output_variables=["diagnostics.simulator_test.dynamic_state"],
        )
        super().__init__(name=name, config={}, metadata=metadata)
        # Only set execution_group if explicitly provided
        if execution_group is not None:
            self.execution_group = execution_group
        self.call_count = 0

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.call_count += 1
        
        # Report values based on current state without proposing state writes.
        updates = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                # Create small update based on current value
                update = value * 0.01  # 1% change
                updates[f"{key}_update"] = update
        
        return {"diagnostics.simulator_test.dynamic_state": updates}


class TestRELICSimulatorL1:
    """L1 unit tests for RELICSimulator."""

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_initialization_default_config(self):
        """Test simulator initialization with default configuration."""
        config = {
            'time_step': 60.0,
            'max_time': 3600.0,
            'output_frequency': 10,
            'checkpoint_frequency': 600.0
        }
        
        simulator = _declared_simulator(config)
        
        assert simulator.time_step == 60.0
        assert simulator.max_time == 3600.0
        assert simulator.output_frequency == 10
        assert simulator.checkpoint_frequency == 600.0
        assert simulator.current_time == 0.0
        assert simulator.step_count == 0
        assert simulator.state == SimulationState.UNINITIALIZED
        assert len(simulator.functors) == 0
        assert simulator._groups['fast'] == []
        assert simulator._groups['medium'] == []
        assert simulator._groups['slow'] == []

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_initialization_with_auto_scale_enabled(self):
        """Test simulator initialization with auto-scale enabled."""
        config = {
            'time_step': 60.0,
            'enable_auto_scale': True
        }
        
        # Mock ScaleManager to avoid import issues
        with patch('relic.core.simulator.HAS_SCALE_MANAGER', True):
            with patch('relic.core.simulator.ScaleManager') as mock_scale_manager:
                mock_scale_manager.return_value = Mock()
                
                simulator = _declared_simulator(config)
                
                assert simulator.enable_auto_scale is True
                assert simulator.scale_manager is not None

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_initialization_with_calendar_types(self):
        """Test simulator initialization with different calendar types."""
        for calendar_type in ['noleap', '360_day']:
            config = {
                'calendar_type': calendar_type,
                'time_step': 60.0
            }
            
            simulator = _declared_simulator(config)
            
            assert simulator._calendar_type == calendar_type
            assert not hasattr(simulator, "_calendar_month_lengths")

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_time_origin_is_a_date_of_the_simulator_calendar(self):
        """time_origin (ISO date and time of t = 0) is read from the settings and checked in the
        configured calendar; it is optional (None) until a time-stepped forcing read needs it."""
        simulator = _declared_simulator({'calendar_type': '360_day', 'time_step': 60.0})
        assert simulator._time_origin is None
        simulator = _declared_simulator(
            {'calendar_type': '360_day', 'time_step': 60.0, 'time_origin': '2000-02-30T00:00:00'}
        )
        assert simulator._time_origin == '2000-02-30T00:00:00'
        for origin in ('2001-02-29T00:00:00', 'not a date', 20000101):
            with pytest.raises(ValueError, match="time_origin"):
                _declared_simulator({'calendar_type': 'noleap', 'time_step': 60.0, 'time_origin': origin})

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_initialization_with_invalid_calendar_type(self):
        """Invalid scientific runtime calendars must fail fast."""
        config = {
            'calendar_type': 'invalid_calendar',
            'time_step': 60.0
        }

        with pytest.raises(
            ValueError,
            match="Unsupported calendar_type 'invalid_calendar'",
        ):
            _declared_simulator(config)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_initialize_with_components(self):
        """Test simulator initialization with StateManager and DependencyGraph."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        state_manager = StateManager()
        dependency_graph = DependencyGraphResolver()
        
        simulator.initialize(state_manager, dependency_graph)
        
        assert simulator.state_manager is state_manager
        assert simulator.dependency_graph is dependency_graph
        assert simulator.state == SimulationState.INITIALIZED

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_initialize_state_with_initial_conditions(self):
        """Test simulator initialization with initial conditions."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        initial_conditions = {
            'temperature': 295.0,
            'moisture': 0.3,
            'carbon_pool': 100.0
        }
        
        simulator.initialize_state(initial_conditions)
        
        assert simulator.state == SimulationState.INITIALIZED
        assert simulator.state_manager is not None
        assert simulator.dependency_graph is not None
        
        # Check initial state values
        state = simulator.state_manager.get_state()
        assert state['temperature'].item() == 295.0
        assert torch.allclose(state['moisture'], torch.tensor(0.3), atol=1e-6)
        assert state['carbon_pool'].item() == 100.0

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_register_functor_basic(self):
        """Test basic functor registration."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)

        functor = MockFunctor("test_functor", execution_group="fast")
        simulator.register_functor(functor)

        assert "test_functor" in simulator.functors
        assert simulator.functors["test_functor"] is functor
        assert "test_functor" in simulator._groups["fast"]
        # Registration publishes the committed provider view, so the group
        # orders are already frozen for the next step.
        assert simulator._groups_dirty is False
        view = simulator.get_committed_provider_view()
        assert view.provider("test_functor").functor is functor
        assert view.order_fast == ("test_functor",)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_register_functor_with_explicit_group(self):
        """Test functor registration with explicit execution group."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)

        functor = MockFunctor("slow_functor", execution_group="slow")
        simulator.register_functor(functor, execution_group="slow")

        assert "slow_functor" in simulator._groups["slow"]
        assert "slow_functor" not in simulator._groups["fast"]

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_register_functor_duplicate_name(self):
        """Test functor registration with duplicate name."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)

        functor1 = MockFunctor("test_functor", execution_group="fast")
        functor2 = MockFunctor("test_functor", execution_group="medium")

        simulator.register_functor(functor1)

        # Should raise error for duplicate name
        with pytest.raises(ValueError, match="already registered"):
            simulator.register_functor(functor2)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_determine_execution_group_from_attribute(self):
        """Test execution group determination from functor attribute."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Test with explicit execution_group attribute
        functor = MockFunctor("test_functor")
        functor.execution_group = "medium"
        
        group = simulator._determine_execution_group(functor)
        assert group == "medium"

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_determine_execution_group_from_characteristic_time(self):
        """Test execution group determination from characteristic time."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Test fast characteristic time
        functor_fast = MockFunctor("fast_functor")
        functor_fast.characteristic_time_days = 0.5
        assert simulator._determine_execution_group(functor_fast) == "fast"
        
        # Test medium characteristic time
        functor_medium = MockFunctor("medium_functor")
        functor_medium.characteristic_time_days = 5.0
        assert simulator._determine_execution_group(functor_medium) == "medium"
        
        # Test slow characteristic time
        functor_slow = MockFunctor("slow_functor")
        functor_slow.characteristic_time_days = 20.0
        assert simulator._determine_execution_group(functor_slow) == "slow"

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_determine_execution_group_from_config(self):
        """Test execution group determination from config mapping."""
        config = {
            'time_step': 60.0,
            'process_groups': {
                'special_functor': 'medium'
            }
        }
        simulator = _declared_simulator(config)
        
        functor = MockFunctor("special_functor")
        group = simulator._determine_execution_group(functor)
        assert group == "medium"

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_compute_group_execution_orders(self):
        """Test computation of group execution orders."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Register functors in different groups
        simulator.register_functor(MockFunctor("fast1", execution_group="fast"))
        simulator.register_functor(MockFunctor("fast2", execution_group="fast"))
        simulator.register_functor(MockFunctor("medium1", execution_group="medium"))
        simulator.register_functor(MockFunctor("slow1", execution_group="slow"))
        
        # Compute orders
        simulator._compute_group_execution_orders()
        
        assert simulator._groups_dirty is False
        assert len(simulator._order_fast) == 2
        assert len(simulator._order_medium) == 1
        assert len(simulator._order_slow) == 1
        assert "fast1" in simulator._order_fast
        assert "fast2" in simulator._order_fast

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_step_basic_execution(self):
        """Test basic step execution with fast group."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize simulator
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        # Register functor
        functor = MockFunctor("test_functor", execution_group="fast", return_value=2.0)
        simulator.register_functor(functor)
        
        # Execute step
        updates = simulator.step()
        
        assert functor.call_count == 1
        assert functor.last_time == 0.0
        assert torch.allclose(functor.last_dt, torch.tensor(60.0, dtype=functor.last_dt.dtype))
        assert "test_functor_output" in updates
        assert updates["test_functor_output"].item() == 2.0
        assert simulator.step_count == 1
        assert simulator.current_time == 60.0

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_cpu_step_accumulates_real_wall_time(self):
        """CPU execution must contribute nonzero wall time to the simulator total."""
        simulator = _declared_simulator(
            {
                "time_step": 60.0,
                "device": "cpu",
                "output_frequency": 100,
            }
        )
        simulator.initialize_state({"temperature": 295.0})
        simulator.register_functor(MockFunctor("cpu_timing", execution_group="fast"))

        simulator.step()

        assert simulator.total_execution_time > 0.0

    def test_cuda_step_timing_events_require_explicit_profiling(
        self,
        monkeypatch,
    ):
        """CUDA events must not synchronize ordinary non-profiled steps."""
        simulator = Mock(device=torch.device("cuda"), enable_profiling=False)
        monkeypatch.setattr(
            simulator_execution_loop.device_policy,
            "is_cuda_available",
            lambda: True,
        )
        monkeypatch.setattr(
            torch.cuda,
            "Event",
            lambda **_kwargs: pytest.fail(
                "CUDA timing event must not be created when profiling is disabled"
            ),
        )

        assert simulator_execution_loop._initialize_step_timing_events(
            simulator
        ) == (None, None)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_nonadaptive_step_uses_one_transaction_rng_snapshot(self, monkeypatch):
        """Fixed steps must not duplicate the outer transaction RNG snapshot."""
        original_capture = simulator_execution_loop._capture_rng_state
        capture_count = 0

        def counted_capture(simulator):
            nonlocal capture_count
            capture_count += 1
            return original_capture(simulator)

        monkeypatch.setattr(
            simulator_execution_loop,
            "_capture_rng_state",
            counted_capture,
        )
        simulator = _declared_simulator(
            {
                "time_step": 60.0,
                "device": "cpu",
                "adaptive_dt_enabled": False,
            }
        )
        simulator.initialize_state({"temperature": 295.0})
        simulator.register_functor(
            MockFunctor("fixed_step", execution_group="fast")
        )

        simulator.step()

        assert capture_count == 1

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_step_uses_lightweight_state_manager_savepoint(self, monkeypatch):
        """Ordinary steps must not deep-copy durable StateManager history."""
        simulator = _declared_simulator(
            {
                "time_step": 60.0,
                "device": "cpu",
                "adaptive_dt_enabled": False,
            }
        )
        simulator.initialize_state({"temperature": 295.0})
        simulator.register_functor(
            MockFunctor("lightweight_savepoint", execution_group="fast")
        )
        manager = simulator.state_manager
        original_create = manager.create_transaction_savepoint
        calls = 0

        def counted_create():
            nonlocal calls
            calls += 1
            return original_create()

        monkeypatch.setattr(manager, "create_transaction_savepoint", counted_create)
        monkeypatch.setattr(
            manager,
            "checkpoint",
            lambda: pytest.fail(
                "ordinary Simulator.step must not call durable checkpoint"
            ),
        )

        simulator.step()

        assert calls == 1

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_optional_stage_profiling_is_stable_and_semantics_neutral(self):
        """Opt-in stage timing must not alter updates or committed state."""
        expected_stage_keys = {
            "manager_snapshot",
            "runtime_snapshot",
            "functor_state_snapshot",
            "rng_snapshot",
            "fast_group",
            "medium_group",
            "slow_group",
            "distributed_collective",
            "cross_domain_validation",
            "conservation",
            "state_apply_updates",
            "output",
            "diagnostic_export",
            "checkpoint",
            "cpu_gpu_transfer",
            "cuda_compute",
            "functor_forward",
            "functor_validation",
            "functor_conservation",
            "step_compute",
            "step_total",
        }

        def build(enable_profiling):
            simulator = _declared_simulator(
                {
                    "time_step": 60.0,
                    "device": "cpu",
                    "output_frequency": 1,
                    "enable_profiling": enable_profiling,
                }
            )
            simulator.initialize_state({"temperature": 295.0})
            simulator.register_functor(
                MockFunctor("profiled_functor", execution_group="fast", return_value=2.0)
            )
            return simulator

        baseline = build(False)
        profiled = build(True)
        assert baseline.enable_profiling is False
        assert profiled.enable_profiling is True

        baseline_updates = baseline.step()
        profiled_updates = profiled.step()

        torch.testing.assert_close(
            profiled_updates["profiled_functor_output"],
            baseline_updates["profiled_functor_output"],
        )
        baseline_state = baseline.state_manager.get_state()
        profiled_state = profiled.state_manager.get_state()
        assert set(profiled_state) == set(baseline_state)
        for variable_name in baseline_state:
            torch.testing.assert_close(
                profiled_state[variable_name],
                baseline_state[variable_name],
            )

        assert getattr(baseline, "_performance_diagnostics", None) is None
        diagnostics = profiled._performance_diagnostics
        assert diagnostics["schema_version"] == 1
        assert diagnostics["accepted_steps"] == 1
        assert set(diagnostics["last_step_s"]) == expected_stage_keys
        assert set(diagnostics["cumulative_s"]) == expected_stage_keys
        assert diagnostics["last_step_s"]["step_total"] > 0.0
        assert diagnostics["last_step_s"]["step_compute"] > 0.0
        assert diagnostics["last_step_s"]["functor_forward"] > 0.0
        assert diagnostics["last_step_s"]["functor_validation"] > 0.0
        assert diagnostics["last_step_s"]["functor_conservation"] >= 0.0
        assert diagnostics["last_step_s"]["state_apply_updates"] > 0.0
        assert diagnostics["last_step_s"]["cpu_gpu_transfer"] > 0.0
        assert all(value >= 0.0 for value in diagnostics["last_step_s"].values())

        functor_diagnostics = diagnostics["functors"]["profiled_functor"]
        assert functor_diagnostics["calls"] == 1
        assert functor_diagnostics["last_step_calls"] == 1
        assert set(functor_diagnostics["last_step_s"]) == {
            "state_validation",
            "forward",
            "local_conservation",
            "framework_validation",
            "total",
        }
        assert functor_diagnostics["last_step_s"]["forward"] > 0.0
        assert functor_diagnostics["last_step_s"]["framework_validation"] > 0.0

        summary = profiled.get_simulation_summary()
        assert summary["performance"]["accepted_steps"] == 1
        assert summary["performance"]["last_step_s"] == diagnostics["last_step_s"]
        summary["performance"]["accepted_steps"] = 99
        assert profiled._performance_diagnostics["accepted_steps"] == 1

        profiled.reset()
        assert profiled._performance_diagnostics is None

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_failed_step_truncates_append_only_outputs_without_cloning_history(
        self,
        monkeypatch,
    ):
        """Late failure must restore output lengths while preserving historical tensors."""
        simulator = _declared_simulator(
            {
                "time_step": 60.0,
                "device": "cpu",
                "output_frequency": 1,
                "enable_profiling": True,
            }
        )
        simulator.initialize_state({"temperature": 295.0})
        simulator.register_functor(
            MockFunctor("append_only", execution_group="fast", return_value=2.0)
        )
        simulator.step()

        history = simulator.time_series["append_only_output"]
        historical_value = history[-1]
        original_length = len(history)
        original_time = simulator.current_time
        original_step = simulator.step_count
        original_profiled_steps = simulator._performance_diagnostics["accepted_steps"]

        def fail_commit():
            raise RuntimeError("late observer failure")

        monkeypatch.setattr(
            simulator.state_manager,
            "commit_provenance_transaction",
            fail_commit,
        )

        with pytest.raises(RuntimeError, match="late observer failure"):
            simulator.step()

        assert simulator.time_series["append_only_output"] is history
        assert len(history) == original_length
        assert history[-1] is historical_value
        assert simulator.current_time == original_time
        assert simulator.step_count == original_step
        assert simulator._performance_diagnostics["accepted_steps"] == original_profiled_steps

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_failed_step_truncates_provenance_history_without_cloning_records(
        self,
        monkeypatch,
    ):
        """Late failure must drop the failed step's writer records and keep the earlier ones."""
        simulator = _declared_simulator({"time_step": 60.0, "device": "cpu"})
        simulator.initialize_state({"temperature": 295.0})
        simulator.register_functor(
            MockFunctor("provenance_history", execution_group="fast", return_value=2.0)
        )
        simulator.step()
        simulator.step()

        detail = simulator._step_provenance_detail
        records = detail["provenance_history_output"]
        accepted_records = list(records)
        assert len(accepted_records) == 2

        def fail_commit():
            raise RuntimeError("late observer failure")

        monkeypatch.setattr(
            simulator.state_manager,
            "commit_provenance_transaction",
            fail_commit,
        )

        with pytest.raises(RuntimeError, match="late observer failure"):
            simulator.step()

        assert simulator._step_provenance_detail is detail
        assert list(detail) == ["provenance_history_output"]
        assert detail["provenance_history_output"] is records
        assert len(records) == len(accepted_records)
        assert all(
            current is accepted for current, accepted in zip(records, accepted_records)
        )

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_step_runtime_savepoint_restores_provenance_history_exactly(self):
        """Rollback truncates appended records, drops new variables and restores the owner."""
        first, second = ("writer", "replace", 0), ("writer", "replace", 1)
        records = [first, second]
        detail = {"pool": records}
        holder = SimpleNamespace(_step_provenance_detail=detail)

        savepoint = simulator_execution_loop._snapshot_step_runtime(holder)

        records.append(("writer", "replace", 2))
        detail["new_pool"] = [("other", "replace", 2)]
        holder._step_provenance_detail = {"replaced": [("other", "replace", 2)]}

        simulator_execution_loop._restore_step_runtime(holder, savepoint)

        assert holder._step_provenance_detail is detail
        assert detail == {"pool": [first, second]}
        assert detail["pool"] is records
        assert records[0] is first and records[1] is second

        del records[1:]
        with pytest.raises(
            RuntimeError,
            match="_step_provenance_detail history was truncated",
        ):
            simulator_execution_loop._restore_step_runtime(holder, savepoint)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_step_runtime_savepoint_does_not_visit_provenance_history(self, monkeypatch):
        """Opening a step transaction must cost the same for any recorded history length."""
        original_clone = simulator_execution_loop._clone_runtime_value
        visited = 0

        def counted_clone(value):
            nonlocal visited
            visited += 1
            return original_clone(value)

        monkeypatch.setattr(
            simulator_execution_loop,
            "_clone_runtime_value",
            counted_clone,
        )

        def clone_visits(history_length):
            nonlocal visited
            holder = SimpleNamespace(
                _step_provenance_detail={
                    "pool": [
                        ("writer", "replace", step) for step in range(history_length)
                    ]
                }
            )
            visited = 0
            simulator_execution_loop._snapshot_step_runtime(holder)
            return visited

        assert clone_visits(2000) == clone_visits(2)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_step_multi_rate_execution(self):
        """Test multi-rate step execution with different groups."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize simulator
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        # Register functors in different groups
        fast_functor = MockFunctor("fast_functor", execution_group="fast", return_value=1.0)
        medium_functor = MockFunctor("medium_functor", execution_group="medium", return_value=2.0)
        slow_functor = MockFunctor("slow_functor", execution_group="slow", return_value=3.0)
        
        simulator.register_functor(fast_functor)
        simulator.register_functor(medium_functor)
        simulator.register_functor(slow_functor)
        
        # Execute step (only fast group should execute)
        updates = simulator.step()
        
        assert fast_functor.call_count == 1
        assert medium_functor.call_count == 0  # Should not execute (not at day boundary)
        assert slow_functor.call_count == 0    # Should not execute (not at month boundary)
        assert "fast_functor_output" in updates

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_step_with_custom_dt(self):
        """Test step execution with custom time step."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize simulator
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        # Register functor
        functor = MockFunctor("test_functor", execution_group="fast")
        simulator.register_functor(functor)
        
        # Execute step with custom dt
        custom_dt = 120.0
        updates = simulator.step(dt=custom_dt)
        
        assert torch.allclose(functor.last_dt, torch.tensor(custom_dt, dtype=functor.last_dt.dtype))
        assert simulator.current_time == custom_dt

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_run_basic_simulation(self):
        """Test basic simulation run."""
        config = {
            'time_step': 60.0,
            'max_time': 300.0  # 5 minutes
        }
        simulator = _declared_simulator(config)
        
        # Initialize simulator
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        # Register functor
        functor = MockFunctor("test_functor", execution_group="fast", return_value=0.1)
        simulator.register_functor(functor)
        
        # Run simulation
        final_state = simulator.run()
        
        assert simulator.state == SimulationState.COMPLETED
        assert simulator.current_time == 300.0
        assert simulator.step_count == 5  # 300s / 60s
        assert functor.call_count == 5

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_run_with_custom_parameters(self):
        """Test simulation run with custom parameters."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize simulator
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        # Register functor
        functor = MockFunctor("test_functor", execution_group="fast")
        simulator.register_functor(functor)
        
        # Run with custom parameters
        start_time = 100.0
        end_time = 400.0
        dt = 50.0
        
        final_state = simulator.run(start_time=start_time, end_time=end_time, dt=dt)
        
        assert simulator.current_time == end_time
        assert simulator.step_count == 6  # (400-100)/50 = 6
        assert functor.call_count == 6

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_day_boundary_detection(self):
        """Test day boundary detection for medium group execution."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Test day boundary crossing
        # Start just before day boundary
        simulator.current_time = 86300.0  # 23:58:20
        simulator._previous_day_index = 0
        
        # Should detect day boundary for future time; the predicate is pure (D82),
        # the interval bookkeeping only advances through the explicit commit.
        assert simulator._should_execute_medium(future_time=86400.0) is True
        assert simulator._previous_day_index == 0
        simulator_time_ops.commit_rate_group_interval(simulator, "medium", 86400.0)
        assert simulator._previous_day_index == 1
        
        # Should not detect day boundary within same day
        assert simulator._should_execute_medium(future_time=86500.0) is False

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_month_boundary_detection(self):
        """Test month boundary detection for slow group execution."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Test month boundary crossing
        # Start just before month boundary (end of January, noleap calendar)
        simulator.current_time = 31 * 86400.0 - 100.0  # Just before Feb 1
        simulator._previous_month_index = 0
        
        # Should detect month boundary for future time; the predicate is pure (D82).
        assert simulator._should_execute_slow(future_time=31 * 86400.0 + 100.0) is True
        assert simulator._previous_month_index == 0
        simulator_time_ops.commit_rate_group_interval(simulator, "slow", 31 * 86400.0 + 100.0)
        assert simulator._previous_month_index == 1

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_slow_month_boundary_uses_simulator_calendar_bridge(self, monkeypatch):
        """Slow scheduling must not call a second calendar implementation."""
        simulator = _declared_simulator({"calendar_type": "noleap", "time_step": 60.0})
        simulator._slow_interval_seconds = None
        simulator._previous_month_index = 1
        calls = []

        def delegated_month_index(seconds):
            calls.append(seconds)
            return 2

        monkeypatch.setattr(simulator, "_compute_month_index", delegated_month_index)

        assert simulator._should_execute_slow(future_time=60 * 86400.0) is True
        assert calls == [60 * 86400.0]
        assert simulator._previous_month_index == 1
        simulator_time_ops.commit_rate_group_interval(simulator, "slow", 60 * 86400.0)
        assert calls == [60 * 86400.0, 60 * 86400.0]
        assert simulator._previous_month_index == 2

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_time_ops_has_no_second_month_index_owner(self):
        """Calendar month arithmetic is owned only by PeriodIndexer."""
        assert not hasattr(simulator_time_ops, "compute_month_index")

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_computation_day_index(self):
        """Test day index computation."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Test day indices
        assert simulator._compute_day_index(0.0) == 0
        assert simulator._compute_day_index(43200.0) == 0  # Half day
        assert simulator._compute_day_index(86400.0) == 1  # 1 day
        assert simulator._compute_day_index(172800.0) == 2  # 2 days

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_computation_month_index_noleap(self):
        """Test month index computation with noleap calendar."""
        config = {'calendar_type': 'noleap', 'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Test month indices for noleap calendar
        assert simulator._compute_month_index(0.0) == 0  # Jan 1
        assert simulator._compute_month_index(31 * 86400.0) == 1  # Feb 1
        assert simulator._compute_month_index(59 * 86400.0) == 2  # Mar 1 (28 days in Feb)
        assert simulator._compute_month_index(365 * 86400.0) == 12  # Next year Jan

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_computation_month_index_360_day(self):
        """Test month index computation with 360-day calendar."""
        config = {'calendar_type': '360_day', 'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Test month indices for 360-day calendar
        assert simulator._compute_month_index(0.0) == 0  # Jan 1
        assert simulator._compute_month_index(30 * 86400.0) == 1  # Feb 1
        assert simulator._compute_month_index(60 * 86400.0) == 2  # Mar 1
        assert simulator._compute_month_index(360 * 86400.0) == 12  # Next year Jan

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_month_index_delegates_to_canonical_period_indexer(self, monkeypatch):
        """Simulator month scheduling must delegate to the CF calendar owner."""
        calls = {}

        def delegated_month_index(indexer, elapsed_seconds, calendar):
            calls["indexer"] = indexer
            calls["elapsed_seconds"] = elapsed_seconds
            calls["calendar"] = calendar
            return 7

        monkeypatch.setattr(
            PeriodIndexer,
            "elapsed_month_index",
            delegated_month_index,
        )
        simulator = _declared_simulator({"calendar_type": "noleap", "time_step": 60.0})
        elapsed_seconds = torch.tensor(123.0, dtype=torch.float64)

        assert simulator._compute_month_index(elapsed_seconds) == 7
        assert calls["indexer"] is simulator._calendar_period_indexer
        assert torch.equal(calls["elapsed_seconds"], elapsed_seconds)
        assert calls["calendar"] == "noleap"

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_pause_and_resume(self):
        """Test simulation pause and resume functionality."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize simulator
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        # Start running
        simulator.state = SimulationState.RUNNING
        
        # Pause
        simulator.pause()
        assert simulator.state == SimulationState.PAUSED
        
        # Resume
        simulator.resume()
        assert simulator.state == SimulationState.RUNNING

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_reset(self):
        """Test simulation reset functionality."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize and run a step
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        functor = MockFunctor("test_functor", "fast")
        simulator.register_functor(functor)
        simulator.step()
        
        # Verify step was executed
        assert simulator.step_count == 1
        assert simulator.current_time == 60.0
        
        # Reset
        simulator.reset()
        
        # Verify reset
        assert simulator.state == SimulationState.INITIALIZED
        assert simulator.step_count == 0
        assert simulator.current_time == 0.0
        assert len(simulator.time_series) == 0

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_get_current_state(self):
        """Test getting current system state."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize with state manager
        initial_conditions = {'temperature': 295.0, 'moisture': 0.3}
        simulator.initialize_state(initial_conditions)
        
        # Get current state
        state = simulator._get_current_state()
        
        assert 'temperature' in state
        assert 'moisture' in state
        assert 'time' in state
        assert 'step' in state
        assert state['temperature'].item() == 295.0
        assert torch.allclose(state['moisture'], torch.tensor(0.3), atol=1e-6)
        assert state['time'].item() == 0.0
        assert state['step'].item() == 0

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_store_outputs(self):
        """Test output storage functionality."""
        config = {'time_step': 60.0, 'output_frequency': 1}
        simulator = _declared_simulator(config)
        
        # Initialize
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        # Store outputs
        outputs = {
            'temperature_change': torch.tensor(1.0),
            'moisture_change': torch.tensor(0.1)
        }
        simulator._store_outputs(outputs)
        
        assert len(simulator.time_series) == 2
        assert 'temperature_change' in simulator.time_series
        assert 'moisture_change' in simulator.time_series
        assert len(simulator.time_series['temperature_change']) == 1

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_checkpoint_creation(self):
        """Test checkpoint creation and restoration."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize and run a step
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        functor = MockFunctor("test_functor", "fast")
        simulator.register_functor(functor)
        simulator.step()
        
        # Create checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
            receipt = simulator._create_checkpoint(checkpoint_path)

            assert os.path.isfile(checkpoint_path)
            assert receipt["path"] == os.path.abspath(checkpoint_path)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_get_simulation_summary(self):
        """Test simulation summary generation."""
        config = {'time_step': 60.0, 'max_time': 3600.0}
        simulator = _declared_simulator(config)
        
        # Run some steps
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        functor = MockFunctor("test_functor", "fast")
        simulator.register_functor(functor)
        simulator.step()
        simulator.step()
        
        # Get summary
        summary = simulator.get_simulation_summary()
        
        assert summary['state'] == SimulationState.RUNNING.value
        assert summary['current_time'] == 120.0
        assert summary['step_count'] == 2
        assert summary['num_functors'] == 1
        assert summary['time_step'] == 60.0
        assert summary['max_time'] == 3600.0
        assert summary['progress_percent'] == (120.0 / 3600.0) * 100
        assert 'groups' in summary
        assert summary['groups']['fast'] == 1

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_scale_consistency_check_disabled(self):
        """Test scale consistency check when auto-scale is disabled."""
        config = {'time_step': 60.0, 'enable_auto_scale': False}
        simulator = _declared_simulator(config)
        
        # Create state with mixed scales
        state = {
            'patch_var': torch.tensor(1.0),  # Scalar
            'grid_var': torch.ones(10, 10)    # Grid
        }
        
        scale_check = simulator._check_scale_consistency(state)
        
        assert scale_check['needs_regrid'] is False
        assert scale_check['reason'] == 'auto_scale disabled'

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_scale_consistency_check_enabled_no_facade(self):
        """Requesting auto-scale without an available ScaleManager fails at construction."""
        config = {'time_step': 60.0, 'enable_auto_scale': True}

        with patch('relic.core.simulator.HAS_SCALE_MANAGER', False):
            with pytest.raises(RuntimeError, match="enable_auto_scale=True.*unavailable"):
                _declared_simulator(config)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_determine_regrid_strategy_reads_the_declared_spatial_basis(self):
        """The strategy follows resolve_spatial_basis(name, metadata); names carry no meaning."""
        simulator = _declared_simulator({'time_step': 60.0})

        cases = [
            ('soil_carbon', {'spatial_basis': 'area_density'}, 'conservative'),
            ('carbon_total', {'spatial_basis': 'cell_total'}, 'conservative'),
            ('air_temperature', {'variable_type': 'intensive'}, 'conservative'),
            ('land_type', {'spatial_basis': 'area_density'}, 'conservative'),
            ('carbon_pool', {'spatial_basis': 'categorical'}, 'nearest'),
            ('temperature', {'variable_type': 'categorical'}, 'nearest'),
        ]
        for name, metadata, expected in cases:
            assert simulator._determine_regrid_strategy(name, metadata) == expected, name

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_determine_regrid_strategy_rejects_missing_or_ambiguous_basis(self):
        """Undeclared, bare-extensive, or inconsistent declarations raise; no name fallback."""
        simulator = _declared_simulator({'time_step': 60.0})

        with pytest.raises(ValueError, match="spatial_basis"):
            simulator._determine_regrid_strategy('carbon_pool', {})
        with pytest.raises(ValueError, match="ambiguous"):
            simulator._determine_regrid_strategy('biomass', {'variable_type': 'extensive'})
        with pytest.raises(ValueError, match="inconsistent"):
            simulator._determine_regrid_strategy(
                'soil_carbon', {'variable_type': 'intensive', 'spatial_basis': 'area_density'}
            )
        with pytest.raises(ValueError, match="no metadata"):
            simulator._determine_regrid_strategy('soil_carbon', None)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_simulator_holds_no_spatial_name_heuristics(self):
        """The spatial quantity basis has one owner: relic.coupling.scales.resolve_spatial_basis."""
        import inspect

        import relic.core.simulator.regrid_strategy_ops as strategy_ops
        import relic.core.simulator.regridding_ops as regridding_ops

        simulator = _declared_simulator({'time_step': 60.0})
        assert not hasattr(simulator, '_is_extensive_variable')
        assert not hasattr(simulator, '_derive_target_grid_info')
        assert not hasattr(strategy_ops, 'is_extensive_variable')
        assert not hasattr(strategy_ops, 'derive_target_grid_info')
        for module in (strategy_ops, regridding_ops):
            source = inspect.getsource(module)
            for keyword in ('"biomass"', '"temperature"', '"category"', '"flux"', '.lower()'):
                assert keyword not in source, (module.__name__, keyword)
        assert 'resolve_spatial_basis' in inspect.getsource(strategy_ops)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_cross_domain_validation_no_validator(self):
        """Test cross-domain validation when validator is unavailable."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Should not raise error when validator is not available
        simulator._validate_cross_domain_coupling(['fast', 'medium'])

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_step_from_uninitialized_state(self):
        """Test step execution from uninitialized state should fail."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Should raise error when not initialized
        with pytest.raises(RuntimeError, match="Cannot step simulation in state"):
            simulator.step()

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_step_execution_with_state_manager(self):
        """Test step execution with StateManager integration."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize with state manager
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        # Register state-dependent functor
        functor = MockStateFunctor("state_functor", "fast")
        simulator.register_functor(functor)
        
        # Execute step
        updates = simulator.step()
        
        assert functor.call_count == 1
        assert len(updates) > 0
        # Should have updates based on state variables
        assert any('temperature_update' in key for key in updates.keys())
        assert simulator.get_state()['temperature'].item() == pytest.approx(295.0)
        assert 'temperature_update' not in simulator.get_state()
        assert updates['diagnostics.simulator_test.dynamic_state.temperature_update'].item() == pytest.approx(2.95)

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_factory_function(self):
        """Test simulator factory function."""
        config = {
            'time_step': 120.0,
            'max_time': 7200.0
        }
        
        simulator = create_simulator(config)
        
        assert isinstance(simulator, RELICSimulator)
        assert simulator.time_step == 120.0
        assert simulator.max_time == 7200.0


class TestRELICSimulatorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_step_without_state_manager(self):
        """Test step execution fails fast when StateManager is missing."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Manually set to initialized state
        simulator.state = SimulationState.INITIALIZED
        
        # Register functor
        functor = MockFunctor("test_functor", execution_group="fast")
        simulator.register_functor(functor)

        with pytest.raises(RuntimeError, match="StateManager is required"):
            simulator.step()

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_functor_with_dependencies(self):
        """Test functor registration with dependencies."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Create functors with dependencies
        functor1 = MockFunctor("functor1", execution_group="fast")
        functor2 = MockFunctor("functor2", execution_group="fast")
        functor2.get_dependencies = lambda: ["functor1"]
        
        # Register functors
        simulator.register_functor(functor1)
        simulator.register_functor(functor2)
        
        # Check that dependency graph was updated
        assert simulator.dependency_graph is not None

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_empty_group_execution(self):
        """Test execution of empty groups."""
        config = {'time_step': 60.0}
        simulator = _declared_simulator(config)
        
        # Initialize but don't register any functors
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        # Execute step
        updates = simulator.step()
        
        # Should complete without error
        assert isinstance(updates, dict)
        assert len(updates) == 0

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_large_time_step_simulation(self):
        """Test simulation with large time steps."""
        config = {
            'time_step': 3600.0,  # 1 hour
            'max_time': 86400.0  # 1 day
        }
        simulator = _declared_simulator(config)
        
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        functor = MockFunctor("test_functor", "fast")
        simulator.register_functor(functor)
        
        # Run simulation
        final_state = simulator.run()
        
        assert simulator.step_count == 24  # 24 hours
        assert functor.call_count == 24

    @pytest.mark.skipif(not SIMULATOR_AVAILABLE, reason="Simulator components not available")
    def test_simulation_with_zero_time_step(self):
        """Test simulation behavior with very small time steps."""
        config = {'time_step': 1.0}  # 1 second
        simulator = _declared_simulator(config)
        
        initial_conditions = {'temperature': 295.0}
        simulator.initialize_state(initial_conditions)
        
        functor = MockFunctor("test_functor", "fast")
        simulator.register_functor(functor)
        
        # Execute single step
        updates = simulator.step()
        
        assert simulator.current_time == 1.0
        assert functor.call_count == 1


@pytest.mark.parametrize(
    "failure", ["write_failure", "missing_writer", "missing_output", "missing_manager"],
)
def test_requested_diagnostic_export_failure_is_visible(
    failure, tmp_path, monkeypatch,
):
    from relic.core.state import ConservationSummary
    from relic.core.simulator.diagnostics import export_diagnostics
    from relic.data.io.writer import DataWriter

    simulator = _declared_simulator({
        "device": "cpu",
        "conservation_checks": False,
        "strict_dependencies": False,
        "enable_diagnostic_export": True,
        "diagnostic_output_path": str(tmp_path / "diagnostics.nc"),
        "diagnostic_output_frequency": 1,
    })
    simulator.initialize_state({"export_source_output": torch.tensor(0.0)})
    simulator.register_functor(MockFunctor("export_source", "fast", return_value=2.0))
    simulator.state_manager._conservation_history.extend(
        ConservationSummary(timestamp=timestamp, balance_error={"C": 0.0}, total_violations=0, tolerance_value=1e-6)
        for timestamp in (0.0, 0.5)
    )
    if failure == "write_failure":
        def fail_write(self, *args, **kwargs):
            assert simulator.state_manager._provenance_transaction_active is False
            raise OSError("injected diagnostic disk failure")

        monkeypatch.setattr(DataWriter, "_write_with_format", fail_write)
        with pytest.raises(OSError, match="injected diagnostic disk failure"):
            simulator.step(dt=1.0)
        assert simulator.current_time == 1.0
        assert simulator.step_count == 1
        assert simulator.get_state()["export_source_output"].item() == 2.0
        assert len(simulator._pending_external_deliveries) == 1
        pending = simulator._pending_external_deliveries[0]
        assert pending["operation"] == "diagnostic"
        assert pending["snapshot"] is not None
        assert pending["attempts"] == 1
        assert "injected diagnostic disk failure" in pending["error"]
    else:
        attribute, message = {
            "missing_writer": ("diagnostic_writer", "DiagnosticWriter"),
            "missing_output": ("diagnostic_output_path", "output path"),
            "missing_manager": ("state_manager", "StateManager"),
        }[failure]
        setattr(simulator, attribute, None)
        with pytest.raises(RuntimeError, match=message):
            export_diagnostics(simulator)
    assert not (tmp_path / "diagnostics.nc").exists()


def test_diagnostic_export_snapshot_is_independent_of_live_runtime(tmp_path, monkeypatch):
    from relic.core.state import ConservationSummary
    from relic.core.simulator.diagnostics import capture_diagnostic_export, export_diagnostics
    import xarray as xr

    path = tmp_path / "frozen.nc"
    simulator = _declared_simulator({
        "device": "cpu", "enable_diagnostic_export": True,
        "diagnostic_output_path": str(path),
    })
    simulator.initialize_state({"temperature": torch.tensor(295.0)})
    summary = ConservationSummary(timestamp=1.0, balance_error={"C": 0.0002}, total_violations=0, tolerance_value=1e-6)
    simulator.state_manager._conservation_history.append(summary)
    simulator.diagnostic_reference_date = "2000-01-01"
    simulator.diagnostic_writer.metadata_registry["carbon_balance_error"] = {
        **simulator.diagnostic_writer.metadata_registry["carbon_balance_error"],
        "long_name": "Frozen residual",
    }
    snapshot = capture_diagnostic_export(simulator)
    assert snapshot is not None
    from relic.data.io.diagnostic_writer import DiagnosticWriter
    monkeypatch.setattr(
        DiagnosticWriter, "convert_summaries_to_dataset",
        lambda *a, **k: pytest.fail("Frozen diagnostics must not be converted again"),
    )
    summary.balance_error["C"] = 7.0
    simulator.diagnostic_writer.metadata_registry["carbon_balance_error"]["long_name"] = "Changed residual"
    simulator.diagnostic_reference_date = "2050-01-01"
    simulator.diagnostic_output_path = tmp_path / "other.nc"
    simulator.state_manager._conservation_history.clear()
    result = export_diagnostics(object(), snapshot=snapshot, delivery_key="accepted:1")
    assert result.validation_passed
    assert result.output_path == path
    before = path.read_bytes()
    repeated = export_diagnostics(object(), snapshot=snapshot, delivery_key="accepted:1")
    assert repeated.replayed and repeated.sha256 == result.sha256
    assert path.read_bytes() == before
    assert not (tmp_path / "other.nc").exists()
    with xr.open_dataset(path, decode_times=False) as stored:
        assert stored.carbon_balance_error.item() == pytest.approx(0.0002)
        assert stored.carbon_balance_error.attrs["long_name"] == "Frozen residual"
        assert "2000-01-01" in stored.time.attrs["units"]


def test_explicit_diagnostic_capture_uses_requested_target_without_periodic_switch(tmp_path):
    from relic.core.state import ConservationSummary
    from relic.core.simulator.diagnostics import capture_diagnostic_export, export_diagnostics

    simulator = _declared_simulator({"device": "cpu"})
    simulator.initialize_state({"temperature": torch.tensor(295.0)})
    simulator.state_manager._conservation_history.append(
        ConservationSummary(timestamp=1.0, balance_error={"C": 0.0002}, total_violations=0, tolerance_value=1e-6)
    )
    path = tmp_path / "explicit.nc"
    snapshot = capture_diagnostic_export(simulator, output_path=path, reference_date="2001-01-01")
    result = export_diagnostics(object(), snapshot=snapshot, delivery_key="explicit")
    assert result.validation_passed and path.exists()
    assert simulator.enable_diagnostic_export is False


@pytest.mark.parametrize("reference_date", ["not-a-date", "2001-02-29", "", False])
def test_diagnostic_capture_validates_reference_before_acceptance(tmp_path, reference_date):
    from relic.core.simulator.diagnostics import capture_diagnostic_export
    from relic.coupling.scales.temporal.core import ValidationError
    from relic.core.state import ConservationSummary

    simulator = _declared_simulator({"device": "cpu"})
    simulator.initialize_state({"temperature": torch.tensor(295.0)})
    simulator.state_manager._conservation_history.append(
        ConservationSummary(timestamp=1.0, balance_error={"C": 0.0002}, total_violations=0, tolerance_value=1e-6)
    )
    path = tmp_path / "invalid.nc"
    with pytest.raises((TypeError, ValueError, ValidationError), match="reference|CF|date"):
        capture_diagnostic_export(simulator, output_path=path, reference_date=reference_date)
    assert not path.exists()
    assert simulator.current_time == 0.0
    assert simulator.state_manager.conservation_history[0].balance_error["C"] == 0.0002


@pytest.mark.parametrize("invalid", ["nonscalar", "collision"])
def test_diagnostic_capture_rejects_static_payload_failures_before_publication(tmp_path, invalid):
    from relic.core.simulator.diagnostics import capture_diagnostic_export
    from relic.core.state import ConservationSummary
    from xarray import MergeError

    simulator = _declared_simulator({"device": "cpu"})
    simulator.initialize_state({"temperature": torch.tensor(295.0)})
    if invalid == "nonscalar":
        summaries = [ConservationSummary(timestamp=0.0, custom_checks={"vector": torch.tensor([1.0, 2.0])}, total_violations=0, tolerance_value=1e-6)]
    else:
        summaries = [
            ConservationSummary(timestamp=0.0, custom_checks={"a.b": 1.0}, total_violations=0, tolerance_value=1e-6),
            ConservationSummary(timestamp=1.0, custom_checks={"a_b": 2.0}, total_violations=0, tolerance_value=1e-6),
        ]
    simulator.state_manager._conservation_history.extend(summaries)
    path = tmp_path / "uncreated" / "invalid.nc"
    with pytest.raises((TypeError, ValueError, MergeError)):
        capture_diagnostic_export(simulator, output_path=path)
    assert not path.parent.exists()
    assert simulator.step_count == 0 and simulator.current_time == 0.0


def run_relic_simulator_l1_tests():
    """Run all RELICSimulator L1 unit tests."""
    print("Running RELICSimulator L1 Unit Tests...")
    print("=" * 60)
    
    if not SIMULATOR_AVAILABLE:
        print("WARNING: RELICSimulator components not available. Install dependencies to run tests.")
        return False
    
    test_instance = TestRELICSimulatorL1()
    test_edge_cases = TestRELICSimulatorEdgeCases()
    
    all_test_methods = []
    
    # Collect all test methods
    for test_class in [test_instance, test_edge_cases]:
        methods = [method for method in dir(test_class) if method.startswith('test_')]
        all_test_methods.extend([getattr(test_class, method) for method in methods])
    
    passed = 0
    failed = 0
    
    for test_method in all_test_methods:
        try:
            print(f"Running {test_method.__name__}... ", end="")
            test_method()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RELICSimulator L1 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All RELICSimulator L1 unit tests PASSED!")
        return True
    else:
        print("Some RELICSimulator L1 tests FAILED. Check the implementation.")
        return False


def test_cross_domain_validation_tracks_only_evaluable_couplings():
    """BiosphereAtmosphereCoupling emits only boundary_layer_conductance.

    Its water and energy budgets therefore cannot be evaluated from its outputs, so the
    simulator does not track it; soil-vegetation and water-carbon checks stay tracked.
    """
    from types import SimpleNamespace

    from relic.core.simulator import validation_ops

    functor = SimpleNamespace(metadata=SimpleNamespace(output_variables=["x"]))
    names = ["biosphere_atmosphere", "soil_vegetation", "water_carbon"]
    simulator = SimpleNamespace(
        functors={name: functor for name in names},
        _groups={"fast": list(names)},
    )

    assert validation_ops._collect_executed_coupling_modules(simulator, ["fast"]) == [
        "soil_vegetation",
        "water_carbon",
    ]
    assert validation_ops._resolve_coupling_type("biosphere_atmosphere") is None


class TestSimulatorDiagnosticsPropagation:
    """Diagnostics collection and reports raise their failures; no sentinel is stored."""

    def test_a_failing_functor_diagnosis_propagates(self):
        from types import SimpleNamespace

        from relic.core.simulator.diagnostics import collect_functor_diagnostics

        manager = StateManager()

        def failing(state, updates):
            raise RuntimeError("diagnosis broke")

        functor = SimpleNamespace(name="probe", diagnose_conservation=failing)
        simulator = SimpleNamespace(state_manager=manager)
        with pytest.raises(RuntimeError, match="diagnosis broke"):
            collect_functor_diagnostics(simulator, functor, {}, {})
        assert not any(key.startswith("probe.") for key in manager._current_diagnostics)

    def test_empty_functor_diagnostics_store_no_sentinel(self):
        from types import SimpleNamespace

        from relic.core.simulator.diagnostics import collect_functor_diagnostics

        manager = StateManager()
        functor = SimpleNamespace(name="probe", diagnose_conservation=lambda state, updates: {})
        collect_functor_diagnostics(SimpleNamespace(state_manager=manager), functor, {}, {})
        assert not any(key.startswith("probe.") for key in manager._current_diagnostics)

    def test_the_uncalled_group_conservation_report_helper_is_removed(self):
        """StateManager commit-time ledgers own the per-step conservation report;
        the group report helper had no caller and is gone."""
        import relic.core.simulator as simulator_package
        from relic.core.simulator import diagnostics

        assert not hasattr(diagnostics, "generate_group_conservation_report")
        assert not hasattr(simulator_package, "generate_group_conservation_report")
        assert not hasattr(RELICSimulator, "_generate_group_conservation_report")

    @pytest.mark.parametrize("count", [2.5, "abc", True, 0])
    def test_an_invalid_substep_count_raises(self, count):
        simulator = _declared_simulator({'time_step': 60.0})
        simulator.initialize_state({'temperature': 295.0})
        functor = MockStateFunctor("substep_probe", "fast")
        simulator.register_functor(functor)
        functor.metadata.substep_count = count
        with pytest.raises(Exception) as error:
            simulator.step()
        assert "substep_count" in str(error.value) or "substep_count" in str(error.value.__cause__)

    @pytest.mark.parametrize("count", [2.5, True, "2"])
    def test_metadata_rejects_a_non_integer_substep_count(self, count):
        from relic.core.functor import FunctorMetadata

        with pytest.raises((TypeError, ValueError), match="substep_count"):
            FunctorMetadata(name="substep_meta", references=["probe"], output_variables=["x"], substep_count=count)


class TestSimulatorOptionalCapabilities:
    """Absence is recorded; failures of RELIC's own modules and requested capabilities raise."""

    def test_functor_validation_findings_reach_the_simulator_report(self):
        """A governed call's framework validation findings (here an error the
        Functor explicitly downgrades) land in the simulator's queryable report."""
        from relic.validation.framework import functor_integration
        from relic.validation.framework.registry import RuleMetadata, get_global_registry

        simulator = _declared_simulator({'time_step': 60.0})
        simulator.initialize_state({'temperature': 295.0})
        functor = MockStateFunctor("validation_report_probe", "fast")
        functor._original_config["strict_validation"] = False
        simulator.register_functor(functor)

        def forced_rule(context):
            results = []
            functor_integration._add_error(
                results, message="forced simulator error", category="probe",
                layer="L1", scope=functor.name,
            )
            return results

        registry = get_global_registry()
        registry.register_rule(
            forced_rule,
            RuleMetadata(
                rule_id="validation_report_probe_forced",
                name="forced findings",
                description="Simulator validation report probe",
                layer="L1",
                tags=[functor.name],
                enabled=True,
            ),
            overwrite=True,
        )
        try:
            simulator.step()
        finally:
            registry.unregister_rule("validation_report_probe_forced")

        findings = simulator.get_validation_findings()
        assert [record["functor"] for record in findings] == ["validation_report_probe"]
        assert findings[0]["disposition"] == "downgraded"
        assert [entry["message"] for entry in findings[0]["errors"]] == ["forced simulator error"]
        findings[0]["errors"].clear()
        assert simulator.get_validation_findings()[0]["errors"]

    def test_optional_symbol_probe_distinguishes_absence_from_failure(self, tmp_path, monkeypatch):
        import relic.core.simulator.main as simulator_main

        (tmp_path / "relic_probe_broken_module.py").write_text(
            "raise RuntimeError('module body failed')\n", encoding="utf-8"
        )
        (tmp_path / "relic_probe_partial_module.py").write_text("present = 1\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))

        assert simulator_main._load_optional_symbols("relic_probe_absent_module", ("x",)) == (False, {"x": None})
        with pytest.raises(RuntimeError, match="module body failed"):
            simulator_main._load_optional_symbols("relic_probe_broken_module", ("x",))
        with pytest.raises(ImportError, match="missing"):
            simulator_main._load_optional_symbols("relic_probe_partial_module", ("present", "missing"))

    def test_cross_domain_validator_import_failure_propagates(self, monkeypatch):
        import relic.core.simulator.main as simulator_main

        def _broken(name):
            raise RuntimeError(f"{name} failed at import")

        monkeypatch.setattr(simulator_main, "CrossDomainValidator", None)
        monkeypatch.setattr(simulator_main, "import_module", _broken)
        with pytest.raises(RuntimeError, match="failed at import"):
            simulator_main._load_cross_domain_validator()

    def test_requested_diagnostic_export_is_never_switched_off(self):
        import logging
        from types import SimpleNamespace

        from relic.core.simulator.init_ops import initialize_diagnostic_export

        def _settings(key, default=None):
            return default

        class _BrokenWriter:
            def __init__(self):
                raise OSError("output directory is read-only")

        simulator = SimpleNamespace(enable_diagnostic_export=True, _get_setting=_settings)
        with pytest.raises(OSError, match="read-only"):
            initialize_diagnostic_export(simulator, logging.getLogger(__name__), True, _BrokenWriter)
        assert simulator.enable_diagnostic_export is True
        with pytest.raises(RuntimeError, match="DiagnosticWriter"):
            initialize_diagnostic_export(simulator, logging.getLogger(__name__), False, None)
        assert simulator.enable_diagnostic_export is True

    @staticmethod
    def _cross_domain_simulator(monkeypatch, mode=None):
        from types import SimpleNamespace

        import relic.core.simulator.validation_ops as validation_ops

        config = {"time_step": 60.0}
        if mode is not None:
            config["cross_domain_validation_mode"] = mode
        simulator = _declared_simulator(config)
        simulator.set_state_manager(StateManager())
        simulator.cross_domain_validator = SimpleNamespace()
        monkeypatch.setattr(validation_ops, "_collect_executed_coupling_modules", lambda sim, groups: ["soil_vegetation"])
        monkeypatch.setattr(
            validation_ops,
            "_validate_single_coupling",
            lambda sim, name, state: (["water budget residual 3.2 kg m-2"], ["stress factor at bound"]),
        )
        return simulator

    def test_cross_domain_error_findings_raise_by_default(self, monkeypatch):
        simulator = self._cross_domain_simulator(monkeypatch)
        with pytest.raises(RuntimeError, match="water budget residual"):
            simulator._validate_cross_domain_coupling(["fast", "medium"])
        report = simulator.get_cross_domain_reports()[-1]
        assert report["errors"] == {"soil_vegetation": ["water budget residual 3.2 kg m-2"]}
        assert report["warnings"] == {"soil_vegetation": ["stress factor at bound"]}

    def test_report_mode_records_findings_in_the_queryable_report(self, monkeypatch):
        simulator = self._cross_domain_simulator(monkeypatch, mode="report")
        simulator._validate_cross_domain_coupling(["fast", "medium"])
        report = simulator.get_cross_domain_reports()[-1]
        assert report["mode"] == "report"
        assert report["errors"] == {"soil_vegetation": ["water budget residual 3.2 kg m-2"]}
        assert report["time"] == simulator.current_time and report["step"] == simulator.step_count

    def test_unknown_cross_domain_validation_mode_is_rejected(self):
        with pytest.raises(ValueError, match="cross_domain_validation_mode"):
            _declared_simulator({"time_step": 60.0, "cross_domain_validation_mode": "log"})

    def test_cross_domain_validation_failure_is_not_a_warning(self, monkeypatch):
        from types import SimpleNamespace

        import relic.core.simulator.validation_ops as validation_ops

        def _crashing(simulator, coupling_name, state):
            raise RuntimeError("water budget terms have mismatched shapes")

        monkeypatch.setattr(validation_ops, "_collect_executed_coupling_modules", lambda sim, groups: ["soil_vegetation"])
        monkeypatch.setattr(validation_ops, "_validate_single_coupling", _crashing)
        simulator = SimpleNamespace(
            cross_domain_validator=object(),
            state_manager=SimpleNamespace(_state_variables={}),
        )
        with pytest.raises(RuntimeError, match="mismatched shapes"):
            validation_ops.validate_cross_domain_coupling(simulator, ["fast", "medium"])


if __name__ == "__main__":
    run_relic_simulator_l1_tests()


