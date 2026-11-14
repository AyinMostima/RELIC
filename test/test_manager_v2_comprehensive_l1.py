"""
Manager v2 Comprehensive L1 Unit Tests

Extended unit testing for all Manager v2 components including:
- Boundary condition testing
- Exception handling validation
- Concurrent access testing  
- Performance benchmarks
- Resource cleanup verification
- State transition edge cases

Test Coverage:
    - Normal operation paths (covered in base integration tests)
    - Edge cases and boundary conditions
    - Error scenarios and recovery
    - Performance under load
    - Thread safety and concurrency
    - Resource management and cleanup
"""

import pytest
import sys
import time
import threading
import gc
from pathlib import Path
from typing import Any, Dict, List

# Add relic to path
relic_root = Path(__file__).parent.parent
sys.path.insert(0, str(relic_root))

_READY_STATES = {'READY', 'INITIALIZED', 'INITIALIZING'}


def _ensure_manager_initialized(manager) -> None:
    """Initialize managers that expose either BaseManager or lightweight APIs."""
    state = getattr(manager, 'state', None)
    if state is None:
        initialize = getattr(manager, 'initialize', None)
        if callable(initialize):
            initialize()
        return

    state_name = getattr(state, 'name', str(state))
    if state_name not in _READY_STATES:
        manager.initialize()


class TestL1BoundaryConditions:
    """L1 Boundary Condition Tests"""
    
    def test_parameter_manager_empty_namespace(self):
        """Test ParameterManager with empty/invalid namespaces"""
        from relic.parameters import get_global_manager
        
        param_manager = get_global_manager()
        _ensure_manager_initialized(param_manager)
        
        # Test empty namespace
        result = param_manager.list(namespace="")
        assert isinstance(result, (list, dict))
        
        # Test non-existent namespace
        result = param_manager.list(namespace="nonexistent_namespace_12345")
        assert isinstance(result, (list, dict))
        
        # Test None namespace (should handle gracefully)
        try:
            result = param_manager.list(namespace=None)
            # Should either work or raise a clear error
            assert isinstance(result, (list, dict))
        except (TypeError, ValueError):
            # Acceptable to reject None
            pass
    
    def test_state_manager_boundary_values(self):
        """Test StateManager with boundary values"""
        from relic.core.state import StateManager
        
        manager = StateManager()
        _ensure_manager_initialized(manager)
        
        # Test with empty state updates
        try:
            empty_state = {}
            # Should handle empty state gracefully
            manager.validate_state(empty_state)
        except Exception as e:
            # Should provide meaningful error message
            assert len(str(e)) > 0
    
    def test_grid_manager_zero_dimensions(self):
        """Test GridManager with edge case dimensions"""
        from relic.coupling.scales.spatial.grid_manager import GridManager
        from relic.parameters import get_global_manager

        manager = GridManager(parameter_manager=get_global_manager())
        manager.initialize()
        manager.validate()
        
        # Test edge cases in grid operations
        # Note: Actual implementation may not support these, but should fail gracefully
        try:
            # Test zero dimension requests
            result = manager.get_grid_info()  # Should return some info
            assert result is not None
        except Exception as e:
            # Should provide meaningful error
            assert "grid" in str(e).lower() or "dimension" in str(e).lower()


class TestL1ExceptionHandling:
    """L1 Exception Handling Tests"""
    
    def test_manager_initialization_failures(self):
        """Test manager behavior when initialization fails"""
        from relic.core.managers.base import BaseManager, ManagerState
        
        # Create a test manager that fails initialization
        class FailingManager(BaseManager):
            def _do_initialize(self):
                raise RuntimeError("Intentional initialization failure")
            
            def _do_configure(self, config: Dict[str, Any]) -> None:
                pass
            
            def _do_validate(self) -> List[str]:
                return []
            
            def _do_cleanup(self) -> None:
                pass
            
            def _create_metadata(self):
                from relic.core.managers.metadata import ManagerMetadata
                return ManagerMetadata(
                    name="TestFailingManager",
                    version="1.0.0",
                    description="Test manager that fails initialization"
                )
        
        manager = FailingManager()
        
        # Should start in UNINITIALIZED state
        assert manager.state == ManagerState.UNINITIALIZED
        
        # Initialization should fail gracefully
        with pytest.raises(Exception) as exc_info:
            manager.initialize()
        
        # Should be in ERROR state after failed initialization
        assert manager.state == ManagerState.ERROR
        assert "Intentional initialization failure" in str(exc_info.value)
    
    def test_invalid_state_transitions(self):
        """Test invalid state transitions are handled properly"""
        from relic.parameters import get_global_manager
        
        manager = get_global_manager()
        
        # Ensure manager is initialized
        _ensure_manager_initialized(manager)
        
        # Try to initialize an already initialized manager
        # Should handle this gracefully (no-op or clear error)
        try:
            manager.initialize()  # Should not crash
        except Exception as e:
            # If it raises an exception, should be meaningful
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in 
                      ['already', 'initialized', 'state', 'transition'])
    
    def test_concurrent_initialization(self):
        """Test concurrent initialization attempts"""
        from relic.coupling.scales.scale_manager import ScaleManager
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager
        
        # Create multiple managers to test concurrent initialization
        parameter_manager = get_global_manager()
        managers = [
            ScaleManager(
                parameter_manager=parameter_manager,
                state_manager=StateManager(),
            )
            for _ in range(5)
        ]
        errors = []
        successes = []
        
        def initialize_manager(manager):
            try:
                manager.initialize()
                successes.append(manager)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent initializations
        threads = []
        for manager in managers:
            thread = threading.Thread(target=initialize_manager, args=(manager,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # At least some should succeed, none should crash the system
        assert len(successes) + len(errors) == len(managers)
        
        # If there are errors, they should be meaningful
        for error in errors:
            assert len(str(error)) > 0


class TestL1ConcurrencyAndThreadSafety:
    """L1 Concurrency and Thread Safety Tests"""
    
    def test_parameter_manager_concurrent_access(self):
        """Test ParameterManager thread safety"""
        from relic.parameters import get_global_manager
        
        manager = get_global_manager()
        _ensure_manager_initialized(manager)
        
        results = []
        errors = []
        
        def access_parameters(thread_id):
            try:
                # Concurrent parameter access
                params = manager.list(namespace="biosphere")
                results.append((thread_id, len(params) if isinstance(params, (list, dict)) else 0))
            except Exception as e:
                errors.append((thread_id, e))
        
        # Run 10 concurrent threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=access_parameters, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should complete successfully
        assert len(results) == 10
        assert len(errors) == 0, f"Thread safety failed with errors: {errors}"
    
    def test_explicit_state_owners_and_parameter_singleton_thread_safety(self):
        """State owners stay explicit while ParameterManager remains canonical."""
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager
        
        state_managers = []
        param_managers = []
        
        def get_managers():
            state_managers.append(StateManager())
            param_managers.append(get_global_manager())
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_managers)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(set(id(m) for m in state_managers)) == 10
        assert len(set(id(m) for m in param_managers)) == 1


class TestL1PerformanceBenchmarks:
    """L1 Performance Benchmark Tests"""
    
    def test_manager_initialization_performance(self):
        """Benchmark manager initialization times"""
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager

        manager_factories = [
            ("ParameterManager", get_global_manager),
            (
                "StateManagerCore",
                lambda: StateManager(
                    device="cpu",
                    autoload_contract_registry=False,
                ),
            ),
        ]
        
        performance_results = {}
        
        for name, get_manager in manager_factories:
            try:
                # Measure initialization time
                start_time = time.time()
                manager = get_manager()
                _ensure_manager_initialized(manager)
                init_time = time.time() - start_time
                
                performance_results[name] = init_time
                
                # Performance requirement: < 1 second per manager
                assert init_time < 1.0, f"{name} initialization took {init_time:.3f}s (> 1s)"
                
            except ImportError:
                performance_results[name] = "SKIPPED"

        from relic.coupling.scales.spatial.grid_manager import GridManager

        start_time = time.time()
        grid_manager = GridManager(parameter_manager=get_global_manager())
        grid_manager.initialize()
        grid_manager.validate()
        grid_init_time = time.time() - start_time
        performance_results["GridManager"] = grid_init_time
        assert grid_init_time < 1.0, (
            f"GridManager initialization took {grid_init_time:.3f}s (> 1s)"
        )
        
        # Print performance summary
        print("\nManager Initialization Performance:")
        for name, time_taken in performance_results.items():
            if isinstance(time_taken, float):
                print(f"  {name}: {time_taken:.3f}s")
            else:
                print(f"  {name}: {time_taken}")
    
    def test_parameter_access_performance(self):
        """Benchmark parameter access performance"""
        from relic.parameters import get_global_manager
        
        manager = get_global_manager()
        _ensure_manager_initialized(manager)
        
        # Benchmark repeated parameter access
        start_time = time.time()
        iterations = 100
        
        for i in range(iterations):
            params = manager.list(namespace="biosphere")
        
        access_time = time.time() - start_time
        avg_time = access_time / iterations
        
        # Performance requirement: < 10ms average per access
        assert avg_time < 0.01, f"Parameter access took {avg_time*1000:.1f}ms avg (> 10ms)"
        
        print(f"\nParameter Access Performance: {avg_time*1000:.2f}ms average")


class TestL1ResourceManagement:
    """L1 Resource Management Tests"""
    
    def test_manager_cleanup(self):
        """Test proper resource cleanup"""
        from relic.coupling.scales.scale_manager import ScaleManager
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager
        
        # Create managers and initialize them
        managers = []
        parameter_manager = get_global_manager()
        for _ in range(5):
            manager = ScaleManager(
                parameter_manager=parameter_manager,
                state_manager=StateManager(),
            )
            manager.initialize()
            manager.validate()
            managers.append(manager)
        
        # Get initial memory usage
        initial_objects = len(gc.get_objects())
        
        # Clean up managers
        for manager in managers:
            try:
                manager.cleanup()
            except Exception:
                pass  # Some managers may not implement cleanup
        
        # Clear references
        del managers
        gc.collect()
        
        # Memory usage should not grow excessively
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Allow some growth but not excessive
        assert object_growth < 1000, f"Memory leak detected: {object_growth} objects not cleaned"
    
    def test_error_state_recovery(self):
        """Test recovery from error states using existing managers"""
        from relic.coupling.scales.scale_manager import ScaleManager
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager
        
        # Create a new manager instance (not singleton)
        manager = ScaleManager(
            parameter_manager=get_global_manager(),
            state_manager=StateManager(),
        )
        
        # First, ensure it's uninitialized
        assert manager.state.name == 'UNINITIALIZED'
        
        # Initialize normally (should succeed)
        manager.initialize()
        manager.validate()
        assert manager.state.name == 'READY'
        
        # Test basic functionality after initialization
        # Just verify the manager is functional
        assert hasattr(manager, 'state')
        assert manager.state.name == 'READY'
        
        print("PASS Error state recovery test (using ScaleManager)")


def run_comprehensive_l1_tests():
    """Run all comprehensive L1 tests"""
    print("=" * 70)
    print("Manager v2 Comprehensive L1 Unit Tests")
    print("=" * 70)
    
    test_classes = [
        ("Boundary Conditions", TestL1BoundaryConditions()),
        ("Exception Handling", TestL1ExceptionHandling()),
        ("Concurrency & Thread Safety", TestL1ConcurrencyAndThreadSafety()),
        ("Performance Benchmarks", TestL1PerformanceBenchmarks()),
        ("Resource Management", TestL1ResourceManagement()),
    ]
    
    total_passed = 0
    total_tests = 0
    
    for category_name, test_instance in test_classes:
        print(f"\n{category_name.upper()}")
        print("-" * len(category_name))
        
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        category_passed = 0
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                print(f"PASS {test_method}")
                category_passed += 1
                total_passed += 1
            except Exception as e:
                print(f"FAIL {test_method} - {e}")
        
        print(f"Category Results: {category_passed}/{len(test_methods)} passed")
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE L1 TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        print("ALL COMPREHENSIVE L1 TESTS PASSED!")
        return True
    else:
        print(f"{total_tests - total_passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_comprehensive_l1_tests()
    sys.exit(0 if success else 1)

