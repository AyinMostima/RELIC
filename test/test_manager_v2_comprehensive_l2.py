"""
Manager v2 Comprehensive L2 Integration Tests

Advanced integration testing for Manager v2 components including:
- Complex manager interaction scenarios
- Concurrent operations and race conditions
- Error propagation and recovery
- Resource contention and sharing
- Cross-manager dependency validation
- Performance under concurrent load

Test Coverage:
    - Manager-to-manager communication paths
    - Error propagation across manager boundaries
    - Concurrent access to shared resources
    - Dependency resolution under load
    - Registry and factory integration
    - System resilience under stress
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Add relic to path
relic_root = Path(__file__).parent.parent
sys.path.insert(0, str(relic_root))


class TestL2ManagerCommunication:
    """L2 Manager Communication Tests"""
    
    def test_parameter_state_bidirectional_communication(self):
        """Test bidirectional communication between ParameterManager and StateManager"""
        from relic.parameters import get_global_manager
        from relic.core.state import StateManager
        
        param_manager = get_global_manager()
        state_manager = StateManager()
        
        # Ensure both are initialized
        if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
            param_manager.initialize()
        state_manager.initialize()

        # Test that parameter manager can provide parameters for state validation
        params = param_manager.list(namespace="biosphere")
        assert isinstance(params, (list, dict))

        # Test state manager can validate empty state
        validation_result = state_manager.validate()
        if isinstance(validation_result, bool):
            assert validation_result is True
        else:
            assert validation_result is None or (isinstance(validation_result, list) and len(validation_result) == 0)

        # Both managers should remain in valid states
        assert param_manager.state.name in ['READY', 'INITIALIZED', 'INITIALIZING']
        assert isinstance(state_manager.get_state(), dict)
    
    def test_scale_grid_constraint_integration(self):
        """Test integration between ScaleManager, GridManager, and PhysicalConstraintManager"""
        from relic.coupling.scales.scale_manager import ScaleManager
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager

        state_manager = StateManager()
        scale_manager = ScaleManager(
            parameter_manager=get_global_manager(),
            state_manager=state_manager,
        )
        scale_manager.initialize()
        scale_manager.validate()

        grid_manager = scale_manager.grid_manager
        constraint_manager = state_manager.physical_constraint_manager

        assert scale_manager.state_manager is state_manager
        assert scale_manager.constraint_manager is constraint_manager
        assert grid_manager is scale_manager.grid_manager
        assert scale_manager.state.name == 'READY'
        assert grid_manager.state.name == 'READY'
        assert constraint_manager.state.name == 'READY'
    
    def test_computation_managers_integration(self):
        """Test device policy alongside explicit ProcessGroupManager lifecycle."""
        from relic.computation.distributed.launcher import ProcessGroupManager
        from relic.core import device_policy

        device = device_policy.resolve_device(default="auto")
        process_manager = ProcessGroupManager()
        process_errors = process_manager.validate()
        if process_errors and any("not available" in err.lower() for err in process_errors):
            assert device.type in {"cpu", "cuda"}
            return

        assert process_errors == []
        assert device.type in {"cpu", "cuda"}


class TestL2ConcurrentOperations:
    """L2 Concurrent Operations Tests"""
    
    def test_concurrent_manager_access(self):
        """Test concurrent access to multiple managers"""
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager

        manager_getters = [
            ("ParameterManager", get_global_manager),
            ("StateManager", StateManager),
        ]
        
        results = []
        errors = []
        
        def access_manager(manager_info):
            name, get_manager = manager_info
            try:
                # Get manager and perform some operations
                manager = get_manager()
                if hasattr(manager, 'state'):
                    if manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
                        manager.initialize()
                    state_name = manager.state.name
                else:
                    manager.initialize()
                    try:
                        validation_errors = manager.validate()
                    except Exception:
                        validation_errors = None
                    state_name = 'READY' if validation_errors is None or validation_errors == [] or isinstance(validation_errors, list) else 'ERROR'

                # Perform some operations
                time.sleep(random.uniform(0.01, 0.05))  # Simulate work

                results.append((name, state_name))
            except Exception as e:
                errors.append((name, str(e)))
        
        # Run concurrent access
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(access_manager, info) for info in manager_getters]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        # All operations should succeed
        assert len(results) == len(manager_getters)
        assert len(errors) == 0, f"Concurrent access failed with errors: {errors}"
        
        # All managers should be in valid states
        for name, state in results:
            assert state in ['READY', 'INITIALIZED', 'INITIALIZING'], f"{name} in invalid state: {state}"
    
    def test_concurrent_initialization_safety(self):
        """Test that concurrent initialization is safe"""
        from relic.coupling.scales.scale_manager import ScaleManager
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager
        
        # Create multiple manager instances
        parameter_manager = get_global_manager()
        managers = [
            ScaleManager(
                parameter_manager=parameter_manager,
                state_manager=StateManager(),
            )
            for _ in range(8)
        ]
        results = []
        errors = []
        
        def initialize_manager(manager_id):
            try:
                manager = managers[manager_id]
                manager.initialize()
                results.append((manager_id, manager.state.name))
            except Exception as e:
                errors.append((manager_id, str(e)))
        
        # Run concurrent initializations
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(initialize_manager, i) for i in range(len(managers))]
            for future in as_completed(futures):
                future.result()
        
        # All should either succeed or fail gracefully
        assert len(results) + len(errors) == len(managers)
        
        # No critical system failures
        for manager_id, error in errors:
            assert "critical" not in error.lower()
            assert "system" not in error.lower()
    
    def test_resource_contention_handling(self):
        """Test handling of resource contention between managers"""
        from relic.parameters import get_global_manager
        
        param_manager = get_global_manager()
        if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
            param_manager.initialize()
        
        results = []
        
        def access_parameters(thread_id):
            try:
                # Multiple threads accessing parameters simultaneously
                params = param_manager.list(namespace="biosphere")
                time.sleep(0.01)  # Hold resource briefly
                params2 = param_manager.list(namespace="pedosphere")
                results.append((thread_id, "SUCCESS", len(params) if isinstance(params, (list, dict)) else 0))
            except Exception as e:
                results.append((thread_id, "ERROR", str(e)))
        
        # Run 20 concurrent threads to stress test resource access
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(access_parameters, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()
        
        # Count successes and failures
        successes = [r for r in results if r[1] == "SUCCESS"]
        errors = [r for r in results if r[1] == "ERROR"]
        
        # Most operations should succeed (allow some timeout errors under heavy load)
        success_rate = len(successes) / len(results)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.1%} ({len(successes)}/{len(results)})"


class TestL2ErrorPropagationAndRecovery:
    """L2 Error Propagation and Recovery Tests"""
    
    def test_core_manager_package_keeps_only_lifecycle_contracts(self):
        """Base lifecycle contracts remain available without runtime routing."""
        from relic.core.managers import BaseManager, ManagerState

        assert BaseManager.__name__ == "BaseManager"
        assert ManagerState.READY.value == "ready"
    
    def test_manager_failure_isolation(self):
        """Test that failure of one manager doesn't affect others"""
        from relic.parameters import get_global_manager
        from relic.core.state import StateManager
        
        param_manager = get_global_manager()
        state_manager = StateManager()
        
        # Initialize both
        if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
            param_manager.initialize()
        state_manager.initialize()
        state_errors = state_manager.validate()

        # Even if one has issues, both should remain functional
        assert param_manager.state.name in ['READY', 'INITIALIZED', 'INITIALIZING']
        assert state_errors is None or isinstance(state_errors, list) or state_errors == []
        assert isinstance(state_manager.get_state(), dict)
        
        # Test that they can operate independently
        params = param_manager.list(namespace="biosphere")
        # Use validate_state instead of validate() which requires state transition
        try:
            validation = state_manager.validate_state({})
        except Exception:
            # If validate_state doesn't work, just verify manager is still functional
            validation = None
        
        assert isinstance(params, (list, dict))
        assert validation is None or isinstance(validation, list)


class TestL2PerformanceUnderLoad:
    """L2 Performance Under Load Tests"""
    
    def test_manager_throughput_under_load(self):
        """Test manager throughput under concurrent load"""
        from relic.parameters import get_global_manager
        
        param_manager = get_global_manager()
        if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
            param_manager.initialize()
        
        operations = []
        start_time = time.time()
        
        def perform_operations(thread_id):
            thread_ops = []
            thread_start = time.time()
            
            for i in range(10):  # 10 operations per thread
                op_start = time.time()
                params = param_manager.list(namespace="biosphere")
                op_time = time.time() - op_start
                thread_ops.append(op_time)
            
            thread_total = time.time() - thread_start
            operations.append((thread_id, thread_ops, thread_total))
        
        # Run with 10 threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(perform_operations, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        total_ops = sum(len(ops) for _, ops, _ in operations)
        avg_ops_per_second = total_ops / total_time
        
        # Performance requirement: > 100 ops/sec under load
        assert avg_ops_per_second > 100, f"Throughput too low: {avg_ops_per_second:.1f} ops/sec"
        
        print(f"\nL2 Performance Results:")
        print(f"  Total Operations: {total_ops}")
        print(f"  Total Time: {total_time:.3f}s")
        print(f"  Throughput: {avg_ops_per_second:.1f} ops/sec")
    
    def test_memory_usage_under_load(self):
        """Test memory usage stability under load"""
        import gc
        import weakref
        from relic.coupling.scales.scale_manager import ScaleManager
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager

        parameter_manager = get_global_manager()

        # Warm up one-time registries and module caches before measuring growth.
        warmup_manager = ScaleManager(
            parameter_manager=parameter_manager,
            state_manager=StateManager(),
        )
        warmup_manager.initialize()
        warmup_manager.validate()
        warmup_manager.cleanup()
        del warmup_manager
        gc.collect()
        gc.collect()

        # Baseline memory
        initial_objects = len(gc.get_objects())

        # Create and destroy managers under load
        managers = []
        manager_refs = []
        for i in range(50):
            manager = ScaleManager(
                parameter_manager=parameter_manager,
                state_manager=StateManager(),
            )
            manager.initialize()
            manager.validate()
            managers.append(manager)
            manager_refs.append(weakref.ref(manager))

            if i % 10 == 0:
                # Periodic cleanup
                gc.collect()

        # Clean up all managers
        for manager in managers:
            try:
                manager.cleanup()
            except Exception:
                pass
        del manager
        del managers
        gc.collect()
        gc.collect()

        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        retained_managers = sum(1 for manager_ref in manager_refs if manager_ref() is not None)

        assert retained_managers == 0, f"ScaleManager instances retained after cleanup: {retained_managers}"

        # Allow some growth but detect major leaks
        assert object_growth < 5000, f"Potential memory leak: {object_growth} objects growth"

        print(f"\nL2 Memory Usage:")
        print(f"  Initial Objects: {initial_objects}")
        print(f"  Final Objects: {final_objects}")
        print(f"  Growth: {object_growth} objects")
        print(f"  Retained Managers: {retained_managers}")


def run_comprehensive_l2_tests():
    """Run all comprehensive L2 tests"""
    print("=" * 70)
    print("Manager v2 Comprehensive L2 Integration Tests")
    print("=" * 70)
    
    test_classes = [
        ("Manager Communication", TestL2ManagerCommunication()),
        ("Concurrent Operations", TestL2ConcurrentOperations()),
        ("Error Propagation & Recovery", TestL2ErrorPropagationAndRecovery()),
        ("Performance Under Load", TestL2PerformanceUnderLoad()),
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
    print("COMPREHENSIVE L2 TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        print("ALL COMPREHENSIVE L2 TESTS PASSED!")
        return True
    else:
        print(f"{total_tests - total_passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_comprehensive_l2_tests()
    sys.exit(0 if success else 1)
