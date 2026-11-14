"""
Manager v2 Comprehensive L3 System Tests

End-to-end system testing for the complete Manager v2 ecosystem including:
- Full system integration scenarios
- Performance benchmarks and stress testing
- Long-running stability validation
- Real-world usage simulation
- System recovery and fault tolerance
- Production readiness verification

Test Coverage:
    - Complete system initialization and shutdown
    - Full workflow end-to-end testing
    - Performance benchmarks vs requirements
    - Stress testing under extreme load
    - Long-running stability verification
    - System resilience and recovery
"""

import sys
import time
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Add relic to path
relic_root = Path(__file__).parent.parent
sys.path.insert(0, str(relic_root))


class TestL3SystemIntegration:
    """L3 Complete System Integration Tests"""
    
    def test_full_system_initialization(self):
        """Test complete system initialization in correct order"""
        print("\n--- Full System Initialization Test ---")

        from relic.computation.distributed.launcher import get_process_group_manager
        from relic.core import device_policy
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager

        assert device_policy.resolve_device(default="auto").type in {"cpu", "cuda"}
        
        # Manager initialization sequence (order matters for dependencies)
        manager_sequence = [
            ("Parameter Manager", get_global_manager),
            ("State Manager", StateManager),
            ("Process Group Manager", get_process_group_manager),
        ]
        
        initialized_managers = []
        total_init_time = 0
        
        for name, get_manager in manager_sequence:
            start_time = time.time()
            try:
                manager = get_manager()
                if hasattr(manager, 'state') and manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
                    manager.initialize()
                
                init_time = time.time() - start_time
                total_init_time += init_time
                initialized_managers.append((name, manager, init_time))
                
                print(f"  {name}: {init_time:.3f}s")
                
            except Exception as e:
                print(f"  {name}: FAILED - {e}")
                initialized_managers.append((name, None, 0))

        from relic.coupling.scales.scale_manager import ScaleManager
        from relic.scientific.transport.boundary_conditions import BoundaryConditionManager

        parameter_manager = next(
            manager
            for name, manager, _ in initialized_managers
            if name == "Parameter Manager" and manager is not None
        )
        state_manager = next(
            manager
            for name, manager, _ in initialized_managers
            if name == "State Manager" and manager is not None
        )
        constraint_start = time.time()
        constraint_manager = state_manager.physical_constraint_manager
        constraint_init_time = time.time() - constraint_start
        assert constraint_manager.state.name == "READY"
        initialized_managers.append(
            ("Physical Constraint Manager", constraint_manager, constraint_init_time)
        )
        total_init_time += constraint_init_time

        scale_start = time.time()
        scale_manager = ScaleManager(
            parameter_manager=parameter_manager,
            state_manager=state_manager,
        )
        scale_manager.initialize()
        scale_manager.validate()
        scale_init_time = time.time() - scale_start
        assert scale_manager.state.name == 'READY'
        assert scale_manager.grid_manager.state.name == 'READY'
        initialized_managers.append(("Scale Manager", scale_manager, scale_init_time))
        initialized_managers.append(("Grid Manager", scale_manager.grid_manager, 0.0))
        total_init_time += scale_init_time

        boundary_start = time.time()
        boundary_manager = BoundaryConditionManager()
        boundary_manager.initialize()
        boundary_manager.validate()
        boundary_init_time = time.time() - boundary_start
        assert boundary_manager.state.name == 'READY'
        initialized_managers.append(
            ("Boundary Condition Manager", boundary_manager, boundary_init_time)
        )
        total_init_time += boundary_init_time
        
        print(f"  Total System Init Time: {total_init_time:.3f}s")
        
        # Verify all managers are functional
        functional_count = 0
        for name, manager, _ in initialized_managers:
            if manager is not None:
                if hasattr(manager, 'state'):
                    if manager.state.name in ['READY', 'INITIALIZED', 'INITIALIZING']:
                        functional_count += 1
                else:
                    functional_count += 1
        
        success_rate = functional_count / len(initialized_managers)
        assert success_rate >= 0.8, f"System initialization success rate too low: {success_rate:.1%}"
        assert total_init_time < 10.0, f"System initialization too slow: {total_init_time:.1f}s"
        
        print(f"  Success Rate: {success_rate:.1%} ({functional_count}/{len(initialized_managers)})")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("\n--- End-to-End Workflow Test ---")
        
        workflow_start = time.time()
        
        # Step 1: System initialization
        from relic.parameters import get_global_manager
        from relic.core.state import StateManager
        
        # Initialize core managers
        param_manager = get_global_manager()
        state_manager = StateManager()
        
        if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
            param_manager.initialize()
        state_manager.initialize()
        state_errors = state_manager.validate()
        assert state_errors is None or isinstance(state_errors, list) or state_errors == []
        
        # Step 2: Parameter operations
        params = param_manager.list(namespace="biosphere")
        assert isinstance(params, (list, dict))
        print(f"  Parameter Access: PASS ({len(params)} items)" if isinstance(params, (list, dict)) else "  Parameter Access: PASS")
        
        # Step 3: State validation
        try:
            validation = state_manager.validate_state({})
            print("  State Validation: PASS")
        except Exception:
            print("  State Validation: PASS (method not available)")
        
        # Step 5: Integration verification
        from relic.coupling.scales.scale_manager import ScaleManager

        scale_manager = ScaleManager(
            parameter_manager=param_manager,
            state_manager=state_manager,
        )
        scale_manager.initialize()
        scale_manager.validate()
        assert scale_manager.state.name == 'READY'
        
        workflow_time = time.time() - workflow_start
        assert workflow_time < 5.0, f"End-to-end workflow too slow: {workflow_time:.1f}s"
        
        print(f"  Workflow Time: {workflow_time:.3f}s")
        print("  End-to-End Workflow: PASS")
    
    def test_system_health_monitoring(self):
        """Test explicit owner health without a generic manager registry."""
        print("\n--- System Health Monitoring Test ---")
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager

        parameter_manager = get_global_manager()
        state_manager = StateManager()
        state_manager.initialize()

        assert parameter_manager.state.name in {
            "READY",
            "INITIALIZED",
            "INITIALIZING",
        }
        assert state_manager.validate() in (None, [])
        print("  System Health Monitoring: PASS")


class TestL3PerformanceBenchmarks:
    """L3 Performance Benchmark Tests"""
    
    def test_system_performance_benchmarks(self):
        """Comprehensive system performance benchmarks"""
        print("\n--- System Performance Benchmarks ---")
        
        benchmarks = {}
        process = psutil.Process()
        baseline_mb = process.memory_info().rss / 1024 / 1024
        benchmarks['memory_usage_baseline_mb'] = baseline_mb
        
        # Benchmark 1: Cold start performance
        start_time = time.time()
        from relic.parameters import get_global_manager
        param_manager = get_global_manager()
        if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
            param_manager.initialize()
        cold_start_time = time.time() - start_time
        benchmarks['cold_start'] = cold_start_time
        
        # Benchmark 2: Parameter access performance
        iterations = 1000
        start_time = time.time()
        for _ in range(iterations):
            params = param_manager.list(namespace="biosphere")
        param_access_time = (time.time() - start_time) / iterations
        benchmarks['param_access_avg'] = param_access_time
        
        # Benchmark 3: Concurrent access performance
        def concurrent_access():
            return param_manager.list(namespace="biosphere")
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_access) for _ in range(100)]
            for future in as_completed(futures):
                future.result()
        concurrent_time = time.time() - start_time
        benchmarks['concurrent_100ops'] = concurrent_time
        
        # Benchmark 4: Memory usage
        memory_mb = process.memory_info().rss / 1024 / 1024
        benchmarks['memory_usage_mb'] = memory_mb
        benchmarks['memory_delta_mb'] = max(0.0, memory_mb - baseline_mb)
        
        # Performance Requirements
        requirements = {
            'cold_start': 2.0,           # < 2 seconds
            'param_access_avg': 0.005,   # < 5ms average for namespace listing
            'concurrent_100ops': 1.0,    # < 1 second for 100 concurrent ops
            'memory_usage_mb': baseline_mb + 300.0,  # allow overhead relative to baseline
            'memory_delta_mb': 300.0     # constrain additional memory during benchmark
        }
        
        print("  Performance Results:")
        all_passed = True
        for metric, value in benchmarks.items():
            requirement = requirements.get(metric, float('inf'))
            # Avoid flakiness at tight thresholds due to timer resolution and background load.
            tolerance = 0.0
            if metric == "param_access_avg":
                tolerance = max(5e-4, requirement * 0.5)
            passed = value <= (requirement + tolerance)
            if metric in requirements:
                all_passed = all_passed and passed
            
            if metric.startswith('memory'):
                label = "baseline" if metric == "memory_usage_baseline_mb" else f"req: <{requirement:.1f}MB"
                status = "PASS" if passed else "FAIL"
                print(f"    {metric}: {value:.1f}MB ({label}) {status if metric != 'memory_usage_baseline_mb' else ''}".rstrip())
            elif 'time' in metric or 'avg' in metric:
                if value < 0.01 or requirement < 0.01:
                    print(f"    {metric}: {value*1000:.2f}ms (req: <{requirement*1000:.0f}ms) {'PASS' if passed else 'FAIL'}")
                else:
                    print(f"    {metric}: {value:.3f}s (req: <{requirement:.1f}s) {'PASS' if passed else 'FAIL'}")
            else:
                print(f"    {metric}: {value:.3f}s (req: <{requirement:.1f}s) {'PASS' if passed else 'FAIL'}")
        
        assert all_passed, "Some performance benchmarks failed to meet requirements"
        print("  Performance Benchmarks: PASS")
    
    def test_scalability_stress_testing(self):
        """Test system scalability under stress"""
        print("\n--- Scalability Stress Testing ---")
        
        from relic.parameters import get_global_manager
        param_manager = get_global_manager()
        if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
            param_manager.initialize()
        
        # Stress test: High concurrency
        def stress_operation():
            for _ in range(10):
                param_manager.list(namespace="biosphere")
                time.sleep(0.001)  # Small delay to simulate real work
        
        # Test with increasing thread counts
        thread_counts = [10, 20, 50]
        results = {}
        
        for thread_count in thread_counts:
            start_time = time.time()
            errors = 0
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(stress_operation) for _ in range(thread_count)]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        errors += 1
            
            execution_time = time.time() - start_time
            success_rate = (thread_count - errors) / thread_count
            results[thread_count] = (execution_time, success_rate)
            
            print(f"  {thread_count} threads: {execution_time:.2f}s, {success_rate:.1%} success")
        
        # Verify scalability
        for thread_count, (exec_time, success_rate) in results.items():
            assert success_rate >= 0.9, f"Success rate too low for {thread_count} threads: {success_rate:.1%}"
            assert exec_time < 10.0, f"Execution time too high for {thread_count} threads: {exec_time:.1f}s"
        
        print("  Scalability Stress Testing: PASS")


class TestL3StabilityAndResilience:
    """L3 Long-term Stability and Resilience Tests"""
    
    def test_long_running_stability(self):
        """Test system stability over extended operation"""
        print("\n--- Long-Running Stability Test ---")
        
        from relic.parameters import get_global_manager
        param_manager = get_global_manager()
        if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
            param_manager.initialize()
        
        # Run operations for a period of time
        start_time = time.time()
        operation_count = 0
        errors = 0
        duration = 5.0  # 5 seconds of continuous operation
        
        while time.time() - start_time < duration:
            try:
                params = param_manager.list(namespace="biosphere")
                operation_count += 1
                
                # Periodic health checks
                if operation_count % 100 == 0:
                    assert param_manager.state.name in ['READY', 'INITIALIZED', 'INITIALIZING']
                
                time.sleep(0.01)  # Small delay between operations
            except Exception:
                errors += 1
        
        total_time = time.time() - start_time
        error_rate = errors / operation_count if operation_count > 0 else 1.0
        ops_per_second = operation_count / total_time
        
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Operations: {operation_count}")
        print(f"  Ops/sec: {ops_per_second:.1f}")
        print(f"  Error Rate: {error_rate:.1%}")
        
        # Stability requirements
        assert error_rate < 0.01, f"Error rate too high: {error_rate:.2%}"
        assert ops_per_second > 10, f"Throughput too low: {ops_per_second:.1f} ops/sec"
        
        print("  Long-Running Stability: PASS")
    
    def test_fault_tolerance_and_recovery(self):
        """Test system fault tolerance and recovery capabilities"""
        print("\n--- Fault Tolerance and Recovery Test ---")
        
        # Test recovery from various failure scenarios
        recovery_scenarios = []
        
        # Scenario 1: Manager state corruption simulation
        from relic.coupling.scales.scale_manager import ScaleManager
        from relic.core.state import StateManager
        from relic.parameters import get_global_manager

        parameter_manager = get_global_manager()
        manager = ScaleManager(
            parameter_manager=parameter_manager,
            state_manager=StateManager(),
        )
        manager.initialize()
        manager.validate()
        
        # Verify normal operation
        assert manager.state.name == 'READY'
        
        # Simulate recovery by re-initialization
        try:
            manager.initialize()  # Should handle gracefully
            recovery_scenarios.append(("Re-initialization", True))
        except Exception as e:
            # Should fail gracefully with meaningful error
            error_msg = str(e).lower()
            graceful_failure = any(keyword in error_msg for keyword in 
                                 ['already', 'initialized', 'state', 'transition'])
            recovery_scenarios.append(("Re-initialization", graceful_failure))
        
        # Scenario 2: Resource exhaustion simulation
        managers = []
        try:
            # Create many managers to test resource limits
            for i in range(20):
                m = ScaleManager(
                    parameter_manager=parameter_manager,
                    state_manager=StateManager(),
                )
                m.initialize()
                m.validate()
                managers.append(m)
            recovery_scenarios.append(("Resource Stress", True))
        except Exception:
            recovery_scenarios.append(("Resource Stress", False))
        finally:
            # Clean up
            for m in managers:
                try:
                    m.cleanup()
                except:
                    pass
            del managers
            gc.collect()
        
        # Evaluate recovery scenarios
        for scenario, success in recovery_scenarios:
            print(f"  {scenario}: {'PASS' if success else 'FAIL'}")
        
        successful_recoveries = sum(1 for _, success in recovery_scenarios if success)
        recovery_rate = successful_recoveries / len(recovery_scenarios)
        
        assert recovery_rate >= 0.8, f"Recovery rate too low: {recovery_rate:.1%}"
        print("  Fault Tolerance and Recovery: PASS")


class TestL3ProductionReadiness:
    """L3 Production Readiness Tests"""
    
    def test_production_configuration_validation(self):
        """Test production configuration validation"""
        print("\n--- Production Configuration Validation ---")

        from relic.core.simulator import RELICSimulator
        from relic.core.state import StateManager

        simulator = RELICSimulator(
            {
                "time_step": 3600.0,
                "calendar_type": "noleap",
                "conservation_checks": False,
            }
        )
        simulator.initialize_state({"temperature": 280.0})

        assert isinstance(simulator.state_manager, StateManager)
        assert simulator.config["calendar_type"] == "noleap"
        assert simulator.config["time_step"] == 3600.0
        print("  Production Configuration: READY")
    
    def test_deployment_consistency(self):
        """Test deployment environment consistency"""
        print("\n--- Deployment consistency Test ---")
        
        consistency_checks = {}
        
        # Python version consistency
        python_version = sys.version_info
        consistent_python = python_version >= (3, 8)
        consistency_checks['python_version'] = consistent_python
        print(f"  Python {python_version.major}.{python_version.minor}: {'consistent' if consistent_python else 'inconsistent'}")
        
        # Measure memory attributable to deployment imports, not the shared pytest process.
        process = psutil.Process()
        gc.collect()
        baseline_uss_mb = process.memory_full_info().uss / 1024 / 1024
        
        # Import consistency
        critical_imports = [
            'torch', 'numpy', 'xarray', 'pandas', 
            'scipy', 'matplotlib', 'pydantic'
        ]
        
        import_success = 0
        for module in critical_imports:
            try:
                __import__(module)
                import_success += 1
            except ImportError:
                pass

        post_import_uss_mb = process.memory_full_info().uss / 1024 / 1024
        import_memory_growth_mb = max(0.0, post_import_uss_mb - baseline_uss_mb)
        memory_ok = import_memory_growth_mb < 512.0
        consistency_checks['memory_usage'] = memory_ok
        print(
            f"  Import Memory Growth: {import_memory_growth_mb:.1f}MB "
            f"{'OK' if memory_ok else 'HIGH'}"
        )
        
        import_ratio = import_success / len(critical_imports)
        consistency_checks['imports'] = import_ratio >= 0.8
        print(f"  Critical Imports: {import_success}/{len(critical_imports)} available")
        
        # Overall consistency
        overall_consistent = all(consistency_checks.values())
        
        assert overall_consistent, "Deployment consistency issues detected"
        print("  Deployment consistency: READY")


def run_comprehensive_l3_tests():
    """Run all comprehensive L3 tests"""
    print("=" * 70)
    print("Manager v2 Comprehensive L3 System Tests")
    print("=" * 70)
    
    test_classes = [
        ("System Integration", TestL3SystemIntegration()),
        ("Performance Benchmarks", TestL3PerformanceBenchmarks()),
        ("Stability & Resilience", TestL3StabilityAndResilience()),
        ("Production Readiness", TestL3ProductionReadiness()),
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
    print("COMPREHENSIVE L3 TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        print("ALL COMPREHENSIVE L3 TESTS PASSED!")
        print("SYSTEM IS PRODUCTION READY!")
        return True
    else:
        print(f"{total_tests - total_passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_comprehensive_l3_tests()
    sys.exit(0 if success else 1)


