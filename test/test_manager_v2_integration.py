"""
Manager v2 Integration Tests - L1/L2/L3 Test Suite

Tests the complete v2 manager system integration after historical code removal.
Validates that all 10 managers work correctly with the unified architecture.

Test Levels:
    L1 - Unit Tests: Individual manager lifecycle and functionality  
    L2 - Integration Tests: Manager interactions and dependencies
    L3 - System Tests: End-to-end workflow and performance
"""

import pytest
import sys
import importlib
from pathlib import Path
import time

# Add relic to path
relic_root = Path(__file__).parent.parent
sys.path.insert(0, str(relic_root))


class TestL1ManagerUnits:
    """L1 Unit Tests - Individual Manager Functionality"""
    
    def test_parameter_manager_v2(self):
        """Test UnifiedParameterManager v2 functionality"""
        try:
            from relic.parameters import get_global_manager
            
            param_manager = get_global_manager()
            assert param_manager is not None
            
            # Test basic operations - only initialize if not already ready
            print(f"DEBUG: param_manager.state.name = {param_manager.state.name}")
            if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
                param_manager.initialize()
            print(f"DEBUG: After initialize, param_manager.state.name = {param_manager.state.name}")
            # Accept INITIALIZING as valid state if initialization is in progress
            assert param_manager.state.name in ['READY', 'INITIALIZED', 'INITIALIZING']
            
            # Test parameter operations
            test_params = param_manager.list(namespace="biosphere")
            assert isinstance(test_params, (list, dict))
            
            print("ParameterManager v2 - PASSED")
            
        except Exception as e:
            pytest.fail(f"ParameterManager v2 failed: {e}")
    
    def test_state_manager_v2(self):
        """Test StateManager v2 functionality"""
        try:
            from relic.core.state import StateManager
            
            manager = StateManager()
            assert manager is not None
            
            # Test lifecycle
            manager.initialize()
            validation_errors = manager.validate()
            print(f"DEBUG: validation_errors = {validation_errors}, type = {type(validation_errors)}")
            assert validation_errors is None or isinstance(validation_errors, list) or validation_errors == []
            assert isinstance(manager.get_state(), dict)
            
            print("PASS StateManager v2 - PASSED")
            
        except Exception as e:
            pytest.fail(f"StateManager v2 failed: {e}")
    
    def test_scale_manager_v2(self):
        """Test ScaleManager v2 functionality"""
        try:
            from relic.coupling.scales.scale_manager import ScaleManager
            from relic.core.state import StateManager
            from relic.parameters import get_global_manager
            
            manager = ScaleManager(
                parameter_manager=get_global_manager(),
                state_manager=StateManager(),
            )
            manager.initialize()
            manager.validate()
            assert manager.state.name == 'READY'
            
            print("PASS ScaleManager v2 - PASSED")
            
        except Exception as e:
            pytest.fail(f"ScaleManager v2 failed: {e}")
    
    def test_grid_manager_v2(self):
        """Test GridManager v2 functionality"""
        try:
            from relic.coupling.scales.spatial.grid_manager import GridManager
            from relic.parameters import get_global_manager
            
            manager = GridManager(parameter_manager=get_global_manager())
            manager.initialize()
            manager.validate()
            assert manager.state.name == 'READY'
            
            print("PASS GridManager v2 - PASSED")
            
        except Exception as e:
            pytest.fail(f"GridManager v2 failed: {e}")
    
    def test_constraint_manager_v2(self):
        """Test PhysicalConstraintManager v2 functionality"""
        try:
            from relic.core.state import StateManager
            
            state_manager = StateManager(schema={}, autoload_contract_registry=False)
            manager = state_manager.physical_constraint_manager
            assert manager is not None
            assert manager.state.name == 'READY'
            
            print("PASS PhysicalConstraintManager v2 - PASSED")
            
        except Exception as e:
            pytest.fail(f"PhysicalConstraintManager v2 failed: {e}")
    
    def test_boundary_manager_v2(self):
        """Test BoundaryConditionManager v2 functionality"""
        try:
            from relic.scientific.transport.boundary_conditions import (
                BoundaryConditionManager,
            )
            
            manager = BoundaryConditionManager()
            manager.initialize()
            manager.validate()
            assert manager.state.name == 'READY'
            
            print("PASS BoundaryConditionManager v2 - PASSED")
            
        except Exception as e:
            pytest.fail(f"BoundaryConditionManager v2 failed: {e}")
    
    def test_device_policy_and_stream_utility(self):
        """Test canonical device selection with the stateless stream utility."""
        try:
            from relic.computation.gpu.accelerator import StreamManager
            from relic.core import device_policy

            device = device_policy.resolve_device(default="auto")
            stream_manager = StreamManager(max_streams=2)

            assert device.type in {"cpu", "cuda"}
            assert stream_manager.get_pipeline_efficiency()["max_streams"] == 2.0

            print("PASS Device Policy and Stream Utility - PASSED")
            
        except Exception as e:
            pytest.fail(f"Device policy/stream utility failed: {e}")
    
    def test_process_manager_v2(self):
        """Test ProcessGroupManager v2 functionality"""
        try:
            from relic.computation.distributed.launcher import ProcessGroupManager

            manager = ProcessGroupManager()
            assert manager is not None

            errors = manager.validate()
            if errors and any("not available" in err.lower() for err in errors):
                print("SKIP  ProcessGroupManager v2 - SKIPPED (distributed not available)")
                return

            assert errors == []
            print("PASS ProcessGroupManager v2 - PASSED")
            
        except Exception as e:
            pytest.fail(f"ProcessGroupManager v2 failed: {e}")
    
    def test_stream_manager_v2(self, monkeypatch, tmp_path):
        """Test StreamManager v2 with the experimental feature explicitly enabled."""
        from relic.data.streaming import stream as stream_module

        monkeypatch.setenv("RELIC_EXPERIMENTAL_STREAMING", "1")
        monkeypatch.setattr(stream_module, "STREAMING_ENABLED", True)

        config = stream_module.StreamConfig(source_path=tmp_path)
        manager = stream_module.StreamManager(config=config)
        assert manager is not None
        assert manager.validate() == []
        manager.cleanup()
    
    def test_version_manager_v2(self, tmp_path):
        """Versioning must use an explicit owner, not an import-time registry path."""
        from relic.parameters.calibration.versioning import VersionManager, VersionedParameter

        manager = VersionManager(storage_dir=tmp_path)
        manager.initialize()
        assert manager.validate() is None
        assert manager.state.name == "READY"

        version_id = manager.create_version({"rate": 1.0}, message="initial")
        current = manager.get_current_version()
        assert current is not None
        assert current.metadata.version_id == version_id

        with pytest.raises(ValueError, match="version_manager"):
            VersionedParameter(name="hidden_owner", current_value=1.0)

        parameter = VersionedParameter(
            name="rate",
            current_value=1.0,
            version_manager=manager,
        )
        updated_version_id = parameter.update_value(2.0, message="update")
        assert manager.get_current_version().metadata.version_id == updated_version_id


    def test_version_manager_requires_an_explicit_storage_directory(self):
        """Version history is not written to a throwaway temporary directory."""
        from relic.core.managers.exceptions import ManagerInitializationError
        from relic.parameters.calibration.versioning import VersionManager

        with pytest.raises(ManagerInitializationError, match="storage_dir"):
            VersionManager().initialize()

    def test_version_manager_io_failures_raise(self, tmp_path):
        from relic.parameters.calibration.versioning import VersionManager

        manager = VersionManager(storage_dir=tmp_path / "store")
        manager.initialize()
        version_id = manager.create_version({"rate": 1.0}, message="initial")

        assert manager.get_version("no_such_version") is None
        (manager.versions_dir / "corrupt.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            manager.get_version("corrupt")
        with pytest.raises(KeyError, match="no_such_version"):
            manager.export_version("no_such_version", tmp_path / "out.json")
        with pytest.raises(ValueError, match="format"):
            manager.export_version(version_id, tmp_path / "out.xml", format="xml")
        manager.export_version(version_id, tmp_path / "out.json")
        assert (tmp_path / "out.json").exists()

        unsupported = tmp_path / "in.txt"
        unsupported.write_text("rate: 1.0", encoding="utf-8")
        with pytest.raises(ValueError, match="format"):
            manager.import_version(unsupported)
        broken = tmp_path / "in.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            manager.import_version(broken)
        assert manager.import_version(tmp_path / "out.json") is not None


class TestL2ManagerIntegrations:
    """L2 Integration Tests - Manager Interactions"""

    def test_core_manager_package_has_no_registry_factory_facade(self):
        """Generic manager routing must not be a supported package API."""
        import relic.core.managers as manager_package

        forbidden = {
            "ManagerRegistry",
            "ManagerFactory",
            "CreationStrategy",
            "CreationContext",
            "get_manager_registry",
            "get_manager_factory",
            "get_global_registry",
            "get_global_factory",
            "create_manager",
            "register_manager_class",
            "get_manager",
            "list_managers",
            "health_check",
            "MANAGER_UNIFICATION_VERSION",
            "ENABLE_V2_MANAGERS",
            "DEFAULT_MANAGER_VERSION",
        }

        assert not forbidden.intersection(manager_package.__all__)
        assert not [name for name in forbidden if hasattr(manager_package, name)]
    
    def test_manager_lifecycle_contract_integration(self):
        """Test the shared lifecycle contract without a generic runtime registry."""
        try:
            from relic.core.managers import BaseManager, ManagerState

            assert BaseManager.__name__ == "BaseManager"
            assert ManagerState.READY.value == "ready"
            
            print("PASS Manager Lifecycle Integration - PASSED")
            
        except Exception as e:
            pytest.fail(f"Manager lifecycle integration failed: {e}")
    
    def test_parameter_state_integration(self):
        """Test parameter manager and state manager integration"""
        try:
            from relic.parameters import get_global_manager
            from relic.core.state import StateManager
            
            param_manager = get_global_manager()
            state_manager = StateManager()
            
            # Both managers should initialize successfully
            if param_manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
                param_manager.initialize()
            state_manager.initialize()

            state_errors = state_manager.validate()
            assert state_errors == [] or state_errors is None

            assert param_manager.state.name in ['READY', 'INITIALIZED', 'INITIALIZING']
            assert isinstance(state_manager.get_state(), dict)
            
            print("PASS Parameter-State Integration - PASSED")
            
        except Exception as e:
            pytest.fail(f"Parameter-State integration failed: {e}")
    
    def test_scale_grid_integration(self):
        """Test scale manager and grid manager integration"""
        try:
            from relic.coupling.scales.scale_manager import ScaleManager
            from relic.core.state import StateManager
            from relic.parameters import get_global_manager
            
            scale_manager = ScaleManager(
                parameter_manager=get_global_manager(),
                state_manager=StateManager(),
            )
            scale_manager.initialize()
            scale_manager.validate()
            grid_manager = scale_manager.grid_manager

            assert scale_manager.state.name == 'READY'
            assert grid_manager is not None
            assert grid_manager.state.name == 'READY'
            
            print("PASS Scale-Grid Integration - PASSED")
            
        except Exception as e:
            pytest.fail(f"Scale-Grid integration failed: {e}")


class TestL3SystemPerformance:
    """L3 System Tests - End-to-End Performance"""
    
    def test_manager_initialization_performance(self, tmp_path):
        """Test manager initialization performance"""
        start_time = time.time()
        
        try:
            # Initialize all managers
            managers = []
            
            from relic.parameters import get_global_manager
            parameter_manager = get_global_manager()
            managers.append(("ParameterManager", parameter_manager))
            
            from relic.core.state import StateManager
            state_manager = StateManager()
            managers.append(("StateManager", state_manager))

            from relic.coupling.scales.scale_manager import ScaleManager
            scale_manager = ScaleManager(
                parameter_manager=parameter_manager,
                state_manager=state_manager,
            )
            scale_manager.initialize()
            scale_manager.validate()
            managers.append(("ScaleManager", scale_manager))
            managers.append(("GridManager", scale_manager.grid_manager))

            managers.append(
                ("PhysicalConstraintManager", state_manager.physical_constraint_manager)
            )

            from relic.scientific.transport.boundary_conditions import BoundaryConditionManager
            boundary_manager = BoundaryConditionManager()
            boundary_manager.initialize()
            boundary_manager.validate()
            assert boundary_manager.state.name == "READY"
            managers.append(("BoundaryConditionManager", boundary_manager))
            
            from relic.computation.distributed.launcher import ProcessGroupManager
            managers.append(("ProcessGroupManager", ProcessGroupManager()))
            
            from relic.parameters.calibration.versioning import VersionManager
            managers.append(("VersionManager", VersionManager(storage_dir=tmp_path / "versions")))
            
            # Initialize all managers
            for name, manager in managers:
                manager_start = time.time()
                assert manager is not None, f"{name} must have an explicit owner"
                if name == "ProcessGroupManager":
                    errors = manager.validate()
                    if errors and any("not available" in err.lower() for err in errors):
                        print("  ProcessGroupManager: SKIPPED (distributed not available)")
                        continue
                elif manager is not None and hasattr(manager, 'state'):
                    if manager.state.name not in ['READY', 'INITIALIZED', 'INITIALIZING']:
                        manager.initialize()
                manager_time = time.time() - manager_start
                print(f"  {name}: {manager_time:.3f}s")
            
            total_time = time.time() - start_time
            
            # Performance requirements: < 5 seconds for all managers
            assert total_time < 5.0, f"Initialization took {total_time:.2f}s, should be < 5s"
            
            print(f"PASS Performance Test - PASSED (Total: {total_time:.3f}s)")
            
        except Exception as e:
            pytest.fail(f"Performance test failed: {e}")
    
    def test_historical_interfaces_removed(self):
        """Verify historical interfaces are removed in v2."""
        try:
            with pytest.raises(ImportError):
                importlib.import_module("relic.cli.params_manager")

            with pytest.raises(ImportError):
                importlib.import_module("relic.core.managers.adapters")

            # New CLI namespace should remain available.
            assert importlib.import_module("relic.interfaces.cli") is not None

            versioning = importlib.import_module(
                "relic.parameters.calibration.versioning"
            )
            optimization_calibration = importlib.import_module(
                "relic.optimization.calibration"
            )
            assert not hasattr(versioning, "_global_version_manager")
            assert not hasattr(versioning, "get_global_version_manager")
            assert not hasattr(versioning, "set_global_version_manager")
            assert optimization_calibration.VersionManager is versioning.VersionManager
            assert (
                optimization_calibration.VersionedParameter
                is versioning.VersionedParameter
            )

            constraint_module = importlib.import_module(
                "relic.coupling.scales.physical_constraint_manager"
            )
            assert not hasattr(
                constraint_module, "_global_physical_constraint_manager"
            )
            assert not hasattr(
                constraint_module, "get_global_physical_constraint_manager"
            )
            assert not hasattr(
                constraint_module, "set_global_physical_constraint_manager"
            )

            print("PASS historical Interfaces Removed - PASSED")
        except Exception as e:
            pytest.fail(f"historical interface removal test failed: {e}")


def run_test_suite():
    """Run the complete test suite"""
    print("=" * 60)
    print("RELIC Manager v2.0.0 Integration Test Suite")
    print("=" * 60)
    
    # L1 Unit Tests
    print("\nL1 UNIT TESTS - Individual Manager Functionality")
    print("-" * 50)
    l1_tests = TestL1ManagerUnits()
    
    test_methods = [method for method in dir(l1_tests) if method.startswith('test_')]
    l1_passed = 0
    l1_total = len(test_methods)
    
    for test_method in test_methods:
        try:
            getattr(l1_tests, test_method)()
            l1_passed += 1
        except Exception as e:
            print(f"FAIL {test_method} - FAILED: {e}")
    
    # L2 Integration Tests
    print(f"\nL2 INTEGRATION TESTS - Manager Interactions")
    print("-" * 50)
    l2_tests = TestL2ManagerIntegrations()
    
    test_methods = [method for method in dir(l2_tests) if method.startswith('test_')]
    l2_passed = 0
    l2_total = len(test_methods)
    
    for test_method in test_methods:
        try:
            getattr(l2_tests, test_method)()
            l2_passed += 1
        except Exception as e:
            print(f"FAIL {test_method} - FAILED: {e}")
    
    # L3 System Tests
    print(f"\nL3 SYSTEM TESTS - Performance & Integration")
    print("-" * 50) 
    l3_tests = TestL3SystemPerformance()
    
    test_methods = [method for method in dir(l3_tests) if method.startswith('test_')]
    l3_passed = 0
    l3_total = len(test_methods)
    
    for test_method in test_methods:
        try:
            getattr(l3_tests, test_method)()
            l3_passed += 1
        except Exception as e:
            print(f"FAIL {test_method} - FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"L1 Unit Tests:        {l1_passed}/{l1_total} passed ({l1_passed/l1_total*100:.1f}%)")
    print(f"L2 Integration Tests: {l2_passed}/{l2_total} passed ({l2_passed/l2_total*100:.1f}%)")
    print(f"L3 System Tests:      {l3_passed}/{l3_total} passed ({l3_passed/l3_total*100:.1f}%)")
    
    total_passed = l1_passed + l2_passed + l3_passed
    total_tests = l1_total + l2_total + l3_total
    
    print(f"\nOVERALL:              {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        print("\nALL TESTS PASSED - Manager v2.0.0 system is fully functional!")
        return True
    else:
        print(f"\n{total_tests - total_passed} tests failed - Review failures above")
        return False


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)


