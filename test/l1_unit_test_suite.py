"""
L1 Unit Test Suite for Manager V2 Migration

Comprehensive unit tests for all migrated managers to ensure
BaseManager integration and individual functionality.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add relic to path
sys.path.insert(0, os.path.abspath('.'))

# Import test modules - add them to path individually
test_dir = os.path.join(os.path.dirname(__file__), 'unit', 'managers')
sys.path.insert(0, test_dir)

import test_state_manager
import test_scale_manager  
import test_grid_manager
import test_physical_constraint_manager

TestStateManagerV2Migration = test_state_manager.TestStateManagerV2Migration
TestScaleManagerV2Migration = test_scale_manager.TestScaleManagerV2Migration
TestGridManagerV2Migration = test_grid_manager.TestGridManagerV2Migration
TestPhysicalConstraintManagerV2Migration = test_physical_constraint_manager.TestPhysicalConstraintManagerV2Migration


class L1UnitTestSuite:
    """L1 Unit Test Suite for all migrated managers."""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
    def run_manager_tests(self, test_class, manager_name):
        """Delegate a complete manager test file to the canonical pytest runner."""
        print(f"\n{'='*60}")
        print(f"Testing {manager_name}")
        print(f"{'='*60}")

        module = sys.modules[test_class.__module__]
        test_file = Path(module.__file__).resolve()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-q"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
        if result.returncode == 0:
            self.test_results['passed'] += 1
            print(f"\n{manager_name} suite: [OK]")
            return 1, 0

        self.test_results['failed'] += 1
        self.test_results['errors'].append({
            'manager': manager_name,
            'test': test_file.name,
            'error': f"pytest return code {result.returncode}",
        })
        print(f"\n{manager_name} suite: [FAIL] return code {result.returncode}")
        return 0, 1
    
    def run_all_tests(self):
        """Run all L1 unit tests."""
        print("\n" + "="*80)
        print("L1 UNIT TEST SUITE - Manager V2 Migration")
        print("="*80)
        
        # Define test suites
        test_suites = [
            (TestStateManagerV2Migration, "StateManager"),
            (TestScaleManagerV2Migration, "ScaleManager"),
            (TestGridManagerV2Migration, "GridManager"),
            (TestPhysicalConstraintManagerV2Migration, "PhysicalConstraintManager"),
        ]
        
        # Run each test suite
        for test_class, manager_name in test_suites:
            try:
                self.run_manager_tests(test_class, manager_name)
            except Exception as e:
                print(f"Failed to run tests for {manager_name}: {e}")
                self.test_results['failed'] += 1
        
        # Print summary
        self.print_summary()
        
        return self.test_results['failed'] == 0
    
    def print_summary(self):
        """Print test suite summary."""
        print("\n" + "="*80)
        print("L1 TEST SUITE SUMMARY")
        print("="*80)
        
        total = self.test_results['passed'] + self.test_results['failed'] + self.test_results['skipped']
        print(f"Total suites: {total}")
        print(f"Passed: {self.test_results['passed']} ({self.test_results['passed']/total*100:.1f}%)")
        print(f"Failed: {self.test_results['failed']} ({self.test_results['failed']/total*100:.1f}%)")
        print(f"Skipped: {self.test_results['skipped']} ({self.test_results['skipped']/total*100:.1f}%)")
        
        if self.test_results['errors']:
            print("\nFailed Suites:")
            for error in self.test_results['errors']:
                print(f"  - {error['manager']}.{error['test']}: {error['error'][:50]}...")
        
        print("\n" + "="*80)
        if self.test_results['failed'] == 0:
            print("L1 TEST SUITE: [OK] ALL TESTS PASSED")
        else:
            print(f"L1 TEST SUITE: [FAIL] {self.test_results['failed']} TESTS FAILED")
        print("="*80)
    
    def verify_manager_integration(self):
        """Verify that all managers integrate properly."""
        print("\n" + "="*60)
        print("Manager Integration Verification")
        print("="*60)

        from relic.core.managers.base import BaseManager, ManagerState
        from relic.core.state import StateManager
        from relic.coupling.scales.spatial.grid_manager import GridManager
        from relic.coupling.scales.physical_constraint_manager import (
            PhysicalConstraintManager,
        )
        from relic.parameters import get_global_manager

        all_ok = True

        try:
            grid_manager = GridManager(parameter_manager=get_global_manager())
            assert isinstance(grid_manager, BaseManager)
            grid_manager.initialize()
            grid_manager.validate()
            assert grid_manager.state == ManagerState.READY
            print("  [OK] GridManager explicit-owner integration verified")
        except Exception as e:
            print(f"  [FAIL] GridManager integration: {e}")
            all_ok = False

        try:
            manager = PhysicalConstraintManager()
            assert isinstance(manager, BaseManager)
            manager.initialize()
            manager.validate()
            assert manager.state == ManagerState.READY

            state_manager = StateManager(
                schema={},
                autoload_contract_registry=False,
            )
            owned_manager = state_manager.physical_constraint_manager
            assert isinstance(owned_manager, PhysicalConstraintManager)
            assert owned_manager.state == ManagerState.READY
            assert owned_manager is state_manager.physical_constraint_manager

            print("  [OK] PhysicalConstraintManager explicit-owner integration verified")
        except Exception as e:
            print(f"  [FAIL] PhysicalConstraintManager integration: {e}")
            all_ok = False
        
        if all_ok:
            print("\n[OK] All manager integrations verified successfully")
        else:
            print("\n[FAIL] Some manager integrations failed")
        
        return all_ok


def run_l1_test_suite():
    """Main entry point for L1 test suite."""
    suite = L1UnitTestSuite()
    
    # Run all unit tests
    tests_passed = suite.run_all_tests()
    
    # Verify manager integration
    integration_ok = suite.verify_manager_integration()
    
    # Final result
    success = tests_passed and integration_ok
    
    print("\n" + "="*80)
    if success:
        print("L1 UNIT TEST SUITE: [OK] COMPLETE SUCCESS")
    else:
        print("L1 UNIT TEST SUITE: [FAIL] SOME TESTS FAILED")
    print("="*80 + "\n")
    
    return success


if __name__ == "__main__":
    success = run_l1_test_suite()
    sys.exit(0 if success else 1)

