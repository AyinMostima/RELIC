"""
Complete Manager v2 Test Suite Runner

Runs all comprehensive Manager v2 tests:
- Original integration tests
- L1 comprehensive unit tests
- L2 comprehensive integration tests  
- L3 comprehensive system tests

Provides complete testing coverage and verification.
"""

import sys
import subprocess
import time
from pathlib import Path

# Add relic to path
relic_root = Path(__file__).parent.parent
sys.path.insert(0, str(relic_root))


def run_test_suite(test_file, description):
    """Run a test suite and return results"""
    print(f"\n{'='*70}")
    print(f"Running {description}")
    print(f"File: {test_file}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Run the test file
        result = subprocess.run([
            sys.executable, test_file
        ], cwd=relic_root, capture_output=True, text=True, timeout=300)
        
        execution_time = time.time() - start_time
        
        # Parse results
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
        
        print(output)
        
        if error and "warning" not in error.lower():
            print(f"STDERR:\n{error}")
        
        print(f"\nExecution Time: {execution_time:.2f}s")
        print(f"Result: {'PASS' if success else 'FAIL'}")
        
        return success, execution_time, output
        
    except subprocess.TimeoutExpired:
        print("ERROR: Test suite timed out (5 minutes)")
        return False, 300, ""
    except Exception as e:
        print(f"ERROR: Failed to run test suite: {e}")
        return False, 0, ""


def main():
    """Run all Manager v2 test suites"""
    print("Manager v2 Complete Test Suite Runner")
    print("=" * 70)
    
    # Define test suites
    test_suites = [
        ("test/test_manager_v2_integration.py", "Original Manager v2 Integration Tests"),
        ("test/test_manager_v2_comprehensive_l1.py", "L1 Comprehensive Unit Tests"),
        ("test/test_manager_v2_comprehensive_l2.py", "L2 Comprehensive Integration Tests"),
        ("test/test_manager_v2_comprehensive_l3.py", "L3 Comprehensive System Tests"),
    ]
    
    results = []
    total_time = 0
    
    for test_file, description in test_suites:
        success, exec_time, output = run_test_suite(test_file, description)
        results.append((description, success, exec_time, output))
        total_time += exec_time
        
        if not success:
            print(f"\n[ERROR]  {description} FAILED")
        else:
            print(f"\n[OK] {description} PASSED")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUITE SUMMARY")
    print("=" * 70)
    
    passed_suites = 0
    total_suites = len(test_suites)
    
    for description, success, exec_time, output in results:
        status = "PASS" if success else "FAIL"
        print(f"{description:50} {status:4} ({exec_time:.1f}s)")
        if success:
            passed_suites += 1
    
    print(f"\nSuite Results: {passed_suites}/{total_suites} passed ({passed_suites/total_suites*100:.1f}%)")
    print(f"Total Execution Time: {total_time:.1f}s")
    
    # Extract test counts from outputs
    total_tests_run = 0
    total_tests_passed = 0
    
    for description, success, exec_time, output in results:
        if success:
            # Try to extract test counts from output
            lines = output.split('\n')
            for line in lines:
                if "passed" in line and "/" in line and "%" in line:
                    try:
                        # Extract numbers like "15/15 passed (100.0%)"
                        parts = line.split()
                        for part in parts:
                            if "/" in part:
                                passed, total = map(int, part.split('/'))
                                total_tests_passed += passed
                                total_tests_run += total
                                break
                    except:
                        pass
    
    if total_tests_run > 0:
        print(f"Individual Tests: {total_tests_passed}/{total_tests_run} passed ({total_tests_passed/total_tests_run*100:.1f}%)")
    
    # Final verdict
    if passed_suites == total_suites:
        print("\n[OK] ALL TEST SUITES PASSED! [OK]")
        print("Manager v2 system is fully functional and production ready!")
        return True
    else:
        failed_count = total_suites - passed_suites
        print(f"\n[ERROR]{failed_count} test suite(s) failed")
        print("System requires fixes before production deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

