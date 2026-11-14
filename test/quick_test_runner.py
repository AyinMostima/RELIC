#!/usr/bin/env python3
"""
Quick Test Runner for Anthroposphere Module

A simplified test runner for quick validation and coverage checking.
This script is designed for rapid feedback during development.

Usage:
    python quick_test_runner.py [test_type]
    
Test Types:
    L1      - Run unit tests only
    L2      - Run integration tests only  
    L3      - Run system tests only
    all     - Run all tests (default)
    coverage- Run tests with coverage
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*50)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    execution_time = time.time() - start_time
    
    print(f"Execution time: {execution_time:.2f}s")
    print(f"Exit code: {result.returncode}")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    success = result.returncode == 0
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    
    return success


def run_l1_tests():
    """Run L1 unit tests"""
    test_dirs = [
        "test/anthroposphere/management/",
        "test/anthroposphere/emissions/",
        "test/anthroposphere/land_use/",
        "test/anthroposphere/test_inventory_l1_improved.py"
    ]
    
    test_files = []
    for test_dir in test_dirs:
        if os.path.isfile(test_dir):
            test_files.append(test_dir)
        elif os.path.isdir(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.startswith("test_") and file.endswith(".py"):
                        test_files.append(os.path.join(root, file))
    
    if not test_files:
        print("No L1 test files found!")
        return False
    
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"] + test_files
    return run_command(cmd, "L1 Unit Tests")


def run_l2_tests():
    """Run L2 integration tests"""
    test_file = "test/anthroposphere/test_integration_l2.py"
    
    if not os.path.exists(test_file):
        print(f"L2 test file not found: {test_file}")
        return False
    
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", test_file]
    return run_command(cmd, "L2 Integration Tests")


def run_l3_tests():
    """Run L3 system tests"""
    test_file = "test/anthroposphere/test_system_l3.py"
    
    if not os.path.exists(test_file):
        print(f"L3 test file not found: {test_file}")
        return False
    
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", test_file]
    return run_command(cmd, "L3 System Tests")


def run_all_tests():
    """Run all tests"""
    success = True
    
    print("\n" + "="*70)
    print("RUNNING ALL ANTHROPOSPHERE TESTS")
    print("="*70)
    
    # L1 Tests
    if not run_l1_tests():
        success = False
    
    # L2 Tests
    if not run_l2_tests():
        success = False
    
    # L3 Tests
    if not run_l3_tests():
        success = False
    
    return success


def run_coverage_tests():
    """Run tests with coverage"""
    try:
        import coverage
    except ImportError:
        print("Coverage package not installed. Install with: pip install coverage")
        return False
    
    print("\n" + "="*70)
    print("RUNNING TESTS WITH COVERAGE")
    print("="*70)
    
    # Initialize coverage
    cov = coverage.Coverage(
        source=["relic.scientific.anthroposphere"],
        omit=["*/tests/*", "*/test_*", "*/__pycache__/*"]
    )
    cov.start()
    
    # Run all tests
    success = run_all_tests()
    
    # Stop coverage and generate report
    cov.stop()
    cov.save()
    
    print("\n" + "="*50)
    print("COVERAGE REPORT")
    print('='*50)
    
    # Console report
    cov.report()
    
    # HTML report
    try:
        cov.html_report(directory="coverage_html")
        print(f"\nDetailed HTML coverage report generated in: coverage_html/")
    except Exception as e:
        print(f"Error generating HTML report: {e}")
    
    # XML report for CI
    try:
        cov.xml_report(outfile="coverage.xml")
        print("XML coverage report generated: coverage.xml")
    except Exception as e:
        print(f"Error generating XML report: {e}")
    
    # Check coverage threshold
    total_coverage = cov.report()
    print(f"\nTotal coverage: {total_coverage:.1f}%")
    
    if total_coverage >= 80:
        print("[OK] Coverage meets requirement (>= 80%)")
    else:
        print(f"[FAIL] Coverage below requirement (>= 80%, got {total_coverage:.1f}%)")
        success = False
    
    return success


def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = [
        "torch",
        "numpy", 
        "pytest"
    ]
    
    optional_packages = [
        "coverage",
        "psutil",
        "matplotlib",
        "pandas"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package} (required)")
            missing_required.append(package)

    for package in optional_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[OPTIONAL] {package} (optional)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\nMissing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\nMissing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install " + " ".join(missing_optional))
    
    print("All required dependencies available!")
    return True


def cleanup_test_files():
    """Clean up test artifacts"""
    print("\nCleaning up test artifacts...")
    
    artifacts = [
        ".coverage",
        "coverage.xml",
        "coverage_html/",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/"
    ]
    
    import glob
    import shutil
    
    cleaned = 0
    for pattern in artifacts:
        if pattern.endswith("/"):
            # Directory
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    cleaned += 1
                    print(f"Removed directory: {path}")
        else:
            # File pattern
            for path in glob.glob(pattern):
                if os.path.isfile(path):
                    os.remove(path)
                    cleaned += 1
                    print(f"Removed file: {path}")
    
    print(f"Cleaned {cleaned} artifacts")


def print_usage():
    """Print usage information"""
    print("""
Quick Test Runner for Anthroposphere Module

Usage: python quick_test_runner.py [test_type]

Test Types:
    L1          - Run unit tests only
    L2          - Run integration tests only  
    L3          - Run system tests only
    all         - Run all tests (default)
    coverage    - Run tests with coverage analysis
    check       - Check dependencies only
    clean       - Clean test artifacts

Examples:
    python quick_test_runner.py           # Run all tests
    python quick_test_runner.py L1       # Run unit tests only
    python quick_test_runner.py coverage # Run with coverage
    python quick_test_runner.py clean    # Clean artifacts
    """)


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = "all"
    
    print("RELIC Anthroposphere Module Quick Test Runner")
    print("=" * 55)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nCannot run tests due to missing dependencies.")
        sys.exit(1)
    
    # Change to project root directory
    os.chdir(project_root)
    
    success = True
    
    if test_type == "l1":
        success = run_l1_tests()
    elif test_type == "l2":
        success = run_l2_tests()
    elif test_type == "l3":
        success = run_l3_tests()
    elif test_type == "all":
        success = run_all_tests()
    elif test_type == "coverage":
        success = run_coverage_tests()
    elif test_type == "check":
        # Dependencies already checked above
        success = True
    elif test_type == "clean":
        cleanup_test_files()
        success = True
    else:
        print(f"Unknown test type: {test_type}")
        print_usage()
        sys.exit(1)
    
    # Print final result
    print("\n" + "="*55)
    if success:
        print("RESULT: ALL TESTS PASSED [OK]")
        sys.exit(0)
    else:
        print("RESULT: SOME TESTS FAILED [FAIL]")
        sys.exit(1)


if __name__ == "__main__":
    main()

