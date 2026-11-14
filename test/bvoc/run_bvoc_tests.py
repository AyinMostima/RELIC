"""
BVOC Test Suite Runner

This script provides a comprehensive test runner for the BVOC module test suite,
allowing execution of different test levels (L1, L2, L3) individually or in 
combination, with detailed reporting and validation.

Usage
-----
python run_bvoc_tests.py [--level L1|L2|L3|all] [--verbose] [--report] [--quick]

Test Levels
-----------
- L1 (Unit): Individual component testing with mocked dependencies
- L2 (Integration): Component interaction and data flow testing
- L3 (System): End-to-end validation and performance benchmarks

Requirements
------------
- pytest >= 6.0
- torch >= 1.9.0
- numpy >= 1.20.0
- All BVOC module dependencies

Examples
--------
# Run all tests with detailed reporting
python run_bvoc_tests.py --level all --verbose --report

# Run only unit tests quickly
python run_bvoc_tests.py --level L1 --quick

# Run integration tests with performance monitoring
python run_bvoc_tests.py --level L2 --verbose

References
----------
- RELIC Testing Framework: Hierarchical validation approach
- Scientific Software Testing: Best practices for model validation
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add relic to path for testing
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns
    -------
    Dict[str, bool]
        Dictionary indicating availability of each dependency
    """
    dependencies = {}
    
    try:
        import pytest
        dependencies['pytest'] = True
        print(f"[OK] pytest {pytest.__version__}")
    except ImportError:
        dependencies['pytest'] = False
        print("[ERROR] pytest not available")
    
    try:
        import torch
        dependencies['torch'] = True
        print(f"[OK] torch {torch.__version__}")
    except ImportError:
        dependencies['torch'] = False
        print("[ERROR] torch not available")
    
    try:
        import numpy
        dependencies['numpy'] = True
        print(f"[OK] numpy {numpy.__version__}")
    except ImportError:
        dependencies['numpy'] = False
        print("[ERROR] numpy not available")
    
    # Check BVOC module availability
    try:
        from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
        dependencies['bvoc_module'] = MEGANModel is not None
        print("[OK] BVOC module available")
    except ImportError as e:
        dependencies['bvoc_module'] = False
        print(f"[ERROR] BVOC module not available: {e}")
    
    return dependencies


def run_test_level(level: str, verbose: bool = False, quick: bool = False) -> Dict[str, Any]:
    """
    Run specific test level.
    
    Parameters
    ----------
    level : str
        Test level (L1, L2, L3)
    verbose : bool
        Enable verbose output
    quick : bool
        Run quick subset of tests
        
    Returns
    -------
    Dict[str, Any]
        Test results and statistics
    """
    test_files = {
        'L1': 'test_bvoc_unit.py',
        'L2': 'test_bvoc_integration.py', 
        'L3': 'test_bvoc_system.py'
    }
    
    if level not in test_files:
        raise ValueError(f"Unknown test level: {level}")
    
    test_file = Path(__file__).parent / test_files[level]
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest', str(test_file)]
    
    if verbose:
        cmd.extend(['-v', '-s'])
    else:
        cmd.append('-q')
    
    # Add quick mode markers
    if quick:
        cmd.extend(['-m', 'not slow'])
    
    # Add output formats
    cmd.extend([
        '--tb=short',
        '--durations=5',
        f'--junit-xml=test_results_{level.lower()}.xml'
    ])
    
    print(f"\nRunning {level} tests: {test_files[level]}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run tests
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        end_time = time.time()
        
        return {
            'level': level,
            'file': test_files[level],
            'duration': end_time - start_time,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
    except Exception as e:
        end_time = time.time()
        return {
            'level': level,
            'file': test_files[level],
            'duration': end_time - start_time,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False,
            'error': str(e)
        }


def parse_test_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse test output to extract statistics.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Raw test results
        
    Returns
    -------
    Dict[str, Any]
        Parsed statistics
    """
    stats = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': 0,
        'warnings': 0
    }
    
    stdout = results.get('stdout', '')
    
    # Parse pytest output
    lines = stdout.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Look for test summary line
        if '======' in line and 'test session starts' in line:
            continue
        elif '======' in line and ('failed' in line or 'passed' in line or 'error' in line):
            # Parse summary line like "1 failed, 2 passed in 0.5s"
            parts = line.split()
            for i, part in enumerate(parts):
                if part.isdigit():
                    count = int(part)
                    if i + 1 < len(parts):
                        next_part = parts[i + 1].lower()
                        if 'passed' in next_part:
                            stats['passed'] = count
                        elif 'failed' in next_part:
                            stats['failed'] = count
                        elif 'skipped' in next_part:
                            stats['skipped'] = count
                        elif 'error' in next_part:
                            stats['errors'] = count
        
        # Count warnings
        if 'warning' in line.lower():
            stats['warnings'] += 1
    
    stats['total_tests'] = stats['passed'] + stats['failed'] + stats['skipped'] + stats['errors']
    
    return stats


def generate_test_report(all_results: List[Dict[str, Any]], output_file: Optional[str] = None) -> str:
    """
    Generate comprehensive test report.
    
    Parameters
    ----------
    all_results : List[Dict[str, Any]]
        Results from all test levels
    output_file : str, optional
        Output file path for report
        
    Returns
    -------
    str
        Formatted test report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BVOC MODULE TEST SUITE REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    total_duration = sum(r['duration'] for r in all_results)
    total_success = all(r['success'] for r in all_results)
    
    # Executive summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Overall Result: {'PASS' if total_success else 'FAIL'}")
    report_lines.append(f"Total Duration: {total_duration:.2f} seconds")
    report_lines.append(f"Test Levels Run: {len(all_results)}")
    report_lines.append("")
    
    # Detailed results by level
    for result in all_results:
        level = result['level']
        success = result['success']
        duration = result['duration']
        
        report_lines.append(f"{level} TESTS ({result['file']})")
        report_lines.append("-" * 40)
        report_lines.append(f"Status: {'PASS' if success else 'FAIL'}")
        report_lines.append(f"Duration: {duration:.2f} seconds")
        report_lines.append(f"Return Code: {result['return_code']}")
        
        # Parse and include statistics
        stats = parse_test_results(result)
        if stats['total_tests'] > 0:
            report_lines.append(f"Total Tests: {stats['total_tests']}")
            report_lines.append(f"Passed: {stats['passed']}")
            report_lines.append(f"Failed: {stats['failed']}")
            report_lines.append(f"Skipped: {stats['skipped']}")
            report_lines.append(f"Errors: {stats['errors']}")
            report_lines.append(f"Warnings: {stats['warnings']}")
        
        # Include stderr if there were issues
        if not success and result.get('stderr'):
            report_lines.append("\nErrors/Warnings:")
            report_lines.append(result['stderr'][:500])  # Truncate for readability
            if len(result['stderr']) > 500:
                report_lines.append("... (truncated)")
        
        report_lines.append("")
    
    # Performance analysis
    if len(all_results) > 1:
        report_lines.append("PERFORMANCE ANALYSIS")
        report_lines.append("-" * 40)
        for result in all_results:
            tests_per_second = "N/A"
            stats = parse_test_results(result)
            if stats['total_tests'] > 0 and result['duration'] > 0:
                tests_per_second = f"{stats['total_tests'] / result['duration']:.1f}"
            
            report_lines.append(f"{result['level']}: {result['duration']:.2f}s ({tests_per_second} tests/s)")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 40)
    
    if total_success:
        report_lines.append("[OK] All tests passed successfully")
        report_lines.append("[OK] BVOC module is ready for production use")
    else:
        report_lines.append("[WARN] Some tests failed - review issues before deployment")
        
        failed_levels = [r['level'] for r in all_results if not r['success']]
        if failed_levels:
            report_lines.append(f"[WARN] Failed levels: {', '.join(failed_levels)}")
    
    # Fast tests should complete quickly
    fast_threshold = 30  # seconds
    slow_levels = [r['level'] for r in all_results if r['duration'] > fast_threshold]
    if slow_levels:
        report_lines.append(f"[WARN] Slow test levels (>{fast_threshold}s): {', '.join(slow_levels)}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = '\n'.join(report_lines)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    
    return report_text


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="BVOC Module Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bvoc_tests.py --level all --verbose --report
  python run_bvoc_tests.py --level L1 --quick
  python run_bvoc_tests.py --level L2 --verbose
        """
    )
    
    parser.add_argument(
        '--level', 
        choices=['L1', 'L2', 'L3', 'all'],
        default='all',
        help='Test level to run (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick subset of tests'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate detailed test report'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file for test report'
    )
    
    args = parser.parse_args()
    
    print("BVOC Module Test Suite Runner")
    print("=" * 50)
    
    # Check dependencies
    print("\nChecking dependencies...")
    deps = check_dependencies()
    
    missing_deps = [name for name, available in deps.items() if not available]
    if missing_deps:
        print(f"\n[ERROR] Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies before running tests.")
        return 1
    
    print("\n[OK] All dependencies available")
    
    # Determine which levels to run
    if args.level == 'all':
        levels_to_run = ['L1', 'L2', 'L3']
    else:
        levels_to_run = [args.level]
    
    # Run tests
    print(f"\nRunning test levels: {', '.join(levels_to_run)}")
    if args.quick:
        print("Mode: Quick (skipping slow tests)")
    
    all_results = []
    
    for level in levels_to_run:
        try:
            result = run_test_level(level, verbose=args.verbose, quick=args.quick)
            all_results.append(result)
            
            # Print immediate results
            if result['success']:
                print(f"[OK] {level} tests PASSED ({result['duration']:.2f}s)")
            else:
                print(f"[FAIL] {level} tests FAILED ({result['duration']:.2f}s)")
                if args.verbose:
                    print(f"Error: {result.get('stderr', 'Unknown error')}")
                    
        except Exception as e:
            print(f"[ERROR] {level} tests ERROR: {e}")
            all_results.append({
                'level': level,
                'success': False,
                'duration': 0.0,
                'error': str(e)
            })
    
    # Generate summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total_success = all(r['success'] for r in all_results)
    total_duration = sum(r['duration'] for r in all_results)
    
    if total_success:
        print("[OK] ALL TESTS PASSED")
    else:
        failed_levels = [r['level'] for r in all_results if not r['success']]
        print(f"[FAIL] TESTS FAILED: {', '.join(failed_levels)}")
    
    print(f"Total Duration: {total_duration:.2f} seconds")
    
    # Generate detailed report if requested
    if args.report:
        output_file = args.output or f"bvoc_test_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        report = generate_test_report(all_results, output_file)
        
        if not args.verbose:
            # Print just the summary if not verbose
            print("\nReport Summary:")
            print(report.split("RECOMMENDATIONS")[0].split("PERFORMANCE ANALYSIS")[0])
    
    # Return appropriate exit code
    return 0 if total_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

