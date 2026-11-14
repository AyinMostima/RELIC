#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 System Test Runner

This script provides a convenient way to run L3 system tests with proper
environment setup and result reporting.

Usage:
    python run_l3_tests.py [test_name] [--verbose] [--save-plots]

Examples:
    python run_l3_tests.py                                    # Run all L3 tests
    python run_l3_tests.py test_complete_daily_cycle_simulation # Run specific test
    python run_l3_tests.py --verbose                          # Run with detailed output
    python run_l3_tests.py --save-plots                       # Save diagnostic plots

Authors: Yuchong Long
License: MIT
"""

import sys
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'l3_test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def check_environment():
    """Check if the environment is properly set up for L3 testing."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python version {sys.version} < 3.8")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.device_count()} devices")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available - using CPU")
    except ImportError:
        issues.append("PyTorch not available")
    
    # Check key dependencies
    required_packages = [
        'numpy', 'matplotlib', 'psutil', 'pytest'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing package: {package}")
    
    # Check relic package
    try:
        import relic
        print(f"RELIC package location: {relic.__file__}")
    except ImportError:
        issues.append("RELIC package not available")
    
    if issues:
        print("Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Environment check passed")
        return True

def create_test_report(test_results, output_dir):
    """Create a comprehensive test report."""
    report = {
        'test_run_info': {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'total_tests': len(test_results),
        },
        'test_results': test_results,
        'summary': {
            'passed': sum(1 for r in test_results if r.get('passed', False)),
            'failed': sum(1 for r in test_results if not r.get('passed', True)),
            'total_tests': len(test_results),
            'total_time': sum(r.get('duration', 0) for r in test_results),
        }
    }
    
    # Save JSON report
    report_file = output_dir / f"l3_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create summary report
    summary_file = output_dir / "l3_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("RELIC L3 System Test Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Test run: {report['test_run_info']['timestamp']}\n")
        f.write(f"Environment: Python {report['test_run_info']['python_version']}\n")
        f.write(f"PyTorch: {report['test_run_info']['pytorch_version']}\n")
        f.write(f"CUDA: {'Available' if report['test_run_info']['cuda_available'] else 'Not available'}\n\n")
        
        f.write(f"Results: {report['summary']['passed']}/{report['summary']['total_tests']} passed\n")
        f.write(f"Total time: {report['summary']['total_time']:.1f} seconds\n\n")
        
        f.write("Individual Test Results:\n")
        f.write("-" * 25 + "\n")
        for result in test_results:
            status = "PASS" if result.get('passed', False) else "FAIL"
            f.write(f"{result.get('name', 'Unknown'):.<40} {status:>6} ({result.get('duration', 0):.1f}s)\n")
        
        if any(not r.get('passed', True) for r in test_results):
            f.write("\nFailure Details:\n")
            f.write("-" * 16 + "\n")
            for result in test_results:
                if not result.get('passed', True):
                    f.write(f"\n{result.get('name', 'Unknown')}:\n")
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
    
    print(f"Test report saved to: {report_file}")
    print(f"Summary saved to: {summary_file}")
    
    return report

def run_specific_test(test_name, output_dir, verbose=False, save_plots=False):
    """Run a specific L3 test."""
    test_file = Path(__file__).parent / "test_end_to_end.py"
    
    # Prepare pytest arguments
    pytest_args = [
        str(test_file) + f"::TestL3EndToEndSystem::{test_name}",
        "-v" if verbose else "-q",
        "-s",  # Don't capture output
        "--tb=short",
        f"--basetemp={output_dir / 'pytest_tmp'}",
    ]
    
    if save_plots:
        pytest_args.append("--capture=no")
    
    print(f"Running test: {test_name}")
    start_time = time.time()
    
    # Run the test
    exit_code = pytest.main(pytest_args)
    
    duration = time.time() - start_time
    
    result = {
        'name': test_name,
        'passed': exit_code == 0,
        'duration': duration,
        'exit_code': exit_code,
    }
    
    if exit_code != 0:
        result['error'] = f"Test failed with exit code {exit_code}"
    
    return result

def run_all_tests(output_dir, verbose=False, save_plots=False):
    """Run all L3 system tests."""
    test_methods = [
        'test_complete_daily_cycle_simulation',
        'test_annual_ecosystem_simulation',
        'test_multi_site_comparison',
        'test_extreme_conditions_stress_testing',
        'test_performance_benchmarking',
        'test_system_robustness_and_recovery',
        'test_scientific_accuracy_validation',
    ]
    
    results = []
    total_start_time = time.time()
    
    print("Running RELIC L3 System Tests")
    print("=" * 40)
    
    for i, test_name in enumerate(test_methods, 1):
        print(f"\n[{i}/{len(test_methods)}] Running {test_name}...")
        
        result = run_specific_test(test_name, output_dir, verbose, save_plots)
        results.append(result)
        
        status = "PASSED" if result['passed'] else "FAILED"
        print(f"  {status} ({result['duration']:.1f}s)")
        
        if not result['passed']:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    total_duration = time.time() - total_start_time
    
    print(f"\nTest suite completed in {total_duration:.1f}s")
    
    # Create comprehensive report
    report = create_test_report(results, output_dir)
    
    # Print summary
    passed = report['summary']['passed']
    total = report['summary']['total_tests']
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("[OK] All L3 system tests PASSED!")
        return 0
    else:
        print(f"[ERROR] {total - passed} test(s) FAILED")
        return 1

def create_diagnostic_plots(output_dir):
    """Create diagnostic plots from test results."""
    try:
        # Look for result files
        result_files = list(output_dir.glob("*_results.json"))
        
        if not result_files:
            print("No result files found for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("RELIC L3 System Test Results", fontsize=16)
        
        # Plot 1: Daily cycle (if available)
        daily_cycle_file = output_dir / "daily_cycle_results.json"
        if daily_cycle_file.exists():
            with open(daily_cycle_file) as f:
                data = json.load(f)
            
            ax = axes[0, 0]
            hours = data.get('time', [])
            gpp = data.get('gross_primary_productivity', [])
            par = data.get('par', [])
            
            ax2 = ax.twinx()
            ax.plot(hours, gpp, 'g-', label='GPP', linewidth=2)
            ax2.plot(hours, par, 'y--', label='PAR', alpha=0.7)
            
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('GPP (kg C m-2 s-1)', color='g')
            ax2.set_ylabel('PAR (umol m-2 s-1)', color='orange')
            ax.set_title('Daily Cycle Patterns')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Annual patterns (if available)
        annual_file = output_dir / "annual_simulation_results.json"
        if annual_file.exists():
            with open(annual_file) as f:
                data = json.load(f)
            
            ax = axes[0, 1]
            
            # Plot for first ecosystem type
            if data:
                first_eco = list(data.keys())[0]
                eco_data = data[first_eco]
                days = eco_data.get('day', [])
                gpp = eco_data.get('gross_primary_productivity', [])
                temp = eco_data.get('air_temperature', [])
                
                ax2 = ax.twinx()
                ax.plot(days, gpp, 'g-', label='GPP', linewidth=2)
                ax2.plot(days, [t - 273.15 for t in temp], 'r--', label='Temperature', alpha=0.7)
                
                ax.set_xlabel('Day of Year')
                ax.set_ylabel('GPP (kg C m-2 s-1)', color='g')
                ax2.set_ylabel('Temperature (degC)', color='r')
                ax.set_title(f'Annual Patterns ({first_eco})')
                ax.grid(True, alpha=0.3)
        
        # Plot 3: Multi-site comparison (if available)
        multisite_file = output_dir / "multi_site_results.json"
        if multisite_file.exists():
            with open(multisite_file) as f:
                data = json.load(f)
            
            ax = axes[1, 0]
            
            site_names = []
            mean_gpp = []
            
            for site, site_data in data.items():
                site_names.append(site_data.get('ecosystem_type', site))
                gpp_values = site_data.get('gross_primary_productivity', [])
                mean_gpp.append(np.mean(gpp_values) if gpp_values else 0)
            
            bars = ax.bar(range(len(site_names)), mean_gpp, color=['green', 'brown', 'gold', 'darkgreen', 'lightgreen'][:len(site_names)])
            ax.set_xlabel('Site/Ecosystem Type')
            ax.set_ylabel('Mean GPP (kg C m-2 s-1)')
            ax.set_title('Multi-Site GPP Comparison')
            ax.set_xticks(range(len(site_names)))
            ax.set_xticklabels(site_names, rotation=45, ha='right')
            ax.grid(True, axis='y', alpha=0.3)
        
        # Plot 4: Performance metrics (if available)
        perf_file = output_dir / "performance_benchmark_results.json"
        if perf_file.exists():
            with open(perf_file) as f:
                data = json.load(f)
            
            ax = axes[1, 1]
            
            if 'single_step' in data:
                batch_sizes = data['single_step']['batch_sizes']
                times = data['single_step']['times']
                
                ax.loglog(batch_sizes, times, 'bo-', linewidth=2, markersize=8)
                ax.set_xlabel('Batch Size')
                ax.set_ylabel('Execution Time (s)')
                ax.set_title('Performance Scaling')
                ax.grid(True, alpha=0.3)
                
                # Add ideal scaling line for reference
                ideal_times = [times[0] * (b / batch_sizes[0]) for b in batch_sizes]
                ax.loglog(batch_sizes, ideal_times, 'r--', alpha=0.5, label='Linear scaling')
                ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / "l3_system_test_diagnostics.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Diagnostic plots saved to: {plot_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error creating diagnostic plots: {str(e)}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run RELIC L3 System Tests")
    parser.add_argument('test_name', nargs='?', help='Specific test to run')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save-plots', action='store_true', help='Save diagnostic plots')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please install missing dependencies.")
        return 1
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("l3_test_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test results will be saved to: {output_dir}")
    
    try:
        if args.test_name:
            # Run specific test
            result = run_specific_test(args.test_name, output_dir, args.verbose, args.save_plots)
            
            # Create simple report
            create_test_report([result], output_dir)
            
            if args.save_plots:
                create_diagnostic_plots(output_dir)
            
            return 0 if result['passed'] else 1
        else:
            # Run all tests
            exit_code = run_all_tests(output_dir, args.verbose, args.save_plots)
            
            if args.save_plots:
                create_diagnostic_plots(output_dir)
            
            return exit_code
            
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 130
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

