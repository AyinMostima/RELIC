"""
Transport and Coupling Test Runner

This script runs all the created transport and coupling tests and generates
a comprehensive test report. It validates L1, L2, and L3 test coverage and
provides detailed reporting on test results.

Usage:
    python run_transport_coupling_tests.py [--level L1|L2|L3|all] [--verbose]

Test Categories:
- L1: Unit tests for individual components
- L2: Integration tests for component interactions
- L3: System tests for end-to-end workflows
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Test runner for transport and coupling tests."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_root = self.project_root / "test"
        self.results = {}

    def discover_tests(self, level: str = "all") -> List[str]:
        """Discover test files based on level."""
        test_patterns = {
            "L1": [
                "test/transport/test_linear_advection_l1.py",
                "test/transport/test_transport_operators_l1.py",
                "test/coupling/test_temporal_aggregation_l1.py",
                "test/coupling/test_spatial_aggregation_l1.py"
            ],
            "L2": [
                "test/integration/test_transport_coupling_integration_l2.py"
            ],
            "L3": [
                "test/system/test_transport_coupling_system_l3.py"
            ]
        }

        if level == "all":
            return test_patterns["L1"] + test_patterns["L2"] + test_patterns["L3"]
        else:
            return test_patterns.get(level, [])

    def run_test_file(self, test_file: str, verbose: bool = False) -> Dict[str, Any]:
        """Run a single test file and return results."""
        test_path = self.project_root / test_file

        if not test_path.exists():
            return {
                "status": "skipped",
                "reason": f"Test file not found: {test_file}",
                "duration": 0.0
            }

        cmd = [sys.executable, "-m", "pytest", str(test_path)]

        if not verbose:
            cmd.extend(["-q", "--tb=no"])
        else:
            cmd.extend(["-v", "--tb=short"])

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            duration = time.time() - start_time

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "reason": "Test execution timed out after 5 minutes",
                "duration": 300.0
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e),
                "duration": 0.0
            }

    def run_tests(self, level: str = "all", verbose: bool = False) -> Dict[str, Any]:
        """Run all tests for the specified level."""
        test_files = self.discover_tests(level)

        if not test_files:
            return {
                "level": level,
                "total_files": 0,
                "results": {},
                "summary": {
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "errors": 0,
                    "timeouts": 0
                }
            }

        print(f"\n{'='*60}")
        print(f"Running {level} Transport and Coupling Tests")
        print(f"{'='*60}")
        print(f"Found {len(test_files)} test files")
        print()

        results = {}
        summary = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "timeouts": 0
        }

        total_duration = 0.0

        for i, test_file in enumerate(test_files, 1):
            print(f"[{i}/{len(test_files)}] Running {test_file}...", end=" ")

            result = self.run_test_file(test_file, verbose)
            results[test_file] = result

            status_symbol = {
                "passed": "[OK]",
                "failed": "[FAIL]",
                "skipped": "[SKIP]",
                "error": "[ERROR]",
                "timeout": "[TIMEOUT]"
            }.get(result["status"], "[?]")

            print(f" {status_symbol} ({result['duration']:.2f}s)")

            summary[result["status"]] += 1
            total_duration += result["duration"]

            if result["status"] == "failed" and verbose:
                print(f"  Error: {result.get('stderr', 'Unknown error')}")

        print(f"\n{'='*60}")
        print(f"Test Results Summary - {level.upper()}")
        print(f"{'='*60}")
        print(f"Total files: {len(test_files)}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Errors: {summary['errors']}")
        print(f"Timeouts: {summary['timeouts']}")
        print(f"Total duration: {total_duration:.2f}s")

        if summary['failed'] > 0:
            print(f"\nFailed tests:")
            for test_file, result in results.items():
                if result["status"] == "failed":
                    print(f"  - {test_file}")

        return {
            "level": level,
            "total_files": len(test_files),
            "results": results,
            "summary": summary,
            "total_duration": total_duration
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("# Transport and Coupling Test Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary table
        report.append("## Summary")
        report.append("")
        report.append("| Level | Total | Passed | Failed | Skipped | Duration |")
        report.append("|-------|-------|--------|--------|---------|----------|")

        if isinstance(results, list):
            # Multiple levels
            for level_result in results:
                level = level_result["level"]
                summary = level_result["summary"]
                duration = level_result["total_duration"]
                report.append(f"| {level.upper()} | {level_result['total_files']} | "
                           f"{summary['passed']} | {summary['failed']} | "
                           f"{summary['skipped']} | {duration:.2f}s |")
        else:
            # Single level
            level = results["level"]
            summary = results["summary"]
            duration = results["total_duration"]
            report.append(f"| {level.upper()} | {results['total_files']} | "
                       f"{summary['passed']} | {summary['failed']} | "
                       f"{summary['skipped']} | {duration:.2f}s |")

        report.append("")

        # Detailed results
        if isinstance(results, list):
            for level_result in results:
                report.append(f"## {level_result['level'].upper()} Level Details")
                report.append("")
                for test_file, result in level_result["results"].items():
                    status = result["status"].upper()
                    duration = result["duration"]
                    report.append(f"### {test_file}")
                    report.append(f"- **Status**: {status}")
                    report.append(f"- **Duration**: {duration:.2f}s")
                    if result["status"] in ["failed", "error", "timeout"]:
                        report.append(f"- **Reason**: {result.get('reason', 'Unknown')}")
                    report.append("")
        else:
            report.append("## Detailed Results")
            report.append("")
            for test_file, result in results["results"].items():
                status = result["status"].upper()
                duration = result["duration"]
                report.append(f"### {test_file}")
                report.append(f"- **Status**: {status}")
                report.append(f"- **Duration**: {duration:.2f}s")
                if result["status"] in ["failed", "error", "timeout"]:
                    report.append(f"- **Reason**: {result.get('reason', 'Unknown')}")
                report.append("")

        # Test coverage analysis
        report.append("## Test Coverage Analysis")
        report.append("")
        report.append("### L1 - Unit Tests")
        report.append("- LinearAdvection class functionality")
        report.append("- Transport operators (CFL, conservation, gradients)")
        report.append("- TemporalAggregator class functionality")
        report.append("- SpatialScaling class functionality")
        report.append("- Parameter system integration")
        report.append("- Error handling and edge cases")
        report.append("")

        report.append("### L2 - Integration Tests")
        report.append("- Transport with temporal scaling integration")
        report.append("- Transport with spatial scaling integration")
        report.append("- Multi-scale transport workflows")
        report.append("- Parameter system integration across modules")
        report.append("- Dependency resolution between components")
        report.append("")

        report.append("### L3 - System Tests")
        report.append("- Complete end-to-end workflows")
        report.append("- Real-world scenario simulation")
        report.append("- Performance benchmarking")
        report.append("- Conservation law verification")
        report.append("- External data integration")
        report.append("- Comparison with analytical solutions")
        report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        all_passed = True
        if isinstance(results, list):
            for level_result in results:
                if level_result["summary"]["failed"] > 0:
                    all_passed = False
                    break
        else:
            all_passed = results["summary"]["failed"] == 0

        if all_passed:
            report.append("[SUCCESS] **All tests passed!** The transport and coupling modules are functioning correctly.")
            report.append("")
            report.append("### Next Steps")
            report.append("- Consider adding performance benchmarks")
            report.append("- Add more complex real-world scenarios")
            report.append("- Implement additional analytical solution comparisons")
        else:
            report.append("[FAILURE] **Some tests failed!** Please review and fix the failing tests.")
            report.append("")
            report.append("### Required Actions")
            report.append("- Fix failing unit tests before proceeding")
            report.append("- Review integration test failures")
            report.append("- Address system test issues")
            report.append("- Ensure proper parameter system configuration")

        return "\n".join(report)


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Run transport and coupling tests")
    parser.add_argument(
        "--level",
        choices=["L1", "L2", "L3", "all"],
        default="all",
        help="Test level to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate test report file"
    )

    args = parser.parse_args()

    runner = TestRunner()

    if args.level == "all":
        # Run all levels
        levels = ["L1", "L2", "L3"]
        all_results = []

        for level in levels:
            result = runner.run_tests(level, args.verbose)
            all_results.append(result)

        final_results = all_results
    else:
        # Run specific level
        final_results = runner.run_tests(args.level, args.verbose)

    # Generate report
    report = runner.generate_report(final_results)

    if args.report:
        report_file = project_root / "test" / "transport_coupling_test_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_file}")
    else:
        print("\n" + "="*60)
        print("REPORT")
        print("="*60)
        print(report)

    # Return appropriate exit code
    if isinstance(final_results, list):
        total_failed = sum(result["summary"]["failed"] for result in final_results)
    else:
        total_failed = final_results["summary"]["failed"]

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

