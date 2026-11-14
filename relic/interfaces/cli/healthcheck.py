"""
Health Check CLI Module

Comprehensive health diagnostics for all RELIC subsystems including
computation layer, data I/O, parameter management, optimization algorithms,
and stochastic processes.

This module provides unified health checking with detailed diagnostics,
automated problem detection, and actionable recommendations for users.

Features:
    - System-wide health overview
    - Individual subsystem diagnostics
    - Performance benchmarking
    - Dependency validation
    - Configuration verification
    - Automated problem resolution suggestions

Commands:
    all         Complete system health check
    compute     Computational optimization health
    data        Data I/O and streaming health  
    params      Parameter system health
    optimize    Optimization algorithms health
    stochastic  Stochastic processes health
    orchestration Graph orchestration health
    config      Configuration system health

Authors: Yuchong Long
"""

import argparse
import logging
from typing import Dict, Any
import time
import traceback
import json

from .utils import (
    print_success, print_error, print_warning, print_info, print_header,
    format_table, save_output, ProgressReporter
)

from ...validation.healthcheck import (
    HealthChecker as _CoreHealthChecker,
    STATUS_HEALTHY,
    STATUS_DEGRADED,
)

logger = logging.getLogger(__name__)

class HealthChecker(_CoreHealthChecker):
    """CLI wrapper for health checks with formatted output."""

    def check_all(self, detailed: bool = False, benchmark: bool = False) -> Dict[str, Any]:
        """Perform complete system health check with CLI output."""
        print_header("RELIC System Health Check")

        with ProgressReporter(description="Running health checks") as progress:
            self.results["compute"] = self._check_compute_health(detailed)
            progress.update(1, "Compute layer")

            self.results["data"] = self._check_data_health(detailed)
            progress.update(1, "Data I/O")

            self.results["params"] = self._check_params_health(detailed)
            progress.update(1, "Parameters")

            self.results["optimize"] = self._check_optimize_health(detailed)
            progress.update(1, "Optimization")

            self.results["stochastic"] = self._check_stochastic_health(detailed)
            progress.update(1, "Stochastic")

            self.results["config"] = self._check_config_health(detailed)
            progress.update(1, "Configuration")

            self.results["orchestration"] = self._check_orchestration_health(detailed)
            progress.update(1, "Orchestration")

            if benchmark:
                self.results["benchmark"] = self._run_basic_benchmark()
                progress.update(1, "Benchmark")

        self._aggregate_results()
        self._print_summary()

        return {
            "overall_status": self.overall_status,
            "subsystems": self.results,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommendations": self.recommendations,
            "timestamp": time.time(),
        }

    def _print_summary(self) -> None:
        """Print health check summary."""
        print("")
        print_header("Health Check Summary")

        if self.overall_status == STATUS_HEALTHY:
            print_success(f"Overall Status: {self.overall_status.upper()}")
        elif self.overall_status == STATUS_DEGRADED:
            print_warning(f"Overall Status: {self.overall_status.upper()}")
        else:
            print_error(f"Overall Status: {self.overall_status.upper()}")

        print("")
        print("Subsystem Status:")
        status_data = []
        for subsystem, result in self.results.items():
            status = result["status"]
            status_data.append(
                {
                    "Subsystem": subsystem.title(),
                    "Status": status.upper(),
                    "Details": result.get("error", "OK" if status == STATUS_HEALTHY else status),
                }
            )

        print(format_table(status_data))

        if self.warnings:
            print("")
            print("Warnings:")
            for warning in self.warnings:
                print_warning(f"  - {warning}")

        if self.errors:
            print("")
            print("Errors:")
            for error in self.errors:
                print_error(f"  - {error}")

        if self.recommendations:
            print("")
            print("Recommendations:")
            for rec in self.recommendations:
                print_info(f"  - {rec}")



def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Setup argument parser for health check commands"""
    subparsers = parser.add_subparsers(dest='health_command', help='Health check commands')
    
    # All systems health check
    all_parser = subparsers.add_parser('all', help='Complete system health check')
    all_parser.add_argument('--detailed', action='store_true', help='Detailed diagnostics')
    all_parser.add_argument('--benchmark', action='store_true', help='Include performance benchmark')
    all_parser.add_argument(
        '--gpu-benchmark',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'With --benchmark: run (--gpu-benchmark) or skip (--no-gpu-benchmark) the GPU '
            'part; required unless the configuration sets enable_gpu_benchmark'
        ),
    )
    all_parser.add_argument('--output', type=str, help='Save report to file')
    all_parser.add_argument('--format', choices=['json', 'yaml', 'text'], default='text', help='Report format')
    
    # Individual subsystem checks
    for subsystem in ['compute', 'data', 'params', 'optimize', 'stochastic', 'orchestration', 'config']:
        sub_parser = subparsers.add_parser(subsystem, help=f'{subsystem.title()} subsystem health check')
        sub_parser.add_argument('--detailed', action='store_true', help='Detailed diagnostics')
        sub_parser.add_argument('--output', type=str, help='Save report to file')
        sub_parser.add_argument('--format', choices=['json', 'yaml', 'text'], default='text', help='Report format')

def execute(args: Any, config: Dict[str, Any]) -> bool:
    """Execute health check command"""
    try:
        # The GPU benchmark switch is an explicit request option: the CLI flag
        # sets enable_gpu_benchmark in the health configuration; without the
        # flag the configuration must declare it before a benchmark runs.
        health_config = dict(config)
        gpu_benchmark = getattr(args, 'gpu_benchmark', None)
        if gpu_benchmark is not None:
            health_config['enable_gpu_benchmark'] = gpu_benchmark
        checker = HealthChecker(health_config)

        if not hasattr(args, 'health_command') or not args.health_command:
            print_error("Health check command required")
            return False
        
        if args.health_command == 'all':
            result = checker.check_all(
                detailed=getattr(args, 'detailed', False),
                benchmark=getattr(args, 'benchmark', False)
            )
        else:
            # Individual subsystem check
            subsystem = args.health_command
            detailed = getattr(args, 'detailed', False)
            
            if subsystem == 'compute':
                subsystem_result = checker._check_compute_health(detailed)
            elif subsystem == 'data':
                subsystem_result = checker._check_data_health(detailed)
            elif subsystem == 'params':
                subsystem_result = checker._check_params_health(detailed)
            elif subsystem == 'orchestration':
                subsystem_result = checker._check_orchestration_health(detailed)
            elif subsystem == 'optimize':
                subsystem_result = checker._check_optimize_health(detailed)
            elif subsystem == 'stochastic':
                subsystem_result = checker._check_stochastic_health(detailed)
            elif subsystem == 'config':
                subsystem_result = checker._check_config_health(detailed)
            else:
                print_error(f"Unknown subsystem: {subsystem}")
                return False
            
            result = {
                'subsystem': subsystem,
                'status': subsystem_result['status'],
                'details': subsystem_result,
                'timestamp': time.time()
            }
            
            # Print individual results
            print_header(f"{subsystem.title()} Health Check")
            status = result['status']
            if status == STATUS_HEALTHY:
                print_success(f"Status: {status.upper()}")
            elif status == STATUS_DEGRADED:
                print_warning(f"Status: {status.upper()}")
            else:
                print_error(f"Status: {status.upper()}")
        
        # Save output if requested
        output_path = getattr(args, 'output', None)
        output_format = args.format
        
        if output_path:
            if output_format == 'text':
                # For text format, save a formatted report
                formatted_report = f"RELIC Health Check Report\n{'='*40}\n\n"
                formatted_report += json.dumps(result, indent=2, default=str)
                
                with open(output_path, 'w', encoding="utf-8") as f:
                    f.write(formatted_report)
            else:
                save_output(result, output_path, output_format)
        
        final_status = result['overall_status'] if args.health_command == 'all' else result['status']
        return final_status in [STATUS_HEALTHY, STATUS_DEGRADED]
        
    except (
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        KeyError,
    ) as e:
        print_error(f"Health check failed: {e}")
        logger.error(f"Health check error: {e}")
        if getattr(args, 'verbose', False):
            traceback.print_exc()
        return False
