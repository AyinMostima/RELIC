"""
Compute CLI Module

Command-line interface for computational optimization operations including
GPU acceleration, JIT compilation, parallel processing, and distributed computing.

This module provides direct access to the computation layer with unified
health checking, performance benchmarking, and distributed computing validation.

Features:
    - Comprehensive health diagnostics
    - Performance benchmarking with multiple targets
    - Distributed computing validation
    - GPU memory monitoring
    - JIT compilation testing
    - Experimental feature support

Commands:
    health      Computational health diagnostics
    bench       Performance benchmarking
    ddp-check   Distributed computing validation
    splitting-sensitivity  Operator splitting order sensitivity diagnostics
    dynamical-benchmark    Dynamical feature-space benchmark diagnostics

Authors: Yuchong Long
"""

import argparse
from typing import Dict, Any

from relic.parameters import get_global_manager

from .utils import (
    print_success, print_error, print_warning, print_info, print_header,
    save_output, ProgressReporter, check_experimental_feature
)
from ...computation import cli_ops as _compute_cli_ops


def _build_io():
    """Build IO callback map for computation CLI ops."""
    return {
        "print_success": print_success,
        "print_error": print_error,
        "print_warning": print_warning,
        "print_info": print_info,
        "print_header": print_header,
        "save_output": save_output,
        "ProgressReporter": ProgressReporter,
        "check_experimental_feature": check_experimental_feature,
    }

# Parameter-database keys of the diagnostic settings that the command line may omit:
# {subcommand: {namespace attribute: (key, units)}}.
_PARAMETER_DATABASE_SETTINGS = {
    'splitting-sensitivity': {
        'n_permutations': ('computation.diagnostics.splitting_sensitivity.max_permutations', 'dimensionless'),
    },
    'dynamical-benchmark': {
        'series_length': ('computation.diagnostics.dynamical_benchmark.series_length', 'dimensionless'),
        'dt': ('computation.diagnostics.dynamical_benchmark.sampling_interval', 's'),
        'frequency': ('computation.diagnostics.dynamical_benchmark.dominant_frequency', 's-1'),
    },
}


def setup_parser(parser):
    """Setup argument parser for compute commands"""
    subparsers = parser.add_subparsers(dest='compute_command', help='Compute commands')
    
    # Health check
    health_parser = subparsers.add_parser('health', help='Computational health check')
    # Verbose output is the global --verbose/-v option (args.verbose); a subcommand
    # --verbose would share its dest and reset it to False.
    health_parser.add_argument('--output', type=str, help='Save report to file')
    health_parser.add_argument('--format', choices=['json', 'yaml', 'text'], default='text')
    
    # Performance benchmark
    bench_parser = subparsers.add_parser('bench', help='Performance benchmark')
    bench_parser.add_argument(
        '--target',
        choices=['photosynthesis', 'transport', 'decomposition'],
        required=True,
        help='Target function to benchmark'
    )
    bench_parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device for benchmark'
    )
    bench_parser.add_argument('--jit', action='store_true', help='Test JIT compilation')
    bench_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    bench_parser.add_argument('--n-points', type=int, default=1000, help='Number of points')
    bench_parser.add_argument('--output', type=str, help='Save results to file')
    
    # DDP check
    ddp_parser = subparsers.add_parser('ddp-check', help='Distributed computing check')
    ddp_parser.add_argument('--world-size', type=int, default=2, help='Number of processes')
    ddp_parser.add_argument('--output', type=str, help='Save results to file')

    # Splitting sensitivity diagnostic
    split_parser = subparsers.add_parser(
        'splitting-sensitivity',
        help='Quantify operator splitting order dependency',
    )
    split_parser.add_argument(
        '--group-order',
        type=str,
        default='',
        help='Comma-separated functor order (default uses simulator fast group order)',
    )
    split_parser.add_argument('--dt', type=float, required=True, help='Time step [s]')
    split_parser.add_argument(
        '--n-permutations',
        type=int,
        default=None,
        help=(
            'Maximum permutation samples for sensitivity scan '
            '(when omitted: computation.diagnostics.splitting_sensitivity.max_permutations)'
        ),
    )
    split_parser.add_argument('--output', type=str, help='Save results to file')

    # Dynamical feature benchmark
    dyn_parser = subparsers.add_parser(
        'dynamical-benchmark',
        help='Compute spectral and embedding-based dynamical benchmark metrics',
    )
    dyn_parser.add_argument(
        '--series-length',
        type=int,
        default=None,
        help='Number of samples in synthetic trajectory (when omitted: computation.diagnostics.dynamical_benchmark.series_length)',
    )
    dyn_parser.add_argument(
        '--dt',
        type=float,
        default=None,
        help='Sampling interval [s] (when omitted: computation.diagnostics.dynamical_benchmark.sampling_interval)',
    )
    dyn_parser.add_argument(
        '--frequency',
        type=float,
        default=None,
        help=(
            'Synthetic signal dominant frequency [Hz] '
            '(when omitted: computation.diagnostics.dynamical_benchmark.dominant_frequency)'
        ),
    )
    dyn_parser.add_argument('--output', type=str, help='Save results to file')

def execute(args, config: Dict[str, Any]) -> bool:
    """Execute compute command with minimal interface-layer validation."""
    if args is None:
        return False
    if not hasattr(args, "compute_command") or not args.compute_command:
        return False
    # Diagnostic settings not stated on the command line are read from the
    # parameter database (their single source) and passed on explicitly.
    settings = _PARAMETER_DATABASE_SETTINGS.get(args.compute_command, {})
    resolved = {
        dest: get_global_manager().get(key, units=units, required=True)
        for dest, (key, units) in settings.items()
        if getattr(args, dest) is None
    }
    if resolved:
        args = argparse.Namespace(**{**vars(args), **resolved})
    return _compute_cli_ops.execute(args, config, _build_io())
