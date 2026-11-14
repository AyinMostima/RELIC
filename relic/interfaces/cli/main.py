"""
RELIC Unified CLI Interface

This is the main entry point for all RELIC command-line operations.
It provides a unified interface to all subsystems including parameter management,
optimization algorithm-component inspection, data I/O, computational acceleration,
and stochastic processes. Optimization execution remains [N] runtime_unbound.

Architecture:
    - Modular design with dedicated subcommands for each functional area
    - Backward consistency with historical CLI interfaces
    - Unified configuration management across all operations
    - Windows-friendly with proper PowerShell integration
    - Experimental feature isolation via environment variables

Usage:
    relic <subcommand> [options]
    relic-cli <subcommand> [options]  # Alternative entry point

Available Subcommands:
    params          Parameter management and conversion
    optimization    [N] Algorithm components only; runtime_unbound
    data            Data I/O operations and streaming (experimental)
    compute         Computational optimization and acceleration
    stochastic      Stochastic processes and uncertainty quantification
    orchestration   Graph orchestration validation and compilation
    healthcheck     System-wide health diagnostics

Global Options:
    --config, -c    Configuration file path (YAML/JSON)
    --verbose, -v   Verbose logging output
    --quiet, -q     Suppress non-essential output
    --help, -h      Show help information
    --version       Show version information

Environment Variables:
    RELIC_CONFIG_PATH          Default configuration file
    RELIC_LOG_LEVEL           Logging level (DEBUG, INFO, WARNING, ERROR)
    RELIC_EXPERIMENTAL_CLI     Enable experimental CLI features
    RELIC_NO_WARNINGS         Suppress deprecation warnings

Examples:
    # Parameter operations
    relic params list --namespace photosynthesis
    relic params get vcmax25 jmax25
    relic params set vcmax25=50.0 --validate
    
    # Optimization capability boundary
    relic optimization --help  # [N] runtime_unbound
    
    # Data operations
    relic data io read climate_data.nc --variables temperature,precipitation
    relic data io write processed_data.nc --format netcdf4 --compression zstd
    
    # Computational optimization
    relic compute health --verbose
    relic compute bench --target photosynthesis --device cuda
    
    # Health diagnostics
    relic healthcheck all --detailed
    relic healthcheck compute --report-file health_report.json

Authors: Yuchong Long
License: MIT
"""

import sys
import argparse
import logging
from importlib import import_module
from pathlib import Path
from typing import Optional
import traceback

from relic.core.config.validator import ValidationSeverity

# Logging is configured by main() when a command runs; importing this module
# leaves the process root logger untouched.
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__min_python_version__ = "3.8"

# Import CLI submodules with error handling
params = import_module("relic.interfaces.cli.params")
optimization = import_module("relic.interfaces.cli.optimization")
data = import_module("relic.interfaces.cli.data")
compute = import_module("relic.interfaces.cli.compute")
stochastic = import_module("relic.interfaces.cli.stochastic")
healthcheck = import_module("relic.interfaces.cli.healthcheck")
utils = import_module("relic.interfaces.cli.utils")
CLI_MODULES_AVAILABLE = True
_CLI_IMPORT_ERROR: Optional[Exception] = None

# Import configuration system
from relic.core.services import load_runtime_config
from relic.core.services.cli_runtime import (
    check_python_version,
    setup_logging,
    handle_warnings,
    print_system_info,
    print_experimental_features,
)
CONFIG_SYSTEM_AVAILABLE = True
_CONFIG_IMPORT_ERROR: Optional[Exception] = None

def create_main_parser():
    """Create the main argument parser with all subcommands"""
    parser = argparse.ArgumentParser(
        prog='relic',
        description='RELIC Unified Command-Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    relic params list --category photosynthesis
    relic optimization --help  # [N] runtime_unbound
    relic data io read input.nc --variables temperature
    relic compute health --verbose
    relic healthcheck all

For detailed help on any subcommand:
    relic <subcommand> --help

Environment Variables:
    RELIC_CONFIG_PATH     Default configuration file
    RELIC_LOG_LEVEL       Logging level (DEBUG, INFO, WARNING, ERROR)
    RELIC_EXPERIMENTAL_*  Enable experimental features
    RELIC_NO_WARNINGS     Suppress deprecation warnings

For more information, visit: https://github.com/relic/relic
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path (YAML or JSON)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'RELIC CLI v{__version__}'
    )
    parser.add_argument(
        '--no-warnings',
        action='store_true',
        help='Suppress warning messages'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest='subcommand',
        title='Available Subcommands',
        description='Choose a functional area for operations',
        help='Subcommand help'
    )
    
    if not CLI_MODULES_AVAILABLE:
        raise RuntimeError(
            "CLI modules are unavailable. Install required dependencies and retry."
        ) from _CLI_IMPORT_ERROR
    
    # Parameter management subcommand
    params_parser = subparsers.add_parser(
        'params',
        help='Parameter management and conversion operations',
        description='Manage, validate, and convert parameters in the RELIC system'
    )
    params.setup_parser(params_parser)
    
    # Optimization subcommand
    opt_parser = subparsers.add_parser(
        'optimization',
        aliases=['opt'],
        help='[N] Optimization algorithm components; runtime_unbound',
        description=(
            '[N] Optimization algorithm components only; production execution '
            'is runtime_unbound'
        )
    )
    optimization.setup_parser(opt_parser)
    
    # Data operations subcommand
    data_parser = subparsers.add_parser(
        'data',
        help='Data I/O operations and streaming',
        description='Data input/output operations and stream processing'
    )
    data.setup_parser(data_parser)
    
    # Computational optimization subcommand
    compute_parser = subparsers.add_parser(
        'compute',
        help='Computational optimization and acceleration',
        description='GPU acceleration, JIT compilation, parallel/distributed computing'
    )
    compute.setup_parser(compute_parser)
    
    # Stochastic processes subcommand
    stoch_parser = subparsers.add_parser(
        'stochastic',
        aliases=['stoch'],
        help='Stochastic processes and uncertainty quantification',
        description='Random processes, weather generation, SDE solving, SSA'
    )
    stochastic.setup_parser(stoch_parser)

    # Orchestration subcommand
    orchestration_parser = subparsers.add_parser(
        'orchestration',
        aliases=['graph'],
        help='Graph orchestration validation and compilation',
        description='Validate and compile Graph IR workflows'
    )
    orchestration_subparsers = orchestration_parser.add_subparsers(
        dest='orchestration_command',
        help='Orchestration commands'
    )
    validate_parser = orchestration_subparsers.add_parser(
        'validate',
        help='Validate Graph IR'
    )
    validate_parser.add_argument('path', type=str)
    validate_parser.add_argument('--strict', action='store_true', default=False)

    compile_parser = orchestration_subparsers.add_parser(
        'compile',
        help='Compile Graph IR'
    )
    compile_parser.add_argument('path', type=str)
    compile_parser.add_argument('--strict', action='store_true', default=False)
    compile_parser.add_argument('--auto-adapt', action='store_true', default=False)
    compile_parser.add_argument('--run-bundle', type=str, default=None)
    
    # Health check subcommand
    health_parser = subparsers.add_parser(
        'healthcheck',
        aliases=['health'],
        help='System-wide health diagnostics',
        description='Comprehensive health checking across all RELIC subsystems'
    )
    healthcheck.setup_parser(health_parser)
    
    return parser

def _handle_early_cli_flags(parser: argparse.ArgumentParser) -> Optional[int]:
    """Handle no-args and version-only early exits."""
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    if '--version' in sys.argv:
        print_system_info(
            __version__,
            CONFIG_SYSTEM_AVAILABLE,
            load_runtime_config if CONFIG_SYSTEM_AVAILABLE else None,
        )
        return 0
    return None

def _load_and_validate_config(args: argparse.Namespace) -> dict:
    """Load runtime configuration and enforce validation failure policy.

    CLI boundary: a configuration that cannot be loaded, or one with CRITICAL
    validation messages, stops the command with exit code 1; ERROR messages
    are logged. No substituted empty configuration is ever used.
    """
    try:
        config, validation_result, _ = load_runtime_config(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Failed to load configuration: %s", exc)
        sys.exit(1)
    if not validation_result.is_valid:
        logger.error("Configuration validation failed:")
        for severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR):
            for message in validation_result.get_messages_by_severity(severity):
                logger.error("  - [%s] %s", severity.value, message)
        if validation_result.num_critical > 0:
            sys.exit(1)
    return config

def _run_orchestration_subcommand(args: argparse.Namespace) -> bool:
    """Execute orchestration validate/compile commands and emit payload."""
    from relic.orchestration.cli import run_compile, run_validate, _write_output

    graph_path = Path(args.path)
    if args.orchestration_command == 'validate':
        payload = run_validate(graph_path, strict=args.strict)
        success = len(payload.get("errors", [])) == 0
    elif args.orchestration_command == 'compile':
        bundle_dir = Path(args.run_bundle) if args.run_bundle else None
        payload = run_compile(
            graph_path,
            auto_adapt=args.auto_adapt,
            strict=args.strict,
            run_bundle_dir=bundle_dir,
        )
        success = True
    else:
        logger.error("Orchestration command required")
        return False
    _write_output(payload)
    return success

def _dispatch_subcommand(args: argparse.Namespace, config: dict) -> Optional[bool]:
    """Dispatch CLI subcommands to their module handlers."""
    handler_map = {
        'params': params.execute,
        'optimization': optimization.execute,
        'opt': optimization.execute,
        'data': data.execute,
        'compute': compute.execute,
        'stochastic': stochastic.execute,
        'stoch': stochastic.execute,
        'healthcheck': healthcheck.execute,
        'health': healthcheck.execute,
    }
    if args.subcommand in ['orchestration', 'graph']:
        return _run_orchestration_subcommand(args)
    handler = handler_map.get(args.subcommand)
    if handler is None:
        return None
    return handler(args, config)

def main():
    """Main CLI entry point"""
    if not CONFIG_SYSTEM_AVAILABLE:
        raise RuntimeError(
            "Configuration runtime is unavailable. Install required dependencies and retry."
        ) from _CONFIG_IMPORT_ERROR

    # Early system checks
    check_python_version(__min_python_version__)
    
    # Parse arguments
    parser = create_main_parser()
    early_exit_code = _handle_early_cli_flags(parser)
    if early_exit_code is not None:
        return early_exit_code
    
    args = parser.parse_args()

    # Setup logging and warnings. The command-line entry point owns the root
    # handler: basicConfig installs it (a no-op when the process already has
    # one), and setup_logging sets the level and format from the options.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    setup_logging(args.verbose, args.quiet)
    handle_warnings(args.no_warnings)
    
    # Handle experimental features
    print_experimental_features()
    
    # Load configuration
    config = _load_and_validate_config(args)
    
    # Check if subcommand was provided
    if not hasattr(args, 'subcommand') or not args.subcommand:
        parser.print_help()
        return 1
    
    try:
        success = _dispatch_subcommand(args, config)
        if success is None:
            logger.error(f"Unknown subcommand: {args.subcommand}")
            return 1
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except (
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        KeyError,
    ) as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

def cli_entry_point():
    """Entry point for relic-cli command"""
    sys.exit(main())

if __name__ == '__main__':
    sys.exit(main())
