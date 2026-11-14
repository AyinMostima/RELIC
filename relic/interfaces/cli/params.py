"""
Parameter Management CLI Module

Command-line interface for RELIC parameter management operations.
Migrated from relic/cli/params_manager.py with enhancements for
the unified interface architecture.

Features:
- List parameters by namespace or pattern
- Get/set parameter values with validation  
- Validate parameter consistency and bounds
- Export parameters to files
- Compare parameter sets
- Create parameter snapshots

Authors: Yuchong Long
"""

import argparse
import logging
from pathlib import Path
from typing import Any

# Import parameter management components
from ...parameters import operations as param_ops

logger = logging.getLogger(__name__)


def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Setup argument parser for parameter commands"""
    subparsers = parser.add_subparsers(
        dest='params_command', 
        help='Parameter management operations'
    )
    
    # List command
    list_parser = subparsers.add_parser('list', help='List parameters')
    list_parser.add_argument('--namespace', '-n', help='Filter by namespace')
    list_parser.add_argument('--pattern', '-p', help='Filter by regex pattern')
    list_parser.add_argument('--category', '-c', help='Filter by category')
    list_parser.add_argument('--details', '-d', action='store_true', 
                           help='Show detailed information')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get parameter value')
    get_parser.add_argument('key', help='Parameter key')
    get_parser.add_argument('--default', help='Default value if not found')
    get_parser.add_argument('--namespace', '-n', help='Parameter namespace')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set parameter value')
    set_parser.add_argument('key', help='Parameter key')
    set_parser.add_argument('value', help='Parameter value')
    set_parser.add_argument('--units', '-u', help='Parameter units')
    set_parser.add_argument('--no-validate', action='store_true', 
                          help='Skip validation')
    set_parser.add_argument('--namespace', '-n', help='Parameter namespace')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate parameters')
    validate_parser.add_argument('--strict', action='store_true', 
                               help='Use strict validation')
    validate_parser.add_argument('--namespace', '-n', help='Limit to namespace')
    validate_parser.add_argument('--fix', action='store_true',
                                help='Attempt to fix issues')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export parameters')
    export_parser.add_argument('--namespace', '-n', help='Namespace to export')
    export_parser.add_argument('--format', '-f', choices=['yaml', 'json'],
                             default='yaml', help='Output format')
    export_parser.add_argument('--output', '-o', type=Path, 
                             help='Output file (stdout if not specified)')
    
    # Diff command  
    diff_parser = subparsers.add_parser('diff', help='Compare parameter files')
    diff_parser.add_argument('file1', type=Path, help='First parameter file')
    diff_parser.add_argument('file2', type=Path, help='Second parameter file')
    diff_parser.add_argument('--ignore-metadata', action='store_true',
                           help='Ignore metadata differences')
    
    # Audit command
    audit_parser = subparsers.add_parser('audit', help='Audit parameter registry')
    audit_parser.add_argument('--namespace', '-n', help='Filter by namespace')
    audit_parser.add_argument('--expected-params', nargs='+', 
                            help='List of expected parameter keys')
    audit_parser.add_argument('--include-unused', action='store_true',
                            help='Include unused parameters in report')
    audit_parser.add_argument('--format', '-f', 
                            choices=['table', 'json', 'yaml', 'markdown'],
                            default='table', help='Output format')
    audit_parser.add_argument('--output', '-o', type=Path,
                            help='Output file (stdout if not specified)')
    audit_parser.add_argument('--clear-missing', action='store_true',
                            help='Clear missing requests tracking after audit')


def execute(args: Any, config: Any) -> bool:
    """Execute parameter command with minimal interface-layer validation."""
    if args is None:
        logger.error("Missing CLI arguments for params command")
        return False
    if not hasattr(args, "params_command") or not args.params_command:
        logger.error("Missing params subcommand")
        return False
    return bool(param_ops.execute_cli(args))


# Direct aliases to unified parameter operations (avoid passthrough wrappers).
export_parameters = param_ops.export_parameters
diff_parameters = param_ops.diff_parameters


# Direct re-exports from operations module (RED-003 remediation).
# These are re-exported directly to avoid unnecessary wrapper indirection.
# See: parameter_access_analysis.md for rationale.

from ...parameters.operations import (  # noqa: E402, F401
    get_parameter,
    list_parameters,
    set_parameter,
    validate_parameters,
)

__all__ = [
    "setup_parser",
    "execute",
    "export_parameters",
    "diff_parameters",
    "get_parameter",
    "list_parameters",
    "set_parameter",
    "validate_parameters",
]
