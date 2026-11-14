"""
Data CLI Module

Command-line interface for RELIC data I/O operations and streaming.
Provides comprehensive data management functionality including file I/O,
format conversion, quality validation, and experimental streaming capabilities.

Features:
- Data file reading with multiple format support (NetCDF, Zarr, CSV, HDF5)
- Format conversion with compression options
- Bounded single-variable CF standard-name/unit metadata validation
- Quality control and gap filling
- Experimental streaming data processing
- Health monitoring for data subsystems

Authors: Yuchong Long
"""

import argparse
import os
import logging
import traceback
from typing import Any

from relic.data.cli_ops import DataCLI

logger = logging.getLogger(__name__)


def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Setup argument parser for data commands"""
    subparsers = parser.add_subparsers(
        dest='data_command',
        help='Data management operations'
    )
    
    # I/O operations subcommand
    io_parser = subparsers.add_parser('io', help='Data I/O operations')
    io_subparsers = io_parser.add_subparsers(dest='io_command', help='I/O commands')
    
    # Read command
    read_parser = io_subparsers.add_parser('read', help='Read data file')
    read_parser.add_argument('path', type=str, help='Input file path')
    read_parser.add_argument('--variables', type=str, help='Comma-separated variable names')
    read_parser.add_argument('--format', type=str, choices=['auto', 'netcdf4', 'zarr', 'csv'], 
                           default='auto', help='Input format')
    read_parser.add_argument('--to-torch', action='store_true', help='Output as PyTorch tensors')
    read_parser.add_argument('--device', type=str, default='cpu', help='Target device for tensors')
    read_parser.add_argument('--dtype', type=str, default='float32', help='Target tensor dtype')
    read_parser.add_argument('--info-only', action='store_true', help='Show file info only')
    
    # Write command
    write_parser = io_subparsers.add_parser('write', help='Write data file')
    write_parser.add_argument('input', type=str, help='Input file path')
    write_parser.add_argument('output', type=str, help='Output file path')
    write_parser.add_argument('--format', type=str, choices=['netcdf4', 'zarr', 'csv'],
                            default='netcdf4', help='Output format')
    write_parser.add_argument('--compression', type=str, 
                            choices=['none', 'zlib', 'gzip', 'zstd'],
                            default='zstd', help='Compression method')
    write_parser.add_argument('--compression-level', type=int, default=4,
                            help='Compression level (1-9)')
    
    # Convert command
    convert_parser = io_subparsers.add_parser('convert', help='Convert between formats')
    convert_parser.add_argument('input', type=str, help='Input file path')
    convert_parser.add_argument('output', type=str, help='Output file path')
    convert_parser.add_argument('--target-format', type=str, required=True,
                              choices=['netcdf4', 'zarr', 'csv'], help='Target format')
    convert_parser.add_argument('--compression', type=str, default='zstd', help='Compression')
    
    # Check I/O health
    io_check_parser = io_subparsers.add_parser('check', help='I/O health check')
    io_check_parser.add_argument('--detailed', action='store_true', help='Detailed check')
    
    # Quality control subcommand
    quality_parser = subparsers.add_parser('quality', help='Data quality operations')
    quality_subparsers = quality_parser.add_subparsers(dest='quality_command', help='Quality commands')
    
    # Validate command
    validate_parser = quality_subparsers.add_parser('validate', help='Validate data quality')
    validate_parser.add_argument('input', type=str, help='Input file path')
    validate_parser.add_argument(
        '--cf-variable-metadata-check',
        action='store_true',
        help='Validate the single-variable CF standard-name/unit subset',
    )
    validate_parser.add_argument('--output', type=str, help='Validation report file')
    
    # Clean command
    clean_parser = quality_subparsers.add_parser('clean', help='Clean data issues')
    clean_parser.add_argument('input', type=str, help='Input file path')
    clean_parser.add_argument('output', type=str, help='Output file path')
    clean_parser.add_argument('--fill-gaps', action='store_true', help='Fill missing data gaps')
    clean_parser.add_argument('--remove-outliers', action='store_true', help='Remove outliers')
    
    # Streaming subcommands (experimental)
    if os.getenv('RELIC_EXPERIMENTAL_STREAMING', '').lower() in ('1', 'true'):
        stream_parser = subparsers.add_parser('stream', help='Data streaming (experimental)')
        stream_subparsers = stream_parser.add_subparsers(dest='stream_command', help='Streaming commands')
        
        # Run streaming
        stream_run_parser = stream_subparsers.add_parser('run', help='Run streaming pipeline')
        stream_run_parser.add_argument('--source', type=str, required=True, help='Source directory')
        stream_run_parser.add_argument('--output', type=str, help='Output directory')

        # Streaming health check
        stream_check_parser = stream_subparsers.add_parser('check', help='Streaming health check')
        stream_check_parser.add_argument('--detailed', action='store_true', help='Detailed check')


def execute(args: Any, config: Any) -> bool:
    """Execute data command"""
    if not hasattr(args, "data_command") or not args.data_command:
        print("Error: data command requires a subcommand")
        print("Available commands: io, quality, stream (experimental)")
        return False

    try:
        cli = DataCLI()
        return bool(cli.run(args, config))
    except (
        ImportError,
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
        KeyError,
    ) as exc:
        logger.error("Error executing data command: %s", exc)
        if hasattr(args, "verbose") and args.verbose:
            traceback.print_exc()
        return False
