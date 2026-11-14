"""
Input/Output Module for Stable-Isotope Tracking

This module provides input/output functionality for the stable-isotope tracking
system, including candidate NetCDF output with bounded CF metadata.

Main Components:
    - IsotopeNetCDFWriter: CF-1.8 compliant NetCDF output
    - IsotopeDataBuffer: Data buffering and aggregation
    - Utility functions for file validation and management

The IO module is designed to integrate with the isotope tracking service
and heartbeat functor to provide regular output of isotope tracking results
in standardized scientific formats.

Author: Yuchong Long
Created: 2024-08-29
Version: 1.0.0
"""

from .writer import (
    IsotopeNetCDFWriter,
    IsotopeDataBuffer,
    NetCDFWriteError,
    AggregationError,
    MetadataError,
    CompressionLevel,
    create_isotope_netcdf_writer,
    validate_netcdf_file,
    CF_VERSION
)

# Export all public components
__all__ = [
    'IsotopeNetCDFWriter',
    'IsotopeDataBuffer', 
    'NetCDFWriteError',
    'AggregationError',
    'MetadataError',
    'CompressionLevel',
    'create_isotope_netcdf_writer',
    'validate_netcdf_file',
    'CF_VERSION'
]
