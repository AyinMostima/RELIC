"""
Data I/O Module

Unified data input/output operations for RELIC with bounded CF metadata,
automatic format detection, and optimal performance characteristics.

This module provides high-level interfaces for reading and writing scientific
datasets in various formats (NetCDF, Zarr, CSV, HDF5) with automatic CF
standardization, time coordinate processing, and quality control.

Core Components:
- DataReader: High-level data reading with CF standardization
- DataWriter: High-level data writing with compression and compliance
- DataIO: Unified adapter for all I/O operations
- Health checking and diagnostics

Key Features:
- Automatic format detection and optimization
- Explicit standard-name/unit subset validation without dataset claims
- PyTorch tensor integration
- Performance monitoring and optimization
- Comprehensive error handling and recovery

Examples:
    # Simple data reading
    from relic.data.io import read
    dataset = read("climate_data.nc")
    
    # Read variables as tensors
    from relic.data.io import DataIO
    io = DataIO()
    tensors = io.read_tensors("model_output.nc", ["temperature", "precipitation"])
    
    # Write with compression
    from relic.data.io import write
    write(dataset, "output.nc", compression="zstd", compression_level=6)
    
    # Health check
    from relic.data.io import io_healthcheck
    status = io_healthcheck()
    print(f"I/O System Status: {status.overall_status.value}")

References:
    - CF Conventions 1.8: http://cfconventions.org/cf-conventions/cf-conventions.html
    - RELIC Data Processing Architecture
    - xarray documentation: http://xarray.pydata.org/en/stable/
"""

from typing import Any

from .reader import (
    DataReader,
    ReadConfig, 
    ReadResult,
    read_dataset,
    read_tensors
)

from .writer import (
    DataWriter,
    WriteConfig,
    WriteResult,
    write_dataset,
    write_tensors
)

from .adapter import (
    DataIO,
    IOOperation,
    IOPerformanceMetrics,
    get_default_io,
    read,
    write,
    convert_format,
    get_file_info
)

from .healthcheck import (
    IOHealthChecker,
    HealthStatus,
    HealthCheckResult,
    SystemHealthReport,
    io_healthcheck
)

# Import configurations and types from external modules
from ...coupling.external.common.io import DataFormat, CompressionType

__all__ = [
    # Core classes
    'DataReader',
    'DataWriter', 
    'DataIO',
    
    # Configuration classes
    'ReadConfig',
    'WriteConfig',
    
    # Result classes
    'ReadResult',
    'WriteResult',
    'IOPerformanceMetrics',
    
    # High-level convenience functions
    'read',
    'write',
    'read_dataset',
    'read_tensors', 
    'write_dataset',
    'write_tensors',
    'convert_format',
    'get_file_info',
    
    # Default instance
    'default_io',
    'get_default_io',
    
    # Health checking
    'IOHealthChecker',
    'HealthStatus',
    'HealthCheckResult', 
    'SystemHealthReport',
    'io_healthcheck',
    
    # Enums and types
    'IOOperation',
    'DataFormat',
    'CompressionType'
]


def __getattr__(name: str) -> Any:
    if name == "default_io":
        return get_default_io()
    raise AttributeError(f"module 'relic.data.io' has no attribute '{name}'")

# Module metadata
__version__ = '1.0.0'
__author__ = 'Yuchong Long'
__description__ = 'Unified data I/O with bounded CF variable metadata support'
