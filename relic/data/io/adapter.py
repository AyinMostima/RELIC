"""
Data I/O Adapter Module

Provides unified high-level API for data I/O operations in RELIC.
This module serves as the primary interface for all data reading and writing
operations, integrating IOHandler, TimeHandler, and CF conventions.

The adapter encapsulates complex I/O operations behind simple, consistent
interfaces while preserving bounded CF metadata, performance, and
seamless integration with the broader RELIC ecosystem.

Key Features:
- Unified API for read/write operations across all formats
- Automatic format detection and optimization
- Scope-explicit standard-name/unit metadata handling
- Integration with RELIC state management
- Performance monitoring and optimization
- Error handling and recovery

Architecture:
DataIO (high-level API)
  ??? DataReader (reading operations)
  ??? DataWriter (writing operations) 
  ??? IOHandler (low-level I/O operations)

Mathematical Framework:
Data I/O operations maintain scientific integrity through:
- Format-agnostic data representation: D = F?^1(raw_data, format_spec)
- CF standardization: D_cf = CF(D, conventions, metadata)
- Performance optimization: D_opt = O(D_cf, chunking, compression)

References:
- RELIC Architecture Documentation
- CF Conventions 1.8: http://cfconventions.org/cf-conventions/cf-conventions.html
- xarray I/O guide: http://xarray.pydata.org/en/stable/io.html
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
from copy import deepcopy
from enum import Enum

import numpy as np
import xarray as xr
import torch
from torch import Tensor

# Import our modules
from .reader import DataReader, ReadConfig, ReadResult
from .writer import DataWriter, WriteConfig, WriteResult
from ..standards.cf_conventions import CFConventions
from ...coupling.external.common.io import DataFormat
from ...core.variable_mapper import variable_name_mapper


class IOOperation(Enum):
    """Supported I/O operations."""
    READ = "read"
    WRITE = "write"
    COPY = "copy"
    CONVERT = "convert"


@dataclass
class IOPerformanceMetrics:
    """
    Performance metrics for I/O operations.
    
    Tracks throughput, memory usage, and optimization effectiveness
    across different I/O operations and file formats.
    """
    operation: IOOperation
    format: DataFormat
    file_size: int                          # File size in bytes
    data_size: int                          # Uncompressed data size in bytes
    operation_time: float                   # Total operation time in seconds
    throughput_mbps: float                  # Throughput in MB/s
    memory_peak: Optional[float] = None     # Peak memory usage in MB
    compression_ratio: Optional[float] = None  # Compression ratio achieved
    cache_hit_rate: Optional[float] = None  # Cache hit rate percentage


class DataIO:
    """
    Unified high-level data I/O interface for RELIC.
    
    Provides comprehensive data I/O capabilities with automatic optimization,
    bounded CF metadata handling and integration with RELIC components.
    
    This class serves as the primary entry point for all data I/O operations
    within the RELIC ecosystem, offering both convenience functions and
    advanced configuration options.
    """
    
    def __init__(self, 
                 reader_config: Optional[ReadConfig] = None,
                 writer_config: Optional[WriteConfig] = None):
        """
        Initialize DataIO with optional configurations.
        
        Parameters:
            reader_config: Configuration for reading operations
            writer_config: Configuration for writing operations
        """
        # Initialize components
        self.reader = DataReader(reader_config)
        self.writer = DataWriter(writer_config)
        self.cf_conventions = CFConventions()
        
        # Performance tracking
        self._performance_history: List[IOPerformanceMetrics] = []
        
        # Variable name mapping for RELIC integration
        self._variable_mapper = variable_name_mapper
    
    # High-level read operations
    def read(self, 
             source: Union[str, Path, List[Union[str, Path]]],
             **kwargs) -> Union[xr.Dataset, Dict[str, Tensor]]:
        """
        Read data from file(s) with automatic optimization.
        
        Parameters:
            source: File path(s) to read
            **kwargs: Configuration overrides
            
        Returns:
            Dataset or tensor dictionary based on configuration
            
        Examples:
            # Read NetCDF dataset
            data = io.read("climate_data.nc")
            
            # Read specific variables as tensors
            tensors = io.read("model_output.nc", 
                            variables=["temperature", "precipitation"],
                            to_torch=True)
            
            # Read with custom chunking
            data = io.read("large_file.nc", chunks={"time": 100, "lat": 50})
        """
        time.time()
        
        try:
            result = self.reader.read_dataset(source, **kwargs)
            
            # Track performance
            self._record_performance(
                operation=IOOperation.READ,
                file_size=result.file_size,
                operation_time=result.load_time,
                format=DataFormat.NETCDF4  # Default assumption
            )
            
            return result.data
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            warnings.warn(f"Read operation failed: {e}")
            raise
    
    def read_variables(self, 
                      source: Union[str, Path],
                      variables: List[str],
                      **kwargs) -> Union[Dict[str, xr.DataArray], Dict[str, Tensor]]:
        """
        Read specific variables from dataset.
        
        Parameters:
            source: Source file path
            variables: List of variable names to read
            **kwargs: Configuration overrides
            
        Returns:
            Dictionary mapping variable names to data arrays or tensors
        """
        kwargs.update({'variables': variables})
        result = self.reader.read_dataset(source, **kwargs)
        
        if isinstance(result.data, dict):
            return result.data
        else:
            # Extract variables from dataset
            var_dict = {}
            for var_name in variables:
                if var_name in result.data.data_vars:
                    var_dict[var_name] = result.data[var_name]
            return var_dict
    
    def read_tensors(self, 
                    source: Union[str, Path],
                    variables: List[str],
                    device: str = 'cpu',
                    dtype: torch.dtype = torch.float32) -> Dict[str, Tensor]:
        """
        Read variables as PyTorch tensors with specified device and dtype.
        
        Parameters:
            source: Source file path
            variables: List of variable names
            device: Target device ('cpu', 'cuda', etc.)
            dtype: Target tensor dtype
            
        Returns:
            Dictionary of variable tensors
        """
        return self.reader.read_tensors(
            source, variables,
            torch_device=device,
            torch_dtype=dtype
        )
    
    # High-level write operations
    def write(self,
              data: Union[xr.Dataset, Dict[str, Tensor], Dict[str, np.ndarray]],
              output_path: Union[str, Path],
              **kwargs) -> WriteResult:
        """
        Write data to file with automatic optimization.
        
        Parameters:
            data: Data to write (Dataset, tensors, or arrays)
            output_path: Output file path
            **kwargs: Configuration overrides
            
        Returns:
            WriteResult with operation details
            
        Examples:
            # Write xarray Dataset
            io.write(dataset, "output.nc")
            
            # Write tensors with metadata template
            io.write(tensor_dict, "output.nc", template=template_dataset)
            
            # Write with compression
            io.write(dataset, "compressed.nc", compression="zstd", compression_level=6)
        """
        time.time()
        
        try:
            if isinstance(data, xr.Dataset):
                result = self.writer.write_dataset(data, output_path, **kwargs)
            elif isinstance(data, dict):
                # Handle tensor or array dictionaries
                template = kwargs.pop('template', None)
                tensors = {}
                for name, value in data.items():
                    if isinstance(value, Tensor):
                        tensors[name] = value
                    elif isinstance(value, np.ndarray):
                        array = value if value.flags.writeable else value.copy()
                        tensors[name] = torch.from_numpy(array)
                    else:
                        raise TypeError(f"Variable {name} must be a Tensor or NumPy array")
                result = self.writer.write_tensors(tensors, output_path, template, **kwargs)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Track performance
            self._record_performance(
                operation=IOOperation.WRITE,
                file_size=result.file_size,
                operation_time=result.write_time,
                format=kwargs.get('format', DataFormat.NETCDF4),
                compression_ratio=result.compression_ratio
            )
            
            return result
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            warnings.warn(f"Write operation failed: {e}")
            raise
    
    def write_tensors(self,
                     tensors: Dict[str, Tensor],
                     output_path: Union[str, Path],
                     coordinates: Optional[Dict[str, np.ndarray]] = None,
                     metadata: Optional[Dict[str, Dict[str, Any]]] = None,
                     **kwargs) -> WriteResult:
        """
        Write PyTorch tensors with optional coordinates and metadata.
        
        Parameters:
            tensors: Dictionary of tensors to write
            output_path: Output file path
            coordinates: Coordinate arrays
            metadata: Per-variable attributes with explicit dims and optional
                xarray encoding; coordinate attributes use their coordinate
                names, and global dataset attributes use the global key.
            **kwargs: Configuration overrides
            
        Returns:
            WriteResult with operation details
        """
        # Create template dataset if coordinates provided
        template = None
        if coordinates or metadata:
            template = self._create_template_dataset(coordinates, metadata, tensors=tensors)
        
        return self.writer.write_tensors(tensors, output_path, template, **kwargs)
    
    # Format conversion operations
    def convert_format(self,
                      source_path: Union[str, Path],
                      target_path: Union[str, Path],
                      target_format: str,
                      **kwargs) -> Tuple[ReadResult, WriteResult]:
        """
        Convert data from one format to another.
        
        Parameters:
            source_path: Source file path
            target_path: Target file path
            target_format: Target format ("netcdf4", "zarr", "csv")
            **kwargs: Configuration overrides for read/write
            
        Returns:
            Tuple of (read_result, write_result)
            
        Examples:
            # Convert NetCDF to Zarr
            read_result, write_result = io.convert_format(
                "data.nc", "data.zarr", "zarr"
            )
            
            # Convert with compression optimization
            io.convert_format(
                "uncompressed.nc", "compressed.nc", "netcdf4",
                compression="zstd", compression_level=9
            )
        """
        target_data_format = DataFormat(target_format)

        # Read source data
        read_kwargs = {
            k.removeprefix('read_'): v
            for k, v in kwargs.items()
            if k.startswith('read_')
        }
        # Format conversion requires the dataset's coordinate and metadata
        # contract, even when ordinary reads default to tensor dictionaries.
        read_kwargs['to_torch'] = False
        read_result = self.reader.read_dataset(source_path, **read_kwargs)
        if not isinstance(read_result.data, xr.Dataset):
            raise TypeError("Format conversion requires a Dataset from DataReader")
        
        # Write to target format
        write_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('read_')}
        write_kwargs['format'] = target_format
        write_result = self.writer.write_dataset(read_result.data, target_path, **write_kwargs)
        
        # Track conversion performance
        self._record_performance(
            operation=IOOperation.CONVERT,
            file_size=write_result.file_size,
            operation_time=read_result.load_time + write_result.write_time,
            format=target_data_format,
            compression_ratio=write_result.compression_ratio
        )
        
        return read_result, write_result
    
    def copy_with_processing(self,
                           source_path: Union[str, Path],
                           target_path: Union[str, Path],
                           processing_func: Callable[[xr.Dataset], xr.Dataset],
                           **kwargs) -> Tuple[ReadResult, WriteResult]:
        """
        Copy data with custom processing applied.
        
        Parameters:
            source_path: Source file path
            target_path: Target file path  
            processing_func: Function to apply to dataset
            **kwargs: Configuration overrides
            
        Returns:
            Tuple of (read_result, write_result)
            
        Example:
            # Copy with unit conversion
            def convert_temperature(ds):
                ds['temperature'] = ds['temperature'] - 273.15  # K to C
                ds['temperature'].attrs['units'] = 'degrees_C'
                return ds
            
            io.copy_with_processing("input.nc", "output.nc", convert_temperature)
        """
        # Read source data
        read_kwargs = {**kwargs, 'to_torch': False}
        read_result = self.reader.read_dataset(source_path, **read_kwargs)
        if not isinstance(read_result.data, xr.Dataset):
            raise TypeError("Dataset processing requires a Dataset from DataReader")
        
        # Apply processing
        try:
            processed_data = processing_func(read_result.data)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            raise ValueError(f"Processing function failed: {e}")
        
        # Write processed data
        write_result = self.writer.write_dataset(processed_data, target_path, **kwargs)
        
        return read_result, write_result
    
    # Utility and introspection methods
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive information about a data file.
        
        Parameters:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            'path': str(file_path),
            'size_bytes': file_path.stat().st_size,
            'format': None,
            'variables': [],
            'dimensions': {},
            'coordinates': [],
            'global_attributes': {},
            'cf_conventions_declared': False,
            'dataset_conformance_checked': False,
            'cf_validation_scope': 'not_evaluated',
            'time_coverage': None
        }
        
        # Try to read file metadata without loading data
        with xr.open_dataset(file_path, decode_times=False) as ds:
            info['format'] = 'netcdf'
            info['variables'] = list(ds.data_vars.keys())
            info['dimensions'] = dict(ds.sizes)
            info['coordinates'] = list(ds.coords.keys())
            info['global_attributes'] = dict(ds.attrs)

            # Record only declaration presence; no dataset conformance is inferred.
            info['cf_conventions_declared'] = 'Conventions' in ds.attrs

            # Get time coverage if available
            if 'time' in ds.coords:
                time_values = ds.time.values
                info['time_coverage'] = {
                    'start': str(time_values[0]),
                    'end': str(time_values[-1]),
                    'length': len(time_values)
                }

        
        return info
    
    def list_formats(self) -> Dict[str, List[str]]:
        """
        List supported I/O formats and their capabilities.
        
        Returns:
            Dictionary mapping format categories to format lists
        """
        return {
            'read_formats': ['netcdf4', 'netcdf3', 'hdf5', 'zarr', 'csv'],
            'write_formats': ['netcdf4', 'netcdf3', 'zarr', 'csv'],
            'compression_methods': ['none', 'zlib', 'gzip', 'zstd', 'blosc'],
            'chunking_strategies': ['auto', 'time_first', 'spatial_first', 'custom']
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from recent I/O operations.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self._performance_history:
            return {'message': 'No performance data available'}
        
        # Compute statistics
        read_ops = [p for p in self._performance_history if p.operation == IOOperation.READ]
        write_ops = [p for p in self._performance_history if p.operation == IOOperation.WRITE]
        
        stats: Dict[str, Union[int, float]] = {
            'total_operations': len(self._performance_history),
            'read_operations': len(read_ops),
            'write_operations': len(write_ops),
        }
        
        if read_ops:
            read_throughputs = [op.throughput_mbps for op in read_ops]
            stats['read_throughput_avg'] = float(np.mean(read_throughputs))
            stats['read_throughput_max'] = float(np.max(read_throughputs))
        
        if write_ops:
            write_throughputs = [op.throughput_mbps for op in write_ops]
            stats['write_throughput_avg'] = float(np.mean(write_throughputs))
            stats['write_throughput_max'] = float(np.max(write_throughputs))
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all I/O caches and performance history."""
        self.reader.io_handler.clear_cache()
        self._performance_history.clear()
    
    # RELIC integration methods
    def standardize_for_relic(self, 
                              dataset: xr.Dataset,
                              target_variables: Optional[List[str]] = None) -> xr.Dataset:
        """
        Apply local RELIC name mapping without inventing CF metadata.
        
        Parameters:
            dataset: Input dataset
            target_variables: Variables to standardize (None = all)
            
        Returns:
            Dataset with local variable names and validated explicit metadata
        """
        processed = dataset.copy()

        # Validate explicit metadata before any rename or attribute mutation.
        for var_name in processed.data_vars:
            validation = self.cf_conventions.validate_cf_compliance(
                dict(processed[var_name].attrs)
            )
            if not validation.is_valid:
                raise ValueError(
                    f"CF metadata invalid for '{var_name}': "
                    + "; ".join(validation.issues)
                )
        
        # Get variables to process
        if target_variables is None:
            target_variables = list(dataset.data_vars.keys())
        
        # Apply variable name mapping
        name_mappings = {}
        for var_name in target_variables:
            if var_name in processed.data_vars:
                # Get RELIC internal name
                internal_name = self._variable_mapper.to_module(var_name)
                if (
                    internal_name != var_name
                    and internal_name not in processed.data_vars
                ):
                    name_mappings[var_name] = internal_name
        
        # Rename variables
        if name_mappings:
            processed = processed.rename(name_mappings)
        
        return processed
    
    # Private helper methods
    def _record_performance(self, 
                           operation: IOOperation,
                           file_size: int,
                           operation_time: float,
                           format: DataFormat,
                           compression_ratio: Optional[float] = None) -> None:
        """Record performance metrics for an I/O operation."""
        throughput = (file_size / (1024 * 1024)) / max(operation_time, 0.001)  # MB/s
        
        metrics = IOPerformanceMetrics(
            operation=operation,
            format=format,
            file_size=file_size,
            data_size=file_size,  # Estimate
            operation_time=operation_time,
            throughput_mbps=throughput,
            compression_ratio=compression_ratio
        )
        
        self._performance_history.append(metrics)
        
        # Keep only recent history (last 100 operations)
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]
    
    def _create_template_dataset(self, 
                                coordinates: Optional[Dict[str, np.ndarray]] = None,
                                metadata: Optional[Dict[str, Dict[str, Any]]] = None,
                                *, tensors: Optional[Dict[str, Tensor]] = None) -> xr.Dataset:
        """Bind declared axes and attributes for the existing tensor writer."""
        coords = coordinates or {}
        attrs = metadata or {}
        
        # Create empty dataset with coordinates
        ds = xr.Dataset(coords=coords)
        
        # Add global attributes
        ds.attrs = deepcopy(attrs.get('global', {}))
        for name in ds.coords:
            declaration = attrs.get(str(name), {})
            ds[name].attrs.update(deepcopy({
                key: value for key, value in declaration.items() if key not in ("dims", "encoding")
            }))
            ds[name].encoding.update(deepcopy(declaration.get("encoding", {})))
        for name, tensor in (tensors or {}).items():
            declaration = attrs.get(name)
            if declaration is None:
                continue
            if name in ds.coords:
                raise ValueError(f"Tensor variable {name} conflicts with a coordinate")
            dims = declaration.get("dims", () if tensor.ndim == 0 else None)
            if isinstance(dims, str):
                dims = (dims,)
            if (
                not isinstance(dims, (list, tuple))
                or any(not isinstance(dim, str) or not dim for dim in dims)
                or len(dims) != tensor.ndim or len(set(dims)) != len(dims)
            ):
                raise ValueError(f"Variable {name} requires explicit, unique dims matching tensor dimensions")
            # Templates carry shape/metadata only. The broadcast view avoids
            # allocating a second full scientific array before DataWriter binds
            # and freezes the actual tensor values.
            ds[name] = xr.Variable(
                tuple(dims), np.broadcast_to(np.array(0.0), tuple(tensor.shape)),
                attrs=deepcopy({key: value for key, value in declaration.items() if key not in ("dims", "encoding")}),
                encoding=deepcopy(declaration.get("encoding", {})),
            )
        
        return ds


# Create default instance lazily to avoid import-time side effects
_DEFAULT_IO: Optional[DataIO] = None


def get_default_io() -> DataIO:
    """Return the shared DataIO instance, creating it on first use."""
    global _DEFAULT_IO
    if _DEFAULT_IO is None:
        _DEFAULT_IO = DataIO()
    return _DEFAULT_IO

# Convenience functions using default instance
def read(source: Union[str, Path, List[Union[str, Path]]], **kwargs) -> Union[xr.Dataset, Dict[str, Tensor]]:
    """Read data using default DataIO instance."""
    return get_default_io().read(source, **kwargs)

def write(data: Union[xr.Dataset, Dict[str, Tensor]], 
         output_path: Union[str, Path], 
         **kwargs) -> WriteResult:
    """Write data using default DataIO instance."""
    return get_default_io().write(data, output_path, **kwargs)

def convert_format(source_path: Union[str, Path],
                  target_path: Union[str, Path], 
                  target_format: str,
                  **kwargs) -> Tuple[ReadResult, WriteResult]:
    """Convert format using default DataIO instance."""
    return get_default_io().convert_format(source_path, target_path, target_format, **kwargs)

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get file information using default DataIO instance."""
    return get_default_io().get_file_info(file_path)
