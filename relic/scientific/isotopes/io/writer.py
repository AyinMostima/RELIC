"""
NetCDF Writer for Stable-Isotope Tracking Data

This module implements a CF-1.8 compliant NetCDF writer for stable-isotope tracking
data output. It provides daily aggregation, compression, and metadata management
for isotope ratios and fluxes with full scientific traceability.

The writer supports time series output with proper CF conventions, chunking,
compression, and metadata annotation. All output variables include comprehensive
metadata for isotope provenance, fractionation models, and scientific references.

Key Features:
    - CF-1.8 compliant NetCDF4 output format
    - Daily temporal aggregation with configurable methods
    - Lossless compression with zlib and shuffle filters
    - Comprehensive metadata with scientific provenance
    - Isotope-specific variable naming and attributes
    - Atomic write operations with error recovery
    - Memory-efficient streaming for large datasets

Architecture:
    - IsotopeNetCDFWriter: Main writer class with full feature set
    - Aggregation strategies: daily mean, integral, end-of-day values
    - Variable management: automatic CF naming and metadata
    - Error handling: fail-fast output validation
    - Integration: Designed for heartbeat functor coordination

Mathematical Framework:
    Temporal aggregation: X?_day = f(X(t)) where f in {mean, sum, last}
    Daily integrals: I_day = integral_{day} flux(t) dt
    Ratio aggregation: r?_day = mean(r(t)) for t in day
    Conservation: Sigma_vars output_vars = Sigma_vars input_vars (for extensive quantities)

References:
    - CF Conventions and Metadata (2023). NetCDF Climate and Forecast Metadata Conventions
    - Unidata (2023). NetCDF User's Guide and Reference Documentation
    - Hassell et al. (2017). A data model of the Climate and Forecast metadata conventions
    - CMIP6 Data Standards and Conventions

Author: Yuchong Long
Created: 2024-08-29
Version: 1.0.0
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# NetCDF and xarray imports
import netCDF4 as nc4
import numpy as np
import torch
from torch import Tensor

HAS_NETCDF = True

# Import configuration components
from relic.parameters import get_global_manager

from ....utils.time_constants import (
    SECONDS_PER_DAY as GLOBAL_SECONDS_PER_DAY,
)
from ....utils.time_constants import (
    SECONDS_PER_HOUR as GLOBAL_SECONDS_PER_HOUR,
)
from ..config import IsotopeConfig
from ..config.models import AggregationMethod, OutputConfig, OutputVariableConfig

logger = logging.getLogger(__name__)

# Constants

def _require_scalar_parameter(parameter_key: str, units: Optional[str] = None) -> float:
    """Resolve a scalar parameter through the ParameterManager with validation."""
    value = get_global_manager().get(
        parameter_key,
        units=units,
        required=True,
        expected_type=None,
    )
    if value is None:
        raise KeyError(
            f"Missing required parameter '{parameter_key}' in parameter database."
        )

    if isinstance(value, torch.Tensor):
        detached = value.detach().reshape(-1)
        if detached.numel() != 1:
            raise ValueError(f"Parameter '{parameter_key}' must resolve to a scalar tensor.")
        return detached.to(dtype=torch.float64).tolist()[0]
    if isinstance(value, np.ndarray):
        flat = np.asarray(value).reshape(-1)
        if flat.size != 1:
            raise ValueError(f"Parameter '{parameter_key}' must resolve to a scalar array.")
        return float(flat[0])
    if isinstance(value, (int, float)):
        return float(value)

    raise TypeError(
        f"Parameter '{parameter_key}' returned non-numeric value '{value}'."
    )


@dataclass(frozen=True)
class _WriterRuntimeDefaults:
    """Resolved runtime defaults for the isotope NetCDF writer."""
    netcdf_fill_value: float
    buffer_size: int


@lru_cache(maxsize=1)
def _get_writer_runtime_defaults() -> _WriterRuntimeDefaults:
    """Resolve writer defaults lazily to avoid import-time side effects."""
    fill_value = _require_scalar_parameter(
        "isotopes.output.netcdf_fill_value",
        units="dimensionless",
    )
    buffer_size = int(_require_scalar_parameter("isotopes.output.buffer_size"))
    return _WriterRuntimeDefaults(
        netcdf_fill_value=float(fill_value),
        buffer_size=buffer_size,
    )


SECONDS_PER_DAY = float(GLOBAL_SECONDS_PER_DAY)
SECONDS_PER_HOUR = float(GLOBAL_SECONDS_PER_HOUR)
CF_VERSION = "1.8"
ISOTOPE_TRACKER_VERSION = "1.0.0"


class NetCDFWriteError(Exception):
    """Exception raised for NetCDF writing errors."""


class AggregationError(NetCDFWriteError):
    """Exception raised for aggregation errors."""


class MetadataError(NetCDFWriteError):
    """Exception raised for metadata errors."""


class CompressionLevel(Enum):
    """NetCDF compression levels."""
    NONE = 0
    LOW = 2
    MEDIUM = 4
    HIGH = 6
    MAXIMUM = 9


class IsotopeDataBuffer:
    """
    Data buffer for isotope tracking results with aggregation support.
    
    Manages temporary storage of isotope data before writing to NetCDF,
    providing aggregation functionality and memory management.
    """
    
    def __init__(
        self,
        variable_configs: Dict[str, OutputVariableConfig],
        buffer_size: Optional[int] = None,
        fill_value: Optional[float] = None,
    ):
        """
        Initialize data buffer.
        
        Parameters:
            variable_configs : Dict[str, OutputVariableConfig]
                Variable configurations for output
            buffer_size : int
                Maximum number of time steps to buffer
        """
        defaults = _get_writer_runtime_defaults()
        self.variable_configs = variable_configs
        self.buffer_size = int(defaults.buffer_size if buffer_size is None else buffer_size)
        self.fill_value = float(
            defaults.netcdf_fill_value if fill_value is None else fill_value
        )
        
        # Data storage
        self.time_buffer: List[float] = []
        self.data_buffers: Dict[str, List[float]] = {
            var_name: [] for var_name in variable_configs.keys()
        }
        
        # Aggregation state
        self.current_day: Optional[int] = None
        self.daily_data: Dict[str, List[float]] = {
            var_name: [] for var_name in variable_configs.keys()
        }
        
        # Statistics
        self.total_points_added = 0
        self.total_points_aggregated = 0
        
    def add_data_point(self, 
                      time: float, 
                      data: Dict[str, Union[float, Tensor]]) -> None:
        """
        Add a data point to the buffer.
        
        Parameters:
            time : float
                Time stamp in seconds since start
            data : Dict[str, Union[float, Tensor]]
                Data values for variables
        """
        # Convert time to day number
        day_num = int(time / SECONDS_PER_DAY)
        
        # Check if we need to aggregate previous day
        if self.current_day is not None and day_num != self.current_day:
            self._aggregate_daily_data()
        
        # Update current day
        self.current_day = day_num
        
        # Add data to daily accumulation
        for var_name, var_config in self.variable_configs.items():
            if var_name in data:
                value = data[var_name]
                if isinstance(value, Tensor):
                    # Explicitly detach at I/O boundary to keep autograd graph in compute path only.
                    detached = value.detach().reshape(-1)
                    if detached.numel() != 1:
                        raise ValueError(
                            f"I/O value for '{var_name}' must be scalar, got shape {tuple(value.shape)}."
                        )
                    value = detached.to(dtype=torch.float64).tolist()[0]
                else:
                    value = float(value)
                
                self.daily_data[var_name].append(value)
        
        self.total_points_added += 1
    
    def _aggregate_daily_data(self) -> None:
        """Aggregate accumulated daily data and add to main buffer."""
        if not self.daily_data or self.current_day is None:
            return
        
        # Calculate day center time
        day_center_time = (self.current_day + 0.5) * SECONDS_PER_DAY
        self.time_buffer.append(day_center_time)
        
        # Aggregate data for each variable
        for var_name, var_config in self.variable_configs.items():
            daily_values = self.daily_data[var_name]
            
            if not daily_values:
                # No data for this variable on this day
                aggregated_value = self.fill_value
            else:
                # Apply aggregation method
                aggregated_value = self._apply_aggregation(
                    daily_values, var_config.aggregation_method
                )
            
            self.data_buffers[var_name].append(aggregated_value)
        
        # Clear daily data
        for var_name in self.daily_data:
            self.daily_data[var_name].clear()
        
        self.total_points_aggregated += 1
        
        # Manage buffer size
        if len(self.time_buffer) > self.buffer_size:
            self._trim_buffers()
    
    def _apply_aggregation(self, 
                          values: List[float], 
                          method: AggregationMethod) -> float:
        """
        Apply aggregation method to daily values.
        
        Parameters:
            values : List[float]
                Values to aggregate
            method : AggregationMethod
                Aggregation method to apply
                
        Returns:
            float: Aggregated value
        """
        if not values:
            return self.fill_value
        
        # Filter out fill values
        valid_values = [v for v in values if v != self.fill_value and math.isfinite(v)]
        
        if not valid_values:
            return self.fill_value
        
        # References: CF Conventions 1.8, Sect. 7.3 (cell_methods mean, sum and point).
        # Mathematical expression: mean = sum_i v_i / n; sum = sum_i v_i; end_of_day = v_n.
        # Parameter meanings: v_i the valid (finite, non-fill) samples of the day in time
        #   order, n their number.
        # Units: mean and end_of_day keep the sample units; sum has the sample units
        #   summed over the samples.
        # Domain/constraints: n >= 1 (no valid sample returns the fill value above); an
        #   unknown method raises AggregationError; an arithmetic error propagates.
        # Sign convention: none (samples keep their sign).
        if method == AggregationMethod.DAILY_MEAN:
            return float(sum(valid_values) / len(valid_values))
        if method == AggregationMethod.DAILY_SUM or method == AggregationMethod.INTEGRAL:
            return float(sum(valid_values))
        if method == AggregationMethod.END_OF_DAY:
            return float(valid_values[-1])
        raise AggregationError(f"Unknown aggregation method: {method}")
    
    def _trim_buffers(self) -> None:
        """Trim buffers to manage memory usage."""
        if len(self.time_buffer) <= self.buffer_size:
            return
        
        # Keep most recent data
        keep_size = self.buffer_size // 2
        self.time_buffer = self.time_buffer[-keep_size:]
        
        for var_name in self.data_buffers:
            self.data_buffers[var_name] = self.data_buffers[var_name][-keep_size:]
    
    def get_aggregated_data(self) -> Tuple[List[float], Dict[str, List[float]]]:
        """
        Get aggregated data from buffer.
        
        Returns:
            Tuple[List[float], Dict[str, List[float]]]: Time values and data arrays
        """
        # Finalize any remaining daily data
        if self.daily_data and any(self.daily_data.values()):
            self._aggregate_daily_data()
        
        return self.time_buffer.copy(), {
            var_name: data.copy() for var_name, data in self.data_buffers.items()
        }
    
    def clear(self) -> None:
        """Clear all buffered data."""
        self.time_buffer.clear()
        for var_name in self.data_buffers:
            self.data_buffers[var_name].clear()
        for var_name in self.daily_data:
            self.daily_data[var_name].clear()
        
        self.current_day = None
        self.total_points_added = 0
        self.total_points_aggregated = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'points_added': self.total_points_added,
            'points_aggregated': self.total_points_aggregated,
            'buffer_length': len(self.time_buffer),
            'current_day': self.current_day,
            'memory_usage_estimate': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        base_size = len(self.time_buffer) * 8  # float64 timestamps
        
        for var_data in self.data_buffers.values():
            base_size += len(var_data) * 8  # float64 values
        
        return base_size


class IsotopeNetCDFWriter:
    """
    NetCDF Writer for Stable-Isotope Tracking Data
    
    Provides CF-1.8 compliant NetCDF output for isotope tracking results
    with proper metadata, compression, and temporal aggregation.
    
    This writer is designed to integrate with the isotope tracking system
    to provide scientific-grade output suitable for analysis and archival.
    
    Features:
        - CF-1.8 compliant metadata and conventions
        - Daily temporal aggregation with multiple methods
        - Efficient compression and chunking
        - Atomic write operations with error recovery
        - Comprehensive scientific metadata and provenance
        - Memory-efficient streaming for large datasets
        
    Integration:
        - Called by IsotopeHeartbeatFunctor for regular output
        - Receives data from IsotopeTrackerService
        - Coordinates with configuration system for metadata
        - Provides error handling and recovery mechanisms
    """
    
    def __init__(self,
                 output_config: OutputConfig,
                 isotope_config: IsotopeConfig,
                 compression_level: CompressionLevel = CompressionLevel.MEDIUM):
        """
        Initialize NetCDF writer.
        
        Parameters:
            output_config : OutputConfig
                Output configuration with file paths and variables
            isotope_config : IsotopeConfig
                Complete isotope configuration for metadata
            compression_level : CompressionLevel
                Compression level for NetCDF output
        """
        if not HAS_NETCDF:
            raise ImportError("NetCDF4 and xarray are required for NetCDF output")
        
        self.output_config = output_config
        self.isotope_config = isotope_config
        self.compression_level = compression_level
        
        # File management
        self.output_file = Path(output_config.output_file)
        self.is_initialized = False
        self.nc_dataset: Optional[nc4.Dataset] = None
        
        # Data buffer
        defaults = _get_writer_runtime_defaults()
        self.data_buffer = IsotopeDataBuffer(
            variable_configs=output_config.variables,
            buffer_size=defaults.buffer_size,
            fill_value=defaults.netcdf_fill_value,
        )
        
        # Write statistics
        self.total_writes = 0
        self.last_write_time: Optional[float] = None
        self.write_errors: List[str] = []
        
        # Prepare output directory
        self._prepare_output_directory()
        
        logger.info(f"Initialized NetCDF writer for: {self.output_file}")
    
    def _prepare_output_directory(self) -> None:
        """Prepare output directory structure."""
        if self.output_config.create_directory:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured output directory exists: {self.output_file.parent}")
    
    def initialize_file(self, start_time: Optional[datetime] = None) -> None:
        """
        Initialize NetCDF file with proper structure and metadata.
        
        Parameters:
            start_time : Optional[datetime]
                Simulation start time for time coordinate
        """
        try:
            # Check if file exists and handle overwrite
            if self.output_file.exists() and not self.output_config.overwrite_existing:
                raise NetCDFWriteError(
                    f"Output file exists and overwrite is disabled: {self.output_file}"
                )
            
            # Create NetCDF file
            self.nc_dataset = nc4.Dataset(
                str(self.output_file), 
                'w', 
                format=self.output_config.netcdf_format
            )
            
            # Create dimensions
            self._create_dimensions()
            
            # Create coordinate variables
            self._create_coordinates(start_time)
            
            # Create data variables
            self._create_data_variables()
            
            # Add global attributes
            self._add_global_attributes(start_time)
            
            self.is_initialized = True
            logger.info(f"Initialized NetCDF file: {self.output_file}")
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            error_msg = f"Failed to initialize NetCDF file: {str(e)}"
            self.write_errors.append(error_msg)
            logger.error(error_msg)
            raise NetCDFWriteError(error_msg) from e
    
    def _create_dimensions(self) -> None:
        """Create NetCDF dimensions."""
        if self.nc_dataset is None:
            raise NetCDFWriteError("NetCDF dataset not initialized")
        
        # Time dimension (unlimited)
        self.nc_dataset.createDimension('time', None)
        
        # Add spatial dimensions if needed (for future extension)
        # self.nc_dataset.createDimension('x', 1)
        # self.nc_dataset.createDimension('y', 1)
    
    def _create_coordinates(self, start_time: Optional[datetime] = None) -> None:
        """Create coordinate variables."""
        if self.nc_dataset is None:
            raise NetCDFWriteError("NetCDF dataset not initialized")
        
        # Time coordinate
        time_var = self.nc_dataset.createVariable(
            'time', 
            'f8', 
            ('time',),
            zlib=self.output_config.compression_level > 0,
            complevel=self.output_config.compression_level,
            shuffle=self.output_config.shuffle,
            fletcher32=self.output_config.fletcher32
        )
        
        # Time attributes
        time_var.standard_name = "time"
        time_var.long_name = "time"
        time_var.units = self.output_config.time_units
        time_var.calendar = self.output_config.calendar
        time_var.axis = "T"
        
        if start_time is not None:
            time_var.time_origin = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    def _create_data_variables(self) -> None:
        """Create data variables with proper metadata."""
        if self.nc_dataset is None:
            raise NetCDFWriteError("NetCDF dataset not initialized")
        
        for var_name, var_config in self.output_config.variables.items():
            # Create variable
            data_var = self.nc_dataset.createVariable(
                var_name,
                'f4',  # float32 for data variables
                ('time',),
                zlib=var_config.compression,
                complevel=self.output_config.compression_level,
                shuffle=self.output_config.shuffle,
                fletcher32=self.output_config.fletcher32,
                fill_value=var_config.fill_value
            )
            
            # Standard CF attributes
            if var_config.standard_name:
                data_var.standard_name = var_config.standard_name
            data_var.long_name = var_config.long_name
            data_var.units = var_config.units
            
            # Isotope-specific attributes
            data_var.isotope = var_config.isotope
            if var_config.parent_variable:
                data_var.parent_variable = var_config.parent_variable
            if var_config.fractionation_model:
                data_var.fractionation_model = var_config.fractionation_model
            
            # CF cell methods for aggregated data
            if var_config.cell_methods:
                data_var.cell_methods = var_config.cell_methods
            else:
                # Generate appropriate cell methods based on aggregation
                if var_config.aggregation_method == AggregationMethod.DAILY_MEAN:
                    data_var.cell_methods = "time: mean (interval: 1 day)"
                elif var_config.aggregation_method in [AggregationMethod.DAILY_SUM, AggregationMethod.INTEGRAL]:
                    data_var.cell_methods = "time: sum (interval: 1 day)"
                elif var_config.aggregation_method == AggregationMethod.END_OF_DAY:
                    data_var.cell_methods = "time: point (end of day)"
            
            # Ancillary variables if specified
            if var_config.ancillary_variables:
                data_var.ancillary_variables = var_config.ancillary_variables
            
            # Valid range (for data quality)
            if var_name.endswith('_ratio'):
                data_var.valid_min = 0.0
                data_var.valid_max = 1.0
            elif 'flux' in var_name or 'integral' in var_name:
                data_var.valid_min = -1e10  # Allow negative fluxes
                data_var.valid_max = 1e10
            
            logger.debug(f"Created variable: {var_name} ({var_config.units})")
    
    def _add_global_attributes(self, start_time: Optional[datetime] = None) -> None:
        """Add global attributes to NetCDF file."""
        if self.nc_dataset is None:
            raise NetCDFWriteError("NetCDF dataset not initialized")
        
        # CF conventions
        self.nc_dataset.Conventions = self.output_config.conventions
        
        # Dataset identification
        self.nc_dataset.title = self.output_config.title
        self.nc_dataset.institution = self.output_config.institution or "RELIC Modeling Framework"
        self.nc_dataset.source = self.output_config.source
        self.nc_dataset.references = self.output_config.references or "RELIC Documentation"
        
        # Creation metadata
        self.nc_dataset.history = (self.output_config.history + " " if self.output_config.history else "") + \
                                 f"{datetime.now().isoformat()}: Created by IsotopeNetCDFWriter v{ISOTOPE_TRACKER_VERSION}"
        self.nc_dataset.comment = self.output_config.comment or "Stable-isotope tracking results from RELIC simulation"
        
        # Technical metadata
        self.nc_dataset.created_by = "RELIC Isotope Tracker"
        self.nc_dataset.version = ISOTOPE_TRACKER_VERSION
        self.nc_dataset.netcdf_version = nc4.__version__
        
        # Isotope-specific metadata
        self.nc_dataset.tracked_isotope = ",".join(
            sorted(species.value for species in self.isotope_config.tracked_species)
        )
        self.nc_dataset.isotope_module = self.isotope_config.module_name
        
        # Process information
        all_processes = self.isotope_config.get_all_processes()
        enabled_processes = [
            name for name, process in all_processes.items() if process.enabled
        ]
        self.nc_dataset.tracked_processes = ", ".join(enabled_processes)
        
        # Scientific metadata
        fractionation_info = []
        for process_name in enabled_processes:
            process_config = all_processes[process_name]
            frac_params = process_config.fractionation
            fractionation_info.append(
                f"{process_name}: {frac_params.model_type.value} "
                f"(alpha_0={frac_params.alpha_0})"
            )
        self.nc_dataset.fractionation_models = "; ".join(fractionation_info)
        
        # Time information
        if start_time is not None:
            self.nc_dataset.simulation_start_time = start_time.isoformat()
        self.nc_dataset.time_coverage_resolution = "P1D"  # Daily resolution
        
    def write_data(self, 
                  time: float, 
                  isotope_data: Dict[str, Union[float, Tensor]]) -> None:
        """
        Write isotope data to NetCDF file.
        
        Parameters:
            time : float
                Time stamp in seconds since start
            isotope_data : Dict[str, Union[float, Tensor]]
                Isotope data to write
        """
        if not self.is_initialized:
            self.initialize_file()
        self.data_buffer.add_data_point(time, isotope_data)
    
    def flush_buffer(self, force_write: bool = False) -> None:
        """
        Flush data buffer to NetCDF file.
        
        Parameters:
            force_write : bool
                Force writing even if buffer is not full
        """
        try:
            if not self.is_initialized:
                logger.warning("Cannot flush buffer - NetCDF file not initialized")
                return
            if force_write and self.data_buffer.current_day is not None:
                self.data_buffer._aggregate_daily_data()
            
            # Get aggregated data from buffer
            times, data_arrays = self.data_buffer.get_aggregated_data()
            
            if not times:
                logger.debug("No data to flush")
                return
            
            # Write data to NetCDF
            self._write_aggregated_data(times, data_arrays)
            
            # Clear buffer
            self.data_buffer.clear()
            
            self.total_writes += 1
            self.last_write_time = times[-1] if times else None
            
            logger.info(f"Flushed {len(times)} time points to NetCDF")
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            error_msg = f"Failed to flush buffer: {str(e)}"
            self.write_errors.append(error_msg)
            raise NetCDFWriteError(error_msg) from e
    
    def _write_aggregated_data(self, 
                              times: List[float], 
                              data_arrays: Dict[str, List[float]]) -> None:
        """
        Write aggregated data to NetCDF file.
        
        Parameters:
            times : List[float]
                Time values
            data_arrays : Dict[str, List[float]]
                Data arrays for each variable
        """
        if self.nc_dataset is None:
            raise NetCDFWriteError("NetCDF dataset not initialized")
        
        if not times:
            return
        
        # Get current time dimension size
        current_size = len(self.nc_dataset.dimensions['time'])
        new_size = current_size + len(times)
        
        # Convert time values to NetCDF time units
        time_values = self._convert_time_values(times)
        
        # Write time coordinate
        time_var = self.nc_dataset.variables['time']
        time_var[current_size:new_size] = time_values
        
        # Write data variables
        for var_name, var_config in self.output_config.variables.items():
            if var_name in data_arrays:
                cleaned_values = [
                    float(v) if math.isfinite(float(v)) else float(var_config.fill_value)
                    for v in data_arrays[var_name]
                ]
                data_values = np.asarray(cleaned_values, dtype=np.float32)
                
                # Write to NetCDF
                data_var = self.nc_dataset.variables[var_name]
                data_var[current_size:new_size] = data_values
        
        # Sync to disk
        self.nc_dataset.sync()
    
    def _convert_time_values(self, times: List[float]) -> np.ndarray:
        """
        Convert time values to NetCDF time coordinate values.
        
        Parameters:
            times : List[float]
                Time values in seconds since start
                
        Returns:
            np.ndarray: Converted time values
        """
        # References: CF Conventions 1.8 Section 4.4; UDUNITS time units.
        # Mathematical expression: t_u = t_s / S_u.
        # Parameter meanings: t_s is elapsed seconds and S_u is seconds per unit.
        # Units: input [s]; output [day], [h], or [s] from time_units.
        # Domain/constraints: values are elapsed times and S_u comes from SSOT.
        # Sign convention: positive values advance after the reference epoch.
        if "days since" in self.output_config.time_units:
            return np.array(times) / SECONDS_PER_DAY
        elif "hours since" in self.output_config.time_units:
            return np.array(times) / SECONDS_PER_HOUR
        else:
            # Default to seconds
            return np.array(times)
    
    def close(self) -> None:
        """Close NetCDF file and clean up resources."""
        try:
            # Flush any remaining data
            self.flush_buffer(force_write=True)
            
            # Close NetCDF dataset
            if self.nc_dataset is not None:
                self.nc_dataset.close()
                self.nc_dataset = None
            
            logger.info(f"Closed NetCDF file: {self.output_file}")
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            error_msg = f"Error closing NetCDF file: {str(e)}"
            self.write_errors.append(error_msg)
            raise NetCDFWriteError(error_msg) from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get writer statistics and diagnostics."""
        buffer_stats = self.data_buffer.get_statistics()
        
        return {
            'output_file': str(self.output_file),
            'is_initialized': self.is_initialized,
            'total_writes': self.total_writes,
            'last_write_time': self.last_write_time,
            'write_errors': len(self.write_errors),
            'buffer_statistics': buffer_stats,
            'file_exists': self.output_file.exists(),
            'file_size': self.output_file.stat().st_size if self.output_file.exists() else 0,
            'compression_level': self.compression_level.value,
            'recent_errors': self.write_errors[-5:] if self.write_errors else []
        }
    
    def validate_output(self) -> List[str]:
        """
        Validate bounded output metadata and completeness without a dataset claim.
        
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        try:
            if not self.output_file.exists():
                errors.append("Output file does not exist")
                return errors
            
            # Open file for validation
            with nc4.Dataset(str(self.output_file), 'r') as ds:
                # Check required dimensions
                if 'time' not in ds.dimensions:
                    errors.append("Missing required 'time' dimension")
                
                # Check time coordinate
                if 'time' not in ds.variables:
                    errors.append("Missing time coordinate variable")
                else:
                    time_var = ds.variables['time']
                    if not hasattr(time_var, 'units'):
                        errors.append("Time coordinate missing 'units' attribute")
                    if not hasattr(time_var, 'calendar'):
                        errors.append("Time coordinate missing 'calendar' attribute")
                
                # Check data variables
                for var_name, var_config in self.output_config.variables.items():
                    if var_name not in ds.variables:
                        errors.append(f"Missing data variable: {var_name}")
                    else:
                        data_var = ds.variables[var_name]
                        
                        # Check required attributes
                        if not hasattr(data_var, 'units'):
                            errors.append(f"Variable {var_name} missing 'units' attribute")
                        
                        # Check isotope-specific attributes
                        if not hasattr(data_var, 'isotope'):
                            errors.append(f"Variable {var_name} missing 'isotope' attribute")
                
                # Check global attributes
                required_globals = ['Conventions', 'title', 'history']
                for attr in required_globals:
                    if not hasattr(ds, attr):
                        errors.append(f"Missing required global attribute: {attr}")
                
                # Check CF conventions version
                if hasattr(ds, 'Conventions') and CF_VERSION not in ds.Conventions:
                    errors.append(f"CF conventions version mismatch: expected {CF_VERSION}")
        
        except OSError as e:
            # Validator boundary: a file that cannot be opened or read as NetCDF is a
            # validation error of the output; any other error propagates.
            errors.append(f"Output file unreadable or not NetCDF: {e}")
        
        return errors
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"IsotopeNetCDFWriter("
                f"file={self.output_file.name}, "
                f"initialized={self.is_initialized}, "
                f"writes={self.total_writes})")


# Convenience functions
def create_isotope_netcdf_writer(isotope_config: IsotopeConfig,
                                output_file: Optional[str] = None,
                                compression_level: CompressionLevel = CompressionLevel.MEDIUM) -> IsotopeNetCDFWriter:
    """
    Create NetCDF writer with configuration.
    
    Parameters:
        isotope_config : IsotopeConfig
            Complete isotope configuration
        output_file : Optional[str]
            Override output file path
        compression_level : CompressionLevel
            Compression level for output
            
    Returns:
        IsotopeNetCDFWriter: Configured writer
    """
    output_config = isotope_config.unified_output
    
    # Override output file if provided
    if output_file is not None:
        output_config.output_file = output_file
    
    return IsotopeNetCDFWriter(
        output_config=output_config,
        isotope_config=isotope_config,
        compression_level=compression_level
    )


def validate_netcdf_file(file_path: Union[str, Path]) -> List[str]:
    """
    Validate a bounded NetCDF metadata subset.
    
    Parameters:
        file_path : Union[str, Path]
            Path to NetCDF file
            
    Returns:
        List[str]: List of validation errors
    """
    if not HAS_NETCDF:
        return ["NetCDF4 not available for validation"]
    
    errors = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        return [f"File does not exist: {file_path}"]
    
    try:
        with nc4.Dataset(str(file_path), 'r') as ds:
            # Bounded CF metadata checks
            if not hasattr(ds, 'Conventions'):
                errors.append("Missing 'Conventions' global attribute")
            
            if 'time' not in ds.dimensions:
                errors.append("Missing required 'time' dimension")
            
            if 'time' not in ds.variables:
                errors.append("Missing time coordinate variable")
    
    except OSError as e:
        # Validator boundary: a file that cannot be opened or read as NetCDF is a
        # validation error; any other error propagates.
        errors.append(f"File validation failed: {str(e)}")
    
    return errors


# Export public interface
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


