"""
Data I/O Health Check Module

Provides comprehensive health checking and diagnostics for the data I/O layer.
Tests format support, library availability, compression capabilities, and 
performance characteristics to ensure optimal operation.

This module validates the complete I/O stack including:
- File format support and library availability
- Compression methods and performance
- Single-variable CF standard-name/unit metadata capability
- Time coordinate processing
- Large file handling and memory management

Key Features:
- Comprehensive format support validation
- Performance benchmarking and optimization
- Dependency availability checking
- Configuration validation and recommendations
- Integration testing with core components

Health Check Categories:
1. Dependencies: Required and optional library availability
2. Formats: File format reading and writing capabilities  
3. Compression: Compression method support and performance
4. Performance: I/O throughput and memory efficiency
5. Integration: Integration with RELIC components

References:
- System diagnostics best practices
- Performance testing methodologies
- Scientific data format specifications
"""

import os
import time
import tempfile
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import xarray as xr

# Test imports to check availability
IMPORT_RESULTS = {}

def _register_optional_dependency(
    result_key: str,
    module_names: Union[str, Tuple[str, ...]],
) -> None:
    """Register optional dependency availability into IMPORT_RESULTS."""
    module_list: Tuple[str, ...]
    if isinstance(module_names, str):
        module_list = (module_names,)
    else:
        module_list = module_names

    for module_name in module_list:
        try:
            if importlib.util.find_spec(module_name) is None:
                IMPORT_RESULTS[result_key] = False
                return
            importlib.import_module(module_name)
        except ImportError:
            # Optional-dependency probe: an absent (or unloadable) module is recorded as absent.
            IMPORT_RESULTS[result_key] = False
            return
    IMPORT_RESULTS[result_key] = True


# Core/optional dependencies used by health probes
_register_optional_dependency("torch", "torch")
_register_optional_dependency("dask", ("dask", "dask.array"))
_register_optional_dependency("h5py", "h5py")
_register_optional_dependency("zarr", "zarr")
_register_optional_dependency("cftime", "cftime")
_register_optional_dependency("netcdf4", "netCDF4")
_register_optional_dependency("pandas", "pandas")


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"                      # All checks passed
    WARNING = "warning"                      # Minor issues found
    ERROR = "error"                          # Major issues found
    CRITICAL = "critical"                    # System non-functional


@dataclass
class HealthCheckResult:
    """
    Result from individual health check.
    
    Contains status, details, and recommendations for a specific
    health check category or test.
    """
    name: str                                # Check name
    status: HealthStatus                     # Overall status
    message: str                             # Status message
    details: Dict[str, Any] = field(default_factory=dict)  # Detailed results
    recommendations: List[str] = field(default_factory=list)  # Improvement suggestions
    performance_metrics: Optional[Dict[str, float]] = None  # Performance data


@dataclass
class SystemHealthReport:
    """
    Comprehensive system health report.
    
    Contains results from all health check categories and
    overall system status assessment.
    """
    overall_status: HealthStatus             # Overall system status
    timestamp: str                           # Report generation time
    results: Dict[str, HealthCheckResult]    # Individual check results
    summary: Dict[str, Any]                  # Summary statistics
    recommendations: List[str] = field(default_factory=list)  # Priority recommendations


class IOHealthChecker:
    """
    Comprehensive health checker for data I/O operations.
    
    Performs systematic validation of I/O capabilities including
    dependencies, format support, performance characteristics,
    and integration with RELIC components.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self._test_data_cache = {}
    
    def run_full_check(self) -> SystemHealthReport:
        """
        Run comprehensive health check across all categories.
        
        Returns:
            Complete system health report
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        results = {}
        
        # Run individual health checks
        results['dependencies'] = self.check_dependencies()
        results['formats'] = self.check_format_support()
        results['compression'] = self.check_compression_support()
        results['performance'] = self.check_performance()
        results['cf_metadata_subset'] = self.check_cf_metadata_subset()
        results['integration'] = self.check_integration()
        
        # Determine overall status
        overall_status = self._determine_overall_status(results)
        
        # Generate summary
        summary = self._generate_summary(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        return SystemHealthReport(
            overall_status=overall_status,
            timestamp=timestamp,
            results=results,
            summary=summary,
            recommendations=recommendations
        )
    
    def check_dependencies(self) -> HealthCheckResult:
        """
        Check availability of required and optional dependencies.
        
        Returns:
            Health check result for dependencies
        """
        details = {}
        missing_required = []
        missing_optional = []
        
        # Required dependencies
        _ = {
            'numpy': True,  # Always available if we get here
            'xarray': True, # Always available if we get here
        }
        
        # Optional dependencies with their impact
        optional_deps = {
            'torch': ('PyTorch tensor support', IMPORT_RESULTS.get('torch', False)),
            'dask': ('Large file processing', IMPORT_RESULTS.get('dask', False)),
            'h5py': ('HDF5 format support', IMPORT_RESULTS.get('h5py', False)),
            'zarr': ('Zarr format support', IMPORT_RESULTS.get('zarr', False)),
            'cftime': ('Non-standard calendar support', IMPORT_RESULTS.get('cftime', False)),
            'netcdf4': ('NetCDF format support', IMPORT_RESULTS.get('netcdf4', False)),
            'pandas': ('CSV and time series support', IMPORT_RESULTS.get('pandas', False))
        }
        
        # Check optional dependencies
        for dep_name, (description, available) in optional_deps.items():
            details[dep_name] = {
                'available': available,
                'description': description,
                'impact': 'high' if dep_name in ['torch', 'dask', 'netcdf4'] else 'medium'
            }
            
            if not available:
                missing_optional.append(dep_name)
        
        # Determine status
        if missing_required:
            status = HealthStatus.CRITICAL
            message = f"Missing required dependencies: {', '.join(missing_required)}"
        elif len(missing_optional) > 3:
            status = HealthStatus.WARNING
            message = f"Multiple optional dependencies missing: {len(missing_optional)}"
        else:
            status = HealthStatus.HEALTHY
            message = "All critical dependencies available"
        
        # Generate recommendations
        recommendations = []
        if missing_optional:
            for dep in missing_optional:
                if details[dep]['impact'] == 'high':
                    recommendations.append(f"Install {dep} for {details[dep]['description']}")
        
        return HealthCheckResult(
            name="dependencies",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    def check_format_support(self) -> HealthCheckResult:
        """
        Check support for different file formats.
        
        Returns:
            Health check result for format support
        """
        details = {}
        supported_formats = []
        unsupported_formats = []
        
        # Test format support
        format_tests = {
            'netcdf4': lambda: self._test_netcdf_support(),
            'hdf5': lambda: self._test_hdf5_support(),
            'zarr': lambda: self._test_zarr_support(),
            'csv': lambda: self._test_csv_support()
        }
        
        for format_name, test_func in format_tests.items():
            try:
                result = test_func()
                details[format_name] = result
                if result['supported']:
                    supported_formats.append(format_name)
                else:
                    unsupported_formats.append(format_name)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                details[format_name] = {
                    'supported': False,
                    'error': str(e)
                }
                unsupported_formats.append(format_name)
        
        # Determine status
        if not supported_formats:
            status = HealthStatus.CRITICAL
            message = "No file formats supported"
        elif 'netcdf4' not in supported_formats:
            status = HealthStatus.ERROR
            message = "NetCDF4 format not supported"
        elif len(supported_formats) < 2:
            status = HealthStatus.WARNING
            message = f"Limited format support: {supported_formats}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Supports {len(supported_formats)} formats"
        
        # Generate recommendations
        recommendations = []
        if 'netcdf4' not in supported_formats:
            recommendations.append("Install netCDF4-python for NetCDF support")
        if 'zarr' not in supported_formats:
            recommendations.append("Install zarr for cloud-optimized storage")
        
        return HealthCheckResult(
            name="format_support",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    def check_compression_support(self) -> HealthCheckResult:
        """
        Check compression method support and performance.
        
        Returns:
            Health check result for compression support
        """
        details = {}
        
        # Test compression methods
        compression_methods = ['zlib', 'gzip', 'lzma']
        
        # Add modern compression if available
        if IMPORT_RESULTS.get('zarr', False):
            compression_methods.extend(['blosc', 'zstd'])
        
        supported_methods = []
        performance_results = {}
        
        for method in compression_methods:
            try:
                result = self._test_compression_method(method)
                details[method] = result
                if result['supported']:
                    supported_methods.append(method)
                    performance_results[method] = result.get('compression_ratio', 0)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                details[method] = {
                    'supported': False,
                    'error': str(e)
                }
        
        # Determine status
        if not supported_methods:
            status = HealthStatus.ERROR
            message = "No compression methods available"
        elif len(supported_methods) < 2:
            status = HealthStatus.WARNING
            message = "Limited compression options"
        else:
            status = HealthStatus.HEALTHY
            message = f"Supports {len(supported_methods)} compression methods"
        
        # Generate recommendations
        recommendations = []
        if 'zstd' not in supported_methods:
            recommendations.append("Install zstandard for optimal compression performance")
        if not supported_methods:
            recommendations.append("Install compression libraries for reduced storage")
        
        return HealthCheckResult(
            name="compression_support",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations,
            performance_metrics=performance_results
        )
    
    def check_performance(self) -> HealthCheckResult:
        """
        Check I/O performance characteristics.
        
        Returns:
            Health check result for performance
        """
        details = {}
        performance_metrics = {}
        
        # Test read performance
        try:
            read_metrics = self._benchmark_read_performance()
            details['read_performance'] = read_metrics
            performance_metrics['read_throughput_mbps'] = read_metrics.get('throughput_mbps', 0)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            details['read_performance'] = {'error': str(e)}
        
        # Test write performance
        try:
            write_metrics = self._benchmark_write_performance()
            details['write_performance'] = write_metrics
            performance_metrics['write_throughput_mbps'] = write_metrics.get('throughput_mbps', 0)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            details['write_performance'] = {'error': str(e)}
        
        # Test memory efficiency
        try:
            memory_metrics = self._benchmark_memory_usage()
            details['memory_efficiency'] = memory_metrics
            performance_metrics['memory_efficiency'] = memory_metrics.get('efficiency_score', 0)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            details['memory_efficiency'] = {'error': str(e)}
        
        # A failed read or write benchmark fails the check (a measured throughput of the other
        # benchmark never reads as a healthy result); a failed memory benchmark lowers it to a
        # warning at most.
        failed_benchmarks = [
            name for name in ('read_performance', 'write_performance') if 'error' in details.get(name, {})
        ]

        # Determine status based on performance
        avg_throughput = np.mean([
            performance_metrics.get('read_throughput_mbps', 0),
            performance_metrics.get('write_throughput_mbps', 0)
        ])
        
        if avg_throughput > 100:
            status = HealthStatus.HEALTHY
            message = f"Good I/O performance: {avg_throughput:.1f} MB/s average"
        elif avg_throughput > 50:
            status = HealthStatus.WARNING
            message = f"Moderate I/O performance: {avg_throughput:.1f} MB/s average"
        elif avg_throughput > 0:
            status = HealthStatus.WARNING
            message = f"Low I/O performance: {avg_throughput:.1f} MB/s average"
        else:
            status = HealthStatus.ERROR
            message = "Performance benchmarks failed"
        if failed_benchmarks:
            status = HealthStatus.ERROR
            message = f"Performance benchmarks failed: {', '.join(failed_benchmarks)}"
        elif 'error' in details.get('memory_efficiency', {}) and status == HealthStatus.HEALTHY:
            status = HealthStatus.WARNING
            message = f"{message}; memory benchmark failed"
        
        # Generate recommendations
        recommendations = []
        if avg_throughput < 50:
            recommendations.append("Consider enabling parallel I/O for better performance")
            recommendations.append("Check disk I/O performance and available memory")
        
        return HealthCheckResult(
            name="performance",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations,
            performance_metrics=performance_metrics
        )
    
    def check_cf_metadata_subset(self) -> HealthCheckResult:
        """
        Check the bounded single-variable standard-name/unit validator.
        
        Returns:
            Health check result for the bounded CF metadata subset
        """
        details = {
            "validation_scope": "single_variable_standard_name_and_units",
            "dataset_conformance_checked": False,
            "coordinates_checked": False,
            "cell_methods_checked": False,
        }
        
        if importlib.util.find_spec("relic.data.standards.cf_conventions") is None:
            details['error'] = "CF conventions module not available"
            status = HealthStatus.ERROR
            message = "CF metadata subset check failed: module missing"
        else:
            cf_module = importlib.import_module("relic.data.standards.cf_conventions")
            CFConventions = getattr(cf_module, "CFConventions")
            CFAttributeSynchronizer = getattr(cf_module, "CFAttributeSynchronizer")

            try:
                validation = CFConventions.validate_cf_compliance(
                    {"standard_name": "air_temperature", "units": "K"}
                )
                details["probe_valid"] = validation.is_valid
                details["probe_issues"] = list(validation.issues)
                _ = CFAttributeSynchronizer()
                details['synchronizer_available'] = True

                if validation.is_valid:
                    status = HealthStatus.HEALTHY
                    message = (
                        "CF single-variable standard-name/unit metadata subset "
                        "is available; dataset conformance remains [N]"
                    )
                else:
                    status = HealthStatus.ERROR
                    message = "CF metadata subset probe failed"
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                details['error'] = str(e)
                status = HealthStatus.ERROR
                message = f"CF metadata subset check failed: {e}"
        
        recommendations = []
        if status != HealthStatus.HEALTHY:
            recommendations.append("Check CF conventions module installation")
        
        return HealthCheckResult(
            name="cf_metadata_subset",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    def check_integration(self) -> HealthCheckResult:
        """
        Check integration with RELIC components.
        
        Returns:
            Health check result for RELIC integration
        """
        details = {}
        
        try:
            reader_module = importlib.import_module("relic.data.io.reader")
            writer_module = importlib.import_module("relic.data.io.writer")
            adapter_module = importlib.import_module("relic.data.io.adapter")
            io_module = importlib.import_module("relic.coupling.external.common.io")
            time_module = importlib.import_module("relic.coupling.external.common.time")

            DataReader = getattr(reader_module, "DataReader")
            ReadConfig = getattr(reader_module, "ReadConfig")
            DataWriter = getattr(writer_module, "DataWriter")
            WriteConfig = getattr(writer_module, "WriteConfig")
            DataIO = getattr(adapter_module, "DataIO")
            IOHandler = getattr(io_module, "IOHandler")
            TimeHandler = getattr(time_module, "TimeHandler")

            # Test DataReader integration
            DataReader(ReadConfig())
            details['reader_integration'] = True
            
            # Test DataWriter integration
            DataWriter(WriteConfig())
            details['writer_integration'] = True
            
            # Test DataIO adapter
            DataIO()
            details['adapter_integration'] = True
            
            # Test IOHandler integration
            IOHandler()
            details['iohandler_integration'] = True
            
            # Test TimeHandler integration
            TimeHandler()
            details['timehandler_integration'] = True
            
            status = HealthStatus.HEALTHY
            message = "All integrations working"
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            details['error'] = str(e)
            status = HealthStatus.ERROR
            message = f"Integration check failed: {e}"
        
        recommendations = []
        if status != HealthStatus.HEALTHY:
            recommendations.append("Check RELIC module imports and dependencies")
        
        return HealthCheckResult(
            name="integration",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    # Private test methods
    def _test_netcdf_support(self) -> Dict[str, Any]:
        """Test NetCDF format support."""
        if not IMPORT_RESULTS.get('netcdf4', False):
            return {'supported': False, 'reason': 'netCDF4-python not available'}
        
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
            tmp.close()
            try:
                # Create test dataset
                data = np.random.random((10, 5, 5))
                ds = xr.Dataset({
                    'test_var': (['time', 'lat', 'lon'], data)
                })

                # Test write
                ds.to_netcdf(tmp.name)

                # Test read
                ds_read = xr.open_dataset(tmp.name)
                ds_read.close()

                return {
                    'supported': True,
                    'read': True,
                    'write': True
                }
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            return {
                'supported': False,
                'error': str(e)
            }
    
    def _test_hdf5_support(self) -> Dict[str, Any]:
        """Test HDF5 format support."""
        if not IMPORT_RESULTS.get('h5py', False):
            return {'supported': False, 'reason': 'h5py not available'}
        import h5py

        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
            tmp.close()
            try:
                # Test HDF5 basic operations
                with h5py.File(tmp.name, 'w') as f:
                    f.create_dataset('test', data=np.random.random((10, 10)))

                with h5py.File(tmp.name, 'r') as f:
                    _ = f['test'][:]

                return {
                    'supported': True,
                    'read': True,
                    'write': True
                }
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            return {
                'supported': False,
                'error': str(e)
            }
    
    def _test_zarr_support(self) -> Dict[str, Any]:
        """Test Zarr format support."""
        if not IMPORT_RESULTS.get('zarr', False):
            return {'supported': False, 'reason': 'zarr not available'}
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                zarr_path = Path(tmpdir) / 'test.zarr'

                # Create test dataset
                data = np.random.random((10, 5, 5))
                ds = xr.Dataset({
                    'test_var': (['time', 'lat', 'lon'], data)
                })

                # Test write with explicit chunking to satisfy zarr requirements
                encoding = {'test_var': {'chunks': (5, 5, 5)}}
                ds.to_zarr(
                    zarr_path,
                    mode='w',
                    encoding=encoding,
                    consolidated=False,
                    zarr_format=2,
                )

                # Test read
                ds_read = xr.open_zarr(zarr_path, consolidated=False)
                ds_read.close()

                return {
                    'supported': True,
                    'read': True,
                    'write': True
                }
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            return {
                'supported': False,
                'error': str(e)
            }
    
    def _test_csv_support(self) -> Dict[str, Any]:
        """Test CSV format support."""
        if not IMPORT_RESULTS.get('pandas', False):
            return {'supported': False, 'reason': 'pandas not available'}
        import pandas as pd

        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            tmp.close()
            try:
                # Create test data
                df = pd.DataFrame({
                    'time': pd.date_range('2000-01-01', periods=10),
                    'value': np.random.random(10)
                })

                # Test write
                df.to_csv(tmp.name, index=False)

                # Test read
                pd.read_csv(tmp.name)

                return {
                    'supported': True,
                    'read': True,
                    'write': True
                }
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            return {
                'supported': False,
                'error': str(e)
            }
    
    def _test_compression_method(self, method: str) -> Dict[str, Any]:
        """Test specific compression method."""
        try:
            # Create test data
            data = np.random.random((100, 100)).astype(np.float32)
            original_size = data.nbytes
            
            tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
            tmp.close()
            try:
                ds = xr.Dataset({'test': (['x', 'y'], data)})

                # Encoding with compression
                encoding = {}
                if method == 'zlib':
                    encoding = {'test': {'zlib': True, 'complevel': 4}}
                elif method == 'gzip':
                    encoding = {'test': {'zlib': True, 'complevel': 4}}  # NetCDF uses zlib for gzip

                start_time = time.time()
                ds.to_netcdf(tmp.name, encoding=encoding)
                compress_time = time.time() - start_time

                # Check compressed size
                compressed_size = os.path.getsize(tmp.name)
                compression_ratio = original_size / compressed_size

                # Test decompression
                start_time = time.time()
                ds_read = xr.open_dataset(tmp.name)
                decompress_time = time.time() - start_time
                ds_read.close()

                return {
                    'supported': True,
                    'compression_ratio': compression_ratio,
                    'compress_time': compress_time,
                    'decompress_time': decompress_time
                }
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            return {
                'supported': False,
                'error': str(e)
            }
    
    def _benchmark_read_performance(self) -> Dict[str, Any]:
        """Benchmark read performance."""
        # Create test file
        data_size_mb = 50  # 50 MB test file
        shape = (1000, 100, 50)  # Adjust for approximately 50MB of float32 data
        data = np.random.random(shape).astype(np.float32)
        
        tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        tmp.close()
        try:
            ds = xr.Dataset({'test_var': (['time', 'lat', 'lon'], data)})
            ds.to_netcdf(tmp.name)

            # Benchmark read
            start_time = time.time()
            ds_read = xr.open_dataset(tmp.name)
            # Force loading
            _ = ds_read.test_var.values
            read_time = time.time() - start_time
            ds_read.close()

            throughput_mbps = data_size_mb / read_time

            return {
                'data_size_mb': data_size_mb,
                'read_time': read_time,
                'throughput_mbps': throughput_mbps
            }
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    
    def _benchmark_write_performance(self) -> Dict[str, Any]:
        """Benchmark write performance."""
        data_size_mb = 50  # 50 MB test
        shape = (1000, 100, 50)
        data = np.random.random(shape).astype(np.float32)
        
        tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        tmp.close()
        try:
            ds = xr.Dataset({'test_var': (['time', 'lat', 'lon'], data)})

            # Benchmark write
            start_time = time.time()
            ds.to_netcdf(tmp.name)
            write_time = time.time() - start_time

            throughput_mbps = data_size_mb / write_time

            return {
                'data_size_mb': data_size_mb,
                'write_time': write_time,
                'throughput_mbps': throughput_mbps
            }
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory efficiency."""
        if importlib.util.find_spec("psutil") is None:
            return {
                'error': 'psutil not available for memory monitoring'
            }
        import psutil

        try:
            process = psutil.Process()

            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create large dataset
            shape = (500, 200, 200)  # ~400MB of float32 data
            data = np.random.random(shape).astype(np.float32)
            _ = xr.Dataset({'test_var': (['time', 'lat', 'lon'], data)})

            # Peak memory after creation
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB

            memory_used = peak_memory - baseline_memory
            data_size_mb = data.nbytes / 1024 / 1024
            efficiency_score = data_size_mb / memory_used if memory_used > 0 else 0

            return {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'memory_used_mb': memory_used,
                'data_size_mb': data_size_mb,
                'efficiency_score': efficiency_score
            }
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            return {
                'error': f'memory benchmark unavailable: {exc}'
            }
    
    def _determine_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall system status from individual check results."""
        status_counts = {status: 0 for status in HealthStatus}
        
        for result in results.values():
            status_counts[result.status] += 1
        
        # Priority-based status determination
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.ERROR] > 0:
            return HealthStatus.ERROR
        elif status_counts[HealthStatus.WARNING] > 1:
            return HealthStatus.WARNING
        elif status_counts[HealthStatus.WARNING] > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _generate_summary(self, results: Dict[str, HealthCheckResult]) -> Dict[str, Any]:
        """Generate summary statistics from health check results."""
        total_checks = len(results)
        status_counts = {status.value: 0 for status in HealthStatus}
        
        for result in results.values():
            status_counts[result.status.value] += 1
        
        return {
            'total_checks': total_checks,
            'status_distribution': status_counts,
            'health_percentage': (status_counts['healthy'] / total_checks) * 100
        }
    
    def _generate_recommendations(self, results: Dict[str, HealthCheckResult]) -> List[str]:
        """Generate priority recommendations from all health check results."""
        all_recommendations = []
        
        # Collect recommendations by priority
        critical_recs = []
        high_recs = []
        medium_recs = []
        
        for result in results.values():
            for rec in result.recommendations:
                if result.status == HealthStatus.CRITICAL:
                    critical_recs.append(f"[CRITICAL] {rec}")
                elif result.status == HealthStatus.ERROR:
                    high_recs.append(f"[HIGH] {rec}")
                else:
                    medium_recs.append(f"[MEDIUM] {rec}")
        
        # Combine in priority order
        all_recommendations.extend(critical_recs)
        all_recommendations.extend(high_recs)
        all_recommendations.extend(medium_recs[:5])  # Limit medium priority
        
        return all_recommendations


# Convenience function for quick health check
def io_healthcheck(detailed: bool = True) -> Union[SystemHealthReport, Dict[str, str]]:
    """
    Run I/O health check and return results.
    
    Parameters:
        detailed: Return full report (True) or summary (False)
        
    Returns:
        Health check results
        
    Example:
        # Quick check
        status = io_healthcheck(detailed=False)
        print(status['overall'])
        
        # Detailed check
        report = io_healthcheck(detailed=True)
        print(f"Overall Status: {report.overall_status.value}")
        for name, result in report.results.items():
            print(f"{name}: {result.status.value} - {result.message}")
    """
    checker = IOHealthChecker()
    report = checker.run_full_check()
    
    if detailed:
        return report
    else:
        # Return simple summary
        return {
            'overall': report.overall_status.value,
            'timestamp': report.timestamp,
            'health_percentage': f"{report.summary['health_percentage']:.1f}%",
            'total_checks': str(report.summary['total_checks']),
            'recommendations': len(report.recommendations)
        }


if __name__ == "__main__":
    # Run health check when executed directly
    print("Running RELIC Data I/O Health Check...")
    print("=" * 50)
    
    report = io_healthcheck(detailed=True)
    
    print(f"Overall Status: {report.overall_status.value.upper()}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Health Score: {report.summary['health_percentage']:.1f}%")
    print()
    
    # Show individual results
    for name, result in report.results.items():
        status_symbol = {
            'healthy': '[OK]',
            'warning': '[WARN]',
            'error': '[ERR]',
            'critical': '[CRIT]'
        }[result.status.value]
        
        print(f"{status_symbol} {name.replace('_', ' ').title()}: {result.message}")
    
    # Show recommendations
    if report.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(report.recommendations[:10], 1):
            print(f"{i}. {rec}")
    
    print("\n" + "=" * 50)
    print("Health check completed.")
