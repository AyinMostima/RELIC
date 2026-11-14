"""
I/O Operations for External Model Coupling

This module provides unified I/O capabilities for reading diverse external
model formats and writing standardized outputs. Handles NetCDF, HDF5, and
other scientific data formats with optimal performance and memory efficiency.

Key Features:
- Lazy loading and chunking for large datasets
- Automatic format detection and CF declaration inspection
- Parallel I/O operations and caching strategies
- Data compression and format conversion utilities
- Integration with xarray, dask, and cloud storage

Mathematical Framework:
I/O operations preserve data integrity through:
- Checksums and data validation
- Metadata preservation and enhancement
- Coordinate system normalization
- Unit and calendar standardization

Performance Optimization:
- Chunking strategy: balance I/O vs memory usage
- Parallel reads: concurrent file access
- Caching: store frequently accessed metadata
- Compression: minimize storage and transfer

References:
- NetCDF Best Practices: https://www.unidata.ucar.edu/software/netcdf/docs/BestPractices.html
- xarray I/O: http://xarray.pydata.org/en/stable/io.html  
- Dask Arrays: https://docs.dask.org/en/latest/array.html
- CF Conventions: http://cfconventions.org/cf-conventions/cf-conventions.html
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import time

import xarray as xr

from relic.data.standards.cf_conventions import CFConventions
import pandas as pd
import dask
import h5py

logger = logging.getLogger(__name__)

from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace

HAS_DASK = True
HAS_H5PY = True
HAS_ZARR = True


CF_DECLARED_SCOPE = "declared_dataset_scope"


class DataFormat(Enum):
    """Supported external data formats."""
    NETCDF4 = "netcdf4"                 # NetCDF-4/HDF5
    NETCDF3 = "netcdf3"                 # Classic NetCDF
    HDF5 = "hdf5"                       # HDF5
    ZARR = "zarr"                       # Zarr (cloud-optimized)
    GRIB = "grib"                       # GRIB meteorological format
    CSV = "csv"                         # CSV files
    PARQUET = "parquet"                 # Apache Parquet
    AUTO = "auto"                       # Automatic detection


class CompressionType(Enum):
    """Data compression options."""
    NONE = "none"                       # No compression
    ZLIB = "zlib"                       # zlib compression
    GZIP = "gzip"                       # gzip compression  
    LZMA = "lzma"                       # LZMA compression
    BLOSC = "blosc"                     # Blosc compression (fast)
    ZSTD = "zstd"                       # Zstandard compression
    AUTO = "auto"                       # Automatic selection


@dataclass
class IOConfig:
    """
    Configuration for I/O operations.
    
    Controls performance, caching, compression, and format-specific
    options for reading and writing external model data.
    """
    # Performance settings
    chunks: Optional[Dict[str, int]] = None     # Chunking strategy
    parallel: Optional[bool] = None             # Enable parallel I/O
    cache_size: Optional[int] = None            # Cache size in MB  
    max_memory: Optional[int] = None            # Max memory usage in MB
    
    # Format options
    format: DataFormat = DataFormat.AUTO       # Data format
    decode_times: bool = True                  # Defer CF time decoding when canonical processing owns it
    compression: CompressionType = CompressionType.AUTO
    compression_level: int = 4                  # Compression level (1-9)
    
    # Quality control
    validate_checksums: Optional[bool] = None   # Verify data checksums
    strict_cf: Optional[bool] = None            # Strict bounded metadata checks
    fill_missing: Optional[bool] = None         # Fill missing coordinate data
    missing_data_warning_threshold: Optional[float] = None  # Warning threshold for missing data fraction
    
    # Caching
    enable_cache: Optional[bool] = None         # Enable metadata caching
    cache_dir: Optional[Path] = None            # Cache directory
    
    # Cloud storage (if supported)
    s3_endpoint: Optional[str] = None           # S3 endpoint URL
    s3_credentials: Optional[Dict] = None       # S3 access credentials


@dataclass 
class IOResult:
    """
    Results from I/O operations.
    
    Contains loaded data, metadata, performance metrics, and
    quality assessment information.
    """
    data: xr.Dataset                            # Loaded dataset
    format_detected: DataFormat                 # Detected/used format
    file_size: int                              # File size in bytes
    load_time: float                            # Loading time in seconds
    
    # Quality metrics
    cf_compliant: bool = False                  # True only after the declared-scope check passed
    cf_validation_scope: str = "not_evaluated"  # "declared_dataset_scope" when the check ran
    dataset_conformance_checked: bool = False
    cf_conventions_declared: bool = False
    cf_conformance_issues: Tuple[str, ...] = ()
    missing_data_fraction: float = 0.0          # Fraction of missing data
    checksum_verified: bool = True              # Checksum verification
    
    # Performance metrics
    chunk_info: Dict[str, Any] = field(default_factory=dict)  # Chunking details
    memory_usage: Optional[float] = None        # Peak memory usage (MB)
    
    # Diagnostics
    warnings: List[str] = field(default_factory=list)  # Quality warnings
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __post_init__(self) -> None:
        if not isinstance(self.cf_compliant, bool):
            raise TypeError("cf_compliant must be bool")
        checked = self.dataset_conformance_checked is True
        declared_scope = self.cf_validation_scope == CF_DECLARED_SCOPE
        if checked != declared_scope:
            raise ValueError(
                "dataset_conformance_checked is true exactly when cf_validation_scope "
                f"is {CF_DECLARED_SCOPE!r}"
            )
        if self.cf_compliant and not checked:
            raise ValueError("IOResult cannot claim dataset-level CF conformance without the declared-scope check")
        if self.cf_compliant and self.cf_conformance_issues:
            raise ValueError("IOResult cannot claim CF conformance while conformance issues are recorded")


class IOHandler:
    """
    Unified I/O handler for external model data.
    
    Provides high-performance, format-agnostic data loading with
    automatic optimization, quality control, and bounded CF declaration
    inspection for diverse external model outputs.
    """
    
    def __init__(
        self,
        config: Optional[IOConfig] = None,
        parameter_manager=None,
        parameter_manager_resolver: Optional[Callable[[], Any]] = None,
    ):
        """
        Initialize I/O handler with configuration.
        
        Parameters:
            config: I/O configuration options
        """
        self._parameter_manager = parameter_manager
        self._parameter_manager_resolver: Optional[Callable[[], Any]] = parameter_manager_resolver
        self._io_defaults_source: str = "parameter_manager"
        self._io_defaults = self._load_io_parameters()
        self.config = config or IOConfig()
        self._apply_parameter_defaults()
        self.missing_data_warning_threshold = (
            self.config.missing_data_warning_threshold
            if self.config.missing_data_warning_threshold is not None
            else self._io_defaults["missing_data_warning_threshold"]
        )
        self._format_handlers = self._initialize_handlers()
        self._cache = {}
        
        # Set up dask if available
        if HAS_DASK and self.config.parallel:
            self._setup_dask()

    def _get_parameter_manager(self):
        """Resolve ParameterManager lazily to avoid import-time side effects."""
        if self._parameter_manager is None and self._parameter_manager_resolver is not None:
            self._parameter_manager = self._parameter_manager_resolver()
        if self._parameter_manager is None:
            self._parameter_manager = get_global_manager()
        return self._parameter_manager

    @property
    def io_defaults_source(self) -> str:
        """Return the source used to populate I/O defaults."""
        return self._io_defaults_source

    def _load_io_parameters(self) -> Dict[str, Any]:
        """Load I/O defaults from ParameterManager."""
        pm = self._get_parameter_manager()
        if pm is None:
            raise RuntimeError("IOHandler requires ParameterManager for IO defaults.")

        def _fetch(name: str, cast_to: Callable, units: str = "dimensionless") -> Any:
            try:
                value = pm.get(
                    name,
                    namespace=resolve_namespace(ParameterNamespace.COUPLING_EXTERNAL_IO),
                    units=units,
                )
            except KeyError as exc:
                raise KeyError(f"Missing required parameter coupling.external.io.{name}") from exc
            if value is None:
                raise KeyError(f"Missing required parameter coupling.external.io.{name}")
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                value = value.detach().cpu()
                if hasattr(value, "item"):
                    value = value.item()
            try:
                return cast_to(value)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001 - explicit audit
                raise TypeError(
                    f"Parameter coupling.external.io.{name} must be convertible to {cast_to.__name__}: {exc}"
                ) from exc

        self._io_defaults_source = "parameter_manager"
        return {
            "missing_data_warning_threshold": float(
                _fetch("missing_data_warning_threshold", float)
            ),
            "validate_checksums": bool(_fetch("validate_checksums", bool)),
            "cache_size_mb": int(_fetch("cache_size_mb", int)),
            "enable_cache": bool(_fetch("enable_cache", bool)),
            "parallel_io": bool(_fetch("parallel_io", bool)),
            "fill_missing": bool(_fetch("fill_missing_coordinates", bool)),
            "strict_cf": bool(_fetch("strict_cf", bool)),
        }

    def _apply_parameter_defaults(self) -> None:
        """Apply ParameterManager defaults to IOConfig."""
        cfg = self.config
        defaults = self._io_defaults
        if cfg.cache_size is None:
            cfg.cache_size = defaults["cache_size_mb"]
        if cfg.validate_checksums is None:
            cfg.validate_checksums = defaults["validate_checksums"]
        if cfg.enable_cache is None:
            cfg.enable_cache = defaults["enable_cache"]
        if cfg.parallel is None:
            cfg.parallel = defaults["parallel_io"]
        if cfg.fill_missing is None:
            cfg.fill_missing = defaults["fill_missing"]
        if cfg.strict_cf is None:
            cfg.strict_cf = defaults["strict_cf"]
        if cfg.missing_data_warning_threshold is None:
            cfg.missing_data_warning_threshold = defaults["missing_data_warning_threshold"]

    
    def _initialize_handlers(self) -> Dict[DataFormat, Callable]:
        """
        Initialize format-specific handlers.
        
        Returns:
            Dictionary mapping formats to handler functions
        """
        handlers = {
            DataFormat.NETCDF4: self._read_netcdf,
            DataFormat.NETCDF3: self._read_netcdf,
            DataFormat.CSV: self._read_csv,
        }
        
        if HAS_H5PY:
            handlers[DataFormat.HDF5] = self._read_hdf5
        
        if HAS_ZARR:
            handlers[DataFormat.ZARR] = self._read_zarr
        
        return handlers
    
    def _setup_dask(self) -> None:
        """Configure dask for parallel processing."""
        if HAS_DASK:
            # Configure dask for optimal I/O
            dask.config.set({
                'array.chunk-size': f"{self.config.cache_size}MiB",
                'array.slicing.split_large_chunks': True
            })
    
    def read(self, source: Union[str, Path, List[Union[str, Path]]]) -> IOResult:
        """
        Read external model data from file(s).
        
        Parameters:
            source: File path(s) to read
            
        Returns:
            I/O result with loaded data and metadata
            
        Raises:
            IOError: If file cannot be read
            ValueError: If format is not supported
        """
        start_time = time.time()
        
        # Handle multiple files
        if isinstance(source, (list, tuple)):
            return self._read_multiple_files(source)
        
        source_path = Path(source)
        
        if not source_path.exists():
            raise IOError(f"File not found: {source_path}")
        
        # Detect format
        format_detected = self._detect_format(source_path)
        
        if format_detected not in self._format_handlers:
            raise ValueError(f"Unsupported format: {format_detected}")
        
        # Check cache first
        cache_key = self._get_cache_key(source_path)
        if self.config.enable_cache and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            if self._is_cache_valid(cached_result, source_path):
                return cached_result
        
        # Read data using appropriate handler
        handler = self._format_handlers[format_detected]
        dataset = handler(source_path)
        
        # Post-process dataset
        processed_dataset = self._postprocess_dataset(dataset)
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(processed_dataset, source_path)
        
        elapsed_time = time.time() - start_time
        file_size = source_path.stat().st_size
        
        # Create result
        result = IOResult(
            data=processed_dataset,
            format_detected=format_detected,
            file_size=file_size,
            load_time=elapsed_time,
            **quality_metrics
        )
        
        # Cache result
        if self.config.enable_cache:
            self._cache[cache_key] = result
        
        return result
    
    def _detect_format(self, file_path: Path) -> DataFormat:
        """
        Detect data format from file extension and content.
        
        Parameters:
            file_path: Path to file
            
        Returns:
            Detected data format
        """
        if self.config.format != DataFormat.AUTO:
            return self.config.format
        
        suffix = file_path.suffix.lower()
        
        # Format detection by extension
        format_map = {
            '.nc': DataFormat.NETCDF4,
            '.nc4': DataFormat.NETCDF4, 
            '.netcdf': DataFormat.NETCDF4,
            '.h5': DataFormat.HDF5,
            '.hdf5': DataFormat.HDF5,
            '.hdf': DataFormat.HDF5,
            '.zarr': DataFormat.ZARR,
            '.csv': DataFormat.CSV,
            '.parquet': DataFormat.PARQUET,
            '.grib': DataFormat.GRIB,
            '.grb': DataFormat.GRIB,
            '.grib2': DataFormat.GRIB
        }
        
        if suffix in format_map:
            return format_map[suffix]
        
        # Content-based detection for ambiguous cases
        try:
            # Try NetCDF first (most common)
            with xr.open_dataset(file_path, engine='netcdf4') as _:
                return DataFormat.NETCDF4
        except (OSError, ValueError):
            # Declared meaning: the netCDF4 backend refuses a file of another format.
            pass
        
        if HAS_H5PY:
            try:
                with h5py.File(file_path, 'r') as _:
                    return DataFormat.HDF5
            except OSError:
                # Declared meaning: h5py refuses a file that is not HDF5.
                pass
        
        # Default error path
        raise ValueError(f"Could not determine format for file: {file_path}")
    
    def _read_netcdf(self, file_path: Path) -> xr.Dataset:
        """
        Read NetCDF file with optimized settings.
        
        Parameters:
            file_path: Path to NetCDF file
            
        Returns:
            xarray Dataset
        """
        if self.config.enable_cache is False and self.config.chunks is None:
            return xr.load_dataset(
                file_path,
                decode_times=self.config.decode_times,
                decode_coords='all'
            )
        open_kwargs = {
            'chunks': self.config.chunks,
            'cache': self.config.enable_cache,
            'decode_times': self.config.decode_times,
            'decode_coords': 'all'
        }
        
        # Remove None values
        open_kwargs = {k: v for k, v in open_kwargs.items() if v is not None}
        
        return xr.open_dataset(file_path, **open_kwargs)
    
    def _read_hdf5(self, file_path: Path) -> xr.Dataset:
        """
        Read HDF5 file using xarray.
        
        Parameters:
            file_path: Path to HDF5 file
            
        Returns:
            xarray Dataset
        """
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 support")
        
        return xr.open_dataset(
            file_path,
            engine='h5netcdf',
            chunks=self.config.chunks,
            decode_times=self.config.decode_times,
        )
    
    def _read_zarr(self, file_path: Path) -> xr.Dataset:
        """
        Read Zarr format data.
        
        Parameters:
            file_path: Path to Zarr store
            
        Returns:
            xarray Dataset  
        """
        if not HAS_ZARR:
            raise ImportError("zarr required for Zarr support")
        
        return xr.open_zarr(
            file_path,
            chunks=self.config.chunks,
            decode_times=self.config.decode_times,
        )
    
    def _read_csv(self, file_path: Path) -> xr.Dataset:
        """
        Read CSV file and convert to xarray Dataset.
        
        Parameters:
            file_path: Path to CSV file
            
        Returns:
            xarray Dataset
        """
        # Read with pandas first
        df = pd.read_csv(file_path, low_memory=False)
        
        # Convert to xarray Dataset
        # This is a simplified conversion - would need enhancement for complex cases
        return df.to_xarray()
    
    def _read_multiple_files(self, file_paths: List[Union[str, Path]]) -> IOResult:
        """
        Read and concatenate multiple files.
        
        Parameters:
            file_paths: List of file paths
            
        Returns:
            Combined I/O result
        """
        if not file_paths:
            raise ValueError("No files provided")
        
        start_time = time.time()
        datasets = []
        total_size = 0
        all_warnings = []
        
        for file_path in file_paths:
            result = self.read(file_path)
            datasets.append(result.data)
            total_size += result.file_size
            all_warnings.extend(result.warnings)
        
        # Concatenate the files along time; files that cannot be concatenated raise (they are never
        # merged instead, which would pair records that do not belong together).
        combined_dataset = xr.concat(datasets, dim='time')
        
        elapsed_time = time.time() - start_time
        
        return IOResult(
            data=combined_dataset,
            format_detected=DataFormat.NETCDF4,  # Assume NetCDF for combined
            file_size=total_size,
            load_time=elapsed_time,
            warnings=all_warnings
        )
    
    def _postprocess_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Post-process dataset for bounded metadata handling and optimization.
        
        Parameters:
            dataset: Raw dataset
            
        Returns:
            Processed dataset
        """
        processed = dataset.copy()
        
        # Decode CF coordinates if not already done
        if not hasattr(processed, '_CF_decoded'):
            processed = xr.decode_cf(
                processed,
                decode_times=self.config.decode_times,
            )
        
        # Fill missing coordinate data if requested
        if self.config.fill_missing:
            processed = self._fill_missing_coordinates(processed)
        
        # Optimize chunking for performance
        if HAS_DASK and self.config.chunks:
            processed = self._optimize_chunks(processed)
        
        return processed
    
    def _fill_missing_coordinates(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Fill missing coordinate information.
        
        Parameters:
            dataset: Dataset with potentially missing coordinates
            
        Returns:
            Dataset with completed coordinates
        """
        # This is a placeholder for coordinate completion logic
        # Would include operations like:
        # - Inferring time bounds from time coordinates
        # - Computing grid cell areas and weights  
        # - Adding standard CF coordinate attributes
        return dataset
    
    def _optimize_chunks(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Optimize data chunking for performance.
        
        Parameters:
            dataset: Dataset to rechunk
            
        Returns:
            Rechunked dataset
        """
        if not HAS_DASK:
            return dataset
        
        # Auto-chunk strategy based on available memory
        _ = self.config.cache_size * 1024 * 1024  # Convert MB to bytes
        
        rechunked_vars = {}
        for var_name, data_var in dataset.data_vars.items():
            if hasattr(data_var.data, 'chunks'):
                # Already chunked - optimize existing chunks
                rechunked_vars[var_name] = data_var.chunk('auto')
            else:
                # Not chunked - apply chunking
                rechunked_vars[var_name] = data_var.chunk(self.config.chunks)
        
        return dataset.assign(rechunked_vars)
    
    def _compute_quality_metrics(self, dataset: xr.Dataset, file_path: Path) -> Dict[str, Any]:
        """
        Compute data quality metrics.
        
        Parameters:
            dataset: Dataset to analyze  
            file_path: Source file path
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        warnings_list = []
        
        cf_declared = self._inspect_cf_declaration(dataset)
        # Bounded dataset-level conformance for the declared CF scope
        # (Conventions, units, standard names, coordinates, time and
        # calendar, bounds, missing data, cell methods); the owner of the
        # rules is CFConventions and the scope name is recorded verbatim.
        conformance = CFConventions.validate_dataset_conformance(dataset)
        metrics['cf_validation_scope'] = conformance.scope
        metrics['dataset_conformance_checked'] = True
        metrics['cf_conventions_declared'] = cf_declared
        metrics['cf_compliant'] = bool(conformance.is_valid)
        metrics['cf_conformance_issues'] = tuple(conformance.issues)
        warnings_list.extend(f"CF conformance: {issue}" for issue in conformance.issues)
        warnings_list.extend(f"CF conformance warning: {warning}" for warning in conformance.warnings)

        if not cf_declared and self.config.strict_cf:
            warnings_list.append(
                "CF Conventions declaration is missing; the declared-scope conformance check failed"
            )
        
        # Compute missing data statistics  
        total_values = 0
        missing_values = 0
        lazy_missing_counts = []
        
        for var_name, data_var in dataset.data_vars.items():
            var_size = data_var.size
            missing_count = data_var.isnull().sum()
            missing_data = missing_count.data
            if HAS_DASK and hasattr(missing_data, "compute"):
                lazy_missing_counts.append(missing_data)
                var_missing = 0
            else:
                var_missing = int(missing_count.item())
            
            total_values += var_size
            missing_values += var_missing

        if lazy_missing_counts:
            computed_counts = dask.compute(*lazy_missing_counts)
            missing_values += sum(int(value) for value in computed_counts)
        
        missing_fraction = missing_values / total_values if total_values > 0 else 0
        metrics['missing_data_fraction'] = missing_fraction
        
        if missing_fraction > self.missing_data_warning_threshold:
            warnings_list.append(
                f"High fraction of missing data: {missing_fraction:.2%} "
                f"(threshold {self.missing_data_warning_threshold:.2%})"
            )
        
        # Checksum verification (if available)
        checksum_verified = self._verify_checksums(dataset, file_path)
        metrics['checksum_verified'] = checksum_verified
        
        if not checksum_verified and self.config.validate_checksums:
            warnings_list.append("Checksum verification failed or not available")
        
        metrics['warnings'] = warnings_list
        
        return metrics
    
    def _inspect_cf_declaration(self, dataset: xr.Dataset) -> bool:
        """
        Check only whether a CF Conventions global declaration is present.
        
        Parameters:
            dataset: Dataset to check
            
        Returns:
            True when a CF declaration token is present
        """
        required_global_attrs = ['Conventions']
        
        for attr in required_global_attrs:
            if attr not in dataset.attrs:
                return False
            
            if 'CF' not in str(dataset.attrs[attr]):
                return False
        
        return True
    
    def _verify_checksums(self, dataset: xr.Dataset, file_path: Path) -> bool:
        """
        Verify data checksums if available.
        
        Parameters:
            dataset: Dataset to verify
            file_path: Source file path
            
        Returns:
            True if checksums verified or not available
        """
        # Look for checksum information in global attributes
        checksum_attrs = ['checksum_md5', 'md5sum', 'data_checksum']
        
        stored_checksum = None
        for attr in checksum_attrs:
            if attr in dataset.attrs:
                stored_checksum = dataset.attrs[attr]
                break
        
        if stored_checksum is None:
            # No stored checksum - cannot verify but not an error
            return True
        
        # Compute file checksum
        computed_checksum = self._compute_file_checksum(file_path)
        
        return computed_checksum == stored_checksum
    
    def _compute_file_checksum(self, file_path: Path) -> str:
        """
        Compute MD5 checksum of file.
        
        Parameters:
            file_path: Path to file
            
        Returns:
            MD5 checksum as hex string
        """
        hash_md5 = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_cache_key(self, file_path: Path) -> str:
        """
        Generate cache key for file.
        
        Parameters:
            file_path: Path to file
            
        Returns:
            Cache key string
        """
        # Use file path and modification time for cache key
        stat = file_path.stat()
        key_data = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result: IOResult, file_path: Path) -> bool:
        """
        Check if cached result is still valid.
        
        Parameters:
            cached_result: Previously cached result
            file_path: Path to source file
            
        Returns:
            True if cache is valid
        """
        try:
            file_path.stat()
            # Cache is valid if file hasn't been modified
            # This is a simplified check - could be enhanced
            return True
        except (OSError, IOError):
            return False
    
    def write(self, 
              dataset: xr.Dataset,
              output_path: Union[str, Path],
              format: Optional[DataFormat] = None,
              **kwargs) -> None:
        """
        Write dataset to file with specified format.
        
        Parameters:
            dataset: Dataset to write
            output_path: Output file path
            format: Output format (auto-detect if None)
            **kwargs: Additional format-specific options
        """
        output_path = Path(output_path)
        
        if format is None:
            format = self._detect_format_from_extension(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write using appropriate method
        if format in [DataFormat.NETCDF4, DataFormat.NETCDF3]:
            self._write_netcdf(dataset, output_path, format, **kwargs)
        elif format == DataFormat.ZARR:
            self._write_zarr(dataset, output_path, **kwargs)
        elif format == DataFormat.CSV:
            self._write_csv(dataset, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def _detect_format_from_extension(self, file_path: Path) -> DataFormat:
        """Detect format from file extension for writing."""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.nc', '.nc4', '.netcdf']:
            return DataFormat.NETCDF4
        elif suffix == '.zarr':
            return DataFormat.ZARR  
        elif suffix == '.csv':
            return DataFormat.CSV
        else:
            return DataFormat.NETCDF4  # Default
    
    def _write_netcdf(self, 
                      dataset: xr.Dataset, 
                      output_path: Path,
                      format: DataFormat,
                      **kwargs) -> None:
        """Write NetCDF file."""
        engine = 'netcdf4' if format == DataFormat.NETCDF4 else 'netcdf3'
        
        encoding = kwargs.get('encoding', {})
        
        # Apply compression if requested
        if self.config.compression != CompressionType.NONE:
            compression_kwargs = self._get_compression_kwargs()
            
            for var_name in dataset.data_vars:
                if var_name not in encoding:
                    encoding[var_name] = {}
                encoding[var_name].update(compression_kwargs)
        
        dataset.to_netcdf(output_path, engine=engine, encoding=encoding, **kwargs)
    
    def _write_zarr(self, dataset: xr.Dataset, output_path: Path, **kwargs) -> None:
        """Write Zarr format."""
        if not HAS_ZARR:
            raise ImportError("zarr required for Zarr output")
        
        dataset.to_zarr(output_path, **kwargs)
    
    def _write_csv(self, dataset: xr.Dataset, output_path: Path, **kwargs) -> None:
        """Write CSV file (simplified)."""
        # Convert to DataFrame first
        df = dataset.to_dataframe()
        df.to_csv(output_path, **kwargs)
    
    def _get_compression_kwargs(self) -> Dict[str, Any]:
        """Get compression settings for NetCDF output."""
        if self.config.compression == CompressionType.ZLIB:
            return {
                'zlib': True,
                'complevel': self.config.compression_level
            }
        elif self.config.compression == CompressionType.GZIP:
            return {
                'zlib': True,  # gzip uses zlib in NetCDF
                'complevel': self.config.compression_level
            }
        else:
            return {'zlib': True, 'complevel': 4}  # Default
    
    def clear_cache(self) -> None:
        """Clear I/O cache."""
        self._cache.clear()
