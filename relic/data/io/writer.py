"""
Data Writer Module

Provides unified data writing interfaces for RELIC data processing layer.
Preserves explicit CF metadata, compression, and format settings without
claiming dataset-level CF conformance.

This module offers high-level APIs for writing scientific data in various formats
(NetCDF, Zarr, CSV) with bounded CF metadata handling, compression optimization,
and metadata synchronization.

Key Features:
- Unified writing interface for multiple formats
- Bounded standard-name/unit metadata validation
- Optimal compression and chunking strategies
- Time coordinate bounds computation
- Global attributes management
- PyTorch tensor input support

Mathematical Framework:
Data writing ensures scientific integrity through:
- CF-1.8 metadata standardization
- Time bounds computation: bounds = B(time_coords, method)
- Compression optimization: C(data, method, level) -> compressed_data
- Coordinate encoding: E(coords, calendar, units) -> cf_coords

References:
- CF Conventions 1.8: http://cfconventions.org/cf-conventions/cf-conventions.html
- NetCDF Best Practices: https://www.unidata.ucar.edu/software/netcdf/docs/BestPractices.html
- Zarr specifications: https://zarr.readthedocs.io/en/stable/spec.html
"""

import logging
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import asdict, dataclass, field
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import xarray as xr
import torch
from torch import Tensor

# Import existing modules
from ..standards.cf_conventions import CFConventions, CFAttributeSynchronizer
from ...coupling.external.common.io import IOHandler, IOConfig, DataFormat, CompressionType
from ...coupling.external.common.time import TimeHandler
from ...coupling.scales.temporal.core import CalendarError, ValidationError

logger = logging.getLogger(__name__)

_DELIVERY_ATTRIBUTES = (
    "relic_delivery_key", "relic_delivery_created_utc", "relic_delivery_options_sha256",
)


@dataclass
class WriteConfig:
    """
    Configuration for data writing operations.
    
    Controls output format, compression, bounded CF metadata, and metadata
    handling for scientific data writing operations.
    """
    # Format and output settings
    format: DataFormat = DataFormat.NETCDF4      # Output format
    compression: CompressionType = CompressionType.ZSTD  # Compression method
    compression_level: int = 4                   # Compression level (1-9)
    
    # Chunking strategy
    chunks: Optional[Dict[str, int]] = None      # Custom chunking strategy
    auto_chunk: bool = True                      # Enable automatic chunking
    chunk_time_first: bool = True                # Prioritize time dimension
    
    # Bounded CF variable metadata handling
    cf_sync: bool = True                         # Propagate explicit CF metadata
    strict_cf: bool = False                      # Raise on bounded metadata failures
    add_bounds: bool = True                      # Add coordinate bounds if missing
    update_history: bool = True                  # Update history attribute
    
    # Global attributes
    title: Optional[str] = None                  # Dataset title
    institution: Optional[str] = None            # Institution name
    source: Optional[str] = None                 # Data source/method
    comment: Optional[str] = None                # Additional comments
    references: Optional[str] = None             # References
    
    # Performance settings
    parallel: bool = False                       # Explicitly sequential until a parallel writer is wired
    validate_output: bool = True                 # Validate written output
    create_dirs: bool = True                     # Create output directories


@dataclass
class WriteResult:
    """
    Result from data writing operations.
    
    Contains information about the write operation, output file properties,
    and any warnings or issues encountered during writing.
    """
    # Output information
    output_path: Path                            # Written file path
    file_size: int = 0                          # Output file size in bytes
    write_time: float = 0.0                     # Writing time in seconds
    compression_ratio: Optional[float] = None    # Achieved compression ratio
    
    # Quality and validation
    cf_compliant: bool = False                   # Dataset-level CF conformance is unbound
    validation_passed: bool = True               # Output validation status
    warnings: List[str] = field(default_factory=list)  # Writing warnings
    
    # Metadata information
    global_attributes_added: Dict[str, str] = field(default_factory=dict)
    variables_written: List[str] = field(default_factory=list)
    coordinates_added: List[str] = field(default_factory=list)
    bounds_computed: List[str] = field(default_factory=list)
    delivery_key: Optional[str] = None
    delivery_created_utc: Optional[str] = None
    sha256: Optional[str] = None
    replayed: bool = False


class DataWriter:
    """
    Unified data writer for scientific datasets.
    
    Provides high-level scientific data writing without inventing CF identity,
    units, or dataset-level conformance claims.
    Built on IOHandler foundation with enhanced standardization capabilities.
    """
    
    def __init__(self, config: Optional[WriteConfig] = None):
        """
        Initialize data writer with configuration.
        
        Parameters:
            config: Writing configuration options
        """
        self.config = config or WriteConfig()
        
        # Initialize underlying components
        io_config = IOConfig(
            parallel=self.config.parallel,
            compression=self.config.compression,
            compression_level=self.config.compression_level
        )
        
        self.io_handler = IOHandler(io_config)
        self.time_handler = TimeHandler()
        self.cf_conventions = CFConventions()
        self.cf_synchronizer = CFAttributeSynchronizer(strict_mode=self.config.strict_cf)
        
    
    @staticmethod
    def _copy_dataset_value(value: Any, *, tensors: bool) -> Any:
        """Copy xarray's public descriptor across the Tensor/NumPy boundary."""
        if isinstance(value, Tensor):
            copied = value.detach().cpu().clone()
            return copied if tensors else copied.numpy()
        if isinstance(value, np.ndarray):
            if tensors and value.dtype.kind in "biufc":
                return torch.from_numpy(np.array(value, dtype=value.dtype.newbyteorder("="), copy=True))
            return value.copy()
        if isinstance(value, np.dtype):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: DataWriter._copy_dataset_value(item, tensors=tensors) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(DataWriter._copy_dataset_value(item, tensors=tensors) for item in value)
        return deepcopy(value)

    def _matching_delivery_timestamp(
        self, output_path: Path, config: WriteConfig, delivery_key: Optional[str], options_sha: str,
    ) -> Optional[str]:
        """Read the existing native receipt contract without changing a target."""
        if delivery_key is None:
            return None
        if not isinstance(delivery_key, str) or not delivery_key.strip():
            raise ValueError("Delivery key must be a nonempty string")
        if config.format not in (DataFormat.NETCDF4, DataFormat.NETCDF3, DataFormat.ZARR):
            raise ValueError("Delivery receipts require a NetCDF or Zarr dataset contract")
        if not output_path.exists():
            return None
        with self._open_output(output_path, config) as previous:
            if previous.attrs.get("relic_delivery_key") != delivery_key:
                return None
            if previous.attrs.get("relic_delivery_options_sha256") != options_sha:
                raise ValueError("Delivery options contract conflicts with existing artifact")
            stamp = previous.attrs.get("relic_delivery_created_utc")
            if not isinstance(stamp, str) or not stamp.endswith("Z"):
                raise ValueError("Existing delivery timestamp contract is invalid")
            datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            return stamp

    def prepare_dataset(
        self, dataset: xr.Dataset, output_path: Union[str, Path], *,
        delivery_key: Optional[str] = None, **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare an independent candidate without creating files or directories.

        Xarray owns the dataset descriptor. Numeric arrays use the existing
        checkpoint Tensor vocabulary; other array/codec types remain native
        values and require explicit support before durable serialization.
        """
        output_path = Path(output_path).resolve()
        config = self._merge_config(kwargs)
        if config.parallel:
            raise NotImplementedError(
                "DataWriter parallel writing is not wired to a verified "
                "scheduler; use the default sequential writer."
            )
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("Dataset preparation requires an xarray Dataset")
        config_values = asdict(config)
        config_values.update(format=config.format.value, compression=config.compression.value)
        options_sha = hashlib.sha256(json.dumps(
            config_values, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        result = WriteResult(output_path=output_path, delivery_key=delivery_key)
        result.delivery_created_utc = self._matching_delivery_timestamp(
            output_path, config, delivery_key, options_sha,
        ) or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        processed = self._process_write_pipeline(dataset, config, result)
        if not result.validation_passed:
            raise ValueError("Dataset metadata validation failed: " + "; ".join(result.warnings))
        if delivery_key is not None:
            processed.attrs.update({
                "relic_delivery_key": delivery_key,
                "relic_delivery_created_utc": result.delivery_created_utc,
                "relic_delivery_options_sha256": options_sha,
            })
        encoding = self._prepare_encoding(processed, config)
        processed.load()
        result_values = asdict(result)
        result_values["output_path"] = str(output_path)
        return self._copy_dataset_value({
            "dataset": processed.to_dict(data="array", encoding=True),
            "config": config_values, "encoding": encoding,
            "result": result_values, "options_sha256": options_sha,
        }, tensors=True)

    def write_dataset(self, 
                     dataset: Optional[xr.Dataset],
                     output_path: Union[str, Path],
                     *,
                     delivery_key: Optional[str] = None,
                     snapshot: Optional[Dict[str, Any]] = None,
                     **kwargs) -> WriteResult:
        """
        Write xarray Dataset to file with full processing pipeline.
        
        Parameters:
            dataset: Dataset to write
            output_path: Output file path
            **kwargs: Additional configuration overrides
            
        Returns:
            WriteResult with operation details and metadata
            
        Raises:
            IOError: If file cannot be written
            ValueError: If data processing fails
        """
        output_path = Path(output_path).resolve()
        effective_config = self._merge_config(kwargs) if snapshot is None else WriteConfig()
        result = WriteResult(output_path=output_path, delivery_key=delivery_key)
        try:
            if snapshot is None:
                if dataset is None:
                    raise ValueError("A dataset or prepared snapshot is required")
                snapshot = self.prepare_dataset(dataset, output_path, delivery_key=delivery_key, **kwargs)
            else:
                if dataset is not None or kwargs:
                    raise ValueError("Prepared snapshot conflicts with raw dataset or write options")
                prepared_path = Path(snapshot["result"]["output_path"]).resolve()
                if os.path.normcase(str(prepared_path)) != os.path.normcase(str(output_path)):
                    raise ValueError("Prepared snapshot conflicts with output path")
                if delivery_key is not None and snapshot["result"]["delivery_key"] not in (None, delivery_key):
                    raise ValueError("Prepared snapshot conflicts with delivery key")
            config_values = dict(snapshot["config"])
            config_values["format"] = DataFormat(config_values["format"])
            config_values["compression"] = CompressionType(config_values["compression"])
            effective_config = WriteConfig(**config_values)
            result_values = self._copy_dataset_value(snapshot["result"], tensors=False)
            result_values["output_path"] = output_path
            result = WriteResult(**result_values)
            if result.delivery_key is None:
                result.delivery_key = delivery_key
            delivery_key = result.delivery_key
            processed_dataset = xr.Dataset.from_dict(
                self._copy_dataset_value(snapshot["dataset"], tensors=False),
            )
            encoding = self._copy_dataset_value(snapshot["encoding"], tensors=False)
            if delivery_key is not None:
                # Bind a request to an unbound candidate without changing its
                # data, preparation timestamp, configuration, or encoding.
                processed_dataset.attrs.update({
                    "relic_delivery_key": delivery_key,
                    "relic_delivery_created_utc": result.delivery_created_utc,
                    "relic_delivery_options_sha256": snapshot["options_sha256"],
                })
            stamp = self._matching_delivery_timestamp(
                output_path, effective_config, delivery_key, snapshot["options_sha256"],
            )
            same_delivery = stamp is not None
            if same_delivery and stamp != result.delivery_created_utc:
                raise ValueError("Existing delivery timestamp conflicts with prepared snapshot")
            if effective_config.create_dirs:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            if same_delivery:
                self._validate_output(output_path, processed_dataset, effective_config, encoding)
                result.replayed = True
                self._artifact_receipt(output_path, result)
                return result

            start_time = time.time()
            temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
            backup = None
            try:
                self._write_with_format(processed_dataset, temporary, effective_config, encoding)
                if effective_config.validate_output or delivery_key is not None:
                    self._validate_output(temporary, processed_dataset, effective_config, encoding)
                if output_path.is_dir():
                    backup = output_path.with_name(f".{output_path.name}.{uuid4().hex}.previous")
                    try:
                        os.replace(output_path, backup)
                        os.replace(temporary, output_path)
                    except BaseException:
                        # Either rename can complete before an interruption is
                        # observed. Restore the old version only if publication
                        # has not established a visible target.
                        if backup.exists() and not output_path.exists():
                            os.replace(backup, output_path)
                            backup = None
                        raise
                else:
                    os.replace(temporary, output_path)
            finally:
                published_backup = (
                    backup is not None and backup.exists()
                    and output_path.exists() and not temporary.exists()
                )
                if temporary.exists():
                    self._remove_temporary_artifact(temporary)
                if published_backup and backup is not None:
                    try:
                        self._remove_temporary_artifact(backup)
                    except OSError as exc:
                        result.warnings.append(f"Published output; previous artifact cleanup failed at {backup}: {exc}")
                        logger.warning(result.warnings[-1])
            result.write_time = time.time() - start_time
            self._artifact_receipt(output_path, result)
            
        except (CalendarError, ValidationError, NotImplementedError):
            # Scientific time evidence is mandatory even in non-strict output mode.
            raise
        except (KeyError, ValueError, TypeError, RuntimeError, OSError):
            # A failed write raises in every mode; a result without a written artifact is never
            # returned (strict_cf only governs bounded metadata findings).
            result.validation_passed = False
            result.cf_compliant = False
            raise
        
        return result

    @staticmethod
    def _remove_temporary_artifact(path: Path) -> None:
        """Remove only a staging or previous artifact created by this write."""
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    @staticmethod
    def _open_output(path: Path, config: WriteConfig) -> xr.Dataset:
        if config.format == DataFormat.ZARR:
            return xr.open_zarr(path, consolidated=False)
        return xr.open_dataset(path)

    @staticmethod
    def _artifact_receipt(path: Path, result: WriteResult) -> None:
        """Hash actual file bytes, or sorted relative paths and bytes in a group."""
        digest = hashlib.sha256()
        total = 0
        paths = sorted(
            (item for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(path).as_posix(),
        ) if path.is_dir() else [path]
        for item in paths:
            size = item.stat().st_size
            if path.is_dir():
                relative = item.relative_to(path).as_posix().encode("utf-8")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(size.to_bytes(8, "big"))
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            total += size
        result.sha256 = digest.hexdigest()
        result.file_size = total
    
    def write_tensors(self,
                     tensors: Dict[str, Tensor],
                     output_path: Union[str, Path],
                     template: Optional[xr.Dataset] = None,
                     **kwargs) -> WriteResult:
        """
        Write PyTorch tensors as scientific dataset.
        
        Parameters:
            tensors: Dictionary mapping variable names to tensors
            output_path: Output file path
            template: Template dataset for coordinates and metadata
            **kwargs: Additional configuration overrides
            
        Returns:
            WriteResult with operation details
            
        Raises:
            ValueError: If tensor conversion fails
        """
        # Convert tensors to xarray Dataset
        dataset = self._tensors_to_dataset(tensors, template)
        
        # Write dataset
        return self.write_dataset(dataset, output_path, **kwargs)
    
    def _merge_config(self, overrides: Dict[str, Any]) -> WriteConfig:
        """
        Merge configuration with overrides.
        
        Parameters:
            overrides: Configuration overrides
            
        Returns:
            Merged configuration
        """
        # Create copy of current config
        merged = WriteConfig(
            format=DataFormat(overrides.get('format', self.config.format.value)),
            compression=CompressionType(overrides.get('compression', self.config.compression.value)),
            compression_level=overrides.get('compression_level', self.config.compression_level),
            chunks=overrides.get('chunks', self.config.chunks),
            auto_chunk=overrides.get('auto_chunk', self.config.auto_chunk),
            chunk_time_first=overrides.get('chunk_time_first', self.config.chunk_time_first),
            cf_sync=overrides.get('cf_sync', self.config.cf_sync),
            strict_cf=overrides.get('strict_cf', self.config.strict_cf),
            add_bounds=overrides.get('add_bounds', self.config.add_bounds),
            update_history=overrides.get('update_history', self.config.update_history),
            title=overrides.get('title', self.config.title),
            institution=overrides.get('institution', self.config.institution),
            source=overrides.get('source', self.config.source),
            comment=overrides.get('comment', self.config.comment),
            references=overrides.get('references', self.config.references),
            parallel=overrides.get('parallel', self.config.parallel),
            validate_output=overrides.get('validate_output', self.config.validate_output),
            create_dirs=overrides.get('create_dirs', self.config.create_dirs)
        )
        
        return merged
    
    def _process_write_pipeline(self, 
                               dataset: xr.Dataset, 
                               config: WriteConfig,
                               result: WriteResult) -> xr.Dataset:
        """
        Process dataset through writing pipeline.
        
        Parameters:
            dataset: Input dataset
            config: Effective configuration
            result: Result object to update
            
        Returns:
            Processed dataset ready for writing
        """
        processed = dataset.copy()
        for attribute in _DELIVERY_ATTRIBUTES:
            processed.attrs.pop(attribute, None)
        
        # Step 1: Neutral global attributes (no Conventions claim)
        processed = self._ensure_global_attributes(processed, config, result)
        
        # Step 2: CF standardization and synchronization
        if config.cf_sync:
            processed = self._synchronize_cf_attributes(processed, config, result)
        
        # Step 3: Add time bounds if missing
        processed = self._add_time_bounds(processed, config, result)
        
        # Step 4: Attach uncertainty metadata (if uncertainty variables exist)
        processed = self._attach_uncertainty_metadata(processed, config, result)
        
        # Step 5: Optimize chunking if enabled
        if config.auto_chunk:
            processed = self._optimize_chunking(processed, config, result)
        
        return processed
    
    def _ensure_global_attributes(self, 
                                 dataset: xr.Dataset, 
                                 config: WriteConfig,
                                 result: WriteResult) -> xr.Dataset:
        """
        Preserve explicit global attributes without declaring CF conformance.

        This method sets the neutral descriptive global attributes early in the
        pipeline: creation_date, the optional title, institution, source,
        comment, and references of ``config``, and the history entry when
        ``config.update_history`` is set. It writes no ``Conventions``
        attribute: the writer checks only the single-variable
        standard-name/unit subset and the time coordinate, not dataset-level
        CF conformance, and it also writes Zarr and CSV products, so it cannot
        declare that a product follows a CF version. A ``Conventions`` value
        the caller supplied passes through unchanged as the caller's own
        declaration.

        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update

        Returns:
            Dataset with the neutral global attributes merged over the caller's
        """
        processed = dataset.copy()
        
        # Get current global attributes
        global_attrs = processed.attrs.copy()
        
        # Create neutral global attributes without a Conventions claim.
        cf_attrs = self.cf_conventions.create_global_attributes(
            title=config.title,
            institution=config.institution,
            source=config.source,
            comment=config.comment,
            references=config.references
        )
        cf_attrs["creation_date"] = global_attrs.get(
            "creation_date", result.delivery_created_utc or cf_attrs["creation_date"],
        )
        
        # Merge with existing attributes (CF attributes take precedence)
        global_attrs.update(cf_attrs)
        
        # Update history if requested
        if config.update_history:
            current_history = global_attrs.get('history', '')
            created = (
                datetime.fromisoformat(result.delivery_created_utc.replace("Z", "+00:00"))
                if result.delivery_created_utc else datetime.now(timezone.utc)
            )
            timestamp = created.strftime('%Y-%m-%d %H:%M:%S UTC')
            new_entry = f"{timestamp}: Processed by RELIC DataWriter"
            
            if current_history:
                global_attrs['history'] = f"{current_history}; {new_entry}"
            else:
                global_attrs['history'] = new_entry
        
        # Set updated attributes
        processed.attrs = global_attrs
        result.global_attributes_added = cf_attrs
        
        return processed

    def _add_time_bounds(self, 
                        dataset: xr.Dataset, 
                        config: WriteConfig,
                        result: WriteResult) -> xr.Dataset:
        """
        Validate and serialize the canonical CF time coordinate.

        Existing source bounds are validated and preserved. New interval bounds
        are computed only when ``config.add_bounds`` is true. Missing or
        conflicting unit, calendar, and bounds evidence always fails closed.
        
        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update
            
        Returns:
            Dataset with time bounds added
        """
        processed = dataset.copy()
        
        if "time" not in processed.coords:
            return processed

        time_var = processed["time"]
        if time_var.ndim != 1:
            raise ValidationError("CF time coordinate must be one-dimensional")

        units = self.time_handler.detect_time_units(processed)
        calendar = self.time_handler.detect_calendar(processed)
        time_values = time_var.values
        if np.issubdtype(time_var.dtype, np.datetime64) or time_var.dtype.kind == "O":
            # CF time coding changes representation, not the declared calendar.
            # Xarray performs datetime/cftime conversion; TimeHandler validates
            # the numerical time and interval support through its sole owner.
            variable = time_var.variable.copy(deep=True)
            for key in ("units", "calendar"):
                variable.attrs.pop(key, None)
            variable.encoding.update(units=units, calendar=calendar, dtype=np.dtype("float64"))
            time_values = xr.coders.CFDatetimeCoder().encode(variable, name="time").values.copy()
        bounds_declarations = [
            (source, str(value))
            for source, value in (
                ("time.attrs", time_var.attrs.get("bounds")),
                ("time.encoding", time_var.encoding.get("bounds")),
            )
            if value is not None
        ]
        bounds_names = {name for _, name in bounds_declarations}
        if len(bounds_names) > 1:
            evidence = ", ".join(
                f"{source}={name}" for source, name in bounds_declarations
            )
            raise ValidationError(f"Conflicting CF time bounds declarations: {evidence}")

        source_bounds_name = next(iter(bounds_names), None)
        source_bounds = None
        source_bounds_dims = None
        source_bounds_attrs = None
        source_bounds_encoding = {}
        if source_bounds_name is not None:
            if source_bounds_name not in processed.variables:
                raise ValidationError(
                    f"Declared CF time bounds variable is missing: {source_bounds_name}"
                )
            source_variable = processed[source_bounds_name]
            if source_variable.ndim != 2 or source_variable.dims[0] != time_var.dims[0]:
                raise ValidationError(
                    "CF time bounds must be two-dimensional with time as the first dimension"
                )
            # CF bounds inherit the coordinate contract; an explicit conflicting
            # declaration is an error in the original TimeHandler owner.
            bounds_contract = xr.Dataset({"time": source_variable.variable})
            try:
                self.time_handler.detect_time_units(bounds_contract, source_units=units)
                self.time_handler.detect_calendar(bounds_contract, source_calendar=calendar)
            except (CalendarError, ValidationError) as exc:
                raise ValidationError(f"CF time bounds contract conflicts with time: {exc}") from exc
            source_bounds = source_variable.values
            source_bounds_dims = source_variable.dims
            source_bounds_attrs = source_variable.attrs.copy()
            source_bounds_encoding = source_variable.encoding.copy()
            # The numerical boundary representation carries the validated time
            # contract in attributes; keep all other declared storage encoding.
            source_bounds_attrs.update(units=units, calendar=calendar)
            for key in ("units", "calendar"):
                source_bounds_encoding.pop(key, None)
            if np.issubdtype(source_variable.dtype, np.datetime64) or source_variable.dtype.kind == "O":
                bounds_variable = source_variable.variable.copy(deep=True)
                for key in ("units", "calendar"):
                    bounds_variable.attrs.pop(key, None)
                bounds_variable.encoding.update(units=units, calendar=calendar, dtype=np.dtype("float64"))
                source_bounds = xr.coders.CFDatetimeCoder().encode(bounds_variable, name=source_bounds_name).values.copy()

        time_coord = self.time_handler.create_time_coordinate(
            values=time_values,
            units=units,
            calendar=calendar,
            compute_bounds=config.add_bounds and source_bounds is None,
            bounds=source_bounds,
        )
        serialized = self.time_handler.to_cf_compliant(time_coord)

        original_time_attrs = time_var.attrs.copy()
        original_time_encoding = time_var.encoding.copy()
        for key in ("units", "calendar", "bounds"):
            original_time_encoding.pop(key, None)
        canonical_time_attrs = serialized["time"]["attrs"].copy()
        final_bounds_name = source_bounds_name
        if source_bounds_name is not None:
            canonical_time_attrs["bounds"] = source_bounds_name
        elif time_coord.bounds is None:
            canonical_time_attrs.pop("bounds", None)
        else:
            final_bounds_name = "time_bounds"
            if final_bounds_name in processed.variables:
                raise ValidationError(
                    "Undeclared time_bounds variable would be overwritten by generated bounds"
                )

        original_time_attrs.update(canonical_time_attrs)
        processed = processed.assign_coords(
            {
                "time": (
                    time_var.dims,
                    serialized["time"]["data"],
                    original_time_attrs,
                )
            }
        )
        processed["time"].encoding.update(original_time_encoding)

        if source_bounds_name is not None:
            processed[source_bounds_name] = xr.Variable(
                source_bounds_dims, source_bounds, attrs=source_bounds_attrs or {},
                encoding=source_bounds_encoding,
            )
            if processed[source_bounds_name].dims != source_bounds_dims:
                raise ValidationError("Source time-bounds dimensions changed during serialization")
        elif final_bounds_name is not None:
            bounds_payload = serialized["time_bounds"]
            processed[final_bounds_name] = (
                (time_var.dims[0], "bounds"),
                bounds_payload["data"],
                bounds_payload["attrs"],
            )
            result.bounds_computed.append("time")
        
        return processed

    def _attach_uncertainty_metadata(self, 
                                   dataset: xr.Dataset, 
                                   config: WriteConfig,
                                   result: WriteResult) -> xr.Dataset:
        """
        Attach uncertainty variable metadata and relationships.
        
        Identifies uncertainty variables (e.g., *_std, *_ci_lower, *_ci_upper)
        and establishes CF-1.8 ancillary_variables relationships.
        
        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update
            
        Returns:
            Dataset with uncertainty metadata attached
        """
        processed = dataset.copy()
        
        # Identify base variables and their uncertainty counterparts
        base_vars = {}
        uncertainty_vars = {}
        
        for var_name in processed.data_vars.keys():
            if var_name.endswith('_std'):
                base_name = var_name[:-4]  # Remove '_std'
                if base_name in processed.data_vars:
                    if base_name not in uncertainty_vars:
                        uncertainty_vars[base_name] = []
                    uncertainty_vars[base_name].append(var_name)
                    base_vars[base_name] = processed[base_name]
            elif var_name.endswith('_ci_lower'):
                base_name = var_name[:-9]  # Remove '_ci_lower'
                if base_name in processed.data_vars:
                    if base_name not in uncertainty_vars:
                        uncertainty_vars[base_name] = []
                    uncertainty_vars[base_name].append(var_name)
                    base_vars[base_name] = processed[base_name]
            elif var_name.endswith('_ci_upper'):
                base_name = var_name[:-9]  # Remove '_ci_upper'
                if base_name in processed.data_vars:
                    if base_name not in uncertainty_vars:
                        uncertainty_vars[base_name] = []
                    uncertainty_vars[base_name].append(var_name)
                    base_vars[base_name] = processed[base_name]
        
        # Process each base variable and its uncertainty variables
        for base_name, uncertainty_list in uncertainty_vars.items():
            try:
                base_var = processed[base_name]
                base_attrs = base_var.attrs.copy()
                
                # Set ancillary_variables attribute on base variable
                processed[base_name].attrs['ancillary_variables'] = ' '.join(uncertainty_list)
                
                # Update metadata for each uncertainty variable
                for uncert_var_name in uncertainty_list:
                    uncert_var = processed[uncert_var_name]
                    uncert_attrs = uncert_var.attrs.copy()
                    
                    # Copy key metadata from base variable
                    if 'standard_name' in base_attrs:
                        uncert_attrs['standard_name'] = base_attrs['standard_name']
                    if 'units' in base_attrs:
                        uncert_attrs['units'] = base_attrs['units']
                    if 'cell_methods' in base_attrs:
                        uncert_attrs['cell_methods'] = base_attrs['cell_methods']
                    if 'coordinates' in base_attrs:
                        uncert_attrs['coordinates'] = base_attrs['coordinates']
                    
                    # Set descriptive long_name based on uncertainty type
                    if uncert_var_name.endswith('_std'):
                        uncert_attrs['long_name'] = f"Standard deviation of {base_attrs.get('long_name', base_name)}"
                        uncert_attrs['comment'] = f"Standard deviation uncertainty for {base_name}"
                    elif uncert_var_name.endswith('_ci_lower'):
                        uncert_attrs['long_name'] = f"Lower confidence interval of {base_attrs.get('long_name', base_name)}"
                        uncert_attrs['comment'] = f"Lower confidence interval bound for {base_name}"
                    elif uncert_var_name.endswith('_ci_upper'):
                        uncert_attrs['long_name'] = f"Upper confidence interval of {base_attrs.get('long_name', base_name)}"
                        uncert_attrs['comment'] = f"Upper confidence interval bound for {base_name}"
                    
                    # Update the variable attributes
                    processed[uncert_var_name].attrs = uncert_attrs
                    
            except (KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
                # Metadata boundary (strict_cf governs metadata findings): strict raises; otherwise
                # the finding is recorded and the result is marked failed, never read as success.
                result.warnings.append(f"Uncertainty metadata attachment failed for {base_name}: {e}")
                result.validation_passed = False
                if config.strict_cf:
                    raise
        
        return processed

    def _synchronize_cf_attributes(self, 
                                  dataset: xr.Dataset, 
                                  config: WriteConfig,
                                  result: WriteResult) -> xr.Dataset:
        """
        Propagate and validate explicit CF metadata for all variables.
        
        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update
            
        Returns:
            Dataset copy with bounded metadata results
        """
        processed = dataset.copy()
        result.cf_compliant = False
        scope_message = (
            "dataset-level CF conformance was not evaluated; only the "
            "single-variable standard-name/unit subset is available"
        )
        if scope_message not in result.warnings:
            result.warnings.append(scope_message)
        
        for var_name, data_var in processed.data_vars.items():
            try:
                metadata = data_var.attrs.copy()
                updated_metadata = self.cf_conventions.sync_variable_attributes(
                    torch.from_numpy(data_var.values), metadata
                )
                processed[var_name].attrs = updated_metadata
                validation = self.cf_conventions.validate_cf_compliance(
                    updated_metadata
                )
                result.warnings.extend(
                    f"{var_name}: {warning}"
                    for warning in validation.warnings
                )
                
            except (KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
                # Metadata boundary (strict_cf governs metadata findings): strict raises; otherwise
                # the declared attributes are kept unchanged and the result is marked failed.
                result.warnings.append(f"CF synchronization failed for {var_name}: {e}")
                result.validation_passed = False
                processed[var_name].attrs = data_var.attrs.copy()
                if config.strict_cf:
                    raise ValueError(
                        f"CF standard-name metadata invalid for '{var_name}': {e}"
                    ) from e
        
        # Record written variables
        result.variables_written = list(processed.data_vars.keys())
        
        return processed
    
    def _update_global_attributes(self, 
                                 dataset: xr.Dataset, 
                                 config: WriteConfig,
                                 result: WriteResult) -> xr.Dataset:
        """
        Update caller-owned global attributes without a CF conformance claim.
        
        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update
            
        Returns:
            Dataset with updated global attributes
        """
        processed = dataset.copy()
        
        # Get current global attributes
        global_attrs = processed.attrs.copy()
        
        # Create neutral global attributes.
        cf_attrs = self.cf_conventions.create_global_attributes(
            title=config.title,
            institution=config.institution,
            source=config.source,
            comment=config.comment,
            references=config.references
        )
        
        # Merge with existing attributes
        global_attrs.update(cf_attrs)
        
        # Update history if requested
        if config.update_history:
            current_history = global_attrs.get('history', '')
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            new_entry = f"{timestamp}: Processed by RELIC DataWriter"
            
            if current_history:
                global_attrs['history'] = f"{current_history}; {new_entry}"
            else:
                global_attrs['history'] = new_entry
        
        # Set updated attributes
        processed.attrs = global_attrs
        result.global_attributes_added = cf_attrs
        
        return processed
    
    def _optimize_chunking(self, 
                          dataset: xr.Dataset, 
                          config: WriteConfig,
                          result: WriteResult) -> xr.Dataset:
        """
        Optimize chunking strategy for performance.
        
        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update
            
        Returns:
            Dataset with optimized chunking
        """
        if config.chunks:
            # Use explicit chunking
            # An explicit chunking request that cannot be applied raises; it is never dropped.
            processed = dataset.chunk(config.chunks)
        else:
            # Auto-chunking strategy
            processed = dataset
            
            # For large datasets, apply automatic chunking
            total_size = sum(var.nbytes for var in dataset.data_vars.values())
            if total_size > 100 * 1024 * 1024:  # > 100MB
                if config.chunk_time_first and 'time' in dataset.dims:
                    # Time-first chunking strategy
                    chunk_dict = {'time': min(100, len(dataset.time))}
                    processed = dataset.chunk(chunk_dict)
                else:
                    processed = dataset.chunk('auto')
        
        return processed
    
    def _get_compression_settings(self, config: WriteConfig, variable: xr.DataArray) -> Dict[str, Any]:
        """
        Get compression settings based on configuration.
        
        Returns:
            Dictionary of compression settings
        """
        method = config.compression
        if method == CompressionType.AUTO:
            method = CompressionType.ZSTD if config.format != DataFormat.NETCDF3 else CompressionType.NONE
        if config.format == DataFormat.ZARR:
            from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec
            if method == CompressionType.NONE:
                return {"compressors": []}
            if method == CompressionType.ZSTD:
                codec = ZstdCodec(level=config.compression_level)
            elif method == CompressionType.GZIP:
                codec = GzipCodec(level=config.compression_level)
            elif method in (CompressionType.ZLIB, CompressionType.BLOSC):
                from zarr.core.array_spec import ArrayConfig, ArraySpec
                from zarr.core.buffer import default_buffer_prototype
                from zarr.core.dtype import parse_data_type
                codec = BloscCodec(
                    cname="zlib" if method == CompressionType.ZLIB else "zstd",
                    clevel=config.compression_level,
                )
                # Let Zarr resolve dtype-dependent codec settings before they
                # enter the frozen standard codec dictionary.
                codec = codec.evolve_from_array_spec(ArraySpec(
                    shape=variable.shape,
                    dtype=parse_data_type(np.dtype(variable.encoding.get("dtype", variable.dtype)), zarr_format=3),
                    fill_value=None, config=ArrayConfig.from_dict({}),
                    prototype=default_buffer_prototype(),
                ))
            else:
                raise ValueError(f"Unsupported Zarr compression: {method.value}")
            return {"compressors": [codec.to_dict()]}
        if method == CompressionType.NONE or config.format == DataFormat.CSV:
            return {}
        if config.format == DataFormat.NETCDF3:
            raise ValueError("NetCDF3 requires compression='none'")
        if method in (CompressionType.ZLIB, CompressionType.GZIP):
            # NetCDF's GZIP option denotes the HDF5 deflate filter, named zlib.
            return {"zlib": True, "complevel": config.compression_level}
        if method == CompressionType.ZSTD:
            return {"compression": "zstd", "complevel": config.compression_level}
        if method == CompressionType.BLOSC:
            return {"compression": "blosc_zstd", "complevel": config.compression_level}
        raise ValueError(f"Unsupported NetCDF compression: {method.value}")
    
    def _prepare_encoding(self, 
                         dataset: xr.Dataset, 
                         config: WriteConfig) -> Dict[str, Dict[str, Any]]:
        """
        Prepare encoding dictionary for variables.
        
        Parameters:
            dataset: Dataset to encode
            config: Configuration
            
        Returns:
            Encoding dictionary
        """
        encoding = {}
        
        for var_name, data_var in dataset.data_vars.items():
            compression_settings = self._get_compression_settings(config, data_var)
            var_encoding = {
                key: value for key, value in data_var.encoding.items()
                if key in ("dtype", "_FillValue", "missing_value", "scale_factor", "add_offset")
            }
            
            # Add compression settings
            if compression_settings and (data_var.ndim or config.format == DataFormat.ZARR):
                var_encoding.update(compression_settings)
            
            # Add chunking if available
            if hasattr(data_var.data, 'chunks') and data_var.data.chunks:
                chunk_key = "chunks" if config.format == DataFormat.ZARR else "chunksizes"
                var_encoding[chunk_key] = tuple(c[0] for c in data_var.data.chunks)
            
            # Fill value handling
            if '_FillValue' not in data_var.attrs and '_FillValue' not in var_encoding and np.issubdtype(data_var.dtype, np.floating):
                var_encoding['_FillValue'] = np.nan
            
            encoding[var_name] = var_encoding
        
        return encoding
    
    def _write_with_format(self, 
                          dataset: xr.Dataset, 
                          output_path: Path,
                          config: WriteConfig,
                          encoding: Dict[str, Dict[str, Any]]) -> None:
        """
        Write dataset using appropriate format handler.
        
        Parameters:
            dataset: Dataset to write
            output_path: Output path
            config: Configuration
            encoding: Variable encoding settings
        """
        if config.format in [DataFormat.NETCDF4, DataFormat.NETCDF3]:
            # Use xarray's netcdf writer directly to avoid encoding conflicts
            dataset.to_netcdf(
                output_path,
                encoding=encoding,
                format="NETCDF3_CLASSIC" if config.format == DataFormat.NETCDF3 else "NETCDF4"
            )
        elif config.format == DataFormat.ZARR:
            # Use xarray's zarr writer
            dataset.to_zarr(
                output_path,
                encoding=encoding,
                mode='w', zarr_format=3, consolidated=False,
            )
        elif config.format == DataFormat.CSV:
            # Convert to CSV (simplified)
            df = dataset.to_dataframe()
            df.to_csv(output_path)
        else:
            raise ValueError(f"Unsupported output format: {config.format}")
    
    def _validate_output(
        self, output_path: Path,
        expected: Optional[xr.Dataset] = None, config: Optional[WriteConfig] = None,
        encoding: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Validate written output file.
        
        Parameters:
            output_path: Path to output file
            expected: Frozen candidate to compare with the stored dataset
        """
        try:
            effective = config or self.config
            if effective.format == DataFormat.CSV:
                import pandas as pd
                pd.read_csv(output_path)
                return
            with self._open_output(output_path, effective) as opened:
                actual = opened.load()
                if expected is not None:
                    normalized = xr.decode_cf(expected.copy(deep=True))
                    if effective.format in (DataFormat.NETCDF4, DataFormat.NETCDF3):
                        # NetCDF attributes have a type and length, but no
                        # distinction between a scalar and a length-one vector.
                        # Normalize only this storage equivalence; retain every
                        # attribute name and value in the content comparison.
                        for view in (normalized, actual):
                            for owner in (view, *view.variables.values()):
                                for key, value in tuple(owner.attrs.items()):
                                    if isinstance(value, (list, tuple, np.ndarray)):
                                        array = np.asarray(value)
                                        if array.shape == (1,):
                                            owner.attrs[key] = array.item()
                    xr.testing.assert_identical(normalized, actual)
                    for name, variable in normalized.variables.items():
                        if variable.dtype != actual[name].dtype:
                            raise ValueError(f"Output dtype contract mismatch for {name}")
                        declared = {**variable.encoding, **(encoding or {}).get(str(name), {})}
                        for key in ("dtype", "_FillValue", "missing_value", "scale_factor", "add_offset", "units", "calendar"):
                            if key not in declared:
                                continue
                            reference = declared[key]
                            observed = actual[name].encoding.get(key, actual[name].attrs.get(key))
                            if key == "dtype":
                                if np.dtype(reference) != np.dtype(observed):
                                    raise ValueError(f"Output storage dtype contract mismatch for {name}")
                            elif reference is None:
                                if observed is not None:
                                    raise ValueError(f"Output {key} contract mismatch for {name}")
                            else:
                                np.testing.assert_equal(observed, reference, err_msg=f"Output {key} contract mismatch for {name}")
            
        except (AssertionError, KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            raise ValueError(f"Output content validation failed: {e}") from e
    
    def _tensors_to_dataset(self, 
                           tensors: Dict[str, Tensor],
                           template: Optional[xr.Dataset] = None) -> xr.Dataset:
        """
        Convert PyTorch tensors to xarray Dataset.
        
        Parameters:
            tensors: Dictionary of tensors
            template: Template dataset for coordinates
            
        Returns:
            xarray Dataset
        """
        data_vars = {}
        coords = {}
        
        # Use only explicit variable dimensions; never guess from coordinates.
        for var_name, tensor in tensors.items():
            numpy_data = tensor.detach().cpu().numpy().copy()
            if template is not None and var_name in template.data_vars:
                variable = template[var_name]
                if tuple(tensor.shape) != variable.shape:
                    raise ValueError(f"Tensor shape for {var_name} does not match its template")
                data_vars[var_name] = xr.Variable(
                    variable.dims, numpy_data, attrs=deepcopy(variable.attrs),
                    encoding=deepcopy(variable.encoding),
                )
            elif tensor.ndim == 0:
                data_vars[var_name] = xr.Variable((), numpy_data)
            else:
                raise ValueError(f"Nonscalar tensor {var_name} requires template variable dimensions")
        
        # Use template coordinates if available
        if template is not None:
            coords = {name: variable.copy(deep=True) for name, variable in template.coords.items()}
            # Create dataset with template coordinates
            dataset = xr.Dataset(data_vars, coords=coords)
            # Copy global attributes from template
            dataset.attrs = deepcopy(template.attrs)
        else:
            # Create simple dataset without coordinates
            dataset = xr.Dataset(data_vars)
        
        return dataset
    
# High-level convenience functions
def write_dataset(dataset: Optional[xr.Dataset],
                 output_path: Union[str, Path],
                 format: Optional[str] = None,
                 compression: Optional[str] = None,
                 cf_sync: Optional[bool] = None,
                 *, delivery_key: Optional[str] = None,
                 snapshot: Optional[Dict[str, Any]] = None,
                 **kwargs) -> WriteResult:
    """
    Convenience function to write dataset with default settings.
    
    Parameters:
        dataset: Dataset to write
        output_path: Output file path
        format: Output format ("netcdf4", "zarr", "csv")
        compression: Compression method ("zstd", "zlib", "none")
        cf_sync: Enable CF-1.8 synchronization
    """
    overrides = dict(kwargs)
    for name, value in (("format", format), ("compression", compression), ("cf_sync", cf_sync)):
        if value is not None:
            overrides[name] = value
    return DataWriter().write_dataset(
        dataset, output_path, delivery_key=delivery_key, snapshot=snapshot, **overrides,
    )


def write_tensors(tensors: Dict[str, Tensor],
                 output_path: Union[str, Path],
                 template: Optional[xr.Dataset] = None,
                 format: str = "netcdf4",
                 *, delivery_key: Optional[str] = None,
                 **kwargs) -> WriteResult:
    """
    Convenience function to write tensors as dataset.
    
    Parameters:
        tensors: Dictionary of tensors
        output_path: Output file path
        template: Template dataset for coordinates and metadata
        format: Output format
    """
    config = WriteConfig(
        format=DataFormat(format),
        cf_sync=True
    )
    
    writer = DataWriter(config)
    return writer.write_tensors(tensors, output_path, template, delivery_key=delivery_key, **kwargs)
