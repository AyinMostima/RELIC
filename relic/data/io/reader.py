"""
Data Reader Module

Provides unified data reading interfaces for RELIC data processing layer.
Built on IOHandler with bounded CF variable metadata validation and preservation.

This module offers high-level APIs for reading scientific data in various formats
(NetCDF, Zarr, CSV, HDF5) with automatic format detection, scoped metadata checks,
time coordinate processing, and variable standardization.

Key Features:
- Unified reading interface for multiple formats
- Automatic CF-1.8 standardization
- Time coordinate processing and validation
- Variable subsetting and unit conversion
- PyTorch tensor output support
- Quality control and validation

Mathematical Framework:
Data reading preserves scientific integrity through:
- CF-1.8 metadata standardization
- Time coordinate normalization: t_std = T(t_raw, calendar, bounds)
- Unit conversion: x_std = U(x_raw, units_raw, units_target)
- Quality validation: Q(data, metadata, constraints)

References:
- CF Conventions 1.8: http://cfconventions.org/cf-conventions/cf-conventions.html
- xarray documentation: http://xarray.pydata.org/en/stable/
- PyTorch tensor operations: https://pytorch.org/docs/stable/tensors.html
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
import torch
from torch import Tensor

# Import existing modules
from ..standards.cf_conventions import CFConventions, CFAttributeSynchronizer
from ..standards.unit_converter import unit_converter, ConversionContext
from ...coupling.external.common.io import IOHandler, IOConfig, DataFormat
from ...coupling.external.common.time import TimeHandler
from ...coupling.external.common.validators import Validator
from ...coupling.scales.temporal.core import ValidationError


@dataclass
class ReadConfig:
    """
    Configuration for data reading operations.

    Controls format detection, CF standardization, time processing,
    and output format preferences for scientific data reading.
    """
    # Format and I/O settings
    format: DataFormat = DataFormat.AUTO         # Data format preference
    chunks: Optional[Dict[str, int]] = None      # Chunking strategy for large files
    parallel: bool = True                        # Enable parallel I/O
    cache_size: int = 128                        # Cache size in MB

    # CF standardization
    cf_validate: bool = True                     # Validate bounded CF metadata subset
    cf_standardize: bool = True                  # Process explicit CF metadata only
    strict_cf: bool = False                      # Raise on bounded metadata failures

    # Time coordinate processing
    process_time: bool = True                    # Enable time coordinate processing
    time_bounds: bool = True                     # Compute time bounds if missing
    standard_calendar: bool = False              # Reject unbound cross-calendar mapping

    # Variable selection and conversion
    variables: Optional[List[str]] = None        # Variables to read (None = all)
    exclude_vars: Optional[List[str]] = None     # Variables to exclude
    standard_names_only: bool = False            # Only return CF standard named vars

    # Unit conversion settings
    apply_unit_conversion: bool = False          # Enable unit conversion
    target_units: Optional[Dict[str, str]] = None  # Target units for specific variables
    file_path: Optional[str] = None              # File path (for convenience)

    # Quality control
    validate_data: bool = True                   # Enable data validation
    fill_missing: bool = False                   # Fill missing coordinate data
    quality_flags: bool = False                  # Include quality flags if available

    # Output format
    to_torch: bool = False                       # Convert to PyTorch tensors
    torch_dtype: torch.dtype = torch.float32    # Target tensor dtype
    torch_device: str = 'cpu'                    # Target tensor device


@dataclass
class ReadResult:
    """
    Result from data reading operations.

    Contains loaded data in requested format, standardized metadata,
    quality metrics, and performance information.
    """
    # Primary data
    data: Union[xr.Dataset, Dict[str, Tensor]]   # Loaded data
    metadata: Dict[str, Any]                     # Standardized metadata

    # Quality and validation
    cf_compliant: bool = False                   # Dataset-level CF conformance is unbound
    validation_passed: bool = True               # Data validation status
    success: bool = True                         # Overall success status
    warnings: List[str] = field(default_factory=list)  # Quality warnings

    # Performance metrics
    load_time: float = 0.0                       # Loading time in seconds
    file_size: int = 0                           # Total file size in bytes
    memory_usage: Optional[float] = None         # Peak memory usage in MB

    # Time coordinate info
    time_processed: bool = False                 # Time coordinates processed
    time_bounds_computed: bool = False           # Time bounds computed
    calendar_converted: bool = False             # Calendar converted

    # Variable information
    variables_read: List[str] = field(default_factory=list)    # Data variables this call returned
    variables_standardized: Dict[str, str] = field(default_factory=dict)  # Name mappings


class DataReader:
    """
    Unified data reader for scientific datasets.
    
    Provides high-level interface for reading scientific data with automatic
    CF-1.8 standardization, time coordinate processing, and quality control.
    Built on IOHandler foundation with enhanced standardization capabilities.
    """
    
    def __init__(self, config: Optional[ReadConfig] = None):
        """
        Initialize data reader with configuration.
        
        Parameters:
            config: Reading configuration options
        """
        self.config = config or ReadConfig()
        
        # Initialize underlying components
        io_config = IOConfig(
            chunks=self.config.chunks,
            parallel=self.config.parallel,
            cache_size=self.config.cache_size,
            format=self.config.format,
            strict_cf=self.config.strict_cf,
            fill_missing=self.config.fill_missing,
            validate_checksums=True
        )
        
        self.io_handler = IOHandler(io_config)
        self.time_handler = TimeHandler()
        self.cf_conventions = CFConventions()
        self.cf_synchronizer = CFAttributeSynchronizer(strict_mode=self.config.strict_cf)
        
        # Initialize validator if needed
        if self.config.validate_data:
            self.validator = Validator()
        else:
            self.validator = None
    
    def read_dataset(self, 
                    path: Union[str, Path, List[Union[str, Path]]], 
                    **kwargs) -> ReadResult:
        """
        Read dataset from file(s) with full processing pipeline.
        
        Parameters:
            path: File path(s) to read
            **kwargs: Additional configuration overrides
            
        Returns:
            ReadResult with processed dataset and metadata
            
        Raises:
            IOError: If file cannot be read
            ValueError: If data processing fails
        """
        # Override config with kwargs
        effective_config = self._merge_config(kwargs)
        
        # Read raw data using IOHandler
        io_runtime_fields = (
            "chunks",
            "parallel",
            "cache_size",
            "format",
            "strict_cf",
            "fill_missing",
            "enable_cache",
            "decode_times",
        )
        previous_io_config = {
            field_name: getattr(self.io_handler.config, field_name)
            for field_name in io_runtime_fields
        }
        canonical_time_ingress = effective_config.process_time
        self.io_handler.config.chunks = effective_config.chunks
        self.io_handler.config.parallel = effective_config.parallel
        self.io_handler.config.cache_size = effective_config.cache_size
        self.io_handler.config.format = effective_config.format
        self.io_handler.config.strict_cf = effective_config.strict_cf
        self.io_handler.config.fill_missing = effective_config.fill_missing
        if (
            effective_config.apply_unit_conversion or canonical_time_ingress
        ) and previous_io_config["enable_cache"]:
            self.io_handler.config.enable_cache = False
        self.io_handler.config.decode_times = not canonical_time_ingress
        try:
            io_result = self.io_handler.read(path)
        finally:
            for field_name, previous_value in previous_io_config.items():
                setattr(self.io_handler.config, field_name, previous_value)
        
        # Initialize result
        result = ReadResult(
            data=io_result.data,
            metadata={},
            cf_compliant=io_result.cf_compliant,
            load_time=io_result.load_time,
            file_size=io_result.file_size,
            memory_usage=io_result.memory_usage,
            warnings=io_result.warnings.copy()
        )
        
        try:
            # Process dataset through pipeline
            processed_data = self._process_pipeline(
                io_result.data, 
                effective_config, 
                result
            )
            if isinstance(processed_data, xr.Dataset):
                if effective_config.apply_unit_conversion:
                    processed_data = processed_data.load()
                    processed_data = processed_data.copy(deep=True)
                    self._safe_close_dataset(io_result.data)
                    self._safe_close_dataset(processed_data)
            else:
                self._safe_close_dataset(io_result.data)

            result.data = processed_data
            
        except (KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            # A failed pipeline raises in every mode: raw, unprocessed data are never returned in
            # place of the processed data (strict_cf only governs bounded metadata findings).
            e.add_note("Data processing failed; raw data are not returned in place of processed data")
            raise

        # The data variables this call returns: after selection, and after tensor
        # conversion when tensors are requested.
        delivered = result.data
        result.variables_read = [
            str(name)
            for name in (delivered.data_vars if isinstance(delivered, xr.Dataset) else delivered)
        ]

        return result

    def _safe_close_dataset(self, dataset: Optional[xr.Dataset]) -> None:
        """
        Close xarray datasets to release file handles safely.

        Parameters:
            dataset: Dataset to close if it supports close()
        """
        if dataset is None:
            return
        close_fn = getattr(dataset, "close", None)
        if callable(close_fn):
            # A failed close raises: an unreleased file handle is never reported as released.
            close_fn()
    
    def read_tensors(self,
                    path: Union[str, Path, List[Union[str, Path]]],
                    variables: List[str],
                    **kwargs) -> Dict[str, Tensor]:
        """
        Read specific variables as PyTorch tensors.
        
        Parameters:
            path: File path(s) to read
            variables: List of variable names to read
            **kwargs: Additional configuration overrides
            
        Returns:
            Dictionary mapping variable names to tensors
            
        Raises:
            ValueError: If variables not found or conversion fails
        """
        # Force tensor output and set variables
        config_override = {
            'to_torch': True,
            'variables': variables,
            **kwargs
        }
        
        result = self.read_dataset(path, **config_override)
        
        if isinstance(result.data, dict):
            return result.data
        else:
            # Convert xarray Dataset to tensor dict
            return self._dataset_to_tensors(result.data, variables)
    
    def _merge_config(self, overrides: Dict[str, Any]) -> ReadConfig:
        """
        Merge configuration with overrides.
        
        Parameters:
            overrides: Configuration overrides
            
        Returns:
            Merged configuration
        """
        # Create copy of current config
        merged = ReadConfig(
            format=overrides.get('format', self.config.format),
            chunks=overrides.get('chunks', self.config.chunks),
            parallel=overrides.get('parallel', self.config.parallel),
            cache_size=overrides.get('cache_size', self.config.cache_size),
            cf_validate=overrides.get('cf_validate', self.config.cf_validate),
            cf_standardize=overrides.get('cf_standardize', self.config.cf_standardize),
            strict_cf=overrides.get('strict_cf', self.config.strict_cf),
            process_time=overrides.get('process_time', self.config.process_time),
            time_bounds=overrides.get('time_bounds', self.config.time_bounds),
            standard_calendar=overrides.get('standard_calendar', self.config.standard_calendar),
            variables=overrides.get('variables', self.config.variables),
            exclude_vars=overrides.get('exclude_vars', self.config.exclude_vars),
            standard_names_only=overrides.get('standard_names_only', self.config.standard_names_only),
            validate_data=overrides.get('validate_data', self.config.validate_data),
            fill_missing=overrides.get('fill_missing', self.config.fill_missing),
            quality_flags=overrides.get('quality_flags', self.config.quality_flags),
            apply_unit_conversion=overrides.get('apply_unit_conversion', self.config.apply_unit_conversion),
            target_units=overrides.get('target_units', self.config.target_units),
            file_path=overrides.get('file_path', self.config.file_path),
            to_torch=overrides.get('to_torch', self.config.to_torch),
            torch_dtype=overrides.get('torch_dtype', self.config.torch_dtype),
            torch_device=overrides.get('torch_device', self.config.torch_device)
        )
        
        return merged
    
    def _process_pipeline(self, 
                         dataset: xr.Dataset, 
                         config: ReadConfig,
                         result: ReadResult) -> Union[xr.Dataset, Dict[str, Tensor]]:
        """
        Process dataset through standardization pipeline.
        
        Parameters:
            dataset: Raw dataset from IOHandler
            config: Effective configuration
            result: Result object to update
            
        Returns:
            Processed dataset in requested format
        """
        processed = dataset.copy()
        
        # Step 1: Variable selection
        if config.variables or config.exclude_vars:
            processed = self._select_variables(processed, config, result)
        
        # Step 2: CF standardization
        if config.cf_standardize:
            processed = self._standardize_cf(processed, config, result)
        
        # Step 3: Unit conversion
        if config.apply_unit_conversion:
            processed = self._apply_unit_conversion(processed, config, result)
        
        # Step 4: Time coordinate processing
        if config.process_time and 'time' in processed.coords:
            processed = self._process_time_coordinates(processed, config, result)
        
        # Step 5: Data validation
        if config.validate_data and self.validator:
            self._validate_data(processed, config, result)
        
        # Step 6: Output format conversion
        if config.to_torch:
            return self._convert_to_tensors(processed, config, result)
        else:
            return processed
    
    def _select_variables(self, 
                         dataset: xr.Dataset, 
                         config: ReadConfig,
                         result: ReadResult) -> xr.Dataset:
        """
        Select/exclude variables from dataset.
        
        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update
            
        Returns:
            Dataset with selected variables
        """
        available_vars = list(dataset.data_vars.keys())
        selected_vars = available_vars.copy()
        
        # Apply inclusion filter
        if config.variables:
            # Support both original and standard names
            final_selection = []
            for var_name in config.variables:
                if var_name in dataset.variables:
                    final_selection.append(var_name)
                else:
                    found = False
                    if self.cf_conventions.is_valid_standard_name(var_name):
                        standard_name = self.cf_conventions.get_standard_name(
                            var_name
                        )
                        for av in available_vars:
                            var_attrs = dataset[av].attrs
                            if var_attrs.get('standard_name') == standard_name:
                                final_selection.append(av)
                                found = True
                                break
                    
                    if not found:
                        result.warnings.append(f"Variable '{var_name}' not found in dataset")
            
            selected_vars = final_selection
        
        # Apply exclusion filter
        if config.exclude_vars:
            for var_name in config.exclude_vars:
                if var_name in selected_vars:
                    selected_vars.remove(var_name)
        
        # Apply standard names filter
        if config.standard_names_only:
            standard_vars = []
            for var_name in selected_vars:
                var_attrs = dataset[var_name].attrs
                if 'standard_name' in var_attrs:
                    standard_name = var_attrs['standard_name']
                    if self.cf_conventions.is_valid_standard_name(standard_name):
                        standard_vars.append(var_name)
            selected_vars = standard_vars
        
        # References: CF Conventions 1.12 Section 7.1; xarray Dataset.drop_vars.
        # Mathematical expression: D_out = D_in without unselected data fields.
        # Parameter meanings: D denotes the dataset, including coordinate support.
        # Units: each source variable retains its original units and bounds.
        # Domain/constraints: preserve coordinates even when bounds introduce an
        # extra dimension; explicit exclusions still apply and dangling time
        # bounds declarations are rejected by the original time-processing owner.
        # Sign convention: no numerical values or signed fluxes are transformed.
        # Dataset[selected_vars] can silently discard a coordinate on an extra
        # bounds dimension. Dropping data variables preserves that support.
        dropped = [name for name in available_vars if name not in selected_vars]
        dropped.extend(name for name in (config.exclude_vars or [])
                       if name in dataset.coords and name not in dropped)
        return dataset.drop_vars(dropped)
    
    def _standardize_cf(self, 
                       dataset: xr.Dataset, 
                       config: ReadConfig,
                       result: ReadResult) -> xr.Dataset:
        """
        Validate explicit standard-name metadata without inventing attributes.
        
        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update
            
        Returns:
            Dataset copy with source metadata preserved
        """
        processed = dataset.copy()
        result.variables_standardized = {}
        result.cf_compliant = False
        scope_message = (
            "dataset-level CF conformance was not evaluated; only the "
            "single-variable standard-name/unit subset is available"
        )
        if scope_message not in result.warnings:
            result.warnings.append(scope_message)

        for var_name, data_var in processed.data_vars.items():
            attrs = data_var.attrs.copy()
            if not config.cf_validate:
                continue
            validation = self.cf_conventions.validate_cf_compliance(attrs)
            if validation.warnings:
                result.warnings.extend(
                    f"{var_name}: {warning}" for warning in validation.warnings
                )
            if validation.is_valid:
                continue
            result.validation_passed = False
            result.success = False
            result.warnings.extend(
                f"{var_name}: {issue}" for issue in validation.issues
            )
            if config.strict_cf:
                raise ValueError(
                    f"CF standard-name metadata invalid for '{var_name}': "
                    + "; ".join(validation.issues)
                )

        return processed
    
    def _process_time_coordinates(self, 
                                 dataset: xr.Dataset, 
                                 config: ReadConfig,
                                 result: ReadResult) -> xr.Dataset:
        """
        Process time coordinates for standardization.
        
        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update
            
        Returns:
            Dataset with processed time coordinates
        """
        # References: CF Conventions 1.12, Sections 4.4 and 7.1.
        # Expression: t = u * scale + epoch and I_i = [b_i0, b_i1).
        # Parameters: units/calendar bind scale and epoch; bounds bind support.
        processed = dataset.copy()
        time_var = processed.time
        units = self.time_handler.detect_time_units(processed)
        calendar = self.time_handler.detect_calendar(processed)

        attr_bounds = time_var.attrs.get("bounds")
        encoding_bounds = time_var.encoding.get("bounds")
        declared_bounds = [
            str(value)
            for value in (attr_bounds, encoding_bounds)
            if value is not None
        ]
        if len(set(declared_bounds)) > 1:
            raise ValidationError(
                "Conflicting CF time-bounds variable declarations"
            )

        bounds_name = declared_bounds[0] if declared_bounds else None
        source_bounds = None
        source_bounds_dims = None
        source_bounds_attrs: Dict[str, Any] = {}
        if bounds_name is not None:
            if bounds_name not in processed.variables:
                raise ValidationError(
                    f"Declared CF time-bounds variable is missing: {bounds_name}"
                )
            bounds_variable = processed[bounds_name]
            if bounds_variable.ndim != 2 or bounds_variable.dims[0] != "time":
                raise ValidationError(
                    "CF time bounds must have dimensions ('time', bounds_dimension)"
                )
            source_bounds = bounds_variable.values
            source_bounds_dims = bounds_variable.dims
            source_bounds_attrs = dict(bounds_variable.attrs)

        time_coord = self.time_handler.create_time_coordinate(
            values=time_var.values,
            units=units,
            calendar=calendar,
            compute_bounds=config.time_bounds,
            bounds=source_bounds,
        )

        if config.standard_calendar and time_coord.calendar != "gregorian":
            original_calendar = time_coord.calendar
            time_coord = self.time_handler.convert_calendar(
                time_coord,
                "gregorian",
            )
            result.calendar_converted = time_coord.calendar != original_calendar

        cf_time_info = self.time_handler.to_cf_compliant(time_coord)
        processed = processed.assign_coords(time=cf_time_info["time"]["data"])
        processed.time.attrs = cf_time_info["time"]["attrs"]

        if time_coord.bounds is not None:
            output_bounds_name = bounds_name or "time_bounds"
            output_bounds_dims = source_bounds_dims or ("time", "bounds")
            processed[output_bounds_name] = (
                output_bounds_dims,
                cf_time_info["time_bounds"]["data"],
            )
            processed[output_bounds_name].attrs = {
                **source_bounds_attrs,
                **cf_time_info["time_bounds"]["attrs"],
            }
            processed.time.attrs["bounds"] = output_bounds_name
            result.time_bounds_computed = source_bounds is None

        result.time_processed = True
        return processed
    
    def _validate_data(self, 
                      dataset: xr.Dataset, 
                      config: ReadConfig,
                      result: ReadResult) -> None:
        """
        Validate dataset using comprehensive validators.
        
        Parameters:
            dataset: Dataset to validate
            config: Configuration
            result: Result object to update
        """
        try:
            # Use the validator from coupling/external/common/validators
            validation_report = self.validator.validate(dataset)
            result.cf_compliant = False
            result.warnings.extend(validation_report.warnings)
            scope_message = (
                "dataset-level CF conformance was not evaluated; external "
                "adapter checks are not a CF dataset validator"
            )
            if scope_message not in result.warnings:
                result.warnings.append(scope_message)

            validation_passed = not validation_report.errors
            if not validation_passed:
                result.validation_passed = False
                result.warnings.extend(validation_report.errors)
                
                if config.strict_cf:
                    raise ValueError(
                        "Dataset validation failed: "
                        f"{validation_report.errors}"
                    )
                
        except (KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
            result.validation_passed = False
            result.warnings.append(f"Data validation error: {e}")
            if config.strict_cf:
                raise
    
    def _apply_unit_conversion(self, 
                              dataset: xr.Dataset, 
                              config: ReadConfig,
                              result: ReadResult) -> xr.Dataset:
        """
        Convert CF-consistent variables to their canonical or requested unit.

        The target is ``config.target_units[var]`` when given, otherwise the
        canonical unit of the variable's standard name in the CF table owned by
        ``CFConventions``. Both the source and the target unit must be
        CF-consistent with the standard name (the same relation
        ``_standardize_cf`` validates); the numbers are converted only by the
        canonical ``UnitConverter``. A variable that cannot be converted keeps
        its values and units, and the read is marked failed (non-strict) or
        raises (``strict_cf``); it is never reported as a success.

        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update

        Returns:
            Dataset with converted units
        """
        # References: CF Conventions 1.13, Sections 3.1 and 3.3 (units and the
        #   standard-name canonical unit); UDUNITS-2 via CFConventions and
        #   UnitConverter.
        # Mathematical expression: v_target = U(v_source; u_source -> u_target)
        #   with u_target = u_canonical(standard_name) unless requested.
        # Parameter meanings: U is UnitConverter.convert; u_source and u_target
        #   must both be convertible to u_canonical(standard_name).
        # Units: labels from the variable attributes and the CF table.
        # Domain/constraints: variables without units or standard_name, or whose
        #   standard name defines no canonical unit, are left unchanged; a
        #   CF-inconsistent source or target unit and a conversion without a rule
        #   are failures, not silent pass-throughs. Both labels are
        #   UDUNITS-convertible to one canonical unit, so the conversion is a
        #   same-dimension scale or affine offset and needs no thermodynamic or
        #   density context; the context is empty (no dataset means, no
        #   defaults). A quantity change that would need T, p, or a density
        #   (for example ppm -> mol m-3) changes the standard name and is
        #   rejected above; a rule that still asks for context raises and is
        #   recorded as a failure.
        # Sign convention: unit conversion preserves the sign of the values.
        processed = dataset.copy()

        for var_name in list(processed.data_vars):
            data_var = processed[var_name]
            current_units = data_var.attrs.get('units', '')
            standard_name = data_var.attrs.get('standard_name', '')
            if not current_units or not standard_name:
                continue

            try:
                if config.target_units and var_name in config.target_units:
                    target_units = config.target_units[var_name]
                else:
                    target_units = self.cf_conventions.get_standard_units(standard_name)
                if target_units is None:
                    continue
                for role, units in (("source", current_units), ("target", target_units)):
                    if not self.cf_conventions.are_standard_name_units_equivalent(
                        standard_name, units
                    ):
                        raise ValueError(
                            f"{var_name}: {role} units {units!r} are not CF-consistent "
                            f"with standard_name {standard_name!r}; values not converted"
                        )
            except ValueError as exc:
                self._record_unit_conversion_failure(str(exc), config, result)
                continue

            if current_units == target_units:
                continue
            try:
                converted_data = unit_converter.convert(
                    np.asarray(data_var.values),
                    standard_name,
                    current_units,
                    target_units,
                    ConversionContext(),
                )
            except ValueError as exc:
                self._record_unit_conversion_failure(
                    f"Unit conversion failed for {var_name}: {exc}; "
                    f"units kept as {current_units!r}",
                    config,
                    result,
                )
                continue

            converted_var = data_var.copy(data=converted_data)
            converted_var.attrs['units'] = target_units
            converted_var.attrs['original_units'] = current_units
            processed[var_name] = converted_var
            result.warnings.append(
                f"Converted {var_name} from {current_units} to {target_units}"
            )

        return processed

    @staticmethod
    def _record_unit_conversion_failure(
        message: str,
        config: ReadConfig,
        result: ReadResult,
    ) -> None:
        """Fail the read explicitly: raise in strict mode, else mark it failed."""
        if config.strict_cf:
            raise ValueError(message)
        result.warnings.append(message)
        result.success = False
        result.validation_passed = False

    def _convert_to_tensors(self, 
                           dataset: xr.Dataset, 
                           config: ReadConfig,
                           result: ReadResult) -> Dict[str, Tensor]:
        """
        Convert xarray Dataset to PyTorch tensor dictionary.
        
        Parameters:
            dataset: Input dataset
            config: Configuration
            result: Result object to update
            
        Returns:
            Dictionary of variable tensors
        """
        tensors = {}
        
        for var_name, data_var in dataset.data_vars.items():
            try:
                # Convert to numpy first
                if hasattr(data_var.data, 'compute'):
                    # Dask array - compute to memory
                    numpy_data = data_var.data.compute()
                else:
                    numpy_data = data_var.values
                
                # Convert to PyTorch tensor
                tensor = torch.from_numpy(numpy_data).to(
                    dtype=config.torch_dtype,
                    device=config.torch_device
                )
                
                tensors[var_name] = tensor
                
            except (KeyError, ValueError, TypeError, RuntimeError, OSError) as e:
                # A variable that cannot be converted raises; it is never dropped from the result.
                e.add_note(f"Tensor conversion failed for {var_name}")
                raise
        
        return tensors
    
    def _dataset_to_tensors(self, 
                           dataset: xr.Dataset, 
                           variables: List[str]) -> Dict[str, Tensor]:
        """
        Convert specific variables from dataset to tensors.
        
        Parameters:
            dataset: Source dataset
            variables: Variable names to convert
            
        Returns:
            Dictionary of tensors
        """
        tensors = {}
        
        for var_name in variables:
            if var_name in dataset.data_vars:
                data_var = dataset[var_name]
                if hasattr(data_var.data, 'compute'):
                    numpy_data = data_var.data.compute()
                else:
                    numpy_data = data_var.values
                
                tensor = torch.from_numpy(numpy_data).to(
                    dtype=self.config.torch_dtype,
                    device=self.config.torch_device
                )
                tensors[var_name] = tensor
        
        return tensors


# High-level convenience functions
def read_dataset(path: Union[str, Path, List[Union[str, Path]]], 
                format: str = "auto", 
                chunks: Optional[Dict[str, int]] = None,
                cf_validate: bool = True) -> xr.Dataset:
    """
    Convenience function to read dataset with default settings.
    
    Parameters:
        path: File path(s) to read
        format: Data format ("auto", "netcdf4", "zarr", "csv")
        chunks: Chunking strategy for large files
        cf_validate: Enable CF-1.8 validation
        
    Returns:
        xarray Dataset with processed data
    """
    config = ReadConfig(
        format=DataFormat(format),
        chunks=chunks,
        cf_validate=cf_validate,
        to_torch=False
    )
    
    reader = DataReader(config)
    result = reader.read_dataset(path)
    
    return result.data


def read_tensors(path: Union[str, Path, List[Union[str, Path]]],
                variables: List[str],
                to_torch: bool = True) -> Dict[str, Tensor]:
    """
    Convenience function to read variables as PyTorch tensors.
    
    Parameters:
        path: File path(s) to read
        variables: Variable names to read
        to_torch: Convert to PyTorch tensors
        
    Returns:
        Dictionary mapping variable names to tensors
    """
    config = ReadConfig(
        variables=variables,
        to_torch=to_torch,
        cf_standardize=True
    )
    
    reader = DataReader(config)
    result = reader.read_tensors(path, variables)
    
    return result
