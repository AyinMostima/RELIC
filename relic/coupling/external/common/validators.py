"""
Data Validation for External Model Coupling

This module provides comprehensive validation capabilities for external
model data, reporting bounded metadata, physical consistency, and quality
control before integration into RELIC scientific workflows.

Validation Framework:
1. Format Compliance: CF-1.8 conventions, NetCDF standards
2. Physical Consistency: Value ranges, units, dimensionality
3. Temporal Validation: Time coordinate completeness and ordering
4. Spatial Validation: Geographic bounds, grid regularity
5. Scientific Quality: Missing data patterns, outlier detection

Mathematical Framework:
Validation uses statistical tests and physical constraints:
- Range checks: x_min <= x <= x_max for each variable
- Continuity tests: |nablat| < threshold for temporal derivatives
- Conservation checks: integralintegral flux dA dt = expected total
- Outlier detection: |x - mu|/sigma > k (k = coupling.external.validation.outlier_zscore_threshold)

Quality Control Metrics:
- Completeness: fraction of non-missing data
- Consistency: compliance with expected patterns
- Accuracy: deviation from expected ranges/distributions
- Reliability: temporal/spatial continuity measures

References:
- CF Checker: https://github.com/cedadev/cf-checker
- Quality Control Best Practices: WMO Guide to QC Procedures
- Statistical Process Control: Montgomery (2020) Introduction to SPC
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from relic.parameters import get_global_manager

from ....utils.physical_constants import get_physical_constant
from .datasource import DataSource
from .schema import InternalSchema
from .validation_types import ValidationReport


class ValidationCategory(Enum):
    """Categories of validation checks."""
    FORMAT_COMPLIANCE = "format_compliance"      # CF, NetCDF standards
    PHYSICAL_CONSISTENCY = "physical_consistency"  # Value ranges, units
    TEMPORAL_QUALITY = "temporal_quality"        # Time coordinate checks
    SPATIAL_QUALITY = "spatial_quality"          # Geographic consistency
    DATA_COMPLETENESS = "data_completeness"      # Missing data assessment
    STATISTICAL_OUTLIERS = "statistical_outliers"  # Outlier detection
    METADATA_QUALITY = "metadata_quality"        # Attribute completeness


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"                       # Informational message
    WARNING = "warning"                 # Potential issue, data usable
    ERROR = "error"                     # Serious issue, may affect results
    CRITICAL = "critical"               # Critical issue, data unusable


@dataclass
class ExternalValidationIssue:
    """
    Individual validation issue with detailed information.
    
    Encapsulates specific problems found during validation
    with context for diagnosis and correction.
    """
    category: ValidationCategory            # Type of validation check
    severity: ValidationSeverity            # Issue severity
    message: str                           # Human-readable description
    variable: Optional[str] = None         # Affected variable name
    location: Optional[str] = None         # Specific location in data
    expected: Optional[Any] = None         # Expected value/range
    actual: Optional[Any] = None           # Actual value found
    suggestion: Optional[str] = None       # Recommended fix
    reference: Optional[str] = None        # Documentation reference


ValidationIssue = ExternalValidationIssue


# Declared parameter of each threshold of ValidationConfig. The missing-data thresholds are the
# external IO warning threshold and the adapter quality-control limit (one value per concept); the
# others live in coupling.external.validation (relic/parameters/database/relic/coupling/
# external_validation.yaml), each with its references or its convention label.
VALIDATION_PARAMETER_KEYS: Dict[str, str] = {
    "max_missing_fraction": "coupling.external.io.missing_data_warning_threshold",
    "missing_fraction_error": "coupling.external.regrid.missing_threshold",
    "outlier_threshold": "coupling.external.validation.outlier_zscore_threshold",
    "outlier_fraction_warning": "coupling.external.validation.outlier_fraction_warning",
    "outlier_fraction_error": "coupling.external.validation.outlier_fraction_error",
    "require_monotonic_time": "coupling.external.validation.require_monotonic_time",
    "max_time_gap_hours": "coupling.external.validation.max_time_gap_hours",
    "require_global_coverage": "coupling.external.validation.require_global_coverage",
    "global_longitude_span_min": "coupling.external.validation.global_longitude_span_min",
    "global_latitude_span_min": "coupling.external.validation.global_latitude_span_min",
    "require_cf_conventions": "coupling.external.validation.require_cf_conventions",
    "cf_version_min": "coupling.external.validation.cf_version_min",
    "require_standard_names": "coupling.external.validation.require_standard_names",
}


@dataclass
class ValidationConfig:
    """
    Configuration for validation operations.

    Every threshold field is None unless set explicitly; Validator takes an unset field from its
    declared parameter (VALIDATION_PARAMETER_KEYS) and keeps an explicit value.
    """
    # Categories to check (if empty, check all)
    enabled_categories: List[ValidationCategory] = field(default_factory=list)

    # Data completeness: missing fraction above which a field is a WARNING, and above which an ERROR
    max_missing_fraction: Optional[float] = None
    missing_fraction_error: Optional[float] = None

    # Statistical outliers: |z| threshold and the flagged fractions of a WARNING and an ERROR
    outlier_threshold: Optional[float] = None
    outlier_fraction_warning: Optional[float] = None
    outlier_fraction_error: Optional[float] = None

    # Temporal validation
    require_monotonic_time: Optional[bool] = None
    max_time_gap_hours: Optional[float] = None

    # Spatial validation: spans (deg) treated as global when global coverage is required
    require_global_coverage: Optional[bool] = None
    global_longitude_span_min: Optional[float] = None
    global_latitude_span_min: Optional[float] = None

    # CF declaration and variable-metadata subset only
    require_cf_conventions: Optional[bool] = None
    cf_version_min: Optional[str] = None
    require_standard_names: Optional[bool] = None

    # Custom validation functions
    custom_validators: List[Callable] = field(default_factory=list)

    def resolved(self) -> "ValidationConfig":
        """A copy whose unset thresholds hold their declared parameter values (required keys)."""
        manager = get_global_manager()
        unset = {
            name: manager.get(key, required=True)
            for name, key in VALIDATION_PARAMETER_KEYS.items()
            if getattr(self, name) is None
        }
        return replace(self, **unset)


class Validator:
    """
    Comprehensive data validation system for external model data.
    
    Provides systematic quality control and compliance checking
    to ensure external data meets RELIC integration requirements.
    Generates detailed reports for diagnosis and correction.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator with configuration.
        
        Parameters:
            config: Validation configuration options
        """
        self.config = (config if config is not None else ValidationConfig()).resolved()
        self.schema = InternalSchema()
        self._validators = self._initialize_validators()
    
    def _initialize_validators(self) -> Dict[ValidationCategory, Callable]:
        """
        Initialize category-specific validation functions.
        
        Returns:
            Dictionary mapping categories to validation functions
        """
        return {
            ValidationCategory.FORMAT_COMPLIANCE: self._validate_format_compliance,
            ValidationCategory.PHYSICAL_CONSISTENCY: self._validate_physical_consistency,
            ValidationCategory.TEMPORAL_QUALITY: self._validate_temporal_quality,
            ValidationCategory.SPATIAL_QUALITY: self._validate_spatial_quality,
            ValidationCategory.DATA_COMPLETENESS: self._validate_data_completeness,
            ValidationCategory.STATISTICAL_OUTLIERS: self._validate_statistical_outliers,
            ValidationCategory.METADATA_QUALITY: self._validate_metadata_quality
        }
    
    def validate(self,
                data: Union[xr.Dataset, xr.DataArray, Dict[str, Any]],
                source: DataSource = DataSource.UNKNOWN,
                variable_mapping: Optional[Dict[str, str]] = None) -> ValidationReport:
        """
        Perform comprehensive validation of external model data.
        
        Parameters:
            data: Data to validate
            source: Data source type
            variable_mapping: Mapping from external to internal names
            
        Returns:
            Comprehensive validation report
        """
        issues = []
        metadata = {}
        
        # Convert to xarray Dataset if needed
        dataset = self._standardize_input(data)
        
        # Determine which categories to validate
        categories_to_check = self.config.enabled_categories or list(ValidationCategory)
        
        # Run validation checks
        for category in categories_to_check:
            if category in self._validators:
                try:
                    category_issues = self._validators[category](dataset, source, variable_mapping)
                    issues.extend(category_issues)
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                    # Validation errors should not crash the system
                    issues.append(ExternalValidationIssue(
                        category=category,
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation check failed: {str(e)}",
                        suggestion="Check data format and validator configuration"
                    ))
        
        # Run custom validators
        for custom_validator in self.config.custom_validators:
            try:
                custom_issues = custom_validator(dataset)
                if isinstance(custom_issues, list):
                    issues.extend(custom_issues)
                elif isinstance(custom_issues, ExternalValidationIssue):
                    issues.append(custom_issues)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.FORMAT_COMPLIANCE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Custom validator failed: {str(e)}"
                ))
        
        # Compile statistics
        metadata['total_issues'] = len(issues)
        metadata['issues_by_severity'] = self._count_by_severity(issues)
        metadata['issues_by_category'] = self._count_by_category(issues)
        
        # Build report
        report = ValidationReport(
            source_path="",  # Will be set by caller
            validation_time=pd.Timestamp.now().isoformat(),
            cf_compliant=False,
            errors=[i.message for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]],
            warnings=[i.message for i in issues if i.severity == ValidationSeverity.WARNING]
        )
        
        # Add detailed issue information to metadata
        metadata['detailed_issues'] = issues
        
        return report
    
    def _standardize_input(self, data: Union[xr.Dataset, xr.DataArray, Dict[str, Any]]) -> xr.Dataset:
        """
        Convert input data to xarray Dataset.
        
        Parameters:
            data: Input data in various formats
            
        Returns:
            Standardized xarray Dataset
        """
        if isinstance(data, xr.Dataset):
            return data
        elif isinstance(data, xr.DataArray):
            return data.to_dataset()
        elif isinstance(data, dict):
            # Try to construct Dataset from dictionary
            data_vars = {}
            coords = {}
            
            for key, value in data.items():
                if isinstance(value, (xr.DataArray, np.ndarray)):
                    data_vars[key] = value
                elif key in ['time', 'lat', 'lon', 'latitude', 'longitude']:
                    coords[key] = value
            
            return xr.Dataset(data_vars, coords=coords)
        else:
            raise ValueError(f"Cannot convert {type(data)} to xarray Dataset")
    
    def _validate_format_compliance(self,
                                  dataset: xr.Dataset,
                                  source: DataSource,
                                  variable_mapping: Optional[Dict[str, str]]) -> List[ExternalValidationIssue]:
        """
        Validate CF convention compliance and format standards.
        
        Parameters:
            dataset: Dataset to validate
            source: Data source type
            variable_mapping: Variable name mapping
            
        Returns:
            List of format compliance issues
        """
        issues = []
        
        # Check for CF Conventions attribute
        if self.config.require_cf_conventions:
            conventions = dataset.attrs.get('Conventions', '')
            if not conventions:
                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.FORMAT_COMPLIANCE,
                    severity=ValidationSeverity.ERROR,
                    message="Missing required 'Conventions' global attribute",
                    suggestion="Add 'Conventions' attribute with CF version (e.g., 'CF-1.8')",
                    reference="CF Conventions Section 2.6.2"
                ))
            elif 'CF' not in conventions:
                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.FORMAT_COMPLIANCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Conventions attribute '{conventions}' does not mention CF",
                    expected="CF-1.6 or later",
                    actual=conventions
                ))
            else:
                # Check CF version
                cf_version_match = re.search(r'CF-(\d+\.?\d*)', conventions)
                if cf_version_match:
                    cf_version = cf_version_match.group(1)
                    if float(cf_version) < float(self.config.cf_version_min):
                        issues.append(ExternalValidationIssue(
                            category=ValidationCategory.FORMAT_COMPLIANCE,
                            severity=ValidationSeverity.WARNING,
                            message=f"CF version {cf_version} is older than recommended minimum {self.config.cf_version_min}",
                            suggestion=f"Update to CF-{self.config.cf_version_min} or later"
                        ))
        
        # Check coordinate variables
        required_coords = ['time']
        spatial_coords = ['lat', 'latitude', 'lon', 'longitude', 'y', 'x']
        
        for coord in required_coords:
            coord_found = any(alias in dataset.coords for alias in [coord])
            if not coord_found and coord == 'time':
                # Check for time-related coordinates
                time_aliases = ['time', 't', 'time_counter']
                time_found = any(alias in dataset.coords for alias in time_aliases)
                if not time_found:
                    issues.append(ExternalValidationIssue(
                        category=ValidationCategory.FORMAT_COMPLIANCE,
                        severity=ValidationSeverity.ERROR,
                        message="No time coordinate found",
                        suggestion="Add time coordinate with appropriate units and calendar",
                        reference="CF Conventions Section 4.4"
                    ))
        
        # Check for spatial coordinates
        spatial_found = any(coord in dataset.coords for coord in spatial_coords)
        if not spatial_found:
            issues.append(ExternalValidationIssue(
                category=ValidationCategory.FORMAT_COMPLIANCE,
                severity=ValidationSeverity.WARNING,
                message="No recognizable spatial coordinates found",
                suggestion="Add latitude/longitude or x/y coordinates",
                reference="CF Conventions Section 4"
            ))
        
        # Check variable attributes for standard names
        if self.config.require_standard_names:
            for var_name, var in dataset.data_vars.items():
                if 'standard_name' not in var.attrs:
                    issues.append(ExternalValidationIssue(
                        category=ValidationCategory.FORMAT_COMPLIANCE,
                        severity=ValidationSeverity.WARNING,
                        message=f"Variable '{var_name}' missing standard_name attribute",
                        variable=var_name,
                        suggestion="Add CF standard_name attribute",
                        reference="CF Standard Names Table"
                    ))
        
        return issues
    
    def _validate_physical_consistency(self,
                                     dataset: xr.Dataset,
                                     source: DataSource,
                                     variable_mapping: Optional[Dict[str, str]]) -> List[ExternalValidationIssue]:
        """
        Validate physical consistency of variable values.
        
        Parameters:
            dataset: Dataset to validate
            source: Data source type  
            variable_mapping: Variable name mapping
            
        Returns:
            List of physical consistency issues
        """
        issues = []
        
        # Get variable definitions for validation
        for var_name, var in dataset.data_vars.items():
            # Map to internal variable if possible
            internal_name = None
            if variable_mapping and var_name in variable_mapping:
                internal_name = variable_mapping[var_name]
            
            var_def = None
            if internal_name:
                var_def = self.schema.get_variable(internal_name)
            
            # Check value ranges
            var_data = var.values
            
            if np.issubdtype(var_data.dtype, np.number):
                finite_data = var_data[np.isfinite(var_data)]
                
                if len(finite_data) > 0:
                    var_min = np.min(finite_data)
                    var_max = np.max(finite_data)
                    
                    # Check against variable definition ranges
                    if var_def:
                        if var_def.valid_min is not None and var_min < var_def.valid_min:
                            issues.append(ExternalValidationIssue(
                                category=ValidationCategory.PHYSICAL_CONSISTENCY,
                                severity=ValidationSeverity.ERROR,
                                message=f"Variable '{var_name}' has values below valid minimum",
                                variable=var_name,
                                expected=f">= {var_def.valid_min}",
                                actual=f"min = {var_min}",
                                suggestion="Check units and data processing"
                            ))
                        
                        if var_def.valid_max is not None and var_max > var_def.valid_max:
                            issues.append(ExternalValidationIssue(
                                category=ValidationCategory.PHYSICAL_CONSISTENCY,
                                severity=ValidationSeverity.ERROR,
                                message=f"Variable '{var_name}' has values above valid maximum",
                                variable=var_name,
                                expected=f"<= {var_def.valid_max}",
                                actual=f"max = {var_max}",
                                suggestion="Check units and data processing"
                            ))
                    
                    # Check for suspicious values (very large/small)
                    if np.abs(var_min) > 1e10 or np.abs(var_max) > 1e10:
                        issues.append(ExternalValidationIssue(
                            category=ValidationCategory.PHYSICAL_CONSISTENCY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Variable '{var_name}' has extremely large values",
                            variable=var_name,
                            actual=f"range: [{var_min}, {var_max}]",
                            suggestion="Verify units and scaling factors"
                        ))
        
        return issues
    
    def _validate_temporal_quality(self,
                                 dataset: xr.Dataset,
                                 source: DataSource,
                                 variable_mapping: Optional[Dict[str, str]]) -> List[ExternalValidationIssue]:
        """
        Validate temporal coordinate quality and consistency.
        
        Parameters:
            dataset: Dataset to validate
            source: Data source type
            variable_mapping: Variable name mapping
            
        Returns:
            List of temporal quality issues
        """
        issues = []
        
        # Find time coordinate
        time_coord = None
        time_name = None
        for coord_name in ['time', 't', 'time_counter']:
            if coord_name in dataset.coords:
                time_coord = dataset.coords[coord_name]
                time_name = coord_name
                break
        
        if time_coord is None:
            issues.append(ExternalValidationIssue(
                category=ValidationCategory.TEMPORAL_QUALITY,
                severity=ValidationSeverity.ERROR,
                message="No time coordinate found for temporal validation",
                suggestion="Add time coordinate to dataset"
            ))
            return issues
        
        # Check monotonicity
        if self.config.require_monotonic_time and len(time_coord) > 1:
            time_values = time_coord.values
            
            elapsed = self._elapsed_seconds(time_coord)
            time_diffs = np.diff(time_values if elapsed is None else elapsed)
            
            non_increasing = np.sum(time_diffs <= 0)
            if non_increasing > 0:
                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.TEMPORAL_QUALITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Time coordinate is not monotonic: {non_increasing} non-increasing steps",
                    variable=time_name,
                    suggestion="Sort data by time or fix time coordinate"
                ))
        
        # Check for large temporal gaps of decoded times (numbers without decoded units have no
        # hours to compare with the threshold).
        elapsed = self._elapsed_seconds(time_coord) if len(time_coord) > 1 else None
        if elapsed is not None:
            # References: CF 1.13 time coordinates; WMO quality control.
            # Mathematical expression: Delta t_h = Delta t_s / S_h.
            # Parameter meanings: Delta t_s is a gap between consecutive records; S_h is seconds/hour.
            # Units: Delta t_s in s (datetime64 or cftime differences); thresholds in hours.
            # Domain/constraints: S_h > 0 from the physical-constant SSOT; an unsupported time value
            #   type raises (the category handler records it as a failed check).
            # Sign convention: positive Delta t_h advances forward in time.
            seconds_per_hour = float(get_physical_constant("seconds_per_hour"))
            max_gap = float(np.max(np.diff(elapsed)) / seconds_per_hour)
            if max_gap > self.config.max_time_gap_hours:
                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.TEMPORAL_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Large temporal gap detected: {max_gap:.1f} hours",
                    variable=time_name,
                    expected=f"<= {self.config.max_time_gap_hours} hours",
                    actual=f"{max_gap:.1f} hours",
                    suggestion="Check for missing time steps"
                ))

        return issues
    
    @staticmethod
    def _elapsed_seconds(time_coord: xr.DataArray) -> Optional[np.ndarray]:
        """Seconds of every record since the first for decoded times (datetime64 or cftime).

        Undecoded numeric times return None (their unit is not known here); any other value type
        raises TypeError.
        """
        values = np.asarray(time_coord.values)
        if values.dtype.kind == "M":
            return (values - values[0]).astype("timedelta64[ns]").astype(np.int64) / 1.0e9
        if values.dtype.kind == "O" and values.size and isinstance(values[0], cftime.datetime):
            return np.array([(value - values[0]).total_seconds() for value in values], dtype=float)
        if values.dtype.kind in "fiu":
            return None
        raise TypeError(f"Unsupported time coordinate values of dtype {values.dtype}.")

    def _validate_spatial_quality(self,
                                dataset: xr.Dataset,
                                source: DataSource,
                                variable_mapping: Optional[Dict[str, str]]) -> List[ExternalValidationIssue]:
        """
        Validate spatial coordinate quality and coverage.
        
        Parameters:
            dataset: Dataset to validate
            source: Data source type
            variable_mapping: Variable name mapping
            
        Returns:
            List of spatial quality issues
        """
        issues = []
        
        # Find spatial coordinates
        lon_coord = None
        lat_coord = None
        
        for lon_name in ['lon', 'longitude', 'x', 'XLONG']:
            if lon_name in dataset.coords:
                lon_coord = dataset.coords[lon_name]
                break
        
        for lat_name in ['lat', 'latitude', 'y', 'XLAT']:
            if lat_name in dataset.coords:
                lat_coord = dataset.coords[lat_name]
                break
        
        if lon_coord is None or lat_coord is None:
            issues.append(ExternalValidationIssue(
                category=ValidationCategory.SPATIAL_QUALITY,
                severity=ValidationSeverity.WARNING,
                message="Spatial coordinates not found or not recognized",
                suggestion="Ensure latitude/longitude coordinates are properly named"
            ))
            return issues
        
        # Check coordinate ranges
        lon_values = lon_coord.values
        lat_values = lat_coord.values
        
        # Handle 2D coordinate arrays
        if lon_values.ndim > 1:
            lon_min, lon_max = np.nanmin(lon_values), np.nanmax(lon_values)
            lat_min, lat_max = np.nanmin(lat_values), np.nanmax(lat_values)
        else:
            lon_min, lon_max = np.min(lon_values), np.max(lon_values)
            lat_min, lat_max = np.min(lat_values), np.max(lat_values)
        
        # Validate longitude range
        if lon_min < -360 or lon_max > 360:
            issues.append(ExternalValidationIssue(
                category=ValidationCategory.SPATIAL_QUALITY,
                severity=ValidationSeverity.ERROR,
                message=f"Longitude values outside valid range: [{lon_min}, {lon_max}]",
                expected="[-360, 360]",
                actual=f"[{lon_min}, {lon_max}]",
                suggestion="Check longitude coordinate units and values"
            ))
        
        # Validate latitude range
        if lat_min < -90 or lat_max > 90:
            issues.append(ExternalValidationIssue(
                category=ValidationCategory.SPATIAL_QUALITY,
                severity=ValidationSeverity.ERROR,
                message=f"Latitude values outside valid range: [{lat_min}, {lat_max}]",
                expected="[-90, 90]", 
                actual=f"[{lat_min}, {lat_max}]",
                suggestion="Check latitude coordinate units and values"
            ))
        
        # Check global coverage if required
        if self.config.require_global_coverage:
            lon_span = lon_max - lon_min
            lat_span = lat_max - lat_min
            
            lon_span_min = self.config.global_longitude_span_min
            lat_span_min = self.config.global_latitude_span_min
            if lon_span < lon_span_min:
                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.SPATIAL_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Longitude coverage ({lon_span:.1f}deg) may not be global",
                    expected=f">= {lon_span_min}deg",
                    actual=f"{lon_span:.1f}deg"
                ))

            if lat_span < lat_span_min:
                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.SPATIAL_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Latitude coverage ({lat_span:.1f}deg) may not be global",
                    expected=f">= {lat_span_min}deg",
                    actual=f"{lat_span:.1f}deg"
                ))
        
        return issues
    
    def _validate_data_completeness(self,
                                  dataset: xr.Dataset,
                                  source: DataSource,
                                  variable_mapping: Optional[Dict[str, str]]) -> List[ExternalValidationIssue]:
        """
        Validate data completeness and missing value patterns.
        
        Parameters:
            dataset: Dataset to validate
            source: Data source type
            variable_mapping: Variable name mapping
            
        Returns:
            List of data completeness issues
        """
        issues = []
        
        for var_name, var in dataset.data_vars.items():
            # Count missing values
            if hasattr(var, 'isnull'):
                missing_count = var.isnull().sum().item()
            else:
                # Handle numpy arrays
                var_values = var.values
                missing_count = np.isnan(var_values).sum() if np.issubdtype(var_values.dtype, np.floating) else 0
            
            total_count = var.size
            missing_fraction = missing_count / total_count if total_count > 0 else 0
            
            if missing_fraction > self.config.max_missing_fraction:
                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.DATA_COMPLETENESS,
                    severity=(
                        ValidationSeverity.ERROR
                        if missing_fraction > self.config.missing_fraction_error
                        else ValidationSeverity.WARNING
                    ),
                    message=f"Variable '{var_name}' has high fraction of missing data: {missing_fraction:.2%}",
                    variable=var_name,
                    expected=f"<= {self.config.max_missing_fraction:.1%}",
                    actual=f"{missing_fraction:.2%}",
                    suggestion="Check data processing and quality control"
                ))
        
        return issues
    
    def _validate_statistical_outliers(self,
                                     dataset: xr.Dataset,
                                     source: DataSource,
                                     variable_mapping: Optional[Dict[str, str]]) -> List[ExternalValidationIssue]:
        """
        Detect statistical outliers in data variables.
        
        Parameters:
            dataset: Dataset to validate
            source: Data source type
            variable_mapping: Variable name mapping
            
        Returns:
            List of statistical outlier issues
        """
        issues = []
        
        for var_name, var in dataset.data_vars.items():
            var_values = var.values
            
            if not np.issubdtype(var_values.dtype, np.number):
                continue
            
            # Remove missing values for statistics
            finite_values = var_values[np.isfinite(var_values)]

            # A spread needs two values; no further minimum sample size is assumed (see below).
            if finite_values.size < 2:
                continue

            mean = np.mean(finite_values)
            std = np.std(finite_values)

            if std == 0:  # Constant values
                continue

            # References: Iglewicz and Hoaglin (1993) How to Detect and Handle Outliers (z-score
            #   rule); Shiffler (1988) Am. Stat. 42(1):79-80 (largest attainable z-score).
            # Mathematical expression: z_i = |x_i - mean(x)| / s with s = sqrt(sum_i (x_i - mean(x))^2 / n);
            #   f = #{i : z_i > k} / n; WARNING when f > f_w, ERROR when f >= f_e; max_i z_i <= sqrt(n - 1).
            # Parameter meanings: x_i are the n finite values of one field; s is their population
            #   standard deviation; k = outlier_threshold; f_w and f_e are outlier_fraction_warning and
            #   outlier_fraction_error. Since max z_i <= sqrt(n - 1), fewer than 1 + k^2 values cannot
            #   contain an outlier, so the check needs no minimum sample size beyond n >= 2.
            # Units: x_i in the units of the field; z_i, k, f, f_w and f_e dimensionless.
            # Domain/constraints: n >= 2 and s > 0 (constant fields are skipped); 0 <= f <= 1.
            # Sign convention: z_i is an absolute deviation; values on either side of the mean count.
            z_scores = np.abs((finite_values - mean) / std)
            outlier_count = np.sum(z_scores > self.config.outlier_threshold)
            outlier_fraction = outlier_count / finite_values.size

            if outlier_fraction > self.config.outlier_fraction_warning:
                severity = (
                    ValidationSeverity.ERROR
                    if outlier_fraction >= self.config.outlier_fraction_error
                    else ValidationSeverity.WARNING
                )

                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.STATISTICAL_OUTLIERS,
                    severity=severity,
                    message=f"Variable '{var_name}' has {outlier_fraction:.2%} statistical outliers",
                    variable=var_name,
                    expected=f"<= {self.config.outlier_fraction_warning:.2%}",
                    actual=f"{outlier_fraction:.2%}",
                    suggestion="Review data for measurement errors or extreme events"
                ))
        
        return issues
    
    def _validate_metadata_quality(self,
                                 dataset: xr.Dataset,
                                 source: DataSource,
                                 variable_mapping: Optional[Dict[str, str]]) -> List[ExternalValidationIssue]:
        """
        Validate metadata completeness and quality.
        
        Parameters:
            dataset: Dataset to validate
            source: Data source type
            variable_mapping: Variable name mapping
            
        Returns:
            List of metadata quality issues
        """
        issues = []
        
        # Check global attributes
        recommended_global_attrs = [
            'title', 'source', 'institution', 'comment', 'references'
        ]
        
        missing_attrs = []
        for attr in recommended_global_attrs:
            if attr not in dataset.attrs:
                missing_attrs.append(attr)
        
        if missing_attrs:
            issues.append(ExternalValidationIssue(
                category=ValidationCategory.METADATA_QUALITY,
                severity=ValidationSeverity.INFO,
                message=f"Missing recommended global attributes: {', '.join(missing_attrs)}",
                suggestion="Add descriptive global attributes for documentation"
            ))
        
        # Check variable attributes
        for var_name, var in dataset.data_vars.items():
            recommended_var_attrs = ['long_name', 'units']
            missing_var_attrs = []
            
            for attr in recommended_var_attrs:
                if attr not in var.attrs:
                    missing_var_attrs.append(attr)
            
            if missing_var_attrs:
                issues.append(ExternalValidationIssue(
                    category=ValidationCategory.METADATA_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Variable '{var_name}' missing attributes: {', '.join(missing_var_attrs)}",
                    variable=var_name,
                    suggestion="Add descriptive variable attributes"
                ))
        
        return issues
    
    def _count_by_severity(self, issues: List[ExternalValidationIssue]) -> Dict[str, int]:
        """Count issues by severity level."""
        counts = {severity.value: 0 for severity in ValidationSeverity}
        for issue in issues:
            counts[issue.severity.value] += 1
        return counts
    
    def _count_by_category(self, issues: List[ExternalValidationIssue]) -> Dict[str, int]:
        """Count issues by category."""
        counts = {category.value: 0 for category in ValidationCategory}
        for issue in issues:
            counts[issue.category.value] += 1
        return counts
    
    def generate_report_html(self, report: ValidationReport, issues: List[ExternalValidationIssue]) -> str:
        """
        Generate HTML validation report.
        
        Parameters:
            report: Basic validation report
            issues: Detailed validation issues
            
        Returns:
            HTML report string
        """
        html = f"""
        <html>
        <head><title>Data Validation Report</title></head>
        <body>
        <h1>Data Validation Report</h1>
        <p><strong>Validation Time:</strong> {report.validation_time}</p>
        <p><strong>Dataset CF conformance checked:</strong> No</p>
        <p><strong>CF validation scope:</strong> {report.cf_validation_scope}</p>
        
        <h2>Summary</h2>
        <p>Total Issues: {len(issues)}</p>
        
        <h2>Issues by Severity</h2>
        <ul>
        """
        
        severity_counts = self._count_by_severity(issues)
        for severity, count in severity_counts.items():
            if count > 0:
                html += f"<li>{severity.title()}: {count}</li>"
        
        html += """
        </ul>
        
        <h2>Detailed Issues</h2>
        """
        
        for issue in issues:
            html += f"""
            <div style="margin: 10px; padding: 10px; border: 1px solid #ccc;">
            <h4>{issue.severity.value.title()}: {issue.category.value.replace('_', ' ').title()}</h4>
            <p><strong>Message:</strong> {issue.message}</p>
            """
            
            if issue.variable:
                html += f"<p><strong>Variable:</strong> {issue.variable}</p>"
            if issue.expected:
                html += f"<p><strong>Expected:</strong> {issue.expected}</p>"
            if issue.actual:
                html += f"<p><strong>Actual:</strong> {issue.actual}</p>"
            if issue.suggestion:
                html += f"<p><strong>Suggestion:</strong> {issue.suggestion}</p>"
                
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html

