"""
Unified Validation Result Models

This module defines the standardized Pydantic models for validation results,
ensuring consistent structure across all validation rules and layers.

Mathematical Framework:
    All validation results follow a standardized schema that supports:
    - Hierarchical validation (L1-L4 layers)
    - Domain-specific tagging and categorization
    - Statistical metrics with proper units and metadata
    - CF-1.8 compliant variable naming and units
    
References:
    - Pydantic documentation: https://pydantic-docs.helpmanual.io/
    - CF-1.8 Conventions: https://cfconventions.org/
    - RELIC validation specification
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing_extensions import TypedDict


class ValidationLevel(str, Enum):
    """
    Validation result severity levels.
    
    Follows standard logging levels with validation-specific semantics:
        INFO: Informational result, no action needed
        WARNING: Potential issue that should be reviewed
        ERROR: Definite problem that may affect results
        CRITICAL: Severe issue requiring immediate attention
    """
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationCode(str, Enum):
    """
    Standardized validation codes with namespaced categories.
    
    Naming Convention:
        [DOMAIN]_[CATEGORY]_[SPECIFIC_ISSUE]
        
    Categories:
        CF_*: CF-1.8 convention compliance
        MASS_BALANCE_*: Mass/energy conservation
        SCORE_*: Probabilistic scoring metrics  
        PARAM_*: Parameter validation
        COUPLING_*: Inter-domain coupling
        INTERFACE_*: Module interface validation
        *_OK: Success codes indicating passing validation
    """
    # Success Codes (Passing Validation)
    CF_COMPLIANCE_OK = "CF_COMPLIANCE_OK"
    CF_UNIT_OK = "CF_UNIT_OK"
    CF_NAME_OK = "CF_NAME_OK"
    CF_STANDARD_NAME_OK = "CF_STANDARD_NAME_OK"
    CF_UNIT_CONVERSION_OK = "CF_UNIT_CONVERSION_OK"
    
    MASS_BALANCE_OK_CARBON = "MASS_BALANCE_OK_CARBON"
    MASS_BALANCE_OK_NITROGEN = "MASS_BALANCE_OK_NITROGEN"
    MASS_BALANCE_OK_WATER = "MASS_BALANCE_OK_WATER"
    MASS_BALANCE_OK = "MASS_BALANCE_OK"  # Generic mass balance OK
    ENERGY_BALANCE_OK = "ENERGY_BALANCE_OK"
    
    SCORE_CRPS_OK = "SCORE_CRPS_OK"
    PIT_UNIFORMITY_OK = "PIT_UNIFORMITY_OK"
    BRIER_SCORE_OK = "BRIER_SCORE_OK"
    COVERAGE_PROBABILITY_OK = "COVERAGE_PROBABILITY_OK"
    
    PARAM_BOUNDS_OK = "PARAM_BOUNDS_OK"
    PARAM_SCHEMA_OK = "PARAM_SCHEMA_OK"
    PARAM_TYPE_OK = "PARAM_TYPE_OK"
    PARAM_CORRELATION_OK = "PARAM_CORRELATION_OK"
    PARAM_LOG_SCALE_OK = "PARAM_LOG_SCALE_OK"
    PARAM_OK = "PARAM_OK"  # Generic parameter validation OK
    
    COUPLING_CONTRACT_OK = "COUPLING_CONTRACT_OK"
    INTERFACE_VARIABLE_OK = "INTERFACE_VARIABLE_OK"
    INTERFACE_TYPE_OK = "INTERFACE_TYPE_OK"
    STRESS_FACTOR_OK = "STRESS_FACTOR_OK"
    
    GENERIC_VALIDATION_OK = "GENERIC_VALIDATION_OK"
    NUMERICAL_STABILITY_OK = "NUMERICAL_STABILITY_OK"
    
    # CF-1.8 Convention Issues
    CF_UNIT_MISMATCH = "CF_UNIT_MISMATCH"
    CF_NAME_MISSING = "CF_NAME_MISSING"
    CF_STANDARD_NAME_INVALID = "CF_STANDARD_NAME_INVALID"
    CF_UNIT_CONVERSION_FAILED = "CF_UNIT_CONVERSION_FAILED"
    
    # Conservation Law Violations
    MASS_BALANCE_FAIL_CARBON = "MASS_BALANCE_FAIL_CARBON"
    MASS_BALANCE_FAIL_NITROGEN = "MASS_BALANCE_FAIL_NITROGEN"
    MASS_BALANCE_FAIL_WATER = "MASS_BALANCE_FAIL_WATER"
    ENERGY_BALANCE_FAIL = "ENERGY_BALANCE_FAIL"
    
    # Mass balance warnings (less severe)
    MASS_BALANCE_WARN_CARBON = "MASS_BALANCE_WARN_CARBON"
    MASS_BALANCE_WARN_NITROGEN = "MASS_BALANCE_WARN_NITROGEN"
    MASS_BALANCE_WARN_WATER = "MASS_BALANCE_WARN_WATER"
    ENERGY_BALANCE_WARN = "ENERGY_BALANCE_WARN"
    
    # Probabilistic Scoring Issues
    SCORE_CRPS_HIGH = "SCORE_CRPS_HIGH"
    SCORE_CRPS_CRITICAL = "SCORE_CRPS_CRITICAL"
    PIT_UNIFORMITY_REJECT_KS = "PIT_UNIFORMITY_REJECT_KS"
    PIT_UNIFORMITY_REJECT_CVM = "PIT_UNIFORMITY_REJECT_CVM"
    PIT_UNIFORMITY_REJECT_CHI2 = "PIT_UNIFORMITY_REJECT_CHI2"
    BRIER_SCORE_HIGH = "BRIER_SCORE_HIGH"
    COVERAGE_PROBABILITY_FAIL = "COVERAGE_PROBABILITY_FAIL"
    
    # Parameter Validation Issues
    PARAM_BOUND_VIOLATION = "PARAM_BOUND_VIOLATION"
    PARAM_SCHEMA_MISSING = "PARAM_SCHEMA_MISSING"
    PARAM_TYPE_MISMATCH = "PARAM_TYPE_MISMATCH"
    PARAM_CORRELATION_VIOLATION = "PARAM_CORRELATION_VIOLATION"
    PARAM_LOG_SCALE_INCONSISTENT = "PARAM_LOG_SCALE_INCONSISTENT"
    
    # Coupling and Interface Issues
    COUPLING_CONTRACT_VIOLATION = "COUPLING_CONTRACT_VIOLATION"
    INTERFACE_VARIABLE_MISSING = "INTERFACE_VARIABLE_MISSING"
    INTERFACE_TYPE_MISMATCH = "INTERFACE_TYPE_MISMATCH"
    STRESS_FACTOR_OUT_OF_BOUNDS = "STRESS_FACTOR_OUT_OF_BOUNDS"
    
    # Generic validation codes
    GENERIC_VALIDATION_FAILED = "GENERIC_VALIDATION_FAILED"
    MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"
    NUMERICAL_INSTABILITY = "NUMERICAL_INSTABILITY"


class ValidationDetail(BaseModel):
    """
    Detailed diagnostic information for a validation result.
    
    Contains specific values, thresholds, locations, and metadata
    that help users understand and resolve validation issues.
    
    Fields:
        variable_name: Name of the variable involved
        cf_standard_name: CF-1.8 standard name if applicable
        units: Physical units of the variable/metric
        value: Actual measured/computed value
        expected: Expected value or acceptable range
        threshold: Threshold used for pass/fail determination  
        location: Spatial/temporal location information
        statistics: Additional statistical information
        metadata: Extra diagnostic data
    """
    variable_name: Optional[str] = Field(default=None, description="Variable name")
    cf_standard_name: Optional[str] = Field(default=None, description="CF-1.8 standard name")
    units: Optional[str] = Field(default=None, description="Physical units")
    
    # Numerical values (allow both scalars and tensors)
    value: Optional[Union[float, int, str]] = Field(default=None, description="Actual value")
    expected: Optional[Union[float, int, str, tuple]] = Field(default=None, description="Expected value/range")
    threshold: Optional[Union[float, int]] = Field(default=None, description="Pass/fail threshold")
    
    # Location information
    location: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Spatial/temporal indices")
    
    # Statistical metrics (allowing complex types for comprehensive validation data)
    statistics: Optional[Dict[str, Union[float, int, str, bool, list, dict, Any]]] = Field(
        default_factory=dict, 
        description="Statistical measures (p-value, confidence intervals, lists, nested data, etc.)"
    )
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional diagnostic information"
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow torch.Tensor and other types


class ValidationResultModel(BaseModel):
    """
    Standardized validation result model.
    
    This is the core result structure returned by all validation rules,
    ensuring consistency across the entire validation framework.
    
    Mathematical Expression:
        Each result represents an evaluation of a validation rule r:
        result = r(state, parameters, metadata) -> ValidationResultModel
        
    Fields:
        level: Severity level (INFO/WARNING/ERROR/CRITICAL)
        code: Standardized validation code
        message: Human-readable description
        passed: Explicit pass/fail indicator
        details: List of detailed diagnostic information
        layer: Validation layer (L1/L2/L3/L4)  
        scope: Module/functor that generated this result
        tags: Categorization tags for filtering and grouping
        timestamp: When the validation was performed
        metrics: Quantitative metrics for the validation
    """
    level: ValidationLevel = Field(..., description="Validation severity level")
    code: ValidationCode = Field(..., description="Standardized validation code")
    message: str = Field(..., description="Human-readable validation message")
    passed: bool = Field(
        default=True,
        description="Explicit pass/fail indicator (True=passed, False=failed)"
    )
    
    details: List[ValidationDetail] = Field(
        default_factory=list,
        description="Detailed diagnostic information"
    )
    
    layer: str = Field(..., description="Validation layer (L1/L2/L3/L4)")
    scope: str = Field(..., description="Module/functor scope")
    
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization (cf, conservation, coupling, etc.)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Validation execution timestamp"
    )
    
    metrics: Dict[str, Union[float, int]] = Field(
        default_factory=dict,
        description="Quantitative validation metrics"
    )
    
    @field_validator('tags', mode='before')
    @classmethod
    def normalize_tags(cls, v):
        """Normalize tags to lowercase for consistency."""
        if isinstance(v, (list, tuple)):
            return [tag.lower() if isinstance(tag, str) else tag for tag in v]
        return v
    
    @field_validator('scope')
    @classmethod
    def validate_scope(cls, v):
        """Ensure scope follows module.function or functor_name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Scope must be a non-empty string")
        return v
    
    def model_post_init(self, __context):
        """Auto-determine passed flag based on level and code if not explicitly set."""
        # Success codes always mean passed
        if "_OK" in str(self.code):
            self.passed = True
        # Error/Critical levels always mean failed
        elif self.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]:
            self.passed = False
        # Info level with non-failure code means passed
        elif self.level == ValidationLevel.INFO and not "_FAIL" in str(self.code):
            self.passed = True
    
    def is_passing(self) -> bool:
        """Check if validation result indicates a passing state."""
        return self.passed
    
    def is_failing(self) -> bool:
        """Check if validation result indicates a failing state."""
        return self.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
    
    def add_detail(self, 
                   variable_name: str,
                   value: Union[float, str], 
                   **kwargs) -> None:
        """Convenience method to add validation details."""
        detail = ValidationDetail(
            variable_name=variable_name,
            value=value,
            **kwargs
        )
        self.details.append(detail)


class _ReportSummary(TypedDict):
    """Fixed structure of counts derived from the report's own results."""

    total: int
    by_level: Dict[str, int]
    by_layer: Dict[str, int]


class ValidationReportModel(BaseModel):
    """
    Comprehensive validation report containing multiple results.
    
    Aggregates validation results from multiple rules and provides
    summary statistics and categorization for easy analysis.
    
    Fields:
        results: List of individual validation results
        summary: Aggregated statistics by level/layer
        cf_summary: bounded CF variable-metadata summary
        conservation_summary: Conservation law compliance summary
        execution_metadata: Information about the validation run
    """
    results: List[ValidationResultModel] = Field(
        default_factory=list,
        description="Individual validation results"
    )
    
    summary: _ReportSummary = Field(
        default_factory=lambda: _ReportSummary(total=0, by_level={}, by_layer={}),
        description="Summary statistics by level and layer"
    )
    
    cf_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="bounded CF variable-metadata summary"
    )
    
    conservation_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conservation law compliance summary" 
    )
    
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation run metadata"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Report generation timestamp"
    )
    
    def model_post_init(self, __context):
        """Initialize summary after model creation."""
        self._update_summary()
    
    def __len__(self) -> int:
        """Return number of validation results."""
        return len(self.results)
    
    def add_result(self, result: ValidationResultModel) -> None:
        """Add a validation result to the report."""
        self.results.append(result)
        self._update_summary()
    
    def get_results_by_level(self, level: ValidationLevel) -> List[ValidationResultModel]:
        """Filter results by validation level."""
        return [r for r in self.results if r.level == level]
    
    def get_results_by_layer(self, layer: str) -> List[ValidationResultModel]:
        """Filter results by validation layer."""
        return [r for r in self.results if r.layer == layer]
    
    def get_results_by_tag(self, tag: str) -> List[ValidationResultModel]:
        """Filter results by tag."""
        tag_lower = tag.lower()
        return [r for r in self.results if tag_lower in r.tags]
    
    def has_errors(self) -> bool:
        """Check if report contains any ERROR or CRITICAL results."""
        return any(r.is_failing() for r in self.results)
    
    def has_warnings(self) -> bool:
        """Check if report contains any WARNING results."""
        return any(r.level == ValidationLevel.WARNING for r in self.results)
    
    def _update_summary(self) -> None:
        """Update summary statistics after adding results."""
        # Count by level
        level_counts = {}
        for level in ValidationLevel:
            level_counts[level.value] = len(self.get_results_by_level(level))
        
        # Count by layer  
        layer_counts = {}
        layers = set(r.layer for r in self.results)
        for layer in layers:
            layer_counts[layer] = len(self.get_results_by_layer(layer))
        
        self.summary = {
            "by_level": level_counts,
            "by_layer": layer_counts,
            "total": len(self.results)
        }
        
        # Update CF summary
        cf_results = self.get_results_by_tag("cf")
        self.cf_summary = {
            "total_cf_validations": len(cf_results),
            "cf_errors": len([r for r in cf_results if r.is_failing()]),
            "cf_warnings": len([r for r in cf_results if r.level == ValidationLevel.WARNING])
        }
        
        # Update conservation summary
        conservation_results = self.get_results_by_tag("conservation")
        self.conservation_summary = {
            "total_conservation_checks": len(conservation_results),
            "conservation_failures": len([r for r in conservation_results if r.is_failing()]),
            "conservation_warnings": len([r for r in conservation_results if r.level == ValidationLevel.WARNING])
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return self.model_dump(mode="json")


@dataclass
class ValidationContext:
    """
    Context information passed to validation rules.

    Provides additional information that validation rules may need
    to perform their checks, such as state manager reference,
    configuration parameters, and execution context.
    """
    state_manager: Optional[Any] = None
    config: Optional[Dict[str, Any]] = None
    execution_layer: Optional[str] = None
    domain_filter: Optional[List[str]] = None
    tolerance_overrides: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    functor_name: Optional[str] = None
    functor_metadata: Optional[Dict[str, Any]] = None
    time: Optional[float] = None
    dt: Optional[float] = None


# Type aliases for convenience
ValidationResults = List[ValidationResultModel]
ValidationReport = ValidationReportModel
ValidationRuleFunction = callable  # Function signature: (context: ValidationContext) -> ValidationResults
