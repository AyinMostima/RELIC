"""
Parameter Validation Component for RELIC

This module provides comprehensive parameter validation including
type checking, bounds enforcement, unit consistency, and scientific
constraint validation.

It serves as the validation layer for the unified parameter management
system, ensuring all parameters meet scientific and technical requirements.

Key Features:
    - Type and format validation
    - Physical bounds checking
    - Unit consistency verification
    - Inter-parameter relationship validation
    - Scientific constraint enforcement
    - Custom validation rules

Design Patterns:
    - Strategy Pattern for validation rules
    - Chain of Responsibility for multi-level validation
    - Specification Pattern for constraint definition

References:
    - CF-1.8 Convention for Climate and Forecast Metadata
    - UDUNITS-2 unit specification standard
    - Gamma, E. et al. (1994). Design Patterns
"""

import re
import inspect
import warnings
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import logging

import torch
from torch import Tensor

from relic.data.standards import dimensionality as dimensionality_module


# Import shared types
from .types import (
    ValidationLevel,
    ValidationResult,
    ValidationContext,
    ParameterValue
)
from .spec import componentwise_bounds, parameter_dtype_is_sequence

# Import ParameterSpec - will be injected at runtime to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .manager import ParameterSpec
else:
    ParameterSpec = None

logger = logging.getLogger(__name__)

# Type aliases
ValidationRule = Callable[[str, Any, 'ParameterSpec', ValidationContext], ValidationResult]


class ValidationType(Enum):
    """Types of parameter validation."""
    TYPE = "type"              # Data type validation
    BOUNDS = "bounds"          # Numerical bounds checking
    UNITS = "units"            # Unit consistency
    RELATIONSHIP = "relationship"  # Inter-parameter constraints
    SCIENTIFIC = "scientific"  # Domain-specific rules
    CUSTOM = "custom"          # User-defined validation


class ParameterValidator:
    """
    Comprehensive parameter validation system.
    
    Provides multi-level validation with type checking, bounds enforcement,
    unit verification, and scientific constraint validation.
    
    This class implements a flexible validation framework that can be
    extended with custom rules while maintaining core scientific validity.
    
    Architecture:
        - Pluggable validation rules
        - Multi-level validation (basic, standard, strict)
        - Context-aware validation
        - Detailed error reporting
        
    Mathematical Model:
        Validation function: V : P x C -> {valid, invalid} x ErrorList
        where P is parameter space, C is context
        Constraint satisfaction:  C(p) where C are constraint predicates
        
    Examples:
        >>> validator = ParameterValidator()
        >>> is_valid, errors = validator.validate_value(
        ...     "temperature", 300.0, temperature_spec
        ... )
    """
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        """Initialize parameter validator.
        
        Parameters:
            level : ValidationLevel
                Default validation level
        """
        # Import ParameterSpec at runtime to avoid circular import
        global ParameterSpec
        if ParameterSpec is None:
            from .manager import ParameterSpec as _ParameterSpec
            ParameterSpec = _ParameterSpec
            
        # Set validation level
        self.validation_level = level
        
        # Validation hooks
        self._validation_hooks: List[Callable] = []
        
        # Custom validators (for backwards consistency)
        self.custom_validators: Dict[str, Callable] = {}
        
        # Built-in validation rules
        self._rules: Dict[ValidationType, List[ValidationRule]] = {
            ValidationType.TYPE: [self._validate_type],
            ValidationType.BOUNDS: [self._validate_bounds],
            ValidationType.UNITS: [self._validate_units],
            ValidationType.RELATIONSHIP: [],
            # No scientific constraint is inferred from key names; bounds carry it.
            ValidationType.SCIENTIFIC: [],
            ValidationType.CUSTOM: []
        }
        
        # Validation cache for performance
        self._validation_cache: Dict[Tuple[str, Any], ValidationResult] = {}
        
        logger.debug("ParameterValidator initialized")
    
    def validate(self, 
                 specs: Dict[str, ParameterSpec],
                 level: Optional[ValidationLevel] = None,
                 context: Optional[ValidationContext] = None) -> List[str]:
        """
        Validate multiple parameters against specifications.
        
        Parameters:
            specs : Dict[str, ParameterSpec]
                Parameter specifications to validate
            level : ValidationLevel
                Validation strictness level
            context : Optional[ValidationContext]
                Additional validation context
                
        Returns:
            List[str]: List of validation errors (empty if valid)
            
        Examples:
            >>> errors = validator.validate(param_specs, ValidationLevel.STRICT)
            >>> if errors:
            ...     print(f"Validation failed: {errors}")
        """
        effective_level = level or self.validation_level
        if effective_level == ValidationLevel.NONE:
            return []
        
        errors = []
        context = context or ValidationContext()
        context.strict_mode = (effective_level == ValidationLevel.STRICT)
        # In STRICT mode, treat warnings as errors to enforce hard validation.
        context.warnings_as_errors = context.warnings_as_errors or context.strict_mode
        
        for key, spec in specs.items():
            result = self.validate_value(
                key, spec.value, spec, effective_level, context
            )
            if not result.is_valid:
                errors.extend(result.errors)
        
        # Validate relationships if standard or strict
        if effective_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            rel_errors = self._validate_relationships(specs, context)
            errors.extend(rel_errors)
        
        return errors
    
    def validate_value(self,
                      key: str,
                      value: ParameterValue,
                      spec: ParameterSpec,
                      level: Optional[ValidationLevel] = None,
                      context: Optional[ValidationContext] = None) -> ValidationResult:
        """
        Validate single parameter value against specification.
        
        Parameters:
            key : str
                Parameter key
            value : ParameterValue
                Value to validate
            spec : ParameterSpec
                Parameter specification
            level : ValidationLevel
                Validation strictness
            context : Optional[ValidationContext]
                Additional context
                
        Returns:
            ValidationResult: (is_valid, errors)
            
        Examples:
            >>> is_valid, errors = validator.validate_value(
            ...     "vcmax25", 60.0, vcmax_spec
            ... )
        """
        effective_level = level or self.validation_level
        cache_key = None
        try:
            cache_key = (key, value, effective_level)
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]
        except TypeError:
            # Unhashable values (e.g., dicts) skip cache
            cache_key = None
        
        errors = []
        warnings: List[str] = []
        error_types: List[str] = []
        context = context or ValidationContext()
        context.strict_mode = (effective_level == ValidationLevel.STRICT)
        # Propagate strictness to warning handling so soft validations escalate in STRICT runs.
        context.warnings_as_errors = context.warnings_as_errors or context.strict_mode
        
        # Apply validation rules based on level
        if effective_level == ValidationLevel.BASIC:
            validation_types = [ValidationType.TYPE, ValidationType.BOUNDS]
        elif effective_level == ValidationLevel.STANDARD:
            validation_types = [
                ValidationType.TYPE,
                ValidationType.BOUNDS,
                ValidationType.UNITS,
                ValidationType.SCIENTIFIC,
                ValidationType.CUSTOM
            ]
        else:  # STRICT
            validation_types = list(ValidationType)
        
        # Run validation rules
        for val_type in validation_types:
            for rule in self._rules.get(val_type, []):
                rule_result = rule(key, value, spec, context)
                if getattr(rule_result, "warnings", None):
                    warnings.extend(rule_result.warnings)
                if not rule_result.is_valid:
                    errors.extend(rule_result.errors)
                    if hasattr(rule_result, "error_types"):
                        error_types.extend(rule_result.error_types)
        
        # Run custom validation from spec
        if hasattr(spec, 'validate_value'):
            spec_errors = spec.validate_value(value)
            errors.extend(spec_errors)
            if spec_errors:
                error_types.extend(["custom"] * len(spec_errors))
        
        result = ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            error_types=error_types
        )
        if cache_key is not None:
            self._validation_cache[cache_key] = result
        
        # Call validation hooks
        self._call_validation_hooks(key, value, spec)
        
        return result
    
    def add_rule(self, 
                 validation_type: ValidationType,
                 rule: ValidationRule) -> None:
        """
        Add custom validation rule.
        
        Parameters:
            validation_type : ValidationType
                Type of validation
            rule : ValidationRule
                Validation function
                
        Examples:
            >>> def my_rule(key, value, spec, context):
            ...     if value < 0:
            ...         return ValidationResult(False, [f"{key} must be positive"])
            ...     return ValidationResult(True, [])
            >>> validator.add_rule(ValidationType.CUSTOM, my_rule)
        """
        self._rules[validation_type].append(rule)
        self._validation_cache.clear()  # Clear cache when rules change
    
    def add_custom_validator(self, name: str, validator_func: Callable) -> None:
        """
        Add a custom validator function.
        
        Parameters:
            name : str
                Name of the validator
            validator_func : Callable
                Validator function that either:
                1. Takes (value) and returns bool (simple validator)
                2. Takes (key, value, spec) and returns ValidationResult (full validator)
                
        Examples:
            >>> def positive_validator(value):
            ...     return value > 0
            >>> validator.add_custom_validator("positive", positive_validator)
            
            >>> def full_validator(key, value, spec):
            ...     return ValidationResult(value > 0, [] if value > 0 else [f"{key} invalid"])
            >>> validator.add_custom_validator("full", full_validator)
        """
        def wrapper(key: str, value: Any, spec: ParameterSpec, context: ValidationContext) -> ValidationResult:
            """Wrapper to handle different validator function signatures."""
            try:
                # Check if validator_func expects full signature (key, value, spec)
                sig = inspect.signature(validator_func)
                param_count = len(sig.parameters)
                
                if param_count >= 3:
                    # Full signature validator - expects (key, value, spec) -> ValidationResult
                    result = validator_func(key, value, spec)
                    if isinstance(result, ValidationResult):
                        return result
                    else:
                        # Convert boolean to ValidationResult
                        return ValidationResult(
                            is_valid=bool(result),
                            errors=[] if result else [f"{key}: Failed {name} validation"]
                        )
                else:
                    # Simple signature validator - expects (value) -> bool
                    is_valid = validator_func(value)
                    return ValidationResult(
                        is_valid=bool(is_valid),
                        errors=[] if is_valid else [f"{key}: Failed {name} validation"]
                    )
            except (TypeError, ValueError, RuntimeError) as e:
                logger.warning("Custom validator %s failed for %s: %s", name, key, e)
                if context.strict_mode:
                    raise
                return ValidationResult(is_valid=False, errors=[f"{key}: Validation error in {name}: {str(e)}"])
        
        self.custom_validators[name] = validator_func
        self.add_rule(ValidationType.CUSTOM, wrapper)
    
    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._validation_cache.clear()
    
    # Built-in validation rules
    
    def _validate_type(self, 
                      key: str,
                      value: Any,
                      spec: ParameterSpec,
                      context: ValidationContext) -> ValidationResult:
        """Validate parameter type."""
        expected_type = spec.dtype
        errors = []
        warnings: List[str] = []
        error_types: List[str] = []
        
        if isinstance(value, Tensor):
            # Check for NaN values
            if torch.isnan(value).any():
                errors.append(f"{key}: Tensor contains NaN values")
                error_types.append("type")
            
            # Check for Inf values
            if torch.isinf(value).any():
                errors.append(f"{key}: Tensor contains Inf values")
                error_types.append("type")
            
            # Check dtype consistency
            if expected_type == 'float32' and value.dtype != torch.float32:
                errors.append(f"{key}: Expected float32 tensor, got {value.dtype}")
                error_types.append("type")
            elif expected_type == 'float64' and value.dtype != torch.float64:
                errors.append(f"{key}: Expected float64 tensor, got {value.dtype}")
                error_types.append("type")
                
        elif expected_type in ['float32', 'float64']:
            if not isinstance(value, (int, float)):
                try:
                    float(value)
                except (TypeError, ValueError):
                    msg = f"{key}: Expected numeric value, got {type(value).__name__}"
                    if context.strict_mode or context.warnings_as_errors:
                        errors.append(msg)
                        error_types.append("type")
                    else:
                        warnings.append(msg)
                
        elif expected_type in ['int32', 'int64']:
            # Note: bool is a subclass of int in Python, but should not be treated as integer here.
            if isinstance(value, bool) or not isinstance(value, int):
                msg = None
                if isinstance(value, float):
                    if value.is_integer():
                        msg = f"{key}: Expected integer value, got float {value}"
                    else:
                        msg = f"{key}: Expected integer value, got non-integer float {value}"
                else:
                    try:
                        cast_value = int(value)
                        msg = f"{key}: Expected integer value, got {type(value).__name__} (cast to {cast_value})"
                    except (TypeError, ValueError, OverflowError):
                        msg = f"{key}: Expected integer value, got {type(value).__name__}"
                if msg:
                    if context.strict_mode or context.warnings_as_errors:
                        errors.append(msg)
                        error_types.append("type")
                    else:
                        warnings.append(msg)
                
        elif expected_type == 'bool':
            if not isinstance(value, bool):
                msg = f"{key}: Expected boolean value, got {type(value).__name__}"
                if context.strict_mode or context.warnings_as_errors:
                    errors.append(msg)
                    error_types.append("type")
                else:
                    warnings.append(msg)
                
        elif expected_type == 'str':
            if not isinstance(value, str):
                msg = f"{key}: Expected string value, got {type(value).__name__}"
                if context.strict_mode or context.warnings_as_errors:
                    errors.append(msg)
                    error_types.append("type")
                else:
                    warnings.append(msg)
        
        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            error_types=error_types,
        )
    
    def _validate_bounds(self,
                        key: str,
                        value: ParameterValue,
                        spec: ParameterSpec,
                        context: ValidationContext) -> ValidationResult:
        """Validate parameter bounds."""
        # A list-valued parameter that declares one (minimum, maximum) pair per
        # component has no scalar interval to order a value against; every
        # candidate is judged by the componentwise rule that
        # ParameterSpec.validate_value applies (spec.componentwise_bounds_errors).
        if (
            parameter_dtype_is_sequence(spec.dtype)
            and componentwise_bounds(getattr(spec, "bounds", None)) is not None
        ):
            return ValidationResult(is_valid=True, errors=[], error_types=[])

        if not isinstance(value, (int, float, Tensor)):
            return ValidationResult(is_valid=True, errors=[], error_types=[])  # Skip bounds check for non-numeric
        
        if isinstance(value, Tensor):
            if value.numel() == 0:
                return ValidationResult(is_valid=True, errors=[], error_types=[])
            if torch.isnan(value).any():
                return ValidationResult(
                    is_valid=False,
                    errors=[f"{key}: Value contains NaN"],
                    error_types=["bounds"],
                )
            if torch.isinf(value).any():
                return ValidationResult(
                    is_valid=False,
                    errors=[f"{key}: Value contains infinite values"],
                    error_types=["bounds"],
                )
            if value.numel() == 1:
                numeric_value = float(value.item())
                tensor_stats = None
            else:
                tensor_min = float(value.min().item())
                tensor_max = float(value.max().item())
                tensor_stats = (tensor_min, tensor_max)
        else:
            numeric_value = float(value)
            tensor_stats = None
        
        # Check for NaN/Inf values first (scalar path only)
        import math
        errors = []
        error_types: List[str] = []
        if tensor_stats is None:
            if math.isnan(numeric_value):
                errors.append(f"{key}: Value is NaN")
                error_types.append("bounds")
                return ValidationResult(is_valid=False, errors=errors, error_types=error_types)
            if math.isinf(numeric_value):
                errors.append(f"{key}: Value is infinite")
                error_types.append("bounds")
                return ValidationResult(is_valid=False, errors=errors, error_types=error_types)
        
        # Handle parameters without bounds (None) or malformed bounds safely
        bounds = getattr(spec, "bounds", None)
        if bounds is None:
            return ValidationResult(is_valid=True, errors=[], error_types=[])

        try:
            min_val, max_val = bounds
        except (TypeError, ValueError):
            errors.append(f"{key}: Invalid bounds specification {bounds}")
            error_types.append("bounds")
            return ValidationResult(is_valid=False, errors=errors, error_types=error_types)

        # Allow open-ended bounds using None
        if tensor_stats is None:
            if min_val is not None and numeric_value < min_val:
                errors.append(f"{key}: Value {numeric_value} outside bounds (below minimum {min_val})")
                error_types.append("bounds")
            if max_val is not None and numeric_value > max_val:
                errors.append(f"{key}: Value {numeric_value} outside bounds (above maximum {max_val})")
                error_types.append("bounds")
        else:
            tensor_min, tensor_max = tensor_stats
            if min_val is not None and tensor_min < min_val:
                errors.append(f"{key}: Tensor min {tensor_min} outside bounds (below minimum {min_val})")
                error_types.append("bounds")
            if max_val is not None and tensor_max > max_val:
                errors.append(f"{key}: Tensor max {tensor_max} outside bounds (above maximum {max_val})")
                error_types.append("bounds")
        
        return ValidationResult(is_valid=(len(errors) == 0), errors=errors, error_types=error_types)
    
    def _validate_units_canonical(
        self,
        key: str,
        spec: ParameterSpec,
        context: ValidationContext,
    ) -> ValidationResult:
        """Validate units through the canonical dimensionality owner."""
        errors: List[str] = []
        warnings: List[str] = []
        error_types: List[str] = []
        units_raw = spec.units

        if not isinstance(units_raw, str) or not units_raw.strip():
            message = f"{key}: Invalid unit specification '{units_raw}'"
            if context.strict_mode or context.warnings_as_errors:
                errors.append(message)
                error_types.append("units")
            else:
                warnings.append(message)
            return ValidationResult(
                is_valid=not errors,
                errors=errors,
                warnings=warnings,
                error_types=error_types,
            )

        units_str = units_raw.strip()
        canonical_dimension = dimensionality_module.dimension_category_from_unit(
            units_str
        )
        if canonical_dimension is None:
            message = f"{key}: Unknown unit '{units_str}'"
            malformed = "_" in units_str or bool(
                re.fullmatch(r"[A-Za-z\\s]+", units_str)
            )
            if context.strict_mode or context.warnings_as_errors or malformed:
                errors.append(message)
                error_types.append("units")
            else:
                warnings.append(message)
        else:
            declared_dimension = str(getattr(spec, "dimension", "") or "").strip()
            if (
                declared_dimension
                and not dimensionality_module.is_declared_dimension_compatible(
                    units_str,
                    declared_dimension,
                )
            ):
                message = (
                    f"{key}: Unit '{units_str}' inconsistent with dimension "
                    f"'{declared_dimension}' (canonical '{canonical_dimension}')"
                )
                if context.strict_mode or context.warnings_as_errors:
                    errors.append(message)
                    error_types.append("units")
                else:
                    warnings.append(message)

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            error_types=error_types,
        )

    def _validate_units(self,
                       key: str,
                       value: ParameterValue,
                       spec: ParameterSpec,
                       context: ValidationContext) -> ValidationResult:
        """Validate parameter units."""
        return self._validate_units_canonical(key, spec, context)

    def _validate_relationships(self,
                               specs: Dict[str, ParameterSpec],
                               context: ValidationContext) -> List[str]:
        """
        Validate inter-parameter relationships.
        
        Parameters:
            specs : Dict[str, ParameterSpec]
                All parameter specifications
            context : ValidationContext
                Validation context
                
        Returns:
            List[str]: Relationship validation errors
        """
        errors = []
        
        # Check Jmax/Vcmax ratio for photosynthesis (only for flux-like capacity parameters)
        def _is_capacity_candidate(spec: ParameterSpec, key: str) -> bool:
            units_lower = str(getattr(spec, "units", "") or "").lower()
            dim_lower = str(getattr(spec, "dimension", "") or "").lower()
            key_lower = key.lower()
            area_term = ("m-2" in units_lower) or ("m2" in units_lower)
            time_term = ("s-1" in units_lower) or ("/s" in units_lower)
            flux_like_units = any(token in units_lower for token in ["umol", "mol", "kg", "g"]) and area_term and time_term
            return flux_like_units or dim_lower == "flux" or "assimilation" in key_lower

        vcmax_candidates: List[str] = []
        jmax_candidates: List[str] = []

        for key in specs:
            key_lower = key.lower()
            if "vcmax" in key_lower and _is_capacity_candidate(specs[key], key):
                vcmax_candidates.append(key)
            elif "jmax" in key_lower and _is_capacity_candidate(specs[key], key):
                jmax_candidates.append(key)

        def _pick_capacity(keys: List[str]) -> Optional[str]:
            for candidate in keys:
                if "25" in candidate.lower():
                    return candidate
            return keys[-1] if keys else None

        vcmax_key = _pick_capacity(vcmax_candidates)
        jmax_key = _pick_capacity(jmax_candidates)

        if vcmax_key and jmax_key:
            try:
                vcmax = float(specs[vcmax_key].value)
                jmax = float(specs[jmax_key].value)
            except (TypeError, ValueError):
                vcmax = 0.0
                jmax = 0.0

            ratio = jmax / vcmax if vcmax > 0 else 0

            if ratio < 1.5:
                msg = f"Jmax/Vcmax ratio {ratio:.2f} unusually low (typical: 1.5-2.5)"
                if context.strict_mode:
                    errors.append(msg)
                else:
                    warnings.warn(msg)
            elif ratio > 3.0:
                msg = f"Jmax/Vcmax ratio {ratio:.2f} unusually high (typical: 1.5-2.5)"
                if context.strict_mode:
                    errors.append(msg)
                else:
                    warnings.warn(msg)

        # Check Q10 values (dimensionless Q10 factors only)
        for key, spec in specs.items():
            key_lower = key.lower()
            if "q10" not in key_lower:
                continue
            if any(term in key_lower for term in ["interval", "temperature_interval", "delta", "difference"]):
                continue
            if "factor" in key_lower:
                continue
            units_lower = str(getattr(spec, "units", "") or "").lower()
            if units_lower and units_lower not in {"dimensionless", "1"}:
                continue
            try:
                q10 = float(spec.value)
            except (TypeError, ValueError):
                continue
            if q10 < 1.0:
                errors.append(f"{key}: Q10 value {q10} must be >= 1.0")
            elif q10 > 5.0:
                msg = f"{key}: Q10 value {q10} unusually high (typical: 1.5-3.0)"
                if context.strict_mode:
                    errors.append(msg)
                else:
                    warnings.warn(msg)
        
        return errors
    
    def add_validation_hook(self, hook: Callable[[str, Any, ParameterSpec], None]) -> None:
        """
        Add validation hook for custom processing.
        
        Parameters:
            hook : Callable[[str, Any, ParameterSpec], None]
                Hook function called during validation
                Takes (key, value, spec) parameters
                
        Examples:
            >>> def log_validation(key, value, spec):
            ...     print(f"Validating {key}: {value}")
            >>> validator.add_validation_hook(log_validation)
        """
        self._validation_hooks.append(hook)
    
    def remove_validation_hook(self, hook: Callable) -> None:
        """Remove validation hook."""
        if hook in self._validation_hooks:
            self._validation_hooks.remove(hook)
    
    def _call_validation_hooks(self, key: str, value: Any, spec: ParameterSpec) -> None:
        """Call all registered validation hooks."""
        for hook in self._validation_hooks:
            try:
                hook(key, value, spec)
            except (TypeError, ValueError, RuntimeError) as e:
                logger.warning("Validation hook failed for %s: %s", key, e)


# Convenience validation functions

def validate_parameter_bounds(value: float,
                             bounds: Tuple[float, float],
                             key: str = "parameter") -> ValidationResult:
    """
    Quick bounds validation.
    
    Parameters:
        value : float
            Value to check
        bounds : Tuple[float, float]
            (min, max) bounds
        key : str
            Parameter name for error messages
            
    Returns:
        ValidationResult: Validation result with is_valid and errors
        
    Examples:
        >>> result = validate_parameter_bounds(50.0, (0, 100))
        >>> assert result.is_valid == True
    """
    errors = []
    if value < bounds[0]:
        errors.append(f"{key} value {value} below minimum {bounds[0]}")
    if value > bounds[1]:
        errors.append(f"{key} value {value} above maximum {bounds[1]}")
    
    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)


def validate_parameter_type(value: Any,
                           expected_type: str,
                           key: str = "parameter") -> ValidationResult:
    """
    Quick type validation.
    
    Parameters:
        value : Any
            Value to check
        expected_type : str
            Expected type name
        key : str
            Parameter name
            
    Returns:
        ValidationResult: Validation result with is_valid and errors
        
    Examples:
        >>> result = validate_parameter_type(5.0, "float32")
        >>> assert result.is_valid == True
    """
    type_map = {
        'float32': (int, float),
        'float64': (int, float),
        'int32': int,
        'int64': int,
        'bool': bool,
        'str': str
    }
    
    errors = []
    expected = type_map.get(expected_type)
    if expected and not isinstance(value, expected):
        errors.append(f"{key} expected type {expected_type}, got {type(value).__name__}")
    
    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)


def validate_parameter_value(value: Any,
                           bounds: Optional[Tuple[float, float]] = None,
                           *,
                           dtype: str) -> ValidationResult:
    """
    Quick parameter value validation combining type and bounds checks.

    Parameters:
        value : Any
            Value to validate
        bounds : Optional[Tuple[float, float]]
            (min, max) bounds for numeric validation
        dtype : str
            Expected data type (required, keyword-only: it is the declared
            dtype of the parameter, with no universal value)

    Returns:
        ValidationResult: Validation result with is_valid and errors

    Examples:
        >>> result = validate_parameter_value(50.0, (0, 100), dtype="float32")
        >>> assert result.is_valid == True
    """
    errors = []
    
    # Type validation
    type_result = validate_parameter_type(value, dtype)
    if not type_result.is_valid:
        errors.extend(type_result.errors)
    
    # Bounds validation
    if bounds is not None and isinstance(value, (int, float)):
        bounds_result = validate_parameter_bounds(value, bounds)
        if not bounds_result.is_valid:
            errors.extend(bounds_result.errors)
    
    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)


def check_physical_consistency(specs: Dict[str, ParameterSpec]) -> ValidationResult:
    """
    Check physical consistency across multiple parameters.
    
    Parameters:
        specs : Dict[str, ParameterSpec]
            Dictionary of parameter specifications to check
            
    Returns:
        ValidationResult: (is_valid, errors)
        
    Examples:
        >>> specs = {"temperature": temp_spec, "pressure": pressure_spec}
        >>> is_valid, errors = check_physical_consistency(specs)
    """
    errors = []
    
    # Check for basic physical constraints
    for key, spec in specs.items():
        if 'temperature' in key.lower():
            temp_value = float(spec.value)
            if spec.units.lower() in ['k', 'kelvin']:
                if temp_value < 0:
                    errors.append(f"Temperature {key} cannot be negative Kelvin: {temp_value}")
                elif temp_value < 100:  # Very low temperature for Earth systems
                    errors.append(f"Temperature {key} extremely low for Earth systems: {temp_value} K")
        
        elif 'pressure' in key.lower():
            pressure_value = float(spec.value)
            if pressure_value < 0:
                errors.append(f"Pressure {key} cannot be negative: {pressure_value}")
    
    # Check inter-parameter relationships
    temp_keys = [k for k in specs if 'temperature' in k.lower()]
    pressure_keys = [k for k in specs if 'pressure' in k.lower()]
    
    # Simple checks for common physical relationships
    if temp_keys and pressure_keys:
        # All temperatures should be reasonable for Earth systems
        for temp_key in temp_keys:
            temp_spec = specs[temp_key]
            temp_value = float(temp_spec.value)
            if temp_spec.units.lower() in ['k', 'kelvin']:
                if temp_value > 373.15:  # Above boiling point
                    for pressure_key in pressure_keys:
                        pressure_spec = specs[pressure_key]
                        pressure_value = float(pressure_spec.value)
                        # If pressure is standard atmospheric, temperature shouldn't be too high
                        if 90000 <= pressure_value <= 110000:  # ~1 atm
                            if temp_value > 400:  # Very high temperature at standard pressure
                                errors.append(
                                    f"High temperature {temp_key}={temp_value}K with standard "
                                    f"pressure {pressure_key}={pressure_value}Pa may be inconsistent"
                                )
    
    # Check vapor pressure vs total pressure relationships
    vapor_pressure_keys = [k for k in specs if 'vapor_pressure' in k.lower()]
    total_pressure_keys = [k for k in specs if 'pressure' in k.lower() and 'vapor' not in k.lower()]
    
    for vapor_key in vapor_pressure_keys:
        vapor_spec = specs[vapor_key]
        vapor_value = float(vapor_spec.value)
        
        for pressure_key in total_pressure_keys:
            pressure_spec = specs[pressure_key]
            pressure_value = float(pressure_spec.value)
            
            # Vapor pressure cannot exceed total pressure
            if vapor_value > pressure_value:
                errors.append(
                    f"Vapor pressure {vapor_key}={vapor_value}Pa cannot exceed "
                    f"total pressure {pressure_key}={pressure_value}Pa"
                )
    
    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)


def validate_parameter_relationships(specs: Dict[str, ParameterSpec]) -> ValidationResult:
    """
    Validate relationships between parameters.
    
    Parameters:
        specs : Dict[str, ParameterSpec]
            Parameter specifications to check
            
    Returns:
        ValidationResult: (is_valid, errors)
    """
    errors = []
    
    # Check photosynthesis parameter relationships
    vcmax_spec = None
    jmax_spec = None
    
    for key, spec in specs.items():
        if 'vcmax' in key.lower():
            vcmax_spec = spec
        elif 'jmax' in key.lower():
            jmax_spec = spec
    
    if vcmax_spec and jmax_spec:
        vcmax = float(vcmax_spec.value)
        jmax = float(jmax_spec.value)
        
        if vcmax > 0:
            ratio = jmax / vcmax
            if ratio < 1.2:
                errors.append(f"Jmax/Vcmax ratio {ratio:.2f} too low (should be > 1.2)")
            elif ratio > 3.0:
                errors.append(f"Jmax/Vcmax ratio {ratio:.2f} too high (should be < 3.0)")
    
    # Check soil water parameter relationships
    field_capacity_spec = None
    wilting_point_spec = None
    saturation_spec = None
    
    for key, spec in specs.items():
        if 'field_capacity' in key.lower():
            field_capacity_spec = spec
        elif 'wilting_point' in key.lower():
            wilting_point_spec = spec
        elif 'saturation' in key.lower():
            saturation_spec = spec
    
    # Validate soil water relationships: wilting_point < field_capacity < saturation
    if wilting_point_spec and field_capacity_spec:
        wp = float(wilting_point_spec.value)
        fc = float(field_capacity_spec.value)
        if wp >= fc:
            errors.append(f"Wilting point ({wp}) must be less than field capacity ({fc})")
    
    if field_capacity_spec and saturation_spec:
        fc = float(field_capacity_spec.value)
        sat = float(saturation_spec.value)
        if fc >= sat:
            errors.append(f"Field capacity ({fc}) must be less than saturation ({sat})")
    
    return ValidationResult(is_valid=(len(errors) == 0), errors=errors)
