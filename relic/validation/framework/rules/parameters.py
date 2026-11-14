"""
Parameter Validation Rules

This module contains validation rules for parameter specifications, distributions,
bounds checking, and consistency validation. Rules are designed to work with
the RELIC parameter system and stochastic distribution specifications.

Key Validation Areas:
    - Parameter boundary validation
    - Distribution parameter consistency
    - Log-scale parameter validation
    - Spatial correlation length validation
    - Parameter schema completeness
    - Type and unit consistency

Mathematical Framework:
    For parameter p with specification S(p):
    - Bounds validation: min_bound <= p <= max_bound
    - Distribution consistency: D(p|theta) well-defined for parameters theta
    - Log-scale consistency: log_scale  p > 0  bounds > 0
    - Correlation validation: |rho(p_1,p_2)| <= correlation_length

References:
    - RELIC parameter system specification
    - Statistical distribution theory
    - Spatial correlation modeling
"""

from typing import Optional, Any
import logging

import torch
from torch import Tensor
import numpy as np

from ..models import (
    ValidationResultModel, ValidationLevel, ValidationCode,
    ValidationContext, ValidationResults, ValidationDetail
)
from ..decorators import validated_rule

logger = logging.getLogger(__name__)

# Import parameter system components
from ....parameters import (
    get_global_manager,
    ParameterSpec,
    ParameterValidator,
    ValidationLevel as ParamValidationLevel,
)
from ....parameters.types import ValidationContext as ParamValidationContext
from ....stochastic.distributions import DistributionSpec


def _build_spec_for_validation(
    param_name: str,
    param_value: Any,
    param_bounds: Any,
    param_units: str,
    param_info_obj: Any,
) -> "ParameterSpec":
    """Build the ParameterSpec of a declaration; an invalid declaration raises."""
    spec_data = {
        "key": param_name,
        "value": param_value,
        "bounds": param_bounds,
        "units": param_units or "dimensionless",
    }
    if isinstance(param_info_obj, dict):
        for field in (
            "dtype",
            "category",
            "description",
            "references",
            "tags",
            "dimension",
            "provenance",
            "requires_grad",
            "allowed_values",
        ):
            if field in param_info_obj:
                spec_data[field] = param_info_obj[field]
    return ParameterSpec.from_dict(spec_data)


def _extract_parameter_fields(param_info: Any) -> tuple[Any, Any, str]:
    if isinstance(param_info, ParameterSpec):
        return (
            getattr(param_info, "value", None),
            getattr(param_info, "bounds", None),
            getattr(param_info, "units", "dimensionless"),
        )
    if isinstance(param_info, dict):
        return (
            param_info.get("value"),
            param_info.get("bounds"),
            param_info.get("units", "dimensionless"),
        )
    return (
        getattr(param_info, "value", None),
        getattr(param_info, "bounds", None),
        getattr(param_info, "units", "dimensionless"),
    )


def _append_invalid_declaration(
    *,
    results: ValidationResults,
    param_name: str,
    error: Exception,
) -> None:
    """A declaration that cannot form a ParameterSpec is a failed result."""
    results.append(ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.GENERIC_VALIDATION_FAILED,
        message=(
            f"Parameter '{param_name}' cannot be bounds-checked: its declaration is "
            f"not a valid ParameterSpec ({error})"
        ),
        passed=False,
        layer="L1",
        scope="parameter_bounds_validation",
        tags=["parameters", "invalid_declaration"],
        details=[ValidationDetail(variable_name=param_name, value=str(error))],
    ))


def _resolve_bound_violation_detail(
    param_value: Any,
    param_bounds: Any,
) -> tuple[Optional[float], Optional[dict[str, float]]]:
    detail_value: Optional[float] = None
    stats: Optional[dict[str, float]] = None
    if isinstance(param_value, Tensor):
        if param_value.numel() == 1:
            detail_value = float(param_value.item())
        else:
            min_val = float(param_value.min().item())
            max_val = float(param_value.max().item())
            mean_val = float(param_value.mean().item())
            stats = {
                "min_value": min_val,
                "max_value": max_val,
                "mean_value": mean_val,
            }
            if isinstance(param_bounds, (list, tuple)) and len(param_bounds) == 2:
                min_bound, max_bound = param_bounds
                if min_bound is not None and min_val < min_bound:
                    detail_value = min_val
                elif max_bound is not None and max_val > max_bound:
                    detail_value = max_val
                else:
                    detail_value = min_val
            else:
                detail_value = min_val
        return detail_value, stats

    try:
        detail_value = float(param_value)
    except (TypeError, ValueError):
        detail_value = None
    return detail_value, stats


def _append_validator_bound_errors(
    *,
    results: ValidationResults,
    param_name: str,
    param_bounds: Any,
    param_units: str,
    bound_errors: list[str],
    detail_value: Optional[float],
    stats: Optional[dict[str, float]],
) -> None:
    for err_msg in bound_errors:
        detail_kwargs: dict[str, Any] = {
            "variable_name": param_name,
            "value": detail_value,
            "expected": str(param_bounds),
            "units": param_units,
        }
        if stats is not None:
            detail_kwargs["statistics"] = stats
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.PARAM_BOUND_VIOLATION,
            message=err_msg,
            layer="L1",
            scope="parameter_bounds_validation",
            tags=["parameters", "bounds_violation"],
            details=[ValidationDetail(**detail_kwargs)],
        ))


def _apply_validator_bounds(
    *,
    param_name: str,
    param_value: Any,
    param_spec: "ParameterSpec",
    param_bounds: Any,
    param_units: str,
    results: ValidationResults,
    validator_state: dict[str, Any],
) -> tuple[bool, int]:
    if validator_state.get("validator") is None:
        validator_state["validator"] = ParameterValidator(level=ParamValidationLevel.BASIC)
    if validator_state.get("context") is None:
        validator_state["context"] = ParamValidationContext()
    result = validator_state["validator"].validate_value(
        param_name,
        param_value,
        param_spec,
        level=ParamValidationLevel.BASIC,
        context=validator_state.get("context"),
    )
    bound_errors: list[str] = []
    error_types = getattr(result, "error_types", []) or []
    if error_types and len(error_types) == len(result.errors):
        for err_msg, err_type in zip(result.errors, error_types):
            if err_type == "bounds":
                bound_errors.append(err_msg)
    elif not result.is_valid:
        bound_errors = list(result.errors)
    if not bound_errors:
        return True, 0

    detail_value, stats = _resolve_bound_violation_detail(param_value, param_bounds)
    _append_validator_bound_errors(
        results=results,
        param_name=param_name,
        param_bounds=param_bounds,
        param_units=param_units,
        bound_errors=bound_errors,
        detail_value=detail_value,
        stats=stats,
    )
    return True, 1


def _extract_log_scale_fields(param_spec: Any) -> tuple[bool, Any, Any]:
    if isinstance(param_spec, dict):
        return (
            bool(param_spec.get("log_scale", False)),
            param_spec.get("bounds"),
            param_spec.get("value"),
        )
    return (
        bool(getattr(param_spec, "log_scale", False)),
        getattr(param_spec, "bounds", None),
        getattr(param_spec, "value", None),
    )


def _collect_log_scale_violations(bounds: Any, current_value: Any) -> list[str]:
    violations: list[str] = []
    if bounds:
        min_bound, max_bound = bounds
        if min_bound <= 0:
            violations.append(f"minimum bound ({min_bound}) <= 0")
        if max_bound <= 0:
            violations.append(f"maximum bound ({max_bound}) <= 0")
    if current_value is None:
        return violations
    if isinstance(current_value, Tensor):
        if torch.any(current_value <= 0):
            violations.append("current value contains non-positive values")
        return violations
    if isinstance(current_value, (list, tuple, np.ndarray)):
        if any(v <= 0 for v in current_value):
            violations.append("current value contains non-positive values")
        return violations
    if current_value <= 0:
        violations.append(f"current value ({current_value}) <= 0")
    return violations


@validated_rule(
    rule_id="parameter_bounds_validation",
    layer="L1",
    name="Parameter Bounds Validation",
    description="Validates that parameter values are within specified bounds",
    domains=["parameters", "bounds"],
    tags=["parameters", "bounds", "validation"],
    references=["RELIC Parameter Specification"]
)
def validate_parameter_bounds(context: ValidationContext) -> ValidationResults:
    """
    Validate parameter values against specified bounds.
    
    Mathematical Expression:
        For parameter p with bounds [min_val, max_val]:
        Valid(p)  min_val <= p <= max_val
        
    Parameters:
        context: Validation context containing parameter manager
        
    Returns:
        List of validation results for parameter bounds
    """
    results = []
    
    # Try to get parameters from metadata first, then from state_manager
    parameters = None
    if context.metadata and 'parameters' in context.metadata:
        parameters = context.metadata['parameters']
    elif context.state_manager:
        # Get parameter manager from context
        param_manager = getattr(context.state_manager, '_parameter_manager', None)
        if param_manager and hasattr(param_manager, 'get_all_parameters'):
            parameters = param_manager.get_all_parameters()
    
    if not parameters:
        return [ValidationResultModel(
            level=ValidationLevel.INFO,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message="No parameters found to validate bounds",
            layer="L1",
            scope="parameter_bounds_validation", 
            tags=["parameters", "no_data"]
        )]
    
    try:
        bounds_violations = 0
        invalid_declarations = 0
        total_parameters = 0
        validator_state: dict[str, Any] = {"validator": None, "context": None}
        for param_name, param_info in parameters.items():
            total_parameters += 1

            param_value, param_bounds, param_units = _extract_parameter_fields(param_info)
            if param_value is None or param_bounds is None:
                continue

            if isinstance(param_info, ParameterSpec):
                spec_for_validation = param_info
            else:
                # ParameterSpec.from_dict rejects an invalid declaration (for
                # example inverted bounds) with ValueError/TypeError: that is
                # a failed result for this parameter, never a fallback
                # comparison against the invalid bounds.
                try:
                    spec_for_validation = _build_spec_for_validation(
                        param_name,
                        param_value,
                        param_bounds,
                        param_units,
                        param_info,
                    )
                except (ValueError, TypeError) as exc:
                    invalid_declarations += 1
                    _append_invalid_declaration(results=results, param_name=param_name, error=exc)
                    continue
            _, violation_count = _apply_validator_bounds(
                param_name=param_name,
                param_value=param_value,
                param_spec=spec_for_validation,
                param_bounds=param_bounds,
                param_units=param_units,
                results=results,
                validator_state=validator_state,
            )
            bounds_violations += violation_count
        
        # Summary result
        if bounds_violations == 0 and invalid_declarations == 0:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_BOUNDS_OK,
                message=f"All {total_parameters} parameters within bounds",
                layer="L1",
                scope="parameter_bounds_validation",
                tags=["parameters", "success"],
                metrics={
                    "total_parameters": total_parameters,
                    "bounds_violations": bounds_violations
                }
            ))
        else:
            results.append(ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.PARAM_BOUND_VIOLATION,
                message=(
                    f"{bounds_violations}/{total_parameters} parameters violate bounds; "
                    f"{invalid_declarations} declarations are invalid"
                ),
                layer="L1", 
                scope="parameter_bounds_validation",
                tags=["parameters", "bounds_summary"],
                metrics={
                    "total_parameters": total_parameters,
                    "bounds_violations": bounds_violations,
                    "invalid_declarations": invalid_declarations,
                    "violation_rate": bounds_violations / total_parameters if total_parameters > 0 else 0.0
                }
            ))
        
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Parameter bounds validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Parameter bounds validation failed: {str(e)}",
            layer="L1",
            scope="parameter_bounds_validation",
            tags=["parameters", "execution_error"]
        ))
    
    return results


def _build_distribution_model(
    dist_spec: Any,
) -> Any:
    if isinstance(dist_spec, DistributionSpec) or not isinstance(dist_spec, dict):
        return dist_spec

    spec_dict = {}
    if 'distribution_type' in dist_spec:
        from ....stochastic.distributions import DistributionType

        dist_type_str = str(dist_spec['distribution_type']).lower()
        spec_dict['dist_type'] = DistributionType(dist_type_str)
    if 'parameters' in dist_spec:
        spec_dict['params'] = dist_spec['parameters']
    for field in ['bounds', 'log_scale', 'integer_valued', 'correlation_length', 'correlation_nugget']:
        if field in dist_spec:
            spec_dict[field] = dist_spec[field]
    # DistributionSpec owns the correlated-field contract: with a
    # correlation_length both kernel (a CorrelationKernel) and
    # correlation_nugget (finite, >= 0) are declared; without one neither is.
    # The kernel name is converted here; an unknown name is rejected, and a
    # missing kernel or nugget is left for DistributionSpec to reject.
    if 'kernel' in dist_spec:
        spec_dict['kernel'] = _correlation_kernel(dist_spec['kernel'])
    return DistributionSpec(**spec_dict)


def _correlation_kernel(value: Any) -> Any:
    """Return the CorrelationKernel named by ``value`` (a member or its string value)."""
    from ....stochastic.distributions import CorrelationKernel

    if isinstance(value, CorrelationKernel):
        return value
    try:
        return CorrelationKernel(str(value).lower())
    except ValueError as exc:
        allowed = ", ".join(kernel.value for kernel in CorrelationKernel)
        raise ValueError(
            f"Unknown correlation kernel {value!r}; expected one of: {allowed}"
        ) from exc


def _extract_distribution_fields(dist_spec: Any) -> tuple[str, dict]:
    if isinstance(dist_spec, dict):
        distribution_type = str(dist_spec.get('distribution_type', '')).lower()
        params = dist_spec.get('parameters', {}) or {}
        return distribution_type, params
    if isinstance(dist_spec, DistributionSpec):
        distribution_type = dist_spec.dist_type.value.lower()
        params = dist_spec.params or {}
        return distribution_type, params
    distribution_type = str(getattr(dist_spec, 'distribution_type', '')).lower()
    params = getattr(dist_spec, 'parameters', {}) or {}
    return distribution_type, params


def _validate_scale_parameter(
    dist_name: str,
    params: dict,
    *,
    label: str,
    tags: list[str],
) -> list[ValidationResultModel]:
    if 'scale' not in params and 'std' not in params:
        return []
    scale = params.get('scale', params.get('std', 1.0))
    if scale > 0:
        return []
    return [
        ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.PARAM_TYPE_MISMATCH,
            message=f"{label} distribution '{dist_name}' has non-positive scale: {scale}",
            layer="L1",
            scope="distribution_parameter_consistency",
            tags=tags,
            details=[ValidationDetail(
                variable_name=dist_name,
                value=scale,
                expected="> 0",
                metadata={"distribution_type": label.lower(), "parameter": "scale"},
            )] if label == "Normal" else [],
        )
    ]


def _validate_uniform_parameters(dist_name: str, params: dict) -> list[ValidationResultModel]:
    low = params.get('low', params.get('a', 0.0))
    high = params.get('high', params.get('b', 1.0))
    if low < high:
        return []
    return [
        ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.PARAM_TYPE_MISMATCH,
            message=f"Uniform distribution '{dist_name}' has low >= high: [{low}, {high}]",
            layer="L1",
            scope="distribution_parameter_consistency",
            tags=["distributions", "uniform", "bounds_violation"],
            details=[ValidationDetail(
                variable_name=dist_name,
                value=f"[{low}, {high}]",
                expected="low < high",
                metadata={"distribution_type": "uniform"},
            )],
        )
    ]


def _validate_gamma_parameters(dist_name: str, params: dict) -> list[ValidationResultModel]:
    alpha = params.get('alpha', params.get('concentration', 1.0))
    beta = params.get('beta', params.get('rate', 1.0))
    errors: list[ValidationResultModel] = []
    if alpha <= 0:
        errors.append(
            ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.PARAM_TYPE_MISMATCH,
                message=f"Gamma distribution '{dist_name}' has non-positive alpha: {alpha}",
                layer="L1",
                scope="distribution_parameter_consistency",
                tags=["distributions", "gamma", "alpha_violation"],
            )
        )
    if beta <= 0:
        errors.append(
            ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.PARAM_TYPE_MISMATCH,
                message=f"Gamma distribution '{dist_name}' has non-positive beta: {beta}",
                layer="L1",
                scope="distribution_parameter_consistency",
                tags=["distributions", "gamma", "beta_violation"],
            )
        )
    return errors


def _validate_beta_parameters(dist_name: str, params: dict) -> list[ValidationResultModel]:
    alpha = params.get('alpha', params.get('concentration1', 1.0))
    beta = params.get('beta', params.get('concentration0', 1.0))
    if alpha > 0 and beta > 0:
        return []
    return [
        ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.PARAM_TYPE_MISMATCH,
            message=f"Beta distribution '{dist_name}' has non-positive parameters: alpha={alpha}, beta={beta}",
            layer="L1",
            scope="distribution_parameter_consistency",
            tags=["distributions", "beta", "parameter_violation"],
        )
    ]


def _validate_distribution(
    dist_name: str,
    distribution_type: str,
    params: dict,
) -> list[ValidationResultModel]:
    if distribution_type == 'normal':
        return _validate_scale_parameter(
            dist_name,
            params,
            label="Normal",
            tags=["distributions", "normal", "scale_violation"],
        )
    if distribution_type == 'lognormal':
        return _validate_scale_parameter(
            dist_name,
            params,
            label="Log-normal",
            tags=["distributions", "lognormal", "scale_violation"],
        )
    if distribution_type == 'uniform':
        return _validate_uniform_parameters(dist_name, params)
    if distribution_type == 'gamma':
        return _validate_gamma_parameters(dist_name, params)
    if distribution_type == 'beta':
        return _validate_beta_parameters(dist_name, params)
    return []


@validated_rule(
    rule_id="distribution_parameter_consistency",
    layer="L1",
    name="Distribution Parameter Consistency",
    description="Validates consistency of distribution parameters",
    domains=["stochastic", "distributions"],
    tags=["distributions", "consistency", "parameters"],
    references=["Statistical Distribution Theory", "RELIC Stochastic Specification"]
)
def validate_distribution_parameter_consistency(context: ValidationContext) -> ValidationResults:
    """
    Validate consistency of statistical distribution parameters.
    
    Mathematical Expression:
        For distribution D with parameters theta = {theta_1, theta_2, ...}:
        Consistent(theta)  i: theta in Domain(theta)  Constraints(theta) satisfied
        
    Parameters:
        context: Validation context containing distribution specifications
        
    Returns:
        List of validation results for distribution consistency
    """
    results = []
    
    # Continue with validation even if parameter system is not available
    # We can still validate basic distribution parameter consistency
    
    try:
        # Get distribution specifications from context
        metadata = context.metadata or {}
        distributions = metadata.get('distributions', {})
        
        if not distributions:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.MISSING_REQUIRED_DATA,
                message="No distribution specifications to validate",
                layer="L1",
                scope="distribution_parameter_consistency",
                tags=["distributions", "no_data"]
            )]
        
        total_distributions = 0
        consistency_violations = 0
        
        for dist_name, dist_spec in distributions.items():
            total_distributions += 1
            
            try:
                distribution_model = _build_distribution_model(dist_spec)
                distribution_type, params = _extract_distribution_fields(distribution_model)
                errors = _validate_distribution(dist_name, distribution_type, params)
                consistency_violations += len(errors)
                results.extend(errors)
                
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                consistency_violations += 1
                results.append(ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.PARAM_SCHEMA_MISSING,
                    message=f"Distribution '{dist_name}' validation failed: {str(e)}",
                    layer="L1",
                    scope="distribution_parameter_consistency",
                    tags=["distributions", "validation_error"],
                    details=[ValidationDetail(
                        variable_name=dist_name,
                        metadata={"error": str(e)}
                    )]
                ))
        
        # Summary result
        if consistency_violations == 0:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_TYPE_OK,
                message=f"All {total_distributions} distributions have consistent parameters",
                layer="L1",
                scope="distribution_parameter_consistency",
                tags=["distributions", "success"],
                metrics={
                    "total_distributions": total_distributions,
                    "consistency_violations": consistency_violations
                }
            ))
        
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Distribution parameter validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Distribution parameter validation failed: {str(e)}",
            layer="L1",
            scope="distribution_parameter_consistency",
            tags=["distributions", "execution_error"]
        ))
    
    return results


@validated_rule(
    rule_id="log_scale_parameter_validation",
    layer="L1",
    name="Log-Scale Parameter Validation",
    description="Validates consistency of log-scale parameter specifications",
    domains=["parameters", "log_scale"],
    tags=["parameters", "log_scale", "consistency"],
    references=["RELIC Parameter Log-Scale Specification"]
)
def validate_log_scale_parameters(context: ValidationContext) -> ValidationResults:
    """
    Validate consistency of log-scale parameter specifications.
    
    Mathematical Expression:
        For log-scale parameter p:
        Valid(p)  log_scale = True  (p > 0  bounds > 0)
        
    Parameters:
        context: Validation context containing parameter specifications
        
    Returns:
        List of validation results for log-scale parameters
    """
    results = []
    
    try:
        # Get parameter specifications from context
        metadata = context.metadata or {}
        parameters = metadata.get('parameters', {})
        
        if not parameters:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.MISSING_REQUIRED_DATA,
                message="No parameters to validate for log-scale consistency",
                layer="L1",
                scope="log_scale_parameter_validation",
                tags=["parameters", "log_scale", "no_data"]
            )]
        
        total_log_scale = 0
        log_scale_violations = 0
        
        for param_name, param_spec in parameters.items():
            log_scale, bounds, current_value = _extract_log_scale_fields(param_spec)
            
            if not log_scale:
                continue
                
            total_log_scale += 1
            violations = _collect_log_scale_violations(bounds, current_value)
            
            if violations:
                log_scale_violations += 1
                results.append(ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.PARAM_LOG_SCALE_INCONSISTENT,
                    message=f"Log-scale parameter '{param_name}': {'; '.join(violations)}",
                    layer="L1",
                    scope="log_scale_parameter_validation",
                    tags=["parameters", "log_scale", "violation"],
                    details=[ValidationDetail(
                        variable_name=param_name,
                        value=str(current_value) if current_value is not None else "None",
                        expected="All values > 0 for log-scale parameters",
                        metadata={
                            "log_scale": log_scale,
                            "bounds": bounds,
                            "violations": violations
                        }
                    )]
                ))
        
        # Summary result
        if total_log_scale == 0:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_LOG_SCALE_INCONSISTENT,
                message="No log-scale parameters found to validate",
                layer="L1",
                scope="log_scale_parameter_validation",
                tags=["parameters", "log_scale", "none_found"]
            ))
        elif log_scale_violations == 0:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_LOG_SCALE_OK,
                message=f"All {total_log_scale} log-scale parameters are consistent",
                layer="L1",
                scope="log_scale_parameter_validation",
                tags=["parameters", "log_scale", "success"],
                metrics={
                    "total_log_scale_parameters": total_log_scale,
                    "log_scale_violations": log_scale_violations
                }
            ))
        
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Log-scale parameter validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Log-scale parameter validation failed: {str(e)}",
            layer="L1",
            scope="log_scale_parameter_validation",
            tags=["parameters", "log_scale", "execution_error"]
        ))
    
    return results


@validated_rule(
    rule_id="spatial_correlation_validation",
    layer="L1",
    name="Spatial Correlation Validation",
    description="Validates spatial correlation length specifications for parameters",
    domains=["parameters", "spatial", "correlation"],
    tags=["parameters", "spatial", "correlation"],
    references=["Spatial Statistics Theory", "RELIC Spatial Parameter Specification"]
)
def validate_spatial_correlation(context: ValidationContext) -> ValidationResults:
    """
    Validate spatial correlation length specifications.
    
    Mathematical Expression:
        For parameters p_1, p_2 with correlation length lambda:
        |Corr(p_1(x), p_2(x+d))| <= exp(-d/lambda) for distance d
        Valid(lambda)  lambda > 0
        
    Parameters:
        context: Validation context with spatial parameter specifications
        
    Returns:
        List of validation results for spatial correlation
    """
    results = []
    
    try:
        # Get spatial parameter specifications from context
        metadata = context.metadata or {}
        spatial_params = metadata.get('spatial_parameters', {})
        
        if not spatial_params:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.MISSING_REQUIRED_DATA,
                message="No spatial parameters to validate for correlation",
                layer="L1",
                scope="spatial_correlation_validation",
                tags=["parameters", "spatial", "no_data"]
            )]
        
        total_spatial = 0
        correlation_violations = 0
        
        for param_name, spatial_spec in spatial_params.items():
            total_spatial += 1
            
            # Extract correlation length
            if isinstance(spatial_spec, dict):
                correlation_length = spatial_spec.get('correlation_length')
                spatial_dims = spatial_spec.get('spatial_dimensions', ['x', 'y'])
            else:
                correlation_length = getattr(spatial_spec, 'correlation_length', None)
                spatial_dims = getattr(spatial_spec, 'spatial_dimensions', ['x', 'y'])
            
            if correlation_length is None:
                results.append(ValidationResultModel(
                    level=ValidationLevel.WARNING,
                    code=ValidationCode.PARAM_CORRELATION_VIOLATION,
                    message=f"Spatial parameter '{param_name}' lacks correlation length specification",
                    layer="L1",
                    scope="spatial_correlation_validation",
                    tags=["parameters", "spatial", "missing_correlation_length"],
                    details=[ValidationDetail(
                        variable_name=param_name,
                        expected="correlation_length > 0",
                        metadata={"spatial_dimensions": spatial_dims}
                    )]
                ))
                continue
            
            # Validate correlation length is positive
            if isinstance(correlation_length, (list, tuple)):
                # Different correlation lengths for different dimensions
                for i, corr_len in enumerate(correlation_length):
                    if corr_len <= 0:
                        correlation_violations += 1
                        dim_name = spatial_dims[i] if i < len(spatial_dims) else f"dim_{i}"
                        results.append(ValidationResultModel(
                            level=ValidationLevel.ERROR,
                            code=ValidationCode.PARAM_CORRELATION_VIOLATION,
                            message=f"Spatial parameter '{param_name}' has non-positive correlation length in {dim_name}: {corr_len}",
                            layer="L1",
                            scope="spatial_correlation_validation",
                            tags=["parameters", "spatial", "correlation_violation"],
                            details=[ValidationDetail(
                                variable_name=f"{param_name}_{dim_name}",
                                value=corr_len,
                                expected="> 0",
                                units="spatial_units"
                            )]
                        ))
            else:
                # Single correlation length for all dimensions
                if correlation_length <= 0:
                    correlation_violations += 1
                    results.append(ValidationResultModel(
                        level=ValidationLevel.ERROR,
                        code=ValidationCode.PARAM_CORRELATION_VIOLATION,
                        message=f"Spatial parameter '{param_name}' has non-positive correlation length: {correlation_length}",
                        layer="L1",
                        scope="spatial_correlation_validation",
                        tags=["parameters", "spatial", "correlation_violation"],
                        details=[ValidationDetail(
                            variable_name=param_name,
                            value=correlation_length,
                            expected="> 0",
                            units="spatial_units"
                        )]
                    ))
            
            # Check correlation length is reasonable (not too large or small)
            if isinstance(correlation_length, (int, float)) and correlation_length > 0:
                if correlation_length > 1e6:  # Very large correlation length
                    results.append(ValidationResultModel(
                        level=ValidationLevel.WARNING,
                        code=ValidationCode.PARAM_CORRELATION_VIOLATION,
                        message=f"Spatial parameter '{param_name}' has very large correlation length: {correlation_length}",
                        layer="L1",
                        scope="spatial_correlation_validation",
                        tags=["parameters", "spatial", "large_correlation"]
                    ))
                elif correlation_length < 1e-6:  # Very small correlation length
                    results.append(ValidationResultModel(
                        level=ValidationLevel.WARNING,
                        code=ValidationCode.PARAM_CORRELATION_VIOLATION,
                        message=f"Spatial parameter '{param_name}' has very small correlation length: {correlation_length}",
                        layer="L1",
                        scope="spatial_correlation_validation",
                        tags=["parameters", "spatial", "small_correlation"]
                    ))
        
        # Summary result
        if correlation_violations == 0:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_CORRELATION_OK,
                message=f"All {total_spatial} spatial parameters have valid correlation lengths",
                layer="L1",
                scope="spatial_correlation_validation",
                tags=["parameters", "spatial", "success"],
                metrics={
                    "total_spatial_parameters": total_spatial,
                    "correlation_violations": correlation_violations
                }
            ))
        
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Spatial correlation validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Spatial correlation validation failed: {str(e)}",
            layer="L1",
            scope="spatial_correlation_validation",
            tags=["parameters", "spatial", "execution_error"]
        ))
    
    return results


@validated_rule(
    rule_id="parameter_schema_completeness",
    layer="L1",
    name="Parameter Schema Completeness",
    description="Validates completeness of parameter schema specifications",
    domains=["parameters", "schema"],
    tags=["parameters", "schema", "completeness"],
    references=["RELIC Parameter Schema Specification"]
)
def validate_parameter_schema_completeness(context: ValidationContext) -> ValidationResults:
    """
    Validate completeness of parameter schema specifications.
    
    Parameters:
        context: Validation context with parameter schema
        
    Returns:
        List of validation results for schema completeness
    """
    results = []
    
    try:
        # Get parameter schema from context
        metadata = context.metadata or {}
        parameters = metadata.get('parameters', {})
        
        if not parameters:
            return [ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.PARAM_SCHEMA_MISSING,
                message="No parameter schema found to validate",
                layer="L1",
                scope="parameter_schema_completeness",
                tags=["parameters", "schema", "missing"]
            )]
        
        total_parameters = 0
        missing_fields = 0
        # `default`/`default_value` is intentionally deprecated in parameter schema.
        required_fields = ['bounds', 'units', 'description']
        
        for param_name, param_spec in parameters.items():
            total_parameters += 1
            missing_for_param = []
            
            for field in required_fields:
                if isinstance(param_spec, dict):
                    field_value = param_spec.get(field)
                else:
                    field_value = getattr(param_spec, field, None)
                
                if field_value is None:
                    missing_for_param.append(field)
            
            if missing_for_param:
                missing_fields += 1
                results.append(ValidationResultModel(
                    level=ValidationLevel.WARNING,
                    code=ValidationCode.PARAM_SCHEMA_MISSING,
                    message=f"Parameter '{param_name}' missing required fields: {', '.join(missing_for_param)}",
                    layer="L1",
                    scope="parameter_schema_completeness",
                    tags=["parameters", "schema", "incomplete"],
                    details=[ValidationDetail(
                        variable_name=param_name,
                        expected=f"Required fields: {', '.join(required_fields)}",
                        metadata={"missing_fields": missing_for_param}
                    )]
                ))
        
        # Summary result
        completion_rate = (total_parameters - missing_fields) / total_parameters if total_parameters > 0 else 0.0
        
        if missing_fields == 0:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_SCHEMA_OK,
                message=f"All {total_parameters} parameters have complete schema",
                layer="L1",
                scope="parameter_schema_completeness",
                tags=["parameters", "schema", "complete"],
                metrics={
                    "total_parameters": total_parameters,
                    "completion_rate": completion_rate
                }
            ))
        else:
            level = ValidationLevel.WARNING if completion_rate > 0.8 else ValidationLevel.ERROR
            results.append(ValidationResultModel(
                level=level,
                code=ValidationCode.PARAM_SCHEMA_MISSING,
                message=f"Parameter schema completion rate: {completion_rate:.1%} ({total_parameters - missing_fields}/{total_parameters})",
                layer="L1",
                scope="parameter_schema_completeness",
                tags=["parameters", "schema", "summary"],
                metrics={
                    "total_parameters": total_parameters,
                    "parameters_with_missing_fields": missing_fields,
                    "completion_rate": completion_rate
                }
            ))
            
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Parameter schema validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Parameter schema validation failed: {str(e)}",
            layer="L1",
            scope="parameter_schema_completeness",
            tags=["parameters", "schema", "execution_error"]
        ))
    
    return results


# Additional validation functions for test consistency
def check_parameter_bounds():
    """Check parameter bounds validation across the system."""

    try:
        manager = get_global_manager()
        errors = []
        validator = ParameterValidator(level=ParamValidationLevel.BASIC)
        context = ParamValidationContext()

        # Validate all parameters and keep only bounds errors.
        for key, spec in manager.parameters.items():
            result = validator.validate_value(
                key,
                spec.value,
                spec,
                level=ParamValidationLevel.BASIC,
                context=context,
            )
            error_types = getattr(result, "error_types", []) or []
            if error_types and len(error_types) == len(result.errors):
                for err_msg, err_type in zip(result.errors, error_types):
                    if err_type == "bounds":
                        errors.append(f"{key}: {err_msg}")
            elif not result.is_valid:
                errors.extend([f"{key}: {error}" for error in result.errors])

        return errors
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Parameter bounds check failed: {e}")
        return [f"Bounds check error: {e}"]


def check_parameter_units():
    """Check parameter units consistency across the system."""

    try:
        manager = get_global_manager()
        errors = []
        validator = ParameterValidator(level=ParamValidationLevel.STANDARD)
        context = ParamValidationContext()

        for key, spec in manager.parameters.items():
            result = validator.validate_value(
                key,
                spec.value,
                spec,
                level=ParamValidationLevel.STANDARD,
                context=context,
            )
            error_types = getattr(result, "error_types", []) or []
            if error_types and len(error_types) == len(result.errors):
                for err_msg, err_type in zip(result.errors, error_types):
                    if err_type == "units":
                        errors.append(f"{key}: {err_msg}")
            elif not result.is_valid:
                errors.extend([f"{key}: {error}" for error in result.errors])

        return errors
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Parameter units check failed: {e}")
        return [f"Units check error: {e}"]


def check_parameter_relationships():
    """Check parameter relationships consistency."""

    try:
        manager = get_global_manager()
        errors = []
        
        # Check for specific parameter relationships
        params = {key: spec.value for key, spec in manager.parameters.items()}
        
        # Water content relationships
        field_cap = None
        wilting_point = None
        saturation = None
        
        for key, value in params.items():
            if "field_capacity" in key:
                field_cap = float(value) if hasattr(value, '__float__') else value
            elif "wilting_point" in key:
                wilting_point = float(value) if hasattr(value, '__float__') else value
            elif "saturation" in key:
                saturation = float(value) if hasattr(value, '__float__') else value
        
        # Check water content hierarchy: wilting_point < field_capacity < saturation
        if wilting_point is not None and field_cap is not None:
            if wilting_point >= field_cap:
                errors.append(f"wilting_point ({wilting_point}) >= field_capacity ({field_cap})")
        
        if field_cap is not None and saturation is not None:
            if field_cap >= saturation:
                errors.append(f"field_capacity ({field_cap}) >= saturation ({saturation})")
        
        return errors
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Parameter relationships check failed: {e}")
        return [f"Relationship check error: {e}"]


# Export validation rule functions
__all__ = [
    "validate_parameter_bounds",
    "validate_distribution_parameter_consistency", 
    "validate_log_scale_parameters",
    "validate_spatial_correlation",
    "validate_parameter_schema_completeness",
    "check_parameter_bounds",
    "check_parameter_units", 
    "check_parameter_relationships"
]
