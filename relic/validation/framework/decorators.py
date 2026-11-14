"""
Validation Framework Decorators

This module provides decorators for standardizing validation rules for
the unified framework structure.

Key Decorators:
    @validated_rule: Standardizes new validation rules with metadata
    @require_state_manager: Ensures state manager access for validation rules
    @validation_cache: Caches validation results for performance

Design Principles:
    - Automatic result standardization
    - Metadata extraction and registration
    - Error handling and logging
    - Performance optimization through caching

Mathematical Framework:
    Decorator composition: D_1(D_2(...(D(f)))) where f is validation rule

References:
    - Python decorator patterns
    - Adapter pattern (GoF Design Patterns)
"""

from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import logging
import time
import traceback

from .models import (
    ValidationResultModel, ValidationLevel, ValidationCode,
    ValidationContext, ValidationResults
)
from .registry import ValidationRegistry, RuleMetadata, get_global_registry

logger = logging.getLogger(__name__)


def validated_rule(rule_id: str,
                  layer: str,
                  name: Optional[str] = None,
                  description: Optional[str] = None,
                  domains: Optional[List[str]] = None,
                  tags: Optional[List[str]] = None,
                  dependencies: Optional[List[str]] = None,
                  cf_requirements: Optional[List[str]] = None,
                  units_contract: Optional[Dict[str, str]] = None,
                  references: Optional[List[str]] = None,
                  version: str = "1.0.0",
                  author: Optional[str] = None,
                  registry: Optional[ValidationRegistry] = None) -> Callable:
    """
    Decorator for standardizing validation rules.

    This decorator automatically:
    - Registers the rule in the validation registry
    - Standardizes function signature and return types
    - Adds error handling and logging
    - Ensures consistent result structure

    Parameters:
        rule_id: Unique identifier for the rule
        layer: Validation layer (L1/L2/L3/L4)
        name: Human-readable rule name
        description: Detailed description
        domains: Applicable scientific domains
        tags: Categorization tags
        dependencies: Other rules that must run first
        cf_requirements: bounded CF variable-metadata requirements
        units_contract: Expected units for inputs/outputs
        references: Scientific literature references
        version: Rule version
        author: Rule author/maintainer
        registry: Custom registry (uses global if None)

    Returns:
        Decorated validation rule function

    Example:
        >>> @validated_rule("unit_conversion_check", "L1",
        ...                domains=["hydrosphere"], tags=["units", "cf"])
        ... def check_unit_conversion(context: ValidationContext) -> ValidationResults:
        ...     # Validation logic here
        ...     return [ValidationResultModel(...)]
    """
    def decorator(func: Callable) -> Callable:
        # Extract metadata
        actual_name = name or func.__name__.replace('_', ' ').title()
        actual_description = description or func.__doc__ or f"Validation rule: {actual_name}"
        actual_domains = domains or []
        actual_tags = tags or []
        actual_dependencies = dependencies or []
        actual_cf_requirements = cf_requirements or []
        actual_units_contract = units_contract or {}
        actual_references = references or []

        # Create rule metadata
        metadata = RuleMetadata(
            rule_id=rule_id,
            name=actual_name,
            description=actual_description,
            layer=layer,
            domains=actual_domains,
            tags=actual_tags,
            dependencies=actual_dependencies,
            cf_requirements=actual_cf_requirements,
            units_contract=actual_units_contract,
            references=actual_references,
            version=version,
            author=author
        )

        @wraps(func)
        def wrapper(context: ValidationContext) -> ValidationResults:
            start_time = time.time()

            try:
                # Set execution context
                if context.execution_layer is None:
                    context.execution_layer = layer

                logger.debug(f"Executing validation rule: {rule_id}")

                # Call original function
                results = func(context)

                # Ensure results is a list
                if not isinstance(results, list):
                    if results is None:
                        results = []
                    else:
                        results = [results]

                # Validate result types
                validated_results = []
                for result in results:
                    if isinstance(result, ValidationResultModel):
                        # Ensure layer and scope are set
                        if not result.layer:
                            result.layer = layer
                        if not result.scope:
                            result.scope = rule_id
                        validated_results.append(result)
                    else:
                        # Convert non-standard results
                        logger.warning(f"Rule {rule_id} returned non-standard result: {type(result)}")
                        converted = _convert_to_validation_result(result, rule_id, layer)
                        validated_results.append(converted)

                execution_time = time.time() - start_time
                logger.debug(f"Rule {rule_id} completed in {execution_time:.3f}s with {len(validated_results)} results")

                return validated_results

            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                execution_time = time.time() - start_time
                logger.error(f"Rule {rule_id} failed after {execution_time:.3f}s: {str(e)}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")

                # Create error result
                error_result = ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.GENERIC_VALIDATION_FAILED,
                    message=f"Validation rule '{rule_id}' execution failed: {str(e)}",
                    layer=layer,
                    scope=rule_id,
                    tags=["execution_error"] + actual_tags,
                    metrics={"execution_time_seconds": execution_time}
                )

                return [error_result]

        # Store metadata on function
        wrapper._validation_metadata = metadata
        wrapper._original_function = func

        # Register rule in registry
        target_registry = registry or get_global_registry()
        target_registry.register_rule(wrapper, metadata, overwrite=True)

        return wrapper

    return decorator


def require_state_manager(func: Callable) -> Callable:
    """
    Decorator to ensure validation rule has access to state manager.

    Parameters:
        func: Validation rule function

    Returns:
        Decorated function with state manager validation
    """
    @wraps(func)
    def wrapper(context: ValidationContext) -> ValidationResults:
        if context.state_manager is None:
            error_result = ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.MISSING_REQUIRED_DATA,
                message=f"Validation rule '{func.__name__}' requires state manager access",
                layer=context.execution_layer or "unknown",
                scope=func.__name__,
                tags=["state_manager", "dependency_error"]
            )
            return [error_result]

        return func(context)

    return wrapper


def validation_cache(ttl_seconds: int = 300,
                    cache_key_func: Optional[Callable] = None) -> Callable:
    """
    Decorator to cache validation results for performance optimization.

    Parameters:
        ttl_seconds: Time-to-live for cache entries
        cache_key_func: Function to generate cache key from context

    Returns:
        Decorated function with result caching
    """
    cache_storage = {}

    def default_cache_key(context: ValidationContext) -> str:
        """Generate default cache key from context."""
        key_parts = [
            str(id(context.state_manager)),
            str(hash(frozenset(context.config.items()) if context.config else 0)),
            context.execution_layer or "unknown"
        ]
        return "|".join(key_parts)

    def decorator(func: Callable) -> Callable:
        actual_cache_key_func = cache_key_func or default_cache_key

        @wraps(func)
        def wrapper(context: ValidationContext) -> ValidationResults:
            # Generate cache key
            cache_key = f"{func.__name__}:{actual_cache_key_func(context)}"
            current_time = time.time()

            # Check cache
            if cache_key in cache_storage:
                cached_result, timestamp = cache_storage[cache_key]
                if current_time - timestamp < ttl_seconds:
                    logger.debug(f"Cache hit for validation rule: {func.__name__}")
                    return cached_result
                else:
                    # Expired cache entry
                    del cache_storage[cache_key]

            # Execute function
            result = func(context)

            # Store in cache
            cache_storage[cache_key] = (result, current_time)
            logger.debug(f"Cached result for validation rule: {func.__name__}")

            return result

        # Add cache management methods
        def clear_cache():
            cache_storage.clear()

        def get_cache_stats():
            return {
                "entries": len(cache_storage),
                "ttl_seconds": ttl_seconds
            }

        wrapper.clear_cache = clear_cache
        wrapper.get_cache_stats = get_cache_stats

        return wrapper

    return decorator


def _convert_to_validation_result(result: Any, rule_id: str, layer: str) -> ValidationResultModel:
    """Convert arbitrary result to ValidationResultModel."""
    return ValidationResultModel(
        level=ValidationLevel.WARNING,
        code=ValidationCode.GENERIC_VALIDATION_FAILED,
        message=f"Converted non-standard result: {str(result)[:100]}",
        layer=layer,
        scope=rule_id,
        tags=["converted", "non_standard"],
        metadata={"original_result": str(result)[:500]}  # Truncate for safety
    )
