from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Dict, Optional

from ..types import (
    ParameterValue,
    ValidationContext,
    UnitValidationMode,
    ParameterCategory,
)
from ..spec import ParameterSpec
from ...data.standards.unit_converter import unit_converter, ConversionContext

logger = logging.getLogger(__name__)


def should_skip_unit_validation(manager: Any, spec: Optional[ParameterSpec]) -> bool:
    if spec is None:
        return True
    category = str(getattr(spec, "category", ""))
    dtype = str(getattr(spec, "dtype", "")).lower()
    if category == ParameterCategory.PROCESS_COUPLING.value and dtype in {"str", "string"}:
        return True
    return False


def validate_units_for_spec(
    manager: Any,
    key: str,
    spec: Optional[ParameterSpec],
    strict: bool,
    source: str,
) -> Optional[Dict[str, Any]]:
    if should_skip_unit_validation(manager, spec):
        return None
    if spec is None:
        return None
    units_str = str(getattr(spec, "units", "")).strip()
    context = ValidationContext(
        all_parameters=manager.storage.get_all(),
        strict_mode=strict,
    )
    context.warnings_as_errors = strict
    result = manager.validator._validate_units(key, spec.value, spec, context)
    errors = list(result.errors)
    warnings_list = list(result.warnings)

    if hasattr(unit_converter, "get_dimension"):
        dimension = unit_converter.get_dimension(units_str)
        if dimension is None and not strict:
            warnings_list.append(
                f"{key}: Unit '{units_str}' has no known dimension"
            )

    if errors or warnings_list:
        issue = {
            "key": key,
            "units": units_str,
            "errors": errors,
            "warnings": warnings_list,
            "source": source,
        }
        if strict:
            message = "; ".join(errors + warnings_list)
            raise ValueError(f"Unit validation failed for {key}: {message}")
        return issue
    return None


def check_units_for_spec(manager: Any, key: str, spec: Optional[ParameterSpec], source: str) -> None:
    if manager.unit_validation_mode == UnitValidationMode.OFF:
        return
    if key in manager._unit_validation_checked:
        return
    strict = manager.unit_validation_mode == UnitValidationMode.STRICT
    issue = validate_units_for_spec(manager, key, spec, strict=strict, source=source)
    if issue:
        manager._unit_validation_issues.append(issue)
    manager._unit_validation_checked.add(key)


def convert_parameter_units(
    manager: Any,
    value: ParameterValue,
    spec: ParameterSpec,
    target_units: str,
) -> ParameterValue:
    """
    Convert parameter value to target units with caching.

    A conversion error propagates from its single attempt, and a failure of
    the conversion cache itself propagates as well; neither is retried or
    bypassed. Only a successful conversion is cached. The result depends only
    on the parameter key, its units, the target units and the value, which
    form the cache key.
    """
    if not manager._cache_enabled or manager._unit_conversion_cache is None:
        return convert_parameter_units_impl(manager, value, spec, target_units)

    source_units = spec.units

    if source_units == target_units:
        return value

    if not isinstance(value, (int, float)):
        return convert_parameter_units_impl(manager, value, spec, target_units)

    cache_key = (spec.key, source_units, target_units, value)

    cached_result = manager._unit_conversion_cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    result = convert_parameter_units_impl(manager, value, spec, target_units)
    manager._unit_conversion_cache.put(cache_key, result)
    return result


def _normalize_unit_alias(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    unit_str = unit.strip()
    alias_map = {
        "g/mol": "g mol-1",
        "kg/mol": "kg mol-1",
    }
    unit_str = alias_map.get(unit_str, unit_str)
    unit_str = re.sub(r"([A-Za-z]+)\^(-?\d+)", r"\1\2", unit_str)
    return unit_str


def _should_warn_on_unit_conversion(
    source_units: Optional[str],
    target_units: Optional[str],
) -> bool:
    if source_units is None or target_units is None:
        return False
    silent_pairs = {("K", "degC"), ("degC", "K")}
    return (source_units, target_units) not in silent_pairs


def _is_dimensionless_equivalent(unit: Optional[str]) -> bool:
    return unit in {"", "dimensionless", "1", None}


def convert_parameter_units_impl(
    manager: Any,
    value: ParameterValue,
    spec: ParameterSpec,
    target_units: str,
) -> ParameterValue:
    """Convert parameter value to target units.

    A conversion the authoritative unit converter cannot perform raises
    ValueError naming the parameter, in every validation level: an unconverted
    value is never returned in place of the requested units.
    """
    source_units = _normalize_unit_alias(spec.units)
    target_units = _normalize_unit_alias(target_units)

    if source_units == target_units:
        return value

    dtype = str(getattr(spec, "dtype", "")).lower()
    non_numeric_types = (str, bytes, bool, dict, list, tuple, set)
    if (
        (isinstance(value, non_numeric_types) or dtype in {"str", "string", "bool", "boolean", "dict", "list", "object", "json"})
        and _is_dimensionless_equivalent(source_units)
        and _is_dimensionless_equivalent(target_units)
    ):
        return value

    if source_units and target_units and source_units != target_units:
        if _should_warn_on_unit_conversion(source_units, target_units):
            warnings.warn(
                f"Parameter '{spec.key}' requested with units '{target_units}' "
                f"but stored units are '{source_units}'. Performing conversion.",
                UserWarning,
                stacklevel=3,
            )

    # unit_converter.convert raises ValueError for an unsupported conversion or
    # insufficient context and TypeError for an unsupported value container;
    # both are re-raised with the parameter named. Other errors propagate.
    try:
        context = ConversionContext()
        converted_value = unit_converter.convert(value, spec.key, source_units, target_units, context)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Unit conversion failed for parameter '{spec.key}': "
            f"{source_units} -> {target_units}. Error: {exc}"
        ) from exc
    logger.debug(
        f"Converted parameter '{spec.key}' from {source_units} to {target_units}: "
        f"{value} -> {converted_value}"
    )
    return converted_value
