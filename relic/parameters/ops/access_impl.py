from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from ..types import ParameterCategory, ParameterValue
from .key_normalization import normalize_parameter_key

logger = logging.getLogger(__name__)

_NOT_FOUND = object()


def _is_interface_mapping_spec(spec: Any) -> bool:
    return (
        str(getattr(spec, "category", "")) == ParameterCategory.PROCESS_COUPLING.value
        and str(spec.dtype).lower() in {"str", "string"}
    )


def _apply_type_validation(
    manager: Any,
    value: Any,
    expected_type: Optional[type],
    type_key: str,
) -> Any:
    if expected_type is None:
        return value
    return manager._validate_type(value, expected_type, type_key)


def _resolve_from_storage(
    manager: Any,
    lookup_key: str,
    type_key: str,
    history_keys: Tuple[str, ...],
    units: Optional[str],
    expected_type: Optional[type],
) -> Any:
    value = manager.storage.get(lookup_key)
    if value is None:
        return _NOT_FOUND

    for key in history_keys:
        manager._access_history.add(key)

    spec = manager.parameters.get(lookup_key)
    if spec is not None:
        manager._check_units_for_spec(lookup_key, spec, source="get")
        if spec.requires_grad and spec.key in manager.nn_parameters:
            value = manager.nn_parameters[spec.key]
        if units is not None:
            value = manager._convert_parameter_units(value, spec, units)

    return _apply_type_validation(manager, value, expected_type, type_key)


def _resolve_from_parameters(
    manager: Any,
    lookup_key: str,
    type_key: str,
    history_keys: Tuple[str, ...],
    units: Optional[str],
    expected_type: Optional[type],
) -> Any:
    if lookup_key not in manager.parameters:
        return _NOT_FOUND

    for key in history_keys:
        manager._access_history.add(key)

    spec = manager.parameters[lookup_key]
    manager._check_units_for_spec(lookup_key, spec, source="get")
    value = manager.nn_parameters[spec.key] if spec.requires_grad and spec.key in manager.nn_parameters else spec.value

    if units is not None and not _is_interface_mapping_spec(spec):
        value = manager._convert_parameter_units(value, spec, units)

    return _apply_type_validation(manager, value, expected_type, type_key)


def _resolve_from_database(
    manager: Any,
    lookup_key: str,
    units: Optional[str],
    expected_type: Optional[type],
) -> Any:
    spec = manager.database.get_parameter(lookup_key)
    if spec is None:
        return _NOT_FOUND

    manager._check_units_for_spec(lookup_key, spec, source="get_db")
    manager.parameters[lookup_key] = spec
    manager.storage.set(lookup_key, spec.value)
    manager._access_history.add(lookup_key)
    if spec.requires_grad and spec.key not in manager.nn_parameters:
        try:
            manager._register_nn_parameter(spec)
        except TypeError as exc:
            logger.warning(
                "Skipping non-numeric learnable parameter %s: %s",
                spec.key,
                exc,
            )
    value = manager.nn_parameters.get(spec.key, spec.value)
    if units is not None:
        value = manager._convert_parameter_units(value, spec, units)
    return _apply_type_validation(manager, value, expected_type, lookup_key)


def get_impl(
    manager: Any,
    full_key: str,
    default: Any = None,
    units: Optional[str] = None,
    expected_type: Optional[type] = None,
    normalize: bool = True,
) -> ParameterValue:
    """
    Implementation of get() without caching layer.

    This helper centralizes the parameter lookup logic used by the manager.
    """
    if normalize:
        full_key = normalize_parameter_key(full_key)

    value = _resolve_from_storage(
        manager,
        lookup_key=full_key,
        type_key=full_key,
        history_keys=(full_key,),
        units=units,
        expected_type=expected_type,
    )
    if value is not _NOT_FOUND:
        return value

    value = _resolve_from_parameters(
        manager,
        lookup_key=full_key,
        type_key=full_key,
        history_keys=(full_key,),
        units=units,
        expected_type=expected_type,
    )
    if value is not _NOT_FOUND:
        return value

    mapped_key = manager.name_mapper.to_standard(full_key)
    mapped_key = normalize_parameter_key(mapped_key)
    if mapped_key != full_key:
        value = _resolve_from_storage(
            manager,
            lookup_key=mapped_key,
            type_key=full_key,
            history_keys=(full_key, mapped_key),
            units=units,
            expected_type=expected_type,
        )
        if value is not _NOT_FOUND:
            return value

        value = _resolve_from_parameters(
            manager,
            lookup_key=mapped_key,
            type_key=full_key,
            history_keys=(full_key, mapped_key),
            units=units,
            expected_type=expected_type,
        )
        if value is not _NOT_FOUND:
            return value

    value = _resolve_from_database(
        manager,
        lookup_key=full_key,
        units=units,
        expected_type=expected_type,
    )
    if value is not _NOT_FOUND:
        return value

    manager._missing_requests[full_key] += 1
    if "." in full_key:
        tail_key = full_key.split(".")[-1]
        manager._missing_requests[tail_key] += 1
    return default
