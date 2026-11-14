from __future__ import annotations

from typing import Any, Optional, Dict

from ..types import ParameterValue, UnitValidationMode
from ..ops.key_normalization import normalize_parameter_key
from ..ops.namespaces import namespace_keys
from ..ops.type_utils import caller_owned_value

_NOT_FOUND = object()


def _can_use_direct_fast_path(
    units: Optional[str],
    expected_type: Optional[type],
) -> bool:
    return units is None and expected_type is None


def _try_direct_storage_hit(
    manager: Any,
    raw_key: Any,
    direct_key: str,
    units: Optional[str],
    expected_type: Optional[type],
) -> Any:
    if manager.unit_validation_mode != UnitValidationMode.OFF:
        return _NOT_FOUND
    if not _can_use_direct_fast_path(units=units, expected_type=expected_type):
        return _NOT_FOUND
    if not hasattr(manager.storage, "exists"):
        return _NOT_FOUND

    for candidate in (raw_key, direct_key):
        if not isinstance(candidate, str):
            continue
        if not manager.storage.exists(candidate):
            continue

        # Every successful read is recorded in the access history (the
        # restart contract reads it), whichever read path served it.
        if candidate in manager.nn_parameters:
            manager._access_history.add(candidate)
            return manager._unwrap_trainable_value(
                manager.nn_parameters[candidate],
                expected_type,
            )

        direct_value = manager.storage.get(candidate)
        if direct_value is not None:
            manager._access_history.add(candidate)
            return manager._unwrap_trainable_value(
                direct_value,
                expected_type,
            )

    return _NOT_FOUND


def _compose_full_key(key: str, namespace: Optional[str]) -> str:
    raw_full_key = f"{namespace}.{key}" if namespace and not key.startswith(namespace) else key
    return normalize_parameter_key(raw_full_key)


def _build_cache_key(
    manager: Any,
    full_key: str,
    units: Optional[str],
    expected_type: Optional[type],
) -> Any:
    return (
        full_key,
        units or "raw",
        manager._format_expected_type(expected_type),
        getattr(manager.validation_level, "value", str(manager.validation_level)),
        getattr(manager.unit_validation_mode, "value", str(manager.unit_validation_mode)),
    )


def _try_get_cached_value(
    manager: Any,
    full_key: str,
    units: Optional[str],
    expected_type: Optional[type],
) -> Any:
    if not manager._cache_enabled or manager._parameter_cache is None:
        return _NOT_FOUND

    cache_key = _build_cache_key(
        manager=manager,
        full_key=full_key,
        units=units,
        expected_type=expected_type,
    )
    cached_value = manager._parameter_cache.get(cache_key)
    if cached_value is None:
        return _NOT_FOUND

    manager._access_history.add(full_key)
    return cached_value


def _put_cached_value(
    manager: Any,
    full_key: str,
    units: Optional[str],
    expected_type: Optional[type],
    result: Any,
    default: Any,
) -> None:
    if not manager._cache_enabled or manager._parameter_cache is None:
        return
    if result is default:
        return

    cache_key = _build_cache_key(
        manager=manager,
        full_key=full_key,
        units=units,
        expected_type=expected_type,
    )
    manager._parameter_cache.put(cache_key, result)


def _fetch_unwrapped_value(
    manager: Any,
    full_key: str,
    default: Any,
    units: Optional[str],
    expected_type: Optional[type],
) -> Any:
    result = manager._get_impl(full_key, default, units, expected_type)
    return manager._unwrap_trainable_value(result, expected_type)


def _build_namespace_tree(
    manager: Any,
    full_key: str,
    child_keys: Any,
) -> Dict[str, Any]:
    prefix = f"{full_key}."
    prefix_len = len(prefix)
    members = [child_key for child_key in child_keys if child_key.startswith(prefix)]
    # A key names a parameter or a namespace, never both (architecture
    # overview Section 6.1), so every member has one place in the tree. A
    # member that is the dot prefix of another member raises before any value
    # is read, whatever the member order; the loop below then descends only
    # into sub-trees it created and never writes into a stored value.
    member_set = set(members)
    for child_key in members:
        parts = child_key[prefix_len:].split(".")
        for depth in range(1, len(parts)):
            holder = prefix + ".".join(parts[:depth])
            if holder in member_set:
                raise ValueError(
                    f"Namespace tree '{full_key}': parameter key '{holder}' is also "
                    f"the namespace of parameter key '{child_key}'; a key names a "
                    "parameter or a namespace, never both"
                )

    tree: Dict[str, Any] = {}
    for child_key in members:
        # A database member the manager does not hold yet is taken in as a
        # single-key metadata read takes it (specification, stored value, unit
        # check, learnable binding). Each leaf is the stored value, or the
        # specification value of a key that has none stored.
        spec = manager.get_parameter(child_key)
        value = manager.storage.get(child_key)
        if value is None and spec is not None:
            value = spec.value
        remainder = child_key[prefix_len:]
        cursor = tree
        parts = remainder.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return tree


def _maybe_resolve_namespace_tree(manager: Any, full_key: str) -> Any:
    if manager.get_parameter(full_key) is not None:
        return _NOT_FOUND

    # One membership rule for every namespace query (ops.namespaces): the tree
    # holds every key below "<full_key>." that the manager holds or the
    # database defines, whatever was loaded on demand before this read.
    child_keys = namespace_keys(manager, full_key)
    if child_keys:
        return _build_namespace_tree(
            manager=manager,
            full_key=full_key,
            child_keys=child_keys,
        )
    return _NOT_FOUND


def _raise_if_required_missing(required: bool, result: Any, default: Any, full_key: str) -> None:
    if required and (result is default or result is None):
        raise KeyError(f"Required parameter '{full_key}' not found")


def get_value(
    manager: Any,
    key: str,
    default: Any = None,
    *,
    namespace: Optional[str] = None,
    units: Optional[str] = None,
    expected_type: Optional[type] = None,
    required: bool = False,
) -> ParameterValue:
    """
    Get parameter value by key with optional namespace filtering and unit conversion.

    Notes
    -----
    This helper keeps access logic centralized while the manager remains the facade.
    Every value it hands out from the manager (direct storage read, query-result
    cache, resolved read, namespace tree) passes caller_owned_value, the one copy
    rule of value reads; the query cache keeps the stored object, never a copy a
    caller holds. A caller-supplied default is returned as it is.
    """
    raw_key = key if isinstance(key, str) else key
    direct_key = normalize_parameter_key(key)
    # A failure of the direct storage read propagates; it is not retried on the
    # slow path.
    direct_result = _try_direct_storage_hit(
        manager=manager,
        raw_key=raw_key,
        direct_key=direct_key,
        units=units,
        expected_type=expected_type,
    )
    if direct_result is not _NOT_FOUND:
        return caller_owned_value(direct_result)

    full_key = _compose_full_key(key=key, namespace=namespace)

    cached_value = _try_get_cached_value(
        manager=manager,
        full_key=full_key,
        units=units,
        expected_type=expected_type,
    )
    if cached_value is not _NOT_FOUND:
        return caller_owned_value(cached_value)

    result = _fetch_unwrapped_value(
        manager=manager,
        full_key=full_key,
        default=default,
        units=units,
        expected_type=expected_type,
    )

    if result is None and default is None and not required:
        # A failure of the namespace lookup (database enumeration, listing)
        # propagates; only a namespace without children leaves the result None.
        namespace_tree = _maybe_resolve_namespace_tree(manager=manager, full_key=full_key)
        if namespace_tree is not _NOT_FOUND:
            # A tree is a view over many keys: it is built from the keys held
            # at this read and never enters the per-key query cache.
            return caller_owned_value(namespace_tree)

    _put_cached_value(
        manager=manager,
        full_key=full_key,
        units=units,
        expected_type=expected_type,
        result=result,
        default=default,
    )
    _raise_if_required_missing(required=required, result=result, default=default, full_key=full_key)

    if result is default:
        return result
    return caller_owned_value(result)


