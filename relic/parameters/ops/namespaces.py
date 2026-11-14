from __future__ import annotations

import fnmatch
import logging
from typing import Any, Optional, List, Dict, Union

from ..types import ParameterCategory, ParameterValue
from ..spec import ParameterSpec
from .key_normalization import normalize_parameter_key
from .type_utils import caller_owned_value

logger = logging.getLogger(__name__)


def list_parameters(
    manager: Any,
    namespace: Optional[str] = None,
    pattern: Optional[str] = None,
    category: Optional[ParameterCategory] = None,
) -> List[str]:
    """
    List parameter keys with optional filtering.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    namespace : Optional[str]
        Select the keys below "<namespace>." (the ParameterStorage.list rule).
    pattern : Optional[str]
        fnmatch wildcard pattern (the ParameterStorage.list rule).
    category : Optional[ParameterCategory]
        Filter by parameter category.

    Returns
    -------
    List[str]
        Matching parameter keys.
    """
    storage_result = manager.storage.list(namespace=namespace, pattern=pattern)

    # ParameterStorage.list returns List[str]; any other result is a broken
    # storage owner and raises instead of silently listing nothing.
    if not isinstance(storage_result, list):
        raise TypeError(
            f"storage.list must return a list of parameter keys, got {type(storage_result).__name__}"
        )

    # Registered specifications not (yet) in storage take the same namespace and
    # pattern filters that ParameterStorage.list applied to the stored keys.
    registered = [
        key
        for key in manager.parameters.keys()
        if (not namespace or key.startswith(f"{namespace}."))
        and (not pattern or fnmatch.fnmatch(key, pattern))
    ]
    filtered_keys = list(set(storage_result).union(registered))

    if category:
        category_filtered: List[str] = []
        for key in filtered_keys:
            if key in manager.parameters and manager.parameters[key].category == category:
                category_filtered.append(key)
        filtered_keys = category_filtered

    return sorted(filtered_keys)


def namespace_keys(manager: Any, namespace: str) -> List[str]:
    """
    Complete key list of one namespace.

    The one membership rule of every namespace query (get_namespace_parameters
    and the tree that manager.get("<namespace>") returns): the keys below
    "<namespace>." (the ParameterStorage.list rule; the namespace key itself
    is not a member) that the manager holds or the database defines. The
    database is enumerated in full, so the result does not depend on which
    keys were loaded on demand before the query.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    namespace : str
        Canonical namespace (e.g., "hydrosphere.evapotranspiration").

    Returns
    -------
    List[str]
        Sorted parameter keys of the namespace.
    """
    # A database that cannot be enumerated fails the query; a namespace built
    # from the lazily cached keys alone would be silently incomplete.
    db_params = manager.database.load_all()
    prefix = f"{namespace}."
    held_keys = manager.list(namespace=namespace)
    database_keys = [key for key in db_params if key.startswith(prefix)]
    return sorted(set(held_keys).union(database_keys))


def get_namespace_parameters(
    manager: Any,
    namespace: str,
    include_subnamespaces: bool = True,
    return_specs: bool = False,
) -> Dict[str, Union[ParameterValue, ParameterSpec]]:
    """
    Retrieve all parameters within the specified namespace.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    namespace : str
        Namespace (e.g., "relic.pedosphere.physical").
    include_subnamespaces : bool, optional
        Whether to include nested namespaces beneath the given prefix.
    return_specs : bool, optional
        If True, return ParameterSpec objects instead of raw values.

    Returns
    -------
    Dict[str, Union[ParameterValue, ParameterSpec]]
        Mapping of parameter keys to values or specifications.
    """
    if not namespace:
        raise ValueError("namespace must be a non-empty string")

    canonical_namespace = normalize_parameter_key(namespace.rstrip("."))
    namespace_candidates = [canonical_namespace]
    if canonical_namespace.startswith("relic.") and len(canonical_namespace.split(".")) > 1:
        namespace_candidates.append(canonical_namespace[len("relic."):])

    keys: List[str] = []
    namespace_used = canonical_namespace
    for candidate in namespace_candidates:
        combined_keys = namespace_keys(manager, candidate)
        if combined_keys:
            keys = combined_keys
            namespace_used = candidate
            break

    prefix = f"{namespace_used}."
    results: Dict[str, Union[ParameterValue, ParameterSpec]] = {}

    for key in keys:
        if not include_subnamespaces and key.startswith(prefix):
            suffix = key[len(prefix):]
            if "." in suffix:
                continue

        spec = manager._ensure_parameter_spec(key)
        if return_specs:
            if spec is None:
                raise KeyError(
                    f"No ParameterSpec available for key '{key}' in namespace '{canonical_namespace}'"
                )
            results[key] = spec
        else:
            # A value read hands out a caller-owned copy (the copy rule of
            # manager.get); return_specs gives the specifications the
            # manager holds, as get_parameter does.
            if spec is not None:
                results[key] = caller_owned_value(spec.value)
            else:
                results[key] = manager.get(key)

    return results
