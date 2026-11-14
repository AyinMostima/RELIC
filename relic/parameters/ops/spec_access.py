from __future__ import annotations

import logging
from typing import Any, Optional

from ..spec import ParameterSpec
from .key_normalization import normalize_parameter_key

logger = logging.getLogger(__name__)


def ensure_parameter_spec(manager: Any, key: str) -> Optional[ParameterSpec]:
    """
    Ensure a ParameterSpec object is available for the given key.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    key : str
        Canonical parameter key.

    Returns
    -------
    Optional[ParameterSpec]
        Parameter specification if it exists in memory or the database.
    """
    canonical_key = normalize_parameter_key(key)
    spec = manager.parameters.get(canonical_key)
    if spec is not None:
        return spec

    db_spec = manager.database.get_parameter(canonical_key)
    if db_spec is not None:
        manager.parameters[canonical_key] = db_spec
        if hasattr(manager, "storage") and manager.storage:
            manager.storage.set(canonical_key, db_spec.value)
    return db_spec


def get_spec(manager: Any, key: str) -> Optional[ParameterSpec]:
    """
    Get parameter specification by key.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    key : str
        Parameter key.

    Returns
    -------
    Optional[ParameterSpec]
        Parameter specification or None.
    """
    canonical_key = normalize_parameter_key(key)
    spec = manager.parameters.get(canonical_key)
    if spec is None and canonical_key != key:
        spec = manager.parameters.get(key)
    if spec is None:
        spec = manager.database.get_parameter(canonical_key)
        if spec is not None:
            manager.parameters[canonical_key] = spec
            manager.storage.set(canonical_key, spec.value)
    if spec is not None:
        manager._check_units_for_spec(spec.key, spec, source="get_spec")
    if spec is not None and getattr(spec, "requires_grad", False) and spec.key not in manager.nn_parameters:
        try:
            manager._register_nn_parameter(spec)
        except TypeError as exc:
            logger.warning(
                "Skipping non-numeric learnable parameter %s: %s",
                spec.key,
                exc,
            )
    return spec


def get_parameter(manager: Any, key: str, namespace: Optional[str] = None) -> Optional[ParameterSpec]:
    """Return ParameterSpec metadata through the canonical manager entry point."""
    full_key = f"{namespace}.{key}" if namespace and not key.startswith(namespace) else key
    full_key = normalize_parameter_key(full_key)
    return get_spec(manager, full_key)


def exists(manager: Any, key: str) -> bool:
    """Return True if the parameter exists in storage or loaded specs."""
    canonical_key = normalize_parameter_key(key)
    if hasattr(manager, "storage") and manager.storage:
        return manager.storage.exists(canonical_key)
    return canonical_key in manager.parameters
