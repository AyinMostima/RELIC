"""
Core architectural constants shared across RELIC modules.

This module is limited to non-physical framework constants such as
scale hierarchy metadata. Physical constants MUST be resolved through
``relic.utils.physical_constants.get_physical_constant`` with SSOT data
stored in ``relic/parameters/database/relic/physical/constants.yaml``.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, List, Optional, Tuple, cast
import logging

logger = logging.getLogger(__name__)


_SCALE_HIERARCHY_CACHE: Optional[List[str]] = None


def _load_parameter_symbols() -> Tuple[Callable[[], Any], Any, Callable[[Any], str]]:
    """Resolve parameter access symbols lazily to avoid static cross-package coupling."""
    manager_module = importlib.import_module("relic.parameters.manager")
    namespaces_module = importlib.import_module("relic.parameters.namespaces")

    get_global_manager = cast(
        Callable[[], Any],
        getattr(manager_module, "get_global_manager", None),
    )
    parameter_namespace = getattr(namespaces_module, "ParameterNamespace", None)
    resolve_namespace = cast(
        Callable[[Any], str],
        getattr(namespaces_module, "resolve_namespace", None),
    )

    if get_global_manager is None or parameter_namespace is None or resolve_namespace is None:
        raise RuntimeError("Failed to resolve parameter namespace helpers for scale hierarchy")
    return get_global_manager, parameter_namespace, resolve_namespace


def _load_scale_hierarchy(parameter_manager: Any = None) -> List[str]:
    """Load scale hierarchy from the parameter database."""
    get_global_manager, parameter_namespace, resolve_namespace = _load_parameter_symbols()
    pm = parameter_manager or get_global_manager()
    if pm is None:
        raise RuntimeError("Parameter manager not initialized for scale hierarchy")
    namespace = resolve_namespace(parameter_namespace.COUPLING_SCALES_HIERARCHY)
    levels = pm.get(
        "levels",
        namespace=namespace,
        expected_type=list,
        required=True,
    )
    if not isinstance(levels, list) or not levels:
        raise ValueError("Scale hierarchy levels missing or invalid")
    return list(levels)


def get_scale_hierarchy(parameter_manager: Any = None, refresh: bool = False) -> List[str]:
    """
    Return the configured scale hierarchy, reloading from the parameter database when requested.
    """
    global _SCALE_HIERARCHY_CACHE
    if refresh or _SCALE_HIERARCHY_CACHE is None:
        _SCALE_HIERARCHY_CACHE = _load_scale_hierarchy(parameter_manager)
    return list(_SCALE_HIERARCHY_CACHE)


def refresh_scale_hierarchy(parameter_manager: Any = None) -> List[str]:
    """Force refresh of the cached hierarchy from the parameter database."""
    return get_scale_hierarchy(parameter_manager=parameter_manager, refresh=True)


SCALE_HIERARCHY = get_scale_hierarchy()

__all__ = ["SCALE_HIERARCHY", "get_scale_hierarchy", "refresh_scale_hierarchy"]
