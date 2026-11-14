from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, List, Union, cast

from ..types import ParameterCategory, ParameterValue
from ..spec import ParameterSpec
from ..manager import UnifiedParameterManager
from ...core.variable_mapper import variable_name_mapper

logger = logging.getLogger(__name__)

_global_manager: Optional[UnifiedParameterManager] = None
# Reentrant: initialize() of the new manager may reach get_global_manager()
# on the same thread and must then receive that manager.
_global_manager_lock = threading.RLock()


def get_global_manager(
    preload_all: bool = False,
    warmup_cache: bool = False,
) -> UnifiedParameterManager:
    """
    Get the process-wide parameter manager (the only global entry point).

    Returns:
        UnifiedParameterManager: the process-wide instance

    The instance is created once under a lock and initialized on first
    access. ``UnifiedParameterManager(...)`` itself always builds an
    independent manager; code that needs a separate owner constructs one and
    injects it explicitly.
    """
    global _global_manager
    if _global_manager is None:
        with _global_manager_lock:
            if _global_manager is None:
                manager = UnifiedParameterManager(
                    preload_all=preload_all,
                    warmup_cache=warmup_cache,
                )
                _global_manager = manager
                try:
                    manager.initialize()
                    # The process-wide variable aliases come from the canonical
                    # manager's catalogs; a failed refresh raises and leaves no
                    # half-built global manager behind.
                    variable_name_mapper.refresh_from_parameter_manager(manager)
                except BaseException:
                    _global_manager = None
                    raise
                logger.info("Process-wide UnifiedParameterManager created")
                return manager
    if preload_all and not getattr(_global_manager, "_preloaded", False):
        _global_manager._preload_parameters(warmup_cache=warmup_cache)
    return _global_manager


def reset_global_manager() -> None:
    """
    Reset the global parameter manager instance.

    Used primarily for testing and debugging. Creates a fresh instance
    on next access to get_global_manager().
    """
    global _global_manager
    with _global_manager_lock:
        _global_manager = None


# Convenience functions for direct access

def get(
    key: str,
    default: Any = None,
    namespace: Optional[str] = None,
    units: Optional[str] = None,
    expected_type: Optional[type] = None,
    required: bool = False,
) -> ParameterValue:
    """Convenience function - get parameter value from global manager."""
    return get_global_manager().get(
        key=key,
        default=default,
        namespace=namespace,
        units=units,
        expected_type=expected_type,
        required=required,
    )


def set_value(
    key: str,
    value: ParameterValue,
    units: Optional[str] = None,
    validate: bool = True,
    notify: bool = True,
) -> None:
    """Convenience function - set parameter value in global manager."""
    get_global_manager().set_value(key, value, units, validate, notify)


def list_parameters(
    namespace: Optional[str] = None,
    pattern: Optional[str] = None,
    category: Optional[ParameterCategory] = None,
) -> List[str]:
    """Convenience function - list parameters from global manager."""
    keys = get_global_manager().list(namespace, pattern, category)
    return [str(key) for key in keys]


def debug_report_units() -> Dict[str, Any]:
    """
    Convenience function - generate a unit validation report using the global manager.
    """
    report = get_global_manager().debug_report_units()
    if isinstance(report, dict):
        return cast(Dict[str, Any], report)
    return {}


def get_namespace_parameters(
    namespace: str,
    include_subnamespaces: bool = True,
    return_specs: bool = False,
) -> dict[str, Union[ParameterValue, ParameterSpec]]:
    """
    Convenience wrapper for retrieving namespace-scoped parameters.
    """
    result = get_global_manager().get_namespace_parameters(
        namespace=namespace,
        include_subnamespaces=include_subnamespaces,
        return_specs=return_specs,
    )
    if isinstance(result, dict):
        return cast(dict[str, Union[ParameterValue, ParameterSpec]], result)
    return {}
