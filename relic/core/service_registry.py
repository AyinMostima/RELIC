"""Core service registry for shared component initialization."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional, cast
import logging

logger = logging.getLogger(__name__)

_COMPONENT_CACHE: Dict[str, Any] = {}


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Load a symbol lazily to keep core free from static cross-package imports."""
    module = importlib.import_module(module_name)
    symbol = getattr(module, symbol_name, None)
    if symbol is None:
        raise RuntimeError(f"Failed to resolve {symbol_name} from {module_name}")
    return symbol


def clear_service_cache() -> None:
    """Clear cached core services."""
    _COMPONENT_CACHE.clear()


def get_parameter_manager() -> Any:
    """Return the global ParameterManager singleton instance."""
    get_global_manager = cast(
        Callable[[], Any],
        _load_symbol("relic.parameters", "get_global_manager"),
    )

    manager = get_global_manager()
    logger.info("Parameter manager initialized")
    return manager


def get_simulator(config: Optional[Dict[str, Any]] = None) -> Any:
    """Create an isolated simulator session through the canonical factory."""
    create_simulator = cast(
        Callable[[Optional[Dict[str, Any]]], Any],
        _load_symbol("relic.core.simulator", "create_simulator"),
    )

    simulator = create_simulator(config)
    logger.info("Isolated simulator session created")
    return simulator


def get_data_io_manager() -> Any:
    """Return the DataIO manager instance (cached)."""
    if "data_io_manager" in _COMPONENT_CACHE:
        return _COMPONENT_CACHE["data_io_manager"]

    data_io_cls = cast(Callable[[], Any], _load_symbol("relic.data.io", "DataIO"))

    manager = data_io_cls()
    _COMPONENT_CACHE["data_io_manager"] = manager
    logger.info("Data I/O manager initialized")
    return manager


__all__ = [
    "clear_service_cache",
    "get_parameter_manager",
    "get_simulator",
    "get_data_io_manager",
]
