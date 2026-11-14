"""
Optional integration hooks for RELIC extensions.

These helpers do not patch or modify existing RELIC classes. They only call
public APIs that already exist on simulator and orchestration registry objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .discovery import discover_extensions
from .loader import _partition_valid_extensions, load_extensions, unload_extension
from .params import load_extension_parameters

if TYPE_CHECKING:
    from relic.core.simulator import RELICSimulator
    from relic.orchestration.node_registry import NodeRegistry


def load_and_register_extensions(
    simulator: "RELICSimulator",
    *,
    parameter_manager: Any,
    extra_dirs: Optional[List[Path]] = None,
    validate: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Discover, validate, load, and register extension functors on a simulator.

    ``parameter_manager`` is required: RELICSimulator owns no parameter
    manager, so the caller names the manager that receives the extension
    parameters (normally ``relic.parameters.get_global_manager()``), and
    every loaded functor is bound to that same manager before registration.
    Extensions are validated first; parameters are loaded only for the
    extensions that passed. Strict mode (the default) raises on any
    validation failure; ``strict=False`` skips invalid extensions and lists
    them with their reasons in ``summary["skipped_extensions"]``. A
    registration refused by the simulator raises the simulator's error; the
    loader records of the loaded functors are kept, so the extension can be
    released with ``unload_from_runtime``.
    """

    discovered = discover_extensions(extra_dirs=extra_dirs)
    loadable, skipped = _partition_valid_extensions(
        discovered,
        validate=validate,
        strict=strict,
    )

    parameter_loaded_count = 0
    parameter_source_paths: List[str] = []
    for info in loadable:
        param_summary = load_extension_parameters(parameter_manager, info)
        parameter_loaded_count += int(param_summary["loaded_count"])
        parameter_source_paths.extend(
            [str(path) for path in param_summary["source_paths"]]
        )

    # Validation already ran for exactly these extensions.
    loaded = load_extensions(
        validate=False,
        strict=strict,
        discovered_infos=loadable,
    )
    skipped_extensions = skipped + loaded.skipped_extensions

    summary: Dict[str, Any] = {
        "loaded_count": len(loaded),
        "registered_count": 0,
        "parameter_loaded_count": parameter_loaded_count,
        "parameter_source_paths": sorted(set(parameter_source_paths)),
        "registered_functors": [],
        "skipped_extensions": skipped_extensions,
    }

    for functor_name, functor in loaded.items():
        functor.parameter_manager = parameter_manager
        execution_group = getattr(functor, "_extension_execution_group", None)
        simulator.register_functor(functor, execution_group=execution_group)
        summary["registered_count"] += 1
        summary["registered_functors"].append(functor_name)

    return summary


def register_with_orchestration(
    registry: "NodeRegistry",
    extra_dirs: Optional[List[Path]] = None,
    validate: bool = True,
    strict: bool = True,
) -> List[str]:
    """
    Register extension functors with an existing orchestration NodeRegistry.
    """

    loaded = load_extensions(
        extra_dirs=extra_dirs,
        validate=validate,
        strict=strict,
    )

    registered_node_type_ids: List[str] = []
    for functor in loaded.values():
        definition = registry.register_functor_instance(functor)
        registered_node_type_ids.append(definition.node_type_id)
    return registered_node_type_ids


def unload_from_runtime(
    simulator: "RELICSimulator",
    extension_name: str,
    registry: Optional["NodeRegistry"] = None,
    strict: bool = False,
    native_drain_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Unload an extension from simulator and optional orchestration registry.

    ``native_drain_timeout`` bounds the wait for native plugin calls still in
    flight; an expired drain keeps the loader record so the unload can be
    retried (see ``unload_extension``).
    """
    return unload_extension(
        extension_name,
        simulator=simulator,
        registry=registry,
        strict=strict,
        native_drain_timeout=native_drain_timeout,
    )
