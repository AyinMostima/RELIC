"""Bootstrap helpers for UnifiedParameterManager initialization."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from .nn_params import bind_learnable_value

logger = logging.getLogger(__name__)


def adopt_database_specs(manager: Any, specs: Dict[str, Any]) -> None:
    """Take database specifications into the manager with their owners in step.

    Every specification gets its stored value, and every learnable one a live
    binding holding that value (``bind_learnable_value``). A learnable
    specification without a tensor form is a malformed database entry: it
    raises TypeError before any owner changes.
    """
    for key, spec in specs.items():
        if getattr(spec, "requires_grad", False):
            try:
                spec.to_tensor(manager.device)
            except TypeError as exc:
                raise TypeError(
                    f"Learnable parameter {key} has no tensor form and cannot be bound: {exc}"
                ) from exc
    manager.parameters.update(specs)
    for key, spec in specs.items():
        manager.storage.set(key, spec.value)
        bind_learnable_value(manager, spec)


def preload_parameters(manager: Any, warmup_cache: bool = True) -> None:
    """Complete the manager with the database specifications it does not hold yet.

    A key the manager already holds (loaded on demand, registered, or written
    with set_value) keeps its specification, stored value and live binding: a
    preload requested after first use never puts a held value back to its
    database value. Caches are warmed on request.
    """
    if manager._preloaded:
        return
    logger.info("Loading parameters from database...")
    loaded_params = manager.database.load_all()
    missing_params = {
        key: spec for key, spec in loaded_params.items() if key not in manager.parameters
    }
    for key, spec in missing_params.items():
        manager._check_units_for_spec(key, spec, source="preload")
    adopt_database_specs(manager, missing_params)

    logger.info("Loaded %s parameters into memory", len(missing_params))

    if warmup_cache and manager._cache_enabled:
        # A failing warm-up read propagates; preload is then not marked done.
        warmup_stats = manager.warmup_cache()
        logger.info(
            "Cache warmup: %s/%s parameters pre-loaded",
            warmup_stats.get("cached_count"),
            warmup_stats.get("attempted_count"),
        )

    manager._preloaded = True


def build_runtime_registry(manager: Any) -> Dict[str, Dict[str, Any]]:
    """
    Build the runtime registry view for consistency tests.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Runtime registry mapping parameter keys to serialized specs.
    """
    runtime_reg: Dict[str, Dict[str, Any]] = {}

    if hasattr(manager.database, "cache") and manager.database.cache:
        for key, spec in manager.database.cache.items():
            if isinstance(spec, dict):
                runtime_reg[key] = spec
            elif hasattr(spec, "to_dict"):
                runtime_reg[key] = spec.to_dict()
            elif hasattr(spec, "__dict__"):
                value = getattr(spec, "value", None)
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.tolist()
                runtime_reg[key] = {
                    "value": value,
                    "units": getattr(spec, "units", None),
                    "bounds": getattr(spec, "bounds", None),
                    "description": getattr(spec, "description", ""),
                }

    for key, spec in manager.parameters.items():
        if isinstance(spec, dict):
            runtime_reg[key] = spec
        elif hasattr(spec, "to_dict"):
            runtime_reg[key] = spec.to_dict()
        elif hasattr(spec, "__dict__"):
            value = getattr(spec, "value", None)
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.tolist()
            runtime_reg[key] = {
                "value": value,
                "units": getattr(spec, "units", None),
                "bounds": getattr(spec, "bounds", None),
                "description": getattr(spec, "description", ""),
            }

    return runtime_reg


def load_initial_parameters(manager: Any) -> None:
    """Load initial parameters from the database, each with its stored value and binding.

    Every database specification replaces what the manager holds under its
    key and is unit-checked before it is adopted (check-only records an
    issue, strict raises), and no query result or unit conversion cached
    before the load is served after it. The callers are a database switch
    (do_configure with a new database_path) and debug_report_units on an
    empty manager; initialize() loads nothing.
    """
    try:
        db_parameters = manager.database.load_all()
        # A unit check recorded for a held specification does not cover the
        # database specification that replaces it under the same key.
        manager._unit_validation_checked.difference_update(db_parameters)
        for key, spec in db_parameters.items():
            manager._check_units_for_spec(key, spec, source="initialize")
        adopt_database_specs(manager, db_parameters)
        logger.info("Loaded %s parameters from database", len(db_parameters))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise RuntimeError(
            "Failed to load initial parameters from YAML database; "
            "fallback defaults are disabled by architecture policy."
        ) from exc
    finally:
        # Cached results were computed from the specifications, stored values
        # and bindings this load replaces (also when it stopped part way);
        # clear both caches, as set_value and delete do.
        manager.clear_cache()

