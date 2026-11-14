"""Lifecycle helpers for UnifiedParameterManager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import torch

from ..database_manager import ParameterDatabase
from ..types import ValidationLevel
from ...core.managers.metadata import ManagerMetadata, ManagerCapability, CapabilityType

logger = logging.getLogger(__name__)


def create_default_metadata(manager: Any) -> ManagerMetadata:
    """Create default metadata for UnifiedParameterManager."""
    return ManagerMetadata(
        name="UnifiedParameterManager",
        version="2.0.0",
        description=(
            "Unified parameter management system with PyTorch integration, "
            "hierarchical configuration loading, scientific validation, "
            "and CF-1.8 unit checking"
        ),
        capabilities=[
            ManagerCapability(type=CapabilityType.PARAMETER_MANAGEMENT),
            ManagerCapability(type=CapabilityType.VALIDATION),
            ManagerCapability(type=CapabilityType.DEVICE_MANAGEMENT),
            ManagerCapability(type=CapabilityType.LIFECYCLE_MANAGEMENT),
            ManagerCapability(type=CapabilityType.PYTORCH_INTEGRATION),
        ],
        dependencies=[
            # VariableNameMapper dependency is injected directly when needed.
        ],
        tags={"singleton", "pytorch", "parameters", "validation"},
        custom_attributes={
            "managed_resources": ["parameters", "nn_parameters", "parameter_database"],
            "configuration_schema": {
                "database_path": {"type": "string", "description": "Path to parameter database"},
                "validation_level": {"type": "string", "enum": ["none", "basic", "standard", "strict"]},
                "unit_validation_mode": {
                    "type": "string",
                    "enum": ["off", "check-only", "strict"],
                    "description": "Unit validation mode for parameter specs.",
                },
                "device": {"type": "string", "description": "PyTorch device (cpu/cuda)"},
                "enforce_bounds_error": {
                    "type": "boolean",
                    "description": "If true, raise errors on bounds violations instead of clamping.",
                },
            },
        },
    )


def do_initialize(manager: Any) -> None:
    """Initialize the parameter manager (BaseManager lifecycle hook).

    Prepares the database location and loads nothing: a manager built with
    preload_all holds the whole database when its constructor returns, and
    any other manager loads on demand. Specifications, stored values and live
    bindings held at this point stay as they are; replacing them belongs to a
    database switch (do_configure with a new database_path).
    """
    manager.database_path.mkdir(parents=True, exist_ok=True)

    if not hasattr(manager.database, "base_path"):
        manager.database = ParameterDatabase(manager.database_path, validator=manager.validator)

    logger.info("UnifiedParameterManager _do_initialize completed")


def do_configure(manager: Any, config: Dict[str, Any]) -> None:
    """Configure the parameter manager (BaseManager lifecycle hook)."""
    settings = dict(config) if config else {}
    if "validation_level" in settings:
        validation_level_str = settings["validation_level"]
        if isinstance(validation_level_str, str):
            manager.validation_level = ValidationLevel(validation_level_str)
        else:
            manager.validation_level = validation_level_str
    if "unit_validation_mode" in settings:
        manager.set_unit_validation_mode(settings["unit_validation_mode"], recheck=True)

    if "bounds_policy" in settings:
        manager.set_bounds_policy(settings["bounds_policy"])
        manager._bounds_policy_explicit = True

    if "enforce_bounds_error" in settings:
        manager.enforce_bounds_error = bool(settings["enforce_bounds_error"])

    if "device" in settings:
        device_str = settings["device"]
        if isinstance(device_str, str):
            manager.device = torch.device(device_str)
        else:
            manager.device = device_str
        if manager.nn_parameters:
            manager.to(manager.device)

    if "database_path" in settings:
        new_path = Path(settings["database_path"])
        if new_path != manager.database_path:
            held_path, held_database = manager.database_path, manager.database
            try:
                manager.database_path = new_path
                manager.database = ParameterDatabase(manager.database_path, validator=manager.validator)
                manager._load_initial_parameters()
            except BaseException:
                # A rejected database is never recorded. The failures that
                # load_initial_parameters reports before adopting (database
                # read, unit checks, tensor forms) leave the held
                # specifications as they are, so the manager stays on the
                # database that defines them.
                manager.database_path, manager.database = held_path, held_database
                raise

    logger.info("UnifiedParameterManager _do_configure completed")


def do_validate(manager: Any) -> List[str]:
    """Validate parameter manager state (BaseManager lifecycle hook)."""
    errors: List[str] = []

    if not manager.database_path.exists():
        errors.append("Parameter database path does not exist")

    for key, spec in manager.parameters.items():
        spec_errors = spec.validate_value(spec.value)
        if spec_errors:
            errors.extend([f"{key}: {error}" for error in spec_errors])

    if len(errors) > 5:
        errors = errors[:5] + [f"... and {len(errors) - 5} more validation errors"]

    for key, param in manager.nn_parameters.items():
        if param.device != manager.device:
            errors.append(f"Parameter {key} on wrong device: {param.device} != {manager.device}")

    logger.debug("UnifiedParameterManager _do_validate completed")
    return errors


def do_cleanup(manager: Any) -> None:
    """Cleanup parameter manager resources (BaseManager lifecycle hook)."""
    manager.change_callbacks.clear()
    manager.change_history.clear()
    manager._missing_requests.clear()
    manager.nn_parameters.clear()

    logger.info("UnifiedParameterManager _do_cleanup completed")
