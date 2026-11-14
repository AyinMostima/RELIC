"""Metrics and grouping helpers for parameter manager."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Dict, List

from ..spec import ParameterSpec
from ..types import ParameterCategory, ValidationContext

logger = logging.getLogger(__name__)


def get_parameters_by_category(manager, category: ParameterCategory) -> List[ParameterSpec]:
    """Return parameters filtered by category."""
    return [spec for spec in manager.parameters.values() if spec.category == category]


def get_statistics(manager) -> Dict[str, Any]:
    """Return parameter statistics by category and training flags."""
    by_category = defaultdict(int)
    for spec in manager.parameters.values():
        by_category[spec.category] += 1

    with_gradients = sum(1 for spec in manager.parameters.values() if spec.requires_grad)

    return {
        "total_parameters": len(manager.parameters),
        "by_category": dict(by_category),
        "with_gradients": with_gradients,
        "nn_parameters": len(manager.nn_parameters),
        "device": str(manager.device),
        "validation_level": manager.validation_level.value,
    }


def get_status(manager) -> Dict[str, Any]:
    """Return comprehensive parameter manager status."""
    learnable_params = sum(1 for spec in manager.parameters.values() if spec.requires_grad)

    categories = defaultdict(int)
    for spec in manager.parameters.values():
        categories[spec.category.value] += 1

    return {
        "total_parameters": len(manager.parameters),
        "learnable_parameters": learnable_params,
        "nn_parameters": len(manager.nn_parameters),
        "parameter_categories": dict(categories),
        "validation_level": manager.validation_level.value,
        "unit_validation_mode": manager.unit_validation_mode.value,
        "unit_validation_issues": len(manager._unit_validation_issues),
        "device": str(manager.device),
        "change_callbacks": len(manager.change_callbacks),
        "change_history_length": len(manager.change_history),
        "database_path": str(manager.database_path),
    }


def debug_report_units(manager) -> Dict[str, Dict[str, Any]]:
    """
    Generate a basic unit validation report for all registered parameters.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from parameter key to its units, dimension, status,
        and any unit-related validation errors.
    """
    report: Dict[str, Dict[str, Any]] = {}

    if manager.validator is None:
        return report

    if not manager.parameters:
        manager._load_initial_parameters()

    context = ValidationContext()
    invalid_count = 0

    for key, spec in manager.parameters.items():
        units = getattr(spec, "units", None)
        dimension = getattr(spec, "dimension", None)

        result = manager.validator._validate_units(key, spec.value, spec, context)

        status = "ok" if result.is_valid else "invalid"
        if not result.is_valid:
            invalid_count += 1

        report[key] = {
            "units": units,
            "dimension": dimension,
            "status": status,
            "errors": result.errors,
        }

    if invalid_count > 0:
        logger.warning("Unit validation found %d parameters with issues", invalid_count)
    else:
        logger.info("Unit validation: all parameter units passed basic checks")

    return report
