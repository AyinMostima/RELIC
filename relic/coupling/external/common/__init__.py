"""
Common utilities for external model coupling.

This package exposes a unified public API via lazy attribute resolution to
avoid import cycles between data, coupling, and core layers.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    # base_adapter
    "ExternalAdapter": (".base_adapter", "ExternalAdapter"),
    "TimeWindow": (".base_adapter", "TimeWindow"),
    "SpatialWindow": (".base_adapter", "SpatialWindow"),
    # validation_types
    "ValidationReport": (".validation_types", "ValidationReport"),
    # datasource
    "DataSource": (".datasource", "DataSource"),
    # schema
    "InternalSchema": (".schema", "InternalSchema"),
    "VariableDefinition": (".schema", "VariableDefinition"),
    "VariableCategory": (".schema", "VariableCategory"),
    # unit conversion
    "UnitConverter": ("relic.data.standards.unit_converter", "UnitConverter"),
    "ConversionContext": ("relic.data.standards.unit_converter", "ConversionContext"),
    # time
    "TimeHandler": (".time", "TimeHandler"),
    "TimeCoordinate": (
        "relic.coupling.scales.temporal.core",
        "TimeCoordinate",
    ),
    # regrid
    "RegridOperator": (".regrid", "RegridOperator"),
    "RegridMethod": (".regrid", "RegridMethod"),
    "RegridConfig": (".regrid", "RegridConfig"),
    "RegridResult": (".regrid", "RegridResult"),
    # io
    "IOHandler": (".io", "IOHandler"),
    "IOConfig": (".io", "IOConfig"),
    "IOResult": (".io", "IOResult"),
    "DataFormat": (".io", "DataFormat"),
    "CompressionType": (".io", "CompressionType"),
    # validators
    "Validator": (".validators", "Validator"),
    "ValidationConfig": (".validators", "ValidationConfig"),
    "ValidationIssue": (".validators", "ValidationIssue"),
    "ValidationCategory": (".validators", "ValidationCategory"),
    "ValidationSeverity": (".validators", "ValidationSeverity"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attr_name = target
    if module_name.startswith("."):
        module = importlib.import_module(module_name, __name__)
    else:
        module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    # Base classes and enums
    "ExternalAdapter",
    "DataSource",
    "TimeWindow",
    "SpatialWindow",
    "ValidationReport",
    # Schema and standards
    "InternalSchema",
    "VariableDefinition",
    "VariableCategory",
    # Units and conversions
    "UnitConverter",
    "ConversionContext",
    # Time handling
    "TimeHandler",
    "TimeCoordinate",
    # Spatial regridding
    "RegridOperator",
    "RegridMethod",
    "RegridConfig",
    "RegridResult",
    # I/O operations
    "IOHandler",
    "IOConfig",
    "IOResult",
    "DataFormat",
    "CompressionType",
    # Validation
    "Validator",
    "ValidationConfig",
    "ValidationIssue",
    "ValidationCategory",
    "ValidationSeverity",
]
