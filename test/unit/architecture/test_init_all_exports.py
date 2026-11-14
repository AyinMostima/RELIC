"""Governance tests for package-level __all__ exports."""

from __future__ import annotations

import importlib


def test_core_data_parameters_all_exports_are_accessible() -> None:
    """Each symbol listed in __all__ must be attribute-accessible."""
    modules = ["relic.core", "relic.data", "relic.parameters"]

    for module_name in modules:
        module = importlib.import_module(module_name)
        exports = getattr(module, "__all__", [])
        assert isinstance(exports, list)
        for symbol in exports:
            assert hasattr(module, symbol), f"{module_name} missing export '{symbol}'"
