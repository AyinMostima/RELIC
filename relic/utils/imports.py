"""Shared import helper utilities."""

from __future__ import annotations

import importlib.util


def module_available(module_name: str) -> bool:
    """Return True when the target module can be resolved."""
    return importlib.util.find_spec(module_name) is not None


__all__ = ["module_available"]
