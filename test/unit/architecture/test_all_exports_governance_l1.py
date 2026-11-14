"""L1 governance tests for package __all__ export integrity."""

from __future__ import annotations

import importlib
from typing import List, Tuple


MODULES_UNDER_TEST = (
    "relic.data",
    "relic.optimization",
    "relic.parameters",
    "relic.scientific",
)


def _collect_export_failures(module_name: str) -> List[Tuple[str, str]]:
    """Collect failing exports as (symbol, error) pairs."""
    module = importlib.import_module(module_name)
    failures: List[Tuple[str, str]] = []
    for symbol in getattr(module, "__all__", []):
        try:
            getattr(module, symbol)
        except Exception as exc:  # pragma: no cover - failure path only
            failures.append((symbol, f"{type(exc).__name__}: {exc}"))
    return failures


def test_all_exports_are_accessible() -> None:
    """Every symbol declared in __all__ must be accessible at runtime."""
    errors: List[str] = []
    for module_name in MODULES_UNDER_TEST:
        failures = _collect_export_failures(module_name)
        for symbol, message in failures:
            errors.append(f"{module_name}.{symbol} -> {message}")
    assert not errors, "Invalid __all__ exports detected:\n" + "\n".join(errors)
