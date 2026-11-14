"""Governance tests for package-level __all__ export integrity."""

from __future__ import annotations

import importlib
import pkgutil
from typing import List

import pytest
import relic
from relic.data._placeholder import ExperimentalFeatureError


def _collect_package_modules_with_all() -> List[str]:
    """Collect importable package modules under relic that define __all__."""
    modules: List[str] = ["relic"]
    seen: set[str] = set(modules)
    queue: list[tuple[object, str]] = [(relic.__path__, "relic.")]

    while queue:
        path, prefix = queue.pop()
        for modinfo in pkgutil.iter_modules(path, prefix):
            if not modinfo.ispkg or modinfo.name in seen:
                continue
            seen.add(modinfo.name)
            modules.append(modinfo.name)
            try:
                module = importlib.import_module(modinfo.name)
            except ExperimentalFeatureError:
                # Placeholder packages refuse import without their explicit
                # RELIC_EXPERIMENTAL_* opt-in; any other import error surfaces.
                continue
            sub_path = getattr(module, "__path__", None)
            if sub_path is not None:
                queue.append((sub_path, f"{modinfo.name}."))

    selected: List[str] = []
    for module_name in sorted(set(modules)):
        try:
            module = importlib.import_module(module_name)
        except ExperimentalFeatureError:
            continue
        if hasattr(module, "__all__"):
            selected.append(module_name)
    return selected


PACKAGE_MODULES_WITH_ALL = _collect_package_modules_with_all()


@pytest.mark.parametrize("module_name", PACKAGE_MODULES_WITH_ALL)
def test_all_exports_are_defined(module_name: str) -> None:
    """Every symbol declared in __all__ must be attribute-accessible."""
    module = importlib.import_module(module_name)
    exported = getattr(module, "__all__", [])

    assert isinstance(exported, (list, tuple)), (
        f"{module_name}.__all__ must be list/tuple, got {type(exported)}"
    )

    for symbol in exported:
        assert isinstance(symbol, str), f"{module_name}.__all__ contains non-string: {symbol!r}"
        assert hasattr(module, symbol), (
            f"{module_name}.__all__ exports '{symbol}' but attribute is not accessible"
        )
