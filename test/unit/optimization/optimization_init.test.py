"""L1 tests for the relic.optimization package exports (relic/optimization/__init__.py).

The four subpackages are RELIC's own modules with hard dependencies only
(torch, numpy, scipy). A failure while importing them is an error at import
time; it is never recorded as an unavailable optimizer with None exports.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
from types import ModuleType

import pytest

import relic.optimization as optimization


def test_broken_subpackage_import_fails_the_package_import(monkeypatch) -> None:
    broken = ModuleType("relic.optimization.sensitivity")
    broken.__spec__ = importlib.machinery.ModuleSpec("relic.optimization.sensitivity", None)
    monkeypatch.setitem(sys.modules, "relic.optimization.sensitivity", broken)
    try:
        with pytest.raises(ImportError, match="MorrisAnalyzer|SensitivityAnalyzer"):
            importlib.reload(optimization)
    finally:
        monkeypatch.undo()
        importlib.reload(optimization)


def test_every_declared_export_is_bound_to_a_real_object() -> None:
    for name in optimization.__all__:
        assert getattr(optimization, name) is not None, name


def test_available_optimizers_report_components_without_runtime_claims() -> None:
    families = optimization.get_available_optimizers()

    assert set(families) == {"calibration", "assimilation", "sensitivity", "uncertainty"}
    for record in families.values():
        assert record["component_available"] is True
        assert record["error"] is None
        assert record["runtime_ready"] is False
