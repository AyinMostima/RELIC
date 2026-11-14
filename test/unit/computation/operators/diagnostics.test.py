"""Tests for the diagnostics operators in relic.computation.operators.diagnostics.

ConservationChecker reads its relative tolerance
(error = |dM - dt integral(S dV)| / |M_old|) from the ParameterManager key
computation.operators.diagnostics.tolerance without a Python fallback.
"""

from __future__ import annotations

import pytest

import relic.computation.operators.base as operator_base
from relic.computation.operators.diagnostics import ConservationChecker
from relic.parameters import get_global_manager

TOLERANCE_KEY = "computation.operators.diagnostics.tolerance"


class _ManagerWithoutTolerance:
    """Delegating manager view in which the tolerance key is absent.

    The key is hidden from both the value lookup (get) and the YAML spec
    lookup (get_parameter) that OperatorBase.get_parameter reads first.
    """

    hidden_key = TOLERANCE_KEY

    def __init__(self, manager):
        self._manager = manager

    def get(self, key, default=None, **kwargs):
        if key == self.hidden_key:
            return default
        return self._manager.get(key, default=default, **kwargs)

    def get_parameter(self, key, namespace=None):
        if key == self.hidden_key:
            return None
        return self._manager.get_parameter(key, namespace)


class TestConservationCheckerTolerance:
    def test_tolerance_is_read_from_the_parameter_database(self):
        manager = get_global_manager()
        original = manager.get(TOLERANCE_KEY, required=True)
        assert ConservationChecker().tolerance == pytest.approx(original)
        try:
            manager.set_value(TOLERANCE_KEY, 2.0e-6)
            assert ConservationChecker().tolerance == pytest.approx(2.0e-6)
        finally:
            manager.set_value(TOLERANCE_KEY, original)
        assert manager.get(TOLERANCE_KEY, required=True) == original

    def test_missing_tolerance_key_is_an_error_not_a_python_default(self, monkeypatch):
        view = _ManagerWithoutTolerance(get_global_manager())
        monkeypatch.setattr(operator_base, "get_global_manager", lambda: view)
        with pytest.raises(KeyError, match=f"Required parameter '{TOLERANCE_KEY}' not found"):
            ConservationChecker()

    def test_explicit_config_tolerance_still_overrides(self):
        checker = ConservationChecker(config={"tolerance": 5.0e-5})
        assert checker.tolerance == pytest.approx(5.0e-5)


STRICT_MODE_KEY = "computation.operators.diagnostics.strict_mode"


class _ManagerWithoutStrictMode(_ManagerWithoutTolerance):
    """Delegating manager view in which the strict_mode key is absent."""

    hidden_key = STRICT_MODE_KEY


class TestConservationCheckerStrictMode:
    def test_strict_mode_is_a_bool_parameter_with_full_metadata(self):
        from pathlib import Path

        import yaml

        path = (
            Path(__file__).resolve().parents[4]
            / "relic" / "parameters" / "database" / "relic" / "computation" / "operators.yaml"
        )
        entry = yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][STRICT_MODE_KEY]
        for field_name in ("value", "units", "dtype", "bounds", "category", "description",
                           "references", "tags", "requires_grad", "provenance"):
            assert field_name in entry, field_name
        assert entry["dtype"] == "bool"
        manager = get_global_manager()
        original = manager.get(STRICT_MODE_KEY, required=True)
        assert original is entry["value"]
        assert ConservationChecker().strict_mode is original
        try:
            manager.set_value(STRICT_MODE_KEY, True)
            assert ConservationChecker().strict_mode is True
        finally:
            manager.set_value(STRICT_MODE_KEY, original)
        assert manager.get(STRICT_MODE_KEY, required=True) is original

    def test_missing_strict_mode_key_is_an_error_not_a_python_default(self, monkeypatch):
        view = _ManagerWithoutStrictMode(get_global_manager())
        monkeypatch.setattr(operator_base, "get_global_manager", lambda: view)
        with pytest.raises(KeyError, match=f"Required parameter '{STRICT_MODE_KEY}' not found"):
            ConservationChecker()
