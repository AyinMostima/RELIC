"""Tests for placeholder module experimental import gates."""

import importlib
import sys

import pytest

from relic.data._placeholder import ExperimentalFeatureError


def _reload_module(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_quality_placeholder_requires_opt_in(monkeypatch):
    monkeypatch.delenv("RELIC_EXPERIMENTAL_DATA_QUALITY", raising=False)
    sys.modules.pop("relic.data.quality", None)
    with pytest.raises(ExperimentalFeatureError):
        importlib.import_module("relic.data.quality")


def test_quality_placeholder_imports_with_opt_in(monkeypatch):
    monkeypatch.setenv("RELIC_EXPERIMENTAL_DATA_QUALITY", "1")
    module = _reload_module("relic.data.quality")
    assert hasattr(module, "get_available_components")


def test_transformers_placeholder_requires_opt_in(monkeypatch):
    monkeypatch.delenv("RELIC_EXPERIMENTAL_DATA_TRANSFORMERS", raising=False)
    sys.modules.pop("relic.data.transformers", None)
    with pytest.raises(ExperimentalFeatureError):
        importlib.import_module("relic.data.transformers")


def test_transformers_placeholder_imports_with_opt_in(monkeypatch):
    monkeypatch.setenv("RELIC_EXPERIMENTAL_DATA_TRANSFORMERS", "1")
    module = _reload_module("relic.data.transformers")
    assert hasattr(module, "get_available_transformers")


def test_optional_data_modules_expose_import_errors_other_than_the_placeholder_gate(monkeypatch):
    """The placeholder gate and a missing optional dependency read as absent (None); any other
    import error of RELIC's own module propagates."""
    import relic.data as data_package

    def gated(name, package=None):
        raise ExperimentalFeatureError("experimental placeholder code")

    monkeypatch.setattr(data_package.importlib, "import_module", gated)
    assert data_package._import_optional_data_module("quality") is None

    def missing(name, package=None):
        raise ModuleNotFoundError("No module named 'optional_backend'")

    monkeypatch.setattr(data_package.importlib, "import_module", missing)
    assert data_package._import_optional_data_module("quality") is None
    assert "quality" in data_package.OPTIONAL_IMPORT_ERRORS

    def broken(name, package=None):
        raise KeyError("module bug")

    monkeypatch.setattr(data_package.importlib, "import_module", broken)
    with pytest.raises(KeyError, match="module bug"):
        data_package._import_optional_data_module("quality")
