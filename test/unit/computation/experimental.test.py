"""L1 tests for the experimental feature registry (relic/computation/experimental.py).

Only a missing optional dependency (ImportError) is recorded as 'unavailable';
any other import failure propagates. Unknown feature names, failed cleanups
and failed feature factories raise instead of being logged and ignored.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

import relic.computation.experimental as experimental


@pytest.fixture(autouse=True)
def _fresh_dependency_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experimental, "_DEPENDENCY_CACHE", {})


def _fake_spec(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    real_find_spec = importlib.util.find_spec

    def find_spec(module_name, *args, **kwargs):
        if module_name == name:
            return SimpleNamespace(name=name)
        return real_find_spec(module_name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _failing_import(monkeypatch: pytest.MonkeyPatch, name: str, error: Exception) -> None:
    real_import = importlib.import_module

    def import_module(module_name, *args, **kwargs):
        if module_name == name:
            raise error
        return real_import(module_name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)


def test_missing_optional_dependency_is_recorded_with_its_cause(monkeypatch) -> None:
    _fake_spec(monkeypatch, "probe_optional_pkg")
    _failing_import(monkeypatch, "probe_optional_pkg", ImportError("DLL load failed"))

    status = experimental.check_dependency_available("probe_optional_pkg")

    assert status["available"] is False
    assert "DLL load failed" in status["import_error"]


def test_non_import_failure_of_a_dependency_propagates(monkeypatch) -> None:
    _fake_spec(monkeypatch, "probe_broken_pkg")
    _failing_import(monkeypatch, "probe_broken_pkg", RuntimeError("module init crashed"))

    with pytest.raises(RuntimeError, match="module init crashed"):
        experimental.check_dependency_available("probe_broken_pkg")


def test_dependencies_are_declared_by_import_name() -> None:
    """A distribution name such as 'scikit-learn' can never be imported."""
    for feature in experimental._EXPERIMENTAL_FEATURES.values():
        for dependency in feature["dependencies"]:
            assert dependency.isidentifier(), dependency


def test_unknown_feature_name_is_an_error() -> None:
    with pytest.raises(ValueError, match="triton_kernals"):
        experimental.is_experimental_enabled("triton_kernals")


def test_failed_cleanup_is_raised_and_the_feature_stays_enabled() -> None:
    manager = experimental.ExperimentalFeatureManager()
    manager.enable_feature("streaming", force=True)

    def failing_cleanup():
        raise OSError("scratch store locked")

    manager.register_cleanup_callback("streaming", failing_cleanup)

    with pytest.raises(OSError, match="scratch store locked"):
        manager.disable_feature("streaming")
    assert manager.is_enabled("streaming")


def test_cleanup_all_disables_what_it_can_and_raises_the_aggregate() -> None:
    manager = experimental.ExperimentalFeatureManager()
    for name in ("streaming", "cupy_acceleration", "ml_acceleration"):
        manager.enable_feature(name, force=True)

    def failing_cleanup():
        raise OSError("device busy")

    manager.register_cleanup_callback("cupy_acceleration", failing_cleanup)

    with pytest.raises(RuntimeError, match="cupy_acceleration") as excinfo:
        manager.cleanup_all()
    assert isinstance(excinfo.value.__cause__, OSError)
    assert manager.get_enabled_features() == ["cupy_acceleration"]


def test_failed_feature_factory_is_raised_not_returned_as_none() -> None:
    manager = experimental.ExperimentalFeatureManager()
    manager.enable_feature("streaming", force=True)

    def failing_factory():
        raise ValueError("bad chunk layout")

    with pytest.raises(ValueError, match="bad chunk layout"):
        manager.get_feature("streaming", failing_factory)


def test_safe_import_substitutes_only_for_a_missing_module(monkeypatch) -> None:
    monkeypatch.setattr(experimental, "is_experimental_enabled", lambda _name: True)
    monkeypatch.setattr(
        experimental,
        "validate_experimental_feature",
        lambda _name: {"valid": True, "errors": [], "warnings": []},
    )
    _failing_import(monkeypatch, "probe_missing_mod", ImportError("No module named probe"))

    sentinel = object()
    assert (
        experimental.safe_import_experimental(
            "streaming", "probe_missing_mod", alternative=sentinel, allow_alternative=True
        )
        is sentinel
    )

    _failing_import(monkeypatch, "probe_crashing_mod", RuntimeError("import side effect failed"))
    with pytest.raises(RuntimeError, match="import side effect failed"):
        experimental.safe_import_experimental(
            "streaming", "probe_crashing_mod", alternative=sentinel, allow_alternative=True
        )


def test_package_exports_the_single_experimental_flag_owner() -> None:
    """relic.computation re-exports the experimental owner; it keeps no second copy."""
    import relic.computation as computation

    assert computation.is_experimental_enabled is experimental.is_experimental_enabled
