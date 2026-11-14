"""Invariant tests for feature flag manager behavior."""

from __future__ import annotations

import pytest

from relic.core.feature_flags import FeatureFlagManager, require_feature


def test_require_feature_decorator_blocks_and_allows_execution() -> None:
    """Decorator must block disabled feature and allow enabled feature."""
    manager = FeatureFlagManager()
    manager.reset_to_defaults()

    @require_feature("strict_conservation_enabled")
    def _guarded() -> str:
        return "ok"

    with pytest.raises(RuntimeError, match="strict_conservation_enabled"):
        _guarded()

    manager.enable("strict_conservation_enabled")
    assert _guarded() == "ok"


def test_register_callback_receives_state_changes() -> None:
    """Callbacks must be invoked when flag state changes."""
    manager = FeatureFlagManager()
    manager.reset_to_defaults()
    events = []

    def _cb(feature_name: str, enabled: bool) -> None:
        events.append((feature_name, enabled))

    manager.register_callback("o3_stress_enabled", _cb)
    manager.enable("o3_stress_enabled")
    manager.disable("o3_stress_enabled")

    assert ("o3_stress_enabled", True) in events
    assert ("o3_stress_enabled", False) in events


def test_validate_feature_consistency_reports_conflicts() -> None:
    """Consistency validator should flag known conflict patterns."""
    manager = FeatureFlagManager()
    manager.reset_to_defaults()

    manager.enable("full_richards_solver")
    manager.disable("advanced_numerical_solvers")

    warnings = manager.validate_feature_consistency()
    assert any("full_richards_solver" in item for item in warnings)


@pytest.mark.parametrize(
    "name", ["conservation_checks", "enzyme_kinetics_enabled", "model_config", "dict"]
)
def test_undeclared_flag_name_raises_instead_of_reading_false(name: str) -> None:
    """A name that is not a declared FeatureFlagConfig field is rejected, never read as off."""
    manager = FeatureFlagManager()
    manager.reset_to_defaults()

    for operation in (manager.is_enabled, manager.enable, manager.disable):
        with pytest.raises(ValueError, match="Unknown feature flag"):
            operation(name)
    assert name not in manager.get_all_flags()


def test_enabled_and_disabled_features_partition_the_declared_flags() -> None:
    """The enabled and disabled lists and the status report read the declared flags."""
    manager = FeatureFlagManager()
    manager.reset_to_defaults()
    manager.enable("canopy_layers_enabled")
    manager.disable("full_richards_solver")
    try:
        enabled = manager.get_enabled_features()
        disabled = manager.get_disabled_features()
        report = manager.get_feature_status_report()
    finally:
        manager.reset_to_defaults()

    assert sorted(enabled + disabled) == sorted(manager.get_all_flags())
    assert "canopy_layers_enabled" in enabled
    assert "full_richards_solver" in disabled
    assert "[+] canopy_layers_enabled" in report
    assert "[-] full_richards_solver" in report


def test_every_flag_name_read_in_relic_is_declared() -> None:
    """Each literal flag name the relic package reads names a declared flag."""
    import re
    from pathlib import Path

    import relic
    from relic.core.config.models import FeatureFlagConfig

    pattern = re.compile(
        r"(?:feature_manager\.is_enabled|feature_enabled|require_feature)\(\s*['\"]([^'\"]+)['\"]"
    )
    package_root = Path(relic.__file__).parent
    read_names = {
        (path.relative_to(package_root).as_posix(), match.group(1))
        for path in package_root.rglob("*.py")
        for match in pattern.finditer(path.read_text(encoding="utf-8-sig"))
    }

    assert read_names
    undeclared = sorted(
        (source, name) for source, name in read_names
        if name not in FeatureFlagConfig.model_fields
    )
    assert undeclared == []

def test_callback_failures_are_raised_after_every_callback_ran(monkeypatch) -> None:
    """A failing listener does not hide behind a warning or stop the others."""
    manager = FeatureFlagManager()
    manager.reset_to_defaults()
    monkeypatch.setattr(manager, "_callbacks", {})
    events = []

    def _failing(feature_name: str, enabled: bool) -> None:
        raise ValueError("listener rejected the change")

    def _recording(feature_name: str, enabled: bool) -> None:
        events.append((feature_name, enabled))

    manager.register_callback("o3_stress_enabled", _failing)
    manager.register_callback("o3_stress_enabled", _recording)
    try:
        with pytest.raises(ExceptionGroup) as error:
            manager.enable("o3_stress_enabled")
        assert [str(item) for item in error.value.exceptions] == ["listener rejected the change"]
        assert events == [("o3_stress_enabled", True)]
        assert manager.is_enabled("o3_stress_enabled") is True
    finally:
        manager.reset_to_defaults()
