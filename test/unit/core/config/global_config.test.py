"""Tests for global configuration cache loading behavior."""

from __future__ import annotations

import pytest

import relic.core.config as config_module

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("validate", [False, True])
def test_get_global_config_propagates_validation_policy(
    monkeypatch,
    tmp_path,
    validate,
) -> None:
    calls = {}
    loaded_config = object()

    class RecordingLoader:
        def __init__(self, *, validate_on_load):
            calls["validate_on_load"] = validate_on_load

        def load_config(
            self,
            config_path,
            overrides,
            env_overrides,
        ):
            calls["load"] = (config_path, overrides, env_overrides)
            return loaded_config

    monkeypatch.setattr(config_module, "_GLOBAL_CONFIG_CACHE", None)
    monkeypatch.setattr(config_module, "ConfigLoader", RecordingLoader)

    config_file = tmp_path / "runtime.yaml"
    config_file.write_text("simulation: {}", encoding="utf-8")
    result = config_module.get_global_config(
        config_path=str(config_file),
        validate=validate,
        env_overrides=False,
    )

    assert result is loaded_config
    assert calls["validate_on_load"] is validate
    assert calls["load"] == (config_file, None, False)


def _write_config(path, time_step):
    path.write_text(
        f"simulation:\n  time_step: {time_step}\n  end_time: 86400.0\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def isolated_resolution(monkeypatch):
    """No named file, no default candidate, no cached global configuration."""
    import relic.core.config.loader as loader_module

    monkeypatch.delenv("RELIC_CONFIG_PATH", raising=False)
    monkeypatch.delenv("RELIC_SIMULATION_TIME_STEP", raising=False)
    monkeypatch.setattr(loader_module, "DEFAULT_CONFIG_PATHS", [])
    monkeypatch.setattr(config_module, "_GLOBAL_CONFIG_CACHE", None)
    return loader_module


def test_one_configuration_source_owner():
    import relic.core.config.loader as loader_module

    assert config_module.resolve_config_source is loader_module.resolve_config_source
    assert not hasattr(config_module, "_find_default_config_path")
    assert not hasattr(loader_module, "CONFIG_SEARCH_PATHS")
    assert not hasattr(loader_module.ConfigLoader, "_discover_config_file")


@pytest.mark.parametrize("named_by", ["argument", "environment"])
def test_a_named_file_that_does_not_exist_raises(isolated_resolution, monkeypatch, tmp_path, named_by):
    missing = tmp_path / "missing.yaml"
    _write_config(tmp_path / "candidate.yaml", 60.0)
    monkeypatch.setattr(isolated_resolution, "DEFAULT_CONFIG_PATHS", [str(tmp_path / "candidate.yaml")])
    if named_by == "argument":
        with pytest.raises(FileNotFoundError, match="missing.yaml"):
            config_module.get_global_config(config_path=str(missing))
    else:
        monkeypatch.setenv("RELIC_CONFIG_PATH", str(missing))
        with pytest.raises(FileNotFoundError, match="missing.yaml"):
            config_module.get_global_config()


def test_without_a_named_file_the_first_existing_candidate_is_loaded(isolated_resolution, monkeypatch, tmp_path):
    first = _write_config(tmp_path / "first.yaml", 75.0)
    second = _write_config(tmp_path / "second.yaml", 90.0)
    monkeypatch.setattr(
        isolated_resolution,
        "DEFAULT_CONFIG_PATHS",
        [str(tmp_path / "absent.yaml"), str(first), str(second)],
    )
    assert isolated_resolution.resolve_config_source() == first
    assert config_module.get_global_config().simulation.time_step == 75.0


def test_without_any_candidate_the_global_default_configuration_is_used_and_logged(
    isolated_resolution, monkeypatch, tmp_path, caplog
):
    import logging

    monkeypatch.setattr(isolated_resolution, "DEFAULT_CONFIG_PATHS", [str(tmp_path / "absent.yaml")])
    caplog.set_level(logging.INFO, logger=isolated_resolution.logger.name)

    assert isolated_resolution.resolve_config_source() is None
    config = config_module.get_global_config()

    # Loaded values equal the RELICConfig defaults (compared in serialized form: a
    # loaded enum field holds its value, the unvalidated default the member).
    default = config_module.RELICConfig()
    assert config.model_dump(mode="json", exclude={"created_at"}) == default.model_dump(
        mode="json", exclude={"created_at"}
    )
    assert any("global default configuration" in record.getMessage() for record in caplog.records)


def test_the_cache_never_masks_a_changed_environment(isolated_resolution, monkeypatch, tmp_path):
    first = _write_config(tmp_path / "first.yaml", 60.0)
    second = _write_config(tmp_path / "second.yaml", 120.0)

    monkeypatch.setenv("RELIC_CONFIG_PATH", str(first))
    cached = config_module.get_global_config()
    assert cached.simulation.time_step == 60.0
    assert config_module.get_global_config() is cached

    monkeypatch.setenv("RELIC_CONFIG_PATH", str(second))
    assert config_module.get_global_config().simulation.time_step == 120.0

    monkeypatch.setenv("RELIC_SIMULATION_TIME_STEP", "30.0")
    assert config_module.get_global_config().simulation.time_step == 30.0

    monkeypatch.delenv("RELIC_SIMULATION_TIME_STEP")
    _write_config(second, 150.0)
    import os

    stat = second.stat()
    os.utime(second, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert config_module.get_global_config().simulation.time_step == 150.0
