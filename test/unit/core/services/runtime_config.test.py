"""Tests for shared CLI and API runtime configuration loading."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import relic.core.config.loader as config_loader
from relic.core.services import runtime_config

pytestmark = pytest.mark.unit


def _forbid_cached_global_config(monkeypatch) -> None:
    """The runtime loader reads its source directly, never the cached global config."""

    def cached_global_config(*args, **kwargs):
        raise AssertionError("load_runtime_config must not consult get_global_config")

    monkeypatch.setattr(runtime_config.core_config, "get_global_config", cached_global_config)


def test_the_config_path_variable_has_one_owner():
    """load_runtime_config names the variable through CONFIG_PATH_ENV_VAR."""
    import inspect

    import relic.core.config.loader as loader
    from relic.core.services import runtime_config

    assert "RELIC_CONFIG_PATH" not in inspect.getsource(runtime_config)
    parameter = inspect.signature(runtime_config.load_runtime_config).parameters["env_var"]
    assert parameter.default == loader.CONFIG_PATH_ENV_VAR


def test_explicit_existing_path_precedes_environment(
    monkeypatch,
    tmp_path,
) -> None:
    explicit_path = tmp_path / "explicit.yaml"
    environment_path = tmp_path / "environment.yaml"
    explicit_path.write_text("simulation: {}", encoding="utf-8")
    environment_path.write_text("simulation: {}", encoding="utf-8")
    monkeypatch.setenv("TEST_RELIC_CONFIG", str(environment_path))
    load_calls = []
    validation_result = object()

    def load_config(path, *, validate):
        load_calls.append((path, validate))
        return {"source": "explicit"}

    monkeypatch.setattr(runtime_config.core_config, "load_config", load_config)
    monkeypatch.setattr(
        runtime_config.core_config,
        "validate_configuration",
        lambda config: validation_result,
    )

    config, validation, source = runtime_config.load_runtime_config(
        str(explicit_path),
        env_var="TEST_RELIC_CONFIG",
    )

    assert config == {"source": "explicit"}
    assert validation is validation_result
    assert source == explicit_path
    assert load_calls == [(str(explicit_path), False)]


def test_environment_existing_path_is_used_without_explicit_path(
    monkeypatch,
    tmp_path,
) -> None:
    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text("simulation: {}", encoding="utf-8")
    monkeypatch.setenv("TEST_RELIC_CONFIG", str(environment_path))
    load_calls = []

    def load_config(path, *, validate):
        load_calls.append((path, validate))
        return {"source": "environment"}

    monkeypatch.setattr(runtime_config.core_config, "load_config", load_config)
    monkeypatch.setattr(
        runtime_config.core_config,
        "validate_configuration",
        lambda config: "valid",
    )

    config, validation, source = runtime_config.load_runtime_config(
        env_var="TEST_RELIC_CONFIG"
    )

    assert config == {"source": "environment"}
    assert validation == "valid"
    assert source == environment_path
    assert load_calls == [(str(environment_path), False)]


def test_no_candidate_uses_global_default_configuration_and_no_source(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.delenv("TEST_RELIC_CONFIG", raising=False)
    monkeypatch.setattr(config_loader, "DEFAULT_CONFIG_PATHS", [])
    _forbid_cached_global_config(monkeypatch)
    validated = []
    monkeypatch.setattr(
        runtime_config.core_config,
        "validate_configuration",
        lambda config: validated.append(config) or "valid",
    )
    caplog.set_level(logging.INFO, logger=runtime_config.logger.name)

    config, validation, source = runtime_config.load_runtime_config(
        env_var="TEST_RELIC_CONFIG"
    )

    # created_at is the construction timestamp; every other field is a default.
    expected = runtime_config.core_config.RELICConfig().model_dump(exclude={"created_at"})
    assert {key: value for key, value in config.items() if key != "created_at"} == expected
    assert validation == "valid"
    assert source is None
    assert len(validated) == 1
    assert isinstance(validated[0], runtime_config.core_config.RELICConfig)
    assert any("global default configuration" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("candidate_source", ["explicit", "environment"])
def test_missing_explicit_or_environment_file_raises(
    monkeypatch,
    tmp_path,
    candidate_source,
) -> None:
    """A named configuration file that does not exist is an error, never a fallback."""
    missing_path = tmp_path / "missing.yaml"
    monkeypatch.delenv("TEST_RELIC_CONFIG", raising=False)
    explicit_path = None
    if candidate_source == "explicit":
        explicit_path = str(missing_path)
    else:
        monkeypatch.setenv("TEST_RELIC_CONFIG", str(missing_path))
    _forbid_cached_global_config(monkeypatch)
    # An existing default candidate is never used in place of a named file.
    candidate = tmp_path / "relic_config.yaml"
    candidate.write_text("simulation: {}", encoding="utf-8")
    monkeypatch.setattr(config_loader, "DEFAULT_CONFIG_PATHS", [str(candidate)])

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        runtime_config.load_runtime_config(
            explicit_path,
            env_var="TEST_RELIC_CONFIG",
        )


def test_default_location_file_is_reported_as_the_source(
    monkeypatch,
    tmp_path,
) -> None:
    default_path = tmp_path / "relic_config.yaml"
    default_path.write_text("simulation: {}", encoding="utf-8")
    monkeypatch.delenv("TEST_RELIC_CONFIG", raising=False)
    monkeypatch.setattr(config_loader, "DEFAULT_CONFIG_PATHS", [str(tmp_path / "absent.yaml"), str(default_path)])
    _forbid_cached_global_config(monkeypatch)
    load_calls = []

    def load_config(path, *, validate):
        load_calls.append((path, validate))
        return {"source": "default-location"}

    monkeypatch.setattr(runtime_config.core_config, "load_config", load_config)
    monkeypatch.setattr(runtime_config.core_config, "validate_configuration", lambda config: "valid")

    config, _, source = runtime_config.load_runtime_config(env_var="TEST_RELIC_CONFIG")

    assert config == {"source": "default-location"}
    assert source == default_path
    assert load_calls == [(str(default_path), False)]


def test_model_config_is_normalized_after_single_wrapper_validation(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text("simulation: {}", encoding="utf-8")

    class ConfigModel:
        def model_dump(self):
            return {"simulation": {"device": "cpu"}}

    model = ConfigModel()
    validated = []
    monkeypatch.setattr(
        runtime_config.core_config,
        "load_config",
        lambda path, *, validate: model,
    )
    monkeypatch.setattr(
        runtime_config.core_config,
        "validate_configuration",
        lambda config: validated.append(config) or "valid",
    )

    config, validation, source = runtime_config.load_runtime_config(
        str(config_path)
    )

    assert config == {"simulation": {"device": "cpu"}}
    assert validation == "valid"
    assert source == Path(config_path)
    assert validated == [model]


def test_config_that_is_neither_mapping_nor_model_is_rejected(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "opaque.yaml"
    config_path.write_text("simulation: {}", encoding="utf-8")
    monkeypatch.setattr(runtime_config.core_config, "load_config", lambda path, *, validate: ["opaque"])
    monkeypatch.setattr(runtime_config.core_config, "validate_configuration", lambda config: "valid")

    with pytest.raises(TypeError, match="list"):
        runtime_config.load_runtime_config(str(config_path))
