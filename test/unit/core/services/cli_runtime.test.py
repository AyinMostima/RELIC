"""Unit tests for CLI runtime helpers (relic.core.services.cli_runtime)."""

import logging

import pytest

from relic.core.services import cli_runtime


def test_invalid_log_level_environment_value_is_rejected(monkeypatch):
    monkeypatch.setenv("RELIC_LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValueError, match="VERBOSE"):
        cli_runtime.setup_logging()


def test_system_info_reports_the_effective_log_level_and_config_source(monkeypatch):
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.WARNING)
    try:
        lines = cli_runtime.get_system_info_lines(
            "test",
            True,
            lambda: ({"simulation": {}}, None, None),
        )
    finally:
        root.setLevel(previous)

    assert any(line.startswith("Log Level:") and line.endswith("WARNING") for line in lines)
    assert any(line.startswith("Config File:") and "global default configuration" in line for line in lines)


def test_explicit_empty_experimental_variable_list_checks_nothing(monkeypatch):
    monkeypatch.setenv("RELIC_EXPERIMENTAL_WEB", "1")

    assert cli_runtime.get_experimental_features([]) == []
    assert "RELIC_EXPERIMENTAL_WEB" in cli_runtime.get_experimental_features()
