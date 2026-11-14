"""Unit tests for API dependency services (relic.core.services.api_dependencies)."""

from types import SimpleNamespace

import pytest

import relic.computation as computation_module
from relic.core.services import api_dependencies


def test_request_without_app_config_uses_the_global_configuration_mapping():
    config = api_dependencies.get_config_from_request(None)

    assert config
    assert "simulation" in config


def test_request_with_a_non_mapping_app_config_is_rejected():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=["not", "a", "mapping"])))

    with pytest.raises(TypeError, match="mapping"):
        api_dependencies.get_config_from_request(request)


def test_computation_health_rejects_a_non_mapping_report(monkeypatch):
    monkeypatch.setattr(computation_module, "healthcheck", lambda verbose=False: ["healthy"])

    with pytest.raises(TypeError, match="mapping"):
        api_dependencies.get_computation_health()


def test_dependency_check_reports_only_optional_dependencies():
    report = api_dependencies.check_api_dependencies()

    assert set(report) == {"optional_dependencies"}
    assert set(report["optional_dependencies"]) == {"numpy", "torch", "pydantic"}
