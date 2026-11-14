"""Unit tests for configuration API route failure propagation."""

import asyncio

import pytest

from relic.interfaces.api.routes import config as config_routes
from relic.interfaces.api.schemas import ConfigValidationRequest


def test_config_validation_failure_reaches_the_api_error_boundary(monkeypatch):
    """A configuration that cannot be validated raises; the app error handler sets the HTTP status."""

    def _unreadable_configuration(candidate, strict=False):
        raise OSError("configuration schema file is unreadable")

    monkeypatch.setattr(config_routes.core_config, "validate_configuration", _unreadable_configuration)
    request = ConfigValidationRequest(config={})

    with pytest.raises(OSError, match="configuration schema file is unreadable"):
        asyncio.run(config_routes.validate_config(request))


def test_config_info_requires_the_validation_result_attributes(monkeypatch):
    """A validation result without is_valid is a defect, not a valid configuration."""

    class _IncompleteValidationResult:
        num_errors = 0
        num_warnings = 0

    monkeypatch.setattr(
        config_routes,
        "load_runtime_config",
        lambda: ({}, _IncompleteValidationResult(), None),
    )

    with pytest.raises(AttributeError, match="is_valid"):
        asyncio.run(config_routes.config_info())
