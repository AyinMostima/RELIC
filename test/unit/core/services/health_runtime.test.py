"""Unit tests for the health payload service (relic.core.services.health_runtime)."""

import asyncio

import pytest

import relic.validation.healthcheck as healthcheck_module
from relic.core.services import health_runtime


class _RecordingChecker:
    received: list = []

    def __init__(self, config):
        type(self).received.append(config)

    async def check_all_async(self, detailed=False, benchmark=False, serialize_details=True):
        return {"overall_status": "HEALTHY"}


def test_health_checker_requires_an_explicit_configuration_mapping(monkeypatch):
    monkeypatch.setattr(healthcheck_module, "HealthChecker", _RecordingChecker)

    with pytest.raises(TypeError, match="configuration mapping"):
        health_runtime._load_health_checker(None)
    with pytest.raises(TypeError, match="configuration mapping"):
        asyncio.run(health_runtime.get_overall_health_payload(config=None))


def test_health_checker_receives_a_copy_of_the_caller_configuration(monkeypatch):
    _RecordingChecker.received.clear()
    monkeypatch.setattr(healthcheck_module, "HealthChecker", _RecordingChecker)
    config = {"enable_gpu_benchmark": False}

    payload = asyncio.run(health_runtime.get_overall_health_payload(config=config))

    assert payload == {"overall_status": "HEALTHY"}
    assert _RecordingChecker.received == [config]
    assert _RecordingChecker.received[0] is not config
