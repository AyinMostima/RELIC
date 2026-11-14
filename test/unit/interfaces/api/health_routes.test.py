"""Unit tests for health API routes."""

import asyncio

import pytest


def _run(coro):
    return asyncio.run(coro)


def test_run_system_benchmark_route(monkeypatch):
    from relic.interfaces.api.routes import health as routes
    import relic.validation.healthcheck as healthcheck_mod

    class DummyChecker:
        received = []

        def __init__(self, config):
            self.config = config
            type(self).received.append(config)

        def run_system_benchmark(self):
            return {"ok": True}

    monkeypatch.setattr(healthcheck_mod, "HealthChecker", DummyChecker)
    monkeypatch.setattr(routes, "get_config", lambda: {"security": {}})

    response = _run(routes.run_system_benchmark(background_tasks=None, gpu_benchmark=True))
    assert response.status == "success"
    assert response.data["ok"] is True
    assert DummyChecker.received == [{"security": {}, "enable_gpu_benchmark": True}]


def _health_backend_failure(*_args, **_kwargs):
    raise RuntimeError("health backend unavailable")


async def _async_health_backend_failure(*_args, **_kwargs):
    _health_backend_failure()


@pytest.mark.parametrize(
    ("patched_name", "route_name", "route_kwargs"),
    [
        ("get_overall_health_payload", "get_overall_health", {}),
        ("get_subsystem_health_payload", "get_compute_health", {}),
        ("get_subsystem_health_payload", "get_data_health", {}),
        ("get_subsystem_health_payload", "get_params_health", {}),
        ("get_subsystem_health_payload", "get_optimize_health", {}),
        ("get_subsystem_health_payload", "get_stochastic_health", {}),
        ("get_subsystem_health_payload", "get_orchestration_health", {}),
        ("get_subsystem_health_payload", "get_config_health", {}),
        ("run_system_benchmark_payload", "run_system_benchmark", {"background_tasks": None}),
    ],
)
def test_health_route_failure_reaches_the_api_error_boundary(
    monkeypatch, patched_name, route_name, route_kwargs
):
    """A failing health route raises; the app error handler sets the HTTP error status."""
    from relic.interfaces.api.routes import health as routes

    replacement = (
        _health_backend_failure
        if patched_name == "run_system_benchmark_payload"
        else _async_health_backend_failure
    )
    monkeypatch.setattr(routes, patched_name, replacement)
    monkeypatch.setattr(routes, "get_config", lambda: {})

    with pytest.raises(RuntimeError, match="health backend unavailable"):
        _run(getattr(routes, route_name)(**route_kwargs))


def test_api_health_routes_require_fastapi():
    """Without fastapi the API routes raise ImportError; no mock router is substituted."""
    import subprocess
    import sys
    from pathlib import Path

    probe = (
        "import sys\n"
        "sys.modules['fastapi'] = None\n"
        "import relic.interfaces.api.routes.health\n"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode != 0
    assert "fastapi" in completed.stderr


@pytest.mark.parametrize(
    ("health_status", "http_status", "body_status"),
    [("HEALTHY", 200, "success"), ("DEGRADED", 200, "warning"), ("UNHEALTHY", 503, "error"), ("UNAVAILABLE", 503, "error")],
)
def test_health_routes_answer_with_the_http_status_of_the_health(
    monkeypatch, health_status, http_status, body_status
):
    import json

    from relic.interfaces.api.routes import health as routes

    async def _overall(**_kwargs):
        return {"overall_status": health_status}

    async def _subsystem(_name, **_kwargs):
        return {"status": health_status}

    monkeypatch.setattr(routes, "get_overall_health_payload", _overall)
    monkeypatch.setattr(routes, "get_subsystem_health_payload", _subsystem)
    monkeypatch.setattr(routes, "get_config", lambda: {})

    for route in (routes.get_overall_health, routes.get_compute_health, routes.get_config_health):
        response = _run(route())
        assert response.status_code == http_status
        body = json.loads(response.body)
        assert body["status"] == body_status
