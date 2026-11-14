"""Unit tests for the RELIC web interface app factory."""

import pytest
from fastapi.testclient import TestClient

import relic.interfaces.web.app as web_app_module
from relic.interfaces.web.app import create_web_app


def test_create_web_app_basic():
    app = create_web_app(enable_cors=False, debug=False)
    assert app.title == "RELIC Web Interface"


@pytest.mark.parametrize("page", ["/", "/health"])
def test_page_render_failure_returns_error_status(monkeypatch, page):
    """A page whose health payload fails renders the error page with HTTP 500."""

    def _missing_subsystem_table(health_checker_cls, config, detailed=True):
        raise KeyError("subsystems")

    monkeypatch.setattr(web_app_module, "build_health_payload", _missing_subsystem_table)
    client = TestClient(create_web_app(enable_cors=False, debug=False), raise_server_exceptions=False)

    response = client.get(page)

    assert response.status_code == 503
    assert "subsystems" in response.text


@pytest.mark.parametrize(
    ("overall_status", "expected_http_status"),
    [("HEALTHY", 200), ("DEGRADED", 200), ("UNHEALTHY", 503), ("unavailable", 503)],
)
def test_status_endpoint_http_status_reflects_health(monkeypatch, overall_status, expected_http_status):
    monkeypatch.setattr(
        web_app_module,
        "build_health_payload",
        lambda health_checker_cls, config, detailed=True: {"overall_status": overall_status},
    )
    client = TestClient(create_web_app(enable_cors=False, debug=False), raise_server_exceptions=False)

    response = client.get("/status")

    assert response.status_code == expected_http_status
    assert response.json()["overall_status"] == overall_status


def test_status_endpoint_health_check_failure_returns_service_unavailable(monkeypatch):
    def _crashed_check(health_checker_cls, config, detailed=True):
        raise RuntimeError("data layer health probe crashed")

    monkeypatch.setattr(web_app_module, "build_health_payload", _crashed_check)
    client = TestClient(create_web_app(enable_cors=False, debug=False), raise_server_exceptions=False)

    response = client.get("/status")

    assert response.status_code == 503
    assert "probe crashed" in response.json()["error"]


def test_health_views_receive_the_web_app_config(monkeypatch):
    received = []

    def _recording_payload(health_checker_cls, config, detailed=True):
        received.append(config)
        return {"overall_status": "HEALTHY"}

    monkeypatch.setattr(web_app_module, "build_health_payload", _recording_payload)
    app_config = {"enable_gpu_benchmark": False, "security": {"api_auth": {"enabled": False}}}
    client = TestClient(create_web_app(config=app_config, enable_cors=False, debug=False))

    assert client.get("/status").status_code == 200
    assert received == [app_config]


def test_home_page_without_overall_status_is_an_error_page(monkeypatch):
    """The health payload always carries overall_status; a missing one is not shown as 'unknown'."""
    monkeypatch.setattr(
        web_app_module,
        "build_health_payload",
        lambda health_checker_cls, config, detailed=True: {"subsystems": {}},
    )
    client = TestClient(create_web_app(enable_cors=False, debug=False), raise_server_exceptions=False)

    response = client.get("/")

    assert response.status_code == 503
    assert "overall_status" in response.text
