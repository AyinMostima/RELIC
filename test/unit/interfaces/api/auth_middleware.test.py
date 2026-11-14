"""Unit tests for API authentication middleware behavior."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from relic.interfaces.api.server import create_app


def _graph_payload():
    return {
        "schema_version": "1.0",
        "nodes": {
            "1": {
                "class_type": "Functor:relic.scientific.placeholder.Dummy",
                "inputs": {},
                "outputs": [],
                "config": {},
            }
        },
        "links": [],
        "metadata": {},
    }


def _client_with_auth() -> TestClient:
    app = create_app(
        enable_cors=False,
        enable_docs=False,
        config={
            "security": {
                "api_auth": {
                    "enabled": True,
                    "api_keys": ["unit-auth-key"],
                }
            }
        },
    )
    return TestClient(app)


def test_auth_middleware_blocks_non_health_without_api_key():
    client = _client_with_auth()
    response = client.post("/graph/validate?strict=false", json=_graph_payload())
    assert response.status_code == 401
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["data"]["error"]["type"] == "AuthenticationError"


def test_auth_middleware_allows_health_without_api_key():
    client = _client_with_auth()
    response = client.get("/health/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"success", "warning"}


def test_metrics_endpoint_available_without_api_key():
    client = _client_with_auth()
    response = client.get("/health/metrics")
    assert response.status_code == 200
    assert "relic_api_requests_total" in response.text
