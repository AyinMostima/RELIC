import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from relic.interfaces.api.server import create_app

_TEST_API_KEY = "test-graph-key"


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


def _build_client() -> TestClient:
    app = create_app(
        enable_cors=False,
        enable_docs=False,
        config={
            "security": {
                "api_auth": {
                    "enabled": True,
                    "api_keys": [_TEST_API_KEY],
                }
            }
        },
    )
    # Route failures reach the app-level error handler, which returns the
    # structured error body with an HTTP error status instead of raising here.
    return TestClient(app, raise_server_exceptions=False)


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": _TEST_API_KEY}


def test_graph_validate_endpoint():
    client = _build_client()
    payload = _graph_payload()

    response = client.post("/graph/validate?strict=false", json=payload, headers=_auth_headers())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "errors" in data["data"]


def test_graph_compile_rejects_unregistered_node():
    client = _build_client()
    payload = _graph_payload()

    response = client.post("/graph/compile", json=payload, headers=_auth_headers())
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["data"]["error"]["type"] == "ValidationError"
    assert "authoritative NodeRegistry definition" in data["message"]


def test_graph_submit_rejects_unregistered_node_without_history():
    client = _build_client()
    payload = _graph_payload()

    submit_response = client.post("/graph/submit", json=payload, headers=_auth_headers())
    assert submit_response.status_code == 400
    submit_data = submit_response.json()
    assert submit_data["status"] == "error"
    assert submit_data["data"]["error"]["type"] == "ValidationError"
    assert "authoritative NodeRegistry definition" in submit_data["message"]


def test_graph_history_and_artifact_lookups_return_not_found():
    client = _build_client()

    history = client.get("/graph/history/relic-test-unknown-run", headers=_auth_headers())
    artifact = client.get(
        "/graph/artifacts/relic-test-unknown-run/plan", headers=_auth_headers()
    )

    assert history.status_code == 404
    assert history.json()["status"] == "error"
    assert "Run history not found" in history.json()["message"]
    assert artifact.status_code == 404
    assert "Artifact not found" in artifact.json()["message"]
