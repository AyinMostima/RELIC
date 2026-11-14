"""Unit tests for the API error boundary: every error body is an APIResponse."""

from fastapi.testclient import TestClient

from relic.interfaces.api.server import create_app


def _client() -> TestClient:
    app = create_app(
        enable_cors=False,
        enable_docs=False,
        config={"security": {"api_auth": {"enabled": False}}},
    )
    return TestClient(app, raise_server_exceptions=False)


def test_route_http_exception_uses_the_api_response_body():
    response = _client().get("/graph/history/relic-test-unknown-run")

    assert response.status_code == 404
    body = response.json()
    assert body["status"] == "error"
    assert "Run history not found" in body["message"]
    assert body["data"]["error"]["type"] == "HTTPException"


def test_unknown_path_uses_the_api_response_body():
    response = _client().get("/relic-test-no-such-endpoint")

    assert response.status_code == 404
    body = response.json()
    assert body["status"] == "error"
    assert body["data"]["error"]["type"] == "HTTPException"


def test_structured_http_exception_detail_keeps_its_type_and_fields():
    from fastapi import HTTPException

    app = create_app(
        enable_cors=False,
        enable_docs=False,
        config={"security": {"api_auth": {"enabled": False}}},
    )

    @app.get("/relic-test-structured-error")
    async def _structured_error():
        raise HTTPException(
            status_code=422,
            detail={"type": "SDEValidationError", "message": "process is not bound", "bound": ["a"]},
        )

    response = TestClient(app, raise_server_exceptions=False).get("/relic-test-structured-error")

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["message"] == "process is not bound"
    assert body["data"]["error"]["type"] == "SDEValidationError"
    assert body["data"]["error"]["details"]["bound"] == ["a"]


def test_request_validation_error_uses_the_api_response_body():
    response = _client().post("/graph/validate", json=["not", "a", "mapping"])

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["data"]["error"]["type"] == "RequestValidationError"
    assert body["data"]["error"]["details"]["errors"]
