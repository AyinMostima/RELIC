"""Middleware settings are stated explicitly and validated; the rate limit has one source."""

import dataclasses
import inspect

import pytest
from fastapi import FastAPI

from relic.interfaces.api import middleware
from relic.interfaces.api.middleware import (
    APIAuthConfig,
    RateLimitingMiddleware,
    RequestLoggingMiddleware,
)
from relic.interfaces.api.server import create_app
from relic.parameters import get_global_manager

pytestmark = pytest.mark.unit

RATE_KEY = "interfaces.api.rate_limit.requests_per_minute"
WINDOW_KEY = "interfaces.api.rate_limit.window_seconds"


def _rate_limiter(app):
    for entry in app.user_middleware:
        if entry.cls is RateLimitingMiddleware:
            return entry.kwargs
    raise AssertionError("RateLimitingMiddleware is not installed")


@pytest.mark.parametrize("name", ["requests_per_minute", "window_seconds"])
def test_rate_limiter_settings_are_required(name):
    parameter = inspect.signature(RateLimitingMiddleware.__init__).parameters[name]
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"requests_per_minute": 0, "window_seconds": 60}, ValueError),
        ({"requests_per_minute": -5, "window_seconds": 60}, ValueError),
        ({"requests_per_minute": 10, "window_seconds": 0}, ValueError),
        ({"requests_per_minute": 2.5, "window_seconds": 60}, TypeError),
        ({"requests_per_minute": True, "window_seconds": 60}, TypeError),
        ({"requests_per_minute": 10, "window_seconds": "60"}, TypeError),
    ],
)
def test_rate_limiter_rejects_invalid_settings_instead_of_correcting_them(kwargs, error):
    with pytest.raises(error):
        RateLimitingMiddleware(FastAPI(), **kwargs)


def test_rate_limiter_keeps_valid_settings_unchanged():
    limiter = RateLimitingMiddleware(FastAPI(), requests_per_minute=7, window_seconds=30)
    assert (limiter.requests_per_minute, limiter.window_seconds) == (7, 30)


def test_create_app_reads_the_rate_limit_from_the_parameter_database():
    manager = get_global_manager()
    kwargs = _rate_limiter(create_app(enable_cors=False, enable_docs=False))

    assert kwargs["requests_per_minute"] == manager.get(RATE_KEY, required=True)
    assert kwargs["window_seconds"] == manager.get(WINDOW_KEY, required=True)
    assert inspect.signature(create_app).parameters["requests_per_minute"].default is None


def test_create_app_passes_an_explicit_rate_limit_through():
    kwargs = _rate_limiter(create_app(enable_cors=False, enable_docs=False, requests_per_minute=5))

    assert kwargs["requests_per_minute"] == 5


def test_auth_config_and_request_logging_have_no_literal_defaults():
    fields = {f.name: f for f in dataclasses.fields(APIAuthConfig)}
    for name in ("exempt_path_prefixes", "header_name"):
        assert fields[name].default is dataclasses.MISSING, name
    parameter = inspect.signature(RequestLoggingMiddleware.__init__).parameters["log_level"]
    assert parameter.default is inspect.Parameter.empty


def test_api_key_header_name_has_one_source():
    default = inspect.signature(middleware.extract_api_key).parameters["header_name"].default

    assert default is middleware.DEFAULT_API_KEY_HEADER
    assert middleware.build_api_auth_config({}).header_name == middleware.DEFAULT_API_KEY_HEADER


def test_request_logging_rejects_an_unknown_level_instead_of_using_info():
    with pytest.raises(ValueError, match="VERBOSE"):
        RequestLoggingMiddleware(FastAPI(), log_level="verbose")

    assert RequestLoggingMiddleware(FastAPI(), log_level="debug").log_level == 10


def _request_without_client(headers=()):
    from starlette.requests import Request

    return Request({"type": "http", "method": "GET", "path": "/data", "headers": list(headers)})


def test_requests_without_a_client_address_share_one_rate_limit_bucket():
    """A request with neither X-Forwarded-For nor a transport address is counted in one shared bucket."""
    import asyncio

    first = RateLimitingMiddleware._get_client_ip(_request_without_client())
    second = RateLimitingMiddleware._get_client_ip(_request_without_client())
    assert first == second == middleware.MISSING_CLIENT_ADDRESS_BUCKET

    limiter = RateLimitingMiddleware(FastAPI(), requests_per_minute=1, window_seconds=60)
    assert asyncio.run(limiter._check_rate_limit(first)) is True
    assert asyncio.run(limiter._check_rate_limit(second)) is False


def test_forwarded_address_takes_precedence_over_the_missing_address_bucket():
    request = _request_without_client([(b"x-forwarded-for", b"203.0.113.7, 10.0.0.1")])

    assert RateLimitingMiddleware._get_client_ip(request) == "203.0.113.7"


def test_rate_limit_documentation_states_the_shared_missing_address_bucket():
    for text in (RateLimitingMiddleware.__doc__, RateLimitingMiddleware._get_client_ip.__doc__):
        assert text is not None
        assert "shared" in text.lower()
        assert "MISSING_CLIENT_ADDRESS_BUCKET" in text
