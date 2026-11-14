"""Unit tests for computation API benchmark scope and failure propagation."""

import asyncio

import pytest

from relic.interfaces.api.routes import compute as compute_routes
from relic.interfaces.api.routes.compute import _run_benchmark
from relic.interfaces.api.schemas import (
    ComputeBenchmarkRequest,
    ComputeHealthRequest,
)


def _benchmark_request() -> ComputeBenchmarkRequest:
    return ComputeBenchmarkRequest(
        target="transport",
        device="cpu",
        batch_size=1,
        n_points=8,
        n_runs=1,
        test_jit=False,
    )


def _compute_backend_failure(*_args, **_kwargs):
    raise RuntimeError("compute backend unavailable")


@pytest.mark.parametrize(
    ("patched_name", "route_name", "request_factory"),
    [
        ("healthcheck", "run_compute_healthcheck", ComputeHealthRequest),
        ("_run_benchmark", "run_compute_benchmark", _benchmark_request),
    ],
)
def test_compute_route_failure_reaches_the_api_error_boundary(
    monkeypatch, patched_name, route_name, request_factory
):
    """A failing compute route raises; the app error handler sets the HTTP error status."""
    monkeypatch.setattr(compute_routes, patched_name, _compute_backend_failure)
    route = getattr(compute_routes, route_name)

    with pytest.raises(RuntimeError, match="compute backend unavailable"):
        asyncio.run(route(request_factory()))


def test_compute_benchmark_is_explicitly_synthetic_and_non_scientific():
    request = _benchmark_request()

    payload = _run_benchmark(request)

    assert payload["benchmark_scope"] == "synthetic_single_environment_microbenchmark"
    assert payload["scientific_equivalence"] is False
    assert payload["scientific_runtime_owner"] is None
    assert payload["target_label"] == "transport"



def test_ddp_check_failure_reaches_the_api_error_boundary(monkeypatch):
    monkeypatch.setattr(compute_routes, "distributed_healthcheck", _compute_backend_failure)

    with pytest.raises(RuntimeError, match="compute backend unavailable"):
        asyncio.run(compute_routes.run_ddp_check())


def test_ddp_check_reports_only_the_performed_check(monkeypatch):
    """The check takes no request: nothing is echoed as if it had been used."""
    monkeypatch.setattr(compute_routes, "distributed_healthcheck", lambda: {"status": "ok"})

    response = asyncio.run(compute_routes.run_ddp_check())

    assert response.data == {"status": "ok"}
