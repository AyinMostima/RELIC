"""
Compute API routes.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import torch
from relic.core import device_policy
from fastapi import APIRouter

from relic.computation import get_device_info, healthcheck
from relic.computation.benchmarks import (
    SYNTHETIC_BENCHMARK_SCOPE,
    create_benchmark_function,
    create_test_inputs,
)
from relic.computation.distributed.launcher import distributed_healthcheck

from ..schemas import (
    APIResponse,
    ComputeBenchmarkRequest,
    ComputeHealthRequest,
    create_success_response,
)

router = APIRouter()


def _run_benchmark(request: ComputeBenchmarkRequest) -> Dict[str, Any]:
    explicit_device = None if request.device == "auto" else request.device
    device = device_policy.resolve_device(
        explicit_device=explicit_device,
        module_name=__name__,
        default="auto",
    )

    benchmark_func = create_benchmark_function(request.target)
    inputs = create_test_inputs(request.target, request.batch_size, request.n_points, device)

    # Warmup
    with torch.no_grad():
        _ = benchmark_func(inputs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings = []
    with torch.no_grad():
        for _ in range(request.n_runs):
            start = time.perf_counter()
            _ = benchmark_func(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

    avg_time = sum(timings) / max(1, len(timings))
    throughput = (request.batch_size * request.n_points) / max(avg_time, 1e-12)
    return {
        "benchmark_scope": getattr(
            benchmark_func,
            "relic_benchmark_scope",
            SYNTHETIC_BENCHMARK_SCOPE,
        ),
        "scientific_equivalence": False,
        "scientific_runtime_owner": None,
        "target_label": request.target,
        "device": str(device),
        "n_runs": request.n_runs,
        "avg_time_seconds": avg_time,
        "min_time_seconds": min(timings),
        "max_time_seconds": max(timings),
        "effective_points_per_second": throughput,
    }


@router.get("/", response_model=APIResponse)  # type: ignore[misc]
async def compute_info() -> APIResponse:
    return create_success_response(
        message="Compute endpoints are available.",
        data={
            "status": "ready",
            "endpoints": ["healthcheck", "benchmark", "ddp-check"],
            "health": healthcheck(verbose=False),
            "devices": get_device_info(),
        },
    )


@router.post("/healthcheck", response_model=APIResponse)  # type: ignore[misc]
async def run_compute_healthcheck(request: ComputeHealthRequest) -> APIResponse:
    # Failures propagate to the app-level error handler (middleware.handle_errors),
    # which returns the structured error body with an HTTP error status.
    payload = healthcheck(verbose=request.detailed)
    if request.include_benchmark:
        benchmark_request = ComputeBenchmarkRequest(
            target="photosynthesis",
            device="auto",
            batch_size=16,
            n_points=128,
            n_runs=3,
        )
        payload["quick_benchmark"] = _run_benchmark(benchmark_request)
    return create_success_response(
        message="Compute health check completed.",
        data=payload,
    )


@router.post("/benchmark", response_model=APIResponse)  # type: ignore[misc]
async def run_compute_benchmark(request: ComputeBenchmarkRequest) -> APIResponse:
    results = _run_benchmark(request)
    return create_success_response(
        message="Compute benchmark completed.",
        data=results,
    )


@router.post("/ddp-check", response_model=APIResponse)  # type: ignore[misc]
async def run_ddp_check() -> APIResponse:
    # distributed_healthcheck takes no arguments, so the route takes no request
    # fields and reports only the check that was performed.
    status = distributed_healthcheck()
    return create_success_response(
        message="Distributed health check completed.",
        data=status,
    )
