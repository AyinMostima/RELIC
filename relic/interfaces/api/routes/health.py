"""
Health Check API Routes

REST API endpoints for RELIC system health monitoring and diagnostics.
Provides comprehensive health status across all subsystems with detailed
diagnostics and performance metrics.

Endpoints:
    GET /health              Overall system health
    GET /health/compute      Computational layer health
    GET /health/data         Data I/O layer health
    GET /health/params       Parameter system health
    GET /health/optimize     Optimization algorithms health
    GET /health/stochastic   Stochastic processes health
    GET /health/orchestration Graph orchestration health
    GET /health/config       Configuration system health
    POST /health/benchmark   System performance benchmark

Authors: Yuchong Long
"""

from typing import Any

# fastapi is required by the API package (server, routes); without it the
# import fails with the original ImportError instead of a mock router.
from fastapi import APIRouter, BackgroundTasks, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from ..schemas import APIResponse, create_success_response
from ..dependencies import get_config, get_logger
from ....core.services.health_runtime import (
    get_overall_health_payload,
    get_subsystem_health_payload,
    run_system_benchmark_payload,
)

logger = get_logger(__name__)
router = APIRouter()

# HTTP and body status per normalized health status (HealthChecker contract):
# HEALTHY answers 200/success, DEGRADED 200/warning, any other status 503/error.
_HEALTH_HTTP_STATUS = {
    "HEALTHY": (200, "success"),
    "DEGRADED": (200, "warning"),
}


def _health_response(message: str, payload: dict[str, Any], health_status: Any) -> JSONResponse:
    """Wrap a health payload so the HTTP status reflects the reported health."""
    normalized = str(health_status).upper()
    http_status, body_status = _HEALTH_HTTP_STATUS.get(normalized, (503, "error"))
    response = create_success_response(message=f"{message}: {normalized}", data=payload)
    response.status = body_status
    return JSONResponse(status_code=http_status, content=jsonable_encoder(response))


# Route failures propagate to the app-level error handler
# (middleware.handle_errors), which returns the structured error body with an
# HTTP error status; a failed health query never answers HTTP 200.
@router.get("/", response_model=APIResponse)
async def get_overall_health():
    """Get comprehensive system health status"""
    config = get_config()
    health_result = await get_overall_health_payload(
        config=config,
        detailed=False,
        benchmark=False,
        serialize_details=True,
    )

    return _health_response(
        "System health check completed",
        health_result,
        health_result["overall_status"],
    )

@router.get("/compute", response_model=APIResponse)
async def get_compute_health():
    """Get computational layer health status"""
    config = get_config()
    status = await get_subsystem_health_payload(
        "compute",
        config=config,
        detailed=True,
        serialize_details=True,
    )

    return _health_response(
        "Compute health check completed",
        status,
        status["status"],
    )

@router.get("/data", response_model=APIResponse)
async def get_data_health():
    """Get data I/O layer health status"""
    config = get_config()
    data_status = await get_subsystem_health_payload(
        "data",
        config=config,
        detailed=True,
        serialize_details=True,
    )

    return _health_response(
        "Data health check completed",
        data_status,
        data_status["status"],
    )

@router.get("/params", response_model=APIResponse)
async def get_params_health():
    """Get parameter system health status"""
    config = get_config()
    status = await get_subsystem_health_payload(
        "params",
        config=config,
        detailed=True,
        serialize_details=True,
    )

    return _health_response(
        "Parameter health check completed",
        status,
        status["status"],
    )

@router.get("/optimize", response_model=APIResponse)
async def get_optimize_health():
    """Get optimization algorithms health status"""
    config = get_config()
    result = await get_subsystem_health_payload(
        "optimize",
        config=config,
        detailed=True,
        serialize_details=True,
    )

    return _health_response(
        "Optimization health check completed",
        result,
        result["status"],
    )

@router.get("/stochastic", response_model=APIResponse)
async def get_stochastic_health():
    """Get stochastic processes health status"""
    config = get_config()
    result = await get_subsystem_health_payload(
        "stochastic",
        config=config,
        detailed=True,
        serialize_details=True,
    )

    return _health_response(
        "Stochastic health check completed",
        result,
        result["status"],
    )

@router.get("/orchestration", response_model=APIResponse)
async def get_orchestration_health():
    """Get graph orchestration health status"""
    config = get_config()
    result = await get_subsystem_health_payload(
        "orchestration",
        config=config,
        detailed=True,
        serialize_details=True,
    )

    return _health_response(
        "Orchestration health check completed",
        result,
        result["status"],
    )

@router.get("/config", response_model=APIResponse)
async def get_config_health():
    """Get configuration system health status"""
    config = get_config()
    status = await get_subsystem_health_payload(
        "config",
        config=config,
        detailed=True,
        serialize_details=True,
    )

    return _health_response(
        "Configuration health check completed",
        status,
        status["status"],
    )

@router.post("/benchmark", response_model=APIResponse)
async def run_system_benchmark(
    background_tasks: BackgroundTasks,
    gpu_benchmark: bool = Query(..., description="Run (true) or skip (false) the GPU benchmark"),
):
    """Run comprehensive system performance benchmark.

    ``gpu_benchmark`` is a required query parameter: the GPU part of the
    benchmark is always an explicit request option.
    """
    health_config = {**get_config(), "enable_gpu_benchmark": gpu_benchmark}
    benchmark_payload = run_system_benchmark_payload(health_config)

    return create_success_response(
        message="System benchmark completed",
        data=benchmark_payload,
    )
