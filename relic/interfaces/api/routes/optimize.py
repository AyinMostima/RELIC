"""
Optimization API routes.
"""

from __future__ import annotations

from typing import Any, Dict, NoReturn

from fastapi import APIRouter, HTTPException, status

from ..schemas import (
    APIResponse,
    CalibrationRequest,
    SensitivityRequest,
    UncertaintyRequest,
)

router = APIRouter()

_RUNTIME_UNBOUND_MESSAGE = (
    "[N] optimization runtime_unbound: production evaluation requires a "
    "caller-supplied sealed GraphIR/ExecutionPlan/GraphExecutor binding, the "
    "canonical simulator lifecycle, explicit initial state and output "
    "projection, unit and observation contracts, and restart-safe RNG semantics."
)


def _raise_runtime_unbound() -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "status": "[N]",
            "runtime_ready": False,
            "runtime_blocker": "runtime_unbound",
            "code": "runtime_unbound",
            "message": _RUNTIME_UNBOUND_MESSAGE,
        },
    )


@router.get("/", response_model=APIResponse)
async def optimize_info() -> APIResponse:
    return APIResponse(
        status="warning",
        message=_RUNTIME_UNBOUND_MESSAGE,
        data={
            "status": "[N]",
            "runtime_ready": False,
            "runtime_blocker": "runtime_unbound",
            "endpoints": ["calibrate", "assimilate", "sensitivity", "uncertainty"],
            "component_algorithms": {
                "calibration": ["differential_evolution", "nelder_mead", "l_bfgs_b"],
                "sensitivity": ["morris", "sobol"],
                "uncertainty": ["ensemble", "bootstrap"],
            },
        },
    )


@router.post("/calibrate", response_model=APIResponse)
async def calibrate_parameters(request: CalibrationRequest) -> APIResponse:
    _raise_runtime_unbound()


@router.post("/assimilate", response_model=APIResponse)
async def assimilate_data(payload: Dict[str, Any]) -> APIResponse:
    _raise_runtime_unbound()


@router.post("/sensitivity", response_model=APIResponse)
async def run_sensitivity(request: SensitivityRequest) -> APIResponse:
    _raise_runtime_unbound()


@router.post("/uncertainty", response_model=APIResponse)
async def run_uncertainty_quantification(request: UncertaintyRequest) -> APIResponse:
    _raise_runtime_unbound()
