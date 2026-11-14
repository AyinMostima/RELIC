"""
Graph Orchestration API Routes

REST API endpoints for graph validation and compilation.
These endpoints expose the Graph IR workflow interface for orchestration.

Endpoints:
    POST /graph/validate
    POST /graph/compile
    POST /graph/submit
    GET /graph/history/{run_id}
    GET /graph/artifacts/{run_id}/{artifact_id}
    WS  /graph/ws

Authors: Yuchong Long
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from ..middleware import is_websocket_request_authorized
from ..schemas import APIResponse, create_success_response
from ....orchestration.api_service import (
    validate_graph_payload,
    compile_graph_payload,
    submit_graph_payload,
    get_run_history as _get_run_history,
    get_run_artifact as _get_run_artifact,
    get_run_status as _get_run_status,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Route failures propagate to the app-level error handler
# (middleware.handle_errors), which returns the structured error body with an
# HTTP error status (an invalid graph raises ValueError -> 400).
@router.post("/validate", response_model=APIResponse)
async def validate_graph_endpoint(payload: Dict[str, Any], strict: bool = True):
    """Validate a Graph IR payload"""
    result = validate_graph_payload(payload, strict=strict)
    return create_success_response(
        message="Graph validation completed",
        data=result,
    )


@router.post("/compile", response_model=APIResponse)
async def compile_graph_endpoint(
    payload: Dict[str, Any],
    auto_adapt: bool = False,
    strict: bool = True,
    run_bundle_dir: Optional[str] = None,
):
    """Compile a Graph IR payload"""
    result, _ = compile_graph_payload(
        payload,
        auto_adapt=auto_adapt,
        strict=strict,
        run_bundle_dir=run_bundle_dir,
    )
    return create_success_response(
        message="Graph compilation completed",
        data=result,
    )


@router.post("/submit", response_model=APIResponse)
async def submit_graph_endpoint(
    payload: Dict[str, Any],
    auto_adapt: bool = False,
    strict: bool = True,
    run_bundle_dir: Optional[str] = None,
):
    """Submit a Graph IR payload for compilation"""
    run_id, report = submit_graph_payload(
        payload,
        auto_adapt=auto_adapt,
        strict=strict,
        run_bundle_dir=run_bundle_dir,
    )
    return create_success_response(
        message="Graph submission completed",
        data={
            "run_id": run_id,
            "status": "completed",
            "report": report,
        },
    )


@router.get("/history/{run_id}", response_model=APIResponse)
async def get_graph_history(run_id: str):
    """Get run history for a submitted graph"""
    record = _get_run_history(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run history not found for run '{run_id}'; submit a graph first.",
        )
    return create_success_response(
        message="Run history retrieved",
        data=record,
    )


@router.get("/artifacts/{run_id}/{artifact_id}", response_model=APIResponse)
async def get_graph_artifact(run_id: str, artifact_id: str):
    """Get artifact metadata for a run bundle"""
    artifact_path = _get_run_artifact(run_id, artifact_id)
    if artifact_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact not found: '{artifact_id}' for run '{run_id}'.",
        )
    return create_success_response(
        message="Artifact metadata retrieved",
        data={"path": artifact_path},
    )


@router.websocket("/ws")
async def graph_ws_endpoint(websocket: WebSocket):
    """WebSocket channel for graph run status updates."""
    auth_config = getattr(getattr(websocket, "app", None), "state", None)
    resolved_auth = getattr(auth_config, "api_auth_config", None)
    if resolved_auth is not None and not is_websocket_request_authorized(
        headers=websocket.headers,
        query_params=websocket.query_params,
        auth_config=resolved_auth,
    ):
        await websocket.close(code=1008, reason="Authentication required")
        return

    await websocket.accept()
    try:
        while True:
            run_id = await websocket.receive_text()
            if run_id.strip().lower() == "close":
                await websocket.close()
                return
            status_payload = _get_run_status(run_id)
            if status_payload is None:
                await websocket.send_json(
                    {"type": "error", "run_id": run_id, "message": "Run history not found"}
                )
                continue
            await websocket.send_json(
                {"type": "status", **status_payload}
            )
    except WebSocketDisconnect:
        return
