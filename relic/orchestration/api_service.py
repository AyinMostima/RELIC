"""Graph orchestration service helpers for API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging
import time
import uuid

logger = logging.getLogger(__name__)

_RUN_HISTORY: Dict[str, Dict[str, Any]] = {}


def _build_registry():
    from relic.orchestration.ssot_catalog import SSOTCatalog
    from relic.orchestration.node_registry import NodeRegistry

    catalog = SSOTCatalog.from_parameter_manager()
    return NodeRegistry(ssot_catalog=catalog)


def _load_graph_payload(payload: Dict[str, Any]):
    from relic.orchestration.graph_ir import GraphIR

    graph_payload = payload.get("graph", payload)
    if not isinstance(graph_payload, dict):
        raise TypeError("Graph payload must be a mapping")
    return GraphIR.from_dict(graph_payload)


def _record_run(
    run_id: str,
    status: str,
    report: Dict[str, Any],
    run_bundle: Optional[Dict[str, str]],
) -> None:
    events = _RUN_HISTORY.get(run_id, {}).get("events", [])
    events.append({"type": status, "timestamp": time.time()})
    _RUN_HISTORY[run_id] = {
        "status": status,
        "report": report,
        "run_bundle": run_bundle or {},
        "events": events,
        "timestamp": time.time(),
    }


def validate_graph_payload(payload: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """Validate a Graph IR payload and return errors/warnings."""
    from relic.orchestration.topology_algebra import get_topology_signature
    from relic.orchestration.validator import build_audit_summary, validate_graph

    graph = _load_graph_payload(payload)
    registry = _build_registry()
    report = validate_graph(graph, registry, strict=strict)
    audit_metadata = {"topology_signature": get_topology_signature(graph)}
    audit_summary = build_audit_summary(report, metadata=audit_metadata)
    return {
        "errors": [issue.__dict__ for issue in report.errors],
        "warnings": [issue.__dict__ for issue in report.warnings],
        "assertion_audit": dict(report.assertion_audit),
        "audit": audit_summary,
    }


def compile_graph_payload(
    payload: Dict[str, Any],
    auto_adapt: bool = False,
    strict: bool = True,
    include_topology: bool = True,
    run_bundle_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """Compile a Graph IR payload and optionally write a run bundle."""
    from relic.orchestration.compiler import compile_graph, plan_to_dict, write_run_bundle

    graph = _load_graph_payload(payload)
    registry = _build_registry()
    plan = compile_graph(
        graph,
        registry,
        auto_adapt=auto_adapt,
        strict=strict,
        include_topology=include_topology,
    )
    result = plan_to_dict(plan)
    run_bundle = None
    if run_bundle_dir:
        run_id = uuid.uuid4().hex
        bundle_path = Path(run_bundle_dir) / run_id
        run_bundle = write_run_bundle(graph, plan, bundle_path)
        result["run_bundle"] = run_bundle
    return result, run_bundle


def submit_graph_payload(
    payload: Dict[str, Any],
    auto_adapt: bool = False,
    strict: bool = True,
    include_topology: bool = True,
    run_bundle_dir: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Compile and record a Graph IR payload submission."""
    report, run_bundle = compile_graph_payload(
        payload,
        auto_adapt=auto_adapt,
        strict=strict,
        include_topology=include_topology,
        run_bundle_dir=run_bundle_dir,
    )
    run_id = uuid.uuid4().hex
    _record_run(run_id, "completed", report, run_bundle)
    return run_id, report


def get_run_history(run_id: str) -> Optional[Dict[str, Any]]:
    """Return stored run history for a graph submission."""
    return _RUN_HISTORY.get(run_id)


def get_run_artifact(run_id: str, artifact_id: str) -> Optional[str]:
    """Return artifact path for a stored run bundle."""
    record = _RUN_HISTORY.get(run_id)
    if record is None:
        return None
    run_bundle = record.get("run_bundle", {})
    return run_bundle.get(artifact_id)


def get_run_status(run_id: str) -> Optional[Dict[str, Any]]:
    """Return the current status payload for websocket updates."""
    record = _RUN_HISTORY.get(run_id)
    if record is None:
        return None
    return {
        "run_id": run_id,
        "status": record.get("status"),
        "events": record.get("events", []),
        "timestamp": record.get("timestamp"),
    }


__all__ = [
    "validate_graph_payload",
    "compile_graph_payload",
    "submit_graph_payload",
    "get_run_history",
    "get_run_artifact",
    "get_run_status",
]
