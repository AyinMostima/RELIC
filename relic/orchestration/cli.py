"""
CLI helpers for orchestration validation and compilation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .graph_io import load_graph
from .ssot_catalog import SSOTCatalog
from .node_registry import NodeRegistry
from .topology_algebra import get_topology_signature
from .validator import build_audit_summary, validate_graph
from .compiler import compile_graph, plan_to_dict, write_run_bundle


def _build_registry() -> NodeRegistry:
    catalog = SSOTCatalog.from_parameter_manager()
    return NodeRegistry(ssot_catalog=catalog)


def run_validate(path: Path, strict: bool = True, include_topology: bool = True) -> Dict[str, Any]:
    graph = load_graph(path)
    registry = _build_registry()
    report = validate_graph(graph, registry, strict=strict)
    audit_metadata: Dict[str, Any] = {}
    if include_topology:
        audit_metadata["topology_signature"] = get_topology_signature(graph)
    audit_summary = build_audit_summary(report, metadata=audit_metadata)
    return {
        "errors": [issue.__dict__ for issue in report.errors],
        "warnings": [issue.__dict__ for issue in report.warnings],
        "assertion_audit": dict(report.assertion_audit),
        "audit": audit_summary,
    }


def run_compile(
    path: Path,
    auto_adapt: bool = False,
    strict: bool = True,
    include_topology: bool = True,
    run_bundle_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    graph = load_graph(path)
    registry = _build_registry()
    plan = compile_graph(
        graph,
        registry,
        auto_adapt=auto_adapt,
        strict=strict,
        include_topology=include_topology,
    )
    payload = plan_to_dict(plan)
    if run_bundle_dir is not None:
        payload["run_bundle"] = write_run_bundle(graph, plan, run_bundle_dir)
    return payload


def _write_output(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
