"""
Execution plan utilities for multi-scale coupling.

Produces a manifest that documents scale levels, regrid steps, and hashes
without altering runtime execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import yaml
import torch

from relic.parameters import get_global_manager

logger = logging.getLogger(__name__)


@dataclass
class LevelSpec:
    name: str
    time_step: Optional[float] = None
    spatial_resolution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegridSpec:
    source: str
    target: str
    method: str = "hierarchical"
    time_ratio: Optional[int] = None
    space_mapping: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStep:
    source: str
    target: str
    method: str
    time_ratio: Optional[int] = None


def load_scaling_specs(
    path: Path,
    parameter_manager: Optional[Any] = None,
) -> Tuple[List[LevelSpec], List[RegridSpec]]:
    if not path.exists():
        raise FileNotFoundError(f"Scaling specs file not found: {path}")
    param_manager = get_global_manager()
    data = param_manager.load_yaml_file(path, namespace="coupling.scales.execution_plan")
    levels_raw = data.get("levels", []) or []
    regrids_raw = data.get("regrids", []) or []

    def _resolve_time_step(entry: Dict[str, Any]) -> Optional[float]:
        raw_time_step = entry.get("time_step")
        time_step_key = entry.get("time_step_parameter")
        if time_step_key:
            key = str(time_step_key).strip()
            resolved = param_manager.get(
                key,
                required=True,
                expected_type=(float, int, torch.Tensor),
            )
            if isinstance(resolved, torch.Tensor):
                return float(resolved.detach().cpu().item())
            return float(resolved)
        if raw_time_step is None:
            return None
        if isinstance(raw_time_step, torch.Tensor):
            return float(raw_time_step.detach().cpu().item())
        return float(raw_time_step)

    levels = [
        LevelSpec(
            name=str(entry.get("name", "")).strip(),
            time_step=_resolve_time_step(entry),
            spatial_resolution=entry.get("spatial_resolution"),
            metadata=dict(entry.get("metadata", {}) or {}),
        )
        for entry in levels_raw
    ]
    regrids = [
        RegridSpec(
            source=str(entry.get("source", "")).strip(),
            target=str(entry.get("target", "")).strip(),
            method=str(entry.get("method", "hierarchical")),
            time_ratio=entry.get("time_ratio"),
            space_mapping=entry.get("space_mapping"),
            metadata=dict(entry.get("metadata", {}) or {}),
        )
        for entry in regrids_raw
    ]
    return levels, regrids


def validate_topology(levels: List[LevelSpec], regrids: List[RegridSpec]) -> None:
    errors: List[str] = []
    level_names = [lvl.name for lvl in levels if lvl.name]
    if len(level_names) != len(set(level_names)):
        errors.append("Duplicate level names detected in LevelSpec")
    name_set = set(level_names)
    for spec in regrids:
        if spec.source not in name_set:
            errors.append(f"RegridSpec source '{spec.source}' is not a defined level")
        if spec.target not in name_set:
            errors.append(f"RegridSpec target '{spec.target}' is not a defined level")
        if spec.time_ratio is not None and int(spec.time_ratio) <= 0:
            errors.append(f"RegridSpec time_ratio must be positive for {spec.source}->{spec.target}")
    if errors:
        raise ValueError("Execution plan topology invalid: " + "; ".join(errors))


def build_execution_plan(levels: List[LevelSpec], regrids: List[RegridSpec]) -> List[ExecutionStep]:
    if regrids:
        return [
            ExecutionStep(
                source=spec.source,
                target=spec.target,
                method=spec.method,
                time_ratio=spec.time_ratio,
            )
            for spec in regrids
        ]
    steps: List[ExecutionStep] = []
    for idx in range(len(levels) - 1):
        steps.append(
            ExecutionStep(
                source=levels[idx].name,
                target=levels[idx + 1].name,
                method="hierarchical",
                time_ratio=None,
            )
        )
    return steps


def build_manifest(levels: List[LevelSpec], regrids: List[RegridSpec], steps: List[ExecutionStep]) -> Dict[str, Any]:
    payload = {
        "schema_version": "1.0.0",
        "levels": [vars(level) for level in levels],
        "regrids": [vars(spec) for spec in regrids],
        "steps": [vars(step) for step in steps],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()[:12]
    payload["hash"] = digest
    return payload


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")



