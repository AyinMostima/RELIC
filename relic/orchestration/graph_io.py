"""
Graph IR I/O utilities.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema

from .graph_ir import GraphIR


def _should_validate_schema(validate_schema: bool) -> bool:
    if validate_schema:
        return True
    env_flag = os.getenv("RELIC_GRAPH_SCHEMA_VALIDATE", "0").strip().lower()
    return env_flag in {"1", "true", "yes", "on"}


def validate_graph_schema(payload: Dict[str, Any], schema_path: Optional[Path] = None) -> None:
    """Validate Graph IR payload against the JSON schema."""
    schema_file = schema_path or Path(__file__).with_name("graph_schema.json")
    schema = json.loads(schema_file.read_text(encoding="utf-8-sig"))
    jsonschema.validate(instance=payload, schema=schema)


def load_graph(
    path: Path,
    *,
    validate_schema: bool = False,
    schema_path: Optional[Path] = None,
) -> GraphIR:
    """Load a Graph IR file from JSON or YAML."""
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to load YAML graph files.") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Graph payload in {path} must be a mapping.")
    if _should_validate_schema(validate_schema):
        validate_graph_schema(payload, schema_path=schema_path)
    return GraphIR.from_dict(payload)


def save_graph(graph: GraphIR, path: Path) -> None:
    """Save a Graph IR to JSON."""
    payload = graph.to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
