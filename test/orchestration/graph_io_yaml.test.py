"""L1 tests for Graph IR YAML loading support."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from relic.orchestration.graph_io import load_graph


def _minimal_graph_payload() -> dict:
    return {
        "schema_version": "1.0.0",
        "nodes": {
            "0": {
                "class_type": "State:ExternalForcing",
                "outputs": ["precipitation_flux"],
            }
        },
        "links": [],
        "metadata": {},
    }


def test_load_graph_from_yaml() -> None:
    yaml = pytest.importorskip("yaml")
    temp_dir = Path(tempfile.mkdtemp(prefix="graph-io-yaml-"))
    try:
        payload = _minimal_graph_payload()
        yaml_path = temp_dir / "graph.yaml"
        yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

        graph = load_graph(yaml_path)
        loaded = graph.to_dict()
        assert loaded["schema_version"] == "1.0.0"
        assert "0" in loaded["nodes"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_graph_from_json_still_works() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="graph-io-json-"))
    try:
        payload = _minimal_graph_payload()
        json_path = temp_dir / "graph.json"
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        graph = load_graph(json_path)
        loaded = graph.to_dict()
        assert loaded["schema_version"] == "1.0.0"
        assert loaded["links"] == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
