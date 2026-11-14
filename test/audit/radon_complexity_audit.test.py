from __future__ import annotations

import json
from pathlib import Path

from test.audit import radon_complexity_audit as audit_module

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "analysis_results" / "radon_complexity_summary.json"
INPUT_PATH = ROOT / "analysis_results" / "radon_cc.json"


def test_radon_complexity_summary_is_deterministic_and_complete(
    tmp_path,
    monkeypatch,
) -> None:
    canonical_before = SUMMARY_PATH.read_bytes()
    canonical_mtime_before = SUMMARY_PATH.stat().st_mtime_ns
    input_before = INPUT_PATH.read_bytes()
    input_mtime_before = INPUT_PATH.stat().st_mtime_ns
    isolated_output = tmp_path / SUMMARY_PATH.name
    monkeypatch.setattr(audit_module, "OUTPUT_PATH", isolated_output)
    audit_module.main()

    assert isolated_output.exists(), "radon complexity summary was not generated"
    payload = json.loads(isolated_output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0.0"
    metrics = payload["metrics"]
    assert metrics["files_scanned"] > 0
    assert metrics["entries_top_level"] > 0
    assert metrics["entries_flat"] >= metrics["entries_top_level"]
    assert metrics["complexity_gt15_flat"] >= metrics["complexity_gt15_top_level"]
    assert metrics["complexity_gt30_flat"] <= metrics["complexity_gt15_flat"]

    hotspots = payload["top_hotspots_flat"]
    assert hotspots, "top hotspots should not be empty"
    assert all("complexity" in item for item in hotspots)

    monitored = payload["monitored_hotspots"]
    key_a = "relic\\core\\simulator\\execution_group.py::execute_group"
    key_b = "relic\\core\\functor\\functor_base.py::get_parameter_value"
    assert key_a in monitored
    assert key_b in monitored
    assert monitored[key_a]["present"] is True
    assert monitored[key_b]["present"] is True
    assert SUMMARY_PATH.read_bytes() == canonical_before
    assert SUMMARY_PATH.stat().st_mtime_ns == canonical_mtime_before
    assert INPUT_PATH.read_bytes() == input_before
    assert INPUT_PATH.stat().st_mtime_ns == input_mtime_before
