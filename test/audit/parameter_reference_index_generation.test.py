from __future__ import annotations

import json
from pathlib import Path

from test.audit import scientific_audit_scan as scan_module

ROOT = Path(__file__).resolve().parents[2]
INDEX_FILE = ROOT / "analysis_results" / "parameter_reference_index.json"
REPORT_FILE = ROOT / "analysis_results" / "scientific_audit_scan_20260213.json"


def test_parameter_reference_index_is_generated_with_classification_schema(
    tmp_path,
):
    index_before = INDEX_FILE.read_bytes()
    index_mtime_before = INDEX_FILE.stat().st_mtime_ns
    report_before = REPORT_FILE.read_bytes()
    report_mtime_before = REPORT_FILE.stat().st_mtime_ns
    scan_module.main(result_dir=tmp_path)
    isolated_index = tmp_path / INDEX_FILE.name

    assert isolated_index.exists(), "parameter_reference_index.json was not generated"
    payload = json.loads(isolated_index.read_text(encoding="utf-8"))

    required_keys = {
        "schema_version",
        "scan_roots",
        "python_files_scanned",
        "text_files_scanned",
        "classification_counts",
        "orphan_candidates",
        "dynamic_call_sites",
    }
    assert required_keys.issubset(payload.keys())
    assert payload["python_files_scanned"] > 0
    assert payload["text_files_scanned"] > 0

    allowed = {
        "definitely_unreferenced",
        "possible_dynamic_reference",
        "configuration_only_reference",
    }
    class_counts = payload["classification_counts"]
    assert set(class_counts.keys()) == allowed
    assert sum(class_counts.values()) == len(payload["orphan_candidates"])

    candidate_classes = {item.get("classification") for item in payload["orphan_candidates"]}
    assert candidate_classes.issubset(allowed)
    assert (tmp_path / REPORT_FILE.name).is_file()
    assert INDEX_FILE.read_bytes() == index_before
    assert INDEX_FILE.stat().st_mtime_ns == index_mtime_before
    assert REPORT_FILE.read_bytes() == report_before
    assert REPORT_FILE.stat().st_mtime_ns == report_mtime_before
