from __future__ import annotations

import json
from pathlib import Path

from test.audit import extension_validation_summary as summary_module

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "analysis_results" / "extension_validation_summary.json"


def test_extension_validation_summary_is_consistent(tmp_path, monkeypatch) -> None:
    canonical_before = OUTPUT_PATH.read_bytes()
    canonical_mtime_before = OUTPUT_PATH.stat().st_mtime_ns
    isolated_output = tmp_path / OUTPUT_PATH.name
    monkeypatch.setattr(summary_module, "OUTPUT_PATH", isolated_output)
    summary_module.main()

    assert isolated_output.exists(), "extension validation summary was not generated"
    payload = json.loads(isolated_output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0.0"
    assert payload["total_passed"] > 0
    assert payload["pass"] is True

    targets = payload["targets"]
    assert "integration_e2e" in targets
    assert "unit_extensions" in targets
    assert targets["integration_e2e"]["pass"] is True
    assert targets["unit_extensions"]["pass"] is True
    assert OUTPUT_PATH.read_bytes() == canonical_before
    assert OUTPUT_PATH.stat().st_mtime_ns == canonical_mtime_before
