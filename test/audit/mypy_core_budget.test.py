from __future__ import annotations

import json
from pathlib import Path

from test.audit import mypy_core_budget as budget_module

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "analysis_results" / "mypy_core_budget_summary.json"


def test_mypy_core_budget_non_regression(tmp_path, monkeypatch) -> None:
    canonical_before = OUTPUT_PATH.read_bytes()
    canonical_mtime_before = OUTPUT_PATH.stat().st_mtime_ns
    isolated_output = tmp_path / OUTPUT_PATH.name
    monkeypatch.setattr(budget_module, "OUTPUT_PATH", isolated_output)
    budget_module.main()

    assert isolated_output.exists(), "mypy core budget summary was not generated"
    payload = json.loads(isolated_output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0.0"
    assert isinstance(payload["error_count"], int)
    assert isinstance(payload["max_errors"], int)
    assert payload["error_count"] >= 0
    assert payload["error_count"] <= payload["max_errors"]
    assert payload["budget_violation"] == 0
    assert payload["pass"] is True
    assert OUTPUT_PATH.read_bytes() == canonical_before
    assert OUTPUT_PATH.stat().st_mtime_ns == canonical_mtime_before
