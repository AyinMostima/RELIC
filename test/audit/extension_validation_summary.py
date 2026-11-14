"""Consolidate extension validation evidence into one machine-readable summary."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "analysis_results" / "extension_validation_summary.json"
BASE_TEMP_ROOT = Path(tempfile.gettempdir()) / "relic-pytest-extension"

TEST_TARGETS = {
    "integration_e2e": "test\\integration\\extensions\\test_extension_e2e.py",
    "unit_extensions": "test\\unit\\extensions",
}


def _parse_pytest_counts(output: str) -> Dict[str, int]:
    counts = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }
    patterns = {
        "passed": r"(\d+)\s+passed",
        "failed": r"(\d+)\s+failed",
        "errors": r"(\d+)\s+error",
        "skipped": r"(\d+)\s+skipped",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            counts[key] = int(match.group(1))
    return counts


def _run_target(label: str, target: str) -> Dict[str, object]:
    BASE_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    run_basetemp = BASE_TEMP_ROOT / f"{label}_{uuid4().hex[:8]}"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--basetemp",
        str(run_basetemp),
        target,
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    merged_output = "\n".join([completed.stdout or "", completed.stderr or ""]).strip()
    counts = _parse_pytest_counts(merged_output)
    return {
        "label": label,
        "target": target,
        "command": " ".join(cmd),
        "return_code": completed.returncode,
        "counts": counts,
        "output_excerpt": merged_output[-1200:],
        "pass": completed.returncode == 0 and counts["failed"] == 0 and counts["errors"] == 0,
    }


def main() -> None:
    results = {label: _run_target(label, target) for label, target in TEST_TARGETS.items()}
    all_pass = all(item["pass"] for item in results.values())
    total_passed = sum(int(item["counts"]["passed"]) for item in results.values())

    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets": results,
        "total_passed": total_passed,
        "pass": all_pass and total_passed > 0,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[OK] summary={OUTPUT_PATH}")
    print(f"[OK] pass={payload['pass']} total_passed={payload['total_passed']}")


if __name__ == "__main__":
    main()
