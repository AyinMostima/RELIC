"""Run mypy strict on relic/core and enforce non-regression error budget."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "test" / "audit" / "mypy_core_error_budget.json"
OUTPUT_PATH = REPO_ROOT / "analysis_results" / "mypy_core_budget_summary.json"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_mypy_summary(text: str) -> tuple[int, int]:
    match = re.search(r"Found\s+(\d+)\s+errors?\s+in\s+(\d+)\s+files?", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1


def main() -> None:
    budget = _read_json(CONFIG_PATH)
    target_path = str(budget.get("target_path", "relic\\core"))
    max_errors = int(budget.get("max_errors", -1))
    mode = str(budget.get("mode", "strict"))

    cmd = [sys.executable, "-m", "mypy"]
    if mode == "strict":
        cmd.append("--strict")
    cmd.append(target_path)

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    merged = "\n".join([completed.stdout or "", completed.stderr or ""]).strip()
    error_count, file_count = _parse_mypy_summary(merged)

    # mypy returns non-zero when errors exist, so parse result text instead of exit code.
    budget_violation = 0 if (error_count >= 0 and error_count <= max_errors) else 1

    result = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(cmd),
        "target_path": target_path,
        "mode": mode,
        "return_code": completed.returncode,
        "error_count": error_count,
        "file_count": file_count,
        "max_errors": max_errors,
        "budget_violation": budget_violation,
        "pass": budget_violation == 0,
        "output_excerpt": merged[-2000:],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[OK] summary={OUTPUT_PATH}")
    print(
        "[OK] errors={errors} max={max_errors} violation={violation}".format(
            errors=error_count,
            max_errors=max_errors,
            violation=budget_violation,
        )
    )


if __name__ == "__main__":
    main()
