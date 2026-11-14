"""
Daily audit report generator for coupling JSONL logs.

Reads JSONL records produced by AuditCollector and outputs a Markdown summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _build_hotspot_matrix(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for record in records:
        operation = str(record.get("operation") or "unknown")
        variables = record.get("variables") or []
        if not isinstance(variables, list):
            variables = []
        op_bucket = matrix.setdefault(operation, {})
        for var in variables:
            key = str(var)
            op_bucket[key] = op_bucket.get(key, 0) + 1
    return matrix


def _top_violations(records: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    def score(rec: Dict[str, Any]) -> float:
        try:
            return float(rec.get("relative_error") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted(records, key=score, reverse=True)
    return ordered[:limit]


def _render_markdown(records: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Coupling Audit Daily Report")
    lines.append("")
    lines.append(f"Total records: {len(records)}")
    lines.append("")

    matrix = _build_hotspot_matrix(records)
    lines.append("## Hotspot Matrix")
    if not matrix:
        lines.append("- No records available.")
    else:
        for operation, vars_map in matrix.items():
            if not vars_map:
                continue
            top_vars = sorted(vars_map.items(), key=lambda x: x[1], reverse=True)[:10]
            summary = ", ".join(f"{name}={count}" for name, count in top_vars)
            lines.append(f"- {operation}: {summary}")
    lines.append("")

    lines.append("## Top Violations")
    violations = _top_violations(records)
    if not violations:
        lines.append("- No violations found.")
    else:
        lines.append("| Rank | Operation | Variables | Relative Error | Timestamp |")
        lines.append("| --- | --- | --- | --- | --- |")
        for idx, record in enumerate(violations, start=1):
            variables = record.get("variables") or []
            if isinstance(variables, list):
                var_text = ", ".join(str(v) for v in variables)
            else:
                var_text = str(variables)
            rel_error = record.get("relative_error", 0.0)
            timestamp = record.get("timestamp", "unknown")
            lines.append(f"| {idx} | {record.get('operation', 'unknown')} | {var_text} | {rel_error} | {timestamp} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate daily coupling audit report.")
    parser.add_argument("--input", required=True, help="Path to JSONL audit log.")
    parser.add_argument("--output", required=True, help="Path to write Markdown report.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = _load_records(input_path)
    report = _render_markdown(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
