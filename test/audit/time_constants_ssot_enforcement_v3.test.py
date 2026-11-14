"""Repository-wide audit for executable time-constant SSOT usage."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RELIC_ROOT = REPO_ROOT / "relic"
ARTIFACT_PATH = REPO_ROOT / "analysis_results" / "time_constants_ssot_audit.json"
FORBIDDEN_TIME_LITERALS = {
    3600.0,
    86400.0,
    31557600.0,
    365.0,
    365.25,
}

# Whitelist keys are (repository-relative path, nearest AST owner, numeric value).
# Counts are exact so a new literal in an approved owner cannot pass silently.
ALLOWED_LITERALS: dict[tuple[str, str, float], dict[str, Any]] = {
    (
        "relic/coupling/scales/temporal/core.py",
        "_datetime_to_day_index",
        365.0,
    ): {
        "count": 1,
        "category": "calendar_owner_math",
        "reason": "Canonical proleptic calendar ordinal arithmetic.",
    },
    (
        "relic/coupling/scales/temporal/core.py",
        "_raw_calendar_day_index",
        365.0,
    ): {
        "count": 1,
        "category": "calendar_owner_math",
        "reason": "Canonical Gregorian raw day-index arithmetic.",
    },
    (
        "relic/coupling/scales/temporal/core.py",
        "_day_index_to_datetime",
        365.0,
    ): {
        "count": 1,
        "category": "calendar_owner_math",
        "reason": "Canonical inverse Gregorian year-length arithmetic.",
    },
}


def _docstring_constant_nodes(tree: ast.AST) -> set[ast.Constant]:
    docstrings: set[ast.Constant] = set()
    for owner in ast.walk(tree):
        body = getattr(owner, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(first.value)
    return docstrings


def _nearest_owner(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return current.name
    return "<module>"


def _collect_time_literals() -> tuple[list[dict[str, Any]], str, int]:
    hits: list[dict[str, Any]] = []
    fingerprint = hashlib.sha256()
    files = sorted(RELIC_ROOT.rglob("*.py"))
    for path in files:
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        source_bytes = path.read_bytes()
        source = source_bytes.decode("utf-8-sig")
        fingerprint.update(relative_path.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(source_bytes)
        fingerprint.update(b"\0")
        tree = ast.parse(source, filename=relative_path)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        docstrings = _docstring_constant_nodes(tree)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Constant)
                or node in docstrings
                or isinstance(node.value, bool)
                or not isinstance(node.value, (int, float))
            ):
                continue
            numeric_value = float(node.value)
            if numeric_value not in FORBIDDEN_TIME_LITERALS:
                continue
            hits.append(
                {
                    "path": relative_path,
                    "line": node.lineno,
                    "owner": _nearest_owner(node, parents),
                    "value": numeric_value,
                    "source": lines[node.lineno - 1].strip(),
                }
            )
    hits.sort(key=lambda item: (item["path"], item["line"], item["value"]))
    return hits, fingerprint.hexdigest(), len(files)


def _generate_report() -> dict[str, Any]:
    hits, source_fingerprint, source_file_count = _collect_time_literals()
    observed_counts = Counter(
        (hit["path"], hit["owner"], hit["value"])
        for hit in hits
    )
    violations = [
        hit
        for hit in hits
        if (hit["path"], hit["owner"], hit["value"])
        not in ALLOWED_LITERALS
    ]
    whitelist_mismatches: list[dict[str, Any]] = []
    for key, specification in sorted(ALLOWED_LITERALS.items()):
        observed = observed_counts.get(key, 0)
        expected = int(specification["count"])
        if observed != expected:
            whitelist_mismatches.append(
                {
                    "path": key[0],
                    "owner": key[1],
                    "value": key[2],
                    "expected_count": expected,
                    "observed_count": observed,
                    "reason": specification["reason"],
                }
            )

    generator_relative = Path(__file__).resolve().relative_to(REPO_ROOT)
    report = {
        "schema_version": "1.0.0",
        "generator": generator_relative.as_posix(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_root": "relic",
        "source_file_count": source_file_count,
        "source_fingerprint_sha256": source_fingerprint,
        "forbidden_values": sorted(FORBIDDEN_TIME_LITERALS),
        "raw_hit_count": len(hits),
        "allowed_hit_count": len(hits) - len(violations),
        "violation_count": len(violations),
        "whitelist_mismatch_count": len(whitelist_mismatches),
        "passed": not violations and not whitelist_mismatches,
        "whitelist": [
            {
                "path": key[0],
                "owner": key[1],
                "value": key[2],
                **specification,
            }
            for key, specification in sorted(ALLOWED_LITERALS.items())
        ],
        "hits": hits,
        "violations": violations,
        "whitelist_mismatches": whitelist_mismatches,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def test_time_constants_use_ssot_across_relic_executable_code() -> None:
    """Reject unapproved executable time literals across the RELIC package."""
    report = _generate_report()
    diagnostics = [
        f"{item['path']}:{item['line']} owner={item['owner']} value={item['value']}"
        for item in report["violations"]
    ]
    diagnostics.extend(
        "whitelist mismatch "
        f"{item['path']} owner={item['owner']} value={item['value']} "
        f"expected={item['expected_count']} observed={item['observed_count']}"
        for item in report["whitelist_mismatches"]
    )
    assert report["passed"], (
        "Time-constant SSOT audit failed; see "
        f"{ARTIFACT_PATH.relative_to(REPO_ROOT).as_posix()}:\n"
        + "\n".join(diagnostics)
    )
