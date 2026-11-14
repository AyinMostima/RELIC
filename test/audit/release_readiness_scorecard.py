"""Render a downstream release-readiness scorecard from immutable audit reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "analysis_results"
OUTPUT_PATH = ANALYSIS_ROOT / "release_readiness_scorecard.json"

INPUT_FILES = {
    "unified_gate": ANALYSIS_ROOT / "unified_scientific_gate.json",
    "static_audit": ANALYSIS_ROOT / "scientific_static_audit.json",
    "radon_summary": ANALYSIS_ROOT / "radon_complexity_summary.json",
    "parameter_reference_index": ANALYSIS_ROOT / "parameter_reference_index.json",
    "dependency_audit": ANALYSIS_ROOT / "dependency_audit_summary.json",
    "extension_validation": ANALYSIS_ROOT / "extension_validation_summary.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")
    for encoding in encodings:
        try:
            payload = json.loads(path.read_text(encoding=encoding))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _is_binary_int(value: Any) -> bool:
    return type(value) is int and value in {0, 1}


def _clip(score: float) -> float:
    return max(0.0, min(5.0, round(score, 2)))


def _bool_to_bonus(value: bool, yes: float, no: float) -> float:
    return yes if value else no


def _derive_dimensions(data: dict[str, dict[str, Any]]) -> dict[str, float]:
    unified = _mapping(data.get("unified_gate"))
    evaluation = _mapping(unified.get("evaluation"))
    metrics = _mapping(unified.get("metrics"))
    gate_pass = evaluation.get("pass") is True

    static_audit = _mapping(data.get("static_audit"))
    static_metrics = _mapping(static_audit.get("metrics"))

    radon = _mapping(data.get("radon_summary"))
    radon_metrics = _mapping(radon.get("metrics"))
    gt30_flat = _safe_int(radon_metrics.get("complexity_gt30_flat"))
    top_hotspots = radon.get("top_hotspots_flat", [])
    if isinstance(top_hotspots, list) and top_hotspots:
        top_complexity = _safe_int(_mapping(top_hotspots[0]).get("complexity"))
    else:
        top_complexity = 0

    parameter_index = _mapping(data.get("parameter_reference_index"))
    orphan_candidates = parameter_index.get("orphan_candidates", [])
    if isinstance(orphan_candidates, list):
        orphan_count = len(orphan_candidates)
        possible_dynamic = sum(
            1
            for item in orphan_candidates
            if _mapping(item).get("classification") == "possible_dynamic_reference"
        )
    else:
        orphan_count = _safe_int(orphan_candidates)
        possible_dynamic = 0
    dynamic_ratio = (possible_dynamic / orphan_count) if orphan_count > 0 else 0.0

    dependency = _mapping(data.get("dependency_audit"))
    vulnerabilities = _safe_int(dependency.get("vulnerability_count"))
    accepted_vulnerabilities = _safe_int(dependency.get("accepted_vulnerability_count"))
    unresolved_skips = _safe_int(dependency.get("critical_skip_without_policy"))
    policy_issues = _list_count(dependency.get("policy_issues"))
    dependency_pass_violation = 0 if dependency.get("pass") is True else 1
    scan_health_value = metrics.get("dependency.scan_health_violation")
    scan_health_violation = scan_health_value if _is_binary_int(scan_health_value) else 1

    extension = _mapping(data.get("extension_validation"))
    extension_pass = extension.get("pass") is True

    functor_global_manager_calls = _safe_int(static_metrics.get("global_manager_calls_functor"))
    output_contract_issues = _safe_int(metrics.get("output_contract.issue_count"))

    d1 = _clip(
        2.0
        + _bool_to_bonus(gate_pass, 2.0, 0.0)
        + _bool_to_bonus(extension_pass, 1.0, 0.2)
        - min(output_contract_issues * 1.0, 2.0)
    )
    d2 = _clip(4.2 - min(gt30_flat / 40.0, 2.2) - min(top_complexity / 300.0, 1.0))
    d3 = _clip(
        2.2
        + _bool_to_bonus(gate_pass, 2.0, 0.0)
        - min(functor_global_manager_calls * 0.5, 1.5)
        - min(output_contract_issues * 1.0, 2.0)
    )
    d4 = _clip(
        4.4
        - min(vulnerabilities * 0.3, 2.5)
        - min(accepted_vulnerabilities * 0.1, 0.5)
        - min(policy_issues * 0.15, 1.5)
        - min(unresolved_skips * 0.8, 2.5)
        - min(dependency_pass_violation * 0.8, 0.8)
        - min(scan_health_violation * 1.2, 1.2)
    )
    d5 = _clip(
        3.8
        + _bool_to_bonus(functor_global_manager_calls == 0, 0.6, -1.0)
        - min(gt30_flat / 45.0, 1.8)
    )
    d6 = _clip(4.0 - min(top_complexity / 80.0, 3.0) - min(gt30_flat / 30.0, 1.0))
    d7 = _clip(3.4 + _bool_to_bonus(extension_pass, 1.1, 0.2) + min(dynamic_ratio, 0.5))

    return {
        "D1_completeness_edge_cases": d1,
        "D2_performance_scalability": d2,
        "D3_reliability_fault_tolerance": d3,
        "D4_security_compliance": d4,
        "D5_maintainability_observability": d5,
        "D6_cost_complexity": d6,
        "D7_extensibility": d7,
    }


def _derive_decision(
    data: dict[str, dict[str, Any]],
    dimensions: dict[str, float],
) -> dict[str, Any]:
    unified = _mapping(data.get("unified_gate"))
    evaluation = _mapping(unified.get("evaluation"))
    metrics = _mapping(unified.get("metrics"))
    gate_pass = evaluation.get("pass") is True
    threshold_violations = evaluation.get("threshold_violations", [])
    consistency_violations = evaluation.get("consistency_violations", [])
    dependency = _mapping(data.get("dependency_audit"))

    blockers = []
    for name in INPUT_FILES:
        report = data.get(name)
        if not isinstance(report, dict) or not report:
            blockers.append(f"{name}_evidence_missing")

    unified_evidence_valid = (
        bool(unified)
        and unified.get("evidence_mode") == "immutable_reports"
        and isinstance(unified.get("metrics"), dict)
        and isinstance(unified.get("evaluation"), dict)
        and isinstance(evaluation.get("pass"), bool)
        and isinstance(threshold_violations, list)
        and isinstance(consistency_violations, list)
        and _is_binary_int(metrics.get("dependency.scan_health_violation"))
    )
    if unified and not unified_evidence_valid:
        blockers.append("unified_gate_evidence_invalid")
    elif unified_evidence_valid and not gate_pass:
        blockers.append("unified_gate_failed")
    if unified_evidence_valid and threshold_violations:
        blockers.append("threshold_violations_present")
    if unified_evidence_valid and consistency_violations:
        blockers.append("consistency_violations_present")

    dependency_evidence_valid = bool(dependency) and isinstance(dependency.get("pass"), bool)
    if dependency and not dependency_evidence_valid:
        blockers.append("dependency_audit_evidence_invalid")
    elif dependency_evidence_valid and dependency.get("pass") is not True:
        blockers.append("dependency_audit_failed")

    if unified_evidence_valid and metrics.get("dependency.scan_health_violation") == 1:
        blockers.append("dependency_scan_unhealthy")

    overall = round(sum(dimensions.values()) / len(dimensions), 2)
    if blockers:
        status = "blocked"
    elif overall >= 4.0:
        status = "ready"
    elif overall >= 3.0:
        status = "ready_with_debt"
    else:
        status = "not_ready"

    return {
        "overall_score": overall,
        "status": status,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _build_scorecard(
    artifacts: dict[str, dict[str, Any]],
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    dimensions = _derive_dimensions(artifacts)
    decision = _derive_decision(artifacts, dimensions)
    return {
        "schema_version": "1.0.0",
        "scoring_policy_version": "2026-07-30-display-only",
        "role": "display_only_downstream_consumer",
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: str(path.relative_to(REPO_ROOT)).replace("/", "\\")
            for name, path in INPUT_FILES.items()
        },
        "dimensions": dimensions,
        "decision": decision,
    }


def main() -> int:
    artifacts = {name: _read_json(path) for name, path in INPUT_FILES.items()}
    result = _build_scorecard(artifacts)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[OK] scorecard={OUTPUT_PATH}")
    print(
        "[INFO] overall={overall} status={status}".format(
            overall=result["decision"]["overall_score"],
            status=result["decision"]["status"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
