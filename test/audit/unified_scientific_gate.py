"""Unified audit gate runner for high-risk scientific governance checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ANALYSIS_DIR = REPO_ROOT / "analysis_results"
BASELINE_PATH = REPO_ROOT / "test" / "audit" / "scientific_audit_baseline.json"
OUTPUT_PATH = ANALYSIS_DIR / "unified_scientific_gate.json"


def _resolve_relative_existing(*relative_paths: str) -> str:
    """Return the first existing repo-relative path, fallback to the first candidate."""
    for rel in relative_paths:
        if (REPO_ROOT / rel).exists():
            return rel
    return relative_paths[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _dependency_gate_metrics(
    dependency_summary: dict[str, Any],
) -> dict[str, int]:
    """Return fail-closed dependency metrics from an immutable audit report."""
    pip_audit_value = dependency_summary.get("pip_audit")
    pip_audit = pip_audit_value if isinstance(pip_audit_value, dict) else {}
    input_report_value = dependency_summary.get("input_report")
    input_report = input_report_value if isinstance(input_report_value, dict) else {}
    policy_issues_value = dependency_summary.get("policy_issues")
    policy_issues_valid = isinstance(policy_issues_value, list) and all(
        isinstance(item, str) for item in policy_issues_value
    )
    policy_issues = policy_issues_value if policy_issues_valid else []
    input_issues_value = input_report.get("integrity_issues")
    input_issues_valid = isinstance(input_issues_value, list) and all(
        isinstance(item, str) for item in input_issues_value
    )
    input_issues = input_issues_value if input_issues_valid else []
    alignment_issues_value = dependency_summary.get("audit_alignment_issues")
    alignment_issues_valid = isinstance(alignment_issues_value, list) and all(
        isinstance(item, str) for item in alignment_issues_value
    )
    alignment_issues = alignment_issues_value if alignment_issues_valid else []
    unaccepted_value = dependency_summary.get("vulnerability_count")
    accepted_value = dependency_summary.get("accepted_vulnerability_count")
    critical_input_value = dependency_summary.get("critical_input_missing")
    input_integrity_value = dependency_summary.get("input_integrity_violations")
    critical_skip_value = dependency_summary.get("critical_skip_without_policy")
    unaccepted_valid = _is_nonnegative_int(unaccepted_value)
    accepted_valid = _is_nonnegative_int(accepted_value)
    critical_input_valid = _is_nonnegative_int(critical_input_value)
    input_integrity_valid = _is_nonnegative_int(input_integrity_value)
    critical_skip_valid = _is_nonnegative_int(critical_skip_value)
    unaccepted_count = unaccepted_value if unaccepted_valid else 0
    accepted_count = accepted_value if accepted_valid else 0
    critical_input_count = critical_input_value if critical_input_valid else 0
    input_integrity_count = input_integrity_value if input_integrity_valid else 0
    critical_skip_count = critical_skip_value if critical_skip_valid else 0
    raw_vulnerability_count = accepted_count + unaccepted_count
    mode = dependency_summary.get("mode")
    return_code = pip_audit.get("return_code")
    dependencies_count_value = pip_audit.get("dependencies_count")
    return_code_valid = type(return_code) is int and return_code in {0, 1}
    dependencies_count_valid = (
        type(dependencies_count_value) is int and dependencies_count_value > 0
    )
    error = pip_audit.get("error")
    command = pip_audit.get("command")
    stderr = pip_audit.get("stderr")
    direct_count = input_report.get("direct_dependency_count")
    resolved_count = input_report.get("resolved_dependency_count")
    requirements_sha256 = input_report.get("requirements_sha256")
    dependencies_sha256 = pip_audit.get("dependencies_sha256")
    sha_valid = (
        isinstance(requirements_sha256, str)
        and len(requirements_sha256) == 64
        and isinstance(dependencies_sha256, str)
        and dependencies_sha256 == requirements_sha256
    )
    issue_count_matches = input_integrity_count == (
        len(input_issues) + len(alignment_issues)
    )

    schema_healthy = (
        dependency_summary.get("schema_version") == "2.0.0"
        and isinstance(pip_audit_value, dict)
        and isinstance(input_report_value, dict)
        and input_report.get("kind") == "installed_project_runtime_closure"
        and _is_nonnegative_int(direct_count)
        and direct_count > 0
        and _is_nonnegative_int(resolved_count)
        and resolved_count > 0
        and input_issues_valid
        and alignment_issues_valid
        and policy_issues_valid
        and unaccepted_valid
        and accepted_valid
        and critical_input_valid
        and input_integrity_valid
        and critical_skip_valid
        and isinstance(dependency_summary.get("pass"), bool)
        and mode == "project_environment"
        and return_code_valid
        and dependencies_count_valid
        and dependencies_count_value == resolved_count
        and sha_valid
        and issue_count_matches
        and isinstance(command, str)
        and bool(command.strip())
        and isinstance(stderr, str)
        and "error" in pip_audit
    )

    return_code_matches_payload = (return_code == 0 and raw_vulnerability_count == 0) or (
        return_code == 1 and raw_vulnerability_count > 0
    )
    scan_healthy = schema_healthy and error is None and return_code_matches_payload

    return {
        "dependency.accepted_vulnerability_count": accepted_count,
        "dependency.critical_input_missing": critical_input_count,
        "dependency.critical_skip_without_policy": critical_skip_count,
        "dependency.input_integrity_violations": input_integrity_count,
        "dependency.policy_issues": len(policy_issues),
        "dependency.raw_vulnerability_count": raw_vulnerability_count,
        "dependency.report_pass_violation": (
            0 if schema_healthy and dependency_summary.get("pass") is True else 1
        ),
        "dependency.scan_health_violation": 0 if scan_healthy else 1,
        "dependency.vulnerability_count": unaccepted_count,
    }


def _static_audit_gate_metrics(
    static_audit: dict[str, Any],
) -> dict[str, int]:
    """Return source-policy metrics owned by the canonical static report."""
    metrics_value = static_audit.get("metrics")
    issues_value = static_audit.get("issues")
    metrics = metrics_value if isinstance(metrics_value, dict) else {}
    issues = issues_value if isinstance(issues_value, dict) else {}

    metric_fields = (
        "global_manager_calls_functor",
        "global_manager_files_functor",
        "non_ascii_files",
        "forward_missing_template",
    )
    issue_fields = (
        "functor_global_manager_calls",
        "removed_api_calls",
        "default_numeric_in_get_parameter_value",
        "state_get_default_numeric",
        "direct_state_assignment",
        "category_theory_imports",
    )
    metric_fields_valid = all(_is_nonnegative_int(metrics.get(field)) for field in metric_fields)
    issue_fields_valid = all(isinstance(issues.get(field), list) for field in issue_fields)
    files_scanned = static_audit.get("files_scanned")
    functor_count = static_audit.get("functor_count")
    non_ascii_files = static_audit.get("non_ascii_files")
    forward_template_missing = static_audit.get("forward_template_missing")
    report_lists_valid = isinstance(non_ascii_files, list) and isinstance(
        forward_template_missing, list
    )

    global_manager_records = issues.get("functor_global_manager_calls", [])
    global_manager_files = {
        item.get("file")
        for item in global_manager_records
        if isinstance(item, dict) and isinstance(item.get("file"), str)
    }
    count_consistent = (
        metric_fields_valid
        and issue_fields_valid
        and report_lists_valid
        and metrics.get("global_manager_calls_functor") == len(global_manager_records)
        and metrics.get("global_manager_files_functor") == len(global_manager_files)
        and metrics.get("non_ascii_files") == len(non_ascii_files)
        and metrics.get("forward_missing_template") == len(forward_template_missing)
    )
    schema_healthy = (
        isinstance(metrics_value, dict)
        and isinstance(issues_value, dict)
        and type(files_scanned) is int
        and files_scanned > 0
        and _is_nonnegative_int(functor_count)
        and count_consistent
    )

    def metric_value(field: str) -> int:
        value = metrics.get(field)
        return value if _is_nonnegative_int(value) else 0

    def issue_count(field: str) -> int:
        value = issues.get(field)
        return len(value) if isinstance(value, list) else 0

    return {
        "source.forward_missing_template": metric_value("forward_missing_template"),
        "source.non_ascii_files": metric_value("non_ascii_files"),
        "static.category_theory_imports": issue_count("category_theory_imports"),
        "static.default_numeric_in_get_parameter_value": issue_count(
            "default_numeric_in_get_parameter_value"
        ),
        "static.direct_state_assignment": issue_count("direct_state_assignment"),
        "static.global_manager_calls": metric_value("global_manager_calls_functor"),
        "static.global_manager_files": metric_value("global_manager_files_functor"),
        "static.removed_api_calls": issue_count("removed_api_calls"),
        "static.report_schema_violation": 0 if schema_healthy else 1,
        "static.state_get_default_numeric": issue_count("state_get_default_numeric"),
    }


def _collect_metrics() -> dict[str, Any]:
    output_contract = _read_json(ANALYSIS_DIR / "output_contract_scan.json")
    output_overlap = _read_json(ANALYSIS_DIR / "output_overlap_scan.json")
    static_audit = _read_json(ANALYSIS_DIR / "scientific_static_audit.json")
    runtime_scan_path = REPO_ROOT / _resolve_relative_existing(
        "test/audit_functor_runtime_scan.json",
        "test/reports/audit_functor_runtime_scan.json",
    )
    runtime_scan = _read_json(runtime_scan_path)
    scientific_scan = _read_json(ANALYSIS_DIR / "scientific_audit_scan_20260213.json")
    radon_summary = _read_json(ANALYSIS_DIR / "radon_complexity_summary.json")
    dependency_summary = _read_json(ANALYSIS_DIR / "dependency_audit_summary.json")
    extension_summary = _read_json(ANALYSIS_DIR / "extension_validation_summary.json")
    mypy_budget = _read_json(ANALYSIS_DIR / "mypy_core_budget_summary.json")

    runtime_summary = runtime_scan.get("summary", {})
    python_reports = scientific_scan.get("python_reports", [])
    radon_metrics = radon_summary.get("metrics", {})
    radon_monitored = radon_summary.get("monitored_hotspots", {})
    extension_targets = extension_summary.get("targets", {})

    hardcoded_pattern_hits = 0
    for report in python_reports:
        hardcoded_pattern_hits += len(report.get("hardcoded_patterns", []))

    extension_failures = 0
    for target in extension_targets.values():
        if isinstance(target, dict) and not target.get("pass", False):
            extension_failures += 1

    extension_total_passed = int(extension_summary.get("total_passed", 0) or 0)
    extension_total_passed_min_violation = 1 if extension_total_passed <= 0 else 0

    monitored_missing = 0
    for label in (
        "relic\\core\\simulator\\execution_group.py::execute_group",
        "relic\\core\\functor\\functor_base.py::get_parameter_value",
    ):
        spot = radon_monitored.get(label, {})
        if not isinstance(spot, dict) or not spot.get("present", False):
            monitored_missing += 1

    execute_group_complexity = int(
        radon_monitored.get(
            "relic\\core\\simulator\\execution_group.py::execute_group",
            {},
        ).get("complexity")
        or 0
    )
    get_parameter_value_complexity = int(
        radon_monitored.get(
            "relic\\core\\functor\\functor_base.py::get_parameter_value",
            {},
        ).get("complexity")
        or 0
    )

    metrics = {
        "output_contract.issue_count": int(output_contract.get("issue_count", 0)),
        "output_overlap.ungoverned_duplicate_count": int(
            output_overlap.get("ungoverned_duplicate_count", 0)
        ),
        "output_overlap.effective_conflict_count": int(
            output_overlap.get("effective_conflict_count", 0)
        ),
        "scientific_audit_scan.hardcoded_pattern_hits": hardcoded_pattern_hits,
        "runtime.metadata_issues": int(runtime_summary.get("metadata_issues", 0)),
        "runtime.ungoverned_output_overlaps": int(
            runtime_summary.get("ungoverned_output_overlaps", 0)
        ),
        "runtime.effective_output_conflicts": int(
            runtime_summary.get("effective_output_conflicts", 0)
        ),
        "radon.complexity_gt15_flat": int(radon_metrics.get("complexity_gt15_flat", 0)),
        "radon.complexity_gt30_flat": int(radon_metrics.get("complexity_gt30_flat", 0)),
        "radon.hotspot_execute_group": execute_group_complexity,
        "radon.hotspot_get_parameter_value": get_parameter_value_complexity,
        "radon.monitored_hotspots_missing": monitored_missing,
        "extension.validation_failures": extension_failures,
        "extension.total_passed_min_violation": extension_total_passed_min_violation,
        "typing.mypy_core_budget_violation": int(mypy_budget.get("budget_violation", 1)),
    }
    metrics.update(_static_audit_gate_metrics(static_audit))
    metrics.update(_dependency_gate_metrics(dependency_summary))
    return metrics


def _evaluate_gate(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    threshold_violations = []
    thresholds = baseline.get("thresholds", {})
    for key, rule in thresholds.items():
        if key not in metrics:
            threshold_violations.append(
                {"type": "missing_metric", "metric": key, "message": "Metric not collected"}
            )
            continue
        max_value = rule.get("max")
        value = metrics[key]
        if max_value is not None and value > max_value:
            threshold_violations.append(
                {
                    "type": "threshold",
                    "metric": key,
                    "value": value,
                    "max": max_value,
                }
            )

    consistency_violations = []
    for rule in baseline.get("consistency_rules", []):
        left = rule.get("left")
        right = rule.get("right")
        op = rule.get("op")
        if left not in metrics or right not in metrics:
            consistency_violations.append(
                {
                    "type": "missing_metric",
                    "left": left,
                    "right": right,
                    "message": "Consistency metric not collected",
                }
            )
            continue
        if op == "==" and metrics[left] != metrics[right]:
            consistency_violations.append(
                {
                    "type": "consistency",
                    "left": left,
                    "right": right,
                    "left_value": metrics[left],
                    "right_value": metrics[right],
                }
            )
        elif op != "==":
            consistency_violations.append(
                {"type": "unsupported_operator", "operator": op, "left": left, "right": right}
            )

    passed = not threshold_violations and not consistency_violations
    return {
        "pass": passed,
        "threshold_violations": threshold_violations,
        "consistency_violations": consistency_violations,
    }


def run_unified_gate() -> dict[str, Any]:
    """Validate immutable scanner reports without mutating their evidence."""
    baseline = _read_json(BASELINE_PATH)
    metrics = _collect_metrics()
    evaluation = _evaluate_gate(metrics, baseline)
    result = {
        "baseline_version": baseline.get("version"),
        "evidence_mode": "immutable_reports",
        "metrics": metrics,
        "evaluation": evaluation,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    result = run_unified_gate()
    if result["evaluation"]["pass"]:
        print("[OK] unified scientific audit gate passed")
        return 0
    print("[ERROR] unified scientific audit gate failed")
    print(json.dumps(result["evaluation"], indent=2, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
