"""Unified scientific audit gate regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from test.audit import unified_scientific_gate as gate
from test.audit.scientific_audit_scan import _mask_non_executable_text


def _dependency_report(
    *,
    passes: bool,
    mode: str = "project_environment",
    return_code: int | None = 1,
    error: str | None = None,
    dependencies_count: int = 12,
    unaccepted_count: int = 0,
    accepted_count: int = 1,
    policy_issues: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "mode": mode,
        "pass": passes,
        "policy_issues": policy_issues or [],
        "vulnerability_count": unaccepted_count,
        "accepted_vulnerability_count": accepted_count,
        "critical_input_missing": 0,
        "input_integrity_violations": 0,
        "critical_skip_without_policy": 0,
        "input_report": {
            "kind": "installed_project_runtime_closure",
            "direct_dependency_count": 1,
            "resolved_dependency_count": dependencies_count,
            "requirements_sha256": "A" * 64,
            "critical_packages_missing_from_declarations": [],
            "critical_packages_missing_from_resolution": [],
            "integrity_issues": [],
        },
        "audit_alignment_issues": [],
        "pip_audit": {
            "command": "python -m pip_audit",
            "return_code": return_code,
            "stderr": "",
            "error": error,
            "dependencies_count": dependencies_count,
            "dependencies_sha256": "A" * 64,
        },
    }


def _passing_metrics(baseline: dict[str, Any]) -> dict[str, int]:
    metrics = {key: 0 for key in baseline["thresholds"]}
    return metrics


def _static_report() -> dict[str, Any]:
    return {
        "files_scanned": 1,
        "functor_count": 1,
        "non_ascii_files": [],
        "forward_template_missing": [],
        "metrics": {
            "global_manager_calls_functor": 0,
            "global_manager_files_functor": 0,
            "non_ascii_files": 0,
            "forward_missing_template": 0,
        },
        "issues": {
            "functor_global_manager_calls": [],
            "removed_api_calls": [],
            "default_numeric_in_get_parameter_value": [],
            "state_get_default_numeric": [],
            "direct_state_assignment": [],
            "category_theory_imports": [],
        },
    }


def test_scientific_pattern_scan_ignores_docstrings_and_comments() -> None:
    source = (
        '"""Example: torch.tensor(298.15)."""\n'
        "# torch.tensor(273.15)\n"
        "value = torch.tensor(310.15)\n"
    )

    masked = _mask_non_executable_text(source)

    assert "torch.tensor(298.15)" not in masked
    assert "torch.tensor(273.15)" not in masked
    assert "torch.tensor(310.15)" in masked


def test_dependency_metrics_accept_only_a_completed_passing_scan() -> None:
    metrics = gate._dependency_gate_metrics(_dependency_report(passes=True))

    assert metrics == {
        "dependency.accepted_vulnerability_count": 1,
        "dependency.critical_input_missing": 0,
        "dependency.critical_skip_without_policy": 0,
        "dependency.input_integrity_violations": 0,
        "dependency.policy_issues": 0,
        "dependency.raw_vulnerability_count": 1,
        "dependency.report_pass_violation": 0,
        "dependency.scan_health_violation": 0,
        "dependency.vulnerability_count": 0,
    }


@pytest.mark.parametrize(
    "report",
    [
        _dependency_report(
            passes=False,
            mode="policy_only",
            return_code=None,
            dependencies_count=0,
            accepted_count=0,
        ),
        _dependency_report(
            passes=False,
            return_code=2,
            error="pip-audit failed",
            dependencies_count=0,
            accepted_count=0,
        ),
        _dependency_report(
            passes=False,
            return_code=0,
            dependencies_count=0,
            accepted_count=0,
        ),
    ],
)
def test_dependency_metrics_fail_closed_for_non_evidence(
    report: dict[str, Any],
) -> None:
    metrics = gate._dependency_gate_metrics(report)

    assert metrics["dependency.report_pass_violation"] == 1
    assert metrics["dependency.scan_health_violation"] == 1


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("policy_issues",), "expired"),
        (("input_report", "integrity_issues"), {}),
        (("critical_input_missing",), True),
        (("pip_audit", "return_code"), True),
        (("pip_audit", "dependencies_count"), True),
        (("accepted_vulnerability_count",), True),
    ],
)
def test_dependency_metrics_reject_malformed_report_fields(
    field_path: tuple[str, ...],
    invalid_value: object,
) -> None:
    report = _dependency_report(passes=True)
    target: dict[str, Any] = report
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = invalid_value

    metrics = gate._dependency_gate_metrics(report)

    assert metrics["dependency.report_pass_violation"] == 1
    assert metrics["dependency.scan_health_violation"] == 1


def test_static_metrics_are_consumed_from_the_canonical_report() -> None:
    metrics = gate._static_audit_gate_metrics(_static_report())

    assert metrics == {
        "source.forward_missing_template": 0,
        "source.non_ascii_files": 0,
        "static.category_theory_imports": 0,
        "static.default_numeric_in_get_parameter_value": 0,
        "static.direct_state_assignment": 0,
        "static.global_manager_calls": 0,
        "static.global_manager_files": 0,
        "static.removed_api_calls": 0,
        "static.report_schema_violation": 0,
        "static.state_get_default_numeric": 0,
    }


@pytest.mark.parametrize(
    ("section", "field", "invalid_value"),
    [
        ("metrics", "non_ascii_files", True),
        ("metrics", "forward_missing_template", "0"),
        ("issues", "removed_api_calls", {}),
    ],
)
def test_static_metrics_fail_closed_for_malformed_report(
    section: str,
    field: str,
    invalid_value: object,
) -> None:
    report = _static_report()
    report[section][field] = invalid_value

    metrics = gate._static_audit_gate_metrics(report)

    assert metrics["static.report_schema_violation"] == 1


def test_collect_metrics_does_not_scan_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = {
        "output_contract_scan.json": {"issue_count": 0},
        "output_overlap_scan.json": {
            "ungoverned_duplicate_count": 0,
            "effective_conflict_count": 0,
        },
        "scientific_static_audit.json": _static_report(),
        "audit_functor_runtime_scan.json": {
            "summary": {
                "metadata_issues": 0,
                "ungoverned_output_overlaps": 0,
                "effective_output_conflicts": 0,
            }
        },
        "scientific_audit_scan_20260213.json": {"python_reports": []},
        "radon_complexity_summary.json": {
            "metrics": {
                "complexity_gt15_flat": 0,
                "complexity_gt30_flat": 0,
            },
            "monitored_hotspots": {
                "relic\\core\\simulator\\execution_group.py::execute_group": {
                    "present": True,
                    "complexity": 1,
                },
                "relic\\core\\functor\\functor_base.py::get_parameter_value": {
                    "present": True,
                    "complexity": 1,
                },
            },
        },
        "dependency_audit_summary.json": _dependency_report(passes=True),
        "extension_validation_summary.json": {
            "targets": {"example": {"pass": True}},
            "total_passed": 1,
        },
        "mypy_core_budget_summary.json": {"budget_violation": 0},
    }

    monkeypatch.setattr(
        gate,
        "_read_json",
        lambda path: reports[path.name],
    )
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *args, **kwargs: pytest.fail("unified gate must not scan source files"),
    )

    metrics = gate._collect_metrics()

    assert metrics["static.report_schema_violation"] == 0
    assert metrics["dependency.scan_health_violation"] == 0


def test_evaluate_gate_rejects_failed_dependency_report() -> None:
    baseline = gate._read_json(gate.BASELINE_PATH)
    metrics = _passing_metrics(baseline)
    metrics["dependency.report_pass_violation"] = 1

    evaluation = gate._evaluate_gate(metrics, baseline)

    assert evaluation["pass"] is False
    assert {item["metric"] for item in evaluation["threshold_violations"]} == {
        "dependency.report_pass_violation"
    }


def test_run_unified_gate_is_a_pure_immutable_report_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = gate._read_json(gate.BASELINE_PATH)
    metrics = _passing_metrics(baseline)
    output_path = tmp_path / "unified_scientific_gate.json"
    read_paths: list[Path] = []

    def fake_read_json(path: Path) -> dict[str, Any]:
        read_paths.append(path)
        if path == gate.BASELINE_PATH:
            return baseline
        raise AssertionError(f"Unexpected direct read: {path}")

    monkeypatch.setattr(
        gate,
        "_run_python_script",
        lambda *args, **kwargs: pytest.fail("unified gate must not execute scanners"),
        raising=False,
    )
    monkeypatch.setattr(gate, "_read_json", fake_read_json)
    monkeypatch.setattr(gate, "_collect_metrics", lambda: metrics)
    monkeypatch.setattr(gate, "OUTPUT_PATH", output_path)

    result = gate.run_unified_gate()

    assert result["evaluation"]["pass"] is True
    assert result["evidence_mode"] == "immutable_reports"
    assert read_paths == [gate.BASELINE_PATH]
    assert output_path.is_file()
