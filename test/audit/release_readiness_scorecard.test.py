from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from test.audit import release_readiness_scorecard as scorecard


def _artifacts(
    *,
    unified_pass: bool = True,
    dependency_pass: bool = True,
    scan_health_violation: int = 0,
    accepted_vulnerabilities: int = 0,
    unaccepted_vulnerabilities: int = 0,
    policy_issues: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "unified_gate": {
            "evidence_mode": "immutable_reports",
            "metrics": {
                "dependency.scan_health_violation": scan_health_violation,
                "output_contract.issue_count": 0,
            },
            "evaluation": {
                "pass": unified_pass,
                "threshold_violations": [],
                "consistency_violations": [],
            },
        },
        "static_audit": {"metrics": {"global_manager_calls_functor": 0}},
        "radon_summary": {
            "metrics": {"complexity_gt30_flat": 0},
            "top_hotspots_flat": [],
        },
        "parameter_reference_index": {"orphan_candidates": []},
        "dependency_audit": {
            "pass": dependency_pass,
            "vulnerability_count": unaccepted_vulnerabilities,
            "accepted_vulnerability_count": accepted_vulnerabilities,
            "critical_skip_without_policy": 0,
            "policy_issues": policy_issues or [],
        },
        "extension_validation": {"pass": True},
    }


def test_release_readiness_scorecard_schema_and_bounds() -> None:
    payload = scorecard._build_scorecard(
        _artifacts(),
        generated_at_utc="2026-07-30T00:00:00+00:00",
    )

    assert payload["schema_version"] == "1.0.0"
    assert payload["scoring_policy_version"] == "2026-07-30-display-only"
    assert payload["role"] == "display_only_downstream_consumer"
    assert payload["generated_at_utc"] == "2026-07-30T00:00:00+00:00"

    dimensions = payload["dimensions"]
    assert len(dimensions) == 7
    for value in dimensions.values():
        assert 0.0 <= float(value) <= 5.0

    decision = payload["decision"]
    assert decision["status"] in {"blocked", "ready", "ready_with_debt", "not_ready"}

    expected_overall = round(sum(float(v) for v in dimensions.values()) / 7.0, 2)
    assert float(decision["overall_score"]) == expected_overall


def test_missing_gate_and_dependency_reports_fail_closed() -> None:
    artifacts = _artifacts()
    artifacts["unified_gate"] = {}
    artifacts["dependency_audit"] = {}

    decision = scorecard._derive_decision(
        artifacts,
        scorecard._derive_dimensions(artifacts),
    )

    assert decision["status"] == "blocked"
    assert set(decision["blockers"]) == {
        "unified_gate_evidence_missing",
        "dependency_audit_evidence_missing",
    }


@pytest.mark.parametrize("report_name", list(scorecard.INPUT_FILES))
@pytest.mark.parametrize("invalid_report", [{}, []])
def test_each_required_report_is_fail_closed(
    report_name: str,
    invalid_report: object,
) -> None:
    artifacts: dict[str, Any] = _artifacts()
    artifacts[report_name] = invalid_report

    decision = scorecard._derive_decision(
        artifacts,
        scorecard._derive_dimensions(artifacts),
    )

    assert decision["status"] == "blocked"
    assert f"{report_name}_evidence_missing" in decision["blockers"]


def test_invalid_unified_and_dependency_evidence_are_blockers() -> None:
    artifacts = _artifacts()
    artifacts["unified_gate"]["metrics"] = {}
    artifacts["dependency_audit"]["pass"] = "yes"

    decision = scorecard._derive_decision(
        artifacts,
        scorecard._derive_dimensions(artifacts),
    )

    assert decision["status"] == "blocked"
    assert "unified_gate_evidence_invalid" in decision["blockers"]
    assert "dependency_audit_evidence_invalid" in decision["blockers"]


@pytest.mark.parametrize("invalid_value", [True, "0", -1, 2])
def test_invalid_scan_health_metric_is_fail_closed(
    invalid_value: object,
) -> None:
    artifacts = _artifacts()
    artifacts["unified_gate"]["metrics"]["dependency.scan_health_violation"] = invalid_value

    decision = scorecard._derive_decision(
        artifacts,
        scorecard._derive_dimensions(artifacts),
    )

    assert decision["status"] == "blocked"
    assert "unified_gate_evidence_invalid" in decision["blockers"]


def test_dependency_failure_and_unhealthy_scan_are_visible_blockers() -> None:
    artifacts = _artifacts(
        dependency_pass=False,
        scan_health_violation=1,
    )

    decision = scorecard._derive_decision(
        artifacts,
        scorecard._derive_dimensions(artifacts),
    )

    assert decision["status"] == "blocked"
    assert "dependency_audit_failed" in decision["blockers"]
    assert "dependency_scan_unhealthy" in decision["blockers"]


def test_security_score_discloses_accepted_and_policy_debt() -> None:
    clean_score = scorecard._derive_dimensions(_artifacts())["D4_security_compliance"]
    debt_score = scorecard._derive_dimensions(
        _artifacts(
            dependency_pass=False,
            accepted_vulnerabilities=2,
            unaccepted_vulnerabilities=1,
            policy_issues=["expired review"],
        )
    )["D4_security_compliance"]

    assert debt_score < clean_score


def test_main_writes_blocked_scorecard_but_returns_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(dependency_pass=False)
    output_path = tmp_path / "release_readiness_scorecard.json"

    monkeypatch.setattr(
        scorecard,
        "_read_json",
        lambda path: artifacts[
            next(name for name, input_path in scorecard.INPUT_FILES.items() if input_path == path)
        ],
    )
    monkeypatch.setattr(scorecard, "OUTPUT_PATH", output_path)

    assert scorecard.main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["decision"]["status"] == "blocked"
    assert "dependency_audit_failed" in payload["decision"]["blockers"]
