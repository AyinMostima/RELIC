from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "test" / "audit" / "dependency_audit_summary.py"
MODULE_NAME = "dependency_audit_summary_owner"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load dependency audit owner: {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def _policy(
    *,
    package_review_date: str = "2026-12-31",
    vulnerability_review_date: str = "2026-12-31",
    vulnerability_id: str = "PYSEC-TEST-1",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "entries": {
            "torch": {
                "reason": "Validated CUDA runtime constraint.",
                "mitigation": "Use trusted model artifacts only.",
                "owner": "platform-security",
                "review_date": package_review_date,
                "vulnerabilities": {
                    vulnerability_id: {
                        "reason": "Upgrade is pending validation.",
                        "mitigation": "Reject untrusted inputs.",
                        "owner": "platform-security",
                        "review_date": vulnerability_review_date,
                    }
                },
            }
        },
    }


def _input_report() -> dict[str, Any]:
    requirements = "sympy==1.13.3\ntorch==2.13.0\n"
    return {
        "kind": "installed_project_runtime_closure",
        "project_metadata_path": "pyproject.toml",
        "project_metadata_sha256": "A" * 64,
        "python_executable": r"python",
        "environment_prefix": r"<CONDA_ENV_RELIC>",
        "direct_dependency_count": 1,
        "direct_dependencies": ["torch"],
        "resolved_dependency_count": 2,
        "packages": [
            {
                "name": "sympy",
                "installed_version": "1.13.3",
                "audit_version": "1.13.3",
            },
            {
                "name": "torch",
                "installed_version": "2.13.0+cu130",
                "audit_version": "2.13.0",
            },
        ],
        "requirements_text": requirements,
        "requirements_sha256": hashlib.sha256(requirements.encode("utf-8")).hexdigest().upper(),
        "critical_packages": ["torch"],
        "critical_packages_missing_from_declarations": [],
        "critical_packages_missing_from_resolution": [],
        "integrity_issues": [],
    }


def _dependency_payload(
    *,
    return_code: int = 0,
    vulnerability_id: str | None = None,
    aliases: list[str] | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    torch_vulnerabilities = []
    if vulnerability_id is not None:
        torch_vulnerabilities.append({"id": vulnerability_id, "aliases": aliases or []})
    torch: dict[str, Any] = {
        "name": "torch",
        "version": "2.13.0",
        "vulns": torch_vulnerabilities,
    }
    if skip_reason is not None:
        torch["skip_reason"] = skip_reason
    return {
        "command": "python -m pip_audit --no-deps --disable-pip -r {temporary_project_closure}",
        "return_code": return_code,
        "stderr": "",
        "error": None,
        "dependencies": [
            {"name": "sympy", "version": "1.13.3", "vulns": []},
            torch,
        ],
    }


def _configure_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    policy: dict[str, Any] | None = None,
) -> None:
    manifest_path = tmp_path / "dependency_audit_manifest.json"
    policy_path = tmp_path / "dependency_skip_allowlist.json"
    project_path = tmp_path / "pyproject.toml"
    output_path = tmp_path / "reports" / "dependency_audit_summary.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "audit_scope": "installed_project_runtime_closure",
                "project_metadata_path": "pyproject.toml",
                "critical_packages": ["torch"],
            }
        ),
        encoding="utf-8",
    )
    policy_path.write_text(json.dumps(policy or _policy()), encoding="utf-8")
    project_path.write_text(
        '[project]\nname = "fixture"\nversion = "1.0.0"\ndependencies = ["torch>=2.0"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "AUDIT_MANIFEST_PATH", manifest_path, raising=False)
    monkeypatch.setattr(MODULE, "SKIP_POLICY_PATH", policy_path)
    monkeypatch.setattr(MODULE, "PROJECT_METADATA_PATH", project_path, raising=False)
    monkeypatch.setattr(MODULE, "OUTPUT_PATH", output_path)


def _completed_process(
    *,
    return_code: int,
    stdout: str,
    stderr: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(returncode=return_code, stdout=stdout, stderr=stderr)


class _Distribution:
    def __init__(self, version: str, requirements: list[str] | None = None) -> None:
        self.version = version
        self.requires = requirements or []


def test_project_closure_resolves_extras_markers_and_local_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_inputs(monkeypatch, tmp_path)
    MODULE.PROJECT_METADATA_PATH.write_text(
        (
            '[project]\nname = "fixture"\nversion = "1.0.0"\n'
            'dependencies = ["torch>=2.0", "dask[complete]>=2026.1"]\n'
        ),
        encoding="utf-8",
    )
    distributions = {
        "torch": _Distribution("2.13.0+cu130", ["sympy>=1.13"]),
        "sympy": _Distribution("1.13.3"),
        "dask": _Distribution(
            "2026.8.0",
            ['distributed==2026.8.0; extra == "complete"'],
        ),
        "distributed": _Distribution("2026.8.0"),
    }
    monkeypatch.setattr(
        MODULE.metadata,
        "distribution",
        lambda name: distributions[name],
        raising=False,
    )

    report = MODULE._collect_project_environment_input()

    assert report["direct_dependencies"] == ["dask", "torch"]
    assert report["resolved_dependency_count"] == 4
    packages = {item["name"]: item for item in report["packages"]}
    assert packages["torch"] == {
        "name": "torch",
        "installed_version": "2.13.0+cu130",
        "audit_version": "2.13.0",
    }
    assert report["critical_packages_missing_from_declarations"] == []
    assert report["critical_packages_missing_from_resolution"] == []
    assert report["integrity_issues"] == []


@pytest.mark.parametrize(
    ("torch_version", "missing_distribution", "expected_issue"),
    [
        ("1.9.0", None, "installed version violates declaration"),
        ("2.13.0+cu130", "sympy", "missing installed distribution: sympy"),
    ],
)
def test_project_closure_fails_closed_on_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    torch_version: str,
    missing_distribution: str | None,
    expected_issue: str,
) -> None:
    _configure_inputs(monkeypatch, tmp_path)
    distributions = {
        "torch": _Distribution(torch_version, ["sympy>=1.13"]),
        "sympy": _Distribution("1.13.3"),
    }

    def find_distribution(name: str):
        if name == missing_distribution:
            raise MODULE.metadata.PackageNotFoundError(name)
        return distributions[name]

    monkeypatch.setattr(MODULE.metadata, "distribution", find_distribution, raising=False)

    report = MODULE._collect_project_environment_input()

    assert any(expected_issue in issue for issue in report["integrity_issues"])


def test_pip_audit_uses_and_removes_temporary_project_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    payload = {"dependencies": [{"name": "torch", "version": "2.13.0", "vulns": []}]}

    def run(command, **kwargs):
        requirements_path = Path(command[-1])
        observed["command"] = command
        observed["path"] = requirements_path
        observed["text"] = requirements_path.read_text(encoding="utf-8")
        return _completed_process(return_code=0, stdout=json.dumps(payload))

    monkeypatch.setattr(MODULE.subprocess, "run", run)

    result = MODULE._run_pip_audit_json("torch==2.13.0\n", timeout_seconds=15)

    assert observed["text"] == "torch==2.13.0\n"
    assert "--no-deps" in observed["command"]
    assert "--disable-pip" in observed["command"]
    assert not observed["path"].exists()
    assert "{temporary_project_closure}" in result["command"]
    assert result["error"] is None


def test_rc1_with_valid_json_is_a_completed_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "dependencies": [
            {
                "name": "torch",
                "version": "2.13.0",
                "vulns": [{"id": "PYSEC-TEST-1", "aliases": []}],
            }
        ]
    }
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: _completed_process(return_code=1, stdout=json.dumps(payload)),
    )

    result = MODULE._run_pip_audit_json("torch==2.13.0\n", timeout_seconds=15)

    assert result["return_code"] == 1
    assert result["error"] is None
    assert result["dependencies"] == payload["dependencies"]


@pytest.mark.parametrize(
    ("return_code", "stdout", "stderr", "expected_error"),
    [
        (2, "", "resolver failure", "resolver failure"),
        (1, "pip-audit emitted no payload", "", "JSON payload not found"),
        (1, '{"dependencies": [}', "", "invalid pip-audit JSON"),
        (0, '{"dependencies": []}', "", "zero dependencies"),
    ],
)
def test_pip_audit_execution_failures_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    stdout: str,
    stderr: str,
    expected_error: str,
) -> None:
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: _completed_process(
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        ),
    )

    result = MODULE._run_pip_audit_json("torch==2.13.0\n", timeout_seconds=15)

    assert expected_error in result["error"]
    assert result["dependencies"] == []


def test_exact_project_closure_passes_without_persistent_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_inputs(monkeypatch, tmp_path, policy={"schema_version": "1.0.0", "entries": {}})
    monkeypatch.setattr(MODULE, "_collect_project_environment_input", _input_report, raising=False)
    monkeypatch.setattr(MODULE, "_run_pip_audit_json", lambda *args, **kwargs: _dependency_payload())

    report = MODULE.build_dependency_audit_report(now=NOW)

    assert report["schema_version"] == "2.0.0"
    assert report["mode"] == "project_environment"
    assert report["critical_input_missing"] == 0
    assert report["input_integrity_violations"] == 0
    assert report["pip_audit"]["dependencies_count"] == 2
    assert report["pass"] is True


@pytest.mark.parametrize("mutation", ["missing", "version"])
def test_audit_output_must_exactly_match_input_closure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    _configure_inputs(monkeypatch, tmp_path, policy={"schema_version": "1.0.0", "entries": {}})
    payload = _dependency_payload()
    if mutation == "missing":
        payload["dependencies"] = payload["dependencies"][1:]
    else:
        payload["dependencies"][0]["version"] = "0.0.1"
    monkeypatch.setattr(MODULE, "_collect_project_environment_input", _input_report, raising=False)
    monkeypatch.setattr(MODULE, "_run_pip_audit_json", lambda *args, **kwargs: payload)

    report = MODULE.build_dependency_audit_report(now=NOW)

    assert report["input_integrity_violations"] > 0
    assert report["pass"] is False


def test_expired_vulnerability_exception_is_unaccepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_inputs(monkeypatch, tmp_path, policy=_policy(vulnerability_review_date="2026-06-30"))
    monkeypatch.setattr(MODULE, "_collect_project_environment_input", _input_report, raising=False)
    monkeypatch.setattr(
        MODULE,
        "_run_pip_audit_json",
        lambda *args, **kwargs: _dependency_payload(
            return_code=1,
            vulnerability_id="PYSEC-TEST-1",
        ),
    )

    report = MODULE.build_dependency_audit_report(now=NOW)

    assert report["accepted_vulnerability_count"] == 0
    assert report["vulnerability_count"] == 1
    assert report["pass"] is False


def test_advisory_alias_with_current_policy_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(MODULE, "_collect_project_environment_input", _input_report, raising=False)
    monkeypatch.setattr(
        MODULE,
        "_run_pip_audit_json",
        lambda *args, **kwargs: _dependency_payload(
            return_code=1,
            vulnerability_id="CVE-ALIAS-1",
            aliases=["PYSEC-TEST-1"],
        ),
    )

    report = MODULE.build_dependency_audit_report(now=NOW)

    assert report["accepted_vulnerability_count"] == 1
    assert report["vulnerability_count"] == 0
    assert report["pass"] is True


def test_critical_package_skip_requires_current_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_inputs(monkeypatch, tmp_path, policy=_policy(package_review_date="2026-06-30"))
    monkeypatch.setattr(MODULE, "_collect_project_environment_input", _input_report, raising=False)
    monkeypatch.setattr(
        MODULE,
        "_run_pip_audit_json",
        lambda *args, **kwargs: _dependency_payload(skip_reason="not on index"),
    )

    report = MODULE.build_dependency_audit_report(now=NOW)

    assert report["critical_skip_without_policy"] == 1
    assert report["pass"] is False


@pytest.mark.parametrize(("passes", "expected_exit"), [(True, 0), (False, 1)])
def test_main_exit_code_matches_report_pass_and_always_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    passes: bool,
    expected_exit: int,
) -> None:
    _configure_inputs(monkeypatch, tmp_path)
    report = {
        "schema_version": "2.0.0",
        "generated_at_utc": NOW.isoformat(),
        "mode": "project_environment",
        "policy_issues": [] if passes else ["scan failed"],
        "input_report": _input_report(),
        "critical_input_missing": 0,
        "input_integrity_violations": 0,
        "critical_skip_without_policy": 0,
        "vulnerability_count": 0,
        "accepted_vulnerability_count": 0,
        "vulnerability_exceptions": [],
        "unaccepted_vulnerabilities": [],
        "skipped_packages": [],
        "pip_audit": {
            "command": "python -m pip_audit --no-deps --disable-pip",
            "return_code": 0 if passes else 2,
            "stderr": "",
            "error": None if passes else "scan failed",
            "dependencies_count": 2 if passes else 0,
            "dependencies_sha256": "B" * 64,
        },
        "pass": passes,
    }
    monkeypatch.setattr(
        MODULE,
        "build_dependency_audit_report",
        lambda **kwargs: report,
        raising=False,
    )

    exit_code = MODULE.main()

    assert exit_code == expected_exit
    assert MODULE.OUTPUT_PATH.is_file()
    assert json.loads(MODULE.OUTPUT_PATH.read_text(encoding="utf-8")) == report
