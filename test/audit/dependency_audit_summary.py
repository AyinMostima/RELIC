"""Dependency governance audit summary for reproducibility and security."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "analysis_results"

AUDIT_MANIFEST_PATH = REPO_ROOT / "test" / "audit" / "dependency_audit_manifest.json"
SKIP_POLICY_PATH = REPO_ROOT / "test" / "audit" / "dependency_skip_allowlist.json"
PROJECT_METADATA_PATH = REPO_ROOT / "pyproject.toml"
OUTPUT_PATH = ANALYSIS_ROOT / "dependency_audit_summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _validate_review_date(
    value: Any,
    *,
    label: str,
    current_date: date,
    issues: list[str],
) -> date | None:
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{label} missing field: review_date")
        return None
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError:
        issues.append(f"{label} has invalid review_date: {value}")
        return None
    if parsed < current_date:
        issues.append(f"{label} expired on {parsed.isoformat()}")
    return parsed


def _validate_policy_schema(
    policy: dict[str, Any],
    *,
    current_date: date | None = None,
) -> list[str]:
    issues: list[str] = []
    reference_date = current_date or datetime.now(timezone.utc).date()
    if policy.get("schema_version") != "1.0.0":
        issues.append("policy.schema_version must be 1.0.0")
    entries = policy.get("entries", {})
    if not isinstance(entries, dict):
        return ["policy.entries is not a mapping"]
    for package_name, config in entries.items():
        if not isinstance(config, dict):
            issues.append(f"policy entry for {package_name} must be object")
            continue
        for field in ("reason", "mitigation", "owner"):
            value = config.get(field)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"policy entry for {package_name} missing field: {field}")
        _validate_review_date(
            config.get("review_date"),
            label=f"package policy for {package_name}",
            current_date=reference_date,
            issues=issues,
        )
        vulnerabilities = config.get("vulnerabilities", {})
        if vulnerabilities and not isinstance(vulnerabilities, dict):
            issues.append(f"policy vulnerabilities for {package_name} must be object")
            continue
        for vuln_id, vuln_config in vulnerabilities.items():
            if not isinstance(vuln_config, dict):
                issues.append(f"vulnerability policy for {package_name}:{vuln_id} must be object")
                continue
            for field in ("reason", "mitigation", "owner"):
                value = vuln_config.get(field)
                if not isinstance(value, str) or not value.strip():
                    issues.append(
                        f"vulnerability policy for {package_name}:{vuln_id} missing field: {field}"
                    )
            _validate_review_date(
                vuln_config.get("review_date"),
                label=(f"vulnerability policy for {package_name}:{vuln_id}"),
                current_date=reference_date,
                issues=issues,
            )
    return issues


def _matching_vulnerability_policy(
    policy_entries: dict[str, Any],
    package_name: str,
    vulnerability: dict[str, Any],
    *,
    current_date: date,
) -> dict[str, Any] | None:
    package_policy = {k.lower(): v for k, v in policy_entries.items()}.get(
        package_name.lower(),
        {},
    )
    if not isinstance(package_policy, dict):
        return None
    vulnerability_policy = package_policy.get("vulnerabilities", {})
    if not isinstance(vulnerability_policy, dict):
        return None
    identifiers = {str(vulnerability.get("id", "")).strip()}
    aliases = vulnerability.get("aliases", [])
    if isinstance(aliases, list):
        identifiers.update(str(alias).strip() for alias in aliases)
    identifiers.discard("")
    matching_keys = sorted(identifiers & {str(key).strip() for key in vulnerability_policy})
    for key in matching_keys:
        config = vulnerability_policy.get(key)
        if not isinstance(config, dict):
            continue
        raw_review_date = config.get("review_date")
        if not isinstance(raw_review_date, str):
            continue
        try:
            review_date = date.fromisoformat(raw_review_date)
        except ValueError:
            continue
        if review_date >= current_date:
            return config
    return None


def _has_current_package_policy(
    policy_entries: dict[str, Any],
    package_name: str,
    *,
    current_date: date,
) -> bool:
    package_policy = {k.lower(): v for k, v in policy_entries.items()}.get(
        package_name.lower(),
    )
    if not isinstance(package_policy, dict):
        return False
    raw_review_date = package_policy.get("review_date")
    if not isinstance(raw_review_date, str):
        return False
    try:
        review_date = date.fromisoformat(raw_review_date)
    except ValueError:
        return False
    return review_date >= current_date


def _marker_applies(requirement: Requirement, extras: set[str]) -> bool:
    if requirement.marker is None:
        return True
    environment = default_environment()
    contexts = extras or {""}
    return any(
        requirement.marker.evaluate({**environment, "extra": extra})
        for extra in contexts
    )


def _collect_project_environment_input() -> dict[str, Any]:
    """Resolve the declared runtime closure from the verified Python environment."""

    integrity_issues: list[str] = []
    try:
        manifest = _read_json(AUDIT_MANIFEST_PATH)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        manifest = {}
        integrity_issues.append(f"audit manifest could not be read: {exc}")
    if not isinstance(manifest, dict):
        manifest = {}
        integrity_issues.append("audit manifest root is not an object")
    if manifest.get("schema_version") != "2.0.0":
        integrity_issues.append("audit manifest schema_version must be 2.0.0")
    if manifest.get("audit_scope") != "installed_project_runtime_closure":
        integrity_issues.append(
            "audit manifest scope must be installed_project_runtime_closure"
        )
    if manifest.get("project_metadata_path") != "pyproject.toml":
        integrity_issues.append("audit manifest project_metadata_path must be pyproject.toml")

    raw_critical = manifest.get("critical_packages", [])
    if not isinstance(raw_critical, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_critical
    ):
        integrity_issues.append("audit manifest critical_packages must be a string list")
        raw_critical = []
    critical_packages = sorted(
        {canonicalize_name(item) for item in raw_critical}
    )

    project_sha256 = ""
    direct_requirements: list[Requirement] = []
    try:
        project_bytes = PROJECT_METADATA_PATH.read_bytes()
        project_sha256 = hashlib.sha256(project_bytes).hexdigest().upper()
        project = tomllib.loads(project_bytes.decode("utf-8"))
        raw_dependencies = project.get("project", {}).get("dependencies", [])
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_dependencies
        ):
            integrity_issues.append("pyproject project.dependencies must be a string list")
            raw_dependencies = []
        for text in raw_dependencies:
            try:
                direct_requirements.append(Requirement(text))
            except Exception as exc:
                integrity_issues.append(f"invalid project dependency {text!r}: {exc}")
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        integrity_issues.append(f"project metadata could not be read: {exc}")

    resolved: dict[str, dict[str, str]] = {}
    requested_extras: dict[str, set[str]] = defaultdict(set)
    direct_names: set[str] = set()
    queue: deque[str] = deque()

    def add_requirement(
        requirement: Requirement,
        marker_extras: set[str],
        *,
        direct: bool = False,
    ) -> None:
        if not _marker_applies(requirement, marker_extras):
            return
        name = canonicalize_name(requirement.name)
        if direct:
            direct_names.add(name)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            integrity_issues.append(f"missing installed distribution: {name}")
            return
        try:
            installed_version = Version(distribution.version)
        except InvalidVersion as exc:
            integrity_issues.append(f"invalid installed version for {name}: {exc}")
            return
        if requirement.specifier and installed_version not in requirement.specifier:
            integrity_issues.append(
                "installed version violates declaration: "
                f"{name} {installed_version} not in {requirement.specifier}"
            )
        first_visit = name not in resolved
        resolved[name] = {
            "name": name,
            "installed_version": str(installed_version),
            "audit_version": installed_version.public,
        }
        new_extras = set(requirement.extras) - requested_extras[name]
        requested_extras[name].update(requirement.extras)
        if first_visit or new_extras:
            queue.append(name)

    for requirement in direct_requirements:
        add_requirement(requirement, {""}, direct=True)
    while queue:
        package_name = queue.popleft()
        distribution = metadata.distribution(package_name)
        marker_extras = requested_extras[package_name] or {""}
        for text in distribution.requires or []:
            try:
                requirement = Requirement(text)
            except Exception as exc:
                integrity_issues.append(
                    f"invalid Requires-Dist for {package_name}: {text}: {exc}"
                )
                continue
            add_requirement(requirement, marker_extras)

    packages = [resolved[name] for name in sorted(resolved)]
    requirements_text = "".join(
        f"{item['name']}=={item['audit_version']}\n" for item in packages
    )
    requirements_sha256 = hashlib.sha256(
        requirements_text.encode("utf-8")
    ).hexdigest().upper()
    critical_missing_declarations = sorted(
        set(critical_packages) - direct_names
    )
    critical_missing_resolution = sorted(
        set(critical_packages) - set(resolved)
    )
    for package_name in critical_missing_declarations:
        integrity_issues.append(
            f"critical package is not directly declared: {package_name}"
        )
    for package_name in critical_missing_resolution:
        integrity_issues.append(
            f"critical package is not resolved: {package_name}"
        )

    return {
        "kind": "installed_project_runtime_closure",
        "project_metadata_path": "pyproject.toml",
        "project_metadata_sha256": project_sha256,
        "python_executable": str(Path(sys.executable).resolve()),
        "environment_prefix": str(Path(sys.prefix).resolve()),
        "direct_dependency_count": len(direct_names),
        "direct_dependencies": sorted(direct_names),
        "resolved_dependency_count": len(packages),
        "packages": packages,
        "requirements_text": requirements_text,
        "requirements_sha256": requirements_sha256,
        "critical_packages": critical_packages,
        "critical_packages_missing_from_declarations": critical_missing_declarations,
        "critical_packages_missing_from_resolution": critical_missing_resolution,
        "integrity_issues": sorted(set(integrity_issues)),
    }


def _run_pip_audit_json(
    requirements_text: str,
    *,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    semantic_command = (
        f"{sys.executable} -m pip_audit -f json -S --progress-spinner off "
        "--no-deps --disable-pip -r {temporary_project_closure}"
    )
    with tempfile.TemporaryDirectory(prefix="relic-dependency-audit-") as temp_dir:
        requirement_path = Path(temp_dir) / "project-runtime-closure.txt"
        requirement_path.write_text(requirements_text, encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "pip_audit",
            "-f",
            "json",
            "-S",
            "--progress-spinner",
            "off",
            "--no-deps",
            "--disable-pip",
            "-r",
            str(requirement_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "command": semantic_command,
                "return_code": None,
                "stderr": "",
                "error": f"pip-audit timed out after {timeout_seconds} seconds",
                "dependencies": [],
            }
        except OSError as exc:
            return {
                "command": semantic_command,
                "return_code": None,
                "stderr": "",
                "error": f"pip-audit could not start: {exc}",
                "dependencies": [],
            }
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode not in (0, 1):
        return {
            "command": semantic_command,
            "return_code": completed.returncode,
            "stderr": stderr.strip(),
            "error": stderr.strip() or "pip-audit failed",
            "dependencies": [],
        }

    start = stdout.find("{")
    if start < 0:
        return {
            "command": semantic_command,
            "return_code": completed.returncode,
            "stderr": stderr.strip(),
            "error": "pip-audit JSON payload not found in stdout",
            "dependencies": [],
        }

    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as exc:
        return {
            "command": semantic_command,
            "return_code": completed.returncode,
            "stderr": stderr.strip(),
            "error": f"invalid pip-audit JSON: {exc.msg}",
            "dependencies": [],
        }
    if not isinstance(payload, dict):
        return {
            "command": semantic_command,
            "return_code": completed.returncode,
            "stderr": stderr.strip(),
            "error": "pip-audit JSON root is not an object",
            "dependencies": [],
        }
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        return {
            "command": semantic_command,
            "return_code": completed.returncode,
            "stderr": stderr.strip(),
            "error": "pip-audit dependencies is not a list",
            "dependencies": [],
        }
    if not dependencies:
        return {
            "command": semantic_command,
            "return_code": completed.returncode,
            "stderr": stderr.strip(),
            "error": "pip-audit returned zero dependencies",
            "dependencies": [],
        }
    vulnerability_count = sum(
        len(item.get("vulns", []))
        for item in dependencies
        if isinstance(item, dict) and isinstance(item.get("vulns", []), list)
    )
    inconsistent_exit = (completed.returncode == 0 and vulnerability_count > 0) or (
        completed.returncode == 1 and vulnerability_count == 0
    )
    if inconsistent_exit:
        return {
            "command": semantic_command,
            "return_code": completed.returncode,
            "stderr": stderr.strip(),
            "error": ("pip-audit return code is inconsistent with its vulnerability payload"),
            "dependencies": [],
        }
    return {
        "command": semantic_command,
        "return_code": completed.returncode,
        "stderr": stderr.strip(),
        "error": None,
        "dependencies": dependencies,
    }


def _validate_audit_alignment(
    input_report: dict[str, Any],
    dependencies: list[Any],
) -> tuple[list[str], str]:
    issues: list[str] = []
    expected = {
        str(item.get("name")): str(item.get("audit_version"))
        for item in input_report.get("packages", [])
        if isinstance(item, dict)
    }
    actual: dict[str, str] = {}
    for item in dependencies:
        if not isinstance(item, dict):
            issues.append("pip-audit dependency entry is not an object")
            continue
        raw_name = item.get("name")
        raw_version = item.get("version")
        if not isinstance(raw_name, str) or not raw_name.strip():
            issues.append("pip-audit dependency name is invalid")
            continue
        if not isinstance(raw_version, str) or not raw_version.strip():
            issues.append(f"pip-audit dependency version is invalid: {raw_name}")
            continue
        name = canonicalize_name(raw_name)
        if name in actual:
            issues.append(f"pip-audit returned duplicate dependency: {name}")
            continue
        actual[name] = raw_version
    for name in sorted(set(expected) - set(actual)):
        issues.append(f"pip-audit omitted project dependency: {name}")
    for name in sorted(set(actual) - set(expected)):
        issues.append(f"pip-audit returned unbound dependency: {name}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name] != actual[name]:
            issues.append(
                f"pip-audit version mismatch for {name}: "
                f"expected {expected[name]}, observed {actual[name]}"
            )
    identity_text = "".join(
        f"{name}=={actual[name]}\n" for name in sorted(actual)
    )
    digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest().upper()
    return issues, digest


def build_dependency_audit_report(
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build one fail-closed dependency evidence record."""

    generated_at = now or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    current_date = generated_at.date()
    policy_issues: list[str] = []
    try:
        policy = _read_json(SKIP_POLICY_PATH)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        policy = {}
        policy_issues.append(f"dependency policy could not be read: {exc}")
    if not isinstance(policy, dict):
        policy = {}
        policy_issues.append("dependency policy root is not an object")
    policy_issues.extend(
        _validate_policy_schema(policy, current_date=current_date)
    )

    input_report = _collect_project_environment_input()
    raw_input_issues = input_report.get("integrity_issues", [])
    input_issues = (
        list(raw_input_issues)
        if isinstance(raw_input_issues, list)
        else ["input_report.integrity_issues is not a list"]
    )
    requirements_text = input_report.get("requirements_text", "")
    if input_issues or not isinstance(requirements_text, str) or not requirements_text:
        pip_audit_result: dict[str, Any] = {
            "command": None,
            "return_code": None,
            "stderr": "",
            "error": "project runtime closure input is invalid",
            "dependencies": [],
        }
    else:
        pip_audit_result = _run_pip_audit_json(requirements_text)

    raw_dependencies = pip_audit_result.get("dependencies", [])
    dependencies = raw_dependencies if isinstance(raw_dependencies, list) else []
    alignment_issues, dependencies_sha256 = _validate_audit_alignment(
        input_report,
        dependencies,
    )
    vulnerability_exceptions: list[dict[str, str]] = []
    unaccepted_vulnerabilities: list[dict[str, str]] = []
    skipped_packages: list[dict[str, str]] = []
    raw_policy_entries = policy.get("entries", {})
    policy_entries = raw_policy_entries if isinstance(raw_policy_entries, dict) else {}
    for item in dependencies:
        if not isinstance(item, dict):
            continue
        vulns = item.get("vulns", [])
        package_name = str(item.get("name", ""))
        if isinstance(vulns, list):
            for vulnerability in vulns:
                if not isinstance(vulnerability, dict):
                    unaccepted_vulnerabilities.append(
                        {"name": package_name, "id": ""}
                    )
                    continue
                exception = _matching_vulnerability_policy(
                    policy_entries,
                    package_name,
                    vulnerability,
                    current_date=current_date,
                )
                record = {
                    "name": package_name,
                    "id": str(vulnerability.get("id", "")),
                }
                if exception is None:
                    unaccepted_vulnerabilities.append(record)
                else:
                    vulnerability_exceptions.append(record)
        skip_reason = item.get("skip_reason")
        if skip_reason:
            skipped_packages.append(
                {"name": package_name, "reason": str(skip_reason)}
            )

    critical_packages = [
        canonicalize_name(item)
        for item in input_report.get("critical_packages", [])
        if isinstance(item, str)
    ]
    critical_input_missing = sum(
        len(input_report.get(field, []))
        for field in (
            "critical_packages_missing_from_declarations",
            "critical_packages_missing_from_resolution",
        )
        if isinstance(input_report.get(field), list)
    )
    critical_skip_without_policy = 0
    for package_name in critical_packages:
        if any(
            canonicalize_name(item["name"]) == package_name
            for item in skipped_packages
        ) and not _has_current_package_policy(
            policy_entries,
            package_name,
            current_date=current_date,
        ):
            critical_skip_without_policy += 1

    accepted_vulnerability_count = len(vulnerability_exceptions)
    vulnerability_count = len(unaccepted_vulnerabilities)
    raw_vulnerability_count = accepted_vulnerability_count + vulnerability_count
    return_code = pip_audit_result.get("return_code")
    input_integrity_violations = len(input_issues) + len(alignment_issues)
    scan_complete = (
        return_code in {0, 1}
        and pip_audit_result.get("error") is None
        and len(dependencies) > 0
        and input_integrity_violations == 0
        and (
            (return_code == 0 and raw_vulnerability_count == 0)
            or (return_code == 1 and raw_vulnerability_count > 0)
        )
    )
    pass_gate = (
        scan_complete
        and not policy_issues
        and critical_input_missing == 0
        and critical_skip_without_policy == 0
        and not skipped_packages
        and vulnerability_count == 0
    )

    return {
        "schema_version": "2.0.0",
        "generated_at_utc": generated_at.isoformat(),
        "mode": "project_environment",
        "policy_issues": policy_issues,
        "input_report": input_report,
        "audit_alignment_issues": alignment_issues,
        "critical_input_missing": critical_input_missing,
        "input_integrity_violations": input_integrity_violations,
        "critical_skip_without_policy": critical_skip_without_policy,
        "vulnerability_count": vulnerability_count,
        "accepted_vulnerability_count": accepted_vulnerability_count,
        "vulnerability_exceptions": vulnerability_exceptions,
        "unaccepted_vulnerabilities": unaccepted_vulnerabilities,
        "skipped_packages": skipped_packages,
        "pip_audit": {
            "command": pip_audit_result.get("command"),
            "return_code": return_code,
            "stderr": pip_audit_result.get("stderr"),
            "error": pip_audit_result.get("error"),
            "dependencies_count": len(dependencies),
            "dependencies_sha256": dependencies_sha256,
        },
        "pass": pass_gate,
    }


def main() -> int:
    result = build_dependency_audit_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    status = "[OK]" if result["pass"] else "[ERROR]"
    print(f"{status} summary={OUTPUT_PATH}")
    print(
        f"{status} pass={result['pass']} "
        f"unaccepted_vulnerabilities="
        f"{result['vulnerability_count']} "
        f"accepted_vulnerabilities="
        f"{result['accepted_vulnerability_count']} "
        f"skips={len(result['skipped_packages'])} "
        f"input_integrity_violations="
        f"{result['input_integrity_violations']} "
        f"critical_input_missing={result['critical_input_missing']}"
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
