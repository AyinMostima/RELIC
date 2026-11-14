# ruff: noqa: N999

import hashlib
import json
from pathlib import Path

import pytest

from relic.core.provenance import ProvenanceRecorder

EXPECTED_BUNDLE_FILES = {
    "workflow.json",
    "compile_report.json",
    "runtime.log.jsonl",
    "env_snapshot.txt",
    "git_status.json",
    "inputs_manifest.sha256",
    "prov.jsonld",
    "manifest.json",
    "checksums.sha256",
}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_checksums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative_path = line.partition("  ")
        assert separator == "  "
        records[relative_path] = digest
    return records


def _create_bundle(root: Path) -> tuple[Path, Path, ProvenanceRecorder]:
    run_dir = root / "bundle"
    run_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = run_dir / "workflow.json"
    report_path = run_dir / "compile_report.json"
    workflow_path.write_text("{}", encoding="utf-8")
    report_path.write_text("{}", encoding="utf-8")
    external_input = root / "forcing.csv"
    external_input.write_text("time,value\n0,1\n", encoding="utf-8")

    recorder = ProvenanceRecorder(output_dir=run_dir, run_id="test_run")
    recorder.record_run_bundle(
        artifacts={
            "workflow": workflow_path,
            "compile_report": report_path,
            "runtime_log": run_dir / "runtime.log.jsonl",
        },
        input_files=[external_input],
        metadata={
            "publication": {
                "gate": "G3",
                "profiles": ["PROFILE-001", "PROFILE-002"],
            }
        },
    )
    return run_dir, external_input, recorder


def test_provenance_recorder_creates_verifiable_exact_bundle(tmp_path: Path) -> None:
    run_dir, external_input, recorder = _create_bundle(tmp_path)

    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert actual_files == EXPECTED_BUNDLE_FILES

    inputs_lines = (run_dir / "inputs_manifest.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    assert inputs_lines == [f"{_sha256(external_input)}  {external_input.resolve()}"]
    assert all("workflow.json" not in line for line in inputs_lines)
    assert all("runtime.log.jsonl" not in line for line in inputs_lines)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "artifacts": {
            "checksums": "checksums.sha256",
            "compile_report": "compile_report.json",
            "env_snapshot": "env_snapshot.txt",
            "git_status": "git_status.json",
            "inputs_manifest": "inputs_manifest.sha256",
            "manifest": "manifest.json",
            "prov": "prov.jsonld",
            "runtime_log": "runtime.log.jsonl",
            "workflow": "workflow.json",
        },
        "files": sorted(EXPECTED_BUNDLE_FILES),
        "metadata": {
            "publication": {
                "gate": "G3",
                "profiles": ["PROFILE-001", "PROFILE-002"],
            }
        },
        "run_id": "test_run",
        "schema": "relic.run_bundle",
        "schema_version": "1.0.0",
    }

    checksums = _read_checksums(run_dir / "checksums.sha256")
    assert set(checksums) == EXPECTED_BUNDLE_FILES - {"checksums.sha256"}
    assert checksums["runtime.log.jsonl"] == _sha256(run_dir / "runtime.log.jsonl")
    assert "compile_graph" in (run_dir / "runtime.log.jsonl").read_text(
        encoding="utf-8"
    )
    for relative_path, digest in checksums.items():
        assert digest == _sha256(run_dir / relative_path)

    assert recorder.verify_run_bundle() == {
        "checksums_verified": 8,
        "file_count": 9,
        "run_id": "test_run",
        "schema": "relic.run_bundle",
        "schema_version": "1.0.0",
        "valid": True,
    }
    assert ProvenanceRecorder(output_dir=run_dir).verify_run_bundle()["valid"] is True


def test_missing_declared_input_fails_before_bundle_is_sealed(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bundle"
    run_dir.mkdir()
    workflow_path = run_dir / "workflow.json"
    workflow_path.write_text("{}", encoding="utf-8")
    missing_input = tmp_path / "missing.csv"

    recorder = ProvenanceRecorder(output_dir=run_dir, run_id="test_run")
    with pytest.raises(FileNotFoundError, match="Declared input file does not exist"):
        recorder.record_run_bundle(
            artifacts={"workflow": workflow_path},
            input_files=[missing_input],
        )

    assert not (run_dir / "manifest.json").exists()
    assert not (run_dir / "checksums.sha256").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "tampered_file",
        "missing_file",
        "extra_file",
        "duplicate_manifest_path",
        "duplicate_checksum_path",
        "unsafe_relative_path",
        "wrong_schema",
    ],
)
def test_verify_run_bundle_rejects_invalid_or_tampered_bundle(
    tmp_path: Path,
    mutation: str,
) -> None:
    run_dir, _, _ = _create_bundle(tmp_path)
    manifest_path = run_dir / "manifest.json"
    checksums_path = run_dir / "checksums.sha256"

    if mutation == "tampered_file":
        (run_dir / "workflow.json").write_text('{"tampered": true}', encoding="utf-8")
    elif mutation == "missing_file":
        (run_dir / "workflow.json").unlink()
    elif mutation == "extra_file":
        (run_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif mutation == "duplicate_manifest_path":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"].append("workflow.json")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "duplicate_checksum_path":
        first_line = checksums_path.read_text(encoding="utf-8").splitlines()[0]
        with checksums_path.open("a", encoding="utf-8") as handle:
            handle.write(first_line + "\n")
    elif mutation == "unsafe_relative_path":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][0] = "../outside.txt"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "wrong_schema":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "99.0.0"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        raise AssertionError(f"Unknown mutation: {mutation}")

    with pytest.raises(ValueError):
        ProvenanceRecorder(output_dir=run_dir).verify_run_bundle()
