"""
Run bundle provenance recorder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import torch
from torch import Tensor

from .models import ProvActivity, ProvAgent, ProvEntity
from .serializer import serialize_prov_jsonld

_RUN_BUNDLE_SCHEMA = "relic.run_bundle"
_RUN_BUNDLE_SCHEMA_VERSION = "1.0.0"
_MANIFEST_NAME = "manifest.json"
_CHECKSUMS_NAME = "checksums.sha256"
_MANIFEST_KEYS = {
    "schema",
    "schema_version",
    "run_id",
    "metadata",
    "artifacts",
    "files",
}
_CONTROL_ARTIFACT_PATHS = {
    "env_snapshot": "env_snapshot.txt",
    "git_status": "git_status.json",
    "inputs_manifest": "inputs_manifest.sha256",
    "prov": "prov.jsonld",
    "runtime_log": "runtime.log.jsonl",
    "manifest": _MANIFEST_NAME,
    "checksums": _CHECKSUMS_NAME,
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _safe_ascii(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode("ascii")


def _write_text_atomic(path: Path, content: str) -> None:
    """Publish one complete local record, retaining the old file on interruption."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True))


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_repo_root(start: Path) -> Path | None:
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _list_packages() -> list[str]:
    """List every installed distribution as name==version for the environment snapshot.

    A distribution without Name or Version metadata (for example a broken
    *.dist-info directory) cannot be recorded; it raises instead of being left
    out of the snapshot. A missing header reads as None before Python 3.12 and
    raises KeyError later, so the header is tested for presence first.
    """
    entries: list[str] = []
    for dist in metadata.distributions():
        package_metadata = dist.metadata
        location = getattr(dist, "_path", dist)
        if package_metadata is None or "Name" not in package_metadata or not package_metadata["Name"]:
            raise RuntimeError(
                f"Installed distribution at {location} has no Name metadata; "
                "the environment snapshot cannot list it"
            )
        name = package_metadata["Name"]
        version = dist.version
        if not version:
            raise RuntimeError(
                f"Installed distribution {name} at {location} has no Version metadata; "
                "the environment snapshot cannot list it"
            )
        entries.append(f"{name}=={version}")
    return sorted(set(entries))


def _write_env_snapshot(path: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = [
        f"timestamp_utc={timestamp}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        "packages:",
    ]
    lines.extend(_list_packages())
    content = "\n".join(_safe_ascii(line) for line in lines) + "\n"
    path.write_text(content, encoding="utf-8")


def _get_git_info(repo_root: Path | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False,
        "head": None,
        "branch": None,
        "is_dirty": None,
        "repo_root": str(repo_root) if repo_root else None,
    }
    if repo_root is None:
        return info
    # Git is an optional external tool: without a git executable the record
    # states available=False. Any failure to run an existing git executable
    # (an invalid repo_root, a permission error) propagates.
    git_executable = shutil.which("git")
    if git_executable is None:
        return info
    head = subprocess.run(
        [git_executable, "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    branch = subprocess.run(
        [git_executable, "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        [git_executable, "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode == 0:
        info["head"] = head.stdout.strip()
    if branch.returncode == 0:
        info["branch"] = branch.stdout.strip()
    if status.returncode == 0:
        info["is_dirty"] = bool(status.stdout.strip())
    info["available"] = head.returncode == 0
    return info


def _resolve_external_inputs(
    files: Iterable[Path],
    bundle_root: Path,
) -> list[Path]:
    seen: dict[str, Path] = {}
    for file_path in files:
        candidate = Path(file_path)
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(
                f"Declared input file does not exist: {candidate}"
            )
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(bundle_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"Declared input file must be external to the run bundle: {resolved}"
            )
        seen[str(resolved)] = resolved
    return sorted(seen.values(), key=lambda item: str(item))


def _write_inputs_manifest(path: Path, files: Iterable[Path]) -> None:
    lines: list[str] = []
    for resolved in files:
        digest = _hash_file(resolved)
        lines.append(f"{digest}  {resolved}")
    content = "\n".join(lines)
    if lines:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _commit_context_record(commit_context: Any) -> dict[str, Any]:
    """JSON form of the commit context of one state notification.

    References: design 20261019_cr_a04_design_01, D7 (StateManager
        commit_context); RFC 8259 (JSON numbers are finite).
    Mathematical expression: none; non-finite floats become null, the rule of
        the descriptive value sums of record_state_update.
    Parameter meanings: commit_context is the copy StateManager delivers
        (plain values and the element_net_change mapping of the commit).
    Units: those of the context values.
    Domain/constraints: a mapping with non-empty string keys whose values are
        JSON values; anything else raises ValueError naming the commit context.
    Sign convention: not applicable.
    """
    def finite(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if type(value) is float and not math.isfinite(value):
            return None
        return value

    if not isinstance(commit_context, dict) or any(
        not isinstance(key, str) or not key for key in commit_context
    ):
        raise ValueError("Provenance commit context must be a mapping with non-empty string keys")
    try:
        record = json.loads(json.dumps(finite(commit_context), ensure_ascii=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Provenance commit context must be JSON-safe") from exc
    return record


def _copy_json_object(payload: dict[str, Any] | None) -> dict[str, Any]:
    value = payload or {}
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError("Run bundle metadata must be JSON-safe") from exc
    if not isinstance(normalized, dict) or normalized != value:
        raise TypeError("Run bundle metadata must preserve its JSON representation")
    return normalized


def _validate_relative_path(relative_path: Any) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Run bundle paths must be non-empty strings")
    if "\\" in relative_path or "\x00" in relative_path or ":" in relative_path:
        raise ValueError(f"Unsafe run bundle relative path: {relative_path!r}")
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or pure_path.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ValueError(f"Unsafe run bundle relative path: {relative_path!r}")
    return relative_path


def _bundle_relative_path(bundle_root: Path, path: Path) -> str:
    resolved_root = bundle_root.resolve()
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Run bundle artifact must be inside output_dir: {path}"
        ) from exc
    return _validate_relative_path(relative_path)


def _write_checksums(
    bundle_root: Path,
    checksums_path: Path,
    relative_paths: Iterable[str],
) -> None:
    lines: list[str] = []
    for relative_path in sorted(relative_paths):
        normalized = _validate_relative_path(relative_path)
        if normalized == _CHECKSUMS_NAME:
            continue
        target = bundle_root.joinpath(*PurePosixPath(normalized).parts)
        if not target.is_file():
            raise FileNotFoundError(
                f"Run bundle file is missing before checksum sealing: {normalized}"
            )
        lines.append(f"{_hash_file(target)}  {normalized}")
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _read_json_document(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON document is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return _copy_json_object(payload)


def _read_runtime_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        try:
            event = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid runtime JSON record {index}: {path}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"Runtime JSON record {index} must be an object: {path}")
        events.append(_copy_json_object(event))
    return events


def _merge_prov_documents(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge native PROV records without duplicating or replacing their meaning."""
    merged = deepcopy(current)
    if merged.get("@context", incoming["@context"]) != incoming["@context"]:
        raise ValueError("Provenance context conflict")
    merged["@context"] = incoming["@context"]
    for kind in ("entity", "activity", "agent"):
        target = merged.setdefault(kind, {})
        if not isinstance(target, dict) or any(not isinstance(value, dict) for value in target.values()):
            raise ValueError(f"Invalid provenance {kind} collection")
        for identity, value in incoming.get(kind, {}).items():
            if identity in target and target[identity] != value:
                raise ValueError(f"Provenance content conflict for {identity}")
            target[identity] = deepcopy(value)
    for kind in ("wasGeneratedBy", "wasDerivedFrom", "wasAssociatedWith"):
        target = merged.setdefault(kind, [])
        if not isinstance(target, list) or any(not isinstance(value, dict) for value in target):
            raise ValueError(f"Invalid provenance {kind} collection")
        for relation in incoming.get(kind, []):
            identity = {key: value for key, value in relation.items() if key.startswith("prov:")}
            matches = [value for value in target if {key: item for key, item in value.items() if key.startswith("prov:")} == identity]
            if matches and (len(matches) != 1 or matches[0] != relation):
                raise ValueError(f"Provenance relation conflict in {kind}")
            if not matches:
                target.append(deepcopy(relation))
    return merged


def _append_runtime_event(path: Path, payload: dict[str, Any]) -> None:
    events = _read_runtime_events(path)
    notification_id = payload.get("notification_id")
    if notification_id is not None:
        matches = [event for event in events if event.get("notification_id") == notification_id]
        if matches:
            if len(matches) != 1 or {key: value for key, value in matches[0].items() if key != "timestamp_utc"} != payload:
                raise ValueError(f"Runtime notification content conflict: {notification_id}")
            return
    event = dict(payload)
    event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    events.append(event)
    _write_text_atomic(path, "".join(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n" for value in events))


class ProvenanceRecorder:
    """
    Run-level provenance recorder.

    Parameters
    ----------
    output_dir : Path
        Output directory for run bundle artifacts.
    run_id : str, optional
        Identifier for the run bundle.
    repo_root : Path, optional
        Repository root for git status collection.
    """

    def __init__(
        self,
        output_dir: Path,
        run_id: str | None = None,
        repo_root: Path | None = None,
    ) -> None:
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError("Provenance run_id must be a nonempty string")
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.repo_root = repo_root or _find_repo_root(self.output_dir)

    def record_run_bundle(
        self,
        artifacts: dict[str, Path],
        input_files: list[Path] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Record provenance artifacts for a run bundle.

        Parameters
        ----------
        artifacts : dict
            Mapping of artifact names to paths.
        input_files : list of Path, optional
            Additional input files to include in the manifest.
        metadata : dict, optional
            Metadata to attach to provenance activity.

        Returns
        -------
        dict
            Mapping of artifact names to file paths (as strings).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        bundle_root = self.output_dir.resolve()
        prior_prov_path = self.output_dir / "prov.jsonld"
        prior_prov = _read_json_document(prior_prov_path) if prior_prov_path.exists() else serialize_prov_jsonld([], [], [])
        prior_prov = _merge_prov_documents(prior_prov, serialize_prov_jsonld([], [], []))
        external_inputs = _resolve_external_inputs(input_files or [], bundle_root)
        manifest_metadata = _copy_json_object(metadata)

        reserved_names = set(_CONTROL_ARTIFACT_PATHS) - {"runtime_log"}
        collisions = reserved_names.intersection(artifacts)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"Artifact names are reserved by the run bundle: {names}")

        runtime_path = Path(
            artifacts.get("runtime_log", self.output_dir / "runtime.log.jsonl")
        )
        _bundle_relative_path(bundle_root, runtime_path)
        _read_runtime_events(runtime_path)

        env_path = self.output_dir / "env_snapshot.txt"
        git_status_path = self.output_dir / "git_status.json"
        inputs_manifest_path = self.output_dir / "inputs_manifest.sha256"
        prov_path = self.output_dir / "prov.jsonld"
        manifest_path = self.output_dir / _MANIFEST_NAME
        checksums_path = self.output_dir / _CHECKSUMS_NAME

        normalized_artifacts: dict[str, Path] = {}
        seen_paths: dict[str, str] = {}
        artifact_items = dict(artifacts)
        artifact_items["runtime_log"] = runtime_path
        for name, raw_path in artifact_items.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Run bundle artifact names must be non-empty strings")
            artifact_path = Path(raw_path)
            relative_path = _bundle_relative_path(bundle_root, artifact_path)
            planned_runtime = name == "runtime_log" and not artifact_path.exists()
            if not planned_runtime and not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Declared run bundle artifact does not exist: {artifact_path}"
                )
            previous_name = seen_paths.get(relative_path)
            if previous_name is not None:
                raise ValueError(
                    "Duplicate run bundle artifact path "
                    f"{relative_path!r}: {previous_name!r} and {name!r}"
                )
            seen_paths[relative_path] = name
            normalized_artifacts[name] = artifact_path

        activity_id = f"compile_graph:{quote(self.run_id, safe='')}"
        previous_activity = prior_prov.get("activity", {}).get(activity_id, {})
        activity = ProvActivity(
            identifier=activity_id,
            label="compile_graph",
            start_time=previous_activity.get("prov:startTime", datetime.now(timezone.utc).isoformat()),
            attributes={"run_id": self.run_id, "relic:metadata": manifest_metadata},
        )
        agent = ProvAgent(
            identifier="relic",
            label="relic",
            attributes={"prov:type": "softwareAgent"},
        )
        entities: list[ProvEntity] = []
        entity_prefix = f"artifact:{quote(self.run_id, safe='')}:"
        for key, path in normalized_artifacts.items():
            entities.append(ProvEntity(identifier=entity_prefix + quote(key, safe=""), label=Path(path).name))
        entities.append(ProvEntity(identifier=entity_prefix + "env_snapshot", label=env_path.name))
        entities.append(ProvEntity(identifier=entity_prefix + "git_status", label=git_status_path.name))
        entities.append(
            ProvEntity(identifier=entity_prefix + "inputs_manifest", label=inputs_manifest_path.name)
        )

        relations = {
            "wasGeneratedBy": [
                {"prov:entity": entity.identifier, "prov:activity": activity.identifier}
                for entity in entities
            ],
            "wasAssociatedWith": [
                {"prov:activity": activity.identifier, "prov:agent": agent.identifier}
            ],
        }
        prov_payload = serialize_prov_jsonld(
            entities=entities,
            activities=[activity],
            agents=[agent],
            relations=relations,
        )
        merged_prov = _merge_prov_documents(prior_prov, prov_payload)
        # Complete all identity/content checks before changing a sealed bundle.
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        if not runtime_path.exists():
            _write_text_atomic(runtime_path, "")
        _write_env_snapshot(env_path)
        git_info = _get_git_info(self.repo_root)
        _write_json(git_status_path, git_info)
        _write_inputs_manifest(inputs_manifest_path, external_inputs)
        _write_json(prov_path, merged_prov)

        _append_runtime_event(
            runtime_path,
            {
                "event": "compile_graph",
                "run_id": self.run_id,
                "artifacts": {
                    key: str(value) for key, value in normalized_artifacts.items()
                },
            },
        )

        output: dict[str, str] = {
            key: str(value) for key, value in normalized_artifacts.items()
        }
        output.update(
            {
                "env_snapshot": str(env_path),
                "git_status": str(git_status_path),
                "inputs_manifest": str(inputs_manifest_path),
                "prov": str(prov_path),
                "runtime_log": str(runtime_path),
                "manifest": str(manifest_path),
                "checksums": str(checksums_path),
            }
        )

        artifact_manifest: dict[str, str] = {}
        for name, raw_path in output.items():
            relative_path = _bundle_relative_path(bundle_root, Path(raw_path))
            previous_name = seen_paths.get(relative_path)
            if previous_name is not None and previous_name != name:
                raise ValueError(
                    "Duplicate run bundle artifact path "
                    f"{relative_path!r}: {previous_name!r} and {name!r}"
                )
            seen_paths[relative_path] = name
            artifact_manifest[name] = relative_path
        file_manifest = sorted(artifact_manifest.values())
        if len(file_manifest) != len(set(file_manifest)):
            raise ValueError("Run bundle manifest contains duplicate file paths")

        _write_json(
            manifest_path,
            {
                "schema": _RUN_BUNDLE_SCHEMA,
                "schema_version": _RUN_BUNDLE_SCHEMA_VERSION,
                "run_id": self.run_id,
                "metadata": manifest_metadata,
                "artifacts": artifact_manifest,
                "files": file_manifest,
            },
        )
        _write_checksums(
            bundle_root=bundle_root,
            checksums_path=checksums_path,
            relative_paths=file_manifest,
        )
        self.verify_run_bundle()
        return output

    def verify_run_bundle(self) -> dict[str, Any]:
        """
        Verify the exact file tree and hashes of an existing run bundle.

        Returns
        -------
        dict
            Verification receipt for the sealed run bundle.

        Raises
        ------
        ValueError
            If schema, paths, file tree, or checksums are invalid.
        """
        bundle_root = self.output_dir.resolve()
        if not bundle_root.is_dir() or self.output_dir.is_symlink():
            raise ValueError(f"Run bundle directory is invalid: {self.output_dir}")

        manifest_path = bundle_root / _MANIFEST_NAME
        checksums_path = bundle_root / _CHECKSUMS_NAME
        if not manifest_path.is_file():
            raise ValueError(f"Run bundle manifest is missing: {manifest_path}")
        if not checksums_path.is_file():
            raise ValueError(f"Run bundle checksums are missing: {checksums_path}")

        manifest = _read_json_document(manifest_path)
        if set(manifest) != _MANIFEST_KEYS:
            raise ValueError(
                "Run bundle manifest must contain exactly: "
                + ", ".join(sorted(_MANIFEST_KEYS))
            )
        if manifest.get("schema") != _RUN_BUNDLE_SCHEMA:
            raise ValueError("Unsupported run bundle schema")
        if manifest.get("schema_version") != _RUN_BUNDLE_SCHEMA_VERSION:
            raise ValueError("Unsupported run bundle schema version")
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Run bundle run_id must be a non-empty string")
        if not isinstance(manifest.get("metadata"), dict):
            raise ValueError("Run bundle metadata must be a JSON object")

        raw_files = manifest.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("Run bundle files must be a non-empty list")
        relative_files = [_validate_relative_path(item) for item in raw_files]
        if len(relative_files) != len(set(relative_files)):
            raise ValueError("Run bundle manifest contains duplicate file paths")
        if relative_files != sorted(relative_files):
            raise ValueError("Run bundle manifest file paths must be sorted")

        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, dict) or not raw_artifacts:
            raise ValueError("Run bundle artifacts must be a non-empty object")
        artifact_paths: dict[str, str] = {}
        for name, relative_path in raw_artifacts.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Run bundle artifact names must be non-empty strings"
                )
            artifact_paths[name] = _validate_relative_path(relative_path)
        if len(artifact_paths.values()) != len(set(artifact_paths.values())):
            raise ValueError("Run bundle artifacts contain duplicate file paths")
        if set(artifact_paths.values()) != set(relative_files):
            raise ValueError("Run bundle artifacts do not cover the exact file manifest")
        for name, relative_path in _CONTROL_ARTIFACT_PATHS.items():
            if artifact_paths.get(name) != relative_path:
                raise ValueError(
                    f"Run bundle control artifact {name!r} must be {relative_path!r}"
                )

        actual_files: set[str] = set()
        actual_directories: set[str] = set()
        for path in bundle_root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Run bundle must not contain symbolic links: {path}")
            relative_path = path.relative_to(bundle_root).as_posix()
            _validate_relative_path(relative_path)
            if path.is_dir():
                actual_directories.add(relative_path)
            elif path.is_file():
                actual_files.add(relative_path)
            else:
                raise ValueError(f"Unsupported run bundle filesystem entry: {path}")

        expected_files = set(relative_files)
        if actual_files != expected_files:
            missing = sorted(expected_files - actual_files)
            extra = sorted(actual_files - expected_files)
            raise ValueError(
                f"Run bundle file tree mismatch; missing={missing}, extra={extra}"
            )
        expected_directories: set[str] = set()
        for relative_path in relative_files:
            parent = PurePosixPath(relative_path).parent
            while parent != PurePosixPath("."):
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        if actual_directories != expected_directories:
            missing = sorted(expected_directories - actual_directories)
            extra = sorted(actual_directories - expected_directories)
            raise ValueError(
                f"Run bundle directory tree mismatch; missing={missing}, extra={extra}"
            )

        checksum_records: dict[str, str] = {}
        try:
            checksum_lines = checksums_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ValueError(
                f"Run bundle checksum manifest is unreadable: {checksums_path}"
            ) from exc
        for line in checksum_lines:
            digest, separator, relative_path = line.partition("  ")
            if separator != "  " or not _SHA256_PATTERN.fullmatch(digest):
                raise ValueError(f"Invalid run bundle checksum record: {line!r}")
            normalized_path = _validate_relative_path(relative_path)
            if normalized_path in checksum_records:
                raise ValueError(
                    f"Duplicate run bundle checksum path: {normalized_path}"
                )
            checksum_records[normalized_path] = digest

        expected_checksum_paths = expected_files - {_CHECKSUMS_NAME}
        if set(checksum_records) != expected_checksum_paths:
            missing = sorted(expected_checksum_paths - set(checksum_records))
            extra = sorted(set(checksum_records) - expected_checksum_paths)
            raise ValueError(
                f"Run bundle checksum coverage mismatch; missing={missing}, extra={extra}"
            )
        for relative_path, expected_digest in checksum_records.items():
            target = bundle_root.joinpath(*PurePosixPath(relative_path).parts)
            actual_digest = _hash_file(target)
            if actual_digest != expected_digest:
                raise ValueError(
                    f"Run bundle checksum mismatch for {relative_path}: "
                    f"expected={expected_digest}, actual={actual_digest}"
                )

        return {
            "valid": True,
            "schema": _RUN_BUNDLE_SCHEMA,
            "schema_version": _RUN_BUNDLE_SCHEMA_VERSION,
            "run_id": run_id,
            "file_count": len(expected_files),
            "checksums_verified": len(checksum_records),
        }

    def get_notification_context(self) -> dict[str, str]:
        """Freeze the current recorder destination and run identity as values."""
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("Provenance run_id must be a nonempty string")
        return {"run_id": self.run_id, "output_dir": str(Path(self.output_dir).resolve())}

    def record_state_update(
        self, step: int, variable: str, pre_value: Any, post_value: Any,
        functor_increments: dict[str, Any], unit: str, *,
        notification_context: dict[str, str] | None = None,
        commit_context: dict[str, Any] | None = None,
    ) -> None:
        """Confirm one state notification across the existing PROV and runtime files.

        Step is the StateManager update index. A shared Functor contribution
        activity is not a claim about the number of individual kernel calls.
        Local writes are serialized; two-file recovery reuses the notification
        identity instead of claiming an atomic transaction across both files.
        The optional commit context of StateManager.apply_updates (stage, role,
        interval and element net change of the commit) is stored in the runtime
        event as ``commit_context``; the PROV document and the content digest
        witness the values only. A replay must carry the same context.
        """
        from relic.core.simulator.checkpoint import _encode_checkpoint_value, _payload_sha256

        context = self.get_notification_context() if notification_context is None else dict(notification_context)
        if set(context) != {"run_id", "output_dir"} or any(not isinstance(value, str) or not value.strip() for value in context.values()):
            raise ValueError("Invalid provenance notification context")
        if type(step) is not int or step < 0:
            raise ValueError("Provenance update index must be a nonnegative integer")
        if not isinstance(variable, str) or not variable or not isinstance(unit, str) or not unit:
            raise ValueError("Provenance variable and unit must be nonempty strings")
        if not isinstance(functor_increments, dict) or any(not isinstance(name, str) or not name for name in functor_increments):
            raise ValueError("Provenance increments require explicit string source names")
        commit_record = None if commit_context is None else _commit_context_record(commit_context)
        run_id = context["run_id"]
        output_dir = Path(context["output_dir"]).resolve()
        values = _encode_checkpoint_value({
            "run_id": run_id, "step": step, "variable": variable, "unit": unit,
            "pre_value": pre_value, "post_value": post_value, "increments": functor_increments,
        }, path="provenance_notification")
        digest = _payload_sha256(values)

        def value_sum(value: Any) -> float | None:
            # This descriptive reduction is not the complete content witness.
            # JSON null denotes a sum not representable as a finite real float.
            if isinstance(value, Tensor):
                if value.is_complex():
                    return None
                total = float(value.sum(dtype=torch.float64).item())
            elif type(value) in (float, int, bool):
                total = float(value)
            else:
                raise TypeError("Provenance state values must be tensors or numeric scalars")
            return total if math.isfinite(total) else None

        entity_id = f"state:{quote(run_id, safe='')}:{quote(variable, safe='')}@step{step + 1}"
        incoming = serialize_prov_jsonld([], [], [])
        incoming["entity"][entity_id] = {
            "prov:type": "prov:Entity", "prov:label": f"{variable} (update {step + 1})",
            "relic:run_id": run_id, "relic:unit": unit,
            "relic:pre_value_sum": value_sum(values["pre_value"]), "relic:post_value_sum": value_sum(values["post_value"]),
            "relic:notification_id": entity_id, "relic:content_sha256": digest,
        }
        incoming["wasGeneratedBy"] = []
        for name, delta in sorted(values["increments"].items()):
            activity_id = f"functor:{quote(run_id, safe='')}:{quote(name, safe='')}@step{step}"
            incoming["activity"][activity_id] = {
                "prov:type": "prov:Activity", "prov:label": f"{name} contribution to update {step}",
                "relic:run_id": run_id, "relic:update_index": step,
            }
            incoming["wasGeneratedBy"].append({
                "prov:entity": entity_id, "prov:activity": activity_id,
                "relic:delta_unit": unit, "relic:delta_value_sum": value_sum(delta),
                "relic:delta_sha256": _payload_sha256({"value": values["increments"][name]}),
            })
        incoming["wasDerivedFrom"] = [{
            "prov:generatedEntity": entity_id,
            "prov:usedEntity": f"state:{quote(run_id, safe='')}:{quote(variable, safe='')}@step{step}",
        }]
        event = dict(event="state_update", run_id=run_id, notification_id=entity_id,
                     content_sha256=digest, step=step, variable=variable, unit=unit,
                     pre_value_sum=value_sum(values["pre_value"]), post_value_sum=value_sum(values["post_value"]))
        if commit_record is not None:
            event["commit_context"] = commit_record
        prov_path = output_dir / "prov.jsonld"
        runtime_path = output_dir / "runtime.log.jsonl"
        current = _read_json_document(prov_path) if prov_path.exists() else serialize_prov_jsonld([], [], [])
        merged = _merge_prov_documents(current, incoming)
        records = _read_runtime_events(runtime_path)
        matching = [record for record in records if record.get("notification_id") == entity_id]
        if matching and (len(matching) != 1 or {key: value for key, value in matching[0].items() if key != "timestamp_utc"} != event):
            raise ValueError(f"Runtime notification content conflict: {entity_id}")
        _copy_json_object(merged)
        _copy_json_object(event)
        output_dir.mkdir(parents=True, exist_ok=True)
        if merged != current:
            _write_json(prov_path, merged)
        _append_runtime_event(runtime_path, event)

    def emit_lineage_event(
        self,
        node_id: str,
        event_type: str,
        cache_key: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Emit a lightweight OpenLineage-style event for node execution."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lineage_path = self.output_dir / "lineage.jsonl"
        payload: dict[str, Any] = {
            "event_type": str(event_type).upper(),
            "node_id": str(node_id),
            "run_id": self.run_id,
        }
        if cache_key:
            payload["cache_key"] = str(cache_key)
        if metrics:
            payload["metrics"] = dict(metrics)
        _append_runtime_event(lineage_path, payload)
