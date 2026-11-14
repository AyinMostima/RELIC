"""Audit gate for provenance SSOT structure in anthropogenic parameter YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILES = (
    REPO_ROOT / "relic" / "parameters" / "database" / "relic" / "anthroposphere" / "land_use.yaml",
    REPO_ROOT / "relic" / "parameters" / "database" / "relic" / "anthroposphere" / "management.yaml",
)
FLAT_PROVENANCE_FIELDS = {"source", "schema_version", "created_by", "last_modified"}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{path} must parse to dict"
    return data


def test_provenance_is_mapping_and_not_flattened() -> None:
    violations: list[str] = []

    for path in TARGET_FILES:
        data = _load_yaml(path)
        file_schema_version = str(data.get("schema_version", "")).strip()
        params = data.get("parameters")
        assert isinstance(params, dict), f"{path} missing parameters mapping"

        for key, spec in params.items():
            if not isinstance(spec, dict):
                violations.append(f"{path}::{key} spec is not dict")
                continue

            provenance = spec.get("provenance")
            if not isinstance(provenance, dict):
                violations.append(f"{path}::{key} provenance must be dict")
                continue

            for required in ("source", "schema_version"):
                if required not in provenance:
                    violations.append(f"{path}::{key} provenance missing '{required}'")

            if file_schema_version:
                psv = str(provenance.get("schema_version", "")).strip()
                if psv != file_schema_version:
                    violations.append(
                        f"{path}::{key} provenance.schema_version={psv} != file.schema_version={file_schema_version}"
                    )

            flattened = sorted(field for field in FLAT_PROVENANCE_FIELDS if field in spec)
            if flattened:
                violations.append(
                    f"{path}::{key} has flattened provenance fields: {', '.join(flattened)}"
                )

    assert not violations, "Provenance structure violations found:\n" + "\n".join(violations)
