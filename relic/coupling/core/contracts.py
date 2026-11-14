"""
Coupling contracts loader and validator.

Defines a lightweight schema for coupling contracts that describe
expected update_mode and scaling metadata for modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"
REQUIRED_TOP_LEVEL = {"schema_version", "contracts"}
REQUIRED_CONTRACT_FIELDS = {"module", "variables", "update_mode", "time_scale", "space_scale"}
ALLOWED_UPDATE_MODES = {"accumulate", "replace"}


def load_contracts(path: Path) -> Dict[str, Any]:
    """Load coupling contracts from YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Contracts file must contain a YAML mapping at the top level.")
    return data


def validate_contracts(data: Dict[str, Any]) -> List[str]:
    """Validate contract structure and return issues."""
    issues: List[str] = []
    missing_top = REQUIRED_TOP_LEVEL - set(data.keys())
    if missing_top:
        issues.append(f"Missing top-level keys: {sorted(missing_top)}")
        return issues

    if str(data.get("schema_version")) != SCHEMA_VERSION:
        issues.append(
            f"Unsupported schema_version '{data.get('schema_version')}', expected '{SCHEMA_VERSION}'"
        )

    contracts = data.get("contracts")
    if not isinstance(contracts, list):
        issues.append("contracts must be a list of contract entries")
        return issues

    for idx, entry in enumerate(contracts):
        if not isinstance(entry, dict):
            issues.append(f"contracts[{idx}] must be a mapping")
            continue
        missing = REQUIRED_CONTRACT_FIELDS - set(entry.keys())
        if missing:
            issues.append(f"contracts[{idx}] missing fields: {sorted(missing)}")
            continue
        if not isinstance(entry.get("module"), str):
            issues.append(f"contracts[{idx}].module must be a string")
        variables = entry.get("variables")
        if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
            issues.append(f"contracts[{idx}].variables must be a list of strings")
        update_mode = str(entry.get("update_mode")).lower()
        if update_mode not in ALLOWED_UPDATE_MODES:
            issues.append(
                f"contracts[{idx}].update_mode must be one of {sorted(ALLOWED_UPDATE_MODES)}"
            )
        for field in ("time_scale", "space_scale"):
            if not isinstance(entry.get(field), str):
                issues.append(f"contracts[{idx}].{field} must be a string")
    return issues


def validate_contracts_file(path: Path, *, strict: bool = False) -> Dict[str, Any]:
    """Load and validate contracts file, raising when strict."""
    data = load_contracts(path)
    issues = validate_contracts(data)
    if issues:
        message = "; ".join(issues)
        if strict:
            raise ValueError(f"Contracts validation failed: {message}")
        logger.warning("Contracts validation issues: %s", message)
    return data
