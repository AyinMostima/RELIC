"""
Output overlap governance helpers.

This module is the single runtime source of truth for overlap policy
classification used by simulator registration and execution.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


OVERLAP_POLICY_AGGREGATED = "aggregated"
OVERLAP_POLICY_MUTUALLY_EXCLUSIVE = "mutually_exclusive"

_ALLOWED_POLICIES = {
    OVERLAP_POLICY_AGGREGATED,
    OVERLAP_POLICY_MUTUALLY_EXCLUSIVE,
}


def _policy_file_path() -> Path:
    return Path(__file__).with_name("output_overlap_whitelist.json")


@lru_cache(maxsize=1)
def load_output_overlap_policies() -> Dict[str, Dict[str, Any]]:
    """
    Load overlap policy entries from canonical JSON.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from variable name to validated policy entry.
    """
    path = _policy_file_path()
    if not path.exists():
        raise FileNotFoundError(f"Output overlap policy file not found: {path}")

    # The canonical file is read completely or the load fails: a read or JSON
    # error propagates and every entry must be a valid declaration.
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the overlap policy file must hold a mapping of variables")

    parsed: Dict[str, Dict[str, Any]] = {}
    for variable, entry in raw.items():
        if not isinstance(variable, str) or not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {variable!r} must map a variable name to a mapping")
        policy = entry.get("policy")
        reason = entry.get("reason")
        if policy not in _ALLOWED_POLICIES:
            raise ValueError(
                f"{path}: entry {variable!r} has policy {policy!r}; expected one of {sorted(_ALLOWED_POLICIES)}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: entry {variable!r} needs a non-empty reason")
        normalized: Dict[str, Any] = {"policy": policy, "reason": reason}
        parsed[variable] = normalized

    return parsed


def get_output_overlap_policy(variable: str) -> Optional[str]:
    """Return policy name for a variable, or None when unmanaged."""
    if not isinstance(variable, str):
        return None
    return load_output_overlap_policies().get(variable, {}).get("policy")


def is_governed_output_overlap(variable: str) -> bool:
    """Return True if variable has a valid overlap governance entry."""
    return get_output_overlap_policy(variable) in _ALLOWED_POLICIES
