"""Parameter key canonicalization utility.

This module maps deprecated parameter keys to canonical keys using
`relic\\parameters\\schemas\\tombstones.yaml`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_tombstones(tombstone_path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load tombstone mapping from YAML.

    Parameters
    ----------
    tombstone_path : Path | None
        Optional explicit path to tombstone YAML.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from deprecated key to metadata with canonical key.
    """
    if tombstone_path is None:
        tombstone_path = Path(__file__).resolve().parent / "schemas" / "tombstones.yaml"
    if not tombstone_path.exists():
        return {}
    with tombstone_path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle) or {}
    tombstones = data.get("tombstones", {})
    if not isinstance(tombstones, dict):
        return {}
    return tombstones


def canonicalize_keys(
    config: Dict[str, Any],
    tombstones: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str]]:
    """Canonicalize top-level keys in a config mapping.

    Parameters
    ----------
    config : Dict[str, Any]
        Parameter mapping to normalize.
    tombstones : Dict[str, Dict[str, Any]]
        Tombstone lookup.

    Returns
    -------
    Tuple[Dict[str, Any], List[str]]
        Canonicalized mapping and warning messages.
    """
    normalized: Dict[str, Any] = {}
    warnings: List[str] = []
    for key, value in config.items():
        if key in tombstones:
            entry = tombstones.get(key, {}) or {}
            canonical = str(entry.get("canonical", key))
            deprecated_in = str(entry.get("deprecated_in", "?"))
            warnings.append(
                f"[WARNING] Parameter '{key}' is deprecated since {deprecated_in}. Use '{canonical}'."
            )
            normalized[canonical] = value
        else:
            normalized[key] = value
    return normalized, warnings


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonicalize deprecated parameter keys.")
    parser.add_argument("config", help="Input YAML config file path.")
    parser.add_argument(
        "--tombstones",
        default=None,
        help="Optional tombstone YAML path. Defaults to relic/parameters/schemas/tombstones.yaml.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_cli()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] File not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8-sig") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        print("[ERROR] Input YAML root must be a mapping.", file=sys.stderr)
        return 1

    tombstone_path = Path(args.tombstones) if args.tombstones else None
    tombstones = load_tombstones(tombstone_path)
    canonicalized, warnings = canonicalize_keys(loaded, tombstones)

    for message in warnings:
        print(message, file=sys.stderr)
    yaml.safe_dump(canonicalized, stream=sys.stdout, sort_keys=True, allow_unicode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
