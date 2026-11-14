"""Configuration helpers for stochastic workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from relic.parameters import get_global_manager


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_file.suffix.lower() in [".yaml", ".yml"]:
        param_manager = get_global_manager()
        payload = param_manager.load_yaml_file(config_file, namespace="cli.stochastic")
    elif config_file.suffix.lower() == ".json":
        with open(config_file, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    else:
        raise ValueError(f"Unsupported config format: {config_file.suffix}")

    # An empty document declares no settings. Any other payload that is not
    # a mapping (a list or a scalar) is rejected instead of being read as an
    # empty configuration that silently selects every default.
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(
            f"Configuration file {config_path} must contain a mapping, got "
            f"{type(payload).__name__}"
        )
    return payload


def coerce_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize configuration into a plain dict."""
    return dict(config) if config else {}


def get_section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Extract a dict section; an absent section is empty, a non-mapping raises."""
    settings = coerce_config(config)
    section = settings.get(key)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise TypeError(
            f"Configuration section '{key}' must be a mapping, got {type(section).__name__}"
        )
    return section
