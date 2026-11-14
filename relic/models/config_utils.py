"""Shared configuration helpers for model modules."""

from __future__ import annotations

from typing import Any, Dict


def coerce_config_dict(config: Any) -> Dict[str, Any]:
    """Normalize heterogeneous config objects into a plain dictionary.

    None (an absent section) is the empty mapping. A value that is not a
    mapping raises TypeError; it is never replaced by an empty config,
    which would silently select every default.
    """
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "_original_config"):
        original = getattr(config, "_original_config")
        if isinstance(original, dict):
            return dict(original)
        raise TypeError(
            "config._original_config must be a mapping, got "
            f"{type(original).__name__}"
        )
    try:
        return dict(config)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"config must be a mapping, got {type(config).__name__}"
        ) from exc


__all__ = ["coerce_config_dict"]
