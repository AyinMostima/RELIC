"""Parameter key normalization utilities for RELIC.

This module enforces canonical parameter keys. Alias mapping is not applied.
"""

from __future__ import annotations


def normalize_parameter_key(key: str) -> str:
    """Return canonical key without alias mapping."""
    if not isinstance(key, str):
        return key
    return key.strip()


__all__ = ["normalize_parameter_key"]
