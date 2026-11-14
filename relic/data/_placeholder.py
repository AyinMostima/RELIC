"""Shared helpers for placeholder subpackages."""

from __future__ import annotations

import os
from typing import Iterable, List


class ExperimentalFeatureError(RuntimeError):
    """Raised when a placeholder module is used without explicit experimental opt-in."""


def list_placeholder_exports(exports: Iterable[str]) -> List[str]:
    """Return a stable list view for placeholder package exports."""
    return list(exports)


def require_experimental_feature(
    *,
    feature_name: str,
    env_var: str,
    module_name: str,
) -> None:
    """Enforce explicit environment opt-in before importing placeholder modules."""
    enabled = os.getenv(env_var, "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        raise ExperimentalFeatureError(
            f"{module_name} is experimental placeholder code. "
            f"Set {env_var}=1 to import {feature_name}."
        )


__all__ = ["ExperimentalFeatureError", "list_placeholder_exports", "require_experimental_feature"]
