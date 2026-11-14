"""Shared dependency parsing helpers for internal coupling functors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def extract_dependency_overrides(
    config: dict[str, Any] | None,
    candidate_keys: Sequence[str],
) -> list[str]:
    """Collect dependency override names from configured keys."""
    if not isinstance(config, dict):
        return []
    overrides: list[str] = []
    for key in candidate_keys:
        if key not in config:
            continue
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            overrides.extend([item for item in value if item is not None])
        else:
            overrides.append(value)
    return validate_dependency_names(overrides)


def validate_dependency_names(dependencies: Iterable[Any]) -> list[str]:
    """Validate dependency names to unique, non-empty strings."""
    if isinstance(dependencies, str):
        dependencies = (dependencies,)
    validated: list[str] = []
    for dep in dependencies:
        if not isinstance(dep, str):
            raise TypeError(f"Dependency names must be strings, got {type(dep).__name__}")
        dep_name = dep.strip()
        if not dep_name:
            raise ValueError("Dependency names must be non-empty strings")
        if dep_name not in validated:
            validated.append(dep_name)
    return validated


def resolve_dependencies(
    *,
    current_dependencies: Iterable[Any] | None,
    config: dict[str, Any] | None,
    default_target: str,
    override_keys: Sequence[str],
) -> tuple[list[str], bool]:
    """
    Resolve final dependency list and whether it is user-defined.

    Returns
    -------
    Tuple[List[str], bool]
        Final dependency names and ``explicit_from_user`` flag.
    """
    explicit_from_user = False
    explicit_dependencies = validate_dependency_names(current_dependencies or [])
    if explicit_dependencies:
        explicit_from_user = True

    if not explicit_dependencies:
        overrides = extract_dependency_overrides(config, override_keys)
        if overrides:
            explicit_dependencies = overrides
            explicit_from_user = True

    if not explicit_dependencies:
        explicit_dependencies = validate_dependency_names([default_target])
        explicit_from_user = False

    return explicit_dependencies, explicit_from_user
