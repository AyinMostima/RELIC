"""Contract tests for internal coupling dependency parsing."""

from __future__ import annotations

import pytest

from relic.coupling.internal.dependency_utils import (
    extract_dependency_overrides,
    resolve_dependencies,
    validate_dependency_names,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("dependencies", "expected"),
    [
        ("soil", ["soil"]),
        ([" soil ", "vegetation", "soil"], ["soil", "vegetation"]),
        (("atmosphere", "soil"), ["atmosphere", "soil"]),
        ([], []),
    ],
)
def test_validate_dependency_names_normalizes_and_stably_deduplicates(
    dependencies,
    expected,
) -> None:
    assert validate_dependency_names(dependencies) == expected


@pytest.mark.parametrize(
    ("dependencies", "error_type"),
    [
        (["soil", 3], TypeError),
        (["soil", None], TypeError),
        (["soil", "  "], ValueError),
    ],
)
def test_validate_dependency_names_rejects_invalid_entries(
    dependencies,
    error_type,
) -> None:
    with pytest.raises(error_type):
        validate_dependency_names(dependencies)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, []),
        ({}, []),
        ({"target": " soil "}, ["soil"]),
        (
            {
                "target": ["soil", None, "vegetation"],
                "targets": ("vegetation", "atmosphere"),
            },
            ["soil", "vegetation", "atmosphere"],
        ),
    ],
)
def test_extract_dependency_overrides_supports_alias_value_shapes(
    config,
    expected,
) -> None:
    assert extract_dependency_overrides(
        config,
        ("target", "targets"),
    ) == expected


@pytest.mark.parametrize(
    (
        "current_dependencies",
        "config",
        "expected",
        "explicit_from_user",
    ),
    [
        (
            ["current"],
            {"target": "configured"},
            ["current"],
            True,
        ),
        (
            [],
            {"target": "configured"},
            ["configured"],
            True,
        ),
        (
            None,
            {},
            ["default"],
            False,
        ),
        (
            "soil",
            {"target": "configured"},
            ["soil"],
            True,
        ),
    ],
)
def test_resolve_dependencies_applies_precedence_and_explicit_flag(
    current_dependencies,
    config,
    expected,
    explicit_from_user,
) -> None:
    dependencies, explicit = resolve_dependencies(
        current_dependencies=current_dependencies,
        config=config,
        default_target="default",
        override_keys=("target",),
    )

    assert dependencies == expected
    assert explicit is explicit_from_user
