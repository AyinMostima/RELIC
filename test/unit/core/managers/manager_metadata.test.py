"""L1 tests for manager metadata version checks.

Module under test: relic/core/managers/metadata.py. An unparseable version or
version constraint is an invalid metadata state and raises; it must not skip
the check or report a dependency as satisfied.
"""

from __future__ import annotations

import dataclasses

import pytest

from relic.core.managers.metadata import (
    CapabilityType,
    DependencyType,
    ManagerCapability,
    ManagerDependency,
    ManagerMetadata,
    ManagerResource,
    ResourceType,
)


def _metadata(name: str, version: str) -> ManagerMetadata:
    return ManagerMetadata(name=name, version=version)


class TestCapabilityVersion:
    def test_invalid_capability_version_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="not-a-version"):
            ManagerCapability(type=CapabilityType.VALIDATION, version="not-a-version")

    def test_unparseable_version_does_not_skip_the_major_version_check(self) -> None:
        provided = ManagerCapability(type=CapabilityType.VALIDATION)
        required = ManagerCapability(type=CapabilityType.VALIDATION)
        provided.version = "broken"

        with pytest.raises(ValueError, match="broken"):
            provided.matches_requirement(required)

    def test_major_version_mismatch_does_not_match(self) -> None:
        provided = ManagerCapability(type=CapabilityType.VALIDATION, version="2.0.0")
        required = ManagerCapability(type=CapabilityType.VALIDATION, version="1.4.0")

        assert provided.matches_requirement(required) is False


class TestDependencyConstraint:
    def test_invalid_constraint_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="grid_manager"):
            ManagerDependency(
                target="grid_manager",
                type=DependencyType.REQUIRED,
                version_constraint=">=bad",
            )

    @pytest.mark.parametrize(
        ("constraint", "target_version", "expected"),
        [
            (">=1.0.0", "1.2.0", True),
            (">=1.0.0", "0.9.0", False),
            ("==2.1.0", "2.1.0", True),
            ("==2.1.0", "2.1.1", False),
            ("<2.0", "2.1.0", False),
            (">=1.0,<2.0", "1.5.0", True),
        ],
    )
    def test_constraint_is_evaluated_not_assumed_satisfied(
        self,
        constraint: str,
        target_version: str,
        expected: bool,
    ) -> None:
        dependency = ManagerDependency(
            target="grid_manager",
            type=DependencyType.REQUIRED,
            version_constraint=constraint,
        )
        available = {"grid_manager": _metadata("grid_manager", target_version)}

        assert dependency.is_satisfied_by(available) is expected

    def test_missing_optional_target_is_satisfied_and_required_is_not(self) -> None:
        optional = ManagerDependency(target="absent", type=DependencyType.OPTIONAL, optional=True)
        required = ManagerDependency(target="absent", type=DependencyType.REQUIRED)

        assert optional.is_satisfied_by({}) is True
        assert required.is_satisfied_by({}) is False


class TestInteroperability:
    def test_unparseable_version_does_not_skip_interoperability_check(self) -> None:
        first = _metadata("first", "1.0.0")
        second = _metadata("second", "1.0.0")
        second.version = "broken"

        with pytest.raises(ValueError, match="broken"):
            first.check_interoperability(second)

    def test_major_version_mismatch_is_reported(self) -> None:
        interoperable, issues = _metadata("first", "1.0.0").check_interoperability(
            _metadata("second", "2.0.0")
        )

        assert interoperable is False
        assert any("Major version" in issue for issue in issues)


class TestLiteralDefaults:
    def test_resource_unit_has_no_placeholder_default(self) -> None:
        # The unit of a resource quantity (bytes, MB, ...) has no universal value.
        assert ManagerResource.__dataclass_fields__["unit"].default is dataclasses.MISSING
        with pytest.raises(TypeError, match="unit"):
            ManagerResource(type=ResourceType.SYSTEM_MEMORY)
        resource = ManagerResource(type=ResourceType.SYSTEM_MEMORY, unit="bytes")
        assert ManagerResource.from_dict(resource.to_dict()).unit == "bytes"

    def test_dependency_carries_no_unread_initialization_order(self) -> None:
        dependency = ManagerDependency(target="grid_manager", type=DependencyType.REQUIRED)
        assert "initialization_order" not in ManagerDependency.__dataclass_fields__
        assert "initialization_order" not in dependency.to_dict()
        assert ManagerDependency.from_dict(dependency.to_dict()) == dependency
