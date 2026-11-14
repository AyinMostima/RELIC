"""L1 tests for parameter version records.

Module under test: relic/parameters/calibration/versioning.py. A version record
states the branch it was created on; the record carries no default branch.
"""

from __future__ import annotations

import dataclasses

import pytest

from relic.parameters.calibration.versioning import VersionMetadata


def test_version_metadata_branch_is_a_required_record_field() -> None:
    assert VersionMetadata.__dataclass_fields__["branch_name"].default is dataclasses.MISSING
    with pytest.raises(TypeError, match="branch_name"):
        VersionMetadata(version_id="v1")

    metadata = VersionMetadata(version_id="v1", branch_name="experiment")
    restored = VersionMetadata.from_dict(metadata.to_dict())
    assert restored.branch_name == "experiment"
