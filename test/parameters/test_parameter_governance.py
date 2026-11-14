"""Tests for parameter governance extensions.

References:
- RELIC parameter SSOT governance requirements.
- Pydantic schema validation rules for parameter metadata.

Mathematical expression:
- Key canonicalization is a deterministic mapping f(old_key) = canonical_key.

Parameter meanings:
- aliases: legacy key list for a canonical parameter.
- canonical_name: SSOT key.
- deprecation: lifecycle metadata for migration.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from relic.parameters.canonicalize import canonicalize_keys, load_tombstones
from relic.parameters.schemas.parameter_schema import ParameterSpec


class TestParameterSpecGovernance:
    """Validate governance fields in ParameterSpec."""

    def test_governance_defaults(self) -> None:
        spec = ParameterSpec(
            value=1.0,
            units="kg",
            dtype="float64",
            bounds=[0.0, 10.0],
            category="test",
            description="test parameter",
            references=["RELIC parameter governance test"],
            tags=[],
            dimension="mass",
            requires_grad=False,
            provenance={},
        )
        assert spec.aliases == []
        assert spec.canonical_name is None
        # default_origin had no reader and no YAML declared it: the field is removed.
        assert "default_origin" not in ParameterSpec.model_fields
        assert spec.deprecation is None

    def test_governance_population(self) -> None:
        spec = ParameterSpec(
            value=2.0,
            units="kg",
            dtype="float64",
            bounds=[0.0, 10.0],
            category="test",
            description="test parameter",
            references=["RELIC parameter governance test"],
            tags=[],
            dimension="mass",
            requires_grad=False,
            provenance={},
            aliases=["legacy.mass.key"],
            canonical_name="relic.mass.key",
            deprecation={
                "deprecated_in": "2.0.0",
                "removed_in": None,
                "tombstone_to": "relic.mass.key",
            },
        )
        assert spec.aliases == ["legacy.mass.key"]
        assert spec.canonical_name == "relic.mass.key"
        assert isinstance(spec.deprecation, dict)

    @staticmethod
    def _valid_payload() -> dict:
        return {
            "value": 1.0,
            "units": "kg",
            "dtype": "float64",
            "bounds": [0.0, 10.0],
            "category": "test",
            "description": "test parameter",
            "references": ["RELIC parameter governance test"],
            "tags": [],
            "dimension": "mass",
            "requires_grad": False,
            "provenance": {},
        }

    def test_bounds_are_required(self) -> None:
        payload = self._valid_payload()
        payload.pop("bounds")

        with pytest.raises(ValidationError):
            ParameterSpec(**payload)

    @pytest.mark.parametrize("description", ["", "   "])
    def test_description_must_be_nonempty(self, description: str) -> None:
        payload = self._valid_payload()
        payload["description"] = description

        with pytest.raises(ValidationError):
            ParameterSpec(**payload)

    def test_references_must_be_nonempty(self) -> None:
        payload = self._valid_payload()
        payload["references"] = []

        with pytest.raises(ValidationError):
            ParameterSpec(**payload)


class TestCanonicalization:
    """Validate key canonicalization behavior."""

    def test_canonicalize_passthrough_when_no_tombstone(self) -> None:
        config = {"a": 1, "b": 2}
        normalized, warnings = canonicalize_keys(config, tombstones={})
        assert normalized == config
        assert warnings == []

    def test_canonicalize_rewrites_deprecated_key(self) -> None:
        config = {"old.key": 3.0, "other.key": 2.0}
        tombstones = {
            "old.key": {
                "canonical": "new.key",
                "deprecated_in": "2.0.0",
            }
        }
        normalized, warnings = canonicalize_keys(config, tombstones=tombstones)
        assert "old.key" not in normalized
        assert normalized["new.key"] == 3.0
        assert normalized["other.key"] == 2.0
        assert len(warnings) == 1
        assert "deprecated" in warnings[0].lower()

    def test_load_tombstones_schema_file(self) -> None:
        tombstone_path = (
            Path(__file__).resolve().parents[2]
            / "relic"
            / "parameters"
            / "schemas"
            / "tombstones.yaml"
        )
        tombstones = load_tombstones(tombstone_path)
        assert isinstance(tombstones, dict)
