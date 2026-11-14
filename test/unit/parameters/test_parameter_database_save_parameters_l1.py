"""
L1 tests for ParameterDatabase.save_parameters portability.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import yaml

from relic.parameters import ParameterDatabase, ParameterSpec


def test_save_parameters_writes_portable_yaml() -> None:
    """Ensure YAML output contains no python-specific tags and includes schema headers."""
    repo_root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(dir=str(repo_root)) as tmpdir:
        tmp_path = Path(tmpdir)
        db = ParameterDatabase(tmp_path)
        spec = ParameterSpec(
            key="test.saved.param",
            value=1.0,
            units="dimensionless",
            bounds=(0.0, 2.0),
            description="Test parameter for save_parameters portability",
            references=["RELIC ParameterManager persistence test"],
            dimension="dimensionless", dtype="float32", category="model_parameter",
        )

        db.save_parameters({"test.saved.param": spec}, tier="calibrated", filename="saved_parameters.yaml")

        saved_path = tmp_path / "calibrated" / "saved_parameters.yaml"
        text = saved_path.read_text(encoding="utf-8")
        assert "!!python/" not in text

        data = yaml.safe_load(text)
        assert isinstance(data, dict)
        assert data.get("schema_version") == "1.0.0"
        assert data.get("tier") == "calibrated"
        assert "namespace" in data
        assert "parameters" in data
        assert isinstance(data["parameters"]["test.saved.param"]["bounds"], list)


def test_save_parameters_rejects_incomplete_metadata() -> None:
    """Reject snapshots that would persist empty required metadata."""
    repo_root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(dir=str(repo_root)) as tmpdir:
        db = ParameterDatabase(Path(tmpdir))
        spec = ParameterSpec(
            key="test.incomplete.param",
            value=1.0,
            units="dimensionless",
            bounds=(0.0, 2.0), dtype="float32", category="model_parameter",
        )

        with pytest.raises(ValueError, match="incomplete persistence metadata"):
            db.save_parameters(
                {spec.key: spec},
                tier="calibrated",
                filename="saved_parameters.yaml",
            )
