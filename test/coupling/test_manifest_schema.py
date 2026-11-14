"""Tests for coupling plugin manifest schema validation."""

from __future__ import annotations

from relic.coupling.contracts.manifest_validator import ManifestValidator
from relic.coupling.protocol import State
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog, SSOTEntry


def _minimal_valid_manifest() -> dict:
    return {
        "name": "echo_plugin",
        "interface_version": "1.0",
        "metadata": {"provenance": {"git": "abc123", "pkg": "echo-plugin-1.0.0"}},
        "ports": [
            {
                "name": "plugin_echo_value",
                "direction": "inout",
                "type_id": "F64",
                "shape_hint": [],
                "units": "dimensionless",
                "update_mode": "INCREMENT",
                "per_timestep": False,
            }
        ],
    }


def test_valid_manifest_passes() -> None:
    validator = ManifestValidator(catalog=SSOTCatalog())
    errors = validator.validate(_minimal_valid_manifest())
    assert errors == []


def test_missing_required_field_fails() -> None:
    manifest = _minimal_valid_manifest()
    manifest.pop("name")
    validator = ManifestValidator(catalog=SSOTCatalog())
    errors = validator.validate(manifest)
    assert errors
    assert any("name" in err for err in errors)


def test_invalid_type_id_fails() -> None:
    manifest = _minimal_valid_manifest()
    manifest["ports"][0]["type_id"] = "FLOAT128"
    validator = ManifestValidator(catalog=SSOTCatalog())
    errors = validator.validate(manifest)
    assert errors
    assert any("type_id" in err for err in errors)


def test_invalid_update_mode_fails() -> None:
    manifest = _minimal_valid_manifest()
    manifest["ports"][0]["update_mode"] = "UNKNOWN"
    validator = ManifestValidator(catalog=SSOTCatalog())
    errors = validator.validate(manifest)
    assert errors
    assert any("update_mode" in err for err in errors)


def test_state_manifest_emits_new_fields() -> None:
    state = State(name="plugin_state")
    port = PortSpec(
        name="plugin_echo_value",
        standard_name="plugin_echo_value",
        units="dimensionless",
        direction="inout",
        per_timestep=True,
        conservative=True,
    )
    state.add_port_spec(port, direction="export", data=1.0)
    manifest = state.manifest()
    assert manifest[0]["direction"] == "export"
    assert manifest[0]["per_timestep"] is True
    assert manifest[0]["conservative"] is True


def test_manifest_dimension_mismatch_fails() -> None:
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="air_temperature",
            standard_name="air_temperature",
            units="K",
        )
    )
    manifest = _minimal_valid_manifest()
    manifest["ports"][0].update(
        {
            "name": "air_temperature",
            "standard_name": "air_temperature",
            "units": "W m-2",
        }
    )
    validator = ManifestValidator(catalog=catalog)
    errors = validator.validate(manifest)
    assert any("dimensionally incompatible" in err for err in errors)


def test_manifest_ssot_checks_do_not_silently_skip_a_failed_catalog(monkeypatch):
    """A catalog that fails to load raises; the SSOT consistency checks are never skipped."""
    import pytest

    from relic.coupling.contracts import manifest_validator
    from relic.orchestration.ssot_catalog import SSOTCatalog

    def failing(*args, **kwargs):
        raise ValueError("catalog conflict")

    monkeypatch.setattr(SSOTCatalog, "from_parameter_manager", classmethod(lambda cls, *a, **k: failing()))
    validator = manifest_validator.ManifestValidator()
    with pytest.raises(ValueError, match="catalog conflict"):
        validator._resolve_catalog()
