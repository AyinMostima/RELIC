from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from relic.extensions.protocol import (
    MANIFEST_FILENAME,
    ManifestError,
    parse_manifest,
)


def _write_manifest(root: Path, payload: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_FILENAME
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _base_manifest() -> dict:
    return {
        "name": "demo_extension",
        "version": "0.1.0",
        "author": "tester",
        "description": "demo",
        "min_relic_version": "1.0.0",
        "functors": [
            {
                "class": "demo.functor.DemoFunctor",
                "bridge": "TransportMechanismFunctor",
                "category": "transport_mechanism",
                "execution_group": "fast",
            }
        ],
        "parameters": [{"path": "parameters.yaml"}],
        "dependencies": {"required": ["radiation"], "optional": ["phenology"]},
    }


def test_parse_manifest_valid(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, _base_manifest())
    manifest = parse_manifest(manifest_path)

    assert manifest.name == "demo_extension"
    assert manifest.version == "0.1.0"
    assert manifest.functors[0].class_path == "demo.functor.DemoFunctor"
    assert manifest.functors[0].bridge == "TransportMechanismFunctor"
    assert manifest.functors[0].execution_group == "fast"


def test_parse_manifest_missing_required_field(tmp_path: Path) -> None:
    payload = _base_manifest()
    payload.pop("name")
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(ManifestError):
        parse_manifest(manifest_path)


def test_parse_manifest_invalid_functor_declaration(tmp_path: Path) -> None:
    payload = _base_manifest()
    payload["functors"][0]["bridge"] = "UnknownBridge"
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(ManifestError):
        parse_manifest(manifest_path)


def test_parse_manifest_invalid_version_format(tmp_path: Path) -> None:
    payload = _base_manifest()
    payload["version"] = "1.0"
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(ManifestError):
        parse_manifest(manifest_path)



def test_manifest_execution_group_is_judged_by_the_simulator_predicate(
    tmp_path: Path, monkeypatch
) -> None:
    """The manifest has no execution-group set of its own: dependency_ops decides."""
    import relic.core.simulator.dependency_ops as dependency_ops
    import relic.extensions.protocol as protocol

    assert not hasattr(protocol, "_VALID_EXECUTION_GROUPS")
    monkeypatch.setattr(dependency_ops, "_is_valid_execution_group", lambda value: False)
    manifest_path = _write_manifest(tmp_path, _base_manifest())

    with pytest.raises(ManifestError, match="execution_group 'fast'"):
        parse_manifest(manifest_path)


def test_manifest_rejects_an_invalid_execution_group_with_the_runtime_message(
    tmp_path: Path,
) -> None:
    payload = _base_manifest()
    payload["functors"][0]["execution_group"] = "fastest"
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(ManifestError) as error:
        parse_manifest(manifest_path)

    message = str(error.value)
    assert "Invalid execution_group 'fastest' from" in message
    assert "expected one of ['fast', 'medium', 'slow']" in message


def test_manifest_functor_without_execution_group_is_rejected(tmp_path: Path) -> None:
    """The execution group is declared per functor; nothing is implied."""
    from relic.extensions.protocol import FunctorDeclaration

    payload = _base_manifest()
    payload["functors"][0].pop("execution_group")
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(ManifestError, match="execution_group"):
        parse_manifest(manifest_path)
    assert FunctorDeclaration.model_fields["execution_group"].is_required()
