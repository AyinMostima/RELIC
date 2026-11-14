from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from relic.extensions.discovery import build_extension_info
from relic.extensions.params import (
    load_extension_parameters,
    save_extension_parameters,
)
from relic.extensions.protocol import MANIFEST_FILENAME
from relic.parameters.manager import UnifiedParameterManager


@pytest.fixture(autouse=True)
def _reset_manager_singleton(fresh_global_parameter_manager):
    """Each test runs on a fresh parameter manager; the previous one is put back."""
    yield


def _create_manager(tmp_path: Path) -> UnifiedParameterManager:
    manager = UnifiedParameterManager(
        database_path=tmp_path / "parameter_db",
        preload_all=False,
        warmup_cache=False,
    )
    manager.initialize()
    return manager


def _write_extension(
    root: Path,
    *,
    name: str,
    namespace_in_yaml: str | None,
    namespace_prefix: str | None = None,
    calibrated_output: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    parameter_payload = {
        "schema_version": "1.0.0",
        "tier": "extension",
        "description": "extension parameter file",
        "parameters": {
            "decomposition_rate": {
                "value": 1.0e-7,
                "units": "1/s",
                "description": "First-order decomposition rate constant",
                "bounds": [0.0, 1.0e-4],
                "dtype": "float32",
                "category": "biological_parameter",
                "references": ["Lloyd and Taylor (1994)"],
                "tags": ["extension"],
                "requires_grad": False,
                "provenance": {
                    "source": "unit_test_extension",
                    "schema_version": "1.0.0",
                },
            }
        },
    }
    if namespace_in_yaml is not None:
        parameter_payload["namespace"] = namespace_in_yaml

    (root / "parameters.yaml").write_text(
        yaml.safe_dump(parameter_payload, sort_keys=False),
        encoding="utf-8",
    )

    parameter_decl = {"path": "parameters.yaml"}
    if namespace_prefix is not None:
        parameter_decl["namespace_prefix"] = namespace_prefix

    manifest_payload = {
        "name": name,
        "version": "0.1.0",
        "author": "unit-test",
        "description": "extension params test",
        "min_relic_version": "1.0.0",
        "functors": [
            {
                "class": "functor.Dummy",
                "bridge": "TransportMechanismFunctor",
                "category": "transport_mechanism",
                "execution_group": "fast",
            }
        ],
        "parameters": [parameter_decl],
        "dependencies": {"required": [], "optional": []},
    }
    if calibrated_output is not None:
        manifest_payload["calibrated_output"] = calibrated_output

    (root / MANIFEST_FILENAME).write_text(
        yaml.safe_dump(manifest_payload, sort_keys=False),
        encoding="utf-8",
    )
    return root


def test_load_extension_parameters(tmp_path: Path) -> None:
    manager = _create_manager(tmp_path)
    extension_root = _write_extension(
        tmp_path / "ext_load",
        name="ext_load",
        namespace_in_yaml="extensions.ext_load",
    )
    info = build_extension_info(extension_root, source="directory")

    summary = load_extension_parameters(manager, info)

    assert summary["loaded_count"] >= 1
    value = manager.get("extensions.ext_load.decomposition_rate", required=True)
    assert float(value) == 1.0e-7


def test_save_extension_parameters_to_extension_dir(tmp_path: Path) -> None:
    manager = _create_manager(tmp_path)
    extension_root = _write_extension(
        tmp_path / "ext_save_default",
        name="ext_save_default",
        namespace_in_yaml="extensions.ext_save_default",
    )
    info = build_extension_info(extension_root, source="directory")
    load_extension_parameters(manager, info)

    output_path = save_extension_parameters(manager, info)

    assert output_path.parent == (extension_root / "calibrated").resolve()
    assert output_path.exists()
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert "parameters" in payload
    assert "extensions.ext_save_default.decomposition_rate" in payload["parameters"]


def test_custom_output_dir(tmp_path: Path) -> None:
    manager = _create_manager(tmp_path)
    extension_root = _write_extension(
        tmp_path / "ext_save_custom",
        name="ext_save_custom",
        namespace_in_yaml="extensions.ext_save_custom",
    )
    info = build_extension_info(extension_root, source="directory")
    load_extension_parameters(manager, info)

    custom_dir = tmp_path / "custom_calibrated_output"
    output_path = save_extension_parameters(manager, info, output_dir=custom_dir)

    assert output_path.parent == custom_dir.resolve()
    assert output_path.exists()


def test_namespace_prefix(tmp_path: Path) -> None:
    manager = _create_manager(tmp_path)
    extension_root = _write_extension(
        tmp_path / "ext_prefix",
        name="ext_prefix",
        namespace_in_yaml=None,
        namespace_prefix="extensions.prefixed",
    )
    info = build_extension_info(extension_root, source="directory")

    load_extension_parameters(manager, info)

    value = manager.get("extensions.prefixed.decomposition_rate", required=True)
    assert float(value) == 1.0e-7


def test_unreadable_namespace_source_error_propagates(tmp_path: Path) -> None:
    """A parameter file whose namespace cannot be read must not drop the namespace."""
    manager = _create_manager(tmp_path)
    extension_root = _write_extension(
        tmp_path / "ext_corrupt_namespace",
        name="ext_corrupt_namespace",
        namespace_in_yaml="extensions.custom_namespace",
    )
    info = build_extension_info(extension_root, source="directory")
    load_extension_parameters(manager, info)

    # Corrupt the declared parameter file after its parameters were loaded:
    # the bytes are not valid UTF-8, so the namespace cannot be read.
    (extension_root / "parameters.yaml").write_bytes(b"namespace: \xff\xfe extensions.custom\n")

    with pytest.raises(UnicodeDecodeError):
        save_extension_parameters(manager, info)


def test_no_core_pollution(tmp_path: Path) -> None:
    manager = _create_manager(tmp_path)
    extension_name = "ext_no_pollution_9d2a5"
    extension_root = _write_extension(
        tmp_path / extension_name,
        name=extension_name,
        namespace_in_yaml=f"extensions.{extension_name}",
    )
    info = build_extension_info(extension_root, source="directory")
    load_extension_parameters(manager, info)

    output_path = save_extension_parameters(manager, info)
    assert output_path.exists()

    project_root = Path(__file__).resolve().parents[3]
    core_calibrated_dir = project_root / "relic" / "parameters" / "database" / "calibrated"
    expected_core_output = core_calibrated_dir / f"{extension_name}_optimized_parameters.yaml"
    assert not expected_core_output.exists()
