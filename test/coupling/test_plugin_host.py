"""Tests for C ABI plugin host and functor adapter."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List, Sequence, Tuple

import numpy as np
import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateMode
from relic.coupling.plugin import PluginFunctorAdapter, PluginHost
from relic.coupling.contracts.manifest_validator import ManifestValidator
from relic.orchestration.ssot_catalog import SSOTCatalog


# A plugin adapter declares the scales it couples (required config keys, MA-21 item 202).
PLUGIN_SCALES = {"source_scale": "external_grid", "target_scale": "grid_cell", "coupling_type": "spatial"}


def _shared_lib_suffix() -> str:
    if os.name == "nt":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _build_commands(source_path: Path, output_path: Path) -> List[Sequence[str]]:
    return [
        ("gcc", "-shared", "-O2", "-std=c99", "-o", str(output_path), str(source_path)),
        ("clang", "-shared", "-O2", "-std=c99", "-o", str(output_path), str(source_path)),
    ]


def _compile_shared_library(source_path: Path, output_path: Path) -> Tuple[bool, str]:
    commands = _build_commands(source_path, output_path)
    errors: List[str] = []
    for command in commands:
        compiler = str(command[0])
        if shutil.which(compiler) is None:
            errors.append(f"{compiler}: not found")
            continue
        try:
            subprocess.run(
                list(command),
                check=True,
                capture_output=True,
                text=True,
            )
            return True, compiler
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            errors.append(f"{compiler}: {stderr}")
    return False, "; ".join(errors)


@pytest.fixture(scope="session")
def echo_plugin_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    source_path = repo_root / "examples" / "plugins" / "echo_plugin.c"
    output_path = repo_root / "examples" / "plugins" / f"echo_plugin{_shared_lib_suffix()}"
    if output_path.exists():
        return output_path

    ok, reason = _compile_shared_library(source_path, output_path)
    if not ok:
        pytest.skip(f"Cannot compile echo plugin: {reason}")
    return output_path


@pytest.fixture(scope="session")
def invalid_plugin_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("plugin_invalid")
    source_path = tmp_dir / "missing_symbol_plugin.c"
    source_path.write_text(
        "\n".join(
            [
                "#include <stddef.h>",
                "int plugin_probe(char* out, size_t len) { (void)out; (void)len; return 0; }",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_dir / f"missing_symbol_plugin{_shared_lib_suffix()}"
    ok, reason = _compile_shared_library(source_path, output_path)
    if not ok:
        pytest.skip(f"Cannot compile invalid plugin fixture: {reason}")
    return output_path


def test_probe_returns_valid_manifest(echo_plugin_path: Path) -> None:
    host = PluginHost(str(echo_plugin_path))
    manifest = host.probe()
    validator = ManifestValidator(catalog=SSOTCatalog())
    errors = validator.validate(manifest)
    assert errors == []


def test_negotiate_accepts_host_caps(echo_plugin_path: Path) -> None:
    host = PluginHost(str(echo_plugin_path))
    manifest = host.negotiate({"zero_copy": True, "interface_version": "1.0"})
    assert manifest["name"] == "echo_plugin"
    assert manifest["interface_version"] == "1.0"


def test_exchange_identity(echo_plugin_path: Path) -> None:
    host = PluginHost(str(echo_plugin_path))
    values = np.asarray([42.5], dtype=np.float64)
    rc = host.exchange("plugin_echo_value", values, timestamp=0.0)
    assert rc == 0
    assert pytest.approx(float(values[0]), rel=1e-12) == 42.5


def test_plugin_adapter_as_functor(echo_plugin_path: Path) -> None:
    adapter = PluginFunctorAdapter(
        name="echo_adapter", plugin_path=str(echo_plugin_path), config=dict(PLUGIN_SCALES)
    )
    assert any(
        base.__module__ == "relic.core.functor.scale_coupling_bridge"
        and base.__name__ == "ScaleCouplingFunctor"
        for base in type(adapter).__mro__
    )
    updates = adapter.forward(
        {"plugin_echo_value": torch.tensor(7.0, dtype=torch.float64)},
        time=0.0,
        dt=1.0,
    )
    assert "plugin_echo_value" in updates
    item = updates["plugin_echo_value"]
    assert item.mode == UpdateMode.ACCUMULATE
    if isinstance(item.value, torch.Tensor):
        assert pytest.approx(float(item.value.item()), rel=1e-12) == 7.0
    else:
        assert pytest.approx(float(item.value), rel=1e-12) == 7.0

    simulator = RELICSimulator({"conservation_checks": False})
    simulator.register_functor(adapter)
    assert simulator.functors[adapter.name] is adapter


def test_plugin_adapter_keeps_instance_declared_dependencies(tmp_path: Path) -> None:
    """A plugin adapter must not drop add_dependency/config declarations.

    The adapter has no static Functor dependency; the bridge default returns
    exactly the instance declaration, so graph links attached by GraphExecutor
    (set_dependencies) and explicit add_dependency calls reach the simulator.
    """
    adapter = PluginFunctorAdapter(
        name="echo_adapter_dependencies",
        plugin_path=str(tmp_path / f"unused_plugin{_shared_lib_suffix()}"),
        config={"dependencies": ["upstream_provider"], **PLUGIN_SCALES},
    )
    assert adapter.get_dependencies() == ["upstream_provider"]
    adapter.add_dependency("probe_dep")
    assert adapter.get_dependencies() == ["upstream_provider", "probe_dep"]


def test_plugin_adapter_missing_required_input_raises_key_error(echo_plugin_path: Path) -> None:
    adapter = PluginFunctorAdapter(
        name="echo_adapter_missing_input",
        plugin_path=str(echo_plugin_path),
        config=dict(PLUGIN_SCALES),
    )
    with pytest.raises(KeyError, match="Missing required plugin input port"):
        adapter.forward({}, time=0.0, dt=1.0)


def test_missing_symbol_raises(invalid_plugin_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Missing required plugin symbol"):
        PluginHost(str(invalid_plugin_path))
