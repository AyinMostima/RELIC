"""The extension scaffold states the bridge and the execution group explicitly."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from relic.extensions import cli as ext_cli
from relic.extensions.protocol import MANIFEST_FILENAME


def _scaffold(tmp_path: Path, *extra: str) -> int:
    return ext_cli.main(["scaffold", "demo_ext", "--output", str(tmp_path / "demo_ext"), *extra])


@pytest.mark.parametrize(
    "argv",
    [
        ["--execution-group", "fast"],
        ["--bridge", "biological_process"],
    ],
)
def test_scaffold_requires_bridge_and_execution_group(tmp_path, argv):
    with pytest.raises(SystemExit) as exc_info:
        _scaffold(tmp_path, *argv)

    assert exc_info.value.code == 2
    assert not (tmp_path / "demo_ext").exists()


def test_scaffold_writes_the_stated_execution_group(tmp_path):
    """The group is the one given, not one implied by the bridge class."""
    assert _scaffold(tmp_path, "--bridge", "biological_process", "--execution-group", "fast") == 0

    manifest = yaml.safe_load((tmp_path / "demo_ext" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["functors"][0]["bridge"] == "BiologicalProcessFunctor"
    assert manifest["functors"][0]["execution_group"] == "fast"


def test_scaffold_rejects_an_unknown_execution_group(tmp_path, capsys):
    assert _scaffold(tmp_path, "--bridge", "physical_field", "--execution-group", "fastest") == 1

    assert "fastest" in capsys.readouterr().out
    assert not (tmp_path / "demo_ext").exists()


def test_scaffold_has_no_bridge_to_group_mapping():
    assert not hasattr(ext_cli, "_EXECUTION_GROUP_MAP")
