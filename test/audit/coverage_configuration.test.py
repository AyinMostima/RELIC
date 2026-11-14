"""Audit the single native Coverage.py configuration owner."""

from __future__ import annotations

from pathlib import Path
import tomllib

from coverage import Coverage
import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_enables_relocatable_branch_coverage_for_relic() -> None:
    config = Coverage(config_file=True).config

    assert config.branch is True
    assert config.source is None
    assert [Path(path).resolve() for path in config.source_dirs] == [
        (ROOT / "relic").resolve()
    ]
    assert config.relative_files is True
    assert config.parallel is True


def test_report_defaults_expose_missing_branches_without_hiding_empty_files() -> None:
    config = Coverage(config_file=True).config

    assert config.show_missing is True
    assert config.skip_covered is False
    assert config.skip_empty is True
    assert config.precision == 2


def test_dev_extra_declares_native_coverage_toml_support() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
    assert "coverage[toml]>=7.10,<8" in dev_dependencies


def test_single_ci_workflow_owns_serial_full_suite_coverage_gate() -> None:
    workflow_paths = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    assert workflow_paths == [ROOT / ".github" / "workflows" / "ci.yml"]

    workflow_text = workflow_paths[0].read_text(encoding="utf-8")
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["serial-branch-coverage"]["steps"]
    uses = [step["uses"] for step in steps if "uses" in step]
    scripts = "\n".join(step.get("run", "") for step in steps)

    assert uses == [
        "actions/checkout@v6",
        "actions/setup-python@v6",
        "actions/upload-artifact@v7",
    ]
    assert "python-version" in steps[1]["with"]
    assert steps[1]["with"]["python-version"] == "3.10"
    assert "coverage run -m pytest" in scripts
    assert "-p no:pytest_cov -p no:xdist -q test" in scripts
    assert "coverage combine" in scripts
    assert "coverage report" in scripts
    assert "--cov" not in scripts
    assert "-n auto" not in scripts
