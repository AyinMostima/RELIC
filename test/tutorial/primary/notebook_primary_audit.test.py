"""Regression tests for the primary-notebook audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = PROJECT_ROOT / "tools" / "notebook_primary_audit.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("notebook_primary_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load notebook_primary_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_notebook(path: Path, source: str) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Learning objective: verify imports."),
            nbformat.v4.new_code_cell(
                source,
                outputs=[
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stdout",
                        text="[OK] imported\n",
                    )
                ],
                execution_count=1,
            ),
        ]
    )
    nbformat.write(notebook, path)


def test_registry_label_is_not_misread_as_try_import(tmp_path: Path) -> None:
    notebook_path = tmp_path / "registry_label.ipynb"
    _write_notebook(
        notebook_path,
        "# Service Registry: canonical owner\nfrom relic.core import RELICSimulator\n",
    )

    result = _load_audit_module().analyze_notebook(notebook_path)

    assert result["try_except_import"] is False


def test_actual_try_import_is_detected(tmp_path: Path) -> None:
    notebook_path = tmp_path / "try_import.ipynb"
    _write_notebook(
        notebook_path,
        "try:\n    import optional_package\nexcept ImportError:\n    optional_package = None\n",
    )

    result = _load_audit_module().analyze_notebook(notebook_path)

    assert result["try_except_import"] is True
