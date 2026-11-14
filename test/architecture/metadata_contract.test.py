"""
Metadata contract tests for pedosphere functors.

This file enforces explicit metadata declarations required by the project SSOT:
- Every FunctorMetadata(...) call in pedosphere modules must explicitly set
  version and execution_phase.
- NitrogenCycling must explicitly declare nitrogen conservation metadata.
"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PEDOSPHERE_ROOT = PROJECT_ROOT / "relic" / "scientific" / "pedosphere"
NITROGEN_FILE = PEDOSPHERE_ROOT / "nutrient_cycling" / "nitrogen.py"


def _iter_metadata_calls(py_file: Path) -> list[ast.Call]:
    tree = ast.parse(py_file.read_text(encoding="utf-8-sig"), filename=str(py_file))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "FunctorMetadata":
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == "FunctorMetadata":
            calls.append(node)
    return calls


def test_pedosphere_functor_metadata_has_explicit_version_and_execution_phase() -> None:
    missing_fields: list[str] = []

    for py_file in PEDOSPHERE_ROOT.rglob("*.py"):
        calls = _iter_metadata_calls(py_file)
        for call in calls:
            kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
            absent = [field for field in ("version", "execution_phase") if field not in kw_names]
            if absent:
                rel = py_file.relative_to(PROJECT_ROOT)
                missing_fields.append(f"{rel}:{call.lineno} missing {', '.join(absent)}")

    assert not missing_fields, "Missing metadata fields:\n" + "\n".join(missing_fields)


def test_nitrogen_cycling_metadata_declares_nitrogen_conservation() -> None:
    calls = _iter_metadata_calls(NITROGEN_FILE)
    assert calls, f"No FunctorMetadata call found in {NITROGEN_FILE}"

    conservation_node = None
    for kw in calls[0].keywords:
        if kw.arg == "conservation_laws":
            conservation_node = kw.value
            break

    assert conservation_node is not None, "NitrogenCycling metadata must define conservation_laws"
    assert isinstance(conservation_node, ast.List), "conservation_laws must be a list literal"

    values = []
    for elt in conservation_node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            values.append(elt.value.lower())

    assert "nitrogen" in values, "NitrogenCycling conservation_laws must include 'nitrogen'"
