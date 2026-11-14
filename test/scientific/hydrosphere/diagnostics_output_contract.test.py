"""Hydrosphere diagnostics output keys must be declared in Functor metadata."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HYDRO_ROOT = REPO_ROOT / "relic" / "scientific" / "hydrosphere"


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _collect_declared_outputs(class_node: ast.ClassDef) -> set[str]:
    declared: set[str] = set()
    for fn in [n for n in class_node.body if isinstance(n, ast.FunctionDef)]:
        if fn.name != "__init__":
            continue
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call) or _base_name(call.func) != "FunctorMetadata":
                continue
            for kw in call.keywords:
                if kw.arg != "output_variables" or not isinstance(
                    kw.value, (ast.List, ast.Tuple)
                ):
                    continue
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        declared.add(elt.value)
    return declared


def _collect_forward_diagnostic_keys(class_node: ast.ClassDef) -> set[str]:
    returned: set[str] = set()
    for fn in [n for n in class_node.body if isinstance(n, ast.FunctionDef)]:
        if fn.name != "forward":
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Constant):
                continue
            if isinstance(node.value, str) and node.value.startswith("diagnostics."):
                returned.add(node.value)
    return returned


def test_hydrosphere_forward_diagnostics_are_declared_in_metadata() -> None:
    violations: list[str] = []

    for path in HYDRO_ROOT.rglob("*.py"):
        src = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            bases = [_base_name(base) for base in cls.bases]
            if not any("Functor" in base for base in bases):
                continue

            declared = _collect_declared_outputs(cls)
            diagnostics_in_forward = _collect_forward_diagnostic_keys(cls)
            missing = sorted(diagnostics_in_forward - declared)
            if missing:
                violations.append(
                    f"{path.as_posix()}::{cls.name} missing diagnostics declarations: "
                    + ", ".join(missing)
                )

    assert not violations, (
        "Found hydrosphere diagnostics keys returned by forward() "
        "but not declared in metadata:\n" + "\n".join(sorted(violations))
    )

