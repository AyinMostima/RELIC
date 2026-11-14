"""Parameter-access lint checks for scientific modules."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = REPO_ROOT / "relic" / "scientific"


def _iter_scientific_python_files() -> Iterable[Path]:
    return sorted(SCIENTIFIC_ROOT.rglob("*.py"))


def _is_functor_like(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            base_name = base.id
        elif isinstance(base, ast.Attribute):
            base_name = base.attr
        else:
            continue
        if base_name.endswith("Functor") or base_name == "FunctorBase":
            return True
    return False


def test_functor_classes_do_not_call_require_parameter_value() -> None:
    offenders: list[str] = []

    for path in _iter_scientific_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_functor_like(node):
                continue

            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "require_parameter_value"
                ):
                    offenders.append(f"{path}:{inner.lineno}")

    assert not offenders, (
        "Functor classes must use FunctorBase parameter APIs instead of "
        "standalone require_parameter_value(). Offenders:\n"
        + "\n".join(offenders)
    )


def test_functor_classes_do_not_call_get_global_manager_directly() -> None:
    offenders: list[str] = []

    for path in _iter_scientific_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_functor_like(node):
                continue

            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                if isinstance(inner.func, ast.Name) and inner.func.id == "get_global_manager":
                    offenders.append(f"{path}:{inner.lineno}")

    assert not offenders, (
        "Functor classes must not call get_global_manager() directly. "
        "Use self.get_parameter_value(..., required=...) in Functor code.\n"
        + "\n".join(offenders)
    )


def test_functor_classes_do_not_override_require_parameter() -> None:
    offenders: list[str] = []

    for path in _iter_scientific_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_functor_like(node):
                continue

            has_override = any(
                isinstance(inner, ast.FunctionDef) and inner.name == "_require_parameter"
                for inner in node.body
            )
            if not has_override:
                continue
            offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "Functor classes must not define _require_parameter. "
        "Use self.get_parameter_value(..., required=...) directly.\n"
        + "\n".join(offenders)
    )
