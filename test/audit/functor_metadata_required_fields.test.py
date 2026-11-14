"""Audit gate for mandatory Functor metadata fields across scientific modules."""

from __future__ import annotations

import ast
from pathlib import Path

from test.audit.collect_functor_inventory import scan


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = REPO_ROOT / "relic" / "scientific"
REQUIRED_FIELDS = {"spatial_scale", "variable_kind"}


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _extract_functor_metadata_keyword_fields(class_node: ast.ClassDef) -> set[str]:
    fields: set[str] = set()
    for fn in [n for n in class_node.body if isinstance(n, ast.FunctionDef)]:
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call):
                continue
            call_name = _base_name(call.func)
            if call_name != "FunctorMetadata":
                continue
            fields.update({kw.arg for kw in call.keywords if kw.arg})
    return fields


def test_all_scientific_functors_declare_scale_and_variable_kind() -> None:
    violations: list[str] = []
    unavailable_pairs = {
        (row["file_path"], row["class_name"])
        for row in scan()
        if row.get("process_status") == "process"
        and row.get("implementation_status") == "not_implemented_fail_closed"
    }

    for path in SCIENTIFIC_ROOT.rglob("*.py"):
        src = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            bases = [_base_name(base) for base in cls.bases]
            if not any("Functor" in base for base in bases):
                continue
            relative_path = str(path.relative_to(REPO_ROOT)).replace("/", "\\")
            if (relative_path, cls.name) in unavailable_pairs:
                continue

            fields = _extract_functor_metadata_keyword_fields(cls)
            missing = sorted(REQUIRED_FIELDS - fields)
            if missing:
                violations.append(
                    f"{path.as_posix()}::{cls.name} missing metadata fields: {', '.join(missing)}"
                )

    assert not violations, "Found Functors missing metadata scale/kind fields:\n" + "\n".join(
        sorted(violations)
    )
