"""Audit gate for conservation_laws policy on scientific Functors."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = REPO_ROOT / "relic" / "scientific"
EMPTY_LAW_ALLOWLIST = {
    "LandUseParameterMapping",
    "PedosphereInterfaceMapper",
    # Reads atmospheric O3 as forcing and advances no ozone stock: the uptake is
    # partitioned into REPLACE rate outputs, so no closed ledger exists (MA-21 12.1).
    "OzoneStress",
}
EMPTY_KIND_ALLOWLIST = {
    "VariableKind.DIAGNOSTIC",
    "VariableKind.FORCING",
}


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def test_empty_conservation_laws_follow_policy() -> None:
    violations: list[str] = []

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

            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                for call in ast.walk(fn):
                    if not isinstance(call, ast.Call) or _call_name(call.func) != "FunctorMetadata":
                        continue

                    conservation_value = None
                    variable_kind = None
                    for kw in call.keywords:
                        if kw.arg == "conservation_laws":
                            conservation_value = kw.value
                        if kw.arg == "variable_kind":
                            variable_kind = ast.unparse(kw.value)

                    if conservation_value is None:
                        continue
                    if not isinstance(conservation_value, (ast.List, ast.Tuple)):
                        continue
                    if len(conservation_value.elts) != 0:
                        continue

                    if cls.name in EMPTY_LAW_ALLOWLIST:
                        continue
                    if variable_kind in EMPTY_KIND_ALLOWLIST:
                        continue

                    violations.append(
                        f"{path.as_posix()}::{cls.name} has empty conservation_laws with variable_kind={variable_kind}"
                    )

    assert not violations, "Empty conservation_laws policy violations:\n" + "\n".join(
        sorted(violations)
    )
