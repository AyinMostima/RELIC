"""Audit test for strict scientific parameter access unification.

Hard rules:
1) Removed wrappers must not appear in relic/scientific/.
2) Functor classes must not call get_global_manager() directly.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = PROJECT_ROOT / "relic" / "scientific"

FORBIDDEN_PATTERNS = {
    "ensure_parameter_manager(": re.compile(r"\bensure_parameter_manager\s*\("),
    "require_parameter_value(": re.compile(r"\brequire_parameter_value\s*\("),
    "resolve_parameter_manager(": re.compile(r"\bresolve_parameter_manager\s*\("),
    "require_parameter_manager(": re.compile(r"\brequire_parameter_manager\s*\("),
}

GLOBAL_STAGE_BUDGET = 0


def _scan_occurrences() -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for file_path in sorted(SCIENTIFIC_ROOT.rglob("*.py")):
        rel = file_path.relative_to(PROJECT_ROOT).as_posix()
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for pattern_name, pattern in FORBIDDEN_PATTERNS.items():
            count = len(pattern.findall(text))
            if count > 0:
                rows.append((rel, pattern_name, count))
    return rows


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


def test_scientific_parameter_access_stage_budget() -> None:
    """Stage budget prevents new forbidden scientific access calls from growing."""
    rows = _scan_occurrences()
    total = sum(count for _, _, count in rows)
    assert total <= GLOBAL_STAGE_BUDGET, (
        f"Forbidden scientific parameter access calls exceed stage budget: "
        f"{total} > {GLOBAL_STAGE_BUDGET}\\n"
        f"Violations: {rows}"
    )


def test_ensure_parameter_manager_fully_removed() -> None:
    """Ensure ensure_parameter_manager is completely absent from scientific modules."""
    rows = _scan_occurrences()
    epm_rows = [(rel, count) for rel, pat, count in rows if pat == "ensure_parameter_manager("]
    assert not epm_rows, (
        f"ensure_parameter_manager() must not appear in relic/scientific/: {epm_rows}"
    )


def test_functor_classes_do_not_call_get_global_manager() -> None:
    """Non-Functor helpers may call get_global_manager; Functor classes may not."""
    offenders: list[tuple[str, int]] = []
    for file_path in sorted(SCIENTIFIC_ROOT.rglob("*.py")):
        rel = file_path.relative_to(PROJECT_ROOT).as_posix()
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_functor_like(node):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    if inner.func.id == "get_global_manager":
                        offenders.append((rel, inner.lineno))
    assert not offenders, f"Functor classes must not call get_global_manager(): {offenders}"
