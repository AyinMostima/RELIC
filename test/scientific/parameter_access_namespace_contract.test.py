from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_FILES = [
    REPO_ROOT / "relic/scientific/anthroposphere/emissions/species_mapping.py",
    REPO_ROOT / "relic/scientific/anthroposphere/management/coordinator.py",
    REPO_ROOT / "relic/scientific/biosphere/growth/biomass_allocation.py",
]


def _contains_forbidden_global_manager_call(source: str) -> list[str]:
    violations: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "get_global_manager":
            violations.append(f"line {node.lineno}: get_global_manager(...)")
        if isinstance(func, ast.Attribute) and func.attr == "get_global_manager":
            violations.append(f"line {node.lineno}: {ast.unparse(func)}(...)")
    return violations


def test_scientific_modules_do_not_call_get_global_manager_directly():
    all_violations: list[str] = []
    for path in TARGET_FILES:
        source = path.read_text(encoding="utf-8-sig")
        violations = _contains_forbidden_global_manager_call(source)
        for violation in violations:
            all_violations.append(f"{path}: {violation}")
    assert not all_violations, (
        "Forbidden direct get_global_manager() calls found:\n" + "\n".join(all_violations)
    )
