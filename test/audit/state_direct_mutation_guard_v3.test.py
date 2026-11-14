"""
Audit gate to prevent in-place mutation of input state dictionaries.
"""

from __future__ import annotations

import ast
from pathlib import Path


_SCIENTIFIC_ROOT = Path("relic/scientific")


def _is_state_subscript(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    return isinstance(node.value, ast.Name) and node.value.id == "state"


def _scan_file(path: Path) -> list[str]:
    issues: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return issues

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if _is_state_subscript(target):
                    issues.append(f"{path.as_posix()}:{node.lineno}")
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            if _is_state_subscript(node.target):
                issues.append(f"{path.as_posix()}:{node.lineno}")
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if _is_state_subscript(node.target):
                issues.append(f"{path.as_posix()}:{node.lineno}")
            self.generic_visit(node)

    Visitor().visit(tree)
    return issues


def test_scientific_modules_do_not_mutate_input_state_in_place() -> None:
    """
    References:
    - docs/source/architecture_overview.md (FluxDict update contract).
    - RELIC state update policy via StateManager.apply_updates.

    Mathematical expression:
    - state_{n+1} = apply_updates(state_n, FluxDict), not in-place state[key] = value in functors.

    Parameter meanings:
    - state: input state dictionary passed into forward().
    - FluxDict: returned update mapping consumed by StateManager.
    """
    violations: list[str] = []
    for py_file in _SCIENTIFIC_ROOT.rglob("*.py"):
        violations.extend(_scan_file(py_file))

    assert not violations, (
        "Found in-place writes to input `state[...]` in scientific modules.\n"
        + "\n".join(violations)
    )
