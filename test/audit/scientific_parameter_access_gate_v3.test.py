"""
Audit gate for scientific parameter access layering.

This gate enforces the 2026-02 architecture rule:
- Scientific Functor classes must use self.get_parameter_value(...)
- get_global_manager().get(...) is only allowed in non-Functor utilities/infrastructure
"""

from __future__ import annotations

import ast
from pathlib import Path


_SCIENTIFIC_ROOT = Path("relic/scientific")
_FUNCTOR_BASE_TOKENS = (
    "Functor",
    "BiologicalProcess",
    "TransportMechanism",
    "PhysicalField",
    "ScaleCoupling",
)
_REMOVED_PARAMETER_ACCESS_TOKENS = (
    "_require_parameter",
    "get_parameter_by_path",
    "require_parameter_value",
    "resolve_parameter_manager",
    "require_parameter_manager",
    "ParameterAccessorMixin",
    "manager.get_parameter_value",
)


def _is_functor_like_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        base_text = ast.unparse(base) if hasattr(ast, "unparse") else ""
        if any(token in base_text for token in _FUNCTOR_BASE_TOKENS):
            return True
    return False


def _scan_file_for_functor_global_manager_calls(path: Path) -> list[str]:
    issues: list[str] = []
    try:
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
    except OSError:
        return [f"{path.as_posix()}:unreadable"]
    except SyntaxError as exc:
        return [f"{path.as_posix()}:{exc.lineno}:syntax error: {exc.msg}"]

    class_stack: list[tuple[str, bool]] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            class_stack.append((node.name, _is_functor_like_class(node)))
            self.generic_visit(node)
            class_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if class_stack and class_stack[-1][1]:
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "get":
                    inner = func.value
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == "get_global_manager"
                    ):
                        class_name = class_stack[-1][0]
                        issues.append(
                            f"{path.as_posix()}:{node.lineno} in class {class_name}"
                        )
            self.generic_visit(node)

    Visitor().visit(tree)
    return issues


def test_scientific_functors_do_not_call_get_global_manager_directly() -> None:
    """
    References:
    - docs/source/architecture_overview.md, parameter access hard lock (2026-02).

    Mathematical expression:
    - p = PM.resolve(namespace.key, source_order), accessed through self.get_parameter_value(...).

    Parameter meanings:
    - PM: ParameterManager.
    - namespace.key: canonical parameter path in YAML SSOT.
    - source_order: ordered resolution chain in parameter system.
    """
    violations: list[str] = []
    for py_file in _SCIENTIFIC_ROOT.rglob("*.py"):
        violations.extend(_scan_file_for_functor_global_manager_calls(py_file))

    assert not violations, (
        "Functor-level get_global_manager().get(...) calls are forbidden.\n"
        + "\n".join(violations)
    )


def test_scientific_modules_do_not_reintroduce_removed_parameter_accessors() -> None:
    """
    References:
    - docs/source/architecture_overview.md, parameter access single-entry hard lock.
    - docs/source/BIHSS_LaTeX_source.tex, strict BIHSS audit implementation gates.

    Mathematical expression:
    - theta = PM.get(namespace.key) for utility code, and theta = self.get_parameter_value(...)
      inside Functor code.

    Parameter meanings:
    - theta: model parameter value from YAML SSOT.
    - PM: global ParameterManager used only outside Functor runtime methods.
    - namespace.key: canonical parameter path.
    """
    violations: list[str] = []
    for py_file in sorted(_SCIENTIFIC_ROOT.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8-sig")
        for token in _REMOVED_PARAMETER_ACCESS_TOKENS:
            if token in source:
                violations.append(f"{py_file.as_posix()}: contains {token}")
        if "from relic.parameters import get as" in source:
            violations.append(
                f"{py_file.as_posix()}: use get_global_manager().get(...) instead of relic.parameters.get alias"
            )

    assert not violations, (
        "Removed or non-canonical parameter accessors found in relic/scientific.\n"
        + "\n".join(violations)
    )
