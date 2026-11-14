from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COUPLING_ROOT = ROOT / "relic" / "coupling"


FUNCTOR_BASES = {
    "FunctorBase",
    "ScaleCouplingFunctor",
    "BiologicalProcessFunctor",
    "TransportMechanismFunctor",
    "PhysicalFieldFunctor",
    "ExternalAdapter",
}

MANAGER_NAMES = {"pm", "manager", "param_manager", "parameter_manager"}


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_coupling_functor_like(class_node: ast.ClassDef) -> bool:
    bases = {_base_name(b) for b in class_node.bases}
    if bases & FUNCTOR_BASES:
        return True
    if class_node.name.endswith("Functor") or class_node.name.endswith("Adapter"):
        return True
    return False


def test_coupling_functor_classes_do_not_call_get_global_manager_or_manager_get():
    offenders: list[str] = []

    for path in sorted(COUPLING_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8-sig", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_coupling_functor_like(node):
                continue

            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue

                if isinstance(inner.func, ast.Name) and inner.func.id == "get_global_manager":
                    offenders.append(f"{path}:{inner.lineno} {node.name} uses get_global_manager()")
                    continue

                if isinstance(inner.func, ast.Attribute) and inner.func.attr == "get":
                    recv = inner.func.value
                    if isinstance(recv, ast.Name) and recv.id in MANAGER_NAMES:
                        offenders.append(f"{path}:{inner.lineno} {node.name} uses {recv.id}.get(...)")
                    elif (
                        isinstance(recv, ast.Attribute)
                        and isinstance(recv.value, ast.Name)
                        and recv.value.id == "self"
                        and recv.attr in {"parameter_manager", "_parameter_manager", "_param_manager"}
                    ):
                        offenders.append(
                            f"{path}:{inner.lineno} {node.name} uses self.{recv.attr}.get(...)"
                        )

    assert not offenders, (
        "Coupling Functor/Adapter classes must use self.get_parameter_value(...) as parameter entrypoint.\n"
        + "\n".join(offenders)
    )
