import ast
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT = REPO_ROOT / "relic" / "scientific"
OUT_DIR = REPO_ROOT / "analysis_results"
forward_template_present = importlib.import_module(
    "test.audit.list_forward_template_missing"
).forward_template_present

BRIDGES = {
    "BiologicalProcessFunctor": "BiologicalProcess",
    "PhysicalFieldFunctor": "PhysicalField",
    "TransportMechanismFunctor": "TransportMechanism",
    "ScaleCouplingFunctor": "ScaleCoupling",
}

META_REQUIRED = [
    "name",
    "version",
    "description",
    "category",
    "execution_phase",
    "mathematical_expression",
    "references",
    "parameters",
    "input_variables",
    "output_variables",
    "spatial_scale",
    "variable_kind",
]

# Restrict placeholder scan to explicit backlog markers and runtime stubs.
# Terms like "fallback"/"simplified" are often legitimate model descriptions
# and produced excessive false positives in previous reports.
PLACEHOLDER_RE = re.compile(
    r"\b(placeholder|todo|fixme)\b",
    re.IGNORECASE,
)
NOT_IMPLEMENTED_RE = re.compile(r"\bNotImplementedError\b")


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _node_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception:
            return ""
    return ""


def _is_numeric_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_numeric_constant(node.operand)
    return False


def _extract_functor_metadata_fields(class_node: ast.ClassDef) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for sub in ast.walk(class_node):
        if isinstance(sub, ast.Call):
            fn = _node_name(sub.func)
            if fn.endswith("FunctorMetadata") or fn == "FunctorMetadata":
                for kw in sub.keywords:
                    if kw.arg is None:
                        continue
                    if isinstance(kw.value, ast.Constant):
                        fields[kw.arg] = kw.value.value
                    elif isinstance(kw.value, (ast.List, ast.Tuple)):
                        values: list[Any] = []
                        all_literal = True
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Constant):
                                values.append(elt.value)
                            else:
                                all_literal = False
                                break
                        fields[kw.arg] = values if all_literal else "__sequence__"
                    elif isinstance(kw.value, ast.Dict):
                        fields[kw.arg] = "__dict__"
                    else:
                        fields[kw.arg] = _node_name(kw.value)
    return fields


def _forward_signature(class_node: ast.ClassDef) -> list[str] | None:
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            args = [a.arg for a in item.args.args]
            if item.args.vararg:
                args.append("*" + item.args.vararg.arg)
            if item.args.kwarg:
                args.append("**" + item.args.kwarg.arg)
            return args
    return None


def _class_line_map(class_node: ast.ClassDef) -> dict[str, int]:
    return {
        "line": getattr(class_node, "lineno", -1),
        "end_line": getattr(class_node, "end_lineno", -1),
    }


def scan_file(path: Path) -> dict[str, Any]:
    src = path.read_text(encoding="utf-8-sig", errors="ignore")
    tree = ast.parse(src)

    functors: list[dict[str, Any]] = []
    file_issues: dict[str, list[dict[str, Any]]] = {
        "global_manager_calls": [],
        "functor_global_manager_calls": [],
        "removed_api_calls": [],
        "default_numeric_in_get_parameter_value": [],
        "state_get_default_numeric": [],
        "direct_state_assignment": [],
        "category_theory_imports": [],
        "placeholder_markers": [],
        "notimplemented_markers": [],
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "category_theory" in node.module:
                file_issues["category_theory_imports"].append(
                    {"line": node.lineno, "module": node.module}
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "category_theory" in alias.name:
                    file_issues["category_theory_imports"].append(
                        {"line": node.lineno, "module": alias.name}
                    )

        if isinstance(node, ast.Call):
            fn = _node_name(node.func)
            if fn.endswith("get_global_manager") or fn == "get_global_manager":
                file_issues["global_manager_calls"].append({"line": node.lineno, "call": fn})
            if (
                fn.endswith("get_parameter_value")
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "manager"
            ):
                file_issues["removed_api_calls"].append({"line": node.lineno, "call": fn})
            if fn.endswith("get_parameter_value"):
                for kw in node.keywords:
                    if kw.arg in {"default", "default_value"} and _is_numeric_constant(kw.value):
                        file_issues["default_numeric_in_get_parameter_value"].append(
                            {
                                "line": node.lineno,
                                "arg": kw.arg,
                            }
                        )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "state"
            ):
                for kw in node.keywords:
                    if kw.arg == "default" and _is_numeric_constant(kw.value):
                        file_issues["state_get_default_numeric"].append({"line": node.lineno})

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "state"
                ):
                    file_issues["direct_state_assignment"].append({"line": node.lineno})

        if isinstance(node, ast.ClassDef):
            bases = [_node_name(b) for b in node.bases]
            base_join = " ".join(bases)
            if "Functor" in base_join or any(b in base_join for b in BRIDGES):
                for inner in ast.walk(node):
                    if not isinstance(inner, ast.Call):
                        continue
                    fn = _node_name(inner.func)
                    if fn.endswith("get_global_manager") or fn == "get_global_manager":
                        file_issues["functor_global_manager_calls"].append(
                            {
                                "line": inner.lineno,
                                "call": fn,
                                "class_name": node.name,
                            }
                        )
                inferred = "Unknown"
                for bridge, abstract in BRIDGES.items():
                    if bridge in bases or bridge in base_join:
                        inferred = abstract
                        break
                metadata_fields = _extract_functor_metadata_fields(node)
                missing_meta = [k for k in META_REQUIRED if k not in metadata_fields]
                functors.append(
                    {
                        "file": _display_path(path),
                        "class_name": node.name,
                        "bases": bases,
                        "inferred_abstract": inferred,
                        "forward_signature": _forward_signature(node),
                        "metadata_fields": metadata_fields,
                        "missing_metadata_fields": missing_meta,
                        **_class_line_map(node),
                    }
                )

    for m in PLACEHOLDER_RE.finditer(src):
        line = src.count("\n", 0, m.start()) + 1
        file_issues["placeholder_markers"].append({"line": line, "token": m.group(1).lower()})
    for m in NOT_IMPLEMENTED_RE.finditer(src):
        line = src.count("\n", 0, m.start()) + 1
        file_issues["notimplemented_markers"].append({"line": line, "token": "notimplementederror"})

    lines = src.splitlines()
    forward_template_missing_lines = [
        index + 1
        for index, line in enumerate(lines)
        if line.lstrip().startswith("def forward(") and not forward_template_present(lines, index)
    ]
    return {
        "functors": functors,
        "issues": file_issues,
        "contains_non_ascii": any(
            ord(character) > 127 for character in src if character != "\ufeff"
        ),
        "forward_template_missing_lines": forward_template_missing_lines,
    }


def _display_path(path: Path) -> str:
    try:
        display_path = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        display_path = path.resolve()
    return display_path.as_posix().replace("/", "\\")


def build_report(root: Path = ROOT) -> dict[str, Any]:
    """Build the canonical static-audit report without writing artifacts."""
    results: dict[str, Any] = {
        "files_scanned": 0,
        "functor_count": 0,
        "functors": [],
        "non_ascii_files": [],
        "forward_template_missing": [],
        "issues": {
            "global_manager_calls": [],
            "functor_global_manager_calls": [],
            "removed_api_calls": [],
            "default_numeric_in_get_parameter_value": [],
            "state_get_default_numeric": [],
            "direct_state_assignment": [],
            "category_theory_imports": [],
            "placeholder_markers": [],
            "notimplemented_markers": [],
        },
    }

    for path in sorted(root.rglob("*.py")):
        try:
            file_result = scan_file(path)
        except SyntaxError as exc:
            results["issues"]["placeholder_markers"].append(
                {
                    "file": _display_path(path),
                    "line": getattr(exc, "lineno", -1),
                    "token": "syntax_error",
                }
            )
            continue
        results["files_scanned"] += 1
        results["functors"].extend(file_result["functors"])
        for key, arr in file_result["issues"].items():
            if not arr:
                continue
            for item in arr:
                item["file"] = _display_path(path)
            results["issues"][key].extend(arr)
        if file_result["contains_non_ascii"]:
            results["non_ascii_files"].append(_display_path(path))
        results["forward_template_missing"].extend(
            {
                "file": _display_path(path),
                "line": line_number,
            }
            for line_number in file_result["forward_template_missing_lines"]
        )

    results["functor_count"] = len(results["functors"])
    total_global_manager_calls = len(results["issues"]["global_manager_calls"])
    functor_global_manager_calls = len(results["issues"]["functor_global_manager_calls"])
    functor_global_manager_files = len(
        {item["file"] for item in results["issues"]["functor_global_manager_calls"]}
    )
    results["metrics"] = {
        "global_manager_calls_total": total_global_manager_calls,
        "global_manager_calls_functor": functor_global_manager_calls,
        "global_manager_files_functor": functor_global_manager_files,
        "global_manager_calls_non_functor": max(
            total_global_manager_calls - functor_global_manager_calls,
            0,
        ),
        "non_ascii_files": len(results["non_ascii_files"]),
        "forward_missing_template": len(results["forward_template_missing"]),
    }
    return results


def main() -> int:
    results = build_report(ROOT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "scientific_static_audit.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")

    summary_lines = [
        f"files_scanned={results['files_scanned']}",
        f"functor_count={results['functor_count']}",
        f"global_manager_calls_total={results['metrics']['global_manager_calls_total']}",
        f"global_manager_calls_functor={results['metrics']['global_manager_calls_functor']}",
        f"global_manager_files_functor={results['metrics']['global_manager_files_functor']}",
        f"global_manager_calls_non_functor={results['metrics']['global_manager_calls_non_functor']}",
        f"non_ascii_files={results['metrics']['non_ascii_files']}",
        f"forward_missing_template={results['metrics']['forward_missing_template']}",
    ]
    for key, arr in results["issues"].items():
        summary_lines.append(f"{key}={len(arr)}")

    out_txt = OUT_DIR / "scientific_static_audit_summary.txt"
    out_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
