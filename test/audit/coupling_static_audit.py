import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path("relic") / "coupling"
OUT_DIR = Path("analysis_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def _forward_signature(class_node: ast.ClassDef) -> Optional[List[str]]:
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            args = [a.arg for a in item.args.args]
            if item.args.vararg:
                args.append("*" + item.args.vararg.arg)
            if item.args.kwarg:
                args.append("**" + item.args.kwarg.arg)
            return args
    return None


def _extract_functor_metadata_fields(class_node: ast.ClassDef) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for sub in ast.walk(class_node):
        if not isinstance(sub, ast.Call):
            continue
        fn = _node_name(sub.func)
        if not (fn.endswith("FunctorMetadata") or fn == "FunctorMetadata"):
            continue
        for kw in sub.keywords:
            if kw.arg is None:
                continue
            if isinstance(kw.value, ast.Constant):
                fields[kw.arg] = kw.value.value
            elif isinstance(kw.value, (ast.List, ast.Tuple)):
                literal_values: List[Any] = []
                all_literal = True
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        literal_values.append(elt.value)
                    else:
                        all_literal = False
                        break
                fields[kw.arg] = literal_values if all_literal else "__sequence__"
            elif isinstance(kw.value, ast.Dict):
                fields[kw.arg] = "__dict__"
            else:
                fields[kw.arg] = _node_name(kw.value)
    return fields


def _is_functor_like(bases: List[str]) -> bool:
    joined = " ".join(bases)
    if "FunctorBase" in joined:
        return True
    if any(k in joined for k in BRIDGES):
        return True
    if "Functor" in joined:
        return True
    return False


def _infer_abstract(bases: List[str]) -> str:
    joined = " ".join(bases)
    for bridge, abstract in BRIDGES.items():
        if bridge in joined:
            return abstract
    return "Unknown"


def _class_line_map(class_node: ast.ClassDef) -> Dict[str, int]:
    return {
        "line": getattr(class_node, "lineno", -1),
        "end_line": getattr(class_node, "end_lineno", -1),
    }


def scan_file(path: Path) -> Dict[str, Any]:
    src = path.read_text(encoding="utf-8-sig", errors="ignore")
    tree = ast.parse(src)

    functors: List[Dict[str, Any]] = []
    file_issues: Dict[str, List[Dict[str, Any]]] = {
        "category_theory_imports": [],
        "removed_manager_api_usage": [],
        "functor_get_global_manager_calls": [],
        "functor_manager_get_calls": [],
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "category_theory" in node.module:
                file_issues["category_theory_imports"].append({"line": node.lineno, "module": node.module})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "category_theory" in alias.name:
                    file_issues["category_theory_imports"].append({"line": node.lineno, "module": alias.name})

        if isinstance(node, ast.Call):
            fn = _node_name(node.func)
            if fn.endswith("get_parameter_value") and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "manager":
                    file_issues["removed_manager_api_usage"].append({"line": node.lineno, "call": fn})

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [_node_name(b) for b in node.bases]
        if not _is_functor_like(bases):
            continue

        inferred = _infer_abstract(bases)
        metadata_fields = _extract_functor_metadata_fields(node)
        missing_meta = [k for k in META_REQUIRED if k not in metadata_fields]
        forward_sig = _forward_signature(node)

        # Forward contract in 2026-02 SSOT: forward(state, time, dt, **kwargs)
        forward_ok = False
        if forward_sig:
            forward_ok = forward_sig[:4] == ["self", "state", "time", "dt"]

        # Functor-only parameter access lock: forbid get_global_manager()/manager.get() usage inside class methods.
        functor_get_global_manager_calls: List[int] = []
        functor_manager_get_calls: List[int] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = _node_name(sub.func)
                if fn.endswith("get_global_manager") or fn == "get_global_manager":
                    functor_get_global_manager_calls.append(getattr(sub, "lineno", -1))

                # Heuristic: manager-like variables calling .get() inside functor code
                if isinstance(sub.func, ast.Attribute) and sub.func.attr == "get":
                    recv = _node_name(sub.func.value)
                    if recv in {"pm", "manager", "param_manager", "parameter_manager"}:
                        functor_manager_get_calls.append(getattr(sub, "lineno", -1))

        if functor_get_global_manager_calls:
            file_issues["functor_get_global_manager_calls"].append(
                {"class": node.name, "lines": sorted(set(functor_get_global_manager_calls))}
            )
        if functor_manager_get_calls:
            file_issues["functor_manager_get_calls"].append(
                {"class": node.name, "lines": sorted(set(functor_manager_get_calls))}
            )

        functors.append(
            {
                "file": str(path).replace("\\\\", "/"),
                "class_name": node.name,
                "bases": bases,
                "inferred_abstract": inferred,
                "forward_signature": forward_sig,
                "forward_signature_ok": forward_ok,
                "metadata_fields": metadata_fields,
                "missing_metadata_fields": missing_meta,
                **_class_line_map(node),
            }
        )

    return {"functors": functors, "issues": file_issues}


def main() -> None:
    results: Dict[str, Any] = {
        "files_scanned": 0,
        "functor_count": 0,
        "functors": [],
        "issues": {
            "category_theory_imports": [],
            "removed_manager_api_usage": [],
            "functor_get_global_manager_calls": [],
            "functor_manager_get_calls": [],
        },
    }

    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            file_result = scan_file(path)
        except SyntaxError as exc:
            results["issues"]["removed_manager_api_usage"].append(
                {
                    "file": str(path).replace("\\\\", "/"),
                    "line": getattr(exc, "lineno", -1),
                    "call": "syntax_error",
                }
            )
            continue
        results["files_scanned"] += 1
        results["functors"].extend(file_result["functors"])
        for key, arr in file_result["issues"].items():
            if not arr:
                continue
            for item in arr:
                item["file"] = str(path).replace("\\\\", "/")
            results["issues"][key].extend(arr)

    results["functor_count"] = len(results["functors"])

    out_json = OUT_DIR / "coupling_static_audit.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")

    summary_lines = [
        f"files_scanned={results['files_scanned']}",
        f"functor_count={results['functor_count']}",
    ]
    for key, arr in results["issues"].items():
        summary_lines.append(f"{key}={len(arr)}")

    out_txt = OUT_DIR / "coupling_static_audit_summary.txt"
    out_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()

