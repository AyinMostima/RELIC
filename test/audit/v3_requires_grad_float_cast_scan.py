import ast
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


def build_param_index() -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for y in Path("relic/parameters/database/relic").rglob("*.yaml"):
        data = yaml.safe_load(y.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        params = data.get("parameters", {})
        if not isinstance(params, dict):
            continue
        for k, v in params.items():
            if isinstance(v, dict):
                index[k] = v
    return index


def const_str(node) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def resolve_key(name: Optional[str], namespace: Optional[str]) -> Optional[str]:
    if not name:
        return None
    if "." in name:
        return name
    if namespace:
        return f"{namespace}.{name}"
    return name


def extract_get_parameter_value_call(node: ast.Call) -> Optional[Tuple[Optional[str], Optional[str]]]:
    if isinstance(node.func, ast.Attribute):
        if node.func.attr != "get_parameter_value":
            return None
    else:
        return None
    name = None
    namespace = None
    if node.args:
        name = const_str(node.args[0])
    for kw in node.keywords:
        if kw.arg == "namespace":
            namespace = const_str(kw.value)
    return name, namespace


def main():
    param_index = build_param_index()
    findings = []

    for py in Path("relic/scientific").rglob("*.py"):
        text = py.read_text(encoding="utf-8-sig", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Name) and node.func.id == "float"):
                continue
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Call):
                continue
            inner = node.args[0]
            parsed = extract_get_parameter_value_call(inner)
            if parsed is None:
                continue
            name, namespace = parsed
            full_key = resolve_key(name, namespace)
            if not full_key:
                continue
            spec = param_index.get(full_key)
            requires_grad = bool(spec.get("requires_grad")) if isinstance(spec, dict) else None
            findings.append(
                {
                    "file": str(py).replace("\\", "/"),
                    "line": node.lineno,
                    "parameter_key": full_key,
                    "requires_grad": requires_grad,
                    "units": spec.get("units") if isinstance(spec, dict) else None,
                }
            )

    out = Path("test/audit/v3_requires_grad_float_cast_scan.json")
    out.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK]", out.as_posix())
    print("count", len(findings))
    high = [f for f in findings if f.get("requires_grad") is True]
    print("requires_grad_true_count", len(high))
    for item in high[:40]:
        print(item)


if __name__ == "__main__":
    main()
