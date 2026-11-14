from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[2]
PEDOSPHERE_CODE_ROOT = ROOT / "relic" / "scientific" / "pedosphere"
PARAM_ROOT = ROOT / "relic" / "parameters" / "database" / "relic"
PEDOSPHERE_PARAM_ROOT = PARAM_ROOT / "pedosphere"

PARAM_CALL_NAMES: Set[str] = {
    "get_parameter_value",
    "require_param",
    "_require_parameter",
    "require_scalar",
    "_scalar_param",
    "_param",
}


def _iter_yaml_entries() -> Dict[str, Dict]:
    entries: Dict[str, Dict] = {}
    for yaml_path in PARAM_ROOT.rglob("*.yaml"):
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue

        namespace = str(payload.get("namespace", "")).strip()
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            continue

        for raw_key, record in parameters.items():
            key = str(raw_key)
            full_key = key if (not namespace or key.startswith(f"{namespace}.")) else f"{namespace}.{key}"
            entries[full_key] = {
                "file": yaml_path,
                "record": record if isinstance(record, dict) else {},
            }
    return entries


def _extract_call_name(call: ast.Call) -> Optional[str]:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _extract_literal_arg(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _iter_parameter_requests() -> Iterable[Tuple[str, int]]:
    for py_file in PEDOSPHERE_CODE_ROOT.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            call_name = _extract_call_name(node)
            if call_name not in PARAM_CALL_NAMES:
                continue

            key: Optional[str] = None
            namespace: Optional[str] = None

            if node.args:
                key = _extract_literal_arg(node.args[0])

            for kw in node.keywords:
                if kw.arg in {"name", "key"}:
                    key = _extract_literal_arg(kw.value) or key
                elif kw.arg == "namespace":
                    namespace = _extract_literal_arg(kw.value) or namespace

            if not key or not namespace:
                continue

            full_key = key if key.startswith(f"{namespace}.") else f"{namespace}.{key}"
            yield full_key, getattr(node, "lineno", -1)


def test_pedosphere_referenced_parameters_do_not_define_default_field():
    yaml_entries = _iter_yaml_entries()
    unexpected_default: List[str] = []

    for full_key, _line in _iter_parameter_requests():
        entry = yaml_entries.get(full_key)
        if entry is None:
            continue
        record = entry["record"]
        if "default" in record:
            relative_file = entry["file"].relative_to(ROOT)
            unexpected_default.append(f"{full_key} ({relative_file})")

    assert not unexpected_default, (
        "Referenced parameters must not define `default` field:\n"
        + "\n".join(sorted(set(unexpected_default)))
    )


def test_pedosphere_rate_units_no_year_minus_one_alias():
    violations: List[str] = []
    pattern = re.compile(r"^\s*units:\s*year-1\s*$")

    for yaml_path in PEDOSPHERE_PARAM_ROOT.rglob("*.yaml"):
        for idx, line in enumerate(yaml_path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.match(line):
                relative_file = yaml_path.relative_to(ROOT)
                violations.append(f"{relative_file}:{idx}")

    assert not violations, (
        "Found non-normalized rate units `year-1`; use `yr-1` instead:\n"
        + "\n".join(violations)
    )
