from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis_results"
OUT_PATH = OUT_DIR / "deep_scientific_audit_v3_report.json"

TARGET_PY_DIRS = [
    ROOT / "relic" / "scientific",
    ROOT / "relic" / "coupling",
]
TARGET_YAML_DIR = ROOT / "relic" / "parameters" / "database" / "relic"
THEORY_DIR = ROOT / "docs" / "source" / "theory"

BRIDGE_MAP = {
    "PhysicalFieldFunctor": "PhysicalField",
    "BiologicalProcessFunctor": "BiologicalProcess",
    "TransportMechanismFunctor": "TransportMechanism",
    "ScaleCouplingFunctor": "ScaleCoupling",
}

REQUIRED_PARAM_FIELDS = [
    "value",
    "units",
    "dtype",
    "category",
    "description",
    "references",
    "tags",
    "dimension",
    "requires_grad",
    "provenance",
]

DIMENSION_EXPECTED_BY_UNIT = {
    "s-1": "frequency",
    "day-1": "frequency",
    "d-1": "frequency",
    "yr-1": "frequency",
    "J mol-1": "energy",
    "W m-2": "energy_flux",
    "kg m-2 s-1": "mass_flux",
    "m3 m-3": "volume_fraction",
    "m2 m-2": "area_ratio",
    "dimensionless": "dimensionless",
}

KEY_RE = re.compile(
    r"""(?:self\.get_parameter_value|get_global_manager\(\)\.get|manager\.get)\(\s*["']([^"']+)["']""",
    re.MULTILINE,
)

GET_PARAM_DEFAULT_RE = re.compile(
    r"""self\.get_parameter_value\([^)]*default\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)""",
    re.MULTILINE,
)

STATE_GET_DEFAULT_CONST_RE = re.compile(
    r"""state\.get\([^)]*,\s*(?:torch\.tensor\()?[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\)?""",
    re.MULTILINE,
)

TORCH_TENSOR_CONST_RE = re.compile(
    r"""torch\.tensor\(\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*(?:,|\))""",
    re.MULTILINE,
)

REGISTER_BUFFER_CONST_RE = re.compile(
    r"""register_buffer\([^)]*torch\.tensor\(\s*[-+]?\d+(?:\.\d+)?""",
    re.MULTILINE,
)

STATE_DIRECT_WRITE_RE = re.compile(r"""state\[[^\]]+\]\s*=""")
CATEGORY_THEORY_IMPORT_RE = re.compile(r"""category_theory""")
MANAGER_GET_PARAMETER_VALUE_RE = re.compile(r"""manager\.get_parameter_value\(""")


def list_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob(f"*{suffix}") if p.is_file())


def _node_to_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _extract_constant_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_list_of_strings(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple)):
        values: list[str] = []
        for elt in node.elts:
            sval = _extract_constant_str(elt)
            if sval is not None:
                values.append(sval)
        return values
    return []


def _extract_metadata_calls(class_node: ast.ClassDef) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for node in ast.walk(class_node):
        if isinstance(node, ast.Call):
            fn = _node_to_text(node.func)
            if fn.endswith("FunctorMetadata"):
                kw: dict[str, Any] = {}
                for item in node.keywords:
                    if item.arg is None:
                        continue
                    if item.arg in ("input_variables", "output_variables", "conservation_laws", "references"):
                        kw[item.arg] = _extract_list_of_strings(item.value)
                    else:
                        kw[item.arg] = _node_to_text(item.value)
                calls.append(
                    {
                        "lineno": node.lineno,
                        "keywords": kw,
                    }
                )
    return calls


def _extract_default_namespace(class_node: ast.ClassDef) -> str | None:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_default_parameter_namespace":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return):
                    sval = _extract_constant_str(sub.value)
                    if sval:
                        return sval
    return None


def _has_super_init_call(class_node: ast.ClassDef) -> bool:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = _node_to_text(sub.func).replace(" ", "")
                    if fn in ("super().__init__", "FunctorBase.__init__"):
                        return True
    return False


def _extract_forward_signature(class_node: ast.ClassDef) -> dict[str, Any] | None:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            args = [a.arg for a in node.args.args]
            return {
                "lineno": node.lineno,
                "args": args,
            }
    return None


def _has_method(class_node: ast.ClassDef, method_name: str) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == method_name for node in class_node.body)


def _read_text_robust(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="ignore")


def scan_python_file(path: Path) -> dict[str, Any]:
    text = _read_text_robust(path)
    data: dict[str, Any] = {
        "file": str(path.relative_to(ROOT)).replace("/", "\\"),
        "line_count": text.count("\n") + 1,
        "category_theory_import": bool(CATEGORY_THEORY_IMPORT_RE.search(text)),
        "manager_get_parameter_value_call": bool(MANAGER_GET_PARAMETER_VALUE_RE.search(text)),
        "get_global_manager_call": "get_global_manager(" in text,
        "parameter_default_numeric_hits": [m.group(0) for m in GET_PARAM_DEFAULT_RE.finditer(text)],
        "state_get_default_constant_hits": [m.group(0) for m in STATE_GET_DEFAULT_CONST_RE.finditer(text)],
        "torch_tensor_constant_hits": [m.group(0) for m in TORCH_TENSOR_CONST_RE.finditer(text)],
        "register_buffer_constant_hits": [m.group(0) for m in REGISTER_BUFFER_CONST_RE.finditer(text)],
        "state_direct_write_hits": [m.group(0) for m in STATE_DIRECT_WRITE_RE.finditer(text)],
        "references_comment_present": "# References:" in text,
        "parameter_keys_used": sorted(set(KEY_RE.findall(text))),
        "classes": [],
    }
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        data["syntax_error"] = f"{exc.lineno}:{exc.offset} {exc.msg}"
        return data

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [_node_to_text(b) for b in node.bases]
        bridge_base = next((b for b in bases if b.split(".")[-1] in BRIDGE_MAP), None)
        abstraction = BRIDGE_MAP.get(bridge_base.split(".")[-1], None) if bridge_base else None
        metadata_calls = _extract_metadata_calls(node)
        # Dependency declarations are not parsed from source literals; the
        # runtime declaration is reported by test.audit.collect_functor_inventory.
        forward_signature = _extract_forward_signature(node)
        cls = {
            "name": node.name,
            "lineno": node.lineno,
            "bases": bases,
            "bridge_base": bridge_base,
            "abstraction": abstraction,
            "has_super_init": _has_super_init_call(node),
            "has_forward": forward_signature is not None,
            "forward_signature": forward_signature,
            "has_compute_rate": _has_method(node, "compute_rate"),
            "has_compute_flux": _has_method(node, "compute_flux"),
            "default_parameter_namespace": _extract_default_namespace(node),
            "metadata_calls": metadata_calls,
        }
        data["classes"].append(cls)
    return data


def scan_yaml_file(path: Path) -> dict[str, Any]:
    rel = str(path.relative_to(ROOT)).replace("/", "\\")
    text = _read_text_robust(path)
    result: dict[str, Any] = {
        "file": rel,
        "has_fraction_units": "/" in text,
        "uses_units_one": bool(re.search(r"units:\s*['\"]?1['\"]?\s*$", text, flags=re.MULTILINE)),
        "parameter_count": 0,
        "parameters": [],
        "functor_metadata": {},
        "parse_error": None,
    }
    try:
        obj = yaml.safe_load(text) or {}
    except Exception as exc:
        result["parse_error"] = str(exc)
        return result

    params = obj.get("parameters", {})
    if not isinstance(params, dict):
        return result

    result["parameter_count"] = len(params)
    fm = obj.get("functor_metadata")
    if isinstance(fm, dict):
        result["functor_metadata"] = fm

    for key, spec in params.items():
        row: dict[str, Any] = {"key": key, "missing_fields": [], "dimension_mismatch": None}
        if not isinstance(spec, dict):
            row["missing_fields"] = REQUIRED_PARAM_FIELDS[:]
            result["parameters"].append(row)
            continue
        missing = [f for f in REQUIRED_PARAM_FIELDS if f not in spec]
        row["missing_fields"] = missing
        references = spec.get("references")
        row["empty_references"] = isinstance(references, list) and len(references) == 0
        unit = spec.get("units")
        dim = spec.get("dimension")
        if isinstance(unit, str) and unit in DIMENSION_EXPECTED_BY_UNIT and isinstance(dim, str):
            expected = DIMENSION_EXPECTED_BY_UNIT[unit]
            if dim != expected:
                row["dimension_mismatch"] = {"unit": unit, "dimension": dim, "expected": expected}
        row["contains_bounds"] = "bounds" in spec
        result["parameters"].append(row)
    return result


def guess_theory_coverage(scientific_py_files: list[Path], theory_files: list[Path]) -> list[dict[str, Any]]:
    theory_names = {p.stem.lower() for p in theory_files}
    missing: list[dict[str, Any]] = []
    for path in scientific_py_files:
        if path.name.startswith("__init__"):
            continue
        if path.stem in {"base", "common", "utils", "registry", "models", "parameters", "io", "writer"}:
            continue
        stem = path.stem.lower()
        parts = [p.lower() for p in path.parts]
        folder = parts[-2] if len(parts) >= 2 else ""
        likely = any(
            key in theory_names
            for key in (
                stem,
                folder,
                f"{folder}_{stem}" if folder else stem,
            )
        )
        if not likely:
            missing.append(
                {
                    "file": str(path.relative_to(ROOT)).replace("/", "\\"),
                    "stem": stem,
                    "folder": folder,
                }
            )
    return missing


def main() -> None:
    py_files: list[Path] = []
    for base in TARGET_PY_DIRS:
        py_files.extend(list_files(base, ".py"))

    yaml_files = list_files(TARGET_YAML_DIR, ".yaml")
    theory_files = list_files(THEORY_DIR, ".md")

    py_scans = [scan_python_file(p) for p in py_files]
    yaml_scans = [scan_yaml_file(p) for p in yaml_files]

    key_to_files: dict[str, list[str]] = defaultdict(list)
    for row in yaml_scans:
        for p in row.get("parameters", []):
            key = p.get("key")
            if isinstance(key, str):
                key_to_files[key].append(row["file"])

    used_keys = set()
    for row in py_scans:
        used_keys.update(row.get("parameter_keys_used", []))

    exact_orphans = sorted(k for k in key_to_files if k not in used_keys)

    output_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in py_scans:
        for cls in row.get("classes", []):
            for meta in cls.get("metadata_calls", []):
                out_vars = meta.get("keywords", {}).get("output_variables")
                if not isinstance(out_vars, list):
                    continue
                for ov in out_vars:
                    output_map[ov].append(
                        {
                            "file": row["file"],
                            "class": cls["name"],
                            "line": meta.get("lineno"),
                        }
                    )

    output_overlaps = {
        k: v
        for k, v in output_map.items()
        if len(v) > 1 and not str(k).startswith("diagnostics.")
    }

    theory_missing = guess_theory_coverage(
        [p for p in py_files if "relic/scientific" in str(p).replace("\\", "/")],
        theory_files,
    )

    summary = {
        "python_file_count": len(py_files),
        "yaml_file_count": len(yaml_files),
        "theory_file_count": len(theory_files),
        "files_read_confirmed": len(py_scans) == len(py_files) and len(yaml_scans) == len(yaml_files),
    }

    report = {
        "summary": summary,
        "python_scans": py_scans,
        "yaml_scans": yaml_scans,
        "output_overlaps": output_overlaps,
        "exact_orphan_parameter_keys": exact_orphans,
        "theory_missing_candidates": theory_missing,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] report: {OUT_PATH}")
    print(
        "[OK] counts:",
        json.dumps(summary, ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
