from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = ROOT / "relic" / "scientific"
PARAM_ROOT = ROOT / "relic" / "parameters" / "database" / "relic"
ARCH_PATH = ROOT / "docs" / "source" / "architecture_overview.md"
PHYSICAL_CONSTANTS_PATH = PARAM_ROOT / "physical" / "constants.yaml"
OUTPUT_DIR = ROOT / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


BRIDGE_BASES = {
    "PhysicalFieldFunctor",
    "BiologicalProcessFunctor",
    "TransportMechanismFunctor",
    "ScaleCouplingFunctor",
}

ABSTRACT_BASES = {
    "PhysicalField",
    "BiologicalProcess",
    "TransportMechanism",
    "ScaleCoupling",
}

REQUIRED_METADATA_FIELDS = [
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
    "conservation_laws",
]

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


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("/", "\\")


def read_text_with_bom(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def get_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = get_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return get_name(node.value)
    if isinstance(node, ast.Call):
        return get_name(node.func)
    return ""


def unparse_safe(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def literal_value(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


@dataclass
class ClassAudit:
    file: str
    class_name: str
    line: int
    bases: list[str]
    bridge_base: str
    abstract_base: str
    has_super_init: bool
    super_init_args: dict[str, str]
    has_forward: bool
    forward_signature: list[str]
    forward_has_expected_signature: bool
    direct_state_assignment: list[int]
    has_get_dependencies: bool
    metadata_fields_found: list[str]
    metadata_missing_fields: list[str]
    metadata_references_empty: bool
    metadata_parameters_empty: bool
    metadata_inputs_empty: bool
    metadata_outputs_empty: bool
    parameter_calls: list[dict[str, Any]]
    uses_get_global_manager: list[int]
    uses_manager_get_parameter_value_removed: list[int]
    state_get_default_numeric: list[dict[str, Any]]
    torch_tensor_literal: list[dict[str, Any]]
    register_buffer_literal: list[dict[str, Any]]
    abs_on_flux_like: list[dict[str, Any]]


def scan_class(class_node: ast.ClassDef, path: Path, source: str) -> ClassAudit:
    bases = [get_name(b) for b in class_node.bases]
    bridge_base = next((b for b in bases if b.split(".")[-1] in BRIDGE_BASES), "")
    abstract_base = next((b for b in bases if b.split(".")[-1] in ABSTRACT_BASES), "")

    has_super_init = False
    super_init_args: dict[str, str] = {}
    has_forward = False
    forward_signature: list[str] = []
    forward_has_expected_signature = False
    direct_state_assignment: list[int] = []
    has_get_dependencies = False
    metadata_fields_found: list[str] = []
    metadata_missing_fields: list[str] = []
    metadata_references_empty = False
    metadata_parameters_empty = False
    metadata_inputs_empty = False
    metadata_outputs_empty = False
    parameter_calls: list[dict[str, Any]] = []
    uses_get_global_manager: list[int] = []
    uses_removed_manager_get_parameter_value: list[int] = []
    state_get_default_numeric: list[dict[str, Any]] = []
    torch_tensor_literal: list[dict[str, Any]] = []
    register_buffer_literal: list[dict[str, Any]] = []
    abs_on_flux_like: list[dict[str, Any]] = []

    for node in ast.walk(class_node):
        if isinstance(node, ast.FunctionDef):
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    func_name = get_name(n.func)

                    if node.name == "__init__" and func_name == "super().__init__":
                        has_super_init = True
                        for kw in n.keywords:
                            super_init_args[kw.arg or ""] = unparse_safe(kw.value)

                    if func_name.endswith("get_parameter_value"):
                        name_arg = None
                        default_arg = None
                        namespace_arg = None
                        if n.args:
                            name_arg = literal_value(n.args[0]) if len(n.args) >= 1 else None
                        for kw in n.keywords:
                            if kw.arg == "name":
                                name_arg = literal_value(kw.value)
                            elif kw.arg == "default":
                                default_arg = literal_value(kw.value)
                            elif kw.arg == "namespace":
                                namespace_arg = literal_value(kw.value)
                        parameter_calls.append(
                            {
                                "line": n.lineno,
                                "func": func_name,
                                "name": name_arg,
                                "default": default_arg,
                                "namespace": namespace_arg,
                                "raw": unparse_safe(n),
                            }
                        )

                    if func_name.endswith("get_global_manager"):
                        uses_get_global_manager.append(n.lineno)
                    if func_name.endswith("manager.get_parameter_value") or func_name == "manager.get_parameter_value":
                        uses_removed_manager_get_parameter_value.append(n.lineno)

                    if func_name.endswith("state.get") or func_name == "state.get":
                        default_node = None
                        if len(n.args) >= 2:
                            default_node = n.args[1]
                        for kw in n.keywords:
                            if kw.arg == "default":
                                default_node = kw.value
                        val = literal_value(default_node)
                        if isinstance(val, (int, float)):
                            state_get_default_numeric.append({"line": n.lineno, "default": val, "raw": unparse_safe(n)})

                    if func_name.endswith("torch.tensor") or func_name == "torch.tensor":
                        val = literal_value(n.args[0]) if n.args else None
                        if isinstance(val, (int, float)):
                            torch_tensor_literal.append({"line": n.lineno, "value": val, "raw": unparse_safe(n)})

                    if func_name.endswith("register_buffer") or func_name == "register_buffer":
                        if len(n.args) >= 2 and isinstance(n.args[1], ast.Call):
                            callee = get_name(n.args[1].func)
                            if callee.endswith("torch.tensor") or callee == "torch.tensor":
                                val = literal_value(n.args[1].args[0]) if n.args[1].args else None
                                if isinstance(val, (int, float)):
                                    register_buffer_literal.append({"line": n.lineno, "value": val, "raw": unparse_safe(n)})

                    if func_name == "abs":
                        expr = unparse_safe(n.args[0]) if n.args else ""
                        if any(k in expr.lower() for k in ["flux", "gradient", "grad", "diffusion", "fick", "transport"]):
                            abs_on_flux_like.append({"line": n.lineno, "expr": expr})

            if node.name == "forward":
                has_forward = True
                forward_signature = [arg.arg for arg in node.args.args]
                expected = ["self", "state", "environment", "time", "dt"]
                forward_has_expected_signature = forward_signature[:5] == expected
                for n in ast.walk(node):
                    if isinstance(n, ast.Assign):
                        for target in n.targets:
                            if isinstance(target, ast.Subscript):
                                value_name = get_name(target.value)
                                if value_name == "state":
                                    direct_state_assignment.append(n.lineno)
                    if isinstance(n, ast.AugAssign):
                        if isinstance(n.target, ast.Subscript):
                            value_name = get_name(n.target.value)
                            if value_name == "state":
                                direct_state_assignment.append(n.lineno)

            if node.name == "get_dependencies":
                has_get_dependencies = True

        if isinstance(node, ast.Call) and get_name(node.func).endswith("FunctorMetadata"):
            found = {}
            for kw in node.keywords:
                if kw.arg:
                    found[kw.arg] = kw.value
            metadata_fields_found = sorted(found.keys())
            metadata_missing_fields = [f for f in REQUIRED_METADATA_FIELDS if f not in found]
            refs = literal_value(found.get("references"))
            params = literal_value(found.get("parameters"))
            iv = literal_value(found.get("input_variables"))
            ov = literal_value(found.get("output_variables"))
            metadata_references_empty = isinstance(refs, list) and len(refs) == 0
            metadata_parameters_empty = isinstance(params, dict) and len(params) == 0
            metadata_inputs_empty = isinstance(iv, list) and len(iv) == 0
            metadata_outputs_empty = isinstance(ov, list) and len(ov) == 0

    return ClassAudit(
        file=rel(path),
        class_name=class_node.name,
        line=class_node.lineno,
        bases=bases,
        bridge_base=bridge_base,
        abstract_base=abstract_base,
        has_super_init=has_super_init,
        super_init_args=super_init_args,
        has_forward=has_forward,
        forward_signature=forward_signature,
        forward_has_expected_signature=forward_has_expected_signature,
        direct_state_assignment=sorted(set(direct_state_assignment)),
        has_get_dependencies=has_get_dependencies,
        metadata_fields_found=metadata_fields_found,
        metadata_missing_fields=metadata_missing_fields,
        metadata_references_empty=metadata_references_empty,
        metadata_parameters_empty=metadata_parameters_empty,
        metadata_inputs_empty=metadata_inputs_empty,
        metadata_outputs_empty=metadata_outputs_empty,
        parameter_calls=parameter_calls,
        uses_get_global_manager=sorted(set(uses_get_global_manager)),
        uses_manager_get_parameter_value_removed=sorted(set(uses_removed_manager_get_parameter_value)),
        state_get_default_numeric=state_get_default_numeric,
        torch_tensor_literal=torch_tensor_literal,
        register_buffer_literal=register_buffer_literal,
        abs_on_flux_like=abs_on_flux_like,
    )


def scan_scientific_python_files() -> dict[str, Any]:
    classes: list[ClassAudit] = []
    parse_errors: list[dict[str, Any]] = []
    all_files = [p for p in SCIENTIFIC_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    all_files = sorted(all_files)

    for path in all_files:
        text = read_text_with_bom(path)
        try:
            tree = ast.parse(text)
        except Exception as exc:
            parse_errors.append({"file": rel(path), "error": str(exc)})
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(scan_class(node, path, text))

    return {
        "scientific_python_file_count": len(all_files),
        "class_count": len(classes),
        "classes": [asdict(c) for c in classes],
        "parse_errors": parse_errors,
    }


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(read_text_with_bom(path))
    except Exception:
        return None


def scan_yaml_parameters(class_rows: list[dict[str, Any]]) -> dict[str, Any]:
    param_key_to_files: dict[str, list[str]] = defaultdict(list)
    missing_fields: list[dict[str, Any]] = []
    empty_references: list[dict[str, Any]] = []
    bad_dimension_value: list[dict[str, Any]] = []
    files = sorted([p for p in PARAM_ROOT.rglob("*.yaml") if "__pycache__" not in p.parts])

    declared_dimensions = {
        "length",
        "temperature",
        "time",
        "frequency",
        "energy",
        "energy_flux",
        "mass_flux",
        "molar_flux",
        "molar_mass",
        "volume_fraction",
        "area_ratio",
        "specific_heat_capacity",
        "mass_per_area",
        "emission_factor",
        "thermal_time",
        "dimensionless",
        "configuration",
        "unit_conversion",
        "none",
    }

    for yml in files:
        data = load_yaml(yml)
        if not isinstance(data, dict):
            continue
        params = data.get("parameters", {})
        if not isinstance(params, dict):
            continue
        for key, spec in params.items():
            param_key_to_files[key].append(rel(yml))
            if not isinstance(spec, dict):
                missing_fields.append({"file": rel(yml), "key": key, "missing": REQUIRED_PARAM_FIELDS})
                continue
            missing = [f for f in REQUIRED_PARAM_FIELDS if f not in spec]
            if missing:
                missing_fields.append({"file": rel(yml), "key": key, "missing": missing})
            refs = spec.get("references")
            if refs is None or (isinstance(refs, list) and len(refs) == 0):
                empty_references.append({"file": rel(yml), "key": key})
            dim = spec.get("dimension")
            if isinstance(dim, str) and dim not in declared_dimensions:
                bad_dimension_value.append({"file": rel(yml), "key": key, "dimension": dim})

    duplicates = [
        {"key": k, "files": v}
        for k, v in param_key_to_files.items()
        if len(v) > 1
    ]

    referenced_keys = set()
    for row in class_rows:
        for call in row.get("parameter_calls", []):
            name = call.get("name")
            if isinstance(name, str):
                referenced_keys.add(name)

    orphan_candidates = []
    for key in sorted(param_key_to_files):
        short = key.split(".")[-1]
        if key not in referenced_keys and short not in referenced_keys:
            orphan_candidates.append({"key": key, "files": param_key_to_files[key]})

    physical_constants = load_yaml(PHYSICAL_CONSTANTS_PATH)
    physical_constant_keys = set()
    if isinstance(physical_constants, dict):
        params = physical_constants.get("parameters", {})
        if isinstance(params, dict):
            physical_constant_keys = set(params.keys())

    duplicate_physical_constants = []
    for key, files_hit in param_key_to_files.items():
        if key in physical_constant_keys and any("physical\\constants.yaml" not in f for f in files_hit):
            duplicate_physical_constants.append({"key": key, "files": files_hit})

    return {
        "yaml_file_count": len(files),
        "parameter_key_count": len(param_key_to_files),
        "missing_fields": missing_fields,
        "empty_references": empty_references,
        "bad_dimension_value": bad_dimension_value,
        "duplicate_keys": duplicates,
        "orphan_candidates": orphan_candidates,
        "duplicate_physical_constants": duplicate_physical_constants,
    }


def scan_raw_patterns() -> dict[str, Any]:
    files = sorted([p for p in SCIENTIFIC_ROOT.rglob("*.py") if "__pycache__" not in p.parts])
    findings = []

    patterns = {
        "placeholder_text": re.compile(r"\b(placeholder|demo|simplified|todo|example|fallback)\b", re.IGNORECASE),
        "hardcoded_R_8_314": re.compile(r"\bR\s*=\s*8\.314\b"),
        "direct_manager_get": re.compile(r"get_global_manager\s*\("),
        "removed_manager_get_parameter_value": re.compile(r"\.get_parameter_value\s*\("),
    }

    for path in files:
        text = read_text_with_bom(path)
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            for name, rgx in patterns.items():
                if rgx.search(line):
                    findings.append({"file": rel(path), "line": i, "pattern": name, "text": line.strip()})

    return {"pattern_findings": findings}


def main() -> None:
    scientific = scan_scientific_python_files()
    yaml_report = scan_yaml_parameters(scientific["classes"])
    raw_report = scan_raw_patterns()
    report = {
        "scientific": scientific,
        "yaml": yaml_report,
        "raw": raw_report,
    }

    out_path = OUTPUT_DIR / "scientific_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] report written: {rel(out_path)}")
    print(f"[OK] scientific files: {scientific['scientific_python_file_count']}, classes: {scientific['class_count']}")
    print(f"[OK] yaml files: {yaml_report['yaml_file_count']}, params: {yaml_report['parameter_key_count']}")


if __name__ == "__main__":
    main()
