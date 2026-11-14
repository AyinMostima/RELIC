from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "relic" / "scientific" / "anthroposphere" / "land_use" / "transitions.py"
TILLAGE_SOURCE = (
    ROOT
    / "relic"
    / "scientific"
    / "anthroposphere"
    / "management"
    / "tillage.py"
)
COORDINATOR_SOURCE = (
    ROOT
    / "relic"
    / "scientific"
    / "anthroposphere"
    / "management"
    / "coordinator.py"
)
OVERLAP_POLICY = ROOT / "relic" / "core" / "simulator" / "output_overlap_whitelist.json"
LAND_USE_PARAMETERS = (
    ROOT
    / "relic"
    / "parameters"
    / "database"
    / "relic"
    / "anthroposphere"
    / "land_use.yaml"
)
CERTIFICATES = {
    SOURCE: ROOT
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "anthroposphere_land_use_transitions_land_use_transitions.yaml",
    TILLAGE_SOURCE: ROOT
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "anthroposphere_management_tillage_tillage_management.yaml",
    COORDINATOR_SOURCE: ROOT
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "anthroposphere_management_coordinator_management_coordinator.yaml",
}
REMOVED_SCM_CONTRACTS = (
    ROOT
    / "relic"
    / "contracts"
    / "variables"
    / "anthroposphere"
    / "decomposition_modifier.yaml",
    ROOT
    / "relic"
    / "contracts"
    / "variables"
    / "anthroposphere"
    / "infiltration_modifier.yaml",
    ROOT
    / "relic"
    / "contracts"
    / "variables"
    / "anthroposphere"
    / "total_tillage_modifier.yaml",
)


def _extract_modifier_keys(tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "LandUseTransitions":
            continue
        for member in node.body:
            if not isinstance(member, ast.FunctionDef) or member.name != "_calculate_parameter_modifiers":
                continue
            for inner in ast.walk(member):
                if not isinstance(inner, ast.Assign):
                    continue
                for target in inner.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "modifiers"
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                    ):
                        keys.add(target.slice.value)
    return keys


def _extract_declared_output_variables(tree: ast.AST) -> set[str]:
    outputs: set[str] = set()
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        class_dict_keys: dict[str, set[str]] = {}
        for member in class_node.body:
            target = None
            value = None
            if isinstance(member, ast.Assign) and len(member.targets) == 1:
                target = member.targets[0]
                value = member.value
            elif isinstance(member, ast.AnnAssign):
                target = member.target
                value = member.value
            if not isinstance(target, ast.Name) or not isinstance(value, ast.Dict):
                continue
            keys = {
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            class_dict_keys[target.id] = keys

        for node in ast.walk(class_node):
            if not isinstance(node, ast.Call):
                continue
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name != "FunctorMetadata":
                continue
            for kw in node.keywords:
                if kw.arg != "output_variables":
                    continue
                if isinstance(kw.value, ast.List):
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            outputs.add(elt.value)
                    continue
                if (
                    isinstance(kw.value, ast.Call)
                    and isinstance(kw.value.func, ast.Name)
                    and kw.value.func.id == "list"
                    and len(kw.value.args) == 1
                    and isinstance(kw.value.args[0], ast.Attribute)
                ):
                    outputs.update(class_dict_keys.get(kw.value.args[0].attr, set()))
    return outputs


def test_land_use_has_no_modifier_secondary_runtime():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    modifier_keys = _extract_modifier_keys(tree)
    declared_outputs = _extract_declared_output_variables(tree)

    assert not modifier_keys
    class_methods = {
        member.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "LandUseTransitions"
        for member in node.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_calculate_parameter_modifiers" not in class_methods
    assert declared_outputs == {
        "land_use_fraction",
        "diagnostics.land_use_transitions.delta_fraction",
        "diagnostics.land_use_transitions.dominant_type_index",
        "diagnostics.land_use_transitions.maximum_fraction",
        "diagnostics.land_use_transitions.soil_carbon_input_rate_change",
        "diagnostics.land_use_transitions.canonical_iphs_runtime",
    }


def test_unused_management_modifiers_are_absent_from_runtime_interfaces():
    forbidden = {
        "decomposition_modifier",
        "infiltration_modifier",
        "total_tillage_modifier",
    }

    land_use_tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    assert forbidden.isdisjoint(_extract_modifier_keys(land_use_tree))
    assert forbidden.isdisjoint(_extract_declared_output_variables(land_use_tree))

    for source_path in (TILLAGE_SOURCE, COORDINATOR_SOURCE):
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert forbidden.isdisjoint(_extract_declared_output_variables(tree))

    coordinator_source = COORDINATOR_SOURCE.read_text(encoding="utf-8")
    assert '"decomposition_modifier"' not in coordinator_source
    assert '"total_tillage_modifier"' not in coordinator_source
    assert "tillage.decomposition_modifier" not in coordinator_source


def test_removed_modifier_ports_have_no_overlap_or_parameter_ssot_entries():
    policy = json.loads(OVERLAP_POLICY.read_text(encoding="utf-8"))
    assert "decomposition_modifier" not in policy
    assert "infiltration_modifier" not in policy

    land_use_parameters = LAND_USE_PARAMETERS.read_text(encoding="utf-8")
    assert "decomposition_modifier" not in land_use_parameters
    assert "infiltration_modifier" not in land_use_parameters


def test_management_certificates_match_current_functor_outputs():
    forbidden = {
        "decomposition_modifier",
        "infiltration_modifier",
        "total_tillage_modifier",
    }

    for source_path, certificate_path in CERTIFICATES.items():
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        expected = _extract_declared_output_variables(tree)
        certificate = yaml.safe_load(certificate_path.read_text(encoding="utf-8"))
        declared = set(certificate["output_variables"])
        iphs_outputs = set(certificate["interface_map"]["iphs_outputs"])

        assert declared == expected
        assert iphs_outputs == expected
        assert forbidden.isdisjoint(declared)


def test_removed_modifier_ports_are_absent_from_scm_registry_inputs():
    stale = [str(path.relative_to(ROOT)) for path in REMOVED_SCM_CONTRACTS if path.exists()]
    assert not stale, "Removed modifier contracts still autoload into SCM: " + ", ".join(stale)
