from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
import re
import runpy
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relic.computation.operators.iphs_certificate import (
    IPHS_LEVEL_DEFINITIONS,
    load_iphs_equivalence_certificate,
)
from relic.core.functor.iphs_metadata import ensure_iphs_equivalent_expression

SCIENTIFIC_ROOT = PROJECT_ROOT / "relic" / "scientific"
THEORY_ROOT = PROJECT_ROOT / "docs" / "source" / "theory"
CERTIFICATE_ROOT = THEORY_ROOT / "iphs_certificates"
OUTPUT_DIR = PROJECT_ROOT / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENERIC_RUNTIME_USAGE = "Production forward path is the single canonical runtime entry point for this process."
GENERIC_STATE_ADVANCER = "J_h e_E + M_h e_S + G_h u_h + N_h r_h"
IMPLEMENTATION_STATUS_IMPLEMENTED = "implemented"
IMPLEMENTATION_STATUS_NOT_IMPLEMENTED = "not_implemented_fail_closed"
IMPLEMENTATION_STATUS_UNKNOWN = "unknown"


ABSTRACTION_MAP = {
    "PhysicalFieldFunctor": "PhysicalField",
    "PhysicalField": "PhysicalField",
    "BiologicalProcessFunctor": "BiologicalProcess",
    "BiologicalProcess": "BiologicalProcess",
    "TransportMechanismFunctor": "TransportMechanism",
    "TransportMechanism": "TransportMechanism",
    "ScaleCouplingFunctor": "ScaleCoupling",
    "ScaleCoupling": "ScaleCoupling",
    "PhenologyBase": "BiologicalProcess",
}

ABSTRACTION_BY_CATEGORY = {
    "physical_field": "PhysicalField",
    "biological_process": "BiologicalProcess",
    "transport_mechanism": "TransportMechanism",
    "scale_coupling": "ScaleCoupling",
}

ROLE_BY_CATEGORY = {
    "physical_field": "field",
    "biological_process": "reaction",
    "transport_mechanism": "transport",
    "scale_coupling": "scale_coupling",
}

ROLE_BY_ABSTRACTION = {
    "PhysicalField": "field",
    "BiologicalProcess": "reaction",
    "TransportMechanism": "transport",
    "ScaleCoupling": "scale_coupling",
}

MANAGEMENT_FILES = {
    "irrigation.py",
    "harvest.py",
    "grazing.py",
    "tillage.py",
    "fertilization.py",
    "fire_management.py",
    "coordinator.py",
}

MANAGEMENT_ROLE_BY_FILE = {
    "coordinator.py": "scale_coupling",
    "fertilization.py": "port",
    "fire_management.py": "reaction",
    "grazing.py": "reaction",
    "harvest.py": "port",
    "irrigation.py": "port",
    "tillage.py": "port",
}

THEORY_ALIASES = {
    "anthroposphere\\emissions\\atmosphere_interface_contract.py": "anthroposphere_emissions_mapping.md",
    "anthroposphere\\emissions\\coupling_to_atmosphere.py": "anthropogenic_emissions.md",
    "anthroposphere\\emissions\\sector_allocation.py": "anthroposphere_sector_allocation.md",
    "anthroposphere\\emissions\\species_mapping.py": "anthroposphere_emissions_mapping.md",
    "anthroposphere\\inventory.py": "anthropogenic_emissions.md",
    "anthroposphere\\interface_mapping.py": "anthroposphere_emissions_mapping.md",
    "anthroposphere\\land_use\\mapping.py": "anthroposphere_land_use_transitions.md",
    "anthroposphere\\land_use\\transitions.py": "anthroposphere_land_use_transitions.md",
    "anthroposphere\\management\\coordinator.py": "anthroposphere_management_coordinator.md",
    "anthroposphere\\management\\fertilization.py": "anthroposphere_management_fertilization.md",
    "anthroposphere\\management\\fire_management.py": "anthroposphere_management_fire.md",
    "anthroposphere\\management\\grazing.py": "anthroposphere_management_grazing.md",
    "anthroposphere\\management\\harvest.py": "anthroposphere_management_harvest.md",
    "anthroposphere\\management\\irrigation.py": "anthroposphere_management_irrigation.md",
    "anthroposphere\\management\\tillage.py": "anthroposphere_management_tillage.md",
    "atmosphere\\chemistry.py": "chemistry.md",
    "atmosphere\\environmental_forcing.py": "atmosphere_environmental_forcing.md",
    "atmosphere\\radiation.py": "radiation.md",
    "atmosphere\\radiation_energy_conservation.py": "radiation.md",
    "atmosphere\\transport\\AtmosphericAdvectionDiffusionFunctor.py": "atmosphere_advection_diffusion.md",
    "atmosphere\\transport\\dry_deposition_aggregator.py": "atmosphere_dry_deposition_aggregator.md",
    "atmosphere\\transport\\eddy_diffusion_blm.py": "atmosphere_eddy_diffusion_blm.md",
    "atmosphere\\turbulence.py": "turbulence.md",
    "biosphere\\bvoc\\megan_model.py": "bvoc_emissions.md",
    "biosphere\\bvoc\\responses.py": "bvoc_emissions.md",
    "biosphere\\bvoc\\synthesis\\isoprene.py": "bvoc_emissions.md",
    "biosphere\\bvoc\\synthesis\\monoterpene.py": "bvoc_emissions.md",
    "biosphere\\bvoc\\synthesis\\sesquiterpene.py": "bvoc_emissions.md",
    "biosphere\\growth\\biomass_allocation.py": "growth.md",
    "biosphere\\growth\\phenology_growth.py": "growth.md",
    "biosphere\\growth\\structural_growth.py": "growth.md",
    "biosphere\\phenology\\base.py": "phenology.md",
    "biosphere\\phenology\\models.py": "phenology.md",
    "biosphere\\phenology\\phenology_functor.py": "phenology.md",
    "biosphere\\photosynthesis\\farquhar.py": "photosynthesis.md",
    "biosphere\\photosynthesis\\stomatal.py": "photosynthesis.md",
    "biosphere\\respiration\\autotrophic_respiration.py": "respiration.md",
    "biosphere\\respiration\\growth_respiration.py": "respiration.md",
    "biosphere\\respiration\\maintenance_respiration.py": "respiration.md",
    "biosphere\\stress\\nutrient_stress.py": "stress.md",
    "biosphere\\stress\\ozone_stress.py": "stress.md",
    "biosphere\\stress\\thermal_stress.py": "stress.md",
    "biosphere\\stress\\water_stress.py": "stress.md",
    "biosphere\\surrogates\\simple_photosynthesis.py": "surrogates.md",
    "hydrosphere\\evapotranspiration\\penman_monteith.py": "hydrosphere_penman_monteith.md",
    "hydrosphere\\evapotranspiration\\priestley_taylor.py": "hydrosphere_priestley_taylor.md",
    "hydrosphere\\groundwater\\dynamics.py": "hydrosphere_groundwater_dynamics.md",
    "hydrosphere\\runoff\\subsurface.py": "hydrosphere_subsurface_runoff.md",
    "hydrosphere\\runoff\\surface.py": "hydrosphere_surface_runoff.md",
    "hydrosphere\\snow\\albedo.py": "hydrosphere_snow_albedo.md",
    "hydrosphere\\snow\\snowpack.py": "hydrosphere_snowpack.md",
    "hydrosphere\\soil_moisture\\bucket.py": "hydrosphere_bucket.md",
    "hydrosphere\\soil_moisture\\richards.py": "hydrosphere_richards.md",
    "hydrosphere\\transport\\capillary_rise.py": "hydrosphere_capillary_rise.md",
    "hydrosphere\\transport\\vertical_percolation.py": "hydrosphere_vertical_percolation.md",
    "isotopes\\functor.py": "isotopes.md",
    "pedosphere\\carbon_pools\\carbon_pools_functor.py": "decomposition.md",
    "pedosphere\\carbon_pools\\pools_matrix.py": "decomposition.md",
    "pedosphere\\decomposition\\first_order.py": "decomposition.md",
    "pedosphere\\decomposition\\microbial_enzyme.py": "decomposition.md",
    "pedosphere\\interface_mapping.py": "pedosphere_transport.md",
    "pedosphere\\methane\\budget_coupler.py": "wetland_methane.md",
    "pedosphere\\methane\\oxidation.py": "wetland_methane.md",
    "pedosphere\\methane\\production.py": "wetland_methane.md",
    "pedosphere\\methane\\scales.py": "wetland_methane.md",
    "pedosphere\\methane\\transport.py": "wetland_methane.md",
    "pedosphere\\nutrient_cycling\\denitrification.py": "denitrification.md",
    "pedosphere\\nutrient_cycling\\mineralization.py": "mineralization.md",
    "pedosphere\\nutrient_cycling\\nitrification.py": "nitrification.md",
    "pedosphere\\nutrient_cycling\\nitrogen.py": "nitrogen_cycling.md",
    "pedosphere\\nutrient_cycling\\nitrogen_leaching.py": "nitrogen_leaching.md",
    "pedosphere\\nutrient_cycling\\root_nutrient_uptake.py": "root_nutrient_uptake.md",
    "pedosphere\\nutrient_cycling\\stoichiometry.py": "stoichiometry_constraint.md",
    "pedosphere\\transport\\gas_diffusion.py": "pedosphere_transport.md",
    "pedosphere\\transport\\SoilWaterTransportFunctor.py": "soil_hydraulics.md",
    "pedosphere\\transport\\solute_advection_diffusion.py": "pedosphere_transport.md",
    "transport\\advection.py": "transport_linear_advection.md",
    "transport\\advection_diffusion.py": "transport_advection_diffusion.md",
    "transport\\diffusion.py": "transport_fickian_diffusion.md",
}


def _name_of(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _name_of(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    return ""


def _const_str(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                parts.append("{expr}")
        return "".join(parts)
    if isinstance(node, ast.Attribute):
        return _name_of(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_const_str(elt) for elt in node.elts]
        return "|".join(value for value in values if value)
    if isinstance(node, ast.Dict):
        keys = [_const_str(key) for key in node.keys]
        return "|".join(key for key in keys if key)
    if isinstance(node, ast.DictComp):
        return "dynamic_parameter_mapping"
    return ""


def _find_method(cls: ast.ClassDef, method_name: str) -> ast.FunctionDef | None:
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == method_name:
            return stmt
    return None


def _method_return_sequence(cls: ast.ClassDef, method_name: str) -> str:
    fn = _find_method(cls, method_name)
    if fn is None:
        return ""
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Return):
            return _const_str(stmt.value)
    return ""


def _call_self_method_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        if node.func.value.id == "self":
            return node.func.attr
    return ""


def _assigned_value(scope: ast.AST, name: str) -> ast.AST | None:
    for stmt in ast.walk(scope):
        if isinstance(stmt, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in stmt.targets):
                return stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
                return stmt.value
    return None


def _method_argument_default(cls: ast.ClassDef, name: str) -> ast.AST | None:
    methods = [stmt for stmt in cls.body if isinstance(stmt, ast.FunctionDef)]
    methods.sort(key=lambda method: method.name != "__init__")
    for method in methods:
        positional = list(method.args.args)
        defaults = [None] * (len(positional) - len(method.args.defaults)) + list(
            method.args.defaults
        )
        for argument, default in zip(positional, defaults):
            if argument.arg == name and default is not None:
                return default
        for argument, default in zip(method.args.kwonlyargs, method.args.kw_defaults):
            if argument.arg == name and default is not None:
                return default
    return None


def _static_string_sequence(
    node: ast.AST | None,
    *,
    cls: ast.ClassDef,
    module: ast.Module | None,
    environment: dict[str, str] | None = None,
    seen: set[str] | None = None,
) -> list[str] | None:
    """Resolve metadata string sequences without importing scientific modules."""

    if node is None:
        return None
    environment = environment or {}
    seen = set() if seen is None else set(seen)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[str] = []
        for element in node.elts:
            target = element.value if isinstance(element, ast.Starred) else element
            resolved = _static_string_sequence(
                target,
                cls=cls,
                module=module,
                environment=environment,
                seen=seen,
            )
            if resolved is None:
                return None
            values.extend(resolved)
        return values
    if isinstance(node, ast.Name):
        if node.id in environment:
            return [environment[node.id]]
        if node.id in seen:
            return None
        seen.add(node.id)
        value = None
        for method in cls.body:
            if isinstance(method, ast.FunctionDef):
                value = _assigned_value(method, node.id)
                if value is not None:
                    break
        if value is None:
            value = _method_argument_default(cls, node.id)
        if value is None and module is not None:
            value = _assigned_value(module, node.id)
        return _static_string_sequence(
            value,
            cls=cls,
            module=module,
            environment=environment,
            seen=seen,
        )
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            value = _assigned_value(cls, node.attr)
            return _static_string_sequence(
                value,
                cls=cls,
                module=module,
                environment=environment,
                seen=seen | {node.attr},
            )
        return None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                resolved = _static_string_sequence(
                    value.value,
                    cls=cls,
                    module=module,
                    environment=environment,
                    seen=seen,
                )
                if resolved is None or len(resolved) != 1:
                    return None
                parts.append(resolved[0])
            else:
                return None
        return ["".join(parts)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string_sequence(
            node.left,
            cls=cls,
            module=module,
            environment=environment,
            seen=seen,
        )
        right = _static_string_sequence(
            node.right,
            cls=cls,
            module=module,
            environment=environment,
            seen=seen,
        )
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Dict):
        keys: list[str] = []
        for key, value in zip(node.keys, node.values):
            target = value if key is None else key
            resolved = _static_string_sequence(
                target,
                cls=cls,
                module=module,
                environment=environment,
                seen=seen,
            )
            if resolved is None:
                return None
            keys.extend(resolved)
        return keys
    if isinstance(node, ast.ListComp) and len(node.generators) == 1:
        generator = node.generators[0]
        if generator.ifs or generator.is_async or not isinstance(generator.target, ast.Name):
            return None
        iterable = _static_string_sequence(
            generator.iter,
            cls=cls,
            module=module,
            environment=environment,
            seen=seen,
        )
        if iterable is None:
            return None
        result: list[str] = []
        for item in iterable:
            resolved = _static_string_sequence(
                node.elt,
                cls=cls,
                module=module,
                environment={**environment, generator.target.id: item},
                seen=seen,
            )
            if resolved is None:
                return None
            result.extend(resolved)
        return result
    if isinstance(node, ast.Call):
        function_name = _name_of(node.func)
        if function_name in {"sorted", "list", "tuple", "set"} and node.args:
            resolved = _static_string_sequence(
                node.args[0],
                cls=cls,
                module=module,
                environment=environment,
                seen=seen,
            )
            if resolved is None:
                return None
            if function_name == "sorted":
                return sorted(resolved)
            if function_name in {"list", "tuple"}:
                return list(resolved)
            return list(dict.fromkeys(resolved))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "keys":
            return _static_string_sequence(
                node.func.value,
                cls=cls,
                module=module,
                environment=environment,
                seen=seen,
            )
        if function_name == "dict.fromkeys" and node.args:
            resolved = _static_string_sequence(
                node.args[0],
                cls=cls,
                module=module,
                environment=environment,
                seen=seen,
            )
            return None if resolved is None else list(dict.fromkeys(resolved))
    return None


def _configuration_default_ports(cls: ast.ClassDef) -> dict[str, str]:
    """Local names bound to a configuration lookup config.get("key", "default").

    A port selected by configuration is declared statically by its documented
    default; only used to resolve metadata input and output variables.
    """
    defaults: dict[str, str] = {}
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for stmt in ast.walk(method):
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "get"
                and len(stmt.value.args) == 2
                and all(
                    isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                    for argument in stmt.value.args
                )
            ):
                continue
            defaults.setdefault(stmt.targets[0].id, stmt.value.args[1].value)
    return defaults


def _static_string_mapping(node: ast.AST | None, *, cls: ast.ClassDef) -> dict[str, str] | None:
    """Resolve self.NAME or NAME to a class-level dict literal of string constants."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        name = node.attr
    elif isinstance(node, ast.Name):
        name = node.id
    else:
        return None
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in stmt.targets
        ):
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == name:
            value = stmt.value
        else:
            continue
        if not isinstance(value, ast.Dict):
            return None
        mapping: dict[str, str] = {}
        for key, item in zip(value.keys, value.values):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(item, ast.Constant)
                and isinstance(item.value, str)
            ):
                return None
            mapping[key.value] = item.value
        return mapping
    return None


def _class_constant(cls: ast.ClassDef, name: str) -> str:
    for stmt in cls.body:
        target_name = ""
        value_node: ast.AST | None = None
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    target_name = target.id
                    value_node = stmt.value
                    break
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == name:
                target_name = stmt.target.id
                value_node = stmt.value
        if target_name == name:
            return _const_str(value_node)
    return ""


def _metadata_category_value(category: str) -> str:
    if not category:
        return ""
    return category.split(".")[-1].strip().lower()


def _infer_abstraction(base_names: list[str], metadata_category: str = "") -> str:
    for base in base_names:
        short = base.split(".")[-1]
        if short in ABSTRACTION_MAP:
            return ABSTRACTION_MAP[short]
    category = _metadata_category_value(metadata_category)
    return ABSTRACTION_BY_CATEGORY.get(category, "Unknown")


def _snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return value.strip("_").lower()


def _process_id(file_path: str, class_name: str) -> str:
    module = file_path[:-3].replace("\\", ".")
    return f"{module}.{class_name}"


def _certificate_path(process_id: str) -> str:
    safe_name = _snake_case(process_id.replace("relic.scientific.", ""))
    return str(CERTIFICATE_ROOT / f"{safe_name}.yaml")


def _theory_path(file_path: str, class_name: str) -> tuple[str, str]:
    rel = file_path.replace("relic\\scientific\\", "")
    candidates: list[str] = []
    if rel in THEORY_ALIASES:
        candidates.append(THEORY_ALIASES[rel])

    path = Path(rel)
    stem = _snake_case(path.stem)
    parts = [part for part in path.parts[:-1] if part not in {"__pycache__"}]
    if parts:
        candidates.append("_".join([_snake_case(parts[0]), stem]) + ".md")
        candidates.append("_".join(_snake_case(part) for part in [*parts, stem]) + ".md")
    candidates.append(stem + ".md")
    candidates.append(_snake_case(class_name) + ".md")

    seen: set[str] = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        candidate_path = THEORY_ROOT / candidate
        if candidate_path.exists():
            return str(candidate_path), "existing"

    fallback = unique_candidates[0] if unique_candidates else f"{_snake_case(class_name)}.md"
    return str(THEORY_ROOT / fallback), "missing"


def _canonical_role(row: dict[str, Any]) -> str:
    rel = row["file_path"].replace("relic\\scientific\\", "")
    category = _metadata_category_value(row.get("metadata_category", ""))
    variable_kind = str(row.get("metadata_variable_kind", "")).lower()
    output_text = str(row.get("metadata_output_variables", "")).lower()
    output_names = [
        item.strip()
        for item in output_text.split("|")
        if item.strip()
    ]
    non_diagnostic_outputs = [
        item for item in output_names if not item.startswith("diagnostics.")
    ]
    class_name = str(row.get("class_name", "")).lower()
    filename = Path(rel).name

    if "\\management\\" in rel and filename in MANAGEMENT_FILES:
        return MANAGEMENT_ROLE_BY_FILE.get(filename, "port")
    if "environmental_forcing" in rel:
        return "port"
    if (
        str(row.get("abstraction", "")) == "BiologicalProcess"
        and (
            "growth_rate_" in output_text
            or "carbon_storage_change" in output_text
            or "leaf_area_growth" in output_text
        )
    ):
        return "reaction"
    if (
        "diagnostic" in variable_kind
        or (output_names and not non_diagnostic_outputs)
        or "validator" in class_name
    ):
        return "observation"
    if "interface_mapping" in rel or rel.endswith("\\mapping.py"):
        return "scale_coupling"
    return ROLE_BY_CATEGORY.get(category) or ROLE_BY_ABSTRACTION.get(
        str(row.get("abstraction", "")),
        "unknown",
    )


def _process_status(row: dict[str, Any]) -> str:
    if row.get("parse_error"):
        return "parse_error"
    if _support_only_reason(row):
        return "support_only"
    if row.get("canonical_role") == "unknown" and not row.get("forward_signature"):
        return "support_only"
    return "process"


def _support_only_reason(row: dict[str, Any]) -> str:
    """Return the audited reason why a candidate is not a runtime process."""
    if row.get("canonical_role") == "unknown" and not row.get("forward_signature"):
        return "No executable scientific forward contract was found."
    return ""


def _interface_status(row: dict[str, Any]) -> str:
    if row.get("forward_signature"):
        return "captured"
    if row.get("abstraction") != "Unknown":
        return "inherited"
    return "missing"


def _io_status(row: dict[str, Any]) -> str:
    if row.get("implementation_status") == IMPLEMENTATION_STATUS_NOT_IMPLEMENTED:
        return "not_applicable_runtime_unbound"
    if row.get("metadata_input_variables") or row.get("metadata_output_variables"):
        return "captured"
    return "missing"


def _structural_hooks(cls: ast.ClassDef) -> str:
    hook_names = {
        "declared_forms",
        "build_hodge_terms",
        "energy_terms",
        "entropy_terms",
        "reversible_structure",
        "dissipative_structure",
        "reaction_stoichiometry",
        "ports",
        "diagnostic_operator",
        "get_dec_form_metadata",
        "build_hodge_contract",
        "contribute_iphs_structure",
    }
    present = sorted(
        stmt.name
        for stmt in cls.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name in hook_names
    )
    return "|".join(present)


def _certificate_status(certificate_path: str) -> tuple[str, str]:
    path = Path(certificate_path)
    if not path.exists():
        return "missing", ""
    try:
        certificate = load_iphs_equivalence_certificate(path)
        audit = certificate.validate()
    except Exception as exc:
        return "invalid", f"{type(exc).__name__}: {exc}"
    if audit.get("passed"):
        return "valid", ""
    return "invalid", "|".join(str(issue) for issue in audit.get("issues", ()))


def _implementation_status(certificate_path: str) -> tuple[str, str]:
    """Derive executable availability from the certificate-owned runtime claim."""

    path = Path(certificate_path)
    if not path.exists():
        return IMPLEMENTATION_STATUS_UNKNOWN, "missing_certificate"
    try:
        certificate = load_iphs_equivalence_certificate(path)
    except Exception as exc:
        return IMPLEMENTATION_STATUS_UNKNOWN, f"{type(exc).__name__}: {exc}"

    replacement_entrypoint = str(
        certificate.runtime_replacement_map.get("replacement_entrypoint", "")
    ).strip()
    if certificate.declares_fail_closed_nonimplementation():
        return IMPLEMENTATION_STATUS_NOT_IMPLEMENTED, replacement_entrypoint
    return IMPLEMENTATION_STATUS_IMPLEMENTED, ""


def _certificate_level_status(
    certificate_path: str,
) -> tuple[str, str, str, str, str, str, str]:
    path = Path(certificate_path)
    if not path.exists():
        return "", "", "missing", "missing", "", "missing", ""
    try:
        certificate = load_iphs_equivalence_certificate(path)
        audit = certificate.validate()
    except Exception as exc:
        return "", "", "invalid", f"{type(exc).__name__}: {exc}", "", "invalid", ""

    level = certificate.iphs_level
    if level is None:
        return "", "", "missing", "iphs_level is required", "", "missing", ""
    definition = IPHS_LEVEL_DEFINITIONS.get(level)
    level_name = str(definition["name"]) if definition else "invalid"
    runtime_level = audit.get("runtime_verified_level")
    runtime_status = str(audit.get("runtime_evidence_status", "missing"))
    runtime_issue_count = str(audit.get("runtime_evidence_issue_count", 0))
    if audit.get("passed"):
        return (
            str(level),
            level_name,
            "valid",
            "certificate",
            "" if runtime_level is None else str(runtime_level),
            runtime_status,
            runtime_issue_count,
        )
    return (
        str(level),
        level_name,
        "invalid",
        "certificate",
        "" if runtime_level is None else str(runtime_level),
        runtime_status,
        runtime_issue_count,
    )


def _split_inventory_variables(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split("|") if item.strip()}


def _compare_interface_variable_sets(
    metadata_values: set[str],
    certificate_values: set[str],
) -> tuple[bool, bool]:
    if metadata_values == certificate_values:
        return True, False
    if not metadata_values:
        return True, True

    collection_placeholders = {
        value
        for value in metadata_values
        if value.endswith("_inputs") or value.endswith("_outputs")
    }
    scalar_placeholders = {
        value
        for value in metadata_values
        if value.endswith("_key") or value in {"source_key", "tracer_key"}
    }
    wildcard_values = {value for value in metadata_values if value.endswith(".*")}
    concrete_metadata = metadata_values - collection_placeholders - scalar_placeholders - wildcard_values
    concrete_certificate = set(certificate_values)
    for wildcard in wildcard_values:
        prefix = wildcard[:-1]
        concrete_certificate = {
            value for value in concrete_certificate if not value.startswith(prefix)
        }
    if not concrete_metadata.issubset(concrete_certificate):
        return False, False
    remaining_certificate = concrete_certificate - concrete_metadata
    if collection_placeholders:
        return True, True
    if len(remaining_certificate) <= len(scalar_placeholders):
        return True, True
    return False, False


def _certificate_contract_consistency(
    certificate_path: str,
    *,
    metadata_inputs: str,
    metadata_outputs: str,
    theory_document_path: str,
) -> tuple[str, str]:
    path = Path(certificate_path)
    if not path.exists():
        return "missing_certificate", "missing_certificate"
    try:
        certificate = load_iphs_equivalence_certificate(path)
    except Exception:
        return "invalid_certificate", "invalid_certificate"

    metadata_input_set = _split_inventory_variables(metadata_inputs)
    metadata_output_set = _split_inventory_variables(metadata_outputs)
    certificate_input_set = set(certificate.input_variables)
    certificate_output_set = set(certificate.output_variables)
    mismatches: list[str] = []
    dynamic_binding = False
    input_match, input_dynamic = _compare_interface_variable_sets(
        metadata_input_set,
        certificate_input_set,
    )
    output_match, output_dynamic = _compare_interface_variable_sets(
        metadata_output_set,
        certificate_output_set,
    )
    dynamic_binding = input_dynamic or output_dynamic
    if not input_match:
        mismatches.append(
            "inputs:metadata={} certificate={}".format(
                sorted(metadata_input_set),
                sorted(certificate_input_set),
            )
        )
    if not output_match:
        mismatches.append(
            "outputs:metadata={} certificate={}".format(
                sorted(metadata_output_set),
                sorted(certificate_output_set),
            )
        )
    interface_status = (
        "matched"
        if not mismatches and not dynamic_binding
        else "dynamic_binding"
        if not mismatches
        else "mismatch:" + ";".join(mismatches)
    )

    theory_path = Path(theory_document_path)
    if theory_path.is_absolute():
        try:
            theory_path = theory_path.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
    certificate_theory_path = Path(certificate.theory_document.replace("\\", "/"))
    normalized_theory = str(theory_path).replace("/", "\\").lower()
    normalized_certificate_theory = str(certificate_theory_path).replace("/", "\\").lower()
    theory_status = (
        "matched"
        if normalized_theory == normalized_certificate_theory
        else "mismatch:{}!={}".format(
            normalized_theory,
            normalized_certificate_theory,
        )
    )
    return interface_status, theory_status


def _state_advancement_status(certificate_path: str) -> str:
    path = Path(certificate_path)
    if not path.exists():
        return "missing_certificate"
    try:
        certificate = load_iphs_equivalence_certificate(path)
    except Exception:
        return "invalid_certificate"

    state_advancement = certificate.state_advancement_map
    if not state_advancement:
        return "missing"

    required_keys = (
        "potential_basis",
        "effort_variable",
        "state_advancer",
        "runtime_usage",
    )
    missing = [
        key
        for key in required_keys
        if not str(state_advancement.get(key, "")).strip()
    ]
    if missing:
        return "incomplete:" + ",".join(missing)
    return "complete"


def _state_advancement_specificity_status(certificate_path: str) -> str:
    path = Path(certificate_path)
    if not path.exists():
        return "missing_certificate"
    try:
        certificate = load_iphs_equivalence_certificate(path)
    except Exception:
        return "invalid_certificate"

    state_advancement = certificate.state_advancement_map
    runtime_usage = str(state_advancement.get("runtime_usage", "")).strip()
    state_advancer = str(state_advancement.get("state_advancer", "")).strip()
    explicit_stoichiometry = str(
        state_advancement.get("explicit_stoichiometric_usage", "")
    ).strip()

    issues: list[str] = []
    if not runtime_usage:
        issues.append("runtime_usage")
    if runtime_usage == GENERIC_RUNTIME_USAGE:
        issues.append("generic_runtime_usage")
    if state_advancer == GENERIC_STATE_ADVANCER or state_advancer.startswith(
        GENERIC_STATE_ADVANCER + " or "
    ):
        issues.append("generic_state_advancer")
    if "N_h r_h" in state_advancer and not explicit_stoichiometry:
        issues.append("missing_explicit_stoichiometric_usage")
    if issues:
        return "incomplete:" + ",".join(issues)
    return "specific"


def _source_state_advancement_status(row: dict[str, Any], source_text: str) -> str:
    if row.get("implementation_status") == IMPLEMENTATION_STATUS_NOT_IMPLEMENTED:
        return "not_applicable_runtime_unbound"
    role = str(row.get("canonical_role", ""))
    lower_source = source_text.lower()
    if role == "reaction":
        required_tokens = {
            "advance_reaction_network_call": "advance_reaction_network(",
            "stoichiometric_N_keyword": "stoichiometric_n=",
            "rates_keyword": "rates=",
        }
        missing = [
            label
            for label, token in required_tokens.items()
            if token not in lower_source
        ]
        if "n_h r_h" not in lower_source and "n_h * r_h" not in lower_source:
            missing.append("N_h_r_h_comment_or_doc")
        if missing:
            return "missing:" + ",".join(missing)
        return "complete"

    if role == "transport":
        evidence_tokens = (
            "_compute_iphs",
            "finalize_transport_updates(",
            "hodge",
            "darcy",
            "richards",
            "port flow",
            "port-flow",
            "flux divergence",
            "mass transfer",
            "overland flow",
        )
        if any(token in lower_source for token in evidence_tokens):
            return "complete"
        return "missing:transport_operator_evidence"

    if role in {"port", "scale_coupling", "field", "observation"}:
        evidence_tokens = (
            "port",
            "chain map",
            "chain_map",
            "observation",
            "diagnostic",
            "constraint",
            "operator",
            "potential",
            "field",
        )
        if any(token in lower_source for token in evidence_tokens):
            return "complete"
        return f"missing:{role}_operator_evidence"

    return "not_applicable"


def _runtime_replacement_status(certificate_path: str, row: dict[str, Any]) -> str:
    if row.get("implementation_status") == IMPLEMENTATION_STATUS_NOT_IMPLEMENTED:
        return IMPLEMENTATION_STATUS_NOT_IMPLEMENTED
    path = Path(certificate_path)
    if not path.exists():
        return "missing_certificate"
    try:
        certificate = load_iphs_equivalence_certificate(path)
        audit = certificate.validate()
    except Exception:
        return "invalid_certificate"
    if audit.get("runtime_evidence_status") == "verified":
        return "canonical_iphs_runtime_verified"
    runtime_map = certificate.runtime_replacement_map
    if (
        certificate.iphs_level is not None
        and certificate.iphs_level <= 3
        and runtime_map.get("runtime_uses_iphs") is False
        and runtime_map.get("legacy_removed") is False
        and isinstance(runtime_map.get("replacement_entrypoint"), str)
        and bool(runtime_map.get("replacement_entrypoint", "").strip())
        and row.get("source_state_advancement_status") == "complete"
    ):
        return "canonical_wrapper_declared"
    return "runtime_not_verified"


def _composition_contract_status(certificate_path: str, row: dict[str, Any]) -> str:
    path = Path(certificate_path)
    if not path.exists():
        return "missing_certificate"
    try:
        certificate = load_iphs_equivalence_certificate(path)
    except Exception:
        return "invalid_certificate"
    if certificate.composition_ports:
        return "ports_declared"
    if certificate.composition_management_map:
        return "composition_map_declared"
    if row.get("dependencies") or row.get("iphs_structural_hooks"):
        return "metadata_declared"
    return "not_declared"


def _overflow_routing_status(certificate_path: str) -> str:
    path = Path(certificate_path)
    if not path.exists():
        return "missing_certificate"
    try:
        certificate = load_iphs_equivalence_certificate(path)
    except Exception:
        return "invalid_certificate"
    if certificate.overflow_routing_map:
        return "routing_map_declared"
    return "missing"


def _parameter_management_status(
    certificate_path: str,
    metadata_parameters: str,
) -> str:
    """Classify parameter binding without inventing keys for pure morphisms."""

    try:
        certificate = load_iphs_equivalence_certificate(Path(certificate_path))
    except (OSError, TypeError, ValueError, KeyError, ImportError):
        return "missing_metadata"
    if certificate.parameter_free:
        return "parameter_free"
    return "metadata_declared" if metadata_parameters else "missing_metadata"


def _extract_forward_signature(cls: ast.ClassDef) -> str:
    fn = _find_method(cls, "forward")
    if fn is None:
        return ""
    args = []
    for a in fn.args.args:
        args.append(a.arg)
    return f"forward({', '.join(args)})"


def _format_dependency_declaration(declaration: Any) -> str:
    """Inventory text of one runtime dependency declaration.

    A list keeps its declaration order as ``a,b``; a structured declaration is
    written per category as ``required=a|b;optional=c;lagged=d`` (empty
    categories omitted).
    """
    if isinstance(declaration, (list, tuple)):
        return ",".join(str(name) for name in declaration)
    if isinstance(declaration, dict):
        return ";".join(
            f"{category}={'|'.join(str(name) for name in declaration[category])}"
            for category in ("required", "optional", "lagged")
            if declaration.get(category)
        )
    raise TypeError(f"Unsupported dependency declaration: {declaration!r}")


def _declared_dependencies(rel: str, class_name: str) -> str:
    """Static dependency declaration of one class from its single owner.

    The value is the class's own ``get_dependencies()`` evaluated on an
    instance built with its documented minimal arguments (the shared
    ``test.utils.functor_minimal_construction`` owner), so config-driven,
    structured and instance-level default declarations are reported exactly
    as the simulator receives them; source literals are never parsed. A class
    that is not a Functor, or that cannot be constructed minimally (listed and
    re-verified with its reason), has no runtime declaration and reports "".
    """
    module_parts = rel[:-3].replace("\\", ".").split(".")
    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    return _runtime_dependency_text(f"{'.'.join(module_parts)}:{class_name}")


@lru_cache(maxsize=None)
def _runtime_dependency_text(class_path: str) -> str:
    """Cached per class: construction is deterministic for the minimal arguments."""
    from relic.core.functor import FunctorBase
    from test.utils.functor_minimal_construction import (
        NOT_MINIMALLY_CONSTRUCTIBLE,
        construct_minimal,
        resolve_class,
        verify_not_minimally_constructible,
    )

    functor_cls = resolve_class(class_path)
    if not (isinstance(functor_cls, type) and issubclass(functor_cls, FunctorBase)):
        return ""
    if class_path in NOT_MINIMALLY_CONSTRUCTIBLE:
        verify_not_minimally_constructible(class_path)
        return ""
    return _format_dependency_declaration(construct_minimal(class_path).get_dependencies())


def _extract_parameter_namespace(cls: ast.ClassDef) -> str:
    init_fn = _find_method(cls, "__init__")
    if init_fn is None:
        return ""
    for stmt in ast.walk(init_fn):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Attribute) and target.attr in {"parameter_namespace", "namespace"}:
                    return _const_str(stmt.value)
    return ""


def _extract_metadata_from_call(
    call: ast.Call,
    cls: ast.ClassDef | None = None,
    module: ast.Module | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    func_name = _name_of(call.func).split(".")[-1]
    if func_name != "FunctorMetadata":
        return result
    for kw in call.keywords:
        if kw.arg is None:
            continue
        if kw.arg in {
            "name",
            "version",
            "description",
            "category",
            "execution_phase",
            "mathematical_expression",
            "spatial_scale",
            "variable_kind",
        }:
            if kw.arg == "name" and cls is not None:
                resolved_name = _static_string_sequence(
                    kw.value,
                    cls=cls,
                    module=module,
                )
                result[kw.arg] = (
                    resolved_name[0]
                    if resolved_name is not None and len(resolved_name) == 1
                    else _const_str(kw.value)
                )
            else:
                result[kw.arg] = _const_str(kw.value)
        if kw.arg in {"input_variables", "output_variables", "conservation_laws"}:
            method_name = _call_self_method_name(kw.value)
            if cls is not None and method_name:
                result[kw.arg] = _method_return_sequence(cls, method_name)
            elif cls is not None:
                resolved = _static_string_sequence(
                    kw.value,
                    cls=cls,
                    module=module,
                    environment=_configuration_default_ports(cls),
                )
                result[kw.arg] = (
                    "|".join(resolved) if resolved is not None else _const_str(kw.value)
                )
            else:
                result[kw.arg] = _const_str(kw.value)
        if kw.arg == "references":
            if isinstance(kw.value, (ast.List, ast.Tuple)):
                refs = []
                for x in kw.value.elts:
                    refs.append(_const_str(x))
                result["references"] = "|".join([r for r in refs if r])
            else:
                result["references"] = _const_str(kw.value)
        if kw.arg == "parameters":
            result["parameters_decl"] = _name_of(kw.value) or _const_str(kw.value)
    return result


def _extract_metadata(
    cls: ast.ClassDef,
    module: ast.Module | None = None,
) -> dict[str, str]:
    """Return the constructor metadata declaration of one Functor class.

    The inventory identity of a class is the metadata its default constructor
    declares.  Metadata calls inside ``__init__`` are therefore read first;
    helper methods that build metadata for non-default constructor selections
    are only consulted when ``__init__`` declares none.
    """

    def _collect(scope: ast.AST) -> dict[str, str]:
        found: dict[str, str] = {}
        for stmt in ast.walk(scope):
            if isinstance(stmt, ast.Call):
                # Path 1: metadata=FunctorMetadata(...) in super().__init__(...)
                for kw in stmt.keywords:
                    if kw.arg == "metadata" and isinstance(kw.value, ast.Call):
                        meta = _extract_metadata_from_call(kw.value, cls, module)
                        found.update(meta)
                # Path 2: direct FunctorMetadata(...) assignment in init
                meta = _extract_metadata_from_call(stmt, cls, module)
                if meta:
                    found.update(meta)
        return found

    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            found = _collect(node)
            if found:
                return found
    return _collect(cls)


def _is_candidate_functor(cls: ast.ClassDef) -> bool:
    base_names = [_name_of(b).split(".")[-1] for b in cls.bases]
    if any(name.endswith("Functor") for name in base_names):
        return True
    if cls.name.endswith("Functor"):
        return True
    return False


def _is_abstract_class(cls: ast.ClassDef) -> bool:
    """Return whether a class directly declares an unimplemented ABC method."""
    abstract_decorators = {
        "abstractmethod",
        "abstractclassmethod",
        "abstractstaticmethod",
        "abstractproperty",
    }
    for statement in cls.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in statement.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if _name_of(target).split(".")[-1] in abstract_decorators:
                return True
    return False


InventoryClassEntry = tuple[Path, ast.Module, ast.ClassDef]
InventoryClassIndex = dict[str, list[InventoryClassEntry]]


def _inventory_class_hierarchy(
    cls: ast.ClassDef,
    path: Path,
    module: ast.Module,
    class_index: InventoryClassIndex,
    visited: set[tuple[str, str]] | None = None,
) -> list[InventoryClassEntry]:
    """Return a class and uniquely resolved bases without importing runtime code."""
    marker = (str(path.resolve()), cls.name)
    visited = set(visited or set())
    if marker in visited:
        return []
    visited.add(marker)
    entries: list[InventoryClassEntry] = [(path, module, cls)]
    local_classes = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.ClassDef)
    }
    for base in cls.bases:
        base_name = _name_of(base).split(".")[-1]
        if not base_name:
            continue
        local_base = local_classes.get(base_name)
        if local_base is not None:
            base_entry: InventoryClassEntry | None = (path, module, local_base)
        else:
            candidates = class_index.get(base_name, [])
            base_entry = candidates[0] if len(candidates) == 1 else None
        if base_entry is None:
            continue
        entries.extend(
            _inventory_class_hierarchy(
                base_entry[2],
                base_entry[0],
                base_entry[1],
                class_index,
                visited,
            )
        )
    return entries


def _metadata_sequence_value(
    node: ast.AST | None,
    *,
    cls: ast.ClassDef,
    module: ast.Module,
    current_values: list[str],
    attribute_name: str,
) -> list[str] | None:
    """Resolve a finite metadata sequence while preserving current declarations."""
    if node is None:
        return None
    if isinstance(node, ast.Starred):
        return _metadata_sequence_value(
            node.value,
            cls=cls,
            module=module,
            current_values=current_values,
            attribute_name=attribute_name,
        )
    if isinstance(node, ast.Name):
        assigned = _assigned_value(cls, node.id)
        if assigned is not None and assigned is not node:
            return _metadata_sequence_value(
                assigned,
                cls=cls,
                module=module,
                current_values=current_values,
                attribute_name=attribute_name,
            )
    if isinstance(node, ast.Attribute):
        dotted = _name_of(node)
        if dotted.endswith(f"metadata.{attribute_name}"):
            return list(current_values)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[str] = []
        for element in node.elts:
            resolved = _metadata_sequence_value(
                element,
                cls=cls,
                module=module,
                current_values=current_values,
                attribute_name=attribute_name,
            )
            if resolved is None:
                return None
            values.extend(resolved)
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _metadata_sequence_value(
            node.left,
            cls=cls,
            module=module,
            current_values=current_values,
            attribute_name=attribute_name,
        )
        right = _metadata_sequence_value(
            node.right,
            cls=cls,
            module=module,
            current_values=current_values,
            attribute_name=attribute_name,
        )
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Call):
        function_name = _name_of(node.func)
        if function_name in {"list", "tuple", "set", "sorted", "dict.fromkeys"} and node.args:
            argument_values = _metadata_sequence_value(
                node.args[0],
                cls=cls,
                module=module,
                current_values=current_values,
                attribute_name=attribute_name,
            )
            if argument_values is None:
                return None
            if function_name == "sorted":
                return sorted(argument_values)
            return (
                list(dict.fromkeys(argument_values))
                if function_name in {"set", "dict.fromkeys"}
                else list(argument_values)
            )
    if (
        isinstance(node, (ast.ListComp, ast.GeneratorExp))
        and len(node.generators) == 1
        and isinstance(node.generators[0].target, ast.Name)
        and not node.generators[0].is_async
    ):
        # A configuration-dependent selection over a finite sequence: the filters
        # select ports per configuration, so the union over configurations is the
        # unfiltered image. The element is the item itself or a constant-mapping
        # rename MAP.get(item, item) of a class-level dict literal.
        generator = node.generators[0]
        items = _metadata_sequence_value(
            generator.iter,
            cls=cls,
            module=module,
            current_values=current_values,
            attribute_name=attribute_name,
        )
        if items is None:
            return None
        element = node.elt
        if isinstance(element, ast.Name) and element.id == generator.target.id:
            return list(items)
        if (
            isinstance(element, ast.Call)
            and isinstance(element.func, ast.Attribute)
            and element.func.attr == "get"
            and len(element.args) == 2
            and all(
                isinstance(argument, ast.Name) and argument.id == generator.target.id
                for argument in element.args
            )
        ):
            mapping = _static_string_mapping(element.func.value, cls=cls)
            if mapping is None:
                return None
            return [mapping.get(item, item) for item in items]
        return None
    return _static_string_sequence(
        node,
        cls=cls,
        module=module,
    )


def _effective_metadata_sequence(
    cls: ast.ClassDef,
    path: Path,
    module: ast.Module,
    class_index: InventoryClassIndex,
    initial_values: list[str],
    attribute_name: str,
) -> list[str]:
    """Apply finite metadata sequence mutations declared by a class hierarchy."""
    values = list(dict.fromkeys(initial_values))
    for _, owner_module, owner_class in _inventory_class_hierarchy(
        cls,
        path,
        module,
        class_index,
    ):
        initializer = next(
            (
                node
                for node in owner_class.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        if initializer is None:
            continue
        operations = sorted(
            (
                node
                for node in ast.walk(initializer)
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Call))
            ),
            key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
        )
        for node in operations:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = _name_of(node.func.value)
                if not owner.endswith(f"metadata.{attribute_name}") or not node.args:
                    continue
                resolved = _metadata_sequence_value(
                    node.args[0],
                    cls=owner_class,
                    module=owner_module,
                    current_values=values,
                    attribute_name=attribute_name,
                )
                if resolved is None:
                    continue
                if node.func.attr == "append" and len(resolved) == 1:
                    values = list(dict.fromkeys([*values, resolved[0]]))
                elif node.func.attr == "extend":
                    values = list(dict.fromkeys([*values, *resolved]))
                continue

            if isinstance(node, ast.Assign):
                targets = node.targets
                value_node = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value_node = node.value
            else:
                continue
            if value_node is None or not any(
                _name_of(target).endswith(f"metadata.{attribute_name}")
                for target in targets
            ):
                continue
            resolved = _metadata_sequence_value(
                value_node,
                cls=owner_class,
                module=owner_module,
                current_values=values,
                attribute_name=attribute_name,
            )
            if resolved is not None:
                if any(isinstance(target, ast.Subscript) for target in targets):
                    # A slice assignment (metadata.input_variables[i:j] = ...) replaces
                    # ports in one configuration only; the declared interface of the
                    # class is the union over its configurations.
                    values = list(dict.fromkeys([*values, *resolved]))
                else:
                    values = list(dict.fromkeys(resolved))
    return values


def _hierarchy_source_text(
    cls: ast.ClassDef,
    path: Path,
    module: ast.Module,
    class_index: InventoryClassIndex,
) -> str:
    """Collect source evidence from the concrete class and uniquely resolved bases."""
    source_parts: list[str] = []
    seen_paths: set[Path] = set()
    for owner_path, _, _ in _inventory_class_hierarchy(
        cls,
        path,
        module,
        class_index,
    ):
        resolved_path = owner_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        source_parts.append(owner_path.read_text(encoding="utf-8-sig"))
    return "\n".join(source_parts)


@lru_cache(maxsize=1)
def scan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    class_index: InventoryClassIndex = {}
    for index_path in sorted(SCIENTIFIC_ROOT.rglob("*.py")):
        try:
            index_source = index_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            index_source = index_path.read_text(
                encoding="utf-8-sig",
                errors="ignore",
            )
        try:
            index_module = ast.parse(index_source)
        except SyntaxError:
            continue
        for index_class in [
            node for node in index_module.body if isinstance(node, ast.ClassDef)
        ]:
            class_index.setdefault(index_class.name, []).append(
                (index_path, index_module, index_class)
            )
    for path in sorted(SCIENTIFIC_ROOT.rglob("*.py")):
        rel = path.relative_to(PROJECT_ROOT).as_posix().replace("/", "\\")
        try:
            src = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            src = path.read_text(encoding="utf-8-sig", errors="ignore")

        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            rows.append(
                {
                    "file_path": rel,
                    "class_name": "<PARSE_ERROR>",
                    "base_classes": "",
                    "abstraction": "Unknown",
                    "forward_signature": "",
                    "execution_phase": "",
                    "dependencies": "",
                    "parameter_namespace": "",
                    "metadata_name": "",
                    "metadata_category": "",
                    "metadata_references": "",
                    "metadata_math": "",
                    "metadata_input_variables": "",
                    "metadata_output_variables": "",
                    "metadata_conservation_laws": "",
                    "metadata_parameters": "",
                    "metadata_spatial_scale": "",
                    "metadata_variable_kind": "",
                    "iphs_structural_hooks": "",
                    "process_id": rel[:-3].replace("\\", "."),
                    "canonical_role": "unknown",
                    "process_status": "parse_error",
                    "process_status_reason": "Source file could not be parsed.",
                    "implementation_status": IMPLEMENTATION_STATUS_UNKNOWN,
                    "implementation_status_reason": "source_parse_error",
                    "theory_document_path": "",
                    "theory_document_status": "missing",
                    "equivalence_certificate_path": "",
                    "equivalence_certificate_status": "missing",
                    "equivalence_certificate_audit": "",
                    "iphs_level": "",
                    "iphs_level_name": "",
                    "iphs_level_status": "missing",
                    "iphs_level_source": "missing",
                    "runtime_verified_level": "",
                    "runtime_evidence_status": "missing",
                    "runtime_evidence_issue_count": "",
                    "certificate_interface_consistency": "missing",
                    "certificate_theory_consistency": "missing",
                    "input_output_baseline_status": "missing",
                    "interface_baseline_status": "missing",
                    "parameter_management_status": "missing_metadata",
                    "state_advancement_status": "missing_certificate",
                    "state_advancement_specificity_status": "missing_certificate",
                    "source_state_advancement_status": "missing_source_evidence",
                    "runtime_replacement_status": "legacy_or_unmigrated",
                    "composition_contract_status": "not_declared",
                    "overflow_routing_status": "missing",
                    "duplicate_process_group": "",
                    "duplicate_process_status": "not_evaluated",
                    "completion_status": "pending",
                    "parse_error": f"SyntaxError:{e.msg}@{e.lineno}:{e.offset}",
                }
            )
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if _is_abstract_class(node):
                continue
            base_names = [_name_of(b) for b in node.bases]
            metadata = _extract_metadata(node, tree)
            if not _is_candidate_functor(node) and not metadata:
                continue
            base_abstraction = _infer_abstraction(base_names, metadata.get("category", ""))
            metadata_input_variables = "|".join(
                _effective_metadata_sequence(
                    node,
                    path,
                    tree,
                    class_index,
                    [
                        value
                        for value in metadata.get("input_variables", "").split("|")
                        if value
                    ],
                    "input_variables",
                )
            )
            metadata_output_variables = "|".join(
                _effective_metadata_sequence(
                    node,
                    path,
                    tree,
                    class_index,
                    [
                        value
                        for value in metadata.get("output_variables", "").split("|")
                        if value
                    ],
                    "output_variables",
                )
            )
            metadata_conservation_laws = metadata.get("conservation_laws", "")
            metadata_math = ensure_iphs_equivalent_expression(
                metadata.get("mathematical_expression", ""),
                category=metadata.get("category", ""),
                variable_kind=metadata.get("variable_kind", ""),
                input_variables=[
                    value for value in metadata_input_variables.split("|") if value
                ],
                output_variables=[
                    value for value in metadata_output_variables.split("|") if value
                ],
                conservation_laws=[
                    value for value in metadata_conservation_laws.split("|") if value
                ],
                functor_name=metadata.get("name", node.name),
            )
            row = {
                "file_path": rel,
                "class_name": node.name,
                "base_classes": ",".join(base_names),
                "abstraction": base_abstraction,
                "forward_signature": _extract_forward_signature(node),
                "execution_phase": metadata.get("execution_phase", ""),
                "dependencies": _declared_dependencies(rel, node.name),
                "parameter_namespace": _extract_parameter_namespace(node),
                "metadata_name": metadata.get("name", ""),
                "metadata_category": metadata.get("category", ""),
                "metadata_references": metadata.get("references", ""),
                "metadata_math": metadata_math,
                "metadata_input_variables": metadata_input_variables,
                "metadata_output_variables": metadata_output_variables,
                "metadata_conservation_laws": metadata_conservation_laws,
                "metadata_parameters": metadata.get("parameters_decl", ""),
                "metadata_spatial_scale": metadata.get("spatial_scale", ""),
                "metadata_variable_kind": metadata.get("variable_kind", ""),
                "iphs_structural_hooks": _structural_hooks(node),
                "parse_error": "",
            }
            process_id = _process_id(rel, node.name)
            theory_path, theory_status = _theory_path(rel, node.name)
            certificate_path = _certificate_path(process_id)
            certificate_status, certificate_audit = _certificate_status(certificate_path)
            implementation_status, implementation_status_reason = (
                _implementation_status(certificate_path)
            )
            (
                iphs_level,
                iphs_level_name,
                iphs_level_status,
                iphs_level_source,
                runtime_verified_level,
                runtime_evidence_status,
                runtime_evidence_issue_count,
            ) = _certificate_level_status(certificate_path)
            state_advancement_status = _state_advancement_status(certificate_path)
            state_advancement_specificity_status = _state_advancement_specificity_status(
                certificate_path
            )
            row["process_id"] = process_id
            row["canonical_role"] = _canonical_role(row)
            row["process_status"] = _process_status(row)
            row["process_status_reason"] = _support_only_reason(row)
            row["implementation_status"] = implementation_status
            row["implementation_status_reason"] = implementation_status_reason
            row["theory_document_path"] = theory_path
            row["theory_document_status"] = theory_status
            row["equivalence_certificate_path"] = certificate_path
            row["equivalence_certificate_status"] = certificate_status
            row["equivalence_certificate_audit"] = certificate_audit
            row["iphs_level"] = iphs_level
            row["iphs_level_name"] = iphs_level_name
            row["iphs_level_status"] = iphs_level_status
            row["iphs_level_source"] = iphs_level_source
            row["runtime_verified_level"] = runtime_verified_level
            row["runtime_evidence_status"] = runtime_evidence_status
            row["runtime_evidence_issue_count"] = runtime_evidence_issue_count
            (
                row["certificate_interface_consistency"],
                row["certificate_theory_consistency"],
            ) = _certificate_contract_consistency(
                certificate_path,
                metadata_inputs=metadata_input_variables,
                metadata_outputs=metadata_output_variables,
                theory_document_path=theory_path,
            )
            row["state_advancement_status"] = state_advancement_status
            row["state_advancement_specificity_status"] = state_advancement_specificity_status
            if implementation_status == IMPLEMENTATION_STATUS_NOT_IMPLEMENTED:
                row["state_advancement_status"] = "not_applicable_runtime_unbound"
                row["state_advancement_specificity_status"] = (
                    "not_applicable_runtime_unbound"
                )
            row["source_state_advancement_status"] = _source_state_advancement_status(
                row,
                _hierarchy_source_text(
                    node,
                    path,
                    tree,
                    class_index,
                ),
            )
            row["input_output_baseline_status"] = _io_status(row)
            row["interface_baseline_status"] = _interface_status(row)
            row["parameter_management_status"] = _parameter_management_status(
                certificate_path,
                row["metadata_parameters"],
            )
            row["runtime_replacement_status"] = _runtime_replacement_status(
                certificate_path,
                row,
            )
            row["composition_contract_status"] = _composition_contract_status(
                certificate_path,
                row,
            )
            row["overflow_routing_status"] = _overflow_routing_status(certificate_path)
            common_complete = (
                row["process_status"] == "process"
                and row["implementation_status"] == IMPLEMENTATION_STATUS_IMPLEMENTED
                and row["canonical_role"] != "unknown"
                and row["theory_document_status"] != "missing"
                and row["equivalence_certificate_status"] == "valid"
                and row["iphs_level_status"] == "valid"
                and row["certificate_interface_consistency"] == "matched"
                and row["certificate_theory_consistency"] == "matched"
                and row["state_advancement_status"] == "complete"
                and row["state_advancement_specificity_status"] == "specific"
                and row["source_state_advancement_status"] == "complete"
                and row["input_output_baseline_status"] == "captured"
                and row["interface_baseline_status"] in {"captured", "inherited"}
                and row["runtime_replacement_status"]
                in {"canonical_iphs_runtime_verified", "canonical_wrapper_declared"}
                and row["composition_contract_status"]
                in {
                    "ports_declared",
                    "composition_map_declared",
                    "metadata_declared",
                }
                and row["overflow_routing_status"] == "routing_map_declared"
            )
            if row["process_status"] == "support_only":
                row["completion_status"] = "not_applicable_support_only"
            elif row["implementation_status"] == IMPLEMENTATION_STATUS_NOT_IMPLEMENTED:
                row["completion_status"] = IMPLEMENTATION_STATUS_NOT_IMPLEMENTED
            elif not common_complete:
                row["completion_status"] = "pending"
            elif row["runtime_evidence_status"] == "verified" and int(
                row["runtime_verified_level"] or 0
            ) >= 4:
                row["completion_status"] = "strict_complete"
            else:
                row["completion_status"] = "classified_non_strict"
            rows.append(
                row
            )
    _mark_duplicate_groups(rows)
    return rows


def _mark_duplicate_groups(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("process_status") != "process":
            continue
        key = (
            str(row.get("canonical_role", "")),
            str(row.get("metadata_output_variables", "")),
        )
        if not key[1]:
            continue
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        duplicate = len(group_rows) > 1
        ids = "|".join(sorted(str(row["process_id"]) for row in group_rows))
        for row in group_rows:
            row["duplicate_process_group"] = ids if duplicate else ""
            row["duplicate_process_status"] = "candidate_duplicate" if duplicate else "unique"

    for row in rows:
        row.setdefault("duplicate_process_group", "")
        row.setdefault("duplicate_process_status", "not_evaluated")


def _build_publication_provenance_binding() -> dict[str, Any]:
    """Delegate source identity and command identity to the publication owner."""

    owner_path = (
        PROJECT_ROOT
        / "scripts"
        / "publication"
        / "02_validate_g1_evidence.py"
    )
    namespace = runpy.run_path(
        str(owner_path),
        run_name="_relic_publication_provenance_owner",
    )
    builder = namespace.get("build_g8_source_fingerprint")
    if not callable(builder):
        raise RuntimeError(
            "publication source fingerprint owner does not expose "
            "build_g8_source_fingerprint"
        )
    fingerprint = builder(PROJECT_ROOT)
    if not isinstance(fingerprint, dict):
        raise TypeError("publication source fingerprint must be a dictionary")

    command_spec_builder = namespace.get("get_g8_command_spec")
    if not callable(command_spec_builder):
        raise RuntimeError(
            "publication provenance owner does not expose get_g8_command_spec"
        )
    command_spec = command_spec_builder()
    if not isinstance(command_spec, list):
        raise TypeError("publication command specification must be a list")
    matching_commands = [
        item
        for item in command_spec
        if isinstance(item, dict)
        and item.get("command_id") == "collect_functor_inventory"
    ]
    if len(matching_commands) != 1:
        raise RuntimeError(
            "publication command specification must contain exactly one "
            "collect_functor_inventory command"
        )
    arguments = matching_commands[0].get("arguments")
    if not isinstance(arguments, list) or not all(
        isinstance(argument, str) for argument in arguments
    ):
        raise TypeError(
            "collect_functor_inventory publication arguments must be a string list"
        )

    provenance_owner = namespace.get("G8_PROVENANCE_OWNER")
    if not isinstance(provenance_owner, str) or not provenance_owner:
        raise RuntimeError(
            "publication provenance owner does not expose G8_PROVENANCE_OWNER"
        )
    return {
        "fingerprint": fingerprint,
        "command": list(arguments),
        "owner": provenance_owner,
    }


def build_phase8_audit_envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind the Phase 8 row audit to one live publication source identity."""

    binding = _build_publication_provenance_binding()
    fingerprint = binding["fingerprint"]
    process_rows = [row for row in rows if row.get("process_status") == "process"]
    support_rows = [
        row for row in rows if row.get("process_status") == "support_only"
    ]
    levels = {
        f"level_{level}": sum(
            1 for row in process_rows if str(row.get("iphs_level")) == str(level)
        )
        for level in range(0, 6)
    }
    return {
        "schema_version": "2.0.0",
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "producer": "test.audit.collect_functor_inventory",
            "command": binding["command"],
            "return_code": 0,
            "source_fingerprint_owner": binding["owner"],
            "source_fingerprint_schema_version": fingerprint.get(
                "schema_version"
            ),
            "source_fingerprint_sha256": fingerprint.get(
                "fingerprint_sha256"
            ),
            "git_head": fingerprint.get("git_head"),
            "source_file_count": fingerprint.get("file_count"),
            "dirty_tree_bound": True,
            "excluded_prefixes": fingerprint.get("excluded_prefixes", []),
            "excluded_files": fingerprint.get("excluded_files", []),
        },
        "summary": {
            "row_count": len(rows),
            "process_count": len(process_rows),
            "support_only_count": len(support_rows),
            "levels": levels,
            "strict_process_count": levels["level_4"] + levels["level_5"],
            "completion_status_counts": {
                status: sum(
                    1
                    for row in rows
                    if str(row.get("completion_status")) == status
                )
                for status in sorted(
                    {
                        str(row.get("completion_status"))
                        for row in rows
                        if row.get("completion_status")
                    }
                )
            },
        },
        "rows": rows,
    }


def main() -> None:
    rows = scan()

    json_path = OUTPUT_DIR / "scientific_functor_inventory.json"
    csv_path = OUTPUT_DIR / "scientific_functor_inventory.csv"
    phase8_json_path = OUTPUT_DIR / "phase8_scientific_process_audit.json"
    missing_certificates_path = OUTPUT_DIR / "phase8_missing_certificates.json"
    missing_theory_path = OUTPUT_DIR / "phase8_missing_theory_documents.json"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    missing_certificates = [
        row
        for row in rows
        if row["process_status"] == "process"
        and row["equivalence_certificate_status"] == "missing"
    ]
    missing_theory = [
        row
        for row in rows
        if row["process_status"] == "process"
        and row["theory_document_status"] == "missing"
    ]
    missing_certificates_path.write_text(
        json.dumps(missing_certificates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    missing_theory_path.write_text(
        json.dumps(missing_theory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = [
        "file_path",
        "class_name",
        "process_id",
        "base_classes",
        "abstraction",
        "canonical_role",
        "process_status",
        "process_status_reason",
        "implementation_status",
        "implementation_status_reason",
        "forward_signature",
        "execution_phase",
        "dependencies",
        "parameter_namespace",
        "metadata_name",
        "metadata_category",
        "metadata_references",
        "metadata_math",
        "metadata_input_variables",
        "metadata_output_variables",
        "metadata_conservation_laws",
        "metadata_parameters",
        "metadata_spatial_scale",
        "metadata_variable_kind",
        "iphs_structural_hooks",
        "theory_document_path",
        "theory_document_status",
        "equivalence_certificate_path",
        "equivalence_certificate_status",
        "equivalence_certificate_audit",
        "iphs_level",
        "iphs_level_name",
        "iphs_level_status",
        "iphs_level_source",
        "runtime_verified_level",
        "runtime_evidence_status",
        "runtime_evidence_issue_count",
        "certificate_interface_consistency",
        "certificate_theory_consistency",
        "state_advancement_status",
        "state_advancement_specificity_status",
        "source_state_advancement_status",
        "input_output_baseline_status",
        "interface_baseline_status",
        "parameter_management_status",
        "runtime_replacement_status",
        "composition_contract_status",
        "overflow_routing_status",
        "duplicate_process_group",
        "duplicate_process_status",
        "completion_status",
        "parse_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    phase8_json_path.write_text(
        json.dumps(
            build_phase8_audit_envelope(rows),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[OK] rows={len(rows)}")
    print(f"[OK] json={json_path}")
    print(f"[OK] csv={csv_path}")
    print(f"[OK] phase8_audit={phase8_json_path}")
    print(f"[OK] missing_certificates={len(missing_certificates)}")
    print(f"[OK] missing_theory_documents={len(missing_theory)}")


if __name__ == "__main__":
    main()
