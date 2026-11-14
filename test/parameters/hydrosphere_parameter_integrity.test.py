from __future__ import annotations

import ast
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple

import yaml

from relic.parameters import get_global_manager
from relic.parameters.ops.key_normalization import normalize_parameter_key

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "relic" / "scientific" / "hydrosphere"
PARAM_ROOT_HYDRO = REPO_ROOT / "relic" / "parameters" / "database" / "relic" / "hydrosphere"
PARAM_ROOT_ALL = REPO_ROOT / "relic" / "parameters" / "database" / "relic"

IGNORED_ORPHAN_PREFIXES = (
    "hydrosphere.state_defaults.",
)


def _gather_yaml_keys(root: Path) -> Tuple[Set[str], Set[str], Dict[str, str], Dict[str, int]]:
    """Collect parameter keys from YAML database under the given root."""
    full_keys: Set[str] = set()
    raw_keys: Set[str] = set()
    full_to_raw: Dict[str, str] = {}
    raw_counts: Dict[str, int] = defaultdict(int)
    for path in root.rglob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not data or "parameters" not in data:
            continue
        namespace = (data.get("namespace") or "").strip()
        for key in data["parameters"].keys():
            full_key = key
            if namespace and not key.startswith(namespace + "."):
                full_key = f"{namespace}.{key}"
            full_norm = normalize_parameter_key(full_key)
            raw_norm = normalize_parameter_key(key)
            full_keys.add(full_norm)
            raw_keys.add(raw_norm)
            full_to_raw[full_norm] = raw_norm
            raw_counts[raw_norm] += 1
    return full_keys, raw_keys, full_to_raw, raw_counts


class _ParameterCallVisitor(ast.NodeVisitor):
    """AST visitor for extracting parameter usage with literal keys."""

    def __init__(self) -> None:
        self.keys: Set[str] = set()
        self.units_map: Dict[str, Set[str]] = {}
        self._allowed_full_prefixes = (
            "hydrosphere.",
            "relic.",
            "validation.",
            "psychrometrics.",
            "biosphere.",
            "atmosphere.",
            "pedosphere.",
            "anthroposphere.",
            "coupling.",
        )
        self._helper_namespace_overrides = {
            "_numerical_param": "hydrosphere.numerical",
            "_sm_param": "hydrosphere.soil_moisture",
            "_get_param_flag": "hydrosphere.transport.vertical_percolation",
            "_get_boundary_default": "hydrosphere.soil_moisture",
            "_require_manager_flag": "hydrosphere.transport.vertical_percolation",
            "_load_psychrometric_param": None,
        }
        self._scopes: list[Dict[str, str]] = [{}]
        self._namespace_stack: list[str | None] = [None]

    def _resolve_name(self, name: str) -> str | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scopes.append({})
        self._namespace_stack.append(self._namespace_stack[-1])
        self.generic_visit(node)
        self._scopes.pop()
        self._namespace_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._scopes[-1][target.id] = node.value.value
                    if target.id.endswith("_namespace") or target.id == "namespace":
                        self._namespace_stack[-1] = node.value.value
        if isinstance(node.value, ast.Dict):
            self._extract_dict_param_specs(node.value)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Record literal parameter declarations returned by required_parameters."""
        value = node.value
        if isinstance(value, (ast.Tuple, ast.List)):
            for element in value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    literal = element.value
                    if literal.startswith(self._allowed_full_prefixes):
                        self._record(literal, None)
        self.generic_visit(node)

    def _looks_like_key(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_.]+", value))

    def _extract_dict_param_specs(self, node: ast.Dict) -> None:
        namespace_hint = self._namespace_stack[-1]
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                key_value = value.value
                if not self._looks_like_key(key_value):
                    continue
                if namespace_hint and not key_value.startswith(self._allowed_full_prefixes):
                    self._record(f"{namespace_hint}.{key_value}", None)
                elif key_value.startswith(self._allowed_full_prefixes):
                    self._record(key_value, None)
                continue
            if not isinstance(value, (ast.Tuple, ast.List)):
                continue
            if not value.elts:
                continue
            key_node = value.elts[0]
            units_node = value.elts[1] if len(value.elts) > 1 else None
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                continue
            units = None
            if isinstance(units_node, ast.Constant) and isinstance(units_node.value, str):
                units = units_node.value
            key_value = key_node.value
            if namespace_hint and not key_value.startswith(self._allowed_full_prefixes):
                self._record(f"{namespace_hint}.{key_value}", units)
            else:
                self._record(key_value, units)

    def _record(self, key: str, units: str | None) -> None:
        if key.startswith("hydrosphere.") and key.count(".") == 1:
            return
        canonical = normalize_parameter_key(key)
        self.keys.add(canonical)
        if units:
            self.units_map.setdefault(canonical, set()).add(units)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = None
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        if func_name in {"get_parameter_value", "_require_parameter"}:
            key = None
            namespace = None
            units = None
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                key = node.args[0].value
            if key is None and node.args and isinstance(node.args[0], ast.Name):
                key = self._resolve_name(node.args[0].id)
            for kw in node.keywords:
                if kw.arg == "namespace" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    namespace = kw.value.value
                if kw.arg == "units" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    units = kw.value.value
            if namespace is None:
                for kw in node.keywords:
                    if kw.arg == "namespace" and isinstance(kw.value, ast.Name):
                        namespace = self._resolve_name(kw.value.id)
            if key and namespace:
                full_key = key
                if not key.startswith(namespace + "."):
                    full_key = f"{namespace}.{key}"
                self._record(full_key, units)
            elif key and key.startswith(self._allowed_full_prefixes):
                self._record(key, units)

        if func_name in {
            "_get_param",
            "_get_param_raw",
            "_get_param_sequence",
            "_require",
            "_require_manager_param",
            "_require_manager_param_raw",
            "_require_manager_param_scalar",
            "_require_manager_param_sequence",
        }:
            namespace = None
            key = None
            units = None
            if node.args:
                first = node.args[0]
                second = node.args[1] if len(node.args) > 1 else None
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    key = first.value
                elif isinstance(first, ast.Name):
                    key = self._resolve_name(first.id)
                if second is not None:
                    if isinstance(first, ast.Constant) and isinstance(first.value, str) and isinstance(
                        second, ast.Constant
                    ) and isinstance(second.value, str):
                        if first.value.startswith(self._allowed_full_prefixes):
                            namespace = first.value
                            key = second.value
                        elif second.value.startswith(self._allowed_full_prefixes):
                            namespace = second.value
                            key = first.value
                        else:
                            key = first.value
                    if (
                        isinstance(first, ast.Name)
                        and isinstance(second, ast.Constant)
                        and isinstance(second.value, str)
                    ):
                        resolved = self._resolve_name(first.id)
                        if resolved and resolved.startswith(self._allowed_full_prefixes):
                            namespace = resolved
                            key = second.value
            for kw in node.keywords:
                if kw.arg == "units" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    units = kw.value.value
                if kw.arg == "namespace" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    namespace = kw.value.value
                if kw.arg == "namespace" and isinstance(kw.value, ast.Name):
                    namespace = self._resolve_name(kw.value.id)
            if key and namespace:
                full_key = key
                if not key.startswith(namespace + "."):
                    full_key = f"{namespace}.{key}"
                self._record(full_key, units)
            elif key and key.startswith(self._allowed_full_prefixes):
                if len(node.args) > 1 and not isinstance(node.args[1], ast.Constant):
                    self.generic_visit(node)
                    return
                self._record(key, units)

        if func_name == "get":
            key = None
            namespace = None
            units = None
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                key = node.args[0].value
            for kw in node.keywords:
                if kw.arg == "namespace" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    namespace = kw.value.value
                if kw.arg == "units" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    units = kw.value.value
                if kw.arg == "namespace" and isinstance(kw.value, ast.Name):
                    namespace = self._resolve_name(kw.value.id)
            if namespace and key:
                full_key = key
                if not key.startswith(namespace + "."):
                    full_key = f"{namespace}.{key}"
                self._record(full_key, units)
            elif key and key.startswith(self._allowed_full_prefixes):
                self._record(key, units)

        if func_name in self._helper_namespace_overrides:
            key = None
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                key = node.args[0].value
            if key is None and node.args and isinstance(node.args[0], ast.Name):
                key = self._resolve_name(node.args[0].id)
            if key:
                namespace = self._helper_namespace_overrides[func_name]
                if namespace:
                    full_key = key if key.startswith(namespace + ".") else f"{namespace}.{key}"
                    self._record(full_key, None)
                else:
                    self._record(key, None)

        self.generic_visit(node)


def _gather_code_keys() -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Collect parameter keys referenced by hydrosphere code."""
    keys: Set[str] = set()
    units_map: Dict[str, Set[str]] = {}
    for path in CODE_ROOT.rglob("*.py"):
        try:
            # Use utf-8-sig so files with BOM are parsed instead of being silently skipped.
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        visitor = _ParameterCallVisitor()
        visitor.visit(tree)
        keys |= visitor.keys
        for key, units in visitor.units_map.items():
            units_map.setdefault(key, set()).update(units)
    return keys, units_map


def test_hydrosphere_parameters_defined() -> None:
    """All literal parameter keys referenced by hydrosphere code exist in YAML."""
    yaml_full, yaml_raw, _, _ = _gather_yaml_keys(PARAM_ROOT_ALL)
    yaml_all = yaml_full | yaml_raw
    code_keys, _ = _gather_code_keys()
    missing = sorted(key for key in code_keys if key not in yaml_all)
    assert not missing, "Missing hydrosphere parameters in YAML: " + ", ".join(missing)


def test_hydrosphere_parameters_units_consistent() -> None:
    """Literal units in code match YAML definitions via ParameterManager validation."""
    code_keys, units_map = _gather_code_keys()
    manager = get_global_manager()

    errors = []
    for key in sorted(code_keys):
        units_set = units_map.get(key, set())
        for units in sorted(units_set):
            try:
                manager.get(key, units=units)
            except Exception as exc:
                errors.append(f"{key} units {units}: {exc}")
    assert not errors, "Units mismatch for hydrosphere parameters: " + "; ".join(errors)


def test_groundwater_runtime_state_has_no_parameter_defaults() -> None:
    """Keep canonical groundwater state and boundary ports StateManager-owned."""
    yaml_full, _, _, _ = _gather_yaml_keys(PARAM_ROOT_HYDRO)
    forbidden = {
        "hydrosphere.state_defaults.soil_drainage",
        "hydrosphere.state_defaults.lateral_inflow",
        "hydrosphere.state_defaults.groundwater_pumping",
        "hydrosphere.state_defaults.groundwater_storage",
        "hydrosphere.state_defaults.water_table_depth",
    }
    retained = sorted(forbidden & yaml_full)
    assert not retained, "Stale groundwater state defaults in YAML: " + ", ".join(retained)


def test_richards_flux_failure_default_is_retired() -> None:
    """No flux default replaces a failed Richards solve (fail-fast, no fallback value).

    RichardsSoilMoisture raises on solver failure and no hydrosphere code reads
    hydrosphere.state_defaults.vertical_flux_default, so neither the YAML database
    nor the hydrosphere parameter inventory may list it.
    """
    from relic.parameters.database.relic import hydrosphere as hydrosphere_database

    retired = "hydrosphere.state_defaults.vertical_flux_default"
    yaml_full, _, _, _ = _gather_yaml_keys(PARAM_ROOT_HYDRO)
    assert retired not in yaml_full
    listed = {
        key
        for keys in hydrosphere_database.KEY_PARAMETERS.values()
        for key in keys
    }
    assert retired not in listed


UNREAD_STATE_DEFAULTS = (
    "hydrosphere.state_defaults.pressure_head_surface_default",
    "hydrosphere.state_defaults.pressure_head_subsurface_default",
    "hydrosphere.state_defaults.surface_water_flux",
    "hydrosphere.state_defaults.drainage_input",
    "hydrosphere.state_defaults.elevation",
    "hydrosphere.state_defaults.relative_humidity",
    "hydrosphere.state_defaults.evapotranspiration_flux",
    # Former SnowAlbedo and SnowPack input fallbacks: a missing input now raises.
    "hydrosphere.state_defaults.precipitation_flux",
    "hydrosphere.state_defaults.snowfall_flux",
    "hydrosphere.state_defaults.snow_water_equivalent",
    "hydrosphere.state_defaults.sublimation_flux",
    "hydrosphere.state_defaults.snow_depth",
    "hydrosphere.state_defaults.snow_nondimensional_age",
    "hydrosphere.state_defaults.snow_surface_age",
    "hydrosphere.state_defaults.solar_zenith_angle",
)


def test_unread_state_defaults_are_retired() -> None:
    """No hydrosphere Functor reads these defaults, not by literal key and not through
    namespace hydrosphere.state_defaults: SnowAlbedo and SnowPack raise on a missing
    input instead of substituting a default. The last key of the namespace, the
    reference temperature of the static Priestley-Taylor Hodge contract, is now
    hydrosphere.evapotranspiration.hodge_reference_air_temperature, so the namespace
    and its file are gone. Neither the YAML database nor the parameter inventory
    declares the retired keys."""
    from relic.parameters.database.relic import hydrosphere as hydrosphere_database

    yaml_full, _, _, _ = _gather_yaml_keys(PARAM_ROOT_HYDRO)
    listed = {
        key
        for keys in hydrosphere_database.KEY_PARAMETERS.values()
        for key in keys
    }
    assert [key for key in UNREAD_STATE_DEFAULTS if key in yaml_full] == []
    assert [key for key in UNREAD_STATE_DEFAULTS if key in listed] == []
    assert [key for key in yaml_full if key.startswith("hydrosphere.state_defaults.")] == []
    assert "state_defaults" not in hydrosphere_database.PARAMETER_FILES
    assert "state_defaults" not in hydrosphere_database.PARAMETER_NAMESPACES
    assert "hydrosphere.evapotranspiration.hodge_reference_air_temperature" in yaml_full


def test_hydrosphere_no_orphan_parameters() -> None:
    """Hydrosphere YAML parameters should be referenced by code (with explicit ignores)."""
    yaml_full, _, full_to_raw, raw_counts = _gather_yaml_keys(PARAM_ROOT_HYDRO)
    code_keys, _ = _gather_code_keys()

    orphans = sorted(
        key for key in yaml_full
        if (
            key not in code_keys
            and not key.startswith(IGNORED_ORPHAN_PREFIXES)
            and not (
                raw_counts.get(full_to_raw.get(key, ""), 0) == 1
                and full_to_raw.get(key, "") in code_keys
            )
        )
    )
    assert not orphans, "Orphan hydrosphere parameters in YAML: " + ", ".join(orphans)
