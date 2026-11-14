import ast
import hashlib
from importlib import resources
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
RELIC_ROOT = REPO_ROOT / "relic"
TEST_ROOT = REPO_ROOT / "test"


def _iter_python_files():
    return sorted(RELIC_ROOT.rglob("*.py"))


def _iter_test_python_files():
    return sorted(TEST_ROOT.rglob("*.py"))


def _parse_file(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(
            path.read_text(encoding="utf-8-sig", errors="ignore"),
            filename=str(path),
        )


def test_import_try_blocks_do_not_use_broad_exception_handlers_l1():
    violations = []
    for path in _iter_python_files():
        try:
            tree = _parse_file(path)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue

            has_direct_import = any(
                isinstance(stmt, (ast.Import, ast.ImportFrom))
                for stmt in node.body
            )
            if not has_direct_import:
                continue

            for handler in node.handlers:
                if handler.type is None:
                    violations.append(f"{path}:{node.lineno} bare except")
                    break

                if isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
                    violations.append(f"{path}:{node.lineno} except Exception")
                    break

                if isinstance(handler.type, ast.Tuple):
                    names = [
                        elt.id
                        for elt in handler.type.elts
                        if isinstance(elt, ast.Name)
                    ]
                    if "Exception" in names:
                        violations.append(f"{path}:{node.lineno} except (..., Exception, ...)")
                        break

    assert not violations, "Found broad handlers on import try-blocks:\n" + "\n".join(violations[:30])


def test_get_global_manager_import_path_is_not_mixed_l1():
    pattern_public = "from relic.parameters import get_global_manager"
    pattern_internal = "from relic.parameters.manager import get_global_manager"

    violations = []
    for path in _iter_python_files():
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        if pattern_public in text and pattern_internal in text:
            violations.append(str(path))

    assert not violations, "Mixed get_global_manager import paths in files:\n" + "\n".join(violations)


def test_get_global_manager_import_uses_public_api_outside_parameters_package_l1():
    violations = []
    parameters_root = RELIC_ROOT / "parameters"

    for path in _iter_python_files():
        if path.is_relative_to(parameters_root):
            continue

        try:
            tree = _parse_file(path)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "relic.parameters.manager":
                continue
            if any(alias.name == "get_global_manager" for alias in node.names):
                violations.append(f"{path}:{node.lineno}")

    for path in _iter_test_python_files():
        try:
            tree = _parse_file(path)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "relic.parameters.manager":
                continue
            if any(alias.name == "get_global_manager" for alias in node.names):
                violations.append(f"{path}:{node.lineno}")

    assert not violations, (
        "Use public API import path for get_global_manager "
        "(from relic.parameters import get_global_manager). Violations:\n"
        + "\n".join(violations)
    )


def test_no_direct_parameters_manager_imports_in_relic_sources_l1():
    """All relic sources must use public parameters API, not manager internals."""
    offenders = []
    patterns = (
        "from relic.parameters.manager import",
        "from ..parameters.manager import",
        "from ...parameters.manager import",
    )

    for path in _iter_python_files():
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        if any(pattern in text for pattern in patterns):
            offenders.append(str(path))

    assert not offenders, (
        "Direct imports from relic.parameters.manager are forbidden in relic sources:\n"
        + "\n".join(offenders)
    )


def test_require_parameter_is_centralized_in_functor_base_l1():
    target_files = [
        RELIC_ROOT / "scientific" / "biosphere" / "surrogates" / "simple_photosynthesis.py",
        RELIC_ROOT / "scientific" / "biosphere" / "growth" / "biomass_allocation.py",
        RELIC_ROOT / "scientific" / "biosphere" / "phenology" / "phenology_functor.py",
        RELIC_ROOT / "scientific" / "biosphere" / "stress" / "ozone_stress.py",
        RELIC_ROOT / "scientific" / "anthroposphere" / "management" / "coordinator.py",
    ]

    offenders = []
    for file_path in target_files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        if "def _require_parameter(" in text:
            offenders.append(str(file_path))

    assert not offenders, "_require_parameter must be centralized in FunctorBase:\n" + "\n".join(offenders)


def test_parameter_metadata_access_uses_canonical_manager_method_l1():
    """Reject calls to removed public ParameterManager metadata aliases."""
    violations = []
    for path in (*_iter_python_files(), *_iter_test_python_files()):
        try:
            tree = _parse_file(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"get_spec", "get_parameter_spec"}:
                violations.append(f"{path}:{node.lineno}:{node.func.attr}")

    assert not violations, (
        "Use manager.get_parameter(...) for ParameterSpec metadata. Violations:\n"
        + "\n".join(violations)
    )


def test_validation_types_use_framework_or_domain_specific_names_l1():
    """Prevent new generic validation type definitions outside the SSOT module."""

    allowed_generic = {
        RELIC_ROOT / "validation" / "framework" / "models.py",
    }
    generic_names = {
        "ValidationLevel",
        "ValidationResult",
        "ValidationIssue",
        "ValidationContext",
    }
    violations = []

    for path in _iter_python_files():
        try:
            tree = _parse_file(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in generic_names:
                if path not in allowed_generic:
                    violations.append(f"{path}:{node.lineno}:{node.name}")

    assert not violations, (
        "Use relic.validation.framework.models or a domain-specific name for "
        "validation types. Violations:\n" + "\n".join(violations)
    )


def test_canonical_management_entrypoints_are_unique_l1():
    """Protect core lifecycle managers, registries, and compiler facade from duplication."""

    expected = {
        "UnifiedParameterManager": RELIC_ROOT / "parameters" / "manager.py",
        "StateManager": RELIC_ROOT / "core" / "state" / "manager.py",
        "ValidationRegistry": RELIC_ROOT / "validation" / "framework" / "registry.py",
        "NodeRegistry": RELIC_ROOT / "orchestration" / "node_registry.py",
        "SSOTCatalog": RELIC_ROOT / "orchestration" / "ssot_catalog.py",
        "ContractRegistry": RELIC_ROOT / "contracts" / "contract_loader.py",
        "RELICSimulator": RELIC_ROOT / "core" / "simulator" / "main.py",
        "RELICModelCompiler": RELIC_ROOT / "orchestration" / "compiler.py",
    }
    class_locations = {name: [] for name in expected}
    generic_stream_locations = []

    for path in _iter_python_files():
        try:
            tree = _parse_file(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name in class_locations:
                class_locations[node.name].append(path)
            if node.name == "StreamManager":
                generic_stream_locations.append(f"{path}:{node.lineno}")

    violations = []
    for name, expected_path in expected.items():
        locations = class_locations[name]
        if locations != [expected_path]:
            rendered = ", ".join(str(path) for path in locations) or "<missing>"
            violations.append(f"{name}: expected {expected_path}, found {rendered}")

    assert not generic_stream_locations, (
        "Generic StreamManager class definitions are ambiguous; use a domain-specific "
        "class name with a compatibility alias instead:\n"
        + "\n".join(generic_stream_locations)
    )
    assert not violations, "Duplicate or missing management entrypoints:\n" + "\n".join(violations)


def test_runtime_resources_are_declared_as_package_data_l1():
    """Wheel builds must retain canonical runtime schemas and assets."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    patterns = set(project["tool"]["setuptools"]["package-data"]["relic"])
    required_patterns = {
        "**/*.yaml",
        "**/*.yml",
        "**/*.json",
        "extensions/templates/**/*.template",
        "interfaces/web/static/**/*.css",
        "interfaces/web/static/**/*.js",
        "interfaces/web/templates/**/*.html",
        "coupling/plugin/*.h",
    }

    assert required_patterns <= patterns


def test_official_cf_v94_resource_is_exact_and_narrowly_packaged_l1():
    """The bundled official XML is the only CF standard-name artifact."""
    resource_name = "cf_standard_name_table.xml"
    resource_path = RELIC_ROOT / "data" / "standards" / resource_name

    assert resource_path.is_file()
    payload = resources.files("relic.data.standards").joinpath(
        resource_name
    ).read_bytes()
    assert len(payload) == 4_554_075
    assert hashlib.sha256(payload).hexdigest() == (
        "39a11ab9ccc327756da1bada1f268eebabdb7cee0cc908464962197bb892165c"
    )

    root = ET.fromstring(payload)
    entries = root.findall("entry")
    aliases = root.findall("alias")
    entry_ids = [entry.get("id") for entry in entries]
    alias_ids = [alias.get("id") for alias in aliases]
    assert root.tag == "standard_name_table"
    assert root.findtext("version_number") == "94"
    assert root.findtext("last_modified") == "2026-06-09T17:23:36Z"
    assert len(entries) == 5_071
    assert len(aliases) == 599
    assert len(set(entry_ids)) == len(entry_ids)
    assert len(set(alias_ids)) == len(alias_ids)
    assert set(entry_ids) & set(alias_ids) == {
        "convective_precipitation_rate",
        "ocean_volume",
    }
    assert sum(
        not (entry.findtext("canonical_units") or "").strip()
        for entry in entries
    ) == 17

    multi_target_aliases = {
        alias.get("id"): tuple(
            target.text for target in alias.findall("entry_id")
        )
        for alias in aliases
        if len(alias.findall("entry_id")) > 1
    }
    assert multi_target_aliases == {
        "surface_carbon_dioxide_mole_flux": (
            "surface_downward_mole_flux_of_carbon_dioxide",
            "surface_upward_mole_flux_of_carbon_dioxide",
        )
    }

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    patterns = set(project["tool"]["setuptools"]["package-data"]["relic"])
    assert "data/standards/cf_standard_name_table.xml" in patterns
    assert "**/*.xml" not in patterns
    assert list(resource_path.parent.glob("cf_standard_name_table*.xml")) == [
        resource_path
    ]
    assert not list(resource_path.parent.glob("cf_standard_name_table*.json"))
