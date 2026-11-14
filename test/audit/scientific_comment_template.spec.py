"""Audit gate for scientific forward-comment template completeness."""

from __future__ import annotations

import ast
from pathlib import Path

from test.audit.list_forward_template_missing import (
    SIX_ELEMENT_MARKERS,
    forward_template_present,
    function_template_present,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = REPO_ROOT / "relic" / "scientific"
ARCHITECTURE_OVERVIEW = REPO_ROOT / "docs" / "source" / "architecture_overview.md"
O3_SIX_ELEMENT_METHODS = {
    Path("anthroposphere/management/tillage.py"): {
        "_get_tillage_intensity_factor",
        "_get_tillage_depth_factor",
        "_calculate_time_decay_factor",
        "_calculate_porosity_modifier",
        "_calculate_carbon_flush",
        "_check_tillage_schedule",
        "forward",
        "_check_conservation",
        "check_conservation_laws",
        "_compute_rate_core",
    },
    Path("anthroposphere/management/harvest.py"): {
        "build_hodge_contract",
        "reversible_structure",
        "dissipative_structure",
        "_get_crop_characteristics",
        "_check_harvest_schedule",
        "forward",
        "check_conservation_laws",
        "_check_conservation",
        "_canonical_iphs_harvest_partition",
        "_storage_potential",
    },
    Path("anthroposphere/management/grazing.py"): {
        "build_hodge_contract",
        "reversible_structure",
        "dissipative_structure",
        "_get_livestock_characteristics",
        "_calculate_daily_intake",
        "_calculate_excreta_return",
        "_calculate_trampling_damage",
        "_calculate_grazing_pressure",
        "_check_grazing_schedule",
        "_assemble_grazing_reaction_outputs",
        "forward",
        "check_conservation_laws",
    },
    Path("anthroposphere/management/fertilization.py"): {
        "_event_duration",
        "_p2o5_to_elemental_p_fraction",
        "get_fertilizer_properties",
        "_event_overlap_seconds",
        "_event_element_masses",
        "reversible_structure",
        "dissipative_structure",
        "forward",
        "check_conservation_laws",
    },
    Path("anthroposphere/management/irrigation.py"): {
        "_kg_m2_to_mm_factor",
        "_standard_moisture_response",
        "forward",
        "check_conservation_laws",
    },
    Path("anthroposphere/inventory.py"): {
        "forward",
    },
    Path("anthroposphere/interface_mapping.py"): {
        "forward",
    },
    Path("anthroposphere/emissions/coupling_to_atmosphere.py"): {
        "forward",
    },
    Path("isotopes/functor.py"): {
        "forward",
    },
}


def test_forward_template_detector_supports_multiline_signatures() -> None:
    lines = [
        "def forward(",
        "    self,",
        "    state,",
        "):",
        '    """A long process docstring.',
        "    More scientific context.",
        '    """',
        "    # References: Example (2026).",
        "    # Mathematical expression: y = f(x).",
        "    # Parameter meanings: x is the input and y is the output.",
        "    return state",
    ]

    assert forward_template_present(lines, 0)


def test_six_element_detector_supports_multiline_signatures() -> None:
    lines = [
        "def forward(",
        "    self,",
        "    state,",
        "):",
        '    """A scientific calculation."""',
        "    # References: Example (2026).",
        "    # Mathematical expression: y = f(x).",
        "    # Parameter meanings: x is input and y is output.",
        "    # Units: x [1]; y [1].",
        "    # Domain/constraints: x and y are finite.",
        "    # Sign convention: positive y is an outward flux.",
        "    return state",
    ]

    assert function_template_present(lines, 0, SIX_ELEMENT_MARKERS)


def test_architecture_ssot_declares_only_the_six_element_comment_contract() -> None:
    """Architecture documentation must match the executable O3 audit."""
    text = ARCHITECTURE_OVERVIEW.read_text(encoding="utf-8")
    section = text.split(
        "### 7.1 Scientific Process Code Comment Template (Mandatory)",
        maxsplit=1,
    )[1].split("### 7.2 Theory Documentation First", maxsplit=1)[0]

    for marker in SIX_ELEMENT_MARKERS:
        assert marker in section.lower()
    assert "References + Formulas + Parameter meanings" not in text


def test_scientific_forward_functions_have_comment_template() -> None:
    violations: list[str] = []
    for path in SCIENTIFIC_ROOT.rglob("*.py"):
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            if not line.lstrip().startswith("def forward("):
                continue
            if not forward_template_present(lines, i):
                violations.append(f"{path.as_posix()}:{i + 1}")

    assert not violations, (
        "Missing References/Mathematical expression/Parameter meanings template "
        "in scientific forward functions:\n" + "\n".join(sorted(violations))
    )


def test_o3_target_calculation_methods_have_six_element_comments() -> None:
    violations: list[str] = []
    for relative_path, method_names in O3_SIX_ELEMENT_METHODS.items():
        path = SCIENTIFIC_ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tree = ast.parse(text, filename=str(path))
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in method_names
        }
        missing_methods = sorted(method_names - methods.keys())
        for method_name in missing_methods:
            violations.append(f"{relative_path.as_posix()}::{method_name}:missing")
        for method_name, node in methods.items():
            if not function_template_present(
                lines,
                node.lineno - 1,
                SIX_ELEMENT_MARKERS,
            ):
                violations.append(
                    f"{relative_path.as_posix()}::{method_name}:{node.lineno}"
                )

    assert not violations, (
        "O3 calculation methods missing six-element scientific comments:\n"
        + "\n".join(violations)
    )
