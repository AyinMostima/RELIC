"""
Audit checks for removed default fallback patterns in key scientific modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_radiation_functor_no_default_passthrough_to_super_get_parameter_value() -> None:
    """
    References:
    - architecture_overview.md Section 19 fail-fast parameter access lock.

    Mathematical expression:
    - N/A (source-pattern gate).

    Parameter meanings:
    - Prevents fallback-by-default in Functor parameter retrieval.
    """
    content = _read("relic/scientific/atmosphere/radiation.py")
    assert "default=default" not in content


def test_stoichiometry_no_state_get_default_flux_fallbacks() -> None:
    """
    References:
    - architecture_overview.md forward-only and fail-fast diagnostics requirements.

    Mathematical expression:
    - N/A (source-pattern gate).

    Parameter meanings:
    - diagnostics.carbon_fluxes and diagnostics.nitrogen_fluxes must not use default dict fallback.
    """
    content = _read("relic/scientific/pedosphere/nutrient_cycling/stoichiometry.py")
    assert "state.get('diagnostics.carbon_fluxes', {})" not in content
    assert "state.get('diagnostics.nitrogen_fluxes', {})" not in content


def test_gas_diffusion_no_nested_state_get_default_pattern() -> None:
    """
    References:
    - architecture_overview.md fail-fast state dependency policy.

    Mathematical expression:
    - N/A (source-pattern gate).

    Parameter meanings:
    - Required state variables must not rely on nested default fallback generation.
    """
    content = _read("relic/scientific/pedosphere/transport/gas_diffusion.py")
    assert "state.get(actual_var, state.get(var_name))" not in content


def test_environmental_forcing_has_no_co2_runtime_responsibility() -> None:
    """Lock EnvironmentalForcing to its canonical VPD-only responsibility."""
    content = _read("relic/scientific/atmosphere/environmental_forcing.py")
    assert 'input_variables=["air_temperature", "relative_humidity"]' in content
    assert 'output_variables=["vapor_pressure_deficit"]' in content
    assert "atmospheric_co2_ppm" not in content
    assert "atmospheric_co2_concentration" not in content
    assert "co2_concentration" not in content


def test_scientific_runtime_state_get_calls_do_not_hide_missing_state() -> None:
    """
    References:
    - phase8_scientific_iphs_standardization_plan.md Section 2.
    - BIHSS_LaTeX_source.tex strict BIHSS audit gates.

    Mathematical expression:
    - x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h.

    Parameter meanings:
    - state, working_state, and merged_state are typed runtime state maps; their
      missing required entries must be explicit ports, diagnostics, or
      ParameterManager-backed defaults, not hidden dictionary fallbacks.
    """

    failures: list[str] = []
    scientific_root = PROJECT_ROOT / "relic" / "scientific"
    checked_names = {"state", "working_state", "merged_state"}
    for source_path in sorted(scientific_root.rglob("*.py")):
        if "__pycache__" in source_path.parts:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "get"):
                continue
            receiver = func.value
            if not isinstance(receiver, ast.Name) or receiver.id not in checked_names:
                continue
            if len(node.args) < 2:
                continue
            scope: list[str] = []
            current: ast.AST = node
            while current in parents:
                current = parents[current]
                if isinstance(current, (ast.ClassDef, ast.FunctionDef)):
                    scope.append(current.name)
            rel_path = source_path.relative_to(PROJECT_ROOT)
            scope_name = ".".join(reversed(scope))
            failures.append(f"{rel_path}:{node.lineno}:{scope_name}")

    assert not failures
