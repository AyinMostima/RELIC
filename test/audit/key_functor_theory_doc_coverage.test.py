"""
Coverage checks for dedicated theory docs required by key pending-issue modules.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THEORY_DIR = PROJECT_ROOT / "docs" / "source" / "theory"


def test_dedicated_theory_docs_exist_for_key_modules() -> None:
    """
    References:
    - docs/source/architecture_overview.md documentation-first mandate.

    Mathematical expression:
    - N/A (documentation contract gate).

    Parameter meanings:
    - Each key scientific process must map to at least one dedicated theory document.
    """
    required_docs = [
        "root_nutrient_uptake.md",
        "stoichiometry_constraint.md",
        "multi_isotope_pools.md",
    ]
    for doc_name in required_docs:
        assert (THEORY_DIR / doc_name).exists(), f"Missing dedicated theory doc: {doc_name}"


def test_dedicated_theory_docs_contain_required_formula_markers() -> None:
    """
    References:
    - Root uptake, stoichiometry, and isotope conservation process documentation.

    Mathematical expression:
    - Verifies presence of key equation markers in each dedicated theory document.

    Parameter meanings:
    - Marker tokens correspond to process-defining variables and tolerances.
    """
    checks = {
        "root_nutrient_uptake.md": [
            "U_i(z) = Vmax_i * C_i(z) / (Km_i + C_i(z)) * f_T * f_W * W_root(z)",
            "Delta pool_i = - U_i_area * M_i * dt",
        ],
        "stoichiometry_constraint.md": [
            "N_target = C_flux / CN_target",
            "e_C_mass = abs((C_inputs - C_outputs) - sum(Delta C_pools))",
        ],
        "multi_isotope_pools.md": [
            "epsilon_mass = abs(M_new - M_old) / (abs(M_old) + epsilon)",
            "delta = (R_sample / R_standard - 1.0) * 1000.0",
        ],
    }
    for doc_name, markers in checks.items():
        content = (THEORY_DIR / doc_name).read_text(encoding="utf-8")
        for marker in markers:
            assert marker in content, f"Missing marker in {doc_name}: {marker}"


def test_key_modules_reference_dedicated_theory_docs() -> None:
    """
    References:
    - Architecture policy requiring code-document alignment for scientific modules.

    Mathematical expression:
    - N/A (metadata/reference contract gate).

    Parameter meanings:
    - Source modules must include explicit references to dedicated theory docs.
    """
    source_checks = {
        "relic/scientific/pedosphere/nutrient_cycling/root_nutrient_uptake.py": "docs/source/theory/root_nutrient_uptake.md",
        "relic/scientific/pedosphere/nutrient_cycling/stoichiometry.py": "docs/source/theory/stoichiometry_constraint.md",
    }
    for source_path, marker in source_checks.items():
        content = (PROJECT_ROOT / source_path).read_text(encoding="utf-8")
        assert marker in content, f"Missing theory doc reference marker in {source_path}: {marker}"

    deleted_isotope_source = PROJECT_ROOT / (
        "relic/scientific/isotopes/tracking/multi_isotope_pools.py"
    )
    historical_isotope_doc = PROJECT_ROOT / (
        "docs/source/theory/multi_isotope_pools.md"
    )
    assert not deleted_isotope_source.exists()
    assert historical_isotope_doc.read_text(encoding="utf-8").startswith(
        "# [H]"
    )
