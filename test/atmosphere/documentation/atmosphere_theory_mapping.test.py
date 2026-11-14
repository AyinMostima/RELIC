"""
L1 documentation mapping checks for atmosphere modules.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_target_atmosphere_modules_have_theory_documents() -> None:
    """Ensure critical atmosphere modules have dedicated theory docs."""
    mapping = {
        "relic/scientific/atmosphere/environmental_forcing.py": "docs/source/theory/atmosphere_environmental_forcing.md",
        "relic/scientific/atmosphere/transport/dry_deposition_aggregator.py": "docs/source/theory/atmosphere_dry_deposition_aggregator.md",
        "relic/scientific/atmosphere/transport/eddy_diffusion_blm.py": "docs/source/theory/atmosphere_eddy_diffusion_blm.md",
        "relic/scientific/atmosphere/transport/AtmosphericAdvectionDiffusionFunctor.py": "docs/source/theory/atmosphere_advection_diffusion.md",
    }

    for module_path, doc_path in mapping.items():
        module_file = REPO_ROOT / module_path
        theory_file = REPO_ROOT / doc_path
        assert module_file.exists(), f"Missing module file: {module_path}"
        assert theory_file.exists(), f"Missing theory document: {doc_path}"
