from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = ROOT / "relic" / "scientific" / "hydrosphere" / "evapotranspiration" / "priestley_taylor.py"


def test_priestley_taylor_docstring_does_not_claim_hardcoded_default_alpha():
    source = SOURCE_FILE.read_text(encoding="utf-8")
    assert "default 1.26" not in source
    assert "hydrosphere.evapotranspiration.alpha_pt" in source


def test_priestley_taylor_alpha_pt_is_loaded_from_parameter_system():
    source = SOURCE_FILE.read_text(encoding="utf-8")
    pattern = r"self\.alpha_pt\s*=\s*self\.get_parameter_value\(\s*\"alpha_pt\""
    assert re.search(pattern, source, re.DOTALL)

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "alpha_pt"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (int, float))
            ):
                raise AssertionError("self.alpha_pt must not be assigned a numeric literal")


def test_dynamic_alpha_separates_equilibrium_basis_and_empirical_prescription():
    source = SOURCE_FILE.read_text(encoding="utf-8")
    helper = source.split("def _compute_dynamic_alpha", 1)[1].split("\n    def ", 1)[0]
    assert "Priestley and Taylor (1972) provide the equilibrium evaporation basis" in helper
    assert "author-prescribed RELIC sensitivity formulation" in helper
    assert "provenance for this exact product has not been verified" in helper
    assert "SI Brochure, 9th edition, for the pressure conversion" in helper
