"""
Audit gate for physical constants SSOT category placement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from relic.utils.physical_constants import get_physical_constant


_PARAM_ROOT = Path("relic/parameters/database/relic")
_SSOT_FILE = Path("relic/parameters/database/relic/physical/constants.yaml")
_PATTERN = re.compile(r"^\s*category:\s*physical_constant\s*$", re.MULTILINE)


def test_physical_constant_category_only_exists_in_ssot_file() -> None:
    """
    References:
    - docs/source/architecture_overview.md Physical Constants SSOT policy.
    - relic/parameters/database/relic/physical/constants.yaml.

    Mathematical expression:
    - N/A (metadata placement gate).

    Parameter meanings:
    - category=physical_constant marks canonical physical constants only.
    """
    violations: list[str] = []
    for yaml_file in _PARAM_ROOT.rglob("*.yaml"):
        content = yaml_file.read_text(encoding="utf-8")
        if yaml_file.resolve() == _SSOT_FILE.resolve():
            continue
        if _PATTERN.search(content):
            violations.append(yaml_file.as_posix())

    assert not violations, (
        "Found category=physical_constant outside SSOT file:\n"
        + "\n".join(sorted(violations))
    )


def test_mean_earth_radius_is_not_duplicated_in_domain_parameters() -> None:
    """Mean Earth radius must have only the physical constants SSOT entry."""
    domain_parameter_file = Path(
        "relic/parameters/database/relic/coupling/scales_parameters.yaml"
    )
    content = domain_parameter_file.read_text(encoding="utf-8")

    assert "coupling.scales.physical.earth_radius:" not in content


def test_generic_year_length_is_not_duplicated_in_domain_parameters() -> None:
    """Year length must come from the selected calendar definition."""
    domain_parameter_file = Path(
        "relic/parameters/database/relic/coupling/scales_parameters.yaml"
    )
    content = domain_parameter_file.read_text(encoding="utf-8")

    assert "coupling.scales.physical.days_per_year:" not in content


def test_isoprene_molar_mass_has_fixed_physical_constant_contract() -> None:
    """Isoprene molar mass must be a fixed, non-trainable chemical constant."""
    payload = yaml.safe_load(_SSOT_FILE.read_text(encoding="utf-8"))
    spec = payload["parameters"]["molecular_weights.isoprene"]

    assert get_physical_constant(
        "relic.physical.constants.molecular_weights.isoprene"
    ) == 68.12
    assert spec["units"] == "g mol-1"
    assert spec["dtype"] == "float64"
    assert spec["bounds"] == [68.12, 68.12]
    assert spec["dimension"] == "molar_mass"
    assert spec["category"] == "chemical_constant"
    assert spec["requires_grad"] is False
    assert spec["references"]
    assert spec["provenance"]["source"]


def _domain_yaml_key_violations(forbidden_keys: set[str]) -> list[str]:
    """List domain YAML files outside the SSOT file that contain a forbidden key."""
    violations: list[str] = []
    for yaml_file in _PARAM_ROOT.rglob("*.yaml"):
        if yaml_file.resolve() == _SSOT_FILE.resolve():
            continue
        content = yaml_file.read_text(encoding="utf-8")
        for forbidden in forbidden_keys:
            if forbidden in content:
                violations.append(f"{yaml_file.as_posix()}: {forbidden}")
    return violations


def test_isoprene_molar_mass_has_no_domain_yaml_aliases() -> None:
    """Domain YAML files must not retain alternate isoprene molar masses."""
    violations = _domain_yaml_key_violations(
        {
            "anthroposphere.emissions.molecular_weights.ISOP:",
            "biosphere.bvoc.molecular_weights.isoprene:",
            "biosphere.bvoc.isoprene_synthase.molecular_weight:",
        }
    )

    assert not violations, "Found duplicate isoprene molar-mass keys:\n" + "\n".join(
        sorted(violations)
    )


@pytest.mark.parametrize(
    ("parameter_name", "expected_value", "reference_domain"),
    [
        ("co", 28.010, "ciaaw.org"),
        ("hcho", 30.0260, "webbook.nist.gov"),
    ],
)
def test_species_molar_mass_has_fixed_physical_constant_contract(
    parameter_name: str, expected_value: float, reference_domain: str
) -> None:
    """
    References:
    - CIAAW (2024), https://ciaaw.org/abridged-atomic-weights.htm (CO).
    - NIST Chemistry WebBook, SRD 69, Formaldehyde, CAS 50-00-0,
      https://webbook.nist.gov/cgi/cbook.cgi?ID=50000 (HCHO).

    Mathematical expression:
    - M_i is the selected conventional bulk molar mass of species i.

    Parameter meanings:
    - parameter_name identifies the canonical species molar-mass entry;
      expected_value is its independently sourced reporting convention.

    Units:
    - M_i and expected_value use g mol-1.

    Domain/constraints:
    - Fixed, positive, binary64, non-trainable constants with complete metadata.
      Fixed bounds select a convention, not an isotopic uncertainty model.

    Sign convention:
    - Positive molar masses; this metadata contract sets no flux direction.
    """
    payload = yaml.safe_load(_SSOT_FILE.read_text(encoding="utf-8"))
    spec = payload["parameters"][f"molecular_weights.{parameter_name}"]

    assert get_physical_constant(
        f"relic.physical.constants.molecular_weights.{parameter_name}"
    ) == expected_value
    assert spec["value"] == expected_value
    assert spec["default"] == expected_value
    assert spec["units"] == "g mol-1"
    assert spec["dtype"] == "float64"
    assert spec["bounds"] == [expected_value, expected_value]
    assert spec["dimension"] == "molar_mass"
    assert spec["category"] == "chemical_constant"
    assert spec["requires_grad"] is False
    assert spec["description"]
    assert spec["tags"]
    assert any(reference_domain in reference for reference in spec["references"])
    assert spec["provenance"]["source"]
    assert spec["provenance"]["last_modified"]


@pytest.mark.parametrize(
    "legacy_species",
    ["CO", "FORM"],
)
def test_species_molar_mass_has_no_domain_yaml_aliases(
    legacy_species: str,
) -> None:
    """Domain YAML files must not retain a migrated physical molar mass."""
    violations = _domain_yaml_key_violations(
        {f"anthroposphere.emissions.molecular_weights.{legacy_species}:"}
    )

    assert not violations, (
        f"Found duplicate {legacy_species} molar-mass keys:\n"
        + "\n".join(sorted(violations))
    )


def test_obsolete_isoprene_parameter_migration_script_is_absent() -> None:
    """A one-time compatibility rewriter must not survive the SSOT migration."""
    assert not Path("scripts/migrate_isoprene_parameters.py").exists()


def test_stable_nuclide_relative_atomic_masses_have_exact_ssot_contracts() -> None:
    """
    References:
    - NIST Atomic Weights and Isotopic Compositions for C, N, and O.
    - CIAAW oxygen three-isotope composition and terminology.

    Mathematical expression:
    - A_r(X) = m(X) / (m(12C) / 12), with A_r(12C) = 12 exactly.

    Parameter meanings:
    - value is relative atomic mass [1]; uncertainty is standard uncertainty [1].
    """
    expected = {
        "nuclides.carbon_12.relative_atomic_mass": (12.0, 0.0),
        "nuclides.carbon_13.relative_atomic_mass": (13.00335483507, 2.3e-10),
        "nuclides.nitrogen_14.relative_atomic_mass": (14.00307400443, 2.0e-10),
        "nuclides.nitrogen_15.relative_atomic_mass": (15.00010889888, 6.4e-10),
        "nuclides.oxygen_16.relative_atomic_mass": (15.99491461957, 1.7e-10),
        "nuclides.oxygen_17.relative_atomic_mass": (16.99913175650, 6.9e-10),
        "nuclides.oxygen_18.relative_atomic_mass": (17.99915961286, 7.6e-10),
    }
    payload = yaml.safe_load(_SSOT_FILE.read_text(encoding="utf-8"))

    for key, (value, uncertainty) in expected.items():
        spec = payload["parameters"][key]
        assert get_physical_constant(
            f"relic.physical.constants.{key}"
        ) == value
        assert spec["value"] == value
        assert spec["default"] == value
        assert spec["uncertainty"] == uncertainty
        assert spec["units"] == "dimensionless"
        assert spec["dtype"] == "float64"
        assert spec["bounds"] == [value, value]
        assert spec["dimension"] == "dimensionless"
        assert spec["category"] == "chemical_constant"
        assert spec["requires_grad"] is False
        assert spec["references"]
        assert spec["provenance"]["source"] == (
            "NIST_Atomic_Weights_and_Isotopic_Compositions"
        )
