from pathlib import Path

import pytest

import relic.scientific.anthroposphere as anthroposphere_package
import relic.scientific.anthroposphere.emissions as emissions_package
from relic.scientific.anthroposphere.emissions import (
    atmosphere_interface_contract as interface_module,
)
from relic.scientific.anthroposphere.emissions.atmosphere_interface_contract import (
    ChemicalMechanism,
)
from relic.scientific.anthroposphere.emissions.species_mapping import SpeciesMapper


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_interface_module_keeps_only_the_live_mechanism_taxonomy():
    assert interface_module.__all__ == ["ChemicalMechanism"]
    for removed in (
        "AtmosphereInterfaceContract",
        "CoordinateSystem",
        "InterfaceVariable",
    ):
        assert not hasattr(interface_module, removed)
        with pytest.raises(AttributeError):
            getattr(interface_module, removed)


def test_packages_do_not_export_the_removed_interface_process():
    for package in (emissions_package, anthroposphere_package):
        assert "ChemicalMechanism" in package.__all__
        for removed in (
            "AtmosphereInterfaceContract",
            "CoordinateSystem",
            "InterfaceVariable",
        ):
            assert removed not in package.__all__
            assert not hasattr(package, removed)


def test_species_mapper_uses_the_exact_single_mechanism_enum():
    mapper = SpeciesMapper()
    assert mapper.default_mechanism is ChemicalMechanism.CB05
    assert mapper.get_supported_mechanisms() == [ChemicalMechanism.CB05]
    assert SpeciesMapper.__init__.__globals__["ChemicalMechanism"] is ChemicalMechanism

    unsupported = tuple(
        mechanism
        for mechanism in ChemicalMechanism
        if mechanism is not ChemicalMechanism.CB05
    )
    assert unsupported
    for mechanism in unsupported:
        with pytest.raises(
            NotImplementedError,
            match="lacks a complete typed runtime contract",
        ):
            mapper.get_mechanism_species_list(mechanism)


def test_removed_interface_has_no_production_definition_or_export():
    occurrences = []
    source_root = PROJECT_ROOT / "relic"
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "AtmosphereInterfaceContract" in text:
            occurrences.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert occurrences == []


def test_source_contains_no_duplicate_contract_formula_or_false_structure():
    source_path = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "emissions"
        / "atmosphere_interface_contract.py"
    )
    source = source_path.read_text(encoding="utf-8")
    forbidden_tokens = (
        "ScaleCouplingFunctor",
        "FunctorMetadata",
        "InterfaceVariable",
        "CoordinateSystem",
        "EMISSION_FLUX_VARIABLES",
        "create_interface_mapping_dict",
        "validate_state_variables",
        "check_interface_compliance",
        "convert_mass_flux_to_mixing_ratio_tendency",
        "convert_mixing_ratio_to_number_density",
        "CochainSpaceSpec",
        "HodgeStarSpec",
        "PortSpecIPHS",
        "build_hodge_contract",
        "dissipative_structure",
        "declared_forms",
        "required_parameters",
        "get_dependencies",
        "compute",
        "canonical_iphs_runtime",
        "runtime_path",
        "atmospheric_chemical_potential",
        "anthroposphere_emissions",
        "meteorological_state",
    )

    assert "class ChemicalMechanism(Enum)" in source
    assert source.count("class ChemicalMechanism(Enum)") == 1
    for token in forbidden_tokens:
        assert token not in source


def test_obsolete_direct_interface_suite_and_certificate_are_removed():
    assert not (
        PROJECT_ROOT
        / "test"
        / "unit"
        / "anthroposphere"
        / "emissions"
        / "test_interface_contract.py"
    ).exists()
    assert not (
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_emissions_atmosphere_interface_contract_atmosphere_interface_contract.yaml"
    ).exists()
