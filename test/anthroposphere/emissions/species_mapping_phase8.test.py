from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.contracts.contract_loader import ContractRegistry
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.emissions.species_mapping import (
    SpeciesMapper,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _valid_state() -> dict[str, torch.Tensor]:
    return {
        "sector_emission_flux": torch.arange(
            1,
            5 * 6 * 2 * 2 + 1,
            dtype=torch.float64,
        ).reshape(5, 6, 2, 2) * 1.0e-12,
    }


def test_species_mapper_declares_canonical_iphs_chain_map():
    mapper = SpeciesMapper()

    assert mapper.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert mapper.runtime_path == "canonical_iphs_species_mapping_chain_map"

    forms = mapper.declared_forms()
    assert {form.semantic_name for form in forms} == {
        "sector_inventory_species_mass_flux",
        "mechanism_gas_mass_flux",
        "mechanism_gas_molar_flux",
        "mechanism_aerosol_mass_flux",
    }

    hodge = mapper.build_hodge_contract(strict=True)
    assert hodge.validate(passive=True)["psd"] is True
    assert torch.all(torch.diag(hodge.matrix) > 0.0)

    ports = mapper.ports()
    assert [port.name for port in ports] == [
        "species_mapping_gas_molar_flux",
        "species_mapping_aerosol_mass_flux",
    ]
    assert all(port.port_kind == "matter" for port in ports)

    required = set(mapper.required_parameters())
    assert "anthroposphere.emissions.species_mapping.default_mechanism" in required
    assert "anthroposphere.emissions.molecular_weights.NOx" in required


def test_species_mapper_forward_reports_canonical_diagnostics():
    mapper = SpeciesMapper()

    state = _valid_state()
    result = mapper.forward(state, time=0.0, dt=60.0)

    assert result["diagnostics.species_mapping.canonical_iphs_runtime"].item() == 1.0
    assert set(result) == set(mapper.metadata.output_variables)
    gas = result["mechanism_gas_molar_flux"]
    aerosol = result["mechanism_aerosol_mass_flux"]
    assert isinstance(gas, UpdateItem)
    assert isinstance(aerosol, UpdateItem)
    assert gas.mode is UpdateMode.REPLACE and gas.per_timestep is False
    assert aerosol.mode is UpdateMode.REPLACE and aerosol.per_timestep is False
    assert gas.value.shape == (len(mapper.mechanism_gas_species_order), 2, 2)
    assert aerosol.value.shape == (
        len(mapper.mechanism_aerosol_species_order),
        2,
        2,
    )
    relative_error = result["diagnostics.species_mapping.relative_error"]
    assert torch.all(relative_error <= 1.0e-12)
    assert torch.allclose(
        result["diagnostics.species_mapping.total_input_mass_flux"],
        state["sector_emission_flux"].sum(dim=(0, 1)),
    )
    violations = mapper.check_conservation_laws(state, result)
    assert violations["mass"] <= 1.0e-12


def test_species_mapper_rejects_negative_flux_without_clipping():
    mapper = SpeciesMapper()

    with pytest.raises(ValueError):
        mapper.mass_to_molar_flux(torch.tensor([-1.0], dtype=torch.float32), "NO")

    bad_state = _valid_state()
    bad_state["sector_emission_flux"][0, 1] = -1.0
    with pytest.raises(ValueError):
        mapper.forward(bad_state, time=0.0, dt=60.0)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        mapper.forward(
            _valid_state(),
            time=0.0,
            dt=60.0,
            chemical_mechanism="cb05",
        )


def test_species_mapper_phase8_certificate_is_valid():
    certificate = load_iphs_equivalence_certificate(
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_emissions_species_mapping_species_mapper.yaml"
    )

    audit = certificate.validate()
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False
    assert certificate.dependency_interconnection_map["direct_functor_calls"].startswith("none")
    mapper = SpeciesMapper()
    assert set(certificate.input_variables) == set(mapper.metadata.input_variables)
    assert set(certificate.output_variables) == set(mapper.metadata.output_variables)
    assert set(certificate.parameter_keys) == set(mapper.required_parameters())
    assert set(certificate.output_variables) <= set(certificate.state_map)
    assert set(certificate.input_variables) | set(certificate.output_variables) <= set(
        certificate.unit_map
    )


@pytest.mark.parametrize(
    ("variable", "unit"),
    [
        ("mechanism_gas_molar_flux", "mol m-2 s-1"),
        ("mechanism_aerosol_mass_flux", "kg m-2 s-1"),
    ],
)
def test_species_mapper_scm_matches_runtime_writer(variable: str, unit: str):
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(
        PROJECT_ROOT
        / "relic"
        / "contracts"
        / "variables"
        / "anthroposphere"
        / f"{variable}.yaml"
    )
    contract = registry.get_contract(variable)
    assert contract is not None
    assert contract.owner == "species_mapping"
    assert contract.kind == "flux"
    assert contract.udunits == unit
    assert contract.incrementable is False
    assert contract.raw["writers"]["tendencies"] == [
        {
            "name": "species_mapping",
            "module": (
                "relic.scientific.anthroposphere.emissions.species_mapping"
            ),
            "provides": [variable],
            "unit": unit,
        }
    ]
    assert contract.raw["flux_contract"]["allowed_in"] == ["species_mapping"]
    assert contract.raw["flux_contract"]["allowed_out"] == []


def test_species_mapper_forward_has_no_environment_parameter():
    """sector_emission_flux, the declared input, comes from the state; forward takes
    (state, time, dt) only, so an environment mapping is rejected (item 112)."""
    import inspect

    parameters = inspect.signature(SpeciesMapper.forward).parameters
    assert list(parameters) == ["self", "state", "time", "dt"]


def test_species_mapper_certificate_records_the_forward_signature():
    import yaml

    certificate = yaml.safe_load(
        (
            PROJECT_ROOT
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "anthroposphere_emissions_species_mapping_species_mapper.yaml"
        ).read_text(encoding="utf-8")
    )
    assert certificate["interface_map"]["forward_signature"] == (
        "forward(self, state, time, dt)"
    )
