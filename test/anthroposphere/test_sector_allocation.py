"""SCM, certificate, and documentation tests for SectorAllocation."""

from __future__ import annotations

from pathlib import Path

from relic.computation.operators.iphs_certificate import (
    load_iphs_equivalence_certificate,
)
from relic.contracts.contract_loader import ContractRegistry
from relic.parameters import get_global_manager
from relic.scientific.anthroposphere.emissions import sector_allocation as module
from relic.scientific.anthroposphere.emissions.sector_allocation import (
    SectorAllocation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = PROJECT_ROOT / "relic" / "contracts" / "variables" / "anthroposphere"
CERTIFICATE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "anthroposphere_emissions_sector_allocation_sector_allocation.yaml"
)
THEORY_PATH = (
    PROJECT_ROOT / "docs" / "source" / "theory" / "anthroposphere_sector_allocation.md"
)


def _registry() -> ContractRegistry:
    registry = ContractRegistry(strict=True)
    for filename in (
        "sector_emission_flux.yaml",
        "spatial_distribution.yaml",
        "temporal_factors.yaml",
    ):
        registry.load_contract_file(CONTRACT_ROOT / filename)
    return registry


def test_sector_flux_contract_has_no_review_marker() -> None:
    text = (CONTRACT_ROOT / "sector_emission_flux.yaml").read_text(encoding="utf-8")
    assert "REVIEW_NEEDED" not in text


def test_sector_flux_contract_is_a_derived_diagnostic_with_one_owner() -> None:
    """forward() returns the split with write mode DERIVED: a per-step derived
    quantity that is never committed, so its contract kind is diagnostic."""
    contract = _registry().get_contract("sector_emission_flux")
    assert contract is not None
    assert contract.kind == "diagnostic"
    assert contract.udunits == "kg m-2 s-1"
    assert contract.cf_standard_name is None
    assert contract.owner == "sector_allocation"
    assert contract.incrementable is False
    assert contract.aliases == []


def test_sector_flux_contract_declares_no_committed_writer() -> None:
    """No Functor commits the derived split; the owner keeps write provenance."""
    registry = _registry()
    contract = registry.get_contract("sector_emission_flux")
    assert contract.raw["writers"] == {"increments": [], "tendencies": []}
    assert contract.raw["flux_contract"]["allowed_in"] == []
    assert contract.raw["flux_contract"]["allowed_out"] == []
    assert registry.can_write("sector_emission_flux", "sector_allocation")
    assert not registry.can_write("sector_emission_flux", "SectorAllocator")
    assert not registry.can_write("sector_emission_flux", "inventory")


def test_spatial_distribution_contract_is_namespaced_diagnostic() -> None:
    contract = _registry().get_contract(
        "diagnostics.sector_allocation.spatial_distribution"
    )
    assert contract is not None
    assert contract.kind == "diagnostic"
    assert contract.owner == "sector_allocation"
    assert contract.incrementable is False
    assert contract.udunits == "dimensionless"


def test_temporal_factor_contract_is_namespaced_unity_diagnostic() -> None:
    contract = _registry().get_contract(
        "diagnostics.sector_allocation.temporal_factors"
    )
    assert contract is not None
    assert contract.kind == "diagnostic"
    assert contract.owner == "sector_allocation"
    assert contract.raw["numerics"]["lower_bound"] == 1.0
    assert contract.raw["numerics"]["upper_bound"] == 1.0


def test_diagnostic_contracts_have_no_old_state_names() -> None:
    registry = _registry()
    assert registry.get_contract("spatial_distribution") is None
    assert registry.get_contract("temporal_factors") is None


def test_new_parameter_specs_are_complete() -> None:
    manager = get_global_manager()
    for key in (
        "anthroposphere.emissions.sector_allocation.proxy_floor",
        "anthroposphere.emissions.sector_allocation.cropland_class_index",
    ):
        spec = manager.get_parameter(key)
        assert spec.units == "dimensionless"
        assert spec.bounds is not None
        assert spec.description
        assert spec.references
        assert spec.provenance


def test_certificate_inputs_and_outputs_match_metadata() -> None:
    allocator = SectorAllocation()
    certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
    assert certificate.input_variables == tuple(allocator.metadata.input_variables)
    assert certificate.output_variables == tuple(allocator.metadata.output_variables)


def test_certificate_parameters_match_runtime_declaration() -> None:
    allocator = SectorAllocation()
    certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
    assert certificate.parameter_keys == allocator.required_parameters()


def test_certificate_is_valid_and_honestly_non_strict() -> None:
    certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
    audit = certificate.validate()
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert certificate.iphs_level == 1
    assert audit["runtime_verified_level"] == 0
    assert audit["runtime_evidence_status"] == "descriptive_only"
    assert certificate.composition_ports == ()
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False


def test_certificate_unit_map_covers_all_declared_ports() -> None:
    certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
    names = set(certificate.input_variables) | set(certificate.output_variables)
    assert names <= set(certificate.unit_map)
    assert certificate.unit_map["sector_emission_flux"] == "kg m-2 s-1"


def test_theory_declares_pointwise_mass_closure_and_replace_semantics() -> None:
    text = THEORY_PATH.read_text(encoding="utf-8")
    assert "sum_s E_sector[s,k,x] = E_input[k,x]" in text
    assert "UpdateItem(REPLACE, DERIVED" in text
    assert 'basis="per_second"' in text
    assert "Level 1 (`iphs_compatible`)" in text
    assert "Levels 2--5" in text


def test_theory_contains_no_secondary_temperature_or_policy_formula() -> None:
    text = THEORY_PATH.read_text(encoding="utf-8")
    assert "f_{temp}" not in text
    assert "f_{policy}" not in text
    assert "apply_increment()" not in text


def test_module_exports_only_canonical_class() -> None:
    assert module.__all__ == ["SectorAllocation"]
    assert not hasattr(module, "SectorAllocator")
    assert not hasattr(module, "SectorAllocationFunctor")
    assert not hasattr(module, "SectorProfile")
    assert not hasattr(module, "TemporalProfile")
    assert not hasattr(module, "SpatialAllocation")


def test_source_contains_no_removed_runtime_paths() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    for token in (
        "_resolve_global_scale",
        "_collect_total_emissions",
        "default_emission_factor",
        "temperature_response",
        "policy_response",
        "sector_emissions_tensor",
        "allocation_efficiency",
    ):
        assert token not in source
