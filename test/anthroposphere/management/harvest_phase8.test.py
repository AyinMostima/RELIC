from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.scientific.anthroposphere.management.harvest import HarvestEvent, HarvestManagement


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _state() -> dict[str, torch.Tensor]:
    return {
        "aboveground_biomass": torch.tensor([[4.0]], dtype=torch.float32),
        "belowground_biomass": torch.tensor([[1.0]], dtype=torch.float32),
        "leaf_area_index": torch.tensor([[3.0]], dtype=torch.float32),
        "crop_maturity": torch.tensor([[1.0]], dtype=torch.float32),
        "residue_cover": torch.tensor([[0.4]], dtype=torch.float32),
    }


def test_harvest_declares_canonical_iphs_biomass_residue_port():
    manager = HarvestManagement(config={"calendar_type": "noleap"})

    assert manager.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert manager.runtime_path == "canonical_iphs_harvest_biomass_residue_port"

    forms = manager.declared_forms()
    assert {form.semantic_name for form in forms} >= {
        "harvest_biomass_pool",
        "harvest_residue_flux",
        "harvest_management_port",
    }

    hodge = manager.build_hodge_contract(strict=True)
    audit = hodge.validate(passive=True)
    assert audit["psd"] is True
    assert torch.all(torch.diag(hodge.matrix) >= 0.0)

    ports = manager.ports()
    assert len(ports) == 1
    assert ports[0].name == "harvest_biomass_residue_port"
    assert ports[0].port_kind == "matter"

    energy = manager.energy_terms(_state())
    assert "harvest_free_energy" in energy
    assert "harvest_chemical_potential" in energy


def test_harvest_biomass_is_checked_against_its_physical_domain():
    """Harvest biomass is checked against its physical domain (non-negative mass,
    the contract bounds): a forest stand above 10 kg m-2 aboveground and 6 kg DM
    m-2 belowground is read as given, and a negative mass raises instead of being
    clipped."""
    manager = HarvestManagement(config={"calendar_type": "noleap"})
    state = _state()
    state["aboveground_biomass"] = torch.tensor([[30.0]], dtype=torch.float32)
    state["belowground_biomass"] = torch.tensor([[10.0]], dtype=torch.float32)
    outputs = manager.forward(state, time=0.0, dt=86400.0, day_of_year=270)
    assert all(
        torch.isfinite(item.value if hasattr(item, "value") else item).all()
        for item in outputs.values()
        if isinstance(getattr(item, "value", item), torch.Tensor)
    )
    for key in ("aboveground_biomass", "belowground_biomass"):
        invalid = _state()
        invalid[key] = torch.tensor([[-1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match=key):
            manager.forward(invalid, time=0.0, dt=86400.0, day_of_year=270)


def test_harvest_biomass_has_no_parameter_default():
    """aboveground_biomass and belowground_biomass have no ParameterManager fallback
    in forward or in the partition kernel; the *_biomass_default parameters remain
    only the nominal stocks of the representative harvest-property query."""
    manager = HarvestManagement(config={"calendar_type": "noleap"})
    for key in ("aboveground_biomass", "belowground_biomass"):
        state = _state()
        state.pop(key)
        with pytest.raises(KeyError, match=key):
            manager.forward(state, time=0.0, dt=86400.0, day_of_year=270)
        with pytest.raises(ValueError, match=key):
            manager._canonical_iphs_harvest_partition(state, event=HarvestEvent())


def test_harvest_residue_cover_has_no_default():
    """residue_cover is a declared input that no RELIC Functor writes: a missing value
    raises ValueError in the partition kernel and in forward instead of taking a zero
    (bare-soil) residue cover."""
    manager = HarvestManagement(config={"calendar_type": "noleap"})
    assert "residue_cover" in manager.get_required_inputs()
    state = _state()
    state.pop("residue_cover")
    with pytest.raises(ValueError, match="residue_cover"):
        manager._canonical_iphs_harvest_partition(state, event=HarvestEvent())
    with pytest.raises(ValueError, match="residue_cover"):
        manager.forward(state, time=0.0, dt=86400.0, day_of_year=270)


def test_harvest_reads_no_nominal_reference_state_parameters():
    """The representative property query and its reference state are removed;
    HarvestManagement reads none of the former nominal/default keys."""
    manager = HarvestManagement(config={"calendar_type": "noleap"})
    assert not hasattr(manager, "_get_harvest_properties")
    former = {
        "anthroposphere.management.aboveground_biomass_default",
        "anthroposphere.management.belowground_biomass_default",
        "anthroposphere.management.harvest.grain_moisture_content",
        "anthroposphere.management.irrigation.crop_stage_default",
        "atmosphere.transport.air_temperature_default",
    }
    assert not former & set(manager.required_parameters())
    assert not former & set(manager.metadata.parameters)


def test_harvest_state_advancer_uses_potential_and_mobility():
    manager = HarvestManagement(config={"calendar_type": "noleap"})

    rates = manager._canonical_iphs_harvest_partition(_state(), event=HarvestEvent())

    assert "state_advancer" in rates
    assert "chemical_potential" in rates
    assert "free_energy" in rates
    assert torch.isfinite(rates["state_advancer"]).all()
    assert torch.isfinite(rates["chemical_potential"]).all()
    assert torch.isfinite(rates["free_energy"]).all()
    assert torch.allclose(rates["biomass_removal_rate"], rates["state_advancer"][2])


def test_harvest_forward_reports_canonical_runtime_and_potential_diagnostics():
    manager = HarvestManagement(config={"calendar_type": "noleap"})

    outputs = manager.forward(_state(), time=0.0, dt=86400.0, day_of_year=270)

    values = {name: item.value for name, item in outputs.items()}
    assert values["diagnostics.harvest_management.canonical_iphs_runtime"].item() == 1.0
    assert torch.isfinite(values["harvest_flux"]).all()
    assert torch.isfinite(
        values["diagnostics.harvest.biomass_partition_residual"]
    ).all()
    assert torch.isfinite(values["diagnostics.harvest.carbon_ledger_residual"]).all()
    assert torch.isfinite(values["diagnostics.harvest.nitrogen_ledger_residual"]).all()
    assert torch.isfinite(values["diagnostics.harvest_management.free_energy"]).all()
    assert torch.isfinite(
        values["diagnostics.harvest_management.chemical_potential_norm"]
    ).all()


def test_harvest_conservation_hook_normalizes_each_ledger_before_reduction():
    """Compare only dimensionless residuals across DM, C, and N ledgers."""
    manager = HarvestManagement(config={"calendar_type": "noleap"})
    input_state = {"aboveground_biomass": torch.tensor([4.0])}
    outputs = {
        "diagnostics.harvest.biomass_partition_residual": torch.tensor([0.4]),
        "diagnostics.harvest.carbon_ledger_residual": torch.tensor([0.2]),
        "diagnostics.harvest.nitrogen_ledger_residual": torch.tensor([0.03]),
        "harvest_biomass_carbon_input": torch.tensor([2.0]),
        "harvest_carbon_export_flux": torch.tensor([1.2]),
        "residue_c_flux": torch.tensor([0.8]),
        "harvest_biomass_nitrogen_input": torch.tensor([0.3]),
        "harvest_nitrogen_export_flux": torch.tensor([0.18]),
        "residue_n_flux": torch.tensor([0.12]),
    }

    result = manager.check_conservation_laws(input_state, outputs)

    assert result == {"mass": pytest.approx(0.1)}


def test_harvest_source_has_no_parallel_runtime_terms():
    source = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "management"
        / "harvest.py"
    ).read_text(encoding="utf-8")

    forbidden_terms = (
        "fallback",
        "legacy_mode",
        "use_legacy",
        "enable_legacy",
        "compute_mode",
        "zero_flux_fallback",
        "_calculate_harvest_biomass",
        "_calculate_residue_return",
        "_calculate_root_turnover",
    )
    for term in forbidden_terms:
        assert term not in source


def test_harvest_phase8_certificate_is_valid():
    manager = HarvestManagement(config={"calendar_type": "noleap"})
    certificate = load_iphs_equivalence_certificate(
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_management_harvest_harvest_management.yaml"
    )

    audit = certificate.validate()
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False
    assert certificate.dependency_interconnection_map["direct_functor_calls"] == (
        "none; biomass, crop stage, moisture, residue, and schedule are state "
        "or boundary inputs and forward calls only local harvest helpers"
    )
    assert certificate.state_advancement_map["state_advancer"]
    assert set(certificate.parameter_keys) == set(manager.required_parameters())
    assert not any(
        key.startswith("relic.physical.constants.")
        for key in certificate.parameter_keys
    )
    physical_key = "relic.physical.constants.time.seconds_per_day"
    assert physical_key not in certificate.parameter_map
    assert certificate.parameter_management_map[physical_key]["source"] == (
        "physical constants SSOT via get_physical_constant"
    )
