from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.emissions.sector_allocation import SectorAllocation

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _state() -> dict[str, torch.Tensor]:
    grid = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    state = {
        key: grid * (index * 1.0e-9)
        for index, key in enumerate(SectorAllocation.CANONICAL_INPUTS, start=1)
    }
    state.update({
        "population_density": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=grid.dtype),
        "gdp_density": torch.tensor([[2.0, 2.0], [1.0, 3.0]], dtype=grid.dtype),
        "road_density": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=grid.dtype),
        "land_use_fractions": torch.ones((4, 2, 2), dtype=grid.dtype),
    })
    return state


def test_sector_allocation_has_no_false_iphs_structure_surface():
    allocator = SectorAllocation()

    assert not hasattr(allocator, "PHASE8_RUNTIME_REPLACEMENT_STATUS")
    assert not hasattr(allocator, "runtime_path")
    for name in (
        "compute",
        "declared_forms",
        "build_hodge_contract",
        "build_hodge_terms",
        "reversible_structure",
        "dissipative_structure",
        "ports",
    ):
        assert not hasattr(allocator, name)
    assert "diagnostics.sector_allocation.canonical_iphs_runtime" not in (
        allocator.metadata.output_variables
    )


def test_sector_allocation_forward_conserves_mass_with_complete_update_contract():
    allocator = SectorAllocation()
    state = _state()
    output = allocator.forward(state, time=0.0, dt=3600.0)

    item = output["sector_emission_flux"]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.per_timestep is False
    assert item.unit == "kg m-2 s-1"
    assert item.basis == "per_second"
    assert item.source_functor == allocator.name
    inputs = torch.stack([state[key] for key in allocator.CANONICAL_INPUTS])
    assert torch.allclose(item.value.sum(dim=0), inputs, rtol=1.0e-12, atol=1.0e-18)
    assert allocator.check_conservation_laws(state, output)["mass"] == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_sector_allocation_rejects_invalid_emission_domain():
    allocator = SectorAllocation()
    bad_state = _state()
    bad_state["anthropogenic_nox_flux"] = -torch.ones((2, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="non-negative"):
        allocator.forward(bad_state, time=0.0, dt=3600.0)


def test_sector_allocation_phase8_certificate_is_valid():
    certificate = load_iphs_equivalence_certificate(
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_emissions_sector_allocation_sector_allocation.yaml"
    )

    audit = certificate.validate()
    assert audit["passed"] is True
    assert audit["issue_count"] == 0
    assert certificate.iphs_level == 1
    assert audit["runtime_verified_level"] == 0
    assert audit["runtime_evidence_status"] == "descriptive_only"
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False


def test_sector_allocation_forward_has_no_environment_parameter():
    """The emission rates and proxies come from the declared state; forward takes
    (state, time, dt) only, so an environment mapping is rejected (item 113)."""
    import inspect

    import yaml

    parameters = inspect.signature(SectorAllocation.forward).parameters
    assert list(parameters) == ["self", "state", "time", "dt"]
    certificate = yaml.safe_load(
        (
            PROJECT_ROOT
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "anthroposphere_emissions_sector_allocation_sector_allocation.yaml"
        ).read_text(encoding="utf-8")
    )
    assert certificate["interface_map"]["forward_signature"] == (
        "forward(self, state, time, dt)"
    )
