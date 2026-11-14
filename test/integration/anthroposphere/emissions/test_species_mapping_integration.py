# -*- coding: ascii -*-
"""Integration tests for SpeciesMapper wiring and cross-domain coupling."""
import torch

from relic.contracts.contract_loader import load_contract_registry
from relic.core.simulator import RELICSimulator
from relic.core.state import StateWriteMode, UpdateItem, UpdateMode, partition_outputs
from relic.scientific.anthroposphere.emissions.sector_allocation import (
    SectorAllocation,
)
from relic.scientific.anthroposphere.emissions.species_mapping import SpeciesMapper


def _sample_emissions(device: torch.device) -> dict:
    """Provide representative canonical inventory ports for integration tests."""
    return {
        "anthropogenic_nox_flux": torch.tensor([[80.0]], device=device),
        "anthropogenic_co_flux": torch.tensor([[60.0]], device=device),
        "anthropogenic_so2_flux": torch.tensor([[25.0]], device=device),
        "anthropogenic_voc_flux": torch.tensor([[40.0]], device=device),
        "anthropogenic_nh3_flux": torch.tensor([[3.0]], device=device),
        "anthropogenic_pm25_flux": torch.tensor([[5.0]], device=device),
    }


def _allocate_inventory(emissions: dict) -> UpdateItem:
    """Route canonical inventory ports through the sole sector allocator."""
    reference = emissions["anthropogenic_nox_flux"]
    state = {
        **emissions,
        "population_density": torch.ones_like(reference),
        "gdp_density": torch.ones_like(reference),
        "road_density": torch.ones_like(reference),
        "land_use_fractions": torch.full(
            (4, *reference.shape),
            0.25,
            dtype=reference.dtype,
            device=reference.device,
        ),
    }
    output = SectorAllocation().forward(state, time=0.0, dt=3600.0)
    item = output["sector_emission_flux"]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.per_timestep is False
    return item


def test_species_mapper_forward_outputs():
    """SpeciesMapper should expose typed ports and a closed equivalent-mass ledger."""
    device = torch.device("cpu")
    mapper = SpeciesMapper()
    emissions = _sample_emissions(device)
    sector_item = _allocate_inventory(emissions)
    state = {"sector_emission_flux": sector_item.value}
    outputs = mapper.forward(state, time=0.0, dt=3600.0)

    assert set(outputs) == set(mapper.metadata.output_variables)
    gas = outputs["mechanism_gas_molar_flux"]
    aerosol = outputs["mechanism_aerosol_mass_flux"]
    assert isinstance(gas, UpdateItem) and isinstance(aerosol, UpdateItem)
    assert gas.mode is UpdateMode.REPLACE and gas.per_timestep is False
    assert aerosol.mode is UpdateMode.REPLACE and aerosol.per_timestep is False
    assert gas.value.shape == (len(mapper.mechanism_gas_species_order), 1, 1)
    assert aerosol.value.shape == (
        len(mapper.mechanism_aerosol_species_order),
        1,
        1,
    )
    assert torch.all(
        outputs["diagnostics.species_mapping.relative_error"] < 1.0e-6
    )
    torch.testing.assert_close(
        outputs[
            "diagnostics.species_mapping.inventory_equivalent_output_mass_flux"
        ],
        sector_item.value.sum(dim=(0, 1)),
    )
    assert mapper.check_conservation_laws(state, outputs)["mass"] < 1.0e-6


def test_sector_allocation_output_is_a_derived_quantity_of_the_step():
    """The split is DERIVED: the later same-group consumer reads it, it is never
    committed as state, and its SCM contract declares the same channel."""
    allocator = SectorAllocation()
    mapper = SpeciesMapper()
    emissions = _sample_emissions(torch.device("cpu"))
    reference = emissions["anthropogenic_nox_flux"]
    # No sector_emission_flux initial value: a derived quantity has no state.
    state = {
        **emissions,
        "population_density": torch.ones_like(reference),
        "gdp_density": torch.ones_like(reference),
        "road_density": torch.ones_like(reference),
        "land_use_fractions": torch.full(
            (4, *reference.shape),
            0.25,
            dtype=reference.dtype,
            device=reference.device,
        ),
    }
    simulator = RELICSimulator(
        {
            "time_step": 60.0,
            "max_time": 60.0,
            "conservation_checks": False,
            "device": "cpu",
        }
    )
    simulator.initialize_state(state)
    simulator.register_functor(allocator, execution_group="fast")
    simulator.register_functor(mapper, execution_group="fast")
    assert simulator.execution_order.index("sector_allocation") < (
        simulator.execution_order.index("species_mapping")
    )

    result = simulator.step(dt=60.0)

    # References: Crippa et al. (2020); CMAQ/SMOKE sector allocation.
    # Mathematical expression: sum_s E_sector[s,k,x] = E_input[k,x] and the
    #   mapper input mass M_in[x] = sum_s sum_k E_sector[s,k,x] = sum_k E_input[k,x].
    # Parameter meanings: E_input the six inventory mass fluxes, E_sector the
    #   derived sector split, M_in the mapper's total input mass flux.
    # Units: E_input, E_sector, M_in [kg m-2 s-1].
    # Domain/constraints: non-negative fluxes; one fast-group call.
    # Sign convention: all fluxes are positive emissions into the atmosphere.
    expected = torch.stack([state[key] for key in allocator.CANONICAL_INPUTS])
    torch.testing.assert_close(result["sector_emission_flux"].sum(dim=0), expected)
    torch.testing.assert_close(
        result["diagnostics.species_mapping.total_input_mass_flux"], expected.sum(dim=0)
    )
    assert simulator.state_manager.get_state("sector_emission_flux") is None
    item = allocator.forward(state, time=0.0, dt=60.0)["sector_emission_flux"]
    assert item.write_mode is StateWriteMode.DERIVED
    committed_channel, diagnostic_channel, _ = partition_outputs(
        {"sector_emission_flux": item}
    )
    assert "sector_emission_flux" in diagnostic_channel and not committed_channel
    assert load_contract_registry().get_contract("sector_emission_flux").kind == "diagnostic"
    provenance = simulator.get_variable_provenance("sector_emission_flux")
    assert [entry["functor"] for entry in provenance] == [
        "sector_allocation",
        "sector_allocation",
    ]
    assert [entry["role"] for entry in provenance] == ["writer", "final_writer"]


def test_species_mapping_reads_the_derived_split_when_registered_first():
    """Reversed registration: the declared dependency still runs sector_allocation
    first, so the mapper reads the DERIVED split of the same group call."""
    allocator = SectorAllocation()
    mapper = SpeciesMapper()
    emissions = _sample_emissions(torch.device("cpu"))
    reference = emissions["anthropogenic_nox_flux"]
    state = {
        **emissions,
        "population_density": torch.ones_like(reference),
        "gdp_density": torch.ones_like(reference),
        "road_density": torch.ones_like(reference),
        "land_use_fractions": torch.full(
            (4, *reference.shape),
            0.25,
            dtype=reference.dtype,
            device=reference.device,
        ),
    }
    simulator = RELICSimulator(
        {
            "time_step": 60.0,
            "max_time": 60.0,
            "conservation_checks": False,
            "device": "cpu",
        }
    )
    simulator.initialize_state(state)
    # The consumer is registered before its producer on purpose.
    simulator.register_functor(mapper, execution_group="fast")
    simulator.register_functor(allocator, execution_group="fast")
    order = simulator.execution_order
    assert order is not None
    assert order.index("sector_allocation") < order.index("species_mapping")

    result = simulator.step(dt=60.0)

    # References: Crippa et al. (2020); CMAQ/SMOKE sector allocation.
    # Mathematical expression: M_in[x] = sum_s sum_k E_sector[s,k,x] = sum_k E_input[k,x].
    # Parameter meanings: E_input the six inventory mass fluxes, E_sector the
    #   derived sector split, M_in the mapper's total input mass flux.
    # Units: E_input, E_sector, M_in [kg m-2 s-1].
    # Domain/constraints: non-negative fluxes; one fast-group call.
    # Sign convention: all fluxes are positive emissions into the atmosphere.
    expected = torch.stack([state[key] for key in allocator.CANONICAL_INPUTS])
    torch.testing.assert_close(
        result["diagnostics.species_mapping.total_input_mass_flux"], expected.sum(dim=0)
    )
    assert simulator.state_manager.get_state("sector_emission_flux") is None

