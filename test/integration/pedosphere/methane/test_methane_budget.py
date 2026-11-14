import torch
import pytest

from relic.utils.physical_constants import get_physical_constant

from relic.scientific.pedosphere.methane import (
    MethaneBudgetCoupler,
    MethaneOxidationFunctor,
    MethaneProductionFunctor,
    MethaneScaleCouplingFunctor,
    MethaneTransportFunctor,
)
from relic.core.state import UpdateItem, UpdateMode


def apply_updates(state, updates):
    for key, value in updates.items():
        mode = UpdateMode.ACCUMULATE
        raw_value = value
        if isinstance(value, UpdateItem):
            raw_value = value.value
            mode = value.mode
        tensor = raw_value if isinstance(raw_value, torch.Tensor) else torch.tensor(float(raw_value))
        tensor = tensor.to(torch.float32)
        if key in state and mode != UpdateMode.REPLACE:
            state[key] = state[key] + tensor
        else:
            state[key] = tensor.clone()
    return state


def run_methane_step(state, dt_sec):
    dt_tensor = torch.tensor(float(dt_sec))
    functors = [
        MethaneScaleCouplingFunctor(),
        MethaneProductionFunctor(),
        MethaneOxidationFunctor(),
        MethaneTransportFunctor(),
        MethaneBudgetCoupler(),
    ]
    for functor in functors:
        updates = functor(state, time=0.0, dt=dt_tensor)
        apply_updates(state, updates)
    return state


def test_mass_balance_over_multi_step_budget():
    # Three grid cells with varying saturation
    initial_state = {
        "soil_temperature": torch.tensor(292.0),
        "soil_ph": torch.tensor(6.8),
        "heterotrophic_respiration": torch.tensor(5.5 / SECONDS_PER_DAY),
        "redox_potential": torch.tensor(-190.0),
        "soil_co2_concentration": torch.tensor(0.18),
        "microbial_activity_index": torch.tensor(0.95),
        "ch4_soil_storage": torch.tensor(7.0),
        "land_surface.wetland_fraction": torch.tensor(0.65),
        "water_filled_pore_space": torch.tensor(0.7),
        "methane_data_score": torch.tensor(0.6),
        "root_length_density": torch.tensor(0.2),
        "soil_o2_concentration": torch.tensor(0.02),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
    }

    state = {key: value.clone() for key, value in initial_state.items()}
    dt = 1800.0
    storage_before_step = state["ch4_soil_storage"].clone()
    for _ in range(3):
        storage_before_step = state["ch4_soil_storage"].clone()
        run_methane_step(state, dt)

    residual = state["diagnostics.methane_budget.mass_balance_residual"].abs().max().item()
    applied_storage_delta = state["ch4_soil_storage"] - storage_before_step
    storage_delta = applied_storage_delta.abs().max().item()
    emission = state["ch4_atmospheric_flux"].abs().max().item()
    emission_mass = state[
        "diagnostics.methane_budget.atmospheric_ch4_mass_flux"
    ].abs().max().item()

    assert residual < 5e-4
    assert storage_delta > 0.0
    assert state["ch4_soil_storage"].min().item() >= 0.0
    assert state["ch4_soil_storage"].dtype == storage_before_step.dtype
    torch.testing.assert_close(
        applied_storage_delta,
        state["diagnostics.methane_budget.storage_tendency_net"].to(
            applied_storage_delta.dtype
        )
        * dt,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert state["ch4_atmospheric_flux"].min().item() >= 0.0
    assert state[
        "diagnostics.methane_budget.atmospheric_ch4_mass_flux"
    ].min().item() >= 0.0
    limiter = state["diagnostics.methane_budget.transport_limiter_residual"]
    requested_transport_loss = state["ch4_transport_loss_rate"]
    assert limiter.min().item() >= 0.0
    torch.testing.assert_close(
        state["ch4_atmospheric_flux"],
        requested_transport_loss - limiter,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    expected_storage_rate = (
        state["ch4_production_rate"]
        - state["ch4_oxidation_rate"]
        - state["ch4_atmospheric_flux"]
    )
    storage_rate = (applied_storage_delta / dt).to(expected_storage_rate.dtype)
    torch.testing.assert_close(
        expected_storage_rate,
        storage_rate,
        rtol=1.0e-5,
        atol=1.0e-8,
    )
    for removed_name in (
        "ch4_emission",
        "ch4_emission_mass",
        "ch4_mass_balance_residual",
        "ch4_storage_tendency_net",
        "ch4_soil_storage_delta",
    ):
        assert removed_name not in state
    assert "ch4_oxidation_rate" in state
    assert "ch4_storage_tendency_oxid" in state
    assert "ch4_oxidation_imaple_rate" not in state
    assert "ch4_oxidation_dndc_rate" not in state
    torch.testing.assert_close(
        state["ch4_oxidation_rate"] + state["ch4_storage_tendency_oxid"],
        torch.zeros_like(state["ch4_oxidation_rate"]),
        atol=0.0,
        rtol=0.0,
    )

    ch4_molar_mass = get_physical_constant(
        "relic.physical.constants.molecular_weights.ch4"
    )
    carbon_molar_mass = get_physical_constant(
        "relic.physical.constants.molecular_weights.carbon"
    )
    if isinstance(ch4_molar_mass, torch.Tensor):
        ch4_molar_mass = float(ch4_molar_mass.detach().cpu().item())
    if isinstance(carbon_molar_mass, torch.Tensor):
        carbon_molar_mass = float(carbon_molar_mass.detach().cpu().item())
    conversion_ratio = float(ch4_molar_mass) / float(carbon_molar_mass)
    assert emission_mass == pytest.approx(emission * conversion_ratio, rel=1e-6, abs=1e-12)


def test_transport_diffusion_flux_preserves_directional_sign():
    transport = MethaneTransportFunctor()
    dt = torch.tensor(600.0)

    base_state = {
        "ch4_soil_storage": torch.tensor(1.0),
        "water_filled_pore_space": torch.tensor(0.4),
        "porosity": torch.tensor(0.45),
    }

    emission_case = dict(base_state)
    emission_case["atmospheric_ch4_concentration"] = torch.tensor(0.0)
    emission_updates = transport.forward(emission_case, time=0.0, dt=dt)

    uptake_case = dict(base_state)
    uptake_case["atmospheric_ch4_concentration"] = torch.tensor(100.0)
    uptake_updates = transport.forward(uptake_case, time=0.0, dt=dt)

    emission_flux = emission_updates[
        "diagnostics.methane_transport.diffusion_rate"
    ]
    uptake_flux = uptake_updates[
        "diagnostics.methane_transport.diffusion_rate"
    ]
    assert isinstance(emission_flux, UpdateItem)
    assert isinstance(uptake_flux, UpdateItem)
    assert emission_flux.value.item() > 0.0
    assert uptake_flux.value.item() < 0.0
    for removed_name in (
        "ch4_diffusion_rate",
        "ch4_ebullition_rate",
        "ch4_plant_flux_rate",
    ):
        assert removed_name not in emission_updates
        assert removed_name not in uptake_updates


SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))

