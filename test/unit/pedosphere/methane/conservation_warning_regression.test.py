import warnings

import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.methane import (
    MethaneBudgetCoupler,
    MethaneOxidationFunctor,
    MethaneProductionFunctor,
    MethaneScaleCouplingFunctor,
    MethaneTransportFunctor,
)
from relic.utils.physical_constants import get_physical_constant


SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))


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


def build_state():
    return {
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
        "soil_o2_concentration": torch.tensor(0.02),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
        "ch4_production_rate": torch.tensor(0.0),
        "ch4_oxidation_rate": torch.tensor(0.0),
        "ch4_transport_loss_rate": torch.tensor(0.0),
        "ch4_storage_tendency_prod": torch.tensor(0.0),
        "ch4_storage_tendency_oxid": torch.tensor(0.0),
        "ch4_storage_tendency_trans": torch.tensor(0.0),
    }


def collect_conservation_messages(records):
    return [
        str(record.message)
        for record in records
        if "Conservation law" in str(record.message)
    ]


def initialize_scale_state(state):
    scale = MethaneScaleCouplingFunctor()
    updates = scale(state, time=0.0, dt=torch.tensor(1800.0))
    return apply_updates(state, updates)


def test_production_functor_emits_no_local_conservation_warning():
    state = initialize_scale_state(build_state())
    functor = MethaneProductionFunctor()
    dt = torch.tensor(1800.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        updates = functor(state, time=0.0, dt=dt)

    assert collect_conservation_messages(records) == []
    assert "ch4_storage_tendency_prod" in updates


def test_oxidation_functor_emits_no_local_conservation_warning():
    state = initialize_scale_state(build_state())
    functor = MethaneOxidationFunctor()
    dt = torch.tensor(1800.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        updates = functor(state, time=0.0, dt=dt)

    assert collect_conservation_messages(records) == []
    assert set(updates) == {
        "ch4_oxidation_rate",
        "ch4_storage_tendency_oxid",
        "diagnostics.methane_oxidation.kinetic_rate",
        "diagnostics.methane_oxidation.storage_limiter_residual",
    }
    assert torch.allclose(
        updates["ch4_oxidation_rate"].value
        + updates["ch4_storage_tendency_oxid"].value,
        torch.zeros_like(updates["ch4_oxidation_rate"].value),
        atol=0.0,
        rtol=0.0,
    )


def test_transport_functor_emits_no_local_conservation_warning():
    state = build_state()
    functor = MethaneTransportFunctor()
    dt = torch.tensor(1800.0)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        updates = functor(state, time=0.0, dt=dt)

    assert collect_conservation_messages(records) == []
    assert set(updates) == {
        "ch4_transport_loss_rate",
        "ch4_storage_tendency_trans",
        "diagnostics.methane_transport.diffusion_rate",
        "diagnostics.methane_transport.ebullition_rate",
        "diagnostics.methane_transport.plant_flux_rate",
    }


def test_full_methane_chain_emits_no_local_conservation_warning():
    state = build_state()
    dt = torch.tensor(1800.0)
    functors = [
        MethaneScaleCouplingFunctor(),
        MethaneProductionFunctor(),
        MethaneOxidationFunctor(),
        MethaneTransportFunctor(),
        MethaneBudgetCoupler(),
    ]

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        for functor in functors:
            updates = functor(state, time=0.0, dt=dt)
            apply_updates(state, updates)

    assert collect_conservation_messages(records) == []
    assert (
        state["diagnostics.methane_budget.mass_balance_residual"]
        .abs()
        .max()
        .item()
        < 5e-4
    )
    for removed_name in (
        "ch4_emission",
        "ch4_emission_mass",
        "ch4_mass_balance_residual",
        "ch4_storage_tendency_net",
        "ch4_soil_storage_delta",
    ):
        assert removed_name not in state
