import torch
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


def run_step(state, dt):
    dt_tensor = torch.tensor(float(dt))
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


def build_state(wetland_fraction, wfps, data_score, temperature):
    return {
        "soil_temperature": torch.tensor(temperature),
        "soil_ph": torch.tensor(6.9),
        "heterotrophic_respiration": torch.tensor(5.5),
        "redox_potential": torch.tensor(-190.0),
        "soil_co2_concentration": torch.tensor(0.18),
        "microbial_activity_index": torch.tensor(0.95),
        "ch4_soil_storage": torch.tensor(6.5),
        "land_surface.wetland_fraction": torch.tensor(wetland_fraction),
        "water_filled_pore_space": torch.tensor(wfps),
        "methane_data_score": torch.tensor(data_score),
        "root_length_density": torch.tensor(0.2),
        "soil_o2_concentration": torch.tensor(0.02),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
    }


def test_hybrid_gating_varies_with_wetness():
    dry_state = build_state(0.1, 0.3, 0.2, 291.0)
    wet_state = build_state(0.95, 0.92, 1.0, 295.0)

    run_step(dry_state, 3600.0)
    run_step(wet_state, 3600.0)

    gating_dry = dry_state["methane_gating_weight"].item()
    gating_wet = wet_state["methane_gating_weight"].item()
    production_dry = dry_state["ch4_production_rate"].item()
    production_wet = wet_state["ch4_production_rate"].item()

    assert gating_dry < gating_wet
    assert production_wet >= production_dry
    assert 0.0 <= gating_dry <= 1.0
    assert 0.0 <= gating_wet <= 1.0
    for state in (dry_state, wet_state):
        assert "ch4_oxidation_rate" in state
        assert "ch4_storage_tendency_oxid" in state
        assert "diagnostics.methane_oxidation.kinetic_rate" in state
        assert "diagnostics.methane_oxidation.imaple_rate" not in state
        assert "diagnostics.methane_oxidation.calibrated_rate" not in state
        assert "diagnostics.methane_oxidation.dndc_rate" not in state
        assert "ch4_oxidation_imaple_rate" not in state
        assert "ch4_oxidation_dndc_rate" not in state
        assert "ch4_soil_storage_delta" not in state
        torch.testing.assert_close(
            state["ch4_oxidation_rate"] + state["ch4_storage_tendency_oxid"],
            torch.zeros_like(state["ch4_oxidation_rate"]),
            atol=0.0,
            rtol=0.0,
        )

