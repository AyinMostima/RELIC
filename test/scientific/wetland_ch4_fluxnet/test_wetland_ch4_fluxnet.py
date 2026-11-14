import os

# Allow OpenMP runtimes from MKL and PyTorch to coexist under the test harness.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from relic.scientific.pedosphere.methane import (
    MethaneBudgetCoupler,
    MethaneOxidationFunctor,
    MethaneProductionFunctor,
    MethaneScaleCouplingFunctor,
    MethaneTransportFunctor,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant


SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))
SECONDS_PER_HOUR = float(get_physical_constant("seconds_per_hour"))


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
    for functor in (
        MethaneScaleCouplingFunctor(),
        MethaneProductionFunctor(),
        MethaneOxidationFunctor(),
        MethaneTransportFunctor(),
        MethaneBudgetCoupler(),
    ):
        updates = functor(state, time=0.0, dt=dt_tensor)
        apply_updates(state, updates)
    return state


def build_state(wetland_fraction, wfps, temperature, ph):
    return {
        "soil_temperature": torch.tensor(temperature),
        "soil_ph": torch.tensor(ph),
        "heterotrophic_respiration": torch.tensor(5.5 / float(SECONDS_PER_DAY)),
        "redox_potential": torch.tensor(-200.0),
        "microbial_activity_index": torch.tensor(1.0),
        "ch4_soil_storage": torch.tensor(7.0),
        "land_surface.wetland_fraction": torch.tensor(wetland_fraction),
        "water_filled_pore_space": torch.tensor(wfps),
        "methane_data_score": torch.tensor(0.7),
        "soil_o2_concentration": torch.tensor(0.005),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
    }


def test_site_level_emission_preserves_a_perturbed_synthetic_signal():
    steps = int(SECONDS_PER_DAY / SECONDS_PER_HOUR)
    dt = SECONDS_PER_HOUR
    simulated = []
    state = build_state(0.7, 0.75, 292.0, 6.9)
    for hour in range(steps):
        # introduce diurnal variation in forcing
        state["soil_temperature"] = torch.tensor(292.0 + 2.0 * np.sin(hour / 4.0))
        state["soil_ph"] = torch.tensor(6.8 + 0.1 * np.cos(hour / 6.0))
        run_methane_step(state, dt)
        for key in (
            "ch4_production_rate",
            "ch4_storage_tendency_prod",
            "diagnostics.methane_production.imaple_rate",
            "diagnostics.methane_production.calibrated_rate",
            "ch4_oxidation_rate",
            "ch4_storage_tendency_oxid",
            "diagnostics.methane_oxidation.kinetic_rate",
            "diagnostics.methane_oxidation.storage_limiter_residual",
        ):
            assert key in state
        assert state["ch4_oxidation_rate"].item() >= 0.0
        torch.testing.assert_close(
            state["ch4_oxidation_rate"] + state["ch4_storage_tendency_oxid"],
            torch.zeros_like(state["ch4_oxidation_rate"]),
            atol=0.0,
            rtol=0.0,
        )
        simulated.append(state["ch4_atmospheric_flux"].item())

    for removed_name in (
        "ch4_emission",
        "ch4_emission_mass",
        "ch4_mass_balance_residual",
        "ch4_storage_tendency_net",
        "ch4_soil_storage_delta",
    ):
        assert removed_name not in state

    simulated = np.array(simulated)
    signal_span = max(float(np.ptp(simulated)), 1.0e-8)
    perturbed_signal = simulated * 0.92 + np.linspace(
        -0.02 * signal_span,
        0.02 * signal_span,
        steps,
    )
    perturbed_signal += np.random.default_rng(42).normal(
        0.0,
        0.01 * signal_span,
        size=steps,
    )

    corr = np.corrcoef(simulated, perturbed_signal)[0, 1]
    assert simulated.std() > 1.0e-6
    assert corr > 0.95

