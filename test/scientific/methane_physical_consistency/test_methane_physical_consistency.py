"""Physical consistency checks of the wetland methane chain.

Replaces the former synthetic "global map total" check. A 0.1 Mkm2 synthetic map
run for one day cannot test a global wetland total, so this file checks
(a) the single-cell iMAPLE production against a hand evaluation of Yue et al.
(2024) Eq. 34, (b) the carbon-budget invariant P_iMAPLE <= r Rh <= Rh, and
(c) the order of magnitude of the per-area emission of a site-like
configuration against the FLUXNET-CH4 site extremes of Yue et al. (2024),
Table S3.
"""

import math
import os

# Allow OpenMP runtimes from MKL and PyTorch to coexist under the test harness.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
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

SECONDS_PER_HOUR = float(get_physical_constant("seconds_per_hour"))
SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))
SECONDS_PER_YEAR = float(get_physical_constant("seconds_per_year"))

# Yue et al. (2024), Supplement Table S3: extremes of the 44 FLUXNET-CH4 site
# ranges, US-Ivo minimum 0.02 and JP-SwL maximum 268.37 g CH4 m-2 yr-1.
FLUXNET_CH4_SITE_MINIMUM = 0.02
FLUXNET_CH4_SITE_MAXIMUM = 268.37


def _as_float(value):
    return MethaneProductionFunctor._scalar_from_value(value)


def _production_state(temperature, soil_ph, wetland, rh, weight=0.0):
    return {
        "soil_temperature": torch.tensor(temperature, dtype=torch.float64),
        "soil_ph": torch.tensor(soil_ph, dtype=torch.float64),
        "heterotrophic_respiration": torch.tensor(rh, dtype=torch.float64),
        "redox_potential": torch.tensor(-250.0, dtype=torch.float64),
        "microbial_activity_index": torch.tensor(0.7, dtype=torch.float64),
        "land_surface.wetland_fraction": torch.tensor(wetland, dtype=torch.float64),
        "methane_gating_weight": torch.tensor(weight, dtype=torch.float64),
    }


def _hand_temperature_factor(production, temperature):
    # References: Yue et al. (2024), Eqs. 35-39; Zhu et al. (2014), Eqs. 3-5 and Table 2.
    # Mathematical expression: Q10=clip(r_b Q_b^((T-T_base)/10), Q10_min, Q10_max);
    #   v=(Tmax-T)/(Tmax-Topt); L=ln(Q10)(Tmax-Topt);
    #   a=1+40/L; x=L^2(1+sqrt(a))^2/400; f_T=v^x exp(x(1-v)) on [Tmin,Tmax].
    # Parameter meanings: Tmin, Topt, Tmax are the published 0, 25 and 45 C
    #   limits; r_b, Q_b, T_base and the 1.7-16 clip are the functor's Eq. 39 values.
    # Units: temperatures [K] (differences equal C differences); Q10, f_T [1].
    # Domain/constraints: f_T = 0 outside [Tmin, Tmax].
    # Sign convention: f_T is a nonnegative multiplier.
    t_min = _as_float(production.temperature_min)
    t_opt = _as_float(production.temperature_optimum)
    t_max = _as_float(production.temperature_max)
    if temperature < t_min or temperature > t_max:
        return 0.0
    q10 = _as_float(production.q10) * _as_float(production.q10_sensitivity) ** (
        (temperature - _as_float(production.q10_base_temperature)) / 10.0
    )
    q10 = min(max(q10, _as_float(production.q10_min)), _as_float(production.q10_max))
    span = math.log(q10) * (t_max - t_opt)
    a_t = 1.0 + 40.0 / span
    x_t = span**2 * (1.0 + math.sqrt(a_t)) ** 2 / 400.0
    v_t = (t_max - temperature) / (t_max - t_opt)
    return v_t**x_t * math.exp(x_t * (1.0 - v_t))


def _hand_ph_factor(production, soil_ph):
    # References: Zhu et al. (2014), Eq. 6.
    # Mathematical expression: f_pH=((pH-pHlow)/(pHopt-pHlow))
    #   * ((pHhigh-pH)/(pHhigh-pHopt))^((pHhigh-pHopt)/(pHopt-pHlow)) on [pHlow,pHhigh].
    # Parameter meanings: pHlow=4, pHopt=7, pHhigh=9.
    # Units: all pH quantities and f_pH [1].
    # Domain/constraints: f_pH = 0 outside [pHlow, pHhigh].
    # Sign convention: f_pH is a nonnegative multiplier.
    low = _as_float(production.ph_lower_limit)
    opt = _as_float(production.ph_optimum)
    high = _as_float(production.ph_upper_limit)
    if soil_ph < low or soil_ph > high:
        return 0.0
    return ((soil_ph - low) / (opt - low)) * ((high - soil_ph) / (high - opt)) ** (
        (high - opt) / (opt - low)
    )


@pytest.mark.parametrize(
    "temperature,soil_ph,wetland",
    [
        (275.15, 8.0, 0.2),
        (283.15, 6.0, 0.4),
        (293.15, 7.0, 0.95),
        (298.15, 7.0, 1.0),
        (305.15, 5.0, 0.7),
        (315.15, 8.5, 0.03),
    ],
)
def test_single_cell_production_matches_a_hand_evaluation_of_yue_equation_34(
    temperature, soil_ph, wetland
):
    # References: Yue et al. (2024), Eq. 34.
    # Mathematical expression: P = Rh * r * f_T(Ts) * f_pH(pH) * f_wetland.
    # Parameter meanings: Rh heterotrophic respiration; r CH4:CO2 release ratio.
    # Units: Rh, P [g C m-2 s-1]; r and factors [1].
    # Domain/constraints: w = 0 selects the iMAPLE branch alone.
    # Sign convention: P is a positive methane-carbon source.
    production = MethaneProductionFunctor()
    rh = 2.0 / SECONDS_PER_DAY
    updates = production(
        _production_state(temperature, soil_ph, wetland, rh),
        time=0.0,
        dt=torch.tensor(SECONDS_PER_HOUR),
    )

    expected = (
        rh
        * _as_float(production.release_ratio)
        * _hand_temperature_factor(production, temperature)
        * _hand_ph_factor(production, soil_ph)
        * wetland
    )
    assert expected > 0.0
    for key in ("diagnostics.methane_production.imaple_rate", "ch4_production_rate"):
        # Parameters are float32 ParameterManager tensors, so agreement is to float32 precision.
        assert updates[key].value.item() == pytest.approx(expected, rel=1.0e-6, abs=0.0)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_both_production_branches_obey_the_carbon_budget_invariant(weight):
    # Rounds 50.3 and 51.1: every factor of Eq. 34 (weight 0) and of the
    # calibrated branch Rh r f_T f_Eh A (weight 1) lies in [0, 1] on its domain,
    # so the methane carbon produced cannot exceed r Rh, and r <= 1 keeps it
    # below Rh. A = 1 and Eh = -250 mV give the largest calibrated factors.
    production = MethaneProductionFunctor()
    release_ratio = _as_float(production.release_ratio)
    assert 0.0 < release_ratio <= 1.0
    for temperature in (273.15, 283.15, 298.15, 310.15, 318.15):
        for soil_ph in (4.0, 5.5, 7.0, 8.5, 9.0):
            for wetland in (0.0, 0.3, 1.0):
                for rh in (1.0e-6, 1.0e-4):
                    state = _production_state(temperature, soil_ph, wetland, rh, weight)
                    state["microbial_activity_index"] = torch.tensor(1.0, dtype=torch.float64)
                    rate = production(
                        state,
                        time=0.0,
                        dt=torch.tensor(SECONDS_PER_HOUR),
                    )["ch4_production_rate"].value.item()
                    assert 0.0 <= rate <= release_ratio * rh * (1.0 + 1.0e-12) <= rh


def _apply_updates(state, updates):
    for key, value in updates.items():
        raw = value.value if isinstance(value, UpdateItem) else value
        mode = value.mode if isinstance(value, UpdateItem) else UpdateMode.ACCUMULATE
        tensor = raw if isinstance(raw, torch.Tensor) else torch.tensor(float(raw))
        tensor = tensor.to(torch.float64)
        if key in state and mode != UpdateMode.REPLACE:
            state[key] = state[key] + tensor
        else:
            state[key] = tensor.clone()


def test_site_like_annual_emission_lies_within_the_fluxnet_ch4_site_range():
    # Order-of-magnitude check only: a site-like inundated cell with a moderate
    # heterotrophic respiration of 1 g C m-2 d-1 is integrated hourly for 30 days
    # from an empty store, and the mean emission of the last 5 days is annualized.
    functors = (
        MethaneScaleCouplingFunctor(),
        MethaneProductionFunctor(),
        MethaneOxidationFunctor(),
        MethaneTransportFunctor(),
        MethaneBudgetCoupler(),
    )
    as_tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    state = {
        "soil_temperature": as_tensor(293.15),
        "soil_ph": as_tensor(6.5),
        "heterotrophic_respiration": as_tensor(1.0 / SECONDS_PER_DAY),
        "redox_potential": as_tensor(-200.0),
        "microbial_activity_index": as_tensor(0.7),
        "ch4_soil_storage": as_tensor(0.0),
        "land_surface.wetland_fraction": as_tensor(1.0),
        "water_filled_pore_space": as_tensor(0.95),
        "methane_data_score": as_tensor(0.7),
        "soil_o2_concentration": as_tensor(0.005),
        "porosity": as_tensor(0.45),
        "atmospheric_ch4_concentration": as_tensor(0.0),
    }
    dt = torch.tensor(SECONDS_PER_HOUR, dtype=torch.float64)
    steps = int(30 * SECONDS_PER_DAY / SECONDS_PER_HOUR)
    averaged_steps = int(5 * SECONDS_PER_DAY / SECONDS_PER_HOUR)
    emissions = []
    for _ in range(steps):
        for functor in functors:
            _apply_updates(state, functor(state, time=0.0, dt=dt))
        emissions.append(
            state["diagnostics.methane_budget.atmospheric_ch4_mass_flux"].item()
        )

    annual_emission = sum(emissions[-averaged_steps:]) / averaged_steps * SECONDS_PER_YEAR
    assert FLUXNET_CH4_SITE_MINIMUM <= annual_emission <= FLUXNET_CH4_SITE_MAXIMUM
