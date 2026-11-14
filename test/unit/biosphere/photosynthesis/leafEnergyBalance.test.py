"""Leaf energy balance (LeafEnergyBalance), law level.

References: Monteith (1965); Allen et al. (1998) FAO-56 Eqs. 3, 4, 8, 11-13 and 42; Jones
(2014) Plants and Microclimate, Chapter 9 (linearised leaf temperature with r_HR = r_aW).

Checked here:
- dT equals an independent float64 evaluation of the RESISTANCE form
  dT = [r_a (r_a + r_c) gamma A / (rho c_p) - r_a D] / [gamma (r_a + r_c) + s r_a]
  written in this file from the SSOT constants and the parameter values (no RELIC helper);
- energy closes: A = H + lambda E with lambda E the Penman-Monteith latent heat of FAO-56
  Eq. 3 evaluated independently;
- limits: g_c = 0 gives the dry leaf A / (rho c_p g_a); a very large g_c gives the wet
  surface [gamma A / (rho c_p g_a) - D] / (s + gamma), whose A = 0 value is close to the
  psychrometric wet-bulb depression solved without linearisation; LAI = 0 gives T_air
  (and T_max) bitwise;
- g_a is bitwise the conductance of PenmanMonteithET (one owner);
- domain checks raise and nothing is clipped.
"""

from __future__ import annotations

import math

import pytest
import torch

from relic.parameters import get_global_manager
from relic.scientific.biosphere.photosynthesis.leaf_energy_balance import LeafEnergyBalance
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.utils.physical_constants import get_physical_constant

DAY = 86400.0
P = "diagnostics.leaf_energy_balance."


def _c(key):
    return float(get_physical_constant(key))


def _p(key):
    value = get_global_manager().get(key)
    return value


def _state(gc, *, rn=150.0, t=293.15, tmax=300.15, tmin=286.15, vpd=1000.0, u=2.0, p=101325.0, lai=3.0,
           dtype=torch.float64):
    gc = torch.as_tensor(gc, dtype=dtype).reshape(1, -1)
    n = gc.shape[1]

    def full(v):
        return torch.as_tensor(v, dtype=dtype).expand(1, n).clone() if not torch.is_tensor(v) else v.to(dtype).reshape(1, -1)

    return {"air_temperature": full(t), "air_temperature_max": full(tmax), "air_temperature_min": full(tmin),
            "vapor_pressure_deficit": full(vpd), "wind_speed": full(u), "atmospheric_pressure": full(p),
            "net_radiation": full(rn), "canopy_conductance": gc, "leaf_area_index": full(lai)}


def _run(state, support="daily", dt=DAY):
    model = LeafEnergyBalance(config={"temporal_support": support})
    with torch.no_grad():
        out = model.forward(dict(state), 0.0, dt)
    return {k: (v.value if hasattr(v, "value") else v) for k, v in out.items()}


def _independent(gc, *, rn, t, tmax, tmin, vpd, u, p, g=0.0, daily=True):
    """Float64 resistance form of the linearised balance, written without RELIC code."""
    e0 = _c("relic.physical.constants.psychrometrics.tetens_base_pressure")
    a = _c("relic.physical.constants.psychrometrics.tetens_coefficient_a")
    b = _c("relic.physical.constants.psychrometrics.tetens_coefficient_b")
    c = _c("relic.physical.constants.psychrometrics.saturation_vapor_pressure_slope_coeff")
    k0 = _c("celsius_to_kelvin")
    lam = _c("latent_heat_vaporization")
    cp = _c("air_specific_heat_cp")
    eps = _c("water_molecular_weight") / _c("air_molecular_weight")

    def es(temp):
        tc = temp - k0
        return e0 * math.exp(a * tc / (tc + b))

    tc = t - k0
    s = c * es(t) / (tc + b) ** 2
    gamma = cp * p / (eps * lam)
    e_s = 0.5 * (es(tmax) + es(tmin)) if daily else es(t)
    e_a = max(e_s - vpd, 0.0)
    # Moist-air density rho = (p - (1 - eps) e_a) / (R_d T), R_d = R / M_d (SSOT units g mol-1 -> kg mol-1).
    r_gas = _c("relic.physical.constants.universal.gas_constant")
    m_d = _c("relic.physical.constants.air.molecular_weight_dry") / 1000.0
    rho = (p - (1.0 - eps) * e_a) / (r_gas / m_d * t)
    # FAO-56 Eq. 4 neutral profile, written here.
    z = float(_p("hydrosphere.evapotranspiration.reference_height"))
    d = float(_p("hydrosphere.evapotranspiration.displacement_height"))
    z0m = float(_p("hydrosphere.evapotranspiration.roughness_length"))
    z0h = z0m * float(_p("hydrosphere.physical.heat_to_momentum_roughness_ratio"))
    u_min = float(_p("hydrosphere.numerical.minimum_wind_speed"))
    lo, hi = (float(v) for v in _p("hydrosphere.numerical.aerodynamic_conductance_bounds"))
    kappa = _c("von_karman_constant")
    ga = min(max(kappa ** 2 * max(u, u_min) / (math.log((z - d) / z0m) * math.log((z - d) / z0h)), lo), hi)
    big_a = rn - g
    ra = 1.0 / ga
    if gc == 0.0:
        dt_ = ra * big_a / (rho * cp)
    else:
        rc = 1.0 / gc
        dt_ = (ra * (ra + rc) * gamma * big_a / (rho * cp) - ra * vpd) / (gamma * (ra + rc) + s * ra)
    le_pm = 0.0 if gc == 0.0 else (s * big_a + rho * cp * vpd * ga) / (s + gamma * (1.0 + ga / gc))
    return {"dT": dt_, "H": rho * cp * ga * dt_, "LE_pm": le_pm, "A": big_a, "s": s, "gamma": gamma, "rho": rho,
            "ga": ga, "e_a": e_a}


CASES = [dict(gc=g, rn=rn, vpd=vpd, u=u) for g in (0.0, 1e-4, 2e-3, 1e-2, 5e-2)
         for rn, vpd, u in ((150.0, 1000.0, 2.0), (-40.0, 300.0, 0.5), (400.0, 2500.0, 5.0))]


@pytest.mark.parametrize("case", CASES, ids=[f"gc{c['gc']}_rn{c['rn']}" for c in CASES])
def test_matches_independent_float64_resistance_form(case):
    out = _run(_state(case["gc"], rn=case["rn"], vpd=case["vpd"], u=case["u"]))
    ref = _independent(case["gc"], rn=case["rn"], t=293.15, tmax=300.15, tmin=286.15, vpd=case["vpd"], u=case["u"], p=101325.0)
    dt_value = float(out[P + "leaf_air_temperature_difference"])
    assert dt_value == pytest.approx(ref["dT"], rel=1e-6, abs=1e-6)
    assert float(out["leaf_temperature"]) == pytest.approx(293.15 + ref["dT"], rel=0, abs=1e-6)
    assert float(out["leaf_temperature_max"]) == pytest.approx(300.15 + ref["dT"], rel=0, abs=1e-6)
    assert float(out[P + "sensible_heat_flux"]) == pytest.approx(ref["H"], rel=1e-6, abs=1e-5)
    assert float(out[P + "aerodynamic_conductance"]) == pytest.approx(ref["ga"], rel=1e-6)


@pytest.mark.parametrize("case", [c for c in CASES if c["gc"] > 0], ids=[f"gc{c['gc']}_rn{c['rn']}" for c in CASES if c["gc"] > 0])
def test_energy_closes_with_the_penman_monteith_latent_heat(case):
    out = _run(_state(case["gc"], rn=case["rn"], vpd=case["vpd"], u=case["u"]))
    ref = _independent(case["gc"], rn=case["rn"], t=293.15, tmax=300.15, tmin=286.15, vpd=case["vpd"], u=case["u"], p=101325.0)
    h = float(out[P + "sensible_heat_flux"])
    le = float(out[P + "latent_heat_flux"])
    # Independent float64 lambda E against the Functor H: the parameters are held in
    # float32 by the ParameterManager, so agreement is to about 1e-7 relative.
    assert h + ref["LE_pm"] == pytest.approx(ref["A"], rel=1e-6, abs=1e-4)
    assert le == pytest.approx(ref["LE_pm"], rel=1e-6, abs=1e-4)
    assert le + h == pytest.approx(float(out[P + "available_energy"]), rel=1e-12, abs=1e-9)
    assert float(out[P + "available_energy"]) == case["rn"]  # daily G = 0 (FAO-56 Eq. 42)


def test_instantaneous_support_reads_the_soil_heat_flux():
    state = _state(0.01, rn=300.0)
    for key in ("air_temperature_max", "air_temperature_min"):
        state.pop(key)
    state["soil_heat_flux"] = torch.tensor([[45.0]], dtype=torch.float64)
    out = _run(state, support="instantaneous", dt=1800.0)
    ref = _independent(0.01, rn=300.0, t=293.15, tmax=0.0, tmin=0.0, vpd=1000.0, u=2.0, p=101325.0, g=45.0, daily=False)
    assert float(out[P + "available_energy"]) == 255.0
    assert float(out[P + "leaf_air_temperature_difference"]) == pytest.approx(ref["dT"], rel=1e-6, abs=1e-6)
    assert "leaf_temperature_max" not in out


def test_dry_leaf_limit_at_zero_canopy_conductance():
    out = _run(_state([0.0], rn=150.0))
    ref = _independent(0.0, rn=150.0, t=293.15, tmax=300.15, tmin=286.15, vpd=1000.0, u=2.0, p=101325.0)
    dry = ref["A"] / (ref["rho"] * _c("air_specific_heat_cp") * ref["ga"])
    # Float32-held parameters: agreement to about 1e-7 relative.
    assert float(out[P + "leaf_air_temperature_difference"]) == pytest.approx(dry, rel=1e-6)
    assert float(out[P + "sensible_heat_flux"]) == pytest.approx(150.0, rel=1e-9)
    assert float(out[P + "latent_heat_flux"]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("rn", [0.0, 150.0])
def test_wet_surface_limit_at_large_canopy_conductance(rn):
    out = _run(_state([1.0e4], rn=rn))
    ref = _independent(1.0e4, rn=rn, t=293.15, tmax=300.15, tmin=286.15, vpd=1000.0, u=2.0, p=101325.0)
    cp = _c("air_specific_heat_cp")
    wet = (ref["gamma"] * ref["A"] / (ref["rho"] * cp * ref["ga"]) - 1000.0) / (ref["s"] + ref["gamma"])
    # g_a / g_c ~ 1e-5: the limit is reached to that relative order.
    assert float(out[P + "leaf_air_temperature_difference"]) == pytest.approx(wet, rel=1e-4, abs=1e-4)
    if rn == 0.0:
        # Psychrometric wet-bulb temperature without linearisation: e_s(T_w) - gamma (T - T_w) = e_a
        # with e_a = e_s(T) - D, solved by bisection; the linearised -D / (s + gamma) differs
        # from it by the neglected second-order term 0.5 s' dT^2 / (s + gamma) (about 0.47 K
        # at D = 1 kPa, 20 degC: the linearisation underestimates the depression).
        e0 = _c("relic.physical.constants.psychrometrics.tetens_base_pressure")
        a = _c("relic.physical.constants.psychrometrics.tetens_coefficient_a")
        b = _c("relic.physical.constants.psychrometrics.tetens_coefficient_b")
        k0 = _c("celsius_to_kelvin")

        def es(temp):
            return e0 * math.exp(a * (temp - k0) / (temp - k0 + b))

        e_a = es(293.15) - 1000.0
        lo, hi = 273.15, 293.15
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if es(mid) - ref["gamma"] * (293.15 - mid) > e_a:
                hi = mid
            else:
                lo = mid
        wet_bulb_depression = 0.5 * (lo + hi) - 293.15
        assert wet < 0.0 and wet_bulb_depression < wet
        tc = 293.15 - k0
        c = _c("relic.physical.constants.psychrometrics.saturation_vapor_pressure_slope_coeff")

        def slope(temp):
            return c * es(temp) / (temp - k0 + b) ** 2

        curvature = (slope(293.15 + 0.01) - slope(293.15 - 0.01)) / 0.02
        second_order = 0.5 * curvature * wet ** 2 / (ref["s"] + ref["gamma"])
        assert tc > 0.0 and 0.3 < second_order < 0.6
        assert abs(wet - wet_bulb_depression) == pytest.approx(second_order, rel=0.15)


def test_drying_lowers_conductance_and_raises_the_leaf_air_difference():
    gc = torch.tensor([[2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 1e-4, 0.0]], dtype=torch.float64)
    out = _run(_state(gc))
    difference = out[P + "leaf_air_temperature_difference"].reshape(-1)
    assert bool(torch.all(difference[1:] > difference[:-1]))


def test_no_canopy_gives_air_temperature_bitwise():
    state = _state([5e-3, 0.0, 1e-2], lai=torch.tensor([[3.0, 0.0, 0.0]]), dtype=torch.float32)
    out = _run(state)
    assert torch.equal(out["leaf_temperature"][:, 1:], state["air_temperature"][:, 1:])
    assert torch.equal(out["leaf_temperature_max"][:, 1:], state["air_temperature_max"][:, 1:])
    assert torch.equal(out[P + "no_canopy"], torch.tensor([[False, True, True]]))
    for key in ("leaf_air_temperature_difference", "sensible_heat_flux", "latent_heat_flux"):
        assert bool(torch.all(out[P + key][:, 1:] == 0.0))
    assert float(out[P + "leaf_air_temperature_difference"][0, 0]) != 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aerodynamic_conductance_is_bitwise_that_of_penman_monteith(dtype):
    wind = torch.tensor([[0.0, 0.05, 1.0, 2.0, 7.5, 40.0]], dtype=dtype)
    state = _state([1e-2] * 6, u=wind, dtype=dtype)
    out = _run(state)
    pm = PenmanMonteithET(config={"canopy_resistance_source": "state", "temporal_support": "daily"})
    expected = pm._compute_aerodynamic_conductance(wind, state["air_temperature"])
    assert torch.equal(out[P + "aerodynamic_conductance"], expected)


def test_vapor_pressure_floor_is_flagged_not_silent():
    # D above the Eq. 12 saturation pressure: e_a < 0 is replaced by 0 in rho and flagged.
    out = _run(_state([1e-2, 1e-2], vpd=torch.tensor([[1000.0, 9000.0]])))
    assert torch.equal(out[P + "vapor_pressure_floor_active"], torch.tensor([[False, True]]))


@pytest.mark.parametrize("change, message", [
    ({"canopy_conductance": -1e-3}, "canopy_conductance"),
    ({"vapor_pressure_deficit": -1.0}, "vapor_pressure_deficit"),
    ({"leaf_area_index": -0.1}, "leaf_area_index"),
    ({"wind_speed": -0.1}, "wind_speed"),
    ({"air_temperature_min": 295.0}, "air_temperature_min"),
    ({"air_temperature_max": 290.0}, "air_temperature_min <= air_temperature <= air_temperature_max"),
    ({"net_radiation": float("nan")}, "net_radiation"),
])
def test_domain_violations_raise(change, message):
    state = _state([1e-2])
    for key, value in change.items():
        state[key] = torch.tensor([[value]], dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        _run(state)


def test_daily_support_requires_a_daily_step_and_configuration_is_explicit():
    with pytest.raises(ValueError, match="dt="):
        _run(_state([1e-2]), dt=3600.0)
    with pytest.raises(ValueError, match="temporal_support"):
        LeafEnergyBalance()
    with pytest.raises(ValueError, match="temporal_support"):
        LeafEnergyBalance(config={"temporal_support": "hourly"})


def test_declared_ports_per_support():
    daily = LeafEnergyBalance(config={"temporal_support": "daily"})
    inst = LeafEnergyBalance(config={"temporal_support": "instantaneous"})
    base = ["air_temperature", "vapor_pressure_deficit", "wind_speed", "atmospheric_pressure", "net_radiation",
            "canopy_conductance", "leaf_area_index"]
    assert daily.metadata.input_variables == base + ["air_temperature_max", "air_temperature_min"]
    assert inst.metadata.input_variables == base + ["soil_heat_flux"]
    assert daily.metadata.output_variables[0] == "leaf_temperature"
    assert daily.metadata.output_variables[-1] == "leaf_temperature_max"
    assert "leaf_temperature_max" not in inst.metadata.output_variables
    assert daily.get_dependencies() == []
    out = LeafEnergyBalance(config={"temporal_support": "daily"}).forward(_state([1e-2]), 0.0, DAY)
    assert (out["leaf_temperature"].mode.name, out["leaf_temperature"].unit, out["leaf_temperature"].per_timestep) == ("REPLACE", "K", False)
