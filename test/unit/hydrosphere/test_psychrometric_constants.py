import pytest

from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    PenmanMonteithET,
)
from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import (
    PriestleyTaylorET,
)
from relic.utils.physical_constants import get_physical_constant


def _as_float(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
    except Exception:
        pass
    return float(value)


def test_psychrometric_constants_from_physical_namespace():
    base_pressure = _as_float(
        get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_base_pressure"
        )
    )
    coeff_a = _as_float(
        get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_coefficient_a"
        )
    )
    coeff_b = _as_float(
        get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_coefficient_b"
        )
    )
    slope_coeff = _as_float(
        get_physical_constant(
            "relic.physical.constants.psychrometrics."
            "saturation_vapor_pressure_slope_coeff"
        )
    )

    pt = PriestleyTaylorET()
    assert _as_float(pt.tetens_base_pressure) == pytest.approx(base_pressure)
    assert _as_float(pt.tetens_coeff_a) == pytest.approx(coeff_a)
    assert _as_float(pt.tetens_coeff_b) == pytest.approx(coeff_b)
    assert _as_float(pt.slope_coefficient) == pytest.approx(slope_coeff)

    pm_et = PenmanMonteithET(config={"canopy_resistance_source": "state"})
    assert _as_float(pm_et._tetens_base_pressure) == pytest.approx(base_pressure)
    assert _as_float(pm_et._tetens_coeff_a) == pytest.approx(coeff_a)
    assert _as_float(pm_et._tetens_coeff_b) == pytest.approx(coeff_b)
    assert _as_float(pm_et._tetens_slope_coeff) == pytest.approx(slope_coeff)


def test_saturation_slope_coefficient_has_temperature_interval_units():
    from relic.data.standards.dimensionality import dimensionality_from_unit
    from relic.parameters import get_global_manager

    key = "relic.physical.constants.psychrometrics.saturation_vapor_pressure_slope_coeff"
    coefficient = _as_float(get_physical_constant(key))
    manager = get_global_manager()
    spec = manager.get_parameter(key)
    assert spec is not None
    # References: Allen et al. (1998), FAO-56, equation 13.
    # Mathematical expression: D = c * es / (Tc + b)^2;
    #   [D] = [c] * Pa / K^2 = Pa / K, hence [c] = K.
    # Parameter meanings: c slope coefficient; es saturation vapor pressure;
    #   Tc Celsius temperature; b Tetens temperature offset; D saturation slope.
    # Units: c,Tc+b [K interval]; es [Pa]; D [Pa K-1].
    # Domain/constraints: the original Tetens formulation and constants; temperature
    #   differences use kelvin without an affine absolute-temperature conversion.
    # Sign convention: es,c and D are nonnegative properties, not signed fluxes.
    resulting_dimension = dimensionality_from_unit(f"{spec.units} Pa K-2")
    assert resulting_dimension == dimensionality_from_unit("Pa K-1")
    assert spec.units == "K"
    assert spec.dimension == "temperature"
    assert coefficient == 4098.0
    assert _as_float(manager.get(key, units="K", required=True)) == 4098.0


_ICE_A_KEY = "relic.physical.constants.psychrometrics.tetens_ice_coefficient_a"
_ICE_B_KEY = "relic.physical.constants.psychrometrics.tetens_ice_coefficient_b"


def test_ice_tetens_coefficients_follow_murray_and_match_goff_gratch():
    import math

    from relic.parameters import get_global_manager

    manager = get_global_manager()
    base_pressure = _as_float(
        get_physical_constant("relic.physical.constants.psychrometrics.tetens_base_pressure")
    )
    kelvin_offset = _as_float(get_physical_constant("celsius_to_kelvin"))
    coeff_a = _as_float(get_physical_constant(_ICE_A_KEY))
    coeff_b = _as_float(get_physical_constant(_ICE_B_KEY))
    # References: Murray (1967), J. Appl. Meteor. 6, 203-204 (ice: a = 21.8745584,
    #   T - 7.66 K with reference 273.16 K); Goff and Gratch (1946) as tabulated in the
    #   Smithsonian Meteorological Tables (independent reference over ice).
    # Mathematical expression: e_i = e0 * exp(a_i * Tc / (Tc + b_i)), Tc = T - 273.15;
    #   log10(e_GG / hPa) = -9.09718 (T0/T - 1) - 3.56654 log10(T0/T)
    #   + 0.876793 (1 - T/T0) + log10(6.1071), T0 = 273.16 K.
    # Parameter meanings: e_i Tetens-form saturation vapor pressure over ice; e0 the
    #   base pressure shared with the liquid form; a_i, b_i ice coefficients;
    #   b_i = 273.16 K - 7.66 K expressed in the Celsius convention of the liquid form.
    # Units: e_i, e0, e_GG [Pa]; a_i [1]; b_i, Tc [degC interval]; T, T0 [K].
    # Domain/constraints: 223.15 K <= T <= 273.15 K; both expressions are evaluated at the
    #   same absolute temperature; each bound is just above the measured deviation at that
    #   temperature (9.4e-4, 5.7e-5, 2.75e-3, 7.83e-3, 1.616e-2, 2.88e-2). Writing Murray's
    #   T - 273.16 as Tc = T - 273.15 evaluates his form 0.01 K higher, which raises e_i by
    #   0.08 to 0.13 percent and keeps e_s continuous with the liquid form at Tc = 0.
    # Sign convention: pressures are positive state properties, not signed fluxes.
    assert coeff_a == 21.8745584
    assert coeff_b == 265.5
    assert manager.get_parameter(_ICE_A_KEY).units == "dimensionless"
    assert manager.get_parameter(_ICE_B_KEY).units == "degC"
    for temperature, bound in ((273.15, 1.0e-3), (263.15, 1.0e-4), (253.15, 3.0e-3), (243.15, 8.0e-3), (233.15, 1.7e-2), (223.15, 3.0e-2)):
        reference = 100.0 * 10.0 ** (
            -9.09718 * (273.16 / temperature - 1.0)
            - 3.56654 * math.log10(273.16 / temperature)
            + 0.876793 * (1.0 - temperature / 273.16)
            + math.log10(6.1071)
        )
        celsius = temperature - kelvin_offset
        ice = base_pressure * math.exp(coeff_a * celsius / (celsius + coeff_b))
        assert abs(ice / reference - 1.0) <= bound
    # The liquid form extrapolated below freezing is not a substitute over ice.
    liquid_a = _as_float(get_physical_constant("relic.physical.constants.psychrometrics.tetens_coefficient_a"))
    liquid_b = _as_float(get_physical_constant("relic.physical.constants.psychrometrics.tetens_coefficient_b"))
    for temperature, lower, upper in ((253.15, 0.20, 0.22), (233.15, 0.43, 0.45)):
        reference = 100.0 * 10.0 ** (
            -9.09718 * (273.16 / temperature - 1.0)
            - 3.56654 * math.log10(273.16 / temperature)
            + 0.876793 * (1.0 - temperature / 273.16)
            + math.log10(6.1071)
        )
        celsius = temperature - kelvin_offset
        liquid = base_pressure * math.exp(liquid_a * celsius / (celsius + liquid_b))
        assert lower < liquid / reference - 1.0 < upper
    # At the freezing point the slope ratio ice/liquid follows Clausius-Clapeyron: it equals
    # L_s / L_v(0 degC) to about 0.1 percent (1.132 against 1.134).
    slope_coefficient = _as_float(get_physical_constant(
        "relic.physical.constants.psychrometrics.saturation_vapor_pressure_slope_coeff"))
    slope_ratio = (coeff_a * coeff_b / coeff_b ** 2) / (slope_coefficient / liquid_b ** 2)
    latent_ratio = (
        _as_float(get_physical_constant("relic.physical.constants.water.latent_heat_vaporization_intercept"))
        + _as_float(get_physical_constant("relic.physical.constants.water.latent_heat_fusion"))
    ) / _as_float(get_physical_constant("relic.physical.constants.water.latent_heat_vaporization_intercept"))
    assert slope_ratio == pytest.approx(1.1321, abs=5.0e-4)
    assert abs(slope_ratio / latent_ratio - 1.0) < 2.0e-3


def test_tetens_helpers_evaluate_the_ice_form_with_its_exact_slope():
    import torch

    from relic.scientific.hydrosphere.utils.psychrometrics import (
        compute_saturation_vapor_pressure,
        compute_slope_saturation_vapor_pressure,
    )

    base_pressure = _as_float(
        get_physical_constant("relic.physical.constants.psychrometrics.tetens_base_pressure")
    )
    kelvin_offset = _as_float(get_physical_constant("celsius_to_kelvin"))
    coeff_a = _as_float(get_physical_constant(_ICE_A_KEY))
    coeff_b = _as_float(get_physical_constant(_ICE_B_KEY))
    temperature = torch.tensor([243.15, 258.15, 272.15, 273.15], dtype=torch.float64, requires_grad=True)
    # References: Murray (1967); Clausius-Clapeyron slope of the Tetens form.
    # Mathematical expression: d e_i / dT = a_i * b_i * e_i / (Tc + b_i)^2.
    # Parameter meanings: e_i ice saturation vapor pressure; a_i, b_i ice coefficients;
    #   the slope coefficient of the generic helper is the product a_i * b_i.
    # Units: e_i [Pa]; slope [Pa K-1]; a_i [1]; b_i, Tc [degC interval].
    # Domain/constraints: 243.15 K to 273.15 K; float64; automatic differentiation is
    #   the independent derivative.
    # Sign convention: the slope is positive.
    saturation = compute_saturation_vapor_pressure(
        temperature, tetens_base_pressure=base_pressure, tetens_coeff_a=coeff_a,
        tetens_coeff_b=coeff_b, kelvin_offset=kelvin_offset,
    )
    slope = compute_slope_saturation_vapor_pressure(
        temperature, es_pa=saturation, tetens_coeff_b=coeff_b,
        tetens_slope_coeff=coeff_a * coeff_b, kelvin_offset=kelvin_offset,
    )
    (gradient,) = torch.autograd.grad(saturation.sum(), temperature)
    assert torch.allclose(slope, gradient, rtol=1.0e-12, atol=0.0)
    # Both phases share the base pressure, so the vapor pressure is continuous at 0 degC.
    assert float(saturation[-1].detach()) == pytest.approx(base_pressure, rel=1.0e-15)
