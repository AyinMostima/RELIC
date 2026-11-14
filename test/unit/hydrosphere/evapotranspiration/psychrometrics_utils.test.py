"""Unit tests for hydrosphere psychrometrics utilities.

References:
- Allen, R.G. et al. (1998). FAO-56 Crop Evapotranspiration.
- Tetens (1930) saturation vapor pressure formulation.

Notes:
- ASCII-only strings and comments for cross-platform stability.
"""

import torch

import pytest

from relic.scientific.hydrosphere.utils.psychrometrics import (
    compute_moist_air_density,
    compute_psychrometric_constant,
    compute_saturation_vapor_pressure,
    compute_slope_saturation_vapor_pressure,
)
from relic.utils.physical_constants import get_physical_constant


def _to_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _get_psychrometric_param(key: str) -> float:
    value = get_physical_constant(f"relic.physical.constants.{key}")
    return _to_float(value)


def test_saturation_vapor_pressure_uses_tetens_params() -> None:
    temp_k = torch.tensor(293.15)
    temp_c = temp_k - 273.15
    base_pressure = _get_psychrometric_param("psychrometrics.tetens_base_pressure")
    coeff_a = _get_psychrometric_param("psychrometrics.tetens_coefficient_a")
    coeff_b = _get_psychrometric_param("psychrometrics.tetens_coefficient_b")

    expected = base_pressure * torch.exp(coeff_a * temp_c / (temp_c + coeff_b))
    result = compute_saturation_vapor_pressure(
        temp_k,
        tetens_base_pressure=base_pressure,
        tetens_coeff_a=coeff_a,
        tetens_coeff_b=coeff_b,
    )

    assert torch.allclose(result, expected, rtol=1e-6, atol=1e-8)


def test_slope_saturation_vapor_pressure() -> None:
    temp_k = torch.tensor(293.15)
    temp_c = temp_k - 273.15
    base_pressure = _get_psychrometric_param("psychrometrics.tetens_base_pressure")
    coeff_a = _get_psychrometric_param("psychrometrics.tetens_coefficient_a")
    coeff_b = _get_psychrometric_param("psychrometrics.tetens_coefficient_b")
    slope_coeff = _get_psychrometric_param(
        "psychrometrics.saturation_vapor_pressure_slope_coeff"
    )

    es_pa = base_pressure * torch.exp(coeff_a * temp_c / (temp_c + coeff_b))
    expected = slope_coeff * es_pa / torch.pow(temp_c + coeff_b, 2)

    result = compute_slope_saturation_vapor_pressure(
        temp_k,
        es_pa=es_pa,
        tetens_coeff_b=coeff_b,
        tetens_slope_coeff=slope_coeff,
    )

    assert torch.allclose(result, expected, rtol=1e-6, atol=1e-8)


def test_psychrometric_constant_formula() -> None:
    cp = _to_float(get_physical_constant("air_specific_heat_cp"))
    latent_heat = _to_float(get_physical_constant("latent_heat_vaporization"))
    mw_ratio = _to_float(get_physical_constant("water_molecular_weight")) / _to_float(
        get_physical_constant("air_molecular_weight")
    )

    pressure = torch.tensor(101325.0)
    expected = cp * pressure / (mw_ratio * latent_heat)

    result = compute_psychrometric_constant(
        pressure,
        specific_heat_air=cp,
        molecular_weight_ratio=mw_ratio,
        latent_heat=latent_heat,
    )

    assert torch.allclose(result, expected, rtol=1e-6, atol=1e-8)


def _hand_moist_air_density(temperature: float, pressure: float, vapor_pressure: float) -> float:
    """rho = (p - (1 - M_w / M_d) e) / (R / M_d * T), molar masses converted g -> kg."""
    gas_constant = _to_float(get_physical_constant("gas_constant"))
    dry_air_molar_mass = _to_float(get_physical_constant("air_molecular_weight")) * 1.0e-3
    water_molar_mass = _to_float(get_physical_constant("water_molecular_weight")) * 1.0e-3
    epsilon = water_molar_mass / dry_air_molar_mass
    dry_air_gas_constant = gas_constant / dry_air_molar_mass
    return (pressure - (1.0 - epsilon) * vapor_pressure) / (dry_air_gas_constant * temperature)


def test_moist_air_density_hand_calculation() -> None:
    """25 degC, 101325 Pa, e = 1500 Pa: about 1.177 kg m-3, not the 0 degC dry value 1.293."""
    expected = _hand_moist_air_density(298.15, 101325.0, 1500.0)

    result = compute_moist_air_density(
        torch.tensor(298.15, dtype=torch.float64),
        torch.tensor(101325.0, dtype=torch.float64),
        torch.tensor(1500.0, dtype=torch.float64),
    )

    assert _to_float(result) == pytest.approx(expected, rel=1e-12)
    assert _to_float(result) == pytest.approx(1.1773, abs=5e-4)
    assert abs(_to_float(result) - _to_float(get_physical_constant("air_density_standard"))) > 0.1


@pytest.mark.parametrize("vapor_pressure_label", ["tv_ratio_1_01", "e_1500_pa"])
def test_moist_air_density_agrees_with_fao56_annex3(vapor_pressure_label: str) -> None:
    """FAO-56 Annex 3: rho_a = P / (1.01 (T + 273) R), P [kPa], R = 0.287 kJ kg-1 K-1.

    The factor 1.01 replaces the exact virtual-temperature ratio
    T_v / T = 1 / (1 - (e / p)(1 - epsilon)). With e chosen so that this ratio is
    exactly 1.01, only FAO's rounded constants differ (T + 273 instead of
    T + 273.15: 5.0e-4 relative at 25 degC; R = 0.287 instead of R_d = 0.28705:
    1.9e-4), so the tolerance is 1e-3. For any e in 0 <= e / p <= 0.0262 the ratio
    lies in [1, 1.01], so the approximation itself differs by at most 1 percent
    (tolerance 1.1e-2 including the rounded constants).
    """
    temperature = 298.15
    pressure = 101325.0
    epsilon = _to_float(get_physical_constant("water_molecular_weight")) / _to_float(
        get_physical_constant("air_molecular_weight")
    )
    if vapor_pressure_label == "tv_ratio_1_01":
        vapor_pressure = pressure * (1.0 - 1.0 / 1.01) / (1.0 - epsilon)
        tolerance = 1.0e-3
    else:
        vapor_pressure = 1500.0
        tolerance = 1.1e-2
    fao56 = (pressure / 1000.0) / (1.01 * (temperature - 273.15 + 273.0) * 0.287)

    result = compute_moist_air_density(
        torch.tensor(temperature, dtype=torch.float64),
        torch.tensor(pressure, dtype=torch.float64),
        torch.tensor(vapor_pressure, dtype=torch.float64),
    )

    assert _to_float(result) == pytest.approx(fao56, rel=tolerance)


@pytest.mark.parametrize(
    "temperature, pressure, vapor_pressure, message",
    [
        (0.0, 101325.0, 1500.0, "temperature"),
        (298.15, 0.0, 0.0, "pressure"),
        (298.15, 101325.0, -1.0, "vapor pressure"),
        (298.15, 1500.0, 1500.0, "vapor pressure"),
    ],
)
def test_moist_air_density_rejects_values_outside_its_domain(
    temperature: float, pressure: float, vapor_pressure: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_moist_air_density(
            torch.tensor(temperature), torch.tensor(pressure), torch.tensor(vapor_pressure)
        )


def test_moist_air_density_is_differentiable() -> None:
    """d rho/dp = 1/(R_d T), d rho/de = -(1 - epsilon)/(R_d T), d rho/dT = -rho/T."""
    temperature = torch.tensor(298.15, dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor(101325.0, dtype=torch.float64, requires_grad=True)
    vapor_pressure = torch.tensor(1500.0, dtype=torch.float64, requires_grad=True)

    density = compute_moist_air_density(temperature, pressure, vapor_pressure)
    density.backward()

    gas_constant = _to_float(get_physical_constant("gas_constant"))
    dry_air_molar_mass = _to_float(get_physical_constant("air_molecular_weight")) * 1.0e-3
    epsilon = _to_float(get_physical_constant("water_molecular_weight")) * 1.0e-3 / dry_air_molar_mass
    rd_t = gas_constant / dry_air_molar_mass * 298.15
    assert _to_float(pressure.grad) == pytest.approx(1.0 / rd_t, rel=1e-12)
    assert _to_float(vapor_pressure.grad) == pytest.approx(-(1.0 - epsilon) / rd_t, rel=1e-12)
    assert _to_float(temperature.grad) == pytest.approx(-_to_float(density) / 298.15, rel=1e-12)
