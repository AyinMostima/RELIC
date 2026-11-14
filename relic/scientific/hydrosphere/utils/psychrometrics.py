"""Hydrosphere psychrometric utilities.

Centralizes saturation vapor pressure, slope of SVP curve, psychrometric
constant and moist-air density calculations for evapotranspiration modules.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from relic.utils.physical_constants import get_physical_constant

# Physical-constant SSOT keys of the moist-air equation of state.
_GAS_CONSTANT_KEY = "relic.physical.constants.universal.gas_constant"
_DRY_AIR_MOLAR_MASS_KEY = "relic.physical.constants.air.molecular_weight_dry"
_WATER_MOLAR_MASS_KEY = "relic.physical.constants.water.molecular_weight"


# References: Tetens (1930); Allen et al. (1998) FAO-56; RHESSys compute_saturation_vapor_pressure.c
# Mathematical expression: es = e0 * exp(a * T_c / (T_c + b))
# Parameter meanings: e0 base pressure [Pa], a dimensionless, b temperature offset [degC], T_c air temperature [degC].
def compute_saturation_vapor_pressure(
    temp_K: Tensor,
    *,
    tetens_base_pressure: Optional[float] = None,
    tetens_coeff_a: Optional[float] = None,
    tetens_coeff_b: Optional[float] = None,
    kelvin_offset: Optional[float] = None,
) -> Tensor:
    """Compute saturation vapor pressure using the Tetens formulation."""
    if tetens_base_pressure is None or tetens_coeff_a is None or tetens_coeff_b is None:
        raise ValueError(
            "Tetens parameters must be passed from the calling functor parameter context."
        )
    if kelvin_offset is None:
        kelvin_offset = float(get_physical_constant("celsius_to_kelvin"))
    temp_C = temp_K - kelvin_offset
    return tetens_base_pressure * torch.exp(
        tetens_coeff_a * temp_C / (temp_C + tetens_coeff_b)
    )


# References: Allen et al. (1998) FAO-56
# Mathematical expression: delta = slope_coeff * es / (T_c + b)^2
# Parameter meanings: slope_coeff constant, es saturation vapor pressure [Pa], b temperature offset [degC].
def compute_slope_saturation_vapor_pressure(
    temp_K: Tensor,
    *,
    es_pa: Optional[Tensor] = None,
    tetens_base_pressure: Optional[float] = None,
    tetens_coeff_a: Optional[float] = None,
    tetens_coeff_b: Optional[float] = None,
    tetens_slope_coeff: Optional[float] = None,
    kelvin_offset: Optional[float] = None,
) -> Tensor:
    """Compute slope of saturation vapor pressure curve."""
    if tetens_slope_coeff is None or tetens_coeff_b is None:
        raise ValueError(
            "Tetens slope parameters must be passed from the calling functor parameter context."
        )
    if es_pa is None:
        es_pa = compute_saturation_vapor_pressure(
            temp_K,
            tetens_base_pressure=tetens_base_pressure,
            tetens_coeff_a=tetens_coeff_a,
            tetens_coeff_b=tetens_coeff_b,
            kelvin_offset=kelvin_offset,
        )

    if kelvin_offset is None:
        kelvin_offset = float(get_physical_constant("celsius_to_kelvin"))
    temp_C = temp_K - kelvin_offset
    return tetens_slope_coeff * es_pa / torch.pow(temp_C + tetens_coeff_b, 2)


# References: Allen et al. (1998) FAO-56
# Mathematical expression: gamma = cp * P / (epsilon * lambda)
# Parameter meanings: cp specific heat of air [J kg-1 K-1], P pressure [Pa],
# epsilon molecular weight ratio water to air, lambda latent heat [J kg-1].
def compute_psychrometric_constant(
    pressure: Tensor,
    *,
    specific_heat_air: float,
    molecular_weight_ratio: float,
    latent_heat: float,
) -> Tensor:
    """Compute psychrometric constant."""
    return specific_heat_air * pressure / (molecular_weight_ratio * latent_heat)


def _physical_constant_in_units(key: str, units: str) -> float:
    """Return one physical-constant SSOT value converted from its declared units."""
    from cf_units import Unit

    from relic.parameters import get_global_manager

    spec = get_global_manager().get_parameter(key)
    if spec is None or not spec.units:
        raise ValueError(f"Physical constant '{key}' has no declared units in the SSOT")
    return float(Unit(spec.units).convert(float(get_physical_constant(key)), Unit(units)))


# References: Holton, J.R. and Hakim, G.J. (2012), An Introduction to Dynamic
#   Meteorology, 5th ed., Eqs. 2.63-2.64 (p = rho R_d T_v with the virtual temperature
#   T_v = T / (1 - (e / p)(1 - R_d / R_v))); Peixoto and Oort (1992), Physics of
#   Climate (rho = rho_d + rho_v = (p / (R_d T))(1 - 0.378 e / p)); Allen et al. (1998),
#   FAO-56, Annex 3 (rho_a = P / (1.01 (T + 273) R), where 1.01 approximates T_v / T).
# Mathematical expression: rho = (p - (1 - epsilon) e) / (R_d T), with
#   epsilon = M_w / M_d and R_d = R / M_d; identical to rho = p / (R_d T_v).
# Parameter meanings: rho moist-air density; p total air pressure; e partial pressure
#   of water vapour (actual vapour pressure); T air temperature; R universal gas
#   constant; M_d and M_w molar masses of dry air and water; R_d specific gas constant
#   of dry air; epsilon = M_w / M_d = R_d / R_v.
# Units: rho [kg m-3]; p and e [Pa]; T [K]; R [J mol-1 K-1]; M_d and M_w [kg mol-1]
#   (converted from their SSOT units); R_d [J kg-1 K-1]; epsilon [1].
# Domain/constraints: T > 0, p > 0 and 0 <= e < p, otherwise ValueError; ideal-gas
#   mixture of dry air and water vapour; differentiable in T, p and e.
# Sign convention: rho is a positive state property, not a flux.
def compute_moist_air_density(
    air_temperature: Tensor,
    atmospheric_pressure: Tensor,
    vapor_pressure: Tensor,
) -> Tensor:
    """Moist-air density [kg m-3] from air temperature [K], pressure [Pa] and vapour pressure [Pa]."""
    if bool(torch.any(air_temperature <= 0)):
        raise ValueError("Moist-air density needs air temperature > 0 K")
    if bool(torch.any(atmospheric_pressure <= 0)):
        raise ValueError("Moist-air density needs atmospheric pressure > 0 Pa")
    if bool(torch.any(vapor_pressure < 0)) or bool(torch.any(vapor_pressure >= atmospheric_pressure)):
        raise ValueError("Moist-air density needs 0 <= vapor pressure < atmospheric pressure")
    gas_constant = _physical_constant_in_units(_GAS_CONSTANT_KEY, "J mol-1 K-1")
    dry_air_molar_mass = _physical_constant_in_units(_DRY_AIR_MOLAR_MASS_KEY, "kg mol-1")
    water_molar_mass = _physical_constant_in_units(_WATER_MOLAR_MASS_KEY, "kg mol-1")
    epsilon = water_molar_mass / dry_air_molar_mass
    dry_air_gas_constant = gas_constant / dry_air_molar_mass
    return (atmospheric_pressure - (1.0 - epsilon) * vapor_pressure) / (
        dry_air_gas_constant * air_temperature
    )
