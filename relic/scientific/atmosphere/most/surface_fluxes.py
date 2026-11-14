"""
Surface Fluxes Module for MOST

This module provides functions for calculating surface fluxes of momentum, heat, 
and scalars using Monin-Obukhov Similarity Theory, including aerodynamic resistance
and eddy diffusivity calculations.

Mathematical Expressions:
    Scalar resistance: rah = fh / (k * u*)
    Momentum resistance: ram = fm / (k * u*)
    fx = ln((z-d)/z0x) - psi_x((z-d)/L) + psi_x(z0x/L)
    Eddy diffusivity: K = k * u* * z / phi_h(z/L)
    Momentum flux: tau = rho * u*^2
    Heat flux: H = rho * cp * u* * theta*

References:
    - Paulson, C.A. (1970). J. Appl. Meteorol. 9, 857-861
    - Businger, J.A. et al. (1971). J. Atmos. Sci. 28, 181-189
    - CTSM Technical Note, Eqs. 2.5.55-2.5.57
    - Garratt, J.R. (1992). The Atmospheric Boundary Layer
    - Campbell, G.S. & Norman, J.M. (1998). An Introduction to Environmental Biophysics
    - Brutsaert, W. (1982). Evaporation into the Atmosphere

Parameter Meanings:
    ra: Aerodynamic resistance [s m^-1]
    K: Eddy diffusivity [m^2 s^-1]
    tau: Momentum flux [kg m^-1 s^-2]
    H: Sensible heat flux [W m-2]
"""

import torch
from functools import lru_cache
from typing import Callable, Dict, Optional, Union
from .similarity_functions import psi_m, psi_h, phi_h
from relic.utils.physical_constants import get_physical_constant
from relic.parameters import get_global_manager
from ....parameters.namespaces import ParameterNamespace, resolve_namespace

# Type alias
Tensor = torch.Tensor

# Physical constants come directly from the physical-constants SSOT. This
# avoids making surface-flux primitives depend on the stability solver module.
VON_KARMAN_CONSTANT = get_physical_constant("von_karman_constant")
SPECIFIC_HEAT_AIR = get_physical_constant("air_specific_heat_cp")
LATENT_HEAT_VAPORIZATION = get_physical_constant("latent_heat_vaporization")


def _require_most_parameter(name: str, units: str) -> float:
    """
    Retrieve a MOST parameter without default defaults.
    """
    namespace = resolve_namespace(ParameterNamespace.ATMOSPHERE_MOST)
    full_key = name if name.startswith(namespace) else f"{namespace}.{name}"
    value = get_global_manager().get(
        full_key,
        units=units,
        expected_type=(float, int),
    )
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].tolist())
    return float(value)


@lru_cache(maxsize=1)
def _load_surface_flux_parameters() -> Dict[str, float]:
    """Load and cache MOST surface flux parameters."""
    return {
        "clamp_min_K": _require_most_parameter("clamp_min_K", units="m2 s-1"),
        "bulk_wind_speed_safeguard": _require_most_parameter("bulk_wind_speed_safeguard", units="m s-1"),
        "bulk_temperature_difference_safeguard": _require_most_parameter("bulk_temperature_difference_safeguard", units="K"),
        "roughness_sublayer_c": _require_most_parameter("roughness_sublayer_c", units="dimensionless"),
    }


def _get_surface_flux_parameter(key: str) -> float:
    params = _load_surface_flux_parameters()
    return params[key]


def _validate_resistance_inputs(
    u_star: Tensor,
    z: Tensor,
    roughness_length: Tensor,
    zeta: Tensor,
    d: Optional[Tensor],
    k: float,
) -> tuple[Tensor, Tensor]:
    """Validate integrated MOST profile inputs without numerical fallback."""
    named_tensors = {
        "u_star": u_star,
        "z": z,
        "roughness_length": roughness_length,
        "zeta": zeta,
    }
    if d is not None:
        named_tensors["d"] = d

    for name, value in named_tensors.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must use a floating dtype")
        if value.device != u_star.device or value.dtype != u_star.dtype:
            raise ValueError("MOST resistance inputs must share dtype and device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    k_tensor = torch.as_tensor(k, dtype=u_star.dtype, device=u_star.device)
    if not bool(torch.isfinite(k_tensor).all()) or bool(torch.any(k_tensor <= 0.0)):
        raise ValueError("k must be finite and strictly positive")
    if bool(torch.any(u_star <= 0.0)):
        raise ValueError("u_star must be strictly positive")
    if bool(torch.any(roughness_length <= 0.0)):
        raise ValueError("roughness_length must be strictly positive")

    z_eff = z if d is None else z - d
    if bool(torch.any(z_eff <= roughness_length)):
        raise ValueError("z - d must be greater than the roughness length")
    return z_eff, k_tensor


def _integrated_profile_factor(
    z_eff: Tensor,
    roughness_length: Tensor,
    zeta: Tensor,
    stability_function: Callable[[Tensor], Tensor],
) -> Tensor:
    """Return the exact integrated MOST profile factor between z0 and z."""
    roughness_zeta = zeta * roughness_length / z_eff
    factor = (
        torch.log(z_eff / roughness_length)
        - stability_function(zeta)
        + stability_function(roughness_zeta)
    )
    if not bool(torch.isfinite(factor).all()):
        raise ValueError("MOST profile factor must contain only finite values")
    if bool(torch.any(factor <= 0.0)):
        raise ValueError("MOST profile factor must be strictly positive")
    return factor


def aerodynamic_resistance(
    u_star: Tensor,
    z: Tensor,
    z0h: Tensor,
    zeta: Tensor,
    d: Optional[Tensor] = None,
    k: float = VON_KARMAN_CONSTANT
) -> Tensor:
    """
    Calculate aerodynamic resistance for scalar transport.
    
    Mathematical Expression:
        fh = ln((z-d)/z0h) - psi_h((z-d)/L) + psi_h(z0h/L)
        rah = fh / (k * u*)
    
    This resistance represents the difficulty of transporting scalars (heat, water vapor,
    CO2, etc.) between the surface and the reference height in the atmosphere.
    
    Parameters:
        u_star : Tensor
            Friction velocity [m s-1]
        z : Tensor
            Reference height [m]
        z0h : Tensor
            Heat/scalar roughness length [m]
        zeta : Tensor
            Stability parameter (z-d)/L [dimensionless]
        d : Optional[Tensor]
            Displacement height [m] (default: 0)
        k : float
            von Karman constant (default: VON_KARMAN_CONSTANT)
    
    Returns:
        ra : Tensor
            Aerodynamic resistance [s m^-1]
    
    References:
        - Paulson (1970). J. Appl. Meteorol. 9, 857-861
        - CTSM Technical Note, Eq. 2.5.56
        - Garratt (1992). The Atmospheric Boundary Layer, Ch. 4

    Open/closed system route:
        This diagnostic resistance does not write state or create a source,
        sink, or boundary flux.
    """
    z_eff, k_tensor = _validate_resistance_inputs(
        u_star, z, z0h, zeta, d, k
    )
    heat_factor = _integrated_profile_factor(z_eff, z0h, zeta, psi_h)
    return heat_factor / (k_tensor * u_star)


def aerodynamic_resistance_momentum(
    u_star: Tensor,
    z: Tensor,
    z0m: Tensor,
    zeta: Tensor,
    d: Optional[Tensor] = None,
    k: float = VON_KARMAN_CONSTANT
) -> Tensor:
    """
    Calculate aerodynamic resistance for momentum transport.
    
    Mathematical Expression:
        fm = ln((z-d)/z0m) - psi_m((z-d)/L) + psi_m(z0m/L)
        ram = fm / (k * u*)
    
    Parameters:
        u_star : Tensor
            Friction velocity [m s-1]
        z : Tensor
            Reference height [m]
        z0m : Tensor
            Momentum roughness length [m]
        zeta : Tensor
            Stability parameter [dimensionless]
        d : Optional[Tensor]
            Displacement height [m]
        k : float
            von Karman constant
    
    Returns:
        ram : Tensor
            Momentum aerodynamic resistance [s m^-1]

    References:
        - Paulson (1970). J. Appl. Meteorol. 9, 857-861
        - CTSM Technical Note, Eq. 2.5.55

    Open/closed system route:
        This diagnostic resistance does not write state or create a source,
        sink, or boundary flux.
    """
    z_eff, k_tensor = _validate_resistance_inputs(
        u_star, z, z0m, zeta, d, k
    )
    momentum_factor = _integrated_profile_factor(z_eff, z0m, zeta, psi_m)
    return momentum_factor / (k_tensor * u_star)


def eddy_diffusivity(
    u_star: Tensor,
    z: Tensor,
    zeta: Tensor,
    d: Optional[Tensor] = None,
    k: float = VON_KARMAN_CONSTANT
) -> Tensor:
    """
    Calculate turbulent eddy diffusivity for heat and scalars.
    
    Mathematical Expression:
        Kh = k * u* * (z - d) / phi_h(z/L)
    
    where phi_h is the dimensionless temperature gradient function.
    
    Parameters:
        u_star : Tensor
            Friction velocity [m s-1]
        z : Tensor
            Height above ground [m]
        zeta : Tensor
            Stability parameter (z-d)/L [dimensionless]
        d : Optional[Tensor]
            Displacement height [m]
        k : float
            von Karman constant
    
    Returns:
        Kh : Tensor
            Eddy diffusivity for heat/scalars [m^2 s^-1]
    
    References:
        - Businger et al. (1971)
        - Dyer (1974)
    """
    # Calculate effective height
    if d is not None:
        z_eff = z - d
    else:
        z_eff = z
    
    # Calculate dimensionless gradient
    phi_h_val = phi_h(zeta)
    
    # Eddy diffusivity
    Kh = k * u_star * z_eff / phi_h_val
    
    # Apply minimum constraint using parameter-managed clamp
    return torch.clamp(Kh, min=_get_surface_flux_parameter("clamp_min_K"))


def _positive_density(rho: Union[Tensor, float], reference: Tensor) -> Tensor:
    """Return the caller's air density as a finite positive tensor broadcastable with reference."""
    density = torch.as_tensor(rho, dtype=reference.dtype, device=reference.device)
    if not bool(torch.isfinite(density).all()) or bool(torch.any(density <= 0.0)):
        raise ValueError("Air density rho must be finite and strictly positive [kg m-3]")
    try:
        torch.broadcast_shapes(density.shape, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Air density shape {tuple(density.shape)} does not broadcast with "
            f"{tuple(reference.shape)}"
        ) from exc
    return density


def momentum_flux(
    u_star: Tensor,
    rho: Union[Tensor, float],
) -> Tensor:
    """
    Calculate surface momentum flux (shear stress).

    Mathematical Expression:
        tau = rho * u*^2

    Parameters:
        u_star : Tensor
            Friction velocity [m s-1]
        rho : Tensor or float
            Actual (moist) air density [kg m-3] of the caller, for example from
            compute_moist_air_density; there is no default density.

    Returns:
        tau : Tensor
            Momentum flux [kg m^-1 s^-2] = [N m^-2]
    """
    # References: Stull (1988), Sec. 2.10; Garratt (1992), Ch. 3.
    # Mathematical expression: tau = rho * u*^2.
    # Parameter meanings: rho the actual air density of the caller, u* the
    #   friction velocity.
    # Units: rho [kg m-3]; u* [m s-1]; tau [kg m-1 s-2].
    # Domain/constraints: rho finite and > 0, broadcastable with u*; no default.
    # Sign convention: tau is the magnitude of the downward momentum flux.
    density = _positive_density(rho, u_star)
    return density * u_star * u_star


def sensible_heat_flux(
    u_star: Tensor,
    theta_star: Tensor,
    rho: Union[Tensor, float],
    cp: Optional[float] = None
) -> Tensor:
    """
    Calculate sensible heat flux.
    
    Mathematical Expression:
        H = rho * cp * u* * theta*
    
    Parameters:
        u_star : Tensor
            Friction velocity [m s-1]
        theta_star : Tensor
            Temperature scale [K]
        rho : Tensor or float
            Actual (moist) air density [kg m-3] of the caller; no default.
        cp : Optional[float]
            Specific heat capacity [J kg^-1 K^-1] (default: from parameter system)

    Returns:
        H : Tensor
            Sensible heat flux [W m-2]
    """
    # References: Stull (1988), Sec. 2.10; Garratt (1992), Ch. 3.
    # Mathematical expression: H = rho * cp * u* * theta*.
    # Parameter meanings: rho the actual air density, cp the specific heat of
    #   air at constant pressure, u* friction velocity, theta* temperature scale.
    # Units: rho [kg m-3]; cp [J kg-1 K-1]; u* [m s-1]; theta* [K]; H [W m-2].
    # Domain/constraints: rho finite and > 0, broadcastable with u*; no default.
    # Sign convention: theta* = k (theta_s - theta_a) / fh > 0 for a surface
    #   warmer than the air gives an upward H > 0.
    density = _positive_density(rho, u_star)
    heat_capacity = SPECIFIC_HEAT_AIR if cp is None else cp
    return density * heat_capacity * u_star * theta_star


def latent_heat_flux(
    u_star: Tensor,
    q_star: Tensor,
    rho: Union[Tensor, float],
    Lv: Optional[float] = None
) -> Tensor:
    """
    Calculate latent heat flux.
    
    Mathematical Expression:
        LE = rho * Lv * u* * q*
    
    Parameters:
        u_star : Tensor
            Friction velocity [m s-1]
        q_star : Tensor
            Humidity scale [kg kg^-1]
        rho : Tensor or float
            Actual (moist) air density [kg m-3] of the caller; no default.
        Lv : Optional[float]
            Latent heat of vaporization [J kg^-1] (default: from parameter system)

    Returns:
        LE : Tensor
            Latent heat flux [W m-2]
    """
    # References: Stull (1988), Sec. 2.10; Garratt (1992), Ch. 3.
    # Mathematical expression: LE = rho * Lv * u* * q*.
    # Parameter meanings: rho the actual air density, Lv the latent heat of
    #   vaporization, u* friction velocity, q* humidity scale.
    # Units: rho [kg m-3]; Lv [J kg-1]; u* [m s-1]; q* [kg kg-1]; LE [W m-2].
    # Domain/constraints: rho finite and > 0, broadcastable with u*; no default.
    # Sign convention: q* > 0 for a surface moister than the air (the
    #   orientation of theta*) gives an upward LE > 0.
    density = _positive_density(rho, u_star)
    latent_heat = LATENT_HEAT_VAPORIZATION if Lv is None else Lv
    return density * latent_heat * u_star * q_star


def scalar_flux(
    concentration_diff: Tensor,
    ra: Tensor
) -> Tensor:
    """
    Calculate scalar flux using gradient-resistance approach.
    
    Mathematical Expression:
        F = (C_surface - C_air) / ra
    
    Parameters:
        concentration_diff : Tensor
            Concentration difference (surface - air) [concentration units]
        ra : Tensor
            Aerodynamic resistance [s m^-1]
    
    Returns:
        F : Tensor
            Scalar flux [concentration units * m s-1]
    """
    return concentration_diff / ra


def bulk_transfer_coefficients(
    u_star: Tensor,
    U: Tensor,
    theta_star: Tensor,
    delta_theta: Tensor
) -> Dict[str, Tensor]:
    """
    Calculate bulk transfer coefficients.
    
    Mathematical Expressions:
        CD = (u* / U)^2  (drag coefficient)
        CH = u* * theta* / (U * delta_theta)  (heat transfer coefficient)
    
    Parameters:
        u_star : Tensor
            Friction velocity [m s-1]
        U : Tensor
            Wind speed [m s-1]
        theta_star : Tensor
            Temperature scale [K]
        delta_theta : Tensor
            Temperature difference [K]
    
    Returns:
        coefficients : Dict[str, Tensor]
            - 'CD': Drag coefficient [dimensionless]
            - 'CH': Heat transfer coefficient [dimensionless]
    """
    # Avoid division by zero
    wind_floor = _get_surface_flux_parameter("bulk_wind_speed_safeguard")
    temp_floor = _get_surface_flux_parameter("bulk_temperature_difference_safeguard")
    safe_wind_fill = torch.where(U >= 0.0, torch.full_like(U, wind_floor), torch.full_like(U, -wind_floor))
    safe_temp_fill = torch.where(delta_theta >= 0.0, torch.full_like(delta_theta, temp_floor), torch.full_like(delta_theta, -temp_floor))
    U_safe = torch.where(torch.abs(U) < wind_floor,
                         safe_wind_fill,
                         U)
    
    delta_theta_safe = torch.where(torch.abs(delta_theta) < temp_floor,
                                   safe_temp_fill,
                                   delta_theta)
    
    # Calculate coefficients
    CD = (u_star / U_safe) ** 2
    CH = u_star * theta_star / (U_safe * delta_theta_safe)
    
    return {
        'CD': CD,
        'CH': CH
    }


def roughness_sublayer_correction(
    z: Tensor,
    h: Tensor,
    z0m: Tensor,
    d: Optional[Tensor] = None
) -> Tensor:
    """
    Calculate roughness sublayer correction factor.
    
    The roughness sublayer is the region above the canopy where MOST
    may not apply directly due to canopy-induced turbulence.
    
    Mathematical Expression:
        psi_RSL = exp(-c * (z - d) / h)  for (z - d) < 2h
        psi_RSL = 1  for (z - d) >= 2h
    
    Parameters:
        z : Tensor
            Height above ground [m]
        h : Tensor
            Canopy height [m]
        z0m : Tensor
            Momentum roughness length [m]
        d : Optional[Tensor]
            Displacement height [m]
    
    Returns:
        psi_RSL : Tensor
            Roughness sublayer correction factor [dimensionless]
    
    References:
        - Garratt (1980). Boundary-Layer Meteorol. 19, 471-479
        - Cellier & Brunet (1992). Boundary-Layer Meteorol. 58, 19-31
    """
    # Calculate effective height
    if d is not None:
        z_eff = z - d
    else:
        z_eff = z
    
    # Empirical coefficient loaded from parameter database.
    c = _get_surface_flux_parameter("roughness_sublayer_c")
    
    # Calculate correction
    psi_RSL = torch.ones_like(z)
    
    # Apply correction in roughness sublayer (z_eff < 2h)
    rsl_mask = z_eff < 2.0 * h
    if torch.any(rsl_mask):
        psi_RSL[rsl_mask] = torch.exp(-c * z_eff[rsl_mask] / h[rsl_mask])
    
    return psi_RSL


# Export all functions
__all__ = [
    'aerodynamic_resistance',
    'aerodynamic_resistance_momentum',
    'eddy_diffusivity',
    'momentum_flux',
    'sensible_heat_flux',
    'latent_heat_flux',
    'scalar_flux',
    'bulk_transfer_coefficients',
    'roughness_sublayer_correction'
]


