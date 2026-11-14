"""
Monin-Obukhov Stability Parameters Module

This module provides functions for calculating key stability parameters in MOST theory,
including friction velocity (u*), Monin-Obukhov length (L), and stability parameter (zeta).

Mathematical Expressions:
    fm = ln((z-d)/z0m) - psi_m((z-d)/L) + psi_m(z0m/L)
    Friction velocity: u* = k * U / fm
    Monin-Obukhov length: L = -u*^3 * rho * cp * T / (k * g * H)
    Stability parameter: zeta = (z - d) / L

References:
    - Monin, A.S. & Obukhov, A.M. (1954). Basic laws of turbulent mixing
    - Foken, T. (2006). 50 years of the Monin-Obukhov similarity theory
    - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology

Parameter Meanings:
    u*: Friction velocity [m s-1]
    L: Monin-Obukhov length scale [m]
    zeta: Stability parameter z/L [dimensionless]
    k: von Karman constant (VON_KARMAN_CONSTANT) [dimensionless]
    g: Gravity acceleration (GRAVITY_ACCELERATION) [m s-2]
"""

import math
import torch
from functools import lru_cache
from typing import Dict, Optional, Tuple, Union
from .similarity_functions import psi_m, psi_h
from .surface_fluxes import (
    _integrated_profile_factor,
    _positive_density,
    sensible_heat_flux,
)

# Type alias
Tensor = torch.Tensor

# Physical constants - using unified interface
from ....utils.physical_constants import (
    get_physical_constant,
)
from relic.parameters import get_global_manager
from ....parameters.namespaces import ParameterNamespace, resolve_namespace

# Load physical constants using unified interface
VON_KARMAN_CONSTANT = get_physical_constant('von_karman_constant')
GRAVITY_ACCELERATION = get_physical_constant('gravity_acceleration')
SPECIFIC_HEAT_AIR = get_physical_constant('air_specific_heat_cp')
MOST_NAMESPACE = resolve_namespace(ParameterNamespace.ATMOSPHERE_MOST)


def _require_most_parameter(name: str, units: str) -> float:
    """Load required MOST parameter for stability operators."""
    full_key = name if name.startswith(MOST_NAMESPACE) else f"{MOST_NAMESPACE}.{name}"
    value = get_global_manager().get(
        full_key,
        units=units,
        expected_type=(float, int),
        required=True,
    )
    if isinstance(value, Tensor):
        return float(value.detach().reshape(-1)[0].tolist())
    return float(value)


@lru_cache(maxsize=1)
def _load_stability_parameters() -> Dict[str, float]:
    """Cache MOST stability parameters used by this module."""
    manager = get_global_manager()
    max_iterations = manager.get(
        "atmosphere.turbulence.max_iterations",
        units="dimensionless",
        expected_type=(int, float),
        required=True,
    )
    convergence_tolerance = manager.get(
        "atmosphere.turbulence.convergence_tolerance",
        units="dimensionless",
        expected_type=(float, int),
        required=True,
    )
    neutral_length = manager.get(
        "atmosphere.turbulence.neutral_obukhov_length",
        units="m",
        expected_type=(float, int),
        required=True,
    )

    def scalar(value: object) -> float:
        if isinstance(value, Tensor):
            return float(value.detach().reshape(-1)[0].tolist())
        return float(value)

    return {
        "h_min_safeguard": _require_most_parameter(
            "h_min_safeguard",
            units="W m-2",
        ),
        "monin_obukhov_length_clamp_min": _require_most_parameter(
            "monin_obukhov_length_clamp_min",
            units="m",
        ),
        "monin_obukhov_length_clamp_max": _require_most_parameter(
            "monin_obukhov_length_clamp_max",
            units="m",
        ),
        "max_iterations": scalar(max_iterations),
        "convergence_tolerance": scalar(convergence_tolerance),
        "neutral_obukhov_length": scalar(neutral_length),
    }


def _get_stability_parameter(name: str) -> float:
    """Return cached stability-parameter value."""
    return _load_stability_parameters()[name]


def _validate_tensor_group(reference: Tensor, **values: Tensor) -> None:
    """Validate a differentiable MOST tensor group."""
    if not isinstance(reference, Tensor):
        raise TypeError("MOST inputs must be torch.Tensor values")
    if not torch.is_floating_point(reference):
        raise TypeError("MOST inputs must use floating dtypes")
    for name, value in values.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must use a floating dtype")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError("MOST inputs must share dtype and device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("MOST reference input must contain only finite values")


def _positive_scalar(name: str, value: float, reference: Tensor) -> Tensor:
    """Return one finite positive scalar on the reference dtype and device."""
    scalar = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if scalar.numel() != 1 or not bool(torch.isfinite(scalar).all()):
        raise ValueError(f"{name} must be a finite scalar")
    if bool(torch.any(scalar <= 0.0)):
        raise ValueError(f"{name} must be strictly positive")
    return scalar


def _validate_profile_geometry(
    z: Tensor,
    roughness_length: Tensor,
    d: Tensor,
) -> Tensor:
    """Validate MOST reference-height and roughness geometry."""
    z_eff = z - d
    if bool(torch.any(roughness_length <= 0.0)):
        raise ValueError("MOST roughness length must be strictly positive")
    if bool(torch.any(z_eff <= roughness_length)):
        raise ValueError(
            "MOST reference height minus displacement must exceed roughness length"
        )
    return z_eff


def friction_velocity(
    U: Tensor,
    z: Tensor,
    z0m: Tensor,
    d: Tensor,
    zeta: Tensor,
    k: float = VON_KARMAN_CONSTANT
) -> Tensor:
    """
    Calculate friction velocity using MOST.
    
    Mathematical Expression:
        fm = ln((z-d)/z0m) - psi_m((z-d)/L) + psi_m(z0m/L)
        u* = k * U / fm
    
    Parameters:
        U : Tensor
            Wind speed at height z [m s-1]
        z : Tensor
            Reference height [m]
        z0m : Tensor
            Momentum roughness length [m]
        d : Tensor
            Displacement height [m]
        zeta : Tensor
            Stability parameter (z-d)/L [dimensionless]
        k : float
            von Karman constant (default: VON_KARMAN_CONSTANT)
    
    Returns:
        u_star : Tensor
            Friction velocity [m s-1]
    
    References:
        - Paulson (1970). J. Appl. Meteorol. 9, 857-861
        - Businger et al. (1971). J. Atmos. Sci. 28, 181-189
        - Garratt (1992). The Atmospheric Boundary Layer, Ch. 3

    Open/closed system route:
        Friction velocity is a diagnostic scale. This operator neither writes
        state nor creates a momentum source, sink, or boundary flux.
    """
    _validate_tensor_group(U, z=z, z0m=z0m, d=d, zeta=zeta)
    if bool(torch.any(U <= 0.0)):
        raise ValueError("MOST wind speed must be strictly positive")
    z_eff = _validate_profile_geometry(z, z0m, d)
    k_tensor = _positive_scalar("k", k, U)

    # References: Paulson (1970); Businger et al. (1971).
    # Expression: u* = k U / fm with the complete integrated profile fm.
    # Parameters: U [m s-1], z/d/z0m [m], zeta [1], k [1].
    momentum_factor = _integrated_profile_factor(z_eff, z0m, zeta, psi_m)
    u_star = k_tensor * U / momentum_factor
    if not bool(torch.isfinite(u_star).all()) or bool(torch.any(u_star <= 0.0)):
        raise ValueError("MOST friction velocity must be finite and positive")
    return u_star


def monin_obukhov_length(
    u_star: Tensor,
    rho: Union[Tensor, float],
    cp: float,
    T: Tensor,
    H: Tensor,
    k: float = VON_KARMAN_CONSTANT,
    g: float = GRAVITY_ACCELERATION
) -> Tensor:
    """
    Calculate Monin-Obukhov length.
    
    Mathematical Expression:
        L = -u*^3 * rho * cp * T / (k * g * H)
        
    Physical Interpretation:
        L > 0: Stable conditions (heat flux downward)
        L < 0: Unstable conditions (heat flux upward)
        |L| -> inf: Neutral conditions (small heat flux)
    
    Parameters:
        u_star : Tensor
            Friction velocity [m s-1]
        rho : Tensor or float
            Actual (moist) air density [kg m-3], a scalar or a field
            broadcastable with u_star; no default density
        cp : float
            Specific heat capacity [J kg^-1 K^-1]
        T : Tensor
            Air temperature [K]
        H : Tensor
            Sensible heat flux [W m-2]
        k : float
            von Karman constant (default: VON_KARMAN_CONSTANT)
        g : float
            Gravity acceleration (default: GRAVITY_ACCELERATION m s-2)
    
    Returns:
        L : Tensor
            Monin-Obukhov length [m]
    
    References:
        - Monin & Obukhov (1954). Tr. Inst. Teor. Geofiz. Akad. Nauk SSSR 24, 163-187
        - Obukhov (1971). Boundary-Layer Meteorol. 2, 7-29

    Open/closed system route:
        L is a diagnostic length. The near-neutral heat-flux safeguard only
        selects a finite diagnostic representation and does not alter H.
    """
    _validate_tensor_group(u_star, T=T, H=H)
    if bool(torch.any(u_star <= 0.0)):
        raise ValueError("u_star must be strictly positive")
    if bool(torch.any(T <= 0.0)):
        raise ValueError("air temperature must be strictly positive")
    rho_tensor = _positive_density(rho, u_star)
    cp_tensor = _positive_scalar("cp", cp, u_star)
    k_tensor = _positive_scalar("k", k, u_star)
    g_tensor = _positive_scalar("g", g, u_star)

    # References: Monin and Obukhov (1954); Obukhov (1971).
    # Mathematical expression: L = -rho cp T u*^3 / (k g H).
    # Parameter meanings: rho the caller's actual air density (for example
    #   compute_moist_air_density), cp heat capacity, T air temperature,
    #   u* friction velocity, k von Karman constant, g gravity, H sensible heat flux.
    # Units: rho [kg m-3]; cp [J kg-1 K-1]; T [K]; u* [m s-1]; g [m s-2];
    #   H [W m-2]; L [m].
    # Domain/constraints: rho finite and > 0 (field allowed); |H| below
    #   h_min_safeguard is replaced by +/- h_min_safeguard; L is clamped.
    # Sign convention: upward H > 0 gives L < 0 (unstable).
    H_min = u_star.new_tensor(_get_stability_parameter("h_min_safeguard"))
    sign_h = torch.where(H < 0.0, -torch.ones_like(H), torch.ones_like(H))
    H_safe = torch.where(torch.abs(H) < H_min, sign_h * H_min, H)

    L = -u_star**3 * rho_tensor * cp_tensor * T / (
        k_tensor * g_tensor * H_safe
    )
    L_bounded = torch.clamp(
        L,
        min=_get_stability_parameter("monin_obukhov_length_clamp_min"),
        max=_get_stability_parameter("monin_obukhov_length_clamp_max"),
    )
    if not bool(torch.isfinite(L_bounded).all()):
        raise ValueError("Monin-Obukhov length must be finite")
    return L_bounded


def zeta(z: Tensor, L: Tensor, d: Optional[Tensor] = None) -> Tensor:
    """
    Calculate stability parameter zeta = (z - d) / L.
    
    Parameters:
        z : Tensor
            Height above ground [m]
        L : Tensor
            Monin-Obukhov length [m]
        d : Optional[Tensor]
            Displacement height [m] (default: 0)
    
    Returns:
        zeta : Tensor
            Stability parameter [dimensionless]
    
    Stability Classification:
        zeta < -2: Very unstable
        -2 <= zeta < -0.1: Unstable
        -0.1 <= zeta < 0.1: Near neutral
        0.1 <= zeta < 2: Stable
        zeta >= 2: Very stable
    """
    displacement = torch.zeros_like(z) if d is None else d
    _validate_tensor_group(z, L=L, d=displacement)
    if bool(torch.any(L == 0.0)):
        raise ValueError("Monin-Obukhov length must be nonzero")
    z_eff = z - displacement
    if bool(torch.any(z_eff <= 0.0)):
        raise ValueError("MOST effective height must be strictly positive")
    return z_eff / L


def richardson_number(
    delta_theta: Tensor,
    delta_u: Tensor,
    delta_z: Tensor,
    theta_mean: Tensor,
    g: float = GRAVITY_ACCELERATION
) -> Tensor:
    """
    Calculate bulk Richardson number.
    
    Mathematical Expression:
        Ri = (g / theta) * (delta_theta / delta_z) / (delta_u / delta_z)^2
    
    Parameters:
        delta_theta : Tensor
            Potential temperature difference [K]
        delta_u : Tensor
            Wind speed difference [m s-1]
        delta_z : Tensor
            Height difference [m]
        theta_mean : Tensor
            Mean potential temperature [K]
        g : float
            Gravity acceleration (default: GRAVITY_ACCELERATION m s-2)
    
    Returns:
        Ri : Tensor
            Richardson number [dimensionless]
    
    Stability Classification:
        Ri < 0: Unstable
        0 <= Ri < 0.25: Neutral to slightly stable
        Ri >= 0.25: Stable (turbulence suppressed)
    
    References:
        - Richardson (1920). Proc. R. Soc. London Ser. A 97, 354-373
    """
    _validate_tensor_group(
        delta_theta,
        delta_u=delta_u,
        delta_z=delta_z,
        theta_mean=theta_mean,
    )
    if bool(torch.any(delta_z <= 0.0)):
        raise ValueError("delta_z must be strictly positive")
    if bool(torch.any(theta_mean <= 0.0)):
        raise ValueError("theta_mean must be strictly positive")
    if bool(torch.any(delta_u == 0.0)):
        raise ValueError("delta_u must be nonzero")
    g_tensor = _positive_scalar("g", g, delta_theta)

    # References: Richardson (1920).
    # Expression: Ri = (g/theta) (delta_theta/delta_z)/(delta_u/delta_z)^2.
    # Parameters: delta_theta [K], delta_u [m s-1], delta_z [m], theta [K].
    dtheta_dz = delta_theta / delta_z
    du_dz = delta_u / delta_z
    return (g_tensor / theta_mean) * dtheta_dz / (du_dz**2)


def zeta_from_richardson(Ri: Tensor) -> Tensor:
    """
    Convert Richardson number to stability parameter zeta.
    
    This function provides an approximate conversion from bulk Richardson
    number to the Monin-Obukhov stability parameter zeta (z/L).
    
    Mathematical Expression:
        For stable (Ri > 0):
            zeta = Ri / (1 - 5*Ri)  for Ri < 0.2
            zeta = 1.0              for Ri >= 0.2
        
        For unstable (Ri < 0):
            zeta = Ri
    
    Parameters:
        Ri : Tensor
            Richardson number [dimensionless]
    
    Returns:
        zeta : Tensor
            Stability parameter z/L [dimensionless]
    
    References:
        - Grachev et al. (2007). J. Geophys. Res. 112, D07101
        - Businger (1988). Topics in Micrometeorology
    """
    _validate_tensor_group(Ri)

    # References: Grachev et al. (2007); Businger (1988).
    # Expression: zeta = Ri for Ri <= 0 and Ri/(1-5 Ri) for 0 < Ri < 0.2.
    # Parameter: Ri is the dimensionless bulk Richardson number.
    stable_mask = Ri > 0
    unstable_mask = Ri <= 0
    
    # Initialize output
    zeta = torch.zeros_like(Ri)
    
    # Unstable: Direct approximation
    if unstable_mask.any():
        zeta = torch.where(unstable_mask, Ri, zeta)
    
    # Stable: Use empirical relation
    if stable_mask.any():
        # For moderate stability
        moderate_stable = stable_mask & (Ri < 0.2)
        if moderate_stable.any():
            zeta = torch.where(
                moderate_stable,
                Ri / (1.0 - 5.0 * Ri),
                zeta
            )
        
        # For strong stability, limit zeta
        strong_stable = stable_mask & (Ri >= 0.2)
        if strong_stable.any():
            zeta = torch.where(strong_stable, torch.ones((), dtype=Ri.dtype, device=Ri.device), zeta)
    
    return zeta


def temperature_scale(
    theta_surface: Tensor,
    theta_air: Tensor,
    z: Tensor,
    z0h: Tensor,
    d: Tensor,
    zeta: Tensor,
    k: float = VON_KARMAN_CONSTANT
) -> Tensor:
    """
    Calculate temperature scale theta*.
    
    Mathematical Expression:
        fh = ln((z-d)/z0h) - psi_h((z-d)/L) + psi_h(z0h/L)
        theta* = k * (theta_surface - theta_air) / fh
    
    Parameters:
        theta_surface : Tensor
            Surface potential temperature [K]
        theta_air : Tensor
            Air potential temperature [K]
        z : Tensor
            Reference height [m]
        z0h : Tensor
            Heat roughness length [m]
        d : Tensor
            Displacement height [m]
        zeta : Tensor
            Stability parameter [dimensionless]
        k : float
            von Karman constant
    
    Returns:
        theta_star : Tensor
            Temperature scale [K]

    References:
        - Paulson (1970). J. Appl. Meteorol. 9, 857-861
        - Businger et al. (1971). J. Atmos. Sci. 28, 181-189

    Open/closed system route:
        theta* is a diagnostic scale and does not itself write thermal state
        or create an energy source, sink, or boundary flux.
    """
    _validate_tensor_group(
        theta_surface,
        theta_air=theta_air,
        z=z,
        z0h=z0h,
        d=d,
        zeta=zeta,
    )
    if bool(torch.any(theta_surface <= 0.0)) or bool(torch.any(theta_air <= 0.0)):
        raise ValueError("potential temperatures must be strictly positive")
    z_eff = _validate_profile_geometry(z, z0h, d)
    k_tensor = _positive_scalar("k", k, theta_surface)

    # References: Paulson (1970); Businger et al. (1971).
    # Expression: theta* = k (theta_surface - theta_air) / fh.
    # Parameters: theta [K], z/d/z0h [m], zeta [1], k [1].
    heat_factor = _integrated_profile_factor(z_eff, z0h, zeta, psi_h)
    theta_star = k_tensor * (theta_surface - theta_air) / heat_factor
    if not bool(torch.isfinite(theta_star).all()):
        raise ValueError("MOST temperature scale must be finite")
    return theta_star


def iterative_most_solution(
    U: Tensor,
    theta_air: Tensor,
    theta_surface: Tensor,
    z: Tensor,
    z0m: Tensor,
    z0h: Tensor,
    d: Tensor,
    max_iterations: Optional[int] = None,
    tolerance: Optional[float] = None,
    k: float = VON_KARMAN_CONSTANT,
    *,
    rho: Union[Tensor, float],
    cp: float = SPECIFIC_HEAT_AIR,
    g: float = GRAVITY_ACCELERATION
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Iteratively solve coupled MOST equations for u*, L, and H.
    
    This function solves the nonlinear system of MOST equations through
    iteration to find consistent values of friction velocity and
    Monin-Obukhov length.
    
    Parameters:
        U : Tensor
            Wind speed [m s-1]
        theta_air : Tensor
            Air potential temperature [K]
        theta_surface : Tensor
            Surface potential temperature [K]
        z : Tensor
            Reference height [m]
        z0m : Tensor
            Momentum roughness length [m]
        z0h : Tensor
            Heat roughness length [m]
        d : Tensor
            Displacement height [m]
        max_iterations : Optional[int]
            Maximum number of iterations. None loads the canonical parameter.
        tolerance : Optional[float]
            Relative tolerance for both u* and L. None loads the canonical parameter.
        k : float
            von Karman constant
        rho : Tensor or float
            Actual (moist) air density [kg m-3] of the caller, keyword-only
            and required; there is no default density
        cp : float
            Specific heat [J kg^-1 K^-1]
        g : float
            Gravity [m s-2]
    
    Returns:
        u_star : Tensor
            Friction velocity [m s-1]
        L : Tensor
            Monin-Obukhov length [m]
        H : Tensor
            Sensible heat flux [W m-2]
    
    References:
        - Businger et al. (1971)
        - Paulson (1970). The mathematical representation of wind speed and temperature profiles

    Open/closed system route:
        The solve returns diagnostic scales and the explicit sensible-heat
        boundary flux. It never mutates state and never returns an unconverged
        last iterate.
    """
    iteration_limit = (
        int(_get_stability_parameter("max_iterations"))
        if max_iterations is None
        else max_iterations
    )
    convergence_tolerance = (
        _get_stability_parameter("convergence_tolerance")
        if tolerance is None
        else tolerance
    )
    if isinstance(iteration_limit, bool) or not isinstance(iteration_limit, int):
        raise ValueError("max_iterations must be a positive integer")
    if iteration_limit < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(convergence_tolerance, (float, int)):
        raise ValueError("tolerance must be a finite positive scalar")
    convergence_tolerance = float(convergence_tolerance)
    if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise ValueError("tolerance must be a finite positive scalar")

    _validate_tensor_group(
        U,
        theta_air=theta_air,
        theta_surface=theta_surface,
        z=z,
        z0m=z0m,
        z0h=z0h,
        d=d,
    )
    if bool(torch.any(U <= 0.0)):
        raise ValueError("MOST wind speed must be strictly positive")
    if bool(torch.any(theta_air <= 0.0)) or bool(torch.any(theta_surface <= 0.0)):
        raise ValueError("potential temperatures must be strictly positive")
    _validate_profile_geometry(z, z0m, d)
    _validate_profile_geometry(z, z0h, d)
    _positive_scalar("k", k, U)
    _positive_density(rho, U)
    _positive_scalar("cp", cp, U)
    _positive_scalar("g", g, U)

    # References: Paulson (1970); Businger et al. (1971).
    # Mathematical expression: u* = k U / fm, H = rho cp u* theta*,
    #   L = -rho cp T u*^3 / (k g H), iterated to convergence.
    # Parameter meanings: U wind speed, theta air/surface temperatures,
    #   z/d/z0 heights, rho the caller's actual air density, cp, k, g constants.
    # Units: U, u* [m s-1]; theta [K]; z, d, z0, L [m]; rho [kg m-3]; H [W m-2].
    # Domain/constraints: rho finite and > 0 and broadcastable with U; no
    #   default density; iteration budget and tolerance from ParameterManager.
    # Sign convention: H > 0 upward; L < 0 unstable, L > 0 stable.
    neutral_zeta = torch.zeros_like(U)
    u_star = friction_velocity(U, z, z0m, d, neutral_zeta, k)
    neutral_length = _get_stability_parameter("neutral_obukhov_length")
    L = torch.full_like(u_star, neutral_length)
    epsilon = torch.finfo(U.dtype).tiny

    for _ in range(iteration_limit):
        u_star_old = u_star
        L_old = L
        zeta_val = zeta(z, L_old, d)
        u_star = friction_velocity(U, z, z0m, d, zeta_val, k)
        theta_star = temperature_scale(
            theta_surface,
            theta_air,
            z,
            z0h,
            d,
            zeta_val,
            k,
        )
        H = sensible_heat_flux(u_star, theta_star, rho=rho, cp=cp)
        L = monin_obukhov_length(u_star, rho, cp, theta_air, H, k, g)

        u_scale = torch.maximum(torch.abs(u_star), torch.abs(u_star_old)).clamp_min(
            epsilon
        )
        L_scale = torch.maximum(torch.abs(L), torch.abs(L_old)).clamp_min(epsilon)
        u_relative_change = torch.abs(u_star - u_star_old) / u_scale
        L_relative_change = torch.abs(L - L_old) / L_scale
        converged = (u_relative_change <= convergence_tolerance) & (
            L_relative_change <= convergence_tolerance
        )
        if bool(torch.all(converged)):
            return u_star, L, H

    raise RuntimeError(
        "MOST iteration failed to converge within the configured iteration budget"
    )


# Export all functions
__all__ = [
    'friction_velocity',
    'monin_obukhov_length',
    'zeta',
    'richardson_number',
    'temperature_scale',
    'iterative_most_solution',
    'VON_KARMAN_CONSTANT',
    'GRAVITY_ACCELERATION',
    'SPECIFIC_HEAT_AIR'
]

