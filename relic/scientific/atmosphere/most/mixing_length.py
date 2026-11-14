"""
Mixing Length Module for Turbulence

This module provides functions for calculating mixing length scales and
K-theory diffusivities used in atmospheric boundary layer modeling.

Mathematical Expressions:
    Prandtl mixing length: l = k * z
    K-theory diffusivity: K = k * u* * z / phi(z/L)
    Blackadar mixing length: l = k * z / (1 + k * z / lambda)

References:
    - Prandtl, L. (1925). Bericht uber Untersuchungen zur ausgebildeten Turbulenz
    - Blackadar, A.K. (1962). The vertical distribution of wind and turbulent exchange
    - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology

Parameter Meanings:
    l: Mixing length [m]
    K: Eddy diffusivity [m^2 s^-1]
    k: von Karman constant (VON_KARMAN_CONSTANT)
    lambda: Asymptotic mixing length [m]
"""

import logging
from functools import lru_cache
from typing import Dict, Optional

import torch
from .similarity_functions import phi_m, phi_h
from .stability_parameters import VON_KARMAN_CONSTANT
from relic.parameters import get_global_manager
from ....parameters.namespaces import ParameterNamespace, resolve_namespace

logger = logging.getLogger(__name__)


MOST_NAMESPACE = resolve_namespace(ParameterNamespace.ATMOSPHERE_MOST)


def _require_most_parameter(name: str, units: Optional[str] = None) -> float:
    """
    Retrieve a MOST parameter from the ParameterManager without default.

    Parameters
    ----------
    name : str
        Parameter key relative to the atmosphere.most namespace.
    units : Optional[str]
        Expected units for validation.

    Returns
    -------
    float
        Parameter value.
    """
    full_key = name if name.startswith(MOST_NAMESPACE) else f"{MOST_NAMESPACE}.{name}"
    value = get_global_manager().get(
        full_key,
        units=units,
        expected_type=(float, int),
    )
    return float(value)


@lru_cache(maxsize=1)
def _load_most_parameters() -> Dict[str, float]:
    """Load and cache MOST parameters to avoid repeated lookups."""
    return {
        "c_mu": _require_most_parameter("c_mu", units="dimensionless"),
        "lambda_blackadar": _require_most_parameter("lambda_blackadar", units="m"),
        "canopy_beta": _require_most_parameter("canopy_beta", units="dimensionless"),
        "c_l_local_mixing": _require_most_parameter("c_l_local_mixing", units="dimensionless"),
        "clamp_min_K": _require_most_parameter("clamp_min_K", units="m2 s-1"),
        "clamp_min_mixing_length": _require_most_parameter("clamp_min_mixing_length", units="m"),
        "canopy_displacement_ratio": _require_most_parameter("canopy_displacement_ratio", units="dimensionless"),
    }


def _get_most_parameter(key: str) -> float:
    """Return a cached MOST parameter value."""
    params = _load_most_parameters()
    return params[key]

# Type alias
Tensor = torch.Tensor


def prandtl_mixing_length(
    z: Tensor,
    k: float = VON_KARMAN_CONSTANT,
    d: Optional[Tensor] = None
) -> Tensor:
    """
    Calculate Prandtl mixing length.
    
    Mathematical Expression:
        l = k * (z - d)
    
    This is the classical mixing length for neutral conditions
    in the surface layer.
    
    Parameters:
        z : Tensor
            Height above ground [m]
        k : float
            von Karman constant (default: VON_KARMAN_CONSTANT)
        d : Optional[Tensor]
            Displacement height [m]
    
    Returns:
        l : Tensor
            Mixing length [m]
    
    References:
        - Prandtl (1925). Z. Angew. Math. Mech. 5, 136-139
    """
    if d is not None:
        z_eff = z - d
    else:
        z_eff = z
    
    return k * z_eff


def blackadar_mixing_length(
    z: Tensor,
    lambda_bl: Optional[float] = None,
    k: float = VON_KARMAN_CONSTANT,
    d: Optional[Tensor] = None
) -> Tensor:
    """
    Calculate Blackadar mixing length with asymptotic limit.
    
    Mathematical Expression:
        l = k * z / (1 + k * z / lambda)
    
    This formulation limits the mixing length at higher elevations,
    providing a smooth transition from surface layer to free atmosphere.
    
    Parameters:
        z : Tensor
            Height above ground [m]
        lambda_bl : float
            Asymptotic mixing length [m] (default: 150)
        k : float
            von Karman constant (default: VON_KARMAN_CONSTANT)
        d : Optional[Tensor]
            Displacement height [m]
    
    Returns:
        l : Tensor
            Mixing length [m]
    
    References:
        - Blackadar (1962). J. Geophys. Res. 67, 3095-3102
    """
    if d is not None:
        z_eff = z - d
    else:
        z_eff = z
    
    lambda_value = _get_most_parameter("lambda_blackadar") if lambda_bl is None else lambda_bl

    l = k * z_eff / (1.0 + k * z_eff / lambda_value)
    
    return l


def stability_corrected_mixing_length(
    z: Tensor,
    zeta: Tensor,
    k: float = VON_KARMAN_CONSTANT,
    d: Optional[Tensor] = None
) -> Tensor:
    """
    Calculate mixing length with stability corrections.
    
    Mathematical Expression:
        l = k * (z - d) / phi_m(z/L)
    
    Parameters:
        z : Tensor
            Height above ground [m]
        zeta : Tensor
            Stability parameter (z-d)/L [dimensionless]
        k : float
            von Karman constant
        d : Optional[Tensor]
            Displacement height [m]
    
    Returns:
        l : Tensor
            Stability-corrected mixing length [m]
    
    References:
        - Businger et al. (1971)
    """
    if d is not None:
        z_eff = z - d
    else:
        z_eff = z
    
    # Calculate dimensionless wind shear
    phi_m_val = phi_m(zeta)
    
    # Mixing length with stability correction
    l = k * z_eff / phi_m_val
    
    return l


def k_theory_K(
    u_star: Tensor,
    z: Tensor,
    zeta: Tensor,
    scalar_type: str,
    k: float = VON_KARMAN_CONSTANT,
    d: Optional[Tensor] = None
) -> Tensor:
    """
    Calculate K-theory eddy diffusivity.
    
    Mathematical Expression:
        Km = k * u* * (z - d) / phi_m(z/L)  (momentum)
        Kh = k * u* * (z - d) / phi_h(z/L)  (heat/scalars)
    
    Parameters:
        u_star : Tensor
            Friction velocity [m s-1]
        z : Tensor
            Height above ground [m]
        zeta : Tensor
            Stability parameter [dimensionless]
        scalar_type : str
            Stability-function selection (required): 'momentum' uses phi_m;
            'heat' and 'scalar' use phi_h; any other value raises ValueError
        k : float
            von Karman constant
        d : Optional[Tensor]
            Displacement height [m]
    
    Returns:
        K : Tensor
            Eddy diffusivity [m^2 s^-1]
    
    References:
        - Businger et al. (1971)
        - Hogstrom (1988)
    """
    if d is not None:
        z_eff = z - d
    else:
        z_eff = z
    
    # Calculate appropriate phi function
    if scalar_type == 'momentum':
        phi_val = phi_m(zeta)
    elif scalar_type in ('heat', 'scalar'):
        phi_val = phi_h(zeta)
    else:
        raise ValueError(
            f"scalar_type must be 'momentum', 'heat' or 'scalar', got {scalar_type!r}"
        )
    
    # K-theory diffusivity
    K = k * u_star * z_eff / phi_val
    
    # Apply minimum constraint from parameter system.
    return torch.clamp(K, min=_get_most_parameter("clamp_min_K"))


def tke_mixing_length(
    tke: Tensor,
    dissipation_rate: Tensor,
    c_mu: Optional[float] = None
) -> Tensor:
    """
    Calculate mixing length from turbulent kinetic energy (TKE).
    
    Mathematical Expression:
        l = c_mu^(3/4) * k^(3/2) / epsilon
    
    This is used in TKE-epsilon closure schemes.
    
    Parameters:
        tke : Tensor
            Turbulent kinetic energy [m^2 s^-2]
        dissipation_rate : Tensor
            TKE dissipation rate [m^2 s^-3]
        c_mu : float
            Model constant (default: 0.09)
    
    Returns:
        l : Tensor
            Mixing length [m]
    
    References:
        - Launder & Spalding (1974). Comput. Methods Appl. Mech. Eng. 3, 269-289
    """
    # Avoid division by zero
    eps_safe = torch.where(torch.abs(dissipation_rate) < 1e-10,
                           torch.sign(dissipation_rate) * 1e-10,
                           dissipation_rate)
    
    c_mu_value = _get_most_parameter("c_mu") if c_mu is None else c_mu

    l = c_mu_value**(3 / 4) * tke**(3 / 2) / eps_safe
    
    return torch.clamp(l, min=_get_most_parameter("clamp_min_mixing_length"))


def canopy_mixing_length(
    z: Tensor,
    h_c: Tensor,
    LAI: Tensor,
    beta: Optional[float] = None
) -> Tensor:
    """
    Calculate mixing length within and above vegetation canopy.
    
    Mathematical Expression:
        Within canopy (z < h_c): l = beta * (h_c - d)
        Above canopy (z >= h_c): l = k * (z - d)
    
    Parameters:
        z : Tensor
            Height above ground [m]
        h_c : Tensor
            Canopy height [m]
        LAI : Tensor
            Leaf area index [m^2 m^-2]
        beta : float
            Canopy mixing length coefficient (default: 0.3)
    
    Returns:
        l : Tensor
            Mixing length [m]
    
    References:
        - Wilson & Shaw (1977). J. Appl. Meteorol. 16, 1197-1205
        - Raupach et al. (1996). Boundary-Layer Meteorol. 78, 351-382
    """
    # Estimate displacement height from canopy parameters
    d = _get_most_parameter("canopy_displacement_ratio") * h_c
    
    # Initialize mixing length
    l = torch.zeros_like(z)
    
    beta_value = _get_most_parameter("canopy_beta") if beta is None else beta

    # Within canopy
    within_canopy = z < h_c
    if torch.any(within_canopy):
        l[within_canopy] = beta_value * (h_c[within_canopy] - d[within_canopy])
    
    # Above canopy
    above_canopy = z >= h_c
    if torch.any(above_canopy):
        l[above_canopy] = VON_KARMAN_CONSTANT * (z[above_canopy] - d[above_canopy])
    
    return torch.clamp(l, min=_get_most_parameter("clamp_min_mixing_length"))


def local_mixing_length(
    z: Tensor,
    N_squared: Tensor,
    tke: Tensor,
    c_l: Optional[float] = None
) -> Tensor:
    """
    Calculate local mixing length for stable stratification.
    
    Mathematical Expression:
        l = c_l * sqrt(k) / sqrt(N^2)
    
    where N^2 is the Brunt-Vaisala frequency squared.
    
    Parameters:
        z : Tensor
            Height above ground [m]
        N_squared : Tensor
            Brunt-Vaisala frequency squared [s^-2]
        tke : Tensor
            Turbulent kinetic energy [m^2 s^-2]
        c_l : float
            Model constant (default: 0.76)
    
    Returns:
        l : Tensor
            Local mixing length [m]
    
    References:
        - Delage (1974). Boundary-Layer Meteorol. 7, 109-121
    """
    # Avoid division by zero
    tiny = torch.full_like(N_squared, torch.finfo(N_squared.dtype).tiny)
    N2_safe = torch.where(N_squared < tiny, tiny, N_squared)
    
    c_l_value = _get_most_parameter("c_l_local_mixing") if c_l is None else c_l

    l = c_l_value * torch.sqrt(tke) / torch.sqrt(N2_safe)
    
    # Also apply Prandtl limit
    l_prandtl = prandtl_mixing_length(z)
    
    # Take minimum of local and Prandtl lengths
    return torch.minimum(l, l_prandtl)


# Export all functions
__all__ = [
    'prandtl_mixing_length',
    'blackadar_mixing_length',
    'stability_corrected_mixing_length',
    'k_theory_K',
    'tke_mixing_length',
    'canopy_mixing_length',
    'local_mixing_length'
]


