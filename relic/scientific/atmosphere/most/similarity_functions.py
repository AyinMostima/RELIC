"""
Monin-Obukhov Similarity Functions Module

This module provides the standard stability functions (psi_m and psi_h) used in
Monin-Obukhov Similarity Theory (MOST) for momentum and heat/scalar transport.

Mathematical Expressions:
    Stable conditions (zeta >= 0):
        psi_m = psi_h = -5 * zeta
    
    Unstable conditions (zeta < 0):
        x = (1 - 16 * zeta)^(1/4)
        psi_m = 2*ln((1+x)/2) + ln((1+x^2)/2) - 2*arctan(x) + pi/2
        psi_h = 2*ln((1+x^2)/2)

References:
    - Businger, J.A. et al. (1971). Flux-profile relationships in the atmospheric surface layer
    - Dyer, A.J. (1974). A review of flux-profile relationships
    - Hogstrom, U. (1988). Non-dimensional wind and temperature profiles
    - Garratt, J.R. (1992). The Atmospheric Boundary Layer

Parameter Meanings:
    zeta: Stability parameter z/L [dimensionless]
    psi_m: Momentum stability function [dimensionless]
    psi_h: Heat/scalar stability function [dimensionless]
"""

import torch
import math

from relic.parameters import get_global_manager
from ....parameters.namespaces import ParameterNamespace, resolve_namespace

# Type alias
Tensor = torch.Tensor
MOST_NAMESPACE = resolve_namespace(ParameterNamespace.ATMOSPHERE_MOST)


def _to_float(value: object) -> float:
    """Convert scalar values from ParameterManager to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].tolist())
    return float(value)


def _load_coefficient(name: str) -> float:
    """Load one required MOST coefficient from its canonical parameter key."""
    full_key = name if name.startswith(MOST_NAMESPACE) else f"{MOST_NAMESPACE}.{name}"
    value = get_global_manager().get(
        full_key,
        units="dimensionless",
        expected_type=(float, int),
        required=True,
    )
    return _to_float(value)


def _most_coefficients() -> dict:
    """Load MOST similarity coefficients from the parameter system."""
    return {
        "beta_stable": _load_coefficient("beta_stable"),
        "gamma_unstable": _load_coefficient("gamma_unstable"),
        "stable_limit": _load_coefficient("stable_correction_limit"),
        "unstable_limit": _load_coefficient("unstable_correction_limit"),
    }


def _validate_zeta(zeta: Tensor) -> dict:
    """Validate the configured Businger-Dyer application domain."""
    if not isinstance(zeta, torch.Tensor):
        raise TypeError("zeta must be a torch.Tensor")
    if not zeta.is_floating_point():
        raise ValueError("zeta must use a floating-point dtype")
    if not bool(torch.all(torch.isfinite(zeta))):
        raise ValueError("zeta must contain only finite values")

    coefficients = _most_coefficients()
    stable_limit = coefficients["stable_limit"]
    unstable_limit = coefficients["unstable_limit"]
    if (
        not math.isfinite(stable_limit)
        or not math.isfinite(unstable_limit)
        or stable_limit <= 0.0
        or unstable_limit <= 0.0
    ):
        raise ValueError("Businger-Dyer validity limits must be finite and positive")
    if bool(torch.any((zeta < -unstable_limit) | (zeta > stable_limit))):
        raise ValueError(
            "zeta is outside the configured Businger-Dyer validity domain"
        )
    return coefficients


def psi_m(zeta: Tensor) -> Tensor:
    """
    Calculate momentum stability function using Businger-Dyer formulation.
    
    Mathematical Expression:
        Stable (zeta >= 0): psi_m = -5 * zeta
        Unstable (zeta < 0): psi_m = 2*ln((1+x)/2) + ln((1+x^2)/2) - 2*arctan(x) + pi/2
                            where x = (1 - 16*zeta)^(1/4)
    
    Parameters:
        zeta : Tensor
            Stability parameter z/L [dimensionless]
    Returns:
        psi_m : Tensor
            Momentum stability function [dimensionless]
    
    References:
        - Businger et al. (1971). J. Atmos. Sci. 28, 181-189
        - Dyer (1974). Boundary-Layer Meteorol. 7, 363-372

    Open/closed system route:
        This dimensionless constitutive relation does not mutate state or
        create a material or energy source. Out-of-domain forcing is rejected.
    """
    coeffs = _validate_zeta(zeta)
    beta_m = zeta.new_tensor(coeffs["beta_stable"])
    gamma_m = zeta.new_tensor(coeffs["gamma_unstable"])
    
    # Initialize output
    psi = torch.zeros_like(zeta)
    
    # References: Businger et al. (1971); Dyer (1974).
    # Expression: stable psi_m=-beta*zeta; unstable x=(1-gamma*zeta)^0.25.
    # Parameters: zeta, beta, and gamma are dimensionless.
    # Stable conditions
    stable_mask = zeta >= 0
    if torch.any(stable_mask):
        psi[stable_mask] = -beta_m * zeta[stable_mask]
    
    # Unstable conditions
    unstable_mask = zeta < 0
    if torch.any(unstable_mask):
        zeta_unstable = zeta[unstable_mask]
        x = torch.pow(1.0 - gamma_m * zeta_unstable, 0.25)
        
        psi_unstable = (2.0 * torch.log((1.0 + x) / 2.0) +
                       torch.log((1.0 + x * x) / 2.0) -
                       2.0 * torch.atan(x) + math.pi / 2.0)
        psi[unstable_mask] = psi_unstable
    
    return psi


def psi_h(zeta: Tensor) -> Tensor:
    """
    Calculate heat/scalar stability function using Businger-Dyer formulation.
    
    Mathematical Expression:
        Stable (zeta >= 0): psi_h = -5 * zeta
        Unstable (zeta < 0): psi_h = 2*ln((1+x^2)/2)
                            where x = (1 - 16*zeta)^(1/4)
    
    Parameters:
        zeta : Tensor
            Stability parameter z/L [dimensionless]
    Returns:
        psi_h : Tensor
            Heat/scalar stability function [dimensionless]
    
    References:
        - Businger et al. (1971). J. Atmos. Sci. 28, 181-189
        - Dyer (1974). Boundary-Layer Meteorol. 7, 363-372

    Open/closed system route:
        This dimensionless constitutive relation does not mutate state or
        create a material or energy source. Out-of-domain forcing is rejected.
    """
    coeffs = _validate_zeta(zeta)
    beta_h = zeta.new_tensor(coeffs["beta_stable"])
    gamma_h = zeta.new_tensor(coeffs["gamma_unstable"])
    
    # Initialize output
    psi = torch.zeros_like(zeta)
    
    # References: Businger et al. (1971); Dyer (1974).
    # Expression: stable psi_h=-beta*zeta; unstable psi_h=2*ln((1+x^2)/2).
    # Parameters: zeta, beta, and gamma are dimensionless.
    # Stable conditions
    stable_mask = zeta >= 0
    if torch.any(stable_mask):
        psi[stable_mask] = -beta_h * zeta[stable_mask]
    
    # Unstable conditions
    unstable_mask = zeta < 0
    if torch.any(unstable_mask):
        zeta_unstable = zeta[unstable_mask]
        x = torch.pow(1.0 - gamma_h * zeta_unstable, 0.25)
        
        psi_unstable = 2.0 * torch.log((1.0 + x * x) / 2.0)
        psi[unstable_mask] = psi_unstable
    
    return psi


def phi_m(zeta: Tensor) -> Tensor:
    """
    Calculate dimensionless wind shear phi_m = (kz/u*) * du/dz.
    
    Mathematical Expression:
        Stable: phi_m = 1 + 5*zeta
        Unstable: phi_m = (1 - 16*zeta)^(-1/4)
    
    Parameters:
        zeta : Tensor
            Stability parameter z/L [dimensionless]
    
    Returns:
        phi_m : Tensor
            Dimensionless wind shear [dimensionless]
    
    References:
        - Businger et al. (1971)
    """
    coeffs = _validate_zeta(zeta)
    phi = torch.ones_like(zeta)
    beta_m = zeta.new_tensor(coeffs["beta_stable"])
    gamma_m = zeta.new_tensor(coeffs["gamma_unstable"])
    
    # References: Businger et al. (1971); Dyer (1974).
    # Expression: phi_m=1+beta*zeta (stable), (1-gamma*zeta)^-0.25 (unstable).
    # Parameters: zeta, beta, and gamma are dimensionless.
    # Stable conditions
    stable_mask = zeta >= 0
    if torch.any(stable_mask):
        phi[stable_mask] = 1.0 + beta_m * zeta[stable_mask]
    
    # Unstable conditions
    unstable_mask = zeta < 0
    if torch.any(unstable_mask):
        phi[unstable_mask] = torch.pow(1.0 - gamma_m * zeta[unstable_mask], -0.25)
    
    return phi


def phi_h(zeta: Tensor) -> Tensor:
    """
    Calculate dimensionless temperature gradient phi_h = (kz/theta*) * dtheta/dz.
    
    Mathematical Expression:
        Stable: phi_h = 1 + 5*zeta
        Unstable: phi_h = (1 - 16*zeta)^(-1/2)
    
    Parameters:
        zeta : Tensor
            Stability parameter z/L [dimensionless]
    
    Returns:
        phi_h : Tensor
            Dimensionless temperature gradient [dimensionless]
    
    References:
        - Businger et al. (1971)
    """
    coeffs = _validate_zeta(zeta)
    phi = torch.ones_like(zeta)
    beta_h = zeta.new_tensor(coeffs["beta_stable"])
    gamma_h = zeta.new_tensor(coeffs["gamma_unstable"])
    
    # References: Businger et al. (1971); Dyer (1974).
    # Expression: phi_h=1+beta*zeta (stable), (1-gamma*zeta)^-0.5 (unstable).
    # Parameters: zeta, beta, and gamma are dimensionless.
    # Stable conditions
    stable_mask = zeta >= 0
    if torch.any(stable_mask):
        phi[stable_mask] = 1.0 + beta_h * zeta[stable_mask]
    
    # Unstable conditions
    unstable_mask = zeta < 0
    if torch.any(unstable_mask):
        phi[unstable_mask] = torch.pow(1.0 - gamma_h * zeta[unstable_mask], -0.5)

    return phi


# Export all functions
__all__ = [
    'psi_m',
    'psi_h',
    'phi_m',
    'phi_h'
]

