"""
Soil water retention conversions between matric potential and volumetric content.

References:
    van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic
    conductivity of unsaturated soils. Soil Sci. Soc. Am. J., 44(5), 892-898.
"""

import torch
from torch import Tensor


def matric_potential_to_volumetric(
    psi: Tensor,
    theta_sat: float,
    alpha: float,
    n: float,
) -> Tensor:
    """
    Convert matric potential (Pa, negative) to volumetric water content (m3 m-3).

    Mathematical Expression:
        m = 1 - 1/n
        S_e(psi) = [1 + (alpha * |psi|)^n]^(-m)
        theta(psi) = theta_sat * S_e(psi)

    Parameters
    ----------
    psi : Tensor
        Soil matric potential [Pa], negative values for suction.
    theta_sat : float
        Saturated volumetric water content [m3 m-3].
    alpha : float
        van Genuchten parameter alpha [Pa^-1], inverse air-entry pressure.
    n : float
        van Genuchten parameter n [dimensionless].
    """
    m = 1.0 - 1.0 / n
    se = (1.0 + (alpha * torch.abs(psi)) ** n) ** (-m)
    return torch.as_tensor(theta_sat, device=psi.device, dtype=psi.dtype) * se


def volumetric_to_matric_potential(
    theta: Tensor,
    theta_sat: float,
    alpha: float,
    n: float,
) -> Tensor:
    """
    Convert volumetric water content (m3 m-3) to matric potential (Pa, negative).

    Mathematical Expression:
        m = 1 - 1/n
        S_e = theta / theta_sat
        psi = -((S_e^(-1/m) - 1)^(1/n)) / alpha
    """
    m = 1.0 - 1.0 / n
    se = theta / torch.as_tensor(theta_sat, device=theta.device, dtype=theta.dtype)
    return -((se ** (-1.0 / m) - 1.0) ** (1.0 / n)) / alpha
