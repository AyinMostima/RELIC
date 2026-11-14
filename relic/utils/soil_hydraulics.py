"""
Unified soil hydraulic functions for RELIC.

This module provides shared implementations of van Genuchten and Mualem style
hydraulic functions to avoid duplicated logic across hydrosphere and pedosphere
modules.
"""

from typing import Optional, Union

import torch
from torch import Tensor

Numeric = Union[float, int, Tensor]


def _as_tensor(value: Numeric, like: Tensor) -> Tensor:
    """Return value as a tensor on the same device and dtype as the reference."""
    if isinstance(value, Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


# References: Van Genuchten (1980).
# Mathematical expression: m = 1 - 1/n.
# Parameter meanings: n_vg [-] van Genuchten shape parameter, m_vg [-] derived exponent.
def van_genuchten_m_parameter(n_vg: Numeric, like: Optional[Tensor] = None) -> Tensor:
    """
    Compute the van Genuchten derived shape parameter m from n.

    Parameters
    ----------
    n_vg : float or Tensor
        Van Genuchten n parameter [dimensionless].
    like : Tensor, optional
        Optional reference tensor for dtype/device alignment when n_vg is scalar.

    Returns
    -------
    Tensor
        Van Genuchten m parameter [dimensionless].
    """
    if isinstance(n_vg, Tensor):
        n_value = n_vg
    elif like is not None:
        n_value = _as_tensor(n_vg, like)
    else:
        n_value = torch.as_tensor(n_vg, dtype=torch.float32)
    return 1.0 - 1.0 / n_value


# References: Van Genuchten (1980).
# Mathematical expression: theta(h) = theta_r + (theta_s - theta_r) /
# (1 + (alpha * |h|)^n)^m, m = 1 - 1/n.
# Parameter meanings: pressure_head [m], alpha_vg [m-1], n_vg [-],
# theta_residual [m3 m-3], theta_saturated [m3 m-3].
def van_genuchten_water_retention(
    pressure_head: Tensor,
    alpha_vg: Numeric,
    n_vg: Numeric,
    theta_residual: Numeric,
    theta_saturated: Numeric,
) -> Tensor:
    """
    Van Genuchten water retention curve.

    Parameters
    ----------
    pressure_head : Tensor
        Soil water pressure head [m] (negative for unsaturated).
    alpha_vg : float or Tensor
        Van Genuchten alpha parameter [m-1].
    n_vg : float or Tensor
        Van Genuchten n parameter [dimensionless].
    theta_residual : float or Tensor
        Residual water content [m3 m-3].
    theta_saturated : float or Tensor
        Saturated water content [m3 m-3].

    Returns
    -------
    Tensor
        Volumetric water content [m3 m-3].

    References
    ----------
    - Van Genuchten, M.T. (1980). Soil Sci. Soc. Am. J., 44, 892-898.

    Scientific mathematical expressions
    -----------------------------------
    theta(h) = theta_r + (theta_s - theta_r) / (1 + (alpha * |h|)^n)^m,
    m = 1 - 1/n.
    """
    alpha = _as_tensor(alpha_vg, pressure_head)
    n_value = _as_tensor(n_vg, pressure_head)
    theta_r = _as_tensor(theta_residual, pressure_head)
    theta_s = _as_tensor(theta_saturated, pressure_head)

    m_value = van_genuchten_m_parameter(n_value)
    abs_h = torch.abs(pressure_head)
    denominator = torch.pow(1.0 + torch.pow(alpha * abs_h, n_value), m_value)
    theta = theta_r + (theta_s - theta_r) / denominator
    return theta


# References: Van Genuchten (1980).
# Mathematical expression: Se = (theta - theta_r) / (theta_s - theta_r).
# Parameter meanings: theta [m3 m-3], theta_residual [m3 m-3],
# theta_saturated [m3 m-3], epsilon [dimensionless].
def effective_saturation(
    theta: Tensor,
    theta_residual: Numeric,
    theta_saturated: Numeric,
    epsilon: float = 1e-10,
) -> Tensor:
    """
    Compute effective saturation.

    Parameters
    ----------
    theta : Tensor
        Volumetric water content [m3 m-3].
    theta_residual : float or Tensor
        Residual water content [m3 m-3].
    theta_saturated : float or Tensor
        Saturated water content [m3 m-3].
    epsilon : float, optional
        Small value to avoid division by zero.

    Returns
    -------
    Tensor
        Effective saturation [dimensionless], clamped to [epsilon, 1].

    References
    ----------
    - Van Genuchten, M.T. (1980). Soil Sci. Soc. Am. J., 44, 892-898.

    Scientific mathematical expressions
    -----------------------------------
    Se = (theta - theta_r) / (theta_s - theta_r).
    """
    theta_r = _as_tensor(theta_residual, theta)
    theta_s = _as_tensor(theta_saturated, theta)
    denom = theta_s - theta_r
    denom_safe = torch.clamp(denom, min=float(epsilon))
    Se = (theta - theta_r) / denom_safe
    return torch.clamp(Se, min=float(epsilon), max=1.0)


# References: Van Genuchten (1980).
# Mathematical expression: Se = (theta - theta_r) / (theta_s - theta_r),
# h = -1/alpha * (Se^(-1/m) - 1)^(1/n), m = 1 - 1/n.
# Parameter meanings: theta [m3 m-3], theta_residual [m3 m-3],
# theta_saturated [m3 m-3], alpha_vg [m-1], n_vg [dimensionless].
def van_genuchten_pressure_head(
    theta: Tensor,
    alpha_vg: Numeric,
    n_vg: Numeric,
    theta_residual: Numeric,
    theta_saturated: Numeric,
    epsilon: float = 1e-10,
) -> Tensor:
    """
    Invert van Genuchten water retention to compute pressure head.

    Parameters
    ----------
    theta : Tensor
        Volumetric water content [m3 m-3].
    alpha_vg : float or Tensor
        Van Genuchten alpha parameter [m-1].
    n_vg : float or Tensor
        Van Genuchten n parameter [dimensionless].
    theta_residual : float or Tensor
        Residual water content [m3 m-3].
    theta_saturated : float or Tensor
        Saturated water content [m3 m-3].
    epsilon : float, optional
        Small value to avoid division by zero.

    Returns
    -------
    Tensor
        Soil water pressure head [m] (negative for unsaturated).

    References
    ----------
    - Van Genuchten, M.T. (1980). Soil Sci. Soc. Am. J., 44, 892-898.

    Scientific mathematical expressions
    -----------------------------------
    Se = (theta - theta_r) / (theta_s - theta_r),
    h = -1/alpha * (Se^(-1/m) - 1)^(1/n), m = 1 - 1/n.
    """
    theta_t = _as_tensor(theta, theta)
    theta_r = _as_tensor(theta_residual, theta_t)
    theta_s = _as_tensor(theta_saturated, theta_t)
    alpha = _as_tensor(alpha_vg, theta_t)
    n_value = _as_tensor(n_vg, theta_t)

    denom = torch.clamp(theta_s - theta_r, min=float(epsilon))
    Se = (theta_t - theta_r) / denom
    Se = torch.clamp(Se, min=float(epsilon), max=1.0)
    m_value = van_genuchten_m_parameter(n_value)
    base = torch.pow(Se, -1.0 / m_value) - 1.0
    base = torch.clamp(base, min=0.0)
    return -(1.0 / alpha) * torch.pow(base, 1.0 / n_value)


# References: Van Genuchten (1980); Mualem (1976).
# Mathematical expression: K(Se) = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2,
# m = 1 - 1/n.
# Parameter meanings: effective_saturation [dimensionless], k_saturated [m s-1],
# n_vg [dimensionless], tortuosity [dimensionless], epsilon [dimensionless].
def van_genuchten_mualem_conductivity(
    effective_saturation: Tensor,
    k_saturated: Numeric,
    n_vg: Numeric,
    tortuosity: Numeric = 0.5,
    epsilon: float = 1e-10,
) -> Tensor:
    """
    Van Genuchten Mualem hydraulic conductivity.

    Parameters
    ----------
    effective_saturation : Tensor
        Effective saturation [dimensionless].
    k_saturated : float or Tensor
        Saturated hydraulic conductivity [m s-1].
    n_vg : float or Tensor
        Van Genuchten n parameter [dimensionless].
    tortuosity : float or Tensor, optional
        Mualem tortuosity parameter [dimensionless].
    epsilon : float, optional
        Small value for saturation clamping.

    Returns
    -------
    Tensor
        Hydraulic conductivity [m s-1].

    References
    ----------
    - Van Genuchten, M.T. (1980). Soil Sci. Soc. Am. J., 44, 892-898.
    - Mualem, Y. (1976). Water Resources Research, 12, 513-522.

    Scientific mathematical expressions
    -----------------------------------
    K(Se) = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2,
    m = 1 - 1/n.
    """
    Se = torch.clamp(
        effective_saturation, min=float(epsilon), max=1.0 - float(epsilon)
    )
    Ks = _as_tensor(k_saturated, Se)
    n_value = _as_tensor(n_vg, Se)
    l_value = _as_tensor(tortuosity, Se)
    m_value = van_genuchten_m_parameter(n_value)

    Se_power = torch.pow(Se, 1.0 / m_value)
    inner_term = 1.0 - torch.pow(1.0 - Se_power, m_value)
    inner_term = torch.clamp(inner_term, min=0.0, max=1.0)
    return Ks * torch.pow(Se, l_value) * torch.pow(inner_term, 2.0)
