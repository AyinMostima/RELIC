"""Composable coupling operators for the RELIC coupling layer.

This module provides small, reusable operators that can be composed in coupling
pipelines without changing scientific Functor execution paths.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from ...parameters import get_global_manager


def _safe_get_parameter(key: str, default: Any) -> Any:
    """Read parameter value from ParameterManager with signature compatibility."""
    manager = get_global_manager()
    try:
        value = manager.get(key, required=False)
    except TypeError:
        value = manager.get(key)
    return default if value is None else value


class TemporalLagOperator:
    """Apply temporal lag interpolation or exponential smoothing.

    Mathematical Expression:
        Linear interpolation:
            y_interp(t) = y_prev + w * (y_next - y_prev),
            w = (t - t_prev) / (t_next - t_prev)
        Exponential smoothing:
            y_smooth(t) = alpha * y_current + (1 - alpha) * y_prev

    References:
        - Balaji et al. (2006), J. Climate, coupling interval effects.

    Parameters:
        method: "linear" or "exponential"
        alpha: smoothing weight in [0, 1] for exponential mode [dimensionless]
    """

    def __init__(
        self,
        method: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> None:
        self.method = str(
            method
            or _safe_get_parameter("coupling.operators.temporal_lag.method", "linear")
        ).strip().lower()
        self.alpha = float(
            alpha
            if alpha is not None
            else _safe_get_parameter("coupling.operators.temporal_lag.alpha", 0.5)
        )
        self.alpha = max(0.0, min(1.0, self.alpha))
        if self.method not in {"linear", "exponential"}:
            raise ValueError(
                f"Unsupported temporal lag method '{self.method}'. "
                "Expected 'linear' or 'exponential'."
            )

    def apply(
        self,
        previous_value: Tensor,
        current_value: Tensor,
        *,
        t_prev: float,
        t_current: float,
        t_target: float,
    ) -> Dict[str, Any]:
        """Execute temporal lag operator and return result with diagnostics."""
        if self.method == "linear":
            denom = max(float(t_current - t_prev), 1e-12)
            weight = (float(t_target) - float(t_prev)) / denom
            weight = max(0.0, min(1.0, weight))
            result = previous_value + weight * (current_value - previous_value)
            diagnostics = {"method": "linear", "weight": weight}
            return {"result": result, "diagnostics": diagnostics}

        result = self.alpha * current_value + (1.0 - self.alpha) * previous_value
        diagnostics = {"method": "exponential", "alpha": self.alpha}
        return {"result": result, "diagnostics": diagnostics}


class ConstraintProjector:
    """Project values to constraint manifold.

    Mathematical Expression:
        Non-negative:
            y_proj = max(0, y)
        Bounded:
            y_proj = clamp(y, lb, ub)
        Simplex:
            y_proj = y / sum(y) if sum(y) > 0

    References:
        - Lauritzen et al. (2011), Geosci. Model Dev., shape-preserving transport.

    Parameters:
        constraint_type: "non_negative", "bounded", or "simplex"
        bounds: optional (lb, ub) tuple for bounded projection
    """

    def __init__(
        self,
        constraint_type: Optional[str] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.default_constraint_type = str(
            constraint_type
            or _safe_get_parameter(
                "coupling.operators.constraint_projector.default_type",
                "non_negative",
            )
        ).strip().lower()
        self.bounds = bounds
        if self.default_constraint_type not in {"non_negative", "bounded", "simplex"}:
            raise ValueError(
                f"Unsupported constraint type '{self.default_constraint_type}'."
            )

    def apply(
        self,
        values: Tensor,
        *,
        constraint_type: Optional[str] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Apply projection and return projected field plus diagnostics."""
        resolved_type = str(constraint_type or self.default_constraint_type).strip().lower()
        resolved_bounds = bounds or self.bounds

        if resolved_type == "non_negative":
            projected = torch.clamp_min(values, 0.0)
            clipped = int((values < 0.0).sum().item())
            return {
                "result": projected,
                "diagnostics": {"constraint_type": resolved_type, "clipped_count": clipped},
            }

        if resolved_type == "bounded":
            if resolved_bounds is None:
                raise ValueError("bounds are required for bounded constraint projection.")
            lower, upper = float(resolved_bounds[0]), float(resolved_bounds[1])
            projected = torch.clamp(values, min=lower, max=upper)
            clipped = int(((values < lower) | (values > upper)).sum().item())
            return {
                "result": projected,
                "diagnostics": {
                    "constraint_type": resolved_type,
                    "lower": lower,
                    "upper": upper,
                    "clipped_count": clipped,
                },
            }

        if resolved_type == "simplex":
            positive = torch.clamp_min(values, 0.0)
            total = positive.sum()
            eps = torch.tensor(1e-12, dtype=positive.dtype, device=positive.device)
            projected = positive / torch.maximum(total, eps)
            return {
                "result": projected,
                "diagnostics": {
                    "constraint_type": resolved_type,
                    "sum_after_projection": float(projected.sum().detach().cpu().item()),
                },
            }

        raise ValueError(f"Unsupported constraint type '{resolved_type}'.")


class FluxKernel:
    """Compute coupling flux between compartments with parameterized coefficients.

    Mathematical Expression:
        F = k * (C_source - C_target) * A_interface
        where:
            k is transfer coefficient [m s-1]
            A_interface is interface area [m2]

    References:
        - Clark et al. (2015), Water Resources Research, inter-compartment fluxes.

    Parameters:
        transfer_coefficient_key: parameter key for k
        area_key: parameter key for interface area
    """

    def __init__(
        self,
        transfer_coefficient_key: str = "coupling.operators.flux_kernel.transfer_coefficient",
        area_key: str = "coupling.operators.flux_kernel.interface_area",
    ) -> None:
        self.transfer_coefficient_key = transfer_coefficient_key
        self.area_key = area_key

    def apply(
        self,
        source_concentration: Tensor,
        target_concentration: Tensor,
        *,
        interface_area: Optional[Tensor] = None,
        transfer_coefficient: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute flux tensor and diagnostics."""
        if transfer_coefficient is None:
            k_value = _safe_get_parameter(self.transfer_coefficient_key, 1.0)
            transfer_coefficient = torch.as_tensor(
                k_value,
                dtype=source_concentration.dtype,
                device=source_concentration.device,
            )
        if interface_area is None:
            area_value = _safe_get_parameter(self.area_key, 1.0)
            interface_area = torch.as_tensor(
                area_value,
                dtype=source_concentration.dtype,
                device=source_concentration.device,
            )

        gradient = source_concentration - target_concentration
        flux = transfer_coefficient * gradient * interface_area
        return {
            "result": flux,
            "diagnostics": {
                "transfer_coefficient_key": self.transfer_coefficient_key,
                "area_key": self.area_key,
                "mean_flux": float(flux.detach().mean().cpu().item()),
            },
        }


__all__ = [
    "TemporalLagOperator",
    "ConstraintProjector",
    "FluxKernel",
]

