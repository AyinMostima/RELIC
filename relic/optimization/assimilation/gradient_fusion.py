"""Dual-track gradient fusion for assimilation workflows.

This module fuses gradients from physics-based automatic differentiation and
statistical ensemble estimation without introducing a new assimilation method.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from ...parameters import get_global_manager


class DualTrackGradientFusion:
    """Fuse physical and statistical gradients with uncertainty-aware weighting.

    Mathematical Expression:
        Track A (physics AD): g_phys = dJ/dx
        Track B (statistical): g_stat from ensemble approximation

        Inverse variance weighting:
            w_phys = 1 / var_phys
            w_stat = 1 / var_stat
            g_fused = (w_phys * g_phys + w_stat * g_stat) / (w_phys + w_stat)
            sigma_fused = sqrt(1 / (w_phys + w_stat))

    References:
        - Bocquet et al. (2020), Nonlin. Processes Geophys.
        - Farchi and Bocquet (2018), Geosci. Model Dev.

    Parameters (required YAML keys, no code defaults):
        physics_weight_mode: "inverse_variance", "fixed", or "adaptive"
        fixed_physics_weight: weight of the physics gradient in fixed mode,
            in [0, 1] (outside values are rejected, not clamped)
    """

    def __init__(self) -> None:
        manager = get_global_manager()
        mode = manager.get(
            "optimization.gradient_fusion.physics_weight_mode",
            required=True,
        )
        fixed_weight = manager.get(
            "optimization.gradient_fusion.fixed_physics_weight",
            required=True,
        )

        self.physics_weight_mode = str(mode).strip().lower()
        self.fixed_physics_weight = float(fixed_weight)
        if not 0.0 <= self.fixed_physics_weight <= 1.0:
            raise ValueError(
                "optimization.gradient_fusion.fixed_physics_weight must lie in "
                f"[0, 1], got {self.fixed_physics_weight}"
            )

        allowed_modes = {"inverse_variance", "fixed", "adaptive"}
        if self.physics_weight_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported physics_weight_mode '{self.physics_weight_mode}'. "
                f"Allowed values: {sorted(allowed_modes)}."
            )

    @staticmethod
    def _safe_variance(value: Tensor) -> Tensor:
        """Compute non-zero scalar variance estimate from a tensor.

        A single value has no sample variance; an explicit uncertainty is
        then required (no unit variance is assumed).
        """
        flat = value.reshape(-1).to(dtype=torch.float64)
        if flat.numel() <= 1:
            raise ValueError(
                "a scalar gradient needs an explicit uncertainty; its variance "
                "cannot be estimated from one value"
            )
        var = torch.var(flat, unbiased=False)
        eps = torch.as_tensor(1e-12, dtype=torch.float64, device=flat.device)
        return torch.maximum(var, eps)

    @staticmethod
    def _resolve_uncertainty(
        gradient: Tensor,
        uncertainty: Tensor | None,
    ) -> Tensor:
        """Resolve uncertainty tensor or derive from gradient variance."""
        if uncertainty is None:
            return DualTrackGradientFusion._safe_variance(gradient)
        if uncertainty.numel() == 1:
            return torch.maximum(
                uncertainty.reshape(()).to(dtype=torch.float64),
                torch.as_tensor(1e-12, dtype=torch.float64, device=uncertainty.device),
            )
        return DualTrackGradientFusion._safe_variance(uncertainty)

    def _adaptive_physics_weight(self, grad_physics: Tensor, grad_statistical: Tensor) -> float:
        """Compute adaptive physics weight from gradient directional consistency."""
        phys = grad_physics.reshape(-1).to(dtype=torch.float64)
        stat = grad_statistical.reshape(-1).to(dtype=torch.float64)
        eps = torch.as_tensor(1e-12, dtype=torch.float64, device=phys.device)
        phys_norm = torch.linalg.norm(phys)
        stat_norm = torch.linalg.norm(stat)
        if float(phys_norm.item()) <= 0.0 or float(stat_norm.item()) <= 0.0:
            return self.fixed_physics_weight
        cosine_similarity = torch.dot(phys, stat) / torch.maximum(phys_norm * stat_norm, eps)
        cosine_value = float(torch.clamp(cosine_similarity, -1.0, 1.0).item())
        adaptive_weight = 0.5 + 0.25 * cosine_value
        return max(0.0, min(1.0, adaptive_weight))

    def fuse(
        self,
        grad_physics: Dict[str, Tensor],
        grad_statistical: Dict[str, Tensor],
        physics_uncertainty: Dict[str, Tensor],
        statistical_uncertainty: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Return fused gradients and confidence intervals per variable."""
        fused_gradients: Dict[str, Tensor] = {}
        confidence_intervals: Dict[str, Tensor] = {}

        all_keys = set(grad_physics.keys()) | set(grad_statistical.keys())
        for key in sorted(all_keys):
            grad_p = grad_physics.get(key)
            grad_s = grad_statistical.get(key)

            if grad_p is None and grad_s is None:
                continue
            if grad_p is None:
                fused_gradients[key] = grad_s  # type: ignore[assignment]
                sigma_s = self._resolve_uncertainty(grad_s, statistical_uncertainty.get(key))
                confidence_intervals[key] = torch.sqrt(sigma_s).to(dtype=grad_s.dtype)
                continue
            if grad_s is None:
                fused_gradients[key] = grad_p
                sigma_p = self._resolve_uncertainty(grad_p, physics_uncertainty.get(key))
                confidence_intervals[key] = torch.sqrt(sigma_p).to(dtype=grad_p.dtype)
                continue

            var_p = self._resolve_uncertainty(grad_p, physics_uncertainty.get(key))
            var_s = self._resolve_uncertainty(grad_s, statistical_uncertainty.get(key))
            eps = torch.as_tensor(1e-12, dtype=torch.float64, device=grad_p.device)

            if self.physics_weight_mode == "fixed":
                w_phys = torch.as_tensor(self.fixed_physics_weight, dtype=torch.float64, device=grad_p.device)
                w_stat = torch.as_tensor(1.0 - self.fixed_physics_weight, dtype=torch.float64, device=grad_p.device)
            elif self.physics_weight_mode == "adaptive":
                adaptive_weight = self._adaptive_physics_weight(grad_p, grad_s)
                w_phys = torch.as_tensor(adaptive_weight, dtype=torch.float64, device=grad_p.device)
                w_stat = torch.as_tensor(1.0 - adaptive_weight, dtype=torch.float64, device=grad_p.device)
            else:
                w_phys = 1.0 / torch.maximum(var_p, eps)
                w_stat = 1.0 / torch.maximum(var_s, eps)

            denom = torch.maximum(w_phys + w_stat, eps)
            fused = (w_phys * grad_p + w_stat * grad_s) / denom
            sigma_fused = torch.sqrt(1.0 / denom).to(dtype=fused.dtype)

            fused_gradients[key] = fused
            confidence_intervals[key] = torch.ones_like(fused) * sigma_fused

        return fused_gradients, confidence_intervals


__all__ = ["DualTrackGradientFusion"]

