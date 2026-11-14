"""Conservation guardian for SCM domains.

References:
- Manzoni and Porporato (2009) Soil Biology and Biochemistry.
- Holton (2012) An Introduction to Dynamic Meteorology.

Mathematical expressions:
- dS/dt = sum(F_in) - sum(F_out)
- Delta S = (sum_in - sum_out) * dt

Parameter meanings:
- domain: conservation domain name (e.g., carbon.total)
- flux_in/flux_out: dict of named flux tensors or scalars
- storage_before/storage_after: dict of pool tensors or scalars
- dt: time step in seconds
- tolerance: dict with atol/rtol for residual check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class BudgetCheckResult:
    domain: str
    members: List[str]
    sum_flux_in: float
    sum_flux_out: float
    delta_storage: float
    residual: float
    passed: bool
    tolerance_used: float
    invariant_drift: Optional[float] = None


class ConservationGuardian:
    """Domain-level conservation checker."""

    def __init__(
        self,
        contract_registry: Any,
        default_atol: float = 1e-10,
        default_rtol: float = 1e-8,
    ) -> None:
        self._registry = contract_registry
        self._default_atol = float(default_atol)
        self._default_rtol = float(default_rtol)

    @staticmethod
    def _sum_value(value: Tensor | float | int) -> float:
        if isinstance(value, Tensor):
            return float(value.sum().item())
        return float(value)

    def _sum_flux(self, fluxes: Dict[str, Tensor | float | int], members: List[str]) -> float:
        if not fluxes:
            return 0.0
        if not members:
            return sum(self._sum_value(v) for v in fluxes.values())
        total = 0.0
        for key, value in fluxes.items():
            if any(member in str(key) for member in members):
                total += self._sum_value(value)
        return total

    def compute_invariant_drift(
        self,
        incidence_row: NDArray[np.float64],  # one row of left-null-space of B
        flux_vector: Dict[str, Tensor | float | int],
        dt: float,
    ) -> float:
        """
        Noether invariant drift = sum(c_i * f_i) * dt
        where c_i is the i-th coefficient from the left null space of B.

        References:
        - Noether (1918), invariant conservation from symmetry
        - Manzoni and Porporato (2009), mass balance in ecosystem models
        Mathematical expression:
        - drift = (sum_j c_j * F_j) * dt
        Parameters:
        - incidence_row: conservation law coefficients from left null space
        - flux_vector: named flux tensors
        - dt: time step (seconds)
        """
        coefficients = np.asarray(incidence_row, dtype=np.float64).reshape(-1)
        if coefficients.size == 0:
            return 0.0
        if not flux_vector:
            return 0.0

        ordered_flux_names = list(flux_vector.keys())
        if len(ordered_flux_names) != coefficients.size:
            raise ValueError(
                "incidence_row length must match flux_vector length: "
                f"{coefficients.size} != {len(ordered_flux_names)}"
            )

        flux_values = np.array(
            [self._sum_value(flux_vector[name]) for name in ordered_flux_names],
            dtype=np.float64,
        )
        drift = float(np.dot(coefficients, flux_values) * float(dt))
        return drift

    def check_domain(
        self,
        domain: str,
        flux_in: Dict[str, Tensor | float | int],
        flux_out: Dict[str, Tensor | float | int],
        storage_before: Dict[str, Tensor | float | int],
        storage_after: Dict[str, Tensor | float | int],
        dt: float,
        tolerance: Optional[Dict[str, float]] = None,
    ) -> BudgetCheckResult:
        """Check conservation for a domain."""
        members = list(self._registry.get_domain_members(domain)) if self._registry is not None else []

        total_flux_in = self._sum_flux(flux_in, members)
        total_flux_out = self._sum_flux(flux_out, members)

        storage_change = 0.0
        for member in members:
            before = storage_before.get(member, torch.tensor(0.0))
            after = storage_after.get(member, torch.tensor(0.0))
            storage_change += self._sum_value(after) - self._sum_value(before)

        expected_change = (total_flux_in - total_flux_out) * float(dt)
        residual = abs(storage_change - expected_change)

        tol = tolerance or {}
        atol = float(tol.get("atol", self._default_atol))
        rtol = float(tol.get("rtol", self._default_rtol))
        threshold = atol + rtol * abs(expected_change)
        passed = residual <= threshold

        if not passed:
            logger.error(
                "Conservation breach in domain %s: residual=%s threshold=%s",
                domain,
                f"{residual:.3e}",
                f"{threshold:.3e}",
            )

        return BudgetCheckResult(
            domain=domain,
            members=members,
            sum_flux_in=total_flux_in,
            sum_flux_out=total_flux_out,
            delta_storage=storage_change,
            residual=residual,
            passed=passed,
            tolerance_used=threshold,
        )
