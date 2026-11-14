"""
Conservation checking utilities for coupling modules.

Provides reusable mass/energy conservation diagnostics to avoid duplicated
implementations across spatial regridders and internal coupling functors.

Mathematical Expressions:
    Mass/Extensive conservation:
        source_total = sum_i (source_i * A_i)  # if areas provided else sum_i source_i
        target_total = sum_j (target_j * A_j)
        error_abs = |source_total - target_total|
        error_rel = error_abs / max(|source_total|, eps)

    Energy conservation:
        imbalance = sum(energy_in) - sum(energy_out)
        error_rel = |imbalance| / max(|energy_in|, eps)

    Acceptance (ledger_acceptance, the one rule of every ledger):
        accepted iff error_abs <= max(tol * max(|source_total|, eps),
                                      gamma_k(u_dtype) * sum_i |a_i|)
        with the rounding term only when the dtype of the returned values is given.

References:
    - Jones, P.W. (1999). Conservative remapping schemes for climate models.
    - CF-1.8 Conventions: Conservation for regridding and aggregation.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class ConservationDiagnostics:
    """Canonical container for conservation check diagnostics.

    This is the Single Source of Truth (SSOT) for conservation diagnostics
    used across the coupling layer -- both by the audit system and by
    the conservation enforcer.

    Parameters
    ----------
    property_name : str
        Name of the conserved property (e.g. 'water', 'carbon').
    is_conserved : bool
        Whether conservation was satisfied within tolerance.
    absolute_error : float
        Absolute difference between source and target totals.
    relative_error : float
        Relative difference (dimensionless).
    source_total : float
        Total of the source field.
    target_total : float
        Total of the target field.
    tolerance : float
        Tolerance threshold used for the check.
    enforcement_applied : bool
        Whether a correction was applied to enforce conservation.
    correction_factor : Optional[float]
        Multiplicative or additive correction factor, if applied.
    timestamp : Optional[float]
        Epoch timestamp of the diagnostic record.
    metadata : Dict[str, Any]
        Arbitrary additional metadata.

    References
    ----------
    - Jones, P.W. (1999). Conservative remapping schemes for climate models.
    """

    property_name: str
    is_conserved: bool
    absolute_error: float
    relative_error: float
    source_total: float
    target_total: float
    tolerance: float = 0.0
    enforcement_applied: bool = False
    correction_factor: Optional[float] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConservationFixer:
    """Enforce conservation with multiplicative or additive correction.

    Mathematical Expression:
        Multiplicative:
            target_corrected = target * (source_total / target_total)
        Additive:
            target_corrected = target + (source_total - target_total) / n_target

    References:
        - Jones, P.W. (1999). Conservative remapping schemes.
        - Lauritzen et al. (2010). Atmospheric transport schemes.

    Parameters:
        method: "multiplicative" or "additive"
        tolerance: absolute error tolerance for correction trigger [same unit as totals]
    """

    def __init__(
        self,
        method: Optional[str] = None,
        tolerance: Optional[float] = None,
    ) -> None:
        from ...parameters import get_global_manager

        self._param_manager = get_global_manager()
        configured_method = method or self._get_parameter(
            "coupling.operators.conservation_fixer.method",
            default="multiplicative",
        )
        configured_tolerance = tolerance
        if configured_tolerance is None:
            configured_tolerance = self._get_parameter(
                "coupling.operators.conservation_fixer.tolerance",
                default=1e-12,
            )
        self.method = str(configured_method).strip().lower()
        if self.method not in {"multiplicative", "additive"}:
            raise ValueError(
                f"Unsupported conservation correction method '{self.method}'. "
                "Expected 'multiplicative' or 'additive'."
            )
        self.tolerance = float(configured_tolerance)

    def _get_parameter(self, key: str, default: Any) -> Any:
        """Get parameter value with manager signature compatibility."""
        try:
            value = self._param_manager.get(key, required=False)
        except TypeError:
            value = self._param_manager.get(key)
        if value is None:
            return default
        return value

    @staticmethod
    def _compute_total(values: Tensor, areas: Optional[Tensor]) -> Tensor:
        """Compute extensive total with optional area weighting."""
        if areas is None:
            return torch.sum(values)
        return torch.sum(values * areas)

    def apply(
        self,
        source: Tensor,
        target: Tensor,
        *,
        source_areas: Optional[Tensor] = None,
        target_areas: Optional[Tensor] = None,
        variable_name: str = "unknown",
    ) -> Tuple[Tensor, ConservationDiagnostics]:
        """Apply configured correction and return corrected target plus diagnostics.

        Mathematical Expression:
            source_total = sum(source * A_source)
            target_total = sum(target * A_target)
            If |source_total - target_total| > tolerance:
                multiplicative -> target *= source_total / target_total
                additive       -> target += (source_total - target_total) / n_target

        Parameters:
            source: source field values
            target: target field values to correct
            source_areas: source cell areas
            target_areas: target cell areas
            variable_name: name used in diagnostics
        """
        pre_check = check_mass_conservation(
            source,
            target,
            source_areas=source_areas,
            target_areas=target_areas,
            tolerance=self.tolerance,
            variable_name=variable_name,
        )

        source_total = self._compute_total(source, source_areas)
        target_total = self._compute_total(target, target_areas)
        absolute_error = torch.abs(source_total - target_total)

        corrected = target.clone()
        correction_factor: Optional[float] = None
        enforcement_applied = False

        if float(absolute_error.detach().cpu().item()) > self.tolerance:
            enforcement_applied = True
            if self.method == "multiplicative":
                eps = torch.tensor(1e-12, device=target.device, dtype=target.dtype)
                denom = target_total if torch.abs(target_total) > eps else eps
                scale = source_total / denom
                corrected = target * scale
                correction_factor = float(scale.detach().cpu().item())
            else:
                n_target = max(int(target.numel()), 1)
                delta = (source_total - target_total) / float(n_target)
                corrected = target + delta
                correction_factor = float(delta.detach().cpu().item())

        post_check = check_mass_conservation(
            source,
            corrected,
            source_areas=source_areas,
            target_areas=target_areas,
            tolerance=self.tolerance,
            variable_name=variable_name,
        )

        diagnostics = ConservationDiagnostics(
            property_name=variable_name,
            is_conserved=bool(post_check["conserved"]),
            absolute_error=float(post_check["absolute_error"]),
            relative_error=float(post_check["relative_error"]),
            source_total=float(post_check["source_total"]),
            target_total=float(post_check["target_total"]),
            tolerance=self.tolerance,
            enforcement_applied=enforcement_applied,
            correction_factor=correction_factor,
            metadata={
                "method": self.method,
                "pre_absolute_error": float(pre_check["absolute_error"]),
                "pre_relative_error": float(pre_check["relative_error"]),
            },
        )
        return corrected, diagnostics


def reduction_rounding_bound(dtype: torch.dtype, term_count: int) -> float:
    """Relative forward-error bound gamma_k of a floating-point reduction of k rounded steps."""
    # References: Higham, N.J. (2002) Accuracy and Stability of Numerical Algorithms, 2nd ed.,
    #   SIAM, Lemma 3.1 (gamma_k) and section 4.2 (recursive summation, |fl(S) - S| <=
    #   gamma_(n-1) sum_i |a_i|); bibliographic basis: the codeliteraturedatabase knowledge base
    #   holds no numerical-analysis source for this bound.
    # Mathematical expression: u = eps(dtype) / 2; gamma_k = k u / (1 - k u).
    # Parameter meanings: eps machine epsilon of the coarsest floating dtype in which the
    #   reduction was evaluated or its operands and result are held (coarsest_floating_dtype);
    #   u unit roundoff; k number of rounded steps (products, additions and the final division
    #   of the reduction).
    # Units: dimensionless; it multiplies an activity sum_i |a_i| in ledger units.
    # Domain/constraints: floating dtypes only (others raise TypeError); k u < 1, otherwise
    #   the reduction has no meaningful error bound and the call raises ValueError.
    # Sign convention: a non-negative bound.
    if not dtype.is_floating_point:
        raise TypeError(f"A rounding bound needs a floating dtype; got {dtype}.")
    unit_roundoff = float(torch.finfo(dtype).eps) / 2.0
    steps = max(int(term_count), 1)
    if steps * unit_roundoff >= 1.0:
        raise ValueError(
            f"A reduction of {steps} rounded steps in {dtype} has no floating-point error bound."
        )
    return steps * unit_roundoff / (1.0 - steps * unit_roundoff)


# Floor of the relative scale s = max(|L_src|, floor) of every ledger: a ledger whose source
# amount is zero is judged by its absolute residual and the rounding term of ledger_acceptance.
LEDGER_SCALE_FLOOR = 1.0e-12


def ledger_relative_scale(source_total: float) -> float:
    """Relative scale s = max(|L_src|, LEDGER_SCALE_FLOOR) of a ledger with source amount L_src."""
    return max(abs(float(source_total)), LEDGER_SCALE_FLOOR)


def require_relative_tolerance(value: Any, name: str) -> float:
    """A relative ledger tolerance as a float: a finite real number in (0, 1), else ValueError."""
    # References: ledger_acceptance (this module): tol * s bounds the accepted residual of a
    #   ledger with relative scale s.
    # Mathematical expression: 0 < tol < 1 (a relative tolerance of 1 or more accepts any result,
    #   and a NaN tolerance compares false with every residual).
    # Parameter meanings: tol the relative tolerance; name the parameter key or argument that
    #   supplied it (reported in the error).
    # Units: dimensionless.
    # Domain/constraints: a bool, a value that is not a real number, NaN, an infinity, zero, a
    #   negative value or a value of 1 or more raises ValueError; there is no replacement value.
    # Sign convention: tol > 0.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite relative tolerance in (0, 1), got {value!r}.")
    tolerance = float(value)
    if not (math.isfinite(tolerance) and 0.0 < tolerance < 1.0):
        raise ValueError(f"{name} must be a finite relative tolerance in (0, 1), got {value!r}.")
    return tolerance


def coarsest_floating_dtype(*dtypes: torch.dtype) -> torch.dtype:
    """The floating dtype with the largest machine epsilon among the given dtypes.

    Integer dtypes are exact and are skipped; at least one floating dtype is required. A
    reduction whose operands or result are held in several dtypes is bounded by the rounding
    of the coarsest one (a float32 field aggregated into a float64 result keeps float32
    rounding in its operands).
    """
    floating = [dtype for dtype in dtypes if dtype.is_floating_point]
    if not floating:
        raise TypeError(f"No floating dtype among {dtypes}; a rounding bound needs one.")
    return max(floating, key=lambda dtype: float(torch.finfo(dtype).eps))


def ledger_acceptance(
    *,
    absolute_error: float,
    relative_scale: float,
    activity: float,
    tolerance: float,
    rounding_dtype: Optional[torch.dtype],
    term_count: int,
) -> Dict[str, Any]:
    """The one acceptance rule of a conservation ledger residual (shared by every ledger)."""
    # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210 (conservative ledgers);
    #   Higham, N.J. (2002) Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
    #   section 4.2 (rounding bound of a reduction; bibliographic basis, see
    #   reduction_rounding_bound).
    # Mathematical expression: accepted iff e <= max(tol * s, gamma_k(u_dtype) * a), with
    #   s = max(|L_src|, 1e-12) the relative scale of the ledger and gamma_k from
    #   reduction_rounding_bound; without a result dtype the rounding term is 0 and the rule
    #   is e / s <= tol.
    # Parameter meanings: e = |L_tgt - L_src| ledger residual; L_src source total; s relative
    #   scale; a = sum_i |a_i| activity of the source terms; tol parameter-backed relative
    #   tolerance (read from the YAML parameter database by the caller); k number of rounded
    #   steps of the reduction that produced the returned value (2n + 2 for n source terms:
    #   products, additions and the final division, each side); u_dtype unit roundoff of the
    #   coarsest floating dtype of the reduction's operands and result (rounding_dtype).
    # Units: e, s, a and the accepted residual in the ledger units; tol and gamma dimensionless.
    # Domain/constraints: tol > 0; e, s and a finite for an accepted ledger (a NaN residual is
    #   never accepted); the rounding term only admits the rounding of that dtype, so a wrong
    #   result (a residual of order |L_src|) is still rejected.
    # Sign convention: e, s, a and the accepted residual are non-negative.
    rounding = 0.0
    if rounding_dtype is not None:
        rounding = reduction_rounding_bound(rounding_dtype, term_count) * float(activity)
    accepted_residual = max(float(tolerance) * float(relative_scale), rounding)
    return {
        "accepted": bool(float(absolute_error) <= accepted_residual),
        "accepted_residual": accepted_residual,
        "rounding_allowance": rounding,
    }


def check_mass_conservation(
    source: Tensor,
    target: Tensor,
    *,
    source_areas: Optional[Tensor] = None,
    target_areas: Optional[Tensor] = None,
    tolerance: float = 1e-9,
    variable_name: str = "unknown",
    rounding_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float | str]:
    """
    Check mass (or any extensive quantity) conservation between source and target.

    Parameters
    ----------
    source : Tensor
        Source values [n_source]
    target : Tensor
        Target values [n_target]
    source_areas : Optional[Tensor], optional
        Source cell areas [n_source], by default None
    target_areas : Optional[Tensor], optional
        Target cell areas [n_target], by default None
    tolerance : float, optional
        Relative tolerance (dimensionless), by default 1e-9
    variable_name : str, optional
        Variable name for diagnostics, by default "unknown"
    rounding_dtype : Optional[torch.dtype], optional
        Coarsest floating dtype in which the reduction that produced the target values was
        evaluated or its operands and result are held (coarsest_floating_dtype). When given,
        the ledger is evaluated in float64 and the residual is accepted up to the rounding
        bound of that dtype (ledger_acceptance); by default None (no rounding term).

    Returns
    -------
    Dict[str, float | str]
        Dictionary with keys: conserved(bool->float), absolute_error, relative_error,
        source_total, target_total, activity, rounding_allowance, accepted_residual.
    """
    # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210 (conservative ledgers);
    #   ledger_acceptance (this module) for the acceptance rule.
    # Mathematical expression: L_src = sum_i s_i A_i (sum_i s_i without areas),
    #   L_tgt = sum_j t_j A_j; e = |L_src - L_tgt|; s = max(|L_src|, 1e-12);
    #   epsilon = e / s; a = sum_i |s_i A_i|; conserved = ledger_acceptance(e, s, a, tol,
    #   rounding_dtype, k = 2 n + 2) with n source terms.
    # Parameter meanings: s_i, t_j source and target values; A_i, A_j optional areas or
    #   support measures; tol relative tolerance supplied by the caller from the YAML
    #   parameter database; rounding_dtype coarsest floating dtype of the reduction.
    # Units: L, e, s and a in the ledger units (value units times area units); epsilon and
    #   tol dimensionless.
    # Domain/constraints: with rounding_dtype the ledger is evaluated in float64, so only the
    #   rounding of the returned values enters the accepted residual; without it the rule is
    #   epsilon <= tol.
    # Sign convention: signed totals; e, s, a and epsilon are non-negative.
    if rounding_dtype is not None:
        source = source.to(torch.float64)
        target = target.to(torch.float64)
        if source_areas is not None:
            source_areas = torch.as_tensor(source_areas).to(device=source.device, dtype=torch.float64)
        if target_areas is not None:
            target_areas = torch.as_tensor(target_areas).to(device=target.device, dtype=torch.float64)
    eps = torch.tensor(LEDGER_SCALE_FLOOR, device=source.device, dtype=source.dtype)

    def _total(values: Tensor, areas: Optional[Tensor]) -> Tensor:
        if areas is None:
            return torch.sum(values)
        return torch.sum(values * areas)

    source_total = _total(source, source_areas)
    target_total = _total(target, target_areas)

    abs_error = torch.abs(source_total - target_total)
    denom = torch.maximum(torch.abs(source_total), eps)
    rel_error = abs_error / denom
    source_terms = source if source_areas is None else source * source_areas
    activity = float(torch.sum(torch.abs(source_terms)).detach().cpu().item())
    acceptance = ledger_acceptance(
        absolute_error=float(abs_error.detach().cpu().item()),
        relative_scale=float(denom.detach().cpu().item()),
        activity=activity,
        tolerance=tolerance,
        rounding_dtype=rounding_dtype,
        term_count=2 * source.numel() + 2,
    )
    return {
        "conserved": float(acceptance["accepted"]),
        "absolute_error": float(abs_error.detach().cpu().item()),
        "relative_error": float(rel_error.detach().cpu().item()),
        "source_total": float(source_total.detach().cpu().item()),
        "target_total": float(target_total.detach().cpu().item()),
        "variable_name": variable_name,
        "activity": activity,
        "rounding_allowance": acceptance["rounding_allowance"],
        "accepted_residual": acceptance["accepted_residual"],
    }


def check_energy_conservation(
    energy_in: Tensor,
    energy_out: Tensor,
    *,
    tolerance: float = 1e-9,
    rounding_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """
    Check energy conservation between input and output energy terms.

    Parameters
    ----------
    energy_in : Tensor
        Incoming energy components.
    energy_out : Tensor
        Outgoing energy components.
    tolerance : float, optional
        Relative tolerance (dimensionless), by default 1e-9
    rounding_dtype : Optional[torch.dtype], optional
        Coarsest floating dtype of the computation of the output terms; when given the ledger
        is evaluated in float64 and accepted up to the rounding bound of that dtype
        (ledger_acceptance).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: conserved(bool->float), imbalance, relative_error, activity,
        rounding_allowance, accepted_residual.
    """
    # References: ledger_acceptance (this module) for the acceptance rule.
    # Mathematical expression: I = sum_i E_in,i - sum_j E_out,j; s = max(|sum_i E_in,i|, 1e-12);
    #   epsilon = |I| / s; a = sum_i |E_in,i|; conserved = ledger_acceptance(|I|, s, a, tol,
    #   rounding_dtype, k = 2 n + 2) with n input terms.
    # Parameter meanings: E_in, E_out energy terms; tol relative tolerance supplied by the
    #   caller; rounding_dtype coarsest floating dtype of the computation of the outputs.
    # Units: I, s and a in the energy units of the terms; epsilon and tol dimensionless.
    # Domain/constraints: with rounding_dtype the ledger is evaluated in float64; without it the
    #   rule is epsilon <= tol.
    # Sign convention: I is positive when the inputs exceed the outputs; s and a non-negative.
    if rounding_dtype is not None:
        energy_in = energy_in.to(torch.float64)
        energy_out = energy_out.to(torch.float64)
    eps = torch.tensor(LEDGER_SCALE_FLOOR, device=energy_in.device, dtype=energy_in.dtype)
    imbalance = torch.sum(energy_in) - torch.sum(energy_out)
    denom = torch.maximum(torch.abs(torch.sum(energy_in)), eps)
    rel_error = torch.abs(imbalance) / denom
    activity = float(torch.sum(torch.abs(energy_in)).detach().cpu().item())
    acceptance = ledger_acceptance(
        absolute_error=float(torch.abs(imbalance).detach().cpu().item()),
        relative_scale=float(denom.detach().cpu().item()),
        activity=activity,
        tolerance=tolerance,
        rounding_dtype=rounding_dtype,
        term_count=2 * energy_in.numel() + 2,
    )
    return {
        "conserved": float(acceptance["accepted"]),
        "imbalance": float(imbalance.detach().cpu().item()),
        "relative_error": float(rel_error.detach().cpu().item()),
        "activity": activity,
        "rounding_allowance": acceptance["rounding_allowance"],
        "accepted_residual": acceptance["accepted_residual"],
    }

