"""Resource-activity invasion threshold on a finite matrix family (G12-THRESHOLD).

Bounded contract: for a linearized resource-activity subsystem with a
nonnegative growth operator ``G`` and a nonsingular M-matrix loss operator
``Delta`` the invasion threshold is ``R_act = rho(Delta^-1 G)`` and the sign
of ``R_act - 1`` equals the sign of the spectral bound ``s(G - Delta)``
(BIHSS appendix, Metzler threshold lemma). Inputs outside the premises are
refused; the non-hyperbolic case ``R_act = 1`` is reported as critical and
no stability conclusion is drawn from it. No positive equilibrium,
transcritical bifurcation, or global persistence is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class InvasionThresholdCertificate:
    """Outcome of the threshold contract on one matrix pair."""

    activity_threshold: float
    spectral_bound: float
    verdict: str
    hyperbolic: bool
    resource_background: Optional[Tuple[float, ...]]
    tolerance: float


def _parameter_default(key: str) -> float:
    """Read a default tolerance through the canonical infrastructure parameter entry."""
    from relic.parameters import get_global_manager

    return get_global_manager().get(key, required=True)


def is_nonnegative(matrix: Tensor, *, tolerance: float = 0.0) -> bool:
    """Entries are >= -tolerance; the default 0.0 means an exact check."""
    return bool(torch.all(matrix >= -tolerance))


def is_z_matrix(matrix: Tensor, *, tolerance: float = 0.0) -> bool:
    """Off-diagonal entries are <= tolerance; the default 0.0 means an exact check."""
    off_diagonal = matrix - torch.diag(torch.diagonal(matrix))
    return bool(torch.all(off_diagonal <= tolerance))


def is_nonsingular_m_matrix(matrix: Tensor, *, tolerance: Optional[float] = None) -> bool:
    """Z-matrix whose inverse exists and is entrywise nonnegative.

    References: Berman and Plemmons (1994) Nonnegative Matrices in the
    Mathematical Sciences, ch. 6 (condition N38 of Theorem 6.2.3).
    Mathematical expression: Delta is a nonsingular M-matrix iff Delta is a
    Z-matrix and Delta^-1 >= 0 entrywise.
    Parameter meanings: Delta square loss operator; tolerance absolute
    slack for the sign tests (None reads ParameterManager key
    validation.iphs.m_matrix_sign_tolerance).
    Units: entries of Delta are rates [s-1] (or the model's rate unit).
    Domain/constraints: square finite matrices only; singular Delta is
    refused.
    Sign convention: losses appear with positive diagonal, exchanges with
    nonpositive off-diagonal entries.
    """
    if tolerance is None:
        tolerance = _parameter_default("validation.iphs.m_matrix_sign_tolerance")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Delta must be a square matrix")
    if not torch.isfinite(matrix).all():
        raise ValueError("Delta must be finite")
    if not is_z_matrix(matrix, tolerance=tolerance):
        return False
    try:
        inverse = torch.linalg.inv(matrix)
    except RuntimeError:
        return False
    if not torch.isfinite(inverse).all():
        return False
    return is_nonnegative(inverse, tolerance=tolerance)


def spectral_radius(matrix: Tensor) -> float:
    eigenvalues = torch.linalg.eigvals(matrix)
    return float(torch.max(torch.abs(eigenvalues)).real)


def spectral_bound(matrix: Tensor) -> float:
    eigenvalues = torch.linalg.eigvals(matrix)
    return float(torch.max(eigenvalues.real))


def resource_background(loss: Tensor, supply: Tensor) -> Tensor:
    """R0 = Lambda_R^-1 I_R for a positive-diagonal resource loss operator."""
    if loss.ndim != 2 or loss.shape[0] != loss.shape[1]:
        raise ValueError("Lambda_R must be square")
    if not is_nonsingular_m_matrix(loss):
        raise ValueError("Lambda_R must be a nonsingular M-matrix")
    return torch.linalg.solve(loss, supply.reshape(-1))


def activity_threshold(
    growth: Tensor | Callable[[Tensor], Tensor],
    loss: Tensor,
    *,
    resource_loss: Optional[Tensor] = None,
    resource_supply: Optional[Tensor] = None,
    tolerance: Optional[float] = None,
) -> InvasionThresholdCertificate:
    """Evaluate R_act = rho(Delta^-1 G(R0)) with the Metzler sign equivalence.

    References: BIHSS manuscript, conclusion C5 and appendix lemma
    "Metzler threshold"; Berman and Plemmons (1994); Diekmann, Heesterbeek
    and Metz (1990) J. Math. Biol. 28 (next-generation threshold);
    van den Driessche and Watmough (2002) Math. Biosci. 180.
    Mathematical expression: R_act = rho(Delta^-1 G(R0)), R0 =
    Lambda_R^-1 I_R; sign(R_act - 1) = sign(s(G - Delta)).
    Parameter meanings: G nonnegative growth operator (a matrix or a
    callable of the resource background), Delta nonsingular M-matrix loss
    operator, Lambda_R resource loss operator, I_R resource supply.
    Units: G and Delta in the activity rate unit [s-1]; R_act dimensionless.
    Domain/constraints: G >= 0 and Delta a nonsingular M-matrix are
    premises and are checked; |R_act - 1| <= tolerance is the critical
    (non-hyperbolic) case with no stability verdict. A tolerance of None
    reads ParameterManager key validation.iphs.invasion_threshold_critical_band.
    Sign convention: R_act > 1 means the activity invades the activity-free
    background; R_act < 1 means it decays.
    """
    if tolerance is None:
        tolerance = _parameter_default("validation.iphs.invasion_threshold_critical_band")
    background: Optional[Tensor] = None
    if callable(growth):
        if resource_loss is None or resource_supply is None:
            raise ValueError("a growth callable requires resource_loss and resource_supply")
        background = resource_background(resource_loss, resource_supply)
        growth_matrix = growth(background)
    else:
        growth_matrix = growth
    growth_matrix = torch.as_tensor(growth_matrix, dtype=torch.float64)
    loss = torch.as_tensor(loss, dtype=torch.float64)
    if growth_matrix.shape != loss.shape or growth_matrix.ndim != 2:
        raise ValueError("G and Delta must be square matrices of the same shape")
    if not torch.isfinite(growth_matrix).all():
        raise ValueError("G must be finite")
    if not is_nonnegative(growth_matrix):
        raise ValueError("G must be entrywise nonnegative (premise of the threshold lemma)")
    if not is_nonsingular_m_matrix(loss):
        raise ValueError("Delta must be a nonsingular M-matrix (premise of the threshold lemma)")
    next_generation = torch.linalg.solve(loss, growth_matrix)
    threshold = spectral_radius(next_generation)
    bound = spectral_bound(growth_matrix - loss)
    hyperbolic = abs(threshold - 1.0) > tolerance
    if not hyperbolic:
        verdict = "critical"
    elif threshold > 1.0:
        verdict = "invasion"
    else:
        verdict = "decay"
    # The lemma's sign equivalence is checked on the actual numbers; a
    # disagreement means a premise was violated numerically and is refused.
    if hyperbolic and (bound > tolerance) != (threshold > 1.0):
        raise ValueError(
            f"spectral bound s(G - Delta) = {bound:.6g} and R_act = {threshold:.6g} disagree in sign"
        )
    return InvasionThresholdCertificate(
        activity_threshold=threshold,
        spectral_bound=bound,
        verdict=verdict,
        hyperbolic=hyperbolic,
        resource_background=None if background is None else tuple(float(v) for v in background),
        tolerance=tolerance,
    )


__all__ = [
    "InvasionThresholdCertificate",
    "activity_threshold",
    "is_nonnegative",
    "is_nonsingular_m_matrix",
    "is_z_matrix",
    "resource_background",
    "spectral_bound",
    "spectral_radius",
]
