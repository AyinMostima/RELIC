"""Relative storage (exergy-type) Lyapunov contract on an affine ledger leaf (G12-EXERGY).

Bounded contract for a closed dissipative system with storage ``F`` and a
linear conserved ledger ``C x = const``: the relative storage
``V(x) = F(x) - F(x*) - grad F(x*)^T (x - x*)`` is a Lyapunov candidate on
the leaf only when ``x*`` is a constrained stationary point
(``grad F(x*) = C^T lambda``) and the Hessian is positive definite on
``ker C``; then ``dot V = dot F - grad F(x*)^T dot x = -D(x) <= 0`` along
ledger-preserving trajectories. The contract computes the full ``dot V``
(never dropping the linear term), checks every premise, and refuses to
conclude when a premise fails (the BIHSS counterexample with
``grad F(x*) != 0`` is the canonical refusal). Curved constraints are
outside the scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class RelativeStorageCertificate:
    stationary: bool
    multiplier: Optional[Tuple[float, ...]]
    gradient_residual: float
    hessian_min_eigenvalue_on_leaf: float
    definite_on_leaf: bool
    verdict: str
    tolerance: float


def relative_storage(
    storage: Callable[[Tensor], Tensor],
    x: Tensor,
    x_star: Tensor,
) -> Tensor:
    """V(x) = F(x) - F(x*) - grad F(x*)^T (x - x*).

    References: BIHSS manuscript, theorem "exergy Lyapunov stability";
    Willems (1972) Arch. Rational Mech. Anal. 45 (storage functions).
    Mathematical expression: V(x) = F(x) - F(x*) - grad F(x*)^T (x - x*).
    Parameter meanings: F storage functional, x state, x* reference point.
    Units: units of F (energy or exergy per unit support).
    Domain/constraints: F must be differentiable at x*; V is evaluated
    exactly, including the linear term.
    Sign convention: V >= 0 near a constrained stationary point with a
    definite Hessian on the leaf; V is not sign-definite otherwise.
    """
    reference = x_star.detach().clone().requires_grad_(True)
    f_star = storage(reference)
    (gradient,) = torch.autograd.grad(f_star, reference)
    return storage(x) - f_star.detach() - torch.dot(gradient.detach().reshape(-1), (x - x_star).reshape(-1))


def relative_storage_rate(
    storage: Callable[[Tensor], Tensor],
    x: Tensor,
    x_dot: Tensor,
    x_star: Tensor,
) -> Tensor:
    """Full chain rule dot V = grad F(x)^T x_dot - grad F(x*)^T x_dot.

    The linear term is kept; it vanishes only when grad F(x*) = C^T lambda
    and C x_dot = 0, which the certificate verifies separately.
    """
    state = x.detach().clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(storage(state), state)
    reference = x_star.detach().clone().requires_grad_(True)
    (gradient_star,) = torch.autograd.grad(storage(reference), reference)
    return torch.dot((gradient - gradient_star).reshape(-1), x_dot.reshape(-1))


def _parameter_default(key: str) -> float:
    """Read a default tolerance through the canonical infrastructure parameter entry."""
    from relic.parameters import get_global_manager

    return get_global_manager().get(key, required=True)


def _leaf_basis(ledger: Optional[Tensor], dimension: int, dtype: torch.dtype) -> Tensor:
    if ledger is None:
        return torch.eye(dimension, dtype=dtype)
    # References: Golub and Van Loan (2013), Matrix Computations, 4th ed.,
    #   sec. 5.4.1 (numerical rank); the numpy.linalg.matrix_rank default cut.
    # Mathematical expression: C = U S V^T; rank = #{sigma_i > eps max(m, n)
    #   sigma_max}; ker C is spanned by the rows of V^T beyond the rank.
    # Parameter meanings: C ledger matrix (m rows, n state columns), sigma_i its
    #   singular values, eps the machine epsilon of the working dtype.
    # Units: sigma_i carry the ledger coefficient units; the cut is relative, so
    #   the rank is invariant under a rescaling of C.
    # Domain/constraints: a numerical roundoff guard, not a model parameter; a
    #   zero ledger has rank 0 and leaves the whole state space as the leaf.
    # Sign convention: not applicable.
    ledger = ledger.to(dtype)
    _, singular, vh = torch.linalg.svd(ledger, full_matrices=True)
    cutoff = float(torch.finfo(dtype).eps) * float(max(ledger.shape)) * float(singular.max())
    rank = int((singular > cutoff).sum())
    return vh[rank:].transpose(0, 1)


def certify_relative_storage(
    storage: Callable[[Tensor], Tensor],
    x_star: Tensor,
    *,
    ledger: Optional[Tensor] = None,
    tolerance: Optional[float] = None,
) -> RelativeStorageCertificate:
    """Check the premises of the relative-storage Lyapunov theorem at x*.

    References: BIHSS manuscript theorem "exergy Lyapunov stability" and
    its counterexample; Khalil (2002) Nonlinear Systems, Theorem 4.1.
    Mathematical expression: grad F(x*) = C^T lambda (least-squares
    residual within tolerance); z^T grad^2 F(x*) z > 0 for z in ker C.
    Parameter meanings: F storage, x* candidate equilibrium, C ledger
    matrix (rows are conserved linear ledgers), lambda multiplier.
    Units: as F and x.
    Domain/constraints: affine ledger leaves only; without a ledger the
    unconstrained conditions grad F(x*) = 0 and grad^2 F(x*) > 0 apply.
    The stationarity test is residual <= tolerance max(g_floor, ||grad F(x*)||)
    with g_floor = validation.iphs.relative_storage_gradient_scale_floor, and
    the definiteness test is lambda_min > tolerance; a tolerance of None
    reads ParameterManager key validation.iphs.relative_storage_tolerance.
    Sign convention: a positive minimum leaf eigenvalue is required for
    strict definiteness; zero or negative refuses the verdict.
    """
    if tolerance is None:
        tolerance = _parameter_default("validation.iphs.relative_storage_tolerance")
    reference = x_star.detach().clone().to(torch.float64).requires_grad_(True)
    value = storage(reference)
    (gradient,) = torch.autograd.grad(value, reference, create_graph=True)
    dimension = reference.numel()
    hessian_rows = []
    for index in range(dimension):
        (row,) = torch.autograd.grad(gradient.reshape(-1)[index], reference, retain_graph=True)
        hessian_rows.append(row.reshape(-1))
    hessian = torch.stack(hessian_rows)
    gradient = gradient.detach().reshape(-1)
    multiplier: Optional[Tuple[float, ...]] = None
    if ledger is None:
        residual = float(torch.linalg.norm(gradient))
    else:
        ledger = ledger.to(torch.float64)
        solution = torch.linalg.lstsq(ledger.transpose(0, 1), gradient.reshape(-1, 1)).solution.reshape(-1)
        residual = float(torch.linalg.norm(ledger.transpose(0, 1) @ solution - gradient))
        multiplier = tuple(float(v) for v in solution)
    # References: Khalil (2002) Theorem 4.1 (stationarity premise); Hairer,
    #   Norsett and Wanner (1993), Sect. II.4 (mixed absolute/relative scale).
    # Mathematical expression: ||C^T lambda - grad F(x*)|| <= tol * max(g_floor, ||grad F(x*)||).
    # Parameter meanings: tol the stationarity tolerance; g_floor =
    #   validation.iphs.relative_storage_gradient_scale_floor (RELIC convention 1.0).
    # Units: residual, ||grad F|| and g_floor in units of grad F on the declared
    #   normalized basis; tol dimensionless.
    # Domain/constraints: relative above the floor, absolute (tol * g_floor) below it.
    # Sign convention: only magnitudes are compared.
    gradient_floor = float(_parameter_default("validation.iphs.relative_storage_gradient_scale_floor"))
    stationary = residual <= tolerance * max(gradient_floor, float(torch.linalg.norm(gradient)))
    basis = _leaf_basis(ledger, dimension, torch.float64)
    if basis.shape[1] == 0:
        min_eigenvalue = float("inf")
        definite = True
    else:
        reduced = basis.transpose(0, 1) @ hessian.detach() @ basis
        reduced = 0.5 * (reduced + reduced.transpose(0, 1))
        min_eigenvalue = float(torch.linalg.eigvalsh(reduced).min())
        definite = min_eigenvalue > tolerance
    if not stationary:
        verdict = "refused: x* is not a constrained stationary point (grad F(x*) not in range C^T)"
    elif not definite:
        verdict = "refused: Hessian of F is not positive definite on the ledger leaf"
    else:
        verdict = "lyapunov_candidate"
    return RelativeStorageCertificate(
        stationary=stationary,
        multiplier=multiplier,
        gradient_residual=residual,
        hessian_min_eigenvalue_on_leaf=min_eigenvalue,
        definite_on_leaf=definite,
        verdict=verdict,
        tolerance=tolerance,
    )


def feedback_power_ledger(
    storage_rate: Tensor,
    port_power: Tensor,
    dissipation: Tensor,
    boundary_flow: Tensor,
    *,
    tolerance: Optional[float] = None,
) -> Tuple[bool, float]:
    """Open-system balance dot F = y_F^T u - D + Phi_F (BIHSS passivity form).

    References: BIHSS manuscript appendix "passivity forms of open systems";
    Willems (1972).
    Mathematical expression: residual = dot F - (y^T u - D + Phi); the ledger
    closes when |residual| <= tolerance * scale, scale = max(p_floor, |dot F|,
    |y^T u|, |D|, |Phi|), p_floor = validation.iphs.feedback_power_ledger_scale_floor.
    Parameter meanings: dot F storage rate, y^T u port power, D >= 0
    dissipation, Phi boundary exergy flow.
    Units: power (units of F per second).
    Domain/constraints: D must be nonnegative; a negative D is refused. A
    tolerance of None reads ParameterManager key
    validation.iphs.feedback_power_ledger_rtol.
    Sign convention: supplied power positive into the system; dissipation
    positive as a loss.
    """
    if tolerance is None:
        tolerance = _parameter_default("validation.iphs.feedback_power_ledger_rtol")
    if bool(dissipation < 0):
        raise ValueError("dissipation must be nonnegative")
    residual = float(storage_rate - (port_power - dissipation + boundary_flow))
    # References: Willems (1972) (dissipation inequality); Hairer, Norsett and
    #   Wanner (1993), Sect. II.4 (mixed absolute/relative scale).
    # Mathematical expression: scale = max(p_floor, |dot F|, |y^T u|, |D|, |Phi|).
    # Parameter meanings: p_floor = validation.iphs.feedback_power_ledger_scale_floor
    #   (RELIC convention 1.0); the other terms as in the balance above.
    # Units: all terms and p_floor in power units (F per second) of the declared basis.
    # Domain/constraints: relative above the floor, absolute (tol * p_floor) below it.
    # Sign convention: magnitudes only; the balance keeps its own signs.
    power_floor = float(_parameter_default("validation.iphs.feedback_power_ledger_scale_floor"))
    scale = max(power_floor, abs(float(storage_rate)), abs(float(port_power)), abs(float(dissipation)), abs(float(boundary_flow)))
    return abs(residual) <= tolerance * scale, residual


__all__ = [
    "RelativeStorageCertificate",
    "certify_relative_storage",
    "feedback_power_ledger",
    "relative_storage",
    "relative_storage_rate",
]
