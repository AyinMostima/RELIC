"""Bounded hybrid-event runtime: guard crossing, event-time sensitivity and saltation.

Theory owner: docs/source/theory/hybrid_events.md. This module is a numerical helper
of the solver layer for one transversal guard crossing inside one explicit step. It
is not a Functor, does not advance StateManager state, and does not locate events
inside implicit solvers. Grazing, simultaneous events and a general production
saltation driver remain [N] and are refused rather than approximated.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

Number = Union[Tensor, float, int]
_TRANSVERSALITY_TOLERANCE_KEY = "computation.solvers.hybrid_events.transversality_tolerance"

__all__ = [
    "event_time_sensitivity",
    "locate_guard_crossing",
    "saltation_matrix",
    "sensitivity_jump",
]


def _as_tensor(value: Number, *, like: Tensor | None = None, name: str) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value
    else:
        if like is not None:
            tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        else:
            tensor = torch.as_tensor(value, dtype=torch.get_default_dtype())
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor or number.")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite.")
    return tensor


def _require_transversal(denominator: Tensor, tolerance: Optional[float]) -> None:
    # References: Hiskens and Pai (2000) (transversality condition of the
    #   saltation matrix); Shampine and Thompson (2000) (event location).
    # Mathematical expression: transversal  <=>  |d| > tol, with d = g_start - g_end
    #   (location) or d = g_x f_minus + g_t (sensitivity, saltation).
    # Parameter meanings: tol = explicit transversality_tol, or the YAML key
    #   computation.solvers.hybrid_events.transversality_tolerance (RELIC
    #   convention 1e-12) when the caller passes None.
    # Units: d and tol in the caller's guard units (location) or guard units
    #   per second (sensitivity, saltation).
    # Domain/constraints: tol > 0; |d| <= tol is refused as grazing.
    # Sign convention: only the magnitude of d is tested.
    if tolerance is None:
        from relic.parameters import get_global_manager

        tolerance = float(get_global_manager().get(_TRANSVERSALITY_TOLERANCE_KEY, required=True))
    if tolerance <= 0.0:
        raise ValueError("transversality_tol must be positive.")
    if bool((torch.abs(denominator) <= tolerance).any().item()):
        raise ValueError(
            "grazing or tangential event: |g_x f_minus + g_t| <= transversality_tol; "
            "the event-time sensitivity is undefined and is not extrapolated."
        )


def locate_guard_crossing(
    g_start: Number,
    g_end: Number,
    t_start: Number,
    dt: Number,
    *,
    transversality_tol: Optional[float] = None,
) -> Tensor:
    """Locate one guard zero crossing inside an explicit step by linear interpolation.

    References:
      - Goebel, Sanfelice and Teel (2012), Hybrid Dynamical Systems, event detection.
      - RELIC theory page hybrid_events.md, Section 2.3.
    Mathematical expression:
      tau = t_start + dt * g_start / (g_start - g_end), valid when g_start and g_end
      bracket zero (g_start * g_end <= 0 with g_start != g_end); exact for a guard
      affine in time over the step.
    Parameter meanings:
      g_start, g_end are the guard values at the step start and end; t_start is the
      step start time; dt the step length.
    Units:
      g_start, g_end in the guard units; t_start, dt and tau in seconds.
    Domain/constraints:
      finite inputs; dt > 0; the pair must bracket zero (a sign change or an exact zero
      at the end); |g_start - g_end| <= transversality_tol is refused as grazing; no
      extrapolation outside [t_start, t_start + dt].
    Sign convention:
      tau increases with dt; the returned value keeps autograd through g_start and
      g_end so d tau / d theta follows the implicit event-time formula.
    """
    g0 = _as_tensor(g_start, name="g_start")
    g1 = _as_tensor(g_end, like=g0, name="g_end")
    t0 = _as_tensor(t_start, like=g0, name="t_start")
    step = _as_tensor(dt, like=g0, name="dt")
    if bool((step <= 0).any().item()):
        raise ValueError("dt must be positive.")
    if bool((g0 * g1 > 0).any().item()):
        raise ValueError("guard values do not bracket zero; no crossing inside the step.")
    difference = g0 - g1
    _require_transversal(difference, transversality_tol)
    fraction = g0 / difference
    return t0 + step * fraction


def event_time_sensitivity(
    g_x: Number,
    g_theta: Number,
    g_t: Number,
    f_minus: Number,
    s_minus: Number,
    *,
    transversality_tol: Optional[float] = None,
) -> Tensor:
    """Event-time sensitivity of a transversal guard crossing.

    References:
      - Goebel, Sanfelice and Teel (2012); Kong et al. (2023), arXiv:2306.06862, Eq. 9.
    Mathematical expression:
      d tau / d theta = -(g_x S_minus + g_theta) / (g_x f_minus + g_t).
    Parameter meanings:
      g_x guard gradient with respect to the state, g_theta with respect to the
      parameter, g_t with respect to time; f_minus the pre-event rate of the guarded
      state; S_minus the pre-event parameter sensitivity of the state.
    Units:
      g_x [guard unit per state unit], g_theta [guard unit per theta unit], g_t [guard
      unit s-1], f_minus [state unit s-1], S_minus [state unit per theta unit]; result
      [s per theta unit].
    Domain/constraints:
      scalar or elementwise tensors; |g_x f_minus + g_t| must exceed
      transversality_tol (grazing is refused).
    Sign convention:
      a positive result delays the event when theta increases.
    """
    gx = _as_tensor(g_x, name="g_x")
    gt = _as_tensor(g_theta, like=gx, name="g_theta")
    gtime = _as_tensor(g_t, like=gx, name="g_t")
    fm = _as_tensor(f_minus, like=gx, name="f_minus")
    sm = _as_tensor(s_minus, like=gx, name="s_minus")
    denominator = gx * fm + gtime
    _require_transversal(denominator, transversality_tol)
    return -(gx * sm + gt) / denominator


def sensitivity_jump(
    s_minus: Number,
    r_x: Number,
    r_theta: Number,
    r_t: Number,
    f_minus: Number,
    f_plus: Number,
    tau_theta: Number,
) -> Tensor:
    """Post-event parameter sensitivity from the pre-event one (scalar-state form).

    References:
      - Kong et al. (2023), arXiv:2306.06862, Section III-A.
    Mathematical expression:
      S_plus = R_x S_minus + R_theta + (R_x f_minus + R_t - f_plus) * (d tau / d theta).
    Parameter meanings:
      R_x, R_theta, R_t partial derivatives of the reset map; f_minus, f_plus the
      pre- and post-event rates; tau_theta the event-time sensitivity.
    Units:
      S_plus, S_minus, R_theta [state unit per theta unit]; R_x [1]; R_t, f_minus,
      f_plus [state unit s-1]; tau_theta [s per theta unit].
    Domain/constraints:
      finite inputs; elementwise for scalar states or diagonal resets; matrix
      resets must use saltation_matrix.
    Sign convention:
      S_plus - S_minus is the jump of the parameter sensitivity across the event.
    """
    sm = _as_tensor(s_minus, name="s_minus")
    rx = _as_tensor(r_x, like=sm, name="r_x")
    rt = _as_tensor(r_theta, like=sm, name="r_theta")
    rtime = _as_tensor(r_t, like=sm, name="r_t")
    fm = _as_tensor(f_minus, like=sm, name="f_minus")
    fp = _as_tensor(f_plus, like=sm, name="f_plus")
    tt = _as_tensor(tau_theta, like=sm, name="tau_theta")
    return rx * sm + rt + (rx * fm + rtime - fp) * tt


def saltation_matrix(
    r_x: Tensor,
    f_minus: Tensor,
    f_plus: Tensor,
    r_t: Tensor,
    normal: Tensor,
    g_t: Number,
    *,
    transversality_tol: Optional[float] = None,
) -> Tensor:
    """Saltation matrix for state perturbations across a transversal event.

    References:
      - Kong et al. (2023), arXiv:2306.06862, Eq. 9; di Bernardo et al. (2008).
    Mathematical expression:
      S = R_x + (f_plus - R_x f_minus - R_t) n^T / (n^T f_minus + g_t), n = g_x^T.
    Parameter meanings:
      R_x (n, n) reset Jacobian; f_minus, f_plus (n,) pre/post-event vector fields;
      R_t (n,) time derivative of the reset; normal (n,) the guard gradient.
    Units:
      R_x and S [1]; f_minus, f_plus, R_t [state unit s-1]; normal [guard unit per
      state unit]; g_t [guard unit s-1].
    Domain/constraints:
      square R_x matching the vector lengths; |n^T f_minus + g_t| must exceed
      transversality_tol.
    Sign convention:
      x_plus perturbation = S times x_minus perturbation.
    """
    rx = _as_tensor(r_x, name="r_x")
    fm = _as_tensor(f_minus, like=rx, name="f_minus")
    fp = _as_tensor(f_plus, like=rx, name="f_plus")
    rt = _as_tensor(r_t, like=rx, name="r_t")
    nvec = _as_tensor(normal, like=rx, name="normal")
    gt = _as_tensor(g_t, like=rx, name="g_t")
    if rx.ndim != 2 or rx.shape[0] != rx.shape[1]:
        raise ValueError("r_x must be a square matrix.")
    dimension = rx.shape[0]
    for name, vector in (("f_minus", fm), ("f_plus", fp), ("r_t", rt), ("normal", nvec)):
        if vector.ndim != 1 or vector.shape[0] != dimension:
            raise ValueError(f"{name} must be a rank-1 tensor of length {dimension}.")
    denominator = torch.dot(nvec, fm) + gt
    _require_transversal(denominator.reshape(()), transversality_tol)
    numerator = fp - rx @ fm - rt
    return rx + torch.outer(numerator, nvec) / denominator
