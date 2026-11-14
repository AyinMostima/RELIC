"""Owner tests for relic/computation/solvers/hybrid_events.py (ledger D73).

References: Goebel, Sanfelice and Teel (2012); Kong et al. (2023) arXiv:2306.06862;
RELIC theory page docs/source/theory/hybrid_events.md.
Mathematical expression: tau_theta = -(g_x S_minus + g_theta)/(g_x f_minus + g_t);
S_plus = R_x S_minus + R_theta + (R_x f_minus + R_t - f_plus) tau_theta;
S = R_x + (f_plus - R_x f_minus - R_t) n^T / (n^T f_minus + g_t).
Parameter meanings: see the module docstrings; the manuscript counterexample uses
x(0)=theta, guard x=1, identity reset, f_minus=1, f_plus=2, horizon T=2.
Units: dimensionless normalized test coordinates and seconds.
Domain/constraints: transversal events only; grazing is refused.
Sign convention: tau increases with dt; S_plus - S_minus is the sensitivity jump.
"""

from __future__ import annotations

import pytest
import torch

from relic.computation.solvers.hybrid_events import (
    event_time_sensitivity,
    locate_guard_crossing,
    saltation_matrix,
    sensitivity_jump,
)

F64 = torch.float64


def _terminal_state(theta: torch.Tensor, *, f_minus: float, f_plus: float, threshold: float, horizon: float, reset_shift: float = 0.0) -> torch.Tensor:
    # Explicit one-step-per-segment hybrid trajectory used as the differentiable oracle.
    x0 = theta
    g_start = x0 - threshold
    g_end = x0 + f_minus * horizon - threshold
    tau = locate_guard_crossing(g_start, g_end, 0.0, horizon)
    x_minus = x0 + f_minus * tau
    # The reset adds reset_shift * theta so that R_theta = reset_shift in the formula check.
    x_plus = x_minus + reset_shift * theta
    return x_plus + f_plus * (horizon - tau)


def test_manuscript_counterexample_by_autograd_formula_and_finite_difference():
    theta = torch.tensor(0.3, dtype=F64, requires_grad=True)
    terminal = _terminal_state(theta, f_minus=1.0, f_plus=2.0, threshold=1.0, horizon=2.0)
    (autograd,) = torch.autograd.grad(terminal, theta)
    assert float(autograd) == pytest.approx(2.0, abs=1e-12)
    tau_theta = event_time_sensitivity(1.0, 0.0, 0.0, 1.0, 1.0)
    assert float(tau_theta) == pytest.approx(-1.0)
    jump = sensitivity_jump(1.0, 1.0, 0.0, 0.0, 1.0, 2.0, tau_theta)
    assert float(jump) == pytest.approx(2.0)
    h = 1e-6
    plus = _terminal_state(torch.tensor(0.3 + h, dtype=F64), f_minus=1.0, f_plus=2.0, threshold=1.0, horizon=2.0)
    minus = _terminal_state(torch.tensor(0.3 - h, dtype=F64), f_minus=1.0, f_plus=2.0, threshold=1.0, horizon=2.0)
    assert float((plus - minus) / (2 * h)) == pytest.approx(2.0, abs=1e-7)
    # The superseded expression without the event-time term misses the jump.
    assert float(sensitivity_jump(1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0)) == pytest.approx(1.0)


def test_reset_shift_and_fixed_time_event():
    theta = torch.tensor(0.3, dtype=F64, requires_grad=True)
    terminal = _terminal_state(theta, f_minus=1.0, f_plus=2.0, threshold=1.0, horizon=2.0, reset_shift=1.0)
    (autograd,) = torch.autograd.grad(terminal, theta)
    tau_theta = event_time_sensitivity(1.0, 0.0, 0.0, 1.0, 1.0)
    assert float(autograd) == pytest.approx(float(sensitivity_jump(1.0, 1.0, 1.0, 0.0, 1.0, 2.0, tau_theta)), abs=1e-12)
    # Fixed-time event g = t - t_e: g_x = 0, g_t = 1 -> tau_theta = 0 and S_plus = R_x S_minus + R_theta.
    fixed = event_time_sensitivity(0.0, 0.0, 1.0, 1.0, 1.0)
    assert float(fixed) == 0.0
    assert float(sensitivity_jump(1.0, 1.0, 0.5, 0.0, 1.0, 2.0, fixed)) == pytest.approx(1.5)


def test_grazing_is_refused_not_extrapolated():
    with pytest.raises(ValueError, match="grazing"):
        event_time_sensitivity(1.0, 0.0, -1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="grazing"):
        locate_guard_crossing(torch.tensor(-1e-14, dtype=F64), torch.tensor(1e-14, dtype=F64), 0.0, 1.0, transversality_tol=1e-12)
    with pytest.raises(ValueError, match="do not bracket"):
        locate_guard_crossing(0.5, 0.7, 0.0, 1.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        locate_guard_crossing(-0.5, 0.5, 0.0, 0.0)


def test_located_crossing_autograd_matches_the_implicit_event_time_formula():
    # Guard g = x(t) - theta with x(t) = x0 + f t: crossing at tau = (theta - x0)/f.
    theta = torch.tensor(3.0, dtype=F64, requires_grad=True)
    x0, rate, dt = 1.0, 0.5, 10.0
    g_start = x0 - theta
    g_end = x0 + rate * dt - theta
    tau = locate_guard_crossing(g_start, g_end, 0.0, dt)
    assert float(tau) == pytest.approx((3.0 - x0) / rate)
    (d_tau,) = torch.autograd.grad(tau, theta)
    expected = event_time_sensitivity(1.0, -1.0, 0.0, rate, 0.0)
    assert float(d_tau) == pytest.approx(float(expected)) == pytest.approx(1.0 / rate)


def test_saltation_matrix_reduces_to_the_scalar_jump_and_to_the_reset_jacobian_for_fixed_time_events():
    r_x = torch.eye(2, dtype=F64)
    f_minus = torch.tensor([1.0, 0.0], dtype=F64)
    f_plus = torch.tensor([2.0, 0.0], dtype=F64)
    r_t = torch.zeros(2, dtype=F64)
    normal = torch.tensor([1.0, 0.0], dtype=F64)
    saltation = saltation_matrix(r_x, f_minus, f_plus, r_t, normal, 0.0)
    # (f_plus - f_minus) n^T / (n^T f_minus) adds 1 to the (0, 0) entry.
    torch.testing.assert_close(saltation, torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=F64))
    # A fixed-time event has n = 0 and g_t = 1: the saltation matrix is the reset Jacobian
    # plus the rate jump times the (zero) normal, i.e. R_x itself.
    fixed = saltation_matrix(r_x, f_minus, f_plus, r_t, torch.zeros(2, dtype=F64), 1.0)
    torch.testing.assert_close(fixed, r_x)
    with pytest.raises(ValueError, match="grazing"):
        saltation_matrix(r_x, torch.tensor([0.0, 1.0], dtype=F64), f_plus, r_t, normal, 0.0)
    with pytest.raises(ValueError, match="square"):
        saltation_matrix(torch.ones((2, 3), dtype=F64), f_minus, f_plus, r_t, normal, 0.0)


def test_default_transversality_tolerance_is_read_from_the_parameter_database():
    """Without an explicit transversality_tol the three owners read one YAML key."""
    from pathlib import Path

    import yaml

    import relic
    from relic.parameters import get_global_manager

    key = "computation.solvers.hybrid_events.transversality_tolerance"
    database = Path(relic.__file__).parent / "parameters" / "database" / "relic" / "computation" / "solvers.yaml"
    entry = yaml.safe_load(database.read_text(encoding="utf-8"))["parameters"][key]
    for field_name in ("value", "units", "dtype", "bounds", "category", "description",
                       "references", "tags", "requires_grad", "provenance"):
        assert field_name in entry, field_name
    manager = get_global_manager()
    original = manager.get(key, required=True)
    assert original == pytest.approx(1.0e-12)
    near_grazing = (torch.tensor(-1e-14, dtype=F64), torch.tensor(1e-14, dtype=F64), 0.0, 1.0)
    r_x = torch.eye(2, dtype=F64)
    f_minus = torch.tensor([1.0e-14, 0.0], dtype=F64)
    f_plus = torch.tensor([2.0, 0.0], dtype=F64)
    normal = torch.tensor([1.0, 0.0], dtype=F64)
    with pytest.raises(ValueError, match="grazing"):
        locate_guard_crossing(*near_grazing)
    with pytest.raises(ValueError, match="grazing"):
        event_time_sensitivity(1.0, 0.0, 0.0, 1.0e-14, 1.0)
    with pytest.raises(ValueError, match="grazing"):
        saltation_matrix(r_x, f_minus, f_plus, torch.zeros(2, dtype=F64), normal, 0.0)
    try:
        manager.set_value(key, 1.0e-20)
        assert float(locate_guard_crossing(*near_grazing)) == pytest.approx(0.5)
        assert float(event_time_sensitivity(1.0, 0.0, 0.0, 1.0e-14, 1.0)) == pytest.approx(-1.0e14)
        saltation_matrix(r_x, f_minus, f_plus, torch.zeros(2, dtype=F64), normal, 0.0)
    finally:
        manager.set_value(key, original)
    assert manager.get(key, required=True) == original
