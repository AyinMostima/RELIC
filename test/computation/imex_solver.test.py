"""
Tests for IMEX Euler solver (L1 unit checks).

These tests validate the IMEX Euler update against a linear split system
with an analytical one-step update.
"""

import torch

from relic.computation.solvers import SolverFactory, SolverStatus


def _linear_explicit(y, t, a):
    return a * y


def _linear_implicit(y, t, b):
    return b * y


def test_imex_euler_linear_split_single_step():
    """IMEX Euler matches the analytical one-step update for linear split."""
    a = -1.0
    b = -5.0
    y0 = torch.tensor([1.0])
    dt = 0.1
    explicit_substeps = 4

    solver = SolverFactory.create(
        solver_name="imex_euler",
        name="imex_test",
        config={
            "max_iterations": 50,
            "rtol": 1.0e-8,
            "atol": 1.0e-10,
            "explicit_substeps": explicit_substeps,
            "damping_factor": 1.0,
        },
    )

    result = solver.solve(
        explicit_func=lambda y, t: _linear_explicit(y, t, a),
        implicit_func=lambda y, t: _linear_implicit(y, t, b),
        y_current=y0,
        t_current=0.0,
        dt=dt,
    )

    dt_explicit = dt / float(explicit_substeps)
    explicit_factor = (1.0 + dt_explicit * a) ** explicit_substeps
    expected = y0 * explicit_factor / (1.0 - dt * b)

    assert result.status == SolverStatus.SUCCESS
    assert torch.allclose(result.solution, expected, atol=1.0e-6, rtol=1.0e-6)


def test_imex_euler_linear_split_no_substeps():
    """IMEX Euler reduces to single explicit step when substeps=1."""
    a = -0.5
    b = -2.0
    y0 = torch.tensor([2.0])
    dt = 0.2
    explicit_substeps = 1

    solver = SolverFactory.create(
        solver_name="imex_euler",
        name="imex_test_single",
        config={
            "max_iterations": 50,
            "rtol": 1.0e-8,
            "atol": 1.0e-10,
            "explicit_substeps": explicit_substeps,
            "damping_factor": 1.0,
        },
    )

    result = solver.solve(
        explicit_func=lambda y, t: _linear_explicit(y, t, a),
        implicit_func=lambda y, t: _linear_implicit(y, t, b),
        y_current=y0,
        t_current=0.0,
        dt=dt,
    )

    explicit_factor = 1.0 + dt * a
    expected = y0 * explicit_factor / (1.0 - dt * b)

    assert result.status == SolverStatus.SUCCESS
    assert torch.allclose(result.solution, expected, atol=1.0e-6, rtol=1.0e-6)
