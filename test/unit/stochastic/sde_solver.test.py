"""Fixed-step Euler-Maruyama solver contract (relic.stochastic.sde_solver).

Covers caller-supplied Wiener increments (the Brownian-tree path handed in by
a Functor-owned substream), their validation, the generator-drawn default, and
the fixed-step-only boundary of the standalone solver.
"""

from __future__ import annotations

import math

import pytest
import torch

from relic.stochastic.sde_solver import EulerMaruyamaSolver, SDEParameters


def _solver(dt: float = 0.5) -> EulerMaruyamaSolver:
    return EulerMaruyamaSolver(
        SDEParameters(dt=dt, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=lambda x, t: -0.5 * x,
        diffusion_func=lambda x, t: torch.full_like(x, 2.0),
    )


def test_supplied_increments_replace_the_solver_draw_exactly():
    solver = _solver(dt=0.5)
    x0 = torch.tensor([1.0, -2.0], dtype=torch.float64)
    increments = torch.tensor([[[0.3, -0.1], [0.2, 0.4]]], dtype=torch.float64)  # (1, 2, 2)
    result = solver.sample(x0, num_samples=1, num_steps=2, seed=0, wiener_increments=increments)
    x1 = x0 + (-0.5 * x0) * 0.5 + 2.0 * increments[0, 0]
    x2 = x1 + (-0.5 * x1) * 0.5 + 2.0 * increments[0, 1]
    torch.testing.assert_close(result["solution"][0, 1], x1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result["solution"][0, 2], x2, rtol=0.0, atol=0.0)
    # The same call without increments draws from the seeded stream instead.
    drawn = solver.sample(x0, num_samples=1, num_steps=2, seed=0)
    assert not torch.allclose(drawn["solution"][0, 1], x1)


def test_generator_draw_matches_sqrt_dt_scaling():
    solver = _solver(dt=0.25)
    x0 = torch.zeros(3, dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)
    result = solver.sample(x0, num_samples=1, num_steps=1, generator=generator)
    generator.manual_seed(7)
    xi = torch.randn((1, 3), dtype=torch.float64, generator=generator)
    expected = 2.0 * xi[0] * math.sqrt(0.25)
    torch.testing.assert_close(result["solution"][0, 1], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "increments, error",
    [
        (torch.zeros(2, 1, 2, dtype=torch.float64), ValueError),
        (torch.zeros(1, 2, 2, dtype=torch.float64), ValueError),
        (torch.zeros(1, 1, 2, dtype=torch.float32), ValueError),
        (torch.full((1, 1, 2), float("nan"), dtype=torch.float64), ValueError),
        ([[[0.0, 0.0]]], TypeError),
    ],
)
def test_invalid_supplied_increments_are_rejected(increments, error):
    solver = _solver()
    with pytest.raises(error):
        solver.sample(
            torch.zeros(2, dtype=torch.float64), num_samples=1, num_steps=1, seed=0,
            wiener_increments=increments,
        )


def test_standalone_solver_stays_fixed_step():
    with pytest.raises(ValueError, match="fixed-step"):
        EulerMaruyamaSolver(
            SDEParameters(dt=0.1, adaptive=True, method="euler_maruyama", diffusion_structure="diagonal"),
            drift_func=lambda x, t: x,
            diffusion_func=lambda x, t: x,
        )


def test_drawn_seed_is_recorded_and_reproduces_the_path():
    solver = _solver()
    x0 = torch.zeros(2, dtype=torch.float64)
    first = solver.sample(x0, num_samples=3, num_steps=4)

    assert isinstance(first["seed"], int)
    again = solver.sample(x0, num_samples=3, num_steps=4, seed=first["seed"])
    assert torch.equal(first["solution"], again["solution"])


def test_explicit_seed_is_recorded():
    result = _solver().sample(torch.zeros(1, dtype=torch.float64), num_samples=1, num_steps=1, seed=5)
    assert result["seed"] == 5


def test_sde_method_and_diffusion_structure_are_required():
    with pytest.raises(TypeError, match="method"):
        SDEParameters(dt=0.1, diffusion_structure="diagonal")
    with pytest.raises(TypeError, match="diffusion_structure"):
        SDEParameters(dt=0.1, method="euler_maruyama")
    parameters = SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal")
    assert (parameters.method, parameters.diffusion_structure) == ("euler_maruyama", "diagonal")
