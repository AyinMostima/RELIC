"""Scientific validation for the bounded one-dimensional ACP column baseline.

References:
    Crank (1975), The Mathematics of Diffusion.
    Patankar (1980), Numerical Heat Transfer and Fluid Flow.
    LeVeque (2007), Finite Difference Methods for ODEs and PDEs.

Mathematical expression:
    dc/dt = K d2c/dz2,
    c(z, t) = c0 + A exp(-K k2 t) cos(k z),
    k = pi/L, and dc/dz = 0 at z = 0 and z = L.

Parameter meanings:
    c is tracer mass concentration [kg m-3], K is the constant manufactured
    diffusivity [m2 s-1], z is height [m], L is column height [m], and t is
    time [s]. This is a passive uniform-grid verification case, not a full
    atmospheric boundary-layer closure.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.transport.diffusion import FickianDiffusion


def _advance_fixed_step_column(
    *,
    cell_count: int,
    final_time: float = 40.0,
) -> tuple[float, float, float]:
    """Advance the cosine eigenmode with a fixed stable explicit time step.

    References:
        Crank (1975); Patankar (1980); LeVeque (2007).
    Mathematical expression:
        c_i^(n+1) = c_i^n + dt K (c_(i+1)-2c_i+c_(i-1))/dz2.
        The chosen dt satisfies K dt/dz2 = 0.2 and therefore dt = O(dz2).
    Parameter meanings:
        cell_count is N [-], dz is L/N [m], and final_time is T [s].
    """

    length = 100.0
    diffusivity = 1.0
    spacing = length / cell_count
    nominal_dt = 0.2 * spacing * spacing / diffusivity
    step_count = round(final_time / nominal_dt)
    dt = final_time / step_count
    wave_number = math.pi / length
    centers = (torch.arange(cell_count, dtype=torch.float64) + 0.5) * spacing
    initial = 2.0 + 0.5 * torch.cos(wave_number * centers)
    concentration = initial.clone()
    functor = FickianDiffusion(
        spatial_dims=1,
        diffusivity=diffusivity,
        grid_spacing=spacing,
        boundary_method="zero",
        non_negative=True,
        track_conservation=True,
    )

    maximum_ledger_error = 0.0
    for step_index in range(step_count):
        result = functor.forward(
            {"tracer_concentration": concentration},
            time=step_index * dt,
            dt=dt,
        )
        update = result["tracer_concentration"]
        assert isinstance(update, UpdateItem)
        assert update.mode is UpdateMode.ACCUMULATE
        assert update.per_timestep is True
        assert update.basis == "per_step"
        assert update.unit == "kg m-3"
        torch.testing.assert_close(
            result["diagnostics.diffusion_stability"],
            torch.tensor(0.2, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-14,
        )
        concentration = concentration + update.value
        maximum_ledger_error = max(
            maximum_ledger_error,
            abs(float(result["diagnostics.conservation_error"])),
        )

    exact = 2.0 + 0.5 * math.exp(
        -diffusivity * wave_number * wave_number * final_time
    ) * torch.cos(wave_number * centers)
    l2_error = float(torch.sqrt(torch.mean((concentration - exact) ** 2)))
    mass_drift = float(
        torch.abs(torch.sum(concentration - initial) * spacing)
    )
    return l2_error, maximum_ledger_error, mass_drift


def test_zero_flux_cosine_manufactured_solution_converges_second_order() -> None:
    """Require second-order combined error on three successively refined grids."""

    # References: Crank (1975); Patankar (1980); LeVeque (2007).
    # Expression: p_h = log(E_h/E_(h/2))/log(2), with dt proportional to dz2.
    # Parameters: E_h is the L2 error and p_h is the observed order [-].
    results = [_advance_fixed_step_column(cell_count=n) for n in (20, 40, 80)]
    errors = [result[0] for result in results]
    observed_orders = [
        math.log(errors[index] / errors[index + 1]) / math.log(2.0)
        for index in range(2)
    ]

    assert all(error > 0.0 for error in errors)
    assert all(1.8 <= order <= 2.2 for order in observed_orders)
    assert errors[2] < 4.0e-7
    assert max(result[1] for result in results) < 1.0e-12
    assert max(result[2] for result in results) < 1.0e-12


def test_closed_column_source_storage_and_boundary_ledgers_close() -> None:
    """Require the independent source-storage ledger to close exactly."""

    # References: Patankar (1980); Hundsdorfer and Verwer (2003).
    # Expression: Delta M - dt integral(S dV) + dt sum(j_out A) = 0.
    # Parameters: M is stored tracer mass [kg], S is source [kg m-3 s-1],
    # j_out is outward boundary flux [kg m-2 s-1], and dt is time [s].
    spacing = 2.0
    dt = 0.1
    concentration = torch.full((8,), 1.5, dtype=torch.float64)
    source = torch.zeros_like(concentration)
    source[0] = 0.25
    functor = FickianDiffusion(
        spatial_dims=1,
        diffusivity=1.0,
        grid_spacing=spacing,
        boundary_method="zero",
        non_negative=True,
        track_conservation=True,
    )

    result = functor.forward(
        {
            "tracer_concentration": concentration,
            "source_term": source,
        },
        time=0.0,
        dt=dt,
    )
    concentration_new = concentration + result["tracer_concentration"].value
    storage_change = torch.sum(concentration_new - concentration) * spacing
    source_input = dt * torch.sum(source) * spacing
    boundary_flux = result[
        "tracer_concentration_diffusive_boundary_flux"
    ]
    assert isinstance(boundary_flux, UpdateItem)
    assert boundary_flux.mode is UpdateMode.REPLACE
    assert boundary_flux.unit == "kg m-2 s-1"
    assert boundary_flux.basis == "per_second"
    assert boundary_flux.per_timestep is False
    assert boundary_flux.value.shape == (2,)
    torch.testing.assert_close(
        boundary_flux.value,
        torch.zeros(2, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    boundary_export = dt * torch.sum(boundary_flux.value)

    torch.testing.assert_close(
        storage_change - source_input + boundary_export,
        torch.tensor(0.0, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert float(result["diagnostics.conservation_error"]) < 1.0e-14


def test_source_and_diffusivity_paths_remain_differentiable() -> None:
    """Require finite nonzero gradients without tensor-to-scalar warnings."""

    # References: Crank (1975); Patankar (1980).
    # Expression: c_next = c + dt(div(K grad(c)) + S).
    # Parameters: K is diffusivity [m2 s-1] and S is source [kg m-3 s-1].
    diffusivity = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    concentration = torch.tensor(
        [1.0, 1.2, 1.4, 1.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    source = torch.tensor(
        [0.2, 0.0, 0.0, 0.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    functor = FickianDiffusion(
        spatial_dims=1,
        diffusivity=diffusivity,
        grid_spacing=2.0,
        boundary_method="zero",
        non_negative=True,
        track_conservation=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Converting a tensor with requires_grad=True to a scalar",
            category=UserWarning,
        )
        result = functor.forward(
            {
                "tracer_concentration": concentration,
                "source_term": source,
            },
            time=0.0,
            dt=0.1,
        )

    concentration_new = concentration + result["tracer_concentration"].value
    weights = torch.arange(1.0, 5.0, dtype=torch.float64)
    objective = torch.sum(weights * concentration_new)
    objective.backward()

    assert diffusivity.grad is not None
    assert source.grad is not None
    assert concentration.grad is not None
    assert torch.isfinite(diffusivity.grad)
    assert torch.all(torch.isfinite(source.grad))
    assert torch.all(torch.isfinite(concentration.grad))
    torch.testing.assert_close(
        diffusivity.grad,
        torch.tensor(-0.0025, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        source.grad,
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        concentration.grad,
        torch.tensor([1.025, 2.0, 3.0, 3.975], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_explicit_column_rejects_unstable_fixed_time_step() -> None:
    """Reject an unstable requested step instead of changing it internally."""

    # References: von Neumann stability analysis; LeVeque (2007).
    # Expression: sigma = K dt/dz2 <= 1/2 for 1-D forward Euler diffusion.
    # Parameters: sigma is the diffusion number [-], K [m2 s-1], dt [s],
    # and dz [m].
    functor = FickianDiffusion(
        spatial_dims=1,
        diffusivity=1.0,
        grid_spacing=1.0,
        boundary_method="zero",
        non_negative=True,
        track_conservation=True,
    )
    concentration = torch.ones(8, dtype=torch.float64)

    with pytest.raises(ValueError, match="Diffusion stability violation"):
        functor.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.51,
        )
