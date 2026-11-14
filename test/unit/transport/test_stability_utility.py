"""Unit tests for shared transport stability utility."""

from __future__ import annotations

import math

from relic.scientific.transport.stability import compute_stable_timestep


def test_compute_stable_timestep_cfl_limited() -> None:
    """Advective CFL limit should reduce dt when velocity is large."""
    dt, diag = compute_stable_timestep(
        dt=10.0,
        grid_spacing=1.0,
        velocity_magnitude=2.0,
        diffusivity=0.0,
        cfl_limit=1.0,
        safety_factor=1.0,
        min_timestep=1e-9,
        max_timestep=1e9,
    )
    assert math.isclose(dt, 0.5)
    assert math.isclose(diag["cfl_number"], 20.0)
    assert math.isclose(diag["advective_limit"], 0.5)


def test_compute_stable_timestep_diffusion_limited() -> None:
    """Diffusive stability limit should reduce dt when diffusivity is large."""
    dt, diag = compute_stable_timestep(
        dt=10.0,
        grid_spacing=1.0,
        velocity_magnitude=0.0,
        diffusivity=1.0,
        diffusion_stability_limit=0.5,
        safety_factor=1.0,
        min_timestep=1e-9,
        max_timestep=1e9,
    )
    assert math.isclose(dt, 0.5)
    assert math.isclose(diag["diffusion_number"], 10.0)
    assert math.isclose(diag["diffusive_limit"], 0.5)


def test_compute_stable_timestep_combined_constraint() -> None:
    """Combined advection-diffusion should choose the strictest dt limit."""
    dt, diag = compute_stable_timestep(
        dt=10.0,
        grid_spacing=1.0,
        velocity_magnitude=1.0,
        diffusivity=10.0,
        cfl_limit=1.0,
        diffusion_stability_limit=0.5,
        safety_factor=1.0,
        min_timestep=1e-9,
        max_timestep=1e9,
    )
    assert math.isclose(diag["advective_limit"], 1.0)
    assert math.isclose(diag["diffusive_limit"], 0.05)
    assert math.isclose(dt, 0.05)


def test_compute_stable_timestep_edge_cases() -> None:
    """Edge cases should remain bounded and stable."""
    dt_no_flow, diag_no_flow = compute_stable_timestep(
        dt=5.0,
        grid_spacing=2.0,
        velocity_magnitude=0.0,
        diffusivity=0.0,
        min_timestep=1e-6,
        max_timestep=1e6,
    )
    assert math.isclose(dt_no_flow, 5.0)
    assert math.isclose(diag_no_flow["cfl_number"], 0.0)
    assert math.isclose(diag_no_flow["diffusion_number"], 0.0)

    dt_zero_spacing, _ = compute_stable_timestep(
        dt=2.0,
        grid_spacing=0.0,
        velocity_magnitude=10.0,
        diffusivity=10.0,
        min_timestep=1e-3,
        max_timestep=100.0,
    )
    assert math.isclose(dt_zero_spacing, 2.0)

