"""Shared numerical stability utilities for transport functors.

This module centralizes CFL and diffusion stability constraints used by
advection, diffusion, and advection-diffusion transport processes.

References:
    - Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial
      difference equations of mathematical physics. Mathematische Annalen, 100.
    - Hundsdorfer, W. and Verwer, J.G. (2003). Numerical Solution of
      Time-Dependent Advection-Diffusion-Reaction Equations. Springer.

Mathematical expressions:
    - CFL condition: CFL = |v| * dt / dx <= CFL_max
    - Diffusion stability: D * dt / dx^2 <= S_max
    - Stable step: dt_stable = min(CFL_max * dx / |v|, S_max * dx^2 / D)

Parameter meanings:
    - dt: Proposed time step [s]
    - velocity_magnitude: Maximum velocity magnitude [m s-1]
    - diffusivity: Diffusion coefficient [m2 s-1]
    - grid_spacing: Minimum grid spacing [m]
    - cfl_limit: Maximum CFL number [dimensionless]
    - diffusion_stability_limit: Maximum diffusion stability number [dimensionless]
    - safety_factor: Safety multiplier applied to stability limits [dimensionless]
    - min_timestep: Lower bound for final time step [s]
    - max_timestep: Upper bound for final time step [s]
"""

from __future__ import annotations

from typing import Dict, Tuple


def compute_stable_timestep(
    dt: float,
    grid_spacing: float,
    velocity_magnitude: float = 0.0,
    diffusivity: float = 0.0,
    cfl_limit: float = 1.0,
    diffusion_stability_limit: float = 0.5,
    safety_factor: float = 0.9,
    min_timestep: float = 1e-6,
    max_timestep: float = 1e6,
) -> Tuple[float, Dict[str, float]]:
    """Compute a numerically stable time step for advection-diffusion updates."""
    requested_dt = float(dt)
    effective_dt = requested_dt
    dx = float(grid_spacing)
    velocity = max(float(velocity_magnitude), 0.0)
    diffusion = max(float(diffusivity), 0.0)

    diagnostics: Dict[str, float] = {
        "cfl_number": 0.0,
        "diffusion_number": 0.0,
        "advective_limit": float("inf"),
        "diffusive_limit": float("inf"),
        "was_reduced": 0.0,
    }

    # Degenerate spacing: only apply global min/max bounds.
    if dx <= 0.0:
        bounded_dt = max(min(effective_dt, float(max_timestep)), float(min_timestep))
        diagnostics["was_reduced"] = 1.0 if bounded_dt < requested_dt else 0.0
        return bounded_dt, diagnostics

    safe_factor = max(float(safety_factor), 0.0)

    if velocity > 0.0:
        advective_limit = float(cfl_limit) * dx / velocity
        diagnostics["advective_limit"] = advective_limit
        diagnostics["cfl_number"] = velocity * requested_dt / dx
        effective_dt = min(effective_dt, advective_limit * safe_factor)

    if diffusion > 0.0:
        diffusive_limit = float(diffusion_stability_limit) * (dx ** 2) / diffusion
        diagnostics["diffusive_limit"] = diffusive_limit
        diagnostics["diffusion_number"] = diffusion * requested_dt / (dx ** 2)
        effective_dt = min(effective_dt, diffusive_limit * safe_factor)

    bounded_dt = max(min(effective_dt, float(max_timestep)), float(min_timestep))
    diagnostics["was_reduced"] = 1.0 if bounded_dt < requested_dt else 0.0
    return bounded_dt, diagnostics

