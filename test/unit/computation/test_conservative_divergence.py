import torch

from relic.computation.operators.functional import (
    compute_conservative_divergence_2d,
    compute_conservative_divergence_3d,
)


def test_conservative_divergence_2d_periodic_telescopes_to_zero():
    torch.manual_seed(42)
    dtype = torch.float64
    batch, ny, nx = 2, 32, 32
    dx, dy = 1.0, 1.0

    flux_x = torch.randn(batch, ny, nx, dtype=dtype)
    flux_y = torch.randn(batch, ny, nx, dtype=dtype)

    div = compute_conservative_divergence_2d(
        flux_x,
        flux_y,
        dx,
        dy,
        boundary_method="periodic",
    )
    net = (div.sum() * dx * dy).item()

    assert abs(net) < 1.0e-12


def test_conservative_divergence_3d_periodic_telescopes_to_zero():
    torch.manual_seed(42)
    dtype = torch.float64
    batch, nz, ny, nx = 1, 16, 16, 16
    dx = dy = dz = 1.0

    flux_x = torch.randn(batch, nz, ny, nx, dtype=dtype)
    flux_y = torch.randn(batch, nz, ny, nx, dtype=dtype)
    flux_z = torch.randn(batch, nz, ny, nx, dtype=dtype)

    div = compute_conservative_divergence_3d(
        flux_x,
        flux_y,
        flux_z,
        dx,
        dy,
        dz,
        boundary_method="periodic",
    )
    net = (div.sum() * dx * dy * dz).item()

    assert abs(net) < 1.0e-12
