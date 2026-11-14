import torch

# ASCII-only comments.
# Test that FickianDiffusion.forward returns increments (delta), not new state,
# and that the delta matches explicit Euler with a simple 1D setup.

from relic.scientific.transport.diffusion import FickianDiffusion
from relic.computation.operators.functional import compute_laplacian_1d
from relic.core.state import UpdateItem, UpdateMode


def test_diffusion_forward_returns_delta_and_matches_explicit_euler():
    device = torch.device('cpu')
    spatial_dims = 1
    K = 1.0e-3  # m2 s-1
    dx = 1.0
    dt = 0.1

    # Create a simple 1D concentration with a peak in the middle
    c = torch.zeros(1, 11, device=device)
    c[..., 5] = 1.0

    diff = FickianDiffusion(
        spatial_dims=spatial_dims,
        diffusivity=K,
        grid_spacing=(dx,),
        time_integration='explicit',
        device=device,
    )

    state = {
        'tracer_concentration': c.clone(),
        'source_term': torch.zeros_like(c),
    }

    result = diff.forward(state, time=0.0, dt=dt)

    # Must return increment
    assert 'tracer_concentration' in result
    update = result['tracer_concentration']
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    delta = update.value

    # Expected explicit Euler: c_new = c + dt * (K * laplacian)
    lap = compute_laplacian_1d(c, dx, boundary_method="zero")
    expected_delta = dt * (K * lap)

    # Compare interior points to avoid boundary policy differences
    assert torch.allclose(delta[..., 1:-1], expected_delta[..., 1:-1], atol=1e-7, rtol=1e-6)

