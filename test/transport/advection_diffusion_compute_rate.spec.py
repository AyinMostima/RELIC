"""L1 tests for AdvectionDiffusion.compute_rate state-dict interface."""

from __future__ import annotations

import pytest
import torch

from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from test.utils.forward_call import call_forward


def _build_module(spatial_dims: int) -> AdvectionDiffusion:
    grid_spacing = tuple([0.1] * spatial_dims)
    return AdvectionDiffusion(
        spatial_dims=spatial_dims,
        diffusivity=1.0e-5,
        grid_spacing=grid_spacing,
    )


def _extract_tracer_rate(output):
    """Normalize forward output to tracer-rate tensor."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if "tracer_concentration" in output:
            return output["tracer_concentration"]
        if "advection_diffusion_flux_tracer_concentration" in output:
            return output["advection_diffusion_flux_tracer_concentration"]
    raise TypeError(f"Unsupported advection-diffusion output type: {type(output)}")


def test_compute_rate_state_dict_tracer_concentration_shape_and_tensor_type() -> None:
    """compute_rate must accept state dict input without Tensor truth-value errors."""
    ad = _build_module(spatial_dims=1)
    concentration = torch.linspace(0.0, 1.0, 32, dtype=torch.float32)
    state = {
        "tracer_concentration": concentration,
        "velocity_field": torch.zeros_like(concentration),
    }

    output = call_forward(ad, state=state)
    rate = _extract_tracer_rate(output)

    assert isinstance(rate, torch.Tensor)
    assert rate.shape == concentration.shape
    assert torch.isfinite(rate).all()


@pytest.mark.parametrize("concentration_key", ["concentration_field", "concentration"])
def test_compute_rate_state_dict_rejects_noncanonical_concentration_keys(concentration_key: str) -> None:
    """Only tracer_concentration is a production state port."""
    ad = _build_module(spatial_dims=1)
    concentration = torch.linspace(0.0, 1.0, 24, dtype=torch.float32)
    state = {
        concentration_key: concentration,
        "velocity_field": torch.zeros_like(concentration),
    }

    with pytest.raises(ValueError, match="tracer_concentration"):
        call_forward(ad, state=state)


def test_compute_rate_state_dict_missing_concentration_raises_value_error() -> None:
    """compute_rate must fail fast when no concentration key is present."""
    ad = _build_module(spatial_dims=1)
    state = {"velocity_field": torch.zeros(16, dtype=torch.float32)}

    with pytest.raises(ValueError, match="State must contain"):
        call_forward(ad, state=state)


def test_compute_rate_state_dict_supports_velocity_field_encoding() -> None:
    """compute_rate should accept velocity encoded as a packed velocity_field tensor."""
    ad = _build_module(spatial_dims=2)
    concentration = torch.rand(1, 10, 12, dtype=torch.float32)
    velocity_field = torch.zeros(1, 10, 12, 2, dtype=torch.float32)
    velocity_field[..., 0] = 0.1
    velocity_field[..., 1] = -0.05
    state = {
        "tracer_concentration": concentration,
        "velocity_field": velocity_field,
    }

    with pytest.raises(ValueError, match="CFL stability violation"):
        call_forward(ad, state=state, dt=1.0)

    output = call_forward(ad, state=state, dt=0.5)
    rate = _extract_tracer_rate(output)

    assert isinstance(rate, torch.Tensor)
    assert rate.shape == concentration.shape
    # The multidimensional CFL condition is dt * (|u| / dx + |v| / dy).
    assert "diagnostics.effective_time_step" not in output
    assert float(output["diagnostics.cfl_number"]) == pytest.approx(0.75)


def test_compute_rate_state_dict_3d_shape_preserved() -> None:
    """compute_rate should preserve tensor shape for 3D inputs."""
    ad = _build_module(spatial_dims=3)
    concentration = torch.rand(1, 6, 5, 4, dtype=torch.float32)
    velocity_field = torch.zeros(1, 6, 5, 4, 3, dtype=torch.float32)
    state = {
        "tracer_concentration": concentration,
        "velocity_field": velocity_field,
    }

    output = call_forward(ad, state=state)
    rate = _extract_tracer_rate(output)

    assert isinstance(rate, torch.Tensor)
    assert rate.shape == concentration.shape


def test_compute_rate_3d_respects_z_direction_diffusion() -> None:
    """3D diffusion must include z-direction curvature and not collapse to 2D."""
    ad = _build_module(spatial_dims=3)
    z_profile = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=torch.float32)
    concentration = z_profile.view(1, 1, 1, 4).expand(1, 5, 6, 4).clone()
    zeros = torch.zeros_like(concentration)
    state = {
        "tracer_concentration": concentration,
        "velocity_field": torch.stack((zeros, zeros, zeros), dim=-1),
    }

    output = call_forward(ad, state=state, dt=0.01)
    rate = _extract_tracer_rate(output)

    assert isinstance(rate, torch.Tensor)
    assert rate.shape == concentration.shape
    assert torch.max(torch.abs(rate)).item() > 0.0
