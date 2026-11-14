"""
Gradient-flow smoke tests for differentiable transport operators.
"""

from __future__ import annotations

import torch

from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.core.state import UpdateItem, UpdateMode


def test_advection_diffusion_supports_backward() -> None:
    """AdvectionDiffusion forward result should keep gradient connection."""
    model = AdvectionDiffusion(
        spatial_dims=1,
        diffusivity=1e-5,
        grid_spacing=(0.01,),
        splitting_method="combined",
    )
    model.track_conservation = False

    concentration = torch.linspace(0.1, 1.0, 10, dtype=torch.float32).unsqueeze(0)
    concentration = concentration.requires_grad_(True)
    state = {
        "tracer_concentration": concentration,
        "velocity_field": torch.zeros_like(concentration),
        "source_term": torch.zeros_like(concentration),
    }

    output = model.forward(state, time=0.0, dt=0.1)
    tracer_update = output["tracer_concentration"]
    assert isinstance(tracer_update, UpdateItem)
    assert tracer_update.mode == UpdateMode.ACCUMULATE
    assert tracer_update.per_timestep is True
    loss = tracer_update.value.sum()
    loss.backward()

    assert concentration.grad is not None
    assert torch.isfinite(concentration.grad).all()


def test_fickian_diffusion_supports_backward() -> None:
    """FickianDiffusion forward result should keep gradient connection."""
    model = FickianDiffusion(
        spatial_dims=1,
        diffusivity=1e-5,
        grid_spacing=(0.01,),
    )
    model.track_conservation = False

    concentration = torch.linspace(0.1, 1.0, 10, dtype=torch.float32).unsqueeze(0)
    concentration = concentration.requires_grad_(True)
    state = {
        "tracer_concentration": concentration,
        "source_term": torch.zeros_like(concentration),
    }

    output = model.forward(state, time=0.0, dt=0.1)
    tracer_update = output["tracer_concentration"]
    assert isinstance(tracer_update, UpdateItem)
    assert tracer_update.mode == UpdateMode.ACCUMULATE
    assert tracer_update.per_timestep is True
    loss = tracer_update.value.sum()
    loss.backward()

    assert concentration.grad is not None
    assert torch.isfinite(concentration.grad).all()
