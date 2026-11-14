"""Unit tests for AdvectionDiffusion state-default handling."""

import pytest
import torch

from relic.scientific.transport.advection_diffusion import AdvectionDiffusion


def _build_functor() -> AdvectionDiffusion:
    return AdvectionDiffusion(spatial_dims=1, diffusivity=0.1, grid_spacing=1.0)


def test_absent_source_term_selects_explicit_zero_source():
    functor = _build_functor()
    template = torch.ones(5, dtype=torch.float32)

    prepared = functor._prepare_source_term(None, template)

    torch.testing.assert_close(prepared, torch.zeros_like(template))
    assert not hasattr(functor, "source_term_default")


def test_source_term_scalar_is_rejected_as_non_tensor_state():
    functor = _build_functor()
    template = torch.ones(3, dtype=torch.float32)

    with pytest.raises(TypeError, match="source_term must be a torch.Tensor"):
        functor._prepare_source_term(2.0, template)

