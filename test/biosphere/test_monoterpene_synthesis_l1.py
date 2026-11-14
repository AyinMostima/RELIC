#!/usr/bin/env python3
"""
L1 unit test for mechanistic monoterpene synthesis.

Validates forward outputs and carbon diagnostics.
"""

import pytest
import torch

from relic.scientific.biosphere.bvoc.synthesis.monoterpene import MonoterpeneSynthesis


@pytest.fixture
def monoterpene_functor():
    """Create MonoterpeneSynthesis from the project parameter database."""
    return MonoterpeneSynthesis()


def test_forward_and_diagnostics(monoterpene_functor):
    """Ensure forward pass produces outputs and diagnostics are finite."""
    state = {
        "air_temperature": torch.tensor([303.15], dtype=torch.float32),
        "gross_photosynthesis": torch.tensor([12.0], dtype=torch.float32),
        MonoterpeneSynthesis.POOL_STATE_KEY: monoterpene_functor.reset_pools(),
    }
    outputs = monoterpene_functor(state=state, time=0.0, dt=60.0)
    assert "monoterpenes_emission" in outputs
    assert outputs["monoterpenes_emission"].value.shape == torch.Size([1])
    assert outputs["monoterpenes_emission"].unit == "mol m-2 s-1"
    assert outputs["monoterpenes_emission"].source_functor == monoterpene_functor.name
    diagnostics = monoterpene_functor.diagnose_conservation(state, outputs)
    assert "bvoc_carbon_allocation_flux" in diagnostics
    allocation = diagnostics["bvoc_carbon_allocation_flux"]
    assert torch.isfinite(allocation).all()
    assert torch.all(allocation >= 0)


