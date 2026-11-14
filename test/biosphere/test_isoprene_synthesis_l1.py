#!/usr/bin/env python3
"""
L1 unit test for mechanistic isoprene synthesis.

Validates forward execution and conservation diagnostics without altering state.
"""

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.bvoc.synthesis.isoprene import IsopreneSynthesis


@pytest.fixture
def isoprene_functor():
    """Create the canonical mechanistic isoprene Functor."""
    return IsopreneSynthesis(name="test_isoprene_synthesis")


def test_forward_and_diagnostics(isoprene_functor):
    """Ensure forward pass produces outputs and diagnostics are emitted."""
    state = {
        "air_temperature": torch.tensor([303.15], dtype=torch.float32),  # K
        "gross_photosynthesis": torch.tensor([10.0], dtype=torch.float32),  # umol m-2 s-1
        "electron_transport_rate": torch.tensor([20.0], dtype=torch.float32),  # umol e- m-2 s-1
        "atmospheric_co2_ppm": torch.tensor([400.0], dtype=torch.float32),  # umol mol-1
        "leaf_area_index": torch.tensor([1.0], dtype=torch.float32),
    }

    outputs = isoprene_functor(state=state, time=0.0, dt=60.0)
    emission = outputs["isoprene_emission"]
    assert isinstance(emission, UpdateItem)
    assert emission.mode is UpdateMode.REPLACE
    assert emission.per_timestep is False
    assert emission.unit == "mol m-2 s-1"
    assert emission.value.shape == torch.Size([1])
    assert "isoprenes_emission" not in outputs
    assert "isoprene_emission_mechanistic" not in outputs

    diagnostics = isoprene_functor.diagnose_conservation(state, outputs)
    carbon_allocation = diagnostics["bvoc_carbon_allocation_flux"]
    assert torch.isfinite(carbon_allocation).all()
    assert torch.all(carbon_allocation >= 0)


