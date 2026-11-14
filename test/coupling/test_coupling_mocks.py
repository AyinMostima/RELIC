"""Contract tests for shared coupling mocks."""

import torch

from test.fixtures.coupling_mocks import DummyPhotosynthesis


def test_dummy_photosynthesis_returns_existing_value():
    mock = DummyPhotosynthesis(name="photo_contract")
    state = {"net_assimilation_rate": torch.tensor(2.5)}
    updates = mock.forward(state, time=0.0, dt=1.0)

    assert updates["net_assimilation_rate"] is state["net_assimilation_rate"]
    assert mock.get_dependencies() == []


def test_dummy_photosynthesis_default_device_respected_cpu():
    device = torch.device("cpu")
    mock = DummyPhotosynthesis(name="photo_device", device=device)
    updates = mock.forward({}, time=0.0, dt=1.0)

    assert torch.is_tensor(updates["net_assimilation_rate"])
    assert updates["net_assimilation_rate"].device == device


