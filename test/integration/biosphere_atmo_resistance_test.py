import pytest
import torch

from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.core.state.types import UpdateItem


def _build_minimal_state(friction_velocity: float = 0.5):
    return {
        "friction_velocity": torch.tensor(friction_velocity, dtype=torch.float64),
        "air_temperature": torch.tensor(298.15, dtype=torch.float64),
        "atmospheric_pressure": torch.tensor(101325.0, dtype=torch.float64),
    }


def test_friction_velocity_is_required():
    coupling = BiosphereAtmosphereCoupling(name="ba_strict", config={})
    state = _build_minimal_state()
    del state["friction_velocity"]

    with pytest.raises(ValueError, match="friction_velocity"):
        coupling.forward(state, 0.0, 1.0)


def test_boundary_conductance_follows_square_root_wind_response():
    coupling = BiosphereAtmosphereCoupling(name="ba_boundary", config={})
    low = coupling.forward(_build_minimal_state(0.5), 0.0, 1.0)
    high = coupling.forward(_build_minimal_state(2.0), 0.0, 1.0)

    low_item = low["boundary_layer_conductance"]
    high_item = high["boundary_layer_conductance"]
    assert isinstance(low_item, UpdateItem)
    assert isinstance(high_item, UpdateItem)
    torch.testing.assert_close(high_item.value, 2.0 * low_item.value)


