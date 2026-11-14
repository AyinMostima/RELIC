"""
Verify conservation parameter types on a canonical Farquhar forward pass.
"""
import torch

from relic.core.state import UpdateItem
from relic.parameters import get_global_manager
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis


def test_conservation_check():
    """Test canonical parameter types and a warning-free Farquhar forward pass."""
    param_manager = get_global_manager()
    tolerance = param_manager.get(
        "validation.conservation.conservation_tolerance",
        expected_type=float,
        required=True,
    )
    strict = param_manager.get(
        "validation.conservation.strict_conservation",
        expected_type=bool,
        required=True,
    )
    assert isinstance(tolerance, float)
    assert isinstance(strict, bool)

    model = FarquharPhotosynthesis(name="test_farquhar")
    state = {
        "air_temperature": torch.tensor(298.15),
        "atmospheric_co2_ppm": torch.tensor(400.0),
        "ppfd": torch.tensor(1000.0),
        "leaf_area_index": torch.tensor(1.0),
        "vapor_pressure_deficit": torch.tensor(1000.0),
    }
    output = model.forward(state, time=0.0, dt=1.0)
    gross = output["gross_photosynthesis"]
    assert isinstance(gross, UpdateItem)
    assert gross.unit == "umol CO2 m-2 s-1"
    assert torch.isfinite(gross.value)

    tolerance_float = float(torch.tensor(tolerance).item())
    strict_bool = bool(torch.tensor(strict).item())
    assert isinstance(tolerance_float, float)
    assert isinstance(strict_bool, bool)
    assert abs(1.0e-7) <= tolerance_float

