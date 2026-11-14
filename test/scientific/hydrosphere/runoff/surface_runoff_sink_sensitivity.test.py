import torch

from relic.scientific.hydrosphere.runoff.surface import SurfaceRunoff


def _scalar(value: torch.Tensor) -> float:
    if hasattr(value, "value"):
        value = value.value
    tensor = value.detach().to(dtype=torch.float32)
    if tensor.numel() == 1:
        return float(tensor.item())
    return float(tensor.mean().item())


def test_surface_runoff_flux_is_monotonic_with_precipitation():
    low_model = SurfaceRunoff(name="surface_precipitation_low")
    high_model = SurfaceRunoff(name="surface_precipitation_high")
    base_state = {"soil_moisture": torch.tensor(0.40)}
    low = low_model.forward(
        {**base_state, "precipitation_flux": torch.tensor(1.0e-4)},
        time=0.0,
        dt=3600.0,
    )["surface_runoff_flux"]
    high = high_model.forward(
        {**base_state, "precipitation_flux": torch.tensor(5.0e-3)},
        time=0.0,
        dt=3600.0,
    )["surface_runoff_flux"]

    assert _scalar(high) >= _scalar(low)
