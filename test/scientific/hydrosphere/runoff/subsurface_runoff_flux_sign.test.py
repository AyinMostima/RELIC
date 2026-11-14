import torch

from relic.scientific.hydrosphere.runoff.subsurface import SubsurfaceRunoff


def _scalar(value: torch.Tensor) -> float:
    if hasattr(value, "value"):
        value = value.value
    tensor = value.detach().to(dtype=torch.float32)
    if tensor.numel() == 1:
        return float(tensor.item())
    return float(tensor.mean().item())


def test_subsurface_runoff_flux_increases_with_groundwater_storage():
    low_model = SubsurfaceRunoff(name="subsurface_flux_low")
    high_model = SubsurfaceRunoff(name="subsurface_flux_high")
    low = low_model.forward(
        {
            "soil_moisture": torch.tensor(0.4),
            "drainage_flux": torch.tensor(0.0),
            "groundwater_storage": torch.tensor(20.0),
        },
        time=0.0,
        dt=3600.0,
    )["subsurface_runoff_flux"]
    high = high_model.forward(
        {
            "soil_moisture": torch.tensor(0.4),
            "drainage_flux": torch.tensor(0.0),
            "groundwater_storage": torch.tensor(120.0),
        },
        time=0.0,
        dt=3600.0,
    )["subsurface_runoff_flux"]

    assert _scalar(high) >= _scalar(low)
