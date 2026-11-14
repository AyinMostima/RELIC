import pytest
import torch

from relic.scientific.hydrosphere.runoff.subsurface import SubsurfaceRunoff


@pytest.mark.parametrize(
    "soil_moisture,groundwater_storage",
    [
        (torch.tensor(0.4, dtype=torch.float32), torch.tensor(120.0, dtype=torch.float32)),
        (0.4, 120.0),
    ],
)
def test_subsurface_runoff_forward_accepts_supported_state_types(
    soil_moisture, groundwater_storage
):
    model = SubsurfaceRunoff(name="subsurface_input_contract")
    fluxes = model.forward(
        {
            "soil_moisture": soil_moisture,
            "drainage_flux": torch.tensor(0.0, dtype=torch.float32),
            "groundwater_storage": groundwater_storage,
        },
        time=0.0,
        dt=3600.0,
    )
    assert "subsurface_runoff_flux" in fluxes
    value = fluxes["subsurface_runoff_flux"]
    value = value.value if hasattr(value, "value") else value
    assert torch.isfinite(value).all()


@pytest.mark.parametrize(
    "state",
    [
        {
            "soil_moisture": {"bad": "value"},
            "drainage_flux": 0.0,
            "groundwater_storage": 120.0,
        },
        {
            "soil_moisture": 0.4,
            "drainage_flux": 0.0,
            "groundwater_storage": {"bad": "value"},
        },
        {
            "soil_moisture": 0.4,
            "drainage_flux": {"bad": "value"},
            "groundwater_storage": 120.0,
        },
    ],
)
def test_subsurface_runoff_forward_rejects_unsupported_state_types(state):
    model = SubsurfaceRunoff(name="subsurface_input_contract_invalid")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        model.forward(state, time=0.0, dt=3600.0)
