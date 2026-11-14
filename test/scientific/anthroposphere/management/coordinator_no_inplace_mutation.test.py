import torch

from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator


def _clone_state(state):
    cloned = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def test_management_coordinator_forward_does_not_mutate_input_state():
    model = ManagementCoordinator(name="management_no_inplace")
    state = {
        "day_of_year": torch.tensor(120.0, dtype=torch.float32),
        "soil_moisture": torch.tensor([0.25, 0.24, 0.23], dtype=torch.float32),
        "soil_nitrogen": torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32),
        "soil_organic_carbon": torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32),
        "crop_maturity": torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32),
        "aboveground_biomass": torch.tensor([1.0, 0.9, 0.8], dtype=torch.float32),
        "air_temperature": torch.tensor([294.0, 293.5, 293.0], dtype=torch.float32),
        "irrigation_flux": torch.zeros(3, dtype=torch.float32),
        "harvest_flux": torch.zeros(3, dtype=torch.float32),
        "grazing_flux": torch.zeros(3, dtype=torch.float32),
        "co2_emission": torch.zeros(3, dtype=torch.float32),
        "field_capacity": torch.full((3,), 0.35, dtype=torch.float32),
        "wilting_point": torch.full((3,), 0.15, dtype=torch.float32),
        "evapotranspiration_flux": torch.full((3,), 3.0e-6, dtype=torch.float32),
        "precipitation_flux": torch.zeros(3, dtype=torch.float32),
    }
    state_before = _clone_state(state)

    result = model.forward(
        state=state,
        time=0.0,
        dt=3600.0,
        calendar_type="noleap",
    )

    assert isinstance(result, dict)
    assert set(state.keys()) == set(state_before.keys())
    for key in state_before:
        value_before = state_before[key]
        value_after = state[key]
        if isinstance(value_before, torch.Tensor):
            assert torch.equal(value_before, value_after)
        else:
            assert value_before == value_after
