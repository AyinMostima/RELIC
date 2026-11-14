import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.emissions.sector_allocation import SectorAllocation


def _clone_state(state):
    cloned = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def test_sector_allocation_forward_does_not_mutate_input_state():
    model = SectorAllocation()
    grid = torch.arange(1, 17, dtype=torch.float64).reshape(1, 4, 4)
    state = {
        key: grid * (index * 1.0e-6)
        for index, key in enumerate(model.CANONICAL_INPUTS, start=1)
    }
    state.update(
        {
            "population_density": grid + 10.0,
            "gdp_density": torch.flip(grid, dims=(-1,)) + 20.0,
            "road_density": grid * 0.5 + 1.0,
            "land_use_fractions": torch.stack(
                (
                    torch.full_like(grid, 0.1),
                    torch.full_like(grid, 0.2),
                    torch.full_like(grid, 0.4),
                    torch.full_like(grid, 0.3),
                ),
                dim=1,
            ),
        }
    )
    state_before = _clone_state(state)

    result = model.forward(state=state, time=0.0, dt=3600.0)

    item = result["sector_emission_flux"]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.per_timestep is False
    assert item.value.shape == (5, 6, 1, 4, 4)
    inputs = torch.stack([state[key] for key in model.CANONICAL_INPUTS])
    assert torch.allclose(item.value.sum(dim=0), inputs, rtol=1.0e-12, atol=1.0e-18)
    assert set(state.keys()) == set(state_before.keys())
    for key in state_before:
        value_before = state_before[key]
        value_after = state[key]
        if isinstance(value_before, torch.Tensor):
            assert torch.equal(value_before, value_after)
        else:
            assert value_before == value_after
