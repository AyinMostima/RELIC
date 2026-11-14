import torch

from relic.scientific.anthroposphere.land_use.transitions import LandUseTransitions


def _clone_state(state):
    cloned = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def test_land_use_transitions_forward_does_not_mutate_input_state():
    model = LandUseTransitions(name="land_use_no_inplace")
    state = {
        "land_use_fraction": torch.tensor(
            [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.04, 0.03],
            dtype=torch.float32,
        ),
        "time_index": torch.tensor(0, dtype=torch.int64),
    }
    state_before = _clone_state(state)
    transition_matrix_before = model.transition_matrix.clone()

    result = model.forward(state=state, time=0.0, dt=3600.0)

    assert isinstance(result, dict)
    assert set(state.keys()) == set(state_before.keys())
    for key in state_before:
        value_before = state_before[key]
        value_after = state[key]
        if isinstance(value_before, torch.Tensor):
            assert torch.equal(value_before, value_after)
        else:
            assert value_before == value_after
    assert torch.equal(model.transition_matrix, transition_matrix_before)
    assert not hasattr(model, "current_fractions")
