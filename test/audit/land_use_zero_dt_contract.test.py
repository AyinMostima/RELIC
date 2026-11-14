import inspect

import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.land_use.transitions import LandUseTransitions


def _unwrap(value):
    if hasattr(value, "value"):
        return value.value
    return value


def _build_state() -> dict:
    fractions = torch.tensor(
        [0.25, 0.20, 0.12, 0.08, 0.10, 0.10, 0.05, 0.05, 0.05],
        dtype=torch.float32,
    )
    return {
        "land_use_fraction": fractions,
        "time_index": torch.tensor(0, dtype=torch.int64),
        "year": torch.tensor(2026.0, dtype=torch.float32),
    }


def test_land_use_zero_dt_preserves_vector_contract() -> None:
    model = LandUseTransitions(name="land_use_zero_dt_contract")
    state = _build_state()
    base = state["land_use_fraction"].clone()

    outputs = model.forward(state=state, time=0.0, dt=0.0)

    vector_update = _unwrap(outputs["land_use_fraction"])
    delta = _unwrap(outputs["diagnostics.land_use_transitions.delta_fraction"])
    assert tuple(vector_update.shape) == tuple(base.shape)
    assert torch.allclose(vector_update.sum(), torch.tensor(1.0, dtype=vector_update.dtype), atol=1e-6)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)
    assert torch.equal(vector_update, base)
    for output in outputs.values():
        assert isinstance(output, UpdateItem)
        assert output.mode is UpdateMode.REPLACE
        assert output.per_timestep is False
        assert output.source_functor == model.name


def test_land_use_forward_signature_is_forward_only() -> None:
    sig = inspect.signature(LandUseTransitions.forward)
    params = list(sig.parameters.values())
    assert all(p.kind != inspect.Parameter.VAR_POSITIONAL for p in params)
    assert [p.name for p in params[:4]] == ["self", "state", "time", "dt"]


def test_land_use_output_keys_are_declared_in_metadata() -> None:
    model = LandUseTransitions(name="land_use_output_contract")
    state = _build_state()
    outputs = model.forward(
        state=state,
        time=0.0,
        dt=3600.0,
    )
    declared = set(model.metadata.output_variables)
    actual = set(outputs.keys())
    assert actual == declared
