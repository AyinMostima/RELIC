"""Canonical contract tests for EnvironmentalForcingFunctor."""

import warnings

import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.atmosphere.environmental_forcing import EnvironmentalForcingFunctor


def _canonical_state() -> dict[str, torch.Tensor]:
    """Return valid external forcing inputs for the VPD diagnostic."""
    return {
        "air_temperature": torch.tensor(298.15, dtype=torch.float32),
        "relative_humidity": torch.tensor(0.5, dtype=torch.float32),
    }


def test_environmental_forcing_declares_exact_canonical_interface():
    """The functor derives only VPD and never echoes external forcing."""
    functor = EnvironmentalForcingFunctor(device=torch.device("cpu"))

    assert functor.metadata.input_variables == [
        "air_temperature",
        "relative_humidity",
    ]
    assert functor.metadata.output_variables == ["vapor_pressure_deficit"]
    assert set(functor.metadata.input_variables).isdisjoint(functor.metadata.output_variables)
    assert not getattr(functor, "optional_input_variables", set())
    assert not hasattr(functor, "defaults")


def test_environmental_forcing_returns_vpd_pa_as_explicit_replace_update():
    """At 25 C and RH 0.5, VPD is approximately 1583.889 Pa."""
    functor = EnvironmentalForcingFunctor(device=torch.device("cpu"))

    result = functor.forward(state=_canonical_state(), time=0.0, dt=3600.0)

    assert set(result) == {"vapor_pressure_deficit"}
    update = result["vapor_pressure_deficit"]
    assert isinstance(update, UpdateItem)
    assert update.mode is UpdateMode.REPLACE
    assert update.per_timestep is True
    assert update.unit == "Pa"
    assert update.source_functor == functor.name
    assert update.value.item() == pytest.approx(1583.889, abs=1.0e-3)


def test_environmental_forcing_runtime_checkpoint_records_vpd_writer_and_device():
    """The canonical simulator must commit the 25 C, RH 0.5 VPD checkpoint."""
    dt = 60.0
    simulator = RELICSimulator(
        {
            "max_time": dt,
            "time_step": dt,
            "device": "cpu",
            "conservation_checks": True,
            "strict_dependencies": True,
        }
    )
    simulator.initialize_state(_canonical_state())
    functor = EnvironmentalForcingFunctor(device=torch.device("cpu"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulator.register_functor(functor, execution_group="fast")
        updates = simulator.step(dt=dt)

    actual = updates["vapor_pressure_deficit"]
    expected = torch.tensor(1583.889, dtype=actual.dtype, device=actual.device)
    absolute_error = abs(actual.item() - 1583.889)
    relative_error = absolute_error / 1583.889
    provenance = simulator.get_variable_provenance("vapor_pressure_deficit")

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-3)
    assert absolute_error == pytest.approx(3.8085937603682396e-05)
    assert relative_error == pytest.approx(2.4045837557860684e-08)
    assert {record["functor"] for record in provenance} == {"environmental_forcing"}
    assert {record["role"] for record in provenance} == {
        "writer",
        "final_writer",
    }
    assert {record["mode"] for record in provenance} == {"UpdateMode.REPLACE"}
    assert actual.device == simulator.get_effective_device("environmental_forcing")
    assert caught == []


def test_environmental_forcing_is_independent_of_dt():
    """An instantaneous forcing diagnostic must not scale with the time step."""
    functor = EnvironmentalForcingFunctor(device=torch.device("cpu"))

    first = functor.forward(_canonical_state(), time=0.0, dt=1.0)
    second = functor.forward(_canonical_state(), time=0.0, dt=86400.0)

    assert torch.equal(
        first["vapor_pressure_deficit"].value,
        second["vapor_pressure_deficit"].value,
    )


@pytest.mark.parametrize("missing_key", ["air_temperature", "relative_humidity"])
def test_environmental_forcing_requires_both_external_inputs(missing_key):
    """Missing forcing must fail instead of selecting a parameter default."""
    functor = EnvironmentalForcingFunctor(device=torch.device("cpu"))
    state = _canonical_state()
    state.pop(missing_key)

    with pytest.raises(KeyError, match=missing_key):
        functor.forward(state=state, time=0.0, dt=1.0)


@pytest.mark.parametrize("key", ["air_temperature", "relative_humidity"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
        torch.tensor(float("-inf")),
        torch.empty(0),
    ],
)
def test_environmental_forcing_rejects_nonfinite_or_empty_inputs(
    key,
    invalid_value,
):
    """Invalid external forcing must not be replaced or sanitized."""
    functor = EnvironmentalForcingFunctor(device=torch.device("cpu"))
    state = _canonical_state()
    state[key] = invalid_value

    with pytest.raises(ValueError, match=key):
        functor.forward(state=state, time=0.0, dt=1.0)


@pytest.mark.parametrize("invalid_rh", [-0.01, 1.01])
def test_environmental_forcing_rejects_relative_humidity_outside_unit_interval(
    invalid_rh,
):
    """Relative humidity is validated, not clamped into its physical range."""
    functor = EnvironmentalForcingFunctor(device=torch.device("cpu"))
    state = _canonical_state()
    state["relative_humidity"] = torch.tensor(invalid_rh)

    with pytest.raises(ValueError, match="relative_humidity"):
        functor.forward(state=state, time=0.0, dt=1.0)


@pytest.mark.parametrize("key", ["air_temperature", "relative_humidity"])
def test_environmental_forcing_requires_tensor_inputs(key):
    """The differentiable scientific interface accepts tensors only."""
    functor = EnvironmentalForcingFunctor(device=torch.device("cpu"))
    state = _canonical_state()
    state[key] = 298.15 if key == "air_temperature" else 0.5

    with pytest.raises(TypeError, match=key):
        functor.forward(state=state, time=0.0, dt=1.0)


def test_environmental_forcing_rejects_nonfinite_computed_vpd():
    """A finite input near the Magnus singularity must fail closed."""
    functor = EnvironmentalForcingFunctor(device=torch.device("cpu"))
    state = _canonical_state()
    state["air_temperature"] = torch.tensor(
        functor.celsius_to_kelvin - functor.vpd_b - 0.01,
        dtype=torch.float32,
    )
    assert torch.isfinite(state["air_temperature"])

    with pytest.raises(ValueError, match="vapor_pressure_deficit"):
        functor.forward(state=state, time=0.0, dt=1.0)
