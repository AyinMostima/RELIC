"""L1 contract tests for the canonical WaterStress diagnostic."""

import pytest
import torch

from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.biosphere.stress.water_stress import WaterStress


CANONICAL_INPUTS = {"soil_water_potential"}
CANONICAL_OUTPUTS = {"water_stress_factor"}


def _value(result: dict[str, UpdateItem]) -> torch.Tensor:
    return result["water_stress_factor"].value


def test_water_stress_uses_biological_bridge_and_exact_interface():
    model = WaterStress()

    assert isinstance(model, BiologicalProcessFunctor)
    assert isinstance(model, torch.nn.Module)
    assert set(model.metadata.input_variables) == CANONICAL_INPUTS
    assert set(model.metadata.output_variables) == CANONICAL_OUTPUTS
    assert model.get_dependencies() == []
    assert model.get_provides() == ["water_stress_factor"]


def test_water_stress_returns_one_explicit_replace_diagnostic():
    model = WaterStress()
    result = model(
        {"soil_water_potential": torch.tensor(-0.5)},
        time=0.0,
        dt=3600.0,
    )

    assert set(result) == CANONICAL_OUTPUTS
    update = result["water_stress_factor"]
    assert isinstance(update, UpdateItem)
    assert update.mode is UpdateMode.REPLACE
    assert update.per_timestep is False
    assert update.unit == "1"
    assert update.source_functor == "water_stress"

    expected = (-0.5 - (-1.5)) / (-0.033 - (-1.5))
    torch.testing.assert_close(update.value, torch.tensor(expected))


@pytest.mark.parametrize(
    ("soil_water_potential", "expected"),
    [
        (-2.0, 0.0),
        (-1.5, 0.0),
        (-0.7665, 0.5),
        (-0.033, 1.0),
        (-0.01, 1.0),
    ],
)
def test_water_stress_piecewise_linear_response(soil_water_potential, expected):
    model = WaterStress()
    result = model(
        {"soil_water_potential": torch.tensor(soil_water_potential)},
        time=0.0,
        dt=60.0,
    )

    torch.testing.assert_close(_value(result), torch.tensor(expected), atol=1.0e-6, rtol=0.0)


def test_water_stress_is_instantaneous_and_dt_invariant():
    model = WaterStress()
    state = {"soil_water_potential": torch.tensor([-1.2, -0.5, -0.02])}

    short_step = _value(model(state, time=0.0, dt=1.0))
    long_step = _value(model(state, time=0.0, dt=86400.0))

    torch.testing.assert_close(short_step, long_step)
    assert torch.all((short_step >= 0.0) & (short_step <= 1.0))
    assert torch.all(short_step[1:] >= short_step[:-1])


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"water_potential": torch.tensor(-0.5)},
        {"leaf_water_potential": torch.tensor(-0.5)},
    ],
)
def test_water_stress_rejects_missing_canonical_input_and_aliases(state):
    with pytest.raises(KeyError, match="soil_water_potential"):
        WaterStress().forward(state, time=0.0, dt=60.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_water_stress_rejects_nonfinite_input(bad_value):
    with pytest.raises(ValueError, match="finite"):
        WaterStress().forward(
            {"soil_water_potential": torch.tensor(bad_value)},
            time=0.0,
            dt=60.0,
        )


def test_water_stress_requires_tensor_state_input():
    with pytest.raises(TypeError, match="torch.Tensor"):
        WaterStress().forward(
            {"soil_water_potential": -0.5},
            time=0.0,
            dt=60.0,
        )


def test_water_stress_rejects_positive_matric_potential():
    with pytest.raises(ValueError, match="non-positive"):
        WaterStress().forward(
            {"soil_water_potential": torch.tensor(0.01)},
            time=0.0,
            dt=60.0,
        )


def test_water_stress_parameter_inventory_has_no_orphan_runtime_defaults():
    model = WaterStress()
    assert {
        "wilting_point_potential",
        "field_capacity_potential",
        "global.factor_min",
        "global.factor_max",
    }.issubset(model.metadata.parameters)
    assert {
        "osmotic_adjustment_capacity",
        "default_water_potential",
        "stomatal_limitation_coefficient",
        "stress_factor_min",
        "stress_factor_max",
    }.isdisjoint(model.metadata.parameters)

    manager = get_global_manager()
    for key in (
        "biosphere.stress.water.osmotic_adjustment_capacity",
        "biosphere.stress.water.default_water_potential",
        "biosphere.stress.water.stomatal_limitation_coefficient",
    ):
        with pytest.raises(KeyError):
            manager.get(key, required=True)
