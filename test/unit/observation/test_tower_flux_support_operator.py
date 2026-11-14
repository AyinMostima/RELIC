import pytest
import torch

from relic.core.state import StateWriteMode
from relic.coupling.observation.tower_flux import TowerFluxSupportOperator
from relic.validation.framework.rules.observation_diagnostic import (
    validate_observation_diagnostic_payload,
)


def _valid_state(dtype=torch.float64):
    return {
        "footprint_weights": torch.tensor([0.25, 0.75], dtype=dtype),
        "footprint_kernel_mass": torch.tensor(1.0, dtype=dtype),
        "footprint_cell_area": torch.tensor([100.0, 200.0], dtype=dtype),
        "footprint_data_coverage_fraction": torch.tensor(1.0, dtype=dtype),
        "measurement_height": torch.tensor(30.0, dtype=dtype),
        "time_bounds": torch.tensor([0.0, 1800.0], dtype=dtype),
        "footprint_density_available": torch.tensor(1.0, dtype=dtype),
        "flux_footprint_valid": torch.tensor(1.0, dtype=dtype),
    }


@pytest.fixture
def operator(monkeypatch):
    instance = TowerFluxSupportOperator()
    requested = []

    def get_parameter_value(name, *, required=False, **kwargs):
        requested.append((name, required))
        if name == (
            "coupling.observation.tower_flux.support_closure_tolerance"
        ):
            return 1.0e-6
        raise KeyError(name)

    monkeypatch.setattr(instance, "get_parameter_value", get_parameter_value)
    instance._requested_parameter_keys = requested
    return instance


def _run(operator, flux=None, state=None):
    weighted_flux = (
        flux
        if flux is not None
        else torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    )
    return operator.forward(
        state=_valid_state() if state is None else state,
        time=0.0,
        dt=torch.tensor(1800.0),
        subprocess_outputs={
            "support_adapter": {
                operator.FOOTPRINT_FLUX: weighted_flux,
            }
        },
    )


def test_uniform_flux_identity_preserves_units_diagnostics_and_gradient(operator):
    weighted_flux = torch.tensor(
        2.5,
        dtype=torch.float64,
        requires_grad=True,
    )

    outputs = _run(operator, flux=weighted_flux)

    tower_flux = outputs[operator.TOWER_FLUX]
    torch.testing.assert_close(
        tower_flux.value,
        weighted_flux,
        rtol=0.0,
        atol=0.0,
    )
    assert tower_flux.unit == "mol m-2 s-1"
    assert tower_flux.write_mode == StateWriteMode.DERIVED
    torch.testing.assert_close(
        outputs[operator.SUPPORT_INTEGRAL].value,
        torch.tensor(1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        outputs[operator.SUPPORT_AREA].value,
        torch.tensor(300.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        outputs[operator.WEIGHT_SUM].value,
        torch.tensor(1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        outputs[operator.COVERAGE_FRACTION].value,
        torch.tensor(1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        outputs[operator.EXTRAPOLATION_FLAG].value,
        torch.tensor(0.0, dtype=torch.float64),
    )
    assert all(
        update.write_mode == StateWriteMode.DERIVED
        for update in outputs.values()
    )
    tower_flux.value.backward()
    torch.testing.assert_close(weighted_flux.grad, torch.tensor(1.0))
    assert operator._requested_parameter_keys == [
        (
            "coupling.observation.tower_flux.support_closure_tolerance",
            True,
        )
    ]


def test_dependency_supplied_flux_passes_full_functor_lifecycle(operator):
    weighted_flux = torch.tensor(2.5, dtype=torch.float64)

    outputs = operator(
        _valid_state(),
        time=0.0,
        dt=torch.tensor(1800.0),
        subprocess_outputs={
            "support_adapter": {
                operator.FOOTPRINT_FLUX: weighted_flux,
            }
        },
    )

    torch.testing.assert_close(
        outputs[operator.TOWER_FLUX].value,
        weighted_flux,
        rtol=0.0,
        atol=0.0,
    )


def test_observation_and_diagnostic_specs_freeze_identity_scope(operator):
    observation = operator.observation_operator()
    diagnostic = operator.diagnostic_operator()

    assert observation.input_state_vars == tuple(
        operator.metadata.input_variables
    )
    assert observation.output_name == operator.TOWER_FLUX
    assert observation.output_units == "mol m-2 s-1"
    assert observation.support == "flux_footprint"
    assert observation.averaging_window == "interval_mean"
    assert observation.conservative_projection_required is False
    assert observation.metadata["height_transfer_factor_applied"] is False
    assert observation.metadata["qc_applied"] is False
    assert diagnostic.writes_state is False
    assert tuple(diagnostic.metadata["outputs"]) == tuple(
        operator.metadata.output_variables
    )
    assert validate_observation_diagnostic_payload(
        [observation],
        [diagnostic],
    ).passed


@pytest.mark.parametrize(
    "missing_field",
    [
        "output_name",
        "support",
        "footprint_description",
        "error_model",
    ],
)
def test_observation_validator_rejects_incomplete_operator_contract(
    operator,
    missing_field,
):
    observation = operator.observation_operator()
    payload = {
        "name": observation.name,
        "input_state_vars": observation.input_state_vars,
        "output_name": observation.output_name,
        "output_units": observation.output_units,
        "support": observation.support,
        "averaging_window": observation.averaging_window,
        "footprint_description": observation.footprint_description,
        "error_model": observation.error_model,
        "conservative_projection_required": False,
        "metadata": observation.metadata,
    }
    payload[missing_field] = ""

    result = validate_observation_diagnostic_payload([payload], [])

    assert result.passed is False
    assert missing_field in result.message


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("footprint_weights", [0.25, -0.25], "non-negative"),
        ("footprint_weights", [0.25, float("nan")], "finite"),
        ("footprint_weights", [0.25, 0.70], "sum to one"),
        ("footprint_kernel_mass", 0.90, "kernel mass"),
        (
            "footprint_data_coverage_fraction",
            0.90,
            "data coverage",
        ),
        ("measurement_height", 0.0, "measurement height"),
        ("footprint_density_available", 0.0, "density"),
        ("flux_footprint_valid", 0.0, "validity"),
    ],
)
def test_identity_fails_closed_when_footprint_gate_is_invalid(
    operator,
    field,
    value,
    match,
):
    state = _valid_state()
    state[field] = torch.as_tensor(value, dtype=torch.float64)

    with pytest.raises(ValueError, match=match):
        _run(operator, state=state)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("footprint_cell_area", [100.0], "cell-area shape"),
        ("footprint_cell_area", [100.0, 0.0], "cell areas"),
        ("time_bounds", [0.0], "two time bounds"),
        ("time_bounds", [1800.0, 0.0], "strictly increasing"),
    ],
)
def test_identity_rejects_invalid_geometry_or_time_support(
    operator,
    field,
    value,
    match,
):
    state = _valid_state()
    state[field] = torch.as_tensor(value, dtype=torch.float64)

    with pytest.raises(ValueError, match=match):
        _run(operator, state=state)


def test_identity_rejects_ambiguous_link_and_state_input(operator):
    state = _valid_state()
    state[operator.FOOTPRINT_FLUX] = torch.tensor(2.5)

    with pytest.raises(RuntimeError, match="ambiguous"):
        _run(operator, state=state)


def test_identity_rejects_missing_upstream_flux(operator):
    with pytest.raises(KeyError, match="footprint-weighted flux"):
        operator.forward(
            state=_valid_state(),
            time=0.0,
            dt=torch.tensor(1800.0),
            subprocess_outputs={},
        )


def test_operator_loads_closure_tolerance_from_parameter_manager():
    operator = TowerFluxSupportOperator()

    outputs = _run(operator)

    torch.testing.assert_close(
        outputs[operator.TOWER_FLUX].value,
        torch.tensor(2.5, dtype=torch.float64),
    )
