from copy import deepcopy

import pytest
import torch

from relic.core.state import StateWriteMode, UpdateItem, UpdateMode
from relic.coupling.observation.tower_concentration import (
    TowerConcentrationProfileOperator,
)
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.utils.physical_constants import get_physical_constant
from relic.validation.framework.rules.observation_diagnostic import (
    validate_observation_diagnostic_payload,
)


def _point_state(
    concentration: torch.Tensor | None = None,
    *,
    pressure_key: str = "atmospheric_pressure",
) -> dict[str, torch.Tensor]:
    dtype = torch.float64
    profile = (
        concentration
        if concentration is not None
        else torch.tensor([1.0e-9, 3.0e-9, 5.0e-9, 7.0e-9], dtype=dtype)
    )
    return {
        "tracer_concentration": profile,
        "air_temperature": torch.full((4,), 300.0, dtype=dtype),
        pressure_key: torch.full((4,), 100000.0, dtype=dtype),
        "layer_center_height": torch.tensor(
            [0.0, 20.0, 40.0, 60.0],
            dtype=dtype,
        ),
    }


def _expected_ppbv(mass_concentration: torch.Tensor) -> torch.Tensor:
    gas_constant = float(get_physical_constant("gas_constant"))
    molar_mass = float(
        get_physical_constant(
            "relic.physical.constants.molecular_weights.isoprene"
        )
    ) * 1.0e-3
    return (
        mass_concentration
        * gas_constant
        * 300.0
        / (molar_mass * 100000.0)
        * 1.0e9
    )


def test_point_linear_three_height_ppbv_is_derived_and_differentiable():
    concentration = torch.tensor(
        [1.0e-9, 3.0e-9, 5.0e-9, 7.0e-9],
        dtype=torch.float64,
        requires_grad=True,
    )
    operator = TowerConcentrationProfileOperator()

    outputs = operator.forward(
        state=_point_state(concentration),
        time=0.0,
        dt=1800.0,
        subprocess_outputs={
            "fickian_diffusion": {
                "tracer_concentration": UpdateItem(
                    value=torch.full_like(concentration, 99.0),
                    mode=UpdateMode.REPLACE,
                )
            }
        },
    )

    expected_mass = torch.tensor(
        [2.0e-9, 4.0e-9, 6.0e-9],
        dtype=torch.float64,
    )
    raw = outputs[operator.RAW_PPBV]
    torch.testing.assert_close(raw.value, _expected_ppbv(expected_mass))
    assert raw.unit == "ppbv"
    assert raw.write_mode == StateWriteMode.DERIVED
    torch.testing.assert_close(
        outputs[operator.SUPPORT_WEIGHT_SUM].value,
        torch.ones(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        outputs[operator.WINDOW_COVERAGE].value,
        torch.ones(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        outputs[operator.WINDOW_COMPLETE].value,
        torch.ones(3, dtype=torch.float64),
    )
    assert raw.value[0] < raw.value[1] < raw.value[2]

    raw.value.sum().backward()
    assert concentration.grad is not None
    assert bool(torch.all(concentration.grad > 0))


def test_point_linear_reconstructs_primitives_before_local_conversion():
    concentration = torch.tensor(
        [1.0e-9, 2.0e-9, 5.0e-9, 9.0e-9],
        dtype=torch.float64,
        requires_grad=True,
    )
    temperature = torch.tensor(
        [280.0, 292.0, 306.0, 320.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    pressure = torch.tensor(
        [101000.0, 99000.0, 96000.0, 92000.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    operator = TowerConcentrationProfileOperator()
    outputs = operator.forward(
        state={
            "tracer_concentration": concentration,
            "air_temperature": temperature,
            "atmospheric_pressure": pressure,
            "layer_center_height": torch.tensor(
                [0.0, 20.0, 40.0, 60.0],
                dtype=torch.float64,
            ),
        },
        time=0.0,
        dt=1800.0,
    )

    weights = torch.tensor(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ],
        dtype=torch.float64,
    )
    gas_constant = float(get_physical_constant("gas_constant"))
    molar_mass = (
        float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.isoprene"
            )
        )
        * 1.0e-3
    )
    # References: Seinfeld and Pandis (2016); CF Conventions 1.8.
    # Expression: chi(z_H)=1e9*(W*rho)*R*(W*T)/(M*(W*p)).
    # Parameters: W reconstructs the point [1], rho is [kg m-3],
    # T is [K], p is [Pa], R is [J mol-1 K-1], and M is [kg mol-1].
    expected = (
        1.0e9
        * (weights @ concentration)
        * gas_constant
        * (weights @ temperature)
        / (molar_mass * (weights @ pressure))
    )
    converted_then_interpolated = weights @ (
        1.0e9
        * concentration
        * gas_constant
        * temperature
        / (molar_mass * pressure)
    )

    assert not torch.allclose(
        expected,
        converted_then_interpolated,
        rtol=1.0e-10,
        atol=0.0,
    )
    torch.testing.assert_close(
        outputs[operator.RAW_PPBV].value,
        expected,
        rtol=1.0e-12,
        atol=0.0,
    )
    outputs[operator.RAW_PPBV].value.sum().backward()
    assert concentration.grad is not None
    assert temperature.grad is not None
    assert pressure.grad is not None
    assert bool(torch.all(concentration.grad > 0))
    assert bool(torch.all(temperature.grad > 0))
    assert bool(torch.all(pressure.grad < 0))


def test_singleton_cell_column_broadcasts_surface_temperature_and_pressure():
    """A single horizontal cell may carry one explicit vertical tracer axis."""
    concentration = torch.tensor(
        [[1.0e-9, 3.0e-9, 5.0e-9, 7.0e-9]],
        dtype=torch.float64,
    )
    state = _point_state(concentration)
    state["air_temperature"] = torch.tensor([300.0], dtype=torch.float64)
    state["atmospheric_pressure"] = torch.tensor(
        [100000.0],
        dtype=torch.float64,
    )
    scalar_operator = TowerConcentrationProfileOperator()

    scalar_outputs = scalar_operator.forward(
        state=state,
        time=0.0,
        dt=1800.0,
    )
    profile_operator = TowerConcentrationProfileOperator()
    profile_outputs = profile_operator.forward(
        state=_point_state(concentration),
        time=0.0,
        dt=1800.0,
    )

    expected_mass = torch.tensor(
        [2.0e-9, 4.0e-9, 6.0e-9],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        scalar_outputs[scalar_operator.RAW_PPBV].value,
        _expected_ppbv(expected_mass),
    )
    torch.testing.assert_close(
        scalar_outputs[scalar_operator.RAW_PPBV].value,
        profile_outputs[profile_operator.RAW_PPBV].value,
        rtol=0.0,
        atol=0.0,
    )


def test_multiple_horizontal_cells_fail_closed():
    """The bounded ACP operator must not hide a horizontal aggregation."""
    state = _point_state(
        torch.full((2, 4), 1.0e-9, dtype=torch.float64)
    )
    operator = TowerConcentrationProfileOperator()

    with pytest.raises(ValueError, match="exactly one horizontal cell"):
        operator.forward(state=state, time=0.0, dt=1800.0)


def test_finite_layer_overlap_uses_independent_normalized_rows():
    operator = TowerConcentrationProfileOperator(
        support_mode="finite_layer_overlap"
    )
    state = {
        "tracer_concentration": torch.tensor(
            [1.0e-9, 3.0e-9, 5.0e-9, 7.0e-9],
            dtype=torch.float64,
        ),
        "air_temperature": torch.full((4,), 300.0, dtype=torch.float64),
        "atmospheric_pressure": torch.full(
            (4,),
            100000.0,
            dtype=torch.float64,
        ),
        "layer_interface_height": torch.tensor(
            [0.0, 20.0, 40.0, 60.0, 80.0],
            dtype=torch.float64,
        ),
    }

    outputs = operator.forward(
        state=state,
        time=0.0,
        dt=1800.0,
    )

    expected_mass = torch.tensor(
        [1.0e-9, 3.0e-9, 5.0e-9],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        outputs[operator.RAW_PPBV].value,
        _expected_ppbv(expected_mass),
    )
    torch.testing.assert_close(
        outputs[operator.SUPPORT_WEIGHT_SUM].value,
        torch.ones(3, dtype=torch.float64),
    )


def test_finite_overlap_averages_local_mole_fraction_not_mean_primitives():
    """Finite support must average the converted local mole fraction."""

    concentration = torch.tensor(
        [1.0e-9, 2.0e-9, 5.0e-9, 9.0e-9],
        dtype=torch.float64,
    )
    temperature = torch.tensor(
        [280.0, 292.0, 306.0, 320.0],
        dtype=torch.float64,
    )
    pressure = torch.tensor(
        [101000.0, 99000.0, 96000.0, 92000.0],
        dtype=torch.float64,
    )
    interfaces = torch.tensor(
        [0.0, 12.0, 28.0, 48.0, 80.0],
        dtype=torch.float64,
    )
    operator = TowerConcentrationProfileOperator(
        support_mode="finite_layer_overlap"
    )

    outputs = operator.forward(
        state={
            "tracer_concentration": concentration,
            "air_temperature": temperature,
            "atmospheric_pressure": pressure,
            "layer_interface_height": interfaces,
        },
        time=0.0,
        dt=1800.0,
    )

    weights = torch.tensor(
        [
            [0.7, 0.3, 0.0, 0.0],
            [0.0, 0.3, 0.7, 0.0],
            [0.0, 0.0, 0.3, 0.7],
        ],
        dtype=torch.float64,
    )
    gas_constant = float(get_physical_constant("gas_constant"))
    molar_mass = (
        float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.isoprene"
            )
        )
        * 1.0e-3
    )
    # References: Seinfeld and Pandis (2016); CF Conventions 1.8.
    # Expression: chi_H = W[1e9 * rho * R * T / (M * p)].
    # Parameters: W is interval support [1], rho is mass concentration
    # [kg m-3], T is temperature [K], p is pressure [Pa], and M is
    # isoprene molar mass [kg mol-1].
    layer_ppbv = (
        1.0e9
        * concentration
        * gas_constant
        * temperature
        / (molar_mass * pressure)
    )
    expected = weights @ layer_ppbv
    mean_primitives = (
        1.0e9
        * (weights @ concentration)
        * gas_constant
        * (weights @ temperature)
        / (molar_mass * (weights @ pressure))
    )

    assert not torch.allclose(
        expected,
        mean_primitives,
        rtol=1.0e-10,
        atol=0.0,
    )
    torch.testing.assert_close(
        outputs[operator.RAW_PPBV].value,
        expected,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_dry_basis_requires_explicit_dry_air_pressure():
    operator = TowerConcentrationProfileOperator(air_basis="dry")
    wet_state = _point_state()

    with pytest.raises(KeyError, match="dry_air_pressure"):
        operator.forward(state=wet_state, time=0.0, dt=1800.0)

    dry_outputs = operator.forward(
        state=_point_state(pressure_key="dry_air_pressure"),
        time=0.0,
        dt=1800.0,
    )
    assert operator.observation_operator().metadata["air_basis"] == "dry"
    assert bool(torch.all(torch.isfinite(dry_outputs[operator.RAW_PPBV].value)))


def test_duration_weighted_window_uses_true_step_overlap():
    operator = TowerConcentrationProfileOperator()
    first = torch.full((4,), 1.0e-9, dtype=torch.float64)
    second = torch.full((4,), 3.0e-9, dtype=torch.float64)

    partial = operator.forward(
        state=_point_state(first),
        time=0.0,
        dt=600.0,
    )
    completed = operator.forward(
        state=_point_state(second),
        time=600.0,
        dt=1200.0,
    )

    torch.testing.assert_close(
        partial[operator.WINDOW_COVERAGE].value,
        torch.full((3,), 1.0 / 3.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        partial[operator.WINDOW_COMPLETE].value,
        torch.zeros(3, dtype=torch.float64),
    )
    expected_mass = torch.full(
        (3,),
        (1.0e-9 * 600.0 + 3.0e-9 * 1200.0) / 1800.0,
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        completed[operator.RAW_PPBV].value,
        _expected_ppbv(expected_mass),
    )
    torch.testing.assert_close(
        completed[operator.WINDOW_COVERAGE].value,
        torch.ones(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        completed[operator.WINDOW_COMPLETE].value,
        torch.ones(3, dtype=torch.float64),
    )


def test_window_buffers_roundtrip_through_standard_state_dict():
    uninterrupted = TowerConcentrationProfileOperator()
    restarted = TowerConcentrationProfileOperator()
    first = torch.full((4,), 1.0e-9, dtype=torch.float64)
    second = torch.full((4,), 3.0e-9, dtype=torch.float64)

    uninterrupted.forward(
        state=_point_state(first),
        time=0.0,
        dt=600.0,
    )
    restarted.load_state_dict(deepcopy(uninterrupted.state_dict()))

    expected = uninterrupted.forward(
        state=_point_state(second),
        time=600.0,
        dt=1200.0,
    )
    actual = restarted.forward(
        state=_point_state(second),
        time=600.0,
        dt=1200.0,
    )

    for output_name in uninterrupted.metadata.output_variables:
        torch.testing.assert_close(
            actual[output_name].value,
            expected[output_name].value,
        )
    for key, expected_value in uninterrupted.state_dict().items():
        torch.testing.assert_close(restarted.state_dict()[key], expected_value)


def test_point_support_outside_column_fails_closed():
    operator = TowerConcentrationProfileOperator()
    with torch.no_grad():
        operator.measurement_heights[-1] = 100.0

    with pytest.raises(ValueError, match="outside the modeled column"):
        operator.forward(
            state=_point_state(),
            time=0.0,
            dt=1800.0,
        )


def test_observation_and_diagnostic_contracts_are_read_only():
    operator = TowerConcentrationProfileOperator()
    observation = operator.observation_operator()
    diagnostic = operator.diagnostic_operator()

    assert observation.output_name == operator.RAW_PPBV
    assert observation.output_units == "ppbv"
    assert observation.support == "three_height_vertical_profile"
    assert observation.averaging_window == "duration_weighted_window"
    assert observation.metadata["air_basis"] == "wet"
    assert observation.metadata["qc_applied"] is False
    assert "atmospheric_pressure" in observation.input_state_vars
    assert "air_pressure" not in observation.input_state_vars
    assert diagnostic.writes_state is False
    assert validate_observation_diagnostic_payload(
        [observation],
        [diagnostic],
    ).passed


def test_all_concentration_outputs_have_canonical_variable_contracts():
    operator = TowerConcentrationProfileOperator()
    catalog = SSOTCatalog.from_parameter_manager()

    for output_name in operator.metadata.output_variables:
        entry = catalog.get(output_name)
        assert entry is not None, output_name
        assert entry.write_mode == "derived"
        assert entry.shape == {"dimensions": ["height"]}
