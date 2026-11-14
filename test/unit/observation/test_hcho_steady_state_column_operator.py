"""L1 tests of the HCHO steady-state column relation owner (Marais et al., 2012).

Hand-computed cases use Marais et al. (2012) Eq. (4), p. 6226, and the Fig. 9
parameterization, p. 6229: S = m Omega_NO2 + c for Omega_NO2 <= 1e15 molecule
cm-2 and S = S_high above, with E_ISOP in atoms C cm-2 s-1 (Eq. 2, p. 6225).
"""

import pytest
import torch

from relic.core.state import StateWriteMode, UpdateMode
from relic.coupling.observation.hcho_steady_state_column import (
    HchoSteadyStateColumnOperator,
)
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant

PREFIX = "coupling.observation.hcho_steady_state_column"
MOLECULE_CM2 = "molecule cm-2"
MOL_M2 = "mol m-2"


def _no2(values):
    return torch.tensor(values, dtype=torch.float64)


def _molecule_cm2_to_mol_m2(values: torch.Tensor) -> torch.Tensor:
    avogadro = float(get_physical_constant("avogadro_constant"))
    return values * 1.0e4 / avogadro


def test_standard_set_hand_values_and_switch_at_1e15():
    operator = HchoSteadyStateColumnOperator()
    result = operator.emission_column_sensitivity(
        _no2([0.5e15, 1.0e15, 2.0e15]),
        no2_column_units=MOLECULE_CM2,
    )
    # 1340 * 0.5 + 523 = 1193 s; 1340 * 1.0 + 523 = 1863 s (boundary is
    # linear); 2e15 > threshold gives the constant 1870 s.
    torch.testing.assert_close(
        result.seconds,
        _no2([1193.0, 1863.0, 1870.0]),
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    assert result.linear_regime.tolist() == [True, True, False]
    assert result.below_calibrated_range.tolist() == [False, False, False]


def test_paulot_sensitivity_set_hand_values():
    operator = HchoSteadyStateColumnOperator(coefficient_set="marais2012_paulot")
    result = operator.emission_column_sensitivity(
        _no2([0.5e15, 1.0e15, 2.0e15]),
        no2_column_units=MOLECULE_CM2,
    )
    # 1600 * 0.5 + 721 = 1521 s; 1600 + 721 = 2321 s; constant 2230 s.
    torch.testing.assert_close(
        result.seconds,
        _no2([1521.0, 2321.0, 2230.0]),
        rtol=1.0e-12,
        atol=1.0e-9,
    )


def test_below_calibrated_range_is_flagged_and_never_clipped():
    operator = HchoSteadyStateColumnOperator()
    result = operator.emission_column_sensitivity(
        _no2([-0.1e15, 0.1e15, 0.2e15, 0.3e15]),
        no2_column_units=MOLECULE_CM2,
    )
    # Linear relation is still evaluated: 523 - 134 = 389 s and 523 + 134 = 657 s.
    torch.testing.assert_close(
        result.seconds,
        _no2([389.0, 657.0, 791.0, 925.0]),
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    # 0.2e15 is the lower end of the calibrated range and is not flagged.
    assert result.below_calibrated_range.tolist() == [True, True, False, False]
    assert result.linear_regime.tolist() == [True, True, True, True]


def test_mol_m2_input_matches_molecule_cm2_input():
    operator = HchoSteadyStateColumnOperator()
    molecule = _no2([0.15e15, 0.5e15, 1.0e15, 3.0e15])
    from_molecule = operator.emission_column_sensitivity(
        molecule,
        no2_column_units=MOLECULE_CM2,
    )
    from_mol = operator.emission_column_sensitivity(
        _molecule_cm2_to_mol_m2(molecule),
        no2_column_units=MOL_M2,
    )
    torch.testing.assert_close(from_mol.seconds, from_molecule.seconds)
    assert torch.equal(from_mol.linear_regime, from_molecule.linear_regime)
    assert torch.equal(
        from_mol.below_calibrated_range,
        from_molecule.below_calibrated_range,
    )


def test_steady_state_column_hand_value_uses_per_carbon_emission():
    operator = HchoSteadyStateColumnOperator()
    sensitivity = operator.emission_column_sensitivity(
        _no2([0.5e15]),
        no2_column_units=MOLECULE_CM2,
    )
    emission = torch.tensor([2.0e-9], dtype=torch.float64)
    background = torch.tensor([1.0e-5], dtype=torch.float64)
    column = operator.steady_state_column(
        emission,
        sensitivity.seconds,
        background,
    )
    # Omega = S n_C E + B = 1193 s * 5 * 2e-9 mol m-2 s-1 + 1e-5 mol m-2.
    torch.testing.assert_close(
        column,
        torch.tensor([1193.0 * 5.0 * 2.0e-9 + 1.0e-5], dtype=torch.float64),
        rtol=1.0e-12,
        atol=0.0,
    )


def test_carbon_atoms_come_from_the_chemistry_owner():
    operator = HchoSteadyStateColumnOperator()
    manager = get_global_manager()
    assert operator.isoprene_carbon_atoms == pytest.approx(
        float(manager.get("atmosphere.chemistry.isoprene_carbon_atoms"))
    )


@pytest.mark.parametrize(
    "coefficient_set,expected",
    [
        (
            "marais2012_standard",
            {
                "low_nox_slope": (1.34e-12, 1.2e-13),
                "low_nox_intercept": (523.0, 86.0),
                "high_nox_constant": (1870.0, 251.0),
            },
        ),
        (
            "marais2012_paulot",
            {
                "low_nox_slope": (1.6e-12, 1.6e-13),
                "low_nox_intercept": (721.0, 114.0),
                "high_nox_constant": (2230.0, 174.0),
            },
        ),
    ],
)
def test_registered_coefficients_units_and_references(coefficient_set, expected):
    manager = get_global_manager()
    units = {
        "low_nox_slope": "s cm2 molecule-1",
        "low_nox_intercept": "s",
        "high_nox_constant": "s",
    }
    for name, (value, sd) in expected.items():
        key = f"{PREFIX}.{coefficient_set}.{name}"
        sd_key = f"{key}_standard_deviation"
        spec = manager.get_parameter(key)
        sd_spec = manager.get_parameter(sd_key)
        assert float(spec.value) == pytest.approx(value, rel=1.0e-12)
        assert float(sd_spec.value) == pytest.approx(sd, rel=1.0e-12)
        assert spec.units == units[name]
        assert sd_spec.units == units[name]
        joined = " ".join(str(item) for item in spec.references)
        assert "10.5194/acp-12-6219-2012" in joined
        assert "6229" in joined and "Fig. 9" in joined
    operator = HchoSteadyStateColumnOperator(coefficient_set=coefficient_set)
    coefficients = operator.coefficients()
    deviations = operator.coefficient_standard_deviations()
    for name, (value, sd) in expected.items():
        assert coefficients[name] == pytest.approx(value, rel=1.0e-12)
        assert deviations[name] == pytest.approx(sd, rel=1.0e-12)


def test_registered_regime_threshold_and_calibrated_lower_end():
    manager = get_global_manager()
    threshold = manager.get_parameter(f"{PREFIX}.no2_regime_threshold")
    lower = manager.get_parameter(f"{PREFIX}.calibrated_no2_lower_end")
    assert float(threshold.value) == pytest.approx(1.0e15, rel=1.0e-12)
    assert float(lower.value) == pytest.approx(2.0e14, rel=1.0e-12)
    assert threshold.units == MOLECULE_CM2
    assert lower.units == MOLECULE_CM2


def test_gradients_follow_the_regimes():
    operator = HchoSteadyStateColumnOperator()
    no2 = _no2([0.5e15, 2.0e15]).requires_grad_(True)
    emission = torch.tensor([2.0e-9, 3.0e-9], dtype=torch.float64, requires_grad=True)
    background = torch.tensor([1.0e-5, 1.0e-5], dtype=torch.float64, requires_grad=True)
    sensitivity = operator.emission_column_sensitivity(
        no2,
        no2_column_units=MOLECULE_CM2,
    )
    column = operator.steady_state_column(emission, sensitivity.seconds, background)
    column.sum().backward()
    torch.testing.assert_close(
        emission.grad,
        sensitivity.seconds.detach() * 5.0,
    )
    torch.testing.assert_close(background.grad, torch.ones(2, dtype=torch.float64))
    # dOmega/dOmega_NO2 = n_C E m in the linear regime and 0 above the switch.
    torch.testing.assert_close(
        no2.grad,
        torch.tensor([5.0 * 2.0e-9 * 1.34e-12, 0.0], dtype=torch.float64),
        rtol=1.0e-10,
        atol=0.0,
    )


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError, match="coefficient_set"):
        HchoSteadyStateColumnOperator(coefficient_set="palmer2003")
    operator = HchoSteadyStateColumnOperator()
    with pytest.raises(ValueError, match="no2_column_units"):
        operator.emission_column_sensitivity(_no2([0.5e15]), no2_column_units="ppb")
    with pytest.raises(ValueError, match="finite"):
        operator.emission_column_sensitivity(
            _no2([float("nan")]),
            no2_column_units=MOLECULE_CM2,
        )
    seconds = torch.tensor([1193.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="non-negative"):
        operator.steady_state_column(
            torch.tensor([-1.0e-9], dtype=torch.float64),
            seconds,
            torch.tensor([0.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="finite"):
        operator.steady_state_column(
            torch.tensor([1.0e-9], dtype=torch.float64),
            seconds,
            torch.tensor([float("inf")], dtype=torch.float64),
        )


def test_forward_returns_derived_replace_diagnostics():
    operator = HchoSteadyStateColumnOperator()
    no2_molecule = _no2([0.1e15, 0.5e15, 2.0e15])
    state = {
        operator.ISOPRENE_EMISSION_INPUT: torch.full((3,), 2.0e-9, dtype=torch.float64),
        operator.NO2_COLUMN_INPUT: _molecule_cm2_to_mol_m2(no2_molecule),
        operator.BACKGROUND_COLUMN_INPUT: torch.full((3,), 1.0e-5, dtype=torch.float64),
    }
    outputs = operator.forward(state=state, time=0.0, dt=3600.0)
    assert set(outputs) == set(operator.metadata.output_variables)
    for item in outputs.values():
        assert item.mode == UpdateMode.REPLACE
        assert item.write_mode == StateWriteMode.DERIVED
        assert item.per_timestep is False
    seconds = torch.tensor([657.0, 1193.0, 1870.0], dtype=torch.float64)
    torch.testing.assert_close(
        outputs[operator.SENSITIVITY_OUTPUT].value,
        seconds,
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    torch.testing.assert_close(
        outputs[operator.COLUMN_OUTPUT].value,
        seconds * 5.0 * 2.0e-9 + 1.0e-5,
        rtol=1.0e-12,
        atol=0.0,
    )
    assert outputs[operator.COLUMN_OUTPUT].unit == MOL_M2
    assert outputs[operator.BELOW_CALIBRATED_OUTPUT].value.tolist() == [1.0, 0.0, 0.0]
    assert outputs[operator.LINEAR_REGIME_OUTPUT].value.tolist() == [1.0, 1.0, 0.0]
