"""L1 tests for variational assimilation (relic/optimization/assimilation/variational.py).

References:
    Talagrand and Courtier (1987), Q. J. R. Meteorol. Soc. 113, 1311-1328.
Mathematical expression:
    J(x) = 1/2 (x - x_b)^T B^-1 (x - x_b) + 1/2 (Hx - y)^T R^-1 (Hx - y),
    minimized by steepest descent x <- x - s grad J with the declared step s.

4D-Var is refused instead of returning a 3D-Var analysis, the configuration
offers only settings the minimizer reads, and an analysis failure keeps its
cause.
"""

from __future__ import annotations

import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.optimization.assimilation import (
    FourDVar,
    IdentityOperator,
    Observation,
    ObservationSet,
    ThreeDVar,
    VariationalConfig,
)
from relic.optimization.assimilation.base import AssimilationMethod
from relic.parameters import get_global_manager


def _runtime():
    state_manager = StateManager()
    simulator = RELICSimulator()
    simulator.set_state_manager(state_manager)
    return simulator, state_manager, get_global_manager()


def _observation(value: float) -> ObservationSet:
    observation = Observation(
        value=torch.tensor([value], dtype=torch.float32),
        error_variance=torch.tensor([1.0], dtype=torch.float32),
        coordinates=(0.0, 0.0, 0.0),
        time=0.0,
        variable_name="GPP",
        quality_flag=0,
    )
    return ObservationSet(observations=[observation], time=0.0)


def test_four_dimensional_var_is_not_replaced_by_three_dimensional_var() -> None:
    four_d = FourDVar(*_runtime(), VariationalConfig(method=AssimilationMethod.VARIATIONAL_4D, max_iterations=5))

    with pytest.raises(NotImplementedError, match="4D-Var"):
        four_d.assimilate(
            {"GPP": torch.tensor([10.0])}, _observation(12.0), IdentityOperator(["GPP"])
        )


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("line_search_method", "armijo"),
        ("observation_error_scale", 2.0),
        ("regularization_weight", 0.1),
        ("assimilation_window", 6.0),
        ("checkpoint_frequency", 5),
        ("use_preconditioning", True),
        ("preconditioner_type", "limited_memory"),
    ],
)
def test_settings_the_minimizer_never_reads_are_not_accepted(field_name, value) -> None:
    with pytest.raises(TypeError):
        VariationalConfig(method=AssimilationMethod.VARIATIONAL_3D, **{field_name: value})


def test_declared_gradient_step_size_drives_the_descent() -> None:
    """One steepest-descent step from x_b = 10 with B = R = 1 and y = 12.

    grad J(x_b) = (x_b - x_b) + (x_b - y) = -2, so x_1 = x_b + 2 s.
    """
    config = VariationalConfig(
        method=AssimilationMethod.VARIATIONAL_3D,
        max_iterations=1, convergence_tolerance=1.0e-12, gradient_step_size=0.25
    )
    result = ThreeDVar(*_runtime(), config).assimilate(
        {"GPP": torch.tensor([10.0])}, _observation(12.0), IdentityOperator(["GPP"])
    )

    assert result.analysis_mean["GPP"].item() == pytest.approx(10.5)


@pytest.mark.parametrize("step", [0.0, -0.1, float("nan")])
def test_invalid_gradient_step_size_is_rejected(step) -> None:
    with pytest.raises(ValueError, match="gradient_step_size"):
        VariationalConfig(method=AssimilationMethod.VARIATIONAL_3D, gradient_step_size=step)


def test_three_dimensional_var_failure_keeps_its_cause() -> None:
    class FailingOperator(IdentityOperator):
        def forward(self, state):
            raise ValueError("operator domain error")

    three_d = ThreeDVar(*_runtime(), VariationalConfig(method=AssimilationMethod.VARIATIONAL_3D, max_iterations=2))

    with pytest.raises(RuntimeError, match="operator domain error") as excinfo:
        three_d.assimilate(
            {"GPP": torch.tensor([10.0])}, _observation(12.0), FailingOperator(["GPP"])
        )
    assert isinstance(excinfo.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# Round 5: the caller chooses 3D-Var or 4D-Var, and the minimizer settings
# come from optimization.assimilation.variational.* (YAML), not code literals.
# ---------------------------------------------------------------------------


def test_the_variational_method_is_required_and_must_be_variational() -> None:
    with pytest.raises(TypeError):
        VariationalConfig()
    with pytest.raises(ValueError, match="VARIATIONAL"):
        VariationalConfig(method=AssimilationMethod.ENSEMBLE_KALMAN_FILTER)


def test_each_variational_assimilator_uses_its_own_method() -> None:
    assert ThreeDVar(*_runtime()).var_config.method is AssimilationMethod.VARIATIONAL_3D
    assert FourDVar(*_runtime()).var_config.method is AssimilationMethod.VARIATIONAL_4D
    with pytest.raises(ValueError, match="VARIATIONAL_3D"):
        ThreeDVar(*_runtime(), VariationalConfig(method=AssimilationMethod.VARIATIONAL_4D))
    with pytest.raises(ValueError, match="VARIATIONAL_4D"):
        FourDVar(*_runtime(), VariationalConfig(method=AssimilationMethod.VARIATIONAL_3D))


def test_minimizer_settings_are_read_from_the_parameter_database() -> None:
    expected = {
        "max_iterations": 100,
        "convergence_tolerance": 1.0e-4,
        "gradient_step_size": 0.01,
        "background_error_scale": 1.0,
    }
    manager = get_global_manager()
    keys = {name: f"optimization.assimilation.variational.{name}" for name in expected}
    originals = {key: manager.get(key, required=True) for key in keys.values()}
    config = VariationalConfig(method=AssimilationMethod.VARIATIONAL_3D)
    for name, value in expected.items():
        assert getattr(config, name) == pytest.approx(value)
    try:
        manager.set_value(keys["gradient_step_size"], 0.25)
        assert VariationalConfig(method=AssimilationMethod.VARIATIONAL_3D).gradient_step_size == 0.25
    finally:
        for key, value in originals.items():
            manager.set_value(key, value)
