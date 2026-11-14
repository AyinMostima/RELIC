"""Tests for derivative-free calibration: parallel safety and error propagation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import relic.parameters.calibration.calibrator as calibrator_module
from relic.parameters.calibration.calibrator import (
    CalibrationConfig,
    CalibrationMethod,
    MSEObjective,
    NashSutcliffeObjective,
    ObjectiveFunction,
    ParameterBounds,
    ParameterCalibrator,
)


class _QuadraticObjective(ObjectiveFunction):
    def __call__(self, parameters, observations) -> float:
        return float((parameters["test_param"] - float(observations.item())) ** 2)


def _config(**overrides) -> CalibrationConfig:
    values = {
        "method": CalibrationMethod.DIFFERENTIAL_EVOLUTION,
        "target_parameters": ["test_param"],
        "parameter_bounds": {
            "test_param": ParameterBounds(lower=0.0, upper=1.0, scale="linear"),
        },
        "max_iterations": 1,
        "population_size": 2,
        "verbose": False,
    }
    values.update(overrides)
    return CalibrationConfig(**values)


def test_calibration_parallel_is_opt_in() -> None:
    assert _config().parallel is False


@pytest.mark.parametrize(
    "config",
    [
        _config(parallel=True),
        _config(parallel=False, method_kwargs={"workers": 2}),
        _config(parallel=False, method_kwargs={"workers": lambda func, items: []}),
    ],
)
def test_differential_evolution_rejects_unisolated_parallel_objectives(
    config: CalibrationConfig,
) -> None:
    calibrator = ParameterCalibrator(parameter_manager=Mock(), hooks=Mock())

    with pytest.raises(RuntimeError, match="isolated objective factory"):
        calibrator._differential_evolution(
            config=config,
            objective_function=_QuadraticObjective(),
            observations=torch.tensor(0.5),
            initial_parameters={"test_param": 0.5},
        )


def test_sequential_differential_evolution_does_not_create_worker_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_differential_evolution(func, bounds, **kwargs):
        captured.update(kwargs)
        assert float(func(np.array([0.5]))) == 0.0
        return SimpleNamespace(
            x=np.array([0.5]),
            nit=1,
            success=True,
            message="ok",
        )

    monkeypatch.setattr(
        calibrator_module,
        "differential_evolution",
        fake_differential_evolution,
    )
    calibrator = ParameterCalibrator(parameter_manager=Mock(), hooks=Mock())

    result = calibrator._differential_evolution(
        config=_config(parallel=False),
        objective_function=_QuadraticObjective(),
        observations=torch.tensor(0.5),
        initial_parameters={"test_param": 0.5},
    )

    assert "workers" not in captured
    assert result.success is True
    assert result.convergence_history == [0.0]


def _diverging_model(parameters):
    raise RuntimeError("model diverged")


@pytest.mark.parametrize("objective_class", [MSEObjective, NashSutcliffeObjective])
def test_objective_propagates_a_model_failure_instead_of_returning_inf(objective_class) -> None:
    objective = objective_class(_diverging_model, variable_names=["gpp"])

    with pytest.raises(RuntimeError, match="model diverged"):
        objective({"test_param": 0.5}, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("objective_class", [MSEObjective, NashSutcliffeObjective])
def test_objective_raises_for_a_missing_output_variable(objective_class) -> None:
    objective = objective_class(
        lambda parameters: {"other": torch.tensor([1.0, 2.0])},
        variable_names=["gpp"],
    )

    with pytest.raises(KeyError, match="gpp"):
        objective({"test_param": 0.5}, torch.tensor([1.0, 2.0]))


def test_nash_sutcliffe_objective_rejects_constant_observations() -> None:
    """NSE is undefined when the observations have zero variance."""
    objective = NashSutcliffeObjective(
        lambda parameters: torch.tensor([1.0, 2.0]),
        variable_names=["gpp"],
    )

    with pytest.raises(ValueError, match="variance"):
        objective({"test_param": 0.5}, torch.tensor([3.0, 3.0]))


def test_objective_values_for_valid_output() -> None:
    model = lambda parameters: torch.tensor([1.0, 3.0])  # noqa: E731
    observations = torch.tensor([1.0, 2.0])

    assert MSEObjective(model, ["gpp"])({}, observations) == pytest.approx(0.5)
    # NSE = 1 - 1 / 0.5 = -1, the minimized objective is 1 - NSE = 2.
    assert NashSutcliffeObjective(model, ["gpp"])({}, observations) == pytest.approx(2.0)


def test_rmse_objective_is_the_square_root_of_the_mse_objective() -> None:
    from relic.parameters.calibration.calibrator import RMSEObjective

    model = lambda parameters: torch.tensor([1.0, 3.0])  # noqa: E731
    observations = torch.tensor([1.0, 2.0])

    assert RMSEObjective(model, ["gpp"])({}, observations) == pytest.approx(0.5 ** 0.5)
    with pytest.raises(RuntimeError, match="model diverged"):
        RMSEObjective(_diverging_model, ["gpp"])({}, observations)


def test_differential_evolution_callback_propagates_objective_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingObjective(ObjectiveFunction):
        def __call__(self, parameters, observations) -> float:
            raise RuntimeError("objective broke in callback")

    def fake_differential_evolution(func, bounds, **kwargs):
        kwargs["callback"](np.array([0.5]), 0.1)
        return SimpleNamespace(x=np.array([0.5]), nit=1, success=True, message="ok")

    monkeypatch.setattr(calibrator_module, "differential_evolution", fake_differential_evolution)
    calibrator = ParameterCalibrator(parameter_manager=Mock(), hooks=Mock())

    with pytest.raises(RuntimeError, match="objective broke in callback"):
        calibrator._differential_evolution(
            config=_config(parallel=False),
            objective_function=_FailingObjective(),
            observations=torch.tensor(0.5),
            initial_parameters={"test_param": 0.5},
        )


def test_calibration_metrics_are_absent_without_model_output() -> None:
    """Without a model_runner no metric is computable; none is fabricated."""
    calibrator = ParameterCalibrator(parameter_manager=Mock(), hooks=Mock())

    metrics = calibrator._calculate_metrics(
        {"test_param": 0.5}, _QuadraticObjective(), torch.tensor(0.5)
    )

    assert metrics is None


def _metrics_for(model_values, observations):
    calibrator = ParameterCalibrator(parameter_manager=Mock(), hooks=Mock())
    objective = MSEObjective(lambda parameters: torch.tensor(model_values), ["gpp"])
    return calibrator._calculate_metrics({}, objective, torch.tensor(observations))


def test_calibration_metrics_reject_observations_without_variance() -> None:
    with pytest.raises(ValueError, match="variance"):
        _metrics_for([1.0, 2.0], [3.0, 3.0])


def test_calibration_metrics_reject_percent_bias_of_zero_sum_observations() -> None:
    with pytest.raises(ValueError, match="percent bias"):
        _metrics_for([1.0, 2.0], [-1.0, 1.0])


def test_calibration_metrics_values() -> None:
    metrics = _metrics_for([1.0, 3.0], [1.0, 2.0])

    assert metrics.mse == pytest.approx(0.5)
    assert metrics.rmse == pytest.approx(0.5 ** 0.5)
    assert metrics.mae == pytest.approx(0.5)
    # SS_res = 1, SS_tot = 0.5: NSE = 1 - SS_res / SS_tot = -1.
    assert metrics.nash_sutcliffe == pytest.approx(-1.0)
    # The two series are perfectly collinear: Pearson r = 1 and R2 = r^2 = 1
    # (Moriasi et al. 2007, p. 887: R2 is the collinearity measure in [0, 1]).
    assert metrics.correlation == pytest.approx(1.0)
    assert metrics.r_squared == pytest.approx(1.0)
    assert metrics.bias == pytest.approx(0.5)
    assert metrics.percent_bias == pytest.approx(100.0 / 3.0)


def test_calibration_metrics_pearson_r_of_anticorrelated_series() -> None:
    metrics = _metrics_for([3.0, 2.0, 1.0], [1.0, 2.0, 3.0])

    assert metrics.correlation == pytest.approx(-1.0)
    assert metrics.r_squared == pytest.approx(1.0)
    assert metrics.nash_sutcliffe == pytest.approx(-3.0)


def test_calibration_metrics_reject_model_output_without_variance() -> None:
    # Pearson r (and R2 = r^2) is undefined for a constant model output; no value is substituted.
    with pytest.raises(ValueError, match="correlation"):
        _metrics_for([2.0, 2.0], [1.0, 2.0])


def test_calibration_metrics_record_has_no_defaulted_statistic() -> None:
    import dataclasses

    from relic.parameters.calibration.results import CalibrationMetrics

    fields = CalibrationMetrics.__dataclass_fields__
    for name in (
        "mse", "rmse", "mae", "r_squared", "nash_sutcliffe", "bias", "percent_bias", "correlation",
    ):
        assert fields[name].default is dataclasses.MISSING, name
    assert not hasattr(CalibrationMetrics, "compute_additional_metrics")


def test_parameter_bounds_scale_is_a_required_declared_choice() -> None:
    with pytest.raises(TypeError, match="scale"):
        ParameterBounds(lower=0.0, upper=1.0)
    with pytest.raises(ValueError, match="scale"):
        ParameterBounds(lower=0.0, upper=1.0, scale="cubic")
    assert ParameterBounds(lower=1.0e-3, upper=1.0, scale="log").scale == "log"


def test_calibration_config_numeric_defaults_are_read_from_the_parameter_database() -> None:
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    config = CalibrationConfig(
        method=CalibrationMethod.DIFFERENTIAL_EVOLUTION,
        target_parameters=["test_param"],
        parameter_bounds={"test_param": ParameterBounds(lower=0.0, upper=1.0, scale="linear")},
    )
    assert config.max_iterations == manager.get("optimization.calibration.max_iterations", required=True)
    assert config.tolerance == manager.get("optimization.calibration.objective_tolerance", required=True)
    assert "validation_split" not in CalibrationConfig.__dataclass_fields__

    with pytest.raises(TypeError, match="method"):
        CalibrationConfig.create_simple()
    simple = CalibrationConfig.create_simple(method="differential_evolution")
    assert simple.max_iterations == config.max_iterations
    assert simple.tolerance == config.tolerance


def test_calibration_metrics_propagate_a_model_failure() -> None:
    calibrator = ParameterCalibrator(parameter_manager=Mock(), hooks=Mock())
    objective = MSEObjective(_diverging_model, variable_names=["gpp"])

    with pytest.raises(RuntimeError, match="model diverged"):
        calibrator._calculate_metrics({"test_param": 0.5}, objective, torch.tensor([1.0, 2.0]))
