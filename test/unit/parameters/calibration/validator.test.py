"""Tests for calibration validation: SSOT runtime conversion and the rule error contract."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import relic.parameters.calibration.validator as validator_module
import relic.utils.physical_constants as physical_constants_module
from relic.parameters import get_global_manager
from relic.parameters.calibration.results import CalibrationMetrics
from relic.parameters.calibration.validator import (
    CalibrationValidator,
    ComputationalValidationRule,
    ConvergenceValidationRule,
    ParameterBoundsValidationRule,
    PhysicalValidationRule,
    StatisticalValidationRule,
    UncertaintyValidationRule,
    ValidationCategory,
    ValidationRule,
    ValidationSeverity,
)
from relic.utils.time_constants import SECONDS_PER_HOUR


def _long_runtime_issues(issues):
    return [issue for issue in issues if issue.message.startswith("Very long runtime")]


def test_runtime_hours_use_seconds_per_hour_ssot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elapsed_time = 2.0 * float(SECONDS_PER_HOUR)
    result = SimpleNamespace(elapsed_time=elapsed_time, iteration_count=1000)
    rule = ComputationalValidationRule(
        max_runtime_hours=3.0,
        min_runtime_seconds=0.0,
    )
    original_get_constant = physical_constants_module.get_physical_constant

    def get_constant(name):
        if name == "seconds_per_hour":
            return float(SECONDS_PER_HOUR)
        return original_get_constant(name)

    monkeypatch.setattr(
        physical_constants_module,
        "get_physical_constant",
        get_constant,
    )
    assert _long_runtime_issues(rule.validate(result)) == []

    monkeypatch.setattr(
        physical_constants_module,
        "get_physical_constant",
        lambda name: float(SECONDS_PER_HOUR) / 2.0,
    )
    issues = _long_runtime_issues(rule.validate(result))

    assert len(issues) == 1
    assert issues[0].details["runtime_hours"] == pytest.approx(4.0)


def test_runtime_conversion_has_no_local_hour_literal() -> None:
    source = inspect.getsource(ComputationalValidationRule.validate)

    assert "3600" not in source
    assert "get_physical_constant" in source
    assert not hasattr(validator_module, "SECONDS_PER_HOUR")


def _broken_constraint(value: float) -> bool:
    raise ValueError("constraint evaluation broke")


def test_physical_constraint_failure_is_not_logged_as_a_passing_check() -> None:
    rule = PhysicalValidationRule(physical_constraints={"vcmax25": _broken_constraint})
    result = SimpleNamespace(optimized_parameters={"vcmax25": 50.0})

    with pytest.raises(ValueError, match="constraint evaluation broke"):
        rule.validate(result)


class _RaisingRule(ValidationRule):
    def __init__(self, error: Exception) -> None:
        super().__init__("Raising rule", ValidationCategory.STATISTICAL)
        self._error = error

    def validate(self, result):
        raise self._error


def _validator_with(rule: ValidationRule) -> CalibrationValidator:
    validator = CalibrationValidator(variable_category="streamflow")
    validator.rules = [rule]
    return validator


def test_rule_that_cannot_evaluate_malformed_result_data_fails_the_validation() -> None:
    validator = _validator_with(_RaisingRule(TypeError("objective_value is None")))

    outcome = validator.validate(SimpleNamespace(metrics=None))

    assert outcome.is_valid is False
    assert [issue.severity for issue in outcome.issues] == [ValidationSeverity.ERROR]
    assert "objective_value is None" in outcome.issues[0].message


@pytest.mark.parametrize(
    "error",
    [RuntimeError("parameter backend unavailable"), AttributeError("rule bug"), OSError("disk")],
    ids=["runtime", "attribute", "os"],
)
def test_rule_implementation_or_environment_failure_propagates(error: Exception) -> None:
    validator = _validator_with(_RaisingRule(error))

    with pytest.raises(type(error), match=str(error)):
        validator.validate(SimpleNamespace(metrics=None))


# --- Literature thresholds: Moriasi et al. (2007), Trans. ASABE 50(3):885-900 ---------------


def _metrics(**values) -> CalibrationMetrics:
    base = dict(
        mse=0.01, rmse=0.1, mae=0.08, r_squared=0.9, nash_sutcliffe=0.9,
        bias=0.0, percent_bias=0.0, correlation=0.95,
    )
    base.update(values)
    return CalibrationMetrics(**base)


def _statistical_issues(category: str, **metric_values):
    rule = StatisticalValidationRule(variable_category=category)
    return rule.validate(SimpleNamespace(metrics=_metrics(**metric_values)))


def test_statistical_rule_requires_a_declared_variable_category() -> None:
    with pytest.raises(TypeError, match="variable_category"):
        StatisticalValidationRule()
    with pytest.raises(ValueError, match="streamflow"):
        StatisticalValidationRule(variable_category="carbon_flux")
    with pytest.raises(TypeError, match="variable_category"):
        CalibrationValidator()


def test_statistical_thresholds_are_the_moriasi_2007_satisfactory_bounds() -> None:
    manager = get_global_manager()
    expected_pbias = {"streamflow": 25.0, "sediment": 55.0, "nitrogen": 70.0, "phosphorus": 70.0}
    for category, pbias in expected_pbias.items():
        rule = StatisticalValidationRule(variable_category=category)
        assert rule.max_bias_percent == pytest.approx(pbias)
        assert rule.max_bias_percent == manager.get(
            f"validation.calibration.statistical.max_abs_percent_bias.{category}", required=True
        )
        assert rule.min_nash_sutcliffe == pytest.approx(0.5)
        assert rule.min_r_squared == pytest.approx(0.5)


def test_nse_below_the_literature_bound_is_flagged() -> None:
    # 0.3 <= NSE <= 0.5 was accepted with the former local bound 0.3.
    issues = _statistical_issues("streamflow", nash_sutcliffe=0.4)
    assert [issue.severity for issue in issues] == [ValidationSeverity.WARNING]
    assert "Nash-Sutcliffe" in issues[0].message


def test_satisfactory_bounds_are_strict_inequalities() -> None:
    # Satisfactory means R2 > 0.5 (p. 887), NSE > 0.50 and |PBIAS| < bound (Table 4, p. 891).
    assert len(_statistical_issues("streamflow", nash_sutcliffe=0.5)) == 1
    assert len(_statistical_issues("streamflow", r_squared=0.5)) == 1
    assert len(_statistical_issues("streamflow", percent_bias=-25.0)) == 1
    assert _statistical_issues("streamflow", nash_sutcliffe=0.5001, r_squared=0.5001, percent_bias=24.9) == []


def test_percent_bias_bound_depends_on_the_declared_variable_category() -> None:
    assert _statistical_issues("sediment", percent_bias=52.0) == []
    assert len(_statistical_issues("streamflow", percent_bias=52.0)) == 1
    assert _statistical_issues("nitrogen", percent_bias=-69.0) == []
    assert len(_statistical_issues("phosphorus", percent_bias=70.0)) == 1


def test_only_an_nse_below_zero_is_an_error() -> None:
    # Moriasi et al. (2007, p. 887): NSE < 0 means the observed mean predicts better
    # than the model (unacceptable). The other unsatisfactory ratings are warnings.
    assert [i.severity for i in _statistical_issues("streamflow", nash_sutcliffe=-0.1)] == [
        ValidationSeverity.ERROR
    ]
    assert [i.severity for i in _statistical_issues("streamflow", r_squared=0.2)] == [
        ValidationSeverity.WARNING
    ]
    assert [i.severity for i in _statistical_issues("sediment", percent_bias=80.0)] == [
        ValidationSeverity.WARNING
    ]


def test_rule_thresholds_are_read_from_the_parameter_database() -> None:
    manager = get_global_manager()
    convergence = ConvergenceValidationRule()
    assert convergence.min_iterations == manager.get(
        "validation.calibration.convergence.min_iterations", required=True)
    assert convergence.max_stagnation_ratio == manager.get(
        "validation.calibration.convergence.max_stagnation_ratio", required=True)
    assert convergence.convergence_threshold == manager.get(
        "validation.calibration.convergence.min_total_improvement", required=True)
    assert ParameterBoundsValidationRule().tolerance == manager.get(
        "validation.calibration.bounds.tolerance", required=True)
    uncertainty = UncertaintyValidationRule()
    assert uncertainty.max_relative_uncertainty == manager.get(
        "validation.calibration.uncertainty.max_relative_uncertainty", required=True)
    assert "min_confidence_interval_coverage" not in inspect.signature(UncertaintyValidationRule).parameters
    for rule_class in (ConvergenceValidationRule, ParameterBoundsValidationRule, UncertaintyValidationRule):
        for parameter in list(inspect.signature(rule_class).parameters.values()):
            assert parameter.default is None, (rule_class.__name__, parameter.name)


def _cross_validation_result(nse: float, rate: float):
    return SimpleNamespace(
        metrics=_metrics(nash_sutcliffe=nse),
        target_parameters=["rate"],
        optimized_parameters={"rate": rate},
    )


def test_cross_validation_thresholds_are_read_from_the_parameter_database() -> None:
    manager = get_global_manager()
    validator = CalibrationValidator(variable_category="streamflow")
    expected = {
        "cross_validation_max_nse_drop": "validation.calibration.cross_validation.max_nse_drop",
        "cross_validation_max_nse_std": "validation.calibration.cross_validation.max_nse_std",
        "cross_validation_max_parameter_cv": "validation.calibration.cross_validation.max_parameter_cv",
    }
    for attribute, key in expected.items():
        assert getattr(validator, attribute) == manager.get(key, required=True), attribute

    training = _cross_validation_result(0.9, 1.0)
    # A drop of 0.25 NSE exceeds the declared bound; a drop of 0.15 does not.
    overfit = validator.validate_cross_validation(
        training, [_cross_validation_result(0.65, 1.0), _cross_validation_result(0.65, 1.0)])
    assert any("overfitting" in issue.message for issue in overfit.issues)
    stable = validator.validate_cross_validation(
        training, [_cross_validation_result(0.75, 1.0), _cross_validation_result(0.75, 1.0)])
    assert not any("overfitting" in issue.message for issue in stable.issues)
