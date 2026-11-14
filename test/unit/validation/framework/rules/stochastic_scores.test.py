"""L1 contract tests for stochastic validation rule adapters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relic.stochastic import validation as stochastic_validation
from relic.validation.framework.models import (
    ValidationCode,
    ValidationContext,
    ValidationLevel,
)
from relic.validation.framework.rules import stochastic_scores

pytestmark = pytest.mark.unit


def _context(**tolerances: float) -> ValidationContext:
    return ValidationContext(
        metadata={
            "observations": torch.tensor([0.0, 1.0]),
            "ensemble_forecasts": torch.tensor(
                [[-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]]
            ),
        },
        tolerance_overrides=tolerances,
    )


class _PITResult:
    payload: dict[str, object] = {}
    alpha: float | None = None

    def __init__(self, alpha: float) -> None:
        type(self).alpha = alpha

    def compute_score(self, observations, forecasts):
        return dict(type(self).payload)


class _CRPSResult:
    payload: dict[str, object] = {}

    def __init__(self, decompose: bool) -> None:
        assert decompose is True

    def compute_score(self, observations, forecasts):
        return dict(type(self).payload)


def test_pit_rule_reports_cvm_pass_and_propagates_alpha(monkeypatch) -> None:
    _PITResult.payload = {
        "ks_pvalue": 0.8,
        "ks_reject": False,
        "ks_statistic": 0.1,
        "cvm_pvalue": 0.7,
        "cvm_reject": False,
        "cvm_statistic": 0.05,
        "pit_mean": 0.5,
        "pit_std": 0.25,
        "n_observations": 2,
    }
    monkeypatch.setattr(stochastic_scores, "PITValidator", _PITResult)

    results = stochastic_scores.validate_pit_uniformity(_context(alpha=0.1))

    cvm_results = [result for result in results if "cvm_test" in result.tags]
    assert len(cvm_results) == 1
    assert cvm_results[0].code == ValidationCode.PIT_UNIFORMITY_OK
    assert cvm_results[0].level == ValidationLevel.INFO
    assert "Cramer-von Mises" in cvm_results[0].message
    assert _PITResult.alpha == pytest.approx(0.1)


def test_pit_rule_reports_cvm_rejection_with_statistic(monkeypatch) -> None:
    _PITResult.payload = {
        "ks_pvalue": 0.8,
        "ks_reject": False,
        "cvm_pvalue": 0.01,
        "cvm_reject": True,
        "cvm_statistic": 0.42,
        "pit_mean": 0.7,
        "pit_std": 0.1,
        "n_observations": 2,
    }
    monkeypatch.setattr(stochastic_scores, "PITValidator", _PITResult)

    results = stochastic_scores.validate_pit_uniformity(_context(alpha=0.05))

    rejected = [
        result
        for result in results
        if result.code == ValidationCode.PIT_UNIFORMITY_REJECT_CVM
    ]
    assert len(rejected) == 1
    assert rejected[0].level == ValidationLevel.ERROR
    assert rejected[0].details[0].variable_name == "cvm_test"
    assert rejected[0].details[0].value == pytest.approx(0.42)
    assert rejected[0].details[0].threshold == pytest.approx(0.05)


def test_pit_rule_does_not_emit_stale_anderson_darling_labels(
    monkeypatch,
) -> None:
    _PITResult.payload = {
        "ks_pvalue": 0.8,
        "ks_reject": False,
        "cvm_pvalue": 0.7,
        "cvm_reject": False,
        "cvm_statistic": 0.05,
        "pit_mean": 0.5,
        "pit_std": 0.25,
        "n_observations": 2,
    }
    monkeypatch.setattr(stochastic_scores, "PITValidator", _PITResult)

    results = stochastic_scores.validate_pit_uniformity(_context())

    assert all("ad_test" not in result.tags for result in results)
    assert all("Anderson-Darling" not in result.message for result in results)


def test_crps_rule_reports_true_decomposition_components(monkeypatch) -> None:
    _CRPSResult.payload = {
        "crps_mean": 0.3,
        "crps_std": 0.1,
        "reliability": 0.2,
        "resolution": 0.1,
        "uncertainty": 0.2,
    }
    monkeypatch.setattr(stochastic_scores, "CRPSValidator", _CRPSResult)

    results = stochastic_scores.validate_crps_ensemble(_context())

    decomposition = [
        result for result in results if "decomposition" in result.tags
    ]
    assert len(decomposition) == 1
    assert decomposition[0].metrics == {
        "reliability": pytest.approx(0.2),
        "resolution": pytest.approx(0.1),
        "uncertainty": pytest.approx(0.2),
    }
    assert decomposition[0].metrics["reliability"] - decomposition[0].metrics[
        "resolution"
    ] + decomposition[0].metrics["uncertainty"] == pytest.approx(0.3)


def test_crps_rule_uses_default_tolerances_when_overrides_are_none(
    monkeypatch,
) -> None:
    _CRPSResult.payload = {
        "crps_mean": 0.3,
        "crps_std": 0.1,
        "reliability": 0.2,
        "resolution": 0.1,
        "uncertainty": 0.2,
    }
    monkeypatch.setattr(stochastic_scores, "CRPSValidator", _CRPSResult)
    context = _context()
    context.tolerance_overrides = None

    results = stochastic_scores.validate_crps_ensemble(context)

    assert results[0].code == ValidationCode.SCORE_CRPS_OK


def test_brier_rule_preserves_murphy_decomposition() -> None:
    context = ValidationContext(
        metadata={
            "observations": torch.tensor([-1.0, -2.0, 1.0, 2.0]),
            "forecast_probabilities": torch.tensor([0.2, 0.2, 0.8, 0.8]),
            "threshold": 0.0,
        },
        tolerance_overrides={},
    )

    results = stochastic_scores.validate_brier_score(context)

    primary = results[0]
    assert primary.code == ValidationCode.BRIER_SCORE_OK
    reconstructed = (
        primary.metrics["reliability"]
        - primary.metrics["resolution"]
        + primary.metrics["uncertainty"]
    )
    assert reconstructed == pytest.approx(primary.metrics["brier_score"])


def test_coverage_rule_accepts_tensor_inputs_without_truth_value_error() -> None:
    context = ValidationContext(
        metadata={
            "observations": torch.tensor([0.0, 1.0]),
            "lower_bounds": torch.tensor([-0.1, 0.9]),
            "upper_bounds": torch.tensor([0.1, 1.1]),
            "nominal_coverage": 0.9,
        },
        tolerance_overrides={"coverage_tolerance": 0.2},
    )

    results = stochastic_scores.validate_coverage_probability(context)

    assert results[0].code == ValidationCode.COVERAGE_PROBABILITY_OK
    assert results[0].metrics["empirical_coverage"] == pytest.approx(1.0)
    assert all("execution_error" not in result.tags for result in results)


@pytest.mark.parametrize(
    ("score", "level", "code"),
    [
        (0.5, ValidationLevel.INFO, ValidationCode.SCORE_CRPS_OK),
        (1.5, ValidationLevel.WARNING, ValidationCode.SCORE_CRPS_HIGH),
        (2.5, ValidationLevel.CRITICAL, ValidationCode.SCORE_CRPS_CRITICAL),
    ],
)
def test_crps_rule_classifies_thresholds(
    monkeypatch,
    score: float,
    level: ValidationLevel,
    code: ValidationCode,
) -> None:
    _CRPSResult.payload = {"crps_mean": score, "crps_std": 0.1}
    monkeypatch.setattr(stochastic_scores, "CRPSValidator", _CRPSResult)

    results = stochastic_scores.validate_crps_ensemble(_context())

    assert results[0].level == level
    assert results[0].code == code


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"ensemble_forecasts": [[0.0, 1.0]]}, "requires observations"),
        ({"observations": [0.0]}, "requires observations"),
    ],
)
def test_crps_rule_reports_missing_inputs(metadata, message: str) -> None:
    results = stochastic_scores.validate_crps_ensemble(
        ValidationContext(metadata=metadata, tolerance_overrides={})
    )

    assert results[0].code == ValidationCode.MISSING_REQUIRED_DATA
    assert message in results[0].message


@pytest.mark.parametrize(
    ("observations", "forecasts", "message"),
    [
        ([[0.0]], [[0.0, 1.0]], "Observations must be 1D"),
        ([0.0], [0.0, 1.0], "Ensemble forecasts must be 2D"),
    ],
)
def test_crps_rule_reports_shape_errors(
    observations,
    forecasts,
    message: str,
) -> None:
    context = ValidationContext(
        metadata={
            "observations": observations,
            "ensemble_forecasts": forecasts,
        },
        tolerance_overrides={},
    )

    results = stochastic_scores.validate_crps_ensemble(context)

    assert results[0].code == ValidationCode.GENERIC_VALIDATION_FAILED
    assert message in results[0].message


def test_pit_rule_reports_ks_and_chi_squared_rejections(monkeypatch) -> None:
    _PITResult.payload = {
        "ks_pvalue": 0.01,
        "ks_reject": True,
        "ks_statistic": 0.4,
        "cvm_pvalue": 0.7,
        "cvm_reject": False,
        "cvm_statistic": 0.05,
        "chi2_pvalue": 0.02,
        "chi2_reject": True,
        "chi2_statistic": 12.0,
        "pit_mean": 0.7,
        "pit_std": 0.1,
        "n_observations": 100,
    }
    monkeypatch.setattr(stochastic_scores, "PITValidator", _PITResult)

    results = stochastic_scores.validate_pit_uniformity(_context())

    codes = {result.code for result in results}
    assert ValidationCode.PIT_UNIFORMITY_REJECT_KS in codes
    assert ValidationCode.PIT_UNIFORMITY_REJECT_CHI2 in codes


def test_pit_rule_reports_missing_inputs() -> None:
    results = stochastic_scores.validate_pit_uniformity(
        ValidationContext(metadata={}, tolerance_overrides={})
    )

    assert results[0].code == ValidationCode.MISSING_REQUIRED_DATA


@pytest.mark.parametrize(
    ("score", "reliability", "level", "extra_warning"),
    [
        (0.2, 0.05, ValidationLevel.INFO, False),
        (0.4, 0.05, ValidationLevel.WARNING, False),
        (0.6, 0.2, ValidationLevel.ERROR, True),
    ],
)
def test_brier_rule_classifies_score_and_reliability(
    monkeypatch,
    score: float,
    reliability: float,
    level: ValidationLevel,
    extra_warning: bool,
) -> None:
    def fake_brier(observations, probabilities, threshold):
        assert isinstance(observations, torch.Tensor)
        assert isinstance(probabilities, torch.Tensor)
        return SimpleNamespace(
            score=score,
            decomposition={
                "base_rate": 0.5,
                "reliability": reliability,
                "resolution": 0.1,
                "uncertainty": 0.25,
            },
            sample_size=len(observations),
        )

    monkeypatch.setattr(stochastic_validation, "brier_score", fake_brier)
    context = ValidationContext(
        metadata={
            "observations": [-1.0, 1.0],
            "forecast_probabilities": [0.2, 0.8],
            "threshold": 0.0,
        },
        tolerance_overrides={},
    )

    results = stochastic_scores.validate_brier_score(context)

    assert results[0].level == level
    assert ("reliability" in results[-1].tags) is extra_warning


def test_brier_rule_reports_missing_inputs() -> None:
    results = stochastic_scores.validate_brier_score(
        ValidationContext(metadata={}, tolerance_overrides={})
    )

    assert results[0].code == ValidationCode.MISSING_REQUIRED_DATA


@pytest.mark.parametrize(
    (
        "empirical",
        "pvalue",
        "width",
        "tolerance",
        "level",
        "width_tag",
    ),
    [
        (0.9, 0.5, 1.0, 0.05, ValidationLevel.INFO, None),
        (0.9, 0.03, 1.0, 0.05, ValidationLevel.WARNING, None),
        (0.9, 0.005, 1.0, 0.05, ValidationLevel.ERROR, None),
        (0.7, 0.5, 1.0, 0.05, ValidationLevel.WARNING, None),
        (0.9, 0.5, 0.001, 0.05, ValidationLevel.INFO, "narrow"),
        (0.9, 0.5, 11.0, 0.05, ValidationLevel.INFO, "wide"),
    ],
)
def test_coverage_rule_classifies_probability_and_width(
    monkeypatch,
    empirical: float,
    pvalue: float,
    width: float,
    tolerance: float,
    level: ValidationLevel,
    width_tag: str | None,
) -> None:
    def fake_coverage(observations, lower, upper, nominal):
        assert isinstance(observations, torch.Tensor)
        assert isinstance(lower, torch.Tensor)
        assert isinstance(upper, torch.Tensor)
        return SimpleNamespace(
            score=empirical,
            pvalue=pvalue,
            decomposition={
                "n_covered": 2,
                "interval_width_mean": width,
            },
        )

    monkeypatch.setattr(
        stochastic_validation,
        "coverage_probability",
        fake_coverage,
    )
    context = ValidationContext(
        metadata={
            "observations": [0.0, 1.0],
            "lower_bounds": [-0.1, 0.9],
            "upper_bounds": [0.1, 1.1],
            "nominal_coverage": 0.9,
        },
        tolerance_overrides={"coverage_tolerance": tolerance},
    )

    results = stochastic_scores.validate_coverage_probability(context)

    assert results[0].level == level
    if width_tag is None:
        assert len(results) == 1
    else:
        assert width_tag in results[1].tags


def test_coverage_rule_reports_missing_inputs() -> None:
    results = stochastic_scores.validate_coverage_probability(
        ValidationContext(metadata={}, tolerance_overrides={})
    )

    assert results[0].code == ValidationCode.MISSING_REQUIRED_DATA


def test_crps_rule_never_fills_missing_decomposition_components(monkeypatch) -> None:
    _CRPSResult.payload = {
        "crps_mean": 0.3,
        "crps_std": 0.1,
        "resolution": 0.1,
        "uncertainty": 0.2,
    }
    monkeypatch.setattr(stochastic_scores, "CRPSValidator", _CRPSResult)

    results = stochastic_scores.validate_crps_ensemble(_context())

    assert not any("decomposition" in result.tags for result in results)
    failures = [result for result in results if result.code == ValidationCode.GENERIC_VALIDATION_FAILED]
    assert len(failures) == 1 and "reliability" in failures[0].message
    assert failures[0].passed is False


def test_pit_rule_never_reads_a_missing_p_value_as_a_pass(monkeypatch) -> None:
    _PITResult.payload = {
        "ks_reject": False,
        "cvm_pvalue": 0.7,
        "cvm_reject": False,
        "cvm_statistic": 0.05,
        "pit_mean": 0.5,
        "pit_std": 0.25,
        "n_observations": 2,
    }
    monkeypatch.setattr(stochastic_scores, "PITValidator", _PITResult)

    results = stochastic_scores.validate_pit_uniformity(_context())

    assert not any("ks_test" in result.tags for result in results)
    failures = [result for result in results if result.code == ValidationCode.GENERIC_VALIDATION_FAILED]
    assert len(failures) == 1 and "ks_pvalue" in failures[0].message


class _ThresholdManager:
    """ParameterManager stand-in serving the stochastic-score thresholds."""

    def __init__(self, values):
        self.values = values
        self.requested = []

    def get(self, key, **kwargs):
        self.requested.append(key)
        assert kwargs.get("required") is True
        return self.values[key]


def test_crps_rule_reads_its_thresholds_from_the_parameter_manager(monkeypatch) -> None:
    _CRPSResult.payload = {"crps_mean": 0.3, "crps_std": 0.1}
    manager = _ThresholdManager({
        "validation.stochastic_scores.crps_warning_threshold": 0.2,
        "validation.stochastic_scores.crps_critical_threshold": 0.5,
    })
    monkeypatch.setattr(stochastic_scores, "CRPSValidator", _CRPSResult)
    monkeypatch.setattr(stochastic_scores, "get_global_manager", lambda: manager)

    results = stochastic_scores.validate_crps_ensemble(_context())

    assert results[0].code == ValidationCode.SCORE_CRPS_HIGH
    assert "validation.stochastic_scores.crps_warning_threshold" in manager.requested


def test_crps_rule_explicit_override_wins_over_the_parameter_manager(monkeypatch) -> None:
    _CRPSResult.payload = {"crps_mean": 0.3, "crps_std": 0.1}
    manager = _ThresholdManager({
        "validation.stochastic_scores.crps_warning_threshold": 0.2,
        "validation.stochastic_scores.crps_critical_threshold": 0.5,
    })
    monkeypatch.setattr(stochastic_scores, "CRPSValidator", _CRPSResult)
    monkeypatch.setattr(stochastic_scores, "get_global_manager", lambda: manager)

    results = stochastic_scores.validate_crps_ensemble(_context(crps_warning=0.4))

    assert results[0].code == ValidationCode.SCORE_CRPS_OK


def test_stochastic_score_thresholds_are_complete_parameter_entries() -> None:
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    keys = [
        "crps_warning_threshold", "crps_critical_threshold", "pit_significance_level",
        "brier_event_threshold", "brier_warning_threshold", "brier_critical_threshold",
        "brier_reliability_warning_threshold", "nominal_interval_coverage",
        "coverage_deviation_tolerance", "coverage_error_significance_level",
        "coverage_warning_significance_level", "interval_width_narrow_threshold",
        "interval_width_wide_threshold",
    ]
    for name in keys:
        spec = manager.get_parameter(f"validation.stochastic_scores.{name}")
        assert spec is not None, name
        assert spec.units == "dimensionless" and spec.references, name
        lower, upper = spec.bounds
        assert lower <= float(spec.value) <= upper, name
