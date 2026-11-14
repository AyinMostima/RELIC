"""
Stochastic Scoring Rules for Validation

This module contains probabilistic scoring rules for forecast validation,
including CRPS, PIT, Brier scores, and quantile scores. These rules implement
strictly proper scoring functions as defined in Gneiting & Raftery (2007).

Key Scoring Rules:
    - CRPS: Continuous Ranked Probability Score
    - PIT: Probability Integral Transform consistency test
    - Brier Score: Binary event probability scoring
    - Quantile Score: Quantile forecast evaluation
    - Coverage Probability: Prediction interval validation

Mathematical Framework:
    Proper scoring rule S(F, y) where F is forecast distribution, y is observation:
    - CRPS: S_CRPS(F,y) = integral(F(t) - H(t-y))^2 dt
    - PIT: z = F(y) should be ~ Uniform(0,1) for calibrated forecasts
    - Brier: S_B(p,y,u) = (p - I_{y>u})^2 for threshold u
    - QS: S_Q(F,y,tau) = tau(y-q_tau)I_{y>q_tau} + (tau-1)(y-q_tau)I_{y<=q_tau}

References:
    - Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules, 
      prediction, and estimation. J. Am. Stat. Assoc. 102, 359-378.
    - Matheson, J.E. & Winkler, R.L. (1976). Scoring rules for continuous
      probability distributions. Management Science 22, 1087-1096.
    - Hersbach, H. (2000). Decomposition of the CRPS for ensemble forecasts.
      Weather Forecast. 15, 559-570.
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
import numpy as np

from ..models import (
    ValidationResultModel, ValidationLevel, ValidationCode,
    ValidationContext, ValidationResults, ValidationDetail
)
from ..decorators import validated_rule
from ....parameters import get_global_manager

logger = logging.getLogger(__name__)

# Import stochastic validation classes
from ....stochastic.validation import PITValidator, CRPSValidator

# Screening thresholds and significance levels of these rules live in the
# parameter database (validation/stochastic_scores.yaml).
_THRESHOLD_NAMESPACE = "validation.stochastic_scores"


def _threshold(name: str, overrides: Mapping[str, Any], override_key: str) -> float:
    """Return an explicit context override, else the ParameterManager value.

    ``overrides`` is the context ``tolerance_overrides`` (or, for the event
    definition and nominal coverage, the context ``metadata``); a key it does
    not carry is read from ``validation.stochastic_scores.<name>`` with unit
    validation and ``required=True`` (no literal default).
    """
    if override_key in overrides:
        return float(overrides[override_key])
    return float(
        get_global_manager().get(
            f"{_THRESHOLD_NAMESPACE}.{name}",
            units="dimensionless",
            required=True,
        )
    )


@validated_rule(
    rule_id="crps_ensemble_validation",
    layer="L1",
    name="CRPS Ensemble Validation",
    description="Validates forecast quality using Continuous Ranked Probability Score",
    domains=["stochastic", "forecasts", "ensembles"],
    tags=["crps", "proper_scoring", "ensemble"],
    units_contract={"crps": "same_as_observations"},
    references=[
        "Matheson & Winkler (1976). Scoring rules for continuous probability distributions",
        "Gneiting & Raftery (2007). Strictly proper scoring rules"
    ]
)
def validate_crps_ensemble(context: ValidationContext) -> ValidationResults:
    """
    Validate forecast quality using CRPS for ensemble predictions.
    
    Mathematical Expression:
        CRPS(F,y) = integral_{-inf}^{inf} [F(t) - H(t-y)]^2 dt
        For ensemble: CRPS = E|X - y| - (1/2)E|X - X'|
    
    Parameters:
        context: Validation context containing observations and forecasts
        
    Returns:
        List of validation results for CRPS scores
    """
    results = []

    # Extract data from context
    metadata = context.metadata or {}
    observations = metadata.get('observations')
    ensemble_forecasts = metadata.get('ensemble_forecasts')
    tolerance_overrides = context.tolerance_overrides or {}
    
    if observations is None or ensemble_forecasts is None:
        return [ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message="CRPS validation requires observations and ensemble forecasts",
            layer="L1",
            scope="crps_ensemble_validation",
            tags=["crps", "missing_data"]
        )]
    
    try:
        # Convert to tensors if needed
        if not isinstance(observations, Tensor):
            observations = torch.tensor(observations, dtype=torch.float32)
        if not isinstance(ensemble_forecasts, Tensor):
            ensemble_forecasts = torch.tensor(ensemble_forecasts, dtype=torch.float32)
        
        # Validate tensor shapes
        if observations.dim() != 1:
            results.append(ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message=f"Observations must be 1D tensor, got shape {observations.shape}",
                layer="L1",
                scope="crps_ensemble_validation",
                tags=["crps", "shape_error"]
            ))
            return results
        
        if ensemble_forecasts.dim() != 2:
            results.append(ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message=f"Ensemble forecasts must be 2D tensor, got shape {ensemble_forecasts.shape}",
                layer="L1",
                scope="crps_ensemble_validation",
                tags=["crps", "shape_error"]
            ))
            return results
        
        # Use CRPS validator
        crps_validator = CRPSValidator(decompose=True)
        crps_result = crps_validator.compute_score(observations, ensemble_forecasts)
        
        # Extract CRPS statistics
        # CRPSValidator.compute_score always returns these keys; a missing
        # score is a defect (KeyError -> failed result), never a filled value.
        crps_mean = crps_result['crps_mean']
        crps_std = crps_result['crps_std']
        
        # Thresholds: explicit tolerance_overrides, else ParameterManager.
        crps_warning_threshold = _threshold("crps_warning_threshold", tolerance_overrides, "crps_warning")
        crps_critical_threshold = _threshold("crps_critical_threshold", tolerance_overrides, "crps_critical")
        
        # Evaluate CRPS score
        if crps_mean > crps_critical_threshold:
            level = ValidationLevel.CRITICAL
            code = ValidationCode.SCORE_CRPS_CRITICAL
            message = f"CRPS critically high: {crps_mean:.4f} > {crps_critical_threshold}"
        elif crps_mean > crps_warning_threshold:
            level = ValidationLevel.WARNING
            code = ValidationCode.SCORE_CRPS_HIGH
            message = f"CRPS elevated: {crps_mean:.4f} > {crps_warning_threshold}"
        else:
            level = ValidationLevel.INFO
            code = ValidationCode.SCORE_CRPS_OK
            message = f"CRPS within acceptable range: {crps_mean:.4f}"
        
        results.append(ValidationResultModel(
            level=level,
            code=code,
            message=message,
            layer="L1",
            scope="crps_ensemble_validation",
            tags=["crps", "ensemble"],
            details=[ValidationDetail(
                variable_name="crps_score",
                value=crps_mean,
                units="same_as_observations",
                statistics={
                    "crps_std": crps_std,
                    "n_observations": len(observations),
                    "n_ensemble_members": ensemble_forecasts.shape[1]
                }
            )],
            metrics={
                "crps_mean": crps_mean,
                "crps_std": crps_std,
                "crps_warning_threshold": crps_warning_threshold,
                "crps_critical_threshold": crps_critical_threshold
            }
        ))
        
        # Add decomposition results if available
        if 'uncertainty' in crps_result:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.SCORE_CRPS_OK,
                message="CRPS decomposition computed",
                layer="L1",
                scope="crps_ensemble_validation",
                tags=["crps", "decomposition"],
                metrics={
                    "reliability": crps_result['reliability'],
                    "resolution": crps_result['resolution'],
                    "uncertainty": crps_result['uncertainty'],
                }
            ))
        
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"CRPS ensemble validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"CRPS ensemble validation failed: {str(e)}",
            layer="L1",
            scope="crps_ensemble_validation",
            tags=["crps", "execution_error"]
        ))
    
    return results


@validated_rule(
    rule_id="pit_uniformity_test",
    layer="L1",
    name="PIT Uniformity Test",
    description="Tests forecast calibration using Probability Integral Transform",
    domains=["stochastic", "forecasts", "calibration"],
    tags=["pit", "uniformity", "calibration"],
    references=[
        "Diebold et al. (1998). Evaluating density forecasts",
        "Gneiting et al. (2007). Probabilistic forecasts, calibration and sharpness"
    ]
)
def validate_pit_uniformity(context: ValidationContext) -> ValidationResults:
    """
    Test forecast calibration using PIT uniformity tests.
    
    Mathematical Expression:
        PIT value: z_i = F_i(y_i) where F_i is forecast CDF
        Null hypothesis: {z_i} ~ i.i.d. Uniform(0,1)
        Tests: Kolmogorov-Smirnov, Cramer-von Mises, Chi-squared
    
    Parameters:
        context: Validation context containing observations and forecasts
        
    Returns:
        List of validation results for PIT uniformity tests
    """
    results = []

    # Extract data from context
    metadata = context.metadata or {}
    observations = metadata.get('observations')
    forecast_ensemble = metadata.get('ensemble_forecasts')
    tolerance_overrides = context.tolerance_overrides or {}
    alpha = _threshold("pit_significance_level", tolerance_overrides, "alpha")
    
    if observations is None or forecast_ensemble is None:
        return [ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message="PIT validation requires observations and ensemble forecasts",
            layer="L1",
            scope="pit_uniformity_test",
            tags=["pit", "missing_data"]
        )]
    
    try:
        # Convert to tensors if needed
        if not isinstance(observations, Tensor):
            observations = torch.tensor(observations, dtype=torch.float32)
        if not isinstance(forecast_ensemble, Tensor):
            forecast_ensemble = torch.tensor(forecast_ensemble, dtype=torch.float32)
        
        # Use PIT validator
        pit_validator = PITValidator(alpha=alpha)
        pit_result = pit_validator.compute_score(observations, forecast_ensemble)
        
        # Extract test statistics
        # PITValidator.compute_score always returns the KS and CvM results
        # and the PIT statistics; the chi-squared test is optional and read
        # only where it ran. A missing value is a defect (KeyError -> failed
        # result), never a p-value of 1.0 that reads as a pass.
        ks_pvalue = pit_result['ks_pvalue']
        ks_reject = pit_result['ks_reject']
        cvm_pvalue = pit_result['cvm_pvalue']
        cvm_reject = pit_result['cvm_reject']

        pit_mean = pit_result['pit_mean']
        pit_std = pit_result['pit_std']
        
        # Kolmogorov-Smirnov test result
        if ks_reject:
            results.append(ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.PIT_UNIFORMITY_REJECT_KS,
                message=f"PIT uniformity rejected by KS test (p={ks_pvalue:.4f} < {alpha})",
                layer="L1",
                scope="pit_uniformity_test",
                tags=["pit", "ks_test", "reject"],
                details=[ValidationDetail(
                    variable_name="ks_test",
                    value=ks_pvalue,
                    threshold=alpha,
                    statistics={
                        "test_statistic": pit_result['ks_statistic']
                    }
                )]
            ))
        else:
            results.append(ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PIT_UNIFORMITY_OK,
                message=f"PIT uniformity passed KS test (p={ks_pvalue:.4f} >= {alpha})",
                layer="L1",
                scope="pit_uniformity_test",
                tags=["pit", "ks_test", "pass"],
                metrics={"ks_pvalue": ks_pvalue}
            ))
        
        # Cramer-von Mises test result (if available)
        if 'cvm_reject' in pit_result:
            if cvm_reject:
                results.append(ValidationResultModel(
                    level=ValidationLevel.ERROR,
                    code=ValidationCode.PIT_UNIFORMITY_REJECT_CVM,
                    message=(
                        "PIT uniformity rejected by Cramer-von Mises test "
                        f"(p={cvm_pvalue:.4f} < {alpha})"
                    ),
                    layer="L1",
                    scope="pit_uniformity_test",
                    tags=["pit", "cvm_test", "reject"],
                    details=[ValidationDetail(
                        variable_name="cvm_test",
                        value=pit_result['cvm_statistic'],
                        threshold=alpha,
                        statistics={"pvalue": cvm_pvalue}
                    )]
                ))
            else:
                results.append(ValidationResultModel(
                    level=ValidationLevel.INFO,
                    code=ValidationCode.PIT_UNIFORMITY_OK,
                    message=(
                        "PIT uniformity passed Cramer-von Mises test "
                        f"(p={cvm_pvalue:.4f} >= {alpha})"
                    ),
                    layer="L1",
                    scope="pit_uniformity_test",
                    tags=["pit", "cvm_test", "pass"],
                    metrics={
                        "cvm_pvalue": cvm_pvalue,
                        "cvm_statistic": pit_result['cvm_statistic'],
                    }
                ))
        
        # Chi-squared test result (if available)
        if 'chi2_reject' in pit_result:
            chi2_pvalue = pit_result['chi2_pvalue']
            if pit_result['chi2_reject']:
                results.append(ValidationResultModel(
                    level=ValidationLevel.WARNING,  # Less severe than KS/CVM
                    code=ValidationCode.PIT_UNIFORMITY_REJECT_CHI2,
                    message=f"PIT uniformity rejected by Chi-squared test (p={chi2_pvalue:.4f})",
                    layer="L1",
                    scope="pit_uniformity_test",
                    tags=["pit", "chi2_test", "reject"],
                    details=[ValidationDetail(
                        variable_name="chi2_test",
                        value=chi2_pvalue,
                        threshold=alpha,
                        statistics={
                            "chi2_statistic": pit_result['chi2_statistic']
                        }
                    )]
                ))
        
        # PIT statistics summary
        results.append(ValidationResultModel(
            level=ValidationLevel.INFO,
            code=ValidationCode.PIT_UNIFORMITY_OK,
            message=f"PIT statistics: mean={pit_mean:.3f}, std={pit_std:.3f}",
            layer="L1",
            scope="pit_uniformity_test",
            tags=["pit", "statistics"],
            metrics={
                "pit_mean": pit_mean,
                "pit_std": pit_std,
                "expected_mean": 0.5,
                "expected_std": np.sqrt(1.0/12),  # Uniform(0,1) std
                "n_observations": pit_result['n_observations']
            }
        ))
        
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"PIT uniformity validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"PIT uniformity validation failed: {str(e)}",
            layer="L1",
            scope="pit_uniformity_test",
            tags=["pit", "execution_error"]
        ))
    
    return results


@validated_rule(
    rule_id="brier_score_validation",
    layer="L1",
    name="Brier Score Validation",
    description="Validates probability forecasts using Brier score for binary events",
    domains=["stochastic", "probability", "binary"],
    tags=["brier", "probability", "binary_events"],
    references=[
        "Brier, G.W. (1950). Verification of forecasts expressed in terms of probability"
    ]
)
def validate_brier_score(context: ValidationContext) -> ValidationResults:
    """
    Validate probability forecasts using Brier score.
    
    Mathematical Expression:
        BS = (1/n) Sigma (p - o)^2
        where p is forecast probability, o in {0,1} is binary outcome
        Decomposition: BS = Reliability - Resolution + Uncertainty
    
    Parameters:
        context: Validation context with probability forecasts and binary outcomes
        
    Returns:
        List of validation results for Brier scores
    """
    results = []
    
    # Extract data from context
    metadata = context.metadata or {}
    observations = metadata.get('observations')
    forecast_probabilities = metadata.get('forecast_probabilities')
    # Binary event definition: explicit context metadata, else ParameterManager.
    threshold = _threshold("brier_event_threshold", metadata, "threshold")
    tolerance_overrides = context.tolerance_overrides or {}
    
    if observations is None or forecast_probabilities is None:
        return [ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message="Brier score validation requires observations and probability forecasts",
            layer="L1",
            scope="brier_score_validation",
            tags=["brier", "missing_data"]
        )]

    from ....stochastic.validation import brier_score

    try:
        # Convert to tensors if needed
        if not isinstance(observations, Tensor):
            observations = torch.tensor(observations, dtype=torch.float32)
        if not isinstance(forecast_probabilities, Tensor):
            forecast_probabilities = torch.tensor(forecast_probabilities, dtype=torch.float32)

        # Use Brier score function
        brier_result = brier_score(observations, forecast_probabilities, threshold)
        
        # Extract Brier score and decomposition
        bs_mean = brier_result.score
        # brier_score always returns this decomposition; no value is filled.
        base_rate = brier_result.decomposition['base_rate']
        reliability = brier_result.decomposition['reliability']
        resolution = brier_result.decomposition['resolution']
        uncertainty = brier_result.decomposition['uncertainty']
        
        # Define thresholds for Brier score evaluation
        bs_warning_threshold = _threshold("brier_warning_threshold", tolerance_overrides, "brier_warning")
        bs_critical_threshold = _threshold("brier_critical_threshold", tolerance_overrides, "brier_critical")
        
        # Evaluate Brier score
        if bs_mean > bs_critical_threshold:
            level = ValidationLevel.ERROR
            code = ValidationCode.BRIER_SCORE_HIGH
            message = f"Brier score critically high: {bs_mean:.4f} > {bs_critical_threshold}"
        elif bs_mean > bs_warning_threshold:
            level = ValidationLevel.WARNING
            code = ValidationCode.BRIER_SCORE_HIGH
            message = f"Brier score elevated: {bs_mean:.4f} > {bs_warning_threshold}"
        else:
            level = ValidationLevel.INFO
            code = ValidationCode.BRIER_SCORE_OK
            message = f"Brier score within acceptable range: {bs_mean:.4f}"
        
        results.append(ValidationResultModel(
            level=level,
            code=code,
            message=message,
            layer="L1",
            scope="brier_score_validation",
            tags=["brier", "probability"],
            details=[ValidationDetail(
                variable_name="brier_score",
                value=bs_mean,
                threshold=threshold,
                units="dimensionless",
                statistics={
                    "base_rate": base_rate,
                    "sample_size": brier_result.sample_size
                }
            )],
            metrics={
                "brier_score": bs_mean,
                "reliability": reliability,
                "resolution": resolution, 
                "uncertainty": uncertainty,
                "base_rate": base_rate
            }
        ))
        
        # Additional analysis for reliability
        reliability_warning = _threshold(
            "brier_reliability_warning_threshold", tolerance_overrides, "brier_reliability_warning"
        )
        if reliability > reliability_warning:  # High reliability indicates poor calibration
            results.append(ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.BRIER_SCORE_HIGH,
                message=f"High reliability component indicates poor calibration: {reliability:.4f}",
                layer="L1",
                scope="brier_score_validation",
                tags=["brier", "calibration", "reliability"]
            ))
        
    except ImportError as e:
        logger.error(f"Brier score validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Brier score validation failed: {str(e)}",
            layer="L1",
            scope="brier_score_validation",
            tags=["brier", "execution_error"]
        ))
    
    return results


@validated_rule(
    rule_id="coverage_probability_validation",
    layer="L1",
    name="Coverage Probability Validation",
    description="Validates prediction interval coverage probability",
    domains=["stochastic", "intervals", "coverage"],
    tags=["coverage", "intervals", "probability"],
    references=["Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences"]
)
def validate_coverage_probability(context: ValidationContext) -> ValidationResults:
    """
    Validate prediction interval coverage probability.
    
    Mathematical Expression:
        Coverage = (1/n) Sigma I(lower_i <= y_i <= upper_i)
        Test: Binomial test for nominal coverage level
    
    Parameters:
        context: Validation context with prediction intervals and observations
        
    Returns:
        List of validation results for coverage probability
    """
    results = []
    
    # Extract data from context
    metadata = context.metadata or {}
    observations = metadata.get('observations')
    lower_bounds = metadata.get('lower_bounds')
    upper_bounds = metadata.get('upper_bounds')
    # Nominal coverage: explicit context metadata, else ParameterManager.
    nominal_coverage = _threshold("nominal_interval_coverage", metadata, "nominal_coverage")
    tolerance_overrides = context.tolerance_overrides or {}
    
    if any(
        value is None
        for value in (observations, lower_bounds, upper_bounds)
    ):
        return [ValidationResultModel(
            level=ValidationLevel.WARNING,
            code=ValidationCode.MISSING_REQUIRED_DATA,
            message="Coverage validation requires observations, lower_bounds, and upper_bounds",
            layer="L1",
            scope="coverage_probability_validation",
            tags=["coverage", "missing_data"]
        )]

    from ....stochastic.validation import coverage_probability

    try:
        # Convert to tensors if needed
        if not isinstance(observations, Tensor):
            observations = torch.tensor(observations, dtype=torch.float32)
        if not isinstance(lower_bounds, Tensor):
            lower_bounds = torch.tensor(lower_bounds, dtype=torch.float32)
        if not isinstance(upper_bounds, Tensor):
            upper_bounds = torch.tensor(upper_bounds, dtype=torch.float32)

        # Use coverage probability function
        coverage_result = coverage_probability(observations, lower_bounds, upper_bounds, nominal_coverage)
        
        # Extract results
        empirical_coverage = coverage_result.score
        pvalue = coverage_result.pvalue
        # coverage_probability always returns these values; no value is filled.
        n_covered = coverage_result.decomposition['n_covered']
        interval_width_mean = coverage_result.decomposition['interval_width_mean']
        
        # Define acceptable deviation from nominal coverage
        coverage_tolerance = _threshold("coverage_deviation_tolerance", tolerance_overrides, "coverage_tolerance")
        coverage_error_alpha = _threshold(
            "coverage_error_significance_level", tolerance_overrides, "coverage_error_alpha"
        )
        coverage_warning_alpha = _threshold(
            "coverage_warning_significance_level", tolerance_overrides, "coverage_warning_alpha"
        )
        coverage_deviation = abs(empirical_coverage - nominal_coverage)

        # Evaluate coverage probability
        if pvalue < coverage_error_alpha:  # Highly significant deviation
            level = ValidationLevel.ERROR
            code = ValidationCode.COVERAGE_PROBABILITY_FAIL
            message = f"Coverage probability significantly different from nominal (p={pvalue:.4f})"
        elif pvalue < coverage_warning_alpha:  # Significant deviation
            level = ValidationLevel.WARNING
            code = ValidationCode.COVERAGE_PROBABILITY_FAIL
            message = f"Coverage probability may differ from nominal (p={pvalue:.4f})"
        elif coverage_deviation > coverage_tolerance:
            level = ValidationLevel.WARNING
            code = ValidationCode.COVERAGE_PROBABILITY_FAIL
            message = f"Coverage deviation exceeds tolerance: {coverage_deviation:.3f} > {coverage_tolerance}"
        else:
            level = ValidationLevel.INFO
            code = ValidationCode.COVERAGE_PROBABILITY_OK
            message = f"Coverage probability acceptable: {empirical_coverage:.3f} (nominal: {nominal_coverage})"
        
        results.append(ValidationResultModel(
            level=level,
            code=code,
            message=message,
            layer="L1",
            scope="coverage_probability_validation",
            tags=["coverage", "intervals"],
            details=[ValidationDetail(
                variable_name="coverage_probability",
                value=empirical_coverage,
                expected=nominal_coverage,
                units="dimensionless",
                statistics={
                    "pvalue": pvalue,
                    "n_covered": n_covered,
                    "n_total": len(observations),
                    "interval_width_mean": interval_width_mean
                }
            )],
            metrics={
                "empirical_coverage": empirical_coverage,
                "nominal_coverage": nominal_coverage,
                "coverage_deviation": coverage_deviation,
                "pvalue": pvalue,
                "coverage_ratio": empirical_coverage / nominal_coverage if nominal_coverage > 0 else float('nan')
            }
        ))
        
        # Additional warning for very narrow or wide intervals
        narrow_width = _threshold("interval_width_narrow_threshold", tolerance_overrides, "interval_width_narrow")
        wide_width = _threshold("interval_width_wide_threshold", tolerance_overrides, "interval_width_wide")
        if interval_width_mean < narrow_width:
            results.append(ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.COVERAGE_PROBABILITY_FAIL,
                message=f"Prediction intervals very narrow: mean width = {interval_width_mean:.4f}",
                layer="L1",
                scope="coverage_probability_validation",
                tags=["coverage", "interval_width", "narrow"]
            ))
        elif interval_width_mean > wide_width:
            results.append(ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.COVERAGE_PROBABILITY_FAIL,
                message=f"Prediction intervals very wide: mean width = {interval_width_mean:.4f}",
                layer="L1",
                scope="coverage_probability_validation",
                tags=["coverage", "interval_width", "wide"]
            ))
        
    except ImportError as e:
        logger.error(f"Coverage probability validation failed: {str(e)}")
        results.append(ValidationResultModel(
            level=ValidationLevel.ERROR,
            code=ValidationCode.GENERIC_VALIDATION_FAILED,
            message=f"Coverage probability validation failed: {str(e)}",
            layer="L1",
            scope="coverage_probability_validation",
            tags=["coverage", "execution_error"]
        ))
    
    return results


# Export validation rule functions
__all__ = [
    "validate_crps_ensemble",
    "validate_pit_uniformity", 
    "validate_brier_score",
    "validate_coverage_probability"
]
