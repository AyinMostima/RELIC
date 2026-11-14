"""
Calibration Result Validation System

This module provides comprehensive validation and quality assurance for
parameter calibration results. It includes statistical tests, physical
constraint checking, and scientific plausibility assessment.

Scientific Background:
    Calibration validation is essential to ensure that optimized parameters
    are physically meaningful and lead to reliable model predictions. 
    Validation includes:
    - Statistical significance testing
    - Physical constraint verification
    - Cross-validation analysis
    - Predictive performance assessment

References:
    - Refsgaard & Henriksen (2004). Modelling guidelines--terminology and guiding principles.
    - Gupta et al. (2008). Decomposition of the mean squared error and NSE performance criteria.
    - Pianosi et al. (2016). Sensitivity analysis of environmental models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..manager import get_global_manager
from .results import CalibrationResult

# Namespace of the per-variable-category percent-bias bounds (Moriasi et al. 2007, Table 4).
_PBIAS_BOUNDS_NAMESPACE = "validation.calibration.statistical.max_abs_percent_bias"


def _stated_or_database(value, key: str):
    """Return the stated rule setting, or the parameter-database value of ``key``."""
    return value if value is not None else get_global_manager().get(key, required=True)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks"""
    STATISTICAL = "statistical"
    PHYSICAL = "physical"
    COMPUTATIONAL = "computational"
    SCIENTIFIC = "scientific"
    CONVERGENCE = "convergence"
    UNCERTAINTY = "uncertainty"


@dataclass
class CalibrationValidationIssue:
    """Individual validation issue"""
    category: ValidationCategory
    severity: ValidationSeverity
    parameter: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None
    
    def __str__(self) -> str:
        param_str = f" (Parameter: {self.parameter})" if self.parameter else ""
        return f"[{self.severity.value.upper()}] {self.category.value}{param_str}: {self.message}"


@dataclass
class CalibrationValidationResult:
    """Result of calibration validation"""
    is_valid: bool
    issues: List[CalibrationValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[CalibrationValidationIssue]:
        """Get issues of specific severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[CalibrationValidationIssue]:
        """Get issues of specific category"""
        return [issue for issue in self.issues if issue.category == category]
    
    def has_critical_issues(self) -> bool:
        """Check if there are critical validation issues"""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    def has_errors(self) -> bool:
        """Check if there are error-level validation issues"""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    def get_quality_score(self) -> float:
        """
        Calculate overall quality score (0-100)
        
        Returns:
            Quality score where 100 is perfect, 0 is completely invalid
        """
        if not self.issues:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.INFO: 1,
            ValidationSeverity.WARNING: 5,
            ValidationSeverity.ERROR: 20,
            ValidationSeverity.CRITICAL: 50
        }
        
        total_penalty = sum(severity_weights[issue.severity] for issue in self.issues)
        
        # Calculate score (capped at 0)
        score = max(0.0, 100.0 - total_penalty)
        return score


ValidationIssue = CalibrationValidationIssue
ValidationResult = CalibrationValidationResult


class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, name: str, category: ValidationCategory):
        self.name = name
        self.category = category
    
    @abstractmethod
    def validate(self, result: CalibrationResult) -> List[CalibrationValidationIssue]:
        """
        Validate calibration result
        
        Parameters:
            result: Calibration result to validate
            
        Returns:
            List of validation issues
        """


class ConvergenceValidationRule(ValidationRule):
    """Validate calibration convergence"""
    
    def __init__(self,
                 min_iterations: Optional[int] = None,
                 max_stagnation_ratio: Optional[float] = None,
                 convergence_threshold: Optional[float] = None):
        super().__init__("Convergence Validation", ValidationCategory.CONVERGENCE)
        self.min_iterations = int(_stated_or_database(
            min_iterations, "validation.calibration.convergence.min_iterations"))
        self.max_stagnation_ratio = float(_stated_or_database(
            max_stagnation_ratio, "validation.calibration.convergence.max_stagnation_ratio"))
        self.convergence_threshold = float(_stated_or_database(
            convergence_threshold, "validation.calibration.convergence.min_total_improvement"))
    
    def validate(self, result: CalibrationResult) -> List[CalibrationValidationIssue]:
        issues = []
        
        # Check if calibration succeeded
        if not result.success:
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=ValidationSeverity.CRITICAL,
                parameter=None,
                message=f"Calibration failed: {result.termination_reason}",
                recommendation="Re-run calibration with different settings or check model implementation"
            ))
        
        # Check minimum iterations
        if result.iteration_count < self.min_iterations:
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=ValidationSeverity.WARNING,
                parameter=None,
                message=f"Very few iterations ({result.iteration_count} < {self.min_iterations})",
                recommendation="Consider increasing iteration limit or checking convergence criteria"
            ))
        
        # Check for stagnation in convergence history
        if len(result.convergence_history) > 10:
            recent_values = result.convergence_history[-int(len(result.convergence_history) * 0.3):]
            if len(set(np.round(recent_values, 10))) == 1:
                stagnation_ratio = len(recent_values) / len(result.convergence_history)
                if stagnation_ratio > self.max_stagnation_ratio:
                    issues.append(CalibrationValidationIssue(
                        category=self.category,
                        severity=ValidationSeverity.WARNING,
                        parameter=None,
                        message=f"Convergence stagnation detected ({stagnation_ratio:.1%} of iterations)",
                        recommendation="Check if global optimum was reached or if algorithm got stuck in local minimum"
                    ))
        
        # Check convergence improvement
        if len(result.convergence_history) >= 2:
            improvement = result.convergence_history[0] - result.convergence_history[-1]
            if improvement < self.convergence_threshold:
                issues.append(CalibrationValidationIssue(
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    parameter=None,
                    message=f"Very small improvement achieved ({improvement:.2e})",
                    recommendation="Consider stricter convergence criteria or different optimization method"
                ))
        
        return issues


class ParameterBoundsValidationRule(ValidationRule):
    """Validate parameter bounds adherence"""
    
    def __init__(self, tolerance: Optional[float] = None):
        super().__init__("Parameter Bounds Validation", ValidationCategory.PHYSICAL)
        self.tolerance = float(_stated_or_database(tolerance, "validation.calibration.bounds.tolerance"))
    
    def validate(self, result: CalibrationResult) -> List[CalibrationValidationIssue]:
        issues = []
        
        for param_name in result.target_parameters:
            if param_name not in result.optimized_parameters:
                issues.append(CalibrationValidationIssue(
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    parameter=param_name,
                    message="Parameter missing from optimized results",
                    recommendation="Check calibration implementation"
                ))
                continue
            
            optimized_value = result.optimized_parameters[param_name]
            
            if param_name in result.parameter_bounds:
                bounds = result.parameter_bounds[param_name]
                
                # Extract bounds (handle different bound formats)
                if hasattr(bounds, 'lower') and hasattr(bounds, 'upper'):
                    lower, upper = bounds.lower, bounds.upper
                elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    lower, upper = bounds[0], bounds[1]
                else:
                    continue
                
                # Check bounds violations
                if optimized_value < lower - self.tolerance:
                    issues.append(CalibrationValidationIssue(
                        category=self.category,
                        severity=ValidationSeverity.ERROR,
                        parameter=param_name,
                        message=f"Parameter below lower bound: {optimized_value:.6f} < {lower:.6f}",
                        details={'value': optimized_value, 'lower_bound': lower},
                        recommendation="Check optimization constraints or parameter bounds"
                    ))
                
                if optimized_value > upper + self.tolerance:
                    issues.append(CalibrationValidationIssue(
                        category=self.category,
                        severity=ValidationSeverity.ERROR,
                        parameter=param_name,
                        message=f"Parameter above upper bound: {optimized_value:.6f} > {upper:.6f}",
                        details={'value': optimized_value, 'upper_bound': upper},
                        recommendation="Check optimization constraints or parameter bounds"
                    ))
                
                # Check if parameter is at boundary (might indicate constraint issues)
                if abs(optimized_value - lower) < self.tolerance:
                    issues.append(CalibrationValidationIssue(
                        category=self.category,
                        severity=ValidationSeverity.WARNING,
                        parameter=param_name,
                        message=f"Parameter at lower bound: {optimized_value:.6f}",
                        recommendation="Consider expanding parameter bounds if physically justified"
                    ))
                
                if abs(optimized_value - upper) < self.tolerance:
                    issues.append(CalibrationValidationIssue(
                        category=self.category,
                        severity=ValidationSeverity.WARNING,
                        parameter=param_name,
                        message=f"Parameter at upper bound: {optimized_value:.6f}",
                        recommendation="Consider expanding parameter bounds if physically justified"
                    ))
        
        return issues


class StatisticalValidationRule(ValidationRule):
    """Validate statistical performance metrics against the Moriasi et al. (2007) ratings.

    The caller declares the variable category of the calibrated output; its
    percent-bias bound is the one declared for that category (streamflow,
    sediment, nitrogen, phosphorus). A category without a declared bound raises.
    """

    def __init__(self,
                 variable_category: str,
                 min_r_squared: Optional[float] = None,
                 min_nash_sutcliffe: Optional[float] = None,
                 max_rmse_threshold: Optional[float] = None,
                 max_bias_percent: Optional[float] = None):
        super().__init__("Statistical Performance Validation", ValidationCategory.STATISTICAL)
        declared = get_global_manager().get_namespace_parameters(_PBIAS_BOUNDS_NAMESPACE)
        categories = sorted(key.rsplit(".", 1)[-1] for key in declared)
        if variable_category not in categories:
            raise ValueError(
                f"variable_category {variable_category!r} has no declared percent-bias bound; "
                f"declared categories: {categories}"
            )
        self.variable_category = variable_category
        self.min_r_squared = float(_stated_or_database(
            min_r_squared, "validation.calibration.statistical.min_r_squared"))
        self.min_nash_sutcliffe = float(_stated_or_database(
            min_nash_sutcliffe, "validation.calibration.statistical.min_nash_sutcliffe"))
        self.max_rmse_threshold = max_rmse_threshold
        self.max_bias_percent = float(_stated_or_database(
            max_bias_percent, f"{_PBIAS_BOUNDS_NAMESPACE}.{variable_category}"))
    
    def validate(self, result: CalibrationResult) -> List[CalibrationValidationIssue]:
        issues = []
        
        if not result.metrics:
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=ValidationSeverity.WARNING,
                parameter=None,
                message="No statistical metrics available for validation",
                recommendation="Ensure metrics are calculated during calibration"
            ))
            return issues
        
        metrics = result.metrics

        # References: Moriasi et al. (2007), Trans. ASABE 50(3):885-900, p. 887 (R2 > 0.5;
        #   NSE < 0 unacceptable) and Table 4, p. 891 (satisfactory NSE and PBIAS ratings).
        # Mathematical expression: satisfactory iff R2 > R2_min, NSE > NSE_min and
        #   |PBIAS| < PBIAS_max(category); NSE < 0 means the observed mean predicts better.
        # Parameter meanings: R2 = r^2 (Pearson r), NSE the Nash-Sutcliffe efficiency, PBIAS
        #   the percent bias; R2_min, NSE_min, PBIAS_max from validation.calibration.statistical.*.
        # Units: R2 and NSE dimensionless; PBIAS and PBIAS_max in percent.
        # Domain/constraints: the ratings are stated for continuous long-term simulation at a
        #   monthly time step; a violated rating is a WARNING, NSE < 0 an ERROR.
        # Sign convention: only |PBIAS| is rated; the sign of the bias is not.
        # R-squared validation
        if not metrics.r_squared > self.min_r_squared:
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=ValidationSeverity.WARNING,
                parameter=None,
                message=f"Low R-squared: {metrics.r_squared:.3f} <= {self.min_r_squared:.3f}",
                details={'r_squared': metrics.r_squared},
                recommendation="Consider model structure improvements or additional calibration parameters"
            ))

        # Nash-Sutcliffe validation
        if not metrics.nash_sutcliffe > self.min_nash_sutcliffe:
            severity = ValidationSeverity.ERROR if metrics.nash_sutcliffe < 0.0 else ValidationSeverity.WARNING
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=severity,
                parameter=None,
                message=f"Low Nash-Sutcliffe efficiency: {metrics.nash_sutcliffe:.3f} <= {self.min_nash_sutcliffe:.3f}",
                details={'nash_sutcliffe': metrics.nash_sutcliffe},
                recommendation="Model performance is not satisfactory; consider model reformulation"
            ))
        
        # RMSE validation (if threshold provided)
        if self.max_rmse_threshold and metrics.rmse > self.max_rmse_threshold:
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=ValidationSeverity.WARNING,
                parameter=None,
                message=f"High RMSE: {metrics.rmse:.6f} > {self.max_rmse_threshold:.6f}",
                details={'rmse': metrics.rmse},
                recommendation="Consider model improvements or parameter bounds adjustment"
            ))
        
        # Bias validation
        if not abs(metrics.percent_bias) < self.max_bias_percent:
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=ValidationSeverity.WARNING,
                parameter=None,
                message=(
                    f"High bias: {metrics.percent_bias:.1f}% outside +-{self.max_bias_percent:.1f}% "
                    f"({self.variable_category})"
                ),
                details={'percent_bias': metrics.percent_bias, 'variable_category': self.variable_category},
                recommendation="Check for systematic model errors or parameter bias"
            ))
        
        return issues


class PhysicalValidationRule(ValidationRule):
    """Validate physical plausibility of parameters"""
    
    def __init__(self, physical_constraints: Optional[Dict[str, Callable[[float], bool]]] = None):
        super().__init__("Physical Plausibility Validation", ValidationCategory.PHYSICAL)
        self.physical_constraints = physical_constraints or {}
        
        # Default physical constraints for common parameters
        self._add_default_constraints()
    
    def _add_default_constraints(self):
        """Add common physical constraints"""
        default_constraints = {
            # Photosynthesis parameters
            'vcmax25': lambda x: 0 < x < 200,  # mol m-2 s-1
            'jmax25': lambda x: 0 < x < 400,   # mol m-2 s-1
            'alpha': lambda x: 0 < x < 1,      # Quantum efficiency
            
            # Temperature response parameters
            'q10': lambda x: 1 < x < 5,        # Q10 factor
            'tref': lambda x: 273 < x < 310,   # Reference temperature (K)
            
            # Soil parameters
            'ksat': lambda x: x > 0,           # Saturated hydraulic conductivity
            'porosity': lambda x: 0 < x < 1,   # Soil porosity
            'field_capacity': lambda x: 0 < x < 1,  # Field capacity
            
            # Carbon pool parameters
            'c_fast': lambda x: x > 0,         # Fast carbon pool
            'c_slow': lambda x: x > 0,         # Slow carbon pool
            'decomp_rate': lambda x: x > 0,    # Decomposition rate
        }
        
        # Only add if not already specified
        for param, constraint in default_constraints.items():
            if param not in self.physical_constraints:
                self.physical_constraints[param] = constraint
    
    def validate(self, result: CalibrationResult) -> List[CalibrationValidationIssue]:
        issues = []
        
        for param_name, value in result.optimized_parameters.items():
            # Check for NaN or infinite values
            if not np.isfinite(value):
                issues.append(CalibrationValidationIssue(
                    category=self.category,
                    severity=ValidationSeverity.CRITICAL,
                    parameter=param_name,
                    message=f"Parameter has invalid value: {value}",
                    recommendation="Check optimization algorithm and objective function implementation"
                ))
                continue
            
            # Check physical constraints. An error raised by a constraint function
            # propagates (CalibrationValidator.validate records malformed-data errors).
            for constraint_param, constraint_func in self.physical_constraints.items():
                if constraint_param in param_name.lower() or param_name.lower() in constraint_param:
                    if not constraint_func(value):
                        issues.append(CalibrationValidationIssue(
                            category=self.category,
                            severity=ValidationSeverity.WARNING,
                            parameter=param_name,
                            message=f"Parameter may be physically implausible: {value:.6f}",
                            details={'value': value, 'constraint': constraint_param},
                            recommendation="Verify parameter bounds and physical interpretation"
                        ))
        
        return issues


class UncertaintyValidationRule(ValidationRule):
    """Validate parameter uncertainty estimates"""
    
    def __init__(self, max_relative_uncertainty: Optional[float] = None):
        super().__init__("Parameter Uncertainty Validation", ValidationCategory.UNCERTAINTY)
        self.max_relative_uncertainty = float(_stated_or_database(
            max_relative_uncertainty, "validation.calibration.uncertainty.max_relative_uncertainty"))
    
    def validate(self, result: CalibrationResult) -> List[CalibrationValidationIssue]:
        issues = []
        
        if not result.parameter_uncertainties:
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=ValidationSeverity.INFO,
                parameter=None,
                message="No uncertainty estimates available",
                recommendation="Consider running uncertainty quantification analysis"
            ))
            return issues
        
        for param_name, uncertainty in result.parameter_uncertainties.items():
            # Check relative uncertainty
            if param_name in result.optimized_parameters:
                optimized_value = result.optimized_parameters[param_name]
                if optimized_value != 0:
                    relative_uncertainty = uncertainty.std / abs(optimized_value)
                    
                    if relative_uncertainty > self.max_relative_uncertainty:
                        issues.append(CalibrationValidationIssue(
                            category=self.category,
                            severity=ValidationSeverity.WARNING,
                            parameter=param_name,
                            message=f"High parameter uncertainty: {relative_uncertainty:.1%}",
                            details={'relative_uncertainty': relative_uncertainty},
                            recommendation="Parameter may be poorly constrained by observations"
                        ))
            
            # Check confidence interval sanity
            ci_lower, ci_upper = uncertainty.confidence_interval_95
            if ci_lower >= ci_upper:
                issues.append(CalibrationValidationIssue(
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    parameter=param_name,
                    message="Invalid confidence interval",
                    recommendation="Check uncertainty estimation method"
                ))
            
            # Check if uncertainty estimates are reasonable
            if uncertainty.std <= 0:
                issues.append(CalibrationValidationIssue(
                    category=self.category,
                    severity=ValidationSeverity.ERROR,
                    parameter=param_name,
                    message="Invalid uncertainty estimate (non-positive standard deviation)",
                    recommendation="Check uncertainty quantification implementation"
                ))
        
        return issues


class ComputationalValidationRule(ValidationRule):
    """Validate computational aspects of calibration"""
    
    def __init__(self,
                 max_runtime_hours: float = 24.0,
                 min_runtime_seconds: float = 1.0):
        super().__init__("Computational Validation", ValidationCategory.COMPUTATIONAL)
        self.max_runtime_hours = max_runtime_hours
        self.min_runtime_seconds = min_runtime_seconds
    
    def validate(self, result: CalibrationResult) -> List[CalibrationValidationIssue]:
        issues = []

        from relic.utils.physical_constants import get_physical_constant

        # References: RELIC calibration reporting and SI time-unit definitions.
        # Mathematical expression: t_h = t_s / S_h.
        # Parameter meanings: t_s is elapsed runtime; S_h is seconds per hour.
        # Units: elapsed_time [s] is converted to runtime_hours [h].
        # Domain/constraints: elapsed_time >= 0 and S_h > 0 from the SSOT.
        # Sign convention: positive runtime advances wall-clock elapsed time.
        seconds_per_hour = float(get_physical_constant("seconds_per_hour"))
        runtime_hours = result.elapsed_time / seconds_per_hour
        
        if runtime_hours > self.max_runtime_hours:
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=ValidationSeverity.WARNING,
                parameter=None,
                message=f"Very long runtime: {runtime_hours:.1f} hours",
                details={'runtime_hours': runtime_hours},
                recommendation="Consider algorithm optimization or parallel processing"
            ))
        
        if result.elapsed_time < self.min_runtime_seconds:
            issues.append(CalibrationValidationIssue(
                category=self.category,
                severity=ValidationSeverity.WARNING,
                parameter=None,
                message=f"Very short runtime: {result.elapsed_time:.2f} seconds",
                recommendation="Check if calibration actually ran or converged too quickly"
            ))
        
        # Check iteration efficiency
        if result.iteration_count > 0 and result.elapsed_time > 0:
            time_per_iteration = result.elapsed_time / result.iteration_count
            if time_per_iteration > 60:  # More than 1 minute per iteration
                issues.append(CalibrationValidationIssue(
                    category=self.category,
                    severity=ValidationSeverity.INFO,
                    parameter=None,
                    message=f"Slow iteration speed: {time_per_iteration:.1f} seconds/iteration",
                    recommendation="Consider computational optimization"
                ))
        
        return issues


class CalibrationValidator:
    """
    Main calibration validation engine
    
    This class coordinates multiple validation rules and provides comprehensive
    assessment of calibration results including statistical, physical, and
    computational validation.
    """
    
    def __init__(self, variable_category: str):
        """
        Parameters:
            variable_category: Category of the calibrated output variable
                (e.g. 'streamflow'); it selects the declared percent-bias bound
                of the statistical rule and has no default.
        """
        self.rules: List[ValidationRule] = []
        self.logger = logging.getLogger(__name__)
        self.statistical_rule = StatisticalValidationRule(variable_category=variable_category)
        # Cross-validation thresholds (RELIC convention values in the parameter database).
        manager = get_global_manager()
        self.cross_validation_max_nse_drop = float(
            manager.get("validation.calibration.cross_validation.max_nse_drop", required=True))
        self.cross_validation_max_nse_std = float(
            manager.get("validation.calibration.cross_validation.max_nse_std", required=True))
        self.cross_validation_max_parameter_cv = float(
            manager.get("validation.calibration.cross_validation.max_parameter_cv", required=True))

        # Add default validation rules
        self._add_default_rules()

    def _add_default_rules(self):
        """Add default validation rules"""
        self.add_rule(ConvergenceValidationRule())
        self.add_rule(ParameterBoundsValidationRule())
        self.add_rule(self.statistical_rule)
        self.add_rule(PhysicalValidationRule())
        self.add_rule(UncertaintyValidationRule())
        self.add_rule(ComputationalValidationRule())
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.rules.append(rule)
        self.logger.info(f"Added validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule by name"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                self.logger.info(f"Removed validation rule: {rule_name}")
                return True
        return False
    
    def validate(self, result: CalibrationResult) -> CalibrationValidationResult:
        """
        Validate calibration result using all registered rules
        
        Parameters:
            result: Calibration result to validate
            
        Returns:
            Comprehensive validation result
        """
        all_issues = []
        rule_summaries = {}
        
        self.logger.info(f"Validating calibration result with {len(self.rules)} rules")
        
        # Run all validation rules. A rule that cannot evaluate the result because
        # the result data is malformed (TypeError, ValueError, KeyError) is recorded
        # as an ERROR issue, which makes the validation invalid. Any other error (a
        # rule implementation bug, an unavailable backend, an I/O failure) is not a
        # finding about the result and propagates.
        for rule in self.rules:
            try:
                issues = rule.validate(result)
            except (TypeError, ValueError, KeyError) as e:
                self.logger.error(f"Validation rule '{rule.name}' failed: {e}")
                all_issues.append(CalibrationValidationIssue(
                    category=ValidationCategory.COMPUTATIONAL,
                    severity=ValidationSeverity.ERROR,
                    parameter=None,
                    message=f"Validation rule '{rule.name}' failed: {str(e)}",
                    recommendation="Check the calibration result data this rule evaluates"
                ))
                continue
            all_issues.extend(issues)

            rule_summaries[rule.name] = {
                'category': rule.category.value,
                'issue_count': len(issues),
                'severities': {severity.value: len([i for i in issues if i.severity == severity])
                             for severity in ValidationSeverity}
            }

            self.logger.debug(f"Rule '{rule.name}' found {len(issues)} issues")
        
        # Determine overall validity
        is_valid = not any(
            issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
            for issue in all_issues
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, result)
        
        # Create summary
        summary = {
            'total_issues': len(all_issues),
            'rule_summaries': rule_summaries,
            'quality_score': None,  # Will be calculated by CalibrationValidationResult
            'critical_issues': len([i for i in all_issues if i.severity == ValidationSeverity.CRITICAL]),
            'error_issues': len([i for i in all_issues if i.severity == ValidationSeverity.ERROR]),
            'warning_issues': len([i for i in all_issues if i.severity == ValidationSeverity.WARNING]),
            'info_issues': len([i for i in all_issues if i.severity == ValidationSeverity.INFO]),
        }
        
        # Create validation result
        validation_result = CalibrationValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            summary=summary,
            recommendations=recommendations
        )
        
        # Calculate and add quality score to summary
        summary['quality_score'] = validation_result.get_quality_score()
        
        self.logger.info(f"Validation complete: {len(all_issues)} issues found, "
                        f"quality score: {summary['quality_score']:.1f}")
        
        return validation_result
    
    def _generate_recommendations(self, 
                                 issues: List[CalibrationValidationIssue],
                                 result: CalibrationResult) -> List[str]:
        """Generate high-level recommendations based on validation issues"""
        recommendations = []
        
        # Count issues by category and severity
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        
        if critical_issues:
            recommendations.append(
                "CRITICAL: Calibration results are unreliable and should not be used. "
                "Re-run calibration with corrected settings."
            )
        
        if error_issues:
            recommendations.append(
                "ERROR: Significant issues detected. Review calibration setup and "
                "consider adjusting bounds, methods, or model implementation."
            )
        
        # Category-specific recommendations
        categories = {issue.category for issue in issues}
        
        if ValidationCategory.CONVERGENCE in categories:
            recommendations.append(
                "Consider adjusting convergence criteria, iteration limits, or "
                "trying different optimization algorithms."
            )
        
        if ValidationCategory.PHYSICAL in categories:
            recommendations.append(
                "Review parameter bounds and physical constraints to ensure "
                "realistic parameter values."
            )
        
        if ValidationCategory.STATISTICAL in categories:
            recommendations.append(
                "Poor statistical performance may indicate need for model "
                "structure improvements or additional calibration parameters."
            )
        
        if ValidationCategory.UNCERTAINTY in categories:
            recommendations.append(
                "High parameter uncertainty suggests parameters may be poorly "
                "constrained by available observations."
            )
        
        # Performance-based recommendations use the statistical rule's declared bounds
        if result.metrics:
            if not result.metrics.nash_sutcliffe > self.statistical_rule.min_nash_sutcliffe:
                recommendations.append(
                    "Low Nash-Sutcliffe efficiency suggests fundamental model "
                    "limitations. Consider model reformulation."
                )

            if not abs(result.metrics.percent_bias) < self.statistical_rule.max_bias_percent:
                recommendations.append(
                    "High bias indicates systematic model errors. Check for "
                    "missing processes or parameter biases."
                )
        
        return recommendations
    
    def validate_cross_validation(self, 
                                 training_result: CalibrationResult,
                                 validation_results: List[CalibrationResult]) -> CalibrationValidationResult:
        """
        Validate cross-validation results
        
        Parameters:
            training_result: Result from training dataset
            validation_results: Results from validation datasets
            
        Returns:
            Cross-validation assessment
        """
        issues = []
        
        if not validation_results:
            issues.append(CalibrationValidationIssue(
                category=ValidationCategory.STATISTICAL,
                severity=ValidationSeverity.WARNING,
                parameter=None,
                message="No cross-validation results provided",
                recommendation="Perform cross-validation to assess model generalization"
            ))
            return CalibrationValidationResult(is_valid=True, issues=issues)
        
        # Compare training vs validation performance
        if training_result.metrics and all(vr.metrics for vr in validation_results):
            train_nse = training_result.metrics.nash_sutcliffe
            val_nse_values = [vr.metrics.nash_sutcliffe for vr in validation_results]
            mean_val_nse = np.mean(val_nse_values)
            std_val_nse = np.std(val_nse_values)
            
            # Check for overfitting
            performance_drop = train_nse - mean_val_nse
            if performance_drop > self.cross_validation_max_nse_drop:
                issues.append(CalibrationValidationIssue(
                    category=ValidationCategory.STATISTICAL,
                    severity=ValidationSeverity.WARNING,
                    parameter=None,
                    message=f"Possible overfitting: training NSE ({train_nse:.3f}) >> "
                           f"validation NSE ({mean_val_nse:.3f})",
                    details={
                        'training_nse': train_nse,
                        'validation_nse_mean': mean_val_nse,
                        'validation_nse_std': std_val_nse,
                        'performance_drop': performance_drop
                    },
                    recommendation="Consider regularization or reducing model complexity"
                ))
            
            # Check validation consistency
            if std_val_nse > self.cross_validation_max_nse_std:
                issues.append(CalibrationValidationIssue(
                    category=ValidationCategory.STATISTICAL,
                    severity=ValidationSeverity.WARNING,
                    parameter=None,
                    message=f"High variability in validation performance (std: {std_val_nse:.3f})",
                    recommendation="Model performance may be sensitive to dataset characteristics"
                ))
        
        # Check parameter consistency across validation runs
        param_variations = {}
        for param in training_result.target_parameters:
            if param in training_result.optimized_parameters:
                train_value = training_result.optimized_parameters[param]
                val_values = [
                    vr.optimized_parameters.get(param, train_value) 
                    for vr in validation_results
                    if param in vr.optimized_parameters
                ]
                
                if val_values:
                    param_std = np.std(val_values)
                    param_mean = np.mean(val_values)
                    cv = param_std / abs(param_mean) if param_mean != 0 else float('inf')
                    
                    param_variations[param] = {
                        'training_value': train_value,
                        'validation_mean': param_mean,
                        'validation_std': param_std,
                        'coefficient_of_variation': cv
                    }
                    
                    if cv > self.cross_validation_max_parameter_cv:
                        issues.append(CalibrationValidationIssue(
                            category=ValidationCategory.UNCERTAINTY,
                            severity=ValidationSeverity.WARNING,
                            parameter=param,
                            message=f"High parameter variation across validation runs (CV: {cv:.2f})",
                            recommendation="Parameter may be poorly constrained"
                        ))
        
        summary = {
            'cross_validation_runs': len(validation_results),
            'parameter_variations': param_variations
        }
        
        is_valid = not any(
            issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
            for issue in issues
        )
        
        return CalibrationValidationResult(
            is_valid=is_valid,
            issues=issues,
            summary=summary,
            recommendations=self._generate_recommendations(issues, training_result)
        )
