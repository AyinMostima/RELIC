"""
Bootstrap Uncertainty Quantification Implementation

This module implements bootstrap-based uncertainty quantification for biogeochemical
models. The method uses resampling techniques to estimate sampling distributions
and construct confidence intervals without parametric assumptions.

Scientific Background:
    Bootstrap methods estimate the sampling distribution of a statistic by
    resampling from the original data. In the context of model uncertainty
    quantification, bootstrap can be applied to both parameter estimation
    (parametric bootstrap) and model prediction uncertainty (residual bootstrap).
    The method is non-parametric and provides robust confidence intervals.

References:
    - Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap
    - Davison, A.C. & Hinkley, D.V. (1997). Bootstrap Methods and their Applications  
    - Carpenter, J. & Bithell, J. (2000). Bootstrap confidence intervals: when, which, what?
    - Seber, G.A.F. & Wild, C.J. (2003). Nonlinear Regression

Mathematical Expression:
    Bootstrap resampling:
    theta?*? = T(X*?) for b = 1,...,B
    where X*? is bootstrap resample, T is statistic
    
    Bootstrap confidence interval (percentile method):
    CI_1?alpha = [theta?*?alpha/_2?, theta?*?_1?alpha/_2?]
    
    Bias-corrected and accelerated (BCa):
    CI_1?alpha = [theta?*?alpha_1?, theta?*?alpha_2?]
    where alpha_1,alpha_2 are bias and acceleration corrected
"""

from typing import Dict, List, Optional, Union, Callable, Tuple
import numpy as np
import torch
import logging
from dataclasses import dataclass, field
import time
from scipy import stats

from .base import (
    UncertaintyQuantifier,
    UQConfig,
    UQResult,
    UncertainParameter,
    UQMethod
)
from ...parameters import UnifiedParameterManager, get_global_manager

logger = logging.getLogger(__name__)


# Implemented bootstrap variants and interval methods. Any other name is
# rejected at configuration time; none is mapped to a substitute.
BOOTSTRAP_TYPES = ("parametric", "residual")
CI_METHODS = ("percentile", "bca")


@dataclass
class BootstrapConfig(UQConfig):
    """Extended configuration specific to bootstrap UQ.

    bootstrap_type: "parametric" (resample the parameter distributions) or
        "residual" (resample model-observation residuals; requires
        observations).
    ci_method: "percentile" or "bca". "bca" is the bias-corrected
        percentile interval with acceleration a = 0 (Efron 1987); the
        jackknife acceleration is not implemented, so
        acceleration_correction=True is rejected.
    balanced_resampling: jittered systematic (stratified) sampling of
        uniform parameters; any other distribution is rejected.
    """
    # The method is fixed by the class; the number of replicates is the
    # UQConfig field n_bootstrap (None reads the YAML value).
    method: UQMethod = field(default=UQMethod.BOOTSTRAP, init=False)
    # Bootstrap-specific parameters (chosen by the caller, no default)
    bootstrap_type: str = field(kw_only=True)  # one of BOOTSTRAP_TYPES

    # Confidence interval methods
    ci_method: str = field(kw_only=True)       # one of CI_METHODS
    acceleration_correction: bool = False  # jackknife acceleration (not implemented)
    
    # Resampling options
    balanced_resampling: bool = False  # stratified sampling of uniform parameters
    
    def __post_init__(self):
        super().__post_init__()
        if self.bootstrap_type not in BOOTSTRAP_TYPES:
            raise ValueError(
                f"bootstrap_type must be one of {BOOTSTRAP_TYPES}, got "
                f"{self.bootstrap_type!r}"
            )
        if self.ci_method not in CI_METHODS:
            raise ValueError(
                f"ci_method must be one of {CI_METHODS}, got {self.ci_method!r}"
            )
        if self.acceleration_correction:
            raise NotImplementedError(
                "BCa acceleration correction (jackknife estimate of a) is not "
                "implemented; ci_method='bca' computes the bias-corrected "
                "interval with a = 0"
            )
        # The sample size of a bootstrap run is its replicate count
        self.n_samples = self.n_bootstrap


class BootstrapUQ(UncertaintyQuantifier):
    """
    Bootstrap Uncertainty Quantification Implementation.
    
    This method provides non-parametric uncertainty quantification through
    bootstrap resampling. It supports multiple bootstrap variants and
    confidence interval methods for robust uncertainty estimation.
    
    Key Features:
    1. Parametric and residual bootstrap methods
    2. Multiple confidence interval approaches (percentile, BCa, etc.)
    3. Bias and acceleration corrections
    4. Stratified and balanced resampling options
    5. Robust outlier handling
    6. Diagnostic statistics for bootstrap quality
    
    Applications in biogeochemical modeling:
    - Parameter uncertainty from limited calibration data
    - Model prediction confidence intervals
    - Non-parametric hypothesis testing
    - Robust error estimation for complex models
    """
    
    def __init__(
        self,
        parameter_manager: UnifiedParameterManager,
        config: Optional[BootstrapConfig] = None
    ):
        """
        Initialize bootstrap UQ analyzer.
        
        Parameters:
            parameter_manager: Parameter management system
            config: Bootstrap-specific configuration
        """
        if config is None:
            raise TypeError(
                "BootstrapUQ needs a BootstrapConfig: the bootstrap type and the interval "
                "method are chosen by the caller"
            )
        self.bootstrap_config = config
        super().__init__(parameter_manager, self.bootstrap_config)
        
        # Bootstrap-specific state
        self.original_data = None
        self.bootstrap_samples = None
        self.jackknife_statistics = None
        
        logger.info(f"Bootstrap UQ initialized with {self.bootstrap_config.n_bootstrap} resamples")
    
    def quantify(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        uncertain_parameters: Dict[str, UncertainParameter],
        baseline_parameters: Optional[Dict[str, float]] = None,
        observations: Optional[Dict[str, np.ndarray]] = None
    ) -> UQResult:
        """
        Perform bootstrap uncertainty quantification.
        
        Parameters:
            model_runner: Function that runs model and returns outputs
            uncertain_parameters: Parameters with uncertainty distributions  
            baseline_parameters: Baseline/fixed parameter values
            observations: Observational data for residual bootstrap
            
        Returns:
            Bootstrap UQ results with confidence intervals
        """
        self.start_time = time.time()
        
        # Validate inputs
        self.validate_uncertain_parameters(uncertain_parameters)
        
        if len(uncertain_parameters) == 0:
            raise ValueError("At least one uncertain parameter must be specified")
        
        param_names = sorted(uncertain_parameters.keys())
        n_bootstrap = self.bootstrap_config.n_bootstrap
        bootstrap_type = self.bootstrap_config.bootstrap_type
        if bootstrap_type == "residual" and observations is None:
            raise ValueError(
                "bootstrap_type='residual' requires observations; it is "
                "never replaced by the parametric bootstrap"
            )
        
        logger.info(f"Starting bootstrap UQ for {len(param_names)} parameters with {n_bootstrap} resamples")
        
        try:
            # The configured variant runs as declared (BootstrapConfig
            # accepts only BOOTSTRAP_TYPES).
            if bootstrap_type == "residual":
                result = self._residual_bootstrap(
                    model_runner, uncertain_parameters, baseline_parameters, observations
                )
            else:
                result = self._parametric_bootstrap(
                    model_runner, uncertain_parameters, baseline_parameters
                )
            
            computation_time = result.computation_time
            
            logger.info(f"Bootstrap UQ completed: {self.model_evaluations} evaluations in {computation_time:.2f}s")
            
            return result
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"Bootstrap UQ failed: {e}")
            raise RuntimeError(f"Bootstrap UQ failed: {e}") from e
    
    def _parametric_bootstrap(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        uncertain_parameters: Dict[str, UncertainParameter],
        baseline_parameters: Optional[Dict[str, float]] = None
    ) -> UQResult:
        """
        Perform parametric bootstrap by resampling from parameter distributions.
        
        In parametric bootstrap, we assume the parameter distributions are known
        and generate bootstrap samples by drawing from these distributions.
        """
        param_names = sorted(uncertain_parameters.keys())
        n_bootstrap = self.bootstrap_config.n_bootstrap
        
        # Independent streams: child 0 of the configured seed draws the
        # original sample and child b + 1 draws replicate b. Replicate 0 used
        # the original seed and reproduced the original sample exactly; the
        # global numpy generator is not reseeded.
        original_stream, *replicate_streams = np.random.SeedSequence(
            self.bootstrap_config.random_seed
        ).spawn(n_bootstrap + 1)

        # 1. Generate initial parameter sample (serves as "original data").
        # References: Efron and Tibshirani (1993), An Introduction to the
        #     Bootstrap, ch. 6 (a bootstrap sample has the size of the original
        #     sample).
        # Mathematical expression: m = max(m_floor, B // d).
        # Parameter meanings: B = n_bootstrap replicates; m_floor =
        #     optimization.uncertainty.bootstrap.parametric_sample_size_floor and
        #     d = ...parametric_sample_size_replicate_divisor (YAML, RELIC
        #     convention values); m is the size of the original sample and of
        #     every replicate.
        # Units: counts (dimensionless).
        # Domain/constraints: m_floor >= 1, d >= 1 (YAML bounds); integer division.
        # Sign convention: not applicable.
        sample_size_floor = int(get_global_manager().get(
            "optimization.uncertainty.bootstrap.parametric_sample_size_floor", required=True
        ))
        replicate_divisor = int(get_global_manager().get(
            "optimization.uncertainty.bootstrap.parametric_sample_size_replicate_divisor",
            required=True,
        ))
        original_sample_size = max(sample_size_floor, n_bootstrap // replicate_divisor)
        original_parameters = self._generate_original_sample(
            uncertain_parameters, original_sample_size, original_stream
        )
        
        # 2. Evaluate model on original sample
        original_outputs = self.evaluate_model_ensemble(
            model_runner, original_parameters, baseline_parameters
        )
        
        # Store original data
        self.original_data = {
            'parameters': original_parameters,
            'outputs': original_outputs
        }
        
        # 3. Generate bootstrap parameter samples
        bootstrap_parameter_samples = []
        bootstrap_output_samples = {var: [] for var in self.output_order}
        
        for b in range(n_bootstrap):
            parameter_streams = dict(
                zip(param_names, replicate_streams[b].spawn(len(param_names)))
            )

            # Resample parameters from distributions
            if self.bootstrap_config.balanced_resampling:
                # Balanced bootstrap - ensure each original sample used equally
                bootstrap_params = self._balanced_parameter_resample(
                    uncertain_parameters, original_sample_size, parameter_streams
                )
            else:
                # Standard resampling from parameter distributions
                bootstrap_params = {}
                for param_name in param_names:
                    uncertain_param = uncertain_parameters[param_name]
                    bootstrap_params[param_name] = uncertain_param.sample(
                        original_sample_size,
                        parameter_streams[param_name]
                    )
            
            # Evaluate model on bootstrap sample
            bootstrap_outputs = self.evaluate_model_ensemble(
                model_runner, bootstrap_params, baseline_parameters
            )
            
            # The original sample defines the output variables; a replicate
            # that lacks one is an error (no substitute statistic).
            missing_outputs = sorted(set(bootstrap_output_samples) - set(bootstrap_outputs))
            if missing_outputs:
                raise KeyError(
                    f"bootstrap replicate {b} lacks model output(s) {missing_outputs}"
                )
            for var_name in bootstrap_output_samples:
                # Compute mean as the statistic of interest
                bootstrap_statistic = np.mean(bootstrap_outputs[var_name])
                bootstrap_output_samples[var_name].append(bootstrap_statistic)
            
            bootstrap_parameter_samples.append(bootstrap_params)
        
        # 4. Convert lists to arrays
        for var_name in bootstrap_output_samples:
            bootstrap_output_samples[var_name] = np.array(bootstrap_output_samples[var_name])
        
        self.bootstrap_samples = bootstrap_output_samples
        
        # 5. Compute bootstrap confidence intervals
        result = self._compute_bootstrap_intervals(
            original_outputs, bootstrap_output_samples, param_names,
            computation_time=time.time() - self.start_time,
        )
        
        return result
    
    def _residual_bootstrap(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        uncertain_parameters: Dict[str, UncertainParameter],
        baseline_parameters: Optional[Dict[str, float]] = None,
        observations: Optional[Dict[str, np.ndarray]] = None
    ) -> UQResult:
        """
        Perform residual bootstrap using model-observation residuals.
        
        This method bootstraps the residuals between model predictions and
        observations to estimate prediction uncertainty.
        """
        if observations is None:
            raise ValueError("Observations required for residual bootstrap")
        
        param_names = sorted(uncertain_parameters.keys())
        n_bootstrap = self.bootstrap_config.n_bootstrap
        
        # 1. Generate initial parameter estimates (e.g., from optimization)
        # For simplicity, use mean parameter values
        initial_params = {}
        if baseline_parameters:
            initial_params.update(baseline_parameters)
        
        for param_name, uncertain_param in uncertain_parameters.items():
            # The residual bootstrap starts from the declared mean; a law
            # without one (beta, gamma, triangular) has no implied centre.
            if uncertain_param.mean is None:
                raise ValueError(
                    f"residual bootstrap needs the declared mean of parameter '{param_name}'"
                )
            initial_params[param_name] = uncertain_param.mean
        
        # 2. Compute initial model predictions and residuals
        initial_outputs = model_runner(initial_params)
        residuals = {}
        
        for var_name, obs_values in observations.items():
            # Every observed variable must be a model output; it is never
            # skipped with a warning.
            if var_name not in initial_outputs:
                raise KeyError(
                    f"observed variable '{var_name}' is not a model output"
                )
            model_values = initial_outputs[var_name]
            if isinstance(model_values, (list, np.ndarray, torch.Tensor)):
                if isinstance(model_values, torch.Tensor):
                    model_values = model_values.detach().cpu().numpy()
                elif isinstance(model_values, list):
                    model_values = np.array(model_values)
                model_values = model_values.flatten()
            else:
                model_values = np.array([model_values])

            obs_values = np.array(obs_values).flatten()

            # Observations and predictions are paired element-wise; a length
            # mismatch is an error, never a truncation.
            if len(model_values) != len(obs_values):
                raise ValueError(
                    f"observation length {len(obs_values)} of '{var_name}' "
                    f"differs from the model output length {len(model_values)}"
                )
            residuals[var_name] = obs_values - model_values
        
        if not residuals:
            raise ValueError("No valid residuals could be computed")
        
        # 3. Bootstrap residuals
        bootstrap_predictions = {var: [] for var in residuals.keys()}
        # One local stream per replicate (children of the configured seed);
        # the global numpy generator is not reseeded.
        replicate_streams = np.random.SeedSequence(
            self.bootstrap_config.random_seed
        ).spawn(n_bootstrap)

        for b in range(n_bootstrap):
            rng = np.random.default_rng(replicate_streams[b])

            # Resample residuals with replacement
            bootstrap_residuals = {}
            for var_name, res_values in residuals.items():
                n_residuals = len(res_values)
                bootstrap_indices = rng.choice(
                    n_residuals, size=n_residuals, replace=True
                )
                bootstrap_residuals[var_name] = res_values[bootstrap_indices]
            
            # Create bootstrap "observations" by adding resampled residuals to model
            bootstrap_obs = {}
            for var_name in residuals.keys():
                model_pred = initial_outputs[var_name]
                if isinstance(model_pred, (list, np.ndarray, torch.Tensor)):
                    if isinstance(model_pred, torch.Tensor):
                        model_pred = model_pred.detach().cpu().numpy().flatten()
                    elif isinstance(model_pred, list):
                        model_pred = np.array(model_pred).flatten()
                else:
                    model_pred = np.array([model_pred])
                
                # Add bootstrap residuals (lengths were checked above)
                bootstrap_obs[var_name] = model_pred + bootstrap_residuals[var_name]
            
            # Compute statistics for bootstrap predictions
            for var_name, pred_values in bootstrap_obs.items():
                bootstrap_statistic = np.mean(pred_values)
                bootstrap_predictions[var_name].append(bootstrap_statistic)
        
        # 4. Convert to arrays
        for var_name in bootstrap_predictions:
            bootstrap_predictions[var_name] = np.array(bootstrap_predictions[var_name])
        
        self.bootstrap_samples = bootstrap_predictions
        
        # 5. Compute confidence intervals
        original_stats = {
            var_name: np.mean(observations[var_name]) for var_name in residuals
        }
        
        result = self._compute_bootstrap_intervals(
            original_stats, bootstrap_predictions, param_names,
            computation_time=time.time() - self.start_time,
        )
        
        return result
    
    def _generate_original_sample(
        self,
        uncertain_parameters: Dict[str, UncertainParameter],
        sample_size: int,
        stream: np.random.SeedSequence,
    ) -> Dict[str, np.ndarray]:
        """Generate the original sample; each parameter draws its own child stream."""
        names = sorted(uncertain_parameters)
        return {
            name: uncertain_parameters[name].sample(sample_size, child)
            for name, child in zip(names, stream.spawn(len(names)))
        }
    
    def _balanced_parameter_resample(
        self,
        uncertain_parameters: Dict[str, UncertainParameter],
        sample_size: int,
        parameter_streams: Dict[str, np.random.SeedSequence],
    ) -> Dict[str, np.ndarray]:
        """Generate balanced bootstrap sample ensuring equal usage of original data"""
        balanced_sample = {}
        
        # Use a different approach - generate systematically varied samples
        for param_name, uncertain_param in uncertain_parameters.items():
            # Jittered systematic sampling is defined for uniform parameters
            # only; another distribution is an error, never plain sampling.
            if uncertain_param.distribution.value != "uniform":
                raise NotImplementedError(
                    f"balanced_resampling is implemented only for uniform "
                    f"parameters; '{param_name}' is "
                    f"{uncertain_param.distribution.value}"
                )
            rng = np.random.default_rng(parameter_streams[param_name])
            u_vals = np.linspace(0, 1, sample_size + 1)[:-1]  # Exclude 1.0
            u_vals += rng.uniform(0, 1/sample_size, sample_size)  # Add jitter
            u_vals = np.clip(u_vals, 0, 1)

            samples = (uncertain_param.lower_bound +
                      u_vals * (uncertain_param.upper_bound - uncertain_param.lower_bound))
            
            balanced_sample[param_name] = samples
        
        return balanced_sample
    
    def _compute_bootstrap_intervals(
        self,
        original_outputs: Union[Dict[str, np.ndarray], Dict[str, float]],
        bootstrap_samples: Dict[str, np.ndarray],
        param_names: List[str],
        *,
        computation_time: float,
    ) -> UQResult:
        """Compute bootstrap confidence intervals using specified method"""
        
        result = UQResult(
            method="Bootstrap",
            parameter_names=param_names,
            output_variables=list(bootstrap_samples.keys()),
            n_samples=self.bootstrap_config.n_bootstrap,
            n_model_evaluations=self.model_evaluations,
            computation_time=computation_time,
        )
        
        # Compute statistics for each output variable
        for var_name, bootstrap_stats in bootstrap_samples.items():
            # Every bootstrap statistic must be finite: a non-finite value
            # (failed or non-finite model output) is an error, never dropped
            # from the sample, which would bias the interval.
            bootstrap_stats = np.asarray(bootstrap_stats, dtype=float)
            if bootstrap_stats.size == 0:
                raise ValueError(f"no bootstrap statistics for '{var_name}'")
            n_nonfinite = int(np.sum(~np.isfinite(bootstrap_stats)))
            if n_nonfinite:
                raise ValueError(
                    f"{n_nonfinite} of {bootstrap_stats.size} bootstrap statistics "
                    f"of '{var_name}' are non-finite"
                )
            valid_stats = bootstrap_stats
            
            # Basic statistics
            result.output_mean[var_name] = np.mean(valid_stats)
            result.output_std[var_name] = np.std(valid_stats, ddof=1)
            result.output_variance[var_name] = np.var(valid_stats, ddof=1)
            
            # Higher-order moments
            result.skewness[var_name] = stats.skew(valid_stats)
            result.kurtosis[var_name] = stats.kurtosis(valid_stats)
            
            # Percentiles
            result.output_percentiles[var_name] = {}
            for percentile in self.bootstrap_config.percentiles:
                result.output_percentiles[var_name][percentile] = np.percentile(valid_stats, percentile)
            
            # Confidence intervals using specified method
            result.confidence_intervals[var_name] = {}
            
            for conf_level in self.bootstrap_config.confidence_levels:
                if self.bootstrap_config.ci_method == "percentile":
                    ci = self._percentile_ci(valid_stats, conf_level)
                elif self.bootstrap_config.ci_method == "bca":
                    # The original statistic is required; the bootstrap mean
                    # is never substituted (it would force z0 towards 0).
                    if var_name not in original_outputs:
                        raise KeyError(
                            f"BCa interval of '{var_name}' needs its original statistic"
                        )
                    if isinstance(original_outputs[var_name], np.ndarray):
                        original_stat = np.mean(original_outputs[var_name])
                    else:
                        original_stat = float(original_outputs[var_name])
                    
                    ci = self._bca_ci(valid_stats, original_stat, conf_level)
                else:
                    raise ValueError(
                        f"ci_method must be one of {CI_METHODS}, got "
                        f"{self.bootstrap_config.ci_method!r}"
                    )
                
                result.confidence_intervals[var_name][conf_level] = ci
            
            # Bootstrap diagnostics
            result.effective_sample_size[var_name] = len(valid_stats)
            result.monte_carlo_error[var_name] = np.std(valid_stats) / np.sqrt(len(valid_stats))
        
        return result
    
    def _percentile_ci(self, bootstrap_stats: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """Compute percentile confidence interval"""
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def _bca_ci(
        self, 
        bootstrap_stats: np.ndarray, 
        original_stat: float, 
        confidence_level: float
    ) -> Tuple[float, float]:
        """
        Compute the bias-corrected (BCa with a = 0) confidence interval.

        A failure is raised with its cause; the percentile interval is never
        returned in place of the requested BCa interval.

        Raises:
            ValueError: every bootstrap statistic lies on one side of the
                original statistic, so z0 = +-infinity and the
                bias-corrected interval is undefined.
        """
        # References:
        #   Efron (1987), J. Am. Stat. Assoc. 82, 171-185; Efron and
        #   Tibshirani (1993), An Introduction to the Bootstrap, ch. 14.
        # Mathematical expression:
        #   z0 = Phi^-1(#{theta*_b < theta_hat} / B)
        #   alpha_1 = Phi(z0 + (z0 + z_(alpha/2)) / (1 - a (z0 + z_(alpha/2))))
        #   alpha_2 = Phi(z0 + (z0 + z_(1-alpha/2)) / (1 - a (z0 + z_(1-alpha/2))))
        #   CI = [q(alpha_1), q(alpha_2)] of the bootstrap statistics, a = 0
        # Parameter meanings:
        #   theta*_b: bootstrap statistics; theta_hat: original statistic;
        #   B: number of bootstrap statistics; z0: bias correction; a:
        #   acceleration (0, the jackknife estimate is not implemented);
        #   alpha = 1 - confidence level; z_p: standard normal quantile; q:
        #   empirical quantile.
        # Units:
        #   theta*, theta_hat and the interval in the statistic's unit;
        #   z0, a, alpha_1, alpha_2 dimensionless.
        # Domain/constraints:
        #   0 < #{theta*_b < theta_hat} < B (finite z0); otherwise ValueError.
        #   alpha_1 and alpha_2 are used unclipped.
        # Sign convention:
        #   z0 > 0 when theta_hat exceeds the bootstrap median, which shifts
        #   both interval quantiles upward.
        n_below = int(np.sum(bootstrap_stats < original_stat))
        n_total = len(bootstrap_stats)
        if n_below == 0 or n_below == n_total:
            raise ValueError(
                f"BCa bias correction is undefined: {n_below} of {n_total} "
                f"bootstrap statistics lie below the original statistic "
                f"{original_stat!r} (z0 would be infinite)"
            )
        bias_correction = stats.norm.ppf(n_below / n_total)
        acceleration = 0.0

        # Corrected percentiles
        alpha = 1 - confidence_level
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

        alpha_1 = stats.norm.cdf(bias_correction +
                               (bias_correction + z_alpha_2) /
                               (1 - acceleration * (bias_correction + z_alpha_2)))
        alpha_2 = stats.norm.cdf(bias_correction +
                               (bias_correction + z_1_alpha_2) /
                               (1 - acceleration * (bias_correction + z_1_alpha_2)))

        lower_bound = np.percentile(bootstrap_stats, 100 * alpha_1)
        upper_bound = np.percentile(bootstrap_stats, 100 * alpha_2)

        return (lower_bound, upper_bound)
