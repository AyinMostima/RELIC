"""
Uncertainty Quantification Base Classes and Interfaces

This module provides the fundamental abstractions for uncertainty quantification
in the RELIC optimization framework. It defines interface contracts and common
functionality for probabilistic analysis of model predictions.

Scientific Background:
    Uncertainty quantification (UQ) aims to characterize and propagate uncertainties
    in model inputs through to model outputs. Methods include ensemble approaches,
    Bayesian inference, and statistical resampling techniques. UQ is essential for
    understanding model reliability and supporting decision-making under uncertainty.

References:
    - Smith, R.C. (2014). Uncertainty Quantification: Theory, Implementation, and Applications
    - Xiu, D. (2010). Numerical Methods for Stochastic Computations
    - Ghanem, R.G. et al. (2017). Handbook of Uncertainty Quantification  
    - Kennedy, M.C. & O'Hagan, A. (2001). Bayesian calibration of computer models

Mathematical Expression:
    Forward UQ: Given X ~ p(x), find Y = f(X) and characterize p(y)
    Inverse UQ: Given observations y_obs, find posterior p(x|y_obs)
    
    Moments of output distribution:
    E[Y] = integral f(x) p(x) dx
    Var[Y] = integral [f(x) - E[Y]]^2 p(x) dx
    
    Credible intervals and prediction bounds for uncertainty characterization
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import logging
from scipy.stats import beta as scipy_beta
from scipy.stats import gamma as scipy_gamma

from relic.parameters import UnifiedParameterManager, get_global_manager

logger = logging.getLogger(__name__)


def _finite_scalar(value: Any, label: str, *, positive: bool = False) -> torch.Tensor:
    """Return one finite float64 scalar or raise."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite scalar")
    scalar = torch.as_tensor(value, dtype=torch.float64)
    if scalar.numel() != 1 or not bool(torch.isfinite(scalar).item()):
        raise ValueError(f"{label} must be a finite scalar")
    if positive and not bool((scalar > 0.0).item()):
        raise ValueError(f"{label} must be a positive finite scalar")
    return scalar.reshape(())


def _finite_or_infinite_scalar(value: Any, label: str) -> torch.Tensor:
    """Return one float64 bound (finite or infinite, never NaN) or raise."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a real scalar")
    scalar = torch.as_tensor(value, dtype=torch.float64)
    if scalar.numel() != 1 or bool(torch.isnan(scalar).item()):
        raise ValueError(f"{label} must be a real scalar")
    return scalar.reshape(())


class UQMethod(Enum):
    """Available uncertainty quantification methods"""
    ENSEMBLE = "ensemble"
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"
    POLYNOMIAL_CHAOS = "polynomial_chaos"
    MONTE_CARLO = "monte_carlo"


class DistributionType(Enum):
    """Parameter distribution types for UQ"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    GAMMA = "gamma"
    TRIANGULAR = "triangular"


@dataclass
class UncertainParameter:
    """
    Uncertain parameter definition for UQ analysis.
    
    Defines the probabilistic characteristics of a parameter including
    distribution type, moments, and physical constraints.
    """
    name: str
    distribution: DistributionType
    # Distribution parameters: a NORMAL or LOGNORMAL parameter declares mean
    # and std; a UNIFORM parameter derives them from its bounds.
    mean: Optional[float] = None
    std: Optional[float] = None
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    # Additional distribution parameters
    alpha: Optional[float] = None  # Beta/Gamma shape parameter
    beta: Optional[float] = None   # Beta/Gamma scale parameter
    mode: Optional[float] = None   # Triangular mode
    # Constraints and metadata
    integer_valued: bool = False
    log_transform: bool = False
    # Declared units are compared with the ParameterSpec units of the
    # parameter key of the same name (validate_uncertain_parameters).
    units: str = ""
    description: str = ""
    
    def __post_init__(self):
        # Validate bounds
        if self.lower_bound >= self.upper_bound:
            raise ValueError(f"Invalid bounds for parameter {self.name}")
        
        # References: Johnson, Kotz and Balakrishnan (1995), Continuous
        #     Univariate Distributions, Vol. 2, ch. 26 (uniform law moments).
        # Mathematical expression: U(a, b): mean = (a + b) / 2,
        #     std = (b - a) / sqrt(12).
        # Parameter meanings: a, b lower and upper bounds.
        # Units: mean, std, a and b in the parameter units.
        # Domain/constraints: derived only when neither mean nor std is
        #     declared; a NORMAL or LOGNORMAL law has no default moments.
        # Sign convention: not applicable.
        if self.distribution == DistributionType.UNIFORM:
            if self.mean is None and self.std is None:
                self.mean = (self.lower_bound + self.upper_bound) / 2
                self.std = (self.upper_bound - self.lower_bound) / np.sqrt(12)

        elif self.distribution in (DistributionType.NORMAL, DistributionType.LOGNORMAL):
            if self.mean is None or self.std is None:
                raise ValueError(
                    f"{self.distribution.value} parameter '{self.name}' needs explicit "
                    "mean and std"
                )
        
        elif self.distribution == DistributionType.TRIANGULAR:
            # The mode is a defining parameter of the triangular law; the
            # mean is not substituted for it.
            if self.mode is None:
                raise ValueError(
                    f"Triangular parameter '{self.name}' requires an explicit mode"
                )
    
    def sample(self, n_samples: int, random_state: Any = None) -> np.ndarray:
        """Draw samples of the declared (bounded) distribution.

        random_state is None, an integer seed, a numpy SeedSequence or a numpy
        Generator; the draws use that local stream and never reseed the global
        numpy generator. The unit draws are mapped by transform_unit_samples,
        the single inverse-CDF owner shared with EnsembleUQ.
        """
        rng = np.random.default_rng(random_state)
        unit = torch.as_tensor(rng.random(n_samples), dtype=torch.float64)
        samples = self.transform_unit_samples(unit).detach().cpu().numpy().copy()
        if self.integer_valued:
            samples = samples.astype(int)
        return samples

    def transform_unit_samples(self, unit_samples: torch.Tensor) -> torch.Tensor:
        """Map unit samples u in [0, 1] to the declared distribution truncated to its bounds.

        References:
            Devroye (1986), Non-Uniform Random Variate Generation, Sect. II.2
            (inversion; truncated distributions by restricting u to
            [F(a), F(b)]); Adams et al., Dakota User's Manual (bounded normal and
            bounded lognormal are truncated distributions).

        Mathematical Expression:
            ``x = F^{-1}(F(a) + u (F(b) - F(a)))`` for normal, lognormal and gamma;
            ``x = a + u (b - a)`` (uniform), ``x = a + F_Beta^{-1}(u) (b - a)``
            (beta on [a, b]); triangular with ``c = (m-a)/(b-a)``:
            ``a + sqrt(u(b-a)(m-a))`` for ``u < c``, else ``b - sqrt((1-u)(b-a)(b-m))``.

        Parameter Meanings:
            ``u`` unit sample, ``F`` the declared CDF (normal mean/std; lognormal
            with arithmetic mean/std converted to log-space mu and sigma; gamma
            shape alpha and scale beta), ``[a, b]`` the declared bounds,
            ``m`` the triangular mode.

        Units:
            ``x``, ``a``, ``b``, mean, std, mode and the gamma scale carry the
            parameter units; ``u``, ``F`` and the shapes are dimensionless.

        Domain/constraints:
            ``u`` must lie in [0, 1]; the bounds must carry positive probability
            mass (``F(b) > F(a)``). Bounds truncate the distribution: clamping
            (the former behaviour) moved the probability outside [a, b] onto the
            bounds, a censored law that was never declared. An infinite bound
            with ``u`` at 0 or 1 gives a non-finite sample, which raises.

        Sign convention:
            Not applicable (sampling transform).
        """
        if not isinstance(self.distribution, DistributionType):
            raise TypeError(f"Parameter '{self.name}' distribution must be DistributionType")
        unit = torch.as_tensor(unit_samples, dtype=torch.float64)
        if unit.dim() != 1 or unit.numel() == 0:
            raise ValueError("unit_samples must be a non-empty one-dimensional tensor")
        if not bool(torch.isfinite(unit).all()) or bool(((unit < 0.0) | (unit > 1.0)).any()):
            raise ValueError("unit_samples must lie in [0, 1]")

        lower = _finite_or_infinite_scalar(self.lower_bound, f"{self.name}.lower_bound")
        upper = _finite_or_infinite_scalar(self.upper_bound, f"{self.name}.upper_bound")
        if not bool((lower < upper).item()):
            raise ValueError(f"Parameter '{self.name}' has invalid bounds")
        finite_bounds = bool(torch.isfinite(lower).item()) and bool(torch.isfinite(upper).item())
        distribution = self.distribution
        standard_normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
        )

        if distribution is DistributionType.UNIFORM:
            if not finite_bounds:
                raise ValueError(f"Uniform parameter '{self.name}' requires finite bounds")
            samples = lower + unit * (upper - lower)
        elif distribution is DistributionType.NORMAL:
            mean = _finite_scalar(self.mean, f"{self.name}.mean")
            std = _finite_scalar(self.std, f"{self.name}.std", positive=True)
            cdf_lower = standard_normal.cdf((lower - mean) / std)
            cdf_upper = standard_normal.cdf((upper - mean) / std)
            probability = self._truncated_probability(unit, cdf_lower, cdf_upper)
            samples = mean + std * standard_normal.icdf(probability)
        elif distribution is DistributionType.LOGNORMAL:
            mean = _finite_scalar(self.mean, f"{self.name}.mean", positive=True)
            std = _finite_scalar(self.std, f"{self.name}.std", positive=True)
            log_std = torch.sqrt(torch.log1p((std / mean).square()))
            log_mean = torch.log(mean) - 0.5 * log_std.square()
            cdf_lower = (
                standard_normal.cdf((torch.log(lower) - log_mean) / log_std)
                if bool((lower > 0.0).item())
                else torch.tensor(0.0, dtype=torch.float64)
            )
            cdf_upper = standard_normal.cdf((torch.log(upper) - log_mean) / log_std)
            probability = self._truncated_probability(unit, cdf_lower, cdf_upper)
            samples = torch.exp(log_mean + log_std * standard_normal.icdf(probability))
        elif distribution is DistributionType.BETA:
            if self.alpha is None or self.beta is None:
                raise ValueError(f"Beta parameter '{self.name}' requires explicit alpha and beta")
            alpha = _finite_scalar(self.alpha, f"{self.name}.alpha", positive=True)
            beta = _finite_scalar(self.beta, f"{self.name}.beta", positive=True)
            if not finite_bounds:
                raise ValueError(f"Beta parameter '{self.name}' requires finite bounds")
            beta_unit = scipy_beta.ppf(unit.numpy(), a=alpha.item(), b=beta.item())
            samples = lower + torch.as_tensor(beta_unit.copy(), dtype=torch.float64) * (upper - lower)
        elif distribution is DistributionType.GAMMA:
            # The shape and scale are required; they are never estimated from
            # mean and std and written back into the declaration.
            if self.alpha is None or self.beta is None:
                raise ValueError(f"Gamma parameter '{self.name}' requires explicit alpha and beta")
            alpha = _finite_scalar(self.alpha, f"{self.name}.alpha", positive=True)
            scale = _finite_scalar(self.beta, f"{self.name}.beta", positive=True)
            cdf_lower = torch.as_tensor(
                float(scipy_gamma.cdf(lower.item(), a=alpha.item(), scale=scale.item())),
                dtype=torch.float64,
            )
            cdf_upper = torch.as_tensor(
                float(scipy_gamma.cdf(upper.item(), a=alpha.item(), scale=scale.item())),
                dtype=torch.float64,
            )
            probability = self._truncated_probability(unit, cdf_lower, cdf_upper)
            gamma_samples = scipy_gamma.ppf(probability.numpy(), a=alpha.item(), scale=scale.item())
            samples = torch.as_tensor(gamma_samples.copy(), dtype=torch.float64)
        elif distribution is DistributionType.TRIANGULAR:
            if self.mode is None:
                raise ValueError(f"Triangular parameter '{self.name}' requires an explicit mode")
            mode = _finite_scalar(self.mode, f"{self.name}.mode")
            if not finite_bounds:
                raise ValueError(f"Triangular parameter '{self.name}' requires finite bounds")
            if not bool(((mode >= lower) & (mode <= upper)).item()):
                raise ValueError(f"Triangular parameter '{self.name}' mode must lie within bounds")
            split = (mode - lower) / (upper - lower)
            left = lower + torch.sqrt(unit * (upper - lower) * (mode - lower))
            right = upper - torch.sqrt((1.0 - unit) * (upper - lower) * (upper - mode))
            samples = torch.where(unit < split, left, right)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        if self.log_transform:
            if not bool((samples > 0.0).all().item()):
                raise ValueError(f"Parameter '{self.name}' log_transform requires positive samples")
            samples = torch.log(samples)
        if self.integer_valued:
            samples = torch.round(samples)
        if not bool(torch.isfinite(samples).all().item()):
            raise ValueError(f"Parameter '{self.name}' generated non-finite samples")
        return samples

    def _truncated_probability(
        self, unit: torch.Tensor, cdf_lower: torch.Tensor, cdf_upper: torch.Tensor
    ) -> torch.Tensor:
        """Probability F(a) + u (F(b) - F(a)); bounds without mass raise."""
        mass = cdf_upper - cdf_lower
        if not bool((mass > 0.0).item()):
            raise ValueError(
                f"Bounds of parameter '{self.name}' carry no probability mass of the "
                "declared distribution"
            )
        return cdf_lower + unit * mass
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate probability density function"""
        # Basic implementation - would be expanded for each distribution
        if self.distribution == DistributionType.UNIFORM:
            return np.where((x >= self.lower_bound) & (x <= self.upper_bound),
                          1.0 / (self.upper_bound - self.lower_bound), 0.0)
        elif self.distribution == DistributionType.NORMAL:
            return np.exp(-0.5 * ((x - self.mean) / self.std)**2) / (self.std * np.sqrt(2 * np.pi))
        else:
            # No constant placeholder density is returned.
            raise NotImplementedError(
                f"pdf is implemented for uniform and normal parameters only, "
                f"not {self.distribution.value}"
            )


@dataclass
class UQConfig:
    """Configuration for uncertainty quantification analysis.

    method is chosen by the caller (keyword-only, no default; EnsembleConfig
    and BootstrapConfig fix it). A sample size or tolerance left at None
    reads optimization.uncertainty.* from the parameter database.
    """
    # Method selection (no default)
    method: UQMethod = field(kw_only=True)

    # Sampling settings (None reads the YAML value)
    n_samples: Optional[int] = None
    n_bootstrap: Optional[int] = None
    random_seed: Optional[int] = 42
    
    # Output settings (None reads optimization.uncertainty.confidence_levels
    # and optimization.uncertainty.percentiles)
    output_variables: List[str] = field(default_factory=list)
    confidence_levels: Optional[List[float]] = None
    percentiles: Optional[List[float]] = None
    
    # Statistical options
    compute_moments: bool = True
    compute_correlations: bool = True
    sensitivity_indices: bool = False
    
    # Computational settings
    parallel_execution: bool = False
    batch_size: int = 50
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    # Convergence and quality control
    convergence_check: bool = True
    convergence_tolerance: Optional[float] = None

    # Failure handling (explicit policy)
    allow_failed_samples: bool = False
    failed_sample_value: float = float("nan")
    allow_missing_outputs: bool = False
    missing_output_value: float = float("nan")

    def __post_init__(self):
        manager = get_global_manager()
        if self.n_samples is None:
            self.n_samples = int(manager.get("optimization.uncertainty.n_samples", required=True))
        if self.n_bootstrap is None:
            self.n_bootstrap = int(manager.get(
                "optimization.uncertainty.bootstrap.n_bootstrap", required=True
            ))
        if self.convergence_tolerance is None:
            self.convergence_tolerance = float(manager.get(
                "optimization.uncertainty.ensemble.convergence_tolerance", required=True
            ))
        if self.confidence_levels is None:
            self.confidence_levels = [
                float(level)
                for level in manager.get("optimization.uncertainty.confidence_levels", required=True)
            ]
        if self.percentiles is None:
            self.percentiles = [
                float(percentile)
                for percentile in manager.get("optimization.uncertainty.percentiles", required=True)
            ]


@dataclass
class UQResult:
    """
    Results from uncertainty quantification analysis.
    
    Contains statistical summaries, distributions, and metadata
    from the UQ analysis process.
    """
    # Method information
    method: str
    parameter_names: List[str]
    output_variables: List[str]
    
    # Central tendencies and spread
    output_mean: Dict[str, float] = field(default_factory=dict)
    output_std: Dict[str, float] = field(default_factory=dict)
    output_variance: Dict[str, float] = field(default_factory=dict)
    
    # Distribution characteristics
    output_percentiles: Dict[str, Dict[float, float]] = field(default_factory=dict)  # output -> percentile -> value
    confidence_intervals: Dict[str, Dict[float, Tuple[float, float]]] = field(default_factory=dict)  # output -> level -> (low, high)
    
    # Full distributions (optional, memory intensive)
    output_samples: Optional[Dict[str, np.ndarray]] = None
    parameter_samples: Optional[Dict[str, np.ndarray]] = None
    
    # Statistical measures
    skewness: Dict[str, float] = field(default_factory=dict)
    kurtosis: Dict[str, float] = field(default_factory=dict)
    
    # Correlations and dependencies
    parameter_correlations: Optional[np.ndarray] = None
    output_correlations: Optional[np.ndarray] = None
    parameter_output_correlations: Optional[Dict[str, Dict[str, float]]] = None
    
    # Computational metadata (stated by the producer; no default)
    n_samples: int = field(kw_only=True)
    n_model_evaluations: int = field(kw_only=True)
    computation_time: float = field(kw_only=True)
    convergence_achieved: bool = False
    
    # Quality metrics and diagnostics
    effective_sample_size: Dict[str, float] = field(default_factory=dict)
    monte_carlo_error: Dict[str, float] = field(default_factory=dict)
    
    def get_credible_interval(
        self, 
        output_variable: str, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Get credible interval for output variable"""
        if (output_variable in self.confidence_intervals and 
            confidence_level in self.confidence_intervals[output_variable]):
            return self.confidence_intervals[output_variable][confidence_level]
        else:
            # Calculate from percentiles if available
            alpha = 1 - confidence_level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            
            if (output_variable in self.output_percentiles and
                lower_percentile in self.output_percentiles[output_variable] and
                upper_percentile in self.output_percentiles[output_variable]):
                
                return (self.output_percentiles[output_variable][lower_percentile],
                       self.output_percentiles[output_variable][upper_percentile])
            
            # No zero-width interval is fabricated for a missing result.
            raise KeyError(
                f"No {confidence_level} interval or matching percentiles are "
                f"available for output '{output_variable}'"
            )
    
    def get_prediction_bounds(
        self, 
        output_variable: str,
        coverage_probability: float = 0.95
    ) -> Tuple[float, float]:
        """Get prediction bounds accounting for both parameter and model uncertainty"""
        # For now, same as credible interval - would be extended for full Bayesian treatment
        return self.get_credible_interval(output_variable, coverage_probability)


class UncertaintyQuantifier(ABC):
    """
    Abstract base class for uncertainty quantification methods.
    
    Defines the common interface for all UQ algorithms including
    ensemble methods, bootstrap, and Bayesian approaches.
    """
    
    def __init__(
        self,
        parameter_manager: UnifiedParameterManager,
        config: UQConfig
    ):
        """
        Initialize base uncertainty quantifier.
        
        Parameters:
            parameter_manager: Parameter management system
            config: UQ analysis configuration
        """
        self.parameter_manager = parameter_manager
        self.config = config
        
        # Setup computational environment
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        
        # Runtime tracking
        self.model_evaluations = 0
        self.start_time = None
        
        # Results storage
        self.samples_generated = 0
        self.output_order = []
        
        logger.info(f"UQ analyzer initialized with method={config.method.value}")
    
    @abstractmethod
    def quantify(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        uncertain_parameters: Dict[str, UncertainParameter],
        baseline_parameters: Optional[Dict[str, float]] = None,
        observations: Optional[Dict[str, np.ndarray]] = None
    ) -> UQResult:
        """
        Perform uncertainty quantification.
        
        Parameters:
            model_runner: Function that runs model and returns outputs
            uncertain_parameters: Parameters with uncertainty distributions
            baseline_parameters: Baseline/fixed parameter values
            observations: Observational data for calibration/validation
            
        Returns:
            Uncertainty quantification results
        """
    
    def validate_uncertain_parameters(self, uncertain_parameters: Dict[str, UncertainParameter]) -> None:
        """Validate uncertain parameters and check for conflicts.

        Declared units are compared with the units of the ParameterSpec of the
        same key; units declared for a name that is not a parameter key cannot
        be checked and are refused.
        """
        for param_name, uncertain_param in uncertain_parameters.items():
            # Check if parameter exists in parameter manager
            known = self.parameter_manager.exists(param_name)
            if not known:
                logger.warning(f"Parameter '{param_name}' not found in parameter manager")
            if uncertain_param.units:
                if not known:
                    raise ValueError(
                        f"Parameter '{param_name}' declares units '{uncertain_param.units}' "
                        "but is not a parameter key, so the units cannot be checked"
                    )
                spec_units = str(self.parameter_manager.get_parameter(param_name).units).strip()
                if uncertain_param.units.strip() != spec_units:
                    raise ValueError(
                        f"Parameter '{param_name}' declares units '{uncertain_param.units}' but "
                        f"the parameter database declares '{spec_units}'"
                    )
            
            # Check distribution validity
            if uncertain_param.lower_bound >= uncertain_param.upper_bound:
                raise ValueError(f"Invalid bounds for parameter '{param_name}'")
            
            # Check physical constraints
            if param_name.endswith('_rate') or param_name.endswith('_fraction'):
                if uncertain_param.lower_bound < 0:
                    logger.warning(f"Parameter '{param_name}' has negative lower bound")
    
    def generate_parameter_samples(
        self,
        uncertain_parameters: Dict[str, UncertainParameter],
        n_samples: int
    ) -> Dict[str, np.ndarray]:
        """
        Generate samples from parameter distributions.
        
        Parameters:
            uncertain_parameters: Parameters with uncertainty distributions
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary of parameter samples
        """
        parameter_samples = {}

        # One child stream of the configured seed per parameter (in sorted
        # name order): passing the same seed to every parameter drew
        # identical unit streams, and the global numpy generator is not reseeded.
        names = sorted(uncertain_parameters)
        streams = np.random.SeedSequence(self.config.random_seed).spawn(len(names))
        for param_name, stream in zip(names, streams):
            parameter_samples[param_name] = uncertain_parameters[param_name].sample(n_samples, stream)
        
        self.samples_generated = n_samples
        logger.debug(f"Generated {n_samples} samples for {len(uncertain_parameters)} parameters")
        
        return parameter_samples
    
    def evaluate_model_ensemble(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        parameter_samples: Dict[str, np.ndarray],
        baseline_parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate model for ensemble of parameter samples.
        
        Parameters:
            model_runner: Model evaluation function
            parameter_samples: Parameter sample arrays
            baseline_parameters: Fixed parameter values
            
        Returns:
            Model output arrays
        """
        param_names = list(parameter_samples.keys())
        n_samples = len(parameter_samples[param_names[0]])
        
        # Combine with baseline parameters
        if baseline_parameters is None:
            baseline_parameters = {}
        
        output_samples = {}
        first_evaluation = True
        
        # Evaluate in batches
        batch_size = self.config.batch_size
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            
            for sample_idx in range(batch_start, batch_end):
                # Create parameter dictionary for this sample
                param_dict = baseline_parameters.copy()
                for param_name in param_names:
                    param_dict[param_name] = parameter_samples[param_name][sample_idx]
                
                try:
                    # Evaluate model
                    model_outputs = model_runner(param_dict)
                    
                    # Initialize output arrays on first evaluation
                    if first_evaluation:
                        if not model_outputs:
                            if self.config.allow_missing_outputs:
                                self.output_order = ['output']
                            else:
                                raise KeyError("Model runner returned no outputs")
                        else:
                            self.output_order = sorted(model_outputs.keys())
                        for var_name in self.output_order:
                            output_samples[var_name] = np.zeros(n_samples)
                        first_evaluation = False
                    
                    # Store outputs
                    for var_name in self.output_order:
                        if var_name not in model_outputs:
                            if self.config.allow_missing_outputs:
                                output_samples[var_name][sample_idx] = float(
                                    self.config.missing_output_value
                                )
                                continue
                            raise KeyError(
                                f"Missing output '{var_name}' from model runner"
                            )
                        value = model_outputs[var_name]
                        if isinstance(value, (list, np.ndarray, torch.Tensor)):
                            if isinstance(value, torch.Tensor):
                                value = value.detach().cpu().numpy()
                            elif isinstance(value, list):
                                value = np.array(value)
                            # Aggregate multi-dimensional outputs
                            aggregated_value = self._aggregate_output(value)
                            output_samples[var_name][sample_idx] = aggregated_value
                        else:
                            output_samples[var_name][sample_idx] = float(value)
                    
                    self.model_evaluations += 1
                    
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                    logger.error(f"Model evaluation failed for sample {sample_idx}: {e}")
                    if not self.config.allow_failed_samples:
                        raise RuntimeError(
                            f"Model evaluation failed for sample {sample_idx}"
                        ) from e
                    if first_evaluation:
                        self.output_order = ['output']
                        output_samples['output'] = np.zeros(n_samples)
                        first_evaluation = False
                    for var_name in output_samples.keys():
                        output_samples[var_name][sample_idx] = float(
                            self.config.failed_sample_value
                        )
        
        logger.debug(f"Evaluated {n_samples} ensemble members")
        return output_samples
    
    def _aggregate_output(self, output_array: np.ndarray) -> float:
        """Aggregate multi-dimensional model output to scalar"""
        # Use mean as default aggregation
        return np.mean(output_array.flatten())
    
    def compute_statistics(
        self,
        output_samples: Dict[str, np.ndarray],
        parameter_samples: Dict[str, np.ndarray],
        *,
        computation_time: float,
    ) -> UQResult:
        """
        Compute statistical summaries from samples.
        
        Parameters:
            output_samples: Model output samples
            parameter_samples: Parameter samples
            
        Returns:
            Statistical summary results
        """
        result = UQResult(
            method=self.config.method.value,
            parameter_names=list(parameter_samples.keys()),
            output_variables=list(output_samples.keys()),
            n_samples=self.samples_generated,
            n_model_evaluations=self.model_evaluations,
            computation_time=computation_time,
        )
        
        # Non-finite outputs are expected only as the declared substitutes of
        # the explicit failure policy (allow_failed_samples or
        # allow_missing_outputs); only then are they left out of the
        # statistics. Otherwise a non-finite model output is an error.
        failure_policy = (
            self.config.allow_failed_samples or self.config.allow_missing_outputs
        )
        for var_name, samples in output_samples.items():
            finite = np.isfinite(samples)
            if not failure_policy and not bool(np.all(finite)):
                raise ValueError(
                    f"{int(np.sum(~finite))} of {len(samples)} samples of output "
                    f"'{var_name}' are non-finite and no failure policy is enabled"
                )
            valid_samples = samples[finite]
            
            if len(valid_samples) == 0:
                raise ValueError(
                    f"output '{var_name}' has no finite sample to summarize"
                )
            
            # Central tendencies and spread
            result.output_mean[var_name] = np.mean(valid_samples)
            result.output_std[var_name] = np.std(valid_samples, ddof=1)
            result.output_variance[var_name] = np.var(valid_samples, ddof=1)
            
            # Higher-order moments
            if self.config.compute_moments:
                from scipy import stats
                result.skewness[var_name] = stats.skew(valid_samples)
                result.kurtosis[var_name] = stats.kurtosis(valid_samples)
            
            # Percentiles
            result.output_percentiles[var_name] = {}
            for percentile in self.config.percentiles:
                result.output_percentiles[var_name][percentile] = np.percentile(valid_samples, percentile)
            
            # Confidence intervals
            result.confidence_intervals[var_name] = {}
            for conf_level in self.config.confidence_levels:
                alpha = 1 - conf_level
                lower_p = 100 * alpha / 2
                upper_p = 100 * (1 - alpha / 2)
                lower_bound = np.percentile(valid_samples, lower_p)
                upper_bound = np.percentile(valid_samples, upper_p)
                result.confidence_intervals[var_name][conf_level] = (lower_bound, upper_bound)
            
            # Monte Carlo error estimate
            result.monte_carlo_error[var_name] = result.output_std[var_name] / np.sqrt(len(valid_samples))
        
        # Correlations if requested
        if self.config.compute_correlations:
            result.parameter_correlations = self._compute_correlations(parameter_samples)
            result.output_correlations = self._compute_correlations(output_samples)
            result.parameter_output_correlations = self._compute_cross_correlations(
                parameter_samples, output_samples
            )
        
        return result
    
    def _compute_correlations(self, samples_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute correlation matrix for samples"""
        var_names = sorted(samples_dict.keys())
        n_vars = len(var_names)

        if n_vars < 2:
            return np.array([[1.0]])

        # Stack samples into matrix
        sample_matrix = np.column_stack([samples_dict[name] for name in var_names])

        # Pearson correlation of the jointly finite pairs. It is undefined
        # (NaN) with fewer than two pairs or a constant series; 0.0 would
        # falsely report 'uncorrelated'.
        corr_matrix = np.eye(n_vars, dtype=float)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                x = sample_matrix[:, i]
                y = sample_matrix[:, j]
                valid_mask = np.isfinite(x) & np.isfinite(y)
                if int(np.sum(valid_mask)) < 2:
                    correlation = float("nan")
                else:
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    x_std = float(np.std(x_valid))
                    y_std = float(np.std(y_valid))
                    if x_std == 0.0 or y_std == 0.0:
                        correlation = float("nan")
                    else:
                        correlation = float(np.corrcoef(x_valid, y_valid)[0, 1])
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation

        return corr_matrix
    
    def _compute_cross_correlations(
        self, 
        parameter_samples: Dict[str, np.ndarray],
        output_samples: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Compute correlations between parameters and outputs"""
        cross_corr = {}
        
        for param_name, param_values in parameter_samples.items():
            cross_corr[param_name] = {}
            for output_name, output_values in output_samples.items():
                # Remove any invalid pairs
                valid_mask = np.isfinite(param_values) & np.isfinite(output_values)
                if np.sum(valid_mask) > 1:
                    valid_param_values = param_values[valid_mask]
                    valid_output_values = output_values[valid_mask]
                    if float(np.std(valid_param_values)) == 0.0 or float(np.std(valid_output_values)) == 0.0:
                        # Undefined for a constant series (not 'uncorrelated').
                        cross_corr[param_name][output_name] = float("nan")
                    else:
                        cross_corr[param_name][output_name] = float(
                            np.corrcoef(valid_param_values, valid_output_values)[0, 1]
                        )
                else:
                    cross_corr[param_name][output_name] = float("nan")

        return cross_corr
