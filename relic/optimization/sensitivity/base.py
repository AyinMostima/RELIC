"""
Sensitivity Analysis Base Classes and Interfaces

This module provides the fundamental abstractions for global sensitivity analysis
in the RELIC optimization framework. It defines interface contracts and common
functionality shared across different sensitivity analysis methods.

Scientific Background:
    Global Sensitivity Analysis (GSA) explores the full parameter space to quantify
    the contribution of each parameter to variations in model outputs. Unlike local
    sensitivity analysis, GSA accounts for parameter interactions and non-linear
    relationships throughout the entire parameter space.

References:
    - Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer
    - Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments
    - Sobol, I.M. (1993). Sensitivity estimates for nonlinear mathematical models
    - Campolongo, F. et al. (2007). An effective screening design for sensitivity analysis

Mathematical Expression:
    Total output variance decomposition:
    V(Y) = sum?V? + sum?<?V?? + ... + V_1_2...?
    
    First-order sensitivity index: S? = V?/V(Y)
    Total-order sensitivity index: S?? = (V(Y) - V??)/V(Y)
    
    Where V? is variance due to parameter i alone,
    V?? is variance due to interaction between i and j,
    and V?? is variance when parameter i is fixed
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, Sequence, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import logging

from relic.parameters import UnifiedParameterManager, get_global_manager

logger = logging.getLogger(__name__)

# Declared choices. A value outside these tuples is rejected instead of being
# replaced by the mean aggregation or an empty ranking.
AGGREGATION_METHODS = ("mean", "sum", "max", "std", "final")
INDEX_TYPES = ("total", "first")
DISTRIBUTIONS = ("uniform", "normal", "lognormal", "triangular")


class SensitivityMethod(Enum):
    """Available sensitivity analysis methods"""
    MORRIS = "morris"
    SOBOL = "sobol"
    FAST = "fast"
    DELTA = "delta"
    LOCAL = "local"


@dataclass
class ParameterRange:
    """
    Parameter range definition for sensitivity analysis.
    
    Defines the sampling space for a single parameter including
    bounds, distribution type, and any constraints.
    """
    name: str
    lower_bound: float
    upper_bound: float
    distribution: str = field(kw_only=True)  # "uniform", "normal", "lognormal", "triangular"
    # Additional distribution parameters
    mean: Optional[float] = None
    std: Optional[float] = None
    # Constraints
    integer_valued: bool = False
    log_scale: bool = False
    # Metadata. Declared units are compared with the ParameterSpec units of
    # the parameter key of the same name by validate_parameter_ranges.
    units: str = ""
    description: str = ""

    def __post_init__(self):
        if self.lower_bound >= self.upper_bound:
            raise ValueError(f"Lower bound must be less than upper bound for parameter {self.name}")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution '{self.distribution}' for parameter {self.name}; "
                f"expected one of {DISTRIBUTIONS}"
            )
        # A non-uniform distribution is described by its own parameters. They
        # are required instead of being derived from the bounds (no implied
        # 3-sigma std, geometric-mean median, log std of 0.5 or midpoint mode).
        if self.distribution in ("normal", "lognormal"):
            if self.mean is None or self.std is None:
                raise ValueError(
                    f"{self.distribution} range for parameter {self.name} needs explicit "
                    "mean and std"
                )
            if not math.isfinite(self.std) or self.std <= 0.0:
                raise ValueError(
                    f"{self.distribution} range for parameter {self.name} needs a finite "
                    f"positive std, got {self.std}"
                )
            if self.distribution == "lognormal" and not self.mean > 0.0:
                raise ValueError(
                    f"lognormal range for parameter {self.name} needs a positive median "
                    f"(mean), got {self.mean}"
                )
        if self.distribution == "triangular":
            if self.mean is None or not self.lower_bound <= self.mean <= self.upper_bound:
                raise ValueError(
                    f"triangular range for parameter {self.name} needs an explicit mode "
                    f"(mean) inside [{self.lower_bound}, {self.upper_bound}], got {self.mean}"
                )
    
    def sample(self, n_samples: int, random_state: Any = None) -> np.ndarray:
        """Generate random samples from parameter distribution.

        random_state is None, an integer seed or a numpy Generator. Draws use
        that local generator; the global numpy random state is not touched.
        normal: N(mean, std); lognormal: exp(N(log(mean), std)) with mean the
        median; triangular: mode = mean. Normal and lognormal draws are
        clipped to the bounds.
        """
        rng = np.random.default_rng(random_state)

        if self.distribution == "uniform":
            samples = rng.uniform(self.lower_bound, self.upper_bound, n_samples)
        elif self.distribution == "normal":
            samples = rng.normal(self.mean, self.std, n_samples)
            samples = np.clip(samples, self.lower_bound, self.upper_bound)
        elif self.distribution == "lognormal":
            samples = rng.lognormal(np.log(self.mean), self.std, n_samples)
            samples = np.clip(samples, self.lower_bound, self.upper_bound)
        else:
            samples = rng.triangular(self.lower_bound, self.mean, self.upper_bound, n_samples)
        
        # Apply log scale if requested
        if self.log_scale:
            samples = np.log10(samples)
            
        # Apply integer constraint if requested
        if self.integer_valued:
            samples = np.round(samples).astype(int)
            
        return samples


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis.

    method and aggregation_method are chosen by the caller (keyword-only,
    no default; MorrisConfig and SobolConfig fix their method). The sample
    settings belong to the method (n_trajectories r to MorrisConfig,
    n_samples N to SobolConfig). A tolerance left at None reads
    optimization.sensitivity.convergence_tolerance.
    """
    # Method selection (no default)
    method: SensitivityMethod = field(kw_only=True)

    # Sampling settings (the sample size is a MorrisConfig / SobolConfig field)
    random_seed: Optional[int] = 42

    # Output settings. An empty output_variables list analyses every output
    # the model returns; a non-empty list selects exactly those outputs.
    output_variables: List[str] = field(default_factory=list)
    aggregation_method: str = field(kw_only=True)  # one of AGGREGATION_METHODS
    
    # Computational settings
    batch_size: int = 10
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    # Quality control
    convergence_check: bool = True
    convergence_tolerance: Optional[float] = None

    def __post_init__(self):
        manager = get_global_manager()
        if self.convergence_tolerance is None:
            self.convergence_tolerance = float(manager.get(
                "optimization.sensitivity.convergence_tolerance", required=True
            ))
        if self.aggregation_method not in AGGREGATION_METHODS:
            raise ValueError(
                f"Unknown aggregation_method '{self.aggregation_method}'; "
                f"expected one of {AGGREGATION_METHODS}"
            )


@dataclass
class SensitivityResult:
    """
    Results from sensitivity analysis.
    
    Contains sensitivity indices, statistics, and metadata from
    the sensitivity analysis process.
    """
    # Method information
    method: str
    parameter_names: List[str]
    output_variables: List[str]
    
    # Primary results
    first_order_indices: Dict[str, Dict[str, float]] = field(default_factory=dict)  # output -> param -> index
    total_order_indices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    interaction_indices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Statistical measures
    parameter_rankings: Dict[str, List[str]] = field(default_factory=dict)  # output -> ranked param names
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    
    # Method-specific results
    morris_mu: Optional[Dict[str, Dict[str, float]]] = None  # Morris mu* values
    morris_sigma: Optional[Dict[str, Dict[str, float]]] = None  # Morris sigma values
    
    # Computational metadata (stated by the producer; no default)
    n_model_evaluations: int = field(kw_only=True)
    computation_time: float = field(kw_only=True)
    convergence_achieved: bool = False
    sample_matrix: Optional[np.ndarray] = None
    
    # Quality metrics
    bootstrap_statistics: Dict[str, Any] = field(default_factory=dict)
    variance_explained: Dict[str, float] = field(default_factory=dict)
    
    def get_most_sensitive_parameters(
        self, 
        output_variable: str, 
        n_top: int = 10,
        index_type: str = "total"
    ) -> List[Tuple[str, float]]:
        """
        Get the most sensitive parameters for a given output.
        
        Returns list of (parameter_name, sensitivity_index) tuples
        sorted by decreasing sensitivity.
        """
        if output_variable not in self.output_variables:
            raise ValueError(f"Output variable '{output_variable}' not found in results")
        if index_type not in INDEX_TYPES:
            raise ValueError(f"Unknown index_type '{index_type}'; expected one of {INDEX_TYPES}")

        # Morris results carry mu* as first-order measures only; a request for
        # indices the method did not produce raises instead of an empty ranking.
        available = self.total_order_indices if index_type == "total" else self.first_order_indices
        if output_variable not in available:
            raise KeyError(
                f"{self.method} result has no {index_type}-order indices for "
                f"'{output_variable}'"
            )
        indices = available[output_variable]
        
        # Sort by sensitivity index
        sorted_params = sorted(indices.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_params[:n_top]
    
    def get_interaction_strength(self, output_variable: str) -> float:
        """Calculate the strength of parameter interactions.

        Raises KeyError when the result lacks first- or total-order indices
        for the output (for example a Morris result). Returns NaN when the
        total-order sum is not positive, because the ratio is undefined.
        """
        # References: Saltelli et al. (2008), Global Sensitivity Analysis: The
        #     Primer, Wiley, Sect. 4.6 (sum of S_i versus sum of ST_i).
        # Mathematical expression: I = (sum_i ST_i - sum_i S_i) / sum_i ST_i.
        # Parameter meanings: S_i first-order index, ST_i total-order index of
        #     parameter i for the given output.
        # Units: all quantities are dimensionless variance fractions.
        # Domain/constraints: defined only for sum_i ST_i > 0; otherwise NaN.
        # Sign convention: I > 0 means variance carried by interactions.
        for label, indices in (("first-order", self.first_order_indices),
                               ("total-order", self.total_order_indices)):
            if output_variable not in indices:
                raise KeyError(
                    f"{self.method} result has no {label} indices for '{output_variable}'"
                )

        first_order_sum = sum(self.first_order_indices[output_variable].values())
        total_order_sum = sum(self.total_order_indices[output_variable].values())
        if total_order_sum > 0:
            return (total_order_sum - first_order_sum) / total_order_sum
        return float("nan")


class SensitivityAnalyzer(ABC):
    """
    Abstract base class for sensitivity analysis methods.
    
    Defines the common interface for all sensitivity analysis algorithms
    including Morris screening, Sobol indices, and other GSA methods.
    """
    
    def __init__(
        self,
        parameter_manager: UnifiedParameterManager,
        config: SensitivityConfig
    ):
        """
        Initialize base sensitivity analyzer.
        
        Parameters:
            parameter_manager: Parameter management system
            config: Sensitivity analysis configuration
        """
        self.parameter_manager = parameter_manager
        self.config = config
        
        # Setup computational environment
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        
        # Runtime tracking
        self.model_evaluations = 0
        self.start_time = None
        self.output_order: Optional[List[str]] = None
        
        logger.info(f"Sensitivity analyzer initialized with method={config.method.value}")
    
    @abstractmethod
    def analyze(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        parameter_ranges: Dict[str, ParameterRange],
        baseline_parameters: Optional[Dict[str, float]] = None
    ) -> SensitivityResult:
        """
        Perform sensitivity analysis.
        
        Parameters:
            model_runner: Function that runs model and returns outputs
            parameter_ranges: Parameter sampling ranges
            baseline_parameters: Baseline parameter values (optional)
            
        Returns:
            Sensitivity analysis results
        """
    
    def validate_parameter_ranges(self, parameter_ranges: Dict[str, ParameterRange]) -> None:
        """Validate parameter ranges and check for conflicts.

        Declared units of a range are compared with the units of the
        ParameterSpec of the same key; units declared for a name that is not
        a parameter key cannot be checked and are refused.
        """
        for param_name, param_range in parameter_ranges.items():
            # Check if parameter exists in parameter manager
            known = not hasattr(self.parameter_manager, 'exists') or self.parameter_manager.exists(param_name)
            if not known:
                logger.warning(f"Parameter '{param_name}' not found in parameter manager")
            if param_range.units:
                if not known:
                    raise ValueError(
                        f"Range '{param_name}' declares units '{param_range.units}' but "
                        "'{param_name}' is not a parameter key, so the units cannot be checked"
                    )
                spec_units = str(self.parameter_manager.get_parameter(param_name).units).strip()
                if param_range.units.strip() != spec_units:
                    raise ValueError(
                        f"Range '{param_name}' declares units '{param_range.units}' but the "
                        f"parameter database declares '{spec_units}'"
                    )
            
            # Check range validity
            if param_range.lower_bound >= param_range.upper_bound:
                raise ValueError(f"Invalid range for parameter '{param_name}'")
            
            # Check physical constraints
            if param_name.endswith('_rate') or param_name.endswith('_fraction'):
                if param_range.lower_bound < 0:
                    logger.warning(f"Parameter '{param_name}' has negative lower bound")

    @staticmethod
    def require_uniform_ranges(parameter_ranges: Dict[str, ParameterRange], method: str) -> None:
        """Refuse ranges whose declared sampling the design cannot honour.

        The Morris grid and the Sobol unit-cube design map a uniform coordinate
        linearly onto [lower_bound, upper_bound]. A range declaring another
        distribution, log10 scaling or integer values would otherwise be
        sampled as a plain uniform range without notice.
        """
        for name, param_range in parameter_ranges.items():
            declared = []
            if param_range.distribution != "uniform":
                declared.append(f"distribution='{param_range.distribution}'")
            if param_range.log_scale:
                declared.append("log_scale=True")
            if param_range.integer_valued:
                declared.append("integer_valued=True")
            if declared:
                raise NotImplementedError(
                    f"{method} samples uniform ranges only; parameter '{name}' declares "
                    + ", ".join(declared)
                )

    @staticmethod
    def model_parameters(
        parameter_names: Sequence[str],
        values: Sequence[float],
        baseline_parameters: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Parameters for one model run: baseline values, overridden by the sample."""
        params: Dict[str, Any] = dict(baseline_parameters) if baseline_parameters else {}
        params.update(zip(parameter_names, values))
        return params

    def collect_model_outputs(self, model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Scalar values of the analysed outputs of one model run.

        The first run of an analysis fixes the output order: the declared
        config.output_variables, or every returned output. Later runs must
        return the same outputs. Array outputs are reduced with
        config.aggregation_method. A missing or non-finite output raises.
        """
        declared = list(self.config.output_variables)
        if self.output_order is None:
            order = declared if declared else sorted(model_outputs.keys())
            if not order:
                raise KeyError("Model runner returned no outputs")
            self.output_order = order
        elif not declared and set(model_outputs.keys()) != set(self.output_order):
            raise KeyError(
                f"Model outputs changed between runs: expected {sorted(self.output_order)}, "
                f"got {sorted(model_outputs.keys())}"
            )

        values: Dict[str, float] = {}
        for name in self.output_order:
            if name not in model_outputs:
                raise KeyError(f"Missing output '{name}' from model runner")
            value = model_outputs[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, (list, tuple, np.ndarray)):
                value = self.aggregate_outputs(
                    np.asarray(value, dtype=float).reshape(1, -1), self.config.aggregation_method
                )[0]
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Model output '{name}' is not finite ({value})")
            values[name] = value
        return values
    
    def create_sample_matrix(
        self, 
        parameter_ranges: Dict[str, ParameterRange],
        n_samples: int
    ) -> np.ndarray:
        """
        Create sample matrix for parameter space exploration.
        
        Parameters:
            parameter_ranges: Parameter sampling ranges
            n_samples: Number of samples to generate
            
        Returns:
            Sample matrix [n_samples x n_parameters]
        """
        n_params = len(parameter_ranges)
        sample_matrix = np.zeros((n_samples, n_params))

        # One local generator for the whole matrix: every column draws its own
        # stream (reseeding per column made equal ranges identical columns),
        # and the global numpy random state is left untouched.
        rng = np.random.default_rng(self.config.random_seed)
        
        param_names = sorted(parameter_ranges.keys())
        for i, param_name in enumerate(param_names):
            param_range = parameter_ranges[param_name]
            samples = param_range.sample(n_samples, rng)
            sample_matrix[:, i] = samples
        
        return sample_matrix
    
    def evaluate_model_batch(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        sample_matrix: np.ndarray,
        parameter_names: List[str],
        baseline_parameters: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Evaluate model for batch of parameter samples.
        
        Parameters:
            model_runner: Model evaluation function
            sample_matrix: Parameter samples [n_samples x n_params]
            parameter_names: Names of parameters
            baseline_parameters: Fixed values passed with every sample
            
        Returns:
            Model outputs [n_samples x n_outputs]
        """
        n_samples = sample_matrix.shape[0]
        outputs_list = []
        self.output_order = None
        
        for i in range(n_samples):
            param_dict = self.model_parameters(parameter_names, sample_matrix[i], baseline_parameters)
            
            try:
                values = self.collect_model_outputs(model_runner(param_dict))
                outputs_list.append([values[var] for var in self.output_order])
                
                self.model_evaluations += 1
                
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                logger.error(f"Model evaluation failed for sample {i}: {e}")
                raise RuntimeError(
                    f"Model evaluation failed for sample {i}"
                ) from e
        
        return np.array(outputs_list)
    
    def aggregate_outputs(
        self, 
        outputs: np.ndarray, 
        method: str = "mean"
    ) -> np.ndarray:
        """
        Aggregate model outputs for sensitivity calculation.
        
        Parameters:
            outputs: Model outputs [n_samples x n_outputs]
            method: Aggregation method
            
        Returns:
            Aggregated outputs [n_samples]
        """
        if method == "mean":
            return np.mean(outputs, axis=1)
        elif method == "sum":
            return np.sum(outputs, axis=1)
        elif method == "max":
            return np.max(outputs, axis=1)
        elif method == "std":
            return np.std(outputs, axis=1)
        elif method == "final":
            return outputs[:, -1]  # Last column/time point
        raise ValueError(
            f"Unknown aggregation method '{method}'; expected one of {AGGREGATION_METHODS}"
        )
