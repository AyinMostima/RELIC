"""
Sobol Sensitivity Analysis Implementation

This module implements the Sobol variance-based global sensitivity analysis
for biogeochemical models. The method decomposes output variance into 
contributions from individual parameters and their interactions.

Scientific Background:
    Sobol indices quantify the contribution of each input parameter to the
    output variance through functional ANOVA decomposition. First-order indices
    measure direct effects, while total-order indices include interactions.
    The method provides a comprehensive understanding of parameter importance
    and interaction effects.

References:
    - Sobol, I.M. (1993). Sensitivity estimates for nonlinear mathematical models
    - Saltelli, A. (2002). Making best use of model evaluations to compute sensitivity indices
    - Saltelli, A. et al. (2010). Variance based sensitivity analysis of model output

Mathematical Expression:
    ANOVA decomposition:
    f(x) = f_0 + Sigma?f?(x?) + Sigma?<?f??(x?,x?) + ... + f_1_2...?(x_1,...,x?)
    
    Variance decomposition:
    V = Sigma? V? + Sigma?<? V?? + ... + V_1_2...?
    
    Sobol indices:
    - First-order: S? = V?/V = V[E(Y|X?)]/V(Y)
    - Total-order: S?? = 1 - V[E(Y|X??)]/V(Y)
    
    Where X?? denotes all parameters except parameter i
"""

from typing import Dict, List, Optional, Callable, Any
import numpy as np
import logging
from dataclasses import dataclass, field
import time
from scipy.stats import qmc

from .base import (
    SensitivityAnalyzer,
    SensitivityConfig,
    SensitivityResult,
    ParameterRange,
    SensitivityMethod
)
from ...parameters import UnifiedParameterManager, get_global_manager

logger = logging.getLogger(__name__)

# The one implemented estimator: 'saltelli' = Saltelli (2010) A/B/AB_i
# matrices with the Jansen (1999) squared-difference estimators of the
# first- and total-order indices. No other estimator is offered, and
# second-order indices are not computed.
ESTIMATORS = ("saltelli",)


@dataclass
class SobolConfig(SensitivityConfig):
    """Extended configuration specific to Sobol method.

    The method is fixed by the class; the estimator is chosen by the caller
    (no default). n_samples (N) None reads optimization.sensitivity.sobol.n_samples,
    skip_values None reads optimization.sensitivity.sobol.skip_values, and
    convergence_samples None is N // d_j for the divisors d_j of
    optimization.sensitivity.sobol.convergence_sample_divisors.
    """
    method: SensitivityMethod = field(default=SensitivityMethod.SOBOL, init=False)
    # Sobol-specific parameters
    n_samples: Optional[int] = None    # Base sample size N (None reads YAML)
    skip_values: Optional[int] = None  # Sobol sequence skip (None reads YAML)
    scramble: bool = True              # Scramble Sobol sequence

    # Estimator (the one implemented: "saltelli"; required, no default)
    estimator: str = field(kw_only=True)
    
    # Convergence settings  
    convergence_check: bool = True
    convergence_samples: List[int] = None  # Sample sizes for convergence check
    
    # Computational optimization
    batch_size: int = 100             # Batch size for model evaluations
    
    def __post_init__(self):
        super().__post_init__()
        manager = get_global_manager()
        if self.n_samples is None:
            self.n_samples = int(manager.get("optimization.sensitivity.sobol.n_samples", required=True))
        if self.skip_values is None:
            self.skip_values = int(manager.get(
                "optimization.sensitivity.sobol.skip_values", required=True
            ))
        
        # Validate Sobol-specific parameters
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.skip_values < 0:
            raise ValueError("skip_values must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.estimator not in ESTIMATORS:
            raise ValueError(
                f"Unknown Sobol estimator '{self.estimator}'; available: {ESTIMATORS}"
            )
        
        # Convergence sample sizes not stated by the caller: N // d_j for the
        # declared divisors (optimization.sensitivity.sobol.convergence_sample_divisors).
        if self.convergence_samples is None:
            divisors = manager.get(
                "optimization.sensitivity.sobol.convergence_sample_divisors", required=True
            )
            self.convergence_samples = [self.n_samples // int(divisor) for divisor in divisors]
        if self.convergence_check:
            sizes = list(self.convergence_samples)
            valid = (
                len(sizes) >= 2
                and all(
                    isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= self.n_samples
                    for n in sizes
                )
                and all(a < b for a, b in zip(sizes, sizes[1:]))
            )
            if not valid:
                raise ValueError(
                    "convergence_samples must hold at least two strictly increasing sample "
                    f"sizes in [1, n_samples={self.n_samples}], got {sizes}"
                )


class SobolAnalyzer(SensitivityAnalyzer):
    """
    Sobol Variance-Based Global Sensitivity Analysis.
    
    This implementation provides comprehensive Sobol sensitivity analysis
    with first-order and total-order indices.
    The method is based on variance decomposition and provides quantitative
    measures of parameter importance and interaction effects.
    
    Key Features:
    1. First-order Sobol indices (main effects)
    2. Total-order Sobol indices (including interactions)
    3. One estimator: Saltelli (2010) design with the Jansen (1999) estimators
    4. Convergence analysis and bootstrap confidence intervals
    5. Efficient Sobol sequence sampling
    
    Applications in biogeochemical modeling:
    - Complete parameter importance ranking
    - Interaction effect quantification
    - Model simplification guidance
    - Uncertainty propagation analysis
    """
    
    def __init__(
        self,
        parameter_manager: UnifiedParameterManager,
        config: Optional[SobolConfig] = None
    ):
        """
        Initialize Sobol analyzer.
        
        Parameters:
            parameter_manager: Parameter management system
            config: Sobol-specific configuration
        """
        if config is None:
            raise TypeError(
                "SobolAnalyzer needs a SobolConfig: the aggregation method and the "
                "estimator are chosen by the caller"
            )
        self.sobol_config = config
        super().__init__(parameter_manager, self.sobol_config)
        
        # Sobol-specific state
        self.sample_matrices = None
        self.model_outputs = None
        
        logger.info(f"Sobol analyzer initialized with {self.sobol_config.n_samples} base samples")
    
    def analyze(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        parameter_ranges: Dict[str, ParameterRange],
        baseline_parameters: Optional[Dict[str, float]] = None
    ) -> SensitivityResult:
        """
        Perform Sobol sensitivity analysis.
        
        Parameters:
            model_runner: Function that runs model and returns outputs
            parameter_ranges: Parameter sampling ranges
            baseline_parameters: Baseline parameter values (optional)
            
        Returns:
            Sobol sensitivity analysis results
        """
        self.start_time = time.time()
        
        # Validate inputs
        self.validate_parameter_ranges(parameter_ranges)
        
        if len(parameter_ranges) == 0:
            raise ValueError("At least one parameter range must be specified")
        self.require_uniform_ranges(parameter_ranges, "Sobol analysis")
        
        param_names = sorted(parameter_ranges.keys())
        n_params = len(param_names)
        
        logger.info(f"Starting Sobol analysis for {n_params} parameters")
        
        try:
            # 1. Generate Sobol sample matrices
            sample_matrices = self._generate_sobol_samples(parameter_ranges, param_names)
            
            # 2. Evaluate model on all sample points
            model_outputs = self._evaluate_sobol_samples(
                model_runner, sample_matrices, param_names, baseline_parameters
            )
            
            # 3. Calculate Sobol indices
            sobol_indices = self._calculate_sobol_indices(model_outputs, n_params)
            
            # 4. Perform convergence analysis if requested
            convergence_data = None
            if self.sobol_config.convergence_check:
                convergence_data = self._analyze_convergence(model_outputs, n_params)
            
            # 5. Create sensitivity result
            computation_time = time.time() - self.start_time
            result = self._create_sobol_result(
                sobol_indices, param_names, convergence_data,
                n_model_evaluations=self.model_evaluations,
                computation_time=computation_time,
            )
            result.convergence_achieved = self._check_convergence(convergence_data) if convergence_data else False
            
            logger.info(f"Sobol analysis completed: {self.model_evaluations} evaluations in {computation_time:.2f}s")
            
            return result
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"Sobol analysis failed: {e}")
            raise RuntimeError(f"Sobol analysis failed: {e}") from e
    
    def _generate_sobol_samples(
        self,
        parameter_ranges: Dict[str, ParameterRange], 
        param_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Generate Sobol sample matrices A, B, and AB_i matrices.
        
        Returns dictionary with sample matrices for Sobol estimation.
        """
        n_params = len(param_names)
        n_samples = self.sobol_config.n_samples
        
        # Draw a single Sobol design in 2D dimensions and split it into
        # independent A/B blocks. Using two consecutive draws from a D-dimensional
        # generator creates correlated base matrices and biases pick-freeze
        # estimators, especially for additive reference models.
        # The scrambling draws from the declared random_seed so a seeded analysis
        # is reproducible. scipy's fast_forward(0) raises, so a zero skip is not
        # forwarded.
        sobol_gen = qmc.Sobol(
            d=2 * n_params,
            scramble=self.sobol_config.scramble,
            rng=self.sobol_config.random_seed,
        )
        if self.sobol_config.skip_values > 0:
            sobol_gen.fast_forward(self.sobol_config.skip_values)
        base_samples = sobol_gen.random(n_samples)
        samples_A = base_samples[:, :n_params]
        samples_B = base_samples[:, n_params:]
        
        # Transform samples to parameter ranges
        def transform_samples(samples):
            transformed = np.zeros_like(samples)
            for i, param_name in enumerate(param_names):
                param_range = parameter_ranges[param_name]
                transformed[:, i] = (param_range.lower_bound + 
                                   samples[:, i] * (param_range.upper_bound - param_range.lower_bound))
            return transformed
        
        A = transform_samples(samples_A)
        B = transform_samples(samples_B)
        
        # Create AB_i matrices (A with i-th column from B)
        AB_matrices = {}
        for i in range(n_params):
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            AB_matrices[f'AB_{i}'] = AB_i
        
        sample_matrices = {
            'A': A,
            'B': B,
            **AB_matrices
        }
        
        self.sample_matrices = sample_matrices
        
        total_samples = n_samples * (2 + n_params)
        logger.debug(f"Generated Sobol sample matrices: {total_samples} total evaluations needed")
        
        return sample_matrices
    
    def _evaluate_sobol_samples(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        sample_matrices: Dict[str, np.ndarray],
        param_names: List[str],
        baseline_parameters: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Evaluate model on all Sobol sample matrices.
        
        Returns nested dictionary: {matrix_name: {output_var: values}}.
        All matrices share one output set; every run must return every
        analysed output as a finite value.
        """
        model_outputs = {}
        self.output_order = None
        
        for matrix_name, samples in sample_matrices.items():
            logger.debug(f"Evaluating matrix {matrix_name} with {len(samples)} samples")
            
            matrix_outputs = {}
            
            # Evaluate in batches for memory efficiency
            batch_size = self.sobol_config.batch_size
            n_samples = len(samples)
            
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_samples = samples[batch_start:batch_end]
                
                for i, sample in enumerate(batch_samples):
                    sample_idx = batch_start + i
                    param_dict = self.model_parameters(param_names, sample, baseline_parameters)
                    
                    try:
                        values = self.collect_model_outputs(model_runner(param_dict))
                        for var_name, value in values.items():
                            if var_name not in matrix_outputs:
                                matrix_outputs[var_name] = np.zeros(n_samples)
                            matrix_outputs[var_name][sample_idx] = value
                        
                        self.model_evaluations += 1
                        
                    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                        logger.error(f"Model evaluation failed for matrix {matrix_name}, sample {sample_idx}: {e}")
                        raise RuntimeError(
                            f"Model evaluation failed for matrix {matrix_name}, sample {sample_idx}"
                        ) from e
            
            model_outputs[matrix_name] = matrix_outputs
        
        self.model_outputs = model_outputs
        return model_outputs
    
    def _calculate_sobol_indices(
        self,
        model_outputs: Dict[str, Dict[str, np.ndarray]],
        n_params: int
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate Sobol first-order and total-order indices.
        
        Returns: {output_var: {index_type: {param_idx: value}}}
        """
        sobol_indices = {}
        
        for var_name in self.output_order:
            # Extract outputs for this variable
            f_A = model_outputs['A'][var_name]
            f_B = model_outputs['B'][var_name]
            f_AB = {i: model_outputs[f'AB_{i}'][var_name] for i in range(n_params)}
            
            # Estimate output variance from the independent A/B ensembles only.
            # The Saltelli/Jansen estimators assume variance is computed from the
            # base sample pairs, not from the hybrid AB_i matrices.
            variance_samples = np.concatenate([f_A, f_B])
            total_variance = np.var(variance_samples, ddof=1)

            # Indices of an output without variance are undefined; they are not
            # computed against a substituted variance floor.
            if not np.isfinite(total_variance) or total_variance <= 0.0:
                raise ValueError(
                    f"Sobol indices of output '{var_name}' are undefined: output variance "
                    f"is {total_variance}"
                )
            
            # Initialize indices
            first_order = {}
            total_order = {}

            # References: Saltelli et al. (2010), Comput. Phys. Commun. 181,
            #     259-270 (A, B, AB_i design); Jansen (1999), Comput. Phys. Commun.
            #     117, 35-43 (squared-difference estimators).
            # Mathematical expression: S_i = 1 - E[(f_B - f_AB_i)^2] / (2 V(Y)),
            #     ST_i = E[(f_A - f_AB_i)^2] / (2 V(Y)), V(Y) = sample variance of
            #     [f_A, f_B] with ddof = 1.
            # Parameter meanings: f_A, f_B = outputs on the independent base
            #     matrices, f_AB_i = outputs on A with column i taken from B,
            #     E = sample mean over the N base samples.
            # Units: S_i and ST_i are dimensionless; the outputs and V(Y) carry the
            #     output units and their square.
            # Domain/constraints: V(Y) > 0 and finite (checked above). Finite-
            #     sample estimates are reported unclipped: S_i may be slightly
            #     negative and sum_i S_i may exceed 1 through estimator noise.
            # Sign convention: a larger index means a larger share of V(Y).
            for i in range(n_params):
                first_order[i] = float(1.0 - 0.5 * np.mean((f_B - f_AB[i]) ** 2) / total_variance)
                total_order[i] = float(0.5 * np.mean((f_A - f_AB[i]) ** 2) / total_variance)
            
            sobol_indices[var_name] = {
                'first_order': first_order,
                'total_order': total_order,
                'total_variance': total_variance
            }
        
        return sobol_indices
    
    def _analyze_convergence(
        self,
        model_outputs: Dict[str, Dict[str, np.ndarray]],
        n_params: int
    ) -> Dict[str, Any]:
        """Analyze convergence of Sobol indices with sample size"""
        convergence_data = {
            var_name: {
                'sample_sizes': [],
                'first_order_convergence': {i: [] for i in range(n_params)},
                'total_order_convergence': {i: [] for i in range(n_params)}
            }
            for var_name in self.output_order
        }

        # Indices of every analysed output on the leading n_conv base samples
        # (SobolConfig guarantees 1 <= n_conv <= n_samples).
        for n_conv in self.sobol_config.convergence_samples:
            subset_outputs = {
                matrix_name: {
                    var_name: outputs[var_name][:n_conv] for var_name in self.output_order
                }
                for matrix_name, outputs in model_outputs.items()
            }
            subset_indices = self._calculate_sobol_indices(subset_outputs, n_params)

            for var_name in self.output_order:
                convergence_data[var_name]['sample_sizes'].append(n_conv)
                for i in range(n_params):
                    convergence_data[var_name]['first_order_convergence'][i].append(
                        subset_indices[var_name]['first_order'][i]
                    )
                    convergence_data[var_name]['total_order_convergence'][i].append(
                        subset_indices[var_name]['total_order'][i]
                    )
        
        return convergence_data
    
    def _check_convergence(self, convergence_data: Dict[str, Any]) -> bool:
        """Check if Sobol indices have converged"""
        if not convergence_data:
            return False
        
        tolerance = self.sobol_config.convergence_tolerance

        # References: Sarrazin, Pianosi and Wagener (2016), Environ. Model.
        #     Softw. 79, 135-152 (convergence of sensitivity indices judged on an
        #     absolute scale, threshold 0.05).
        # Mathematical expression: converged when |S_i(N_k) - S_i(N_{k-1})| <= tol
        #     and |ST_i(N_k) - ST_i(N_{k-1})| <= tol for every parameter i and
        #     output, N_{k-1} < N_k the last two convergence sample sizes.
        # Parameter meanings: S_i, ST_i = first- and total-order indices, N_k =
        #     base sample sizes, tol = convergence_tolerance.
        # Units: indices and tol are dimensionless variance fractions.
        # Domain/constraints: at least two sample sizes per output; an index near
        #     zero is compared on the same absolute scale (no relative-change
        #     denominator floor).
        # Sign convention: only the magnitude of the change matters.
        for var_name, var_data in convergence_data.items():
            if len(var_data['sample_sizes']) < 2:
                raise ValueError(
                    f"Convergence of '{var_name}' needs at least two sample sizes, got "
                    f"{var_data['sample_sizes']}"
                )
            
            for param_idx, first_order_vals in var_data['first_order_convergence'].items():
                total_order_vals = var_data['total_order_convergence'][param_idx]
                change_first = abs(first_order_vals[-1] - first_order_vals[-2])
                change_total = abs(total_order_vals[-1] - total_order_vals[-2])
                if change_first > tolerance or change_total > tolerance:
                    return False
        
        return True
    
    def _create_sobol_result(
        self,
        sobol_indices: Dict[str, Dict[str, Dict[str, float]]],
        param_names: List[str],
        convergence_data: Optional[Dict[str, Any]],
        *,
        n_model_evaluations: int,
        computation_time: float,
    ) -> SensitivityResult:
        """Create SensitivityResult from Sobol indices"""
        
        first_order_indices = {}
        total_order_indices = {}
        parameter_rankings = {}
        variance_explained = {}
        
        for var_name, indices in sobol_indices.items():
            # Convert parameter indices to names
            first_order_indices[var_name] = {
                param_names[i]: value for i, value in indices['first_order'].items()
            }
            total_order_indices[var_name] = {
                param_names[i]: value for i, value in indices['total_order'].items()
            }
            
            # Create parameter ranking based on total-order indices
            sorted_params = sorted(
                total_order_indices[var_name].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            parameter_rankings[var_name] = [param_name for param_name, _ in sorted_params]
            
            # Share of variance explained by first-order effects, reported as
            # estimated (it can exceed 1 through estimator noise).
            variance_explained[var_name] = float(sum(first_order_indices[var_name].values()))
        
        # Create result
        result = SensitivityResult(
            method="Sobol",
            parameter_names=param_names,
            output_variables=list(sobol_indices.keys()),
            first_order_indices=first_order_indices,
            total_order_indices=total_order_indices,
            parameter_rankings=parameter_rankings,
            variance_explained=variance_explained,
            n_model_evaluations=n_model_evaluations,
            computation_time=computation_time,
        )
        
        # Add convergence data if available
        if convergence_data:
            result.bootstrap_statistics['convergence_data'] = convergence_data
        
        return result
