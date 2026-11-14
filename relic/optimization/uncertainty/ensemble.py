"""
Ensemble Uncertainty Quantification Implementation

This module implements ensemble-based uncertainty quantification for biogeochemical
models. The method uses Monte Carlo sampling to propagate parameter uncertainties
through model simulations and characterize output distributions.

Scientific Background:
    Ensemble uncertainty quantification uses Monte Carlo sampling to propagate
    parametric uncertainties through complex models. By running multiple model
    realizations with different parameter sets drawn from their uncertainty
    distributions, the method builds empirical output distributions conditional
    on those declared parameter distributions. Structural uncertainty is not
    represented unless the caller explicitly encodes it in the model ledger.

References:
    - Beven, K. & Freer, J. (2001). Equifinality, data assimilation, and uncertainty estimation
    - Vrugt, J.A. et al. (2003). A Shuffled Complex Evolution Metropolis algorithm for optimization
    - Duan, Q. et al. (2007). Model Parameter Estimation Experiment (MOPEX): An overview
    - Clark, M.P. et al. (2008). Framework for Understanding Structural Errors (FUSE)

Mathematical Expression:
    Forward uncertainty propagation:
    theta_i ~ p(theta), i = 1,...,N (parameter ensemble)
    y_i = f(theta_i) (model evaluations)
    
    Empirical output distribution:
    p_N(y) = (1/N) sum_i delta(y - y_i)
    
    Statistical moments:
    mu_N = (1/N) sum_i y_i (mean)
    sigma_N^2 = (1/(N-1)) sum_i (y_i - mu_N)^2 (variance)
    
    Where N is ensemble size, theta are parameters, y are outputs
"""

from typing import Any, Callable, Dict, Optional
import numpy as np
import torch
import logging
from dataclasses import dataclass, field
import time

from scipy.stats import qmc

from .base import (
    UncertaintyQuantifier,
    UQConfig,
    UQResult,
    UncertainParameter,
    UQMethod,
)
# Use stochastic ensemble runner as backend
from ...stochastic.ensemble_runner import (
    CallableContract as StochasticCallableContract,
    ExecutionConfig as StochasticExecutionConfig,
    ExecutionMode as StochasticExecutionMode,
    StatisticType,
    EnsembleRunner as StochasticEnsembleRunner
)
from ...parameters import UnifiedParameterManager, get_global_manager

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig(UQConfig):
    """Extended configuration specific to ensemble UQ.

    The method is fixed by the class; the sampling strategy is chosen by the
    caller (no default). ensemble_size and ensemble_inflation left at None
    read optimization.uncertainty.ensemble.*, outlier_threshold reads the
    key stochastic.ensemble.outlier_threshold shared with ExecutionConfig.
    """
    method: UQMethod = field(default=UQMethod.ENSEMBLE, init=False)
    # Ensemble-specific parameters
    ensemble_size: Optional[int] = None  # Number of ensemble members
    sampling_strategy: str = field(kw_only=True)  # "random", "lhs", "sobol"

    # Ensemble management
    ensemble_inflation: Optional[float] = None  # Inflate ensemble spread if needed
    localization: bool = False         # Spatial localization
    adaptive_sampling: bool = False    # Adaptive ensemble size
    
    # Computational optimization
    parallel_evaluation: bool = False
    parallel_callable_contract: StochasticCallableContract = (
        StochasticCallableContract.UNSPECIFIED
    )
    n_workers: int = 4
    chunk_size: int = 100
    
    # Quality control
    outlier_detection: bool = False
    outlier_threshold: Optional[float] = None  # Standard deviations (z-score)

    def __post_init__(self):
        super().__post_init__()
        manager = get_global_manager()
        if self.ensemble_size is None:
            self.ensemble_size = int(manager.get(
                "optimization.uncertainty.ensemble.ensemble_size", required=True
            ))
        if self.ensemble_inflation is None:
            self.ensemble_inflation = float(manager.get(
                "optimization.uncertainty.ensemble.inflation", required=True
            ))
        if self.outlier_threshold is None:
            self.outlier_threshold = float(manager.get(
                "stochastic.ensemble.outlier_threshold", required=True
            ))
        # The sample size of an ensemble is its member count
        self.n_samples = self.ensemble_size
        if (
            isinstance(self.ensemble_size, bool)
            or not isinstance(self.ensemble_size, int)
            or self.ensemble_size <= 0
        ):
            raise ValueError("ensemble_size must be a positive integer")
        if self.sampling_strategy not in {"random", "lhs", "sobol"}:
            raise ValueError(
                "sampling_strategy must be one of: random, lhs, sobol"
            )
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer or None")
        for field_name in ("n_workers", "chunk_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in ("ensemble_inflation", "outlier_threshold"):
            value = getattr(self, field_name)
            scalar = torch.as_tensor(value, dtype=torch.float64)
            if (
                isinstance(value, bool)
                or scalar.numel() != 1
                or not bool(torch.isfinite(scalar).item())
                or float(value) <= 0.0
            ):
                raise ValueError(f"{field_name} must be a positive finite number")
        for field_name in (
            "localization",
            "adaptive_sampling",
            "parallel_evaluation",
            "outlier_detection",
            "convergence_check",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean")
        for field_name in ("localization", "adaptive_sampling"):
            if getattr(self, field_name):
                raise NotImplementedError(
                    f"{field_name} is [N] and has no operational UQ contract"
                )
        if not isinstance(
            self.parallel_callable_contract,
            StochasticCallableContract,
        ):
            raise TypeError(
                "parallel_callable_contract must use the canonical "
                "CallableContract enum"
            )


class EnsembleUQ(UncertaintyQuantifier):
    """
    Ensemble Uncertainty Quantification Implementation.
    
    This method provides robust uncertainty quantification through Monte Carlo
    ensemble simulations. Sequential and thread-parallel execution both use the
    canonical stochastic ``EnsembleRunner``. Thread execution is available only
    when the caller explicitly certifies a deterministic, pure model callable.
    
    Key Features:
    1. Monte Carlo parameter sampling with multiple strategies
    2. Ensemble-based uncertainty propagation  
    3. Empirical distribution characterization
    4. Caller-certified deterministic-pure thread evaluation
    5. Explicitly enabled outlier detection and quality control
    6. Heuristic convergence monitoring with a non-vacuous evidence floor

    Non-operational capabilities:
        Localization and adaptive sampling are [N] and fail closed during
        configuration. They are not inferred from metadata or labels.
    
    Applications in biogeochemical modeling:
    - Parameter uncertainty propagation
    - Model prediction confidence bounds
    - Risk assessment and decision support
    - Sensitivity analysis complementary to GSA
    """
    
    def __init__(
        self,
        parameter_manager: UnifiedParameterManager,
        config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize ensemble UQ analyzer.
        
        Parameters:
            parameter_manager: Parameter management system
            config: Ensemble-specific configuration
        """
        if config is None:
            raise TypeError(
                "EnsembleUQ needs an EnsembleConfig: the sampling strategy is chosen by the caller"
            )
        self.ensemble_config = config
        super().__init__(parameter_manager, self.ensemble_config)
        
        # Ensemble-specific state
        self.ensemble_parameters = None
        self.ensemble_outputs = None
        self.quality_flags = None
        
        logger.info(f"Ensemble UQ initialized with {self.ensemble_config.ensemble_size} members")
    
    def quantify(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        uncertain_parameters: Dict[str, UncertainParameter],
        baseline_parameters: Optional[Dict[str, float]] = None,
        observations: Optional[Dict[str, np.ndarray]] = None
    ) -> UQResult:
        """
        Perform ensemble uncertainty quantification.
        
        Parameters:
            model_runner: Function that runs model and returns outputs
            uncertain_parameters: Parameters with uncertainty distributions
            baseline_parameters: Baseline/fixed parameter values
            observations: Observational data (not used in basic ensemble method)
            
        Returns:
            Ensemble UQ results
        """
        self.start_time = time.time()
        self.model_evaluations = 0
        self.samples_generated = 0
        
        # Validate inputs
        self.validate_uncertain_parameters(uncertain_parameters)
        
        if len(uncertain_parameters) == 0:
            raise ValueError("At least one uncertain parameter must be specified")
        
        param_names = sorted(uncertain_parameters.keys())
        ensemble_size = self.ensemble_config.ensemble_size
        
        logger.info(f"Starting ensemble UQ for {len(param_names)} parameters with {ensemble_size} members")
        
        try:
            # 1. Generate parameter ensemble
            parameter_samples = self._generate_parameter_ensemble(uncertain_parameters)
            if parameter_samples:
                first_param = next(iter(parameter_samples.keys()))
                self.samples_generated = int(len(parameter_samples[first_param]))
            
            # 2. Evaluate every member through the canonical numerical owner
            ensemble_outputs = self._evaluate_ensemble(
                model_runner, parameter_samples, baseline_parameters
            )
            
            # 3. Quality control and outlier detection
            if self.ensemble_config.outlier_detection:
                ensemble_outputs, outlier_mask = self._filter_outliers(ensemble_outputs)
                # Sync parameter_samples with filtered outputs
                if outlier_mask is not None:
                    parameter_samples = {
                        k: v[~outlier_mask] for k, v in parameter_samples.items()
                    }
            
            # 4. Compute uncertainty statistics
            computation_time = time.time() - self.start_time
            result = self.compute_statistics(
                ensemble_outputs, parameter_samples, computation_time=computation_time
            )

            retained_members = int(len(next(iter(ensemble_outputs.values()))))
            removed_members = self.samples_generated - retained_members
            self.quality_flags = {
                "requested": ensemble_size,
                "generated": self.samples_generated,
                "retained": retained_members,
                "removed": removed_members,
            }
            result.n_samples = retained_members
            result.n_model_evaluations = self.model_evaluations
            
            # 5. Add ensemble-specific metadata
            result.output_samples = ensemble_outputs
            result.parameter_samples = parameter_samples
            
            # 6. Check convergence if requested
            if self.ensemble_config.convergence_check:
                result.convergence_achieved = self._check_ensemble_convergence(ensemble_outputs)
            
            logger.info(f"Ensemble UQ completed: {self.model_evaluations} evaluations in {computation_time:.2f}s")
            
            return result
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"Ensemble UQ failed: {e}")
            raise RuntimeError(f"Ensemble UQ failed: {e}")
    
    def _generate_parameter_ensemble(
        self,
        uncertain_parameters: Dict[str, UncertainParameter]
    ) -> Dict[str, np.ndarray]:
        """Generate a parameter ensemble from an explicit distribution ledger.

        References:
            McKay et al. (1979), Technometrics 21, 239-245.
            Sobol (1967), USSR Computational Mathematics 7, 86-112.

        Mathematical Expression:
            ``u_i in (0, 1)^d`` is generated by Monte Carlo, Latin hypercube,
            or a scrambled Sobol design, and ``theta_ij = F_j^{-1}(u_ij)``.

        Parameter Meanings:
            ``u_i`` is the unit-cube design point, ``F_j`` is the declared
            marginal CDF, ``theta_ij`` is parameter ``j`` for member ``i``,
            ``d`` is the parameter count, and ``N`` is the ensemble size.

        Returns:
            Dictionary of one-dimensional NumPy arrays at the legacy UQ API
            boundary. Sampling and transforms remain local and Torch-first.
        """
        if not isinstance(uncertain_parameters, dict) or not uncertain_parameters:
            raise ValueError("uncertain_parameters must be a non-empty mapping")
        for ledger_name, parameter in uncertain_parameters.items():
            if not isinstance(ledger_name, str) or not ledger_name.strip():
                raise ValueError("Parameter ledger keys must be non-empty strings")
            if not isinstance(parameter, UncertainParameter):
                raise TypeError(
                    f"Parameter ledger entry '{ledger_name}' must be UncertainParameter"
                )
            if parameter.name != ledger_name:
                raise ValueError(
                    f"Parameter ledger key '{ledger_name}' must equal "
                    f"UncertainParameter.name '{parameter.name}'"
                )
        param_names = sorted(uncertain_parameters.keys())
        n_params = len(param_names)
        ensemble_size = self.ensemble_config.ensemble_size
        strategy = self.ensemble_config.sampling_strategy

        unit_samples = self._generate_unit_design(
            strategy=strategy,
            ensemble_size=ensemble_size,
            n_params=n_params,
            seed=self.ensemble_config.random_seed,
        )
        parameter_samples = {
            param_name: uncertain_parameters[param_name].transform_unit_samples(
                unit_samples[:, index]
            )
            .detach()
            .cpu()
            .numpy()
            .copy()
            for index, param_name in enumerate(param_names)
        }
        
        # Apply ensemble inflation if requested
        if self.ensemble_config.ensemble_inflation != 1.0:
            parameter_samples = self._inflate_ensemble(parameter_samples, uncertain_parameters)
        
        self.ensemble_parameters = parameter_samples
        self.samples_generated = int(ensemble_size)
        logger.debug(f"Generated parameter ensemble with {ensemble_size} members")
        
        return parameter_samples

    @staticmethod
    def _generate_unit_design(
        strategy: str,
        ensemble_size: int,
        n_params: int,
        seed: Optional[int],
    ) -> torch.Tensor:
        """Generate a local unit-cube design without mutating global RNG state.

        References:
            McKay et al. (1979), Technometrics 21, 239-245.
            Owen (1998), SIAM Journal on Numerical Analysis 34, 1884-1910.

        Mathematical Expression:
            Monte Carlo uses ``u_ij ~ Uniform(0, 1)``. LHS stratifies every
            marginal into ``N`` equiprobable intervals. Scrambled Sobol points
            use a randomized low-discrepancy digital net with ``N = 2^m``.

        Parameter Meanings:
            ``strategy`` selects the design, ``ensemble_size`` is ``N``,
            ``n_params`` is ``d``, and ``seed`` identifies the local stream.
        """
        if strategy == "random":
            generator = torch.Generator(device="cpu")
            if seed is None:
                generator.seed()
            else:
                generator.manual_seed(int(seed))
            return torch.rand(
                (ensemble_size, n_params),
                generator=generator,
                dtype=torch.float64,
            )

        local_rng = np.random.default_rng(seed)
        if strategy == "lhs":
            design = qmc.LatinHypercube(
                d=n_params,
                scramble=True,
                rng=local_rng,
            ).random(ensemble_size)
        elif strategy == "sobol":
            if ensemble_size & (ensemble_size - 1):
                raise ValueError(
                    "Sobol ensemble_size must be a power of two for random_base2"
                )
            exponent = ensemble_size.bit_length() - 1
            design = qmc.Sobol(
                d=n_params,
                scramble=True,
                rng=local_rng,
            ).random_base2(exponent)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        return torch.as_tensor(design.copy(), dtype=torch.float64)

    def _inflate_ensemble(
        self,
        parameter_samples: Dict[str, np.ndarray],
        uncertain_parameters: Dict[str, UncertainParameter]
    ) -> Dict[str, np.ndarray]:
        """Apply bounded multiplicative inflation around the ensemble mean.

        References:
            Anderson and Anderson (1999), Monthly Weather Review 127, 2741-2758.

        Mathematical Expression:
            ``theta_i' = mean(theta) + lambda(theta_i - mean(theta))``.

        Parameter Meanings:
            ``lambda`` is ``ensemble_inflation`` and ``theta_i`` is one member.
        """
        inflation_factor = self.ensemble_config.ensemble_inflation
        
        inflated_samples = {}
        for param_name, samples in parameter_samples.items():
            uncertain_param = uncertain_parameters[param_name]
            values = torch.as_tensor(samples, dtype=torch.float64)
            if values.dim() != 1 or values.numel() == 0:
                raise ValueError(
                    f"Parameter '{param_name}' samples must be non-empty and one-dimensional"
                )
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"Parameter '{param_name}' samples must be finite")

            ensemble_mean = torch.mean(values)
            inflated = ensemble_mean + float(inflation_factor) * (
                values - ensemble_mean
            )

            lower = torch.as_tensor(
                uncertain_param.lower_bound,
                dtype=torch.float64,
            )
            upper = torch.as_tensor(
                uncertain_param.upper_bound,
                dtype=torch.float64,
            )
            if uncertain_param.log_transform:
                lower = (
                    torch.log(lower)
                    if bool((lower > 0.0).item())
                    else torch.tensor(float("-inf"), dtype=torch.float64)
                )
                upper = (
                    torch.log(upper)
                    if bool(torch.isfinite(upper).item())
                    else upper
                )
            # Inflation about the ensemble mean has no bound handling (Anderson
            # and Anderson 1999); a member pushed outside the declared bounds is
            # an error, not clamped onto the bound.
            outside = (inflated < lower) | (inflated > upper)
            if bool(outside.any()):
                raise ValueError(
                    f"ensemble inflation {inflation_factor} moves {int(outside.sum())} member(s) "
                    f"of '{param_name}' outside its bounds [{float(lower)}, {float(upper)}]"
                )
            if uncertain_param.integer_valued:
                inflated = torch.round(inflated)
            if not bool(torch.isfinite(inflated).all().item()):
                raise ValueError(
                    f"Parameter '{param_name}' inflation generated non-finite samples"
                )
            inflated_samples[param_name] = inflated.detach().cpu().numpy().copy()
        
        logger.debug(f"Applied ensemble inflation factor: {inflation_factor}")
        return inflated_samples
    
    def _evaluate_ensemble(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        parameter_samples: Dict[str, np.ndarray],
        baseline_parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate scalar ensemble members through the canonical numerical owner.

        The UQ layer adapts its NumPy/Python boundary to ``EnsembleRunner`` but
        does not own workers, retries, member scheduling, or result stacking.
        Thread mode passes the caller declaration to the canonical runner,
        which alone owns and enforces callable-purity policy.
        """
        if not callable(model_runner):
            raise TypeError("model_runner must be callable")
        if self.config.allow_failed_samples or self.config.allow_missing_outputs:
            raise ValueError(
                "EnsembleRunner execution is fail-closed; substitution policies "
                "require a separately audited sequential substitution workflow"
            )
        if not isinstance(parameter_samples, dict) or not parameter_samples:
            raise ValueError("parameter_samples must be a non-empty mapping")

        tensor_samples: Dict[str, torch.Tensor] = {}
        sample_dtypes: Dict[str, torch.dtype] = {}
        ensemble_size: Optional[int] = None
        for param_name, samples in parameter_samples.items():
            if not isinstance(param_name, str) or not param_name.strip():
                raise ValueError("Parameter sample names must be non-empty strings")
            tensor = torch.as_tensor(samples, device=self.device)
            if tensor.dim() != 1 or tensor.shape[0] == 0:
                raise ValueError(
                    f"parameter_samples['{param_name}'] must be a non-empty "
                    "one-dimensional ledger"
                )
            if tensor.dtype == torch.bool or tensor.is_complex():
                raise TypeError(
                    f"parameter_samples['{param_name}'] must be real numeric values"
                )
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(
                    f"parameter_samples['{param_name}'] must contain only finite values"
                )
            sample_dtypes[param_name] = tensor.dtype
            if not tensor.is_floating_point():
                transport_tensor = tensor.to(dtype=torch.float64)
                if not torch.equal(
                    transport_tensor.to(dtype=tensor.dtype),
                    tensor,
                ):
                    raise ValueError(
                        f"parameter_samples['{param_name}'] cannot be transported "
                        "through EnsembleRunner without integer precision loss"
                    )
                tensor = transport_tensor
            if ensemble_size is None:
                ensemble_size = tensor.shape[0]
            elif tensor.shape[0] != ensemble_size:
                raise ValueError(
                    "All parameter samples must have the same ensemble size"
                )
            tensor_samples[param_name] = tensor
        assert ensemble_size is not None

        if baseline_parameters is None:
            fixed_inputs: Dict[str, Any] = {}
        elif isinstance(baseline_parameters, dict):
            fixed_inputs = dict(baseline_parameters)
        else:
            raise TypeError("baseline_parameters must be a mapping when provided")
        for param_name in tensor_samples:
            fixed_inputs.pop(param_name, None)

        def scalar_model_adapter(inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
            adapted_inputs = dict(inputs)
            for param_name in tensor_samples:
                value = adapted_inputs[param_name]
                if not isinstance(value, torch.Tensor) or value.numel() != 1:
                    raise ValueError(
                        f"Sampled input '{param_name}' must be a scalar tensor"
                    )
                adapted_inputs[param_name] = (
                    value.detach()
                    .to(dtype=sample_dtypes[param_name])
                    .item()
                )

            outputs = model_runner(adapted_inputs)
            if not isinstance(outputs, dict) or not outputs:
                raise ValueError("model_runner must return a non-empty output mapping")

            tensor_outputs: Dict[str, torch.Tensor] = {}
            for output_name, output_value in outputs.items():
                if not isinstance(output_name, str) or not output_name.strip():
                    raise ValueError("Model output names must be non-empty strings")
                value = torch.as_tensor(output_value, device=self.device)
                if value.dtype == torch.bool or value.is_complex():
                    raise TypeError(
                        f"Model output '{output_name}' must be a real numeric scalar"
                    )
                if value.dim() != 0:
                    raise ValueError(
                        f"Model output '{output_name}' must be scalar per member"
                    )
                if not bool(torch.isfinite(value).item()):
                    raise ValueError(
                        f"Model output '{output_name}' must be finite"
                    )
                tensor_outputs[output_name] = value
            return tensor_outputs

        exec_config = StochasticExecutionConfig(
            mode=(
                StochasticExecutionMode.PARALLEL_THREAD
                if self.ensemble_config.parallel_evaluation
                else StochasticExecutionMode.SEQUENTIAL
            ),
            n_workers=self.ensemble_config.n_workers,
            chunk_size=self.ensemble_config.chunk_size,
            outlier_detection=False,
            outlier_threshold=self.ensemble_config.outlier_threshold,
            convergence_check=False,
            callable_contract=self.ensemble_config.parallel_callable_contract,
        )
        runner = StochasticEnsembleRunner(
            config=exec_config,
            device=self.device,
            parameter_manager=self.parameter_manager,
        )
        results = runner.run_ensemble(
            model_callable=scalar_model_adapter,
            parameter_samples=tensor_samples,
            fixed_inputs=fixed_inputs,
            statistics=[StatisticType.MEAN],
            store_samples=True,
            seed=self.ensemble_config.random_seed,
        )
        if not isinstance(results.samples, dict) or not results.samples:
            raise RuntimeError("EnsembleRunner returned no raw sample ledger")

        ensemble_outputs: Dict[str, np.ndarray] = {}
        for output_name, samples in results.samples.items():
            if not isinstance(samples, torch.Tensor):
                raise TypeError(
                    f"Ensemble output '{output_name}' must be a torch.Tensor"
                )
            if samples.shape != (ensemble_size,):
                raise ValueError(
                    f"Ensemble output '{output_name}' must have shape "
                    f"({ensemble_size},)"
                )
            if not bool(torch.isfinite(samples).all().item()):
                raise ValueError(
                    f"Ensemble output '{output_name}' must contain only finite values"
                )
            ensemble_outputs[output_name] = (
                samples.detach().cpu().numpy().copy()
            )

        self.model_evaluations += ensemble_size
        return ensemble_outputs
    
    def _filter_outliers(
        self, ensemble_outputs: Dict[str, np.ndarray]
    ) -> tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """Remove complete member rows using a Torch z-score criterion.

        References:
            Iglewicz and Hoaglin (1993), How to Detect and Handle Outliers.

        Mathematical Expression:
            A member is marked when ``|y_i - mean(y)| / s_y > tau`` for any
            declared output, where ``s_y`` is the sample standard deviation.

        Parameter Meanings:
            ``y_i`` is member ``i``, ``s_y`` is output spread, and ``tau`` is
            ``outlier_threshold``. The union mask is applied to every output.

        Returns:
            Tuple of (filtered_outputs, outlier_mask).  outlier_mask is None
            when no outliers were detected.
        """
        if not isinstance(ensemble_outputs, dict) or not ensemble_outputs:
            raise ValueError("ensemble_outputs must be a non-empty mapping")

        output_tensors: Dict[str, torch.Tensor] = {}
        n_original: Optional[int] = None
        for var_name, outputs in ensemble_outputs.items():
            if not isinstance(var_name, str) or not var_name.strip():
                raise ValueError("Ensemble output names must be non-empty strings")
            values = torch.as_tensor(outputs, dtype=torch.float64)
            if values.dim() != 1 or values.numel() == 0:
                raise ValueError(
                    f"Ensemble output '{var_name}' must be non-empty and one-dimensional"
                )
            if n_original is None:
                n_original = values.shape[0]
            elif values.shape[0] != n_original:
                raise ValueError("All ensemble outputs must preserve member alignment")
            output_tensors[var_name] = values
        assert n_original is not None

        outlier_mask = torch.zeros(n_original, dtype=torch.bool)
        for outputs in output_tensors.values():
            finite = torch.isfinite(outputs)
            outlier_mask |= ~finite
            finite_outputs = outputs[finite]
            if finite_outputs.numel() < 2:
                continue
            std_output = torch.std(finite_outputs, correction=1)
            if bool((std_output > 1.0e-12).item()):
                mean_output = torch.mean(finite_outputs)
                z_scores = torch.abs(outputs[finite] - mean_output) / std_output
                finite_indices = torch.nonzero(finite, as_tuple=False).flatten()
                outlier_mask[finite_indices] |= (
                    z_scores > self.ensemble_config.outlier_threshold
                )

        n_outliers = int(torch.count_nonzero(outlier_mask).item())
        if n_outliers > 0:
            if n_outliers == n_original:
                raise RuntimeError(
                    "Outlier filtering removed every ensemble member"
                )
            logger.info(f"Removed {n_outliers} outlier ensemble members")
            filtered_outputs = {
                var_name: outputs[~outlier_mask].detach().cpu().numpy().copy()
                for var_name, outputs in output_tensors.items()
            }
            return (
                filtered_outputs,
                outlier_mask.detach().cpu().numpy().copy(),
            )
        
        return ensemble_outputs, None
    
    def _check_ensemble_convergence(self, ensemble_outputs: Dict[str, np.ndarray]) -> bool:
        """Evaluate a heuristic running-mean stability diagnostic.

        References:
            Robert and Casella (2004), Monte Carlo Statistical Methods.

        Mathematical Expression:
            ``r_k = |m_k - m_(k-1)| / |m_(k-1)|`` (no absolute floor) and the
            diagnostic passes only when ``max_k r_k <= tolerance``.

        Parameter Meanings:
            ``m_k`` is a prefix mean, ``r_k`` is relative mean drift, and the
            evidence floor is ``n_min`` members
            (optimization.uncertainty.ensemble.convergence_min_members). The
            monitored prefixes start at ``N // d_start`` and advance by
            ``max(1, (N // d_start) // d_stride)`` members
            (optimization.uncertainty.ensemble.convergence_first_prefix_divisor
            and ...convergence_stride_divisor). All three are RELIC convention
            values, not the windows of Geweke (1992) or Gelman and Rubin
            (1992). This is not a proof of distributional convergence or
            effective sample size.
        """
        if not isinstance(ensemble_outputs, dict) or not ensemble_outputs:
            return False
        tolerance = torch.as_tensor(
            self.ensemble_config.convergence_tolerance,
            dtype=torch.float64,
        )
        if (
            tolerance.numel() != 1
            or not bool(torch.isfinite(tolerance).item())
            or not bool((tolerance > 0.0).item())
        ):
            raise ValueError("convergence_tolerance must be positive and finite")

        # References: Robert and Casella (2004), Monte Carlo Statistical
        #     Methods, ch. 12 (monitoring the running Monte Carlo average).
        # Mathematical expression: the diagnostic is evaluated only for N >= n_min.
        # Parameter meanings: N ensemble members of one output; n_min =
        #     optimization.uncertainty.ensemble.convergence_min_members (YAML).
        # Units: member counts (dimensionless).
        # Domain/constraints: N < n_min reports not converged (no vacuous pass).
        # Sign convention: not applicable.
        min_members = int(get_global_manager().get(
            "optimization.uncertainty.ensemble.convergence_min_members", required=True
        ))
        # References: Robert and Casella (2004), ch. 12 (running-mean monitoring);
        #     the window constants are RELIC convention values (YAML).
        # Mathematical expression: n_0 = N // d_start; s = max(1, n_0 // d_stride);
        #     monitored prefix lengths n_k = n_0 + k s < N.
        # Parameter meanings: N ensemble members; d_start =
        #     optimization.uncertainty.ensemble.convergence_first_prefix_divisor;
        #     d_stride = optimization.uncertainty.ensemble.convergence_stride_divisor.
        # Units: member counts (dimensionless).
        # Domain/constraints: d_start, d_stride >= 1 (YAML bounds); n_0 < 1 or
        #     fewer than two monitored prefixes report not converged.
        # Sign convention: not applicable.
        start_divisor = int(get_global_manager().get(
            "optimization.uncertainty.ensemble.convergence_first_prefix_divisor", required=True
        ))
        stride_divisor = int(get_global_manager().get(
            "optimization.uncertainty.ensemble.convergence_stride_divisor", required=True
        ))
        expected_size: Optional[int] = None
        for var_name, outputs in ensemble_outputs.items():
            values = torch.as_tensor(outputs, dtype=torch.float64)
            if values.dim() != 1 or values.numel() < min_members:
                return False
            if not bool(torch.isfinite(values).all().item()):
                return False
            if expected_size is None:
                expected_size = values.numel()
            elif values.numel() != expected_size:
                raise ValueError("All ensemble outputs must preserve member alignment")

            first_prefix = values.numel() // start_divisor
            if first_prefix < 1:
                return False
            step = max(1, first_prefix // stride_divisor)
            prefix_lengths = torch.arange(
                first_prefix,
                values.numel(),
                step,
                dtype=torch.int64,
            )
            if prefix_lengths.numel() < 2:
                return False
            cumulative = torch.cumsum(values, dim=0)
            running_means = (
                cumulative[prefix_lengths - 1]
                / prefix_lengths.to(dtype=torch.float64)
            )
            # References: Robert and Casella (2004), Monte Carlo Statistical
            #     Methods, ch. 12 (monitoring the running Monte Carlo average).
            # Mathematical expression: r_k = |m_k - m_(k-1)| / |m_(k-1)|.
            # Parameter meanings: m_k = mean of the first n_k members.
            # Units: m_k in output units; r_k dimensionless.
            # Domain/constraints: no absolute floor on |m_(k-1)| (a 1e-6 floor made
            #     every output with |m| << 1e-6 look converged); a zero previous
            #     mean gives r_k = 0 only for zero change, else inf.
            # Sign convention: only the magnitude of the change matters.
            changes = torch.abs(torch.diff(running_means))
            previous = torch.abs(running_means[:-1])
            relative_changes = torch.where(
                previous > 0,
                changes / torch.where(previous > 0, previous, torch.ones_like(previous)),
                torch.where(changes > 0, torch.full_like(changes, float("inf")), changes),
            )
            max_relative_change = torch.max(relative_changes)
            if bool((max_relative_change > tolerance).item()):
                logger.debug(
                    "Ensemble not converged for %s: max change = %.4f",
                    var_name,
                    max_relative_change.item(),
                )
                return False

        return expected_size is not None
