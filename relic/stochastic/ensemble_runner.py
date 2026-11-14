"""
Ensemble Execution and Statistical Aggregation for RELIC

This module provides the single numerical ensemble aggregation owner used by
RELIC. It executes an explicitly supplied callable and preserves the leading
ensemble axis. Governed RELIC simulator member isolation, stochastic
substreams, and durable restart are separate runtime obligations and are not
implied by this numerical interface.

Key Features:
    - Batch vectorized execution with PyTorch optimization
    - Parallel execution using caller-certified pure threads
    - Optional convergence diagnostics over the ensemble axis
    - NetCDF-oriented statistical aggregation metadata
    - Explicit ParameterManager dependency injection
    - Optional member-aligned outlier filtering

Mathematical Framework:
    Ensemble statistics:
        mu = (1/N) sum_i x_i (sample mean)
        sigma^2 = (1/(N-1)) sum_i (x_i - mu)^2 (sample variance)
        q_alpha = F_N^{-1}(alpha) (empirical quantile)
    
    Convergence criterion:
        |mu_j - mu_{j-1}| / max(|mu_{j-1}|, epsilon) < delta
        
    Quality control:
        Outliers: |x_i - mu| > tau * sigma where tau is the threshold

References:
    - Robert, C.P. & Casella, G. (2004). Monte Carlo Statistical Methods. Springer
    - Liu, J.S. (2001). Monte Carlo Strategies in Scientific Computing. Springer  
    - Gelman, A. et al. (2013). Bayesian Data Analysis. CRC Press, 3rd edition
    - Soize, C. (2017). Uncertainty Quantification. Springer

Authors: Yuchong Long
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import logging
import time

import numpy as np
import torch
from torch import Tensor

from .category import StochasticProcess, ProcessMetadata
from relic.parameters import get_global_manager

logger = logging.getLogger(__name__)


def quantile_label(level: Any) -> str:
    """Label of a quantile level: 'q' plus the exact percent, two-digit integer part.

    0.05 -> 'q05', 0.29 -> 'q29', 0.025 -> 'q02.5', 0.975 -> 'q97.5', 1 -> 'q100'.
    The percent is formed from the shortest decimal that round-trips the value
    at its own precision (a float32 level 0.1 is 'q10', not 'q10.000000149'),
    so distinct levels never share a label (int(level * 100) truncated 0.29
    to 28 and merged 0.97 with 0.975; '.2f' merged 0.025 with 0.03). This is
    the one naming rule for EnsembleRunner statistics and quantile scores.
    """
    if isinstance(level, torch.Tensor):
        if level.numel() != 1:
            raise ValueError("quantile level must be a scalar")
        level = level.detach().cpu().numpy().reshape(())[()]
    if isinstance(level, bool) or not isinstance(level, (int, float, np.integer, np.floating)):
        raise TypeError(f"quantile level must be a real number, got {type(level).__name__}")
    text = np.format_float_positional(level, unique=True, trim="-")
    percent = format((Decimal(text) * 100).normalize(), "f")
    integer_part, _, fraction = percent.partition(".")
    label = f"q{int(integer_part):02d}"
    return f"{label}.{fraction}" if fraction else label


class ExecutionMode(Enum):
    """Execution modes; process execution remains fail-closed [N]."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREAD = "parallel_thread" 
    PARALLEL_PROCESS = "parallel_process"
    VECTORIZED = "vectorized"


class CallableContract(Enum):
    """Caller certification used by the canonical ensemble execution owner."""

    UNSPECIFIED = "unspecified"
    DETERMINISTIC_PURE = "deterministic_pure"


class StatisticType(Enum):
    """Statistical aggregation types.""" 
    MEAN = "mean"
    STD = "std"
    VAR = "var"
    QUANTILE = "quantile"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    IQR = "iqr"


@dataclass
class ExecutionConfig:
    """
    Configuration for ensemble execution.
    
    Parameters:
        mode (ExecutionMode): Execution strategy
        callable_contract (CallableContract): Explicit callable side-effect contract
        n_workers (int): Number of parallel workers
        chunk_size (int): Batch size for vectorized execution
        outlier_detection (bool): Enable member-aligned outlier removal
        outlier_threshold (Optional[float]): Z-score threshold for outliers;
            None reads stochastic.ensemble.outlier_threshold
        convergence_check (bool): Enable convergence monitoring
        convergence_window (Optional[int]): Window size for convergence
            assessment; None reads stochastic.ensemble.convergence_window
        convergence_tolerance (Optional[float]): Relative tolerance for
            convergence; None reads stochastic.ensemble.convergence_tolerance
    """
    mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    callable_contract: CallableContract = CallableContract.UNSPECIFIED
    n_workers: int = 4
    chunk_size: int = 100
    outlier_detection: bool = False
    outlier_threshold: Optional[float] = None
    convergence_check: bool = False
    convergence_window: Optional[int] = None
    convergence_tolerance: Optional[float] = None

    def __post_init__(self) -> None:
        """Resolve unstated settings from YAML, then reject invalid contracts."""
        # The parameter database (stochastic.ensemble.*) is the single source
        # of every screening and convergence setting a caller does not state.
        manager = get_global_manager()
        if self.outlier_threshold is None:
            self.outlier_threshold = float(manager.get(
                "stochastic.ensemble.outlier_threshold", required=True
            ))
        if self.convergence_window is None:
            self.convergence_window = int(manager.get(
                "stochastic.ensemble.convergence_window", required=True
            ))
        if self.convergence_tolerance is None:
            self.convergence_tolerance = float(manager.get(
                "stochastic.ensemble.convergence_tolerance", required=True
            ))
        if not isinstance(self.mode, ExecutionMode):
            raise TypeError("mode must be an ExecutionMode")
        if not isinstance(self.callable_contract, CallableContract):
            raise TypeError("callable_contract must be a CallableContract")
        for field_name in ("n_workers", "chunk_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.convergence_window, bool)
            or not isinstance(self.convergence_window, int)
            or self.convergence_window < 2
        ):
            raise ValueError("convergence_window must be an integer >= 2")
        for field_name in ("outlier_threshold", "convergence_tolerance"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a positive finite number")
            scalar = torch.as_tensor(value, dtype=torch.float64)
            if not bool(torch.isfinite(scalar).item()) or float(value) <= 0.0:
                raise ValueError(f"{field_name} must be a positive finite number")
        if not isinstance(self.outlier_detection, bool):
            raise TypeError("outlier_detection must be boolean")
        if not isinstance(self.convergence_check, bool):
            raise TypeError("convergence_check must be boolean")


@dataclass
class EnsembleResults:
    """
    Container for numerical ensemble results and explicit export ledgers.
    
    Parameters:
        statistics (Dict[str, Tensor]): Statistical summaries by variable
        samples (Optional[Dict[str, Tensor]]): Raw ensemble samples (if stored)
        metadata (Dict[str, Any]): Provenance and caller-supplied export ledgers
        execution_info (Dict[str, Any]): Runtime execution information
        convergence_info (Optional[Dict[str, Any]]): Convergence diagnostics
        outliers_removed (Dict[str, int]): Count of outliers removed per variable
        
    Export Contract:
        NetCDF export requires ``metadata['variable_dimensions']`` for every
        statistic. Optional variable attributes must be supplied through
        ``metadata['variable_attributes']``. The numerical runner never
        invents coordinate names, units, standard names, or CF compliance.
    """
    statistics: Dict[str, Tensor]
    samples: Optional[Dict[str, Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_info: Dict[str, Any] = field(default_factory=dict)
    convergence_info: Optional[Dict[str, Any]] = None
    outliers_removed: Dict[str, int] = field(default_factory=dict)

    def to_netcdf_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert results using caller-supplied dimensions and attributes.

        Returns:
            Dict: NetCDF variable specifications with data, attributes, dimensions

        Raises:
            ValueError: If an explicit dimension or attribute ledger is invalid
            TypeError: If a statistic is not a tensor
        """
        dimension_ledger = self.metadata.get("variable_dimensions")
        if not isinstance(dimension_ledger, dict):
            raise ValueError(
                "NetCDF export requires metadata['variable_dimensions']"
            )
        attribute_ledger = self.metadata.get("variable_attributes", {})
        if not isinstance(attribute_ledger, dict):
            raise ValueError("metadata['variable_attributes'] must be a mapping")

        variables: Dict[str, Dict[str, Any]] = {}
        for var_name, data in self.statistics.items():
            if not isinstance(data, Tensor):
                raise TypeError(f"Statistic '{var_name}' must be a torch.Tensor")
            dimensions = dimension_ledger.get(var_name)
            if not isinstance(dimensions, (list, tuple)):
                raise ValueError(
                    f"variable_dimensions must declare statistic '{var_name}'"
                )
            if len(dimensions) != data.dim():
                raise ValueError(
                    f"Statistic '{var_name}' has rank {data.dim()} but declares "
                    f"{len(dimensions)} dimensions"
                )
            if any(not isinstance(name, str) or not name.strip() for name in dimensions):
                raise ValueError(
                    f"Statistic '{var_name}' dimensions must be non-empty strings"
                )
            if len(set(dimensions)) != len(dimensions):
                raise ValueError(
                    f"Statistic '{var_name}' dimensions must be unique"
                )

            attributes = attribute_ledger.get(var_name, {})
            if not isinstance(attributes, dict):
                raise ValueError(
                    f"variable_attributes['{var_name}'] must be a mapping"
                )

            variables[var_name] = {
                "data": data.detach().cpu().numpy(),
                "attrs": dict(attributes),
                "dims": list(dimensions),
            }

        return variables


class EnsembleRunner(StochasticProcess):
    """
    Numerical ensemble execution engine with statistical aggregation.
    
    Provides batch execution of an explicit callable with statistical
    aggregation and optional quality control. This class is not a second RELIC
    simulator, StateManager, ParameterManager, checkpoint owner, or stochastic
    runtime.
    
    Mathematical Framework:
        Ensemble forward model: Y = {f(theta_i, x, t) : i = 1,...,N}
        Statistical moments: E[Y], Var[Y], quantiles Q_alpha[Y]
        Quality control: remove complete members when |y_i - mu| > tau sigma
        
    Supported numerical modes:
        - Sequential execution
        - Batched vectorized execution for genuinely batch-aware callables
        - Thread execution for caller-certified pure callables

    Non-operational capability:
        Process execution is [N] until a Windows-tested pickling, worker-state,
        cancellation, and stochastic-substream contract is executable.
    """
    
    def __init__(self,
                 config: Optional[ExecutionConfig] = None,
                 device: torch.device = torch.device("cpu"),
                 parameter_manager=None):
        """
        Initialize ensemble runner.

        Parameters:
            config (Optional[ExecutionConfig]): Execution configuration
            device (torch.device): Computation device (CPU/GPU)
            parameter_manager: Optional parameter manager instance (dependency injection)
        """
        # Initialize metadata
        metadata = ProcessMetadata(
            name="EnsembleRunner",
            description="Batch ensemble execution with statistical aggregation",
            units={"statistics": "various", "convergence": "dimensionless"},
            parameters={"device": str(device)},
            references="Robert & Casella (2004), Gelman et al. (2013)"
        )
        super().__init__(metadata)

        self.config = config or ExecutionConfig()
        self.device = device
        self.parameter_manager = parameter_manager  # Store injected parameter manager

        # Execution state
        self.execution_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'execution_time': 0.0
        }
        self._last_parallel_metadata: Optional[Dict[str, Any]] = None

        # Convergence monitoring
        self.convergence_history = {}
        logger.info(
            "EnsembleRunner initialized: %s mode, device=%s",
            self.config.mode.value,
            device,
        )
    
    def sample(self,
               input_state: Tensor,
               num_samples: int = 1,
               seed: Optional[int] = None,
               **kwargs) -> Dict[str, Tensor]:
        """
        Interface for StochasticProcess consistency (not used directly).

        The numerical owner has no standalone stochastic sampling law. Callers
        must use ``run_ensemble`` with an explicit model callable and ledger.
        """
        raise NotImplementedError(
            "EnsembleRunner.sample is undefined; use run_ensemble with an "
            "explicit callable and parameter ledger"
        )
    
    def log_probability(self, input_state: Tensor, output_state: Tensor, **kwargs) -> Tensor:
        """Not applicable for ensemble runner."""
        raise NotImplementedError("EnsembleRunner does not compute log probabilities")
    
    def moments(self, input_state: Tensor, **kwargs) -> Dict[str, Tensor]:
        """Not applicable for ensemble runner."""
        raise NotImplementedError("Use run_ensemble() to compute ensemble moments")
    
    def run_ensemble(self,
                    model_callable: Callable[[Dict[str, Any]], Dict[str, Tensor]],
                    parameter_samples: Dict[str, Tensor],
                    fixed_inputs: Optional[Dict[str, Any]] = None,
                    statistics: List[StatisticType] = None,
                    quantile_levels: Optional[List[float]] = None,
                    store_samples: bool = False,
                    seed: Optional[int] = None) -> EnsembleResults:
        """
        Execute ensemble simulations with statistical aggregation.
        
        Parameters:
            model_callable (Callable): Model function to execute
                Signature: f(inputs: Dict[str, Any]) -> Dict[str, Tensor]
            parameter_samples (Dict[str, Tensor]): Parameter samples [num_samples, ...]
            fixed_inputs (Optional[Dict]): Fixed inputs for all ensemble members
            statistics (List[StatisticType]): Statistics to compute
            quantile_levels (Optional[List[float]]): Quantile levels (0-1)
            store_samples (bool): Whether to store raw samples
            seed (Optional[int]): Caller-owned substream identifier recorded as
                provenance. The runner never reseeds process-global RNG state.
            
        Returns:
            EnsembleResults: Aggregated statistics and metadata
            
        Mathematical Expression:
            For each output variable y:
                mu = (1/N) sum_i y_i
                sigma^2 = (1/(N-1)) sum_i (y_i - mu)^2
                q_alpha = inf{x : F_N(x) >= alpha} where F_N is empirical CDF
        """
        start_time = time.time()

        num_samples = self._validate_run_contract(
            model_callable=model_callable,
            parameter_samples=parameter_samples,
            fixed_inputs=fixed_inputs,
        )

        if statistics is None:
            statistics = [StatisticType.MEAN, StatisticType.STD, StatisticType.QUANTILE]
        if quantile_levels is None:
            quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        self._validate_statistics_contract(statistics, quantile_levels)
        self._reset_run_state(num_samples)

        logger.info(
            "Running ensemble: %s samples, %s mode",
            num_samples,
            self.config.mode.value,
        )

        if self.config.mode == ExecutionMode.VECTORIZED:
            raw_results = self._run_vectorized(model_callable, parameter_samples, fixed_inputs)
        elif self.config.mode in {
            ExecutionMode.PARALLEL_THREAD,
            ExecutionMode.PARALLEL_PROCESS,
        }:
            raw_results = self._run_parallel(
                model_callable,
                parameter_samples,
                fixed_inputs,
            )
        else:
            raw_results = self._run_sequential(
                model_callable,
                parameter_samples,
                fixed_inputs,
            )

        if self.config.outlier_detection:
            raw_results, outliers_removed = self._remove_outliers(raw_results)
        else:
            outliers_removed = {}

        aggregated_stats = self._compute_statistics(
            raw_results,
            statistics,
            quantile_levels,
        )
        convergence_info = (
            self._check_convergence(raw_results)
            if self.config.convergence_check
            else None
        )

        execution_time = time.time() - start_time
        self.execution_stats['execution_time'] = execution_time
        if self._last_parallel_metadata is not None:
            self.execution_stats.update(self._last_parallel_metadata)
        metadata = self._build_metadata(parameter_samples)
        metadata['substream_identifier'] = seed

        results = EnsembleResults(
            statistics=aggregated_stats,
            samples=raw_results if store_samples else None,
            metadata=metadata,
            execution_info=self.execution_stats.copy(),
            convergence_info=convergence_info,
            outliers_removed=outliers_removed,
        )

        logger.info(
            "Ensemble completed: %s/%s successful in %.2fs",
            self.execution_stats['successful_runs'],
            num_samples,
            execution_time,
        )
        return results

    def _validate_run_contract(
        self,
        model_callable: Callable[[Dict[str, Any]], Dict[str, Tensor]],
        parameter_samples: Dict[str, Tensor],
        fixed_inputs: Optional[Dict[str, Any]],
    ) -> int:
        """Validate the ensemble ledger before any model evaluation."""
        if not callable(model_callable):
            raise TypeError("model_callable must be callable")
        if (
            self.config.mode
            in {ExecutionMode.PARALLEL_THREAD, ExecutionMode.PARALLEL_PROCESS}
            and self.config.callable_contract
            is not CallableContract.DETERMINISTIC_PURE
        ):
            raise NotImplementedError(
                "[N] Parallel ensemble execution requires "
                "callable_contract=deterministic_pure; this certification means "
                "the callable is deterministic, reentrant, thread-safe, and does "
                "not mutate runtime state or consume process-global RNG streams"
            )
        if self.config.mode is ExecutionMode.PARALLEL_PROCESS:
            if self.device.type != "cpu":
                raise NotImplementedError(
                    "[N] Process ensemble execution currently requires a CPU runner"
                )
            import pickle

            try:
                pickle.dumps(model_callable)
            except (pickle.PickleError, AttributeError, TypeError) as exc:
                raise TypeError(
                    "Process ensemble model_callable must be importable and picklable"
                ) from exc
        if not isinstance(parameter_samples, dict) or not parameter_samples:
            raise ValueError("Parameter samples must be a non-empty mapping")
        if fixed_inputs is not None and not isinstance(fixed_inputs, dict):
            raise TypeError("fixed_inputs must be a mapping when provided")
        if self.config.mode is ExecutionMode.PARALLEL_PROCESS:
            for name, value in (fixed_inputs or {}).items():
                if isinstance(value, Tensor) and value.device.type != "cpu":
                    raise ValueError(
                        f"fixed_inputs['{name}'] must be on CPU for process mode"
                    )
            try:
                pickle.dumps(fixed_inputs)
            except (pickle.PickleError, AttributeError, TypeError) as exc:
                raise TypeError(
                    "Process ensemble fixed_inputs must be serializable values"
                ) from exc

        ensemble_size: Optional[int] = None
        for name, samples in parameter_samples.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Parameter sample names must be non-empty strings")
            if not isinstance(samples, Tensor):
                raise TypeError(
                    f"parameter_samples['{name}'] must be a torch.Tensor"
                )
            if (
                self.config.mode is ExecutionMode.PARALLEL_PROCESS
                and samples.device.type != "cpu"
            ):
                raise ValueError(
                    f"parameter_samples['{name}'] must be on CPU for process mode"
                )
            if samples.dim() == 0:
                raise ValueError(
                    f"parameter_samples['{name}'] must have at least one dimension"
                )
            if samples.shape[0] == 0:
                raise ValueError(
                    f"parameter_samples['{name}'] must contain at least one member"
                )
            if not bool(torch.isfinite(samples).all().item()):
                raise ValueError(
                    f"parameter_samples['{name}'] must contain only finite values"
                )
            if ensemble_size is None:
                ensemble_size = samples.shape[0]
            elif samples.shape[0] != ensemble_size:
                raise ValueError(
                    "All parameter samples must have the same ensemble size"
                )

        overlap = set(parameter_samples).intersection(fixed_inputs or {})
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                f"fixed_inputs and parameter_samples overlap for: {names}"
            )
        assert ensemble_size is not None
        return ensemble_size

    @staticmethod
    def _validate_statistics_contract(
        statistics: List[StatisticType],
        quantile_levels: List[float],
    ) -> None:
        """Validate requested statistics and empirical quantile levels."""
        if not isinstance(statistics, list) or not statistics:
            raise ValueError("statistics must be a non-empty list")
        if any(not isinstance(statistic, StatisticType) for statistic in statistics):
            raise TypeError("statistics entries must be StatisticType values")
        if StatisticType.QUANTILE in statistics:
            if not isinstance(quantile_levels, list) or not quantile_levels:
                raise ValueError("quantile_levels must be non-empty for quantiles")
            for level in quantile_levels:
                if not isinstance(level, (int, float)) or not 0.0 <= float(level) <= 1.0:
                    raise ValueError("quantile levels must lie in [0, 1]")
            names = [quantile_label(level) for level in quantile_levels]
            if len(set(names)) != len(names):
                raise ValueError(
                    f"quantile levels must be distinct, got {quantile_levels}"
                )


    def _reset_run_state(self, num_samples: int) -> None:
        """Reset per-run counters so evidence never leaks across executions."""
        self.execution_stats = {
            'total_runs': num_samples,
            'successful_runs': 0,
            'failed_runs': 0,
            'execution_time': 0.0,
        }
        self.convergence_history = {}
        self._last_parallel_metadata = None
    
    def _run_vectorized(self,
                       model_callable: Callable,
                       parameter_samples: Dict[str, Tensor],
                       fixed_inputs: Optional[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Run ensemble using vectorized model evaluation.
        
        This is the most efficient mode for models that support batch processing.
        """
        logger.debug("Starting vectorized ensemble execution")
        
        num_samples = next(iter(parameter_samples.values())).shape[0]
        chunk_size = min(self.config.chunk_size, num_samples)
        if chunk_size <= 0:
            raise ValueError("Vectorized chunk_size must be positive")

        chunks: Dict[str, List[Tensor]] = {}
        expected_keys: Optional[Tuple[str, ...]] = None
        expected_shapes: Dict[str, torch.Size] = {}
        expected_dtypes: Dict[str, torch.dtype] = {}

        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            chunk_size_actual = end_idx - start_idx

            chunk_inputs = fixed_inputs.copy() if fixed_inputs else {}
            for key, samples in parameter_samples.items():
                chunk_inputs[key] = samples[start_idx:end_idx]

            try:
                chunk_outputs = model_callable(chunk_inputs)
            except Exception as exc:
                self.execution_stats['failed_runs'] += chunk_size_actual
                raise RuntimeError(
                    f"Ensemble vectorized chunk {start_idx}:{end_idx} failed: {exc}"
                ) from exc

            keys = self._validate_output_mapping(
                chunk_outputs,
                context=f"vectorized chunk {start_idx}:{end_idx}",
            )
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(
                    f"Model output keys changed in vectorized chunk {start_idx}:{end_idx}"
                )

            for var_name in expected_keys:
                values = chunk_outputs[var_name]
                if values.dim() == 0 or values.shape[0] != chunk_size_actual:
                    raise ValueError(
                        f"Vectorized output '{var_name}' must preserve the leading "
                        f"chunk axis of size {chunk_size_actual}"
                    )
                trailing_shape = values.shape[1:]
                if var_name not in expected_shapes:
                    expected_shapes[var_name] = trailing_shape
                    expected_dtypes[var_name] = values.dtype
                    chunks[var_name] = []
                elif trailing_shape != expected_shapes[var_name]:
                    raise ValueError(
                        f"Vectorized output '{var_name}' trailing shape changed in "
                        f"chunk {start_idx}:{end_idx}"
                    )
                elif values.dtype != expected_dtypes[var_name]:
                    raise ValueError(
                        f"Vectorized output '{var_name}' dtype changed in chunk "
                        f"{start_idx}:{end_idx}"
                    )
                chunks[var_name].append(values.to(self.device))

            self.execution_stats['successful_runs'] += chunk_size_actual

        assert expected_keys is not None
        return {
            var_name: torch.cat(chunks[var_name], dim=0)
            for var_name in expected_keys
        }
    
    def _run_parallel(self,
                     model_callable: Callable,
                     parameter_samples: Dict[str, Tensor],
                     fixed_inputs: Optional[Dict[str, Any]]) -> Dict[str, Tensor]:
        """Run caller-certified pure members through the canonical executor."""
        from ..computation.parallel.executor import (
            ParallelBackend,
            ParallelConfig,
            ParallelExecutor,
        )

        num_samples = next(iter(parameter_samples.values())).shape[0]
        param_sets = [
            self._member_inputs(parameter_samples, fixed_inputs, index)
            for index in range(num_samples)
        ]
        backend = (
            ParallelBackend.PROCESSES
            if self.config.mode is ExecutionMode.PARALLEL_PROCESS
            else ParallelBackend.THREADS
        )
        executor = ParallelExecutor(
            ParallelConfig(
                backend=backend,
                max_workers=self.config.n_workers,
                chunk_size=1,
                allow_sequential=False,
            )
        )
        try:
            member_results = executor.parallel_map(
                model_callable,
                param_sets,
                complexity="high",
            )
        except Exception as exc:
            self._last_parallel_metadata = executor.get_last_execution_metadata()
            actual_count = int(
                self._last_parallel_metadata.get("actual_count", 0)
            )
            self.execution_stats['successful_runs'] = actual_count
            self.execution_stats['failed_runs'] = max(
                1,
                num_samples - actual_count,
            )
            raise RuntimeError(
                f"Ensemble {self.config.mode.value} execution failed: {exc}"
            ) from exc

        self._last_parallel_metadata = executor.get_last_execution_metadata()
        if len(member_results) != num_samples:
            self.execution_stats['successful_runs'] = len(member_results)
            self.execution_stats['failed_runs'] = num_samples - len(member_results)
            raise RuntimeError(
                "Parallel ensemble returned an incomplete member ledger"
            )
        self.execution_stats['successful_runs'] = num_samples
        return self._stack_member_outputs(member_results)

    def _member_inputs(
        self,
        parameter_samples: Dict[str, Tensor],
        fixed_inputs: Optional[Dict[str, Any]],
        index: int,
    ) -> Dict[str, Any]:
        """Build one member input without retaining the ensemble dimension."""
        inputs = fixed_inputs.copy() if fixed_inputs else {}
        inputs.update(
            {
                name: samples[index]
                for name, samples in parameter_samples.items()
            }
        )
        return inputs
    
    def _run_sequential(self,
                       model_callable: Callable,
                       parameter_samples: Dict[str, Tensor],
                       fixed_inputs: Optional[Dict[str, Any]]) -> Dict[str, Tensor]:
        """Run members sequentially and fail on the first invalid member."""
        logger.debug("Starting sequential ensemble execution")

        num_samples = next(iter(parameter_samples.values())).shape[0]
        member_results: List[Dict[str, Tensor]] = []
        for index in range(num_samples):
            inputs = self._member_inputs(parameter_samples, fixed_inputs, index)
            try:
                result = model_callable(inputs)
            except Exception as exc:
                self.execution_stats['failed_runs'] += 1
                raise RuntimeError(
                    f"Ensemble member {index} failed: {exc}"
                ) from exc
            member_results.append(result)
            self.execution_stats['successful_runs'] += 1

        return self._stack_member_outputs(member_results)

    @staticmethod
    def _validate_output_mapping(
        result: Any,
        context: str,
    ) -> Tuple[str, ...]:
        """Validate one output ledger without altering its values or shapes."""
        if not isinstance(result, dict) or not result:
            raise ValueError(f"Model output for {context} must be a non-empty mapping")

        keys: List[str] = []
        for var_name, values in result.items():
            if not isinstance(var_name, str) or not var_name:
                raise ValueError(
                    f"Model output keys for {context} must be non-empty strings"
                )
            if not isinstance(values, Tensor):
                raise TypeError(
                    f"Model output '{var_name}' for {context} must be a torch.Tensor"
                )
            if values.numel() == 0:
                raise ValueError(
                    f"Model output '{var_name}' for {context} must be non-empty"
                )
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(
                    f"Model output '{var_name}' for {context} must contain only finite values"
                )
            keys.append(var_name)
        return tuple(sorted(keys))

    def _stack_member_outputs(
        self,
        member_results: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Stack validated member results on exactly one leading sample axis."""
        expected_keys: Optional[Tuple[str, ...]] = None
        expected_shapes: Dict[str, torch.Size] = {}
        expected_dtypes: Dict[str, torch.dtype] = {}
        values_by_name: Dict[str, List[Tensor]] = {}

        for index, result in enumerate(member_results):
            keys = self._validate_output_mapping(result, context=f"member {index}")
            if expected_keys is None:
                expected_keys = keys
                for var_name in keys:
                    expected_shapes[var_name] = result[var_name].shape
                    expected_dtypes[var_name] = result[var_name].dtype
                    values_by_name[var_name] = []
            elif keys != expected_keys:
                raise ValueError(f"Model output keys changed for member {index}")

            assert expected_keys is not None
            for var_name in expected_keys:
                values = result[var_name]
                if values.shape != expected_shapes[var_name]:
                    raise ValueError(
                        f"Model output '{var_name}' shape changed for member {index}"
                    )
                if values.dtype != expected_dtypes[var_name]:
                    raise ValueError(
                        f"Model output '{var_name}' dtype changed for member {index}"
                    )
                values_by_name[var_name].append(values.to(self.device))

        if expected_keys is None:
            raise ValueError("Ensemble execution produced no member outputs")
        return {
            var_name: torch.stack(values_by_name[var_name], dim=0)
            for var_name in expected_keys
        }
    
    def _remove_outliers(self, raw_results: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, int]]:
        """
        Remove complete ensemble members using an aligned z-score ledger.

        For every output coordinate, z_i = |x_i - mean(x)| / std(x). A member
        is removed from every output when any declared coordinate exceeds the
        configured threshold. This preserves cross-variable member alignment.
        """
        if not raw_results:
            raise ValueError("Outlier detection requires non-empty results")
        ensemble_size = next(iter(raw_results.values())).shape[0]
        combined_mask = torch.zeros(
            ensemble_size,
            dtype=torch.bool,
            device=next(iter(raw_results.values())).device,
        )
        outliers_removed: Dict[str, int] = {}

        for var_name, values in raw_results.items():
            if values.shape[0] != ensemble_size:
                raise ValueError("All results must share the same ensemble axis")
            if ensemble_size < 3:
                outliers_removed[var_name] = 0
                continue

            mean_value = values.mean(dim=0)
            std_value = values.std(dim=0, unbiased=True)
            positive_scale = std_value > torch.finfo(values.dtype).eps
            z_scores = torch.where(
                positive_scale,
                torch.abs(values - mean_value) / torch.where(
                    positive_scale,
                    std_value,
                    torch.ones_like(std_value),
                ),
                torch.zeros_like(values),
            )
            member_mask = (z_scores > self.config.outlier_threshold).reshape(
                ensemble_size,
                -1,
            ).any(dim=1)
            outliers_removed[var_name] = int(member_mask.sum().item())
            combined_mask = combined_mask | member_mask.to(combined_mask.device)

        keep_mask = ~combined_mask
        if not bool(keep_mask.any().item()):
            raise ValueError("Outlier filtering removed every ensemble member")
        return (
            {
                var_name: values[keep_mask.to(values.device)]
                for var_name, values in raw_results.items()
            },
            outliers_removed,
        )
    
    def _compute_statistics(self,
                           results: Dict[str, Tensor],
                           statistics: List[StatisticType],
                           quantile_levels: List[float]) -> Dict[str, Tensor]:
        """
        Compute statistical aggregations along the ensemble axis.
        
        Mathematical Expression:
            Sample mean: mu = (1/N) sum_i x_i
            Sample std: sigma = sqrt[(1/(N-1)) sum_i (x_i - mu)^2]
            Quantile: q_alpha = inf{x : F_N(x) >= alpha}
        """
        aggregated: Dict[str, Tensor] = {}

        for var_name, values in results.items():
            if values.dim() == 0 or values.shape[0] == 0:
                raise ValueError(
                    f"Result '{var_name}' must preserve a non-empty ensemble axis"
                )
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"Result '{var_name}' contains non-finite values")

            for stat_type in statistics:
                if stat_type == StatisticType.MEAN:
                    stat_name = f"{var_name}_mean"
                    aggregated[stat_name] = values.mean(dim=0)

                # The unbiased sample spread needs at least two members; for a
                # single member it is omitted instead of reported as zero.
                elif stat_type == StatisticType.STD:
                    if values.shape[0] > 1:
                        aggregated[f"{var_name}_std"] = values.std(dim=0, unbiased=True)

                elif stat_type == StatisticType.VAR:
                    if values.shape[0] > 1:
                        aggregated[f"{var_name}_var"] = values.var(dim=0, unbiased=True)

                elif stat_type == StatisticType.MIN:
                    stat_name = f"{var_name}_min"
                    aggregated[stat_name] = values.amin(dim=0)

                elif stat_type == StatisticType.MAX:
                    stat_name = f"{var_name}_max"
                    aggregated[stat_name] = values.amax(dim=0)

                elif stat_type == StatisticType.MEDIAN:
                    stat_name = f"{var_name}_median"
                    aggregated[stat_name] = values.median(dim=0).values

                elif stat_type == StatisticType.IQR:
                    stat_name = f"{var_name}_iqr"
                    q75 = torch.quantile(values, 0.75, dim=0)
                    q25 = torch.quantile(values, 0.25, dim=0)
                    aggregated[stat_name] = q75 - q25

                elif stat_type == StatisticType.QUANTILE:
                    for level in quantile_levels:
                        stat_name = f"{var_name}_{quantile_label(level)}"
                        aggregated[stat_name] = torch.quantile(
                            values,
                            level,
                            dim=0,
                        )

        return aggregated
    
    def _check_convergence(self, results: Dict[str, Tensor]) -> Dict[str, Any]:
        """Check convergence of ensemble statistics."""
        convergence_info = {}
        window_size = self.config.convergence_window
        tolerance = self.config.convergence_tolerance
        # Running means at n_1, n_1 + s, ..., N with s = max(1, n_1 // d);
        # d = stochastic.ensemble.convergence_stride_divisor (RELIC convention).
        stride_divisor = int(get_global_manager().get(
            "stochastic.ensemble.convergence_stride_divisor", required=True
        ))

        for var_name, values in results.items():
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(
                    f"Convergence input '{var_name}' contains non-finite values"
                )
            sample_count = values.shape[0]

            if sample_count < window_size:
                convergence_info[var_name] = {'converged': False, 'reason': 'insufficient_samples'}
                continue

            running_means = []
            stop_indices = list(
                range(
                    window_size,
                    sample_count + 1,
                    max(1, window_size // stride_divisor),
                )
            )
            if stop_indices[-1] != sample_count:
                stop_indices.append(sample_count)
            for stop_index in stop_indices:
                running_mean = values[:stop_index].mean(dim=0)
                running_means.append(running_mean)

            if len(running_means) < 2:
                convergence_info[var_name] = {'converged': False, 'reason': 'insufficient_windows'}
                continue

            running_means_tensor = torch.stack(running_means)
            # References: Robert and Casella (2004), Monte Carlo Statistical
            #     Methods, ch. 12 (monitoring the running Monte Carlo average).
            # Mathematical expression: r_k = |m_k - m_(k-1)| / |m_(k-1)|,
            #     converged when max_k r_k < tol.
            # Parameter meanings: m_k = running mean over the first n_k members,
            #     tol = convergence_tolerance.
            # Units: m_k in the output units; r_k and tol are dimensionless.
            # Domain/constraints: no absolute floor on |m_(k-1)| (a 1e-6 floor
            #     made every output with |m| << 1e-6 look converged); a zero
            #     previous mean gives r_k = 0 only for zero change, else inf.
            # Sign convention: only the magnitude of the change matters.
            changes = torch.abs(torch.diff(running_means_tensor, dim=0))
            previous = torch.abs(running_means_tensor[:-1])
            rel_changes = torch.where(
                previous > 0,
                changes / torch.where(previous > 0, previous, torch.ones_like(previous)),
                torch.where(changes > 0, torch.full_like(changes, float("inf")), changes),
            )
            max_rel_change = rel_changes.max().item()

            converged = max_rel_change < tolerance
            convergence_info[var_name] = {
                'converged': converged,
                'max_rel_change': max_rel_change,
                'n_windows': len(running_means)
            }
        
        all_converged = bool(convergence_info) and all(
            info.get('converged', False)
            for info in convergence_info.values()
        )
        convergence_info['overall_converged'] = all_converged
        
        return convergence_info
    
    def _build_metadata(self, parameter_samples: Dict[str, Tensor]) -> Dict[str, Any]:
        """Build metadata without resolving hidden global dependencies."""
        metadata = {
            'title': 'Ensemble simulation results',
            'source': 'RELIC stochastic ensemble runner',
            'metadata_profile': 'explicit-ledger',
            'institution': 'Yuchong Long',
            'ensemble_size': len(next(iter(parameter_samples.values()))),
            'execution_mode': self.config.mode.value,
            'created': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        }
        
        # Add parameter information
        param_info = {}
        for param_name, samples in parameter_samples.items():
            detached = samples.detach()
            param_info[param_name] = {
                'mean': detached.mean().item(),
                'std': detached.std(unbiased=False).item(),
                'min': detached.amin().item(),
                'max': detached.amax().item(),
            }

            if self.parameter_manager is not None:
                parameter_table = getattr(
                    self.parameter_manager,
                    'parameters',
                    None,
                )
                param_spec = (
                    parameter_table.get(param_name)
                    if parameter_table is not None
                    else None
                )
                if param_spec is not None:
                    param_info[param_name]['units'] = param_spec.units
                    param_info[param_name]['description'] = param_spec.description

        metadata['parameter_statistics'] = param_info
        return metadata
