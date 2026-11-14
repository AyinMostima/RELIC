"""
Ensemble execution helpers for RELICSimulator.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from ...parameters import get_global_manager
from .checkpoint import (
    _collect_parameter_managers,
    capture_runtime_transaction,
    restore_runtime_transaction,
)

logger = logging.getLogger(__name__)


def sample_parameters(
    distributions: Dict[str, Any],
    num_samples: int,
    seed: Optional[int] = None,
) -> Dict[str, Tensor]:
    """
    Sample declared parameter distributions with a local Torch RNG substream.

    References:
        Robert, C. P. and Casella, G. (2004). Monte Carlo Statistical
        Methods, 2nd ed., Springer.

    Mathematical expressions:
        Normal: X = mu + sigma Z, Z ~ N(0, 1), sigma >= 0.
        Lognormal: sigma_l^2 = log(1 + (s / m)^2) and
        mu_l = log(m) - sigma_l^2 / 2, with m > 0 and s >= 0.
        Uniform: X = a + (b - a) U, U ~ U(0, 1), with a < b.

    Parameter meanings:
        ``mean`` and ``std`` are the arithmetic mean and standard deviation;
        ``low`` and ``high`` are finite uniform bounds; ``value`` is a finite
        deterministic value. ``seed`` selects a reproducible local substream
        and never mutates Python, NumPy, or Torch process-global RNG state.
    """
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if seed is not None and (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or seed > 2**63 - 1
    ):
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    if not isinstance(distributions, dict):
        raise ValueError("distributions must be a mapping of parameter specifications")

    generator = torch.Generator(device="cpu")
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    samples: Dict[str, Tensor] = {}
    dtype = torch.get_default_dtype()
    required_fields = {
        "normal": ("mean", "std"),
        "lognormal": ("mean", "std"),
        "uniform": ("low", "high"),
        "constant": ("value",),
    }

    def required_scalar(
        param_name: str,
        dist_spec: Dict[str, Any],
        field_name: str,
    ) -> Tensor:
        try:
            value = torch.as_tensor(dist_spec[field_name], dtype=dtype, device="cpu")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Distribution '{param_name}' field '{field_name}' must be a finite scalar"
            ) from exc
        if value.numel() != 1:
            raise ValueError(
                f"Distribution '{param_name}' field '{field_name}' must be a finite scalar"
            )
        value = value.reshape(())
        if not bool(torch.isfinite(value).item()):
            raise ValueError(
                f"Distribution '{param_name}' field '{field_name}' must be finite"
            )
        return value

    for param_name, dist_spec in distributions.items():
        if not isinstance(param_name, str) or not param_name.strip():
            raise ValueError("Distribution parameter names must be non-empty strings")
        if not isinstance(dist_spec, dict):
            raise ValueError(f"Distribution '{param_name}' requires a mapping specification")
        if "type" not in dist_spec:
            raise ValueError(f"Distribution '{param_name}' requires explicit field 'type'")

        dist_type = dist_spec["type"]
        if dist_type not in required_fields:
            raise ValueError(
                f"Unsupported distribution '{dist_type}' for parameter '{param_name}'"
            )
        missing_fields = [
            field_name
            for field_name in required_fields[dist_type]
            if field_name not in dist_spec
        ]
        if missing_fields:
            fields = ", ".join(missing_fields)
            raise ValueError(
                f"Distribution '{param_name}' of type '{dist_type}' requires fields: {fields}"
            )

        if dist_type == "normal":
            mean = required_scalar(param_name, dist_spec, "mean")
            std = required_scalar(param_name, dist_spec, "std")
            if bool((std < 0).item()):
                raise ValueError(f"Distribution '{param_name}' std must be non-negative")
            standard_normal = torch.randn(
                num_samples,
                generator=generator,
                dtype=dtype,
                device="cpu",
            )
            draws = mean + std * standard_normal

        elif dist_type == "lognormal":
            mean = required_scalar(param_name, dist_spec, "mean")
            std = required_scalar(param_name, dist_spec, "std")
            if bool((mean <= 0).item()):
                raise ValueError(f"Distribution '{param_name}' mean must be positive")
            if bool((std < 0).item()):
                raise ValueError(f"Distribution '{param_name}' std must be non-negative")
            log_variance = torch.log1p(torch.square(std / mean))
            log_mean = torch.log(mean) - 0.5 * log_variance
            log_std = torch.sqrt(log_variance)
            standard_normal = torch.randn(
                num_samples,
                generator=generator,
                dtype=dtype,
                device="cpu",
            )
            draws = torch.exp(log_mean + log_std * standard_normal)

        elif dist_type == "uniform":
            low = required_scalar(param_name, dist_spec, "low")
            high = required_scalar(param_name, dist_spec, "high")
            if bool((low >= high).item()):
                raise ValueError(f"Distribution '{param_name}' requires low < high")
            unit_uniform = torch.rand(
                num_samples,
                generator=generator,
                dtype=dtype,
                device="cpu",
            )
            draws = low + (high - low) * unit_uniform

        else:
            value = required_scalar(param_name, dist_spec, "value")
            draws = torch.ones(num_samples, dtype=dtype, device="cpu") * value

        if not bool(torch.isfinite(draws).all().item()):
            raise ValueError(
                f"Distribution '{param_name}' produced non-finite samples"
            )
        samples[param_name] = draws

        logger.debug(
            "Sampled %s: mean=%.3f, std=%.3f",
            param_name,
            samples[param_name].mean().item(),
            samples[param_name].std(unbiased=False).item(),
        )

    return samples


def run_ensemble_simulation(
    self: Any,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    dt: Optional[float] = None,
    *,
    num_samples: int,
    outlier_detection: bool,
    parameter_distributions: Optional[Dict[str, Any]] = None,
    execution_mode: str = "sequential",
    statistics: Optional[List[str]] = None,
    quantile_levels: Optional[List[float]] = None,
    store_samples: bool = False,
    seed: Optional[int] = None,
) -> Any:
    """Run ensemble simulations through the shared Simulator runtime.

    Sequential execution is the only operational strategy because the current
    Simulator, StateManager, and ParameterManager are shared mutable owners.
    Other declared strategies fail closed until each member owns an independent
    runtime.
    """
    from ...stochastic.ensemble_runner import (
        EnsembleRunner,
        ExecutionConfig,
        ExecutionMode,
        StatisticType,
    )

    # Validate inputs
    if end_time is None:
        end_time = self.max_time
    if dt is None:
        dt = self.time_step

    known_execution_modes = {
        "vectorized",
        "parallel_thread",
        "parallel_process",
        "sequential",
    }
    if execution_mode not in known_execution_modes:
        supported_modes = ", ".join(sorted(known_execution_modes))
        raise ValueError(
            f"Unsupported ensemble execution mode '{execution_mode}'. "
            f"Expected one of: {supported_modes}"
        )
    if execution_mode != "sequential":
        raise NotImplementedError(
            f"[N] Simulator ensemble mode '{execution_mode}' requires an "
            "independent runtime per member; the current Simulator, StateManager, "
            "and ParameterManager are shared mutable owners"
        )
    resolved_execution_mode = ExecutionMode.SEQUENTIAL

    logger.info("Starting ensemble simulation: %s samples, mode=%s", num_samples, execution_mode)

    # Every member restores one runtime transaction (state, RNG) and replays it;
    # members differ only by their sampled parameters. Without a distribution all
    # members are the same run, so the request is rejected instead of running N
    # copies of it.
    if not parameter_distributions:
        raise ValueError(
            "run_ensemble requires parameter_distributions: members replay one runtime "
            "transaction and differ only by their sampled parameters"
        )
    parameter_samples = sample_parameters(parameter_distributions, num_samples, seed)

    parameter_manager = None
    parameter_keys: List[str] = []
    original_parameter_values: Dict[str, Any] = {}
    if parameter_samples:
        parameter_keys = list(parameter_samples)
        bound_functors = {
            name: functor
            for name, functor in self.functors.items()
            if getattr(functor, "_parameter_manager", None) is not None
        }
        parameter_managers = _collect_parameter_managers(bound_functors)
        owners_by_parameter: Dict[str, Any] = {}
        missing_parameters: List[str] = []
        for name in parameter_keys:
            owners = []
            for candidate in parameter_managers:
                get_parameter = getattr(candidate, "get_parameter", None)
                if not callable(get_parameter):
                    raise TypeError(
                        "Functor-bound ParameterManager must implement get_parameter"
                    )
                if get_parameter(name) is not None:
                    owners.append(candidate)
            if not owners:
                missing_parameters.append(name)
                continue
            if len(owners) > 1:
                raise ValueError(
                    f"Ensemble parameter '{name}' has multiple ParameterManager owners"
                )
            owners_by_parameter[name] = owners[0]

        if missing_parameters:
            names = ", ".join(sorted(missing_parameters))
            raise KeyError(
                "Ensemble parameters are not registered in ParameterManager: "
                f"{names}"
            )

        unique_owners = {id(owner): owner for owner in owners_by_parameter.values()}
        if len(unique_owners) != 1:
            raise ValueError(
                "Ensemble parameter ledger spans multiple ParameterManager instances"
            )
        parameter_manager = next(iter(unique_owners.values()))
        if not callable(getattr(parameter_manager, "get_batch", None)):
            raise TypeError("Functor-bound ParameterManager must implement get_batch")
        if not callable(getattr(parameter_manager, "set_batch", None)):
            raise TypeError("Functor-bound ParameterManager must implement set_batch")

        original_values = parameter_manager.get_batch(parameter_keys)
        original_parameter_values = {
            name: (
                value.detach().clone()
                if isinstance(value, Tensor)
                else deepcopy(value)
            )
            for name, value in zip(parameter_keys, original_values)
        }

    runtime_transaction = capture_runtime_transaction(self)

    # The running-mean convergence diagnostic is enabled from the same member
    # floor as EnsembleUQ (optimization.uncertainty.ensemble.convergence_min_members,
    # YAML SSOT); below it an ensemble is never checked for convergence.
    convergence_min_members = get_global_manager().get(
        "optimization.uncertainty.ensemble.convergence_min_members",
        required=True,
        expected_type=int,
    )
    # outlier_threshold, convergence_window and convergence_tolerance are the
    # runner's stochastic.ensemble.* database values (ExecutionConfig reads them).
    config = ExecutionConfig(
        mode=resolved_execution_mode,
        outlier_detection=outlier_detection,
        convergence_check=num_samples >= convergence_min_members,
    )

    ensemble_runner = EnsembleRunner(
        config=config,
        device=self.device if hasattr(self, "device") else torch.device("cpu"),
        parameter_manager=parameter_manager,
    )

    # Parse statistics
    stat_types = []
    if statistics is None:
        statistics = ["mean", "std", "quantile"]

    stat_map = {
        "mean": StatisticType.MEAN,
        "std": StatisticType.STD,
        "var": StatisticType.VAR,
        "quantile": StatisticType.QUANTILE,
        "min": StatisticType.MIN,
        "max": StatisticType.MAX,
        "median": StatisticType.MEDIAN,
        "iqr": StatisticType.IQR,
    }

    for stat_name in statistics:
        if stat_name in stat_map:
            stat_types.append(stat_map[stat_name])

    # Define model callable
    def model_callable(inputs: Dict[str, Any]) -> Dict[str, Tensor]:
        """Execute one member through restore-and-replay shared-owner semantics."""
        restore_runtime_transaction(self, runtime_transaction)
        try:
            if parameter_manager is not None:
                missing_inputs = [name for name in parameter_keys if name not in inputs]
                if missing_inputs:
                    names = ", ".join(sorted(missing_inputs))
                    raise KeyError(f"Ensemble member is missing parameters: {names}")
                member_updates: Dict[str, Any] = {}
                for param_name in parameter_keys:
                    param_value = inputs[param_name]
                    if isinstance(param_value, Tensor):
                        if param_value.numel() != 1:
                            raise ValueError(
                                "Sequential ensemble parameters must be scalar: "
                                f"{param_name}"
                            )
                        param_value = param_value.detach().reshape(()).item()
                    member_updates[param_name] = param_value
                parameter_manager.set_batch(
                    member_updates,
                    validate=True,
                    notify=True,
                )

            # Run simulation
            self.reset()
            self.run(start_time=start_time, end_time=end_time, dt=dt)

            # Extract outputs
            final_state = self._get_current_state()
            outputs = {}
            for var_name, var_value in final_state.items():
                if isinstance(var_value, Tensor) and var_name not in ["time", "step"]:
                    # Aggregate spatial/temporal dimensions to scalar
                    if var_value.numel() == 1:
                        outputs[var_name] = var_value.squeeze()
                    else:
                        outputs[var_name] = var_value.mean()  # Average over spatial points

            return outputs

        finally:
            try:
                if parameter_manager is not None:
                    parameter_manager.set_batch(
                        original_parameter_values,
                        validate=True,
                        notify=True,
                    )
            finally:
                restore_runtime_transaction(self, runtime_transaction)

    # Run ensemble
    try:
        results = ensemble_runner.run_ensemble(
            model_callable=model_callable,
            parameter_samples=parameter_samples,
            fixed_inputs=None,
            statistics=stat_types,
            quantile_levels=quantile_levels,
            store_samples=store_samples,
            seed=seed,
        )
    finally:
        try:
            if parameter_manager is not None:
                parameter_manager.set_batch(
                    original_parameter_values,
                    validate=True,
                    notify=True,
                )
        finally:
            restore_runtime_transaction(self, runtime_transaction)

    logger.info(
        "Ensemble completed: %s/%s successful",
        results.execution_info.get("successful_runs", 0),
        num_samples,
    )

    return results
