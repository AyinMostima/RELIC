# Calibration, Sensitivity Analysis, and Uncertainty Quantification

RELIC exposes parameter calibration, global sensitivity analysis backed by
Sensitivity Analysis Library in Python (SALib), Uncertainty Quantification (UQ),
and data assimilation components as Python APIs over the canonical parameter
system. Every
model callback receives a parameter mapping; the callback is responsible for
running the intended RELIC model and returning the documented output shape.

## Availability

```python
from relic.optimization import get_available_optimizers

status = get_available_optimizers()
print(status)
```

The returned mapping reports import availability for calibration,
assimilation, sensitivity, and uncertainty families. Import availability does
not guarantee that a model callback, observation mapping, parameter range, or
optional numerical backend is valid.

## Parameter calibration

Learnable parameters are owned by ParameterManager. Obtain their live
`nn.Parameter` objects through `manager.get_trainable_parameters()` for the
manager calibration path. A Functor holds its own manager's binding by
reference, so module `parameters()` and `state_dict()` omit it.
`parameter_sensitivity(parameter_name, state, *, time, dt, ...)` uses
functional substitution of module members and raises `TypeError` for a held
manager binding. Re-registering an unchanged shape and dtype updates its
value in place, preserving references held by Functors and the optimizer.

### `ParameterBounds`

```text
ParameterBounds(
    lower: float,
    upper: float,
    scale: str,
    initial: Optional[float] = None,
)
```

`lower` must be less than `upper`. An explicit `initial` value must lie inside
the interval. `scale` is required and is `linear` or `log`; any other value
raises `ValueError`.

### `CalibrationConfig`

```text
CalibrationConfig(
    method: CalibrationMethod,
    target_parameters: list[str],
    parameter_bounds: dict[str, ParameterBounds],
    max_iterations: int = ...,
    tolerance: float = ...,
    population_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    parallel: bool = False,
    n_workers: Optional[int] = None,
    method_kwargs: dict[str, Any] = {},
    cross_validation_folds: Optional[int] = None,
    save_progress: bool = True,
    progress_file: Optional[pathlib.Path] = None,
    verbose: bool = True,
)
```

| Parameter | Meaning |
| --- | --- |
| `method` | Optimization algorithm. |
| `target_parameters` | Fully qualified parameter keys to calibrate. |
| `parameter_bounds` | Bounds keyed by every target parameter. |
| `max_iterations` | Maximum optimizer iterations or generations; when not stated, `optimization.calibration.max_iterations`. |
| `tolerance` | Convergence tolerance; when not stated, `optimization.calibration.objective_tolerance`. |
| `population_size` | Population size for population-based methods. |
| `random_seed` | Reproducibility seed for NumPy and PyTorch sampling. |
| `parallel`, `n_workers` | Parallel-evaluation request and worker limit. |
| `method_kwargs` | Algorithm-specific options. |
| `cross_validation_folds` | Optional fold count. |
| `save_progress`, `progress_file` | Progress persistence controls. |
| `verbose` | Optimizer logging control. |

Available `CalibrationMethod` values are differential evolution, genetic
algorithm, simulated annealing, particle swarm, Bayesian optimization,
gradient descent, Adam, and Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS).

The enum is a shared method identity, not one eight-algorithm execution facade.
`ParameterCalibrator` owns only the five derivative-free methods: differential
evolution, genetic algorithm, simulated annealing, particle swarm, and Bayesian
optimization. `GradientCalibrator` separately owns gradient descent, Adam, and
L-BFGS. Passing a gradient method to `ParameterCalibrator.calibrate(...)`
raises before logging, hooks, RNG, ParameterManager mutation, objective
evaluation, dispatch, or progress effects.

### `ParameterCalibrator` (derivative-free owner)

```text
ParameterCalibrator(
    parameter_manager: Optional[UnifiedParameterManager] = None,
    hooks: Optional[CalibrationHooks] = None,
)
```

```text
calibrator.calibrate(
    config: CalibrationConfig,
    objective_function: ObjectiveFunction,
    observations: torch.Tensor,
    initial_parameters: Optional[dict[str, float]] = None,
) -> CalibrationResult
```

`ObjectiveFunction.__call__(parameters, observations)` returns a scalar value
to minimize. The built-in objective classes include Mean Squared Error (`MSEObjective`),
Root Mean Squared Error (`RMSEObjective`), and `NashSutcliffeObjective`.

```python
import torch

from relic.parameters.calibration import (
    CalibrationConfig,
    CalibrationMethod,
    MSEObjective,
    ParameterBounds,
    ParameterCalibrator,
)


def model_runner(parameters):
    rate = parameters["model.rate"]
    return torch.tensor([rate, 2.0 * rate], dtype=torch.float32)


config = CalibrationConfig(
    method=CalibrationMethod.DIFFERENTIAL_EVOLUTION,
    target_parameters=["model.rate"],
    parameter_bounds={
        "model.rate": ParameterBounds(
            lower=0.1,
            upper=2.0,
            scale="linear",
            initial=0.5,
        )
    },
    max_iterations=50,
    random_seed=42,
    parallel=False,
    save_progress=False,
)
objective = MSEObjective(model_runner, variable_names=[])
observations = torch.tensor([1.0, 2.0], dtype=torch.float32)

calibrator = ParameterCalibrator()
result = calibrator.calibrate(
    config,
    objective,
    observations,
    initial_parameters={"model.rate": 0.5},
)
print(result.success, result.optimized_parameters)
```

The target key must be valid for the active parameter manager and the callback
must return values compatible with the observation tensor. A model failure or
a missing output variable in the built-in scalar objectives raises; there is
no infinite penalty value. A failed calibration triggers the
`CALIBRATION_ERROR` hook event and re-raises.

### `GradientCalibrator` (gradient owner)

```text
GradientCalibrationConfig(
    method: CalibrationMethod,
    learning_rate: float = ...,
    max_iterations: int = ...,
    tolerance: float = ...,
    beta1: float = ...,
    beta2: float = ...,
    epsilon: float = ...,
    weight_decay: float = ...,
    max_eval: Optional[int] = None,
    history_size: int = ...,
    line_search_fn: Optional[str] = None,
    patience: int = ...,
    min_delta: float = ...,
    verbose: bool = True,
    save_progress: bool = True,
    progress_file: Optional[pathlib.Path] = None,
)
```

```text
GradientCalibrator(
    parameter_manager: UnifiedParameterManager,
    config: Optional[GradientCalibrationConfig] = None,
    hooks: Optional[CalibrationHooks] = None,
)

gradient_calibrator.calibrate(
    objective_function,
    observations,
    target_parameters: Optional[list[str]] = None,
    parameter_bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> CalibrationResult
```

`method` is required. Every numeric setting that is not stated (shown as
`...`) is read from `optimization.calibration.*` in the parameter database;
the Adam settings there are the Algorithm 1 values of Kingma and Ba (2015).
`GradientCalibrator` requires both the caller's ParameterManager and a
`config`; a missing one raises `ValueError`, so implicit process-global
calibration state and a default optimizer are both rejected. The objective must
return one finite differentiable scalar and every requested target must produce
a finite gradient. Candidate values remain isolated from canonical manager
state until one accepted point is committed as an atomic manager batch.

```python
import torch

from relic.parameters import get_global_manager
from relic.parameters.calibration import (
    CalibrationMethod,
    GradientCalibrationConfig,
    GradientCalibrator,
)

manager = get_global_manager()
snapshot = manager.snapshot()
target_key = "stochastic.weather.temperature.reversion_rate"
try:
    grad_config = GradientCalibrationConfig(
        method=CalibrationMethod.ADAM,
        learning_rate=0.05,
        max_iterations=80,
        tolerance=1e-4,
        verbose=False,
        save_progress=False,
    )
    grad_calibrator = GradientCalibrator(manager, config=grad_config)
    observations = torch.tensor([0.2, 0.4], dtype=torch.float64)

    def differentiable_objective(params, obs):
        alpha = params[target_key]
        prediction = torch.stack([alpha, 2.0 * alpha])
        return torch.mean((prediction - obs) ** 2)

    grad_result = grad_calibrator.calibrate(
        differentiable_objective,
        observations,
        target_parameters=[target_key],
        parameter_bounds={target_key: (0.01, 1.0)},
    )
    print(round(grad_result.optimized_parameters[target_key], 2))
finally:
    manager.restore(snapshot)
```

### `CalibrationResult`

| Result group | Fields |
| --- | --- |
| Identity | `method`, `target_parameters`, `parameter_bounds` |
| Parameters | `initial_parameters`, `optimized_parameters` |
| Objective | `objective_value`, `initial_objective_value` |
| Convergence | `convergence_history`, `iteration_count`, `elapsed_time` |
| Status | `success`, `status`, `termination_reason` |
| Analysis | `metrics`, `parameter_uncertainties`, `sensitivity_analysis` |
| Evidence | `model_outputs`, `observations`, `calibrator_config`, `random_seed` |

Check `success`, `status`, and `termination_reason` before using optimized
parameters. A returned result is not equivalent to convergence or scientific
acceptance.

## Global sensitivity analysis

### `ParameterRange`

```text
ParameterRange(
    name: str,
    lower_bound: float,
    upper_bound: float,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    integer_valued: bool = False,
    log_scale: bool = False,
    units: str = "",
    description: str = "",
    *,
    distribution: str,
)
```

`distribution` is keyword-only and required: `uniform`, `normal`,
`lognormal`, or `triangular`. Bounds must be ordered. `normal` and
`lognormal` need an explicit `mean` and a finite positive `std` (for
`lognormal`, `mean` is the positive median), and `triangular` needs its mode
as `mean` inside the bounds; nothing is derived from the bounds.
`sample(n_samples, random_state=None)` draws from a local generator, clips
normal and lognormal draws to the bounds, and applies integer or log scaling
when requested. Declared `units` must equal the units of the `ParameterSpec`
of the parameter key of the same name; units declared for a name that is not
a parameter key raise `ValueError` when an analyzer validates the ranges.
Morris and Sobol analysis accept only uniform ranges without log scaling or
integer values; any other declaration raises `NotImplementedError`.

### Callback contract

Both analyzers use:

```text
analyze(
    model_runner: Callable[
        [dict[str, float]],
        dict[str, float],
    ],
    parameter_ranges: dict[str, ParameterRange],
    baseline_parameters: Optional[dict[str, float]] = None,
) -> SensitivityResult
```

The runner accepts one parameter mapping and returns scalar outputs by name.
`baseline_parameters` supplies fixed parameters and is merged with each
sampled mapping.

### `MorrisAnalyzer`

```text
MorrisAnalyzer(
    parameter_manager: UnifiedParameterManager,
    config: Optional[MorrisConfig] = None,
)
```

`config` is required: an analyzer built without it raises `TypeError`.
`MorrisConfig` takes `aggregation_method` as a required keyword (`mean`,
`sum`, `max`, `std`, or `final`); its numeric fields `n_trajectories`,
`n_levels`, and `grid_jump`, when left unset, read
`optimization.sensitivity.morris.*`. The constructor requires a positive
trajectory count and grid jump, at least two levels, and
2 x `grid_jump` <= `n_levels`.

Use Morris screening to rank many parameters with fewer evaluations. The
result contains elementary-effect statistics, output-wise sensitivity
indices, execution counts, timing, and convergence metadata.

### `SobolAnalyzer`

```text
SobolAnalyzer(
    parameter_manager: UnifiedParameterManager,
    config: Optional[SobolConfig] = None,
)
```

`config` is required here as well (`TypeError` without it). `SobolConfig`
takes `aggregation_method` and `estimator` as required keywords; only the
`saltelli` estimator is implemented (`janon`, `glen`, and `owen` raise
`NotImplementedError`). Its other fields are `n_samples`, `skip_values`,
`scramble`, `calc_second_order`, `convergence_check`, `convergence_samples`,
and `batch_size`; `n_samples` and `skip_values`, when left unset, read
`optimization.sensitivity.sobol.*`. Sample and batch counts must be positive.

Use Sobol analysis for first-order and total-order variance decomposition;
second-order indices are not implemented (`calc_second_order=True` raises
`NotImplementedError`). An empty parameter-range mapping raises `ValueError`;
analysis failures are wrapped in `RuntimeError`.

```python
from relic.optimization.sensitivity import (
    MorrisAnalyzer,
    MorrisConfig,
    ParameterRange,
)
from relic.parameters import get_global_manager

ranges = {
    "rate": ParameterRange(
        name="rate",
        lower_bound=0.1,
        upper_bound=2.0,
        distribution="uniform",
    )
}


def runner(parameters):
    return {"response": parameters["rate"] ** 2}


analyzer = MorrisAnalyzer(
    get_global_manager(),
    MorrisConfig(n_trajectories=10, n_levels=4, random_seed=42, aggregation_method="mean"),
)
sensitivity = analyzer.analyze(runner, ranges)
print(sensitivity.method, sensitivity.n_model_evaluations)
```

## Uncertainty quantification

### `UncertainParameter`

```text
UncertainParameter(
    name: str,
    distribution: DistributionType,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    lower_bound: float = -inf,
    upper_bound: float = inf,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    mode: Optional[float] = None,
    integer_valued: bool = False,
    log_transform: bool = False,
    units: str = "",
    description: str = "",
)
```

Supported distributions are uniform, normal, lognormal, beta, gamma, and
triangular. Bounds must be ordered. `NORMAL` and `LOGNORMAL` parameters
must give `mean` and `std`; a `UNIFORM` parameter without them takes the
moments of its bounds, and a triangular parameter needs an explicit `mode`.
`alpha`, `beta`, and `mode` configure distribution-specific shape. `sample()`
applies physical bounds and requested integer or log transforms.

### `EnsembleUQ` and `BootstrapUQ`

```text
EnsembleUQ(
    parameter_manager: UnifiedParameterManager,
    config: Optional[EnsembleConfig] = None,
)

BootstrapUQ(
    parameter_manager: UnifiedParameterManager,
    config: Optional[BootstrapConfig] = None,
)
```

`config` is required: either class built without it raises `TypeError`.
`EnsembleConfig` needs `sampling_strategy` (`random`, Latin Hypercube Sampling `lhs`, or `sobol`) and
`BootstrapConfig` needs `bootstrap_type` (`parametric` or `residual`) and
`ci_method` (`percentile` or `bca`), each as a keyword.

Both expose:

```text
quantify(
    model_runner: Callable[
        [dict[str, float]],
        dict[str, float],
    ],
    uncertain_parameters: dict[str, UncertainParameter],
    baseline_parameters: Optional[dict[str, float]] = None,
    observations: Optional[dict[str, numpy.ndarray]] = None,
) -> UQResult
```

`EnsembleUQ` propagates sampled parameter distributions through the callback.
Its configuration controls ensemble size, sampling, parallel evaluation,
localization, adaptive sampling, chunking, and outlier handling.
`BootstrapUQ` supports parametric and residual resampling; residual bootstrap
uses `observations`.

`UQResult` reports output means, standard deviations, variances, percentiles,
confidence intervals, optional samples, skewness, kurtosis, correlations,
sample and evaluation counts, timing, convergence, effective sample sizes,
and Monte Carlo error.

By default, failed samples and missing outputs are not accepted. Change
`allow_failed_samples` or `allow_missing_outputs` only when the resulting
sentinel values are explicitly handled downstream.

```python
from relic.optimization.uncertainty import (
    DistributionType,
    EnsembleConfig,
    EnsembleUQ,
    UncertainParameter,
)
from relic.parameters import get_global_manager

uq = EnsembleUQ(
    get_global_manager(),
    EnsembleConfig(
        ensemble_size=16,
        sampling_strategy="lhs",
        random_seed=42,
        parallel_evaluation=False,
    ),
)
uq_result = uq.quantify(
    lambda p: {"flux": 2.0 * p["alpha"]},
    {
        "alpha": UncertainParameter(
            name="alpha",
            distribution=DistributionType.UNIFORM,
            lower_bound=0.1,
            upper_bound=0.5,
        )
    },
)
print(round(uq_result.output_mean["flux"], 2), uq_result.n_model_evaluations)
```

## Data assimilation

`relic.optimization.assimilation` provides component-level state estimation
algorithms sharing the `Assimilator`, `ObservationSet`, and
`ObservationOperator` abstractions:

| Class | Configuration | Status and scope |
| --- | --- | --- |
| `EnsembleKalmanFilter` | `EnKFConfig` | Ensemble Kalman Filter (EnKF) single-step stochastic state update with optional localization and inflation. |
| `ThreeDVar` | `VariationalConfig(method=AssimilationMethod.VARIATIONAL_3D)` | Three-Dimensional Variational (3D-Var) single-time variational state analysis. |
| `FourDVar` | `VariationalConfig(method=AssimilationMethod.VARIATIONAL_4D)` | `[N]` Four-Dimensional Variational (4D-Var) framework shell (`relic\optimization\assimilation\variational.py`); defines the class skeleton without an implemented adjoint model, multi-time window gradient integration, or production runtime binding. |

## Identifiability

```text
compute_identifiability(
    jacobian: dict[str, dict[str, Tensor]] | Tensor | numpy.ndarray,
    tol: float = 1e-6,
) -> dict[str, Any]
```

The function computes the singular values, rank, nullity, condition number,
and unidentifiable directions of the observation Jacobian.

```python
import torch

from relic.optimization.identifiability import compute_identifiability

jacobian = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 0.0],
    ],
    dtype=torch.float64,
)
result = compute_identifiability(jacobian)
print(result["rank"], result["nullity"])
```

Expected output:

```text
1 1
```

## CLI boundary

Use the Python APIs on this page for bounded component-level calibration,
sensitivity analysis, and UQ. The public optimization CLI and HTTP routes are
explicitly `[N] runtime_unbound`: they reject before executing an objective,
constructing an optimizer workflow, or mutating parameters because no sealed
GraphIR/ExecutionPlan/GraphExecutor and simulator lifecycle is bound. Their
command and schema presence must not be treated as an end-to-end model
workflow.

## Related documentation

- [Architecture](../architecture.md)
- [Configuration and Parameters](configuration_and_parameters.md)
- [State, Functors, and Simulation](state_functors_and_simulation.md)
- [Parameter Contracts API](../api/parameters_contracts.md)
- [Science index](../science/index.rst)
- [Theory index](../theory/index.rst)
