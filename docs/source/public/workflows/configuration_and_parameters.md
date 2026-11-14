# Configuration and Parameters

RELIC separates structural configuration from scientific parameter values.
`RELICConfig` validates subsystem structure across 15 Pydantic V2 `BaseModel`
classes in `relic.core.config.models` (base `ConfigurationModel`, root
`RELICConfig` at schema `2.0.0`, 10 subsystem configuration models, and 3
nested `DataConfig` sub-models `DataIOConfig`, `DataQualityConfig`, and
`DataTransformConfig`, totaling 13 subsystem and nested models beneath
`RELICConfig`), while `UnifiedParameterManager` serves as the Single Source of
Truth (SSOT) for versioned scientific values and metadata. `RELICSimulator`
consumes a runtime mapping and is not a direct consumer of a `RELICConfig`
object.

## Canonical owners

| Concern | Public API | Canonical implementation |
| --- | --- | --- |
| Typed configuration model | `RELICConfig` | `relic\core\config\models.py` |
| Configuration loading | `ConfigLoader`, `load_config` | `relic\core\config\loader.py` |
| Parameter value ownership | `UnifiedParameterManager` | `relic\parameters\manager.py` |
| Parameter metadata | `ParameterSpec` | `relic\parameters\spec.py` |
| Functor parameter access | `FunctorBase.get_parameter_value()` | `relic\core\functor\functor_base.py` |
| Physical constants | `get_physical_constant()` | `relic\utils\physical_constants.py` |

See [Parameter Contracts API](../api/parameters_contracts.md) for the full
reference.

## `RELICConfig`

```text
RELICConfig(
    *,
    config_version: str = "2.0.0",
    simulation_name: str = "relic_simulation",
    description: str = "",
    created_at: datetime = ...,
    physics: PhysicsConfig = ...,
    simulation: SimulationConfig = ...,
    stochastic: StochasticConfig = ...,
    optimization: OptimizationConfig = ...,
    surrogates: dict[str, SurrogateConfig] = ...,
    data: DataConfig = ...,
    parameters: ParametersConfig = ...,
    computation: ComputationConfig = ...,
    logging: LoggingConfig = ...,
    feature_flags: FeatureFlagConfig = ...,
)
```

| Parameter | Meaning |
| --- | --- |
| `config_version` | Version of the typed configuration schema. |
| `simulation_name` | File-system-safe simulation identifier. |
| `description` | Human-readable model configuration description. |
| `created_at` | Configuration creation timestamp; defaults to the time the object is built. |
| `physics` | Switches and settings for physical and biological process families. |
| `simulation` | Typed time, grid, integration, checkpoint, and conservation settings. |
| `stochastic` | Stochastic process and sampling settings. |
| `optimization` | Typed optimization and calibration settings. |
| `surrogates` | Surrogate configurations keyed by Functor name. |
| `data` | Input, output, Climate and Forecast (CF) metadata, quality-control, and transform settings. |
| `parameters` | Parameter database and validation settings. |
| `computation` | Precision, parallelism, distributed, and memory settings. |
| `logging` | Logging and monitoring settings. |
| `feature_flags` | Explicit feature switches. |

`RELICConfig` rejects unknown fields, validates assignments, serializes
enums as values, and validates cross-section constraints. Examples of
validation failures include an end time not greater than the start time, or a
time step longer than the configured duration.

### Load and serialize typed configuration

```python
from pathlib import Path
import tempfile

from relic.core.config import RELICConfig

config = RELICConfig.from_yaml(r"configs\simulation_minimal.yaml")
print(config.config_version, config.simulation.time_step)

with tempfile.TemporaryDirectory() as tmpdir:
    exported_path = Path(tmpdir) / "exported_config.yaml"
    config.to_yaml(exported_path)
    reloaded = RELICConfig.from_yaml(exported_path)
    assert reloaded.simulation.time_step == config.simulation.time_step

as_mapping = config.to_dict()
yaml_text = config.to_yaml()
schema = config.to_schema()
```

| Method | Return value | Exceptions |
| --- | --- | --- |
| `RELICConfig.from_yaml(path)` | Validated `RELICConfig` | Same as `ConfigLoader().load_config(path)`: it applies environment overrides, writes `parameters.parameter_overrides` to the ParameterManager, and runs `ConfigValidator`. Every load failure raises `relic.core.config.loader.ConfigurationError` (see below). |
| `to_dict()` | Nested Python mapping | Pydantic serialization exceptions are propagated. |
| `to_yaml(path=None)` | YAML text when `path` is omitted; otherwise `None` | File errors are propagated. |
| `to_schema()` | JSON Schema mapping | Model schema errors are propagated. |

`relic.core.config` imports PyYAML, Hydra, OmegaConf, and toml when it is
imported, so none of these methods runs without them.

### `ConfigLoader` and `load_config`

```python
from relic.core.config import ConfigLoader, load_config

loader = ConfigLoader()
config = loader.load_config(
    config_path=r"configs\simulation_minimal.yaml",
    overrides={"simulation.time_step": 1800.0},
    env_overrides=True,
)

config_unvalidated = load_config(
    r"configs\simulation_minimal.yaml",
    validate=False,
)
```

`ConfigLoader.load_config(config_path=None, overrides=None,
env_overrides=True)` is the single configuration load path. It reads the file
as UTF-8, applies the dotted overrides, applies environment overrides when
`env_overrides` is true, builds `RELICConfig`, writes
`parameters.parameter_overrides` to the ParameterManager (with bounds
validation), and runs `ConfigValidator`. `overrides` is a mapping such as
`{"simulation.time_step": 1800.0}` or a list of `"dotted.key=value"` strings;
an item without `=` raises. An environment variable is read only when its name
is `RELIC_` plus the upper-case path of a `RELICConfig` field joined with
`"_"` characters, for example `RELIC_SIMULATION_TIME_STEP` for
`simulation.time_step`; other `RELIC_*` variables are not configuration. The
module-level `load_config(config_path=None, overrides=None, validate=True)`
uses `ConfigLoader(validate_on_load=validate)`; `validate=False` is the only
way to skip `ConfigValidator`. `ConfigValidator` accepts enum fields given as
strings in YAML, such as `integration_method: "euler"`.

A named configuration (the `config_path` argument, else `RELIC_CONFIG_PATH`)
that does not exist raises `FileNotFoundError`. Without a named configuration
the first existing file of `relic.core.config.loader.DEFAULT_CONFIG_PATHS` is
read, and when none exists the global default configuration (`RELICConfig`
defaults, with overrides and environment overrides applied) is used and the
choice is logged; `relic.core.config.resolve_config_source` is this one rule
for `ConfigLoader`, `get_global_config` and `load_runtime_config`. Every other
load failure raises `relic.core.config.loader.ConfigurationError`, with the
original exception as `__cause__` where there is one: an unsupported suffix,
invalid YAML, a model-validation error, a malformed override, a
`parameter_overrides` key that is not a registered parameter, and
`ConfigValidator` errors.

## Runtime configuration mapping

`RELICSimulator` reads a Python mapping with `.get()`. Do not pass a
`RELICConfig` instance directly. Map typed settings to the runtime key names:

```python
from relic.core.config import RELICConfig
from relic.core.simulator import RELICSimulator

typed = RELICConfig.from_yaml(r"configs\simulation_minimal.yaml")
runtime_config = {
    "time_step": typed.simulation.time_step,
    "max_time": typed.simulation.end_time,
    "output_frequency": 1,  # simulator steps between time_series records
    "enable_checkpointing": False,
    "time_integration_method": typed.simulation.integration_method,
    "device": "auto",
}
simulator = RELICSimulator(runtime_config)
```

Runtime keys may be placed at the mapping root or under a `simulator`,
`runtime`, or `computation` section. `RELICSimulator` reads each key from the
root first and otherwise from the first of these sections, in that order, that
contains it; `device` and `require_gpu` use the same lookup. The
`device_overrides` and `module_device_overrides` mappings are merged in the
order root, `simulator`, `runtime`, `computation`; the later entry wins for the
same name.
`SimulationConfig` has no `output_frequency` or checkpoint field, so these
runtime keys are given explicitly. It declares no adaptive-step field either:
adaptive stepping is configured by the runtime keys `adaptive_dt_enabled`,
`adaptive_atol`, `adaptive_rtol`, `adaptive_dt_min` and `adaptive_dt_max`, whose
defaults are the `computation.scheduling.adaptive.*` parameters. Its
`end_time` defaults to one day and its `time_step` must not exceed one hour
(`SECONDS_PER_DAY` and `SECONDS_PER_HOUR` of `relic.utils.time_constants`). The simulator does not rename
typed fields such as `simulation.end_time` to `max_time` or
`simulation.integration_method` to `time_integration_method`.

### Main runtime keys

| Key | Meaning |
| --- | --- |
| `time_step` | Default step duration passed to Functors. |
| `max_time` | Default end time used by `run()` when no explicit end time is supplied. |
| `output_frequency` | Number of simulator steps between records of Functor output values in `time_series` (positive integer, default 1). |
| `enable_checkpointing` | Opt in to automatic checkpoint writes; defaults to `False`. |
| `checkpoint_directory` | Explicit automatic-checkpoint directory; required when checkpointing is enabled. |
| `checkpoint_frequency` | Simulation-time interval in seconds between automatic checkpoints (default 3600.0); must be positive when checkpointing is enabled. |
| `time_integration_method` | Transport time-integration method: `euler`, `implicit_midpoint`, or `crank_nicolson`, the allowed values of `transport.numerical.time_integration_method`, whose value is used when the key is unset. Another value raises `ValueError` at construction. It is passed to Functors whose `forward()` declares `time_integration_method`. |
| `implicit_midpoint_max_iter`, `implicit_midpoint_tolerance` | Iteration limit and convergence tolerance of the implicit transport steps; unset, `transport.numerical.implicit_midpoint_max_iter` and `transport.numerical.implicit_midpoint_tolerance` are used, and configured values are checked against their bounds. |
| `splitting_mode` | Operator-splitting mode used by the execution loop. |
| `strict_dependencies` | Escalate dependency and ordinary output-overlap conflicts where supported. |
| `conservation_checks` | `True` or `False` enables conservation accounting in the runtime commit path; a mapping may set only the StateManager keys `enabled`, `raise_on_violation`, `rollback_on_failure` (each `True` or `False`) and `tolerance`. Another type, an unknown key, or an invalid value raises `ValueError` naming the key and the value. |
| `conservation_raise_on_violation`, `conservation_rollback_on_failure` | Set `raise_on_violation` and `rollback_on_failure` of that mapping. Only `True` or `False` is accepted; a string such as `"false"` or an integer `0`/`1` raises `ValueError`, and a value that contradicts the same key in `conservation_checks` raises. |
| `conservation_tolerance` | Explicit override of the StateManager commit-time ledger tolerance, whose default is `validation.conservation.conservation_tolerance`. It must be a finite real number within the bounds declared for that parameter (currently `[1e-10, 1e-3]`); any other value raises `ValueError` naming the key and the value. |
| `strict_conservation_enabled` | Escalate configured conservation failures. |
| `adaptive_dt_enabled` | Enable adaptive time-step control. |
| `adaptive_atol`, `adaptive_rtol` | Absolute and relative adaptive error tolerances. |
| `adaptive_dt_min`, `adaptive_dt_max` | Bounds for the adaptive time step. |
| `adaptive_monitor_vars` | State variables used by adaptive error control. |
| `device` | Global device policy: `auto`, `cpu`, `cuda`, or `cuda:N`. |
| `require_gpu` | Reject CPU fallback when `true`. |
| `device_overrides` | Device requests keyed by Functor name. |
| `module_device_overrides` | Device requests keyed by module name. |
| `state_units` | Units of initial-state variables as a mapping of variable name to unit; read by `initialize_state()` when it builds the `StateManager`. A unit that disagrees with a variable contract or catalog declaration raises `ValueError`. |
| `enable_distributed` | Enable simulator-side distributed execution context. |
| `enable_diagnostic_export` | Enable configured diagnostic export. |
| `diagnostic_output_frequency` | Diagnostic export interval. |
| `diagnostic_output_path` | Diagnostic output path. |
| `diagnostic_reference_date` | Reference time used in diagnostic time coordinates. |

There is no implicit automatic-checkpoint directory. To enable automatic
persistence, map all required values explicitly:

```python
from pathlib import Path
import tempfile

from relic.core.simulator import RELICSimulator

with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_dir = Path(tmpdir) / "automatic"
    runtime_config = {
        "time_step": 1800.0,
        "max_time": 3600.0,
        "enable_checkpointing": True,
        "checkpoint_directory": str(checkpoint_dir),
        "checkpoint_frequency": 3600.0,  # [s] simulation time
    }
    simulator = RELICSimulator(runtime_config)
    assert simulator.enable_checkpointing is True
```

`checkpoint_directory=None` or a non-positive frequency is rejected while
automatic checkpointing is enabled. Manual
`simulator.create_checkpoint(checkpoint_path)` is independent of the automatic
switch and always requires an explicit, non-existing target file.

## `ParameterSpec`

```text
ParameterSpec(
    key: str,
    value: Any,
    units: str,
    dtype: str,
    bounds: Optional[tuple[Any, Any]],
    category: ParameterCategory,
    default: Any = None,
    description: str = "",
    references: list[str] = [],
    tags: set[str] = set(),
    dimension: str = "",
    provenance: dict[str, Any] = {},
    requires_grad: bool = False,
    allowed_values: Optional[list[Any]] = None,
)
```

| Parameter | Meaning |
| --- | --- |
| `key` | Fully qualified parameter identifier. |
| `value` | Current parameter value. |
| `default` | Optional declared fallback retained as metadata. |
| `units` | Canonical physical units or `dimensionless` (required). |
| `dtype` | Declared storage or tensor data type (required). |
| `bounds` | Inclusive admissible interval (required): a numeric `dtype` writes an unbounded side as `-inf` or `inf` (in YAML `-.inf` / `.inf`), and `None` (`null`) states that no interval applies, which is accepted only for a non-numeric `dtype` such as a text value, a boolean switch, a list or a mapping. Numeric endpoints, values, and defaults must be YAML numbers (`1.0e+7`, `1.0e-6`); text such as `1.0e7` or `1e-6` is rejected without conversion. |
| `category` | Parameter classification used by validation and discovery (required). |
| `description` | Scientific meaning of the parameter. |
| `references` | Literature or authoritative source identifiers. |
| `tags` | Search and grouping labels. |
| `dimension` | Physical dimension independent of unit spelling. |
| `provenance` | Source and transformation metadata. |
| `requires_grad` | Whether optimization may treat the value as trainable. |
| `allowed_values` | Optional finite admissible set for categorical values. |

Scientific YAML entries are expected to carry `value`, `units`, `dtype`,
`bounds`, `category`, `description`, `references`, `tags`, `requires_grad`,
and `provenance`. Physical constants belong only in
`relic\parameters\database\relic\physical\constants.yaml`.

## `UnifiedParameterManager`

In production mode (with `RELIC_SKIP_CALIBRATED` unset), `UnifiedParameterManager`
loads 4,643 runtime parameters (810 trainable parameters with `requires_grad=True`
accessible via `get_trainable_parameters(with_gradients=True)`) from 138 YAML
files under `relic\parameters\database\` (134 canonical `relic\` files
containing 4,642 leaf entries and 4 `calibrated\` override files containing
109 leaf entries, totaling 4,751 YAML leaf entries).

```text
manager.get(
    key,
    default=None,
    namespace=None,
    units=None,
    expected_type=None,
    required=False,
    bounds=None,
)
```

| Argument | Meaning |
| --- | --- |
| `key` | Parameter key, optionally resolved relative to `namespace`. |
| `default` | Value returned when the key is absent and `required` is false. |
| `namespace` | Namespace prefix used during resolution. |
| `units` | Requested output units; a requested conversion is always checked (reading has no validation switch, and any other keyword raises `TypeError`). |
| `expected_type` | Required Python or tensor-compatible value type; an integer value is widened when `float` is expected, a bool is never a number, and no value is narrowed. |
| `required` | Raise `KeyError` when the key is absent. |
| `bounds` | Optional access-time bounds. Out-of-range values are clamped with a `UserWarning`. |

`get_parameter(key)` returns the `ParameterSpec` metadata object, not the
parameter value.

```python
from relic.parameters import get_global_manager

manager = get_global_manager()
density = manager.get(
    "relic.physical.constants.air.density_stp",
    required=True,
    units="kg m-3",
)
spec = manager.get_parameter("relic.physical.constants.air.density_stp")

print(density)
print(spec.units, spec.description)
```

Other manager operations include `set_value`, `register`, `list`,
`snapshot`, `restore`, `diff`, `export`, `freeze`, `unfreeze`,
`get_trainable_parameters`, and `validate_parameters`. Use them on the
canonical manager; do not create a second parameter registry.

## Functor parameter access

Inside a `FunctorBase` subclass, use the Functor-owned resolver:

```python
from relic.core.functor import BiologicalProcessFunctor, FunctorMetadata
from relic.core.functor.functor_types import (
    ProcessCategory,
    SpatialScale,
    VariableKind,
)


class ParameterLookupDemo(BiologicalProcessFunctor):
    def __init__(self):
        metadata = FunctorMetadata(
            name="param_lookup_demo",
            description="Resolve a governed parameter via FunctorBase.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            mathematical_expression="dx/dt = -alpha * x",
            references=["Executable documentation example"],
            parameters={
                "stochastic.weather.temperature.reversion_rate": "Mean reversion rate [day-1]"
            },
            input_variables=["x"],
            output_variables=["x"],
            spatial_scale=SpatialScale.POINT,
            variable_kind=VariableKind.STATE,
        )
        super().__init__(name="param_lookup_demo", config={}, metadata=metadata)

    def read_reversion_rate(self) -> float:
        param = self.get_parameter_value(
            "stochastic.weather.temperature.reversion_rate",
            required=True,
            units="day-1",
        )
        raw = param.value if hasattr(param, "value") else param
        return float(raw.detach()) if hasattr(raw, "detach") else float(raw)

    def forward(self, state, time, dt, **kwargs):
        return {}


demo = ParameterLookupDemo()
print(round(demo.read_reversion_rate(), 4))
```

```text
FunctorBase.get_parameter_value(
    name,
    *,
    units=None,
    bounds=None,
    validate=True,
    namespace=None,
    source_order=("manager", "config"),
    expected_type=None,
    required=False,
)
```

`source_order` controls lookup precedence among the YAML database
(`"manager"`) and the Functor config (`"config"`, which `set_parameter_value`
writes); any other name raises `ValueError`. There is no caller-supplied
default and no dynamic registration: a missing parameter is `None`, and
`required=True` makes it fail immediately.

Non-Functor code uses `get_global_manager().get(...)`. Scientific Functors
must not call the global manager directly, hardcode a scientific fallback, or
mutate manager caches.

## Physical constants

```python
from relic.utils.physical_constants import get_physical_constant

rho_air = get_physical_constant("air_density_standard")
```

`get_physical_constant(name)` resolves the canonical constant database; the
name `air_density_standard` resolves to
`relic.physical.constants.air.density_stp` (1.293 kg m-3), the density of dry air
at 0 degC and 101.325 kPa. It is a reference constant, not the air density of a
given state: process code that needs the density of the current air computes the
moist-air density from temperature, pressure, and vapour pressure with
`relic.scientific.hydrosphere.utils.psychrometrics.compute_moist_air_density`
(as `PenmanMonteithET` does).
Do not duplicate a constant in Python or in a domain parameter file.

## Related documentation

- [Architecture](../architecture.md)
- [Core Runtime API](../api/core_runtime.md)
- [Parameter Contracts API](../api/parameters_contracts.md)
- [State, Functors, and Simulation](state_functors_and_simulation.md)
- [Science index](../science/index.rst)
- [Theory index](../theory/index.rst)
