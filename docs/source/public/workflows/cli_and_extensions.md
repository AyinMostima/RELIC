# CLI and Extensions

RELIC exposes Command-Line Interface (CLI) modules, a FastAPI Representational
State Transfer (REST) and WebSocket server for inspection and bounded
operations, and a manifest-driven extension protocol for external Functors.
Command availability does not replace the Python lifecycle or Graph
Intermediate Representation (`GraphIR`) structural validation.

## Unified command module

Invoke the unified CLI (`relic\interfaces\cli\main.py`) through Python:

```powershell
python -m relic.interfaces.cli.main --help
python -m relic.interfaces.cli.main --version
```

| Subcommand (alias) | Sub-operations | Scope and status |
| --- | --- | --- |
| `params` | `list`, `get`, `set`, `validate`, `export`, `diff`, `audit` | Inspect, modify, validate, export, compare, and audit governed parameters. |
| `optimization` (`opt`) | `calibrate`, `assimilate`, `sensitivity`, `uncertainty` | `[N]` `runtime_unbound`: optimization and Uncertainty Quantification (UQ) CLI entrypoints reject execution until bound to a sealed `GraphIR`/`ExecutionPlan`/`GraphExecutor` runtime. |
| `data` | `io`, `quality` | Data I/O, format conversion, quality checks, and optional streaming checks when `RELIC_EXPERIMENTAL_STREAMING=1` is set. |
| `compute` | `health`, `bench`, `ddp-check`, `splitting-sensitivity`, `dynamical-benchmark` | Computational health, benchmarks, distributed diagnostics, operator-splitting sensitivity, and dynamical benchmark metrics. |
| `stochastic` (`stoch`) | `uncertainty`, `weather`, `sde`, `gillespie`, `validate` | Parameter uncertainty propagation, bound Stochastic Differential Equation (SDE) and Gillespie Stochastic Simulation Algorithm (SSA) runs, validation, and `[N]` `weather` placeholder. |
| `orchestration` (`graph`) | `validate`, `compile` | Validate and compile `GraphIR` workflows, optionally emitting a sealed run bundle. |
| `healthcheck` (`health`) | `all`, `compute`, `data`, `params`, `optimize`, `stochastic`, `orchestration`, `config` | Subsystem and whole-system health diagnostics. |

The package does not install a bare `relic` console script. Always use
`python -m ...` in the active environment.

## Parameter commands

```powershell
python -m relic.interfaces.cli.main params list --namespace relic.physical.constants
python -m relic.interfaces.cli.main params get relic.physical.constants.air.density_stp
python -m relic.interfaces.cli.main params validate --strict
python -m relic.interfaces.cli.main params export --namespace relic.physical.constants --format yaml --output output\physical_constants.yaml
```

`params set` changes the active manager value; it is not a substitute for
updating the canonical YAML parameter definition when a durable architecture
change is intended.

## Data and health commands

```powershell
python -m relic.interfaces.cli.main data io check --detailed
python -m relic.interfaces.cli.main data io read data\forcing.nc --info-only
python -m relic.interfaces.cli.main health all
python -m relic.interfaces.cli.main health all --benchmark --no-gpu-benchmark
python -m relic.interfaces.cli.main compute health
```

`--benchmark` requires `--gpu-benchmark` or `--no-gpu-benchmark` unless the
configuration sets `enable_gpu_benchmark`.

Use command-specific `--help` for required paths and formats:

```powershell
python -m relic.interfaces.cli.main data io convert --help
```

## Orchestration commands

Unified form:

```powershell
python -m relic.interfaces.cli.main orchestration validate path\to\graph.yaml --strict
python -m relic.interfaces.cli.main orchestration compile path\to\graph.yaml --strict --run-bundle output\run_bundle
```

Dedicated form:

```powershell
python -m relic.orchestration validate path\to\graph.yaml --strict
python -m relic.orchestration compile path\to\graph.yaml --strict --run-bundle output\run_bundle
```

`path\to\graph.yaml` stands for a GraphIR file that you write; the repository
ships no example GraphIR file. A `.yaml` or `.yml` file is read as YAML and any
other suffix as JSON. The payload must be a mapping with a non-empty
`schema_version` and a `nodes` mapping in which every node names a
`class_type`; see [Graph Validation and Execution](graph_validation_and_execution.md).

Both forms include the topology audit; only the dedicated form provides
`--no-topology-audit`. Both paths load the GraphIR and use the canonical
registry, validation, and compile functions. `--auto-adapt` (compile only)
lowers each required context-free `unit_cast` into an executable adapter node
and validates the graph again; any other required adapter type is rejected
(`unsupported_auto_adapter`). The adapter rules are described in
[Graph Validation and Execution](graph_validation_and_execution.md).

## Layered runner boundary

```powershell
python -m relic.interfaces.cli.run_layered --help
```

The layered runner provides an explicit state-oriented runtime shell. It does
not map every `RELICConfig` field to runtime keys and does not
discover and register a complete scientific model. Use the Python workflow or
a validated GraphIR when scientific Functor composition is required.

## Extension manifest

Every extension root contains:

```text
relic_extension.yaml
```

Top-level `ExtensionManifest` fields are:

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | Yes | Non-empty extension identity. |
| `version` | Yes | Semantic version in `X.Y.Z` form. |
| `author` | No | Author or organization. |
| `description` | No | Extension purpose. |
| `min_relic_version` | No | Minimum compatible RELIC semantic version. |
| `calibrated_output` | No | Optional calibrated-output declaration. |
| `functors` | Yes | One or more Functor declarations. |
| `parameters` | No | Parameter-file declarations. |
| `dependencies` | No | Required and optional Functor dependencies. |

Each Functor declaration contains:

| Field | Allowed values or meaning |
| --- | --- |
| `class` | Dotted import path to the concrete class. |
| `bridge` | `BiologicalProcessFunctor`, `PhysicalFieldFunctor`, `TransportMechanismFunctor`, or `ScaleCouplingFunctor`. |
| `category` | `biological_process`, `physical_field`, `transport_mechanism`, or `scale_coupling`. |
| `execution_group` | `fast`, `medium`, or `slow`; required, and a manifest without it fails validation. |
| `config` | Constructor configuration mapping. |

Parameter declarations contain `path` and an optional `namespace_prefix`.
Dependency declarations contain `required` and `optional` Functor-name lists.

```yaml
name: example_extension
version: 1.0.0
author: Example Team
description: Example RELIC transport extension.
min_relic_version: 1.0.0

functors:
  - class: example_extension.transport.ExampleTransport
    bridge: TransportMechanismFunctor
    category: transport_mechanism
    execution_group: medium
    config: {}

parameters:
  - path: parameters\transport.yaml
    namespace_prefix: extension.example

dependencies:
  required: []
  optional: []
```

Unknown manifest fields are ignored by the protocol model. Invalid semantic
versions, empty names, missing Functors, invalid class paths, or unsupported
bridge, category, and execution-group values raise manifest validation
errors.

## Extension discovery

```text
discover_extensions(
    extra_dirs: Optional[list[pathlib.Path]] = None,
    use_entry_points: bool = True,
    use_env_dirs: bool = True,
) -> list[ExtensionInfo]
```

Discovery order is:

1. caller-provided `extra_dirs`;
2. semicolon-separated directories in `RELIC_EXTENSION_DIRS`;
3. Python entry points in the `relic.extensions` group.

Duplicate extension names keep the first discovered entry.

```powershell
$env:RELIC_EXTENSION_DIRS = "C:\models\relic_ext_a;C:\models\relic_ext_b"
python -m relic.extensions.cli list
Remove-Item Env:RELIC_EXTENSION_DIRS
```

`ExtensionInfo` reports name, root path, discovery source, and parsed
manifest. Discovery proves that a manifest can be found and parsed; it does
not instantiate the declared Functors.

## Extension loading

```python
from pathlib import Path
from relic.extensions import load_single_extension, unload_extension

loaded = load_single_extension(
    Path(r"extensions\custom_functor"),
    validate=True,
    strict=True,
    populate_core_registry=True,
)
print(sorted(loaded))
unload_extension("custom_functor_example")
```

```text
load_single_extension(
    extension_path: pathlib.Path,
    validate: bool = True,
    strict: bool = True,
    populate_core_registry: bool = False,
) -> LoadedFunctors
```

```text
load_extensions(
    extra_dirs: Optional[list[pathlib.Path]] = None,
    validate: bool = True,
    strict: bool = True,
    populate_core_registry: bool = False,
    discovered_infos: Optional[list[ExtensionInfo]] = None,
) -> LoadedFunctors
```

| Argument | Meaning |
| --- | --- |
| `validate` | Validate the manifest, import path, bridge/category contract, parameters, and construction surface before loading. |
| `strict` | Default True: a validation failure raises `ExtensionValidationError` before anything is loaded and a duplicate instance name raises `ExtensionLoadError`. `strict=False` loads the valid extensions and lists each skipped extension or duplicate with its reason in `skipped_extensions`. |
| `populate_core_registry` | Add loaded classes to the existing scientific registry. It does not create another registry. |
| `discovered_infos` | Reuse an explicit discovery result without repeating filesystem and entry-point discovery. |

Duplicate Functor names across extensions raise `ExtensionLoadError` in strict
mode; in non-strict mode the first instance is kept and the duplicate is
listed in `skipped_extensions`. Use `get_loaded_extensions()` and
`get_extension_registry()` for inspection. `unload_extension()` removes
extension-owned simulator and node-registry entries when `simulator=` and
`registry=` are passed explicitly.

## Extension commands

```powershell
python -m relic.extensions.cli --help
python -m relic.extensions.cli scaffold my_extension --bridge transport_mechanism --execution-group fast --output C:\models\my_extension
python -m relic.extensions.cli validate C:\models\my_extension
python -m relic.extensions.cli list --source all
python -m relic.extensions.cli info my_extension
```

Scaffold bridge choices are `biological_process`, `physical_field`,
`transport_mechanism`, and `scale_coupling`. `--execution-group` (`fast`,
`medium`, or `slow`) is required.

Validation confirms the extension protocol and runtime construction checks.
It does not prove scientific validity, conservation, unit correctness for all
states, or Gate 0 completeness. A scientific extension still requires theory
documentation, parameter metadata, Functor metadata, state and port
contracts, and the applicable structural assertions.

## Register a loaded extension

```python
from pathlib import Path

from relic.core.simulator import RELICSimulator
from relic.extensions import load_single_extension, unload_extension

loaded = load_single_extension(
    Path(r"extensions\custom_functor"),
    validate=True,
    strict=True,
    populate_core_registry=False,
)
simulator = RELICSimulator({"time_step": 60.0})
for functor in loaded.values():
    simulator.register_functor(functor)
print(sorted(simulator.functors))
unload_extension("custom_functor_example", simulator=simulator)
```

Or register the instance for `GraphIR`:

```python
from pathlib import Path

from relic.extensions import load_single_extension, unload_extension
from relic.orchestration import NodeRegistry, SSOTCatalog

loaded = load_single_extension(
    Path(r"extensions\custom_functor"),
    validate=True,
    strict=True,
    populate_core_registry=False,
)
registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
node_definition = registry.register_functor_instance(
    next(iter(loaded.values()))
)
print(node_definition.node_type_id)
unload_extension("custom_functor_example", registry=registry)
```

Registration applies the same duplicate-name, output-ownership, dependency,
bridge, device, and state rules as built-in Functors.

## Stochastic CLI boundary

`stochastic sde` executes a bound continuous-SDE process through the canonical
GraphIR/Functor path (strict compile, `GraphExecutor`, StateManager commit).
The seed is mandatory because the Functor owns its Wiener substream, and the
drift/diffusion coefficients come from the ParameterManager database, so the
command accepts no per-run parameter overrides:

```powershell
python -m relic.interfaces.cli.main stochastic --seed 7 --output sde_run --format netcdf sde --process stochastic_air_temperature --samples 8 --steps 24 --dt 3600 --initial-value 1.0
```

`stochastic gillespie` executes a bound marked-jump process the same way
(direct-SSA event clock plus the Functor's mark law, counts and marks
starting from zero); arbitrary reaction-network declarations are rejected:

```powershell
python -m relic.interfaces.cli.main stochastic --seed 7 --output rain_run --format netcdf gillespie --process stochastic_rainfall --samples 8 --steps 24 --dt 21600
```

The NetCDF output of both commands carries `trajectory_<state>` on
`(sample, time)` for every bound state, the elapsed clock `time_seconds` on
`time`, ensemble and exact analytic moments, and the compile assertion audit.
`stochastic weather` remains `[N]` until its forcing operator has the same
kind of validated runtime binding.

## Optimization CLI boundary

The optimization subcommands (`calibrate`, `assimilate`, `sensitivity`, and
`uncertainty`) are visible in the unified CLI and marked `[N]`
`runtime_unbound`; they fail closed before running an optimizer or mutating
parameters because no sealed `GraphIR`/`ExecutionPlan`/`GraphExecutor` runtime
is bound. Use
[Calibration, Sensitivity Analysis, and Uncertainty Quantification](calibration_sensitivity_and_uq.md)
for supported Python component operation.

## FastAPI REST and WebSocket server

`relic\interfaces\api\server.py` exposes `create_app()` and `run_server()` for
HTTP and WebSocket inspection and orchestration. `create_app(enable_docs=False)`
registers 38 application routes (37 HTTP endpoints plus 1 WebSocket endpoint
`/graph/ws`); `create_app(enable_docs=True)` (the default) adds 4 OpenAPI and
interactive documentation routes (`/openapi.json`, `/docs`,
`/docs/oauth2-redirect`, and `/redoc`) for 42 total entries on `app.routes`:

| Route group | Count | Endpoints |
| --- | --- | --- |
| Root and metrics | 2 HTTP | `GET /`, `GET /health/metrics` |
| Health (`/health`) | 9 HTTP | `GET /health/`, `GET /health/compute`, `GET /health/data`, `GET /health/params`, `GET /health/optimize`, `GET /health/stochastic`, `GET /health/orchestration`, `GET /health/config`, `POST /health/benchmark` |
| Graph (`/graph`) | 5 HTTP + 1 WS | `POST /graph/validate`, `POST /graph/compile`, `POST /graph/submit`, `GET /graph/history/{run_id}`, `GET /graph/artifacts/{run_id}/{artifact_id}`, `WS /graph/ws` |
| Data (`/data`) | 5 HTTP | `GET /data/`, `POST /data/inspect`, `POST /data/read`, `POST /data/write`, `POST /data/convert` |
| Compute (`/compute`) | 4 HTTP | `GET /compute/`, `POST /compute/healthcheck`, `POST /compute/benchmark`, `POST /compute/ddp-check` |
| Optimization (`/optimization`) | 5 HTTP (`[N]`) | `GET /optimization/`, `POST /optimization/calibrate`, `POST /optimization/assimilate`, `POST /optimization/sensitivity`, `POST /optimization/uncertainty` (`[N]` `runtime_unbound`) |
| Stochastic (`/stochastic`) | 4 HTTP | `GET /stochastic/`, `POST /stochastic/weather` (`[N]`), `POST /stochastic/sde`, `POST /stochastic/gillespie` |
| Configuration (`/config`) | 3 HTTP | `GET /config/`, `GET /config/effective`, `POST /config/validate` |
| OpenAPI docs (`enable_docs=True`) | 4 HTTP | `GET /openapi.json`, `GET /docs`, `GET /docs/oauth2-redirect`, `GET /redoc` |

All non-health application routers (`/graph/*`, `/data/*`, `/compute/*`,
`/optimization/*`, `/stochastic/*`, and `/config/*`) enforce API key
authentication (`X-API-Key` header or `Authorization: Bearer <token>`,
configured through `RELIC_API_KEY`, `RELIC_API_KEYS`, or
`security.api_auth.api_keys`), while rate limiting is governed by the
parameters `interfaces.api.rate_limit.enabled` and
`interfaces.api.rate_limit.requests_per_minute`.

```python
from relic.interfaces.api.server import create_app

app_core = create_app(enable_docs=False)
app_full = create_app(enable_docs=True)
print(len(app_core.routes), len(app_full.routes))
```

## Related documentation

- [Architecture](../architecture.md)
- [Orchestration](../orchestration.md)
- [Graph Validation and Execution](graph_validation_and_execution.md)
- [Scientific Structure and Extensions API](../api/scientific_structure_extensions.md)
- [Science index](../science/index.rst)
- [Theory index](../theory/index.rst)
