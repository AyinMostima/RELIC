# Installation and Imports

Recomposable Ecosystem Library for Integrated Coupling (RELIC) declares package
version `1.0.0`, runtime requirement `Python >= 3.11`, core dependency
`torch >= 2.0.0`, and optional dependency group `[dev]` in `pyproject.toml`.

## Requirements and Conda environments

Package metadata requires Python 3.11 or newer (`requires-python = ">=3.11"`)
together with PyTorch (`torch >= 2.0.0`), Pydantic V2 (`pydantic >= 2.0.0`),
Hydra (`hydra-core >= 1.3.0`), NetCDF4 (`netCDF4 >= 1.6.0`), Zarr
(`zarr >= 2.14.0`), SALib (`SALib >= 1.4.0`), and FastAPI (`fastapi >= 0.100.0`).
Workloads are separated across three dedicated Conda environments selected by
the executed workload:

| Conda environment | Role |
| --- | --- |
| `RELIC` | Model execution, scientific simulation, parameter and graph validation, and model-dependent preparation. |
| `dataanalysis` | Independent data preparation, postprocessing, statistical analysis, and publication figure generation. |
| `sphinx_env` | Sphinx HTML documentation generation and public-documentation verification gates. |

Install the package in editable mode inside the `RELIC` Conda environment:

```powershell
cd C:\path\to\RELIC_PROJECT
conda activate RELIC
python --version
python -m pip install -e .
```

Install the `[dev]` optional extra only when development and static-analysis
tools are required:

```powershell
conda activate RELIC
python -m pip install -e ".[dev]"
```

The editable installation keeps imports attached to the current checkout.
Package installation does not initialize a simulator, load scientific
parameters, or register Functors.

## Audit and native I/O prerequisites

The repository gates and I/O paths depend on the following local conditions on
the reference Windows workstation. They are not installed by `pip`, and they
qualify a single host environment, not the RELIC package definition.

- `rg` (ripgrep) must be resolvable on `PATH`. The architecture audits under
  `test\architecture` and the device-policy check
  `rg "torch.cuda.is_available\(" relic` run it as a subprocess. Verify it with
  `Get-Command rg` and `rg --version`, and add the actual installation directory
  to `$env:PATH` when needed. A copy that happens to ship with an editor or
  assistant installation is not a RELIC package dependency.
- NetCDF filter plugins: the pip `netCDF4` wheel ships name-mangled copies of
  `hdf5.dll`, `netcdf.dll`, `zstd.dll`, `blosc.dll`, `libbz2.dll`, `szip.dll` and
  `zlib.dll` in `netcdf4.libs`, while the HDF5 filter plugins in
  `netCDF4\plugins` import them by plain name. Copies of those libraries under
  their unmangled names must exist in the plugin directory, otherwise every
  `compression="zstd"` write (the `DataWriter` default) fails with
  `NetCDF: Filter error: undefined filter encountered`. Importing `netCDF4`
  alone does not qualify the configured compression: verify the selected filter
  with a real write/read round trip before using it for scientific outputs, and
  repeat that qualification after reinstalling or upgrading the native packages,
  which removes the copies. Host-specific DLL repairs belong to the environment,
  not to RELIC source or generated documentation.
- `RELIC_SKIP_CALIBRATED`: when set to `1`, `true`, `yes` or `on`, the
  parameter database manager excludes the calibrated parameter tier for that
  process. The repository test session sets it to `1` by default, together with
  `RELIC_SKIP_LEGACY=1`, so test runs use the uncalibrated tiers; production
  runs must leave both unset to load calibrated values. The public science
  reference describes the production view, including calibrated parameters; its
  freshness check therefore runs the generator's `--check` in a child process
  with `RELIC_SKIP_CALIBRATED`, `RELIC_SKIP_LEGACY` and `RELIC_FAST_INIT`
  removed from that child's environment. Do not regenerate production defaults
  from the altered test-session singleton, and do not remove calibration files
  to make a freshness comparison pass.
- Experimental data feature flags: placeholder quality-control components in
  `relic.data.quality` require `RELIC_EXPERIMENTAL_DATA_QUALITY=1` at import
  time, and experimental streaming components in `relic.data.streaming` require
  `RELIC_EXPERIMENTAL_STREAMING=1`. Both remain disabled by default in
  standard runs.
- Test discovery: the repository test-runner configuration owns discovery
  (`test_*.py`, `*_test.py`, `*.test.py` and `*.spec.py`, importlib import mode,
  strict configuration and markers). The test-suite root `conftest.py` skips the
  Windows `nul` placeholder and any file whose dot-separated stem components are
  not valid identifiers, including Explorer duplicate copies such as
  `name (1).py`. This exclusion is a discovery rule, not evidence that the
  excluded copies were executed or verified. Ruff excludes the same duplicate
  copies through `extend-exclude` in `pyproject.toml`.

## Stable package imports

On Windows, optional NetCDF compression requires the selected HDF5 filter
plugin and its native DLL dependencies inside the active environment.
Qualify the chosen compression with an actual write/read round trip before
using it for scientific outputs; importing `netCDF4` alone is insufficient.
Repeat that qualification after replacing native packages.

The root package exposes the main construction and inspection entry points:

```python
import relic

print(relic.__version__)

simulator = relic.create_default_simulator(
    {
        "time_step": 3600.0,
        "max_time": 86400.0,
        "device": "auto",
    }
)
```

`create_default_simulator(config=None)` replaces `None` with an empty mapping
and delegates to `create_simulator(config)`. `create_simulator` constructs a
`RELICSimulator`; it does not initialize a `StateManager` or dependency graph.

For library code, import classes from their canonical public owners:

```python
from relic.core import (
    BiologicalProcessFunctor,
    FunctorBase,
    FunctorMetadata,
    RELICSimulator,
    SimulationState,
    StateManager,
    StateVariableSpec,
)
from relic.parameters import ParameterSpec, UnifiedParameterManager
from relic.parameters import get_global_manager
```

| API | Purpose | Return value |
| --- | --- | --- |
| `relic.create_simulator(config)` | Construct a simulator from a runtime configuration mapping. | `RELICSimulator` |
| `relic.create_default_simulator(config=None)` | Construct a simulator with an empty mapping when no configuration is supplied. | `RELICSimulator` |
| `relic.get_version_info()` | Report package version information. | `dict[str, str]` |
| `relic.core.StateManager` | Own prognostic and diagnostic tensors. | State manager instance |
| `relic.core.FunctorBase` | Provide the executable lifecycle for scientific modules. | Abstract runtime base |
| `relic.parameters.get_global_manager()` | Access the process-wide parameter owner from non-Functor code. | `UnifiedParameterManager` |

See [Core Runtime API](../api/core_runtime.md) and
[Parameter Contracts API](../api/parameters_contracts.md) for the complete
class surfaces.

## Lazy imports

Root-level optional components are resolved lazily by default. The following
values enable lazy imports: `1`, `true`, and `yes`.

Set the behavior before importing RELIC:

```powershell
$env:RELIC_LAZY_IMPORT = "1"
python -c "import relic; print(relic.__version__)"
```

Disable lazy imports for an import-surface diagnostic:

```powershell
$env:RELIC_LAZY_IMPORT = "0"
python -c "import relic; print(relic.get_version_info())"
Remove-Item Env:RELIC_LAZY_IMPORT
```

In lazy mode an `AttributeError` means the name is not exported or its
subpackage is missing; an error raised while importing a subpackage propagates
unchanged. With `RELIC_LAZY_IMPORT=0` a failing subpackage makes
`import relic` raise. Prefer the canonical subpackage import when the owner is
known.

## Command-module entry points

The installed package does not declare a bare `relic` console script. Invoke
the supported command modules through the active Python interpreter:

```powershell
python -m relic.interfaces.cli.main --help
python -m relic.orchestration --help
python -m relic.extensions.cli --help
python -m relic.interfaces.cli.run_layered --help
```

| Module | Scope |
| --- | --- |
| `relic.interfaces.cli.main` | Unified parameter, optimization, data, computation, stochastic, orchestration, and health commands. |
| `relic.orchestration` | GraphIR validation and compilation (`validate`, `compile`); `compile --run-bundle` also writes a run bundle. Graph execution uses `GraphExecutor` or `compile_and_execute` in Python. |
| `relic.extensions.cli` | Extension discovery, inspection, scaffolding, and manifest validation. |
| `relic.interfaces.cli.run_layered` | Layered runtime shell for explicitly configured state-oriented runs. |

The layered runner does not discover or register a complete scientific model
automatically. Use the Python runtime workflow when Functor registration and
state ownership must be explicit.

## Import diagnostics

Use a short import check after installation:

```powershell
python -c "import relic; from relic.core import RELICSimulator, StateManager; print(relic.__version__)"
```

Typical failures are:

| Failure | Meaning | Action |
| --- | --- | --- |
| `ModuleNotFoundError: relic` | The package is not installed in the active interpreter. | Activate `RELIC` and run the editable installation command. |
| `ImportError` for an optional backend | A feature-specific dependency is unavailable. | Install the relevant project extra or use a supported backend already present in the environment. |
| `AttributeError` on a root package name | The name is not a root export. | Import from the canonical owner shown in the API reference. |
| CUDA initialization error | The requested runtime device cannot be resolved. | Use `device: "auto"` or follow [Devices and Distributed Execution](devices_and_distributed.md). |

## Related documentation

- [Architecture](../architecture.md)
- [Workflow index](index.rst)
- [Core Runtime API](../api/core_runtime.md)
- [Scientific Structure and Extensions API](../api/scientific_structure_extensions.md)
- [Science index](../science/index.rst)
- [Theory index](../theory/index.rst)
