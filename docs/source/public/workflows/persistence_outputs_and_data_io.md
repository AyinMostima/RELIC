# Persistence, Outputs, and Data I/O

RELIC provides three distinct persistence layers:

1. `StateManager` snapshots for in-memory transactional restore.
2. `RELICSimulator` checkpoints for runtime continuation.
3. Data and provenance writers for durable scientific outputs.

These layers serve different contracts and are not interchangeable.

## State snapshots

`StateManager.checkpoint()` returns an in-memory mapping containing state
tensors, counters, diagnostics, history, conservation records, and pending
provenance transaction data. `restore(snapshot)` accepts only a valid mapping
produced by that state manager contract.

```python
import torch

from relic.core.state import StateManager
from relic.core.state.types import StateVariableSpec, UpdateItem, UpdateMode

manager = StateManager(
    {"x": StateVariableSpec(name="x", units="dimensionless")}
)
manager.initialize({"x": torch.tensor(1.0)})

snapshot = manager.checkpoint()
manager.apply_updates(
    {
        "x": UpdateItem(
            value=torch.tensor(1.0),
            mode=UpdateMode.ACCUMULATE,
            unit="dimensionless",
            per_timestep=True,
        )
    },
    check_conservation=False,
)
manager.restore(snapshot)

print(float(manager.get_state()["x"]))
```

Expected output:

```text
1.0
```

`restore()` raises `TypeError` for a non-mapping snapshot and `ValueError`
when the required state mapping is absent. Use snapshots around local
transactions; they are not a stable cross-version file format.

## Simulator checkpoints

```text
RELICSimulator.create_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]

RELICSimulator.restore_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    compatibility_certificate: RestartCompatibilityCertificate | None = None,
    migration_certificate: GraphMigrationCertificate | None = None,
) -> dict[str, Any]
```

`relic\core\simulator\checkpoint.py` is the sole owner of the durable
checkpoint format. Without `snapshot`, the public create call captures the
live state and requires an explicit, non-existing target path; an existing
file raises `FileExistsError`. It creates parent directories, writes a unique
sibling temporary file, and atomically commits the target. `snapshot` is a
frozen checkpoint payload captured at an accepted step boundary for a
checkpoint delivery: it is written as given and never rebuilt from live
state, and an existing target is accepted only when its verified payload is
identical, in which case the receipt is returned again; a different file
raises `FileExistsError`. The call never selects an implicit output directory
and never overwrites an existing checkpoint.

```python
from pathlib import Path
import tempfile
import torch

from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.core.state.types import StateVariableSpec
from relic.scientific.atmosphere.stochastic_temperature import (
    OrnsteinUhlenbeckTemperatureFunctor,
)


def build_stochastic_simulator(seed: int) -> RELICSimulator:
    state_mgr = StateManager(
        {
            "stochastic_air_temperature_anomaly": StateVariableSpec(
                name="stochastic_air_temperature_anomaly",
                units="K",
            )
        },
        device=torch.device("cpu"),
    )
    state_mgr.initialize(
        {"stochastic_air_temperature_anomaly": torch.tensor([0.5], dtype=torch.float64)}
    )
    sim = RELICSimulator(
        {"time_step": 3600.0, "device": "cpu", "conservation_checks": False}
    )
    sim.initialize(state_mgr, DependencyGraphResolver())
    sim.register_functor(
        OrnsteinUhlenbeckTemperatureFunctor(config={"seed": seed}),
        execution_group="fast",
    )
    return sim


with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_path = Path(tmpdir) / "restart" / "step_000001.pt"
    simulator = build_stochastic_simulator(seed=42)
    simulator.step(3600.0)
    create_receipt = simulator.create_checkpoint(checkpoint_path)
    simulator.step(3600.0)
    continuous_val = float(
        simulator.get_state()["stochastic_air_temperature_anomaly"].detach()[0]
    )

    restored_simulator = build_stochastic_simulator(seed=42)
    restore_receipt = restored_simulator.restore_checkpoint(checkpoint_path)
    restored_simulator.step(3600.0)
    restarted_val = float(
        restored_simulator.get_state()["stochastic_air_temperature_anomaly"].detach()[0]
    )
    print(create_receipt["sha256"] == restore_receipt["sha256"])
    print(continuous_val == restarted_val)
```

Both operations return `path`, `format_version`, `size_bytes`, file `sha256`,
canonical `payload_sha256`, and `provider_epoch`. Creation leaves
`provider_epoch=None`; a successful restore reports the newly committed view
epoch. Automatic checkpoint names are
`checkpoint_directory/<run_id>/checkpoint_step_<12-digit-step>.pt`.
`run_id` belongs to the simulator, survives restart, and changes on reset.

Format 3 includes the frozen `delivery_state`: accepted pending obligations,
receipts, job definitions, and flat checkpoint-node references. Restoring a
checkpoint reconciles accepted notifications and delivery work through the
existing owners; it does not recapture newer live outputs or deliver effects
during state restoration. Retry retains each original run and destination.
Delivery is at-least-once: every obligation carries an idempotency key and a
receipt, so redelivery to an idempotent consumer is effectively-once.
Cross-system exactly-once delivery requires the counterpart to take part in
the protocol and is not claimed.
Only obligations present in the persisted checkpoint are durably recoverable.

The versioned `relic.simulator.checkpoint` payload contains:

- simulator time, counters, configuration-relevant runtime fields, and output
  history;
- one validated in-memory `StateManager.checkpoint()` snapshot;
- every registered Functor's `{state_dict, parameter_bindings}` entry: standard
  PyTorch module state (including safe `get_extra_state()` data) and live
  values of manager bindings held by reference, including nested Functors;
- Functor topology, dependency order/groups, StateManager schema, scientific
  configuration/metadata/parameter contracts, resolved devices, and
  distributed rank/world-size signature;
- Python, Torch CPU, optional legacy global NumPy, and CUDA random number generator (RNG) states;
- a canonical payload Secure Hash Algorithm 256-bit (SHA-256) integrity record.

Restore uses `torch.load(..., map_location="cpu", weights_only=True)`, requires
the exact `relic.simulator.checkpoint` format and `format_version=3`, and has
no unsafe deserialization fallback. Only version 3 is written. A version-2
file is upgraded in memory: held bindings move from module state into
`parameter_bindings`; an unknown state entry raises `ValueError`. The
upgraded payload is resealed and validated, and `format_upgrade` in the
restore receipt records `from_format_version`, `to_format_version`, and
`moved_bindings`. Binding restore requires matching key sets, shape, and
dtype and preserves the live object, gradient flag, and device.
CUDA RNG presence must match the resolved runtime symmetrically, including the
CUDA device count; when pinning `RELICSimulator` to `"device": "cpu"` on a
CUDA-capable host, pass `device=torch.device("cpu")` to an externally
constructed `StateManager` so both owners resolve CPU tensors symmetrically.
The current single-file format fails closed on both create
and restore when an active distributed runtime has `world_size > 1`.

Restore requires a prepared, compatible simulator. It validates payload
integrity, topology, scientific contracts, devices, distributed context,
StateManager state, Functor state dictionaries, and RNG data before applying
the restart. A failure before provider-view publication restores the pre-restore
runtime fields, StateManager, Functor state, RNG, accepted pending work, and
registration metadata. Once the new provider view is published, its epoch
and scientific state remain committed even if retirement, cleanup, or
receipt I/O fails; that error is reported as post-publication failure and
must not rewind a world that readers may already have observed.

A checkpoint does not capture the state of external services or data sources.
An external adapter stores only its source and model binding
(`get_extra_state()`), and restoring that binding into an adapter bound to
another source or model raises `ValueError`. Its opened datasets are not part
of the checkpoint; a restarted run opens them again through `discover()` and
`open()`. Adapter `forward(state, time, dt, *, calendar_type, time_origin)`
selects forcing at `time_origin + time` using its declared
`time_interpolation`; it does not own a separate mutable time cursor.
An explicitly bound forcing sequence has the simulator-owned cursor, which
advances only on accepted steps and is restored by the checkpoint.

The default restore is exact. The same owner accepts a
`RestartCompatibilityCertificate` for bounded declared contract/parameter
differences and state renames that preserve units, shape, and dtype. It binds
the actual source and target topology digests; it cannot waive kernel,
implementation, Functor-type, execution-order, or runtime-environment changes.
Its forcing policy either preserves the exact cursor or explicitly clears it.
`GraphMigrationCertificate` handles declared renaming/addition/retirement in
its restricted graph family, including history requirements and mapped owners.
The two certificate arguments are mutually exclusive. Neither provides
arbitrary code equivalence, unit conversion, or shape migration.

`StateManager.checkpoint()` remains an in-memory transaction primitive. It is
embedded in the durable simulator payload but is not an independent file
format or a second persistence owner. Likewise, Functors use
`state_dict()`/`load_state_dict()` and PyTorch
`get_extra_state()`/`set_extra_state()` without introducing a parallel restart
API. Stochastic Functors built on `FunctorSubstreamMixin`, including
`OrnsteinUhlenbeckTemperatureFunctor` (`relic.scientific.atmosphere.stochastic_temperature`)
and `MarkedPoissonRainfallFunctor` (`relic.scientific.atmosphere.stochastic_rainfall`),
persist their Wiener or marked-jump substream cursor in `extra_state`, so
accepted-step rollback and `restore_checkpoint()` replay the identical
stochastic sample path. Manager bindings held by reference are outside module
state; the same checkpoint owner captures and restores them through
`parameter_bindings`.

Automatic checkpoints are disabled by default. Enabling them requires all
three runtime settings:

```python
from pathlib import Path
import tempfile

from relic.core.simulator import RELICSimulator

with tempfile.TemporaryDirectory() as tmpdir:
    simulator = RELICSimulator(
        {
            "enable_checkpointing": True,
            "checkpoint_directory": str(Path(tmpdir) / "automatic"),
            "checkpoint_frequency": 3600.0,
        }
    )
    print(simulator.enable_checkpointing)
```

The frequency must be positive. Automatic persistence occurs only after an
accepted scientific and provenance step commit, using
`checkpoint_step_<12-digit-step>.pt` in the explicit directory's `run_id`
subdirectory. File errors
propagate, and there is no public checkpoint cleanup API.

A bitwise-equal uninterrupted run and split restart run is strong dynamic
evidence only for the tested graph, inputs, fixed-step protocol, dependency
stack, and device environment. It does not promise cross-platform,
cross-version, cross-device, or multi-rank bitwise identity.

## Run bundles

```python
from pathlib import Path
import tempfile

from relic.orchestration import GraphIR, NodeRegistry, SSOTCatalog, compile_graph
from relic.orchestration.compiler import write_run_bundle

with tempfile.TemporaryDirectory() as tmpdir:
    graph = GraphIR(
        schema_version="1.0",
        nodes={},
        metadata={"name": "bundle_demo"},
    )
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    plan = compile_graph(graph, registry, strict=True)
    artifacts = write_run_bundle(
        graph,
        plan,
        Path(tmpdir) / "run_bundle",
    )
    print(len(artifacts), "manifest" in artifacts)
```

```text
write_run_bundle(
    graph: GraphIR,
    plan: ExecutionPlan,
    output_dir: pathlib.Path,
    *,
    run_id: str | None = None,
    input_files: list[pathlib.Path] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]
```

`write_run_bundle` is imported from `relic.orchestration.compiler`; the
`relic.orchestration` package does not export it. `graph` must be the
canonical `GraphIR` that `plan` was compiled from; another type raises
`TypeError` and a different graph raises `ValueError`. `run_id` names the
provenance run (default: the current UTC time, `YYYYMMDDTHHMMSSZ`).
`input_files` lists the input files to hash; when it is omitted,
`graph.metadata["input_files"]` is used. `metadata` adds caller entries to the
bundle metadata; the keys `schema_version` and `graph_metadata` are reserved
and raise `ValueError`.

The bundle records the declarative workflow, compile result, runtime log
target, environment and repository evidence, input checksums when input files
are declared, and W3C Provenance Data Model (PROV) metadata.

| Artifact | Purpose |
| --- | --- |
| `workflow.json` | Serialized GraphIR. |
| `execution_plan.json` | Serialized execution steps. |
| `compile_report.json` | Validation, structural audit, adapters, and compile metadata. |
| `runtime.log.jsonl` | Runtime event stream target. |
| `env_snapshot.txt` | Execution-environment snapshot. |
| `git_status.json` | Repository-state evidence. |
| `inputs_manifest.sha256` | Hash manifest for declared input files. |
| `prov.jsonld` | Machine-readable provenance graph. |
| `manifest.json` | Run identity, metadata, and the path of every bundle artifact. |
| `checksums.sha256` | Hashes of the bundle files. |

`write_run_bundle()` creates the target directory, verifies the sealed bundle,
and returns normalized artifact paths. It raises `TypeError` if the
provenance recorder does not return the required mapping.

## Conservation diagnostic output

```python
from pathlib import Path
import tempfile
import torch

from relic.core.state import StateManager
from relic.core.state.types import StateVariableSpec, UpdateItem, UpdateMode
from relic.data.io.diagnostic_writer import export_diagnostics_to_netcdf

state_manager = StateManager(
    {
        "carbon_pool": StateVariableSpec(
            name="carbon_pool",
            units="kg C m-2",
            conserved=True,
            accumulates="carbon.total",
        )
    }
)
state_manager.initialize({"carbon_pool": torch.tensor(2.0)})
state_manager.apply_updates(
    {
        "carbon_pool": UpdateItem(
            value=torch.tensor(0.0),
            mode=UpdateMode.ACCUMULATE,
            unit="kg C m-2",
            per_timestep=True,
            element="carbon",
        )
    },
    dt=3600.0,
    check_conservation=True,
)

with tempfile.TemporaryDirectory() as tmpdir:
    diag_result = export_diagnostics_to_netcdf(
        summaries=state_manager.conservation_history,
        time_coords=[3600.0],
        output_path=Path(tmpdir) / "diagnostics.nc",
        reference_date="1850-01-01",
    )
    print(diag_result.validation_passed)
```

```text
export_diagnostics_to_netcdf(
    summaries: list[ConservationSummary],
    time_coords: list[float],
    output_path: pathlib.Path,
    reference_date: str = "1850-01-01",
    *,
    delivery_key: Optional[str] = None,
    metadata_registry: Optional[dict[str, dict[str, Any]]] = None,
) -> WriteResult
```

The writer converts conservation summaries to an xarray dataset and delegates
Climate and Forecast (CF) attribute synchronization and output creation to `DataWriter`, without
adding time bounds. It returns the `WriteResult` and raises `OSError` when the
write or its output validation fails. `metadata_registry` replaces the
default per-variable diagnostic metadata. `delivery_key` is stored as the
global attribute `relic_delivery_key`, so a retried delivery with the same key
and options keeps the original creation timestamp.
`summaries` and `time_coords` must describe the same sequence of simulation
steps.

## `DataIO`

```python
from relic.data.io import DataIO, ReadConfig, WriteConfig

io = DataIO(
    reader_config=ReadConfig(cf_validate=True, cf_standardize=True),
    writer_config=WriteConfig(cf_sync=True, validate_output=True),
)
```

```text
DataIO(
    reader_config: Optional[ReadConfig] = None,
    writer_config: Optional[WriteConfig] = None,
)
```

### Read configuration

| `ReadConfig` field | Default | Meaning |
| --- | --- | --- |
| `format` | `AUTO` | Detect the supported source format. |
| `chunks` | `None` | Optional xarray chunk mapping. |
| `parallel` | `True` | Enable supported parallel I/O behavior. |
| `cache_size` | `128` | Reader cache size in MB. |
| `cf_validate` | `True` | Check each variable's explicit `standard_name` and `units` with `CFConventions.validate_cf_compliance`; the check runs only inside the `cf_standardize` step. |
| `cf_standardize` | `True` | Run the explicit-metadata CF step. It never renames variables, infers attributes, or changes values or units. With `cf_validate`, invalid metadata sets `ReadResult.success` and `ReadResult.validation_passed` to `False`, or raises `ValueError` when `strict_cf` is set. The step always adds a warning that dataset-level CF conformance was not evaluated. |
| `strict_cf` | `False` | Raise for metadata (CF standard-name and unit) findings; without it such findings are recorded and the result is marked failed. Every other failure (a failed processing pipeline, tensor conversion, write, chunking request or file close) raises in both modes; raw or partial data are never returned in its place. |
| `process_time` | `True` | Process time coordinates. |
| `time_bounds` | `True` | Compute missing time bounds where supported. |
| `standard_calendar` | `False` | Keep the source calendar. `True` accepts only `gregorian` and raises for any other calendar, because no cross-calendar conversion exists. |
| `variables` | `None` | Variables to include; `None` means all. |
| `exclude_vars` | `None` | Variables to exclude. |
| `standard_names_only` | `False` | Return only variables with standard-name support. |
| `apply_unit_conversion` | `False` | Convert CF-consistent variables to `target_units` or to the CF canonical unit of their `standard_name` (see below). |
| `target_units` | `None` | Variable-to-target-unit mapping; each unit must be CF-consistent with the variable's `standard_name`. |
| `file_path` | `None` | Not read by the reader; pass the path to the read call. |
| `validate_data` | `True` | Run data validation. |
| `fill_missing` | `False` | Fill supported missing coordinate data. |
| `quality_flags` | `False` | Accepted but not used; the reader adds no quality flags. |
| `to_torch` | `False` | Return a tensor mapping in place of an xarray dataset. |
| `torch_dtype` | `torch.float32` | Tensor output type. |
| `torch_device` | `"cpu"` | Tensor output device. |

### Unit conversion on read

With `apply_unit_conversion=True`, the reader converts each variable that has
both `units` and `standard_name` attributes. The target is `target_units[name]`
when given, otherwise the canonical unit of the standard name in the CF table.
Variables without these attributes, or whose standard name defines no
canonical unit, are left unchanged. The source and the target unit must both
be CF-consistent with the standard name.

The values are converted only by the shared `UnitConverter`. It uses an
explicit conversion rule when one exists; otherwise it converts a pair of
units of the same dimension that differ only by a scale factor through
UDUNITS, for example `kg m-2 day-1` to `kg m-2 s-1`. Affine pairs without a
rule, such as `degF` to `K`, and pairs of different dimensions still need an
explicit rule. The reader passes an empty `ConversionContext`: it supplies no
temperature, pressure, or density, so a conversion that needs one fails.

A converted variable keeps `standard_name` and its other attributes, gets the
new `units`, and records the previous unit in `original_units`; a note is
added to `ReadResult.warnings`. A variable with a CF-inconsistent unit, or one
that cannot be converted, keeps its values and units, and the read is marked
failed: `ReadResult.success` and
`ReadResult.validation_passed` become `False` and the reason is added to
`warnings`. With `strict_cf=True` the read raises `ValueError` instead.
`DataIO.read()` returns only the data, so inspect the `ReadResult` from
`DataIO.reader.read_dataset(...)` or `convert_format(...)`, or set
`strict_cf=True`.

Conversions of model state that need context take it from
`UnitConverter.get_conversion_context(state, *, layer_thickness_variable=None, layer_thickness=None, temperature=None, pressure=None)`.
The layer thickness (the context field `soil_depth`, in m) comes only from the
state variable that the converted variable declares as its layer thickness
(the `layer_thickness_variable` of its variable contract, which the transport
bridge reads), read pointwise, or from an explicit `layer_thickness`. Giving both raises
`ValueError`, as does a named variable that is missing from `state`; no fixed
state name such as `soil_depth` or `root_zone_depth` is looked up.
Temperature is read only from `air_temperature` and pressure only from
`atmospheric_pressure`; the alias `air_pressure` is resolved upstream by the
ContractRegistry, and no other name is tried. A conversion that needs another
temperature or pressure, such as a leaf or soil temperature, receives it
through the explicit `temperature` [K] or `pressure` [Pa] argument, which
takes precedence over the state. `water_density` is taken from `state` when
present. State fields are kept as tensors. No default values are filled in: a
field that stays `None` makes the conversion that needs it raise.

### Write configuration

| `WriteConfig` field | Default | Meaning |
| --- | --- | --- |
| `format` | `NETCDF4` | Output format. |
| `compression` | `ZSTD` | Compression method. |
| `compression_level` | `4` | Compression level from 1 through 9. |
| `chunks` | `None` | Explicit chunk mapping. |
| `auto_chunk` | `True` | Select chunks automatically. |
| `chunk_time_first` | `True` | Prefer time-oriented chunks. |
| `cf_sync` | `True` | Synchronize CF attributes. |
| `strict_cf` | `False` | Raise for metadata (CF standard-name and unit) findings; without it such findings are recorded and the result is marked failed. Every other failure (a failed processing pipeline, tensor conversion, write, chunking request or file close) raises in both modes; raw or partial data are never returned in its place. |
| `add_bounds` | `True` | Add coordinate bounds when supported. |
| `update_history` | `True` | Update the dataset history attribute. |
| `title`, `institution`, `source`, `comment`, `references` | `None` | Optional global metadata. |
| `parallel` | `False` | Sequential writing; `True` raises `NotImplementedError`. |
| `validate_output` | `True` | Validate the written output. |
| `create_dirs` | `True` | Create missing parent directories. |

## Read, write, and convert

```python
from pathlib import Path
import tempfile
import xarray as xr

from relic.data.io import DataIO, ReadConfig, WriteConfig

io = DataIO(
    reader_config=ReadConfig(cf_validate=True, cf_standardize=True),
    writer_config=WriteConfig(cf_sync=True, validate_output=True),
)
source_dataset = xr.Dataset(
    {
        "air_temperature": (
            ("time",),
            [288.15, 289.15],
            {"standard_name": "air_temperature", "units": "K"},
        ),
        "precipitation_flux": (
            ("time",),
            [1.0e-5, 2.0e-5],
            {"standard_name": "precipitation_flux", "units": "kg m-2 s-1"},
        ),
    },
    coords={
        "time": (
            ("time",),
            [0.0, 3600.0],
            {
                "standard_name": "time",
                "units": "seconds since 2000-01-01 00:00:00",
                "calendar": "noleap",
            },
        )
    },
)

with tempfile.TemporaryDirectory() as tmpdir:
    forcing_path = Path(tmpdir) / "forcing.nc"
    io.write(source_dataset, forcing_path, format="netcdf4")

    dataset = io.read(
        forcing_path,
        variables=["air_temperature", "precipitation_flux"],
    )
    tensors = io.read(
        forcing_path,
        variables=["air_temperature"],
        to_torch=True,
        torch_device="cpu",
    )
    write_result = io.write(
        dataset,
        Path(tmpdir) / "forcing_standardized.nc",
        format="netcdf4",
    )
    read_result, conversion_result = io.convert_format(
        forcing_path,
        Path(tmpdir) / "forcing.zarr",
        "zarr",
    )
    print(sorted(tensors), write_result.validation_passed, conversion_result.validation_passed)
```

| Method | Return value |
| --- | --- |
| `read(source, **kwargs)` | `xarray.Dataset` or `dict[str, Tensor]` |
| `read_variables(source, variables, **kwargs)` | Variable mapping |
| `read_tensors(source, variables, device="cpu", dtype=torch.float32)` | Tensor mapping |
| `write(data, output_path, **kwargs)` | `WriteResult` |
| `write_tensors(tensors, output_path, coordinates=None, metadata=None, **kwargs)` | `WriteResult` |
| `convert_format(source_path, target_path, target_format, **kwargs)` | `(ReadResult, WriteResult)` |
| `get_file_info(file_path)` | File metadata mapping; a file whose metadata cannot be read raises (no mapping carrying an `error` entry). |

`ReadResult` retains loaded data, metadata, CF and validation flags, warnings,
timing, file size, time-processing status, and standardized-variable mappings.
`WriteResult` reports the output path, size, timing, compression ratio, CF and
validation flags, warnings, variables, coordinates, and computed bounds.

Interpret every CF result by its declared scope. The existing
`CFConventions.validate_cf_compliance(...)` checks one variable's official
standard name and compatible units; it reports
`single_variable_standard_name_and_units`, not whole-dataset conformance.
`CFConventions.validate_dataset_conformance(...)` evaluates
`declared_dataset_scope`: the Conventions declaration, units, official names,
axis values, time units/calendar, coordinates and bounds references/shapes,
missing-value consistency, and cell-method syntax/dimensions. Grid mappings,
discrete sampling geometries, gathering compression, external variables,
and group hierarchies are outside that scope. Name Table v94 membership
cannot itself authorize a dataset Conventions declaration.

The external `IOHandler.read(...)` reports its dataset check through
`IOResult.cf_validation_scope`, `dataset_conformance_checked`, and
`cf_conformance_issues`; `cf_compliant=True` requires that the declared check
actually ran and found no issue. Other result types and unchecked paths must
not inherit this claim merely from a similarly named boolean field.

Supported read formats include NetCDF4, NetCDF3, HDF5, Zarr, and CSV.
Supported write formats include NetCDF4, NetCDF3, Zarr, and CSV. A missing
file raises `FileNotFoundError`; unsupported data or formats raise
`ValueError`; strict CF violations raise `ValueError`.

For conversion overrides, prefix reader-only options with `read_`:

```python
from pathlib import Path
import tempfile
import xarray as xr

from relic.data.io import DataIO

io = DataIO()
sample = xr.Dataset(
    {
        "air_temperature": (
            ("time",),
            [288.15, 289.15],
            {"standard_name": "air_temperature", "units": "K"},
        )
    },
    coords={
        "time": (
            ("time",),
            [0.0, 3600.0],
            {
                "standard_name": "time",
                "units": "seconds since 2000-01-01 00:00:00",
                "calendar": "noleap",
            },
        )
    },
)

with tempfile.TemporaryDirectory() as tmpdir:
    src = Path(tmpdir) / "forcing.nc"
    dst = Path(tmpdir) / "forcing_subset.nc"
    io.write(sample, src, format="netcdf4")
    read_res, conv_res = io.convert_format(
        src,
        dst,
        "netcdf4",
        read_variables=["air_temperature"],
        compression="zstd",
        compression_level=6,
    )
    print(read_res.success, conv_res.validation_passed)
```

## Experimental data quality and streaming gates

Two experimental data modules remain gated by explicit environment variables
checked at import time:

- `relic.data.quality` (`QualityValidator`, `RangeRule`, `ValidationIssue`)
  requires `RELIC_EXPERIMENTAL_DATA_QUALITY=1` before import; otherwise
  `require_experimental_feature` raises `RuntimeError`.
- `relic.data.streaming` requires `RELIC_EXPERIMENTAL_STREAMING=1` before
  import; otherwise the exported names raise `StreamingDisabledError` (or
  `RuntimeError`).

### Streaming windows

```text
StreamWindow(
    size: timedelta,
    step: Optional[timedelta] = None,
    aggregation_method: str = "mean",
    quality_checks: bool = True,
)
```

| Field | Meaning |
| --- | --- |
| `size` | Window length; a window is complete when the newest record is at least `size` after the window start. The reduced output is stamped at the window centre. |
| `step` | Advance of the window start; `None` uses `size`. |
| `aggregation_method` | Reduction over time: `mean`, `sum`, `min`, or `max`, written as `cell_methods = "time: <method>"`. Any other value, including `"MEAN"`, raises `ValueError` when the window is built. |
| `quality_checks` | Run quality control on each window. |

The window has no coverage or gap fields; passing `min_data_fraction` or
`max_gap_duration` raises `TypeError`. Coverage and gap limits come from the
quality-control parameters `coupling.scales.temporal.coverage_threshold` and
`coupling.scales.temporal.quality_control.*`.

With `quality_checks=True`, the window time series of each variable is
checked before the reduction with
`QualityController.validate_for_operation(series, time_coord, "aggregation")`.
A series fails when its valid fraction is below the coverage threshold or its
quality grade is below `acceptable`; the reduced variable then gets the
attribute `quality_flags = "validation_failed"`. A variable without a time
dimension gets `"not_assessed"`. A numeric time axis needs CF `units` and
`calendar` attributes, and window data without a `time` dimension raises.
Errors propagate to the stream consumer; a failed window is not dropped with a
warning.

## CF attributes after scaling

`CFAttributeSynchronizer` in `relic.data.standards.cf_conventions` carries
explicit CF metadata through a processing step. It does not change or
unit-convert the data tensor, and it does not infer units, variable roles,
grids, or coordinates. Each `sync_after_*` method passes the data and
metadata through `CFConventions.sync_variable_attributes`, which derives two
metadata values from the data: `shape` (`list(data.shape)`) and, for
non-empty data, `actual_range` (`[float(data.min()), float(data.max())]`).
The range is `[nan, nan]` when the data contain NaN. For empty data
`actual_range` is not written, so a value already in the input metadata is
kept. The method-specific attributes in the table below are written on top
of these two fields.

`sync_variable_attributes` first checks the standard name and units of the
input metadata and raises `ValueError` for an issue even when `strict_mode`
is `False`; the only exception is when every issue is an undefined unit
contract for the standard name. With `strict_mode=True`, the updated
metadata is checked again, and an issue found by this bounded CF check
raises `ValueError`.

| Method | Behavior |
| --- | --- |
| `sync_after_aggregation(data, metadata, aggregation_method, dimension="time", *, source_label, target_label)` | Sets the `<dimension>: <method>` cell method and appends `Aggregated <dimension> from <source_label> to <target_label> (<dimension>: <method>)` to `history`. |
| `sync_after_disaggregation(data, metadata, disaggregation_method, dimension="area", *, source_label, target_label)` | Same, with a `Disaggregated ...` entry. The allowed cell methods depend on `dimension`: `sum` (each fine cell holds its own total) or `mean` (each fine cell holds a mean) for `area`, and `sum`, `mean`, or `point` (instantaneous states) for `time`. Other methods and other dimensions raise `ValueError`. |
| `sync_after_regridding(data, metadata, regrid_method, source_grid, target_grid)` | Requires `shape` in both grids and a non-empty string for each name in `CF_REGRID_TARGET_GRID_FIELDS` (`grid_mapping`, `lat_name`, `lon_name`) in `target_grid`; writes `grid_mapping` and `coordinates` and appends a regridding entry to `history`. |
| `sync_after_temporal_aggregation(data, metadata, time_coord, operation, method, coverage_stats, *, source_label, target_label)` | Calls `sync_after_aggregation` (`operation="aggregation"`) or `sync_after_disaggregation` (`operation="disaggregation"`) with `dimension="time"`. The variable keeps its `standard_name`, `units`, and `long_name`, gets the `time: <method>` cell method, and gains the history entry `Aggregated time from <source_label> to <target_label> (time: <method>)` (or `Disaggregated time ...`). Coverage fields come from the required `coverage_stats` (`coverage_fraction`, or `valid_points` and `total_points`): `coverage_fraction`, `coverage_threshold` read from `coupling.scales.temporal.coverage_threshold`, and `quality_state` (`good_data` or `insufficient_coverage`). The attributes of the result time coordinate are returned under `time_coordinate`, with `bounds` only when the coordinate has bounds; in strict mode only these time-coordinate attributes get the temporal check. |

The source and target labels are required, non-empty scale labels such as
`"hourly"` and `"daily"`; a missing or empty label raises.

## Data I/O command checks

```powershell
python -m relic.interfaces.cli.main data io check
python -m relic.interfaces.cli.main data io read data\forcing.nc
python -m relic.interfaces.cli.main data io convert data\forcing.nc output\forcing.zarr --target-format zarr
```

Use `--help` on the selected subcommand before an operation because
format-specific arguments are command-specific.

`data io check` runs the I/O health checks. The I/O performance check is an
error when its read or write benchmark fails (a failed memory benchmark lowers
a healthy result to a warning); the streaming throughput check of the
experimental `data stream check` is an error when a benchmark raises.

## Related documentation

- [Architecture](../architecture.md)
- [State, Functors, and Simulation](state_functors_and_simulation.md)
- [Graph Validation and Execution](graph_validation_and_execution.md)
- [Orchestration](../orchestration.md)
- [Science index](../science/index.rst)
- [Theory index](../theory/index.rst)
