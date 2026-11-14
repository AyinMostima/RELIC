# Operations Quick Start

This page is the shortest supported route into RELIC. Detailed procedures,
arguments, return values, exceptions, and current runtime limitations live in
the task-specific workflow pages; they are not duplicated here.

## Select a workflow

| Task | Detailed guide |
| --- | --- |
| Install, import, and inspect package availability | [Installation and imports](workflows/installation_and_imports.md) |
| Load configuration and retrieve parameters | [Configuration and parameters](workflows/configuration_and_parameters.md) |
| Define state, register Functors, and run the simulator | [State, Functors, and simulation](workflows/state_functors_and_simulation.md) |
| Validate, compile, and execute GraphIR | [Graph validation and execution](workflows/graph_validation_and_execution.md) |
| Snapshot state, restart, write output, and use Data I/O | [Persistence, outputs, and Data I/O](workflows/persistence_outputs_and_data_io.md) |
| Couple internal scales or external models | [Coupling workflows](workflows/coupling_workflows.md) |
| Calibrate parameters or run sensitivity and uncertainty analysis | [Calibration, sensitivity, and UQ](workflows/calibration_sensitivity_and_uq.md) |
| Select devices or initialize distributed execution | [Devices and distributed execution](workflows/devices_and_distributed.md) |
| Use CLI modules or load an extension | [CLI and extensions](workflows/cli_and_extensions.md) |

## Environment preflight

`pyproject.toml` specifies `Python >= 3.11` and `torch >= 2.0`, with optional
development dependencies under the `[dev]` extra. Workloads are separated across
three Conda environments:

| Conda environment | Assigned workload |
| --- | --- |
| `RELIC` | Model execution, scientific simulation, and model-dependent preparation |
| `dataanalysis` | Independent data preparation, postprocessing, analysis, and figure generation |
| `sphinx_env` | Public documentation generation |

Use the `RELIC` environment and the module entry points provided by the
package:

```powershell
conda activate RELIC
python -B -c "import relic; print(relic.__version__)"
python -B -m relic.interfaces.cli.main --help
python -B -m relic.orchestration --help
python -B -m relic.extensions.cli --help
```

The package does not declare a bare `relic` console script. The Python module
commands above are the stable callable surfaces documented for this
architecture version:

- `python -m relic.interfaces.cli.main` exposes seven top-level subcommands:
  `params`, `optimization` (alias `opt`, `[N]` `runtime_unbound`; `FourDVar` in
  `relic.optimization.assimilation` is an unimplemented `[N]` framework shell
  without an adjoint model), `data`, `compute`, `stochastic` (alias `stoch`),
  `orchestration` (alias `graph`), and `healthcheck` (alias `health`).
- `python -m relic.orchestration` exposes `validate` and `compile` for
  `GraphIR` files.
- `RELICSimulator(None)` normalizes `None` to the empty configuration `{}`,
  defaulting `enable_auto_scale` to `False`. `resolve_device(...)` follows a
  GPU-first policy for distributed and `auto` selection with automatic fallback
  to CPU when CUDA is unavailable (unless `require_gpu=True`), whereas an
  explicit `enable_distributed: true` setting that cannot be satisfied raises
  `RuntimeError`.

## First successful run

Use the [complete minimal example](workflows/state_functors_and_simulation.md#complete-minimal-example)
below as the installation benchmark. Save it to a temporary script and run it
from the repository root or from an environment where the package is installed:

```python
import torch

from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.functor import BiologicalProcessFunctor, FunctorMetadata
from relic.core.functor.functor_types import (
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.core.state.types import StateVariableSpec, UpdateItem, UpdateMode


class IncrementFunctor(BiologicalProcessFunctor):
    def __init__(self) -> None:
        metadata = FunctorMetadata(
            name="increment",
            description="Minimal governed state increment example.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            mathematical_expression="x_next = x + 1",
            references=["Executable documentation example"],
            input_variables=["x"],
            output_variables=["x"],
            spatial_scale=SpatialScale.POINT,
            variable_kind=VariableKind.STATE,
        )
        super().__init__(name="increment", config={}, metadata=metadata)

    def forward(self, state, time, dt, **kwargs):
        return {
            "x": UpdateItem(
                value=torch.ones_like(state["x"]),
                mode=UpdateMode.ACCUMULATE,
                unit="dimensionless",
                per_timestep=True,
                source_functor=self.name,
            )
        }


manager = StateManager(
    {"x": StateVariableSpec(name="x", units="dimensionless")}
)
manager.initialize({"x": torch.tensor(1.0)})

simulator = RELICSimulator({"time_step": 60.0})
simulator.initialize(manager, DependencyGraphResolver())
simulator.register_functor(IncrementFunctor(), execution_group="fast")

updates = simulator.step(60.0)
print(float(simulator.get_state()["x"]))
print(sorted(updates))
print(simulator.step_count)
```

```powershell
cd C:\path\to\RELIC
conda activate RELIC
python -B relic_minimal_run.py
```

The expected output is:

```text
2.0
['x']
1
```

Interpret the three lines separately: the managed state advanced from 1 to 2,
the simulator returned the single governed update key, and one step committed.
The example is complete only when all three values match. It verifies the
basic StateManager/Functor/simulator lifecycle; it does not validate a real
ecological process, conservation law, GraphIR assertion set, or long-run
numerical accuracy.

## Benchmark-style acceptance checklist

| Layer | What to inspect | Accept when |
| --- | --- | --- |
| Environment | Python path, RELIC version, Torch/device report | They match the intended environment and device policy. |
| Inputs | State schema, forcing interval, units, masks, parameter keys | Required values exist, are finite, and share the declared support and time basis. |
| Configuration | Resolved runtime and parameter configuration | No required value came from an undocumented fallback. |
| Graph/runtime | Registration summary, validation report, execution order | Owners are unique, blocking issues are empty, and the expected Functors execute. |
| Numerics | State values, residuals, conservation or analytic oracle | Values and residuals meet an explicit tolerance with compatible units. |
| Outputs | Expected files, variables, dimensions, timestamps, metadata | The artifact set is complete and readable, not merely present. |
| Restart | Checkpoint receipt and continuous/restart comparison | An exact restart reproduces the uninterrupted state within the tested setup; a certified compatible or migration restart reproduces only the certified scope. |
| Provenance | Configuration, code/source identity, hashes, logs | Every reported result can be traced to the same run. |

For scientific conservation, use a hand-integrable function, manufactured
solution, or independently calculated ledger as the numerical oracle.
"No exception" is never the conservation acceptance criterion.

## Minimal runtime sequence

```text
Create a configuration dictionary
  -> define and initialize StateManager
  -> construct each scientific Functor through its canonical bridge
  -> register Functors and execution groups
  -> initialize simulator state
  -> call step(...) or run(...)
  -> inspect state, diagnostics, and provenance
  -> checkpoint or export through the existing owners
```

Use the [architecture guide](architecture.md) to determine ownership, the
[scientific class reference](science/index.rst) to select processes, and the
[theory guide](theory/index.rst) to verify equations and applicability before
assembling a model.

## Troubleshooting order

Work from the outer contract toward the scientific kernel. This prevents a
unit or configuration fault from being misdiagnosed as a process-law defect.

| Order | Check | Typical evidence |
| --- | --- | --- |
| 1 | Environment and import identity | Python executable, package path, version, optional dependency import. |
| 2 | Configuration and parameter resolution | Resolved keys, namespaces, units, bounds, and source provenance. |
| 3 | State and input contracts | Missing variables, shape/device mismatch, non-finite data, time and support coverage. |
| 4 | Registration and GraphIR | Duplicate writers, unresolved dependencies, port/unit mismatch, failed structural assertion. |
| 5 | Numerical and scientific behavior | Analytic oracle, conservation ledger, domain bounds, entropy/energy sign, convergence or restart comparison. |
| 6 | Output and persistence | File receipt, variable metadata, timestamp coverage, hash, checkpoint compatibility. |

Keep the first failing layer as the active diagnosis. Do not add a fallback or
relax a downstream validation rule to hide an upstream contract error.

## Immediate stop conditions

Correct the model or configuration before continuing when:

- a required state variable, ParameterManager entry, unit, or shape is absent;
- more than one non-diagnostic Functor claims the same state output;
- graph validation or a required structural assertion fails;
- a strict reaction ledger lacks a conservation matrix or violates internal
  `C N = 0`; an open strict ledger must also declare its route, exchange,
  ledger rate, and units;
- clipping, overflow, an unmapped species, or an un-injected source lacks an
  explicit residual route;
- a requested GPU is unavailable while `require_gpu` is true; or
- a StateManager transaction, checkpoint restore, or external-data validation
  fails.

Warnings and diagnostics are observations; they are not alternate scientific
state-update paths.
