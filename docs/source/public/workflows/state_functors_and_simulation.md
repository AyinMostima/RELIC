# State, Functors, and Simulation

The executable RELIC lifecycle has one state owner and one simulation owner.
`StateManager` validates and commits tensors. Scientific Functors calculate
`UpdateItem` objects. `RELICSimulator` orders Functors, supplies time and
device context, commits each group call's updates through `StateManager`, and
wraps all group commits of a step in one accepted-step transaction.

## Runtime ownership

| Responsibility | Canonical class | Implementation |
| --- | --- | --- |
| Variable schema and tensor state | `StateManager` | `relic\core\state\manager.py` |
| Variable contract | `StateVariableSpec` | `relic\core\state\types.py` |
| Explicit state update | `UpdateItem` | `relic\core\state\types.py` |
| Scientific lifecycle | `FunctorBase` | `relic\core\functor\functor_base.py` |
| Scientific metadata | `FunctorMetadata` | `relic\core\functor\functor_types.py` |
| Contract-to-runtime integration | Four Functor bridge classes | `relic\core\functor\biological_bridge.py`, `transport_bridge.py`, `physical_field_bridge.py`, and `scale_coupling_bridge.py`; `functor_bridges.py` re-exports them |
| Registration and time integration | `RELICSimulator` | `relic\core\simulator\main.py` |
| Dependency ordering | `DependencyGraphResolver` | `relic\core\dependency_graph.py` |

The scientific architecture behind this lifecycle is described in
[Architecture](../architecture.md) and
[Scientific Structure and Extensions API](../api/scientific_structure_extensions.md).

## `StateVariableSpec`

```text
StateVariableSpec(
    name: str,
    units: str,
    bounds: Optional[tuple[float, float]] = None,
    conserved: bool = False,
    spatial_dims: int = 0,
    required: bool = True,
    default: Optional[torch.Tensor] = None,
    write_mode: StateWriteMode = StateWriteMode.MUTABLE,
    cf_standard_name: Optional[str] = None,
    kind: str = "state",
    owner: Optional[str] = None,
    writers: list[str] = [],
    aliases: list[str] = [],
    incrementable: bool = True,
    accumulates: Optional[str] = None,
    flux_in: list[str] = [],
    flux_out: list[str] = [],
    numerics: Optional[dict[str, float]] = None,
    description: str = "",
    default_value: Optional[float] = None,
    variable_type: StateVariableType = StateVariableType.SCALAR,
)
```

| Parameter | Meaning |
| --- | --- |
| `name` | Canonical state-variable identifier. |
| `units` | Units enforced when updates are committed. |
| `bounds` | Optional inclusive lower and upper limits. |
| `conserved` | Whether the variable participates in conservation accounting. |
| `spatial_dims` | Number of spatial dimensions represented by the tensor. |
| `required` | Whether initialization requires the variable. |
| `default` | Optional tensor default. |
| `write_mode` | `MUTABLE`, `IMMUTABLE`, or `DERIVED` write policy. |
| `cf_standard_name` | Optional Climate and Forecast (CF) standard name used by data and unit tooling. |
| `kind` | Semantic variable role, normally `state` for prognostic state. |
| `owner` | Declared primary owner. |
| `writers` | Functors authorized to write the variable. |
| `aliases` | Accepted alternate names. |
| `incrementable` | Whether incremental updates are permitted. |
| `accumulates` | Conservation domain accumulated by the variable. |
| `flux_in`, `flux_out` | Declared incoming and outgoing ledger terms. |
| `numerics` | Variable-specific numerical controls. |
| `description` | Scientific meaning. |
| `default_value` | Scalar compatibility default. |
| `variable_type` | Storage role: scalar (default), grid, concentration, or parameter. Only concentration is read at runtime, by the conservation filter. |

## `UpdateItem`

```text
UpdateItem(
    value: Any,
    mode: UpdateMode = UpdateMode.ACCUMULATE,
    write_mode: Optional[StateWriteMode] = None,
    unit: Optional[str] = None,
    basis: str = "per_step",
    per_timestep: bool = False,
    element: Optional[str] = None,
    source_functor: Optional[str] = None,
)
```

| Parameter | Meaning |
| --- | --- |
| `value` | Tensor or tensor-compatible value to commit. |
| `mode` | `ACCUMULATE`, `REPLACE`, or `FLUX`. |
| `write_mode` | Optional update-side write policy override. |
| `unit` | Units of `value`; the manager converts compatible units to the schema unit. |
| `basis` | Legacy integration basis, `per_step` or `per_second`. |
| `per_timestep` | `True` when `value` is already integrated over the current step. |
| `element` | Optional conservation-ledger element such as carbon or water. |
| `source_functor` | Writer identity for provenance and conflict accounting. |

New code should state `per_timestep` explicitly. A `per_second` update requires
`dt`. Unsupported basis values, unknown variables, incompatible units, or
write-policy violations raise an exception before a valid commit is accepted.

## `StateManager`

```text
StateManager(
    schema: Optional[dict[str, StateVariableSpec]] = None,
    **kwargs,
)
```

The manager converts initial values to tensors on its current device. Public
reads clone tensors by default, preventing accidental mutation of owned state.

| Method | Purpose | Return value |
| --- | --- | --- |
| `register_variable(spec)` | Add a variable contract before initialization or use. | `None` |
| `initialize(initial_state=None)` | Validate and establish the owned state. | `None` |
| `set_state(state_or_name, value=None, validate=True)` | Replace one or more stored values. | `None` |
| `get_state(variables=None, clone=True)` | Read all or selected state variables. | Tensor mapping |
| `checkpoint()` | Capture state, counters, diagnostics, and transaction metadata. | Restorable mapping |
| `restore(snapshot)` | Restore a mapping produced by `checkpoint()`. | `None` |
| `apply_increment(var_name, increment, dt=None)` | Commit one `UpdateItem`. | Updated tensor |
| `apply_updates(updates, dt=None, check_conservation=True, rollback_on_failure=False, update_sources=None, update_increments=None)` | Commit a set of updates transactionally. | Updated tensor mapping |

Use `get_state(clone=False)` only for performance-critical read-only access.
The canonical simulation update path is `apply_updates()`, not direct mutation
of tensors returned by the manager. `checkpoint()`/`restore()` are full
in-memory snapshots used by the adaptive retry and embedded in the durable
simulator checkpoint; the accepted-step rollback uses
`create_transaction_savepoint()`/`restore_transaction_savepoint()`. Neither is
a second file format or an independently durable persistence system.

```python
import torch
from relic.core.state import StateManager
from relic.core.state.types import StateVariableSpec, UpdateItem, UpdateMode

manager = StateManager(
    {
        "carbon_pool": StateVariableSpec(
            name="carbon_pool",
            units="kg C m-2",
            bounds=(0.0, 100.0),
            conserved=True,
            accumulates="carbon.total",
        )
    }
)
manager.initialize({"carbon_pool": torch.tensor(2.0)})
snapshot = manager.checkpoint()

manager.apply_updates(
    {
        "carbon_pool": UpdateItem(
            value=torch.tensor(0.25),
            mode=UpdateMode.ACCUMULATE,
            unit="kg C m-2",
            per_timestep=True,
            element="carbon",
            source_functor="documented_input",
        )
    },
    dt=3600.0,
    check_conservation=False,
)
manager.restore(snapshot)
print(float(manager.get_state()["carbon_pool"]))
```

Expected output:

```text
2.0
```

## `FunctorBase` and bridge classes

Every executable scientific process follows:

```text
scientific abstraction -> bridge class -> FunctorBase -> parameter system
```

| Scientific contract | Executable bridge | Intended use |
| --- | --- | --- |
| `PhysicalField` | `PhysicalFieldFunctor` | Spatial or temporal physical fields. |
| `BiologicalProcess` | `BiologicalProcessFunctor` | Biological transformations and rates. |
| `TransportMechanism` | `TransportMechanismFunctor` | Advective, diffusive, or managed transport. |
| `ScaleCoupling` | `ScaleCouplingFunctor` | Spatial, temporal, or cross-domain coupling. |

Concrete Functors implement:

```text
forward(
    state: dict[str, torch.Tensor],
    time: float,
    dt: float,
    **kwargs,
) -> dict[str, UpdateItem]
```

The `state` mapping is an execution snapshot. `time` and `dt` always describe
the same real interval. A group call covers `[t_s, t_s + dt_group]`; with
`h_sub = dt_group / substep_count`, sub-step `k` (counting from 0) receives
`time = t_s + k * h_sub` and `dt = h_sub`. The start `t_s` and the length
`dt_group` depend on the call:

- fast group in `lie` mode: `t_s` is the step start and `dt_group` is the
  step length `h`;
- `strang` mode with two or more fast Functors: the first half-steps
  (`dt_group = h/2`) and the middle full step of the last fast Functor
  (`dt_group = h`) start at the step start, and the second half-steps
  (`dt_group = h/2`) start at the step start plus `h/2`;
- medium or slow group: `t_s` is the group's last execution time `t_last`
  and `dt_group = t_future - t_last`, where `t_future` is the end of the
  current step. A non-positive `dt_group` raises `RuntimeError`.

For example, with `t_last = 0` and a step from 1 s to 2 s, a medium call
receives `time = 0` and `dt = 2`, the interval `[0, 2]` that it accounts for;
`t_last` then becomes 2. `dt` never starts before `time`.

The result maps names to state updates, diagnostics
(`diagnostics.*` names or `DERIVED` outputs), or top-level `ExternalIntent`
requests; the simulator commits only the state updates and delivers requests
after the step is accepted. A Functor must not mutate shared state directly or
call another Functor as a hidden dependency.

### `FunctorMetadata`

Metadata is the executable contract used for dependency ordering, scale
detection, validation, provenance, and structural compilation.

| Metadata group | Important fields |
| --- | --- |
| Identity | `name`, `version`, `description`, `category`, `execution_phase` |
| Science | `mathematical_expression`, `references`, `parameters`, `conservation_laws` |
| Data contract | `input_variables`, `output_variables`, `optional_output_variables` |
| Scale and scheduling | `spatial_dimensions`, `spatial_scale`, `temporal_scale`, `variable_kind`, `substep_count`, `sub_flow_accuracy` |
| Composition | `coupling_interfaces`, `port_power_contracts` |
| Structure | `form_contracts`, `hodge_contracts`, `iphs_terms`, `stoichiometric_contracts`, `structural_role` |
| Concurrency | asynchronous capability (`async_safe`, `stream_hint`: declarations only; the simulator runs Functors sequentially) and `conflict_resolution_policy` (enforced for output overlaps at registration) |

`optional_output_variables` must be a sequence of names (not a string), a
subset of `output_variables`, and free of duplicates. `spatial_dimensions`
must be 0, 1, 2, or 3, and `substep_count` must be at least 1; the
simulator splits each group interval into `substep_count` sub-steps.

Execution-order dependencies are not a `FunctorMetadata` field. They are
declared on the Functor instance through `config["dependencies"]` (a list
or a single name), `set_dependencies(...)`, `add_dependency(...)`, and
`remove_dependency(...)`, and exposed by `get_dependencies()`, which returns
a list or a mapping with `required`, `optional`, and `lagged` keys. A class
that adds a static declaration returns
`self._merge_declared_dependencies(static)`, so the instance names are merged
with it without being dropped.

Declare both `spatial_scale` and `variable_kind`. Empty input and output
declarations generate a warning when the `FunctorMetadata` object is
constructed, normally in the Functor's `__init__` and therefore before
registration; registration itself does not issue this warning. Diagnostic
outputs use the `diagnostics.*` namespace when they should not participate
in ordinary output overlap ownership.

## `RELICSimulator`

```text
RELICSimulator(config: Optional[dict[str, Any]] = None)
```

The simulator is a `torch.nn.Module`. Passing `None` (as in
`RELICSimulator(None)`) substitutes an empty runtime mapping and defaults
`enable_auto_scale=False` and `enable_distributed=False`. The simulator starts
in `UNINITIALIZED` and requires a `StateManager` and dependency resolver before
normal execution; `initialize()` attaches them, and `initialize_state()` creates
any that is missing.

| Method | Purpose | Return value |
| --- | --- | --- |
| `initialize(state_manager, dependency_graph)` | Attach the canonical state and dependency owners. | `None` |
| `initialize_state(initial_conditions)` | Install the initial values and ready the simulator. Without an attached `StateManager`, it first builds one from declarations only: the variable contract (which takes precedence), the Single Source of Truth (SSOT) variable catalog, `validation.numerical.state_variable_bounds`, and the runtime key `state_units` (units only). Undeclared units and conflicting declarations raise `ValueError`. It creates a `DependencyGraphResolver` when none is attached. An attached manager is kept and receives the values through `set_state(..., validate=True)`. | `None` |
| `register_functor(functor, execution_group=None)` | Register and device-migrate one Functor. | `None` |
| `step(dt=None)` | Execute one transactional time step. | Merged update mapping |
| `run(start_time=0, end_time=None, dt=None)` | Run until the selected end time. | Final state mapping |
| `run_ensemble(...)` | Execute the configured ensemble path. | `EnsembleResults` |
| `pause()` | Move a running simulator to `PAUSED`. | `None` |
| `resume()` | Resume a paused simulator. | `None` |
| `reset()` | Reset runtime counters and execution state. | `None` |
| `get_state()` | Read current managed state. | State mapping |
| `create_checkpoint(checkpoint_path, *, snapshot=None)` | Write one explicit, atomic, non-overwriting durable checkpoint of the live state, or of a frozen `snapshot` payload. | Integrity receipt mapping |
| `restore_checkpoint(checkpoint_path, *, compatibility_certificate=None, migration_certificate=None)` | Strictly validate and restore a checkpoint: exact by default, or a certified compatible or migration restart. | Integrity receipt mapping |
| `get_functor_summary()` | Inspect registered Functors and their contracts. | Summary mapping |
| `get_variable_provenance(var_name=None)` | Inspect the last-writer provenance tracked for variables. | Provenance mapping |

Execution groups are `fast`, `medium`, and `slow`. The group follows one
priority chain: the `execution_group` argument, the Functor attribute, the
characteristic time, the temporal scale, `process_groups`, and `fast` only
when nothing is declared. An invalid declaration, or a `process_groups` entry
that contradicts the attribute, the characteristic time, or the temporal
scale, raises `ValueError` before anything is registered; the argument
overrides the entry. The simulator derives dependency order
separately for each group. Declared cross-group reads are checked each time
a provider view is published (registration, retirement, or restore), not on
every step, and in both directions. A slower consumer that reads a faster
group's declared output gets the warning "Cross-group sampling lag". A
faster consumer that reads a slower group's output, for example a fast
Functor reading a medium or slow output, gets "Cross-group hold lag": it
holds the value committed when the producer last ran. Listing the producer
in the consumer's `lagged` dependencies acknowledges the read, which is then
logged only at debug level. With `strict_cross_group_dependencies=True` (a
bool, default `False`) any unacknowledged read raises `RuntimeError` and the
publication is rolled back. With `resolve_cross_group_deps=True`, slow
consumers with an unacknowledged sampling lag are first moved to the medium
group; nothing is moved into the fast group.

### Registration rules

- The object must inherit from `FunctorBase`; otherwise `TypeError` is raised.
- Functor names must be unique; duplicates raise `ValueError`.
- A concrete scientific Functor should inherit through the matching bridge.
- Critical state-output conflicts and mutually exclusive writers raise
  `ValueError`.
- Ordinary output overlaps warn by default and can be escalated with
  `strict_dependencies=True`.
- Device migration occurs at registration, after the Functor's initial CPU
  placeholder has been created.

### Simulation states

| State | Meaning | Normal transition |
| --- | --- | --- |
| `UNINITIALIZED` | State and dependency owners are not attached. | `initialize()` to `INITIALIZED` |
| `INITIALIZED` | Ready for a step or run. | `step()` or `run()` to `RUNNING` |
| `RUNNING` | Time integration is active. | `pause()`, completion, or failure |
| `PAUSED` | Execution is suspended. | `resume()` or `run()` |
| `COMPLETED` | Requested run interval finished. | A supported new `run()` or `reset()` |
| `FAILED` | An execution exception terminated the run. | Inspect the exception, then rebuild or reset as appropriate |

`step()` promotes `INITIALIZED` to `RUNNING`. A failed step restores the
StateManager savepoint, simulator step bookkeeping, Functor `state_dict()` and
extra state, random number generator (RNG) states, and the provenance transaction before re-raising; it
does not undo external files or services. A failed `run()` sets
the simulator state to `FAILED` and re-raises the exception. Stateless
execution is not a supported normal path.

### Checkpoint ownership and boundaries

`relic\core\simulator\checkpoint.py` is the only durable checkpoint owner.
Manual creation without a `snapshot` requires an explicit non-existing path; there is no
implicit destination and no public cleanup API. Automatic checkpointing
defaults to off and requires `enable_checkpointing=True`, an
explicit `checkpoint_directory`, and a positive `checkpoint_frequency`.
Automatic files are committed atomically after accepted scientific and
provenance step state has committed.

The strict `relic.simulator.checkpoint` format writes `format_version=3` and
is restored with `weights_only=True`. Each Functor entry has `state_dict`
(including safe PyTorch extra state) and `parameter_bindings` for live values
held by reference from its manager. Version 2 is upgraded in memory with a
`format_upgrade` restore receipt. Binding restoration checks key sets, shapes,
and dtypes and preserves live identity, gradient flags, and devices. The
payload binds StateManager state, these Functor entries,
scientific/topology/parameter contracts, resolved device and distributed
signatures, and Python, Torch CPU, legacy global NumPy, and symmetric CUDA
RNG states. The current
single-file format rejects an active `world_size > 1`. Create and restore
return receipts with path, format version, size, file hash, and payload hash;
restore compensates in-memory owners only before provider-view publication.
After publication, retirement or receipt errors preserve the committed
scientific state and new epoch while reporting the failure.

A Functor with replay-critical mutable data outside parameters and buffers
implements `get_extra_state()` and `set_extra_state()` so that PyTorch owns
the extension through `state_dict()`. It must not add another checkpoint API.
Bitwise equality from a continuous/restart test is evidence for that exact
graph, input, fixed-step, dependency, and device setup, not a cross-platform
guarantee.

## Complete minimal example

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
    def __init__(self):
        metadata = FunctorMetadata(
            name="increment",
            description="Minimal state increment example.",
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

Expected output:

```text
2.0
['x']
1
```

`time_series` records Functor outputs every `output_frequency` steps; it is not a
complete snapshot of every managed state variable. Use `get_state()` or an
explicit persistence workflow when a complete state snapshot is required.

## Related documentation

- [Architecture](../architecture.md)
- [Configuration and Parameters](configuration_and_parameters.md)
- [Core Runtime API](../api/core_runtime.md)
- [Graph Validation and Execution](graph_validation_and_execution.md)
- [Science index](../science/index.rst)
- [Theory index](../theory/index.rst)
