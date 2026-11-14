# Core Runtime API

This reference describes the public runtime architecture exported by
`relic.core` and the canonical configuration types used to construct that
runtime. It covers scientific contracts, executable Functor bridges, state
ownership, dependency ordering, simulation control, device selection, and
validated configuration loading.

## Stable imports and ownership

Prefer the root package for the 35 public classes exported by
`relic.core`:

```python
from relic.core import (
    PhysicalField,
    BiologicalProcess,
    TransportMechanism,
    ScaleCoupling,
    SpatialScalingBase,
    TemporalScalingBase,
    HierarchicalScalingBase,
    FunctorBase,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    FunctorState,
    SpatialScale,
    VariableKind,
    SubFlowAccuracy,
    FunctorException,
    ConfigurationError,
    DependencyError,
    ExecutionError,
    ConservationError,
    BiologicalProcessFunctor,
    TransportMechanismFunctor,
    PhysicalFieldFunctor,
    ScaleCouplingFunctor,
    RELICSimulator,
    SimulationState,
    StateManager,
    StateVariableSpec,
    StateVariableType,
    DependencyGraph,
    DependencyType,
    DependencyEdge,
    ExecutionNode,
    CyclicDependencyError,
    RELICConfig,
)
```

The following ownership rules are part of the API contract:

| Concern | Canonical owner | Boundary |
| --- | --- | --- |
| Scientific interface | `relic.core.abstractions` | The four abstract contracts define scientific meaning but are not executable runtime modules. |
| Executable scientific module | `relic.core.functor.FunctorBase` plus one approved bridge | A concrete process follows `abstract contract -> bridge -> FunctorBase -> ParameterManager`. |
| Scientific parameters | `FunctorBase.get_parameter_value()` and the YAML parameter database | A scientific Functor must not create a second parameter accessor or hardcode scientific defaults. |
| Prognostic state | `relic.core.state.StateManager` | State changes are committed through `UpdateItem` and `apply_updates()`. |
| Runtime orchestration | `relic.core.simulator.RELICSimulator` | Registration, execution groups, stepping, checkpoint coordination, and device migration belong here. |
| Dependency order | `relic.core.DependencyGraph` | This is a public alias of `DependencyGraphResolver`, not a second graph implementation. |
| Device resolution | `relic.core.device_policy` | CUDA detection and fallback must not be duplicated in scientific modules. |
| Typed configuration | `relic.core.config.RELICConfig` and `ConfigLoader` | Configuration models validate structure; they do not replace the scientific parameter database. |

`relic.core.category_theory` is a design blueprint and is not a runtime export.

## Scientific abstract contracts

The four contracts below are abstract base classes. Concrete scientific
modules normally inherit from the corresponding bridge class and do not
instantiate a contract directly.

### `PhysicalField`

Implementation path: `relic.core.abstractions.PhysicalField`

```python
PhysicalField(device: Optional[torch.device] = None)
```

| Parameter | Meaning |
| --- | --- |
| `device` | Initial tensor device. Executable Functors are migrated by the simulator at registration. |

A concrete field declares the class attribute `SPATIAL_DIMS`: `1`, `2` or `3`
for a field with a coordinate query, or `None` for a point model whose state
is read from StateManager. A missing declaration or a `bool`/non-int value
raises `TypeError`; an int outside 1-3 raises `ValueError`. The abstraction
keeps no field identity, units, initial value or time flag: the field's
identity is its declared metadata output and its units are that variable's
ContractRegistry units.

Subclass contract:

| Method | Purpose |
| --- | --- |
| `get_value_at(coordinates)` | Evaluate the field at coordinates of shape `[n, SPATIAL_DIMS]`; a point model raises `NotImplementedError`. |
| `compute_gradient(coordinates)` | Return the spatial gradient. |
| `interpolate(source_coordinates, target_coordinates)` | Map field values between coordinate sets. |
| `update_field(time_step, source_terms, boundary_conditions=None)` | Advance the field under sources and boundary conditions. |
| `validate_coordinates(coordinates)` | Check coordinate rank against `SPATIAL_DIMS`; a point model raises `RuntimeError`. |

Unimplemented abstract methods prevent direct construction of an incomplete
subclass.

### `BiologicalProcess`

Implementation path: `relic.core.abstractions.BiologicalProcess`

```python
BiologicalProcess(
    process_name: str,
    allometric_scaling: Optional[float] = None,
    device: Optional[torch.device] = None,
)
```

| Parameter | Meaning |
| --- | --- |
| `process_name` | Non-empty process identifier. |
| `allometric_scaling` | Optional allometric exponent or scaling coefficient owned by the concrete process. |
| `device` | Initial tensor device before simulator registration. |

`forward(state, time, dt, **kwargs)` is the sole public scientific computation
entry point. `parameter_sensitivity(parameter_name, state, *, time, dt,
target_output=None, ...)` provides finite-difference sensitivity analysis. The
parameter is a tensor member of the Functor or of a submodule (`nn.Parameter`,
buffer or tensor attribute, named by its qualified name such as `"vcmax25"` or
`"kinetics.k"`); each evaluation replaces it through `torch.func.functional_call`
and runs the governed call `self(state, time=time, dt=dt)`, so the Functor is
never mutated. Only real members are replaced (`named_parameters()`,
`named_buffers()`, or a tensor in the module's `__dict__`). A non-tensor
attribute, or a name of a class that provides attributes through its own
`__getattr__` (such a value cannot be replaced by `functional_call`), raises
`TypeError`; an unknown name or submodule path raises `KeyError`. The default step follows the perturbed tensor's dtype (Press et
al. 1992, section 5.7): `h = eps_m**(1/2) * x_c` for the forward and
`h = eps_m**(1/3) * x_c` for the central difference, with
`eps_m = torch.finfo(theta.dtype).eps`, $x_c = \lvert\theta\rvert$ where `theta != 0` and
`optimization.sensitivity.finite_difference.zero_parameter_scale` where
`theta == 0`, and `h` rounded so that `theta + h` is exactly representable. An explicit
`absolute_eps` or `relative_eps` (each optional) takes precedence: the step
is the largest given bound of `absolute_eps` and `relative_eps * x_c`. The scientific inputs
come from the state only, and `target_output` selects the
output of the `target` aggregation. A stochastic Functor (a
`FunctorSubstreamMixin` substream owner) is differenced with common random
numbers: every evaluation starts from the substream position saved before the
first one, and the position is restored after the last one, so the call does
not advance the stream. The abstract contract deliberately does not supply a
generic temperature law: `temperature_function(temperature)` raises
`RuntimeError` until the concrete process implements its documented response.
The contract carries no temperature or pH response labels. Outside a Functor, `get_parameter_value(param_name)` (here and on
`ScaleCoupling`) reads the object's config, then the manager namespaces; a
parameter found in neither raises `KeyError`, and there is no caller default.

### `TransportMechanism`

Implementation path: `relic.core.abstractions.TransportMechanism`

```python
TransportMechanism(
    transport_type: str,
    peclet_number: Optional[float] = None,
    device: Optional[torch.device] = None,
)
```

| Parameter | Meaning |
| --- | --- |
| `transport_type` | Descriptive label, one of `TransportMechanism.TRANSPORT_TYPES`: `diffusion`, `advection`, `advection_diffusion`, `turbulent`, `root_uptake`, `gas_exchange`, `boundary_layer_diffusion`, `dry_deposition`, `vertical_percolation`, `capillary_rise`, `solute_diffusion`, `managed_application`, or `interphase_mass_transfer`. A label implies no coefficient. |
| `peclet_number` | Optional advection-to-diffusion ratio supplied by the concrete closure. |
| `device` | Initial tensor device before simulator registration. |

The constant scalar diffusivity is not a constructor input and has no
default: `diffusivity` is `None` until a process whose law uses a constant D
calls `set_scalar_diffusivity(value)` after reading its parameters. The setter
is the only route; it accepts a real number or a one-element floating tensor
(including a live ParameterManager-owned `nn.Parameter`, stored as the same
object so a calibration update reaches the attribute and the law together),
requires a finite value `> 0` [m2 s-1], and raises `TypeError` (bool, None,
string, integer tensor) or `ValueError` (non-finite, non-positive,
multi-element) otherwise.

Key methods are `forward(state, time, dt, **kwargs)`,
`boundary_conditions(...)`, `compute_diffusivity_tensor(...)`, and
`get_transport_metadata()`. The base boundary method raises `RuntimeError`:
a concrete process owns its boundary law (normal distance, face coefficient,
supported boundary types) and rejects unknown types and missing boundary
values without assuming zero. The base diffusivity method raises
`RuntimeError`; a concrete process must declare its own correlation, units,
parameters, and validity range. `get_transport_metadata()` reports
`transport_type`, `diffusivity`, `peclet_number`, and `has_velocity_field`;
medium properties such as tortuosity and porosity are owned and exported by
the concrete process that uses them. An unknown transport type raises
`ValueError`; a helper that needs D raises `RuntimeError` while it is unset.

### `ScaleCoupling`

Implementation path: `relic.core.abstractions.ScaleCoupling`

```python
ScaleCoupling(
    source_scale: str,
    target_scale: str,
    coupling_type: str,
    conservation_properties: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
)
```

| Parameter | Meaning |
| --- | --- |
| `source_scale` | Non-empty source resolution or organizational scale. |
| `target_scale` | Non-empty target resolution or organizational scale. |
| `coupling_type` | `spatial`, `temporal`, `hierarchical`, or `stochastic`. |
| `conservation_properties` | Quantities preserved by the transformation; defaults to `["mass"]`. |
| `device` | Initial tensor device before simulator registration. |

Subclasses implement `upscale(fine_data)` and `downscale(coarse_data)`.
`coupling_type` has no default. An empty scale name or an unknown coupling
type raises `ValueError`; a missing argument raises `TypeError`.

## Specialized scaling contracts

These are secondary abstract contracts, not executable Functors by
themselves.

### `SpatialScalingBase`

Implementation path: `relic.core.abstractions.SpatialScalingBase`

```python
SpatialScalingBase(
    source_scale: str,
    target_scale: str,
    method: str,
    preserve: str,
    device: Optional[torch.device] = None,
)
```

`method` declares the regridding scheme and `preserve` the quantity preserved
by `regrid(data, source_grid, target_grid, method, preserve)`; neither has a
default, in the constructor or in `regrid`. A `preserve` value of `none`
creates no conservation declaration.

### `TemporalScalingBase`

Implementation path: `relic.core.abstractions.TemporalScalingBase`

```python
TemporalScalingBase(
    source_scale: str,
    target_scale: str,
    method: str,
    pattern_type: str,
    device: Optional[torch.device] = None,
)
```

`method` declares the temporal semantics selected by a concrete runtime.
Neither `method` nor `pattern_type` has a default, and
`resample(data, source_dt, target_dt, method, pattern=None)` requires its
`method` as well.
`source_dt` and `target_dt` remain resolution descriptors, not substitutes for
explicit calendar bounds. `pattern_type` names an optional reconstruction
profile; it is executable only when a concrete implementation binds that
profile to explicit bounds, units, diagnostics, and an independent
conservation oracle. A `sum` label alone does not prove mass preservation.

### `HierarchicalScalingBase`

Implementation path: `relic.core.abstractions.HierarchicalScalingBase`

```python
HierarchicalScalingBase(
    source_scale: str,
    target_scale: str,
    hierarchy_levels: List[str],
    weight_type: str,
    rule: str,
    device: Optional[torch.device] = None,
    name: Optional[str] = None,
    metadata: Optional[FunctorMetadata] = None,
)
```

| Parameter | Meaning |
| --- | --- |
| `hierarchy_levels` | Ordered organizational levels participating in aggregation and disaggregation. |
| `weight_type` | Weight source such as `area`, `biomass`, `uniform`, or a concrete custom scheme. |
| `rule` | Hierarchical aggregation rule. `weight_type` and `rule` have no default. |
| `name` | Optional identity; otherwise derived from source and target scales. |
| `metadata` | Optional Functor metadata when combined with an executable bridge. |

The constructor fixes `coupling_type="hierarchical"` and
`conservation_properties=["mass", "energy"]`. The contract declares no
multi-level method of its own: a concrete class implements the level transfer
through the `ScaleCoupling` methods `upscale(fine_data)` and
`downscale(coarse_data)`, with an explicit spatial basis for each variable.

## Functor runtime

### `FunctorBase`

Implementation path: `relic.core.functor.functor_base.FunctorBase`

```python
FunctorBase(
    name: str,
    config: Dict[str, Any],
    metadata: Optional[FunctorMetadata] = None,
    **kwargs: Any,
)
```

Recognized constructor inputs:

| Parameter | Meaning |
| --- | --- |
| `name` | Unique runtime identity containing only letters, digits, `_`, and `-`. |
| `config` | Runtime settings and dependency declarations. Scientific values remain owned by the YAML parameter database. |
| `metadata` | Scientific and orchestration declaration. A minimal object is created when omitted. |
| `parameter_manager` | Optional injected `UnifiedParameterManager`. It must expose `database`, its `ParameterDatabase`, because dotted-name resolution in `get_parameter_value()` asks it which namespaces the parameter files declare. |
| `device` | Pre-registration device. CPU is the safe placeholder when omitted. |

`FunctorBase` inherits from `torch.nn.Module` and remains abstract until
`get_dependencies()` and `forward(...)` are implemented.

Primary methods:

| Method | Purpose |
| --- | --- |
| `get_dependencies()` | Return the declaration the dependency resolver reads: a list of required Functor names, or a mapping with `required`, `optional`, and `lagged` lists. The bridge default returns only the instance names; a class with a static declaration returns `self._merge_declared_dependencies(static)`. An instance name joins `required` unless the static declaration already lists it, in which case its static category is kept. Optional and lagged dependencies come only from this declaration; there is no separate attribute for them. |
| `forward(state, time, dt, **kwargs)` | Compute updates without committing global state. |
| `wrap_update(value, mode=None, write_mode=None, per_timestep=False)` | Create an explicit `UpdateItem`. |
| `get_parameter_manager()` | Return the injected or global canonical manager. |
| `manager_binding_key(value)` | Return the key whose live binding is this object in the Functor's own manager, or `None`. |
| `held_manager_parameters()` | Return manager keys and live bindings held by reference; these are absent from module parameter and state dictionaries. |
| `get_parameter_value(...)` | Resolve and validate one parameter through the canonical Functor path. |
| `set_dependencies(dependencies)`, `add_dependency(functor_name)`, `remove_dependency(functor_name)` | Maintain the instance names that `get_dependencies()` merges; `config["dependencies"]` (a list or one string) seeds them at construction. `set_dependencies` requires a list of non-empty strings and raises `TypeError` or `ValueError` otherwise. |
| `get_required_inputs()` | Return the declared inputs every call must provide: `metadata.input_variables` minus `optional_input_variables`, in declaration order. It is the single required-input rule; `validate_state`, the validation pipeline, and `NodeRegistry.register_functor_instance()` read requiredness only from it. A Functor declares an optional input by listing it in `optional_input_variables` (in `__init__`, or where a configured value such as a bound velocity field replaces the state input) and does not override this method. |
| `validate_state(state)` | Return validation errors for the inputs of `get_required_inputs()` that are missing or not numeric, and for constraints. |
| `check_conservation_laws(input_state, output_updates)` | Perform metadata-first local conservation accounting. |
| `get_execution_stats()`, `reset_stats()` | Read or reset execution timing and error counters. |
| `to_device(device)` | Move the module and update its runtime device. |
| `state_dict()`, `load_state_dict(state_dict, strict=True)` | Standard PyTorch module state, including `get_extra_state()` / `set_extra_state()` where defined. Durable files belong to `RELICSimulator.create_checkpoint(...)` / `restore_checkpoint(...)`. |

A learnable binding from the Functor's own ParameterManager is held by
reference. Module `parameters()`, `state_dict()`, `.to()`, `.double()`, and
`to_device()` do not copy, save, or convert it. All holders and the optimizer
share the same binding and accumulate gradients there. Registration relocates
held bindings in place per key; two registered holders assigned different
devices raise `ValueError`, and failed registration rolls relocation back.
`parameter_sensitivity` refuses a held binding with `TypeError`; use the
manager calibration path and `get_trainable_parameters()` instead.

Canonical parameter lookup:

```python
get_parameter_value(
    name: str,
    *,
    units: Optional[str] = None,
    bounds: Optional[Tuple[float, float]] = None,
    validate: bool = True,
    namespace: Optional[str] = None,
    source_order: Tuple[str, ...] = ("manager", "config"),
    expected_type: Optional[type] = None,
    required: bool = False,
) -> Any
```

| Parameter | Meaning |
| --- | --- |
| `name` | Relative or fully qualified parameter key. |
| `units` | Expected units checked against the parameter specification. |
| `bounds` | Optional caller constraint in addition to the stored specification. |
| `validate` | Enable units, bounds, and type validation. |
| `namespace` | Explicit namespace; otherwise inferred from the Functor category and module. |
| `source_order` | Resolution priority: a non-empty tuple of distinct `"manager"` (the YAML parameter database) and `"config"` (this Functor's config, which `set_parameter_value` writes) entries; any other name raises `ValueError`. There is no caller-supplied default and no `"default"` source. |
| `expected_type` | Required Python type when a scalar result is expected. |
| `required` | Raise `ValueError` when no source holds a value; otherwise an absent parameter is `None`. |

Numeric values are returned as tensors on `self.device` unless a scalar
`expected_type` requires scalar conversion.

Lifecycle:

1. Construction validates `name` and `config`, loads dependencies, registers
   validation metadata, and enters `FunctorState.INITIALIZED`.
2. The simulator registers the Functor, checks output ownership, assigns an
   execution group, resolves graph dependencies, and migrates the device.
3. `__call__` requires `time` and a positive `dt`, preprocesses and validates
   state, enters `EXECUTING`, invokes `forward`, postprocesses updates, checks
   local conservation and validation rules, then enters `COMPLETED`. It
   returns the whole output mapping. The simulator classifies it with
   `partition_outputs`: state proposals enter the group call's provisional
   state and commit, `diagnostics.*` and `DERIVED` outputs are ingested as
   diagnostics, and top-level `ExternalIntent` values are excluded from local
   conservation and delivered only after the step is accepted. A direct
   `forward(...)` call skips this lifecycle.
4. A supported runtime failure is recorded and wrapped in `ExecutionError`;
   the Functor enters `FAILED`.

Framework validation in step 3 runs the L1 rules registered from the Functor
metadata. It is strict by default: an ERROR or CRITICAL finding fails the call
with `ExecutionError`, and no message class is exempt. `config["strict_validation"]`
(a bool, default `True`; another type raises `TypeError` at construction) is
the only switch: `False` is the explicit opt-in that downgrades those findings,
and only into a caller-supplied report. `__call__(..., validation_report=list)`
(keyword-only, not passed to `forward`) receives one record per call with
findings: `functor`, `time`, `dt`, `strict_validation`, `errors` and `warnings`
(each a list of `{"level", "code", "message", "scope"}`) and `disposition`
(`"raised"`, `"downgraded"` or `"passed_with_warnings"`). A downgrade without a
report raises `ExecutionError`, so a downgraded finding is never only a log
line. The removed key `strict_validation_errors` raises `ConfigurationError`.
The Functor keeps no findings: the simulator passes its run-level report
(`RELICSimulator.get_validation_findings()`). The validation context carries
a mapping output; a bare tensor output (the `torch.nn.Module` hook contract)
names no variables and is passed as `None`.

An invalid name or non-dictionary config raises `ValueError`. Invalid
dependency configuration raises the Functor `ConfigurationError`. Durable
checkpoint validation and file errors belong to the simulator checkpoint API;
Functor state dictionaries follow the standard PyTorch loading contract.

### `FunctorMetadata`

Implementation path: `relic.core.functor.functor_types.FunctorMetadata`

`FunctorMetadata` is a dataclass, not an execution engine.

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | required | Runtime and documentation identity. |
| `version` | `"1.0.0"` | Version of the concrete Functor contract. |
| `description` | `""` | Human-readable process description. |
| `category` | `BIOLOGICAL_PROCESS` | One of the four scientific abstraction categories. |
| `execution_phase` | `MAIN_PROCESSES` | Orchestration phase. |
| `mathematical_expression` | `""` | Declared scientific or IPHS-equivalent expression. |
| `references` | empty list | Scientific references supporting the process. |
| `parameters` | empty dict | The ParameterManager keys the Functor reads, each with its meaning and `[units]`: a full key, or a full-key family written with `*` or `<pft>` whose every match is read. A dict entry may state `description`, `units` and `namespace`; the key is then the namespace joined with the entry name, and a namespace whose last segment is the entry name is itself the key. |
| `input_variables` | empty list | Required state or forcing reads. |
| `output_variables` | empty list | Declared writes used for ownership and provenance checks. |
| `optional_output_variables` | empty list | Optional outputs; each must also occur in `output_variables`. The unit rule of the validation layer reports a missing declared output unless it is optional; `PenmanMonteithET` declares `canopy_storage` optional when canopy interception is off. |
| `conservation_laws` | empty list | Locally declared conserved quantities. |
| `spatial_dimensions` | `0` | Spatial rank from `0` through `3`. |
| `temporal_scale` | `UNKNOWN` | Characteristic execution timescale, parsed by `TemporalScale.from_value`: a `TemporalScale` member, a member value (case-insensitive, surrounding whitespace ignored), or `None`; there are no aliases. `UNKNOWN` (also `None` and `"unknown"`) means not declared. Any other string, the empty string included, raises `ValueError` listing the member values, and any other type raises `TypeError`, when the metadata is built. |
| `substep_count` | `1` | Integer number of internal substeps, at least 1 (a bool, a non-integer or a value below 1 raises at construction, and a value changed afterwards raises at the call).A call over `dt` runs `n` substeps of `h = dt / n`; each substep sees the state advanced by the earlier substeps (an amount by its value, a per-second `ACCUMULATE` rate by its value times `h`). The substep outputs merge by the one merge rule (see `step()`): amounts add, and per-second rates form the time-weighted mean `sum_k q_k h / dt`, so the committed change of a constant rate `q` is `q * dt` for every `n`. |
| `sub_flow_accuracy` | `UNDECLARED` | `SubFlowAccuracy` of the Functor's own sub-flow over one call: `EXACT`, `ORDER_2`, `ORDER_1`, or `UNDECLARED`. Bounds the effective order of operator splitting. |
| `coupling_interfaces` | empty list | Named coupling interfaces or ports. |
| `form_contracts` | empty tuple | Declared typed-form contracts. |
| `hodge_contracts` | empty tuple | Declared Hodge operator contracts. |
| `iphs_terms` | empty tuple | Declared IPHS structural terms. |
| `port_power_contracts` | empty tuple | Effort-flow power and units contracts. |
| `stoichiometric_contracts` | empty tuple | Reaction network and stoichiometry contracts. |
| `structural_role` | empty tuple | Structural roles used by validation and certificates. |
| `spatial_scale` | `UNKNOWN` | Metadata-first spatial scale, parsed by `SpatialScale.from_value`: a `SpatialScale` member, a member value (case-insensitive, surrounding whitespace ignored), or `None`; there are no aliases. `UNKNOWN` (also `None` and `"unknown"`) means not declared and is the only case the scale check leaves to its heuristics. Any other string, the empty string included, raises `ValueError` listing the member values, and any other type raises `TypeError`, when the metadata is built. |
| `variable_kind` | `UNKNOWN` | Primary output role. |
| `async_safe` | `False` | Whether asynchronous scheduling is declared safe. |
| `stream_hint` | `None` | Optional execution stream hint. |
| `conflict_resolution_policy` | `INHERIT` | Output-overlap policy. |

Post-initialization normalizes scale enums and validates the name,
`spatial_dimensions`, `substep_count`, and optional outputs. Invalid
declarations raise `TypeError` or `ValueError`. Missing descriptions,
references, or both input and output declarations produce warnings.

### Executable bridge classes

The bridge facade is `relic.core.functor.functor_bridges`.

#### `BiologicalProcessFunctor`

Implementation path:
`relic.core.functor.biological_bridge.BiologicalProcessFunctor`

```python
BiologicalProcessFunctor(
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[FunctorMetadata] = None,
    device: Optional[torch.device] = None,
    **kwargs: Any,
)
```

Recognized scientific keyword arguments are `process_name` and
`allometric_scaling`. It also
accepts `parameter_manager` through `FunctorBase`. The bridge supplies no
generic temperature law: a concrete Functor that needs one implements
`temperature_function` (the abstract contract raises until it does). It
supplies unit diagnostics and the MRO-safe `get_parameter_value(...)` route to
`FunctorBase`. Drivers are state inputs;
the bridge has no state/environment merge. A concrete subclass must override
`forward(...)`.

#### `TransportMechanismFunctor`

Implementation path:
`relic.core.functor.transport_bridge.TransportMechanismFunctor`

```python
TransportMechanismFunctor(
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[FunctorMetadata] = None,
    **kwargs: Any,
)
```

A concrete class declares its label as the class attribute `TRANSPORT_TYPE`,
which the bridge reads before `TransportMechanism.__init__`; an undeclared
label (`None`, the bridge value) raises `TypeError`, and the keywords
`transport_type` and `diffusivity` are rejected with `TypeError`. Recognized
keyword arguments are `peclet_number`, `device`, `velocity_field`,
`grid_spacing`, `spatial_dims`, `advection_scheme`, `diffusion_scheme`,
`non_negative`, `splitting_method`, `time_integration`, and
`parameter_manager`.

Key utilities are:

| Method | Purpose |
| --- | --- |
| `merge_source_sink(source, sink, **kwargs)` | Build a canonical transport input mapping. |
| `flux_to_updates(fluxes, dt=None, flux_key=None)` | Pass through an update dict or convert one flux to an increment `flux * dt`. |
| `validate_transport_units(...)` | Validate tensor outputs or area-per-time unit strings. |
| `finalize_transport_updates(outputs, dt, state=None, mapping=None, drop_diagnostics=True)` | Convert the outputs declared in `mapping` (output key to target state variable) to increments `F * dt`, with `F` converted from its declared unit to the per-second rate of the target contract unit. An output that is neither mapped nor in the `diagnostics.` namespace raises `ValueError`, and a non-numeric value raises `TypeError`. `diagnostics.` outputs are dropped, or passed through when `drop_diagnostics=False`. Target bounds and budgets are enforced by StateManager and the conservation layers, not here. |

Scalar `flux_to_updates` calls require both `dt` and `flux_key`; otherwise
`TypeError` is raised. A concrete subclass must override `forward(...)`.

`finalize_transport_updates` reads each flux unit from the state's `units`,
`output_units`, or `flux_units` mapping and converts the flux to the
per-second rate of the unit in the target's variable contract
(`<udunits> s-1`). An undeclared flux unit (including `state=None`), a
target without a variable contract, a declared target unit that contradicts
the contract, or a unit pair the UnitConverter cannot convert raises
`ValueError`. A content-rate conversion between
`m3 m-3 s-1` and `kg m-2 s-1` reads the layer thickness named by
`layer_thickness_variable` in the contract of the volumetric side, pointwise
from the state; an undeclared or absent thickness raises `ValueError`.
Increments are committed exactly as `F * dt`, with no magnitude filter. When
`transport.numerical.track_conservation` is true, the result also carries
`diagnostics.transport.*` entries: the total flux and the number of fluxes per
target and, where the state holds the target and its sum is nonzero, the
relative change `abs(sum(delta)) / abs(sum(x))`. Errors in these diagnostics
propagate.

#### `PhysicalFieldFunctor`

Implementation path:
`relic.core.functor.physical_field_bridge.PhysicalFieldFunctor`

```python
PhysicalFieldFunctor(
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[FunctorMetadata] = None,
    **kwargs: Any,
)
```

Recognized keyword arguments are `device` and `parameter_manager`; the
removed field keywords `field_name`, `units`, `default_value`, `spatial_dims`
and `temporal_dependence` raise `TypeError`. A concrete class declares
`SPATIAL_DIMS` (see `PhysicalField`) and implements `forward` and
`get_value_at`: the bridge supplies neither, so a class without them cannot be
constructed. `update_field(...)` delegates through the canonical
`FunctorBase.__call__` path. `update_field_to_flux(field_update, dt,
field_name)` returns `{field_name: field_update / dt}` for a declared output.

The default gradient represents a spatially constant field and returns zeros.
`interpolate` and `compute_laplacian` require concrete implementations.

#### `ScaleCouplingFunctor`

Implementation path:
`relic.core.functor.scale_coupling_bridge.ScaleCouplingFunctor`

```python
ScaleCouplingFunctor(
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[FunctorMetadata] = None,
    **kwargs: Any,
)
```

Recognized keyword arguments are `source_scale`, `target_scale`,
`coupling_type`, `conservation_properties`, `device`, and
`parameter_manager`. `source_scale`, `target_scale`, and `coupling_type` are
required: a missing one raises `TypeError`, and the bridge supplies no
default scale or coupling type. Dots in `name` are normalized to `_` characters.
`get_namespaced_parameter(...)` provides typed access through the canonical
Functor parameter route. `upscale(...)` and `downscale(...)` require an
explicit matching payload from the concrete `forward(...)`; a missing payload
raises `RuntimeError`. `check_conservation_bridge(...)` is diagnostic and
does not commit state.

The bridge classes are integration bases. They are not complete scientific
processes and must not be registered as placeholder calculations.

### Functor enums

| Class and implementation path | Values |
| --- | --- |
| `ProcessCategory` in `relic.core.functor.functor_types` | `PHYSICAL_FIELD`, `BIOLOGICAL_PROCESS`, `TRANSPORT_MECHANISM`, `SCALE_COUPLING` |
| `ExecutionPhase` in `relic.core.functor.functor_types` | `INITIALIZATION`, `PREPROCESSING`, `MAIN_PROCESSES`, `COUPLING`, `POSTPROCESSING`, `VALIDATION` |
| `FunctorState` in `relic.core.functor.functor_types` | `UNINITIALIZED`, `INITIALIZED`, `EXECUTING`, `COMPLETED`, `FAILED`, `DISABLED` |
| `SpatialScale` in `relic.core.functor.functor_types` | `POINT`, `PATCH`, `GRID_CELL`, `HILLSLOPE`, `BASIN`, `LANDSCAPE`, `GRID_2D`, `GRID_3D`, `REGIONAL`, `GLOBAL`, `UNKNOWN` |
| `VariableKind` in `relic.core.functor.functor_types` | `STATE`, `FLUX`, `DIAGNOSTIC`, `PARAMETER`, `FORCING`, `UNKNOWN` |
| `SubFlowAccuracy` in `relic.core.functor.functor_types` | `EXACT`, `ORDER_2`, `ORDER_1`, `UNDECLARED` |
| `TemporalScale` in `relic.core.functor.functor_types` | `SECOND`, `MINUTE`, `HOUR`, `SUB_DAILY`, `DAY`, `WEEK`, `MONTH`, `SEASON`, `YEAR`, `STATIC`, `STEADY_STATE`, `UNKNOWN` |
| `ConflictResolutionPolicy` in `relic.core.functor.functor_types` | `INHERIT`, `WARN`, `ERROR`, `EXCLUSIVE` |

`TemporalScale` and `ConflictResolutionPolicy` are canonical subpackage
types but are not root `relic.core` exports.

### Functor exceptions

All five classes are implemented in
`relic.core.functor.functor_exceptions`:

| Class | Meaning |
| --- | --- |
| `FunctorException` | Base exception for the Functor layer. |
| `ConfigurationError` | Invalid Functor configuration. |
| `DependencyError` | Functor dependency declaration or resolution error. |
| `ExecutionError` | Standard wrapper emitted by `FunctorBase.__call__`. |
| `ConservationError` | Severe concrete local-conservation failure. |

## State runtime

### `StateManager`

Implementation path: `relic.core.state.manager.StateManager`

```python
StateManager(
    schema: Optional[Dict[str, StateVariableSpec]] = None,
    **kwargs: Any,
)
```

Recognized constructor inputs:

| Parameter | Meaning |
| --- | --- |
| `schema` | Variable schema keyed by canonical state name. When omitted, the default schema holds one specification per loaded variable contract, or only `time` when no contracts are loaded. |
| `config` | Compatibility source for `schema` when `schema` is omitted. |
| `device` | Runtime state device resolved through `device_policy`; invalid or unavailable CUDA requests fall back to policy resolution. |
| `conservation_checks` | `None` (defaults), a bool (the `enabled` switch), or a mapping with only the keys `CONSERVATION_CHECK_KEYS` of `relic.core.state.manager`: `enabled`, `raise_on_violation`, `rollback_on_failure` (each a bool) and `tolerance` (a finite real within the bounds declared for `validation.conservation.conservation_tolerance`, currently `[1e-10, 1e-3]`). `validate_conservation_checks` checks it at construction; another type, an unknown key, or an invalid value raises `ValueError` naming the key and the value. |
| `physical_constraint_manager` | Optional injected canonical constraint manager. |
| `autoload_contract_registry` | Whether the packaged variable contracts are loaded. The default comes from `RELIC_AUTOLOAD_CONTRACT_REGISTRY` (default `1`; `0`, `false`, `no`, or `off` disable it). They are read through `relic.contracts.load_contract_registry()`, strictly and once per process: a missing directory raises `FileNotFoundError`, and a malformed or duplicate contract raises `ValueError` naming the file. Opting out is logged at INFO and turns off the ownership, alias, writer-permission, and increment checks in `apply_updates()`. |

Primary methods:

| Method | Purpose |
| --- | --- |
| `initialize(initial_state=None)` | Install the supplied values or, when none are supplied, populate required variables not yet set from tensor defaults, scalar defaults, or zero. A value outside its declared bounds raises `ValueError`. |
| `initialize_state(initial_state)` | Check the values against the declared bounds, then reset state, install the values, and mark an external reset. A value outside its declared bounds raises `ValueError` before anything is reset; nothing is clamped. |
| `register_variable(spec)` | Add one `StateVariableSpec`; non-spec inputs raise `TypeError`. |
| `set_state(state, value=None, *, validate=True)` | Clone and move one variable or a mapping to the manager device; with `validate`, a value outside its declared bounds raises `ValueError`. |
| `update_state(updates, *, validate=False)` | Convenience replacement merge; simulation advancement should use `apply_updates`. |
| `get_state(variables=None, *, clone=True)` | Return all state, selected variables, or a single value. Cloning is the safe default. |
| `apply_increment(var_name, increment, dt=None)` | Convert units and rate basis, then commit one `UpdateItem`. |
| `apply_updates(updates, dt=None, check_conservation=True, rollback_on_failure=False, update_sources=None, update_increments=None)` | Validate and atomically commit a batch of updates. |
| `checkpoint()` and `restore(snapshot)` | Snapshot and restore state, counters, diagnostics, conservation history, and pending provenance. |
| `generate_conservation_report(...)` | Produce and optionally store a `ConservationSummary`. |
| `check_strict_conservation(updates, dt=None, store_history=True)` | Run the strict conservation route when enabled. |
| `get_diagnostics()` | Return clipping and update diagnostics. |
| `ingest_module_diagnostics(module_name, diagnostics)` | Store a module's diagnostics for the step as real scalars: a real number, or a real tensor stored as its mean. Any other value (a string, `None`, a sequence, a mapping, a complex tensor) raises `TypeError` naming the key, and none of the call's diagnostics is stored. |
| `conservation_tolerance` (property) | The commit-time ledger tolerance. Its default has one source, the parameter `validation.conservation.conservation_tolerance` (`CONSERVATION_TOLERANCE_PARAMETER`); a missing parameter raises, with no fallback. An assignment is validated like `conservation_checks["tolerance"]` (a finite real within the declared bounds, currently `[1e-10, 1e-3]`) and an invalid value raises `ValueError` without changing the tolerance. |
| `configure(config=None, **settings)` | Apply post-construction settings. Only `STATE_MANAGER_CONFIGURABLE_KEYS` are accepted, currently `conservation_tolerance` (validated as above); an undeclared key, a key given both in `config` and as a keyword with different values, or an invalid value raises `ValueError` before anything changes. |

Transactional semantics:

- `ACCUMULATE` adds an increment; shapes must match an existing tensor.
- `REPLACE` installs the supplied value.
- `FLUX` is converted to an accumulated increment. A rate-like value is
  multiplied by `dt` when `per_timestep` is false and `dt` is supplied.
- `DERIVED` writes diagnostics without mutating prognostic state.
- `IMMUTABLE` writes are rejected in strict RunSpecs and otherwise skipped
  with a warning.
- Alias writes, unauthorized owners, and unauthorized increment writers raise
  `PermissionError` when a ContractRegistry is active.
- Bounds are constrained and clipping losses are recorded.
- Element mismatch, shape mismatch, strict conservation failure, ordinary
  conservation failure configured to raise, or final state validation failure
  restores the pre-update snapshot and raises an exception.

### State data classes

#### `StateVariableSpec`

Implementation path: `relic.core.state.types.StateVariableSpec`

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | required | Canonical variable name. |
| `units` | required | Canonical stored units. |
| `bounds` | `None` | Optional lower and upper physical bounds. |
| `conserved` | `False` | Whether the variable participates in conservation accounting. |
| `spatial_dims` | `0` | Spatial rank. |
| `required` | `True` | Whether initialization must create the variable. |
| `default` | `None` | Optional tensor default. |
| `write_mode` | `MUTABLE` | Default state write semantics. |
| `cf_standard_name` | `None` | Optional CF standard name. |
| `kind` | `"state"` | Semantic role such as state or flux. |
| `owner` | `None` | Canonical replacement writer. |
| `writers` | empty list | Additional declared writers. |
| `aliases` | empty list | Read aliases resolved to the canonical name. |
| `incrementable` | `True` | Whether accumulated updates are permitted. |
| `accumulates` | `None` | Optional target accumulated by this variable. |
| `flux_in` | empty list | Declared incoming flux names. |
| `flux_out` | empty list | Declared outgoing flux names. |
| `numerics` | `None` | Variable-specific numerical settings. |
| `description` | `""` | Human-readable meaning. |
| `default_value` | `None` | Optional scalar compatibility default. |
| `variable_type` | `SCALAR` | Storage role only (`StateVariableType`); the quantity basis is `SpatialQuantityBasis` and prognostic/diagnostic is `kind`. Only `CONCENTRATION` changes behaviour: the conservation check skips updates to a non-conserved `CONCENTRATION` variable. |

#### `UpdateItem`

Implementation path: `relic.core.state.types.UpdateItem`

| Field | Default | Meaning |
| --- | --- | --- |
| `value` | required | Tensor or numeric update payload. |
| `mode` | `ACCUMULATE` | Accumulate, replace, or flux application mode. |
| `write_mode` | `None` | Per-update override of mutable, immutable, or derived semantics. |
| `unit` | `None` | Units of the update payload. |
| `basis` | `"per_step"` | Compatibility basis; prefer `per_timestep`. |
| `per_timestep` | `False` | `True` when the payload is already integrated over the current step. |
| `element` | `None` | Optional conserved-element label. |
| `source_functor` | `None` | Writer identity used for authorization and provenance. |

`UpdateItem` is canonical in `relic.core.state` but is not exported from the
root package.

#### Conservation result carriers

`ConservationSummary` and `ElementConservationReport` are dataclasses in
`relic.core.state.types`.

| Class | Fields |
| --- | --- |
| `ConservationSummary` | `total_violations`, `warnings`, `custom_checks`, `timestamp`, `carbon_report`, `nitrogen_report`, `phosphorus_report`, `balance_error`, `dt`, `dynamic_tolerance_used`, `tolerance_value` |
| `ElementConservationReport` | required `balance_error` and `relative_error` |

### State enums

| Class | Values |
| --- | --- |
| `StateVariableType` | `SCALAR`, `GRID`, `CONCENTRATION`, `PARAMETER` (storage role only; the quantity basis is `SpatialQuantityBasis`) |
| `UpdateMode` | `ACCUMULATE`, `REPLACE`, `FLUX` |
| `StateWriteMode` | `MUTABLE`, `IMMUTABLE`, `DERIVED` |

Only `StateVariableType` is a root `relic.core` export. The other two are
imported from `relic.core.state`.

## Dependency graph

### `DependencyGraph`

Public import: `from relic.core import DependencyGraph`

Implementation class:
`relic.core.dependency_graph.DependencyGraphResolver`

```python
DependencyGraph(use_networkx: Optional[bool] = None)
```

`use_networkx` keeps a NetworkX mirror of the graph for introspection; it is
never sorted. `topological_sort` has one path, Kahn's algorithm over the
declared adjacency, and breaks ties by Functor registration order. A Functor
joins the graph under a non-empty string name (or an explicit `functor_id`);
otherwise `add_functor` raises `ValueError`.

| Method | Purpose |
| --- | --- |
| `add_functor(functor, functor_id=None)` | Create an execution node and read its dependency declaration; an existing identifier raises `ValueError`. |
| `add_edge(source, target, dependency_type=DATA, weight=1.0, required=True)` | Declare that `source` depends on `target`. |
| `topological_sort(handle_optional=True)` | Build edges and return dependency-before-dependent order. |
| `get_parallel_groups()` | Group nodes by dependency level for reporting. The simulator runs Functors sequentially; nodes on one level are not executed concurrently. |
| `validate_dependencies()` | Return missing dependency descriptions without executing. |
| `add_functor_dynamic(...)`, `remove_functor(...)` | Mutate a graph and optionally rebuild order. |
| `get_execution_stats()` | Return node and execution-order statistics. |
| `visualize(output_file=None)` | Return or persist a graph representation. |

Missing required dependencies raise `MissingDependencyError` from
`relic.core.dependency_graph`. Cycles raise
`CyclicDependencyError`. Removing an unknown node raises `KeyError`.

`DependencyGraph` only orders and inspects nodes; it does not execute
Functors. Execution goes through the simulator and StateManager transactional
path.

### Dependency carriers and enums

`DependencyEdge` is a dataclass:

| Field | Default | Meaning |
| --- | --- | --- |
| `source` | required | Functor that declares the dependency. |
| `target` | required | Functor that must execute first. |
| `dependency_type` | `DATA` | Data, temporal, resource, or optional relation. |
| `weight` | `1.0` | Stored edge metadata; not used by the topological sort. |
| `required` | `True` | Whether a missing target is fatal. |
| `metadata` | empty dict | Additional edge metadata. |

`ExecutionNode` is a dataclass:

| Field | Default | Meaning |
| --- | --- | --- |
| `functor_id` | required | Node identity. |
| `functor` | required | Attached Functor instance. |
| `dependencies` | empty set | Required and optional dependency names. |
| `lagged_dependencies` | empty set | Recorded dependencies excluded from the current-step DAG. |
| `dependents` | empty set | Reverse dependency set. |
| `in_degree` | `0` | Number of dependent relations. |
| `out_degree` | `0` | Number of declared dependencies. |
| `execution_time` | `None` | Optional timing statistic. |
| `metadata` | empty dict | Functor metadata payload. |

`DependencyType` values are `DATA`, `TEMPORAL`, `RESOURCE`, and `OPTIONAL`.
`CyclicDependencyError(cycle)` stores the detected node sequence.

## Simulator

### `RELICSimulator`

Implementation path: `relic.core.simulator.main.RELICSimulator`

```python
RELICSimulator(config: Optional[Dict[str, Any]] = None)
```

The constructor accepts a runtime mapping. Keys may be placed at the root or
under `simulator` or `runtime` unless a key is explicitly described as a
nested mapping.

#### Runtime configuration keys

| Group | Key | Default | Meaning |
| --- | --- | --- | --- |
| Time | `time_step` | `60.0` | Fast-group step size in seconds. |
| Time | `max_time` | `SECONDS_PER_DAY` (86400 s) | Default run end time in seconds. |
| Time | `output_frequency` | `1` | State-output interval in accepted steps. |
| Checkpoint | `enable_checkpointing` | `False` | Opt in to automatic checkpoint writes. |
| Checkpoint | `checkpoint_directory` | `None` | Explicit automatic-checkpoint directory; required when automatic checkpointing is enabled. |
| Checkpoint | `checkpoint_frequency` | `SECONDS_PER_HOUR` (3600 s) | Positive simulation-time interval between automatic checkpoints when enabled. |
| Time | `calendar_type` | `"noleap"` | `noleap` or `360_day` calendar for boundaries. |
| Time | `medium_interval_seconds` | `SECONDS_PER_DAY` (86400 s) | Medium-group interval. |
| Time | `slow_interval_seconds` | `None` | Optional slow-group interval. |
| Adaptive | `adaptive_dt_enabled` | `False` | Enable update-magnitude RMS tolerance control with at most one retry. |
| Adaptive | `adaptive_atol` | YAML `computation.scheduling.adaptive.atol` (`1e-6`) | Absolute RMS update tolerance; with `adaptive_dt_enabled` it must be finite and `> 0` (`ValueError`). |
| Adaptive | `adaptive_rtol` | YAML `computation.scheduling.adaptive.rtol` (`1e-3`) | Relative RMS update tolerance; with `adaptive_dt_enabled` it must be finite and `>= 0`. |
| Adaptive | `adaptive_dt_min` | YAML `computation.scheduling.adaptive.dt_min_seconds` (`1.0`) | Minimum adaptive step in seconds; with `adaptive_dt_enabled` finite and `> 0`. |
| Adaptive | `adaptive_dt_max` | `dt_max_factor * time_step` (YAML `computation.scheduling.adaptive.dt_max_factor`, `10`) | Maximum adaptive step in seconds; with `adaptive_dt_enabled` finite and `>= adaptive_dt_min` (a small `time_step` needs an explicit `adaptive_dt_min`). |
| Adaptive | `adaptive_monitor_vars` | `None` | Optional state names used by the update-magnitude monitor. |
| Integration | `splitting_mode` | `"lie"` | `lie` (sequential) or `strang` (palindromic composition over the whole fast plan order, `2n - 1` sub-executions; the second half-steps start at `t + h/2`). The sub-executions share one dependency-output map: a Functor that declares `subprocess_outputs` sees each provider's most recent earlier sub-execution, which under `strang` is the provider's first half-step over `[t, t + h/2]`. The effective order is the lower of the splitting order and the declared `sub_flow_accuracy` of the fast Functors; see `get_splitting_order_report()` and the numerics theory page. |
| Integration | `time_integration_method` | parameter-backed: `transport.numerical.time_integration_method` = `"euler"` | Transport time integration method, one of the parameter's `allowed_values` `euler`, `implicit_midpoint`, and `crank_nicolson`; passed to each Functor whose `forward` declares this keyword. |
| Integration | `implicit_midpoint_max_iter` | parameter-backed: `transport.numerical.implicit_midpoint_max_iter` = `50`, an integer from `1` through `100` | Maximum fixed-point (Picard) iterations of an implicit-midpoint or Crank-Nicolson transport step; running out raises `RuntimeError`. |
| Integration | `implicit_midpoint_tolerance` | parameter-backed: `transport.numerical.implicit_midpoint_tolerance` = `1e-10`, from `1e-15` through `1e-6` | Fixed-point convergence tolerance of those steps. |
| Integration | `resolve_cross_group_deps` | `False` | Move `slow` consumers that read outputs of faster groups without a `lagged` declaration into the `medium` group (slow to medium only). A Functor already scheduled in `slow` moves only when both groups last ran at the same time; otherwise the registration is rejected. Fast-group reads do not change. |
| Integration | `strict_cross_group_dependencies` | `False` | Reject a registration, retirement, or restore that leaves a cross-group read without a `lagged` declaration. |
| Graph | `strict_dependencies` | `False` | Escalate output ownership and dependency conflicts. |
| Graph | `process_groups` | unset (no entries) | Mapping from Functor name to `fast`, `medium`, or `slow`. An entry sets the group only of a Functor that declares no `execution_group` attribute, characteristic time, or temporal scale; an entry that disagrees with such a declaration raises `ValueError` (remove the entry, or pass the group as the registration argument, which overrides it), and an agreeing entry is accepted. A setting that is not a mapping, or an entry outside the three groups, raises `ValueError`. |
| Graph | `allow_bucket_surface_coexistence` | `False` | Explicitly permit the otherwise exclusive bucket and surface closure pair. |
| State | `state_units` | `None` | Mapping from initial-state variable name to one exact unit string. Read only when `initialize_state(...)` creates the StateManager, and only for units. |
| Conservation | `conservation_checks` | `True` | `True` enables and `False` disables StateManager conservation checks; a mapping may set only `enabled`, `raise_on_violation`, `rollback_on_failure` (bools) and `tolerance`. Another type, an unknown key, or an invalid value raises `ValueError`. Inside `step()` the checks run at every group commit (this setting enabled in StateManager and the simulator switch `enable_conservation_checks` true); StateManager's per-commit `generate_conservation_report(...)` is suspended for the step transaction, so an accepted step adds no summary to `conservation_history`. It is a setting, not a feature flag. |
| Conservation | `conservation_raise_on_violation`, `conservation_rollback_on_failure` | unset | Bool overrides of the same `conservation_checks` keys; a non-bool value (for example `"false"`, `0`, `1`) or a value contradicting `conservation_checks` raises `ValueError`. |
| Conservation | `conservation_tolerance` | unset (`validation.conservation.conservation_tolerance`) | Override of the StateManager commit-time ledger tolerance: a finite real number within the bounds declared for that parameter (currently `[1e-10, 1e-3]`); otherwise `ValueError`. The simulator checks it, like the other conservation settings, with the StateManager rules. |
| Conservation | `strict_conservation_enabled` | `False` | Enable strict element-level checks. |
| Scale | `enable_auto_scale` | `False` | Enable ScaleManager initialization and scale checks (`False` for both `None` and `{}`). |
| Scale | `strict_scale_detection` | `True` | Treat unresolved scale detection as strict. |
| Scale | `auto_scale` | empty dict | Nested scale settings. |
| Scale | `auto_scale.mode` | parameter-backed | ScaleManager default mode; its `coupling.scales.defaults.precision.<mode>.tolerance` is the facade ledger tolerance of calls without a mode. |
| Scale | `auto_scale.enforce_conservation` | parameter-backed | Require conservative regridding. |
| Scale | `auto_scale.cf_strict_mode` | parameter-backed | Enforce CF-aware scale metadata. |
| Scale | `auto_scale.source_grid` | `None` | Source grid descriptor. Required, with `auto_scale.target_grid`, whenever a detected scale mismatch triggers automatic regridding and the scale check supplies no grid; missing descriptors raise `ValueError` because geometry is never inferred from tensor shape or scale labels. |
| Scale | `auto_scale.target_grid` | `None` | Target grid descriptor, a mapping (`TypeError` otherwise). Required, with `auto_scale.source_grid`, whenever automatic regridding runs and the scale check supplies no grid; missing descriptors raise `ValueError`. Unless its geometry entries equal those of the source descriptor exactly (an identity regrid, which returns the state unchanged without any metadata lookup), it must also declare the non-empty strings `grid_mapping`, `lat_name`, and `lon_name` (`CF_REGRID_TARGET_GRID_FIELDS`); they are never inferred. Once the ScaleManager lifecycle has attached its CF synchronizer, during `initialize(...)` or `initialize_state(...)`, a descriptor without them raises `ValueError` naming the missing fields, and the check is repeated before each regrid. Each regridded variable needs a ScaleManager catalog entry (`KeyError` otherwise); its `spatial_basis` selects the method: `nearest` for `categorical`, `conservative` otherwise. |
| Scale | `auto_scale_mode` | `None` | Flat compatibility key for `auto_scale.mode`. |
| Scale | `auto_scale_enforce_conservation` | `None` | Flat compatibility key for conservation. |
| Scale | `auto_scale_cf_strict_mode` | `None` | Flat compatibility key for CF strictness. |
| Device | `device` | `"auto"` | `auto`, `cpu`, `cuda`, or `cuda:N`. |
| Device | `require_gpu` | `False` | Raise `RuntimeError` without CPU fallback when CUDA is unavailable. |
| Device | `device_overrides` | empty dict | Per-Functor device map; an entry the device policy cannot resolve raises `RuntimeError` naming the Functor, with the policy error as cause. |
| Device | `module_device_overrides` | empty dict | Per-module device map propagated to `device_policy`; an unresolvable entry raises `RuntimeError` naming the module. |
| Distributed | `enable_distributed` | `False` | Enable the optional process-group runtime. A request that cannot be met (runtime not installed, process-group validation errors, a setup failure) raises `RuntimeError` with the cause; without CUDA the ranks run on the CPU (device policy). |
| Distributed | `local_rank` | runtime rank | CUDA index for the local distributed process; an index without a visible CUDA device raises `RuntimeError` (set `device: cpu` to run the ranks on the CPU). |
| Validation | `cross_domain_validation_mode` | `"raise"` | `raise` fails the step on an ERROR finding of the cross-domain coupling validator; `report` is the explicit opt-in that only records it. Every step with findings is recorded in `get_cross_domain_reports()`. Another value raises `ValueError`. |
| Diagnostics | `enable_diagnostic_export` | `False` |Enable DiagnosticWriter output; an unavailable writer module or a writer that cannot be created raises, the export is never switched off. |
| Diagnostics | `diagnostic_output_frequency` | `10` | Diagnostic export interval in steps. |
| Diagnostics | `diagnostic_output_path` | `"build/diagnostics/diagnostics.nc"` | Diagnostic destination. |
| Diagnostics | `diagnostic_reference_date` | `"1850-01-01"` | Time-axis reference for diagnostics. |
| IRF | `irf_specs_path` | `None` | Optional IRF specification path; a configured file that cannot be read or parsed raises in `initialize(...)` whatever `irf_strict` is. |
| IRF | `irf_strict` | `False` | Validation findings of the IRF report raise errors; when `False`, findings are logged as warnings. Loading failures, validator failures, and run-spec resolution failures raise in both modes. |

The four adaptive settings above are read only when `adaptive_dt_enabled` is
true; a configuration key of the same name overrides the YAML value.
`RELICSimulator(None)` is the empty configuration `{}`: every key takes its
`_get_setting` default. The default `time_step` is 60 s (single source
`_DEFAULT_TIME_STEP_SECONDS`), and `enable_auto_scale` defaults to `False`
both for `None` and for a mapping without the key.

An unconfigured integration setting (`time_integration_method`,
`implicit_midpoint_max_iter`, or `implicit_midpoint_tolerance`) takes the
value of its `transport.numerical.*` parameter. A configured value is checked
against that parameter's dtype and `allowed_values` or bounds, and a violation
raises `ValueError` naming the setting when the simulator is constructed.

#### Simulator methods

| Method | Purpose and important parameters |
| --- | --- |
| `get_effective_device(functor_name)` | Resolve per-Functor override before the simulator default. |
| `initialize(state_manager, dependency_graph)` | Attach the canonical owners, load optional IRF specs, and enter `INITIALIZED`. With auto-scale enabled and a StateManager attached, it also runs the ScaleManager lifecycle; lifecycle failures raise and do not switch auto-scale off. |
| `initialize_state(initial_conditions)` | Create missing owners, install initial state, and enter `INITIALIZED`. When the simulator creates the StateManager, each initial-state variable takes its units and bounds only from declarations: its SCM contract, otherwise its SSOT catalog entry, `validation.numerical.state_variable_bounds`, and the `state_units` setting (units only). Conflicting declarations raise `ValueError`, and a variable left without units raises `ValueError` listing all such variables; nothing is inferred from variable names. Values outside declared bounds raise `ValueError`. |
| `register_functor(functor, execution_group=None)` | Validate a `FunctorBase`, output ownership, dependencies, group, and device before registration. The group is resolved before any registration state changes, by one priority chain: the `execution_group` argument, the Functor attribute `execution_group`, `characteristic_time_days`, `metadata.temporal_scale`, the `process_groups` entry, and `fast` when nothing is declared. The argument takes precedence over every other rule, including a conflicting `process_groups` entry. Every declared value is validated, also when a higher rule decides; an invalid value raises `ValueError` and registers nothing. |
| `unregister_functor(functor_name, *, strict=True)` | Remove a Functor and rebuild graph state; unknown names raise `KeyError` when strict. |
| `register_event_trigger(group, trigger_fn)` | Register a predicate `trigger_fn(state, time)` for the `medium` or `slow` group; other groups raise `ValueError`. It sees the state after the groups already run in the step and the step start time, and it is consulted only when neither the interval nor a pending bootstrap request starts the group. An exception or a result that is neither `bool` nor `int` aborts the step. |
| `step(dt=None)` | Execute one accepted-step transaction and return the aggregated updates of the accepted attempt (a rejected adaptive attempt contributes nothing). Successive outputs of one name merge by one rule: a plain diagnostic is the last observation, an `UpdateItem` follows its mode (`REPLACE` last value, `ACCUMULATE` sum), and a plain state payload is summed; return a diagnostic as an `ACCUMULATE` `UpdateItem` when its whole-step total is wanted. An `ACCUMULATE` item with `per_timestep=False` and a per-second basis (`"per_second"`, or `"instantaneous"` for a boundary-rate aggregate) is a rate, not an amount: an output over `h` within a merged interval `H` adds with the weight `h / H`, which gives the time-weighted mean over substeps, the sum over the Functors of one group call, and, under `strang`, the rate over the step (a Functor run as two half-steps contributes the mean of its two rates). The diagnostics StateManager stores for a Functor (`get_diagnostics()`) follow the same rule over the Functor's executions in the step, so a stored value equals the returned one; a rejected attempt leaves no stored diagnostics. During execution a boundary-rate reader sees the sum of current-composition writers' latest contributions, with a provider's interval mean visible after its call. The complete `h / H` mean is published only at composition completion; stage/final restatements add no second physical transfer. The adaptive metric maps update names to StateManager targets and counts each target once. `dt=None` uses `time_step`; the adaptive proposal is used only by `run(...)`. |
| `get_splitting_order_report()` | Return a new dictionary with the splitting mode, the fast order, the declared `sub_flow_accuracy` of each fast Functor, the splitting order limit, and the effective order (`"unverified"` when any fast Functor is undeclared). It is derived from the committed provider view; nothing is stored. |
| `run(start_time=0.0, end_time=None, dt=None)` | Run until `end_time` or `max_time` and return final state. |
| `pause()`, `resume()`, `reset()` | Control the simulation state machine. |
| `create_checkpoint(checkpoint_path)` | Atomically write one explicit non-overwriting durable checkpoint and return its integrity receipt. |
| `restore_checkpoint(checkpoint_path, *, compatibility_certificate=None, migration_certificate=None)` | Restore an exact checkpoint or an explicitly certified compatible/migration restart and return its receipt. |
| `get_functor_summary()` | Return category, group, variable counts, scale, and kind for registered Functors. |
| `get_variable_provenance(var_name=None)` | Return writer and update-mode history. |
| `get_cross_domain_reports()` | Return copies of the cross-domain validation records: one per step with findings (`time`, `step`, `mode`, `couplings`, and the `errors` / `warnings` messages by coupling). |
| `get_validation_findings()` | Return copies of the framework validation records of the governed Functor calls (one per call with findings; fields as in the `FunctorBase` lifecycle). |
| `get_simulation_summary()` | Return runtime, conservation, diagnostic, and scale statistics. |

Durable checkpoint serialization, validation, rollback, and receipts are owned
only by `relic.core.simulator.checkpoint`. There is no implicit
checkpoint destination and no public cleanup API. Automatic checkpointing
remains disabled unless `enable_checkpointing=True`; that
setting requires an explicit `checkpoint_directory` and positive
`checkpoint_frequency`. An automatic write occurs only after an accepted
scientific/provenance step commit.

The active `relic.simulator.checkpoint` file format requires
`format_version=3` and is loaded with `weights_only=True`; only version 3 is
written. Each Functor entry contains `state_dict` (including PyTorch extra
state) and `parameter_bindings` (live values of manager bindings held by
reference, including nested Functors). Restore writes those values through
the owning manager without replacing binding identity, gradient flag, or
device; live and saved keys, shapes, and dtypes must match. Version 2 is
upgraded in memory, with `format_upgrade` in the restore receipt recording
`from_format_version`, `to_format_version`, and `moved_bindings`. The payload
also binds the StateManager snapshot, scientific/topology contracts, resolved
device and distributed signatures, and Python/Torch CPU/optional legacy
global NumPy/CUDA RNG states. CUDA RNG presence is symmetric with the resolved runtime. The
current single-file contract fails closed when an active distributed runtime
has `world_size > 1`. Create and restore receipts contain the resolved path,
format version, file size, file SHA-256, and payload SHA-256.

Stochastic Functors that inherit `FunctorSubstreamMixin` -- including the two
certified Level 2 (`[I]`) runtime-bound processes
`OrnsteinUhlenbeckTemperatureFunctor` and `MarkedPoissonRainfallFunctor` --
serialize their independent deterministic substream cursor via PyTorch
`get_extra_state()` and `set_extra_state()`, so checkpoint/restart and
rejected-step rollback replay identical stochastic realizations without relying
on global RNG call order.

`StateManager.checkpoint()` is the full in-memory snapshot embedded in this
format and used by the adaptive retry; the accepted-step rollback uses
`create_transaction_savepoint()`. Neither is a second durable persistence
system. A bitwise-equal
continuous/restart experiment is run-specific evidence and is not a general
cross-platform or cross-device reproducibility guarantee.

Provider lifecycle and forcing position:

| Entry | Contract |
| --- | --- |
| `provider_epoch` | Monotonic identity of the published provider view. |
| `get_committed_provider_view()` | Read the immutable `CommittedProviderView` with provider identities, groups, and execution orders. |
| `acquire_provider_view()` | Acquire a context-managed reader token; retirement retains the view until readers and device work finish. |
| `get_lifecycle_receipts()` | Read registration and retirement outcomes in commit order. |
| `bind_forcing_cursor(sequence_sha256=..., length=..., consumed=0)` | Bind a forcing sequence identity and accepted-step position. |
| `get_forcing_cursor()` | Read a copy of the current forcing cursor, or `None`. |
| `clear_forcing_cursor()` | Explicitly abandon or unbind the sequence. |

`step()` acquires one reader token before its first group call and releases it
when the step ends; every group, sub-step, and compensation of that step uses
that view. A registration, retirement, or topology transition published during
or after the step applies from the next step. Retired views are released only
after their readers leave and their device fences complete; a native plugin
library is unloaded by its host only after draining.

A forcing cursor advances only on an accepted step, rolls back with rejected
work, and survives checkpoint restore. Rebinding a different partially
consumed sequence is refused. Provider identity consists of a stable instance
UID and implementation, contract, and binding digests; epoch alone does not
establish semantic equivalence. Checkpoint restore publishes one fresh epoch
after the science/runtime restoration succeeds. Failures before publication
are compensated; failures after publication preserve the committed state.

Ensemble execution:

```python
run_ensemble(
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
) -> EnsembleResults
```

`parameter_distributions` declares distributions by parameter key and is
required: members restore one runtime transaction (state, RNG) and replay it,
so they differ only by their sampled parameters; `None` or an empty mapping
raises `ValueError`.
`execution_mode` accepts `sequential`, `vectorized`, `parallel_thread`, or
`parallel_process`, but only `sequential` runs. The simulator,
StateManager, and ParameterManager are shared mutable owners, so the other
modes raise `NotImplementedError` until each member has an independent
runtime, and an unknown mode raises `ValueError`. `num_samples` and
`outlier_detection` are required: the member count sets the Monte Carlo
precision, and outlier removal drops members whose z-score exceeds the
runner's threshold, so the caller decides both. The running-mean convergence
diagnostic is evaluated when `num_samples` reaches
`optimization.uncertainty.ensemble.convergence_min_members`. `statistics`,
`quantile_levels`, and `store_samples` control aggregation and storage; `seed`
controls reproducibility.

Registration rejects non-`FunctorBase` objects with `TypeError` and duplicate
names with `ValueError`. Output overlaps warn or raise according to
`strict_dependencies` and each Functor conflict policy. `step()` promotes
`INITIALIZED` to `RUNNING`; other non-running states are rejected. Each step
captures a StateManager transaction savepoint, the simulator step bookkeeping
(time, counters, interval trackers, bootstrap marks, the adaptive proposal,
output history length, provenance detail, and the forcing cursor), every
registered Functor's `state_dict()` including extra state and its held
manager binding values in `parameter_bindings`, and the Python,
Torch CPU, legacy global NumPy, and (for a CUDA runtime) CUDA RNG states, and
it opens a provenance transaction. A failure before acceptance restores all of
these and re-raises. Other Python attributes, external files, and external
services are not rolled back; accepted external deliveries are retried, not
undone. `run()` accepts `INITIALIZED`, `PAUSED`,
or `COMPLETED`; success enters `COMPLETED`, while a supported runtime error
enters `FAILED` and is re-raised. `pause()` and `resume()` are no-ops outside
their valid source states.

### `SimulationState`

Implementation path: `relic.core.simulator.main.SimulationState`

Values: `UNINITIALIZED`, `INITIALIZED`, `RUNNING`, `PAUSED`, `COMPLETED`, and
`FAILED`.

### Simulator aliases and factory

`SimulatorConfig`, `TimeSeriesData`, and `CheckpointData` are typing aliases,
not classes. `create_simulator(config)` is a thin factory that returns
`RELICSimulator(config)`.

## Device policy

`relic.core.device_policy` exports the functions below, the completion-fence
class `DeviceFence`, and the type alias `DeviceLike` (`str` or
`torch.device`).

| Function | Purpose |
| --- | --- |
| `is_cuda_available()` | Defensive CUDA availability query. |
| `resolve_device(explicit_device=None, *, module_name=None, config=None, default="cpu", distributed=False, require_gpu=False)` | Resolve explicit, configured, override, environment, process-default, and fallback devices in priority order. |
| `distributed_default_device(*, module_name=None, config=None, require_gpu=False)` | CUDA-first distributed resolution with CPU fallback. |
| `set_default_device(device, *, require_gpu=False)` | Set the process-wide default. |
| `set_device_overrides(overrides, *, clear_existing=False, require_gpu=False)` | Set per-module runtime overrides. |
| `get_device_overrides()` | Return a copy of current overrides. |
| `record_device_fence(devices)` | Return a `DeviceFence` recorded on the current stream of each CUDA device in `devices`. |

`resolve_device` evaluates seven precedence tiers in order:

1. `explicit_device` argument when not `None`.
2. `config` per-Functor and per-module overrides (`device_overrides` and `module_device_overrides` at the root, `simulator`, or `runtime` section).
3. Process-wide runtime overrides registered via `set_device_overrides(...)`, followed by `RELIC_DEVICE_OVERRIDES_JSON` environment overrides. Across tiers 2 and 3, the full `module_name` is checked across all three override sources before its final dotted component.
4. `config` global `device` (`root`, `simulator`, or `runtime` section).
5. Process-wide default device registered via `set_default_device(...)`.
6. `RELIC_DEFAULT_DEVICE` environment variable.
7. Distributed default (`torch.device("cuda")` when CUDA is available, falling back to `torch.device("cpu")` when `distributed=True`) or the caller's `default` argument (`"cpu"` by default).

Non-mapping override containers raise `TypeError`, malformed JSON in
`RELIC_DEVICE_OVERRIDES_JSON` raises `ValueError`, and unparseable device
strings in an override map raise `RuntimeError` naming the source and key. An
unavailable CUDA request (`"auto"` or `"cuda"`) falls back to
`torch.device("cpu")` unless `require_gpu=True`, which raises `RuntimeError`.
An out-of-range `cuda:N` index falls back to `cuda:0` when at least one CUDA
device is visible or `cpu` when none is visible, unless `require_gpu=True`
(`RuntimeError`). On `RELICSimulator`, an explicit `enable_distributed: true`
request that cannot be met raises `RuntimeError`.

```python
import torch
from relic.core import RELICSimulator, StateVariableSpec
from relic.core.device_policy import resolve_device
from relic.core.state import UpdateItem, UpdateMode

sim = RELICSimulator(None)
assert sim.enable_auto_scale is False
spec = StateVariableSpec(name="soil_carbon", units="kg m-2")
item = UpdateItem(value=torch.tensor(0.5), mode=UpdateMode.ACCUMULATE, unit="kg m-2")
dev = resolve_device(config={"device": "cpu"})
assert spec.units == "kg m-2" and item.basis == "per_step" and dev.type == "cpu"
```

`DeviceFence(devices)` records one CUDA event on the current stream of each
distinct CUDA device in `devices`; CPU devices are ignored, and a fence
without CUDA events is complete. `completed` is true once the work enqueued
on those streams before the record has finished. `wait(timeout=None)` blocks
until then and returns `False` if `timeout` seconds pass first, and
`describe()` returns the devices, the event count, and the completion flag.
Work enqueued later or on other streams is not covered. The simulator records
such fences for its provider views and releases a retired view only after
they complete.

## Typed configuration API

### `RELICConfig`

Root public import: `from relic.core import RELICConfig`

Implementation path: `relic.core.config.models.RELICConfig`

All `ConfigurationModel` subclasses validate assignment, reject extra fields,
serialize enum values, strip surrounding string whitespace, and provide
`to_dict()` and `to_json()`.

| Field | Default | Meaning |
| --- | --- | --- |
| `config_version` | `"2.0.0"` | Configuration schema version. |
| `simulation_name` | `"relic_simulation"` | File-system-safe simulation identifier. |
| `description` | `""` | Simulation description. |
| `created_at` | object-creation factory | Configuration creation metadata. |
| `physics` | `PhysicsConfig()` | Process-selection settings. |
| `simulation` | `SimulationConfig()` | Time, grid, solver, and conservation settings. |
| `stochastic` | `StochasticConfig()` | Uncertainty and stochastic-process settings. |
| `optimization` | `OptimizationConfig()` | Parameter optimization settings. |
| `surrogates` | empty dict | Surrogate configuration by Functor name. |
| `data` | `DataConfig()` | I/O, quality, and transformation settings. |
| `parameters` | `ParametersConfig()` | Parameter database and validation settings. |
| `computation` | `ComputationConfig()` | Device, precision, compilation, and parallel settings. |
| `logging` | `LoggingConfig()` | Logging and monitoring settings. |
| `feature_flags` | `FeatureFlagConfig()` | Explicit optional capability switches. |

Key methods:

| Method | Purpose |
| --- | --- |
| `to_schema()` | Generate the configuration JSON Schema. |
| `to_yaml(path=None)` | Return YAML or write it to a path. |
| `from_yaml(path)` | `ConfigLoader().load_config(path)`, the single configuration load path: environment overrides, `parameters.parameter_overrides` written to the ParameterManager, and cross-section validation. |
| `merge_with(other)` | Deep-merge a mapping or another config, with `other` taking precedence. |
| `validate_for_simulation()` | Return readiness warnings for paths, CUDA, parameters, and stochastic sampling. |

Direct `RELICConfig(...)` construction raises `pydantic.ValidationError` for
invalid types, ranges, extra fields, cross-field constraints, or simulation
names. A nonmatching major schema version warns. `from_yaml(path)` raises
`relic.core.config.loader.ConfigurationError` for every load failure; a
wrapped exception, such as a Pydantic error, is kept as its `__cause__`.
PyYAML, `toml`, Hydra, and OmegaConf are import-time requirements of
`relic.core.config`.

### Core configuration submodels

#### `PhysicsConfig`

Implementation path: `relic.core.config.models.PhysicsConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `carbon_cycle` | `True` | Enable carbon-cycle processes. |
| `water_balance` | `True` | Enable water-balance processes. |
| `nitrogen_cycle` | `False` | Enable nitrogen-cycle processes. |
| `phosphorus_cycle` | `False` | Enable phosphorus-cycle processes. |
| `bvoc_emissions` | `False` | Enable BVOC emissions. |
| `atmospheric_chemistry` | `False` | Enable atmospheric chemistry. |
| `aerosol_dynamics` | `False` | Enable aerosol dynamics. |
| `photosynthesis_model` | `"farquhar"` | `farquhar`, `simplified`, or `none`. |
| `respiration_model` | `"q10"` | `q10`, `arrhenius`, or `none`. |
| `phenology_model` | `"static"` | `static`, `dynamic`, or `satellite`. |
| `allocation_model` | `"fixed"` | `fixed`, `dynamic`, or `optimal`. |
| `soil_biogeochemistry` | `True` | Enable soil biogeochemistry. |
| `decomposition_model` | `"first_order"` | `first_order`, `mimics`, or `century`. |
| `evapotranspiration_model` | `"penman_monteith"` | `penman_monteith`, `priestley_taylor`, or `simple`. |
| `runoff_model` | `"simple"` | `topmodel`, `curve_number`, or `simple`. |
| `soil_moisture_model` | `"layers"` | `richards`, `bucket`, or `layers`. |
| `molecular_diffusion` | `True` | Enable molecular diffusion. |
| `turbulent_transport` | `False` | Enable turbulent transport. |
| `advection` | `False` | Enable advection. |

BVOC emissions require the carbon cycle. Other questionable combinations
produce model warnings without synthesizing new process implementations.

#### `SimulationConfig`

Implementation path: `relic.core.config.models.SimulationConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `start_time` | `0.0` | Start time in seconds; must be non-negative. |
| `end_time` | `SECONDS_PER_DAY` (86400 s) | End time in seconds; must exceed `start_time`. |
| `time_step` | `60.0` | Default step in seconds; greater than zero and at most `SECONDS_PER_HOUR`. |
| `spatial_resolution` | `1.0` | Positive grid resolution in meters. |
| `grid_size_x` | `100` | X cells, from `1` through `10000`. |
| `grid_size_y` | `100` | Y cells, from `1` through `10000`. |
| `grid_size_z` | `10` | Vertical layers, from `1` through `100`. |
| `integration_method` | `EULER` | Numerical integration enum. |
| `tolerance_relative` | `1e-6` | Positive relative tolerance, at most `0.1`. |
| `tolerance_absolute` | `1e-9` | Positive absolute tolerance, at most `0.01`. |
| `max_iterations` | `100` | Solver iteration cap, from `1` through `10000`. |
| `convergence_criterion` | `1e-8` | Positive iterative convergence threshold. |
| `strict_conservation` | `False` | Typed strict C/N/P conservation switch. |
| `conservation_checks` | `True` | Enable conservation validation. |
| `log_conservation_violations` | `True` | Emit conservation violations to configured logs. |

The model rejects invalid time ordering and a step longer than the
simulation.

#### `StochasticConfig`

Implementation path: `relic.core.config.models.StochasticConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `enable_monte_carlo` | `False` | Enable Monte Carlo analysis. |
| `num_realizations` | `100` | Realization count from `1` through `10000`. |
| `random_seed` | `None` | Optional non-negative seed. |
| `parameter_sampling` | `"normal"` | `uniform`, `normal`, `lognormal`, or `beta`. |
| `correlation_structure` | `"none"` | `none`, `exponential`, `gaussian`, or `matern`. |
| `correlation_length` | `1000.0` | Positive spatial correlation length in meters. |
| `enable_weather_generator` | `False` | Enable stochastic weather generation. |
| `weather_model` | `"richardson"` | `richardson`, `lars-wg`, or `markov`. |
| `enable_sde` | `False` | Enable stochastic differential equations. |
| `sde_solver` | `"euler_maruyama"` | `euler_maruyama`, `milstein`, or `runge_kutta`. |
| `noise_type` | `"white"` | `white`, `colored`, or `jump`. |
| `compute_pit` | `True` | Compute probability integral transform diagnostics. |
| `compute_crps` | `True` | Compute continuous ranked probability score. |
| `compute_coverage` | `True` | Compute interval coverage. |
| `confidence_levels` | `[0.5, 0.9, 0.95]` | Requested interval probabilities. |

#### `OptimizationConfig`

Implementation path: `relic.core.config.models.OptimizationConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `enable_optimization` | `False` | Enable optimization. |
| `optimization_method` | `"gradient"` | `gradient`, `evolutionary`, `bayesian`, or `hybrid`. |
| `learning_rate` | `0.01` | Positive gradient learning rate, at most `1`. |
| `optimizer` | `"adam"` | `sgd`, `adam`, `lbfgs`, or `rmsprop`. |
| `max_epochs` | `100` | Positive epoch limit. |
| `population_size` | `50` | Evolutionary population from `10` through `1000`. |
| `mutation_rate` | `0.1` | Mutation probability from `0` through `1`. |
| `crossover_rate` | `0.9` | Crossover probability from `0` through `1`. |
| `acquisition_function` | `"ei"` | `ei`, `pi`, or `ucb`. |
| `num_initial_points` | `10` | At least five Bayesian initial points. |
| `objective` | `"mse"` | `mse`, `mae`, `likelihood`, or `custom`. |
| `regularization` | `"none"` | `none`, `l1`, `l2`, or `elastic`. |
| `regularization_strength` | `0.01` | Non-negative regularization strength. |
| `multi_objective` | `False` | Enable multi-objective optimization. |
| `pareto_front_size` | `50` | Pareto population, at least ten. |

#### `SurrogateConfig`

Implementation path: `relic.core.config.models.SurrogateConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `class_path` | `"relic.models.surrogates.ONNXSurrogate"` | Fully qualified surrogate class. |
| `model_path` | required | Existing model artifact path. |
| `input_variables` | required | Non-empty canonical inference inputs. |
| `output_variables` | required | Non-empty canonical outputs. |
| `input_mapping` | empty dict | State-name to model-input mapping. |
| `output_mapping` | empty dict | Model-output to RELIC output mapping. |
| `units_conversion` | empty dict | Variable to source/target units pair. |
| `device` | `AUTO` | Inference device. |
| `batch_size` | `1` | Batch limit from `1` through `1024`. |
| `providers` | empty list | Optional ONNX execution providers. |
| `session_options` | empty dict | Runtime session options. |
| `conservation_check` | `True` | Enable surrogate conservation validation. |
| `conservation_tolerance` | `1e-6` | Positive maximum conservation error. |
| `dependencies` | empty list | Required Functor names. |
| `uncertainty_quantification` | `False` | Request uncertainty estimates. |
| `cache_predictions` | `True` | Cache identical inputs. |
| `cache_size` | `1000` | Maximum cached entries; `0` means unlimited. |
| `model_metadata` | empty dict | Preloaded model metadata. |
| `notes` | `""` | Additional model notes. |

Construction fails when the artifact does not exist, input or output lists are
empty, or mappings refer to undeclared variables.

#### `DataConfig`

Implementation path: `relic.core.config.models.DataConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `io` | `DataIOConfig()` | Paths, format, compression, and chunking. |
| `quality` | `DataQualityConfig()` | Quality-control policy. |
| `transforms` | `DataTransformConfig()` | Input transformations and coordinate system. |
| `output_variables` | `["gross_primary_productivity", "net_primary_productivity", "evapotranspiration_flux", "soil_moisture"]` | Default output names. |
| `output_frequency` | `"hourly"` | `timestep`, `hourly`, `daily`, or `monthly`. |
| `global_attributes` | institution and source only | Caller-owned global output attributes; no `Conventions` claim is synthesized. |

`DataConfig` contains no dataset-level CF switch. Reader and writer operations
may validate or preserve an explicitly supplied single-variable standard-name
and unit subset, but dimensions, coordinates, bounds, grid mappings, cell
methods, and global attributes are not jointly certified. Full-dataset CF
conformance remains `[N]`.

`DataIOConfig` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `input_path` | `"./data/input"` | Input directory. |
| `output_path` | `"./data/output"` | Output directory. |
| `format` | `NETCDF4` | Primary output format. |
| `compression` | `True` | Enable compression. |
| `compression_level` | `4` | Compression level from `1` through `9`. |
| `chunking` | `True` | Enable chunked output. |
| `chunk_size` | `{"time": 100, "x": 50, "y": 50}` | Chunk dimensions. |

`DataQualityConfig` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `enable_qc` | `True` | Enable quality checks. |
| `missing_value_strategy` | `"interpolate"` | `interpolate`, `forward_fill`, `mean`, or `drop`. |
| `outlier_detection` | `"zscore"` | `zscore`, `iqr`, `isolation`, or `none`. |
| `outlier_threshold` | `3.0` | Positive method-specific threshold. |
| `smooth_input` | `False` | Enable input smoothing. |
| `smoothing_window` | `5` | Window size of at least three. |

`DataTransformConfig` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `standardize` | `False` | Standardize inputs. |
| `log_transform` | empty list | Variables receiving a log transform. |
| `coordinate_system` | `"cartesian"` | `cartesian`, `geographic`, or `projected`. |
| `projection` | `None` | Optional PROJ specification. |

#### `ParametersConfig`

Implementation path: `relic.core.config.models.ParametersConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `database_path` | `"relic/parameters/database"` | Parameter database root. |
| `load_from_database` | `True` | Load the canonical database. |
| `database_precedence` | `["calibrated", "relic"]` | Ordered calibrated and baseline sources. |
| `validation_level` | `STANDARD` | Parameter validation strictness. |
| `validate_units` | `True` | Validate units. |
| `validate_bounds` | `True` | Validate declared bounds. |
| `validate_relationships` | `False` | Validate cross-parameter relationships. |
| `allow_overrides` | `True` | Allow explicit runtime overrides. |
| `parameter_overrides` | empty dict | Parameter-key to numeric override. |
| `sensitivity_analysis` | `False` | Enable sensitivity analysis. |
| `sensitivity_method` | `"sobol"` | `local`, `morris`, `sobol`, or `fast`. |
| `sensitivity_samples` | `1000` | Sample count of at least `100`. |
| `track_changes` | `True` | Record parameter changes. |
| `save_calibrated` | `True` | Persist calibrated parameters. |
| `calibration_metadata` | empty dict | Calibration provenance metadata. |

This model configures the existing ParameterManager. It does not authorize
scientific values to be embedded directly in Functor code.

#### `ComputationConfig`

Implementation path: `relic.core.config.models.ComputationConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `device` | `CPU` | Typed configuration device. |
| `precision` | `FLOAT32` | Numerical precision. |
| `enable_gpu` | `True` | Permit GPU execution when available. |
| `gpu_id` | `0` | Non-negative CUDA index. |
| `enable_amp` | `False` | Enable automatic mixed precision. |
| `gpu_memory_fraction` | `0.9` | Requested fraction from `0.1` through `1`. |
| `compile_backend` | `"inductor"` | Explicit `torch.compile` backend request for a caller that invokes `maybe_compile`. |
| `compile_mode` | `"default"` | Explicit `torch.compile` mode request for a caller that invokes `maybe_compile`. |
| `parallel_backend` | `THREADS` | Thread, process, Dask, or Ray backend. |
| `num_workers` | `None` | Positive worker count or automatic selection. |
| `chunk_size` | `"auto"` | `auto`, `small`, `medium`, `large`, or `adaptive`. |
| `distributed_backend` | `"gloo"` | `gloo`, `nccl`, or `mpi`. |
| `enable_distributed` | `False` | Enable distributed execution. |
| `world_size` | `1` | Positive process count. |
| `rank` | `0` | Non-negative process rank. |
| `enable_gradient_checkpointing` | `False` | Trade recomputation for lower memory use. |
| `enable_memory_efficient_attention` | `False` | Enable memory-efficient attention where supported. |
| `clear_cache_frequency` | `100` | Cache-clear interval; `0` disables it. |
| `enable_profiling` | `False` | Enable profiling. |
| `profile_memory` | `False` | Include memory profiling. |
| `profile_wait_steps` | `5` | Non-negative pre-profile wait. |
| `profile_warmup_steps` | `5` | Non-negative warmup length. |
| `benchmark_mode` | `False` | Enable benchmark behavior. |

These configuration fields do not auto-compile scientific Functors. The only
public callable adapter is `relic.computation.jit.maybe_compile(...)`, which
wraps `torch.compile` for one caller-supplied callable. It does not certify
arbitrary Python purity or scientific-kernel equivalence. Its numerical
validation option is `[N]` and rejects before compilation or callable
execution. Synthetic benchmark success is scoped to one current environment,
target label, input batch, backend, and device; it is not cross-backend,
cross-device, release, or long-integration evidence.

#### `LoggingConfig`

Implementation path: `relic.core.config.models.LoggingConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `level` | `INFO` | Global log level. |
| `console_output` | `True` | Enable console logging. |
| `file_output` | `True` | Enable file logging. |
| `log_dir` | `"./logs"` | Log directory. |
| `format` | standard detailed format | Message format. |
| `date_format` | standard timestamp format | Timestamp rendering format. |
| `module_levels` | RELIC module defaults | Per-module log levels. |
| `log_performance` | `False` | Log performance metrics. |
| `performance_interval` | `100` | Positive metric interval in steps. |
| `debug_mode` | `False` | Enable debug behavior. |
| `trace_calls` | `False` | Trace calls. |
| `log_tensors` | `False` | Log tensor statistics. |
| `enable_tensorboard` | `False` | Enable TensorBoard integration. |
| `tensorboard_dir` | `"./runs"` | TensorBoard directory. |
| `enable_wandb` | `False` | Enable Weights and Biases integration. |
| `wandb_project` | `"relic"` | Weights and Biases project name. |

#### `FeatureFlagConfig`

Implementation path: `relic.core.config.models.FeatureFlagConfig`

| Field | Default | Meaning |
| --- | --- | --- |
| `full_richards_solver` | `False` | Enable the full Richards solver. |
| `o3_stress_enabled` | `False` | Enable ozone stress. |
| `root_nutrient_uptake_enabled` | `False` | Enable root nutrient uptake. |
| `canopy_layers_enabled` | `False` | Enable multilayer canopy radiation. |
| `strict_conservation_enabled` | `False` | Enable strict element conservation. |
| `radiation_conservation` | `False` | Enable the radiation energy conservation validator of `RadiationFunctor` (which also requires its `enable_conservation_check` config switch). |
| `canopy_interception_diagnostics` | `False` | Enable detailed interception diagnostics. |
| `dynamic_snow_albedo` | `True` | Enable dynamic snow albedo. |
| `advanced_numerical_solvers` | `False` | Enable advanced solver paths. |
| `parameter_sensitivity_analysis` | `False` | Enable parameter sensitivity analysis. |
| `external_model_coupling` | `False` | Enable external-model coupling. |

Each field may be overridden by
`RELIC_ENABLE_<FIELD_NAME>=true/false`. The fields are the only flags. The
process-wide `relic.core.feature_flags.feature_manager` (`FeatureFlagManager`)
accepts exactly these field names in `is_enabled`, `enable`, `disable`,
`toggle`, `update_flag`, and `register_callback`; any other name, such as the
simulator setting `conservation_checks`, raises `ValueError` and never reads
as disabled. `get_all_flags()`, `get_enabled_features()`,
`get_disabled_features()`, and `get_feature_status_report()` report the
declared flags. The model itself has no name-lookup methods.

### Configuration enums

| Enum | Values |
| --- | --- |
| `IntegrationMethod` | `EULER` (`"euler"`), `IMPLICIT_MIDPOINT` (`"implicit_midpoint"`), `CRANK_NICOLSON` (`"crank_nicolson"`); the members equal the `allowed_values` of `transport.numerical.time_integration_method` |
| `DeviceType` | `CPU`, `CUDA`, `AUTO`, `MPS` |
| `PrecisionType` | `FLOAT16`, `BFLOAT16`, `FLOAT32`, `FLOAT64` |
| `ParallelBackend` | `THREADS`, `PROCESSES`, `DASK`, `RAY` |
| `ParameterValidationLevel` | `NONE`, `BASIC`, `STANDARD`, `STRICT`, `CUSTOM` |
| `OutputFormat` | `NETCDF4`, `HDF5`, `CSV`, `JSON`, `PARQUET` |
| `LogLevel` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

### `ConfigLoader`

Public import: `from relic.core.config import ConfigLoader`

Implementation path: `relic.core.config.loader.ConfigLoader`

```python
ConfigLoader(
    env_prefix: str = "RELIC",
    validate_on_load: bool = True,
    cache_configs: bool = True,
    use_hydra: bool = True,
    parameter_manager: Optional[Any] = None,
)
```

| Parameter | Meaning |
| --- | --- |
| `env_prefix` | Prefix scanned for environment overrides. |
| `validate_on_load` | Run `ConfigValidator` after Pydantic construction. |
| `cache_configs` | Cache validated configurations and invalidate file entries by modification time. |
| `use_hydra` | Compose a configuration directory through Hydra and OmegaConf; with `False`, only single files load. |
| `parameter_manager` | Inject the manager that receives `parameters.parameter_overrides`; `get_global_manager()` is used when omitted. |

`load_config(config_path=None, overrides=None, env_overrides=True)` reads a
UTF-8 YAML, JSON, or TOML file chosen by suffix, or composes a Hydra
directory. The source follows the one rule
`relic.core.config.resolve_config_source(config_path=None, *, env_var="RELIC_CONFIG_PATH")`,
which `get_global_config` and `load_runtime_config` also use: a named source
(the argument, else `RELIC_CONFIG_PATH`) that does not exist raises
`FileNotFoundError`; without one, the first existing file of
`relic.core.config.loader.DEFAULT_CONFIG_PATHS` is read; when none exists the
global default configuration (`RELICConfig` defaults) is the base and the choice
is logged. No directory is searched for other file names. The instance cache
key holds the source, the overrides and, with `env_overrides`, the set
configuration environment variables. `get_global_config(...)` caches per the
resolved source and its modification time, the overrides, both flags and those
variables, so a changed `RELIC_CONFIG_PATH`, file or override variable is read
again. `overrides` is a
list of dotted `key=value` strings or a mapping from dotted keys to values,
and it is always applied. Source precedence is base configuration, overrides,
then environment overrides. It builds `RELICConfig` and writes
`parameters.parameter_overrides` to the injected manager or
`get_global_manager()` with `set_value(..., validate=True)`; a key that is not
a registered parameter raises. It then runs `ConfigValidator` when
`validate_on_load` is true, where an error-level message fails the load, and
caches the result.

Environment overrides are matched against the `RELICConfig` field tree. Each
leaf field has exactly one variable name: `env_prefix` and the upper-case
field path joined by `_`, for example `RELIC_PHYSICS_CARBON_CYCLE` for
`physics.carbon_cycle` and `RELIC_DATA_IO_COMPRESSION_LEVEL` for
`data.io.compression_level`. Field names that contain `_` characters therefore
work. Other variables with the prefix, such as `RELIC_CONFIG_PATH`, are not
configuration fields and are ignored by the loader; `FeatureFlagConfig` reads
its own `RELIC_ENABLE_<FIELD_NAME>` variables. Values are parsed as `true` or
`false`, then `null` or `none`, then integer, then float, and are otherwise
kept as strings. `RELIC_LOG_LEVEL` is read by the command-line entry
`relic.interfaces.cli.main` when neither `--verbose` nor `--quiet` is given: it
must name a logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL), otherwise
it raises `ValueError`; unset means INFO.

Other public methods are `save_config(...)`, `save_resolved(...)`,
`export_schema(...)`, `set_parameter_manager(...)`, `clear_cache()`, and
`get_statistics()`.

Invalid files, unsupported formats, Pydantic failures, or invalid validation
results raise `relic.core.config.loader.ConfigurationError`.

### Configuration validation classes

Public imports are available from `relic.core.config`.

#### `ConfigValidator`

```python
ConfigValidator(calendar_type: Optional[str] = None)
```

The validator owns an ordered rule list. `add_rule(rule)` rejects duplicate
`rule_id` values; `remove_rule(rule_id)` reports whether a rule was removed.
`validate(config, levels=None, use_cache=True)` applies the selected
`ConfigValidationStage` values and returns a `ConfigValidationResult`.
An exception inside a rule is captured as an error message and does not
escape the validation pass.

The six default rules, in priority order, are `temperature_bounds`,
`timestep_stability`, `conservation_consistency`, `parameter_bounds`,
`integration_consistency`, and `memory_usage`. `calendar_type` (`noleap` or
`360_day`) is passed to `integration_consistency`. Without it, a simulation
longer than 360 days is reported as an error, because that check depends on
the calendar year length; with it, a simulation longer than one calendar year
gets an `INFO` suggestion to set the runtime key `adaptive_dt_enabled`. `ConfigLoader`
constructs its validator without a calendar, so with `validate_on_load=True`
such a configuration fails to load with `ConfigurationError`.

#### `ValidationMessage`

Implementation path: `relic.core.config.validator.ValidationMessage`

| Field | Default | Meaning |
| --- | --- | --- |
| `level` | required | Validation stage. |
| `severity` | required | Message severity. |
| `message` | required | Human-readable issue. |
| `location` | `""` | Configuration location. |
| `parameter_name` | `""` | Related parameter. |
| `current_value` | `None` | Observed value. |
| `suggested_value` | `None` | Recommended value. |
| `suggested_action` | `""` | Recommended action. |
| `reference_url` | `""` | Optional reference. |

#### `ConfigValidationResult`

Implementation path:
`relic.core.config.validator.ConfigValidationResult`

| Field | Default | Meaning |
| --- | --- | --- |
| `is_valid` | `True` | Recomputed as false when errors or critical messages exist. |
| `messages` | empty list | Collected `ValidationMessage` objects. |
| `validation_time` | `0.0` | Validation duration in seconds. |
| `num_errors` | `0` | Error count. |
| `num_warnings` | `0` | Warning count. |
| `num_info` | `0` | Informational count. |
| `num_critical` | `0` | Critical count. |

`ValidationResult` is a compatibility alias of
`ConfigValidationResult`. It is not a second result class.
`ValidationSeverity` values are `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
`ConfigValidationStage` values are `SYNTACTIC`, `SEMANTIC`, `SCIENTIFIC`,
and `PERFORMANCE`.

## Current live-code limitations

These boundaries describe the implemented API and prevent documentation from
promising behavior that the runtime does not provide.

1. `relic.core.DependencyGraph` is an alias of `DependencyGraphResolver`.
   Direct `add_functor()` raises `ValueError` for an existing node ID but
   performs no output-owner, bridge, or device check. Use
   `RELICSimulator.register_functor()` for canonical duplicate-name,
   output-owner, dependency, and device checks.
2. `relic.core.ConfigurationError` belongs to the Functor exception hierarchy.
   `ConfigLoader` raises the distinct
   `relic.core.config.loader.ConfigurationError`.
3. The Functor call forms are `functor(state, time, dt)` and
   `functor(state, time=..., dt=...)`. Any other positional form (for
   example an environment mapping after the state), or `time`/`dt` given both
   positionally and by keyword, raises `TypeError`: scientific inputs come
   only from the state.
4. Typed and runtime configuration names are not a direct one-to-one mapping:
   `SimulationConfig.integration_method` corresponds to
   `time_integration_method`; `SimulationConfig.end_time` corresponds to
   `run(end_time=...)` and does not map directly to `max_time`.
   `ComputationConfig.device` defaults to CPU, while
   `RELICSimulator(config=None)` uses CUDA-first automatic resolution.
   Convert these fields explicitly at the application boundary.
5. `FunctorBase` and `RELICSimulator` are `torch.nn.Module` objects, which
   does not make a run globally smooth. Bound clipping (`torch.clamp`, zero
   gradient for clipped elements; detached when `preserve_gradients` is
   false), event and rate-group triggers, adaptive accept/reject and
   step-size choice, sub-step counts, and detached diagnostic values are not
   differentiable. Graph node caching never replays autograd-bearing values.
   Verify gradients on the executed graph before a sensitivity or calibration
   claim.

## Non-class root exports

The remaining root exports are not classes:

| Export | Kind |
| --- | --- |
| `DependencyList`, `DependencyDeclaration` | Dependency typing aliases. |
| `StateDict`, `EnvironmentDict`, `FluxDict` | Mapping typing aliases. |
| `create_simulator(config)` | `RELICSimulator` factory. |
| `load_config(...)` | Default `ConfigLoader` convenience function. |
| `SCALE_HIERARCHY` | Scale-order data. |
| `get_scale_hierarchy()`, `refresh_scale_hierarchy()` | Scale hierarchy accessors. |
| `__version__`, `__phase__`, `__author__`, `__license__` | Package metadata. |
