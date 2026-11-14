(api-parameters-contracts)=
# Parameters and Parameter Contracts

This page documents the canonical parameter-management API used by RELIC.
`UnifiedParameterManager` is the only concrete parameter manager. It owns
parameter lookup, validation, persistence, unit conversion, trainable
parameters, caching, and snapshots. `ParameterSpec` is the canonical metadata
contract for one parameter.

:::{important}
There is no importable class named `ParameterManager`. In architecture prose,
that name denotes the parameter-management role. Python code obtains the
process-wide `UnifiedParameterManager` with `get_global_manager()`.
Constructing `UnifiedParameterManager(...)` directly creates an independent
manager, which must be injected explicitly where it is used.

Scientific `FunctorBase` subclasses must not call the global manager directly.
They retrieve values only through
`self.get_parameter_value("key", required=...)`. Non-Functor infrastructure
uses `get_global_manager().get(...)`.
:::

## Stable imports

```python
from relic.parameters import (
    BoundsPolicy,
    ParameterDatabase,
    ParameterDefinition,
    ParameterSchema,
    ParameterSnapshot,
    ParameterSpec,
    ParameterStorage,
    ParameterValidator,
    UnifiedParameterManager,
    UnitValidationMode,
    get_global_manager,
    list_parameters,
    reset_global_manager,
    set_value,
)

from relic.parameters.types import (
    ParameterValidationContext,
    ParameterValidationLevel,
    ParameterValidationResult,
)
```

The following direct module paths are canonical runtime integration paths but
are not re-exported by their package roots:

```python
from relic.coupling.scales.physical_constraint_manager import (
    ConstraintSpec,
    ConstraintType,
    PhysicalConstraintManager,
)

from relic.validation.conservation_guardian import (
    BudgetCheckResult,
    ConservationGuardian,
)
```

`ValidationLevel` and `ValidationContext` in `relic.parameters`, and
`ValidationResult` in `relic.parameters.types`, are compatibility aliases for
the domain-prefixed parameter types. New documentation and code should use
the domain-prefixed names above to avoid confusion with
`relic.validation.framework`.

## Ownership model

| Component | Canonical owner | Public responsibility |
| --- | --- | --- |
| Manager | `UnifiedParameterManager` | Resolve, validate, mutate, cache, persist, and expose parameters |
| Specification | `ParameterSpec` | Store one value together with units, bounds, provenance, and optimization metadata |
| Database | `ParameterDatabase` | Read and write the native and calibrated parameter tiers |
| Runtime storage | `ParameterStorage` | Provide thread-safe value storage, namespace indexing, and snapshots |
| Validation | `ParameterValidator` | Apply type, bounds, units, relationship, and scientific validation rules |
| Snapshot value | `ParameterSnapshot` | Hold an immutable copy of parameter values |
| Serialization envelope | `ParameterSchema` | Validate and version the parameter interchange mapping |
| Decorator metadata | `ParameterDefinition` | Carry metadata for the compatibility decorator API |
| Physical constraints | `PhysicalConstraintManager` | Apply manager-owned constraints during state and scale integration |
| Domain budgets | `ConservationGuardian` | Produce non-blocking domain conservation diagnostics |

These components form one parameter system. Applications must not create a
second parameter registry, cache, schema, or lookup facade.

## `UnifiedParameterManager`

Implementation:
`relic.parameters.manager.UnifiedParameterManager`

Stable import:

```python
from relic.parameters import UnifiedParameterManager
```

### Constructor

```python
UnifiedParameterManager(
    database_path: Optional[Path] = None,
    metadata: Optional[ManagerMetadata] = None,
    preload_all: bool = False,
    warmup_cache: bool = False,
    **kwargs,
)
```

| Parameter | Type and default | Purpose |
| --- | --- | --- |
| `database_path` | `Optional[Path] = None` | Parameter database root. `None` selects `relic\parameters\database`. |
| `metadata` | `Optional[ManagerMetadata] = None` | Base-manager identity and lifecycle metadata. The manager creates canonical metadata when omitted. |
| `preload_all` | `bool = False` | Load the complete parameter database during construction without relying on lazy namespace loading. A later `initialize()` loads nothing. |
| `warmup_cache` | `bool = False` | Warm value caches after preloading. It has no independent effect when `preload_all` is false. |
| `cache_enabled` | keyword-only through `kwargs`, default `True` | Enable parameter-value and unit-conversion caches. |
| `cache_max_size` | keyword-only through `kwargs`, default `1000` | Maximum number of parameter-value cache entries. |
| `cache_ttl` | keyword-only through `kwargs`, default `None` | Optional cache time-to-live. |

Each construction builds a new, independent manager; the class has no
singleton hook. The process-wide manager is obtained only through
`get_global_manager()`, and normal application code should use it. Code that
needs a separate owner constructs one and injects it explicitly, for example
through the `parameter_manager` argument of a Functor. An injected manager
must expose `database`, its `ParameterDatabase`: dotted-name resolution in
`FunctorBase.get_parameter_value()` and in the external coupling adapters
calls `relic.parameters.namespaces.is_declared_namespace(namespace, manager)`,
which asks `manager.database.declares_namespace(...)`. An independent manager
never rewrites the process-wide variable name mapper.

Under the default packaged database (`relic\parameters\database`), `138` YAML
files define `4,751` leaf entries (`134` native `relic` tier files with `4,642`
entries and `4` `calibrated` tier files with `109` entries), resolving after
tier precedence to `4,643` active runtime parameters (`810` learnable
parameters with `requires_grad=True`). Two distinct precedence rules govern
resolution: inside `ParameterDatabase`, later tiers override earlier tiers in
the order `relic`, enabled `calibrated` scopes (`global`, `regional`,
`site_specific`), and `add_source(...)` paths; inside a Functor,
`FunctorBase.get_parameter_value(..., source_order=("manager", "config"))`
queries `"manager"` (`UnifiedParameterManager`) before `"config"`
(Functor-local overrides) by default.

### `get()`

```python
manager.get(
    key: str,
    default: Any = None,
    namespace: Optional[str] = None,
    units: Optional[str] = None,
    expected_type: Optional[type] = None,
    required: bool = False,
    bounds: Optional[Tuple[float, float]] = None,
) -> ParameterValue
```

| Parameter | Purpose |
| --- | --- |
| `key` | Canonical or normalizable dot-notation parameter key. |
| `default` | Value returned when no parameter can be resolved and `required` is false. |
| `namespace` | Optional namespace prefix applied before key normalization. |
| `units` | Requested output units. The stored value is converted when the units are compatible. |
| `expected_type` | Require the resolved value to match a Python type. |
| `required` | Raise `KeyError` when the parameter is missing; when `False`, return `default`. |
| `bounds` | Optional read-time lower and upper bounds for a scalar value. |

Returns the resolved value, not its metadata. A request for a namespace prefix
(not a leaf key, without `default`, and with `required=False`) returns a nested
mapping of the stored values of every key below `<namespace>.` that the manager
holds or the database defines, the membership of `get_namespace_parameters`.
The mapping is built at each read and belongs to the caller. Reading has no
validation switch: a requested unit conversion is always checked, and any other
keyword raises `TypeError`. Every successful read, including the direct storage
path, is recorded in the access history that restart contracts read.

Raises:

- `KeyError` when `required=True` and the key cannot be resolved.
- `TypeError` when `expected_type` is incompatible with the resolved value.
- Unit-conversion or validation exceptions when the requested conversion or
  validation cannot be completed.

For a Python scalar or one-element Tensor, `bounds` clamps the returned value
and emits `UserWarning` when clamping occurs. It does not clamp multi-element
Tensors.

### `get_parameter()`

```python
manager.get_parameter(
    key: str,
    namespace: Optional[str] = None,
) -> Optional[ParameterSpec]
```

Returns the full `ParameterSpec` or `None`. This method is for metadata
inspection, including units, bounds, references, provenance, and
`requires_grad`. A key that the manager does not hold yet is read from the
database, which lazily loads the parameter files whose declared namespace owns
the key (see `ParameterDatabase`); a learnable value is then registered with
the manager's PyTorch parameter collection.

### Mutation and persistence

```python
manager.set_value(
    key: str,
    value: ParameterValue,
    units: Optional[str] = None,
    validate: bool = True,
    notify: bool = True,
) -> None
```

Updates a value, optionally validates it, invalidates affected caches, and
notifies registered observers. `key` must name a registered parameter (loaded
or in the database): an unknown key raises `KeyError` and changes nothing, so a
misspelled key never becomes a new parameter. A new parameter is registered
first with `register_parameter(...)`. Validation failures
raise `ValueError`. Out-of-bounds behavior is controlled by `BoundsPolicy` and
the manager's validation configuration; an invalid `RELIC_BOUNDS_POLICY` or
`RELIC_UNIT_VALIDATION_MODE` environment value raises `ValueError` when the
manager is created. `units` names the unit of `value`: the value is converted
into the declared unit of the specification, which is never replaced; the
dimensionless aliases `"1"` and `"dimensionless"` need no conversion, and a
pair the unit owner cannot convert raises `ValueError` without writing.
`set_batch(updates, validate=True, notify=True)` commits the whole batch (or
restores every owner on a failure), then notifies every changed key; callback
failures of all keys are raised together as one `ExceptionGroup` afterwards,
and the changes stay committed.

```python
manager.set_parameter(
    spec: ParameterSpec,
    tier: str,
) -> None
```

Persists one complete specification to the database tier and sets that
parameter's specification, stored value and live binding together; no other
specification is loaded into the manager. An existing binding of the same shape
and dtype keeps its `nn.Parameter` object, gradient flag, and device and takes
the new value in place; a specification that is no longer learnable loses its binding. A
learnable specification without a tensor form raises `TypeError` before
anything is written. `spec` must be a `ParameterSpec`; otherwise `TypeError` is
raised. Loading the whole database (`preload_all`, a new `database_path`)
gives every specification its stored value and every learnable
specification its binding in the same way.

```python
manager.register_parameter(spec: ParameterSpec) -> None
```

Registers a specification in the existing manager and storage layer. This is a
lower-level registration operation used by database and extension loading. It
does not create another parameter owner. `register_parameter` and
`set_parameter(spec, tier)` use the same `bind_learnable_value` rule, including
database re-registration: matching shape and dtype preserve the live binding
object, gradient flag, and device and copy the value in place. Changed shape
or dtype creates a new binding; a non-learnable specification has none.
`set_parameter` requires the destination `tier` explicitly.

### Discovery and namespace access

```python
manager.list(
    namespace: Optional[str] = None,
    pattern: Optional[str] = None,
    category: Optional[ParameterCategory] = None,
) -> List[str]
```

Returns matching canonical keys.

```python
manager.get_namespace_parameters(
    namespace: str,
    include_subnamespaces: bool = True,
    return_specs: bool = False,
) -> Dict[str, Union[ParameterValue, ParameterSpec]]
```

Returns values by default or complete specifications when
`return_specs=True`.

### Validation, optimization, and snapshots

```python
manager.validate_parameters(
    keys: Optional[List[str]] = None,
) -> List[Dict[str, str]]
```

Returns zero or more records with `parameter` and `error` keys. Ordinary
validation failures are represented in the returned list.

```python
manager.get_trainable_parameters() -> Dict[str, torch.nn.Parameter]
manager.freeze(keys: Optional[List[str]] = None) -> None
manager.unfreeze(keys: Optional[List[str]] = None) -> None
manager.to(
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> UnifiedParameterManager
```

These methods expose and migrate the manager-owned PyTorch parameters. Device
assignment of scientific Functors remains owned by the simulator device
policy; this API does not authorize a scientific module to select its own
device.

```python
manager.snapshot() -> Dict[str, Any]
manager.restore(snapshot: Dict[str, Any]) -> None
manager.create_snapshot(description: str = "") -> ParameterSnapshot
manager.restore_snapshot(snapshot: ParameterSnapshot) -> None
manager.get_diff(snapshot: ParameterSnapshot) -> Dict[str, Tuple[Any, Any]]
manager.diff(other_snapshot: Dict[str, Any]) -> Dict[str, Any]
```

The manager snapshot keeps each owner of parameter state apart and exactly as
it is: `parameters` holds the serialized specifications with their declared
values (type and dtype kept), `storage_snapshot` the storage's own snapshot of
the stored values (a specification whose value is not stored yet stays without
one), `nn_values` a copy of every live learnable `nn.Parameter` value (an
optimizer may have moved it away from the declared value) and
`nn_requires_grad` its gradient flag, next to the device, the validation
settings and the unit issues. `restore()` validates the whole snapshot before
it changes anything: it requires a `parameters` mapping, stored values that
agree with their specifications, gradient flags only for learnable
specifications, and one live value per flag with the shape and dtype of its
specification; otherwise it raises `ValueError`. It then restores the stored
values, the specifications, the live values (copied into the existing
bindings, across devices, without replacing them; bindings created after the
snapshot are removed) and the settings, so every value with its type and
dtype, and every specification, equals its state at snapshot time.

`create_snapshot()` captures every parameter's value exactly (its stored
value, or its declared value while none is stored) and the live learnable
values in `metadata["nn_values"]`; `restore_snapshot()` writes those values
back, without validation, clamping or conversion, to the parameters that still
exist and copies the live values into their bindings. Parameters registered
after the snapshot are kept. `get_diff()`, `diff()` and the experiment diff
compare resolved values: the live learnable value over the stored or declared
value, which is the value `manager.get(...)` returns.

## Global manager functions

### `get_global_manager()`

```python
get_global_manager(
    preload_all: bool = False,
    warmup_cache: bool = False,
) -> UnifiedParameterManager
```

Returns the process-wide canonical manager. The first call creates it under
a reentrant lock, calls `initialize()`, and refreshes the process-wide
variable name mapper from it; if either step fails, the global manager is
reset and the error is raised. Later calls return the same object; a later
call with `preload_all=True` preloads it when it was not preloaded. Such a
late preload adopts the database specifications of the keys the manager does
not hold yet; a key it already holds (loaded on demand, registered, or
written with `set_value`) keeps its specification, stored value and live
binding. `initialize()` loads nothing: a manager constructed with
`preload_all=True` holds the whole database when its constructor returns, and
a value written or a learnable binding moved before `initialize()` is kept. A
database switch (`configure` with a new `database_path`) unit-checks every
specification of the new database under the manager's unit validation mode
(check-only records an issue, strict raises before anything is adopted),
replaces the held specifications, stored values and binding values with those
of the new database, then clears the query cache and the unit-conversion cache
as a whole (also when the load fails part way), so no lookup cached from the
previous database survives the switch. A switch that fails keeps the previous
`database_path` and database, the ones that define the held specifications.

### `reset_global_manager()`

```python
reset_global_manager() -> None
```

Clears the process-wide manager instance. This is a lifecycle and isolation
operation, not a routine scientific workflow operation.

### Convenience accessors

```python
get(
    key: str,
    default: Any = None,
    namespace: Optional[str] = None,
    units: Optional[str] = None,
    expected_type: Optional[type] = None,
    required: bool = False,
) -> ParameterValue

set_value(
    key: str,
    value: ParameterValue,
    units: Optional[str] = None,
    validate: bool = True,
    notify: bool = True,
) -> None

list_parameters(
    namespace: Optional[str] = None,
    pattern: Optional[str] = None,
    category: Optional[ParameterCategory] = None,
) -> List[str]

get_namespace_parameters(
    namespace: str,
    include_subnamespaces: bool = True,
    return_specs: bool = False,
) -> Dict[str, Union[ParameterValue, ParameterSpec]]
```

These functions delegate to the process-wide manager from
`get_global_manager()`. They are convenience entry points, not alternate
managers. `get`, `set_value`, and `list_parameters` are exported by
`relic.parameters`; `get_namespace_parameters` is imported from
`relic.parameters.manager`.

## `ParameterSpec`

Implementation:
`relic.parameters.spec.ParameterSpec`

Stable import:

```python
from relic.parameters import ParameterSpec
```

### Constructor

```python
ParameterSpec(
    key: str,
    value: ParameterValue,
    units: str,
    dtype: str,
    bounds: Optional[Tuple[Any, Any]],
    category: ParameterCategory,
    default: Optional[ParameterValue] = None,
    description: str = "",
    references: List[str] = [],
    tags: Set[str] = set(),
    dimension: str = "",
    provenance: MetadataDict = {},
    requires_grad: bool = False,
    allowed_values: Optional[List[Any]] = None,
)
```

The displayed empty collections are produced by dataclass factories; instances
do not share mutable defaults. `units`, `dtype`, `bounds`, and `category`
have no default:
a numeric `dtype` declares a `(lower, upper)` interval and writes an
unbounded side as `-inf` or `inf` (`bounds=(-inf, inf)` is explicitly
unbounded); `bounds=None` states that no interval applies and is accepted
only for a non-numeric `dtype` (text, boolean switch, enumeration, list or
mapping). `relic.parameters.spec.parameter_bounds_error(bounds, dtype)` is
the one rule: `ParameterSpec` raises `ValueError` on a violation, and
`has_complete_parameter_metadata`, the structural parameter audit and the
parameter YAML schema read the same rule.

A `list` or `tuple` dtype may declare one `[minimum, maximum]` pair of numbers
per component in place of a single scalar interval, for example
`[[223.15, 273.15], [323.15, 373.15]]` for a two-component value.
`relic.parameters.spec.componentwise_bounds_errors(value, pairs)` is the one
rule for such a declaration: the value is a list or tuple with one finite real
component per pair, each inside its closed pair. `validate_value()` returns the
violations, a validated `set_value(...)` and a validated
`FunctorBase.get_parameter_value(...)` raise `ValueError`, and no component is
clamped. A flat `(lower, upper)` interval on a list value is not applied to its
components.

A numeric parameter declares its bounds endpoints, value, and default as
numbers. `parameter_bounds_error` rejects textual numeric endpoints, and
`ParameterSpec` rejects textual numeric values or defaults; neither is
converted by `from_dict`. With PyYAML's YAML 1.1 rules, write `1.0e+7`
or `1.0e-6`: `1.0e7` and `1e-6` are strings and are rejected.

| Field | Purpose | Validation and failure behavior |
| --- | --- | --- |
| `key` | Unique dot-notation identifier. | Empty or malformed keys raise `ValueError`. |
| `value` | Current scalar, configuration value, or Tensor. | Checked by `validate_value()`. |
| `default` | Optional declared default. | Stored on the object and serialized by `to_dict()`. |
| `units` | Physical units, normally CF/UDUNITS-compatible. | Unit validation is performed by the manager and validator. |
| `dtype` | Declared numeric or configuration type. | Controls Tensor conversion and numeric normalization. |
| `bounds` | Lower and upper physical limits; a numeric `dtype` writes an open side as `-inf` or `inf`, and `None` (no interval applies) is valid only for a non-numeric `dtype`. | Required. `None` or a `None` endpoint with a numeric `dtype`, an invalid shape, or lower greater than upper raises `ValueError`. |
| `category` | `ParameterCategory` classification. | Required. Strings are normalized; unsupported types raise `TypeError`. |
| `description` | Scientific meaning and usage context. | Free text. |
| `references` | Literature, standard, or dataset citations. | Stored as a list of strings. |
| `tags` | Search and workflow labels. | A list is normalized to a set. |
| `dimension` | Physical-dimension description. | Used by unit and scientific checks. |
| `provenance` | Source, schema, version, and calibration metadata. | An empty mapping is replaced with basic user-defined provenance. |
| `requires_grad` | Enable trainable PyTorch representation. | Non-floating gradient values are converted when possible or rejected. |
| `allowed_values` | Enumerated values for categorical parameters. | Values outside the declaration are reported by `validate_value()`. |

### `validate_value()`

```python
spec.validate_value(value: ParameterValue) -> List[str]
```

Returns validation messages without raising for ordinary invalid values.
Checks include categorical membership, finite numeric values, scalar bounds,
non-negative biological rates and concentrations, and physically valid
absolute temperatures.

### Tensor conversion

```python
spec.to_tensor(
    device: Optional[torch.device] = None,
) -> torch.Tensor

spec.from_tensor(tensor: torch.Tensor) -> None
```

`to_tensor()` maps declared boolean, integer, and floating dtypes to PyTorch.
String and categorical values raise `TypeError`. A non-floating Tensor cannot
retain gradients; conversion failure raises `TypeError`.

`from_tensor()` accepts only a scalar Tensor. A non-Tensor raises `TypeError`;
a non-scalar Tensor raises `ValueError`.

### Mapping and serialization interface

`ParameterSpec` implements `Mapping[str, Any]` and provides:

```python
spec.to_dict() -> Dict[str, Any]
spec.model_dump(**kwargs) -> Dict[str, Any]
ParameterSpec.from_dict(data: Dict[str, Any]) -> ParameterSpec
```

`from_dict()` requires `value`, normalizes category, dtype, tags, bounds, and
categorical values, and moves selected additional metadata into provenance.

`to_dict()` and `model_dump()` also emit `default`, in the same Tensor wire
format as `value`, and `allowed_values`, as a list or `None`. `from_dict()`
decodes a Tensor-encoded `default` and normalizes `allowed_values` to a list
or `None`, so both fields round-trip.

## `ParameterSchema`

Implementation:
`relic.parameters.serialization.ParameterSchema`

Stable public import:

```python
from relic.parameters import ParameterSchema
```

Status: stable serialization helper. It owns the parameter interchange
envelope, not parameter lookup, per-entry metadata, or database persistence.
The class has no constructor parameters and is intended as a namespace for its
version and static methods.

```python
from relic.parameters import ParameterSchema

assert ParameterSchema.VERSION == "1.0.0"
```

The canonical envelope has a `parameters` mapping. Each member is a mapping
that contains at least `value`. `version`, `timestamp`, and `metadata` may
accompany it.

### `validate_schema()`

```python
ParameterSchema.validate_schema(
    data: Dict[str, Any],
) -> bool
```

Returns `True` only when:

- `data` contains `parameters`;
- `parameters` is a mapping;
- every parameter entry is a mapping containing `value`.

It returns `False` for an incomplete canonical mapping. The current method is
a lightweight structural check; it does not validate all `ParameterSpec`
fields, units, bounds, or provenance. Inputs outside the declared mapping
contract may raise ordinary Python type errors.

### `migrate_schema()`

```python
ParameterSchema.migrate_schema(
    data: Dict[str, Any],
    from_version: Optional[str] = None,
) -> Dict[str, Any]
```

Returns a shallow copy of a canonical mapping and adds
`ParameterSchema.VERSION` when `version` is absent.

Raises `ValueError` when `data` is not a mapping, when `parameters` is not a
mapping, or when the canonical `parameters` field is missing.

Current implementation limits:

- Flat legacy parameter maps are rejected without migration.
- `from_version` is accepted by the signature but does not currently select a
  version-specific migration path.
- `ParameterSchema` must not be treated as a second `ParameterSpec` schema or
  validation registry.

## `ParameterDefinition`

Implementation:
`relic.parameters.decorators.ParameterDefinition`

Stable public import:

```python
from relic.parameters import ParameterDefinition
```

Status: exported compatibility and convenience metadata for the parameter
decorator API. It is not the canonical runtime parameter object and must not
replace `ParameterSpec`.

```python
ParameterDefinition(
    name: str,
    default: Any,
    units: str,
    bounds: tuple,
    description: str = "",
    requires_grad: bool = False,
    validate: bool = False,
    register: bool = False,
)
```

| Field | Purpose |
| --- | --- |
| `name` | Parameter name attached to a decorated property. |
| `default` | Local fallback used by the decorator. |
| `units` | Declared physical units (required; there is no universal unit). |
| `bounds` | Lower and upper limits passed to decorator-managed access (required; `(-inf, inf)` declares an unbounded parameter). |
| `description` | Human-readable meaning. |
| `requires_grad` | Request a trainable value when decorator registration is used. |
| `validate` | Request decorator-time validation. |
| `register` | Request registration of a missing decorated parameter. |

The dataclass has no validation methods or `__post_init__`. Invalid names,
bounds, units, or defaults are not rejected at construction and can fail only
when the owning decorator or manager uses them.

The decorators `managed_parameters`, `parameter`, `validated_parameter`, and
`learnable_parameter` attach or consume this metadata. `managed_parameters`
registers only an attribute `<name>` that declares a complete `<name>_spec`
mapping (`units`, `bounds`, `dtype`, and `description`); an incomplete spec
raises `ValueError`, and an attribute without a spec is not a managed
parameter. They remain exported,
but production scientific Functors must use YAML-backed `ParameterSpec`
entries and `self.get_parameter_value(...)`. A decorated property is not an
alternative parameter owner or an approved way to hardcode scientific
defaults.

## Manager-owned support classes

### `ParameterDatabase`

```python
ParameterDatabase(
    base_path: Path,
    validator: Optional[ParameterValidator] = None,
)
```

The database owns two precedence tiers:

- `relic`: native framework parameters.
- `calibrated`: calibration and optimized values.

Construction creates the `relic` and `calibrated` tier directories under
`base_path` and initializes an empty parameter cache and a per-file header
cache; it builds no file index. Every read follows one load plan:

1. the native `relic` files (the base tier);
2. the enabled calibrated scopes `global`, `regional`, and `site_specific`, in
   that order (`RELIC_CALIBRATED_SCALES` selects scopes, top-level calibrated
   snapshot files take part only with `RELIC_INCLUDE_CALIBRATED_SNAPSHOTS=1`,
   and `RELIC_SKIP_CALIBRATED=1` excludes the calibrated tier);
3. the sources added with `add_source()`.

Later entries take precedence, and a key defined in two base-tier files
raises `ValueError` naming both files. The namespace of a YAML file is its
top-level `namespace:` header line, read from the raw bytes and cached by
modification time and size; file and directory names play no role. JSON files
and YAML files without that header are read only by the full loads. Primary
methods include:

```python
database.load_all() -> Dict[str, ParameterSpec]
database.load_all_with_priority(
    source_priority: List[str],
) -> Dict[str, ParameterSpec]
database.get_parameter(key: str) -> Optional[ParameterSpec]
database.set_parameter(
    spec: ParameterSpec,
    tier: str = "calibrated",
) -> None
database.add_source(source_path: Path) -> None
database.load_namespace(namespace: str) -> None
database.declares_namespace(namespace: str) -> bool
database.declared_namespaces(
    directory: Optional[Path] = None,
) -> List[str]
```

`get_parameter(key)` returns a cached specification directly. Otherwise it
loads the files whose declared namespace is a dot prefix of the key, together
with every file of a related (ancestor or descendant) namespace, in full-load
precedence.
If that does not define the key, it runs `load_all()`; with
`fail_on_validation_errors` set, it raises `KeyError` instead. That flag comes
from `RELIC_FAIL_ON_PARAMETER_VALIDATION` (`1`, `true`, or `yes`) or, when
that variable is unset, from a true `CI` variable or a `RELIC_ENV` of `ci`,
`prod`, or `production`.

`load_namespace(namespace)` reads the files of related namespaces in the same
way, `declares_namespace(namespace)` reports whether a participating file
declares the namespace, an ancestor, or a descendant, and
`declared_namespaces()` lists the declared namespaces in load order.
`load_all_with_priority(source_priority)` accepts only `relic` and
`calibrated`, highest precedence first, reads those tiers of the load plan
without the additional sources, raises `ValueError` for any other name, and
leaves the cache unchanged. `add_source()` raises `FileNotFoundError` for a
path that does not exist.

Application code normally reaches this object through
`UnifiedParameterManager.database`. Direct construction is appropriate only
for controlled infrastructure that intentionally owns a database root.

### `ParameterStorage`

```python
ParameterStorage() -> ParameterStorage
```

Provides thread-safe in-memory CRUD, namespace indexing, access statistics,
and immutable snapshots.

```python
storage.get(key, default=None) -> Optional[ParameterValue]
storage.set(key, value) -> None
storage.delete(key) -> bool
storage.exists(key) -> bool
storage.list(namespace=None, pattern=None) -> List[str]
storage.get_all() -> Dict[str, ParameterValue]
storage.snapshot(description="", *, parameters=None, metadata=None) -> ParameterSnapshot
storage.restore(snapshot: ParameterSnapshot) -> None
storage.diff(snapshot1=None, snapshot2=None) -> Dict[str, Tuple[Any, Any]]
```

`storage.list(namespace=...)` returns every key below `"<namespace>."`
(children and deeper descendants) with one rule for every namespace; a key
of a sibling prefix (`a.bc.x` for namespace `a.b`) is never returned.

`UnifiedParameterManager.delete(key)` removes the specification, the storage
value and any registered `nn.Parameter`, then clears the query cache and the
unit-conversion cache as a whole, so no cached lookup of the deleted key or of
its namespace/unit variants survives the deletion. `storage.snapshot()`
captures the stored values, or the explicit `parameters` values without
changing live storage, each with its type and dtype; `metadata` is recorded
with the snapshot. `storage.restore(snapshot)` replaces the stored values and
then notifies the manager. The manager requires a specification for every
restored value, the loaded one or the database one; a value without any
specification raises `KeyError`, and no specification is invented. It brings
specification values and live bindings in line with the restored values and
drops the specifications and bindings of parameters absent from the snapshot.
When a restore callback raises, its error propagates and the storage takes
back the values it held before the restore. `manager.restore(snapshot)` writes
the stored values directly and sets the snapshotted specifications exactly
itself.

This is the manager's value layer, not a public alternative to
`UnifiedParameterManager`.

### `ParameterSnapshot`

```python
ParameterSnapshot(
    timestamp: datetime,
    _parameters: ParameterDict,
    metadata: Dict[str, Any] = {},
    description: str = "",
)
```

The frozen dataclass deep-copies its values and metadata. The `parameters`
property returns a read-only `MappingProxyType`. A manager snapshot records the
live learnable values under `metadata["nn_values"]`.

### `ParameterValidator`

```python
ParameterValidator(
    level: ParameterValidationLevel = ParameterValidationLevel.STANDARD,
)
```

```python
validator.validate(
    specs: Dict[str, ParameterSpec],
    level: Optional[ParameterValidationLevel] = None,
    context: Optional[ParameterValidationContext] = None,
) -> List[str]

validator.validate_value(
    key: str,
    value: ParameterValue,
    spec: ParameterSpec,
    level: Optional[ParameterValidationLevel] = None,
    context: Optional[ParameterValidationContext] = None,
) -> ParameterValidationResult
```

Validation levels:

- `NONE`: no validation.
- `BASIC`: type and range checks.
- `STANDARD`: standard scientific and relationship checks.
- `STRICT`: full checks with strict context.
- `CUSTOM`: caller-defined policy.

`ParameterValidationResult` contains `is_valid`, `errors`, `warnings`, `level`,
and machine-readable `error_types`. `ParameterValidationContext` contains
related parameter values, environmental context, `strict_mode`, and
`warnings_as_errors`.

### `ParameterValidationResult`

```python
ParameterValidationResult(
    is_valid: bool,
    errors: List[str] = [],
    warnings: List[str] = [],
    error_types: List[str] = [],
)
```

The result reports validation status without requiring an exception.
`aggregate(results)` combines errors, warnings, error categories, and joint
validity across multiple results.

### `ParameterValidationContext`

```python
ParameterValidationContext(
    all_parameters: Dict[str, ParameterValue] = {},
    environment: Dict[str, Any] = {},
    strict_mode: bool = False,
    warnings_as_errors: bool = False,
)
```

The context supplies related values and environmental information to
cross-parameter and scientific validators. Strict validation enables
`strict_mode` and treats warnings as errors.

### `BoundsPolicy` and `UnitValidationMode`

`BoundsPolicy` selects `WARN_AND_CLAMP` or `ERROR`.
`UnitValidationMode` selects `OFF`, `CHECK_ONLY`, or `STRICT`.

## Physical constraint integration

### Stability and ownership

`PhysicalConstraintManager`, `ConstraintSpec`, and `ConstraintType` are defined
in:

```text
relic\coupling\scales\physical_constraint_manager.py
```

They are not exported by `relic.coupling.scales` or `relic.coupling`.
Their direct module path is the canonical infrastructure integration path, not
a general package-root public API.

Each `StateManager` owns its physical-constraint manager. It first uses an
explicit `physical_constraint_manager=` constructor injection; otherwise its
read-only `physical_constraint_manager` property lazily constructs,
initializes, validates, and caches one instance for that StateManager:

```python
from relic.core.state import StateManager

state_manager = StateManager(autoload_contract_registry=False)
constraint_manager = state_manager.physical_constraint_manager
assert constraint_manager is not None
```

There is no module-global physical-constraint getter or setter. Scientific
Functors propose their state updates through the existing contract and do not
create competing constraint owners. Infrastructure that needs explicit
construction uses the same class and its normal lifecycle.

### `ConstraintSpec`

```python
ConstraintSpec(
    name: str,
    type: ConstraintType,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    smoothness: Optional[float] = None,
    strict: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
)
```

| Field | Purpose |
| --- | --- |
| `name` | Constraint identity used for filtering and diagnostics. |
| `type` | Constraint operation selected from `ConstraintType`. |
| `min_value` | Optional lower bound for `BOUNDED`. |
| `max_value` | Optional upper bound for `BOUNDED`. |
| `smoothness` | Optional smoothing scale used by `SMOOTH`. |
| `strict` | Declared strictness flag carried by the specification. |
| `metadata` | Optional provenance and variable context. |

`ConstraintType` defines `NON_NEGATIVE`, `BOUNDED`, `MONOTONIC`, `CONVEX`,
`SMOOTH`, and `CONSERVE_SHAPE`.

`ConstraintSpec` parses `type` at construction: a `ConstraintType` or its
value string; another value raises `ValueError` naming the types. The
manager's validation phase reports a bounded specification with no limits or
with `min_value >= max_value`. `add_constraint` takes only a `ConstraintSpec`
with a known type, and a type changed after the constraint was added raises
`ValueError` when the constraints are applied; no constraint is skipped.

### `PhysicalConstraintManager`

```python
PhysicalConstraintManager(
    constraints: Optional[List[ConstraintSpec]] = None,
    method: Optional[str] = None,
    preserve_gradients: Optional[bool] = None,
    device: Optional[torch.device] = None,
    parameter_manager: Optional[Any] = None,
)
```

| Parameter | Purpose |
| --- | --- |
| `constraints` | Initial constraint specifications (`ConstraintSpec` entries; another entry raises `ValueError`). `initialize()` copies them, so constraints added later do not survive `reset()` and the next `initialize()`. |
| `method` | Application strategy, one of the `allowed_values` of `coupling.scales.defaults.physical_constraint_method` (`clip`, `scale`, `redistribute`, `project`); another value raises `ValueError` naming them, at construction and at `configure` (a rejected call changes nothing). |
| `preserve_gradients` | Clone data while preserving autograd connectivity without detaching tensors. With the `clip` method, clipped elements still receive zero gradient through `torch.clamp`. `True` or `False` (another value raises `ValueError`); `None` takes the parameter. |
| `device` | Device for manager operations and adaptive parameters. |
| `parameter_manager` | Explicit parameter owner used by initialization; the canonical global parameter owner is resolved only when this is absent. |

Direct construction creates an uninitialized `BaseManager`/`nn.Module`.
Infrastructure calls `initialize()` and `validate()`. Normal state updates
obtain the initialized, cached owner through
`state_manager.physical_constraint_manager`.

When explicit settings are omitted, initialization resolves
`physical_constraint_method`, `preserve_gradients`, and `constraint_device`
from `coupling.scales.defaults` through the canonical parameter manager.
Missing required parameter values propagate lookup errors, and a
`physical_constraint_method` parameter outside its `allowed_values` raises
`ValueError` naming them; `initialize()` reports such a failure, like every
initialization failure, as `ManagerInitializationError` with the error as its
cause and leaves the manager in the `ERROR` state. The constraint operations
never treat an unknown method as `clip`: they raise `ValueError`. If neither a
manager nor all explicit settings are available, initialization raises
`RuntimeError`, reported the same way.

An explicitly injected `parameter_manager` controls initialization and
subsequent parameter lookups. Only when it is absent does `_do_initialize()`
obtain the canonical `get_global_manager()`.

### Constraint operations

```python
manager.add_constraint(
    constraint: ConstraintSpec,
) -> None

manager.apply_constraints(
    data: torch.Tensor,
    constraint_names: Optional[List[str]] = None,
    return_mask: bool = False,
    *,
    bounds: Optional[
        Tuple[Optional[float], Optional[float]]
    ] = None,
    variable_name: Optional[str] = None,
    scale: Optional[str] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

| Parameter | Purpose |
| --- | --- |
| `data` | Tensor to constrain. |
| `constraint_names` | Optional subset of registered constraint names. |
| `return_mask` | Return a boolean violation mask with the constrained Tensor. |
| `bounds` | Runtime lower and upper limits used to create a temporary bounded constraint. |
| `variable_name` | Variable identity for the catalog bounds lookup and diagnostics. |
| `scale` | Optional scale namespace used during bounds lookup. |

If `bounds` is absent and `variable_name` is present, the bounds are the
`bounds` field of the variable's scale-catalog entry
(`coupling.scales.<scale>.variables`, then `coupling.scales.defaults.variables`,
matched like every catalog lookup); a variable that no catalog declares, or an
entry without a `bounds` field, raises `KeyError`, and `bounds: null` means no
bound. No other namespace, `<name>_bounds` key or clamp key is consulted. The
`ScaleManager` facade passes the bounds of the variable's resolved metadata
(call metadata, else its catalog entry) and asks for no lookup when that
declaration has none. It records violations but does
not create another state or conservation owner. Calling this method before
manager initialization can fail because runtime attributes are not available.
Tensor shape and operator errors propagate from PyTorch.

```python
manager.get_violation_report() -> Dict[str, Any]
manager.reset_tracking() -> None
```

The report contains total violations, check count, violation rate, and up to
the ten most recent history entries.

`configure` accepts only `method`, `preserve_gradients`, `device` and
`constraints`: an unknown key, an invalid method, a `preserve_gradients` that
is not `True` or `False`, a `device` that is not a torch device, and
`constraints` that are not a list of mappings of `ConstraintSpec` fields (with
`name` and `type`, a known type and no other field) raise `ValueError` before
the call changes anything. Manager
`validate()` reports invalid methods, unavailable CUDA devices, and malformed
bounded constraints through the BaseManager validation result.

## Conservation guardian

### Stability and ownership

`ConservationGuardian` is the canonical fourth-layer domain budget checker
named by the architecture, but it is not re-exported from
`relic.validation`. Use the direct module path only for validation
infrastructure:

```python
from relic.validation.conservation_guardian import (
    BudgetCheckResult,
    ConservationGuardian,
)
```

`StateManager` owns commit-time state ledgers and invokes the guardian through
its conservation integration. `ConservationGuardian` is a non-blocking
diagnostic checker; it is not another StateManager, ledger, generic validation
result, or scientific process.

### `BudgetCheckResult`

```python
BudgetCheckResult(
    domain: str,
    members: List[str],
    sum_flux_in: float,
    sum_flux_out: float,
    delta_storage: float,
    residual: float,
    passed: bool,
    tolerance_used: float,
    invariant_drift: Optional[float] = None,
)
```

| Field | Purpose |
| --- | --- |
| `domain` | Conservation-domain name. |
| `members` | State variables obtained from the contract registry. |
| `sum_flux_in` | Aggregated incoming flux. |
| `sum_flux_out` | Aggregated outgoing flux. |
| `delta_storage` | Observed storage change. |
| `residual` | Absolute difference between observed and expected storage change. |
| `passed` | Whether the residual is within tolerance. |
| `tolerance_used` | Computed absolute-plus-relative threshold. |
| `invariant_drift` | Optional separately computed Noether invariant drift. |

The dataclass performs no constructor-time validation. It is a domain result
owned by the guardian and must not be confused with
`relic.validation.framework.ValidationResultModel`.

### `ConservationGuardian`

```python
ConservationGuardian(
    contract_registry: Any,
    default_atol: float = 1.0e-10,
    default_rtol: float = 1.0e-8,
)
```

| Parameter | Purpose |
| --- | --- |
| `contract_registry` | Existing registry that supplies `get_domain_members(domain)`. |
| `default_atol` | Default absolute residual tolerance. |
| `default_rtol` | Default relative residual tolerance. |

The constructor converts tolerances to `float`; incompatible values raise
`TypeError` or `ValueError`. It does not currently enforce non-negative or
finite tolerances.

```python
guardian.check_domain(
    domain: str,
    flux_in: Dict[str, Union[torch.Tensor, float, int]],
    flux_out: Dict[str, Union[torch.Tensor, float, int]],
    storage_before: Dict[str, Union[torch.Tensor, float, int]],
    storage_after: Dict[str, Union[torch.Tensor, float, int]],
    dt: float,
    tolerance: Optional[Dict[str, float]] = None,
) -> BudgetCheckResult
```

The domain balance is:

$$
\Delta S_{\mathrm{expected}}
= \left(\sum F_{\mathrm{in}} - \sum F_{\mathrm{out}}\right) \Delta t.
$$

The threshold is:

$$
\tau
= \mathrm{atol}
+ \mathrm{rtol}
  \lvert\Delta S_{\mathrm{expected}}\rvert.
$$

A failed budget logs an error and returns `passed=False`; it does not raise
merely because the budget failed. Registry, conversion, or Tensor failures can
still propagate. `check_domain()` does not populate `invariant_drift`; that
diagnostic is computed separately.

```python
guardian.compute_invariant_drift(
    incidence_row: numpy.ndarray,
    flux_vector: Dict[
        str,
        Union[torch.Tensor, float, int],
    ],
    dt: float,
) -> float
```

Returns zero for an empty coefficient row or empty flux vector. Otherwise the
coefficient count must equal the number of flux entries; a mismatch raises
`ValueError`.

Scientific modules must not reproduce these tolerance and budget loops in
their `forward()` methods. Local checks remain owned by `FunctorBase`,
commit-time ledgers by `StateManager`, and domain diagnostics by
`ConservationGuardian`.

## Physical constants and anti-redundancy rules

Physical constants have one database owner:

```text
relic\parameters\database\relic\physical\constants.yaml
```

Non-Functor infrastructure reads them through:

```python
from relic.utils.physical_constants import get_physical_constant

kappa = float(get_physical_constant("von_karman_constant"))
assert 0.39 < kappa < 0.42
```

Do not duplicate physical constants in Python or domain parameter files.
Do not add another manager, parameter accessor, cache, database facade, or
compatibility wrapper. In particular, removed access chains such as
`get_parameter_by_path()`, `_require_parameter()`,
`require_parameter_value()`, and `ParameterAccessorMixin` are not part of the
public architecture.
