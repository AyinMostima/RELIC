(orchestration-validation-api)=
# Orchestration, Contracts, and Validation API

This page documents the canonical public types that describe a RELIC workflow,
validate its interfaces, compile it into an execution plan, and execute that
plan through the existing simulator. It also documents the variable-contract
and unified-validation APIs used by the same architecture.

The signatures below use `...` for a field backed by a fresh
`default_factory` container. Canonical owner paths identify the module that
implements a type. Convenience imports identify the supported shorter import
when one exists.

## Graph IR data model

### `GraphIR`

Canonical owner: `relic.orchestration.graph_ir.GraphIR`

Convenience import:

```python
from relic.orchestration import GraphIR
```

Constructor:

```python
GraphIR(
    schema_version: str,
    nodes: Dict[str, NodeSpec],
    links: List[LinkSpec] = ...,
    metadata: Dict[str, Any] = ...,
    assertion: Dict[str, Any] = ...,
)
```

Parameters:

- `schema_version` identifies the GraphIR schema understood by the loader and
  compiler.
- `nodes` maps each workflow-local node identifier to a `NodeSpec`.
- `links` contains explicit directed connections between node ports.
- `metadata` carries workflow-level information that is preserved in the
  compile report.
- `assertion` contains structural assertion payloads evaluated by graph
  validation. DEC, IPHS, reaction-network, port, state-domain, and related
  Gate 0 checks enter the compiler through this field.

Key methods:

```python
GraphIR.from_dict(payload: Dict[str, Any]) -> GraphIR
graph.to_dict() -> Dict[str, Any]
```

`from_dict()` accepts the serialized GraphIR form. If explicit links are
absent, it infers links from input bindings of the form
`{"link": [source_node, source_port]}`. `to_dict()` returns the corresponding
plain mapping and omits `assertion` only when that mapping is empty.

Failure semantics:

- A non-mapping payload, nodes value, node payload, inputs value, metadata
  value, or assertion value raises `TypeError`.
- A non-list outputs or links value raises `TypeError`.
- A missing or empty `schema_version` raises `ValueError`.
- A node without `class_type` raises `ValueError`.
- Parsing does not prove that referenced nodes and ports exist. Those checks
  belong to `validate_graph()`.

### `NodeSpec`

Canonical owner: `relic.orchestration.graph_ir.NodeSpec`

Convenience import:

```python
from relic.orchestration import NodeSpec
```

Constructor:

```python
NodeSpec(
    class_type: str,
    inputs: Dict[str, Any] = ...,
    outputs: List[str] = ...,
    config: Dict[str, Any] = ...,
)
```

Parameters:

- `class_type` is the node-definition key, registered Functor name, or
  importable class reference resolved by the execution layer.
- `inputs` describes literal values and upstream link bindings.
- `outputs` lists the variables or output ports declared by the node.
- `config` is the node-local configuration passed to the resolved Functor.
  The executor also reads `execution_group` from this mapping;
  `validate_graph` rejects a value other than `fast`, `medium`, `slow`, or
  `None` in strict and non-strict mode, with the message `register_functor`
  raises (naming the node's `config.execution_group` as the source).

`NodeSpec` is a declarative record. Port types and units come from
`NodeDefinition` and `SSOTCatalog`, not from this class.

### `LinkEndpoint`

Canonical owner: `relic.orchestration.graph_ir.LinkEndpoint`

Constructor:

```python
LinkEndpoint(node: str, port: str)
```

- `node` is a workflow-local node identifier.
- `port` is the input or output port name on that node.

`LinkEndpoint` is not re-exported by `relic.orchestration`. Direct construction
therefore uses its canonical owner path.

### `LinkSpec`

Canonical owner: `relic.orchestration.graph_ir.LinkSpec`

Convenience import:

```python
from relic.orchestration import LinkSpec
```

Constructor:

```python
LinkSpec(
    source: LinkEndpoint,
    target: LinkEndpoint,
)
```

- `source` identifies the producing node and output port.
- `target` identifies the consuming node and input port.

The dataclass does not check endpoint existence. `validate_graph()` performs
that check against the complete graph.

## Graph serialization

Canonical owner: `relic.orchestration.graph_io`

Convenience imports:

```python
from relic.orchestration import load_graph, save_graph
```

### `load_graph`

```python
load_graph(
    path: Path,
    *,
    validate_schema: bool = False,
    schema_path: Optional[Path] = None,
) -> GraphIR
```

- `path` selects a JSON, YAML, or YML graph document.
- `validate_schema` enables JSON Schema validation before GraphIR parsing.
- `schema_path` overrides the packaged `graph_schema.json`.

Schema validation is also enabled when `RELIC_GRAPH_SCHEMA_VALIDATE` is set to
a true value. Invalid JSON, YAML, schema content, or filesystem access raises
the corresponding parser, `jsonschema`, or I/O exception. A decoded root that
is not a mapping raises `ValueError`. Loading YAML without PyYAML raises
`RuntimeError`.

### `save_graph`

```python
save_graph(graph: GraphIR, path: Path) -> None
```

The function writes the `GraphIR.to_dict()` representation as sorted,
indented JSON. Filesystem errors propagate to the caller.

## Port and node contracts

### `PortSpec`

Canonical owner: `relic.orchestration.port_spec.PortSpec`

Convenience import:

```python
from relic.orchestration import PortSpec
```

Constructor:

```python
PortSpec(
    name: str,
    standard_name: str,
    units: Optional[str] = None,
    quantity: Optional[str] = None,
    cf_standard_name: Optional[str] = None,
    element: Optional[str] = None,
    datatype: str = "tensor",
    semantics: Optional[str] = None,
    update_mode: Optional[str] = None,
    write_mode: Optional[str] = None,
    required: bool = True,
    context_requirements: Dict[str, str] = ...,
    shape: Optional[Dict[str, Any]] = None,
    direction: str = "inout",
    per_timestep: bool = False,
    conservative: bool = False,
    form_degree: Optional[int] = None,
    primality: Optional[str] = None,
    extensivity: Optional[str] = None,
    complex_id: Optional[str] = None,
    physical_domain: Optional[str] = None,
    orientation_convention: Optional[str] = None,
    boundary_role: Optional[str] = None,
    port_kind: Optional[str] = None,
    power_conjugate: Optional[str] = None,
    spatial_kind: Optional[str] = None,
    area_basis: Optional[str] = None,
    temporal_basis: Optional[str] = None,
    vertical_basis: Optional[str] = None,
    declared_support_fields: frozenset[str] = ...,
)
```

Identity and physical-semantics parameters:

- `name` is the local port or variable name.
- `standard_name` is the canonical RELIC standard name.
- `units` is the UDUNITS-compatible unit string supplied by SSOT metadata.
- `quantity` identifies the physical quantity category.
- `cf_standard_name` carries the CF-compatible standard name when declared.
- `element` identifies a conserved element such as carbon, nitrogen, or water.

Runtime and state-update parameters:

- `datatype` declares the runtime value category; the default is `tensor`.
- `semantics` distinguishes meanings such as state, flux, tendency, or
  increment.
- `update_mode` describes the requested update operation.
- `write_mode` describes ownership or mutability semantics.
- `required` marks a mandatory input or an optional output.
- `context_requirements` declares conversion context keys required by the
  receiving port.
- `shape` carries declared dimensional or shape metadata.
- `direction` declares input, output, or bidirectional use.
- `per_timestep` marks a value whose interpretation is tied to a time step.
- `conservative` identifies a port that participates in a conservation
  contract.

DEC and structural parameters:

- `form_degree` is the discrete differential-form degree.
- `primality` identifies primal or dual complex placement.
- `extensivity` identifies extensive or intensive interpretation.
- `complex_id` identifies the associated discrete complex.
- `physical_domain` names the physical domain represented by the cochain.
- `orientation_convention` declares the sign and orientation convention.
- `boundary_role` identifies an interior, boundary, source, or sink role.

IPHS port parameters:

- `port_kind` identifies the port-Hamiltonian role.
- `power_conjugate` identifies the paired effort or flow variable.

Support parameters (ordered by
`relic.orchestration.port_spec.PORT_SUPPORT_FIELDS = ("shape", "spatial_kind", "area_basis", "temporal_basis", "vertical_basis")`):

- `spatial_kind` is `leaf`, `canopy`, `grid_cell`, `footprint`,
  `tower_height`, or `vertical_column`.
- `area_basis` is `leaf_area`, `ground_area`, or `grid_area`.
  `support_aggregation` reads it as the area measure of its values and
  weights; the quantity basis is the SSOT `spatial_basis`.
- `temporal_basis` is `instantaneous`, `interval_mean`, or
  `interval_integral`.
- `vertical_basis` is `none`, `height_level`, or `layer_integral`.
- `declared_support_fields` names the support fields (`shape` and the four
  fields above) that were declared explicitly; every support field with a
  value is added at construction.

A support value outside these vocabularies, or an unknown name in
`declared_support_fields`, raises `ValueError`. `NodeRegistry` fills the
support fields from the `support` mapping of the SSOT entry; a `support` value
that is not a mapping raises `ValueError`.

Key methods:

```python
port.to_dict() -> Dict[str, Any]
port.is_support_declared(field_name: str) -> bool
port.missing_support_declarations() -> tuple[str, ...]
```

`is_support_declared()` raises `ValueError` for a name that is not a support
field. `missing_support_declarations()` returns the support fields without an
explicit declaration.

`PortSpec` is a data contract and does not validate compatibility by itself.
`can_connect()` and `validate_graph()` own compatibility decisions.

### `NodeDefinition`

Canonical owner: `relic.orchestration.port_spec.NodeDefinition`

Convenience import:

```python
from relic.orchestration import NodeDefinition
```

Constructor:

```python
NodeDefinition(
    node_type_id: str,
    inputs: Dict[str, PortSpec] = ...,
    outputs: Dict[str, PortSpec] = ...,
    metadata: Dict[str, Any] = ...,
)
```

- `node_type_id` is the key used by `NodeRegistry`.
- `inputs` maps input names to typed `PortSpec` records.
- `outputs` maps output names to typed `PortSpec` records.
- `metadata` preserves relevant Functor metadata.

```python
definition.to_dict() -> Dict[str, Any]
```

The method returns a plain serializable mapping.

## SSOT projection and node registry

### `SSOTEntry`

Canonical owner: `relic.orchestration.ssot_catalog.SSOTEntry`

Constructor:

```python
SSOTEntry(
    name: str,
    standard_name: str,
    units: Optional[str],
    spatial_basis: Optional[SpatialQuantityBasis] = None,
    quantity: Optional[str] = None,
    cf_standard_name: Optional[str] = None,
    element: Optional[str] = None,
    semantics: Optional[str] = None,
    update_mode: Optional[str] = None,
    write_mode: Optional[str] = None,
    context_requirements: Dict[str, str] = ...,
    datatype: Optional[str] = None,
    shape: Optional[Dict[str, Any]] = None,
    dimensionality: Optional[Dimensionality] = None,
    references: List[str] = ...,
    raw: Dict[str, Any] = ...,
)
```

- `name` is the catalog key.
- `standard_name` is the declared standard name, or the key when none is
  declared. Together with a `cell_methods` qualifier kept in `raw`, it forms
  the entry's CF identity.
- `units` and `dimensionality` describe the physical dimension.
- `spatial_basis` is the declared spatial representation (`intensive`,
  `cell_total`, `area_density`, or `categorical`), resolved by
  `resolve_spatial_basis` from the entry's `spatial_basis`, or else from a
  `variable_type` of `intensive` or `categorical`. It is `None` when the entry
  declares neither key; a `variable_type` of `extensive` without
  `spatial_basis`, an unknown value, or an inconsistent pair raises
  `ValueError`.
  `quantity` names the physical quantity.
- `cf_standard_name` and `element` provide standards and conservation tags.
- `semantics`, `update_mode`, and `write_mode` describe state interaction.
- `context_requirements` lists required conversion context.
- `datatype` and `shape` describe runtime representation.
- `references` preserves source references from parameter metadata.
- `raw` preserves the complete source payload, including structural fields not
  promoted to direct attributes.

`SSOTEntry` is not re-exported by `relic.orchestration`.

### `SSOTCatalog`

Canonical owner: `relic.orchestration.ssot_catalog.SSOTCatalog`

Convenience import:

```python
from relic.orchestration import SSOTCatalog
```

Constructor:

```python
SSOTCatalog() -> SSOTCatalog
```

Factory:

```python
SSOTCatalog.from_parameter_manager(manager: Any = None) -> SSOTCatalog
```

- `manager` is an optional `UnifiedParameterManager`-compatible object. When
  omitted, the canonical global manager is used.

Key members:

```python
catalog.add_entry(entry: SSOTEntry) -> None
catalog.get(name: str) -> Optional[SSOTEntry]
catalog.get_by_standard_name(
    standard_name: Optional[str],
    *,
    cell_methods: Optional[str] = None,
) -> Optional[SSOTEntry]
```

`get(name)` returns the entry of that variable name, which is the unique
catalog key. Otherwise it follows an explicit alias, from the parameter table
`coupling.external.variables.aliases` or from a variable-contract alias, and
then falls back to `get_by_standard_name(name)` without a `cell_methods`
qualifier; it returns `None` when nothing matches. `get_by_standard_name()`
treats the CF identity, `standard_name` together with `cell_methods` (CF-1.8
section 7.3), as one key: a request without `cell_methods` selects only the
entry that declares none, and `None` is returned when no entry holds the
identity.

The catalog keeps no conflict list; every conflict raises `ValueError`:

- A second declaration of an existing variable name is merged when it agrees
  and raises when it contradicts in units (dimensionless spellings count as
  equal), semantics, CF standard name, update or write mode, spatial basis,
  quantity, element, datatype, context requirements, `standard_name`, or
  `cell_methods`.
- One CF identity belongs to exactly one variable; a second variable that
  declares it raises. This rule is owned by
  `relic.data.standards.cf_conventions.register_cf_identity_holder`.
- An entry name that collides with an alias, an alias that names another
  existing variable or is claimed by two variables, an alias whose target is
  unknown, and an alias entry that is not a pair of non-empty names raise.

`from_parameter_manager()` reads
`coupling.external.variables.variable_definitions` and
`coupling.scales.defaults.variables`, both required, then merges the packaged
variable contracts read through `load_contract_registry()`, comparing units
and CF standard name, and finally registers the optional alias table. A
contract alias that names a parameter-catalog entry merges that entry into the
contract variable. The factory raises `RuntimeError` when no manager is
available, `KeyError` when a required payload is missing, `TypeError` when a
required payload or a present alias table is not a mapping, and `ValueError`
when the contract registry reports conflicts.

Ownership boundary:

`SSOTCatalog` is an orchestration projection of `UnifiedParameterManager`-backed
metadata. It does not own parameter values, bounds, units, or scientific
defaults and is not a second SSOT service.

### `NodeRegistry`

Canonical owner: `relic.orchestration.node_registry.NodeRegistry`

Convenience import:

```python
from relic.orchestration import NodeRegistry
```

Constructor:

```python
NodeRegistry(
    ssot_catalog: SSOTCatalog,
    definitions: Dict[str, NodeDefinition] = ...,
)
```

- `ssot_catalog` supplies canonical metadata for generated ports.
- `definitions` optionally seeds the `node_type_id -> NodeDefinition` mapping.

Key methods:

```python
registry.register_definition(
    node_type_id: str,
    definition: NodeDefinition,
) -> None

registry.unregister_definition(node_type_id: str) -> bool
registry.get_definition(node_type_id: str) -> Optional[NodeDefinition]

registry.register_functor_instance(
    functor: Any,
    node_type_id: Optional[str] = None,
) -> NodeDefinition

registry.remove_definitions_by_functor_names(
    functor_names: Iterable[str],
) -> List[str]

registry.build_from_node_spec(
    node_type_id: str,
    input_names: Iterable[str],
    output_names: Iterable[str],
) -> NodeDefinition

registry.resolve_class_type(class_type: str) -> Optional[type]
```

`register_functor_instance()` derives port names from
`FunctorMetadata.input_variables` and `FunctorMetadata.output_variables`. A
Functor without metadata raises `ValueError`. An input port is required exactly
when the Functor's `get_required_inputs()` names it (the `FunctorBase` rule:
declared inputs minus `optional_input_variables`); an object without that
method raises `TypeError`, and a required name that is not a declared input
raises `ValueError`. If an SSOT entry is unavailable,
the registry can still build a partial port whose units and semantic fields are
unset; graph validation determines whether that contract is acceptable.

`register_definition()` replaces an existing entry with the same key.
`resolve_class_type()` first follows the dotted `runtime_class` in the
metadata of a registered definition (a value that is not a dotted string
raises `ValueError`; one that does not name a class raises `TypeError`), then
handles `Functor:<module>.<Class>` references, and returns `None` for other
formats. Module import failures propagate.

Ownership boundary:

`NodeRegistry` owns orchestration node definitions only. It does not replace
the scientific Functor registry, `UnifiedParameterManager`, or
`ValidationRegistry`.

## Graph validation

The graph validator uses graph-scoped result types. They are distinct from the
Pydantic models in the Unified Validation Framework.

### `GraphValidationIssue`

Canonical owner: `relic.orchestration.validator.GraphValidationIssue`

Constructor:

```python
GraphValidationIssue(
    code: str,
    message: str,
    node_id: Optional[str] = None,
    port: Optional[str] = None,
    expected: Optional[str] = None,
    got: Optional[str] = None,
    hint: Optional[str] = None,
)
```

- `code` is a machine-readable graph-validation code.
- `message` is the reader-facing description.
- `node_id` and `port` locate the issue.
- `expected` and `got` describe a mismatch.
- `hint` provides a corrective action.

The module-local name `ValidationIssue` is an alias of
`GraphValidationIssue`. It is not a Unified Validation Framework type.

### `ConnectionDecision`

Canonical owner: `relic.orchestration.validator.ConnectionDecision`

Convenience import:

```python
from relic.orchestration import ConnectionDecision
```

Constructor:

```python
ConnectionDecision(
    ok: bool,
    errors: List[GraphValidationIssue] = ...,
    warnings: List[GraphValidationIssue] = ...,
    requires_adapter: bool = False,
    adapter_type: Optional[str] = None,
)
```

- `ok` reports whether the connection is acceptable.
- `errors` contains blocking incompatibilities.
- `warnings` contains non-blocking findings.
- `requires_adapter` marks a convertible but non-identical interface.
- `adapter_type` names the suggested adapter when one is known.

### Graph `ValidationReport`

Canonical owner: `relic.orchestration.validator.ValidationReport`

Convenience import:

```python
from relic.orchestration import ValidationReport
```

Constructor:

```python
ValidationReport(
    errors: List[GraphValidationIssue] = ...,
    warnings: List[GraphValidationIssue] = ...,
    assertion_audit: Dict[str, Any] = ...,
)
```

- `errors` contains blocking graph findings.
- `warnings` contains advisory findings.
- `assertion_audit` contains normalized structural-assertion evidence.

This dataclass must not be confused with
`relic.validation.ValidationReportModel`.

### `can_connect`

Canonical owner: `relic.orchestration.validator.can_connect`

Convenience import:

```python
from relic.orchestration import can_connect
```

Signature:

```python
can_connect(
    source: PortSpec,
    target: PortSpec,
    unit_converter: UnitConverter,
    context: Optional[ConversionContext] = None,
    strict: bool = True,
) -> ConnectionDecision
```

- `source` and `target` are the output and input contracts.
- `unit_converter` is the canonical conversion service used for compatibility
  checks.
- `context` supplies required conversion context such as time-step or area
  information.
- `strict` promotes optional semantic and structural mismatches to errors.

The function checks datatype, quantity, element, CF standard name, units,
state-update semantics, conversion context, DEC/IPHS structural fields, and
explicit support declarations across `PORT_SUPPORT_FIELDS` (emitting
`missing_port_support_metadata` when a support field is declared on only one
endpoint, and `<field>_mismatch` when both endpoints declare differing
values; promoted to blocking errors when `strict=True` and advisory warnings
when `strict=False`). Ordinary incompatibilities are returned inside
`ConnectionDecision` without raising exceptions. A convertible mismatch sets
`requires_adapter=True`.

### `validate_graph`

Canonical owner: `relic.orchestration.validator.validate_graph`

Convenience import:

```python
from relic.orchestration import validate_graph
```

Signature:

```python
validate_graph(
    graph: GraphIR,
    registry: NodeRegistry,
    unit_converter: Optional[UnitConverter] = None,
    strict: bool = True,
    context: Optional[ConversionContext] = None,
    ssot_catalog: Optional[SSOTCatalog] = None,
) -> ValidationReport
```

- `graph` is the GraphIR to inspect.
- `registry` supplies typed node definitions.
- `unit_converter` overrides the default converter.
- `strict` controls promotion of compatible-but-incomplete contracts.
- `context` supplies conversion context.
- `ssot_catalog` overrides the catalog used for dimensional checks.

The report covers structural assertions, missing nodes and ports, mutually
exclusive output overlaps, dimensional consistency, link compatibility, and
adapter recommendations. Validation findings are returned in the report.
`compile_graph()` is responsible for rejecting a report that contains errors.

## Adapters

Adapters represent explicit transformations. An adapter insertion recorded by
the compiler is metadata; only `compile_graph(..., auto_adapt=True)` lowers a
context-free unit conversion into an executable adapter step, and every other
adapter must be authored explicitly.

### `AdapterResult`

Canonical owner: `relic.orchestration.adapters.AdapterResult`

Constructor:

```python
AdapterResult(
    value: Any,
    metadata: Dict[str, Any],
)
```

- `value` is the transformed result.
- `metadata` records the adapter type and relevant conversion context.

### `AdapterBase`

Canonical owner: `relic.orchestration.adapters.AdapterBase`

Constructor:

```python
AdapterBase() -> AdapterBase
```

Interface:

```python
adapter.apply(value: Any, **kwargs: Any) -> AdapterResult
```

The base implementation raises `NotImplementedError`.

### `UnitCastAdapter`

Canonical owner: `relic.orchestration.adapters.UnitCastAdapter`

Convenience import:

```python
from relic.orchestration import UnitCastAdapter
```

Constructor:

```python
UnitCastAdapter(
    unit_converter: Optional[UnitConverter] = None,
)
```

- `unit_converter` overrides the canonical shared UnitConverter.

Method:

```python
adapter.apply(
    value: Any,
    variable_name: str,
    source_units: str,
    target_units: str,
    context: Optional[ConversionContext] = None,
) -> AdapterResult
```

- `value` is the value to convert.
- `variable_name` selects variable-specific conversion semantics.
- `source_units` and `target_units` identify the conversion.
- `context` supplies context-dependent conversion quantities.

Conversion and context errors from UnitConverter propagate.

### `TemporalResampleAdapter`

Canonical owner:
`relic.orchestration.adapters.TemporalResampleAdapter`

Convenience import:

```python
from relic.orchestration import TemporalResampleAdapter
```

Constructor:

```python
TemporalResampleAdapter() -> TemporalResampleAdapter
```

Method:

```python
adapter.apply(
    value: Any,
    **kwargs: Any,
) -> AdapterResult
```

- `value` is the input sequence or field.
- `kwargs["sampling_mode"]` is `window` (default) or `current_step`; any
  other value raises `ValueError`.
- `kwargs["temporal_resampler"]` is an optional callable applied to `value`
  in window mode; combining it with `current_step` raises `ValueError`.
- Without a resampler, `kwargs["method"]` must be `interval_mean` or
  `interval_integral`; otherwise `RuntimeError` is raised.

In window mode the built-in methods form the duration-weighted sum of the
samples along `kwargs["axis"]` (default `0`) with
`kwargs["sample_durations"]` (default one per sample): `interval_integral`
returns that sum and `interval_mean` divides it by the total duration. The
durations must be one-dimensional, match the axis length, be finite and
non-negative, and have a positive sum; otherwise `ValueError` is raised. In
`current_step` mode only `interval_mean` is allowed and no `axis` may be
given; `sample_durations` must be a scalar equal to the runtime
`kwargs["dt"]`, and the value is returned unchanged as a zero-order hold over
the step. A missing duration or `dt` in that mode raises `RuntimeError`.
Errors raised by a provided resampler propagate.

### `SemanticAdapter`

Canonical owner: `relic.orchestration.adapters.SemanticAdapter`

Convenience import:

```python
from relic.orchestration import SemanticAdapter
```

Constructor:

```python
SemanticAdapter() -> SemanticAdapter
```

Method:

```python
adapter.apply(
    value: Any,
    **kwargs: Any,
) -> AdapterResult
```

- `value` is a flux-like value.
- `kwargs["dt"]` is the required time-step multiplier used to form an
  increment.

Missing `dt` raises `RuntimeError`. Multiplication errors propagate.

### `AreaBasisTransformAdapter`

Canonical owner: `relic.orchestration.adapters.AreaBasisTransformAdapter`

Convenience import:

```python
from relic.orchestration import AreaBasisTransformAdapter
```

Constructor:

```python
AreaBasisTransformAdapter() -> AreaBasisTransformAdapter
```

Method:

```python
adapter.apply(
    value: Any,
    **kwargs: Any,
) -> AdapterResult
```

- `kwargs["source_area_basis"]` and `kwargs["target_area_basis"]` must be
  distinct members of `{"leaf_area", "ground_area"}` (`ValueError`
  otherwise).
- `kwargs["leaf_area_index"]` supplies finite, non-negative leaf area per
  ground area (`RuntimeError` when omitted; strictly positive for
  `ground_area` to `leaf_area` division).

### `SurfaceFluxToVolumeSourceAdapter`

Canonical owner:
`relic.orchestration.adapters.SurfaceFluxToVolumeSourceAdapter`

Convenience import:

```python
from relic.orchestration import SurfaceFluxToVolumeSourceAdapter
```

Constructor:

```python
SurfaceFluxToVolumeSourceAdapter() -> SurfaceFluxToVolumeSourceAdapter
```

Method:

```python
adapter.apply(
    value: Any,
    **kwargs: Any,
) -> AdapterResult
```

- Converts a non-negative surface molar flux (`mol m-2 s-1`) into a
  volumetric mass source (`kg m-3 s-1`) injected strictly into layer index
  `0` along `kwargs.get("vertical_axis", -1)`, with zero source in upper
  layers.
- Requires positive `cell_area` (`m2`), positive `layer_volumes` (`m3`), and
  `molar_mass_g_per_mol_constant_key` resolved through
  `get_physical_constant(...)`.

### `SupportAggregationAdapter`

Canonical owner: `relic.orchestration.adapters.SupportAggregationAdapter`

Convenience import:

```python
from relic.orchestration import SupportAggregationAdapter
```

Constructor:

```python
SupportAggregationAdapter() -> SupportAggregationAdapter
```

Method:

```python
adapter.apply(
    value: Any,
    **kwargs: Any,
) -> AdapterResult
```

- Reduces a field to an explicit spatial support using `aggregate_spatial_alias`
  governed by `quantity_basis` (from SSOT `spatial_basis`), bound
  `source_port_spec` and `target_port_spec` (matching `area_basis`), and
  `conservation_tolerance` (`coupling.scales.defaults.conservation_tolerance`).
- Accepts optional `weights`, `strategy` (`ScalingStrategy`), and
  `enforce_conservation` (`bool`, default `False`).

### `ExecutableAdapterFunctor`

Canonical owner: `relic.orchestration.adapters.ExecutableAdapterFunctor`

Convenience import:

```python
from relic.orchestration import ExecutableAdapterFunctor
```

Constructor:

```python
ExecutableAdapterFunctor(
    name: Optional[str] = None,
    adapter_type: str = "",
    source_node: str = "",
    source_port: str = "",
    target_port: str = "",
    config: Optional[Dict[str, Any]] = None,
)
```

- Inherits `ScaleCouplingFunctor` (`ProcessCategory.SCALE_COUPLING`,
  `ExecutionPhase.COUPLING`) and executes one registered `AdapterRegistry`
  entry inside the governed Functor lifecycle.
- `source_port` and `target_port` must be non-empty and distinct to preserve
  single-writer provenance.
- `config` accepts `source_mode` (`"subprocess_outputs"` default, or
  `"state"`) and state-variable bindings `sample_durations_state`,
  `leaf_area_index_state`, `weights_state`, `cell_area_state`, and
  `layer_volumes_state`. Duplicating `dt` or the governed support inputs in
  `config` raises `ValueError`.
- `bind_node_definition(definition)` binds the authoritative input/output
  `PortSpec` contracts and `support_quantity_basis`; `last_adapter_metadata`
  records the latest `AdapterResult.metadata`.

### `AdapterRegistry`

Canonical owner: `relic.orchestration.adapters.AdapterRegistry`

Convenience import:

```python
from relic.orchestration import AdapterRegistry
```

Constructor:

```python
AdapterRegistry(
    unit_converter: Optional[UnitConverter] = None,
)
```

- `unit_converter` overrides the canonical converter used by the built-in unit
  adapter and compatibility suggestions.

Key methods:

```python
registry.get(adapter_type: str) -> Optional[AdapterBase]

registry.suggest_unit_adapter(
    source: PortSpec,
    target: PortSpec,
) -> Optional[str]

registry.register_executable_node(
    *,
    node_registry: NodeRegistry,
    node_type_id: str,
    functor: ExecutableAdapterFunctor,
    source_port: PortSpec,
    target_port: PortSpec,
    context_ports: Optional[Dict[str, PortSpec]] = None,
) -> NodeDefinition
```

The built-in adapter keys are `unit_cast` (`UnitCastAdapter`),
`temporal_resample` (`TemporalResampleAdapter`), `semantic_adapter` and its
synonym `flux_to_increment` (one shared `SemanticAdapter`),
`area_basis_transform` (leaf-area and ground-area flux densities converted
through the leaf area index), `surface_flux_to_volume_source` (an areal molar
flux injected as a mass source into the lowest volume layer only), and
`support_aggregation` (weighted aggregation to an explicit spatial support:
the quantity basis is the SSOT `spatial_basis` of the port variable, carried
by `NodeDefinition.metadata["support_quantity_basis"]`;
`ExecutableAdapterFunctor` supplies that basis, the bound PortSpecs and
`coupling.scales.defaults.conservation_tolerance`, which configuration must
not declare; the ledger is reported in `last_adapter_metadata`, and
`enforce_conservation: true` makes a ledger residual raise when it exceeds
both the relative tolerance times the source amount and the floating-point
rounding bound of the reduction).
`AdapterRegistry` registers six concrete adapter classes across seven keys:
`unit_cast`, `temporal_resample`, `semantic_adapter`, `flux_to_increment`
(shared `SemanticAdapter` instance), `area_basis_transform`,
`surface_flux_to_volume_source`, and `support_aggregation`.
`register_executable_node(...)` validates unit, shape, context-port, and
SSOT `spatial_basis` contracts before registering an
`ExecutableAdapterFunctor` definition in `NodeRegistry`.
`get()` returns `None` for an unknown key.
`suggest_unit_adapter()` returns `unit_cast` only when the declared units
differ and the canonical converter reports a supported conversion.

```python
from relic.orchestration import AdapterRegistry

adapter_reg = AdapterRegistry()
expected_keys = (
    "unit_cast",
    "temporal_resample",
    "semantic_adapter",
    "flux_to_increment",
    "area_basis_transform",
    "surface_flux_to_volume_source",
    "support_aggregation",
)
assert all(adapter_reg.get(k) is not None for k in expected_keys)
assert adapter_reg.get("semantic_adapter") is adapter_reg.get("flux_to_increment")
```

## Compilation

### `AdapterInsertion`

Canonical owner: `relic.orchestration.compiler.AdapterInsertion`

Constructor:

```python
AdapterInsertion(
    adapter_type: str,
    source_node: str,
    source_port: str,
    target_node: str,
    target_port: str,
    details: Dict[str, Any] = ...,
)
```

- `adapter_type` names the suggested transformation.
- `source_node` and `source_port` locate the producer.
- `target_node` and `target_port` locate the consumer.
- `details` records unit and semantic mismatch context.

An insertion is metadata. With `auto_adapt=True`, an unambiguous context-free
unit conversion is lowered from it into an executable `ExecutableAdapterFunctor`
step; other adapter types are rejected and must be authored explicitly.

### `ExecutionStep`

Canonical owner: `relic.orchestration.compiler.ExecutionStep`

Constructor:

```python
ExecutionStep(
    node_id: str,
    class_type: str,
    inputs: Dict[str, Any],
    outputs: List[str],
    config: Dict[str, Any],
)
```

- `node_id` is the workflow-local instance name.
- `class_type` is resolved to the concrete Functor class.
- `inputs`, `outputs`, and `config` are copied from the compiled node.

Steps are emitted in deterministic topological order, with ties broken by
node identifier. This is the registration order; at run time the simulator
places each step in its rate group and re-derives that group's order from the
dependencies, using registration order only as a tie-break.

### `CompileReport`

Canonical owner: `relic.orchestration.compiler.CompileReport`

Convenience import:

```python
from relic.orchestration import CompileReport
```

Constructor:

```python
CompileReport(
    validation: ValidationReport,
    adapter_insertions: List[AdapterInsertion] = ...,
    metadata: Dict[str, Any] = ...,
)
```

- `validation` is the graph-scoped validation report.
- `adapter_insertions` records recommended explicit transformations.
- `metadata` contains schema and graph metadata plus the normalized global
  structural audit. A topology signature is included when requested.

### `ExecutionPlan`

Canonical owner: `relic.orchestration.compiler.ExecutionPlan`

Convenience import:

```python
from relic.orchestration import ExecutionPlan
```

Constructor:

```python
ExecutionPlan(
    steps: List[ExecutionStep],
    report: CompileReport,
)
```

- `steps` is the ordered node execution sequence.
- `report` contains validation, structural-audit, and adapter evidence.

### `compile_graph`

Canonical owner: `relic.orchestration.compiler.compile_graph`

Convenience import:

```python
from relic.orchestration import compile_graph
```

Signature:

```python
compile_graph(
    graph: GraphIR,
    registry: NodeRegistry,
    adapter_registry: Optional[AdapterRegistry] = None,
    auto_adapt: bool = False,
    strict: bool = True,
    context: Optional[ConversionContext] = None,
    optimization_passes: Optional[List[OptimizationPass]] = None,
    enable_optimization: bool = False,
    include_topology: bool = False,
) -> ExecutionPlan
```

- `graph` is the GraphIR to compile.
- `registry` supplies node and port definitions.
- `adapter_registry` supplies conversion compatibility checks.
- `auto_adapt` records adapter insertions in `CompileReport` and applies
  triple-guard automatic lowering only when (1) `can_connect` selects a
  single unambiguous `unit_cast` without `multiple_adapters_required`
  (otherwise `ambiguous_adapter` or `unsupported_auto_adapter` is recorded),
  (2) `UnitConverter` executes the conversion with `context=None`, and (3)
  both endpoint nodes declare the same explicit `execution_group`. Qualifying
  edges are lowered into `ExecutableAdapterFunctor` nodes and revalidated;
  any other adapter requirement raises `ValueError`.
- `strict` enables strict graph and structural-audit rejection, rejects an
  unlowered executable-adapter requirement, and produces the compile-run
  witness (class resolution, construction, and registration without a step)
  that `GraphExecutor` requires before execution.
- `context` supplies conversion context.
- `optimization_passes` overrides the ordered optimization pass list.
- `enable_optimization` enables optimization before final planning.
- `include_topology` adds the topology signature to compile metadata.

Failure semantics:

- Graph-validation errors raise `ValueError`.
- Validation errors introduced or exposed by optimization raise `ValueError`.
- A cycle or unresolved topological dependency raises `ValueError`.
- A failed enabled structural audit in strict mode raises `ValueError`.

### `RELICModelCompiler`

Canonical owner: `relic.orchestration.compiler.RELICModelCompiler`

Convenience import:

```python
from relic.orchestration import RELICModelCompiler
```

Constructor:

```python
RELICModelCompiler(
    registry: NodeRegistry,
    adapter_registry: Optional[AdapterRegistry] = None,
)
```

- `registry` is the node-definition registry passed to every compilation.
- `adapter_registry` optionally fixes the adapter registry used by the facade.

Method:

```python
compiler.compile(
    graph: GraphIR,
    **compile_kwargs: Any,
) -> ExecutionPlan
```

`compile_kwargs` are forwarded unchanged to `compile_graph()`.

Ownership boundary:

`RELICModelCompiler` is a thin facade. `compile_graph()` is the canonical graph
compiler and structural-audit path. The facade does not own a second compiler,
validator, registry, or execution model.

### `compile_and_execute`

Canonical owner: `relic.orchestration.compiler.compile_and_execute`

Convenience import:

```python
from relic.orchestration import compile_and_execute
```

Signature:

```python
compile_and_execute(
    graph: GraphIR,
    registry: NodeRegistry,
    initial_state: Dict[str, Any],
    n_steps: int,
    dt: float,
    output_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    cache: Optional[Any] = None,
    simulator_config: Optional[Mapping[str, Any]] = None,
    **compile_kwargs: Any,
) -> Dict[str, Any]
```

- `graph`, `registry`, and `compile_kwargs` configure `compile_graph()`.
- `initial_state` initializes the canonical StateManager.
- `n_steps` is the number of simulator steps.
- `dt` is the simulator time step in seconds. Both are required: there is no
  default number of steps and no default time step.
- `output_dir` enables provenance output.
- `cache_dir` creates a `NodeCache` when `cache` is not supplied.
- `cache` injects an existing cache object and takes precedence over
  `cache_dir`.
- `simulator_config` is passed to `GraphExecutor`, which copies it into each
  fresh `RELICSimulator`; a non-mapping value raises `TypeError`.

Compilation and execution errors propagate. A non-mapping return from
`GraphExecutor.execute()` raises `TypeError`.

## Execution and cache

### `ExecutionStrategy`

Canonical owner: `relic.orchestration.executor.ExecutionStrategy`

Members:

- `ExecutionStrategy.PUSH` evaluates every node each time its rate group
  executes.
- `ExecutionStrategy.DIRTY` can reuse the previous output when the tracked
  inputs are unchanged.

The enum is not re-exported by `relic.orchestration`.

### `GraphExecutor`

Canonical owner: `relic.orchestration.executor.GraphExecutor`

Convenience import:

```python
from relic.orchestration import GraphExecutor
```

Constructor:

```python
GraphExecutor(
    plan: Any,
    state_manager: Optional[StateManager] = None,
    cache: Optional[NodeCache] = None,
    output_dir: Optional[Path] = None,
    registry: Optional[NodeRegistry] = None,
    strategy: ExecutionStrategy = ExecutionStrategy.PUSH,
    dirty_include_time: bool = True,
    simulator_config: Optional[Mapping[str, Any]] = None,
)
```

- `plan` is expected to expose the `steps` field of an `ExecutionPlan`.
- `state_manager` injects the canonical StateManager; a new one is created
  when omitted.
- `cache` enables node-output caching.
- `output_dir` enables provenance recording at that location.
- `registry` assists class-type resolution.
- `strategy` selects push or dirty evaluation.
- `dirty_include_time` controls whether temporal context participates in dirty
  input hashes.
- `simulator_config` is copied into each fresh `RELICSimulator`; a non-mapping
  value raises `TypeError`.

Key methods:

```python
executor.execute(
    initial_state: Dict[str, Tensor],
    n_steps: int,
    dt: float,
) -> Dict[str, Tensor]

executor.create_checkpoint(
    checkpoint_path: str | Path,
) -> Dict[str, Any]

executor.execute_from_checkpoint(
    initial_state: Dict[str, Tensor],
    checkpoint_path: str | Path,
    *,
    n_steps: int,
    dt: float,
    compatibility_certificate: Any = None,
) -> Dict[str, Tensor]

executor.migrate_from_checkpoint(
    initial_state: Dict[str, Tensor],
    checkpoint_path: str | Path,
    migration_certificate: Any,
    *,
    n_steps: int,
    dt: float,
) -> Dict[str, Tensor]

executor.execute_forcing_sequence(
    initial_state: Dict[str, Tensor],
    forcing_sequence: Sequence[Mapping[str, Tensor]],
    *,
    dt: float,
) -> Dict[str, Tensor]

executor.execute_forcing_sequence_from_checkpoint(
    initial_state: Dict[str, Tensor],
    checkpoint_path: str | Path,
    forcing_sequence: Sequence[Mapping[str, Tensor]],
    *,
    dt: float,
) -> Dict[str, Tensor]

executor.preflight() -> Dict[str, Any]
executor.invalidate_node_closure(node_id: str) -> Tuple[str, ...]
executor.mark_dirty(node_id: str) -> None
executor.set_strategy(strategy: ExecutionStrategy) -> None
```

`execute()` first verifies the plan seal and the strict compile-run witness
(a plan compiled without `strict=True` is refused), builds the planned
Functors, and confirms the previous run's accepted external deliveries. It
then resets and initializes the StateManager, replaces the simulator with a
fresh canonical `RELICSimulator`, creates a `DependencyGraphResolver`, binds
each Functor's call governor (cache, RNG audit, lineage), registers each
Functor with the `execution_group` from its step configuration, calls
`step(dt)` `n_steps` times, and returns `StateManager.get_state()`.

`create_checkpoint(checkpoint_path)` requires a prepared execution and
delegates directly to the canonical simulator checkpoint API.
`execute_from_checkpoint(...)` rebuilds the same declared graph lifecycle from
`initial_state`, delegates strict restore to the simulator, advances the
requested fixed steps, and returns the managed state. Without
`compatibility_certificate` the restore is exact; with a
`RestartCompatibilityCertificate` it is the certified compatible restart.
`migrate_from_checkpoint(...)` prepares the target plan and restores a
checkpoint of another graph under a `GraphMigrationCertificate`.
`execute_forcing_sequence(...)` binds the simulator forcing cursor to the
digest of the normalized sequence and applies one snapshot per accepted step;
`execute_forcing_sequence_from_checkpoint(...)` restores a mid-sequence
checkpoint whose cursor names the same sequence and is not exhausted, then
consumes the remaining snapshots. `preflight()` rebuilds the compile-run
witness without executing a step. `invalidate_node_closure(node_id)` drops the
cache entries and dirty-tracking data of the node and of every transitive
consumer, and returns the affected node identifiers. `GraphExecutor` owns no
checkpoint format, serializer, restart manager, or cleanup path.

A short `class_type` (no `:` and no `.`) that the node registry does not
resolve is looked up with `relic.scientific.get_functor_path`, as given, then
lowercased, then in snake case; an unregistered name raises `KeyError`. A
`Functor:<path>` whose path contains no `.` raises `ValueError`. Import
errors of a dotted or `module:Class` path, such as `ModuleNotFoundError` or
`AttributeError`, propagate, as do Functor construction, registration, and
simulator-step failures.
For both ordinary and restored execution, `n_steps` and `dt` are required,
`n_steps` must be a nonnegative integer, and `dt` must be a finite positive
scalar; there is no default time step. Thus `n_steps=0` is valid
for restore-only verification, while booleans, negative counts, non-finite
values, and non-positive `dt` are rejected.

Ownership boundary:

`GraphExecutor` bridges an `ExecutionPlan` to `RELICSimulator` and
`StateManager`. It is not a second simulator and does not replace the
FunctorBase lifecycle. Its checkpoint methods preserve the same sole-owner
boundary: `relic.core.simulator.checkpoint` owns the strict versioned,
`weights_only=True` file format, scientific/topology/device/distributed
signatures, RNG symmetry, atomic write, rollback, and receipt.

### `NodeCache`

Canonical owner: `relic.orchestration.cache.NodeCache`

Convenience import:

```python
from relic.orchestration import NodeCache
```

Constructor:

```python
NodeCache(cache_dir: Path)
```

- `cache_dir` is created when absent and stores serialized node outputs.

Key methods:

```python
NodeCache.hash_object(obj: Any) -> str

cache.compute_key(
    node_id: str,
    inputs_hash: str,
    params_hash: str,
    code_hash: str,
) -> str

cache.get(key: str) -> Optional[Any]
cache.put(key: str, value: Any) -> None
cache.has(key: str) -> bool
cache.invalidate(node_id: str) -> int
cache.clear_all() -> int
```

Keys bind node identity, input state, parameters, and implementation code.
`invalidate()` and `clear_all()` return the number of removed entries.
Invalid key syntax raises `ValueError`. Reads validate the entry envelope,
key, payload length, and digest before deserialization. A damaged or unreadable
entry emits a warning, attempts eviction, and becomes a cache miss; failed
eviction emits a further warning. Write serialization and filesystem errors
propagate. The cache is for trusted local serialized outputs; its content
digest detects corruption and does not authenticate arbitrary pickle files.

Caching is off unless a node's `config.cache_policy` is `deterministic_pure`
and its Functor declares the same `cache_classification`; any other policy
value, a missing matching declaration, or replay-critical instance state on
the Functor raises `ValueError`. An eligible node is still executed without
caching when it has PyTorch hooks, declares parameters or holds a parameter
manager, has a non-empty `state_dict()`, has trainable parameters, or reads
tensors with `requires_grad` or `grad_fn`. Outputs that carry an
`ExternalIntent` or an autograd graph are neither stored nor replayed, and a
node declared `deterministic_pure` that consumes runtime RNG raises
`RuntimeError`.

`GraphExecutor` also binds time, time step, keyword inputs, execution contract,
and provider identity to the cache key. A provider epoch change triggers plan
reconciliation; the epoch alone is not a semantic cache identity. Reuse still
requires matching declared dependencies, parameters, implementation, and
provider binding.

## SCM variable contracts

### `VariableContract`

Canonical owner: `relic.contracts.contract_loader.VariableContract`

Convenience import:

```python
from relic.contracts import VariableContract
```

Constructor:

```python
VariableContract(
    variable: str,
    kind: str,
    udunits: str,
    cf_standard_name: Optional[str] = None,
    owner: Optional[str] = None,
    aliases: List[str] = ...,
    incrementable: bool = True,
    accumulates: Optional[str] = None,
    flux_in: List[str] = ...,
    flux_out: List[str] = ...,
    writers: List[str] = ...,
    increment_writers: List[str] = ...,
    tendency_writers: List[str] = ...,
    atol: float = 1e-12,
    rtol: float = 1e-9,
    layer_thickness_variable: Optional[str] = None,
    raw: Dict[str, Any] = ...,
)
```

Identity and ownership parameters:

- `variable` is the canonical state variable name.
- `kind` classifies state, diagnostic, parameter, or another declared kind.
- `udunits` is the canonical unit string.
- `cf_standard_name` stores the CF-compatible name.
- `owner` identifies the sole owner when the contract declares one.
- `aliases` lists accepted input aliases.

Write and conservation parameters:

- `incrementable` permits accumulated multi-writer updates when true.
- `accumulates` identifies the conservation domain.
- `flux_in` and `flux_out` define the allowed conservation-ledger fluxes.
- `writers` lists all authorized writers.
- `increment_writers` lists authorized accumulated-increment writers.
- `tendency_writers` lists authorized tendency writers.

Numerical, layer, and provenance parameters:

- `atol` and `rtol` are contract-level numerical comparison tolerances.
- `layer_thickness_variable` names the state variable [m] that holds the
  thickness of the layer this variable describes, or is `None` when the
  contract declares no layer. The registry reads it from the top-level
  contract key `layer_thickness_variable`; a value other than `None` must be
  a non-empty string, otherwise loading the contract raises `ValueError`.
  The packaged `soil_moisture` contract declares
  `layer_thickness_variable: soil_depth`. `TransportMechanismFunctor`
  converts each mapped flux from its declared unit to the per-second rate of
  the target's contract `udunits`. A content-rate conversion between
  `m3 m-3 s-1` and `kg m-2 s-1` uses the `layer_thickness_variable` of the
  contract of the volumetric side (the target when its contract unit is
  volumetric, otherwise the flux) and raises `ValueError` when that contract
  declares none.
  `UnitConverter.get_conversion_context(state, layer_thickness_variable=...)`
  reads the thickness pointwise from the state and raises `ValueError` when
  that state field is absent. The declared flux unit is read from the state's
  `units`, `output_units`, or `flux_units` mapping; an undeclared flux unit
  (including `state=None`), a target without a contract, or an unconvertible
  pair raises `ValueError`. Depth-rate water fluxes use the UnitConverter
  liquid-water-equivalent rule with the SSOT constant `water.density_liquid`;
  no state density enters it.
- `raw` preserves the complete YAML payload.

Normal application code obtains these values from YAML contracts. The Python
defaults describe the dataclass shape and do not replace the contract files as
the source of scientific metadata.

### `ContractRegistry`

Canonical owner: `relic.contracts.contract_loader.ContractRegistry`

Convenience import:

```python
from relic.contracts import ContractRegistry
```

Constructor:

```python
ContractRegistry(
    schema_path: Optional[Path] = None,
    strict: bool = False,
)
```

- `schema_path` overrides the packaged SCM variable schema.
- `strict` converts schema, parsing, and duplicate-contract findings into
  exceptions where supported.

The constructor also loads the conservation-domain ledger.

Packaged contracts are read only through
`load_contract_registry(contracts_dir: Optional[Path] = None) -> ContractRegistry`
(`from relic.contracts import load_contract_registry, PACKAGED_CONTRACTS_DIR`).
It loads one directory strictly, by default `PACKAGED_CONTRACTS_DIR`, and
returns the same registry for that resolved directory for the rest of the
process; a failed load is not cached. Its consumers, including StateManager,
the transport bridge, `UnitSystem`, and `SSOTCatalog`, share that registry
and treat it as read-only.

Loading and lookup methods:

```python
registry.load_contracts_dir(contracts_dir: Path) -> None
registry.load_contract_file(path: Path) -> None
registry.load_domain_ledger(force_reload: bool = False) -> None

registry.get_owner(variable: str) -> Optional[str]
registry.get_contract(variable: str) -> Optional[VariableContract]
registry.is_alias(name: str) -> bool
registry.resolve_alias(name: str) -> str
registry.list_variables() -> List[str]
registry.list_domains() -> List[str]
registry.conflicts -> List[str]
```

Write and conservation methods:

```python
registry.can_write(variable: str, functor_name: str) -> bool
registry.is_incrementable(variable: str) -> bool
registry.can_increment(variable: str, functor_name: str) -> bool
registry.get_flux_whitelist(variable: str) -> Dict[str, List[str]]
registry.get_domain_members(domain: str) -> List[str]
registry.to_state_variable_spec(var_name: str) -> Optional[StateVariableSpec]
```

Failure and compatibility semantics:

- `load_contracts_dir()` raises `FileNotFoundError` for a missing directory.
- In strict mode, an unavailable schema, invalid contract document, schema
  violation, or duplicate variable can raise its native exception or
  `ValueError`.
- In non-strict mode, recoverable load failures are logged. The first duplicate
  variable is retained and the conflict is recorded.
- A declared contract with no owner or writers describes externally supplied
  data, not an unrestricted scientific output.
- An undeclared variable remains writable and incrementable for compatibility
  with transient diagnostics.

Ownership boundary:

`ContractRegistry` owns SCM variable ownership, writer permissions, flux
whitelists, and conservation-domain membership. It is distinct from
`SSOTCatalog` and `ValidationRegistry`.

## Unified Validation Framework

The Unified Validation Framework owns L1 through L4 rule registration,
execution, result aggregation, and reporting. Its models are independent of
the graph-scoped `ValidationReport`.

Recommended imports:

```python
from relic.validation import (
    ValidationCode,
    ValidationContext,
    ValidationEngine,
    ValidationLevel,
    ValidationRegistry,
    ValidationReportModel,
    ValidationReporter,
    ValidationResultModel,
)

from relic.validation.framework import RuleMetadata, ValidationDetail
```

### `ValidationLevel`

Canonical owner: `relic.validation.framework.models.ValidationLevel`

Members:

- `INFO` reports successful or informational evidence.
- `WARNING` reports a non-blocking concern.
- `ERROR` reports a failed validation.
- `CRITICAL` reports a failure that can terminate fail-fast execution.

### `ValidationCode`

Canonical owner: `relic.validation.framework.models.ValidationCode`

`ValidationCode` is the single enum for standardized validation outcomes.
Members are grouped by responsibility:

- CF and units:
  `CF_COMPLIANCE_OK`, `CF_UNIT_OK`, `CF_NAME_OK`,
  `CF_STANDARD_NAME_OK`, `CF_UNIT_CONVERSION_OK`,
  `CF_UNIT_MISMATCH`, `CF_NAME_MISSING`,
  `CF_STANDARD_NAME_INVALID`, and `CF_UNIT_CONVERSION_FAILED`.
- Conservation:
  `MASS_BALANCE_OK_CARBON`, `MASS_BALANCE_OK_NITROGEN`,
  `MASS_BALANCE_OK_WATER`, `MASS_BALANCE_OK`, `ENERGY_BALANCE_OK`,
  `MASS_BALANCE_FAIL_CARBON`, `MASS_BALANCE_FAIL_NITROGEN`,
  `MASS_BALANCE_FAIL_WATER`, `ENERGY_BALANCE_FAIL`,
  `MASS_BALANCE_WARN_CARBON`, `MASS_BALANCE_WARN_NITROGEN`,
  `MASS_BALANCE_WARN_WATER`, and `ENERGY_BALANCE_WARN`.
- Probabilistic scoring:
  `SCORE_CRPS_OK`, `PIT_UNIFORMITY_OK`, `BRIER_SCORE_OK`,
  `COVERAGE_PROBABILITY_OK`, `SCORE_CRPS_HIGH`,
  `SCORE_CRPS_CRITICAL`, `PIT_UNIFORMITY_REJECT_KS`,
  `PIT_UNIFORMITY_REJECT_CVM`, `PIT_UNIFORMITY_REJECT_CHI2`,
  `BRIER_SCORE_HIGH`, and `COVERAGE_PROBABILITY_FAIL`.
- Parameters:
  `PARAM_BOUNDS_OK`, `PARAM_SCHEMA_OK`, `PARAM_TYPE_OK`,
  `PARAM_CORRELATION_OK`, `PARAM_LOG_SCALE_OK`, `PARAM_OK`,
  `PARAM_BOUND_VIOLATION`, `PARAM_SCHEMA_MISSING`,
  `PARAM_TYPE_MISMATCH`, `PARAM_CORRELATION_VIOLATION`, and
  `PARAM_LOG_SCALE_INCONSISTENT`.
- Coupling and interfaces:
  `COUPLING_CONTRACT_OK`, `INTERFACE_VARIABLE_OK`,
  `INTERFACE_TYPE_OK`, `STRESS_FACTOR_OK`,
  `COUPLING_CONTRACT_VIOLATION`, `INTERFACE_VARIABLE_MISSING`,
  `INTERFACE_TYPE_MISMATCH`, and `STRESS_FACTOR_OUT_OF_BOUNDS`.
- Generic execution:
  `GENERIC_VALIDATION_OK`, `NUMERICAL_STABILITY_OK`,
  `GENERIC_VALIDATION_FAILED`, `MISSING_REQUIRED_DATA`, and
  `NUMERICAL_INSTABILITY`.

Consumers should reference `ValidationCode` enum members directly and avoid maintaining a second code list.

### `ValidationDetail`

Canonical owner: `relic.validation.framework.models.ValidationDetail`

Constructor:

```python
ValidationDetail(
    *,
    variable_name: Optional[str] = None,
    cf_standard_name: Optional[str] = None,
    units: Optional[str] = None,
    value: Optional[Union[float, int, str]] = None,
    expected: Optional[Union[float, int, str, tuple]] = None,
    threshold: Optional[Union[float, int]] = None,
    location: Optional[Dict[str, Any]] = ...,
    statistics: Optional[Dict[str, Any]] = ...,
    metadata: Optional[Dict[str, Any]] = ...,
)
```

- `variable_name`, `cf_standard_name`, and `units` identify the quantity.
- `value`, `expected`, and `threshold` describe the comparison.
- `location` identifies spatial or temporal indices.
- `statistics` stores quantitative diagnostics.
- `metadata` stores additional structured evidence.

Pydantic validates supplied field types. Arbitrary scientific object types are
allowed inside supported metadata containers.

### `ValidationResultModel`

Canonical owner: `relic.validation.framework.models.ValidationResultModel`

Constructor:

```python
ValidationResultModel(
    *,
    level: ValidationLevel,
    code: ValidationCode,
    message: str,
    passed: bool = True,
    details: List[ValidationDetail] = ...,
    layer: str,
    scope: str,
    tags: List[str] = ...,
    timestamp: datetime = ...,
    metrics: Dict[str, Union[float, int]] = ...,
)
```

- `level` is the severity.
- `code` is the standardized outcome code.
- `message` is the reader-facing result.
- `passed` is the explicit pass/fail flag.
- `details` carries variable-level evidence.
- `layer` identifies L1, L2, L3, or L4.
- `scope` identifies the module, Functor, or rule scope.
- `tags` supports filtering and aggregation.
- `timestamp` records result creation time.
- `metrics` stores scalar quantitative results.

Key methods:

```python
result.is_passing() -> bool
result.is_failing() -> bool

result.add_detail(
    variable_name: str,
    value: Union[float, str],
    **kwargs: Any,
) -> None
```

Tags are normalized to lowercase. `scope` must be a non-empty string.
Success codes containing `_OK` force `passed=True`; `ERROR` and `CRITICAL`
force `passed=False`. Invalid Pydantic fields raise `ValidationError`.

### `ValidationReportModel`

Canonical owner: `relic.validation.framework.models.ValidationReportModel`

Constructor:

```python
ValidationReportModel(
    *,
    results: List[ValidationResultModel] = ...,
    summary: Dict[str, Dict[str, int]] = ...,
    cf_summary: Dict[str, Any] = ...,
    conservation_summary: Dict[str, Any] = ...,
    execution_metadata: Dict[str, Any] = ...,
    timestamp: datetime = ...,
)
```

- `results` contains the individual rule results.
- `summary` aggregates counts by level and layer.
- `cf_summary` aggregates CF-tagged results.
- `conservation_summary` aggregates conservation-tagged results.
- `execution_metadata` stores timing and rule-count information.
- `timestamp` records report creation time.

Key methods:

```python
report.add_result(result: ValidationResultModel) -> None
report.get_results_by_level(level: ValidationLevel) -> List[ValidationResultModel]
report.get_results_by_layer(layer: str) -> List[ValidationResultModel]
report.get_results_by_tag(tag: str) -> List[ValidationResultModel]
report.has_errors() -> bool
report.has_warnings() -> bool
report.to_dict() -> Dict[str, Any]
```

The report recalculates summary, CF, and conservation aggregates from
`results`.

### `ValidationContext`

Canonical owner: `relic.validation.framework.models.ValidationContext`

Constructor:

```python
ValidationContext(
    state_manager: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    execution_layer: Optional[str] = None,
    domain_filter: Optional[List[str]] = None,
    tolerance_overrides: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output: Optional[Dict[str, Any]] = None,
    functor_name: Optional[str] = None,
    functor_metadata: Optional[Dict[str, Any]] = None,
    time: Optional[float] = None,
    dt: Optional[float] = None,
)
```

- `state_manager` exposes the canonical state owner when a rule requires it.
- `config` carries validation configuration.
- `execution_layer` is set by the engine for the current layer.
- `domain_filter` restricts scientific domains.
- `tolerance_overrides` provides explicit numerical tolerances to the rules that
  accept them. The stochastic score rules read each threshold from the
  override key given first below, and otherwise from the dimensionless
  parameter `validation.stochastic_scores.<key>` named in parentheses (a missing
  parameter raises): `crps_warning` (`crps_warning_threshold`), `crps_critical`
  (`crps_critical_threshold`), `alpha` (`pit_significance_level`),
  `brier_warning` (`brier_warning_threshold`), `brier_critical`
  (`brier_critical_threshold`), `brier_reliability_warning`
  (`brier_reliability_warning_threshold`), `coverage_tolerance`
  (`coverage_deviation_tolerance`), `coverage_error_alpha`
  (`coverage_error_significance_level`), `coverage_warning_alpha`
  (`coverage_warning_significance_level`), `interval_width_narrow`
  (`interval_width_narrow_threshold`), and `interval_width_wide`
  (`interval_width_wide_threshold`). The Brier `threshold` and the
  `nominal_coverage` are read from the context metadata, or else from
  `brier_event_threshold` and `nominal_interval_coverage` in the same
  namespace. The conservation budget rules (water, carbon, energy, nitrogen) refuse
  an override of their budget tolerance: each compares the relative residual
  `max_i |r_i| / sum_k |t_k,i|` of its budget with its one key,
  `validation.conservation.<budget>_conservation_tolerance`. A budget with a
  missing term is not evaluable: the rule returns one `ERROR` result with
  `MISSING_REQUIRED_DATA`, `passed=False`, no residual, and the missing terms
  named; a missing term is never taken as zero, so a ground heat flux of zero
  (for example FAO-56 Eq. 42 for day-long steps) is supplied explicitly. The
  carbon rule evaluates each identity whose defining term is supplied
  (`net_primary_productivity`, `net_ecosystem_exchange`, `storage_change`). The
  nitrogen rule closes the total soil mineral nitrogen (NH4 plus NO3): its terms are
  deposition, fixation, fertilizer and net mineralization against denitrification,
  leaching, volatilization, plant uptake and the storage change; nitrification is an
  internal transfer and is not a term. When the context carries no flux dictionary of
  a budget at all, the rule returns one `WARNING` result whose message and detail read
  `not applicable (domain absent)`: no check was performed. The plausibility warnings
  of the carbon and energy rules read their limits from
  `validation.conservation.plausibility.*`.
- `metadata` carries execution context and control flags.
- `state`, `parameters`, and `output` inject rule inputs.
- `functor_name` and `functor_metadata` identify a Functor scope.
- `time` and `dt` provide temporal context.

This dataclass does not validate scientific state by itself.

### `RuleMetadata`

Canonical owner: `relic.validation.framework.registry.RuleMetadata`

Constructor:

```python
RuleMetadata(
    rule_id: str,
    name: str,
    description: str,
    layer: str,
    domains: List[str] = ...,
    tags: List[str] = ...,
    inputs_schema: Optional[Dict[str, Any]] = None,
    outputs_schema: Optional[Dict[str, Any]] = None,
    dependencies: List[str] = ...,
    execution_order: int = 0,
    cf_requirements: List[str] = ...,
    units_contract: Dict[str, str] = ...,
    references: List[str] = ...,
    version: str = "1.0.0",
    author: Optional[str] = None,
    enabled: bool = True,
    parallel_safe: bool = True,
    requires_state_manager: bool = False,
    required_data: List[str] = ...,
)
```

Identity and classification parameters:

- `rule_id`, `name`, and `description` identify and explain the rule.
- `layer` selects the validation layer.
- `domains` selects applicable scientific domains.
- `tags` supports filtering and grouping.

Contract and dependency parameters:

- `inputs_schema` and `outputs_schema` describe rule data contracts.
- `dependencies` lists prerequisite rule identifiers.
- `execution_order` supplies an ordering hint.
- `cf_requirements` lists applicable CF requirements.
- `units_contract` maps values to required units.
- `references` records scientific or standards references.

Lifecycle parameters:

- `version` identifies the rule contract version.
- `author` records the maintainer when supplied.
- `enabled` controls rule discovery.
- `parallel_safe` declares whether concurrent execution is permitted.
- `requires_state_manager` marks a StateManager dependency.
- `required_data` lists required context data.

Tags and domains are normalized to lowercase, and domain names are added to
the tag set. An unsupported layer raises `ValueError`.

### `ValidationRegistry`

Canonical owner: `relic.validation.framework.registry.ValidationRegistry`

Convenience import:

```python
from relic.validation import ValidationRegistry
```

Constructor:

```python
ValidationRegistry() -> ValidationRegistry
```

Registration and lookup:

```python
registry.register_rule(
    rule_function: Callable,
    metadata: RuleMetadata,
    overwrite: bool = False,
) -> None

registry.unregister_rule(rule_id: str) -> bool
registry.get_rule(rule_id: str) -> Optional[Callable]
registry.get_metadata(rule_id: str) -> Optional[RuleMetadata]
registry.list_rules(**filters: Any) -> List[str]
```

Filtered lookup:

```python
registry.get_rules_by_layer(
    layer: str,
    enabled_only: bool = True,
) -> Dict[str, Callable]

registry.get_rules_by_domain(
    domains: Union[str, List[str]],
    enabled_only: bool = True,
) -> Dict[str, Callable]

registry.get_rules_by_tags(
    tags: Union[str, List[str]],
    enabled_only: bool = True,
) -> Dict[str, Callable]
```

Dependency and lifecycle methods:

```python
registry.resolve_dependencies(rule_ids: List[str]) -> List[str]
registry.enable_rule(rule_id: str) -> bool
registry.disable_rule(rule_id: str) -> bool
registry.get_rule_statistics() -> Dict[str, Any]
registry.validate_registry_integrity() -> Dict[str, Any]
registry.export_registry() -> Dict[str, Any]
registry.clear_registry() -> None
```

Registering an existing `rule_id` without `overwrite=True` raises
`ValueError`. A missing dependency or dependency cycle also raises
`ValueError`. A rule function whose first parameter is not context-compatible
currently emits a warning and does not reject registration.

Canonical global owner:

```python
from relic.validation.framework.registry import get_global_registry

registry = get_global_registry()
```

`relic.validation.framework.validation_registry` dynamically resolves the
current canonical global registry and avoids retaining an obsolete alias after
the registry is replaced.

### `ValidationEngine`

Canonical owner: `relic.validation.framework.engine.ValidationEngine`

Convenience import:

```python
from relic.validation import ValidationEngine
```

Constructor:

```python
ValidationEngine(registry: Optional[ValidationRegistry] = None)
```

- `registry` injects a rule registry. The canonical global registry is used
  when omitted.

Rules run sequentially in the calling thread; the engine has no worker pool
and no per-rule timeout. The execution metadata of the report
(`rules_passed`, `rules_failed`, `execution_time_seconds`, `end_time`) are
final: they are counted before the report is generated.

Primary method:

```python
engine.run(
    layers: Optional[Union[str, List[str]]] = None,
    domains: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    enabled_only: bool = True,
    context: Optional[ValidationContext] = None,
    fail_fast: bool = False,
    include_metrics: bool = True,
    **kwargs: Any,
) -> ValidationReportModel
```

- `layers` selects one or more L1 through L4 layers; omission selects the
  default layer order.
- `domains` and `tags` filter rule metadata.
- `enabled_only` excludes disabled rules when true.
- `context` supplies state and execution data; a context is created when
  omitted.
- `fail_fast` stops after a critical failure and raises at the end of the
  interrupted run.
- `include_metrics` includes execution timing and counts.
- `kwargs` is accepted for API compatibility.

Additional methods:

```python
engine.generate_report(
    results: List[ValidationResultModel],
    metrics: Optional[ExecutionMetrics] = None,
) -> ValidationReportModel

engine.get_execution_status() -> Dict[str, Any]
engine.stop_execution() -> bool
engine.get_registry_info() -> Dict[str, Any]
```

An invalid layer raises `ValueError`. Dependency, layer, and execution-strategy
failures are normally converted into a `CRITICAL` `ValidationResultModel`.
When `fail_fast=True` and a critical failure occurs, `run()` raises
`RuntimeError`. `stop_execution()` is best-effort and records
`stop_requested` in context metadata.

### `ValidationReporter`

Canonical owner: `relic.validation.framework.reporters.ValidationReporter`

Convenience import:

```python
from relic.validation import ValidationReporter
```

Constructor:

```python
ValidationReporter(cf_conventions: Optional[CFConventions] = None)
```

- `cf_conventions` overrides the default CF conventions handler.

Serialization methods:

```python
reporter.to_json(
    report: ValidationReportModel,
    indent: int = 2,
    include_metadata: bool = True,
) -> str

reporter.to_yaml(
    report: ValidationReportModel,
    include_metadata: bool = True,
) -> str

reporter.to_csv(
    report: ValidationReportModel,
    include_details: bool = False,
) -> str

reporter.to_markdown(
    report: ValidationReportModel,
    include_details: bool = True,
    max_detail_length: int = 100,
) -> str

reporter.to_netcdf_dict(
    report: ValidationReportModel,
    variable_prefix: str = "validation_",
) -> Dict[str, Any]

reporter.to_cli_summary(
    report: ValidationReportModel,
    width: int = 80,
    show_details: bool = False,
) -> str

reporter.save_report(
    report: ValidationReportModel,
    file_path: Union[str, Path],
    format: str = "auto",
    **kwargs: Any,
) -> None
```

The reporter module imports PyYAML when it is imported, so PyYAML is required
to use it. `save_report()` raises `ValueError` for an unsupported format.
Serialization and filesystem failures otherwise propagate.

## Command-line entry points and runtime boundaries

- `python -m relic.orchestration` (`relic.orchestration.__main__`) exposes
  `validate <graph>` and `compile <graph>` (`--strict`, `--auto-adapt`,
  `--output`) over `load_graph`, `validate_graph`, and `compile_graph`.
- `python -m relic.interfaces.cli.main` exposes seven top-level subcommands:
  `params`, `optimization` (`opt`, `[N]`), `data`, `compute`, `stochastic`
  (`stoch`), `orchestration` (`graph`), and `healthcheck` (`health`).
- Certified Level 2 (`[I]`) stochastic processes
  (`OrnsteinUhlenbeckTemperatureFunctor` and `MarkedPoissonRainfallFunctor`)
  compile and execute through `GraphIR` and `GraphExecutor` with
  `FunctorSubstreamMixin` checkpoint replay. By contrast, `optimization` CLI
  subcommands are `[N]` (`runtime_unbound`), and `FourDVar` in
  `relic.optimization.assimilation` is an unimplemented `[N]` framework shell
  without an adjoint model (`compute_gradient` raises `NotImplementedError`).

## Canonical ownership summary

- `compile_graph()` owns GraphIR compilation; `RELICModelCompiler` is its
  facade.
- `RELICSimulator` and `StateManager` own runtime advancement;
  `GraphExecutor` bridges a plan to those owners.
- `UnifiedParameterManager` owns parameter and SSOT metadata; `SSOTCatalog` is a
  read-only orchestration projection.
- `NodeRegistry` owns node definitions only.
- `ContractRegistry` owns variable write and conservation contracts.
- `ValidationRegistry` owns unified validation rules.
- Graph `ValidationReport` describes GraphIR compatibility and assertions.
- `ValidationReportModel` describes framework-wide L1 through L4 results.
- Adapter insertions in a compile report are metadata; only `auto_adapt=True`
  lowers context-free unit conversions into executable adapter steps.
