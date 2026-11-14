# Graph Validation and Execution

RELIC orchestration converts a declarative Graph Intermediate Representation
(`GraphIR`) into an `ExecutionPlan`, then executes that plan through the
existing `RELICSimulator`. The compiler does not create a parallel simulator,
parameter manager, state manager, or validation registry.

## Orchestration pipeline

```text
GraphIR
  -> SSOTCatalog and NodeRegistry
  -> validate_graph()
  -> compile_graph()
  -> ExecutionPlan
  -> GraphExecutor
  -> RELICSimulator and StateManager
```

Canonical owners are listed in [Orchestration](../orchestration.md) and
[Orchestration and Validation API](../api/orchestration_validation.md).

## Graph data classes

```text
GraphIR(
    schema_version: str,
    nodes: dict[str, NodeSpec],
    links: list[LinkSpec] = [],
    metadata: dict[str, Any] = {},
    assertion: dict[str, Any] = {},
)

NodeSpec(
    class_type: str,
    inputs: dict[str, Any] = {},
    outputs: list[str] = [],
    config: dict[str, Any] = {},
)

LinkEndpoint(node: str, port: str)
LinkSpec(source: LinkEndpoint, target: LinkEndpoint)
```

| Field | Meaning |
| --- | --- |
| `schema_version` | Version of the GraphIR serialization contract. It must be non-empty. |
| `nodes` | Node identifier to `NodeSpec` mapping. |
| `links` | Explicit typed-port connections. Links can also be inferred from node input link declarations. |
| `metadata` | Workflow-level metadata, including optional input manifests and role hints. |
| `assertion` | Gate 0 structural assertion payload. |
| `class_type` | Registered node type, importable Functor class, or supported scientific registry name. |
| `inputs` | Literal inputs or links to upstream node ports. |
| `outputs` | Output names exposed by the node. |
| `config` | Constructor and execution settings, including an optional `execution_group`. |

`GraphIR.from_dict(payload)` rejects a non-mapping payload, an empty schema
version, non-mapping nodes, missing class types, invalid input/output shapes,
or invalid metadata and assertion shapes.

## Load and save a graph

```python
from pathlib import Path
import tempfile

from relic.orchestration import GraphIR
from relic.orchestration.graph_io import load_graph, save_graph

with tempfile.TemporaryDirectory() as tmpdir:
    workflow_path = Path(tmpdir) / "workflow.json"
    source_graph = GraphIR(
        schema_version="1.0",
        nodes={},
        metadata={"name": "roundtrip_demo"},
    )
    save_graph(source_graph, workflow_path)
    loaded_graph = load_graph(workflow_path, validate_schema=True)
    print(loaded_graph.schema_version, loaded_graph.metadata["name"])
```

```text
load_graph(
    path: pathlib.Path,
    *,
    validate_schema: bool = False,
    schema_path: Optional[pathlib.Path] = None,
) -> GraphIR
```

The repository ships no example GraphIR file. JSON and YAML input are
supported: a `.yaml` or `.yml` suffix selects YAML, and any other suffix JSON.
JSON Schema validation is opt-in through
`validate_schema=True` or the `RELIC_GRAPH_SCHEMA_VALIDATE` environment
variable. YAML loading raises `RuntimeError` when PyYAML is unavailable.
Malformed files propagate JSON, YAML, schema, type, or value errors.
`save_graph()` writes JSON.

## `SSOTCatalog`

`SSOTCatalog` resolves variable semantics from the Single Source of Truth (SSOT)
parameter database and variable contracts.

```python
from relic.orchestration import SSOTCatalog

catalog = SSOTCatalog.from_parameter_manager()
entry = catalog.get("air_temperature")
daily_max = catalog.get_by_standard_name(
    "air_temperature",
    cell_methods="time: maximum",
)
```

`SSOTCatalog.from_parameter_manager(manager=None)` loads, from the supplied
`UnifiedParameterManager` or the global manager when omitted, the variable
definitions (`coupling.external.variables.variable_definitions`), the scale
defaults (`coupling.scales.defaults.variables`), the variable contracts of
`ContractRegistry` with their aliases, and the optional alias table
`coupling.external.variables.aliases`. Missing required SSOT payloads raise an
exception. The catalog does not collect conflicts: loading raises `ValueError`
when a variable is declared twice with contradicting fields, when two
variables claim one Climate and Forecast (CF) identity (`standard_name` plus `cell_methods`), when an
alias names another variable, is claimed by two variables, or targets an
unknown variable, and when the contract registry reports conflicts.
`get(name)` looks up a variable name, then an explicit alias, then the
unqualified standard name.
`get_by_standard_name(standard_name, *, cell_methods=None)` returns the one
entry that holds that identity, or `None`. In the example, `entry` is
`air_temperature` and `daily_max` is `air_temperature_max` (its alias
`daily_maximum_air_temperature` resolves to it), both in K.

## `NodeRegistry`

```text
NodeRegistry(
    ssot_catalog: SSOTCatalog,
    definitions: dict[str, NodeDefinition] = {},
)
```

| Method | Purpose | Return value |
| --- | --- | --- |
| `register_definition(node_type_id, definition)` | Register a typed node definition. | `None` |
| `unregister_definition(node_type_id)` | Remove a registered definition. | `bool` |
| `get_definition(node_type_id)` | Look up a definition. | `NodeDefinition` or `None` |
| `register_functor_instance(functor, node_type_id=None)` | Derive ports from `FunctorMetadata` and register the result. | `NodeDefinition` |
| `resolve_class_type(class_type)` | Resolve a registered class type to an implementation class when available. | Class or `None` |

A Functor without metadata cannot be registered and raises `ValueError`.
The registry derives ports from declared input and output variables and the
SSOT catalog; it does not infer missing scientific semantics from the
Functor's implementation.

## Validate a graph

```text
validate_graph(
    graph: GraphIR,
    registry: NodeRegistry,
    unit_converter=None,
    strict: bool = True,
    context=None,
    ssot_catalog=None,
) -> ValidationReport
```

`ValidationReport` contains:

| Field | Meaning |
| --- | --- |
| `errors` | Blocking `GraphValidationIssue` objects. |
| `warnings` | Non-blocking validation issues. |
| `assertion_audit` | Per-assertion Gate 0 evidence and results. |

Check `report.errors` explicitly:

```python
from relic.orchestration import GraphIR, NodeRegistry, SSOTCatalog, validate_graph

graph = GraphIR(schema_version="1.0", nodes={}, metadata={"name": "validation_check"})
registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
report = validate_graph(graph, registry, strict=True)
if report.errors:
    messages = [issue.message for issue in report.errors]
    raise ValueError(messages)
print(len(report.errors))
```

Validation covers registered node types, port existence, data types, units,
quantities, semantics, write/update modes, scale compatibility, dependency
structure, and declared structural assertions. `strict=False` can demote
supported compatibility findings to warnings; it does not supply missing
scientific structure.

## Gate 0 structural assertions

The current assertion operations are:

| Assertion operation | Contract checked |
| --- | --- |
| `cochain_complex_check` | Discrete Exterior Calculus (DEC) cochain-complex structure and boundary composition. |
| `hodge_psd_check` | Hodge or dissipative operator positivity requirements. |
| `iphs_structure_check` | Declared Interconnected Port-Hamiltonian System (IPHS) interconnection, dissipation, ports, and structure. |
| `chain_map_check` | Morphism or regridding compatibility between complexes. |
| `entropy_step_check` | Entropy-production sign and step-level evidence. |
| `port_power_units_check` | Effort-flow power-unit compatibility. |
| `port_matching_check` | Declared port interconnection and direction compatibility. |
| `overflow_routing_check` | Explicit routing of clipping, overflow, or residual terms. |
| `equivalence_certificate_check` | Migration certificate completeness and binding. |
| `reaction_network_check` | Stoichiometry, rates, total affinity-rate power/dissipation, internal `C N = 0`, and any additional open exchange ledger. Without an explicit positive-temperature and unit binding, $A^{\mathsf{T}} r$ is not a thermodynamic entropy-production rate. |
| `stochastic_sdae_check` | Declared stochastic differential-algebraic structure. |
| `state_domain_check` | State-variable domain and ownership declarations. |
| `observation_diagnostic_check` | Observation and diagnostic separation from prognostic updates. |

Strict requirements depend on process role:

| Process role | Required assertion additions |
| --- | --- |
| Ordinary dynamic process | Cochain, Hodge, IPHS, entropy, power units, port matching, overflow, certificate, and state domain checks. |
| Reaction or matter transformation | Ordinary dynamic set plus `reaction_network_check`. |
| Coupling, regridding, or morphism | Ordinary dynamic set plus `chain_map_check`. |
| Stochastic process | Ordinary dynamic set plus `stochastic_sdae_check`. |
| Observation or diagnostic process | Certificate, state domain, and `observation_diagnostic_check`. |

An explicit `required_structural_assertions` declaration can add requirements.
Passing Gate 0 establishes the declared structural checks; it does not by
itself establish a strict IPHS level. IPHS level ownership remains with the
equivalence certificate.

The reaction rule performs a bounded point audit. Constitutive-domain
validity, open-route application, energy/entropy pairing, and time-integrated
single-owner/no-double-count execution remain `[N]`.

## Compile a graph

```text
compile_graph(
    graph: GraphIR,
    registry: NodeRegistry,
    adapter_registry=None,
    auto_adapt: bool = False,
    strict: bool = True,
    context=None,
    optimization_passes=None,
    enable_optimization: bool = False,
    include_topology: bool = False,
) -> ExecutionPlan
```

The compiler validates, optionally applies the supplied optimization passes,
topologically sorts the graph, and returns:

```text
ExecutionPlan(
    steps: list[ExecutionStep],
    report: CompileReport,
)
```

Each `ExecutionStep` contains `node_id`, `class_type`, `inputs`, `outputs`,
and `config`. The compile report contains the validation report, adapter
insertion descriptions, and audit metadata.

Validation errors, failed strict structural audits, cycles, or unresolved
dependencies raise `ValueError`. `RELICModelCompiler` is only a facade over
`compile_graph()`:

```python
from relic.orchestration import (
    GraphIR,
    NodeRegistry,
    RELICModelCompiler,
    SSOTCatalog,
)

graph = GraphIR(schema_version="1.0", nodes={})
registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
compiler = RELICModelCompiler(registry)
plan = compiler.compile(graph, strict=True)
print(len(plan.steps))
```

`ExecutionPlan` is a dataclass. It does not provide `execute()`,
`build_simulator()`, or `to_dict()` methods. Use `GraphExecutor`,
`compile_and_execute`, or the module-level `plan_to_dict()` function.

### Adapter registry and `unit_cast` lowering guard

`AdapterRegistry` (`relic\orchestration\adapters.py`) registers 7 adapter keys
across 6 concrete `AdapterBase` subclasses, wrapped for runtime execution by
`ExecutableAdapterFunctor`:

| Registry key | Adapter class | Purpose |
| --- | --- | --- |
| `unit_cast` | `UnitCastAdapter` | Affine or multiplicative unit conversion through `UnitConverter`. |
| `temporal_resample` | `TemporalResampleAdapter` | Explicit temporal aggregation or disaggregation across step frequencies. |
| `semantic_adapter`, `flux_to_increment` | `SemanticAdapter` | Semantic conversion between per-second rates and per-step increments (`flux_to_increment` aliases `semantic_adapter`). |
| `area_basis_transform` | `AreaBasisTransformAdapter` | Conversion between per-cell totals and per-area densities using explicit cell areas. |
| `surface_flux_to_volume_source` | `SurfaceFluxToVolumeSourceAdapter` | Conversion between surface fluxes and volumetric sources across layer thickness. |
| `support_aggregation` | `SupportAggregationAdapter` | Explicit spatial or vertical support aggregation with declared weights. |

```python
from relic.orchestration.adapters import AdapterRegistry

adapter_registry = AdapterRegistry()
print(type(adapter_registry.get("unit_cast")).__name__)
print(
    adapter_registry.get("semantic_adapter")
    is adapter_registry.get("flux_to_increment")
)
```

When `auto_adapt=True`, `compile_graph()` records insertions in
`plan.report.adapter_insertions` and automatically lowers only `unit_cast`
insertions that satisfy all three guard conditions in `relic\orchestration\compiler.py`:

1. **Pure context-free unit contract**: only `unit_cast` is eligible, neither
   endpoint port declares `context_requirements`, all non-unit port contract
   fields match (`semantics`, `per_timestep`, `conservative`, and strict
   `can_connect` after substituting `target.units`), and
   `UnitConverter.convert` succeeds with `context=None`.
2. **Matching explicit `execution_group` on non-`State:` endpoints**: the
   source node is not a declarative `State:` boundary, both endpoint nodes
   declare an explicit `execution_group` in `{"fast", "medium", "slow"}`, and
   both groups are equal.
3. **Distinct source and target port names**: `source_port` and `target_port`
   differ so single-writer state ownership is preserved without overwriting the
   upstream variable.

Any other adapter type, cross-group link, context-dependent conversion, or
same-name port pair raises `ValueError` during automatic lowering and must be
inserted explicitly as a graph node.

### Command-line orchestration

`relic\orchestration\__main__.py` exposes standalone validation and
compilation from the shell without launching an execution session:

```powershell
python -m relic.orchestration validate path\to\graph.json
python -m relic.orchestration compile path\to\graph.json --auto-adapt --output output\plan.json
```

## Execute a plan

```text
GraphExecutor(
    plan,
    state_manager=None,
    cache=None,
    output_dir=None,
    registry=None,
    strategy=ExecutionStrategy.PUSH,
    dirty_include_time=True,
    simulator_config=None,
)
```

| Parameter | Meaning |
| --- | --- |
| `plan` | Validated, compiled `ExecutionPlan`. |
| `state_manager` | Optional state owner; a manager is created when omitted. |
| `cache` | Optional `NodeCache`. |
| `output_dir` | Optional provenance output directory. |
| `registry` | Registry used to resolve classes and registered instances. |
| `strategy` | `PUSH` evaluates every node each time its rate group runs; `DIRTY` evaluates changed nodes. |
| `dirty_include_time` | Include time in dirty-input hashing. |
| `simulator_config` | Optional runtime mapping copied into each fresh `RELICSimulator`. |

```text
executor.execute(
    initial_state: dict[str, torch.Tensor],
    n_steps: int,
    dt: float,
) -> dict[str, torch.Tensor]
```

`n_steps` and `dt` (seconds) are required; there is no default time step.

Execution first requires the strict compile-run witness, so a plan compiled
without `strict=True` is refused. It then resets and initializes the state
manager, creates the dependency resolver and a fresh simulator, resolves every
plan step to a Functor, registers each Functor, advances `n_steps`, and
returns the final managed state. Unknown or
unimportable classes, invalid Functor construction, state-contract errors,
and runtime execution errors are propagated.

## Minimal orchestration example

This example exercises the complete graph plumbing without claiming a
scientific process:

```python
import torch

from relic.orchestration import (
    GraphExecutor,
    GraphIR,
    NodeRegistry,
    SSOTCatalog,
    compile_graph,
    validate_graph,
)

graph = GraphIR(
    schema_version="1.0",
    nodes={},
    metadata={"name": "empty_plumbing_example"},
    assertion={},
)
registry = NodeRegistry(SSOTCatalog.from_parameter_manager())

report = validate_graph(graph, registry, strict=True)
if report.errors:
    raise RuntimeError([issue.message for issue in report.errors])

plan = compile_graph(graph, registry, strict=True)
executor = GraphExecutor(plan, registry=registry)
final_state = executor.execute(
    {"x": torch.tensor(1.0)},
    n_steps=1,
    dt=3600.0,
)

print(len(plan.steps))
print(float(final_state["x"]))
```

Expected output:

```text
0
1.0
```

For a scientific graph, register real Functor definitions and provide the
role-specific Gate 0 evidence before compilation.

## Convenience execution

```text
compile_and_execute(
    graph,
    registry,
    initial_state,
    n_steps: int,
    dt: float,
    output_dir=None,
    cache_dir=None,
    cache=None,
    simulator_config=None,
    **compile_kwargs,
) -> dict[str, Any]
```

`n_steps` and `dt` are required; the caller chooses the step count and interval.
This function calls `compile_graph()`, constructs `GraphExecutor` with
`simulator_config`, and returns the final state. It does not bypass
validation or create an alternate runtime.

## Related documentation

- [Architecture](../architecture.md)
- [Orchestration](../orchestration.md)
- [Orchestration and Validation API](../api/orchestration_validation.md)
- [State, Functors, and Simulation](state_functors_and_simulation.md)
- [Science index](../science/index.rst)
- [Theory: DEC, IPHS, and GENERIC](../theory/dec_iphs_generic.md)
