# Graph Orchestration and Typed Ports

This document describes the implemented GraphIR path in RELIC. It is an
orchestration layer over the existing simulator, not a second simulator,
scheduler, state manager, parameter manager, validation framework, or compiler.
The package-level ownership and Phase 8 boundaries remain authoritative in the
[RELIC Architecture Overview](architecture_overview.md).

## 1. Canonical Owners

| Responsibility | Canonical owner |
| --- | --- |
| Serializable graph schema | `relic\orchestration\graph_ir.py` |
| Typed port and node contracts | `relic\orchestration\port_spec.py` |
| SSOT-backed variable metadata | `relic\orchestration\ssot_catalog.py` |
| Node/class resolution | `relic\orchestration\node_registry.py` |
| Connection and graph validation | `relic\orchestration\validator.py` |
| Adapter implementations/registry | `relic\orchestration\adapters.py` |
| Compilation and run-bundle serialization | `relic\orchestration\compiler.py` |
| Plan execution and simulator bridging | `relic\orchestration\executor.py` |
| Downstream state ownership/writer constraints | `relic\contracts\contract_loader.py`, enforced by `StateManager.apply_updates(...)` instead of GraphIR validation |

`RELICModelCompiler` is only a facade over `compile_graph(...)`. New graph
features extend these owners; they do not introduce an alternative GraphIR,
port schema, assertion service, registry, or compiler.

## 2. Resolved Port Contract

`PortSpec` is the structural port schema. `SSOTCatalog` supplies variable
identity and units from ParameterManager-backed catalogs. `can_connect(...)`
and `validate_graph(...)` use those orchestration contracts; they do not accept
or query a `ContractRegistry`. State ownership and writer constraints are
enforced later by `StateManager.apply_updates(...)` when contracts are enabled.

:::{figure} ../../assets/readme/relic_typed_ports.png
:alt: Resolved RELIC port contract assembled from PortSpec, SSOT metadata, and state ownership contracts.
:width: 95%
:align: center

The figure is an end-to-end contract view: compilation resolves structural
`PortSpec` fields and SSOT identity metadata, while the later StateManager
commit applies `ContractRegistry` responsibility constraints. It does not imply
that GraphIR validation itself queries all three owners. The "port passport" is
this combined lifecycle view; not every displayed field belongs directly to
the `PortSpec` dataclass.

The adapter chain on the right is conceptual. With `auto_adapt=True` the
compiler lowers an unambiguous context-free unit conversion into an executable
`ExecutableAdapterFunctor` node between the two endpoints (both endpoints need
the same explicit `execution_group`); temporal, semantic, and
context-dependent conversions still have to be authored explicitly before
runtime.
:::

### 2.1 PortSpec fields

| Group | Fields |
| --- | --- |
| Identity | `name`, `standard_name`, `cf_standard_name`, `quantity`, `element` |
| Data contract | `units`, `datatype`, `shape`, `required`, `context_requirements` |
| Update semantics | `semantics`, `update_mode`, `write_mode`, `direction`, `per_timestep`, `conservative` |
| DEC/IPHS structure | `form_degree`, `primality`, `extensivity`, `complex_id`, `physical_domain`, `orientation_convention`, `boundary_role`, `port_kind`, `power_conjugate` |
| Support contract | `spatial_kind`, `area_basis`, `temporal_basis`, `vertical_basis`, `declared_support_fields` (`PORT_SUPPORT_FIELDS` adds `shape`) |

`default_chain` is not a `PortSpec` field. Defaults and parameter bindings belong
to node configuration and the existing parameter system. Units and standard
names are resolved from SSOT metadata instead of being redeclared in graph
documents.

## 3. GraphIR Schema

`GraphIR` has five top-level fields:

- `schema_version: str`
- `nodes: dict[str, NodeSpec]`
- `links: list[LinkSpec]`
- `metadata: dict`
- `assertion: dict`

Each `NodeSpec` contains `class_type`, `inputs`, `outputs`, and `config`.
`GraphIR.from_dict(...)` requires `assertion` to be a mapping. A direct list is
invalid. Multiple structural assertions are represented inside a mapping
container such as `assertions`, `items`, or `checks`, or through the supported
`metadata.structural_assertions` fallback.

The following minimal graph illustrates the validation and compilation schema:

```json
{
  "schema_version": "1.0.0",
  "nodes": {
    "forcing": {
      "class_type": "State:ExternalForcing",
      "outputs": ["precipitation_flux"]
    },
    "bucket": {
      "class_type": "Functor:relic.scientific.hydrosphere.soil_moisture.bucket.BucketSoilMoisture",
      "inputs": {
        "precipitation_flux": {
          "link": ["forcing", "precipitation_flux"]
        }
      },
      "outputs": ["soil_moisture"],
      "config": {}
    }
  },
  "links": [
    {
      "from": {"node": "forcing", "port": "precipitation_flux"},
      "to": {"node": "bucket", "port": "precipitation_flux"}
    }
  ],
  "metadata": {}
}
```

This graph is executable when `initial_state` supplies
`precipitation_flux`. After validation and optional optimization,
`compile_graph(...)` lowers the exact `State:ExternalForcing` declaration to a
same-named `{"state": "precipitation_flux"}` input binding. The boundary is
retained in the declarative GraphIR and run-bundle workflow, but it is not
emitted as an `ExecutionStep`. The compile report records lowered nodes and
links, required initial-state keys, and executable node IDs under
`metadata.state_boundary_lowering`.

The lowering is deliberately narrow. External boundaries cannot define inputs,
runtime config, incoming links, aliases, duplicate outputs, or duplicate state
owners. Any other `State:*` node that remains after optional optimization is
rejected instead of being imported as a runtime class. `GraphExecutor.execute(...)`
checks the sorted required initial-state keys before resetting or initializing
StateManager.

`NodeSpec.inputs` is intentionally a generic mapping, but the current executor
consumes `{"link": ["node_id", "port_name"]}` to derive dependencies and
`{"state": "variable_name"}` for flat StateManager reads. Compiler constant
folding may emit `{"value": ...}` in an optimized graph. The
orchestration layer currently has no generic `parameter_key` input resolver;
parameters must remain in Functor configuration and the canonical
ParameterManager path. Do not describe any additional input payload as
executable until its resolver and tests exist.

If the top-level `links` list is absent or empty, `GraphIR.from_dict(...)`
infers links from node input payloads. Explicit links remain preferable for
auditable workflow files.

## 4. Validation and Structural Assertions

`can_connect(...)` and `validate_graph(...)` evaluate typed compatibility and
graph-wide constraints. Relevant checks include:

1. node and port existence;
2. datatype, quantity, element, and local `standard_name` or bilateral explicit `cf_standard_name` identity compatibility;
3. unit convertibility and required conversion context;
4. semantics, update, and write-mode compatibility;
5. DEC structural port metadata (`form_degree`, `primality`, `extensivity`, `complex_id`, plus declared `orientation_convention` and `physical_domain`) and port support contracts (`shape`, `spatial_kind`, `area_basis`, `temporal_basis`, `vertical_basis`, rejecting one-sided declarations or mismatches without an explicit support adapter);
6. overlapping outputs and mutually exclusive output closures;
7. cycles, required bindings, topology, and structural assertions.

Validation returns coded errors and warnings; it does not silently invent units,
ports, owners, or adapters.

Phase 8 strict graphs declare the role-appropriate subset of the canonical
structural operations listed in the architecture overview. The complete
supported set is not an unconditional checklist for every graph:
`validator.py` derives required operations from the declared roles and
structure. GraphIR dispatch calls the shared rule implementations directly;
decorated wrappers of those implementations are also registered with the
framework `ValidationRegistry`. These are two entry paths to shared validation
logic, not two validation systems.

## 5. Adapters

`AdapterRegistry` (`relic\orchestration\adapters.py`) registers six concrete
adapter classes across seven adapter type keys:

- `UnitCastAdapter` (`unit_cast`) for UDUNITS and SSOT-backed unit conversion through `UnitConverter`;
- `TemporalResampleAdapter` (`temporal_resample`) for conservative window reductions (`interval_mean`, `interval_integral`) and explicit `current_step` zero-order hold alignment;
- `SemanticAdapter` (`semantic_adapter`, also registered as `flux_to_increment`) for rate-to-increment multiplication by runtime `dt`;
- `AreaBasisTransformAdapter` (`area_basis_transform`) for `leaf_area` and `ground_area` flux-density conversion with `leaf_area_index`;
- `SurfaceFluxToVolumeSourceAdapter` (`surface_flux_to_volume_source`) for conservative lowest-layer volumetric mass-source injection from an areal molar flux using `cell_area`, `layer_volumes`, and a physical-constants molar-mass key;
- `SupportAggregationAdapter` (`support_aggregation`) for support reduction governed by the port variable's SSOT `spatial_basis` and `coupling.scales.defaults.conservation_tolerance`.

Explicit graph-edge adapters run inside the Functor lifecycle through
`ExecutableAdapterFunctor` (a `ScaleCouplingFunctor` registered with
`AdapterRegistry.register_executable_node(...)`), which selects its input via
`config.source_mode` (`"subprocess_outputs"` by default, or `"state"`).

`compile_graph(..., auto_adapt=False)` does not synthesize a conversion.
Under `strict=True` a connection that requires an executable adapter is
rejected; only non-strict compilation keeps it as warning metadata, and a
non-strict plan is not executable. With `auto_adapt=True`, each insertion
recorded in `CompileReport.adapter_insertions` must pass three guards: it must
be a context-free `unit_cast`, both endpoint ports must agree on all non-unit
contract fields under a strict `can_connect(...)` probe, and both endpoint nodes
must declare the same explicit `execution_group` in `{"fast", "medium", "slow"}`.
Passing insertions are lowered into `ExecutableAdapterFunctor` steps and the
graph is revalidated. Other adapter types raise `ValueError`, and a remaining
adapter requirement is rejected in strict mode. Adapter decisions must not infer
a missing conservation law, scientific closure, effort/flow pairing, or state
owner.

## 6. Compilation and Execution

The canonical path is:

```text
GraphIR
  -> validate_graph(...)
  -> compile_graph(...)
       -> optional context-free unit-adapter lowering (auto_adapt=True)
       -> optional optimization passes
       -> revalidation after graph transformation
       -> external-forcing boundary lowering
       -> stable topological ExecutionStep order (node-id tie-break)
       -> strict compile-run witness (registration only, no step) and plan seal
  -> ExecutionPlan + CompileReport

For plans compiled with strict=True (passed compile-run witness):
  ExecutionPlan
    -> GraphExecutor
    -> RELICSimulator
    -> execution-group StateManager.apply_updates(...)
```

`compile_graph(...)` always validates; calling `validate_graph(...)` separately
is an optional preflight, not an alternative to compiler validation. The Python
API defaults to `strict=True`, while both CLI surfaces require an explicit
`--strict` flag and otherwise use non-strict mode. A plan compiled without
`strict=True` carries a skipped compile-run witness; `GraphExecutor` refuses
to execute it.

In memory, the unified compiler audit is
`ExecutionPlan.report.metadata["audit_summary"]`.
`plan_to_dict(...)` serializes it as `report.audit`. There is no
nested `report` attribute on the `CompileReport` dataclass.

`GraphExecutor` supports `ExecutionStrategy.PUSH` and
`ExecutionStrategy.DIRTY`; dirty execution is controlled through
`set_strategy(...)` and `mark_dirty(node_id)`. Execution reuses one
`RELICSimulator` per prepared run and its per-group-call StateManager commit
path. Strict compilation resolves, constructs, and registers every node once
in a registration-only simulator (no StateManager, no step) to produce the
compile-run witness; this proves class resolution and registration, not a
successful step or scientific validity. The ExecutionStep order is the
registration order. At run time the simulator orders each rate group
separately from the attached dependencies (ties by registration index), runs
fast, medium, and slow in that order, and executes the Functors of a group
sequentially.

## 7. Command-Line and API Surfaces

Reliable Windows command entry points:

```powershell
conda activate RELIC
python -m relic.orchestration validate path\to\graph.json --strict
python -m relic.orchestration compile path\to\graph.json --strict --run-bundle path\to\bundle
python -m relic.interfaces.cli.main orchestration validate path\to\graph.json --strict
```

The `relic` package defines no package-root `__main__` module or installed
`[project.scripts]` console-script declaration, so `python -m relic` and a bare
`relic` command must not be documented as guaranteed entry points.

When the optional API runtime dependencies are installed, the graph router
exposes:

- `POST /graph/validate`
- `POST /graph/compile`
- `POST /graph/submit`
- `GET /graph/history/{run_id}`
- `GET /graph/artifacts/{run_id}/{artifact_id}`
- `WS /graph/ws`

The source presence of API/Web routes does not prove that FastAPI, Uvicorn, or
Jinja dependencies are installed in a base environment.

## 8. NUOPC/ESMF Conceptual Mapping

This table is documentation-only; RELIC does not currently implement a NUOPC
runtime adapter.

| NUOPC phase | RELIC concept | Boundary |
| --- | --- | --- |
| advertise | `NodeRegistry` + resolved `PortSpec` | Publish typed, SSOT-backed contracts |
| realize | `compile_graph(...)` | Validate and compile GraphIR |
| import/export | `StateManager` ingestion/update paths | Move declared state and diagnostics |
| run | `RELICSimulator` | Execute registered Functors and rate groups |
| finalize | Run-bundle writer | Persist graph, plan, audit, and provenance |
| mediator | Existing scale/coupling services | Handle only explicitly declared mappings |

The mapping does not authorize embedded NUOPC code, a second mediator runtime,
or inferred regridding/scale semantics.

## 9. Verification

Run the focused orchestration contract before relying on an edited schema:

```powershell
conda activate RELIC
python -m pytest test\orchestration\orchestration_core.test.py -q
python -m pytest test\orchestration\graph_executor.test.py -q
python -m sphinx -E -a -W --keep-going -b html docs\source docs\build\html
```

Passing these focused tests proves the documented graph/port/compiler contract
at their scope. It does not substitute for full regression or performance
validation.
