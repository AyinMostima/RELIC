# Extension Contracts

## Reuse-first decision

Before adding a module, identify the existing canonical owner. If a stable
implementation already satisfies the required behavior, extend or configure it;
do not create another manager, registry, compiler, runtime, loader, or schema.

| Needed capability | Extend this owner |
| --- | --- |
| Scientific process | Existing scientific package and one canonical Bridge Class |
| Runtime value or metadata | `UnifiedParameterManager` and the parameter YAML database |
| State ownership or writer permission | `ContractRegistry` under `relic\contracts\` |
| Graph node or port behavior | Existing GraphIR, `NodeRegistry`, `PortSpec`, validator, and compiler |
| Physical constant | `physical\constants.yaml` and `get_physical_constant(...)` |
| Device behavior | `relic\core\device_policy.py` and simulator registration |
| Validation rule | Existing `ValidationRegistry` and validation result model |
| Python extension discovery | `relic\extensions\` and `relic_extension.yaml` |
| C ABI or shared-library coupling | `relic\coupling\plugin\` |

## Adding a scientific Functor

1. Select the primary abstraction: `PhysicalField`,
   `BiologicalProcess`, `TransportMechanism`, or `ScaleCoupling`.
2. Inherit through the corresponding Bridge Class and therefore
   `FunctorBase` and `torch.nn.Module`.
3. Declare inputs, outputs, dependencies, stage, scale, variable kind,
   references, equations, units, and conserved quantities.
4. Add complete parameter YAML entries and access runtime values only through
   `self.get_parameter_value(...)`.
5. Implement one `forward(state, time, dt, **kwargs)` path.
6. Return declared state updates, diagnostics under `diagnostics.*` (or
   `DERIVED` outputs), and any checkpoint, diagnostic-export, or topology
   request as a top-level `ExternalIntent`; let the simulator commit state
   through `StateManager` and deliver requests after the step is accepted. Do
   not perform external I/O inside `forward`.
7. Register the Functor through the existing simulator and dependency graph.
8. Keep replay-critical mutable data in parameters, buffers, or
   `get_extra_state()`/`set_extra_state()`; other Python attributes are not
   restored by step rollback or checkpoint restore.

Do not add `compute_rate()` or `compute_flux()` as a parallel execution path.
Do not use a legacy fallback mode for a process claimed as strict IPHS.

## Parameter extension

Every scientific parameter entry declares:

| Required metadata | Purpose |
| --- | --- |
| `value`, `dtype`, `units` | Runtime value contract |
| `bounds` | Valid range |
| `category`, `description`, `tags` | Discovery and meaning |
| `references`, `provenance` | Scientific and data lineage |
| `requires_grad` | Optimization and differentiability |

Reuse an existing semantic parameter before adding a new key. In the live
database, `138` YAML files (`134` in the base `relic` tier with `4,642` leaf
entries and `4` in the `calibrated` tier with `109` leaf entries, totaling
`4,751` leaf entries) load `4,643` default runtime parameters (`810` with
`requires_grad=True`). Physical constants reside exclusively in
`relic\parameters\database\relic\physical\constants.yaml` and are accessed
through `get_physical_constant(...)`. When a process is replaced or merged,
remove or explicitly deprecate orphaned keys; do not keep two namespaces for
the same quantity.

## Variable contract extension

Add variable contracts under `relic\contracts\variables\` using the existing
schema. Declare canonical name, kind, units, aliases, owner, permitted writers,
increment behavior, flux whitelist, conservation tolerances, and domain
membership as applicable.

Ownership is not inferred from output names. Optional externally supplied state
uses no production owner, no increment permission, and explicit provenance.
When configuration selects one implementation from a closed set, selection is
proved by the existing graph or bundle construction path; do not add a
conditional-owner registry.

## Graph and adapter extension

Extend `GraphIR`, `PortSpec`, `NodeRegistry`, `validator.py`, and
`compiler.py` only when the capability belongs to orchestration. Preserve the
single serialized graph and issue models.

An adapter is executable only when it is represented by authored runtime
behavior. Compiler recommendation metadata is not sufficient. Adapters must
not infer missing scientific closure, conservation, ownership, or
effort-flow pairing.

## External extension paths

Python and manifest-based extensions use `relic\extensions\` and
`relic_extension.yaml`. Language-agnostic C ABI and shared-library coupling
uses `PluginHost` and `PluginFunctorAdapter` under
`relic\coupling\plugin\`.

After loading, both paths rejoin the same Functor registration,
`UnifiedParameterManager`, dependency, simulator, and `StateManager` lifecycle.

## Scientific structure claims

A process may claim only the level supported by its executable path and
`IPHSEquivalenceCertificate`. In the live repository, `90` unique concrete
scientific `Functor` classes are registered across the four canonical bridge
families (`BiologicalProcessFunctor`: `37`, `TransportMechanismFunctor`: `27`,
`PhysicalFieldFunctor`: `14`, `ScaleCouplingFunctor`: `12`; `0` uncertified)
and are backed by `92` YAML certificates: a primary 90-class distribution
across Levels 0 through 5 of `4 / 15 / 39 / 32 / 0 / 0`, plus `2` secondary
`constructor_selection` certificates bringing the 92-certificate totals to
`4 / 15 / 39 / 32 / 1 / 1`. Typed cochains (`CochainSpaceSpec`), linear chain
maps (`LinearChainMap`, validated in `__post_init__` for two-dimensional
matrix rank, `(target.n_dofs, source.n_dofs)` shape, finite entries, and
positive `unit_factor`), Hodge star specifications (`HodgeStarSpec`), and
declared IPHS operators (`IPHSStructureSpec`) provide structural contracts,
not automatic scientific closure.

For reaction and matter-transformation processes, verify actual donor and
recipient increments, units, support, time integration, external exchanges,
and exactly one state advance. Where a process declares and uses `N r`, its
single production path must evaluate that declared operator and rates; closed
ledgers verify `C N=0`, and open exchanges require routed amounts on the
same unit/support/time basis. A mature pool model without an `N r`
representation verifies its actual increments and physical budget directly;
it must not invent identity stoichiometry to satisfy an unrelated claim.
Clipping and overflow remain explicit, and dependencies use metadata, ports,
chain maps, or `IntegrationPlan`.

Strict dissipative-pH and GENERIC claims are conditional on the corresponding
runtime point and governed process-certificate witness:
`FickianDiffusion(strict_runtime_point=True, spatial_dims=1)` at Level 4
(`{strict_runtime_point: true, spatial_dims: 1}`) and
`ChemistryFunctor(physical_boundary="closed_constant_volume")` at Level 5
(`{physical_boundary: closed_constant_volume}`). Both are documented in
[Scientific Structure and Extensions API](api/scientific_structure_extensions.md).
Ordinary graph `strict=True` validates the declared graph contract; it does
not by itself request strict IPHS or authorize a broader thermodynamic claim.

## Explicit unavailable scientific boundaries

Four public scientific entries are currently non-operational:
`EmissionInventory` (`runtime_domain_unbound`), `AtmosphereCoupling`
(`runtime_dependency_unbound`), `AnthroposphereInterfaceMapping`
(`runtime_metadata_domain_unbound`), and `IsotopeHeartbeatFunctor`
(`runtime_unbound`). Constructors and `forward(...)` reject before state or
output effects, including an unsafe constructor bypass. Empty runtime
I/O/ports are the unavailable contract, not successful zero-valued models.

Re-enablement requires, respectively: source-bound target-domain inventory
fractions and allocation; provenance-bound six-species providers in the same
graph; exact SCM/official-CF/unit/support/coordinate and governed consumer
binding; and isotope component UpdateItems with complete transfer routes,
parent-process time ownership, and restart behavior. Their class presence and
Level 0 records do not establish runnable processes. The generated science
reference is the per-class source-backed detail.

## Anti-redundancy boundary

The following additions are non-compliant when an owner already exists:

- a second simulator, `StateManager`, or `UnifiedParameterManager`;
- another `ValidationRegistry`, `NodeRegistry`, or compiler;
- a parallel `PortSpec`, variable-contract schema, or IPHS level taxonomy;
- module-local device selection or physical constants;
- direct Functor-to-Functor execution;
- hidden source, sink, clipping, or overflow branches; and
- compatibility wrappers that recreate removed parameter or execution APIs.

An extension is complete only when its contract, owner, runtime path, and
configuration agree. Successful import or graph compilation alone is not
evidence of executed scientific behavior.
