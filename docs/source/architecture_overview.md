# RELIC Architecture Overview

This document serves to unify the understanding of the RELIC architecture, development workflow, and extension specifications.
The goal is to ensure that new features strictly align with the existing system, avoiding redundant implementations, redundant frameworks, and duplicate functionalities.

## 1. Document Positioning and Mandatory Principles

This document is the package-level architecture specification for `relic\`, not a conceptual draft. All descriptions herein reflect the authoritative current state of the repository.

Evidence boundaries:

- Stable ownership, lifecycle, and extension rules belong in this document.
- Generated audit counts and migration levels belong to the evidence artifacts under
  `analysis_results\` and are valid only for the commit/tree and command recorded by
  that artifact.
- Plans describe intended work; they are not proof that a runtime capability exists.
- A Sphinx build, static scan, or targeted test does not imply a full regression,
  performance qualification, or production-release decision.

Mandatory Principles:

- Always check for existing capabilities before extending.
- Prioritize extending existing modules over building a parallel, duplicate system.
- Scientific processes must follow the strict inheritance chain from abstract contracts through bridge classes and `FunctorBase` to the parameter system.
- Parameters, units, and physical constants must be centrally managed. Scattered hardcoding is strictly forbidden.
- Theoretical documentation, parameters, code, and tests must form a closed loop.
- Scientific imports must preserve native library functions. Convert scalar inputs
  at their own boundaries; do not replace PyTorch functions globally.
- Test files reside in `test\` (outside the package); do not create test directories inside `relic\`.
- RELIC-IPHS-DEC structural enforcement is additive over the existing architecture. It must reuse `FunctorBase`, `StateManager`, `ParameterManager`, `ValidationRegistry`, `NodeRegistry`, `PortSpec`, `SSOTCatalog`, `validate_graph(...)`, and `compile_graph(...)`.
- Each process has one documented production law through the existing governed lifecycle. Mature empirical and numerical formulations are valid when their applicability, interfaces, budgets, and outputs are verified. A claimed strict IPHS process must also bind its canonical IPHS runtime and mathematical evidence; descriptive levels 0-3 do not make that strict claim.

## 2. Package-Level Structure (Authoritative Directory)

`relic\` Top-Level Directories:

- `relic\contracts\`: Variable contracts, domain constraints, and contract loaders.
- `relic\core\`: The four major abstractions, Functor subsystem, simulator, state management, configurations, and services.
- `relic\coupling\`: Internal coupling, external coupling, scale coupling, protocol layers, and tools.
- `relic\data\`: I/O, standards, quality control, streaming data, and transformations.
- `relic\extensions\`: Manifest-based discovery, validation, loading, and registration of external or user Functors through existing owners.
- `relic\interfaces\`: CLI, API, Web, and Notebook interface layers.
- `relic\models\`: Entities (basin/patch/landscape) and surrogate model helpers.
- `relic\monitoring\`: Narrow Prometheus-style metric helpers; it is not a scheduler or simulation runtime.
- `relic\optimization\`: Calibration, assimilation, sensitivity, and uncertainty analysis.
- `relic\orchestration\`: Graph IR, validation, compilation, execution, and caching.
- `relic\parameters\`: Parameter manager, parameter database, validators, converters, and operations layer.
- `relic\scientific\`: Implementations of scientific processes across the planetary spheres.
- `relic\stochastic\`: Stochastic processes and uncertainty simulation.
- `relic\computation\`: Solvers and operators, parallel/distributed/GPU/JIT processing.
- `relic\utils\`: Physical constants access, tensor and soil utilities.
- `relic\validation\`: Validation framework, rules, and cross-domain verification.

Two extension paths have distinct owners:

- `relic\extensions\` owns Python and manifest-based extension discovery through
  `relic_extension.yaml`.
- `relic\coupling\plugin\` owns language-agnostic C ABI and shared-library
  coupling through `PluginHost` and `PluginFunctorAdapter`.

Native plugin lifecycle (owner `PluginHost`): `load` binds the library,
resolves the three C ABI symbols, and verifies the plugin's declared
`interface_version` against `COUPLER_PLUGIN_ABI_VERSION` before any exchange
(a mismatch raises `PluginABIError` and releases the handle); the manifest
`metadata.abi` record (language, standard, `__cplusplus`, compiler, target,
pointer width) is persisted with the load receipt. `retire` refuses new
native calls while calls already inside the library continue; `unload`
retires, drains the in-flight counter (the Python return from a C entry
point is the only completion signal the ABI offers), and releases the
handle through `FreeLibrary`/`dlclose` only when nothing is in flight. An
expired `drain_timeout` leaves the plugin retired with an observable receipt
so the release can be retried; a repeated unload is refused; `reload` binds
the same path again. `relic\extensions\loader.unload_extension` orders the
work per functor as simulator retirement (its `LifecycleReceipt` is consumed
into the summary), native retire/drain/release, and only then loader-record
clearing; a functor whose native release is incomplete keeps its record and
`complete` is False. The reference native plugin is
`examples\plugins\relay_plugin.cpp` (C++23, built by
`build_relay_plugin.bat`).

Both paths must adapt executable processes into the existing Functor and
simulator lifecycle. Neither path authorizes a second registry, scheduler, or
runtime.

### 2.1 Architecture Visualization

#### Selected Module Dependency Graph

The diagram shows selected package-level dependencies that are important for
extension and ownership decisions. It is not an exhaustive import graph.

All architecture diagrams in this document use the same visual contract. Node
fill denotes only a semantic role (`owner`, `runtime`, `contract`, `evidence`,
or `external`); it never denotes implementation status. Status is always
written as an explicit prefix and reinforced by border style:

| Status | Meaning | Border contract |
| --- | --- | --- |
| `[I]` | Implemented with executable evidence | solid, `#4B6F5A` |
| `[N]` | Not implemented; fail-closed | dashed, `#9B6435` |
| `[H]` | Historical or pre-final-tree observation | dotted, `#6C5B7B` |
| `[M]` | Measured and bound to artifact/hash | thick, `#315A78`, artifact/hash named |
| `[X]` | Rejected, invalid, or superseded | dash-dot, `#8A4545`, explicit `X` label |

The theory labels `THM`, `SPEC`, and `HYP` are orthogonal to these implementation
states. A theorem label is not executable evidence, and `[M]` is invalid without
the named measurement artifact and source identity.

```{mermaid}
flowchart TB
    subgraph "relic/"
        CORE["[I] core/"]
        SCI["[I] scientific/"]
        PARAM["[I] parameters/"]
        ORCH["[I] orchestration/"]
        COUP["[I] coupling/"]
        DATA["[I] data/"]
        EXT["[I] extensions/"]
        MODELS["[I] models/"]
        MON["[I] monitoring/"]
        OPT["[I] optimization/"]
        COMP["[I] computation/"]
        VALID["[I] validation/"]
        IFACE["[I] interfaces/"]
        STOCH["[I] stochastic/"]
        UTILS["[I] utils/"]
        CONTRACTS["[I] contracts/"]
    end

    SCI --> CORE
    SCI --> PARAM
    SCI --> UTILS
    COUP --> CORE
    COUP --> SCI
    COUP --> PARAM
    ORCH --> CORE
    ORCH --> VALID
    OPT --> CORE
    OPT --> PARAM
    OPT --> STOCH
    COMP --> CORE
    IFACE --> CORE
    IFACE --> PARAM
    IFACE --> OPT
    CORE --> PARAM
    CORE --> UTILS
    CORE --> CONTRACTS
    VALID --> CORE
    VALID --> PARAM
    DATA --> UTILS
    EXT --> CORE
    EXT --> PARAM
    EXT --> ORCH
    MODELS --> CORE
    STOCH --> CORE

    class CORE,PARAM,ORCH owner
    class SCI,COUP,MODELS,OPT,COMP,STOCH runtime
    class VALID,CONTRACTS contract
    class DATA,EXT,IFACE,MON,UTILS external
    class CORE,SCI,PARAM,ORCH,COUP,DATA,EXT,MODELS,MON,OPT,COMP,VALID,IFACE,STOCH,UTILS,CONTRACTS statusI

    classDef owner fill:#DCE8F2,color:#17212B
    classDef runtime fill:#E1ECE5,color:#17212B
    classDef contract fill:#E8E1EC,color:#17212B
    classDef evidence fill:#F2E6D8,color:#17212B
    classDef external fill:#ECEFF1,color:#17212B
    classDef statusI stroke:#4B6F5A,stroke-width:2px
    classDef statusN stroke:#9B6435,stroke-width:2px,stroke-dasharray:7 4
    classDef statusH stroke:#6C5B7B,stroke-width:2px,stroke-dasharray:2 3
    classDef statusM stroke:#315A78,stroke-width:4px
    classDef statusX stroke:#8A4545,stroke-width:2px,stroke-dasharray:8 3 2 3
```

#### Core Execution Chain Flowchart

```{mermaid}
flowchart LR
    A["[I] Scientific Contract\n(abstractions.py)"] --> B["[I] Canonical Bridge\n*_bridge.py"]
    B --> C["[I] FunctorBase\n(functor_base.py)"]
    C --> D["[I] Register\n(registration.py)"]
    D --> E["[I] Dependency Resolution\n(dependency_graph.py)"]
    E --> G["[I] Step Transaction\n(execution_loop.py)"]
    G --> F["[I] Group Calls: fast, medium, slow\n(execution_group.py)"]
    F --> H["[I] Commit per Group Call\n(StateManager.apply_updates)"]
    H --> J["[I] Step Accept or Rollback\n(execution_loop.py)"]
    J --> K["[I] Post-Accept Delivery\n(checkpoint, external intents)"]
    H --> I["[I] Diagnostics\n(diagnostics.py)"]

    class B,C,D,E,H owner
    class F,G,J runtime
    class A contract
    class I evidence
    class K external
    class A,B,C,D,E,F,G,H,I,J,K statusI

    classDef owner fill:#DCE8F2,color:#17212B
    classDef runtime fill:#E1ECE5,color:#17212B
    classDef contract fill:#E8E1EC,color:#17212B
    classDef evidence fill:#F2E6D8,color:#17212B
    classDef external fill:#ECEFF1,color:#17212B
    classDef statusI stroke:#4B6F5A,stroke-width:2px
    classDef statusN stroke:#9B6435,stroke-width:2px,stroke-dasharray:7 4
    classDef statusH stroke:#6C5B7B,stroke-width:2px,stroke-dasharray:2 3
    classDef statusM stroke:#315A78,stroke-width:4px
    classDef statusX stroke:#8A4545,stroke-width:2px,stroke-dasharray:8 3 2 3
```

#### Parameter System Four-Layer Call Architecture

```{mermaid}
flowchart TB
    L4["[I] Layer 4: Non-Functor Code\ninjected manager.get() or bootstrap global"]
    L3["[I] Layer 3: Scientific Modules\nself.get_parameter_value()"]
    L2["[I] Layer 2: Manager SSOT\nmanager.get()"]
    L1["[I] Layer 1: Database Internal\ndatabase_manager.get_parameter()"]

    L4 --> L2
    L3 --> L2
    L2 --> L1

    YAML[("[I] YAML Database\nparameters/database/")]
    L1 --> YAML

    class L2,L1,YAML owner
    class L3 runtime
    class L4 external
    class L4,L3,L2,L1,YAML statusI

    classDef owner fill:#DCE8F2,color:#17212B
    classDef runtime fill:#E1ECE5,color:#17212B
    classDef contract fill:#E8E1EC,color:#17212B
    classDef evidence fill:#F2E6D8,color:#17212B
    classDef external fill:#ECEFF1,color:#17212B
    classDef statusI stroke:#4B6F5A,stroke-width:2px
    classDef statusN stroke:#9B6435,stroke-width:2px,stroke-dasharray:7 4
    classDef statusH stroke:#6C5B7B,stroke-width:2px,stroke-dasharray:2 3
    classDef statusM stroke:#315A78,stroke-width:4px
    classDef statusX stroke:#8A4545,stroke-width:2px,stroke-dasharray:8 3 2 3
```

## 3. Core Execution Main Chain

RELIC Main Execution Path:

```text
Abstract Contracts (core/abstractions.py)
  -> Bridge Functors (core/functor/*_bridge.py; functor_bridges.py re-exports)
  -> FunctorBase (core/functor/functor_base.py)
  -> DependencyGraphResolver (core/dependency_graph.py)
  -> RELICSimulator (core/simulator/main.py)
  -> Execution group lowers typed outputs to UpdateItem records
  -> StateManager.apply_updates(...) for each executed group call
     (each Strang sub-execution is its own call)
  -> Validation and Diagnostics
```

Graph entry path: `GraphIR -> compile_graph(...)` (validation, optional
unit-adapter lowering, topological `ExecutionStep` list, structural audit,
strict compile-run witness, plan seal; no state advance) `-> GraphExecutor`
(verifies seal and witness, builds the planned Functors, binds the call
governor, and passes each step's `config.execution_group` to
`register_functor` as the stage declaration; `validate_graph` rejects any
value other than `fast`, `medium`, `slow`, or `None` in strict and
non-strict validation alike, with the message `register_functor` raises)
`-> RELICSimulator.step(...)`,
the only scientific state advance. Each prepared run uses one fresh
`RELICSimulator`. The strict compile-run witness registers the Functors into
a separate registration-only simulator that has no StateManager and never
steps. Its call to the existing registration owner uses the private
`_prepare_device=False` contract, retaining class, bridge, port, writer,
dependency, transaction and publication checks while deferring device
preparation to actual execution registration. Compile and preflight preserve
the device and existing autograd association of borrowed ParameterManager
bindings. The private flag is included in coordinated registration votes.
Plans compiled with
`strict=False` carry a skipped witness, and `GraphExecutor` refuses to execute
them.

Declared input projection: inside the call governor each planned Functor
kernel receives only the state keys its `ExecutionStep` binds
(`inputs.<port>.link` or `inputs.<port>.state`), the keys that DIRTY input
hashes and cache keys already use (`GraphExecutor._declared_state_keys`). An
optional input the step does not bind is absent for the kernel even when a
forcing boundary, a declared state or another node holds that name; other
nodes' outputs and the bookkeeping keys `time`, `step` and `state_var` are
absent as well. What a kernel reads from the shared state is therefore a
subset of the bindings its step declares; a bound input need not be read (a
kernel that accepts alternative inputs uses one of those bound and ignores
the others). Provider outputs passed as `subprocess_outputs` follow the
Functor's dependency declaration (`get_dependencies()`: the link sources of
the step merged with any static declaration), not this projection.
`compile_graph` reports the projection of every executable node
and its unbound optional inputs, each with the graph members that hold the
name, under `report.metadata["input_projection"]` (strict and non-strict
compilation; a disclosure, not a validation issue). State validation, the
local conservation hook and the validation pipeline of `FunctorBase.__call__`
still read the whole shared state. A `RELICSimulator` driven without a
compiled plan has no declared bindings and keeps name-based reads.

Main Chain Constraints:

- Business/scientific logic is computed exclusively within Functors.
- Prognostic state writes are handled by `StateManager.apply_updates(...)`; diagnostic
  ingestion remains a separate StateManager path.
- Inside each rate group the order is a topological order of the in-group
  required and present optional edges (`DependencyGraphResolver`, ties by
  registration index). Groups run fast, then medium, then slow; cross-group
  and lagged dependencies are not ordered by the graph.
- The `orchestration` layer compiles GraphIR into a sealed `ExecutionPlan` and
  prepares it: `GraphExecutor` builds and wraps the planned Functors and
  registers them into one `RELICSimulator`, which owns group assignment,
  per-group ordering, stepping, and the per-group-call
  `StateManager.apply_updates(...)` commits. The orchestration layer never
  advances or commits scientific state itself.
- Fast, medium, and slow groups apply their updates when each group executes. There is
  no second, single end-of-step state commit after all rate groups.

### 3.1 Seven Layers and the Control Plane (Owner Map)

The mathematical manuscript (`docs\source\RELIC_Mathematical_Foundations_Revised_v2_CN.tex`,
chapter "七层数学基础" and appendix "统一符号") partitions RELIC into seven layers plus an
orthogonal control plane. This section binds each layer to its canonical owner in the
live tree. It extends the main chain above; it does not introduce a second registry,
scheduler, or evidence framework. Symbols follow the manuscript: `Psi_ker` is the raw
kernel (`forward`), `Psi_gov` the governed operator (`FunctorBase.__call__`), `U_i`
the UpdateItem proposal set, `D_i` diagnostics, `E_ext` external intents, `T_g` a group
subtransaction, `T_step` the accepted-step outer transaction, `Omega_e` the committed
provider view at epoch `e`, and `G_decl` / `G_res` / `G_exec` / `G_phys` the declared,
resolved, executable, and physical-exchange graphs.

| Layer | Mathematical object | Inputs and outputs (units) | Canonical owner (symbol) | Legal entry and forbidden bypass | Error / compensation boundary | Status | Tests and exclusions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 Semantic | parameter space `Theta`; variable contracts; unit / support / time morphisms | YAML spec (`value`, `units`, `bounds`, `dtype`) to resolved values; CF names; support and calendar | `relic\parameters` (`ParameterManager`, `get_parameter_value`), `relic\contracts\contract_loader.py` (`ContractRegistry`), `relic\data\standards\` (units, dimensionality), `relic\coupling\scales\temporal\` (overlap weights) | `self.get_parameter_value(...)` inside Functors, `get_global_manager().get(...)` outside; forbidden: hardcoded constants, `manager.cache` mutation, second schema registry | invalid key / unit / bounds raise before any state write; conservation tolerances come from `validation\conservation.yaml`, the accuracy of scale-mapping operators from the scale precision policy (`coupling.scales.defaults`, Section 20.3) | [I] | `test\integration\parameters`, `test\unit\coupling\scales\temporal`, `test\audit\yaml_dimension_unit_mapping_v3.test.py`; excluded: unit inference from code |
| 2 Declaration | `G_decl` (declared dependencies), typed ports, assertions | `FunctorMetadata` + `relic_extension.yaml` + GraphIR nodes/links to validated declaration | `relic\orchestration\graph_ir.py`, `node_registry.py` (`PortSpec`, `SSOTCatalog`), `validator.py` (`assertion` field), `relic\core\functor\functor_types.py` | `compile_graph(..., strict=...)` / `validate_graph(...)`; forbidden: custom compilers, nested `strict=False` weakening explicit outer requests | declaration errors reject before resolution; optional targets may be missing, required targets are refused up front | [I] | `test\orchestration\orchestration_core.test.py`, `ncs_compiler_mvp.test.py`, `test\unit\test_dependency_graph.py` |
| 3 Resolution | provider identity `nu_i` (implementation / contract / binding digests), `Omega_e`, epoch `e`, `P_e`, `G_res` and `G_exec` | registry + declarations to immutable `CommittedProviderView`, execution orders, DIRTY-tracked plan | `relic\core\simulator\registration.py` (`_ProviderLifecycleState`, `_publish_provider_view`), `relic\core\dependency_graph.py`, `relic\orchestration\compiler.py` / `executor.py` (cache keyed by digests) | `register_functor` / `unregister_functor` / `apply_topology_transition`; a step reads through one `ProviderViewToken`; forbidden: mutating `functors` / `_groups` outside the owner (the next step is refused) | prepare before mutate, CAS at publication, snapshot compensation, `LifecycleReceipt` with `rolled_back` / `compensation_incomplete`; dispose waits for readers and device fences | [I] | `test\core\simulator_provider_lifecycle.test.py`, `simulator_topology_transition.test.py`, `device_fence_lifecycle.test.py`, `test\orchestration\graph_executor.test.py`; excluded: tensor-shape dimension change |
| 4 Execution | `Psi_gov = post o Psi_ker o pre` with validation and local conservation; proposal classification into `U_i`, `D_i`, `E_ext` | state view, `time`, `dt` to UpdateItem / diagnostic / ExternalIntent | `relic\core\functor\functor_base.py` (`__call__`, `set_call_governor`), bridges in `core\functor\*_bridge.py`, `relic\core\state\types.py` (`partition_outputs`) | normal scheduling calls the Functor object; `forward` is the kernel API only; forbidden: `compute_rate` / `compute_flux`, direct Functor-to-Functor calls, an intent that is also a declared output | kernel errors are wrapped as `ExecutionError`; local conservation hook may warn or raise per policy; raw `forward` tests prove nothing about governance | [I] | `test\audit\forward_only_functor_api.test.py`, FunctorBase suites, `test\core\test_functor_substream_l1.py`; excluded: global smoothness across events / clipping |
| 5 Transaction | `T_g` (`Commit_h` per group), `T_step` (savepoint / restore of state, runtime, Functor state, provenance, supported RNG, forcing cursor), checkpoint / replay partial maps (exact, compatible, migration) | accepted proposals to committed prognostic state; accepted step to deliveries, checkpoints, receipts | `relic\core\state\manager.py` (`apply_updates`, savepoints), `relic\core\simulator\execution_group.py`, `execution_loop.py` (`step_simulation`, deliveries), `checkpoint.py` (single serializer, certificates) | group commit happens inside each group; deliveries and checkpoints run only after acceptance and are retried by `retry_external_deliveries`; forbidden: a second end-of-step commit, hidden state writes, checkpoint inside the scientific savepoint | rejected step restores everything captured; I/O failure never claims the science step failed; multi-rank checkpoint is refused until a coordinated format exists | [I] | `test\unit\test_simulator_l1.py`, `test\core\forcing_cursor_restart.test.py`, `checkpoint_compatible_restart.test.py`, `checkpoint_graph_migration.test.py`, `simulator_registration_and_step_guards.test.py` |
| 6 Scientific structure | `G_phys` (stocks and compatible fluxes), DEC cochains and Hodge, dissipative pH `(J - R)`, GENERIC `L / M`, reaction `N_h r_h`, ledgers `C_ledger`, BIHSS sub-semantics | declared structure to assembled operators, certificates, rule verdicts | `relic\computation\operators\dec\`, `iphs.py`, `iphs_certificate.py` (only level owner), `relic\validation\framework\rules\` | claims are requested explicitly (positive `required_level`, strict assembly, `phase8_strict`); a mature empirical law may be the single production law; forbidden: inferring `H`, efforts, or affinities from empirical code; treating `G_exec` incidence as a physical conservation proof | missing evidence keeps a claim at `[N]`; certificates validate their own contracts | [I] for existing owners; strict runtime claims per certificate | `test\unit\validation\framework\rules\`, `test\audit\phase8_scientific_inventory.test.py`, IPHS operator suites; excluded: arbitrary reaction networks (G07 [N]), Level 4/5 for non-claimed processes |
| 7 Evidence | claim status, owner, tree / hash, test, artifact, scope and exclusion | run outputs to fingerprinted evidence bundle | G8 command spec and validators under `scripts\` (`02_validate_g1_evidence.py`), `analysis_results\` (refreshed only under H04 authorization), repair receipts in `output_audit_tmp\relic-repair-20260912\` | evidence is produced by the owners' commands; forbidden: a second claim framework, hand-edited `passed` fields, counting group runs as a full run | stale fingerprints mark runs `[H]` / failed; `[H]` rows in the manuscript claim matrix are historical | H04 pending | `test\audit\audit_gate_suite.test.py`; excluded: any statement of whole-system formal verification |
| Control plane (orthogonal) | `Gamma_ctrl`: registry, plan, cache, resource handles, epoch, receipts; transition relation `->` and normalization to quiescence | lifecycle requests to receipts and epochs; never a scientific state write | `registration.py` (lifecycle), `relic\computation\distributed\launcher.py` (`coordinate_transaction`), `relic\core\device_policy.py` (`DeviceFence`), `relic\coupling\plugin\host.py` (`PluginHost`), `relic\extensions\loader.py` (`unload_extension`) | lifecycle changes are recoverable (receipts, compensation, drain); accepted scientific history is not locally reversible (needs checkpoint / migration / replay); forbidden: swapping providers mid-step, releasing a native library or a retired view while work is in flight | crash-stop ranks, partitions, timeouts, and dispose failures are observable outcomes, never silent | [I] | `test\computation\distributed\coordination.test.py`, `test\coupling\plugin_host_lifecycle.test.py`, `test\unit\extensions\test_loader_unload.py` |

Boundaries that must stay distinct:

- `G_decl`, `G_res` / `G_exec`, and `G_phys` are three graphs. A software incidence
  structure in `G_exec` is not a physical conservation proof; budgets are verified on
  `G_phys` ledgers.
- Normal execution writes through `Psi_gov`; `forward` is only the kernel. Outputs are
  partitioned three ways (`partition_outputs`): state proposals go to `T_g`, diagnostics
  are ingested separately, external intents are delivered after `T_step` accepts.
- Group commit (`T_g`), the accepted-step transaction (`T_step`), post-commit checkpoint
  and delivery I/O, and the external world `Omega_ext` are four scopes. Only the first
  two are inside the in-memory savepoint; the last two are retried and receipted, never
  rolled into the scientific step.
- Claims about the strict scientific-structure layer are per process and per certificate;
  a mature model without those claims is complete when its own contracts hold.

## 4. Core Layer Detailed Description

### 4.1 The Four Abstractions

File: `relic\core\abstractions.py`

Four conceptual interface abstractions:

- `PhysicalField`
- `BiologicalProcess`
- `TransportMechanism`
- `ScaleCoupling`

Implementation Requirements:

- Any scientific Functor must explicitly correspond to one of the above four categories.
- Free-form implementations that "run but lack an abstraction semantic" are strictly prohibited.

:::{figure} ../../assets/readme/relic_scientific_contracts.png
:alt: Four RELIC scientific abstractions and their canonical bridge Functors converging on FunctorBase.
:width: 95%
:align: center

Every scientific module satisfies one abstract contract through its canonical
bridge; `FunctorBase` provides the shared `torch.nn.Module` lifecycle.
:::

### 4.2 Bridge Layer and Base Class

Canonical bridge implementations:

- `relic\core\functor\physical_field_bridge.py`
- `relic\core\functor\biological_bridge.py`
- `relic\core\functor\transport_bridge.py`
- `relic\core\functor\scale_coupling_bridge.py`

`relic\core\functor\functor_bridges.py` is the stable re-export facade; it is not
a second implementation. The shared base class is
`relic\core\functor\functor_base.py`.

Bridge Classes:

- `PhysicalFieldFunctor`
- `BiologicalProcessFunctor`
- `TransportMechanismFunctor`
- `ScaleCouplingFunctor`

Base Class:

- `FunctorBase(nn.Module, abc.ABC)`

Key Constraints:

- Unified parameter access and validation entry point.
- Unified dependency declaration entry point: `get_dependencies()`. A concrete
  override returns `self._merge_declared_dependencies(static)`, the single merge
  owner: a name declared on the instance through `config["dependencies"]`,
  `set_dependencies(...)`, or `add_dependency(...)` joins `required` unless the
  static declaration already lists it, in which case its static category is
  kept. The result is normalized by
  `relic.core.dependency_graph.normalize_dependency_declaration` (Section 4.3).
  Optionality comes only from this declaration; there is no
  `optional_dependencies` attribute channel.
- Unified scientific metadata entry point.
- Unified conservation check entry point.
- The explicit `get_parameter_value(...)` overrides in biological and scale
  bridges are required MRO disambiguation. They must not be removed as
  apparently duplicated wrappers.
- A `TransportMechanismFunctor` subclass declares its descriptive label as the
  class attribute `TRANSPORT_TYPE` (one of `TransportMechanism.TRANSPORT_TYPES`);
  the bridge reads it before `TransportMechanism.__init__`, and a missing
  declaration or a `transport_type=` / `diffusivity=` constructor keyword raises
  `TypeError`. No label implies a scalar diffusivity: `diffusivity` is `None`
  unless the process law uses one constant scalar coefficient, which the
  subclass sets after its parameter reads through the only setter
  `set_scalar_diffusivity(value)` (finite and positive; a live learnable
  parameter object is stored as is). `FickianDiffusion` calls it only for a
  scalar D (a number or a 0-d tensor, stored as the same object); a tensor
  field D leaves `diffusivity` `None`. The transport kernels `apply_diffusion`
  and `advance_advection_diffusion` (and the `transport.diffusion` /
  `transport.advection_diffusion` operators) require
  `diffusivity_layout="scalar_field"` or `"directional"` for a tensor
  diffusivity with `ndim >= 1`; the shape never selects a reading, and the grid
  Peclet diagnostic uses the same per-axis reading. The base class provides no generic
  boundary-flux law: `boundary_conditions()` raises unless a concrete transport
  Functor implements its own.
- A `PhysicalFieldFunctor` subclass declares the class attribute `SPATIAL_DIMS`
  (1, 2, or 3 for the coordinate dimension of its spatial query, `None` for a
  point model whose state lives in StateManager); a missing declaration and the
  constructor keywords `field_name`, `units`, `default_value`, `spatial_dims`,
  and `temporal_dependence` raise `TypeError`. The field identity is the
  declared `output_variables` and its units are the variable contracts'; the
  bridge provides no default `forward`, `get_value_at`, or `evolve`, and a point
  model's `get_value_at` raises `NotImplementedError`. A bool or non-integer
  `SPATIAL_DIMS` raises `TypeError`, an integer outside 1-3 `ValueError`;
  `update_field_to_flux(field_update, dt, field_name)` returns
  `{field_name: field_update / dt}` (all three arguments required).
- `BiologicalProcess.parameter_sensitivity(parameter_name, state, *, time, dt,
  ...)` replaces the perturbed tensor member (an `nn.Parameter`, buffer, or
  tensor attribute, by its qualified name) only inside
  `torch.func.functional_call` and evaluates the governed call
  `self(state, time=time, dt=dt)`, so the Functor is never modified. Without
  an explicit step the step follows the perturbed tensor's dtype (Press et al.
  1992, Sect. 5.7): `h = eps_m^(1/2) x_c` (forward) or `eps_m^(1/3) x_c`
  (central), `eps_m = torch.finfo(dtype).eps`, `x_c = |theta|` for a nonzero
  parameter and `optimization.sensitivity.finite_difference.zero_parameter_scale`
  at zero, and `h` is adjusted so that `theta + h` is exact; an explicit
  `absolute_eps` and/or `relative_eps` takes precedence as the largest of
  `absolute_eps` and `relative_eps * x_c` given. Only real members
  (`named_parameters()`, `named_buffers()`, tensors in `__dict__`) are
  perturbed: a value supplied by a class `__getattr__` raises `TypeError`, an
  unknown name `KeyError`. The parameter and buffer versions are compared
  around every evaluation, and a forward that mutates any of them in place
  raises `RuntimeError` (invariant I1; reported, never restored). Concrete
  Functors do not override it. For a stochastic Functor (a `FunctorSubstreamMixin` substream
  owner) every evaluation starts from the substream position saved before the
  first one (common random numbers; Glasserman 2004, Sect. 7.1), and the
  position is restored after the last, so the call does not advance the
  stream; a private generator outside the substream owner is not covered and
  must move to the substream.
- A learnable parameter has one owner, the ParameterManager, which keeps one `nn.Parameter` per key in `nn_parameters`. A Functor attribute that is, by object identity, a binding of the Functor's own manager is held by reference (`FunctorBase.__setattr__`): it is not a module parameter, so the Functor's `parameters()`, `state_dict()`, `.to()`, `.double()`, and `to_device` never copy, save, or convert it, and every holder reads the same theta (the gradients of all holders accumulate in that one binding). `manager_binding_key(value)` names the key of such a binding and `held_manager_parameters()` lists the bindings a Functor holds, including references in nested `Mapping` keys and values, lists, tuples, sets and frozensets. Container identities prevent repeated traversal of aliases and cycles; ordinary objects and other Functors are not recursively inspected, and a clone, view or binding owned by another manager is not accepted. There is no other parameter-listing API (`get_torch_parameters` is removed). `parameter_sensitivity` raises `TypeError` for a held binding, whose sensitivity goes through the manager calibration path (`get_trainable_parameters`). A durable checkpoint (format version 3, Section 4.4.3) saves and restores the value of a held binding through its manager in the Functor entry's `parameter_bindings`; registration relocates a held binding to the Functor's effective device in place, per key, and a binding held by two registered Functors on different devices raises `ValueError` (Section 4.4.2).
- A `ScaleCouplingFunctor` subclass passes `source_scale`, `target_scale`, and
  `coupling_type` to the bridge explicitly in its `__init__` (a class configured
  per instance reads them from its own configuration, where they are
  required); the bridge invents none of them and a missing one raises
  `TypeError`. `ScaleCoupling` and the spatial, temporal, and hierarchical
  scaling bases have no default method arguments (including the `method` of
  `regrid` and `resample`). The bridges provide no generic temperature law and
  no `enzyme_kinetics` label; a process that needs a temperature response
  implements its own documented law.
- Scientific inputs come only from the state: the governed call accepts
  `time` and `dt` (two positional arguments or keywords) and the integration
  control keywords; an `environment` keyword or any other positional argument
  raises `TypeError`. There is no environment input channel in `relic`, and
  the test helper `call_forward` provides none.
- Validation strictness is configuration: `strict_validation` (bool, default
  True) is read on every call. A Functor keeps no validation state; the
  findings of a non-strict call go into the caller's report
  (`__call__(..., validation_report=list)`, owned for a run by
  `RELICSimulator.get_validation_findings()`), and a non-strict downgrade
  without such a report raises `ExecutionError`.
- Differentiability boundary: `nn.Module` membership and tensor state make
  gradients available only along the executed tensor path. Bound clipping
  (`torch.clamp`, zero gradient for clipped elements; detached when
  `PhysicalConstraintManager.preserve_gradients` is false), event and
  rate-group triggers, adaptive accept/reject and step-size choice, sub-step
  counts, discrete topology transitions, and detached diagnostic or audit
  values are not differentiable. Graph caches never replay autograd-bearing
  values. A gradient claim has to be checked on the executed graph.
- Registration requires a `FunctorBase` instance and exactly one canonical
  bridge matching the declared `ProcessCategory`. Non-Functor objects,
  unbridged scientific classes, and bridge/category mismatches raise
  `TypeError` before registration mutates simulator state. The existing
  `_require_functor_base(...)` and `_require_category_bridge(...)` checks in
  `core\simulator\registration.py` own this boundary.

### 4.3 Dependency Graph

File: `relic\core\dependency_graph.py`
Core Class: `DependencyGraphResolver`

Key Interfaces:

- `add_functor(...)`
- `add_edge(...)`
- `topological_sort(...)`
- `get_parallel_groups()`
- `nodes` (canonical `functor_id -> ExecutionNode` mapping)
- `normalize_dependency_declaration(declaration, *, owner)` returning
  `DeclaredDependencies` (`required`, `optional`, `lagged`, `same_step`,
  `canonical()`): the single normalizer of a `get_dependencies()` declaration,
  shared by the resolver and the simulator (registration edges and contract
  digest, missing-provider check, group execution orders, cross-group lag
  governance). It accepts `None`, a list or tuple of names (all `required`), or
  a mapping whose keys are a subset of `required`, `optional`, and `lagged`
  with list or tuple values. A repeated name inside one category is kept once.
  A non-string name, a non-sequence category value, or any other declaration
  type raises `TypeError`; an unknown category key (for example a
  `{name: "optional"}` shorthand), an empty name, or a name in two categories
  raises `ValueError`.

Purpose:

- Detect dependency cycles.
- Compute a valid topological execution order.
- Break ties by registration order. When several nodes are ready at once,
  the one sorting path -- Kahn's algorithm over the declared adjacency with a
  min-heap on the Functor registration index (dependents released in the same
  order) -- takes the earliest registered Functor. The optional NetworkX graph
  (`use_networkx`) is an introspection mirror and is never sorted: a mirror
  that lost a node would shorten the order. The former NetworkX path and its
  fallback to Kahn were removed on 2026-09-26. The registration
  index is saved in the mutation snapshot, so a restored checkpoint keeps
  the order (2026-09-21).
- Report dependency levels. `get_parallel_groups()` groups Functors by their
  longest-path level in the sorted order; Functors on one level have no
  same-step edge between them. The list is a candidate grouping used for
  reporting (`get_execution_stats()`, `max_parallel_width`); ties inside the
  sorted order still follow the registration index. The execution loop runs every group call as a
  sequential loop over its order, so no thread, process, or rank runs
  Functors of one level concurrently. GPU parallelism inside one Functor's
  tensor operations and cross-rank reductions of updates are separate
  mechanisms.
- The sort enforces every resolved same-step edge,
  including present optional providers. Missing optional providers are omitted
  only when requested; lagged declarations do not constrain this step's DAG.
  In-degree accounting and dependent release use the same resolved edge set.
- Preserve each consumer's original required and optional dependency
  declarations when a provider is removed. Required-provider removal is
  rejected before mutation; optional edges are unbound but their declarations
  remain so that re-registration can bind them again.

### 4.4 Simulator

Main File: `relic\core\simulator\main.py`

Core Class:

- `RELICSimulator`

Key Methods:

- `initialize(...)`
- `initialize_state(...)`
- `register_functor(...)`
- `_execute_group(group_order, dt, *, start_time, composition=None)` --
  Runs one group call over the interval `[start_time, start_time + dt]`;
  `start_time` is required and must be a finite real number.
  `composition` is the `GroupComposition` (`execution_group.py`) of the
  fast-group composition the call belongs to. It owns dependency outputs,
  per-Functor merged diagnostics, per-writer stage boundary rates, and the
  private whole-interval boundary-rate accumulator. `_execute_fast_group`
  shares it across all sub-executions and finalizes boundary rates after
  successful completion. Without it the group call creates and finalizes
  its own empty composition.
- `step(...)`
- `run(...)`
- `run_ensemble(...)` -- `num_samples` and `outlier_detection` are given by the
  caller; the convergence diagnostic runs when `num_samples` reaches
  `optimization.uncertainty.ensemble.convergence_min_members`.
- `create_checkpoint(checkpoint_path)` -- Writes one explicit durable checkpoint
  and returns its integrity receipt.
- `restore_checkpoint(checkpoint_path, *, compatibility_certificate=None,
  migration_certificate=None)` -- Strictly validates schema, version, source
  identity, and integrity before restoring and returns its receipt. Without a
  certificate the restore is exact; a `RestartCompatibilityCertificate` allows
  only certified contract differences, and a `GraphMigrationCertificate` maps
  one saved graph onto one target graph inside a restricted family
  (Section 4.4.3). It is not arbitrary graph migration.
- `get_functor_summary()` -- Returns a metadata summary (inputs, outputs, scale, kind, etc.) of all registered Functors.
- `get_variable_provenance(var_name=None, *, include_retention=False,
  include_commit_context=False)` -- Queries the retained writer-provenance
  window; commit context and StateManager commit indices are opt-in.
- `get_splitting_order_report()` -- Returns a fresh dictionary with the
  effective order of the fast-group composition, derived from the committed
  provider view and the declared `FunctorMetadata.sub_flow_accuracy` of the
  fast Functors (Section 4.4.1); nothing is stored.

- `get_composition_plan()` -- Pure export of the committed provider view as
  simulator-owned `IntegrationPlan` stages; it advances no state and stores no
  second plan. GraphExecutor can compare a declared plan with this export
  (Section 5.2).

Modular Subfiles (Maintaining Separation of Concerns):

- `relic\core\simulator\registration.py` -- Executes output variable overlap detection (`_detect_output_overlap`) and owns transactional registration and retirement rollback. Both operations use one private snapshot of the existing owners. On failure or cancellation, compensation attempts module membership, dependency graph, group/order and bootstrap flags, and the conservation ledger independently. A compensation failure is reported together with the original error in an exception group; it is never reported as successful recovery. Device migration is the final owner commit action, after graph, ledger, and metadata work. Its compensation attempts all captured nested device owners, including when another owner's compensation is cancelled. The same owner mints provider identity and publishes the committed view: each registered provider carries one lifecycle `provider_uid` (never reused after retirement) and three content digests, the implementation digest (canonical SHA-256 of `FunctorBase.get_kernel_identity()`, or a narrowed non-strict identity with the rejection reason when the kernel is untraceable), the contract digest (class, bridge, metadata, dependency declaration normalized by `normalize_dependency_declaration`, canonical config, parameter namespace; it is computed before any owner mutates, so an invalid dependency declaration rejects the registration without a state change), and the binding digest (name, execution group, bound parameter values, bootstrap flag). `_publish_provider_view` is the single publication point: it recomputes group orders, applies cross-group promotion, refuses publication while a required dependency is unmet, compare-and-swaps the epoch at the swap point, freezes an immutable `CommittedProviderView` (epoch, `key -> ProviderIdentity`, groups, execution orders, registry signature), and retires the previous view. Every registration and retirement appends a `LifecycleReceipt` (operation, UID, digests, epochs, managed delta per owner, completed and compensated action identifiers, recoverable resources, status `committed`/`rolled_back`/`compensation_incomplete`). The empty registry is published as epoch 1 at simulator construction; checkpoint restore republishes under a fresh epoch. `RELICSimulator.get_committed_provider_view()`, `acquire_provider_view()`, `provider_epoch`, and `get_lifecycle_receipts()` are the read surface; `DependencyGraphResolver` never holds an epoch. The same owner commits declared hybrid topology transitions (`TopologyTransition`, `declare_topology_transition`, `apply_topology_transition`): guard and membership checks and the planned ledger audit run before any mutation, providers are retired and connected with publication deferred, the affine state remap `x^+ = P x^- + R eta` is one transactional StateManager write by the declared writer `topology:<transition>`, exactly one provider epoch is published, and a `LifecycleReceipt` records the ledger residual; a failure compensates the registry through the registration snapshot and restores the remapped variables. Transitions are requested by Functors as `ExternalIntent` operation `topology`, validated by the delivery owner while the step can still roll back, and committed after acceptance, so the requesting step runs on the view it started with.
- Reader admission and quiescent swap: a step transaction admits itself through `acquire_provider_view()`, which refuses a registry that differs from the committed view or dirty group membership, and holds one `ProviderViewToken` for the whole step. A view retired while readers hold tokens stays `pending_dispose`; the last token release, on whichever thread, runs `dispose_view`, so an old reader finishes entirely on its old view and a reader admitted after publication sees only the new one. Token release is idempotent and happens in the step `finally` on success, failure, or cancellation. A failed disposal is recorded as `dispose_failed` with its error and retried through `retry_dispose`; it never reverts the committed view. `wait_quiescent(epoch, timeout)` observes disposal.
- `relic\core\simulator\execution_group.py` -- Maintains `_step_provenance_detail` for variable write tracking.
- `relic\core\simulator\execution_loop.py` -- Freezes one committed provider view per step transaction (`acquire_provider_view()` before the first group; token release in the step `finally`) so fast, medium, slow, substeps, and compensation resolve providers and orders from `simulator._active_step_view`; group orders and `_groups_dirty` are no longer step-transaction attributes. Governs declared cross-group reads at view publication (`_warn_cross_group_dependencies`: sampling-lag and hold-lag warnings, optional slow-to-medium move, optional strict rejection) and computes the effective splitting order (`splitting_order_report`).
- `relic\core\simulator\scale_ops.py` -- Label-level scale consistency check over the retained outputs of the group being executed. Tier 1 uses `FunctorMetadata.spatial_scale` of the declaring Functor, parsed by `SpatialScale.from_value` (an unrecognized label raises instead of reaching Tier 2; only `UNKNOWN` counts as undeclared); if any checked output has a declared scale, only declared outputs are compared and undeclared tensors are ignored. Tier 2 (only when no checked output is declared) infers a label from the name tokens `basin`/`hillslope`/`patch` and then from tensor rank and element count. `UNKNOWN` never triggers regridding. A mismatch selects a label of maximal `_SCALE_PREFERENCE` rank; it provides no geometry. With auto-scale off and `strict_scale_detection` on, a mismatch raises; with auto-scale on, `regridding_ops.py` requires explicit `source_grid`/`target_grid` descriptors (the target with CF grid fields), selects the method from the catalogued `spatial_basis`, and regrids a working copy of the state (Section 8.3). The same file computes the per-group execution orders (`compute_group_execution_orders`): an empty group gets an empty order, and a missing provider, a cycle, or any other sort failure propagates with a note naming the group and its members. The previous orders stay in place, the groups stay dirty, and the registration owner rolls the publication back; no unsorted registration order is installed.
- `relic\core\simulator\dependency_ops.py` -- Execution-group classification.
  The fast and medium characteristic-time thresholds are the parameter-database
  entries `computation.scheduling.fast_group_max_characteristic_days` and
  `computation.scheduling.medium_group_max_characteristic_days`
  (`relic\parameters\database\relic\computation\scheduling.yaml`).
  `resolve_scheduling_thresholds()` reads them. Python holds no literal day
  thresholds.
- `relic\core\simulator\diagnostics.py`
- `relic\core\simulator\time_ops.py` -- Interval predicates
  (`should_execute_medium/slow` read tracker state and do not change it), the
  one interval commit `commit_rate_group_interval()`, and event trigger checks
  that fail fast: a trigger that raises or returns a value that is neither
  `bool` nor `int` aborts the step
  with `RuntimeError`.
- `relic\core\simulator\init_ops.py` -- Calendar configuration and interval initialization.
- `relic\core\simulator\state_init.py` -- Builds the initial-state schema from
  declarations only; nothing is inferred from a variable name and there is no
  generic fallback range. A variable with an SCM contract takes its contract
  specification, and a contract alias takes the specification of its canonical
  contract under its own name. Any other variable takes units and bounds from
  its SSOT catalog entry, units and the ParameterManager keys of its lower and
  upper bound (read in those units) from
  `validation.numerical.state_variable_bounds[<variable>]` (current
  configuration: no entry; `daylength`, the former only entry, is declared by
  its SCM contract), and units
  only from the simulator setting `state_units`. The contract replaces the
  other declarations; among the others, units that disagree, or bounds
  declared both by the catalog and by `state_variable_bounds`, raise
  `ValueError`, and `state_units` that disagree with the contract units raise
  as well. A variable without declared bounds is unbounded, and only a
  contract makes a variable conserved (through `accumulates`). After the
  contract overrides, one `ValueError` lists every variable that still has no
  units, and no StateManager is kept. The file also initializes the interval
  trackers and registers the test-only bootstrap Functors on an empty
  simulator when `enable_bootstrap_modules` is set and bootstrap registration
  is allowed (`RELIC_ALLOW_BOOTSTRAP_MODULES=true`, or, unless that variable
  is set false, a pytest process or the `allow_bootstrap_modules` setting); a
  disallowed request raises `RuntimeError`. An absent `bootstrap_functors` setting
  requests `bootstrap_functor_name`, else the parameter
  `computation.simulator_bootstrap.default_functor_name`; an explicit list,
  even an empty one, is used as given; an unregistered name raises `KeyError`.
- `relic\core\simulator\checkpoint.py` -- Sole owner of the durable checkpoint
  format, strict validation, atomic file commit, restore rollback, and receipt
  generation.

#### 4.4.1 Multi-Rate Spatiotemporal Execution Architecture

RELIC employs a **multi-rate execution model**, partitioning Functors into three speed groups:

| Speed Group | Execution Frequency            | Default Trigger Condition                     | Effective dt               |
| ----------- | ------------------------------ | --------------------------------------------- | -------------------------- |
| `fast`      | Every step                     | Unconditional                                 | `step_dt`                  |
| `medium`    | Configurable interval          | Every 86400s (1 day) or event triggered       | `t_future - t_last_medium` |
| `slow`      | Configurable interval/Calendar | Month boundary, configurable, or event driven | `t_future - t_last_slow`   |

**Group Assignment (at registration):**

The first `initialize(state_manager=..., dependency_graph=...)` call uses the
same `_initialize_time_tracking` owner as `initialize_state(...)`. Both entry
paths start medium and slow interval tracking at the configured initial time;
ordinary groups do not gain an implicit first-step bootstrap. Rebinding owners
on an initialized, running, or restored simulator preserves the accumulated
intervals. Explicit `force_initial_execution` and event triggers retain their
separate execution contracts.

`register_functor(functor, execution_group=...)` fixes the group once, before
any registration state changes, by one priority chain
(`dependency_ops.determine_execution_group`, called through
`registration._resolve_execution_group`); the first declared rule decides:

1. the `execution_group` argument of `register_functor`;
2. the Functor attribute `execution_group`;
3. `characteristic_time_days` `tau` compared with `T_f` and `T_m`: `fast` if
   `tau <= T_f`, `medium` if `T_f < tau <= T_m`, `slow` otherwise;
4. `FunctorMetadata.temporal_scale`, mapped by `_resolve_temporal_scale_group`
   (`TemporalScale.UNKNOWN` means no declaration);
5. the configuration entry `process_groups[functor_name]`, which fills in only
   for a Functor that declares none of rules 2-4;
6. `fast`, only when nothing is declared.

Every declared value is validated, also when a higher rule decides; the
scheduler has no silent fallback. An argument, attribute, or `process_groups`
value outside `fast`, `medium`, `slow`, a `characteristic_time_days` that is
not a finite non-negative real number of days, a `metadata.temporal_scale`
that `TemporalScale.from_value` rejects (`TypeError` for a value that is not
a string), and a `process_groups` setting that is not a mapping raise
`ValueError`, and the rejected registration leaves no Functor,
group member, dependency-graph node, or provider UID behind. `FunctorMetadata`
converts `temporal_scale` with `TemporalScale.from_value` when it is built:
`None` and `"unknown"` give `UNKNOWN` (no declaration); there are no aliases,
only member values (case-insensitive) parse;
any other unrecognized string, the empty string included, raises
`ValueError` there, and a value of another type raises `TypeError`. A
`process_groups` entry
that disagrees with the group of rules 2-4 raises `ValueError` naming the rule
and both groups: remove the entry, or pass the group as the `register_functor`
argument, which overrides the entry. An agreeing entry is accepted. A resolved
group that is not one of the simulator's groups is an invariant violation
(`RuntimeError`).

`T_f` and `T_m` are the parameters
`computation.scheduling.fast_group_max_characteristic_days` and
`computation.scheduling.medium_group_max_characteristic_days` in
`relic\parameters\database\relic\computation\scheduling.yaml` (current
configuration example: 1 day and 10 day). They classify processes. They are
not execution intervals: the medium and slow intervals come from
`medium_interval_seconds` and `slow_interval_seconds` or the calendar month.
`process_groups` never overrides the explicit argument, the Functor attribute,
the characteristic time, or the temporal scale.

**Configuration Keys (read via `_get_setting()`):**

| Key                                   | Default Value           | Purpose                                                            |
| ------------------------------------- | ----------------------- | ------------------------------------------------------------------ |
| `time_step`                           | `60.0` s (configuration default; single source `_DEFAULT_TIME_STEP_SECONDS` in `relic\core\simulator\main.py`) | Fast-group step; `RELICSimulator(None)` is the empty configuration `{}` |
| `max_time`                            | `SECONDS_PER_DAY` (86400 s) | End time of `run()` when no `end_time` is given |
| `medium_interval_seconds`             | `SECONDS_PER_DAY` (1 day, `relic.utils.time_constants`) | Trigger interval for the medium group (seconds)                    |
| `slow_interval_seconds`               | `None` (Calendar month) | Trigger interval for slow group; fallback to month if `None`       |
| `time_origin`                         | `None`                  | ISO 8601 date and time of t = 0 in `calendar_type` (for example `2000-01-01T00:00:00`); checked at construction; passed to every Functor whose `forward` declares `time_origin` (`execution_group._build_integration_kwargs`, like `calendar_type`); an external adapter reading forcing at the simulation time raises without it |
| `adaptive_dt_enabled`                 | `False`                 | Enable committed-state-change RMS step control with at most one rollback retry |
| `adaptive_atol` / `adaptive_rtol`     | YAML `computation.scheduling.adaptive.atol` / `.rtol` (`1e-6` / `1e-3`); a config key of the same name overrides | Scales used by the committed-state-change RMS metric |
| `adaptive_dt_min` / `adaptive_dt_max` | YAML `computation.scheduling.adaptive.dt_min_seconds` (`1.0`) / `dt_max_factor` x dt (factor `10`); a config key of the same name overrides | Lower and upper bounds for adaptive stepping; database values pass the same validation as overrides |
| `adaptive_monitor_vars`               | `None` (All)            | Monitor error norms strictly on specified variables                |
| `splitting_mode` | YAML `computation.scheduling.splitting_mode` (currently `lie`) | Validated `lie` or `strang`; unknown values raise at construction |
| `splitting_policy` | YAML `computation.scheduling.splitting_policy` (currently `warn`) | `warn` or `require_declared`; the latter refuses split Strang members not shown stateless when their sub-flow accuracy is undeclared |
| `fast_order` | YAML `computation.scheduling.fast_order` (currently `forward`) | `forward` dependency order or `reverse`; reverse refuses same-step fast dependencies |
| `substep_overrides` | YAML `computation.scheduling.substep_overrides` (currently `{}`) | Per Functor, exactly `{count: n}` or `{max_seconds: s}`; overrides metadata for each call |
| `multirate_exchanges` | YAML `computation.scheduling.multirate_exchanges` (currently `[]`) | Explicit `{id, variable, consumer_group, mode}` records; `medium`/`slow` consumers and `hold`/`interval_mean` modes |
| `resolve_cross_group_deps`            | `False`                 | Move `slow` consumers with an unacknowledged sampling lag into the `medium` group (from `slow` to `medium` only; a scheduled slow member moves only when `t_last_slow == t_last_medium`, otherwise the publication fails; fast-group reads are never changed) |
| `strict_cross_group_dependencies`     | `False`                 | Reject (roll back) a registration, retirement, or restore whose published plan has an unacknowledged cross-group read |

An attempted simulator step is one transaction over what is actually captured:
the StateManager transaction savepoint, each registered Functor's `state_dict`
(parameters, buffers, and `get_extra_state` payloads, reloaded with
`strict=True`), the simulator attributes listed in
`execution_loop._STEP_TRANSACTION_ATTRIBUTES` (time and step counters, interval
trackers, bootstrap marks, the adaptive proposal, provenance buffers, and the
forcing cursor), the provenance transaction, and all supported random-number
generators. Other Python attributes of a Functor and anything outside the
process (files, services, external deliveries already handed over) are not
rolled back. Rejected adaptive attempts and whole-step exceptions restore Python,
NumPy, Torch CPU, and Torch CUDA RNG state through the capture/restore helpers
owned by `relic\core\simulator\checkpoint.py`. Retrying therefore replays the
same substream. For Functors using `FunctorSubstreamMixin`, the existing
Brownian tree pins a rejected increment, partitions it for the shorter retry,
and retains the pending remainder. The marked-jump path retains rejected
arrivals and marks. Standalone Euler-Maruyama remains fixed-step; high-order
Levy-area schemes and adaptive subdivision without the path ledger remain `[N]`.

**dt Data Flow (`run()` chooses `dt_in`; `step_simulation()` executes it):**

```text
run():    dt_in = _adaptive_next_dt when adaptive, else dt,
          truncated to end_time - current_time
step(dt): dt_in = dt (self.time_step when dt is None); step_simulation()
          never uses _adaptive_next_dt to choose dt (it only writes the next
          proposal; the step transaction snapshots and restores it)
dt_in -> _execute_fast_with_adaptive_retry -> _execute_fast_group(dt)
      -> [splitting_mode=strang? palindromic split over the fast plan order:
          F1(dt/2) -> ... -> F(n-1)(dt/2) -> Fn(dt) -> F(n-1)(dt/2) -> ... -> F1(dt/2)]
      -> [error_norm > 1 and dt > dt_min? snapshot rollback, shorter retry]
      -> [error_norm > 1 and dt <= dt_min? accept with WARNING, no retry]
      -> [error_norm OK? grow _adaptive_next_dt for next step]
      -> future_time = current_time + dt  (dt returned: dt_in, or the retry)
      -> [should_execute_medium(future_time) OR event_trigger?]   (pure predicate)
         -> medium_group(effective_dt = future_time - last_medium_time)
         -> commit_rate_group_interval("medium", future_time)
      -> [should_execute_slow(future_time) OR event_trigger?]     (pure predicate)
         -> slow_group(effective_dt = future_time - last_slow_time)
         -> commit_rate_group_interval("slow", future_time)
```

Time argument seen by a Functor: every group call passes the interval it
accounts for, as `(time, dt) = (start_time, dt_group)`, and sub-step `k` (from
0) of that call receives `time = start_time + k * h_sub` with
`h_sub = dt_group / substep_count` (`execution_group._run_functor_substeps`).
The start is set by the caller:

- fast group in `lie` mode: `start_time = t_current`, `dt_group = h`;
- `strang` mode: the first half-steps and the middle full step start at
  `t_current`, the second half-steps start at `t_current + h/2`;
- medium and slow groups: `start_time = t_last` of that group,
  `dt_group = dt_eff = t_future - t_last`. Example: with `t_last = 0`,
  `t_current = 1`, and `t_future = 2`, the medium group is called with
  `time = 0` and `dt = 2`, the interval `[0, 2]` it accounts for.

The owner guarantees the interval. Whether a production process samples its
non-autonomous forcing at the right times inside that interval is still a
property of that process and must be checked per process.

**Palindromic Splitting (`splitting_mode="strang"`):**

For the fast plan order `F1, ..., Fn` (n >= 2) one step is the palindromic
composition

```text
Phi_h = Phi_1^{h/2} o ... o Phi_{n-1}^{h/2} o Phi_n^{h} o Phi_{n-1}^{h/2} o ... o Phi_1^{h/2}
```

with `2n - 1` sub-executions (Strang 1968; Hairer, Lubich and Wanner 2006,
Sect. II.5). When every `Phi_i` is the exact sub-flow, this palindromic
composition equals its own adjoint and is second order (local error
`O(h^3)`) for smooth sub-flows; for non-autonomous sub-flows this also needs
the stage times above (a second half-step started at `t` instead of `t + h/2`
leaves an `O(h^2)` local error). The palindromic call order does not by
itself make a composition of numerical sub-integrators symmetric. The order
claim is therefore bound to the sub-integrators: each Functor may declare
`FunctorMetadata.sub_flow_accuracy` (`SubFlowAccuracy.EXACT`, `ORDER_2`,
`ORDER_1`, or the default `UNDECLARED`), and the effective order of one step is
`q = min(s, min_i p_i)` with `s = 1` for `lie` and `s = 2` for `strang`
(`EXACT` counts as infinite order; with one fast Functor `q = p_1`; any
`UNDECLARED` Functor makes `q` "unverified"). `get_splitting_order_report()`
returns this result. Under `strang`, a view publication that changes the
effective order to a value below 2 or to "unverified" logs one warning; an
unchanged result is not logged again. `SubFlowAccuracy` is importable from
`relic.core` and `relic.core.functor`. Regularity and boundary treatment are
not checked. A
persistent regression test measures the order through `_execute_fast_group`
on a rotation problem with a time-dependent coefficient: error ratios on step
halving of about 4.00 for `strang` with exact sub-flows, about 1.99 for `lie`,
and about 2.01 for `strang` with explicit Euler sub-flows. The earlier form
split only the first Functor; for `n >= 3` that is not symmetric and is first
order in general. With `n = 1` the mode is the same as `lie`. The medium and
slow groups are not part of the composition; they run after it. Every
sub-execution goes through `_execute_group()` and is its own group commit, so
sub-stepping and StateManager governance are unchanged. Plain `lie` splitting
is first order under the same sub-flow conditions.

**Splitting admission at provider-view publication:**

`execution_loop.resolve_splitting_admissibility` checks the declared SCM
inputs/outputs and dependencies before registration, retirement, or restore
publishes a view. With at least two fast members under Strang, S1 refuses an
SCM flux path between stateful members, including paths through stateless
relays; shared-state coupling is not such a flux path. Unclassified endpoints
are flagged instead of proved safe. Without an SCM registry, two stateful
fast members cannot establish this admission. S3 with `require_declared`
refuses an undeclared member unless it is shown stateless; a declaration of
first order does not establish second order. S5 permits `reverse` only without
required or present optional same-step fast edges; lagged edges do not impose
that order. A refusal preserves the committed view through the existing
rollback owner. Lie admits and reports the flux-path flag.

`get_splitting_order_report()` also returns member classifications and their
basis, refused/flagged edges, exchange records, configured sub-step controls
(including unmatched names), effective fast sub-step seconds, and
`cross_group_composition=lie_fast_medium_slow`. Effective seconds refer to the
configured fast step, not an independently measured adaptive trajectory.

Dependency outputs (`subprocess_outputs`): the `2n - 1` sub-executions of one
fast-group composition share one dependency-output map
(`GroupComposition.dependency_outputs`). Each sub-execution
records its Functor's merged outputs, replacing that Functor's earlier entry,
and a consumer receives for each declared provider the outputs of that
provider's most recent earlier sub-execution (a required provider missing from
the map raises). Providers precede consumers in the fast order, so every
`strang` sub-execution of a consumer, including its second half-step over
`[t + h/2, t + h]`, sees each provider's first half-step outputs over
`[t, t + h/2]`; in `lie` mode it sees the provider's outputs over `[t, t + h]`.
An entry is accumulated over its own sub-execution interval, not over the
consumer's. The composition (`GroupComposition`) is created empty for each
fast-group attempt (a rejected adaptive attempt's composition is discarded
with the attempt) and dropped when it returns. Medium and slow group calls are compositions of their
own; a provider in another execution group is never in the map, and its values
are read from the committed state with the declared cross-group lag.

Merge rule for successive outputs: the outputs of successive executions -- the
substeps of one Functor call, the Functors of one group call (their
diagnostics), and the sub-executions of one `strang` composition -- merge by
one rule, `execution_group.merge_sequential_updates`, and `step()` folds the
group results into its returned mapping with the same decision
(`replaces_earlier_output`). The channels come from `partition_outputs`. A
plain diagnostic (a `diagnostics.*` or `__*` name returned as a plain value) is
an observation: the last value is returned. An `UpdateItem` follows its mode:
`REPLACE` keeps the last value; an `ACCUMULATE` amount (`per_timestep=True` or
a per-step basis; `FLUX` arrives lowered to such an amount) is summed; an
`ACCUMULATE` per-second rate (`per_timestep=False` with basis `per_second`, or
`instantaneous` for a boundary-rate aggregate) adds with the weight `h / H`
(`coverage`), the length of the interval its execution accounts for over the
length of the merged interval, so that the merged rate times `H` is the
integrated amount: the time-weighted mean over the substeps of one call
(`h / H = 1 / substep_count`), the sum over the Functors of one group call
(`1` each), and the rate over a `strang` step (half-steps `1/2`, the middle
`1`). Items of one name must agree on mode, write mode, unit, basis,
`per_timestep`, and element. A plain state payload is an increment
and is summed. A diagnostic whose whole-step total is wanted must be returned
as an `ACCUMULATE` `UpdateItem` (for example with `write_mode=DERIVED`).
Stored diagnostics follow the same rule: the composition keeps each
Functor's diagnostics merged over its sub-executions
(`GroupComposition.diagnostics`, UpdateItem modes kept), and after each
sub-execution StateManager stores the merged payload for that Functor
(`ingest_module_diagnostics` replaces its entries; it stores real scalars
only, a real number or a real tensor as its mean, and any other value such
as a string, `None`, a sequence, a mapping, or a complex tensor raises
`TypeError` naming the key, with none of that call's diagnostics stored).
After the step the stored
value equals the returned one: an `ACCUMULATE` amount diagnostic of a Functor
that runs twice in a `strang` composition is stored as the sum, a per-second
rate diagnostic as the mean of its two half-step rates. A rejected adaptive
attempt's stored diagnostics are restored with the attempt's StateManager
checkpoint, and its composition is discarded. The visible boundary rate during a composition is the sum of the latest
emitted contribution of each writer in that composition. A new emission
replaces only that writer's contribution; no previous-step mean is added.
Within a provider's substeps its latest reduced rate is visible; after its
call, its interval mean is visible and retained in `subprocess_outputs`.
An absent emission holds the latest stage observation, whereas the private
whole-interval accumulator counts only emitted contributions, weighted by
`h / H`. The complete mean is published through StateManager only after the
composition finishes, before the adaptive metric. An `ACCUMULATE` boundary contribution is reduced at its writer substep;
a physical `REPLACE` call mean uses the existing group-commit reducer once. Physical boundary
amounts are checked once over their own intervals; stage and final-rate
restatements add no second transfer or boundary-budget entry.

The 2026-10-20 P-N evidence revised the step-5 last-child decision for an
actual physical boundary port. In `_run_functor_substeps`, a substepped
`REPLACE` rate is selected only through the existing SCM writer
`allowed_in`/`allowed_out` route and StateManager boundary validator
(rate units, direction and actual interval). Its call value is
`sum_k q_k dt_k / dt_call`; every actual child must emit it. Missing
coverage or an explicit `instantaneous`/`interval_integral` temporal
support conflicting with this rate integral refuses and rolls back the
step. Absent temporal-support metadata is not filled with a default: the
existing physical port and commit-interval contract govern the amount.
Ordinary state or observation `REPLACE`, `DERIVED` outputs and the
existing `ACCUMULATE` rate rule keep their semantics. In the constructed
P-N pair this selects `n2_production`; the transformation observations
`nitrification_flux` and `denitrification_flux` have no physical route
and keep the last child value. The existing wider S2 publication selector
still acts on committed SCM flux call values; it does not certify all SCM
fluxes as having the same temporal support. This correction changes the
reported physical rate, not the reaction law or the number of state
advances.

An SCM flux `REPLACE` item with `per_timestep=False` has an additional
composition publication rule (S2): `GroupComposition` accumulates each
writer's committed call rates with coverage `h_k / H` and publishes the
complete interval mean after the composition. During calls, readers retain
the latest stage rate. A missing emission from one of that writer's calls,
incomplete interval coverage, or a second writer is refused. The ordinary
`REPLACE` merge rule is unchanged; the final call returns the completed mean,
so the step result and published state agree. This restatement is not a
second physical transfer and does not change the process law.

**Adaptive Update-Magnitude Metric (Implementation Contract):**

```
tol_k = atol + rtol * |x_k|                  x_k: state before the fast group
u_k = x'_k - x_k                             x'_k: value StateManager committed
e_k = mean_elements((u_k / tol_k)^2)
Err = sqrt((1/m) * sum_k e_k)                m monitored written state
                                             variables, equal weight
Err > 1 and dt > dt_min:
          reject, roll back, dt_retry = max(dt * max(0.1, 0.9 / sqrt(Err)), dt_min)
          (< dt, because the factor is at most 0.9 and dt_min < dt), run the
          fast group once more over [t, t + dt_retry] and accept that result;
          _adaptive_next_dt = dt_retry
Err > 1 and dt <= dt_min:
          no shorter retry exists: accept the attempt without rollback, log a
          WARNING naming Err and dt_min, and propose dt_next as for an
          accepted step
Err <= 1: accept, dt_next = dt * min(5.0, max(0.1, 0.9 / sqrt(Err)))
Err == 0: accept, dt_next = 2 * dt (the formula is undefined at zero; Err is
          also 0 when no written, monitored state variable has a numeric
          pre-step value, so an unmeasured step doubles the proposal)
dt_next is then clipped to [dt_min, dt_max]
```

`u_k = x'_k - x_k` is the change StateManager committed to variable `k` over
the fast group (`x_k` the pre-step snapshot, `x'_k` the committed value), so
every update form is measured by what was committed: a `REPLACE` value, an
integrated amount, a per-second rate or `FLUX` integrated once over the real
interval (substeps, `strang` half-steps), a declared boundary-rate aggregate
stored as a rate, and clipping or distributed reduction at commit. Only
actual written state targets with a numeric pre-step value enter (diagnostics
and `DERIVED` outputs are excluded). The existing StateManager target rule
maps `delta_x` to `x`; targets are deduplicated before monitor filtering and
pre/post-state lookup. This does not authorize SCM read aliases as write
names. A missing committed value or a shape change raises. Until 2026-09-23 the metric used the returned `REPLACE` value itself,
which gave `Err` close to `1/rtol` whatever the step length. Until 2026-09-26
a per-second rate entered as the rate itself (`x = 300 kg m-2`,
`q = 0.006 kg m-2 s-1`, `dt = 60 s` gave `Err = 0.02`, accepted with a 300 s
proposal, instead of `Err = 1.2`, rejected once, retry 49.3 s), and a steady
boundary rate `q0 = 0.5` gave `Err = 998` instead of 0. Only fast-group updates enter
the metric. Each monitored variable is first averaged over its tensor elements and
then weighted equally with the other variables.

This metric is a control heuristic over committed state changes. It is not a
Richardson estimate, an embedded-pair estimate, a local-truncation-error
estimate, or a CFL guarantee. A rejected first attempt is rolled back and
retried at most once, and only when it is longer than `adaptive_dt_min`; the
retry interval lies inside the rejected one (so a Brownian-tree Functor
refines its pinned increment), and the retry result is accepted without a
second error check, so an accepted step can have `Err > 1`. An attempt that
is not longer than `adaptive_dt_min` is accepted with a warning. Until
2026-09-23 the retry was `max(..., dt_min)` even for such an attempt, so it
could be longer than the attempt and `run(...)` could pass its end time
(example: `dt_min = 300 s`, end time 500 s, run ended at 600 s; now 500 s). `run(...)` uses
`_adaptive_next_dt` as the next step size; a direct `step(dt)` call uses the
`dt` it is given, and the proposal is used only when `run(...)` takes the
next step.

This heuristic is the only step-size controller used by the Simulator. The
solver-library `AdaptiveTimeStepper` and the coupling-layer
`AdaptiveTimestepController` and `CFLChecker` were Gustafsson-style PI
controllers with no runtime callers. They were removed on 2026-09-21 together
with the `coupling.scales.temporal.adaptive.*` parameters. Do not add a second
controller. The solver library has no step-size control: on 2026-09-23 the
unused `forward(..., adaptive=True)` option of the solvers sharing
`_ODEForwardMixin` (`BackwardEulerSolver`, `ImplicitMidpointSolver`,
`CrankNicolsonSolver`, `RK4Solver`) and its `_adaptive_step` loop were removed
together with the `computation.solvers.rk4.adaptive` key, because the option
advanced time by the proposed next step instead of the step taken and had no
runtime caller; passing `adaptive` now raises `TypeError`. `forward` integrates
on a fixed grid whose last point equals `t_end` exactly (no residual step
shorter than a roundoff-scaled tolerance), returns float64 time points, and
stops at the first failing step. `step` raises `SolverStepError` (a
`RuntimeError`) when the step status is not `SUCCESS`: `MAX_ITERATIONS` when
the fixed-point iteration of `BackwardEulerSolver`, `ImplicitMidpointSolver`,
or `CrankNicolsonSolver` stays finite but does not converge within
`max_iterations`, and `DIVERGED` when an `RK4Solver` solution has a
non-finite element. In the three implicit solvers a non-finite iterate never
becomes a status: `SolverBase.check_convergence` raises
`ValueError("residual must contain only finite values")` (or the matching
`reference` message), and `forward` propagates that `ValueError`.

**Event Trigger Mechanism:**

- Registered via `simulator.register_event_trigger(group, fn)`.
- `fn(state: Dict, current_time: float) -> bool`, forces execution of the group when returning `True`. `state` is the current state after the groups that already ran in this step (the fast group, and the medium group before the slow check); `current_time` is the step start `t_current`, not `t_future`. Triggers are consulted only when neither the interval predicate nor a bootstrap mark already started the group.
- Supports ecological trigger conditions such as GDD thresholds, sudden temperature drops, or soil moisture extremes.
- A trigger that raises, or returns a value that is neither `bool` nor `int`,
  aborts the step with `RuntimeError` naming the trigger and the group
  (`check_event_triggers`). The failure is not turned into "not triggered".
  The step transaction rolls back as for any other step error.

**Rate-Group Predicates and Interval Commit:**

- `should_execute_medium(simulator, future_time)` and
  `should_execute_slow(simulator, future_time)` compare interval indices and
  change nothing. Calling them repeatedly has no effect.
- The trackers (`_previous_medium_index` / `_previous_day_index`,
  `_previous_slow_index` / `_previous_month_index`) advance only in
  `commit_rate_group_interval(simulator, group, future_time)`, which
  `_execute_rate_group_if_needed()` calls after the group has run. They are
  step-transaction attributes, so a rejected or failed step restores them.
- `effective_dt = future_time - last_execution_time` must be positive. A
  non-positive value means the last-execution bookkeeping is inconsistent,
  and the step raises `RuntimeError`. The raw `dt` is not substituted.
- Trigger conditions: `medium` fires when
  `floor(t_future / medium_interval_seconds)` exceeds the committed medium
  index. `slow` uses `floor(t_future / slow_interval_seconds)` when that key
  is set, and otherwise the month index of the configured calendar
  (`_compute_month_index` through `PeriodIndexer`, so month lengths follow the
  calendar, including `360_day`; no fixed 30-day month is assumed). A Functor
  with `force_initial_execution=True` marks its group for bootstrap. When the
  interval predicate is false, the mark starts the group. The mark is consumed
  whenever the group executes, whatever started it (boundary, mark, or
  trigger), so a group that ran for its first boundary does not run once more
  at the next step (example: `step(100)` then `step(50)` runs the medium group
  once). The mark is a transaction attribute and is restored on rejection. A
  registered event trigger can also start the group; triggers are checked
  after the interval predicate and the bootstrap mark.
- A group runs at most once per step. When one step crosses several
  boundaries, the group runs once with `effective_dt` covering the whole
  elapsed interval; the skipped boundaries are not replayed one by one.

**Sub-stepping (metadata or per-run override):**

- Without an override, `FunctorMetadata.substep_count` supplies the count
  (default 1). `substep_overrides[name]` instead supplies a positive integer
  `count` or a finite positive `max_seconds=s`, giving
  `n=ceil(dt_call/s)` for each group call, including each Strang half-call.
- `registration.py` includes the override in the provider binding digest;
  `execution_group.resolve_call_substep_count` is the single count owner.
  The group executor divides the actual call interval into these sub-steps.
- Sub-step `k` (from 0) receives `time = start_time + k * h_sub`, where
  `start_time` is the start of the group call (see the time argument above);
  the next sub-step reads a local provisional state, not a global write, in
  which an amount is added and an ordinary per-second ACCUMULATE rate `q`
  advances its target by `q * h_sub`. A declared boundary-rate aggregate
  instead uses the composition's stage view: the latest emitted reduced
  contribution of each writer is summed, without adding the old state rate.
  After a provider call, its own interval mean replaces its contribution in
  that view. Later Functors read this stage view; ordinary per-second state
  tendencies still advance their local targets once over `dt_group`.
- Sub-step results are merged by the single merge rule
  `merge_sequential_updates` (`execution_group.py`): a plain diagnostic keeps
  the last observation, a REPLACE item the last value, an ACCUMULATE amount or
  a plain state increment is summed, and a per-second ACCUMULATE rate forms the
  time-weighted mean `sum_k q_k h_sub / dt_group`. A substepped physical
  REPLACE boundary port also publishes its actual child-interval mean
  under the qualified port rule above, before S2 averages group calls.
  StateManager integrates that
  rate once over `dt_group` (or keeps it as the boundary-rate aggregate), so a
  constant rate `q` commits `q * dt_group` for any `substep_count`.

**Cross-Group Sampling Lag and Its Mitigation:**

- Multi-rate execution has sampling lag in both directions. A `fast`
  consumer of a `medium` or `slow` output reads the value committed when that
  producer last ran. A `medium` or `slow` consumer of a faster producer sees
  the producer's state only when its own group runs.
- `_warn_cross_group_dependencies()` runs when a provider view is published,
  so a move is part of the published plan and not a per-step mutation. It
  compares declared inputs and outputs across groups with the ranks
  `fast = 0`, `medium = 1`, `slow = 2`. A read by a slower consumer from a
  faster producer is a sampling lag (warning text "Cross-group sampling
  lag"); a read by a faster consumer from a slower producer is a hold lag
  ("Cross-group hold lag": the consumer holds the value committed when the
  producer last ran). Both directions are warned at every publication.
- A read is acknowledged when the consumer lists the producer in the `lagged`
  part of its dependency declaration; an acknowledged read is logged at DEBUG
  only. With `strict_cross_group_dependencies=True` every unacknowledged read
  raises `RuntimeError`, and the registration, retirement, or restore that
  would publish the plan is rolled back.
- `resolve_cross_group_deps=True` moves every `slow` consumer with an
  unacknowledged sampling lag into the `medium` group (iterated to a fixed
  point, logged at INFO, recorded as `promoted_to_medium` in the publication
  history). A moved Functor is a member of one group only, so it is
  integrated once per medium interval; a pending slow bootstrap mark of a
  moved Functor moves with it. A Functor that was already a slow member of
  the previously committed view moves only when `t_last_slow ==
  t_last_medium`; otherwise the publication raises `RuntimeError` naming the
  interval that would be left uncovered (`t_last_slow < t_last_medium`) or
  integrated twice (`t_last_slow > t_last_medium`), because one group commit
  integrates with one interval and no per-Functor catch-up exists. A provider
  first published in this view has no accounting history and is placed in
  the medium group directly. The check uses step-transaction attributes that
  are saved in checkpoints, so it gives the same result after a restore. A
  moved Functor stays in the medium group; retiring the producer later does
  not move it back. Nothing moves into `fast`, and fast-group reads do not
  change.
- A `lagged` declaration removes the edge from the current step's dependency
  graph (`DependencyGraphResolver` keeps it out of the adjacency). It does
  not by itself make the consumer read the previous accepted step: the value
  read is whatever the state holds when the consumer runs, which follows from
  the execution order and the group commits. None of these mechanisms
  computes or enforces a staleness bound.

**Declared interval-mean exchange:**

Without an exchange declaration, cross-group reads retain the hold semantics
above. An admitted `interval_mean` record makes the existing execution loop
commit `exchange_mean__<id>` through StateManager under writer `exchange:<id>`
after adaptive acceptance and before medium/slow calls. It updates
`m <- m + (vbar-m) h/(tau+h)` for `tau>0` and sets `m=vbar` for `tau=0`,
where `vbar` is the committed fast-step value, `h` the accepted fast interval,
and `tau=t_step_start-t_last_consumer`, all times in seconds. This is the
exact mean of the declared piecewise-constant reconstruction, not a
continuous-time accuracy claim or a physical budget term. The consumer call
receives a temporary input overlay of its original variable with this mean.
A consumer group writing that variable, an undeclared unit, a Functor writing
the reserved mean, or an unsupported interpolation mode is refused. Means
and interval trackers roll back and restore through the existing state and
checkpoint owners; cross-group composition remains sequential.

**Prohibitions:**

- Writing custom scheduling logic outside of `time_ops.py` / `execution_loop.py`.
- Hardcoding step intervals without using `_get_setting()` configuration keys.
- Bypassing the `_execute_group()` sub-stepping mechanism to write manual loops.
- Passing raw `dt` instead of `effective_dt` to medium/slow groups.
- Mutating interval trackers outside `commit_rate_group_interval()`, masking a
  non-positive `effective_dt`, or swallowing event-trigger exceptions.
- Adding a second step-size controller or literal execution-group thresholds.

Extension Rule:

- Always extend preferentially within the corresponding submodule. Never regress to a "monolithic single file" structure.

**Bounded writer provenance (CR-A-01):**

`execution_loop.step_simulation` owns retention of the existing
`_step_provenance_detail` audit lists. The runtime configuration key
`max_provenance_records_per_variable` defaults to 256 and accepts only a
positive integer (validated before a step); the ordinary `_get_setting`
root/section lookup applies. After all groups and finalization succeed, and
before automatic or explicit external-intent checkpoint preparation, each
variable keeps its most recent M writer records. A trim replaces the list;
it never truncates the earlier list held by the append-only savepoint. A
pre-acceptance failure therefore restores the previous window and its
cumulative eviction count exactly. The last writer is always retained.

`get_variable_provenance(var_name=None, *, include_retention=False,
include_commit_context=False)` queries this
rolling window, not a complete execution history. Its default record shapes
are unchanged. `include_retention=True` returns `records` plus
`evicted_records` (an integer for one variable, or a per-variable mapping for
all variables); zero means no records have been evicted from that variable's
retained audit history. Eviction counts are step-transaction runtime fields,
so the existing checkpoint serializer saves and restores them and the existing
graph migration maps their variable names. The configuration remains part of
the exact restart identity. Stable-boundary manual checkpoints serialize the
same already-trimmed state; no second retention or persistence owner exists.

`include_commit_context=True` also returns each writer record's
`commit_context` and StateManager `commit_index`, separating the two Strang
half-calls and successive steps. Context contains step, composition start and
interval, stage, role, interval start and `dt`; default record shapes stay
unchanged. This retained window is not a full-trajectory ledger once records
have been evicted.

This bounds writer-record storage by O(V M), for a fixed set of V tracked
variables, plus the current step's writes. Scientific state, numerical memory,
forcing cursor, RNG, and per-step increment attribution are not discarded.
It does not bound caller-owned forcing arrays, requested time-series outputs,
or growth in the number of distinct variables introduced by topology changes.
StateManager's separate history and conservation ledgers use its existing
`max_history_length` option; full-run memory claims require measurements of
all these owners and the input/output workflow. Internal direct group calls
remain scoped group APIs, not accepted whole steps.

#### 4.4.2 Device Management Policy

RELIC uses a **3-tier configurable device assignment** system. Hardcoding device selections within scientific modules is strictly prohibited.

**Resolution Priority (Highest to Lowest):**

| Tier | Source                       | Location                                                   |
| ---- | ---------------------------- | ---------------------------------------------------------- |
| 1    | Per-Functor Override         | `SimulatorConfig["device_overrides"]["functor_name"]`      |
| 2    | Simulator Global Default     | `SimulatorConfig["device"]` (`auto`/`cpu`/`cuda`/`cuda:N`) |
| 3    | Pre-Registration Placeholder | `torch.device("cpu")` inside `FunctorBase.__init__`        |

**Lifecycle:** `FunctorBase.__init__` sets `self.device = cpu` (placeholder).
During registration, `_maybe_migrate_functor_device()` attempts to migrate the
Functor to its effective device (override > global). Query the resolved policy
via `simulator.get_effective_device("functor_name")`. Resolution failures abort
registration. Migration failures or cancellation trigger device compensation
and registration rollback; incomplete compensation reports both causes.
Use `require_gpu` when CPU fallback during policy resolution is not acceptable.

Held ParameterManager bindings (Section 4.2) follow the Functor: `_plan_binding_relocations` in `registration.py` plans, per key, the relocation of every held binding that is not on the effective device, and `relocate_binding(manager, key, device)` (`relic\parameters\ops\nn_params.py`) moves it in place (Parameter object identity and accumulated `.grad` values are retained, so every holder reads the moved tensor). A binding already held by another registered Functor on a different device raises `ValueError` naming the parameter, both Functors, and their devices (assign them one device through `device` / `device_overrides`); a failed registration moves relocated bindings back.

A device relocation can recreate the binding's `AccumulateGrad` association. An existing forward graph may retain its original gradient edge while a later lookup through the moved Parameter resolves a new edge. Compile and preflight use registration with device preparation deferred, preserving the borrowed binding and its live graph association. Actual execution registration performs the existing device resolution, shared-binding conflict checks, completion fences and relocation.

The table above describes the Functor registration lifecycle. The general
`relic\core\device_policy.py::resolve_device(...)` entry point also considers
explicit arguments, module/runtime/environment overrides, process defaults, and
distributed defaults. Scientific modules must not reproduce that precedence.

**Device completion fences (E01):** a Python return from a kernel launch is
not completion of the device work. `relic\core\device_policy.py` owns
`DeviceFence` / `record_device_fence(...)`: CUDA events recorded on the
current stream of every CUDA device named (an index-less `cuda` resolves to
the current device; a CPU-only fence is vacuously complete). The provider
lifecycle binds fences to its reader tokens and retirements: releasing a
`ProviderViewToken` records a fence over the view's provider devices, and
retiring a view records one too; a retired view whose readers are gone but
whose fences are pending has status `pending_device_fence` and stays in
`retired_views` until `wait_quiescent`/`retry_dispose`/a later release finds
the fences complete (`device_fence_status(epoch)` exposes the pending
devices). `wait_quiescent` awaits device completion outside the lifecycle
lock. Device migration at registration awaits a fence on the source device
before copying tensors (`managed_delta["device"]["source_fence"]` in the
receipt) and `to_device` re-creates the substream generator on the target;
migration across device types after draws stays refused. Registration and
retirement receipts carry `provider_view.retired_view` with the fence status
observed at publication. Real CUDA evidence is
`test\core\device_fence_lifecycle.test.py`; its GPU cases are skipped, never
counted as passed, without a device.

**Key Files:**

- `relic\core\simulator\main.py`: `_resolve_runtime_device`, `_resolve_single_device`, `get_effective_device`
- `relic\core\simulator\registration.py`: `_maybe_migrate_functor_device`, `ProviderViewToken`, `_ProviderLifecycleState.device_fence_status`
- `relic\core\device_policy.py`: Unified entry point for policy resolution, module-level overrides, and `DeviceFence`.
  Override maps (config `device_overrides` / `module_device_overrides` at root,
  `simulator` or `runtime`, `set_device_overrides(...)`,
  `RELIC_DEVICE_OVERRIDES_JSON`) are parsed strictly: an unparseable device or a
  CUDA device under `require_gpu=True` without CUDA raises `RuntimeError` naming
  source and key, a non-mapping section raises `TypeError`, environment text that
  is not a JSON object raises `ValueError`; no override is dropped with a warning.
  Override maps keep the requested device (syntax-checked, not resolved); a
  request resolves only the selected override, with its own `require_gpu` and
  CPU fallback, so an unavailable override of another module does not fail it.
  The environment parse is cached per raw text. `get_device_overrides()`
  returns the runtime overrides resolved under the default policy;
  `set_device_overrides(..., require_gpu=True)` also checks availability before
  storing.
  `is_cuda_available()` propagates a torch probe exception instead of reporting
  no CUDA.

**Configuration Example:**

```yaml
simulator:
  device: "auto"
  require_gpu: false
  device_overrides:
    farquhar_photosynthesis: "cuda"
    management_coordinator: "cpu"
  module_device_overrides:
    relic.interfaces.api.routes.compute: "cpu"
```

**Execution Reminders (Must Comply):**

- Default strategy for distributed computing: prioritize GPU; fallback automatically to CPU if CUDA is unavailable. An explicit `device: cpu` is honored on every rank (the GPU-first rule is the default for `auto`, not an override), so the ranks of one process group share one device identity.
- All modules must be configurable independently for devices, preferentially using `module_device_overrides` (module-level) and `device_overrides` (Functor-level).
- `torch.cuda.is_available()` is permitted _only_ within `relic\core\device_policy.py`; scattered calls elsewhere are strictly forbidden.
- Self-check command: `rg "torch.cuda.is_available\(" relic` (expected to only hit `relic\core\device_policy.py`).
- Standalone Functor instances (unregistered in Simulator) use the CPU placeholder by default; explicitly pass `device` or enforce `to_device("cuda")` if a GPU is required.

**Prohibitions (Inside Scientific Modules):**

| Forbidden Pattern                                   | Correct Alternative                                     |
| --------------------------------------------------- | ------------------------------------------------------- |
| Calling `torch.cuda.is_available()` in `__init__`   | Let the Simulator assign the device during registration |
| `device or torch.device("cuda")`                    | `device if device is not None else torch.device("cpu")` |
| Hardcoding `self.device = torch.device("cuda")`     | Accept a `device` argument, default to `None` or `cpu`  |
| Calling `_resolve_runtime_device()` within Functors | _Only_ `RELICSimulator` calls this function             |

#### 4.4.3 Durable Checkpoint and Restart Contract

The sole durable checkpoint owner is
`relic\core\simulator\checkpoint.py`. `RELICSimulator` and
`GraphExecutor` expose public delegation methods; they do not define another
serializer, format, checkpoint registry, or persistence lifecycle.

**Explicit and automatic creation:**

- `RELICSimulator.create_checkpoint(checkpoint_path)` requires an explicit
  file path. The public create path refuses to overwrite an existing target,
  creates missing parent directories, writes a uniquely named sibling
  temporary file, and atomically commits it with `os.replace(...)`.
- Automatic checkpointing is opt-in. `enable_checkpointing` defaults to
  `False`; setting it to `True` requires an explicit
  `checkpoint_directory` and a positive `checkpoint_frequency`.
- Automatic and explicit checkpoint candidates are fully captured before the
  step's acceptance boundary, without file I/O. Preparation projects the
  accepted provenance flags, original auto-report setting, and temporary
  source-map cleanup only into the independent snapshot. The ordinary capture
  API still requires a stable boundary. File publication follows acceptance
  of the scientific state, simulator time and step counters, and outputs.
  The automatic name is
  `checkpoint_step_<12-digit-step>.pt` inside the current `run_id` subdirectory
  of the configured checkpoint directory. The simulator owns `run_id`; reset
  starts a new run and retains previously accepted pending deliveries.
  There is no implicit checkpoint destination.
- Simulator identity changes and StateManager attachment share one binding
  path through `StateManager.set_provenance_recorder(..., run_id=...)`.
  The recorder's environment snapshot lists every installed distribution; one
  without Name or Version metadata raises. `git_status.json` records
  `available: false` only without a repository root or a git executable on
  PATH; any other git execution failure propagates.
  Native Recorder notifications freeze their run and destination when staged;
  retries retain those values even after a later reset or observer reconfiguration.
  Restore applies the saved identity, including rollback on a failed restore.
  Run identity and StateManager replacement are prohibited during an active step.
  GraphExecutor confirms the previous simulator's accepted deliveries before
  resetting its StateManager or replacing the simulator. Attachment, delivery,
  and reset errors propagate; a failed delivery retains the previous run.
- Diagnostic file export also runs after accepted science and provenance
  commit. In-memory time-series output remains inside the step savepoint.
  A post-commit file error must not roll back accepted scientific state;
  delivery receipts and retry behavior are separate from state acceptance.
- The existing execution loop combines requests by normalized resource
  identity, including Windows case aliases. Conflicting operations or options,
  an existing target for a new checkpoint, and invalid diagnostic epochs are
  rejected during preparation. The `topology` operation names one declared
  transition (payload
  `transition` equal to the resource); an undeclared transition is rejected
  during preparation, and an accepted request is committed by the
  registration owner when the queue drains, with a receipt carrying the
  transition name, the epochs, and the lifecycle receipt id. A complete prepared batch enters the simulator's
  pending queue immediately after acceptance, before cleanup or file delivery.
  An interrupted retry uses the original frozen payload and the existing I/O
  owner's receipt; it does not recapture newer live state.
- StateManager's closed provenance-transaction flag identifies acceptance.
  Failure or cancellation before that boundary restores the step savepoint,
  including RNG and previously absent temporary runtime fields. Cancellation
  after that boundary retains the accepted state and prepared delivery batch.
  Native provenance notifications retain their original run, destination, and
  values in durable checkpoints. Restore reconciles accepted notifications,
  reuses or reconstructs the native Recorder without delivery during restore,
  and confirms them through the existing retry entry point. Process-local
  transaction capture and restore require confirmed notifications and file
  deliveries; the durable
  owner's internal rollback image preserves pending work without acknowledging it.
  In-memory acceptance alone is not durable: recovery covers only the obligations
  present in a successfully persisted checkpoint's frozen delivery closure.
- Checkpoint file errors propagate. There is no public checkpoint-placeholder
  cleanup API; checkpoint correctness must not depend on scanning or deleting
  unrelated files.

**Strict file and restart boundary:**

- The active format identifier is `relic.simulator.checkpoint`, with exact integer `format_version=3` validation (`CHECKPOINT_FORMAT_VERSION`); only version 3 is written. Each Functor entry is `{state_dict, parameter_bindings}`: `state_dict` is the module state, and `parameter_bindings` holds the live value of every ParameterManager binding the Functor (or a nested Functor) holds by reference, captured and written back through the owning manager (`capture_binding_values` / `restore_binding_values` in `relic\parameters\ops\nn_params.py`), which keeps each binding's object, gradient flag, and device; a saved value must match the live binding's shape and dtype, and the saved and live key sets must be equal. The in-memory runtime transaction uses the same entry. A version-2 file (a held binding stored as a Functor `state_dict` entry) is read through `_upgrade_version_2_payload`: each such entry of the target Functor moves to `parameter_bindings`, an entry that is neither module state nor a held binding raises `ValueError`, the upgraded payload is resealed and validated, and the restore receipt records `format_upgrade` (`from_format_version`, `to_format_version`, `moved_bindings`). Restore uses
  `torch.load(..., map_location="cpu", weights_only=True)` and provides no
  unsafe or legacy deserialization fallback.
- The same envelope stores external work in `delivery_state`: ordered pending
  progress, receipts, job definitions, and flat checkpoint-node references.
  Each frozen root carries its own pending obligations. Selecting a root takes
  its reachable definitions without recapturing state or embedding nested
  delivery tables. Symbolic references may cycle; serialized values do not.
  Diagnostic snapshots use the existing DataWriter preparation and checkpoint
  value grammar. A delivery still publishes through its original I/O owner.
- The payload binds simulator runtime state, a `StateManager` snapshot,
  registered Functor state, topology, scientific contracts, the resolved
  device/distributed environment, RNG state, and a SHA-256 integrity record.
  Topology validation includes Functor names/types, execution order and
  groups, dependency nodes/edges, simulator and StateManager contracts,
  Functor configuration/metadata/parameter bindings, and accessed parameter
  specifications.
- `FunctorBase.get_kernel_identity()` is the sole owner of the current
  source-traceable Python kernel identity. Checkpoint scientific contracts and
  disk/DIRTY cache keys consume the same payload. It binds the actual class
  source, bound Python function, semantic code contents, Python cache tag,
  defaults, and supported immutable closure cells. A changed kernel or missing
  identity is rejected by the existing topology check before restore mutation.
  Paths, line numbers, process addresses, and `repr` fallbacks are not identity
  fields; same-class permanent code-hash caching is not used. The
  registration owner consumes the same payload into the provider
  implementation digest of the committed view; checkpoint topology binds the
  provider identity digests per Functor next to the kernel identity.
- This is a bounded Python-kernel contract, not complete dependency, external
  helper, native ABI, compiled-backend, or cross-environment equivalence.
  Untraceable source, opaque callables, foreign receivers, and mutable closure
  or default values are not certified. The ordinary uncached call path does
  not acquire an unrelated identity-certification requirement.
- Functor persistence uses the standard `torch.nn.Module.state_dict()` /
  `load_state_dict(..., strict=True)` contract. A Functor with additional
  replay-critical mutable state must implement PyTorch
  `get_extra_state()` / `set_extra_state()` using the checkpoint-safe value
  grammar; it must not create a parallel persistence channel.
  A ParameterManager binding held by reference is not module state; its value travels in `parameter_bindings`.
- `FunctorBase` does not own `save_checkpoint(...)` or
  `load_checkpoint(...)`. Durable restart is exposed only by the simulator
  checkpoint owner; Functors expose state dictionaries and optional extra
  state.
- Python, Torch CPU, the legacy global `numpy.random` state (optional), and
  CUDA RNG states are restored symmetrically. Independent generators (for
  example `numpy.random.Generator` objects or library-private streams) are
  covered only when their owning Functor exposes them through
  `get_extra_state()`, as `StochasticSubstream` does. A CUDA-resolved runtime requires CUDA RNG state; a CPU-only
  runtime rejects unexpected CUDA RNG state, and CUDA device-count mismatch is
  fail-closed.
- The current format is one file for one process. Creation and restore both
  fail closed when an active distributed runtime reports `world_size > 1`;
  coordinated multi-rank checkpointing is not implied by the presence of
  distributed runtime support nor by the coordinated registry protocol
  below (a coordinated checkpoint format is a separate, unimplemented step).
- Restore validates the complete checkpoint before mutation. Its existing
  runtime transaction backup and registration metadata snapshot compensate
  `StateManager`, Functor state, runtime fields, RNG, accepted pending work,
  and provider metadata when application or publication fails before the
  committed view/epoch changes. The lifecycle lock covers that commit boundary;
  there is no second scientific-state backup. After publication, retirement
  failure or receipt I/O cannot rewind the reader-visible world. Those errors
  remain observable with the committed epoch. Local evaluation transaction
  restore does not publish a new epoch. Incomplete compensation reports both
  the original error and compensation errors, never a false rollback success.
- The simulator owns one forcing cursor (`bind_forcing_cursor`,
  `get_forcing_cursor`, `clear_forcing_cursor`): the canonical SHA-256 of an
  externally supplied forcing sequence, its length, and the number of
  snapshots consumed by accepted steps. It advances only inside an accepted
  step (`_advance_forcing_cursor` in the step finalization), rolls back with
  the step transaction, refuses a step past the sequence end before any
  mutation, and is a validated `runtime` field of the checkpoint. Binding a
  different sequence while one is partially consumed is refused. An external
  adapter's read at the simulation time is independent of this cursor: it is a
  function of t (`time_origin + time` in `calendar_type`, see the coupling
  section), so a restored run reads the records of its own time, and no adapter
  read position is checkpointed. `ExternalAdapter` serializes only
  its source/model binding as standard extra state, so a checkpoint cannot be
  restored into an adapter bound to another source or model. Connection
  handles and cached datasets are resources re-opened by their owner.
- Compatible restart is a separate claim from exact restart and is owned by
  the same serializer. `restore_checkpoint(path,
  compatibility_certificate=RestartCompatibilityCertificate(...))` accepts a
  checkpoint whose topology differs from the live simulator only where the
  certificate says so: the certificate must name the real saved and live
  topology digests (mint them with `compute_restart_topology_digests`),
  lists the scientific-contract leaf paths that may differ, bounds numeric
  parameter drift with a relative tolerance (non-numeric values need their
  exact path), and maps renamed state variables whose units, shape, and
  dtype are unchanged. Derived provider `binding_digest`/`contract_digest`
  differences are accepted only together with a certified underlying
  difference of the same Functor. Functor names and types, execution order,
  groups, dependency nodes and edges, runtime environment, kernel identity,
  implementation digests, StateManager type, and accessed parameter
  specifications are never certifiable; hidden Functor state and RNG state
  are always restored exactly. A certificate whose digests do not match, or
  any field beyond the declared contract, is rejected before mutation. The
  restore receipt reports `restart_mode` and the accepted differences.
  `GraphExecutor.execute_from_checkpoint(..., compatibility_certificate=...)`
  passes the certificate through. Cross-graph migration is not implied.
- Cross-graph migration replay is the third restart claim, again owned by
  the same serializer: `restore_checkpoint(path,
  migration_certificate=GraphMigrationCertificate(...))` maps one saved graph
  onto one target graph inside a restricted family. The certificate binds the
  real saved and live topology digests, an injective `functor_map`
  (double writers are rejected), declared `retired_functors` and
  `new_functors`, a `state_variable_map` plus declared
  `retired_state_variables`, compatible-restart terms for the mapped
  Functors, `replay_mode` (`switch` continues forward; `counterfactual`
  requires the saved history to be complete), and the explicit
  `new_functor_rng="fresh_substream"` coupling. On the common domain the
  migration is checked before any mutation: mapped Functors keep their type,
  execution group, and required/optional/lagged relations among mapped
  Functors; saved state variables must be mapped, kept, or declared retired
  (never silently dropped); mapped variables keep units, shape, and dtype.
  The saved payload is then transformed into the target namespace (renamed
  Functor state and contracts, live state for new variables, fresh live
  state_dict for new Functors, renamed time series and per-step bookkeeping;
  retired variables keep their time series), audited with the compatible
  machinery (a certified rename changes only the declared name and the two
  name-bearing provider digests), validated, and committed atomically with
  the existing rollback. Identity migration equals exact restore; composing
  two migrations equals the direct one on the common domain.
  `GraphExecutor.migrate_from_checkpoint(...)` prepares the target plan and
  passes the certificate through.
- Delivery settings (`enable_checkpointing`, `checkpoint_frequency`,
  `checkpoint_directory`, `enable_diagnostic_export`,
  `diagnostic_output_frequency`, `diagnostic_output_path`) are incidental
  paths and cadences and are excluded from the scientific contract digest;
  every other simulator configuration key remains part of the exact restart
  identity.

Both create and restore return a receipt containing the resolved `path`,
`format_version`, `size_bytes`, file `sha256`, and canonical
`payload_sha256`. The checkpoint owner exposes `capture_checkpoint_payload`
for capturing that same format without writing a file. Its `create_checkpoint`
accepts a frozen `snapshot`; delivery never recaptures live state. A retry
against an existing target validates both payloads and returns the actual
artifact receipt only when their content matches. Conflicting or damaged
targets remain unchanged. The direct API without a snapshot still requires
an unused target. Temporary-file cleanup also runs on interruption.

A demonstrated bitwise match between an uninterrupted run
and a split restore run is dynamic evidence for that tested graph, inputs,
fixed-step protocol, software stack, and device environment. It is not a
general guarantee of bitwise identity across platforms, dependency versions,
device types, or distributed layouts.

#### 4.4.4 Calendar and Management-Time Contract

`relic\coupling\scales\temporal\calendar.py` is the adapter around the existing
`PeriodIndexer`/`CalendarHandler` calendar owner. Simulator-supported calendars
are `noleap` and `360_day`; scientific and management code must receive the
resolved simulator calendar explicitly and must not derive day of year from a
Julian-year constant.

For elapsed time `t` in seconds, `s_d=86400 s day-1`, and calendar-year length
`D_y` in days, the canonical management index is
`doy(t)=1+floor((t mod (D_y s_d))/s_d)`, with `D_y=365` for `noleap` and
`D_y=360` for `360_day`. Event windows that cross a year boundary use circular
distance modulo `D_y`. Fertilization, fire, grazing, harvest, tillage, and the
management coordinator reuse this contract; they must not create a second
calendar helper or assume 365/365.25 days independently.

### 4.5 State Management

File: `relic\core\state\manager.py`
Core Class: `StateManager`

Key Methods:

- `initialize(...)`
- `reset()`
- `get_state(...)`
- `checkpoint()`
- `restore(snapshot)`
- `create_transaction_savepoint()`
- `restore_transaction_savepoint(savepoint)`
- `apply_increment(...)`
- `apply_updates(...)`
- `update_state(...)` (convenience/integration-test merge API; not the canonical
  simulator commit path)
- `ingest_module_diagnostics(...)`
- `generate_conservation_report(...)`
- `set_conservation_ledger(...)`

`apply_updates(..., commit_context=None)` validates optional plain scalar
context before any state work. Nonempty string keys map only to `None`,
`bool`, `int`, finite `float`, or `str`; reserved ledger keys cannot be
supplied by callers. Emitted provenance events receive a copy augmented with
`conservation_checked` and signed `element_net_change`. An unevaluable ledger
is recorded as `element_net_change=None` plus `element_net_change_error`,
not zero or a successful conservation check. Recorders receive the keyword
only when their signature accepts it. Pending contexts use the existing
checkpoint grammar and replay owner; no parallel ledger is introduced.

State Transaction Constraints:

- All state changes must be atomically committed within the `StateManager`.
- `apply_updates` is the common normalization and commit owner;
  `apply_increment` delegates to it. Where schema/SCM target units are declared,
  explicit update units must be compatible. Absolute replacements use absolute
  conversion; increments use difference conversion, so temperature offsets
  are not added to temperature changes. Rate intervals are integrated once;
  declared instantaneous boundary rates retain their exact unit/time contract.
  Finite scalar Tensor intervals preserve their integration gradients.
- Budget differences, reductions, clipping disclosures and automatically
  generated provenance increments use promoted precision before subtraction
  or reduction. Stored-state dtype is unchanged. Requested transfers cannot
  certify closure when floating-point storage loses a donor/recipient change.
  The configured blocking policy uses the same ParameterManager tolerances;
  explicit nonblocking policy still reports the actual residual. Explicit
  contributor maps remain the supplied contributions, distinct from generated
  post-minus-pre attribution.
- `relic\core\state\types.py::partition_outputs` is the single classifier
  for state candidates, explicit diagnostics, and `ExternalIntent` requests.
  It preserves the existing `UpdateItem`, tuple, and mapping boundary forms;
  `StateManager` remains the owner of coercion and write-mode resolution.
- `ExternalIntent` contains immutable data descriptors only. It is neither a
  scientific variable nor a diagnostic. `apply_updates` rejects direct,
  wrapped, or nested requests and validates the whole candidate batch before
  applying any valid state values. Unknown nonnumeric values are errors.
- `StateManager(conservation_checks=...)` owns the validation of its
  conservation settings (`CONSERVATION_CHECK_KEYS`,
  `validate_conservation_checks`, `validate_conservation_setting` in
  `relic\core\state\manager.py`): `None` keeps the defaults, a bool is the
  `enabled` switch, and a mapping may carry only `enabled`,
  `raise_on_violation`, `rollback_on_failure` (bools) and `tolerance` (a
  finite real within the declared bounds of
  `validation.conservation.conservation_tolerance`, currently [1e-10, 1e-3]);
  any other type, key, or value raises `ValueError`. The simulator's
  `state_init` only merges its flat settings into this mapping through the
  same functions. The `conservation_tolerance` property and `configure(...)`
  (which accepts only `STATE_MANAGER_CONFIGURABLE_KEYS`, currently
  `conservation_tolerance`, and raises for any other key) validate through
  `validate_conservation_setting` as well. The default tolerance has one
  source, `validation.conservation.conservation_tolerance`, read without
  fallback; a missing parameter raises.
- Execution-group diagnostics retain gradients and do not enter the next
  substep's provisional scientific state. Undeclared scientific outputs are
  rejected instead of relabeled as diagnostics. New provisional Tensor states
  retain their original precision when no existing state supplies a dtype.
  Accepted-step requests are prepared and delivered by the simulator loop;
  their frozen recovery closure is described in Section 4.4.3.
- `create_transaction_savepoint()` / `restore_transaction_savepoint(savepoint)`
  are the lightweight savepoint used by the accepted-step transaction.
  `checkpoint()` / `restore(snapshot)` are the full in-memory snapshot used by
  the adaptive retry and embedded in the durable checkpoint. Neither is a
  second file format or an independently durable persistence system.
- Boundary clipping, conservation checks, and diagnostic collections are executed in this unified layer.
- Links directly with the `relic\contracts\contract_loader.py` contract system.
- Alias resolution, ownership/writer authorization, and incrementability checks
  occur on the canonical `apply_updates(...)` path when contracts are loaded.
  Contract loading is on by default and strict: a load failure raises in
  `StateManager.__init__` (Section 21.2). `update_state(...)` does no contract
  checks and is never an equivalent governed commit.

## 5. Orchestration Layer

Directory: `relic\orchestration\`

Key Files and Responsibilities:

- `graph_ir.py`: `GraphIR`, `NodeSpec`, `LinkSpec`
- `graph_io.py`: Graph file read/write
- `port_spec.py`: Port type definitions
- `ssot_catalog.py`: Variable and semantic catalog
- `node_registry.py`: Node definition and class resolution
- `validator.py`: Validation of graph connection legality
- `adapters.py`: Edge adapter registry (`AdapterRegistry`) and `ExecutableAdapterFunctor`
- `compiler.py`: Graph compilation and execution plan generation
- `executor.py`: Plan execution and state bridging
- `cache.py`: Node-level caching

Compilation Optimizations:

- `IdentityEliminationPass`
- `ConstantFoldingPass`

Execution Strategies (Actual Implementation):

- `ExecutionStrategy.PUSH`
- `ExecutionStrategy.DIRTY`
- `GraphExecutor.execute(initial_state, n_steps, dt)` and
  `compile_and_execute(..., n_steps, dt, ...)` have no default step count or
  time step.
- `GraphExecutor.set_strategy(...)`
- `GraphExecutor.mark_dirty(node_id)`
- `GraphExecutor.create_checkpoint(checkpoint_path)`
- `GraphExecutor.execute_forcing_sequence(initial_state, forcing_sequence, *,
  dt)` binds the simulator forcing cursor to the canonical digest of the
  normalized sequence and applies one snapshot per accepted step;
  `execute_forcing_sequence_from_checkpoint(initial_state, checkpoint_path,
  forcing_sequence, *, dt)` restores a mid-sequence checkpoint, requires the
  restored cursor to name that exact sequence (digest and length) and not to
  be exhausted, and consumes the remaining snapshots. For a tested graph,
  input sequence, fixed-step protocol, software stack, and device, a split run
  has matched the uninterrupted run bitwise in state and time series; this is
  dynamic evidence, not a general guarantee (Section 4.4.3).
- `GraphExecutor.execute_from_checkpoint(initial_state, checkpoint_path, *,
  n_steps, dt)`

`GraphExecutor` delegates checkpoint creation and restore to its one canonical
`RELICSimulator`. Creation requires a prepared execution. Restart rebuilds the
declared graph lifecycle from `initial_state`, strictly restores the durable
checkpoint, and then executes the requested fixed steps. `n_steps` must be a
nonnegative integer and `dt` must be a finite positive scalar.

Forcing sequences use the existing finite `Sequence[Mapping[str, Tensor]]`
contract, including stable indexed sources. Validation visits every index before
preparing execution, hashing uses the same canonical ordered tensor bytes, and
the original sequence is retained without materializing all samples. Length and
values must remain stable and repeatable throughout validation, hashing, and
execution. Each execution step copies that sample's mapping before device
alignment, so consuming it does not mutate the source dictionary. A source
without a length and repeatable integer indexing (such as a generator) is not
accepted. Restarts still compare the complete sequence digest and length and
continue at the simulator's saved consumed index. The executor adds only
constant sample storage; caller-owned arrays and output series remain separate
memory costs. The existing DataReader is not replaced by this interface.

`ExecutableAdapterFunctor` selects its input through `config.source_mode`.
The default `subprocess_outputs` reads the declared upstream Functor output;
explicit `state` reads only `source_port` admitted by the compiled input
projection, including a lowered `State:ExternalForcing` binding. The selector
is part of the registered adapter contract and is removed before invoking
the existing transformation; an omitted selector adds no default key to the
existing contract. Unknown selectors and missing inputs raise, without
switching input channels. State mode reads the simulator's projected state
view, including same-group provisional updates; it does not promise equality
with a provider's raw `UpdateItem` proposal or per-call output freshness.
Port semantics, transformation laws and StateManager ownership are unchanged.

Caching Semantics:

- Caching is disabled by default. A node is cache-eligible only when its
  existing execution declaration explicitly classifies it as
  `deterministic_pure` and its inputs and outputs do not participate in an
  autograd graph.
- `NodeCache` and `GraphExecutor` are the existing owners; no second cache
  manager or execution path may be introduced.
- An eligible node whose cache key cannot be computed or whose output cannot
  be written raises (`ExecutionError` from the governed call); caching is never
  silently disabled. `NodeCache` evicts an unreadable entry, an envelope that
  fails validation, or a payload whose classes no longer resolve with a
  `RuntimeWarning` and reports a miss; any other restore error propagates.
- Eligible keys bind GraphIR/ExecutionPlan identity, node/Functor identity,
  provider and registry contract, configuration and ParameterManager binding,
  solver contract, forcing/calendar/cursor contract, and stochastic
  substream/path identity. A changed contract must miss the cache.
- Stateful and effectful Functors, stochastic nodes, and tensors with
  `requires_grad` or `grad_fn` are never cached. A hit may replay only a pure
  value; it must never stand in for hidden-state advancement or preserve a
  misleading detached autograd flag.
- Output qualification uses `core.state.partition_outputs`: an
  `ExternalIntent` cannot enter either disk caching or DIRTY reuse. The same
  check applies to existing hits before replay; disqualified disk entries are
  invalidated through `NodeCache`. Autograd inspection includes
  `UpdateItem.value` in inputs and outputs. A non-gradient `UpdateItem`
  remains cacheable with its original update metadata.
- `NodeCache.invalidate(node_id)` invalidates entries by node ID inside this
  single owner. `GraphExecutor.invalidate_node_closure(node_id)` extends it to
  the dependency closure: the node plus every transitive consumer reached
  through plan links or shared-state reads of its declared outputs (aliases
  resolved through the StateManager contract registry); each member loses its
  disk entries, DIRTY last output, and last input hash.
- Cache keys and DIRTY input hashes also bind the provider identity consumed
  from the simulator's committed provider view: the node's implementation,
  contract, and binding digests, the same triple for its transitive upstream
  providers, and the deterministic-pure RNG contract. `GraphExecutor` records
  the reconciled (simulator, epoch) and re-reads the view before each governed
  call; a new epoch with unchanged identities keeps every entry, while a
  changed digest invalidates that node's closure so no old-epoch result can
  serve a new step and unrelated nodes keep their entries.

### 5.1 Structural Ecology Assertion Interface

Structural assertions extend the existing GraphIR validation path; they do not
introduce a second framework.

Theoretical Document Entry: `docs\source\theory\structural_ecology_foundation.md`.

Sole Interface Constraints:

- Top-level optional `GraphIR` field: `assertion`.
- One assertion item uses `phi`, `op`, `scope`, optional `context`, and optional
  `audit_key`.
- Multiple items may be provided under `assertions`, `items`, or `checks`.
- When the top-level field is empty, the validator may read
  `metadata.structural_assertions` or
  `metadata.mandatory_structural_assertions`.
- Building a secondary assertion entry (e.g., `assertions_v2`, standalone assertion services, or parallel compilers) is prohibited.

Semantic Mapping (from Theory to Existing Implementation):

- `phi` (Object Layer Proposition): Syntactic assertion checks inside `graph_ir.py` and `validator.py`.
- `op` (Operational Protocol): Parsed dynamically within `validator.py` onto existing Node IDs, `class_type` entries, SSOT variables, and `NodeDefinition` metadata protocol lists.
- `scope` (Extrapolation Boundary): Validated for type limits and non-null states inside `validator.py`.

Unified Audit Output (Strictly Single Exit):

- In memory, compiler audit data is stored at
  `ExecutionPlan.report.metadata["audit_summary"]`.
- `plan_to_dict(...)` serializes that summary as `report.audit`; this serialized
  path is not a field on the `CompileReport` dataclass.
- Conservation and topology evidence is folded into the same audit summary
  instead of being emitted by a second report service.
- The API and CLI serialize the same audit payload so structural evidence has
  one schema.

Compatibility and Behavioral Constraints:

- Ordinary `compile_graph(..., strict=True)` does not itself require an IPHS
  assertion or a thermodynamic certificate. Explicit structural and advanced
  requests retain their applicable assertion obligations; the current metadata
  trigger limitation is documented in Section 5.2.
- Malformed assertions are reported through `validate_graph(...)` using the
  existing issue model.
- Assertions add structural validation and audit evidence; they do not modify
  solvers, timestepping, or Functor execution.

### 5.2 RELIC-IPHS-DEC Gate 0 Enforcement Layer

The RELIC-IPHS-DEC layer provides the canonical operators and checks for
declared DEC/IPHS structures. It does not replace the simulator, parameter
system, state manager, validation engine, graph compiler, or orchestration
model, and it does not require every mature process to be reformulated as
IPHS. It exposes structural objects, certificates, and validators through
existing package boundaries.

Current canonical entry points:

| Function | Existing implementation |
| --- | --- |
| Typed cochain spaces, DEC complexes, Hodge specs | `relic\computation\operators\dec\forms.py`, `complex.py`, `hodge.py` |
| IPHS structure specification and assembly | `relic\computation\operators\iphs.py` (`IPHSStructureSpec`, `assemble_iphs_structure(...)`) |
| Stochastic IPHS/SDAE structural specs | `relic\computation\operators\stochastic_sdae.py` |
| IPHS equivalence certificates | `relic\computation\operators\iphs_certificate.py` |
| Structure-aware splitting metadata | `relic\computation\solvers\geometric.py` (`IntegrationStage`, `IntegrationPlan`) |
| Structural validation rules | `relic\validation\framework\rules\*.py`; decorated wrappers are registered with the existing `ValidationRegistry` |
| GraphIR structural assertion dispatch | `relic\orchestration\validator.py`, which calls the shared rule implementations from `GraphIR.assertion` or `graph.metadata.structural_assertions`; applicable obligations depend on actual structures and explicit claims |
| Typed cochain state-write boundary | `relic\computation\operators\dec\adapters.py`, lowering typed cochains to existing `UpdateItem` records |
| Optional structural protocol | `relic\core\functor\iphs_protocol.py` (`IPHSDECFunctor`); this augments, and never replaces, the four-contract/FunctorBase lifecycle |
| IPHS metadata normalization | `relic\core\functor\iphs_metadata.py`; generated canonical-expression metadata is descriptive and is not strict-runtime evidence |
| Global structural compile audit | `relic\orchestration\compiler.py`, stored in memory at `ExecutionPlan.report.metadata["audit_summary"]` and serialized by `plan_to_dict(...)` as `report.audit` |
| Compiler facade | `RELICModelCompiler`, which delegates to `compile_graph(...)` and is not a second compiler |

Supported canonical structural assertion ops:

```text
cochain_complex_check
hodge_psd_check
iphs_structure_check
chain_map_check
entropy_step_check
port_power_units_check
port_matching_check
overflow_routing_check
equivalence_certificate_check
reaction_network_check
stochastic_sdae_check
state_domain_check
observation_diagnostic_check
```

Structural semantics:

- The supported operations above are not an unconditional checklist for every
  ordinary model. The target contract selects obligations from actual
  structures and explicit scientific claims, without duplicate operations.
  Ordinary `compile_graph(..., strict=True)` validates the declared graph,
  registered implementations, ports, dependencies, writer ownership,
  assertions, and compile-run witness; that argument alone does not request
  strict dPH or GENERIC evidence.
- `strict_structural_assertions` applies strict checks to the structures and
  assertions actually declared or required. It does not request an IPHS
  formulation or add unused Hodge, entropy, or certificate objects.
- `iphs_certificate.py::resolve_iphs_level_requirement(...)` is the single
  level-request normalizer. Validation rules consume it directly;
  `validator.py::_required_iphs_level(...)` adapts Graph scopes and issues,
  and the compiler reuses that adapter. Descriptive `iphs_level`
  values 0-3 and an explicit zero minimum do not request runtime certification.
  A positive graph-metadata or `iphs_assembly` `required_level` requires
  actual equivalence/runtime evidence. It cannot be satisfied by a local
  matrix-only audit. Invalid levels are rejected even with no IPHS assertion.
- The `iphs_structure` rule validates local mathematical structure, not a
  process runtime certificate. Its IPHS-specific `strict=true` requests at
  least Level 4 structure; ordinary graph content strictness does not.
  An explicit positive level requires a non-empty structure payload.
  An assertion-local `iphs_structure_check.context.required_level` selects
  local structural conditions only; it does not add a graph-wide runtime
  certification request or replace `equivalence_certificate_check`.
- Explicit Level 4/5, `phase8_strict`, and strict IPHS assembly requests retain
  their role-appropriate mathematical and executable-evidence requirements.
  Nested false flags and lower thresholds cannot weaken those requirements.
  A declared certificate still receives its applicable complete map, parameter,
  and interface checks without promoting a descriptive level to Level 4.
- `cochain_complex_check` enforces `D_next D_k = 0`; strict mode also requires the complete derivative and support sequence implied by the declared dimension, finite tensors, valid masks/orientations, and positive measures.
- `hodge_psd_check` enforces compatible shape, finite values, symmetry, and positive semidefiniteness of the supplied matrix. Strict mode also requires an explicit `role` and `definiteness`, same-complex primal/dual complementary spaces, compatible domain/orientation/boundary role, ParameterManager-backed parameter and metric keys, convertible operator/domain/codomain units, and tolerances not above `sqrt(machine epsilon)`. A strict `metric_hodge` must be symmetric positive definite; only a strict `material_hodge` may be positive semidefinite, and then its linearly independent `kernel_basis` must have dimension equal to the numerical nullity with `H K` vanishing within tolerance. These are checks of the supplied matrix, not of mesh provenance or refinement convergence.
- `iphs_structure_check` separates two formulations. Dissipative pH validates
  skew `J`, symmetric PSD resistance `R`, and `xdot=(J-R)e_H+Gu`; GENERIC
  validates skew `J`, symmetric PSD mobility `M`, and
  `xdot=J e_E+M e_S+Gu`. Here `J,R,M:X*->X`, `G:U->X`,
  `G^T:X*->U*`, and a feedback map `D:U*->U` produces
  `R_port=G D G^T`. Strict level 4/5 assembly requires runtime-linked
  contributions and the gradients needed by the claimed level; an empty
  assembly is a failure.
- The strict reaction target permits one of two mutually exclusive
  representations. In the independent representation, `N:F->X`, `r in F`,
  and `+N r` is appended once. Its balances
  are `dH/dt=-e_H^T R e_H+y_H^T u+e_H^T N r` for dissipative pH and
  `dE/dt=y_E^T u+e_E^T N r`,
  `dS/dt=e_S^T M e_S+y_S^T u+e_S^T N r` for GENERIC. In the absorbed
  representation, the entire irreversible term represents the reactions,
  `N r=-R e_H` or `N r=M e_S` (checked componentwise by
  `IPHSStructureSpec`); the independent `+N r` term is then absent. A certificate must bind the selected representation to
  one executable owner and prove any projection/equivalence and residual route.
  Executing both representations is a validation error. The implemented
  representation selector and its process-level evidence boundary are specified
  once in Section 7.3 below; pointwise mutual exclusion alone is not a
  time-integrated physical-closure proof.
- `chain_map_check` checks derivative commutation `R_{k+1} D_source = D_target R_k` within `tol` for the supplied matrices; for ordinary maps it does not check ledgers. A declared BIHSS morphism also checks, on caller-supplied sample vectors and scalars, ledger preservation `C_t Phi (+ C_port with a declared open route) = C_s`, the mapped state, energy/entropy storage error within `storage_tol`, port-map and port-power agreement within `power_tol`, and entropy-production non-decrease within `entropy_tol`. Passing samples do not prove these identities for every state or grid. `LinearChainMap` (Section 8.4) separately checks `C_t (u M) = C_s`; an ordinary conservative remap, derivative commutation, and ledger preservation are three distinct conditions.
- `port_matching_check` validates port declarations only: typed effort/flow cochain metadata, parseable units with `[power_units]=[effort][flow]`, equal effort/flow `n_dofs`, sign/kind/role/route vocabulary, and, per matched group, at least one `into_system` and one `out_of_system` port whose declared incidence weights satisfy `|sum_i s_i w_i| <= tolerance`, with one power unit, port kind, element, and flow-space signature. It reads no effort or flow values and does not evaluate the per-step power sum `sum_i e_i^T f_i`; a passed check is not an energy or mass balance. Unmatched ports must be declared open boundaries; strict empty ports require an explicit `closed_system=true` declaration.
- `overflow_routing_check` enforces explicit routing or rejection for overflow,
  clipping residuals, unconsumed fluxes, and balance residuals. A strict empty
  payload always fails. `no_residuals=true` is not an implemented witness and
  cannot replace at least one complete canonical route.
- `state_domain_check` and `observation_diagnostic_check` require non-empty
  strict declarations and coverage of the state or observation variables
  claimed by the Graph role; an empty mapping is not evidence.
- `equivalence_certificate_check` validates the IPHS equivalence or capability
  actually claimed, including applicable input/output, parameter, composition,
  runtime, reference, and level evidence. Level 4/5 evidence must bind to
  import-resolvable runtime owners, concrete state/value keys, units, theory
  anchors, ParameterManager metadata, and existing module-specific tests;
  prose and assigned-zero placeholders are rejected. An explicit advanced
  requirement cannot be weakened by nested `strict=false` or a lower
  `required_level`. Ordinary model availability does not require inventing an
  IPHS certificate.
- `reaction_network_check` separates strict stoichiometric, basis, rate, unit,
  and physical open-ledger checks from the affinity obligation. Direct
  `validate_reaction_network_payload(..., strict=True)` retains the full
  affinity contract by default. Its caller may set `require_affinity=False`
  for an ordinary structural model without an affinity declaration; the
  Graph adapter derives that requirement from the actual requested claims.
  This never bypasses `C_h N_h=0`, finite non-empty N/C/r, basis ordering,
  declared units, or actual open-exchange accounting.
- Declared affinity, sign, order, and unit fields remain validated. Explicit
  `require_affinity=True` increases the obligation; False cannot override an
  advanced outer request. A non-strict numerical-point audit may omit unused
  metadata, but cannot use that omission to hide invalid fields it provides.
  The physical convention is `A_phys=-N_h^T mu`, and `A_phys^T r` is power;
  the numerical audit does not construct chemical potentials or a kinetic law.
  `[A_phys]` is energy per reaction extent, `[r]` is extent per time on the
  declared support, and their aggregate pairing must be nonnegative within
  tolerance. Elementwise positivity is a stronger optional condition.
- `[N]` Without an actual temperature and entropy-balance binding, affinity
  power alone does not establish reaction entropy production. The closed-box
  runtime point below supplies its own binding. For the common positive-temperature isothermal
  case, `sigma_rxn=A_phys^T r/T`; this is not an unconditional formula for a
  nonisothermal model. Open exchange still requires a physical route, finite
  exchange and ledger rates, consistent units, and `C_h q_open` equal to the
  declared ledger rate. A route name or diagnostic residual does not close a
  physical budget.
- `stochastic_sdae_check` remains one ValidationRegistry rule but validates the
  requested semantics independently: `continuous_sde`, `marked_jump`,
  `memory`, `ale`, `descriptor_dae`, and `topology_change`. A declaration from
  one class cannot satisfy another class's non-empty obligation. The rule also
  enforces stochastic owner binding, FDT covariance evidence, conservative
  noise-flux routing, descriptor constraints, jump/reset ledgers, passive
  memory kernels, ALE Hodge-energy correction routing, topology remap audits,
  and restart/RNG reproducibility evidence. For `continuous_sde`,
  `marked_jump`, `memory`, `ale`, `descriptor_dae`, and `topology_change`
  the fail-closed `[N]` expectation is lifted only by an
  `execution_bindings` entry whose `functor_class` resolves to an importable
  Functor with the matching category attribute (`stochastic_category`,
  `memory_category`, `geometry_category`, `constraint_category`, or
  `topology_category`) and,
  for the stochastic semantics, an admissible `solver_name`
  (`euler_maruyama`; `direct_ssa`/`gillespie`/`next_reaction`) and a
  `solver_owner` that is an existing `relic.stochastic` class with `sample`;
  whose `forward_owner` is that class's `forward`; whose `state_vars` are
  declared by the stochastic terms, jump resets, memory kernels, ALE
  geometry corrections, descriptor constraints, or hybrid topology
  transitions and cover the class `STATE_VARIABLE`/`STATE_VARIABLES`; whose
  ownership scope is `rng_state_scope=functor_extra_state` (stochastic),
  `history_state_scope` in `state_manager`/`functor_extra_state` (memory),
  `geometry_state_scope=state_manager` (ALE),
  `algebraic_state_scope=state_manager` (DAE, with `descriptor_restart`
  declaring `projection_on_restore=true`), or
  `topology_state_scope=state_manager` (topology change, with
  `topology_restart` naming `checkpoint_owner`
  `relic.core.simulator.checkpoint`, `graph_epoch_owner` the registration
  owner `relic.core.simulator.registration`, and `event_queue_state_key`
  the delivery queue owner
  `relic.core.simulator.execution_loop.retry_external_deliveries`);
  descriptor constraints may declare `algebraic_elimination` for index-1
  systems solved by the implicit function theorem; and whose
  `validation_test` names an existing test under `test\`. The rule resolves
  owners but the named test carries the executed evidence. A payload that
  declares random terms or jump resets must name a `relic.stochastic`
  runtime owner; a purely deterministic hybrid payload names its executing
  RELIC owner. Deterministic management events (fertilization, harvest)
  keep their calendar/overlap contract and are not marked jumps.
- Every `IntegrationStage` declares `solver_name`, `structure_kind`,
  `dt_fraction`, and `coupling_mode`; `IntegrationStage.from_dict` rejects a
  payload without them.
- `IntegrationPlan.validate()` checks staged, lagged, Lie, or Strang coupling metadata without reimplementing simulator scheduling. For `scheme="strang"` it only requires matching `operator_kind` and `dt_fraction` at the first and last stage; it does not check a full palindrome. It checks the declared stage graph only: its report carries `execution_semantics="metadata_only"`, `verified=False`, `runtime_ready=False`, and `solver_execution_count=0`. A structurally valid plan is not evidence that a solver ran or that the Simulator executes that schedule.
- Runtime evidence is separate from that static validation.
  `RELICSimulator.get_composition_plan()` derives simulator-owned stages from
  the committed fast order and the medium/slow groups, including stage times,
  sub-steps, triggers and exchanges. Restatement, composition-mean and exchange
  commits are not extra solver stages. For a declared
  `graph.metadata.integration_plan` whose stages all use `runtime_owner=simulator`,
  GraphExecutor compares the canonical schedules before stepping and again
  after restore. Mixed runtime owners or a mismatch are refused; no simulator
  stages gives `not_simulator_owned`, and no declaration gives `not_declared`.
  A match is exposed by `integration_plan_witness` with both schedule digests;
  its `executed_stage_check` compares observed writer commit contexts with the
  declared stages. A changed provider epoch that changes the schedule marks
  the witness `superseded` and stops further steps. This is bounded execution
  evidence, not a proof of a numerical order or an IPHS capability.
  The plan is derived from the existing exact-restart contract (including
  splitting controls and provider bindings); checkpoint format 3 stores no
  second composition-plan field. Sub-flow metadata or accessed ParameterSpec
  changes, even descriptive fields, can change that scientific identity.
- `advance_reaction_network(...)` supports declared runtime
  `conservation_matrix` validation and records an audit when `C_h` is missing.
  `C_h N_h != 0` is rejected when strict conservation is evaluated. A missing
  matrix is fail-closed only when the Phase 8 caller sets
  `require_conservation_matrix=True` and does not declare an allowed
  `open_system_route`; this is a caller obligation, not the function's
  unconditional default behavior.
- Open scientific ledgers that use `advance_reaction_network(...)` must
  declare its existing `open_system_route` contract: `port`, `boundary_export`,
  `managed_export`, `diagnostic_residual`, or `observation`. Route labels alone
  do not establish physical closure. Actual exchanges require numerical
  budgets; diagnostic residuals disclose discrepancies and do not close them.
  Mature pool models without `N_h` verify their actual increments and budgets
  directly instead of constructing identity stoichiometry.
- The low-level `advance_reaction_network(...)` compatibility helper permits a
  declared open route to replace `C_h` when its caller selects that policy.
  This does not weaken the current strict reaction rule: its closed and open
  payloads both provide `C_h` and prove `C_h N_h=0`; open payloads then add the
  separately routed exchange ledger described above.
- `assemble_iphs_structure(...)` embeds local Functor contributions into the
  declared global state basis and concatenates independent port and reaction
  columns. A skew port interconnection over the ports declared `internal`
  contributes `G_int C G_int^T` to `J`, where `G_int` holds the columns of `G`
  of those ports; a PSD feedback `u_int=-D y_int`, `y_int=G_int^T e_H`,
  contributes `G_int D G_int^T` to resistance `R`, hence the drift term is
  `-G_int D G_int^T e_H`; it never enters GENERIC mobility `M`. A payload with
  `C` or `D` must declare an explicit `internal` or `external` role for every
  column of `G`; ports without roles raise `ValueError`, and ports not declared
  `internal` remain external inputs `u` (van der Schaft and Jeltsema 2014,
  Sect. 6.2: the interconnection acts only on the interconnected ports).
  `formulation="generic"` with `port_dissipation_D` is rejected. Contributions
  are embedded by exact `state_vars` name: contributions naming the same
  variable act on one coordinate, and different names are never identified,
  aliased, or unit-converted. Declared `energy_gradient`/`entropy_gradient`
  contributions are summed coordinatewise; the assembler constructs no `H`,
  `E`, or `S` functional, effort, affinity, or shared-state equivalence. It has
  no `storage_kind` argument, so the Level 4 reaction pairing is reachable only
  through a directly constructed `IPHSStructureSpec`.
- `compile_graph(...)` validates declared system-level IPHS assembly from `graph.metadata.iphs_assembly` without creating a second compiler. The metadata may include local contribution `state_vars`, `port_interconnection_C`, and `port_dissipation_D`. The compiler forwards `poisson_operator` and, only together with `constant_field` or `state_field`, `poisson_field` (an importable reference `package.module:attribute` resolved and checked to be callable at compile time), `poisson_domain` (a non-empty list of equal-length numeric state vectors), and `jacobi_tol`; any other operator carrying these keys is rejected. The tolerances `skew_tol`, `psd_tol`, `degeneracy_tol`, and `jacobi_tol` may be declared in the metadata; a declared null or non-number is rejected, and an omitted key selects the parameter default. It may also collect explicitly declared node-local `config.iphs_contribution` payloads when `collect_node_contributions=True` or `contribution_node_ids` is supplied; this is structured declaration assembly, not inference from free-form theory text.
- GraphIR validation rejects malformed assertions and mismatched audit keys,
  preserves assertions through optimization, and revalidates the optimized
  graph before compiler acceptance. Where structural obligations are required,
  missing or duplicate operations fail. User-authored `passed=true` fields
  remain unverified declarations, including for ordinary models.

IPHS standardization level:

- The only runtime owner for process level is `IPHSEquivalenceCertificate` in `relic\computation\operators\iphs_certificate.py`.
- Every migrated certificate declares `iphs_level` and `iphs_level_evidence`. The Phase 8 inventory reads those fields; it must not infer an independent level taxonomy.
- Certificate validity and executable availability are separate inventory
  axes. `implementation_status` is derived from the certificate-owned runtime
  claim and may be `implemented`, `not_implemented_fail_closed`, or `unknown`;
  class presence or a level number alone must not decide availability.
- A row is `not_implemented_fail_closed` only when its certificate declares
  level 0, `operational_claim=false`, `runtime_uses_iphs=false`,
  `legacy_removed=false`, a `replacement_entrypoint` beginning with `none_`,
  empty input/output/composition-port declarations, and an `iphs_expression`
  beginning with `[N]`.
- Such an explicit unavailable row remains certificate-invalid,
  `runtime_verified_level=0`, and `runtime_evidence_status=descriptive_only`;
  its runtime I/O and state-advance fields are
  `not_applicable_runtime_unbound`. This is truthful non-implementation
  evidence, not a successful executable certificate. Certificate checks apply
  to each declared scope. Descriptive levels 0-3 and
  `runtime_verified_level=0` do not by themselves prove that an ordinary mature
  model is unavailable; executable availability requires evidence from its
  actual governed runtime, state updates, and outputs.
- The evidence-key hierarchy accumulates from interface evidence toward strict
  evidence: `0` ordinary empirical, `1` IPHS-compatible, `2` IPHS-wrapped,
  `3` IPHS-structured, `4` strict dissipative pH, and `5` strict GENERIC.
  The level-4 and level-5 governing equations are formulation-specific and
  mutually exclusive; level 5 does not execute the level-4 `R` term in
  addition to `M`.
- For levels 1-3, `runtime_replacement_map.runtime_uses_iphs` and `legacy_removed` must both be `false`; these fields are verified strict-runtime claims, not declaration markers. Levels 4-5 may set both to `true` only with import-resolvable runtime owners, removed-symbol evidence, ParameterManager bindings, and an existing module-specific validation test.
- `level_4` is non-downgradable `formulation=dissipative_ph`: it requires
  `M=0`, symmetric PSD `R`, and an executable closed-port check. For the
  reaction-free single-point structural profile (`N=0`),
  `dH/dt=e_H^T(J-R)e_H=-e_H^T R e_H <= 0`; nonzero independent reactions
  also require `e_H^T N r=0` when `H` is total energy, or
  `e_H^T N r<=0` when `H` is a dissipative/free-energy storage, unless the
  difference is carried by a typed thermal or chemical port. Typed port-power
  evidence is required when `G` is nonzero.
- `level_5` is non-downgradable `formulation=generic`: it requires `R=0`, an
  energy/entropy pair, `J e_S=0`, `M e_E=0`,
  `e_S^T M e_S>=0`, and Jacobi evidence. With `N=0`, these imply closed-system
  energy conservation and entropy non-decrease. A nonzero independent reaction
  term also requires `e_E^T N r=0` and `e_S^T N r>=0`.
  `[I]` The existing spec supports four `poisson_operator` policies.
  `zero_field` evaluates `J(x)=0` on the declared finite coordinate basis; its
  state derivatives and Jacobi residual vanish identically, and each local `J`
  and reversible feedback `C` must be exactly zero before dtype conversion and
  assembly (cancellation or a numerical tolerance is not a field-identity
  proof). `constant_field` executes the declared constant `J` at every state (an
  optional callable is checked equal to `J` on declared `poisson_domain`
  samples), so `dJ/dx=0` and the Jacobi identity holds identically
  (`constant_field_exact`). `state_field` requires `poisson_field`,
  `poisson_domain`, and `reversible_operator_state_dependence="state_dependent"`;
  it checks skew symmetry and the autograd Jacobi residual at each declared
  sample within `jacobi_tol` (`sampled_domain_jacobi_residual`). At a Level 5
  runtime point it also evaluates the skew defect `max(abs(J + J^T))` and the Jacobi
  residual at the bound runtime state (`runtime_point_jacobi_point=bound_state`),
  or at the first declared sample when no state is bound
  (`declared_reference_sample`), and rejects the point when the skew defect
  exceeds `skew_tol` or the Jacobi residual exceeds `jacobi_tol`
  (`runtime_point_skew_residual`, `runtime_point_jacobi_residual`). The evidence
  is the finite set of declared samples plus the evaluated runtime points, not a
  continuum proof. The production drift evaluates `J(x)` on the autograd path,
  so it carries the `dJ/dx` and parameter sensitivities of the executed map;
  only the audit checks read detached values. The IPHS tolerances default to
  the parameters `validation.iphs.skew_tolerance`, `psd_tolerance`,
  `degeneracy_tolerance`, and `jacobi_tolerance` (current configuration example:
  1e-10, 1e-10, 1e-10, 1e-8); explicit values override them. The
  default `matrix_snapshot`, a `constant` label, a detached tensor, or a text
  certificate establish no field identity. `[I]` Process-level E/S effort
  linkage is executable for GENERIC runtime points on any of the three bases:
  `IPHSStructureSpec.validate(required_level=5, state=x, energy_value=E,
  entropy_value=S, reaction_rates=r)` binds the actual process potentials,
  verifies `J e_S=0`, `M e_E=0`, `e_S^T M e_S>=0`, the closed-reaction pairing
  `e_E^T N r=0`, `e_S^T N r>=0`, the absorbed `M e_S=N r` identity and the
  selected Jacobi basis, and returns `runtime_verified_level=5` with
  `process_level_status=runtime_point_verified`. `[N]` No production process
  declares a nonzero reversible field yet. Level 5 runtime points with declared
  port columns and any `matrix_snapshot` reversible operator remain
  fail-closed. In this
  implementation the `J` field carries the theoretical GENERIC Poisson operator
  commonly denoted `L` when `formulation=generic`.
- `[I]` Process-bound certificate acceptance: `IPHSEquivalenceCertificate.validate(...,
  runtime_point_audit=...)` accepts the audit returned by the strict runtime
  point only together with a governed-execution witness
  (`governed_execution.entry="FunctorBase.__call__"`, the executed Functor,
  the committed non-diagnostic output variables, and the accepted step count).
  Without that witness a Level 4/5 certificate stays `runtime_binding=descriptive`
  and keeps the `process_bound_runtime_unavailable [N]` issue; with an
  inconsistent witness it is rejected. The witness is a mapping the caller
  assembles from an executed step (for example from
  `RELICSimulator.get_variable_provenance`); the certificate owner checks only
  its fields and does not query StateManager, so it is inspectable evidence,
  not an automatically issued or unforgeable per-step certificate. The layers
  prove different things: the structural pre-gate checks operator matrices;
  the runtime point checks the continuous-time drift residuals at the bound
  pre-step state (not the discrete time step); `FunctorBase.__call__` and the
  StateManager commit are the governed execution and state advance; the
  caller witness links one audit to one committed step. The rule
  `equivalence_certificate_check` forwards a mapping payload's
  `runtime_point_audit` to the certificate owner. `generic_degeneracy` accepts
  `reversible_operator_state_dependence=zero_field` only with
  `jacobi_identity_basis=canonical_zero_field`, `constant` only with
  `constant_field_exact`, and `state_dependent` only with
  `sampled_domain_jacobi_residual` plus an integer `jacobi_sample_count >= 1`
  (bool rejected) and a positive finite `jacobi_tolerance`; this evidence
  check does not examine runtime-point residuals. For the sampled basis the
  Level 5 check of a supplied `runtime_point_audit` requires
  `jacobi_max_residual` and `runtime_point_jacobi_residual` to be finite and
  at most the audit's own `jacobi_tolerance` (the operator's `jacobi_tol`,
  by default `validation.iphs.jacobi_tolerance`; it is not compared with the declared
  evidence `jacobi_tolerance`), a positive integer `jacobi_sample_count`, and
  only a finite `runtime_point_skew_residual`. The runtime-point skew defect
  is bounded by `skew_tol` (by default `validation.iphs.skew_tolerance`) inside
  `IPHSStructureSpec`, which raises `ValueError` otherwise;
  `runtime_replacement_map`
  may declare `legacy_never_existed=true` for a process introduced directly on
  the canonical runtime, in which case `removed_legacy_symbols` may be empty.
  A runtime-point certificate is identified by its `constructor_selection`
  field (a mapping of constructor keyword to JSON scalar, allowed only at
  Level 4 or 5 and checked against the real constructor signature); the
  process-contract generator reads this field. The first Level 5 runtime point
  is the constructor selection
  `ChemistryFunctor(physical_boundary="closed_constant_volume")`
  (`docs\source\theory\iphs_certificates\atmosphere_chemistry_chemistry_functor_closed_constant_volume.yaml`);
  its shared audit tolerance is `validation.iphs.runtime_point_tolerance`. Its
  domain is a closed, rigid (constant-volume), adiabatic `NO + O3 -> NO2 + O2` box on
  scalar state `(n_NO, n_NO2, n_O3, n_O2, U)` with fixed `V`: `n_i>0`, `V>0`,
  temperature inverted inside `[temperature_min, temperature_max]` on one
  NASA7 interval, affinity `A>0`, and forward mass action only. Before emitting
  explicit Euler increments, the actual tensor candidate `x + dt * drift`
  must satisfy the same thermodynamic domain, including `n_i>0`, valid
  temperature inversion and `A>0`. The pure property check neither evaluates
  a second drift nor advances state; it is not a discrete entropy or stability
  certificate. The ordinary
  `ChemistryFunctor` main certificate remains Level 3.
- A reaction tendency wrapped as `N_h r_h`, especially an identity stoichiometry wrapper, is at most `level_2` unless the certificate proves true stoichiometry, effort/affinity direction, and thermodynamic closure.
- `[M]` The reaction-free `N=0` Level 4 matrix/effort pre-gate is executable.
  A pre-gate alone keeps process acceptance `[N]`; the runtime-linked Fickian
  selection described below supplies the additional storage, canonical-drift,
  governed-execution and StateManager witness for its declared scope. `[I]` Level 5 GENERIC process acceptance is executable
  through the runtime point plus the process-bound certificate witness above
  on the `zero_field`, `constant_field`, and `state_field` bases; `[N]` Level 5
  acceptance with port columns or a `matrix_snapshot` reversible operator
  remains fail-closed, and no production process binds a nonzero field. `[I]` Nonzero absorbed reaction coupling into the Level 5 runtime
  is verified by the GENERIC pairing `e_E^T N r=0`, `e_S^T N r>=0` and
  `M e_S=N r`. `[I]` Level 4 dissipative-pH process acceptance is executable
  through `IPHSStructureSpec.validate(required_level=4, state=x, energy_value=H)`:
  the effort `e_H=dH/dx` is derived on the bound process state, the existing
  single-point power audit returns `runtime_verified_level=4`, and the same
  process-bound certificate witness applies. Explicit `energy_effort` alone
  remains the pre-gate and keeps its `[N]` rejection. The first Level 4 runtime
  point is the constructor selection
  `FickianDiffusion(strict_runtime_point=True)`
  (`transport_diffusion_fickian_diffusion_strict_runtime_point.yaml`), restricted
  to `spatial_dims=1`, one unbatched rank-1 state with at least two cells,
  finite positive diffusivities, and a zero-flux (`zero`, `neumann`, `open`) or
  `periodic` boundary. Its storage `H=0.5 c^T c` has units kg2 m-6 and is a
  quadratic norm, not joules. The runtime point rebuilds
  `R_h=d_h^T diag(K_f) d_h/dx^2` and rejects the step when `-R_h e_H` does not
  reproduce the source-free production finite-volume rate at the pre-step
  state; a nonzero `source_term` is rejected under this selection before
  integration (fail-closed), so the audited drift is the whole production rate.
  The main `FickianDiffusion` certificate
  remains Level 3. `[I]`
  The Level 4 reaction pairing is executable at the operator level: a bound
  runtime point with `reaction_rates` requires an explicit
  `IPHSStructureSpec.storage_kind` (`total_energy` requires `e_H^T N r=0`,
  `free_energy` requires `e_H^T N r<=0`), an independent reaction enters the
  power balance as `e_H^T N r`, and an absorbed reaction must satisfy
  `N r=-R e_H`; `[N]` no production dPH process with a reaction is bound yet,
  and Level 4 points with external port columns remain fail-closed because
  `port_input` cannot accompany bound potentials (internal port columns
  eliminated through `C`/`D` feedback are accepted by the operator).
- Never copy a generated level distribution into stable architecture claims.
  Read the current distribution from
  `analysis_results\phase8_scientific_process_audit.json` and verify its recorded
  source commit/tree before quoting it.

Strict IPHS composition boundary:

- `IPHSStructureSpec.canonical_drift(state=x, energy_value=E, entropy_value=S)`
  can derive runtime efforts from scalar autograd values connected to the same
  state. dPH supplies only its energy/storage value; GENERIC supplies both E and S.
  This route is mutually exclusive with stored or explicit precomputed efforts.
  Detached values, unrelated graphs, incompatible dtype/device, and non-scalar
  potentials are rejected. The caller constructs potentials with autograd enabled;
  the operator does not reconstruct missing graphs or physical definitions.
  Runtime autograd values belong to the scientific call, not static GraphIR
  metadata or a checkpoint. This low-level binding is not a process certificate
  or a StateManager commit, and strict process gates still require those links.
- FickianDiffusion's Level 3 audit is an unweighted concentration norm. It
  includes periodic seams and the existing Dirichlet/extrapolated half-cell
  boundary losses, using production coefficient normalization and harmonic
  means. Its units are not joules or thermodynamic entropy units. The local
  diagonal Hodge material map preserves coefficients and dtype and rejects
  unmapped DOF counts; it is not the global FV generator. Manager-selected
  diffusivity is read through one current-value entry point by production and
  audits; Hodge candidate construction does not mutate that binding.
- RELIC can compose declared strict IPHS contributions into a global structural form, but it cannot infer missing physical closure laws from ordinary empirical Functor code.
- A module participates in automatic strict field generation only when it declares the state basis, storage or free-energy functional, effort basis, reversible structure, dissipative structure, port incidence, reaction stoichiometry, units, ledgers, and residual routing needed by its role.
- A reaction contribution based on `N_h r_h` is not automatically strict IPHS.
  A ParameterManager-backed empirical rate and ledger routing are necessary but
  not sufficient: strict status also requires the formulation-specific
  energy/entropy pairing and proof that the reaction is executed exactly once.
- Empty `energy_terms()` or `entropy_terms()` means the module may still provide a structural contribution, but it must not be documented as strict IPHS/GENERIC energy or entropy closure.

Prohibitions:

- Do not create `assertions_v2`, standalone structural assertion runners, or parallel validation services.
- Do not create a second port schema. Extend or view the existing `PortSpec` and `PortSpecIPHS` contracts.
- Do not introduce a second compiler. Use `compile_graph(...)`; `RELICModelCompiler` is only a facade.
- Add IPHS/DEC checks to the existing shared rule implementations and decorate
  their framework wrappers for `ValidationRegistry`; do not create a parallel
  registry or a second GraphIR assertion service.
- Do not route IPHS residuals through diagnostics if they change prognostic state.
- Do not hide open-system exchange inside reaction or transport updates.

### 5.3 Dynamic Evidence and Release Claims

Architecture capability and migration completion are separate claims. A
certificate reports `runtime_verified_level` 4 or 5 only when
`IPHSEquivalenceCertificate.validate(..., runtime_point_audit=...)` accepts a
bound runtime-point audit together with the governed-execution witness and all
strict evidence maps; inventory presence, generated metadata, a certificate
object, a strict graph compile, or an `N_h r_h` wrapper alone is insufficient.
The Phase 8 inventory maps each process class to one main certificate;
constructor-scoped runtime-point certificates are not inventory rows, so a zero
Level 4/5 inventory count does not mean the named runtime points are absent,
and a passing named runtime point does not promote its main process
certificate.

Use the following generated evidence instead of freezing counts in this SSOT:

- `analysis_results\phase8_scientific_process_audit.json` for process inventory
  and certificate-derived levels.
- `analysis_results\final_closeout_evidence_manifest.json` for the commands,
  return codes, artifact hashes, and explicit exclusions of a closeout run.
- `analysis_results\dependency_audit_summary.json` for dependency policy status.

Publication provenance has one owner:
`scripts\publication\02_validate_g1_evidence.py`. Reuse its source-fingerprint
logic; do not add a second git/hash collector. The compatibility inventory
`scientific_functor_inventory.json` remains a bare row list for existing
consumers. `phase8_scientific_process_audit.json` is the provenance-bearing
envelope with `provenance`, `summary`, and `rows` fields.
Each row records `implementation_status` and
`implementation_status_reason` independently from certificate validity and
IPHS level. Generated summaries report that partition; they must never infer
implementation from row count, level distribution, or certificate-file
presence.
`final_closeout_evidence_manifest.json` is a thin G8 index over the executed
evidence, not a second store of duplicated test output.

Generated claims use the following boundary: `[I]` implemented and exercised,
`[N]` not implemented and fail-closed, `[H]` historical context only, `[M]`
measured from the named bound artifact, and `[X]` contradicted or retired.
An unbound artifact or a command timeout cannot be promoted to `[I]` or `[M]`.

Before quoting any result, compare the artifact's recorded commit/tree with the
live worktree and rerun the owning command when they differ. Targeted chains,
static scans, documentation builds, and wheel-import checks remain bounded
evidence and must not be expanded into full-regression, performance, or release
claims.

## 6. Parameter System

### 6.1 Canonical Manager Access Surface

File: `relic\parameters\manager.py`

`UnifiedParameterManager` has two canonical retrieval methods:

| Purpose | Public method | Return contract |
| --- | --- | --- |
| Runtime value lookup | `manager.get(key, ...)` | Resolved scalar, tensor, mapping, or caller-supplied default |
| Metadata lookup | `manager.get_parameter(key, namespace=None)` | `ParameterSpec` or `None` |

`get_parameter(...)` preserves the same key normalization and optional
namespace composition used by the internal specification lookup. The helper
`relic\parameters\ops\spec_access.py::get_spec(manager, key)` is an internal
implementation detail only; it is not a public manager method.

A value read hands out an object the caller owns. `manager.get(...)` on every
path (direct storage read, query-result cache, unit conversion, namespace
tree), the values of `get_namespace_parameters(...)`,
`FunctorBase.get_parameter_value(...)` (through `manager.get`) and
`get_namespace_dict(...)` pass
`relic\parameters\ops\type_utils.py::caller_owned_value`, the one copy rule:
a `dict`, `list`, `set` or `tuple` is copied at every depth with its container
type, a Tensor (a learnable `nn.Parameter` binding included, Section 4.2)
keeps its identity at any depth, and any other object is returned as it is.
The query-result cache keeps the stored object, never a copy a caller holds.
Writes follow the same rule: `set_value` stores a copy of a written
container, `load_yaml_file` and `load_omegaconf_config` cache the loaded
mapping and return a copy, and the change history and every change callback
receive their own copies of the old and new value. A stored value therefore
changes only through the mutation owners (`set_value`, `set_parameter`,
`register_parameter`, snapshot restore). `get_parameter(key)` and
`get_namespace_parameters(..., return_specs=True)` return the specification
objects the manager holds, without a copy: they are the metadata lookup, the
`value` field is the stored value, and callers do not change them. A
specification passed to `register_parameter` or `set_parameter` is held as
given and becomes the manager's object.

Public compatibility aliases `get_spec(...)`, `get_parameter_spec(...)`,
`clear()`, and `load_database(...)` are removed. Tests and infrastructure use
`get_parameter(...)`, `clear_parameters()`, and `load_from_database(...)`
directly. Do not restore transition aliases or parallel metadata accessors.

This is a read-access contract, not a claim that the manager has only two public
methods. Lifecycle and mutation operations such as
`register_parameter(spec)`, `set_value(...)`, `list(...)`,
`clear_parameters()`, and
`load_from_database(...)` remain separate public responsibilities.
`set_value(key, value)` changes a registered parameter only; an unknown key
raises `KeyError` (register it with `register_parameter()` first).
`set_parameter(spec, tier)` replaces one specification, its stored value and its trainable binding and does not reload the database. `set_parameter` and `register_parameter` (including every re-registration by a database reload) follow one binding rule (`bind_learnable_value`): an existing binding of the same shape and dtype keeps its `nn.Parameter` object, gradient flag, and device and takes the new value in place, so the Functors holding it by reference (Section 4.2) and the optimizer keep one theta; a changed shape or dtype creates a new binding, and a specification that is not learnable has none. `ParameterStorage.restore(snapshot)` rolls storage
back and re-raises when a restore callback fails; the manager callback rejects
stored values without a specification (`KeyError`). `ParameterCalibrator`
metrics are `None` when the objective exposes no `model_runner`; observations
without variance (R2, NSE) or with zero sum (percent bias) raise `ValueError`
instead of returning sentinel values. Parameter listing (`manager.list`,
`get_namespace_parameters`, audit `namespace_filter`) follows the
`ParameterStorage.list` rule for every source: a namespace selects the keys
below `<namespace>.` (not the namespace key itself, not a sibling prefix) and a
pattern is an fnmatch pattern. Namespace membership has one rule,
`relic\parameters\ops\namespaces.py::namespace_keys`: the keys below
`<namespace>.` that the manager holds or the database defines, with the
database enumerated in full, so the result does not depend on which keys were
loaded on demand before. `get_namespace_parameters` and the nested tree that
`manager.get("<namespace>")` returns for a key that names no parameter (an
optional read without a default) both use it. The tree holds copies of the
stored values (the copy rule above), is built at each read and never enters
the query-result cache. A
database key names a parameter or a namespace, never both: no key is the dot
prefix of another key, so every key has one place in the tree
(`test\unit\parameters\test_yaml_duplicate_keys_l1.py`). The tree builder
(`relic\parameters\ops\accessors.py::_build_namespace_tree`) applies the same
rule to the members of each read, whatever their source: when one member is
the dot prefix of another, `manager.get("<namespace>")` raises `ValueError`
naming both keys before any value is read, and no stored value changes.
Registration does not apply it (keys also enter the manager on demand and by
preload without registration, and a source added later can bring the other
key), and single-key reads and `get_namespace_parameters` of such keys are
unaffected. The
unit-conversion cache key is (parameter key, source units, target units,
value).
`delete(key)` removes an in-memory specification, storage entry and trainable
binding. With caching enabled, it clears both query-result and unit-conversion
caches, including namespace- and unit-dependent queries. A missing key raises
`KeyError`; deletion does not remove a persistent YAML definition.

Runtime context determines the value-access entry point:

- Scientific `FunctorBase` implementations use only
  `self.get_parameter_value(..., required=True/False)`.
  Config values with an explicit `units` field are converted from those source
  units to the canonical parameter units on each config read, through the
  existing parameter-unit owner. The original Tensor and unit descriptor remain
  unchanged; repeated reads build fresh conversion graphs and observe updated
  source values. Requested output-unit conversion and bounds checks then use
  the canonical metadata. Unit conversion remains active when optional value
  validation is disabled. A manager-only request does not consult config units.
  Its ordered source resolution (`source_order`, only `"manager"` and
  `"config"`; any other entry raises `ValueError`) advances past a manager only
  when the optional lookup returns `None` (or no manager is available). There is
  no caller-supplied default, no local or `"default"` source, and no automatic
  registration of a missing key: a missing optional parameter is `None`, and
  `set_parameter_value` writes the instance configuration, the one
  instance-level source. Manager validation, unit conversion, and backend errors
  propagate; they do not select a different source. An explicit config-first
  order still resolves the configuration before reading a manager value.
  `manager.get(key, units=...)` raises `ValueError` ("Unit conversion failed for
  parameter ...") for a conversion the authoritative converter cannot perform,
  at every `ValidationLevel` including `NONE` (the read path has no `validate`
  switch); an
  unconverted value is never returned or cached. Parameter change callbacks run
  after the change is committed: every callback is notified, then the failures
  are raised together as an `ExceptionGroup` naming the key; the change stays
  committed.
- `FunctorBase._get_state_variable(state, key, units=None,
  validate_range=(lo, hi), dtype=None)` reads the state only and returns the
  value unchanged; any element outside the declared domain, NaN included,
  raises `ValueError` and is never clipped. A missing key raises `KeyError`
  (NOT FOUND): there is no default tier and no `default_param_name`. The
  governed call (`FunctorBase.__call__` calling `validate_state`) requires every
  declared non-optional input; readers pass the physical domain of the
  variable (its contract bounds) as `validate_range`, and an optional port
  such as a source term uses its declared closed case (for example S = 0),
  not a parameter.
- Non-Functor helpers constructed by an existing lifecycle owner receive a
  required `UnifiedParameterManager` explicitly and call `manager.get(...)`.
  Their constructors must not retain an optional-`None` global fallback.
- Top-level infrastructure, scripts, coupling utilities, and configuration
  loaders may use `get_global_manager().get(...)` only when no explicit owner
  injection point exists.
- Directory location does not determine context. A helper under
  `relic\scientific\` is not a Functor unless it participates in the Functor
  lifecycle.

`get_global_manager()` (`relic\parameters\ops\manager_registry.py`) is the
only process-wide entry point. It creates the manager once under a reentrant
lock, runs `initialize()`, and then refreshes the process-wide variable-name
mapper from that manager's catalogs; if either step fails, the error
propagates and no global manager is kept. Constructing
`UnifiedParameterManager(...)` always builds an independent instance, which
is used only through explicit injection.

A preload requested after first use (`get_global_manager(preload_all=True)`
on an existing manager, as issued when the solver registry is first imported, or
`_preload_parameters()`) completes the manager
(`relic\parameters\ops\bootstrap.py::preload_parameters`): it adopts the
database specification of every key the manager does not hold yet. A key the
manager already holds (loaded on demand, registered, or written with
`set_value`) keeps its specification, stored value and live binding, so a
value written before the first full load is what every later read returns.

`initialize()` loads nothing
(`relic\parameters\ops\lifecycle.py::do_initialize` prepares the database
location only). A manager constructed with `preload_all=True` holds the whole
database when its constructor returns, so a value written or a learnable
binding moved between construction and `initialize()` is kept, as it is for a
manager that loads on demand. The database switch
(`configure({"database_path": new_path})`,
`bootstrap.py::load_initial_parameters`) is the lifecycle step that replaces
the held specifications, stored values and binding values with those of the
new database. Every specification it adopts is unit-checked under
`unit_validation_mode` first, whatever was checked under the previous
database (check-only records an issue, strict raises before anything is
adopted). A switch that fails restores `database_path` and `database`, so the
manager stays on the database that defines the specifications it holds and a
key loaded on demand later (also after `reset()` and `initialize()`) comes
from there, never from the rejected database. The switch clears the
query-result and unit-conversion caches as a whole, also when the load fails
part way, so no query result cached from the previous database is served:
`get(key)`, `get(key, units=...)`, `get_parameter(key).value` and
`storage.get(key)` agree on the value the new database defines
(`test\parameters\parameter_manager_lazy_test.py`).

Do not reintroduce `_require_parameter(...)`, `get_parameter_by_path(...)`,
`require_parameter_value(...)`, `resolve_parameter_manager(...)`,
`require_parameter_manager(...)`, `ParameterAccessorMixin`, or any parallel
parameter proxy.

`ParameterSpec` has no default for `units`, `dtype`, `bounds`, or `category`, and the
`bounds` field must be present. A parameter with a numeric `dtype` declares a
`(lower, upper)` interval and writes an unbounded side as `-inf` or `inf` (YAML
`-.inf` / `.inf`), never `None`; `None` states that no interval applies and is
accepted only for a non-numeric `dtype` (text, boolean switch, enumeration, list,
or mapping). `relic.parameters.spec.parameter_bounds_error` is the single rule read
by `ParameterSpec`, `has_complete_parameter_metadata`, the structural parameter
audit, and the parameter YAML schema.

A list-valued parameter (`dtype` `list` or `tuple`) may declare one
`[minimum, maximum]` pair of numbers per component in place of a single interval,
for example `[[223.15, 273.15], [323.15, 373.15]]` for a two-component
`[low, high]` value (the `bounds` field holds two entries, so the declaration
describes two-component values). `relic.parameters.spec.componentwise_bounds_errors`
is the single rule for such a declaration, read by `ParameterSpec.validate_value`
(validated `set_value`, database loading, `validate_parameters`) and by
`FunctorBase._validate_parameter_value` (every validated `get_parameter_value`
read, manager value or configuration candidate): the value is a list or tuple with
one finite real component per pair, each inside its closed pair. A violation
raises `ValueError` from `set_value` and `get_parameter_value` and is reported by
database loading (raised under `fail_on_validation_errors`); no component is
clamped. A flat `(lower, upper)` interval or `None` on a list-valued parameter is
not applied to its components.

A numeric parameter states its bounds endpoints, value and default as numbers: `parameter_bounds_error` rejects a text endpoint of a numeric dtype and `ParameterSpec` rejects a text value or default of a numeric dtype; neither is converted, and `ParameterSpec.from_dict` does not turn numeric strings into numbers. PyYAML follows YAML 1.1, so a database entry writes `1.0e+7` / `1.0e-6` (`1.0e7` and `1e-6` load as strings).

`ParameterSpec.to_dict()` and `from_dict()` own parameter field projection,
including `default` and `allowed_values`; the public parameter serialization
helpers delegate to them. Tensor-valued current/default fields use the existing
Tensor wire format for every shape, including scalar and single-element values.
The decoder requires an explicit canonical dtype, shape, and boolean gradient
flag. It preserves empty trailing dimensions while rejecting dimensions that
contradict the nested data. A restored categorical spec retains its validation
constraints. This value projection does not prove a complete provider binding:
live optimized values still come through `manager.get(...)`, and implementation,
request-source, lifecycle, and committed-view contracts retain their own owners.
Category normalization belongs only to `parameters\category_aliases.py`:
canonical names and declared aliases accept normalized case/whitespace, while
unknown or non-text categories raise an error, and an omitted category is
an error as well.

ParameterManager snapshots keep each owner of parameter state apart and exactly
as it is, without synchronizing or replacing live storage/spec objects during
capture: the specifications with their declared values, the stored values (a
specification whose value is not stored yet stays without one), and a copy of
every live `nn_parameters` value under `nn_values`, recorded through the
existing storage snapshot owner. Full snapshots also record actual gradient
enablement. Restore validates the whole snapshot first (stored values agree
with their specifications, gradient flags only for learnable specifications,
one live value per flag with its specification's shape and dtype, compatible
live bindings), then restores the stored values, the exact specifications and
the live values into the existing objects, so every value keeps its type and
dtype; a learnable float stays a float. Cross-device copies retain each
existing binding and its device; bindings created after the snapshot are
removed; snapshot gradient flags are restored explicitly. Existing optimizer
references are preserved. Optimizer momentum and other optimizer-owned state
are outside this parameter snapshot contract. Snapshot diffs and experiment
diffs compare resolved values (the live learnable value over the stored or
declared value), decode the same Tensor wire, and report numeric changes
separately from dtype/shape/gradient contract changes.

### 6.2 Parameter Reference Specifications

The `references` field for all parameters must not be an empty list. Newly added parameters must include:

- At least one authoritative citation (DOI or complete literature information).
- Citation format examples: `"doi:10.1104/pp.125.4.1938"` or `"Bernacchi et al. (2001) Plant Physiol 125:1938-1948"`.

### 6.3 Parameter Cleanup Specifications When Modifying Scientific Processes (Mandatory)

When modifying, replacing, or deleting a scientific process (Functor), **you must synchronously clean up redundant parameters in the parameter database**:

1. **Deleting a process**: Delete all parameter entries corresponding to this process, or explicitly mark them as `deprecated`.
2. **Replacing a process**: Replace old parameters with the new process's parameters, and delete parameters unique to the old process.
3. **Merging processes**: Eliminate duplicate parameters after merging, retaining entries with higher precision or more authoritative citations.
4. **Refactoring a process**: The namespace of the parameter keys must be updated synchronously; orphan parameters under the old namespace must not be retained.

Operating procedure:

```text
Before modifying a scientific process -> List all parameter keys associated with this process
After modifying a scientific process -> Check which parameters are no longer referenced by any code
After confirming orphan parameters -> Delete from YAML, and explain the reason for deletion
Run parameter tests -> pytest test\unit\parameters\ -v
```

Forbidden behaviors:

- Retaining unused orphan parameters after deleting or replacing a scientific process.
- Modifying a parameter namespace without synchronously updating the access path in the code.
- Adding a new parameter without checking if a parameter with the same meaning already exists.

### 6.4 Physical Constants, Units, and Time SSOT

- Physical constants are defined only in
  `relic\parameters\database\relic\physical\constants.yaml` and are read through
  `relic\utils\physical_constants.py::get_physical_constant(...)`.
- `relic\utils\time_constants.py` derives its exported conversions from the same
  parameter database. Scientific modules must not hardcode conversion values.
- Unit declarations use the existing unit/parameter validation path; do not
  create a domain-local unit registry or repeat constants in domain YAML.
- Extending a constant means adding a complete YAML entry and using the existing
  generic accessor. Add Python code only when a genuinely new access behavior is
  required; a per-constant mapping or wrapper is not required.

## 7. Scientific Process Layer (`scientific`) Specifications

Directory: `relic\scientific\`

Sphere modules:

- `relic\scientific\atmosphere\`
- `relic\scientific\biosphere\`
- `relic\scientific\hydrosphere\`
- `relic\scientific\pedosphere\`
- `relic\scientific\anthroposphere\`
- `relic\scientific\isotopes\`
- `relic\scientific\transport\`

Registration mechanism:

- File: `relic\scientific\__init__.py`
- `FUNCTOR_REGISTRY`: maps each registered name to its `"module:Class"` path. It is filled once, when
  `relic.scientific` is imported, so name resolution does not depend on
  process history.
- `register_functor(cls)`: a plain function; no module applies it as a
  decorator. It binds the class name (`FUNCTOR_NAME`, else `metadata.name`,
  else `__name__`), its lowercase form, and the snake_case class name. One
  name binds one class: binding a name to a different class raises
  `ValueError`, and registering the same class again changes nothing.
  Extensions add classes only through this call (`relic\extensions\loader.py`
  with `populate_core_registry=True`).
- `register_all_functors()`: the registry bootstrap, called once by the last
  statement of the module; it walks every `relic.scientific` subpackage and
  registers each `FunctorBase` subclass. Calling it again changes nothing.
  Command-line entry modules (`*_cli.py`, `is_cli_entry_module()`) are not
  imported: they define no Functor and run only through `python -m`.
- `get_functor_path(name)`: tries the name as given, lowercased, and in
  snake_case; a miss raises `KeyError` and triggers no scan.

Stable-isotope runtime is currently `[N] runtime_unbound`.
`wire_isotope_tracker(...)` is the sole public assembly request boundary and
rejects before reading state, constructing runtime objects, or mutating the
simulator. `IsotopeHeartbeatFunctor` is a fail-closed
`ScaleCouplingFunctor`: both construction and `forward(...)` reject until
governed component UpdateItems, complete route coverage, parent-process time
binding, and restart contracts exist. It does not invoke
`IsotopeTrackerService`; the former heartbeat/service/NetCDF runtime path has
been removed. `IsotopeTrackerService` is itself a fail-closed candidate
boundary and owns no prognostic pools or event ledger. No isotope-specific
StateManager callback exists.

`MultiIsotopeTracker`, species configurations, and fractionation utilities are
internal candidate components; they must not self-register, appear as an
executable package runtime, or be cited as Phase 8 implementation evidence.
The canonical configuration root remains
`relic.scientific.isotopes.config.IsotopeConfig`, whose runtime enablement
defaults to false. Configuration presence does not upgrade the `[N]` runtime
status.

For photosynthesis and hybrid MEGAN pathways, `MEGANModel.forward(...)` is the
sole photosynthesis-carbon projection owner. It resolves the carbon floor
through ParameterManager once per forward call, shares the projected carbon
across compound calculations, and emits the explicit floor residual. The
standard pathway neither requires `net_photosynthesis` nor declares or emits
that residual.

The atmospheric `ChemistryFunctor` owns its NASA7 standard properties, ideal-gas
mixture potentials, and internal-energy temperature inversion. These methods
reuse ParameterManager, physical constants, and the registered scalar solver;
they do not introduce a thermodynamics service or advance state. The inverse
captures one parameter projection and explicitly passes coefficient scalars to
the existing implicit-gradient solver. Units, continuous-fit domain restrictions,
and the thermodynamic derivative contract are specified in `theory/chemistry.md`.
Their availability does not establish strict reaction execution or certification.

### 7.1 Scientific Process Code Comment Template (Mandatory)

Any computational code involving ecological, transport, conservation, or physical-chemical processes must precede the computation block with exactly the following six English elements. This is the only accepted scientific comment structure:

```python
# References:
#   - Farquhar et al. (1980), Planta.
# Mathematical expression:
#   A = min(Wc, Wj, Wp) - Rd
# Parameter meanings:
#   vcmax25: maximum carboxylation rate at 25C [umol CO2 m-2 s-1]
#   jmax25: maximum electron transport rate at 25C [umol e- m-2 s-1]
# Units:
#   A, Wc, Wj, Wp, and Rd [umol CO2 m-2 s-1]; jmax25 [umol e- m-2 s-1]
# Domain/constraints:
#   temperature, CO2, and radiation inputs are finite and within declared bounds.
# Sign convention:
#   positive A is net carbon uptake by the canopy.
```

The former three-element abbreviation is not accepted. Units embedded in
`Parameter meanings` do not replace the standalone `Units` element, and every
calculation must state its enforced domain and flux/source/sink sign convention.

### 7.1.1 Process Contract Inventory (G00 working view)

`docs\generate_process_contract_inventory.py` renders
`docs\source\theory\process_contract_inventory.md` from the IPHS certificates and
the live functor inventory scan (`test\audit\collect_functor_inventory.py`). For
every scientific process it lists the computation form actually used (empirical
rate, pool-transfer ODE, storage-discharge ODE, spatial PDE discretization,
algebraic diagnostic, managed boundary exchange, conservative interface mapping,
SDE, marked jump, memory kernel, ALE, index-1 DAE, hybrid topology event, or a
claimed strict IPHS runtime point), the applicability of the additional G04-G12
contracts, the first primary source, the theory page status, the parameter and
acceptance-test premises, the runtime evidence status, and the remaining gap with
its owning task card. The page is derived: certificates and the inventory scanner
stay the owners, the classifier rules are the whole logic and live in the
generator, and `--check` plus `test\audit\process_contract_inventory.test.py`
keep it current. A process without a thermodynamic claim is complete when its
own mature contract holds; missing IPHS objects are not defects for it.

### 7.1.2 Unit Consistency Gate (G01)

Every certificate `unit_map` entry is a parseable unit string (never prose) and
has the same dimensionality as the SCM contract and the SSOT catalog entry of the
same variable; spelling may differ (`m^-2` versus `m-2`, a species qualifier such
as `kg C`) but the dimension may not. An output that carries no physical quantity
(a status token or a report dictionary) is written as `non-quantity`, which is
accepted only for variables without a contract. Placeholder contract units
(`dimensionless` for a physical quantity) are contract defects: on 2026-09-14 the
hydrology stubs `rain_flux`, `snow_melt_flux`, `snowfall` (kg m-2 s-1),
`snow_depth` (m), `snow_surface_age` (day), `snow_water_equivalent` (mm),
`hydraulic_conductivity` (m s-1; its contract was deleted on 2026-09-24
because no Functor reads or writes it as state; it survives only as the nested
Richards diagnostic key `diagnostics.soil_moisture_richards.hydraulic_conductivity`,
which is also the Richards certificate `unit_map` key) and the atmospheric
transport diagnostics
`diagnostics.eddy_diffusivity` (m2 s-1), `diagnostics.advection_rate` and
`diagnostics.diffusion_rate` (kg m-3 s-1) were corrected from the process code.
The gate is `test\audit\certificate_unit_consistency.test.py`; rate-versus-total
and `dt` semantics stay with the state-advancement and direct-mutation audits and
each process's own tests.

### 7.2 Theory Documentation First

Before adding a new scientific process, establish the theoretical documentation in `docs\source\theory\` first, then implement the code.
The theoretical document must contain at least:

- References
- Core mathematical expressions
- Parameter meanings
- Units
- Domain and applicability constraints
- Sign conventions for fluxes, gradients, sources, and sinks

### 7.3 Mature Scientific Processes and Conditional IPHS Canonicalization

Each executable scientific process has one documented production law
integrated into the existing RELIC lifecycle. Mature empirical, conventional
numerical, and existing canonical operator formulations may be used when their
assumptions, interfaces, budgets, and numerical outputs are valid. The
equations below apply only to the corresponding dPH or GENERIC claim:

```text
dissipative_ph:
  x_dot = (J_h - R_h) e_H + G_h u_h + N_h r_h
  J_h.T = -J_h
  R_h = R_h.T >= 0
  dH/dt = -e_H.T R_h e_H + y_H.T u_h + e_H.T N_h r_h

generic:
  x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h
  J_h.T = -J_h
  M_h = M_h.T >= 0
  J_h e_S = 0
  M_h e_E = 0
  dE/dt = y_E.T u_h + e_E.T N_h r_h
  dS/dt = e_S.T M_h e_S + y_S.T u_h + e_S.T N_h r_h
```

A GENERIC Poisson claim requires Jacobi-identity evidence over its declared
state domain. `zero_field` establishes it for `J(x)=0` and `constant_field` for
a constant `J`; `state_field` provides finite-point residual evidence, not a
continuum proof: skew symmetry and the Jacobi residual are checked at its
declared `poisson_domain` samples and, at a Level 5 runtime point, at the
evaluated point (the bound state, or the first declared sample when no state
is bound). `matrix_snapshot` does not prove global constancy or state
derivatives.
The field contract and process-acceptance boundary are specified above.
The shared reaction conditions for an independent strict internal ledger are:

```text
C_h N_h = 0
generic: e_E.T N_h r_h = 0 and e_S.T N_h r_h >= 0
dissipative_ph total energy: e_H.T N_h r_h = 0
dissipative_ph free energy: e_H.T N_h r_h <= 0
```

The equations above use the independent-reaction representation. An absorbed
representation instead requires the entire irreversible term to equal the
reaction rate, `N_h r_h=-R_h e_H` for dissipative pH or `N_h r_h=M_h e_S` for
GENERIC, and removes the independent `+N_h r_h` term.
`[I]` The existing spec and assembler implement the mutually exclusive
`reaction_representation="independent"|"absorbed"` policies. Absorption retains
the real `N` and runtime `r`, checks their product against the entire
irreversible rate on every evaluation, and omits the independent addition.
The componentwise comparison allows only dtype/reduction roundoff; structural
degeneracy tolerances cannot alter the production law. Contribution policies
must agree with the assembly. Mixed absorbed/independent reactions or partial
irreversible decompositions need an explicit decomposition and are not inferred.
This pointwise representation check does not establish physical E/S closure
or a governed StateManager advance. `[I]` The closed constant-volume NO+O3
selection binds actual E/S, absorbed nonzero reaction drift, one governed state
advance and restart evidence; the closed one-dimensional Fickian selection
binds Level 4 storage and its finite-volume drift. These constructor-scoped
runtime-point certificates do not promote the ordinary main process
certificates. `[N]` General unbound processes, dPH reaction processes and
strict points with declared ports remain unavailable. Matrix-only audits and
zero-field Jacobi pre-gates alone do not establish process acceptance.

Levels 1-3 may expose a declared subexpression such as `N_h r_h`, a transport
operator, or typed ports without claiming either strict closure. `R_h` acts on
the energy effort only; GENERIC `M_h` acts on the entropy effort only. A
dissipative feedback `u_int=-D y`, `y=G^T e_H`, contributes `G D G^T` to
`R_h` and therefore enters the drift with a minus sign.

Runtime replacement rules:

- Normal scheduling calls the Functor object through `FunctorBase.__call__()`;
  `forward(state, time, dt, **kwargs)` implements its scientific kernel.
- The production path executes the single documented process law. A mature
  empirical closure may be that law when its scientific and architectural
  contracts are satisfied. Existing correct Nr, DEC, and IPHS implementations
  remain reusable; this scope change does not justify a second formula.
- Migrated modules must not expose a legacy mode, fallback formula, duplicate executable process law, or hidden branch.
- For a process declaring the independent IPHS reaction representation, the
  production calculation path must explicitly evaluate
  declared `N_h` and `r_h` as `N_h r_h`. An absorbed representation must prove
  `N_h r_h=-R_h e_H` or `N_h r_h=M_h e_S` (whole irreversible term) and must not execute a second
  reaction term. DEC cochain declarations, Hodge/mobility terms, matter ports,
  `J_h`/`M_h` metadata, entropy audits, or a canonical-runtime flag are not
  sufficient by themselves.

Required per-module closure:

1. The module still satisfies one of the four abstract contracts and the existing bridge class.
2. All parameters resolve through `self.get_parameter_value(...)` in Functor code and YAML metadata in the ParameterManager database.
3. All state writes go through the existing StateManager path and, when
   contracts are enabled, its ContractRegistry enforcement.
4. All composition-facing declarations use existing FunctorMetadata, NodeRegistry, PortSpec, SSOTCatalog, ValidationRegistry, and simulator scheduling/device policy.
5. The matching `docs\source\theory` document states the equation actually
   executed, references, applicability, parameters, units, support, time
   semantics, state updates, diagnostics, and independent numerical oracles.
6. A claimed IPHS equivalence or strict capability has a validated certificate
   from the existing certificate owner. No inapplicable H/E/S, effort,
   affinity, or certificate placeholder is required for an ordinary model.
7. Ports, budgets, reaction networks, residual routing, and staged/lagged
   coupling are verified where actually used, through the existing rules and
   executable process tests. Verify real output values and physical balances,
   not only schema acceptance or successful execution.
8. Phase 8 strict GraphIR payloads provide `GraphIR.assertion` or `graph.metadata.structural_assertions`; missing structural assertions are validation errors.

Failure conditions:

- A migrated module changes input or output keys, tensor shapes, units, dtype, device placement, dependency semantics, or write mode without an explicit adapter and tests.
- A migrated module leaves unconsumed flux, clipping residual, overflow, or open-system exchange untracked.
- A migrated module uses direct Functor calls instead of declared dependency, port, chain-map, or IntegrationPlan metadata.
- A module has code, theory, parameter YAML, or an applicable certificate that
  disagrees with its actual scientific contract.
- The same physical transfer is advanced by two independent production laws,
  or the executed law disagrees with its declared representation, parameter
  binding, physical budget, or documented assumptions.

## 8. Contracts, Data, Coupling, Computation

### 8.1 Contracts

- `relic\contracts\contract_loader.py`
- `relic\contracts\schemas\scm_variable_schema.yaml`
- `relic\contracts\variables\` (variable contracts organized by sphere)

### 8.2 Data

Directory: `relic\data\`

Key sub-layers:

- `io\`
- `standards\`
- `quality\`
- `streaming\`
- `transformers\`

`DataWriter` is the output owner for `WriteResult` and local file delivery.
NetCDF and Zarr accept `delivery_key`: a repeated key must match complete
decoded data, attributes, dtype, CF storage declarations, and effective write
options before the existing artifact receipt is returned. A new key may
replace the previous output. File SHA256 uses actual bytes; Zarr SHA256 binds
sorted relative paths, lengths, and bytes, with total size summed across files.
The native `relic_delivery_*` attributes belong to this owner. No manifest or
second writer implements this policy.

`DataWriter.prepare_dataset(...)` runs the existing processing and encoding
chain once, fixes chunking before materialization, and copies an independent
xarray descriptor with its effective configuration and preparation metadata.
It creates no output files or directories. Numeric arrays use Tensor values
for checkpoint encoding; Zarr codecs use their standard configuration
dictionaries. Other native object types require explicit serialization support
before they can be persisted.

`write_dataset(None, path, snapshot=candidate)` consumes those prepared values
through the same publication and receipt path. It rejects raw data, path or
configuration overrides and conflicting bound keys. An unbound candidate may
be copied and bound to an explicit request key without repeating processing;
the original candidate stays unchanged. Its UTC timestamp denotes preparation,
not retry time. Snapshot replay never adopts an existing artifact's timestamp
to change the candidate; ordinary raw writes retain the same-key timestamp
lookup needed for their existing value-equivalent replay contract.

Writes validate a staging artifact before publication. NetCDF uses file
replacement; replacing a Windows Zarr directory uses rename and compensation
and does not promise an uninterrupted atomic directory switch. Cancellation
cleans the staging artifact. A post-publication receipt failure can be retried
without writing the same content again. These contracts describe serialized
local delivery: at-least-once redelivery with an idempotency key and receipt per
obligation gives effectively-once semantics for idempotent consumers; distributed
exactly-once delivery is not implied.

Tensor export retains explicit variable dimensions, coordinates, attributes,
encoding, and precision. Nonscalar tensors require a template variable;
coordinate lengths alone do not establish its axes. `DataIO.write_tensors`
accepts per-variable `dims`, ordinary attributes, and optional xarray
`encoding`; coordinate metadata uses the coordinate name and dataset metadata
uses `global`. It binds these to the existing DataWriter template. Format
conversion and dataset processing retain Dataset metadata instead of invoking
the reader's optional tensor-only projection.

Diagnostic budget values retain the StateManager/UnitConverter canonical
basis: C/N/P use grams per area with a separate element attribute; water uses
kilograms per area. Metadata may not relabel those values into another basis.
Default diagnostic metadata does not add CF masking ranges: large residuals
and anomalous factors remain observable. A generic tolerance shared across
gram and kilogram ledgers has no invented common physical unit. Scalar
diagnostics retain their original key across time records, and output-name or
metadata collisions fail before publication. The simulator captures the fully
prepared diagnostic candidate before accepting the scientific step; scalar and
cross-record identity errors therefore remain rollbackable. Delivery does not
repeat summary conversion or retain a second copy of the original summaries.

### 8.3 Coupling

Directory: `relic\coupling\`

Key sub-layers:

- `configs\`
- `contracts\`
- `core\`
- `external\`
- `internal\`
- `observation\`
- `plugin\`
- `protocol\`
- `scales\`
- `tools\`

Coupling truth boundaries:

- An emitted interface rate is not a donor deduction. `SoilVegetationCoupling`
  writes `root_uptake_profile` (kg m-3 s-1, `REPLACE`) and never advances soil
  moisture; its variable contract declares no consumer, so root uptake is
  currently an exposed port, not a soil-water sink. A budget closes only when a
  registered consumer integrates the port exactly once.
- `port_matching_check`, `port_power_units_check`, and `chain_map_check` are
  declaration or sample checks. A closure claim names the consuming Functor,
  the single StateManager commit, the unit/support/time basis, and the actual
  residual; a `diagnostic_residual` route discloses a residual and never closes
  a budget.
- `external\common\base_adapter.py::ExternalAdapter` is a template whose
  `forward` raises `NotImplementedError`; the concrete WRF, CESM, CMIP6, VIC,
  and SWAT adapters read model-output datasets. Their
  `forward(state, time, dt, *, calendar_type, time_origin)` returns exactly the
  provided variables at the simulation time. The provided variables are the
  configuration's `provided_variables` (each a mapping target of the source) or,
  when it is None, every mapping target of the source; they are the adapter's
  declared `output_variables` (no inputs, `variable_kind` FORCING), each written
  in the units of its variable contract, whose `writers.tendencies` lists the
  adapters, and the adapter declares `ConflictResolutionPolicy.EXCLUSIVE`, so
  registration refuses any second writer of a provided variable in either
  registration order. The read goes through the one read path (`read` calling
  `_read_field`: schema mapping, declared source transform such as WRF
  de-accumulation, layered soil water, the WRF `Q2` water vapour mixing ratio w
  converted to specific humidity q = w/(1+w), or the WRF grid-relative
  `U10`/`V10` rotated to earth-relative components with the file's
  `COSALPHA`/`SINALPHA`, u_e = u cos(a) - v sin(a), v_e = v cos(a) + u sin(a),
  a missing rotation field raising; catalog unit conversion), then
  `_read_records_at`: the record times are decoded (cftime/xarray; numeric CF
  time coordinates with their units and calendar), the dataset calendar must be
  the simulator `calendar_type` (CF synonyms such as `365_day` = `noleap`; no
  conversion), and the value at `time_origin + time` follows the configuration's
  `time_interpolation` (`lower`: record k for t_k <= t < t_(k+1); `linear`:
  f1 = (t_(k+1) - t)/(t_(k+1) - t_k) on the two records around t; CDEPS/E3SM
  `shr_tInterp_getFactors`, `forcing_record_weights`); a time outside the
  records, a missing `time_origin`, `calendar_type` or `time_interpolation`, and
  a time-dependent field without a decoded time coordinate raise; the read keeps
  no position, so it is a function of t alone; the units a
  file declares are authoritative, and an adapter table only fills units a
  file does not declare (`units_source`). External fields are mapped by
  meaning: daily extrema, lowest-model-level fields, and energy versus water
  fluxes are never aliased to another quantity, and a many-to-one mapping
  needs a declared canonical source. Catalog identity for name mapping is
  `(standard_name, cell_methods)`, owned by
  `relic\data\standards\cf_conventions.py` and shared by `SSOTCatalog` and the
  variable name mapper; one CF identity belongs to one variable, so the
  external radiation, latent heat, and evapotranspiration fields map to the
  contract names `shortwave_down`, `longwave_down`, `latent_heat_flux`, and
  `evapotranspiration_flux`. The 10 m near-surface wind (`eastward_wind`,
  `northward_wind`; WRF `U10`/`V10`, CMIP6 `uas`/`vas`) and the transport-grid
  advection wind (`wind_u`, `wind_v`) are different variables with different
  vertical support. An external catalog entry declares `standard_name`,
  `long_name`, `units`, and `category` (a missing field raises `KeyError`, an
  unknown category `ValueError`; no entry is skipped and no category is
  assumed). A variable's valid range has one referenced declaration, and every
  other catalog that lists the variable repeats it: the evapotranspiration
  flux is signed (CF `water_evapotranspiration_flux`, upward positive, dew
  negative) and the latent heat flux range is the latent heat of vaporization
  times that range. An adapter's metadata expression is
  `ExternalAdapter.mapping_expression(source)`: the time-indexed read, the
  mapping pairs, and the formula of each source transform the source actually
  declares. Ingestion regridding runs through NumPy and carries no
  gradient. Neither implies live two-way coupling with a running external
  model or certification against an external coupling standard.

Scale labels and physical support are separate contracts. `SpatialScale`,
`TemporalScale`, and `VariableKind` (`relic\core\functor\functor_types.py`) are
Functor-level labels: they drive the label-level scale check, execution-group
classification, and STATE output-overlap governance. They do not encode
geometry, cell areas, time bounds, or quantity representation. Quantity
representation is declared to the scale operator that uses it:
`spatial_basis` (`SpatialQuantityBasis` in
`relic\coupling\scales\types.py`: `intensive`, `cell_total`, `area_density`,
`categorical`) and `temporal_semantics` (interval_mean, rate,
interval_accumulation, state_at_boundary), together with explicit units.
`SpatialQuantityBasis` is the only quantity-basis enumeration: the temporal
operators use it to recognise categorical variables, and the former
`VariableType` enumerations of the temporal, hierarchical, and external
layers were removed on 2026-09-23. `resolve_spatial_basis(name, metadata)`
reads the declaration from the variable catalog or the caller's metadata and
raises when it is missing or inconsistent; the ambiguous word `extensive` is
rejected, because it could mean a cell total or an area density.
`convert_spatial_basis` converts between `cell_total` and `area_density` with
explicit cell areas. No spatial operator infers the basis from a variable
name or unit.

Automatic spatial regridding (`relic\core\simulator\scale_ops.py`,
`regridding_ops.py`) only runs when `enable_auto_scale` is on and the label
check reports more than one known label among the outputs of the group being
executed. It requires explicit `source_grid` and `target_grid` descriptors;
tensor shape and scale labels are never used to reconstruct geometry, and a
missing descriptor raises `ValueError`. The target descriptor must also carry
the CF grid fields `grid_mapping`, `lat_name`, and `lon_name`, which are passed
to the CF synchronizer; a descriptor without them is rejected when the
ScaleManager lifecycle runs. The method follows the catalogued
`spatial_basis`: `categorical` uses nearest neighbour, all other bases use the
conservative method; a variable without a catalog entry raises `KeyError`.
The regridded tensors replace entries in a copy of the state that is used as
the group's working input; the transform itself commits nothing to
StateManager. The ScaleManager lifecycle (bind, reset, initialize, configure,
validate) runs only when the simulator has a StateManager; `configure` receives
only `enforce_conservation` and `cf_strict_mode`: the facade ledger tolerance
is not simulator configuration, and its one source is
`coupling.scales.defaults.precision.<mode>.tolerance`, which the ScaleManager
reads when it initializes. `BaseManager.configure` checks a call through the
hook `_validate_config` before it stores or applies anything (ScaleManager
declares its settings there; a manager that declares none accepts the call as
before): an error raised by that check (for ScaleManager a `ValueError` naming
the rejected setting) reaches the caller unchanged and leaves the lifecycle
state and the settings as they were; only an error raised while the checked
settings are applied (`_do_configure`) is a lifecycle failure
(`ManagerInitializationError`, state `ERROR`, recovery through `reset()`).
`BaseManager.cleanup()` raises a `ManagerError` whose cause is the failure of
`_do_cleanup` and leaves the manager in `ERROR` (never forced to
`UNINITIALIZED`); an interrupted cleanup (`KeyboardInterrupt`, `SystemExit`,
cancellation) also leaves `ERROR` and propagates unchanged. `reset()`
propagates the error and a later `cleanup()` or `reset()` retries it.
`AsyncBaseManager`, `SingletonManagerMixin`, cleanup callbacks and the
`DeviceAwareManagerMixin` memory helpers were removed (no users). For
ScaleManager, `reset()` discards configured values: `initialize()` starts every
configurable setting from its constructor argument or its
`coupling.scales.defaults` parameter, which is why the simulator re-applies its
`enforce_conservation` and `cf_strict_mode` payload after `reset()`. Without a
StateManager the
simulator raises its own `RuntimeError`, and a lifecycle failure raises
instead of switching automatic scaling off. The same holds at construction
(`init_ops.initialize_scale_manager`): when automatic scaling is requested, a
ScaleManager that cannot be built or is unavailable, or an invalid mode,
raises. The CF grid fields are required only when a regrid can occur: two
descriptors with literally equal geometry are an identity mapping and need
none. The CF field list is the public constant
`CF_REGRID_TARGET_GRID_FIELDS` of `relic\data\standards\cf_conventions.py`,
and the variable declaration comes from the public
`ScaleManager.get_variable_metadata(name, scale=...)`.

The spatial operators share one conservation rule: an amount is conserved
on the support that the mapping covers, and the amount outside it is reported
(`<variable>_uncovered_source_amount`, `uncovered_source_amount`), never
rescaled into the covered cells. `ConservativeRemap` and
`ConservativeRegridder` use overlap weights without a whole-domain ratio;
both need declared geometry (overlap areas, or cell bounds from the file or an
explicit grid descriptor; bounds are never inferred from coordinate spacing,
so, for example, WRF curvilinear area densities cannot be regridded
conservatively and the call raises). `HierarchicalAggregator` resolves the
method per variable from its declared basis (categorical: area-weighted mode;
categorical disaggregation raises) and maps children to parents through a membership matrix
whose rows sum to 1 for shared children and computes
`Y_j = sum_i x_i A_i M_ij / A_j` for area densities or `X M` for cell
totals. `SpatialAggregation` and the graph-edge `SupportAggregationAdapter`
share one whole-support law (`aggregate_spatial_alias`): area densities and
intensive fields take the area-weighted mean (the basis is kept), cell totals
the sum, categorical fields the area-weighted mode; a caller strategy may only
name the law of the basis. The support adapter takes the basis from the SSOT
catalog `spatial_basis` of its port variables; the port `area_basis` is only
the area measure. Every amount ledger (hierarchical facade, the
covered-support ledger of `ConservativeRegridder` that `regrid_variable`
audits, `SpatialAggregation`, `SupportAggregationAdapter`) is evaluated in
every mode from the returned value and accepted by the one rule
`ledger_acceptance` (`relic\coupling\core\conservation_checker.py`; the
regridder ledger row by row through `covered_support_acceptance`, the
hierarchical facade ledger batch by batch through
`ConservationEnforcer.check_conservation(..., batch_ndim=...)`, so the residuals
of different batches never cancel and a call is conserved only when every batch
is): residual
`<= max(tol |L|, gamma_k(u) sum_i |a_i|)`, with `u` the unit roundoff of the
coarsest floating dtype of the reduction and `k` its count of rounded steps
(`2n + 2` for a reduction over `n` cells, `2n + 8` for a regridder row). The
strict path raises above it,
the other paths report it; no multiplicative correction is applied. The
tolerance `tol` has one source per ledger:
`coupling.scales.defaults.precision.<mode>.tolerance` of the call's mode for
the `ScaleManager` facade (its hierarchical ledger, its `regrid_variable`
audit, and the conservative regridder it builds for the call;
`ScaleManager.ledger_tolerance(mode)`), and
`coupling.scales.defaults.conservation_tolerance` for `SpatialAggregation`,
`ConservativeRemap`, the support adapter, and every regridder built without an
explicit tolerance; `ConservationEnforcer` holds no tolerance and every check
passes one. The scale audit records stay in `AuditCollector` (its own API and
JSONL sink); they never enter the StateManager diagnostics channel, which
holds numeric diagnostics only. Regridder reuse is keyed by the content
fingerprint of the grids
and arguments (`grid_utils.content_fingerprint`), never by shape or object
identity. `ScaleManager.regrid_variable` takes an explicit method (`strategy`);
a missing or unknown method raises before its audit record is written, and the
facade's default mode is the required
`coupling.scales.defaults.auto_scale_mode` (no conservative fallback). See
`docs\source\theory\scale_coupling.md` before chaining operators.

### 8.4 Computation

Directory: `relic\computation\`

Key sub-layers:

- `operators\`
- `solvers\`
- `gpu\`
- `parallel\`
- `distributed\`
- `jit\`
- `config\`

`OperatorMetadata` declares `input_units`, `output_units`, and
`spatial_dimensions`; the built-in operators declare the units documented by
their `apply()`. A solver call changes no solver tensors or attributes (I1):
per-call iterations, status, and residual are `SolverResult` fields, and
`ThomasAlgorithmSolver` records its Varah estimate in the result metadata.

`DistributedConfig` is the single owner of the distributed launch defaults;
`ProcessGroupManager`, `init_distributed`, and `setup_distributed_simulation`
accept only a configuration object or overrides of its fields.

**Coordinated registry transactions (E02).** `relic\computation\distributed\launcher.py`
owns `coordinate_transaction(operation, prepare=..., apply=..., compensate=...,
rollback_prepare=...)`, `decide_coordination(votes)`, `CoordinationOutcome`, and
`process_group_world()`. The world is the real `torch.distributed` group
(rank, world size); a mocked or half-initialized manager is not a world.
Phases and collective order per transaction: local prepare and vote gather
(`all_gather_object`), pure decision and decision gather (every rank must
compute the same decision or the transaction ends `decision_divergence`),
local apply and apply gather, and, only after an apply failure, compensation
on the ranks that applied and a compensation gather. The decision commits
only when every rank prepared successfully and every identity field agrees
(`epoch`, `contract`, `binding`, `device`, `rng`); keys prefixed `local_`
(such as the per-process provider UID) are recorded, never compared. Failure
model: crash-stop ranks, no Byzantine tolerance; a lost peer surfaces as a
collective error or the process-group timeout and the local outcome is
`partition` for that phase with `applied` telling whether the local apply had
happened; there is no unconditional atomicity or liveness under partition.
`RELICSimulator.register_functor` / `unregister_functor` run this protocol
whenever `enable_distributed` is on and the real group has `world_size > 1`
(`_coordination_active`), with the registration owner's contract and
binding digests, the resolved device, the stochastic substream seed, and the
provider epoch as the vote; a non-`committed` outcome raises with the
reasons and is kept in `get_coordination_outcomes()`. The distributed
ensemble-mean reduction of updates runs inside group execution before the
StateManager write (`_reduce_updates_across_ranks`; reducing after the write
averaged a stale copy), over sorted variable names, on raw tensors and
`UpdateItem` payloads alike. Real two-rank evidence:
`test\computation\distributed\coordination.test.py` with
`test\fixtures\distributed_coordination_worker.py`.

**Bounded mathematical contracts (G12).** Three finite-domain, decidable
contracts live in the operator layer and are documented in
`docs\source\theory\bounded_mathematical_contracts.md`. None of them is a
general theorem prover; inputs outside a stated premise are refused with a
reason. `relic\computation\operators\dec\chain_map.py` owns
`LinearChainMap` on declared `CochainSpaceSpec` objects: `y = M x`, identity,
composition, ledger preservation `C_t (u M) = C_s` with the unit factor `u`,
an operator-norm error bound, exact equality as the only equivalence, and
`close_within` as a distance bound that is deliberately not transitive.
`relic\computation\operators\invasion_threshold.py` owns
`activity_threshold`: for a nonnegative growth operator `G` (matrix or a
callable of the resource background `R0 = Lambda_R^-1 I_R`) and a
nonsingular M-matrix loss `Delta`, it computes `R_act = rho(Delta^-1 G)` and
re-checks the Metzler threshold lemma `sign(R_act - 1) = sign(s(G - Delta))`;
`R_act = 1` is reported as critical without a stability verdict, and no
nonlinear bifurcation or persistence claim is made.
`relic\computation\operators\relative_storage.py` owns the relative
storage `V = F(x) - F(x*) - grad F(x*)^T (x - x*)`, its full chain-rule rate
(the linear term is never dropped), `certify_relative_storage` (constrained
stationarity `grad F(x*) = C^T lambda` and Hessian positivity on `ker C`,
affine ledger leaves only), and `feedback_power_ledger`. Canonical consumers
with executable evidence: `LitterfallInterfaceMapper` as a declared
`LinearChainMap` (ledger preservation only; derivative commutation is not part
of this contract) with
carbon and nitrogen ledgers, the activity-free linearization of
`MicrobialEnzymeDecomposition`, and closed one-dimensional `FickianDiffusion`
whose reported dissipation matches the storage rate. The default tolerances of
these owners (`preserves_ledger`, `is_nonsingular_m_matrix`,
`activity_threshold`, `certify_relative_storage`, `feedback_power_ledger`,
and the `HodgeStarSpec` symmetry, PSD, and kernel checks) are the
`validation.iphs.*` parameters; an explicit argument overrides them, and a
default of 0.0 in `is_nonnegative`, `is_z_matrix`, or `validate_complex` means
an exact check. `category_theory.py` remains a blueprint and is not imported
by these owners.

The registered `bracketed_bisection` solver is owned by
`relic\computation\solvers\nonlinear.py`. It solves independent scalar equations
over tensor batches, with explicit scalar or batch-shaped parameters. For smooth
simple roots it provides first and second implicit parameter derivatives through
the returned root and a shared residual pullback, without storing the bisection
iteration graph or constructing a dense batch Jacobian. Singular root derivatives
are rejected. This numerical capability does not establish a thermodynamic
process closure or support coupled vector root systems.

## 9. Optimization, Stochastic, Models, Interfaces, Validation

### 9.1 Optimization

Directory: `relic\optimization\`

Sub-layers:

- `calibration\`
- `assimilation\`
- `sensitivity\`
- `uncertainty\`

The Monte Carlo controls of `EnsembleUQ` and `BootstrapUQ` (convergence
evidence floor and monitoring window, parametric bootstrap sample size) are
the `optimization.uncertainty.*` YAML convention values; the running-mean
diagnostic is a heuristic, not the Geweke or Gelman-Rubin test.

**Managed observation feedback (2026-09-14).** `relic\coupling\observation\
state_feedback.py` (`ObservationStateFeedbackFunctor`, a
`ScaleCouplingFunctor`) is the single governed state-feedback writer. It reuses
the identity observation operator of `relic\optimization\assimilation\base.py`,
forms the innovation `d = y - x` against a direct observation held in the
state, gates it with `|d| / sqrt(B + R) <= nu_max`, and proposes the scalar
Kalman analysis `x_a = x + B / (B + R) d` as a `REPLACE` update that the
StateManager commits (the increment is reported on the `observation` route as
a diagnostic together with the gain and analysis variance). The target
variable's SCM contract must list `observation_state_feedback` as a tendency
writer (`soil_moisture` does); any other target is refused by the contract.
`B`, `R`, `nu_max` live in `coupling.observation.state_feedback.*`. The
read-only observation operators (`TowerFluxSupportOperator`,
`TowerConcentrationProfileOperator`, `PostObservationQualityOperator`) remain
read-only. Evidence: `test\unit\observation\test_state_feedback_functor.py`
(hand-computed analysis, governed commit with provenance, gate refusal without
state change, analytic parameter gradient, durable restart).

**Satellite column operators (2026-10-18).** `relic\coupling\observation\
hcho_steady_state_column.py` (`HchoSteadyStateColumnOperator`) owns the
steady-state HCHO column `Omega = S n_C E + B` with the NOx-dependent
column-emission sensitivity `S` of Marais et al. (2012) (coefficient sets
`marais2012_standard`, primary, and `marais2012_paulot`, sensitivity, under
`coupling.observation.hcho_steady_state_column.*`; `n_C` from
`atmosphere.chemistry.isoprene_carbon_atoms`; NO2 columns below the
calibrated range are flagged, never clipped). `relic\coupling\observation\
satellite_column_kernel.py` (`SatelliteColumnKernelOperator`) owns hybrid
product layers `p = a + b p_s`, layer edges, air and tracer partial columns
`dp / (g M_air)`, the amount-conserving pressure remap (unmapped amount
returned) and the averaging-kernel column `N_v = sum_T A_l x_pc,l` with the
registered tropopause truncation variants. Both are read-only
`ScaleCouplingFunctor`s in the postprocessing phase; their `observation.*`
inputs have no SCM contracts, so they are not compiled into governed graphs
and research code calls their law methods. Theory:
`docs\source\theory\observation_operators.md`. Evidence:
`test\unit\observation\test_hcho_steady_state_column_operator.py`,
`test\unit\observation\test_satellite_column_kernel_operator.py`.

### 9.2 Stochastic

Directory: `relic\stochastic\`

Key files:

- `distributions.py`
- `ensemble_runner.py`
- `weather_generator.py` (theory:
  `docs\source\theory\stochastic_weather_generator.md`; no relative humidity
  or vapour pressure deficit output, see the WXGEN note there)
- `sde_solver.py`
- `gillespie.py`
- `validation.py`

IPHS/SDAE policy:

`relic\stochastic\` is the canonical owner of standalone stochastic algorithms
and services. Its current `StochasticProcess` hierarchy owns the Gillespie, SDE,
weather-generation, and ensemble implementations. No domain-level manager,
global cache, or API dependency shell exists. Callers construct the existing
algorithm classes with explicit inputs, while public API/CLI execution remains
fail-closed until the required scientific Functor, GraphIR, StateManager,
ParameterManager, unit, conservation, RNG, and restart bindings exist.
These standalone classes are not `FunctorBase` subclasses, and the presence of
`StochasticConfig` in `relic\core\config\models.py` does not prove that every
standalone service is already wired through that configuration.

`EulerMaruyamaSolver` is a numeric-only standalone interface for the declared
diagonal Ito equation `dX_t=a(X_t,t)dt+B(X_t,t)dW_t`. Its constructor requires
explicit drift and diffusion callables and its local seeded Torch generator
does not mutate the global RNG. It contains no decomposition, temperature, or
population process formulas and is not registered in the deterministic/
algebraic `SolverRegistry`. Moving it into `SolverBase` would expand that
registry beyond its current responsibility without supplying the missing
scientific binding.

The scientific binding of a standalone algorithm is a Functor, not the
algorithm itself. `relic\core\functor\stochastic_substream.py`
(`FunctorSubstreamMixin`) is the single owner of the Functor-owned
`torch.Generator` contract: seed validation, the substream position (seed,
draw count, generator bytes) as module extra state, device migration only
before the first draw, and the same-device check before a draw. It does not
touch global Python/NumPy/Torch RNG state, which stays with
`relic\core\simulator\checkpoint.py`. Two scientific Functors use it:

- `relic\scientific\atmosphere\stochastic_temperature.py`
  (`OrnsteinUhlenbeckTemperatureFunctor`, PhysicalField contract,
  `continuous_sde`) advances `stochastic_air_temperature_anomaly` with one
  fixed Euler-Maruyama step per governed call through `EulerMaruyamaSolver`,
  reading `alpha` and `sigma` only through `get_parameter_value`.
- `relic\scientific\atmosphere\stochastic_rainfall.py`
  (`MarkedPoissonRainfallFunctor`, PhysicalField contract, `marked_jump`)
  advances `stochastic_rainfall_event_count` and `stochastic_rainfall_depth`
  by sampling the arrivals of each fixed step with the direct `GillespieSSA`
  method (one constant-hazard channel) and exponential depth marks from the
  same substream; it emits the uncompensated increments together with the
  predictable compensators `lambda dt` and `lambda alpha dt` as diagnostics,
  and its per-cell arrival cap rejects a step instead of truncating it.

Both expose their exact discrete moments through a `discrete_moments`
classmethod, and accepted-step rollback and checkpoint restart replay the
same path for both.

The deterministic hybrid semantics `memory` and `ale` are bound the same
way, by scientific Functors instead of managers:

- `relic\scientific\pedosphere\transport\two_region_mass_transfer.py`
  (`TwoRegionMassTransferFunctor`, TransportMechanism contract, transport
  type `interphase_mass_transfer`, `memory`) advances the mobile and
  immobile solute concentrations with the exact exponential solution of the
  two-region exchange; the immobile concentration is the hidden realization
  of the exponential Volterra kernel, the extended storage
  `0.5 (theta_m c_m^2 + theta_im c_im^2)` never increases, and
  `check_conservation_laws` reports the theta-weighted ledger residual.
- `relic\scientific\pedosphere\transport\moving_domain_diffusion.py`
  (`MovingDomainDiffusionFunctor`, TransportMechanism contract, `ale`)
  advances a column of length `moving_domain_length` under a prescribed
  boundary velocity with the canonical finite-volume diffusive rate, the
  swept relative flux `c_face w` at every face, and cell volumes that
  satisfy the discrete geometric conservation law exactly; it emits the
  moving-face port flux, the column ledger residual, the GCL residual, and
  the Hodge-energy correction `-0.5 c^T dot(star) c`. Uniform stretching
  only; refinement, non-uniform mappings, and remap are `[N]`.
- `relic\scientific\pedosphere\transport\equilibrium_sorption.py`
  (`EquilibriumSorptionFunctor`, PhysicalField contract, `descriptor_dae`)
  advances the total solute mass `y = theta C + rho_b S` as the differential
  state and re-solves the algebraic phase concentrations of the Freundlich
  isotherm every step: the index-1 constraint is eliminated by the existing
  `BracketedBisectionSolver` on its exact bracket (implicit parameter
  gradients), inconsistent input triples are rejected instead of projected,
  and the constraint residual, degradation sink, and solver iterations are
  emitted; non-convergence fails the step. Transport coupling and
  competitive multi-component sorption are `[N]`.
- `relic\scientific\hydrosphere\runoff\surface_ponding.py`
  (`HortonianPondingEventFunctor`, PhysicalField contract,
  `topology_change`; `SurfacePondingFunctor`, TransportMechanism contract)
  realizes one real connectivity change of the process graph: the detector
  partitions the rainfall depth increment into infiltration and excess
  `E = max(0, dH - I_max dt)` and requests `ponding_onset` or
  `ponding_cessation` as a `topology` intent; the registration owner commits
  the request after the accepted step with the affine remap on
  `(h, E, a)`, an exact water-depth ledger audit (declared infiltration
  `Pi = -h^-` at cessation), and one provider epoch; the connected store
  drains ponded water with the exact linear-reservoir recession
  `h' = (h + E) exp(-dt/tau)`. Execution plans stay static: a plan-only
  executor refuses a post-transition checkpoint and the restart applies the
  same transition first. Tensor-shape dimension changes and multi-process
  swaps are `[N]`.
- Empirical histories that participate in production results are owned by
  canonical state or module extra state, never by Python attributes:
  `MaintenanceRespiration` keeps its acclimation temperature window as
  extra state (rolled back and restored with the step) and `GSIPhenology`
  keeps its rolling GSI history as StateManager state. Such empirical
  memories claim no passivity and are not `memory` semantic bindings.

`relic\stochastic\workflows.py` is the single public
graph owner: `BOUND_STOCHASTIC_PROCESSES` names the bound processes with
their semantic, `build_bound_stochastic_graph(...)` (and its semantic-checked
`build_bound_continuous_sde_graph` / `build_bound_marked_jump_graph` callers)
builds the `NodeRegistry`/`GraphIR` with the `stochastic_sdae_check`
execution binding, and `execute_bound_stochastic_process(...)` runs it through
`compile_graph(..., strict=True)`, `GraphExecutor`, and the StateManager
commit, then reconstructs every state trajectory from the recorded increments
and rejects any mismatch with the committed state. The HTTP routes
`POST /stochastic/sde` and `POST /stochastic/gillespie` and the CLI commands
`stochastic sde` and `stochastic gillespie` call that owner with a mandatory
seed; unbound process names, arbitrary reaction-network declarations, and
per-run parameter overrides (coefficients belong to the ParameterManager
database) fail closed with HTTP 422 or a `ValueError`, and
`get_available_samplers()` reports `runtime_ready=True` with
`bound_processes` for `EulerMaruyamaSolver` and `GillespieSSA`. The
weather-forcing route remains `[N]`. Variable steps on one Wiener path are
owned by the simulator adaptive step control together with the Functor-owned
Brownian tree (`FunctorSubstreamMixin.draw_wiener_increment`, pending-increment
ledger in the substream extra state, `record_rejected_trial_path` called by the
adaptive retry after the substream rewind): a rejected trial's increment
`Delta W(h)` is pinned on the path and the shorter retry draws the
bridge-conditioned `Delta W(h1) ~ N((h1/h) Delta W(h), h1 (h - h1) / h)`, the
remainder staying pending for the next step, so `Delta W(h) = Delta W(h1) +
Delta W(h - h1)` and fixed-step rewinds still replay the same trial. The
marked-jump analogue keeps the rejected trial's arrivals and marks as a
pending event set (`MarkedPoissonRainfallFunctor` event path ledger), so the
retry consumes exactly the events inside its window. The standalone
Euler-Maruyama solver stays fixed-step and accepts caller-supplied
`wiener_increments`. Adaptive Wiener-increment subdivision
without a Brownian bridge is also `[N]`.

A stochastic term embedded in a scientific Functor uses the same lifecycle,
parameter, state, unit, RNG, and applicable restart contracts as its
deterministic process. IPHS certificates and FDT evidence apply only to the
corresponding claimed structure or fluctuation-dissipation model. Conservative
stochastic transport uses its actual conservative flux and boundary
discretization; DEC cochains are required when DEC is the chosen method.
Deterministic management events require correct timing, reset semantics, and
budgets without requiring a Poisson or SSA formulation. Actual stochastic
jumps reuse the existing suitable event algorithm and restartable RNG substream.
Do not create a second stochastic solver, random runtime, SSA engine, ensemble
runner, or probabilistic validation path inside a scientific module.

### 9.3 Models

Directory: `relic\models\`

Key files:

- `basin.py`
- `patch.py`
- `landscape.py`
- `surrogates.py`
- `pytorch_surrogate.py`

### 9.4 Interfaces

Directory: `relic\interfaces\`

Entry points:

- CLI: `relic\interfaces\cli\main.py`
- API: `relic\interfaces\api\server.py`
- Web: `relic\interfaces\web\app.py`

API routes do not catch errors; `middleware.handle_errors` returns the
structured APIResponse error body with an HTTP error status (`HTTPException`
keeps its status, including routing 404/405; `RequestValidationError` 422;
`ValueError` 400; `FileNotFoundError` 404; `PermissionError` 403;
`ImportError` 503; otherwise 500). The API `/health/*` routes and the web
`/status` endpoint answer HTTP 200 for a HEALTHY (API body `success`) or
DEGRADED (`warning`) status and 503 otherwise. The web health views pass the
web app configuration explicitly to `HealthChecker` through
`core.services.web_runtime.build_health_payload(health_checker_cls, config,
detailed)`; a health check that cannot be completed answers 503 (the error
page for the HTML views). A web page that fails to render returns the error
page with HTTP 500. A health benchmark states its GPU part explicitly:
`HealthChecker` reads the required boolean health-configuration key
`enable_gpu_benchmark` (no schema default); the CLI sets it with
`health all --benchmark --gpu-benchmark|--no-gpu-benchmark`, the API with the
required query parameter `POST /health/benchmark?gpu_benchmark=true|false`.
Interface configuration is explicit: `core.services.health_runtime` takes the
caller's configuration mapping (None is a `TypeError`), the API does not start
when its configuration cannot be loaded, the CLI exits 1 when the configuration
cannot be loaded or has CRITICAL validation messages, and an invalid
`RELIC_LOG_LEVEL` is an error.
`core.services.runtime_config.load_runtime_config` reads the configuration file
named by the caller (CLI `--config`) or by `RELIC_CONFIG_PATH`; a named file
that does not exist raises `FileNotFoundError`. Without a named file it loads
the first existing path of `core.config.loader.DEFAULT_CONFIG_PATHS` (the one
rule is `core.config.loader.resolve_config_source`, shared by `ConfigLoader`
and `get_global_config`, whose cache key holds the resolved source, its mtime,
the overrides, both switches and the configuration environment variables)
(`./relic_config.yaml`, `./config/relic_config.yaml`,
`./examples/configs/default.yaml`, `~/.relic/config.yaml`,
`/etc/relic/config.yaml`) and reports it as the source; when none exists it
uses the global default configuration (`RELICConfig()` defaults), logs that
choice, and `relic info` shows "global default configuration". The web
interface raises `FileNotFoundError` naming a missing packaged `templates` or
`static` directory (an incomplete installation) and creates no directory.
Request schemas and CLI options carry no literal default for a scientific
quantity, a method or sampling choice, or a result field (for example the
calibration method, uncertainty sample count and sampling strategy, and the
realization count of the SDE, marked-jump and weather requests; the
sensitivity request requires the options of its method). A setting with a
documented value is read from the parameter database when omitted
(`validation.stochastic_scores.nominal_interval_coverage`,
`computation.diagnostics.*`). Boundary defaults remain only for software
switches (format, device, compression, benchmark workload) and are passed to
the internal call explicitly. Every request model (`schemas.RequestModel`,
including the nested `ParameterBound`) rejects unknown fields with HTTP 422;
fields that carry caller-defined keys are typed as mappings at field level,
and graph and assimilation payloads are plain mappings checked by their
owners. The request rate limit has one source,
`interfaces.api.rate_limit.requests_per_minute` and `.window_seconds`, and
invalid values raise; requests with neither an X-Forwarded-For entry nor a
transport client address share one rate-limit bucket
(`MISSING_CLIENT_ADDRESS_BUCKET`). Health status fields are read directly: a subsystem
result without `status`, a probe payload without `overall`, or an
unrecognized raw status is an error.
CLI commands report a failure with a non-zero exit code.

### 9.5 Validation

Directory: `relic\validation\`

The stochastic score rules (CRPS, PIT, Brier, interval coverage) read their
screening thresholds and significance levels from `validation.stochastic_scores.*`
through ParameterManager (`required=True`); a key in the context
`tolerance_overrides` (for the Brier event threshold and the nominal coverage:
in the context `metadata`) is an explicit override, and no literal default
exists. The CRPS, event and interval-width thresholds are compared in the
native units of the validated variable, which the generic rules do not
distinguish; they are screening values, not per-variable acceptance limits.
`relic.stochastic.validation.coverage_probability` follows the same rule:
without `nominal_coverage` it reads `nominal_interval_coverage`, and its
Clopper-Pearson interval uses `coverage_interval_confidence_level`.
`PITValidator` and `pit_test` without `alpha` read
`validation.stochastic_scores.pit_significance_level`, and
`relic.stochastic.workflows.run_validation_analysis` uses the stated
`nominal_coverage`, reads `nominal_interval_coverage` when it is None, and
raises when the namespace has no such attribute.
The parameter bounds rule reports a declaration that cannot form a
`ParameterSpec` as a failed result (`invalid_declaration`), never as a
comparison against its invalid bounds.

Core:

- `framework\` (engine, rules, registry, reporters)
- `cross_domain_validator.py`
- `conservation_guardian.py`
- `healthcheck.py`

Validation type taxonomy:

- Framework-wide validation severity and result types are owned by
  `relic\validation\framework\models.py`.
- The current framework model names are `ValidationLevel`,
  `ValidationContext`, `ValidationResultModel`, `ValidationDetail`, and
  `ValidationReportModel`.
- Domain validators may define domain-prefixed types such as
  `ParameterValidationResult`, `PFTValidationResult`,
  `CalibrationValidationResult`, `ConfigValidationResult`, or
  `ExternalValidationIssue` when their payload shape is domain-specific.
- Generic class names `ValidationLevel`, `ValidationResult`,
  `ValidationIssue`, and `ValidationContext` must not be newly defined outside
  the framework owner. `ValidationResult` and `ValidationIssue` may still occur
  as compatibility aliases in domain modules; they are not current framework
  model class names. New code uses the framework models or a domain-prefixed
  type directly.
- The architecture guard
  `test\unit\architecture\test_import_and_parameter_hygiene_l1.py`
  enforces this taxonomy and also protects the uniqueness of the canonical
  manager, registry, simulator, catalog, and compiler entry points.

## 10. Testing Architecture and Gating Specifications

Testing directory: `test\` (outside the package)

Main test hierarchy:

- `test\unit\`
- `test\integration\`
- `test\benchmark\`
- `test\system\`
- `test\scientific\`
- `test\validation\`
- `test\invariants\`
- `test\audit\`, `test\architecture\`, `test\lint\`
- `test\config\`, `test\parameters\`, `test\unit_conversion\`
- `test\orchestration\`, `test\regression\`, `test\verification\`

Invariant tests key files:

- `test\invariants\side_effects.test.py`
- `test\invariants\units.test.py`
- `test\invariants\conservation.test.py`
- `test\invariants\reversibility.test.py`
- `test\invariants\commutativity.test.py`
- `test\invariants\dt_scaling.test.py`
- `test\invariants\bounds_clipping.test.py`

Gating rules:

- `pytest.ini` is the test-discovery and marker SSOT. It enables importlib mode,
  strict configuration, strict markers, and recognizes `test_*.py`,
  `*_test.py`, `*.test.py`, and `*.spec.py`.
- Tests that write durable evidence, require serial execution, require a GPU, or
  require external data must declare the existing `writes_artifacts`, `serial`,
  `gpu`, or `external_data` marker.
- Do not proceed to the design of the next test file until the current test file has fully passed.
- Do not create functionally overlapping test files.
- Test naming and module boundaries must be clear.
- A test that writes to a process-wide owner restores it. Tests that write to
  the global ParameterManager request
  `test\conftest.py::restore_global_parameter_manager` from a module-level
  autouse fixture; a module-scoped setup snapshots and restores the manager
  itself. A test that needs a fresh manager requests
  `test\conftest.py::fresh_global_parameter_manager` (an empty slot; the next
  `get_global_manager()` builds a new manager, and the original object is
  restored afterwards); a module or class scope applies the same `setattr`
  through `pytest.MonkeyPatch.context()`. Tests that clear
  `PFTRegistry._instance` request
  `test\conftest.py::restore_pft_registry_singleton`. `test\validation\conftest.py` restores the validation registry, the
  Functor-metadata integrator and the rule module namespaces after every
  validation test. Resetting a global (`reset_global_manager()`,
  `reset_global_registry()`, `PFTRegistry._instance = None`) is not a restore.

`test\conftest.py::pytest_ignore_collect` excludes the Windows `nul`
placeholder and Python filenames whose dot-separated stem components are
not valid identifiers, including Explorer duplicate copies. This is a
discovery rule, not evidence that excluded files were executed. Ruff likewise
excludes `**/*(1).py`; user copies are not project verification targets.

The test session defaults `RELIC_SKIP_CALIBRATED=1` and
`RELIC_SKIP_LEGACY=1`, which change the loaded parameter view. Public science
reference freshness must use the production parameter view. The existing
`test\unit\publication\generate_public_science_reference.test.py` therefore
runs the generator in a child process without `RELIC_SKIP_CALIBRATED`,
`RELIC_SKIP_LEGACY`, or `RELIC_FAST_INIT`. Do not remove calibration files or
reuse an altered test singleton to regenerate production defaults.

## 11. Operational Specifications (Execution Sequence)

### 11.1 Before Starting a Task

1. Globally search whether the existing capabilities already exist.
2. If it already exists and meets the core requirements, mark it directly: `Completed (Feature Exists/No Action Needed)`.
3. If it does not exist, determine the minimal extension surface and record the scope of impact.

### 11.2 During Development

1. Modify documentation and parameters first, then modify the code.
2. Test immediately after completing a single module.
3. Proceed to the next module only after passing the test.
4. All parameters go through the parameter system, all units go through the unit system, and all constants go through SSOT.
5. Ecological/physical/chemical/transport calculations must attach the six-element "References + Mathematical expression + Parameter meanings + Units + Domain/constraints + Sign convention" comment block.

### 11.3 Before Finishing

1. Run the minimal executable verification.
2. Run related tests.
3. Check the consistency between paths and documentation.
4. Check whether duplicate modules or parallel implementations have been introduced.

## 12. Extension Methods (By Object)

### 12.1 Extending a Scientific Functor

1. Add theoretical documentation in `docs\source\theory\`.
2. Add parameter entries to the parameter database YAML (complete metadata, non-empty references, correct dimension).
3. Select the bridge class and implement the Functor.
4. Completely fill out `FunctorMetadata`; declare every input in
   `input_variables` and list the inputs a call may omit in
   `optional_input_variables` (in `__init__`, or where a configured value such
   as a bound velocity field replaces the state input).
   `FunctorBase.get_required_inputs()` (declared minus optional, in declaration
   order) is the only required-input rule; `validate_state`, the validation
   pipeline, and `NodeRegistry.register_functor_instance()` read it. No Functor
   overrides it, and no Functor filters `validate_state` messages to make an
   input optional. `FunctorMetadata.parameters` lists the full keys the
   Functor reads (`required_parameters()`, certificate `parameter_keys`) with
   meaning and units; input variables belong only to `input_variables`. An
   entry is an exact key or a whole-key family written with `*` or `<pft>` (every
   matching key is read); a dict entry carries `description` and `units`, and a
   `namespace` is joined as the generator joins it. Framework keys (bridge
   response keys, local conservation keys, Hodge keys) are not listed; a Functor
   that calls `_enforce_forward_conservation` lists `conservation_tolerance` and
   `strict_conservation`. `test\scientific\metadata_required_fields.test.py`
   enforces this contract. `optional_output_variables` must be a subset of
   `output_variables`; a missing optional output is not reported by the unit
   rule. The
   science-reference generator matches parameter rows to full
   ParameterManager keys exactly (a declared `*`/`<pft>` pattern matches whole
   keys); a declared name that is no key is rendered as not declared in the
   parameter database and is never mapped to a key ending with the same name.
5. Register in `relic\scientific\` (explicitly or via automatic scanning).
6. Add the corresponding test file to `test\`.
7. Proceed to the next module only after passing the test.

### 12.1.1 Modifying or Deleting a Scientific Functor (Mandatory Process)

1. **Before modifying**: List all parameter keys associated with this Functor.
2. **After modifying**: Check which parameters are no longer referenced by any code (search for `get_parameter_value` and `manager.get` calls).
3. **Delete orphan parameters**: Delete parameter entries no longer in use from the parameter database YAML.
4. **Synchronize namespace**: If the process is renamed, the namespace of the parameter keys must be updated synchronously.
5. **Run verification**: `pytest test\unit\parameters\ -v`.
6. **Update theoretical document**: Synchronously update the theoretical documentation under `docs\source\theory\`.

### 12.2 Extending a Parameter

1. Choose the correct namespace and tier. A key must not be the dot prefix of
   another key (Section 6.1): next to a namespace `<group>.*`, name the value
   `<group>_<quantity>`.
2. Add complete fields in the parameter database YAML.
3. Validate `units`, `dtype`, `bounds`, `references`, `provenance`.
4. Use them hierarchically in code: inside a Functor, use only
   `self.get_parameter_value(...)`; owner-constructed non-Functor helpers use
   required constructor injection plus `manager.get(...)`; top-level
   infrastructure without an injection owner may use
   `get_global_manager().get(...)`. Use
   `manager.get(..., bounds=(lower, upper))` when boundary clipping is needed.

### 12.3 Extending a Physical Constant

1. Write only to `relic\parameters\database\relic\physical\constants.yaml`.
2. Read it through the existing generic
   `relic\utils\physical_constants.py::get_physical_constant(...)` accessor;
   do not add a per-constant wrapper or duplicate mapping.
3. Execute constant-related tests.

### 12.4 Extending an Orchestration Node Capability

1. Extend within the `orchestration` layer; do not duplicate orchestration logic in the business layer.
2. Prefer reusing `NodeRegistry`, `PortSpec`, `validate_graph`, `compile_graph`.
3. If optimization is needed, use the compilation Pass extension method.

### 12.5 Extending a Variable Contract

1. Add YAML in the corresponding sphere layer in `relic\contracts\variables\`.
2. Follow `scm_variable_schema.yaml`.
3. Ensure variable names and units are strictly consistent with parameters/states.

### 12.6 Extending a Test File

1. Place under the corresponding module directory in `test\`.
2. Name clearly, do not use generic names.
3. A single file tests only a single module or single functional domain.
4. Fully pass the current file before proceeding to the next.

### 12.7 Loading an External or User Extension

1. For Python or manifest-based extensions, reuse `relic\extensions\` for
   discovery, validation, loading, and registration. Declare the extension in
   `relic_extension.yaml`; each functor declaration states `execution_group`,
   and a manifest without it is rejected.
2. For language-agnostic C ABI or shared-library coupling, reuse
   `relic\coupling\plugin\`, `PluginHost`, and `PluginFunctorAdapter`.
3. Register adapted Functors through the existing `RELICSimulator`,
   `NodeRegistry`, and ParameterManager paths.
4. Validate bridge inheritance, metadata, dependencies, and parameter files
   before registration. The validator instantiates each functor with the
   `config` its manifest declaration carries, exactly as the loader will.
5. Unload through `unload_extension` / `unload_from_runtime`; never clear a
   loader record before the simulator retirement and the native release have
   both completed, and retry a summary whose `complete` is False.
6. Do not create another plugin manifest, discovery service, loader, registry,
   scheduler, or runtime.
7. Validation runs for every discovered extension before any is loaded;
   `strict` defaults to True for `load_extensions`, `load_single_extension`,
   `load_and_register_extensions`, and `register_with_orchestration`.
   Non-strict loads return `LoadedFunctors` whose `skipped_extensions` names
   every skipped extension or duplicate and its reason. Import, construction
   (no retry without the declared config), and core-registry failures raise
   `ExtensionLoadError` with the original cause; Functors constructed before
   the failure stay recorded so `unload_extension` can release them.
   `load_and_register_extensions` requires an explicit `parameter_manager`,
   loads parameters only for validated extensions, and binds the loaded
   Functors to that manager. Discovery raises on an invalid manifest or an
   unresolvable entry point. Unloading an extension that is not loaded raises
   `KeyError`.

## 13. Metadata Integrity Specifications

Functor metadata must be able to answer:

- What this process does.
- When it is executed.
- What the input and output variables are.
- On what literature and equations it is based.
- What conservation laws are involved.
- What the parameter names, units, and physical meanings are.
- **What is the spatial scale of this process's output** (`spatial_scale`, should be a `SpatialScale` enum value).
- **What is the Functor's declared role label** (`variable_kind`, a `VariableKind` value; not a per-variable quantity representation).

:::{figure} ../../assets/readme/relic_variable_roles.png
:alt: State, forcing, parameter, flux, and diagnostic roles around a RELIC Functor.
:width: 95%
:align: center

`variable_kind` is one Functor-level role label. At runtime it is used by
output-overlap governance (two STATE writers of one output are rejected) and
by introspection; state advancement is selected by each `UpdateItem`, and
conservation membership by the state contracts and `conservation_laws`. It
does not declare the quantity basis, which scale operators receive as
explicit `spatial_basis`/`temporal_semantics` (Section 8.3). `UNKNOWN`
is an undeclared sentinel, not a sixth scientific role.
:::

### 13.1 category_theory.py State Declaration

The file `relic\core\category_theory.py` acts as a **DESIGN BLUEPRINT** and is **not integrated into runtime code**.

- This file is not imported by any runtime module.
- It is not exported from `relic.core`.
- It is strictly forbidden to instantiate classes defined in it within production code.
- It is retained solely as a theoretical reference document.

Parameter metadata must be able to answer:

- Where the parameter originates from.
- What the applicable units and ranges are.
- Whether it is learnable and optimizable.
- Whether the version tracking and source record are complete.

## 14. Anti-Redundancy Red Lines

Forbidden behaviors:

- Creating a secondary simulator/state/parameter manager inside `relic\`.
- Bypassing the bridge layer to directly write "contract-less Functors."
- Defining physical constants in non-SSOT locations.
- Hardcoding values within scientific processes that should be parameterized.
- Creating scattered test files outside of `test\`.
- Switching to the next test file before the current test file is completed.
- **Retaining unused orphan parameters after deleting or replacing a scientific process.**
- **Adding new parameters without checking whether a parameter with the same meaning already exists.**
- **Using incorrect physical quantity classifications for the parameter's
  `dimension` field** (validate against the parameter schema and existing
  unit/dimensionality services).
- **Leaving the `references` field of a parameter as an empty list.**
- **Using `units: 1` or fractional units (e.g., `day-1`).**
- **Failing to declare `input_variables` and `output_variables` when adding a new Functor** (triggers empty declaration warning).
- **Using `UNKNOWN` for `spatial_scale` while the output variables need to participate in scale consistency checks.**
- **Instantiating classes in `category_theory.py` or importing it into runtime code** (this file is only a design blueprint).
- **Ignoring output variable overlap warnings during registration (`_detect_output_overlap`).**
- **Creating a second checkpoint serializer, format/version registry, restart
  manager, or StateManager-backed file persistence path instead of delegating
  to `relic\core\simulator\checkpoint.py`.**
- **Reintroducing an implicit checkpoint destination, unsafe/legacy load
  fallback, or a destructive checkpoint cleanup API.**

## 15. Windows Execution and Check Commands

Architecture audits invoke `rg` as a subprocess. Resolve its actual installed
path with `Get-Command rg` and verify `rg --version`; the audit process must
inherit that directory on `$env:PATH`. A user-local tool installation directory
is not a stable dependency path.

Windows NetCDF compression requires the selected HDF5 filter and its native
DLL dependencies to load inside the active environment. Qualify a configured
filter with a real write/read round trip; importing `netCDF4` alone is
insufficient. Repeat the qualification after replacing native packages.
Host-specific DLL repair records belong to environment evidence and do not
establish compatibility on other hosts.

Suggested commands (PowerShell):

```powershell
conda activate RELIC
python -c "import relic; print('[OK] import relic')"
python -m relic.interfaces.cli.main --help
python -m relic.orchestration --help
pytest test\invariants\ -v

conda activate sphinx_env
powershell -NoProfile -ExecutionPolicy Bypass -File docs\build_public_docs.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File docs\build_rag_docs.ps1
```

`docs\build\` is generated output. Never hand-edit its HTML or copy an old build
as evidence; always rebuild it from `docs\source\public\` through the existing
`docs\build_public_docs.ps1` owner. The public build emits matched English and
Chinese trees under `docs\build\en\` and `docs\build\zh\`, omits internal
migration plans, and does not copy source pages. `docs\build_rag_docs.ps1`
derives the bilingual RAG corpus under `docs\rag\` from that validated HTML.
Both scripts run in the dedicated `sphinx_env`; the public builder calls the
sibling RELIC interpreter only for live scientific-reference generation and
freshness checks. Publication verification must also check link closure,
rendered table shape, paired language routes, duplicate pages/tables, math
nodes, and non-English script content.

After scientific-source, parameter, or certificate changes, regenerate with
`docs\generate_public_science_reference.py --write` and
`docs\generate_process_contract_inventory.py`, then run each with
`--check`. Process main certificates and additional constructor-scoped
runtime-point certificates remain distinct in the inventory. Strict Sphinx
success alone does not prove freshness; the generator checks and existing
document guards are separate requirements.

After changing public source text, extract fresh gettext catalogs from
`docs\source\public`, reconcile changed message IDs with
`docs\locale\zh_CN\LC_MESSAGES`, translate the changed messages, and compile
the corresponding MO files before rebuilding HTML and RAG. Zero fuzzy or
empty entries in an old catalog is not evidence of zero source drift. Never
hand-edit generated HTML, RAG, PDF, or translation binaries. Both publication
scripts explicitly return zero only after all checks and publication succeed;
Robocopy status 1-7 denotes successful copying, while status 8 or above fails.

### 15.1 Executable Tutorial and User-Guide Contract

The canonical executable tutorial layer is `tutorial\primary`: 30 notebooks
(`00` through `29`) organized as foundation, theory, and applied/advanced
tracks. `tutorial\docs\primary` explains the learning routes but does not
replace this architecture SSOT or the production modules exercised by the
notebooks.

Tutorial notebooks must:

- locate the repository from the current working directory or the explicit
  `RELIC_PROJECT_ROOT` environment variable; a developer-machine absolute path
  is not an allowed fallback;
- call current public RELIC owners and canonical state/output names rather
  than restoring deleted aliases or compatibility modules;
- state references, equations, parameter meanings, units, domains, and sign
  conventions for scientific calculations;
- preserve current fail-closed and capability-health results instead of
  replacing them with tutorial-only success claims; and
- use analytic, hand-computed, ledger, restart, or holdout oracles appropriate
  to the demonstrated claim. Execution without an exception is not a
  scientific acceptance criterion.

The release workflow is owned by `tools\notebook_primary_audit.py` and
`tools\execute_primary_notebooks.py`. A tutorial release is accepted only when
the same batch report covers all selected notebooks with `status="ok"`, zero
error outputs, zero unexplained empty code outputs, and output cells saved from
that run. The report belongs in
`reports\notebook_primary_execution_final.json`; its counts are release
evidence, not stable architecture constants.

The API and Web source trees exist, but their servers require their optional
runtime packages to be installed. A base-package import or CLI help check does
not prove that FastAPI/Uvicorn/Jinja-backed interfaces are available.

Path formatting specifications:

- Use backslashes `\`.
- Use drive-letter root paths, e.g., `C:\`.

## 16. Quick Review Checklist

- Do all paths actually exist?
- Does the documentation only describe the existing architecture with no outdated paths?
- Is a single architectural narrative maintained without duplicated sections?
- Are "operational specifications + extension methods + metadata info" clearly documented?
- Do scientific process specifications cover all six mandatory elements: "References + Mathematical expression + Parameter meanings + Units + Domain/constraints + Sign convention"?
- Are parameters/units/constants strictly routed through their unified systems?
- Is the testing strategy consistent with the `test\` structure?

## 17. Standalone Utility Parameter Access

Section 6.1 is the canonical parameter-access contract. Standalone utilities
follow its non-Functor path and must not be used to bypass Functor namespace,
metadata, or lifecycle checks.

## 18. Forward-Only Execution Contract

This section is a hard architectural contract for scientific functors.

- The scientific kernel API is `forward(state, time, dt, **kwargs)`. Normal
  simulator execution calls the Functor object through
  `FunctorBase.__call__()` to apply lifecycle and governance checks.
- For subclasses of `BiologicalProcessFunctor` and `TransportMechanismFunctor`, do not define `compute_rate()` or `compute_flux()`.
- Bridge classes must not contain dual delegation (`forward <-> compute_*`) and must not provide silent zero-output fallback behavior.
- Do not introduce compatibility aliases, deprecated wrappers, or warning-based transition layers for `compute_rate` / `compute_flux`.
- Simulator integration, templates, examples, and governance tests use object
  calls. Direct `forward(...)` tests are kernel tests; their results cannot
  establish that governance obligations were executed.

`FunctorBase.set_call_governor(...)` binds one runtime policy callback at the
existing timed-call boundary. It receives the standard `nn.Module` call as a
continuation; `GraphExecutor._wrap_functor(...)` binds its declared input
projection, cache, RNG audit, and lineage policy there without replacing
`forward`. Validation precedes this
boundary, and postprocessing, conservation, and output validation still run
when that policy returns a cached raw result.

Local or global PyTorch forward/pre/backward hooks make a node ineligible for
cache and DIRTY-result reuse, so hooks execute through PyTorch's own dispatcher.
Autograd and hook behavior retain native output restrictions: full backward
hooks do not support arbitrary dictionary outputs. This interface does not
add a second executor, scientific kernel, or checkpoint state owner.

Mandatory guardrails:

- Static guard script: `scripts\architecture\assert_forward_only_functor_api.py`
- CI audit test: `test\audit\forward_only_functor_api.test.py`

## 19. Parameter Access Enforcement

Section 6.1 is the single normative definition. Static architecture checks may
enforce that contract, but they must not create another accessor, compatibility
shim, or transition API.

## 20. Conservation Check Architecture

RELIC uses four complementary conservation layers for mass, energy, and element
conservation. They do not all share the same blocking semantics.

### 20.1 4-layer architecture

```
Layer 1  FunctorBase call path  __call__() invokes _check_local_conservation()
Layer 2  FunctorBase helpers    check_conservation_laws() evaluates declared outputs
Layer 3  StateManager           commit-time element ledgers and diagnostics
Layer 4  Guardian               ConservationGuardian.check_domain() domain balance
```

Inside an accepted-step transaction every group commit runs the Layer 3
commit-time checks (`apply_updates` with `check_conservation`: the simulator
switch `enable_conservation_checks` and the StateManager switch validated from
the `conservation_checks` setting, `conservation_checks["enabled"]`), while
StateManager's per-commit `generate_conservation_report(...)` is suspended for
the transaction (`set_auto_report_enabled(False)`). The step loop builds no
report of its own: a report needs the committed, normalized UpdateItems and the
interval of every commit (a boundary flux is a REPLACE rate over its commit
interval), which only StateManager holds; the returned mapping of payloads of
`step()` carries neither. An accepted step therefore adds no summary to
`conservation_history`. `conservation_checks` is a setting, not a feature flag.

### 20.2 Standard implementation pattern

Simulator execution calls the Functor object, not `forward(...)` directly.
`FunctorBase.__call__(...)` therefore owns the normal post-forward
`_check_local_conservation(...)` hook. Scientific modules must not copy a
second generic tolerance loop into every `forward(...)`.

A small number of modules that intentionally support direct `forward(...)`
invocation may call `_enforce_forward_conservation(...)` explicitly. This is a
special direct-call safeguard and uses its documented parameter keys; it does
not replace the normal `__call__()` hook.

### 20.3 Configuration SSOT

File: `relic\parameters\database\relic\validation\conservation.yaml`

Current values and bounds are owned only by the YAML file and are intentionally
not duplicated here.

| Parameter | Effect |
| --- | --- |
| `conservation_tolerance` | Threshold used by `_enforce_forward_conservation(...)` for intentional direct `forward(...)` calls |
| `strict_conservation` | When enabled, that direct-forward safeguard raises `ValueError` above `conservation_tolerance` |
| `functor_tolerance` | Warning threshold for the canonical `FunctorBase.__call__()` local hook |
| `functor_severe_threshold` | `ConservationError` threshold for custom conservation checkers on the canonical hook; the base checker does not raise on this key |
| `water_conservation_tolerance`, `carbon_conservation_tolerance`, `nitrogen_conservation_tolerance`, `phosphorus_conservation_tolerance`, `energy_conservation_tolerance` | Dimensionless relative tolerance of the matching budget check of the validation rules and coupling contracts (one key per budget; no per-call override) |
| `plausibility.*` (GPP upper bound, absolute net radiation bound, latent heat lower bound, Bowen ratio upper bound) | Warning bounds of the carbon and energy validation rules; model plausibility bounds without a literature value |
| `stoichiometry.*` (ratio tolerance and the C:N, C:P, N:P ranges) | The one source of `check_cnp_stoichiometry_impl`, which takes no tolerance argument |

The four layers above judge physical budgets and take their tolerances from
`validation\conservation.yaml`. The StateManager default,
`SoilVegetationCoupling`, and the history rule `ConservationRule` (every
recorded amount against the first; the cells of the compared records and the
step count enter the rounding count of `ledger_acceptance`) read
`validation.conservation.conservation_tolerance`.
`budget_closure(tolerance_parameter, residual, terms, *, summands=None,
rounding_activity=None)` in `relic\validation\cross_domain_coupling_ops.py` is
the one relative-residual definition, `max_i |r_i| / sum_k |t_k,i|` (the gross
activity of the terms of the identity, no floor), judged by `ledger_acceptance`
against the one key its caller names; operands are compared in float64 and the
rounding term uses the unit roundoff of the coarsest floating dtype actually
rounded. `rounding_activity` (finite and at least the gross activity, else
`ValueError`) widens only the rounding term, for an identity whose residual also
rounds operands larger than its terms (the G3 bucket storage rate from two
float32 contents); each element is then judged separately. The budget checks of
the validation rules and coupling
contracts and the budget identities of the G3 compile witnesses use
`validation.conservation.<budget>_conservation_tolerance` (water, carbon,
nitrogen, phosphorus, energy), and the witness tower-support comparisons use
`coupling.observation.tower_flux.support_closure_tolerance`; a per-call
override is refused. A budget with a missing term is not evaluable
(`passed=False`, no residual, missing terms named); no term is taken as zero,
so a day-long `G = 0` (FAO-56 Eq. 42) is supplied by the caller. The nitrogen
budget is the total soil mineral N budget (CASA-CNP, Wang et al. 2010);
nitrification is internal and not a budget term. A rule whose whole flux mapping
is absent returns WARNING with `passed=True` and states `not applicable (domain
absent)`; this is not a passed budget.
The accuracy of a scale-mapping operator (spatial regridding, aggregation, and
remapping) is not a fifth layer but the scale precision policy:
`coupling.scales.defaults.precision.<mode>.tolerance` for the ScaleManager
facade (per call and mode, with the mode's dtype and enforcement flag;
`ScaleManager.ledger_tolerance(mode)`) and
`coupling.scales.defaults.conservation_tolerance` for mode-less scale
operators and regridders. Each check reads exactly one of these keys; neither
family falls back to the other, and no shared resolver chooses between them.

### 20.4 FunctorBase.check_conservation_laws

The base method uses declared `output_variables` from `FunctorMetadata` to
select relevant state updates, aggregates changes for each declared law, and
returns `{law_name: violation_magnitude}`. If no output variables are declared,
the current implementation returns no local results; there is no keyword-based
heuristic fallback. Empty metadata is therefore a coverage gap, not an implicit
conservation check.

### 20.5 ConservationGuardian

File: `relic\validation\conservation_guardian.py`

Domain-level conservation checking using mass balance:

```
delta_storage = (flux_in - flux_out) * dt
```

Uses configurable tolerances (atol/rtol). The `check_domain()` method logs errors but does not raise exceptions (non-blocking). The `compute_invariant_drift()` method evaluates the linear-invariant drift `(c^T F) dt` for a caller-supplied left-null-space row `c` of a physical exchange incidence matrix (`G_phys`). It is a non-blocking diagnostic, not a symmetry (Noether) proof, and it must not be built from the execution dependency graph.

### 20.6 Forbidden

- Hardcoding conservation tolerance values in `forward()` methods.
- Bypassing `FunctorBase.__call__()` in simulator execution.
- Disabling declared conservation metadata to avoid the shared check.
- Implementing custom conservation logic that bypasses `FunctorBase.check_conservation_laws()`.
- Adding ad hoc warning or exception policies outside the canonical
  `__call__()` conservation hook or the documented direct-forward safeguard.
- Inconsistently mixing patterns across scientific modules.

## 21. SSOT Subsystems

Beyond the Parameter System and physical/unit/time rules in Section 6, RELIC
enforces SSOT policies for the following subsystems.

### 21.1 SSOTCatalog -- Orchestration Variable Metadata

File: `relic\orchestration\ssot_catalog.py`

The `SSOTCatalog` loads variable metadata from the centralized parameter system. It must not hardcode units, bounds, or defaults.
`validate_graph` without an explicit catalog loads
`SSOTCatalog.from_parameter_manager()`; a load failure raises.

**Data sources (loaded in this order):**

- `coupling.external.variables.variable_definitions` -- external variable definitions
- `coupling.scales.defaults.variables` -- scale-default variable definitions
- the ContractRegistry variable contracts, read through `load_contract_registry()` -- merged into the entries; contract aliases become catalog aliases
- `coupling.external.variables.aliases` -- optional alias table mapping each alias to a variable name

**Selected entry fields (`SSOTEntry`):**
| Field | Type | Description |
| ---------------- | -------------- | -------------------------------------------- |
| `name` | str | Variable key |
| `standard_name` | str | Declared `standard_name`, else the variable key; with `cell_methods` it forms the entry's CF identity, which one variable holds |
| `cf_standard_name` | str | Optional CF controlled-vocabulary name; `None` remains absent and never falls back to local `standard_name` |
| `units` | str | UDUNITS-compatible unit string |
| `spatial_basis` | SpatialQuantityBasis or None | Declared spatial representation, resolved by `resolve_spatial_basis`; `None` when undeclared |
| `quantity` | str | Physical quantity (e.g., `temperature`) |
| `element` | str | Tracked element (`carbon`, `nitrogen`, etc.) |
| `semantics` | str | `state`, `flux`, `diagnostic` |
| `write_mode` | str | `mutable`, `derived`, `immutable` |
| `dimensionality` | Dimensionality | Auto-derived from units |

**API:**

```python
from relic.orchestration.ssot_catalog import SSOTCatalog
catalog = SSOTCatalog.from_parameter_manager()
entry = catalog.get("soil_temperature")  # by name, explicit alias, or unqualified standard_name
daily_max = catalog.get_by_standard_name("air_temperature", cell_methods="time: maximum")
```

**Conflict detection:** The catalog keeps no conflict list. `from_parameter_manager()`
raises `ValueError` on every conflict, so a catalog that loads has none. A second
declaration of one variable name is merged when it agrees and raises when it
contradicts on `units` (`1` and dimensionless spellings count as equal), `semantics`,
`cf_standard_name`, `update_mode`, `write_mode`, `spatial_basis`, `quantity`,
`element`, `datatype`, or a `context_requirements` key; a contract merge compares only
`units` and `cf_standard_name`. One CF identity (`standard_name` plus `cell_methods`)
belongs to one variable: a second variable declaring it raises
(`cf_conventions.register_cf_identity_holder`). A variable contract that declares
`dimension.cf_standard_name` holds that CF identity, qualified by its optional
`dimension.cell_methods` (`VariableContract.cell_methods`), in the same holder
registry after its aliases are merged: two contracts, or a contract and a catalog
entry of another variable, that declare one identity raise. The entry's local
`standard_name` (its connection identity) is unchanged, and `get_by_standard_name`
resolves a contract CF identity to its holder (for example `air_temperature` with
`time: maximum` resolves to `air_temperature_max`). An entry name that is already an
alias, an alias that names another variable or is claimed by two variables, an
alias-table entry that is not a name pair or targets an unknown variable, a contract
alias that is itself a variable contract, and a non-empty `ContractRegistry.conflicts`
also raise. A parameter entry listed under a contract alias is merged into the
contract variable, which keeps the only entry.
When a VariableContract exists, its optional CF-name decision is canonical:
identical explicit names merge, an absent existing value may receive the
contract value, and any different non-null value or an existing non-null value
against a contract `None` fails closed. Blank and non-string CF-name fields are
invalid. Local `standard_name` remains a separate connection identity.
`NodeRegistry` projects this optional field to `PortSpec` without fallback:
missing catalog entries and explicit `None` remain `None`, while local port
identity continues through `standard_name`. Port serialization retains an
explicit JSON/YAML null instead of inventing a CF controlled-vocabulary name.
Connection validation uses a deterministic identity mode. When both ports have
explicit CF names, those exact case-sensitive names are compared and a shared
CF identity may bridge different local names; all remaining unit, quantity,
shape, support, mode, and semantic gates still apply. When either CF name is
absent, no CF issue is emitted and local `standard_name` values are compared
instead. Local mismatches use `standard_name_mismatch`; bilateral explicit CF
mismatches use `cf_standard_name_mismatch`. Non-strict warnings from either
taxonomy still make the semantic audit fail.

The registered metadata rule is `cf_standard_name_syntax` in
`relic\validation\framework\rules\cross_domain.py`. It checks only an explicit
RELIC controlled-vocabulary base identity against `[a-z][a-z0-9_]*`; missing or
`None` remains neutral. A syntax pass does not establish official CF table
membership, alias resolution, standard-name modifier semantics, canonical-unit
compatibility, quantity semantics, or dataset-wide CF conformance. Those
capabilities belong to the canonical `CFConventions` owner and its
fingerprinted official table and executable negative oracles described below.
The lexical rule supplies none of that additional evidence. CF features outside
the explicitly implemented dataset scope remain `[N]`. The public
`CrossDomainValidator.validate_cf_compliance` facade and its former
`cross_domain_interface_ops` helper have been removed. The historical
unregistered framework wrapper has also been removed; cross-domain CF metadata
now has one registered lexical-syntax rule and no parallel compliance callable.
This entrypoint uniqueness is architectural evidence only, not official CF
conformance evidence.

Official standard-name vocabulary data is bundled once at
`relic\data\standards\cf_standard_name_table.xml`. The immutable artifact is CF
Standard Name Table v94 (2026-06-09), with source SHA-256
`39a11ab9ccc327756da1bada1f268eebabdb7cee0cc908464962197bb892165c`.
It is package data, not a second runtime registry: `CFConventions` remains the
only permitted parser and query owner. Runtime network refresh, generated
JSON/YAML mirrors, and ParameterManager-backed official membership are
forbidden. The Standard Name Table version is independent of a dataset's CF
Conventions version and cannot authorize a `Conventions` global attribute.

`CFConventions` now parses that XML offline into separate entry and alias
indexes. Exact entries take precedence over the two v94 cross-kind ID
collisions; alias targets remain provenance. The single multi-target alias is
not collapsed to one quantity. Undefined canonical units remain `None`; dB uses
only the finite CF FAQ spelling set. All other defined unit relations use
`cf_units.Unit.is_convertible(...)`. `UnitConverter` executes the
same-dimension pure-ratio pairs among them (zero offset) through UDUNITS.
When no `UNIT_CONVERSIONS` key matches, two cross-dimension rules apply
without a table entry: the liquid-water-equivalent rule `F = rho_w * d`
(depth or depth rate to areal mass or mass flux, and back) for variables
declared liquid water by name (`lwe_*`, or names normalized to `water_flux`,
`water_amount`, or `precipitation_flux`, such as `runoff_flux` or
`snow_melt_flux`), with `rho_w` the SSOT constant
`relic.physical.constants.water.density_liquid` (1000 kg m-3), never a state
density (`5 mm day-1` converts to `5.787037e-5 kg m-2 s-1`; the unit pair alone never
selects this rule, so the variable `generic` raises); and the element
mass/amount rule `n_E = m_E / M_E` for labels carrying the same single
element token (C, N, or P), with `M_E` from
`relic.physical.constants.molecular_weights.<element>`. Affine pairs (for
example `degF` to `K`) and every other cross-dimension pair still need an
explicit table rule, and
the DataReader converts only when both units are CF-consistent with the
variable's `standard_name`. The existing broad method name
`validate_cf_compliance(...)` is temporarily retained only because production
callers still exist; its result explicitly states
`validation_scope=single_variable_standard_name_and_units`. Dataset-level
conformance is owned by `CFConventions.validate_dataset_conformance(...)`
(2026-09-14), whose result names its scope verbatim
(`declared_dataset_scope`): the `Conventions` declaration (2.6.1), variable
and coordinate `units` (3.1, boundary variables inherit units per 7.1),
official standard-name membership and canonical-unit compatibility (3.3),
`axis` values, time coordinate units and the CF calendar vocabulary (4.4),
`coordinates` references (5), `bounds` references and shapes (7.1),
`_FillValue`/`missing_value` consistency (2.5.1), and `cell_methods` syntax
and dimension references (7.3). Grid mappings, discrete sampling geometries,
compression by gathering, external variables, and group hierarchies are
outside the scope and are neither accepted nor rejected. The API name
alignment of the single-variable method remains `[N]`.

DataIO, DataReader, and DataWriter preserve explicit metadata and never infer a
standard name or canonical unit from a local variable key. Invalid non-strict
metadata returns a failed result instead of a conformance claim; strict mode
raises before mutation. Writer-side implicit canonical-unit conversion and its
second hardcoded unit table no longer exist. Spatial regridding requires an
explicit numerical method and explicit target-grid metadata; missing roles or
methods do not default to bilinear.

DataReader variable selection drops unselected data variables while preserving
coordinate variables, including source bounds on an extra bounds dimension.
An explicitly requested coordinate is recognized without being reported as a
data variable in `ReadResult.variables_read`. Explicit coordinate exclusions
remain effective; excluding declared time bounds raises in the existing time
ingress instead of synthesizing support or returning the unfiltered dataset.

Dataset-level CF conformance is evaluated only by the declared-scope check.
`IOHandler.read(...)` runs it on every loaded dataset: `IOResult` reports
`cf_validation_scope="declared_dataset_scope"`,
`dataset_conformance_checked=True`, the `cf_conformance_issues` found, and
`cf_compliant=True` only when the check ran and found no issue (the dataclass
refuses any other combination). Paths that do not run the check keep
`not_evaluated` and `cf_compliant=False`. `DataConfig`, public API requests,
CLI flags, scale audits, external adapter reports, stochastic descriptions, and
I/O health/file-info records still expose no positive dataset-conformance
switch; a `Conventions` attribute, conservative numerical method, successful
metadata synchronization, or absence of local validation errors cannot turn
those fields true. Evidence:
`test\coupling\external\common\test_cf_dataset_conformance.py` (real
netCDF4 products, raw-file cross-check, one negative per rule).

### 21.2 ContractRegistry -- State Variable Contracts (SCM)

File: `relic\contracts\contract_loader.py`

The `ContractRegistry` manages per-variable contracts that define ownership, aliases, flux whitelists, writer permissions, and conservation tolerances.

**Selected contract fields (`VariableContract`):**
| Field | Type | Description |
| ---------------------- | ----- | -------------------------------- |
| `variable` | str | Canonical variable name (one projection: the SCM `udunits`, the NodeRegistry port units from `SSOTCatalog`, and the StateManager default schema units must agree, "1" and "dimensionless" being the same unit; `test\unit\contracts\scm_ssot_projection.test.py` is the gate and no second schema registry exists) |
| `kind` | str | Variable kind: state, flux, forcing, diagnostic or parameter; a declared output returned with `write_mode=DERIVED` (never committed, visible to later Functors of the same group call) is `diagnostic` with no committed writer, for example `sector_emission_flux` |
| `udunits` | str | Unit specification (physical units, never a placeholder; stub contracts that said `dimensionless` for `temperature`, `wind_speed`, `gpp`, `net_assimilation`, `canopy_conductance`, `carbon_flux` were corrected on 2026-09-14: `temperature` and `wind_speed` are forcing in K and m s-1, `gpp` is a flux in umol m-2 s-1, `canopy_conductance` is m s-1; the `net_assimilation` and `carbon_flux` contracts were deleted on 2026-09-24 because no Functor reads or writes them as state; the `net_assimilation` alias entry and the generic `carbon_flux` entry were removed from `coupling.scales.defaults.variables` and `coupling.scales.daily.variables` as well: no Functor, contract or production lookup names `carbon_flux` (ScaleManager, HierarchicalAggregator, SpatialAggregation, the scale adapters and SSOTCatalog resolve variables only by the names their callers pass), and the tests that used it as a YAML area-density sample use the contracted area density `net_primary_productivity` (kg C m-2 s-1 per unit ground area) instead; `carbon_flux` remains a UnitConverter conversion category (the `carbon_flux_*` transforms), which is not a variable name) |
| `cf_standard_name` | str or None | Optional CF standard name (`dimension.cf_standard_name`); with `cell_methods` it forms the CF identity that this one variable holds in `SSOTCatalog` |
| `cell_methods` | str or None | CF-1.8 section 7.3 qualifier of `cf_standard_name` (`dimension.cell_methods`, for example `time: maximum`); a non-empty string or absent, validated at load |
| `owner` | str | Functor that owns this variable |
| `aliases` | list | Alternative names |
| `writers` | list | Functors allowed to write |
| `flux_in` / `flux_out` | list | Flux components for conservation |
| `atol` / `rtol` | float | Conservation tolerances |
| `layer_thickness_variable` | str or None | Optional state variable holding the thickness of the layer that this variable describes (for example `soil_depth` for `soil_moisture`); unit conversions that need a layer thickness read it pointwise from this variable |

**Selected API:**
| Method | Purpose |
| -------------------------------------------- | ------------------------------- |
| `registry.get_contract(variable)` | Get the complete variable contract |
| `registry.get_owner(variable)` | Get owning Functor |
| `registry.resolve_alias(name)` | Resolve alias to canonical name |
| `registry.is_alias(name)` | Check whether a name is an alias |
| `registry.is_incrementable(variable)` | Check whether increment semantics are declared |
| `registry.can_increment(variable, functor_name)` | Check increment permission |
| `registry.can_write(variable, functor_name)` | Check write permission |
| `registry.get_flux_whitelist(variable)` | Get permitted conservation fluxes |
| `registry.to_state_variable_spec(variable)` | Convert a contract to the StateManager specification (the contract kind goes to `spec.kind`; the storage role is `PARAMETER` for parameters and `GRID` otherwise; `numerics.lower_bound`/`upper_bound` become `spec.bounds` exactly as declared: `(lb, ub)`, `(lb, +inf)`, `(-inf, ub)`, or `None`, and `lower_bound > upper_bound` raises `ValueError`; `conserved` is true when the contract declares `accumulates`) |
| `registry.get_domain_members(domain)` | Get conservation domain members |
| `registry.list_variables()` | List all registered variables |
| `registry.list_domains()` | List conservation domains |

**StateManager integration:** `StateManager.__init__` loads the packaged
contracts by default through `relic.contracts.load_contract_registry()`, the
single contract-directory reader (`PACKAGED_CONTRACTS_DIR`,
`relic\contracts\variables\`). Loading is strict and cached once per resolved
directory per process (a failed load is not cached); the same registry is
shared read-only by its other readers (the transport bridge, `UnitSystem`,
`SSOTCatalog`). `UnitSystem` keeps no variable-unit table of its own:
`get_contract_unit` is a derived view of this registry, and
`get_standard_unit` falls back to its unit-category table (temperature,
pressure, carbon_flux, radiation, ...) only for names without a contract.
`UnitSystem.convert` is the variable-independent entry of the single converter
owner (`unit_converter.convert` with variable name `generic`); a
variable-dependent conversion such as the liquid-water-equivalent rule calls
`unit_converter.convert(value, variable_name, ...)` with the declared variable
name.
A missing directory raises `FileNotFoundError`, and an
unreadable, schema-invalid, or duplicate contract raises `ValueError` naming
the file. The only opt-out is explicit (`autoload_contract_registry=False`, or
`RELIC_AUTOLOAD_CONTRACT_REGISTRY` set to `0`, `false`, `no`, or `off`; the
default is `1`) and is logged at INFO. When contracts are loaded, the
canonical `apply_updates(...)` path enforces alias resolution,
ownership/writer authorization, and incrementability. `update_state(...)` is a
convenience/integration-test merge through `set_state(...)`, does no contract
checks, and is not the governed simulator commit path.

**Compile-time mutually exclusive owners:** The SCM schema intentionally does
not encode conditional owner expressions. When configuration selects exactly
one implementation from a closed set, the contract must use `owner: null` and
list only the real candidate Functor names in the existing `writers` field.
The existing graph or standard-bundle construction gate must reject unknown
choices and prove that exactly one candidate is instantiated. An observation
aggregator that only projects dependency outputs must never be declared as the
state owner. Optional externally supplied state with no production writer uses
`owner: null`, empty writer lists, `incrementable: false`, and explicit
provenance identifying the state as externally supplied. This is a composition
of the existing contract and build mechanisms, not a second ownership service.

Active contracts covered by an architecture audit must not retain a
`REVIEW_NEEDED` marker. A contract that names a Functor which does not write the
declared variable is an audit failure even if the YAML schema is syntactically
valid.

**Forbidden:**

- Bypassing contract ownership checks when writing state variables.
- Defining variable contracts outside `relic\contracts\`.
- Encoding a diagnostic-only aggregator as the owner of prognostic state.
- Adding a conditional-owner registry, schema variant, or runtime selector in
  place of the existing compile-time graph or bundle selection.

### 21.3 Time Constants SSOT

File: `relic\utils\time_constants.py`

Canonical time values are resolved through ParameterManager. A derived value is
used only when its source entry is absent; it is computed from already resolved
values as shown below. A present entry that fails unit or type validation
raises. Scientific Functors must not define their own conversion constants.

**Available constants:**
| Constant | Source key | Derived when absent |
| -------------------- | ------------------------------------------------------------ | ------------------------------------ |
| `SECONDS_PER_DAY` | `relic.physical.constants.time.seconds_per_day` | None (required) |
| `SECONDS_PER_YEAR` | `relic.physical.constants.time.seconds_per_year` | None (required) |
| `SECONDS_PER_HOUR` | `relic.physical.constants.time.seconds_per_hour` | `SECONDS_PER_DAY / 24` |
| `SECONDS_PER_MINUTE` | `relic.physical.constants.time.seconds_per_minute` | None (required) |
| `DAYS_PER_YEAR` | `coupling.scales.temporal.calendars.gregorian.days_per_year` | `SECONDS_PER_YEAR / SECONDS_PER_DAY` |

**Access pattern:** Module-level `__getattr__` with lazy loading and caching.

```python
from relic.utils.time_constants import SECONDS_PER_DAY
```

**Forbidden:** Hardcoding `86400`, `365.25`, `3600`, or any time conversion factor in scientific modules or Functors.

### 21.4 Infrastructure State and Constraint Ownership

RELIC has one StateManager class and one PhysicalConstraintManager class, but
their lifecycle scopes are distinct. A lifecycle facade is not permission to
create a second state or constraint authority:

| Owner | Entry Point | Scope |
| --- | --- | --- |
| StateManager | `RELICSimulator.state_manager` or explicit constructor injection; no module-global facade | simulation state and transactional writes |
| PhysicalConstraintManager | `StateManager.physical_constraint_manager` | one cached or injected constraint owner per StateManager; no module-global singleton API |

Scientific Functors use their `state` argument and the
`StateManager.physical_constraint_manager` integration path. They must not
replace these owners with module-local global state.

**Forbidden in Functor code:**
| Forbidden Pattern | Why |
| ------------------------------------------ | ------------------------------------------------------- |
| `get_global_manager()` | Bypasses Functor namespace resolution |
| removed `get_global_state_manager()` API | State is passed via `forward(state)` or explicit infrastructure injection |
| removed `get_global_physical_constraint_manager()` API | Use `StateManager.physical_constraint_manager` property |
