# Mathematical Foundations, State, Units, and Conservation

## Architectural contract and notation

The Recomposable Ecosystem Library for Integrated Coupling (RELIC) represents a
terrestrial biogeochemical model as a directed composition of typed scientific
Functors under shared parameter and state governance (*Methods of Ecological
Analysis*, Chapters 1--5, 10, 11, and 14). Each concrete Functor maps declared
read-only state and external forcing tensors to typed update records without
mutating shared storage in place:

$$
\mathcal{F}_{\boldsymbol{\theta}_i}:
(\mathbf{S}_t,\; \boldsymbol{\xi}_t,\; t,\; \Delta t)
\longmapsto
\{\Delta \mathbf{S}_i,\; \mathbf{q}_i,\; \mathbf{d}_i\},
$$

where $\mathbf{S}_t$ is the prognostic state vector, $\boldsymbol{\xi}_t$ is
external forcing, $\boldsymbol{\theta}_i$ is the `ParameterManager`-owned
parameter vector, $\mathbf{q}_i$ denotes flux or tendency outputs, and
$\mathbf{d}_i$ denotes diagnostic observations. `RELICSimulator`
(`relic.core.simulator`) governs topological execution and multi-rate
scheduling, while `StateManager` (`relic.core.state.manager`) executes unit
validation, bounds accounting, elemental conservation ledgers, and atomic
commits.

| Symbol | Meaning | Canonical SI units |
| --- | --- | --- |
| $\mathbf{S}$, $x$ | Prognostic reservoir or intensive state vector | variable-specific |
| $\boldsymbol{\xi}$, $f$ | Externally supplied meteorological or boundary forcing | variable-specific |
| $\boldsymbol{\theta}$ | Versioned model parameter vector | parameter-specific |
| $q$, $\mathbf{F}$ | Scalar transfer rate or vector flux density | storage unit s$^{-1}$, or storage unit m$^{-2}$ s$^{-1}$ |
| $r$ | Stoichiometric reaction or process-rate vector | mol m$^{-3}$ s$^{-1}$, kg m$^{-2}$ s$^{-1}$, or declared equivalent |
| $N$ | Stoichiometric or output-incidence matrix | state unit per reaction extent |
| $C$ | Linear elemental conservation-ledger matrix | element unit per state unit |
| $u$, $y$ | Conjugate boundary-port input and output | effort/flow product unit |
| $J$, $L$ | Reversible skew-symmetric interconnection or Poisson operator | operator-specific |
| $R$, $M$ | Irreversible symmetric positive-semidefinite resistance or mobility operator | operator-specific |
| $E$, $S$, $H$ | Energy, entropy, and storage or free-energy functionals | J, J K$^{-1}$, and declared storage unit |
| $\Delta t$ | Effective execution step duration | s |

Units are resolved from state variable contracts (`relic.contracts`),
certificate unit maps, and `ParameterManager` YAML metadata. A literal
`dimensionless` quantity carries unit $1$; an omitted unit fails validation and
never defaults to dimensionless.

## Four scientific abstractions and Functor lifecycle

Every concrete scientific module implements one of four abstract contracts in
`relic.core.abstractions` and enters the runtime lifecycle through its matching
canonical bridge in `relic.core.functor.functor_bridges`, inheriting from
`FunctorBase` (`relic.core.functor.functor_base`, a subclass of
`torch.nn.Module`):

| Abstract contract (`relic.core.abstractions`) | Canonical bridge (`relic.core.functor.functor_bridges`) | `ProcessCategory` (`relic.core.functor.functor_types`) | Governing mathematical mapping | Representative spheres and processes |
| --- | --- | --- | --- | --- |
| `PhysicalField` | `PhysicalFieldFunctor` | `PHYSICAL_FIELD` | $\phi: \Omega \times \mathcal{T} \rightarrow \mathcal{V}$ | Shortwave/longwave radiation, vapour pressure deficit, snowpack, snow albedo, groundwater head, gas-phase photochemistry |
| `BiologicalProcess` | `BiologicalProcessFunctor` | `BIOLOGICAL_PROCESS` | $\frac{\mathrm{d}x}{\mathrm{d}t} = N\,r_{\mathrm{bio}}(x, \boldsymbol{\xi}; \boldsymbol{\theta})$ | Photosynthesis, stomatal conductance, respiration, phenology, allocation, BVOC synthesis, litter/SOM decomposition, N/CH$_4$ kinetics |
| `TransportMechanism` | `TransportMechanismFunctor` | `TRANSPORT_MECHANISM` | $\frac{\mathrm{d}x}{\mathrm{d}t} = -D_{\mathrm{div}}\mathbf{F} + q_{\mathrm{bnd}}$ | Advection-diffusion, Monin-Obukhov turbulence, dry deposition, evapotranspiration, Richards moisture flow, runoff, solute leaching, managed transfers |
| `ScaleCoupling` | `ScaleCouplingFunctor` | `SCALE_COUPLING` | $y = \mathcal{M}_{a \rightarrow b}(x)$ | Conservative spatial remapping, temporal aggregation, emission sector/species mapping, litterfall and hydrological interface mappers |

Across the live repository, `90` unique concrete scientific `Functor` classes
are registered across these four bridge families (`37`
`BiologicalProcessFunctor`, `27` `TransportMechanismFunctor`, `14`
`PhysicalFieldFunctor`, and `12` `ScaleCouplingFunctor`; `0` uncertified).

### Variable roles and scheduling metadata

`FunctorMetadata` (`relic.core.functor.functor_base`) declares
`input_variables`, `output_variables`, `spatial_scale` (`SpatialScale`),
`temporal_scale` (`TemporalScale`), `variable_kind` (`VariableKind`),
`execution_phase` (`ExecutionPhase`), and `sub_flow_accuracy`
(`SubFlowAccuracy`). Six `VariableKind` enumerators classify tensor I/O
(`STATE`, `FORCING`, `PARAMETER`, `FLUX`, `DIAGNOSTIC`, plus the `UNKNOWN`
sentinel for undeclared roles):

| Role (`VariableKind`) | Runtime meaning | Governed write semantics |
| --- | --- | --- |
| `STATE` | Prognostic reservoir or field carried across steps | `StateManager` transactional `UpdateItem` |
| `FORCING` | Externally supplied meteorological or boundary driver | Read-only input; unmodified by consuming Functors |
| `PARAMETER` | Versioned coefficient, prior, or material property | Resolved via `ParameterManager` (`self.get_parameter_value`) |
| `FLUX` | Rate of exchange, transport, or stoichiometric transformation | Applied as a rate tendency (`per_timestep=False`), step increment, or typed port |
| `DIAGNOSTIC` | Derived observable without prognostic storage ownership | Partitioned to diagnostic channel (`StateWriteMode.DERIVED`) |

`spatial_scale`, `temporal_scale`, and `variable_kind` apply to the Functor as
a whole: `spatial_scale` drives label-level compatibility checks,
`variable_kind` drives single-writer overlap checks
(`relic.core.simulator.output_overlap_governance`), and `temporal_scale`
assigns default `fast`, `medium`, or `slow` execution groups when neither
`execution_group` nor `characteristic_time_days` is specified (any explicit
`process_groups` entry must match or registration raises `ValueError`).

## Spatial quantity basis and temporal overlap laws

Scale labels do not define cell geometry, cell areas, or extensivity. Basis-aware
spatial operators (`ConservativeRemap`, `ConservativeRegridder`,
`SpatialAggregation`, `HierarchicalAggregator`, `ScaleManager`, simulator
auto-regridding, and external-forcing adapters) require an explicit
`SpatialQuantityBasis` (`relic.coupling.scales.types`) via `spatial_basis`
metadata or the `quantity_basis` argument (*Methods of Ecological Analysis*,
Chapters 2 and 14):

| `SpatialQuantityBasis` | Value on cell $i$ | Conserved cell amount | Cross-scale reduction rule |
| --- | --- | --- | --- |
| `intensive` | Intensive state $x_i$ (e.g., K, mol mol$^{-1}$) | Undefined | Area- or overlap-weighted mean |
| `cell_total` | Extensive cell amount $X_i$ (e.g., kg) | $X_i$ | Direct sum $\sum_i X_i$ or overlap-fraction partition |
| `area_density` | Areal density $x_i$ (e.g., kg m$^{-2}$, W m$^{-2}$) | $X_i = x_i A_i$ ($A_i > 0$ in m$^2$) | Area-weighted integral divided by target area $A_J$ |
| `categorical` | Discrete class code $c_i$ | Undefined; arithmetic forbidden | Area-weighted mode (ties broken toward smallest code) or nearest neighbour |

Bare `"extensive"` is rejected with `ValueError` because it fails to distinguish
per-cell totals from areal densities. Exact conversion between `cell_total` and
`area_density` (`convert_spatial_basis`) requires strictly positive cell areas
$A_i > 0$ ($X_i = x_i A_i$); conversions involving `intensive` or `categorical`
raise `ValueError`. Note that `SpatialQuantityBasis` governs spatial support
transfer and is distinct from `CochainSpaceSpec.extensivity` in
`relic.computation.operators.dec.forms`.

For first-order overlap remapping with source-target overlap area $A_{iJ} \ge 0$,
source area $A_i > 0$, target area $A_J > 0$, covered target area
$C_J = \sum_i A_{iJ}$, and covered source fraction
$f_i = \sum_J A_{iJ}/A_i \le 1$, `ConservativeRemap` and `ConservativeRegridder`
evaluate:

$$
Y_J = \sum_i X_i\,\frac{A_{iJ}}{A_i} \quad (\texttt{cell\_total}),
\qquad
y_J = \frac{1}{A_J}\sum_i x_i A_{iJ} \quad (\texttt{area\_density}),
\qquad
y_J = \frac{1}{C_J}\sum_i x_i A_{iJ} \quad (\texttt{intensive}).
$$

Amounts are conserved strictly over covered support:

$$
\sum_J Y_J = \sum_i f_i X_i,
\qquad
U = \sum_i (1 - f_i)\,X_i,
$$

where uncovered source mass $U$ is reported in diagnostics and never rescaled
onto covered target cells. Explicit cell bounds, overlap matrices, or polygon
vertices are mandatory; bounds are never inferred from coordinate spacing.
External-model ingestion reads Climate and Forecast (CF) bounds for 1D
rectilinear coordinates and rejects 2D curvilinear grids lacking explicit cell
corners during conservative remapping (`SpatialScaling.regrid` with
`method="conservative"` on a coordinate-free grid descriptor switches to
non-conservative nearest-neighbour lookup).

In time, `TemporalAggregator` and `TemporalDisaggregator`
(`relic.coupling.scales.temporal`) require explicit `temporal_semantics`
(`interval_mean`, `rate`, `interval_accumulation`, or `state_at_boundary`;
`point_sample` and `categorical` are rejected) together with explicit source and
target units. Over target interval $j$ of duration $\tau_j$ with source-target
overlap durations $O_{ij}$ and validity mask $m_i \in \{0, 1\}$, `interval_mean`
and rate-to-rate mappings evaluate:

$$
\bar{x}_j = \frac{\sum_i m_i O_{ij}\,x_i}{\sum_i m_i O_{ij}},
\qquad
c_j = \frac{1}{\tau_j}\sum_i m_i O_{ij},
$$

returning $\bar{x}_j$ when dimensionless coverage $c_j$ meets
`coupling.scales.temporal.coverage_threshold` and `NaN` below threshold without
extrapolating missing intervals. `TemporalScaling` maps fixed SI durations via
`method` (`mean`, `rate`, `sum`, `total`, `extensive`, or law names) and rejects
calendar-dependent `month` or `year` intervals.

## Universal conservation and balance laws

### Local, stoichiometric, and control-volume balances

For a conserved scalar density $c(\mathbf{x}, t)$ with velocity $\mathbf{v}$,
diffusivity $D$, internal stoichiometric network $N r(c, \boldsymbol{\theta}, \boldsymbol{\xi})$,
and open-boundary exchange $s_{\mathrm{bnd}}$, the continuum and integral
control-volume balances on domain $\Omega$ with outward normal $\mathbf{n}$ are
(*Methods of Ecological Analysis*, Chapters 3--5):

$$
\frac{\partial c}{\partial t} + \nabla \cdot \left(\mathbf{v}c - D\nabla c\right)
= N r + s_{\mathrm{bnd}},
\qquad
\frac{\mathrm{d}}{\mathrm{d}t}\int_{\Omega} c\,\mathrm{d}V
= -\int_{\partial\Omega} \mathbf{F}\cdot\mathbf{n}\,\mathrm{d}A
+ \int_{\Omega} N r\,\mathrm{d}V
+ \int_{\Omega} s_{\mathrm{bnd}}\,\mathrm{d}V.
$$

After spatial discretization into prognostic reservoir vector
$\mathbf{S}(t) \in \mathbb{R}^n$ with internal process stoichiometries
$\{\mathbf{N}_p\}_{p \in \mathcal{P}_{\mathrm{int}}}$ and boundary port
operators $\{\mathbf{B}_q\}_{q \in \mathcal{P}_{\mathrm{ext}}}$, the
semi-discrete coupled evolution is:

$$
\frac{\mathrm{d}\mathbf{S}}{\mathrm{d}t}
= \sum_{p \in \mathcal{P}_{\mathrm{int}}}
\mathbf{N}_p\,\mathbf{J}_p(\mathbf{S}, \boldsymbol{\xi}, \boldsymbol{\theta}, t)
+ \sum_{q \in \mathcal{P}_{\mathrm{ext}}}
\mathbf{B}_q\,\mathbf{u}_q(\mathbf{S}, \boldsymbol{\xi}, \boldsymbol{\theta}, t).
$$

For any conserved elemental ledger matrix $C$ (rows encoding water, carbon,
nitrogen, phosphorus, or charge content per state unit), internal closure
requires the stoichiometric nullspace condition:

$$
C \mathbf{N}_p = 0 \quad \forall p \in \mathcal{P}_{\mathrm{int}}
\qquad \Longrightarrow \qquad
\frac{\mathrm{d}}{\mathrm{d}t}(C\mathbf{S})
= \sum_{q \in \mathcal{P}_{\mathrm{ext}}} C\mathbf{B}_q\,\mathbf{u}_q.
$$

Non-zero changes in $C\mathbf{S}$ must traverse an explicit port, boundary
export, managed export, diagnostic residual, or observation route.

### Energy and entropy balances

When a process declares thermodynamic energy $E(\mathbf{S})$, entropy
$S(\mathbf{S})$, and conjugate port effort-flow pairs $(e_{\mathrm{port}}, f_{\mathrm{port}})$
with port power $P_{\mathrm{port}} = e_{\mathrm{port}}^{\mathsf{T}} f_{\mathrm{port}}$,
the first- and second-law balances take the form:

$$
\frac{\mathrm{d}E}{\mathrm{d}t} = P_{\mathrm{in}} - P_{\mathrm{out}} - D_E,
\quad D_E \ge 0,
\qquad
\frac{\mathrm{d}S}{\mathrm{d}t} = \Phi_S + \sigma,
\quad \sigma \ge 0,
$$

where $D_E$ is internal mechanical/thermal dissipation, $\Phi_S$ is boundary
entropy flux, and $\sigma$ is internal entropy production. Empirical rate laws
without declared potentials and operators are audited through mass/energy budget
ledgers at certificate Levels 1--3 without fabricating synthetic thermodynamic
potentials.

## Transactional StateManager governance (`I1`--`I7`)

`partition_outputs` (`relic.core.state.types`) splits every Functor return
mapping into state updates (`UpdateItem`), diagnostics, and immutable
`ExternalIntent` records according to `UpdateMode` (`ACCUMULATE`, `REPLACE`,
`FLUX`) and `StateWriteMode` (`MUTABLE`, `IMMUTABLE`, `DERIVED`):

$$
\mathbf{S}_{n+1} = \mathbf{S}_n + \Delta t\,\mathbf{q}
\quad (\texttt{ACCUMULATE}/\texttt{FLUX},\; \texttt{per\_timestep=False}),
\qquad
\mathbf{S}_{n+1} = \mathbf{S}_n + \Delta\mathbf{S}
\quad (\texttt{ACCUMULATE},\; \texttt{per\_timestep=True}),
\qquad
\mathbf{S}_{n+1} = \mathbf{S}_{\mathrm{new}}
\quad (\texttt{REPLACE}).
$$

All Functor-to-`StateManager` commits enforce seven runtime invariants (`I1`--`I7`,
*Methods of Ecological Analysis*, Chapter 11):

| ID | Invariant | Mathematical and runtime contract | Enforcement owner |
| --- | --- | --- | --- |
| `I1` | No side effects | `Functor.forward()` operates on read-only tensor views; prognostic mutations occur exclusively via returned `UpdateItem` records | `FunctorBase`, `StateManager` |
| `I2` | Unit consistency | Every `UpdateItem.unit` is dimensionally compatible with the target state contract (`per_timestep=False` rates carry time-derivative units) | `relic.core.units.validator`, `StateManager` |
| `I3` | Conservation closure | Five active ledgers (`water`, `carbon`, `nitrogen`, `phosphorus`, `energy`) satisfy $\lvert r_n^{(E)} \rvert \le \epsilon_E$ over $[t_n, t_n + \Delta t]$ | `relic.core.state.conservation` |
| `I4` | Additive reversibility | Applying $+\delta\mathbf{S}$ followed by $-\delta\mathbf{S}$ under `UpdateMode.ACCUMULATE` restores $\mathbf{S}_n$ to floating-point precision | `StateManager` |
| `I5` | Commutativity of independent increments | Synchronous `ACCUMULATE` increments to disjoint or additive pools commute up to floating-point rounding (`output_overlap_governance` guards shared writers) | `StateManager`, `relic.core.simulator.output_overlap_governance` |
| `I6` | Explicit $\Delta t$-scaling | Rate tendencies (`per_timestep=False`) scale once by committed interval $\Delta t$; integrated increments (`per_timestep=True`) apply without a second $\Delta t$ factor | `UpdateItem`, `StateManager` |
| `I7` | Accounted bounds clipping | Box-bound projection $\widetilde{S}_i = \min(\max(S_i, S_{i,\min}), S_{i,\max})$ records exact truncated mass/energy in `clip_loss` or explicit overflow routes | `StateManager`, `relic.validation.framework.rules.overflow_routing` |

Discrete conservation residuals audited by
`StateManager.generate_conservation_report()` across `validation/conservation.yaml`
tolerances ($\epsilon_{\mathrm{H_2O}} = 10^{-9}$, $\epsilon_{\mathrm{C}} = 10^{-9}$,
$\epsilon_{\mathrm{N}} = 10^{-11}$, $\epsilon_{\mathrm{P}} = 10^{-12}$,
$\epsilon_{\mathrm{E}} = 10^{-6}$) satisfy:

$$
r_n^{(E)}
= \sum_{i \in \mathcal{V}_E}\left(Q_{i,n+1}^{(E)} - Q_{i,n}^{(E)}\right)
- \Delta t\left(\sum_{j \in \mathcal{S}_{\mathrm{ext}}(E)} F_{j,n}^{(E)}
- \sum_{k \in \mathcal{L}_{\mathrm{ext}}(E)} F_{k,n}^{(E)}\right)
+ Q_{\mathrm{clip},n}^{(E)}.
$$

## Centralized ParameterManager and physical-constant SSOT

`ParameterManager` (`relic.parameters.manager`) owns all model coefficients,
bounds, units, dtypes, references, and gradient flags. In the live database
under `relic/parameters/database/`, `138` YAML files (`134` under `relic/` and
`4` under `calibrated/`) define `4,751` leaf entries (`4,642` core + `109`
calibrated overrides), resolving to `4,643` active default runtime parameters of
which `810` carry `with_gradients: true` for differentiable calibration. Inside
a Functor, parameters are retrieved exclusively via
`self.get_parameter_value(name, required=...)`.

Universal and material physical constants (CODATA gas constant $R$,
Stefan-Boltzmann constant $\sigma$, Avogadro constant $N_A$, standard gravity
$g$, von Karman constant $\kappa$, molar masses, and phase-change enthalpies)
reside solely in `relic/parameters/database/relic/physical/constants.yaml` as a
Single Source of Truth (SSOT) and are accessed via `get_physical_constant`
(`relic.utils.physical_constants`). Domain YAML files and Python modules never
redefine physical constants. Dimensional conversions preserve exact stoichiometry;
converting molar assimilation $A$ [$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$] to carbon
mass flux $F_{\mathrm{C}}$ [kg C m$^{-2}$ s$^{-1}$] applies
$F_{\mathrm{C}} = A\,M_{\mathrm{C}}\,10^{-9}$ using SSOT carbon molar mass
$M_{\mathrm{C}}$ [g mol$^{-1}$].

## Piecewise-smooth discrete adjoint differentiation and device policy

Because `FunctorBase` subclasses `torch.nn.Module`, macro-step transitions
$\mathbf{S}_t = \Phi_t(\mathbf{S}_{t-1}, \boldsymbol{\theta})$ ($t = 1, \dots, N_t$)
from initial state $\mathbf{S}_0(\boldsymbol{\theta})$ construct an exact
reverse-mode discrete adjoint graph in `torch.autograd` (*Methods of Ecological
Analysis*, Chapter 10). Because $\boldsymbol{\theta}$ enters every step
transition $\Phi_t$ directly, the total terminal objective gradient
$\frac{\mathrm{d}\mathcal{L}(\mathbf{S}_{N_t}, \boldsymbol{\theta})}{\mathrm{d}\boldsymbol{\theta}}$
accumulates step-wise parameter Jacobians across the full trajectory:

$$
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\theta}}
= \frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}
+ \frac{\partial \mathcal{L}}{\partial \mathbf{S}_{N_t}}
\left(\prod_{k=1}^{N_t} \frac{\partial \Phi_k}{\partial \mathbf{S}_{k-1}}\right)
\frac{\partial \mathbf{S}_0}{\partial \boldsymbol{\theta}}
+ \sum_{t=1}^{N_t}
\frac{\partial \mathcal{L}}{\partial \mathbf{S}_{N_t}}
\left(\prod_{k=t+1}^{N_t} \frac{\partial \Phi_k}{\partial \mathbf{S}_{k-1}}\right)
\frac{\partial \Phi_t}{\partial \boldsymbol{\theta}},
$$

where matrix products are ordered from left ($k = N_t$) to right ($k = 1$ or
$k = t + 1$) and the empty product at $t = N_t$ equals the identity matrix
$\mathbf{I}_n$. Smooth PyTorch tensor paths, linear overlap remaps without
post-clipping, and implicit scalar roots solved via `BracketedBisectionSolver`
(using implicit-function differentiation) propagate exact discrete sensitivities.
Transversal guard crossings require explicit saltation-matrix factors
(`relic.computation.solvers.hybrid_events`; see [Numerical Integration, Splitting, Stability, and Stochastic Processes](numerics.md)), whereas hard bounds
clipping, discrete regime switches, adaptive step-size rejections, and NumPy-based
external ingestion regridding act as non-differentiable boundaries.

Device placement is governed centrally by `relic.core.device_policy` (`device`,
`require_gpu`, `device_overrides`, `module_device_overrides`). `FunctorBase.__init__`
allocates CPU placeholders, and simulator registration migrates modules to the
resolved target device (GPU-first with automatic CPU fallback unless
`require_gpu: true` or an unsatisfied `enable_distributed: true` contract
raises). Scientific modules never invoke `torch.cuda.is_available()` directly.

## Bounded mathematical contracts and runtime boundary (`[I]` vs `[N]`)

Three decidable finite-dimensional operator contracts in
`relic.computation.operators` provide numerical verification against
`validation/iphs.yaml` tolerances (*Methods of Ecological Analysis*, Chapters
1--3 and 10):

1. **Typed linear chain maps (`[I]`)**: `LinearChainMap`
   (`relic.computation.operators.dec.chain_map`) maps declared 10-field
   `CochainSpaceSpec` spaces, verifies unit-scaled ledger preservation
   $\lVert C_t(u M) - C_s \rVert_\infty \le \varepsilon_{\mathrm{cm}}$, and
   bounds perturbation growth by $\lVert u M \rVert_2 \lVert \delta x \rVert_2$.
   Exact `equals` is an equivalence relation; `close_within`
   ($\lVert u_1 M_1 - u_2 M_2 \rVert_2 \le \varepsilon$) is reflexive and
   symmetric without transitivity. Production consumer:
   `LitterfallInterfaceMapper`.
2. **Metzler resource-activity invasion threshold (`[I]`)**:
   `activity_threshold` and `is_nonsingular_m_matrix`
   (`relic.computation.operators.invasion_threshold`) verify non-negative growth
   $G \ge 0$ and nonsingular M-matrix loss $\Delta$ ($Z$-matrix sign structure
   and $\Delta^{-1} \ge 0$), evaluate the next-generation spectral radius
   $\mathcal{R}_{\mathrm{act}} = \rho(\Delta^{-1} G)$, and cross-check
   $\operatorname{sign}(\mathcal{R}_{\mathrm{act}} - 1) = \operatorname{sign}(s(G - \Delta))$.
   Values inside $\lvert \mathcal{R}_{\mathrm{act}} - 1 \rvert \le \varepsilon_{\mathrm{crit}}$
   return `verdict="critical"` and `hyperbolic=False` without claiming global
   bifurcation or persistence. Production consumer: linearized activity-free
   `MicrobialEnzymeDecomposition`.
3. **Affine-leaf relative storage and passivity ledger (`[I]`)**:
   `relative_storage`, `certify_relative_storage`, and `feedback_power_ledger`
   (`relic.computation.operators.relative_storage`) retain the full Bregman
   first-order term
   $V(x) = F(x) - F(x^\ast) - \nabla F(x^\ast)^{\mathsf{T}}(x - x^\ast)$,
   verify constrained stationarity $\nabla F(x^\ast) \in \operatorname{im}(C^{\mathsf{T}})$
   and projected Hessian positivity $Z^{\mathsf{T}}\nabla^2 F(x^\ast) Z \succ 0$
   on an SVD basis $Z$ of $\ker C$ in `float64`, and audit open-system Willems
   power balance $\dot{F} = y_F^{\mathsf{T}} u - \mathcal{D} + \Phi_F$. Curved
   nonlinear conservation manifolds lie outside this contract. Production
   consumer: closed 1D `FickianDiffusion`.

| Architectural or theoretical element | Status | Scope boundary |
| --- | --- | --- |
| Four-pillar ABCs, canonical bridges, `FunctorBase`, `ParameterManager` (`4,643` parameters), `StateManager` (`I1`--`I7`), `GraphIR`, `SpatialQuantityBasis`, and bounded operator contracts | `[I]` | Live runtime governance across `relic.core`, `relic.parameters`, `relic.orchestration`, `relic.coupling`, and `relic.computation` |
| Category-theoretic specification (`relic.core.category_theory`) | `[M]` | Standalone design-specification module; not imported by `GraphIR` compilation or `execution_loop`, with no compiler identity-elision pass |
| Multi-isotope prognostic pool ledger (`IsotopeHeartbeatFunctor`) | `[N]` | Fail-closed `runtime_unbound` Level 0 boundary (historical design context `[H]`); excluded from the five active default conservation ledgers |
| Textbook structural six-tuple $\mathfrak{S} = (\mathcal{K}, \mathcal{C}^\bullet, \mathcal{H}, \mathcal{J}-\mathcal{R}, \mathcal{G}, \mathcal{C}_{\mathrm{ledger}})$ | `[N]` | Analytical synthesis in *Methods of Ecological Analysis* (Chapters 1, 21--24) realized across distributed DEC/IPHS modules without a monolithic wrapper class |
| Five-term total simulation bias decomposition $\lVert \widehat{\mathbf{y}}_T - \mathbf{y}_T^\star \rVert \le E_1^{\mathrm{model}} + E_2^{\mathrm{param}} + E_3^{\mathrm{spatial}} + E_4^{\mathrm{temporal/split}} + E_5^{\mathrm{obs}}$ | `[N]` | Analytical telescoping error bound (*Methods of Ecological Analysis*, Chapters 14, 25--30); no online runtime `E1`--`E5` decomposition estimator is implemented |

## References

- Berman, A. and Plemmons, R. J. (1994). *Nonnegative Matrices in the Mathematical Sciences*. SIAM.
- Bonan, G. B. (2019). *Climate Change and Terrestrial Ecosystem Modeling*. Cambridge University Press.
- de Groot, S. R. and Mazur, P. (1984). *Non-Equilibrium Thermodynamics*. Dover.
- Farquhar, G. D., von Caemmerer, S., and Berry, J. A. (1980). A biochemical model of photosynthetic CO$_2$ assimilation in leaves of C$_3$ species. *Planta* 149, 78--90.
- Griewank, A. and Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed. SIAM.
- Hairer, E., Lubich, C., and Wanner, G. (2006). *Geometric Numerical Integration*, 2nd ed. Springer.
- Jones, P. W. (1999). First- and second-order conservative remapping schemes for grids in spherical coordinates. *Monthly Weather Review* 127, 2204--2210.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- Richards, L. A. (1931). Capillary conduction of liquids through porous mediums. *Physics* 1, 318--333.
- Willems, J. C. (1972). Dissipative dynamical systems, Part I: General theory. *Archive for Rational Mechanics and Analysis* 45, 321--351.
