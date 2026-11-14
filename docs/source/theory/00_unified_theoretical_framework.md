# Unified Theoretical Framework for RELIC

## 1. Process Definition and Scope

RELIC formulates terrestrial and interfacial Earth-system dynamics as a composable, differentiable graph of typed process operators acting on conserved material and energy reservoirs. Across the thirty-chapter mathematical framework of *Methods of Ecological Analysis* (Chapters 1--30), biogeochemical, hydrological, radiative, atmospheric, and anthropogenic dynamics are organized around three architectural contracts:

1. **Four-Pillar Scientific Abstraction (`ProcessCategory`)**: Every scientific module implements one of four abstract contracts in `relic/core/abstractions.py` (`PhysicalField`, `BiologicalProcess`, `TransportMechanism`, or `ScaleCoupling`) and enters the runtime lifecycle through its corresponding bridge class (`PhysicalFieldFunctor`, `BiologicalProcessFunctor`, `TransportMechanismFunctor`, or `ScaleCouplingFunctor` in `relic/core/functor/`) inheriting from `FunctorBase(ABC, torch.nn.Module)`.
2. **Transactional Reservoir-Flux State Governance (`StateManager`)**: Process `forward()` calls operate functionally on read-only tensor views and emit declarative output mappings partitioned by `partition_outputs()` (`relic/core/state/types.py`) into state updates (`UpdateItem`), diagnostics, and immutable `ExternalIntent` descriptors. All prognostic mutations pass through `StateManager` unit validation, bounds accounting, and elemental conservation ledgers.
3. **Centralized Versioned Parameter and Constant Ownership (`ParameterManager`)**: Process parameters, bounds, units, and priors reside in YAML databases under `relic/parameters/database/relic/`, while universal and material constants reside exclusively in `relic/parameters/database/relic/physical/constants.yaml` and are retrieved via `get_physical_constant()`.

### 1.1 Five Interacting Earth-System Spheres

RELIC partitions domain processes across five Earth-system spheres connected by explicit interfacial flux variables:

| Sphere | Physical Domain | Representative Prognostic States | Primary Process Families |
| :--- | :--- | :--- | :--- |
| **Atmosphere** | Surface boundary layer and air column | $T_{\mathrm{air}}$, $q_v$, $[\mathrm{O}_3]$, $[\mathrm{NO}_x]$, BVOC mixing ratios | Shortwave/longwave radiation, Monin-Obukhov turbulence, photochemistry, canopy transport |
| **Biosphere** | Leaf, canopy, root, and stand cohorts | $C_{\mathrm{leaf}}$, $C_{\mathrm{stem}}$, $C_{\mathrm{root}}$, $\mathrm{LAI}$, phenological phase | C3/C4 photosynthesis, stomatal conductance, autotrophic respiration, allocation, mortality, BVOC emission |
| **Hydrosphere** | Canopy interception, snowpack, vadose zone, runoff | $\theta_{\mathrm{soil}}$, $\mathrm{SWE}$, canopy water storage, ponded depth | Infiltration, Penman-Monteith and Priestley-Taylor evapotranspiration, Richards flow, freeze-thaw, baseflow |
| **Pedosphere** | Multi-layer mineral and organic soil profile | $C_{\mathrm{SOM}}$, $N_{\min}$, $P_{\mathrm{avail}}$, microbial biomass, $T_{\mathrm{soil}}$ | Litter/SOM decomposition, enzymatic depolymerization, mineralization-immobilization, nitrification-denitrification, sorption |
| **Anthroposphere** | Managed land tiles and disturbance regimes | Land-use fractions, harvest pools, management schedules | Cropland harvest, irrigation, fertilizer application, tillage, prescribed fire |

| Source Sphere \ Target Sphere | Atmosphere | Biosphere | Hydrosphere | Pedosphere | Anthroposphere |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Atmosphere** | Internal mixing and chemistry | $R_{\mathrm{net}}$, $[\mathrm{CO}_2]$, $T_{\mathrm{air}}$, VPD, $\mathrm{O}_3$ uptake | Precipitation $P$, atmospheric evaporative demand | Surface radiative/convective forcing, N/P deposition | Urban/agricultural exposure boundary |
| **Biosphere** | $A_{\mathrm{net}}$, $R_{\mathrm{auto}}$, BVOCs, canopy roughness | Stand competition and allocation | Transpiration, canopy interception loss | Litterfall, root exudates, root turnover | Harvestable biomass yield |
| **Hydrosphere** | Latent heat $LE$, sensible heat $H$ | Root-zone matric potential $\psi_{\mathrm{soil}}$ | Lateral/vertical water routing | Volumetric moisture $\theta$, advective solute leaching | Irrigation withdrawal demand |
| **Pedosphere** | $R_{\mathrm{hetero}}$, $\mathrm{CH}_4$, $\mathrm{N}_2\mathrm{O}$, $\mathrm{NO}$, $\mathrm{NH}_3$ | Plant mineral N and P uptake | Pore-space hydraulic impedance | Bioturbation and vertical mixing | Soil fertility and carbon stock accounting |
| **Anthroposphere** | Direct emissions and land-cover albedo shift | Sowing, harvest, grazing, thinning | Irrigation supply, drainage modification | Fertilizer input, residue incorporation, tillage | Socio-ecological management feedback |

### 1.2 Scope of Book-Level Syntheses vs Runtime Software Owners

This reference page unifies the live software architecture with the analytical synthesis chapters of *Methods of Ecological Analysis*:

- **Implemented Runtime Contracts (`[I]`)**: Four-pillar Functor bridges, `ParameterManager` YAML resolution, `StateManager` seven-invariant (`I1`--`I7`) transactional updates and five-domain conservation reports (`water`, `carbon`, `nitrogen`, `phosphorus`, `energy`), `GraphIR` dependency compilation (`relic/orchestration/graph_ir.py`), multi-rate `fast`/`medium`/`slow` execution groups ([time_scaling.md](time_scaling.md)), 10-field `CochainSpaceSpec` and bounded mathematical operator contracts ([bounded_mathematical_contracts.md](bounded_mathematical_contracts.md)), four-tier `ValidationLayer` (`L1_UNIT`--`L4_SCIENTIFIC`), and 92 declarative `IPHSEquivalenceCertificate` YAML specifications ([structural_ecology_foundation.md](structural_ecology_foundation.md)).
- **Design Blueprints and Analytical Syntheses (`[M]` / `[H]` / `[N]`)**: `relic/core/category_theory.py` is a `[M]` design specification module not imported by the simulator compiler (no compiler pass eliminates identity Functors, and conservation checks are executed by `StateManager` ledgers and `ValidationLayer` rules, not runtime natural-transformation objects). Multi-isotope pool tracking (`IsotopeHeartbeatFunctor`, [multi_isotope_pools.md](multi_isotope_pools.md)) carries `[H]` (`runtime_unbound`) status. The structural six-tuple $\mathfrak{S}$ (Section 2.2) and the five-term total simulation error decomposition (Section 2.5) are `[N]` theoretical frameworks organizing RELIC modules across Chapters 1 and 21--30 without a single monolithic wrapper class.

Companion structural and numerical foundations are detailed in [structural_ecology_foundation.md](structural_ecology_foundation.md), [bounded_mathematical_contracts.md](bounded_mathematical_contracts.md), [scale_coupling.md](scale_coupling.md), [spatiotemporal_scale_topology.md](spatiotemporal_scale_topology.md), [time_scaling.md](time_scaling.md), [numerical_time_integration.md](numerical_time_integration.md), [hybrid_events.md](hybrid_events.md), [observation_operators.md](observation_operators.md), and [observation_state_feedback.md](observation_state_feedback.md).

---

## 2. Governing Equations

### 2.1 Coupled Reservoir-Flux Evolution System

Let $\mathbf{S}(t) \in \mathcal{X} \subseteq \mathbb{R}^n$ denote the vector of prognostic reservoir states, $\boldsymbol{\xi}(t)$ external environmental forcings, and $\boldsymbol{\theta} \in \Theta \subseteq \mathbb{R}^m$ model parameters. The continuous-time semi-discrete evolution of the coupled system is decomposed into internal stoichiometric/transport transfers $\mathcal{P}_{\mathrm{int}}$ and open-boundary exchange fluxes $\mathcal{P}_{\mathrm{ext}}$:

$$\frac{d\mathbf{S}}{dt} = \sum_{p \in \mathcal{P}_{\mathrm{int}}} \mathbf{N}_p\,\mathbf{J}_p\bigl(\mathbf{S}(t), \boldsymbol{\xi}(t), \boldsymbol{\theta}, t\bigr) + \sum_{q \in \mathcal{P}_{\mathrm{ext}}} \mathbf{B}_q\,\mathbf{u}_q\bigl(\mathbf{S}(t), \boldsymbol{\xi}(t), \boldsymbol{\theta}, t\bigr),$$

where $\mathbf{N}_p \in \mathbb{R}^{n \times r_p}$ is the incidence or stoichiometric matrix of internal process $p$, $\mathbf{J}_p \in \mathbb{R}^{r_p}$ is its process rate vector, $\mathbf{B}_q \in \mathbb{R}^{n \times s_q}$ is the boundary port injection operator, and $\mathbf{u}_q \in \mathbb{R}^{s_q}$ is the external source/sink flux vector. For every conserved material quantity $E \in \{\mathrm{H_2O}, \mathrm{C}, \mathrm{N}, \mathrm{P}\}$ with elemental content vector $\mathbf{m}_E \in \mathbb{R}_{\ge 0}^n$, internal mass conservation requires the nullspace identity

$$\mathbf{m}_E^\top \mathbf{N}_p = \mathbf{0}^\top \qquad \forall p \in \mathcal{P}_{\mathrm{int}},$$

so that total elemental storage $Q^{(E)}(t) = \mathbf{m}_E^\top \mathbf{S}(t)$ changes solely through declared boundary ports:

$$\frac{dQ^{(E)}}{dt} = \sum_{q \in \mathcal{P}_{\mathrm{ext}}} \mathbf{m}_E^\top \mathbf{B}_q\,\mathbf{u}_q\bigl(\mathbf{S}(t), \boldsymbol{\xi}(t), \boldsymbol{\theta}, t\bigr) = S_{\mathrm{ext}}^{(E)}(t) - L_{\mathrm{ext}}^{(E)}(t).$$

### 2.2 Typed Cochain and Port-Hamiltonian Structural Synthesis (`[N]` / `[I]`)

In *Methods of Ecological Analysis* (Chapters 1, 21--24), conservative multi-physics land-atmosphere models are organized by the structural six-tuple

$$\mathfrak{S} = \bigl(\mathcal{K},\; \mathcal{C}^\bullet,\; \mathcal{H},\; \mathcal{J} - \mathcal{R},\; \mathcal{G},\; \mathcal{C}_{\mathrm{ledger}}\bigr),$$

comprising:
1. A finite primal/dual cellular complex $\mathcal{K}$ with discrete coboundary operators $d_k: \mathcal{C}^k \to \mathcal{C}^{k+1}$ satisfying $d_{k+1} \circ d_k = 0$ (`relic/computation/operators/dec/`, `[I]`).
2. Graded typed cochain spaces $\mathcal{C}^\bullet = \bigoplus_k \mathcal{C}^k(\mathcal{K})$ carrying explicit 10-field `CochainSpaceSpec` metadata (`complex_id`, `orientation`, `degree`, `quantity_family`, `physical_unit`, `extensivity`, `PositivityDomain`, `SimplexSupportRule`, `frame_variance`, `measure_pairing`; `relic/computation/operators/dec/forms.py`, `[I]`).
3. A convex thermodynamic or free-energy storage functional $\mathcal{H}(\mathbf{S})$ defining conjugate intensive potentials $\boldsymbol{\mu} = \nabla_{\mathbf{S}}\mathcal{H}(\mathbf{S})$ (such as matric potential $\psi$, temperature $T$, or chemical affinity).
4. An interconnection-dissipation operator pair $(\mathcal{J}(\mathbf{S}), \mathcal{R}(\mathbf{S}))$ with skew-adjoint reversible coupling $\mathcal{J}(\mathbf{S})^\top = -\mathcal{J}(\mathbf{S})$ and symmetric positive semi-definite irreversible dissipation $\mathcal{R}(\mathbf{S}) = \mathcal{R}(\mathbf{S})^\top \succeq 0$, yielding the port-Hamiltonian evolution

$$\frac{d\mathbf{S}}{dt} = \bigl[\mathcal{J}(\mathbf{S}) - \mathcal{R}(\mathbf{S})\bigr]\nabla_{\mathbf{S}}\mathcal{H}(\mathbf{S}) + \mathcal{G}(\mathbf{S})\mathbf{u}_{\mathrm{port}},$$

and the dissipation inequality $\frac{d\mathcal{H}}{dt} = -\nabla_{\mathbf{S}}\mathcal{H}^\top \mathcal{R}(\mathbf{S})\nabla_{\mathbf{S}}\mathcal{H} + \mathbf{y}_{\mathrm{port}}^\top \mathbf{u}_{\mathrm{port}} \le \mathbf{y}_{\mathrm{port}}^\top \mathbf{u}_{\mathrm{port}}$ for conjugate port output $\mathbf{y}_{\mathrm{port}} = \mathcal{G}(\mathbf{S})^\top \nabla_{\mathbf{S}}\mathcal{H}(\mathbf{S})$.
5. Boundary port operator $\mathcal{G}(\mathbf{S})$ and discrete conservation ledger $\mathcal{C}_{\mathrm{ledger}}$ audited over each macro-step $\Delta t$ ([structural_ecology_foundation.md](structural_ecology_foundation.md)).

### 2.3 Exact Discrete Adjoint Sensitivity along Piecewise-Smooth Trajectories

Let $\mathbf{S}_t = \Phi_t(\mathbf{S}_{t-1}, \boldsymbol{\theta})$ ($t = 1, \dots, N_t$) denote the discrete macro-step transition map assembled from `torch.nn.Module` Functors starting from initial state $\mathbf{S}_0(\boldsymbol{\theta})$, and let $\mathcal{L}(\mathbf{S}_{N_t}, \boldsymbol{\theta})$ denote a scalar calibration or assimilation objective at terminal step $N_t$. Because parameters $\boldsymbol{\theta}$ enter every step transition $\Phi_t$ directly in addition to any dependence through $\mathbf{S}_0(\boldsymbol{\theta})$, the exact total parameter gradient accumulated by reverse-mode automatic differentiation (`torch.autograd`) is:

$$\frac{d\mathcal{L}}{d\boldsymbol{\theta}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} + \boldsymbol{\lambda}_0^{\mathsf{T}} \frac{\partial \mathbf{S}_0}{\partial \boldsymbol{\theta}} + \sum_{t=1}^{N_t} \boldsymbol{\lambda}_t^{\mathsf{T}} \frac{\partial \Phi_t(\mathbf{S}_{t-1}, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}},$$

where the discrete adjoint covectors $\boldsymbol{\lambda}_t \in \mathbb{R}^n$ propagate backward from $\boldsymbol{\lambda}_{N_t} = (\partial \mathcal{L} / \partial \mathbf{S}_{N_t})^{\mathsf{T}}$ via

$$\boldsymbol{\lambda}_{t-1}^{\mathsf{T}} = \boldsymbol{\lambda}_t^{\mathsf{T}} \frac{\partial \Phi_t(\mathbf{S}_{t-1}, \boldsymbol{\theta})}{\partial \mathbf{S}_{t-1}} \qquad (t = N_t, N_t - 1, \dots, 1),$$

Expanding the recursion gives the explicit product-sum representation:

$$\frac{d\mathcal{L}}{d\boldsymbol{\theta}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} + \frac{\partial \mathcal{L}}{\partial \mathbf{S}_{N_t}}\left(\prod_{k=1}^{N_t}\frac{\partial \Phi_k}{\partial \mathbf{S}_{k-1}}\right)\frac{\partial \mathbf{S}_0}{\partial \boldsymbol{\theta}} + \sum_{t=1}^{N_t}\frac{\partial \mathcal{L}}{\partial \mathbf{S}_{N_t}}\left(\prod_{k=t+1}^{N_t}\frac{\partial \Phi_k}{\partial \mathbf{S}_{k-1}}\right)\frac{\partial \Phi_t}{\partial \boldsymbol{\theta}},$$

with matrix products ordered from left ($k = N_t$) to right ($k = 1$ or $k = t+1$) and the empty product at $t = N_t$ equal to the identity matrix $\mathbf{I}_n$. Along smooth tensor execution paths, PyTorch constructs this exact discrete adjoint. When a trajectory crosses a transversal hybrid event surface $g(\mathbf{S}, t) = 0$ with state reset $\mathbf{S}^+ = \mathcal{R}_{\mathrm{evt}}(\mathbf{S}^-)$, differentiability requires inserting the saltation matrix $\boldsymbol{\Xi}$ ([hybrid_events.md](hybrid_events.md)); discrete categorical mode switches, integer indexing branches, and offline `xarray` dataset loaders are non-differentiable boundaries.

### 2.4 Spatial Quantity Basis and Cross-Scale Aggregation

Cross-scale projection between fine support cells $i \in \mathcal{I}(C)$ of area $A_i > 0$ and coarse cell $C$ of area $A_C = \sum_{i \in \mathcal{I}(C)} A_i$ is governed by `SpatialQuantityBasis` (`relic/coupling/scales/types.py`), which enforces four explicit bases (`intensive`, `cell_total`, `area_density`, `categorical`) and rejects ambiguous bare `"extensive"` declarations ([scale_coupling.md](scale_coupling.md)):

$$\bar{\phi}_C = \frac{\sum_{i \in \mathcal{I}(C)} W_i\,\phi_i}{\sum_{i \in \mathcal{I}(C)} W_i} \quad (\text{basis} \in \{\text{intensive},\; \text{area\_density}\},\; W_i = A_i\text{ for area\_density}),$$

$$X_C = \sum_{i \in \mathcal{I}(C)} X_i \quad (\text{basis} = \text{cell\_total}), \qquad X_i = x_i\,A_i \quad (\text{convert\_spatial\_basis: area\_density } x_i \leftrightarrow \text{cell\_total } X_i),$$

$$c_C = \arg\max_{c} \sum_{i \in \mathcal{I}(C),\, c_i = c} W_i \quad (\text{basis} = \text{categorical}).$$

This support-transfer contract is orthogonal to `CochainSpaceSpec.extensivity` (`"extensive"` for integrated $k$-cochain pairings versus `"intensive"` for point potentials in `relic/computation/operators/dec/forms.py`).

### 2.5 Five-Term Total Simulation Error Decomposition (`[N]`)

For true Earth-system trajectory $\mathbf{S}^\star(t)$, true observation $\mathbf{y}_T^\star = \mathcal{H}_{\mathrm{obs}}^\star(\mathbf{S}^\star(t_T)) + \boldsymbol{\eta}_T$, and numerical RELIC prediction $\widehat{\mathbf{y}}_T = \mathcal{H}_{\mathrm{obs}}(\mathbf{S}_T^{\mathrm{num}}(\widehat{\boldsymbol{\theta}}))$, *Methods of Ecological Analysis* (Chapters 25--30) decomposes the end-to-end discrepancy into five telescoping structural, parametric, spatial, temporal/splitting, and observational terms:

$$\widehat{\mathbf{y}}_T - \mathbf{y}_T^\star = \underbrace{\mathcal{H}_{\mathrm{obs}}\bigl(\mathbf{S}_T^{\mathrm{num}}(\widehat{\boldsymbol{\theta}})\bigr) - \mathcal{H}_{\mathrm{obs}}\bigl(\mathbf{S}_T^{\Delta x, \mathrm{exact}}(\widehat{\boldsymbol{\theta}})\bigr)}_{\mathbf{e}_4:\;\text{temporal integration and operator splitting}} + \underbrace{\mathcal{H}_{\mathrm{obs}}\bigl(\mathbf{S}_T^{\Delta x, \mathrm{exact}}(\widehat{\boldsymbol{\theta}})\bigr) - \mathcal{H}_{\mathrm{obs}}\bigl(\Pi_{\Delta x}\mathbf{S}_T^{\mathrm{pde}}(\widehat{\boldsymbol{\theta}})\bigr)}_{\mathbf{e}_3:\;\text{spatial discretization and scale projection}} + \underbrace{\mathcal{H}_{\mathrm{obs}}\bigl(\Pi_{\Delta x}\mathbf{S}_T^{\mathrm{pde}}(\widehat{\boldsymbol{\theta}})\bigr) - \mathcal{H}_{\mathrm{obs}}\bigl(\Pi_{\Delta x}\mathbf{S}_T^{\mathrm{pde}}(\boldsymbol{\theta}^\star)\bigr)}_{\mathbf{e}_2:\;\text{parameter estimation uncertainty}} + \underbrace{\mathcal{H}_{\mathrm{obs}}\bigl(\Pi_{\Delta x}\mathbf{S}_T^{\mathrm{pde}}(\boldsymbol{\theta}^\star)\bigr) - \mathcal{H}_{\mathrm{obs}}\bigl(\Pi_{\Delta x}\mathbf{S}^\star(t_T)\bigr)}_{\mathbf{e}_1:\;\text{model structural closure error}} + \underbrace{\mathcal{H}_{\mathrm{obs}}\bigl(\Pi_{\Delta x}\mathbf{S}^\star(t_T)\bigr) - \mathbf{y}_T^\star}_{\mathbf{e}_5:\;\text{observation operator and footprint error}},$$

yielding the triangle-inequality error bound in any normed observable space:

$$\bigl\lVert \widehat{\mathbf{y}}_T - \mathbf{y}_T^\star \bigr\rVert \le E_1^{\mathrm{model}} + E_2^{\mathrm{param}} + E_3^{\mathrm{spatial}} + E_4^{\mathrm{temporal/split}} + E_5^{\mathrm{obs}}.$$

---

## 3. Variable Table

### 3.1 Architectural Metadata, Update, and Scale Enumerations

| Symbol / Type | Owner Module | Exact Members / Domain | Units | Architectural Role |
| :--- | :--- | :--- | :--- | :--- |
| `ProcessCategory` | `relic/core/functor/functor_types.py` | `PHYSICAL_FIELD`, `BIOLOGICAL_PROCESS`, `TRANSPORT_MECHANISM`, `SCALE_COUPLING` | categorical | Maps each Functor to one of the four scientific abstract base classes and canonical parameter namespace prefixes (`physical`, `biosphere`, `transport`, `coupling.scales`). |
| `VariableKind` | `relic/core/functor/functor_types.py` | `STATE`, `FLUX`, `DIAGNOSTIC`, `PARAMETER`, `FORCING`, `UNKNOWN` | categorical | Six-member Functor I/O variable role classification (`UNKNOWN` sentinel when undeclared). |
| `UpdateMode` | `relic/core/state/types.py` | `ACCUMULATE`, `REPLACE`, `FLUX` | categorical | Per-output `UpdateItem` state application semantics inside `StateManager`. |
| `StateWriteMode` | `relic/core/state/types.py` | `MUTABLE`, `IMMUTABLE`, `DERIVED` | categorical | Write-access contract used by `StateManager` and `partition_outputs()`. |
| `StateVariableType` | `relic/core/state/types.py` | `SCALAR`, `GRID`, `CONCENTRATION`, `PARAMETER` | categorical | Storage-role tag in `StateManager` (`CONCENTRATION` informs the conservation filter; spatial extensivity is owned by `SpatialQuantityBasis`). |
| `SpatialScale` | `relic/core/functor/functor_types.py` | `POINT`, `PATCH`, `GRID_CELL`, `HILLSLOPE`, `BASIN`, `LANDSCAPE`, `GRID_2D`, `GRID_3D`, `REGIONAL`, `GLOBAL`, `UNKNOWN` | categorical | Eleven-member spatial label vocabulary (10 concrete scale labels plus `UNKNOWN`), equipped with a total preorder rank without mesh geometry ([spatiotemporal_scale_topology.md](spatiotemporal_scale_topology.md)). |
| `TemporalScale` | `relic/core/functor/functor_types.py` | `SECOND`, `MINUTE`, `HOUR`, `SUB_DAILY`, `DAY`, `WEEK`, `MONTH`, `SEASON`, `YEAR`, `STATIC`, `STEADY_STATE`, `UNKNOWN` | categorical | Twelve-member characteristic period classification driving automatic `fast`/`medium`/`slow` execution-group assignment ([time_scaling.md](time_scaling.md)). |
| `ExecutionPhase` | `relic/core/functor/functor_types.py` | `INITIALIZATION`, `PREPROCESSING`, `MAIN_PROCESSES`, `COUPLING`, `POSTPROCESSING`, `VALIDATION` | categorical | Six ordered intra-step scheduling stages executed by the simulator loop. |
| `SubFlowAccuracy` | `relic/core/functor/functor_types.py` | `EXACT`, `ORDER_2`, `ORDER_1`, `UNDECLARED` | dimensionless | Local sub-flow truncation order declaration ($p \in \{\infty, 2, 1, \text{None}\}$) governing `execution_loop.splitting_order_report`. |
| `SpatialQuantityBasis` | `relic/coupling/scales/types.py` | `INTENSIVE`, `CELL_TOTAL`, `AREA_DENSITY`, `CATEGORICAL` | categorical | Mandatory physical quantity basis for cross-scale spatial aggregation and remapping ([scale_coupling.md](scale_coupling.md)). |

### 3.2 Unified Cross-Sphere Notation and Subscript Conventions

All RELIC theory documents share the following physical symbol and subscript conventions unless locally specialized in a page's Section 3 table:

| Domain | Symbol | Physical Meaning | Canonical SI Units |
| :--- | :--- | :--- | :--- |
| **General** | $t$, $\Delta t$ | Simulation time and macro/sub-step duration | $\mathrm{s}$ |
| **General** | $T$, $T_c$ | Absolute thermodynamic temperature and Celsius temperature | $\mathrm{K}$, ${}^\circ\mathrm{C}$ |
| **General** | $P$, $\rho$ | Pressure and mass density | $\mathrm{Pa}$, $\mathrm{kg\,m^{-3}}$ |
| **General** | $\mathbf{S}$, $\boldsymbol{\theta}$, $\boldsymbol{\xi}$ | Prognostic state vector, parameter vector, and external forcing vector | variable-specific |
| **Radiation & Energy** | $R_{\mathrm{net}}$, $SW_{\mathrm{down}}$, $LW_{\mathrm{down}}$ | Net all-wave radiation, downwelling shortwave, and downwelling longwave irradiance | $\mathrm{W\,m^{-2}}$ |
| **Radiation & Energy** | $\alpha$, $\varepsilon$, $\sigma$ | Broadband/spectral albedo, thermal emissivity, and Stefan-Boltzmann constant | dimensionless, dimensionless, $\mathrm{W\,m^{-2}\,K^{-4}}$ |
| **Radiation & Energy** | $H$, $LE$, $G$, $\Delta S_{\mathrm{canopy}}$ | Sensible heat flux, latent heat flux, ground heat conduction, and canopy heat storage rate | $\mathrm{W\,m^{-2}}$ |
| **Vegetation & Carbon** | $A$, $V_{c\max}$, $R_d$ | Net $\mathrm{CO}_2$ assimilation, maximum Rubisco carboxylation rate, and mitochondrial dark respiration | $\mathrm{\mu mol\,CO_2\,m^{-2}\,s^{-1}}$ |
| **Vegetation & Carbon** | $J_{\max}$ | Potential photosynthetic electron transport capacity | $\mathrm{\mu mol\,e^-\,m^{-2}\,s^{-1}}$ |
| **Vegetation & Carbon** | $g_s$, $\mathrm{LAI}$ | Stomatal conductance to water vapor / $\mathrm{CO}_2$ and one-sided leaf area index | $\mathrm{mol\,m^{-2}\,s^{-1}}$, $\mathrm{m^2\,m^{-2}}$ |
| **Vegetation & Carbon** | $C_i$, $k_i$ | Carbon pool stock density and first-order or specific turnover rate | $\mathrm{g\,C\,m^{-2}}$ (or $\mathrm{kg\,C\,m^{-2}}$), $\mathrm{s^{-1}}$ |
| **Hydrology & Soil** | $\theta$, $\psi$, $K(\theta)$ | Volumetric soil water content, matric/water potential, and unsaturated hydraulic conductivity | $\mathrm{m^3\,m^{-3}}$, $\mathrm{Pa}$ (or $\mathrm{m}$, $\mathrm{MPa}$), $\mathrm{m\,s^{-1}}$ |
| **Hydrology & Soil** | $\mathrm{ET}$, $q$ | Evapotranspiration mass flux and Darcy/Richards volumetric water flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$, $\mathrm{m\,s^{-1}}$ |
| **Turbulence** | $u_*$, $L$, $\kappa$ | Friction velocity, Monin-Obukhov stability length, and von Karman constant | $\mathrm{m\,s^{-1}}$, $\mathrm{m}$, dimensionless |
| **Turbulence** | $z_{0m}$, $z_{0h}$, $r_a$, $K_{\mathrm{eddy}}$ | Momentum/scalar roughness lengths, aerodynamic resistance, and turbulent eddy diffusivity | $\mathrm{m}$, $\mathrm{m}$, $\mathrm{s\,m^{-1}}$, $\mathrm{m^2\,s^{-1}}$ |
| **Subscripts** | $(\cdot)_{25}$ or $(\cdot)_{\mathrm{ref}}$ | Quantity evaluated at reference temperature ($298.15\;\mathrm{K} = 25\,^\circ\mathrm{C}$) | -- |
| **Subscripts** | $(\cdot)_{\min}$, $(\cdot)_{\max}$ | Lower and upper physical or admissibility bounds | -- |
| **Subscripts** | $(\cdot)_{\mathrm{sfc}}$, $(\cdot)_{\mathrm{air}}$, $(\cdot)_{\mathrm{soil}}$, $(\cdot)_{\mathrm{leaf}}$, $(\cdot)_{\mathrm{canopy}}$ | Surface interface, ambient atmosphere, soil profile, leaf-level, and canopy-integrated support | -- |

---

## 4. Parameter Table

Universal physical constants are defined exclusively in `relic/parameters/database/relic/physical/constants.yaml` (`namespace: relic.physical.constants`) and retrieved via `get_physical_constant("<short_key>")` (`relic/core/constants.py`). Domain, scheduling, and conservation governance parameters are retrieved via `self.get_parameter_value("<key>")` inside `FunctorBase` subclasses or `get_global_manager().get("<key>")` (`relic/parameters/manager.py`) in non-Functor infrastructure.

| Parameter Key | Symbol | Default Value | Units | Bounds | Source YAML | Role in Unified Framework |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `relic.physical.constants.universal.gas_constant` | $R$ | `8.314462618` | $\mathrm{J\,mol^{-1}\,K^{-1}}$ | `[8.314462618, 8.314462618]` | `physical/constants.yaml` | CODATA 2018 molar ideal-gas constant across thermodynamics, Arrhenius kinetics, and psychrometrics |
| `relic.physical.constants.universal.stefan_boltzmann_constant` | $\sigma$ | `5.670374419e-08` | $\mathrm{W\,m^{-2}\,K^{-4}}$ | `[5.670374419e-08, 5.670374419e-08]` | `physical/constants.yaml` | Blackbody longwave emission constant in surface and canopy energy balances |
| `relic.physical.constants.universal.avogadro_constant` | $N_A$ | `6.02214076e+23` | $\mathrm{mol^{-1}}$ | `[6.02214076e+23, 6.02214076e+23]` | `physical/constants.yaml` | Photon, molecular, and molar flux conversions |
| `relic.physical.constants.earth.gravity_acceleration` | $g$ | `9.80665` | $\mathrm{m\,s^{-2}}$ | `[9.80665, 9.80665]` | `physical/constants.yaml` | Standard gravitational acceleration in hydrostatic head, Richards flow, and buoyancy stability |
| `relic.physical.constants.turbulence.von_karman_constant` | $\kappa$ | `0.41` | dimensionless | `[0.41, 0.41]` | `physical/constants.yaml` | Logarithmic wind profile and Monin-Obukhov similarity constant |
| `relic.physical.constants.air.specific_heat_cp` | $c_p$ | `1005.0` | $\mathrm{J\,kg^{-1}\,K^{-1}}$ | `[1005.0, 1005.0]` | `physical/constants.yaml` | Isobaric specific heat of dry air in sensible heat and Penman-Monteith equations |
| `computation.scheduling.seconds_per_day` | $S_d$ | `86400.0` | $\mathrm{s\,d^{-1}}$ | `[86400.0, 86400.0]` | `computation/scheduling.yaml` | Exact SI day conversion factor for `TemporalScale` characteristic periods |
| `computation.scheduling.fast_max_days` | $\tau_{\mathrm{fast,max}}$ | `0.0416666667` | $\mathrm{d}$ | `[1.0e-06, 1.0]` | `computation/scheduling.yaml` | Upper characteristic period threshold ($1\;\mathrm{h}$) assigning Functors to the `fast` execution group |
| `computation.scheduling.medium_max_days` | $\tau_{\mathrm{med,max}}$ | `1.0` | $\mathrm{d}$ | `[0.01, 30.0]` | `computation/scheduling.yaml` | Upper characteristic period threshold ($1\;\mathrm{d}$) assigning Functors to the `medium` execution group |
| `validation.conservation.tolerances.water` | $\epsilon_{\mathrm{H_2O}}$ | `1.0e-09` | domain budget unit | `[1.0e-15, 0.001]` | `validation/conservation.yaml` | Default water mass-balance residual tolerance |
| `validation.conservation.tolerances.carbon` | $\epsilon_{\mathrm{C}}$ | `1.0e-09` | domain budget unit | `[1.0e-15, 0.001]` | `validation/conservation.yaml` | Default carbon mass-balance residual tolerance |
| `validation.conservation.tolerances.nitrogen` | $\epsilon_{\mathrm{N}}$ | `1.0e-11` | domain budget unit | `[1.0e-15, 0.001]` | `validation/conservation.yaml` | Default nitrogen mass-balance residual tolerance |
| `validation.conservation.tolerances.phosphorus` | $\epsilon_{\mathrm{P}}$ | `1.0e-12` | domain budget unit | `[1.0e-15, 0.001]` | `validation/conservation.yaml` | Default phosphorus mass-balance residual tolerance |
| `validation.conservation.tolerances.energy` | $\epsilon_{\mathrm{E}}$ | `1.0e-06` | $\mathrm{J\,m^{-2}}$ (or $\mathrm{W\,m^{-2}}$) | `[1.0e-12, 1.0]` | `validation/conservation.yaml` | Default surface/column energy-balance residual tolerance |
| `validation.iphs.strict_certificate_gate_default` | -- | `false` | boolean | `{false, true}` | `validation/iphs.yaml` | Default strictness toggle for `IPHSEquivalenceCertificate` runtime gating |

---

## 5. Numerical Methods and Discretization

### 5.1 Dependency Graph Compilation (`GraphIR`) and Six-Phase Step Lifecycle

Prior to time integration, `GraphIR` (`relic/orchestration/graph_ir.py`) and simulator dependency operators (`relic/core/simulator/dependency_ops.py`) compile registered Functors into a deterministic directed acyclic execution plan:

1. **Phase and Dependency Ordering**: Functors are partitioned across the six ordered `ExecutionPhase` stages (`INITIALIZATION`, `PREPROCESSING`, `MAIN_PROCESSES`, `COUPLING`, `POSTPROCESSING`, `VALIDATION`), topologically sorted by declared input/output variable dependencies with deterministic tie-breaking, and audited via `output_overlap_governance.py` and the 13-key `GraphIR` compilation report (`report.audit`).
2. **Device Placement Policy**: `FunctorBase.__init__` initializes tensor buffers on a CPU placeholder; during simulator registration (`relic/core/simulator/device_ops.py`), modules migrate according to the centralized `relic/core/device_policy.py` SSOT (`device`, `require_gpu`, `device_overrides`, `module_device_overrides`), defaulting to GPU-first with automatic CPU fallback unless `require_gpu: true` or an unsatisfied `enable_distributed: true` constraint raises explicitly.
3. **Functional Step Evaluation and Channel Partition**: Within each sub-step, a Functor receives an immutable view of required state/forcing tensors and returns a dictionary mapped by `partition_outputs()` into `(state, diagnostics, intents)`. `StateManager` applies `UpdateItem` payloads (`ACCUMULATE`, `REPLACE`, or `FLUX`, scaled by $\Delta t$ when `per_timestep=False`) and records diagnostics and external I/O intents without side-effect mutation inside `forward()`.

### 5.2 Multi-Rate Execution Groups and Operator-Splitting Order Governance

To accommodate process time constants spanning seconds (radiation, turbulence) to years (soil organic matter stabilization, stand demographics), `relic/core/simulator/execution_group.py` and `execution_loop.py` partition active Functors into three speed groups (`fast`, `medium`, `slow`) with integer sub-step ratios $N_{\mathrm{fast}} \ge N_{\mathrm{medium}} \ge N_{\mathrm{slow}} \ge 1$ over macro-step $\Delta t$:

- **Lie-Trotter (`lie`) Schedule**: Sequential group composition $\Phi_{\Delta t}^{\mathrm{Lie}} = \Phi_{\mathrm{slow}}(\Delta t) \circ [\Phi_{\mathrm{medium}}(\Delta t / N_{\mathrm{medium}})]^{N_{\mathrm{medium}}} \circ [\Phi_{\mathrm{fast}}(\Delta t / N_{\mathrm{fast}})]^{N_{\mathrm{fast}}}$, achieving global first-order splitting accuracy $\mathcal{O}(\Delta t)$ when all constituent sub-flows satisfy `SubFlowAccuracy` order $p \ge 1$.
- **Strang (`strang`) Symmetric Schedule**: Symmetric palindromic composition across group sub-flows, achieving formal global second-order accuracy $\mathcal{O}(\Delta t^2)$ if and only if all participating Functors are time-reversible/symmetric or declare `SubFlowAccuracy.ORDER_2` or `SubFlowAccuracy.EXACT` ([time_scaling.md](time_scaling.md)). Whenever any active Functor retains the default `SubFlowAccuracy.UNDECLARED`, `execution_loop.splitting_order_report` marks global splitting-order claims as `unverified`.

Individual stiff or differential-algebraic processes within a single Functor call invoke specialized solvers from `relic/computation/solvers/` (IMEX, Rosenbrock-W, BDF, Newton-Krylov, or operator-split integrators; [numerical_time_integration.md](numerical_time_integration.md)) and event guards ([hybrid_events.md](hybrid_events.md)).

### 5.3 Four-Tier Validation Hierarchy and IPHS Certificate Registry

Runtime and offline verification combine two orthogonal governance layers ([bounded_mathematical_contracts.md](bounded_mathematical_contracts.md), [structural_ecology_foundation.md](structural_ecology_foundation.md)):

- **`ValidationLayer` (`relic/validation/framework/layers.py`)**: Four sequential validation tiers: `L1_UNIT` (positivity, finiteness, dimensional/unit consistency, parameter bounds), `L2_INTEGRATION` (cross-Functor coupling compatibility, elemental and energy conservation closure, overflow routing), `L3_BENCHMARK` (analytical solutions, MMS convergence orders, reference benchmark tolerances), and `L4_SCIENTIFIC` (emergent ecological ratios, eddy-covariance/satellite observation plausibility).
- **IPHS Equivalence Certificates (`relic/core/functor/iphs_metadata.py`, `relic/computation/operators/iphs.py`)**: The repository maintains 92 declarative `IPHSEquivalenceCertificate` YAML specifications across six structural levels (`L0 = 4`, `L1 = 15`, `L2 = 39`, `L3 = 32`, `L4 = 1`, `L5 = 1`). Those 92 certificates bind directly to domain process theory pages; `0` certificates bind via `theory_document` to this synthesis overview page.

---

## 6. Conservation, Boundary Conditions, and Ledgers

### 6.1 Five Active Elemental and Energy Budget Ledgers

`StateManager.generate_conservation_report()` (`relic/core/state/conservation.py`) and `relic/core/conservation_registry.py` audit discrete conservation over each step $[t_n, t_n + \Delta t]$ across five active domains configured in `relic/parameters/database/relic/validation/conservation.yaml`: `water`, `carbon`, `nitrogen`, `phosphorus`, and `energy`.

For each conserved material domain $E \in \{\text{water}, \text{carbon}, \text{nitrogen}, \text{phosphorus}\}$, the discrete budget residual is:

$$r_n^{(E)} = \sum_{i \in \mathcal{V}_E} \bigl(Q_{i, n+1}^{(E)} - Q_{i, n}^{(E)}\bigr) - \Delta t \left(\sum_{j \in \mathcal{S}_{\mathrm{ext}}(E)} F_{j, n}^{(E)} - \sum_{k \in \mathcal{L}_{\mathrm{ext}}(E)} F_{k, n}^{(E)}\right) + Q_{\mathrm{clip}, n}^{(E)},$$

which must satisfy $\lvert r_n^{(E)} \rvert \le \epsilon_E$. Internal pool-to-pool transfers (such as litter fragmentation into soil organic matter or root water uptake into transpiration stream storage) cancel identically when stoichiometric weights match. Surface and canopy energy closure enforces

$$r_n^{(\mathrm{energy})} = \Delta S_{\mathrm{canopy}, n} - \Delta t\,\bigl(R_{\mathrm{net}, n} - H_n - LE_n - G_n\bigr), \qquad \bigl\lvert r_n^{(\mathrm{energy})} \bigr\rvert \le \epsilon_{\mathrm{E}}.$$

### 6.2 Seven `StateManager` Invariant Contracts (`I1`--`I7`)

All Functor-to-`StateManager` state transitions obey seven invariant contracts:

| ID | Invariant Name | Mathematical / Software Contract | Enforcement Owner |
| :--- | :--- | :--- | :--- |
| **I1** | No Side Effects | `Functor.forward()` does not mutate `StateManager` storage in place; state changes occur solely through returned mappings | `FunctorBase`, `StateManager` |
| **I2** | Unit Consistency | Every `UpdateItem.unit` is dimensionally compatible with the target registered state variable (`per_timestep=False` rates carry time-derivative units) | `relic/core/units/validator.py`, `StateManager` |
| **I3** | Conservation Closure | Discrete domain residual satisfies $\lvert \Delta Q^{(E)} - (S_{\mathrm{ext}}^{(E)} - L_{\mathrm{ext}}^{(E)})\Delta t + Q_{\mathrm{clip}}^{(E)} \rvert \le \epsilon_E$ | `relic/core/state/conservation.py` |
| **I4** | Additive Reversibility | Sequential application of $+\delta\mathbf{S}$ followed by $-\delta\mathbf{S}$ in `UpdateMode.ACCUMULATE` restores the initial state to floating-point precision | `StateManager` |
| **I5** | Commutativity of Independent Increments | Within a synchronous phase, `ACCUMULATE` increments to disjoint or additive pools are invariant to summation order up to floating-point rounding (`output_overlap_governance.py` governs shared targets) | `StateManager`, `relic/core/simulator/output_overlap_governance.py` |
| **I6** | Explicit $\Delta t$-Scaling | Rate outputs (`per_timestep=False`) scale linearly by sub-step $\Delta t_{\mathrm{sub}}$, whereas pre-integrated step increments (`per_timestep=True`) apply without a second $\Delta t$ factor | `UpdateItem`, `StateManager` |
| **I7** | Accounted Bounds Clipping | Any positivity or box-bound projection $\widetilde{S}_i = \min(\max(S_i, S_{i,\min}), S_{i,\max})$ records the exact truncated mass/energy in a `clip_loss` diagnostic ledger; silent clamping is prohibited | `StateManager`, `relic/validation/framework/rules/overflow_routing.py` |

### 6.3 Multi-Isotope Mass-Balance Scope (`[H]`)

For stable and radiogenic isotope tracers ($^{13}\mathrm{C}$, $^{14}\mathrm{C}$, $^{15}\mathrm{N}$, $^{18}\mathrm{O}$, $^{2}\mathrm{H}$) with pool isotope ratio $R_i$ and process fractionation factor $\alpha_j$, isotopologue conservation requires $\frac{d(S_i R_i)}{dt} = \sum_j F_j R_j \alpha_j - \lambda_{\mathrm{decay}} S_i R_i$ (reducing to $\sum_i S_i R_i = \text{const}$ in a closed non-decaying network). In the live repository, multi-isotope pool tracking (`IsotopeHeartbeatFunctor`, [multi_isotope_pools.md](multi_isotope_pools.md)) is classified as `[H]` (`runtime_unbound`) and is not one of the five default conservation domains in `validation/conservation.yaml`.

---

## 7. Code Mapping

| Architectural Component | File Path | Key Symbols |
| :--- | :--- | :--- |
| Four-pillar scientific ABCs | `relic/core/abstractions.py` | `PhysicalField`, `BiologicalProcess`, `TransportMechanism`, `ScaleCoupling` |
| Functor base lifecycle & bridges | `relic/core/functor/functor_base.py`, `relic/core/functor/functor_bridges.py` | `FunctorBase`, `PhysicalFieldFunctor`, `BiologicalProcessFunctor`, `TransportMechanismFunctor`, `ScaleCouplingFunctor` |
| Core enums & scale/order metadata | `relic/core/functor/functor_types.py` | `ProcessCategory`, `ExecutionPhase`, `SpatialScale`, `TemporalScale`, `VariableKind`, `SubFlowAccuracy` |
| Transactional state & output partition | `relic/core/state/manager.py`, `relic/core/state/types.py` | `StateManager`, `UpdateItem`, `UpdateMode`, `StateWriteMode`, `StateVariableType`, `ExternalIntent`, `partition_outputs` |
| Conservation ledger & registry | `relic/core/state/conservation.py`, `relic/core/conservation_registry.py` | `StateManager.generate_conservation_report`, conservation domain registry |
| Physical constants & unit system | `relic/core/constants.py`, `relic/core/units/validator.py` | `get_physical_constant`, `UnitValidator` |
| Centralized parameter manager | `relic/parameters/manager.py` | `ParameterManager`, `get_global_manager` |
| Device placement SSOT | `relic/core/device_policy.py` | Device resolution (`auto`, `cpu`, `cuda`, `require_gpu`, per-functor/per-module overrides) |
| GraphIR dependency compiler | `relic/orchestration/graph_ir.py` | `GraphIR` compilation, topological ordering, 13-key `report.audit` |
| Multi-rate simulator loop | `relic/core/simulator/execution_loop.py`, `relic/core/simulator/execution_group.py` | `fast`/`medium`/`slow` sub-stepping, `lie`/`strang` schedules, `splitting_order_report` |
| Spatial quantity basis & scale types | `relic/coupling/scales/types.py`, `relic/coupling/scales/scale_manager.py` | `SpatialQuantityBasis`, `convert_spatial_basis`, `ScaleManager` |
| Typed DEC cochain space contracts | `relic/computation/operators/dec/forms.py` | `CochainSpaceSpec`, `CochainTensor`, `PositivityDomain`, `SimplexSupportRule` |
| Four-tier validation & IPHS metadata | `relic/validation/framework/layers.py`, `relic/core/functor/iphs_metadata.py`, `relic/computation/operators/iphs.py` | `ValidationLayer`, `IPHSEquivalenceCertificate`, runtime IPHS certificate evaluators |
| Categorical design blueprint (`[M]`) | `relic/core/category_theory.py` | Standalone category-theoretic specification classes (not imported by simulator runtime) |
| Core YAML parameter databases | `relic/parameters/database/relic/physical/constants.yaml`, `relic/parameters/database/relic/computation/scheduling.yaml`, `relic/parameters/database/relic/validation/conservation.yaml`, `relic/parameters/database/relic/validation/iphs.yaml` | Universal physical constants, multi-rate thresholds, five-domain conservation tolerances, IPHS gate defaults |

---

## 8. Implementation Status

| Capability / Theoretical Contract | Status | Live Verification Notes |
| :--- | :--- | :--- |
| Four-pillar ABC to bridge to `FunctorBase(torch.nn.Module)` chain | `[I]` | Enforced across `relic/core/abstractions.py` and `relic/core/functor/` |
| Centralized `ParameterManager` + `physical/constants.yaml` SSOT | `[I]` | Live YAML validation, bounds metadata, and `get_physical_constant()` lookup |
| `StateManager` transactional updates (`I1`--`I7`) & 5-domain conservation reports | `[I]` | Active for `water`, `carbon`, `nitrogen`, `phosphorus`, and `energy` via `partition_outputs()` and `UpdateItem` |
| `GraphIR` DAG compilation, overlap governance, and `device_policy.py` SSOT | `[I]` | Deterministic topological sort, 13-key `report.audit`, and centralized CPU/CUDA placement |
| Multi-rate `fast`/`medium`/`slow` execution with `SubFlowAccuracy` reporting | `[P]` | `lie` and `strang` group schedules are `[I]`; `SubFlowAccuracy` defaults to `UNDECLARED` unless explicitly declared by each Functor |
| End-to-end `torch.autograd` discrete adjoint differentiation | `[P]` | Exact along smooth `torch.Tensor` execution paths; hybrid event saltation matrices are opt-in ([hybrid_events.md](hybrid_events.md)), and categorical switches / `xarray` loaders are non-differentiable |
| `SpatialQuantityBasis` (`intensive`, `cell_total`, `area_density`, `categorical`) | `[I]` | Enforced in `relic/coupling/scales/types.py`; bare `"extensive"` rejected |
| 10-field `CochainSpaceSpec`, `ValidationLayer` (`L1`--`L4`), and 92 IPHS YAML certificates | `[I]` | Live in `relic/computation/operators/dec/forms.py`, `relic/validation/framework/layers.py`, and `relic/parameters/database/relic/` (`0` certificates bound directly to this overview page) |
| Multi-isotope prognostic conservation (`IsotopeHeartbeatFunctor`) | `[H]` | `runtime_unbound` in default profiles; see [multi_isotope_pools.md](multi_isotope_pools.md) |
| Category-theory morphism/natural-transformation module (`relic/core/category_theory.py`) | `[M]` | Design blueprint module not imported by `GraphIR` or `execution_loop.py`; no compiler identity-elision pass |
| Structural six-tuple $\mathfrak{S}$ and five-term total error decomposition ($E_1$--$E_5$) | `[N]` | Analytical book-level syntheses (*Methods of Ecological Analysis*, Ch. 1, 21--30) organizing individual RELIC components without a single monolithic Python class |

---

## 9. References

1. Bonan, G. B. (2019). *Climate Change and Terrestrial Ecosystem Modeling*. Cambridge University Press. DOI: 10.1017/9781107339217.
2. Farquhar, G. D., von Caemmerer, S., & Berry, J. A. (1980). A biochemical model of photosynthetic $\mathrm{CO}_2$ assimilation in leaves of $\mathrm{C}_3$ species. *Planta*, 149(1), 78--90. DOI: 10.1007/BF00386231.
3. Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM. DOI: 10.1137/1.9780898717761.
4. Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations* (2nd ed.). Springer. DOI: 10.1007/3-540-30666-8.
5. Hirani, A. N. (2003). *Discrete Exterior Calculus*. Ph.D. Thesis, California Institute of Technology.
6. Lloyd, J., & Taylor, J. A. (1994). On the temperature dependence of soil respiration. *Functional Ecology*, 8(3), 315--323. DOI: 10.2307/2389824.
7. Monin, A. S., & Obukhov, A. M. (1954). Basic laws of turbulent mixing in the surface layer of the atmosphere. *Trudy Geofizicheskogo Instituta AN SSSR*, 24(151), 163--187.
8. Monsi, M., & Saeki, T. (1953). Uber den Lichtfaktor in den Pflanzengesellschaften und seine Bedeutung fur die Stoffproduktion. *Japanese Journal of Botany*, 14, 22--52.
9. Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205--234.
10. Parton, W. J., Schimel, D. S., Cole, C. V., & Ojima, D. S. (1987). Analysis of factors controlling soil organic matter levels in Great Plains grasslands. *Soil Science Society of America Journal*, 51(5), 1173--1179. DOI: 10.2136/sssaj1987.03615995005100050015x.
11. Richards, L. A. (1931). Capillary conduction of liquids through porous mediums. *Physics*, 1(5), 318--333. DOI: 10.1063/1.1745010.
12. van der Schaft, A., & Jeltsema, D. (2014). Port-Hamiltonian systems theory: An introductory overview. *Foundations and Trends in Systems and Control*, 1(2--3), 173--378. DOI: 10.1561/2600000002.
