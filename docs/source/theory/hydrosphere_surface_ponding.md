# Hortonian Surface Ponding and Hybrid Topology Transitions

## 1. Process Definition and Scope

`relic/scientific/hydrosphere/runoff/surface_ponding.py` implements plot-scale Hortonian infiltration-excess surface ponding as a hybrid topology-switching pair committed through `relic/core/simulator/registration.py` (`TopologyTransition`, `apply_topology_transition`):

1. `HortonianPondingEventFunctor` (`PhysicalFieldFunctor`, always connected): computes infiltration excess from cumulative `stochastic_rainfall_depth` and emits pure `ExternalIntent("topology", ...)` transition descriptors (`ponding_onset_transition`, `ponding_cessation_transition`).
2. `SurfacePondingFunctor` (`TransportMechanismFunctor`, connected only while `surface_ponding_active` equals 1, declaring `required: ["hortonian_ponding_event"]`): integrates transient ponded surface storage and overland drainage as an analytical linear reservoir.

Multi-process simultaneous topology swaps and runtime tensor-shape dimension changes across topology transitions remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapters 14 and 19 (Section 19.5.2).

## 2. Governing Equations

### 2.1 Stateful Rainfall-Increment Partitioning (`HortonianPondingEventFunctor`)

With cumulative rainfall depth $H_n$ (`stochastic_rainfall_depth`, $\mathrm{mm}$), infiltration capacity $I_{\max}$ ($\mathrm{mm\,h^{-1}}$), and $s_{\mathrm{hour}}$ resolved from `get_physical_constant("seconds_per_hour")` (Horton, 1933; Beven, 2012):

$$
\Delta H_n =
\begin{cases}
0, & n = 1 \;\text{(initial baseline capture after construction or reset)} \\
H_n - H_{n-1}, & n > 1
\end{cases}
$$

$$
E_n = \max\!\left(0,\; \Delta H_n - \frac{I_{\max}\,\Delta t}{s_{\mathrm{hour}}}\right), \qquad
I_{\mathrm{exc},n} = \Delta H_n - E_n = \min\!\left(\Delta H_n,\; \frac{I_{\max}\,\Delta t}{s_{\mathrm{hour}}}\right)
$$

Any decreasing cumulative depth ($\Delta H_n < 0$) raises `ValueError`.

### 2.2 Multi-Cell Onset and Cessation Guards

For binary uniform connectivity flag $a \in \{0, 1\}$ (`surface_ponding_active`) across cells $c$:

$$
\text{onset guard: } a = 0 \;\land\; \max_c E_n \ge E_{\mathrm{on}}, \qquad
\text{cessation guard: } a = 1 \;\land\; \max_c E_n = 0 \;\land\; \max_c h_n \le h_{\mathrm{off}}
$$

### 2.3 Analytical Linear-Reservoir Recession (`SurfacePondingFunctor`)

While connected ($a = 1$), ponded depth $h_n$ ($\mathrm{mm}$) absorbs excess $E_n$ and drains with time scale $\tau > 0$ ($\mathrm{s}$) (Dooge, 1973):

$$
h_0 = h_n + E_n, \qquad
h_{n+1} = h_0\exp\!\left(-\frac{\Delta t}{\tau}\right), \qquad
q_n = h_0 - h_{n+1}, \qquad
\Delta h_n = h_{n+1} - h_n = E_n - q_n
$$

### 2.4 Hybrid Topology Remaps and Ledger Equations

On state triple $x = (h, E, a)^{\mathsf{T}}$ with surface water ledger row $C = (1, 1, 0)$ (Goebel et al., 2012):

$$
\text{onset: } P_{\mathrm{on}} = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix},\;
R_{\mathrm{on}}\eta_{\mathrm{on}} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix},\;
C x^+ = C x^-
$$

$$
\text{cessation: } P_{\mathrm{off}} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix},\;
R_{\mathrm{off}}\eta_{\mathrm{off}} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix},\;
\Pi_{\mathrm{off}} = -h^-,\;
C x^+ = C x^- + \Pi_{\mathrm{off}}
$$

where $\Pi_{\mathrm{off}} = -h^-$ records residual depression infiltration leaving the surface ledger at cessation.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $H$ | Cumulative stochastic rainfall depth | $\mathrm{mm}$ | `stochastic_rainfall_depth` | `state` |
| $a$ | Binary ponding-graph connectivity state ($\{0, 1\}$) | $\mathrm{1}$ | `surface_ponding_active` | `state` |
| $h$ | Surface ponded water depth (input stock and `ACCUMULATE` increment $\Delta h_n$) | $\mathrm{mm}$ | `surface_ponded_depth` | `state` |
| $E$ | Step infiltration-excess rainfall depth | $\mathrm{mm}$ | `surface_ponding_excess` | `state` |
| $E_d, \Delta H, I_{\mathrm{exc}}$ | Event-detector excess, rainfall increment, and infiltrated step diagnostics | $\mathrm{mm}$ | `diagnostics.ponding_event.excess_depth`, `rainfall_increment`, `infiltrated_excess` | `diagnostic` |
| $q_n, r_w$ | Overland runoff depth and ponded water-balance residual ($\Delta h_n - E_n + q_n$) | $\mathrm{mm}$ | `diagnostics.surface_ponding.runoff_depth`, `water_balance_residual` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $I_{\max}$ | `hydrosphere.runoff.ponding_infiltration_capacity` | $\mathrm{mm\,h^{-1}}$ | `hydrosphere/runoff.yaml` | Horton (1933); Beven (2012) |
| $E_{\mathrm{on}}$ | `hydrosphere.runoff.ponding_onset_depth` | $\mathrm{mm}$ | `hydrosphere/runoff.yaml` | Beven (2012) |
| $h_{\mathrm{off}}$ | `hydrosphere.runoff.ponding_cessation_depth` | $\mathrm{mm}$ | `hydrosphere/runoff.yaml` | Beven (2012) |
| $\tau$ | `hydrosphere.runoff.ponding_drainage_time` | $\mathrm{s}$ | `hydrosphere/runoff.yaml` | Dooge (1973) |
| $s_{\mathrm{hour}}$ | `relic.physical.constants.seconds_per_hour` | $\mathrm{s\,h^{-1}}$ | `physical/constants.yaml` | SI time constant |

## 5. Numerical Methods and Discretization

`HortonianPondingEventFunctor` emits `surface_ponding_excess` as a `REPLACE` state and queues `ExternalIntent("topology", ...)` without mutating simulator registration mid-step; `retry_external_deliveries` commits the transition only after the accepting step completes. `SurfacePondingFunctor` integrates linear-reservoir decay analytically via $\exp(-\Delta t / \tau)$ and emits $\Delta h_n$ as `UpdateItem(UpdateMode.ACCUMULATE, per_timestep=True)`. Checkpoints capture `_last_rainfall_depth` and post-transition graph epoch metadata for bitwise-exact restart replay.

## 6. Conservation, Boundary Conditions, and Ledgers

Within connected steps, `SurfacePondingFunctor` satisfies $\Delta h_n - E_n + q_n = 0$ to machine roundoff. Across topology transitions, `apply_topology_transition` verifies $\lvert C x^+ - C x^- - \Pi\rvert$ against tolerance prior to mutating registration or `StateManager` (with writer provenance `topology:<transition>`), rolling back both registration and state on any audit violation and recording `diagnostics.topology_transition.ledger_residual` and `ledger_change`.

## 7. Code Mapping

| Functor / Function | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `HortonianPondingEventFunctor` | `relic/scientific/hydrosphere/runoff/surface_ponding.py` | `PhysicalField` (`SPATIAL_DIMS=None`) | `PhysicalFieldFunctor` | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.PLOT`) |
| `SurfacePondingFunctor` | `relic/scientific/hydrosphere/runoff/surface_ponding.py` | `TransportMechanism` (`TRANSPORT_TYPE="advection"`) | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.PLOT`) |
| `ponding_onset_transition`, `ponding_cessation_transition` | `relic/scientific/hydrosphere/runoff/surface_ponding.py` | `TopologyTransition` factories | Committed via `apply_topology_transition` | Post-step external-intent commit |

## 8. Implementation Status

- `HortonianPondingEventFunctor`: `[I]`, IPHS **Level 2** (`field`, certificate `hydrosphere_runoff_surface_ponding_hortonian_ponding_event_functor.yaml`).
- `SurfacePondingFunctor`: `[I]`, IPHS **Level 2** (`transport`, certificate `hydrosphere_runoff_surface_ponding_surface_ponding_functor.yaml`).
- Multi-process swaps and tensor-shape dimension changes across topology transitions: `[N]`.

Frozen certificate expressions:

```text
E_n = max(0, H_n - H_{n-1} - I_max dt / 3600)
onset request when a = 0 and max E_n >= E_on; cessation request when a = 1, E_n = 0 and max h_n <= h_off; requests are pure ExternalIntent descriptors
h_{n+1} = (h_n + E_n) exp(-dt/tau)
Delta h = h_{n+1} - h_n emitted as ACCUMULATE; q_n and the residual Delta h - E_n + q_n emitted as observations
```

## 9. References

- Beven, K. J. (2012). *Rainfall-Runoff Modelling: The Primer* (2nd ed.). Wiley.
- Dooge, J. C. I. (1973). *Linear Theory of Hydrologic Systems* (Technical Bulletin No. 1468). USDA Agricultural Research Service.
- Goebel, R., Sanfelice, R. G., and Teel, A. R. (2012). *Hybrid Dynamical Systems: Modeling, Stability, and Robustness*. Princeton University Press.
- Horton, R. E. (1933). The role of infiltration in the hydrologic cycle. *Transactions, American Geophysical Union*, 14(1), 446-460.
