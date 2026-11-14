# Spatiotemporal Scale Topology, Dependency Scheduling, and Multi-Rate Splitting

## 1. Process Definition and Scope

`RELICSimulator` (`relic/core/simulator/`) and `DependencyGraphResolver` (`relic/core/dependency_graph.py`) organize registered scientific `FunctorBase` instances across spatial scale labels (`SpatialScale`), temporal scale classifications (`TemporalScale`), directed acyclic dependency graphs (DAGs), three-rate execution zones ($Z_{\mathrm{fast}}, Z_{\mathrm{medium}}, Z_{\mathrm{slow}}$), Lie and palindromic Strang operator splitting, update-magnitude adaptive time stepping, and seven runtime execution-chain invariants (`I1`–`I7`). Theoretical foundations for multi-rate splitting, topological partial orders, and scale-transition adjunctions appear in *Methods of Ecological Analysis*, Chapters 1, 21, 22, and 24.

This page strictly separates executable runtime mechanisms (`[I]`) from mathematical design interpretations (`[M]` / `[N]`):
- **Runtime mechanisms (`[I]`)**: Four-pillar registration checks (`_require_functor_base`, `_require_category_bridge`), registration-indexed topological sorting, three-tier multi-rate scheduling in `elapsed seconds`, calendar validation (`configure_calendar_settings`, rejecting unrecognized calendars with `Unsupported calendar`), cross-group lag detection and `multirate_exchanges` interval-mean reconstruction, Lie/Strang fast-group composition with `GroupComposition`, heuristic RMS adaptive step control with transactional rollback, and `StateManager` conservation/bounds invariants.
- **Design interpretations (`[M]` / `[N]`)**: Category-theoretic and fiber-bundle analogies (`relic/core/category_theory.py`, not imported at runtime) and diagnostic `TemporalZones` dominance monitoring (`computation.temporal_zones.*`). Spatial scale labels carry no mesh geometry; actual geometric transformations execute solely on explicit `GridDescriptor` and `SpatialQuantityBasis` inputs ([scale_coupling.md](scale_coupling.md)).

## 2. Governing Equations

### 2.1 Scale Preorders, Characteristic Time, and Rate-Zone Classification

The spatial scale set $\mathcal{S}$ (`SpatialScale` in `relic/core/functor/functor_types.py`) carries the preference rank $p: \mathcal{S} \cup \{\text{unknown}\} \to \mathbb{Z}$ (`_SCALE_PREFERENCE` in `relic/core/simulator/scale_ops.py`):

$$
p(\text{point}) = 6, \; p(\text{patch}) = 5, \; p(\text{grid\_cell}) = 4, \; p(\text{hillslope}) = 3, \; p(\text{basin}) = p(\text{landscape}) = 2,
$$

$$
p(\text{grid\_2d}) = 1, \; p(\text{grid\_3d}) = 0, \; p(\text{regional}) = -1, \; p(\text{global}) = -2, \; p(\text{unknown}) = -3.
$$

Because $p(\text{basin}) = p(\text{landscape}) = 2$, $s_1 \preceq s_2 \iff p(s_1) \ge p(s_2)$ defines a total preorder (antisymmetry does not hold). For nested fine cells $i \in J$ with full coverage $A_J = \sum_{i \in J} A_i > 0$, broadcast $(P c)_i = c_{J(i)}$ and area-weighted restriction $(R f)_J = A_J^{-1}\sum_{i \in J} A_i f_i$ are weighted inner-product adjoints ($R = M_c^{-1} P^{\mathsf{T}} M_f$, $R P = I_c$, $P R \neq I_f$), whereas nonlinear processes exhibit a generally non-zero Jensen commutation defect $\epsilon_{\mathrm{comm}} = \lVert \mathrm{Agg}(F(x)) - F(\mathrm{Agg}(x)) \rVert$.

For `TemporalScale`, `to_characteristic_time_days` resolves fixed SI durations from `SECONDS_PER_DAY` $S_d$ ($\tau(\text{second}) = 1/S_d$, $\tau(\text{minute}) = S_m/S_d$, $\tau(\text{hour}) = S_h/S_d$, $\tau(\text{day}) = 1$) and requires explicit realized bounds $B = [b_0, b_1]$ ($\tau(t; B) = b_1 - b_0$) for calendar/schedule scales (`sub_daily`, `week`, `month`, `season`, `year`). Rate-zone classification $\gamma$ partitions characteristic times $\tau_c$ (days) against `ParameterManager` thresholds $0 < \tau_f < \tau_m$ (`resolve_scheduling_thresholds` in `relic/core/simulator/dependency_ops.py`):

$$
\gamma(\tau_c) = \begin{cases}
\text{fast}, & \tau_c \le \tau_f, \\
\text{medium}, & \tau_f < \tau_c \le \tau_m, \\
\text{slow}, & \tau_c > \tau_m.
\end{cases}
$$

Precedence in `determine_execution_group` is: explicit `register_functor(execution_group=...)` argument, Functor `execution_group` attribute, `characteristic_time_days` via $\gamma(\tau_c)$, `TemporalScale` class (`sub_daily` $\in Z_{\mathrm{fast}}$, `week` $\in Z_{\mathrm{medium}}$, `month`/`season`/`year` $\in Z_{\mathrm{slow}}$), `process_groups[name]`, and default `"fast"` (contradicting declarations raise `ValueError` and roll back registration).

### 2.2 DAG Level Decomposition and Multi-Rate Interval Boundaries

For registered Functors $V$ and resolved same-step required/present-optional dependency edges $E$ (`lagged` dependencies excluded), topological sorting (`DependencyGraphResolver.topological_sort`) breaks ties deterministically by registration index (Kahn, 1962), raising `CyclicDependencyError` if cyclic. Longest-path depth $\lambda(v) = 0$ if $\mathrm{deps}_{\mathrm{res}}(v) = \emptyset$ and $1 + \max_{u \in \mathrm{deps}_{\mathrm{res}}(v)} \lambda(u)$ otherwise defines edge-free level sets $P_l = \lambda^{-1}(l)$ (`get_parallel_groups()`, reported for diagnostics while executed sequentially).

The simulator clock advances in `elapsed seconds` ($t_{\mathrm{future}} = t_{\mathrm{current}} + h$, $h > 0$). `Z_fast` executes every step over $[t_{\mathrm{current}}, t_{\mathrm{future}}]$. `Z_medium` (interval $T_m = \text{medium\_interval\_seconds}$) and `Z_slow` (interval $T_s = \text{slow\_interval\_seconds}$, or civil month ordinal $m_C(t_{\mathrm{future}}; o) > m_{\mathrm{prev}}$ when $T_s$ is unset) trigger when pure predicates `should_execute_medium` / `should_execute_slow` detect an index crossing:

$$
\left\lfloor \frac{t_{\mathrm{future}}}{T_g} \right\rfloor > i_g, \qquad \Delta t_{\mathrm{eff},g} = t_{\mathrm{future}} - t_{\mathrm{last},g} > 0, \qquad g \in \{\text{medium}, \text{slow}\},
$$

and `commit_rate_group_interval` advances $i_g$ transactionally only after the group completes. For declared `multirate_exchanges` with `mode="interval_mean"`, `_commit_multirate_exchanges` updates `exchange_mean__<id>` across accepted fast steps $h_k$ covering elapsed consumer window $\tau = t_k - t_{\mathrm{last,consumer}}$ (Gear and Wells, 1984):

$$
m_e \leftarrow \begin{cases}
\bar{v}_k, & \tau = 0, \\
m_e + \dfrac{h_k}{\tau + h_k}(\bar{v}_k - m_e) = \dfrac{\sum_j \bar{v}_j h_j}{\sum_j h_j}, & \tau > 0.
\end{cases}
$$

When optional diagnostic `TemporalZones` monitoring (`computation.temporal_zones.monitoring_enabled`) is enabled, `_check_cross_zone_error_transfer` alerts when the zone-dominance ratio $R(v) = \max_z C_z(v) / \sum_z C_z(v)$ ($C_z(v) = \sum_{F \in Z_z} \lvert \Delta v_F \rvert$) exceeds $\theta_{\mathrm{tz}}$.

### 2.3 Lie and Palindromic Strang Operator Splitting

For fast-group order $(F_1, \dots, F_n)$ over step $h = \Delta t$:

1. **Lie-Trotter splitting (`splitting_mode="lie"`, or `"strang"` with $n < 2$)** runs one sequential pass over $[t, t + h]$, carrying local error $\mathcal{O}(h^2)$ with linear Baker-Campbell-Hausdorff remainder $\tfrac{1}{2} h^2 \sum_{k < l} [A_l, A_k] + \mathcal{O}(h^3)$:

   $$\Phi_{\mathrm{Lie}}^{h} = \Phi_{F_n}^{h} \circ \cdots \circ \Phi_{F_1}^{h}.$$

2. **Palindromic Strang splitting (`splitting_mode="strang"`, $n \ge 2$)** executes $2n - 1$ group commits—first half-steps over $[t, t + h/2]$, full middle step of $F_n$ over $[t, t + h]$, and reversed second half-steps starting at $t + h/2$ (Sportisse, 2000; Hairer, Lubich, and Wanner, 2006):

   $$\Phi_{\mathrm{Strang}}^{h} = \Phi_{F_1}^{h/2} \circ \cdots \circ \Phi_{F_{n-1}}^{h/2} \circ \Phi_{F_n}^{h} \circ \Phi_{F_{n-1}}^{h/2} \circ \cdots \circ \Phi_{F_1}^{h/2}.$$

   For two linear exact sub-flows $e^{h A_k}$, the leading local error is $\tfrac{1}{24} h^3 \left(2[A_2, [A_2, A_1]] - [A_1, [A_1, A_2]]\right) + \mathcal{O}(h^4)$. Because numerical sub-integrators cap composed accuracy, `get_splitting_order_report()` bounds effective order from declared `FunctorMetadata.sub_flow_accuracy` ($p_i \in \{1, 2, \infty\}$ or `"unverified"` if any member is `UNDECLARED`) as

   $$q = \min\!\left(s,\; \min_{1 \le i \le n} p_i\right), \qquad s = \begin{cases} 1, & \text{Lie}, \\ 2, & \text{Strang}. \end{cases}$$

   Any Functor with `substep_count = m > 1` (or `substep_overrides` `{count: m}` / `{max_seconds: s}` giving $m = \lceil \Delta t_{\mathrm{call}} / s \rceil$) composes $\Phi_F^{\Delta t_{\mathrm{call}}} = (\Phi_F^{\Delta t_{\mathrm{call}}/m})^m$. Permutation sensitivity is quantified by `compute_splitting_sensitivity` as $S(v) = \operatorname{std}_\sigma(\phi_\sigma(v)) / \operatorname{mean}_\sigma(\lvert\phi_\sigma(v)\rvert)$.

### 2.4 Update-Magnitude Adaptive Step Control and Execution Invariants

When `adaptive_dt_enabled=True`, `_compute_error_norm` evaluates an RMS update-magnitude norm across the $m$ deduplicated prognostic variables actually committed by the fast group ($u_k = x'_k - x_k$; Hairer and Wanner, 1996; Uruci, 2018):

$$
\mathrm{tol}_k = \mathrm{atol} + \mathrm{rtol}\,\lvert x_k \rvert, \qquad
\mathrm{Err} = \sqrt{\frac{1}{m}\sum_{k=1}^{m} \operatorname{mean}_{\mathrm{elements}}\!\left[\left(\frac{u_k}{\mathrm{tol}_k}\right)^2\right]}.
$$

Note that $u_k$ is the committed state change and not an embedded local-truncation-error estimate. When $\mathrm{Err} \le 1$, the step is accepted and the next proposal is $\Delta t_{\mathrm{new}} = \operatorname{clamp}\!\left(\Delta t \cdot \min(5.0, \max(0.1, 0.9 / \sqrt{\mathrm{Err}})),\; \Delta t_{\min},\; \Delta t_{\max}\right)$ ($2\Delta t$ if $\mathrm{Err} = 0$). When $\mathrm{Err} > 1$ and $\Delta t > \Delta t_{\min}$, `_execute_fast_with_adaptive_retry` rolls back the `StateManager` checkpoint, Functor `state_dict`s, RNG states, pinned Brownian increments, and provenance buffers, then executes one retry at $\Delta t_{\mathrm{retry}} = \max(\Delta t_{\min}, \Delta t \cdot \max(0.1, 0.9 / \sqrt{\mathrm{Err}}))$.

External model coupling (`TimeStagger` in `relic/coupling/protocol/time_stagger.py`) separately negotiates commensurable millisecond-quantized steps via $\Delta t_{\mathrm{coupled}} = 10^{-3}\operatorname{lcm}_k(\lfloor 1000\,\Delta t_k \rfloor)\,\mathrm{s}$.

## 3. Variable Table

| Quantity / Field | Meaning | Units | Owner | Role |
| --- | --- | --- | --- | --- |
| `current_time`, `step_dt`, $\Delta t_{\mathrm{eff},g}$ | Simulator clock in `elapsed seconds` and rate-group intervals | $\mathrm{s}$ | `RELICSimulator` | Scheduler clock |
| `order_fast`, `order_medium`, `order_slow` | Registration-tie-broken topological execution lists | `list[str]` | `scale_ops.py` | Frozen per-step provider view |
| `exchange_mean__<id>` | Piecewise-constant fast-to-slower interval mean $m_e$ | `[var]` | `StateManager` (`exchange:<id>`) | Governed multi-rate overlay |
| `Err`, $\Delta t_{\mathrm{new}}$ | Fast-group committed update-magnitude RMS norm and next step | $1$, $\mathrm{s}$ | `execution_loop.py` | Adaptive step heuristic |
| `clip_loss` | Signed bounds-clipping discrepancy $\sum(x_v^{\mathrm{req}} - \operatorname{clamp}(x_v^{\mathrm{req}}, l_v, u_v))$ | `[var]` | `StateManager.get_diagnostics()` | Invariant `I7` ledger |
| `integration_plan_witness` | Schedule-equivalence witness against `get_composition_plan()` | `@dataclass` | `GraphExecutor` | Opt-in provenance witness |

## 4. Parameter Table

| Key | Units | Default (Bounds) | Source YAML | Consumer |
| --- | --- | --- | --- | --- |
| `computation.scheduling.fast_group_max_characteristic_days` | `day` | `1.0` ($[0.001, 30.0]$) | `computation/scheduling.yaml` | `resolve_scheduling_thresholds` ($\tau_f$) |
| `computation.scheduling.medium_group_max_characteristic_days` | `day` | `10.0` ($[0.01, 365.0]$) | `computation/scheduling.yaml` | `resolve_scheduling_thresholds` ($\tau_m$) |
| `computation.scheduling.splitting_mode`, `splitting_policy`, `fast_order` | `str` | `"lie"`, `"warn"`, `"forward"` | `computation/scheduling.yaml` | `_execute_fast_group` & Strang admission |
| `computation.scheduling.substep_overrides`, `multirate_exchanges` | `dict`, `list` | `{}`, `[]` | `computation/scheduling.yaml` | Per-Functor substeps & cross-rate exchanges |
| `computation.scheduling.adaptive.atol`, `rtol`, `dt_min_seconds`, `dt_max_factor` | `1`, `1`, `s`, `1` | `1.0e-6`, `1.0e-3`, `1.0`, `10.0` | `computation/scheduling.yaml` | `_compute_error_norm` & step bounds |
| `computation.scheduling.topology_transition_tolerance` | `dimensionless` | `1.0e-9` ($[0.0, 10^{-3}]$) | `computation/scheduling.yaml` | Topology transition verification |
| `computation.temporal_zones.monitoring_enabled`, `cross_transfer_threshold` | `bool`, `1` | `False`, `0.8` ($[0.0, 1.0]$) | `computation/temporal_zones.yaml` | `_check_cross_zone_error_transfer` |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1.0e-6` ($[10^{-10}, 10^{-3}]$) | `validation/conservation.yaml` | `StateManager` commit-time element ledgers |
| `coupling.scales.defaults.conservation_tolerance` | `dimensionless` | `1.0e-8` ($[10^{-14}, 10^{-4}]$) | `coupling/defaults.yaml` | `ScaleCouplingFunctor.check_conservation` |

## 5. Numerical Methods and Discretization

1. **Calendar and clock initialization**: `configure_calendar_settings` (`relic/core/simulator/init_ops.py`) normalizes CF synonyms (`365_day` to `noleap`, `gregorian` to `standard`), admits `("noleap", "360_day", "standard", "proleptic_gregorian")`, validates any configured ISO-8601 `time_origin` via `parse_time_origin`, and raises `ValueError("Unsupported calendar_type ...")` on `all_leap`, `julian`, or unknown calendars.
2. **Strang admission and sequential update merging**: At provider-view publication, Strang admission refuses direct flux relays between two stateful fast Functors (S1), refuses undeclared `sub_flow_accuracy` under `splitting_policy="require_declared"` unless stateless (S3), and refuses `fast_order="reverse"` when same-step fast dependency edges exist (S5). Across substeps and Strang half-steps, `merge_sequential_updates` (`relic/core/simulator/execution_group.py`) sums `ACCUMULATE` amounts, weights per-second `ACCUMULATE` rates and qualified physical `REPLACE` boundary ports by sub-interval coverage $h_{\mathrm{sub}} / H$, and retains the final observation for ordinary `REPLACE` states and `diagnostics.*` outputs.
3. **Cross-group lag governance**: `_warn_cross_group_dependencies` flags unacknowledged slow-reads-fast sampling lags ($\mathrm{rank}(F_c) > \mathrm{rank}(F_p)$) and fast-reads-slow hold lags ($\mathrm{rank}(F_c) < \mathrm{rank}(F_p)$); `strict_cross_group_dependencies=True` rejects publication, while `resolve_cross_group_deps=True` promotes slow sampling-lag consumers to `medium` only when $t_{\mathrm{last,slow}} = t_{\mathrm{last,medium}}$.

## 6. Conservation, Boundary Conditions, and Ledgers

The execution loop enforces seven structural invariants across every accepted step:
- **`I1` (No Side Effects)**: `Functor.forward` never mutates `StateManager` storage directly.
- **`I2` (Unit Consistency)**: Increment units match target variable contracts (`[\Delta x_v] = [x_v]` for `ACCUMULATE`).
- **`I3` (Conservation)**: Commit-time `StateManager.apply_updates` verifies $\lvert \sum_i w_i \Delta x_i - (\sum_j S_j - \sum_k L_k)\Delta t \rvert \le \varepsilon_{\mathrm{cons}}$ against `validation.conservation.conservation_tolerance`, while scale-coupling operators run `check_conservation` against `coupling.scales.defaults.conservation_tolerance`.
- **`I4` (Reversibility)**, **`I5` (Additive Commutativity of `ACCUMULATE`)**, and **`I6` ($\Delta t$-Scaling of per-second rates)**.
- **`I7` (Disclosed Bounds Clipping)**: Float64 `clip_loss` is recorded per commit without silent mass destruction, while `_check_flux_whitelist` and `output_overlap_governance` (`MUTUALLY_EXCLUSIVE`, `PRIMARY_WRITER`) govern multi-writer access.

## 7. Code Mapping

| Module / Symbol | File Path | Architectural Role |
| --- | --- | --- |
| `SpatialScale`, `TemporalScale`, `VariableKind` | `relic/core/functor/functor_types.py` | Scale and role metadata enums |
| `DependencyGraphResolver` | `relic/core/dependency_graph.py` | Registration-indexed Kahn/NetworkX topological sort and `get_parallel_groups` |
| `determine_execution_group`, `resolve_scheduling_thresholds` | `relic/core/simulator/dependency_ops.py` | Rate-zone assignment against `computation.scheduling.*` |
| `configure_calendar_settings` | `relic/core/simulator/init_ops.py` | Simulator CF calendar and `time_origin` admission gate |
| `should_execute_medium`, `should_execute_slow`, `commit_rate_group_interval`, `check_event_triggers` | `relic/core/simulator/time_ops.py` | Transactional multi-rate boundary predicates and triggers |
| `step_simulation`, `_execute_fast_group`, `_compute_error_norm`, `_execute_fast_with_adaptive_retry` | `relic/core/simulator/execution_loop.py` | Multi-rate driver, Lie/Strang splitting, `multirate_exchanges`, and adaptive retry |
| `GroupComposition`, `merge_sequential_updates` | `relic/core/simulator/execution_group.py` | Substep/Strang update merging, boundary-rate finalization, and overlap checks |
| `check_scale_consistency`, `compute_group_execution_orders` | `relic/core/simulator/scale_ops.py` | Preorder scale-label check and per-zone sub-ordering |
| `TimeStagger` | `relic/coupling/protocol/time_stagger.py` | External NUOPC millisecond-LCM timestep negotiation |

## 8. Implementation Status

- **Implemented (`[I]`)**: Four-pillar bridge enforcement, deterministic registration-indexed DAG sorting, three-tier multi-rate scheduling in `elapsed seconds` with transactional interval commits and `multirate_exchanges` (`hold` and `interval_mean`), Lie and $2n - 1$ palindromic Strang splitting with `get_splitting_order_report()` and `GroupComposition` boundary-rate finalization, committed-change RMS adaptive step retry with full state/RNG/Brownian rollback, `get_composition_plan()` schedule witness, and `StateManager` `I1`–`I7` invariant governance.
- **Metadata-only / Design blueprint (`[M]` / `[N]`)**: Concurrent multi-Functor execution of DAG level sets $P_l$ (levels are reported for introspection and executed sequentially), embedded local-truncation-error or automatic CFL step-size control in the scheduler, automatic runtime checking of category-theoretic functoriality or fiber-bundle curvature (`relic/core/category_theory.py`), and calendar-period negotiation inside external `TimeStagger`.
- **IPHS Equivalence Certificates**: None (`0` certificates reference `spatiotemporal_scale_topology.md`).

## 9. References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. (2009). *Introduction to Algorithms* (3rd ed., Chapter 22). MIT Press.
2. Gear, C. W., and Wells, D. R. (1984). Multirate linear multistep methods. *BIT Numerical Mathematics*, 24(4), 484-502.
3. Hairer, E., Lubich, C., and Wanner, G. (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations* (2nd ed., Section II.5). Springer.
4. Hairer, E., and Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.). Springer.
5. Jones, P. W. (1999). First- and second-order conservative remapping schemes for grids in spherical coordinates. *Monthly Weather Review*, 127(9), 2204-2210.
6. Kahn, A. B. (1962). Topological sorting of large networks. *Communications of the ACM*, 5(11), 558-562.
7. Knoll, D. A., and Keyes, D. E. (2004). Jacobian-free Newton-Krylov methods: A survey of approaches and applications. *Journal of Computational Physics*, 193(2), 357-397.
8. Lawrence, D. M., Fisher, R. A., Koven, C. D., Oleson, K. W., Swenson, S. C., et al. (2019). The Community Land Model version 5: Description of new features, benchmarking, and impact of forcing uncertainty. *Journal of Advances in Modeling Earth Systems*, 11(12), 4245-4287.
9. Mac Lane, S. (1998). *Categories for the Working Mathematician* (2nd ed.). Springer.
10. Sportisse, B. (2000). An analysis of operator splitting techniques in the stiff case. *Journal of Computational Physics*, 161(1), 140-168.
11. Uruci, R. (2018). *Development and Testing of a Photochemical Model Based on the Master Chemical Mechanism* (Appendix A, Eqs. A.5-A.6).
