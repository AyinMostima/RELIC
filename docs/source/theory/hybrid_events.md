# Hybrid Events: Guard Crossing, Event-Time Sensitivity, and Saltation

## 1. Process Definition and Scope

`relic/computation/solvers/hybrid_events.py` implements the bounded numerical helper for state-triggered threshold crossings inside a single explicit RELIC time step. It provides four pure tensor functions (`locate_guard_crossing`, `event_time_sensitivity`, `sensitivity_jump`, and `saltation_matrix`) that evaluate crossing times and post-event trajectory sensitivities across transversal guard surfaces. Theoretical derivations for hybrid-event differentiability and saltation operators appear in *Methods of Ecological Analysis*, Chapter 28.

The module operates under four bounded conditions:
1. A single scalar smooth guard $g(x, \theta, t)$ changes sign at most once inside an explicit step $[t_n, t_n + \Delta t]$.
2. The crossing satisfies strict transversality, $\lvert g_x f^- + g_t \rvert > \varepsilon_{\mathrm{trans}}$ and $\lvert g_{\mathrm{start}} - g_{\mathrm{end}} \rvert > \varepsilon_{\mathrm{trans}}$.
3. Pre-event vector field $f^-$, post-event vector field $f^+$, and reset map $R(x^-, \theta, \tau)$ are differentiable in a neighborhood of the crossing.
4. Multiple events inside a single step are isolated and have fixed ordering.

The helper is solver-layer infrastructure: it is not a `FunctorBase` subclass, does not schedule processes, and does not mutate `StateManager` storage. Its production scientific consumer is `GDDPhenology` (`relic/scientific/biosphere/phenology/models.py`, theory page [phenology.md](phenology.md)), which emits analytic onset-time and leaf-area sensitivity diagnostics when thermal-time accumulation crosses `GDD_start`.

## 2. Governing Equations

### 2.1 Implicit Event Time and Parameter Sensitivity

Along a trajectory $x(t, \theta)$ parameterized by $\theta$, an event time $\tau(\theta)$ is defined implicitly by the zero level set $g(x(\tau(\theta), \theta), \theta, \tau(\theta)) = 0$. Total differentiation with respect to $\theta$ yields the implicit event-time sensitivity (Hiskens and Pai, 2000; Goebel, Sanfelice, and Teel, 2012; Kong et al., 2023):

$$
\frac{d\tau}{d\theta} = -\frac{g_x S^- + g_\theta}{g_x f^- + g_t}, \qquad S^- = \left(\frac{\partial x}{\partial \theta}\right)_{\tau^-},
$$

subject to the transversality condition $g_x f^- + g_t \neq 0$.

### 2.2 Reset Map, Parameter Sensitivity Jump, and State Saltation Matrix

Across a state reset $x^+ = R(x^-, \theta, \tau)$, differentiating $x(\tau^+, \theta)$ accounts for both the direct reset derivatives and the vector-field mismatch evaluated over the shifted switching instant:

$$
S^+ = R_x S^- + R_\theta + \left(R_x f^- + R_t - f^+\right)\frac{d\tau}{d\theta}.
$$

For state perturbations $\delta x^-$ propagating across the guard manifold with normal column vector $n = g_x^{\mathsf{T}} \in \mathbb{R}^d$ and square reset Jacobian $R_x \in \mathbb{R}^{d \times d}$, the saltation matrix $\mathcal{S} \in \mathbb{R}^{d \times d}$ satisfying $\delta x^+ = \mathcal{S}\,\delta x^-$ is (di Bernardo et al., 2008; Kong et al., 2023):

$$
\mathcal{S} = R_x + \frac{\left(f^+ - R_x f^- - R_t\right) n^{\mathsf{T}}}{n^{\mathsf{T}} f^- + g_t}.
$$

Omitting the event-time term $(R_x f^- + R_t - f^+)\,d\tau/d\theta$ yields an inconsistent sensitivity even when the state itself is continuous ($R = \mathrm{id}$). For the benchmark scalar system $x(0) = \theta$, guard $g(x) = x - 1$, $R(x) = x$, $f^- = 1$, $f^+ = 2$, and terminal time $T = 2$ near $\theta = 0$, the exact terminal derivative is $\partial x(2)/\partial \theta = 2$, whereas dropping the event-time jump term produces $1$.

### 2.3 Explicit-Step Guard Location and GDD Phenology Consumer

Over one explicit step $[t_n, t_n + \Delta t]$ with $\Delta t > 0$ and bracketing guard values $g_0 = g(t_n)$, $g_1 = g(t_n + \Delta t)$ satisfying $g_0 g_1 \le 0$ and $\lvert g_0 - g_1 \rvert > \varepsilon_{\mathrm{trans}}$, `locate_guard_crossing` locates the crossing by linear interpolation (Shampine and Thompson, 2000):

$$
\tau = t_n + \Delta t\,\frac{g_0}{g_0 - g_1}.
$$

This relation is exact whenever $g(t)$ is affine over $[t_n, t_n + \Delta t]$ and preserves PyTorch reverse-mode differentiation through $g_0$ and $g_1$.

In `GDDPhenology`, daily growing degree-days accumulate as $\mathrm{GDD}_{n+1} = \mathrm{GDD}_n + \mathrm{GDD}_d\,\Delta t / t_{\mathrm{day}}$ ($t_{\mathrm{day}} = 86400\,\mathrm{s}$; McMaster and Wilhelm, 1997). Dormancy switches to leaf expansion at $g = \mathrm{GDD} - \mathrm{GDD}_{\mathrm{start}} = 0$. For parameter $\theta = \mathrm{GDD}_{\mathrm{start}}$, $g_x = 1$, $g_\theta = -1$, $g_t = 0$, $S^- = 0$, and $f^- = \mathrm{GDD}_d / t_{\mathrm{day}}$, giving

$$
\frac{d\tau}{d\,\mathrm{GDD}_{\mathrm{start}}} = \frac{t_{\mathrm{day}}}{\mathrm{GDD}_d}.
$$

Because leaf area index is continuous across onset ($R = \mathrm{id}$, $R_x = 1$, $R_\theta = 0$, $R_t = 0$) with pre-onset rate $f_{\mathrm{LAI}}^- = 0$ and linear expansion rate $f_{\mathrm{LAI}}^+ = (\mathrm{LAI}_{\max} - \mathrm{LAI}_{\min})(\mathrm{GDD}_d / t_{\mathrm{day}})/\Delta\mathrm{GDD}_{\mathrm{full}}$ (reducing to $\mathrm{LAI}_{\max}(\mathrm{GDD}_d / t_{\mathrm{day}})/\Delta\mathrm{GDD}_{\mathrm{full}}$ when $\mathrm{LAI}_{\min} = 0$), the post-onset LAI sensitivity jump with respect to $\mathrm{GDD}_{\mathrm{start}}$ is

$$
S_{\mathrm{LAI}}^+ - S_{\mathrm{LAI}}^- = \left(f_{\mathrm{LAI}}^- - f_{\mathrm{LAI}}^+\right)\frac{d\tau}{d\,\mathrm{GDD}_{\mathrm{start}}} = -\frac{\mathrm{LAI}_{\max} - \mathrm{LAI}_{\min}}{\Delta\mathrm{GDD}_{\mathrm{full}}}.
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $g_0, g_1$ | Step-start and step-end guard evaluations | `[guard]` (GDD: $\mathrm{^\circ C\,d}$) | `g_start`, `g_end` | Function input |
| $t_n, \Delta t, \tau$ | Step start time, step length, and located crossing time | $\mathrm{s}$ | `t_start`, `dt`, return of `locate_guard_crossing` | Function input / output |
| $g_x, g_\theta, g_t$ | Partial derivatives of guard $g(x, \theta, t)$ | `[guard]/[state]`, `[guard]/[\theta]`, `[guard] s-1` | `g_x` (`normal`), `g_theta`, `g_t` | Function input |
| $f^-, f^+$ | Pre-event and post-event vector fields | `[state] s-1` | `f_minus`, `f_plus` | Function input |
| $R_x, R_\theta, R_t$ | Partial derivatives of reset map $R(x^-, \theta, \tau)$ | $1$, `[state]/[\theta]`, `[state] s-1` | `r_x`, `r_theta`, `r_t` | Function input |
| $S^-, S^+$ | Pre-event and post-event parameter sensitivities $\partial x / \partial \theta$ | `[state]/[\theta]` | `s_minus`, return of `sensitivity_jump` | Function input / output |
| $\mathcal{S}$ | State perturbation saltation matrix | $1$ | return of `saltation_matrix` | Function output |
| $\chi_{\mathrm{onset}}$ | Binary onset-crossing indicator in `GDDPhenology` | $1$ | `diagnostics.phenology.onset_event_flag` | `diagnostic` (`DERIVED`) |
| $\tau - t_n$ | Within-step onset time offset ($0$ without crossing) | $\mathrm{s}$ | `diagnostics.phenology.onset_event_time_offset` | `diagnostic` (`DERIVED`) |
| $d\tau / d\mathrm{GDD}_{\mathrm{start}}$ | Onset event-time sensitivity ($0$ without crossing) | $\mathrm{s\,^\circ C^{-1}\,d^{-1}}$ | `diagnostics.phenology.onset_event_time_sensitivity_gdd_start` | `diagnostic` (`DERIVED`) |
| $\Delta S_{\mathrm{LAI}}$ | Post-onset LAI sensitivity jump ($0$ without crossing) | $\mathrm{m^2\,m^{-2}\,^\circ C^{-1}\,d^{-1}}$ | `diagnostics.phenology.lai_sensitivity_jump_gdd_start` | `diagnostic` (`DERIVED`) |

## 4. Parameter Table

| Symbol | `ParameterManager` / Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $\varepsilon_{\mathrm{trans}}$ | `computation.solvers.hybrid_events.transversality_tolerance` | `[guard]` or `[guard] s-1` (`dimensionless` schema) | `relic/parameters/database/relic/computation/solvers.yaml` (`1.0e-12`, bounds $[10^{-300}, 10^{-3}]$) | Hiskens and Pai (2000); Shampine and Thompson (2000) |
| $T_{\mathrm{freeze}}$ | `relic.physical.constants.T_FREEZE` (`get_physical_constant("T_FREEZE")`) | $\mathrm{K}$ | `relic/parameters/database/relic/physical/constants.yaml` | CODATA / SI water freezing point |
| $t_{\mathrm{day}}$ | `SECONDS_PER_DAY` (`relic/utils/time_constants.py`) | $\mathrm{s\,d^{-1}}$ | Canonical time constant (`86400.0`) | SI day conversion |

## 5. Numerical Methods and Discretization

All four functions coerce numeric or tensor arguments to floating-point `torch.Tensor` instances on the primary input's `dtype` and `device`, rejecting non-floating types (`TypeError`) and non-finite entries (`ValueError`). When `transversality_tol=None`, `_require_transversal` queries `ParameterManager` for `computation.solvers.hybrid_events.transversality_tolerance` with `required=True`; no hardcoded Python fallback exists.

- `locate_guard_crossing` rejects $\Delta t \le 0$, non-bracketing pairs ($g_0 g_1 > 0$), and tangential differences ($\lvert g_0 - g_1 \rvert \le \varepsilon_{\mathrm{trans}}$) with `ValueError`; no extrapolation outside $[t_n, t_n + \Delta t]$ is performed.
- `event_time_sensitivity` and `sensitivity_jump` operate elementwise on scalar or batched scalar states.
- `saltation_matrix` validates that `r_x` is a rank-2 square $(d, d)$ matrix and `f_minus`, `f_plus`, `r_t`, `normal` are rank-1 vectors of length $d$, rejecting $\lvert n^{\mathsf{T}} f^- + g_t \rvert \le \varepsilon_{\mathrm{trans}}$ before forming `torch.outer(fp - rx @ fm - rt, nvec) / denominator`.

## 6. Conservation, Boundary Conditions, and Ledgers

`hybrid_events.py` performs pure analytical sensitivity evaluation and introduces no mass, energy, or stoichiometric source or sink. In `GDDPhenology`, the four onset-sensitivity outputs are emitted strictly as read-only diagnostics (`StateWriteMode.DERIVED`) and leave prognostic thermal-time accumulation and leaf-area trajectories unchanged. Any reset map $R$ supplied by a caller must independently satisfy the caller's physical conservation ledger $C R(x^-, \theta, \tau) = C x^-$.

## 7. Code Mapping

| Symbol / Class | File Path | Architectural Role | Production Consumers / Outputs |
| --- | --- | --- | --- |
| `locate_guard_crossing`, `event_time_sensitivity`, `sensitivity_jump`, `saltation_matrix` | `relic/computation/solvers/hybrid_events.py` | Solver-layer hybrid sensitivity helper (not a Functor) | Consumed by `GDDPhenology`; pure tensor sensitivity operators |
| `GDDPhenology` | `relic/scientific/biosphere/phenology/models.py` | `BiologicalProcess` via `BiologicalProcessFunctor` (`ExecutionPhase.MAIN_PROCESSES`) | Emits `diagnostics.phenology.onset_event_*` and `lai_sensitivity_jump_gdd_start` |

## 8. Implementation Status

- **Implemented (`[I]`)**: Single-crossing explicit-step linear guard location, implicit event-time parameter sensitivity, scalar/elementwise post-event sensitivity jump, full matrix state-perturbation saltation operator, `ParameterManager`-backed grazing refusal, and `GDDPhenology` onset diagnostic emission.
- **Not implemented (`[N]`)**: Tangential or grazing bifurcation analysis ($\lvert g_x f^- + g_t \rvert \to 0$), simultaneous multi-surface corner collisions, root-finding event location embedded inside implicit ODE/DAE steppers, and an automatic graph-wide saltation wrapper across arbitrary Functors.
- **IPHS Equivalence Certificates**: None (`0` certificates reference `hybrid_events.md`, as this page documents solver-layer sensitivity infrastructure; `PhenologyFunctor` / `GDDPhenology` owns its process certificate on [phenology.md](phenology.md)).

## 9. References

1. di Bernardo, M., Budd, C. J., Champneys, A. R., and Kowalczyk, P. (2008). *Piecewise-Smooth Dynamical Systems: Theory and Applications*. Springer.
2. Goebel, R., Sanfelice, R. G., and Teel, A. R. (2012). *Hybrid Dynamical Systems: Modeling, Stability, and Robustness*. Princeton University Press.
3. Hiskens, I. A., and Pai, M. A. (2000). Trajectory sensitivity analysis of hybrid systems. *IEEE Transactions on Circuits and Systems I: Fundamental Theory and Applications*, 47(2), 204-220.
4. Kong, N. J., Payne, J. J., Zhu, J., and Johnson, A. M. (2023). Saltation matrices: The essential tool for linearizing hybrid dynamical systems. *arXiv:2306.06862*.
5. McMaster, G. S., and Wilhelm, W. W. (1997). Growing degree-days: One equation, two interpretations. *Agricultural and Forest Meteorology*, 87(4), 291-300.
6. Shampine, L. F., and Thompson, S. (2000). Event location for ordinary differential equations. *Computers & Mathematics with Applications*, 39(5-6), 43-54.
