# Numerical Integration, Splitting, Stability, Scale Coupling, and Hybrid Events

## Method-of-lines formulation and solver registry

Spatial discretization and operator assembly reduce deterministic RELIC
processes to semi-discrete ordinary or differential-algebraic systems
(*Methods of Ecological Analysis*, Chapters 25, 26, and 28):

$$
\frac{\mathrm{d}x}{\mathrm{d}t}
= f(t, x; \boldsymbol{\theta})
= f_J(t, x) + f_M(t, x) + f_G(t, x) + f_N(t, x),
\qquad
x(t_0) = x_0,
$$

partitioning reversible ($f_J$), dissipative ($f_M$), boundary-port ($f_G$), and
reaction ($f_N$) tendencies. `relic.computation.solvers` is the single solver
owner (`SolverBase`, `SolverRegistry`, `SolverFactory`); scientific Functors
never instantiate private solver registries.

| Solver family | Live RELIC symbols (`relic.computation.solvers`) | Status | Order and stability contract |
| --- | --- | --- | --- |
| Explicit Runge-Kutta | `RK4Solver` (`temporal.py`) | `[I]` | Classical 4-stage order-4 explicit stepper on fixed time grids; conditionally stable |
| Damped fixed-point implicit Euler | `BackwardEulerSolver` (`temporal.py`) | `[I]` | Order-1 implicit Euler solved by damped Picard iteration ($\omega \in (0, 1]$, contraction limit $\Delta t\,L_f < 1$ at $\omega = 1$) |
| Symmetric midpoint & $\vartheta$-trapezoidal | `ImplicitMidpointSolver`, `CrankNicolsonSolver` (`temporal.py`), `transactional_implicit_midpoint_step` (`geometric.py`) | `[I]` | Order-2 symmetric midpoint and $\vartheta$-method ($\vartheta = 0.5$ default) solved by fixed-point iteration ($\frac{1}{2}\Delta t\,L_f < 1$ or $\vartheta\Delta t\,L_f < 1$) |
| Implicit-Explicit (IMEX) Euler | `IMEXEulerSolver` (`temporal.py`) | `[I]` | Order-1 explicit non-stiff + damped fixed-point implicit stiff step ($\Delta t\,L_{f_{\mathrm{imp}}} < 1$ at $\omega = 1$) |
| Tridiagonal direct linear solver | `ThomasAlgorithmSolver`, `solve_tridiagonal_thomas`, `solve_tridiagonal_batch` (`linear.py`) | `[I]` | $O(n)$ non-pivoting LU elimination with mandatory $\lvert p_i \rvert \le \epsilon_{\mathrm{mach}} s_i$ fail-fast guard and optional Varah condition estimate |
| Bracketed scalar root solver | `BracketedBisectionSolver` (`nonlinear.py`) | `[I]` | Guaranteed interval-halving root finder $g(z^\ast) = 0$ with implicit-function autograd backward rule |
| Multistep Backward Differentiation Formula (BDF), linearly implicit Rosenbrock-W, high-order IMEX Additive Runge-Kutta (IMEX-ARK), and Jacobian-free Newton-Krylov | Reference solver formulations (*Methods of Ecological Analysis*, Chapters 25--26) | `[N]` | Analytical stiff-ODE/PDE reference families; not registered in `relic.computation.solvers` (stiff layered PDEs use domain-specific `ThomasAlgorithmSolver` linearizations) |

## Deterministic time-stepping, linear, and nonlinear solvers

### Implemented one-step temporal integrators (`[I]`)

`RK4Solver` advances smooth non-stiff systems on fixed `float64` time grids via
four stage evaluations:

$$
k_1 = f(t_n, x_n), \quad
k_2 = f\!\left(t_n + \tfrac{1}{2}\Delta t,\; x_n + \tfrac{1}{2}\Delta t\,k_1\right), \quad
k_3 = f\!\left(t_n + \tfrac{1}{2}\Delta t,\; x_n + \tfrac{1}{2}\Delta t\,k_2\right), \quad
k_4 = f(t_n + \Delta t,\; x_n + \Delta t\,k_3),
$$

$$
x_{n+1} = x_n + \frac{\Delta t}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right).
$$

`BackwardEulerSolver`, `ImplicitMidpointSolver`, `CrankNicolsonSolver` (weight
$\vartheta = 0.5$ default), and `IMEXEulerSolver` ($f = f_{\mathrm{exp}} + f_{\mathrm{imp}}$)
discretize:

$$
x_{n+1} = x_n + \Delta t\,f(t_{n+1}, x_{n+1}) \;\; (\text{Backward Euler}),
\qquad
x_{n+1} = x_n + \Delta t\,f\!\left(t_n + \tfrac{1}{2}\Delta t,\; \tfrac{1}{2}(x_n + x_{n+1})\right) \;\; (\text{Implicit Midpoint}),
$$

$$
x_{n+1} = x_n + \Delta t\left[(1 - \vartheta)f(t_n, x_n) + \vartheta f(t_{n+1}, x_{n+1})\right] \;\; (\text{Crank-Nicolson } \vartheta),
$$

$$
x_{n+1} = x_n + \Delta t\,f_{\mathrm{exp}}(t_n, x_n) + \Delta t\,f_{\mathrm{imp}}(t_{n+1}, x_{n+1}) \;\; (\text{IMEX Euler}).
$$

All four implicit solvers iterate fixed-point updates (damped Picard
$x^{(k+1)} = \omega G(x^{(k)}) + (1 - \omega)x^{(k)}$ with default
`damping_factor` $\omega = 1.0$ and `max_iterations = 50` in Backward Euler and
IMEX Euler; undamped fixed-point in Implicit Midpoint and Crank-Nicolson)
without internal Newton-Raphson Jacobians. Consequently, A-stability of the
underlying quadrature does not lift the Picard contraction restriction
($\Delta t\,L_f < 1$, $\frac{1}{2}\Delta t\,L_f < 1$, or
$\vartheta\Delta t\,L_f < 1$). Finite non-convergent iterates return
`SolverStatus.MAX_ITERATIONS` from `solve()` and raise `SolverStepError` in
`step()` / `forward()`; non-finite iterates raise `ValueError` immediately.
Standalone solvers perform fixed-grid stepping with no internal adaptive step
resizing.

For discrete-gradient structure preservation on energy $E$ and entropy $S$ with
$\overline{\nabla}F(x_n, x_{n+1})^{\mathsf{T}}(x_{n+1} - x_n) = F(x_{n+1}) - F(x_n)$,
skew $J_h^{\mathsf{T}} = -J_h$ preserves energy
$E(x_{n+1}) - E(x_n) = \Delta t\,\overline{\nabla}E^{\mathsf{T}} J_h \overline{\nabla}E = 0$,
and $M_h \succeq 0$ guarantees discrete entropy production
$S(x_{n+1}) - S(x_n) = \Delta t\,\overline{\nabla}S^{\mathsf{T}} M_h \overline{\nabla}S \ge 0$
when functional, discrete gradient, and accepted step are explicitly bound.

### Reference stiff solver families (`[N]`)

*Methods of Ecological Analysis* (Chapters 25--26) derives four stiff
Jacobian-based time-integration families as analytical benchmarks (`[N]` in
`relic.computation.solvers`):

1. **Variable-order BDF ($k = 1, \dots, 5$)**: $\sum_{j=0}^k \alpha_j x_{n+1-j} = \Delta t\,\beta_0 f(t_{n+1}, x_{n+1})$, $A(\alpha)$-stable for stiff reaction-DAE kinetics;
2. **Linearly implicit Rosenbrock-W**: $(I - \gamma \Delta t J_n) k_i = f(\dots) + \Delta t J_n \sum_{j < i} \gamma_{ij} k_j$ ($J_n \approx \partial f / \partial x$), avoiding nonlinear Newton loops via one linear solve per stage;
3. **High-order IMEX-ARK**: paired explicit/diagonally-implicit Butcher tableaux $(\tilde{A}, \tilde{b}, \tilde{c})$ and $(A, b, c)$ treating advection/reactions explicitly and stiff vertical diffusion implicitly beyond order 1; and
4. **Jacobian-free Newton-Krylov (JFNK)**: inner GMRES/BiCGSTAB solves approximating directional derivatives $J(x)v \approx [f(x + \varepsilon v) - f(x)]/\varepsilon$ for large coupled nonlinear residuals.

### Tridiagonal linear and bracketed nonlinear solvers (`[I]`)

1D vertical diffusion and linearized Richards discretizations assemble
tridiagonal systems $A x = d$ with lower, main, and upper diagonals
$(a_i, b_i, c_i)$. `ThomasAlgorithmSolver` and `solve_tridiagonal_thomas`
(`relic.computation.solvers.linear`) execute forward elimination and back
substitution:

$$
p_i = b_i - a_i c_{i-1}' \;\; (p_0 = b_0),
\qquad
c_i' = \frac{c_i}{p_i},
\qquad
d_i' = \frac{d_i - a_i d_{i-1}'}{p_i},
\qquad
x_n = d_n', \quad x_i = d_i' - c_i' x_{i+1}.
$$

Any non-finite pivot or near-zero pivot satisfying
$\lvert p_i \rvert \le \epsilon_{\mathrm{mach}} s_i$ (where $s_i$ is row $i$'s
$\ell_\infty$ norm) raises `ValueError` without pivot regularization. Enabling
`check_stability` (`computation.solvers.thomas.check_stability`) records the
Varah (1975) condition-number bound in `SolverResult` metadata.

`BracketedBisectionSolver` (`relic.computation.solvers.nonlinear`) solves scalar
algebraic constraints $g(z^\ast; \boldsymbol{\theta}) = 0$ on sign-change
brackets $[a, b]$ ($g(a)g(b) \le 0$) and propagates exact implicit-function
sensitivities through `_ImplicitRootGradient`:

$$
\frac{\partial z^\ast}{\partial \boldsymbol{\theta}}
= -\left(\frac{\partial g(z^\ast, \boldsymbol{\theta})}{\partial z}\right)^{-1}
\frac{\partial g(z^\ast, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}.
$$

## Multi-rate execution, operator splitting, and step control

### Palindromic Strang splitting and `SubFlowAccuracy`

`RELICSimulator` partitions registered Functors into `fast`, `medium`, and
`slow` groups (`relic.core.simulator.execution_group`,
`relic.core.simulator.execution_loop`). For $n \ge 2$ ordered fast-group
processes $F_1, \dots, F_n$, `lie` mode executes $\Phi_n^{\Delta t} \circ \cdots \circ \Phi_1^{\Delta t}$, whereas `strang` mode executes the palindromic
$2n - 1$ composition (*Methods of Ecological Analysis*, Chapter 26):

$$
\Phi_{\Delta t}^{\mathrm{Strang}}
= \Phi_1^{\Delta t / 2} \circ \cdots \circ \Phi_{n-1}^{\Delta t / 2}
\circ \Phi_n^{\Delta t}
\circ \Phi_{n-1}^{\Delta t / 2} \circ \cdots \circ \Phi_1^{\Delta t / 2},
$$

passing stage start time $t$ to the leading half-steps and central full step and
$t + \Delta t / 2$ to the trailing half-steps. Each Functor declares
`FunctorMetadata.sub_flow_accuracy` (`SubFlowAccuracy` in
`relic.core.functor.functor_types`: `EXACT` with $p_i = \infty$, `ORDER_2` with
$p_i = 2$, `ORDER_1` with $p_i = 1$, or `UNDECLARED`), and
`RELICSimulator.get_splitting_order_report()` evaluates the guaranteed one-step
order:

$$
q = \min\!\left(s,\; \min_{1 \le i \le n} p_i\right),
\qquad
s = \begin{cases} 1, & \texttt{lie}, \\ 2, & \texttt{strang}, \end{cases}
$$

reporting `unverified` whenever any participating process leaves
`SubFlowAccuracy.UNDECLARED`. Strang composition with first-order explicit Euler
sub-integrators reduces globally to $q = 1$. In Strang mode with $n \ge 2$,
State Contract Model (SCM) flux edges between stateful fast members (including
stateless relays) are refused at provider publication; `IntegrationPlan`
(`relic.computation.solvers.geometric`) validates static stage DAGs as
`metadata_only` (`verified=False`, `runtime_ready=False`), while runtime
schedule conformance is verified via `RELICSimulator.get_composition_plan()` and
`GraphExecutor.integration_plan_witness`.

### Multi-rate effective interval and boundary-rate integrals

With base fast step $\Delta t_f$, the `medium` group triggers across multiples
of `medium_interval_seconds`, and the `slow` group triggers across multiples of
`slow_interval_seconds` (or calendar month boundaries when unset), receiving the
exact elapsed interval since its prior execution:

$$
\Delta t_{\mathrm{eff}} = t_{\mathrm{future}} - t_{\mathrm{last}},
\qquad
\Delta t_{\mathrm{eff}} > 0.
$$

Each group execution (and each Strang sub-execution) commits immediately through
`StateManager`. Across substeps $k$ of duration $h_k$ within call interval
$H = \sum_k h_k$, `ACCUMULATE` amounts sum directly, `ACCUMULATE` rates
(`per_timestep=False`) form time-weighted averages $\sum_k (h_k / H) q_k$, and
substepped physical boundary-port `REPLACE` rates (`allowed_in` / `allowed_out`)
commit the exact interval mean $\sum_k q_k \Delta t_k / \Delta t_{\mathrm{call}}$.
Declared `multirate_exchanges` supply either `hold` or piecewise-constant
running `interval_mean` overlays (`exchange_mean__<id>`) to slower consumers.

### RMS update-magnitude step controller

`RELICSimulator.run(...)` adapts macro-steps exclusively via the fast-group
committed root-mean-square (RMS) update metric over the $m$ distinct state
variables written by the fast group ($u_k = x'_k - x_k$):

$$
\tau_k = \mathrm{atol} + \mathrm{rtol}\,\lvert x_k \rvert,
\qquad
\mathrm{Err}
= \sqrt{\frac{1}{m}\sum_{k=1}^{m}
\operatorname{mean}\!\left(\frac{u_k}{\tau_k}\right)^2}.
$$

If $\mathrm{Err} > 1$ and $\Delta t > \Delta t_{\min}$, the step rolls back and
retries once with
$\Delta t_{\mathrm{new}} = \max\!\left(\Delta t_{\min},\; \Delta t\max(0.1, 0.9 / \sqrt{\mathrm{Err}})\right)$;
accepted steps propose
$\Delta t_{\mathrm{next}} = \operatorname{clip}\!\left(\Delta t\min(5, \max(0.1, 0.9 / \sqrt{\mathrm{Err}})),\; \Delta t_{\min},\; \Delta t_{\max}\right)$
(doubling when $\mathrm{Err} = 0$). This controller bounds relative state jumps
and is distinct from formal embedded truncation-error or Courant-Friedrichs-Lewy
(CFL) stability control.

## Transport stability and nonlinear Richards closure

For 1D/multi-D advection-diffusion
$\frac{\partial c}{\partial t} + \nabla\cdot(\mathbf{u}c - D\nabla c) = 0$ on
isotropic grid spacing $\Delta x$ in $d$ spatial dimensions, explicit time
stepping is constrained by the Courant-Friedrichs-Lewy (CFL) number, explicit
Fourier diffusion number $\mu$, and cell Peclet number $\mathrm{Pe}$ (*Methods
of Ecological Analysis*, Chapter 25):

$$
\mathrm{CFL} = \frac{\lvert u \rvert\,\Delta t}{\Delta x} \le C_{\max},
\qquad
\mu = \frac{D\,\Delta t}{\Delta x^2} \le \frac{1}{2d},
\qquad
\mathrm{Pe} = \frac{\lvert u \rvert\,\Delta x}{D}.
$$

`apply_advection` (`relic.scientific.transport`) enforces
$\mathrm{CFL} \le C_{\max}$ against `transport.numerical.cfl_advection_max`
(default `1.0`, bounds `[0.5, 1.0]`) on explicit `"euler"` steps and raises
`ValueError` on violation (`upwind` supports all integrators; `lax_wendroff`,
`superbee`, and `van_leer` require `"euler"`; `central` requires
`"implicit_midpoint"` or `"crank_nicolson"`).

Variably saturated vertical soil-water flow obeys the mixed-form Richards
equation with van Genuchten (1980) retention and Mualem (1976) hydraulic
conductivity:

$$
\frac{\partial \theta(h)}{\partial t}
= \frac{\partial}{\partial z}\!\left[K(h)\left(\frac{\partial h}{\partial z} + 1\right)\right] - S(h),
\qquad
S_e = \left[1 + (\alpha \lvert h \rvert)^n\right]^{-m} \;\; \left(m = 1 - \frac{1}{n}\right),
$$

$$
\theta(h) = \theta_r + S_e(\theta_s - \theta_r),
\qquad
K(S_e) = K_s S_e^\ell\left[1 - \left(1 - S_e^{1/m}\right)^m\right]^2,
$$

routing any saturation or residual-moisture clipping
($\theta_r \le \theta \le \theta_s$) into explicit water-budget ledgers.

## Conservative spatial remapping and internal cross-domain couplers

### Basis-governed spatial remapping

Grid-to-grid (`ConservativeRemap`, `ConservativeRegridder`, `BilinearRegridder`,
`NearestNeighborRegridder` in `relic.coupling.scales.spatial`) and nested-unit
hierarchical operators (`HierarchicalAggregator`, `ScaleManager` in
`relic.coupling.scales`) enforce `SpatialQuantityBasis` (`relic.coupling.scales.types`:
`intensive`, `cell_total`, `area_density`, `categorical`; see [Mathematical Foundations](foundations.md)).
Spherical `latlon` cell areas use SSOT `earth_radius` $R_\oplus$ as
$A = R_\oplus^2(\lambda_2 - \lambda_1)(\sin\phi_2 - \sin\phi_1)$, and
`HierarchicalAggregator` additionally supports neutral-drag effective roughness
blending at reference height $z_r$ ($0 < z_{0,i} < z_r$):

$$
z_{0,j} = z_r \exp\!\left[-\left(\sum_i \frac{f_{ij}}{\ln^2(z_r / z_{0,i})}\right)^{-1/2}\right].
$$

### Internal cross-domain coupling ports (`relic.coupling.internal`)

Three `ScaleCouplingFunctor` ports connect soil, vegetation, and surface-layer
variables without duplicating domain process laws (*Methods of Ecological
Analysis*, Chapters 23--24):

1. **`SoilVegetationCoupling` (`relic.coupling.internal.soil_vegetation`)**:
   Normalizes parametric (`exponential`, `linear`, `beta`) or observed root
   density across layer thicknesses $\Delta z_i$ ($\sum_i \beta_i \Delta z_i = 1$)
   and applies the four-threshold Feddes matric-potential stress factor
   $\alpha(\psi_i) \in [0, 1]$ ($\psi_{\mathrm{wilt}} < \psi_{\mathrm{opt,dry}} < \psi_{\mathrm{opt,wet}} < \psi_{\mathrm{anaer}}$)
   to potential transpiration $T_p$:

$$
S_i = \alpha(\psi_i)\,\beta_i\,T_p,
\qquad
T_{\mathrm{actual}} = \sum_i S_i\,\Delta z_i \le T_p.
$$

2. **`BiosphereAtmosphereCoupling` (`relic.coupling.internal.biosphere_atmosphere`)**:
   Maps friction velocity $u_\ast > 0$, air temperature $T > 0$, and pressure
   $P > 0$ to bulk-canopy leaf boundary-layer conductance:

$$
g_{b,\mathrm{ms}} = C_v\sqrt{\frac{u_\ast}{d_{\mathrm{leaf}}}},
\qquad
g_{b,\mathrm{mol}} = g_{b,\mathrm{ms}}\frac{P}{R T}.
$$

3. **`WaterCarbonCoupling` (`relic.coupling.internal.water_carbon`)**:
   Combines soil-moisture factor $f_{\mathrm{soil}}(\theta)$ (`logistic`,
   `linear`, or `piecewise`), hyperbolic vapour-pressure-deficit factor
   $f_{\mathrm{vpd}} = (1 + D / D_{50})^{-1}$, and clamped $Q_{10}$ thermal
   response $f_T$ into bounded physiological and decomposition modifiers
   ($f_{\mathrm{gpp}}$, $f_{Ra}$, $f_{Rh}$, $k_{\mathrm{dec}}$, and
   $f_{\mathrm{overall}}$).

## Transverse hybrid events and saltation-matrix sensitivities

For one transversal zero crossing of a scalar guard $g(x, \boldsymbol{\theta}, t) = 0$
within an explicit step $[t_0, t_0 + \Delta t]$ ($g_0 g_1 \le 0$) followed by a
smooth reset map $x^+ = R(x^-, \boldsymbol{\theta}, \tau)$,
`relic.computation.solvers.hybrid_events` (`locate_guard_crossing`,
`event_time_sensitivity`, `sensitivity_jump`, `saltation_matrix`) locates the
crossing time $\tau$ by linear interpolation and propagates exact first-order
state and parameter sensitivities (*Methods of Ecological Analysis*, Chapter 28):

$$
\tau = t_0 + \Delta t\,\frac{g_0}{g_0 - g_1},
\qquad
\frac{\mathrm{d}\tau}{\mathrm{d}\boldsymbol{\theta}}
= -\frac{g_x S^- + g_{\boldsymbol{\theta}}}{g_x f^- + g_t},
$$

$$
\Xi = R_x + \frac{\left(f^+ - R_x f^- - R_t\right) n^{\mathsf{T}}}{n^{\mathsf{T}} f^- + g_t}
\quad \left(n = g_x^{\mathsf{T}}\right),
\qquad
S^+ = R_x S^- + R_{\boldsymbol{\theta}}
+ \left(R_x f^- + R_t - f^+\right)\frac{\mathrm{d}\tau}{\mathrm{d}\boldsymbol{\theta}},
$$

where $f^-$ and $f^+$ are pre- and post-event vector fields and $S^-$, $S^+$ are
pre- and post-event parameter sensitivities. Tangential or grazing trajectories
satisfying $\lvert g_0 - g_1 \rvert \le \varepsilon_{\mathrm{trans}}$ or
$\lvert n^{\mathsf{T}} f^- + g_t \rvert \le \varepsilon_{\mathrm{trans}}$
(`computation.solvers.hybrid_events.transversality_tolerance`, default `1e-12`)
raise `ValueError` without extrapolation. Production consumer: `GDDPhenology`
onset diagnostics; simultaneous multi-guard crossings and implicit-solver event
location remain `[N]`.

## Stochastic, memory, ALE, DAE, and topology-transition families

`relic.stochastic` provides standalone Ito Stochastic Differential Equation
(SDE) Euler-Maruyama stepping
($x_{n+1} = x_n + a(x_n, t_n)\Delta t + B(x_n, t_n)\Delta W_n$,
$\Delta W_n \sim \mathcal{N}(0, \Delta t\,I)$, strong order $1/2$), Gillespie
direct Stochastic Simulation Algorithm (SSA) reaction jumps
($\tau = -\ln u_1 / a_0(x)$, $x^+ = x^- + \nu_j$), and Monte Carlo ensemble
statistics ($\bar{x} = \frac{1}{M}\sum_{m=1}^M x^{(m)}$,
$s_x^2 = \frac{1}{M-1}\sum_{m=1}^M (x^{(m)} - \bar{x})^2$). Conservative
stochastic forcing acts on flux cochains
($\mathrm{d}x_{\mathrm{noise}} = -D_{d-1,h}\Xi_h\,\mathrm{d}W_t$) so closed
elemental ledgers remain invariant.

Across the certified scientific Functors, six bounded non-standard dynamical
families are wired into `FunctorBase`, `StateManager`, and checkpoint/restart
governance (`[I]`):

| Semantic family | Governed production Functor(s) | Mathematical realization and scope boundary |
| --- | --- | --- |
| `continuous_sde` | `OrnsteinUhlenbeckTemperatureFunctor` (`Level 2`) | Euler-Maruyama Ornstein-Uhlenbeck anomaly step with deterministic per-Functor substream checkpoint/restart replay (`FunctorSubstreamMixin`) |
| `marked_jump` | `MarkedPoissonRainfallFunctor` (`Level 2`) | Constant-hazard Poisson storm arrivals with exponential depth marks and deterministic substream replay; steps exceeding arrival cap fail closed without truncation |
| `memory` | `TwoRegionMassTransferFunctor` (`Level 3`) | Exact exponential integrating-factor realization (`SubFlowAccuracy.EXACT`) of mobile-immobile Volterra memory exchange $f_{\mathrm{mem}}(t) = -\int_0^t K(t-s)e(s)\,\mathrm{d}s$ |
| `ale` | `MovingDomainDiffusionFunctor` (`Level 3`) | 1D Arbitrary Lagrangian-Eulerian (ALE) uniform column stretching satisfying the Geometric Conservation Law (GCL) and moving-metric power identity $e_E^{\mathsf{T}} f_{\mathrm{ALE}} = -\frac{1}{2}x^{\mathsf{T}}\dot{\star}_h x$ |
| `descriptor_dae` | `EquilibriumSorptionFunctor` (`Level 2`) | Total-mass prognostic advance paired with index-1 Freundlich Differential-Algebraic Equation (DAE) scalar root solve via `BracketedBisectionSolver` |
| `topology_change` | `HortonianPondingEventFunctor` & `SurfacePondingFunctor` (`Level 2`) | Guard-triggered ponding topology registration with affine state remap, water-budget closure, and single provider-epoch transition |

## References

- Butcher, J. C. (2016). *Numerical Methods for Ordinary Differential Equations*, 3rd ed. Wiley.
- Feddes, R. A., Kowalik, P. J., and Zaradny, H. (1978). *Simulation of Field Water Use and Crop Yield*. Pudoc.
- Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. *Journal of Physical Chemistry* 81, 2340--2361.
- Lygeros, J., Johansson, K. H., Simic, S. N., Zhang, J., and Sastry, S. S. (2003). Dynamical properties of hybrid automata. *IEEE Transactions on Automatic Control* 48, 2--17.
- Hairer, E., Lubich, C., and Wanner, G. (2006). *Geometric Numerical Integration*, 2nd ed. Springer.
- Hairer, E. and Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Springer.
- Hiskens, I. A. and Pai, M. A. (2000). Trajectory sensitivity analysis of hybrid systems. *IEEE Transactions on Circuits and Systems I* 47, 204--220.
- Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. *Water Resources Research* 12, 513--522.
- Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (1992). *Numerical Recipes in C*, 2nd ed. Cambridge University Press.
- Strang, G. (1968). On the construction and comparison of difference schemes. *SIAM Journal on Numerical Analysis* 5, 506--517.
- van Genuchten, M. T. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. *Soil Science Society of America Journal* 44, 892--898.
- Varah, J. M. (1975). A lower bound for the smallest singular value of a matrix. *Linear Algebra and its Applications* 11, 3--5.
