# Numerical Time Integration and Solver Infrastructure

## 1. Process Definition and Scope

`relic/computation/solvers/` provides the unified solver infrastructure (`SolverBase`, `SolverMetadata`, `SolverResult`, `SolverStatus`, `SolverStepError`, and `SolverFactory`) for time integration of ordinary differential equations, tridiagonal linear solves, and bracketed scalar root finding across RELIC. Theoretical foundations for stiff ODE integration, operator splitting, and implicit differentiation appear in *Methods of Ecological Analysis*, Chapters 21, 22, and 28.

This layer contains seven registered solvers backed by `relic/parameters/database/relic/computation/solvers.yaml`:
- **Temporal ODE solvers (`relic/computation/solvers/temporal.py`)**: `BackwardEulerSolver` (`"backward_euler"`), `ImplicitMidpointSolver` (`"implicit_midpoint"`), `CrankNicolsonSolver` (`"crank_nicolson"`), `IMEXEulerSolver` (`"imex_euler"`), and `RK4Solver` (`"rk4"`), plus convenience driver `solve_ode_system`.
- **Direct tridiagonal linear solver (`relic/computation/solvers/linear.py`)**: `ThomasAlgorithmSolver` (`"thomas_algorithm"`), backed by `solve_tridiagonal_thomas`, `solve_tridiagonal_batch`, and `check_matrix_conditioning`.
- **Differentiable scalar root solver (`relic/computation/solvers/nonlinear.py`)**: `BracketedBisectionSolver` (`"bracketed_bisection"`).

Solvers inherit from `(ABC, torch.nn.Module)` via `SolverBase` (`relic/computation/solvers/base.py`), not from `FunctorBase`, and do not map to the four scientific Functor abstractions. Standalone solver time grids carry no adaptive step-size controller; simulator-level step control resides solely in `relic/core/simulator/execution_loop.py` (documented in [spatiotemporal_scale_topology.md](spatiotemporal_scale_topology.md)).

## 2. Governing Equations

### 2.1 Implicit, Explicit, and IMEX One-Step ODE Schemes

For initial-value problems $y'(t) = f(y, t)$ advancing from $(t_n, y_n)$ over step $\Delta t > 0$ with $t_{n+1} = t_n + \Delta t$:

1. **Backward Euler (`BackwardEulerSolver`, order 1, A- and L-stable)** with damped fixed-point map $G_{\mathrm{BE}}(y) = y_n + \Delta t\,f(y, t_{n+1})$ and relaxation $\omega \in (0, 1]$:

   $$y_{n+1} = y_n + \Delta t\,f(y_{n+1}, t_{n+1}), \qquad r^{(k)} = G_{\mathrm{BE}}(y^{(k)}) - y^{(k)}, \qquad y^{(k+1)} = \omega\,G_{\mathrm{BE}}(y^{(k)}) + (1 - \omega)\,y^{(k)}.$$

2. **Implicit Midpoint (`ImplicitMidpointSolver`, order 2, A-stable, symplectic)** evaluated by undamped fixed-point iteration ($\omega = 1$) at $t_{n+1/2} = t_n + \tfrac{1}{2}\Delta t$:

   $$y_{n+1} = y_n + \Delta t\,f\!\left(\frac{y_n + y_{n+1}}{2},\; t_n + \frac{\Delta t}{2}\right), \qquad y^{(k+1)} = y_n + \Delta t\,f\!\left(\frac{y_n + y^{(k)}}{2},\; t_{n+1/2}\right).$$

3. **Crank-Nicolson / $\theta$-Method (`CrankNicolsonSolver`, order 2 iff $\lvert\theta - 0.5\rvert \le 10^{-10}$, order 1 otherwise; A-stable for $\theta \in [0.5, 1]$)** via undamped fixed-point iteration on $\theta \in [0, 1]$:

   $$y_{n+1} = y_n + \Delta t\left[(1 - \theta) f(y_n, t_n) + \theta f(y_{n+1}, t_{n+1})\right].$$

4. **Split IMEX Euler (`IMEXEulerSolver`, order 1)** for $y' = f_{\mathrm{exp}}(y, t) + g_{\mathrm{imp}}(y, t)$ with $M = \text{explicit\_substeps} \ge 1$ explicit substeps followed by $\omega$-damped implicit fixed-point iteration ($\omega \in (0, 1]$; Ascher, Ruuth, and Spiteri, 1997):

   $$y^{[0]} = y_n, \qquad y^{[i+1]} = y^{[i]} + \frac{\Delta t}{M} f_{\mathrm{exp}}\!\left(y^{[i]},\; t_n + i\frac{\Delta t}{M}\right) \quad (i = 0, \dots, M-1),$$

   $$\tilde{y}^{(k+1)} = y^{[M]} + \Delta t\,g_{\mathrm{imp}}(y^{(k)}, t_{n+1}), \qquad r^{(k)} = \tilde{y}^{(k+1)} - y^{(k)}, \qquad y^{(k+1)} = \omega\,\tilde{y}^{(k+1)} + (1 - \omega)\,y^{(k)}.$$

5. **Classical Fourth-Order Runge-Kutta (`RK4Solver`, order 4, conditionally stable, real stability interval $\approx [-2.78, 0]$)**:

   $$k_1 = f(y_n, t_n), \quad k_2 = f\!\left(y_n + \tfrac{\Delta t}{2}k_1, t_n + \tfrac{\Delta t}{2}\right), \quad k_3 = f\!\left(y_n + \tfrac{\Delta t}{2}k_2, t_n + \tfrac{\Delta t}{2}\right), \quad k_4 = f(y_n + \Delta t\,k_3, t_n + \Delta t),$$

   $$y_{n+1} = y_n + \frac{\Delta t}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right).$$

### 2.2 Fixed-Point Stopping Rule and Exact Binary64 Trajectory Grid

All four implicit ODE solvers test convergence on the **undamped** fixed-point increment $r^{(k)} = \tilde{y}^{(k+1)} - y^{(k)}$ (never on $\omega r^{(k)}$, which would understate the step residual by $\omega$) via `SolverBase.check_convergence`:

$$
\lVert r^{(k)} \rVert_\infty < \max\!\left(\mathrm{atol},\; \mathrm{rtol}\,\lVert \tilde{y}^{(k+1)} \rVert_\infty\right), \qquad \mathrm{atol} > 0,\quad \mathrm{rtol} \ge 0.
$$

`_ODEForwardMixin.forward(ode_func, y0, (t_start, t_end), dt)` partitions $[t_{\mathrm{start}}, t_{\mathrm{end}}]$ using IEEE binary64 rounding bounds ($\varepsilon = \text{sys.float\_info.epsilon}$, $T = \max(\lvert t_{\mathrm{start}}\rvert, \lvert t_{\mathrm{end}}\rvert)$; Higham, 2002):

$$
q = \frac{t_{\mathrm{end}} - t_{\mathrm{start}}}{\Delta t}, \qquad \delta_q = \varepsilon\left(\frac{T}{\Delta t} + q\right), \qquad
N = \begin{cases} \operatorname{round}(q), & \lvert q - \operatorname{round}(q)\rvert \le \delta_q, \\ \lfloor q \rfloor + 1, & \text{otherwise}, \end{cases}
$$

taking $N - 1$ steps of length $\Delta t$ and a positive terminal step closing on $t_N = t_{\mathrm{end}}$ exactly. Returned `time_points` are always `torch.float64` regardless of `y0.dtype`.

### 2.3 Tridiagonal Thomas Elimination, Varah Conditioning Bound, and Implicit Root Gradients

For an $n \times n$ tridiagonal system $A x = d$ with sub-diagonal $a_{1..n-1}$, main diagonal $b_{0..n-1}$, and super-diagonal $c_{0..n-2}$, `solve_tridiagonal_thomas` (and batched `solve_tridiagonal_batch`) performs unpivoted $LU$ elimination (Thomas, 1949; Press et al., 1992):

$$
p_0 = b_0, \quad c'_0 = \frac{c_0}{p_0}, \quad d'_0 = \frac{d_0}{p_0}; \qquad
p_i = b_i - a_i c'_{i-1}, \quad c'_i = \frac{c_i}{p_i}, \quad d'_i = \frac{d_i - a_i d'_{i-1}}{p_i},
$$

followed by $x_{n-1} = d'_{n-1}$ and $x_i = d'_i - c'_i x_{i+1}$. With row $\ell_\infty$ scale $s_i = \lvert a_i\rvert\,\mathbb{I}_{i \ge 1} + \lvert b_i\rvert + \lvert c_i\rvert\,\mathbb{I}_{i \le n-2}$ and dtype machine epsilon $\varepsilon_m = \text{torch.finfo(dtype).eps}$, any non-finite pivot or $\lvert p_i \rvert \le \varepsilon_m s_i$ raises `ValueError` immediately without pivot floor substitution. When `check_stability=True`, `check_matrix_conditioning` evaluates the Gershgorin/Varah upper bound on $\operatorname{cond}_\infty(A)$ for $r_i = \lvert a_i\rvert\,\mathbb{I}_{i \ge 1} + \lvert c_i\rvert\,\mathbb{I}_{i \le n-2}$ (Gershgorin, 1931; Varah, 1975):

$$
\kappa_{\mathrm{Varah}} = \frac{\max_i (\lvert b_i\rvert + r_i)}{\min_i (\lvert b_i\rvert - r_i)} \ge \operatorname{cond}_\infty(A) \qquad \left(\min_i(\lvert b_i\rvert - r_i) > \varepsilon_m \max_i(\lvert b_i\rvert + r_i)\right).
$$

`BracketedBisectionSolver` contracts sign-changing intervals $[a_k, b_k]$ ($F(a_0, \theta) F(b_0, \theta) \le 0$) until $\tfrac{1}{2}\lvert b_k - a_k\rvert \le \mathrm{atol} + \mathrm{rtol}\,\lvert m_k\rvert$ or $F(m_k, \theta) = 0$, and attaches second-order-capable implicit parameter gradients via `_ImplicitRootGradient` and `torch.func.vjp` (Blondel et al., 2022):

$$
\frac{\partial x^*}{\partial \theta} = -\left(\frac{\partial F(x^*, \theta)}{\partial x}\right)^{-1}\frac{\partial F(x^*, \theta)}{\partial \theta}.
$$

## 3. Variable Table

| Symbol / Object | Meaning | Units / Type | Code Name | Role |
| --- | --- | --- | --- | --- |
| $y_n, y_{n+1}$ | Pre-step and post-step state tensors | Caller state units (`torch.Tensor`) | `y_current`, `SolverResult.solution` | Solver input / output |
| $t_n, \Delta t$ | Step start time and positive step size | $\mathrm{s}$ (or caller time unit, `float`) | `t_current`, `dt` | Solver input |
| $a, b, c, d$ | Sub-, main-, super-diagonal and RHS of tridiagonal system | Matrix / RHS units (`[n]` or `[batch, n]`) | `a`, `b`, `c`, `d` | Linear solver input |
| $[a_0, b_0], \theta$ | Strict scalar root bracket ($a_0 < b_0$) and parameter tuple | Root / parameter units (`torch.Tensor`) | `lower`, `upper`, `parameters` | Nonlinear solver input |
| `status` | `SUCCESS`, `MAX_ITERATIONS`, `DIVERGED`, `SINGULAR`, `INVALID_INPUT` | `SolverStatus` | `SolverResult.status` | Exit status |
| `iterations`, `residual_norm` | Iteration / stage count and final $\lVert r \rVert_\infty$ | `int`, `Optional[float]` | `SolverResult.iterations`, `residual_norm` | Step diagnostics |

## 4. Parameter Table

All 29 solver parameters reside in `relic/parameters/database/relic/computation/solvers.yaml` (namespace `computation.solvers`), with global fallbacks in `relic/parameters/database/relic/physical/constants.yaml`:

| Solver Namespace | `ParameterManager` Keys | Default Values | Bounds / Constraint | Reference |
| --- | --- | --- | --- | --- |
| `computation.solvers.backward_euler` | `damping_factor`, `max_iterations`, `rtol`, `atol` | `1.0`, `50`, `1.0e-6`, `1.0e-9` | $\omega \in (0, 1]$, `max_iterations` $\in [1, 10^4]$ | Hairer and Wanner (1996); Kelley (2003) |
| `computation.solvers.crank_nicolson` | `theta`, `max_iterations`, `rtol`, `atol` | `0.5`, `50`, `1.0e-6`, `1.0e-9` | $\theta \in [0, 1]$, `max_iterations` $\in [1, 10^4]$ | Crank and Nicolson (1947) |
| `computation.solvers.implicit_midpoint` | `max_iterations`, `rtol`, `atol` | `100`, `1.0e-6`, `1.0e-8` | `max_iterations` $\in [1, 10^4]$ (distinct from `transport.numerical.*`) | Hairer, Lubich, and Wanner (2006) |
| `computation.solvers.imex_euler` | `explicit_substeps`, `damping_factor`, `max_iterations`, `rtol`, `atol` | `1`, `1.0`, `50`, `1.0e-6`, `1.0e-9` | $M \in [1, 100]$, $\omega \in (0, 1]$ | Ascher, Ruuth, and Spiteri (1997) |
| `computation.solvers.rk4` | `max_iterations`, `rtol`, `atol` | `50`, `1.0e-6`, `1.0e-9` | Resolved by `SolverBase`; unused by fixed 4-stage explicit step | Kutta (1901); Butcher (2008) |
| `computation.solvers.thomas` | `check_stability`, `report_condition`, `condition_number_limit`, `max_iterations`, `rtol`, `atol` | `true`, `false`, `1.0e12`, `1`, `1.0e-6`, `1.0e-8` | `condition_number_limit` $\in [1, 10^{16}]$ | Thomas (1949); Varah (1975); Press et al. (1992) |
| `computation.solvers.bracketed_bisection` | `max_iterations`, `rtol`, `atol` | `80`, `1.0e-10`, `1.0e-10` | `max_iterations` $\in [1, 512]$ | Burden and Faires (2010); Blondel et al. (2022) |
| `computation.solvers.hybrid_events` | `transversality_tolerance` | `1.0e-12` | $[10^{-300}, 10^{-3}]$ (see [hybrid_events.md](hybrid_events.md)) | Hiskens and Pai (2000) |

## 5. Numerical Methods and Discretization

- **Strict configuration and YAML spec validation**: `SolverBase.__init__` accepts only `("max_iterations", "rtol", "atol") | set(metadata.parameters)`; any undeclared config key raises `ValueError`. Overridden values in `config` are validated against `ParameterSpec.validate_value` (bounds and finiteness) before use. Passing `units` or `bounds` in Python code to `SolverBase.get_parameter` raises `ValueError`.
- **Fail-fast step and trajectory semantics**: Because all implicit ODE solvers use Jacobian-free fixed-point iteration, contractivity requires $\Delta t\,L_f$ (scaled by $\theta$, $1/2$, or $\omega$) $< 1$; unconditional A-stability of the underlying discretization does not guarantee fixed-point convergence at arbitrarily large stiff steps. When `solve()` returns `SolverStatus.MAX_ITERATIONS` or `SolverStatus.DIVERGED` (non-finite `RK4Solver` output), `_ODEForwardMixin.step()` and `forward()` raise `SolverStepError` carrying `solver_name`, `status`, `step_index`, `t_current`, `dt`, `iterations`, `residual_norm`, and `metadata`. Non-finite iterates inside implicit fixed-point loops raise `ValueError` in `check_convergence`.
- **Autograd-safe tridiagonal sweep**: `solve_tridiagonal_thomas` accumulates Python lists of scalar `0-d` tensors and calls `torch.stack` once at return, avoiding in-place autograd graph mutation for callers such as `RichardsSoilMoisture`.
- **Removed legacy APIs**: `AdaptiveTimeStepper`, `forward(..., adaptive=True)` (now `TypeError`), `solve_ode_system(..., method="forward_euler")` (now `ValueError`; accepted methods are `"backward_euler"`, `"implicit_midpoint"`, `"crank_nicolson"`, and `"rk4"`), and Thomas pivot floor regularization (`computation.solvers.thomas.regularization`) are permanently absent.

## 6. Conservation, Boundary Conditions, and Ledgers

`SolverBase` enforces runtime state invariance (invariant `I1`): calling `solve()` or `nn.Module.forward()` mutates no internal solver counters or buffers; all per-call diagnostics reside on the returned `SolverResult`. Notice that `SolverBase.check_convergence` tests a scaled successive-iterate increment $\lVert y_{\mathrm{new}} - y_{\mathrm{prev}}\rVert_\infty$, which bounds the true algebraic equation defect only when the fixed-point Lipschitz factor $q < 1$ is bounded away from $1$. Exact quadratic invariant preservation of `ImplicitMidpointSolver` holds up to the converged fixed-point tolerance; transactional rollback of unconverged geometric midpoint steps is governed separately by `transactional_implicit_midpoint_step` in `relic/computation/solvers/geometric.py` (see [structural_ecology_foundation.md](structural_ecology_foundation.md)).

## 7. Code Mapping

| Class / Function | File Path | `SolverFamily` / Registry Name | Role |
| --- | --- | --- | --- |
| `SolverBase`, `SolverMetadata`, `SolverResult`, `SolverStatus`, `SolverStepError` | `relic/computation/solvers/base.py` | Base contract | Parameter validation, `check_convergence`, invariant `I1` |
| `BackwardEulerSolver`, `ImplicitMidpointSolver`, `CrankNicolsonSolver`, `IMEXEulerSolver`, `RK4Solver`, `solve_ode_system` | `relic/computation/solvers/temporal.py` | `ODE_IMPLICIT`, `ODE_IMEX`, `ODE_EXPLICIT` | Fixed-step ODE/IMEX integrators and `_ODEForwardMixin` |
| `ThomasAlgorithmSolver`, `solve_tridiagonal_thomas`, `solve_tridiagonal_batch`, `check_matrix_conditioning` | `relic/computation/solvers/linear.py` | `LINEAR_DIRECT` (`"thomas_algorithm"`) | Unpivoted $O(n)$ tridiagonal direct solver and Varah bound |
| `BracketedBisectionSolver` | `relic/computation/solvers/nonlinear.py` | `NONLINEAR` (`"bracketed_bisection"`) | Batched bracketed bisection with 2nd-order VJP implicit gradients |
| `SolverFactory`, `SolverRegistry`, `register_solver` | `relic/computation/solvers/registry.py` | Registry / factory | Name-based solver instantiation |

## 8. Implementation Status

- **Implemented (`[I]`)**: Fixed-point `BackwardEulerSolver`, `ImplicitMidpointSolver`, `CrankNicolsonSolver`, multirate-explicit `IMEXEulerSolver`, explicit `RK4Solver`, binary64-grid `_ODEForwardMixin.forward`, `SolverStepError` fail-fast propagation, unpivoted `ThomasAlgorithmSolver` with scaling-invariant singularity refusal and Varah $\ell_\infty$ condition estimate, and `BracketedBisectionSolver` with second-order implicit differentiation.
- **Not implemented (`[N]`)**: Newton-Krylov / Jacobian-backed implicit ODE stage solvers inside `temporal.py`, embedded adaptive Runge-Kutta step-size control inside `_ODEForwardMixin`, and partial-pivoting banded $LU$ fallback inside `linear.py`.
- **IPHS Equivalence Certificates**: None (`0` certificates reference `numerical_time_integration.md`).

## 9. References

1. Ascher, U. M., Ruuth, S. J., and Spiteri, R. J. (1997). Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations. *Applied Numerical Mathematics*, 25(2-3), 151-167.
2. Blondel, M., Berthet, Q., Cuturi, M., Frostig, R., Hoyer, S., Llinares-Lopez, F., Pedregosa, F., and Vert, J.-P. (2022). Efficient and modular implicit differentiation. *Advances in Neural Information Processing Systems*, 35, 5230-5242.
3. Burden, R. L., and Faires, J. D. (2010). *Numerical Analysis* (9th ed.). Brooks/Cole.
4. Butcher, J. C. (2008). *Numerical Methods for Ordinary Differential Equations* (2nd ed.). Wiley.
5. Crank, J., and Nicolson, P. (1947). A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type. *Proceedings of the Cambridge Philosophical Society*, 43(1), 50-67.
6. Gershgorin, S. (1931). Uber die Abgrenzung der Eigenwerte einer Matrix. *Izvestiya Akademii Nauk SSSR, Seriya Matematicheskaya*, 6, 749-754.
7. Hairer, E., Lubich, C., and Wanner, G. (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations* (2nd ed.). Springer.
8. Hairer, E., and Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.). Springer.
9. Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
10. Kelley, C. T. (2003). *Solving Nonlinear Equations with Newton's Method*. SIAM.
11. Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (1992). *Numerical Recipes in C: The Art of Scientific Computing* (2nd ed., Section 2.4). Cambridge University Press.
12. Thomas, L. H. (1949). *Elliptic Problems in Linear Difference Equations over a Network*. Watson Scientific Computing Laboratory Report, Columbia University.
13. Varah, J. M. (1975). A lower bound for the smallest singular value of a matrix. *Linear Algebra and its Applications*, 11(1), 3-5.
