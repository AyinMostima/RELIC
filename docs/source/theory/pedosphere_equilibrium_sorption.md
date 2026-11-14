# Soil Equilibrium Sorption and Dissolved Decay

## 1. Process Definition and Scope

`EquilibriumSorptionFunctor` (`relic/scientific/pedosphere/transport/equilibrium_sorption.py`) evolves a point-scale (`SPATIAL_DIMS = None`, `SpatialScale.GRID_CELL`) semi-explicit index-1 differential-algebraic equation (DAE) for a sorbing soil solute subject to instantaneous nonlinear Freundlich phase partitioning and first-order degradation in the dissolved phase (`[I]`).

The differential state is total solute mass per unit bulk soil volume $y$ (`equilibrium_total_solute_mass`), while dissolved concentration $C$ (`equilibrium_dissolved_concentration`) and solid-phase sorbed mass fraction $S$ (`equilibrium_sorbed_concentration`) are canonical `StateManager` algebraic states (`constraint_category = "descriptor_dae"`, `constraint_index = 1`, `elimination_method = "algebraic_elimination"`). Volumetric liquid water content $\theta$ (`soil_moisture`) and dry soil bulk density $\rho_b$ (`bulk_density`) are mandatory runtime tensor inputs in `state`. Spatial advection-dispersion coupling of $y$ and multi-species competitive isotherms are outside this Functor (`[N]`).

## 2. Governing Equations

Total solute mass per unit bulk volume $y \ge 0$ ($\mathrm{kg\,m^{-3}}$) partitions instantaneously between aqueous solution at concentration $C \ge 0$ ($\mathrm{kg\,m^{-3}}$) and solid matrix at sorbed concentration $S \ge 0$ ($\mathrm{kg\,kg^{-1}}$) according to the Freundlich isotherm with coefficient $K_f > 0$ and exponent $n > 0$, while dissolved solute decays at first-order rate $\lambda \ge 0$ ($\mathrm{s^{-1}}$):

$$
\frac{d y}{dt} = -\lambda\,\theta\,C,
$$

$$
0 = g(y, C) = \theta\,C + \rho_b\,K_f\,C^{n} - y, \qquad S = K_f\,C^{n}.
$$

For $C > 0$, $\theta \in (0, 1]$, $\rho_b > 0$, $K_f > 0$, and $n > 0$, the algebraic Jacobian with respect to $C$ is strictly positive:

$$
\frac{\partial g}{\partial C} = \theta + \rho_b\,K_f\,n\,C^{n-1} > 0,
$$

confirming differentiation index $1$ and unique local invertibility $C = C(y)$ by the implicit function theorem (including at $C = 0$ by strict monotonicity on $[0, \infty)$). In the linear special case $n = 1$ with $K_d = K_f$ ($\mathrm{m^3\,kg^{-1}}$), defining volumetric storage capacity $B = \theta + \rho_b\,K_d = \theta\,R$ gives $C = y / B$. Note that `PARAMETERS` declares nominal units $\mathrm{m^3\,kg^{-1}}$ for $K_f$, which is dimensionally exact at $n = 1$ and represents the numerical value in $\mathrm{kg^{1-n}\,m^{3n}\,kg^{-1}}$ when $C$ is expressed in $\mathrm{kg\,m^{-3}}$ for general $n > 0$.

Applying backward Euler over step $\Delta t > 0$ ($\mathrm{s}$) with end-of-step algebraic consistency yields the scalar monotone root problem for $C' = C(t + \Delta t)$:

$$
F(C') = \theta\,C' + \rho_b\,K_f\,\left(C'\right)^{n} + \Delta t\,\lambda\,\theta\,C' - y = 0,
$$

followed by state updates and implicit degradation sink $J_{\mathrm{deg}}$ ($\mathrm{kg\,m^{-3}\,s^{-1}}$):

$$
S' = K_f\,\left(C'\right)^{n}, \qquad J_{\mathrm{deg}} = \lambda\,\theta\,C', \qquad y' = y - \Delta t\,J_{\mathrm{deg}}.
$$

For $n = 1$ and $B = \theta + \rho_b\,K_d$, the implicit update reduces to the closed analytical solution

$$
C' = \frac{y}{B + \Delta t\,\lambda\,\theta}, \qquad y' = \frac{y}{1 + \dfrac{\Delta t\,\lambda\,\theta}{B}}.
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| :--- | :--- | :--- | :--- | :--- |
| $y$ | Total solute mass per unit bulk soil volume (differential state) | $\mathrm{kg\,m^{-3}}$ | `equilibrium_total_solute_mass` | `input` / `output` (`state increment`) |
| $C$ | Dissolved aqueous solute concentration (algebraic state) | $\mathrm{kg\,m^{-3}}$ | `equilibrium_dissolved_concentration` | `input` / `output` (`algebraic state`) |
| $S$ | Solid-phase sorbed solute mass per unit dry soil mass (algebraic state) | $\mathrm{kg\,kg^{-1}}$ | `equilibrium_sorbed_concentration` | `input` / `output` (`algebraic state`) |
| $\theta$ | Volumetric liquid soil water content | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `input` |
| $\rho_b$ | Dry soil bulk density | $\mathrm{kg\,m^{-3}}$ | `bulk_density` | `input` |
| $r_{\mathrm{dae}}$ | Post-step partition constraint residual magnitude $\lvert \theta C' + \rho_b S' - y' \rvert$ | $\mathrm{kg\,m^{-3}}$ | `diagnostics.equilibrium_sorption.constraint_residual` | `diagnostic` |
| $J_{\mathrm{deg}}$ | Implicit dissolved-phase degradation sink $\lambda\,\theta\,C'$ | $\mathrm{kg\,m^{-3}\,s^{-1}}$ | `diagnostics.equilibrium_sorption.degradation_flux` | `diagnostic` |
| $k_{\mathrm{iter}}$ | Bracketed bisection contractions executed on active cells | $1$ | `diagnostics.equilibrium_sorption.solver_iterations` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
| :--- | :--- | :--- | :--- | :--- |
| $K_f$ | `pedosphere.transport.equilibrium_sorption_freundlich_coefficient` | $\mathrm{m^3\,kg^{-1}}$ (nominal; $\mathrm{kg^{1-n}\,m^{3n}\,kg^{-1}}$ for $n > 0$) | `pedosphere/transport.yaml` ($0.002$) | Jury and Horton (2004); Mayes et al. (2012) |
| $n$ | `pedosphere.transport.equilibrium_sorption_freundlich_exponent` | $1$ | `pedosphere/transport.yaml` ($0.8$) | Jury and Horton (2004); Mayes et al. (2012) |
| $\lambda$ | `pedosphere.transport.equilibrium_sorption_dissolved_decay_rate` | $\mathrm{s^{-1}}$ | `pedosphere/transport.yaml` ($10^{-6}$) | First-order aqueous degradation rate |
| $\tau_{\mathrm{cons}}$ | `pedosphere.transport.equilibrium_sorption_consistency_tolerance` | $1$ | `pedosphere/transport.yaml` ($10^{-9}$) | Hairer and Wanner (1996) consistent DAE check |
| $\varepsilon_{\mathrm{abs}}$ | `computation.solvers.bracketed_bisection.atol` | $1$ (interval width in $\mathrm{kg\,m^{-3}}$) | `computation/solvers.yaml` ($10^{-10}$) | Burden and Faires (2010) |
| $\varepsilon_{\mathrm{rel}}$ | `computation.solvers.bracketed_bisection.rtol` | $1$ | `computation/solvers.yaml` ($10^{-10}$) | Burden and Faires (2010) |
| $K_{\max}$ | `computation.solvers.bracketed_bisection.max_iterations` | $1$ | `computation/solvers.yaml` ($80$) | Burden and Faires (2010) |

Instance-level `config["parameters"]` overrides are resolved ahead of `ParameterManager` (`parameter_lookup_source_order = ("config", "manager")`); unknown override keys raise `ValueError`.

## 5. Numerical Methods and Discretization

1. **Consistent Initialization Gate**: Before advancing, `forward` checks that $y$, $C$, $S$, and $\theta$ share identical tensor shape with $y, C, S \ge 0$, $\theta \in (0, 1]$, and $\rho_b > 0$. It then verifies both algebraic constraints against relative tolerance $\tau_{\mathrm{cons}}$ and machine `tiny = torch.finfo(total.dtype).tiny`:

$$
\left\lvert \theta\,C + \rho_b\,S - y \right\rvert \le \tau_{\mathrm{cons}}\,\max\!\left(\lvert y \rvert, \epsilon_{\mathrm{tiny}}\right), \qquad \left\lvert S - K_f\,C^{n} \right\rvert \le \tau_{\mathrm{cons}}\,\max\!\left(\lvert S \rvert, \epsilon_{\mathrm{tiny}}\right).
$$

Any inconsistent initial or restored checkpoint triple raises `ValueError` without silent projection.
2. **Exact Root Bracketing and Differentiable Bisection**: Cells with $y = 0$ bypass the nonlinear solve and receive exact zeros ($C' = 0$, $S' = 0$, $k_{\mathrm{iter}} = 0$). On active cells (`active = total > 0`), $F(0) = -y < 0$ and $F(y/\theta) = \rho_b K_f (y/\theta)^n + \Delta t\,\lambda\,y > 0$ with $F'(C') > 0$, so $[0, y/\theta]$ is an exact bracket. `BracketedBisectionSolver` contracts $[0, y/\theta]$ using `atol`, `rtol`, and `max_iterations` while preserving implicit parameter gradients ($\partial C' / \partial K_f$, $\partial C' / \partial n$, $\partial C' / \partial \lambda$) via implicit differentiation at the converged root. Non-convergence (`result.status` distinct from `SolverStatus.SUCCESS`) raises `RuntimeError`. Note on active-mask broadcasting: `y_active = total[active]` and `theta_active = theta[active]` are 1-D masked slices whereas `bulk_density` is passed to `residual_fn` unsliced; consequently, spatially varying 1-D `bulk_density` tensors require all cells to be active ($y > 0$), while scalar `bulk_density` broadcasts across arbitrary mixed zero/positive `total` tensors.
3. **UpdateItem Semantics**: `equilibrium_total_solute_mass` is emitted as an increment $y' - y = -\Delta t\,J_{\mathrm{deg}}$ with `UpdateMode.ACCUMULATE` (`per_timestep=True`, `unit="kg m-3"`). The re-solved algebraic states `equilibrium_dissolved_concentration` (`kg m-3`) and `equilibrium_sorbed_concentration` (`kg kg-1`), together with the three diagnostics, are emitted with `UpdateMode.REPLACE` (`per_timestep=True`). Calling `get_value_at(coordinates)` raises `NotImplementedError` because `SPATIAL_DIMS = None`.

## 6. Conservation, Boundary Conditions, and Ledgers

- **Total-Mass DAE Ledger**: Solute leaves the closed bulk control volume solely through dissolved-phase first-order decay $\Delta t\,J_{\mathrm{deg}}$. `forward` stores `_last_mass_ledger_residual = (y' - y) + dt * degradation_flux`, and `check_conservation_laws` reports

$$
r_{\mathrm{mass}} = \max \left\lvert \Delta y + \Delta t\,\lambda\,\theta\,C' \right\rvert
$$

under key `"mass"` ($\mathrm{kg\,m^{-3}}$).
- **Algebraic Constraint Audit and Checkpoint Restart**: `diagnostics.equilibrium_sorption.constraint_residual` emits $\lvert \theta C' + \rho_b S' - y' \rvert$ at every step, and GraphIR `descriptor_dae` metadata binds $(y, C, S)$ so restored checkpoints undergo the same consistent-triple verification prior to stepping.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| :--- | :--- | :--- | :--- | :--- |
| `EquilibriumSorptionFunctor` | `relic/scientific/pedosphere/transport/equilibrium_sorption.py` | `PhysicalField` | `PhysicalFieldFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

## 8. Implementation Status

- `EquilibriumSorptionFunctor` index-1 Freundlich equilibrium sorption DAE with implicit dissolved decay, consistent-triple verification, and differentiable bracketed bisection elimination: `[I]` (certified IPHS Level 2 descriptor DAE; certificate `pedosphere_transport_equilibrium_sorption_equilibrium_sorption_functor.yaml`).
- Coupled reactive-transport PDE advancing $y$ with `SoluteAdvectionDiffusion1D` and multi-solute competitive sorption isotherms: `[N]`.

Certificate expressions (`pedosphere_transport_equilibrium_sorption_equilibrium_sorption_functor.yaml`):

```text
legacy_expression: y' = y - dt lambda theta C', C' from theta C' + rho_b K_f C'^n + dt lambda theta C' = y, S' = K_f C'^n
iphs_expression: consistent initial triple verified within the relative tolerance; C' bracketed on [0, y/theta]; constraint residual and solver iterations reported; non-convergence fails the step
```

## 9. References

1. Hairer, E., and Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd ed., Chapter VI). Springer.
2. Jury, W. A., and Horton, R. (2004). *Soil Physics* (6th ed., Chapter 7). Wiley.
3. Burden, R. L., and Faires, J. D. (2010). *Numerical Analysis* (9th ed.). Brooks/Cole.
4. Mayes, M. A., Heal, K. R., Brandt, C. C., Phillips, J. R., and Jardine, P. M. (2012). Relation between soil order and sorption of dissolved organic carbon in temperate subsoils. *Soil Science Society of America Journal*, 76(1), 199-206.
