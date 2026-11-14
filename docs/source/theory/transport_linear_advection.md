# Linear Hyperbolic Advection Transport

## 1. Process Definition and Scope

`LinearAdvection` (`relic/scientific/transport/advection.py`) advances conservative hyperbolic tracer transport (`tracer_concentration`, default $\mathrm{kg\,m^{-3}}$, configurable `tracer_units`) driven by `velocity_field` ($\mathrm{m\,s^{-1}}$) on uniform orthogonal cell-centered grids (`SpatialScale.GRID_CELL`), with optional volumetric `source_term` and conditional open-boundary exchange (`tracer_concentration_advective_boundary_flux`). Factory helper `create_uniform_advection` builds constant-flow multidimensional instances from scalar speed and normalized direction vectors.

Undeclared component velocity state aliases (`u`, `v`, `velocity_x`), hidden ParameterManager velocity fallbacks in `forward()`, local automatic time-step shortening, and non-conservative pointwise concentration clipping remain `[N]`. Shared continuum transport rules appear in `transport.md`, and deeper derivations appear in *Methods of Ecological Analysis*, Chapter 13 (Section 13.3) and Chapter 26 (Section 26.2).

## 2. Governing Equations

### 2.1 Continuum Conservative Advection Balance

For tracer concentration $c(\mathbf{x}, t)$ (`tracer_units`, default $\mathrm{kg\,m^{-3}}$), advective velocity vector $\mathbf{v}(\mathbf{x}, t)$ ($\mathrm{m\,s^{-1}}$), and volumetric source rate $S(\mathbf{x}, t)$ (`tracer_units` $\mathrm{s^{-1}}$), mass conservation in divergence form is:

$$
\frac{\partial c}{\partial t} + \nabla \cdot (\mathbf{v} c) = S, \qquad
\mathbf{q} = \mathbf{v} c, \qquad
\frac{\partial c}{\partial t} = -\nabla \cdot \mathbf{q} + S
$$

### 2.2 Discrete IPHS-DEC Cochain Form and Zero-Dissipation Structure

On the discrete cell complex with primal tracer cochain $c_h$, finite-volume divergence $D_h$, velocity interior contraction $i_{v_h}$, and discrete source rate $s_h$:

$$
\frac{dc_h}{dt} = -D_h q_h + s_h, \qquad
q_h = i_{v_h} c_h, \qquad
M_h = 0, \qquad
\sigma_h = X_h^{\mathsf{T}} M_h X_h = 0
$$

Because pure continuum advection is reversible transport, its dissipative mobility matrix is identically zero ($M_h = 0 \succeq 0$), while `reversible_structure()` exposes the conservative skew interconnection matrix $J_h = -J_h^{\mathsf{T}}$ and `build_hodge_terms()` supplies a passive velocity-magnitude Hodge view from `transport.advection.default_velocity_magnitude` strictly for structural graph audits.

### 2.3 Finite-Volume Face Reconstruction Schemes and Outflow Courant Bound

Along each spatial axis with cell width $\Delta x$ and arithmetic face velocity $u_{i+1/2} = \frac{1}{2}(u_i + u_{i+1})$, `_compute_advection_flux_face_1d` (`relic/computation/operators/functional.py`) evaluates conservative interior face flux $F_{i+1/2}$ using one of four `LinearAdvection` schemes (Courant et al., 1928; Sweby, 1984; LeVeque, 2002):

1. **First-order upwind** (`"upwind"`):

$$
F_{i+1/2}^{\mathrm{up}} =
\begin{cases}
u_{i+1/2}\,c_i, & u_{i+1/2} \ge 0 \\
u_{i+1/2}\,c_{i+1}, & u_{i+1/2} < 0
\end{cases}
$$

2. **Second-order Lax-Wendroff** (`"lax_wendroff"`):

$$
F_{i+1/2}^{\mathrm{LW}} = \frac{1}{2} u_{i+1/2}(c_i + c_{i+1}) - \frac{\Delta t}{2\Delta x} u_{i+1/2}^2 (c_{i+1} - c_i)
$$

3. **TVD flux-limited high-resolution schemes** (`"superbee"` and `"van_leer"`): with scale-invariant successive-gradient ratio $r_{i+1/2} = (\Delta_{\mathrm{up}}/s)/\operatorname{reg}(\Delta_{\mathrm{down}}/s, \varepsilon_{\mathrm{lim}})$ where $s = \max(\lvert\Delta_{\mathrm{up}}\rvert, \lvert\Delta_{\mathrm{down}}\rvert)$ and local face Courant number $C_{i+1/2} = \lvert u_{i+1/2}\rvert \Delta t / \Delta x$:

$$
F_{i+1/2}^{\mathrm{TVD}} = F_{i+1/2}^{\mathrm{up}} + \frac{1}{2}\lvert u_{i+1/2}\rvert (1 - C_{i+1/2})\,\phi(r_{i+1/2})\,(c_{i+1} - c_i)
$$

$$
\phi_{\mathrm{superbee}}(r) = \max\bigl(0,\; \min(2r, 1),\; \min(r, 2)\bigr), \qquad
\phi_{\mathrm{van\_leer}}(r) = \frac{r + \lvert r \rvert}{1 + \lvert r \rvert}
$$

Explicit Euler stepping enforces the multidimensional outward-face Courant condition across all $d$ active spatial axes:

$$
C_{\mathrm{out},j} = \Delta t \sum_{k=1}^{d} \frac{\max(u_{k,j+1/2}, 0) - \min(u_{k,j-1/2}, 0)}{\Delta x_k} \le \mathrm{CFL}_{\mathrm{adv,max}}
$$

which reduces to $\Delta t \sum_{k=1}^{d} \lvert u_k \rvert / \Delta x_k \le \mathrm{CFL}_{\mathrm{adv,max}}$ for spatially uniform flow.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $c_h$ | Transported tracer concentration cochain (`ACCUMULATE`, `per_timestep=True`) | `tracer_units` (default $\mathrm{kg\,m^{-3}}$) | `tracer_concentration` | `state` |
| $\mathbf{v}_h$ | Required advective velocity vector field (state port or constructor/setter tensor) | $\mathrm{m\,s^{-1}}$ | `velocity_field` | `forcing` / `state` |
| $S_h$ | Optional volumetric tracer source or sink rate | `tracer_units` $\mathrm{s^{-1}}$ ($\mathrm{kg\,m^{-3}\,s^{-1}}$) | `source_term` | `flux` |
| $c_{\partial\Omega}$ | Conditional external inflow concentration (required for `"open"` and `"dirichlet"`) | `tracer_units` ($\mathrm{kg\,m^{-3}}$) | `tracer_concentration_boundary_state` | `forcing` |
| $q_{\partial\Omega}$ | Conditional signed outward-positive boundary face flux cochain (`REPLACE`, `per_timestep=False`) | `advective_flux_units` (default $\mathrm{kg\,m^{-2}\,s^{-1}}$; $\mathrm{m\,s^{-1}}$ when dimensionless) | `tracer_concentration_advective_boundary_flux` | `flux` |
| $C_{\max}$ | Maximum cell outward-face Courant number | $\mathrm{1}$ | `diagnostics.cfl_number` | `diagnostic` |
| $\varepsilon_{\mathrm{cons}}$ | Conditional normalized finite-volume mass-balance residual (`track_conservation=True`) | $\mathrm{1}$ | `diagnostics.conservation_error` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\Delta x$ | `transport.advection.grid_spacing` | $\mathrm{m}$ | `transport/advection.yaml` ($1.0$) | Uniform metric spacing |
| $U_{\mathrm{audit}}$ | `transport.advection.default_velocity_magnitude` | $\mathrm{m\,s^{-1}}$ | `transport/advection.yaml` ($1.0$) | Structural Hodge audit speed only |
| $b_{\mathrm{in}}, b_{\mathrm{out}}$ | `transport.advection.inflow_boundary_type`, `outflow_boundary_type` | `str` | `transport/advection.yaml` (`"dirichlet"`, `"open"`) | Open-boundary inflow/outflow roles |
| $\mathrm{CFL}_{\mathrm{adv,max}}$ | `transport.numerical.cfl_advection_max` | $\mathrm{1}$ | `transport/numerical.yaml` ($1.0$) | Courant et al. (1928); LeVeque (2002) |
| $b_{\mathrm{cons}}, b_{\mathrm{nn}}$ | `transport.numerical.track_conservation`, `enforce_non_negative` | `bool` | `transport/numerical.yaml` (`true`, `true`) | Mass ledger and positivity flags |
| $b_{\mathrm{bc}}$ | `transport.numerical.default_boundary_type` | `str` | `transport/numerical.yaml` (`"neumann"`) | Fallback boundary method |
| $\varepsilon_{\mathrm{lim}}$ | `transport.numerical.limiter_epsilon` | $\mathrm{1}$ | `transport/numerical.yaml` ($10^{-12}$) | Sweby (1984) TVD ratio regularizer |

## 5. Numerical Methods and Discretization

`LinearAdvection.forward` resolves a single boundary method (explicit constructor `boundary_method`, then Functor config `boundary_method`, then `transport.numerical.default_boundary_type`) and delegates its sole numerical transport step through `_compute_iphs_advection_update()` to `apply_advection` (`relic/computation/operators/functional.py`). Fully discrete schemes (`"lax_wendroff"`, `"superbee"`, `"van_leer"`) embed $\Delta t$ in their flux and pair exclusively with `"euler"`, whereas `"upwind"` pairs with `"euler"`, `"implicit_midpoint"`, or `"crank_nicolson"`.

The operator returns post-advection concentration $c_{\mathrm{adv}}^{n+1}$ and step-quadrature outward boundary fluxes; `forward()` forms the total step increment $\Delta c = (c_{\mathrm{adv}}^{n+1} - c^n) + \Delta t\,S^n$ once and emits `UpdateItem(mode=UpdateMode.ACCUMULATE, per_timestep=True)` so `StateManager` applies no second $\Delta t$ factor. Any explicit CFL violation raises `ValueError` and must be resolved via simulator time-stepping or `FunctorMetadata.substep_count`.

## 6. Conservation, Boundary Conditions, and Ledgers

Supported `boundary_method` closures are `"periodic"`, `"neumann"`, `"dirichlet"`, `"zero"`, `"open"`, and `"extrapolate"`:

- **Boundary state and signed outward port**: `"open"` and `"dirichlet"` require explicit `tracer_concentration_boundary_state`; all other closures reject it so no hidden boundary tracer reservoir exists. Nonperiodic boundary faces evaluate upwind normal flux $j_{\mathrm{out},f} = (\mathbf{v}_f \cdot \mathbf{n}_f)\,c_{\mathrm{upwind},f}$ (positive leaving the domain, matching `TransportMechanism.boundary_conditions()` face normal flux $F_b = (\mathbf{v}_b \cdot \mathbf{n}_b) c_b$) and pack `[batch, total_face_dofs]` in `("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")` order into `tracer_concentration_advective_boundary_flux` with derived units `advective_flux_units` (`tracer_units` multiplied by $\mathrm{m\,s^{-1}}$). Under the IPHS into-system convention, port flow is $f_{\mathrm{port}} = -j_{\mathrm{out}}$.
- **Open and closed mass balance**: `"periodic"` has an empty boundary ledger ($\Delta M = \Delta t\sum_j S_j V_j$), whereas nonperiodic runs satisfy exact open-system closure:

$$
M^{n+1} - M^n - \Delta t\sum_j S_j V_j + \Delta t\sum_{f \in \partial\Omega} j_{\mathrm{out},f} A_f = 0
$$

- **Positivity**: when `enforce_non_negative` is enabled, negative initial or candidate concentrations raise `ValueError` without pointwise clipping.

## 7. Code Mapping

| Functor / Factory | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `LinearAdvection` | `relic/scientific/transport/advection.py` | `TransportMechanism` (`TRANSPORT_TYPE="advection"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |
| `create_uniform_advection` | `relic/scientific/transport/advection.py` | Uniform constant-flow factory returning `LinearAdvection` | `TransportMechanismFunctor` | Construction helper |

Structural IPHS hooks implemented on `LinearAdvection`: `declared_forms()`, `required_parameters()`, `build_hodge_terms()`, `build_hodge_contract()`, `reversible_structure()`, `dissipative_structure()`, `ports()`, and `boundary_conditions()`.

## 8. Implementation Status

- `LinearAdvection`: `[I]`, IPHS **Level 3** (`transport`, `port_equivalent`, certificate `transport_advection_linear_advection.yaml`). Single conservative finite-volume runtime path via `_compute_iphs_advection_update()` with PSD zero dissipation $M_h = 0$, unit-consistent `advective_flux_units`, and batch-preserving outward boundary cochain routing.

Frozen certificate expressions (`transport_advection_linear_advection.yaml`):

```text
dc/dt + div(v * c) = S
c_dot = -D_h q_h + s_h; q_h = i_v c_h; M_h = 0 is PSD for the conservative component.
```

## 9. References

- Courant, R., Friedrichs, K., and Lewy, H. (1928). Über die partiellen Differenzengleichungen der mathematischen Physik. *Mathematische Annalen*, 100(1), 32-74.
- Hirani, A. N. (2003). *Discrete Exterior Calculus* (Ph.D. thesis). California Institute of Technology.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
- Sweby, P. K. (1984). High resolution schemes using flux limiters for hyperbolic conservation laws. *SIAM Journal on Numerical Analysis*, 21(5), 995-1011.
