# Coupled Advection-Diffusion Transport

## 1. Process Definition and Scope

`AdvectionDiffusion` (`relic/scientific/transport/advection_diffusion.py`) advances coupled hyperbolic-parabolic tracer transport (`tracer_concentration`, $\mathrm{kg\,m^{-3}}$) in $d \in \{1, 2, 3\}$ spatial dimensions on uniform Cartesian cell-centered grids (`SpatialScale.GRID_CELL`). The Functor couples conservative finite-volume advection driven by `velocity_field` ($\mathrm{m\,s^{-1}}$), Fickian diffusion with `diffusivity` ($\mathrm{m^2\,s^{-1}}$), optional volumetric `source_term` ($\mathrm{kg\,m^{-3}\,s^{-1}}$), and conditional open boundary exchange (`tracer_concentration_advection_diffusion_boundary_flux`, $\mathrm{kg\,m^{-2}\,s^{-1}}$) without instantiating child scientific Functors. Domain-specific factory constructors `create_wind_transport` and `create_groundwater_transport` configure atmospheric boundary-layer and porous-media transport instances.

Full off-diagonal dispersion tensors, unstructured meshes, undeclared component velocity aliases (`u`, `v`, `w`, `velocity_x`), mutable runtime velocity/diffusivity setters, and internal automatic time-step subdivision remain `[N]`. Shared transport operator contracts appear in `transport.md`, and deeper derivations appear in *Methods of Ecological Analysis*, Chapter 13 (Sections 13.3 and 13.5) and Chapter 26 (Section 26.2).

## 2. Governing Equations

### 2.1 Continuum Conservative Advection-Diffusion Balance

For scalar concentration $c(\mathbf{x}, t)$ ($\mathrm{kg\,m^{-3}}$), velocity vector $\mathbf{v}(\mathbf{x}, t)$ ($\mathrm{m\,s^{-1}}$), positive-semidefinite diffusivity $K$ ($\mathrm{m^2\,s^{-1}}$, written $D$ in classical continuum notation), and volumetric source rate $S(\mathbf{x}, t)$ ($\mathrm{kg\,m^{-3}\,s^{-1}}$), the conservative mass balance is:

$$
\frac{\partial c}{\partial t} = -\nabla \cdot (\mathbf{v} c) + \nabla \cdot (K \nabla c) + S = -\nabla \cdot \bigl(\mathbf{q}_{\mathrm{adv}} + \mathbf{q}_{\mathrm{diff}}\bigr) + S, \qquad
\mathbf{q}_{\mathrm{adv}} = \mathbf{v} c, \qquad
\mathbf{q}_{\mathrm{diff}} = -K \nabla c
$$

### 2.2 Discrete IPHS-DEC Cochain Formulation and Structural Dissipation

On the discrete primal cell complex with cell cochain $c_h$, finite-volume divergence $D_h$, exterior difference $d_h$, velocity interior contraction $i_{v_h}$, positive diagonal diffusive Hodge star $\star_K$, and open source/boundary operator $G_h u_h$:

$$
\frac{dc_h}{dt} = -D_h\bigl(q_{\mathrm{adv},h} + q_{\mathrm{diff},h}\bigr) + G_h u_h, \qquad
q_{\mathrm{adv},h} = i_{v_h} c_h, \qquad
q_{\mathrm{diff},h} = -\star_K d_h c_h
$$

The advective operator contributes a skew conservative structure ($M_{\mathrm{adv},h} = 0$), whereas the diffusive component contributes positive-semidefinite mobility $M_{\mathrm{diff},h} \succeq 0$ with non-negative quadratic dissipation $\sigma_h = X_h^{\mathsf{T}} M_{\mathrm{diff},h} X_h \ge 0$ for $X_h = d_h c_h$.

### 2.3 Directional Cell Peclet Number and Configuration Factories

1. **Grid Peclet diagnostic** (Patankar, 1980; Hundsdorfer and Verwer, 2003): for strictly positive per-axis diffusivity $K_{i,k} > 0$,

$$
\mathrm{Pe}_{i,k} = \frac{\lvert u_{i,k}\rvert\,\Delta x_k}{K_{i,k}}, \qquad
\mathrm{Pe}_{\max} = \max_{i,k}\,\mathrm{Pe}_{i,k}
$$

At $K = 0$, the ratio is $0/0$ for $u = 0$ and unbounded for $u \ne 0$; constructing `AdvectionDiffusion` with `track_peclet=True` and $K = 0$ raises `ValueError` at initialization, whereas `track_peclet=False` admits $K = 0$ as pure advection.

2. **Atmospheric wind and porous-media groundwater factories** (Stull, 1988; Bear, 1972):

$$
K_{\mathrm{eff,wind}} = K_{\mathrm{molecular}} + K_{\mathrm{turbulent}}, \qquad
\mathbf{v}_{\mathrm{wind}} = U\,\hat{\mathbf{d}}, \quad \lVert\hat{\mathbf{d}}\rVert_2 = 1
$$

$$
\mathbf{v}_{\mathrm{pore}} = \frac{\mathbf{q}_{\mathrm{Darcy}}}{n_e}, \qquad
D_{\mathrm{eff,gw}} = D_m + \alpha_L\,\lVert\mathbf{v}_{\mathrm{pore}}\rVert_2, \quad n_e \in (0, 1]
$$

Both factories reject non-finite or negative transport coefficients, non-unit wind direction vectors, porosity outside $(0, 1]$, or missing multidimensional Darcy velocity axes.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $c_h$ | Transported tracer concentration cochain (`ACCUMULATE`, `per_timestep=True`) | $\mathrm{kg\,m^{-3}}$ | `tracer_concentration` | `state` |
| $\mathbf{v}_h$ | Required advective velocity vector field ($d$ components) | $\mathrm{m\,s^{-1}}$ | `velocity_field` | `forcing` / `state` |
| $S_h$ | Optional volumetric tracer source or sink rate ($S_h = 0$ when omitted) | $\mathrm{kg\,m^{-3}\,s^{-1}}$ | `source_term` | `flux` |
| $c_{\partial\Omega}$ | Conditional external boundary concentration (required for `"dirichlet"` and `"open"`) | $\mathrm{kg\,m^{-3}}$ | `tracer_concentration_boundary_state` | `forcing` |
| $q_{\partial\Omega}$ | Conditional outward-positive total boundary face flux cochain (`REPLACE`, `per_timestep=False`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `tracer_concentration_advection_diffusion_boundary_flux` | `flux` |
| $C_{\max}, \sigma_{\max}$ | Maximum outward-face Courant number and diffusion stability number | $\mathrm{1}$ | `diagnostics.cfl_number`, `diagnostics.diffusion_stability` | `diagnostic` |
| $\mathrm{Pe}_{\max}$ | Conditional maximum directional cell Peclet number (`track_peclet=True`) | $\mathrm{1}$ | `diagnostics.peclet_number` | `diagnostic` |
| $\varepsilon_{\mathrm{cons}}$ | Conditional closed-domain periodic mass-conservation residual | $\mathrm{1}$ | `diagnostics.conservation_error` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $K$ | `transport.diffusion.diffusivity` | $\mathrm{m^2\,s^{-1}}$ | `transport/diffusion.yaml` ($10^{-5}$) | Crank (1975) |
| $\Delta x$ | `transport.advection_diffusion.grid_spacing` | $\mathrm{m}$ | `transport/advection_diffusion.yaml` ($1.0$) | Uniform metric spacing |
| $m_{\mathrm{split}}$ | `transport.advection_diffusion.splitting_method` | `str` | `transport/advection_diffusion.yaml` (`"symmetric"`) | Strang (1968); Hundsdorfer and Verwer (2003) |
| $U_{\mathrm{audit}}$ | `transport.advection.default_velocity_magnitude` | $\mathrm{m\,s^{-1}}$ | `transport/advection.yaml` ($1.0$) | Structural Hodge audit speed only |
| $\mathrm{CFL}_{\mathrm{adv,max}}$ | `transport.numerical.cfl_advection_max` | $\mathrm{1}$ | `transport/numerical.yaml` ($1.0$) | LeVeque (2002) |
| $\sigma_{\mathrm{diff,max}}$ | `transport.numerical.cfl_diffusion_max` | $\mathrm{1}$ | `transport/numerical.yaml` ($0.5$) | Morton and Mayers (2005) |
| $b_{\mathrm{bc}}$ | `transport.numerical.default_boundary_type` | `str` | `transport/numerical.yaml` (`"neumann"`) | Default boundary closure |
| $b_{\mathrm{nn}}, b_{\mathrm{cons}}$ | `transport.numerical.enforce_non_negative`, `track_conservation` | `bool` | `transport/numerical.yaml` (`true`, `true`) | Positivity and ledger flags |

Underlying sub-step evaluations inside `advance_advection_diffusion` also read `transport.numerical.time_integration_method` (`"euler"`), `implicit_midpoint_max_iter` ($50$), `implicit_midpoint_tolerance` ($10^{-10}$), `conservation_tolerance` ($10^{-6}$), and `limiter_epsilon` ($10^{-12}$).

## 5. Numerical Methods and Discretization

`AdvectionDiffusion.forward` delegates its sole numerical update through `_compute_iphs_advection_diffusion_update()` and `OperatorFactory` (`"transport.advection_diffusion"`) to `advance_advection_diffusion` (`relic/computation/operators/functional.py`):

- **`"sequential"` splitting**: advances full-step advection $c^* = \mathcal{L}_{\mathrm{adv}}(\Delta t)\,c^n$, full-step diffusion $\tilde{c} = \mathcal{L}_{\mathrm{diff}}(\Delta t)\,c^*$, and source addition $c^{n+1} = \tilde{c} + \Delta t\,S^n$ ($O(\Delta t)$ splitting error).
- **`"symmetric"` (Strang) splitting** (default): advances half-step source $c^{(1)} = c^n + \frac{\Delta t}{2}S^n$, half-step diffusion $c^{(2)} = \mathcal{L}_{\mathrm{diff}}(\frac{\Delta t}{2})\,c^{(1)}$, full-step advection $c^{(3)} = \mathcal{L}_{\mathrm{adv}}(\Delta t)\,c^{(2)}$, second half-step diffusion $c^{(4)} = \mathcal{L}_{\mathrm{diff}}(\frac{\Delta t}{2})\,c^{(3)}$, and second half-step source $c^{n+1} = c^{(4)} + \frac{\Delta t}{2}S^n$. Splitting commutator error is $O(\Delta t^2)$, whereas global step accuracy is $O(\Delta t^2)$ only with second-order sub-integrators (`"implicit_midpoint"`) and $O(\Delta t)$ under default `"euler"` sub-steps (*Methods of Ecological Analysis*, Chapter 26, Section 26.2.2). Because each diffusion half-step uses $h = \Delta t / 2$, explicit Euler stability admits full steps up to $\Delta t\sum_k K_k/\Delta x_k^2 \le 2\,\sigma_{\mathrm{diff,max}}$ (scaled by $2/3$ under `"dirichlet"`).
- **`"combined"` unsplit explicit update**: evaluates $c^{n+1} = c^n + \Delta t\bigl(R_{\mathrm{adv}}(c^n) + R_{\mathrm{diff}}(c^n) + S^n\bigr)$ under `"euler"` only (rejecting implicit integrators) and enforces $\max_i C_i + w \max_i \sigma_i \le \mathrm{CFL}_{\mathrm{adv,max}}$ with $w = 3$ for `"dirichlet"` and $w = 2$ for all other boundaries.

The integrated increment $\Delta c = c^{n+1} - c^n$ is returned once via `UpdateItem(mode=UpdateMode.ACCUMULATE, per_timestep=True)` with no second $\Delta t$ multiplication.

## 6. Conservation, Boundary Conditions, and Ledgers

Supported `boundary_method` closures are `"periodic"`, `"neumann"`, `"dirichlet"`, `"zero"`, `"open"`, and `"extrapolate"`:

- **Closed versus open routes**: `"periodic"` is the sole unconditional closed-domain closure for coupled advection-diffusion because zero normal gradient (`"neumann"`) closes diffusive face gradients while nonzero boundary velocity still transports advective mass across domain faces. Closed-domain `diagnostics.conservation_error` is emitted only when `track_conservation=True` and `boundary_method == "periodic"`.
- **Conditional boundary effort and total boundary flux port**: `"dirichlet"` supplies `tracer_concentration_boundary_state` to both advection and diffusion; `"open"` supplies it to advective inflow while keeping zero normal diffusive flux; closed configurations reject `tracer_concentration_boundary_state`. Every nonperiodic step packs the combined outward-positive face fluxes (`(1, adv) + (1, diff)` for sequential/combined; `(0.5, diff_1) + (1, adv) + (0.5, diff_2)` for symmetric) into `tracer_concentration_advection_diffusion_boundary_flux` with shape `[batch, total_face_dofs]` in deterministic `("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")` order, satisfying $\Delta M - \Delta t\int_{\Omega} S\,dV + \Delta t\sum_f j_{\mathrm{out},f} A_f = 0$.
- **Positivity**: when `enforce_non_negative` is enabled, negative initial, symmetric half-step, or final concentrations raise `ValueError` without clipping.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `AdvectionDiffusion` | `relic/scientific/transport/advection_diffusion.py` | `TransportMechanism` (`TRANSPORT_TYPE="advection_diffusion"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

Structural IPHS hooks implemented directly on `AdvectionDiffusion`: `declared_forms()`, `required_parameters()`, `build_hodge_terms()`, `reversible_structure()`, `dissipative_structure()`, and `ports()` (`advection_diffusion_boundary_flux` with `positive_power_convention="out_of_system"`).

## 8. Implementation Status

- `AdvectionDiffusion`: `[I]`, IPHS **Level 3** (`transport`, `port_equivalent`, certificate `transport_advection_diffusion_advection_diffusion.yaml`). Single canonical operator path via `_compute_iphs_advection_diffusion_update()` with typed cochain spaces, skew advective and PSD diffusive Hodge declarations, and batch-preserving outward boundary port routing.

Frozen certificate expressions (`transport_advection_diffusion_advection_diffusion.yaml`):

```text
dc/dt = -div(v * c) + div(K * grad(c)) + S
c_dot = -D_h(q_adv_h + q_diff_h) + G_h u_h; q_adv_h = i_v c_h; q_diff_h = -star_K d_h c_h.
```

## 9. References

- Bear, J. (1972). *Dynamics of Fluids in Porous Media*. American Elsevier.
- Crank, J. (1975). *The Mathematics of Diffusion* (2nd ed.). Oxford University Press.
- Hirani, A. N. (2003). *Discrete Exterior Calculus* (Ph.D. thesis). California Institute of Technology.
- Hundsdorfer, W., and Verwer, J. G. (2003). *Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations*. Springer.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
- Morton, K. W., and Mayers, D. F. (2005). *Numerical Solution of Partial Differential Equations* (2nd ed.). Cambridge University Press.
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
- Strang, G. (1968). On the construction and comparison of difference schemes. *SIAM Journal on Numerical Analysis*, 5(3), 506-517.
- Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers.
