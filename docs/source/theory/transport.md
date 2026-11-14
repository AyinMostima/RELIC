# Shared Continuum Transport Theory and Splitting Contracts

## 1. Process Definition and Scope

This document defines the shared mathematical, numerical, stability, and boundary-ledger contracts governing the three RELIC continuum transport Functors on uniform orthogonal cell-centered grids (`SpatialScale.GRID_CELL`) and their shared finite-volume kernels in `relic/computation/operators/functional.py`:

- `LinearAdvection` (`relic/scientific/transport/advection.py`): conservative hyperbolic tracer advection (`transport_linear_advection.md`).
- `FickianDiffusion` (`relic/scientific/transport/diffusion.py`): parabolic Fickian molecular or eddy diffusion (`transport_fickian_diffusion.md`).
- `AdvectionDiffusion` (`relic/scientific/transport/advection_diffusion.py`): coupled advective-diffusive transport via operator splitting or combined explicit stepping (`transport_advection_diffusion.md`).

All three Functors implement the `TransportMechanism` contract via `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`). Unstructured mesh transport, full off-diagonal tensor dispersion, and internal adaptive time-step selection inside transport Functors remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 13 (Sections 13.3 and 13.5) and Chapter 26 (Section 26.2).

## 2. Governing Equations

### 2.1 Hyperbolic Advection, Parabolic Diffusion, and Coupled ADR PDEs

For scalar tracer concentration $c(\mathbf{x}, t)$ (default $\mathrm{kg\,m^{-3}}$), velocity vector $\mathbf{v}(\mathbf{x}, t)$ ($\mathrm{m\,s^{-1}}$), scalar or diagonal diffusivity $D(\mathbf{x}, t)$ ($\mathrm{m^2\,s^{-1}}$, written $K$ in discrete cochain certificates), and volumetric source-sink rate $S(\mathbf{x}, t)$ ($\mathrm{kg\,m^{-3}\,s^{-1}}$), the three continuum conservation laws are:

1. **Conservative linear advection** (plain-text contract: `dc/dt + div(v * c) = S`):

$$
\frac{\partial c}{\partial t} + \nabla \cdot (\mathbf{v} c) = S, \qquad \mathbf{J}_{\mathrm{adv}} = \mathbf{v} c
$$

2. **Fickian diffusion** (plain-text contract: `dc/dt = div(D * grad(c)) + S`):

$$
\mathbf{J}_{\mathrm{diff}} = -D \nabla c, \qquad
\frac{\partial c}{\partial t} = \nabla \cdot (D \nabla c) + S
$$

3. **Combined advection-diffusion-reaction**:

$$
\frac{\partial c}{\partial t} + \nabla \cdot (\mathbf{v} c) = \nabla \cdot (D \nabla c) + S
$$

### 2.2 Discrete Cochain Operators and Stability Bounds

On a $d$-dimensional uniform Cartesian cell-centered mesh ($d \in \{1, 2, 3\}$ for diffusion and coupled ADR; $d \ge 1$ for pure advection) with storage-ordered spacings $(\Delta z, \Delta y, \Delta x)$, cell primal $d$-cochain $c_h$ evolves through discrete divergence $D_h$, exterior difference $d_h$, interior contraction $i_v$, positive diagonal face conductance $\star_K$, and open port operator $G_h$:

$$
\frac{dc_h}{dt} = -D_h\bigl(q_{\mathrm{adv},h} + q_{\mathrm{diff},h}\bigr) + G_h u_h, \qquad
q_{\mathrm{adv},h} = i_v c_h, \qquad
q_{\mathrm{diff},h} = -\star_K d_h c_h
$$

Explicit forward Euler (`"euler"`) integration enforces three dimensionless stability bounds:

1. **One-dimensional reference and multidimensional outward-face Courant bound** (plain-text contract: `CFL = |u| * dt / dx <= 1`):

$$
\mathrm{CFL}_{1\mathrm{D}} = \frac{\lvert u \rvert\,\Delta t}{\Delta x} \le 1, \qquad
C_i = \Delta t \sum_{k=1}^{d} \frac{\max(u_{k,i+1/2}, 0) - \min(u_{k,i-1/2}, 0)}{\Delta x_k} \le \mathrm{CFL}_{\mathrm{adv,max}}
$$

2. **One-dimensional reference and multidimensional parabolic diffusion number** (plain-text contract: `r = D * dt / dx^2 <= 0.5`):

$$
r = \frac{D\,\Delta t}{\Delta x^2} \le 0.5, \qquad
\sigma_i = h \sum_{k=1}^{d} \frac{K_{k,i}}{\Delta x_k^2} \le
\begin{cases}
\dfrac{2}{3}\,\sigma_{\mathrm{diff,max}}, & \text{boundary\_method} = \text{"dirichlet"} \\[1.0ex]
\sigma_{\mathrm{diff,max}}, & \text{otherwise}
\end{cases}
$$

where $h$ is the sub-step duration ($h = \Delta t / 2$ on each symmetric diffusion half-step) and the $2/3$ Dirichlet factor reflects half-cell face distance $\Delta x / 2$ (worst-case diagonal $3K/\Delta x^2$ versus interior $2K/\Delta x^2$; Patankar, 1980).

3. **Combined unsplit explicit ADR bound**:

$$
\max_i C_i + w \max_i \sigma_i \le \mathrm{CFL}_{\mathrm{adv,max}}, \qquad
w = \begin{cases} 3, & \text{boundary\_method} = \text{"dirichlet"} \\ 2, & \text{otherwise} \end{cases}
$$

Violating any active explicit bound raises `ValueError`; transport Functors never alter $\Delta t$ internally (sub-stepping is owned by the simulator or `FunctorMetadata.substep_count`).

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $c$ | Transported tracer concentration state (`ACCUMULATE`, `per_timestep=True`) | $\mathrm{kg\,m^{-3}}$ (or declared `tracer_units`) | `tracer_concentration` | `state` |
| $\mathbf{v}$ | Spatial advective velocity vector field | $\mathrm{m\,s^{-1}}$ | `velocity_field` (or fixed constructor `velocity`) | `forcing` / `state` |
| $D, K$ | Scalar field or directional diffusivity tensor | $\mathrm{m^2\,s^{-1}}$ | `diffusivity` | `parameter` / `state` |
| $S$ | Volumetric tracer source or sink rate | $\mathrm{kg\,m^{-3}\,s^{-1}}$ | `source_term` | `flux` |
| $c_{\partial\Omega}$ | Explicit external tracer state at open/Dirichlet boundaries | $\mathrm{kg\,m^{-3}}$ | `tracer_concentration_boundary_state` | `forcing` |
| $q_{\partial\Omega}$ | Packed outward-positive boundary-face flux cochain (`REPLACE`, `per_timestep=False`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `tracer_concentration_advective_boundary_flux`, `tracer_concentration_diffusive_boundary_flux`, `tracer_concentration_advection_diffusion_boundary_flux` | `flux` |
| $C_{\max}, \sigma_{\max}, \mathrm{Pe}_{\max}, \varepsilon_{\mathrm{cons}}$ | Maximum Courant number, diffusion number, cell Peclet number, and relative conservation residual | $\mathrm{1}$ | `diagnostics.cfl_number`, `diagnostics.diffusion_stability`, `diagnostics.peclet_number`, `diagnostics.conservation_error` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\mathrm{CFL}_{\mathrm{adv,max}}$ | `transport.numerical.cfl_advection_max` | $\mathrm{1}$ | `transport/numerical.yaml` ($1.0$) | LeVeque (2002) |
| $\sigma_{\mathrm{diff,max}}$ | `transport.numerical.cfl_diffusion_max` | $\mathrm{1}$ | `transport/numerical.yaml` ($0.5$) | Crank (1975); Morton and Mayers (2005) |
| $m_{\mathrm{time}}$ | `transport.numerical.time_integration_method` | `str` | `transport/numerical.yaml` (`"euler"`) | Hairer et al. (1993) |
| $N_{\mathrm{fp}}, \tau_{\mathrm{fp}}$ | `transport.numerical.implicit_midpoint_max_iter`, `implicit_midpoint_tolerance` | $\mathrm{1}$ | `transport/numerical.yaml` ($50$, $10^{-10}$) | Fixed-point midpoint/trapezoidal solver |
| $\tau_{\mathrm{cons}}, b_{\mathrm{cons}}$ | `transport.numerical.conservation_tolerance`, `track_conservation` | $\mathrm{1}$, `bool` | `transport/numerical.yaml` ($10^{-6}$, `true`) | Patankar (1980) |
| $b_{\mathrm{nn}}, c_{\min}$ | `transport.numerical.enforce_non_negative`, `non_negative_min_value` | `bool`, $\mathrm{kg\,m^{-3}}$ | `transport/numerical.yaml` (`true`, $0.0$) | Positivity audit contract |
| $b_{\mathrm{bc}}, \varepsilon_{\mathrm{lim}}$ | `transport.numerical.default_boundary_type`, `limiter_epsilon` | `str`, $\mathrm{1}$ | `transport/numerical.yaml` (`"neumann"`, $10^{-12}$) | Sweby (1984) TVD ratio regularizer |

Keys `cfl_safety_factor`, `min_timestep`, `max_timestep`, `adaptive_timestep_factor`, and `default_boundary_value` exist in `transport/numerical.yaml` for external driver tooling and are unread by `LinearAdvection`, `FickianDiffusion`, and `AdvectionDiffusion`.

## 5. Numerical Methods and Discretization

Finite-volume spatial operators in `relic/computation/operators/functional.py` evaluate conservative face-flux differences across interior and boundary faces:

- **Advection schemes and integrator pairing**: `"upwind"` pairs with `"euler"`, `"implicit_midpoint"`, or `"crank_nicolson"`; one-step Lax-Wendroff and TVD flux-limited schemes (`"lax_wendroff"`, `"superbee"`, `"van_leer"`) embed $\Delta t$ directly in their antidiffusive flux and require `"euler"`; `"central"` (operator-layer only) requires an implicit A-stable integrator because forward Euler FTCS has amplification $\lvert g \rvert^2 = 1 + C^2\sin^2(k\Delta x) > 1$.
- **Diffusion face conductance and layout**: interior faces use harmonic mean diffusivity $K_{i+1/2} = 2 K_i K_{i+1}/(K_i + K_{i+1})$ (`harmonic_mean`, zero when either adjacent cell has $K = 0$). Any tensor diffusivity with $\mathrm{ndim} \ge 1$ requires explicit `diffusivity_layout in {"scalar_field", "directional"}`; shape-based layout inference is prohibited.
- **Operator splitting modes** (`advance_advection_diffusion`):
  - `Sequential splitting`: Lie-Trotter composition $c^* = \mathcal{L}_{\mathrm{adv}}(\Delta t)\,c^n$, $\tilde{c} = \mathcal{L}_{\mathrm{diff}}(\Delta t)\,c^*$, $c^{n+1} = \tilde{c} + \Delta t\,S^n$, giving $O(\Delta t)$ splitting accuracy.
  - `Symmetric (Strang) splitting`: five-stage Strang (1968) composition $c^{(1)} = c^n + \frac{\Delta t}{2}S^n$, $c^{(2)} = \mathcal{L}_{\mathrm{diff}}(\frac{\Delta t}{2})\,c^{(1)}$, $c^{(3)} = \mathcal{L}_{\mathrm{adv}}(\Delta t)\,c^{(2)}$, $c^{(4)} = \mathcal{L}_{\mathrm{diff}}(\frac{\Delta t}{2})\,c^{(3)}$, $c^{n+1} = c^{(4)} + \frac{\Delta t}{2}S^n$. Splitting commutator error is $O(\Delta t^2)$; global step order reaches $O(\Delta t^2)$ only when paired with second-order sub-integrators (`"implicit_midpoint"`), remaining $O(\Delta t)$ under default `"euler"` sub-steps (*Methods of Ecological Analysis*, Chapter 26, Theorem 26.3).
  - `Combined` explicit update: single forward Euler step $c^{n+1} = c^n + \Delta t\bigl(R_{\mathrm{adv}}(c^n) + R_{\mathrm{diff}}(c^n) + S^n\bigr)$, rejecting implicit time integrators.

All three Functors commit discrete step increments $\Delta c = c^{n+1} - c^n$ directly as `UpdateItem(mode=UpdateMode.ACCUMULATE, per_timestep=True)` without calling `TransportMechanismFunctor.finalize_transport_updates`.

## 6. Conservation, Boundary Conditions, and Ledgers

Supported `boundary_method` options are `"periodic"`, `"neumann"`, `"dirichlet"`, `"zero"`, `"open"`, and `"extrapolate"`. Periodic domains emit no boundary port; nonperiodic domains pack outward-positive face fluxes in canonical face order `("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")` via `pack_outward_boundary_flux_cochain` into `[batch, total_face_dofs]` (`UpdateMode.REPLACE`, `per_timestep=False`) using integrator-matched quadrature ($j(c^n)$ for `"euler"`, $j(\frac{1}{2}(c^n + c^{n+1}))$ for `"implicit_midpoint"`, and $\frac{1}{2}(j(c^n) + j(c^{n+1}))$ for `"crank_nicolson"`). `_finite_volume_transport_conservation_error` audits exact cell-volume storage, source, and boundary-area closure:

$$
r_{\mathrm{ledger}} = M^{n+1} - M^n - \Delta t\int_{\Omega} S\,dV + \Delta t\sum_{f \in \partial\Omega} j_{\mathrm{out},f} A_f, \qquad
\varepsilon_{\mathrm{cons}} = \max_{\mathrm{batch}} \frac{\lvert r_{\mathrm{ledger}}\rvert}{\max\bigl(\lVert M^n\rVert_1, \lVert M^{n+1}\rVert_1, \lvert\Delta M_{\mathrm{src}}\rvert, \lvert\Delta M_{\partial\Omega}\rvert, \epsilon_{\mathrm{mach}}\bigr)}
$$

When `enforce_non_negative` is enabled, negative input or post-step concentrations raise `ValueError` with zero silent clamping.

## 7. Code Mapping

| Functor / Module | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `LinearAdvection` | `relic/scientific/transport/advection.py` | `TransportMechanism` (`TRANSPORT_TYPE="advection"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |
| `FickianDiffusion` | `relic/scientific/transport/diffusion.py` | `TransportMechanism` (`TRANSPORT_TYPE="diffusion"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |
| `AdvectionDiffusion` | `relic/scientific/transport/advection_diffusion.py` | `TransportMechanism` (`TRANSPORT_TYPE="advection_diffusion"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |
| Finite-volume transport kernels | `relic/computation/operators/functional.py` | Operator computation layer (`apply_advection`, `apply_diffusion`, `advance_advection_diffusion`) | None | Invoked inside transport Functors |

## 8. Implementation Status

- `LinearAdvection`: `[I]`, IPHS **Level 3** (`transport`, certificate `transport_advection_linear_advection.yaml`).
- `FickianDiffusion`: `[I]`, IPHS **Level 3** on its general default constructor surface (`transport_diffusion_fickian_diffusion.yaml`) and IPHS **Level 4** (`closed_dph`) on the constructor-selected 1D closed zero-source point `{strict_runtime_point: true, spatial_dims: 1}` (`transport_diffusion_fickian_diffusion_strict_runtime_point.yaml`).
- `AdvectionDiffusion`: `[I]`, IPHS **Level 3** (`transport`, certificate `transport_advection_diffusion_advection_diffusion.yaml`).

Shared plain-text audit contracts:

```text
dc/dt + div(v * c) = S
dc/dt = div(D * grad(c)) + S
CFL = |u| * dt / dx <= 1
r = D * dt / dx^2 <= 0.5
Sequential splitting
Symmetric (Strang) splitting
```

## 9. References

- Crank, J. (1975). *The Mathematics of Diffusion* (2nd ed.). Oxford University Press.
- Hairer, E., Nørsett, S. P., and Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems* (2nd ed.). Springer.
- Hundsdorfer, W., and Verwer, J. G. (2003). *Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations*. Springer.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
- Morton, K. W., and Mayers, D. F. (2005). *Numerical Solution of Partial Differential Equations* (2nd ed.). Cambridge University Press.
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
- Strang, G. (1968). On the construction and comparison of difference schemes. *SIAM Journal on Numerical Analysis*, 5(3), 506-517.
- Sweby, P. K. (1984). High resolution schemes using flux limiters for hyperbolic conservation laws. *SIAM Journal on Numerical Analysis*, 21(5), 995-1011.
