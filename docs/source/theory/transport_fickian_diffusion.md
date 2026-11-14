# Fickian Diffusion Transport and Closed dPH Runtime Point

## 1. Process Definition and Scope

`FickianDiffusion` (`relic/scientific/transport/diffusion.py`) advances isotropic, spatially varying, or storage-axis directional parabolic diffusion of `tracer_concentration` ($\mathrm{kg\,m^{-3}}$) in $d \in \{1, 2, 3\}$ spatial dimensions on uniform Cartesian cell-centered grids (`SpatialScale.GRID_CELL`), supporting optional volumetric `source_term` ($\mathrm{kg\,m^{-3}\,s^{-1}}$), conditional Dirichlet face effort `tracer_concentration_boundary_state` ($\mathrm{kg\,m^{-3}}$), and outward boundary flux `tracer_concentration_diffusive_boundary_flux` ($\mathrm{kg\,m^{-2}\,s^{-1}}$). Process state naming is fixed to `tracer_concentration` so unit, provenance, and certificate bindings remain invariant across profiles, including the bounded 1D constant-eddy-diffusivity column specialization (`transport.diffusion.acp_manufactured_constant_eddy_diffusivity`, `transport.diffusion.acp_manufactured_vertical_grid_spacing`).

On its general constructor surface, `FickianDiffusion` is certified at IPHS **Level 3** (`transport_diffusion_fickian_diffusion.yaml`). Under the closed one-dimensional constructor selection `FickianDiffusion(spatial_dims=1, boundary_method in {"zero", "neumann", "open", "periodic"}, strict_runtime_point=True)` with zero `source_term` and unbatched 1D state, each emitted step is audited at IPHS **Level 4** (`dissipative_ph`, `transport_diffusion_fickian_diffusion_strict_runtime_point.yaml`). Full off-diagonal tensor diffusion, constructor state renaming, and Level 5 GENERIC thermal-entropy coupling remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 13 (Sections 13.3 and 13.5).

## 2. Governing Equations

### 2.1 Continuum Fickian Law, Anisotropic Diagonal Extension, and Gas-Phase Scaling

For concentration $c(\mathbf{x}, t)$ ($\mathrm{kg\,m^{-3}}$), scalar or storage-ordered diagonal diffusivity $D_k(\mathbf{x}) > 0$ ($\mathrm{m^2\,s^{-1}}$, written $K$ in discrete cochain operators), and volumetric source rate $S(\mathbf{x}, t)$ ($\mathrm{kg\,m^{-3}\,s^{-1}}$), Fick's first and second laws (Crank, 1975; Bird et al., 2007) are:

$$
\mathbf{J} = -D \nabla c, \qquad
\frac{\partial c}{\partial t} = \nabla \cdot (D \nabla c) + S = \sum_{k=1}^{d} \frac{\partial}{\partial x_k}\!\left(D_k \frac{\partial c}{\partial x_k}\right) + S
$$

When optional thermodynamic fields `temperature` $T$ ($\mathrm{K}$) or `pressure` $P$ ($\mathrm{Pa}$) are passed to `get_diffusivity(temperature, pressure)`, base diffusivity $D_0$ scales via Fuller et al. (1966):

$$
D(T, P) = D_0 \left(\frac{T}{T_{\mathrm{ref}}}\right)^{n_T} \left(\frac{P_{\mathrm{ref}}}{P}\right)
$$

preserving autograd gradients on manager-bound or learnable `torch.Tensor` coefficients.

### 2.2 Conservative Finite-Volume Face Law and Level-3 IPHS-DEC Balance

Across adjacent cell centers $L$ and $R$ separated by $\Delta x_k$, interior diffusive flux uses the harmonic face diffusivity $K_f$ (Patankar, 1980):

$$
J_f = -K_f \frac{c_R - c_L}{\Delta x_k}, \qquad
K_f = \frac{2 K_L K_R}{K_L + K_R}
$$

In discrete exterior calculus notation with cell cochain $c_h$, exterior difference $X_h = d_h c_h$, positive-definite face conductance $\star_K = \star_K^{\mathsf{T}} \succeq 0$, discrete divergence $D_h$, and open source/boundary input $G_h u_h$:

$$
\frac{dc_h}{dt} = -D_h q_h + G_h u_h, \qquad
q_h = -\star_K X_h, \qquad
\sigma_h = X_h^{\mathsf{T}} \star_K X_h \ge 0
$$

### 2.3 Quadratic Concentration-Norm Storage and Semi-Discrete Balance

The structural audit defines the unweighted quadratic concentration-norm storage functional $H_h(c_h)$ ($\mathrm{kg^2\,m^{-6}}$) and conjugate effort $e_h$ ($\mathrm{kg\,m^{-3}}$):

$$
H_h(c_h) = \frac{1}{2} c_h^{\mathsf{T}} c_h, \qquad
e_h = \frac{\delta H_h}{\delta c_h} = c_h
$$

As established in *Methods of Ecological Analysis*, Chapter 13 (Section 13.5), $H_h$ is a mathematical $L^2$ Lyapunov norm functional ($\mathrm{kg^2\,m^{-6}}$) whose dissipation rate $\sigma_0$ carries units $\mathrm{kg^2\,m^{-6}\,s^{-1}}$; it represents neither physical energy in joules nor thermodynamic entropy production. Its semi-discrete balance across interior faces and boundary faces $\partial\Omega_k$ is:

$$
\frac{dH_h}{dt} = -\sigma_0 + \sum_i c_i S_i - P_{\mathrm{out},0}, \qquad
P_{\mathrm{out},0} = \sum_{k=1}^{d} \sum_{f \in \partial\Omega_k} \frac{c_f J_{\mathrm{out},f}}{\Delta x_k}
$$

Interior faces (and the periodic seam once) contribute $K_f (\Delta c_f / \Delta x_k)^2$ to $\sigma_0$; `"dirichlet"` half-cell boundary faces ($\Delta x_k / 2$) contribute $2 K_{\mathrm{edge}}((c_{\mathrm{edge}} - c_f)/\Delta x_k)^2$; `"extrapolate"` faces contribute $\frac{1}{2} K_{\mathrm{edge}}(\Delta c_{\mathrm{edge}}/\Delta x_k)^2$; `"zero"`, `"neumann"`, and `"open"` add zero boundary dissipation. Under discrete forward Euler stepping, $H_h(c_h^{n+1}) - H_h(c_h^n)$ includes the discrete quadratic term $\frac{1}{2}\Delta t^2 \lVert \dot{c}_h \rVert_2^2$.

### 2.4 Level 4 dPH Runtime Point (`level_4_dph_runtime_point`)

Under constructor selection `FickianDiffusion(spatial_dims=1, boundary_method in {"zero", "neumann", "open", "periodic"}, strict_runtime_point=True)` on an unbatched 1D grid ($N \ge 2$ cells) with absent or identically zero `source_term` (nonzero `source_term` raises `ValueError` before integration), `_closed_dph_runtime_point` assembles cell-to-face difference matrix $d_h$ (interior faces plus periodic seam), harmonic face vector $K_f$, and symmetric positive-semidefinite global resistance matrix $R_h$:

$$
R_h = \frac{1}{\Delta x^2} d_h^{\mathsf{T}} \operatorname{diag}(K_f)\,d_h, \qquad
\frac{dc_h}{dt} = (J_h - R_h) e_H, \qquad
J_h = 0, \qquad
e_H = \frac{\partial H_h}{\partial c_h} = c_h
$$

$$
\frac{dH_h}{dt} = -e_H^{\mathsf{T}} R_h e_H = -\sum_{f} K_f \left(\frac{\Delta c_f}{\Delta x}\right)^2 \le 0
$$

On every emitted step, `forward()` verifies via autograd inside `IPHSStructureSpec.canonical_drift` and `IPHSStructureSpec.validate(required_level=4, state=c_h, energy_value=H_h)` that $-(R_h e_H)$ reproduces the production finite-volume rate within `validation.iphs.runtime_point_tolerance` and exposes the audit record through `last_strict_iphs_audit()` (cleared on any rejected step).

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $c_h$ | Transported tracer concentration state (`ACCUMULATE`, `per_timestep=True`) | $\mathrm{kg\,m^{-3}}$ | `tracer_concentration` | `state` |
| $S_h$ | Optional volumetric source or sink rate (must be zero when `strict_runtime_point=True`) | $\mathrm{kg\,m^{-3}\,s^{-1}}$ | `source_term` | `flux` |
| $c_{\partial\Omega}$ | Conditional external Dirichlet face concentration (required only for `"dirichlet"`) | $\mathrm{kg\,m^{-3}}$ | `tracer_concentration_boundary_state` | `forcing` |
| $q_{\partial\Omega}$ | Conditional outward-positive boundary face flux cochain (`"dirichlet"` and `"extrapolate"`, `REPLACE`, `per_timestep=False`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `tracer_concentration_diffusive_boundary_flux` | `flux` |
| $\sigma_{\max}$ | Maximum cell parabolic diffusion stability number $\max_i \Delta t \sum_k K_{k,i}/\Delta x_k^2$ | $\mathrm{1}$ | `diagnostics.diffusion_stability` | `diagnostic` |
| $\varepsilon_{\mathrm{cons}}$ | Conditional normalized finite-volume mass-conservation residual (`track_conservation=True`) | $\mathrm{1}$ | `diagnostics.conservation_error` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $D, K$ | `transport.diffusion.diffusivity` (or profile-selected `diffusivity_parameter_key`) | $\mathrm{m^2\,s^{-1}}$ | `transport/diffusion.yaml` ($10^{-5}$) | Crank (1975); Bird et al. (2007) |
| $\Delta x$ | `transport.diffusion.grid_spacing` (or profile-selected `grid_spacing_parameter_key`) | $\mathrm{m}$ | `transport/diffusion.yaml` ($1.0$) | Uniform metric spacing |
| $n_T$ | `transport.diffusion.temperature_exponent` | $\mathrm{1}$ | `transport/diffusion.yaml` ($1.5$) | Fuller et al. (1966) |
| $K_{\mathrm{ACP}}, \Delta z_{\mathrm{ACP}}$ | `transport.diffusion.acp_manufactured_constant_eddy_diffusivity`, `acp_manufactured_vertical_grid_spacing` | $\mathrm{m^2\,s^{-1}}$, $\mathrm{m}$ | `transport/diffusion.yaml` ($1.0$, $10.0$) | Passive-tracer 1D column benchmark |
| $\sigma_{\mathrm{diff,max}}$ | `transport.numerical.cfl_diffusion_max` | $\mathrm{1}$ | `transport/numerical.yaml` ($0.5$) | Explicit Von Neumann bound |
| $\tau_{\mathrm{cons}}, b_{\mathrm{cons}}, b_{\mathrm{nn}}$ | `transport.numerical.conservation_tolerance`, `track_conservation`, `enforce_non_negative` | $\mathrm{1}$, `bool`, `bool` | `transport/numerical.yaml` ($10^{-6}$, `true`, `true`) | Patankar (1980) |
| $b_{\mathrm{bc}}$ | `transport.numerical.default_boundary_type` | `str` | `transport/numerical.yaml` (`"neumann"`) | Default boundary closure |
| $N_{\mathrm{fp}}, \tau_{\mathrm{fp}}$ | `transport.numerical.implicit_midpoint_max_iter`, `implicit_midpoint_tolerance` | $\mathrm{1}$ | `transport/numerical.yaml` ($50$, $10^{-10}$) | Semi-implicit midpoint solver |
| $T_{\mathrm{ref}}, P_{\mathrm{ref}}$ | `reference_temperature`, `reference_pressure` (via `get_physical_constant`) | $\mathrm{K}$, $\mathrm{Pa}$ | `physical/constants.yaml` | Fuller scaling reference state |

## 5. Numerical Methods and Discretization

`FickianDiffusion.forward` delegates once through `OperatorFactory` (`"transport.diffusion"`) to `apply_diffusion` (`relic/computation/operators/functional.py`):

- **Storage-ordered diffusivity layouts**: grid spacing and directional diffusivity follow tensor storage order `(dy, dx)` in 2D and `(dz, dy, dx)` in 3D. Numbers and 0-D tensors resolve to `"scalar_field"` (and bind `TransportMechanism` base scalar diffusivity); any tensor with $\mathrm{ndim} \ge 1$ must explicitly declare `diffusivity_layout="scalar_field"` or `"directional"` without shape-based inference.
- **Time integration and explicit stability**: `time_integration="explicit"` selects operator method `"euler"` and enforces $\max_i \Delta t \sum_k K_{k,i}/\Delta x_k^2 \le \sigma_{\mathrm{diff,max}}$ ($0.5$) for `"zero"`, `"neumann"`, `"open"`, `"periodic"`, and `"extrapolate"`, or $\frac{2}{3}\sigma_{\mathrm{diff,max}}$ ($1/3$) for `"dirichlet"` half-cell boundary conductance. `time_integration="semi_implicit"` selects fixed-point `"implicit_midpoint"` and bypasses the explicit Euler limit. Unstable explicit steps raise `ValueError` with zero internal time-step rescaling.

The integrated step increment $\Delta c = c^{n+1} - c^n$ is returned once as `UpdateItem(mode=UpdateMode.ACCUMULATE, per_timestep=True)`.

## 6. Conservation, Boundary Conditions, and Ledgers

Supported `boundary_method` closures are `"periodic"`, `"zero"`, `"neumann"`, `"open"`, `"dirichlet"`, and `"extrapolate"`:

- **Closed diffusive boundaries**: `"periodic"`, `"zero"`, `"neumann"`, and `"open"` enforce zero normal diffusive boundary exchange and reject `tracer_concentration_boundary_state`.
- **Open diffusive boundaries**: `"dirichlet"` requires `tracer_concentration_boundary_state` evaluated at physical half-cell distance $\Delta x_k / 2$; `"extrapolate"` reconstructs linear ghost cells $2 c_{\mathrm{edge}} - c_{\mathrm{adj}}$. Both emit outward-positive `tracer_concentration_diffusive_boundary_flux` (`UpdateMode.REPLACE`, `per_timestep=False`) in face order `("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")` using integrator-matched boundary quadrature.
- **Mass ledger and positivity**: interior harmonic face fluxes cancel telescopically, satisfying $\Delta M - \Delta t\int_{\Omega} S\,dV + \Delta t\sum_f J_{\mathrm{out},f} A_f = 0$. When `enforce_non_negative` is enabled, negative initial, boundary, or post-step concentrations raise `ValueError` without clamping.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `FickianDiffusion` | `relic/scientific/transport/diffusion.py` | `TransportMechanism` (`TRANSPORT_TYPE="diffusion"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

Structural and runtime audit hooks on `FickianDiffusion`: `declared_forms()`, `required_parameters()`, `build_hodge_terms()`, `build_hodge_contract()`, `energy_terms()`, `entropy_terms()`, `dissipative_structure()`, `ports()`, `_closed_dph_runtime_point()`, and `last_strict_iphs_audit()`.

## 8. Implementation Status

- `FickianDiffusion` (default general surface): `[I]`, IPHS **Level 3** (`transport`, `port_equivalent`, certificate `transport_diffusion_fickian_diffusion.yaml`). Declares typed primal/dual cochain forms, local positive-definite material Hodge diagonal $\star_K$, quadratic $L^2$ norm audit hooks, and outward diffusive matter port `diffusive_boundary_flux`.
- `FickianDiffusion` (closed 1D runtime-point selection `constructor_selection: {strict_runtime_point: true, spatial_dims: 1}`): `[I]`, IPHS **Level 4** (`dissipative_ph`, `port_equivalent`, certificate `transport_diffusion_fickian_diffusion_strict_runtime_point.yaml`). Verifies global finite-volume resistance drift $\dot{c}_h = -R_h e_H$ and monotonic norm decay $\dot{H}_h = -e_H^{\mathsf{T}} R_h e_H \le 0$ on every emitted step when paired with a governed step witness.

Frozen Level-3 certificate expressions (`transport_diffusion_fickian_diffusion.yaml`):

```text
dc/dt = div(K * grad(c)) + S
c_dot = -D_h q_h + G_h u_h; q_h = -star_K X_h; X_h = d_h c_h; H_h = 0.5 * c_h.T c_h; sigma = X_h.T star_K X_h >= 0; dH/dt = -sigma + e_h.T G_h u_h - P_boundary.
```

Frozen Level-4 constructor-selection certificate expressions (`transport_diffusion_fickian_diffusion_strict_runtime_point.yaml`):

```text
dc/dt = div(K * grad(c)) + S
c_dot = (J_h - R_h) e_H; J_h = 0; R_h = d_h^T diag(K_f) d_h / dx^2; e_H = c_h; dH_h/dt = -e_H^T R_h e_H <= 0
```

## 9. References

- Bird, R. B., Stewart, W. E., and Lightfoot, E. N. (2007). *Transport Phenomena* (2nd ed.). John Wiley & Sons.
- Crank, J. (1975). *The Mathematics of Diffusion* (2nd ed.). Oxford University Press.
- Fuller, E. N., Schettler, P. D., and Giddings, J. C. (1966). A new method for prediction of binary gas-phase diffusion coefficients. *Industrial & Engineering Chemistry*, 58(5), 18-27.
- Hirani, A. N. (2003). *Discrete Exterior Calculus* (Ph.D. thesis). California Institute of Technology.
- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
- Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers.
- van der Schaft, A., and Jeltsema, D. (2014). Port-Hamiltonian systems theory: An introductory overview. *Foundations and Trends in Systems and Control*, 1(2-3), 173-378.
