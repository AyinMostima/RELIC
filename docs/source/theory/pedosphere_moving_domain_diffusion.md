# Pedosphere Moving-Domain Column Diffusion

## 1. Process Definition and Scope

`MovingDomainDiffusionFunctor` (`relic/scientific/pedosphere/transport/moving_domain_diffusion.py`) solves one-dimensional solute diffusion on a time-varying vertical soil column $[0, L(t)]$ whose lower boundary moves at prescribed velocity $w_b = \frac{dL}{dt}$ (`[I]`). Typical physical realizations include an unsaturated vadose column above a rising or falling water table or an expanding root-zone profile.

The Functor implements an Arbitrary Lagrangian-Eulerian (ALE) finite-volume discretization on a uniformly stretched reference coordinate $\xi \in [0, 1]$ (`geometry_category = "ale"`, `geometry_mapping = "uniform_stretching"`, `TRANSPORT_TYPE = "diffusion"`). Cell volumes satisfy the discrete Geometric Conservation Law (GCL) identically so uniform concentration fields remain invariant under pure mesh stretching or compression, while boundary entrainment ($w_b > 0$) and release ($w_b < 0$) close the column solute budget to machine precision. Non-uniform mesh mappings, adaptive mesh refinement, and cross-mesh state remapping are not implemented (`[N]`).

## 2. Governing Equations

With reference coordinate $\xi \in [0, 1]$, physical depth $x(\xi, t) = \xi\,L(t)$ measured downward from the fixed upper surface $x = 0$, Jacobian $J_{\mathrm{ALE}} = L(t)$, and linear mesh velocity $w(x, t) = \xi\,w_b$ where $\frac{dL}{dt} = w_b$, the continuous GCL $\frac{\partial J_{\mathrm{ALE}}}{\partial t} = J_{\mathrm{ALE}}\,\frac{\partial w}{\partial x} = w_b$ holds identically. Over a moving control volume $[x_k(t), x_{k+1}(t)]$ with diffusive Fickian flux $F = -D\,\frac{\partial c}{\partial x}$ and diffusivity $D \ge 0$ ($\mathrm{m^2\,s^{-1}}$), Reynolds transport yields the ALE conservation law

$$
\frac{d}{dt}\int_{x_k(t)}^{x_{k+1}(t)} c\,dx = -\left(F_{k+1} - F_k\right) + c(x_{k+1}, t)\,w_{k+1} - c(x_k, t)\,w_k.
$$

Partitioning $[0, L(t)]$ into $N \ge 2$ uniform cells of width $\Delta x = L(t) / N$ (cell volume per unit area $V_i(t) = \Delta x$) sets face velocities

$$
w_k = \frac{k}{N}\,w_b, \qquad k = 0, 1, \dots, N,
$$

with fixed top face $w_0 = 0$ and moving bottom boundary $w_N = w_b$. Over explicit step $\Delta t > 0$ ($\mathrm{s}$), column length and cell volumes advance according to the exact discrete GCL:

$$
L(t + \Delta t) = L(t) + w_b\,\Delta t, \qquad V_i(t + \Delta t) = V_i(t) + \Delta t\,\left(w_{i+1} - w_i\right) = \frac{L(t + \Delta t)}{N}.
$$

Let $\mathrm{rate}_i$ ($\mathrm{kg\,m^{-3}\,s^{-1}}$) denote the closed-boundary 1-D finite-volume diffusive tendency returned by `_compute_diffusion_rate_and_boundary_flux_1d` on grid spacing $\Delta x = L(t)/N$ with zero diffusive flux at $k = 0$ and $k = N$. The swept mass flux $s_k = c_{\mathrm{face},k}\,w_k$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) uses upwind selection relative to face motion ($-w_k$, because fluid crosses face $k$ from cell $k$ when $w_k > 0$ and from cell $k-1$ when $w_k < 0$):

$$
c_{\mathrm{face},0} = c_0 \quad (w_0 = 0), \qquad
c_{\mathrm{face},k} = \begin{cases} c_k, & w_k > 0, \\ c_{k-1}, & w_k \le 0, \end{cases} \quad (1 \le k \le N-1),
$$

$$
c_{\mathrm{face},N} = \begin{cases} c_b, & w_b > 0, \\ c_{N-1}, & w_b \le 0, \end{cases}
$$

where $0$-based cell indices $i \in \{0, \dots, N-1\}$ match PyTorch indexing and $c_b \ge 0$ ($\mathrm{kg\,m^{-3}}$) is the prescribed entrained boundary concentration. Cell areal masses $m_i(t) = c_i(t)\,V_i(t)$ ($\mathrm{kg\,m^{-2}}$) and updated concentrations $c_i(t + \Delta t)$ advance via

$$
m_i(t + \Delta t) = m_i(t) + \Delta t\,\left[\mathrm{rate}_i\,\Delta x + s_{i+1} - s_i\right], \qquad c_i(t + \Delta t) = \frac{m_i(t + \Delta t)}{V_i(t + \Delta t)}.
$$

On the time-dependent $L^2$ Hodge storage functional $H = \frac{1}{2}\,c^{\mathsf{T}} \star c$ with $\star = \mathrm{diag}(L/N)$ and metric time derivative $\dot{\star} = \mathrm{diag}(w_b/N)$, the moving-geometry Hodge power correction is

$$
P_{\mathrm{geo}} = -\frac{1}{2}\,c^{\mathsf{T}}\,\dot{\star}\,c = -\frac{w_b}{2 N}\sum_{i=0}^{N-1} c_i^2.
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| :--- | :--- | :--- | :--- | :--- |
| $c$ | 1-D column cell solute concentration vector ($N \ge 2$) | $\mathrm{kg\,m^{-3}}$ | `moving_domain_solute_concentration` | `input` / `output` (`state increment`) |
| $L$ | Physical column length | $\mathrm{m}$ | `moving_domain_length` | `input` / `output` (`state increment`) |
| $w_b$ | Prescribed lower-boundary velocity $\frac{dL}{dt}$ | $\mathrm{m\,s^{-1}}$ | `moving_domain_boundary_velocity` | `input` (`forcing`) |
| $c_b$ | Solute concentration of fluid entrained when $w_b > 0$ | $\mathrm{kg\,m^{-3}}$ | `moving_boundary_solute_concentration` | `input` (`forcing`) |
| $s_N$ | Swept mass flux across moving lower boundary $c_{\mathrm{face},N}\,w_b$ | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.moving_domain.boundary_mass_flux` | `diagnostic` |
| $r_{\mathrm{mass}}$ | Column step mass-balance residual $\sum_i m_i(t+\Delta t) - \sum_i m_i(t) - \Delta t\,s_N$ | $\mathrm{kg\,m^{-2}}$ | `diagnostics.moving_domain.mass_balance_residual` | `diagnostic` |
| $r_{\mathrm{gcl}}$ | Maximum discrete GCL volume residual $\max_i \lvert V_i(t+\Delta t) - V_i(t) - \Delta t\,(w_{i+1}-w_i) \rvert$ | $\mathrm{m}$ | `diagnostics.moving_domain.gcl_residual` | `diagnostic` |
| $P_{\mathrm{geo}}$ | Moving-metric Hodge energy rate correction $-\frac{1}{2} c^{\mathsf{T}}\dot{\star}c$ | $\mathrm{kg^2\,m^{-5}\,s^{-1}}$ | `diagnostics.moving_domain.geometry_power` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
| :--- | :--- | :--- | :--- | :--- |
| $D$ | `pedosphere.transport.moving_domain_diffusivity` | $\mathrm{m^2\,s^{-1}}$ | `pedosphere/transport.yaml` ($10^{-9}$) | Crank (1975) |
| $\mathrm{Co}_{\mathrm{mesh},\max}$ | `pedosphere.transport.moving_domain_mesh_courant_max` | $1$ | `pedosphere/transport.yaml` ($0.9$) | Donea et al. (2004) |
| $\mathrm{Fo}_{\max}$ | `transport.numerical.cfl_diffusion_max` | $1$ | `transport/numerical.yaml` ($0.5$) | Patankar (1980) von Neumann diffusion bound |

`moving_domain_diffusivity` is bound directly to `set_scalar_diffusivity` without float conversion so autograd gradients with respect to $D$ flow through both `forward` and `dissipative_structure`.

## 5. Numerical Methods and Discretization

1. **Input and Positivity Guards**: `forward` requires all four input tensors to be floating-point and finite, `moving_domain_solute_concentration` to be 1-D with $N \ge 2$ and $c_i \ge 0$, `moving_domain_length` to satisfy $L > 0$ and $L + w_b\,\Delta t > 0$, `moving_boundary_solute_concentration` to satisfy $c_b \ge 0$, and `dt` to be finite and positive. Any post-step concentration with $c_i(t + \Delta t) < -\epsilon_{\mathrm{pos}}$ (where $\epsilon_{\mathrm{pos}} = 10\,\epsilon_{\mathrm{mach}}\,\max(\max_i c_i, 1)$) raises `RuntimeError`, while tiny negative rounding noise within $[-\epsilon_{\mathrm{pos}}, 0)$ is clamped to $0$.
2. **Independent Explicit Stability Inequalities**: With $\Delta x = L(t)/N$, `forward` checks the diffusion Fourier number and the mesh Courant number as two separate inequalities before stepping:

$$
\mathrm{Fo} = \frac{D\,\Delta t}{\left(\Delta x\right)^2} \le \mathrm{Fo}_{\max}, \qquad \mathrm{Co}_{\mathrm{mesh}} = \frac{\lvert w_b \rvert\,\Delta t}{\Delta x} \le \mathrm{Co}_{\mathrm{mesh},\max}.
$$

Violating either bound raises `ValueError`. Note that because both diffusion and upwind mesh-sweeping act simultaneously in a single explicit step, the diagonal self-weight of interior cell $i$ on $m_i(t)$ is proportional to $1 - 2\,\mathrm{Fo} - \frac{\Delta t\,\max(w_i, 0)}{\Delta x}$ (for $w_b < 0$, or $1 - 2\,\mathrm{Fo} - \frac{\Delta t\,\lvert w_{i+1} \rvert}{\Delta x}$ under shrinking upwind withdrawal); satisfying $\mathrm{Fo} \le 0.5$ and $\mathrm{Co}_{\mathrm{mesh}} \le 0.9$ independently permits combined diagonal weights below zero when both ratios approach their individual maxima simultaneously, in which case the post-step negativity check fails closed with `RuntimeError` if negative concentrations arise.
3. **IPHS Level 3 Structural Hooks**: `dissipative_structure` returns $J = 0$, symmetric positive semidefinite diffusive matrix $R_h = d_h^{\mathsf{T}} d_h\,(D / \Delta x^2)$ where $d_h \in \mathbb{R}^{(N-1)\times N}$ is the cell-to-face difference matrix, and metric derivative `hodge_time_derivative` $\dot{\star} = \frac{w_b}{N} I_N$.
4. **UpdateItem Modes**: Both prognostic state updates (`moving_domain_solute_concentration` returning $c(t+\Delta t) - c(t)$ in $\mathrm{kg\,m^{-3}}$, and `moving_domain_length` returning $L(t+\Delta t) - L(t)$ in $\mathrm{m}$) use `UpdateMode.ACCUMULATE` with `per_timestep=True`. All four `diagnostics.moving_domain.*` entries use `UpdateMode.REPLACE` with `per_timestep=True`.

## 6. Conservation, Boundary Conditions, and Ledgers

- **Boundary Conditions**: Diffusive fluxes across top face $x = 0$ and bottom face $x = L(t)$ vanish ($F_0 = F_N = 0$). The top face is stationary ($w_0 = 0 \implies s_0 = 0$). Mass enters or leaves the column exclusively via the swept advective flux $s_N = c_{\mathrm{face},N}\,w_b$ at the moving lower boundary ($s_N > 0$ adds entrained mass $c_b\,w_b\,\Delta t$ during expansion; $s_N < 0$ exports bottom-cell mass $c_{N-1}\,w_b\,\Delta t$ during contraction).
- **Exact Discrete GCL and Mass Ledger**: Telescoping interior swept fluxes $\sum_{i=0}^{N-1}(s_{i+1} - s_i) = s_N - s_0 = s_N$ and zero-sum closed diffusive rates $\sum_{i=0}^{N-1}\mathrm{rate}_i\,\Delta x = 0$ yield exact column mass closure:

$$
\sum_{i=0}^{N-1} m_i(t + \Delta t) - \sum_{i=0}^{N-1} m_i(t) = \Delta t\,s_N.
$$

`check_conservation_laws` extracts `diagnostics.moving_domain.mass_balance_residual` and returns $\max \lvert r_{\mathrm{mass}} \rvert$ under `"mass"` ($\mathrm{kg\,m^{-2}}$).

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| :--- | :--- | :--- | :--- | :--- |
| `MovingDomainDiffusionFunctor` | `relic/scientific/pedosphere/transport/moving_domain_diffusion.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

Diffusive face fluxes delegate to `_compute_diffusion_rate_and_boundary_flux_1d` in `relic/computation/operators/functional.py`.

## 8. Implementation Status

- `MovingDomainDiffusionFunctor` 1-D ALE finite-volume column diffusion on a uniformly stretched domain with exact discrete GCL, upwind swept boundary exchange, and moving-metric Hodge power diagnostic: `[I]` (certified IPHS Level 3 structured moving-domain transport; certificate `pedosphere_transport_moving_domain_diffusion_moving_domain_diffusion_functor.yaml`).
- Non-uniform ALE coordinate mappings, adaptive local mesh refinement, and conservative multi-mesh state remapping: `[N]`.

Certificate expressions (`pedosphere_transport_moving_domain_diffusion_moving_domain_diffusion_functor.yaml`):

```text
legacy_expression: m_i(t+dt) = m_i(t) + dt [rate_i dx + s_{i+1} - s_i]; c_i(t+dt) = m_i(t+dt) / V_i(t+dt); V_i(t+dt) = L(t+dt)/N; L(t+dt) = L + w_b dt
iphs_expression: constant field preserved under pure mesh motion; column mass change equals the swept boundary mass dt s_N; GCL residual zero to rounding
```

## 9. References

1. Crank, J. (1975). *The Mathematics of Diffusion* (2nd ed.). Oxford University Press.
2. Thomas, P. D., and Lombard, C. K. (1979). Geometric conservation law and its application to flow computations on moving grids. *AIAA Journal*, 17(10), 1030-1037.
3. Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
4. Donea, J., Huerta, A., Ponthot, J.-Ph., and Rodriguez-Ferran, A. (2004). Arbitrary Lagrangian-Eulerian methods. In E. Stein, R. de Borst, and T. J. R. Hughes (Eds.), *Encyclopedia of Computational Mechanics* (Vol. 1, Chapter 14). Wiley.
