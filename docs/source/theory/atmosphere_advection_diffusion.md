# Atmospheric Advection-Diffusion Transport

## 1. Process Definition and Scope

`AtmosphericAdvectionDiffusionFunctor` (`relic\scientific\atmosphere\transport\AtmosphericAdvectionDiffusionFunctor.py`) advances one grid-cell atmospheric tracer concentration field under resolved wind advection, molecular diffusion, stability-dependent turbulent eddy diffusion, and an optional volumetric source-sink input. Theoretical foundations for atmospheric scalar transport and finite-volume discretization appear in *Methods of Ecological Analysis*, Chapters 17, 18, and 25.

The process operates in 2D `(batch, y, x)` or 3D `(batch, z, y, x)` tensor storage order while accepting physical cell widths in `(dx, dy[, dz])` order. Stability closure is selected explicitly at construction via `stability_scheme` (`"monin-obukhov"`, `"richardson"`, or `"constant"`). When no `source_term` input is supplied on `atmosphere_tracer_source_port`, the source defaults to $S = 0$ without any background `ParameterManager` source.

## 2. Governing Equations

The continuous scalar advection-diffusion balance for tracer concentration $C$ ($\mathrm{kg\,m^{-3}}$) is

$$\frac{\partial C}{\partial t} + \nabla \cdot (\mathbf{u} C) = \nabla \cdot \left((K_{\mathrm{mol}} + K_{\mathrm{eddy}}) \nabla C\right) + S.$$

Turbulent eddy diffusivity $K_{\mathrm{eddy}}$ ($\mathrm{m^2\,s^{-1}}$) uses one of three stability regimes evaluated at mid-boundary-layer height $z = H_{\mathrm{BL}} / 2$:

1. **Monin-Obukhov scheme (`"monin-obukhov"`)**: moist-air density $\rho$ is evaluated from `temperature` $T$, `atmospheric_pressure` $p$, and `relative_humidity` $\mathrm{RH}$ by `compute_moist_air_density`:

   $$e = \mathrm{RH}\,e_s(T), \qquad \rho = \frac{p - (1 - \varepsilon)e}{R_d T},$$

   $$L = -\frac{\rho c_p T u_*^3}{\kappa g H_s}, \qquad \zeta = \frac{z}{L}, \qquad z = \frac{H_{\mathrm{BL}}}{2}.$$

2. **Bulk Richardson scheme (`"richardson"`)**: using horizontal wind speed magnitude $\lVert \mathbf{U} \rVert$ and configured potential-temperature lapse rate $\partial \theta / \partial z$ (rejecting zero-shear states without denominator regularization):

   $$\mathrm{Ri}_{\mathrm{bulk}} = \frac{g}{T_{\mathrm{ref}}}\frac{\partial \theta / \partial z}{(\lVert \mathbf{U} \rVert / z)^2}.$$

3. **Constant scheme (`"constant"`)**: sets the bounded stability index $\xi = 0$.

For bounded stability index $\xi = \operatorname{clip}(\zeta \text{ or } \mathrm{Ri}_{\mathrm{bulk}}, -\xi_{\max}, \xi_{\max})$ and neutral diffusivity $K_0 = f K_{\max}$, pointwise eddy diffusivity is

$$K_{\mathrm{eddy}} = \min\!\left(K_{\max},\; \begin{cases} K_0 \left(1 + \dfrac{a s}{1 + s}\right), & s = -\xi > 0, \\ \dfrac{K_0}{1 + b \xi}, & \xi \ge 0, \end{cases}\right), \qquad K_{\mathrm{tot}} = K_{\mathrm{mol}} + K_{\mathrm{eddy}}.$$

In discrete structural cochain notation, the Level 3 audit map is

$$\dot{c}_h = -D_h q_h + G_h u_h + s_h, \qquad q_h = v_h c_h - \star_K d_h c_h,$$

where $\star_K$ in static structural assembly uses the nominal upper bound $K_{\mathrm{mol}} + K_{\max}$, whereas `forward()` evaluates the state-dependent $K_{\mathrm{tot}}$ above.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $C$, $\Delta C$ | Tracer concentration input and per-step integrated increment | $\mathrm{kg\,m^{-3}}$ | `tracer_concentration` | `state` (input) / `flux` (`ACCUMULATE`, `per_timestep=True`) |
| $u_x, u_y, u_z$ | Resolved wind velocity components (`wind_w` read only in 3D) | $\mathrm{m\,s^{-1}}$ | `wind_u`, `wind_v`, `wind_w` | `forcing` |
| $T$ | Air temperature (required in `"monin-obukhov"`, optional otherwise) | $\mathrm{K}$ | `temperature` | `forcing` |
| $H_s$ | Upward surface sensible heat flux (required in `"monin-obukhov"`) | $\mathrm{W\,m^{-2}}$ | `sensible_heat_flux` | `forcing` / upstream `flux` |
| $p$ | Atmospheric pressure (required in `"monin-obukhov"`) | $\mathrm{Pa}$ | `atmospheric_pressure` | `forcing` |
| $\mathrm{RH}$ | Relative humidity in $[0, 1]$ (required in `"monin-obukhov"`) | $1$ | `relative_humidity` | `forcing` |
| $S$ | Optional external volumetric tracer source or sink | $\mathrm{kg\,m^{-3}\,s^{-1}}$ | `source_term` | `forcing` |
| $T_{\mathrm{adv}}$ | Advective concentration tendency | $\mathrm{kg\,m^{-3}\,s^{-1}}$ | `diagnostics.advection_rate` | `diagnostic` |
| $T_{\mathrm{diff}}$ | Diffusive concentration tendency | $\mathrm{kg\,m^{-3}\,s^{-1}}$ | `diagnostics.diffusion_rate` | `diagnostic` |
| $C_{\mathrm{adv}}$ | Cell-centered advective Courant sum | $1$ | `diagnostics.atmospheric_advection_diffusion.cfl_number` | `diagnostic` |
| $D_{\mathrm{stab}}$ | Explicit diffusion stability number | $1$ | `diagnostics.diffusion_stability_number` | `diagnostic` |
| $K_{\mathrm{eddy}}$ | Evaluated turbulent eddy diffusivity | $\mathrm{m^2\,s^{-1}}$ | `diagnostics.eddy_diffusivity` | `diagnostic` |
| $\xi$ | Bounded stability index | $1$ | `diagnostics.stability_parameter` | `diagnostic` |
| $j_{\mathrm{out},f}$ | Packed outward boundary face flux cochain | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.boundary_flux_cochain` | `diagnostic` |

## 4. Parameter Table

| Symbol | `ParameterManager` / Physical-Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $h_x, h_y, h_z$ | `atmosphere.grid.dx`, `atmosphere.grid.dy`, `atmosphere.grid.dz` | $\mathrm{m}$ | `atmosphere/grid.yaml` | Stull (1988) |
| $H_{\mathrm{BL}}$ | `atmosphere.boundary_layer.boundary_layer_height` | $\mathrm{m}$ | `atmosphere/boundary_layer.yaml` | Stull (1988) |
| $K_{\max}$ | `atmosphere.turbulence.eddy_diffusivity_max` | $\mathrm{m^2\,s^{-1}}$ | `atmosphere/turbulence.yaml` | Jacobson (2005) |
| $f$ | `atmosphere.turbulence.transport_neutral_diffusivity_fraction` | $1$ | `atmosphere/turbulence.yaml` (`0.1`) | Model empirical closure |
| $a$ | `atmosphere.turbulence.transport_unstable_enhancement_coefficient` | $1$ | `atmosphere/turbulence.yaml` (`2.0`) | Model empirical closure |
| $b$ | `atmosphere.turbulence.transport_stable_suppression_coefficient` | $1$ | `atmosphere/turbulence.yaml` (`10.0`) | Model empirical closure |
| $\xi_{\max}$ | `atmosphere.turbulence.transport_stability_limit` | $1$ | `atmosphere/turbulence.yaml` (`10.0`) | Model empirical closure |
| $u_*$ | `atmosphere.turbulence.minimum_friction_velocity` | $\mathrm{m\,s^{-1}}$ | `atmosphere/turbulence.yaml` | Monin and Obukhov (1954) |
| $\partial \theta / \partial z$ | `atmosphere.stability.potential_temperature_gradient` | $\mathrm{K\,m^{-1}}$ | `atmosphere/stability.yaml` | Richardson (1920) |
| $K_{\mathrm{mol}}$ | `atmosphere.molecular_diffusion.molecular_diffusivity` | $\mathrm{m^2\,s^{-1}}$ | `atmosphere/molecular_diffusion.yaml` | Seinfeld and Pandis (2016) |
| $C_{\mathrm{adv},\max}$ | `transport.numerical.cfl_advection_max` | $1$ | `transport/numerical.yaml` | LeVeque (2007) |
| $D_{\mathrm{stab},\max}$ | `transport.numerical.cfl_diffusion_max` | $1$ | `transport/numerical.yaml` | LeVeque (2007) |
| $\kappa, g, c_p, T_{\mathrm{ref}}$ | `get_physical_constant(...)` (`physical/constants.yaml`) | SI | Physical-constants SSOT | CODATA / Holton and Hakim (2012) |

## 5. Numerical Methods and Discretization

`forward()` delegates spatial discretization and single-step integration to the registered `transport.advection_diffusion` finite-volume operator after reversing physical spacing $(h_x, h_y[, h_z])$ into tensor axis order $(h_z, h_y, h_x)$:

$$\Delta C = \left(T_{\mathrm{adv}} + T_{\mathrm{diff}} + S\right)\Delta t.$$

Before executing the update, the Functor checks individual cell-centered advective and diffusive stability numbers:

$$C_{\mathrm{adv}}(x) = \Delta t \sum_{k=1}^{d} \frac{\lvert u_k(x) \rvert}{h_k} \le C_{\mathrm{adv},\max}, \qquad D_{\mathrm{stab}}(x) = K_{\mathrm{tot}}(x)\Delta t \sum_{k=1}^{d} h_k^{-2} \le D_{\mathrm{stab},\max}.$$

The combined finite-volume operator enforces its joint monotonicity bound and rejects negative post-step concentrations. Exceeding any bound raises immediately without internal step shortening; substepping remains owned by `RELICSimulator`. Both branches of the piecewise eddy-diffusivity formula are evaluated on safe half-lines so `torch.autograd` gradients remain finite across $\xi = 0$.

## 6. Conservation, Boundary Conditions, and Ledgers

Active face descriptors configured via `BoundaryConditionManager` (`set_boundary_condition`) lower onto three homogeneous canonical face laws and survive checkpoint restarts through `get_extra_state` / `set_extra_state`:

| Configured Faces | Canonical Law | Outward Face Flux Law |
| --- | --- | --- |
| All faces zero-gradient Neumann | `neumann` | $j_{\mathrm{diff}} = 0$; advective ghost $c_{\mathrm{ghost}} = c_{\mathrm{adjacent}}$ permits normal-wind inflow/outflow |
| All axes paired periodic | `periodic` | No exterior boundary faces; $c(x + L) = c(x)$ |
| All faces equal static scalar Dirichlet $g$ | `dirichlet` | $j_{\mathrm{diff}} = -K_{\mathrm{tot}}(g - c_{\mathrm{adjacent}})/(h/2)$ outward; advective inflow ghost $g$ |

Mixed, Robin, wall, open, nonzero-Neumann, spatially inhomogeneous Dirichlet, and callable time-varying face descriptors are rejected fail-fast before state mutation. The local `mass` conservation hook verifies the open-system finite-volume ledger:

$$\Delta M - \Delta t \int_{\Omega} S\,dV + \Delta t \sum_{f \in \partial\Omega} j_{\mathrm{out},f} A_f = 0.$$

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `AtmosphericAdvectionDiffusionFunctor` | `relic\scientific\atmosphere\transport\AtmosphericAdvectionDiffusionFunctor.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

## 8. Implementation Status

- **Runtime status**: `[I]` implemented production Functor delegating to `transport.advection_diffusion` and returning `UpdateItem(ACCUMULATE, per_timestep=True)` on `tracer_concentration`.
- **IPHS certificate**: **Level 3** (`docs\source\theory\iphs_certificates\atmosphere_transport_atmospheric_advection_diffusion_functor_atmospheric_advection_diffusion_functor.yaml`, IPHS-structured transport). `energy_terms` and `entropy_terms` provide unweighted quadratic concentration proxies (not thermodynamic energy in joules or physical entropy production), and advection has no runtime-linked skew-symmetric Poisson proof, so Level 4/5 strict runtime status is not claimed.

## 9. References

1. Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers.
2. Monin, A. S., & Obukhov, A. M. (1954). Basic laws of turbulent mixing in the surface layer of the atmosphere. *Trudy Geofiz. Inst. AN SSSR*, 24(151), 163-187.
3. Richardson, L. F. (1920). The supply of energy from and to atmospheric eddies. *Proceedings of the Royal Society of London A*, 97(686), 354-373.
4. Jacobson, M. Z. (2005). *Fundamentals of Atmospheric Modeling* (2nd ed.). Cambridge University Press.
5. Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and Physics* (3rd ed.). Wiley.
6. Holton, J. R., & Hakim, G. J. (2012). *An Introduction to Dynamic Meteorology* (5th ed.). Academic Press.
7. Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop Evapotranspiration* (FAO Irrigation and Drainage Paper 56). FAO.
8. LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.
