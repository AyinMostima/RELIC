# Atmospheric Surface-Layer Turbulence (MOST)

## 1. Process Definition and Scope

`TurbulenceFunctor` (`relic\scientific\atmosphere\turbulence.py`) computes surface-layer turbulent momentum and sensible-heat exchange together with aerodynamic scalar resistance and Monin-Obukhov stability scales using the shared similarity operators in `relic\scientific\atmosphere\most\`. Theoretical foundations for Monin-Obukhov Similarity Theory (MOST) appear in *Methods of Ecological Analysis*, Chapters 17 and 18.

All eight meteorological and geometric inputs are mandatory without default substitution, and `config["parameters"]` overrides are rejected in favor of `ParameterManager` YAML governance.

## 2. Governing Equations

Moist-air density $\rho$ ($\mathrm{kg\,m^{-3}}$) is evaluated from `air_temperature` $T_{\mathrm{air}}$, `atmospheric_pressure` $p$, and `relative_humidity` $\mathrm{RH} \in [0, 1]$ ($e < p$) via `compute_moist_air_density`:

$$e = \mathrm{RH}\,e_s(T_{\mathrm{air}}), \qquad \rho = \frac{p - (1 - \varepsilon)e}{R_d T_{\mathrm{air}}}.$$

With scalar roughness closure $z_{0h} = r_{h/m} z_{0m}$ and effective height $z - d > z_{0m}$, the roughness-corrected integrated momentum and heat profile factors are

$$f_m = \ln\!\left(\frac{z - d}{z_{0m}}\right) - \psi_m\!\left(\frac{z - d}{L}\right) + \psi_m\!\left(\frac{z_{0m}}{L}\right), \qquad f_h = \ln\!\left(\frac{z - d}{z_{0h}}\right) - \psi_h\!\left(\frac{z - d}{L}\right) + \psi_h\!\left(\frac{z_{0h}}{L}\right),$$

where the Businger-Dyer integrated stability corrections for $\zeta \in [-l_u, l_s]$ are

$$\psi_m(\zeta) = \psi_h(\zeta) = -\beta_s \zeta \quad (\zeta \ge 0),$$

$$\psi_m(\zeta) = 2\ln\!\left(\frac{1 + x}{2}\right) + \ln\!\left(\frac{1 + x^2}{2}\right) - 2\arctan(x) + \frac{\pi}{2}, \qquad \psi_h(\zeta) = 2\ln\!\left(\frac{1 + x^2}{2}\right), \qquad x = (1 - \gamma_u \zeta)^{1/4} \quad (\zeta < 0).$$

Friction velocity $u_*$, temperature scale $\theta_*$, upward surface sensible heat flux $H$, surface shear stress $\tau$, Monin-Obukhov length $L$, aerodynamic scalar resistance $r_{ah}$, and neutral drag coefficient $C_{D,N}$ are

$$u_* = \frac{\kappa U}{f_m}, \qquad \theta_* = \frac{\kappa\left(T_{\mathrm{sfc}} - T_{\mathrm{air}}\right)}{f_h}, \qquad H = \rho c_p u_* \theta_*, \qquad \tau = \rho u_*^2,$$

$$L = -\frac{\rho c_p T_{\mathrm{air}} u_*^3}{\kappa g H}, \qquad r_{ah} = \frac{f_h}{\kappa u_*}, \qquad C_{D,N} = \left[\frac{\kappa}{\ln\!\left((z - d)/z_{0m}\right)}\right]^2.$$

In discrete structural cochain notation, the Level 3 transport audit form is

$$\dot{x}_h = -D_h q_h + G_h u_h, \qquad q_h = -\star_{\tau} D_h^{\mathsf{T}} e_h, \qquad y_h = G_h^{\mathsf{T}} e_h,$$

with static reference momentum mobility $\star_{\tau} = \rho_0 \kappa u_{*,\min} z_{\mathrm{ref}}$ ($\mathrm{kg\,m^{-1}\,s^{-1}}$; $\rho_0$ enters only this state-independent structural Hodge matrix, never runtime $\tau$, $H$, or $L$).

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $U$ | Wind speed at reference height ($U \ge U_{\min}$) | $\mathrm{m\,s^{-1}}$ | `wind_speed` | `forcing` |
| $T_{\mathrm{air}}, T_{\mathrm{sfc}}$ | Air temperature at $z$ and surface skin temperature | $\mathrm{K}$ | `air_temperature`, `surface_temperature` | `forcing` / `state` |
| $z, z_{0m}, d$ | Reference height, momentum roughness length, displacement height | $\mathrm{m}$ | `reference_height`, `roughness_length_momentum`, `displacement_height` | `forcing` |
| $p, \mathrm{RH}$ | Atmospheric pressure and relative humidity in $[0, 1]$ | $\mathrm{Pa}$, $1$ | `atmospheric_pressure`, `relative_humidity` | `forcing` |
| $u_*$ | Converged surface-layer friction velocity | $\mathrm{m\,s^{-1}}$ | `friction_velocity` | `diagnostic` (`REPLACE`, `per_timestep=True`) |
| $r_{ah}$ | Aerodynamic scalar resistance | $\mathrm{s\,m^{-1}}$ | `aerodynamic_resistance` | `diagnostic` (`REPLACE`, `per_timestep=True`) |
| $L$ | Converged Monin-Obukhov length | $\mathrm{m}$ | `monin_obukhov_length` | `diagnostic` (`REPLACE`, `per_timestep=True`) |
| $H$ | Surface upward sensible heat flux | $\mathrm{W\,m^{-2}}$ | `sensible_heat_flux` | `flux` (`REPLACE`, `per_timestep=True`) |
| $\tau$ | Surface momentum shear stress | $\mathrm{kg\,m^{-1}\,s^{-2}}$ | `momentum_flux` | `flux` (`REPLACE`, `per_timestep=True`) |
| $\zeta$ | Stability parameter $(z - d)/L$ | $1$ | `diagnostics.turbulence.stability_parameter` | `diagnostic` (`REPLACE`, `per_timestep=True`) |
| $C_{D,N}$ | Neutral momentum drag coefficient | $1$ | `diagnostics.turbulence.drag_coefficient` | `diagnostic` (`REPLACE`, `per_timestep=True`) |

## 4. Parameter Table

| Symbol | `ParameterManager` / Physical-Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $z_{\mathrm{ref}}$ | `atmosphere.turbulence.default_reference_height` | $\mathrm{m}$ | `atmosphere/turbulence.yaml` | Structural Hodge scale |
| $r_{h/m}$ | `atmosphere.turbulence.roughness_heat_ratio` | $1$ | `atmosphere/turbulence.yaml` (`0.1`) | Garratt (1992) |
| $U_{\min}$ | `atmosphere.turbulence.minimum_wind_speed` | $\mathrm{m\,s^{-1}}$ | `atmosphere/turbulence.yaml` (`0.1`) | Operational wind floor check |
| $u_{*,\min}$ | `atmosphere.turbulence.minimum_friction_velocity` | $\mathrm{m\,s^{-1}}$ | `atmosphere/turbulence.yaml` | Static Hodge mobility scale only |
| $N_{\max}, \epsilon_{\mathrm{tol}}, L_0$ | `atmosphere.turbulence.{max_iterations,convergence_tolerance,neutral_obukhov_length}` | $1$, $1$, $\mathrm{m}$ | `atmosphere/turbulence.yaml` | Fixed-point MOST solver |
| $\beta_s, \gamma_u$ | `atmosphere.most.{beta_stable,gamma_unstable}` | $1$ | `atmosphere/most.yaml` (`5.0`, `16.0`) | Businger et al. (1971); Dyer (1974) |
| $l_s, l_u$ | `atmosphere.most.{stable_correction_limit,unstable_correction_limit}` | $1$ | `atmosphere/most.yaml` (`2.0`, `2.0`) | Operational MOST envelope |
| $H_{\min}, L_{\min}, L_{\max}$ | `atmosphere.most.{h_min_safeguard,monin_obukhov_length_clamp_min,monin_obukhov_length_clamp_max}` | $\mathrm{W\,m^{-2}}$, $\mathrm{m}$, $\mathrm{m}$ | `atmosphere/most.yaml` | Near-neutral diagnostic regularization |
| $\kappa, \rho_0, c_p, g, T_0, e_0, a, b$ | `get_physical_constant(...)` (`von_karman_constant`, `air.density_reference`, `air.specific_heat_cp`, `earth.gravity_acceleration`, `celsius_to_kelvin`, Tetens constants) | SI | `physical/constants.yaml` | Physical-constants SSOT |

## 5. Numerical Methods and Discretization

Because $(u_*, \theta_*, H, L)$ are implicitly coupled through $\psi_m$ and $\psi_h$, `forward()` executes a fixed-point iteration initialized from neutral $u_{*,0}$ and `neutral_obukhov_length` $L_0$. At each iteration $n$, $\zeta_n = (z - d)/L_n$ must stay inside $[-l_u, l_s]$ (out-of-envelope values fail fast), and termination requires joint elementwise convergence on both $u_*$ and $L$:

$$\frac{\lvert u_{*,n+1} - u_{*,n} \rvert}{\max(\lvert u_{*,n+1} \rvert, \lvert u_{*,n} \rvert)} \le \epsilon_{\mathrm{tol}}, \qquad \frac{\lvert L_{n+1} - L_n \rvert}{\max(\lvert L_{n+1} \rvert, \lvert L_n \rvert)} \le \epsilon_{\mathrm{tol}}.$$

Exhausting `max_iterations` raises an exception without returning an unconverged iterate. The separate diagnostic helper in `most` providing a mixing-length diffusivity $K = l\,u_*$ is never used as a fallback inside `TurbulenceFunctor`.

## 6. Conservation, Boundary Conditions, and Ledgers

`TurbulenceFunctor` declares the open boundary law `momentum_boundary` and verifies the constitutive surface-stress identity using the same input moist-air density $\rho_{\mathrm{air}}$:

$$R_{\tau} = \max\!\left(\frac{\lvert \tau - \rho_{\mathrm{air}} u_*^2 \rvert}{\max(\lvert \tau \rvert, \lvert \rho_{\mathrm{air}} u_*^2 \rvert)}\right)$$

(with unit denominator scale only when both terms vanish identically). Sensible heat flux $H$ is exported as an open boundary flux to downstream surface-energy and atmospheric-stability consumers; no closed local atmospheric energy reservoir is integrated here.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `TurbulenceFunctor` | `relic\scientific\atmosphere\turbulence.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

## 8. Implementation Status

- **Runtime status**: `[I]` implemented MOST surface-layer exchange Functor emitting seven `UpdateItem(REPLACE, per_timestep=True)` outputs with full `torch.autograd` tensor connectivity.
- **IPHS certificate**: **Level 3** (`docs\source\theory\iphs_certificates\atmosphere_turbulence_turbulence_functor.yaml`, IPHS-structured transport). No internal kinetic/thermal storage functional or strict port-power equality is claimed.

## 9. References

1. Monin, A. S., & Obukhov, A. M. (1954). Basic laws of turbulent mixing in the surface layer of the atmosphere. *Trudy Geofiz. Inst. AN SSSR*, 24(151), 163-187.
2. Businger, J. A., Wyngaard, J. C., Izumi, Y., & Bradley, E. F. (1971). Flux-profile relationships in the atmospheric surface layer. *Journal of the Atmospheric Sciences*, 28(2), 181-189.
3. Dyer, A. J. (1974). A review of flux-profile relationships. *Boundary-Layer Meteorology*, 7(3), 363-372.
4. Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers.
5. Garratt, J. R. (1992). *The Atmospheric Boundary Layer*. Cambridge University Press.
6. Holton, J. R., & Hakim, G. J. (2012). *An Introduction to Dynamic Meteorology* (5th ed.). Academic Press.
