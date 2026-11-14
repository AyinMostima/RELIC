# Atmospheric Boundary-Layer Eddy Diffusion

## 1. Process Definition and Scope

`EddyDiffusionBLMFunctor` (`relic\scientific\atmosphere\transport\eddy_diffusion_blm.py`) models vertical turbulent exchange between a reduced single-box planetary boundary layer and prescribed free-tropospheric background mixing ratios for the ordered species tuple `('no', 'no2', 'o3', 'isoprene')`. Theoretical background on K-theory gradient closure and Monin-Obukhov similarity appears in *Methods of Ecological Analysis*, Chapters 17 and 18.

By default its upstream dependency is `atmospheric_turbulence`, which supplies `friction_velocity` and `monin_obukhov_length`. The GraphIR selection field `surface_layer_source = "external_forcing"` removes that dependency: $u_*$ and $L$ then arrive through `State:ExternalForcing`, for example from an observed tower series, and the closure below is unchanged. A missing value is refused before the first step and no default is substituted. A graph with this selection must not also contain an `atmospheric_turbulence` node, because that node would replace the forcing. When `atmosphere.transport.enabled` is false, `forward()` returns only `diagnostics.eddy_diffusion.enabled = 0`.

## 2. Governing Equations

At mid-boundary-layer effective mixing height $z_{\mathrm{eff}} = h / 2$, the raw stability ratio $\zeta_{\mathrm{raw}}$ is bounded by the intersection of the process cap $l_p$ and the shared Monin-Obukhov operational limits $(l_u, l_s)$:

$$\zeta_{\mathrm{raw}} = \frac{h / 2}{L}, \qquad l_s^* = \min(l_p, l_s), \qquad l_u^* = \min(l_p, l_u),$$

$$\zeta_{\mathrm{eff}} = \operatorname{clip}(\zeta_{\mathrm{raw}}, -l_u^*, l_s^*), \qquad r_{\zeta} = \zeta_{\mathrm{raw}} - \zeta_{\mathrm{eff}}.$$

Using the shared Businger-Dyer scalar similarity function $\phi_h(\zeta_{\mathrm{eff}})$ ($\phi_h = 1 + \beta_h \zeta_{\mathrm{eff}}$ for $\zeta_{\mathrm{eff}} \ge 0$; $\phi_h = (1 - \gamma_h \zeta_{\mathrm{eff}})^{-1/2}$ for $\zeta_{\mathrm{eff}} < 0$), the bounded eddy diffusivity $K_z$ ($\mathrm{m^2\,s^{-1}}$) is

$$K_{\mathrm{raw}} = \frac{\kappa u_* (h / 2)}{\phi_h(\zeta_{\mathrm{eff}})}, \qquad K_z = \operatorname{clip}(K_{\mathrm{raw}}, K_{\min}, K_{\max}).$$

`forward()` evaluates the reduced single-box top-entrainment exchange closure against the configured top-boundary background mixing ratio $C_{\mathrm{bg},s}$ for each species $s \in \{\mathrm{no}, \mathrm{no2}, \mathrm{o3}, \mathrm{isoprene}\}$ (no multi-layer vertical partial differential equation is solved here):

$$\frac{dC_s}{dt} = \frac{K_z}{h^2}\left(C_{\mathrm{bg},s} - C_s\right), \qquad \Delta C_s = \Delta t\,\frac{K_z}{h^2}\left(C_{\mathrm{bg},s} - C_s\right).$$

In discrete structural cochain notation, the Level 3 transport audit form is

$$\dot{x}_h = -D_h q_h + G_h u_h, \qquad q_h = -\star_K D_h^{\mathsf{T}} e_h, \qquad y_h = G_h^{\mathsf{T}} e_h,$$

where $x_h$ is the 4-species mixing-ratio 0-cochain ($\mathrm{mol\,mol^{-1}}$), $D_h^{\mathsf{T}} e_h$ is the vertical gradient 1-cochain ($\mathrm{mol\,mol^{-1}\,m^{-1}}$), $q_h$ is the vertical turbulent flux dual 1-cochain ($\mathrm{mol\,mol^{-1}\,m\,s^{-1}}$), and $\star_K$ is the positive diagonal Hodge mobility anchored at $K_{\min}$.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $C_{\mathrm{NO}}, \Delta C_{\mathrm{NO}}$ | Boundary-layer NO mixing ratio and per-step increment | $\mathrm{mol\,mol^{-1}}$ | `no_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $C_{\mathrm{NO_2}}, \Delta C_{\mathrm{NO_2}}$ | Boundary-layer $\mathrm{NO_2}$ mixing ratio and per-step increment | $\mathrm{mol\,mol^{-1}}$ | `no2_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $C_{\mathrm{O_3}}, \Delta C_{\mathrm{O_3}}$ | Boundary-layer $\mathrm{O_3}$ mixing ratio and per-step increment | $\mathrm{mol\,mol^{-1}}$ | `o3_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $C_{\mathrm{iso}}, \Delta C_{\mathrm{iso}}$ | Boundary-layer isoprene mixing ratio and per-step increment | $\mathrm{mol\,mol^{-1}}$ | `isoprene_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $h$ | Planetary boundary-layer height | $\mathrm{m}$ | `pbl_height` | `forcing` |
| $u_*$ | Surface-layer friction velocity | $\mathrm{m\,s^{-1}}$ | `friction_velocity` | `diagnostic` / upstream input |
| $L$ | Monin-Obukhov length ($\lvert L \rvert > \epsilon_{\mathrm{mach}}$) | $\mathrm{m}$ | `monin_obukhov_length` | `diagnostic` / upstream input |
| $I_{\mathrm{on}}$ | Runtime activation indicator (`0` or `1`) | $1$ | `diagnostics.eddy_diffusion.enabled` | `diagnostic` |
| $K_z$ | Clamped reduced-box eddy diffusivity | $\mathrm{m^2\,s^{-1}}$ | `diagnostics.eddy_diffusion.eddy_diffusivity` | `diagnostic` |
| $\sigma_{\mathrm{diff}}$ | Explicit diffusion stability number | $1$ | `diagnostics.eddy_diffusion.stability_number` | `diagnostic` |
| $\zeta_{\mathrm{eff}}$ | Effective bounded stability parameter | $1$ | `diagnostics.eddy_diffusion.stability_parameter_z_over_L` | `diagnostic` |
| $I_{\mathrm{clamp}}$ | Indicator (`0` or `1`) that $K_{\mathrm{raw}}$ hit $[K_{\min}, K_{\max}]$ | $1$ | `diagnostics.eddy_diffusion.eddy_diffusivity_clamped` | `diagnostic` |
| $r_{\zeta}$ | Stability clipping residual $\zeta_{\mathrm{raw}} - \zeta_{\mathrm{eff}}$ | $1$ | `diagnostics.eddy_diffusion.stability_parameter_clipping_residual` | `diagnostic` |
| $h, u_*, L$ | Echoed turbulence diagnostics | $\mathrm{m}$, $\mathrm{m\,s^{-1}}$, $\mathrm{m}$ | `diagnostics.eddy_diffusion.{pbl_height,friction_velocity,monin_obukhov_length}` | `diagnostic` |

## 4. Parameter Table

| Symbol | `ParameterManager` / Physical-Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $I_{\mathrm{on}}$ | `atmosphere.transport.enabled` | $1$ | `atmosphere/transport.yaml` | Runtime activation switch |
| $\mathcal{S}$ | `atmosphere.transport.target_species` | $1$ | `atmosphere/transport.yaml` (`[no, no2, o3, isoprene]`) | Canonical species schema |
| $K_{\min}, K_{\max}$ | `atmosphere.transport.k_min`, `atmosphere.transport.k_max` | $\mathrm{m^2\,s^{-1}}$ | `atmosphere/transport.yaml` | Stull (1988); Garratt (1992) |
| $l_p$ | `atmosphere.transport.z_over_L_limit` | $1$ | `atmosphere/transport.yaml` | Foken (2006) |
| $u_{*,\min}, u_{*,\max}$ | `atmosphere.transport.ustar_min`, `atmosphere.transport.ustar_max` | $\mathrm{m\,s^{-1}}$ | `atmosphere/transport.yaml` | Stull (1988) |
| $h_{\min}, h_{\max}$ | `atmosphere.transport.pbl_height_min`, `atmosphere.transport.pbl_height_max` | $\mathrm{m}$ | `atmosphere/transport.yaml` | Stull (1988) |
| $C_{\mathrm{bg},s}$ | `atmosphere.transport.background_{no,no2,o3,isoprene}_mixing_ratio` | $\mathrm{mol\,mol^{-1}}$ | `atmosphere/transport.yaml` | Top-boundary background state |
| $\beta_h, \gamma_h$ | `atmosphere.most.beta_stable`, `atmosphere.most.gamma_unstable` | $1$ | `atmosphere/most.yaml` | Businger et al. (1971); Dyer (1974) |
| $l_s, l_u$ | `atmosphere.most.stable_correction_limit`, `atmosphere.most.unstable_correction_limit` | $1$ | `atmosphere/most.yaml` | Operational MOST envelope |
| $c_{\mathrm{CFL}}, \sigma_{\max}$ | `transport.numerical.cfl_safety_factor`, `transport.numerical.max_diffusion_stability` | $1$ | `transport/numerical.yaml` | Explicit stability contract |
| $\kappa$ | `get_von_karman()` (`relic.physical.constants.turbulence.von_karman_constant`) | $1$ | `physical/constants.yaml` | Physical-constants SSOT |

## 5. Numerical Methods and Discretization

Before computing species increments, `forward()` evaluates the canonical diffusion stability number using characteristic length $h_{\mathrm{char}} = \min(h)$ via `compute_diffusion_stability`:

$$\sigma_{\mathrm{diff}} = \frac{K_z \Delta t}{h_{\mathrm{char}}^2} \le \sigma_{\max} c_{\mathrm{CFL}}.$$

Violating this bound or supplying $u_* \notin [u_{*,\min}, u_{*,\max}]$, $h \notin [h_{\min}, h_{\max}]$, or $\lvert L \rvert \le \epsilon_{\mathrm{mach}}$ raises `ValueError` immediately with the maximum admissible $\Delta t$. The inherited constant-diffusivity solver path `_resolve_diffusivity_tensor` is explicitly disabled with `RuntimeError` so all calls pass through the state-dependent MOST closure.

## 6. Conservation, Boundary Conditions, and Ledgers

Each species obeys an open top-boundary entrainment ledger between the boundary-layer box and the fixed free-tropospheric Dirichlet reservoir $C_{\mathrm{bg},s}$:

$$R_s = \Delta C_s - \Delta t\,\frac{K_z}{h^2}\left(C_{\mathrm{bg},s} - C_s\right) = 0.$$

Both stability-parameter saturation ($r_{\zeta}$) and diffusivity clamping ($I_{\mathrm{clamp}}$) are disclosed through read-only diagnostics.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `EddyDiffusionBLMFunctor` | `relic\scientific\atmosphere\transport\eddy_diffusion_blm.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

## 8. Implementation Status

- **Runtime status**: `[I]` implemented reduced-box boundary-layer eddy-diffusion Functor returning `UpdateItem(ACCUMULATE, per_timestep=True, unit="mol mol-1")`.
- **IPHS certificate**: **Level 3** (`docs\source\theory\iphs_certificates\atmosphere_transport_eddy_diffusion_blm_eddy_diffusion_blm_functor.yaml`, IPHS-structured transport). No thermodynamic storage functional or strict port-power balance proof is claimed.

## 9. References

1. Monin, A. S., & Obukhov, A. M. (1954). Basic laws of turbulent mixing in the atmosphere near the ground. *Trudy Geofiz. Inst. AN SSSR*, 24(151), 163-187.
2. Businger, J. A., Wyngaard, J. C., Izumi, Y., & Bradley, E. F. (1971). Flux-profile relationships in the atmospheric surface layer. *Journal of the Atmospheric Sciences*, 28(2), 181-189.
3. Dyer, A. J. (1974). A review of flux-profile relationships. *Boundary-Layer Meteorology*, 7(3), 363-372.
4. Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers.
5. Garratt, J. R. (1992). *The Atmospheric Boundary Layer*. Cambridge University Press.
6. Foken, T. (2006). 50 years of the Monin-Obukhov similarity theory. *Boundary-Layer Meteorology*, 119(3), 431-447.
