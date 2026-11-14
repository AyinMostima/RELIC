# Atmospheric Radiation and Solar Geometry

## 1. Process Definition and Scope

This page defines the atmospheric radiation processes and solar-geometry helpers in `relic\scientific\atmosphere\radiation.py`, `relic\scientific\atmosphere\radiation_energy_conservation.py`, and `relic\scientific\atmosphere\solar_geometry.py`. Theoretical foundations for surface radiative transfer and canopy light attenuation appear in *Methods of Ecological Analysis*, Chapters 17, 18, and 20.

Four components share this domain boundary without overlapping state ownership:

1. `RadiationFunctor`: point-scale (`SPATIAL_DIMS = None`) instantaneous surface shortwave/longwave radiation balance, photosynthetically active photon flux (`ppfd`), and optional compile-time Beer-Lambert multi-layer canopy PAR partitioning controlled solely by the global feature flag `canopy_layers_enabled`.
2. `RadiationField`: spatial radiation field container (`StateManager`-supplied `grid_coordinates` and `radiation_field`) enforcing bounded projection, Shepard inverse-distance spatial query interpolation, and optional `torch.autograd` spatial gradients.
3. `RadiationEnergyConservationValidator`: validation-phase diagnostic Functor auditing shortwave, longwave, and canopy radiative flux balances without advancing prognostic state.
4. `solar_geometry.py`: pure stateless tensor functions (`cos_solar_zenith`) implementing FAO-56 solar declination, seasonal time correction, hour angle, and horizon-clipped solar zenith cosine.

## 2. Governing Equations

### 2.1 Surface Shortwave and Longwave Balance (`RadiationFunctor`)

Clear-sky and cloud-attenuated downwelling shortwave irradiance $SW_{\mathrm{down}}$ ($\mathrm{W\,m^{-2}}$) and net shortwave irradiance $SW_{\mathrm{net}}$ are

$$SW_{\mathrm{down,clear}} = SW_0\,\tau_{\mathrm{atm}}\,\max(0, \cos\theta), \qquad SW_{\mathrm{down}} = SW_{\mathrm{down,clear}}\left(1 - f_{\mathrm{cloud}} C_{\mathrm{cloud}}\right), \qquad SW_{\mathrm{net}} = (1 - \alpha) SW_{\mathrm{down}}.$$

Internal actual vapor pressure $e_{a,\mathrm{Pa}}$ (computed via the shared Tetens psychrometric operator from $T_c = T_{\mathrm{air}} - T_0$ and $\mathrm{RH}$, converted to $\mathrm{hPa}$ without reading or writing `vapor_pressure_deficit`) drives Brutsaert (1975) clear-sky atmospheric emissivity $\varepsilon_{\mathrm{atm}}$ and downwelling longwave irradiance $LW_{\mathrm{down}}$:

$$e_{\mathrm{sat,Pa}} = e_0\exp\!\left(\frac{a\,T_c}{T_c + b}\right), \qquad e_{a,\mathrm{Pa}} = \mathrm{RH}\,e_{\mathrm{sat,Pa}}, \qquad \varepsilon_{\mathrm{atm}} = C_1\left(\frac{e_{a,\mathrm{hPa}}}{T_{\mathrm{air}}}\right)^{C_2}, \qquad LW_{\mathrm{down}} = \varepsilon_{\mathrm{atm}}\sigma T_{\mathrm{air}}^4.$$

Under an opaque, reciprocal, diffuse-grey surface law with broadband emissivity/absorptivity $\varepsilon_{\mathrm{sfc}} \in [0, 1]$, surface-emitted, reflected, total outgoing, net longwave, and total net radiation are

$$LW_{\mathrm{emitted}} = \varepsilon_{\mathrm{sfc}}\sigma T_{\mathrm{sfc}}^4, \qquad LW_{\mathrm{reflected}} = (1 - \varepsilon_{\mathrm{sfc}})LW_{\mathrm{down}}, \qquad LW_{\mathrm{up}} = LW_{\mathrm{emitted}} + LW_{\mathrm{reflected}},$$

$$LW_{\mathrm{net}} = LW_{\mathrm{down}} - LW_{\mathrm{up}} = \varepsilon_{\mathrm{sfc}} LW_{\mathrm{down}} - \varepsilon_{\mathrm{sfc}}\sigma T_{\mathrm{sfc}}^4, \qquad R_{\mathrm{net}} = SW_{\mathrm{net}} + LW_{\mathrm{net}}.$$

### 2.2 PAR, Erbs Direct-Diffuse Partitioning, and Beer-Lambert Canopy Transfer

Total photosynthetically active irradiance $\mathrm{PAR}_{\mathrm{total}}$ ($\mathrm{W\,m^{-2}}$) and canopy-top photon flux density $\mathrm{PPFD}$ ($\mathrm{\mu mol\,photons\,m^{-2}\,s^{-1}}$) are evaluated unconditionally:

$$\mathrm{PAR}_{\mathrm{total}} = f_{\mathrm{PAR}} SW_{\mathrm{down}}, \qquad \mathrm{PPFD} = \mathrm{PAR}_{\mathrm{total}} c_{\mathrm{PPFD}}.$$

Hourly clearness index $k_T = SW_{\mathrm{down}} / \left(SW_0 \max(0, \cos\theta)\right)$ ($k_T = 0$ at night) partitions diffuse fraction $f_{\mathrm{diffuse}}$ via the ParameterManager-backed Erbs et al. (1982) correlation:

$$f_{\mathrm{diffuse}} = \begin{cases} 1 - 0.09\,k_T, & k_T \le 0.22, \\ 0.9511 - 0.1604\,k_T + 4.388\,k_T^2 - 16.638\,k_T^3 + 12.336\,k_T^4, & 0.22 < k_T \le 0.80, \\ 0.165, & k_T > 0.80, \end{cases}$$

$$\mathrm{PAR}_{\mathrm{dir}} = (1 - f_{\mathrm{diffuse}})\mathrm{PAR}_{\mathrm{total}}, \qquad \mathrm{PAR}_{\mathrm{dif}} = f_{\mathrm{diffuse}}\mathrm{PAR}_{\mathrm{total}}.$$

When `canopy_layers_enabled` is true (`model_type="beer_lambert"` is the sole executable canopy transfer law), total `leaf_area_index` $\mathrm{LAI}_{\mathrm{total}}$ is distributed across $N$ layers by either uniform weights $L_i = \mathrm{LAI}_{\mathrm{total}} / N$ or midpoint-quadrature Beta softmax weights $x_i = (i + 0.5)/N$, $\log w_i = (a_\beta - 1)\log x_i + (b_\beta - 1)\log(1 - x_i)$, $L_i = \mathrm{LAI}_{\mathrm{total}}\operatorname{softmax}(\log w)_i$. With cumulative leaf area $L_{\mathrm{cum}}(i) = \sum_{j=1}^{i} L_j$ ($L_{\mathrm{cum}}(0) = 0$), layer absorbed PAR and canopy absorbed fraction are

$$\mathrm{APAR}_i = \mathrm{PAR}_{\mathrm{dir}}\left[e^{-k_b L_{\mathrm{cum}}(i-1)} - e^{-k_b L_{\mathrm{cum}}(i)}\right] + \mathrm{PAR}_{\mathrm{dif}}\left[e^{-k_d L_{\mathrm{cum}}(i-1)} - e^{-k_d L_{\mathrm{cum}}(i)}\right], \qquad \mathrm{fPAR} = \frac{\sum_{i=1}^{N} \mathrm{APAR}_i}{\mathrm{PAR}_{\mathrm{total}}}.$$

### 2.3 Spatial Field Projection (`RadiationField`) and FAO-56 Solar Geometry

`RadiationField` projects input state $R_{\mathrm{state}}$ onto $[R_{\min}, R_{\max}]$, records $\epsilon_{\mathrm{clip}} = R_{\mathrm{clipped}} - R_{\mathrm{unclipped}}$, evaluates Shepard (1968) inverse-distance query weights $w_i = d_i^{-p} / \sum_j d_j^{-p}$ for $d_i = \lVert x_q - x_i \rVert$ above `coincident_tolerance`, and declares the Level 3 structural form $\dot{R}_h = M_h e_R + G_h u_R$ with $M_h = \operatorname{diag}(R_{\max} - R_{\min}) \succeq 0$.

In `solar_geometry.py`, for day-of-year $J$, UTC hour $t$, east longitude $\lambda \in [-180^\circ, 180^\circ)$ ($L_z = 0$, $L_m = -\lambda$), and latitude $\varphi$ ($\mathrm{rad}$), FAO-56 Chapter 3 gives

$$\delta = 0.409\sin\!\left(\frac{2\pi}{365}J - 1.39\right), \qquad b = \frac{2\pi(J - 81)}{364}, \qquad S_c = 0.1645\sin(2b) - 0.1255\cos b - 0.025\sin b,$$

$$\omega = \frac{\pi}{12}\left[\left(t + 0.06667(L_z - L_m) + S_c\right) - 12\right], \qquad \cos\theta_z = \max\!\left(\sin\varphi\sin\delta + \cos\varphi\cos\delta\cos\omega,\; 0\right).$$

Three daily-geometry functions serve the daylight Gauss temporal support (`relic\scientific\atmosphere\daylight_quadrature.py`, documented in `photosynthesis.md`, Section 2.5). `cosine_solar_zenith_from_hour_angle` evaluates the same clipped $\cos\theta_z$ (FAO-56 Eq. 28 integrand, $\delta$ of Eq. 24) for a given hour angle $\omega$; `cosine_solar_zenith_fao56` now calls it and is bitwise unchanged. `sunset_hour_angle_fao56` and `daylight_hours_fao56` give the sunset hour angle (FAO-56 Eq. 25) and the day length (Eq. 34):

$$c = -\tan\varphi\tan\delta, \qquad \omega_s = \begin{cases} \arccos c, & |c| < 1 \\ 0, & c \ge 1 \ \text{(polar night)} \\ \pi, & c \le -1 \ \text{(polar day)} \end{cases}, \qquad N = \frac{24\,\omega_s}{\pi}\ \mathrm{h} \in [0, 24].$$

FAO-56 does not treat $|c| \ge 1$; the two limits are the geometric limits of Eq. 25 (no refraction or solar-disc correction), and $\arccos$ is evaluated only on $|c| < 1$, so no infinite derivative is formed and the result is constant outside. FAO-56 Example 9 (3 September, $J = 246$, $20^\circ$ S: $\omega_s = 1.527\ \mathrm{rad}$, $N = 11.7\ \mathrm{h}$) is reproduced ($1.5269\ \mathrm{rad}$, $11.66\ \mathrm{h}$).

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $T_{\mathrm{air}}, T_{\mathrm{sfc}}, \mathrm{RH}$ | Air temperature, surface temperature, relative humidity | $\mathrm{K}$, $\mathrm{K}$, $1$ | `air_temperature`, `surface_temperature`, `relative_humidity` | `forcing` / `state` |
| $SW_0, \cos\theta, f_{\mathrm{cloud}}$ | Extraterrestrial shortwave potential, zenith cosine, cloud fraction | $\mathrm{W\,m^{-2}}$, $1$, $1$ | `incoming_shortwave_potential`, `cos_zenith`, `cloud_fraction` | `forcing` |
| $\alpha, \varepsilon_{\mathrm{sfc}}$ | Surface broadband shortwave albedo and longwave emissivity | $1$ | `surface_albedo`, `surface_emissivity` | `state` / `forcing` |
| $\mathrm{LAI}_{\mathrm{total}}$ | Canopy leaf area index (required when `canopy_layers_enabled=True`) | $\mathrm{m^2\,m^{-2}}$ | `leaf_area_index` | `state` |
| $SW_{\mathrm{down}}, LW_{\mathrm{down}}, R_{\mathrm{net}}$ | Downwelling shortwave/longwave and net surface radiative flux | $\mathrm{W\,m^{-2}}$ | `shortwave_down`, `longwave_down`, `net_radiation` | `flux` (`REPLACE`, `per_timestep=False`) |
| $SW_{\mathrm{net}}, LW_{\mathrm{net}}$ | Net shortwave and net longwave radiative flux | $\mathrm{W\,m^{-2}}$ | `shortwave_net`, `longwave_net` | `flux` (`REPLACE`, `per_timestep=False`) |
| $\varepsilon_{\mathrm{atm}}, \mathrm{PPFD}$ | Atmospheric emissivity and canopy-top photosynthetic photon flux | $1$, $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$ | `atmospheric_emissivity`, `ppfd` | `flux` / observation (`REPLACE`, `per_timestep=False`) |
| $\mathrm{fPAR}, \mathrm{APAR}_i$ | Conditional canopy absorbed PAR fraction and layer absorbed PAR | $1$, $\mathrm{W\,m^{-2}}$ | `fpar`, `apar_layer` | `flux` (`REPLACE`, `per_timestep=False`) |
| $\text{diag}_{\mathrm{rad}}$ | Unconditional PAR, PPFD, longwave (`longwave_up`, `longwave_emitted`, `longwave_reflected`), vapor-pressure, and clipping diagnostics | $\mathrm{W\,m^{-2}}$, $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$, $\mathrm{Pa}$, $1$ | `diagnostics.radiation.*` | `diagnostic` |
| $x_g, R_{\mathrm{field}}$ | Spatial coordinates and bounded radiation field (`RadiationField`) | $\mathrm{m}$, $\mathrm{W\,m^{-2}}$ | `grid_coordinates`, `radiation_field` | `state` (`REPLACE`, `per_timestep=True`) |
| $\nabla R, \epsilon_{\mathrm{clip}}$ | Conditional spatial gradient and projection residuals (`RadiationField`) | $\mathrm{W\,m^{-3}}$, $\mathrm{W\,m^{-2}}$ | `diagnostics.radiation_field.{gradient,clipping_residual,query_clipping_residual}` | `diagnostic` |
| $r_{\mathrm{energy}}$ | Relative radiative energy balance audit error (`RadiationEnergyConservationValidator`) | $1$ | `diagnostics.energy_conservation_report` | `diagnostic` |

## 4. Parameter Table

| Symbol | `ParameterManager` / Physical-Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $\tau_{\mathrm{atm}}, C_{\mathrm{cloud}}$ | `atmosphere.radiation.{atmospheric_transmittance,cloud_attenuation_factor}` | $1$ | `atmosphere/radiation.yaml` | Campbell and Norman (1998) |
| $C_1, C_2$ | `atmosphere.radiation.{brutsaert_coefficient,brutsaert_exponent}` | $\mathrm{K^{1/7}\,hPa^{-1/7}}$, $1$ | `atmosphere/radiation.yaml` (`1.24`, `1/7`) | Brutsaert (1975); Peng et al. (2025) |
| $f_{\mathrm{PAR}}, c_{\mathrm{PPFD}}$ | `atmosphere.radiation.canopy.{par_fraction_of_sw,ppfd_conversion}` | $1$, $\mathrm{\mu mol\,J^{-1}}$ | `atmosphere/radiation.yaml` (`0.45`, `4.6`) | Jacovides et al. (2003); McCree (1972) |
| $\text{Erbs}$ | `atmosphere.radiation.canopy.{erbs_low_threshold,erbs_high_threshold,erbs_low_coefficients,erbs_mid_coefficients,erbs_high_fraction}` | $1$ | `atmosphere/radiation.yaml` | Erbs et al. (1982) |
| $N, k_b, k_d, a_\beta, b_\beta$ | `atmosphere.radiation.canopy.{model_type,layers,k_b,k_d,lai_profile,beta_params}` | $1$ | `atmosphere/radiation.yaml` | Monsi and Saeki (1953); Song et al. (2021) |
| $R_{\min}, R_{\max}, p, \epsilon_d$ | `atmosphere.radiation.field_bounds.generic_{min,max}_radiation`, `atmosphere.radiation.field_interpolation.{idw_power,coincident_tolerance}` | $\mathrm{W\,m^{-2}}$, $1$, $\mathrm{m}$ | `atmosphere/radiation.yaml` | Shepard (1968) |
| $\epsilon_{\mathrm{audit}}$ | `validation.conservation.radiation.*` | SI / $1$ | `validation/` YAML | Campbell and Norman (1998); Bonan (2019) |
| $\sigma, e_0, a, b, T_0$ | `get_physical_constant(...)` (`stefan_boltzmann_constant`, Tetens constants, `celsius_to_kelvin`) | SI | `physical/constants.yaml` | Physical-constants SSOT |

## 5. Numerical Methods and Discretization

`RadiationFunctor` evaluates instantaneous radiative rates (`UpdateItem(REPLACE, per_timestep=False)`), independent of time step `dt`. Canopy topology is locked at initialization from `canopy_layers_enabled`; unsupported canopy selectors (such as legacy `two_stream` proxies) or domains violating $0 \le SW_{\mathrm{down}} \le SW_0 \max(0, \cos\theta)$ raise immediately without fallback formulas. Shortwave and emissivity projection residuals are always emitted via `diagnostics.radiation.shortwave_clipping_residual` and `diagnostics.radiation.emissivity_clipping_residual`.

`RadiationField.forward` requires `dt > 0` and finite `grid_coordinates` and `radiation_field` on every step, returning `UpdateItem(REPLACE, unit="W m-2", per_timestep=True)` without caching hidden previous-step state.

## 6. Conservation, Boundary Conditions, and Ledgers

Surface and canopy radiation obey open-boundary instantaneous flux identities:

$$R_{\mathrm{net}} - SW_{\mathrm{net}} - LW_{\mathrm{net}} = 0, \qquad SW_{\mathrm{down}} = SW_{\mathrm{net}} + \alpha\,SW_{\mathrm{down}}, \qquad LW_{\mathrm{net}} = LW_{\mathrm{down}} - LW_{\mathrm{up}}, \qquad \sum_{i=1}^{N} \mathrm{APAR}_i + \mathrm{PAR}_{\mathrm{transmitted}} = \mathrm{PAR}_{\mathrm{total}}.$$

`RadiationEnergyConservationValidator` audits explicit `longwave_up` as total outgoing longwave (emitted plus reflected) and rejects `surface_emission` as an alias for total `longwave_up`; when `longwave_up` is omitted in direct point audits, its property-inference path checks the opaque reciprocal greybody identity and preserves partial-ledger scope when observations are incomplete.

## 7. Code Mapping

| Functor Class / Module | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `RadiationFunctor` | `relic\scientific\atmosphere\radiation.py` | `PhysicalField` | `PhysicalFieldFunctor` | `ExecutionPhase.MAIN_PROCESSES` |
| `RadiationField` | `relic\scientific\atmosphere\radiation.py` | `PhysicalField` | `PhysicalFieldFunctor` | `ExecutionPhase.MAIN_PROCESSES` |
| `RadiationEnergyConservationValidator` | `relic\scientific\atmosphere\radiation_energy_conservation.py` | `PhysicalField` | `PhysicalFieldFunctor` | `ExecutionPhase.VALIDATION` |
| `cos_solar_zenith` (pure function) | `relic\scientific\atmosphere\solar_geometry.py` | Stateless utility (non-Functor) | N/A | Callable on demand |
| `cosine_solar_zenith_from_hour_angle`, `sunset_hour_angle_fao56`, `daylight_hours_fao56` (pure functions) | `relic\scientific\atmosphere\solar_geometry.py` | Stateless utility (non-Functor) | N/A | Called by `daylight_quadrature.py` |

## 8. Implementation Status

- **Runtime status**: `[I]` implemented across `RadiationFunctor`, `RadiationField`, `RadiationEnergyConservationValidator`, and `solar_geometry.py`.
- **IPHS certificates**:
  - `RadiationFunctor`: **Level 2** (`docs\source\theory\iphs_certificates\atmosphere_radiation_radiation_functor.yaml`, IPHS-wrapped open-boundary flux process; no stored-energy Hamiltonian or conjugate effort-flow power port is claimed).
  - `RadiationField`: **Level 3** (`docs\source\theory\iphs_certificates\atmosphere_radiation_radiation_field.yaml`, IPHS-structured spatial field container).
  - `RadiationEnergyConservationValidator`: **Level 1** (`docs\source\theory\iphs_certificates\atmosphere_radiation_energy_conservation_radiation_energy_conservation_validator.yaml`, read-only conservation diagnostic).
- **Equivalence certificate ASCII anchors (`RadiationFunctor`)**:

```text
R_net = SW_net + LW_net; SW_net = (1-alpha) * SW_down; LW_net = epsilon*LW_down - epsilon*sigma*T_sfc^4; SW_down = SW_pot * tau * max(0,cos(theta)) * f_cloud; LW_down = epsilon_atm * sigma * T_air^4; epsilon_atm = C1 * (e_a/T_air)^(C2)
R_net = (1-alpha) SW_down + epsilon_sfc LW_down - epsilon_sfc sigma T_sfc^4; PPFD = SW_down f_PAR c_PPFD; retained values are open boundary fluxes and observations
```

## 9. References

1. Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop Evapotranspiration* (FAO Irrigation and Drainage Paper 56). FAO.
2. Brutsaert, W. (1975). On a derivable formula for long-wave radiation from clear skies. *Water Resources Research*, 11(5), 742-744.
3. Campbell, G. S., & Norman, J. M. (1998). *An Introduction to Environmental Biophysics* (2nd ed.). Springer.
4. Monteith, J. L., & Unsworth, M. H. (2013). *Principles of Environmental Physics* (4th ed.). Academic Press.
5. McCree, K. J. (1972). Test of current definitions of photosynthetically active radiation against leaf photosynthesis data. *Agricultural Meteorology*, 10, 443-453.
6. Jacovides, C. P., Tymvios, F. S., Asimakopoulos, D. N., Theofilou, K. M., & Pashiardes, S. (2003). Global photosynthetically active radiation and its relationship with global solar radiation. *Agricultural and Forest Meteorology*, 119, 29-37.
7. Erbs, D. G., Klein, S. A., & Duffie, J. A. (1982). Estimation of the diffuse radiation fraction for hourly, daily and monthly-average global radiation. *Solar Energy*, 28, 293-302.
8. Monsi, M., & Saeki, T. (1953). Uber den Lichtfaktor in den Pflanzengesellschaften und seine Bedeutung fur die Stoffproduktion. *Japanese Journal of Botany*, 14, 22-52.
9. Song, J., Miller, G. R., Cahill, A. T., Aparecido, L. M. T., & Moore, G. W. (2021). Improving canopy radiation transfer in land-surface models with vertically resolved leaf-area distributions. *Journal of Advances in Modeling Earth Systems*, 13, e2020MS002259.
10. Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. *Proceedings of the 1968 23rd ACM National Conference*, 517-524.
11. Nicodemus, F. E. (1965). Directional reflectance and emissivity of an opaque surface. *Applied Optics*, 4(7), 767-775.
12. Peng, J., et al. (2025). Evaluation of clear-sky downward longwave radiation parameterizations. *Atmospheric Measurement Techniques*, 18, 2877-2905.
