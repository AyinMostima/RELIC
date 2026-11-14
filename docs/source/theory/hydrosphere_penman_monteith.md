# Penman-Monteith Evapotranspiration and Canopy Interception

## 1. Process Definition and Scope

`PenmanMonteithET` (`relic/scientific/hydrosphere/evapotranspiration/penman_monteith.py`, using `relic/scientific/hydrosphere/utils/psychrometrics.py`) computes combined energy-balance and aerodynamic water-vapor mass transport (`evapotranspiration_flux`, $\mathrm{kg\,m^{-2}\,s^{-1}}$) and latent heat flux (`latent_heat_flux`, $\mathrm{W\,m^{-2}}$) following FAO-56 (Allen et al., 1998) and Monteith (1965), with optional CLM4.5-style canopy rainfall interception storage (`enable_canopy_interception`).

`PenmanMonteithET` declares `CONFLICT_RESOLUTION = ConflictResolutionPolicy.EXCLUSIVE` (mutually exclusive with `PriestleyTaylorET` on the shared `evapotranspiration_flux` and `latent_heat_flux` ports) and lagged dependencies on `bucket_soil_moisture` and `richards_soil_moisture`. Monin-Obukhov atmospheric stability iteration inside `PenmanMonteithET` and multilayer prognostic leaf energy balance remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Section 19.3.1).

Configuration. `canopy_resistance_source` (required) is `state` (canopy resistance from `surface_resistance` or `canopy_conductance`) or `fao56_reference_surface` (the constant FAO-56 reference $r_s$). `temporal_support` selects the time support of the law: `hourly` (default when absent, the FAO-56 hourly form; bitwise the previous behaviour), `daily` (Section 2.5) or `daylight_gauss` (Section 2.6). The aerodynamic conductance of Section 2.2 is the module-level owner `fao56_aerodynamic_conductance`, shared with `LeafEnergyBalance` (`biosphere_leaf_energy_balance.md`), so both read one aerodynamic resistance.

## 2. Governing Equations

### 2.1 Psychrometric Properties and Virtual-Temperature Moist Air Density

For Kelvin air temperature $T > 0$ ($T_C = T - 273.15$) and atmospheric pressure $P > 0$ ($\mathrm{Pa}$), saturation vapor pressure $e_s$ ($\mathrm{Pa}$), slope $\Delta$ ($\mathrm{Pa\,K^{-1}}$), latent heat $\lambda$ ($\mathrm{J\,kg^{-1}}$), psychrometric constant $\gamma$ ($\mathrm{Pa\,K^{-1}}$), and virtual-temperature moist air density $\rho_a$ ($\mathrm{kg\,m^{-3}}$, Holton and Hakim, 2012) are:

$$
e_s(T_C) = 610.8\exp\!\left(\frac{17.27\,T_C}{T_C + 237.3}\right), \qquad
\Delta = \frac{4098.0\,e_s(T_C)}{(T_C + 237.3)^2}, \qquad
\gamma = \frac{c_p P}{\varepsilon\,\lambda}
$$

$$
\rho_a = \frac{P - (1 - \varepsilon)e_a}{R_d T}, \qquad
R_d = \frac{R}{M_d}, \qquad
\varepsilon = \frac{M_w}{M_d}
$$

where actual vapor pressure $e_a \in [0, P)$ is obtained from `relative_humidity` ($e_a = \mathrm{RH}\,e_s$) or `vapor_pressure_deficit` ($e_a = e_s - \mathrm{VPD}$).

### 2.2 Available Energy and Conductances

Parameter `soil_heat_flux_source` selects `"fraction"` ($G = f_{G,\mathrm{day}} R_n$ for $R_n \ge 0$ and $G = f_{G,\mathrm{night}} R_n$ for $R_n < 0$) or `"state"` (reading required surface `soil_heat_flux` $G$ directly), giving available energy $A = R_n - G$. On the default `hourly` support both modes enforce $\Delta t \le \Delta t_{G,\max}$ (`soil_heat_flux_hourly_max_step`); the fractions are those of FAO-56 Eqs. 45-46 ($f_{G,\mathrm{day}} = 0.1$, $f_{G,\mathrm{night}} = 0.5$ by default). The `daily` and `daylight_gauss` supports replace this step domain by $\Delta t = 86400\,\mathrm{s}$ (Sections 2.5 and 2.6).

When `aerodynamic_resistance` $r_a$ is omitted, logarithmic neutral conductance $g_a = 1 / r_a$ ($\mathrm{m\,s^{-1}}$) uses wind speed clamped to $u_{\mathrm{safe}} = \max(u, u_{\min})$ (`minimum_wind_speed`), momentum roughness $z_{0m}$, heat roughness $z_{0h} = r_{h/m} z_{0m}$ (`heat_to_momentum_roughness_ratio`), displacement height $d$, and von Karman constant $\kappa$:

$$
g_a = \operatorname{clip}\!\left(\frac{\kappa^2 u_{\mathrm{safe}}}{\ln\!\bigl((z_{\mathrm{ref}} - d)/z_{0m}\bigr)\,\ln\!\bigl((z_{\mathrm{ref}} - d)/z_{0h}\bigr)},\; g_{a,\min},\; g_{a,\max}\right)
$$

Canopy conductance $g_c = \max(g_{\min},\, 1/r_s)$ comes from `canopy_conductance`, `surface_resistance`, or LAI-scaled `reference_surface_resistance`.

### 2.3 Penman-Monteith Flux and Nonnegative Projection

Raw vapor mass flux $\mathrm{ET}_{\mathrm{raw}}$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$), nonnegative projected flux $\mathrm{ET}$, and latent heat flux $L_E$ ($\mathrm{W\,m^{-2}}$) are:

$$
\mathrm{ET}_{\mathrm{raw}} = \frac{\Delta(R_n - G) + \rho_a c_p\,\mathrm{VPD}\,g_a}{\lambda\bigl[\Delta + \gamma(1 + g_a / g_c)\bigr]}, \qquad
\mathrm{ET} = \max(\mathrm{ET}_{\mathrm{raw}},\; q_{\min,0}), \qquad
L_E = \lambda\,\mathrm{ET}
$$

Negative condensation candidates ($\mathrm{ET}_{\mathrm{raw}} < 0$) are recorded in `diagnostics.evapotranspiration["negative_et_projection_residual"]`.

### 2.4 Optional Canopy Interception Partitioning

When `enable_canopy_interception` is active with `precipitation_flux` $P_{\mathrm{prec}}$, `canopy_storage` $C^n$ ($\mathrm{mm}$, equivalent to $\mathrm{kg\,m^{-2}}$), and `leaf_area_index` $\mathrm{LAI}$, maximum canopy capacity $C_{\max} = \operatorname{clip}(c_{\mathrm{LAI}}\,\mathrm{LAI},\, C_{\mathrm{bound},\min},\, C_{\mathrm{bound},\max})$ bounds intercepted water and wet-canopy evaporation $E_i$ ($\mathrm{mm}$ over $\Delta t$):

$$
C^* = \min(C^n + P_{\mathrm{prec}}\Delta t,\; C_{\max}), \qquad
E_i = \min\bigl(C^*,\; \eta_{\mathrm{can}}\,\mathrm{ET}\,\Delta t\bigr), \qquad
C^{n+1} = C^* - E_i
$$

Throughfall/drip $T_d = \max(C^n + P_{\mathrm{prec}}\Delta t - C_{\max}, 0) / \Delta t$ and residual transpiration $E_t = \max(\mathrm{ET} - E_i/\Delta t, 0)$ are reported in `diagnostics.evapotranspiration` while primary output `evapotranspiration_flux` remains $\mathrm{ET}$.

### 2.5 Daily Support (`temporal_support: daily`)

The FAO-56 24-hour form (Allen et al., 1998, Chapter 3), qualified for $\Delta t$ = `relic.physical.constants.time.seconds_per_day` only ($\Delta t \ne 86400\,\mathrm{s}$ raises); every input is the 24-hour value of its quantity (daily mean rates). Two terms change:

$$
e_s = \frac{e^0(T_{\max}) + e^0(T_{\min})}{2} \quad \text{(FAO-56 Eq. 12)}, \qquad
G = 0 \quad \text{(FAO-56 Eq. 42)}
$$

with $e^0$ the Tetens form of Eq. 11 (Section 2.1) and $T_{\max}$, $T_{\min}$ the daily extremes `air_temperature_max` and `air_temperature_min` (declared and required only on this support; each inside `temperature_bounds`, $T_{\min} \le T_{\max}$). FAO-56 states that $e^0$ at the daily mean temperature instead of this mean underestimates $e_s$ (the convexity of Eq. 11), hence the vapour pressure deficit and ET. Eq. 42: the day-scale soil heat flux beneath a grass reference surface is relatively small and may be ignored; it sets $G = 0$ for `soil_heat_flux_source: fraction`, whereas `state` reads the supplied daily $G$. $\Delta$, $\gamma$ and $\rho_a$ keep the daily-mean $T$ and $P$, and $e_a$ follows from `relative_humidity` or `vapor_pressure_deficit` with this $e_s$. A daily canopy conductance supplied as the 24-hour mean of the Medlyn samples (Medlyn `daylight_gauss`, daytime $g_c$ times $N/24$ plus the night conductance times $1 - N/24$) enters as $r_s = 1/\overline{g_c}$; this daily PM is not the mean of an instantaneous PM (declared approximation, quantified below).

### 2.6 Daylight Gauss Support (`temporal_support: daylight_gauss`)

The combination equation of Section 2.3 is evaluated at the $n$ Gauss-Legendre instants of the daylight period and at the night sample of the shared owner `relic/scientific/atmosphere/daylight_quadrature.py` (the samples and weights of Farquhar, Medlyn, `LeafEnergyBalance` and MEGAN; construction in `photosynthesis.md`, Section 2.5; Goudriaan, 1986) and averaged with the sample weights $W_i = (N/24)\,w_i$, $W_n = 1 - N/24$. Daily step only ($\Delta t$ = `seconds_per_day`). At sample $k$:

$$
R_{n,k} = R_n - S_n + S_{n,k}, \qquad \sum_k W_k S_{n,k} = S_n \ \Rightarrow\ \sum_k W_k R_{n,k} = R_n
$$

$$
e_{s,k} = e^0(T_k), \qquad D_k = \max\!\left(e^0(T_k) - e_a,\; 0\right), \qquad
G_k = \begin{cases} f_{G,\mathrm{day}}\,R_{n,k}, & R_{n,k} > 0 \\ f_{G,\mathrm{night}}\,R_{n,k}, & R_{n,k} \le 0 \end{cases} \ \text{(FAO-56 Eqs. 45-46)} \ \text{or} \ G_k = G_{\mathrm{state}}
$$

$$
\mathrm{ET}_k = \max\!\left(\frac{\Delta_k(R_{n,k} - G_k) + \rho_{a,k} c_p D_k\, g_a}{\lambda\bigl[\Delta_k + \gamma(1 + g_a/g_{c,k})\bigr]},\; q_{\min,0}\right), \qquad
\mathrm{ET} = \sum_k W_k\,\mathrm{ET}_k
$$

$S_n$ is the daily net shortwave `shortwave_net` of the radiation owner redistributed by the solar elevation into $S_{n,k}$; the daily net longwave $R_n - S_n$ is held constant over the day (the PM has no `longwave_down` or emissivity input and its $R_n$ is the surface net radiation of the radiation owner; this is the same rule as the non-radiative `LeafEnergyBalance` daylight form, not its isothermal radiative form). $T_k$ is the SWAT+ Eq. 1:1.3.1 course of $(T, T_{\max}, T_{\min})$, $e_a$ the daily `actual_vapor_pressure` (replacing `relative_humidity` and `vapor_pressure_deficit`), $g_{c,k}$ the per-sample canopy conductance `canopy_conductance_daylight_samples` of a `daylight_gauss` Medlyn upscaling (with `canopy_resistance_source: state`; it replaces `surface_resistance` and `canopy_conductance`), or the constant FAO-56 reference $r_s$ for `fao56_reference_surface`. Wind, pressure and an optional $r_a$ are constant over the day. Every returned quantity of the law (ET, reference and equilibrium ET, $G$, $A$, $D$, conductances, $\Delta$, $\gamma$) is the weighted mean $\sum_k W_k X_k$, with the lower clamp of ET applied at every sample.

FAO-56 Eqs. 45 and 46 give $G_{hr} = 0.1 R_n$ during daylight and $0.5 R_n$ during nighttime periods for hourly (or shorter) calculations beneath a dense grass cover; the samples are instants. Declared approximations: the dense-grass condition is not met by forests, but the fractions are applied to every surface as on the `hourly` support, so the hourly reference and the daylight method share the $G$ closure; the classifier is $R_{n,k} > 0$ (the classifier of the hourly support; FAO-56 states daylight versus nighttime periods, which differ only where a daylight sample has $R_{n,k} \le 0$); the daily weighted $G$ is not forced to the Eq. 42 value 0; the net longwave is constant over the day (the surface temperature course is not seen); at a sample with $D_k$ floored at 0 the law's $e_{a,k} = e^0(T_k)$ enters the air density only.

Configuration: `{"canopy_resistance_source": "state", "temporal_support": "daylight_gauss", "daylight_gauss_nodes": 3 | 5}` and the optional `daylight_shortwave_shape`. Inputs: `net_radiation`, `shortwave_net`, `air_temperature`, `air_temperature_max`, `air_temperature_min`, `actual_vapor_pressure`, `latitude`, `day_of_year`, `wind_speed`, `atmospheric_pressure`, `canopy_conductance_daylight_samples`, and the optional `soil_heat_flux`, `aerodynamic_resistance` and interception inputs. The outputs are unchanged (24-hour means); the interception step of Section 2.4 acts on the 24-hour means. Daylight keys on another support raise. The `hourly` default and the `daily` support are bitwise the previous implementation (all outputs, the diagnostics dictionary, metadata and optional inputs).

Measured (site validation V1, run `20261108_daylight_support_v1_03` in `BVOCsdata/data/paper_results/paper03_isoprene_mechanism/daylight_support_site_validation`; 15 sites, 27 site-years; reference: the hourly chain with the `hourly` PM; frozen threshold $|\Delta\mathrm{ET}_{\mathrm{annual}}| \le 5\,\%$): median annual ET deviation $+18.7\,\%$ for the all-daily chain (`daily` PM), $-15.5\,\%$ for the `daily` PM fed with the 24-hour mean of the $n = 5$ Medlyn samples, and $+2.2\,\%$ ($n = 3$) and $+3.0\,\%$ ($n = 5$) for the `daylight_gauss` PM, each of which passed at 13 of 15 sites; monthly deviations remain larger.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $R_n$ | Surface net radiation (required) | $\mathrm{W\,m^{-2}}$ | `net_radiation` | `forcing` |
| $G$ | Surface soil heat flux (required when `soil_heat_flux_source="state"`) | $\mathrm{W\,m^{-2}}$ | `soil_heat_flux` | `forcing` |
| $T, u, P$ | Air temperature, wind speed, and atmospheric pressure (required) | $\mathrm{K}$, $\mathrm{m\,s^{-1}}$, $\mathrm{Pa}$ | `air_temperature`, `wind_speed`, `atmospheric_pressure` | `forcing` |
| $\mathrm{RH}, \mathrm{VPD}$ | Relative humidity or vapor pressure deficit (at least one required) | $\mathrm{1}$, $\mathrm{Pa}$ | `relative_humidity`, `vapor_pressure_deficit` | `forcing` |
| $r_s, g_c, r_a$ | Optional surface resistance, canopy conductance, and aerodynamic resistance | $\mathrm{s\,m^{-1}}$, $\mathrm{m\,s^{-1}}$, $\mathrm{s\,m^{-1}}$ | `surface_resistance`, `canopy_conductance`, `aerodynamic_resistance` | `state` |
| $P_{\mathrm{prec}}, C, \mathrm{LAI}$ | Optional interception inputs (precipitation, canopy storage, leaf area index) | $\mathrm{kg\,m^{-2}\,s^{-1}}$, $\mathrm{mm}$, $\mathrm{m^2\,m^{-2}}$ | `precipitation_flux`, `canopy_storage`, `leaf_area_index` | `forcing` / `state` |
| $T_{\max}, T_{\min}$ | Daily maximum and minimum air temperature (`daily`: FAO-56 Eq. 12; `daylight_gauss`: diurnal course) | $\mathrm{K}$ | `air_temperature_max`, `air_temperature_min` | `forcing` |
| $S_n$ | Daily net shortwave of the radiation owner (`daylight_gauss`) | $\mathrm{W\,m^{-2}}$ | `shortwave_net` | `flux` |
| $e_a$ | Daily actual vapour pressure (`daylight_gauss`, replaces RH and VPD) | $\mathrm{Pa}$ | `actual_vapor_pressure` | `forcing` |
| $\varphi$, $J$ | Latitude and day of year of the daylight samples (`daylight_gauss`) | $\mathrm{degrees\_north}$, $1$ | `latitude`, `day_of_year` | `forcing` |
| $g_{c,k}$ | Per-sample Medlyn canopy conductance (`daylight_gauss` with `canopy_resistance_source="state"`) | $\mathrm{m\,s^{-1}}$ | `canopy_conductance_daylight_samples` | `diagnostic` |
| $\mathrm{ET}, L_E$ | Realized evapotranspiration mass flux and latent heat flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$, $\mathrm{W\,m^{-2}}$ | `evapotranspiration_flux`, `latent_heat_flux` | `flux` |
| $\mathrm{VPD}_d, g_a, g_c$ | Resolved VPD, aerodynamic conductance, and canopy conductance diagnostics | $\mathrm{Pa}$, $\mathrm{m\,s^{-1}}$, $\mathrm{m\,s^{-1}}$ | `diagnostics.vapor_pressure_deficit`, `diagnostics.aerodynamic_conductance`, `diagnostics.canopy_conductance` | `diagnostic` |
| $\mathcal{D}_{\mathrm{ET}}$ | Nested psychrometric, energy-port, and interception ledger dictionary | mixed SI | `diagnostics.evapotranspiration` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $s_G, \Delta t_{G,\max}$ | `hydrosphere.evapotranspiration.soil_heat_flux_source`, `hydrosphere.physical.soil_heat_flux_hourly_max_step` | `str`, $\mathrm{s}$ | `hydrosphere/evapotranspiration.yaml`, `physical.yaml` | Allen et al. (1998) |
| $f_{G,\mathrm{day}}, f_{G,\mathrm{night}}$ | `hydrosphere.physical.soil_heat_flux_ratio_daytime`, `hydrosphere.physical.soil_heat_flux_ratio_nighttime` | $\mathrm{1}$ | `hydrosphere/physical.yaml` | Allen et al. (1998) |
| $z_{0m}, d, z_{\mathrm{ref}}, r_{h/m}$ | `hydrosphere.evapotranspiration.roughness_length`, `displacement_height`, `reference_height`, `hydrosphere.physical.heat_to_momentum_roughness_ratio` | $\mathrm{m}$, $\mathrm{m}$, $\mathrm{m}$, $\mathrm{1}$ | `hydrosphere/evapotranspiration.yaml`, `physical.yaml` | Allen et al. (1998); Brutsaert (1982) |
| $g_{\min}, g_{a,\mathrm{ref}}, r_{s,\mathrm{ref}}$ | `hydrosphere.evapotranspiration.minimum_conductance`, `hodge_reference_aerodynamic_conductance`, `hydrosphere.physical.reference_surface_resistance` | $\mathrm{m\,s^{-1}}$, $\mathrm{m\,s^{-1}}$, $\mathrm{s\,m^{-1}}$ | `hydrosphere/evapotranspiration.yaml`, `physical.yaml` | Monteith (1965); Allen et al. (1998) |
| $f_{\mathrm{LAI,max}}, f_{\mathrm{LAI,mod}}$ | `hydrosphere.physical.maximum_lai_factor`, `hydrosphere.physical.moderate_lai_factor` | $\mathrm{1}$ | `hydrosphere/physical.yaml` | Allen et al. (1998) |
| $b_{\mathrm{int}}, c_{\mathrm{LAI}}, \eta_{\mathrm{can}}$ | `hydrosphere.evapotranspiration.enable_canopy_interception`, `interception_capacity_per_LAI`, `canopy_evaporation_efficiency` | `bool`, $\mathrm{mm}$, $\mathrm{1}$ | `hydrosphere/evapotranspiration.yaml` | Oleson et al. (2013) |
| $P_{\min}, P_{\max}, T_{\mathrm{bounds}}, R_{n,\mathrm{bounds}}$ | `hydrosphere.physical.atmospheric_pressure_min`, `atmospheric_pressure_max`, `hydrosphere.numerical.temperature_bounds`, `net_radiation_bounds` | $\mathrm{Pa}$, $\mathrm{K}$, $\mathrm{W\,m^{-2}}$ | `hydrosphere/physical.yaml`, `numerical.yaml` | RELIC domain validation |
| $u_{\mathrm{off}}, u_{\min}, g_{a,\mathrm{bounds}}, q_{\min,\mathrm{ET}}, q_{\min,0}$ | `hydrosphere.numerical.minimum_wind_speed_offset`, `minimum_wind_speed`, `aerodynamic_conductance_bounds`, `evapotranspiration_flux_clamp_min`, `validation.numerical.clamp_min_zero` | $\mathrm{m\,s^{-1}}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `hydrosphere/numerical.yaml`, `validation/numerical.yaml` | RELIC numerical bounds |
| $\Delta t_{\mathrm{day}}$ | `relic.physical.constants.time.seconds_per_day` (`daily`, `daylight_gauss`) | $\mathrm{s}$ | `physical/constants.yaml` | SI |
| $C_{\mathrm{bounds}}, \varepsilon_{\mathrm{mm}}, \varepsilon_{\mathrm{crit}}, q_{d,\min}$ | `hydrosphere.numerical.canopy_storage_bounds`, `mass_conservation_tolerance_mm`, `critical_mass_balance_error_threshold`, `minimum_drainage_flux` | $\mathrm{mm}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `hydrosphere/numerical.yaml` | RELIC interception ledger |

## 5. Numerical Methods and Discretization

Internal tensor conversion normalizes floating inputs to `torch.float32` on the Functor device. Input validation permits tiny negative floating-point wind noise down to `-minimum_wind_speed_offset` before clamping aerodynamic wind speed to `minimum_wind_speed`. Flux and diagnostic ports are emitted as `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`, and `canopy_storage` is emitted as an updated `REPLACE` stock when interception is enabled.

The rate law is one function `_penman_monteith_law(state, dt, support)` with `support` in `hourly`, `daily` and `daylight_sample`; `forward` calls it directly or through `_daylight_gauss_law`, which builds the samples in `float32` (trailing axis of length $n + 1$), applies every input check of the law to every sample, and returns the weighted means before the interception step.

## 6. Conservation, Boundary Conditions, and Ledgers

`PenmanMonteithET` couples an open water-vapor transport port ($q_v = \mathrm{ET}$) to an open latent-heat energy port ($L_E = \lambda q_v$). When canopy interception is active, canopy storage closure verifies $\lvert(C^{n+1} - C^n) - (P_{\mathrm{prec}}\Delta t - T_d\Delta t - E_i)\rvert \le \varepsilon_{\mathrm{mm}}$; when interception is disabled, the Functor owns no water stock and routes negative-ET projection residuals explicitly through `diagnostics.evapotranspiration`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `PenmanMonteithET` | `relic/scientific/hydrosphere/evapotranspiration/penman_monteith.py` | `TransportMechanism` (`TRANSPORT_TYPE="gas_exchange"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `PenmanMonteithET`: `[I]`, IPHS **Level 3** (`transport`, certificate `hydrosphere_evapotranspiration_penman_monteith_penman_monteith_et.yaml`). Dissipative vapor-transport operator $q_v = \star_{\mathrm{ET}} X_v$ paired with latent-heat port $L_E = \lambda q_v$ and explicit projection residuals.

Frozen certificate expressions:

```text
ET = [Delta(Rn-G) + rho_a cp VPD ga] / [lambda(Delta + gamma(1 + ga/gc))], then ET is projected to nonnegative flux and optional canopy interception partitions ET.
q_v = star_ET X_v plus energy port L_E = lambda q_v; projection residuals are routed as diagnostics without hidden sinks.
```

## 9. References

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements* (FAO Irrigation and Drainage Paper 56). FAO.
- Brutsaert, W. (1982). *Evaporation into the Atmosphere: Theory, History, and Applications*. Springer.
- Goudriaan, J. (1986). A simple and fast numerical method for the computation of daily totals of crop photosynthesis. *Agricultural and Forest Meteorology*, 38, 249-254 (cited for the daylight Gauss scheme; text not read).
- Holton, J. R., and Hakim, G. J. (2012). *An Introduction to Dynamic Meteorology* (5th ed.). Academic Press.
- Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205-234.
- Oleson, K. W., Lawrence, D. M., Bonan, G. B., et al. (2013). *Technical Description of Version 4.5 of the Community Land Model (CLM)* (NCAR/TN-503+STR). NCAR.
- SWAT+ Theoretical Documentation, Section 1:1.3.2 Hourly Air Temperature, Eq. 1:1.3.1 (Campbell, 1985, cited there), read 2026-11-06 at swatplus.gitbook.io.
