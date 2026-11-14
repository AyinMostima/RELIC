# Big-Leaf Canopy Energy Balance and Leaf Temperature

## 1. Process Definition and Scope

`LeafEnergyBalance` (`relic/scientific/biosphere/photosynthesis/leaf_energy_balance.py`) is a read-only diagnostic that returns the leaf temperature `leaf_temperature` ($\mathrm{K}$) of the representative big leaf, the canopy treated as one leaf per unit ground area. It solves the steady energy balance $R_n - G = H + \lambda E$ in the closed form obtained by linearising the saturation vapour pressure about the air temperature, the same linearisation that gives the Penman-Monteith combination equation (Monteith, 1965; Allen et al., 1998, Eq. 3). Its latent heat is therefore the Penman-Monteith latent heat for the same available energy, vapour pressure deficit, aerodynamic conductance and canopy conductance, so the leaf temperature and the evapotranspiration of a daily graph share one energy partition.

With the canopy conductance of `MedlynStomatalConductance` (`canopy_conductance_upscaling: effective_leaf_area_index`) the chain soil water $\rightarrow$ water stress $\rightarrow$ Farquhar assimilation $\rightarrow$ $g_s$ $\rightarrow$ $g_c$ $\rightarrow$ latent heat $\rightarrow$ leaf temperature is closed. Consumers read the leaf temperature through explicit options with unchanged defaults: `MEGANModel` config `temperature_source: leaf_temperature` (the MEGAN2.1 $\gamma_T$ and its $T_{24}$, $T_{240}$ history then use the leaf temperature and the states `leaf_temperature_24h_mean`, `leaf_temperature_240h_mean`), and `ThermalStress` `air_temperature_source: leaf_temperature` or `leaf_temperature_max`.

Configuration. `temporal_support` has no default and is one of `daily`, `instantaneous` and `daylight_gauss` (Section 2.1). `energy_balance_form` keeps `linearised_big_leaf` when absent (the form of Section 2, the certificate expression); `radiative_canopy_share` (Section 2.2) adds the radiative coupling, the canopy share of the available energy and a Monin-Obukhov stability correction of $g_a$, and only with it `free_convection: beljaars` (Section 2.3) and `stability_iterations` are accepted.

Not modelled in the default form: a radiative resistance (the supplied $R_n$ is the net radiation of the surface, not the isothermal net radiation) and stability corrections of $g_a$; both are available through `energy_balance_form: radiative_canopy_share`. Not modelled in any form: a canopy heat store, a separate soil surface (two-source, Shuttleworth-Wallace) energy balance, sunlit/shaded leaf classes, and soil sensible heat in the stability of the profile `[N]`. The daily support has no diurnal cycle within a step; the `daylight_gauss` support resolves it at quadrature samples.

## 2. Governing Equations

Available energy, psychrometric terms and density (Allen et al., 1998, Eqs. 8, 11-13 and 42; shared helpers of `relic/scientific/hydrosphere/utils/psychrometrics.py`):

$$
A = R_n - G, \qquad G = 0 \ \text{(daily support, FAO-56 Eq. 42)}, \qquad
s = \left.\frac{d e_s}{dT}\right|_{T_a}, \qquad \gamma = \frac{c_p\, p}{\epsilon\, \lambda}
$$

$$
e_a = \max(e_s - D,\, 0), \quad e_s = \tfrac12\left[e^0(T_{\max}) + e^0(T_{\min})\right] \text{ (daily) or } e^0(T_a), \qquad
\rho = \frac{p - (1-\epsilon)\, e_a}{R_d T_a}
$$

Aerodynamic conductance (FAO-56 Eq. 4, neutral profile; one owner `fao56_aerodynamic_conductance` shared with `PenmanMonteithET`):

$$
g_a = \frac{\kappa^2 \max(u, u_{\min})}{\ln\frac{z-d}{z_{0m}}\,\ln\frac{z-d}{z_{0h}}}, \qquad g_a \in [g_{a,\min}, g_{a,\max}]
$$

Leaf-air temperature difference. Resistance form (Jones, 2014, Chapter 9) with the heat resistance equal to the aerodynamic vapour resistance, $r_{HR} = r_{aW} = r_a = 1/g_a$, and $r_c = 1/g_c$:

$$
T_l - T_a = \frac{r_a (r_a + r_c)\, \gamma A/(\rho c_p) - r_a D}{\gamma (r_a + r_c) + s\, r_a}
$$

Implemented conductance form (numerator and denominator multiplied by $g_a g_c$; no division by $g_c$):

$$
\Delta T = \frac{\gamma (g_a + g_c) A/(\rho c_p) - g_a g_c D}{g_a\left[s\, g_c + \gamma (g_a + g_c)\right]}, \qquad
H = \rho c_p g_a \Delta T, \qquad \lambda E = A - H = \frac{(s A + \rho c_p D g_a)\, g_c}{s\, g_c + \gamma (g_a + g_c)}
$$

Limits: $g_c \to 0$ gives the dry leaf $\Delta T = A/(\rho c_p g_a)$ ($H = A$); $g_c \to \infty$ gives the wet surface $\Delta T = [\gamma A/(\rho c_p g_a) - D]/(s + \gamma)$, whose $A = 0$ value $-D/(s+\gamma)$ is the linearised wet-bulb depression. No canopy ($\mathrm{LAI} \le 0$): $T_l = T_a$, $\Delta T = H = \lambda E = 0$.

Daily maximum (daily support only, declared estimate, no literature law):

$$
T_{l,\max} = T_{\max} + \Delta T
$$

It is exact only if $\Delta T$ were constant over the day. Where daytime available energy exceeds its 24-hour mean, the leaf-air difference at the time of $T_{\max}$ exceeds the daily-mean $\Delta T$, so the estimate is biased low there.

### 2.1 Daylight Gauss temporal support (`temporal_support: daylight_gauss`)

Daily step only ($dt = 86400\,\mathrm{s}$). The instantaneous balance is evaluated at the samples $k$ of `relic.scientific.atmosphere.daylight_quadrature` (Gauss-Legendre nodes over the daylight period of the FAO-56 solar geometry from `latitude` and `day_of_year`, plus the night) and averaged with the sample weights $W_k$ (Goudriaan, 1986; Goudriaan and van Laar, 1994):

$$
R_{n,k} = R_n - S_n + S_{n,k}, \quad \sum_k W_k S_{n,k} = S_n, \qquad D_k = \max\left(e^0(T_k) - e_a,\, 0\right), \qquad G_k = 0, \qquad X = \sum_k W_k X_k
$$

with $T_k$ the SWAT+ Eq. 1:1.3.1 diurnal course of the daily mean, maximum and minimum air temperature, $e_a$ the input `actual_vapor_pressure` (replacing `vapor_pressure_deficit`), $S_n$ the daily net shortwave `shortwave_net` redistributed by the solar elevation (the net longwave is held constant over the day), and $g_{c,k}$ the per-sample Medlyn canopy conductance `canopy_conductance_daylight_samples` (replacing `canopy_conductance`). Wind, pressure and LAI are constant over the day. The outputs are the 24-hour means, `leaf_temperature_daylight_samples` ($T_{l,k}$), and $T_{l,\max} = T_{\max} + \Delta T_m$ with $m$ the daylight sample of the highest air temperature ($T_{l,\max} = T_{\max}$ where $\mathrm{LAI} \le 0$). Declared approximations: $G_k = 0$ at every instant (FAO-56 Eq. 42 holds for the daily sum only, so daytime available energy is overestimated by the daytime soil heat flux), a constant net longwave, and $T_{l,\max}$ exact only if the air maximum fell on the selected sample.

### 2.2 Radiative, canopy-share and stability-corrected form (`energy_balance_form: radiative_canopy_share`)

The radiation inputs `shortwave_net`, `longwave_down` and `surface_emissivity` of the radiation owner replace `net_radiation`. With the isothermal net radiation (the net radiation the surface would receive at air temperature) and the first-order Taylor term of $\varepsilon\sigma T_l^4$ about $T_a$ as the radiative conductance:

$$
R_{ni} = S_n + \varepsilon L_\downarrow - \varepsilon \sigma T_a^4, \qquad g_R = \frac{4 \varepsilon \sigma T_a^3}{\rho c_p}, \qquad A_c = (R_{ni} - G)\left(1 - e^{-k\,\mathrm{LAI}}\right) = (R_{ni} - G)\, k L_{\mathrm{eff}}
$$

$$
\Delta T = \frac{\gamma (g_a + g_c) A_c/(\rho c_p) - g_a g_c D}{\gamma (g_a + g_R)(g_a + g_c) + s\, g_a g_c}, \qquad
H = \rho c_p g_a \Delta T, \qquad L = \rho c_p g_R \Delta T, \qquad \lambda E = A_c - H - L
$$

Since RadiationFunctor writes $L_{net} = \varepsilon L_\downarrow - \varepsilon\sigma T_s^4$, $R_{ni} = R_n + \varepsilon\sigma(T_s^4 - T_a^4)$. The canopy share uses the Beer-Lambert owner `beer_lambert_effective_leaf_area` and `biosphere.photosynthesis.canopy_light_extinction` shared with Farquhar and Medlyn (Monsi and Saeki, 1953; de Pury and Farquhar, 1997). The wind is floored by the balance-local `biosphere.leaf_energy_balance.minimum_wind_speed` (default 0.1 m s$^{-1}$, equal to the hydrosphere floor; Penman-Monteith keeps `hydrosphere.numerical.minimum_wind_speed`). The surface emissivity of the radiation owner is used for the leaf (no leaf-emissivity parameter exists in the database). No chapter or equation number of Jones (2014) or Campbell and Norman (1998) is quoted for this form: it is derived from the Stefan-Boltzmann law and the Taylor term above.

Monin-Obukhov stability of $g_a$ for heat and momentum, from the MOST owner `relic.scientific.atmosphere.most` (Monin and Obukhov, 1954; Businger et al., 1971; Dyer, 1974; Paulson, 1970, the references of that owner):

$$
u_* = \frac{\kappa U}{f_m(\zeta)}, \quad g_a = \frac{\kappa u_*}{f_h(\zeta)}, \quad f_x = \ln\frac{z-d}{z_{0x}} - \psi_x(\zeta) + \psi_x\!\left(\zeta \frac{z_{0x}}{z-d}\right), \quad
L_{MO} = -\frac{\rho c_p T_a u_*^3}{\kappa g H}, \quad \zeta = \frac{z-d}{L_{MO}}
$$

With $\zeta = 0$ the owner gives the FAO-56 Eq. 4 conductance. The self-consistent $\zeta$ is the root of $F(\zeta) = \zeta - Z(\zeta)$ with $Z(\zeta) = \mathrm{clip}((z-d)/L_{MO}(H(\zeta)), -\zeta_u, \zeta_s)$ on the Businger-Dyer domain of the owner ($\zeta_u$ = `atmosphere.most.unstable_correction_limit`, $\zeta_s$ = `atmosphere.most.stable_correction_limit`, both 2). Because $Z$ is clipped to the same interval the root is bracketed in every cell; Section 5 gives the solver. On the daily support the 24-hour mean $\Delta T$ drives $\zeta$ (declared approximation); on the daylight support every sample has its own bracket with the daily wind.

### 2.3 Free-convection velocity (`free_convection: beljaars`, radiative form only)

The stability-corrected wind gains the convective velocity of the CTSM Technical Note (CLM5.0, Eqs. 2.5.24, 2.5.28 and 2.5.29):

$$
U = \sqrt{u^2 + U_c^2}, \qquad U_c = \beta w_* \ (\zeta < 0), \ U_c = 0 \ (\zeta \ge 0), \qquad w_* = \left(\frac{g H z_i}{\rho c_p T_a}\right)^{1/3}, \qquad
\left(\frac{w_*}{u_*}\right)^2 = \left(\frac{-\zeta z_i}{\kappa (z-d)}\right)^{2/3}
$$

so $u_* = u / \sqrt{f_m^2/\kappa^2 - \beta^2 (w_*/u_*)^2}$ with $f_m/\kappa$ from the owner's friction velocity; the 2/3 power is evaluated on $\zeta < 0$ only. $\beta$ = `biosphere.leaf_energy_balance.convective_velocity_coefficient` (1) and $z_i$ = `biosphere.leaf_energy_balance.convective_boundary_layer_height` (1000 m). Absent or `none` keeps $U = u$.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $T_a$ | Air temperature (24-hour mean on the daily support) | $\mathrm{K}$ | `air_temperature` | input |
| $T_{\max}, T_{\min}$ | Daily extrema of the air temperature (daily support) | $\mathrm{K}$ | `air_temperature_max`, `air_temperature_min` | input |
| $D$ | Vapour pressure deficit of the air | $\mathrm{Pa}$ | `vapor_pressure_deficit` | input |
| $u$ | Wind speed at the reference height | $\mathrm{m\,s^{-1}}$ | `wind_speed` | input |
| $p$ | Atmospheric pressure | $\mathrm{Pa}$ | `atmospheric_pressure` | input |
| $R_n$ | Net radiation of the surface, all available to the big leaf | $\mathrm{W\,m^{-2}}$ | `net_radiation` | input |
| $G$ | Soil heat flux (instantaneous support only) | $\mathrm{W\,m^{-2}}$ | `soil_heat_flux` | input |
| $g_c$ | Bulk canopy conductance | $\mathrm{m\,s^{-1}}$ | `canopy_conductance` | input |
| LAI | Leaf area index | $\mathrm{m^2\,m^{-2}}$ | `leaf_area_index` | input |
| $T_l$ | Leaf temperature | $\mathrm{K}$ | `leaf_temperature` | diagnostic |
| $T_{l,\max}$ | Daily maximum leaf temperature estimate | $\mathrm{K}$ | `leaf_temperature_max` | diagnostic |
| $\Delta T, H, \lambda E, A, g_a$ | Balance diagnostics | $\mathrm{K}$, $\mathrm{W\,m^{-2}}$, $\mathrm{m\,s^{-1}}$ | `diagnostics.leaf_energy_balance.*` | diagnostic |
| $e_a$ | Actual vapour pressure (`daylight_gauss`, replaces $D$) | $\mathrm{Pa}$ | `actual_vapor_pressure` | input |
| $g_{c,k}$ | Per-sample canopy conductance (`daylight_gauss`, replaces $g_c$) | $\mathrm{m\,s^{-1}}$ | `canopy_conductance_daylight_samples` | input |
| $\phi$, DOY | Latitude and day of year of the daylight samples (`daylight_gauss`) | $\mathrm{deg}$, $\mathrm{1}$ | `latitude`, `day_of_year` | input |
| $S_n$ | Daily net shortwave (`daylight_gauss`; `radiative_canopy_share`) | $\mathrm{W\,m^{-2}}$ | `shortwave_net` | input |
| $L_\downarrow$, $\varepsilon$ | Downward longwave and surface emissivity (`radiative_canopy_share`, with $S_n$ replacing $R_n$) | $\mathrm{W\,m^{-2}}$, $\mathrm{1}$ | `longwave_down`, `surface_emissivity` | input |
| $T_{l,k}$ | Leaf temperature at the daylight samples (`daylight_gauss`) | $\mathrm{K}$ | `leaf_temperature_daylight_samples` | diagnostic |
| $R_{ni}$, $A_c/A$, $g_R$, $L$, $\zeta$, residual | Radiative-form diagnostics | $\mathrm{W\,m^{-2}}$, $\mathrm{1}$, $\mathrm{m\,s^{-1}}$, $\mathrm{K}$ | `diagnostics.leaf_energy_balance.isothermal_net_radiation`, `canopy_energy_fraction`, `radiative_conductance`, `longwave_emission_change`, `stability_parameter`, `stability_iteration_residual` | diagnostic |
| $U_c$, $U$ | Convective velocity and gusty wind (`free_convection: beljaars`) | $\mathrm{m\,s^{-1}}$ | `diagnostics.leaf_energy_balance.convective_velocity`, `gusty_wind_speed` | diagnostic |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $z_{0m}, d, z$ | `hydrosphere.evapotranspiration.roughness_length`, `displacement_height`, `reference_height` | $\mathrm{m}$ | `hydrosphere/evapotranspiration.yaml` | Allen et al. (1998) Eq. 4 |
| $z_{0h}/z_{0m}$ | `hydrosphere.physical.heat_to_momentum_roughness_ratio` | $\mathrm{1}$ | `hydrosphere/physical.yaml` | Allen et al. (1998) |
| $u_{\min}$, $[g_{a,\min}, g_{a,\max}]$ | `hydrosphere.numerical.minimum_wind_speed`, `aerodynamic_conductance_bounds` | $\mathrm{m\,s^{-1}}$ | `hydrosphere/numerical.yaml` | shared with `PenmanMonteithET` |
| bounds | `hydrosphere.numerical.temperature_bounds`, `net_radiation_bounds`, `hydrosphere.physical.atmospheric_pressure_min`, `_max` | $\mathrm{K}$, $\mathrm{W\,m^{-2}}$, $\mathrm{Pa}$ | hydrosphere YAML | input plausibility (raise, never clip) |
| $\lambda, c_p, \epsilon, \kappa$, Tetens | physical-constant SSOT | SI | `physical/constants.yaml` | Allen et al. (1998) |
| $\sigma$ | physical-constant SSOT `stefan_boltzmann_constant` (`radiative_canopy_share`) | $\mathrm{W\,m^{-2}\,K^{-4}}$ | `physical/constants.yaml` | CODATA 2018 |
| $k$ | `biosphere.photosynthesis.canopy_light_extinction` (`radiative_canopy_share`) | $\mathrm{1}$ | `biosphere/photosynthesis.yaml` | Monsi and Saeki (1953); de Pury and Farquhar (1997) |
| $u_{\min}$ of the balance | `biosphere.leaf_energy_balance.minimum_wind_speed` (`radiative_canopy_share`, replaces `hydrosphere.numerical.minimum_wind_speed`) | $\mathrm{m\,s^{-1}}$ | `biosphere/leaf_energy_balance.yaml` | Allen et al. (1998) Chapter 3 and Eq. 4 |
| $\zeta_s, \zeta_u$ | `atmosphere.most.stable_correction_limit`, `unstable_correction_limit` (`radiative_canopy_share`) | $\mathrm{1}$ | `atmosphere/most.yaml` | Businger et al. (1971); Foken (2006), as cited by the owner |
| $\beta$, $z_i$ | `biosphere.leaf_energy_balance.convective_velocity_coefficient`, `convective_boundary_layer_height` (`free_convection: beljaars`) | $\mathrm{1}$, $\mathrm{m}$ | `biosphere/leaf_energy_balance.yaml` | CTSM Technical Note (CLM5.0) Eqs. 2.5.28-2.5.29 |

## 5. Numerical Methods and Discretization

Every `forward` call validates finite inputs, $T$ inside `temperature_bounds`, $T_{\min} \le T_a \le T_{\max}$ (daily), $R_n$ and $p$ inside their bounds, $D \ge 0$, $u \ge 0$, $g_c \ge 0$ and $\mathrm{LAI} \ge 0$, and raises otherwise; the daily support requires $dt = 86400\,\mathrm{s}$. The denominator $g_a[s g_c + \gamma(g_a+g_c)] \ge \gamma g_a^2 > 0$ is checked, never clamped. The only replacement is $e_a = 0$ in the air density when $D > e_s$, flagged in `diagnostics.leaf_energy_balance.vapor_pressure_floor_active`. The linearisation neglects $\tfrac12 s' \Delta T^2$; at $D = 1\,\mathrm{kPa}$ and $20\,^\circ\mathrm{C}$ the wet-bulb limit differs from the exact psychrometric depression by about $0.47\,\mathrm{K}$. Outputs are `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`. The `daylight_gauss` support also requires $dt = 86400\,\mathrm{s}$ and applies the input checks to every sample.

Stability solver of `radiative_canopy_share`: a fixed number $N$ of bisection steps on the bracket $[-\zeta_u, \zeta_s]$ (`stability_iterations`, default 40, at most 60; bracket width $(\zeta_u + \zeta_s) 2^{-N}$); the loop count never depends on the data. The bisection runs without gradient; a final implicit-function step $\zeta = \zeta^* + (Z(\zeta^*) - \zeta^*)/(1 - Z'(\zeta^*))$ ($Z'$ by a central difference with step $10^{-4}$, $1 - Z'$ floored at $10^{-3}$, result clipped to the domain) carries the derivative of the root, $d\zeta^*/d\theta = (\partial Z/\partial\theta)/(1 - \partial Z/\partial\zeta)$. The diagnostic `stability_iteration_residual` reports $|\Delta T(\zeta) - \Delta T(\zeta^*)|$. Plain and under-relaxed fixed-point iterations were rejected because they cycle across the stable/unstable switch on calm, low-conductance cells, where the bracket cannot. Where $H$ changes sign the owner's $|H|$ safeguard makes $Z$ jump by $2(z-d)/1000\,\mathrm{m}$, and the residual stays bounded by that jump.

## 6. Conservation, Boundary Conditions, and Ledgers

The Functor owns no heat or water store. On canopy cells $A = H + \lambda E$ (radiative form: $A_c = H + L + \lambda E$, with the remainder $A - A_c$ left to the not-modelled soil surface) holds by construction and $\lambda E$ equals the Penman-Monteith latent heat of FAO-56 Eq. 3; it is not projected to non-negative values (dew is kept), whereas `PenmanMonteithET` projects negative ET to zero and floors $g_c$ at `minimum_conductance`. No-canopy cells carry no leaf energy balance; their energy belongs to the soil surface, which is not modelled here.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `LeafEnergyBalance` | `relic/scientific/biosphere/photosynthesis/leaf_energy_balance.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.PATCH`) |

In the global daily graph the node runs after `atmospheric_radiation` and `medlyn_stomatal_conductance` and before `thermal_stress` (heat on `leaf_temperature_max`) and `megan_bvoc_model`; it adds no cycle, since no upstream node reads the leaf temperature in the same step.

## 8. Implementation Status

- `LeafEnergyBalance`: `[I]`, IPHS **Level 1** (`observation`, certificate `biosphere_photosynthesis_leaf_energy_balance_leaf_energy_balance.yaml`).

Frozen certificate expressions:

```text
A = R_n - G (daily G = 0); dT = [gamma (g_a + g_c) A / (rho c_p) - g_a g_c D] / (g_a [s g_c + gamma (g_a + g_c)]); T_leaf = T_air + dT, T_leaf = T_air where LAI <= 0; H = rho c_p g_a dT; lambda E = A - H; daily T_leaf,max = T_max + dT
y_h = O_h x_h through the declared observation operator
```

## 9. References

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop evapotranspiration: Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56.
- Businger, J. A., Wyngaard, J. C., Izumi, Y., and Bradley, E. F. (1971). Flux-profile relationships in the atmospheric surface layer. *Journal of the Atmospheric Sciences*, 28, 181-189 (cited through the MOST owner).
- Campbell, G. S., and Norman, J. M. (1998). *An Introduction to Environmental Biophysics* (2nd ed.). Springer, Chapter 14.
- CTSM Technical Note (CLM5.0). Momentum, Sensible Heat, and Latent Heat Fluxes, Eqs. 2.5.24, 2.5.28 and 2.5.29 (convective velocity, $\beta = 1$, $z_i = 1000$ m).
- de Pury, D. G. G., and Farquhar, G. D. (1997). Simple scaling of photosynthesis from leaves to canopies without the errors of big-leaf models. *Plant, Cell and Environment*, 20, 537-557 (canopy extinction owner).
- Dyer, A. J. (1974). A review of flux-profile relationships. *Boundary-Layer Meteorology*, 7, 363-372 (cited through the MOST owner).
- Goudriaan, J. (1986). A simple and fast numerical method for the computation of daily totals of crop photosynthesis. *Agricultural and Forest Meteorology*, 38, 249-254.
- Goudriaan, J., and van Laar, H. H. (1994). *Modelling Potential Crop Growth Processes*. Kluwer Academic Publishers.
- Jones, H. G. (2014). *Plants and Microclimate* (3rd ed.). Cambridge University Press, Chapter 9.
- Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205-224.
- Monin, A. S., and Obukhov, A. M. (1954). Basic laws of turbulent mixing in the surface layer of the atmosphere. *Trudy Geofizicheskogo Instituta AN SSSR*, 24(151), 163-187 (cited through the MOST owner).
- Monsi, M., and Saeki, T. (1953). Ueber den Lichtfaktor in den Pflanzengesellschaften und seine Bedeutung fuer die Stoffproduktion. *Japanese Journal of Botany*, 14, 22-52 (canopy extinction owner).
- Monteith, J. L., and Unsworth, M. H. (2013). *Principles of Environmental Physics* (4th ed.). Academic Press.
- Paulson, C. A. (1970). The mathematical representation of wind speed and temperature profiles in the unstable atmospheric surface layer. *Journal of Applied Meteorology*, 9, 857-861 (cited through the MOST owner).
