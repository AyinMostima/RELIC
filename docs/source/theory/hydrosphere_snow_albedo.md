# Hydrosphere Snow Albedo Evolution

## 1. Process Definition and Scope

`SnowAlbedo` (`relic/scientific/hydrosphere/snow/albedo.py`) evaluates snow optical aging and broadband `surface_albedo` using either the spectral Biosphere-Atmosphere Transfer Scheme (`albedo_scheme="bats"`, Dickinson et al., 1993; Wiscombe and Warren, 1980) or an exponential age-decay scheme (`albedo_scheme="age_based"`).

`SnowAlbedo` declares an optional dependency on `snow_pack`. Each scheme reads and updates exclusively its own prognostic optical-age state (`snow_nondimensional_age` under `"bats"`, or `snow_surface_age` under `"age_based"`); switching schemes across a checkpoint does not translate between the two age histories. Explicit prognostic snow grain-radius radiative transfer (SNTHERM/SNICAR multi-band Mie scattering) and canopy-masked sub-canopy snow albedo remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Section 19.4.2).

## 2. Governing Equations

### 2.1 Fresh-Snow Reset, Step Aging Order, and No-Snow Continuation

For cells with snow cover (`snow_depth` $z_s > z_{s,\min}$), fresh snowfall `snowfall` $P_{\mathrm{snow}} > \theta_{\mathrm{reset}}$ (`snow_reset_threshold`) resets the starting optical age to zero **prior** to adding the current step's aging increment:

$$
a_s^* =
\begin{cases}
0, & P_{\mathrm{snow}} > \theta_{\mathrm{reset}} \\
a_s^n, & P_{\mathrm{snow}} \le \theta_{\mathrm{reset}}
\end{cases}
$$

Consequently, at the end of a heavy-snowfall step with $z_s > z_{s,\min}$, $a_s^{n+1}$ equals one step's aging increment. Where snow cover is absent ($z_s \le z_{s,\min}$), the active age state is zeroed ($a_s^{n+1} = 0$) and bare-ground albedo continuation applies:

$$
\alpha_{\mathrm{raw}} = \alpha_{\mathrm{old}}\,f_{\mathrm{bare}}, \qquad \alpha_s = \operatorname{clip}(\alpha_{\mathrm{raw}},\; \alpha_{\min},\; \alpha_{\max})
$$

### 2.2 BATS Nondimensional Aging and Spectral Zenith Closure (`"bats"`)

With snow surface temperature $T_s$ clamped to $[T_{\min}, T_{\max}]$, cold/warm decay rate $k_{\mathrm{decay}}(T_s)$ ($\mathrm{s^{-1}}$, switching at $T_{\mathrm{warm}}$), vapor-diffusion grain-growth factor $r_1$, melt-refreeze factor $r_2$, and soot/dust factor $r_3 = c_{\mathrm{soot}}$ (Dickinson et al., 1993; Warren, 1982):

$$
r_1 = \exp\!\left[b_g\left(\frac{1}{T_{\mathrm{ref}}} - \frac{1}{T_s + \epsilon_T}\right)\right], \qquad
r_2 = \min\bigl(r_1^{10},\; 1\bigr), \qquad
\tau_s^{n+1} = \tau_s^* + \Delta t\,k_{\mathrm{decay}}(T_s)(r_1 + r_2 + r_3)
$$

From $f_{\mathrm{age}} = \tau_s^{n+1} / (1 + \tau_s^{n+1})$ and solar zenith angle $\theta_z$ (activating zenith factor $f_z$ when $\cos\theta_z < \cos\theta_{\mathrm{thr}}$ with parameter $b_z$ = `bats_b_parameter`):

$$
f_z =
\begin{cases}
\dfrac{1}{b_z}\!\left(\dfrac{1 + b_z}{1 + 2b_z\cos\theta_z} - 1\right), & \cos\theta_z < \cos\theta_{\mathrm{thr}} \\
0, & \cos\theta_z \ge \cos\theta_{\mathrm{thr}}
\end{cases}
$$

$$
\alpha_{\mathrm{vis},0}' = \alpha_{\mathrm{vis,base}}(1 - c_{\mathrm{vis}} f_{\mathrm{age}}), \qquad
\alpha_{\mathrm{vis}} = \alpha_{\mathrm{vis},0}' + c_z f_z(1 - \alpha_{\mathrm{vis},0}')
$$

$$
\alpha_{\mathrm{nir},0}' = \alpha_{\mathrm{nir,base}}(1 - c_{\mathrm{nir}} f_{\mathrm{age}}), \qquad
\alpha_{\mathrm{nir}} = \alpha_{\mathrm{nir},0}' + c_z f_z(1 - \alpha_{\mathrm{nir},0}')
$$

$$
\alpha_{\mathrm{raw}} = w_{\mathrm{vis}}\alpha_{\mathrm{vis}} + (1 - w_{\mathrm{vis}})\alpha_{\mathrm{nir}}, \qquad
\alpha_s = \operatorname{clip}(\alpha_{\mathrm{raw}},\; \alpha_{\min},\; \alpha_{\max})
$$

### 2.3 Exponential Age-Decay Closure (`"age_based"`)

With $s_{\mathrm{day}}$ from `get_physical_constant("seconds_per_day")`, age state $t_{\mathrm{age}}$ ($\mathrm{day}$), warm sensitivity $\kappa$, and time scale $\tau_{\mathrm{age}}$ ($\mathrm{day}$):

$$
t_{\mathrm{age}}^{n+1} = t_{\mathrm{age}}^* + \frac{\Delta t}{s_{\mathrm{day}}}, \qquad
f_T = \begin{cases} \exp\bigl[\kappa(T - T_{\mathrm{warm}})\bigr], & T > T_{\mathrm{warm}} \\ 1, & T \le T_{\mathrm{warm}} \end{cases}
$$

$$
\alpha_{\mathrm{raw}} = \alpha_{\mathrm{old}} + (\alpha_{\mathrm{fresh}} - \alpha_{\mathrm{old}})\exp\!\left(-\frac{t_{\mathrm{age}}^{n+1} f_T}{\tau_{\mathrm{age}}}\right), \qquad
\alpha_s = \operatorname{clip}(\alpha_{\mathrm{raw}},\; \alpha_{\min},\; \alpha_{\max})
$$

### 2.4 Shortwave Observation Operator and Port Diagnostics

Absorbed shortwave fraction $a_{\mathrm{sw}} = 1 - \alpha_s$ defines passive Hodge operator $\star_\alpha = \operatorname{diag}\bigl(\max(1 - \alpha_s, \varepsilon_{\mathrm{small}})\bigr) \succeq 0$. When optional downwelling shortwave flux $I_{\mathrm{sw}}$ (`shortwave_down`, $\mathrm{W\,m^{-2}}$) is supplied, radiation port power $P_{\mathrm{rad}} = I_{\mathrm{sw}}(1 - \alpha_s)$ is recorded in `diagnostics.snow_albedo`. Optical-age tendency $\dot{a}_s = (a_s^{n+1} - a_s^n)/\Delta t$ is recorded as `snow_nondimensional_age_tendency` ($\mathrm{s^{-1}}$) under `"bats"` and `snow_age_tendency` ($\mathrm{day\,s^{-1}}$) under `"age_based"`.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $P_{\mathrm{snow}}$ | Snowfall mass flux (required) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `snowfall` | `flux` |
| $T_s, T$ | Snow surface temperature and air temperature | $\mathrm{K}$ | `snow_surface_temperature`, `air_temperature` | `state` / `forcing` |
| $\theta_z$ | Solar zenith angle | $\mathrm{rad}$ or $\mathrm{deg}$ | `solar_zenith_angle` | `forcing` |
| $z_s$ | Snowpack geometric depth (required) | $\mathrm{m}$ | `snow_depth` | `state` |
| $I_{\mathrm{sw}}$ | Optional surface downwelling shortwave radiation | $\mathrm{W\,m^{-2}}$ | `shortwave_down` | `forcing` |
| $\alpha_s$ | Broadband snow or bare-ground surface albedo | $\mathrm{1}$ | `surface_albedo` | `diagnostic` |
| $t_{\mathrm{age}}$ | Age-based snow optical age state | $\mathrm{day}$ | `snow_surface_age` | `state` |
| $\tau_s$ | BATS nondimensional snow age state | $\mathrm{1}$ | `snow_nondimensional_age` | `state` |
| $\mathcal{D}_{\mathrm{alb}}$ | Nested absorbed-fraction, clipping-residual, age-tendency, and radiation port-power dictionary | mixed SI | `diagnostics.snow_albedo` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $s_{\mathrm{alb}}$ | `hydrosphere.snow.albedo_scheme` | `str` | `hydrosphere/snow.yaml` | `"bats"` or `"age_based"` |
| $\alpha_{\mathrm{fresh}}, \alpha_{\mathrm{old}}, f_{\mathrm{bare}}$ | `hydrosphere.snow.fresh_snow_albedo`, `old_snow_albedo`, `bare_ground_albedo_factor` | $\mathrm{1}$ | `hydrosphere/snow.yaml` | Warren (1982); Molotch and Bales (2006) |
| $\theta_{\mathrm{reset}}, z_{s,\min}$ | `hydrosphere.snow.snow_reset_threshold`, `minimum_snow_depth` | $\mathrm{kg\,m^{-2}\,s^{-1}}$, $\mathrm{m}$ | `hydrosphere/snow.yaml` | Dickinson et al. (1993) |
| $k_{\mathrm{cold}}, k_{\mathrm{warm}}, T_{\mathrm{warm}}$ | `hydrosphere.snow.albedo_decay_rate_cold`, `albedo_decay_rate_warm`, `temperature_threshold_warm` | $\mathrm{s^{-1}}$, $\mathrm{K}$ | `hydrosphere/snow.yaml` | Dickinson et al. (1993) |
| $b_g, T_{\mathrm{ref}}, \epsilon_T, c_{\mathrm{soot}}$ | `hydrosphere.snow.bats_grain_growth_temperature_factor`, `bats_reference_temperature`, `bats_temperature_epsilon`, `dust_soot_factor` | $\mathrm{K}$, $\mathrm{K}$, $\mathrm{K}$, $\mathrm{1}$ | `hydrosphere/snow.yaml` | Dickinson et al. (1993) |
| $b_z, \cos\theta_{\mathrm{thr}}, c_z$ | `hydrosphere.snow.bats_b_parameter`, `bats_cos_zenith_threshold`, `bats_albedo_zenith_factor` | $\mathrm{1}$ | `hydrosphere/snow.yaml` | Wiscombe and Warren (1980); Dickinson et al. (1993) |
| $\alpha_{\mathrm{vis,base}}, c_{\mathrm{vis}}, \alpha_{\mathrm{nir,base}}, c_{\mathrm{nir}}, w_{\mathrm{vis}}$ | `hydrosphere.snow.bats_albedo_vis_base`, `bats_albedo_vis_age_factor`, `bats_albedo_nir_base`, `bats_albedo_nir_age_factor`, `bats_albedo_average_weight` | $\mathrm{1}$ | `hydrosphere/snow.yaml` | Dickinson et al. (1993) |
| $\kappa, \tau_{\mathrm{age}}$ | `hydrosphere.snow.age_based_temperature_sensitivity`, `aging_time_scale` | $\mathrm{K^{-1}}$, $\mathrm{day}$ | `hydrosphere/snow.yaml` | Molotch and Bales (2006) |
| $\alpha_{\min}, \alpha_{\max}, \alpha_{\mathrm{ref,H}}$ | `hydrosphere.snow.albedo_clamp_min`, `albedo_clamp_max`, `hodge_reference_surface_albedo` | $\mathrm{1}$ | `hydrosphere/snow.yaml` | RELIC domain and Hodge contract |
| $T_{\min}, T_{\max}, \varepsilon_{\mathrm{small}}$ | `hydrosphere.snow.temperature_clamp_min`, `temperature_clamp_max`, `hydrosphere.numerical.epsilon_small` | $\mathrm{K}$, $\mathrm{1}$ | `hydrosphere/snow.yaml`, `numerical.yaml` | RELIC numerical bounds |
| $s_{\mathrm{day}}$ | `relic.physical.constants.seconds_per_day` | $\mathrm{s\,day^{-1}}$ | `physical/constants.yaml` | SI time constant |

## 5. Numerical Methods and Discretization

`forward()` validates every scheme-specific required input before evaluation and emits `UpdateItem(UpdateMode.REPLACE)` updates on differentiable `torch.Tensor` values. `get_value_at()` and `interpolate()` raise `RuntimeError` so `StateManager` remains the sole owner of `surface_albedo` and optical age states.

## 6. Conservation, Boundary Conditions, and Ledgers

`SnowAlbedo` owns no water or thermal energy stock. Its local `energy` partition check `SnowAlbedo.check_conservation_laws` verifies unit shortwave reflectance-absorptance closure $r_E = \max_i\lvert\alpha_{s,i} + a_{\mathrm{sw},i} - 1\rvert = 0$ and routes bound-clipping residuals to `diagnostics.snow_albedo["albedo_clipping_residual"]`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `SnowAlbedo` | `relic/scientific/hydrosphere/snow/albedo.py` | `PhysicalField` (`SPATIAL_DIMS=None`) | `PhysicalFieldFunctor` (`relic/core/functor/physical_field_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `SnowAlbedo`: `[I]`, IPHS **Level 1** (`observation`, certificate `hydrosphere_snow_albedo_snow_albedo.yaml`). Optical reflectance observation with scheme-specific prognostic age update and passive shortwave partition audit.
- Prognostic multi-layer snow grain-size radiative transfer: `[N]`.

Frozen certificate expressions:

```text
alpha_s = clamp(Phi_age(age,T,theta_z), alpha_min, alpha_max), with no-snow bare-ground continuation alpha_old * bare_ground_albedo_factor.
q_rad = star_alpha I_sw, P_rad = I_sw * (1 - alpha_s), and d age_s/dt = (age_s_next - age_s_start) / dt; clipping, no-snow projection, port-power, and age-tendency residuals are routed through diagnostics.snow_albedo.
```

## 9. References

- Dickinson, R. E., Henderson-Sellers, A., and Kennedy, P. J. (1993). *Biosphere-Atmosphere Transfer Scheme (BATS) Version 1e as Coupled to the NCAR Community Climate Model* (NCAR/TN-387+STR). NCAR.
- Molotch, N. P., and Bales, R. C. (2006). Comparison of radar and passive microwave snow water equivalent estimates in the Rio Grande headwaters. *Water Resources Research*, 42, W06417.
- Warren, S. G. (1982). Optical properties of snow. *Reviews of Geophysics and Space Physics*, 20(1), 67-89.
- Wiscombe, W. J., and Warren, S. G. (1980). A model for the spectral albedo of snow. I: Pure snow. *Journal of the Atmospheric Sciences*, 37(12), 2712-2733.
