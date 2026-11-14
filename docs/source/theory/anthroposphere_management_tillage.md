# Agricultural Tillage Disturbance, Surface Modifiers, and Carbon Flush

## 1. Process Definition and Scope

`TillageManagement` (`relic/scientific/anthroposphere/management/tillage.py`) evaluates scheduled agricultural tillage operations (`TillageEvent`) and computes their single-layer soil disturbance index, surface residue-cover removal rate, decaying porosity and roughness modifiers, and operation-step soil $\mathrm{CO_2}$ flush.

The Functor implements `TransportMechanism` (`TRANSPORT_TYPE="managed_application"`) without owning multi-week soil organic matter decomposition kinetics or surface infiltration laws, which belong to pedosphere and hydrosphere Functors. Multi-layer column-stochastic vertical profile mixing $N' = M N$ (*Methods of Ecological Analysis*, Chapter 24, Section 24.5.3, Eqs. 24.14-24.16, and Exercise 24.4) is `[N]` in runtime code. `TillageManagement` evaluates its internal four-pass default `tillage_events` schedule directly and does not consume `tillage_intensity` emitted by `ManagementCoordinator`.

## 2. Governing Equations

### 2.1 Calendar-Window Triggering and Elapsed-Time Decay

At step $[t, t + \Delta t)$ on resolved calendar day $d \in \{1, \dots, Y(C)\}$, elapsed time since tillage advances prior to schedule evaluation as $t_{\mathrm{since}}^{-} = t_{\mathrm{since}} + \Delta t / s_{\mathrm{day}}$. A scheduled event $e$ with target day $d_e$, moisture limit $\theta_{\max,e}$, depth class $d_{\mathrm{cls},e} \in \{\mathrm{shallow}, \mathrm{medium}, \mathrm{deep}, \mathrm{very\_deep}\}$, and intensity class $i_{\mathrm{cls},e} \in \{\mathrm{none}, \mathrm{low}, \mathrm{moderate}, \mathrm{high}, \mathrm{very\_high}\}$ triggers globally when the first matching schedule entry satisfies wrap-around calendar distance, domain-mean moisture, and minimum-interval gates:

$$
\min\!\bigl(\lvert d - d_e \rvert,\; Y(C) - \lvert d - d_e \rvert\bigr) \le W_{\mathrm{till}}, \qquad
\overline{\theta} \le \theta_{\max,e}, \qquad
\overline{t_{\mathrm{since}}^{-}} > T_{\min,\mathrm{till}}
$$

When event $e$ executes, $t_{\mathrm{since}}^{+} = 0$, active factors update to $(I, D) = (I(i_{\mathrm{cls},e}), D(d_{\mathrm{cls},e}))$, and operation count $n_{\mathrm{ops}}$ increments by $1$; on non-operation steps, $t_{\mathrm{since}}^{+} = t_{\mathrm{since}}^{-}$ and $(I, D)$ retain the stored post-operation state (initialized at `TillageIntensity.NONE`, `TillageDepth.SHALLOW`). Post-tillage relaxation and physical-property modifiers follow

$$
f_t = \exp\!\bigl(-k_{\mathrm{decay}}\,t_{\mathrm{since}}^{+}\bigr), \qquad
f_{\phi} = 1 + a_{\phi}\,I\,f_t, \qquad
f_r = 1 + a_r\,I\,f_t
$$

### 2.2 Operation-Step Carbon Flush, Disturbance, and Residue Incorporation

During an executing operation step with intensity factor $I_e = I > 0$ and depth factor $D_e = D$, short-lived tillage-induced carbon oxidation $F_{\mathrm{C}}$ ($\mathrm{kg\,C\,m^{-2}\,s^{-1}}$, Reicosky, 1997) and stoichiometric $\mathrm{CO_2}$ mass flux $F_{\mathrm{CO_2}}$ ($\mathrm{kg\,CO_2\,m^{-2}\,s^{-1}}$) are

$$
F_{\mathrm{C}} = \frac{C_{\mathrm{soil}}\,f_0\,f_p\,I_e\,D_e\,Q_{10}^{(T_{\mathrm{soil}} - T_{\mathrm{std}})/10}}{s_{\mathrm{day}}}, \qquad
F_{\mathrm{CO_2}} = F_{\mathrm{C}}\,\frac{M_{\mathrm{CO_2}}}{M_{\mathrm{C}}}
$$

Simultaneously, soil disturbance index $D_{\mathrm{till}}$, Gaussian moisture-weighted residue incorporation fraction $R_{\mathrm{inc}} \in [0, 1]$, and surface residue-cover tendency $\dot{c}_{\mathrm{res}}$ ($\mathrm{s^{-1}}$) evaluate from soil moisture $\theta$ and optional state `residue_cover` $c_{\mathrm{res}} \in [0, 1]$ (required on operation steps):

$$
D_{\mathrm{till}} = k_D\,I_e\,\frac{\theta}{\theta_{\mathrm{den}}}, \qquad
g(\theta) = \exp\!\left(-\frac{(\theta - \theta_{\mathrm{opt}})^2}{2\sigma_{\theta}^2}\right)
$$

$$
R_{\mathrm{inc}} = \operatorname{clip}\!\bigl(k_R\,I_e\,(a_{\theta} + g(\theta)),\; 0,\; 1\bigr), \qquad
\Delta c_{\mathrm{res}} = -\min(c_{\mathrm{res}},\; R_{\mathrm{inc}}), \qquad
\dot{c}_{\mathrm{res}} = \frac{\Delta c_{\mathrm{res}}}{\Delta t}
$$

On steps without an executing operation, $F_{\mathrm{C}} = 0$, $F_{\mathrm{CO_2}} = 0$, $D_{\mathrm{till}} = 0$, $R_{\mathrm{inc}} = 0$, and $\dot{c}_{\mathrm{res}} = 0$.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\theta, T_{\mathrm{soil}}, C_{\mathrm{soil}}$ | Volumetric soil moisture, tilled-layer soil temperature, and soil organic carbon stock | $\mathrm{m^3\,m^{-3}}$, $\mathrm{K}$, $\mathrm{kg\,C\,m^{-2}}$ | `soil_moisture`, `soil_temperature`, `soil_organic_carbon` | `state` / `forcing` |
| $d, c_{\mathrm{res}}$ | Simulator day-of-year and surface residue-cover fraction (read on operation steps) | $\mathrm{d}$, $\mathrm{1}$ | `day_of_year`, `residue_cover` | `forcing` / `state` |
| $t_{\mathrm{since}}, n_{\mathrm{ops}}$ | Elapsed time since last tillage pass and cumulative executed operation count | $\mathrm{d}$, $\mathrm{1}$ | `days_since_tillage`, `cumulative_operations` | `state` / `diagnostic` |
| $f_{\phi}, f_r, I, f_t$ | Porosity modifier, surface roughness modifier, active intensity factor, and exponential decay factor | $\mathrm{1}$ | `porosity_modifier`, `roughness_modifier`, `tillage_intensity_factor`, `time_decay_factor` | `flux` / `diagnostic` |
| $D_{\mathrm{till}}, R_{\mathrm{inc}}$ | Step soil disturbance index and surface residue incorporation fraction | $\mathrm{1}$ | `soil_disturbance_rate`, `residue_incorporation_rate` | `flux` |
| $F_{\mathrm{C}}, F_{\mathrm{CO_2}}$ | Operation-step soil carbon loss rate and stoichiometric $\mathrm{CO_2}$ emission flux | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$, $\mathrm{kg\,CO_2\,m^{-2}\,s^{-1}}$ | `carbon_loss_rate`, `soil_co2_flux` | `flux` |
| $\dot{c}_{\mathrm{res}}$ | Surface residue-cover removal rate | $\mathrm{s^{-1}}$ | `delta_residue_cover` | `flux` |
| $R_{\mathrm{C}}$ | Available soil-carbon exceedance diagnostic $\max(0, F_{\mathrm{C}}\Delta t - C_{\mathrm{soil}} - \varepsilon_{\mathrm{tol}})$ | $\mathrm{kg\,C\,m^{-2}}$ | `diagnostics.tillage_mass_balance_error` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $k_D, k_R, k_{\mathrm{decay}}$ | `anthroposphere.management.tillage.disturbance_factor`, `anthroposphere.management.tillage.residue_incorporation_rate`, `anthroposphere.management.tillage.effect_decay_rate` | $\mathrm{1}$, $\mathrm{1}$, $\mathrm{d^{-1}}$ | `anthroposphere/management.yaml` | Lal (2004); Six et al. (2002) |
| $a_{\phi}, a_r$ | `anthroposphere.management.tillage.porosity_change`, `anthroposphere.management.tillage.roughness_intensity_coefficient` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Lal (2004) |
| $f_0, f_p$ | `anthroposphere.management.tillage.co2_pulse_base_fraction`, `anthroposphere.management.tillage.co2_pulse_factor` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Reicosky (1997) |
| $\theta_{\mathrm{opt}}, \sigma_{\theta}, \theta_{\mathrm{den}}, a_{\theta}$ | `anthroposphere.management.tillage.moisture_optimum`, `anthroposphere.management.tillage.moisture_sigma`, `anthroposphere.management.tillage.moisture_effect_denominator`, `anthroposphere.management.tillage.residue_moisture_offset` | $\mathrm{m^3\,m^{-3}}$ (first three), $\mathrm{1}$ | `anthroposphere/management.yaml` | Six et al. (2002) |
| $W_{\mathrm{till}}, T_{\min,\mathrm{till}}$ | `anthroposphere.management.tillage.schedule_window_days`, `anthroposphere.management.tillage.minimum_operation_interval_days` | $\mathrm{d}$ | `anthroposphere/management.yaml` | RELIC scheduling window conventions |
| $d_{e,0}, \theta_{\max,e,0}$ | `anthroposphere.management.tillage.default_schedule.*` | $\mathrm{d}$, $\mathrm{m^3\,m^{-3}}$ | `anthroposphere/management.yaml` | Four-pass default tillage scenario |
| $I_{\mathrm{cls}}, D_{\mathrm{cls}}$ | `anthroposphere.management.tillage.*_intensity_factor`, `anthroposphere.management.tillage.*_depth_factor` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Lal (2004); Reicosky (1997) |
| $Q_{10}, \varepsilon_{\mathrm{tol}}$ | `pedosphere.physical.q10_decomposition`, `validation.conservation.conservation_tolerance` | $\mathrm{1}$ | `pedosphere/physical.yaml`, `validation/conservation.yaml` | Davidson and Janssens (2006) |
| $T_{\mathrm{std}}, M_{\mathrm{CO_2}}, M_{\mathrm{C}}, s_{\mathrm{day}}$ | `standard_temperature`, `relic.physical.constants.molecular_weights.co2`, `relic.physical.constants.molecular_weights.carbon`, `relic.physical.constants.time.seconds_per_day` | $\mathrm{K}$, $\mathrm{g\,mol^{-1}}$, $\mathrm{g\,mol^{-1}}$, $\mathrm{s\,d^{-1}}$ | `physical/constants.yaml` | NIST / BIPM SI constants |

## 5. Numerical Methods and Discretization

`TillageManagement.forward()` evaluates schedule gates on spatial means (`soil_moisture.mean()`, `days_since_tillage.mean()`), triggering uniform domain-wide transitions. Because `F_C` divides the operation pulse fraction by fixed $s_{\mathrm{day}} = 86400\,\mathrm{s}$ while resetting `days_since_tillage` to zero (locking out re-execution for $T_{\min,\mathrm{till}}$ days), single-step integration scales cumulative pulse release by $\Delta t / s_{\mathrm{day}}$ when $\Delta t$ differs from one day. Eleven outputs—including `carbon_loss_rate` and `soil_co2_flux`—use `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`, whereas `delta_residue_cover` uses `UpdateItem(UpdateMode.ACCUMULATE, per_timestep=False, unit="s-1")`. No direct `ACCUMULATE` decrement is emitted on `soil_organic_carbon`. `get_extra_state()` and `set_extra_state()` checkpoint `current_intensity`, `current_depth`, `cumulative_tillage_operations`, `porosity_modifier`, and `roughness_modifier` (`format="relic.tillage.runtime_state"`, `version=1`); `tillage_events` is not serialized.

## 6. Conservation, Boundary Conditions, and Ledgers

`carbon_loss_rate` ($F_{\mathrm{C}}$, tagged `element="carbon"`) and `soil_co2_flux` ($F_{\mathrm{CO_2}} = (M_{\mathrm{CO_2}}/M_{\mathrm{C}})F_{\mathrm{C}}$, untagged) express the same oxidised carbon export in elemental-carbon and molecular-$\mathrm{CO_2}$ units; downstream carbon ledgers count `carbon_loss_rate` once. `_check_conservation()` evaluates carbon-stock exceedance $R_{\mathrm{C}} = \max(0, \max(F_{\mathrm{C}}, 0)\Delta t - \max(C_{\mathrm{soil}}, 0) - \varepsilon_{\mathrm{tol}})$ without clipping $F_{\mathrm{C}}$, and `check_conservation_laws()` reports $\max \lvert R_{\mathrm{C}} \rvert$ under `"mass"`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `TillageManagement` | `relic/scientific/anthroposphere/management/tillage.py` | `TransportMechanism` (`TRANSPORT_TYPE="managed_application"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `TillageManagement`: `[I]`, IPHS **Level 2** (`port`, certificate `anthroposphere_management_tillage_tillage_management.yaml`). Single-layer tillage disturbance, residue-cover decrement, decaying physical modifiers, and operation-step $\mathrm{CO_2}$ flush; multi-layer vertical soil-profile mixing matrices ($M$, $A_c$), cell-wise moisture gating, and sub-daily partition-invariant pulse integration remain `[N]`.

Frozen certificate expressions:

```text
Removed generic modifier outputs had no executable downstream composition path.
y_dot = G_h u_h with one canonical output per direct tillage contribution and no duplicated process-law output.
```

## 9. References

- Davidson, E. A., and Janssens, I. A. (2006). Temperature sensitivity of soil carbon decomposition and feedbacks to climate change. *Nature*, 440(7081), 165-173.
- Lal, R. (2004). Soil carbon sequestration impacts on global climate change and food security. *Science*, 304(5677), 1623-1627.
- Paustian, K., Six, J., Elliott, E. T., and Hunt, H. W. (2000). Management options for reducing $\mathrm{CO_2}$ emissions from agricultural soils. *Biogeochemistry*, 48(1), 147-163.
- Reicosky, D. C. (1997). Tillage-induced $\mathrm{CO_2}$ emission from soil. *Nutrient Cycling in Agroecosystems*, 49(1-3), 273-285.
- Six, J., Conant, R. T., Paul, E. A., and Paustian, K. (2002). Stabilization mechanisms of soil organic matter: Implications for C-saturation of soils. *Plant and Soil*, 241(2), 155-176.
