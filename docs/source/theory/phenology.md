# Phenology Process Formulations

## 1. Process Definition and Scope

RELIC provides four mutually exclusive canonical phenology model owners (`StaticPhenology`, `GDDPhenology`, `GSIPhenology`, and `CropPhenology` in `relic/scientific/biosphere/phenology/models.py`), all inheriting from `PhenologyBase` (`BiologicalProcessFunctor`), together with one parameter-free diagnostic aggregator (`PhenologyFunctor` in `relic/scientific/biosphere/phenology/phenology_functor.py`).

At graph-compile or bundle-construction time (`create_standard_biosphere_functors(phenology_model_owner=...)`), exactly one scientific owner is instantiated (`static_phenology`, `gdd_phenology`, `gsi_phenology`, or `crop_phenology`). That selected owner advances canopy leaf area index (`leaf_area_index`), temporal phenology memory in `StateManager`, and the shared leaf/fine-root carbon and nitrogen turnover ledger. `PhenologyFunctor` projects four observations from the selected owner into stable `diagnostics.*` keys without advancing state. Runtime equation switching, soil-water four-factor GSI extensions, and crop water/nitrogen stress phenology couplings remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 21.

## 2. Governing Equations

### 2.1 Static Calendar Phenology (`StaticPhenology`)

With day of year $t$ ($\mathrm{day}$), onset $t_{\mathrm{on}}$, offset $t_{\mathrm{off}}$, full-canopy day $t_{\mathrm{full}} = t_{\mathrm{on}} + \Delta t_{\mathrm{exp}}$, senescence start $t_{\mathrm{sen}} = t_{\mathrm{off}} - \Delta t_{\mathrm{sen}}$, and steepness $k_s$ ($\mathrm{day^{-1}}$), the canopy growth factor is:

$$
f_{\mathrm{growth}}(t) =
\begin{cases}
0, & t < t_{\mathrm{on}} \text{ or } t \ge t_{\mathrm{off}} \\
\frac{1}{2}\bigl[1 + \tanh\bigl(k_s(t - t_{\mathrm{on}})\bigr)\bigr], & t_{\mathrm{on}} \le t < t_{\mathrm{full}} \\
1, & t_{\mathrm{full}} \le t < t_{\mathrm{sen}} \\
\frac{1}{2}\bigl[1 + \tanh\bigl(k_s(t_{\mathrm{off}} - t)\bigr)\bigr], & t_{\mathrm{sen}} \le t < t_{\mathrm{off}}
\end{cases}
$$

and canopy leaf area index interpolates between $\mathrm{LAI}_{\min}$ and $\mathrm{LAI}_{\max}$.

### 2.2 Growing Degree-Day Phenology and Hybrid Event Sensitivity (`GDDPhenology`)

From Kelvin air-temperature extrema $(T_{\max,d}, T_{\min,d})$ and base temperature $T_{\mathrm{base}}$ (White et al., 1997; Richardson et al., 2013):

$$
\mathrm{GDD}_d = \max\!\left(0,\; \frac{\min(T_{\max,d}, T_{\mathrm{lim}}) + T_{\min,d}}{2} - T_{\mathrm{base}}\right), \qquad
\mathrm{GDD}_{n+1} = \mathrm{GDD}_{\mathrm{base}} + \mathrm{GDD}_d\frac{\Delta t}{t_{\mathrm{day}}}
$$

where $\mathrm{GDD}_{\mathrm{base}}$ resets at calendar year wrap detected from `phenology_gdd_previous_day_of_year`. Piecewise linear ramps over $[\mathrm{GDD}_{\mathrm{start}}, \mathrm{GDD}_{\mathrm{full}}]$ and $[\mathrm{GDD}_{\mathrm{end}}, \mathrm{GDD}_{\mathrm{senes}}]$ map $\mathrm{GDD}_{n+1}$ to `phenological_factor`, `leaf_area_index`, `phenology_phase`, and `phenology_senescence_factor`.

Leaf-onset crossing of guard $g = \mathrm{GDD} - \mathrm{GDD}_{\mathrm{start}}$ is evaluated through `relic/computation/solvers/hybrid_events.py`:

$$
\tau - t_n = \Delta t\,\frac{\mathrm{GDD}_{\mathrm{start}} - \mathrm{GDD}_n}{\mathrm{GDD}_{n+1} - \mathrm{GDD}_n}, \qquad
\frac{d\tau}{d\mathrm{GDD}_{\mathrm{start}}} = \frac{t_{\mathrm{day}}}{\mathrm{GDD}_d}, \qquad
\Delta\frac{\partial \mathrm{LAI}}{\partial \mathrm{GDD}_{\mathrm{start}}} = -\frac{\mathrm{LAI}_{\max}}{\Delta\mathrm{GDD}_{\mathrm{full}}}
$$

### 2.3 Growing Season Index Phenology (`GSIPhenology`)

Following Jolly et al. (2005), three piecewise-linear climatic indicators of daily minimum temperature $T_{\min,d}$ (converted from Kelvin to Celsius), vapor pressure deficit $\mathrm{VPD}_d$ ($\mathrm{Pa}$), and photoperiod $L_d$ ($\mathrm{s}$) define instantaneous daily $i\mathrm{GSI}_d$ and its $N = 21$-day rolling mean:

$$
i\mathrm{GSI}_d = f_T(T_{\min,d})\,f_{\mathrm{VPD}}(\mathrm{VPD}_d)\,f_{\mathrm{day}}(L_d), \qquad
\overline{\mathrm{GSI}}_d = \frac{1}{N}\sum_{j=0}^{N-1} i\mathrm{GSI}_{d-j}
$$

Hysteresis thresholds (`onset_threshold`, `offset_threshold`) update `phenology_gsi_in_growing_season` and drive LAI, phase code, and senescence on a strict daily step ($\Delta t = t_{\mathrm{day}}$).

### 2.4 Crop Thermal-Time Phenology (`CropPhenology`)

Within the configured sowing-to-harvest window (including cross-year cycles), potential heat units (PHU), optional vernalization factor $f_v$, development index $\mathrm{DI}$, and harvest index $\mathrm{HI}$ follow Ritchie (1991), McMaster and Wilhelm (1997), and Jones et al. (2003):

$$
f_v = \operatorname{clip}\!\left(\frac{V_{n+1}}{V_{\mathrm{req}}},\; f_{v,\min},\; f_{v,\max}\right), \qquad
\Delta\mathrm{PHU} = \max(0,\; T_K - T_{\mathrm{freeze}} - T_{\mathrm{base}})\frac{\Delta t}{t_{\mathrm{day}}}\,f_v
$$

$$
\mathrm{PHU}_{n+1} = \mathrm{PHU}_n + \Delta\mathrm{PHU}, \qquad
\mathrm{DI} = \operatorname{clip}\!\left(\frac{\mathrm{PHU}_{n+1}}{\mathrm{PHU}_{\mathrm{total}}},\; 0,\; 1\right), \qquad
\mathrm{HI} = \begin{cases} \mathrm{HI}_{\max}\dfrac{\mathrm{DI} - \mathrm{DI}_{\mathrm{HI,start}}}{1 - \mathrm{DI}_{\mathrm{HI,start}}}, & \mathrm{DI} \ge \mathrm{DI}_{\mathrm{HI,start}} \\ 0, & \mathrm{DI} < \mathrm{DI}_{\mathrm{HI,start}} \end{cases}
$$

with $f_v = 1$ when `vernalization_required` is false. Organ carbon allocation remains owned by `BiomassAllocation`.

### 2.5 Common Vegetation Turnover Ledger (`PhenologyBase`) and Diagnostic Projection (`PhenologyFunctor`)

Every selected owner evaluates leaf and fine-root carbon and nitrogen turnover rates ($\mathrm{kg\,element\,m^{-2}\,s^{-1}}$) from senescence factor $f_{\mathrm{sen}}$, turnover coefficients $(k_l, k_r)$, and foliar nitrogen retranslocation efficiency $\eta_N$ (Thornton et al., 2002):

$$
F_{l,C} = C_l\,k_l\,f_{\mathrm{sen}}, \qquad F_{r,C} = C_r\,k_r
$$

$$
F_{l,N} = N_l\,k_l\,f_{\mathrm{sen}}, \qquad
F_{\mathrm{retrans},N} = \eta_N\,F_{l,N}, \qquad
F_{\mathrm{litter},l,N} = (1 - \eta_N)F_{l,N}, \qquad
F_{\mathrm{litter},r,N} = N_r\,k_r
$$

`PhenologyFunctor` projects the selected owner's sub-execution outputs without advancing state:

$$
z_h = P_{\mathrm{owner}}\,y_h
$$

mapping `phenological_factor` to `diagnostics.phenology_state`, `phenology_phase` to `diagnostics.growth_stage`, `leaf_area_index` to `diagnostics.leaf_area_index`, and `phenology_senescence_factor` to `diagnostics.phenology.senescence_factor`.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $t, L_d$ | Calendar day of year and photoperiod | $\mathrm{day}$, $\mathrm{s}$ | `day_of_year`, `daylength` | `forcing` |
| $T, T_{\max,d}, T_{\min,d}, \mathrm{VPD}_d$ | Mean, daily max, daily min air temperature, and VPD | $\mathrm{K}$, $\mathrm{K}$, $\mathrm{K}$, $\mathrm{Pa}$ | `air_temperature`, `air_temperature_max`, `air_temperature_min`, `vapor_pressure_deficit` | `forcing` |
| $C_l, C_r, N_l, N_r, N_{\mathrm{plant}}$ | Leaf/fine-root C and N stocks and labile plant N pool | $\mathrm{kg\,m^{-2}}$ | `leaf_carbon`, `froot_carbon`, `leaf_nitrogen`, `froot_nitrogen`, `plant_nitrogen_pool` | `state` |
| $\mathrm{GDD}, t_{\mathrm{prev}}$ | GDD thermal-time accumulator and previous DOY | $\mathrm{degC\,day}$, $\mathrm{day}$ | `phenology_gdd_accumulated`, `phenology_gdd_previous_day_of_year` | `state` |
| $H_{\mathrm{GSI}}$ | 21-sample GSI ring buffer, write index, valid count, season flag | $1$ | `phenology_gsi_{history,history_index,history_count,in_growing_season}` | `state` |
| $\mathrm{PHU}, s_{\mathrm{sown}}, s_{\mathrm{harv}}, V$ | Crop PHU, sown/harvest flags, vernalization days | $\mathrm{degC\,day}$, $1$, $1$, $\mathrm{day}$ | `phenology_crop_{accumulated_phu,is_sown,is_harvested,vernalization_days}` | `state` |
| $\mathrm{LAI}, \phi, f_{\mathrm{sen}}, \text{phase}, \mathrm{HI}$ | Canopy LAI, activity factor, senescence factor, phase code, crop harvest index | $\mathrm{m^2\,m^{-2}}$, $1$, $1$, $1$, $1$ | `leaf_area_index`, `phenological_factor`, `phenology_senescence_factor`, `phenology_phase`, `harvest_index` | `state` / `diagnostic` |
| $F_{l,C}, F_{r,C}, F_{\mathrm{litter},l,N}, F_{\mathrm{litter},r,N}, F_{\mathrm{retrans},N}$ | Litterfall boundary exports and internal N retranslocation flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `{leaf,froot}_litterfall_{carbon,nitrogen}`, `nitrogen_retranslocation` | `flux` |

## 4. Parameter Table

Typed parameter validation is owned by `PhenologyParameters` and `PhenologyParameterSets` in `relic/scientific/biosphere/phenology/parameters.py`. Certificate parameter-key counts are 18 (`StaticPhenology`), 21 (`GDDPhenology`), 28 (`GSIPhenology`), 36 (`CropPhenology`), and 0 (`PhenologyFunctor`):

| Symbol | ParameterManager Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $\mathrm{LAI}_{\min}, \mathrm{LAI}_{\max}$ | `biosphere.phenology.morphology.{min_lai,max_lai}` | $\mathrm{m^2\,m^{-2}}$ | `phenology/*.yaml` | White et al. (1997) |
| $k_l, k_r, \eta_N$ | `biosphere.phenology.turnover.{leaf_turnover_rate,froot_turnover_rate,n_retrans_efficiency}` | $\mathrm{s^{-1}}$, $\mathrm{s^{-1}}$, $1$ | `phenology/*.yaml` | Thornton et al. (2002) |
| $t_{\mathrm{on}}, t_{\mathrm{off}}, \Delta t_{\mathrm{exp}}, \Delta t_{\mathrm{sen}}, k_s$ | `biosphere.phenology.static.{leaf_onset_doy,leaf_offset_doy,leaf_expansion_days,leaf_senescence_days,transition_steepness}` | $\mathrm{day}$, $\mathrm{day^{-1}}$ | `phenology/*.yaml` | Richardson et al. (2013) |
| $T_{\mathrm{base}}, \mathrm{GDD}_{\mathrm{start}}, \mathrm{GDD}_{\mathrm{end}}, T_{\mathrm{lim}}$ | `biosphere.phenology.gdd.{base_temp,threshold_start,threshold_end,max_temp_limit}` | $\mathrm{K}$, $\mathrm{degC\,day}$, $\mathrm{K}$ | `phenology/*.yaml` | White et al. (1997) |
| $T_{\min,\mathrm{thr}}, T_{\min,\mathrm{opt}}, \mathrm{VPD}_{\min,\max}, L_{\mathrm{thr}}$ | `biosphere.phenology.gsi.{tmin_threshold,tmin_optimal,vpd_min,vpd_max,daylen_threshold}` | $\mathrm{K}$, $\mathrm{Pa}$, $\mathrm{s}$ | `phenology/*.yaml` | Jolly et al. (2005) |
| $\mathrm{PHU}_{\mathrm{total}}, \mathrm{HI}_{\max}, f_{v,\min}, V_{\mathrm{req}}$ | `biosphere.phenology.crop.{phu_total,hi_max,vernalization_min_factor,vernalization_days_required}` | $\mathrm{degC\,day}$, $1$, $1$, $\mathrm{day}$ | `phenology/*.yaml` | Ritchie (1991); Jones et al. (2003) |

## 5. Numerical Methods and Discretization

- All prognostic temporal memory (GDD accumulator, GSI 21-day ring buffer, crop PHU/vernalization/sowing states) resides in `StateManager` via `UpdateItem(UpdateMode.REPLACE, per_timestep=True)` so checkpoint/restart reproduces trajectories without hidden Functor attributes.
- Vegetation C/N pool changes are evaluated once per step via `advance_reaction_network(...)` and returned as pre-integrated `UpdateItem(UpdateMode.ACCUMULATE, per_timestep=True)` increments, whereas litterfall and retranslocation rates use `UpdateMode.REPLACE`.
- `PhenologyFunctor` reads `subprocess_outputs[canonical_model_owner]` injected by the simulator and raises immediately on missing, non-mapping, or non-finite owner payloads.

## 6. Conservation, Boundary Conditions, and Ledgers

Every canonical phenology owner enforces exact local vegetation turnover closure:

$$
\Delta C_{\mathrm{veg}} + \Delta t\,(F_{l,C} + F_{r,C}) = 0, \qquad
\Delta N_{\mathrm{veg}} + \Delta N_{\mathrm{plant}} + \Delta t\,(F_{\mathrm{litter},l,N} + F_{\mathrm{litter},r,N}) = 0
$$

publishing residuals to `diagnostics.phenology.{carbon,nitrogen}_turnover_balance_residual`. Downstream `LitterfallInterfaceMapper` converts the four litterfall boundary exports once from $\mathrm{kg\,m^{-2}\,s^{-1}}$ to $\mathrm{g\,m^{-2}\,s^{-1}}$ for pedosphere decomposition and mineralization so coupled vegetation-soil ledgers satisfy $1000\Delta C_{\mathrm{veg}} + \Delta C_{\mathrm{soil}} + R_h\Delta t = 0$ and $1000(\Delta N_{\mathrm{veg}} + \Delta N_{\mathrm{plant}}) + \Delta N_{\mathrm{soil}} = 0$.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `StaticPhenology` | `relic/scientific/biosphere/phenology/models.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`PhenologyBase`) | `MAIN_PROCESSES` (`PATCH`) |
| `GDDPhenology` | `relic/scientific/biosphere/phenology/models.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`PhenologyBase`) | `MAIN_PROCESSES` (`PATCH`) |
| `GSIPhenology` | `relic/scientific/biosphere/phenology/models.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`PhenologyBase`) | `MAIN_PROCESSES` (`PATCH`) |
| `CropPhenology` | `relic/scientific/biosphere/phenology/models.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`PhenologyBase`) | `MAIN_PROCESSES` (`PATCH`) |
| `PhenologyFunctor` | `relic/scientific/biosphere/phenology/phenology_functor.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`PATCH`) |

## 8. Implementation Status

- `StaticPhenology`, `GDDPhenology`, `GSIPhenology`, and `CropPhenology`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_phenology_models_{static,gdd,gsi,crop}_phenology.yaml`.
- `PhenologyFunctor`: `[I]` Level 1 (`observation`, `diagnostic`) in `biosphere_phenology_phenology_functor_phenology_functor.yaml`.
- Soil-water four-factor GSI closure and crop water/nitrogen stress phenology drivers: `[N]`.

Frozen certificate equivalence expressions:

```text
Prior RELIC extension: GSI_ext = f_T f_VPD f_day f_water. The production runtime no longer exposes this expression or a selectable fallback.
x_dot = N_h r_h with ParameterManager-backed process rates r_h and declared stoichiometric/output map N_h
phi = f_selected(T, radiation, moisture, day_of_year, dt) inside PhenologyFunctor
z_h = P_owner y_h with no state advancement
```

## 9. References

1. Ritchie, J. T. (1991). Wheat phasic development. In *Modeling Plant and Soil Systems*, Agronomy Monograph 31, ASA-CSSA-SSSA, 31-54.
2. McMaster, G. S., and Wilhelm, W. W. (1997). Growing degree-days: one equation, two interpretations. *Agricultural and Forest Meteorology*, 87, 291-300.
3. White, M. A., Thornton, P. E., and Running, S. W. (1997). A continental phenology model for monitoring vegetation responses to interannual climatic variability. *Global Biogeochemical Cycles*, 11, 217-234.
4. Thornton, P. E., Law, B. E., Gholz, H. L., et al. (2002). Modeling and measuring the effects of disturbance history and climate on carbon and water budgets in evergreen needleleaf forests. *Agricultural and Forest Meteorology*, 113, 185-222.
5. Jones, J. W., Hoogenboom, G., Porter, C. H., et al. (2003). The DSSAT cropping system model. *European Journal of Agronomy*, 18, 235-265.
6. Jolly, W. M., Nemani, R., and Running, S. W. (2005). A generalized, bioclimatic index to predict foliar phenology in response to climate. *Global Change Biology*, 11, 619-632.
7. Richardson, A. D., Keenan, T. F., Migliavacca, M., et al. (2013). Climate change, phenology, and phenological control of vegetation feedbacks to the climate system. *Agricultural and Forest Meteorology*, 169, 156-173.
