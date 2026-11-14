# Prescribed Fire, Wildfire Ignition, and Biomass Combustion

## 1. Process Definition and Scope

`FireManagement` (`relic/scientific/anthroposphere/management/fire_management.py`) simulates scheduled prescribed burning and stochastic wildfire occurrences across seven vegetation classes (`grassland`, `shrubland`, `forest_boreal`, `forest_temperate`, `forest_tropical`, `agricultural`, `peat`), converting combustible aboveground biomass and surface litter into trace-gas emissions (`co2_emission`, `co_emission`, `fire_ch4_emission`, `n2o_emission`), surface ash (`ash_deposition`), and open fuel-carbon loss rates (`combustion_flux`, `carbon_loss`, `fire_fuel_carbon_input`).

Wildfire ignition draws from a Functor-owned random substream (`FunctorSubstreamMixin`) requiring an explicit integer `config["seed"]` in $[0, 2^{63})$ (legacy `random_seed` raises `ValueError`). `FireManagement` emits open reaction-port fluxes and resets `days_since_fire`, while donor-pool stock decrements on `aboveground_biomass` and `litter_mass` are not committed directly by `FireManagement`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 24 (Section 24.5.5).

## 2. Governing Equations

### 2.1 Calendar Occurrence Window and Seeded Wildfire Ignition

For simulator day $d \in [1, Y(C)]$ and scheduled `FireEvent` day $d_e$, circular calendar distance $d_C(d, d_e) = \min(\lvert d - d_e\rvert,\; Y(C) - \lvert d - d_e\rvert) \le w_{\mathrm{fire}}$ activates an annual occurrence identifier `index:year` at most once (tracked in `_executed_schedule_occurrences`). Unexecuted `FireType.WILDFIRE` occurrences (unless `force_fire=True`) draw $u \sim \mathcal{U}(0, 1)$ from the Functor substream and ignite when

$$
u < p_{\mathrm{fire}} = s_p\,(1 - M_e)\,\frac{w_e}{w_{\mathrm{thr}}}
$$

where $s_p$ is `fire_probability_scale`, $M_e$ is event fuel moisture, $w_e$ is event wind speed, and $w_{\mathrm{thr}}$ is `wind_speed_threshold`.

### 2.2 Fuel Availability, Dryness-Modified Completeness, and Step-Invariant Rates

Unless overridden by optional state `fuel_load` ($B_{\mathrm{fuel}} \ge 0$), available dry-matter fuel combines `aboveground_biomass` $B_{\mathrm{ag}}$ and `litter_mass` $B_{\mathrm{lit}}$ via vegetation-class fractions:

$$
B_{\mathrm{fuel}} = \alpha_{\mathrm{fuel}}\bigl(\phi_{\mathrm{ag}} B_{\mathrm{ag}} + \phi_{\mathrm{lit}} B_{\mathrm{lit}}\bigr)
$$

Using state `fuel_moisture` $M_{\mathrm{eff}} = M$ for `WILDFIRE` and event moisture $M_{\mathrm{eff}} = M_e$ for non-wildfire events alongside state air `temperature` $T$, the dryness-temperature multiplier $f_m$ and effective combustion completeness $\mathrm{CC}_{\mathrm{eff}}$ are

$$
\Delta T_{\mathrm{eff}} = \operatorname{clip}\!\left(\frac{T - T_{\mathrm{ref}}}{\Delta T_{\mathrm{scale}}},\; c_{\min},\; c_{\max}\right), \qquad
m_{\mathrm{eff}} = \operatorname{clip}\bigl(1 - k_m M_{\mathrm{eff}},\; c_{\min},\; c_{\max}\bigr)
$$

$$
f_m = m_{\mathrm{eff}}\bigl(w_{m,0} + w_{m,T}\Delta T_{\mathrm{eff}}\bigr), \qquad
\mathrm{CC}_{\mathrm{eff}} = \operatorname{clip}\bigl(\mathrm{CC}_{\mathrm{veg}}\,\phi_{\mathrm{int}}\,f_m,\; c_{\min},\; c_{\max}\bigr)
$$

Note that `combustion_efficiency` enters the five-channel positive diagonal Hodge mobility $H_{\mathrm{fire}} = \operatorname{diag}(w_i / \sum_{j=1}^5 w_j)$ (`combustion_completeness`, `combustion_efficiency`, `emission_factor_co2`, `emission_factor_co`, `emission_factor_ch4`) without multiplying $B_{\mathrm{fuel}}$ a second time in `forward()`. Across multiple occurrences $i$ executing in the same step, effective burn fraction $f_{\mathrm{eff},i} = f_{\mathrm{burn},i}(1 - s_{\mathrm{supp},i})$ (or state `burn_fraction` override) compounds non-overlappingly through $\Delta f_{\mathrm{burn},i} = \operatorname{clip}(f_{\mathrm{eff},i}, 0, 1)\,\max(1 - f_{\mathrm{burn,cum}}^{(i-1)}, 0)$ with $f_{\mathrm{burn,cum}}^{(i)} = f_{\mathrm{burn,cum}}^{(i-1)} + \Delta f_{\mathrm{burn},i}$. Event masses per occurrence $i$ are

$$
M_{\mathrm{fuel},i} = B_{\mathrm{fuel}}\,\mathrm{CC}_{\mathrm{eff}}\,\Delta f_{\mathrm{burn},i}, \qquad
M_{\mathrm{C},i} = M_{\mathrm{fuel},i}\,f_{\mathrm{C}}, \qquad
M_{s,i} = \frac{M_{\mathrm{C},i}}{f_{\mathrm{C}}}\frac{\mathrm{EF}_s}{1000}, \qquad
M_{\mathrm{ash},i} = M_{\mathrm{fuel},i}\,f_{\mathrm{ash}}
$$

for $s \in \{\mathrm{CO_2}, \mathrm{CO}, \mathrm{CH_4}, \mathrm{N_2O}\}$. Dividing summed masses by executing step support $\Delta t_{\mathrm{event}} = \Delta t$ and routing through `advance_reaction_network` ($N_h = I_7$, `open_system_route="port"`) guarantees $F_q\,\Delta t = \sum_i M_{q,i}$ for any $\Delta t > 0$. `days_since_fire` updates to $D' = 0$ when an occurrence executes and $D' = D + \Delta t / s_{\mathrm{day}}$ on fire-free steps.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $B_{\mathrm{ag}}, B_{\mathrm{lit}}$ | Aboveground biomass and surface litter dry-matter stocks | $\mathrm{kg\,DM\,m^{-2}}$ | `aboveground_biomass`, `litter_mass` | `state` |
| $M, T$ | Fuel moisture content and air temperature | $\mathrm{1}$, $\mathrm{K}$ | `fuel_moisture`, `temperature` | `forcing` |
| $D$ | Elapsed time since last executed fire occurrence | $\mathrm{d}$ | `days_since_fire` | `state` |
| $d, B_{\mathrm{fuel}}, f_{\mathrm{burn,ov}}$ | Optional day-of-year, fuel-load override, and burn-fraction override | $\mathrm{d}$, $\mathrm{kg\,DM\,m^{-2}}$, $\mathrm{1}$ | `day_of_year`, `fuel_load`, `burn_fraction` | `forcing` |
| $f_{\mathrm{burn,cum}}, F_{\mathrm{comb}}$ | Step burned area fraction and dry-matter combustion flux | $\mathrm{1}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `burned_fraction`, `combustion_flux` | `flux` |
| $F_{\mathrm{C,loss}}, F_{\mathrm{C,in}}$ | Combusted carbon loss and paired fire-fuel carbon port rate | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `carbon_loss`, `fire_fuel_carbon_input` | `flux` |
| $F_{\mathrm{CO2}}, F_{\mathrm{CO}}, F_{\mathrm{CH4}}, F_{\mathrm{N2O}}$ | Atmospheric trace-gas mass emission fluxes (`fire_ch4_emission` distinct from soil `ch4_emission`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `co2_emission`, `co_emission`, `fire_ch4_emission`, `n2o_emission` | `flux` |
| $F_{\mathrm{ash}}$ | Combusted biomass ash deposition mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `ash_deposition` | `flux` |
| $n_{\mathrm{evt}}, \Delta t_{\mathrm{event}}, \epsilon_{\mathrm{C}}$ | Executed event count, step support, and carbon-availability excess | $\mathrm{1}$, $\mathrm{s}$, $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `diagnostics.fire.events_today`, `diagnostics.fire.event_support_seconds`, `diagnostics.fire.carbon_availability_residual` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $T_{\mathrm{ref}}, c_{\min}, c_{\max}$ | `anthroposphere.management.temperature`, `anthroposphere.management.clamp_min`, `anthroposphere.management.clamp_max` | $\mathrm{K}$, $\mathrm{1}$, $\mathrm{1}$ | `anthroposphere/management.yaml` | Moisture-temperature response bounds |
| $\mathrm{CC}_0, \mathrm{CE}_0, f_{\mathrm{C}}, f_{\mathrm{ash}}$ | `anthroposphere.management.fire.combustion_completeness`, `anthroposphere.management.fire.combustion_efficiency`, `anthroposphere.management.fire.biomass_carbon_fraction`, `anthroposphere.management.fire.ash_fraction` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Seiler and Crutzen (1980); Certini (2005) |
| $\mathrm{EF}_0$ | `anthroposphere.management.fire.emission_factor_co2`, `anthroposphere.management.fire.emission_factor_co`, `anthroposphere.management.fire.emission_factor_ch4` | $\mathrm{g\,kg^{-1}}$ | `anthroposphere/management.yaml` | Andreae and Merlet (2001) |
| $s_p, w_{\mathrm{thr}}, w_{\mathrm{fire}}$ | `anthroposphere.management.fire.fire_probability_scale`, `anthroposphere.management.fire.wind_speed_threshold`, `anthroposphere.management.fire.schedule_window_days` | $\mathrm{1}$, $\mathrm{m\,s^{-1}}$, $\mathrm{d}$ | `anthroposphere/management.yaml` | Bond and Keeley (2005) |
| $k_m, w_{m,0}, w_{m,T}, \Delta T_{\mathrm{scale}}$ | `anthroposphere.management.fire.moisture_reduction_coefficient`, `anthroposphere.management.fire.moisture_temp_base_weight`, `anthroposphere.management.fire.moisture_temp_scale_weight`, `anthroposphere.management.fire.temperature_scale_range` | $\mathrm{1}$, $\mathrm{1}$, $\mathrm{1}$, $\mathrm{K}$ | `anthroposphere/management.yaml` | Fuel-moisture combustion response |
| $\phi_{\mathrm{int}}, \theta_{\mathrm{veg}}, \theta_{\mathrm{sched}}$ | `anthroposphere.management.fire.intensity_factor.*`, `anthroposphere.management.fire.vegetation_properties.*`, `anthroposphere.management.fire.default_schedule.*` | $\mathrm{1}$, $\mathrm{g\,kg^{-1}}$, $\mathrm{d}$, $\mathrm{m\,s^{-1}}$ | `anthroposphere/management.yaml` | van der Werf et al. (2017); Andreae and Merlet (2001) |
| $\tau_{\mathrm{cons}}$ | `validation.conservation.conservation_tolerance` | $\mathrm{1}$ | `validation/conservation.yaml` | Hodge metric tolerance |
| $s_{\mathrm{day}}, M_{\mathrm{C}}, M_{\mathrm{CO2}}, M_{\mathrm{CH4}}, M_{\mathrm{O2}}$ | `relic.physical.constants.time.seconds_per_day`, `relic.physical.constants.molecular_weights.carbon`, `relic.physical.constants.molecular_weights.co2`, `relic.physical.constants.molecular_weights.ch4`, `relic.physical.constants.molecular_weights.o2` | $\mathrm{s\,d^{-1}}$, $\mathrm{g\,mol^{-1}}$ | `physical/constants.yaml` | IUPAC molar weights and SI day seconds |

## 5. Numerical Methods and Discretization

All 14 outputs (including updated `days_since_fire`) are wrapped in `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`. Module extra state (`get_extra_state` / `set_extra_state`, format version 3) snapshots `fire_events`, `executed_schedule_occurrences`, and `FunctorSubstreamMixin` RNG state (`seed`, `draw_count`, `generator_state`, `wiener_pending`), ensuring bit-exact rollback on step rejection and common-random-number replay under parameter sensitivity.

## 6. Conservation, Boundary Conditions, and Ledgers

`FireManagement` verifies that elemental carbon emitted across $\mathrm{CO_2}$, $\mathrm{CO}$ ($M_{\mathrm{CO}} = M_{\mathrm{C}} + \frac{1}{2}M_{\mathrm{O_2}}$), and $\mathrm{CH_4}$ never exceeds total combusted carbon `carbon_loss` ($F_{\mathrm{C,loss}}$), allowing non-negative unspeciated pyrogenic carbon or char residual:

$$
\epsilon_{\mathrm{C}} = \operatorname{relu}\!\left(F_{\mathrm{CO2}}\frac{M_{\mathrm{C}}}{M_{\mathrm{CO2}}} + F_{\mathrm{CO}}\frac{M_{\mathrm{C}}}{M_{\mathrm{CO}}} + F_{\mathrm{CH4}}\frac{M_{\mathrm{C}}}{M_{\mathrm{CH4}}} - F_{\mathrm{C,loss}}\right)
$$

`check_conservation_laws()` evaluates $\max\lvert \epsilon_{\mathrm{C}}\rvert$.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `FireManagement` | `relic/scientific/anthroposphere/management/fire_management.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `FireManagement`: `[I]`, IPHS **Level 3** (`reaction`, certificate `anthroposphere_management_fire_management_fire_management.yaml`). Step-invariant combustion reaction-port map with seeded substream wildfire ignition, positive diagonal Hodge mobility $H_{\mathrm{fire}}$, and replay-safe occurrence deduplication.

Frozen certificate expressions:

```text
C_combusted = B_fuel * CC * CE * (1 - M_factor); E_species = C_combusted / f_C * EF_species.
The biomass burning equation defines entries of the standard reaction vector r_h inside the single fire IPHS combustion-port map.
```

## 9. References

- Andreae, M. O., and Merlet, P. (2001). Emission of trace gases and aerosols from biomass burning. *Global Biogeochemical Cycles*, 15(4), 955-966.
- Bond, W. J., and Keeley, J. E. (2005). Fire as a global 'herbivore': the ecology and evolution of flammable ecosystems. *Trends in Ecology & Evolution*, 20(7), 387-394.
- Certini, G. (2005). Effects of fire on properties of forest soils: a review. *Oecologia*, 143(1), 1-10.
- Seiler, W., and Crutzen, P. J. (1980). Estimates of gross and net fluxes of carbon between the biosphere and the atmosphere from biomass burning. *Climatic Change*, 2(3), 207-247.
- van der Werf, G. R., Randerson, J. T., Giglio, L., et al. (2017). Global fire emissions estimates during 1997-2016. *Earth System Science Data*, 9(2), 697-720.
