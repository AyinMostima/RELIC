# Multi-Practice Agricultural Management Coordination and Event Scheduling

## 1. Process Definition and Scope

`ManagementCoordinator` (`relic/scientific/anthroposphere/management/coordinator.py`) aggregates agroecosystem state observations, lagged management flux diagnostics, calendar seasonality, and an internal `EventScheduler` (`relic/scientific/anthroposphere/management/event_scheduler.py`) to emit five normalized management priorities and three next-step practice control signals (`irrigation_demand_flux`, `fertilizer_product_flux`, `tillage_intensity`).

`ManagementCoordinator` executes in `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) with lagged feedback dependencies on `irrigation_management`, `harvest_management`, `grazing_management`, and `fire_management`. Its `optimize_management_practices` routine evaluates a closed-form priority-weighted heuristic projection (no iterative trajectory optimization is solved). Inside `EventScheduler`, spatial conflict detection compares `spatial_extent` tuples by exact equality; general polygon-intersection conflict geometry remains `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 24 (Section 24.5).

## 2. Governing Equations

### 2.1 Five-Axis Priority Projection and Structural Hodge Metric

From `CalendarHandler`, day-of-year $d = \lfloor t / s_{\mathrm{day}}\rfloor \bmod Y(C) + 1$ selects active seasonal windows (`spring`, `summer`, `autumn`, `winter`). Five normalized stress or capacity indicators are averaged over the spatial support for $i \in \{\text{nutrients}, \text{water}, \text{soil\_health}, \text{productivity}, \text{environmental}\}$:

$$
s_N = \operatorname{clip}\!\left(1 - \frac{N}{N_{\mathrm{ref}}},\; 0,\; 1\right), \qquad
s_W = \operatorname{clip}\!\left(1 - \frac{\theta}{\theta_{\mathrm{ref}}},\; 0,\; 1\right), \qquad
s_C = \operatorname{clip}\!\left(\frac{C_{\mathrm{soc}}}{C_{\mathrm{ref}}},\; 0,\; b_{\max}\right)
$$

$$
s_P = \operatorname{clip}\!\left(\frac{M_{\mathrm{crop}}}{P_{\mathrm{ref}}},\; 0,\; b_{\max}\right), \qquad
s_T = \operatorname{clip}\!\left(\frac{\lvert T_{\mathrm{air}} - T_{\mathrm{ref}}\rvert}{T_{\mathrm{ref}}},\; 0,\; 1\right)
$$

With positive base weights $w_i > 0$, unnormalized scores $q_i = w_i(1 + \bar{s}_i)$ project onto the simplex $p_i$ alongside positive diagonal structural Hodge metric $H_m$:

$$
p_i = \frac{q_i}{\sum_{j=1}^{5} q_j}, \qquad
H_m = \operatorname{diag}\!\left(\frac{w_i}{\sum_{j=1}^{5} w_j}\right) \succ 0, \qquad
I_{ab} = \operatorname{clip}\!\bigl(k_{\mathrm{int}}(p_a + p_b),\; I_{\min},\; I_{\max}\bigr)
$$

where $I_{ab}$ represents bounded pairwise practice-interaction multipliers (`fertilization_irrigation`, `tillage_fertilization`, `grazing_fire`, `harvest_irrigation`).

### 2.2 Next-Step Practice Control Projections and Event Queue Efficiency

Given coordination efficiency $\eta_c = \operatorname{clip}(\eta_e\,\eta_0, c_{\min}, c_{\max})$ derived from queue resource-need efficiency $\eta_e = \operatorname{clip}(N_{\mathrm{need}} / \sum_r (R_r / Q_r), 0, 1)$ ($\eta_e = 1$ when $\sum_r R_r / Q_r = 0$), strategy multiplier $\mu_{\mathrm{strat}}$, spring urea reference mass $M_{\mathrm{urea}}$, and crop-maturity factor $f_{\mathrm{mat}} = \overline{\operatorname{clip}(M_{\mathrm{crop}} / P_{\mathrm{ref}}, f_{\mathrm{mat,\min}}, f_{\mathrm{mat,\max}})}$, the three next-step control ports are:

$$
F_{\mathrm{irr,dem}} = \max(E - P,\; 0), \qquad
F_{\mathrm{fert,prod}} = \max\!\left(0,\; \eta_c\,p_{\mathrm{nutrients}}\,I_{\mathrm{fert,irr}}\,\frac{M_{\mathrm{urea}}}{s_{\mathrm{day}}}\,f_{\mathrm{mat}}\,\mu_{\mathrm{strat}}\right)
$$

$$
u_{\mathrm{till}} = \operatorname{clip}\!\bigl(\eta_c\,p_{\mathrm{soil\_health}}\,I_{\mathrm{till,fert}}\,u_{\mathrm{till,0}},\; c_{\min},\; c_{\max}\bigr)
$$

Here $F_{\mathrm{irr,dem}}$ sets effective precipitation equal to total `precipitation_flux` $P$. `EventScheduler` orders `ManagementEvent` records in a min-heap by `(scheduled_time, priority, event_id)` and executes due events satisfying $t_e \le t_c$ subject to resource quotas $0 \le U_r \le Q_r$, blackout windows, and minimum inter-event intervals.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\theta, \theta_{\mathrm{fc}}, \theta_{\mathrm{wp}}$ | Soil volumetric water content, field capacity, and wilting point | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture`, `field_capacity`, `wilting_point` | `state`, `parameter` |
| $N, C_{\mathrm{soc}}$ | Soil mineral nitrogen and soil organic carbon pools | $\mathrm{kg\,N\,m^{-2}}$, $\mathrm{kg\,C\,m^{-2}}$ | `soil_nitrogen`, `soil_organic_carbon` | `state` |
| $M_{\mathrm{crop}}, B_{\mathrm{ag}}$ | Crop maturity index and aboveground biomass carbon | $\mathrm{1}$, $\mathrm{kg\,C\,m^{-2}}$ | `crop_maturity`, `aboveground_biomass` | `state` |
| $T_{\mathrm{air}}, E, P$ | Air temperature, evapotranspiration flux, and precipitation flux | $\mathrm{K}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `air_temperature`, `evapotranspiration_flux`, `precipitation_flux` | `forcing`, `flux` |
| $F_{\mathrm{irr}}, F_{\mathrm{harv}}, F_{\mathrm{graz}}, F_{\mathrm{CO2}}$ | Lagged irrigation, harvest, grazing, and fire $\mathrm{CO_2}$ fluxes | $\mathrm{kg\,m^{-2}\,s^{-1}}$ (water or C) | `irrigation_flux`, `harvest_flux`, `grazing_flux`, `co2_emission` | `flux` |
| $F_{\mathrm{irr,dem}}, F_{\mathrm{fert,prod}}, u_{\mathrm{till}}$ | Next-step irrigation demand, fertilizer product rate, and tillage intensity | $\mathrm{kg\,m^{-2}\,s^{-1}}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$, $\mathrm{1}$ | `irrigation_demand_flux`, `fertilizer_product_flux`, `tillage_intensity` | `control` |
| $p_{\mathrm{water}}, p_{\mathrm{nutrients}}, p_{\mathrm{soil\_health}}, p_{\mathrm{productivity}}, p_{\mathrm{environmental}}$ | Five simplex-normalized management priority outputs | $\mathrm{1}$ | `water_priority`, `nutrient_priority`, `soil_health_priority`, `productivity_priority`, `environmental_priority` | `diagnostic` |
| $R_{\mathrm{mass}}, R_{\mathrm{quota}}$ | Biomass-removal bound excess and resource-quota violation diagnostics | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$, $\mathrm{1}$ | `diagnostics.coordinator_mass_balance_error`, `diagnostics.resource_quota_violation` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $w_i, N_{\mathrm{ref}}, \theta_{\mathrm{ref}}, C_{\mathrm{ref}}, P_{\mathrm{ref}}, T_{\mathrm{ref}}$ | `anthroposphere.management.coordinator.*` | $\mathrm{1}$, $\mathrm{kg\,m^{-2}}$, $\mathrm{m^3\,m^{-3}}$, $\mathrm{K}$ | `anthroposphere/management.yaml` | Multi-objective priority normalization |
| $k_{\mathrm{int}}, I_{\min}, I_{\max}, \eta_0, \mu_{\mathrm{strat}}, Q_r$ | `anthroposphere.management.coordinator.*` | $\mathrm{1}$, quota units | `anthroposphere/management.yaml` | Practice interaction and quota bounds |
| $c_{\min}, c_{\max}$ | `anthroposphere.management.clamp_min`, `anthroposphere.management.clamp_max` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Control clamping bounds |
| $M_{\mathrm{urea}}, u_{\mathrm{till,0}}$ | `anthroposphere.management.fertilization.spring_urea_product_mass`, `anthroposphere.management.tillage.low_intensity_factor` | $\mathrm{kg\,m^{-2}}$, $\mathrm{1}$ | `anthroposphere/management.yaml` | Reference fertilizer and tillage scales |
| $d_{\mathrm{irr,0}}, \mathrm{HI}_0, \eta_{\mathrm{graz,0}}, r_{\mathrm{int,0}}, f_{\mathrm{man,0}}$ | `anthroposphere.management.irrigation.default_application_depth`, `anthroposphere.management.harvest.default_harvest_index`, `anthroposphere.management.grazing.base_efficiency`, `anthroposphere.management.grazing.daily_intake_rate`, `anthroposphere.management.grazing.manure_return_fraction` | $\mathrm{m}$, $\mathrm{1}$, $\mathrm{d^{-1}}$ | `anthroposphere/management.yaml` | Default seasonal schedule event payloads |
| $t_{\mathrm{graz}}, f_{\mathrm{fire,0}}$ | `anthroposphere.management.grazing.default_schedule.spring.*`, `anthroposphere.management.fire.default_fire_type`, `anthroposphere.management.fire.default_fuel_load_fraction`, `anthroposphere.management.fire.fuel_consumption_fraction` | $\mathrm{d}$, `str`, $\mathrm{1}$ | `anthroposphere/management.yaml` | Seasonal grazing and prescribed fire defaults |
| $\tau_{\mathrm{cons}}$ | `validation.conservation.conservation_tolerance` | $\mathrm{1}$ | `validation/conservation.yaml` | Ledger tolerance |
| $s_{\mathrm{day}}, s_{\mathrm{hour}}$ | `relic.physical.constants.seconds_per_day`, `relic.physical.constants.seconds_per_hour` | $\mathrm{s\,d^{-1}}$, $\mathrm{s\,h^{-1}}$ | `physical/constants.yaml` | SI time conversion constants |

## 5. Numerical Methods and Discretization

All 20 outputs (8 public priority and control ports plus 12 `diagnostics.*` tensors) are emitted via `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`. Stateful queue records (`current_season`, `last_coordination_time`, `coordination_history`, `practice_interactions`, and serialized `EventScheduler` state) participate in replay checkpointing through `get_extra_state()` and `set_extra_state()`. Inside `set_extra_state()`, `self.current_season` is assigned immediately before validating `last_coordination_time`, so a non-finite `last_coordination_time` raises `ValueError` after mutating `current_season` unless enclosed by an outer simulator checkpoint savepoint.

## 6. Conservation, Boundary Conditions, and Ledgers

`ManagementCoordinator` does not directly mutate water, carbon, or nutrient pools; it emits control signals consumed by individual practice Functors on subsequent steps while auditing lagged biomass-removal consistency and resource-quota bounds:

$$
R_{\mathrm{mass}} = \operatorname{relu}\!\left(\max(F_{\mathrm{harv}}, 0) + \max(F_{\mathrm{graz}}, 0) - \frac{\max(B_{\mathrm{ag}}, 0)}{\Delta t} - \tau_{\mathrm{cons}}\right) + \operatorname{relu}(-F_{\mathrm{harv}}) + \operatorname{relu}(-F_{\mathrm{graz}})
$$

$$
R_{\mathrm{quota}} = \max_r \bigl(\operatorname{relu}(U_r - Q_r) + \operatorname{relu}(-U_r)\bigr)
$$

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `ManagementCoordinator` | `relic/scientific/anthroposphere/management/coordinator.py` | `ScaleCoupling` (`source_scale="grid_cell"`, `target_scale="management_domain"`, `coupling_type="spatial"`) | `ScaleCouplingFunctor` (`relic/core/functor/scale_coupling_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `ManagementCoordinator`: `[I]`, IPHS **Level 3** (`scale_coupling`, certificate `anthroposphere_management_coordinator_management_coordinator.yaml`). Five-axis priority projection with positive diagonal structural Hodge metric $H_m$, heuristic next-step control generation, and `EventScheduler` queue management; general polygon-overlap spatial conflict resolution in `EventScheduler` remains `[N]`.

Frozen certificate expressions:

```text
P_eff = P_base W_resource W_timing W_conflict.
The non-strict priority projection and positive Hodge metric expose observations and next-step controls through the declared management port.
```

## 9. References

- Dury, J., Schaller, N., Garcia, F., Reynaud, A., and Bergez, J. E. (2012). Models to support cropping plan and crop rotation decisions. A review. *Agronomy for Sustainable Development*, 32(2), 567-580.
- Keating, B. A., Carberry, P. S., Hammer, G. L., et al. (2003). An overview of APSIM, a model designed for farming systems simulation. *European Journal of Agronomy*, 18(3-4), 267-288.
- Littleboy, M., Silburn, D. M., Freebairn, D. M., Woodruff, D. R., Hammer, G. L., and Leslie, J. K. (1992). Impact of soil erosion on production in cropping systems. I. Development and validation of a simulation model. *Australian Journal of Soil Research*, 30(5), 757-774.
