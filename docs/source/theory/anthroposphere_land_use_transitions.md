# Land-Use Transitions and Effective Surface Parameter Observation

## 1. Process Definition and Scope

`LandUseTransitions` (`relic/scientific/anthroposphere/land_use/transitions.py`) advances the nine-class LUH2 fractional land-cover simplex (`primary_forest`, `secondary_forest`, `cropland_c3`, `cropland_c4`, `pasture`, `rangeland`, `urban`, `water`, `barren`) using a continuous-time Markov chain (CTMC) matrix exponential augmented by optional discrete `TransitionEvent` pulses.

`LandUseParameterMapping` (`relic/scientific/anthroposphere/land_use/mapping.py`) evaluates read-only area-weighted diagnostic observations across 25 vegetation, soil, and hydrological surface attributes from `land_use_fractions` without mutating downstream `ParameterManager` entries. Multi-pool carbon and nitrogen stock redistribution across land-use transitions (*Methods of Ecological Analysis*, Chapter 24, Section 24.4.2, Equation 24.7) remains `[N]`; `LandUseTransitions` emits only a diagnostic soil-carbon input-rate tendency. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 24 (Section 24.4).

## 2. Governing Equations

### 2.1 Continuous-Time Markov Generator and Discrete Event Increments

Base annual off-diagonal transition rates $Q^{\mathrm{base}}_{ij} \ge 0$ ($i \ne j$, $\mathrm{yr^{-1}}$) combine gross transition rates (`deforestation_rate`, `afforestation_rate`, `urbanization_rate`, `agricultural_expansion_rate`, `abandonment_rate`) with normalized destination channel weights summing to unity. Optional spatial socioeconomic fields (`gdp_growth` $G$, `population_density` $P$, `market_access` $M$, `urban_suitability` $U$) scale $Q^{\mathrm{base}}$ through a phenomenological support-mean multiplier $s'$:

$$
s = s_0 + w_g \ln\!\left(1 + \frac{\overline{G}}{G_0}\right) + w_p \ln\!\left(1 + \frac{\overline{P}}{P_0}\right) + w_m \overline{M}, \qquad
s' = s\,\max(1 - \overline{U},\; u_{\min})
$$

$$
Q_{ij} = s'\,Q^{\mathrm{base}}_{ij} \quad (i \ne j), \qquad
Q_{ii} = -\sum_{j \ne i} Q_{ij}, \qquad
H_Q = \operatorname{diag}\!\left(\sum_{j \ne i} Q_{ij}\right) \succeq 0
$$

where $H_Q$ defines the Level-3 non-negative diagonal exit-mobility audit metric. Over step $\Delta t_{\mathrm{yr}} = \Delta t / s_{\mathrm{yr}}$, the matrix-exponential increment $\Delta p^{\mathrm{ctmc}}$ and optional `TransitionEvent` transfer $\Delta p^{\mathrm{evt}}$ ($a_e = p_{\mathrm{from}}\,f_e\,m_e$ removed from `from_type` and added to `to_type` when `time_index` falls inside the active event window) advance the simplex $p \in \Delta^8$:

$$
\Delta p^{\mathrm{ctmc}} = p \exp(Q\,\Delta t_{\mathrm{yr}}) - p, \qquad
p(t + \Delta t) = p(t) + \Delta p^{\mathrm{ctmc}} + \Delta p^{\mathrm{evt}}, \qquad
\Delta I_C = \frac{1}{s_{\mathrm{yr}}}\sum_{i=1}^{9} \bigl(p_i(t + \Delta t) - p_i(t)\bigr)\,I_{C,i}
$$

where $I_{C,i}$ ($\mathrm{kg\,C\,m^{-2}\,yr^{-1}}$) is class soil-carbon input capacity and $\Delta I_C$ ($\mathrm{kg\,C\,m^{-2}\,s^{-1}}$) is diagnostic rate change.

### 2.2 Read-Only Effective Surface Parameter Observation

Given simplex fractions $f_i \ge 0$ with $\sum_{i=1}^{9} f_i = 1$ and class attribute table $\theta_{i,m}$, `LandUseParameterMapping` forms the convex observation

$$
\bar{\theta}_m = \sum_{i=1}^{9} f_i\,\theta_{i,m}
$$

across 10 vegetation, 8 soil, and 7 hydrological diagnostic fields (`diagnostics.land_use_parameter_mapping.{vegetation,soil,hydrology}.*`).

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $p_i, f_i$ | Nine-class LUH2 fractional land-use simplex (`[..., 9]` or `[9, ...]`) | $\mathrm{1}$ | `land_use_fractions` | `state` |
| $G, P, M, U$ | Optional GDP growth, population density, market access, and urban suitability drivers | $\mathrm{1}$, $\mathrm{km^{-2}}$, $\mathrm{1}$, $\mathrm{1}$ | `gdp_growth`, `population_density`, `market_access`, `urban_suitability` | `forcing` |
| $A_{\mathrm{grid}}$ | Optional grid-cell area for area-weighted conservation audits | $\mathrm{m^2}$ | `grid_area` | `forcing` |
| $\Delta I_C$ | Diagnostic soil-carbon input-rate change from fraction shifts | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `diagnostics.land_use_transitions.soil_carbon_input_rate_change` | `diagnostic` |
| $\mathrm{tr}(H_Q), \epsilon_{\mathrm{area}}$ | Mobility metric trace and simplex conservation residual | $\mathrm{yr^{-1}}$, $\mathrm{1}$ | `diagnostics.land_use_transitions.mobility_metric_trace`, `diagnostics.land_use_transitions.area_conservation_error` | `diagnostic` |
| $\bar{\theta}_m$ | Area-weighted vegetation, soil, and hydrology parameter observations (25 fields) | class-specific SI | `diagnostics.land_use_parameter_mapping.*` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $r_{\mathrm{def}}, r_{\mathrm{aff}}, r_{\mathrm{urb}}, r_{\mathrm{ag}}, r_{\mathrm{abn}}$ | `anthroposphere.land_use.transition_rates.*` | $\mathrm{yr^{-1}}$ | `anthroposphere/land_use.yaml` | Hurtt et al. (2020) |
| $c_{ij}$ | `anthroposphere.land_use.transition_channels.*` | $\mathrm{1}$ | `anthroposphere/land_use.yaml` | Hurtt et al. (2020) |
| $w_g, w_p, w_m, G_0, P_0, u_{\min}, s_0$ | `anthroposphere.land_use.numerical.driver_*`, `anthroposphere.land_use.numerical.unity_factor` | $\mathrm{1}$, $\mathrm{km^{-2}}$ | `anthroposphere/land_use.yaml` | Phenomenological driver scaling |
| $\theta_{i,\mathrm{veg}}, I_{C,i}$ | `anthroposphere.land_use.vegetation.*`, `anthroposphere.land_use.soil.*`, `anthroposphere.land_use.hydrological.*` | SI units | `anthroposphere/land_use.yaml` | Lawrence et al. (2016); Hurtt et al. (2020) |
| $\tau_{\mathrm{cons}}$ | `validation.conservation.conservation_tolerance` | $\mathrm{1}$ | `validation/conservation.yaml` | Simplex conservation tolerance |
| $s_{\mathrm{yr}}$ | `relic.physical.constants.seconds_per_year` | $\mathrm{s\,yr^{-1}}$ | `physical/constants.yaml` | Julian year seconds |

## 5. Numerical Methods and Discretization

`LandUseTransitions` evaluates $\exp(Q\,\Delta t_{\mathrm{yr}})$ via `torch.matrix_exp` on arbitrary layout permutations of the 9-class axis and writes updated `land_use_fractions` (`StateWriteMode.STATE`) plus six `diagnostics.land_use_transitions.*` fields through `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`. `TransitionEvent` evaluation is stateless across calls; omitting `time_index` or repeating the same `time_index` reapplies active events because no executed-occurrence ledger is persisted. Explicit spatial remapping (`upscale`, `downscale`) requires bound `upscale_bridge` ($R_h$) and `downscale_bridge` ($P_h$) callables. `LandUseParameterMapping` writes 26 `UpdateItem(UpdateMode.REPLACE, per_timestep=False)` diagnostic entries and requires explicit `field_query_bridge` and `field_gradient_bridge` callables for continuous coordinate queries.

## 6. Conservation, Boundary Conditions, and Ledgers

Row-zero generator construction $\sum_j Q_{ij} = 0$ with $Q_{ij} \ge 0$ ($i \ne j$) guarantees stochastic semigroup preservation $p \exp(Q\,\Delta t_{\mathrm{yr}}) \ge 0$ and $\sum_i (p \exp(Q\,\Delta t_{\mathrm{yr}}))_i = \sum_i p_i = 1$ to floating-point precision, while paired event transfers preserve $\sum_i \Delta p^{\mathrm{evt}}_i = 0$. Both Functors verify non-negativity and $\max\lvert \sum_i p_i - 1\rvert \le \tau_{\mathrm{cons}}$ on every step.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `LandUseTransitions` | `relic/scientific/anthroposphere/land_use/transitions.py` | `ScaleCoupling` (`source_scale="fine_grid"`, `target_scale="coarse_grid"`, `coupling_type="spatial"`) | `ScaleCouplingFunctor` (`relic/core/functor/scale_coupling_bridge.py`) | `ExecutionPhase.PREPROCESSING` (`SpatialScale.GRID_CELL`) |
| `LandUseParameterMapping` | `relic/scientific/anthroposphere/land_use/mapping.py` | `PhysicalField` (`SPATIAL_DIMS=2`) | `PhysicalFieldFunctor` (`relic/core/functor/physical_field_bridge.py`) | `ExecutionPhase.PREPROCESSING` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `LandUseTransitions`: `[I]`, IPHS **Level 3** (`scale_coupling`, certificate `anthroposphere_land_use_transitions_land_use_transitions.yaml`). Nine-class CTMC matrix-exponential simplex evolution with non-negative exit-mobility metric $H_Q$; cross-tile biogeochemical C/N stock redistribution remains `[N]`.
- `LandUseParameterMapping`: `[I]`, IPHS **Level 1** (`observation`, certificate `anthroposphere_land_use_mapping_land_use_parameter_mapping.yaml`). Read-only convex diagnostic observation over 25 surface attributes.

Frozen certificate expressions:

```text
Historical public aliases and area/modifier outputs are retained only as equivalence evidence, not runtime ports.
The single CTMC transition path uses H_Q = diag(sum_{j != i} Q_ij) as a non-negative mobility audit metric.
theta_bar = sum_i f_i theta_i, with f_i >= 0 and sum_i f_i = 1.
The process is only IPHS-compatible as a read-only observation; no Hodge, Hamiltonian, dissipation operator, or power port is asserted.
```

## 9. References

- Arneth, A., Sitch, S., Pongratz, J., et al. (2017). Historical carbon dioxide emissions caused by land-use changes are possibly larger than assumed. *Nature Geoscience*, 10(2), 79-84.
- Hurtt, G. C., Chini, L., Sahajpal, R., et al. (2020). Harmonization of global land use change and management for the period 850-2100 (LUH2) for CMIP6. *Geoscientific Model Development*, 13(11), 5425-5464.
- Lawrence, D. M., Hurtt, G. C., Arneth, A., et al. (2016). The Land Use Model Intercomparison Project (LUMIP) contribution to CMIP6: rationale and experimental design. *Geoscientific Model Development*, 9(9), 2973-2998.
