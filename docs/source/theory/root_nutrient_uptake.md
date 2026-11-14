# Root Nutrient Uptake

## 1. Process Definition and Scope

`RootNutrientUptake` (`relic/scientific/pedosphere/nutrient_cycling/root_nutrient_uptake.py`) evaluates plant root acquisition of soil nitrate ($\mathrm{NO_3^-}$), ammonium ($\mathrm{NH_4^+}$), and available phosphate ($\mathrm{PO_4^{3-}}$) across layered or scalar soil profiles, bounded by plant demand, available soil stocks, and remaining plant internal storage capacity.

All 13 declared inputs are mandatory, finite canonical `StateManager` ports; historical `no3_concentration`, `nh4_concentration`, `temperature`, and `root_biomass` aliases are rejected. Mycorrhizal symbiotic uptake pathways, rhizosphere exudate solubilization, and active electrochemical membrane potential coupling remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapters 21 and 23.

## 2. Governing Equations

### 2.1 Soil Bulk Concentration Basis and Root Profile Weights

Canonical bulk-soil molar concentrations ($\mathrm{mol\,m^{-3}}$) are derived directly from areal elemental mass pools ($m_{\mathrm{NO_3}}, m_{\mathrm{NH_4}}, P_{\mathrm{avail}}$ in $\mathrm{g\,element\,m^{-2}}$), layer thickness $z$ (`soil_depth`, $\mathrm{m}$), and elemental molar masses $M_N, M_P$ ($\mathrm{g\,mol^{-1}}$):

$$
C_{\mathrm{NO_3}}(z) = \frac{m_{\mathrm{NO_3}}(z)}{M_N z}, \qquad
C_{\mathrm{NH_4}}(z) = \frac{m_{\mathrm{NH_4}}(z)}{M_N z}, \qquad
C_{\mathrm{PO_4}}(z) = \frac{P_{\mathrm{avail}}(z)}{M_P z}
$$

Normalized root weights $W_{\mathrm{root}}(z)$ sum to unity over the layer axis from positive `root_density` inputs (or exponential depth weighting $\exp(-k_{\mathrm{root}} z_{\mathrm{cum}})$ when `root_density` carries uniform profile geometry).

### 2.2 Potential Michaelis-Menten Kinetics and Environmental Modifiers

For each nutrient ion $i \in \{\mathrm{NO_3^-}, \mathrm{NH_4^+}, \mathrm{PO_4^{3-}}\}$, potential layer uptake ($\mathrm{mol\,m^{-2}\,s^{-1}}$) follows Barber (1995) and Tinker and Nye (2000):

$$
U_i(z) = V_{\max,i}\frac{C_i(z)}{K_{m,i} + C_i(z)}\,f_T(z)\,f_W(z)\,W_{\mathrm{root}}(z), \qquad
U_{i,\mathrm{area}} = \sum_z U_i(z)
$$

Soil temperature $T_s$ ($\mathrm{K}$) is clamped to $[T_{\min}, T_{\max}]$ before evaluating $f_T = Q_{10}^{(T_s - T_{\mathrm{ref}})/\Delta T_{10}}$. When `water_stress_enabled` is active, $f_W = \operatorname{clip}\bigl((\theta - \theta_{\mathrm{wp}})/(\theta_{\mathrm{fc}} - \theta_{\mathrm{wp}}),\, 0,\, 1\bigr)$; otherwise $f_W = 1$.

### 2.3 Demand, Pool-Availability, and Storage-Capacity Bounding

When `demand_regulation_enabled` is active, total kinetic nitrogen and phosphorus rates are first downscaled to satisfy `plant_n_demand` ($\mathrm{mol\,N\,m^{-2}\,s^{-1}}$) and `plant_p_demand` ($\mathrm{g\,P\,m^{-2}\,s^{-1}}$, converted by $M_P$). Layer rates are next bounded by available donor mass over $\Delta t$ and shared remaining plant storage capacity ($S_{N,\max} - S_N$ in $\mathrm{mol\,N\,m^{-2}}$; $S_{P,\max} - P_{\mathrm{plant}}/M_P$ in $\mathrm{mol\,P\,m^{-2}}$). For phosphorus:

$$
U_P = \min\!\left(U_{P,\mathrm{kin}},\; \frac{D_P}{M_P},\; \frac{P_{\mathrm{avail}}}{M_P \Delta t},\; \frac{\max(S_{P,\max} - P_{\mathrm{plant}}/M_P,\, 0)}{\Delta t}\right)
$$

### 2.4 Reaction-Network State Increments

Over step $\Delta t$, donor soil pools and plant storage pools update via:

$$
\Delta\mathrm{pool}_i(z) = -U_i(z)\,M_i\,\Delta t, \qquad
\Delta\mathrm{pool}_i = \sum_z \Delta\mathrm{pool}_i(z) = -U_{i,\mathrm{area}}\,M_i\,\Delta t
$$

$$
\Delta m_{\mathrm{NO_3}} = -M_N U_{\mathrm{NO_3}}\Delta t, \qquad
\Delta m_{\mathrm{NH_4}} = -M_N U_{\mathrm{NH_4}}\Delta t, \qquad
\Delta S_N = (U_{\mathrm{NO_3}} + U_{\mathrm{NH_4}})\Delta t
$$

$$
\Delta P_{\mathrm{avail}} = -M_P U_P \Delta t, \qquad
\Delta P_{\mathrm{plant}} = +M_P U_P \Delta t
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $m_{\mathrm{NO_3}}$ | Soil nitrate nitrogen pool (input stock and per-step increment) | $\mathrm{g\,N\,m^{-2}}$ | `no3_nitrogen` | `state` |
| $m_{\mathrm{NH_4}}$ | Soil ammonium nitrogen pool (input stock and per-step increment) | $\mathrm{g\,N\,m^{-2}}$ | `nh4_nitrogen` | `state` |
| $P_{\mathrm{avail}}$ | Soil available phosphorus pool (input stock and per-step increment) | $\mathrm{g\,P\,m^{-2}}$ | `available_phosphorus` | `state` |
| $S_N$ | Plant internal nitrogen storage pool (input stock and per-step increment) | $\mathrm{mol\,N\,m^{-2}}$ | `plant_n_storage` | `state` |
| $P_{\mathrm{plant}}$ | Plant internal phosphorus storage pool (input stock and per-step increment) | $\mathrm{g\,P\,m^{-2}}$ | `plant_p_storage` | `state` |
| $\theta, \theta_{\mathrm{fc}}, \theta_{\mathrm{wp}}$ | Volumetric soil water content, field capacity, and wilting point | $\mathrm{m^3\,m^{-3}}$ | `soil_water_content`, `field_capacity`, `wilting_point` | `state` |
| $T_s$ | Soil layer temperature | $\mathrm{K}$ | `soil_temperature` | `state` |
| $z$ | Soil layer thickness | $\mathrm{m}$ | `soil_depth` | `state` |
| $\rho_{\mathrm{root}}$ | Root density profile | $\mathrm{kg\,m^{-3}}$ | `root_density` | `state` |
| $D_N$ | Plant nitrogen uptake demand | $\mathrm{mol\,N\,m^{-2}\,s^{-1}}$ | `plant_n_demand` | `flux` |
| $D_P$ | Plant phosphorus uptake demand | $\mathrm{g\,P\,m^{-2}\,s^{-1}}$ | `plant_p_demand` | `flux` |
| $U_N$ | Realized area-integrated molar nitrogen uptake rate | $\mathrm{mol\,N\,m^{-2}\,s^{-1}}$ | `nitrogen_uptake_rate` | `flux` |
| $F_P$ | Realized area-integrated phosphorus mass uptake rate ($M_P U_P$) | $\mathrm{g\,P\,m^{-2}\,s^{-1}}$ | `phosphorus_uptake_rate` | `flux` |
| $U_{\mathrm{NO_3}}(z), U_{\mathrm{NH_4}}(z)$ | Layer-resolved molar nitrate and ammonium uptake rates | $\mathrm{mol\,N\,m^{-2}\,s^{-1}}$ | `no3_uptake_by_layer`, `nh4_uptake_by_layer` | `diagnostic` |
| $F_{\mathrm{PO_4}}(z)$ | Layer-resolved phosphorus mass uptake rate | $\mathrm{g\,P\,m^{-2}\,s^{-1}}$ | `po4_uptake_by_layer` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $V_{\max,N}, V_{\max,P}$ | `pedosphere.nutrient_cycling.vmax_n`, `pedosphere.nutrient_cycling.vmax_p` | $\mathrm{mol\,m^{-2}\,s^{-1}}$ | `pedosphere/nutrient_cycling.yaml` | Barber (1995); Griffiths and York (2020) |
| $K_{m,N}, K_{m,P}$ | `pedosphere.nutrient_cycling.km_n`, `pedosphere.nutrient_cycling.km_p` | $\mathrm{mol\,m^{-3}}$ | `pedosphere/nutrient_cycling.yaml` | Barber (1995); Tinker and Nye (2000) |
| $S_{N,\max}, S_{P,\max}$ | `pedosphere.nutrient_cycling.max_storage_n`, `pedosphere.nutrient_cycling.max_storage_p` | $\mathrm{mol\,m^{-2}}$ | `pedosphere/nutrient_cycling.yaml` | Chapin et al. (1993) |
| $k_{\mathrm{root}}$ | `pedosphere.nutrient_cycling.root_decay_coeff` | $\mathrm{m^{-1}}$ | `pedosphere/nutrient_cycling.yaml` | Tinker and Nye (2000) |
| $Q_{10}, T_{\mathrm{ref}}$ | `pedosphere.nutrient_cycling.q10_value`, `pedosphere.nutrient_cycling.reference_temperature` | $\mathrm{1}$, $\mathrm{K}$ | `pedosphere/nutrient_cycling.yaml` | von Bloh et al. (2018) |
| $b_{\mathrm{dem}}, b_W$ | `pedosphere.nutrient_cycling.demand_regulation_enabled`, `pedosphere.nutrient_cycling.water_stress_enabled` | `bool` | `pedosphere/nutrient_cycling.yaml` | von Bloh et al. (2018) |
| $T_{\min}, T_{\max}$ | `pedosphere.numerical.temperature_clamp_min`, `pedosphere.numerical.temperature_clamp_max` | $\mathrm{K}$ | `pedosphere/numerical.yaml` | RELIC numerical bounds |
| $b_{\mathrm{cons}}$ | `validation.conservation.strict_conservation` | `bool` | `validation/conservation.yaml` | RELIC conservation contract |
| $M_N, M_P, \Delta T_{10}$ | `relic.physical.constants.molecular_weights.nitrogen`, `relic.physical.constants.molecular_weights.phosphorus`, `relic.physical.constants.temperature.q10_interval` | $\mathrm{g\,mol^{-1}}$, $\mathrm{g\,mol^{-1}}$, $\mathrm{K}$ | `physical/constants.yaml` | CIAAW / NIST SRD 69 |

## 5. Numerical Methods and Discretization

Demand, donor-pool, and storage-capacity rate bounds are evaluated algebraically prior to a single `advance_reaction_network` call over explicit step $\Delta t$. Layered donor pools retain their vertical layer dimension in the returned `UpdateItem(ACCUMULATE, per_timestep=True)` tensors while plant storage pools receive area-summed increments. No `delta_*` keys are returned, preventing `StateManager` from multiplying integrated increments by $\Delta t$ a second time.

## 6. Conservation, Boundary Conditions, and Ledgers

The reaction network comprises three signed transfer channels ($\mathrm{NO_3^-}$ to plant N, $\mathrm{NH_4^+}$ to plant N, and available soil P to plant P) with two exact closed conservation rows over the summed profile:

$$
\frac{\Delta m_{\mathrm{NO_3}} + \Delta m_{\mathrm{NH_4}}}{M_N} + \Delta S_N = 0, \qquad
\Delta P_{\mathrm{avail}} + \Delta P_{\mathrm{plant}} = 0
$$

Nitrogen states declare `element="N"` (`accumulates: nitrogen.total`) and phosphorus states declare `element="P"` (`accumulates: phosphorus.total`); `REPLACE` rate and layer-profile outputs carry no stock-ledger `element` tag.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `RootNutrientUptake` | `relic/scientific/pedosphere/nutrient_cycling/root_nutrient_uptake.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `RootNutrientUptake`: `[I]`, IPHS **Level 2** (`reaction`, certificate `pedosphere_nutrient_cycling_root_nutrient_uptake_root_nutrient_uptake.yaml`). Bounded Michaelis-Menten uptake rates lower through `advance_reaction_network` with exact elemental N and P mass-conservation rows; chemical-potential effort pairing is not certified.
- Mycorrhizal uptake and active root exudation couplings: `[N]`.

Frozen audit and certificate expressions:

```text
U_i(z) = Vmax_i * C_i(z) / (Km_i + C_i(z)) * f_T * f_W * W_root(z)
Delta pool_i = - U_i_area * M_i * dt
U = Vmax * C/(Km + C) * f_T * f_W * W_root
x_dot = N_h r_h with ParameterManager-backed process rates r_h and declared stoichiometric/output map N_h
```

## 9. References

- Barber, S. A. (1995). *Soil Nutrient Bioavailability: A Mechanistic Approach* (2nd ed.). Wiley.
- Chapin, F. S., Autumn, K., and Pugnaire, F. (1993). Evolution of suites of traits in response to environmental stress. *American Naturalist*, 142, S78-S92.
- Griffiths, M., and York, L. M. (2020). Targeting root ion uptake kinetics to increase plant productivity and nutrient use efficiency. *Plant Physiology*, 182(4), 1854-1868.
- Tinker, P. B., and Nye, P. H. (2000). *Solute Movement in the Rhizosphere*. Oxford University Press.
- von Bloh, W., Schaphoff, S., Muller, C., Rolinski, S., Waha, K., and Zaehle, S. (2018). Implementing the nitrogen cycle into the dynamic global vegetation, hydrology, and crop growth model LPJmL (version 5.0). *Geoscientific Model Development*, 11, 2789-2812.
