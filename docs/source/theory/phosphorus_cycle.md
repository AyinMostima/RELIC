# Pedosphere Phosphorus Cycling

## 1. Process Definition and Scope

Phosphorus cycling in the live RELIC pedosphere codebase has no standalone `PhosphorusCycleFunctor`. Executable phosphorus dynamics are limited to two cross-process channels (`[I]`):

1. `RootNutrientUptake` (`relic/scientific/pedosphere/nutrient_cycling/root_nutrient_uptake.py`, `[I]`, certified under `root_nutrient_uptake.md`): computes layer-resolved Michaelis-Menten orthophosphate ($\mathrm{PO_4^{3-}}\text{-}\mathrm{P}$) root uptake, enforces plant demand, donor-pool availability, and plant storage-capacity bounds, and transfers elemental phosphorus mass conservatively from `available_phosphorus` to `plant_p_storage`.
2. `PedosphereInterfaceMapper` (`relic/scientific/pedosphere/interface_mapping.py`, `[I]`, certified under `pedosphere_transport.md`): registers `pedosphere.phosphorus.p_po4_fraction` and `pedosphere.phosphorus.p_dust_fraction` to split atmospheric wet and dry phosphorus deposition into solution orthophosphate (`available_phosphorus`) and mineral-associated (`inorganic_phosphorus`) pools inside its auxiliary `map_atmosphere_to_pedosphere` utility.

Standalone soil phosphorus transformation Functors for organic phosphorus mineralization, microbial phosphorus immobilization, reversible Langmuir or Freundlich sorption-desorption, primary mineral weathering, iron/aluminum-oxide occlusion, and dissolved phosphorus leaching are `[N]` (pre-parameterized in `relic/parameters/database/relic/pedosphere/phosphorus.yaml` without a scheduled `FunctorBase` runtime owner). Neither `NitrogenCycling` (`nitrogen.py`), nor `StoichiometryConstraint` (`stoichiometry.py`, which audits carbon-to-nitrogen ratios only), nor `NumericalStabilityController` (`mass_conservation.py`) executes phosphorus pool updates or domain-specific phosphorus audits; global elemental phosphorus ledger verification belongs to `StateManager` and `ConservationGuardian` via `element="P"` (`phosphorus.total`).

## 2. Governing Equations

### 2.1 Implemented Root Orthophosphate Uptake (`[I]`)

In `RootNutrientUptake`, the canonical layer-resolved or scalar plant-available phosphorus pool $P_{\mathrm{avail},i}$ ($\mathrm{g\,P\,m^{-2}}$) in soil layer $i$ of thickness $z_i > 0$ ($\mathrm{m}$) is converted to molar concentration $C_{\mathrm{PO_4},i}$ ($\mathrm{mol\,P\,m^{-3}}$) using the phosphorus atomic weight $M_P$ (`relic.physical.constants.molecular_weights.phosphorus`, Barber, 1995):

$$
C_{\mathrm{PO_4},i} = \frac{P_{\mathrm{avail},i}}{M_P z_i}
$$

Potential molar uptake rate in layer $i$ ($\mathrm{mol\,P\,m^{-2}\,s^{-1}}$) follows Michaelis-Menten kinetics modulated by temperature factor $f_T(T)$, water-stress factor $f_W(\theta_i)$, and normalized root-biomass weight $W_{\mathrm{root},i}$ ($\sum_i W_{\mathrm{root},i} = 1$, Jackson et al., 1996; Tinker and Nye, 2000):

$$
U_{P,\mathrm{pot},i} = V_{\max,P} \frac{C_{\mathrm{PO_4},i}}{K_{m,P} + C_{\mathrm{PO_4},i}} f_T(T) f_W(\theta_i) W_{\mathrm{root},i}, \qquad f_T(T) = Q_{10}^{(T - T_{\mathrm{ref}}) / \Delta T_{10}}
$$

When `water_stress_enabled` is active, $f_W(\theta_i) = \operatorname{clip}((\theta_i - \theta_{\mathrm{wp},i}) / (\theta_{\mathrm{fc},i} - \theta_{\mathrm{wp},i}), 0, 1)$ (fixing $f_W = 1$ when disabled). Summing over layers gives $U_{P,\mathrm{pot}} = \sum_i U_{P,\mathrm{pot},i}$, which is regulated by plant molar demand $D_P$ when `demand_regulation_enabled` is active:

$$
U_{P,\mathrm{req}} = \begin{cases} \min\!\left(U_{P,\mathrm{pot}}, D_P\right), & \text{demand regulation enabled} \\ U_{P,\mathrm{pot}}, & \text{demand regulation disabled} \end{cases}
$$

When `PedosphereInterfaceMapper.map_atmosphere_to_pedosphere` processes annual wet and dry phosphorus deposition fluxes ($F_{\mathrm{dep,wet},P}, F_{\mathrm{dep,dry},P}$ in $\mathrm{g\,P\,m^{-2}\,yr^{-1}}$), conversion by $t_{\mathrm{yr}}$ (`seconds_per_year`) yields instantaneous inputs partitioned by $f_{\mathrm{PO_4}}$ and $f_{\mathrm{dust}}$:

$$
F_{\mathrm{avail},P}^{\mathrm{dep}} = f_{\mathrm{PO_4}} \frac{F_{\mathrm{dep,wet},P}}{t_{\mathrm{yr}}}, \qquad F_{\mathrm{inorg},P}^{\mathrm{dep}} = \left(1 - f_{\mathrm{PO_4}}\right)\frac{F_{\mathrm{dep,wet},P}}{t_{\mathrm{yr}}} + f_{\mathrm{dust}} \frac{F_{\mathrm{dep,dry},P}}{t_{\mathrm{yr}}}
$$

### 2.2 Unimplemented Soil Phosphorus Transformations (`[N]`)

The full multi-pool soil phosphorus balance parameterized in `pedosphere/phosphorus.yaml` (Hedley et al., 1982; McGill and Cole, 1981; Parton et al., 1988; Wang et al., 2010),

$$
\frac{dP_{\mathrm{avail}}}{dt} = F_{\mathrm{min},P} + F_{\mathrm{weather},P} + F_{\mathrm{desorb},P} + F_{\mathrm{dep},P} - F_{\mathrm{immob},P} - F_{\mathrm{sorb},P} - F_{\mathrm{leach},P} - M_P U_P
$$

including phosphatase-mediated organic-P mineralization $F_{\mathrm{min},P}$, microbial immobilization $F_{\mathrm{immob},P}$, Langmuir/Freundlich mineral sorption-desorption ($F_{\mathrm{sorb},P}, F_{\mathrm{desorb},P}$), primary apatite weathering $F_{\mathrm{weather},P}$, Fe/Al-oxide occlusion, and dissolved P leaching $F_{\mathrm{leach},P}$, is `[N]` in the live Functor registry; only the root-uptake sink $-M_P U_P$ (`[I]`) and auxiliary deposition helper (`[I]`) exist in executable Python modules.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $P_{\mathrm{avail}}$ | Plant-available soil phosphorus pool | $\mathrm{g\,P\,m^{-2}}$ | `available_phosphorus` | `RootNutrientUptake` required input and `ACCUMULATE` output (`per_timestep=True`) |
| $S_{\mathrm{plant},P}$ | Plant internal phosphorus mass storage | $\mathrm{g\,P\,m^{-2}}$ | `plant_p_storage` | `RootNutrientUptake` required input and `ACCUMULATE` output (`per_timestep=True`) |
| $D_P$ | Plant molar phosphorus uptake demand | $\mathrm{mol\,P\,m^{-2}\,s^{-1}}$ | `plant_p_demand` | `RootNutrientUptake` required input |
| $z$ | Soil layer thickness / depth vector | $\mathrm{m}$ | `soil_depth` | `RootNutrientUptake` required input |
| $\rho_{\mathrm{root}}$ | Root biomass density by layer | $\mathrm{kg\,m^{-3}}$ | `root_density` | `RootNutrientUptake` required input |
| $\theta, \theta_{\mathrm{fc}}, \theta_{\mathrm{wp}}$ | Soil water content, field capacity, and wilting point | $\mathrm{m^3\,m^{-3}}$ | `soil_water_content`, `field_capacity`, `wilting_point` | `RootNutrientUptake` required inputs |
| $T$ | Soil temperature | $\mathrm{K}$ | `soil_temperature` | `RootNutrientUptake` required input |
| $F_{\mathrm{up},P}$ | Realized areal phosphorus mass uptake rate $M_P U_P$ | $\mathrm{g\,P\,m^{-2}\,s^{-1}}$ | `phosphorus_uptake_rate` | `RootNutrientUptake` flux output (`REPLACE`, `per_timestep=False`) |
| $F_{\mathrm{up,layer},P}$ | Layer-resolved orthophosphate mass uptake rate | $\mathrm{g\,P\,m^{-2}\,s^{-1}}$ | `po4_uptake_by_layer` | `RootNutrientUptake` diagnostic output (`REPLACE`, `per_timestep=False`) |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $V_{\max,P}$ | `pedosphere.nutrient_cycling.vmax_p` | $\mathrm{mol\,P\,m^{-2}\,s^{-1}}$ | `pedosphere/nutrient_cycling.yaml` | Barber (1995) |
| $K_{m,P}$ | `pedosphere.nutrient_cycling.km_p` | $\mathrm{mol\,P\,m^{-3}}$ | `pedosphere/nutrient_cycling.yaml` | Barber (1995); Tinker and Nye (2000) |
| $S_{P,\max}$ | `pedosphere.nutrient_cycling.max_storage_p` | $\mathrm{mol\,P\,m^{-2}}$ | `pedosphere/nutrient_cycling.yaml` ($1.0$) | Chapin et al. (1993) |
| $\beta_{\mathrm{root}}$ | `pedosphere.nutrient_cycling.root_decay_coeff` | $\mathrm{m^{-1}}$ | `pedosphere/nutrient_cycling.yaml` | Jackson et al. (1996) |
| $Q_{10}, T_{\mathrm{ref}}$ | `pedosphere.nutrient_cycling.q10_value`, `pedosphere.nutrient_cycling.reference_temperature` | $\mathrm{1}$, $\mathrm{K}$ | `pedosphere/nutrient_cycling.yaml` ($2.0$, $298.15$) | Barber (1995) |
| $f_{\mathrm{ws,on}}, f_{\mathrm{dem,on}}$ | `pedosphere.nutrient_cycling.water_stress_enabled`, `pedosphere.nutrient_cycling.demand_regulation_enabled` | flag | `pedosphere/nutrient_cycling.yaml` | Process configuration switches |
| $f_{\mathrm{PO_4}}, f_{\mathrm{dust}}$ | `pedosphere.phosphorus.p_po4_fraction`, `pedosphere.phosphorus.p_dust_fraction` | $\mathrm{1}$ | `pedosphere/phosphorus.yaml` ($0.3$, $0.7$) | Wang et al. (2010) |
| $P_{\mathrm{avail},0}, k_{\mathrm{min},P}, Q_{10,\mathrm{min},P}$ | `pedosphere.phosphorus.available_p_initial`, `pedosphere.phosphorus.mineralization_rate_constant`, `pedosphere.phosphorus.mineralization_q10` | $\mathrm{g\,m^{-2}}$, $\mathrm{s^{-1}}$, $\mathrm{1}$ | `pedosphere/phosphorus.yaml` ($5.0$, $3.17 \times 10^{-8}$, $2.5$; `[N]` database-only) | Hedley et al. (1982); McGill and Cole (1981) |
| $K_{d,P}, q_{\max,P}, b_P, k_{\mathrm{des},P}$ | `pedosphere.phosphorus.sorption_kd_linear`, `pedosphere.phosphorus.sorption_langmuir_qmax`, `pedosphere.phosphorus.sorption_langmuir_b`, `pedosphere.phosphorus.desorption_rate` | $\mathrm{1}$, $\mathrm{g\,m^{-2}}$, $\mathrm{1}$, $\mathrm{s^{-1}}$ | `pedosphere/phosphorus.yaml` ($0.5$, $20.0$, $0.1$, $10^{-6}$; `[N]` database-only) | Barrow (1983); Frossard et al. (2000) |
| $k_{\mathrm{weath},P}, k_{\mathrm{occ},P}$ | `pedosphere.phosphorus.weathering_base_rate`, `pedosphere.phosphorus.occlusion_rate_constant` | $\mathrm{s^{-1}}$ | `pedosphere/phosphorus.yaml` ($3.17 \times 10^{-10}$, $10^{-9}$; `[N]` database-only) | Guidry and Mackenzie (2003); Cross and Schlesinger (1995) |
| $M_P, \Delta T_{10}$ | `relic.physical.constants.molecular_weights.phosphorus`, `relic.physical.constants.temperature.q10_interval` | $\mathrm{g\,P\,mol^{-1}}$, $\mathrm{K}$ | `physical/constants.yaml` ($30.97376$, $10.0$) | Physical constants SSOT |

## 5. Numerical Methods and Discretization

Over a finite step $\Delta t > 0$, `RootNutrientUptake.forward` bounds the requested areal molar phosphorus uptake rate $U_{P,\mathrm{req}}$ simultaneously by total available soil phosphorus $P_{\mathrm{avail,tot}} = \sum_i P_{\mathrm{avail},i}$ and remaining plant molar storage capacity $C_{P,\mathrm{rem}} = \max(S_{P,\max} - S_{\mathrm{plant},P} / M_P, 0)$:

$$
U_P = \min\!\left(U_{P,\mathrm{req}},\, \frac{P_{\mathrm{avail,tot}}}{M_P \Delta t},\, \frac{C_{P,\mathrm{rem}}}{\Delta t}\right)
$$

(for $\Delta t = 0$, $U_P = U_{P,\mathrm{req}}$ and step increments vanish). Layer uptake proportions are scaled consistently so $\sum_i U_{P,i} = U_P$, converted to elemental mass fluxes $F_{\mathrm{up},P} = M_P U_P$ and $F_{\mathrm{up,layer},P,i} = M_P U_{P,i}$, and passed through `advance_reaction_network` with stoichiometric vector $N_{h,P} = (-1, +1)^{\mathsf{T}}$:

$$
\Delta P_{\mathrm{avail},i} = -M_P U_{P,i} \Delta t, \qquad \Delta S_{\mathrm{plant},P} = +M_P U_P \Delta t
$$

Both state increments are emitted as `UpdateItem(mode=UpdateMode.ACCUMULATE, per_timestep=True, unit="g P m-2", element="P")`. Missing, non-finite, or negative inputs raise `ValueError` without fallback aliases such as `po4_concentration`.

## 6. Conservation, Boundary Conditions, and Ledgers

Within `RootNutrientUptake`, phosphorus uptake is a closed soil-to-plant internal transfer satisfying exact discrete elemental mass conservation across every step $\Delta t$:

$$
\sum_i \Delta P_{\mathrm{avail},i} + \Delta S_{\mathrm{plant},P} = -M_P \left(\sum_i U_{P,i}\right) \Delta t + M_P U_P \Delta t = 0
$$

Both `available_phosphorus` and `plant_p_storage` accumulate into the `phosphorus.total` conservation domain in `StateManager`. The instantaneous flux outputs `phosphorus_uptake_rate` and `po4_uptake_by_layer` use `UpdateMode.REPLACE` (`per_timestep=False`) without an `element` ledger tag so uptake mass is counted once.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `RootNutrientUptake` | `relic/scientific/pedosphere/nutrient_cycling/root_nutrient_uptake.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` |
| `PedosphereInterfaceMapper` | `relic/scientific/pedosphere/interface_mapping.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.COUPLING` |

## 8. Implementation Status

- **Orthophosphate root uptake and soil-to-plant P mass transfer (`RootNutrientUptake`)**: `[I]` (certified at IPHS Level 2 under `root_nutrient_uptake.md`; zero IPHS equivalence certificates carry `theory_document: docs\source\theory\phosphorus_cycle.md`).
- **Atmospheric phosphorus deposition speciation parameters (`PedosphereInterfaceMapper`)**: `[I]` (parameters registered on `PedosphereInterfaceMapper`, certified at IPHS Level 2 under `pedosphere_transport.md`; deposition splitting resides in the auxiliary method `map_atmosphere_to_pedosphere`).
- **Standalone soil phosphorus mineralization, microbial P immobilization, sorption-desorption, mineral weathering, Fe/Al-oxide occlusion, and dissolved-P leaching Functors**: `[N]` (parameters exist in `relic/parameters/database/relic/pedosphere/phosphorus.yaml`, while no live `FunctorBase` subclass executes these soil P pool transformations).

## 9. References

1. Barber, S. A. (1995). *Soil Nutrient Bioavailability: A Mechanistic Approach* (2nd ed.). John Wiley and Sons.
2. Barrow, N. J. (1983). A mechanistic model for describing the sorption and desorption of phosphate by soil. *Journal of Soil Science*, 34(4), 733-750.
3. Chapin, F. S., Autumn, K., and Pugnaire, F. (1993). Evolution of suites of traits in response to environmental stress. *The American Naturalist*, 142, S78-S92.
4. Cleveland, C. C., and Liptzin, D. (2007). C:N:P stoichiometry in soil: Is there a "Redfield ratio" for the microbial biomass? *Biogeochemistry*, 85(3), 235-252. doi:10.1007/s10533-007-9132-0
5. Cross, A. F., and Schlesinger, W. H. (1995). A literature review and evaluation of the Hedley fractionation: Applications to the biogeochemical cycle of soil phosphorus in natural ecosystems. *Geoderma*, 64(3-4), 197-214. doi:10.1016/0016-7061(94)00023-4
6. Frossard, E., Condron, L. M., Oberson, A., Sinaj, S., and Fardeau, J. C. (2000). Processes governing phosphorus availability in temperate soils. *Journal of Environmental Quality*, 29(1), 15-23.
7. Guidry, M. W., and Mackenzie, F. T. (2003). Experimental study of igneous and sedimentary apatite dissolution. *Geochimica et Cosmochimica Acta*, 67(16), 2949-2963. doi:10.1016/S0016-7037(02)01138-1
8. Hedley, M. J., Stewart, J. W. B., and Chauhan, B. S. (1982). Changes in inorganic and organic soil phosphorus fractions induced by cultivation practices and by laboratory incubations. *Soil Science Society of America Journal*, 46(5), 970-976.
9. Jackson, R. B., Canadell, J., Ehleringer, J. R., Mooney, H. A., Sala, O. E., and Schulze, E. D. (1996). A global analysis of root distributions for terrestrial biomes. *Oecologia*, 108(3), 389-411.
10. McGill, W. B., and Cole, C. V. (1981). Comparative aspects of cycling of organic C, N, S and P through soil organic matter during pedogenesis. *Geoderma*, 26(4), 267-286.
11. Parton, W. J., Stewart, J. W. B., and Cole, C. V. (1988). Dynamics of C, N, P and S in grassland soils: A model. *Biogeochemistry*, 5(1), 109-131.
12. Tinker, P. B., and Nye, P. H. (2000). *Solute Movement in the Rhizosphere*. Oxford University Press.
13. Wang, Y. P., Law, R. M., and Pak, B. (2010). A global model of carbon, nitrogen and phosphorus cycles for the terrestrial biosphere. *Biogeosciences*, 7(7), 2261-2282.
