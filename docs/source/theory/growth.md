# Plant Biomass Allocation and Growth

## 1. Process Definition and Scope

The plant growth subsystem partitions post-respiration net primary productivity (NPP) into structural organ carbon pools (`leaf_biomass_c`, `wood_biomass_c`, `root_biomass_c`) and evaluates two downstream diagnostic observers (`StructuralGrowth` and `PhenologyGrowth`).

`BiomassAllocation` is the sole structural-biomass allocation owner. Construction respiration has already been deducted upstream by `GrowthRespiration` and `AutotrophicRespiration`, so `BiomassAllocation` applies no second synthesis-efficiency deduction. Non-structural carbon reserve pools, hydraulic growth limitations, and size-structured stand demography remain `[N]`. Deeper theoretical context is provided in *Methods of Ecological Analysis*, Chapter 21.

## 2. Governing Equations

### 2.1 Carbon Partition and Organ Stock Advance (`BiomassAllocation`)

Given upstream `net_primary_productivity` ($\mathrm{NPP} \ge 0$) and optional compile-time-selected `bvoc_carbon_allocation_flux` ($F_{\mathrm{BVOC},C} \ge 0$, with $0 \le F_{\mathrm{BVOC},C} \le \mathrm{NPP}$), the allocatable structural-growth budget and organ fluxes are (Potter et al., 1993; Kesselmeier et al., 2002; Franklin et al., 2012):

$$
\mathrm{NPP}_{\mathrm{growth}} = \mathrm{NPP} - F_{\mathrm{BVOC},C}, \qquad
F_i = f_i\,\mathrm{NPP}_{\mathrm{growth}}, \qquad
\sum_{i \in \{\mathrm{leaf},\mathrm{stem},\mathrm{root}\}} f_i = 1
$$

$$
\frac{dB_i}{dt} = F_i, \qquad \Delta B_i = F_i\,\Delta t
$$

### 2.2 Environmental, Phenological, and Allometric Fraction Modifiers

1. **Environmental root modifier** (`enable_environmental_control=True`, Kelvin basis; Aber and Federer, 1992; Poorter et al., 2012; Gao et al., 2025):

$$
q_T = \operatorname{clip}\!\left(m_{\mathrm{root},T}\exp\bigl[b_{\mathrm{root},T}(T - T_{\mathrm{ref,root}})\bigr]\exp\bigl(-c_{\mathrm{root},T}\lvert T - T_{\mathrm{opt}}\rvert\bigr),\; q_{T,\min},\; q_{T,\max}\right)
$$

$$
q_W = 1 + \max\!\left(0,\; \frac{w_{\mathrm{crit}} - f_w}{w_{\mathrm{crit}}}\right), \qquad
f_{\mathrm{root,env}} = f_{\mathrm{root}}\,q_T\,q_W
$$

When $f_{\mathrm{leaf}} + f_{\mathrm{stem}} > 0$, aboveground shares rescale as $(f_{\mathrm{leaf,env}}, f_{\mathrm{stem,env}}) = (f_{\mathrm{leaf}}, f_{\mathrm{stem}})\,(1 - f_{\mathrm{root,env}})/(f_{\mathrm{leaf}} + f_{\mathrm{stem}} + \epsilon_{\mathrm{norm}})$, and the diagnostic environmental modifier is $(q_T + f_w)/2$.

2. **Discrete phenology-phase and LAI threshold controls** (`enable_phenology_control=True`; Potter et al., 1993; Zhang et al., 2011):
   - Dormant phase (`phase == 0`): sets $f_{\mathrm{leaf}} = f_{\mathrm{leaf,dormant}}$ and redistributes the remainder across stem and root proportionally to $f_{\mathrm{stem}}/(f_{\mathrm{stem}} + f_{\mathrm{root}} + \epsilon_{\mathrm{norm}})$ and $f_{\mathrm{root}}/(f_{\mathrm{stem}} + f_{\mathrm{root}} + \epsilon_{\mathrm{norm}})$.
   - Greenup phase (`phase == 1`): sets $f_{\mathrm{leaf}} = f_{\mathrm{greenup}}$; woody plants ($f_{\mathrm{stem}} > f_{\mathrm{woody,thr}}$) split $1 - f_{\mathrm{greenup}}$ by $(e_{\mathrm{stem}}, e_{\mathrm{root}})$, whereas herbaceous plants set $f_{\mathrm{stem}} = f_{\mathrm{stem,herb}}$ and $f_{\mathrm{root}} = 1 - f_{\mathrm{greenup}}$.
   - Mature or senescence phases (`phase in {2, 3}`): discrete canopy thresholds set $f_{\mathrm{leaf}} = f_{\mathrm{leaf,LAImax}}$ with proportional stem/root redistribution when $\mathrm{LAI} \ge \mathrm{LAI}_{\max}$, or apply the greenup leaf-expansion rule when $\mathrm{LAI} < \mathrm{LAI}_{\min}$.

3. **Optional pipe-model and root-shoot allometric blend** (`enable_allometry_control=True`; Shinozaki et al., 1964; Enquist and Niklas, 2002; Franklin et al., 2012):

$$
f_{\mathrm{stem,tgt}} = \operatorname{clip}\!\left(f_{\mathrm{leaf}}\,\operatorname{clip}\!\left(\frac{B_{\mathrm{leaf}}}{\max(B_{\mathrm{wood}}, B_{\min})},\; \epsilon_{\mathrm{norm}},\; r_{\max}\right)^{a_{\mathrm{ls}}},\; f_{\min},\; f_{\max}\right)
$$

$$
R_{\mathrm{rs}} = \operatorname{clip}\!\left(\frac{B_{\mathrm{root}}}{\max(B_{\mathrm{leaf}} + B_{\mathrm{wood}}, B_{\min})},\; \epsilon_{\mathrm{norm}},\; r_{\max}\right)^{b_{\mathrm{rs}}}, \qquad
f_{\mathrm{root,tgt}} = \operatorname{clip}\!\left(\frac{R_{\mathrm{rs}}}{1 + R_{\mathrm{rs}}},\; f_{\min},\; f_{\max}\right)
$$

with $f_{\mathrm{leaf,tgt}} = \operatorname{clip}(1 - f_{\mathrm{stem,tgt}} - f_{\mathrm{root,tgt}}, f_{\min}, f_{\max})$, normalized to $\hat{f}_{i,\mathrm{tgt}}$ and blended via $f_i' = (1 - w_{\mathrm{allo}})f_i + w_{\mathrm{allo}}\hat{f}_{i,\mathrm{tgt}}$, followed by final simplex normalization $\sum_i f_i = 1$.

### 2.3 Leaf-Area Tendency and Activity Observers (`StructuralGrowth`, `PhenologyGrowth`)

`StructuralGrowth` evaluates a diagnostic leaf-area tendency through `advance_reaction_network` with $N_h = [1, -1]$ and $r_h = [r_{\mathrm{growth}}, r_{\mathrm{loss}}]^{\mathsf{T}}$ without writing `leaf_area_index`:

$$
r_{\mathrm{growth}} = \mathrm{SLA}\,F_{\mathrm{leaf}}, \qquad
r_{\mathrm{loss}} = k_{\mathrm{leaf}}\,\mathrm{LAI}, \qquad
r_{\mathrm{net}} = r_{\mathrm{growth}} - r_{\mathrm{loss}}, \qquad
\Delta\mathrm{LAI}_{\mathrm{obs}} = r_{\mathrm{net}}\,\Delta t
$$

`PhenologyGrowth` maps the selected phenology owner's dimensionless `phenological_factor` $\phi \in [0, 1]$ to a binary activity indicator:

$$
g_{\mathrm{active}} = \mathbb{I}[\phi > \phi_{\mathrm{active}}]
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $\mathrm{NPP}$ | Upstream net primary productivity | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `net_primary_productivity` | `flux` |
| $F_{\mathrm{BVOC},C}$ | Optional compile-time-selected BVOC carbon debit | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `bvoc_carbon_allocation_flux` | `flux` |
| $T, f_w$ | Air temperature and soil-water stress factor | $\mathrm{K}$, $1$ | `air_temperature`, `water_stress_factor` | `forcing` |
| $\mathrm{LAI}, \text{phase}, \phi$ | Leaf area index, integer phenology phase, activity factor | $\mathrm{m^2\,m^{-2}}$, $1$, $1$ | `leaf_area_index`, `phenology_phase`, `phenological_factor` | `state` / `diagnostic` |
| $B_{\mathrm{leaf}}, B_{\mathrm{wood}}, B_{\mathrm{root}}$ | Structural organ carbon pools | $\mathrm{kg\,C\,m^{-2}}$ | `leaf_biomass_c`, `wood_biomass_c`, `root_biomass_c` | `state` |
| $F_{\mathrm{leaf}}, F_{\mathrm{stem}}, F_{\mathrm{root}}$ | Organ carbon allocation fluxes | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `growth_rate_leaf`, `growth_rate_stem`, `growth_rate_root` | `flux` |
| $\Delta\mathrm{LAI}_{\mathrm{obs}}, g_{\mathrm{active}}$ | Diagnostic LAI step change and binary growth-active flag | $\mathrm{m^2\,m^{-2}}$, $1$ | `diagnostics.structural_growth.step_leaf_area_delta`, `diagnostics.phenology.growth_active` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $f_{\mathrm{leaf,0}}, f_{\mathrm{stem,0}}, f_{\mathrm{root,0}}$ | `biosphere.growth.allocation.{f_leaf_base,f_stem_base,f_root_base}` or `biosphere.pft.<pft>.allocation.f_*` | $1$ | `growth/allocation.yaml` / `pft/*.yaml` | Poorter et al. (2012); Franklin et al. (2012) |
| $b_{\mathrm{root},T}, T_{\mathrm{ref,root}}, m_{\mathrm{root},T}$ | `biosphere.growth.allocation.{root_alloc_temp_coeff,root_alloc_temp_ref,root_alloc_temp_mult}` | $\mathrm{K^{-1}}$, $\mathrm{K}$, $1$ | `growth/allocation.yaml` | Aber and Federer (1992); Gao et al. (2025) |
| $c_{\mathrm{root},T}, T_{\mathrm{opt}}, w_{\mathrm{crit}}$ | `biosphere.growth.allocation.{temperature_stress_factor,temperature_opt,water_stress_threshold}` | $\mathrm{K^{-1}}$, $\mathrm{K}$, $1$ | `growth/allocation.yaml` | Potter et al. (1993); Poorter et al. (2012) |
| $\mathrm{LAI}_{\min}, \mathrm{LAI}_{\max}, f_{\mathrm{greenup}}$ | `biosphere.growth.allocation.{lai_min_threshold,lai_max_threshold,greenup_f_leaf}` | $\mathrm{m^2\,m^{-2}}$, $\mathrm{m^2\,m^{-2}}$, $1$ | `growth/allocation.yaml` | Potter et al. (1993); Zhang et al. (2011) |
| $a_{\mathrm{ls}}, b_{\mathrm{rs}}, w_{\mathrm{allo}}$ | `biosphere.growth.allocation.{allometry_leaf_stem_exp,allometry_root_shoot_exp,allometric_blend_weight}` | $1$ | `growth/allocation.yaml` | Shinozaki et al. (1964); Enquist and Niklas (2002) |
| $\mathrm{SLA}, k_{\mathrm{leaf}}$ | `biosphere.phenology.morphology.sla`, `biosphere.phenology.turnover.leaf_turnover_rate` | $\mathrm{m^2\,kg^{-1}\,C}$, $\mathrm{s^{-1}}$ | `phenology/*.yaml` | Landsberg and Waring (1997) |
| $\phi_{\mathrm{active}}$ | `biosphere.growth.allocation.growth_active_threshold` | $1$ | `growth/allocation.yaml` | Richardson et al. (2013) context |

## 5. Numerical Methods and Discretization

- `BiomassAllocation` promotes parameters to the widest floating dtype of the state inputs, lowers the three organ rates through `advance_reaction_network(..., open_system_route="port")` with $N_h = I_3$, emits organ fluxes as `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`, and emits pre-integrated stock increments $\Delta B_i = F_i\Delta t$ as `UpdateItem(UpdateMode.ACCUMULATE, per_timestep=True, element="C")`.
- Missing, non-finite, or negative `net_primary_productivity` or `bvoc_carbon_allocation_flux` inputs, as well as $F_{\mathrm{BVOC},C} > \mathrm{NPP}$ or non-integer/unsupported `phenology_phase` codes, raise immediately without clipping.

## 6. Conservation, Boundary Conditions, and Ledgers

`BiomassAllocation` enforces the open NPP partition ledger (`carbon_partition`) both inside `forward()` and via `check_conservation_laws(...)`:

$$
R_C = \mathrm{NPP} - F_{\mathrm{BVOC},C} - F_{\mathrm{leaf}} - F_{\mathrm{stem}} - F_{\mathrm{root}}, \qquad
\epsilon_{\mathrm{part}} = \max\!\left(\frac{\lvert R_C\rvert}{\max(\lvert\mathrm{NPP}\rvert, 1_{[\mathrm{NPP}=0]})}\right)
$$

Stock increments satisfy $\sum_i \Delta B_i - \Delta t\sum_i F_i = 0$ and register under `carbon.total` without double-counting the instantaneous flux ports. `StructuralGrowth` and `PhenologyGrowth` emit `diagnostics.*` only and modify no conserved stock.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `BiomassAllocation` | `relic/scientific/biosphere/growth/biomass_allocation.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`GRID_CELL`) |
| `StructuralGrowth` | `relic/scientific/biosphere/growth/structural_growth.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`PATCH`) |
| `PhenologyGrowth` | `relic/scientific/biosphere/growth/phenology_growth.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`PATCH`) |

## 8. Implementation Status

- `BiomassAllocation`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_growth_biomass_allocation_biomass_allocation.yaml`.
- `StructuralGrowth`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`, diagnostic tendency observer) in `biosphere_growth_structural_growth_structural_growth.yaml`.
- `PhenologyGrowth`: `[I]` Level 1 (`observation`, `diagnostic`) in `biosphere_growth_phenology_growth_phenology_growth.yaml`.
- Non-structural carbon reserve pools, hydraulic growth limitations, and size-structured stand dynamics: `[N]`.

Frozen certificate equivalence expressions:

```text
dB_i/dt = f_i * NPP_eff * eta_i with un-routed phenology suppression and duplicated construction loss
x_dot = N_h r_h with ParameterManager-backed process rates r_h and declared stoichiometric/output map N_h
dLAI/dt = SLA * c_g_to_kg * growth_rate_leaf - (k_leaf / seconds_per_year) * LAI
y_dot = N_h r_h; delta_y = y_dot * dt
f = f(GDD, temperature, photoperiod); rate = f
diagnostics.phenology.growth_active = O_h(phenological_factor)
```

## 9. References

1. Aber, J. D., and Federer, C. A. (1992). A generalized, lumped-parameter model of photosynthesis, evapotranspiration and net primary production in temperate and boreal forest ecosystems. *Oecologia*, 92, 463-474.
2. Potter, C. S., Randerson, J. T., Field, C. B., et al. (1993). Terrestrial ecosystem production: a process model based on global satellite and surface data. *Global Biogeochemical Cycles*, 7, 811-841.
3. Shinozaki, K., Yoda, K., Hozumi, K., and Kira, T. (1964). A quantitative analysis of plant form: the pipe model theory. I. Basic analyses. *Japanese Journal of Ecology*, 14, 97-105.
4. Enquist, B. J., and Niklas, K. J. (2002). Global allocation rules for patterns of biomass partitioning in seed plants. *Science*, 295, 1517-1520.
5. Sands, P. J., and Landsberg, J. J. (2002). Parameterisation of 3-PG for plantation grown *Eucalyptus globulus*. *Forest Ecology and Management*, 163, 273-292.
6. Kesselmeier, J., Ciccioli, P., Kuhn, U., et al. (2002). Volatile organic compound emissions in relation to plant carbon fixation and the terrestrial carbon budget. *Global Biogeochemical Cycles*, 16, 1126.
7. Zhang, Q., Wang, Y. P., Matear, R. J., Pitman, A. J., and Dai, Y. J. (2011). CASA-CNP carbon allocation routines. CSIRO Marine and Atmospheric Research Technical Documentation.
8. Franklin, O., Johansson, J., Dewar, R. C., et al. (2012). Modeling carbon allocation in trees: a search for principles. *Tree Physiology*, 32, 648-666.
9. Poorter, H., Niklas, K. J., Reich, P. B., et al. (2012). Biomass allocation to leaves, stems and roots: meta-analyses of interspecific variation and environmental control. *New Phytologist*, 193, 30-50.
10. Gao, Y., Zhou, Z., Ollinger, S. V., et al. (2025). pnetr: an R package for the PnET family of forest ecosystem models. *Methods in Ecology and Evolution*.
