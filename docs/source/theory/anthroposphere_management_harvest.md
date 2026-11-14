# Agricultural and Forestry Harvest Biomass-Residue Partitioning

## 1. Process Definition and Scope

`HarvestManagement` (`relic/scientific/anthroposphere/management/harvest.py`) partitions `aboveground_biomass` and `belowground_biomass` during scheduled `HarvestEvent` operations into exported product fluxes (`harvest_flux`, `harvest_carbon_export_flux`, `harvest_nitrogen_export_flux`), returned surface-residue and root-mortality fluxes (`residue_c_flux`, `residue_n_flux`), and an additive surface residue-cover tendency (`delta_residue_cover`).

Six `CropType` categories (`cereal_c3`, `cereal_c4`, `legume`, `root_tuber`, `forage`, `forest`) have parameterized carbon fractions and $\mathrm{C:N}$ ratios (`[I]`); `fiber`, `sugar`, `oil_seed`, and `tree_fruit` raise `ValueError` (`[N]`). `HarvestManagement` emits daily-normalized boundary fluxes and `delta_residue_cover` without directly emitting `UpdateItem` stock decrements on `aboveground_biomass` or `belowground_biomass`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 24 (Section 24.5.3).

## 2. Governing Equations

### 2.1 Storage Potential, Effort Vector, and Calendar-Interval Trigger

Over the five-channel biomass cochain $x_h = [B_{\mathrm{ag}}, B_{\mathrm{ag}}, B_{\mathrm{ag}}, B_{\mathrm{bg}}, B_{\mathrm{ag}}]^{\mathsf{T}}$ ($\mathrm{kg\,DM\,m^{-2}}$) and positive diagonal Hodge matrix $H_{\mathrm{harv}} = \operatorname{diag}(h_1, \dots, h_5)$ with $h_i = w_i / \sum_{j=1}^5 w_j$ (`harvest_efficiency`, `default_harvest_index`, `residue_removal_rate`, `root_mortality_rate`, `residue_retention`), quadratic storage potential $G_h(x_h)$ and effort $\mu_h$ are evaluated on every step:

$$
G_h(x_h) = \frac{1}{2}\,x_h^{\mathsf{T}} H_{\mathrm{harv}} x_h, \qquad
\mu_h = \frac{\partial G_h}{\partial x_h} = H_{\mathrm{harv}} x_h
$$

A scheduled `HarvestEvent` at day $d_e$ triggers when circular calendar distance $d_C(d, d_e) = \min(\lvert d - d_e\rvert,\; Y(C) - \lvert d - d_e\rvert) \le W$ and elapsed time since `last_harvest_time_seconds` satisfies $t - t_{\mathrm{last}} > T_{\min}\,s_{\mathrm{day}}$ (or $t_{\mathrm{last}}$ is unset).

### 2.2 Sequential Event Partition and Standard-Day Flux Scaling

For each due event $e$ acting sequentially on local working stocks $(B_{\mathrm{ag}}^{(e)}, B_{\mathrm{bg}}^{(e)}, c_{\mathrm{res}}^{(e)})$, clamped harvest index $\widetilde{\mathrm{HI}}_e = \operatorname{clip}(\mathrm{HI}_e, 0, 1)$, mechanical efficiency $\eta_h$, quality factor $q_e$, residue retention $r_e$, and root mortality fraction $k_{\mathrm{root}}$ form control fractions

$$
u_{H,e} = \operatorname{clip}\bigl(\widetilde{\mathrm{HI}}_e\,\eta_h\,q_e,\; 0,\; 1\bigr), \qquad
u_{R,e} = \operatorname{clip}\bigl((1 - \widetilde{\mathrm{HI}}_e\,\eta_h\,q_e)\,r_e,\; 0,\; 1\bigr), \qquad
u_{D,e} = \operatorname{clip}(k_{\mathrm{root}},\; 0,\; 1)
$$

With control vector $u_e = [u_{H,e}, u_{R,e}, \operatorname{clip}(u_{H,e} + u_{R,e}, 0, 1), u_{D,e}, u_{R,e}]^{\mathsf{T}}$ and diagonal mobility $M_h(u_e) = \operatorname{diag}(u_{e,i} / h_i)$, state advancement $M_h(u_e)\mu_h = u_e \odot x_h^{(e)}$ produces event dry-matter transfers $H_e = u_{H,e} B_{\mathrm{ag}}^{(e)}$, $R_e = u_{R,e} B_{\mathrm{ag}}^{(e)}$, and $D_e = u_{D,e} B_{\mathrm{bg}}^{(e)}$ (raising `ValueError` if $H_e + R_e > B_{\mathrm{ag}}^{(e)}$ or $D_e > B_{\mathrm{bg}}^{(e)}$), alongside elemental masses and capped residue-cover change:

$$
C_{\mathrm{exp},e} = f_{\mathrm{C},e}\,H_e, \qquad
C_{\mathrm{res},e} = f_{\mathrm{C},e}(R_e + D_e), \qquad
C_{\mathrm{in},e} = C_{\mathrm{exp},e} + C_{\mathrm{res},e}, \qquad
N_{x,e} = \frac{C_{x,e}}{r_{\mathrm{CN},e}}
$$

$$
\Delta c_e = \operatorname{clip}\!\bigl(c_{\mathrm{res}}^{(e)} + \mathbf{1}_{B_{\mathrm{ag}}^{(e)} > 0}\,u_{R,e},\; 0,\; 1\bigr) - c_{\mathrm{res}}^{(e)}, \qquad
F_Y = \frac{1}{s_{\mathrm{day}}}\sum_{e} Y_e
$$

for $Y_e \in \{H_e, C_{\mathrm{in},e}, C_{\mathrm{exp},e}, C_{\mathrm{res},e}, N_{\mathrm{in},e}, N_{\mathrm{exp},e}, N_{\mathrm{res},e}, \Delta c_e\}$. Because `last_harvest_time_seconds` locks out re-execution for $T_{\min}$ days while $F_Y$ divides by fixed $s_{\mathrm{day}}$, single-step integration scales event totals by $\Delta t / s_{\mathrm{day}}$ whenever $\Delta t$ differs from $s_{\mathrm{day}}$ (*Methods of Ecological Analysis*, Chapter 24, Exercise 24.5).

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $B_{\mathrm{ag}}, B_{\mathrm{bg}}$ | Aboveground and belowground dry-matter biomass stocks | $\mathrm{kg\,m^{-2}}$ | `aboveground_biomass`, `belowground_biomass` | `state` |
| $c_{\mathrm{res}}, d$ | Surface residue cover fraction (required) and optional day-of-year | $\mathrm{1}$, $\mathrm{d}$ | `residue_cover`, `day_of_year` | `forcing` |
| $F_{\mathrm{harv}}$ | Exported dry-matter harvest flux | $\mathrm{kg\,DM\,m^{-2}\,s^{-1}}$ | `harvest_flux` | `flux` |
| $F_{\mathrm{C,in}}, F_{\mathrm{C,exp}}, F_{\mathrm{C,res}}$ | Total partitioned, exported product, and returned residue+root carbon fluxes | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `harvest_biomass_carbon_input`, `harvest_carbon_export_flux`, `residue_c_flux` | `flux` |
| $F_{\mathrm{N,in}}, F_{\mathrm{N,exp}}, F_{\mathrm{N,res}}$ | Total partitioned, exported product, and returned residue+root nitrogen fluxes | $\mathrm{kg\,N\,m^{-2}\,s^{-1}}$ | `harvest_biomass_nitrogen_input`, `harvest_nitrogen_export_flux`, `residue_n_flux` | `flux` |
| $F_{\mathrm{cov}}$ | Additive surface residue-cover tendency (`ACCUMULATE`, `per_timestep=False`) | $\mathrm{s^{-1}}$ | `delta_residue_cover` | `flux` |
| $n_{\mathrm{evt}}, R_B, R_{\mathrm{C}}, R_{\mathrm{N}}$ | Executed event count, dry-matter partition residual, and C/N ledger residuals | $\mathrm{1}$, $\mathrm{kg\,DM\,m^{-2}}$, $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$, $\mathrm{kg\,N\,m^{-2}\,s^{-1}}$ | `diagnostics.harvest.events_today`, `diagnostics.harvest.biomass_partition_residual`, `diagnostics.harvest.carbon_ledger_residual`, `diagnostics.harvest.nitrogen_ledger_residual` | `diagnostic` |
| $G_h, \lVert \mu_h\rVert_2, c_{\mathrm{iphs}}$ | Storage potential, effort Euclidean norm, and canonical runtime flag | $\mathrm{kg^2\,m^{-4}}$, $\mathrm{kg\,m^{-2}}$, $\mathrm{1}$ | `diagnostics.harvest_management.free_energy`, `diagnostics.harvest_management.chemical_potential_norm`, `diagnostics.harvest_management.canonical_iphs_runtime` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\eta_h, k_{\mathrm{root}}$ | `anthroposphere.management.harvest.harvest_efficiency`, `anthroposphere.management.harvest.root_mortality_rate` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Johnson et al. (2006); Keating et al. (2003) |
| $\mathrm{HI}_0, r_{\mathrm{ret},0}, r_{\mathrm{rem},0}$ | `anthroposphere.management.harvest.default_harvest_index`, `anthroposphere.management.harvest.residue_retention`, `anthroposphere.management.harvest.residue_removal_rate` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Lal (2005); Blanco-Canqui and Lal (2009) |
| $f_{\mathrm{C},e}, r_{\mathrm{CN},e}$ | `anthroposphere.management.harvest.cereal_c3_c_content`, `anthroposphere.management.harvest.cereal_c3_cn_ratio`, `anthroposphere.management.harvest.*` | $\mathrm{kg\,C\,kg^{-1}\,DM}$, $\mathrm{kg\,C\,kg^{-1}\,N}$ | `anthroposphere/management.yaml` | Crop-specific elemental ratios (6 supported crops) |
| $W, T_{\min}, \theta_{\mathrm{sched}}$ | `anthroposphere.management.harvest.schedule_window_days`, `anthroposphere.management.harvest.minimum_event_interval_days`, `anthroposphere.management.harvest.default_schedule.*` | $\mathrm{d}$, $\mathrm{1}$ | `anthroposphere/management.yaml` | Calendar window and default 3-cut schedule |
| $\tau_{\mathrm{cons}}$ | `validation.conservation.conservation_tolerance` | $\mathrm{1}$ | `validation/conservation.yaml` | Partition closure tolerance |
| $s_{\mathrm{day}}$ | `relic.physical.constants.time.seconds_per_day` | $\mathrm{s\,d^{-1}}$ | `physical/constants.yaml` | SI day length ($86400$) |

## 5. Numerical Methods and Discretization

All 15 outputs are wrapped as `UpdateItem` instances with `per_timestep=False`: `delta_residue_cover` uses `UpdateMode.ACCUMULATE` so it composes additively with negative cover burial from `TillageManagement`, while the remaining 14 flux and diagnostic outputs use `UpdateMode.REPLACE`. Replay checkpointing (`get_extra_state` / `set_extra_state`, format version 2) persists `last_harvest_time_seconds`, `harvest_count`, `annual_yield`, and cumulative removal scalars; `harvest_events` itself is configured at construction time.

## 6. Conservation, Boundary Conditions, and Ledgers

`HarvestManagement` enforces dry-matter working-stock partition closure and exact carbon and nitrogen export-plus-residue splitting on every step:

$$
R_B = \operatorname{relu}\!\bigl(\lvert B_{\mathrm{ag}} - B_{\mathrm{removed}} - B_{\mathrm{rem}}\rvert - \tau_{\mathrm{cons}}\bigr), \qquad
R_{\mathrm{C}} = \lvert F_{\mathrm{C,in}} - F_{\mathrm{C,exp}} - F_{\mathrm{C,res}}\rvert, \qquad
R_{\mathrm{N}} = \lvert F_{\mathrm{N,in}} - F_{\mathrm{N,exp}} - F_{\mathrm{N,res}}\rvert
$$

`check_conservation_laws()` returns the maximum scale-normalized residual across $(R_B, R_{\mathrm{C}}, R_{\mathrm{N}})$.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `HarvestManagement` | `relic/scientific/anthroposphere/management/harvest.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `HarvestManagement`: `[I]`, IPHS **Level 3** (`port`, certificate `anthroposphere_management_harvest_harvest_management.yaml`). Storage-potential and diagonal Hodge-mobility harvest partition across grain export, surface residue, root mortality, and additive residue cover; `fiber`, `sugar`, `oil_seed`, and `tree_fruit` `CropType` variants remain `[N]`.

Frozen certificate expressions:

```text
H_biomass = B_total * HI * efficiency; R_C = (B_total - H_biomass) * C_fraction * R_return.
Harvest-index, residue-retention, and root-turnover closures are control coefficients inside the single IPHS harvest biomass-residue port state advancer.
```

## 9. References

- Blanco-Canqui, H., and Lal, R. (2009). Crop residue removal impacts on soil productivity and environmental quality. *Critical Reviews in Plant Sciences*, 28(3), 139-163.
- Johnson, J. M. F., Allmaras, R. R., and Reicosky, D. C. (2006). Estimating source carbon from crop residues, roots and rhizodeposits using the national grain-yield database. *Agronomy Journal*, 98(3), 622-636.
- Keating, B. A., Carberry, P. S., Hammer, G. L., et al. (2003). An overview of APSIM, a model designed for farming systems simulation. *European Journal of Agronomy*, 18(3-4), 267-288.
- Lal, R. (2005). World crop residues production and implications of its use as a biofuel. *Environment International*, 31(4), 575-584.
- Prince, S. D., Haskett, J., Steininger, M., Strand, H., and Wright, R. (2001). Net primary production of U.S. Midwest croplands from agricultural harvest yield data. *Ecological Applications*, 11(4), 1194-1205.
