# Autotrophic Plant Respiration

## 1. Process Definition and Scope

The autotrophic respiration subsystem computes plant metabolic carbon release and net primary productivity (NPP) through three composed `BiologicalProcessFunctor` modules: `MaintenanceRespiration` (organ protein-turnover and ion-gradient maintenance), `GrowthRespiration` (carbon-bounded biosynthesis cost), and `AutotrophicRespiration` (state-driven reaction aggregator evaluating total autotrophic respiration and signed NPP).

`MaintenanceRespiration` and `GrowthRespiration` own their respective rate laws and commit `maintenance_respiration_total` and `growth_respiration_total` to `StateManager`. `AutotrophicRespiration` consumes those committed fluxes together with `gross_primary_productivity` without calling child Functors directly. Type II thermal-sensitivity acclimation ($E_0$ shift), Lloyd-Taylor respiration inside `MaintenanceRespiration`, prognostic soluble-sugar pool dynamics, and VPD or relative-water-content respiration modifiers remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 20.

## 2. Governing Equations

### 2.1 Organ Maintenance Respiration and Type I Acclimation (`MaintenanceRespiration`)

For organs $i \in \{\mathrm{leaf}, \mathrm{stem}, \mathrm{root}\}$ with carbon biomass $B_i$ ($\mathrm{g\,C\,m^{-2}}$) and Kelvin air temperature $T$ (Ryan, 1991; Atkin and Tjoelker, 2003):

$$
Q_{10,i} = Q_{10,\mathrm{base}}\,f_{Q_{10},i}, \qquad
k_{m,i,\mathrm{acc}} =
\begin{cases}
k_{m,i,25}\exp\!\bigl[-a_{\mathrm{acc}}(T_{\mathrm{growth}} - T_{\mathrm{ref}})\bigr], & \text{acclimation active} \\
k_{m,i,25}, & \text{otherwise}
\end{cases}
$$

$$
k_{m,i}(T) = k_{m,i,\mathrm{acc}}\,Q_{10,i}^{(T - T_{\mathrm{ref}})/10}, \qquad
f_{\mathrm{sub}} = \frac{S}{K_{m,\mathrm{sub}} + S}, \qquad
f_w = \operatorname{clip}\!\left(\frac{\psi_{\mathrm{soil}} - \psi_{\mathrm{wilt}}}{\psi_{\mathrm{fc}} - \psi_{\mathrm{wilt}}},\; 0,\; 1\right)
$$

$$
R_{m,i} = 10^{-3}\,k_{m,i}(T)\,B_i\,f_{\mathrm{sub}}\,f_w, \qquad
R_m = \sum_{i \in \{\mathrm{leaf},\mathrm{stem},\mathrm{root}\}} R_{m,i}
$$

Type I acclimation activates when `enable_acclimation=True` and either `growing_temperature` is supplied in `state` or at least three spatial-mean samples occupy the rolling `acclimation_days` ring buffer (`T_growth` equals the valid-window mean). The factor $10^{-3}$ converts $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ to $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ via `UnitConverter`.

### 2.2 Carbon-Bounded Biosynthesis Cost (`GrowthRespiration`)

From upstream `gross_primary_productivity` ($\mathrm{GPP}$) and `maintenance_respiration_total` ($R_m$), available post-maintenance carbon flux and nutrient-modulated growth efficiency $Y_g$ are (Thornley and Cannell, 2000; Amthor, 2000):

$$
C_{\mathrm{avail}} = \max(\mathrm{GPP} - R_m,\; 0), \qquad
r_{NP} = \frac{N_{\mathrm{plant}}}{P_{\mathrm{plant}} + \epsilon_P}, \qquad
Y_g = \operatorname{clip}\!\left(Y_0 + a_{NP}\frac{r_{NP}}{r_{NP} + K_{NP}},\; Y_{\min},\; Y_{\max}\right)
$$

With temperature modifier $f_{T,g} = Q_{10,g}^{(T - T_{\mathrm{ref}})/10}$ and soluble-sugar factor $f_{S,g} = S / (K_{m,g} + S)$:

$$
R_{g,\mathrm{pot}} = (1 - Y_g)\,C_{\mathrm{avail}}\,f_{T,g}\,f_{S,g}, \qquad
R_g = \min(R_{g,\mathrm{pot}},\; C_{\mathrm{avail}}), \qquad
R_{g,\mathrm{res}} = R_{g,\mathrm{pot}} - R_g \ge 0
$$

where $R_{g,\mathrm{res}}$ is emitted as `diagnostics.growth_respiration.carbon_limit_residual`.

### 2.3 Total Autotrophic Respiration and Signed NPP Ledger (`AutotrophicRespiration`)

`AutotrophicRespiration` lowers the committed component fluxes through `advance_reaction_network` ($\dot{x}_h = N_h r_h$):

$$
R_{\mathrm{auto}} = R_m + R_g, \qquad
\mathrm{NPP} = \mathrm{GPP} - R_{\mathrm{auto}}
$$

Signed negative $\mathrm{NPP}$ is preserved whenever $R_{\mathrm{auto}} > \mathrm{GPP}$ without rescaling committed respiration fluxes.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $T, T_{\mathrm{growth}}$ | Air temperature and optional/window-mean growth temperature | $\mathrm{K}$ | `air_temperature`, `growing_temperature` | `forcing` |
| $B_{\mathrm{leaf}}, B_{\mathrm{stem}}, B_{\mathrm{root}}$ | Organ carbon biomass inputs for maintenance | $\mathrm{g\,C\,m^{-2}}$ | `biomass_leaf`, `biomass_stem`, `biomass_root` | `state` |
| $S, \psi_{\mathrm{soil}}$ | Soluble-sugar substrate pool and soil water potential | $\mathrm{g\,C\,m^{-2}}$, $\mathrm{MPa}$ | `soluble_sugar`, `soil_water_potential` | `state` / `forcing` |
| $N_{\mathrm{plant}}, P_{\mathrm{plant}}$ | Plant nitrogen and phosphorus pools | $\mathrm{g\,N\,m^{-2}}$, $\mathrm{g\,P\,m^{-2}}$ | `plant_nitrogen`, `plant_phosphorus` | `state` |
| $\mathrm{GPP}$ | Canopy gross primary productivity | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `gross_primary_productivity` | `flux` |
| $R_m, R_g, R_{\mathrm{auto}}$ | Maintenance, growth, and total autotrophic respiration | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `maintenance_respiration_total`, `growth_respiration_total`, `total_autotrophic_respiration` | `flux` |
| $\mathrm{NPP}$ | Signed net primary productivity | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `net_primary_productivity` | `flux` |
| $R_{g,\mathrm{res}}, r_{C,\mathrm{bal}}$ | Growth carbon-limit residual and NPP balance residual | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `diagnostics.growth_respiration.carbon_limit_residual`, `diagnostics.autotrophic_respiration.carbon_balance_residual` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $k_{m,\mathrm{leaf},25}, k_{m,\mathrm{stem},25}, k_{m,\mathrm{root},25}$ | `biosphere.respiration.maintenance.{rm_leaf_25,rm_stem_25,rm_root_25}` | $\mathrm{s^{-1}}$ | `respiration.yaml` ($1.736\times 10^{-7}, 1.736\times 10^{-8}, 1.157\times 10^{-7}$) | Ryan (1991) |
| $Q_{10,\mathrm{base}}, f_{Q_{10},\mathrm{leaf/stem/root}}$ | `biosphere.respiration.maintenance.{q10_respiration,q10_leaf_factor,q10_stem_factor,q10_root_factor}` | $1$ | `respiration.yaml` ($2.0, 1.0, 0.9, 1.1$) | Atkin and Tjoelker (2003) |
| $a_{\mathrm{acc}}, K_{m,\mathrm{sub}}$ | `biosphere.respiration.maintenance.{acclimation_rate_coeff,km_substrate}` | $\mathrm{K^{-1}}$, $\mathrm{g\,C\,m^{-2}}$ | `respiration.yaml` (acclimation; $10.0$) | Atkin and Tjoelker (2003); Reich et al. (2016) |
| $\psi_{\mathrm{wilt}}, \psi_{\mathrm{fc}}$ | `biosphere.respiration.maintenance.{water_stress_wilting_point,water_stress_field_capacity}` | $\mathrm{MPa}$ | `respiration.yaml` | Gifford (2003) |
| $Y_0, Y_{\min}, Y_{\max}$ | `biosphere.respiration.growth.{ygrow_base,min_growth_efficiency,max_growth_efficiency}` | $1$ | `respiration.yaml` ($0.65, 0.40, 0.75$) | Amthor (2000); Thornley and Cannell (2000) |
| $a_{NP}, K_{NP}, \epsilon_P$ | `biosphere.respiration.growth.{np_response_coeff,np_half_sat}`, `biosphere.validation.epsilon.nutrient` | $1$, $1$, $\mathrm{g\,P\,m^{-2}}$ | `respiration.yaml` / `validation.yaml` ($0.2, 0.0667$) | Thornley and Cannell (2000) |
| $Q_{10,g}, K_{m,g}, \epsilon_{\mathrm{flux}}$ | `biosphere.respiration.growth.{q10_growth,km_substrate_growth}`, `biosphere.validation.epsilon.flux` | $1$, $\mathrm{g\,C\,m^{-2}}$, $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `respiration.yaml` / `validation.yaml` ($2.0, 5.0$) | Amthor (2000) |

## 5. Numerical Methods and Discretization

- All three Functors lower their outputs through `advance_reaction_network(..., open_system_route="port")` and return instantaneous `UpdateItem(UpdateMode.REPLACE, per_timestep=False, element=None)` fluxes and diagnostics.
- `MaintenanceRespiration` persists its `acclimation_days` ring buffer (`buffer`, `count`, `index`, `window_days`) through PyTorch `get_extra_state()` and `set_extra_state()` so rollback and format-3 checkpoint restart replay the exact acclimation window. Air temperature is validated over $[233.15, 333.15]\ \mathrm{K}$.

## 6. Conservation, Boundary Conditions, and Ledgers

`AutotrophicRespiration` enforces the exact open-port carbon identity:

$$
r_{C,\mathrm{bal}} = \mathrm{GPP} - R_{\mathrm{auto}} - \mathrm{NPP} = 0
$$

In the State Contract Manifest (SCM), `total_autotrophic_respiration` is a positive outward boundary flux (`allowed_out: [autotrophic_respiration]`), while signed `net_primary_productivity` authorizes `autotrophic_respiration` in both `allowed_in` and `allowed_out`. Instantaneous respiration fluxes carry `accumulates: null` so `StateManager` does not double-count them as carbon storage pools.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `MaintenanceRespiration` | `relic/scientific/biosphere/respiration/maintenance_respiration.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `MAIN_PROCESSES` (`GRID_CELL`) |
| `GrowthRespiration` | `relic/scientific/biosphere/respiration/growth_respiration.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `MAIN_PROCESSES` (`GRID_CELL`) |
| `AutotrophicRespiration` | `relic/scientific/biosphere/respiration/autotrophic_respiration.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `MAIN_PROCESSES` (`GRID_CELL`) |

## 8. Implementation Status

- `MaintenanceRespiration`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_respiration_maintenance_respiration_maintenance_respiration.yaml`.
- `GrowthRespiration`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_respiration_growth_respiration_growth_respiration.yaml` (depends on photosynthesis owner and `maintenance_respiration`).
- `AutotrophicRespiration`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_respiration_autotrophic_respiration_autotrophic_respiration.yaml` (depends on photosynthesis owner, `maintenance_respiration`, and `growth_respiration`).
- Type II sensitivity acclimation, prognostic soluble-sugar pool dynamics, and VPD/RWC respiration modifiers: `[N]`.

Frozen certificate equivalence expressions:

```text
R(T) = R_ref * Q10^((T-T_ref)/10)
R_g = min((1 - Ygrow) * max(0, GPP - R_m) * f_T * f_S, max(0, GPP - R_m))
x_dot = N_h r_h with ParameterManager-backed process rates r_h and declared stoichiometric/output map N_h
R_auto = R_maintenance + R_growth; NPP = GPP - R_auto
x_dot = N_h r_h with scheduler-provided carbon-port rates r_h and declared stoichiometric/output map N_h
```

## 9. References

1. Ryan, M. G. (1991). Effects of climate change on plant respiration. *Ecological Applications*, 1, 157-167.
2. Lloyd, J., and Taylor, J. A. (1994). On the temperature dependence of soil respiration. *Functional Ecology*, 8, 315-323.
3. Amthor, J. S. (2000). The McCree--de Wit--Penning de Vries--Thornley respiration paradigms: 30 years later. *Annals of Botany*, 86, 1-20.
4. Thornley, J. H. M., and Cannell, M. G. R. (2000). Modelling the components of plant respiration: representation and realism. *Annals of Botany*, 85, 55-67.
5. Atkin, O. K., and Tjoelker, M. G. (2003). Thermal acclimation and the dynamic response of plant respiration to temperature. *Trends in Plant Science*, 8, 343-351.
6. Gifford, R. M. (2003). Plant respiration in productivity models: conceptualisation, representation and issues for global terrestrial carbon-cycle research. *Functional Plant Biology*, 30, 171-186.
7. Reich, P. B., Sendall, K. M., Stefanski, A., et al. (2016). Boreal and temperate trees show strong acclimation of respiration to warming. *Nature*, 531, 633-636.
