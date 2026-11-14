# Hydrosphere Surface Runoff Partitioning

## 1. Process Definition and Scope

`SurfaceRunoff` (`relic/scientific/hydrosphere/runoff/surface.py`) partitions incoming surface water supply into outward overland flow (`surface_runoff_flux`, $\mathrm{kg\,m^{-2}\,s^{-1}}$) and downward soil infiltration (`infiltration_flux`, $\mathrm{kg\,m^{-2}\,s^{-1}}$) using Soil Conservation Service Curve Number (`"scs_cn"`), Hortonian infiltration-excess (`"hortonian"`), Dunne saturation-excess (`"saturation_excess"`), or `"combined"` closures.

Compile-time parameter `hydrosphere.runoff.surface_water_input_source` selects `"precipitation_flux"` (reading `precipitation_flux`) or `"rain_flux_plus_snow_melt_flux"` (reading `rain_flux + snow_melt_flux` and registering `snow_pack` as a required same-step dependency). Horizontal two-dimensional shallow-water overland routing across grid cells and runtime mutation of `curve_number` remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Section 19.5.1).

## 2. Governing Equations

### 2.1 Surface Water Input Rate and Top-Layer Moisture Extraction

From selected mass supply $P_m$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$), liquid water density $\rho_w$, and $s_{\mathrm{hour}}$ (`seconds_per_hour`), equivalent rainfall intensity in $\mathrm{mm\,h^{-1}}$ is $P_{\mathrm{rate}} = (1000\,s_{\mathrm{hour}} / \rho_w)\,P_m$. For layered `soil_moisture` profiles, surface moisture $\theta_{\mathrm{surf}} = \theta[0]$ extracts the uppermost soil cell so top-boundary outputs remain scalar per column.

### 2.2 Individual Runoff Generation Closures

1. **Dimensionally consistent step-integrated SCS-CN** (`"scs_cn"`, Mockus, 1957; USDA NRCS, 1986):

$$
S_0 = \max\!\left(\frac{c_{\mathrm{num}}}{\mathrm{CN}} - c_{\mathrm{off}},\; S_{\min}\right), \qquad
S = S_0\,f_{\mathrm{amc}}\bigl(1 - \alpha_\theta\,\operatorname{clip}(\theta_{\mathrm{surf}}, 0, 1)\bigr), \qquad
I_a = \lambda_a S
$$

$$
P_{\mathrm{event}} = P_{\mathrm{rate}}\frac{\Delta t}{s_{\mathrm{hour}}}, \qquad
E_{\mathrm{scs}} = \max(P_{\mathrm{event}} - I_a,\; 0), \qquad
Q_{\mathrm{event}} = \begin{cases} \dfrac{E_{\mathrm{scs}}^2}{E_{\mathrm{scs}} + S}, & E_{\mathrm{scs}} > 0 \\ 0, & E_{\mathrm{scs}} = 0 \end{cases}, \qquad
Q_{\mathrm{scs}} = Q_{\mathrm{event}}\frac{s_{\mathrm{hour}}}{\Delta t}
$$

2. **Moisture-adjusted Hortonian infiltration-excess** (`"hortonian"`, Horton, 1933):

$$
I_{c,\mathrm{eff}} = I_c\bigl(1 - c_\theta\,\operatorname{clip}(\theta_{\mathrm{surf}}, 0, 1)\bigr), \qquad
Q_h = \max(0,\; P_{\mathrm{rate}} - I_{c,\mathrm{eff}})
$$

3. **Dunne variable-source saturation-excess** (`"saturation_excess"`, Dunne and Black, 1970):

$$
A_{\mathrm{sat}} = \operatorname{clip}\!\left(\frac{\theta_{\mathrm{surf}} - \theta_{\mathrm{thr}}}{1 - \theta_{\mathrm{thr}}},\; 0,\; 1\right), \qquad
Q_s = P_{\mathrm{rate}}\,A_{\mathrm{sat}}
$$

4. **Combined rule** (`"combined"`, controlled by `runoff_combination_method`):

$$
Q_{\mathrm{comb}} =
\begin{cases}
\min\bigl(P_{\mathrm{rate}},\; \max(Q_{\mathrm{scs}}, Q_h, Q_s)\bigr), & \text{method} = \text{"max"} \\
\min\bigl(P_{\mathrm{rate}},\; Q_{\mathrm{scs}} + Q_h + Q_s\bigr), & \text{method} = \text{"sum"}
\end{cases}
$$

### 2.3 Mass-Flux Conversion, Partition Closure, and Structural Mobility

Converting selected rate $Q$ ($\mathrm{mm\,h^{-1}}$) back to volumetric rate $Q_{\mathrm{vol}}$ ($\mathrm{m\,s^{-1}}$) gives bounded overland mass flux $R_{\mathrm{surf}}$, infiltration mass flux $I_{\mathrm{soil}}$, and runoff coefficient $C_R$:

$$
R_{\mathrm{surf}} = \min\bigl(P_m,\; \max(0,\; \rho_w Q_{\mathrm{vol}})\bigr), \qquad
I_{\mathrm{soil}} = P_m - R_{\mathrm{surf}}, \qquad
C_R = \begin{cases} \dfrac{R_{\mathrm{surf}}}{P_m}, & P_m > F_{\min} \\ 0, & P_m \le F_{\min} \end{cases}
$$

The Level-3 structural mobility operator $\star_{\mathrm{surf}} = \operatorname{diag}(\rho_w D_{\mathrm{surf}})$ ($\mathrm{kg\,m^{-1}\,s^{-1}}$) satisfies $\sigma = X_{\mathrm{surf}}^{\mathsf{T}}\star_{\mathrm{surf}} X_{\mathrm{surf}} \ge 0$.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $P$ | Total precipitation mass flux (`surface_water_input_source="precipitation_flux"`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `precipitation_flux` | `forcing` |
| $P_{\mathrm{rain}}, M_{\mathrm{snow}}$ | Rain and snowmelt mass fluxes (`surface_water_input_source="rain_flux_plus_snow_melt_flux"`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `rain_flux`, `snow_melt_flux` | `flux` |
| $\theta$ | Soil volumetric water content (scalar or profile $\theta[0]$) | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `state` |
| $R_{\mathrm{surf}}$ | Outward surface runoff boundary mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `surface_runoff_flux` | `flux` |
| $I_{\mathrm{soil}}$ | Downward soil-infiltration boundary mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `infiltration_flux` | `flux` |
| $C_R, P_{\mathrm{rate}}, Q$ | Runoff coefficient, rainfall rate, and runoff depth rate | $\mathrm{1}$, $\mathrm{mm\,h^{-1}}$ | `diagnostics.surface_runoff.runoff_coefficient`, `precipitation_rate_mm_h`, `runoff_rate_mm_h` | `diagnostic` |
| $S, I_a$ | Effective SCS retention and initial abstraction depths | $\mathrm{mm}$ | `diagnostics.surface_runoff.scs_retention`, `initial_abstraction` | `diagnostic` |
| $r_{\mathrm{nn}}, r_{\mathrm{lim}}, r_w$ | Nonnegative projection, precipitation-cap, and partition balance residuals | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.surface_runoff.runoff_nonnegative_residual`, `precipitation_limit_residual`, `water_balance_residual` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $m_{\mathrm{run}}, m_{\mathrm{comb}}, s_{\mathrm{in}}$ | `hydrosphere.runoff.runoff_method`, `runoff_combination_method`, `surface_water_input_source` | `str` | `hydrosphere/runoff.yaml` | Oleson et al. (2013); Neitsch et al. (2011) |
| $\mathrm{CN}, \lambda_a, S_{\min}$ | `hydrosphere.runoff.curve_number`, `initial_abstraction_ratio`, `scs_storage_minimum` | $\mathrm{1}$, $\mathrm{1}$, $\mathrm{mm}$ | `hydrosphere/runoff.yaml` | USDA NRCS (1986) |
| $c_{\mathrm{num}}, c_{\mathrm{off}}$ | `hydrosphere.physical.scs_numerator_constant`, `scs_offset_constant` | $\mathrm{mm}$ | `hydrosphere/physical.yaml` | SI metric SCS constants ($25400$, $254$) |
| $f_{\mathrm{amc}}, \alpha_\theta$ | `hydrosphere.runoff.antecedent_moisture_factor`, `moisture_adjustment_factor` | $\mathrm{1}$ | `hydrosphere/runoff.yaml` | Neitsch et al. (2011) |
| $I_c, c_\theta$ | `hydrosphere.runoff.infiltration_capacity`, `infiltration_moisture_reduction_factor` | $\mathrm{mm\,h^{-1}}$, $\mathrm{1}$ | `hydrosphere/runoff.yaml` | Horton (1933) |
| $\theta_{\mathrm{thr}}$ | `hydrosphere.runoff.saturated_area_threshold` | $\mathrm{m^3\,m^{-3}}$ | `hydrosphere/runoff.yaml` | Dunne and Black (1970) |
| $D_{\mathrm{surf}}, F_{\min}$ | `hydrosphere.transport.surface_water_diffusivity`, `hydrosphere.numerical.minimum_flux_threshold` | $\mathrm{m^2\,s^{-1}}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `hydrosphere/transport.yaml`, `numerical.yaml` | Structural Hodge diffusivity and threshold |
| $\rho_w, s_{\mathrm{hour}}$ | `relic.physical.constants.water_density`, `relic.physical.constants.seconds_per_hour` | $\mathrm{kg\,m^{-3}}$, $\mathrm{s\,h^{-1}}$ | `physical/constants.yaml` | IAPWS / SI constants |

## 5. Numerical Methods and Discretization

All outputs are instantaneous rates or diagnostics submitted via `UpdateItem(UpdateMode.REPLACE, per_timestep=False)` on differentiable `torch.Tensor` inputs. Only the configured `surface_water_input_source` variables are read, preventing double counting when both `precipitation_flux` and `snow_pack` outputs exist in `StateManager`.

## 6. Conservation, Boundary Conditions, and Ledgers

`SurfaceRunoff` is a zero-storage open boundary partition satisfying exact mass splitting $r_w = P_m - R_{\mathrm{surf}} - I_{\mathrm{soil}} = 0$. `SurfaceRunoff.check_conservation_laws()` evaluates $\max\lvert r_w\rvert$, while `infiltration_flux` enters downstream `RichardsSoilMoisture` through the `surface_soil_water_interface` port (`BucketSoilMoisture` remains the mutually exclusive single-reservoir runoff alternative).

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `SurfaceRunoff` | `relic/scientific/hydrosphere/runoff/surface.py` | `TransportMechanism` (`TRANSPORT_TYPE="advection"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `SurfaceRunoff`: `[I]`, IPHS **Level 3** (`transport`, certificate `hydrosphere_runoff_surface_surface_runoff.yaml`). Exact two-port surface mass partition with positive structural mobility audit $\sigma = X_{\mathrm{surf}}^{\mathsf{T}}\star_{\mathrm{surf}} X_{\mathrm{surf}} \ge 0$.

Frozen certificate expressions:

```text
surface_runoff_flux = min(precipitation, max(0, rho_w * Q(P, CN, theta, Ic))).
m_dot = G_surf u_surf; u_surf = min(P, max(0, rho_w * Q)); sigma = X_surf.T star_surf X_surf >= 0.
```

## 9. References

- Dunne, T., and Black, R. D. (1970). Partial area contributions to storm runoff in a small New England watershed. *Water Resources Research*, 6(5), 1296-1311.
- Horton, R. E. (1933). The role of infiltration in the hydrologic cycle. *Transactions, American Geophysical Union*, 14(1), 446-460.
- Mockus, V. (1957). *Use of Storm Rainfall for Runoff Estimates*. USDA Soil Conservation Service.
- Neitsch, S. L., Arnold, J. G., Kiniry, J. R., and Williams, J. R. (2011). *Soil and Water Assessment Tool Theoretical Documentation Version 2009* (TR-406). Texas Water Resources Institute.
- Oleson, K. W., Lawrence, D. M., Bonan, G. B., et al. (2013). *Technical Description of Version 4.5 of the Community Land Model (CLM)* (NCAR/TN-503+STR). NCAR.
- USDA NRCS (1986). *Urban Hydrology for Small Watersheds* (Technical Release 55). USDA Natural Resources Conservation Service.
