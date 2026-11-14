# Hydrosphere Snowpack Mass Balance

## 1. Process Definition and Scope

`SnowPack` (`relic/scientific/hydrosphere/snow/snowpack.py`) evolves grid-cell snow water equivalent (`snow_water_equivalent`, $\mathrm{mm}$) and bulk geometric `snow_depth` ($\mathrm{m}$) through temperature-threshold precipitation partitioning, degree-day snowmelt, and start-of-step inventory-bounded melt and sublimation withdrawals.

All four inputs (`precipitation_flux`, `air_temperature`, `snow_water_equivalent`, and `sublimation_flux`, including an explicit zero tensor when sublimation is absent) are mandatory finite canonical ports. Multilayer snowpack thermal conduction, liquid-water retention/refreezing within snow pores, and prognostic snow-density compaction remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Section 19.4.1).

## 2. Governing Equations

### 2.1 Rain-Snow Partitioning and Potential Degree-Day Melt

From total `precipitation_flux` $P_{\mathrm{tot}}$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$), Kelvin `air_temperature` $T_{\mathrm{air}}$, rain-snow threshold $T_{\mathrm{rs}}$, melt threshold $T_{\mathrm{melt}}$, degree-day factor $\mathrm{DDF}$ ($\mathrm{mm\,K^{-1}\,day^{-1}}$), and $s_{\mathrm{day}}$ (`seconds_per_day`) (Rango and Martinec, 1995; Hock, 2003):

$$
f_{\mathrm{snow}} =
\begin{cases}
1, & T_{\mathrm{air}} \le T_{\mathrm{rs}} \\
0, & T_{\mathrm{air}} > T_{\mathrm{rs}}
\end{cases}, \qquad
P_{\mathrm{snow,mass}} = f_{\mathrm{snow}}\,P_{\mathrm{tot}}, \qquad
P_{\mathrm{rain,mass}} = (1 - f_{\mathrm{snow}})\,P_{\mathrm{tot}}
$$

$$
M_{0,\mathrm{depth}} = \frac{\mathrm{DDF}\,\max(0,\; T_{\mathrm{air}} - T_{\mathrm{melt}})}{s_{\mathrm{day}}}
$$

Mass fluxes ($\mathrm{kg\,m^{-2}\,s^{-1}}$) and liquid-equivalent depth rates ($\mathrm{mm\,s^{-1}}$) convert via `UnitConverter` using physical-constant liquid water density $\rho_w$.

### 2.2 Sequential Inventory-Bounded Withdrawals and SWE Update

With starting SWE $S^n \ge 0$ ($\mathrm{mm}$), step $\Delta t$ ($\mathrm{s}$), snowfall depth rate $P_s$, potential melt rate $M_0 = M_{0,\mathrm{depth}}$, and potential sublimation depth rate $E_0$, melt takes priority over sublimation against starting stock $S^n$ before new snowfall is added:

$$
M = \min\!\left(M_0,\; \frac{S^n}{\Delta t}\right), \qquad
E_{\mathrm{sub}} = \min\!\left(E_0,\; \frac{\max(S^n - M\Delta t,\; 0)}{\Delta t}\right)
$$

$$
S^{n+1} = \max\bigl(0,\; S^n + (P_s - M - E_{\mathrm{sub}})\Delta t\bigr)
$$

Unmet melt and sublimation demands ($M_0 - M$ and $E_0 - E_{\mathrm{sub}}$) are recorded as explicit limiter residuals in `diagnostics.snow_pack`.

### 2.3 Bulk Snow Depth Conversion

Updated SWE converts from $\mathrm{mm}$ to $\mathrm{m}$ through `UnitConverter` ($S_m = \operatorname{convert}(S^{n+1},\, \mathrm{mm},\, \mathrm{m})$) and scales by the ratio of liquid water density $\rho_w$ to bulk snow density $\rho_{\mathrm{snow}}$:

$$
z_{\mathrm{snow}} = S_m\,\frac{\rho_w}{\rho_{\mathrm{snow}}}
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $P_{\mathrm{tot}}$ | Total precipitation mass flux (required) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `precipitation_flux` | `forcing` |
| $T_{\mathrm{air}}$ | Near-surface air temperature (required) | $\mathrm{K}$ | `air_temperature` | `forcing` |
| $S$ | Snow water equivalent (start-of-step input and end-of-step replacement) | $\mathrm{mm}$ | `snow_water_equivalent` | `state` |
| $E_{0,\mathrm{mass}}$ | Potential snow sublimation mass flux (required) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `sublimation_flux` | `flux` |
| $z_{\mathrm{snow}}$ | Bulk geometric snow depth | $\mathrm{m}$ | `snow_depth` | `state` |
| $M_{\mathrm{mass}}$ | Realized snowmelt mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `snow_melt_flux` | `flux` |
| $P_{\mathrm{rain,mass}}$ | Partitioned liquid rainfall mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `rain_flux` | `flux` |
| $P_{\mathrm{snow,mass}}$ | Partitioned solid snowfall mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `snowfall` | `flux` |
| $\mathcal{D}_{\mathrm{snow}}$ | Nested melt/sublimation limiter and mass-balance residual dictionary | mixed SI | `diagnostics.snow_pack` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\mathrm{DDF}$ | `hydrosphere.snow.degree_day_factor` | $\mathrm{mm\,K^{-1}\,day^{-1}}$ | `hydrosphere/snow.yaml` | Rango and Martinec (1995); Hock (2003) |
| $T_{\mathrm{melt}}$ | `hydrosphere.snow.temperature_threshold_warm` | $\mathrm{K}$ | `hydrosphere/snow.yaml` | Hock (2003) |
| $T_{\mathrm{rs}}$ | `hydrosphere.snow.rain_snow_temperature_threshold` | $\mathrm{K}$ | `hydrosphere/snow.yaml` | Running and Coughlan (1988) |
| $\rho_{\mathrm{snow}}$ | `hydrosphere.snow.snow_density` | $\mathrm{kg\,m^{-3}}$ | `hydrosphere/snow.yaml` | Pomeroy and Gray (1995) |
| $\rho_w, s_{\mathrm{day}}$ | `relic.physical.constants.water_density`, `relic.physical.constants.seconds_per_day` | $\mathrm{kg\,m^{-3}}$, $\mathrm{s\,day^{-1}}$ | `physical/constants.yaml` | IAPWS / SI constants |

## 5. Numerical Methods and Discretization

`SnowPack` preserves the floating `dtype` of its input tensors (`float32` or `float64`) and declares `SUBFLOW_ACCURACY = SubFlowAccuracy.ORDER_1`. On interior steps where neither inventory cap is active ($S^n \ge (M_0 + E_0)\Delta t$), the affine trajectory is exact to machine precision; across depletion or replenishment cap crossings, the sequential start-of-step withdrawal rule has global first-order accuracy $\mathcal{O}(\Delta t)$, so same-fast-group flux coupling to `BucketSoilMoisture` uses Lie splitting or slow-provider hold and excludes Strang second-order qualification. Outputs use `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`.

## 6. Conservation, Boundary Conditions, and Ledgers

Precipitation partitioning conserves incoming atmospheric water mass exactly ($P_{\mathrm{tot}} = P_{\mathrm{snow,mass}} + P_{\mathrm{rain,mass}}$), and the open SWE control volume closes:

$$
r_{\mathrm{SWE}} = (S^{n+1} - S^n) - (P_s - M - E_{\mathrm{sub}})\Delta t = 0
$$

to floating-point precision, with realized `snow_melt_flux` and `rain_flux` supplying downstream soil-moisture and surface-runoff processes when `water_input_source="rain_flux_plus_snow_melt_flux"`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `SnowPack` | `relic/scientific/hydrosphere/snow/snowpack.py` | `PhysicalField` (`SPATIAL_DIMS=None`) | `PhysicalFieldFunctor` (`relic/core/functor/physical_field_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `SnowPack`: `[I]`, IPHS **Level 3** (`field`, certificate `hydrosphere_snow_snowpack_snow_pack.yaml`). Inventory-bounded SWE field evolution $\frac{d\mathrm{SWE}}{dt} = \star_{\mathrm{SWE}}\dot{m}$ with explicit accumulation, melt, and sublimation matter ports.
- Full snowpack internal energy balance and liquid-water refreezing: `[N]`.

Frozen certificate expressions:

```text
SWE_new = max(0, SWE_old + (P_snow - M - E_sub) dt); snow_depth = SWE_mm / 1000 * rho_w / rho_snow.
dSWE/dt = star_SWE m_dot plus accumulation and melt matter ports; melt, sublimation, and nonnegative storage projection residuals are routed to diagnostics.snow_pack.
```

## 9. References

- Hock, R. (2003). Temperature index melt modelling in mountain areas. *Journal of Hydrology*, 282(1-4), 104-115.
- Pomeroy, J. W., and Gray, D. M. (1995). *Snowcover Accumulation, Relocation and Management* (NHRI Science Report No. 7). National Hydrology Research Institute.
- Rango, A., and Martinec, J. (1995). Revisiting the degree-day method for snowmelt computations. *Water Resources Bulletin*, 31(4), 657-669.
- Running, S. W., and Coughlan, J. C. (1988). A general model of forest ecosystem processes for regional applications. I. Hydrologic balance, canopy gas exchange and primary production processes. *Ecological Modelling*, 42(2), 125-154.
