# Biosphere Stress Response Formulations

## 1. Process Definition and Scope

RELIC implements four patch-scale environmental stress Functors under `relic/scientific/biosphere/stress/`, all inheriting from `BiologicalProcessFunctor` and executing in `ExecutionPhase.MAIN_PROCESSES` at `SpatialScale.PATCH`:

1. `WaterStress` (`water_stress.py`): maps canonical `soil_water_potential` ($\mathrm{MPa}$) to `water_stress_factor`.
2. `ThermalStress` (`thermal_stress.py`): maps canonical `air_temperature` ($\mathrm{K}$), or the temperature selected by `air_temperature_source` (Section 2.2.1), to `temperature_stress_factor`, `heat_stress_factor`, and `cold_stress_factor`, or to the subset selected by `stress_outputs`.
3. `NutrientStress` (`nutrient_stress.py`): maps tissue `nitrogen_concentration` and `phosphorus_concentration` ($\mathrm{kg\,element\,kg^{-1}\,dry\,matter}$) to `nutrient_stress_factor` and per-element diagnostic factors.
4. `OzoneStress` (`ozone_stress.py`): evaluates open stomatal ozone uptake, apoplastic antioxidant detoxification, excess damage tendency, and `ozone_stress_factor` through `advance_reaction_network`.

None of the four Functors accepts legacy forcing aliases (`leaf_water_potential`, `temperature`, `leaf_temperature`), prognostic osmotic acclimation (`[N]`), or biomass-dilution critical-N curves (`[N]`); `ThermalStress` reads `leaf_temperature` or `leaf_temperature_max` only when `air_temperature_source` names it as the declared input, never as a fallback alias. Deeper derivations appear in *Methods of Ecological Analysis*, Chapters 20 to 22.

## 2. Governing Equations

### 2.1 Soil Water Potential Stress (`WaterStress`)

With non-positive `soil_water_potential` $\psi_{\mathrm{soil}}$ ($\mathrm{MPa}$), permanent wilting point $\psi_{\mathrm{wilt}}$, field capacity potential $\psi_{\mathrm{fc}}$ ($\psi_{\mathrm{fc}} > \psi_{\mathrm{wilt}}$), and global bounds $0 \le f_{\min} < f_{\max} \le 1$ (Running and Coughlan, 1988; Williams et al., 2001):

$$
f_{\mathrm{water}} = \operatorname{clip}\!\left(\frac{\psi_{\mathrm{soil}} - \psi_{\mathrm{wilt}}}{\psi_{\mathrm{fc}} - \psi_{\mathrm{wilt}}},\; f_{\min},\; f_{\max}\right)
$$

### 2.2 Cardinal and Gaussian Thermal Stress (`ThermalStress`)

From near-surface `air_temperature` $T$ ($\mathrm{K}$), optimal temperature $T_{\mathrm{opt}}$, tolerance width $\sigma$, cardinal thresholds $(T_{\mathrm{heat}}, T_{\mathrm{crit}}, T_{\mathrm{cold}}, T_{\mathrm{freeze}})$, and negative slopes $(k_{\mathrm{heat}}, k_{\mathrm{cold}})$ in $\mathrm{K^{-1}}$ (Medlyn et al., 2002; Way and Oren, 2010; Hatfield and Prueger, 2015):

$$
f_T = \operatorname{clip}\!\left(\exp\!\left[-\frac{1}{2}\left(\frac{T - T_{\mathrm{opt}}}{\sigma}\right)^2\right],\; f_{\min},\; f_{\max}\right)
$$

$$
f_{\mathrm{heat}} = \operatorname{clip}\!\left(\begin{cases} 1, & T \le T_{\mathrm{heat}} \\ \max\bigl(0,\, 1 + k_{\mathrm{heat}}(T - T_{\mathrm{heat}})\bigr), & T_{\mathrm{heat}} < T \le T_{\mathrm{crit}} \\ 0, & T > T_{\mathrm{crit}} \end{cases},\; f_{\min},\; f_{\max}\right)
$$

$$
f_{\mathrm{cold}} = \operatorname{clip}\!\left(\begin{cases} 1, & T \ge T_{\mathrm{cold}} \\ \max\bigl(0,\, 1 + k_{\mathrm{cold}}(T_{\mathrm{cold}} - T)\bigr), & T_{\mathrm{freeze}} \le T < T_{\mathrm{cold}} \\ 0, & T < T_{\mathrm{freeze}} \end{cases},\; f_{\min},\; f_{\max}\right)
$$

The three controls are emitted independently without internal multiplication.

#### 2.2.1 Temperature Source and Output Selection (`air_temperature_source`, `stress_outputs`)

`air_temperature_source` names the one declared input $T$ of the three responses (all in $\mathrm{K}$); the responses themselves are unchanged:

- `air_temperature` (default, previous interface): the near-surface air temperature.
- `air_temperature_max`, `air_temperature_min`: the daily maximum or minimum air temperature (CF-1.8 Section 7.3 `cell_methods` "time: maximum" / "time: minimum"; SCM contracts `air_temperature_max`, `air_temperature_min`), for example the daily maximum for the heat threshold and the daily minimum for the cold threshold.
- `leaf_temperature`, `leaf_temperature_max`: the big-leaf leaf temperature of `LeafEnergyBalance` and its declared daily-maximum estimate $T_{l,\max} = T_{\max} + \Delta T$ (daily support; `biosphere_leaf_energy_balance.md`). The heat threshold of Salvucci and Crafts-Brandner (2004) is a leaf-temperature response of Rubisco activase; the cold and Gaussian parameters stay as declared for air temperature.

A graph binds only same-named ports, so the selected state key is the declared input; a missing key raises. No static dependency is declared for a leaf source: the graph link to `leaf_energy_balance` orders the two nodes. `stress_outputs` selects a non-empty, duplicate-free subset of (`temperature_stress_factor`, `heat_stress_factor`, `cold_stress_factor`), written in that declaration order; the default is all three (previous interface). It lets two instances share the graph without writing the same state key, for example `thermal_stress` (heat from `air_temperature_max` or `leaf_temperature_max`) and `thermal_stress_cold` (`air_temperature_source: air_temperature_min`, `stress_outputs: [cold_stress_factor]`), the heat and cold providers of `MEGANModel` `stress_factors` (`bvoc_emissions.md`, Section 2.1.1). Both keys are wiring selections; no law or parameter is added.

### 2.3 Liebig Tissue Nutrient Stress (`NutrientStress`)

For tissue mass fractions $c_i$ ($\mathrm{kg\,element\,kg^{-1}\,dry\,matter}$), minimum thresholds $c_{i,\min}$, critical thresholds $c_{i,\mathrm{crit}}$, and exponent $p$ across $i \in \{N, P\}$ (Aerts and Chapin, 2000; Agren et al., 2012):

$$
z_i = \frac{c_i - c_{i,\min}}{c_{i,\mathrm{crit}} - c_{i,\min}}, \qquad
f_i = \operatorname{clip}\!\left(\max(z_i, 0)^p,\; f_{\min},\; f_{\max}\right), \qquad
f_{NP} = \min(f_N, f_P)
$$

### 2.4 Stomatal Ozone Uptake and Detoxification Network (`OzoneStress`)

From atmospheric ozone mole fraction $[\mathrm{O_3}]$ (`o3_mixing_ratio`, $\mathrm{mol\,mol^{-1}}$), leaf water-vapor stomatal conductance $g_s$ (`stomatal_conductance`, $\mathrm{mol\,m^{-2}\,s^{-1}}$), H$_2$O/O$_3$ diffusivity ratio $k_{\mathrm{O_3}} = 1.67$, uptake efficiency $\eta_{\mathrm{uptake}}$, detoxification flux capacity $C_{\mathrm{detox}}$ ($\mathrm{mol\,O_3\,m^{-2}\,s^{-1}}$), and apoplastic antioxidant pool capacity $C_{\mathrm{ax}}$ ($\mathrm{mol\,O_3\,m^{-2}}$) (Ploechl et al., 2000; Sitch et al., 2007; Leung et al., 2020):

$$
F_{\mathrm{O_3}} = \eta_{\mathrm{uptake}}\,[\mathrm{O_3}]\,\frac{g_s}{k_{\mathrm{O_3}}}, \qquad
r_h = \begin{bmatrix} F_{\mathrm{O_3}} \\ \min(F_{\mathrm{O_3}}, C_{\mathrm{detox}}) \\ \max(F_{\mathrm{O_3}} - C_{\mathrm{detox}}, 0) \end{bmatrix}, \qquad
\dot{x}_h = N_h r_h
$$

with stoichiometric output incidence matrix $N_h \in \mathbb{R}^{5 \times 3}$ mapping $r_h$ to uptake flux, detoxification flux $D = \min(F_{\mathrm{O_3}}, C_{\mathrm{detox}})$, excess flux, antioxidant-pool tendency $-D$, and damage tendency. Specific antioxidant depletion, downstream stress factor, and nonnegative structural diagnostics are:

$$
k_{\mathrm{ax}} = \frac{D}{C_{\mathrm{ax}}}, \qquad
f_{\mathrm{O_3}} = \operatorname{clip}\!\left(1 - \min\!\left(\frac{F_{\mathrm{O_3}}}{C_{\mathrm{detox}}},\; r_{\max}\right),\; f_{\min},\; f_{\max}\right)
$$

$$
P_{\mathrm{port}} = F_{\mathrm{O_3}}\frac{[\mathrm{O_3}]}{[\mathrm{O_3}]_{\mathrm{thr}}}, \qquad
\sigma_{\mathrm{O_3}} = \min(F_{\mathrm{O_3}}, C_{\mathrm{detox}})\max\!\left(\frac{F_{\mathrm{O_3}}}{C_{\mathrm{detox}}}, 0\right) \ge 0
$$

#### 2.4.1 Supports of the Ozone Chain

The only divisors of the law are the parameters $k_{\mathrm{O_3}}$, $C_{\mathrm{detox}}$, $C_{\mathrm{ax}}$ and $[\mathrm{O_3}]_{\mathrm{thr}}$, never $g_s$ or LAI: $g_s = 0$ (Medlyn `nonpositive_leaf_area: no_canopy`) or $[\mathrm{O_3}] = 0$ gives $F_{\mathrm{O_3}} = 0$ and $f_{\mathrm{O_3}} = \operatorname{clip}(1, f_{\min}, f_{\max}) = 1$ with the default bounds. The variable contracts declare the supports of the chain: `o3_mixing_ratio` is one value per grid cell (`spatial_kind: grid_cell`, `temporal_basis: instantaneous`, no vertical dimension), the well-mixed box (boundary-layer) mole fraction of its writers `atmospheric_chemistry`, `eddy_diffusion_blm` and `dry_deposition_aggregator`, which `OzoneStress` takes as the leaf-surface value (the $R_a \rightarrow 0$ limit of Sitch et al., 2007); `stomatal_conductance` is the leaf-area-basis conductance of its one contract tendency writer `medlyn_stomatal_conductance` (same basis and unit, no conversion), or forcing or state; `ozone_stress_factor` is an instantaneous leaf-support control (`spatial_kind: leaf`) read by `MEGANModel` when its `stress_factors` contain `ozone` and optionally by `FarquharPhotosynthesis`. No provider is declared for `stomatal_conductance`: Farquhar evaluates $A_c$ and $A_j$ at the atmospheric $\mathrm{CO_2}$ mole fraction and outputs no conductance, so the Medlyn link is an explicit graph link.

The law is an instantaneous flux response with no internal history; its outputs are the rates $\dot{x}_h = N_h r_h$, independent of $\Delta t$. In a daily graph ($\Delta t = 86400\,\mathrm{s}$) on the default support it reads the daily-mean $[\mathrm{O_3}]$ and the daily-mean $g_s$, a declared limitation and not a daily law: the daily mean of $g_s[\mathrm{O_3}]$ differs from the product of the daily means by their diurnal covariance, the clipped response is convex in $F_{\mathrm{O_3}}$ (so $f_{\mathrm{O_3}}$ of the mean flux is not the mean of $f_{\mathrm{O_3}}$), and $C_{\mathrm{detox}}$, the flux threshold $Y$ of $\mathrm{POD}_Y$ (Mills et al., 2011) and the $40\ \mathrm{nmol\,mol^{-1}}$ AOT40 threshold are defined on hourly values. $Y$ (`flux_threshold`) is declared but not read by this law.

#### 2.4.2 Daylight Gauss Temporal Support (`temporal_support: daylight_gauss`)

Daily step only. The law is evaluated at the daylight Gauss samples $k$ of `relic/scientific/atmosphere/daylight_quadrature.py` (sample construction and weights $W_k$: `photosynthesis.md`, Section 2.5; Goudriaan, 1986) with the per-sample Medlyn leaf conductance and the daily-mean ozone mole fraction held constant over the day:

$$
F_{\mathrm{O_3},k} = \eta_{\mathrm{uptake}}\,[\mathrm{O_3}]\,\frac{g_{s,k}}{k_{\mathrm{O_3}}}, \qquad
f_{\mathrm{O_3},k} = \operatorname{clip}\!\left(1 - \min\!\left(\max\!\left(\frac{F_{\mathrm{O_3},k}}{C_{\mathrm{detox}}}, 0\right),\; r_{\max}\right),\; f_{\min},\; f_{\max}\right), \qquad
X = \sum_k W_k X_k
$$

for every output $X$ (rates, $f_{\mathrm{O_3}}$ and the diagnostics); $g_{s,n} = g_0$ at night. $C_{\mathrm{detox}}$ is compared with the flux of each instant, not with a daily-mean flux. Configuration: `{"temporal_support": "daylight_gauss", "daylight_gauss_nodes": 3 | 5}` and the optional `daylight_shortwave_shape`; $\Delta t$ must equal `seconds_per_day`; daylight keys on the default support raise; absent `temporal_support` keeps `instantaneous`, bitwise the previous behaviour. Ports: `stomatal_conductance_daylight_samples` (from a `daylight_gauss` Medlyn) replaces `stomatal_conductance`; `latitude` and `day_of_year` are added; the outputs are 24-hour means plus `ozone_stress_factor_daylight_samples` ($[\mathrm{cell}, \mathrm{daylight\_sample}]$) for a `daylight_gauss` `MEGANModel`. Declared approximation: the diurnal course of $[\mathrm{O_3}]$ is not resolved, so its covariance with $g_s$ is neglected. The clipped response converges slowest with $n$: in the design probe the largest error of $f_{\mathrm{O_3}}$ against a one-minute reference was 9.7 % with $n = 3$ and 1.7 % with $n = 5$.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\psi_{\mathrm{soil}}$ | Soil water matric potential ($\le 0$) | $\mathrm{MPa}$ | `soil_water_potential` | `state` |
| $T$ | Near-surface air temperature, or the input selected by `air_temperature_source` (`ThermalStress`) | $\mathrm{K}$ | `air_temperature`, `air_temperature_max`, `air_temperature_min`, `leaf_temperature`, `leaf_temperature_max` | `forcing` / `diagnostic` |
| $c_N, c_P$ | Tissue nitrogen and phosphorus mass fractions | $\mathrm{kg\,kg^{-1}}$ | `nitrogen_concentration`, `phosphorus_concentration` | `state` |
| $[\mathrm{O_3}]$ | Atmospheric ozone mixing ratio | $\mathrm{mol\,mol^{-1}}$ | `o3_mixing_ratio` | `state` |
| $g_s$ | Leaf water-vapor stomatal conductance (forcing, state, or the `medlyn_stomatal_conductance` output) | $\mathrm{mol\,m^{-2}\,s^{-1}}$ | `stomatal_conductance` | `forcing` / `diagnostic` |
| $g_{s,k}$ | Leaf conductance at the daylight samples (`OzoneStress` `daylight_gauss`, replaces $g_s$) | $\mathrm{mol\,m^{-2}\,s^{-1}}$ | `stomatal_conductance_daylight_samples` | `diagnostic` |
| $\varphi$, $J$ | Latitude and day of year of the daylight samples (`OzoneStress` `daylight_gauss`) | $\mathrm{degrees\_north}$, $1$ | `latitude`, `day_of_year` | `forcing` |
| $f_{\mathrm{water}}$ | Soil water stress multiplier | $\mathrm{1}$ | `water_stress_factor` | `diagnostic` |
| $f_T, f_{\mathrm{heat}}, f_{\mathrm{cold}}$ | Gaussian thermal, heat, and cold stress multipliers | $\mathrm{1}$ | `temperature_stress_factor`, `heat_stress_factor`, `cold_stress_factor` | `diagnostic` |
| $f_{NP}, f_N, f_P$ | Combined Liebig nutrient multiplier and per-element diagnostics | $\mathrm{1}$ | `nutrient_stress_factor`, `diagnostics.nutrient_stress.nitrogen_factor`, `diagnostics.nutrient_stress.phosphorus_factor` | `diagnostic` |
| $f_{\mathrm{O_3}}$ | Ozone stress multiplier | $\mathrm{1}$ | `ozone_stress_factor` | `diagnostic` |
| $f_{\mathrm{O_3},k}$ | Ozone stress multiplier at the daylight samples (`daylight_gauss`) | $\mathrm{1}$ | `ozone_stress_factor_daylight_samples` | `diagnostic` |
| $F_{\mathrm{O_3}}$ | Stomatal ozone uptake flux | $\mathrm{mol\,O_3\,m^{-2}\,s^{-1}}$ | `ozone_uptake` | `flux` |
| $k_{\mathrm{ax}}$ | Specific apoplastic antioxidant consumption rate | $\mathrm{s^{-1}}$ | `antioxidant_depletion` | `diagnostic` |
| $D, F_{\mathrm{ex}}, \dot{A}_{\mathrm{ax}}, \dot{D}_{\mathrm{O_3}}, P_{\mathrm{port}}, \sigma_{\mathrm{O_3}}$ | Ozone reaction-network tendencies and structural port diagnostics | $\mathrm{mol\,O_3\,m^{-2}\,s^{-1}}$, $\mathrm{1}$ | `diagnostics.ozone_stress.detoxification_flux`, `diagnostics.ozone_stress.excess_ozone_flux`, `diagnostics.ozone_stress.antioxidant_pool_tendency`, `diagnostics.ozone_stress.damage_tendency`, `diagnostics.ozone_stress.port_power`, `diagnostics.ozone_stress.entropy_production` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $f_{\min}, f_{\max}$ | `biosphere.stress.global.factor_min`, `biosphere.stress.global.factor_max` | $\mathrm{1}$ | `biosphere/stress.yaml` | Shared stress bounds |
| $\psi_{\mathrm{wilt}}, \psi_{\mathrm{fc}}$ | `biosphere.stress.water.wilting_point_potential`, `biosphere.stress.water.field_capacity_potential` | $\mathrm{MPa}$ | `biosphere/stress.yaml` | Running and Coughlan (1988); Cox et al. (1998) |
| $T_{\mathrm{opt}}, \sigma$ | `biosphere.stress.thermal.optimal_temperature`, `biosphere.stress.thermal.temperature_tolerance_width` | $\mathrm{K}$ | `biosphere/stress.yaml` | Medlyn et al. (2002); Way and Oren (2010) |
| $T_{\mathrm{heat}}, T_{\mathrm{crit}}, k_{\mathrm{heat}}$ | `biosphere.stress.thermal.heat_stress_threshold`, `biosphere.stress.thermal.critical_high_temperature`, `biosphere.stress.thermal.heat_stress_coefficient` | $\mathrm{K}$, $\mathrm{K}$, $\mathrm{K^{-1}}$ | `biosphere/stress.yaml` | Hatfield and Prueger (2015) |
| $T_{\mathrm{cold}}, T_{\mathrm{freeze}}, k_{\mathrm{cold}}$ | `biosphere.stress.thermal.cold_stress_threshold`, `biosphere.stress.thermal.freezing_damage_threshold`, `biosphere.stress.thermal.cold_stress_coefficient` | $\mathrm{K}$, $\mathrm{K}$, $\mathrm{K^{-1}}$ | `biosphere/stress.yaml` | Hatfield and Prueger (2015) |
| $c_{N,\mathrm{crit}}, c_{N,\min}, c_{P,\mathrm{crit}}, c_{P,\min}, p$ | `biosphere.stress.nutrient.n_critical_concentration`, `biosphere.stress.nutrient.n_minimum_concentration`, `biosphere.stress.nutrient.p_critical_concentration`, `biosphere.stress.nutrient.p_minimum_concentration`, `biosphere.stress.nutrient.nutrient_stress_exponent` | $\mathrm{kg\,kg^{-1}}$, $\mathrm{1}$ | `biosphere/stress.yaml` | Aerts and Chapin (2000); Agren et al. (2012) |
| $[\mathrm{O_3}]_{\mathrm{thr}}, F_{\mathrm{O_3,thr}}, \eta_{\mathrm{uptake}}$ | `biosphere.stress.ozone.ozone_threshold`, `biosphere.stress.ozone.flux_threshold`, `biosphere.stress.ozone.ozone_uptake_rate` | $\mathrm{mol\,mol^{-1}}$, $\mathrm{mol\,O_3\,m^{-2}\,s^{-1}}$, $\mathrm{1}$ | `biosphere/stress.yaml` | Emberson et al. (2000); Mills et al. (2011) |
| $C_{\mathrm{detox}}, C_{\mathrm{ax}}, r_{\max}$ | `biosphere.stress.ozone.ozone_detox_capacity`, `biosphere.stress.ozone.antioxidant_capacity`, `biosphere.stress.ozone.max_flux_ratio` | $\mathrm{mol\,O_3\,m^{-2}\,s^{-1}}$, $\mathrm{mol\,O_3\,m^{-2}}$, $\mathrm{1}$ | `biosphere/stress.yaml` | Ploechl et al. (2000); Sitch et al. (2007) |
| $k_{\mathrm{O_3}}$ | `relic.physical.constants.conversions.h2o_o3_diffusivity_ratio` | $\mathrm{1}$ | `physical/constants.yaml` | Sitch et al. (2007); Leung et al. (2020) |

## 5. Numerical Methods and Discretization

All four Functors evaluate instantaneous algebraic or open-port rate closures on `torch.Tensor` inputs and emit `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`. Parameter threshold orderings ($\psi_{\mathrm{fc}} > \psi_{\mathrm{wilt}}$, $c_{i,\mathrm{crit}} > c_{i,\min}$, $0 \le f_{\min} < f_{\max} \le 1$) and finite input domains are validated before calculation. If a caller converts stomatal conductance from $\mathrm{m\,s^{-1}}$ to $\mathrm{mol\,m^{-2}\,s^{-1}}$, `UnitConverter` applies $g_{\mathrm{mol}} = g_m P / (R T)$ using canonical `air_temperature` and `atmospheric_pressure`.

The `daylight_gauss` support of `OzoneStress` requires $\Delta t = 86400\,\mathrm{s}$, applies the input checks of the law to every sample, and returns the weighted daily means as `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`; the network increment formed with $\Delta t$ is not returned. `ThermalStress` writes only its `stress_outputs`.

## 6. Conservation, Boundary Conditions, and Ledgers

`WaterStress`, `ThermalStress`, and `NutrientStress` are dimensionless control observations that own no water, heat, nitrogen, or phosphorus pools and declare no mass or energy ledger. `OzoneStress` is an open boundary-port reaction network (`open_system_route="port"`): atmospheric ozone enters from the boundary and apoplastic antioxidant consumption represents an unclosed physiological defense rate without advancing an ozone or antioxidant mass stock.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `WaterStress` | `relic/scientific/biosphere/stress/water_stress.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.PATCH`) |
| `ThermalStress` | `relic/scientific/biosphere/stress/thermal_stress.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.PATCH`) |
| `NutrientStress` | `relic/scientific/biosphere/stress/nutrient_stress.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.PATCH`) |
| `OzoneStress` | `relic/scientific/biosphere/stress/ozone_stress.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.PATCH`) |

## 8. Implementation Status

- `WaterStress`: `[I]`, IPHS **Level 1** (`observation`, certificate `biosphere_stress_water_stress_water_stress.yaml`).
- `ThermalStress`: `[I]`, IPHS **Level 1** (`observation`, certificate `biosphere_stress_thermal_stress_thermal_stress.yaml`).
- `NutrientStress`: `[I]`, IPHS **Level 1** (`observation`, certificate `biosphere_stress_nutrient_stress_nutrient_stress.yaml`).
- `OzoneStress`: `[I]`, IPHS **Level 2** (`reaction`, certificate `biosphere_stress_ozone_stress_ozone_stress.yaml`). Evaluates $\dot{x}_h = N_h r_h$ via `advance_reaction_network`; `sigma_O3` is a bounded detoxification diagnostic without thermodynamic affinity pairing.

Frozen certificate expressions:

```text
f_{water} = (psi - psi_wilt) / (psi_{fc} - psi_{wilt})
f_temp = exp(-0.5 * ((T - T_opt)/sigma)^2)
y_h = O_h x_h through the declared observation operator
z_i=(c_i-c_i,min)/(c_i,crit-c_i,min); f_i=clip(max(z_i,0)^p,f_min,f_max); f_NP=min(f_N,f_P)
y_h = O_h(c_N,c_P;theta) through the declared observation operator
F_O3 = (g_s / k_O3) * [O3] * ozone_uptake_rate
x_dot = N_h r_h with explicit stoichiometric_N and rates evaluated by advance_reaction_network.
```

## 9. References

- Aerts, R., and Chapin, F. S. (2000). The mineral nutrition of wild plants revisited: A re-evaluation of processes and patterns. *Advances in Ecological Research*, 30, 1-67.
- Agren, G. I., Wetterstedt, J. A. M., and Billberger, M. F. K. (2012). Nutrient limitation on terrestrial plant growth: Modeling the interaction between nitrogen and phosphorus. *New Phytologist*, 194(4), 953-960.
- Cox, P. M., Huntingford, C., and Harding, R. J. (1998). A canopy conductance and photosynthesis model for use in a GCM land surface scheme. *Journal of Hydrology*, 212-213, 79-94.
- Emberson, L. D., Ashmore, M. R., Cambridge, H. M., Simpson, D., and Tuovinen, J.-P. (2000). Modelling stomatal ozone flux across Europe. *Environmental Pollution*, 109(3), 403-413.
- Hatfield, J. L., and Prueger, J. H. (2015). Temperature extremes: Effect on plant growth and development. *Weather and Climate Extremes*, 10, 4-10.
- Leung, F., Sitch, S., Tai, A. P. K., Wiltshire, A. J., Gornall, J. L., Folberth, G. A., and Unger, N. (2020). Understanding surface ozone impacts on vegetation using JULES-crop. *Geoscientific Model Development*, 13, 6201-6213.
- Medlyn, B. E., Dreyer, E., Ellsworth, D., et al. (2002). Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data. *Plant, Cell & Environment*, 25(9), 1167-1179.
- Mills, G., Pleijel, H., Braun, S., et al. (2011). New stomatal flux-based critical levels for ozone effects on vegetation. *Atmospheric Environment*, 45(28), 5064-5068.
- Goudriaan, J. (1986). A simple and fast numerical method for the computation of daily totals of crop photosynthesis. *Agricultural and Forest Meteorology*, 38, 249-254 (cited for the daylight Gauss scheme; text not read).
- Ploechl, M., Lyons, T., Ollerenshaw, J., and Barnes, J. (2000). Simulating ozone detoxification in the leaf apoplast through the direct reaction with ascorbate. *Planta*, 210(3), 454-467.
- Running, S. W., and Coughlan, J. C. (1988). A general model of forest ecosystem processes for regional applications. I. Hydrologic balance, canopy gas exchange and primary production processes. *Ecological Modelling*, 42(2), 125-154.
- Salvucci, M. E., and Crafts-Brandner, S. J. (2004). Inhibition of photosynthesis by heat stress. *Plant, Cell & Environment*, 27, 1063-1071.
- Sitch, S., Cox, P. M., Collins, W. J., and Huntingford, C. (2007). Indirect radiative forcing of climate change through ozone effects on the land-carbon sink. *Nature*, 448, 791-794.
- Way, D. A., and Oren, R. (2010). Differential responses to changes in growth temperature between trees from different functional groups and biomes: A review and synthesis of data. *Tree Physiology*, 30(6), 669-688.
- Williams, M., Law, B. E., Anthoni, P. M., and Unsworth, M. H. (2001). Use of a simulation model and ecosystem flux data to examine carbon-water interactions in ponderosa pine. *Tree Physiology*, 21(5), 287-298.
