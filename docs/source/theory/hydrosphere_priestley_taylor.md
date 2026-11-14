# Priestley-Taylor Radiation-Driven Evapotranspiration

## 1. Process Definition and Scope

`PriestleyTaylorET` (`relic/scientific/hydrosphere/evapotranspiration/priestley_taylor.py`) computes equilibrium-scaled water-vapor mass transport (`evapotranspiration_flux`, $\mathrm{kg\,m^{-2}\,s^{-1}}$) and latent heat flux (`latent_heat_flux`, $\mathrm{W\,m^{-2}}$) from surface available energy following Priestley and Taylor (1972).

`PriestleyTaylorET` declares `CONFLICT_RESOLUTION = ConflictResolutionPolicy.EXCLUSIVE` (mutually exclusive with `PenmanMonteithET`) and lagged dependencies on `bucket_soil_moisture` and `richards_soil_moisture`. Explicit aerodynamic wind-resistance coupling and prognostic canopy interception storage belong exclusively to `PenmanMonteithET` (`[N]` in `PriestleyTaylorET`). Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Section 19.3.2).

## 2. Governing Equations

### 2.1 Liquid and Sub-Freezing Ice Psychrometric Regimes

Within admitted Kelvin air-temperature bounds $T \in [T_{\mathrm{lo}}, T_{\mathrm{hi}}]$ (`hydrosphere.numerical.temperature_bounds`, with $T_C = T - 273.15$), parameter `sub_freezing_phase` selects the phase closure where $T < T_f = 273.15\,\mathrm{K}$:

- **`"liquid"`** (default at all admitted temperatures, or where $T \ge T_f$):

$$
e_s(T_C) = e_0\exp\!\left(\frac{a\,T_C}{T_C + b}\right), \qquad
\Delta = \frac{4098.0\,e_s(T_C)}{(T_C + b)^2}, \qquad
\gamma = \frac{c_p P}{\varepsilon\,L_v(T)}
$$

- **`"ice"`** (cellwise where $T < T_f$, using Murray (1967) coefficients $a_i = 21.8745584$, $b_i = 265.5\,^\circ\mathrm{C}$, and sublimation latent heat $L_s = L_{v,0} + L_f$):

$$
e_{s,i}(T_C) = e_0\exp\!\left(\frac{a_i T_C}{T_C + b_i}\right), \qquad
\Delta_i = \frac{a_i b_i\,e_{s,i}(T_C)}{(T_C + b_i)^2}, \qquad
\gamma_i = \frac{c_p P}{\varepsilon\,L_s}
$$

Denote the active phase latent heat, vapor-pressure slope, and psychrometric constant by $(\lambda, \Delta_\phi, \gamma_\phi)$.

### 2.2 Available Energy and Dynamic $\alpha$ Modulation

Parameter `soil_heat_flux_source` selects `"fraction"` ($G = f_G R_n$) or `"state"` (reading required signed `soil_heat_flux` $G$ in $\mathrm{W\,m^{-2}}$), defining available energy $A = R_n - G$.

When `enable_dynamic_alpha` is false, $\alpha = \alpha_{\mathrm{PT}}$. When `enable_dynamic_alpha` is true, $\alpha$ initializes to $\alpha_{\mathrm{PT}}$ and applies optional moisture and VPD modifiers sequentially before final clamping (de Bruin, 1983):

1. If the moisture input $\theta$ is present in `state` (`soil_water_content` on the undeclared route, the declared `soil_moisture` otherwise; Section 2.2.1), with $\theta_{\mathrm{fc}} = f_{\mathrm{fc}}\theta_s$, $\theta_{\mathrm{wp}} = f_{\mathrm{wp}}\theta_r$ (default frame; Section 2.2.1 for the bucket frame), and $\theta_{\mathrm{rel}} = \operatorname{clip}\bigl((\theta - \theta_{\mathrm{wp}})/(\theta_{\mathrm{fc}} - \theta_{\mathrm{wp}}),\, 0,\, 1\bigr)$:

$$
\alpha_{\mathrm{moist}} =
\begin{cases}
\alpha_{\min} + (\alpha_{\mathrm{PT}} - \alpha_{\min})\dfrac{\theta_{\mathrm{rel}}}{\theta_{\mathrm{stress}}}, & \theta_{\mathrm{rel}} < \theta_{\mathrm{stress}} \\
\alpha_{\mathrm{PT}}, & \theta_{\mathrm{rel}} \ge \theta_{\mathrm{stress}}
\end{cases}, \qquad
\alpha \leftarrow \min(\alpha, \alpha_{\mathrm{moist}})
$$

2. If `vapor_pressure_deficit` is passed in `kwargs`, or read from `state` when it is a declared input (Section 2.2.1) (converted from $\mathrm{Pa}$ to $\mathrm{kPa}$ via `UnitConverter`):

$$
f_{\mathrm{vpd}} = \min\!\left(\max\!\left(b_{\mathrm{vpd}} - k_{\mathrm{vpd}}(\mathrm{VPD}_{\mathrm{kPa}} - v_0),\; \frac{\alpha_{\min}}{\alpha_{\mathrm{PT}}}\right),\; f_{\max,1}\right), \qquad
\alpha \leftarrow \alpha\,f_{\mathrm{vpd}}
$$

3. Final bound: $\alpha = \operatorname{clip}(\alpha,\, \alpha_{\min},\, \alpha_{\max})$.

#### 2.2.1 Declared Dynamic-$\alpha$ Inputs and Moisture Frame (`dynamic_alpha_inputs`, `dynamic_alpha_moisture_frame`)

Purpose: let a graph declare the drivers of the dynamic $\alpha$ as ports and describe the soil of the moisture modifier with the same thresholds as the soil store it reads, so that every reader of the store describes one soil (one-soil frame consistency of the paper-01 composition `C15_soil_water_feedback_on_emission_one_soil`; Manabe, 1969, bucket).

`dynamic_alpha_inputs` is a list of distinct names from (`soil_moisture`, `vapor_pressure_deficit`); an absent key declares none and keeps the undeclared route unchanged ($\theta$ from `state["soil_water_content"]` and VPD from the keyword arguments, each applied only when present). A declared name is added to the input variables, required in the state and finite, and requires `hydrosphere.evapotranspiration.enable_dynamic_alpha` true (otherwise `ValueError`):

- `soil_moisture`: the volumetric liquid water content $\theta$ ($\mathrm{m^3\,m^{-3}}$) of the soil store (the `soil_moisture` state of the soil water model, for example `BucketSoilMoisture`), read instead of `soil_water_content`; in a graph it is the store at the start of the step (an explicit, start-of-step evaluation).
- `vapor_pressure_deficit`: the atmospheric VPD $D$ ($\mathrm{Pa}$) of that contract, read from the state instead of the keyword argument; passing it also as a keyword argument raises.

`dynamic_alpha_moisture_frame` selects the thresholds of $\theta_{\mathrm{rel}}$:

$$
\text{porosity\_fractions (default):} \ \theta_{\mathrm{fc}} = f_{\mathrm{fc}}\,\theta_s, \quad \theta_{\mathrm{wp}} = f_{\mathrm{wp}}\,\theta_r; \qquad
\text{bucket\_soil\_moisture:} \ \theta_{\mathrm{fc}} = \theta_{\mathrm{fc,bucket}}, \quad \theta_{\mathrm{wp}} = \theta_{\mathrm{wp,bucket}}
$$

where $\theta_{\mathrm{fc,bucket}}$ and $\theta_{\mathrm{wp,bucket}}$ are the owner parameters `hydrosphere.soil_moisture.field_capacity` and `hydrosphere.soil_moisture.wilting_point` that `BucketSoilMoisture` reads for its ET limiter, drainage threshold and clip (`hydrosphere_bucket.md`). `bucket_soil_moisture` requires `soil_moisture` in `dynamic_alpha_inputs`, declares the two parameters only under this frame, casts them to the device and dtype of $\theta$, and checks $0 \le \theta_{\mathrm{wp}} < \theta_{\mathrm{fc}}$ at construction and on every call (a learnable binding may change after construction). The modifiers of items 1-3 are unchanged. Absent keys keep the previous behaviour bitwise.

### 2.3 Evapotranspiration and Latent Heat Ports

$$
\mathrm{ET}_{\mathrm{raw}} = \alpha\,\frac{\Delta_\phi}{\Delta_\phi + \gamma_\phi}\,\frac{A}{\lambda}, \qquad
\mathrm{ET} = \max(\mathrm{ET}_{\mathrm{raw}},\; q_{\min,0}), \qquad
L_E = \lambda\,\mathrm{ET}
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $R_n$ | Surface net radiation (required) | $\mathrm{W\,m^{-2}}$ | `net_radiation` | `forcing` |
| $T$ | Near-surface air temperature (required) | $\mathrm{K}$ | `air_temperature` | `forcing` |
| $P$ | Atmospheric pressure (required) | $\mathrm{Pa}$ | `atmospheric_pressure` | `forcing` |
| $G$ | Surface soil heat flux (required when `soil_heat_flux_source="state"`) | $\mathrm{W\,m^{-2}}$ | `soil_heat_flux` | `forcing` |
| $\theta$ | Optional dynamic-$\alpha$ soil water content key in `state` | $\mathrm{m^3\,m^{-3}}$ | `soil_water_content` | `state` |
| $\theta$ | Declared dynamic-$\alpha$ soil water content of the store (`dynamic_alpha_inputs` contains `soil_moisture`) | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `state` |
| $\mathrm{VPD}$ | Declared dynamic-$\alpha$ vapour pressure deficit (`dynamic_alpha_inputs` contains `vapor_pressure_deficit`) | $\mathrm{Pa}$ | `vapor_pressure_deficit` | `forcing` |
| $\mathrm{ET}$ | Nonnegative evapotranspiration mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `evapotranspiration_flux` | `flux` |
| $L_E$ | Surface latent heat flux ($\lambda\,\mathrm{ET}$) | $\mathrm{W\,m^{-2}}$ | `latent_heat_flux` | `flux` |
| $\mathcal{D}_{\mathrm{PT}}$ | Nested equilibrium flux, phase latent heat, and negative-ET projection residual dictionary | mixed SI | `diagnostics.evapotranspiration` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\alpha_{\mathrm{PT}}, \alpha_{\min}, \alpha_{\max}$ | `hydrosphere.evapotranspiration.alpha_pt`, `alpha_min`, `alpha_max` | $\mathrm{1}$ | `hydrosphere/evapotranspiration.yaml` | Priestley and Taylor (1972) |
| $s_G, f_G, p_{\mathrm{ice}}$ | `hydrosphere.evapotranspiration.soil_heat_flux_source`, `soil_heat_fraction`, `sub_freezing_phase` | `str`, $\mathrm{1}$, `str` | `hydrosphere/evapotranspiration.yaml` | Allen et al. (1998); Murray (1967) |
| $b_{\alpha}, \theta_{\mathrm{stress}}$ | `hydrosphere.evapotranspiration.enable_dynamic_alpha`, `stress_threshold` | `bool`, $\mathrm{1}$ | `hydrosphere/evapotranspiration.yaml` | Flint and Childs (1991) |
| $\theta_s, \theta_r, f_{\mathrm{fc}}, f_{\mathrm{wp}}$ | `hydrosphere.soil_moisture.porosity`, `residual_water_content`, `hydrosphere.physical.field_capacity_fraction`, `wilting_point_fraction` | $\mathrm{m^3\,m^{-3}}$, $\mathrm{1}$ | `hydrosphere/soil_moisture.yaml`, `physical.yaml` | Flint and Childs (1991) |
| $\theta_{\mathrm{fc,bucket}}, \theta_{\mathrm{wp,bucket}}$ | `hydrosphere.soil_moisture.field_capacity`, `hydrosphere.soil_moisture.wilting_point` (`dynamic_alpha_moisture_frame: bucket_soil_moisture` only) | $\mathrm{m^3\,m^{-3}}$ | `hydrosphere/soil_moisture.yaml` ($0.3, 0.1$) | Manabe (1969); shared with `BucketSoilMoisture` |
| $b_{\mathrm{vpd}}, k_{\mathrm{vpd}}, v_0$ | `hydrosphere.physical.vpd_alpha_baseline`, `vpd_alpha_sensitivity`, `vpd_alpha_threshold` | $\mathrm{1}$, $\mathrm{kPa^{-1}}$, $\mathrm{kPa}$ | `hydrosphere/physical.yaml` | de Bruin (1983) |
| $P_{\mathrm{ref}}, T_{\mathrm{ref,H}}$ | `hydrosphere.evapotranspiration.hodge_reference_atmospheric_pressure`, `hodge_reference_air_temperature` | $\mathrm{Pa}$, $\mathrm{K}$ | `hydrosphere/evapotranspiration.yaml` | RELIC Hodge scaling |
| $P_{\min}, P_{\max}, T_{\mathrm{bounds}}, \delta_{R_n}$ | `hydrosphere.physical.atmospheric_pressure_min`, `atmospheric_pressure_max`, `hydrosphere.numerical.temperature_bounds`, `net_radiation_validation_offset` | $\mathrm{Pa}$, $\mathrm{K}$, $\mathrm{W\,m^{-2}}$ | `hydrosphere/physical.yaml`, `numerical.yaml` | RELIC domain validation |
| $q_{\min,\mathrm{ET}}, q_{d,\min}, q_{\min,0}, f_{\max,1}$ | `hydrosphere.numerical.evapotranspiration_flux_clamp_min`, `minimum_drainage_flux`, `validation.numerical.clamp_min_zero`, `clamp_max_one` | $\mathrm{kg\,m^{-2}\,s^{-1}}$, $\mathrm{1}$ | `hydrosphere/numerical.yaml`, `validation/numerical.yaml` | RELIC numerical bounds |

## 5. Numerical Methods and Discretization

`_ensure_tensor` preserves float64 input precision and promotes lower-precision floating tensors to at least float32 through the existing FunctorBase device conversion, retaining the autograd path. Python scalars and non-floating tensors use `torch.float32`. Float64 drivers therefore support double-precision finite differences of learnable parameters through the same production expression. Default `temperature_bounds` ($[273.15, 323.15]\,\mathrm{K}$) reject sub-freezing temperatures unless the shared bound is explicitly lowered in configuration. Outputs `evapotranspiration_flux`, `latent_heat_flux`, and `diagnostics.evapotranspiration` use `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`.

## 6. Conservation, Boundary Conditions, and Ledgers

`PriestleyTaylorET` owns no water storage pool. Its open transport-energy coupling checks exact phase-consistent latent-heat closure $L_E = \lambda\,\mathrm{ET}$ cell by cell (using $L_s$ where $T < T_f$ in `"ice"` mode and $L_v$ elsewhere) and routes negative available-energy projection residuals to `diagnostics.evapotranspiration`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `PriestleyTaylorET` | `relic/scientific/hydrosphere/evapotranspiration/priestley_taylor.py` | `TransportMechanism` (`TRANSPORT_TYPE="gas_exchange"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `PriestleyTaylorET`: `[I]`, IPHS **Level 3** (`transport`, certificate `hydrosphere_evapotranspiration_priestley_taylor_priestley_taylor_et.yaml`). Equilibrium vapor operator $q_{\mathrm{eq}} = \star_{\mathrm{PT}} A$ coupled with energy port $L_E = \lambda q_{\mathrm{eq}}$.

Frozen certificate expressions:

```text
ET = alpha * (Delta/(Delta+gamma)) * available_energy / lambda, then ET is projected to nonnegative flux.
q_eq = star_PT A plus energy port L_E = lambda q_eq; nonnegative projection and energy residuals are routed as diagnostics.
```

## 9. References

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements* (FAO Irrigation and Drainage Paper 56). FAO.
- de Bruin, H. A. R. (1983). A model for the Priestley-Taylor parameter $\alpha$. *Journal of Climate and Applied Meteorology*, 22(4), 572-578.
- Flint, A. L., and Childs, S. W. (1991). Use of the Priestley-Taylor evaporation equation for soil water limited conditions in a small forest clearcut. *Agricultural and Forest Meteorology*, 56(3-4), 247-260.
- Manabe, S. (1969). Climate and the ocean circulation 1: The atmospheric circulation and the hydrology of the earth's surface. *Monthly Weather Review*, 97(11), 739-774.
- Murray, F. W. (1967). On the computation of saturation vapor pressure. *Journal of Applied Meteorology*, 6(1), 203-204.
- Priestley, C. H. B., and Taylor, R. J. (1972). On the assessment of surface heat flux and evaporation using large-scale parameters. *Monthly Weather Review*, 100(2), 81-92.
