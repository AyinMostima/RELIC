# Hydrosphere Capillary Rise Transport

## 1. Process Definition and Scope

`CapillaryRiseTransport` (`relic/scientific/hydrosphere/transport/capillary_rise.py`) computes steady-state upward liquid water mass flux from a shallow water table toward the root zone and soil surface following Gardner (1958) and Eagleson (1978), supporting both Brooks-Corey and van Genuchten hydraulic parameterizations.

The Functor declares a lagged dependency on `groundwater_dynamics` so start-of-step `water_table_depth` drives the current boundary transfer while `GroundwaterDynamics` and `RichardsSoilMoisture` consume `capillary_rise_flux` within the same step without cyclic dependencies. Transient unsaturated Richards capillary waves inside this module and hysteretic wetting-drying retention curves remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Section 19.8.2).

## 2. Governing Equations

### 2.1 Hydraulic Parameterization Branches

Parameter `hydrosphere.transport.hydraulic_parameterization_method` selects the Gardner power-law conductivity parameters $(a, b)$ in $K(\psi) = a\lvert\psi\rvert^{-b}$ and shape factor $B = 1 + \frac{3}{2(b - 1)}$:

- **`"brooks_corey"`** (with pore-size index $\lambda$ and air-entry head $\psi_{\mathrm{ae}}$):

$$
b = 2 + 3\lambda, \qquad a = K_s\,\lvert\psi_{\mathrm{ae}}\rvert^{b}
$$

- **`"van_genuchten"`** (with shape exponent $n > 1$, $m = 1 - 1/n$, and inverse air-entry parameter $\alpha_{\mathrm{vg}}$):

$$
b = \frac{2}{m}, \qquad a = K_s\left(\frac{1}{\alpha_{\mathrm{vg}}}\right)^b
$$

### 2.2 Surface Boundary Flux and Conductivity Clipping

For finite nonnegative water-table depth $Z$ ($\mathrm{m}$) and maximum capillary height $Z_{\max}$ ($\mathrm{m}$), raw candidate volumetric velocity $w$ and bounded volumetric cochain $u_{\mathrm{vol}}$ ($\mathrm{m\,s^{-1}}$) are:

$$
w(Z) =
\begin{cases}
\dfrac{B\,a}{Z^b}, & 0 < Z \le Z_{\max} \\
0, & Z = 0 \text{ or } Z > Z_{\max}
\end{cases}, \qquad
u_{\mathrm{vol}} = \operatorname{clip}(w(Z),\; 0,\; K_s)
$$

Hodge scaling $\star_{\mathrm{cap}} = \operatorname{diag}(\rho_w)$ maps $u_{\mathrm{vol}}$ to upward boundary mass flux $u_{\mathrm{cap}}$ and conductivity-cap residual $r_K$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$):

$$
u_{\mathrm{cap}} = \rho_w u_{\mathrm{vol}}, \qquad
r_K = \rho_w \max\bigl(w(Z) - K_s,\; 0\bigr)
$$

### 2.3 Layered Diagnostic Profile and Saturation Constraint

When `enable_capillary_depth_profile` is active across $N_L$ layers of thickness $\Delta z$ (layer depth $z_k < Z$) with surface porosity $n_0$ and porosity decay length $p$, the unconstrained diagnostic mass-flux profile is:

$$
q_{\mathrm{prof}}(z_k) = \rho_w\,\operatorname{clip}\!\left(\frac{B\,a}{(Z - z_k)^b},\; 0,\; K_s\right)\exp\!\left(-\frac{z_k}{p}\right)
$$

When optional `soil_moisture_profile` $\theta_k$ (length $N_L$) is supplied, saturation inhibition $f_{\mathrm{sat}}(\theta_k)$ and profile diagnostic residual $r_{\mathrm{sat}}$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) are:

$$
f_{\mathrm{sat}}(\theta_k) = \operatorname{clip}\!\left(\frac{c_{\mathrm{thr}} n_0 - \theta_k}{c_{\mathrm{bw}} n_0},\; 0,\; 1\right), \qquad
q_{\mathrm{lim}}(z_k) = f_{\mathrm{sat}}(\theta_k)\,q_{\mathrm{prof}}(z_k), \qquad
r_{\mathrm{sat}} = \max_k\bigl(q_{\mathrm{prof}}(z_k) - q_{\mathrm{lim}}(z_k)\bigr)
$$

Profile saturation inhibition affects only the diagnostic profile and leaves primary boundary port `capillary_rise_flux` unchanged.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $Z$ | Water-table depth below soil surface ($\ge 0$, required) | $\mathrm{m}$ | `water_table_depth` | `state` |
| $\theta_k$ | Optional layered volumetric soil moisture profile | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture_profile` | `state` |
| $u_{\mathrm{cap}}$ | Upward capillary boundary mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `capillary_rise_flux` | `flux` |
| $r_K$ | Conductivity-cap mass-flux residual | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.capillary_rise.conductivity_clipping_residual` | `diagnostic` |
| $r_{\mathrm{sat}}$ | Profile saturation-constraint diagnostic residual | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.capillary_rise.saturation_constraint_residual` | `diagnostic` |
| $\mathcal{D}_{\mathrm{cap}}$ | Nested read-only profile and operator metadata mapping | mixed SI | `diagnostics.capillary_rise` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $Z_{\max}$ | `hydrosphere.transport.max_capillary_height` | $\mathrm{m}$ | `hydrosphere/transport.yaml` | Eagleson (1978) |
| $m_{\mathrm{hyd}}$ | `hydrosphere.transport.hydraulic_parameterization_method` | `str` | `hydrosphere/transport.yaml` | `"brooks_corey"` or `"van_genuchten"` |
| $\lambda, \psi_{\mathrm{ae}}$ | `hydrosphere.transport.brooks_corey_pore_size_index`, `hydrosphere.transport.air_entry_pressure` | $\mathrm{1}$, $\mathrm{m}$ | `hydrosphere/transport.yaml` | Brooks and Corey (1964); Eagleson (1978) |
| $K_s, \alpha_{\mathrm{vg}}, n$ | `hydrosphere.soil_moisture.saturated_hydraulic_conductivity`, `hydrosphere.soil_moisture.alpha_vg`, `hydrosphere.soil_moisture.n_vg` | $\mathrm{m\,s^{-1}}$, $\mathrm{m^{-1}}$, $\mathrm{1}$ | `hydrosphere/soil_moisture.yaml` | van Genuchten (1980); Gardner (1958) |
| $n_0, p$ | `hydrosphere.groundwater.surface_porosity`, `hydrosphere.groundwater.porosity_decay_length` | $\mathrm{1}$, $\mathrm{m}$ | `hydrosphere/groundwater.yaml` | Wigmosta et al. (1994) |
| $c_{\mathrm{thr}}, c_{\mathrm{bw}}$ | `hydrosphere.transport.capillary_saturation_threshold_factor`, `hydrosphere.transport.capillary_saturation_bandwidth_factor` | $\mathrm{1}$ | `hydrosphere/transport.yaml` | RELIC profile limiter |
| $b_{\mathrm{prof}}, N_L, \Delta z$ | `hydrosphere.transport.enable_capillary_depth_profile`, `hydrosphere.transport.capillary_profile_layer_count`, `hydrosphere.transport.capillary_profile_layer_thickness` | `bool`, `int`, $\mathrm{m}$ | `hydrosphere/transport.yaml` | RELIC profile geometry |
| $\rho_w$ | `relic.physical.constants.water_density` | $\mathrm{kg\,m^{-3}}$ | `physical/constants.yaml` | IAPWS reference density |

## 5. Numerical Methods and Discretization

`CapillaryRiseTransport` evaluates steady-state algebraic expressions without $\Delta t$ integration and emits `capillary_rise_flux` and the two flat residual ports as `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`. Singularity at $Z = 0$ is prevented by masking evaluation strictly to $0 < Z \le Z_{\max}$. Legacy `groundwater_depth` aliases and non-finite or negative `water_table_depth` tensors raise `ValueError`.

## 6. Conservation, Boundary Conditions, and Ledgers

`CapillaryRiseTransport` emits the producer port of an open boundary water transfer (`out_of_system` from groundwater, `into_system` at the Richards bottom boundary) and owns no water storage pool. Its custom `mass` conservation check verifies the candidate-flux limiter identity $F_{\mathrm{cand}} = u_{\mathrm{cap}} + r_K - r_{\mathrm{neg}}$ in $\mathrm{kg\,m^{-2}\,s^{-1}}$; combined aquifer-to-vadose storage conservation closes when paired with `GroundwaterDynamics` and `RichardsSoilMoisture`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `CapillaryRiseTransport` | `relic/scientific/hydrosphere/transport/capillary_rise.py` | `TransportMechanism` (`TRANSPORT_TYPE="capillary_rise"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`, `spatial_dimensions=1`) |

## 8. Implementation Status

- `CapillaryRiseTransport`: `[I]`, IPHS **Level 3** (`transport`, certificate `hydrosphere_transport_capillary_rise_capillary_rise_transport.yaml`). Positive-definite Hodge dissipation $\sigma = u_{\mathrm{vol}}^{\mathsf{T}}\star_{\mathrm{cap}}u_{\mathrm{vol}} \ge 0$ and explicit open-port limiter accounting.

Frozen certificate expressions:

```text
w = B*a/Z^b; capillary_rise_flux = rho_w * clamp(w, 0, Ks).
m_dot = G_cap u_cap; u_cap = rho_w * max(0, min(Ks, B*a/Z^b)); sigma = u_vol.T star_cap u_vol >= 0 with u_vol the volumetric flux cochain [m s-1].
```

## 9. References

- Brooks, R. H., and Corey, A. T. (1964). *Hydraulic Properties of Porous Media* (Hydrology Paper No. 3). Colorado State University.
- Eagleson, P. S. (1978). Climate, soil, and vegetation: 3. A simplified model of soil moisture movement in the liquid phase. *Water Resources Research*, 14(5), 722-730.
- Gardner, W. R. (1958). Some steady-state solutions of the unsaturated moisture flow equation with application to evaporation from a water table. *Soil Science*, 85(4), 228-232.
- van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. *Soil Science Society of America Journal*, 44(5), 892-898.
- Wigmosta, M. S., Vail, L. W., and Lettenmaier, D. P. (1994). A distributed hydrology-vegetation model for complex terrain. *Water Resources Research*, 30(6), 1665-1679.
