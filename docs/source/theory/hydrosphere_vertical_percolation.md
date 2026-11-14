# Hydrosphere Vertical Percolation Transport

## 1. Process Definition and Scope

`VerticalPercolationTransport` (`relic/scientific/hydrosphere/transport/vertical_percolation.py`) is a reduced-order gravity-drainage transport Functor that computes downward bottom-boundary `drainage_flux` ($\mathrm{kg\,m^{-2}\,s^{-1}}$) from field-capacity moisture excess and selectable unsaturated hydraulic conductivity (`HydraulicModel`), with optional liquid-viscosity temperature scaling.

The Functor owns only its open bottom-drainage flux calculation and advances no soil-water storage pool (`BucketSoilMoisture` and `RichardsSoilMoisture` are alternative prognostic storage + drainage owners). Multi-layer matric-potential gradient solving and lateral interflow splitting inside `VerticalPercolationTransport` remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Section 19.8.1).

## 2. Governing Equations

### 2.1 Effective Saturation and Selectable Hydraulic Conductivity

For volumetric `soil_moisture` $\theta \in [\theta_r, \theta_s]$ with effective saturation $S_e = (\theta - \theta_r)/(\theta_s - \theta_r)$ regularized by $\epsilon_S$ (`saturation_epsilon`), `hydrosphere.transport.vertical_percolation.hydraulic_model` selects one `HydraulicModel` closure:

- **`"van_genuchten_mualem"`** ($m = 1 - 1/n$, tortuosity $l$, delegating to `van_genuchten_mualem_conductivity` in `relic/utils/soil_hydraulics.py`; Mualem, 1976; van Genuchten, 1980):

$$
K_0(S_e) = K_s\,S_e^l\left[1 - \left(1 - S_e^{1/m}\right)^m\right]^2
$$

- **`"brooks_corey"`** (pore-size index $\lambda_p$; Brooks and Corey, 1964):

$$
K_0(S_e) = K_s\,S_e^{(2 + 3\lambda_p)/\lambda_p}
$$

- **`"clapp_hornberger"`** (exponent $b$; Clapp and Hornberger, 1978):

$$
K_0(S_e) = K_s\,S_e^{2b + 3}
$$

- **`"simple_exponential"`** (retained cubic power-law closure in code):

$$
K_0(S_e) = K_s\,S_e^3
$$

### 2.2 Viscosity Temperature Correction and Field-Capacity Drainage Law

When `enable_temperature_effects` is active with `soil_temperature` $T$ ($\mathrm{K}$), reference temperature $T_{\mathrm{ref}}$, and viscosity factor $\beta_T$ ($\mathrm{K^{-1}}$):

$$
K(\theta, T) =
\begin{cases}
K_0(S_e)\exp\bigl[\beta_T(T - T_{\mathrm{ref}})\bigr], & \text{temperature effects enabled} \\
K_0(S_e), & \text{temperature effects disabled}
\end{cases}
$$

Downward bottom-drainage mass flux $F_{\mathrm{drain}}$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) over layer thickness $\Delta z$ ($\mathrm{m}$) and drainage time scale $\tau_{\mathrm{drain}}$ ($\mathrm{s}$) is:

$$
F_{\mathrm{drain}} = \frac{\max(0,\; \theta - \theta_{\mathrm{fc}})}{\tau_{\mathrm{drain}}}\,\frac{K(\theta, T)}{K_s}\,\rho_w\,\Delta z
$$

Structural mobility audit uses positive-definite Darcy Hodge operator $\star_K = \operatorname{diag}(\rho_w K_s)$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$), satisfying $\sigma = g_h^{\mathsf{T}}\star_K g_h \ge 0$.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\theta$ | Volumetric soil moisture ($\theta_r \le \theta \le \theta_s$, required) | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `state` |
| $T$ | Soil temperature (within `temperature_bounds`, required) | $\mathrm{K}$ | `soil_temperature` | `state` |
| $F_{\mathrm{drain}}$ | Downward soil-column bottom drainage mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `drainage_flux` | `flux` |
| $S_e, K, K/K_s, \theta_{\mathrm{ex}}, r_w$ | Effective saturation, hydraulic conductivity, relative conductivity, field-capacity excess, and open-boundary verification residual | $\mathrm{1}$, $\mathrm{m\,s^{-1}}$, $\mathrm{1}$, $\mathrm{m^3\,m^{-3}}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.vertical_percolation.effective_saturation`, `hydraulic_conductivity`, `relative_hydraulic_conductivity`, `field_capacity_excess`, `water_balance_residual` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $b_T, m_{\mathrm{hyd}}$ | `hydrosphere.transport.vertical_percolation.enable_temperature_effects`, `hydraulic_model` | `bool`, `str` | `hydrosphere/transport.yaml` | Model selector |
| $\theta_s, \theta_r, \theta_{\mathrm{fc}}, \theta_{\mathrm{wp}}, Z_{\mathrm{root}}$ | `hydrosphere.soil_moisture.porosity`, `residual_water_content`, `field_capacity`, `wilting_point`, `root_zone_depth` | $\mathrm{m^3\,m^{-3}}$, $\mathrm{m}$ | `hydrosphere/soil_moisture.yaml` | Allen et al. (1998); Hillel (1998) |
| $K_s, \alpha_{\mathrm{vg}}, n$ | `hydrosphere.soil_moisture.saturated_hydraulic_conductivity`, `alpha_vg`, `n_vg` | $\mathrm{m\,s^{-1}}$, $\mathrm{m^{-1}}$, $\mathrm{1}$ | `hydrosphere/soil_moisture.yaml` | van Genuchten (1980) |
| $l, \lambda_p, b$ | `hydrosphere.transport.vg_l`, `bc_lambda`, `ch_b_parameter` | $\mathrm{1}$ | `hydrosphere/transport.yaml` | Mualem (1976); Brooks and Corey (1964); Clapp and Hornberger (1978) |
| $\tau_{\mathrm{drain}}, \Delta z$ | `hydrosphere.transport.drainage_time_constant`, `soil_layer_thickness` | $\mathrm{s}$, $\mathrm{m}$ | `hydrosphere/transport.yaml` | Reduced-order drainage scale |
| $\beta_T$ | `hydrosphere.physical.viscosity_temperature_factor` | $\mathrm{K^{-1}}$ | `hydrosphere/physical.yaml` | Liquid water viscosity sensitivity |
| $T_{\mathrm{bounds}}, \epsilon_S, S_{e,\mathrm{bounds}}$ | `hydrosphere.numerical.temperature_bounds`, `saturation_epsilon`, `effective_saturation_bounds` | $\mathrm{K}$, $\mathrm{1}$ | `hydrosphere/numerical.yaml` | RELIC domain guards |
| $\rho_w, T_{\mathrm{ref}}$ | `relic.physical.constants.water_density`, `relic.physical.constants.reference_temperature` | $\mathrm{kg\,m^{-3}}$, $\mathrm{K}$ | `physical/constants.yaml` | IAPWS / standard reference |

## 5. Numerical Methods and Discretization

`VerticalPercolationTransport` validates non-empty floating `torch.Tensor` inputs strictly inside $[\theta_r, \theta_s]$ and `temperature_bounds` without clipping or fallback model chains. Output `drainage_flux` and all five flat `diagnostics.vertical_percolation.*` ports use `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`.

## 6. Conservation, Boundary Conditions, and Ledgers

The Functor declares open boundary ledger `vertical_drainage_boundary` ($R_{\mathrm{boundary}} = F_{\mathrm{computed}} - F_{\mathrm{exported}} = 0$). `VerticalPercolationTransport.check_conservation_laws(...)` independently recomputes constitutive drainage from the input state and checks both the exported `drainage_flux` and the reported `water_balance_residual` diagnostic so tampered outputs cannot self-attest zero error.

## 7. Code Mapping

| Functor / Enum Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `VerticalPercolationTransport` | `relic/scientific/hydrosphere/transport/vertical_percolation.py` | `TransportMechanism` (`TRANSPORT_TYPE="vertical_percolation"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |
| `HydraulicModel` | `relic/scientific/hydrosphere/transport/vertical_percolation.py` | Conductivity closure enumeration | Used by `VerticalPercolationTransport` | Construction validation |

## 8. Implementation Status

- `VerticalPercolationTransport`: `[I]`, IPHS **Level 3** (`transport`, certificate `hydrosphere_transport_vertical_percolation_vertical_percolation_transport.yaml`). Reduced-order gravitational drainage transport with positive Darcy Hodge audit $\sigma = g_h^{\mathsf{T}}\star_K g_h \ge 0$ and independent open-port conservation verification.

Frozen certificate expressions:

```text
F_drain = max(0, theta - theta_fc) * (K/Ks) * rho_w * dz / tau_drain.
theta_dot = -D_h q_h + s_h; q_h = star_K g_h; sigma = g_h.T star_K g_h >= 0.
```

## 9. References

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements* (FAO Irrigation and Drainage Paper 56). FAO.
- Brooks, R. H., and Corey, A. T. (1964). *Hydraulic Properties of Porous Media* (Hydrology Paper No. 3). Colorado State University.
- Buckingham, E. (1907). *Studies on the Movement of Soil Moisture* (Bulletin No. 38). USDA Bureau of Soils.
- Clapp, R. B., and Hornberger, G. M. (1978). Empirical equations for some soil hydraulic properties. *Water Resources Research*, 14(4), 601-604.
- Darcy, H. (1856). *Les Fontaines publiques de la ville de Dijon*. Victor Dalmont.
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. *Water Resources Research*, 12(3), 513-522.
- Richards, L. A. (1931). Capillary conduction of liquids through porous mediums. *Physics*, 1(5), 318-333.
- van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. *Soil Science Society of America Journal*, 44(5), 892-898.
