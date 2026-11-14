# Bucket Soil Moisture Balance

## 1. Process Definition and Scope

`BucketSoilMoisture` (`relic/scientific/hydrosphere/soil_moisture/bucket.py`) advances a single-layer root-zone volumetric soil moisture reservoir $\theta \in [\theta_{\mathrm{wp}}, \theta_{\mathrm{sat}}]$ through a bounded sequential update of surface water influx, extractable evapotranspiration, saturation-excess surface runoff, and power-law gravity drainage.

Compile-time parameter `hydrosphere.soil_moisture.water_input_source` selects one surface water supply contract: `"precipitation_flux"` binds `precipitation_flux` directly, whereas `"rain_flux_plus_snow_melt_flux"` replaces `precipitation_flux` in `FunctorMetadata.input_variables` with `rain_flux` and `snow_melt_flux` and registers a lagged dependency on `snow_pack`. Multilayer root-zone partitioning, capillary groundwater rise coupling, and Hortonian infiltration-excess ponding inside the bucket closure remain `[N]` (handled instead by `RichardsSoilMoisture` and `SurfacePondingFunctor`). Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Sections 19.6 and 19.10).

## 2. Governing Equations

### 2.1 Continuous Volume-Balance Form and Water Input Source

Over effective root-zone depth $Z$ ($\mathrm{m}$) with liquid water density $\rho_w$ ($\mathrm{kg\,m^{-3}}$) and surface mass influx $W_{\mathrm{in}}$ ($P$ under `"precipitation_flux"`, or $P_{\mathrm{rain}} + M_{\mathrm{snow}}$ under `"rain_flux_plus_snow_melt_flux"`, all in $\mathrm{kg\,m^{-2}\,s^{-1}}$) (Manabe, 1969; Milly, 1994):

$$
\frac{d\theta}{dt} = \frac{W_{\mathrm{in}} - \mathrm{ET}_{\mathrm{act}} - Q - D_m}{\rho_w Z}
$$

### 2.2 Sequential Discrete Update Over Step $\Delta t$

1. **Extractable evapotranspiration bound** from start-of-step storage above wilting point $\theta_{\mathrm{wp}}$:

$$
\mathrm{ET}_{\mathrm{act}} = \min\!\left(\mathrm{ET},\; \frac{\max(\theta^n - \theta_{\mathrm{wp}}, 0)\,\rho_w Z}{\Delta t}\right)
$$

The production call requires finite, non-negative water influx and evapotranspiration demand. The extraction cap uses start-of-step soil water. Water influx enters the subsequent provisional-moisture update.

2. **Provisional moisture and saturation-excess runoff** above porosity $\theta_{\mathrm{sat}}$:

$$
\theta_p = \theta^n + \frac{(W_{\mathrm{in}} - \mathrm{ET}_{\mathrm{act}})\Delta t}{\rho_w Z}, \qquad
Q = \frac{\max(\theta_p - \theta_{\mathrm{sat}}, 0)\,\rho_w Z}{\Delta t}, \qquad
\theta_r = \min(\theta_p, \theta_{\mathrm{sat}})
$$

3. **Power-law gravity drainage** bounded by water stored above field capacity $\theta_{\mathrm{fc}}$ (with zero minimum drainage floor below $\theta_{\mathrm{fc}}$):

$$
D_{p,\mathrm{vol}} = f_D K_{\mathrm{sat}}\left(\frac{\theta_r}{\theta_{\mathrm{sat}}}\right)^b, \qquad
D_{\mathrm{vol}} = \min\!\left(D_{p,\mathrm{vol}},\; \frac{\max(\theta_r - \theta_{\mathrm{fc}}, 0)\,Z}{\Delta t}\right), \qquad
D_m = \rho_w D_{\mathrm{vol}}
$$

4. **End-of-step volumetric moisture**:

$$
\theta^{n+1} = \operatorname{clip}\!\left(\theta_r - \frac{D_{\mathrm{vol}}\Delta t}{Z},\; \theta_{\mathrm{wp}},\; \theta_{\mathrm{sat}}\right)
$$

### 2.3 Diagnostic Quantities

Start-of-step moisture $\theta^n$ defines plant-accessibility observations, whereas end-of-step moisture $\theta^{n+1}$ defines updated storage observations:

$$
\mathrm{AWC}^n = \max(0, \theta^n - \theta_{\mathrm{wp}}), \qquad
\mathrm{RWC}^n = \operatorname{clip}\!\left(\frac{\theta^n - \theta_{\mathrm{wp}}}{\theta_{\mathrm{fc}} - \theta_{\mathrm{wp}}},\; 0,\; 1\right), \qquad
f_{w}^n = \mathrm{RWC}^n
$$

$$
S_{\mathrm{def}}^{n+1} = \theta_{\mathrm{sat}} - \theta^{n+1}, \qquad
S_{w}^{n+1} = \theta^{n+1}\,\rho_w Z
$$

The reported `saturation_deficit` is a volumetric water fraction in $\mathrm{m^3\,m^{-3}}$. The reported `soil_water_storage` is an areal mass in $\mathrm{kg\,m^{-2}}$.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\theta$ | Volumetric root-zone soil moisture (start-of-step input and end-of-step replacement) | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `state` |
| $P$ | Total precipitation mass flux (`water_input_source="precipitation_flux"`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `precipitation_flux` | `forcing` |
| $P_{\mathrm{rain}}, M_{\mathrm{snow}}$ | Liquid rain and snowmelt mass fluxes (`water_input_source="rain_flux_plus_snow_melt_flux"`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `rain_flux`, `snow_melt_flux` | `flux` |
| $\mathrm{ET}$ | Potential or upstream evapotranspiration demand | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `evapotranspiration_flux` | `flux` |
| $Q$ | Saturation-excess surface runoff mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `surface_runoff_flux` | `flux` |
| $D_m$ | Deep gravity drainage mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `drainage_flux` | `flux` |
| $\mathcal{D}_{\mathrm{bucket}}$ | Nested diagnostic dictionary (`water_stress_factor`, `saturation_deficit`, `available_water_capacity`, `relative_water_content`, `drainage_rate`, `runoff_rate`, `net_water_input`, `adjusted_evapotranspiration`, `soil_moisture_change_rate`, `soil_water_storage`, `clipped_mass_residual`, `drainage_limiter_residual`) | mixed SI | `diagnostics.soil_moisture_bucket` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $Z$ | `hydrosphere.soil_moisture.soil_depth` | $\mathrm{m}$ | `hydrosphere/soil_moisture.yaml` | Manabe (1969); Milly (1994) |
| $\theta_{\mathrm{fc}}$ | `hydrosphere.soil_moisture.field_capacity` | $\mathrm{m^3\,m^{-3}}$ | `hydrosphere/soil_moisture.yaml` | Milly (1994) |
| $\theta_{\mathrm{wp}}$ | `hydrosphere.soil_moisture.wilting_point` | $\mathrm{m^3\,m^{-3}}$ | `hydrosphere/soil_moisture.yaml` | Manabe (1969) |
| $\theta_{\mathrm{sat}}$ | `hydrosphere.soil_moisture.porosity` | $\mathrm{m^3\,m^{-3}}$ | `hydrosphere/soil_moisture.yaml` | Clapp and Hornberger (1978) |
| $K_{\mathrm{sat}}$ | `hydrosphere.soil_moisture.saturated_hydraulic_conductivity` | $\mathrm{m\,s^{-1}}$ | `hydrosphere/soil_moisture.yaml` | Clapp and Hornberger (1978) |
| $f_D$ | `hydrosphere.soil_moisture.drainage_factor` | $\mathrm{1}$ | `hydrosphere/soil_moisture.yaml` | Calibratable free-drainage scale |
| $b$ | `hydrosphere.soil_moisture.drainage_exponent` | $\mathrm{1}$ | `hydrosphere/soil_moisture.yaml` | Campbell (1974); Clapp and Hornberger (1978) |
| $s_{\mathrm{in}}$ | `hydrosphere.soil_moisture.water_input_source` | `str` | `hydrosphere/soil_moisture.yaml` | RELIC single-provider input contract |
| $\rho_w$ | `relic.physical.constants.water.density_liquid` | $\mathrm{kg\,m^{-3}}$ | `physical/constants.yaml` | CRC Handbook |

## 5. Numerical Methods and Discretization

`BucketSoilMoisture` declares `SUBFLOW_ACCURACY = SubFlowAccuracy.ORDER_1` and `CONFLICT_RESOLUTION = ConflictResolutionPolicy.EXCLUSIVE`, with lagged dependencies on `penman_monteith_et` and `priestley_taylor_et` (plus lagged `snow_pack` when `water_input_source == "rain_flux_plus_snow_melt_flux"`). On the smooth uncapped drainage branch ($W_{\mathrm{in}} = \mathrm{ET} = 0$, $\theta_{\mathrm{fc}} < \theta < \theta_{\mathrm{sat}}$), $\dot{\theta} = -k\theta^b$ with $k = f_D K_{\mathrm{sat}} / (Z\theta_{\mathrm{sat}}^b)$ is integrated by explicit forward Euler ($\theta^{n+1} = \theta^n - \Delta t\,k(\theta^n)^b$), carrying local truncation error $-\frac{1}{2}b k^2 \theta^{2b-1}\Delta t^2 + \mathcal{O}(\Delta t^3)$ and global order 1 (Hairer et al., 1993). All outputs are emitted as `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`.

## 6. Conservation, Boundary Conditions, and Ledgers

After the final moisture projection, the production call adds the projection residual to drainage and clips the corrected drainage at zero. `drainage_limiter_residual` records potential minus corrected drainage in $\mathrm{m\,s^{-1}}$. `clipped_mass_residual` records any remaining imbalance in $\mathrm{kg\,m^{-2}\,s^{-1}}$. The actual water ledger therefore satisfies

$$
r_{\mathrm{mass}} = \rho_w Z(\theta^{n+1} - \theta^n) - (W_{\mathrm{in}} - \mathrm{ET}_{\mathrm{act}} - Q - D_m)\Delta t
= -\mathtt{clipped\_mass\_residual}\,\Delta t.
$$

For a start state in $[\theta_{\mathrm{wp}},\theta_{\mathrm{sat}}]$, ordered hydraulic thresholds and non-negative admitted forcing, the sequential storage limits preserve that interval and the residual is zero in exact arithmetic. The runtime accepts the wider input interval $[0,1]$ and exposes any remaining projection imbalance through this diagnostic. A nonzero diagnostic discloses the unclosed amount. Floating-point ledger checks use the existing conservation owners and their configured tolerances.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `BucketSoilMoisture` | `relic/scientific/hydrosphere/soil_moisture/bucket.py` | `TransportMechanism` (`TRANSPORT_TYPE="vertical_percolation"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `BucketSoilMoisture`: `[I]`, IPHS **Level 3** (`transport`, certificate `hydrosphere_soil_moisture_bucket_bucket_soil_moisture.yaml`). Dissipative transport audit $\sigma = X_h^{\mathsf{T}}\star_K X_h \ge 0$ with exact open-port water balance closure; strict runtime Hamiltonian replacement is not claimed.
- Multilayer bucket cascades and subgrid variable-infiltration curves inside `BucketSoilMoisture`: `[N]`.

Frozen certificate expressions:

```text
theta_new = clamp(theta + (P - ET - Q - D) * dt / (rho_w * Z), theta_wp, theta_sat), with flux correction for mass closure.
theta_dot = (P - ET - Q - D) / (rho_w * Z); sigma = X_h.T star_K X_h >= 0.
```

## 9. References

- Campbell, G. S. (1974). A simple method for determining unsaturated conductivity from moisture retention data. *Soil Science*, 117(6), 311-314.
- Clapp, R. B., and Hornberger, G. M. (1978). Empirical equations for some soil hydraulic properties. *Water Resources Research*, 14(4), 601-604.
- Hairer, E., Norsett, S. P., and Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems* (2nd ed.). Springer.
- Manabe, S. (1969). Climate and the ocean circulation 1: The atmospheric circulation and the hydrology of the earth's surface. *Monthly Weather Review*, 97(11), 739-774.
- Milly, P. C. D. (1994). Climate, soil water storage, and the average annual water balance. *Water Resources Research*, 30(7), 2143-2156.
