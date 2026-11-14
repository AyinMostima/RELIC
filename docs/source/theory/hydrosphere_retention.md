# Soil Water Retention Observation

## 1. Process Definition and Scope

`SoilWaterRetention` (`relic/scientific/hydrosphere/soil_moisture/retention.py`) is a read-only point-field observation operator that converts volumetric `soil_moisture` ($\mathrm{m^3\,m^{-3}}$) into matric `soil_water_potential` ($\mathrm{MPa}$) by delegating retention inversion to `van_genuchten_pressure_head` in `relic/utils/soil_hydraulics.py` (documented in `soil_hydraulics.md`).

In coupled graphs where soil moisture is carried as a volumetric fraction (such as `BucketSoilMoisture`), `SoilWaterRetention` supplies the canonical `soil_water_potential` port consumed by `WaterStress`. Instance configuration accepts optional `soil_moisture_provider` (producer node name) and strict boolean `soil_moisture_lagged` (default `False`) to declare either same-step `required` or previous-step `lagged` scheduling order. Hysteretic wetting-drying curves and dual-porosity retention remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Sections 19.1.2 and 19.2.1).

## 2. Governing Equations

For finite volumetric soil moisture $\theta \in [\theta_r, \theta_s]$ with $\theta_s > \theta_r$, inverse air-entry scale $\alpha > 0$ ($\mathrm{m^{-1}}$), shape parameter $n > 1$, and $m = 1 - 1/n$ (van Genuchten, 1980; Hillel, 1998):

$$
S_e = \max\!\left(\frac{\theta - \theta_r}{\theta_s - \theta_r},\; \epsilon_S\right), \qquad
h = -\frac{1}{\alpha}\left(S_e^{-1/m} - 1\right)^{1/n}
$$

$$
\psi_{\mathrm{Pa}} = \rho_w g\,h, \qquad
\psi_{\mathrm{MPa}} = \operatorname{convert}(\psi_{\mathrm{Pa}},\; \mathrm{Pa},\; \mathrm{MPa})
$$

Endpoint regularization $\epsilon_S = 10^{-10}$ inside `van_genuchten_pressure_head` keeps $h$ finite when $\theta = \theta_r$ while yielding $h = 0$ and $\psi_{\mathrm{MPa}} = 0$ at full saturation ($\theta = \theta_s$). Under unsaturated tension, $h \le 0$ and $\psi_{\mathrm{MPa}} \le 0$.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\theta$ | Volumetric soil water content ($\theta_r \le \theta \le \theta_s$) | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `state` |
| $\psi_{\mathrm{MPa}}$ | Soil water matric potential ($\le 0$) | $\mathrm{MPa}$ | `soil_water_potential` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\alpha$ | `hydrosphere.soil_moisture.alpha_vg` | $\mathrm{m^{-1}}$ | `hydrosphere/soil_moisture.yaml` | van Genuchten (1980) |
| $n$ | `hydrosphere.soil_moisture.n_vg` | $\mathrm{1}$ | `hydrosphere/soil_moisture.yaml` | van Genuchten (1980) |
| $\theta_r$ | `hydrosphere.soil_moisture.residual_water_content` | $\mathrm{m^3\,m^{-3}}$ | `hydrosphere/soil_moisture.yaml` | van Genuchten (1980) |
| $\theta_s$ | `hydrosphere.soil_moisture.porosity` | $\mathrm{m^3\,m^{-3}}$ | `hydrosphere/soil_moisture.yaml` | van Genuchten (1980); Hillel (1998) |
| $\rho_w, g$ | `relic.physical.constants.water_density`, `relic.physical.constants.gravity_acceleration` | $\mathrm{kg\,m^{-3}}$, $\mathrm{m\,s^{-2}}$ | `physical/constants.yaml` | IAPWS / CODATA |

## 5. Numerical Methods and Discretization

Every `forward` call validates $\theta_s > \theta_r$, $\alpha > 0$, $n > 1$, and $\theta \in [\theta_r, \theta_s]$ without clipping out-of-domain states (raising `ValueError` outside $[\theta_r, \theta_s]$). Conversion from $\mathrm{Pa}$ to $\mathrm{MPa}$ delegates to `UnitConverter`. Output `soil_water_potential` is emitted as `UpdateItem(UpdateMode.REPLACE, per_timestep=False)` on differentiable `torch.Tensor` values.

## 6. Conservation, Boundary Conditions, and Ledgers

`SoilWaterRetention` is a stateless observation mapping $y_h = O_h x_h$: it owns no water mass or energy stock, introduces no boundary source or sink, and declares an empty `conservation_laws` list.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `SoilWaterRetention` | `relic/scientific/hydrosphere/soil_moisture/retention.py` | `PhysicalField` (`SPATIAL_DIMS=None`) | `PhysicalFieldFunctor` (`relic/core/functor/physical_field_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `SoilWaterRetention`: `[I]`, IPHS **Level 1** (`observation`, certificate `hydrosphere_soil_moisture_retention_soil_water_retention.yaml`). Pure retention observation operator calling shared `van_genuchten_pressure_head`.

Frozen certificate expressions:

```text
Se = (theta - theta_r) / (theta_s - theta_r); h = -(Se^(-1/m) - 1)^(1/n) / alpha, m = 1 - 1/n; psi = rho_w g h
y_h = O_h x_h through the declared observation operator
```

## 9. References

- Hillel, D. (1998). *Environmental Soil Physics*. Academic Press.
- van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. *Soil Science Society of America Journal*, 44(5), 892-898.
