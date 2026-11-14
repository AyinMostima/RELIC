# Agricultural Irrigation Boundary Transfer and Quota Allocation

## 1. Process Definition and Scope

`IrrigationManagement` (`relic/scientific/anthroposphere/management/irrigation.py`) computes demand-triggered agricultural water withdrawal (`abstraction_flux`), conveyance-adjusted root-zone water delivery (`irrigation_flux`), and domain quota depletion (`irrigation_remaining_quota`) across irrigated cropland grid cells (`land_use_type == 1`).

The Functor implements `TransportMechanism` (`TRANSPORT_TYPE="managed_application"`) under canonical instance name `"irrigation_management"` without owning prognostic soil moisture; root-zone infiltration, drainage, and evapotranspiration remain owned by hydrosphere and pedosphere Functors. `IrrigationManagement` evaluates root-zone moisture deficit directly from `soil_moisture`, `field_capacity`, `wilting_point`, and `precipitation_flux`, and does not consume `irrigation_demand_flux` emitted by `ManagementCoordinator`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 24 (Section 24.5.1 and Exercise 24.2).

## 2. Governing Equations

### 2.1 Moisture-Deficit Trigger and Net Root-Zone Demand

For cell $i$ with volumetric soil moisture $\theta_i \in [0, 1]$, wilting point $\theta_{\mathrm{wp},i} \in [0, 1]$, field capacity $\theta_{\mathrm{fc},i} \in (\theta_{\mathrm{wp},i}, 1]$, integer land-use code $c_{\mathrm{lu},i} \ge 0$, and plant-available water trigger fraction $f_{\mathrm{trig}} \in [0, 1]$, the moisture threshold and discrete activation mask are

$$
\theta_{\mathrm{trig},i} = \theta_{\mathrm{wp},i} + f_{\mathrm{trig}}\bigl(\theta_{\mathrm{fc},i} - \theta_{\mathrm{wp},i}\bigr), \qquad
u_i = \mathbf{1}_{\theta_i < \theta_{\mathrm{trig},i}}\,\mathbf{1}_{c_{\mathrm{lu},i} = 1}
$$

With effective root-zone depth $Z_r > 0$ ($\mathrm{m}$), liquid water density $\rho_w > 0$ ($\mathrm{kg\,m^{-3}}$), and precipitation mass flux $F_{P,i} \ge 0$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) treated at $100\%$ effective rainfall ($P_e = F_P$, simplifying FAO runoff/interception deductions), the step net water-mass requirement ($\mathrm{kg\,m^{-2}}$) is

$$
M_{\mathrm{def},i} = \max\bigl(\theta_{\mathrm{fc},i} - \theta_i,\; 0\bigr)\,Z_r\,\rho_w, \qquad
M_{\mathrm{net},i} = \max\bigl(M_{\mathrm{def},i} - F_{P,i}\,\Delta t,\; 0\bigr)
$$

### 2.2 Conveyance Efficiency, Rate Cap, and Equal-Percentage Quota Share

Integer system code $c_{\mathrm{irr},i} \in \{1, 2, 3, 4\}$ selects learnable application efficiency $\eta_i \in (0, 1]$ (`1`: surface, `2`: sprinkler, `3`: drip, `4`: subsurface). Maximum application rate $r_{\max}$ ($\mathrm{mm\,h^{-1}}$) converts via `UnitConverter` to mass-flux cap $F_{\max} = r_{\max}\rho_w / (1000 \times 3600)$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$), yielding pre-quota gross withdrawal mass $A_i$:

$$
A_i = u_i\,\min\!\left(\frac{M_{\mathrm{net},i}}{\eta_i},\; F_{\max}\,\Delta t\right)
$$

With depth-conversion factor $h_w = 1000 / \rho_w$ ($\mathrm{mm\,m^2\,kg^{-1}}$), requested domain quota depth $D = \sum_i A_i\,h_w$ ($\mathrm{mm}$, summed unweighted across active cells; *Methods of Ecological Analysis*, Chapter 24, Exercise 24.2), and scalar remaining quota $Q \ge 0$ ($\mathrm{mm}$), single-priority WEAP21 equal-percentage rationing (Yates et al., 2005) scales all active cells by $s \in [0, 1]$:

$$
s = \begin{cases} 1, & D \le Q \\ \dfrac{Q}{D}, & D > Q \end{cases}, \qquad
M_{\mathrm{gross},i} = s\,A_i, \qquad
M_{\mathrm{deliv},i} = \eta_i\,M_{\mathrm{gross},i}
$$

Step boundary fluxes and updated scalar quota state are

$$
F_{\mathrm{abs},i} = \frac{M_{\mathrm{gross},i}}{\Delta t}, \qquad
F_{\mathrm{irr},i} = \frac{M_{\mathrm{deliv},i}}{\Delta t}, \qquad
Q' = \begin{cases} Q - \sum_i M_{\mathrm{gross},i}\,h_w, & D \le Q \\ 0, & D > Q \end{cases}
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\theta, \theta_{\mathrm{fc}}, \theta_{\mathrm{wp}}$ | Volumetric soil moisture, field capacity, and wilting point | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture`, `field_capacity`, `wilting_point` | `state` / `forcing` |
| $F_P$ | Surface precipitation mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `precipitation_flux` | `forcing` |
| $c_{\mathrm{lu}}, c_{\mathrm{irr}}$ | Integer land-use class (`1` = irrigated cropland) and irrigation-system code (`1..4`) | $\mathrm{1}$ | `land_use_type`, `irrigation_type` | `forcing` |
| $Q, Q'$ | Scalar domain remaining source-water quota | $\mathrm{mm}$ | `irrigation_remaining_quota` | `state` |
| $F_{\mathrm{irr}}, F_{\mathrm{abs}}$ | Delivered root-zone water flux and gross source withdrawal flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `irrigation_flux`, `abstraction_flux` | `flux` |
| $d_{\mathrm{app}}, \eta_{\mathrm{eff}}, \psi_{\theta}$ | Applied water depth, active system efficiency, and normalized moisture deficit $(\theta_{\mathrm{fc}}-\theta)/(\theta_{\mathrm{fc}}-\theta_{\mathrm{wp}})$ | $\mathrm{mm}$, $\mathrm{1}$, $\mathrm{1}$ | `diagnostics.irrigation.applied_water`, `diagnostics.irrigation.effective_efficiency`, `diagnostics.irrigation.moisture_response` | `diagnostic` |
| $r_w$ | Pointwise conveyance balance residual $\lvert F_{\mathrm{irr}} - \eta F_{\mathrm{abs}} \rvert$ | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.irrigation.water_balance_residual` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\eta_1, \eta_2, \eta_3, \eta_4$ | `anthroposphere.management.irrigation.surface_efficiency`, `anthroposphere.management.irrigation.sprinkler_efficiency`, `anthroposphere.management.irrigation.drip_efficiency`, `anthroposphere.management.irrigation.subsurface_efficiency` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Allen et al. (1998) FAO-56 |
| $f_{\mathrm{trig}}$ | `anthroposphere.management.irrigation.moisture_threshold` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Doorenbos and Pruitt (1977) FAO-24 |
| $Z_r$ | `anthroposphere.management.irrigation.root_zone_depth` | $\mathrm{m}$ | `anthroposphere/management.yaml` | Allen et al. (1998) |
| $r_{\max}$ | `anthroposphere.management.irrigation.max_application_rate` | $\mathrm{mm\,h^{-1}}$ | `anthroposphere/management.yaml` | System hydraulic capacity |
| $\varepsilon_w$ | `anthroposphere.management.irrigation.conservation_tolerance` | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `anthroposphere/management.yaml` | Open-port residual tolerance |
| $\rho_w$ | `water_density` | $\mathrm{kg\,m^{-3}}$ | `physical/constants.yaml` | IAPWS standard liquid water density |

## 5. Numerical Methods and Discretization

`IrrigationManagement.forward()` validates finite domains across all seven inputs without silent defaults or module-internal buffers. The discrete trigger $\theta_i < \theta_{\mathrm{trig},i}$ is piecewise constant with zero gradient with respect to $f_{\mathrm{trig}}$, whereas active delivery remains differentiable with respect to learnable system efficiencies $\eta_1, \dots, \eta_4$. One-dimensional `[N]` or row `[1, N]` spatial outputs are normalized to column `[N, 1]` layout by `_reshape_output()`, and all seven outputs—including scalar `irrigation_remaining_quota`—are emitted via `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`.

## 6. Conservation, Boundary Conditions, and Ledgers

`IrrigationManagement` forms an open two-port water-transfer operator: `abstraction_flux` withdraws mass from the external source-water ledger, `irrigation_flux` delivers $\eta F_{\mathrm{abs}}$ to the root-zone boundary, and $(1 - \eta)F_{\mathrm{abs}}$ represents non-recovered conveyance loss. `forward()` enforces $\max_i \lvert F_{\mathrm{irr},i} - \eta_i F_{\mathrm{abs},i} \rvert \le \varepsilon_w$, and `check_conservation_laws()` reports the maximum absolute residual under `"mass"`. Note that the frozen certificate `iphs_expression` states single-cell quota capping $\min(M_{\mathrm{net}}/\eta, F_{\max}\Delta t, M_{\mathrm{quota}})$, whereas live multi-cell `forward()` applies equal-percentage domain rationing $s = \min(1, Q / D)$ over unweighted cell-depth sum $D = \sum_i A_i h_w$.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `IrrigationManagement` | `relic/scientific/anthroposphere/management/irrigation.py` | `TransportMechanism` (`TRANSPORT_TYPE="managed_application"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `IrrigationManagement`: `[I]`, IPHS **Level 2** (`port`, certificate `anthroposphere_management_irrigation_irrigation_management.yaml`). Non-strict phenomenological managed water-boundary wrapper with explicit source withdrawal, conveyance loss, and `StateManager`-owned quota state; cell-area-weighted volumetric quota summation and sub-100% FAO effective-rainfall partitioning remain `[N]`.

Frozen certificate expressions:

```text
I = (ET_c - P_eff - DeltaS) / E_irr
M_net = max((theta_fc-theta) Z_r rho_w - precipitation_flux dt, 0); M_gross = min(M_net/eta, F_max dt, M_quota); M_delivered = eta M_gross
```

## 9. References

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56. FAO, Rome.
- Doorenbos, J., and Pruitt, W. O. (1977). *Guidelines for Predicting Crop Water Requirements*. FAO Irrigation and Drainage Paper 24. FAO, Rome.
- Hanks, R. J. (1974). Model for predicting plant yield as influenced by water use. *Agronomy Journal*, 66(5), 660-665.
- Yates, D., Sieber, J., Purkey, D., and Huber-Lee, A. (2005). WEAP21—A demand-, priority-, and preference-driven water planning model: Part 1: Model characteristics. *Water International*, 30(4), 487-500.
