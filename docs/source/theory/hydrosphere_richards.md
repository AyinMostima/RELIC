# Mixed-Form Richards Unsaturated Soil Moisture Dynamics

## 1. Process Definition and Scope

`RichardsSoilMoisture` (`relic/scientific/hydrosphere/soil_moisture/richards.py`) and its shared tridiagonal solver `RichardsNumericalSolver` (`relic/scientific/hydrosphere/solvers/richards_solver.py`) integrate one-dimensional vertical unsaturated liquid water flow across $N_z$ soil layers using the mass-conservative Celia et al. (1990) mixed-form discretization and van Genuchten-Mualem hydraulic relations.

`RichardsSoilMoisture` declares `CONFLICT_RESOLUTION = ConflictResolutionPolicy.EXCLUSIVE`, an optional dependency on `surface_runoff`, and lagged dependencies on `penman_monteith_et`, `priestley_taylor_et`, and `capillary_rise_transport`. Distributed volumetric root water-uptake sink profiles $S(z,t)$ (such as Feddes root-sink weighting; evapotranspiration enters at the top boundary in `"atmospheric"` mode), Brooks-Corey conductivity inside `RichardsSoilMoisture`, and two- or three-dimensional lateral variably saturated Richards flow remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapters 7 and 19 (Sections 19.7 and 19.10.2).

## 2. Governing Equations

### 2.1 Vertical Coordinate Sign Convention and Mixed PDE Form

With depth $z$ positive downward ($\mathrm{m}$), pressure head $h \le 0$ ($\mathrm{m}$), and unsaturated hydraulic conductivity $K(\theta)$ ($\mathrm{m\,s^{-1}}$), downward-positive Darcy volumetric flux $q$ ($\mathrm{m\,s^{-1}}$) and the source-free mixed-form Richards equation (Richards, 1931; Celia et al., 1990) are:

$$
q = -K(\theta)\left(\frac{\partial h}{\partial z} - 1\right) = -D(\theta)\frac{\partial \theta}{\partial z} + K(\theta)
$$

$$
\frac{\partial \theta}{\partial t} = -\frac{\partial q}{\partial z} = \frac{\partial}{\partial z}\left[K(\theta)\left(\frac{\partial h}{\partial z} - 1\right)\right] = \frac{\partial}{\partial z}\left[D(\theta)\frac{\partial \theta}{\partial z}\right] - \frac{\partial K(\theta)}{\partial z}
$$

where $D(\theta) = K(\theta)\frac{dh}{d\theta}$ ($\mathrm{m^2\,s^{-1}}$). In the upward-oriented DEC cochain convention of the Level-3 certificate ($q_z = -q$), $\dot{\theta} = -D_z q_z$ with $q_z = \star_K(dh - g_z)$.

### 2.2 Van Genuchten-Mualem Constitutive Closure

With effective saturation $S_e = (\theta - \theta_r)/(\theta_s - \theta_r)$ regularized by `saturation_epsilon`, $m = 1 - 1/n$ ($n > 1$), inverse air-entry scale $\alpha$ ($\mathrm{m^{-1}}$), and parameterized tortuosity exponent $l$ (`hydrosphere.soil_moisture.tortuosity_parameter`) (Mualem, 1976; van Genuchten, 1980):

$$
\theta(h) = \theta_r + (\theta_s - \theta_r)\bigl[1 + (\alpha\lvert h\rvert)^n\bigr]^{-m}, \qquad
h(\theta) = -\frac{1}{\alpha}\left(S_e^{-1/m} - 1\right)^{1/n}
$$

$$
K(\theta) = K_s\,S_e^l\left[1 - \left(1 - S_e^{1/m}\right)^m\right]^2, \qquad
\psi_m = \rho_w g\,h(\theta)
$$

where $\psi_m$ (`soil_matric_potential`, $\mathrm{Pa}$) is derived directly from updated $\theta$ at every layer.

### 2.3 Derived Wetness and Infiltration Capacity Diagnostics

With $\theta_{\mathrm{fc}} = f_{\mathrm{fc}}\theta_s$, $\theta_{\mathrm{wp}} = f_{\mathrm{wp}}\theta_r$, profile-mean moisture $\bar{\theta}$, and top-layer conductivity $K_0 = K(\theta_0)$, relative profile wetness $f_{\mathrm{rel}}$ and diagnostic surface infiltration capacity $I_{\mathrm{cap}}$ ($\mathrm{m\,s^{-1}}$) are:

$$
f_{\mathrm{rel}} = \operatorname{clip}\!\left(\frac{\bar{\theta} - \theta_{\mathrm{wp}}}{\theta_{\mathrm{fc}} - \theta_{\mathrm{wp}}},\; 0,\; 1\right), \qquad
I_{\mathrm{cap}} = K_0\left(1 + \frac{\theta_s - \theta_0}{\theta_s - \theta_r}\right)
$$

$f_{\mathrm{rel}}$ is reported under `diagnostics.soil_moisture_richards.relative_moisture`; biological `water_stress_factor` remains owned by `WaterStress`.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\theta_i$ | Layered volumetric soil moisture profile (input and updated replacement) | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `state` |
| $F_{\mathrm{infil}}$ | Surface infiltration mass flux (optional in atmospheric top BC) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `infiltration_flux` | `flux` |
| $F_{\mathrm{ET}}$ | Surface evapotranspiration mass flux (optional top-boundary demand) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `evapotranspiration_flux` | `flux` |
| $F_{\mathrm{cap}}$ | Upward capillary rise mass flux at bottom boundary (optional) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `capillary_rise_flux` | `flux` |
| $\psi_m$ | Layered soil matric pressure profile ($\le 0$) | $\mathrm{Pa}$ | `soil_matric_potential` | `diagnostic` |
| $D_m$ | Bottom-boundary gravity drainage mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `drainage_flux` | `flux` |
| $\dot{\theta}_i, q_{i+1/2}, f_{\mathrm{rel}}$ | Moisture tendency, interface flux profile, and relative wetness diagnostics | $\mathrm{s^{-1}}$, $\mathrm{m\,s^{-1}}$, $\mathrm{1}$ | `diagnostics.soil_moisture_richards.soil_moisture_change_rate`, `diagnostics.soil_moisture_richards.water_flux_profile`, `diagnostics.soil_moisture_richards.relative_moisture` | `diagnostic` |
| $\mathcal{D}_{\mathrm{rich}}$ | Solver convergence, iteration count, infiltration capacity, and mass-balance dictionary | mixed SI | `diagnostics.soil_moisture_richards` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $N_z, Z$ | `hydrosphere.soil_moisture.layer_count`, `hydrosphere.soil_moisture.soil_depth` | `int`, $\mathrm{m}$ | `hydrosphere/soil_moisture.yaml` | Vertical grid geometry |
| $\theta_r, \theta_s, \alpha, n$ | `hydrosphere.soil_moisture.residual_water_content`, `porosity`, `alpha_vg`, `n_vg` | $\mathrm{m^3\,m^{-3}}$, $\mathrm{m^{-1}}$, $\mathrm{1}$ | `hydrosphere/soil_moisture.yaml` | van Genuchten (1980) |
| $K_s, l$ | `hydrosphere.soil_moisture.saturated_hydraulic_conductivity`, `tortuosity_parameter` | $\mathrm{m\,s^{-1}}$, $\mathrm{1}$ | `hydrosphere/soil_moisture.yaml` | Mualem (1976) |
| $f_{\mathrm{fc}}, f_{\mathrm{wp}}$ | `hydrosphere.physical.field_capacity_fraction`, `wilting_point_fraction` | $\mathrm{1}$ | `hydrosphere/physical.yaml` | Hillel (1998) |
| $b_{\mathrm{NR}}, b_{\mathrm{mass}}$ | `hydrosphere.soil_moisture.richards_enable_newton_raphson`, `richards_enable_mass_conservation_check` | `bool` | `hydrosphere/soil_moisture.yaml` | Celia et al. (1990) |
| $N_{\mathrm{Pic}}, N_{\mathrm{Newt}}, \varepsilon_{\mathrm{conv}}, \varepsilon_{\mathrm{abs}}, \varepsilon_{\mathrm{rel}}$ | `hydrosphere.numerical.richards_max_picard_iterations`, `richards_max_newton_iterations`, `richards_convergence_tolerance`, `richards_newton_absolute_tolerance`, `richards_newton_relative_tolerance` | `int`, $\mathrm{1}$ | `hydrosphere/numerical.yaml` | Celia et al. (1990) |
| $f_{\mathrm{harm}}, c_{\mathrm{CFL}}, \varepsilon_{\mathrm{mass}}, \Delta t_{\min}$ | `hydrosphere.numerical.harmonic_mean_factor`, `richards_cfl_coefficient`, `mass_conservation_tolerance`, `minimum_timestep` | $\mathrm{1}$, $\mathrm{s}$ | `hydrosphere/numerical.yaml` | RELIC solver controls |
| $\epsilon_S, p_{\min}, s_{\min}, t_{1,\min}, \varepsilon_{\mathrm{tri}}, h_{\mathrm{char}}$ | `hydrosphere.numerical.saturation_epsilon`, `minimum_pressure_term`, `minimum_se_term`, `minimum_term1`, `tridiagonal_tolerance`, `characteristic_pressure_head` | $\mathrm{1}$, $\mathrm{m}$ | `hydrosphere/numerical.yaml` | Constitutive guards |
| $\alpha_0, N_{\mathrm{bt}}, c_{\mathrm{Arm}}, \beta_{\mathrm{ls}}, \alpha_{\min}$ | `hydrosphere.numerical.line_search_initial_alpha`, `line_search_max_backtracks`, `line_search_armijo_constant`, `line_search_alpha_reduction_factor`, `line_search_min_alpha` | $\mathrm{1}$, `int` | `hydrosphere/numerical.yaml` | Armijo line search |
| $\rho_w, g$ | `relic.physical.constants.water_density`, `relic.physical.constants.gravity_acceleration` | $\mathrm{kg\,m^{-3}}$, $\mathrm{m\,s^{-2}}$ | `physical/constants.yaml` | IAPWS / CODATA |

## 5. Numerical Methods and Discretization

Over uniform cells $\Delta z = Z / N_z$, finite-volume conservation $\Delta z(\theta_i^{n+1} - \theta_i^n)/\Delta t = q_{i-1/2}^{n+1} - q_{i+1/2}^{n+1}$ uses underflow-safe harmonic interface conductivity:

$$
K_{i+1/2} =
\begin{cases}
f_{\mathrm{harm}}\,K_i\dfrac{K_{i+1}}{K_i + K_{i+1}}, & K_i + K_{i+1} > 0 \\
0, & K_i + K_{i+1} = 0
\end{cases}
$$

Implicit backward-Euler residuals are solved at each substep via modified Picard iteration (Celia et al., 1990) or optional tridiagonal Newton-Raphson iteration (`richards_enable_newton_raphson`) with Armijo backtracking line search (`line_search_*`) and adaptive CFL/convergence substepping down to `minimum_timestep`. Outputs use `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`.

## 6. Conservation, Boundary Conditions, and Ledgers

Top boundary `"atmospheric"` applies net surface volumetric flux $q_{1/2} = (F_{\mathrm{infil}} - F_{\mathrm{ET}})/\rho_w$, while bottom free-drainage / capillary coupling applies $q_{N_z+1/2} = K_{N_z} - F_{\mathrm{cap}}/\rho_w$ (emitting `drainage_flux` $D_m = \rho_w \max(q_{N_z+1/2}, 0)$). When `richards_enable_mass_conservation_check` is active, column mass closure verifies:

$$
\left\lvert \sum_{i=1}^{N_z} \rho_w \Delta z\bigl(\theta_i^{n+1} - \theta_i^n\bigr) - \rho_w(q_{1/2} - q_{N_z+1/2})\Delta t \right\rvert \le \varepsilon_{\mathrm{mass}}
$$

## 7. Code Mapping

| Functor / Solver Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `RichardsSoilMoisture` | `relic/scientific/hydrosphere/soil_moisture/richards.py` | `TransportMechanism` (`TRANSPORT_TYPE="diffusion"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`, `spatial_dimensions=1`) |
| `RichardsNumericalSolver` | `relic/scientific/hydrosphere/solvers/richards_solver.py` | Shared 1D mixed-form tridiagonal solver | Called by `RichardsSoilMoisture` | Internal solver helper |

## 8. Implementation Status

- `RichardsSoilMoisture`: `[I]`, IPHS **Level 3** (`transport`, certificate `hydrosphere_soil_moisture_richards_richards_soil_moisture.yaml`). Source-free 1D mixed-form finite-volume solver with symmetric positive-definite Darcy conductance $\sigma = X_h^{\mathsf{T}}\star_K X_h \ge 0$ (the `- S(theta)` and `+ s_z` terms in the frozen certificate strings denote zero volumetric internal sink in the current 1D boundary-driven implementation).
- Distributed root water-uptake sink term $S(z,t)$, Brooks-Corey conductivity inside `RichardsSoilMoisture`, and multi-dimensional Richards flow: `[N]`.

Frozen certificate expressions:

```text
dtheta/dt = d/dz[K(theta)(dh/dz - 1)] - S(theta), discretized by the canonical mixed-form solver.
theta_dot = -D_z q_z + s_z; q_z = star_K (d h - g_z); sigma = X_h.T star_K X_h >= 0.
```

## 9. References

- Celia, M. A., Bouloutas, E. T., and Zarba, R. L. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. *Water Resources Research*, 26(7), 1483-1496.
- Hillel, D. (1998). *Environmental Soil Physics*. Academic Press.
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. *Water Resources Research*, 12(3), 513-522.
- Richards, L. A. (1931). Capillary conduction of liquids through porous mediums. *Physics*, 1(5), 318-333.
- van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. *Soil Science Society of America Journal*, 44(5), 892-898.
