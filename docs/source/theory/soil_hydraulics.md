# Unified Soil Hydraulics Utility and Pedosphere Soil-Water Transport

## 1. Process Definition and Scope

This document covers two tightly coupled components:

1. `relic/utils/soil_hydraulics.py`: the single stateless utility implementing van Genuchten-Mualem constitutive functions (`van_genuchten_m_parameter`, `van_genuchten_water_retention`, `effective_saturation`, `van_genuchten_pressure_head`, and `van_genuchten_mualem_conductivity`) shared across hydrosphere and pedosphere modules.
2. `SoilWaterTransportFunctor` (`relic/scientific/pedosphere/transport/SoilWaterTransportFunctor.py`): the one-dimensional pedosphere unsaturated Richards water-transport Functor (`SPATIAL_DIMS = 1`) that couples those shared constitutive laws with `RichardsNumericalSolver` modified-Picard integration under `pedosphere.*` and `transport.numerical.*` parameter namespaces.

When coupled with in-model surface hydrology, `SoilWaterTransportFunctor` declares lagged dependencies on `surface_runoff` and the selected evapotranspiration owner (`penman_monteith_et` or `priestley_taylor_et`). Brooks-Corey or Campbell closures inside `soil_hydraulics.py` and `SoilWaterTransportFunctor`, as well as two- or three-dimensional pedosphere Richards grids, remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapters 19 and 23.

## 2. Governing Equations

### 2.1 Shared Van Genuchten-Mualem Constitutive Functions (`relic/utils/soil_hydraulics.py`)

With shape relation $m = 1 - 1/n$ ($n > 1$) and regularization threshold $\epsilon = 10^{-10}$ (Mualem, 1976; van Genuchten, 1980):

$$
S_e(\theta) = \operatorname{clip}\!\left(\frac{\theta - \theta_r}{\max(\theta_s - \theta_r, \epsilon)},\; \epsilon,\; 1\right), \qquad
\theta(h) = \theta_r + \frac{\theta_s - \theta_r}{\bigl[1 + (\alpha\lvert h\rvert)^n\bigr]^m}
$$

$$
h(\theta) = -\frac{1}{\alpha}\bigl[\max(S_e^{-1/m} - 1,\; 0)\bigr]^{1/n}, \qquad
K(S_e) = K_s\,\tilde{S}_e^l\left[1 - \left(1 - \tilde{S}_e^{1/m}\right)^m\right]^2
$$

where $\tilde{S}_e = \operatorname{clip}(S_e, \epsilon, 1 - \epsilon)$ inside `van_genuchten_mualem_conductivity`.

### 2.2 Discrete Cochain Transport and Surface Boundary Conversion (`SoilWaterTransportFunctor`)

Over a 1D vertical column with uniform cell spacing $\Delta z$ (`pedosphere.grid.dz`), discrete divergence $D_h$, discrete gradient $d_h$, gravity head gradient $g_z$ (`gravity_head_gradient`), and symmetric positive-definite Darcy Hodge operator $\star_K = \star_K^{\mathsf{T}} \succeq 0$ (Richards, 1931; Celia et al., 1990):

$$
\dot{\theta}_h = -D_h q_h + G_h u_h, \qquad
q_h = -\star_K X_h, \qquad
X_h = d_h(h_h + z_h), \qquad
\sigma_h = X_h^{\mathsf{T}}\star_K X_h \ge 0
$$

Surface mass fluxes convert to top-boundary volumetric velocity $u_h$ ($\mathrm{m\,s^{-1}}$) via physical-constant liquid water density $\rho_w$, while the free-drainage bottom boundary uses bottom-cell conductivity:

$$
u_h = q_{\mathrm{surface}} = \frac{F_{\mathrm{infil}} - F_{\mathrm{ET}}}{\rho_w}, \qquad
q_{\mathrm{bottom}} = K(\theta_{\mathrm{bottom}})\,g_z
$$

Specific moisture capacity $C(h) = \bigl(\frac{dh}{d\theta}\bigr)^{-1}$ is obtained by automatic differentiation of `van_genuchten_pressure_head` regularized by `water_capacity_epsilon`.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\theta_i$ | 1D soil volumetric water content profile ($\ge 2$ layers; input and updated replacement) | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `state` |
| $h_i$ | Optional pressure-head profile input | $\mathrm{m}$ | `pressure_head` | `state` |
| $F_{\mathrm{infil}}$ | Scalar surface infiltration mass flux ($\ge 0$, required) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `infiltration_flux` | `flux` |
| $F_{\mathrm{ET}}$ | Scalar surface evapotranspiration mass flux ($\ge 0$, required) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `evapotranspiration_flux` | `flux` |
| $r_{\mathrm{clip}}, R_w$ | Saturation-bound clipping residual and open-column water balance residual | $\mathrm{m^3\,m^{-3}}$, $\mathrm{m}$ | `diagnostics.soil_water_transport.clipping_residual`, `water_balance_residual` | `diagnostic` |
| $q_h, N_{\mathrm{iter}}$ | Interface Darcy volumetric flux profile and Picard iteration count | $\mathrm{m\,s^{-1}}$, $\mathrm{1}$ | `diagnostics.soil_water_transport.darcy_flux`, `richards_iterations` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\Delta z$ | `pedosphere.grid.dz` | $\mathrm{m}$ | `pedosphere/grid.yaml` | 1D column spacing |
| $K_s$ | `pedosphere.hydraulic.saturated_conductivity` | $\mathrm{m\,s^{-1}}$ | `pedosphere/hydraulic.yaml` | Mualem (1976) |
| $\theta_{\mathrm{wp}}, \theta_s$ | `pedosphere.moisture.wilting_point`, `pedosphere.soil.porosity` | $\mathrm{m^3\,m^{-3}}$ | `pedosphere/moisture.yaml`, `soil.yaml` | van Genuchten (1980) |
| $\alpha, n$ | `pedosphere.van_genuchten.alpha_vg`, `pedosphere.van_genuchten.n_vg` | $\mathrm{m^{-1}}$, $\mathrm{1}$ | `pedosphere/van_genuchten.yaml` | van Genuchten (1980) |
| $g_z$ | `pedosphere.transport.gravity_head_gradient` | $\mathrm{1}$ | `pedosphere/transport.yaml` | Richards (1931) |
| $\Delta t_{\min}, \varepsilon_{\mathrm{rich}}, N_{\max}$ | `transport.numerical.min_timestep`, `richards_tolerance`, `max_richards_iterations` | $\mathrm{s}$, $\mathrm{1}$, `int` | `transport/numerical.yaml` | Celia et al. (1990) |
| $c_{\mathrm{CFL}}, f_{\mathrm{harm}}, \omega_{\mathrm{ur}}, \varepsilon_{\mathrm{lin}}$ | `pedosphere.numerical.cfl_number`, `harmonic_mean_factor`, `under_relaxation_factor`, `linear_solver_tolerance` | $\mathrm{1}$ | `pedosphere/numerical.yaml` | Solver controls |
| $\epsilon_L, \epsilon_D, \epsilon_C$ | `pedosphere.numerical.epsilon_length`, `epsilon_diffusivity`, `water_capacity_epsilon` | $\mathrm{m}$, $\mathrm{m^2\,s^{-1}}$, $\mathrm{m^{-1}}$ | `pedosphere/numerical.yaml` | Numerical guards |
| $\rho_w$ | `relic.physical.constants.water_density` | $\mathrm{kg\,m^{-3}}$ | `physical/constants.yaml` | IAPWS reference density |

## 5. Numerical Methods and Discretization

`SoilWaterTransportFunctor` enforces a 1D column (`len(soil_moisture) >= 2`, values inside `[wilting_point, porosity]`), positive scalar $\Delta t$ within the configured stability bound, and converged modified-Picard iteration via `RichardsNumericalSolver` (raising an error before returning any `UpdateItem` if convergence is not reached). Updated `soil_moisture` and all four diagnostics use `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`.

## 6. Conservation, Boundary Conditions, and Ledgers

For downward-positive boundary velocities, the open-column depth ledger ($\mathrm{m}$) is:

$$
R_w = \sum_i \bigl(\theta_i^{n+1} - \theta_i^n\bigr)\Delta z - q_{\mathrm{surface}}\Delta t + q_{\mathrm{bottom}}\Delta t
$$

`SoilWaterTransportFunctor.check_conservation_laws()` evaluates `diagnostics.soil_water_transport.water_balance_residual` directly, retaining any saturation-bound `clipping_residual` explicitly without silent state renormalization.

## 7. Code Mapping

| Module / Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `soil_hydraulics` functions | `relic/utils/soil_hydraulics.py` | Stateless constitutive utility | Called by hydrosphere and pedosphere Functors | Library helper |
| `SoilWaterTransportFunctor` | `relic/scientific/pedosphere/transport/SoilWaterTransportFunctor.py` | `TransportMechanism` (`SPATIAL_DIMS=1`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.POINT` / 1D column) |

## 8. Implementation Status

- `relic/utils/soil_hydraulics.py`: `[I]` shared van Genuchten-Mualem constitutive SSOT.
- `SoilWaterTransportFunctor`: `[I]`, IPHS **Level 3** (`transport`, certificate `pedosphere_transport_soil_water_transport_functor_soil_water_transport_functor.yaml`). Typed 1D cochain spaces, symmetric positive-definite Darcy `HodgeStarSpec`, matter boundary ports, and open-column mass-balance ledger.
- Non-1D pedosphere Richards solvers and Brooks-Corey/Campbell utilities inside `soil_hydraulics.py`: `[N]`.

Frozen certificate expressions:

```text
dtheta/dt = div(K(theta) * grad(h + z)) + S
theta_dot = -D_h q_h + G_h u_h; q_h = -star_K X_h; X_h = d(h + z); sigma = X_h.T star_K X_h >= 0.
```

## 9. References

- Celia, M. A., Bouloutas, E. T., and Zarba, R. L. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. *Water Resources Research*, 26(7), 1483-1496.
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. *Water Resources Research*, 12(3), 513-522.
- Richards, L. A. (1931). Capillary conduction of liquids through porous mediums. *Physics*, 1(5), 318-333.
- van Genuchten, M. Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. *Soil Science Society of America Journal*, 44(5), 892-898.
