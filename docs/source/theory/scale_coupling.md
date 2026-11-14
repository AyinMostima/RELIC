# Scale Coupling: Spatial Remapping, Hierarchical Aggregation, and Cross-Domain Ports

## 1. Process Definition and Scope

The RELIC scale-coupling subsystem implements the **`ScaleCoupling`** abstraction (`TemporalScalingBase`, `SpatialScalingBase`, and `HierarchicalScalingBase` in `relic/core/abstractions.py`), bridged into the unified `FunctorBase` lifecycle through `ScaleCouplingFunctor` (`relic/core/functor/scale_coupling_bridge.py`). Every concrete operator declares explicit `source_scale`, `target_scale`, and `coupling_type` arguments without implicit defaults. Theoretical derivations for measure-preserving spatial projection, drag-coefficient roughness blending, and cross-sphere interface ports appear in *Methods of Ecological Analysis*, Chapters 2, 23, and 24.

The subsystem comprises four operational families:
1. **Whole-support and graph-edge spatial reductions**: `SpatialAggregation` (`relic/coupling/scales/adapters.py`) and `SupportAggregationAdapter` (`relic/orchestration/adapters.py`), backed by `aggregate_spatial_alias` (`relic/coupling/scales/alias_utils.py`).
2. **Grid-to-grid spatial remapping**: `ConservativeRemap` (`relic/coupling/scales/adapters.py`), `ConservativeRegridder`, `BilinearRegridder`, `NearestNeighborRegridder`, and `SpatialRegridderFactory` (`relic/coupling/scales/spatial/`), plus non-differentiable external-forcing ingestion regridding (`relic/coupling/external/common/regrid.py`).
3. **Nested-unit hierarchical coupling**: `HierarchicalAggregator` and `HierarchyIndex` (`relic/coupling/scales/hierarchical/`), governed at runtime by the `ScaleManager` facade (`relic/coupling/scales/scale_manager.py`) and simulator auto-regridding (`relic/core/simulator/scale_ops.py`, `relic/core/simulator/regridding_ops.py`).
4. **Internal cross-domain interface ports**: `SoilVegetationCoupling` (`relic/coupling/internal/soil_vegetation.py`), `BiosphereAtmosphereCoupling` (`relic/coupling/internal/biosphere_atmosphere.py`), and `WaterCarbonCoupling` (`relic/coupling/internal/water_carbon.py`).

Temporal interval-overlap kernels, CF calendar arithmetic, and standalone series quality control are specified in [time_scaling.md](time_scaling.md); static multi-scale delta configuration is specified in [multiscale_config.md](multiscale_config.md).

## 2. Governing Equations

### 2.1 Typed Spatial Quantity Basis and Whole-Support Reduction

Every spatial operator requires an explicit `SpatialQuantityBasis` (`relic/coupling/scales/types.py`): `intensive` (state or per-unit property), `cell_total` (per-cell amount $X_i$), `area_density` (areal density $x_i$ with cell amount $X_i = x_i A_i$), or `categorical` (integer class code). Bare `"extensive"` is ambiguous and raises `ValueError`. Exact conversion between the two amount bases on cells of area $A_i > 0$ (`convert_spatial_basis`) is

$$
x_i = \frac{X_i}{A_i}, \qquad X_i = x_i A_i.
$$

In `SpatialAggregation` and `SupportAggregationAdapter`, whole-support reduction (`aggregate_spatial_alias`) maps a tensor over weights $W_i \ge 0$ ($\sum_i W_i > 0$; $W_i = A_i$ required for `area_density`, $W_i = 1$ default for `intensive` and `categorical`) according to its declared basis:

$$
\phi = \frac{\sum_i W_i x_i}{\sum_i W_i} \quad (\text{intensive, area\_density}), \qquad
Q = \sum_i X_i \quad (\text{cell\_total}),
$$

while `categorical` fields select the smallest class code maximizing $\sum_i W_i\,\mathbb{I}[x_i = c]$. Leaf-area and ground-area flux densities convert across leaf area index $\mathrm{LAI} = A_{\mathrm{leaf}} / A_{\mathrm{ground}} > 0$ via $F_{\mathrm{ground}} = F_{\mathrm{leaf}}\,\mathrm{LAI}$ and $F_{\mathrm{leaf}} = F_{\mathrm{ground}} / \mathrm{LAI}$.

### 2.2 First-Order Conservative Remapping and Spatial Interpolation

On explicit overlap areas $A_{iJ} \ge 0$ ($\mathrm{m^2}$) between source cell $i$ (area $A_i > 0$) and target cell $J$ (area $A_J > 0$), let $W_{iJ} = A_{iJ} / A_i$, covered source fraction $f_i = \sum_J W_{iJ} \le 1$, and covered target area $C_J = \sum_i A_{iJ}$. `ConservativeRemap` and `ConservativeRegridder` evaluate one linear differentiable kernel without row renormalization (Jones, 1999):

$$
Y_J = \sum_i X_i W_{iJ} \quad (\text{cell\_total}), \qquad
y_J = \frac{\sum_i x_i A_i W_{iJ}}{A_J} \quad (\text{area\_density, destarea}), \qquad
y_J = \frac{\sum_i x_i A_i W_{iJ}}{C_J} \quad (\text{intensive, fracarea}).
$$

Spherical `latlon` grids compute cell and overlap areas using `earth_radius` $R_\oplus$ (`relic/parameters/database/relic/physical/constants.yaml`) as $A = R_\oplus^2(\lambda_2 - \lambda_1)(\sin\phi_2 - \sin\phi_1)$ for radian bounds $(\lambda_1, \lambda_2) \times (\phi_1, \phi_2)$; planar `cartesian` and `projected` grids use $(x_2 - x_1)(y_2 - y_1)$ or Shapely polygon intersections (`latlon` polygon grids raise `ValueError`). Downscaling exchanges source and target roles ($X_i = \sum_J Y_J A_{iJ}/A_J$, $x_i = \sum_J y_J A_{iJ}/A_i$, or $x_i = \sum_J y_J A_{iJ}/\sum_J A_{iJ}$). Uncovered source amounts

$$
U = \sum_i X_i (1 - f_i) \qquad \text{or} \qquad U = \sum_i x_i A_i (1 - f_i)
$$

are reported explicitly (`uncovered_source_amount`) and never redistributed onto covered targets.

For non-conservative interpolation (`preserve="none"`), `BilinearRegridder` applies separable bilinear weights on strictly monotonic regular `[ny, nx]` grids and scale-invariant Shepard inverse-distance weights $w_i = d_i^{-p} / \sum_k d_k^{-p}$ ($p = \text{weight\_power}$) on irregular grids, assigning $1/m$ across $m$ coincident nodes when $d_i \le \epsilon\max(\lVert p_t\rVert, \lVert p_i\rVert)$ (Shepard, 1968). `NearestNeighborRegridder` selects the closest node within search radius $r$ (`categorical` fields admit nearest-neighbor regridding only).

### 2.3 Hierarchical Aggregation and Disaggregation

For child units $i$ (area $A_i$) and parent units $j$ (declared area $A_j > 0$) linked by `HierarchyIndex.membership_mapping` $M_{ij} \in [0, 1]$ and normalized member-area fractions $f_{ij} = A_i M_{ij} / \sum_k A_k M_{kj}$, `HierarchicalAggregator` evaluates:

$$
y_j = \frac{\sum_i x_i A_i M_{ij}}{A_j} \quad (\text{area\_density}), \qquad
Y_j = \sum_i X_i M_{ij} \quad (\text{cell\_total}), \qquad
y_j = \sum_i f_{ij} x_i \quad (\text{intensive mean}).
$$

Additional intensive kernels include `harmonic_mean` ($\left(\sum_i f_{ij} x_i^{-1}\right)^{-1}$), `geometric_mean` ($\exp(\sum_i f_{ij}\ln x_i)$), `circular_mean` ($\operatorname{atan2}(\sum_i f_{ij}\sin x_i, \sum_i f_{ij}\cos x_i)$), and neutral-drag `effective_roughness_length` at blending height $z_r$ for $0 < z_{0,i} < z_r$ (Byun et al., 1999, Eq. 12-37'):

$$
z_{0,j} = z_r \exp\!\left[-\left(\sum_i \frac{f_{ij}}{\ln^2(z_r / z_{0,i})}\right)^{-1/2}\right].
$$

Conservative disaggregation over a strict partition splits amounts proportionally and broadcasts intensive values without reconstructing sub-parent heterogeneity:

$$
x_i = y_j \frac{A_j}{\sum_{k \in j} A_k} \quad (\text{area\_density}), \qquad
X_i = Y_j \frac{A_i}{\sum_{k \in j} A_k} \quad (\text{cell\_total}), \qquad
x_i = \sum_j M_{ij} y_j \quad (\text{intensive}),
$$

while categorical disaggregation raises `ValueError`.

### 2.4 Soil-Vegetation Feddes Root-Uptake Port

`SoilVegetationCoupling` (`source_scale="soil_layer"`, `target_scale="grid_cell"`, `coupling_type="spatial"`, `VariableKind.FLUX`) maps strictly increasing layer-bottom depths $z_i > 0$ ($\Delta z_i = z_i - z_{i-1}$, $z_0 = 0$, layer midpoints $\bar{z}_i = \tfrac{1}{2}(z_{i-1} + z_i)$, normalized depth $\hat{z}_i = \operatorname{clamp}(\bar{z}_i / Z_{\max}, 0, 1)$) to a normalized root profile $\beta_i$ ($\sum_i \beta_i \Delta z_i = 1$, units $\mathrm{m^{-1}}$; Jackson et al., 1996):
- Observed `root_density_profile` $\rho_{r,i} \ge 0$: $\beta_i = \rho_{r,i} / \sum_k \rho_{r,k}\Delta z_k$.
- Parametric pre-normalization kernel $\tilde{\beta}_i$ (`distribution_type`), normalized by $\beta_i = \tilde{\beta}_i / \sum_k \tilde{\beta}_k \Delta z_k$:

$$
\tilde{\beta}_i = \begin{cases}
a \exp(-a \hat{z}_i Z_{\max}), & \text{exponential}, \\
\max\!\left(c_{\mathrm{lin}}(1 - \hat{z}_i)/Z_{\max},\; 0\right), & \text{linear}, \\
\hat{z}_i^{p - 1}(1 - \hat{z}_i)^{q - 1}, & \text{beta}.
\end{cases}
$$

For layer matric potential $\psi_i$ ($\mathrm{Pa}$) and thresholds $\psi_{\mathrm{wilt}} < \psi_{\mathrm{opt,dry}} < \psi_{\mathrm{opt,wet}} < \psi_{\mathrm{anaer}}$, the four-threshold Feddes reduction $\alpha(\psi_i) \in [0, 1]$ (Feddes, Kowalik, and Zaradny, 1978; Skaggs et al., 2006) and layer sink $S_i$ ($\mathrm{kg\,m^{-3}\,s^{-1}}$) under potential transpiration $T_p$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) are:

$$
\alpha(\psi) = \begin{cases}
0, & \psi \le \psi_{\mathrm{wilt}} \text{ or } \psi \ge \psi_{\mathrm{anaer}}, \\
\dfrac{\psi - \psi_{\mathrm{wilt}}}{\psi_{\mathrm{opt,dry}} - \psi_{\mathrm{wilt}}}, & \psi_{\mathrm{wilt}} < \psi < \psi_{\mathrm{opt,dry}}, \\
1, & \psi_{\mathrm{opt,dry}} \le \psi \le \psi_{\mathrm{opt,wet}}, \\
\dfrac{\psi_{\mathrm{anaer}} - \psi}{\psi_{\mathrm{anaer}} - \psi_{\mathrm{opt,wet}}}, & \psi_{\mathrm{opt,wet}} < \psi < \psi_{\mathrm{anaer}},
\end{cases}
\qquad
S_i = \alpha(\psi_i)\,\beta_i\,T_p, \qquad T_{\mathrm{actual}} = \sum_i S_i \Delta z_i \le T_p.
$$

### 2.5 Biosphere-Atmosphere Leaf Boundary-Layer Conductance Port

`BiosphereAtmosphereCoupling` (`source_scale="canopy_air"`, `target_scale="leaf_surface"`, `coupling_type="spatial"`, `SpatialScale.PATCH`, `VariableKind.DIAGNOSTIC`) implements the CTSM single-layer bulk-canopy leaf boundary-layer closure from canonical `friction_velocity` $u_* > 0$, air temperature $T > 0$, and atmospheric pressure $P > 0$ (Campbell and Norman, 1998; Bonan et al., 2014; CTSM Technical Note Eqs. 2.5.128 and 2.5.135):

$$
U_{\mathrm{av}} = u_*, \qquad g_{b,\mathrm{ms}} = C_v \sqrt{\frac{u_*}{d_{\mathrm{leaf}}}}, \qquad g_{b,\mathrm{mol}} = g_{b,\mathrm{ms}} \frac{P}{R T},
$$

where $R$ is the universal gas constant converted via `unit_converter` (`ConversionContext(temperature=T, pressure=P)`). Stomatal conductance, photosynthesis, and energy balance remain owned by dedicated biosphere Functors (`stomatal_model_contract` in `biosphere_atmosphere.yaml` is explicitly marked `parameter_contract_only_no_production_medlyn_functor`).

### 2.6 Water-Carbon Physiological Modifier Coupling

`WaterCarbonCoupling` (`source_scale="grid_cell"`, `target_scale="grid_cell"`, `coupling_type="spatial"`, `VariableKind.DIAGNOSTIC`) maps volumetric `soil_moisture` $\theta \in [0, 1]$, `vapor_pressure_deficit` $D \ge 0$ ($\mathrm{Pa}$), and `soil_temperature` $T$ ($\mathrm{K}$) to dimensionless modifiers:

1. **Soil-moisture photosynthetic factor $f_{\mathrm{soil}}(\theta)$** (`stress_function_type`):
   - `"logistic"`: $f_{\mathrm{soil}} = \left[1 + \exp\!\left(-k_{\mathrm{log}}(\theta - \theta_{\mathrm{onset}})\right)\right]^{-1}$.
   - `"linear"`: $f_{\mathrm{soil}} = \operatorname{clamp}\!\left((\theta - \theta_{\mathrm{wilt}})/(\theta_{\mathrm{opt}} - \theta_{\mathrm{wilt}}),\; f_{\min},\; f_{\theta,\max}\right)$.
   - `"piecewise"`: $0$ for $\theta < \theta_{\mathrm{wilt}}$; $c_{\mathrm{part}}(\theta - \theta_{\mathrm{wilt}})/(\theta_{\mathrm{onset}} - \theta_{\mathrm{wilt}})$ on $[\theta_{\mathrm{wilt}}, \theta_{\mathrm{onset}})$; $(\theta - \theta_{\mathrm{onset}})/(\theta_{\mathrm{opt}} - \theta_{\mathrm{onset}})$ on $[\theta_{\mathrm{onset}}, \theta_{\mathrm{opt}})$; and $1$ for $\theta \ge \theta_{\mathrm{opt}}$.

2. **VPD hyperbolic factor and canopy combinations** (Jarvis, 1976; Lohammar et al., 1980; Leuning, 1995; Landsberg and Waring, 1997):

   $$
   f_{\mathrm{vpd}} = \frac{1}{1 + D / D_{50}}, \qquad
   f_{\mathrm{gpp}} = \operatorname{clamp}\!\left(f_{\mathrm{soil}} f_{\mathrm{vpd}},\; f_{\min},\; f_{\theta,\max}\right), \qquad
   f_{Ra} = \operatorname{clamp}\!\left(\sqrt{f_{\mathrm{gpp}}},\; f_{\min},\; 1\right), \qquad
   f_{\mathrm{overall}} = \operatorname{clamp}\!\left(\min(f_{\mathrm{soil}}, f_{\mathrm{vpd}}),\; f_{\min},\; 1\right).
   $$

3. **Respiration moisture and $Q_{10}$ temperature modifiers** (Davidson et al., 2012; Moyano et al., 2013; Sierra et al., 2015): With $\theta_{\mathrm{sat}} = \theta_{\mathrm{fc}}$, $\theta_{\mathrm{opt,r}} = \eta_{\mathrm{opt}}\theta_{\mathrm{sat}}$, $\theta_{\mathrm{anaer}} = \eta_{\mathrm{anaer}}\theta_{\mathrm{sat}}$, and shape exponent $p_{\mathrm{shape}}$ (`moisture_shape_parameter`), piecewise drought/optimal/anaerobic response $\tilde{f}_\theta$ is shaped as $f_{\theta,\mathrm{resp}} = \operatorname{clamp}(\tilde{f}_\theta^{1/p_{\mathrm{shape}}}, f_{\theta,\min}, f_{\theta,\max})$. With clamped $Q_{10}$ factor $f_T = \operatorname{clamp}(Q_{10}^{(T - T_{\mathrm{ref}})/10}, f_{T,\min}, f_{T,\max})$ and cap $c_T$:

   $$
   f_{Rh} = \operatorname{clamp}\!\left(f_{\theta,\mathrm{resp}}\min(f_T, c_T),\; f_{\min},\; f_{\theta,\max}\right), \qquad
   k_{\mathrm{dec}} = f_{\mathrm{soil\_carbon}} = f_{\theta,\mathrm{resp}}\,f_T.
   $$

## 3. Variable Table

| Variable / Key | Meaning | Units | Producer / Consumer | Role |
| --- | --- | --- | --- | --- |
| `soil_moisture_profile`, `soil_matric_potential`, `soil_depth_layers`, `potential_transpiration` | Layer $\theta_i$, $\psi_i$, bottoms $z_i$, demand $T_p$ (`root_density_profile` optional) | $\mathrm{m^3\,m^{-3}}$, $\mathrm{Pa}$, $\mathrm{m}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$ | Inputs to `SoilVegetationCoupling` | Coupling inputs |
| `root_uptake_profile`, `effective_transpiration` | Layer water sink $S_i$ and actual transpiration $T_{\mathrm{actual}}$ | $\mathrm{kg\,m^{-3}\,s^{-1}}$, $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `SoilVegetationCoupling` (`REPLACE`, `per_timestep=False`) | Flux ports (+ 4 `diagnostics.soil_vegetation.*`) |
| `friction_velocity`, `air_temperature`, `atmospheric_pressure` | Canonical $u_*$, $T$, and $P$ ($> 0$) | $\mathrm{m\,s^{-1}}$, $\mathrm{K}$, $\mathrm{Pa}$ | Inputs to `BiosphereAtmosphereCoupling` | Coupling inputs |
| `boundary_layer_conductance` | Bulk leaf boundary-layer conductance $g_{b,\mathrm{mol}}$ | $\mathrm{mol\,m^{-2}\,s^{-1}}$ | `BiosphereAtmosphereCoupling` (`REPLACE`, `per_timestep=False`) | Transport coefficient port |
| `soil_moisture`, `vapor_pressure_deficit`, `soil_temperature` | Volumetric moisture $\theta$, VPD $D$, soil temperature $T$ | $\mathrm{m^3\,m^{-3}}$, $\mathrm{Pa}$, $\mathrm{K}$ | Inputs to `WaterCarbonCoupling` | Coupling inputs |
| `gpp_water_stress_factor`, `autotrophic_respiration_modifier`, `heterotrophic_respiration_modifier`, `decomposition_rate_modifier`, `overall_water_stress`, `soil_carbon_turnover_factor` | Dimensionless physiological and soil-C modifiers ($f_{\mathrm{gpp}}$ consumed next step) | $1$ | `WaterCarbonCoupling` (+ 3 `diagnostics.water_carbon.*`) | Dimensionless modifiers |
| `<var>_uncovered_source_amount`, `diagnostics.conservative_remap.*` | Unmapped source mass/amount $U$ and covered-support ledger metrics | `[amount]`, $1$ | `ConservativeRemap`, `ConservativeRegridder`, `HierarchicalAggregator` | Coverage & ledger diagnostics |

## 4. Parameter Table

| Key | Units | Default | Source YAML | Consumer |
| --- | --- | --- | --- | --- |
| `coupling.scales.defaults.conservation_tolerance` | `dimensionless` | `1.0e-8` | `coupling/defaults.yaml` | `SpatialAggregation`, `ConservativeRemap`, `SpatialRegridderBase`, standalone `HierarchicalAggregator` |
| `coupling.scales.defaults.precision.<mode>.tolerance` | `dimensionless` | `1e-12` (`strict_conservative`), `1e-9` (`conservative`), `1e-6` (`balanced`), `1e-3` (`fast_approx`) | `coupling/scales_parameters.yaml` | `ScaleManager` facade (`aggregate`, `disaggregate`, `regrid_variable`) |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1.0e-6` | `validation/conservation.yaml` | `StateManager`, `SoilVegetationCoupling` port check, temporal mappers |
| `coupling.scales.defaults.epsilon`, `hierarchy_numerical_tolerance`, `roughness_blending_height` | `1`, `1`, `m` | `1.0e-15`, `1.0e-6`, `10.0` | `coupling/defaults.yaml` | Bilinear/IDW coincidence, `WeightOps.validate_weights`, roughness $z_r$ |
| `coupling.scales.defaults.auto_scale_mode`, `auto_scale_enforce_conservation` | `str`, `bool` | `"conservative"`, `True` | `coupling/defaults.yaml` | `ScaleManager` default mode and enforcement |
| `coupling.scales.spatial.aggregation_strategy`, `weighting_type`, `preserve_totals`, `boundary_handling_mode`, `regular_search_radius_cell_diagonals` | `str` / `bool` / `1` | `"area_weighted"`, `"area"`, `True`, `"constant"`, `2.0` | `coupling/scales_parameters.yaml` | `SpatialAggregation`, `BilinearRegridder`, `NearestNeighborRegridder` |
| `coupling.scales.spatial.irregular.max_search_radius`, `neighbor_count`, `weight_power` | `dimensionless` | `1.0`, `4`, `2.0` | `coupling/scales_parameters.yaml` | Irregular-grid IDW / nearest search |
| `coupling.scales.soil_vegetation.distribution_type`, `root_depth_max`, `shape_parameter`, `linear_distribution_coeff`, `beta_q_parameter` | `str`, `m`, `m-1`, `1`, `1` | `"exponential"`, `2.0`, `2.0`, `2.0`, `2.0` | `coupling/soil_vegetation.yaml` | `SoilVegetationCoupling` root profile |
| `coupling.scales.soil_vegetation.wilting_point_potential`, `optimal_range`, `anaerobic_point` | `Pa` | `-1.5e6`, `[-1.0e5, -1.0e4]`, `-1.0e3` | `coupling/soil_vegetation.yaml` | `SoilVegetationCoupling` Feddes stress |
| `coupling.scales.biosphere_atmosphere.leaf_characteristic_dimension`, `boundary_layer_transfer_coefficient` | `m`, `m s-0.5` | `0.05`, `0.01` | `coupling/biosphere_atmosphere.yaml` | `BiosphereAtmosphereCoupling` ($d_{\mathrm{leaf}}, C_v$) |
| `coupling.scales.water_carbon.stress_function_type`, `wilting_point_volumetric`, `field_capacity_volumetric`, `optimal_moisture`, `stress_onset_threshold`, `logistic_steepness`, `piecewise_partial_factor` | `str`, `m3 m-3`, `1` | `"logistic"`, `0.05`, `0.35`, `0.25`, `0.15`, `10.0`, `0.5` | `coupling/water_carbon.yaml` | `WaterCarbonCoupling` $f_{\mathrm{soil}}(\theta)$ |
| `coupling.scales.water_carbon.vpd_half_saturation`, `optimal_moisture_fraction`, `moisture_shape_parameter`, `anaerobic_threshold`, `drought_threshold` | `Pa`, `1`, `1`, `1`, `m3 m-3` | `1000.0`, `0.6`, `2.0`, `0.8`, `0.1` | `coupling/water_carbon.yaml` | `WaterCarbonCoupling` $f_{\mathrm{vpd}}$ & $f_{\theta,\mathrm{resp}}$ |
| `coupling.scales.water_carbon.q10_soil`, `reference_temperature_celsius`, `temperature_response_cap`, `temperature_response_min_bound`, `temperature_response_max_bound`, `minimum_stress_factor`, `minimum_moisture_factor`, `maximum_moisture_factor` | `1`, `degC`, `1` | `2.0`, `10.0`, `1.0`, `0.1`, `10.0`, `0.01`, `0.02`, `1.0` | `coupling/water_carbon.yaml` | `WaterCarbonCoupling` $Q_{10}$ and bounds |

## 5. Numerical Methods and Discretization

1. **Content-addressed regridder caching**: `SpatialRegridderFactory` keys cached instances by a SHA-256 `content_fingerprint` (`relic/coupling/scales/spatial/grid_utils.py`) over the regridder class, method, full `GridDescriptor` contents (coordinates, bounds, areas, CRS, masks), constructor options, and `ParameterManager` identity. Grid bounds are mandatory for conservative remapping and are never inferred from coordinate spacing.
2. **Simulator auto-regridding and CF synchronization**: When `enable_auto_scale=True` and `check_scale_consistency` detects mixed `SpatialScale` labels across a rate group's outputs, `perform_regridding` requires explicit `source_grid` and `target_grid` descriptors (declaring `grid_mapping`, `lat_name`, and `lon_name` in `CF_REGRID_TARGET_GRID_FIELDS`), selects `"nearest"` for `categorical` and `"conservative"` for all other bases under `enforce_conservation=True`, updates working-copy inputs without committing to `StateManager`, and synchronizes `cell_methods` (`area: sum`, `area: mean`, or `area: mode`) and `history` via `CFAttributeSynchronizer`.
3. **Differentiability boundary**: In-graph PyTorch operators (`ConservativeRegridder`, `BilinearRegridder`, `HierarchicalAggregator` continuous kernels, `SoilVegetationCoupling`, `BiosphereAtmosphereCoupling`, `WaterCarbonCoupling`) preserve reverse-mode gradients, whereas categorical `mode`/`nearest` and external NetCDF/xarray ingestion (`relic/coupling/external/common/regrid.py`) are non-differentiable.

## 6. Conservation, Boundary Conditions, and Ledgers

All spatial and hierarchical amount ledgers share the rounding-aware acceptance bound `ledger_acceptance` (`relic/coupling/core/conservation_checker.py`; Higham, 2002):

$$
\lvert L_{\mathrm{target}} - L_{\mathrm{cov}} \rvert \le \max\!\left(\mathrm{tol}\,\max(\lvert L_{\mathrm{cov}} \rvert, 10^{-12}),\; \gamma_k(u)\sum_i \lvert \ell_i \rvert\right), \qquad \gamma_k(u) = \frac{k u}{1 - k u},
$$

where $L_{\mathrm{cov}}$ is the covered-support source amount, $\ell_i$ are the covered source terms ($X_i$ or $x_i A_i$), $u$ is the unit roundoff of the coarsest operand/output floating dtype ($k = 2n + 2$ for whole-support/hierarchical reductions; $k = 2n + 8$ per batch row in `covered_support_acceptance`), and `tol` comes from the single owner for that call path (Section 4). Batch axes are audited row-by-row without cross-batch cancellation, and residuals are never hidden by post-hoc rescaling. Opt-in positivity or range clamps (`preserve_positivity=True` on `ConservativeRemap`, `range_clamp=True` under `preserve="none"` on `ConservativeRegridder`, or `positive_definite` on `RegridResult`) disclose signed clip losses (`requested - clipped`) explicitly.

For internal cross-domain ports:
- `SoilVegetationCoupling` verifies port closure $R_{\mathrm{water}} = \sum_i S_i \Delta z_i - T_{\mathrm{actual}} = 0$ within `validation.conservation.conservation_tolerance`. It never mutates `soil_moisture` directly, and no registered soil-water Functor currently consumes `root_uptake_profile`; closing a full soil-column water budget with root extraction requires binding a downstream Richards or bucket sink consumer exactly once.
- `BiosphereAtmosphereCoupling` and `WaterCarbonCoupling` emit transport coefficients and dimensionless stress multipliers (`conservation_laws = []`); `WaterCarbonCoupling.check_conservation_laws` audits definitional modifier identities and $[0, 1]$ stress-factor bounds.

## 7. Code Mapping

| Class / Function | File Path | Architectural Role |
| --- | --- | --- |
| `ScaleCouplingFunctor` | `relic/core/functor/scale_coupling_bridge.py` | Bridge connecting `ScaleCoupling` to `FunctorBase` |
| `SpatialQuantityBasis`, `convert_spatial_basis` | `relic/coupling/scales/types.py` | Four-basis spatial representation contract |
| `SpatialAggregation`, `ConservativeRemap` | `relic/coupling/scales/adapters.py` | Whole-support reduction and overlap-area conservative remap Functors |
| `ConservativeRegridder`, `BilinearRegridder`, `NearestNeighborRegridder`, `SpatialRegridderFactory` | `relic/coupling/scales/spatial/` | Grid-descriptor regridders and SHA-256 content-fingerprint factory |
| `HierarchicalAggregator`, `HierarchyIndex` | `relic/coupling/scales/hierarchical/` | Nested child-parent aggregation/disaggregation and membership index |
| `ScaleManager` | `relic/coupling/scales/scale_manager.py` | Governed facade with mode precision tolerances, CF sync, and `AuditCollector` |
| `ledger_acceptance` | `relic/coupling/core/conservation_checker.py` | Higham $\gamma_k(u)$-aware covered-support amount ledger validator |
| `SupportAggregationAdapter`, `ExecutableAdapterFunctor` | `relic/orchestration/adapters.py` | Compiled graph-edge support and LAI area-basis adapters |
| `SoilVegetationCoupling`, `BiosphereAtmosphereCoupling`, `WaterCarbonCoupling` | `relic/coupling/internal/` | Internal soil-plant, canopy-leaf, and water-carbon coupling Functors |

## 8. Implementation Status

- **Implemented (`[I]`)**: Four-basis `SpatialQuantityBasis` enforcement, Higham rounding-aware covered-support ledgers with non-redistributed `uncovered_source_amount`, spherical `latlon` and planar `ConservativeRegridder`/`BilinearRegridder`/`NearestNeighborRegridder`, SHA-256 `SpatialRegridderFactory` caching, `HierarchicalAggregator` (including CMAQ `effective_roughness_length`), `ScaleManager` precision-policy governance with CF-1.8 `cell_methods` synchronization, `SupportAggregationAdapter`, `SoilVegetationCoupling` Feddes port, `BiosphereAtmosphereCoupling` CTSM bulk leaf conductance port, and `WaterCarbonCoupling` modifiers.
- **Metadata-only / Unbound (`[M]` / `[N]`)**: `stomatal_model_contract` parameters in `biosphere_atmosphere.yaml` have no production Medlyn Functor in `relic/coupling/internal/`; `root_uptake_profile` has no active soil-moisture sink consumer in default hydrosphere profiles; temporal alias pairs (`daily` to/from `monthly`) in `ScaleManager` raise `RuntimeError` (`temporal_alias_periodization_unbound`, `temporal_alias_disaggregation_unbound`); and spherical polygon-intersection conservative remapping is `[N]`.
- **IPHS Equivalence Certificates**: None (`0` certificates reference `scale_coupling.md`; individual scientific process certificates reside on their domain theory pages).

## 9. References

1. Bonan, G. B., Williams, M., Fisher, R. A., and Oleson, K. W. (2014). Modeling stomatal conductance in the Earth system: Linking leaf water-use efficiency and water transport along the soil-plant-atmosphere continuum. *Geoscientific Model Development*, 7(5), 2193-2222.
2. Byun, D. W., and Ching, J. K. S. (Eds.) (1999). *Science Algorithms of the EPA Models-3 Community Multiscale Air Quality (CMAQ) Modeling System* (EPA/600/R-99/030, Chapter 12). U.S. Environmental Protection Agency.
3. Campbell, G. S., and Norman, J. M. (1998). *An Introduction to Environmental Biophysics* (2nd ed.). Springer.
4. Davidson, E. A., Samanta, S., Caramori, S. S., and Savage, K. (2012). The Dual Arrhenius and Michaelis-Menten (DAMM) model, a core biophysical model of microbial respiration regulation. *Global Change Biology*, 18(1), 371-384.
5. Feddes, R. A., Kowalik, P. J., and Zaradny, H. (1978). *Simulation of Field Water Use and Crop Yield*. Pudoc, Wageningen.
6. Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
7. Jackson, R. B., Canadell, J., Ehleringer, J. R., Mooney, H. A., Sala, O. E., and Schulze, E. D. (1996). A global analysis of root distributions for terrestrial biomes. *Oecologia*, 108(3), 389-411.
8. Jarvis, P. G. (1976). The interpretation of the variations in leaf water potential and stomatal conductance found in canopies in the field. *Philosophical Transactions of the Royal Society of London B*, 273(927), 593-610.
9. Jones, P. W. (1999). First- and second-order conservative remapping schemes for grids in spherical coordinates. *Monthly Weather Review*, 127(9), 2204-2210.
10. Kljun, N., Calanca, P., Rotach, M. W., and Schmid, H. P. (2015). A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP). *Geoscientific Model Development*, 8(11), 3695-3713.
11. Landsberg, J. J., and Waring, R. H. (1997). A generalised model of forest productivity using simplified concepts of radiation-use efficiency, carbon balance and partitioning. *Forest Ecology and Management*, 95(3), 209-228.
12. Leuning, R. (1995). A critical appraisal of a combined stomatal-photosynthesis model for $\mathrm{C_3}$ plants. *Plant, Cell & Environment*, 18(4), 339-355.
13. Lohammar, T., Larsson, S., Linder, S., and Falk, S. O. (1980). FAST—Simulation models of gaseous exchange in Scots pine. *Ecological Bulletins*, 32, 505-523.
14. Moyano, F. E., Manzoni, S., and Chenu, C. (2013). Responses of soil heterotrophic respiration to moisture availability: An exploration of processes and models. *Soil Biology and Biochemistry*, 59, 72-85.
15. Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. *Proceedings of the 1968 23rd ACM National Conference*, 517-524.
16. Sierra, C. A., Trumbore, S. E., Davidson, E. A., Vicca, S., and Janssens, I. (2015). Sensitivity of decomposition rates of soil organic matter with respect to simultaneous changes in temperature and moisture. *Journal of Advances in Modeling Earth Systems*, 7(1), 335-356.
17. Skaggs, T. H., van Genuchten, M. T., Shouse, P. J., and Poss, J. A. (2006). Macroscopic approaches to root water uptake as a function of water and salinity stress. *Agricultural Water Management*, 86(1-2), 140-149.
