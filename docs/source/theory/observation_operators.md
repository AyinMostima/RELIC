# Observation Operators and Post-Observation Quality Control

## 1. Process Definition and Scope

`relic/coupling/observation/` implements five read-only `ScaleCoupling` observation and post-observation quality Functors (`TowerFluxSupportOperator`, `TowerConcentrationProfileOperator`, `PostObservationQualityOperator`, `HchoSteadyStateColumnOperator`, and `SatelliteColumnKernelOperator`), all inheriting from `ScaleCouplingFunctor` at `ExecutionPhase.POSTPROCESSING` with `VariableKind.DIAGNOSTIC` and `StateWriteMode.DERIVED`. Theoretical foundations for typed observation morphisms, spatio-temporal support projection, and non-destructive quality control appear in *Methods of Ecological Analysis*, Chapters 2, 23, 27, and 30.

Observation operators project prognostic or flux states onto instrument-equivalent supports via `ObservationOperatorSpec` and `DiagnosticSpec` (`relic/computation/operators/dec/forms.py`) without introducing a second runtime, scheduler, or state store, and without mutating prognostic states (closed-loop state assimilation is isolated in `ObservationStateFeedbackFunctor`, documented on [observation_state_feedback.md](observation_state_feedback.md)). Flux observations ($H_{\mathrm{flux}}$), tower concentration profiles ($H_{\mathrm{conc}}$), and satellite column retrievals ($H_{\mathrm{HCHO}}$, $H_{\mathrm{AK}}$) have distinct physical supports and are never interchangeable.

## 2. Governing Equations

### 2.1 Eddy-Covariance Flux-Footprint Identity (`TowerFluxSupportOperator`)

For surface molar flux density $F_s(x, y)$ ($\mathrm{mol\,m^{-2}\,s^{-1}}$) and receptor flux-footprint density $f(x, y; z_m, \xi)$ ($\mathrm{m^{-2}}$) parameterized by receptor height $z_m > 0$ and boundary-layer state $\xi$ (Finnigan, 2004; Aubinet, Vesala, and Papale, 2012; Kljun et al., 2015), discrete cell-integrated kernel weights over cells $A_i$ are

$$
w_i^{\mathrm{raw}} = \int_{A_i} f(x, y; z_m, \xi)\,dA, \qquad C = \sum_i w_i^{\mathrm{raw}}, \qquad \alpha_i = \frac{w_i^{\mathrm{raw}}}{C}, \qquad \sum_i \alpha_i = 1.
$$

Temporal window resampling executes upstream in `TemporalResampleAdapter`, and spatial reduction $F_{\mathrm{fp}} = \sum_i \alpha_i \overline{F}_{s,i}$ executes in `SupportAggregationAdapter` (`relic/orchestration/adapters.py`). Conditional on full kernel-mass closure ($\lvert C - 1\rvert \le \varepsilon_{\mathrm{fp}}$), weight normalization ($\lvert\sum_i \alpha_i - 1\rvert \le \varepsilon_{\mathrm{fp}}$), valid-data coverage ($\lvert f_{\mathrm{cov}} - 1\rvert \le \varepsilon_{\mathrm{fp}}$), and negligible storage/advection/chemical divergence, `TowerFluxSupportOperator` evaluates the unit-preserving identity without an empirical multiplicative height factor:

$$
H_{\mathrm{flux}}(F_{\mathrm{fp}}) = F_{\mathrm{tower}} = F_{\mathrm{fp}}, \qquad \frac{\partial F_{\mathrm{tower}}}{\partial F_{\mathrm{fp}}} = 1, \qquad A_{\mathrm{supp}} = \sum_{i : \alpha_i > 0} A_i.
$$

### 2.2 Tower Concentration Profile and Volumetric Source Adapter (`TowerConcentrationProfileOperator`)

In the bounded 1D inert-tracer column chain, `SurfaceFluxToVolumeSourceAdapter` (`relic/orchestration/adapters.py`) injects surface molar flux $F_k$ ($\mathrm{mol\,m^{-2}\,s^{-1}}$) with molar mass $M_{\mathrm{iso}}$ ($\mathrm{kg\,mol^{-1}}$) and bottom area $A_k$ ($\mathrm{m^2}$) into lowest finite-volume cell $V_{k,0}$ ($\mathrm{m^3}$):

$$
S_{k,0} = \frac{F_k M_{\mathrm{iso}} A_k}{V_{k,0}}, \qquad S_{k,l>0} = 0, \qquad \sum_l S_{k,l} V_{k,l} = F_k M_{\mathrm{iso}} A_k.
$$

Given post-transport singleton-column mass concentration $\rho_l$ (`shape [1, layer]`, $\mathrm{kg\,m^{-3}}$), temperature $T_l$ ($\mathrm{K}$), universal gas constant $R$, and explicitly selected pressure basis $p_{\mathrm{basis},l}$ (`atmospheric_pressure` for `air_basis="wet"`, `dry_air_pressure` for `air_basis="dry"`, $\mathrm{Pa}$; RH is never used to infer dry pressure), `TowerConcentrationProfileOperator` evaluates three independent receptor rows ($j \in \{0, 1, 2\}$) under two non-commuting vertical-support modes:

1. **Point linear reconstruction (`support_mode="point_linear"`)** using two-point layer-center interpolation row $W_j$ ($\sum_l W_{j,l} = 1$) applied to primitive fields prior to nonlinear ideal-gas conversion:

   $$\chi_j^{\mathrm{point}} = 10^9\,\frac{(W_j \rho)\,R\,(W_j T)}{M_{\mathrm{iso}}\,(W_j p_{\mathrm{basis}})}.$$

2. **Finite layer-overlap averaging (`support_mode="finite_layer_overlap"`)** over interval $B_j = [z_j - \Delta z_j, z_j + \Delta z_j]$ and interfaces $z_{l \pm 1/2}$, averaging locally converted layer mole fractions:

   $$w_{j,l}^{\mathrm{raw}} = \max\!\left(0,\; \min(z_j + \Delta z_j, z_{l+1/2}) - \max(z_j - \Delta z_j, z_{l-1/2})\right), \qquad w_{j,l} = \frac{w_{j,l}^{\mathrm{raw}}}{\sum_m w_{j,m}^{\mathrm{raw}}},$$

   $$\chi_j^{\mathrm{interval}} = \sum_l w_{j,l}\left(10^9\,\frac{\rho_l R T_l}{M_{\mathrm{iso}} p_{\mathrm{basis},l}}\right).$$

Across model step intervals $I_n$ overlapping active non-overlapping observation window $W$ of duration $D_{\mathrm{win}}$ by $\Delta t_n = \lvert W \cap I_n\rvert$, checkpointed registered buffers accumulate the exact duration-weighted window mean:

$$
\overline{\chi}_j = \frac{\sum_n \chi_{j,n}\,\Delta t_n}{\sum_n \Delta t_n}, \qquad c_{\mathrm{win}} = \frac{\sum_n \Delta t_n}{D_{\mathrm{win}}}.
$$

### 2.3 Non-Destructive Post-Observation Bitmask Classification (`PostObservationQualityOperator`)

Consuming `raw_ppbv` ($x_i$) and `window_complete` ($c_i \in \{0, 1\}$) directly from `subprocess_outputs["tower_concentration_profile_operator"]` alongside state gate `instrument_valid` ($v_i \in \{0, 1\}$) and registered 3-receptor instrument contract $(\mathrm{LOD}_i, x_{\min,i}, x_{\max,i}, w_{\mathrm{base},i})$, `PostObservationQualityOperator` forms bitwise-OR integer flags (CF-1.8 `flag_masks`; WMO, 2021):

$$
\mathrm{flag}_i = 1\,\mathbb{I}[\neg\mathrm{finite}(x_i)] \;\lor\; 2\,\mathbb{I}[\mathrm{finite}(x_i) \land x_i < \mathrm{LOD}_i] \;\lor\; 4\,\mathbb{I}[\mathrm{finite}(x_i) \land (x_i < x_{\min,i} \lor x_i > x_{\max,i})] \;\lor\; 8\,\mathbb{I}[\neg c_i] \;\lor\; 16\,\mathbb{I}[\neg v_i],
$$

$$
q_i = (\mathrm{flag}_i == 0), \qquad w_i = w_{\mathrm{base},i}\,\mathbb{I}[q_i].
$$

### 2.4 Steady-State Formaldehyde Column Operator (`HchoSteadyStateColumnOperator`)

Following Palmer et al. (2003) and Marais et al. (2012, Eqs. 2 and 4, Fig. 9), `HchoSteadyStateColumnOperator` maps molecular isoprene emission $E \ge 0$ ($\mathrm{mol\,C_5H_8\,m^{-2}\,s^{-1}}$) and caller-supplied non-isoprene background column $B$ ($\mathrm{mol\,m^{-2}}$) to steady-state HCHO vertical column $\Omega_{\mathrm{HCHO}}$ ($\mathrm{mol\,m^{-2}}$) using carbon stoichiometry $n_C$ (`atmosphere.chemistry.isoprene_carbon_atoms` $= 5$):

$$
\Omega_{\mathrm{HCHO}} = S(\Omega_{\mathrm{NO_2}})\,n_C\,E + B, \qquad
S(\Omega_{\mathrm{NO_2}}) = \begin{cases} m\,\Omega_{\mathrm{NO_2}} + c, & \Omega_{\mathrm{NO_2}} \le \Omega_t, \\ S_{\mathrm{high}}, & \Omega_{\mathrm{NO_2}} > \Omega_t, \end{cases}
$$

where tropospheric $\mathrm{NO_2}$ column $\Omega_{\mathrm{NO_2}}$ (supplied in `"molecule cm-2"` or converted from `"mol m-2"` via $\Omega[\mathrm{molecule\,cm^{-2}}] = \Omega[\mathrm{mol\,m^{-2}}]\,N_A \times 10^{-4}$) switches regime at $\Omega_t = 1.0 \times 10^{15}\,\mathrm{molecule\,cm^{-2}}$. Pixels with $\Omega_{\mathrm{NO_2}} < \Omega_{\mathrm{low}} = 2.0 \times 10^{14}\,\mathrm{molecule\,cm^{-2}}$ remain unclipped on the linear branch and set boolean diagnostic `no2_below_calibrated_range = True`.

### 2.5 Hybrid-Pressure Averaging-Kernel Column Operator (`SatelliteColumnKernelOperator`)

For surface-first hybrid-pressure retrieval layers $k = 0, \dots, L-1$ with strictly decreasing centres $p_k = a_k + b_k p_s$ ($p_s > p_0 > \dots > p_{L-1} > 0$, $\mathrm{Pa}$), layer edges $p_{e,0} = p_s$, $p_{e,L} = 0$, and interior edges ($k = 1, \dots, L-1$) use `"arithmetic_mean"` or `"log_pressure"` midpoints:

$$
p_{e,k}^{\mathrm{arith}} = \frac{p_{k-1} + p_k}{2}, \qquad p_{e,k}^{\mathrm{log}} = \sqrt{p_{k-1} p_k}.
$$

Hydrostatic dry-air partial columns $n_k$ and tracer partial columns $x_{\mathrm{pc},k}$ ($\mathrm{mol\,m^{-2}}$) for mixing ratio $\chi_k \ge 0$ ($\mathrm{mol\,mol^{-1}}$) are

$$
n_k = \frac{p_{e,k} - p_{e,k+1}}{g\,M_{\mathrm{air}}}, \qquad x_{\mathrm{pc},k} = \chi_k\,n_k.
$$

When remapping source partial columns $x_i^{\mathrm{src}}$ on edges $p_e^{\mathrm{src}}$ to product edges $p_e^{\mathrm{tgt}}$, `remap_partial_columns` applies 1D pressure-overlap conservation (Jones, 1999) with $O_{ij} = \max(0, \min(p_{e,i}^{\mathrm{src}}, p_{e,j}^{\mathrm{tgt}}) - \max(p_{e,i+1}^{\mathrm{src}}, p_{e,j+1}^{\mathrm{tgt}}))$ and $\Delta p_i^{\mathrm{src}} = p_{e,i}^{\mathrm{src}} - p_{e,i+1}^{\mathrm{src}}$:

$$
x_j^{\mathrm{tgt}} = \sum_i x_i^{\mathrm{src}}\,\frac{O_{ij}}{\Delta p_i^{\mathrm{src}}}, \qquad x_{\mathrm{unmapped}} = \sum_i x_i^{\mathrm{src}}\,\frac{\Delta p_i^{\mathrm{src}} - \sum_j O_{ij}}{\Delta p_i^{\mathrm{src}}}.
$$

Applying product averaging kernel $A_l = m_l / M$ over tropopause layer subset $T$ (`"inclusive"`: $l \le l_t$; `"exclusive"`: $l < l_t$; `"full_column"`: all $L$ layers, for 0-based `tropopause_layer_index` $l_t$) yields the a-priori-free retrieval-equivalent column (Eskes and Boersma, 2003; De Smedt et al., 2018):

$$
N_{v,\mathrm{model}} = \sum_{l \in T} A_l\,x_{\mathrm{pc},l}.
$$

## 3. Variable Table

| Operator | Key Inputs | Key Outputs (`StateWriteMode.DERIVED`) | Units |
| --- | --- | --- | --- |
| `TowerFluxSupportOperator` | `diagnostics.observation.tower_flux.footprint_weighted_isoprene_flux`, `footprint_weights`, `footprint_kernel_mass`, `footprint_cell_area`, `footprint_data_coverage_fraction`, `measurement_height`, `time_bounds`, `footprint_density_available`, `flux_footprint_valid` | `diagnostics.observation.tower_flux.{tower_equivalent_flux, support_integral, support_area, weight_sum, coverage_fraction, support_extrapolation_flag}` | $\mathrm{mol\,m^{-2}\,s^{-1}}$, $1$, $\mathrm{m^2}$ |
| `TowerConcentrationProfileOperator` | `tracer_concentration` (`[1, layer]`), `air_temperature`, `atmospheric_pressure` or `dry_air_pressure`, `layer_center_height` or `layer_interface_height` | `diagnostics.observation.tower_concentration.{raw_ppbv, support_weight_sum, window_coverage, window_complete}` | $\mathrm{ppbv}$, $1$ |
| `PostObservationQualityOperator` | `raw_ppbv`, `window_complete` (from upstream dependency), `observation.tower_concentration.instrument_valid` | `diagnostics.observation.tower_concentration.{mask, flag, weight}` | $1$ (`bool`, `int64`, `float`) |
| `HchoSteadyStateColumnOperator` | `observation.hcho_column.{isoprene_emission, no2_column, background_column}` | `diagnostics.observation.hcho_column.{steady_state_column, emission_column_sensitivity, no2_linear_regime, no2_below_calibrated_range}` | $\mathrm{mol\,m^{-2}}$, $\mathrm{s}$, `bool` |
| `SatelliteColumnKernelOperator` | `observation.satellite_column.{hybrid_coefficient_a, hybrid_coefficient_b, surface_pressure, model_mixing_ratio, averaging_kernel, tropopause_layer_index}` | `diagnostics.observation.satellite_column.{model_column, air_partial_column}` | $\mathrm{mol\,m^{-2}}$ |

## 4. Parameter Table

| Symbol | `ParameterManager` / Physical-Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $\varepsilon_{\mathrm{fp}}$ | `coupling.observation.tower_flux.support_closure_tolerance` | $1$ | `coupling/observation.yaml` (`1.0e-6`) | Kljun et al. (2015) |
| $z_{\mathrm{meas}}, \Delta z_{\mathrm{half}}$ | `coupling.observation.tower_concentration.{synthetic_measurement_heights, synthetic_support_half_widths}` | $\mathrm{m}$ | `coupling/observation.yaml` (`[10, 30, 50]`, `[5, 5, 5]`) | Stull (1988) |
| $D_{\mathrm{win}}, \varepsilon_{\mathrm{conc}}$ | `coupling.observation.tower_concentration.{window_duration, support_closure_tolerance}` | $\mathrm{s}$, $1$ | `coupling/observation.yaml` (`1800.0`, `1.0e-9`) | CF-1.8; Higham (2002) |
| $\mathrm{LOD}, x_{\min}, x_{\max}, w_{\mathrm{base}}$ | `<prefix>.{detection_limit_ppbv, valid_min_ppbv, valid_max_ppbv, base_weight}` (synthetic fixture: `coupling.observation.tower_concentration.profile_004_synthetic_qc.*`) | $\mathrm{ppbv}$, $1$ | `coupling/observation.yaml` (`[0.001]*3`, `[0]*3`, `[1000]*3`, `[1]*3`) | WMO (2021); CF-1.8 |
| $m, c, S_{\mathrm{high}}$ (`standard`) | `coupling.observation.hcho_steady_state_column.marais2012_standard.{low_nox_slope, low_nox_intercept, high_nox_constant}` (+ `*_standard_deviation`) | $\mathrm{s\,cm^2\,molecule^{-1}}$, $\mathrm{s}$ | `coupling/observation.yaml` ($1.34\times 10^{-12} \pm 1.2\times 10^{-13}$, $523 \pm 86$, $1870 \pm 251$) | Marais et al. (2012) |
| $m, c, S_{\mathrm{high}}$ (`paulot`) | `coupling.observation.hcho_steady_state_column.marais2012_paulot.{low_nox_slope, low_nox_intercept, high_nox_constant}` (+ `*_standard_deviation`) | $\mathrm{s\,cm^2\,molecule^{-1}}$, $\mathrm{s}$ | `coupling/observation.yaml` ($1.60\times 10^{-12} \pm 1.6\times 10^{-13}$, $721 \pm 114$, $2230 \pm 174$) | Marais et al. (2012) |
| $\Omega_t, \Omega_{\mathrm{low}}$ | `coupling.observation.hcho_steady_state_column.{no2_regime_threshold, calibrated_no2_lower_end}` | $\mathrm{molecule\,cm^{-2}}$ | `coupling/observation.yaml` (`1.0e15`, `2.0e14`) | Marais et al. (2012) |
| $n_C$ | `atmosphere.chemistry.isoprene_carbon_atoms` | $1$ | `atmosphere/chemistry.yaml` (`5`) | Stoichiometric constant |
| $\varepsilon_{\mathrm{remap}}$ | `coupling.scales.defaults.conservation_tolerance` | $1$ | `coupling/defaults.yaml` | Jones (1999) |
| $R, M_{\mathrm{iso}}, N_A, g, M_{\mathrm{air}}$ | `get_physical_constant(...)` (`gas_constant`, `relic.physical.constants.molecular_weights.isoprene`, `avogadro_constant`, `gravity_acceleration`, `air_molecular_weight`) | SI / $\mathrm{g\,mol^{-1}}$ | `physical/constants.yaml` ($10^{-3}$ converts $\mathrm{g\,mol^{-1}}$ to $\mathrm{kg\,mol^{-1}}$) | CODATA 2018 |

## 5. Numerical Methods and Discretization

- **Fail-closed support validation**: `TowerFluxSupportOperator` raises if any boolean validity gate is false, if `footprint_weighted_isoprene_flux` is simultaneously present in `StateManager` and `subprocess_outputs` (ambiguous source), or if $C$, $\sum_i \alpha_i$, or $f_{\mathrm{cov}}$ deviates from $1$ by more than $\varepsilon_{\mathrm{fp}}$.
- **Stateful window buffers and restart safety**: `TowerConcentrationProfileOperator` registers `_window_numerator`, `_window_covered_duration`, `_active_window_bounds`, `_window_initialized`, and `_last_step_end` as PyTorch buffers included in `state_dict()` and failed-step rollback transactions. Multi-cell horizontal inputs (`shape[0] != 1`) or out-of-column receptor supports fail closed without implicit horizontal averaging or extrapolation.
- **No inline QC thresholds or silent clipping**: `PostObservationQualityOperator` rejects any `config` key other than `instrument_contract_parameter_prefix` and `"dependencies"`, enforcing $0 \le x_{\min,i} \le \mathrm{LOD}_i \le x_{\max,i}$, $x_{\min,i} < x_{\max,i}$, and $w_{\mathrm{base},i} \ge 0$. Neither `PostObservationQualityOperator` nor `HchoSteadyStateColumnOperator` clips, masks, or gap-fills raw physical predictions; `SatelliteColumnKernelOperator` imposes no positivity constraint on averaging-kernel elements $A_l$ or retrieved columns, and `scale_profile_shape` requires an explicit normalization layer set $K$ with no default.

## 6. Conservation, Boundary Conditions, and Ledgers

All five observation operators emit `UpdateItem(mode=UpdateMode.REPLACE, write_mode=StateWriteMode.DERIVED, per_timestep=False)` and have empty `conservation_laws=[]`, leaving all `StateManager` prognostic mass and energy ledgers untouched. Two auxiliary mass-conservation contracts are enforced strictly on intermediate transfers:
1. `SurfaceFluxToVolumeSourceAdapter` preserves column and domain mass rates $\sum_{k,l} S_{k,l} V_{k,l} = \sum_k F_k M_{\mathrm{iso}} A_k$, and graphs that simultaneously inject a non-zero lower diffusive boundary flux are invalid.
2. `SatelliteColumnKernelOperator.remap_partial_columns` returns `PartialColumnRemap(partial_columns, unmapped_column)` and enforces $\lvert \sum_j x_j^{\mathrm{tgt}} + x_{\mathrm{unmapped}} - \sum_i x_i^{\mathrm{src}} \rvert \le \varepsilon_{\mathrm{remap}} \sum_i \lvert x_i^{\mathrm{src}} \rvert$, preventing silent mass loss when model and product surface pressures differ.

## 7. Code Mapping

| Functor / Adapter Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `TowerFluxSupportOperator` | `relic/coupling/observation/tower_flux.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.POSTPROCESSING` |
| `TowerConcentrationProfileOperator` | `relic/coupling/observation/tower_concentration.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.POSTPROCESSING` |
| `PostObservationQualityOperator` | `relic/coupling/observation/quality.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.POSTPROCESSING` |
| `HchoSteadyStateColumnOperator` | `relic/coupling/observation/hcho_steady_state_column.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.POSTPROCESSING` |
| `SatelliteColumnKernelOperator` | `relic/coupling/observation/satellite_column_kernel.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.POSTPROCESSING` |
| `TemporalResampleAdapter`, `SupportAggregationAdapter`, `SurfaceFluxToVolumeSourceAdapter` | `relic/orchestration/adapters.py` | Adapter / `ScaleCoupling` | `ExecutableAdapterFunctor` (`ScaleCouplingFunctor`) | Upstream compiled adapter step |

## 8. Implementation Status

- **Implemented (`[I]`)**: Validated flux-footprint identity operator, 3-height `point_linear` and `finite_layer_overlap` tower concentration operator with checkpointed duration-weighted window buffers, non-destructive 5-bit `PostObservationQualityOperator`, dual-scheme (`marais2012_standard`, `marais2012_paulot`) steady-state HCHO column operator, and hybrid-pressure `SatelliteColumnKernelOperator` with conservative 1D pressure-overlap remapping and 3 tropopause truncation modes.
- **Not implemented (`[N]`)**: Dynamic 2D footprint-kernel PDE/Lagrangian generation inside `TowerFluxSupportOperator` (weights must be supplied upstream), full non-uniform 3D PBL/canopy reactive chemical transport inside the synthetic 1D concentration chain, automatic jackknife uncertainty propagation inside `HchoSteadyStateColumnOperator`, and moist-air virtual-temperature hydrostatic corrections inside `SatelliteColumnKernelOperator`.
- **IPHS Equivalence Certificates**: None (`0` certificates reference `observation_operators.md`; read-only diagnostic projections do not advance dissipative/Hamiltonian state dynamics).

## 9. References

1. Aubinet, M., Vesala, T., and Papale, D. (Eds.) (2012). *Eddy Covariance: A Practical Guide to Measurement and Data Analysis*. Springer.
2. De Smedt, I., Theys, N., Yu, H., Danckaert, T., Lerot, C., et al. (2018). Algorithm theoretical baseline for formaldehyde retrievals from S5P TROPOMI and from the QA4ECV project. *Atmospheric Measurement Techniques*, 11(4), 2395-2426.
3. Eskes, H. J., and Boersma, K. F. (2003). Averaging kernels for DOAS total-column satellite retrievals. *Atmospheric Chemistry and Physics*, 3(5), 1285-1291.
4. Finnigan, J. (2004). The footprint concept in complex terrain. *Agricultural and Forest Meteorology*, 127(3-4), 117-129.
5. Jones, P. W. (1999). First- and second-order conservative remapping schemes for grids in spherical coordinates. *Monthly Weather Review*, 127(9), 2204-2210.
6. Kljun, N., Calanca, P., Rotach, M. W., and Schmid, H. P. (2015). A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP). *Geoscientific Model Development*, 8(11), 3695-3713.
7. Marais, E. A., Jacob, D. J., Kurosu, T. P., Chance, K., Murphy, J. G., et al. (2012). Isoprene emissions in Africa inferred from OMI observations of formaldehyde columns. *Atmospheric Chemistry and Physics*, 12(14), 6219-6235.
8. Palmer, P. I., Jacob, D. J., Fiore, A. M., Martin, R. V., Chance, K., and Kurosu, T. P. (2003). Mapping isoprene emissions over North America using formaldehyde column observations from space. *Journal of Geophysical Research: Atmospheres*, 108(D6), 4180.
9. Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
10. Seinfeld, J. H., and Pandis, S. N. (2016). *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change* (3rd ed.). Wiley.
11. Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers.
12. World Meteorological Organization (2021). *Guide to Instruments and Methods of Observation* (WMO-No. 8). WMO.
