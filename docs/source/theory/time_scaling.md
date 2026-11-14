# Temporal Scaling: Interval Overlap Kernels, Calendar Arithmetic, and Quality Control

## 1. Process Definition and Scope

`relic/coupling/scales/temporal/` implements explicit-bounds temporal aggregation (`TemporalAggregator`), conservative piecewise-constant temporal disaggregation (`TemporalDisaggregator`), fixed-SI resampling (`TemporalScaling`), CF calendar and month-ordinal arithmetic (`CFTimeProcessor`, `CalendarHandler`, `PeriodIndexer`), bounded CF-1.13 time-coordinate metadata synchronization (`create_temporal_rescaling_metadata`), and standalone pre-ingestion series quality control (`QualityController`, `CoverageEvaluator`, `GapFiller`, `OutlierDetector`). Single-step instantaneous-to-interval-mean adaptation inside compiled graphs resides in `TemporalResampleAdapter` (`relic/orchestration/adapters.py`). Theoretical derivations for measure-preserving temporal projection and multi-calendar support appear in *Methods of Ecological Analysis*, Chapters 21 and 23.

`TemporalAggregator`, `TemporalDisaggregator`, and `TemporalScaling` fulfill the **`ScaleCoupling`** abstraction (`TemporalScalingBase`) via `ScaleCouplingFunctor` (`SpatialScale.GRID_CELL`, `VariableKind.DIAGNOSTIC`). All interval-overlap mappings require explicit half-open interval bounds, explicit `temporal_semantics`, and explicit units; bounds and semantics are never inferred from timestamps or variable names, and mapping operators never perform silent gap interpolation.

## 2. Governing Equations

### 2.1 Half-Open Overlap Kernel and Support Coverage

Given non-overlapping half-open source intervals $I_i = [t_i^-, t_i^+)$ of duration $\delta_i = t_i^+ - t_i^- > 0$ and target intervals $J_j = [T_j^-, T_j^+)$ of duration $\Delta_j = T_j^+ - T_j^- > 0$ sharing one calendar and one CF reference epoch, the exact overlap duration and target-fraction weight are:

$$
O_{ij} = \max\!\left(0,\; \min(t_i^+, T_j^+) - \max(t_i^-, T_j^-)\right), \qquad w_{ij} = \frac{O_{ij}}{\Delta_j}.
$$

With binary validity mask $v_i \in \{0, 1\}$ ($v_i = 0$ for `NaN` or excluded `valid_mask` entries; infinite values raise `ValidationError`), geometric coverage and valid coverage are:

$$
C_j^{\mathrm{geom}} = \sum_i w_{ij} \le 1, \qquad C_j^{\mathrm{valid}} = \sum_i v_i w_{ij}.
$$

Any $C_j^{\mathrm{geom}} > 1$ indicates overlapping source intervals and raises `ValidationError`. Geometric gap $1 - C_j^{\mathrm{geom}}$, missing-data gap $C_j^{\mathrm{geom}} - C_j^{\mathrm{valid}}$, and total gap $1 - C_j^{\mathrm{valid}}$ are reported without rescaling mapped values.

### 2.2 Aggregation Laws by `temporal_semantics`

`TemporalAggregator` (`TemporalConfig.direction == "upscale"`) admits targets satisfying $C_j^{\mathrm{valid}} \ge \theta_{\mathrm{cov}}$ (`coupling.scales.temporal.coverage_threshold`, except extrema/range which require only $C_j^{\mathrm{valid}} > 0$) and outputs `NaN` elsewhere:

1. **Interval mean (`temporal_semantics="interval_mean"`, and `"rate"` kept in rate units)** over observed valid support:

   $$\bar{\phi}_j = \frac{\sum_i v_i O_{ij} \phi_i}{\sum_i v_i O_{ij}}.$$

2. **Interval accumulation (`temporal_semantics="interval_accumulation"`)** under piecewise-constant within-source allocation:

   $$A_j = \sum_i v_i \frac{O_{ij}}{\delta_i} a_i.$$

3. **Rate integrated to amount (`temporal_semantics="rate"`, `AggregationMethod.SUM`)** with time-unit conversion factor $s_\tau$ from coordinate time units to `integration_time_unit`:

   $$A_j = s_\tau \sum_i v_i O_{ij} r_i.$$

4. **State at boundary (`temporal_semantics="state_at_boundary"`)**: selects $\phi(T_j) = \phi(t_i)$ at an exact timestamp match $T_j = t_i$ with identical units and `conservation_level="none"`.

5. **Descriptive statistics** (require `interval_mean`, identical units, and `conservation_level="none"`): duration-weighted population variance $\sigma_j^2 = \sum_i v_i O_{ij}(\phi_i - \bar{\phi}_j)^2 / \sum_i v_i O_{ij}$ (in `variance_units`), extrema $\min/\max\{\phi_i : v_i O_{ij} > 0\}$, range, and lower duration-weighted median (`AggregationMethod.PERCENTILE` raises `ValueError` because CF-1.13 defines no general percentile `cell_methods` token).

### 2.3 Current-Step Zero-Order Hold and Uniform Disaggregation

In `TemporalResampleAdapter` (`mode="current_step"`), a single fixed step $[t_n, t_n + \Delta t)$ reconstructs an instantaneous flux as $\widetilde{F}(\tau) = F(t_n)$, yielding exact one-step interval mean:

$$
\overline{F}_n = \frac{1}{\Delta t} \int_{t_n}^{t_n + \Delta t} \widetilde{F}(\tau)\,d\tau = F(t_n),
$$

valid strictly when $\Delta t > 0$ equals the declared window duration, $t_n$ is the left endpoint, no adaptive or hidden substep support change occurs, and ports transition from `instantaneous` to `interval_mean` on identical shapes.

In `TemporalDisaggregator` (`TemporalConfig.direction == "downscale"`), only `DisaggregationMethod.UNIFORM` is executable (`LINEAR`, `SHAPE_PRESERVING_SPLINE`, `CUBIC_SPLINE`, `CLIMATOLOGY_DIURNAL`, `EXTERNAL_DRIVER`, and `STOCHASTIC` fail closed with `ValueError`). Exchanging coarse index $j$ (duration $\Delta_j$) and fine index $i$ with overlap $O_{ji}$:

$$
\phi_i = \frac{\sum_j v_j O_{ji} \Phi_j}{\sum_j v_j O_{ji}} \quad (\text{mean / rate}), \qquad
a_i = \sum_j v_j \frac{O_{ji}}{\Delta_j} A_j \quad (\text{accumulation}),
$$

while `state_at_boundary` requires an explicit `state_boundary_policy` (`"exact"`, `"left"`, or `"right"`) without extrapolation.

### 2.4 Calendar Admission and Absolute Month Ordinal

`PeriodIndexer.elapsed_month_index(t, calendar, time_origin=None)` and `RELICSimulator._compute_month_index(t)` evaluate the monotone integer month ordinal from year $1$ for elapsed seconds $t \ge 0$ and reference origin $o$ (`0001-01-01` when omitted; `o_utc` after timezone offset conversion in `parse_time_origin` for the simulator):

$$
m_C(t; o) = 12\,\left[y_C(o + t) - 1\right] + \left[q_C(o + t) - 1\right].
$$

Calendar support differs across components by contract:

| Component | Admitted CF Calendars | Notes |
| --- | --- | --- |
| `CFTimeProcessor` | `gregorian`, `proleptic_gregorian`, `noleap`, `all_leap`, `360_day` | Bounds helper `create_time_bounds` (never auto-called by mappers) |
| `TemporalAggregator` | `noleap`, `360_day`, `gregorian`, `proleptic_gregorian` | Explicit interval bounds required; calendar-named frequencies rejected |
| `TemporalDisaggregator` | `noleap`, `360_day` | Separate strict two-calendar admission |
| `configure_calendar_settings` (`RELICSimulator`) | `noleap` (`365_day`), `360_day`, `standard` (`gregorian`), `proleptic_gregorian` | Rejects `all_leap` (`366_day`), `julian`, and unknown names with `Unsupported calendar_type` |

### 2.5 Standalone Quality Control Equations

In `relic/coupling/scales/temporal/quality_control.py` (used by `StreamProcessor`, never invoked inside `TemporalAggregator` or `TemporalDisaggregator`), `GapFiller` fills runs of length $\le g_{\max}$ via `"linear"` ($\phi_{\mathrm{gap}}(t) = \phi_a + (\phi_b - \phi_a)(t - t_a)/(t_b - t_a)$), `"nearest"`, `"climatological"` (series valid mean $\lvert V\rvert^{-1}\sum_{k \in V}\phi_k$), or `"none"` (`"seasonal"` and `"spline"` raise `ValueError`). `OutlierDetector` (for series with $\ge N_{\min}$ finite samples and spread above rounding) flags points under `"zscore"` ($\lvert(\phi_i - \bar{\phi})/\sigma_\phi\rvert > \theta_z$), `"modified_zscore"` ($\lvert 0.6745(\phi_i - \tilde{\phi})/\mathrm{MAD}\rvert > \theta_z$), or `"iqr"` ($\phi_i < Q_1 - k_{\mathrm{iqr}}\,\mathrm{IQR}$ or $\phi_i > Q_3 + k_{\mathrm{iqr}}\,\mathrm{IQR}$); `"isolation_forest"` raises `ValueError`.

## 3. Variable Table

| Symbol / Key | Meaning | Units | Owner | Role |
| --- | --- | --- | --- | --- |
| $I_i = [t_i^-, t_i^+)$, $J_j = [T_j^-, T_j^+)$ | Half-open source and target interval bounds | `time_units` | `source_bounds`, `target_bounds` | Required mapping geometry |
| $O_{ij}$, $w_{ij}$ | Overlap duration and target-fraction weight | `time_units`, $1$ | `temporal/core.py` | Kernel weights |
| $C_j^{\mathrm{geom}}$, $C_j^{\mathrm{valid}}$ | Geometric and valid-sample target coverage | $1$ | `temporal/core.py` | Support diagnostics |
| $\bar{\phi}_j$, $A_j$, $\phi_i$, $a_i$ | Mapped coarse/fine means, rates, or accumulations | `target_units` | `aggregator.py`, `disaggregator.py` | Primary mapped tensor |
| $m_C(t; o)$ | Zero-based absolute civil month ordinal from year 1 | $1$ (integer) | `PeriodIndexer`, `RELICSimulator` | Slow-group boundary index |
| $\epsilon_{\mathrm{rel}}$ | Relative mapping conservation residual on accepted targets | $1$ | `temporal/core.py` | Mapping audit diagnostic |

## 4. Parameter Table

| Key | Units | Default (Bounds) | Source YAML | Consumer |
| --- | --- | --- | --- | --- |
| `coupling.scales.temporal.coverage_threshold` | `dimensionless` | `0.8` ($[0.5, 1.0]$) | `coupling/scales_parameters.yaml` | `TemporalAggregator`, `QualityConfig`, CF metadata |
| `coupling.scales.temporal.disaggregation.allow_negative` | `dimensionless` (bool) | `False` | `coupling/scales_parameters.yaml` | `DisaggregationConfig` |
| `coupling.scales.temporal.disaggregation.use_physical_constraints` | `dimensionless` (bool) | `True` | `coupling/scales_parameters.yaml` | `DisaggregationConfig` |
| `coupling.scales.temporal.disaggregation.time_dim` | `dimensionless` (int) | `0` ($[0, 3]$) | `coupling/scales_parameters.yaml` | `DisaggregationConfig` |
| `coupling.scales.conservative.clamp_min`, `clamp_max` | `dimensionless` | `0.0`, `1.0e10` | `coupling/scales_parameters.yaml` | `TemporalDisaggregator` lower-bound refusal check |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1.0e-6` ($[10^{-10}, 10^{-3}]$) | `validation/conservation.yaml` | Mapping residual check in all 3 temporal operators |
| `coupling.scales.temporal.quality_control.max_gap_length` | `dimensionless` (samples) | `7` ($[0, 1000]$) | `coupling/scales_parameters.yaml` | `QualityController` / `GapFiller` |
| `coupling.scales.temporal.quality_control.gap_filling_method` | `dimensionless` (str) | `"linear"` | `coupling/scales_parameters.yaml` | `GapFiller` |
| `coupling.scales.temporal.quality_control.outlier_method`, `outlier_threshold`, `outlier_min_valid_samples`, `iqr_fence_factor` | `dimensionless` | `"zscore"`, `3.0`, `10`, `1.5` | `coupling/scales_parameters.yaml` | `OutlierDetector` |
| `coupling.scales.temporal.quality_control.grade_excellent_min_fraction`, `grade_excellent_max_gap_length`, `grade_good_min_fraction`, `grade_acceptable_min_fraction`, `grade_poor_min_fraction` | `dimensionless` | `0.95`, `1`, `0.8`, `0.6`, `0.4` | `coupling/scales_parameters.yaml` | `CoverageEvaluator` / `QualityController` |
| `coupling.scales.temporal.calendars.*` | calendar schema | 5 CF calendars | `coupling/temporal_calendars.yaml` | `CFTimeProcessor` / `CalendarHandler` |

## 5. Numerical Methods and Discretization

1. **Fail-closed contract validation**: Both `TemporalAggregator` and `TemporalDisaggregator` reject `point_sample` and `categorical` inputs (`SpatialQuantityBasis.parse` rejects bare `"extensive"`). Interval laws require `measure_units` (plus `integration_time_unit` for `"interval_mean"` and `"rate"`; `"interval_accumulation"` rejects `integration_time_unit`). In `TemporalDisaggregator`, when `use_physical_constraints=True` and `allow_negative=False`, finite inputs below `clamp_min` raise `ValueError` without silent clamping.
2. **Bounded CF-1.13 metadata synchronization**: `CFAttributeSynchronizer.sync_after_temporal_aggregation` updates `cell_methods` in place (`"time: mean"`, `"time: sum"`, `"time: minimum"`, `"time: maximum"`, `"time: variance"`, or `"time: point"`), appends operation `history`, attaches `coverage_fraction`, `coverage_threshold`, and `quality_state`, and emits `time_coordinate` attributes (`standard_name="time"`, `axis="T"`, `units`, `calendar`, and `bounds`).

## 6. Conservation, Boundary Conditions, and Ledgers

At `conservation_level` `"moderate"` (default) or `"strict"` (raises on violation), `"interval_mean"`, `"rate"`, and `"interval_accumulation"` evaluate the relative mapping residual against a required positive `ParameterManager` scale $Q_{\mathrm{ref}}$ (`q_ref_parameter`, in `measure_units`):

$$
\epsilon_{\mathrm{rel}} = \frac{\lvert M_{\mathrm{target}} - M_{\mathrm{source}} \rvert}{\max\!\left(\sum \lvert q_{\mathrm{source}} \rvert,\; Q_{\mathrm{ref}}\right)} \le \varepsilon_{\mathrm{cons}},
$$

under law identifiers `temporal_observed_integral_consistency`, `temporal_rate_amount_consistency`, `temporal_rate_integral_consistency`, or `temporal_interval_accumulation_partition` ($\varepsilon_{\mathrm{cons}} = \text{validation.conservation.conservation\_tolerance}$). Diagnostics always record `physical_ledger_checked = False` because $\epsilon_{\mathrm{rel}}$ audits the mathematical projection on accepted support, not a closed multi-pool physical budget. In `TemporalDisaggregator`, `"interval_accumulation"` at `"moderate"` or `"strict"` requires complete valid coverage ($C_i^{\mathrm{valid}} = 1$) on every accepted target and raises `ValidationError` on partial gaps. Non-time series with zero accepted targets remain `NaN` with `checked = False` (`mapping_excluded_output_mask`), and if no series has accepted targets, `conservation_skip_reason = "no_accepted_target_support"` is recorded without raising.

## 7. Code Mapping

| Module / Class | File Path | Architectural Role |
| --- | --- | --- |
| `build_overlap_plan`, `apply_temporal_kernel` | `relic/coupling/scales/temporal/core.py` | Half-open overlap plan, unit checks, and mapping residual evaluation |
| `TemporalAggregator` | `relic/coupling/scales/temporal/aggregator.py` | Fine-to-coarse `ScaleCouplingFunctor` (`direction="upscale"`) |
| `TemporalDisaggregator` | `relic/coupling/scales/temporal/disaggregator.py` | Coarse-to-fine uniform `ScaleCouplingFunctor` (`direction="downscale"`) |
| `TemporalScaling` | `relic/coupling/scales/temporal/temporal_scaling.py` | Fixed-SI-duration resampling `ScaleCouplingFunctor` |
| `CFTimeProcessor`, `CalendarHandler`, `PeriodIndexer` | `relic/coupling/scales/temporal/calendar.py` | CF calendar bounds construction and `elapsed_month_index` |
| `create_temporal_rescaling_metadata` | `relic/coupling/scales/temporal/cf_functions.py` | Bounded CF time-coordinate and coverage metadata builder |
| `QualityController`, `CoverageEvaluator`, `GapFiller`, `OutlierDetector` | `relic/coupling/scales/temporal/quality_control.py` | Standalone pre-mapping QC and gap-filling helpers |
| `TemporalResampleAdapter` | `relic/orchestration/adapters.py` | Graph-edge `current_step` zero-order hold adapter |

## 8. Implementation Status

- **Implemented (`[I]`)**: Exact half-open overlap weighting, 4 `temporal_semantics` modes, duration-weighted descriptive statistics, `current_step` zero-order hold, uniform conservative disaggregation, 4-calendar aggregator / 2-calendar disaggregator / 4-family simulator origin-aware month ordinal arithmetic, $Q_{\mathrm{ref}}$-normalized mapping conservation diagnostics, standalone `QualityController` grading/gap-filling/outlier detection, and bounded CF time-coordinate attribute synchronization.
- **Not implemented (`[N]`)**: Non-uniform disaggregation (`LINEAR`, `SHAPE_PRESERVING_SPLINE`, `CUBIC_SPLINE`, `CLIMATOLOGY_DIURNAL`, `EXTERNAL_DRIVER`, `STOCHASTIC`), general percentile aggregation (`PERCENTILE`), `"seasonal"`/`"spline"` gap filling, `"isolation_forest"` outlier detection, propagation of measurement uncertainty variances $\sigma_\phi^2$ inside mapping kernels, and dataset-wide NetCDF CF-conformance certification.
- **IPHS Equivalence Certificates**: None (`0` certificates reference `time_scaling.md`).

## 9. References

1. Astrom, K. J., and Murray, R. M. (2008). *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton University Press.
2. Eaton, B., Gregory, J., Drach, B., Taylor, K., Hankin, S., et al. (2024). *NetCDF Climate and Forecast (CF) Metadata Conventions, Version 1.13* (Sections 4.4, 7.1, 7.3, and Appendix E).
3. Iglewicz, B., and Hoaglin, D. C. (1993). *How to Detect and Handle Outliers* (ASQC Basic References in Quality Control, Vol. 16). American Society for Quality Control.
4. World Meteorological Organization (2018). *Guide to Climatological Practices* (WMO-No. 100, 2018 ed.). WMO.
