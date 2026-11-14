# Coupling Workflows

RELIC coupling is implemented through `ScaleCouplingFunctor` and the existing
state, parameter, port, and orchestration contracts. Internal domain exchange,
temporal scaling, spatial regridding, external model adapters, and plugin
coupling use separate interfaces but converge on the same runtime lifecycle.

## Built-in internal coupling

```python
from relic.coupling import (
    create_coupling_functor,
    get_available_couplings,
)
from relic.coupling.internal import get_default_configs

available = get_available_couplings(include_plugin=False)
defaults = get_default_configs()

coupler = create_coupling_functor(
    coupling_type="biosphere_atmosphere",
    name="bio_atm",
    config=defaults["biosphere_atmosphere"],
)
```

| Coupling type | Purpose |
| --- | --- |
| `biosphere_atmosphere` | Exchange declared biosphere and atmosphere variables through the built-in coupling contract. |
| `soil_vegetation` | Evaluate Feddes root-water uptake from a resolved soil profile and expose `root_uptake_profile` (kg m-3 s-1) and `effective_transpiration` (kg m-2 s-1) as replace-mode rate ports. It does not advance or deduct soil moisture; a soil-water budget closes only when a registered soil-water Functor consumes the port exactly once, and no current soil-water Functor does. |

These are the built-in types returned when plugin coupling is excluded.
Unknown types raise `ValueError`. Default configurations are read from the
parameter database; missing required entries raise `KeyError` or
`RuntimeError`.

```text
create_coupling_functor(
    coupling_type: str,
    name: str,
    config: Optional[dict[str, Any]],
    parameter_manager=None,
) -> FunctorBase
```

`name` becomes the registered Functor identity. `config` contains the
coupling-specific contract and parameter references. `parameter_manager`
allows explicit manager injection; otherwise the canonical global manager is
used by the coupling factory.

## Temporal scaling

### `TemporalConfig`

```text
TemporalConfig(
    source_frequency: str,
    target_frequency: str,
    direction: ScalingDirection = ScalingDirection.UPSCALE,
    spatial_basis: SpatialQuantityBasis = SpatialQuantityBasis.INTENSIVE,
    temporal_semantics: Optional[TemporalSemantics] = None,
    conservation_level: ConservationLevel = ConservationLevel.MODERATE,
    conservation_level_overrides: Mapping[str, ConservationLevel | str] = {},
    device: Optional[torch.device] = None,
)
```

| Parameter | Meaning |
| --- | --- |
| `source_frequency` | Frequency token of the source series: `sub-hourly`, `hourly`, `daily`, `weekly`, `monthly`, `seasonal`, or `annual`, optionally with an integer multiplier such as `3hourly`. Tokens are matched without regard to case; `yearly` is accepted for `annual`, and `subhourly`, `sub_hourly`, `minute`, `minutes`, `min`, `mins`, `minutely`, `second`, `seconds`, `sec`, `secs`, and `secondly` for `sub-hourly`. |
| `target_frequency` | Frequency token of the target series. |
| `direction` | `UPSCALE` for fine-to-coarse or `DOWNSCALE` for coarse-to-fine. `TemporalAggregator` requires `UPSCALE`. `TemporalDisaggregator.disaggregate()`, `downscale()`, `to_hourly()`, `to_daily()`, and `forward()` require `DOWNSCALE`, and `TemporalDisaggregator.upscale()` requires `UPSCALE`. A mismatch raises `ValueError`. |
| `spatial_basis` | Spatial representation of the mapped field: a `SpatialQuantityBasis` (`intensive`, `cell_total`, `area_density`, or `categorical`) or its string value. It never selects the temporal law; the temporal operators use it only to reject `categorical` fields, which raise in aggregation and disaggregation. The string `extensive` raises `ValueError` because it does not say whether a value is a cell total or an area density. |
| `temporal_semantics` | Explicit `interval_mean`, `rate`, `interval_accumulation`, or `state_at_boundary`; a call-level value takes precedence. Missing semantics are rejected without inference, and `point_sample` and `categorical` are rejected. |
| `conservation_level` | `none`, `relaxed`, `moderate`, or `strict`; see "Coverage, residuals, and errors" below. |
| `conservation_level_overrides` | Immutable normalized overrides keyed by exact Functor identity. |
| `device` | Optional initial tensor device; simulator registration owns final migration. |

The constructor normalizes the enum fields, parses `spatial_basis` with
`SpatialQuantityBasis.parse`, and freezes the normalized per-Functor
conservation overrides. It validates the frequency tokens but stores them as
given: `TemporalConfig("hourly", "yearly").target_frequency` is `"yearly"`. An
unknown frequency token raises `ValueError`. The valid-coverage threshold is
not a configuration field. Its only owner is the parameter
`coupling.scales.temporal.coverage_threshold` (declared bounds [0.5, 1.0];
current configuration: 0.8), which `TemporalAggregator` reads through
ParameterManager when it is constructed; the aggregator also rejects a value
outside `(0, 1]`. The resolved value is the read-only property
`coverage_threshold`: assigning it raises `AttributeError`, so a different
threshold for one aggregator is set on that parameter (the global
ParameterManager or an injected `parameter_manager`) before construction,
where it is validated. The configuration does not accept `cf_compliant` or
establish complete dataset-level Climate and Forecast (CF) conformance.
Runtime interval bounds, calendar, units, temporal semantics, and coverage are
checked by the temporal operation.

`TemporalConfig.from_resolved_settings(source_frequency=..., target_frequency=..., settings=..., direction=..., spatial_basis=..., device=...)`
builds this configuration from one already resolved settings mapping.
`settings` may contain only `conservation_level`,
`conservation_level_overrides`, and `temporal_semantics`; any other key, such
as `coverage_threshold`, raises `ValueError`. The displayed override mapping is
created by a dataclass factory, not shared across instances.

### `AggregationConfig`

```text
AggregationConfig(
    method: AggregationMethod = AggregationMethod.TIME_WEIGHTED_MEAN,
    time_dim: int = 0,
)
```

`method` selects the statistic: `MEAN`, `TIME_WEIGHTED_MEAN`, `SUM`, `MINIMUM`,
`MAXIMUM`, `MEDIAN`, `VARIANCE`, `STANDARD_DEVIATION`, or `RANGE`.
`PERCENTILE` exists in the enumeration but raises `ValueError`, because CF
defines no general percentile cell-method token. `time_dim` selects the time
axis of the data; a negative value counts from the last axis. The
configuration has no other fields, and any other keyword raises `TypeError`.
Interval labels and closure are not configurable: every interval is the
explicit half-open bound pair `[start, end)` passed to `aggregate()`, and the
target coordinate is the caller's `target_time`.

### `TemporalAggregator`

```text
TemporalAggregator(
    config: TemporalConfig,
    aggregation_config: Optional[AggregationConfig] = None,
    parameter_manager=None,
    metadata: Optional[FunctorMetadata] = None,
)
```

```python
import torch
from relic.coupling.scales.temporal import TemporalAggregator, TemporalConfig

data = torch.tensor([280.0, 282.0, 290.0], dtype=torch.float64)
source_time = torch.tensor([0.5, 1.5, 3.0], dtype=torch.float64)
target_time = torch.tensor([2.0], dtype=torch.float64)
source_bounds = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
target_bounds = torch.tensor([[0.0, 4.0]], dtype=torch.float64)

aggregator = TemporalAggregator(
    TemporalConfig(
        source_frequency="hourly",
        target_frequency="daily",
        conservation_level="none",  # every other level (default "moderate") requires q_ref_parameter
    ),
)
result = aggregator.aggregate(
    data,                      # [n_source, ...]; NaN marks missing values
    source_time, target_time,  # 1-D, in time_units
    variable_name="air_temperature",
    standard_name="air_temperature",
    units="K",
    temporal_semantics="interval_mean",
    target_units="K",
    measure_units="K h",
    integration_time_unit="hour",
    calendar="noleap",
    time_units="hours since 2000-01-01 00:00:00",
    source_bounds=source_bounds,  # [n_source, 2], half-open intervals
    target_bounds=target_bounds,  # [n_target, 2]
)
print(float(result.data[0]))
```

With `data = [280, 282, 290]` K on the source intervals `[0, 1)`, `[1, 2)`, and
`[2, 4)` h and one target interval `[0, 4)` h, the result is the
duration-weighted mean 285.5 K. Missing `temporal_semantics`, `target_units`,
calendar, or `time_units` raise `ValueError`. Every method also needs both
bounds, and the mean and sum methods need `measure_units`; a missing one raises
`ValueError`. `state_at_boundary` returns before these checks and needs
neither bounds nor `measure_units`. Empty source `units` raise the temporal
`ValidationError` described below for the mean and sum methods, which build the
unit contract, and `ValueError` for the descriptive statistics and
`state_at_boundary`, whose source and target units must be identical. Bounds
are never inferred. The calendar must be `noleap` or `360_day`; any other
calendar, including `standard`, raises `ValueError`. Interval means and rates
also need `integration_time_unit`, and an `interval_accumulation` must not
declare one. The aggregator maps fixed intervals only: a `monthly`, `seasonal`,
or `annual` source or target frequency raises `ValueError` in `aggregate()`,
because calendar periods belong to the calendar owners. At every conservation
level other than `none`, including the default `moderate`, `q_ref_parameter`
must also name a ParameterManager key in `measure_units`.

`TemporalAggregator.forward` and `TemporalDisaggregator.forward` require a
non-empty `variable_name` (the key of the returned update; a `standard_name`
never stands in for it) and non-empty `units`. `aggregate`, `disaggregate`, and
`upscale` require `units`; `variable_name`, `standard_name`, and `cell_methods`
are optional there: absent means an unnamed field, no CF identity, and no
earlier cell method, the absent names are omitted from the result metadata
(never published as an empty string), and an empty string raises `ValueError`.
The disaggregator no longer keys an unnamed output as `disaggregated`.

`aggregate()` returns `AggregationResult` with:

| Field | Meaning |
| --- | --- |
| `data` | Aggregated tensor. |
| `metadata` | Units, calendar, time units, `cell_methods` (`time: <method>`), a history entry, and the coverage threshold used. |
| `quality_flags` | Valid fraction, sample count, missing fraction, and insufficient-coverage mask. |
| `diagnostics` | Method, semantics, coverage and gap fractions, observed measure, and conservation diagnostics. |

The method defaults to `TIME_WEIGHTED_MEAN`; the mapping law is fixed by the
explicit `temporal_semantics` together with the method, never by names, units,
or roles. Let $O_{ij}$ be the overlap duration of source interval $I_i$ with
target interval $J_j$, and $m_i=1$ for a finite source value and $m_i=0$ for
NaN (Inf raises). Values keep their sign; overlap durations are non-negative.

- `MEAN` and `TIME_WEIGHTED_MEAN` run the same kernel. They accept
  `interval_mean` and `rate` and return the mean over the observed support,
  $y_j=\sum_im_iO_{ij}x_i/\sum_im_iO_{ij}$.
- `SUM` integrates a `rate` into an interval accumulation,
  $Y_j=\sum_im_iO_{ij}r_i$ with $O_{ij}$ in `integration_time_unit`, converted
  from `measure_units` to `target_units`; it partitions an
  `interval_accumulation` by duration, $Y_j=\sum_im_i(O_{ij}/\lvert I_i\rvert)a_i$.
- `MINIMUM`, `MAXIMUM`, `MEDIAN` (the lower duration-weighted median),
  `VARIANCE` (the duration-weighted population variance),
  `STANDARD_DEVIATION`, and `RANGE` require `interval_mean` and identical
  source and target units, and use the finite samples with positive overlap.
  `VARIANCE` also needs `variance_units`, the square of `target_units` on the
  same scale. `STANDARD_DEVIATION` and `RANGE` accept `units_metadata` only
  when it is absent or `temperature: difference`.
- `state_at_boundary` maps exact boundary instants and does no arithmetic;
  source and target units must be identical.

### Coverage, residuals, and errors

The valid coverage of target interval $J_j$ is
$c_j=\sum_im_iO_{ij}/\lvert J_j\rvert$ (dimensionless). Missing data are never
scaled up. An output whose $c_j$ is below the coverage threshold $\theta$
(`coupling.scales.temporal.coverage_threshold`; current configuration: 0.8) is
NaN and set in `insufficient_coverage_mask`. `MINIMUM`, `MAXIMUM`, and `RANGE`
keep their value whenever $c_j>0$; the mask still records the insufficient
coverage. An output at or above the threshold covers only the observed part: a
rate of 1e-4 kg m-2 s-1 summed over ten hours with one hour missing gives
3.24 kg m-2 with $c_j=0.9$, not 3.6 kg m-2.

The conservation level controls the mapping residual
$R=\sum_jq^{\mathrm{tgt}}_j-\sum_{i,j}q^{\mathrm{src}}_{ij}$, where the
contributions $q$ are measures in `measure_units` on the accepted targets; a
positive $R$ means that the target measure exceeds the source measure. The
relative residual is $\lvert R\rvert/\max(a,q_{\mathrm{ref}})$
(dimensionless), with the source activity
$a=\sum_{i,j}\lvert q^{\mathrm{src}}_{ij}\rvert$ and $q_{\mathrm{ref}}$ the
positive parameter named by `q_ref_parameter`. `none` computes no residual;
`relaxed` computes it without a tolerance comparison; `moderate` compares the
largest relative residual with `validation.conservation.conservation_tolerance`
(the level selects no tolerance) and reports the outcome in `conservation_passed` and `conservation_violation`;
`strict` raises `ValueError` on a violation. The residual is reported and never
used to correct values. `state_at_boundary` and the descriptive statistics
(`MINIMUM` to `RANGE`) have no residual law, so every level other than `none`
raises `ValueError` for them.

An accepted target has a finite value and a valid coverage at or above the
threshold; an accumulation target must also be completely covered. A target
that is not accepted is left out of the residual even when its value is finite.
An output element (one series along the non-time axes) without any accepted
target keeps its output and is excluded explicitly. That output is NaN when no
target reached the threshold, but a `SUM` target at or above the threshold that
is not completely covered keeps its finite partial accumulation. An excluded
element's residual is NaN, it is not checked, and it is flagged in
`mapping_excluded_output_mask` and counted in `mapping_excluded_output_count`,
while `mapping_excluded_target_count` counts all rejected targets. When no
element can be checked, the diagnostics report `conservation_checked=False`,
`conservation_passed=None`, and
`conservation_skip_reason="no_accepted_target_support"`, and `strict` does not
raise. The 3.24 kg m-2 example above is such a case: at `relaxed`, `moderate`,
and `strict` it returns 3.24 kg m-2 with `mapping_excluded_output_count=1` and
this skip reason, so the partial accumulation is never conservation-checked.

`aggregate()` does not wrap errors. Argument, metadata, and domain errors raise
`ValueError`, and an argument of the wrong type raises `TypeError`. An invalid
CF time-units string or unit contract raises the temporal core's
`ValidationError`, exported by `relic.coupling.scales.temporal`.
`RuntimeError` is raised only by internal consistency checks. In a governed
Functor call, `FunctorBase` re-raises a `TypeError`, `ValueError`, `KeyError`,
`AttributeError`, `RuntimeError`, `ImportError`, or `OSError` as
`ExecutionError` with the original error as its cause. The temporal
`ValidationError` derives directly from `Exception`, so it is not wrapped and
reaches the caller unchanged.

### `TemporalDisaggregator`

```text
TemporalDisaggregator(
    config: TemporalConfig,
    disaggregation_config: Optional[DisaggregationConfig] = None,
)
```

```text
DisaggregationConfig(
    method: DisaggregationMethod = DisaggregationMethod.UNIFORM,
    allow_negative: Optional[bool] = None,
    use_physical_constraints: Optional[bool] = None,
    time_dim: Optional[int] = None,
)
```

Fields left as `None` are read from
`coupling.scales.temporal.disaggregation.allow_negative`,
`.use_physical_constraints`, and `.time_dim` (current configuration: `false`,
`true`, and `0`). Only `UNIFORM`, the piecewise-constant reconstruction, runs;
`LINEAR`, the spline methods, the profile methods (`CLIMATOLOGY_DIURNAL`,
`EXTERNAL_DRIVER`), and `STOCHASTIC` raise `ValueError`. With
`use_physical_constraints` true and `allow_negative` false, a finite input
below `coupling.scales.conservative.clamp_min` (current configuration: 0)
raises without clamping. There are no profile, smoothing,
external-weight, stochastic-noise, or tolerance fields; the mapping tolerance
is `validation.conservation.conservation_tolerance`.

```python
import torch
from relic.coupling.scales.temporal import TemporalConfig, TemporalDisaggregator

data = torch.tensor([3.0e-4], dtype=torch.float64)
coarse_time = torch.tensor([1.5], dtype=torch.float64)
fine_time = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
coarse_bounds = torch.tensor([[0.0, 3.0]], dtype=torch.float64)
fine_bounds = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=torch.float64)

disaggregator = TemporalDisaggregator(
    TemporalConfig(
        source_frequency="3hourly",
        target_frequency="hourly",
        direction="downscale",
        conservation_level="none",  # every other level (default "moderate") requires q_ref_parameter
    ),
)
result = disaggregator.disaggregate(
    data,
    coarse_time,
    fine_time,
    variable_name="precipitation_flux",
    standard_name="precipitation_flux",
    units="kg m-2 s-1",
    cell_methods="time: mean",
    temporal_semantics="rate",
    target_units="kg m-2 s-1",
    measure_units="kg m-2",
    integration_time_unit="s",
    calendar="noleap",
    time_units="hours since 2000-01-01 00:00:00",
    source_bounds=coarse_bounds,  # [n_coarse, 2]
    target_bounds=fine_bounds,    # [n_fine, 2]
)
print(result.data.tolist())
```

`disaggregate()`, `downscale()`, and `forward()` take the bounds as
`coarse_bounds` and `fine_bounds` or as `source_bounds` and `target_bounds`
(`upscale()` as `fine_bounds` and `coarse_bounds` or the same two canonical
names), one name per bound; passing both names for one bound raises
`ValueError`, and error messages name the keys the caller used. `to_hourly()`
and `to_daily()` require the keywords `coarse_time`, `fine_time`,
`coarse_bounds`, and `fine_bounds`; `source_bounds` and `target_bounds` do not
replace them there. With coarse interval $I_i$, fine interval $J_j$, and
overlap $O_{ij}$, the fine values are
$x_j=\sum_im_iO_{ij}y_i/\sum_im_iO_{ij}$ for interval means and
rates and $x_j=\sum_im_i(O_{ij}/\lvert I_i\rvert)a_i$ for an
`interval_accumulation`. `DisaggregationResult` contains `data`, `metadata`,
`conservation_error`, and `diagnostics` (reconstruction, bounds relation,
geometric and valid coverage, gap fractions, observed measure, and the
mapping-residual diagnostics).

The disaggregator applies no coverage threshold. A fine interval without a
finite coarse value is NaN for interval means and rates, but receives 0 for an
`interval_accumulation`; its `diagnostics["valid_coverage"]` is 0 in both
cases, so check the coverage before using accumulated amounts. The
conservation levels act as in aggregation, with the residual taken over the
fine intervals that have a finite value and a positive valid coverage. Unlike
aggregation, this includes partially covered accumulation intervals: when an
`interval_accumulation` fine interval has a valid coverage between 0 and 1
(it extends past the coarse bounds or overlaps a coarse interval whose value is
NaN), every level other than `none` raises the temporal `ValidationError`,
because the accumulation residual requires complete coverage. With the level
`none` the call returns the partial amount, for example 1 kg m-2 with valid
coverage 0.5 for a fine interval [2, 4) h over a coarse interval [0, 3) h that
holds 3 kg m-2. `conservation_error` is the largest relative residual over the
checked series: NaN when a residual was computed but no series could be checked, and 0.0 when
the level is `none`. Values are never corrected. The call needs the same
explicit fields as aggregation, including a `noleap` or `360_day` calendar and,
at a level other than `none`, `q_ref_parameter`; `state_at_boundary`
also requires `state_boundary_policy` (`exact`, `left`, or `right`) and
the conservation level `none`.

## Spatial regridding

### `SpatialRegridderFactory`

```text
SpatialRegridderFactory(
    enforce_conservation: bool = False,
    conservation_tolerance: Optional[float] = None,
    device: Optional[torch.device] = None,
    parameter_manager=None,
)
```

| Parameter | Meaning |
| --- | --- |
| `enforce_conservation` | Require the conservative method for every field that is not categorical: a `cell_total`, `area_density`, or `intensive` field, or a field without a declared `spatial_basis`, raises `ValueError` with any other method. The conservative regridders the factory builds then raise `RuntimeError` in `regrid()` when the per-row covered-support ledger is rejected by `covered_support_acceptance` (the relative tolerance and floating-point rounding allowance). A variable's `require_conservation: true` metadata applies the same method rule to that variable. A field declared `categorical` is regridded only by `nearest` or `irregular_nearest`, with or without this flag; any other method raises `ValueError` for it. The weight check at construction does not depend on this flag. |
| `conservation_tolerance` | Positive relative error tolerance; resolved from parameters when omitted. |
| `device` | Working device for regridding tensors; resolved by the device policy when omitted. |
| `parameter_manager` | Canonical parameter owner or explicit injected manager. |

```python
import torch
from relic.coupling.scales.spatial import GridDescriptor, SpatialRegridderFactory

source_grid = GridDescriptor(
    grid_type="regular",
    coord_system="cartesian",
    dims=(1, 2),
    x_coords=torch.tensor([0.5, 1.5], dtype=torch.float64),
    y_coords=torch.tensor([0.5], dtype=torch.float64),
    x_bounds=torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
    y_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    units={"x": "m", "y": "m"},
)
target_grid = GridDescriptor(
    grid_type="regular",
    coord_system="cartesian",
    dims=(1, 1),
    x_coords=torch.tensor([0.75], dtype=torch.float64),
    y_coords=torch.tensor([0.5], dtype=torch.float64),
    x_bounds=torch.tensor([[0.0, 1.5]], dtype=torch.float64),
    y_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    units={"x": "m", "y": "m"},
)
factory = SpatialRegridderFactory(device=torch.device("cpu"))
regridder = factory.create_regridder(
    source_grid,
    target_grid,
    method="conservative",
    variable_metadata={
        "spatial_basis": "area_density",
        "units": "kg m-2",
    },
)
precipitation_source = torch.tensor([[4.0, 8.0]], dtype=torch.float64)
precipitation_target = regridder.regrid(precipitation_source, method="conservative")
print(round(float(precipitation_target.reshape(-1)[0]), 2))
```

`create_regridder()` accepts `conservative`, `bilinear`, `nearest`,
`irregular_bilinear`, and `irregular_nearest`. The method must be explicit,
either as `method` or as `variable_metadata["regridding_method"]`; otherwise
`ValueError` is raised (there is no bilinear default). Invalid methods raise
`ValueError`. The conservative method remaps on a declared quantity basis,
given as `quantity_basis=` or as `variable_metadata["spatial_basis"]`:
`intensive`, `cell_total`, or `area_density`. The string `extensive` raises
`ValueError`, a `categorical` field raises because it needs the
nearest-neighbour method, and `quantity_basis` passed for another method
raises. A conservative regridder built without a basis raises on every remap
that does not pass its own `quantity_basis`; no field is assumed to be an area
density. Grids must declare their cells: `x_bounds` and `y_bounds` from the
data file or an explicit descriptor, or, for an unstructured grid, its vertex
polygons and `cell_areas`. Bounds are never inferred from the coordinate
spacing, and missing bounds raise `ValueError`.

With the overlap fractions $W_{iJ}=A_{iJ}/A_i$ (not renormalized), the covered
fraction $f_i=\sum_JW_{iJ}$ of source cell $i$, and the covered target area
$C_J=\sum_iA_iW_{iJ}$, the regridder computes $Y_J=\sum_iX_iW_{iJ}$ for
`cell_total`, $y_J=\sum_ix_iA_iW_{iJ}/A_J$ for `area_density`, and
$y_J=\sum_ix_iA_iW_{iJ}/C_J$ for `intensive` fields; the output keeps the input
basis. $W$ and $f$ are dimensionless, and values keep their units and sign.
The geometry comes from the declared `coord_system` of the two grids, not from
the grid type. `latlon` grids, `regular` grids included, are spherical: the
overlap of two cells is the lat-lon cell they share, and every cell and
overlap area is $A=R^2(\lambda_2-\lambda_1)(\sin\phi_2-\sin\phi_1)$ in m2,
with longitudes $\lambda$ and latitudes $\phi$ in radians and $R$ the physical
constant `earth_radius`. A 1-degree cell between 60 N and 61 N therefore has
about 0.49 times the area of one between 0 and 1 N. Declared `cell_areas` of a
`latlon` grid must equal these spherical areas within the conservation
tolerance; otherwise `ValueError` is raised. `cartesian` and `projected` grids
are planar, with areas $(x_2-x_1)(y_2-y_1)$ in the squared coordinate units,
and their declared `cell_areas` are used as given. Nothing is reinterpreted as
planar: grids in different coordinate systems, a `latlon` grid with a latitude
bound outside [-90, 90], an unknown coordinate system, and `latlon` polygon
grids raise `ValueError`. The amount is conserved on the
covered support, $\sum_JY_J=\sum_if_iX_i$; the uncovered source amount
$\sum_iX_i(1-f_i)$ is reported as `uncovered_source_amount` in
`last_regrid_report` and is not rescaled into the target. There is no
whole-domain rescaling, so the map is
linear and autograd differentiates exactly this expression. For densities 4
and 8 kg m-2 on two 1 m2 source cells and one 1.5 m2 target cell that covers
the first cell and half of the second, the result is 5.33 kg m-2 (8 kg), 4 kg
are reported as uncovered, and the gradient with respect to the two source
densities is $[2/3, 1/3]$. `last_regrid_report` also records the basis, the
covered and mapped amounts, the covered source fraction by area, the absolute
and relative residual, and the range-clamp residual.

At construction the regridder checks its weights on the covered support,
whatever `enforce_conservation` is: every source covered fraction
$f_i=\sum_JW_{iJ}$ and every target covered fraction
$g_J=\sum_iA_iW_{iJ}/A_J$ is at most 1, and no weight is negative, within the
tolerance. A violation, caused by overlapping cells or by `cell_areas` that
disagree with the cell bounds, raises `ValueError`. Partial or missing
coverage is not an error: it is counted in `coverage_report` and
`verify_conservation()` (fully, partially, and not covered source and target
cells, and the minimum coverages) and logged, and each remap reports its
uncovered amount.

The range clamp is a per-call option: `regrid(..., preserve="none")`,
`upscale()` or `downscale()` with `range_clamp=True`, or a governed call with
`preserve="none"`. It clips the density representation to the range of the
source values and reports the change as `range_clip_residual` (requested
minus clamped amount, the StateManager clip-loss sign: negative when the clamp
adds material); nothing is rescaled after the clamp. With `preserve="mass"` (the default) or
`preserve="energy"`, the clamp stays off. The covered-support ledger is
accepted row by row by `covered_support_acceptance` through
`ledger_acceptance`; a rejected ledger raises `RuntimeError` under
`enforce_conservation` and logs a warning otherwise.

Regridder instances are cached per factory under a content fingerprint of
everything that builds them: the regridder class and method, every field of
both grid descriptors (grid type, coordinate system, dimensions, coordinates,
connectivity, vertices, bounds, cell areas, mask, units, and Coordinate Reference System (CRS)), every other
constructor argument by value, and the parameter manager by identity. A grid
pair with the same shapes but another geometry therefore gets its own
regridder, and an argument without a canonical content encoding raises
`TypeError`. After changing parameters of a live manager, call
`clear_cache()`. `ConservativeRemap` below applies the same basis rule to
explicit overlap areas.

### `AutoRegridder`

```text
AutoRegridder(
    source_grid: GridDescriptor,
    target_grid: GridDescriptor,
    enforce_conservation: bool = False,
    conservation_tolerance: Optional[float] = None,
    device: Optional[torch.device] = None,
    parameter_manager=None,
    strict_direction_inference: bool = True,
    validation_mode: Literal["strict"] = "strict",
)
```

The public operations are `forward()`, `upscale()`, and `downscale()`. There
is no public `regrid()` method. `forward()` requires `data`, and it requires
`variable_metadata` with an explicit `regridding_method` (and `spatial_basis`
for the conservative method) whenever either directional regridder has not
been created yet. `upscale()` and `downscale()` called before any metadata was
set raise `ValueError` (no method is assumed); `downscale()` maps a field on
the target grid to the source grid through the target-to-source regridder's
forward mapping, so the nearest-neighbour method maps in the right direction.
Metadata rebuilds existing
regridders only when it changes the method: a later call that keeps the method
but declares another `spatial_basis` still uses the first basis, so call
`clear_cache()` or use a new `AutoRegridder` before switching the basis.
`forward()` takes the direction from an explicit `operation` or from the
tensor shape against the two grids, and mixed or unresolvable directions raise
`ValueError`. Only strict validation mode is accepted. The governed call
validates the declared input port `source_grid_fields`, so the state passed to
it must contain that key with a numeric tensor; the fields themselves are read
from `data`.

For the regridders themselves, equal source and target cell counts decide no
direction: `forward` raises unless the call names `operation`; the
nearest-neighbour regridder takes the direction from the grid the data fit
and raises when they fit both or neither. `ScaleManager.regrid_variable` maps
the field from the source to the target grid with the regridder's `upscale`
for every method and reports the target `coverage` of the bilinear and
nearest-neighbour regridders.

### Overlap remap and coverage

`ConservativeRemap` remaps on explicit overlap areas $A_{iJ}$ (one area unit
for all areas; m2 in the example below) between source cell $i$ and target
cell $J$, with source areas $A_i$, target areas $A_J$, and the covered target
area $C_J=\sum_iA_{iJ}$. Every variable declares its basis, through
`variable_metadata` (`spatial_basis`) or the `coupling.scales.*.variables`
catalogs. `cell_total` values are per-cell
totals, $Y_J=\sum_iX_iA_{iJ}/A_i$; `area_density` values are amounts per unit
area, $y_J=\sum_ix_iA_{iJ}/A_J$, which needs `target_areas`; `intensive` values
are averaged over the covered target area, $y_J=\sum_ix_iA_{iJ}/C_J$, and every
target cell must receive some overlap. An undeclared variable, the string
`extensive`, and `categorical` fields raise `ValueError`. As an alternative to
`overlap_areas`, `source_grid` and `target_grid` descriptors with declared cell
bounds can be passed; the overlap then comes from `ConservativeRegridder`.

```python
import torch
from relic.coupling.scales.adapters import ConservativeRemap

remap = ConservativeRemap()
litter_carbon_target = remap.upscale(
    torch.tensor([4.0, 8.0], dtype=torch.float64),  # kg per source cell
    variables=["litter_carbon"],
    overlap_areas=torch.tensor([[1.0], [0.5]], dtype=torch.float64),  # m2, [n_source, n_target]
    source_areas=torch.tensor([1.0, 1.0], dtype=torch.float64),       # m2
    variable_metadata={"litter_carbon": {"spatial_basis": "cell_total"}},
)
# litter_carbon_target == [8.0] kg
# remap.last_scaling_audit.metadata["covered_source_fraction"] == 0.75
# remap.last_scaling_audit.metadata["uncovered_source_amount"] == {"litter_carbon": 4.0}
```

The source cells hold 4 kg and 8 kg with covered fractions 1 and 0.5, so the
target receives 8 kg and 4 kg stay outside the target support. Only the
covered part is remapped. The unmapped 4 kg are reported, not routed anywhere:
as `uncovered_source_amount` in the audit metadata, in an audit warning, and,
in a governed call, as the output `litter_carbon_uncovered_source_amount`. The
audit's relative error is 0 because it compares the target with the covered
source part. The value 0.75 is an area coverage, not a mass coverage: 8 of
12 kg are mapped. For an intensive field the audited quantity is the area
integral $\sum_Jy_JC_J$; it is a heat or energy budget only when the field is
itself that density, and a temperature budget needs mass and heat capacity.

`ConservativeRegridder` and `HierarchicalAggregator` take the same declared
basis (`cell_total` or `area_density`) and convert between the two with the
cell areas themselves; declare the representation directly; manual pre-conversion is
unnecessary. With `preserve_positivity=True`, negative outputs are clipped to zero
and the clip residual (in the ledger amount, negative for the material the
clip adds) is reported as
`diagnostics.conservative_remap.positivity_clip_residual`. When
`preserve_mass` or `preserve_energy` is set (both default to true), a relative
covered-support error above the tolerance raises `RuntimeError`. The example
calls the adapter method directly to show the arithmetic.

## External model adapters

External adapters read existing model output. Each adapter is built from its
configuration dataclass, for example `WRFAdapter(config=WRFConfig(root_path=...))`;
a `config` of another type raises `ValueError` (Functor dependencies are
declared with `add_dependency` or `set_dependencies`, not through the adapter
configuration). An adapter needs its configuration: `CESMAdapter`,
`CMIP6Adapter`, and `SWATAdapter` raise without a `CESMConfig`,
`CMIP6Config`, or `SWATConfig`; `VICAdapter` and `WRFAdapter` without a
configuration read `coupling.external.<model>.default_root_path` (a failed
lookup raises); no adapter uses the current directory. `WRFConfig.time_format`
defaults to `None`, which takes `coupling.external.wrf.time_format`; an
explicit format is kept. Multi-file VIC and CESM outputs are concatenated in
the order of their decoded record times (file names and modification times
never order records); overlapping or repeated records, and a file without a
decoded time coordinate, raise. The lifecycle is:

```text
discover -> open -> read -> transform_to_internal -> validate_data -> close
```

| Method | Purpose |
| --- | --- |
| `discover(variables=None)` | Scan `config.root_path` for the configured output files and report the files, time coverage, and available variables. A missing root path raises `FileNotFoundError`, and a root without matching files raises `ValueError`. See below for `variables`. |
| `open()` | Open the discovered files. Raises `ValueError` when no discovered files are recorded (`discover()` was not called, or `close()` cleared them). A `TypeError`, `ValueError`, `KeyError`, `AttributeError`, `RuntimeError`, `ImportError`, or `OSError` raised while opening or parsing is re-raised as `IOError` with the original message; an exception of any other type propagates unchanged. |
| `read(variables=None, ...)` | The one read path of every field: map internal names to external names, apply the field's declared source transform, and convert to the catalog units. Returns float32 tensors keyed by internal name (with a `_<domain>`, `_<stream>`, or `_<dataset>` suffix when several are read), each carrying its read metadata (dimensions, coordinates, attributes, CF horizontal coordinates). WRF, CESM, and SWAT also accept a `domain`, `stream`, or `dataset` selector. With `variables=None`, a mapped field that a dataset does not contain is left out of that dataset's result, listed in `last_read_report["absent_mapped_fields"]` as `{dataset: {internal_name: external_name}}`, and logged. With an explicit `variables` list, a requested variable whose mapped field is in none of the datasets read raises `ValueError`; a field that only some of them contain is left out of the others and recorded the same way. |
| `transform_to_internal(data, target_grid=None, variable_metadata=None)` | Regrid each field to `target_grid` on its declared basis when a target grid is given, apply the catalog quality control, and return float32 tensors. It renames nothing, converts no units, and does no temporal resampling; other keyword arguments raise `TypeError`. |
| `validate_data(data)` | Return a `ValidationReport` from the shared external-data validator without changing the data; this is the adapters' only validation entry point. |
| `close()` | Close the opened datasets and clear the discovered files. |

The `forward()` of a concrete adapter returns every provided variable at
the simulation time through `read()` as replace-mode updates (`variables=`
restricts the names): the simulator passes its `calendar_type` and
`time_origin` settings, the dataset calendar must be `calendar_type`, and the
configuration's `time_interpolation` (`lower` or `linear`, required) evaluates
the records at `time_origin + time`; a time outside the records raises. It
keeps no read position and does not call `transform_to_internal()`.

The `variables` argument of `discover()` takes external model names, whereas
`read()` and `forward()` take internal names. WRF, CESM, and VIC report only
the listed names found in the output, while SWAT ignores the argument. CMIP6
inserts a non-empty list into its directory search pattern as a literal
segment such as `(tas|pr)`, which
matches no directory, so the call raises `ValueError`; call the CMIP6
`discover()` without `variables`.

Current adapter configuration families include:

| Configuration | External family |
| --- | --- |
| `WRFConfig` | Weather Research and Forecasting model output (`wrfout_d*` files). Record times come from the `Times` variable parsed with `time_format`; missing or unparsable times raise, and files are ordered and concatenated along `Time` by these times only, which must increase strictly. |
| `CESMConfig` | Community Earth System Model history output. `soil_layer_thickness` (m, one value per `H2OSOI` layer) is used only when the file has no `DZSOI`. `open()` adds no derived fields to the opened datasets (`CESMConfig` has no `compute_derived`; the former WSPD, Z and THETA had no reader). |
| `CMIP6Config` | CMIP6 climate-model datasets. |
| `VICConfig` | Variable Infiltration Capacity model output. Only `file_format="netcdf"` is accepted; VIC ASCII and binary output raise `ValueError` when the configuration is built, because their columns carry no variable names. `record_interval` with `record_interval_units` declares the interval of water depths per record when the file has no CF time bounds; `time_step` is recorded as a dataset attribute only. |
| `SWATConfig` | Soil and Water Assessment Tool output (`output.hru`, `output_level="hru"`); print settings read from `file.cio` or declared as `nbyr`, `iyr`, `idaf`, `idal`, `iprint`, `nyskip`, and `icalen`, and the run's `calendar`. There is no `time_step` or `skip_warmup` field: the print frequency is IPRINT and the warm-up years are NYSKIP. |

Every adapter configuration (`WRFConfig`, `CESMConfig`, `CMIP6Config`,
`VICConfig`, and `SWATConfig`) has `time_interpolation` (`lower` or `linear`),
the rule of a read at the simulation time; `None` (the default) is not a rule,
and `forward` raises until it is declared. Each configuration also has
`provided_variables`: the catalog variables the adapter writes (mapping
targets of its source; `None` declares every mapping target). They are the
adapter's declared outputs (it reads no state), `forward` returns exactly them
(a provided variable the data lack raises; a request outside them raises),
every update carries its catalog unit, and the adapter is their exclusive
writer (`ConflictResolutionPolicy.EXCLUSIVE`): registering a second writer of
any of them, an internal process such as Penman-Monteith or radiation or
another adapter, raises. SWAT `forward` without an opened dataset of the
configured output level raises an exception; silent empty-forcing return is not permitted.

An adapter's declared expression is its read path, generated from the schema
mapping and the declared source transforms
(`ExternalAdapter.mapping_expression(source)`, one formula per transform kind
in `TRANSFORM_EXPRESSIONS`). `FunctorMetadata` stores the canonical Interconnected Port-Hamiltonian System (IPHS) form
of that declaration (owner, no inputs, the provided variables as outputs);
the read-path formula itself is returned by `mapping_expression`. WRF and
CESM `open()` add no derived fields to the opened datasets; `WRFConfig` has no
`extract_diagnostics` and `CESMConfig` no `compute_derived`.

Adapter availability means that the Python implementation can be imported.
It does not establish that a source file, credentials, variable mapping, unit
conversion, or target grid is ready. Always call `discover()` before `open()`
and validate transformed data before simulation registration.

The `ExternalAdapter` base class is a template: its `forward()` raises
`NotImplementedError`. The configured WRF, CESM, CMIP6, VIC, and SWAT adapters
read existing model-output datasets through this lifecycle. They are not live
two-way couplings with a running external model and do not certify compliance
with an external coupling standard.

### Read path

Variables are mapped by meaning through the `InternalSchema` tables. Every
mapped name is the canonical name of a ContractRegistry variable contract
(never a contract alias, which StateManager refuses to write): WRF `PSFC`,
CESM `PS`, and CMIP6 `ps` map to `atmospheric_pressure`. Each contract lists
the adapter Functor of every source that maps to it (`WRFAdapter`,
`CESMAdapter`, `CMIP6Adapter`, `VICAdapter`, `SWATAdapter`) under
`writers.tendencies` and `provenance.adapter_writers`, in the contract units,
which equal the catalog units. Variables no RELIC process writes are
contracts of kind `forcing` (`convective_precipitation_flux`,
`large_scale_precipitation_flux`, `eastward_wind`, `northward_wind`,
`specific_humidity`, `surface_snow_amount`, and
`mass_content_of_water_in_soil_layer` among them). One CF identity is held by
one variable: surface downwelling shortwave and longwave radiation (WRF
`SWDOWN`, `LWDOWN` and `GLW`; CESM `FSDS` and `FLDS`; CMIP6 `rsds` and
`rlds`) map to `shortwave_down` and `longwave_down` (CF
`surface_downwelling_shortwave_flux_in_air` and
`surface_downwelling_longwave_flux_in_air`), the latent heat flux (WRF `LH`,
CESM `EFLX_LH_TOT`, CMIP6 `hfls`) to `latent_heat_flux` (CF
`surface_upward_latent_heat_flux`, signed, catalog valid range -245 to
1470 W m-2), and VIC `OUT_EVAP` and SWAT `ET` to `evapotranspiration_flux`
(CF `water_evapotranspiration_flux`); the external catalog holds these entries
under the contract names. `eastward_wind` and `northward_wind` are the
near-surface wind at 10 m (WRF `U10` and `V10`, CMIP6 `uas` and `vas`);
`wind_u` and `wind_v` are the wind components of the tracer-transport grid
read by `AtmosphericAdvectionDiffusionFunctor`, a different vertical support,
and hold no CF name. Daily extrema map to `air_temperature_max` and
`air_temperature_min` (`air_temperature` with `cell_methods` `time: maximum`
and `time: minimum`; CMIP6 `tasmax` and `tasmin`, VIC `TMAX` and `TMIN`),
never to `air_temperature`. Convective and large-scale precipitation keep
their own names (CMIP6 `prc` and WRF `RAINC` map to
`convective_precipitation_flux`, WRF `RAINNC` and CESM `PRECL` to
`large_scale_precipitation_flux`; all of them are liquid plus solid), so
`precipitation_flux` comes only from a total such as CMIP6 `pr` or CESM
`PRECT`. CMIP6 `prsn` (solid phase) maps to `snowfall`. CAM writes the snow
rates `PRECSL` and `PRECSC` and no `PRECS`; surface rain
(`PRECT - PRECSC - PRECSL`) and snowfall (`PRECSC + PRECSL`) combine several
fields, so no CESM field maps to `rain_flux` or `snowfall`. Water fluxes such
as VIC `OUT_EVAP` and SWAT `ET` map
to `evapotranspiration_flux` (CF `water_evapotranspiration_flux`), not to the
latent heat flux. The CESM
lowest-model-level fields `TBOT`, `PBOT`, `QBOT`, `UBOT`, and `VBOT` are not
mapped; CESM `PS` maps to `atmospheric_pressure` and `U10` to `wind_speed`, and
`QREFHT` is not mapped. When several external fields map to one internal name,
`read()` uses the declared canonical source: WRF `GLW` for
`longwave_down` and CESM `TREFHT` for `air_temperature`
(`TSA` is the land-only value). Without a declared canonical source the request
raises `ValueError`, and an internal name that the source does not map raises
`KeyError`; for example, `air_temperature` cannot be read from VIC.
`InternalSchema.create_mapping(source, variables)` and
`ExternalAdapter.get_variable_mapping(variables)` return the mapping of every
requested variable or raise `KeyError` naming every variable the source does
not provide.

The units a file declares are authoritative. The VIC and SWAT adapters fill
units from their own tables only for variables whose file declares none. In
these two adapters a data variable records `units_source`: `file` when the file
declares its units and `adapter_table` when the table supplied them; a variable
without units in the file or the table gets no `units_source`. After the
source transform, the unit conversion to the catalog units of
`coupling.external.variables.variable_definitions` runs; a variable missing
from the catalog raises `KeyError`, a field without a units attribute raises
`ValueError`, and an unsupported conversion raises.

Declared source transforms (`InternalSchema.source_transforms()`) run before
the unit conversion. $\rho_w$ is the water density (physical constant
`water_density`, kg m-3):

- Accumulations: WRF `RAINNC` and `RAINC` (with the bucket counters `I_RAINNC`
  and `I_RAINC` and the file attribute `BUCKET_MM`) and `PRCP_ACC` (no bucket)
  accumulate since the simulation start. With the accumulated depth $P_k$ (mm),
  the bucket counter $I_k$, and the bucket size $B$ (mm), the total is
  $A_k=P_k+I_kB$ and the rate is $r_k=\rho_w(A_k-A_{k-1})/(t_k-t_{k-1})$ in
  kg m-2 s-1, with the depth difference converted from mm to m and $t$ in s.
  The rate belongs to $(t_{k-1},t_k]$ and is stored at $t_k$; the first record
  is consumed as the initial accumulation, so the result has one record fewer.
  The file must declare the accumulation in mm. A decreasing accumulation
  raises unless the bucket counter is present and applied, and a counter
  without a positive `BUCKET_MM` raises. Hourly accumulations of 0, 2, and 5 mm
  give 2/3600 and 3/3600 kg m-2 s-1.
- Layered soil water: WRF `SMOIS` and CESM `H2OSOI` are volumetric contents
  $\theta_k$ (m3 m-3) per layer and become $m_k=\rho_w\theta_k\Delta z_k$
  (kg m-2, `mass_content_of_water_in_soil_layer`). The thickness $\Delta z_k$
  (m) comes from the output (`DZS` in wrfout; `DZSOI` in CLM history, whose top
  layers are the `levsoi` layers) or, for CESM output without `DZSOI`, from
  `CESMConfig.soil_layer_thickness` (one value per layer); without either the
  read raises. $\theta=0.3$ and $\Delta z=0.1$ m give 30 kg m-2.
- Water depths: VIC `OUT_SOIL_MOIST` (per `nlayer` layer) and `OUT_SWE` are
  liquid-equivalent depths $d$ and become $m=\rho_wd$ (kg m-2). VIC `OUT_SWE`
  includes snow intercepted by the canopy, whereas the CF name it maps to,
  `surface_snow_amount`, means snow on the ground; the adapter does not
  separate the two.
- Water depth fluxes: VIC `PRCP`, `OUT_EVAP`, `OUT_RUNOFF`, and
  `OUT_BASEFLOW`, and SWAT `PRECIP`, `SURQ_GEN`, `LATQGEN`, and `ET`. A field
  in length per time units becomes $F=\rho_wv$; a water depth $d_k$ per record
  becomes $F_k=\rho_wd_k/(t^{\mathrm{end}}_k-t^{\mathrm{start}}_k)$ in
  kg m-2 s-1. The record interval comes from the CF time bounds of the file's
  `time` coordinate (its `bounds` attribute) or, only when the file declares
  no bounds, from `record_interval` with `record_interval_units` in `VICConfig`
  or `SWATConfig` (declared together, for example `1.0` and `"day"`). The
  interval is never inferred from the coordinate spacing; without either
  declaration the read raises. SWAT `output.hru` records always carry their
  bounds, and a configured `record_interval` then raises.
- Water vapour mixing ratio: WRF `Q2` is the 2 m mixing ratio $w$ (WRF
  Registry `QV at 2 M`, kg kg-1); the specific humidity is $q=w/(1+w)$
  (transform `mixing_ratio`; the file must declare `kg kg-1`).
- Grid-relative wind: WRF `U10` and `V10` are grid-relative; `eastward_wind`
  and `northward_wind` are $u_e=u\cos a-v\sin a$ and $v_e=v\cos a+u\sin a$
  with the file's `COSALPHA` and `SINALPHA` (transform `earth_relative_wind`;
  missing fields raise).

### Transform and quality control

`transform_to_internal()` regrids a field only when `target_grid` is given. The
basis and the positivity come from `variable_metadata[name]`, which must
declare `spatial_basis` and `valid_min`, or from the external variable catalog;
an undeclared variable raises `KeyError`. `cell_total` and `area_density`
fields use conservative remapping, `intensive` fields bilinear interpolation,
and `categorical` fields nearest neighbour. The source grid is built from the
field's CF latitude and longitude coordinates. The regridding step identifies
them by their units (`degrees_north`, `degrees_east`) with a consistent
`standard_name` and `axis`, never by coordinate names; a field without them
raises. The adapters' `open()` differs: WRF sets `standard_name` and `units` on
`XLAT` and `XLONG`, and CESM on `lat` and `lon`, by coordinate name and
overwrites the values in the file, so for these two sources the identification
rests on those names; VIC adds the attributes to `lat` and `lon` only where the
file declares none. Conservative remapping needs the cell bounds that the file
declares through the coordinates' `bounds` attribute (for example `lat_bnds`
and `lon_bnds`); bounds are not inferred from the coordinate spacing. 1-D
coordinates give a `regular` source grid with `coord_system` `latlon`, so the
conservative weights and cell areas are spherical, in m2 (see "Spatial
regridding" above), and a target grid in another coordinate system raises
`ValueError`. Curvilinear (2-D) coordinates carry no bounds, so WRF
area densities, such as radiation and precipitation fluxes, cannot be regridded
conservatively: wrfout has no cell corners, and the call raises. The ingestion
regridding runs through NumPy and xarray, so no gradient flows through it.

The catalog quality control marks values outside the catalog `valid_min` and
`valid_max` as missing (NaN) and never fills missing values. The missing
fraction is recorded under `quality_control` in the tensor's read metadata, and
a field whose missing fraction exceeds
`coupling.external.regrid.missing_threshold` (current configuration: 0.5)
raises `ValueError`. The result is float32; the tensor passed in is not
modified, and its gradient flag is left as the caller set it. `RegridOperator`
itself applies no missing-data threshold: it reports the NaN fraction of the
field it receives as `RegridResult.missing_data_fraction`.

Every external catalog entry declares `standard_name`, `long_name`, `units`,
and `category` (one of the `VariableCategory` names); a missing field or an
unknown category raises when the catalog is loaded; no entry is skipped and
no category is assumed. The valid range of `evapotranspiration_flux` is
declared once, with its references, in the external catalog entry
([-1.0e-4, 6.0e-4] kg m-2 s-1, signed: condensation and dew are negative);
`coupling.scales.*.variables` repeat it, and `latent_heat_flux` has
$\lambda$ times that range ([-245, 1470] W m-2, $\lambda=2.45\times10^{6}$
J kg-1). The upper bound is the energy limit (1470 W m-2 of latent heat, about
1.08 times the total solar irradiance); the weighing-lysimeter alfalfa ET of
Tolk, Evett and Howell (2006, Agron. J. 98(6):1646-1654), 11.3 mm d-1 on
average over 13 advective days and above 15 mm d-1 on 3 days, is its
observational support. `coupling.variables.<name>.bounds` holds exactly the
bounds of the seven outputs registered by `register_coupling_variables`
(boundary-layer conductance, root uptake profile, effective transpiration and
the four soil-vegetation diagnostics), its only reader. The contracts of
both variables declare no clipping bounds.

Each of the following variables has one valid range, declared with its
references in the external catalog entry when the variable is listed there
and otherwise in `coupling.scales.defaults.variables` (the daily catalog
repeats it); every other catalog repeats it: `precipitation_flux` and
`runoff_flux` [0, 0.52] kg m-2 s-1 (0.52 = 31.2 mm in 1 min, the World Meteorological Organization (WMO) greatest
1-min rainfall; the runoff bound is a convention equal to it);
`net_radiation` [-400, 2200] W m-2 (upper bound: $R_n\le S_{down}$ and the
greatest measured surface global irradiance 2177 W m-2, Cordero et al. 2023;
lower bound a convention); `shortwave_down` [0, 2200] W m-2 (the same
observed extreme rounded up); `longwave_down` [0, 700] W m-2 (derived:
$L_{down}=\varepsilon_a\sigma T_a^4\le\sigma T_{max}^4$ = 671 W m-2 with
emissivity at most 1 and the WMO greatest air temperature 56.7 degC);
`sensible_heat_flux` [-200, 1000] W m-2
(convention); `stomatal_conductance` [0, 2.0] mol m-2 s-1 (convention above
the woody-angiosperm maxima of Murray et al. 2019; the leaf-level conductance
to water vapour in the unit of its readers `ozone_stress` and `oxygen_18`);
`canopy_conductance` [0, 0.05] m s-1 (convention, 1.5 times the mean maximum
crop surface conductance of Kelliher et al. 1995); `net_primary_productivity`
[-1.0e-6, 1.0e-6] kg C m-2 s-1 (signed, NPP = GPP - R_a; convention above the
peak hourly GPP of irrigated maize, Suyker et al. 2005); `soil_moisture`
[0, 1] m3 m-3 (a volume fraction). `heterotrophic_respiration` is a
non-negative release (its writers raise on a negative rate and it accumulates
into the `carbon.total` boundary ledger); its contract declares `lower_bound`
0 and the sign convention `non_negative_release_to_atmosphere` (upward
positive, as CF
`surface_upward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_heterotrophic_respiration`),
and it has no catalog range.

The external data validator derives the statistics of its outlier check from
the data; it keeps no per-variable reference statistics. Every threshold of
the validator is a declared parameter; an unset `ValidationConfig` field takes
it and an explicit value is kept: the outlier z-score 4.0, the flagged
fractions 0.05 (WARNING) and 0.1 (ERROR), the largest time gap 48 h, the
oldest accepted CF version 1.6, the global-coverage spans 350 and 170 deg and
the check switches under `coupling.external.validation`; a missing-data
fraction above `coupling.external.io.missing_data_warning_threshold` (0.1) is
a WARNING and above `coupling.external.regrid.missing_threshold` (0.5) an
ERROR. The outlier check assumes no minimum sample size: $n$ values cannot
give a z-score above $\sqrt{n-1}$ (Shiffler 1988).

### SWAT `output.hru`

Only the SWAT 2012 `output.hru` text table is implemented; use
`SWATConfig(root_path=..., output_level="hru", ...)`. `output_level` is
required, and any other value, such as `"sub"`, `"rch"`, or `"rsv"`, raises
`ValueError` when the configuration is built, because the row layouts of
`output.sub`, `output.rch`, and `output.rsv` are not implemented.
`SWATAdapter` requires a `SWATConfig`. `output.std`, `output.wtr`, and SWAT
SQLite outputs are not read, and `SWATConfig` has no `output_format` or
`database_path` field. The text reader follows the SWAT 2012 source:

- Header (headout.f90, format 1020):
  `LULC  HRU       GIS  SUB  MGT  MON   AREAkm2`, then one 10-character field
  per variable, the SWAT name followed by its unit suffix, for example
  `PRECIPmm`, `SOL_TMPdgC`, or `SOLARMJ/m2`. The suffix is the file's unit
  declaration (`units_source = file`). The calendar-date header written with
  ICALEN = 1 (format 1021) raises.
- Rows (hrumon.f90, hruyr.f90, and hruaa.f90; formats 1000, 1001, and 2000)
  are read at their fixed Fortran field positions: HRU as `i5` or `i7`, AREA as
  `e10.5`, and each value as a 10-character field. Short rows and blank,
  overflowed (`*`), or non-numeric fields raise.
- The print settings NBYR, IYR, IDAF, IDAL, IPRINT (0 monthly, 1 daily,
  2 annual), NYSKIP, and ICALEN are read from `file.cio` in `root_path` the way
  SWAT reads that file: by record position, replaying the read statements of
  readfile.f90. The `value | NAME : description` labels that ArcSWAT and QSWAT
  write are optional; a label that names a different setting from the one
  SWAT reads from that record raises. A `file.cio` that ends before ICALEN
  gives ICALEN = 0, SWAT's initial value; one that ends before a read SWAT
  requires raises. Without `file.cio`, the settings are declared in
  `SWATConfig` (`nbyr`, `iyr`, `idaf`, `idal`, `iprint`, `nyskip`, `icalen`). A
  configured value that differs from `file.cio`, an undeclared setting, and
  ICALEN = 1 raise. NYSKIP >= NBYR is set to NBYR - 1, as SWAT does.
- The CF calendar of the run is always declared in `SWATConfig.calendar` and
  must give SWAT's year and month lengths. SWAT 2012 uses a leap year when the
  year is divisible by 4, which is the CF `julian` calendar; `standard` agrees
  for 1901 to 2099.
- SWAT prints only the years after the NYSKIP warm-up years. IDAF applies to
  the first simulated year and IDAL to the last one (0 means the first or the
  last day of the year). With IPRINT = 0, each printed year has its monthly
  blocks followed by one annual block (MON = four-digit year), and the file
  ends with one average-annual block (MON = years averaged, identified by its
  position). With IPRINT = 1 there are only daily blocks (MON = julian day).
  With IPRINT = 2 there is one annual block per printed year and the
  average-annual block. The file must be exactly the block sequence of the
  declared settings; any deviation raises.
- The records are the monthly or daily blocks (IPRINT 0 or 1) or the annual
  blocks (IPRINT 2). The summary blocks are excluded from the time series. The
  dataset attribute `swat_annual_summary_blocks` counts the excluded annual
  blocks (0 for IPRINT 1 and 2; with IPRINT 2 the annual blocks are the
  records), `swat_average_annual_summary_blocks` counts the average-annual
  blocks, and `swat_average_annual_years` is the number of years averaged that
  the average-annual block gives in its MON field (format f4.1), absent when
  the file has no such block. Each record carries CF time bounds (`time_bnds`,
  days since IYR-01-01 in the declared calendar): its calendar month, day, or
  year, clipped to the simulated window from IDAF to IDAL; the time value is
  the middle of that interval.
- The mapped SWAT columns are `PRECIP`, `SURQ_GEN`, `LATQGEN`, `ET`, and
  `SOL_TMP`. The first four are water depths per record and become fluxes over
  the record bounds: in a year that is not a leap year, 28 mm in February and
  31 mm in March are both 1 mm per day (about 1.157e-5 kg m-2 s-1). `SOL_TMP`
  maps to `soil_temperature`. A date `time_window` raises because the records
  have a numeric CF time coordinate.

## Plugin coupling

Plugin coupling is included by `get_available_couplings()` only when the
plugin adapter package is importable.

```python
from relic.coupling import create_coupling_functor

plugin = create_coupling_functor(
    coupling_type="plugin",
    name="external_coupler",
    config={
        "plugin_path": r"C:\models\coupler\plugin.exe",
        "source_scale": "grid_cell",
        "target_scale": "grid_cell",
        "coupling_type": "spatial",
    },
)
print(plugin.name)
```

`plugin_path` or its `path` alias is required. Missing plugin support raises
`RuntimeError`; a missing path raises `ValueError`. A plugin coupling
declares the scales it couples: `source_scale`, `target_scale`, and
`coupling_type` (`spatial`, `temporal`, `hierarchical`, or `stochastic`) are
required config keys; a missing key raises `KeyError`, and another coupling
type raises `ValueError`. A plugin adapter still
must obey the declared input/output, units, update, state, and execution-group
contracts.

## Composition rules

- Register coupling Functors through the simulator or GraphIR pipeline.
- Declare dependencies through metadata, ports, links, chain maps, or an
  `IntegrationPlan`.
- Do not call another Functor directly from a coupling `forward()` method.
- Route residuals and overflow explicitly; do not silently discard them.
- Use `chain_map_check`, `port_matching_check`, and
  `overflow_routing_check` for structural coupling evidence. These checks
  validate declarations or supplied samples; they do not evaluate per-step port
  power or donor/recipient budgets. A closure claim also names the consuming
  Functor, the single state commit, the unit/support/time basis, and the actual
  residual; a diagnostic residual discloses an imbalance and does not close it.
- Read thresholds and conversion settings through the parameter system.

## Related documentation

- [Architecture](../architecture.md)
- [Orchestration](../orchestration.md)
- [Graph Validation and Execution](graph_validation_and_execution.md)
- [Scientific Structure and Extensions API](../api/scientific_structure_extensions.md)
- [Science index](../science/index.rst)
- [Theory index](../theory/index.rst)
