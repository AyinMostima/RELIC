# Multi-Scale Configuration and Unified Metadata Specification

## 1. Process Definition and Scope

`CentralModelBuilder` (`relic/coupling/scales/hierarchical/model_builder.py`, factory `create_model_builder`) extracts per-level incremental declarations from the six scale modules in `relic/models/` (`individual.py`, `patch.py`, `hillslope.py`, `basin.py`, `landscape.py`, and `regional.py`) and composes deduplicated effective Functor lists and typed `VariableSpec` catalogs across the configured spatial hierarchy. Theoretical foundations for typed spatial bases and conservative multi-scale metadata appear in *Methods of Ecological Analysis*, Chapters 2, 23, and 24.

The builder is static configuration and metadata-audit infrastructure. It is not a `FunctorBase` subclass, holds no state tensors, registers no variables in `StateManager`, and executes no field aggregation. Numerical spatial reductions execute separately in `HierarchicalAggregator` (`relic/coupling/scales/hierarchical/aggregator.py`) and the `ScaleManager` facade (`relic/coupling/scales/scale_manager.py`), documented in [scale_coupling.md](scale_coupling.md) and [spatiotemporal_scale_topology.md](spatiotemporal_scale_topology.md).

## 2. Governing Equations

### 2.1 Hierarchy Delta Extraction and Effective Composition

Let $\mathcal{H} = (s_1, \dots, s_6) = (\text{individual}, \text{patch}, \text{hillslope}, \text{basin}, \text{landscape}, \text{regional})$ be the ordered hierarchy read from `coupling.scales.hierarchy.levels` via `relic.core.constants.get_scale_hierarchy()`. From cumulative module lists $F(s_i) = \text{LAYER\_FUNCTORS}$ and $V(s_i) = \text{EXPOSED\_VARIABLES}$, constructor delta extraction retains only names not declared at any earlier hierarchy position:

$$
F_\Delta(s_i) = F(s_i) \setminus \bigcup_{k < i} F(s_k), \qquad
V_\Delta(s_i) = V(s_i) \setminus \bigcup_{k < i} V(s_k).
$$

For target level $s_m \in \mathcal{H}$ with prefix $\operatorname{hierarchy}(s_m) = (s_1, \dots, s_m)$, effective configurations are formed on demand as

$$
F_{\mathrm{eff}}(s_m) = \operatorname{deduplicate}\!\left(\bigsqcup_{i=1}^{m} F_\Delta(s_i)\right), \qquad
V_{\mathrm{eff}}(s_m) = \operatorname{resolve}_{\mathrm{conflicts}}\!\left(\bigsqcup_{i=1}^{m} V_\Delta(s_i)\right),
$$

where $\operatorname{deduplicate}$ preserves first-occurrence declaration order deterministically (independent of `PYTHONHASHSEED`; an effective Functor list is a membership catalog, not a topological execution schedule). Because `basin` precedes `landscape` in `coupling.scales.hierarchy.levels` while `relic/models/basin.py` builds its cumulative lists on `relic/models/landscape.py` (which builds on `patch.py`), `basin` captures both basin and landscape declarations into $F_\Delta(\text{basin})$ and $V_\Delta(\text{basin})$. Consequently, $F_\Delta(\text{landscape}) = \emptyset$ and $V_\Delta(\text{landscape}) = \emptyset$, while $F_{\mathrm{eff}}(\text{landscape})$ contains the hillslope and basin entries even though `relic.models.landscape.LAYER_FUNCTORS` omits them.

### 2.2 Spatial Reductions Represented by `AggregationMethod`

For child spatial units $i$ with area $A_i \ge 0$ ($\mathrm{m^2}$) and parent-membership share $M_{ij} \in [0, 1]$ (`HierarchyIndex.membership_mapping`), the normalized area fraction within parent $j$ is (Jones, 1999; Tague and Band, 2004):

$$
f_{ij} = \frac{A_i M_{ij}}{\sum_k A_k M_{kj}}, \qquad \sum_i f_{ij} = 1 \quad \left(\sum_k A_k M_{kj} > 0\right).
$$

The four operational spatial reductions associated with `VariableSpec` declarations are:

1. **Area-weighted mean (`AggregationMethod.AREA_WEIGHTED_MEAN`)** for `intensive` states and `area_density` fluxes $x_i$ (preserving density units, e.g., $\mathrm{kg\,m^{-2}\,s^{-1}}$):

   $$y_j = \sum_i f_{ij}\,x_i.$$

2. **Cell-total summation (`AggregationMethod.SUM`)** for `cell_total` amounts $X_i$ (e.g., $\mathrm{m^3\,s^{-1}}$ or $\mathrm{kg\,s^{-1}}$; no `AggregationMethod` converts an `area_density` directly into a parent total amount):

   $$Y_j = \sum_i X_i M_{ij}.$$

3. **Area-weighted categorical mode (`AggregationMethod.MODE`)** with deterministic tie-breaking toward the smallest class code $c$ (CF-1.8 Section 7.3):

   $$S_{cj} = \sum_i A_i M_{ij}\,\mathbb{I}[x_i = c], \qquad \operatorname{mode}_j = \min\left\{c : S_{cj} = \max_{c'} S_{c'j}\right\}.$$

4. **Effective aerodynamic roughness length (`AggregationMethod.EFFECTIVE_ROUGHNESS_LENGTH`)** via neutral drag-coefficient blending at blending height $z_r$ for $0 < z_{0,i} < z_r$ (Byun and Ching, 1999, Eqs. 12-37' and 12-48):

   $$z_{0,j} = z_r \exp\!\left[-\left(\sum_i \frac{f_{ij}}{\ln^2(z_r / z_{0,i})}\right)^{-1/2}\right].$$

### 2.3 Basis-Method Admission and Bounded CF `cell_methods` Generation

`VariableSpec.__post_init__` parses `spatial_basis` via `SpatialQuantityBasis.parse` (rejecting bare `"extensive"`) and enforces conservative basis-method compatibility before regenerating `cell_methods`:

| `SpatialQuantityBasis` | Inferred Default (No Hint) | Accepted `AggregationMethod` | Builder `cell_methods` | `ScaleManager` Facade `area:` Token |
| --- | --- | --- | --- | --- |
| `intensive` | `AREA_WEIGHTED_MEAN` | All 9 methods (`SUM` emits warning) | `"area: <op> time: mean"` (`MODE` / `STANDARD_DEVIATION`: no `time:` token) | `area: mean` |
| `cell_total` | `SUM` | `SUM` only | `"area: sum time: mean"` | `area: sum` |
| `area_density` | `AREA_WEIGHTED_MEAN` | `AREA_WEIGHTED_MEAN` only | `"area: mean time: mean"` | `area: mean` |
| `categorical` | `MODE` | `MODE` only | `"area: mode time: point"` | `area: mode` (aggregation only) |

For `EFFECTIVE_ROUGHNESS_LENGTH`, the generated string is `"area: mean (comment: effective roughness length from linearly additive drag coefficients at the blending height) time: mean"`.

## 3. Variable Table

| Data Structure / Field | Meaning | Type / Units | Owner | Role |
| --- | --- | --- | --- | --- |
| `VariableSpec.units` | Physical unit string (`"1"` if omitted in raw dict) | `str` | `model_builder.py` | Critical conflict field |
| `VariableSpec.standard_name` | CF standard name (`None` when absent; never fabricated) | `Optional[str]` | `model_builder.py` | Critical conflict field |
| `VariableSpec.spatial_basis` | `intensive`, `cell_total`, `area_density`, or `categorical` | `SpatialQuantityBasis` | `scales/types.py` | Critical conflict field |
| `VariableSpec.description` | Human-readable variable summary | `str` | `model_builder.py` | Non-critical conflict field |
| `VariableSpec.aggregation` | Declared or inferred spatial reduction label | `AggregationMethod` | `model_builder.py` | Non-critical conflict field |
| `VariableSpec.cell_methods` | Auto-regenerated bounded CF-1.8 `cell_methods` string | `str` | `model_builder.py` | Derived metadata field |
| `VariableSpec.extras` | Remaining unparsed dictionary attributes (`bounds`, flags) | `Dict[str, Any]` | `model_builder.py` | Stored pass-through metadata |
| `ScaleConfig` | Per-level `delta_functors`, `delta_variables`, `coordinate_vars`, `coupling` | `@dataclass` | `model_builder.py` | Builder level record |
| `ConflictReport` | `errors` and `warnings` from `check_conflicts(scale)` | `@dataclass` | `model_builder.py` | Non-raising audit report |

## 4. Parameter Table

| Key / Resource | Meaning | Type / Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| `coupling.scales.hierarchy.levels` | Ordered 6-level spatial scale sequence $\mathcal{H}$ | `list[str]` | `relic/parameters/database/relic/coupling/hierarchy_mapping.yaml` | Tague and Band (2004) |
| `coupling.scales.defaults.roughness_blending_height` | Neutral drag blending height $z_r$ ($10.0$, bounds $[1, 10]$) | $\mathrm{m}$ | `relic/parameters/database/relic/coupling/defaults.yaml` | Byun and Ching (1999) |
| `coupling.scales.defaults.variables`, `coupling.scales.daily.variables` | Runtime spatial-operator variable basis catalogs | `dict` | `relic/parameters/database/relic/coupling/scale_variables.yaml` | RELIC spatial basis SSOT |
| `cf_standard_name_table.xml` | Bundled CF Standard Name Table (version 94) | XML resource | `relic/data/standards/cf_standard_name_table.xml` | CF Metadata Conventions v1.8 |

## 5. Numerical Methods and Discretization

`CentralModelBuilder` performs deterministic set/dictionary composition without caching merged outputs:

1. **Basis and method resolution**: `resolve_spatial_basis(name, entry)` requires explicit `spatial_basis` (checking consistency if `variable_type` is also present) or unambiguous `variable_type` (`"intensive"` or `"categorical"`), raising `ValueError` on bare `"extensive"` or missing keys. Explicit `aggregation_method` values must belong to `AggregationMethod` (`"area_weighted_sum"` is rejected at the builder level because it belongs to `HierarchicalAggregator` `ScalingStrategy` and not `VariableSpec` `AggregationMethod`).
2. **Conflict detection and overrides**: When two definitions of a variable meet in `get_effective_variables` (e.g., after mutating `get_scale_config()`), mismatches in `units`, `standard_name`, or `spatial_basis` raise `ValueError` unless listed in `allow_overrides[var_name]`, which downgrades the mismatch to `UserWarning` with last-definition-wins semantics. Mismatches in `description`, `aggregation`, or `cell_methods` always emit `UserWarning`.
3. **CF and UDUNITS audit**: `validate_cf_and_units(variables)` delegates to `CFConventions.validate_cf_compliance` (`relic/data/standards/cf_conventions.py`). Standard names are checked against entries and aliases in `cf_standard_name_table.xml` (v94); units are checked via `cf_units` / UDUNITS convertibility (for example, `"kg C m-2 s-1"` is flagged as a UDUNITS error because `C` parses as coulomb, whereas canonical CF units are `"kg m-2 s-1"`).
4. **Autograd boundary of executed methods**: In `HierarchicalAggregator`, `area_weighted_mean`, `area_weighted_sum`, and `effective_roughness_length` preserve PyTorch gradients, whereas `mode` uses `torch.unique` and raises `NotImplementedError` on `.backward()` unless inputs or outputs are detached prior to loss assembly.

```python
from relic.coupling.scales.hierarchical.model_builder import create_model_builder

builder = create_model_builder(allow_overrides={"lai": ["units"]})
patch_functors = builder.get_effective_functors("patch")
patch_vars = builder.get_effective_variables("patch")
cf_report = builder.validate_cf_and_units(patch_vars)
```

## 6. Conservation, Boundary Conditions, and Ledgers

`CentralModelBuilder` enforces structural metadata conservation rules (preventing `cell_total` variables from being averaged or `area_density` variables from being summed without area conversion `convert_spatial_basis`), while runtime mass/energy conservation ledgers execute in `HierarchicalAggregator`. Notice the opposite metadata-precedence contracts of the two runtime consumers, neither of which imports `CentralModelBuilder`:

- `HierarchicalAggregator` checks call-level `variable_metadata` first, then `coupling.scales.<source>.variables`, `coupling.scales.<target>.variables`, and `coupling.scales.defaults.variables`. Children with $\sum_j M_{ij} = 0$ are tracked separately as `uncovered_source_amount` without silent redistribution.
- `ScaleManager.aggregate` and `ScaleManager.disaggregate` query `coupling.scales.<source_scale>.variables` and `coupling.scales.defaults.variables` first (`find_catalog_entry` in `relic/coupling/scales/variable_catalog.py`), falling back to the call's `metadata` argument only when `KeyError` indicates the variable is absent from all YAML catalogs. `CFAttributeSynchronizer` updates `cell_methods` and `history` per variable basis, and refuses categorical disaggregation.

## 7. Code Mapping

| Class / Function | File Path | Architectural Role | Notes |
| --- | --- | --- | --- |
| `CentralModelBuilder`, `VariableSpec`, `ScaleConfig`, `AggregationMethod`, `create_model_builder` | `relic/coupling/scales/hierarchical/model_builder.py` | Multi-scale configuration delta builder (not exported by `relic.coupling.scales.hierarchical.__init__`) | Demonstrated in `examples/complete_multiscale_models_demo.py` |
| `SpatialQuantityBasis`, `resolve_spatial_basis`, `convert_spatial_basis` | `relic/coupling/scales/types.py` | Canonical 4-basis spatial representation contract | Rejects bare `"extensive"` |
| `get_scale_functors`, `get_scale_variables` | `relic/models/__init__.py` (`individual.py` .. `regional.py`) | Raw cumulative per-module lists | No cross-level deduplication or basis validation |
| `CFConventions`, `CFAttributeSynchronizer`, `cf_variable_identity` | `relic/data/standards/cf_conventions.py` | Bundled CF-1.8 v94 standard-name and UDUNITS validator | Shared `(standard_name, cell_methods)` identity with `SSOTCatalog` |

## 8. Implementation Status

- **Implemented (`[I]`)**: Deterministic 6-level delta extraction, effective Functor/variable composition, `SpatialQuantityBasis` validation, 9-method `AggregationMethod` enum (`SUM`, `MEAN`, `AREA_WEIGHTED_MEAN`, `VOLUME_WEIGHTED_MEAN`, `MAX`, `MIN`, `STANDARD_DEVIATION`, `MODE`, `EFFECTIVE_ROUGHNESS_LENGTH`), bounded CF `cell_methods` generation, critical/non-critical conflict detection with `allow_overrides`, and `validate_cf_and_units` against CF Standard Name Table v94.
- **Metadata-only / Not implemented in runtime aggregator (`[M]` / `[N]`)**: Five builder `AggregationMethod` labels (`SUM`, `MEAN`, `VOLUME_WEIGHTED_MEAN`, `MAX`, `MIN`, `STANDARD_DEVIATION` as literal strategy strings) serve strictly as metadata labels and are not `HierarchicalAggregator.aggregate` strategy names (`HierarchicalAggregator` implements `area_weighted_mean`, `area_weighted_sum`, `mode`, `nearest`, and `effective_roughness_length`); dataset-level NetCDF CF-compliance certification and automatic wiring from `CentralModelBuilder` into `StateManager` remain `[N]`.
- **IPHS Equivalence Certificates**: None (`0` certificates reference `multiscale_config.md`).

## 9. References

1. Byun, D. W., and Ching, J. K. S. (Eds.) (1999). *Science Algorithms of the EPA Models-3 Community Multiscale Air Quality (CMAQ) Modeling System* (EPA/600/R-99/030, Chapter 12: Meteorology-Chemistry Interface Processor). U.S. Environmental Protection Agency.
2. Eaton, B., Gregory, J., Drach, B., Taylor, K., Hankin, S., et al. (2020). *NetCDF Climate and Forecast (CF) Metadata Conventions, Version 1.8* (Section 7.3: Cell Methods) and CF Standard Name Table Version 94.
3. Jones, P. W. (1999). First- and second-order conservative remapping schemes for grids in spherical coordinates. *Monthly Weather Review*, 127(9), 2204-2210.
4. Rew, R., and Davis, G. (1990). NetCDF: An interface for scientific data access. *IEEE Computer Graphics and Applications*, 10(4), 76-82.
5. Tague, C. L., and Band, L. E. (2004). RHESSys: Regional Hydro-Ecologic Simulation System—An object-oriented approach to spatially distributed modeling of carbon, water, and nutrient cycling. *Earth Interactions*, 8(19), 1-42.
