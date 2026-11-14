# Plant Functional Type Parameterization

## 1. Process Definition and Scope

The Plant Functional Type (PFT) parameterization subsystem provides typed, immutable trait views over the centralized `ParameterManager` database for fifteen broad vegetation categories (`PFTCategory`). PFT definitions supply configuration traits to biosphere Functors at initialization and carry no `forward()` method, state ownership, flux ports, or runtime graph node.

The former `PFTFunctor` is retired (`[H]`) and excluded from `relic.scientific.biosphere.pft` and `relic.scientific.biosphere` exports. Although `PhotosynthesisPathway` enumerates `C3`, `C4`, and `CAM`, executable photosynthesis kernels implement `C3` and `C4` only (`CAM` biochemical execution is `[N]`). Deeper trait-aggregation theory is covered in *Methods of Ecological Analysis*, Chapters 16 and 20.

## 2. Governing Equations

Each `PFTDefinition` organizes trait parameters for category $p \in \mathcal{P}$ into six typed parameter groups:

$$
\theta_{\mathrm{pft}}(p) = \bigl\{\theta_{\mathrm{photo}}(p),\; \theta_{\mathrm{resp}}(p),\; \theta_{\mathrm{alloc}}(p),\; \theta_{\mathrm{phen}}(p),\; \theta_{\mathrm{bvoc}}(p),\; \theta_{\mathrm{stress}}(p)\bigr\}
$$

where organ base allocation fractions satisfy the simplex constraint:

$$
f_{\mathrm{leaf}}(p) + f_{\mathrm{stem}}(p) + f_{\mathrm{root}}(p) = 1, \qquad f_i(p) \in [0, 1]
$$

When multiple PFTs occupy a mosaic tile with area weights $w_p \ge 0$ ($\sum_p w_p = 1$), effective tile-level linear traits follow weighted aggregation $\theta_{\mathrm{eff}} = \sum_p w_p\,\theta_{\mathrm{pft}}(p)$ (Oleson et al., 2013), while nonlinear process responses evaluate the owning biosphere Functor per PFT before flux aggregation.

## 3. Variable Table

PFT parameterization declares zero `StateManager` runtime ports. Its fifteen canonical `PFTCategory` identifiers are:

| Category Group | `PFTCategory` Identifiers | Primary Pathway |
| --- | --- | --- |
| Needleleaf trees | `needleleaf_evergreen_temperate`, `needleleaf_evergreen_boreal`, `needleleaf_deciduous_boreal` | `C3` |
| Broadleaf trees | `broadleaf_evergreen_tropical`, `broadleaf_evergreen_temperate`, `broadleaf_deciduous_tropical`, `broadleaf_deciduous_temperate`, `broadleaf_deciduous_boreal` | `C3` |
| Shrubs | `shrub_evergreen_temperate`, `shrub_deciduous_temperate` | `C3` |
| Grasses | `c3_grass_arctic`, `c3_grass_temperate`, `c4_grass_temperate` | `C3` / `C4` |
| Crops | `crop_temperate`, `crop_tropical` | `C3` / `C4` |

## 4. Parameter Table

Each PFT loads its traits from `relic/parameters/database/relic/biosphere/pft/<pft_name>.yaml` (indexed by `pft/__init__.yaml` alongside `pft/metadata.yaml`). Four process-wide coefficients avoid YAML duplication through `_SHARED_PROCESS_PARAMETER_BINDINGS` in `pft_types.py`:

| Typed `PFTDefinition` Field | Canonical `ParameterManager` Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| `photosynthesis.{vcmax25,jmax25,rd25}` | `biosphere.pft.<pft>.photosynthesis.{vcmax25,jmax25,rd25}` | $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$ | `pft/<pft>.yaml` | Farquhar et al. (1980); Kattge et al. (2011) |
| `photosynthesis.rd_q10` | `biosphere.photosynthesis.rd_q10` | $1$ | `photosynthesis.yaml` (shared) | Oleson et al. (2013) |
| `respiration.{maint_resp_leaf,maint_resp_stem,maint_resp_root}` | `biosphere.pft.<pft>.respiration.maintenance.{rm_leaf_25,rm_stem_25,rm_root_25}` (via `_RESPIRATION_PARAMETER_PATH_MAP`) | $\mathrm{s^{-1}}$ | `pft/<pft>.yaml` | Oleson et al. (2013) |
| `respiration.q10_maint_resp` | `biosphere.respiration.maintenance.q10_respiration` | $1$ | `respiration.yaml` (shared) | Atkin and Tjoelker (2003) |
| `respiration.n_effect_maint_resp` | `biosphere.respiration.growth.np_response_coeff` | $1$ | `respiration.yaml` (shared) | Oleson et al. (2013) |
| `respiration.base_resp_rate` | `biosphere.respiration.maintenance.min_respiration_rate` | $\mathrm{s^{-1}}$ | `respiration.yaml` (shared) | Oleson et al. (2013) |
| `allocation.{f_leaf,f_stem,f_root}` | `biosphere.pft.<pft>.allocation.{f_leaf,f_stem,f_root}` | $1$ | `pft/<pft>.yaml` | Lawrence and Chase (2007) |
| `phenology.*` | `biosphere.pft.<pft>.phenology.*` | $\mathrm{K}$, $\mathrm{day}$, $\mathrm{m^2\,kg^{-1}\,C}$ | `pft/<pft>.yaml` | Oleson et al. (2013) |
| `bvoc.{ef_isoprene,ef_monoterpene,ef_sesquiterpene,ef_other_voc}` | `biosphere.pft.<pft>.bvoc.ef_*` | $\mathrm{\mu g\,g^{-1}\,DW\,h^{-1}}$ | `pft/<pft>.yaml` | Guenther et al. (2012) |
| `stress.*` | `biosphere.pft.<pft>.stress.*` | $\mathrm{MPa}$, $\mathrm{K}$, $1$ | `pft/<pft>.yaml` | Medlyn et al. (2011); Oleson et al. (2013) |

## 5. Numerical Methods and Discretization

`PFTRegistry` is a thread-safe singleton (`PFTRegistry.get_instance()`) that lazily loads PFT YAML files into `ParameterManager` via `load_default_pfts()` (`lightweight=True`) and materializes full `PFTDefinition` dataclasses upon lookup. Missing PFT directories, unindexed PFT names outside `PFTCategory`, or incomplete non-shared trait entries raise `PFTRegistryError` or `ValueError` without fabricating fallback traits.

## 6. Conservation, Boundary Conditions, and Ledgers

PFT configuration advances no state and declares no conservation ledger (`conservation_laws=[]`). Allocation simplex validation ($f_{\mathrm{leaf}} + f_{\mathrm{stem}} + f_{\mathrm{root}} = 1$) and physical parameter-bound checks execute during `PFTDefinition` construction; runtime mass and energy ledgers remain owned by the consuming scientific Functors.

## 7. Code Mapping

| Component | File Path | Abstract Contract | Bridge Class | Role |
| --- | --- | --- | --- | --- |
| `PFTDefinition`, `PFTCategory`, `PhotosynthesisPathway` | `relic/scientific/biosphere/pft/pft_types.py` | Non-Functor typed configuration | None | Immutable PFT trait dataclasses |
| `PFTParameterBase` | `relic/scientific/biosphere/pft/pft_parameter_base.py` | Non-Functor base helper | None | `ParameterManager` lookup and unit/bound checks |
| `PFTRegistry` | `relic/scientific/biosphere/pft/pft_registry.py` | Non-Functor registry | None | Thread-safe YAML loading and PFT lookup |

## 8. Implementation Status

- `PFTDefinition`, `PFTParameterBase`, and `PFTRegistry`: `[I]` non-Functor parameter-support layer (no IPHS process certificate).
- `PFTFunctor`: `[H]` retired and removed from public biosphere exports so PFT traits cannot masquerade as a state-advancing process.
- `PhotosynthesisPathway.CAM` biochemical execution kernel: `[N]` (enum identifier reserved; only `C3` and `C4` execute in `FarquharPhotosynthesis`).

## 9. References

1. Farquhar, G. D., von Caemmerer, S., and Berry, J. A. (1980). A biochemical model of photosynthetic $\mathrm{CO_2}$ assimilation in leaves of $\mathrm{C_3}$ species. *Planta*, 149, 78-90.
2. Lawrence, P. J., and Chase, T. N. (2007). Representing a new MODIS consistent land surface in the Community Land Model (CLM 3.0). *Journal of Geophysical Research: Biogeosciences*, 112, G01023.
3. Kattge, J., Diaz, S., Lavorel, S., et al. (2011). TRY: a global database of plant traits. *Global Change Biology*, 17, 2905-2935.
4. Medlyn, B. E., Duursma, R. A., Eamus, D., et al. (2011). Reconciling the optimal and empirical approaches to modelling stomatal conductance. *Global Change Biology*, 17, 2134-2144.
5. Guenther, A. B., Jiang, X., Heald, C. L., et al. (2012). The Model of Emissions of Gases and Aerosols from Nature version 2.1 (MEGAN2.1). *Geoscientific Model Development*, 5, 1471-1492.
6. Oleson, K. W., Lawrence, D. M., Bonan, G. B., et al. (2013). Technical description of version 4.5 of the Community Land Model (CLM). NCAR Technical Note NCAR/TN-503+STR.
