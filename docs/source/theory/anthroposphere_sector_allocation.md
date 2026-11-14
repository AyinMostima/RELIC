# Anthropogenic Emission Sector Allocation

## 1. Process Definition and Scope

`SectorAllocation` (`relic/scientific/anthroposphere/emissions/sector_allocation.py`) performs a pointwise mass-conservative categorical split of six pre-gridded anthropogenic species mass fluxes (`anthropogenic_nox_flux`, `anthropogenic_co_flux`, `anthropogenic_so2_flux`, `anthropogenic_voc_flux`, `anthropogenic_nh3_flux`, `anthropogenic_pm25_flux`) across five canonical economic sectors (`power`, `transport`, `industry`, `buildings`, `agriculture`) on the same horizontal grid support.

The Functor implements `ScaleCoupling` (`source_scale="inventory_grid"`, `target_scale="sector_species_grid"`, `coupling_type="hierarchical"`) during `ExecutionPhase.PREPROCESSING`, emitting `sector_emission_flux` (`[5, 6, ...spatial]`) for downstream chemical speciation by `SpeciesMapper` (`anthroposphere_emissions_mapping.md`). Coarse-to-fine spatial downscaling across distinct grid resolutions and temporal diurnal-weekly-seasonal modulation are not executed by `SectorAllocation`; `downscale()` raises `RuntimeError` and `diagnostics.sector_allocation.temporal_factors` returns unit factors. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 24 (Sections 24.1 and 24.2).

## 2. Governing Equations

### 2.1 Slice-Local Spatial Proxy Normalization

Each sector $s \in \{1, \dots, 5\} \equiv \{\mathrm{power}, \mathrm{transport}, \mathrm{industry}, \mathrm{buildings}, \mathrm{agriculture}\}$ binds to one non-negative gridded spatial proxy field $r_s(x) \ge 0$ over trailing two-dimensional horizontal axes $(-2, -1)$:

$$
(r_{\mathrm{power}}, r_{\mathrm{transport}}, r_{\mathrm{industry}}, r_{\mathrm{buildings}}, r_{\mathrm{agriculture}})
= \bigl(\rho_{\mathrm{gdp}},\; \rho_{\mathrm{road}},\; \rho_{\mathrm{gdp}},\; \rho_{\mathrm{pop}},\; f_{\mathrm{lu}}[c_{\mathrm{crop}}]\bigr)
$$

where $f_{\mathrm{lu}}[c_{\mathrm{crop}}]$ selects index $c_{\mathrm{crop}}$ (`cropland_class_index`) along the land-use class axis (`ndim - 3`) of `land_use_fractions`. For each leading batch or ensemble slice independently, let $\bar{r}_s = \operatorname{mean}_{(-2,-1)}(r_s)$ denote the two-dimensional spatial mean and $\epsilon_p > 0$ denote `proxy_floor`. The regularized dimensionless spatial proxy $P_s(x)$ is

$$
P_s(x) = \begin{cases}
\dfrac{r_s(x) / \bar{r}_s + \epsilon_p}{1 + \epsilon_p}, & \bar{r}_s > 0 \\[8pt]
0, & \bar{r}_s = 0
\end{cases}
$$

preserving slice-local unit spatial mean ($\operatorname{mean}_{(-2,-1)} P_s = 1$ whenever $\bar{r}_s > 0$) without cross-slice batch leakage.

### 2.2 Pointwise Sector-by-Species Convex Partition

Let $f_s \ge 0$ (`sector_fractions.<s>`) denote sector priors and $q_{s,k} \ge 0$ denote the sparse 12-entry sector-species affinity matrix (`SPECIES_PARAMETER_KEYS`) over species $k \in \{1, \dots, 6\} \equiv \{\mathrm{nox}, \mathrm{co}, \mathrm{so2}, \mathrm{voc}, \mathrm{nh3}, \mathrm{pm25}\}$, with $q_{s,k} = 0$ outside the twelve declared `(sector, species)` pairs (`power`: `nox, so2, pm25`; `transport`: `nox, co, pm25`; `industry`: `nox, so2, voc`; `buildings`: `nox, pm25`; `agriculture`: `nh3`). Unnormalized sector affinity $a_{s,k}(x)$ and normalized convex allocation weights $W_{s,k}(x)$ are

$$
a_{s,k}(x) = f_s\,q_{s,k}\,P_s(x), \qquad
W_{s,k}(x) = \frac{a_{s,k}(x)}{\sum_{s'=1}^{5} a_{s',k}(x)}, \qquad
\sum_{s=1}^{5} W_{s,k}(x) = 1
$$

requiring strictly positive denominator $\sum_{s'=1}^5 a_{s',k}(x) > 0$ at every grid cell (`ValueError` otherwise). Multiplying input species flux $E_k(x) \ge 0$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) by $W_{s,k}(x)$ yields allocated sector-species flux $E^{\mathrm{sec}}_{s,k}(x)$ and inverse aggregation (`upscale()`):

$$
E^{\mathrm{sec}}_{s,k}(x) = W_{s,k}(x)\,E_k(x), \qquad
\sum_{s=1}^{5} E^{\mathrm{sec}}_{s,k}(x) = E_k(x)
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $E_{\mathrm{nox}}, E_{\mathrm{co}}, E_{\mathrm{so2}}$ | Input gridded anthropogenic species mass fluxes | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `anthropogenic_nox_flux`, `anthropogenic_co_flux`, `anthropogenic_so2_flux` | `forcing` |
| $E_{\mathrm{voc}}, E_{\mathrm{nh3}}, E_{\mathrm{pm25}}$ | Input gridded anthropogenic species mass fluxes | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `anthropogenic_voc_flux`, `anthropogenic_nh3_flux`, `anthropogenic_pm25_flux` | `forcing` |
| $\rho_{\mathrm{pop}}, \rho_{\mathrm{gdp}}, \rho_{\mathrm{road}}$ | Non-negative population, GDP, and road spatial proxy densities | $\mathrm{km^{-2}}$, $\mathrm{USD\,m^{-2}}$, $\mathrm{km\,km^{-2}}$ | `population_density`, `gdp_density`, `road_density` | `forcing` |
| $f_{\mathrm{lu}}$ | Gridded land-use class fraction tensor (class axis at `ndim - 3`) | $\mathrm{1}$ | `land_use_fractions` | `forcing` |
| $E^{\mathrm{sec}}_{s,k}$ | Allocated sector-by-species emission mass flux (`[5, 6, ...spatial]`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `sector_emission_flux` | `flux` |
| $W_{s,k}, P_s, \tau_s$ | Normalized allocation weights (`[5, 6, ...spatial]`), normalized sector proxies (`[5, ...spatial]`), and unit temporal factors (`[5]`) | $\mathrm{1}$ | `diagnostics.sector_allocation.allocation_weights`, `diagnostics.sector_allocation.spatial_distribution`, `diagnostics.sector_allocation.temporal_factors` | `diagnostic` |
| $r_{\mathrm{mass}}$ | Pointwise sector-sum mass residual $\sum_s E^{\mathrm{sec}}_{s,k} - E_k$ (`[6, ...spatial]`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.sector_allocation.mass_residual` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $f_s$ | `anthroposphere.emissions.sector_fractions.power`, `anthroposphere.emissions.sector_fractions.transport`, `anthroposphere.emissions.sector_fractions.industry`, `anthroposphere.emissions.sector_fractions.buildings`, `anthroposphere.emissions.sector_fractions.agriculture` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Crippa et al. (2020); McDuffie et al. (2020) |
| $q_{\mathrm{power},k}$ | `anthroposphere.emissions.power_sector.species_fractions.nox`, `anthroposphere.emissions.power_sector.species_fractions.so2`, `anthroposphere.emissions.power_sector.species_fractions.pm25` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Crippa et al. (2020) |
| $q_{\mathrm{transport},k}$ | `anthroposphere.emissions.transport_sector.species_fractions.nox`, `anthroposphere.emissions.transport_sector.species_fractions.co`, `anthroposphere.emissions.transport_sector.species_fractions.pm25` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | McDuffie et al. (2020) |
| $q_{\mathrm{industry},k}$ | `anthroposphere.emissions.industry_sector.species_fractions.nox`, `anthroposphere.emissions.industry_sector.species_fractions.so2`, `anthroposphere.emissions.industry_sector.species_fractions.nmvoc` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Crippa et al. (2020) |
| $q_{\mathrm{buildings},k}, q_{\mathrm{agriculture},\mathrm{nh3}}$ | `anthroposphere.emissions.buildings_sector.species_fractions.nox`, `anthroposphere.emissions.buildings_sector.species_fractions.pm25`, `anthroposphere.emissions.agriculture_sector.species_fractions.nh3` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | McDuffie et al. (2020) |
| $\epsilon_p, c_{\mathrm{crop}}$ | `anthroposphere.emissions.sector_allocation.proxy_floor`, `anthroposphere.emissions.sector_allocation.cropland_class_index` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | SMOKE spatial proxy regularization |
| $\varepsilon_{\mathrm{cons}}$ | `anthroposphere.emissions.conservation_tolerance` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Relative partition tolerance |

## 5. Numerical Methods and Discretization

`SectorAllocation.forward()` requires all ten input tensors to share floating dtype (`float32` or `float64`), device, and spatial shape (`ndim >= 2`, with one extra class dimension at `ndim - 3` on `land_use_fractions`), rejecting runtime `config["parameters"]` overrides or negative values. Notice that `SectorAllocation` normalizes weights $W_{s,k}(x)$ pointwise per grid cell: spatial proxies modulate *relative* sector shares at cell $x$ only for species active in two or more sectors (`nox`, `so2`, `pm25`), whereas single-sector species (`co` in `transport`, `voc` in `industry`, `nh3` in `agriculture`) satisfy $W_{s,k}(x) \equiv 1$ everywhere regardless of proxy gradients. `sector_emission_flux` is emitted via `UpdateItem(UpdateMode.REPLACE, write_mode=StateWriteMode.DERIVED, unit="kg m-2 s-1", basis="per_second", per_timestep=False)` alongside four plain diagnostic tensors.

## 6. Conservation, Boundary Conditions, and Ledgers

Pointwise convex normalization $\sum_{s=1}^5 W_{s,k}(x) = 1$ guarantees exact per-species, per-cell mass-flux conservation $\sum_{s=1}^5 E^{\mathrm{sec}}_{s,k}(x) = E_k(x)$ up to floating-point rounding. `forward()` verifies relative error $\max \lvert r_{\mathrm{mass}} \rvert / \max(\max E_k, \epsilon_{\mathrm{tiny}}) \le \varepsilon_{\mathrm{cons}}$, and `check_conservation_laws()` enforces the same bound from `input_state` and `output_updates`, returning the relative error under `"mass"`. `upscale()` sums `sector_emission_flux` over sector axis `0` (`[5, 6, ...]`), whereas `downscale()` fails closed with `RuntimeError` because sector weights cannot be reconstructed from species totals alone without spatial proxy fields.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `SectorAllocation` | `relic/scientific/anthroposphere/emissions/sector_allocation.py` | `ScaleCoupling` (`inventory_grid` to `sector_species_grid`, `hierarchical`) | `ScaleCouplingFunctor` (`relic/core/functor/scale_coupling_bridge.py`) | `ExecutionPhase.PREPROCESSING` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `SectorAllocation`: `[I]`, IPHS **Level 1** (`scale_coupling`, certificate `anthroposphere_emissions_sector_allocation_sector_allocation.yaml`). Pointwise mass-conservative categorical partition across five sectors and six species; coarse-to-fine multi-resolution spatial disaggregation and diurnal-seasonal temporal scaling remain outside this operator (`[N]`).

Frozen certificate expressions:

```text
Historical dynamic outputs and secondary process-law scaling are retained only as migration evidence, not executable runtime.
The single runtime allocation satisfies sum_s E_s,k,x = E_k,x pointwise for each species and cell; no DEC morphism is asserted.
```

## 9. References

- CMAQ Development Team (2019). *Community Multiscale Air Quality (CMAQ) Modeling System User Guide* (SMOKE Spatial Allocation). U.S. Environmental Protection Agency.
- Crippa, M., Solazzo, E., Huang, G., et al. (2020). High resolution temporal profiles in the Emissions Database for Global Atmospheric Research. *Scientific Data*, 7(1), 121.
- McDuffie, E. E., Smith, S. J., O'Rourke, P., et al. (2020). A global anthropogenic emission inventory of atmospheric pollutants from sector- and fuel-specific sources (1970-2017): an application of the Community Emissions Data System (CEDS). *Earth System Science Data*, 12(4), 3413-3442.
