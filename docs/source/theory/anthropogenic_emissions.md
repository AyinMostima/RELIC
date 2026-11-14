# Anthropogenic Emission Inventories and Atmospheric Injection

## 1. Process Definition and Scope

`EmissionInventory` (`relic/scientific/anthroposphere/inventory.py`) and `AtmosphereCoupling` (`relic/scientific/anthroposphere/emissions/coupling_to_atmosphere.py`) define the external inventory-ingestion boundary and vertical atmospheric-injection boundary of the anthroposphere emissions subsystem (*Methods of Ecological Analysis*, Chapter 24, Sections 24.1 and 24.3).

Both boundaries are fail-closed `[N]` contracts at IPHS **Level 0**. `EmissionInventory` raises `RuntimeError` (`[N] runtime_domain_unbound`) in `__init__` and `forward` before `FunctorBase`, `ParameterManager`, calendar, state, cache, checkpoint, or output initialization because no authenticated source-domain to target-domain spatial disaggregation chain map is bound. `AtmosphereCoupling` raises `RuntimeError` (`[N] runtime_dependency_unbound`) in `__init__` and `forward` because its six canonical surface-rate ports (`anthropogenic_nox_flux`, `anthropogenic_co_flux`, `anthropogenic_so2_flux`, `anthropogenic_voc_flux`, `anthropogenic_nh3_flux`, `anthropogenic_pm25_flux`, $\mathrm{kg\,m^{-2}\,s^{-1}}$) have no executable runtime Graph writer. Executable sector disaggregation (`SectorAllocation`, `[I]`) and CB05 chemical speciation (`SpeciesMapper`, `[I]`) operate when pre-gridded `anthropogenic_*_flux` forcing fields are supplied externally in `StateManager`, as documented in `anthroposphere_sector_allocation.md` and `anthroposphere_emissions_mapping.md`.

## 2. Governing Equations

### 2.1 Conditional Domain Disaggregation Contract

For global annual emission total $E^{\mathrm{global}}_{s,k}$ ($\mathrm{kg\,yr^{-1}}$) of sector $s$ and inventory species $k$, calendar year length $Y(C)$ ($\mathrm{d\,yr^{-1}}$), seconds per day $s_{\mathrm{day}}$, target-domain capture fraction $\beta_{s,k,D} \in [0, 1]$, calendar-normalized temporal profile factor $T_{s,k}(t) \ge 0$, target-cell area $A_i > 0$ ($\mathrm{m^2}$), and domain-conditional spatial proxy weight $P_{s,k,i \mid D} \ge 0$ satisfying $\sum_{i \in D} P_{s,k,i \mid D} = 1$, the conditional cell mass flux `[N]` is:

$$
F_{s,k,i}(t) = \frac{E^{\mathrm{global}}_{s,k}\,\beta_{s,k,D}\,T_{s,k}(t)\,P_{s,k,i \mid D}}{Y(C)\,s_{\mathrm{day}}\,A_i}, \qquad
\sum_{i \in D} A_i\,F_{s,k,i}(t) = \frac{E^{\mathrm{global}}_{s,k}\,\beta_{s,k,D}\,T_{s,k}(t)}{Y(C)\,s_{\mathrm{day}}}
$$

Area cancellation over $i \in D$ recovers the domain-captured rate only when $\beta_{s,k,D}$ and $P_{s,k,i \mid D}$ are jointly bound by an authenticated global-to-regional chain map. Setting $\beta_{s,k,D} = 1$ after normalizing a local proxy over an arbitrary sub-domain inflates regional flux density inversely with sub-domain area and is rejected by `EmissionInventory`.

### 2.2 Conditional Vertical Injection Partition Contract

Given a resolved surface mass flux $F_s(x)$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$), vertical injection efficiency $\eta \in [0, 1]$, and vertical layer profile $P_z(x) \ge 0$ satisfying $\sum_z P_z(x) = 1$, the conditional atmospheric injection partition `[N]` separates three-dimensional plume injection $F^{\mathrm{inj}}_{s,z}(x)$ from residual surface flux $F^{\mathrm{res}}_s(x)$:

$$
F^{\mathrm{inj}}_{s,z}(x) = \eta\,F_s(x)\,P_z(x), \qquad
F^{\mathrm{res}}_s(x) = (1 - \eta)\,F_s(x), \qquad
\sum_z F^{\mathrm{inj}}_{s,z}(x) + F^{\mathrm{res}}_s(x) = F_s(x)
$$

Until an authenticated provider emits the six `anthropogenic_*_flux` ports and binds target writers for prospective `emission_field_3d` ($\mathrm{kg\,m^{-2}\,s^{-1}}$) and `bvoc_emission_field_3d` ($\mathrm{mol\,m^{-2}\,s^{-1}}$), `AtmosphereCoupling` exposes no executable operator.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $F_{\mathrm{NOx}}, F_{\mathrm{CO}}, F_{\mathrm{SO2}}$ | Canonical anthropogenic surface mass fluxes (external forcing for `SectorAllocation` `[I]`; unbound Graph writer `[N]`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `anthropogenic_nox_flux`, `anthropogenic_co_flux`, `anthropogenic_so2_flux` | `forcing` / `flux` |
| $F_{\mathrm{VOC}}, F_{\mathrm{NH3}}, F_{\mathrm{PM25}}$ | Canonical anthropogenic surface mass fluxes (external forcing for `SectorAllocation` `[I]`; unbound Graph writer `[N]`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `anthropogenic_voc_flux`, `anthropogenic_nh3_flux`, `anthropogenic_pm25_flux` | `forcing` / `flux` |
| $F^{\mathrm{inj}}_{s,z}$ | Prospective three-dimensional anthropogenic mass-flux field (`[N]`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `emission_field_3d` | `flux` |
| $F^{\mathrm{bvoc}}_{g,z}$ | Prospective three-dimensional biogenic VOC molar-flux field (`[N]`) | $\mathrm{mol\,m^{-2}\,s^{-1}}$ | `bvoc_emission_field_3d` | `flux` |
| $E^{\mathrm{sec}}_{s,k}$ | Sector-by-species mass flux produced by `SectorAllocation` and consumed by `SpeciesMapper` (`[I]`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `sector_emission_flux` | `flux` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $T_{s,k}$ | `anthroposphere.emissions.inventory.temporal_profiles.*` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Crippa et al. (2018); Olivier et al. (2005) |
| $H_{\mathrm{mix}}, \Delta z_{\mathrm{inj}}$ | `anthroposphere.emissions.vertical_distribution.mixing_height`, `anthroposphere.emissions.vertical_distribution.layer_thickness` | $\mathrm{m}$ | `anthroposphere/emissions_parameters.yaml` | Vertical plume-layer specification (`[N]` at runtime) |
| $u_{\mathrm{em}}$ | `anthroposphere.emissions.uncertainty.factor` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Hoesly et al. (2018) |
| $s_{\mathrm{day}}$ | `relic.physical.constants.time.seconds_per_day` | $\mathrm{s\,d^{-1}}$ | `physical/constants.yaml` | SI calendar conversion ($86400$) |

## 5. Numerical Methods and Discretization

`EmissionInventory` and `AtmosphereCoupling` enforce deterministic fail-closed `RuntimeError` guards on both `__init__` and `forward`. Neither class allocates `torch.Tensor` buffers, queries `ParameterManager` at runtime, submits `UpdateItem` entries to `StateManager`, or mutates replay checkpoints. Active gridded emission pipelines enter at `SectorAllocation` (externally supplied `anthropogenic_*_flux`) and `SpeciesMapper` (`sector_emission_flux`).

## 6. Conservation, Boundary Conditions, and Ledgers

Both Level-0 contracts prevent unverified local algebraic identities from masquerading as open-system mass conservation. `EmissionInventory` rejects local proxy normalization without an authenticated global-to-domain fraction $\beta_{s,k,D}$, and `AtmosphereCoupling` rejects partition evaluation on unbound `anthropogenic_*_flux` providers. Active mass-conservation ledgers in the emissions chain are enforced by `SectorAllocation` and `SpeciesMapper`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `EmissionInventory` | `relic/scientific/anthroposphere/inventory.py` | `ScaleCoupling` | `ScaleCouplingFunctor` (`relic/core/functor/scale_coupling_bridge.py`) | Fail-closed (`[N]`, unbound domain) |
| `AtmosphereCoupling` | `relic/scientific/anthroposphere/emissions/coupling_to_atmosphere.py` | `TransportMechanism` (`TRANSPORT_TYPE="gas_exchange"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | Fail-closed (`[N]`, unbound providers) |

## 8. Implementation Status

- `EmissionInventory`: `[N]`, IPHS **Level 0** (`scale_coupling`, certificate `anthroposphere_inventory_emission_inventory.yaml`). Fail-closed (`runtime_domain_unbound`) pending an authenticated global-to-target spatial disaggregation chain map.
- `AtmosphereCoupling`: `[N]`, IPHS **Level 0** (`transport`, certificate `anthroposphere_emissions_coupling_to_atmosphere_atmosphere_coupling.yaml`). Fail-closed (`runtime_dependency_unbound`) pending executable providers for the six canonical `anthropogenic_*_flux` ports.

Frozen certificate expressions:

```text
The rejected path implicitly set every target-domain fraction to one after normalizing a local proxy.
[N] runtime_domain_unbound; no executable scientific expression is exposed.
The rejected direct-forward path proved only a local partition identity and did not prove provider resolution or governed execution.
[N] runtime_dependency_unbound; no executable scientific expression is exposed.
```

## 9. References

- Crippa, M., Guizzardi, D., Muntean, M., et al. (2018). Gridded emissions of air pollutants for the period 1970-2012 within EDGAR v4.3.2. *Earth System Science Data*, 10(4), 1987-2013.
- Hoesly, R. M., Smith, S. J., Feng, L., et al. (2018). Historical (1750-2014) anthropogenic emissions of reactive gases and aerosols from the Community Emissions Data System (CEDS). *Geoscientific Model Development*, 11(1), 369-408.
- Janssens-Maenhout, G., Crippa, M., Guizzardi, D., et al. (2015). HTAP_v2.2: a mosaic of regional and global emission grid maps for 2008 and 2010 to study hemispheric transport of air pollution. *Atmospheric Chemistry and Physics*, 15(19), 11411-11432.
- Olivier, J. G. J., Van Aardenne, J. A., Dentener, F. J., Pagliari, V., Ganzeveld, L. N., and Peters, J. A. H. W. (2005). Recent trends in global greenhouse gas emissions: regional trends 1970-2000 and spatial distribution of key sources in 2000. *Environmental Sciences*, 2(2-3), 81-99.
