# Anthroposphere Scientific Classes

This reference covers all 14 certificate-backed scientific classes in `relic.scientific.anthroposphere`: 11 executable concrete Functors, 3 fail-closed unavailable boundaries, and 0 abstract contracts. Each entry is generated from the live class, its IPHS certificate, State Contract Model, and ParameterManager metadata. Fail-closed unavailable classes are not instantiated during documentation generation.

The tables distinguish constructor controls, runtime variables, and ParameterManager scientific parameters. Values shown as package defaults belong to the documented RELIC architecture version and must not be copied as hardcoded constants into scientific code.

Certificate wildcard parameter groups (`*`, `<pft>`) are expanded to their exact current YAML entries. Every other declared parameter is matched to the parameter database by its exact full key; a declared name is never mapped to a key that merely ends with the same name.

Exact ParameterManager rows report the live YAML value, bounds, gradient policy, description, and provenance. A row whose value reads `not declared in the parameter database` is intentionally not assigned an invented default: the class metadata or certificate declares the quantity, with the meaning and units it states, but the parameter database has no key of that exact name.

IPHS levels are read from the sole certificate owner. Levels 1 to 3 describe compatibility, wrapping, or declared structure; they are not strict IPHS/GENERIC claims.

The Channel column of each runtime output contract applies `relic.core.state.types.partition_outputs`, the only runtime output channel classifier, to the declared output name. A name that the classifier reserves for diagnostics, such as `diagnostics.*`, goes to the diagnostics channel and is never committed as state. For any other name the channel is decided at run time by the write mode of the returned value: `StateWriteMode.DERIVED` routes the output to the diagnostics channel, and any other write mode sends it to the StateManager commit (a value without a write mode takes the run's semantic-mode rules or the StateManager schema). The generator does not run `forward()`, so it does not predict that write mode. The Kind column is the State Contract Model kind of the variable and does not set its channel.

(class-relic-scientific-anthroposphere-emissions-coupling-to-atmosphere-atmospherecoupling)=
## atmosphere_coupling

Canonical coupling node whose governed scientific runtime is unbound.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.emissions.coupling_to_atmosphere.AtmosphereCoupling` |
| Source | `relic\scientific\anthroposphere\emissions\coupling_to_atmosphere.py:41` |
| Runtime name | `not_applicable_runtime_unbound` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `runtime_unbound` |
| Variable kind | `runtime_unbound` |
| IPHS level | `level_0` |
| Implementation status | `not_implemented_fail_closed` |
| Class status | `fail-closed unavailable boundary` |

:::{warning}
This class is `[N] runtime_dependency_unbound`. Construction and `forward(...)` reject before state access or side effects. Empty runtime input, output, and port declarations below are the documented fail-closed boundary, not missing documentation.
:::

### Constructor and argument meanings

```python
atmosphere_coupling(*args: Any, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `args` | `Any` | `required` | Additional positional arguments accepted by the inherited constructor. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |

### Governing equations and canonical operator

**Literature or process form**

Atmospheric models may vertically allocate prescribed surface emissions only after every required rate has a provenance-bound provider in the same runtime graph.

**Canonical RELIC structural form**

No IPHS formulation is claimed while the required anthropogenic provider dependency is unavailable.

| Entry | Meaning |
| --- | --- |
| `runtime_status` | no executable vertical source operator is registered |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | not_applicable_runtime_dependency_unbound |
| `effort_variable` | not_applicable_runtime_dependency_unbound |
| `state_advancer` | none_runtime_dependency_unbound |
| `runtime_usage` | no state proposal or mutation is produced |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
No executable dependency or composition path is declared while this capability remains fail-closed.

### Scientific parameter and numeric binding contract

This fail-closed class does not resolve or execute scientific parameters.

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `constructor` | rejects before FunctorBase, ParameterManager, metadata, cache, or state effects |
| `forward` | independently rejects unsafe constructor bypass before state access |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthropogenic_emissions.md`.

### References

- CMAQ User Guide, emissions source terms and vertical allocation. (certificate)
- Seinfeld, J.H. and Pandis, S.N. (2016). Atmospheric Chemistry and Physics. (certificate)
- Emmons, L.K. et al. (2010). Description and evaluation of MOZART-4. (certificate)

(class-relic-scientific-anthroposphere-emissions-sector-allocation-sectorallocation)=
## sector_allocation

Split canonical inventory fluxes over a fixed sector/species tensor.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.emissions.sector_allocation.SectorAllocation` |
| Source | `relic\scientific\anthroposphere\emissions\sector_allocation.py:50` |
| Runtime name | `sector_allocation` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
sector_allocation(name: str = 'sector_allocation', config: dict[str, Any] | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'sector_allocation'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>dict[str, Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `anthropogenic_nox_flux` | flux | `kg m-2 s-1` | NOx inventory mass flux [kg m-2 s-1] |
| `anthropogenic_co_flux` | flux | `kg m-2 s-1` | CO inventory mass flux [kg m-2 s-1] |
| `anthropogenic_so2_flux` | flux | `kg m-2 s-1` | SO2 inventory mass flux [kg m-2 s-1] |
| `anthropogenic_voc_flux` | flux | `kg m-2 s-1` | VOC inventory mass flux [kg m-2 s-1] |
| `anthropogenic_nh3_flux` | flux | `kg m-2 s-1` | NH3 inventory mass flux [kg m-2 s-1] |
| `anthropogenic_pm25_flux` | flux | `kg m-2 s-1` | PM25 inventory mass flux [kg m-2 s-1] |
| `population_density` | forcing | `1` | population proxy field |
| `gdp_density` | forcing | `1` | GDP proxy field |
| `road_density` | forcing | `1` | road proxy field |
| `land_use_fractions` | forcing | `1` | land-use class fraction field |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `sector_emission_flux` | diagnostic | `kg m-2 s-1` | decided at run time by the returned write mode | fixed [sector, species, source_shape] mass-flux tensor [kg m-2 s-1] |
| `diagnostics.sector_allocation.allocation_weights` | output port | `1` | diagnostics | pointwise sector weights [1] |
| `diagnostics.sector_allocation.spatial_distribution` | diagnostic | `dimensionless` | diagnostics | spatial-mean-normalized sector proxies [1] |
| `diagnostics.sector_allocation.temporal_factors` | diagnostic | `dimensionless` | diagnostics | unity factors confirming upstream temporal ownership [1] |
| `diagnostics.sector_allocation.mass_residual` | output port | `kg m-2 s-1` | diagnostics | input minus sector-summed output by species and cell [kg m-2 s-1] |

### Governing equations and canonical operator

**Literature or process form**

Emissions preprocessing splits already computed inventory mass fluxes using declared sector priors and spatial surrogates while preserving species totals.

**Canonical RELIC structural form**

Level 1 compatibility records the executed allocation $E_{s,k,x} = W_{s,k,x} E_{k,x}$ with $W_{s,k,x} \geq 0$ and $\sum_{s} W_{s,k,x} = 1$; no IPHS operator is claimed.

| Entry | Meaning |
| --- | --- |
| `allocation_rule` | forward evaluates the pointwise non-negative W_sector_species_proxy and normalizes it only along the sector axis |
| `sector_aggregation` | upscale sums the governed sector axis without constructing a second allocation operator |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | not_applicable_algebraic_flux_partition_without_hamiltonian_potential |
| `effort_variable` | not_applicable_no_energy_or_entropy_effort |
| `state_advancer` | $E_{\mathrm{sector}} = W_{\mathrm{sector},\mathrm{species},\mathrm{proxy}} E_{\mathrm{input}}$ |
| `runtime_usage` | forward replaces the current sector-resolved rate map and emits the exact pointwise residual. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.emissions.sector_fractions.power` | `dimensionless` | `0.25; bounds=(0.1, 0.5)` | `fixed` | Default fraction for power sector | Source: anthroposphere_emissions_database; IPCC 2019 |
| `anthroposphere.emissions.sector_fractions.transport` | `dimensionless` | `0.25; bounds=(0.1, 0.4)` | `fixed` | Default fraction for transport sector | Source: anthroposphere_emissions_database; IPCC 2019 |
| `anthroposphere.emissions.sector_fractions.industry` | `dimensionless` | `0.2; bounds=(0.1, 0.4)` | `fixed` | Default fraction for industry sector | Source: anthroposphere_emissions_database; IPCC 2019 |
| `anthroposphere.emissions.sector_fractions.buildings` | `dimensionless` | `0.15; bounds=(0.05, 0.3)` | `fixed` | Default fraction for buildings sector | Source: anthroposphere_emissions_database; IPCC 2019 |
| `anthroposphere.emissions.sector_fractions.agriculture` | `dimensionless` | `0.1; bounds=(0.05, 0.25)` | `fixed` | Default fraction for agriculture sector | Source: anthroposphere_emissions_database; IPCC 2019 |
| `anthroposphere.emissions.power_sector.species_fractions.nox` | `dimensionless` | `0.75; bounds=(0.0, 1.0)` | `fixed` | Fraction of NOx mass within power-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.power_sector.species_fractions.so2` | `dimensionless` | `0.85; bounds=(0.0, 1.0)` | `fixed` | Fraction of SO2 mass within power-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.power_sector.species_fractions.pm25` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Fraction of PM2.5 mass within power-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.transport_sector.species_fractions.nox` | `dimensionless` | `0.9; bounds=(0.0, 1.0)` | `fixed` | Fraction of NOx mass for transport-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.transport_sector.species_fractions.co` | `dimensionless` | `0.95; bounds=(0.0, 1.0)` | `fixed` | Fraction of CO mass for transport-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.transport_sector.species_fractions.pm25` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Fraction of PM2.5 mass for transport-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.industry_sector.species_fractions.nox` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Fraction of NOx mass for industry-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.industry_sector.species_fractions.so2` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Fraction of SO2 mass for industry-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.industry_sector.species_fractions.nmvoc` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Fraction of NMVOC mass for industry-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.buildings_sector.species_fractions.nox` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Fraction of NOx mass for buildings-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.buildings_sector.species_fractions.pm25` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Fraction of PM2.5 mass for buildings-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.agriculture_sector.species_fractions.nh3` | `dimensionless` | `0.95; bounds=(0.0, 1.0)` | `fixed` | Fraction of NH3 mass for agriculture-sector emissions | Source: anthroposphere_emissions_database; IPCC 2019 Refinement |
| `anthroposphere.emissions.sector_allocation.proxy_floor` | `dimensionless` | `1e-06; bounds=(1e-12, 0.01)` | `fixed` | Positive numerical support applied after proxy normalization to keep the sector weight denominator defined. | Source: RELIC sector allocation numerical policy; CMAQ User Guide (2019), SMOKE spatial surrogate quality assurance |
| `anthroposphere.emissions.sector_allocation.cropland_class_index` | `dimensionless` | `2; bounds=(0.0, 20.0)` | `fixed` | Zero-based land-use class index used as the agriculture allocation proxy. | Source: RELIC land-use mapping convention; RELIC land-use class ordering contract |
| `anthroposphere.emissions.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-09, 0.0001)` | `fixed` | Tolerance for sector allocation conservation checks | Source: anthroposphere_emissions_database; Seinfeld and Pandis (2016) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | sector_emission_flux uses UpdateItem(REPLACE, DERIVED, per_second, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} )$; diagnostics are observation payloads |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_sector_allocation.md`.

### References

- Crippa et al. (2020) DOI:10.1038/s41597-020-0462-2 (FunctorMetadata)
- McDuffie et al. (2020) DOI:10.5194/essd-12-3413-2020 (FunctorMetadata)
- CMAQ User Guide (2019), SMOKE spatial allocation (FunctorMetadata)
- Pielke, R.A. (2002). Mesoscale meteorological modeling. (FunctorMetadata)
- Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming. (FunctorMetadata)
- CMAQ User Guide v5.3 (EPA, 2019). (certificate)
- SMOKE Emissions Modeling System documentation. (certificate)
- Crippa, M. et al. (2020). High resolution temporal profiles in EDGAR. (certificate)
- McDuffie, E.E. et al. (2020). Global anthropogenic emission inventory. (certificate)

(class-relic-scientific-anthroposphere-emissions-species-mapping-speciesmapper)=
## species_mapping

Maps emission inventory species to chemical mechanism species and provides unit conversion utilities.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.emissions.species_mapping.SpeciesMapper` |
| Source | `relic\scientific\anthroposphere\emissions\species_mapping.py:50` |
| Runtime name | `species_mapping` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
species_mapping(name: str = 'species_mapping', config: Dict[str, Any] | None = None, **kwargs)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'species_mapping'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `sector_emission_flux` | diagnostic | `kg m-2 s-1` | fixed [power, transport, industry, buildings, agriculture] by [NOx, CO, SO2, VOC, NH3, PM25] mass-flux tensor [kg m-2 s-1] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `mechanism_gas_molar_flux` | flux | `mol m-2 s-1` | decided at run time by the returned write mode | fixed [NO, NO2, CO, SO2, NH3, PAR, OLE, TOL, XYL, FORM, ALD2] gas flux tensor [mol m-2 s-1] |
| `mechanism_aerosol_mass_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | fixed [BC, OC, NO3, SO4, NH4, OTHER] aerosol flux tensor [kg m-2 s-1] |
| `diagnostics.species_mapping.total_input_mass_flux` | output port | `kg m-2 s-1` | diagnostics | total sector/species input mass flux [kg m-2 s-1] |
| `diagnostics.species_mapping.total_output_mass_flux` | output port | `kg m-2 s-1` | diagnostics | reconstructed true gas plus aerosol output mass flux [kg m-2 s-1] |
| `diagnostics.species_mapping.inventory_equivalent_output_mass_flux` | output port | `kg m-2 s-1` | diagnostics | output converted to the inventory NO2-equivalent basis [kg m-2 s-1] |
| `diagnostics.species_mapping.inventory_basis_adjustment_flux` | output port | `kg m-2 s-1` | diagnostics | true output minus inventory-equivalent output [kg m-2 s-1] |
| `diagnostics.species_mapping.relative_error` | output port | `1` | diagnostics | inventory-equivalent conservation residual [1] |
| `diagnostics.species_mapping.canonical_iphs_runtime` | output port | `1` | diagnostics | marker for the single typed chain-map runtime [1] |

### Governing equations and canonical operator

**Literature or process form**

CMAQ/SMOKE-style emissions preprocessing applies sector-specific speciation before chemistry transport and keeps gas molar flux separate from aerosol mass flux.

**Canonical RELIC structural form**

$$
F_{\mathrm{gas},\mathrm{mol}} = \star_{\mathrm{MW},\mathrm{inv}} S_{\mathrm{gas}} F_{\mathrm{sector}}; F_{\mathrm{aer},\mathrm{mass}} = S_{\mathrm{aer}} F_{\mathrm{sector}} .
$$

| Entry | Meaning |
| --- | --- |
| `S_gas` | forward applies sector-specific VOC and NOx split operators to fixed gas axes |
| `S_aer` | forward applies sector-specific PM composition to the fixed aerosol axis |
| `star_MW_inv` | build_hodge_contract positive molecular-weight inverse Hodge |
| `O_h` | numeric conservation and inventory-basis diagnostics returned by forward |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | sector-resolved inventory mass-flux cochain $u_{\mathrm{sector}}$ |
| `effort_variable` | fixed CB05 maps $S_{\mathrm{gas}}$ and $S_{\mathrm{aer}}$ plus molecular Hodge star_MW_inv |
| `state_advancer` | $F_{\mathrm{gas},\mathrm{mol}} = \star_{\mathrm{MW},\mathrm{inv}} S_{\mathrm{gas}} u_{\mathrm{sector}}$; $F_{\mathrm{aer},\mathrm{mass}} = S_{\mathrm{aer}} u_{\mathrm{sector}}$ |
| `runtime_usage` | forward evaluates these fixed typed outputs and returns REPLACE UpdateItems. |

### Composition channels and executable PortSpec API

Certificate composition channels: `species_mapping_gas_molar_flux`, `species_mapping_aerosol_mass_flux`

Executable `ports()` owner: `SpeciesMapper`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `species_mapping_gas_molar_flux` | `chemical_species_potential [J mol-1]` | `mechanism_gas_molar_flux [mol m-2 s-1]` | `W m-2` | into_system; matter |
| `species_mapping_aerosol_mass_flux` | `aerosol_mass_specific_potential [J kg-1]` | `mechanism_aerosol_mass_flux [kg m-2 s-1]` | `W m-2` | into_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.emissions.species_mapping.default_mechanism` | `dimensionless` | `'cb05'; bounds=None` | `fixed` | Default atmospheric chemistry mechanism used when none is specified | Source: anthropogenic_temporal_profiles; Yarwood et al. (2005) Updates to the CB05 chemical mechanism |
| `anthroposphere.emissions.species_mapping.mass_balance_tolerance` | `dimensionless` | `1e-06; bounds=(1e-09, 0.0001)` | `fixed` | Tolerance for mass balance checks during speciation | Source: anthropogenic_temporal_profiles; CMAQ Technical Documentation |
| `anthroposphere.emissions.partition_factors.no_fraction` | `dimensionless` | `0.9; bounds=(0.5, 1.0)` | `fixed` | Fraction of NOx NO2-equivalent mass assigned to emitted NO | Source: anthroposphere_emissions_database; US EPA emissions modeling guidance for NOx speciation |
| `anthroposphere.emissions.molecular_weights.NO` | `g mol-1` | `30.006; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.molecular_weights.NO2` | `g mol-1` | `46.005; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.molecular_weights.NOx` | `g mol-1` | `46.005; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.molecular_weights.SO2` | `g mol-1` | `64.066; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.molecular_weights.NH3` | `g mol-1` | `17.031; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.molecular_weights.PAR` | `g mol-1` | `14.0; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.molecular_weights.OLE` | `g mol-1` | `28.0; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.molecular_weights.TOL` | `g mol-1` | `92.14; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.molecular_weights.XYL` | `g mol-1` | `106.17; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.molecular_weights.ALD2` | `g mol-1` | `44.053; bounds=(1.0, inf)` | `fixed` | Molecular weight specification. A molar mass is positive and no molecular species is lighter than a hydrogen atom (1.008 g mol-1), so the lower bound is 1 g mol-1; lumped and particulate species have no upper limit, so the upper bound is open (inf). | Source: anthroposphere_emissions_database; IUPAC 2021 |
| `anthroposphere.emissions.speciation.voc_split_fractions.power_plant.PAR` | `dimensionless` | `0.4; bounds=(0.0, 1.0)` | `fixed` | Fraction of power plant VOC emissions allocated to PAR | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.power_plant.OLE` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of power plant VOC emissions allocated to OLE | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.power_plant.TOL` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Fraction of power plant VOC emissions allocated to TOL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.power_plant.XYL` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of power plant VOC emissions allocated to XYL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.power_plant.FORM` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of power plant VOC emissions allocated to FORM | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.power_plant.ALD2` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of power plant VOC emissions allocated to ALD2 | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.transport.PAR` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Fraction of transport VOC emissions allocated to PAR | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.transport.OLE` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Fraction of transport VOC emissions allocated to OLE | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.transport.TOL` | `dimensionless` | `0.15; bounds=(0.0, 1.0)` | `fixed` | Fraction of transport VOC emissions allocated to TOL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.transport.XYL` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of transport VOC emissions allocated to XYL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.transport.FORM` | `dimensionless` | `0.15; bounds=(0.0, 1.0)` | `fixed` | Fraction of transport VOC emissions allocated to FORM | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.transport.ALD2` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of transport VOC emissions allocated to ALD2 | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.industrial.PAR` | `dimensionless` | `0.35; bounds=(0.0, 1.0)` | `fixed` | Fraction of industrial VOC emissions allocated to PAR | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.industrial.OLE` | `dimensionless` | `0.15; bounds=(0.0, 1.0)` | `fixed` | Fraction of industrial VOC emissions allocated to OLE | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.industrial.TOL` | `dimensionless` | `0.25; bounds=(0.0, 1.0)` | `fixed` | Fraction of industrial VOC emissions allocated to TOL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.industrial.XYL` | `dimensionless` | `0.15; bounds=(0.0, 1.0)` | `fixed` | Fraction of industrial VOC emissions allocated to XYL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.industrial.FORM` | `dimensionless` | `0.05; bounds=(0.0, 1.0)` | `fixed` | Fraction of industrial VOC emissions allocated to FORM | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.industrial.ALD2` | `dimensionless` | `0.05; bounds=(0.0, 1.0)` | `fixed` | Fraction of industrial VOC emissions allocated to ALD2 | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.residential.PAR` | `dimensionless` | `0.45; bounds=(0.0, 1.0)` | `fixed` | Fraction of residential VOC emissions allocated to PAR | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.residential.OLE` | `dimensionless` | `0.05; bounds=(0.0, 1.0)` | `fixed` | Fraction of residential VOC emissions allocated to OLE | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.residential.TOL` | `dimensionless` | `0.15; bounds=(0.0, 1.0)` | `fixed` | Fraction of residential VOC emissions allocated to TOL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.residential.XYL` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of residential VOC emissions allocated to XYL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.residential.FORM` | `dimensionless` | `0.15; bounds=(0.0, 1.0)` | `fixed` | Fraction of residential VOC emissions allocated to FORM | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.residential.ALD2` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of residential VOC emissions allocated to ALD2 | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1 |
| `anthroposphere.emissions.speciation.voc_split_fractions.agriculture.PAR` | `dimensionless` | `0.38; bounds=(0.0, 1.0)` | `fixed` | Fraction of agricultural VOC emissions allocated to PAR | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1; IPCC (2019) Refinement to the 2006 Guidelines |
| `anthroposphere.emissions.speciation.voc_split_fractions.agriculture.OLE` | `dimensionless` | `0.07; bounds=(0.0, 1.0)` | `fixed` | Fraction of agricultural VOC emissions allocated to OLE | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1; IPCC (2019) Refinement to the 2006 Guidelines |
| `anthroposphere.emissions.speciation.voc_split_fractions.agriculture.TOL` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of agricultural VOC emissions allocated to TOL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1; IPCC (2019) Refinement to the 2006 Guidelines |
| `anthroposphere.emissions.speciation.voc_split_fractions.agriculture.XYL` | `dimensionless` | `0.05; bounds=(0.0, 1.0)` | `fixed` | Fraction of agricultural VOC emissions allocated to XYL | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1; IPCC (2019) Refinement to the 2006 Guidelines |
| `anthroposphere.emissions.speciation.voc_split_fractions.agriculture.FORM` | `dimensionless` | `0.25; bounds=(0.0, 1.0)` | `fixed` | Fraction of agricultural VOC emissions allocated to FORM | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1; IPCC (2019) Refinement to the 2006 Guidelines |
| `anthroposphere.emissions.speciation.voc_split_fractions.agriculture.ALD2` | `dimensionless` | `0.15; bounds=(0.0, 1.0)` | `fixed` | Fraction of agricultural VOC emissions allocated to ALD2 | Source: anthroposphere_emissions_database; EPA SPECIATE Database v5.1; IPCC (2019) Refinement to the 2006 Guidelines |
| `anthroposphere.emissions.pm_composition.power.bc_fraction` | `dimensionless` | `0.05; bounds=(0.0, 0.2)` | `fixed` | Power-generation BC fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.power.oc_fraction` | `dimensionless` | `0.05; bounds=(0.0, 0.2)` | `fixed` | Power-generation OC fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.power.nitrate_fraction` | `dimensionless` | `0.1; bounds=(0.0, 0.3)` | `fixed` | Power-generation nitrate fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.power.sulfate_fraction` | `dimensionless` | `0.3; bounds=(0.1, 0.6)` | `fixed` | Power-generation sulfate fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.power.ammonium_fraction` | `dimensionless` | `0.05; bounds=(0.0, 0.2)` | `fixed` | Power-generation ammonium fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.transport.bc_fraction` | `dimensionless` | `0.25; bounds=(0.1, 0.4)` | `fixed` | Transport-sector black carbon fraction | Source: literature_synthesis; EPA MOVES2014 |
| `anthroposphere.emissions.pm_composition.transport.oc_fraction` | `dimensionless` | `0.3; bounds=(0.1, 0.5)` | `fixed` | Transport-sector organic carbon fraction | Source: literature_synthesis; EPA MOVES2014 |
| `anthroposphere.emissions.pm_composition.transport.nitrate_fraction` | `dimensionless` | `0.15; bounds=(0.0, 0.4)` | `fixed` | Transport-sector nitrate fraction | Source: literature_synthesis; EPA MOVES2014 |
| `anthroposphere.emissions.pm_composition.transport.sulfate_fraction` | `dimensionless` | `0.1; bounds=(0.0, 0.3)` | `fixed` | Transport-sector sulfate fraction | Source: literature_synthesis; EPA MOVES2014 |
| `anthroposphere.emissions.pm_composition.transport.ammonium_fraction` | `dimensionless` | `0.05; bounds=(0.0, 0.2)` | `fixed` | Transport-sector ammonium fraction | Source: literature_synthesis; EPA MOVES2014 |
| `anthroposphere.emissions.pm_composition.industrial.bc_fraction` | `dimensionless` | `0.12; bounds=(0.0, 0.3)` | `fixed` | Industrial black carbon fraction of primary PM2.5 | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.industrial.oc_fraction` | `dimensionless` | `0.18; bounds=(0.0, 0.4)` | `fixed` | Industrial organic carbon fraction of primary PM2.5 | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.industrial.nitrate_fraction` | `dimensionless` | `0.15; bounds=(0.0, 0.4)` | `fixed` | Industrial nitrate fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.industrial.sulfate_fraction` | `dimensionless` | `0.25; bounds=(0.0, 0.5)` | `fixed` | Industrial sulfate fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.industrial.ammonium_fraction` | `dimensionless` | `0.05; bounds=(0.0, 0.2)` | `fixed` | Industrial ammonium fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.residential.bc_fraction` | `dimensionless` | `0.2; bounds=(0.0, 0.5)` | `fixed` | Residential black carbon fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.residential.oc_fraction` | `dimensionless` | `0.4; bounds=(0.0, 0.7)` | `fixed` | Residential organic carbon fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.residential.nitrate_fraction` | `dimensionless` | `0.1; bounds=(0.0, 0.3)` | `fixed` | Residential nitrate fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.residential.sulfate_fraction` | `dimensionless` | `0.1; bounds=(0.0, 0.3)` | `fixed` | Residential sulfate fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.residential.ammonium_fraction` | `dimensionless` | `0.05; bounds=(0.0, 0.3)` | `fixed` | Residential ammonium fraction | Source: literature_synthesis; EPA AP-42 (2019) |
| `anthroposphere.emissions.pm_composition.agriculture.bc_fraction` | `dimensionless` | `0.05; bounds=(0.0, 0.2)` | `fixed` | Agriculture-sector BC fraction | Source: literature_synthesis; FAO (2018) |
| `anthroposphere.emissions.pm_composition.agriculture.oc_fraction` | `dimensionless` | `0.3; bounds=(0.0, 0.6)` | `fixed` | Agriculture-sector OC fraction | Source: literature_synthesis; FAO (2018) |
| `anthroposphere.emissions.pm_composition.agriculture.nitrate_fraction` | `dimensionless` | `0.2; bounds=(0.0, 0.5)` | `fixed` | Agriculture-sector nitrate fraction | Source: literature_synthesis; FAO (2018) |
| `anthroposphere.emissions.pm_composition.agriculture.sulfate_fraction` | `dimensionless` | `0.15; bounds=(0.0, 0.4)` | `fixed` | Agriculture-sector sulfate fraction | Source: literature_synthesis; FAO (2018) |
| `anthroposphere.emissions.pm_composition.agriculture.ammonium_fraction` | `dimensionless` | `0.05; bounds=(0.0, 0.3)` | `fixed` | Agriculture-sector ammonium fraction | Source: literature_synthesis; FAO (2018) |

### Physical constants (SSOT)

These values are resolved from the physical-constants database. They are not domain parameters and must not be duplicated in scientific YAML or Python code.

| Constant key | Units | Value and fixed bounds | Gradient | Meaning | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `relic.physical.constants.molecular_weights.co` | `g mol-1` | `28.01; bounds=(28.01, 28.01)` | `fixed` | Conventional bulk molar mass of carbon monoxide (CO), the sum of the CIAAW 2024 abridged standard atomic weights C = 12.011 and O = 15.999; a fixed model constant, not an isotopologue-specific mass; the fixed bounds select this convention and do not assert zero natural isotopic variability | Source: CIAAW_abridged_standard_atomic_weights_2024; CIAAW (2024) Abridged Standard Atomic Weights, https://ciaaw.org/abridged-atomic-weights.htm; CIAAW Carbon, standard atomic weight interval [12.0096, 12.0116], https://ciaaw.org/carbon.htm |
| `relic.physical.constants.molecular_weights.hcho` | `g mol-1` | `30.026; bounds=(30.026, 30.026)` | `fixed` | Conventional bulk molar mass of formaldehyde (CH2O), using the NIST Chemistry WebBook SRD 69 molecular weight 30.0260 based on average atomic masses; a fixed reporting convention, not an isotopologue-specific mass or an isotopic uncertainty model | Source: NIST_Chemistry_WebBook_SRD_69; NIST Chemistry WebBook, SRD 69, Formaldehyde, CAS 50-00-0, formula CH2O, molecular weight 30.0260, https://webbook.nist.gov/cgi/cbook.cgi?ID=50000; NIST Chemistry WebBook molecular-weight documentation, https://webbook.nist.gov/chemistry/mw-ser/ |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | physical ports are UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} ) \mathrm{values} \mathrm{returned}$ through the existing FunctorBase/StateManager path |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_emissions_mapping.md`.

### References

- Yarwood, G. et al. (2005). Updates to the CB05 chemical mechanism (FunctorMetadata)
- Kuenen, J. et al. (2014). TNO-MACC_II emission inventory (FunctorMetadata)
- IPCC (2019). Refinement to the 2006 IPCC Guidelines (FunctorMetadata)
- Pielke, R.A. (2002). Mesoscale meteorological modeling. (FunctorMetadata)
- Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming. (FunctorMetadata)
- CMAQ User's Guide v5.3 (EPA, 2019). (certificate)
- Yarwood, G. et al. (2005). Updates to the CB05 chemical mechanism. (certificate)
- Kuenen, J. et al. (2014). TNO-MACC_II emission inventory. (certificate)
- Seinfeld, J.H. and Pandis, S.N. (2016). Atmospheric Chemistry and Physics. (certificate)

(class-relic-scientific-anthroposphere-interface-mapping-anthroposphereinterfacemapping)=
## anthroposphere_interface_mapping

Canonical interface node whose generic metadata runtime is unbound.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.interface_mapping.AnthroposphereInterfaceMapping` |
| Source | `relic\scientific\anthroposphere\interface_mapping.py:47` |
| Runtime name | `not_applicable_runtime_unbound` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `runtime_unbound` |
| Variable kind | `runtime_unbound` |
| IPHS level | `level_0` |
| Implementation status | `not_implemented_fail_closed` |
| Class status | `fail-closed unavailable boundary` |

:::{warning}
This class is `[N] runtime_metadata_domain_unbound`. Construction and `forward(...)` reject before state access or side effects. Empty runtime input, output, and port declarations below are the documented fail-closed boundary, not missing documentation.
:::

### Constructor and argument meanings

```python
anthroposphere_interface_mapping(*args: Any, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `args` | `Any` | `required` | Additional positional arguments accepted by the inherited constructor. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |

### Governing equations and canonical operator

**Literature or process form**

CF metadata must describe the actual variable units and dimensions; equivalent units may be retained, while unverified relabeling and invented coordinate axes are invalid.

**Canonical RELIC structural form**

No IPHS formulation is claimed while the exact anthroposphere metadata domain and governed application path are unavailable.

| Entry | Meaning |
| --- | --- |
| `runtime_status` | no executable metadata chain-map or observation operator is registered |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | not_applicable_runtime_metadata_domain_unbound |
| `effort_variable` | not_applicable_runtime_metadata_domain_unbound |
| `state_advancer` | none_runtime_metadata_domain_unbound |
| `runtime_usage` | no UpdateItem, diagnostic ledger, or state mutation is produced |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
No executable dependency or composition path is declared while this capability remains fail-closed.

### Scientific parameter and numeric binding contract

This fail-closed class does not resolve or execute scientific parameters.

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `constructor` | rejects before FunctorBase, ParameterManager, metadata, registry, or state effects |
| `forward` | independently rejects unsafe constructor bypass before state access |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_emissions_mapping.md`.

### References

- CF Conventions Version 1.13. (certificate)
- RELIC architecture overview, ScaleCoupling contract and governed Functor lifecycle. (certificate)

(class-relic-scientific-anthroposphere-inventory-emissioninventory)=
## emission_inventory

Canonical inventory node whose current scientific runtime is unbound.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.inventory.EmissionInventory` |
| Source | `relic\scientific\anthroposphere\inventory.py:46` |
| Runtime name | `not_applicable_runtime_unbound` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `runtime_unbound` |
| Variable kind | `runtime_unbound` |
| IPHS level | `level_0` |
| Implementation status | `not_implemented_fail_closed` |
| Class status | `fail-closed unavailable boundary` |

:::{warning}
This class is `[N] runtime_domain_unbound`. Construction and `forward(...)` reject before state access or side effects. Empty runtime input, output, and port declarations below are the documented fail-closed boundary, not missing documentation.
:::

### Constructor and argument meanings

```python
emission_inventory(*args: Any, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `args` | `Any` | `required` | Additional positional arguments accepted by the inherited constructor. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |

### Governing equations and canonical operator

**Literature or process form**

Global annual inventories require an explicit source-grid to target-domain partition before a local surface rate can be computed.

**Canonical RELIC structural form**

No IPHS formulation is claimed while the inventory domain binding is unavailable.

| Entry | Meaning |
| --- | --- |
| `runtime_status` | no executable inventory operator is registered |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | not_applicable_runtime_domain_unbound |
| `effort_variable` | not_applicable_runtime_domain_unbound |
| `state_advancer` | none_runtime_domain_unbound |
| `runtime_usage` | no UpdateItem or state mutation is produced |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
No executable dependency or composition path is declared while this capability remains fail-closed.

### Scientific parameter and numeric binding contract

This fail-closed class does not resolve or execute scientific parameters.

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `constructor` | rejects before FunctorBase, ParameterManager, calendar, proxy, or ledger effects |
| `forward` | independently rejects unsafe constructor bypass before state access |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthropogenic_emissions.md`.

### References

- Guizzardi, D. et al. (2025). HTAPv3.2 emission mosaic. DOI:10.5194/essd-17-5915-2025. (certificate)
- Crippa, M. et al. (2024). Spatial distribution of EDGAR v8.0 emissions. DOI:10.5194/essd-16-2811-2024. (certificate)
- CF Conventions, calendar and cell-area contracts. (certificate)

(class-relic-scientific-anthroposphere-land-use-mapping-landuseparametermapping)=
## land_use_parameter_mapping

Land Use Parameter Mapping Functor

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.land_use.mapping.LandUseParameterMapping` |
| Source | `relic\scientific\anthroposphere\land_use\mapping.py:178` |
| Runtime name | `land_use_mapping` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `observation` |
| Spatial scale | `grid_cell` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
land_use_parameter_mapping(name: str = 'land_use_mapping', config: Dict[str, Any] | None = None, *, field_query_bridge: Any | None = None, field_gradient_bridge: Any | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'land_use_mapping'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `field_query_bridge` | <code>typing.Any &#124; None</code> | `None` | Constructor control for field query bridge; see the class runtime contract and ParameterManager table below. |
| `field_gradient_bridge` | <code>typing.Any &#124; None</code> | `None` | Constructor control for field gradient bridge; see the class runtime contract and ParameterManager table below. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `land_use_fraction` | state | `1` | canonical nine-class grid-cell land-use fractional-area observation [1] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `diagnostics.land_use_parameter_mapping.vegetation.lai_max` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.lai_min` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.vcmax25` | output port | `umol CO2 m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.jmax25` | output port | `umol e- m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.root_depth` | output port | `m` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.canopy_height` | output port | `m` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.wood_density` | output port | `kg m-3` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.sla` | output port | `m2 kg-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.leaf_lifespan` | output port | `days` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.c3_fraction` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.vegetation.c4_fraction` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.soil.decomposition_rate` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.soil.nitrification_rate` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.soil.denitrification_rate` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.soil.soil_porosity` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.soil.infiltration_rate` | output port | `mm hr-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.soil.soil_carbon_input` | output port | `kg C m-2 yr-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.soil.tillage_factor` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.soil.erosion_rate` | output port | `kg m-2 yr-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.hydrology.surface_roughness` | output port | `m` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.hydrology.albedo` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.hydrology.emissivity` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.hydrology.interception_capacity` | output port | `mm` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.hydrology.runoff_coefficient` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.hydrology.evaporation_resistance` | output port | `s m-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_parameter_mapping.canonical_iphs_runtime` | output port | `dimensionless` | diagnostics | compatibility runtime marker [1], not strict IPHS evidence |

### Governing equations and canonical operator

**Literature or process form**

Land-surface models preserve sub-grid land units or tiles and aggregate supported grid-cell diagnostics or exchanges using land-cover fractions.

**Canonical RELIC structural form**

$y = O_{\mathrm{land}} ( f , \theta )$, where $O_{\mathrm{land}}$ is a read-only mixed-unit observation map.

| Entry | Meaning |
| --- | --- |
| `O_land` | calculate_weighted_parameters read-only area-weighted observation map |
| `R_query` | externally bound typed source-to-target field chain map |
| `G_gradient` | externally bound typed source-field gradient map |
| `O_h` | canonical runtime diagnostic output |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | none; the process is a read-only observation |
| `effort_variable` | not_applicable_read_only_observation_has_no_thermodynamic_effort |
| `state_advancer` | $y_{h} = O_{\mathrm{land}} ( f_{h} , \theta ) \mathrm{without} \mathrm{prognostic} \mathrm{state} \mathrm{advancement}$ |
| `runtime_usage` | forward evaluates $O_{\mathrm{land}}$ once and returns REPLACE diagnostics without prognostic ownership. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

Executable `ports()` owner: `LandUseParameterMapping`. The default instance returns no fixed PortSpec entries; its ports are configuration-conditional or observational.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.land_use.vegetation.barren.c3_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Sparse vegetation typically C3. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.barren.c4_fraction` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | Minimal C4 on barren land. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.barren.canopy_height` | `m` | `0.1; bounds=(0.05, 0.3)` | `fixed` | Very low canopy on barren land | Source: literature; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `anthroposphere.land_use.vegetation.barren.jmax25` | `umol e- m-2 s-1` | `20.0; bounds=(10.0, 40.0)` | `fixed` | Low Jmax25 for sparse vegetation | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.barren.lai_max` | `dimensionless` | `0.2; bounds=(0.0, 0.5)` | `fixed` | Minimal vegetation on barren land | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.barren.lai_min` | `dimensionless` | `0.0; bounds=(0.0, inf)` | `fixed` | No minimum vegetation on barren land. A leaf area index (m2 m-2) cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.barren.leaf_lifespan` | `days` | `60.0; bounds=(30.0, 120.0)` | `fixed` | Short leaf lifespan on barren land | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.barren.root_depth` | `m` | `0.1; bounds=(0.05, 0.3)` | `fixed` | Shallow roots on barren land | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.vegetation.barren.sla` | `m2 kg-1` | `40.0; bounds=(30.0, 60.0)` | `fixed` | High SLA for sparse vegetation | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.barren.vcmax25` | `umol CO2 m-2 s-1` | `10.0; bounds=(5.0, 20.0)` | `fixed` | Low Vcmax25 for sparse vegetation | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.barren.wood_density` | `kg m-3` | `0.0; bounds=(0.0, inf)` | `fixed` | No woody biomass on barren land. A density cannot be negative (kg m-3); the lower bound is 0. The upper bound stays open for this zero-valued class. | Source: literature; Chave et al. (2009) Biogeosciences 6, 2163-2178; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.cropland_c3.c3_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | C3 fraction for C3 crops. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.cropland_c3.c4_fraction` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | C4 fraction for C3 crops. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538 |
| `anthroposphere.land_use.vegetation.cropland_c3.canopy_height` | `m` | `1.5; bounds=(0.8, 2.5)` | `fixed` | Canopy height for C3 crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190 |
| `anthroposphere.land_use.vegetation.cropland_c3.jmax25` | `umol e- m-2 s-1` | `160.0; bounds=(120.0, 240.0)` | `fixed` | Jmax25 for C3 crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.cropland_c3.lai_max` | `dimensionless` | `4.0; bounds=(2.5, 6.0)` | `fixed` | Maximum LAI for C3 crops | Source: literature; Confalonieri et al. (2013) DOI:10.1016/j.eja.2013.06.003 |
| `anthroposphere.land_use.vegetation.cropland_c3.lai_min` | `dimensionless` | `0.5; bounds=(0.1, 1.0)` | `fixed` | Minimum LAI for C3 crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `anthroposphere.land_use.vegetation.cropland_c3.leaf_lifespan` | `days` | `120.0; bounds=(60.0, 180.0)` | `fixed` | Leaf lifespan for C3 crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Wright et al. (2004) Nature 428, 821-827 |
| `anthroposphere.land_use.vegetation.cropland_c3.root_depth` | `m` | `1.0; bounds=(0.5, 1.5)` | `fixed` | Root depth for C3 crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.cropland_c3.sla` | `m2 kg-1` | `25.0; bounds=(15.0, 40.0)` | `fixed` | SLA for C3 crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Wright et al. (2004) Nature 428, 821-827 |
| `anthroposphere.land_use.vegetation.cropland_c3.vcmax25` | `umol CO2 m-2 s-1` | `80.0; bounds=(60.0, 120.0)` | `fixed` | Vcmax25 for C3 crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.cropland_c3.wood_density` | `kg m-3` | `0.0; bounds=(0.0, inf)` | `fixed` | No woody biomass for annual crops. A density cannot be negative (kg m-3); the lower bound is 0. The upper bound stays open for this zero-valued class. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Chave et al. (2009) Biogeosciences 6, 2163-2178 |
| `anthroposphere.land_use.vegetation.cropland_c4.c3_fraction` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | C3 fraction for C4 crops. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538 |
| `anthroposphere.land_use.vegetation.cropland_c4.c4_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | C4 fraction for C4 crops. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.cropland_c4.canopy_height` | `m` | `2.5; bounds=(1.5, 4.0)` | `fixed` | Canopy height for C4 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190 |
| `anthroposphere.land_use.vegetation.cropland_c4.jmax25` | `umol e- m-2 s-1` | `80.0; bounds=(50.0, 120.0)` | `fixed` | Jmax25 for C4 crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.cropland_c4.lai_max` | `dimensionless` | `5.0; bounds=(3.0, 7.0)` | `fixed` | Maximum LAI for C4 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `anthroposphere.land_use.vegetation.cropland_c4.lai_min` | `dimensionless` | `0.5; bounds=(0.1, 1.0)` | `fixed` | Minimum LAI for C4 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `anthroposphere.land_use.vegetation.cropland_c4.leaf_lifespan` | `days` | `150.0; bounds=(80.0, 200.0)` | `fixed` | Leaf lifespan for C4 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Wright et al. (2004) Nature 428, 821-827 |
| `anthroposphere.land_use.vegetation.cropland_c4.root_depth` | `m` | `1.5; bounds=(0.8, 2.5)` | `fixed` | Root depth for C4 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.cropland_c4.sla` | `m2 kg-1` | `20.0; bounds=(12.0, 35.0)` | `fixed` | SLA for C4 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Wright et al. (2004) Nature 428, 821-827 |
| `anthroposphere.land_use.vegetation.cropland_c4.vcmax25` | `umol CO2 m-2 s-1` | `40.0; bounds=(25.0, 60.0)` | `fixed` | Vcmax25 for C4 crops (lower than C3) | Source: literature; Collatz, Berry and Clark (1998) Oecologia 114, 441-454, DOI:10.1007/s004420050468 |
| `anthroposphere.land_use.vegetation.cropland_c4.wood_density` | `kg m-3` | `0.0; bounds=(0.0, inf)` | `fixed` | No woody biomass for annual crops. A density cannot be negative (kg m-3); the lower bound is 0. The upper bound stays open for this zero-valued class. | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Chave et al. (2009) Biogeosciences 6, 2163-2178 |
| `anthroposphere.land_use.vegetation.pasture.c3_fraction` | `dimensionless` | `0.7; bounds=(0.5, 0.9)` | `fixed` | Mixed C3/C4 grasses favor C3 | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.pasture.c4_fraction` | `dimensionless` | `0.3; bounds=(0.1, 0.5)` | `fixed` | C4 grass component in pasture | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.pasture.canopy_height` | `m` | `0.3; bounds=(0.1, 0.6)` | `fixed` | Canopy height for pasture | Source: literature; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `anthroposphere.land_use.vegetation.pasture.jmax25` | `umol e- m-2 s-1` | `90.0; bounds=(60.0, 130.0)` | `fixed` | Jmax25 for pasture grasses | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.pasture.lai_max` | `dimensionless` | `2.5; bounds=(1.5, 4.0)` | `fixed` | Maximum LAI for managed pasture | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.pasture.lai_min` | `dimensionless` | `1.0; bounds=(0.5, 1.5)` | `fixed` | Minimum LAI for managed pasture | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.pasture.leaf_lifespan` | `days` | `60.0; bounds=(30.0, 120.0)` | `fixed` | Leaf lifespan for pasture | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.pasture.root_depth` | `m` | `0.5; bounds=(0.3, 0.8)` | `fixed` | Root depth for pasture | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.vegetation.pasture.sla` | `m2 kg-1` | `30.0; bounds=(20.0, 45.0)` | `fixed` | SLA for pasture grasses | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.pasture.vcmax25` | `umol CO2 m-2 s-1` | `45.0; bounds=(30.0, 65.0)` | `fixed` | Vcmax25 for pasture grasses | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.pasture.wood_density` | `kg m-3` | `0.0; bounds=(0.0, inf)` | `fixed` | No woody biomass for grasses. A density cannot be negative (kg m-3); the lower bound is 0. The upper bound stays open for this zero-valued class. | Source: literature; Chave et al. (2009) Biogeosciences 6, 2163-2178; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.primary_forest.c3_fraction` | `dimensionless` | `1.0; bounds=(0.8, 1.0)` | `fixed` | Fraction using C3 photosynthesis | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.primary_forest.c4_fraction` | `dimensionless` | `0.0; bounds=(0.0, 0.2)` | `fixed` | Fraction using C4 photosynthesis | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.primary_forest.canopy_height` | `m` | `30.0; bounds=(20.0, 45.0)` | `fixed` | Average canopy height | Source: literature; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `anthroposphere.land_use.vegetation.primary_forest.jmax25` | `umol e- m-2 s-1` | `120.0; bounds=(80.0, 160.0)` | `fixed` | Maximum electron transport rate at 25degC | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.primary_forest.lai_max` | `dimensionless` | `7.0; bounds=(5.0, 9.0)` | `fixed` | Maximum leaf area index for primary forest | Source: literature; Asner et al. (2003) DOI:10.1890/02-4022 |
| `anthroposphere.land_use.vegetation.primary_forest.lai_min` | `dimensionless` | `5.0; bounds=(3.0, 6.0)` | `fixed` | Minimum leaf area index for primary forest | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.primary_forest.leaf_lifespan` | `days` | `365.0; bounds=(200.0, 730.0)` | `fixed` | Average leaf lifespan | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.primary_forest.root_depth` | `m` | `3.0; bounds=(2.0, 5.0)` | `fixed` | Effective rooting depth | Source: literature; Jackson et al. (1996) DOI:10.1007/BF00333714 |
| `anthroposphere.land_use.vegetation.primary_forest.sla` | `m2 kg-1` | `15.0; bounds=(10.0, 25.0)` | `fixed` | Specific leaf area | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.primary_forest.vcmax25` | `umol CO2 m-2 s-1` | `60.0; bounds=(40.0, 80.0)` | `fixed` | Maximum carboxylation rate at 25degC | Source: literature; Kattge et al. (2011) DOI:10.1111/j.1365-2486.2011.02451.x |
| `anthroposphere.land_use.vegetation.primary_forest.wood_density` | `kg m-3` | `600.0; bounds=(400.0, 800.0)` | `fixed` | Wood density | Source: literature; Chave et al. (2006) DOI:10.1111/j.1365-2486.2005.001115.x |
| `anthroposphere.land_use.vegetation.rangeland.c3_fraction` | `dimensionless` | `0.5; bounds=(0.2, 0.8)` | `fixed` | Balanced C3/C4 mix in rangeland | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.rangeland.c4_fraction` | `dimensionless` | `0.5; bounds=(0.2, 0.8)` | `fixed` | C4 grass component in rangeland | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.rangeland.canopy_height` | `m` | `0.5; bounds=(0.2, 1.0)` | `fixed` | Canopy height for rangeland | Source: literature; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `anthroposphere.land_use.vegetation.rangeland.jmax25` | `umol e- m-2 s-1` | `80.0; bounds=(50.0, 120.0)` | `fixed` | Jmax25 for rangeland | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.rangeland.lai_max` | `dimensionless` | `2.0; bounds=(1.0, 3.5)` | `fixed` | Maximum LAI for natural grassland | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.rangeland.lai_min` | `dimensionless` | `0.5; bounds=(0.2, 1.0)` | `fixed` | Minimum LAI for natural grassland | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.rangeland.leaf_lifespan` | `days` | `90.0; bounds=(45.0, 180.0)` | `fixed` | Leaf lifespan for rangeland | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.rangeland.root_depth` | `m` | `1.0; bounds=(0.5, 2.0)` | `fixed` | Root depth for rangeland | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.vegetation.rangeland.sla` | `m2 kg-1` | `28.0; bounds=(18.0, 40.0)` | `fixed` | SLA for rangeland grasses | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.rangeland.vcmax25` | `umol CO2 m-2 s-1` | `40.0; bounds=(25.0, 60.0)` | `fixed` | Vcmax25 for rangeland | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.rangeland.wood_density` | `kg m-3` | `0.0; bounds=(0.0, inf)` | `fixed` | No woody biomass for grasses. A density cannot be negative (kg m-3); the lower bound is 0. The upper bound stays open for this zero-valued class. | Source: literature; Chave et al. (2009) Biogeosciences 6, 2163-2178; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.secondary_forest.c3_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | C3 fraction for secondary forest. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.secondary_forest.c4_fraction` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | C4 fraction for secondary forest. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.secondary_forest.canopy_height` | `m` | `15.0; bounds=(8.0, 25.0)` | `fixed` | Canopy height for secondary forest | Source: literature; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `anthroposphere.land_use.vegetation.secondary_forest.jmax25` | `umol e- m-2 s-1` | `100.0; bounds=(70.0, 140.0)` | `fixed` | Jmax25 for secondary forest | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.secondary_forest.lai_max` | `dimensionless` | `5.0; bounds=(3.0, 7.0)` | `fixed` | Maximum LAI for secondary forest | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.secondary_forest.lai_min` | `dimensionless` | `3.0; bounds=(2.0, 4.0)` | `fixed` | Minimum LAI for secondary forest | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.secondary_forest.leaf_lifespan` | `days` | `300.0; bounds=(180.0, 500.0)` | `fixed` | Leaf lifespan for secondary forest | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.secondary_forest.root_depth` | `m` | `2.0; bounds=(1.0, 3.0)` | `fixed` | Root depth for secondary forest | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.vegetation.secondary_forest.sla` | `m2 kg-1` | `18.0; bounds=(12.0, 28.0)` | `fixed` | SLA for secondary forest | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.secondary_forest.vcmax25` | `umol CO2 m-2 s-1` | `50.0; bounds=(35.0, 70.0)` | `fixed` | Vcmax25 for secondary forest | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.secondary_forest.wood_density` | `kg m-3` | `500.0; bounds=(300.0, 700.0)` | `fixed` | Wood density for secondary forest | Source: literature; Chave et al. (2009) Biogeosciences 6, 2163-2178; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.urban.c3_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Urban vegetation typically C3. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.urban.c4_fraction` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | Minimal C4 in urban areas. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.urban.canopy_height` | `m` | `0.5; bounds=(0.1, 2.0)` | `fixed` | Canopy height for urban vegetation | Source: literature; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `anthroposphere.land_use.vegetation.urban.jmax25` | `umol e- m-2 s-1` | `40.0; bounds=(20.0, 80.0)` | `fixed` | Jmax25 for urban vegetation | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.urban.lai_max` | `dimensionless` | `0.5; bounds=(0.1, 1.0)` | `fixed` | Maximum LAI for urban vegetation | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.urban.lai_min` | `dimensionless` | `0.1; bounds=(0.0, 0.3)` | `fixed` | Minimum LAI for urban vegetation | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.urban.leaf_lifespan` | `days` | `180.0; bounds=(90.0, 365.0)` | `fixed` | Leaf lifespan for urban vegetation | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.urban.root_depth` | `m` | `0.2; bounds=(0.1, 0.5)` | `fixed` | Root depth for urban vegetation | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.vegetation.urban.sla` | `m2 kg-1` | `35.0; bounds=(25.0, 50.0)` | `fixed` | SLA for urban vegetation | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.urban.vcmax25` | `umol CO2 m-2 s-1` | `20.0; bounds=(10.0, 40.0)` | `fixed` | Vcmax25 for urban vegetation | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.urban.wood_density` | `kg m-3` | `0.0; bounds=(0.0, inf)` | `fixed` | Minimal woody biomass. A density cannot be negative (kg m-3); the lower bound is 0. The upper bound stays open for this zero-valued class. | Source: literature; Chave et al. (2009) Biogeosciences 6, 2163-2178; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.water.c3_fraction` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No photosynthesis over water. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.water.c4_fraction` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No photosynthesis over water. A fraction of the vegetation cover lies in [0, 1]. | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.water.canopy_height` | `m` | `0.0; bounds=(0.0, inf)` | `fixed` | No canopy over water. A canopy height cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `anthroposphere.land_use.vegetation.water.jmax25` | `umol e- m-2 s-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No photosynthesis over water. A maximum electron transport rate cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.water.lai_max` | `dimensionless` | `0.0; bounds=(0.0, inf)` | `fixed` | No vegetation over water. A leaf area index (m2 m-2) cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.water.lai_min` | `dimensionless` | `0.0; bounds=(0.0, inf)` | `fixed` | No vegetation over water. A leaf area index (m2 m-2) cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.water.leaf_lifespan` | `days` | `0.0; bounds=(0.0, inf)` | `fixed` | No leaves over water. A leaf lifespan cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.water.root_depth` | `m` | `0.0; bounds=(0.0, inf)` | `fixed` | No roots over water. A rooting depth cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.vegetation.water.sla` | `m2 kg-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No leaves over water. A specific leaf area cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Wright et al. (2004) Nature 428, 821-827; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.vegetation.water.vcmax25` | `umol CO2 m-2 s-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No photosynthesis over water. A maximum carboxylation rate cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Bernacchi et al. (2001) Plant Cell Environ. 24, 253-259 |
| `anthroposphere.land_use.vegetation.water.wood_density` | `kg m-3` | `0.0; bounds=(0.0, inf)` | `fixed` | No woody biomass over water. A density cannot be negative (kg m-3); the lower bound is 0. The upper bound stays open for this zero-valued class. | Source: literature; Chave et al. (2009) Biogeosciences 6, 2163-2178; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.barren.decomposition_rate` | `dimensionless` | `0.5; bounds=(0.3, 0.8)` | `fixed` | Low decomposition on barren land | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.barren.denitrification_rate` | `dimensionless` | `0.5; bounds=(0.3, 0.8)` | `fixed` | Low denitrification on barren land | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.barren.erosion_rate` | `kg m-2 yr-1` | `0.2; bounds=(0.1, 0.5)` | `fixed` | High erosion on exposed soil | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.barren.infiltration_rate` | `mm hr-1` | `15.0; bounds=(5.0, 30.0)` | `fixed` | Moderate infiltration on barren land | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.barren.nitrification_rate` | `dimensionless` | `0.5; bounds=(0.3, 0.8)` | `fixed` | Low nitrification on barren land | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.barren.soil_carbon_input` | `kg C m-2 yr-1` | `0.05; bounds=(0.01, 0.1)` | `fixed` | Very low carbon input | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.barren.soil_porosity` | `dimensionless` | `0.3; bounds=(0.2, 0.4)` | `fixed` | Variable porosity on barren land | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.barren.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage on barren land. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.cropland_c3.decomposition_rate` | `dimensionless` | `1.3; bounds=(1.1, 1.6)` | `fixed` | Enhanced decomposition due to tillage | Source: literature; Six et al. (2002) DOI:10.1016/S0038-0717(02)00054-7 |
| `anthroposphere.land_use.soil.cropland_c3.denitrification_rate` | `dimensionless` | `0.8; bounds=(0.6, 1.0)` | `fixed` | Reduced denitrification in well-aerated soil | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Li et al. (2000) J. Geophys. Res. 105, 9149-9167 |
| `anthroposphere.land_use.soil.cropland_c3.erosion_rate` | `kg m-2 yr-1` | `0.1; bounds=(0.05, 0.2)` | `fixed` | High erosion from bare soil periods | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c3.infiltration_rate` | `mm hr-1` | `25.0; bounds=(15.0, 40.0)` | `fixed` | Moderate infiltration in cropland | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Green & Ampt (1911) J. Agric. Sci. 4, 1-24 |
| `anthroposphere.land_use.soil.cropland_c3.nitrification_rate` | `dimensionless` | `1.2; bounds=(1.0, 1.5)` | `fixed` | Enhanced nitrification in tilled soil | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Li et al. (2000) J. Geophys. Res. 105, 9149-9167 |
| `anthroposphere.land_use.soil.cropland_c3.soil_carbon_input` | `kg C m-2 yr-1` | `0.3; bounds=(0.2, 0.5)` | `fixed` | Lower carbon input from crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c3.soil_porosity` | `dimensionless` | `0.38; bounds=(0.3, 0.45)` | `fixed` | Reduced porosity from tillage | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Rawls et al. (1982) Trans. ASAE 25, 1316-1320 |
| `anthroposphere.land_use.soil.cropland_c3.tillage_factor` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Full tillage impact. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c4.decomposition_rate` | `dimensionless` | `1.3; bounds=(1.1, 1.6)` | `fixed` | Enhanced decomposition due to tillage | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179 |
| `anthroposphere.land_use.soil.cropland_c4.denitrification_rate` | `dimensionless` | `0.8; bounds=(0.6, 1.0)` | `fixed` | Reduced denitrification in well-aerated soil | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Li et al. (2000) J. Geophys. Res. 105, 9149-9167 |
| `anthroposphere.land_use.soil.cropland_c4.erosion_rate` | `kg m-2 yr-1` | `0.1; bounds=(0.05, 0.2)` | `fixed` | High erosion from bare soil periods | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c4.infiltration_rate` | `mm hr-1` | `25.0; bounds=(15.0, 40.0)` | `fixed` | Moderate infiltration in cropland | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Green & Ampt (1911) J. Agric. Sci. 4, 1-24 |
| `anthroposphere.land_use.soil.cropland_c4.nitrification_rate` | `dimensionless` | `1.2; bounds=(1.0, 1.5)` | `fixed` | Enhanced nitrification in tilled soil | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Li et al. (2000) J. Geophys. Res. 105, 9149-9167 |
| `anthroposphere.land_use.soil.cropland_c4.soil_carbon_input` | `kg C m-2 yr-1` | `0.35; bounds=(0.25, 0.55)` | `fixed` | Slightly higher residue from C4 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c4.soil_porosity` | `dimensionless` | `0.38; bounds=(0.3, 0.45)` | `fixed` | Reduced porosity from tillage | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Rawls et al. (1982) Trans. ASAE 25, 1316-1320 |
| `anthroposphere.land_use.soil.cropland_c4.tillage_factor` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Full tillage impact. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.pasture.decomposition_rate` | `dimensionless` | `1.1; bounds=(0.9, 1.3)` | `fixed` | Slightly enhanced decomposition | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.pasture.denitrification_rate` | `dimensionless` | `0.9; bounds=(0.7, 1.1)` | `fixed` | Reduced denitrification | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.pasture.erosion_rate` | `kg m-2 yr-1` | `0.05; bounds=(0.02, 0.1)` | `fixed` | Moderate erosion from overgrazing | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.pasture.infiltration_rate` | `mm hr-1` | `20.0; bounds=(10.0, 35.0)` | `fixed` | Reduced infiltration from compaction | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.pasture.nitrification_rate` | `dimensionless` | `1.1; bounds=(0.9, 1.3)` | `fixed` | Enhanced nitrification from urine patches | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.pasture.soil_carbon_input` | `kg C m-2 yr-1` | `0.4; bounds=(0.25, 0.6)` | `fixed` | Moderate carbon input | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.pasture.soil_porosity` | `dimensionless` | `0.35; bounds=(0.25, 0.45)` | `fixed` | Compacted soil from grazing | Source: literature; Greenwood & McKenzie (2001) DOI:10.1016/S0167-1987(01)00178-0 |
| `anthroposphere.land_use.soil.pasture.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in pasture. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.primary_forest.decomposition_rate` | `dimensionless` | `0.8; bounds=(0.6, 1.0)` | `fixed` | Slower decomposition under forest | Source: literature; Davidson & Janssens (2006) DOI:10.1038/nature04514 |
| `anthroposphere.land_use.soil.primary_forest.denitrification_rate` | `dimensionless` | `1.1; bounds=(0.9, 1.3)` | `fixed` | Enhanced denitrification under forest | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.primary_forest.erosion_rate` | `kg m-2 yr-1` | `0.01; bounds=(0.005, 0.02)` | `fixed` | Very low erosion under forest | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.primary_forest.infiltration_rate` | `mm hr-1` | `50.0; bounds=(30.0, 80.0)` | `fixed` | High infiltration in forest | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.primary_forest.nitrification_rate` | `dimensionless` | `0.9; bounds=(0.7, 1.1)` | `fixed` | Nitrification rate modifier for forest | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.primary_forest.soil_carbon_input` | `kg C m-2 yr-1` | `0.8; bounds=(0.5, 1.2)` | `fixed` | High carbon input from forest litter | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.primary_forest.soil_porosity` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | High porosity in forest soils | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.primary_forest.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in natural forest. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.rangeland.decomposition_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal decomposition rate | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.rangeland.denitrification_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal denitrification rate | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.rangeland.erosion_rate` | `kg m-2 yr-1` | `0.03; bounds=(0.01, 0.06)` | `fixed` | Low to moderate erosion | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.rangeland.infiltration_rate` | `mm hr-1` | `35.0; bounds=(20.0, 50.0)` | `fixed` | Good infiltration in rangeland | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.rangeland.nitrification_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal nitrification rate | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.rangeland.soil_carbon_input` | `kg C m-2 yr-1` | `0.3; bounds=(0.2, 0.5)` | `fixed` | Moderate carbon input from grasses | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.rangeland.soil_porosity` | `dimensionless` | `0.4; bounds=(0.3, 0.5)` | `fixed` | Good porosity in natural grassland | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.rangeland.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in rangeland. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.secondary_forest.decomposition_rate` | `dimensionless` | `0.9; bounds=(0.7, 1.1)` | `fixed` | Intermediate decomposition in secondary forest | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.secondary_forest.denitrification_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal denitrification rate | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.secondary_forest.erosion_rate` | `kg m-2 yr-1` | `0.02; bounds=(0.01, 0.04)` | `fixed` | Low erosion under secondary forest | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.secondary_forest.infiltration_rate` | `mm hr-1` | `40.0; bounds=(25.0, 60.0)` | `fixed` | Good infiltration in secondary forest | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.secondary_forest.nitrification_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal nitrification rate | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.secondary_forest.soil_carbon_input` | `kg C m-2 yr-1` | `0.6; bounds=(0.4, 0.9)` | `fixed` | Moderate carbon input | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.secondary_forest.soil_porosity` | `dimensionless` | `0.42; bounds=(0.32, 0.52)` | `fixed` | Good porosity in secondary forest | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.secondary_forest.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in secondary forest. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.urban.decomposition_rate` | `dimensionless` | `0.2; bounds=(0.1, 0.4)` | `fixed` | Very low decomposition in urban soils | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.urban.denitrification_rate` | `dimensionless` | `0.2; bounds=(0.1, 0.4)` | `fixed` | Low denitrification in urban soils | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.urban.erosion_rate` | `kg m-2 yr-1` | `0.001; bounds=(0.0, 0.005)` | `fixed` | Protected surfaces have minimal erosion | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.urban.infiltration_rate` | `mm hr-1` | `2.0; bounds=(0.5, 5.0)` | `fixed` | Very low infiltration in urban areas | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.urban.nitrification_rate` | `dimensionless` | `0.3; bounds=(0.1, 0.5)` | `fixed` | Low nitrification in urban soils | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.urban.soil_carbon_input` | `kg C m-2 yr-1` | `0.05; bounds=(0.01, 0.1)` | `fixed` | Minimal carbon input | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.urban.soil_porosity` | `dimensionless` | `0.15; bounds=(0.1, 0.25)` | `fixed` | Highly compacted urban soils | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.urban.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in urban areas. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.water.decomposition_rate` | `dimensionless` | `0.0; bounds=(0.0, inf)` | `fixed` | No soil processes over water. The value is a non-negative multiplier of the reference soil process rate (the other land-use classes declare it within [0.1, 1.3]); a negative factor would reverse the process, so the lower bound is 0. | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.water.denitrification_rate` | `dimensionless` | `0.0; bounds=(0.0, inf)` | `fixed` | No soil processes over water. The value is a non-negative multiplier of the reference soil process rate (the other land-use classes declare it within [0.1, 1.3]); a negative factor would reverse the process, so the lower bound is 0. | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.water.erosion_rate` | `kg m-2 yr-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No erosion over water. An erosion rate (soil loss) cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.water.infiltration_rate` | `mm hr-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No infiltration over water. An infiltration rate into the soil cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.water.nitrification_rate` | `dimensionless` | `0.0; bounds=(0.0, inf)` | `fixed` | No soil processes over water. The value is a non-negative multiplier of the reference soil process rate (the other land-use classes declare it within [0.1, 1.3]); a negative factor would reverse the process, so the lower bound is 0. | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.water.soil_carbon_input` | `kg C m-2 yr-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No soil carbon input over water. A soil carbon input flux (inputs only) cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.water.soil_porosity` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No soil over water. Porosity is a volume fraction of the soil and lies in [0, 1]. | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.water.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage over water. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.hydrological.barren.albedo` | `dimensionless` | `0.3; bounds=(0.2, 0.4)` | `fixed` | High albedo for bare soil | Source: literature; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.barren.emissivity` | `dimensionless` | `0.92; bounds=(0.88, 0.96)` | `fixed` | Moderate emissivity for soil | Source: literature; Brutsaert (1975) Water Resour. Res. 11, 742-744; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.barren.evaporation_resistance` | `s m-1` | `150.0; bounds=(80.0, 250.0)` | `fixed` | Moderate resistance over bare soil | Source: literature; Monteith (1965) Symp. Soc. Exp. Biol. 19, 205-234; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.barren.interception_capacity` | `mm` | `0.05; bounds=(0.0, 0.15)` | `fixed` | Minimal interception on barren land | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.hydrological.barren.runoff_coefficient` | `dimensionless` | `0.6; bounds=(0.4, 0.8)` | `fixed` | High runoff from bare soil | Source: literature; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24, 43-69; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.barren.surface_roughness` | `m` | `0.02; bounds=(0.005, 0.05)` | `fixed` | Low roughness for bare soil | Source: literature; Brutsaert (1982) Evaporation into the Atmosphere, Springer; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.cropland_c3.albedo` | `dimensionless` | `0.2; bounds=(0.15, 0.25)` | `fixed` | Moderate albedo for cropland | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `anthroposphere.land_use.hydrological.cropland_c3.emissivity` | `dimensionless` | `0.95; bounds=(0.92, 0.98)` | `fixed` | Moderate emissivity | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Brutsaert (1975) Water Resour. Res. 11, 742-744 |
| `anthroposphere.land_use.hydrological.cropland_c3.evaporation_resistance` | `s m-1` | `100.0; bounds=(50.0, 200.0)` | `fixed` | Low resistance over bare/sparse canopy | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Monteith (1965) Symp. Soc. Exp. Biol. 19, 205-234 |
| `anthroposphere.land_use.hydrological.cropland_c3.interception_capacity` | `mm` | `0.5; bounds=(0.2, 1.0)` | `fixed` | Low interception capacity | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.cropland_c3.runoff_coefficient` | `dimensionless` | `0.3; bounds=(0.2, 0.5)` | `fixed` | Moderate runoff from cropland | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24, 43-69 |
| `anthroposphere.land_use.hydrological.cropland_c3.surface_roughness` | `m` | `0.1; bounds=(0.05, 0.2)` | `fixed` | Low surface roughness for crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Brutsaert (1982) Evaporation into the Atmosphere, Springer |
| `anthroposphere.land_use.hydrological.cropland_c4.albedo` | `dimensionless` | `0.18; bounds=(0.14, 0.23)` | `fixed` | Slightly lower albedo than C3 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `anthroposphere.land_use.hydrological.cropland_c4.emissivity` | `dimensionless` | `0.95; bounds=(0.92, 0.98)` | `fixed` | Moderate emissivity | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Brutsaert (1975) Water Resour. Res. 11, 742-744 |
| `anthroposphere.land_use.hydrological.cropland_c4.evaporation_resistance` | `s m-1` | `120.0; bounds=(70.0, 200.0)` | `fixed` | Moderate resistance | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Monteith (1965) Symp. Soc. Exp. Biol. 19, 205-234 |
| `anthroposphere.land_use.hydrological.cropland_c4.interception_capacity` | `mm` | `0.7; bounds=(0.3, 1.2)` | `fixed` | Moderate interception capacity | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.cropland_c4.runoff_coefficient` | `dimensionless` | `0.25; bounds=(0.15, 0.4)` | `fixed` | Moderate runoff from C4 cropland | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24, 43-69 |
| `anthroposphere.land_use.hydrological.cropland_c4.surface_roughness` | `m` | `0.15; bounds=(0.08, 0.3)` | `fixed` | Slightly higher roughness for taller C4 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Brutsaert (1982) Evaporation into the Atmosphere, Springer |
| `anthroposphere.land_use.hydrological.pasture.albedo` | `dimensionless` | `0.22; bounds=(0.18, 0.28)` | `fixed` | Moderate albedo for grassland | Source: literature; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.pasture.emissivity` | `dimensionless` | `0.96; bounds=(0.93, 0.98)` | `fixed` | High emissivity for grass | Source: literature; Brutsaert (1975) Water Resour. Res. 11, 742-744; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.pasture.evaporation_resistance` | `s m-1` | `80.0; bounds=(40.0, 150.0)` | `fixed` | Low resistance over short grass | Source: literature; Monteith (1965) Symp. Soc. Exp. Biol. 19, 205-234; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.pasture.interception_capacity` | `mm` | `0.3; bounds=(0.1, 0.6)` | `fixed` | Low interception capacity | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.hydrological.pasture.runoff_coefficient` | `dimensionless` | `0.35; bounds=(0.25, 0.5)` | `fixed` | Moderate to high runoff from compacted pasture | Source: literature; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24, 43-69; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.pasture.surface_roughness` | `m` | `0.03; bounds=(0.01, 0.08)` | `fixed` | Low surface roughness for short grass | Source: literature; Brutsaert (1982) Evaporation into the Atmosphere, Springer; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.primary_forest.albedo` | `dimensionless` | `0.12; bounds=(0.08, 0.16)` | `fixed` | Low albedo for forest canopy | Source: literature; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.primary_forest.emissivity` | `dimensionless` | `0.98; bounds=(0.95, 0.99)` | `fixed` | High emissivity for vegetation | Source: literature; Brutsaert (1975) Water Resour. Res. 11, 742-744; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.primary_forest.evaporation_resistance` | `s m-1` | `200.0; bounds=(100.0, 300.0)` | `fixed` | High resistance under forest canopy | Source: literature; Monteith (1965) Symp. Soc. Exp. Biol. 19, 205-234; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.primary_forest.interception_capacity` | `mm` | `2.5; bounds=(1.5, 4.0)` | `fixed` | High canopy interception capacity | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.hydrological.primary_forest.runoff_coefficient` | `dimensionless` | `0.1; bounds=(0.05, 0.2)` | `fixed` | Low runoff from forest | Source: literature; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24, 43-69; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.primary_forest.surface_roughness` | `m` | `2.0; bounds=(1.0, 3.0)` | `fixed` | High surface roughness for forests | Source: literature; Bonan (2008) Ecological Climatology |
| `anthroposphere.land_use.hydrological.rangeland.albedo` | `dimensionless` | `0.25; bounds=(0.2, 0.32)` | `fixed` | Higher albedo for sparse vegetation | Source: literature; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.rangeland.emissivity` | `dimensionless` | `0.96; bounds=(0.93, 0.98)` | `fixed` | High emissivity for grass | Source: literature; Brutsaert (1975) Water Resour. Res. 11, 742-744; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.rangeland.evaporation_resistance` | `s m-1` | `90.0; bounds=(50.0, 150.0)` | `fixed` | Moderate resistance | Source: literature; Monteith (1965) Symp. Soc. Exp. Biol. 19, 205-234; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.rangeland.interception_capacity` | `mm` | `0.4; bounds=(0.2, 0.8)` | `fixed` | Low to moderate interception | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.hydrological.rangeland.runoff_coefficient` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Moderate runoff from rangeland | Source: literature; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24, 43-69; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.rangeland.surface_roughness` | `m` | `0.05; bounds=(0.02, 0.12)` | `fixed` | Low surface roughness for natural grass | Source: literature; Brutsaert (1982) Evaporation into the Atmosphere, Springer; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.secondary_forest.albedo` | `dimensionless` | `0.14; bounds=(0.1, 0.18)` | `fixed` | Slightly higher albedo than primary forest | Source: literature; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.secondary_forest.emissivity` | `dimensionless` | `0.97; bounds=(0.94, 0.99)` | `fixed` | High emissivity for vegetation | Source: literature; Brutsaert (1975) Water Resour. Res. 11, 742-744; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.secondary_forest.evaporation_resistance` | `s m-1` | `150.0; bounds=(80.0, 250.0)` | `fixed` | Moderate resistance | Source: literature; Monteith (1965) Symp. Soc. Exp. Biol. 19, 205-234; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.secondary_forest.interception_capacity` | `mm` | `2.0; bounds=(1.0, 3.0)` | `fixed` | Good interception capacity | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.hydrological.secondary_forest.runoff_coefficient` | `dimensionless` | `0.15; bounds=(0.08, 0.25)` | `fixed` | Low to moderate runoff | Source: literature; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24, 43-69; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.secondary_forest.surface_roughness` | `m` | `1.0; bounds=(0.5, 2.0)` | `fixed` | Moderate surface roughness | Source: literature; Brutsaert (1982) Evaporation into the Atmosphere, Springer; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.urban.albedo` | `dimensionless` | `0.15; bounds=(0.1, 0.25)` | `fixed` | Low albedo for urban surfaces | Source: literature; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.urban.emissivity` | `dimensionless` | `0.9; bounds=(0.85, 0.95)` | `fixed` | Lower emissivity for urban materials | Source: literature; Brutsaert (1975) Water Resour. Res. 11, 742-744; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.urban.evaporation_resistance` | `s m-1` | `500.0; bounds=(300.0, 1000.0)` | `fixed` | Very high resistance over impervious surfaces | Source: literature; Monteith (1965) Symp. Soc. Exp. Biol. 19, 205-234; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.urban.interception_capacity` | `mm` | `0.1; bounds=(0.0, 0.3)` | `fixed` | Very low interception in urban areas | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.hydrological.urban.runoff_coefficient` | `dimensionless` | `0.85; bounds=(0.7, 0.95)` | `fixed` | Very high runoff from impervious surfaces | Source: literature; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24, 43-69; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.urban.surface_roughness` | `m` | `1.5; bounds=(0.5, 3.0)` | `fixed` | High roughness from buildings | Source: literature; Brutsaert (1982) Evaporation into the Atmosphere, Springer; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.water.albedo` | `dimensionless` | `0.08; bounds=(0.05, 0.12)` | `fixed` | Low albedo for water | Source: literature; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.water.emissivity` | `dimensionless` | `0.98; bounds=(0.95, 0.99)` | `fixed` | High emissivity for water | Source: literature; Brutsaert (1975) Water Resour. Res. 11, 742-744; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.water.evaporation_resistance` | `s m-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No resistance for direct evaporation. A resistance to evaporation cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Monteith (1965) Symp. Soc. Exp. Biol. 19, 205-234; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.water.interception_capacity` | `mm` | `0.0; bounds=(0.0, inf)` | `fixed` | No interception over water. An interception storage capacity cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.hydrological.water.runoff_coefficient` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No runoff from water surface. A runoff coefficient is the fraction of precipitation that becomes runoff and lies in [0, 1]. | Source: literature; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24, 43-69; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.hydrological.water.surface_roughness` | `m` | `0.001; bounds=(0.0001, 0.01)` | `fixed` | Very low roughness for water surface | Source: literature; Brutsaert (1982) Evaporation into the Atmosphere, Springer; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | all outputs use explicit UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{false} , \mathrm{unit} , \mathrm{source}_{\mathrm{functor}} )$ |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_land_use_transitions.md`.

### References

- Pongratz et al. (2018) DOI:10.1111/gcb.13988 (FunctorMetadata)
- Lawrence et al. (2016) DOI:10.5194/gmd-9-2973-2016 (FunctorMetadata)
- FAO (2017) Sustainable Soil Management Guidelines (FunctorMetadata)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- Lawrence, D.M. et al. (2016). The Land Use Model Intercomparison Project. (certificate)
- Pongratz, J. et al. (2018). Models meet data for land management. (certificate)
- Bonan, G.B. (2019). Climate Change and Terrestrial Ecosystem Modeling. (certificate)
- Le Moigne, P. (2018). SURFEX Scientific Documentation. (certificate)

(class-relic-scientific-anthroposphere-land-use-transitions-landusetransitions)=
## land_use_transitions

Land Use Transitions Functor

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.land_use.transitions.LandUseTransitions` |
| Source | `relic\scientific\anthroposphere\land_use\transitions.py:103` |
| Runtime name | `land_use_transitions` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
land_use_transitions(name: str | Dict[str, Any] = 'land_use_transitions', config: Dict[str, Any] | None = None, *, upscale_bridge: Any | None = None, downscale_bridge: Any | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | <code>str &#124; typing.Dict[str, typing.Any]</code> | `'land_use_transitions'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `upscale_bridge` | <code>typing.Any &#124; None</code> | `None` | Constructor control for upscale bridge; see the class runtime contract and ParameterManager table below. |
| `downscale_bridge` | <code>typing.Any &#124; None</code> | `None` | Constructor control for downscale bridge; see the class runtime contract and ParameterManager table below. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `land_use_fraction` | state | `1` | Canonical nine-class full-state area fraction on each independent support [1]. |
| `time_index` | forcing | `1` | Required or optional value consumed by forward(). |
| `gdp_per_capita` | forcing | `1` | Required or optional value consumed by forward(). |
| `population_density` | forcing | `1` | Required or optional value consumed by forward(). |
| `market_accessibility` | forcing | `1` | Required or optional value consumed by forward(). |
| `policy_intensity` | forcing | `1` | Required or optional value consumed by forward(). |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `land_use_fraction` | state | `1` | decided at run time by the returned write mode | Canonical nine-class full-state area fraction on each independent support [1]. |
| `diagnostics.land_use_transitions.delta_fraction` | output port | `1` | diagnostics | Per-step observation p(t+dt) - p(t) [1]. |
| `diagnostics.land_use_transitions.dominant_type_index` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_transitions.maximum_fraction` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_transitions.soil_carbon_input_rate_change` | output port | `kg C m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.land_use_transitions.canonical_iphs_runtime` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

A row-sum-zero continuous-time Markov generator preserves total land-use fraction under its matrix exponential.

**Canonical RELIC structural form**

$$
\dot{p} = p Q ( \theta ); p ( t + \mathrm{d}t ) = p \exp ( Q ( \theta ) \mathrm{d}t_{\mathrm{years}} ); Q 1 = 0 .
$$

| Entry | Meaning |
| --- | --- |
| `Q` | Row-sum-zero CTMC generator built by _build_ctmc_generator and _compute_transition_matrix. |
| `exp_Q_dt` | $\mathtt{torch.matrix\_exp} ( Q \cdot \mathrm{d}t_{\mathrm{years}} )$ transition map. |
| `H_Q` | Positive exit-rate Hodge from build_hodge_contract. |
| `R_h` | Externally bound fine-to-coarse typed chain map. |
| `P_h` | Externally bound coarse-to-fine typed chain map. |
| `O_h` | Namespaced observation outputs. |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | Land-use fraction state cochain $p_{h}$. |
| `effort_variable` | CTMC generator $Q_{h}$ and mobility audit $H_{Q}$. |
| `state_advancer` | $p ( t + \mathrm{d}t ) = p ( t ) \exp ( Q_{h} \mathrm{d}t_{\mathrm{years}} ) .$ |
| `runtime_usage` | forward evaluates one matrix exponential and returns one full-state replacement. |

### Composition channels and executable PortSpec API

Certificate composition channels: `land_use_transition_area_fraction_port`

Executable `ports()` owner: `LandUseTransitions`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `land_use_transition_area_fraction_port` | `land_use_transition_potential [1]` | `land_use_transition_port_flow [yr^-1]` | `yr^-1` | into_system; management |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.land_use.transition_rates.abandonment_rate` | `yr-1` | `0.0015; bounds=(0.0, 0.01)` | `fixed` | Annual fraction of agricultural land abandoned or restored to natural vegetation | Source: LUH2 global mean (1990-2015); Hurtt et al. (2020) GMD 13:5425-5464 |
| `anthroposphere.land_use.transition_rates.afforestation_rate` | `yr-1` | `0.002; bounds=(0.0, 0.01)` | `fixed` | Annual fraction of agricultural or degraded land reverting to forests through planting or natural regrowth | Source: LUH2 global mean (1990-2015); Hurtt et al. (2020) GMD 13:5425-5464 |
| `anthroposphere.land_use.transition_rates.agricultural_expansion_rate` | `yr-1` | `0.004; bounds=(0.0, 0.02)` | `fixed` | Annual fraction of natural ecosystems converted to cropland or pasture | Source: LUH2 global mean (1990-2015); Hurtt et al. (2020) GMD 13:5425-5464 |
| `anthroposphere.land_use.transition_rates.deforestation_rate` | `yr-1` | `0.005; bounds=(0.0, 0.02)` | `fixed` | Global mean annual fraction of primary or secondary forest converted to other land uses | Source: LUH2 global mean (1990-2015); Hurtt et al. (2020) GMD 13:5425-5464 |
| `anthroposphere.land_use.transition_rates.urbanization_rate` | `yr-1` | `0.001; bounds=(0.0, 0.005)` | `fixed` | Annual fraction of non-urban land converted to urban/built-up area | Source: LUH2 global mean (1990-2015); IPCC (2019) SRCCL; Hurtt et al. (2020) |
| `anthroposphere.land_use.transition_channels.crops_to_secondary` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Fraction of abandoned crops becoming secondary forest | Source: default; Hurtt et al. (2020) Geosci. Model Dev. 13, 5425-5464; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.transition_channels.pasture_to_crops_c3` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Fraction of pasture converted to C3 crops | Source: default; Hurtt et al. (2020) Geosci. Model Dev. 13, 5425-5464; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.transition_channels.pasture_to_crops_c4` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Fraction of pasture converted to C4 crops | Source: default; Hurtt et al. (2020) Geosci. Model Dev. 13, 5425-5464; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.transition_channels.pasture_to_secondary` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Fraction of abandoned pasture becoming secondary forest | Source: default; Hurtt et al. (2020) Geosci. Model Dev. 13, 5425-5464; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.transition_channels.primary_to_crops` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of deforested primary forest becoming crops | Source: default; Hurtt et al. (2020) Geosci. Model Dev. 13, 5425-5464; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.transition_channels.primary_to_pasture` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of deforested primary forest becoming pasture | Source: default; Hurtt et al. (2020) Geosci. Model Dev. 13, 5425-5464; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.transition_channels.primary_to_secondary` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of deforested primary forest becoming secondary forest | Source: default; Hurtt et al. (2020) Geosci. Model Dev. 13, 5425-5464; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.transition_channels.secondary_to_crops` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Fraction of deforested secondary forest becoming crops | Source: default; Hurtt et al. (2020) Geosci. Model Dev. 13, 5425-5464; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.transition_channels.secondary_to_pasture` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Fraction of deforested secondary forest becoming pasture | Source: default; Hurtt et al. (2020) Geosci. Model Dev. 13, 5425-5464; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.clamp_minimum` | `dimensionless` | `1e-10; bounds=(1e-12, 1e-08)` | `fixed` | Minimum value to prevent division by zero | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.clamp_minimum_zero` | `dimensionless` | `0.0; bounds=(0.0, 0.01)` | `fixed` | Minimum value for clamping fractions | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.default_target_land_use_index` | `dimensionless` | `0; bounds=(0.0, 8.0)` | `fixed` | Default target land use type index when not specified in state (0=primary_forest) | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_gdp_reference` | `dimensionless` | `1000.0; bounds=(1e-12, 100000.0)` | `fixed` | Reference scale for GDP per capita driver normalization | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_gdp_weight` | `dimensionless` | `0.05; bounds=(0.0, 5.0)` | `fixed` | Weight on GDP per capita driver for land use transition scaling | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_market_weight` | `dimensionless` | `0.05; bounds=(0.0, 5.0)` | `fixed` | Weight on market accessibility driver for land use transition scaling | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_policy_min` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | Minimum policy scaling factor applied to transition rates | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_population_reference` | `dimensionless` | `1000.0; bounds=(1e-12, 100000.0)` | `fixed` | Reference scale for population density driver normalization | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_population_weight` | `dimensionless` | `0.02; bounds=(0.0, 2.0)` | `fixed` | Weight on population density driver for land use transition scaling | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_scale_base` | `dimensionless` | `1.0; bounds=(0.0, 100.0)` | `fixed` | Baseline scaling factor for driver-modulated transition rates | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_transition_rate_max` | `dimensionless` | `0.1; bounds=(0.0, 10.0)` | `fixed` | Maximum driver-based transition rate | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_transition_rate_min` | `dimensionless` | `0.0; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Minimum driver-based transition rate | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.driver_transition_rate_scale` | `dimensionless` | `0.05; bounds=(0.0, 5.0)` | `fixed` | Scaling factor for economic and demographic driver transition rates | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.spatial_autocorrelation_max` | `dimensionless` | `1.0; bounds=(0.0, 100.0)` | `fixed` | Maximum clamp for spatial autocorrelation indicator | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.spatial_autocorrelation_min` | `dimensionless` | `-1.0; bounds=(-100.0, 0.0)` | `fixed` | Minimum clamp for spatial autocorrelation indicator | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.spatial_autocorrelation_offset` | `dimensionless` | `0.1; bounds=(0.0, 10.0)` | `fixed` | Offset applied to leading spatial autocorrelation entry | Source: default; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.transition_probability_default` | `dimensionless` | `0.05; bounds=(0.0, 5.0)` | `fixed` | Default transition probability used to initialize spatial probability fields | Source: default; Hurtt et al. (2020) GMD 13:5425-5464; Verburg et al. (2009) Landscape Ecology 24:1167-1181 |
| `anthroposphere.land_use.numerical.unity_factor` | `dimensionless` | `1.0; bounds=(0.9, 1.1)` | `fixed` | Unity factor for calculations | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.numerical.zero_initial` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Zero initial value | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.barren.decomposition_rate` | `dimensionless` | `0.5; bounds=(0.3, 0.8)` | `fixed` | Low decomposition on barren land | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.barren.denitrification_rate` | `dimensionless` | `0.5; bounds=(0.3, 0.8)` | `fixed` | Low denitrification on barren land | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.barren.erosion_rate` | `kg m-2 yr-1` | `0.2; bounds=(0.1, 0.5)` | `fixed` | High erosion on exposed soil | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.barren.infiltration_rate` | `mm hr-1` | `15.0; bounds=(5.0, 30.0)` | `fixed` | Moderate infiltration on barren land | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.barren.nitrification_rate` | `dimensionless` | `0.5; bounds=(0.3, 0.8)` | `fixed` | Low nitrification on barren land | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.barren.soil_carbon_input` | `kg C m-2 yr-1` | `0.05; bounds=(0.01, 0.1)` | `fixed` | Very low carbon input | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.barren.soil_porosity` | `dimensionless` | `0.3; bounds=(0.2, 0.4)` | `fixed` | Variable porosity on barren land | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.barren.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage on barren land. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.cropland_c3.decomposition_rate` | `dimensionless` | `1.3; bounds=(1.1, 1.6)` | `fixed` | Enhanced decomposition due to tillage | Source: literature; Six et al. (2002) DOI:10.1016/S0038-0717(02)00054-7 |
| `anthroposphere.land_use.soil.cropland_c3.denitrification_rate` | `dimensionless` | `0.8; bounds=(0.6, 1.0)` | `fixed` | Reduced denitrification in well-aerated soil | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Li et al. (2000) J. Geophys. Res. 105, 9149-9167 |
| `anthroposphere.land_use.soil.cropland_c3.erosion_rate` | `kg m-2 yr-1` | `0.1; bounds=(0.05, 0.2)` | `fixed` | High erosion from bare soil periods | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c3.infiltration_rate` | `mm hr-1` | `25.0; bounds=(15.0, 40.0)` | `fixed` | Moderate infiltration in cropland | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Green & Ampt (1911) J. Agric. Sci. 4, 1-24 |
| `anthroposphere.land_use.soil.cropland_c3.nitrification_rate` | `dimensionless` | `1.2; bounds=(1.0, 1.5)` | `fixed` | Enhanced nitrification in tilled soil | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Li et al. (2000) J. Geophys. Res. 105, 9149-9167 |
| `anthroposphere.land_use.soil.cropland_c3.soil_carbon_input` | `kg C m-2 yr-1` | `0.3; bounds=(0.2, 0.5)` | `fixed` | Lower carbon input from crops | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c3.soil_porosity` | `dimensionless` | `0.38; bounds=(0.3, 0.45)` | `fixed` | Reduced porosity from tillage | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Rawls et al. (1982) Trans. ASAE 25, 1316-1320 |
| `anthroposphere.land_use.soil.cropland_c3.tillage_factor` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Full tillage impact. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Farquhar et al. (1980) Planta 149, 78-90; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c4.decomposition_rate` | `dimensionless` | `1.3; bounds=(1.1, 1.6)` | `fixed` | Enhanced decomposition due to tillage | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179 |
| `anthroposphere.land_use.soil.cropland_c4.denitrification_rate` | `dimensionless` | `0.8; bounds=(0.6, 1.0)` | `fixed` | Reduced denitrification in well-aerated soil | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Li et al. (2000) J. Geophys. Res. 105, 9149-9167 |
| `anthroposphere.land_use.soil.cropland_c4.erosion_rate` | `kg m-2 yr-1` | `0.1; bounds=(0.05, 0.2)` | `fixed` | High erosion from bare soil periods | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c4.infiltration_rate` | `mm hr-1` | `25.0; bounds=(15.0, 40.0)` | `fixed` | Moderate infiltration in cropland | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Green & Ampt (1911) J. Agric. Sci. 4, 1-24 |
| `anthroposphere.land_use.soil.cropland_c4.nitrification_rate` | `dimensionless` | `1.2; bounds=(1.0, 1.5)` | `fixed` | Enhanced nitrification in tilled soil | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Li et al. (2000) J. Geophys. Res. 105, 9149-9167 |
| `anthroposphere.land_use.soil.cropland_c4.soil_carbon_input` | `kg C m-2 yr-1` | `0.35; bounds=(0.25, 0.55)` | `fixed` | Slightly higher residue from C4 crops | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.cropland_c4.soil_porosity` | `dimensionless` | `0.38; bounds=(0.3, 0.45)` | `fixed` | Reduced porosity from tillage | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Rawls et al. (1982) Trans. ASAE 25, 1316-1320 |
| `anthroposphere.land_use.soil.cropland_c4.tillage_factor` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Full tillage impact. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.pasture.decomposition_rate` | `dimensionless` | `1.1; bounds=(0.9, 1.3)` | `fixed` | Slightly enhanced decomposition | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.pasture.denitrification_rate` | `dimensionless` | `0.9; bounds=(0.7, 1.1)` | `fixed` | Reduced denitrification | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.pasture.erosion_rate` | `kg m-2 yr-1` | `0.05; bounds=(0.02, 0.1)` | `fixed` | Moderate erosion from overgrazing | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.pasture.infiltration_rate` | `mm hr-1` | `20.0; bounds=(10.0, 35.0)` | `fixed` | Reduced infiltration from compaction | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.pasture.nitrification_rate` | `dimensionless` | `1.1; bounds=(0.9, 1.3)` | `fixed` | Enhanced nitrification from urine patches | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.pasture.soil_carbon_input` | `kg C m-2 yr-1` | `0.4; bounds=(0.25, 0.6)` | `fixed` | Moderate carbon input | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.pasture.soil_porosity` | `dimensionless` | `0.35; bounds=(0.25, 0.45)` | `fixed` | Compacted soil from grazing | Source: literature; Greenwood & McKenzie (2001) DOI:10.1016/S0167-1987(01)00178-0 |
| `anthroposphere.land_use.soil.pasture.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in pasture. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.primary_forest.decomposition_rate` | `dimensionless` | `0.8; bounds=(0.6, 1.0)` | `fixed` | Slower decomposition under forest | Source: literature; Davidson & Janssens (2006) DOI:10.1038/nature04514 |
| `anthroposphere.land_use.soil.primary_forest.denitrification_rate` | `dimensionless` | `1.1; bounds=(0.9, 1.3)` | `fixed` | Enhanced denitrification under forest | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.primary_forest.erosion_rate` | `kg m-2 yr-1` | `0.01; bounds=(0.005, 0.02)` | `fixed` | Very low erosion under forest | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.primary_forest.infiltration_rate` | `mm hr-1` | `50.0; bounds=(30.0, 80.0)` | `fixed` | High infiltration in forest | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.primary_forest.nitrification_rate` | `dimensionless` | `0.9; bounds=(0.7, 1.1)` | `fixed` | Nitrification rate modifier for forest | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.primary_forest.soil_carbon_input` | `kg C m-2 yr-1` | `0.8; bounds=(0.5, 1.2)` | `fixed` | High carbon input from forest litter | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.primary_forest.soil_porosity` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | High porosity in forest soils | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.primary_forest.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in natural forest. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.rangeland.decomposition_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal decomposition rate | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.rangeland.denitrification_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal denitrification rate | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.rangeland.erosion_rate` | `kg m-2 yr-1` | `0.03; bounds=(0.01, 0.06)` | `fixed` | Low to moderate erosion | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.rangeland.infiltration_rate` | `mm hr-1` | `35.0; bounds=(20.0, 50.0)` | `fixed` | Good infiltration in rangeland | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.rangeland.nitrification_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal nitrification rate | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.rangeland.soil_carbon_input` | `kg C m-2 yr-1` | `0.3; bounds=(0.2, 0.5)` | `fixed` | Moderate carbon input from grasses | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.rangeland.soil_porosity` | `dimensionless` | `0.4; bounds=(0.3, 0.5)` | `fixed` | Good porosity in natural grassland | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.rangeland.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in rangeland. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.secondary_forest.decomposition_rate` | `dimensionless` | `0.9; bounds=(0.7, 1.1)` | `fixed` | Intermediate decomposition in secondary forest | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.secondary_forest.denitrification_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal denitrification rate | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.secondary_forest.erosion_rate` | `kg m-2 yr-1` | `0.02; bounds=(0.01, 0.04)` | `fixed` | Low erosion under secondary forest | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.secondary_forest.infiltration_rate` | `mm hr-1` | `40.0; bounds=(25.0, 60.0)` | `fixed` | Good infiltration in secondary forest | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.secondary_forest.nitrification_rate` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Normal nitrification rate | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.secondary_forest.soil_carbon_input` | `kg C m-2 yr-1` | `0.6; bounds=(0.4, 0.9)` | `fixed` | Moderate carbon input | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.secondary_forest.soil_porosity` | `dimensionless` | `0.42; bounds=(0.32, 0.52)` | `fixed` | Good porosity in secondary forest | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.secondary_forest.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in secondary forest. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.urban.decomposition_rate` | `dimensionless` | `0.2; bounds=(0.1, 0.4)` | `fixed` | Very low decomposition in urban soils | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.urban.denitrification_rate` | `dimensionless` | `0.2; bounds=(0.1, 0.4)` | `fixed` | Low denitrification in urban soils | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.urban.erosion_rate` | `kg m-2 yr-1` | `0.001; bounds=(0.0, 0.005)` | `fixed` | Protected surfaces have minimal erosion | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.urban.infiltration_rate` | `mm hr-1` | `2.0; bounds=(0.5, 5.0)` | `fixed` | Very low infiltration in urban areas | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.urban.nitrification_rate` | `dimensionless` | `0.3; bounds=(0.1, 0.5)` | `fixed` | Low nitrification in urban soils | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.urban.soil_carbon_input` | `kg C m-2 yr-1` | `0.05; bounds=(0.01, 0.1)` | `fixed` | Minimal carbon input | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.urban.soil_porosity` | `dimensionless` | `0.15; bounds=(0.1, 0.25)` | `fixed` | Highly compacted urban soils | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.urban.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage in urban areas. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.water.decomposition_rate` | `dimensionless` | `0.0; bounds=(0.0, inf)` | `fixed` | No soil processes over water. The value is a non-negative multiplier of the reference soil process rate (the other land-use classes declare it within [0.1, 1.3]); a negative factor would reverse the process, so the lower bound is 0. | Source: literature; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Coleman & Jenkinson (2014) RothC-26.3 Model Description |
| `anthroposphere.land_use.soil.water.denitrification_rate` | `dimensionless` | `0.0; bounds=(0.0, inf)` | `fixed` | No soil processes over water. The value is a non-negative multiplier of the reference soil process rate (the other land-use classes declare it within [0.1, 1.3]); a negative factor would reverse the process, so the lower bound is 0. | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.water.erosion_rate` | `kg m-2 yr-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No erosion over water. An erosion rate (soil loss) cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.water.infiltration_rate` | `mm hr-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No infiltration over water. An infiltration rate into the soil cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Green & Ampt (1911) J. Agric. Sci. 4, 1-24; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.water.nitrification_rate` | `dimensionless` | `0.0; bounds=(0.0, inf)` | `fixed` | No soil processes over water. The value is a non-negative multiplier of the reference soil process rate (the other land-use classes declare it within [0.1, 1.3]); a negative factor would reverse the process, so the lower bound is 0. | Source: literature; Li et al. (2000) J. Geophys. Res. 105, 9149-9167; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.water.soil_carbon_input` | `kg C m-2 yr-1` | `0.0; bounds=(0.0, inf)` | `fixed` | No soil carbon input over water. A soil carbon input flux (inputs only) cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `anthroposphere.land_use.soil.water.soil_porosity` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No soil over water. Porosity is a volume fraction of the soil and lies in [0, 1]. | Source: literature; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.land_use.soil.water.tillage_factor` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | No tillage over water. The tillage factor is the fraction of full-tillage disturbance (0 no tillage, 1 full tillage) and lies in [0, 1]. | Source: literature; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Hurtt et al. (2011) Climatic Change 109, 117-161 |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | All six outputs use explicit UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{false} )$ with units and source_functor. |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_land_use_transitions.md`.

### References

- Hurtt et al. (2020) DOI:10.5194/gmd-13-5425-2020 (FunctorMetadata)
- IPCC (2019) Climate Change and Land Report (FunctorMetadata)
- Verburg et al. (2009) DOI:10.1007/s10980-009-9347-7 (FunctorMetadata)
- Pielke, R.A. (2002). Mesoscale meteorological modeling. (FunctorMetadata)
- Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming. (FunctorMetadata)
- Hurtt, G.C. et al. (2020). Harmonization of global land-use change and management for LUH2. (certificate)
- Hurtt, G.C. et al. (2011). Harmonization of land-use scenarios for 1500-2100. (certificate)
- Verburg, P.H. et al. (2009). Combining top-down and bottom-up dynamics in land-use modeling. (certificate)
- Anderson, W.J. (1991). Continuous-Time Markov Chains. (certificate)
- Ramirez, H. and Le Gorrec, Y. (2022). Irreversible port-Hamiltonian systems. (certificate)

(class-relic-scientific-anthroposphere-management-coordinator-managementcoordinator)=
## management_coordinator

Unified Management Coordinator

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.management.coordinator.ManagementCoordinator` |
| Source | `relic\scientific\anthroposphere\management\coordinator.py:129` |
| Runtime name | `management_coordinator` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `grid_cell` |
| Variable kind | `diagnostic` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
management_coordinator(name: str | Dict[str, Any] = 'management_coordinator', config: Dict[str, Any] | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | <code>str &#124; typing.Dict[str, typing.Any]</code> | `'management_coordinator'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | Required or optional value consumed by forward(). |
| `soil_nitrogen` | forcing | `kg N m-2` | Required or optional value consumed by forward(). |
| `soil_organic_carbon` | forcing | `kg C m-2` | Required or optional value consumed by forward(). |
| `crop_maturity` | forcing | `1` | Required or optional value consumed by forward(). |
| `aboveground_biomass` | state | `kg m-2` | Required or optional value consumed by forward(). |
| `air_temperature` | forcing | `K` | Required or optional value consumed by forward(). |
| `irrigation_flux` | flux | `kg m-2 s-1` | Required or optional value consumed by forward(). |
| `harvest_flux` | flux | `kg DM m-2 s-1` | Required or optional value consumed by forward(). |
| `grazing_flux` | flux | `kg m-2 s-1` | Required or optional value consumed by forward(). |
| `co2_emission` | flux | `kg m-2 s-1` | Required or optional value consumed by forward(). |
| `field_capacity` | forcing | `m3 m-3` | Required or optional value consumed by forward(). |
| `wilting_point` | forcing | `m3 m-3` | Required or optional value consumed by forward(). |
| `evapotranspiration_flux` | flux | `kg m-2 s-1` | Required or optional value consumed by forward(). |
| `precipitation_flux` | forcing | `kg m-2 s-1` | Required or optional value consumed by forward(). |
| `day_of_year` | forcing | `1` | Required or optional value consumed by forward(). |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `coordination_status` | diagnostic | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `total_irrigation_flux` | diagnostic | `kg m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `total_harvest_flux` | diagnostic | `kg DM m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `total_grazing_flux` | diagnostic | `kg DM m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `total_fire_co2_flux` | diagnostic | `kg CO2 m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `irrigation_demand_flux` | diagnostic | `kg m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `fertilizer_product_flux` | diagnostic | `kg product m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `tillage_intensity` | diagnostic | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.active` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.events_executed` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.irrigation_events_count` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.tillage_events_count` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.harvest_events_count` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.grazing_events_count` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.fire_events_count` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.priority_weights_sum` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.interaction_effects_count` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.conflict_resolution_applied` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.coordinator_mass_balance_error` | output port | `kg DM m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.management_coordinator.canonical_iphs_runtime` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

Multi-objective farm management uses configurable priorities and resource constraints; retained coefficients are calibratable policy heuristics rather than universal laws.

**Canonical RELIC structural form**

$q_{i} = w_{i} ( 1 + s_{i} )$; $p_{i} = q_{i} / \sum_{j} ( q_{j} )$; $H_{m} = \operatorname{diag} ( w_{i} / \sum_{j} ( w_{j} ) )$; $0 \leq U_{r} \leq Q_{r}$ for each resource r.

| Entry | Meaning |
| --- | --- |
| `Pi_h` | ParameterManager-backed priority projection from canonical state inputs. |
| `H_m` | Positive-semidefinite diagonal priority Hodge metric with non-negative weights and positive total. |
| `G_m` | Typed management coordination control port. |
| `resource_normalization` | $c = \sum_{r} R_{r} / Q_{r}$ is dimensionless because each requirement and quota pair has the same unit. |
| `state_advance` | No prognostic state advance is owned by this observer/controller. |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | Five-component management-priority observation basis. |
| `effort_variable` | Normalized priority vector and per-resource quota-normalized scheduling score. |
| `state_advancer` | No prognostic state advance; forward returns explicit UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} ) \mathrm{observations}$ and next-step controls. |
| `runtime_usage` | The simulator commits eight public ports; current diagnostic routing keeps twelve namespaced diagnostics ephemeral. |

### Composition channels and executable PortSpec API

Certificate composition channels: `management_coordination_control_port`

Executable `ports()` owner: `ManagementCoordinator`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `management_coordination_control_port` | `management_priority_effort [1]` | `resource_allocation [1]` | `1` | into_system; management |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.management.coordinator.management_strategy` | `dimensionless` | `'conventional'; bounds=None` | `fixed` | Configured management policy used by the coordinator. | Source: calibrated_management_policy; RELIC configurable management policy; site calibration required |
| `anthroposphere.management.coordinator.coordination_mode` | `dimensionless` | `'adaptive'; bounds=None` | `fixed` | Default coordination mode for management scheduling | Source: management_parameter_database; RELIC Architecture Guidelines v2.0 |
| `anthroposphere.management.coordinator.conflict_resolution_strategy` | `dimensionless` | `'priority_based'; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Default conflict resolution strategy for management coordination | Source: management_parameter_database; FAO (2018) Coordinated farm management guidelines |
| `anthroposphere.management.coordinator.coordination_interval_days` | `day` | `1.0; bounds=(0.0416666667, 365.0)` | `fixed` | Interval between coordinator policy evaluations. | Source: calibrated_management_policy; RELIC configurable management policy; site calibration required |
| `anthroposphere.management.coordinator.planning_horizon_days` | `day` | `30.0; bounds=(1.0, 365.0)` | `fixed` | Planning horizon for management scheduling decisions | Source: management_parameter_database; RELIC Architecture Guidelines v2.0 |
| `anthroposphere.management.coordinator.statistics_history_limit` | `dimensionless` | `5; bounds=(1.0, 1000.0)` | `fixed` | Maximum number of recent coordination results returned by statistics. | Source: calibrated_management_policy; RELIC runtime reporting configuration |
| `anthroposphere.management.coordinator.decision_threshold` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Threshold for applying coordination decisions | Source: management_parameter_database; Decision support systems for farm management |
| `anthroposphere.management.coordinator.high_priority_threshold` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Need score above which irrigation receives high priority. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.candidate_need_threshold` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Minimum need score for generating a candidate event. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.coordination_efficiency` | `dimensionless` | `0.85; bounds=(0.7, 0.95)` | `fixed` | Efficiency scalar for coordination decisions | Source: management_parameter_database; Integrated management optimization literature |
| `anthroposphere.management.coordinator.interaction_factor` | `dimensionless` | `1.0; bounds=(0.5, 1.5)` | `fixed` | Interaction multiplier between management practices | Source: management_parameter_database; Matson et al. (1997) |
| `anthroposphere.management.coordinator.interaction_multiplier_min` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Lower bound for phenomenological management interaction multipliers. | Source: calibrated_management_policy; Phenomenological management diagnostic; site calibration required |
| `anthroposphere.management.coordinator.interaction_multiplier_max` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Upper bound for phenomenological management interaction multipliers. | Source: calibrated_management_policy; Phenomenological management diagnostic; site calibration required |
| `anthroposphere.management.coordinator.maturity_factor_min` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Lower bound for the crop-maturity management multiplier. | Source: calibrated_management_policy; Phenomenological management diagnostic; crop calibration required |
| `anthroposphere.management.coordinator.maturity_factor_max` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Upper bound for the crop-maturity management multiplier. | Source: calibrated_management_policy; Phenomenological management diagnostic; crop calibration required |
| `anthroposphere.management.coordinator.harvest_biomass_threshold` | `kg m-2` | `2.0; bounds=(0.5, 5.0)` | `fixed` | Aboveground biomass threshold for harvest readiness | Source: management_parameter_database; Hay (1995) Harvest Index review |
| `anthroposphere.management.coordinator.score_limits.benefit_max` | `dimensionless` | `1.5; bounds=(1.0, 5.0)` | `fixed` | Upper bound for normalized productivity and soil-health benefit scores. | Source: calibrated_management_policy; Phenomenological management diagnostic; site calibration required |
| `anthroposphere.management.coordinator.event_defaults.tillage_intensity_class` | `dimensionless` | `'moderate'; bounds=None` | `fixed` | Default categorical tillage control intent. | Source: calibrated_management_policy; Configurable management policy; site calibration required |
| `anthroposphere.management.coordinator.priority_weights.productivity` | `dimensionless` | `0.15; bounds=(0.0, 1.0)` | `fixed` | Base weight for productivity objectives | Source: management_parameter_database; Integrated crop management studies |
| `anthroposphere.management.coordinator.priority_weights.environmental` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Base weight for environmental impact objectives | Source: management_parameter_database; Climate risk weighting guidance |
| `anthroposphere.management.coordinator.priority_weights.water` | `dimensionless` | `0.25; bounds=(0.0, 1.0)` | `fixed` | Base weight for water management objectives | Source: management_parameter_database; Allen et al. (1998) |
| `anthroposphere.management.coordinator.priority_weights.nutrients` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Base weight for nutrient objectives | Source: management_parameter_database; Lal (2004) |
| `anthroposphere.management.coordinator.priority_weights.soil_health` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Base weight for soil health objectives | Source: management_parameter_database; Robertson & Swinton (2005) |
| `anthroposphere.management.coordinator.objectives.biodiversity_index.weight` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Objective weight for biodiversity index. | Source: default; Matson et al. (1997) |
| `anthroposphere.management.coordinator.objectives.economic_return.weight` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Objective weight for economic return. | Source: default; Robertson and Swinton (2005) |
| `anthroposphere.management.coordinator.objectives.nutrient_use_efficiency.weight` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Objective weight for nutrient use efficiency. | Source: default; Robertson and Swinton (2005) |
| `anthroposphere.management.coordinator.objectives.soil_carbon_sequestration.weight` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Objective weight for soil carbon sequestration. | Source: default; Lal (2004) |
| `anthroposphere.management.coordinator.objectives.water_use_efficiency.weight` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Objective weight for water use efficiency. | Source: default; Robertson and Swinton (2005) |
| `anthroposphere.management.coordinator.objective_scoring.biodiversity_index.event_penalty` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Penalty per management event in the biodiversity proxy. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.objective_scoring.economic_return.harvest_reward` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Reward per harvest event in the economic proxy. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.objective_scoring.neutral_score` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Neutral score for an unrecognized objective. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.objective_scoring.nutrient_use_efficiency.event_ratio_penalty` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Penalty on the fertilization-to-harvest event ratio. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.objective_scoring.soil_carbon_sequestration.tillage_penalty` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Penalty per tillage event in the soil-carbon proxy. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.objective_scoring.water_use_efficiency.event_penalty` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Penalty per irrigation event in the water-efficiency proxy. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.baseline.nutrient_multiplier` | `dimensionless` | `1.0; bounds=(0.0, 2.0)` | `fixed` | Baseline nutrient-control multiplier restored before each strategy change. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.conservation.nutrient_multiplier` | `dimensionless` | `0.75; bounds=(0.0, 2.0)` | `fixed` | Conservation-policy multiplier applied to the nutrient control port. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.conservation.objective_weight_multiplier` | `dimensionless` | `1.6; bounds=(0.0, 2.0)` | `fixed` | Conservation-policy multiplier for carbon and biodiversity weights. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.organic.biodiversity_weight` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Organic-policy weight for the biodiversity objective. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.organic.nutrient_multiplier` | `dimensionless` | `0.8; bounds=(0.0, 2.0)` | `fixed` | Organic-policy multiplier applied to the nutrient control port. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.organic.soil_carbon_weight` | `dimensionless` | `0.4; bounds=(0.0, 1.0)` | `fixed` | Organic-policy weight for the soil-carbon objective. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.precision.efficiency_weight_multiplier` | `dimensionless` | `1.5; bounds=(0.0, 2.0)` | `fixed` | Precision-policy multiplier for efficiency objective weights. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.precision.nutrient_multiplier` | `dimensionless` | `1.1; bounds=(0.0, 2.0)` | `fixed` | Precision-policy multiplier applied to the nutrient control port. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.productivity.nutrient_multiplier` | `dimensionless` | `1.2; bounds=(0.0, 2.0)` | `fixed` | Productivity-policy multiplier applied to the nutrient control port. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.strategies.productivity.objective_weight_multiplier` | `dimensionless` | `1.5; bounds=(0.0, 2.0)` | `fixed` | Productivity-policy multiplier for economic and water-efficiency weights. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.reference.environment_temperature_optimal` | `K` | `298.15; bounds=(273.15, 320.0)` | `fixed` | Optimal air temperature reference for stress scaling | Source: management_parameter_database; Crop physiological temperature ranges |
| `anthroposphere.management.coordinator.reference.productivity_reference` | `dimensionless` | `1.0; bounds=(0.1, 2.0)` | `fixed` | Reference productivity factor for maturity weighting | Source: management_parameter_database; Crop growth stage indices |
| `anthroposphere.management.coordinator.reference.soil_carbon_reference` | `kg m-2` | `30.0; bounds=(1.0, 80.0)` | `fixed` | Reference soil organic carbon for soil health weighting | Source: management_parameter_database; Lal (2004) |
| `anthroposphere.management.coordinator.reference.soil_moisture_optimal` | `m3 m-3` | `0.35; bounds=(0.05, 0.6)` | `fixed` | Reference soil moisture for irrigation stress scaling | Source: management_parameter_database; FAO irrigation thresholds |
| `anthroposphere.management.coordinator.reference.soil_nitrogen_optimal` | `kg N m-2` | `0.01; bounds=(0.001, 0.05)` | `fixed` | Reference optimal soil nitrogen for stress scaling | Source: management_parameter_database; Agronomic nutrient sufficiency ranges |
| `anthroposphere.management.coordinator.resource_quotas.fuel` | `L` | `200.0; bounds=(0.0, 1000.0)` | `fixed` | Annual fuel-volume quota for the configured scheduling domain. | Source: default; Farm management practice defaults |
| `anthroposphere.management.coordinator.resource_quotas.labor` | `h` | `100.0; bounds=(0.0, 500.0)` | `fixed` | Monthly person-hour quota for the configured scheduling domain. | Source: default; Farm management practice defaults |
| `anthroposphere.management.coordinator.resource_quotas.machinery` | `h` | `50.0; bounds=(0.0, 300.0)` | `fixed` | Monthly machine-hour quota for the configured scheduling domain. | Source: default; Farm management practice defaults |
| `anthroposphere.management.coordinator.resource_quotas.nitrogen` | `g m-2 yr-1` | `20.0; bounds=(0.0, 50.0)` | `fixed` | Annual nitrogen quota for coordinated management operations. | Source: default; IPCC 2006 guidelines |
| `anthroposphere.management.coordinator.resource_quotas.phosphorus` | `g m-2 yr-1` | `5.0; bounds=(0.0, 20.0)` | `fixed` | Annual phosphorus quota for coordinated management operations. | Source: default; IPCC 2006 guidelines |
| `anthroposphere.management.coordinator.resource_quotas.water` | `mm` | `800.0; bounds=(0.1, 10000.0)` | `fixed` | Annual cumulative water allocation for coordinator scheduling. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.fire_management.labor` | `h` | `10.0; bounds=(0.0, 1000.0)` | `fixed` | Person-hours required by one prescribed-fire event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.fire_management.water` | `mm` | `50.0; bounds=(0.0, 1000.0)` | `fixed` | Base prescribed-fire water requirement. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.grazing.labor` | `h` | `0.5; bounds=(0.0, 1000.0)` | `fixed` | Person-hours required by one grazing event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.harvest.fuel` | `L` | `20.0; bounds=(0.0, 1000.0)` | `fixed` | Fuel volume required by one harvest event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.harvest.labor` | `h` | `8.0; bounds=(0.0, 1000.0)` | `fixed` | Person-hours required by one harvest event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.harvest.machinery` | `h` | `4.0; bounds=(0.0, 1000.0)` | `fixed` | Machine-hours required by one harvest event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.irrigation.fuel` | `L` | `2.0; bounds=(0.0, 1000.0)` | `fixed` | Fuel volume required by one irrigation event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.irrigation.labor` | `h` | `1.0; bounds=(0.0, 1000.0)` | `fixed` | Person-hours required by one irrigation event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.irrigation.water` | `mm` | `25.0; bounds=(0.0, 1000.0)` | `fixed` | Base irrigation depth required by one candidate event. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.tillage.fuel` | `L` | `15.0; bounds=(0.0, 1000.0)` | `fixed` | Fuel volume required by one tillage event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.tillage.labor` | `h` | `3.0; bounds=(0.0, 1000.0)` | `fixed` | Person-hours required by one tillage event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.resource_requirements.tillage.machinery` | `h` | `2.0; bounds=(0.0, 1000.0)` | `fixed` | Machine-hours required by one tillage event in the scheduling domain. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.fire.base_need` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Baseline prescribed-fire management need. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.fire.off_season_factor` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Off-season prescribed-fire multiplier. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.fire.peak_factor` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Peak-season prescribed-fire multiplier. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.grazing.biomass_reference` | `kg DM m-2` | `1.0; bounds=(0.01, 20.0)` | `fixed` | Aboveground dry-matter reference for grazing readiness. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.grazing.off_season` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Off-season grazing need score. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.harvest.autumn_factor` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Autumn multiplier for harvest readiness. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.harvest.other_factor` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Non-autumn multiplier for harvest readiness. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.tillage.alternative_peak` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Peak-season alternative-strategy tillage need score. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.tillage.conventional_peak` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Peak-season conventional tillage need score. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.need_scores.tillage.off_season` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Off-season tillage need score. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.event_delay_days.fire_management` | `day` | `14.0; bounds=(0.0, 365.0)` | `fixed` | Configured lead time for prescribed-fire candidates. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.event_delay_days.grazing` | `day` | `1.0; bounds=(0.0, 365.0)` | `fixed` | Configured lead time for grazing candidates. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.event_delay_days.harvest` | `day` | `7.0; bounds=(0.0, 365.0)` | `fixed` | Configured lead time for harvest candidates. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.event_delay_days.irrigation` | `day` | `0.041666666666666664; bounds=(0.0, 365.0)` | `fixed` | Configured lead time for irrigation candidates. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.event_delay_days.tillage` | `day` | `3.0; bounds=(0.0, 365.0)` | `fixed` | Configured lead time for tillage candidates. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.event_duration_hours.fire_management` | `h` | `0.5; bounds=(0.01, 168.0)` | `fixed` | Configured execution duration for one prescribed-fire event. | Source: calibrated_management_policy; Configurable farm operation schedule; site calibration required |
| `anthroposphere.management.coordinator.event_duration_hours.grazing` | `h` | `24.0; bounds=(0.01, 720.0)` | `fixed` | Configured execution duration for one grazing event. | Source: calibrated_management_policy; Configurable farm operation schedule; site calibration required |
| `anthroposphere.management.coordinator.event_duration_hours.harvest` | `h` | `4.0; bounds=(0.01, 168.0)` | `fixed` | Configured execution duration for one harvest event. | Source: calibrated_management_policy; Configurable farm operation schedule; site calibration required |
| `anthroposphere.management.coordinator.event_duration_hours.irrigation` | `h` | `1.0; bounds=(0.01, 168.0)` | `fixed` | Configured execution duration for one irrigation event. | Source: calibrated_management_policy; Configurable farm operation schedule; site calibration required |
| `anthroposphere.management.coordinator.event_duration_hours.tillage` | `h` | `2.0; bounds=(0.01, 168.0)` | `fixed` | Configured execution duration for one tillage event. | Source: calibrated_management_policy; Configurable farm operation schedule; site calibration required |
| `anthroposphere.management.coordinator.season_boundaries.autumn_start_day` | `day` | `244.0; bounds=(1.0, 366.0)` | `fixed` | Northern-hemisphere autumn start in canonical day-of-year coordinates. | Source: calibrated_management_policy; Configurable northern-hemisphere management calendar |
| `anthroposphere.management.coordinator.season_boundaries.spring_start_day` | `day` | `60.0; bounds=(1.0, 366.0)` | `fixed` | Northern-hemisphere spring start in canonical day-of-year coordinates. | Source: calibrated_management_policy; Configurable northern-hemisphere management calendar |
| `anthroposphere.management.coordinator.season_boundaries.summer_start_day` | `day` | `152.0; bounds=(1.0, 366.0)` | `fixed` | Northern-hemisphere summer start in canonical day-of-year coordinates. | Source: calibrated_management_policy; Configurable northern-hemisphere management calendar |
| `anthroposphere.management.coordinator.season_boundaries.winter_start_day` | `day` | `335.0; bounds=(1.0, 366.0)` | `fixed` | Northern-hemisphere winter start in canonical day-of-year coordinates. | Source: calibrated_management_policy; Configurable northern-hemisphere management calendar |
| `anthroposphere.management.coordinator.status.neutral_score` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Coordinator status before any accepted scheduling history. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.status.objective_weight` | `dimensionless` | `0.4; bounds=(0.0, 1.0)` | `fixed` | Weight of objective achievement in coordinator status. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.coordinator.status.success_weight` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Weight of scheduling success in coordinator status. | Source: calibrated_management_policy; Phenomenological management policy; site calibration required |
| `anthroposphere.management.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Minimum clamping value for calculations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.clamp_max` | `dimensionless` | `1.0; bounds=(0.0, 10.0)` | `fixed` | Maximum clamping value for calculations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fertilization.spring_urea_product_mass` | `kg product m-2` | `0.002173913043478261; bounds=(0.0, 0.05)` | `fixed` | Urea product mass applied during the spring event | Source: FAO_fertilizer_product_composition_conversion; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.irrigation.default_application_depth` | `mm` | `25.0; bounds=(0.0, 200.0)` | `fixed` | Default irrigation application depth per event. | Source: management_parameter_database; Allen et al. (1998) |
| `anthroposphere.management.tillage.low_intensity_factor` | `dimensionless` | `0.3; bounds=(0.1, 0.6)` | `fixed` | Factor applied for low intensity tillage operations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fire.default_fire_type` | `dimensionless` | `'prescribed'; bounds=None` | `fixed` | Default fire type label for management events. | Source: management_parameter_database; Andreae and Merlet (2001) |
| `anthroposphere.management.fire.default_fuel_load_fraction` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default fraction of aboveground biomass treated as fuel load. | Source: management_parameter_database; Andreae and Merlet (2001) |
| `anthroposphere.management.fire.fuel_consumption_fraction` | `dimensionless` | `0.85; bounds=(0.7, 0.95)` | `fixed` | Fraction of available fuel consumed | Source: management_parameter_database; Seiler and Crutzen (1980) |
| `anthroposphere.management.grazing.base_efficiency` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Baseline efficiency scalar for grazing management operations | Source: literature_synthesis; Milchunas and Lauenroth (1993) DOI:10.2307/2937150 |
| `anthroposphere.management.grazing.daily_intake_rate` | `kg DM kg-1 body weight day-1` | `0.025; bounds=(0.02, 0.03)` | `fixed` | Daily dry matter intake relative to body weight | Source: management_parameter_database; NRC (2007) |
| `anthroposphere.management.grazing.manure_return_fraction` | `dimensionless` | `0.85; bounds=(0.8, 0.9)` | `fixed` | Fraction of consumed nutrients returned as manure | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.grazing.default_schedule.spring.body_weight` | `kg` | `550.0; bounds=(0.0, 55000.0)` | `fixed` | Average animal body weight for spring grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.spring.stocking_rate` | `animals ha-1` | `1.5; bounds=(0.0, 150.0)` | `fixed` | Stocking rate for spring grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.harvest.default_harvest_index` | `dimensionless` | `0.45; bounds=(0.3, 0.7)` | `fixed` | Default fraction of aboveground biomass harvested | Source: management_parameter_database; Hay (1995) |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, $\mathrm{calendar}_{\mathrm{type}} = \mathrm{None} )$ |
| `state_write_mode` | Every output is UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} )$ with explicit unit and source_functor. |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_management_coordinator.md`.

### References

- Allen et al. (1998), FAO Irrigation and Drainage Paper 56 (FunctorMetadata)
- Lal (2004), Science 304, 1623-1627 (FunctorMetadata)
- Matson et al. (1997), Science 277, 504-509 (FunctorMetadata)
- Robertson and Swinton (2005), Frontiers in Ecology and the Environment 3, 38-46 (FunctorMetadata)
- van der Schaft and Maschke (1995), AEU 49, 362-371 (FunctorMetadata)
- Pielke, R.A. (2002). Mesoscale meteorological modeling. (FunctorMetadata)
- Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming. (FunctorMetadata)
- Allen et al. (1998), FAO Irrigation and Drainage Paper 56. (certificate)
- Lal (2004), Science 304, 1623-1627. (certificate)
- Matson et al. (1997), Science 277, 504-509. (certificate)
- Robertson and Swinton (2005), Frontiers in Ecology and the Environment 3, 38-46. (certificate)
- van der Schaft and Maschke (1995), AEU 49, 362-371. (certificate)

(class-relic-scientific-anthroposphere-management-fertilization-fertilizationmanagement)=
## fertilization_management

Map scheduled product mass to canonical nitrogen and phosphorus ports.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.management.fertilization.FertilizationManagement` |
| Source | `relic\scientific\anthroposphere\management\fertilization.py:112` |
| Runtime name | `fertilization_management` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `port` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
fertilization_management(name: str = 'fertilization_management', config: Optional[Dict[str, Any]] = None, events: Optional[Sequence[FertilizerEvent]] = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'fertilization_management'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Dict[str, Any]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `events` | `Optional[Sequence[FertilizerEvent]]` | `None` | Constructor control for events; see the class runtime contract and ParameterManager table below. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `day_of_year` | forcing | `1` | selects scheduled event overlap without introducing a process-law fallback |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `fertilizer_n_flux` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | elemental nitrogen boundary flux routed to the nitrogen cycling consumer |
| `fertilizer_p_flux` | flux | `g P m-2 s-1` | decided at run time by the returned write mode | elemental phosphorus boundary flux associated with the available phosphorus transaction |
| `available_phosphorus` | state | `g P m-2` | decided at run time by the returned write mode | integrated elemental phosphorus increment [g P m^-2] |
| `diagnostics.fertilization.nitrogen_boundary_residual` | output port | `kg N m-2` | diagnostics | scheduled nitrogen mass minus routed nitrogen mass [kg N m^-2] |
| `diagnostics.fertilization.phosphorus_boundary_residual` | output port | `g P m-2` | diagnostics | scheduled phosphorus mass minus routed phosphorus mass [g P m^-2] |
| `diagnostics.fertilization.canonical_iphs_runtime` | output port | `1` | diagnostics | canonical boundary runtime observation [1] |

### Governing equations and canonical operator

**Literature or process form**

Delta M_q equals the sum over scheduled product mass times elemental mass fraction times exact timestep overlap divided by event duration, where q is N or P.

**Canonical RELIC structural form**

$y_{h} = G_{h} u_{h}$ for declared nitrogen and phosphorus managed-boundary ports.

| Entry | Meaning |
| --- | --- |
| `G_N` | managed nitrogen input incidence from scheduled product mass to the nitrogen boundary flow |
| `G_P` | managed phosphorus input incidence from scheduled product mass to the phosphorus boundary flow |
| `J_h` | not declared because management application is not a reversible internal process |
| `M_h` | not declared because management application is not a dissipative internal process |
| `N_h` | not declared because nutrient transformations are owned by pedosphere reaction Functors |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | No storage potential is claimed; the module maps an external managed event to typed boundary flows. |
| `effort_variable` | Scheduled product mass and composition define the managed boundary input. |
| `state_advancer` | $y_{h} = G_{h} u_{h}$ through the declared nitrogen and phosphorus managed-matter ports. |
| `runtime_usage` | FertilizationManagement.forward computes exact event overlap, elemental boundary mass, boundary flux, and the phosphorus StateManager increment once. |

### Composition channels and executable PortSpec API

Certificate composition channels: `fertilization_nitrogen_input_port`, `fertilization_phosphorus_input_port`

Executable `ports()` owner: `FertilizationManagement`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `fertilization_nitrogen_input_port` | `fertilization_nitrogen_application_potential [1]` | `fertilizer_n_flux [kg N m^-2 s^-1]` | `kg N m^-2 s^-1` | into_system; matter |
| `fertilization_phosphorus_input_port` | `fertilization_phosphorus_application_potential [1]` | `fertilizer_p_flux [g P m^-2 s^-1]` | `g P m^-2 s^-1` | into_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.management.fertilization.spring_urea_product_mass` | `kg product m-2` | `0.002173913043478261; bounds=(0.0, 0.05)` | `fixed` | Urea product mass applied during the spring event | Source: FAO_fertilizer_product_composition_conversion; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fertilization.spring_day` | `day` | `120; bounds=(100.0, 140.0)` | `fixed` | Day of year for spring application (early May) | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fertilization.side_dress_ammonium_nitrate_product_mass` | `kg product m-2` | `0.022857142857142857; bounds=(0.0, 0.1)` | `fixed` | Ammonium nitrate product mass for an 80 kg N ha-1 side-dress event | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fertilization.side_dress_day` | `day` | `180; bounds=(160.0, 200.0)` | `fixed` | Day of year for side-dress application (late June) | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fertilization.spring_superphosphate_product_mass` | `kg product m-2` | `0.01; bounds=(0.005, 0.02)` | `fixed` | Superphosphate product mass for a 100 kg product ha-1 spring event | Source: management_parameter_database; Wang et al. (2010) Biogeosciences 7, 2261-2282; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.management.fertilization.phosphorus_day` | `day` | `100; bounds=(90.0, 120.0)` | `fixed` | Day of year for P application (mid April) | Source: management_parameter_database; Wang et al. (2010) Biogeosciences 7, 2261-2282; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.management.fertilization.urea.n_content` | `dimensionless` | `0.46; bounds=(0.44, 0.48)` | `fixed` | Nitrogen content in urea fertilizer | Source: management_parameter_database; FAO Fertilizer Manual |
| `anthroposphere.management.fertilization.ammonium_nitrate.n_content` | `dimensionless` | `0.35; bounds=(0.33, 0.37)` | `fixed` | Nitrogen content in ammonium nitrate | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fertilization.ammonium_sulfate.n_content` | `dimensionless` | `0.21; bounds=(0.2, 0.22)` | `fixed` | Nitrogen content in ammonium sulfate | Source: management_parameter_database; FAO Fertilizer Manual |
| `anthroposphere.management.fertilization.superphosphate.p2o5_mass_fraction` | `dimensionless` | `0.2; bounds=(0.16, 0.22)` | `fixed` | P2O5 mass fraction in superphosphate fertilizer product | Source: management_parameter_database; FAO Fertilizer Manual |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem values are returned through the existing FunctorBase and StateManager path |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_management_fertilization.md`.

### References

- FAO (2000) Fertilizers and Their Use, 4th edition (FunctorMetadata)
- E3SM/ELM Flow of Agriculture Nitrogen design documentation (FunctorMetadata)
- CLM/CTSM FERT_TO_SMINN and NFERTILIZATION history fields (FunctorMetadata)
- BIPM (2019) SI Brochure, 9th edition (FunctorMetadata)
- NIST and IUPAC standard atomic weights (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- FAO (2000). Fertilizers and Their Use, 4th edition. (certificate)
- E3SM Project. ELM Flow of Agriculture Nitrogen design documentation. (certificate)
- CLM and CTSM history fields FERT_TO_SMINN and NFERTILIZATION. (certificate)
- BIPM (2019). SI Brochure, 9th edition. (certificate)
- IUPAC standard atomic weights. (certificate)

(class-relic-scientific-anthroposphere-management-fire-management-firemanagement)=
## fire_management

Fire Management Functor

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.management.fire_management.FireManagement` |
| Source | `relic\scientific\anthroposphere\management\fire_management.py:113` |
| Runtime name | `fire_management` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
fire_management(name: str = 'fire_management', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'fire_management'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

The runtime metadata below is introspected from the documented configuration `seed=7`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `aboveground_biomass` | state | `kg m-2` | Required or optional value consumed by forward(). |
| `litter_mass` | forcing | `kg DM m-2` | Required or optional value consumed by forward(). |
| `fuel_moisture` | forcing | `1` | Required or optional value consumed by forward(). |
| `day_of_year` | forcing | `1` | Required or optional value consumed by forward(). |
| `days_since_fire` | state | `day` | Required or optional value consumed by forward(). |
| `temperature` | forcing | `K` | Required or optional value consumed by forward(). |
| `fuel_load` | forcing | `kg DM m-2` | Required or optional value consumed by forward(). |
| `burn_fraction` | forcing | `1` | Required or optional value consumed by forward(). |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `burned_fraction` | state | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `co2_emission` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `co_emission` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `fire_ch4_emission` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | fire-specific atmospheric methane mass flux [kg CH4 m^-2 s^-1] |
| `n2o_emission` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `ash_deposition` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `days_since_fire` | state | `day` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `fire_fuel_carbon_input` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `carbon_loss` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `combustion_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.fire.events_today` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.fire.event_support_seconds` | output port | `s` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.fire.carbon_availability_residual` | output port | `kg C m^-2 s^-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.fire_management.canonical_iphs_runtime` | output port | `1` | diagnostics | canonical IPHS runtime flag [1] |

### Governing equations and canonical operator

**Literature or process form**

Biomass burning emissions are represented by fuel burned times species emission factors with moisture, combustion completeness, and suppression controls.

**Canonical RELIC structural form**

$\dot{x} = N_{h} r_{h} ( x_{h} , \theta ) + G_{h} u_{\mathrm{fire}}$, with $r_{h} \coloneqq r_{\mathrm{fire}}$; $H_{\mathrm{fire}} = \operatorname{diag} ( w_{i} / \sum_{j} w_{j} )$, $w_{i} \geq 0 .$

| Entry | Meaning |
| --- | --- |
| `N_h` | stoichiometric map from fuel combustion to atmospheric species and ash outputs |
| `r_h` | ParameterManager-backed combustion, emission-factor, ash, and carbon-loss rate tensor passed to advance_reaction_network |
| `H_fire` | PSD combustion mobility built by build_hodge_contract |
| `G_h` | fire_combustion_management_port ignition and suppression input map |
| `O_h` | canonical runtime diagnostic output |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | $G_{\mathrm{fire}} ( x_{h} ) = 0.5 \cdot x_{h}^{\mathsf{T}} \cdot H_{\mathrm{fire}} \cdot x_{h}$ for combustible fuel storage and open emission ports |
| `effort_variable` | $\mu_{\mathrm{fire}} = \frac{\mathrm{d}G_{\mathrm{fire}}}{\mathrm{d}x_{h}}$; combustion completeness, fuel moisture, and emission factors define $r_{h}$ entries |
| `state_advancer` | forward evaluates explicit $\dot{x} = N_{h} r_{h}$ through advance_reaction_network in fire combustion output assembly |
| `runtime_usage` | _assemble_fire_reaction_outputs calls advance_reaction_network with declared $N_{h}$ and $r_{h}$ for combustion, emissions, ash, and carbon loss |
| `explicit_stoichiometric_usage` | _assemble_fire_reaction_outputs materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |
| `event_support` | each calendar occurrence executes once; $r_{h}$ rates are the event-integrated masses divided by the executing step dt ($\Delta_{t,\mathrm{event}} = \mathrm{d}t , \mathrm{reported} \mathrm{as} \mathtt{diagnostics.fire.event\_support\_seconds} )$, so rate * dt returns the event mass for hourly, daily and multi-day steps |

### Composition channels and executable PortSpec API

Certificate composition channels: `fire_combustion_management_port`

Executable `ports()` owner: `FireManagement`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `fire_combustion_management_port` | `fire_management_effort [1]` | `combustion_flux [kg m^-2 s^-1]` | `kg m^-2 s^-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.management.temperature` | `K` | `288.15; bounds=(273.15, 313.15)` | `fixed` | Reference air temperature T_ref of the fire moisture-factor temperature effect clamp((T - T_ref) / T_scale) and air temperature of the default prescribed fire events; not a default of the fire temperature input, which has none | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Minimum clamping value for calculations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.clamp_max` | `dimensionless` | `1.0; bounds=(0.0, 10.0)` | `fixed` | Maximum clamping value for calculations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fire.base_efficiency` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Baseline efficiency scalar for fire management operations | Source: literature_synthesis; van der Werf et al. (2017) DOI:10.5194/essd-9-697-2017 |
| `anthroposphere.management.fire.combustion_completeness` | `dimensionless` | `0.9; bounds=(0.85, 0.95)` | `fixed` | Combustion completeness factor | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fire.combustion_efficiency` | `dimensionless` | `0.9; bounds=(0.8, 1.0)` | `fixed` | Combustion efficiency (CO2 fraction of C emissions) | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.emission_factor_co2` | `g kg-1 DM` | `1680.0; bounds=(1600.0, 1750.0)` | `fixed` | CO2 emission factor for biomass burning | Source: management_parameter_database; Keeling et al. (1976) Tellus 28, 538-551; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.management.fire.emission_factor_co` | `g kg-1 DM` | `92.0; bounds=(60.0, 130.0)` | `fixed` | CO emission factor for biomass burning | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fire.emission_factor_ch4` | `g kg-1 DM` | `4.7; bounds=(2.0, 7.0)` | `fixed` | CH4 emission factor for biomass burning | Source: management_parameter_database; Walter & Heimann (2000) Glob. Biogeochem. Cycles 14, 745-765; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.management.fire.ash_fraction` | `dimensionless` | `0.05; bounds=(0.0, 0.2)` | `fixed` | Fraction of combusted biomass deposited as ash | Source: management_parameter_database; Certini (2005) DOI:10.1007/s00442-004-1788-8 |
| `anthroposphere.management.fire.biomass_carbon_fraction` | `dimensionless` | `0.45; bounds=(0.3, 0.6)` | `fixed` | Carbon fraction of dry biomass (kg C per kg DM) | Source: management_parameter_database; Seiler and Crutzen (1980) |
| `anthroposphere.management.fire.fire_frequency` | `year-1` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Mean annual fire occurrence frequency used for scheduling | Source: literature_synthesis; van der Werf et al. (2017) DOI:10.5194/essd-9-697-2017 |
| `anthroposphere.management.fire.fire_probability_scale` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Scale factor for wildfire ignition probability | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.moisture_reduction_coefficient` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `fixed` | Moisture reduction coefficient for fire spread calculations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fire.moisture_temp_base_weight` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Base weight for moisture temperature coupling | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fire.moisture_temp_scale_weight` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Temperature scaling weight for moisture coupling | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fire.temperature_scale_range` | `K` | `25.0; bounds=(1.0, 100.0)` | `fixed` | Temperature normalization range for moisture scaling | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.fire.wind_speed_threshold` | `m s-1` | `10.0; bounds=(0.0, 30.0)` | `fixed` | Wind speed threshold affecting fire spread | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.schedule_window_days` | `day` | `5.0; bounds=(0.0, 30.0)` | `fixed` | Window size in days for matching scheduled fire events | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.prescribed.burned_fraction` | `dimensionless` | `0.3; bounds=(0.0, 30.0)` | `fixed` | Fraction of area burned in prescribed fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.prescribed.date` | `day` | `100; bounds=(0.0, 10000.0)` | `fixed` | Day of year for prescribed burn | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.prescribed.fuel_moisture` | `dimensionless` | `0.15; bounds=(0.0, 15.0)` | `fixed` | Fuel moisture content for prescribed fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.prescribed.suppression_effectiveness` | `dimensionless` | `0.9; bounds=(0.0, 90.0)` | `fixed` | Suppression effectiveness for prescribed fire (control) | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.prescribed.wind_speed` | `m s-1` | `3.0; bounds=(0.0, 300.0)` | `fixed` | Wind speed during prescribed fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.wildfire.burned_fraction` | `dimensionless` | `0.05; bounds=(0.0, 5.0)` | `fixed` | Fraction of area burned in wildfire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.wildfire.date` | `day` | `240; bounds=(0.0, 24000.0)` | `fixed` | Typical day of year for wildfire season peak | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.wildfire.fuel_moisture` | `dimensionless` | `0.08; bounds=(0.0, 8.0)` | `fixed` | Fuel moisture content for wildfire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.wildfire.suppression_effectiveness` | `dimensionless` | `0.3; bounds=(0.0, 30.0)` | `fixed` | Suppression effectiveness for wildfire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.default_schedule.wildfire.wind_speed` | `m s-1` | `8.0; bounds=(0.0, 800.0)` | `fixed` | Wind speed during wildfire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.intensity_factor.extreme` | `dimensionless` | `1.4; bounds=(0.2, 2.0)` | `fixed` | Combustion completeness multiplier for extreme intensity fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.intensity_factor.high` | `dimensionless` | `1.2; bounds=(0.2, 2.0)` | `fixed` | Combustion completeness multiplier for high intensity fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.intensity_factor.low` | `dimensionless` | `0.8; bounds=(0.2, 2.0)` | `fixed` | Combustion completeness multiplier for low intensity fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.intensity_factor.moderate` | `dimensionless` | `1.0; bounds=(0.2, 2.0)` | `fixed` | Combustion completeness multiplier for moderate intensity fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.agricultural.aboveground_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of aboveground biomass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.agricultural.ch4_ef` | `g kg-1` | `2.7; bounds=(0.0, 270.0)` | `fixed` | CH4 emission factor for agricultural fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.agricultural.co2_ef` | `g kg-1` | `1515.0; bounds=(0.0, 151500.0)` | `fixed` | CO2 emission factor for agricultural fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.agricultural.co_ef` | `g kg-1` | `92.0; bounds=(0.0, 9200.0)` | `fixed` | CO emission factor for agricultural fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.agricultural.combustion_completeness` | `dimensionless` | `0.95; bounds=(0.0, 95.0)` | `fixed` | Combustion completeness for agricultural fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.agricultural.fuel_load_factor` | `dimensionless` | `0.8; bounds=(0.0, 80.0)` | `fixed` | Fuel load factor for agricultural fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.agricultural.litter_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of litter mass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.agricultural.n2o_ef` | `g kg-1` | `0.21; bounds=(0.0, 21.0)` | `fixed` | N2O emission factor for agricultural fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_boreal.aboveground_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of aboveground biomass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_boreal.ch4_ef` | `g kg-1` | `5.1; bounds=(0.0, 509.99999999999994)` | `fixed` | CH4 emission factor for forest boreal fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_boreal.co2_ef` | `g kg-1` | `1489.0; bounds=(0.0, 148900.0)` | `fixed` | CO2 emission factor for forest boreal fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_boreal.co_ef` | `g kg-1` | `127.0; bounds=(0.0, 12700.0)` | `fixed` | CO emission factor for forest boreal fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_boreal.combustion_completeness` | `dimensionless` | `0.75; bounds=(0.0, 75.0)` | `fixed` | Combustion completeness for forest boreal fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_boreal.fuel_load_factor` | `dimensionless` | `2.0; bounds=(0.0, 200.0)` | `fixed` | Fuel load factor for forest boreal fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_boreal.litter_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of litter mass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_boreal.n2o_ef` | `g kg-1` | `0.26; bounds=(0.0, 26.0)` | `fixed` | N2O emission factor for forest boreal fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_temperate.aboveground_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of aboveground biomass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_temperate.ch4_ef` | `g kg-1` | `4.7; bounds=(0.0, 470.0)` | `fixed` | CH4 emission factor for forest temperate fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_temperate.co2_ef` | `g kg-1` | `1637.0; bounds=(0.0, 163700.0)` | `fixed` | CO2 emission factor for forest temperate fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_temperate.co_ef` | `g kg-1` | `88.0; bounds=(0.0, 8800.0)` | `fixed` | CO emission factor for forest temperate fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_temperate.combustion_completeness` | `dimensionless` | `0.8; bounds=(0.0, 80.0)` | `fixed` | Combustion completeness for forest temperate fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_temperate.fuel_load_factor` | `dimensionless` | `1.8; bounds=(0.0, 180.0)` | `fixed` | Fuel load factor for forest temperate fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_temperate.litter_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of litter mass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_temperate.n2o_ef` | `g kg-1` | `0.26; bounds=(0.0, 26.0)` | `fixed` | N2O emission factor for forest temperate fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_tropical.aboveground_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of aboveground biomass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_tropical.ch4_ef` | `g kg-1` | `5.7; bounds=(0.0, 570.0)` | `fixed` | CH4 emission factor for forest tropical fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_tropical.co2_ef` | `g kg-1` | `1643.0; bounds=(0.0, 164300.0)` | `fixed` | CO2 emission factor for forest tropical fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_tropical.co_ef` | `g kg-1` | `93.0; bounds=(0.0, 9300.0)` | `fixed` | CO emission factor for forest tropical fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_tropical.combustion_completeness` | `dimensionless` | `0.7; bounds=(0.0, 70.0)` | `fixed` | Combustion completeness for forest tropical fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_tropical.fuel_load_factor` | `dimensionless` | `2.5; bounds=(0.0, 250.0)` | `fixed` | Fuel load factor for forest tropical fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_tropical.litter_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of litter mass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.forest_tropical.n2o_ef` | `g kg-1` | `0.2; bounds=(0.0, 20.0)` | `fixed` | N2O emission factor for forest tropical fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.grassland.aboveground_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of aboveground biomass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.grassland.ch4_ef` | `g kg-1` | `2.3; bounds=(0.0, 229.99999999999997)` | `fixed` | CH4 emission factor for grassland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.grassland.co2_ef` | `g kg-1` | `1650.0; bounds=(0.0, 165000.0)` | `fixed` | CO2 emission factor for grassland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.grassland.co_ef` | `g kg-1` | `65.0; bounds=(0.0, 6500.0)` | `fixed` | CO emission factor for grassland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.grassland.combustion_completeness` | `dimensionless` | `0.9; bounds=(0.0, 90.0)` | `fixed` | Combustion completeness for grassland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.grassland.fuel_load_factor` | `dimensionless` | `1.0; bounds=(0.0, 100.0)` | `fixed` | Fuel load factor for grassland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.grassland.litter_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of litter mass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.grassland.n2o_ef` | `g kg-1` | `0.21; bounds=(0.0, 21.0)` | `fixed` | N2O emission factor for grassland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.peat.aboveground_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of aboveground biomass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.peat.ch4_ef` | `g kg-1` | `21.0; bounds=(0.0, 2100.0)` | `fixed` | CH4 emission factor for peat fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.peat.co2_ef` | `g kg-1` | `1703.0; bounds=(0.0, 170300.0)` | `fixed` | CO2 emission factor for peat fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.peat.co_ef` | `g kg-1` | `210.0; bounds=(0.0, 21000.0)` | `fixed` | CO emission factor for peat fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.peat.combustion_completeness` | `dimensionless` | `0.6; bounds=(0.0, 60.0)` | `fixed` | Combustion completeness for peat fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.peat.fuel_load_factor` | `dimensionless` | `5.0; bounds=(0.0, 500.0)` | `fixed` | Fuel load factor for peat fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.peat.litter_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of litter mass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.peat.n2o_ef` | `g kg-1` | `0.26; bounds=(0.0, 26.0)` | `fixed` | N2O emission factor for peat fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.shrubland.aboveground_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of aboveground biomass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.shrubland.ch4_ef` | `g kg-1` | `4.7; bounds=(0.0, 470.0)` | `fixed` | CH4 emission factor for shrubland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.shrubland.co2_ef` | `g kg-1` | `1686.0; bounds=(0.0, 168600.0)` | `fixed` | CO2 emission factor for shrubland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.shrubland.co_ef` | `g kg-1` | `107.0; bounds=(0.0, 10700.0)` | `fixed` | CO emission factor for shrubland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.shrubland.combustion_completeness` | `dimensionless` | `0.85; bounds=(0.0, 85.0)` | `fixed` | Combustion completeness for shrubland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.shrubland.fuel_load_factor` | `dimensionless` | `1.2; bounds=(0.0, 120.0)` | `fixed` | Fuel load factor for shrubland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.shrubland.litter_fraction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Fraction of litter mass available as fuel | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.fire.vegetation_properties.shrubland.n2o_ef` | `g kg-1` | `0.26; bounds=(0.0, 26.0)` | `fixed` | N2O emission factor for shrubland fire | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, $\mathrm{calendar}_{\mathrm{type}} = \mathrm{None} , \cdot \cdot \mathrm{kwargs} )$ |
| `state_write_mode` | outputs are returned through existing FunctorBase/StateManager path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_management_fire.md`.

### References

- Bond & Keeley (2005) DOI:10.1016/j.tree.2005.04.025 (FunctorMetadata)
- Certini (2005) DOI:10.1007/s00442-004-1788-8 (FunctorMetadata)
- Seiler & Crutzen (1980) DOI:10.1007/BF00137988 (FunctorMetadata)
- Andreae & Merlet (2001) DOI:10.1029/2000GB001382 (FunctorMetadata)
- van der Werf et al. (2017) DOI:10.5194/essd-9-697-2017 (FunctorMetadata)
- Seiler, W. and Crutzen, P.J. (1980). Estimates of gross and net fluxes of carbon between the biosphere and atmosphere. (certificate)
- Andreae, M.O. and Merlet, P. (2001). Trace gas emissions from biomass burning. (certificate)
- Akagi, S.K. et al. (2011). Compilation of emission factors for open biomass burning. (certificate)
- van der Werf, G.R. et al. (2017). Global fire emissions 1997-2016. (certificate)
- Ramirez, H. and Le Gorrec, Y. (2022). Irreversible port-Hamiltonian systems. (certificate)

(class-relic-scientific-anthroposphere-management-grazing-grazingmanagement)=
## grazing_management

Grazing Management Functor

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.management.grazing.GrazingManagement` |
| Source | `relic\scientific\anthroposphere\management\grazing.py:107` |
| Runtime name | `grazing_management` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
grazing_management(name: str = 'grazing_management', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'grazing_management'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `aboveground_biomass` | state | `kg m-2` | Required or optional value consumed by forward(). |
| `vegetation_quality` | forcing | `1` | Required or optional value consumed by forward(). |
| `day_of_year` | forcing | `1` | Required or optional value consumed by forward(). |
| `soil_moisture` | state | `m3 m-3` | Required or optional value consumed by forward(). |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `aboveground_biomass` | state | `kg m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `grazing_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `dung_c_flux` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `dung_n_flux` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | fecal organic nitrogen return boundary consumed only by litterfall_interface_mapper |
| `urine_n_flux` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | separate urine mineral nitrogen boundary consumed only by the nitrogen cycling route |
| `livestock_carbon_intake_flux` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `livestock_nitrogen_intake_flux` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `livestock_carbon_export_flux` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `livestock_nitrogen_export_flux` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `trampling_damage` | diagnostic | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `grazing_active` | diagnostic | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `current_stocking_rate` | diagnostic | `animals ha-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `utilization_rate` | diagnostic | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `cumulative_intake` | state | `kg m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `days_grazed` | state | `day` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.grazing_pressure` | output port | `animals ha-1` | diagnostics | effective stocking density weighted by forage quality [animals ha^-1] |
| `diagnostics.grazing_management.canonical_iphs_runtime` | output port | `1` | diagnostics | canonical IPHS runtime flag [1] |

### Governing equations and canonical operator

**Literature or process form**

Livestock grazing intake depends on body weight, stocking density, forage quality, and biomass availability; intake carbon and nitrogen are partitioned among excreta and managed livestock exports.

**Canonical RELIC structural form**

$\dot{x} = N_{h} r_{h} ( x_{h} , \theta ) + G_{h} u_{\mathrm{grazing}}$, with $r_{h} \coloneqq r_{\mathrm{grazing}}$; $H_{\mathrm{grazing}} = \operatorname{diag} ( w_{i} / \sum_{j} w_{j} )$, $w_{i} \geq 0 .$

| Entry | Meaning |
| --- | --- |
| `N_h` | non-strict open-port map from biomass removal to carbon, nitrogen, excreta, livestock export, and trampling channels |
| `r_h` | ParameterManager-backed grazing intake, excreta return, livestock export, and trampling rate tensor passed to advance_reaction_network |
| `H_grazing` | PSD removal-return mobility built by build_hodge_contract |
| `G_h` | grazing_biomass_return_port stocking input map |
| `O_h` | canonical runtime and biomass-balance diagnostics |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | $G_{\mathrm{grazing}} ( x_{h} ) = 0.5 \cdot x_{h}^{\mathsf{T}} \cdot H_{\mathrm{grazing}} \cdot x_{h}$ for grazable biomass and return ports |
| `effort_variable` | $\mu_{\mathrm{grazing}} = \frac{\mathrm{d}G_{\mathrm{grazing}}}{\mathrm{d}x_{h}}$; stocking pressure, forage quality, and livestock traits define $r_{h}$ entries |
| `state_advancer` | forward evaluates explicit $\dot{x} = N_{h} r_{h}$ through advance_reaction_network in grazing biomass-return output assembly |
| `runtime_usage` | _assemble_grazing_reaction_outputs calls advance_reaction_network with declared $N_{h}$ and $r_{h}$ for intake, dung, urine, livestock export, and trampling ports |
| `explicit_stoichiometric_usage` | _assemble_grazing_reaction_outputs materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `grazing_biomass_return_port`, `grazing_dung_boundary`, `grazing_urine_boundary`

Executable `ports()` owner: `GrazingManagement`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `grazing_biomass_return_port` | `grazing_stocking_effort [animals ha^-1]` | `grazing_flux [kg m^-2 s^-1]` | `kg m^-2 s^-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.management.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Minimum clamping value for calculations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.clamp_max` | `dimensionless` | `1.0; bounds=(0.0, 10.0)` | `fixed` | Maximum clamping value for calculations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.grazing.base_efficiency` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Baseline efficiency scalar for grazing management operations | Source: literature_synthesis; Milchunas and Lauenroth (1993) DOI:10.2307/2937150 |
| `anthroposphere.management.grazing.digestibility_coefficient` | `dimensionless` | `0.65; bounds=(0.55, 0.75)` | `fixed` | Feed digestibility coefficient | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.grazing.selectivity_factor` | `dimensionless` | `0.8; bounds=(0.5, 1.0)` | `fixed` | Grazing selectivity factor adjusting intake for forage quality | Source: management_parameter_database; Hodgson (1990) |
| `anthroposphere.management.grazing.max_utilization` | `dimensionless` | `0.6; bounds=(0.4, 0.8)` | `fixed` | Maximum allowable fraction of aboveground biomass removed by grazing | Source: management_parameter_database; Allen et al. (2011) |
| `anthroposphere.management.grazing.forage_carbon_fraction` | `dimensionless` | `0.45; bounds=(0.35, 0.5)` | `fixed` | Carbon mass fraction of grazed forage dry matter | Source: management_parameter_database; IPCC (2019) |
| `anthroposphere.management.grazing.functional_response_scalar` | `ha kg-1` | `0.001; bounds=(0.0001, 0.005)` | `fixed` | Scalar k for intake functional response I = 1 - exp(-k * biomass_kg_ha) | Source: management_parameter_database; Hodgson (1990) |
| `anthroposphere.management.grazing.n_retention` | `dimensionless` | `0.15; bounds=(0.05, 0.3)` | `fixed` | Fraction of nitrogen intake retained by livestock | Source: management_parameter_database; Mottet et al. (2017) |
| `anthroposphere.management.grazing.dung_partition` | `dimensionless` | `0.4; bounds=(0.3, 0.5)` | `fixed` | Fraction of excreted nitrogen routed to dung | Source: management_parameter_database; IPCC (2019) |
| `anthroposphere.management.grazing.urine_partition` | `dimensionless` | `0.6; bounds=(0.5, 0.7)` | `fixed` | Fraction of excreted nitrogen routed to urine | Source: management_parameter_database; IPCC (2019) |
| `anthroposphere.management.grazing.trampling_factor` | `dimensionless` | `0.05; bounds=(0.01, 0.1)` | `fixed` | Base vegetation damage factor from trampling | Source: management_parameter_database; Greenwood & McKenzie (2001) |
| `anthroposphere.management.grazing.trampling_moisture_threshold` | `dimensionless` | `0.2; bounds=(0.05, 0.3)` | `fixed` | Soil moisture threshold where trampling damage increases | Source: management_parameter_database; Greenwood & McKenzie (2001) |
| `anthroposphere.management.grazing.trampling_moisture_scale` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `fixed` | Moisture scaling factor for trampling damage | Source: management_parameter_database; Greenwood & McKenzie (2001) |
| `anthroposphere.management.grazing.trampling_stocking_normalization` | `animals ha-1` | `2.0; bounds=(0.5, 5.0)` | `fixed` | Stocking rate normalization for trampling response | Source: management_parameter_database; Greenwood & McKenzie (2001) |
| `anthroposphere.management.grazing.livestock.buffalo.digestibility` | `dimensionless` | `0.62; bounds=(0.0, 1.0)` | `fixed` | Digestibility for buffalo | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.buffalo.intake_rate` | `dimensionless` | `2.3; bounds=(0.0, 229.99999999999997)` | `fixed` | Intake rate for buffalo (% body weight) | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.buffalo.metabolism_factor` | `dimensionless` | `0.95; bounds=(0.0, 95.0)` | `fixed` | Metabolic rate factor for buffalo | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.buffalo.n_content` | `kg kg-1` | `0.03; bounds=(0.0, 1.0)` | `fixed` | Nitrogen content of buffalo biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.buffalo.p_content` | `kg kg-1` | `0.006; bounds=(0.0, 0.6)` | `fixed` | Phosphorus content of buffalo biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.buffalo.trampling_impact` | `dimensionless` | `1.2; bounds=(0.0, 120.0)` | `fixed` | Trampling impact factor for buffalo | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.cattle.digestibility` | `dimensionless` | `0.65; bounds=(0.0, 1.0)` | `fixed` | Digestibility for cattle | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.cattle.intake_rate` | `dimensionless` | `2.5; bounds=(0.0, 250.0)` | `fixed` | Intake rate for cattle (% body weight) | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.cattle.metabolism_factor` | `dimensionless` | `1.0; bounds=(0.0, 100.0)` | `fixed` | Metabolic rate factor for cattle | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.cattle.n_content` | `kg kg-1` | `0.025; bounds=(0.0, 1.0)` | `fixed` | Nitrogen content of cattle biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.cattle.p_content` | `kg kg-1` | `0.005; bounds=(0.0, 0.5)` | `fixed` | Phosphorus content of cattle biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.cattle.trampling_impact` | `dimensionless` | `0.8; bounds=(0.0, 80.0)` | `fixed` | Trampling impact factor for cattle | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.goat.digestibility` | `dimensionless` | `0.68; bounds=(0.0, 1.0)` | `fixed` | Digestibility for goat | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.goat.intake_rate` | `dimensionless` | `3.5; bounds=(0.0, 350.0)` | `fixed` | Intake rate for goat (% body weight) | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.goat.metabolism_factor` | `dimensionless` | `1.1; bounds=(0.0, 110.00000000000001)` | `fixed` | Metabolic rate factor for goat | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.goat.n_content` | `kg kg-1` | `0.025; bounds=(0.0, 1.0)` | `fixed` | Nitrogen content of goat biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.goat.p_content` | `kg kg-1` | `0.005; bounds=(0.0, 0.5)` | `fixed` | Phosphorus content of goat biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.goat.trampling_impact` | `dimensionless` | `0.4; bounds=(0.0, 40.0)` | `fixed` | Trampling impact factor for goat | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.horse.digestibility` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Digestibility for horse | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.horse.intake_rate` | `dimensionless` | `2.0; bounds=(0.0, 200.0)` | `fixed` | Intake rate for horse (% body weight) | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.horse.metabolism_factor` | `dimensionless` | `0.9; bounds=(0.0, 90.0)` | `fixed` | Metabolic rate factor for horse | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.horse.n_content` | `kg kg-1` | `0.02; bounds=(0.0, 1.0)` | `fixed` | Nitrogen content of horse biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.horse.p_content` | `kg kg-1` | `0.004; bounds=(0.0, 0.4)` | `fixed` | Phosphorus content of horse biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.horse.trampling_impact` | `dimensionless` | `1.0; bounds=(0.0, 100.0)` | `fixed` | Trampling impact factor for horse | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.sheep.digestibility` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Digestibility for sheep | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.sheep.intake_rate` | `dimensionless` | `3.0; bounds=(0.0, 300.0)` | `fixed` | Intake rate for sheep (% body weight) | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.sheep.metabolism_factor` | `dimensionless` | `1.2; bounds=(0.0, 120.0)` | `fixed` | Metabolic rate factor for sheep | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.sheep.n_content` | `kg kg-1` | `0.025; bounds=(0.0, 1.0)` | `fixed` | Nitrogen content of sheep biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.sheep.p_content` | `kg kg-1` | `0.005; bounds=(0.0, 0.5)` | `fixed` | Phosphorus content of sheep biomass | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.livestock.sheep.trampling_impact` | `dimensionless` | `0.3; bounds=(0.0, 30.0)` | `fixed` | Trampling impact factor for sheep | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.spring.body_weight` | `kg` | `550.0; bounds=(0.0, 55000.0)` | `fixed` | Average animal body weight for spring grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.spring.duration` | `days` | `150.0; bounds=(0.0, 15000.0)` | `fixed` | Duration of spring grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.spring.start_date` | `day` | `120; bounds=(0.0, 12000.0)` | `fixed` | Start day of year for spring grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.spring.stocking_rate` | `animals ha-1` | `1.5; bounds=(0.0, 150.0)` | `fixed` | Stocking rate for spring grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.spring.supplement_rate` | `kg day-1` | `2.0; bounds=(0.0, 200.0)` | `fixed` | Supplemental feed rate for spring grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.winter.body_weight` | `kg` | `550.0; bounds=(0.0, 55000.0)` | `fixed` | Average animal body weight for winter grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.winter.duration` | `days` | `60.0; bounds=(0.0, 6000.0)` | `fixed` | Duration of winter grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.winter.start_date` | `day` | `300; bounds=(0.0, 30000.0)` | `fixed` | Start day of year for winter grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.winter.stocking_rate` | `animals ha-1` | `0.8; bounds=(0.0, 80.0)` | `fixed` | Stocking rate for winter grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.grazing.default_schedule.winter.supplement_rate` | `kg day-1` | `8.0; bounds=(0.0, 800.0)` | `fixed` | Supplemental feed rate for winter grazing | Source: default; FAO (2000) Fertilizers and Their Use, 4th edition |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | aboveground_biomass, cumulative_intake, and days_grazed use integrated ACCUMULATE updates; instantaneous ports use REPLACE |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_management_grazing.md`.

### References

- Conant et al. (2001) DOI:10.1890/1051-0761(2001)011[0343:GMACIG]2.0.CO;2 (FunctorMetadata)
- Milchunas & Lauenroth (1993) DOI:10.2307/2937150 (FunctorMetadata)
- McNaughton (1985) DOI:10.2307/1942578 (FunctorMetadata)
- Howland et al. (2016) DOI:10.1111/gcb.13199 (FunctorMetadata)
- Conant, R.T. et al. (2001). Grassland management and conversion into grassland: effects on soil carbon. (certificate)
- Milchunas, D.G. and Lauenroth, W.K. (1993). Quantitative effects of grazing on vegetation and soils. (certificate)
- McNaughton, S.J. (1985). Ecology of a grazing ecosystem. (certificate)
- Saggar, S. et al. (2007). Modelling nitrous oxide emissions from grazed grasslands. (certificate)
- Ramirez, H. and Le Gorrec, Y. (2022). Irreversible port-Hamiltonian systems. (certificate)

(class-relic-scientific-anthroposphere-management-harvest-harvestmanagement)=
## harvest_management

Harvest Management Functor

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.management.harvest.HarvestManagement` |
| Source | `relic\scientific\anthroposphere\management\harvest.py:95` |
| Runtime name | `harvest_management` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `port` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
harvest_management(name: str = 'harvest_management', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'harvest_management'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `aboveground_biomass` | state | `kg m-2` | Required or optional value consumed by forward(). |
| `belowground_biomass` | forcing | `kg DM m-2` | Required or optional value consumed by forward(). |
| `residue_cover` | forcing | `1` | Required or optional value consumed by forward(). |
| `day_of_year` | forcing | `1` | Required or optional value consumed by forward(). |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `harvest_flux` | flux | `kg DM m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `harvest_biomass_carbon_input` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `harvest_carbon_export_flux` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `residue_c_flux` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `harvest_biomass_nitrogen_input` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `harvest_nitrogen_export_flux` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `residue_n_flux` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `delta_residue_cover` | flux | `s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.harvest.events_today` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.harvest.biomass_partition_residual` | output port | `kg DM m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.harvest.carbon_ledger_residual` | output port | `kg C m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.harvest.nitrogen_ledger_residual` | output port | `kg N m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.harvest_management.free_energy` | output port | `kg2 m-4` | diagnostics | G_h(x) potential audit |
| `diagnostics.harvest_management.chemical_potential_norm` | output port | `kg m-2` | diagnostics | norm of mu_h effort |
| `diagnostics.harvest_management.canonical_iphs_runtime` | output port | `1` | diagnostics | canonical IPHS runtime flag [1] |

### Governing equations and canonical operator

**Literature or process form**

Harvested biomass follows harvest-index partitioning, while crop residues and root turnover return carbon and nitrogen to managed soil pools.

**Canonical RELIC structural form**

The current Level 3 implementation is a typed open-port partition map. The quadratic quantity $Q_{h} = 0.5 \cdot x_{h}^{\mathsf{T}} \cdot H_{h} \cdot x_{h}$ is diagnostic only; no dissipative pH or GENERIC state equation is claimed.

| Entry | Meaning |
| --- | --- |
| `H_harvest` | positive semidefinite diagnostic metric assembled by build_hodge_contract; it is not a GENERIC mobility or a dissipative pH resistance operator |
| `M_h` | empirical non-negative partition weights $M_{h}$($u_{h}, \theta$); the symbol does not denote the GENERIC entropy-production operator |
| `G_h` | open harvest_biomass_residue_port control and export map |
| `O_h` | quadratic-potential, effort-norm, ledger-residual, and runtime diagnostics |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | $Q_{h} ( x_{h} ) = 0.5 \cdot x_{h}^{\mathsf{T}} \cdot H_{\mathrm{harvest}} \cdot x_{h}$ is a diagnostic quadratic quantity, not a Hamiltonian or thermodynamic potential claim |
| `effort_variable` | $q_{h} = \frac{\mathrm{d}Q_{h}}{\mathrm{d}x_{h}} = H_{\mathrm{harvest}} \cdot x_{h}$ is a diagnostic gradient, not a dPH or GENERIC conjugate effort claim |
| `state_advancer` | the Level 3 typed open-port map partitions biomass into managed export and residue-return outputs; it is not asserted to equal (J-R)$e_{H}$ or J $e_{E}$+M $e_{S}$ |
| `runtime_usage` | forward evaluates _canonical_iphs_harvest_partition for each scheduled event and _storage_potential for the potential audit; Level 4 or Level 5 thermodynamic closure is not claimed. |

### Composition channels and executable PortSpec API

Certificate composition channels: `harvest_biomass_residue_port`

Executable `ports()` owner: `HarvestManagement`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `harvest_biomass_residue_port` | `harvest_management_effort [1]` | `harvest_flux [kg m^-2 s^-1]` | `kg m^-2 s^-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.management.harvest.harvest_efficiency` | `dimensionless` | `0.95; bounds=(0.8, 1.0)` | `fixed` | Efficiency factor applied to harvested biomass | Source: management_parameter_database; FAO Harvest Mechanization Manual (2019) |
| `anthroposphere.management.harvest.default_harvest_index` | `dimensionless` | `0.45; bounds=(0.3, 0.7)` | `fixed` | Default fraction of aboveground biomass harvested | Source: management_parameter_database; Hay (1995) |
| `anthroposphere.management.harvest.schedule_window_days` | `day` | `3.0; bounds=(0.0, 30.0)` | `fixed` | Half-width of the calendar window around a scheduled harvest day within which the event may execute (RELIC scheduling convention) | Source: convention; RELIC scheduling convention; no literature value (formerly the literal 3 in HarvestManagement._check_harvest_schedule) |
| `anthroposphere.management.harvest.minimum_event_interval_days` | `day` | `7.0; bounds=(0.0, 365.0)` | `fixed` | Minimum elapsed time after an accepted harvest before another scheduled harvest executes (RELIC scheduling convention) | Source: convention; RELIC scheduling convention; no literature value (formerly the literal 7 in HarvestManagement._check_harvest_schedule) |
| `anthroposphere.management.harvest.default_schedule.cereal.date` | `day` | `270.0; bounds=(1.0, 366.0)` | `fixed` | Day of year of the default C3 cereal combine harvest | Source: convention; RELIC default management scenario convention (formerly a literal in HarvestManagement._initialize_default_schedule) |
| `anthroposphere.management.harvest.default_schedule.cereal.harvest_index` | `dimensionless` | `0.45; bounds=(0.0, 1.0)` | `fixed` | Harvest index (removed fraction of aboveground biomass) of the default C3 cereal combine harvest | Source: convention; Hay (1995) Harvest index: a review of its use in plant breeding and crop physiology, Ann. Appl. Biol. 126, 197-216 (harvest index of modern cereals near 0.4-0.5); RELIC default management scenario convention (formerly a literal in HarvestManagement._initialize_default_schedule) |
| `anthroposphere.management.harvest.default_schedule.cereal.residue_retention` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Fraction of the non-harvested aboveground biomass retained as residue in the default C3 cereal combine harvest | Source: convention; RELIC default management scenario convention (formerly a literal in HarvestManagement._initialize_default_schedule) |
| `anthroposphere.management.harvest.default_schedule.forage_first_cut.date` | `day` | `150.0; bounds=(1.0, 366.0)` | `fixed` | Day of year of the default first forage cut | Source: convention; RELIC default management scenario convention (formerly a literal in HarvestManagement._initialize_default_schedule) |
| `anthroposphere.management.harvest.default_schedule.forage_first_cut.harvest_index` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Harvest index (removed fraction of aboveground biomass) of the default first forage cut | Source: convention; RELIC default management scenario convention (formerly a literal in HarvestManagement._initialize_default_schedule) |
| `anthroposphere.management.harvest.default_schedule.forage_first_cut.residue_retention` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of the non-harvested aboveground biomass retained as residue in the default first forage cut | Source: convention; RELIC default management scenario convention (formerly a literal in HarvestManagement._initialize_default_schedule) |
| `anthroposphere.management.harvest.default_schedule.forage_second_cut.date` | `day` | `220.0; bounds=(1.0, 366.0)` | `fixed` | Day of year of the default second forage cut | Source: convention; RELIC default management scenario convention (formerly a literal in HarvestManagement._initialize_default_schedule) |
| `anthroposphere.management.harvest.default_schedule.forage_second_cut.harvest_index` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Harvest index (removed fraction of aboveground biomass) of the default second forage cut | Source: convention; RELIC default management scenario convention (formerly a literal in HarvestManagement._initialize_default_schedule) |
| `anthroposphere.management.harvest.default_schedule.forage_second_cut.residue_retention` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Fraction of the non-harvested aboveground biomass retained as residue in the default second forage cut | Source: convention; RELIC default management scenario convention (formerly a literal in HarvestManagement._initialize_default_schedule) |
| `anthroposphere.management.harvest.residue_retention` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Fraction of residue left on field | Source: literature; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.harvest.residue_removal_rate` | `dimensionless` | `0.25; bounds=(0.0, 0.8)` | `fixed` | Fraction of residue removed during harvest operations | Source: management_parameter_database; Lal (2005) |
| `anthroposphere.management.harvest.root_mortality_rate` | `dimensionless` | `0.4; bounds=(0.1, 0.9)` | `fixed` | Root mortality fraction triggered by harvest disturbance | Source: management_parameter_database; Johnson et al. (2006) |
| `anthroposphere.management.harvest.cereal_c3_c_content` | `kg C kg-1 DM` | `0.45; bounds=(0.4, 0.5)` | `fixed` | Carbon content of C3 cereal biomass | Source: management_parameter_database; Johnson et al. (2006) |
| `anthroposphere.management.harvest.cereal_c4_c_content` | `kg C kg-1 DM` | `0.45; bounds=(0.4, 0.5)` | `fixed` | Carbon content of C4 cereal biomass | Source: management_parameter_database; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.management.harvest.forage_c_content` | `kg C kg-1 DM` | `0.45; bounds=(0.4, 0.5)` | `fixed` | Carbon content of forage biomass | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Luyssaert et al. (2007) Glob. Change Biol. 13, 2509-2537 |
| `anthroposphere.management.harvest.forest_c_content` | `kg C kg-1 DM` | `0.5; bounds=(0.45, 0.55)` | `fixed` | Carbon content of harvested forest biomass | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Luyssaert et al. (2007) Glob. Change Biol. 13, 2509-2537 |
| `anthroposphere.management.harvest.legume_c_content` | `kg C kg-1 DM` | `0.45; bounds=(0.4, 0.5)` | `fixed` | Carbon content of legume biomass | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Luyssaert et al. (2007) Glob. Change Biol. 13, 2509-2537 |
| `anthroposphere.management.harvest.root_tuber_c_content` | `kg C kg-1 DM` | `0.42; bounds=(0.35, 0.5)` | `fixed` | Carbon content of root/tuber biomass | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Luyssaert et al. (2007) Glob. Change Biol. 13, 2509-2537 |
| `anthroposphere.management.harvest.cereal_c3_cn_ratio` | `kg C kg-1 N` | `80.0; bounds=(60.0, 100.0)` | `fixed` | Carbon-to-nitrogen ratio for C3 cereal residue | Source: management_parameter_database; Lal (2005) |
| `anthroposphere.management.harvest.cereal_c4_cn_ratio` | `kg C kg-1 N` | `85.0; bounds=(65.0, 105.0)` | `fixed` | Carbon-to-nitrogen ratio for C4 cereal residue | Source: management_parameter_database; Collatz et al. (1992) Aust. J. Plant Physiol. 19, 519-538; Cleveland & Liptzin (2007) Biogeochemistry 85, 235-252 |
| `anthroposphere.management.harvest.forage_cn_ratio` | `kg C kg-1 N` | `30.0; bounds=(20.0, 40.0)` | `fixed` | Carbon-to-nitrogen ratio for forage residue | Source: management_parameter_database; Cleveland & Liptzin (2007) Biogeochemistry 85, 235-252; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.management.harvest.forest_cn_ratio` | `kg C kg-1 N` | `120.0; bounds=(80.0, 180.0)` | `fixed` | Carbon-to-nitrogen ratio for forest residue | Source: management_parameter_database; Cleveland & Liptzin (2007) Biogeochemistry 85, 235-252; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.management.harvest.legume_cn_ratio` | `kg C kg-1 N` | `25.0; bounds=(15.0, 35.0)` | `fixed` | Carbon-to-nitrogen ratio for legume residue | Source: management_parameter_database; Cleveland & Liptzin (2007) Biogeochemistry 85, 235-252; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.management.harvest.root_tuber_cn_ratio` | `kg C kg-1 N` | `40.0; bounds=(25.0, 55.0)` | `fixed` | Carbon-to-nitrogen ratio for root/tuber residue | Source: management_parameter_database; Cleveland & Liptzin (2007) Biogeochemistry 85, 235-252; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | outputs are returned through existing FunctorBase/StateManager path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_management_harvest.md`.

### References

- Prince et al. (2001) DOI:10.1046/j.1365-2699.2001.00567.x (FunctorMetadata)
- Johnson et al. (2006) DOI:10.2134/agronj2005.0179 (FunctorMetadata)
- Lal (2005) DOI:10.1016/j.envint.2004.09.005 (FunctorMetadata)
- Blanco-Canqui & Lal (2009) DOI:10.1080/07352680902776507 (FunctorMetadata)
- Prince, S.D. et al. (2001). Global primary production: A remote sensing approach. (certificate)
- Johnson, J.M.F. et al. (2006). Agricultural opportunities to mitigate greenhouse gas emissions. (certificate)
- Lal, R. (2005). World crop residues production and implications of its use as a biofuel. (certificate)
- Blanco-Canqui, H. and Lal, R. (2009). Crop residue removal impacts on soil productivity and environmental quality. (certificate)
- Grmela, M. and Ottinger, H.C. (1997). Dynamics and thermodynamics of complex fluids. (certificate)
- van der Schaft, A. and Maschke, B. (1995). Port-Hamiltonian systems. (certificate)

(class-relic-scientific-anthroposphere-management-irrigation-irrigationmanagement)=
## IrrigationManagement

Irrigation Management Functor

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.management.irrigation.IrrigationManagement` |
| Source | `relic\scientific\anthroposphere\management\irrigation.py:58` |
| Runtime name | `irrigation_management` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `port` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
IrrigationManagement(name: str = 'irrigation_management', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'irrigation_management'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | soil_moisture mapped to canonical Phase 8 cochain basis |
| `precipitation_flux` | forcing | `kg m-2 s-1` | external precipitation mass-flux forcing mapped to the canonical water port |
| `land_use_type` | forcing | `1` | external land-use composition code controlling the managed port |
| `irrigation_type` | forcing | `1` | external irrigation-system code selecting ParameterManager efficiency |
| `field_capacity` | forcing | `m3 m-3` | external soil hydraulic property controlling root-zone deficit |
| `wilting_point` | forcing | `m3 m-3` | external soil hydraulic property controlling the irrigation trigger |
| `irrigation_remaining_quota` | state | `mm` | remaining source-water quota [mm], owned state; initial value from the scenario initial state |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `irrigation_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `abstraction_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `irrigation_remaining_quota` | state | `mm` | decided at run time by the returned write mode | remaining source-water quota [mm], owned state; initial value from the scenario initial state |
| `diagnostics.irrigation.applied_water` | output port | `mm` | diagnostics | delivered depth of the step (readers integrate) |
| `diagnostics.irrigation.effective_efficiency` | output port | `1` | diagnostics | explicit application-efficiency diagnostic |
| `diagnostics.irrigation.moisture_response` | output port | `1` | diagnostics | explicit soil-moisture response diagnostic |
| `diagnostics.irrigation.water_balance_residual` | output port | `kg m-2 s-1` | diagnostics | explicit delivered-versus-abstracted water residual |

### Governing equations and canonical operator

**Literature or process form**

$$
M_{\mathrm{gross}} = M_{\mathrm{net}} / \eta; M_{\mathrm{delivered}} = \eta M_{\mathrm{gross}}
$$

**Canonical RELIC structural form**

$y = G_{h} u_{h}$ through the declared open management-water port map

| Entry | Meaning |
| --- | --- |
| `G_h` | declared port map |
| `H_h` | positive port metric or conversion Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed root-zone deficit and application-efficiency control |
| `state_advancer` | $y = G_{h} u_{h}$ through the declared open management-water port map |
| `runtime_usage` | IrrigationManagement.forward evaluates one phenomenological port response; strict IPHS runtime structure is not claimed. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.management.irrigation.surface_efficiency` | `dimensionless` | `0.6; bounds=(0.4, 0.7)` | `learnable` | Application efficiency for surface irrigation | Source: management_parameter_database; Howell (2003) |
| `anthroposphere.management.irrigation.sprinkler_efficiency` | `dimensionless` | `0.75; bounds=(0.65, 0.85)` | `learnable` | Application efficiency for sprinkler irrigation | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Siebert et al. (2010) Hydrol. Earth Syst. Sci. 14, 1863-1880 |
| `anthroposphere.management.irrigation.drip_efficiency` | `dimensionless` | `0.9; bounds=(0.85, 0.95)` | `learnable` | Application efficiency for drip irrigation | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Siebert et al. (2010) Hydrol. Earth Syst. Sci. 14, 1863-1880 |
| `anthroposphere.management.irrigation.subsurface_efficiency` | `dimensionless` | `0.92; bounds=(0.9, 0.97)` | `learnable` | Application efficiency for subsurface drip irrigation | Source: management_parameter_database; Ayars et al. (2015) |
| `anthroposphere.management.irrigation.moisture_threshold` | `dimensionless` | `0.35; bounds=(0.3, 0.8)` | `fixed` | Fraction of the plant-available water range (wilting point to field capacity) below which irrigation is triggered | Source: management_parameter_database; Irrigation scheduling guidelines |
| `anthroposphere.management.irrigation.root_zone_depth` | `m` | `0.5; bounds=(0.1, 2.0)` | `fixed` | Soil root zone depth considered in irrigation scheduling | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Siebert et al. (2010) Hydrol. Earth Syst. Sci. 14, 1863-1880 |
| `anthroposphere.management.irrigation.max_application_rate` | `mm hr-1` | `10.0; bounds=(1.0, 50.0)` | `fixed` | Maximum gross irrigation application rate per hour | Source: management_parameter_database; Allen et al. (1998) |
| `anthroposphere.management.irrigation.conservation_tolerance` | `kg m-2 s-1` | `1e-06; bounds=(0.0, 5e-05)` | `fixed` | Permissible irrigation mass conservation discrepancy before warning | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Siebert et al. (2010) Hydrol. Earth Syst. Sci. 14, 1863-1880 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | explicit UpdateItem REPLACE with per_timestep false and source_functor irrigation_management |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_management_irrigation.md`.

### References

- Allen et al. (1998) FAO-56 Crop evapotranspiration (FunctorMetadata, certificate)
- Doorenbos & Pruitt (1977) FAO-24 Crop water requirements (FunctorMetadata)
- Hanks (1974) DOI:10.2134/agronj1974.00021962006600050017x (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- FAO Determination of the irrigation schedule for a field (certificate)
- FAO Scheme irrigation water need and supply (certificate)
- Howell (2001) Enhancing water use efficiency in irrigated agriculture (certificate)

(class-relic-scientific-anthroposphere-management-tillage-tillagemanagement)=
## TillageManagement

Tillage Management Functor

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.anthroposphere.management.tillage.TillageManagement` |
| Source | `relic\scientific\anthroposphere\management\tillage.py:86` |
| Runtime name | `tillage_management` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `port` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
TillageManagement(name: str = 'tillage_management', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'tillage_management'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | soil_moisture mapped to canonical Phase 8 cochain basis |
| `soil_temperature` | forcing | `K` | soil temperature of the tilled layer [K], the Q10 driver of the tillage carbon flush |
| `soil_organic_carbon` | forcing | `kg C m-2` | soil organic carbon stock of the tilled layer [kg C m-2] exposed by the carbon flush (no default) |
| `day_of_year` | forcing | `1` | day_of_year mapped to canonical Phase 8 cochain basis |
| `days_since_tillage` | state | `day` | days_since_tillage mapped to canonical Phase 8 cochain basis |
| `residue_cover` | forcing | `1` | residue cover fraction [1], optional, read only when an operation executes (incorporated cover -min(cover, R)); an operation without it raises |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `porosity_modifier` | state | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `roughness_modifier` | state | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `days_since_tillage` | state | `day` | decided at run time by the returned write mode | days_since_tillage mapped to canonical Phase 8 cochain basis |
| `tillage_intensity_factor` | diagnostic | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `time_decay_factor` | diagnostic | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `cumulative_operations` | diagnostic | `1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `soil_disturbance_rate` | diagnostic | `1` | decided at run time by the returned write mode | dimensionless direct tillage disturbance contribution |
| `residue_incorporation_rate` | diagnostic | `1` | decided at run time by the returned write mode | dimensionless residue-incorporation contribution |
| `carbon_loss_rate` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | operation-step tillage carbon flush F_C = C_soil f_0 f_p I D f_T / S_day [kg C m^-2 s^-1], the one tillage carbon flux, zero outside an operation step; post-tillage enhanced decomposition is not represented |
| `soil_co2_flux` | flux | `kg CO2 m-2 s-1` | decided at run time by the returned write mode | tillage CO2 mass flux F_C M_CO2 / M_C [kg CO2 m^-2 s^-1] |
| `delta_residue_cover` | flux | `s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.tillage_mass_balance_error` | output port | `kg C m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

$D_{\mathrm{tillage}} = k_{\mathrm{disturbance}} I f_{\theta}$; $R_{\mathrm{inc}} = \operatorname{clamp} ( k_{\mathrm{residue}} I ( a_{\theta} + f_{\mathrm{moisture}} ) , 0 , 1 )$; $F_{C} = C_{\mathrm{soil}} f_{\mathrm{0}} f_{p} I D f_{T} / S_{\mathrm{day}} \mathrm{in} \mathrm{an} \mathrm{operation} \mathrm{step}$, 0 otherwise.

**Canonical RELIC structural form**

$\dot{y} = G_{h} u_{h}$ through the declared tillage disturbance, residue, porosity, roughness, and carbon-export ports.

| Entry | Meaning |
| --- | --- |
| `G_h` | declared port map |
| `H_h` | positive port metric or conversion Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{y} = G_{h} u_{h}$ through the declared matter, energy, or management port map |
| `runtime_usage` | TillageManagement.forward applies the canonical IPHS port map and routes declared control, forcing, or interface fluxes through the production output path. |

### Composition channels and executable PortSpec API

Certificate composition channels: `soil_moisture`, `soil_temperature`, `soil_organic_carbon`, `days_since_tillage`, `residue_cover`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `anthroposphere.management.tillage.disturbance_factor` | `dimensionless` | `0.6; bounds=(0.0, 1.5)` | `fixed` | Base soil disturbance scaling for tillage operations | Source: literature; Reicosky 1997 DOI:10.1023/A:1009766510274 |
| `anthroposphere.management.tillage.residue_incorporation_rate` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `fixed` | Fraction of surface residue incorporated by tillage | Source: literature; Six 2002 DOI:10.1023/A:1016125726789 |
| `anthroposphere.management.tillage.effect_decay_rate` | `day-1` | `0.05; bounds=(0.01, 0.2)` | `fixed` | Exponential decay rate for tillage effects | Source: management_parameter_database; Alvarez & Steinbach (2009) |
| `anthroposphere.management.tillage.schedule_window_days` | `day` | `3.0; bounds=(0.0, 30.0)` | `fixed` | Half-width of the calendar window around a scheduled tillage day within which the operation may execute (RELIC scheduling convention) | Source: convention; RELIC scheduling convention; no literature value (formerly the literal 3 in TillageManagement._check_tillage_schedule) |
| `anthroposphere.management.tillage.minimum_operation_interval_days` | `day` | `14.0; bounds=(0.0, 365.0)` | `fixed` | Minimum elapsed time since the last tillage operation before another scheduled operation executes (RELIC scheduling convention) | Source: convention; RELIC scheduling convention; no literature value (formerly the literal 14 in TillageManagement._check_tillage_schedule) |
| `anthroposphere.management.tillage.default_schedule.autumn_chisel.date` | `day` | `290.0; bounds=(1.0, 366.0)` | `fixed` | Day of year of the default autumn chisel pass | Source: convention; RELIC default management scenario convention (formerly a literal in TillageManagement._initialize_default_schedule) |
| `anthroposphere.management.tillage.default_schedule.autumn_chisel.soil_moisture_limit` | `m3 m-3` | `0.28; bounds=(0.0, 1.0)` | `fixed` | Upper soil moisture limit for executing the default autumn chisel pass | Source: convention; Dexter and Bird (2001) Methods for predicting the optimum and the range of soil water contents for tillage, Soil Tillage Res. 57, 203-212 (concept of an upper soil water content for workable tillage; the value is a RELIC scenario convention); RELIC default management scenario convention (formerly a literal in TillageManagement._initialize_default_schedule) |
| `anthroposphere.management.tillage.default_schedule.conservation.date` | `day` | `150.0; bounds=(1.0, 366.0)` | `fixed` | Day of year of the default low-intensity conservation pass | Source: convention; RELIC default management scenario convention (formerly a literal in TillageManagement._initialize_default_schedule) |
| `anthroposphere.management.tillage.default_schedule.conservation.soil_moisture_limit` | `m3 m-3` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Upper soil moisture limit for executing the default low-intensity conservation pass | Source: convention; Dexter and Bird (2001) Methods for predicting the optimum and the range of soil water contents for tillage, Soil Tillage Res. 57, 203-212 (concept of an upper soil water content for workable tillage; the value is a RELIC scenario convention); RELIC default management scenario convention (formerly a literal in TillageManagement._initialize_default_schedule) |
| `anthroposphere.management.tillage.default_schedule.primary.date` | `day` | `90.0; bounds=(1.0, 366.0)` | `fixed` | Day of year of the default deep high-intensity primary pass | Source: convention; RELIC default management scenario convention (formerly a literal in TillageManagement._initialize_default_schedule) |
| `anthroposphere.management.tillage.default_schedule.primary.soil_moisture_limit` | `m3 m-3` | `0.22; bounds=(0.0, 1.0)` | `fixed` | Upper soil moisture limit for executing the default deep high-intensity primary pass | Source: convention; Dexter and Bird (2001) Methods for predicting the optimum and the range of soil water contents for tillage, Soil Tillage Res. 57, 203-212 (concept of an upper soil water content for workable tillage; the value is a RELIC scenario convention); RELIC default management scenario convention (formerly a literal in TillageManagement._initialize_default_schedule) |
| `anthroposphere.management.tillage.default_schedule.secondary.date` | `day` | `110.0; bounds=(1.0, 366.0)` | `fixed` | Day of year of the default shallow secondary pass | Source: convention; RELIC default management scenario convention (formerly a literal in TillageManagement._initialize_default_schedule) |
| `anthroposphere.management.tillage.default_schedule.secondary.soil_moisture_limit` | `m3 m-3` | `0.25; bounds=(0.0, 1.0)` | `fixed` | Upper soil moisture limit for executing the default shallow secondary pass | Source: convention; Dexter and Bird (2001) Methods for predicting the optimum and the range of soil water contents for tillage, Soil Tillage Res. 57, 203-212 (concept of an upper soil water content for workable tillage; the value is a RELIC scenario convention); RELIC default management scenario convention (formerly a literal in TillageManagement._initialize_default_schedule) |
| `anthroposphere.management.tillage.porosity_change` | `dimensionless` | `0.15; bounds=(0.0, 0.5)` | `fixed` | Base porosity increase following tillage | Source: management_parameter_database; Rawls et al. (1982) Trans. ASAE 25, 1316-1320; Arneth et al. (2017) Nat. Geosci. 10, 79-84 |
| `anthroposphere.management.tillage.roughness_intensity_coefficient` | `dimensionless` | `0.5; bounds=(0.0, 2.0)` | `fixed` | Intensity response coefficient for tillage roughness modifier | Source: RELIC parameter database; Six et al. (2002) aggregate disruption and tillage intensity responses |
| `anthroposphere.management.tillage.co2_pulse_base_fraction` | `dimensionless` | `0.001; bounds=(0.0, 0.01)` | `fixed` | Base fractional scaling for tillage induced CO2 pulse emissions | Source: RELIC parameter database; Paustian et al. (2000) soil disturbance pulse response |
| `anthroposphere.management.tillage.co2_pulse_factor` | `dimensionless` | `3.0; bounds=(1.0, 5.0)` | `fixed` | Scaling factor for CO2 pulse emissions following tillage | Source: management_parameter_database; Lal (2005) |
| `anthroposphere.management.tillage.moisture_optimum` | `m3 m-3` | `0.3; bounds=(0.1, 0.5)` | `fixed` | Optimal soil moisture for tillage disturbance response | Source: literature_synthesis; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.tillage.moisture_sigma` | `m3 m-3` | `0.08; bounds=(0.02, 0.2)` | `fixed` | Gaussian width for moisture response under tillage | Source: literature_synthesis; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.tillage.moisture_effect_denominator` | `m3 m-3` | `0.35; bounds=(0.1, 0.6)` | `fixed` | Soil moisture scaling denominator for tillage moisture effect | Source: literature_synthesis; FAO (2000) Fertilizers and Their Use, 4th edition |
| `anthroposphere.management.tillage.residue_moisture_offset` | `dimensionless` | `0.5; bounds=(0.0, 2.0)` | `fixed` | Offset term applied to moisture response in tillage residue incorporation | Source: RELIC parameter database; DNDC process representation of residue incorporation response |
| `anthroposphere.management.tillage.high_intensity_factor` | `dimensionless` | `1.0; bounds=(0.6, 1.5)` | `fixed` | Factor applied for high intensity tillage | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.tillage.low_intensity_factor` | `dimensionless` | `0.3; bounds=(0.1, 0.6)` | `fixed` | Factor applied for low intensity tillage operations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.tillage.moderate_intensity_factor` | `dimensionless` | `0.6; bounds=(0.3, 1.0)` | `fixed` | Factor applied for moderate tillage intensity | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.tillage.none_intensity_factor` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Intensity factor applied when no tillage occurs | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.tillage.very_high_intensity_factor` | `dimensionless` | `1.5; bounds=(1.0, 2.0)` | `fixed` | Factor applied for very high intensity tillage | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.tillage.deep_depth_factor` | `dimensionless` | `1.0; bounds=(0.6, 1.4)` | `fixed` | Depth factor scaling for deep tillage | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.tillage.medium_depth_factor` | `dimensionless` | `0.8; bounds=(0.4, 1.0)` | `fixed` | Depth factor scaling for medium tillage depth | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.tillage.shallow_depth_factor` | `dimensionless` | `0.5; bounds=(0.1, 0.8)` | `fixed` | Depth factor scaling for shallow tillage operations | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `anthroposphere.management.tillage.very_deep_depth_factor` | `dimensionless` | `1.2; bounds=(0.8, 1.5)` | `fixed` | Depth factor for very deep tillage | Source: management_parameter_database; Arneth et al. (2017) Nat. Geosci. 10, 79-84; Smith et al. (2014) Nat. Clim. Change 4, 850-853 |
| `pedosphere.physical.q10_decomposition` | `dimensionless` | `2.0; bounds=(1.5, 3.0)` | `learnable` | Temperature sensitivity coefficient for soil organic matter decomposition, doubling rate per 10K increase; also the temperature factor of the tillage CO2 pulse (oxidation of the exposed organic matter) | Source: Soil respiration literature; Lloyd & Taylor (1994) doi:10.2307/2389824; Kirschbaum (1995) doi:10.1016/0038-0717(94)00242-S |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, $\mathrm{calendar}_{\mathrm{type}} = \mathrm{None} , \cdot \cdot \mathrm{kwargs} )$ |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Anthroposphere theory](../theory/anthroposphere.md).

Certificate theory owner: `docs\source\theory\anthroposphere_management_tillage.md`.

### References

- Lal (2004) DOI:10.1126/science.1097396 (FunctorMetadata, certificate)
- Paustian et al. (2000) DOI:10.1023/A:1005554205788 (FunctorMetadata, certificate)
- Six et al. (2002) DOI:10.1023/A:1016125726789 (FunctorMetadata, certificate)
- Reicosky (1997) DOI:10.1023/A:1009766510274 (FunctorMetadata, certificate)
- Davidson and Janssens (2006) DOI:10.1038/nature04514 (FunctorMetadata, certificate)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
