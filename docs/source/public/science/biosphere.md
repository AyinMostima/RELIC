# Biosphere Scientific Classes

This reference covers all 24 certificate-backed scientific classes in `relic.scientific.biosphere`: 24 executable concrete Functors, 0 fail-closed unavailable boundaries, and 0 abstract contracts. Each entry is generated from the live class, its IPHS certificate, State Contract Model, and ParameterManager metadata. Fail-closed unavailable classes are not instantiated during documentation generation.

The tables distinguish constructor controls, runtime variables, and ParameterManager scientific parameters. Values shown as package defaults belong to the documented RELIC architecture version and must not be copied as hardcoded constants into scientific code.

Certificate wildcard parameter groups (`*`, `<pft>`) are expanded to their exact current YAML entries. Every other declared parameter is matched to the parameter database by its exact full key; a declared name is never mapped to a key that merely ends with the same name.

Exact ParameterManager rows report the live YAML value, bounds, gradient policy, description, and provenance. A row whose value reads `not declared in the parameter database` is intentionally not assigned an invented default: the class metadata or certificate declares the quantity, with the meaning and units it states, but the parameter database has no key of that exact name.

IPHS levels are read from the sole certificate owner. Levels 1 to 3 describe compatibility, wrapping, or declared structure; they are not strict IPHS/GENERIC claims.

The Channel column of each runtime output contract applies `relic.core.state.types.partition_outputs`, the only runtime output channel classifier, to the declared output name. A name that the classifier reserves for diagnostics, such as `diagnostics.*`, goes to the diagnostics channel and is never committed as state. For any other name the channel is decided at run time by the write mode of the returned value: `StateWriteMode.DERIVED` routes the output to the diagnostics channel, and any other write mode sends it to the StateManager commit (a value without a write mode takes the run's semantic-mode rules or the StateManager schema). The generator does not run `forward()`, so it does not predict that write mode. The Kind column is the State Contract Model kind of the variable and does not set its channel.

(class-relic-scientific-biosphere-bvoc-megan-model-meganmodel)=
## MEGANModel

BiologicalProcess bridge for a reduced instantaneous MEGAN closure.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.bvoc.megan_model.MEGANModel` |
| Source | `relic\scientific\biosphere\bvoc\megan_model.py:495` |
| Runtime name | `megan_bvoc_model` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MEGANModel(name: str = 'megan_bvoc_model', config: Dict[str, Any] | None = None, device: torch.device | None = None, coupling_strategy: relic.scientific.biosphere.bvoc.megan_model.CouplingStrategy = <CouplingStrategy.SEQUENTIAL: 'sequential'>, emission_pathway: relic.scientific.biosphere.bvoc.megan_model.EmissionPathway = <EmissionPathway.HYBRID: 'hybrid'>, validate_inputs: bool = True)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'megan_bvoc_model'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `coupling_strategy` | `relic.scientific.biosphere.bvoc.megan_model.CouplingStrategy` | `<CouplingStrategy.SEQUENTIAL: 'sequential'>` | Constructor control for coupling strategy; see the class runtime contract and ParameterManager table below. |
| `emission_pathway` | `relic.scientific.biosphere.bvoc.megan_model.EmissionPathway` | `<EmissionPathway.HYBRID: 'hybrid'>` | Constructor control for emission pathway; see the class runtime contract and ParameterManager table below. |
| `validate_inputs` | `bool` | `True` | Enables fail-fast validation of required runtime inputs. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | canonical canopy driver in K |
| `ppfd` | flux | `umol photons m-2 s-1` | canonical atmospheric-radiation photon port in umol photons m-2 s-1 |
| `leaf_area_index` | state | `m2 m-2` | canonical one-sided canopy leaf-area state in m2 leaf m-2 ground |
| `atmospheric_co2_ppm` | forcing | `umol mol-1` | canonical ambient CO2 forcing in ppm |
| `net_photosynthesis` | flux | `umol CO2 m-2 s-1` | required photosynthesis or hybrid pathway carbon driver in umol C m-2 s-1 |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `isoprene_emission` | flux | `mol m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `monoterpenes_emission` | flux | `mol m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `sesquiterpenes_emission` | flux | `mol m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `ovocs_emission` | flux | `mol m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `total_bvoc_emission` | flux | `mol m-2 s-1` | decided at run time by the returned write mode | exact sum of enabled compound molar fluxes |
| `bvoc_carbon_allocation_flux` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | equivalent instantaneous carbon-allocation proxy obtained from molecular weights and carbon fractions |
| `diagnostics.bvoc.megan.environmental_stress_factor` | diagnostic | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.bvoc.megan.photosynthesis_carbon_floor_residual` | output port | `umol C m-2 s-1` | diagnostics | explicit A_available minus A_n lower-bound projection residual in umol C m-2 s-1 |
| `diagnostics.bvoc.megan.co2_inhibition_factor` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.bvoc.megan.isoprene.gamma_temperature` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.bvoc.megan.isoprene.gamma_light` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.bvoc.megan.monoterpenes.gamma_temperature` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.bvoc.megan.monoterpenes.gamma_light` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.bvoc.megan.sesquiterpenes.gamma_temperature` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.bvoc.megan.sesquiterpenes.gamma_light` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.bvoc.megan.ovocs.gamma_temperature` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.bvoc.megan.ovocs.gamma_light` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

$\gamma_{T,i} = ( 1 - \mathrm{LDF}_{i} ) \exp ( \beta_{i} ( T - T_{s} ) ) + \mathrm{LDF}_{i} E_{\mathrm{opt},i} C_{\mathrm{T2}} \exp ( C_{\mathrm{T1},i} x ) / ( C_{\mathrm{T2}} - C_{\mathrm{T1},i} ( 1 - \exp ( C_{\mathrm{T2}} x ) ) )$, $x = ( 1 / T_{\mathrm{opt}} - 1 / T ) / R ( \mathrm{Guenther} \mathrm{et} \mathrm{al} . 2012 \mathrm{Eq} . 7 \mathrm{with} \mathrm{Eqs} . 8 \mathrm{and} 11 ; \beta_{i} , \mathrm{LDF}_{i} , C_{\mathrm{T1},i} \mathrm{and} C_{\mathrm{eo},i} \mathrm{from} \mathrm{one} \mathrm{Table} 4 \mathrm{row} \mathrm{per} \mathrm{class} ; \mathrm{LDF} = 1 \mathrm{for} \mathrm{isoprene} ; \mathrm{the} \mathrm{lumped} \mathrm{OVOC} \mathrm{class} \mathrm{takes} \sum_{j} w_{j} \gamma_{T,j} \mathrm{over} \mathrm{its} \mathrm{methanol} , \mathrm{acetone} \mathrm{and} \mathrm{acetaldehyde} \mathrm{rows} \mathrm{with} \mathrm{the} \mathrm{Table} 6 \mathrm{emission} \mathrm{shares} w_{j} )$; $\gamma_{P,i} = ( 1 - \mathrm{LDF}_{i} ) + \mathrm{LDF}_{i} \gamma_{P,\mathrm{LDF}}$ with $\gamma_{P,\mathrm{LDF}} = \alpha \mathrm{CL} \mathrm{PPFD} / \sqrt{1 + \alpha ^ 2 \mathrm{PPFD} ^ 2}$, $\mathrm{LDF} = 1$ for isoprene, the Table 4 LDF for monoterpenes (representative Other Monoterpenes value) and sesquiterpenes, and the Table 6 emission-weighted mixture sum_j $w_{j}$ gamma_P,j of the methanol, acetone and bidirectional VOC (acetaldehyde) rows for the lumped OVOC class; $\gamma_{\mathrm{CO2}} = \mathrm{IS}_{\max} - \mathrm{IS}_{\max} \mathrm{Ci} ^ h / ( ( C^\star ) ^ h + \mathrm{Ci} ^ h )$

**Canonical RELIC structural form**

$\dot{x} = N_{h} r_{h}$ with separate molar species and carbon-mass open-port maps

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | reduced dry-leaf-mass emission potentials mapped by $\rho_{f} = \mathrm{LAI} \cdot k_{f}$ to open atmospheric molar species ports plus a separate carbon-mass boundary port |
| `effort_variable` | no independently validated thermodynamic effort is claimed at level 2 |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit molar species and carbon-mass maps |
| `runtime_usage` | MEGANModel.forward calls advance_reaction_network twice with dimension-consistent stoichiometric_N and rates before emitting canonical outputs. |
| `explicit_stoichiometric_usage` | One $N_{h}$ maps enabled molar species to their molar total; a separate $N_{h}$ maps the g C boundary export. |

### Composition channels and executable PortSpec API

Certificate composition channels: `isoprene_emission`, `monoterpenes_emission`, `sesquiterpenes_emission`, `ovocs_emission`, `bvoc_carbon_allocation_flux`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.bvoc.megan.hybrid_weight` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Weighting factor blending photosynthesis-coupled and empirical emission pathways | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012 |
| `biosphere.bvoc.carbon_fraction.isoprene` | `dimensionless` | `0.882; bounds=(0.8, 0.92)` | `fixed` | Carbon mass fraction for isoprene (C5H8, 60 g C per 68 g compound) | Source: chemical_stoichiometry; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.carbon_fraction.monoterpenes` | `dimensionless` | `0.882; bounds=(0.8, 0.92)` | `fixed` | Carbon mass fraction for monoterpenes (C10H16, 120 g C per 136 g compound) | Source: chemical_stoichiometry; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.carbon_fraction.sesquiterpenes` | `dimensionless` | `0.882; bounds=(0.8, 0.92)` | `fixed` | Carbon mass fraction for sesquiterpenes (C15H24, 180 g C per 204 g compound) | Source: chemical_stoichiometry; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.carbon_fraction.ovocs` | `dimensionless` | `0.4618; bounds=(0.4, 0.8)` | `fixed` | Effective carbon mass fraction of the lumped OVOC class (methanol, acetone and acetaldehyde): MEGAN2.1 global-emission-weighted equivalent f_C,eff = sum(w_i f_C,i) with the mass emissions w_i = 99.6, 43.7 and 20.7 Tg yr-1 (Guenther et al. 2012 Table 6) | Source: literature_synthesis; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 6 (global annual emissions for 2000: methanol 99.6, acetone 43.7, acetaldehyde 20.7 Tg yr-1); Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.validation.temperature.min_operational` | `K` | `203.15; bounds=(200.0, 260.0)` | `fixed` | Minimum physically valid canopy temperature for MEGAN inputs. Lowered from 233.15 K on 2026-11-05 to the terrestrial range (Northern Hemisphere record low -67.8 degC) for global daily runs; the temperature response tends to zero there (gamma_T about 1e-9 at 203.15 K with the default coefficients). | Source: MEGAN validation requirements; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012; Niinemets et al. (2010) Biogeosciences 7, 1809-1832, doi:10.5194/bg-7-1809-2010 |
| `biosphere.bvoc.validation.temperature.max_operational` | `K` | `333.15; bounds=(310.0, 360.0)` | `fixed` | Maximum physically valid canopy temperature for MEGAN inputs | Source: MEGAN validation requirements; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012; Niinemets et al. (2010) Biogeosciences 7, 1809-1832, doi:10.5194/bg-7-1809-2010 |
| `biosphere.bvoc.emission_factors.isoprene` | `ug g-1 DW h-1` | `10.0; bounds=(0.0, 100.0)` | `learnable` | Canopy-scale isoprene emission factor under standard conditions | Source: MEGAN2.1 default broadleaf value; Guenther et al. (2012) MEGAN2.1 |
| `biosphere.bvoc.emission_factors.monoterpenes` | `ug g-1 DW h-1` | `2.0; bounds=(0.5, 6.0)` | `learnable` | Baseline monoterpene emission factor for temperate mixed forests | Source: MEGAN2.1 default needleleaf value; Guenther et al. (2012) MEGAN2.1; Sindelarova et al. (2014) Global emission data |
| `biosphere.bvoc.emission_factors.sesquiterpenes` | `ug g-1 DW h-1` | `0.5; bounds=(0.1, 1.5)` | `learnable` | Baseline sesquiterpene emission factor | Source: MEGAN2.1 typical values; Sindelarova et al. (2014) Global emission data |
| `biosphere.bvoc.emission_factors.ovocs` | `ug g-1 DW h-1` | `1.8; bounds=(0.5, 5.0)` | `learnable` | Baseline oxygenated VOCs emission factor for aggregated OVOCs | Source: Aligned with biosphere.bvoc.default.ovocs; Guenther et al. (2012) MEGAN2.1 |
| `biosphere.bvoc.megan.alpha_light` | `m2 s umol-1` | `0.0027; bounds=(0.001, 0.005)` | `learnable` | Light response coefficient for BVOC emissions | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (5); Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Eq. (6) |
| `biosphere.bvoc.megan.cl_light` | `dimensionless` | `1.066; bounds=(0.8, 1.3)` | `learnable` | Light scaling factor for MEGAN light response | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (6); Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Eq. (7) |
| `biosphere.bvoc.megan.ldf_monoterpenes` | `dimensionless` | `0.4; bounds=(0.0, 1.0)` | `fixed` | Light-dependent fraction LDF of the lumped monoterpene class (undifferentiated total C10H16 monoterpenes) in the MEGAN2.1 light activity gamma_L = (1 - LDF) + LDF gamma_P_LDF; representative value | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (3) and Table 4 (LDF = 0.4 for Other Monoterpenes; 0.2 to 0.8 for the individually listed monoterpenes) |
| `biosphere.bvoc.megan.ldf_sesquiterpenes` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Light-dependent fraction LDF of sesquiterpene emission in the MEGAN2.1 light activity gamma_L = (1 - LDF) + LDF gamma_P_LDF | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (3) and Table 4 (LDF = 0.5 for alpha-farnesene, beta-caryophyllene and other sesquiterpenes) |
| `biosphere.bvoc.megan.topt` | `K` | `313.0; bounds=(298.0, 313.0)` | `learnable` | Optimum temperature T_opt of the Eq. (8) light-dependent temperature activity gamma_T,LDF, Eq. (9) T_opt = 313 + 0.6 (T240 - 297) at the fixed standard history T240 = T_s = 297 K, so T_opt = 313 K; class independent and used for every compound class (Guenther et al. 2012) | Source: megan_v2.1_model; Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Eq. (8); Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (9) |
| `biosphere.bvoc.t_standard` | `K` | `303.15; bounds=(298.15, 308.15)` | `fixed` | Standard temperature T_s of the light-independent Eq. (11) response exp(beta (T - T_s)); 303.15 K (30 degC) is paired with the reference temperature of the RELIC dry-mass emission factors (basal emission rates normalized to 30 degC; standard air temperature 303 K of the MEGAN canopy factors). Guenther et al. (2012) use T_s = 297 K together with the MEGAN2.1 area-based Table 2 emission factors; the two pairs must not be mixed. | Source: megan_v2.1_model; Duhl et al. (2008) Biogeosciences 5, 761-777, doi:10.5194/bg-5-761-2008, Sect. 2.1, Eq. (1) (basal emission rates in ug gDW-1 h-1 normalized to a standard temperature, usually 30 degC); Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Sect. 2.1 (standard air temperature 303 K) |
| `biosphere.bvoc.megan.eopt_isoprene` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `learnable` | Optimum activity E_opt of the isoprene Eq. (8) temperature activity at the fixed standard history (T24 = T240 = T_s), the C_eo of the Guenther et al. (2012) Table 4 Isoprene row | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Isoprene: C_eo = 2) and Eq. (10) |
| `biosphere.bvoc.megan.ct1_isoprene` | `J mol-1` | `95000.0; bounds=(80000.0, 110000.0)` | `learnable` | Activation energy C_T1 of the isoprene Eq. (8) temperature activity (Guenther et al. 2012 Table 4 Isoprene row) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (C_T1 = 95 kJ mol-1); Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Eq. (5) |
| `biosphere.bvoc.megan.ct2` | `J mol-1` | `230000.0; bounds=(200000.0, 260000.0)` | `learnable` | Deactivation energy C_T2 of the Eq. (8) light-dependent temperature activity gamma_T,LDF; class independent (Guenther et al. 2012 Sect. 2.2) and used for every compound class | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Sect. 2.2 (C_T2 = 230 kJ mol-1); Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Eq. (5) |
| `biosphere.bvoc.megan.ct1_monoterpenes` | `J mol-1` | `80000.0; bounds=(60000.0, 100000.0)` | `learnable` | Activation energy C_T1 of the Eq. (8) light-dependent temperature activity of the lumped monoterpene class (Guenther et al. 2012 Table 4 Other Monoterpenes row) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Other Monoterpenes: C_T1 = 80 kJ mol-1) and Eq. (8) |
| `biosphere.bvoc.megan.eopt_monoterpenes` | `dimensionless` | `1.83; bounds=(1.0, 3.0)` | `learnable` | Optimum activity E_opt of the Eq. (8) light-dependent temperature activity of the lumped monoterpene class at the fixed standard history (T24 = T240 = T_s), the C_eo of the Guenther et al. (2012) Table 4 Other Monoterpenes row | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Other Monoterpenes: C_eo = 1.83) and Eq. (10) |
| `biosphere.bvoc.megan.ct1_sesquiterpenes` | `J mol-1` | `130000.0; bounds=(110000.0, 150000.0)` | `learnable` | Activation energy C_T1 of the Eq. (8) light-dependent temperature activity of the sesquiterpene class (Guenther et al. 2012 Table 4 Other Sesquiterpenes row) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Other Sesquiterpenes: C_T1 = 130 kJ mol-1) and Eq. (8) |
| `biosphere.bvoc.megan.eopt_sesquiterpenes` | `dimensionless` | `2.37; bounds=(1.0, 3.0)` | `learnable` | Optimum activity E_opt of the Eq. (8) light-dependent temperature activity of the sesquiterpene class at the fixed standard history (T24 = T240 = T_s), the C_eo of the Guenther et al. (2012) Table 4 Other Sesquiterpenes row | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Other Sesquiterpenes: C_eo = 2.37) and Eq. (10) |
| `biosphere.bvoc.megan.ovoc.methanol.global_emission` | `g yr-1` | `99600000000000.0; bounds=(99600000000000.0, 99600000000000.0)` | `fixed` | Global annual methanol emission for 2000 (99.6 Tg yr-1): weight of methanol in the emission-weighted mixture sum_j w_j gamma_j of the activity factors of the lumped OVOC class (methanol, acetone and acetaldehyde) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 6 (global annual emissions for 2000: methanol 99.6, acetone 43.7, acetaldehyde 20.7 Tg yr-1) |
| `biosphere.bvoc.megan.ovoc.methanol.beta` | `K-1` | `0.08; bounds=(0.05, 0.11)` | `learnable` | Temperature coefficient beta of the light-independent Eq. (11) activity of methanol, an OVOC component | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Methanol row) and Eq. (11) |
| `biosphere.bvoc.megan.ovoc.methanol.ldf` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Light-dependent fraction LDF of methanol, an OVOC component, in the Eq. (3) light and Eq. (7) temperature activities | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Methanol row), Eqs. (3) and (7) |
| `biosphere.bvoc.megan.ovoc.methanol.ct1` | `J mol-1` | `60000.0; bounds=(40000.0, 80000.0)` | `learnable` | Activation energy C_T1 of the Eq. (8) light-dependent temperature activity of methanol, an OVOC component | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Methanol row) and Eq. (8) |
| `biosphere.bvoc.megan.ovoc.methanol.eopt` | `dimensionless` | `1.6; bounds=(1.0, 3.0)` | `learnable` | Optimum activity E_opt of the Eq. (8) light-dependent temperature activity of methanol, an OVOC component, at the fixed standard history (T24 = T240 = T_s): the Table 4 C_eo | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Methanol row) and Eq. (10) |
| `biosphere.bvoc.megan.ovoc.acetone.global_emission` | `g yr-1` | `43700000000000.0; bounds=(43700000000000.0, 43700000000000.0)` | `fixed` | Global annual acetone emission for 2000 (43.7 Tg yr-1): weight of acetone in the emission-weighted mixture sum_j w_j gamma_j of the activity factors of the lumped OVOC class (methanol, acetone and acetaldehyde) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 6 (global annual emissions for 2000: methanol 99.6, acetone 43.7, acetaldehyde 20.7 Tg yr-1) |
| `biosphere.bvoc.megan.ovoc.acetone.beta` | `K-1` | `0.1; bounds=(0.07, 0.13)` | `learnable` | Temperature coefficient beta of the light-independent Eq. (11) activity of acetone, an OVOC component | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Acetone row) and Eq. (11) |
| `biosphere.bvoc.megan.ovoc.acetone.ldf` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Light-dependent fraction LDF of acetone, an OVOC component, in the Eq. (3) light and Eq. (7) temperature activities | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Acetone row), Eqs. (3) and (7) |
| `biosphere.bvoc.megan.ovoc.acetone.ct1` | `J mol-1` | `80000.0; bounds=(60000.0, 100000.0)` | `learnable` | Activation energy C_T1 of the Eq. (8) light-dependent temperature activity of acetone, an OVOC component | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Acetone row) and Eq. (8) |
| `biosphere.bvoc.megan.ovoc.acetone.eopt` | `dimensionless` | `1.83; bounds=(1.0, 3.0)` | `learnable` | Optimum activity E_opt of the Eq. (8) light-dependent temperature activity of acetone, an OVOC component, at the fixed standard history (T24 = T240 = T_s): the Table 4 C_eo | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Acetone row) and Eq. (10) |
| `biosphere.bvoc.megan.ovoc.acetaldehyde.global_emission` | `g yr-1` | `20700000000000.0; bounds=(20700000000000.0, 20700000000000.0)` | `fixed` | Global annual acetaldehyde emission for 2000 (20.7 Tg yr-1): weight of acetaldehyde in the emission-weighted mixture sum_j w_j gamma_j of the activity factors of the lumped OVOC class (methanol, acetone and acetaldehyde) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 6 (global annual emissions for 2000: methanol 99.6, acetone 43.7, acetaldehyde 20.7 Tg yr-1) |
| `biosphere.bvoc.megan.ovoc.acetaldehyde.beta` | `K-1` | `0.13; bounds=(0.08, 0.18)` | `learnable` | Temperature coefficient beta of the light-independent Eq. (11) activity of acetaldehyde (Guenther et al. 2012 Table 1 places acetaldehyde in the bidirectional VOC class), an OVOC component | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Bidirectional VOC row) and Eq. (11) |
| `biosphere.bvoc.megan.ovoc.acetaldehyde.ldf` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Light-dependent fraction LDF of acetaldehyde (Guenther et al. 2012 Table 1 places acetaldehyde in the bidirectional VOC class), an OVOC component, in the Eq. (3) light and Eq. (7) temperature activities | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Bidirectional VOC row), Eqs. (3) and (7) |
| `biosphere.bvoc.megan.ovoc.acetaldehyde.ct1` | `J mol-1` | `95000.0; bounds=(80000.0, 110000.0)` | `learnable` | Activation energy C_T1 of the Eq. (8) light-dependent temperature activity of acetaldehyde (Guenther et al. 2012 Table 1 places acetaldehyde in the bidirectional VOC class), an OVOC component | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Bidirectional VOC row) and Eq. (8) |
| `biosphere.bvoc.megan.ovoc.acetaldehyde.eopt` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `learnable` | Optimum activity E_opt of the Eq. (8) light-dependent temperature activity of acetaldehyde (Guenther et al. 2012 Table 1 places acetaldehyde in the bidirectional VOC class), an OVOC component, at the fixed standard history (T24 = T240 = T_s): the Table 4 C_eo | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Bidirectional VOC row) and Eq. (10) |
| `biosphere.bvoc.photosynthetic_coupling.f_carbon_isoprene` | `dimensionless` | `0.015; bounds=(0.005, 0.03)` | `learnable` | Fraction of gross photosynthesis allocated to isoprene synthesis (1.5 percent) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012; Monson and Fall (1989) Plant Physiology 90, 267-274, doi:10.1104/pp.90.1.267 |
| `biosphere.bvoc.photosynthetic_coupling.f_carbon_monoterpenes` | `dimensionless` | `0.005; bounds=(0.001, 0.02)` | `learnable` | Fraction of gross photosynthesis allocated to monoterpene synthesis (0.5%) | Source: grote_niinemets_2008; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.photosynthetic_coupling.f_carbon_sesquiterpenes` | `dimensionless` | `0.002; bounds=(0.0005, 0.01)` | `learnable` | Fraction of gross photosynthesis allocated to sesquiterpene synthesis (0.2 percent) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012 |
| `biosphere.bvoc.photosynthetic_coupling.f_carbon_ovocs` | `dimensionless` | `0.004; bounds=(0.001, 0.015)` | `learnable` | Fraction of gross photosynthesis allocated to OVOC synthesis (0.4 percent) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012 |
| `biosphere.bvoc.molecular_weights.monoterpenes` | `g mol-1` | `136.24; bounds=(0.0, 1362.4)` | `fixed` | Average molecular weight of monoterpenes (C10H16) | Source: chemical_database; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.molecular_weights.sesquiterpenes` | `g mol-1` | `204.36; bounds=(0.0, 2043.6000000000001)` | `fixed` | Average molecular weight of sesquiterpenes (C15H24) | Source: chemical_database; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.molecular_weights.ovocs` | `g mol-1` | `37.87; bounds=(0.0, 1000.0)` | `fixed` | Effective molar mass of the lumped OVOC class (methanol, acetone and acetaldehyde): MEGAN2.1 global-emission-weighted equivalent MW_eff = sum(w_i) / sum(w_i / MW_i) with the mass emissions w_i = 99.6, 43.7 and 20.7 Tg yr-1 (Guenther et al. 2012 Table 6) and MW_i = 32.042, 58.080 and 44.053 g mol-1 | Source: chemical_database; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 6 (global annual emissions for 2000: methanol 99.6, acetone 43.7, acetaldehyde 20.7 Tg yr-1); Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.carbon_atoms.isoprene` | `dimensionless` | `5; bounds=(0.0, 50.0)` | `fixed` | Number of carbon atoms in isoprene | Source: RELIC default; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.carbon_atoms.monoterpenes` | `dimensionless` | `10; bounds=(0.0, 100.0)` | `fixed` | Number of carbon atoms in monoterpenes | Source: RELIC default; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.carbon_atoms.sesquiterpenes` | `dimensionless` | `15; bounds=(0.0, 150.0)` | `fixed` | Number of carbon atoms in sesquiterpenes | Source: RELIC default; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.carbon_atoms.ovocs` | `dimensionless` | `1.456; bounds=(0.0, 60.0)` | `fixed` | Effective carbon atoms per molecule of the lumped OVOC class (methanol, acetone and acetaldehyde): n_C,eff = MW_eff f_C,eff / A_r(C) of the MEGAN2.1 global-emission-weighted equivalent (Guenther et al. 2012 Table 6: 99.6, 43.7 and 20.7 Tg yr-1) | Source: RELIC default; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 6 (global annual emissions for 2000: methanol 99.6, acetone 43.7, acetaldehyde 20.7 Tg yr-1); Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.foliar_density_factor` | `g DW m-2` | `200.0; bounds=(50.0, 500.0)` | `fixed` | Effective dry-leaf mass per unit canonical RELIC LAI leaf area for the reduced BVOC closure | Source: relic_reduced_bvoc_closure_assumption; RELIC reduced mass-normalized BVOC closure assumption; Guenther et al. (2012) supports only the MEGAN response-function heritage |
| `biosphere.bvoc.megan.beta_monoterpenes` | `K-1` | `0.1; bounds=(0.06, 0.12)` | `learnable` | Temperature coefficient beta of the light-independent Eq. (11) activity of the lumped monoterpene class (Guenther et al. 2012 Table 4 Other Monoterpenes row) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Other Monoterpenes: beta = 0.1 K-1) and Sect. 2.3 (MEGAN2.1 uses 0.1, slightly higher than the 0.09 recommended by Guenther et al. 1993) |
| `biosphere.bvoc.megan.beta_sesquiterpenes` | `K-1` | `0.17; bounds=(0.12, 0.22)` | `learnable` | Exponential temperature coefficient for sesquiterpenes (higher sensitivity) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4; Duhl et al. (2008) Biogeosciences 5, 761-777, doi:10.5194/bg-5-761-2008, Table 1 |
| `biosphere.bvoc.co2_inhibition.ismax` | `dimensionless` | `1.344; bounds=(1.0, 2.0)` | `learnable` | Maximum isoprene CO2 activity factor in the Guenther et al. equation 14 response | Source: guenther_2012_equation_14; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (14); Heald et al. (2009) Global Change Biology 15, 1127-1140, doi:10.1111/j.1365-2486.2008.01802.x |
| `biosphere.bvoc.co2_inhibition.c_star` | `ppm` | `585.0; bounds=(500.0, 700.0)` | `learnable` | CO2 scaling parameter for inhibition | Source: possell_hewitt_2011; Possell and Hewitt (2011) Global Change Biology 17, 1595-1610, doi:10.1111/j.1365-2486.2010.02306.x; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (14) |
| `biosphere.bvoc.co2_inhibition.hill_exponent` | `dimensionless` | `1.4614; bounds=(0.5, 3.0)` | `learnable` | Hill exponent in the Guenther et al. equation 14 isoprene CO2 response | Source: guenther_2012_equation_14; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (14); Wilkinson et al. (2009) Global Change Biology 15, 1189-1200, doi:10.1111/j.1365-2486.2008.01803.x |
| `biosphere.bvoc.co2_inhibition.intercellular_to_ambient_ratio` | `dimensionless` | `0.7; bounds=(0.5, 0.9)` | `fixed` | Ratio used to estimate intercellular CO2 from ambient CO2 for the reduced MEGAN response | Source: guenther_2012_equation_14; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Sect. 2.2 (C_i is 70 % of the ambient CO2 concentration); Heald et al. (2009) Global Change Biology 15, 1127-1140, doi:10.1111/j.1365-2486.2008.01802.x |
| `validation.numerical.carbon_flux_min` | `umol m-2 s-1` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Minimum carbon flux for photosynthesis coupling (non-negative) | Source: physical_constraints; Physical constraints |
| `validation.numerical.epsilon` | `dimensionless` | `1e-08; bounds=(1e-09, 0.0001)` | `fixed` | General epsilon value for numerical stability | Source: numerical_methods; IEEE 754 floating point standard |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |
| `validation.conservation.strict_conservation` | `dimensionless` | `False; bounds=None` | `fixed` | Whether to raise exception on conservation violation (true) or just log warning (false) | Source: validation_framework_design; Conservation law validation best practices |

### Physical constants (SSOT)

These values are resolved from the physical-constants database. They are not domain parameters and must not be duplicated in scientific YAML or Python code.

| Constant key | Units | Value and fixed bounds | Gradient | Meaning | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `relic.physical.constants.molecular_weights.isoprene` | `g mol-1` | `68.12; bounds=(68.12, 68.12)` | `fixed` | Molecular weight of isoprene (C5H8) | Source: IUPAC_CIAAW_atomic_weights; IUPAC Periodic Table of the Elements, 2021; CIAAW Standard Atomic Weights |
| `relic.physical.constants.conversions.micrograms_to_grams` | `g ug-1` | `1e-06; bounds=(1e-06, 1e-06)` | `fixed` | Conversion multiplier from micrograms to grams | Source: SI_definition; BIPM SI Brochure 9th Edition (2019) |
| `relic.physical.constants.universal.gas_constant` | `J mol-1 K-1` | `8.314462618; bounds=(8.314462618, 8.314462618)` | `fixed` | Universal gas constant | Source: relic_physical_constants; CODATA 2018 |
| `relic.physical.constants.time.seconds_per_hour` | `s` | `3600.0; bounds=(3600.0, 3600.0)` | `fixed` | Number of seconds in one hour | Source: relic_physical_constants; International System of Units |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{false} )$ for instantaneous fluxes and diagnostics |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\bvoc_emissions.md`.

### References

- Guenther et al. (2012). The Model of Emissions of Gases and Aerosols from Nature version 2.1 (MEGAN2.1): an extended and updated framework for modeling biogenic emissions. Geosci. Model Dev., 5, 1471-1492. (FunctorMetadata, certificate)
- Guenther et al. (2006). Estimates of global terrestrial isoprene emissions using MEGAN (Model of Emissions of Gases and Aerosols from Nature). Atmos. Chem. Phys., 6, 3181-3210. (FunctorMetadata)
- Niinemets et al. (2011). Estimations of isoprenoid emission capacity from enclosure studies: measurements, data processing, quality and standardized measurement protocols. Biogeosciences, 8, 2209-2246. (FunctorMetadata)
- Heald et al. (2009). Global Change Biology. doi:10.1111/j.1365-2486.2008.01802.x. (certificate)
- Wilkinson et al. (2009). Global Change Biology 15, 1189-1200. doi:10.1111/j.1365-2486.2008.01803.x. (certificate)

(class-relic-scientific-biosphere-bvoc-synthesis-isoprene-isoprenesynthesis)=
## IsopreneSynthesis

Mechanistic Isoprene Synthesis Model.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.bvoc.synthesis.isoprene.IsopreneSynthesis` |
| Source | `relic\scientific\biosphere\bvoc\synthesis\isoprene.py:318` |
| Runtime name | `mechanistic_isoprene_synthesis` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
IsopreneSynthesis(name: str = 'mechanistic_isoprene_synthesis', config: Dict[str, Any] | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'mechanistic_isoprene_synthesis'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | grid-cell air-temperature proxy mapped to canonical Phase 8 cochain basis |
| `gross_photosynthesis` | flux | `umol CO2 m-2 s-1` | gross_photosynthesis mapped to canonical Phase 8 cochain basis |
| `electron_transport_rate` | flux | `umol e- m-2 s-1` | electron_transport_rate mapped to canonical Phase 8 cochain basis |
| `atmospheric_co2_ppm` | forcing | `umol mol-1` | ambient CO2 proxy mapped to canonical Phase 8 cochain basis |
| `leaf_area_index` | state | `m2 m-2` | grid-cell leaf area per ground area used by the declared Beer-Lambert canopy bridge |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `isoprene_emission` | flux | `mol m-2 s-1` | decided at run time by the returned write mode | canonical mutually exclusive atmosphere-facing species port |
| `bvoc_carbon_allocation_flux` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | non-negative carbon allocation magnitude exposed to the plant carbon ledger |
| `diagnostics.bvoc.isoprene.emission_rate_molar` | diagnostic | `umol m-2 s-1` | diagnostics | species-resolved molar diagnostic emitted by the canonical IPHS runtime |
| `diagnostics.bvoc.isoprene.emission_rate_mass` | diagnostic | `ug m-2 s-1` | diagnostics | traditional mass observation isolated from the species port |
| `diagnostics.bvoc.isoprene.dmapp_concentration` | diagnostic | `umol m-3` | diagnostics | internal DMAPP observation |
| `diagnostics.bvoc.isoprene.mep_pathway_flux` | diagnostic | `umol m-2 s-1` | diagnostics | internal MEP flux observation |
| `diagnostics.bvoc.isoprene.synthase_activity` | diagnostic | `umol m-2 s-1` | diagnostics | internal synthase-rate observation |
| `diagnostics.bvoc.isoprene.co2_inhibition_factor` | diagnostic | `1` | diagnostics | internal CO2-response observation |
| `diagnostics.bvoc.isoprene.energy_limitation_factor` | diagnostic | `1` | diagnostics | internal energy-response observation |
| `diagnostics.bvoc.isoprene.g3p_availability` | diagnostic | `umol m-2 s-1` | diagnostics | internal G3P allocation observation |
| `diagnostics.bvoc.isoprene.pyruvate_availability` | diagnostic | `umol m-2 s-1` | diagnostics | internal pyruvate allocation observation |
| `diagnostics.bvoc.isoprene.atp_demand` | diagnostic | `umol m-2 s-1` | diagnostics | internal ATP-demand observation |
| `diagnostics.bvoc.isoprene.nadph_demand` | diagnostic | `umol m-2 s-1` | diagnostics | internal NADPH-demand observation |

### Governing equations and canonical operator

**Literature or process form**

MEP Pathway:

$$
\mathrm{G3P} + \mathrm{Pyruvate} \xrightarrow{\mathrm{DXS}} \mathrm{DXP} \xrightarrow{\mathrm{DXR}} \mathrm{MEP} \longrightarrow \mathrm{DMAPP}
$$

Isoprene Synthesis:

$$
\mathrm{DMAPP} \xrightarrow{\mathrm{IS}} \mathrm{Isoprene}
$$

$$
J_{\mathrm{DXS}} = V_{\mathrm{max},\mathrm{DXS}} \cdot [ \mathrm{G3P} ] \cdot [ \mathrm{Pyr} ] / ( ( K_{\mathrm{G3P}} + [ \mathrm{G3P} ] ) \cdot ( K_{\mathrm{Pyr}} + [ \mathrm{Pyr} ] ) )
$$

$$
J_{\mathrm{IS}} ( T_{\mathrm{air}} ) = V_{\mathrm{max},\mathrm{IS}} ( T_{\mathrm{air}} ) \cdot [ \mathrm{DMAPP} ] / ( K_{\mathrm{DMAPP}} + [ \mathrm{DMAPP} ] ) \cdot K_{\mathrm{CO2}} / ( K_{\mathrm{CO2}} + C_{a} )
$$

$$
A_{\mathrm{ATP}} = J_{\mathrm{DMAPP}} \cdot \nu_{\mathrm{ATP}}; A_{\mathrm{NADPH}} = J_{\mathrm{DMAPP}} \cdot \nu_{\mathrm{NADPH}}
$$

The exact proxy and kinetic closure is ParameterManager-backed and is not claimed as a universal leaf-scale law.

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |
| `effective_leaf_area` | $L_{\mathrm{eff}} = ( 1 - \exp ( - k \mathrm{LAI} ) ) / k \mathrm{maps} \mathrm{leaf} - \mathrm{area} \mathrm{kinetics}$ to ground-area ports |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | IsopreneSynthesis.forward calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical outputs. |
| `explicit_stoichiometric_usage` | IsopreneSynthesis.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `air_temperature`, `gross_photosynthesis`, `electron_transport_rate`, `atmospheric_co2_ppm`, `leaf_area_index`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.bvoc.mep.v_max_dxs` | `umol m-2 s-1` | `2.5; bounds=(1.0, 5.0)` | `learnable` | Maximum reaction rate of DXS enzyme in MEP pathway | Source: xiao_vickers_2009_2012; RELIC model default; no verified literature source is cited for this value |
| `biosphere.bvoc.mep.k_g3p` | `umol m-3` | `500.0; bounds=(100.0, 1500.0)` | `learnable` | Michaelis constant for G3P in DXS reaction | Source: vickers_et_al_2009; RELIC model default; no verified literature source is cited for this value |
| `biosphere.bvoc.mep.k_pyruvate` | `umol m-3` | `800.0; bounds=(200.0, 2000.0)` | `learnable` | Michaelis constant for pyruvate in DXS reaction | Source: vickers_et_al_2009; RELIC model default; no verified literature source is cited for this value |
| `biosphere.bvoc.mep.q10_dxs` | `dimensionless` | `2.3; bounds=(1.8, 3.0)` | `learnable` | Q10 temperature coefficient for DXS enzyme activity | Source: sharkey_et_al_2008; Sharkey et al. (2008) Annals of Botany 101, 5-18, doi:10.1093/aob/mcm240 |
| `biosphere.bvoc.mep.atp_per_dmapp` | `dimensionless` | `5.0; bounds=(4.0, 6.0)` | `fixed` | ATP molecules required per DMAPP produced via MEP pathway | Source: eisenreich_et_al_2004; Eisenreich et al. (2004) Cellular and Molecular Life Sciences 61(12), doi:10.1007/s00018-004-3381-z |
| `biosphere.bvoc.mep.nadph_per_dmapp` | `dimensionless` | `2.0; bounds=(2.0, 4.0)` | `fixed` | NADPH molecules required per DMAPP produced via MEP pathway | Source: eisenreich_et_al_2004; Eisenreich et al. (2004) Cellular and Molecular Life Sciences 61(12), doi:10.1007/s00018-004-3381-z |
| `biosphere.bvoc.isoprene_synthase.v_max` | `umol m-2 s-1` | `5.0; bounds=(2.0, 10.0)` | `learnable` | Maximum velocity of isoprene synthase enzyme (leaf-area basis) | Source: rasulov_morfopoulos_compiled; Rasulov et al. (2009) Plant Physiology 149, 1609-1618, doi:10.1104/pp.108.133512; Morfopoulos et al. (2014) New Phytologist 203, 125-139, doi:10.1111/nph.12770 |
| `biosphere.bvoc.isoprene_synthase.k_dmapp` | `umol m-3` | `10.0; bounds=(5.0, 20.0)` | `learnable` | Michaelis constant of isoprene synthase for DMAPP (substrate) | Source: rasulov_schnitzler_compiled; Rasulov et al. (2009) Plant Physiology 149, 1609-1618, doi:10.1104/pp.108.133512; Schnitzler et al. (2010) Plant Biology 12, 302-316, doi:10.1111/j.1438-8677.2009.00284.x |
| `biosphere.bvoc.isoprene_synthase.k_co2_inhibition` | `umol mol-1` | `300.0; bounds=(200.0, 400.0)` | `fixed` | CO2 inhibition constant for isoprene synthase (half-saturation) | Source: RELIC parameter database; Wilkinson et al. (2009) Global Change Biology 15, 1189-1200, doi:10.1111/j.1365-2486.2008.01803.x; Monson and Fall (1989) Plant Physiology 90, 267-274, doi:10.1104/pp.90.1.267 |
| `biosphere.bvoc.isoprene_synthase.q10` | `dimensionless` | `2.3; bounds=(2.0, 2.6)` | `learnable` | Q10 temperature sensitivity factor for isoprene synthase catalytic activity | Source: rasulov_temperature_compiled; Monson and Fall (1989) Plant Physiology 90, 267-274, doi:10.1104/pp.90.1.267, abstract (Q10 1.8-2.4 for leaf isoprene emission, 35-45 degC); Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012 |
| `biosphere.bvoc.isoprene_synthase.enable_co2_inhibition` | `dimensionless` | `True; bounds=(0.0, 100.0)` | `fixed` | Enables CO2 inhibition feedback on isoprene synthase activity based on Wilkinson et al. (2009) | Source: relic_mechanistic_update; Wilkinson et al. (2009) Global Change Biology 15, 1189-1200, doi:10.1111/j.1365-2486.2008.01803.x |
| `biosphere.bvoc.carbon_fraction.isoprene` | `dimensionless` | `0.882; bounds=(0.8, 0.92)` | `fixed` | Carbon mass fraction for isoprene (C5H8, 60 g C per 68 g compound) | Source: chemical_stoichiometry; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.isoprene_synthase.f_g3p` | `dimensionless` | `0.5; bounds=(0.3, 0.7)` | `learnable` | Fraction of G3P from photosynthesis allocated to MEP pathway | Source: vickers_et_al_2009; RELIC model default; no verified literature source is cited for this value |
| `biosphere.bvoc.isoprene_synthase.f_pyruvate` | `dimensionless` | `0.3; bounds=(0.1, 0.5)` | `learnable` | Fraction of pyruvate from photosynthesis allocated to MEP pathway | Source: vickers_et_al_2009; RELIC model default; no verified literature source is cited for this value |
| `biosphere.bvoc.isoprene_synthase.g3p_flux_to_concentration` | `s m-1` | `10.0; bounds=(5.0, 20.0)` | `fixed` | Scaling factor converting G3P flux (umol m-2 s-1) to cytosolic concentration (umol m-3) | Source: relic_mechanistic_update; Lichtenthaler (1999) Annu. Rev. Plant Physiol. Plant Mol. Biol. 50, 47-65, doi:10.1146/annurev.arplant.50.1.47 |
| `biosphere.bvoc.isoprene_synthase.pyruvate_flux_to_concentration` | `s m-1` | `15.0; bounds=(7.5, 30.0)` | `fixed` | Scaling factor converting pyruvate flux (umol m-2 s-1) to cytosolic concentration (umol m-3) | Source: relic_mechanistic_update; Lichtenthaler (1999) Annu. Rev. Plant Physiol. Plant Mol. Biol. 50, 47-65, doi:10.1146/annurev.arplant.50.1.47 |
| `biosphere.bvoc.isoprene_synthase.dmapp_flux_to_concentration` | `s m-1` | `20.0; bounds=(10.0, 40.0)` | `fixed` | Scaling factor converting DMAPP production flux to pool concentration under steady-state assumption | Source: relic_mechanistic_update; Rasulov et al. (2009) Plant Physiology 149, 1609-1618, doi:10.1104/pp.108.133512; Morfopoulos et al. (2014) New Phytologist 203, 125-139, doi:10.1111/nph.12770 |
| `biosphere.bvoc.isoprene_synthase.atp_per_electron_transport` | `dimensionless` | `0.75; bounds=(0.5, 1.0)` | `fixed` | ATP equivalents produced per unit electron transport rate delivered to the MEP pathway | Source: relic_mechanistic_update; RELIC model default; no verified literature source is cited for this value |
| `biosphere.bvoc.isoprene_synthase.nadph_per_electron_transport` | `dimensionless` | `0.5; bounds=(0.25, 0.75)` | `fixed` | NADPH equivalents produced per unit electron transport supporting the MEP pathway | Source: relic_mechanistic_update; RELIC model default; no verified literature source is cited for this value |
| `validation.numerical.coupling_max` | `dimensionless` | `1.0; bounds=(1.0, 1.0)` | `fixed` | Maximum photosynthesis-BVOC coupling factor | Source: physical_constraints; Physical constraints |
| `biosphere.photosynthesis.canopy_light_extinction` | `dimensionless` | `0.5; bounds=(0.2, 0.9)` | `fixed` | Beer-Lambert canopy light extinction coefficient for LAI scaling in canopy photosynthesis | Source: literature_synthesis; Monsi and Saeki (1953) Japanese Journal of Botany 14:22-52; Monteith and Unsworth (2013) Principles of Environmental Physics |
| `validation.numerical.epsilon` | `dimensionless` | `1e-08; bounds=(1e-09, 0.0001)` | `fixed` | General epsilon value for numerical stability | Source: numerical_methods; IEEE 754 floating point standard |
| `validation.numerical.epsilon_energy` | `dimensionless` | `1e-06; bounds=(1e-07, 0.001)` | `fixed` | Epsilon for energy-demand denominators in BVOC pathway calculations | Source: numerical_methods; Numerical methods best practices |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Physical constants (SSOT)

These values are resolved from the physical-constants database. They are not domain parameters and must not be duplicated in scientific YAML or Python code.

| Constant key | Units | Value and fixed bounds | Gradient | Meaning | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `relic.physical.constants.molecular_weights.isoprene` | `g mol-1` | `68.12; bounds=(68.12, 68.12)` | `fixed` | Molecular weight of isoprene (C5H8) | Source: IUPAC_CIAAW_atomic_weights; IUPAC Periodic Table of the Elements, 2021; CIAAW Standard Atomic Weights |
| `relic.physical.constants.temperature.reference_temperature` | `K` | `298.15; bounds=(298.15, 298.15)` | `fixed` | Standard reference temperature (25degC) | Source: relic_physical_constants; IUPAC Gold Book |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\bvoc_emissions.md`.

### References

- Guenther et al. (2012). The Model of Emissions of Gases and Aerosols from Nature version 2.1 (MEGAN2.1). Geoscientific Model Development. (FunctorMetadata, certificate)
- Lichtenthaler (1999). The 1-deoxy-D-xylulose-5-phosphate pathway of isoprenoid biosynthesis in plants. Annual Review of Plant Physiology and Plant Molecular Biology. (FunctorMetadata, certificate)
- Rasulov et al. (2009). Evidence that light, carbon dioxide, and oxygen dependencies of leaf isoprene emission are driven by energy status in hybrid aspen. Plant Physiology. (FunctorMetadata, certificate)
- Vranova et al. (2013). Network analysis of the MVA and MEP pathways for isoprenoid synthesis. Annual Review of Plant Biology. (FunctorMetadata, certificate)
- de Pury and Farquhar (1997). Simple scaling of photosynthesis from leaves to canopies without the errors of big-leaf models. Plant, Cell and Environment. (FunctorMetadata, certificate)
- Koppmann (2007). Volatile Organic Compounds in the Atmosphere. Wiley. (FunctorMetadata, certificate)
- Monsi and Saeki (1953). Uber den Lichtfaktor in den Pflanzengesellschaften und seine Bedeutung fur die Stoffproduktion. Japanese Journal of Botany 14, 22-52. (certificate)

(class-relic-scientific-biosphere-bvoc-synthesis-monoterpene-monoterpenesynthesis)=
## MonoterpeneSynthesis

Phenomenological Stored-Pool Monoterpene Synthesis and Emission Model.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.bvoc.synthesis.monoterpene.MonoterpeneSynthesis` |
| Source | `relic\scientific\biosphere\bvoc\synthesis\monoterpene.py:285` |
| Runtime name | `mechanistic_monoterpene_synthesis` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MonoterpeneSynthesis(name: str = 'mechanistic_monoterpene_synthesis', config: Dict[str, Any] | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'mechanistic_monoterpene_synthesis'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | Grid-cell emitting-foliage temperature proxy supplied through the canonical external forcing state |
| `gross_photosynthesis` | flux | `umol CO2 m-2 s-1` | gross_photosynthesis mapped to canonical Phase 8 cochain basis |
| `monoterpene_pool_concentration` | state | `umol m-2` | StateManager-owned storage cochain used as the unique pool state |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `monoterpenes_emission` | flux | `mol m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `monoterpene_pool_concentration` | state | `umol m-2` | decided at run time by the returned write mode | StateManager-owned storage cochain used as the unique pool state |
| `diagnostics.bvoc.monoterpene.emission_rate_molar` | output port | `umol m-2 s-1` | diagnostics | species-resolved molar diagnostic emitted by the canonical IPHS runtime |
| `diagnostics.bvoc.monoterpene.emission_rate_mass` | diagnostic | `ug m-2 s-1` | diagnostics | mass-basis emission diagnostic, not an alternative atmosphere port |
| `diagnostics.bvoc.monoterpene.pool_tendency` | output port | `umol m-2 s-1` | diagnostics | bounded pool tendency produced by the canonical reaction runtime |
| `diagnostics.bvoc.monoterpene.pool_delta` | output port | `umol m-2` | diagnostics | exact per-step pool increment diagnostic |
| `diagnostics.bvoc.monoterpene.pool_boundary_residual_rate` | output port | `umol m-2 s-1` | diagnostics | explicit signed routing of the bounded-update residual |
| `diagnostics.bvoc.monoterpene.storage_change` | output port | `kg C m-2 s-1` | diagnostics | carbon-mass storage tendency derived from the canonical pool tendency |
| `diagnostics.bvoc.monoterpene.synthesis_rate` | diagnostic | `umol m-2 s-1` | diagnostics | storage-pool synthesis diagnostic |
| `diagnostics.bvoc.monoterpene.pool_capacity_utilization` | diagnostic | `1` | diagnostics | bounded-pool utilization diagnostic |
| `diagnostics.bvoc.monoterpene.temperature_emission_factor` | diagnostic | `1` | diagnostics | temperature-dependent pool release diagnostic normalized at the physical-constant reference temperature |
| `diagnostics.bvoc.monoterpene.carbon_flux_allocated` | output port | `umol CO2 m-2 s-1` | diagnostics | photosynthetic allocation diagnostic |
| `diagnostics.bvoc.monoterpene.degradation_rate` | output port | `umol m-2 s-1` | diagnostics | storage-pool degradation diagnostic |
| `diagnostics.bvoc.monoterpene.seasonal_factor` | output port | `1` | diagnostics | seasonal response diagnostic |
| `bvoc_carbon_allocation_flux` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | positive plant-carbon allocation into stored-pool monoterpene synthesis; distinct from atmosphere-facing stored-pool release |

### Governing equations and canonical operator

**Literature or process form**

Stored-pool studies support separating synthesis from release and a strong temperature response of release. The exact GPP saturation, Q10 synthesis, capacity limiter, and degradation coefficients are calibratable ParameterManager-backed phenomenological choices, not a universal source law.

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | air temperature, gross photosynthesis, and pool concentration are empirical drivers; no thermodynamic effort is claimed |
| `state_advancer` | the bounded pool ledger is evaluated once, and its rate and port cochains are evaluated as $\dot{x} = N_{h} r_{h}$ by advance_reaction_network with an explicit identity $N_{h}$ |
| `runtime_usage` | MonoterpeneSynthesis.forward evaluates one bounded storage ledger, exposes its residual, and wraps the resulting rate and port cochains through the existing reaction-network runtime. |
| `explicit_stoichiometric_usage` | the identity $N_{h}$ wrapper covers emission, synthesis, positive plant-carbon allocation, degradation, pool-tendency, storage-carbon, and boundary-residual cochains; this is level-2 wrapping rather than strict stoichiometric closure. |

### Composition channels and executable PortSpec API

Certificate composition channels: `monoterpenes_emission`, `bvoc_carbon_allocation_flux`, `monoterpene_pool_concentration`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.bvoc.monoterpene.v_max_mts` | `umol m-2 s-1` | `1.2; bounds=(0.5, 3.0)` | `learnable` | Calibratable maximum areal rate in the phenomenological monoterpene synthesis response | Source: relic_phenomenological_calibration; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975; Ghirardo et al. (2010) Plant, Cell & Environment 33, 781-792, doi:10.1111/j.1365-3040.2009.02104.x |
| `biosphere.bvoc.monoterpene.k_gpp` | `umol m-2 s-1` | `15.0; bounds=(8.0, 30.0)` | `fixed` | Calibratable allocated photosynthetic-carbon half-saturation scale in the phenomenological synthesis response | Source: relic_phenomenological_calibration; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.monoterpene.q10_synthase` | `dimensionless` | `2.5; bounds=(1.8, 3.2)` | `learnable` | Q10 coefficient for the calibratable monoterpene synthesis response | Source: relic_phenomenological_calibration; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.monoterpene.pool.capacity` | `umol m-2` | `734.0; bounds=(146.8, 2936.0)` | `fixed` | Maximum monoterpene storage pool capacity converted to area basis using foliar density and monoterpene molecular weight | Source: niinemets_grote_2008; Niinemets et al. (2010) Biogeosciences 7, 1809-1832, doi:10.5194/bg-7-1809-2010; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.monoterpene.pool_initial` | `umol m-2` | `50.0; bounds=(0.0, 500.0)` | `fixed` | Initial monoterpene storage pool concentration at model initialization | Source: grote_niinemets_2008; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.monoterpene.pool.k_degradation` | `s-1` | `5.556e-07; bounds=(1.389e-07, 1.389e-06)` | `learnable` | Degradation rate constant for monoterpenes in storage pool | Source: niinemets_et_al_2010; Niinemets et al. (2010) Biogeosciences 7, 1809-1832, doi:10.5194/bg-7-1809-2010 |
| `biosphere.bvoc.monoterpene.pool.k_emit_25` | `s-1` | `2.222e-05; bounds=(8.333e-06, 4.167e-05)` | `learnable` | Emission rate constant from storage pool at 25 degC | Source: grote_niinemets_2008; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.monoterpene.pool.beta_emission` | `K-1` | `0.09; bounds=(0.06, 0.12)` | `learnable` | Temperature response coefficient for pool emission | Source: grote_niinemets_2008; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Sect. 2.3 (beta = 0.09 recommended by Guenther et al. 1993); Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.photosynthetic_coupling.f_carbon_monoterpenes` | `dimensionless` | `0.005; bounds=(0.001, 0.02)` | `learnable` | Fraction of gross photosynthesis allocated to monoterpene synthesis (0.5%) | Source: grote_niinemets_2008; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.molecular_weights.monoterpenes` | `g mol-1` | `136.24; bounds=(0.0, 1362.4)` | `fixed` | Average molecular weight of monoterpenes (C10H16) | Source: chemical_database; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.carbon_fraction.monoterpenes` | `dimensionless` | `0.882; bounds=(0.8, 0.92)` | `fixed` | Carbon mass fraction for monoterpenes (C10H16, 120 g C per 136 g compound) | Source: chemical_stoichiometry; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.monoterpene.enable_seasonal_variation` | `dimensionless` | `False; bounds=None` | `fixed` | Enable seasonal variation in monoterpene synthesis rates (sinusoidal modulation) | Source: relic_parameter_team; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012 |
| `biosphere.bvoc.monoterpene.seasonal.amplitude` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `learnable` | Amplitude of seasonal modulation for monoterpene synthesis (peak deviation from unity) | Source: seasonal_emission_synthesis; Niinemets et al. (2010) Biogeosciences 7, 1809-1832, doi:10.5194/bg-7-1809-2010; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.monoterpene.seasonal.minimum_factor` | `dimensionless` | `0.1; bounds=(0.0, 0.5)` | `fixed` | Minimum seasonal modulation factor to prevent negative synthesis flux | Source: seasonal_emission_synthesis; Niinemets et al. (2010) Biogeosciences 7, 1809-1832, doi:10.5194/bg-7-1809-2010; Grote and Niinemets (2008) Plant Biology 10, 8-28, doi:10.1055/s-2007-964975 |
| `biosphere.bvoc.monoterpene.seasonal.peak_day` | `day` | `150.0; bounds=(90.0, 210.0)` | `learnable` | Day of year when monoterpene synthesis peaks (Northern Hemisphere default) | Source: seasonal_emission_synthesis; Niinemets et al. (2010) Biogeosciences 7, 1809-1832, doi:10.5194/bg-7-1809-2010; Tingey et al. (1980) Plant Physiology 65, 797-801, doi:10.1104/pp.65.5.797 |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\bvoc_emissions.md`.

### References

- Tingey, D.T., et al. (1980). Influence of light and temperature on monoterpene emission rates from slash pine. Plant Physiology 65, 797-801. (FunctorMetadata, certificate)
- Keeling, C.I. & Bohlmann, J. (2006). Genes, enzymes and chemicals of terpenoid diversity in the constitutive and induced defence of conifers against insects and pathogens. New Phytologist 170, 657-675. (FunctorMetadata, certificate)
- Dudareva, N., et al. (2006). Plant volatiles: recent advances and future perspectives. Critical Reviews in Plant Sciences 25, 417-440. (FunctorMetadata, certificate)
- Niinemets, U. & Reichstein, M. (2002). A model analysis of the effects of nonspecific monoterpenoid storage in leaf tissues on emission kinetics. Global Biogeochemical Cycles 16, 1110. (FunctorMetadata, certificate)
- Niinemets, U., et al. (2014). Bidirectional exchange of biogenic volatiles with vegetation: emission sources, reactions, breakdown and deposition. Plant, Cell & Environment 37, 1790-1809. (certificate)

(class-relic-scientific-biosphere-bvoc-synthesis-sesquiterpene-sesquiterpenestressmodel)=
## SesquiterpeneStressModel

Sesquiterpene stress response observation with threshold activation.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.bvoc.synthesis.sesquiterpene.SesquiterpeneStressModel` |
| Source | `relic\scientific\biosphere\bvoc\synthesis\sesquiterpene.py:781` |
| Runtime name | `sesquiterpene_stress_model` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `observation` |
| Spatial scale | `grid_cell` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
SesquiterpeneStressModel(name: str = 'sesquiterpene_stress_model', config: Dict[str, Any] | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'sesquiterpene_stress_model'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `stress_level` | forcing | `1` | stress_level mapped to canonical Phase 8 cochain basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `stress_response_factor` | diagnostic | `1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |

### Governing equations and canonical operator

**Literature or process form**

$f = \min ( \max ( 1 + \max ( S - S_{\mathrm{th}} , 0 ) \cdot A , 1 ) , 1 + A )$: RELIC reduced threshold-linear calibration of the stress dose-response documented in docs/source/theory/bvoc_emissions.md; per-stress weights and a recovery state are not part of the executed law

**Canonical RELIC structural form**

$y_{h} = O_{h} x_{h}$ through the declared observation operator

| Entry | Meaning |
| --- | --- |
| `O_h` | declared observation map |
| `H_h` | diagnostic metric or identity Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $y_{h} = O_{h} x_{h}$ through the declared observation operator |
| `runtime_usage` | SesquiterpeneStressModel.forward reads stress_level only from the declared state dictionary and returns the canonical stress_response_factor dictionary through the production output path. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.bvoc.sesquiterpene.stress_threshold` | `dimensionless` | `0.7; bounds=(0.5, 0.9)` | `fixed` | Stress level threshold for sesquiterpene emission activation | Source: duhl_et_al_2008; Duhl et al. (2008) Biogeosciences 5, 761-777, doi:10.5194/bg-5-761-2008 |
| `biosphere.bvoc.sesquiterpene.stress_amplification` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `learnable` | Amplification factor for stress-induced sesquiterpene emissions | Source: duhl_et_al_2008; Duhl et al. (2008) Biogeosciences 5, 761-777, doi:10.5194/bg-5-761-2008 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | one UpdateItem($\mathrm{mode} = \mathrm{REPLACE} , \mathrm{unit} = 1 , \mathrm{per}_{\mathrm{timestep}} = \mathrm{False} ) \mathrm{committed}$ through the existing StateManager update path; the algebraic factor is never accumulated |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\bvoc_emissions.md`.

### References

- Copolovici et al. (2012). Journal of Plant Physiology 169:664-672 (FunctorMetadata, certificate)
- Loreto & Schnitzler (2010). Trends in Plant Science 15:154-166 (FunctorMetadata, certificate)
- Niinemets et al. (2013). Frontiers in Plant Science 4:262 (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-bvoc-synthesis-sesquiterpene-sesquiterpenesynthesis)=
## SesquiterpeneSynthesis

Sesquiterpene synthesis model with MEGAN-style activity factors.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.bvoc.synthesis.sesquiterpene.SesquiterpeneSynthesis` |
| Source | `relic\scientific\biosphere\bvoc\synthesis\sesquiterpene.py:55` |
| Runtime name | `sesquiterpene_synthesis` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
SesquiterpeneSynthesis(name: str = 'sesquiterpene_synthesis', config: Dict[str, Any] | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'sesquiterpene_synthesis'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `temperature` | forcing | `K` | required canopy temperature input in K |
| `leaf_area_index` | state | `m2 m-2` | required finite nonnegative canonical RELIC leaf area index in m2 leaf m-2 ground |
| `ppfd` | flux | `umol photons m-2 s-1` | optional photon-flux input in umol photons m-2 s-1 |
| `soil_moisture` | state | `m3 m-3` | optional dimensionless soil-moisture stress driver |
| `water_stress_factor` | diagnostic | `dimensionless` | optional dimensionless water-stress factor |
| `temperature_stress_factor` | diagnostic | `dimensionless` | optional dimensionless temperature-stress factor |
| `nutrient_stress_factor` | diagnostic | `dimensionless` | optional dimensionless nutrient-stress factor |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `sesquiterpenes_emission` | flux | `mol m-2 s-1` | decided at run time by the returned write mode | canonical open atmospheric molar-flux port |
| `bvoc_carbon_allocation_flux` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | positive no-storage plant-carbon allocation proxy equal to emitted sesquiterpene carbon |
| `diagnostics.bvoc.sesquiterpene.emission_rate_molar` | output port | `umol m-2 s-1` | diagnostics | species-resolved molar diagnostic |
| `diagnostics.bvoc.sesquiterpene.gamma_temperature` | output port | `1` | diagnostics | Eq. (7) temperature activity diagnostic |
| `diagnostics.bvoc.sesquiterpene.gamma_light` | output port | `1` | diagnostics | optional light activity diagnostic |
| `diagnostics.bvoc.sesquiterpene.gamma_stress` | output port | `1` | diagnostics | combined environmental stress diagnostic |
| `diagnostics.bvoc.sesquiterpene.gamma_total` | output port | `1` | diagnostics | product of the three activity diagnostics |

### Governing equations and canonical operator

**Literature or process form**

The light factor is Guenther et al. (2012) Eq. (3) with the Table 4 sesquiterpene light-dependent fraction and the Eq. (4) light-dependent activity; the temperature factor is the Eq. (7) blend of the Eq. (11) exponential about the MEGAN standard temperature T_s and the Eq. (8) light-dependent response, with the Table 4 Other Sesquiterpenes row; stress is the most limiting supplied factor. The RELIC reduced closure converts a dry-leaf-mass-normalized factor through LAI k_f; it is not the canonical MEGAN2.1 vegetation-type, ground-area emission-factor formulation. Stored-emitter literature requires synthesis and release to remain distinct when storage is represented; this no-storage closure reports emitted compound carbon only as an explicit allocation proxy.

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | SesquiterpeneSynthesis.forward validates the declared state ports, evaluates one MEGAN-family empirical response rate, maps the mass-normalized rate through required LAI and ParameterManager-backed $k_{f}$, converts it through the existing UnitConverter, derives the positive no-storage plant-carbon allocation proxy, calls advance_reaction_network with explicit stoichiometric_N and rates, and emits the canonical molar and carbon ports plus namespaced diagnostics. |
| `explicit_stoichiometric_usage` | SesquiterpeneSynthesis.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network with `open_system_route=port`. |

### Composition channels and executable PortSpec API

Certificate composition channels: `sesquiterpenes_emission`, `bvoc_carbon_allocation_flux`, `temperature`, `leaf_area_index`, `ppfd`, `soil_moisture`, `water_stress_factor`, `temperature_stress_factor`, `nutrient_stress_factor`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.bvoc.default.sesquiterpenes` | `ug g-1 DW h-1` | `0.45; bounds=(0.1, 2.0)` | `fixed` | Default sesquiterpene emission factor for unknown vegetation types | Source: default_default; Duhl et al. (2008) Biogeosciences 5, 761-777, doi:10.5194/bg-5-761-2008, Sect. 3.1; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 2 |
| `biosphere.bvoc.megan.beta_sesquiterpenes` | `K-1` | `0.17; bounds=(0.12, 0.22)` | `learnable` | Exponential temperature coefficient for sesquiterpenes (higher sensitivity) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4; Duhl et al. (2008) Biogeosciences 5, 761-777, doi:10.5194/bg-5-761-2008, Table 1 |
| `biosphere.bvoc.molecular_weights.sesquiterpenes` | `g mol-1` | `204.36; bounds=(0.0, 2043.6000000000001)` | `fixed` | Average molecular weight of sesquiterpenes (C15H24) | Source: chemical_database; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.carbon_fraction.sesquiterpenes` | `dimensionless` | `0.882; bounds=(0.8, 0.92)` | `fixed` | Carbon mass fraction for sesquiterpenes (C15H24, 180 g C per 204 g compound) | Source: chemical_stoichiometry; Prohaska et al. (2022) Pure Appl. Chem. 94, 573-600, doi:10.1515/pac-2019-0603 (standard atomic weights) |
| `biosphere.bvoc.foliar_density_factor` | `g DW m-2` | `200.0; bounds=(50.0, 500.0)` | `fixed` | Effective dry-leaf mass per unit canonical RELIC LAI leaf area for the reduced BVOC closure | Source: relic_reduced_bvoc_closure_assumption; RELIC reduced mass-normalized BVOC closure assumption; Guenther et al. (2012) supports only the MEGAN response-function heritage |
| `biosphere.bvoc.t_standard` | `K` | `303.15; bounds=(298.15, 308.15)` | `fixed` | Standard temperature T_s of the light-independent Eq. (11) response exp(beta (T - T_s)); 303.15 K (30 degC) is paired with the reference temperature of the RELIC dry-mass emission factors (basal emission rates normalized to 30 degC; standard air temperature 303 K of the MEGAN canopy factors). Guenther et al. (2012) use T_s = 297 K together with the MEGAN2.1 area-based Table 2 emission factors; the two pairs must not be mixed. | Source: megan_v2.1_model; Duhl et al. (2008) Biogeosciences 5, 761-777, doi:10.5194/bg-5-761-2008, Sect. 2.1, Eq. (1) (basal emission rates in ug gDW-1 h-1 normalized to a standard temperature, usually 30 degC); Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Sect. 2.1 (standard air temperature 303 K) |
| `biosphere.bvoc.megan.alpha_light` | `m2 s umol-1` | `0.0027; bounds=(0.001, 0.005)` | `learnable` | Light response coefficient for BVOC emissions | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (5); Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Eq. (6) |
| `biosphere.bvoc.megan.cl_light` | `dimensionless` | `1.066; bounds=(0.8, 1.3)` | `learnable` | Light scaling factor for MEGAN light response | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (6); Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Eq. (7) |
| `biosphere.bvoc.megan.ldf_sesquiterpenes` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Light-dependent fraction LDF of sesquiterpene emission in the MEGAN2.1 light activity gamma_L = (1 - LDF) + LDF gamma_P_LDF | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (3) and Table 4 (LDF = 0.5 for alpha-farnesene, beta-caryophyllene and other sesquiterpenes) |
| `biosphere.bvoc.megan.eopt_sesquiterpenes` | `dimensionless` | `2.37; bounds=(1.0, 3.0)` | `learnable` | Optimum activity E_opt of the Eq. (8) light-dependent temperature activity of the sesquiterpene class at the fixed standard history (T24 = T240 = T_s), the C_eo of the Guenther et al. (2012) Table 4 Other Sesquiterpenes row | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Other Sesquiterpenes: C_eo = 2.37) and Eq. (10) |
| `biosphere.bvoc.megan.ct1_sesquiterpenes` | `J mol-1` | `130000.0; bounds=(110000.0, 150000.0)` | `learnable` | Activation energy C_T1 of the Eq. (8) light-dependent temperature activity of the sesquiterpene class (Guenther et al. 2012 Table 4 Other Sesquiterpenes row) | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Table 4 (Other Sesquiterpenes: C_T1 = 130 kJ mol-1) and Eq. (8) |
| `biosphere.bvoc.megan.topt` | `K` | `313.0; bounds=(298.0, 313.0)` | `learnable` | Optimum temperature T_opt of the Eq. (8) light-dependent temperature activity gamma_T,LDF, Eq. (9) T_opt = 313 + 0.6 (T240 - 297) at the fixed standard history T240 = T_s = 297 K, so T_opt = 313 K; class independent and used for every compound class (Guenther et al. 2012) | Source: megan_v2.1_model; Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Eq. (8); Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Eq. (9) |
| `biosphere.bvoc.megan.ct2` | `J mol-1` | `230000.0; bounds=(200000.0, 260000.0)` | `learnable` | Deactivation energy C_T2 of the Eq. (8) light-dependent temperature activity gamma_T,LDF; class independent (Guenther et al. 2012 Sect. 2.2) and used for every compound class | Source: megan_v2.1_model; Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, doi:10.5194/gmd-5-1471-2012, Sect. 2.2 (C_T2 = 230 kJ mol-1); Guenther et al. (2006) Atmos. Chem. Phys. 6, 3181-3210, doi:10.5194/acp-6-3181-2006, Eq. (5) |

### Physical constants (SSOT)

These values are resolved from the physical-constants database. They are not domain parameters and must not be duplicated in scientific YAML or Python code.

| Constant key | Units | Value and fixed bounds | Gradient | Meaning | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `relic.physical.constants.universal.gas_constant` | `J mol-1 K-1` | `8.314462618; bounds=(8.314462618, 8.314462618)` | `fixed` | Universal gas constant | Source: relic_physical_constants; CODATA 2018 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\bvoc_emissions.md`.

### References

- Guenther et al. (2012). MEGAN2.1. Geosci. Model Dev. 5, 1471-1492. (FunctorMetadata, certificate)
- Duhl et al. (2008). Sesquiterpene emissions from vegetation: a review. (FunctorMetadata, certificate)
- Niinemets et al. (2014). Bidirectional exchange of biogenic volatiles with vegetation. Plant, Cell & Environment 37, 1790-1809. (certificate)

(class-relic-scientific-biosphere-growth-biomass-allocation-biomassallocation)=
## BiomassAllocation

Biomass allocation functor with ParameterManager-driven defaults.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.growth.biomass_allocation.BiomassAllocation` |
| Source | `relic\scientific\biosphere\growth\biomass_allocation.py:69` |
| Runtime name | `biomass_allocation` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
BiomassAllocation(name: str = 'biomass_allocation', config: Dict[str, Any] | None = None, device: torch.device | None = None, pft_name: str | None = None, enable_environmental_control: bool = True, enable_phenology_control: bool = True, enable_allometry_control: bool = False, phenology_model_owner: str = 'gsi_phenology', bvoc_carbon_allocation_owner: str | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'biomass_allocation'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `pft_name` | <code>str &#124; None</code> | `None` | Optional plant functional type namespace used for PFT parameter overrides. |
| `enable_environmental_control` | `bool` | `True` | Enables or disables the environmental control pathway. |
| `enable_phenology_control` | `bool` | `True` | Enables or disables the phenology control pathway. |
| `enable_allometry_control` | `bool` | `False` | Enables or disables the allometry control pathway. |
| `phenology_model_owner` | `str` | `'gsi_phenology'` | Selects the canonical upstream owner used for dependency and single-writer composition. |
| `bvoc_carbon_allocation_owner` | <code>str &#124; None</code> | `None` | Selects the canonical upstream owner used for dependency and single-writer composition. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `net_primary_productivity` | flux | `kg C m-2 s-1` | required post-respiration carbon input from AutotrophicRespiration |
| `air_temperature` | forcing | `K` | conditional finite environmental driver in K |
| `water_stress_factor` | diagnostic | `dimensionless` | conditional bounded environmental control in the interval [0,1] |
| `leaf_area_index` | state | `m2 m-2` | conditional non-negative phenology state used to redistribute allocation |
| `phenology_phase` | diagnostic | `dimensionless` | conditional integer phase observation used to redistribute allocation |
| `leaf_biomass_c` | state | `kg C m-2` | conditional non-negative canonical plant carbon stock for allometric control and per-step ACCUMULATE output |
| `wood_biomass_c` | state | `kg C m-2` | conditional non-negative canonical plant carbon stock for allometric control and per-step ACCUMULATE output |
| `root_biomass_c` | state | `kg C m-2` | conditional non-negative canonical plant carbon stock for allometric control and per-step ACCUMULATE output |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `growth_rate_leaf` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | canonical leaf allocation flux and explicit carbon.total input boundary |
| `growth_rate_stem` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | canonical wood allocation flux and explicit carbon.total input boundary |
| `growth_rate_root` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | canonical root allocation flux and explicit carbon.total input boundary |
| `leaf_biomass_c` | state | `kg C m-2` | decided at run time by the returned write mode | conditional non-negative canonical plant carbon stock for allometric control and per-step ACCUMULATE output |
| `wood_biomass_c` | state | `kg C m-2` | decided at run time by the returned write mode | conditional non-negative canonical plant carbon stock for allometric control and per-step ACCUMULATE output |
| `root_biomass_c` | state | `kg C m-2` | decided at run time by the returned write mode | conditional non-negative canonical plant carbon stock for allometric control and per-step ACCUMULATE output |
| `diagnostics.biomass_allocation.total_growth_rate` | diagnostic | `kg C m-2 s-1` | diagnostics | exact sum of the three organ fluxes |
| `diagnostics.biomass_allocation.allocation_fraction_leaf` | output port | `1` | diagnostics | normalized leaf fraction observation |
| `diagnostics.biomass_allocation.allocation_fraction_stem` | output port | `1` | diagnostics | normalized stem fraction observation |
| `diagnostics.biomass_allocation.allocation_fraction_root` | output port | `1` | diagnostics | normalized root fraction observation |
| `diagnostics.biomass_allocation.allocation_sum` | output port | `1` | diagnostics | exact normalized fraction sum |
| `diagnostics.biomass_allocation.environmental_modifier` | output port | `1` | diagnostics | environmental-control observation only |
| `diagnostics.biomass_allocation.phenology_modifier` | output port | `1` | diagnostics | phenology-control observation only |
| `diagnostics.biomass_allocation.carbon_balance_residual` | output port | `kg C m-2 s-1` | diagnostics | NPP minus the three allocated organ fluxes |

### Governing equations and canonical operator

**Literature or process form**

$\frac{\mathrm{d}B_{i}}{\mathrm{d}t} = f_{i} \cdot ( \mathrm{NPP} - F_{\mathrm{BVOC},C} )$, with $\sum_{i} ( f_{i} ) = 1$, construction respiration accounted before NPP, and the BVOC debit enabled only by an explicit graph-selected owner

**Canonical RELIC structural form**

$\dot{x} = N_{h} r_{h}$ with $r_{h} = f_{i} \cdot ( \mathrm{NPP} - F_{\mathrm{BVOC},C} )$, ParameterManager-backed allocation coefficients, and declared stoichiometric/output map $N_{h}$

| Entry | Meaning |
| --- | --- |
| `N_h` | explicit three-channel identity wrapper for leaf, wood, and root allocation rates |
| `r_h` | vector $[ f_{\mathrm{leaf}} \cdot ( \mathrm{NPP} - F_{\mathrm{BVOC},C} ) , f_{\mathrm{stem}} \cdot ( \mathrm{NPP} - F_{\mathrm{BVOC},C} ) , f_{\mathrm{root}} \cdot ( \mathrm{NPP} - F_{\mathrm{BVOC},C} ) ]$ evaluated once by the production forward path |
| `boundary_route` | net_primary_productivity and the graph-selected bvoc_carbon_allocation_flux close the local partition; growth_rate_leaf, growth_rate_stem, and growth_rate_root are the only carbon.total input boundaries for the tracked structural stocks |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | empirical NPP boundary flux and allocation controls; no Hamiltonian effort is claimed at level 2 |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with a three-channel identity $N_{h}$ and $r_{h} = [ f_{\mathrm{leaf}} \cdot ( \mathrm{NPP} - F_{\mathrm{BVOC},C} ) , f_{\mathrm{stem}} \cdot ( \mathrm{NPP} - F_{\mathrm{BVOC},C} ) , f_{\mathrm{root}} \cdot ( \mathrm{NPP} - F_{\mathrm{BVOC},C} ) ]$ |
| `runtime_usage` | BiomassAllocation.forward resolves the graph-selected BVOC debit, normalizes organ fractions, evaluates all three rates once through advance_reaction_network, and emits the integrated state_delta as per-step ACCUMULATE updates. |
| `explicit_stoichiometric_usage` | The identity $N_{h}$ wrapper is explicit level-2 runtime structure and is not strict stoichiometric, Hamiltonian, or GENERIC evidence. |

### Composition channels and executable PortSpec API

Certificate composition channels: `net_primary_productivity`, `bvoc_carbon_allocation_flux`, `leaf_biomass_c`, `wood_biomass_c`, `root_biomass_c`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.growth.allocation.f_leaf_base` | `dimensionless` | `0.3; bounds=(0.05, 0.8)` | `learnable` | Base leaf allocation fraction | Source: literature_synthesis; Franklin et al. (2012); Poorter et al. (2012) |
| `biosphere.growth.allocation.f_stem_base` | `dimensionless` | `0.35; bounds=(0.05, 0.7)` | `learnable` | Base stem allocation fraction | Source: literature_synthesis; Franklin et al. (2012) |
| `biosphere.growth.allocation.f_root_base` | `dimensionless` | `0.35; bounds=(0.1, 0.8)` | `learnable` | Base root allocation fraction | Source: literature_synthesis; Franklin et al. (2012) |
| `biosphere.growth.allocation.root_alloc_temp_coeff` | `K-1` | `0.1; bounds=(0.05, 0.2)` | `fixed` | Temperature coefficient for root allocation | Source: pnet_model; Gao et al. (2025). pnetr package |
| `biosphere.growth.allocation.root_alloc_temp_ref` | `K` | `280.25; bounds=(273.15, 288.15)` | `fixed` | Reference temperature for root allocation (7.1 degC) | Source: pnet_model; Gao et al. (2025) |
| `biosphere.growth.allocation.root_alloc_temp_mult` | `dimensionless` | `0.68; bounds=(0.3, 1.0)` | `fixed` | Temperature multiplier for root allocation | Source: pnet_model; Gao et al. (2025) |
| `biosphere.growth.allocation.water_stress_threshold` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Water stress threshold for root allocation increase | Source: literature_synthesis; McDowell et al. (2008) |
| `biosphere.growth.allocation.temperature_stress_factor` | `K-1` | `0.02; bounds=(0.01, 0.05)` | `fixed` | Temperature stress response factor | Source: literature_synthesis; Berry & Bjorkman (1980) |
| `biosphere.growth.allocation.temperature_opt` | `K` | `298.15; bounds=(288.15, 308.15)` | `fixed` | Optimal temperature for allocation (25 degC) | Source: literature_synthesis; Berry & Bjorkman (1980) |
| `biosphere.growth.allocation.lai_max_threshold` | `m2 m-2` | `6.0; bounds=(3.0, 12.0)` | `fixed` | LAI threshold to suppress leaf allocation | Source: casa_model; Zhang et al. (2011). CASA-CNP |
| `biosphere.growth.allocation.lai_min_threshold` | `m2 m-2` | `1.0; bounds=(0.1, 3.0)` | `fixed` | LAI threshold to increase leaf allocation | Source: casa_model; Zhang et al. (2011) |
| `biosphere.growth.allocation.greenup_f_leaf` | `dimensionless` | `0.8; bounds=(0.5, 0.95)` | `fixed` | Leaf allocation during greenup phase | Source: casa_model; Zhang et al. (2011); Richardson et al. (2013) |
| `biosphere.growth.allocation.allometry_leaf_stem_exp` | `dimensionless` | `2.0; bounds=(1.5, 2.5)` | `fixed` | Leaf-stem allometric exponent (pipe model) | Source: literature_synthesis; Shinozaki et al. (1964) |
| `biosphere.growth.allocation.allometry_root_shoot_exp` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Root-shoot allometric exponent | Source: literature_synthesis; Enquist & Niklas (2002) |
| `biosphere.growth.allocation.allometric_blend_weight` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `fixed` | Blending weight for allometric constraint adjustment (0=no constraint, 1=full constraint) | Source: literature_synthesis; Franklin et al. (2012); Poorter et al. (2012) |
| `biosphere.growth.allocation.min_allocation_frac` | `dimensionless` | `0.05; bounds=(0.01, 0.1)` | `fixed` | Minimum allowed allocation fraction | Source: numerical_stability; Franklin et al. (2012) |
| `biosphere.growth.allocation.max_allocation_frac` | `dimensionless` | `0.9; bounds=(0.8, 0.95)` | `fixed` | Maximum allowed allocation fraction | Source: numerical_stability; Franklin et al. (2012) |
| `biosphere.growth.allocation.allocation_sum_tolerance` | `dimensionless` | `0.01; bounds=(0.001, 0.1)` | `fixed` | Tolerance for sum = 1.0 constraint | Source: numerical_stability; Numerical requirements |
| `biosphere.growth.allocation.normalization_epsilon` | `dimensionless` | `1e-08; bounds=(1e-12, 0.0001)` | `fixed` | Positive denominator guard used only during allocation normalization | Source: migrated_runtime_literal; Numerical requirements |
| `biosphere.growth.allocation.fraction_zero_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | Absolute tolerance for identifying zero allocation fractions | Source: migrated_runtime_literal; Numerical requirements |
| `biosphere.growth.allocation.woody_stem_fraction_threshold` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Stem allocation threshold used to distinguish woody allocation behavior | Source: migrated_runtime_literal; Franklin et al. (2012); Poorter et al. (2012) |
| `biosphere.growth.allocation.allometry_biomass_floor` | `kg C m-2` | `1e-06; bounds=(1e-12, 0.01)` | `fixed` | Positive biomass floor for stable allometric ratios | Source: migrated_runtime_literal; Numerical requirements |
| `biosphere.growth.allocation.allometry_ratio_max` | `dimensionless` | `10.0; bounds=(1.0, 100.0)` | `fixed` | Upper bound for dimensionless allometric biomass ratios | Source: migrated_runtime_literal; Numerical requirements |
| `biosphere.growth.allocation.emergency_stem_fraction` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `learnable` | Emergency allocation fraction to stem when LAI drops below minimum threshold for woody plants | Source: Biomass allocation model; Internal allocation strategy |
| `biosphere.growth.allocation.emergency_root_fraction` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `learnable` | Emergency allocation fraction to root when LAI drops below minimum threshold for woody plants | Source: Biomass allocation model; Internal allocation strategy |
| `biosphere.growth.allocation.defaults.default_environmental_modifier` | `dimensionless` | `1.0; bounds=(0.0, 2.0)` | `fixed` | Default environmental modifier (no environmental effect) | Source: system_defaults; System defaults |
| `biosphere.growth.allocation.defaults.default_phenological_modifier` | `dimensionless` | `1.0; bounds=(0.0, 2.0)` | `fixed` | Default phenological modifier (no phenology effect) | Source: system_defaults; System defaults |
| `biosphere.growth.allocation.defaults.temp_root_factor_min` | `dimensionless` | `0.5; bounds=(0.1, 1.0)` | `fixed` | Minimum temperature root factor | Source: system_defaults; System defaults |
| `biosphere.growth.allocation.defaults.temp_root_factor_max` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Maximum temperature root factor | Source: system_defaults; System defaults |
| `biosphere.growth.allocation.phenology.dormant_phase_code` | `dimensionless` | `0; bounds=(0.0, 3.0)` | `fixed` | Phenology phase code for dormant phase | Source: literature_synthesis; CASA model phenology documentation |
| `biosphere.growth.allocation.phenology.greenup_phase_code` | `dimensionless` | `1; bounds=(0.0, 3.0)` | `fixed` | Phenology phase code for greenup phase | Source: literature_synthesis; CASA model phenology documentation |
| `biosphere.growth.allocation.phenology.mature_phase_code` | `dimensionless` | `2; bounds=(0.0, 3.0)` | `fixed` | Phenology phase code for mature phase | Source: literature_synthesis; CASA model phenology documentation |
| `biosphere.growth.allocation.phenology.senescence_phase_code` | `dimensionless` | `3; bounds=(0.0, 3.0)` | `fixed` | Phenology phase code for senescence phase | Source: literature_synthesis; CASA model phenology documentation |
| `biosphere.growth.allocation.phenology.dormant_phase_modifier` | `dimensionless` | `0.2; bounds=(0.1, 0.5)` | `fixed` | Growth modifier during dormant phase | Source: literature_synthesis; Richardson et al. (2013) |
| `biosphere.growth.allocation.phenology.greenup_phase_modifier` | `dimensionless` | `1.5; bounds=(1.0, 2.0)` | `fixed` | Growth modifier during greenup phase | Source: literature_synthesis; Richardson et al. (2013) |
| `biosphere.growth.allocation.phenology.mature_phase_modifier` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Growth modifier during mature phase | Source: literature_synthesis; Richardson et al. (2013) |
| `biosphere.growth.allocation.phenology.senescence_phase_modifier` | `dimensionless` | `0.5; bounds=(0.3, 0.7)` | `fixed` | Growth modifier during senescence phase | Source: literature_synthesis; Richardson et al. (2013) |
| `biosphere.growth.allocation.phenology.dormant_leaf_allocation` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Leaf allocation during dormant phase | Source: literature_synthesis; Richardson et al. (2013) |
| `biosphere.growth.allocation.phenology.lai_max_leaf_allocation` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Leaf allocation when LAI exceeds maximum | Source: literature_synthesis; Reich et al. (1997) |
| `biosphere.growth.allocation.phenology.herbaceous_stem_allocation` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Stem allocation for herbaceous plants during greenup | Source: literature_synthesis; Kikuzawa (1999) |
| `biosphere.growth.allocation.phenology.herbaceous_lai_min_stem_allocation` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Stem allocation for herbaceous when LAI < minimum | Source: literature_synthesis; Kikuzawa (1999) |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |
| `biosphere.pft.broadleaf_deciduous_boreal.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.broadleaf_deciduous_temperate.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.broadleaf_deciduous_tropical.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.broadleaf_evergreen_temperate.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.broadleaf_evergreen_tropical.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.c3_grass_arctic.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.c3_grass_temperate.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.c4_grass_temperate.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.crop_temperate.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.crop_tropical.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.needleleaf_deciduous_boreal.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.needleleaf_evergreen_boreal.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.needleleaf_evergreen_temperate.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.shrub_deciduous_temperate.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.shrub_evergreen_temperate.allocation.f_leaf` | `dimensionless` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Fraction of carbon allocated to leaf tissue | Source: literature; doi:10.1175/2010JCLI3879.1 |
| `biosphere.pft.broadleaf_deciduous_boreal.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.broadleaf_deciduous_temperate.allocation.f_stem` | `dimensionless` | `0.5; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.broadleaf_deciduous_tropical.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.broadleaf_evergreen_temperate.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.broadleaf_evergreen_tropical.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.c3_grass_arctic.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.c3_grass_temperate.allocation.f_stem` | `dimensionless` | `0.4; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.c4_grass_temperate.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.crop_temperate.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.crop_tropical.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.needleleaf_deciduous_boreal.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.needleleaf_evergreen_boreal.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.needleleaf_evergreen_temperate.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.shrub_deciduous_temperate.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.shrub_evergreen_temperate.allocation.f_stem` | `dimensionless` | `0.45; bounds=(0.35, 0.55)` | `fixed` | Fraction of carbon allocated to stem tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.broadleaf_deciduous_boreal.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.broadleaf_deciduous_temperate.allocation.f_root` | `dimensionless` | `0.25; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.broadleaf_deciduous_tropical.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.broadleaf_evergreen_temperate.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.broadleaf_evergreen_tropical.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.c3_grass_arctic.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.c3_grass_temperate.allocation.f_root` | `dimensionless` | `0.35; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.c4_grass_temperate.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.crop_temperate.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.crop_tropical.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.needleleaf_deciduous_boreal.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.needleleaf_evergreen_boreal.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.needleleaf_evergreen_temperate.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.shrub_deciduous_temperate.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
| `biosphere.pft.shrub_evergreen_temperate.allocation.f_root` | `dimensionless` | `0.3; bounds=(0.2, 0.45)` | `fixed` | Fraction of carbon allocated to root tissue | Source: literature; Litton et al. (2007) Forests 4:913 |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | organ rates and diagnostics use UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} )$; three biomass stocks use UpdateItem(ACCUMULATE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{True} )$ with already integrated kg C m-2 values |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\growth.md`.

### References

- Gao et al. (2025). pnetr: An R package for the PnET model family (FunctorMetadata, certificate)
- Zhang et al. (2011). CASA-CNP allocation routines (FunctorMetadata, certificate)
- Sands & Landsberg (2002). 3-PG allocation strategies (FunctorMetadata, certificate)
- Franklin et al. (2012). Modeling carbon allocation in trees (FunctorMetadata, certificate)
- Kesselmeier et al. (2002). VOC emissions and the terrestrial carbon budget (FunctorMetadata, certificate)
- Niinemets et al. (2014). Synthesis and release can be uncoupled for stored terpenoids (certificate)

(class-relic-scientific-biosphere-growth-phenology-growth-phenologygrowth)=
## PhenologyGrowth

Observe growth activity from one canonical phenology control port.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.growth.phenology_growth.PhenologyGrowth` |
| Source | `relic\scientific\biosphere\growth\phenology_growth.py:42` |
| Runtime name | `phenology_growth` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `observation` |
| Spatial scale | `patch` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
PhenologyGrowth(name: str = 'phenology_growth', config: Dict[str, Any] | None = None, device: torch.device | None = None, metadata: relic.core.functor.functor_types.FunctorMetadata | None = None, phenology_model_owner: str | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'phenology_growth'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `metadata` | <code>relic.core.functor.functor_types.FunctorMetadata &#124; None</code> | `None` | Optional explicit FunctorMetadata object for advanced composition. |
| `phenology_model_owner` | <code>str &#124; None</code> | `None` | Selects the canonical upstream owner used for dependency and single-writer composition. |

The runtime metadata below is introspected from the documented configuration `phenology_model_owner='gsi_phenology'`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `phenological_factor` | diagnostic | `dimensionless` | required dimensionless control from one selected main phenology owner |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `diagnostics.phenology.growth_active` | diagnostic | `dimensionless` | diagnostics | dimensionless Boolean-valued activity diagnostic |

### Governing equations and canonical operator

**Literature or process form**

Phenology controls ecosystem carbon processes; the exact fixed binary threshold is a calibrated model configuration rather than a universal literature law.

**Canonical RELIC structural form**

$\mathtt{diagnostics.phenology.growth\_active} = I [ \mathrm{phenological}_{\mathrm{factor}} > \mathrm{growth}_{\mathrm{active},\mathrm{threshold}} ]$, with no state advancement

| Entry | Meaning |
| --- | --- |
| `O_h` | declared empirical observation map; no state, Hamiltonian, J, or M operator is asserted |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared selected-owner control and diagnostic basis |
| `effort_variable` | not applicable to this non-strict empirical observation |
| `state_advancer` | none; $\mathtt{diagnostics.phenology.growth\_active} = O_{h} ( \mathrm{phenological}_{\mathrm{factor}} )$ is observation-only |
| `runtime_usage` | PhenologyGrowth.forward evaluates $O_{h}$ once and emits one explicit REPLACE UpdateItem without advancing state. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phenological_factor`, `diagnostics.phenology.growth_active`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.growth.allocation.growth_active_threshold` | `dimensionless` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Calibratable threshold for the selected-owner growth activity diagnostic | Source: calibrated_phenomenological_diagnostic; Richardson et al. (2013), phenological control concept; exact threshold is calibrated |
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
| `state_write_mode` | one dimensionless REPLACE UpdateItem with $\mathrm{per}_{\mathrm{timestep}} = \mathrm{false}$; no conserved state write |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\growth.md`.

### References

- Richardson et al. (2013). Climate change, phenology, and phenological control. Agric For Meteorol 169:156-173 (FunctorMetadata, certificate)
- Gao et al. (2025). pnetr: an R package for the PnET family of forest ecosystem models. (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-growth-structural-growth-structuralgrowth)=
## StructuralGrowth

Observe leaf-carbon-driven LAI tendencies without writing LAI state.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.growth.structural_growth.StructuralGrowth` |
| Source | `relic\scientific\biosphere\growth\structural_growth.py:35` |
| Runtime name | `structural_growth` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `patch` |
| Variable kind | `diagnostic` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
StructuralGrowth(name: str = 'structural_growth', config: Dict[str, Any] | None = None, device: torch.device | None = None, **kwargs: Any)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'structural_growth'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `kwargs` | `typing.Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `growth_rate_leaf` | flux | `kg C m-2 s-1` | BiomassAllocation-owned carbon flux observed in kg C m-2 s-1 |
| `leaf_area_index` | state | `m2 m-2` | phenology-owned absolute LAI observed in m2 m-2 |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `diagnostics.structural_growth.gross_leaf_area_growth_tendency` | output port | `m2 m-2 s-1` | diagnostics | SLA times canonical leaf-carbon growth flux |
| `diagnostics.structural_growth.leaf_area_senescence_tendency` | output port | `m2 m-2 s-1` | diagnostics | turnover rate times absolute LAI |
| `diagnostics.structural_growth.net_leaf_area_tendency` | output port | `m2 m-2 s-1` | diagnostics | signed gross-minus-senescence observation |
| `diagnostics.structural_growth.step_leaf_area_delta` | output port | `m2 m-2` | diagnostics | net tendency integrated exactly once over dt |

### Governing equations and canonical operator

**Literature or process form**

$$
r_{\mathrm{growth}} = \mathrm{SLA} \cdot \mathrm{growth}_{\mathrm{rate},\mathrm{leaf}}; r_{\mathrm{loss}} = k_{\mathrm{leaf}} \cdot \mathrm{leaf}_{\mathrm{area},\mathrm{index}}; r_{\mathrm{net}} = r_{\mathrm{growth}} - r_{\mathrm{loss}}
$$

**Canonical RELIC structural form**

$$
\dot{y} = N_{h} r_{h}; \delta_{y} = \dot{y} \cdot \mathrm{d}t
$$

| Entry | Meaning |
| --- | --- |
| `N_h` | signed row $[ 1 , - 1 ]$ mapping gross leaf-area growth and senescence to the net observation |
| `r_h` | explicit $[ \mathrm{SLA} \cdot \mathrm{growth}_{\mathrm{rate},\mathrm{leaf}} , k_{\mathrm{leaf}} \cdot \mathrm{leaf}_{\mathrm{area},\mathrm{index}} ]$ tendency channels |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | one diagnostic leaf-area observation row |
| `effort_variable` | required growth_rate_leaf and leaf_area_index inputs with ParameterManager-backed SLA and turnover coefficients |
| `state_advancer` | observation $\dot{y} = N_{h} r_{h}$; $\delta_{y} = \dot{y} \cdot \mathrm{d}t$ |
| `runtime_usage` | StructuralGrowth.forward reads the net tendency and step delta returned by advance_reaction_network without advancing leaf_area_index. |
| `explicit_stoichiometric_usage` | `species=(diagnostics.structural_growth.net_leaf_area_tendency)`; `reactions=(leaf_area_growth, leaf_area_senescence)`; $\mathrm{stoichiometric}_{N} = [ [ 1 , - 1 ] ] .$ |

### Composition channels and executable PortSpec API

Certificate composition channels: `growth_rate_leaf`, `leaf_area_index`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.phenology.morphology.sla` | `m2 kg-1 C` | `20.0; bounds=(5.0, 50.0)` | `fixed` | Specific leaf area (alias for specific_leaf_area) | Source: literature_synthesis; Wright et al. (2004) Nature 428, 821-827 |
| `biosphere.phenology.turnover.leaf_turnover_rate` | `s-1` | `7.927e-09; bounds=(3.171e-09, 6.342e-08)` | `learnable` | Base annual leaf turnover rate for deciduous trees | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | four observation-only UpdateItem values use REPLACE; StructuralGrowth never writes leaf_area_index |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\growth.md`.

### References

- Wright et al. (2004). The worldwide leaf economics spectrum. Nature 428:821-827. (FunctorMetadata, certificate)
- Yin et al. (2021). Combining physiological principles and mathematics to improve crop models. (FunctorMetadata)
- Feinberg (2019). Foundations of Chemical Reaction Network Theory. (FunctorMetadata, certificate)
- Yin et al. (2021). On the needs for combining physiological principles and mathematics to improve crop models. (certificate)

(class-relic-scientific-biosphere-phenology-models-cropphenology)=
## CropPhenology

Crop-Specific Phenology Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.phenology.models.CropPhenology` |
| Source | `relic\scientific\biosphere\phenology\models.py:1432` |
| Runtime name | `crop_phenology` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
CropPhenology(name: str = 'crop_phenology', config: Dict[str, Any] | None = None, phenology_parameters: relic.scientific.biosphere.phenology.parameters.PhenologyParameters | None = None, **kwargs) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'crop_phenology'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `phenology_parameters` | <code>relic.scientific.biosphere.phenology.parameters.PhenologyParameters &#124; None</code> | `None` | Constructor control for phenology parameters; see the class runtime contract and ParameterManager table below. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `day_of_year` | forcing | `1` | required calendar forcing port [day] |
| `air_temperature` | forcing | `K` | required temperature forcing port [K] |
| `phenology_crop_accumulated_phu` | state | `degC day` | StateManager-owned crop thermal-time state [degC day] |
| `phenology_crop_is_sown` | state | `dimensionless` | StateManager-owned crop sowing-cycle flag [dimensionless] |
| `phenology_crop_is_harvested` | state | `dimensionless` | StateManager-owned crop harvest-cycle flag [dimensionless] |
| `phenology_crop_vernalization_days` | state | `day` | StateManager-owned crop vernalization state [day] |
| `leaf_carbon` | state | `kg C m-2` | prognostic vegetation carbon pool [kg C m-2] |
| `froot_carbon` | state | `kg C m-2` | prognostic vegetation carbon pool [kg C m-2] |
| `leaf_nitrogen` | state | `kg N m-2` | prognostic vegetation nitrogen pool [kg N m-2] |
| `froot_nitrogen` | state | `kg N m-2` | prognostic vegetation nitrogen pool [kg N m-2] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `leaf_area_index` | state | `m2 m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenology_phase` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.accumulated_phu` | output port | `degC day` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `harvest_index` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenological_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenology_crop_accumulated_phu` | state | `degC day` | decided at run time by the returned write mode | StateManager-owned crop thermal-time state [degC day] |
| `phenology_crop_is_sown` | state | `dimensionless` | decided at run time by the returned write mode | StateManager-owned crop sowing-cycle flag [dimensionless] |
| `phenology_crop_is_harvested` | state | `dimensionless` | decided at run time by the returned write mode | StateManager-owned crop harvest-cycle flag [dimensionless] |
| `phenology_crop_vernalization_days` | state | `day` | decided at run time by the returned write mode | StateManager-owned crop vernalization state [day] |
| `phenology_senescence_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_carbon` | state | `kg C m-2` | decided at run time by the returned write mode | prognostic vegetation carbon pool [kg C m-2] |
| `froot_carbon` | state | `kg C m-2` | decided at run time by the returned write mode | prognostic vegetation carbon pool [kg C m-2] |
| `leaf_nitrogen` | state | `kg N m-2` | decided at run time by the returned write mode | prognostic vegetation nitrogen pool [kg N m-2] |
| `froot_nitrogen` | state | `kg N m-2` | decided at run time by the returned write mode | prognostic vegetation nitrogen pool [kg N m-2] |
| `plant_nitrogen_pool` | state | `kg N m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_litterfall_carbon` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `froot_litterfall_carbon` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_litterfall_nitrogen` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `froot_litterfall_nitrogen` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `nitrogen_retranslocation` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.carbon_turnover_balance_residual` | output port | `kg C m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.nitrogen_turnover_balance_residual` | output port | `kg N m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

Crop phenology equations:

Daily thermal time:

$$
PHU_{daily} = \max(0, T_{avg} - T_{base})
$$

Cumulative PHU:

$$
PHU_{cum} = \sum_{i=sowing}^{t} PHU_{daily,i}
$$

Development index:

$$
DI = \frac{PHU_{cum}}{PHU_{total}}
$$

Harvest index dynamics:

$$
HI(t) = HI_{max} \times \frac{DI - DI_{HI-start}}{1 - DI_{HI-start}}
$$

$$
\text{ for } DI \geq DI_{HI-start}
$$

Biomass partitioning:

$$
f_{grain} = HI(t) \text{ during grain filling}
$$

$$
f_{leaf} = (1 - HI(t)) \times allocation_{leaf}(DI)
$$

$$
f_{stem} = (1 - HI(t)) \times allocation_{stem}(DI)
$$

$$
f_{root} = (1 - HI(t)) \times allocation_{root}(DI)
$$

Development stages:

- Germination: $0 < \mathrm{DI} < 0.15$

- Vegetative: $0.15 < \mathrm{DI} < 0.4$

- Reproductive: $0.4 < \mathrm{DI} < 0.65$

- Grain filling: $0.65 < \mathrm{DI} < 1.0$

- $Maturity: DI \geq 1.0$

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | C and N turnover uses $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network; crop temporal full states use the canonical StateManager replacement path. |
| `runtime_usage` | CropPhenology inherits PhenologyBase.forward, which calls advance_reaction_network for the C and N turnover ledger and emits explicit REPLACE updates for the complete crop temporal state. |
| `explicit_stoichiometric_usage` | PhenologyBase.forward materializes stoichiometric_N and rates for the C and N turnover outputs and lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network; PHU and crop-cycle memory are not misrepresented as reaction states. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.phenology.turnover.n_retrans_efficiency` | `dimensionless` | `0.6; bounds=(0.3, 0.9)` | `learnable` | Leaf nitrogen retranslocation efficiency during senescence | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.constants.max_lai_threshold` | `m2 m-2` | `15.0; bounds=(5.0, 20.0)` | `fixed` | Maximum reasonable leaf area index threshold for validation | Source: model_validation; Bonan et al. (2011). J. Climate, 24(17), 4603-4623. |
| `biosphere.phenology.gdd.base_temp` | `K` | `278.15; bounds=(273.15, 283.15)` | `fixed` | Base temperature for GDD calculation (5degC) | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300 |
| `biosphere.phenology.crop.default_type` | `dimensionless` | `'wheat'; bounds=None` | `fixed` | Default crop type used when crop phenology parameters are requested without explicit specification | Source: agronomic_guidelines; USDA (2010). National Crop Phenology Guidelines. |
| `biosphere.phenology.crop.phu_total` | `degC day` | `1200.0; bounds=(800.0, 2000.0)` | `learnable` | Total potential heat units for crop maturity (generic default) | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.hi_max` | `dimensionless` | `0.5; bounds=(0.3, 0.7)` | `learnable` | Maximum harvest index for crops | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.hi_start_fraction` | `dimensionless` | `0.4; bounds=(0.2, 0.6)` | `learnable` | Development index fraction when harvest index accumulation starts | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.sowing_doy` | `day` | `120; bounds=(1.0, 180.0)` | `fixed` | Default sowing day of year for crops | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.harvest_doy` | `day` | `280; bounds=(200.0, 365.0)` | `fixed` | Default harvest day of year for crops | Source: literature_synthesis; Luyssaert et al. (2007) Glob. Change Biol. 13, 2509-2537; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.vernalization_required` | `dimensionless` | `False; bounds=None` | `fixed` | Whether vernalization is required for crop development | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.vernalization_temp_min` | `K` | `273.15; bounds=(263.15, 283.15)` | `fixed` | Minimum temperature for vernalization (0degC) | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.vernalization_temp_max` | `K` | `283.15; bounds=(273.15, 293.15)` | `fixed` | Maximum temperature for vernalization (10degC) | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.vernalization_days_required` | `day` | `30; bounds=(10.0, 60.0)` | `fixed` | Number of days required for vernalization | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.turnover.leaf_turnover_rate` | `s-1` | `7.927e-09; bounds=(3.171e-09, 6.342e-08)` | `learnable` | Base annual leaf turnover rate for deciduous trees | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.turnover.froot_turnover_rate` | `s-1` | `3.171e-08; bounds=(1.585e-08, 9.512e-08)` | `learnable` | Base annual fine root turnover rate | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.morphology.max_lai` | `dimensionless` | `6.0; bounds=(2.0, 10.0)` | `learnable` | Maximum leaf area index for vegetation | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `biosphere.phenology.morphology.min_lai` | `dimensionless` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Minimum leaf area index (dormant season) | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `biosphere.validation.tolerance.phenology` | `dimensionless` | `1e-06; bounds=(1e-12, 0.0001)` | `fixed` | Tolerance for phenology model numerical validation checks | Source: literature_synthesis; Guenther, A. B. et al. (2012) Geosci. Model Dev., 5, 1471-1492 |
| `biosphere.phenology.activity.germination_activity` | `dimensionless` | `0.3; bounds=(0.1, 0.5)` | `fixed` | Activity level during germination phase | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.activity.vegetative_activity` | `dimensionless` | `1.0; bounds=(0.8, 1.0)` | `fixed` | Maximum growth activity during vegetative phase | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.activity.maturity_senescence` | `dimensionless` | `1.0; bounds=(0.8, 1.0)` | `fixed` | Maximum senescence rate at maturity | Source: literature_synthesis; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948 |
| `biosphere.phenology.crop.di_germination_end` | `dimensionless` | `0.15; bounds=(0.1, 0.2)` | `learnable` | Development index threshold for end of germination stage | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.di_vegetative_peak` | `dimensionless` | `0.25; bounds=(0.2, 0.3)` | `learnable` | Development index at peak vegetative growth | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.di_vegetative_end` | `dimensionless` | `0.4; bounds=(0.3, 0.5)` | `learnable` | Development index threshold for end of vegetative stage | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.di_lai_peak_end` | `dimensionless` | `0.3; bounds=(0.25, 0.35)` | `learnable` | Development index at end of peak LAI maintenance | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.di_reproductive_end` | `dimensionless` | `0.65; bounds=(0.6, 0.7)` | `learnable` | Development index threshold for end of reproductive stage | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.initial_lai` | `m2 m-2` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Initial leaf area index at crop emergence | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.lai_growth_exponent` | `dimensionless` | `0.7; bounds=(0.5, 1.0)` | `learnable` | Exponent for LAI growth factor in grain filling | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.grain_filling_factor` | `dimensionless` | `2.0; bounds=(1.5, 2.5)` | `learnable` | Senescence factor multiplier during grain filling | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.senescence_growth_factor` | `dimensionless` | `1.5; bounds=(1.0, 2.0)` | `learnable` | Growth factor multiplier during early senescence | Source: literature_synthesis; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.senescence_min_factor` | `dimensionless` | `0.2; bounds=(0.1, 0.3)` | `fixed` | Minimum growth factor during senescence | Source: literature_synthesis; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.vernalization_increment` | `day` | `1.0; bounds=(0.5, 1.5)` | `fixed` | Daily increment for vernalization days accumulation | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.crop.vernalization_min_factor` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `learnable` | Configurable empirical lower bound for the crop vernalization multiplier | Source: model_calibration_default; Jones, J.W. et al. (2003). DSSAT cropping system model; coefficient requires crop-specific calibration |
| `biosphere.phenology.crop.year_reset_threshold` | `day` | `50; bounds=(1.0, 100.0)` | `fixed` | Day of year threshold for resetting crop cycle | Source: literature_synthesis; Bondeau et al. (2007) Glob. Change Biol. 13, 679-706 |
| `biosphere.phenology.computational.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Minimum value for tensor clamping operations | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.computational.clamp_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Maximum value for tensor clamping operations | Source: RELIC parameter database; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
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
| `state_write_mode` | integrated C and N pool increments use ACCUMULATE with per_timestep true; crop temporal full states and observations use REPLACE with declared units |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\phenology.md`.

### References

- Jones, J.W., et al. (2003). The DSSAT cropping system model. European Journal of Agronomy, 18(3-4), 235-265. (FunctorMetadata, certificate)
- Keating, B.A., et al. (2003). An overview of APSIM, a model designed for farming systems simulation. European Journal of Agronomy, 18(3-4), 267-288. (FunctorMetadata, certificate)
- Ritchie, J.T. (1991). Wheat phasic development. Agricultural Systems, 36(3), 305-320. (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-phenology-models-gddphenology)=
## GDDPhenology

Growing Degree Days (GDD) Phenology Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.phenology.models.GDDPhenology` |
| Source | `relic\scientific\biosphere\phenology\models.py:441` |
| Runtime name | `gdd_phenology` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
GDDPhenology(name: str = 'gdd_phenology', config: Dict[str, Any] | None = None, phenology_parameters: relic.scientific.biosphere.phenology.parameters.PhenologyParameters | None = None, **kwargs) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'gdd_phenology'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `phenology_parameters` | <code>relic.scientific.biosphere.phenology.parameters.PhenologyParameters &#124; None</code> | `None` | Constructor control for phenology parameters; see the class runtime contract and ParameterManager table below. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `day_of_year` | forcing | `1` | required calendar forcing port [day] |
| `air_temperature` | forcing | `K` | required mean temperature forcing port [K] |
| `air_temperature_max` | forcing | `K` | required maximum temperature forcing port [K] |
| `air_temperature_min` | forcing | `K` | required minimum temperature forcing port [K] |
| `phenology_gdd_accumulated` | state | `degC day` | StateManager-owned cumulative thermal-time state [degC day] |
| `phenology_gdd_previous_day_of_year` | state | `day` | StateManager-owned prior calendar-day state [day] |
| `leaf_carbon` | state | `kg C m-2` | prognostic vegetation carbon pool [kg C m-2] |
| `froot_carbon` | state | `kg C m-2` | prognostic vegetation carbon pool [kg C m-2] |
| `leaf_nitrogen` | state | `kg N m-2` | prognostic vegetation nitrogen pool [kg N m-2] |
| `froot_nitrogen` | state | `kg N m-2` | prognostic vegetation nitrogen pool [kg N m-2] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `leaf_area_index` | state | `m2 m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenological_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenology_senescence_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenology_gdd_accumulated` | state | `degC day` | decided at run time by the returned write mode | StateManager-owned cumulative thermal-time state [degC day] |
| `phenology_gdd_previous_day_of_year` | state | `day` | decided at run time by the returned write mode | StateManager-owned prior calendar-day state [day] |
| `diagnostics.phenology.accumulated_gdd` | output port | `degC day` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.onset_event_flag` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.onset_event_time_offset` | output port | `s` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.onset_event_time_sensitivity_gdd_start` | output port | `s degC-1 day-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.lai_sensitivity_jump_gdd_start` | output port | `m2 m-2 degC-1 day-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `phenology_phase` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_carbon` | state | `kg C m-2` | decided at run time by the returned write mode | prognostic vegetation carbon pool [kg C m-2] |
| `froot_carbon` | state | `kg C m-2` | decided at run time by the returned write mode | prognostic vegetation carbon pool [kg C m-2] |
| `leaf_nitrogen` | state | `kg N m-2` | decided at run time by the returned write mode | prognostic vegetation nitrogen pool [kg N m-2] |
| `froot_nitrogen` | state | `kg N m-2` | decided at run time by the returned write mode | prognostic vegetation nitrogen pool [kg N m-2] |
| `plant_nitrogen_pool` | state | `kg N m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_litterfall_carbon` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `froot_litterfall_carbon` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_litterfall_nitrogen` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `froot_litterfall_nitrogen` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `nitrogen_retranslocation` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.carbon_turnover_balance_residual` | output port | `kg C m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.nitrogen_turnover_balance_residual` | output port | `kg N m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

GDD phenology equations:

Daily GDD:

$$
GDD_{daily} = \max(0, \frac{T_{max} + T_{min}}{2} - T_{base})
$$

Cumulative GDD:

$$
GDD_{cum} = \sum_{i=1}^{t} GDD_{daily,i}
$$

Growth factor:

$$
f_{growth}(GDD_{cum}) =
\begin{cases}
0 & \text{if } GDD_{cum} < GDD_{start} \\
\frac{GDD_{cum} - GDD_{start}}{GDD_{full} - GDD_{start}} & \text{if } GDD_{start} \leq GDD_{cum} < GDD_{full} \\
1 & \text{if } GDD_{full} \leq GDD_{cum} < GDD_{end} \\
1 - \frac{GDD_{cum} - GDD_{end}}{GDD_{senes} - GDD_{end}} & \text{if } GDD_{end} \leq GDD_{cum} < GDD_{senes} \\
0 & \text{if } GDD_{cum} \geq GDD_{senes}
\end{cases}
$$

$$
LAI(t) = LAI_{max} \times f_{growth}(GDD_{cum}(t))
$$

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | GDDPhenology inherits PhenologyBase.forward, which calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical phenology outputs. |
| `explicit_stoichiometric_usage` | PhenologyBase.forward materializes stoichiometric_N and rates for GDDPhenology outputs, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.phenology.turnover.n_retrans_efficiency` | `dimensionless` | `0.6; bounds=(0.3, 0.9)` | `learnable` | Leaf nitrogen retranslocation efficiency during senescence | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.constants.max_lai_threshold` | `m2 m-2` | `15.0; bounds=(5.0, 20.0)` | `fixed` | Maximum reasonable leaf area index threshold for validation | Source: model_validation; Bonan et al. (2011). J. Climate, 24(17), 4603-4623. |
| `biosphere.phenology.gdd.base_temp` | `K` | `278.15; bounds=(273.15, 283.15)` | `fixed` | Base temperature for GDD calculation (5degC) | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300 |
| `biosphere.phenology.gdd.threshold_start` | `degC_day` | `200.0; bounds=(50.0, 400.0)` | `learnable` | GDD threshold for phenological onset (budburst) | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300 |
| `biosphere.phenology.gdd.threshold_end` | `degC_day` | `1200.0; bounds=(800.0, 1600.0)` | `learnable` | GDD threshold for phenological end (senescence) | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300 |
| `biosphere.phenology.gdd.max_temp_limit` | `degC` | `35.0; bounds=(25.0, 45.0)` | `fixed` | Maximum temperature limit for GDD calculation | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300 |
| `biosphere.phenology.turnover.leaf_turnover_rate` | `s-1` | `7.927e-09; bounds=(3.171e-09, 6.342e-08)` | `learnable` | Base annual leaf turnover rate for deciduous trees | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.turnover.froot_turnover_rate` | `s-1` | `3.171e-08; bounds=(1.585e-08, 9.512e-08)` | `learnable` | Base annual fine root turnover rate | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.morphology.max_lai` | `dimensionless` | `6.0; bounds=(2.0, 10.0)` | `learnable` | Maximum leaf area index for vegetation | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `biosphere.phenology.morphology.min_lai` | `dimensionless` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Minimum leaf area index (dormant season) | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `biosphere.validation.tolerance.phenology` | `dimensionless` | `1e-06; bounds=(1e-12, 0.0001)` | `fixed` | Tolerance for phenology model numerical validation checks | Source: literature_synthesis; Guenther, A. B. et al. (2012) Geosci. Model Dev., 5, 1471-1492 |
| `biosphere.phenology.gdd.expansion_phase_gdd` | `degC_day` | `100.0; bounds=(50.0, 200.0)` | `learnable` | GDD duration for leaf expansion phase | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300 |
| `biosphere.phenology.gdd.full_growth_offset` | `degC_day` | `300.0; bounds=(200.0, 500.0)` | `learnable` | GDD offset from start to reach full growth phase | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300 |
| `biosphere.phenology.gdd.reset_day_threshold` | `day` | `10; bounds=(1.0, 30.0)` | `fixed` | Day of year threshold for resetting GDD accumulation | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300 |
| `biosphere.phenology.gdd.senescence_buffer_gdd` | `degC_day` | `200.0; bounds=(100.0, 400.0)` | `learnable` | GDD buffer before senescence initiation | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948 |
| `biosphere.phenology.gdd.senescence_start_offset` | `degC_day` | `200.0; bounds=(100.0, 400.0)` | `learnable` | GDD offset before end threshold to start senescence | Source: literature_synthesis; McMaster & Wilhelm (1997) Agric. For. Meteorol. 87, 291-300; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948 |
| `biosphere.phenology.computational.unity_factor` | `dimensionless` | `1.0; bounds=(1.0, 1.0)` | `fixed` | Unity factor for computational variables | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.computational.zero_initial` | `dimensionless` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Zero initialization value for computational variables | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.computational.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Minimum value for tensor clamping operations | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.computational.clamp_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Maximum value for tensor clamping operations | Source: RELIC parameter database; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.turnover.background_senescence_rate` | `dimensionless` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Background senescence rate during non-senescence periods | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948 |
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
| `state_write_mode` | integrated C and N pool increments use ACCUMULATE with per_timestep true; GDD temporal full states, rates, and observations use REPLACE with declared units |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\phenology.md`.

### References

- McMaster, G.S. & Wilhelm, W.W. (1997). Growing degree-days: one equation, two interpretations. Agricultural and Forest Meteorology, 87(4), 291-300. (FunctorMetadata, certificate)
- Miller, P., et al. (2001). Using growing degree-days to predict plant stages. Montana State University Extension Service. (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-phenology-models-gsiphenology)=
## GSIPhenology

Growing Season Index (GSI) Phenology Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.phenology.models.GSIPhenology` |
| Source | `relic\scientific\biosphere\phenology\models.py:957` |
| Runtime name | `gsi_phenology` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
GSIPhenology(name: str = 'gsi_phenology', config: Dict[str, Any] | None = None, phenology_parameters: relic.scientific.biosphere.phenology.parameters.PhenologyParameters | None = None, **kwargs) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'gsi_phenology'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `phenology_parameters` | <code>relic.scientific.biosphere.phenology.parameters.PhenologyParameters &#124; None</code> | `None` | Constructor control for phenology parameters; see the class runtime contract and ParameterManager table below. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `day_of_year` | forcing | `1` | required calendar forcing port [day] |
| `air_temperature` | forcing | `K` | required mean temperature forcing port [K] |
| `air_temperature_min` | forcing | `K` | required minimum temperature forcing port [K] |
| `vapor_pressure_deficit` | forcing | `Pa` | required atmospheric dryness forcing port [Pa] |
| `daylength` | forcing | `s` | required photoperiod forcing port [s] |
| `phenology_gsi_history` | state | `dimensionless` | StateManager-owned 21-day instantaneous-GSI ring buffer [dimensionless] |
| `phenology_gsi_history_index` | state | `dimensionless` | next ring-buffer write position with integer semantics [dimensionless] |
| `phenology_gsi_history_count` | state | `dimensionless` | number of valid ring-buffer samples with integer semantics [dimensionless] |
| `phenology_gsi_in_growing_season` | state | `dimensionless` | StateManager-owned hysteresis flag constrained to zero or one [dimensionless] |
| `leaf_carbon` | state | `kg C m-2` | prognostic vegetation carbon pool [kg C m-2] |
| `froot_carbon` | state | `kg C m-2` | prognostic vegetation carbon pool [kg C m-2] |
| `leaf_nitrogen` | state | `kg N m-2` | prognostic vegetation nitrogen pool [kg N m-2] |
| `froot_nitrogen` | state | `kg N m-2` | prognostic vegetation nitrogen pool [kg N m-2] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `leaf_area_index` | state | `m2 m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenological_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.growing_season_index` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `phenology_gsi_history` | state | `dimensionless` | decided at run time by the returned write mode | StateManager-owned 21-day instantaneous-GSI ring buffer [dimensionless] |
| `phenology_gsi_history_index` | state | `dimensionless` | decided at run time by the returned write mode | next ring-buffer write position with integer semantics [dimensionless] |
| `phenology_gsi_history_count` | state | `dimensionless` | decided at run time by the returned write mode | number of valid ring-buffer samples with integer semantics [dimensionless] |
| `phenology_gsi_in_growing_season` | state | `dimensionless` | decided at run time by the returned write mode | StateManager-owned hysteresis flag constrained to zero or one [dimensionless] |
| `phenology_senescence_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenology_phase` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_carbon` | state | `kg C m-2` | decided at run time by the returned write mode | prognostic vegetation carbon pool [kg C m-2] |
| `froot_carbon` | state | `kg C m-2` | decided at run time by the returned write mode | prognostic vegetation carbon pool [kg C m-2] |
| `leaf_nitrogen` | state | `kg N m-2` | decided at run time by the returned write mode | prognostic vegetation nitrogen pool [kg N m-2] |
| `froot_nitrogen` | state | `kg N m-2` | decided at run time by the returned write mode | prognostic vegetation nitrogen pool [kg N m-2] |
| `plant_nitrogen_pool` | state | `kg N m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_litterfall_carbon` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `froot_litterfall_carbon` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_litterfall_nitrogen` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `froot_litterfall_nitrogen` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `nitrogen_retranslocation` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.carbon_turnover_balance_residual` | output port | `kg C m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.nitrogen_turnover_balance_residual` | output port | `kg N m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

$$
\mathrm{iGSI} = f_{T} ( T_{\mathrm{min}} ) f_{\mathrm{VPD}} ( \mathrm{VPD} ) f_{\mathrm{day}} ( \mathrm{daylength} ); \mathrm{GSI}_{\mathrm{bar},n} = \operatorname{mean} ( \mathrm{iGSI} \mathrm{over} \mathrm{the} 21 - \mathrm{day} \mathrm{StateManager} \mathrm{window} ) .
$$

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | GSIPhenology inherits PhenologyBase.forward, which calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical phenology outputs. |
| `explicit_stoichiometric_usage` | PhenologyBase.forward materializes stoichiometric_N and rates for GSIPhenology outputs, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.phenology.turnover.n_retrans_efficiency` | `dimensionless` | `0.6; bounds=(0.3, 0.9)` | `learnable` | Leaf nitrogen retranslocation efficiency during senescence | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.constants.max_lai_threshold` | `m2 m-2` | `15.0; bounds=(5.0, 20.0)` | `fixed` | Maximum reasonable leaf area index threshold for validation | Source: model_validation; Bonan et al. (2011). J. Climate, 24(17), 4603-4623. |
| `biosphere.phenology.gsi.tmin_threshold` | `K` | `271.15; bounds=(263.15, 278.15)` | `learnable` | Minimum temperature threshold for GSI calculation | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.tmin_optimal` | `K` | `278.15; bounds=(273.15, 288.15)` | `learnable` | Optimal minimum temperature for GSI calculation | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.vpd_min` | `Pa` | `900.0; bounds=(500.0, 1500.0)` | `learnable` | Minimum VPD for optimal GSI conditions | Source: literature_synthesis; Oren et al. (1999) Plant Cell Environ. 22, 1515-1526 |
| `biosphere.phenology.gsi.vpd_max` | `Pa` | `4100.0; bounds=(3000.0, 6000.0)` | `learnable` | Maximum VPD for GSI calculation | Source: literature_synthesis; Oren et al. (1999) Plant Cell Environ. 22, 1515-1526 |
| `biosphere.phenology.gsi.daylen_threshold` | `s` | `39300.0; bounds=(28800.0, 50400.0)` | `learnable` | Daylight duration threshold for GSI (10.92 hours) | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.onset_threshold` | `dimensionless` | `0.5; bounds=(0.3, 0.8)` | `learnable` | GSI threshold for phenological onset | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.offset_threshold` | `dimensionless` | `0.5; bounds=(0.3, 0.8)` | `learnable` | GSI threshold for phenological offset | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.memory_days` | `day` | `21; bounds=(1.0, 30.0)` | `fixed` | Number of daily samples in the Jolly GSI moving average | Source: literature_synthesis; Jolly, W.M. et al. (2005). Global Change Biology 11(4), 619-632 |
| `biosphere.phenology.turnover.leaf_turnover_rate` | `s-1` | `7.927e-09; bounds=(3.171e-09, 6.342e-08)` | `learnable` | Base annual leaf turnover rate for deciduous trees | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.turnover.froot_turnover_rate` | `s-1` | `3.171e-08; bounds=(1.585e-08, 9.512e-08)` | `learnable` | Base annual fine root turnover rate | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.morphology.max_lai` | `dimensionless` | `6.0; bounds=(2.0, 10.0)` | `learnable` | Maximum leaf area index for vegetation | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `biosphere.phenology.morphology.min_lai` | `dimensionless` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Minimum leaf area index (dormant season) | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `biosphere.validation.tolerance.phenology` | `dimensionless` | `1e-06; bounds=(1e-12, 0.0001)` | `fixed` | Tolerance for phenology model numerical validation checks | Source: literature_synthesis; Guenther, A. B. et al. (2012) Geosci. Model Dev., 5, 1471-1492 |
| `biosphere.phenology.activity.minimal_dormancy` | `dimensionless` | `0.05; bounds=(0.0, 0.2)` | `fixed` | Minimal metabolic activity level during dormancy | Source: literature_synthesis; Chuine (2000) J. Theor. Biol. 207, 337-347 |
| `biosphere.phenology.gsi.expansion_factor` | `dimensionless` | `1.5; bounds=(1.0, 2.0)` | `learnable` | Growth factor multiplier for expansion phase in GSI model | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.expansion_min_clamp` | `dimensionless` | `0.1; bounds=(0.0, 0.2)` | `fixed` | Minimum clamp value for expansion phase growth factor | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.expansion_max_clamp` | `dimensionless` | `1.0; bounds=(0.8, 1.0)` | `fixed` | Maximum clamp value for expansion phase growth factor | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.full_leaf_factor` | `dimensionless` | `1.2; bounds=(1.0, 1.5)` | `learnable` | Growth factor multiplier for full leaf phase in GSI model | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.full_leaf_min_clamp` | `dimensionless` | `0.8; bounds=(0.6, 0.9)` | `fixed` | Minimum clamp value for full leaf phase growth factor | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.gsi.senescence_factor` | `dimensionless` | `0.8; bounds=(0.5, 1.0)` | `learnable` | Growth factor multiplier for senescence phase in GSI model | Source: literature_synthesis; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948 |
| `biosphere.phenology.gsi.senescence_min_clamp` | `dimensionless` | `0.1; bounds=(0.0, 0.2)` | `fixed` | Minimum clamp value for senescence phase growth factor | Source: literature_synthesis; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948 |
| `biosphere.phenology.gsi.senescence_max_clamp` | `dimensionless` | `1.0; bounds=(0.8, 1.0)` | `fixed` | Maximum clamp value for senescence phase growth factor | Source: literature_synthesis; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948 |
| `biosphere.phenology.gsi.memory_multiplier` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `learnable` | Multiplier for GSI memory in senescence rate calculation | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.computational.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Minimum value for tensor clamping operations | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.computational.clamp_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Maximum value for tensor clamping operations | Source: RELIC parameter database; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.turnover.background_senescence_rate` | `dimensionless` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Background senescence rate during non-senescence periods | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948 |
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
| `state_write_mode` | integrated C and N pool increments use ACCUMULATE with per_timestep true; GSI temporal full states, rates, and observations use REPLACE with declared units |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\phenology.md`.

### References

- Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict foliar phenology in response to climate. Global Change Biology, 11(4), 619-632. (FunctorMetadata, certificate)
- White, M.A., et al. (1997). A continental phenology model for monitoring vegetation responses to interannual climatic variability. Global Biogeochemical Cycles, 11(2), 217-234. (FunctorMetadata, certificate)
- Band, L.E., et al. (1993). Regional Hydro-Ecological Simulation System (RHESSys): An object-oriented approach to spatially distributed modeling of carbon, water, and nutrient cycling. Ecological Modelling, 68(3-4), 129-146. (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-phenology-models-staticphenology)=
## StaticPhenology

Static Date-Based Phenology Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.phenology.models.StaticPhenology` |
| Source | `relic\scientific\biosphere\phenology\models.py:112` |
| Runtime name | `static_phenology` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
StaticPhenology(name: str = 'static_phenology', config: Dict[str, Any] | None = None, phenology_parameters: relic.scientific.biosphere.phenology.parameters.PhenologyParameters | None = None, **kwargs) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'static_phenology'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `phenology_parameters` | <code>relic.scientific.biosphere.phenology.parameters.PhenologyParameters &#124; None</code> | `None` | Constructor control for phenology parameters; see the class runtime contract and ParameterManager table below. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `day_of_year` | forcing | `1` | required calendar forcing port [day] |
| `air_temperature` | forcing | `K` | required temperature forcing port [K] |
| `leaf_carbon` | state | `kg C m-2` | prognostic vegetation carbon pool [kg C m-2] |
| `froot_carbon` | state | `kg C m-2` | prognostic vegetation carbon pool [kg C m-2] |
| `leaf_nitrogen` | state | `kg N m-2` | prognostic vegetation nitrogen pool [kg N m-2] |
| `froot_nitrogen` | state | `kg N m-2` | prognostic vegetation nitrogen pool [kg N m-2] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `leaf_area_index` | state | `m2 m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenological_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenology_senescence_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `phenology_phase` | diagnostic | `dimensionless` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_carbon` | state | `kg C m-2` | decided at run time by the returned write mode | prognostic vegetation carbon pool [kg C m-2] |
| `froot_carbon` | state | `kg C m-2` | decided at run time by the returned write mode | prognostic vegetation carbon pool [kg C m-2] |
| `leaf_nitrogen` | state | `kg N m-2` | decided at run time by the returned write mode | prognostic vegetation nitrogen pool [kg N m-2] |
| `froot_nitrogen` | state | `kg N m-2` | decided at run time by the returned write mode | prognostic vegetation nitrogen pool [kg N m-2] |
| `plant_nitrogen_pool` | state | `kg N m-2` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_litterfall_carbon` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `froot_litterfall_carbon` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `leaf_litterfall_nitrogen` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `froot_litterfall_nitrogen` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `nitrogen_retranslocation` | flux | `kg N m-2 s-1` | decided at run time by the returned write mode | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.carbon_turnover_balance_residual` | output port | `kg C m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.phenology.nitrogen_turnover_balance_residual` | output port | `kg N m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

Static phenology equations:

Growth factor:

$$
f_{growth}(t) =
\begin{cases}
0 & \text{if } t < t_{onset} \\
0.5(1 + \tanh(k(t - t_{onset}))) & \text{if } t_{onset} \leq t < t_{full} \\
1 & \text{if } t_{full} \leq t < t_{senes} \\
0.5(1 + \tanh(k(t_{offset} - t))) & \text{if } t_{senes} \leq t < t_{offset} \\
0 & \text{if } t \geq t_{offset}
\end{cases}
$$

$$
LAI(t) = LAI_{max} \times f_{growth}(t)
$$

Where:

- t: day of year [days]

- t_{onset}: leaf onset date [DOY]

- t_{offset}: leaf offset date [DOY]

- $t_{full} = t_{onset} + expansion_{days}$

- $t_{senes} = t_{offset} - senescence_{days}$

- $k = \mathrm{transition}_{\mathrm{steepness}}$: transition steepness [day-1]

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | StaticPhenology inherits PhenologyBase.forward, which calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical phenology outputs. |
| `explicit_stoichiometric_usage` | PhenologyBase.forward materializes stoichiometric_N and rates for StaticPhenology outputs, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.phenology.turnover.n_retrans_efficiency` | `dimensionless` | `0.6; bounds=(0.3, 0.9)` | `learnable` | Leaf nitrogen retranslocation efficiency during senescence | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.constants.max_lai_threshold` | `m2 m-2` | `15.0; bounds=(5.0, 20.0)` | `fixed` | Maximum reasonable leaf area index threshold for validation | Source: model_validation; Bonan et al. (2011). J. Climate, 24(17), 4603-4623. |
| `biosphere.phenology.static.leaf_onset_doy` | `day` | `120; bounds=(60.0, 180.0)` | `fixed` | Day of year for leaf onset in static phenology | Source: literature_synthesis; Chuine, I. (2000). Journal of Theoretical Biology, 207(3), 337-347 |
| `biosphere.phenology.static.leaf_offset_doy` | `day` | `300; bounds=(200.0, 330.0)` | `fixed` | Calendar day of year when leaves reach offset/dormancy stage | Source: literature_synthesis; Chuine, I. (2000). Journal of Theoretical Biology, 207(3), 337-347 |
| `biosphere.phenology.static.leaf_expansion_days` | `day` | `30; bounds=(10.0, 60.0)` | `fixed` | Number of days for leaf expansion from onset to full development | Source: literature_synthesis; Chuine, I. (2000). Journal of Theoretical Biology, 207(3), 337-347 |
| `biosphere.phenology.static.leaf_senescence_days` | `day` | `45; bounds=(20.0, 90.0)` | `fixed` | Number of days for leaf senescence from onset to complete dormancy | Source: literature_synthesis; Chuine, I. (2000). Journal of Theoretical Biology, 207(3), 337-347 |
| `biosphere.phenology.turnover.leaf_turnover_rate` | `s-1` | `7.927e-09; bounds=(3.171e-09, 6.342e-08)` | `learnable` | Base annual leaf turnover rate for deciduous trees | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.turnover.froot_turnover_rate` | `s-1` | `3.171e-08; bounds=(1.585e-08, 9.512e-08)` | `learnable` | Base annual fine root turnover rate | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Gill & Jackson (2000) New Phytol. 147, 13-31 |
| `biosphere.phenology.morphology.max_lai` | `dimensionless` | `6.0; bounds=(2.0, 10.0)` | `learnable` | Maximum leaf area index for vegetation | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `biosphere.phenology.morphology.min_lai` | `dimensionless` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Minimum leaf area index (dormant season) | Source: literature_synthesis; Running & Coughlan (1988) Ecol. Model. 42, 125-154 |
| `biosphere.validation.tolerance.phenology` | `dimensionless` | `1e-06; bounds=(1e-12, 0.0001)` | `fixed` | Tolerance for phenology model numerical validation checks | Source: literature_synthesis; Guenther, A. B. et al. (2012) Geosci. Model Dev., 5, 1471-1492 |
| `biosphere.phenology.static.transition_steepness` | `day-1` | `0.2; bounds=(0.1, 1.0)` | `learnable` | Steepness parameter for sigmoid transitions in static phenology | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.static.sigmoid_midpoint` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Midpoint (t0) for sigmoid transitions controlling timing of senescence onset | Source: literature_synthesis; White, M.A., et al. (1997). Global Biogeochemical Cycles, 11(2), 217-234. |
| `biosphere.phenology.static.sigmoid_steepness` | `dimensionless` | `4.0; bounds=(1.0, 10.0)` | `learnable` | Steepness parameter (k) for sigmoid transitions in static phenology curves | Source: literature_synthesis; White, M.A., et al. (1997). Global Biogeochemical Cycles, 11(2), 217-234. |
| `biosphere.phenology.computational.sigmoid_scale` | `dimensionless` | `0.5; bounds=(0.1, 1.0)` | `fixed` | Scale factor for tanh-to-sigmoid transformation in phenology transitions | Source: literature_synthesis; White et al. (1997). Global Biogeochemical Cycles, 11(2), 217-234. |
| `biosphere.phenology.computational.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Minimum value for tensor clamping operations | Source: literature_synthesis; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.computational.clamp_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Maximum value for tensor clamping operations | Source: RELIC parameter database; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `biosphere.phenology.turnover.background_senescence_rate` | `dimensionless` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Background senescence rate during non-senescence periods | Source: literature_synthesis; Parton et al. (1987) Soil Sci. Soc. Am. J. 51, 1173-1179; Delpierre et al. (2009) Agric. For. Meteorol. 149, 938-948 |
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
| `state_write_mode` | integrated C and N pool increments use ACCUMULATE with per_timestep true; rates and observations use REPLACE with declared units |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\phenology.md`.

### References

- White, M.A., et al. (1997). A continental phenology model for monitoring vegetation responses to interannual climatic variability. Global Biogeochemical Cycles, 11(2), 217-234. (FunctorMetadata, certificate)
- Running, S.W. & Coughlan, J.C. (1988). A general model of forest ecosystem processes for regional applications. Ecological Modelling, 42(2), 125-154. (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-phenology-phenology-functor-phenologyfunctor)=
## PhenologyFunctor

Project one canonical phenology owner's outputs into diagnostics.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.phenology.phenology_functor.PhenologyFunctor` |
| Source | `relic\scientific\biosphere\phenology\phenology_functor.py:30` |
| Runtime name | `biosphere_phenology_functor` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `observation` |
| Spatial scale | `grid_cell` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
PhenologyFunctor(*, config: dict[str, Any] | None = None, device: torch.device | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `config` | <code>dict[str, Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

The runtime metadata below is introspected from the documented configuration `canonical_model_owner='gsi_phenology'`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `leaf_area_index` | state | `m2 m-2` | selected owner leaf-area observation |
| `phenological_factor` | diagnostic | `dimensionless` | selected owner dimensionless development observation |
| `phenology_senescence_factor` | diagnostic | `dimensionless` | selected owner dimensionless senescence response factor |
| `phenology_phase` | diagnostic | `dimensionless` | selected owner phase observation |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `diagnostics.phenology_state` | output port | `dimensionless` | diagnostics | identity projection of phenological_factor |
| `diagnostics.growth_stage` | output port | `dimensionless` | diagnostics | identity projection of phenology_phase |
| `diagnostics.leaf_area_index` | output port | `m2 m-2` | diagnostics | identity projection of leaf_area_index |
| `diagnostics.phenology.senescence_factor` | output port | `dimensionless` | diagnostics | identity projection of phenology_senescence_factor |

### Governing equations and canonical operator

**Literature or process form**

Static, GDD, GSI, and crop equations are evaluated only by their canonical owners in models.py.

**Canonical RELIC structural form**

$z_{h} = P_{\mathrm{owner}} y_{h}$, where $P_{\mathrm{owner}}$ is a parameter-free diagnostic observation projection.

| Entry | Meaning |
| --- | --- |
| `P_owner` | compile-time-selected identity and rename projection over required owner observations |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | selected canonical owner observations |
| `effort_variable` | not applicable to a diagnostic observation projection |
| `state_advancer` | none; $z_{h} = P_{\mathrm{owner}} y_{h}$ is diagnostic only |
| `runtime_usage` | forward validates and projects dependency-injected owner observations without state advancement |

### Composition channels and executable PortSpec API

Certificate composition channels: `selected canonical phenology owner dependency-output channel`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
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
| `forward_signature` | forward(self, state, time, dt, subprocess_outputs) |
| `state_write_mode` | explicit REPLACE per-step diagnostics only; selected canonical model remains the sole scientific state owner |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\phenology.md`.

### References

- White et al. (1997) Global Biogeochemical Cycles 11:217-234 (FunctorMetadata)
- Jolly et al. (2005) Global Change Biology 11:619-632 (FunctorMetadata)
- Richardson et al. (2013) Agricultural and Forest Meteorology 169:156-173 (FunctorMetadata)
- White, M.A. et al. (1997). Global Biogeochemical Cycles 11:217-234. (certificate)
- Jolly, W.M. et al. (2005). Global Change Biology 11:619-632. (certificate)
- McMaster, G.S. and Wilhelm, W.W. (1997). Agricultural and Forest Meteorology 87:291-300. (certificate)
- Jones, J.W. et al. (2003). European Journal of Agronomy 18:235-265. (certificate)
- Richardson, A.D. et al. (2013). Agricultural and Forest Meteorology 169:156-173. (certificate)

(class-relic-scientific-biosphere-photosynthesis-farquhar-farquharphotosynthesis)=
## FarquharPhotosynthesis

Unified Farquhar-von Caemmerer-Berry C3 Photosynthesis Model.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.photosynthesis.farquhar.FarquharPhotosynthesis` |
| Source | `relic\scientific\biosphere\photosynthesis\farquhar.py:79` |
| Runtime name | `farquhar_photosynthesis` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `patch` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
FarquharPhotosynthesis(name: str = 'farquhar_photosynthesis', config: Dict[str, Any] | None = None, device: torch.device | None = None, enable_gpu: bool = True, validate_inputs: bool = True, pft_name: str | None = None, pathway: str | None = None, vcmax25: float | None = None, jmax25: float | None = None, rd25: float | None = None, parameter_manager: Any | None = None, *, environment_source: Literal['canonical_providers', 'external_forcing'] = 'canonical_providers')
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'farquhar_photosynthesis'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `enable_gpu` | `bool` | `True` | Compatibility control accepted by this constructor; effective device placement is still governed by simulator device policy. |
| `validate_inputs` | `bool` | `True` | Enables fail-fast validation of required runtime inputs. |
| `pft_name` | <code>str &#124; None</code> | `None` | Optional plant functional type namespace used for PFT parameter overrides. |
| `pathway` | <code>str &#124; None</code> | `None` | Constructor control for pathway; see the class runtime contract and ParameterManager table below. |
| `vcmax25` | <code>float &#124; None</code> | `None` | Constructor control for vcmax25; see the class runtime contract and ParameterManager table below. |
| `jmax25` | <code>float &#124; None</code> | `None` | Constructor control for jmax25; see the class runtime contract and ParameterManager table below. |
| `rd25` | <code>float &#124; None</code> | `None` | Constructor control for rd25; see the class runtime contract and ParameterManager table below. |
| `parameter_manager` | <code>typing.Any &#124; None</code> | `None` | Constructor control for parameter manager; see the class runtime contract and ParameterManager table below. |
| `environment_source` | `typing.Literal['canonical_providers', 'external_forcing']` | `'canonical_providers'` | Constructor control for environment source; see the class runtime contract and ParameterManager table below. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | required atmospheric temperature forcing in K |
| `atmospheric_co2_ppm` | forcing | `umol mol-1` | required atmospheric CO2 mole-fraction forcing in umol mol-1 |
| `ppfd` | flux | `umol photons m-2 s-1` | required photon flux from atmospheric_radiation, or from external forcing under environment_source = external_forcing, in umol photons m-2 s-1 |
| `vapor_pressure_deficit` | forcing | `Pa` | required atmospheric demand from environmental_forcing, or from external forcing under environment_source = external_forcing, in Pa |
| `vcmax_scaling` | forcing | `1` | optional C3/C4 capacity control |
| `vpmax_scaling` | forcing | `1` | optional C4 PEP capacity control |
| `jmax_scaling` | forcing | `1` | optional electron-transport capacity control |
| `water_stress_factor` | diagnostic | `dimensionless` | optional water limitation control in the unit interval |
| `temperature_stress_factor` | diagnostic | `dimensionless` | optional damage control outside enzyme temperature response in the unit interval |
| `ozone_stress_factor` | diagnostic | `dimensionless` | optional C3 ozone limitation control in the unit interval |
| `leaf_area_index` | state | `m2 m-2` | required canopy state of the Beer-Lambert scaling L_eff = (1 - exp(-k LAI)) / k [m2 m-2]; no default |
| `actual_vapor_pressure` | forcing | `Pa` | actual vapour pressure e_a, read only with temporal_support daylight_gauss (D_k = max(e0(T_k) - e_a, 0) at the daylight samples) |
| `air_temperature_max` | forcing | `K` | daily maximum air temperature, read only with temporal_support daylight_gauss (diurnal course T_k of the samples) |
| `air_temperature_min` | forcing | `K` | daily minimum air temperature, read only with temporal_support daylight_gauss (diurnal course T_k of the samples) |
| `latitude` | forcing | `degrees_north` | latitude of the cell, read only with temporal_support daylight_gauss (daylight geometry of the samples) |
| `day_of_year` | forcing | `1` | day of year, read only with temporal_support daylight_gauss (daylight geometry of the samples) |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `gross_photosynthesis` | flux | `umol CO2 m-2 s-1` | decided at run time by the returned write mode | gross molar assimilation is exposed as an instantaneous replacement port |
| `net_photosynthesis` | flux | `umol CO2 m-2 s-1` | decided at run time by the returned write mode | net molar assimilation equals gross assimilation minus dark respiration |
| `gross_primary_productivity` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | gross assimilation is converted to kg C m-2 s-1 with the carbon molar mass SSOT |
| `electron_transport_rate` | flux | `umol e- m-2 s-1` | decided at run time by the returned write mode | retained public port is consumed by mechanistic isoprene synthesis |
| `diagnostics.photosynthesis.dark_respiration` | diagnostic | `umol CO2 m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.photosynthesis.transport_limited` | diagnostic | `umol CO2 m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.photosynthesis.jmax_actual` | diagnostic | `umol e- m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.photosynthesis.limitation_factor` | diagnostic | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.photosynthesis.pathway_type` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `net_photosynthesis_daylight_samples` | diagnostic | `umol CO2 m-2 s-1` | decided at run time by the returned write mode | net assimilation at the daylight Gauss samples, written only with temporal_support daylight_gauss |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Input declaration difference: `actual_vapor_pressure`, `air_temperature_max`, `air_temperature_min`, `day_of_year`, `latitude`.
Output declaration difference: `net_photosynthesis_daylight_samples`.
:::

### Governing equations and canonical operator

**Literature or process form**

$$
A = \min ( \mathrm{Ac} , \mathrm{Aj} ) - \mathrm{Rd}; \mathrm{Ac} = V_{c,\max} \cdot ( \mathrm{Ci} - \Gamma_{\star} ) / ( \mathrm{Ci} + K_c \cdot ( 1 + O / K_o ) ); \mathrm{Aj} = J \cdot ( \mathrm{Ci} - \Gamma_{\star} ) / ( 4 \cdot ( \mathrm{Ci} + 2 \cdot \Gamma_{\star} ) )
$$

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | open gross-assimilation, net-assimilation, and dark-respiration output ports |
| `effort_variable` | atmospheric CO2, PPFD, VPD, temperature, and optional control drivers; no thermodynamic effort proof is claimed |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ uses an identity-like output incidence and advance_reaction_network with `open_system_route=port` |
| `runtime_usage` | FarquharPhotosynthesis.forward calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical outputs. |
| `explicit_stoichiometric_usage` | FarquharPhotosynthesis.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `vapor_pressure_deficit`, `ppfd`, `gross_primary_productivity`, `electron_transport_rate`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.photosynthesis.pathway` | `dimensionless` | `'C3'; bounds=(-inf, inf)` | `fixed` | Photosynthetic pathway type (C3 or C4) | Source: biological_classification; Farquhar et al. (1980) Planta 149:78-90 |
| `biosphere.photosynthesis.vcmax25` | `umol CO2 m-2 s-1` | `75.3; bounds=(50.0, 120.0)` | `fixed` | Maximum carboxylation rate at 25degC - calibrated for Harvard Forest | Source: flux_tower_calibration; doi:10.1071/PP9900137; Farquhar et al. (1980) Planta 149:78-90 |
| `biosphere.photosynthesis.jmax25` | `umol e- m-2 s-1` | `148.6; bounds=(80.0, 250.0)` | `fixed` | Maximum electron transport rate at 25degC - calibrated for Harvard Forest | Source: flux_tower_calibration; doi:10.1071/PP9900137; Farquhar et al. (1980) Planta 149:78-90 |
| `biosphere.photosynthesis.rd25` | `umol CO2 m-2 s-1` | `1.2; bounds=(0.1, 5.0)` | `learnable` | Dark respiration rate at 25degC | Source: literature_synthesis; Bernacchi et al. (2001) Plant Physiol 125:1938-1948 |
| `biosphere.photosynthesis.kc25` | `umol mol-1` | `404.0; bounds=(300.0, 500.0)` | `fixed` | Michaelis-Menten constant for CO2 at 25degC | Source: literature_synthesis; doi:10.1104/pp.125.4.1938; Bernacchi et al. (2001) Plant Physiol 125:1938-1948 |
| `biosphere.photosynthesis.ko25` | `mmol mol-1` | `248.0; bounds=(200.0, 300.0)` | `fixed` | Michaelis-Menten constant for O2 at 25degC | Source: literature_synthesis; doi:10.1104/pp.125.4.1938; Bernacchi et al. (2001) Plant Physiol 125:1938-1948 |
| `biosphere.photosynthesis.gamma_star25` | `umol mol-1` | `42.75; bounds=(30.0, 60.0)` | `fixed` | CO2 compensation point at 25degC | Source: literature_synthesis; Bernacchi et al. (2001) Plant Physiol 125:1938-1948 |
| `biosphere.photosynthesis.ha_vcmax` | `J mol-1` | `65330.0; bounds=(50000.0, 80000.0)` | `fixed` | Activation energy for Vcmax temperature response | Source: literature_synthesis; doi:10.1104/pp.125.4.1938; Bernacchi et al. (2001) Plant Physiol 125:1938-1948 |
| `biosphere.photosynthesis.hd_vcmax` | `J mol-1` | `149252.0; bounds=(100000.0, 200000.0)` | `fixed` | Deactivation energy for Vcmax | Source: literature_synthesis; Kattge & Knorr (2007) Plant Cell Environ 30:1176-1190 |
| `biosphere.photosynthesis.s_vcmax` | `J mol-1 K-1` | `486.0; bounds=(400.0, 600.0)` | `fixed` | Entropy term for Vcmax temperature response | Source: literature_synthesis; Kattge & Knorr (2007) Plant Cell Environ 30:1176-1190 |
| `biosphere.photosynthesis.ha_jmax` | `J mol-1` | `43540.0; bounds=(30000.0, 60000.0)` | `fixed` | Activation energy for Jmax temperature response | Source: literature_synthesis; doi:10.1104/pp.125.4.1938; Bernacchi et al. (2001) Plant Physiol 125:1938-1948 |
| `biosphere.photosynthesis.hd_jmax` | `J mol-1` | `152044.0; bounds=(100000.0, 200000.0)` | `fixed` | Deactivation energy for Jmax | Source: literature_synthesis; Kattge & Knorr (2007) Plant Cell Environ 30:1176-1190 |
| `biosphere.photosynthesis.s_jmax` | `J mol-1 K-1` | `495.0; bounds=(400.0, 600.0)` | `fixed` | Entropy term for Jmax temperature response | Source: literature_synthesis; Kattge & Knorr (2007) Plant Cell Environ 30:1176-1190 |
| `biosphere.respiration.rd_q10` | `dimensionless` | `2.0; bounds=(1.5, 3.5)` | `fixed` | Q10 temperature coefficient for dark respiration | Source: literature_synthesis; Lloyd & Taylor (1994); Bernacchi et al. (2001) |
| `biosphere.photosynthesis.ha_kc` | `J mol-1` | `79430.0; bounds=(60000.0, 100000.0)` | `fixed` | Activation energy for Kc | Source: literature_synthesis; Bernacchi et al. (2001) Plant Physiol 125:1938-1948 |
| `biosphere.photosynthesis.ha_ko` | `J mol-1` | `36380.0; bounds=(20000.0, 50000.0)` | `fixed` | Activation energy for Ko | Source: literature_synthesis; Bernacchi et al. (2001) Plant Physiol 125:1938-1948 |
| `biosphere.photosynthesis.ha_gamma_star` | `J mol-1` | `37830.0; bounds=(20000.0, 50000.0)` | `fixed` | Activation energy for CO2 compensation point | Source: literature_synthesis; Bernacchi et al. (2001) Plant Physiol 125:1938-1948 |
| `biosphere.photosynthesis.alpha` | `dimensionless` | `0.24; bounds=(0.1, 0.4)` | `fixed` | Quantum efficiency of photosystem II | Source: literature_synthesis; Long et al. (1993) Plant Cell Environ 16:185-194 |
| `biosphere.photosynthesis.theta` | `dimensionless` | `0.7; bounds=(0.5, 0.95)` | `fixed` | Curvature parameter for light response | Source: literature_synthesis; Farquhar & Wong (1984) Aust J Plant Physiol 11:191-210 |
| `biosphere.photosynthesis.o2_concentration` | `mmol mol-1` | `210.0; bounds=(200.0, 220.0)` | `fixed` | Atmospheric O2 concentration | Source: atmospheric_standard; Standard atmospheric composition |
| `biosphere.photosynthesis.vpmax25` | `umol CO2 m-2 s-1` | `80.0; bounds=(20.0, 150.0)` | `learnable` | Maximum PEP carboxylation rate at 25degC (C4 plants) | Source: literature_synthesis; von Caemmerer (2000) Biochemical Models of Leaf Photosynthesis |
| `biosphere.photosynthesis.kp25` | `umol mol-1` | `80.0; bounds=(50.0, 150.0)` | `fixed` | Michaelis constant for PEP carboxylase at 25degC | Source: literature_synthesis; von Caemmerer (2000) Biochemical Models of Leaf Photosynthesis |
| `biosphere.photosynthesis.phi` | `dimensionless` | `0.2; bounds=(0.1, 0.4)` | `fixed` | C4 bundle-sheath leakiness phi = L / Vp, the fraction of the PEP carboxylation rate that leaks back out of the bundle sheath (von Caemmerer 2000; 2021, Eq. 5); the C4 path uses Ap = Vp (1 - phi) | Source: literature_synthesis; von Caemmerer (2000) Biochemical Models of Leaf Photosynthesis |
| `biosphere.photosynthesis.ve_max25` | `umol CO2 m-2 s-1` | `60.0; bounds=(20.0, 100.0)` | `learnable` | Maximum Rubisco carboxylation capacity of the C4 bundle sheath at 25degC, the Rubisco-limited rate V_max of Collatz et al. (1992) and Vcmax of the von Caemmerer (2000) C4 model; the C4 path caps assimilation at Ae = Ve_max in A = min(Ap, Aj, Ae) | Source: literature_synthesis; von Caemmerer (2000) Biochemical Models of Leaf Photosynthesis |
| `biosphere.photosynthesis.ha_vpmax` | `J mol-1` | `70000.0; bounds=(50000.0, 90000.0)` | `fixed` | Activation energy for Vpmax | Source: literature_synthesis; Massad et al. (2007) Plant Cell Environ 30:1284-1298 |
| `biosphere.photosynthesis.hd_vpmax` | `J mol-1` | `150000.0; bounds=(100000.0, 200000.0)` | `fixed` | Deactivation energy for Vpmax | Source: literature_synthesis; Massad et al. (2007) Plant Cell Environ 30:1284-1298 |
| `biosphere.photosynthesis.s_vpmax` | `J mol-1 K-1` | `490.0; bounds=(400.0, 600.0)` | `fixed` | Entropy term for Vpmax | Source: literature_synthesis; Massad et al. (2007) Plant Cell Environ 30:1284-1298 |
| `biosphere.photosynthesis.ha_kp` | `J mol-1` | `40000.0; bounds=(20000.0, 60000.0)` | `fixed` | Activation energy for Kp | Source: literature_synthesis; Massad et al. (2007) Plant Cell Environ 30:1284-1298 |
| `biosphere.photosynthesis.ha_ve_max` | `J mol-1` | `50000.0; bounds=(30000.0, 70000.0)` | `fixed` | Activation energy for Ve_max | Source: literature_synthesis; Massad et al. (2007) Plant Cell Environ 30:1284-1298 |
| `biosphere.photosynthesis.hd_ve_max` | `J mol-1` | `145000.0; bounds=(100000.0, 190000.0)` | `fixed` | Deactivation energy for Ve_max | Source: literature_synthesis; Massad et al. (2007) Plant Cell Environ 30:1284-1298 |
| `biosphere.photosynthesis.s_ve_max` | `J mol-1 K-1` | `485.0; bounds=(400.0, 570.0)` | `fixed` | Entropy term for Ve_max | Source: literature_synthesis; Massad et al. (2007) Plant Cell Environ 30:1284-1298 |
| `biosphere.photosynthesis.vcmax_min_scaling` | `dimensionless` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Minimum Vcmax scaling factor for numerical stability and biological realism in photosynthesis calculations | Source: Farquhar model implementation; Farquhar et al. (1980) Planta 149:78-90 |
| `biosphere.photosynthesis.vcmax_max_scaling` | `dimensionless` | `3.0; bounds=(1.0, 10.0)` | `fixed` | Maximum Vcmax scaling factor for numerical stability and biological realism in photosynthesis calculations | Source: Farquhar model implementation; Farquhar et al. (1980) Planta 149:78-90 |
| `biosphere.photosynthesis.canopy_light_extinction` | `dimensionless` | `0.5; bounds=(0.2, 0.9)` | `fixed` | Beer-Lambert canopy light extinction coefficient for LAI scaling in canopy photosynthesis | Source: literature_synthesis; Monsi and Saeki (1953) Japanese Journal of Botany 14:22-52; Monteith and Unsworth (2013) Principles of Environmental Physics |
| `biosphere.photosynthesis.rubisco_kinetic_factor1` | `dimensionless` | `4.0; bounds=(2.0, 6.0)` | `fixed` | Rubisco kinetic factor for CO2 specificity | Source: literature_synthesis; Farquhar et al. (1980) Planta 149:78-90 |
| `biosphere.photosynthesis.rubisco_kinetic_factor2` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `fixed` | Secondary Rubisco kinetic factor | Source: literature_synthesis; Farquhar et al. (1980) Planta 149:78-90 |
| `biosphere.photosynthesis.c4_pathway_type` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Indicator variable for C4 pathway activation (1.0 for C4) | Source: model_default; Model configuration |
| `validation.numerical.epsilon_divide` | `dimensionless` | `1e-10; bounds=(1e-11, 1e-06)` | `fixed` | Small epsilon value to prevent division by zero | Source: numerical_methods; IEEE 754 floating point standard |
| `validation.numerical.clamp_min_zero` | `dimensionless` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Minimum clamp value for zero-bounded quantities | Source: numerical_methods; Numerical methods best practices |
| `validation.numerical.clamp_min_small` | `dimensionless` | `0.1; bounds=(0.001, 1.0)` | `fixed` | Small positive minimum clamp value | Source: numerical_methods; Numerical methods best practices |
| `validation.numerical.clamp_max_one` | `dimensionless` | `1.0; bounds=(1.0, 1.0)` | `fixed` | Maximum clamp value for unit-bounded quantities | Source: numerical_methods; Numerical methods best practices |
| `validation.numerical.clamp_min_negative` | `MPa` | `-0.1; bounds=(-5.0, -0.001)` | `fixed` | Minimum clamp threshold for negative pressure potentials to avoid numerical instability | Source: numerical_methods; Campbell, G.S. & Norman, J.M. (1998). Environmental Biophysics.; Tyree, M.T. & Zimmermann, M.H. (2002). Xylem Structure and the Ascent of Sap. |
| `validation.numerical.clamp_max_zero` | `MPa` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Maximum clamp threshold for pressure potentials (e.g., water potential) capped at 0 MPa | Source: numerical_methods; Numerical methods best practices |
| `validation.numerical.clamp_min_temperature` | `K` | `233.15; bounds=(200.0, 273.15)` | `fixed` | Minimum valid temperature for biological processes (-40 degC) | Source: biological_constraints; Physiological temperature limits |
| `validation.numerical.clamp_max_temperature` | `K` | `323.15; bounds=(273.15, 373.15)` | `fixed` | Maximum valid temperature for biological processes (50 degC) | Source: biological_constraints; Physiological temperature limits |
| `validation.numerical.clamp_min_co2` | `umol mol-1` | `10.0; bounds=(1.0, 100.0)` | `fixed` | Minimum valid CO2 concentration | Source: atmospheric_data; Atmospheric chemistry |
| `validation.numerical.clamp_max_co2` | `umol mol-1` | `5000.0; bounds=(1000.0, 10000.0)` | `fixed` | Maximum valid CO2 concentration | Source: atmospheric_data; Atmospheric chemistry |
| `validation.numerical.vpd_response_d0` | `Pa` | `1500.0; bounds=(500.0, 3000.0)` | `fixed` | Leuning vapor pressure deficit response scale D0 in canonical Pa units | Source: literature_synthesis; Leuning (1995) Plant Cell Environ 18:339-355 |
| `validation.numerical.c4_electron_factor` | `dimensionless` | `3.0; bounds=(2.0, 4.0)` | `fixed` | Electron transport factor for C4 photosynthesis | Source: literature_synthesis; von Caemmerer (2000) Biochemical Models of Leaf Photosynthesis |
| `validation.numerical.max_ppfd_validation` | `umol photons m-2 s-1` | `3000.0; bounds=(2000.0, 5000.0)` | `fixed` | Maximum valid photosynthetic photon flux density | Source: radiation_data; Maximum observed photosynthetic photon flux density |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | explicit UpdateItem REPLACE with $\mathrm{per}_{\mathrm{timestep}} = \mathrm{false}$ through StateManager |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\photosynthesis.md`.

### References

- Farquhar, G.D., von Caemmerer, S., Berry, J.A. (1980). A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. Planta 149, 78-90 (FunctorMetadata, certificate)
- Medlyn et al. (2002). Temperature response of parameters of a biochemically based model of photosynthesis. Plant, Cell & Environment 25, 1167-1179 (FunctorMetadata, certificate)
- Bernacchi et al. (2001). Improved temperature response functions for models of Rubisco-limited photosynthesis. Plant, Cell & Environment 24, 253-259 (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-photosynthesis-leaf-energy-balance-leafenergybalance)=
## LeafEnergyBalance

Diagnostic big-leaf energy balance: leaf temperature from canopy conductance and radiation.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.photosynthesis.leaf_energy_balance.LeafEnergyBalance` |
| Source | `relic\scientific\biosphere\photosynthesis\leaf_energy_balance.py:79` |
| Runtime name | `leaf_energy_balance` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `observation` |
| Spatial scale | `patch` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
LeafEnergyBalance(name: str = 'leaf_energy_balance', config: dict | None = None, *, parameter_manager: Any = None, device: Any = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'leaf_energy_balance'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `parameter_manager` | `typing.Any` | `None` | Constructor control for parameter manager; see the class runtime contract and ParameterManager table below. |
| `device` | `typing.Any` | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Configuration modes

The declared runtime interface depends on the constructor configuration; there is no default mode. Each documented mode was instantiated. The table lists the variables that only that mode declares; every other variable of the runtime contracts below is declared in all modes.

| Mode | Inputs declared only in this mode | Outputs declared only in this mode |
| --- | --- | --- |
| `temporal_support='daily'` | `air_temperature_max`, `air_temperature_min` | `leaf_temperature_max` |
| `temporal_support='instantaneous'` | `soil_heat_flux` | none |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | required StateManager port in K, the linearisation point of e_s |
| `vapor_pressure_deficit` | forcing | `Pa` | required StateManager port in Pa, the deficit of the air at the reference height |
| `wind_speed` | forcing | `m s-1` | required StateManager port in m s-1 at the reference height (FAO-56 Eq. 4) |
| `atmospheric_pressure` | forcing | `Pa` | required StateManager port in Pa |
| `net_radiation` | flux | `W m-2` | required upstream StateManager port in W m-2, all of it available to the big leaf |
| `canopy_conductance` | state | `m s-1` | required upstream StateManager port in m s-1 (Medlyn g_s L_eff R T / P or forcing); 0 is the dry-leaf limit |
| `leaf_area_index` | state | `m2 m-2` | required StateManager port in m2 m-2; LAI <= 0 declares no canopy |
| `air_temperature_max` | forcing | `K` | required StateManager port in K with temporal_support daily only (e_s of FAO-56 Eq. 12 and T_leaf,max) |
| `air_temperature_min` | forcing | `K` | required StateManager port in K with temporal_support daily only (e_s of FAO-56 Eq. 12) |
| `soil_heat_flux` | forcing | `W m-2` | required signed StateManager port in W m-2 with temporal_support instantaneous only (daily G = 0, FAO-56 Eq. 42) |
| `actual_vapor_pressure` | forcing | `Pa` | actual vapour pressure e_a, read only with temporal_support daylight_gauss (replaces vapor_pressure_deficit) |
| `canopy_conductance_daylight_samples` | diagnostic | `m s-1` | per-sample Medlyn canopy conductance g_c,k, read only with temporal_support daylight_gauss (replaces canopy_conductance) |
| `shortwave_net` | flux | `W m-2` | net shortwave S_n, read with temporal_support daylight_gauss and with energy_balance_form radiative_canopy_share |
| `latitude` | forcing | `degrees_north` | latitude of the cell, read only with temporal_support daylight_gauss |
| `day_of_year` | forcing | `1` | day of year, read only with temporal_support daylight_gauss |
| `longwave_down` | flux | `W m-2` | downward longwave L_down of the radiation owner, read only with energy_balance_form radiative_canopy_share (replaces net_radiation) |
| `surface_emissivity` | forcing | `1` | surface emissivity eps of the radiation owner, read only with energy_balance_form radiative_canopy_share |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `leaf_temperature` | diagnostic | `K` | decided at run time by the returned write mode | canonical leaf-support output T_air + dT, instantaneous REPLACE |
| `diagnostics.leaf_energy_balance.leaf_air_temperature_difference` | output port | `K` | diagnostics | signed dT of the linearised balance, 0 where no canopy |
| `diagnostics.leaf_energy_balance.sensible_heat_flux` | output port | `W m-2` | diagnostics | H = rho c_p g_a dT, positive leaf to air |
| `diagnostics.leaf_energy_balance.latent_heat_flux` | output port | `W m-2` | diagnostics | lambda E = A - H, the Penman-Monteith latent heat, not projected to nonnegative values |
| `diagnostics.leaf_energy_balance.available_energy` | output port | `W m-2` | diagnostics | A = R_n - G |
| `diagnostics.leaf_energy_balance.aerodynamic_conductance` | output port | `m s-1` | diagnostics | g_a of fao56_aerodynamic_conductance, shared with PenmanMonteithET |
| `diagnostics.leaf_energy_balance.no_canopy` | output port | `1` | diagnostics | boolean LAI <= 0 |
| `diagnostics.leaf_energy_balance.vapor_pressure_floor_active` | output port | `1` | diagnostics | boolean e_s - D < 0, replaced by 0 in the air density only |
| `leaf_temperature_max` | diagnostic | `K` | decided at run time by the returned write mode | declared daily-maximum estimate T_max + dT, temporal_support daily only, instantaneous REPLACE |
| `leaf_temperature_daylight_samples` | diagnostic | `K` | decided at run time by the returned write mode | leaf temperature T_l,k at the daylight Gauss samples, temporal_support daylight_gauss only |
| `diagnostics.leaf_energy_balance.isothermal_net_radiation` | output port | `W m-2` | diagnostics | R_ni = S_n + eps L_down - eps sigma T_a^4, radiative_canopy_share only |
| `diagnostics.leaf_energy_balance.canopy_energy_fraction` | output port | `1` | diagnostics | canopy share 1 - exp(-k LAI) of the available energy, radiative_canopy_share only |
| `diagnostics.leaf_energy_balance.radiative_conductance` | output port | `m s-1` | diagnostics | g_R = 4 eps sigma T_a^3 / (rho c_p), radiative_canopy_share only |
| `diagnostics.leaf_energy_balance.longwave_emission_change` | output port | `W m-2` | diagnostics | L = rho c_p g_R dT, radiative_canopy_share only |
| `diagnostics.leaf_energy_balance.stability_parameter` | output port | `1` | diagnostics | self-consistent Monin-Obukhov zeta of the bisection, radiative_canopy_share only |
| `diagnostics.leaf_energy_balance.stability_iteration_residual` | output port | `K` | diagnostics | \\|dT(final) - dT(zeta*)\\| of the stability solver, radiative_canopy_share only |
| `diagnostics.leaf_energy_balance.convective_velocity` | output port | `m s-1` | diagnostics | U_c = beta w* (zeta < 0), free_convection beljaars only |
| `diagnostics.leaf_energy_balance.gusty_wind_speed` | output port | `m s-1` | diagnostics | U = sqrt(u^2 + U_c^2), free_convection beljaars only |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Input declaration difference: `actual_vapor_pressure`, `canopy_conductance_daylight_samples`, `day_of_year`, `latitude`, `longwave_down`, `shortwave_net`, `surface_emissivity`.
Output declaration difference: `diagnostics.leaf_energy_balance.canopy_energy_fraction`, `diagnostics.leaf_energy_balance.convective_velocity`, `diagnostics.leaf_energy_balance.gusty_wind_speed`, `diagnostics.leaf_energy_balance.isothermal_net_radiation`, `diagnostics.leaf_energy_balance.longwave_emission_change`, `diagnostics.leaf_energy_balance.radiative_conductance`, `diagnostics.leaf_energy_balance.stability_iteration_residual`, `diagnostics.leaf_energy_balance.stability_parameter`, `leaf_temperature_daylight_samples`.
:::

### Governing equations and canonical operator

**Literature or process form**

$T_{l}$ - $T_{a} = [ r_{\mathrm{HR}} ( r_{\mathrm{aW}} + r_{c} ) \gamma A / ( \rho c_{p} ) - r_{\mathrm{HR}} D ] / [ \gamma ( r_{\mathrm{aW}} + r_{c} ) + s r_{\mathrm{HR}} ] ( \mathrm{Jones} 2014 , \mathrm{Chapter} 9 )$ with $r_{\mathrm{HR}} = r_{\mathrm{aW}} = r_{a}$; lambda E of Monteith (1965) and Allen et al. (1998) Eq. 3

**Canonical RELIC structural form**

$y_{h} = O_{h} x_{h}$ through the declared observation operator

| Entry | Meaning |
| --- | --- |
| `O_h` | declared steady linearised big-leaf energy-balance observation map |
| `H_h` | diagnostic metric or identity Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | supplied air temperature, VPD, wind, pressure, net radiation, canopy conductance and LAI observation inputs |
| `effort_variable` | not applicable to this level-1 steady diagnostic |
| `state_advancer` | $y_{h} = O_{h} x_{h}$ through the declared observation operator |
| `runtime_usage` | LeafEnergyBalance.forward solves the steady linearised energy balance once per call and emits leaf_temperature (and on the daily support leaf_temperature_max) without advancing any heat or material pool. |

### Composition channels and executable PortSpec API

Certificate composition channels: `air_temperature`, `vapor_pressure_deficit`, `wind_speed`, `atmospheric_pressure`, `net_radiation`, `canopy_conductance`, `leaf_area_index`, `leaf_temperature`, `air_temperature_max`, `air_temperature_min`, `soil_heat_flux`, `leaf_temperature_max`, `actual_vapor_pressure`, `canopy_conductance_daylight_samples`, `shortwave_net`, `latitude`, `day_of_year`, `longwave_down`, `surface_emissivity`, `leaf_temperature_daylight_samples`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.evapotranspiration.roughness_length` | `m` | `2.0; bounds=(0.001, 10.0)` | `learnable` | Momentum roughness length for forest canopy aerodynamics | Source: literature_synthesis; FAO-56 Irrigation and Drainage Paper; Brutsaert (1982) Evaporation into the Atmosphere |
| `hydrosphere.evapotranspiration.displacement_height` | `m` | `13.4; bounds=(0.0, 50.0)` | `fixed` | Zero plane displacement height d of the FAO-56 logarithmic wind profile (FAO-56 relates d = 2/3 h for a canopy of height h; RELIC reads d directly) | Source: empirical_relationship; FAO-56 Irrigation and Drainage Paper; Thom (1975) Boundary Layer Meteorology 9:319-343 |
| `hydrosphere.evapotranspiration.reference_height` | `m` | `30.0; bounds=(2.0, 200.0)` | `fixed` | Reference height for wind speed measurements | Source: measurement_standard; FAO-56 Irrigation and Drainage Paper; WMO Guide to Meteorological Instruments |
| `hydrosphere.physical.heat_to_momentum_roughness_ratio` | `dimensionless` | `0.1; bounds=(0.05, 0.2)` | `fixed` | Ratio of roughness length for heat to momentum (z0h/z0m) | Source: boundary_layer_theory; Brutsaert (1982) Evaporation into the Atmosphere; Garratt (1992) The Atmospheric Boundary Layer |
| `hydrosphere.numerical.minimum_wind_speed` | `m s-1` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Minimum wind speed for aerodynamic calculations to prevent division by zero | Source: meteorological_modeling; Allen et al. (1998) FAO-56 Irrigation Paper; Monteith (1965) Symp. Soc. Exp. Biol. 19:205-224 |
| `hydrosphere.numerical.aerodynamic_conductance_bounds` | `m s-1` | `[0.001, 1.0]; bounds=([0.0001, 0.01], [0.1, 10.0])` | `fixed` | Physical bounds for aerodynamic conductance [min, max] | Source: boundary_layer_meteorology; Monteith (1965) Symp. Soc. Exp. Biol. 19:205-224; Allen et al. (1998) FAO-56 Irrigation Paper |
| `hydrosphere.numerical.temperature_bounds` | `K` | `[273.15, 323.15]; bounds=([183.15, 273.15], [323.15, 373.15])` | `fixed` | Admitted near-surface air temperature for hydrosphere calculations [min, max]. The default keeps 273.15 K as minimum because the default sub-freezing phase of the evapotranspiration laws is liquid. A run with sub_freezing_phase ice may set the minimum down to 183.15 K (range widened on 2026-11-05 from 223.15 K for global daily runs, terrestrial record low -89.2 degC): the Tetens form over ice deviates from Goff-Gratch by -1.6 percent at 233.15 K, -4.7 percent at 213.15 K and -7.2 percent at 203.15 K, where the saturation vapour pressure is below 0.3 Pa. | Source: meteorological_standards; WMO Guide to Meteorological Instruments; WMO Archive of Weather and Climate Extremes |
| `hydrosphere.numerical.net_radiation_bounds` | `W m-2` | `[-500.0, 1500.0]; bounds=([-2000.0, 0.0], [1000.0, 2500.0])` | `fixed` | Reasonable net radiation bounds for ET calculations [min, max] | Source: energy_balance_validation; Allen et al. (1998) FAO Irrigation and Drainage Paper 56; Monteith (1965) Symp. Soc. Exp. Biol. 19:205-224 |
| `hydrosphere.physical.atmospheric_pressure_min` | `Pa` | `44000.0; bounds=(30000.0, 110000.0)` | `fixed` | Minimum plausible surface atmospheric pressure used to guard hydrosphere flux calculations: US Standard Atmosphere 1976 pressure at 6,500 m, above the highest vascular plants (about 6,150 m, Himalaya). Lowered from 70,000 Pa on 2026-11-05 for vegetated plateaus (Tibet, Andes); the psychrometric constant gamma = c_p p / (epsilon L) holds at any pressure. | Source: atmospheric_climatology; US Standard Atmosphere 1976 |
| `hydrosphere.physical.atmospheric_pressure_max` | `Pa` | `110000.0; bounds=(80000.0, 120000.0)` | `fixed` | Maximum plausible surface atmospheric pressure used to guard hydrosphere flux calculations: above the Dead Sea shore (-430 m, about 106.5 kPa) under a strong anticyclone. Raised from 106,000 Pa on 2026-11-05. | Source: atmospheric_climatology; US Standard Atmosphere 1976 |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | UpdateItem REPLACE with per_timestep false |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\biosphere_leaf_energy_balance.md`.

### References

- Monteith, J.L. (1965). Evaporation and environment. Symp. Soc. Exp. Biol. 19, 205-224 (FunctorMetadata, certificate)
- Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56, Eqs. 3, 4 and 42 (FunctorMetadata)
- Jones, H.G. (2014). Plants and Microclimate, 3rd ed. Cambridge University Press, Chapter 9 (FunctorMetadata, certificate)
- Campbell, G.S. and Norman, J.M. (1998). An Introduction to Environmental Biophysics, 2nd ed. Springer, Chapter 14 (FunctorMetadata, certificate)
- Monteith, J.L. and Unsworth, M.H. (2013). Principles of Environmental Physics, 4th ed. Academic Press (FunctorMetadata, certificate)
- Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56, Eqs. 3, 4, 8, 11-13 and 42 (certificate)

(class-relic-scientific-biosphere-photosynthesis-stomatal-medlynstomatalconductance)=
## MedlynStomatalConductance

BiologicalProcess diagnostic, conditional on externally supplied assimilation.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.photosynthesis.stomatal.MedlynStomatalConductance` |
| Source | `relic\scientific\biosphere\photosynthesis\stomatal.py:25` |
| Runtime name | `medlyn_stomatal_conductance` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `observation` |
| Spatial scale | `patch` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MedlynStomatalConductance(name: str = 'medlyn_stomatal_conductance', config: dict | None = None, *, parameter_manager: Any = None, device: Any = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'medlyn_stomatal_conductance'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `parameter_manager` | `typing.Any` | `None` | Constructor control for parameter manager; see the class runtime contract and ParameterManager table below. |
| `device` | `typing.Any` | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

The runtime metadata below is introspected from the documented configuration `surface_conditions='leaf_surface_equals_canopy_air'`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `net_photosynthesis` | flux | `umol CO2 m-2 s-1` | required upstream StateManager port on ground support in umol CO2 m-2 s-1 |
| `leaf_area_index` | state | `m2 m-2` | required StateManager port in m2 m-2 |
| `ppfd` | flux | `umol photons m-2 s-1` | required StateManager port in umol m-2 s-1; selects only the light branch |
| `atmospheric_co2_ppm` | forcing | `umol mol-1` | required StateManager port in umol mol-1, substituted for leaf-surface CO2 by the declared condition |
| `vapor_pressure_deficit` | forcing | `Pa` | required StateManager port in Pa, substituted for leaf-surface VPD by the declared condition |
| `air_temperature` | forcing | `K` | air temperature of the canopy conductance conversion g_c = g_s L_eff R T / P, read only with canopy_conductance_upscaling |
| `atmospheric_pressure` | forcing | `Pa` | air pressure of the canopy conductance conversion g_c = g_s L_eff R T / P, read only with canopy_conductance_upscaling |
| `air_temperature_max` | forcing | `K` | daily maximum air temperature, read only with temporal_support daylight_gauss |
| `air_temperature_min` | forcing | `K` | daily minimum air temperature, read only with temporal_support daylight_gauss |
| `latitude` | forcing | `degrees_north` | latitude of the cell, read only with temporal_support daylight_gauss |
| `day_of_year` | forcing | `1` | day of year, read only with temporal_support daylight_gauss |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `stomatal_conductance` | forcing | `mol m-2 s-1` | decided at run time by the returned write mode | canonical nonnegative leaf-support output, instantaneous REPLACE |
| `diagnostics.medlyn.leaf_net_assimilation` | output port | `umol m-2 s-1` | diagnostics | signed representative leaf assimilation, preserved without clipping |
| `diagnostics.medlyn.dark_residual` | output port | `1` | diagnostics | boolean selection of the dark residual branch g_s = g0 |
| `diagnostics.medlyn.nonpositive_assimilation_residual` | output port | `1` | diagnostics | boolean selection of the illuminated nonpositive-assimilation residual branch g_s = g0 |
| `diagnostics.medlyn.vpd_floor_active` | output port | `1` | diagnostics | boolean activation of D_eff = D_min |
| `canopy_conductance` | state | `m s-1` | decided at run time by the returned write mode | bulk canopy conductance g_c = g_s L_eff R T / P, written only with canopy_conductance_upscaling |
| `diagnostics.medlyn.no_canopy` | output port | `1` | diagnostics | boolean LAI <= 0 of nonpositive_leaf_area no_canopy (g_s = g_c = 0) |
| `stomatal_conductance_daylight_samples` | diagnostic | `mol m-2 s-1` | decided at run time by the returned write mode | stomatal conductance at the daylight Gauss samples, written only with temporal_support daylight_gauss |
| `canopy_conductance_daylight_samples` | diagnostic | `m s-1` | decided at run time by the returned write mode | canopy conductance at the daylight Gauss samples, written only with temporal_support daylight_gauss and canopy_conductance_upscaling |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Input declaration difference: `air_temperature`, `air_temperature_max`, `air_temperature_min`, `atmospheric_pressure`, `day_of_year`, `latitude`.
Output declaration difference: `canopy_conductance`, `canopy_conductance_daylight_samples`, `diagnostics.medlyn.no_canopy`, `stomatal_conductance_daylight_samples`.
:::

### Governing equations and canonical operator

**Literature or process form**

$$
g_{s} = \mathrm{g0} + 1.6 ( 1 + \mathrm{g1} / \sqrt{D_{s}} ) A_{n} / C_{s} ( \mathrm{Medlyn} \mathrm{et} \mathrm{al} . 2011 ; \mathrm{CTSM} \mathrm{technical} \mathrm{note} \mathrm{Eq} . 2.9 . 1 )
$$

**Canonical RELIC structural form**

$y_{h} = O_{h} x_{h}$ through the declared observation operator

| Entry | Meaning |
| --- | --- |
| `O_h` | declared conditional Medlyn observation map with explicit residual branches |
| `H_h` | diagnostic metric or identity Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | supplied net assimilation, LAI, PPFD, canopy-air CO2 and VPD observation inputs |
| `effort_variable` | not applicable to this empirical level-1 observation |
| `state_advancer` | $y_{h} = O_{h} x_{h}$ through the declared observation operator |
| `runtime_usage` | MedlynStomatalConductance.forward evaluates one conditional conductance and its branch diagnostics and emits stomatal_conductance without advancing any material pool. |

### Composition channels and executable PortSpec API

Certificate composition channels: `net_photosynthesis`, `leaf_area_index`, `ppfd`, `atmospheric_co2_ppm`, `vapor_pressure_deficit`, `stomatal_conductance`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `coupling.scales.biosphere_atmosphere.g0` | `mol m-2 s-1` | `0.01; bounds=(0.0, 0.1)` | `fixed` | Generic residual leaf conductance prior for the explicitly declared dark/nonpositive-assimilation branch; retained value is not a measured nighttime calibration or a claim of the CTSM intercept value. | Source: retained_generic_residual_prior_with_explicit_ctsm_style_branch_contract; Ball et al. (1987); CTSM PhotosynthesisMod.F90 at 7d1bc988a5a67660d2988c57a0bf195173c31017, lines 1856-1870 and 1927-1934: separate dark and negative-assimilation conductance treatment; its maximum-resistance cap is not adopted here. |
| `coupling.scales.biosphere_atmosphere.g1` | `kPa^0.5` | `4.0; bounds=(0.0, 20.0)` | `fixed` | Generic Medlyn slope prior multiplying inverse square root leaf-surface VPD in kPa. Value 4.0 is retained; it is not the dimensionless Ball-Berry PFT g1 or the independent P3 comparator value 4.45. | Source: retained_generic_prior_with_corrected_medlyn_dimension; Medlyn et al. (2011), Global Change Biology 17, 2134-2144, doi:10.1111/j.1365-2486.2010.02375.x; modern implemented equation verified against CTSM technical note Eq. 2.9.1.; CTSM Photosynthesis technical note, Eq. 2.9.1: D_s in kPa, requiring g1 in kPa^0.5; https://escomp.github.io/CTSM/tech_note/Photosynthesis/CLM50_Tech_Note_Photosynthesis.html |
| `coupling.scales.biosphere_atmosphere.vpd_min_pa` | `Pa` | `50.0; bounds=(50.0, 50.0)` | `fixed` | Declared CTSM-style VPD regularization: D_eff=max(D_surface,50 Pa), equivalently 0.05 kPa. This is a fixed implementation choice, not an original Medlyn equation term, observation filling, or a numerical acceptance tolerance. | Source: explicit_ctsm_style_regularization_selected_20261015; CTSM PhotosynthesisMod.F90 at 7d1bc988a5a67660d2988c57a0bf195173c31017, lines 87-88 and 1890: max(esat_tv-ceair,50 Pa)*0.001. |
| `biosphere.photosynthesis.canopy_light_extinction` | `dimensionless` | `0.5; bounds=(0.2, 0.9)` | `fixed` | Beer-Lambert canopy light extinction coefficient for LAI scaling in canopy photosynthesis | Source: literature_synthesis; Monsi and Saeki (1953) Japanese Journal of Botany 14:22-52; Monteith and Unsworth (2013) Principles of Environmental Physics |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Physical constants (SSOT)

These values are resolved from the physical-constants database. They are not domain parameters and must not be duplicated in scientific YAML or Python code.

| Constant key | Units | Value and fixed bounds | Gradient | Meaning | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `relic.physical.constants.conversions.h2o_co2_diffusivity_ratio` | `dimensionless` | `1.6; bounds=(1.4, 1.8)` | `fixed` | Diffusivity ratio of water vapor to CO2 in air | Source: boundary_layer_theory; Monteith & Unsworth (2008) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | UpdateItem REPLACE with per_timestep false |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\photosynthesis.md`.

### References

- Medlyn et al. (2011), doi:10.1111/j.1365-2486.2010.02375.x. (FunctorMetadata)
- CTSM Photosynthesis technical note, Eq. 2.9.1. (FunctorMetadata)
- CTSM PhotosynthesisMod.F90, commit 7d1bc988a5a67660d2988c57a0bf195173c31017. (FunctorMetadata)
- de Pury and Farquhar (1997), Plant Cell Environ. 20, 537-557. (FunctorMetadata)
- Medlyn, B.E. et al. (2011). Global Change Biology 17, 2134-2144, doi:10.1111/j.1365-2486.2010.02375.x (certificate)
- CTSM Photosynthesis technical note, Eq. 2.9.1 (certificate)
- CTSM PhotosynthesisMod.F90 at commit 7d1bc988a5a67660d2988c57a0bf195173c31017, lines 87-88 and 1856-1934 (certificate)
- de Pury, D.G.G. and Farquhar, G.D. (1997). Plant, Cell and Environment 20, 537-557 (certificate)

(class-relic-scientific-biosphere-respiration-autotrophic-respiration-autotrophicrespiration)=
## AutotrophicRespiration

Unified Autotrophic Respiration Model.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.respiration.autotrophic_respiration.AutotrophicRespiration` |
| Source | `relic\scientific\biosphere\respiration\autotrophic_respiration.py:23` |
| Runtime name | `autotrophic_respiration` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
AutotrophicRespiration(name: str = 'autotrophic_respiration', device: torch.device | None = None, pft_name: str | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'autotrophic_respiration'` | Functor identity used by registration, provenance, and dependency resolution. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `pft_name` | <code>str &#124; None</code> | `None` | Optional plant functional type namespace used for PFT parameter overrides. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `gross_primary_productivity` | flux | `kg C m-2 s-1` | scheduler state input mapped to the carbon-boundary cochain |
| `maintenance_respiration_total` | flux | `kg C m-2 s-1` | scheduler-produced maintenance flux mapped to r_h[1] |
| `growth_respiration_total` | flux | `kg C m-2 s-1` | scheduler-produced growth flux mapped to r_h[2] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `total_autotrophic_respiration` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | canonical aggregate reaction output |
| `net_primary_productivity` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | unique canonical NPP output computed as GPP minus total autotrophic respiration |
| `diagnostics.autotrophic_respiration.maintenance_component` | output port | `kg C m-2 s-1` | diagnostics | exact observation of the scheduled maintenance input |
| `diagnostics.autotrophic_respiration.growth_component` | output port | `kg C m-2 s-1` | diagnostics | exact observation of the scheduled growth input |
| `diagnostics.autotrophic_respiration.maintenance_fraction` | output port | `1` | diagnostics | maintenance share of total autotrophic respiration |
| `diagnostics.autotrophic_respiration.growth_fraction` | output port | `1` | diagnostics | growth share of total autotrophic respiration |
| `diagnostics.autotrophic_respiration.carbon_balance_residual` | output port | `kg C m-2 s-1` | diagnostics | exact GPP minus respiration minus NPP ledger residual |

### Governing equations and canonical operator

**Literature or process form**

$$
R_{\mathrm{auto}} = R_{\mathrm{maintenance}} + R_{\mathrm{growth}}; \mathrm{NPP} = \mathrm{GPP} - R_{\mathrm{auto}}
$$

**Canonical RELIC structural form**

$\dot{x} = N_{h} r_{h}$ with scheduler-provided carbon-port rates $r_{h}$ and declared stoichiometric/output map $N_{h}$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | scheduler state-port vector [GPP, $R_{m}$, $R_{g}$] evaluated by the production forward path; the diagnostic epsilon is not a rate |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | no thermodynamic effort is claimed at level 2; gross_primary_productivity is the declared carbon input port |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with $r_{h} = [ \mathrm{GPP} , R_{m} , R_{g} ]$ and rows for $R_{a}$ and NPP |
| `runtime_usage` | AutotrophicRespiration.forward reads all three scheduler state inputs, then evaluates total respiration and NPP through advance_reaction_network. |
| `explicit_stoichiometric_usage` | $N_{h}$ rows are $[ 0 , 1 , 1 ]$ for $R_{a}$ and $[ 1 , - 1 , - 1 ]$ for NPP. |

### Composition channels and executable PortSpec API

Certificate composition channels: `maintenance_respiration_total`, `growth_respiration_total`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.validation.epsilon.flux` | `kg C m-2 s-1` | `1e-15; bounds=(1e-18, 1e-10)` | `fixed` | Canonical kg-C threshold used only for near-zero respiration fraction diagnostics | Source: RELIC parameter database; IEEE 754 numerical threshold for diagnostic flux ratios |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | all outputs use UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} )$ with explicit units and writer provenance; non-incrementable fluxes use $\mathrm{element} = \mathrm{None} \mathrm{because} \mathrm{their} \mathrm{SCM} \mathrm{contracts} \mathrm{declare} \mathrm{accumulates} = \mathrm{null}$ |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\respiration.md`.

### References

- Amthor, J.S. (2000). Annals of Botany 86, 1-20. (FunctorMetadata)
- Cannell, M.G.R. and Thornley, J.H.M. (2000). Annals of Botany 85, 45-54. (FunctorMetadata)
- Gifford, R.M. (2003). Functional Plant Biology 30, 171-186. (FunctorMetadata)
- Amthor, J.S. (2000). The McCree-de Wit-Penning de Vries-Thornley respiration paradigms. Annals of Botany 86, 1-20. (certificate)
- Cannell, M.G.R. and Thornley, J.H.M. (2000). Modelling the Components of Plant Respiration. Annals of Botany 85, 45-54. (certificate)
- Gifford, R.M. (2003). Plant respiration in productivity models. Functional Plant Biology 30, 171-186. (certificate)

(class-relic-scientific-biosphere-respiration-growth-respiration-growthrespiration)=
## GrowthRespiration

Growth Respiration Model following CASA framework.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.respiration.growth_respiration.GrowthRespiration` |
| Source | `relic\scientific\biosphere\respiration\growth_respiration.py:39` |
| Runtime name | `growth_respiration` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
GrowthRespiration(name: str = 'growth_respiration', config: Dict[str, Any] | None = None, device: torch.device | None = None, pft_name: str | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'growth_respiration'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `pft_name` | <code>str &#124; None</code> | `None` | Optional plant functional type namespace used for PFT parameter overrides. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `gross_primary_productivity` | flux | `kg C m-2 s-1` | gross_primary_productivity mapped to canonical Phase 8 cochain basis |
| `maintenance_respiration_total` | flux | `kg C m-2 s-1` | maintenance_respiration_total mapped to canonical Phase 8 cochain basis |
| `air_temperature` | forcing | `K` | canonical atmospheric temperature driver in K |
| `plant_nitrogen` | forcing | `g N m-2` | plant_nitrogen mapped to canonical Phase 8 cochain basis |
| `plant_phosphorus` | forcing | `g P m-2` | plant_phosphorus mapped to canonical Phase 8 cochain basis |
| `soluble_sugar` | forcing | `g C m-2` | canonical substrate pool in g C m-2 |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `growth_respiration_total` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.growth_respiration.growth_efficiency` | output port | `1` | diagnostics | bounded construction efficiency observation |
| `diagnostics.growth_respiration.available_carbon` | output port | `kg C m-2 s-1` | diagnostics | carbon remaining after maintenance respiration |
| `diagnostics.growth_respiration.np_ratio_effect` | output port | `1` | diagnostics | nutrient-ratio response observation |
| `diagnostics.growth_respiration.temperature_factor` | output port | `1` | diagnostics | Q10 response observation |
| `diagnostics.growth_respiration.substrate_factor` | output port | `1` | diagnostics | substrate response observation |
| `diagnostics.growth_respiration.carbon_limit_residual` | output port | `kg C m-2 s-1` | diagnostics | unrealized potential created by the explicit carbon availability bound |

### Governing equations and canonical operator

**Literature or process form**

$$
R_{g,\mathrm{potential}} = ( 1 - \mathrm{Ygrow} ) \cdot \max ( 0 , \mathrm{GPP} - R_{m} ) \cdot f_{T} \cdot f_{S}; R_{g} = \min ( R_{g,\mathrm{potential}} , \max ( 0 , \mathrm{GPP} - R_{m} ) )
$$

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | GrowthRespiration.forward calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical outputs. |
| `explicit_stoichiometric_usage` | GrowthRespiration.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `farquhar_photosynthesis`, `maintenance_respiration`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.respiration.growth.ygrow_base` | `dimensionless` | `0.65; bounds=(0.4, 0.75)` | `fixed` | Base growth efficiency (CASA framework) | Source: casa_cnp_model; CASA-CNP model documentation |
| `biosphere.respiration.growth.np_response_coeff` | `dimensionless` | `0.2; bounds=(0.1, 0.3)` | `fixed` | N:P ratio response coefficient | Source: casa_cnp_model; CASA-CNP model documentation |
| `biosphere.respiration.growth.np_half_sat` | `dimensionless` | `0.0667; bounds=(0.05, 0.1)` | `fixed` | N:P ratio half-saturation constant (1/15) | Source: casa_cnp_model; CASA-CNP model documentation |
| `biosphere.respiration.growth.q10_growth` | `dimensionless` | `2.0; bounds=(1.5, 3.0)` | `fixed` | Q10 temperature coefficient for growth respiration | Source: literature_synthesis; Lloyd & Taylor (1994) |
| `biosphere.respiration.growth.km_substrate_growth` | `g C m-2` | `5.0; bounds=(2.0, 10.0)` | `fixed` | Half-saturation constant for substrate limitation in growth | Source: literature_synthesis; Ryan (1991) |
| `biosphere.respiration.growth.min_growth_efficiency` | `dimensionless` | `0.4; bounds=(0.3, 0.5)` | `fixed` | Minimum growth efficiency bound | Source: casa_cnp_model; CASA-CNP model documentation |
| `biosphere.respiration.growth.max_growth_efficiency` | `dimensionless` | `0.75; bounds=(0.7, 0.85)` | `fixed` | Maximum growth efficiency bound | Source: casa_cnp_model; CASA-CNP model documentation |
| `biosphere.validation.epsilon.nutrient` | `g P m-2` | `0.0001; bounds=(1e-06, 0.005)` | `fixed` | Small phosphorus floor used to avoid division-by-zero in growth respiration N:P ratios | Source: respiration_framework_refactor; Penning de Vries et al. (1974) Journal of Theoretical Biology 45:339-377; Poorter et al. (1991) Plant, Cell & Environment 14:963-970 |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | all outputs use UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} , \mathrm{element} = \mathrm{None} )$ with explicit units and writer provenance; the non-incrementable growth flux has accumulates null in its SCM contract |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\respiration.md`.

### References

- Gifford, R.M. (2003). Plant respiration in productivity models (FunctorMetadata, certificate)
- CASA-CNP model documentation (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-respiration-maintenance-respiration-maintenancerespiration)=
## MaintenanceRespiration

Maintenance Respiration Model with Q10 response.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.respiration.maintenance_respiration.MaintenanceRespiration` |
| Source | `relic\scientific\biosphere\respiration\maintenance_respiration.py:56` |
| Runtime name | `maintenance_respiration` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MaintenanceRespiration(name: str = 'maintenance_respiration', config: Dict[str, Any] | None = None, device: torch.device | None = None, enable_acclimation: bool = True, acclimation_days: int = 10, pft_name: str | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'maintenance_respiration'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `enable_acclimation` | `bool` | `True` | Enables or disables the acclimation pathway. |
| `acclimation_days` | `int` | `10` | Constructor control for acclimation days; see the class runtime contract and ParameterManager table below. |
| `pft_name` | <code>str &#124; None</code> | `None` | Optional plant functional type namespace used for PFT parameter overrides. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | air_temperature mapped to canonical Phase 8 cochain basis |
| `biomass_leaf` | forcing | `g C m-2` | biomass_leaf mapped to canonical Phase 8 cochain basis |
| `biomass_stem` | forcing | `g C m-2` | biomass_stem mapped to canonical Phase 8 cochain basis |
| `biomass_root` | forcing | `g C m-2` | biomass_root mapped to canonical Phase 8 cochain basis |
| `soluble_sugar` | forcing | `g C m-2` | soluble_sugar mapped to canonical Phase 8 cochain basis |
| `soil_water_potential` | diagnostic | `MPa` | soil_water_potential mapped to canonical Phase 8 cochain basis |
| `growing_temperature` | forcing | `K` | optional acclimation observation mapped to the canonical Phase 8 basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `maintenance_respiration_total` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.maintenance_respiration.leaf_flux` | output port | `kg C m-2 s-1` | diagnostics | leaf component observation of the total carbon export |
| `diagnostics.maintenance_respiration.stem_flux` | output port | `kg C m-2 s-1` | diagnostics | stem component observation of the total carbon export |
| `diagnostics.maintenance_respiration.root_flux` | output port | `kg C m-2 s-1` | diagnostics | root component observation of the total carbon export |
| `diagnostics.maintenance_respiration.leaf_rate_coefficient` | output port | `s-1` | diagnostics | leaf Q10-adjusted coefficient observation |
| `diagnostics.maintenance_respiration.substrate_factor` | output port | `1` | diagnostics | substrate limitation observation |
| `diagnostics.maintenance_respiration.water_stress_factor` | output port | `1` | diagnostics | water limitation observation |
| `diagnostics.maintenance_respiration.growth_temperature` | output port | `K` | diagnostics | acclimation-temperature observation |

### Governing equations and canonical operator

**Literature or process form**

$$
R ( T ) = R_{\mathrm{ref}} \cdot \mathrm{Q}_{10} ^ ( ( T - T_{\mathrm{ref}} ) / 10 )
$$

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | MaintenanceRespiration.forward calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical outputs. |
| `explicit_stoichiometric_usage` | MaintenanceRespiration.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.respiration.maintenance.rm_leaf_25` | `s-1` | `1.736e-07; bounds=(5.787e-08, 5.787e-07)` | `fixed` | Leaf maintenance respiration rate at 25 degrees C (converted from 0.015 day-1 using 1 day = 86400 s) | Source: literature_synthesis; Ryan (1991); Reich et al. (2008) |
| `biosphere.respiration.maintenance.rm_stem_25` | `s-1` | `1.736e-08; bounds=(1.157e-08, 2.315e-07)` | `fixed` | Stem maintenance respiration rate at 25 degrees C (converted from 0.0015 day-1 using 1 day = 86400 s) | Source: literature_synthesis; Ryan (1991) |
| `biosphere.respiration.maintenance.rm_root_25` | `s-1` | `1.157e-07; bounds=(3.472e-08, 3.472e-07)` | `fixed` | Root maintenance respiration rate at 25 degrees C (converted from 0.010 day-1 using 1 day = 86400 s) | Source: literature_synthesis; Ryan (1991) |
| `biosphere.respiration.maintenance.q10_respiration` | `dimensionless` | `2.0; bounds=(1.5, 3.5)` | `fixed` | Base Q10 temperature coefficient for respiration | Source: literature_synthesis; Lloyd & Taylor (1994) |
| `biosphere.respiration.maintenance.q10_leaf_factor` | `dimensionless` | `1.0; bounds=(0.8, 1.2)` | `fixed` | Leaf-specific Q10 multiplication factor | Source: literature_synthesis; Ryan (1991) |
| `biosphere.respiration.maintenance.q10_stem_factor` | `dimensionless` | `0.9; bounds=(0.7, 1.1)` | `fixed` | Stem-specific Q10 multiplication factor (lower for woody tissue) | Source: literature_synthesis; Ryan (1991) |
| `biosphere.respiration.maintenance.q10_root_factor` | `dimensionless` | `1.1; bounds=(0.9, 1.3)` | `fixed` | Root-specific Q10 multiplication factor | Source: literature_synthesis; Ryan (1991) |
| `biosphere.respiration.maintenance.acclimation_rate_coeff` | `K-1` | `0.05; bounds=(0.01, 0.15)` | `fixed` | Temperature acclimation coefficient | Source: literature_synthesis; Atkin & Tjoelker (2003) |
| `biosphere.respiration.maintenance.km_substrate` | `g C m-2` | `10.0; bounds=(5.0, 25.0)` | `fixed` | Half-saturation constant for substrate limitation | Source: literature_synthesis; Ryan (1991) |
| `biosphere.respiration.maintenance.water_stress_wilting_point` | `MPa` | `-1.5; bounds=(-3.0, -0.5)` | `fixed` | Soil water potential at permanent wilting point | Source: soil_physics_standard; Standard soil physics value |
| `biosphere.respiration.maintenance.water_stress_field_capacity` | `MPa` | `-0.1; bounds=(-0.5, 0.0)` | `fixed` | Soil water potential at field capacity | Source: soil_physics_standard; Standard soil physics value |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\respiration.md`.

### References

- Ryan, M.G. (1991). Effects of climate change on plant respiration (FunctorMetadata, certificate)
- Atkin & Tjoelker (2003). Thermal acclimation and dynamic response (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-stress-nutrient-stress-nutrientstress)=
## NutrientStress

Compute a calibratable N/P tissue-concentration control factor.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.stress.nutrient_stress.NutrientStress` |
| Source | `relic\scientific\biosphere\stress\nutrient_stress.py:50` |
| Runtime name | `nutrient_stress` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `observation` |
| Spatial scale | `patch` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
NutrientStress(name: str = 'nutrient_stress', config: Dict[str, Any] | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'nutrient_stress'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `nitrogen_concentration` | forcing | `kg kg-1` | required plant-tissue nitrogen mass fraction supplied through StateManager |
| `phosphorus_concentration` | forcing | `kg kg-1` | required plant-tissue phosphorus mass fraction supplied through StateManager |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `nutrient_stress_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | canonical bounded control output |
| `diagnostics.nutrient_stress.nitrogen_factor` | output port | `1` | diagnostics | namespaced component observation |
| `diagnostics.nutrient_stress.phosphorus_factor` | output port | `1` | diagnostics | namespaced component observation |

### Governing equations and canonical operator

**Literature or process form**

Greenwood et al. (1990) describe $c_{N,\mathrm{crit}} ( W ) = a W ^ - b$; Agren et al. (2012) treat Liebig limitation as a first-order approximation. Neither source establishes the implemented fixed-threshold power response as a universal law.

**Canonical RELIC structural form**

$y_{h} = O_{h} ( c_{N}, c_{P} ; \theta )$, a read-only phenomenological observation with no state advancement

| Entry | Meaning |
| --- | --- |
| `O_h` | declared observation map |
| `H_h` | diagnostic metric or identity Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | plant-tissue nitrogen and phosphorus mass-fraction observation inputs |
| `effort_variable` | not applicable to this empirical level-1 observation |
| `state_advancer` | none; $y_{h} = O_{h} ( c_{N}, c_{P} ; \theta )$ is read-only |
| `runtime_usage` | NutrientStress.forward evaluates one bounded phenomenological observation and emits an explicit control plus two namespaced diagnostics without advancing nutrient state. |

### Composition channels and executable PortSpec API

Certificate composition channels: `nitrogen_concentration`, `phosphorus_concentration`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.stress.nutrient.n_critical_concentration` | `kg N kg-1 DM` | `0.025; bounds=(0.015, 0.035)` | `fixed` | Calibratable critical nitrogen mass fraction in plant dry matter for the phenomenological stress diagnostic | Source: phenomenological_calibration_prior; Greenwood et al. (1990), critical nitrogen dilution context only |
| `biosphere.stress.nutrient.n_minimum_concentration` | `kg N kg-1 DM` | `0.005; bounds=(0.003, 0.008)` | `fixed` | Calibratable lower nitrogen mass-fraction threshold for the phenomenological stress diagnostic | Source: phenomenological_calibration_prior; Aerts and Chapin (2000), plant nutrient status context |
| `biosphere.stress.nutrient.p_critical_concentration` | `kg P kg-1 DM` | `0.003; bounds=(0.002, 0.005)` | `fixed` | Calibratable critical phosphorus mass fraction in plant dry matter for the phenomenological stress diagnostic | Source: phenomenological_calibration_prior; Aerts and Chapin (2000), plant phosphorus limitation context |
| `biosphere.stress.nutrient.p_minimum_concentration` | `kg P kg-1 DM` | `0.001; bounds=(0.0005, 0.002)` | `fixed` | Calibratable lower phosphorus mass-fraction threshold for the phenomenological stress diagnostic | Source: phenomenological_calibration_prior; Aerts and Chapin (2000), plant phosphorus limitation context |
| `biosphere.stress.nutrient.nutrient_stress_exponent` | `dimensionless` | `1.5; bounds=(1.0, 2.5)` | `fixed` | Calibratable dimensionless exponent in the phenomenological nutrient stress response | Source: phenomenological_calibration_prior; Agren et al. (2012), nutrient limitation context only |
| `biosphere.stress.global.factor_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Global minimum stress factor for clamping (applies to all stress types) | Source: numerical_implementation; Numerical constraint |
| `biosphere.stress.global.factor_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Global maximum stress factor for clamping (applies to all stress types) | Source: numerical_implementation; Numerical constraint |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | UpdateItem REPLACE with per_timestep false |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\stress.md`.

### References

- Greenwood, D.J. et al. (1990). Annals of Botany 66, 425-436 (FunctorMetadata, certificate)
- Agren, G.I. et al. (2012). New Phytologist 194, 953-960 (FunctorMetadata, certificate)
- Aerts, R. and Chapin III, F.S. (2000). Advances in Ecological Research 30, 1-67 (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-stress-ozone-stress-ozonestress)=
## OzoneStress

Ozone Stress Response Model for RELIC.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.stress.ozone_stress.OzoneStress` |
| Source | `relic\scientific\biosphere\stress\ozone_stress.py:80` |
| Runtime name | `ozone_stress` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `patch` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
OzoneStress(name: str = 'ozone_stress', config: Dict[str, Any] | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'ozone_stress'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `o3_mixing_ratio` | state | `mol mol-1` | atmospheric ozone mole fraction (CF mole_fraction_of_ozone_in_air, written by atmospheric_chemistry) mapped to the ozone exposure cochain input. |
| `stomatal_conductance` | forcing | `mol m-2 s-1` | leaf-level stomatal conductance to water vapour [mol m-2 s-1], a caller-supplied forcing (no relic Functor writes it; FarquharPhotosynthesis outputs no conductance), converted to the O3 conductance g_s / k_O3 for the stomatal uptake reaction rate. |
| `latitude` | forcing | `degrees_north` | latitude of the cell, read only with temporal_support daylight_gauss |
| `day_of_year` | forcing | `1` | day of year, read only with temporal_support daylight_gauss |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `ozone_stress_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | same public stress multiplier preserved between legacy and IPHS runtime contracts. |
| `ozone_uptake` | flux | `mol O3 m-2 s-1` | decided at run time by the returned write mode | first row of N_h r_h. |
| `antioxidant_depletion` | diagnostic | `s-1` | decided at run time by the returned write mode | specific antioxidant consumption rate D / C_ax [s-1] of the detoxification reaction row (D = min(F, C_detox)); no dimensionless clamp. |
| `diagnostics.ozone_stress.detoxification_flux` | output port | `mol O3 m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.ozone_stress.excess_ozone_flux` | output port | `mol O3 m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.ozone_stress.antioxidant_pool_tendency` | output port | `mol O3 m-2 s-1` | diagnostics | antioxidant defense sink row of N_h r_h. |
| `diagnostics.ozone_stress.damage_tendency` | output port | `mol O3 m-2 s-1` | diagnostics | unquenched ozone damage row of N_h r_h. |
| `diagnostics.ozone_stress.port_power` | output port | `mol O3 m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.ozone_stress.entropy_production` | output port | `mol O3 m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `ozone_stress_factor_daylight_samples` | diagnostic | `dimensionless` | decided at run time by the returned write mode | ozone stress factor at the daylight Gauss samples, written only with temporal_support daylight_gauss |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Input declaration difference: `day_of_year`, `latitude`.
Output declaration difference: `ozone_stress_factor_daylight_samples`.
:::

### Governing equations and canonical operator

**Literature or process form**

$F_{\mathrm{O3}} = ( g_{s} / k_{\mathrm{O3}} ) \cdot [ \mathrm{O}_{3} ] \cdot \mathrm{ozone}_{\mathrm{uptake},\mathrm{rate}}$, $\mathrm{the} R_{a} \rightarrow 0 \mathrm{limit} \mathrm{of} \mathrm{the} \mathrm{Sitch} \mathrm{et} \mathrm{al} . ( 2007 ) \mathrm{leaf} \mathrm{uptake} [ \mathrm{O}_{3} ] / ( R_{a} + k_{\mathrm{O3}} / g_{l} )$ with $g_{s}$ the leaf-level stomatal conductance to water vapour and $k_{\mathrm{O3}} = 1.67$; $f_{\mathrm{O3}} = 1 - \min ( F_{\mathrm{O3}} / C_{\mathrm{detox}} , r_{\mathrm{max}} )$

**Canonical RELIC structural form**

$\dot{x} = N_{h} r_{h}$ with $r_{h} = [ F_{\mathrm{O3}} , \min ( F_{\mathrm{O3}} , C_{\mathrm{detox}} ) , \max ( F_{\mathrm{O3}} - C_{\mathrm{detox}} , 0 ) ]$ and open port routing.

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric/output incidence matrix in OzoneStress.forward. |
| `r_h` | ParameterManager-backed reaction-rate vector [$F_{\mathrm{O3}}$, detoxification_flux, excess_ozone_flux]. |
| `G_h` | open atmospheric ozone uptake port declared through `open_system_route='port'`. |
| `y_h` | ozone exposure effort $[ \mathrm{O}_{3} ] / [ \mathrm{O}_{3} ] _ \mathrm{threshold}$ used for port-power audit. |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | ozone uptake, detoxification, excess flux, antioxidant-pool tendency, and damage tendency cochain rows. |
| `effort_variable` | ozone exposure effort $[ \mathrm{O}_{3} ] / [ \mathrm{O}_{3} ] _ \mathrm{threshold}$. |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with $r_{h} = [ F_{\mathrm{O3}} , \mathrm{detoxification}_{\mathrm{flux}} , \mathrm{excess}_{\mathrm{ozone},\mathrm{flux}} ] .$ |
| `explicit_stoichiometric_usage` | OzoneStress.forward declares $\mathrm{stoichiometric}_{N} = . . .$ and $\mathrm{rates} = . . . \mathrm{in} \mathrm{advance}_{\mathrm{reaction},\mathrm{network}} .$ |
| `runtime_usage` | OzoneStress.forward calls advance_reaction_network and maps species_rate rows to public outputs and diagnostics. |

### Composition channels and executable PortSpec API

Certificate composition channels: `atmospheric_chemistry`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.stress.ozone.ozone_threshold` | `mol mol-1` | `4e-08; bounds=(2e-08, 8e-08)` | `fixed` | Ozone concentration threshold for stress calculations | Source: literature_synthesis; Mills et al. (2007) |
| `biosphere.stress.ozone.flux_threshold` | `nmol O3 m-2 s-1` | `1.6; bounds=(1.0, 2.5)` | `fixed` | Stomatal flux threshold for POD calculations | Source: literature_synthesis; Mills et al. (2011) |
| `biosphere.stress.ozone.ozone_uptake_rate` | `dimensionless` | `0.8; bounds=(0.6, 1.0)` | `fixed` | Fraction of ozone flux that is actually taken up | Source: literature_synthesis; Emberson et al. (2000) |
| `biosphere.stress.ozone.ozone_detox_capacity` | `mol O3 m-2 s-1` | `2e-09; bounds=(1e-09, 5e-09)` | `fixed` | Plant ozone detoxification capacity | Source: literature_synthesis; Emberson et al. (2000) |
| `biosphere.stress.ozone.antioxidant_capacity` | `mol O3 m-2` | `1e-08; bounds=(5e-09, 2e-08)` | `fixed` | Plant antioxidant capacity for ozone defense | Source: literature_synthesis; Kangasjarvi et al. (2005) |
| `biosphere.stress.ozone.max_flux_ratio` | `dimensionless` | `0.9; bounds=(0.5, 1.0)` | `learnable` | Maximum normalized ozone flux ratio used when converting uptake to stress factor | Source: literature_synthesis; Emberson et al. (2000); Mills et al. (2011) |
| `biosphere.stress.global.factor_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Global minimum stress factor for clamping (applies to all stress types) | Source: numerical_implementation; Numerical constraint |
| `biosphere.stress.global.factor_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Global maximum stress factor for clamping (applies to all stress types) | Source: numerical_implementation; Numerical constraint |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Physical constants (SSOT)

These values are resolved from the physical-constants database. They are not domain parameters and must not be duplicated in scientific YAML or Python code.

| Constant key | Units | Value and fixed bounds | Gradient | Meaning | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `relic.physical.constants.conversions.h2o_o3_diffusivity_ratio` | `dimensionless` | `1.67; bounds=(1.67, 1.67)` | `fixed` | Diffusivity ratio of water vapor to ozone in air: the ratio k_O3 of the leaf (stomatal) resistance for O3 to that for water vapor, so that the stomatal conductance for O3 is g_s,O3 = g_s,H2O / k_O3 | Source: literature; Sitch, S., Cox, P.M., Collins, W.J. and Huntingford, C. (2007). Indirect radiative forcing of climate change through ozone effects on the land-carbon sink. Nature 448(7155), 791-794. doi:10.1038/nature06059; Leung, F., Williams, K., Sitch, S. et al. (2020). Calibrating soybean parameters in JULES 5.0 from the US-Ne2/3 FLUXNET sites and the SoyFACE-O3 experiment. Geosci. Model Dev. 13, 6201-6213, Eq. (5). doi:10.5194/gmd-13-6201-2020 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\stress.md`.

### References

- Fuhrer, J. et al. (1997). Environmental Pollution 97, 91-106 (FunctorMetadata, certificate)
- Mills, G. et al. (2011). Atmospheric Environment 45, 1703-1713 (FunctorMetadata, certificate)
- Emberson, L.D. et al. (2000). Environmental Pollution 109, 403-413 (FunctorMetadata, certificate)
- Sitch, S. et al. (2007). Nature 448, 791-794 (FunctorMetadata, certificate)
- Leung, F. et al. (2020). Geosci. Model Dev. 13, 6201-6213, Eq. (5) (FunctorMetadata, certificate)
- Kangasjarvi, J. et al. (2005). Journal of Experimental Botany 56, 195-209 (certificate)
- Grmela and Ottinger (1997). Physical Review E 56, 6620-6632 (certificate)

(class-relic-scientific-biosphere-stress-thermal-stress-thermalstress)=
## ThermalStress

Thermal Stress Response Model for RELIC.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.stress.thermal_stress.ThermalStress` |
| Source | `relic\scientific\biosphere\stress\thermal_stress.py:51` |
| Runtime name | `thermal_stress` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `observation` |
| Spatial scale | `patch` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
ThermalStress(name: str = 'thermal_stress', config: Dict[str, Any] | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'thermal_stress'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | near-surface air temperature [K], the single temperature driver of the thermal factors |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `temperature_stress_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `heat_stress_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `cold_stress_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |

### Governing equations and canonical operator

**Literature or process form**

$$
f_{\mathrm{temp}} = \exp ( - 0.5 \cdot ( ( T - T_{\mathrm{opt}} ) / \sigma ) ^ 2 )
$$

**Canonical RELIC structural form**

$y_{h} = O_{h} x_{h}$ through the declared observation operator

| Entry | Meaning |
| --- | --- |
| `O_h` | declared observation map |
| `H_h` | diagnostic metric or identity Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $y_{h} = O_{h} x_{h}$ through the declared observation operator |
| `runtime_usage` | ThermalStress.forward applies the canonical observation or diagnostic operator and emits validation diagnostics through the production output path. |

### Composition channels and executable PortSpec API

Certificate composition channels: `air_temperature`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.stress.thermal.optimal_temperature` | `K` | `298.15; bounds=(293.15, 303.15)` | `fixed` | Optimal temperature for plant processes (25 degrees C) | Source: literature_synthesis; Berry & Bjorkman (1980); Salvucci & Crafts-Brandner (2004) |
| `biosphere.stress.thermal.heat_stress_threshold` | `K` | `308.15; bounds=(303.15, 313.15)` | `fixed` | Temperature threshold for heat stress (35 degrees C) | Source: literature_synthesis; Hatfield & Prueger (2015) |
| `biosphere.stress.thermal.critical_high_temperature` | `K` | `318.15; bounds=(313.15, 323.15)` | `fixed` | Critical high temperature causing severe damage (45 degrees C) | Source: literature_synthesis; Wahid et al. (2007) |
| `biosphere.stress.thermal.cold_stress_threshold` | `K` | `278.15; bounds=(273.15, 283.15)` | `fixed` | Temperature threshold for cold stress (5 degrees C) | Source: literature_synthesis; Chinnusamy et al. (2007) |
| `biosphere.stress.thermal.freezing_damage_threshold` | `K` | `271.15; bounds=(268.15, 273.15)` | `fixed` | Temperature causing freezing damage (-2 degrees C) | Source: literature_synthesis; Pearce (2001) |
| `biosphere.stress.thermal.temperature_tolerance_width` | `K` | `10.0; bounds=(5.0, 20.0)` | `fixed` | Temperature tolerance width for Gaussian response function | Source: literature_synthesis; Berry & Bjorkman (1980) |
| `biosphere.stress.thermal.heat_stress_coefficient` | `K-1` | `-0.1; bounds=(-0.2, -0.05)` | `fixed` | Coefficient for heat stress response function | Source: literature_synthesis; Hatfield & Prueger (2015) |
| `biosphere.stress.thermal.cold_stress_coefficient` | `K-1` | `-0.15; bounds=(-0.3, -0.08)` | `fixed` | Coefficient for cold stress response function | Source: literature_synthesis; Chinnusamy et al. (2007) |
| `biosphere.stress.global.factor_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Global minimum stress factor for clamping (applies to all stress types) | Source: numerical_implementation; Numerical constraint |
| `biosphere.stress.global.factor_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Global maximum stress factor for clamping (applies to all stress types) | Source: numerical_implementation; Numerical constraint |
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
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\stress.md`.

### References

- Berry, J. & Bjorkman, O. (1980). Annual Review of Plant Physiology 31, 491-543 (FunctorMetadata, certificate)
- Salvucci, M.E. & Crafts-Brandner, S.J. (2004). Plant, Cell & Environment 27, 1063-1071 (FunctorMetadata, certificate)
- Hatfield, J.L. & Prueger, J.H. (2015). Weather and Climate Extremes 10, 4-10 (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-stress-water-stress-waterstress)=
## WaterStress

Water Stress Response Model for RELIC.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.stress.water_stress.WaterStress` |
| Source | `relic\scientific\biosphere\stress\water_stress.py:43` |
| Runtime name | `water_stress` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `observation` |
| Spatial scale | `patch` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
WaterStress(name: str = 'water_stress', config: Dict[str, Any] | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'water_stress'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_water_potential` | diagnostic | `MPa` | required external or upstream StateManager port in MPa |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `water_stress_factor` | diagnostic | `dimensionless` | decided at run time by the returned write mode | canonical bounded output from the observation map |

### Governing equations and canonical operator

**Literature or process form**

$$
f_{water} = (psi - psi_wilt) / (psi_{fc} - psi_{wilt})
$$

**Canonical RELIC structural form**

$y_{h} = O_{h} x_{h}$ through the declared observation operator

| Entry | Meaning |
| --- | --- |
| `O_h` | declared observation map |
| `H_h` | diagnostic metric or identity Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | soil_water_potential observation input in MPa |
| `effort_variable` | not applicable to this empirical level-1 observation |
| `state_advancer` | $y_{h} = O_{h} x_{h}$ through the declared observation operator |
| `runtime_usage` | WaterStress.forward evaluates one bounded observation and emits water_stress_factor without advancing water state. |

### Composition channels and executable PortSpec API

Certificate composition channels: `soil_water_potential`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.stress.water.wilting_point_potential` | `MPa` | `-1.5; bounds=(-2.0, -1.0)` | `fixed` | Water potential at permanent wilting point | Source: literature_synthesis; Boyer (1982); Hsiao (1973) |
| `biosphere.stress.water.field_capacity_potential` | `MPa` | `-0.033; bounds=(-0.1, -0.01)` | `fixed` | Water potential at field capacity | Source: literature_synthesis; Boyer (1982); Jones (2007) |
| `biosphere.stress.global.factor_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Global minimum stress factor for clamping (applies to all stress types) | Source: numerical_implementation; Numerical constraint |
| `biosphere.stress.global.factor_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Global maximum stress factor for clamping (applies to all stress types) | Source: numerical_implementation; Numerical constraint |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | UpdateItem REPLACE with per_timestep false |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\stress.md`.

### References

- Boyer, J.S. (1982). Plant productivity and environment. Science 218, 443-448 (FunctorMetadata, certificate)
- Hsiao, T.C. (1973). Plant responses to water stress. Annual Review of Plant Physiology 24, 519-570 (FunctorMetadata, certificate)
- Jones, H.G. (2007). Monitoring plant and soil water status. Journal of Experimental Botany 58, 155-173 (FunctorMetadata, certificate)

(class-relic-scientific-biosphere-surrogates-simple-photosynthesis-simplecanopyphotosynthesis)=
## SimpleCanopyPhotosynthesis

Simple canopy photosynthesis and respiration functor.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.biosphere.surrogates.simple_photosynthesis.SimpleCanopyPhotosynthesis` |
| Source | `relic\scientific\biosphere\surrogates\simple_photosynthesis.py:44` |
| Runtime name | `simple_photosynthesis` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
SimpleCanopyPhotosynthesis(name: str = 'simple_photosynthesis', config: Optional[Dict[str, Union[str, float]]] = None, device: Optional[torch.device] = None, **kwargs: Dict[str, Union[str, float]]) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'simple_photosynthesis'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Dict[str, Union[str, float]]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | `Optional[torch.device]` | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `kwargs` | `Dict[str, Union[str, float]]` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `ppfd` | flux | `umol photons m-2 s-1` | ppfd photon-flux input provided by atmospheric_radiation |
| `air_temperature` | forcing | `K` | air_temperature mapped to canonical Phase 8 cochain basis |
| `leaf_area_index` | state | `m2 m-2` | leaf_area_index mapped to canonical Phase 8 cochain basis |
| `soil_moisture` | state | `m3 m-3` | soil_moisture mapped to canonical Phase 8 cochain basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `gross_primary_productivity` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `ecosystem_respiration` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `net_ecosystem_exchange` | flux | `kg C m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.photosynthesis.water_stress` | output port | `1` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.photosynthesis.temperature_factor` | output port | `1` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |

### Governing equations and canonical operator

**Literature or process form**

Rectangular-hyperbola PPFD response and fixed-Q10 empirical respiration response, with positive-to-atmosphere $\mathrm{NEE} = \mathrm{Reco} - \mathrm{GPP}$

**Canonical RELIC structural form**

$y_{h} = N_{h} r_{h}$ with $r_{h} = [ \mathrm{GPP} , \mathrm{Reco} ]$ and $N_{h}$ rows [1,0], [0,1], [-1,1]

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC open carbon-flux port basis |
| `effort_variable` | not claimed for this level 2 empirical reaction wrapper |
| `state_advancer` | $y_{h} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | SimpleCanopyPhotosynthesis.forward calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical outputs. |
| `explicit_stoichiometric_usage` | SimpleCanopyPhotosynthesis.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `gross_primary_productivity`, `ecosystem_respiration`, `net_ecosystem_exchange`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `biosphere.photosynthesis.surrogate.max_gpp_rate` | `kg C m-2 s-1` | `3e-07; bounds=(1e-08, 1e-06)` | `fixed` | Maximum gross primary productivity of the surrogate canopy model per unit leaf area index at saturating light and optimum temperature and soil moisture (3.0e-7 kg C m-2 s-1 is about 25 umol CO2 m-2 s-1); the ground-area rate is this value times the light, temperature and soil moisture factors and the leaf area index; the upper bound equals validation.conservation.plausibility.gross_primary_productivity_max | Source: model_default; Monteith & Unsworth (2013) Principles of Environmental Physics; Baldocchi (2003) Global Change Biology 9:479-492 |
| `biosphere.photosynthesis.surrogate.ppfd_half_saturation` | `umol photons m-2 s-1` | `600.0; bounds=(100.0, 1500.0)` | `fixed` | Half-saturation PPFD for the surrogate rectangular-hyperbola light response | Source: model_default; Baly & Morgan (1934) Nature 133:414 |
| `biosphere.photosynthesis.surrogate.temperature_optimum` | `K` | `298.15; bounds=(273.15, 313.15)` | `fixed` | Optimal canopy temperature for surrogate photosynthesis response | Source: model_default; Monteith & Unsworth (2013) Principles of Environmental Physics |
| `biosphere.photosynthesis.surrogate.temperature_width` | `K` | `12.0; bounds=(5.0, 25.0)` | `fixed` | Gaussian temperature response width for surrogate canopy model | Source: model_default; Monteith & Unsworth (2013) Principles of Environmental Physics |
| `biosphere.photosynthesis.surrogate.base_respiration_rate` | `kg C m-2 s-1` | `4.8e-08; bounds=(1e-09, 1e-06)` | `fixed` | Ecosystem respiration rate of the surrogate module per unit leaf area index at the physical-constant reference temperature (4.8e-8 kg C m-2 s-1 is about 4 umol CO2 m-2 s-1); the ground-area rate is this value times the Q10 temperature factor and the leaf area index; the upper bound equals validation.conservation.plausibility.gross_primary_productivity_max | Source: model_default; Mahecha et al. (2010) Science 329:838-840; Baldocchi (2003) Global Change Biology 9:479-492 |
| `biosphere.photosynthesis.surrogate.respiration_q10` | `dimensionless` | `1.4; bounds=(1.0, 3.0)` | `fixed` | Q10 factor of the surrogate ecosystem respiration against air temperature (1.4 +/- 0.1 across 60 FLUXNET sites in Mahecha et al. 2010); the lower bound 1.0 is no temperature response and the upper bound admits apparent seasonal values | Source: literature; Mahecha et al. (2010) Science 329:838-840 |
| `biosphere.photosynthesis.surrogate.soil_moisture_optimum` | `m3 m-3` | `0.35; bounds=(0.1, 0.6)` | `fixed` | Optimal volumetric soil moisture for surrogate photosynthesis response | Source: model_default; Monteith & Unsworth (2013) Principles of Environmental Physics |
| `biosphere.photosynthesis.surrogate.soil_moisture_width` | `m3 m-3` | `0.15; bounds=(0.05, 0.4)` | `fixed` | Soil moisture response width for surrogate canopy water stress function | Source: model_default; Monteith & Unsworth (2013) Principles of Environmental Physics |
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
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Biosphere theory](../theory/biosphere.md).

Certificate theory owner: `docs\source\theory\surrogates.md`.

### References

- Baly, E.C.C., Morgan, L.B. (1934). Nature 133:414. (FunctorMetadata, certificate)
- Mahecha, M.D., et al. (2010). Science 329:838-840. (FunctorMetadata, certificate)
- Chapin, F.S., et al. (2006). Ecosystems 9:1041-1050. (FunctorMetadata, certificate)
- Baldocchi, D. (2003). Global Change Biology 9:479-492. (FunctorMetadata, certificate)
