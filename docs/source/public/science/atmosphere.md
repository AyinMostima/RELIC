# Atmosphere Scientific Classes

This reference covers all 13 certificate-backed scientific classes in `relic.scientific.atmosphere`: 13 executable concrete Functors, 0 fail-closed unavailable boundaries, and 0 abstract contracts. Each entry is generated from the live class, its IPHS certificate, State Contract Model, and ParameterManager metadata. Fail-closed unavailable classes are not instantiated during documentation generation.

The tables distinguish constructor controls, runtime variables, and ParameterManager scientific parameters. Values shown as package defaults belong to the documented RELIC architecture version and must not be copied as hardcoded constants into scientific code.

Certificate wildcard parameter groups (`*`, `<pft>`) are expanded to their exact current YAML entries. Every other declared parameter is matched to the parameter database by its exact full key; a declared name is never mapped to a key that merely ends with the same name.

Exact ParameterManager rows report the live YAML value, bounds, gradient policy, description, and provenance. A row whose value reads `not declared in the parameter database` is intentionally not assigned an invented default: the class metadata or certificate declares the quantity, with the meaning and units it states, but the parameter database has no key of that exact name.

IPHS levels are read from the sole certificate owner. Levels 1 to 3 describe compatibility, wrapping, or declared structure; they are not strict IPHS/GENERIC claims.

The Channel column of each runtime output contract applies `relic.core.state.types.partition_outputs`, the only runtime output channel classifier, to the declared output name. A name that the classifier reserves for diagnostics, such as `diagnostics.*`, goes to the diagnostics channel and is never committed as state. For any other name the channel is decided at run time by the write mode of the returned value: `StateWriteMode.DERIVED` routes the output to the diagnostics channel, and any other write mode sends it to the StateManager commit (a value without a write mode takes the run's semantic-mode rules or the StateManager schema). The generator does not run `forward()`, so it does not predict that write mode. The Kind column is the State Contract Model kind of the variable and does not set its channel.

(class-relic-scientific-atmosphere-chemistry-chemistryfunctor)=
## ChemistryFunctor

Reduced-form atmospheric chemistry functor using box model approach.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.chemistry.ChemistryFunctor` |
| Source | `relic\scientific\atmosphere\chemistry.py:92` |
| Runtime name | `atmospheric_chemistry` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `field` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
ChemistryFunctor(name: str = 'atmospheric_chemistry', *, physical_boundary: Literal['open_mechanism', 'closed_constant_volume'], config: Mapping[str, Any] | None = None, **kwargs)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'atmospheric_chemistry'` | Functor identity used by registration, provenance, and dependency resolution. |
| `physical_boundary` | `typing.Literal['open_mechanism', 'closed_constant_volume']` | `required` | Constructor control for physical boundary; see the class runtime contract and ParameterManager table below. |
| `config` | <code>typing.Mapping[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

The runtime metadata below is introspected from the documented configuration `physical_boundary='open_mechanism'`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | air_temperature mapped to canonical Phase 8 cochain basis |
| `atmospheric_pressure` | forcing | `Pa` | atmospheric_pressure p of the ideal-gas number density n = p / (k_B T) |
| `shortwave_down` | flux | `W m-2` | shortwave_down solar global irradiance G of J_NO2 = (1 + alpha)(B1 G + B2 G^2) (Trebs et al. 2009) through the radiation input port |
| `cos_zenith` | forcing | `1` | cos_zenith cos(chi) of the MCM ratio rho(chi) = J(O1D)/J(NO2) and of the direct-beam slant path of the spectral temperature factor, a required input |
| `no_mixing_ratio` | state | `mol mol-1` | no_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `no2_mixing_ratio` | state | `mol mol-1` | no2_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `o3_mixing_ratio` | state | `mol mol-1` | o3_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `isoprene_mixing_ratio` | state | `mol mol-1` | isoprene_mixing_ratio mapped to canonical Phase 8 cochain basis when present |
| `co_mixing_ratio` | forcing | `mol mol-1` | co_mixing_ratio [CO] of the OH loss k_OH+CO [CO], a required input |
| `ch4_mixing_ratio` | forcing | `mol mol-1` | ch4_mixing_ratio [CH4] of the OH loss k_OH+CH4 [CH4], a required input |
| `specific_humidity` | forcing | `kg kg-1` | specific_humidity q of the water-vapour mole fraction x_w = (q/M_w) / (q/M_w + (1-q)/M_d) in the O(1D) + H2O fraction f, a required input |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `no_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | no_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `no2_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | no2_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `o3_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | o3_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `isoprene_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | isoprene_mixing_ratio mapped to canonical Phase 8 cochain basis when present |
| `diagnostics.chemistry.photolysis_rate_no2` | output port | `s-1` | diagnostics | canonical photolysis-rate observation |
| `diagnostics.chemistry.isoprene_oxidation_rate` | output port | `mol mol-1 s-1` | diagnostics | canonical isoprene oxidation-rate observation |
| `diagnostics.chemistry.reaction_no_o3_rate` | output port | `mol mol-1 s-1` | diagnostics | canonical NO and ozone reaction-rate observation |
| `diagnostics.chemistry.photolysis_no2_rate` | output port | `mol mol-1 s-1` | diagnostics | canonical NO2 photolysis tendency observation |
| `diagnostics.chemistry.rate_constant_no_o3` | output port | `cm3 molecule-1 s-1` | diagnostics | canonical NO and ozone rate-coefficient observation |
| `diagnostics.chemistry.shortwave_radiation_used` | output port | `W m-2` | diagnostics | canonical resolved radiation observation |
| `diagnostics.chemistry.oh_concentration` | output port | `molecule cm-3` | diagnostics | canonical OH number-density observation |
| `diagnostics.chemistry.air_number_density` | output port | `molecule cm-3` | diagnostics | canonical air number-density observation |
| `diagnostics.chemistry.ro2_no_conversion_rate` | output port | `mol mol-1 s-1` | diagnostics | canonical peroxy radical NO-to-NO2 conversion rate observation |
| `diagnostics.chemistry.ro2_no_branching_fraction` | output port | `dimensionless` | diagnostics | canonical peroxy radical NO branching fraction observation |
| `diagnostics.chemistry.ox_net_chemical_tendency` | output port | `mol mol-1 s-1` | diagnostics | canonical net odd-oxygen chemical production tendency observation |
| `diagnostics.chemistry.carbon_oxidized_rate` | output port | `mol mol-1 s-1` | diagnostics | canonical reactive carbon oxidation export rate observation |

### Governing equations and canonical operator

**Literature or process form**

$\frac{\mathrm{d}\left[\mathrm{NO}\right]}{\mathrm{d}t} = J_{\mathrm{NO2}} \cdot [ \mathrm{NO}_{2} ] - k_{\mathrm{NO},\mathrm{O3}} \cdot [ \mathrm{NO} ] \cdot [ \mathrm{O}_{3} ]$; $\frac{\mathrm{d}\left[\mathrm{NO}_{2}\right]}{\mathrm{d}t} = k_{\mathrm{NO},\mathrm{O3}} \cdot [ \mathrm{NO} ] \cdot [ \mathrm{O}_{3} ] - J_{\mathrm{NO2}} \cdot [ \mathrm{NO}_{2} ]$; $\frac{\mathrm{d}\left[\mathrm{O}_{3}\right]}{\mathrm{d}t} = J_{\mathrm{NO2}} \cdot [ \mathrm{NO}_{2} ] - k_{\mathrm{NO},\mathrm{O3}} \cdot [ \mathrm{NO} ] \cdot [ \mathrm{O}_{3} ]$; $k ( T ) = k_{\mathrm{298}} \cdot \exp ( \mathrm{Ea} / R \cdot ( 1 / T_{\mathrm{ref}} - 1 / T ) )$, $T_{\mathrm{ref}} = 298.15 K$; $J_{\mathrm{NO2}} = ( 1 + \alpha ) \cdot ( \mathrm{B}_{1} \cdot G + \mathrm{B}_{2} \cdot G ^ 2 ) \cdot R_{\mathrm{NO2}} ( T , \theta )$, $G = \mathrm{SW} \mathrm{held}$ at $G_{v} = \mathrm{B}_{1} / ( 2 \lvert \mathrm{B}_{2} \rvert ) \mathrm{above} ( \mathrm{Trebs} \mathrm{et} \mathrm{al} . 2009 )$; $J_{\mathrm{O1D}} = ( 1 + \alpha ) \cdot ( \mathrm{B}_{1} \cdot G + \mathrm{B}_{2} \cdot G ^ 2 ) \cdot \rho ( \theta ) \cdot R_{\mathrm{O1D}} ( T , \theta )$, $\rho = ( \mathrm{l1} / \mathrm{l4} ) \cos ( \theta ) ^ ( \mathrm{m1} - \mathrm{m4} ) \exp ( - ( \mathrm{n1} - \mathrm{n4} ) / \cos ( \theta ) )$; $R_{X} = 1 ( \mathrm{mode} \mathrm{none} )$ or sum sigma_X phi_X F0 exp(-tau/cos(theta)) / same at $T_{\mathrm{ref}}$ (mode spectral_quantum_yield); [OH] = y f $J_{\mathrm{O1D}}$ [O3] / (k_OH_CO[CO] + k_OH_CH4[CH4] + k_OH_NO2[NO2] + k_OH_C5H8[C5H8]), $f = k_{\mathrm{H2O}} [ \mathrm{H2O} ] / ( k_{\mathrm{H2O}} [ \mathrm{H2O} ] + k_{\mathrm{N2}} [ \mathrm{N}_{2} ] + k_{\mathrm{O2}} [ \mathrm{O}_{2} ] )$; $\mathrm{BVOC} + \mathrm{OH} \rightarrow \mathrm{products} ( k_{\mathrm{BVOC},\mathrm{OH}} )$

**Canonical RELIC structural form**

$\dot{x} = N_{h} r_{h}$, with photolysis and kinetic rates as ParameterManager-backed reaction entries and $C_{\mathrm{NOx}} N_{h} = 0$

| Entry | Meaning |
| --- | --- |
| `J_h` | zero reversible operator; reversible_structure returns None |
| `M_h` | positive semidefinite diagonal field Hodge declared for level-3 structural audit, not a strict runtime state advancer |
| `G_h` | declared radiation photolysis port map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | radiation and species fields used by the declared level 3 structural closure; no strict variational effort is claimed |
| `state_advancer` | $\Delta x = \mathrm{d}t N_{h} r_{h}$ with canonical per-step UpdateItem accumulation, admitted only when x + Delta x is non-negative for every retained species |
| `explicit_stoichiometric_usage` | ChemistryFunctor.forward passes the four retained species, five named reactions, signed 4 by 5 $N_{h}$, ParameterManager-backed $r_{h}$, $C_{\mathrm{NOx}}$, dt, and boundary_export route to advance_reaction_network exactly once. |
| `runtime_usage` | ChemistryFunctor.forward evaluates one gas-phase reduced-kinetics path through advance_reaction_network, validates $C_{\mathrm{NOx}} N_{h} = 0$, and rejects an explicit Euler candidate when dt exceeds the exact current-state positivity bound min($x_{i}$ / -$f_{i}$) over negative tendencies. The open boundary route contains unretained O/O2/OH and oxidation products. This is not a general stiff-stability, accuracy, Hamiltonian, entropy, or GENERIC proof. Dry deposition remains exclusively owned by DryDepositionAggregator, and no alternate chemistry formula branch is executable. |

### Composition channels and executable PortSpec API

Certificate composition channels: `atmospheric_radiation_photolysis_port`

Executable `ports()` owner: `ChemistryFunctor`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `atmospheric_radiation_photolysis_port` | `no2_mixing_ratio [mol mol-1]` | `diagnostics.chemistry.photolysis_rate_no2 [s-1]` | `mol mol-1 s-1` | into_system; radiation |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `atmosphere.chemistry.chemistry_temperature_q10` | `dimensionless` | `1.5; bounds=(1.0, 3.0)` | `fixed` | Multiplier of the maximum NO2 photolysis frequency of the Trebs et al. (2009) relation, J_max = (1 + alpha) B1^2 / (4 \\|B2\\|), in the positive diagonal mobility of the level-3 structural Hodge audit of the open mechanism, star_rxn = diag(J_max Q10); it does not enter any production rate. | Source: relic_convention; RELIC convention value (scale of the audit Hodge mobility; no literature value) |
| `atmosphere.chemistry.isoprene_carbon_atoms` | `dimensionless` | `5.0; bounds=(5.0, 5.0)` | `fixed` | Carbon atoms per molecule of isoprene (2-methyl-1,3-butadiene, C5H8, N_C = 5.0) for the reactive carbon molar conservation ledger. | Source: literature; Atkinson, R. and Arey, J. (2003) Atmospheric degradation of volatile organic compounds, Chem. Rev. 103, 4605-4638, doi:10.1021/cr0206420 (isoprene molecular formula C5H8) |
| `atmosphere.chemistry.isoprene_ro2_no2_stoichiometric_yield` | `dimensionless` | `2.0; bounds=(0.0, 4.0)` | `fixed` | Stoichiometric NO-to-NO2 conversions per OH-initiated isoprene oxidation event along the high-NOx peroxy radical propagation branch (RO2 + NO -> RO + NO2 followed by RO + O2 -> carbonyl + HO2 and HO2 + NO -> OH + NO2; gamma_NO2 = 2.0). | Source: literature; Seinfeld, J.H. and Pandis, S.N. (2016) Atmospheric Chemistry and Physics, 3rd ed., Wiley, Chapter 6, Eqs. (6.45)-(6.49) (two NO -> NO2 conversions per OH-initiated hydrocarbon oxidation cycle); Jenkin, M.E. and Clemitshaw, K.C. (2000) Ozone and other secondary photochemical pollutants: chemical processes governing their formation in the planetary boundary layer, Atmos. Environ. 34, 2499-2527, doi:10.1016/S1352-2310(99)00478-1 |
| `atmosphere.chemistry.j_no2_irradiance_linear_coefficient` | `W-1 m2 s-1` | `1.47e-05; bounds=(1.44e-05, 1.5e-05)` | `fixed` | NO2 photolysis frequency from the solar global irradiance G (shortwave_down): linear coefficient B1 = (1.47 +- 0.03) e-5 W-1 m2 s-1 of J_NO2 = (1 + alpha) (B1 G + B2 G^2). G already carries the solar geometry and the cloud attenuation. The concave polynomial rises to its vertex G_v = B1 / (2 \\|B2\\|) = 1519 W m-2; at and above G_v the vertex value is held (monotone extension beyond the fitted curvature). Bounds are the stated 1-sigma fit errors. | Source: literature; Trebs, I., Bohn, B., Ammann, C., Rummel, U., Blumthaler, M., Konigstedt, R., Meixner, F.X., Fan, S. and Andreae, M.O. (2009) Relationship between the NO2 photolysis frequency and the solar global irradiance, Atmos. Meas. Tech. 2, 725-739, doi:10.5194/amt-2-725-2009; Eq. (1)-(2), abstract and Sect. 5: B1 = (1.47 +- 0.03) x 10-5 W-1 m2 s-1, B2 = (-4.84 +- 0.31) x 10-9 W-2 m4 s-1 |
| `atmosphere.chemistry.j_no2_irradiance_quadratic_coefficient` | `W-2 m4 s-1` | `-4.84e-09; bounds=(-5.15e-09, -4.53e-09)` | `fixed` | NO2 photolysis frequency from the solar global irradiance G (shortwave_down): quadratic coefficient B2 = (-4.84 +- 0.31) e-9 W-2 m4 s-1 of J_NO2 = (1 + alpha) (B1 G + B2 G^2). G already carries the solar geometry and the cloud attenuation. The concave polynomial rises to its vertex G_v = B1 / (2 \\|B2\\|) = 1519 W m-2; at and above G_v the vertex value is held (monotone extension beyond the fitted curvature). Bounds are the stated 1-sigma fit errors. | Source: literature; Trebs, I., Bohn, B., Ammann, C., Rummel, U., Blumthaler, M., Konigstedt, R., Meixner, F.X., Fan, S. and Andreae, M.O. (2009) Relationship between the NO2 photolysis frequency and the solar global irradiance, Atmos. Meas. Tech. 2, 725-739, doi:10.5194/amt-2-725-2009; Eq. (1)-(2), abstract and Sect. 5: B1 = (1.47 +- 0.03) x 10-5 W-1 m2 s-1, B2 = (-4.84 +- 0.31) x 10-9 W-2 m4 s-1 |
| `atmosphere.chemistry.k_isoprene_no3_298` | `cm3 molecule-1 s-1` | `6.963486e-13; bounds=(4.39e-13, 1.11e-12)` | `fixed` | Isoprene + NO3 -> products (MCM v3.3.1 NO3 + C5H8 = NISOPO2) rate coefficient at the Arrhenius reference temperature T_ref = 298.15 K, evaluated from the MCM v3.3.1 expression 3.15e-12 exp(-450/T) cm3 molecule-1 s-1; the nighttime isoprene oxidant term of the chemical lifetime diagnostic. | Source: literature; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015, supplement file MCM_C5H8_subset.fac line 112): NO3 + C5H8 = NISOPO2, 3.15e-12 exp(-450/T); Atkinson et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006; datasheet II.A3.69: k = 7.0e-13 cm3 molecule-1 s-1 at 298 K, Delta log k = +-0.2; k = 3.15e-12 exp(-450/T) over 250-390 K |
| `atmosphere.chemistry.k_isoprene_no3_activation_energy` | `J mol-1` | `3741.508; bounds=(2078.6, 5404.4)` | `fixed` | Activation energy Ea = (E/R) R of isoprene + NO3 in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; E/R = 450 K (MCM v3.3.1, IUPAC 2006 II.A3.69 and IUPAC 2015 NO3_VOC8); positive: NO3 addition speeds up with warming. Bounds are the IUPAC reliability E/R +- 200 K. | Source: literature; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015, supplement file MCM_C5H8_subset.fac line 112): NO3 + C5H8 = NISOPO2, 3.15e-12 exp(-450/T); Atkinson et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006; datasheet II.A3.69: k = 3.15e-12 exp(-450/T) over 250-390 K, E/R = 450 K, Delta(E/R) = +-200 K |
| `atmosphere.chemistry.k_isoprene_o3_298` | `cm3 molecule-1 s-1` | `1.27e-17; bounds=(1e-17, 1.5e-17)` | `fixed` | Isoprene + O3 -> products rate coefficient at the Arrhenius reference temperature (IUPAC preferred k298). | Source: literature; Atkinson et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006; datasheet II.A1.157: k = 1.27e-17 cm3 molecule-1 s-1 at 298 K, Delta log k = +-0.10; k = 1.03e-14 exp(-1995/T) over 240-360 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): C5H8 + O3, 1.03e-14 exp(-1995/T) |
| `atmosphere.chemistry.k_isoprene_o3_activation_energy` | `J mol-1` | `16587.4; bounds=(14924.5, 18250.2)` | `fixed` | Activation energy Ea = (E/R) R of isoprene + O3 in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; E/R = 1995 K (IUPAC II.A1.157, MCM v3.3.1); positive: ozonolysis speeds up with warming. Bounds are the IUPAC reliability E/R +- 200 K. | Source: literature; Atkinson et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006; datasheet II.A1.157: k = 1.03e-14 exp(-1995/T), Delta(E/R) = +-200 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): C5H8 + O3, 1.03e-14 exp(-1995/T) |
| `atmosphere.chemistry.k_isoprene_oh_298` | `cm3 molecule-1 s-1` | `1e-10; bounds=(9e-11, 1.1e-10)` | `fixed` | Isoprene + OH -> products rate coefficient at the Arrhenius reference temperature (IUPAC preferred k298). | Source: literature; Atkinson et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006; datasheet II.A2.10: k = 1.0e-10 cm3 molecule-1 s-1 at 298 K, Delta log k = +-0.10; k = 2.7e-11 exp(390/T) over 240-430 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): C5H8 + OH, 2.7e-11 exp(390/T) |
| `atmosphere.chemistry.k_isoprene_oh_activation_energy` | `J mol-1` | `-3242.6; bounds=(-4074.1, -2411.2)` | `fixed` | Activation energy Ea = (E/R) R of isoprene + OH in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; E/R = -390 K (IUPAC II.A2.10, MCM v3.3.1); negative: OH addition slows with warming. Bounds are the IUPAC reliability E/R +- 100 K. | Source: literature; Atkinson et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006; datasheet II.A2.10: k = 2.7e-11 exp(390/T), E/R = -390 K, Delta(E/R) = +-100 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): C5H8 + OH, 2.7e-11 exp(390/T) |
| `atmosphere.chemistry.k_no_o3_298` | `cm3 molecule-1 s-1` | `1.8e-14; bounds=(1.4e-14, 2e-14)` | `fixed` | NO + O3 -> NO2 + O2 rate coefficient at the Arrhenius reference temperature (IUPAC preferred k298). | Source: literature; Atkinson et al. (2004) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume I, Atmos. Chem. Phys. 4, 1461-1738, doi:10.5194/acp-4-1461-2004; datasheet I.A3.54: k = 1.8e-14 cm3 molecule-1 s-1 at 298 K, Delta log k = +-0.08; k = 1.4e-12 exp(-1310/T) over 195-308 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): NO + O3 = NO2, 1.4e-12 exp(-1310/T) |
| `atmosphere.chemistry.k_no_o3_activation_energy` | `J mol-1` | `10891.9; bounds=(9229.1, 12554.8)` | `fixed` | Activation energy Ea = (E/R) R of NO + O3 in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; E/R = 1310 K (IUPAC I.A3.54, MCM v3.3.1); positive: the rate increases with temperature. Bounds are the IUPAC reliability E/R +- 200 K. | Source: literature; Atkinson et al. (2004) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume I, Atmos. Chem. Phys. 4, 1461-1738, doi:10.5194/acp-4-1461-2004; datasheet I.A3.54: k = 1.4e-12 exp(-1310/T), Delta(E/R) = +-200 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): NO + O3 = NO2, 1.4e-12 exp(-1310/T) |
| `atmosphere.chemistry.k_o1d_h2o_298` | `cm3 molecule-1 s-1` | `2e-10; bounds=(0.0, inf)` | `fixed` | O(1D) + H2O -> OH + OH rate coefficient at the Arrhenius reference temperature (tabulated k298); enters the fraction f = k_H2O[H2O] / (k_H2O[H2O] + k_N2[N2] + k_O2[O2]) of O(1D) that forms OH. A rate coefficient is positive; the source states no reliability interval. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Table 7.1: O(1D) + H2O -> OH + OH, k298 = 2.0e-10, k = 1.63e-10 exp(60/T) cm3 molecule-1 s-1 (k = A exp(-B/T), B = -60 K), source of data Sander, S.P. et al. (2006) Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation Number 15, JPL Publication 06-2 |
| `atmosphere.chemistry.k_o1d_h2o_activation_energy` | `J mol-1` | `-498.9; bounds=(-inf, inf)` | `fixed` | Activation energy Ea = B R of O(1D) + H2O -> OH + OH in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; B = -60 K: negative, the rate coefficient increases as the air cools. The Arrhenius form admits either sign of Ea, so the interval is unbounded. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Table 7.1: O(1D) + H2O -> OH + OH, k298 = 2.0e-10, k = 1.63e-10 exp(60/T) cm3 molecule-1 s-1 (k = A exp(-B/T), B = -60 K), source of data Sander, S.P. et al. (2006) Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation Number 15, JPL Publication 06-2 |
| `atmosphere.chemistry.k_o1d_n2_298` | `cm3 molecule-1 s-1` | `3.1e-11; bounds=(0.0, inf)` | `fixed` | O(1D) + N2 -> O(3P) + N2 rate coefficient at the Arrhenius reference temperature (tabulated k298); enters the fraction f = k_H2O[H2O] / (k_H2O[H2O] + k_N2[N2] + k_O2[O2]) of O(1D) that forms OH. A rate coefficient is positive; the source states no reliability interval. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Table 7.1: O(1D) + N2 -> O(3P) + N2, k298 = 3.1e-11, k = 2.15e-11 exp(110/T) cm3 molecule-1 s-1 (k = A exp(-B/T), B = -110 K), source of data Sander, S.P. et al. (2006) Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation Number 15, JPL Publication 06-2 |
| `atmosphere.chemistry.k_o1d_n2_activation_energy` | `J mol-1` | `-914.6; bounds=(-inf, inf)` | `fixed` | Activation energy Ea = B R of O(1D) + N2 -> O(3P) + N2 in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; B = -110 K: negative, the rate coefficient increases as the air cools. The Arrhenius form admits either sign of Ea, so the interval is unbounded. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Table 7.1: O(1D) + N2 -> O(3P) + N2, k298 = 3.1e-11, k = 2.15e-11 exp(110/T) cm3 molecule-1 s-1 (k = A exp(-B/T), B = -110 K), source of data Sander, S.P. et al. (2006) Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation Number 15, JPL Publication 06-2 |
| `atmosphere.chemistry.k_o1d_o2_298` | `cm3 molecule-1 s-1` | `4e-11; bounds=(0.0, inf)` | `fixed` | O(1D) + O2 -> O(3P) + O2 rate coefficient at the Arrhenius reference temperature (tabulated k298); enters the fraction f = k_H2O[H2O] / (k_H2O[H2O] + k_N2[N2] + k_O2[O2]) of O(1D) that forms OH. A rate coefficient is positive; the source states no reliability interval. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Table 7.1: O(1D) + O2 -> O(3P) + O2, k298 = 4.0e-11, k = 3.2e-11 exp(67/T) cm3 molecule-1 s-1 (k = A exp(-B/T), B = -67 K), source of data Sander, S.P. et al. (2006) Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation Number 15, JPL Publication 06-2 |
| `atmosphere.chemistry.k_o1d_o2_activation_energy` | `J mol-1` | `-557.1; bounds=(-inf, inf)` | `fixed` | Activation energy Ea = B R of O(1D) + O2 -> O(3P) + O2 in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; B = -67 K: negative, the rate coefficient increases as the air cools. The Arrhenius form admits either sign of Ea, so the interval is unbounded. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Table 7.1: O(1D) + O2 -> O(3P) + O2, k298 = 4.0e-11, k = 3.2e-11 exp(67/T) cm3 molecule-1 s-1 (k = A exp(-B/T), B = -67 K), source of data Sander, S.P. et al. (2006) Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation Number 15, JPL Publication 06-2 |
| `atmosphere.chemistry.k_oh_ch4_298` | `cm3 molecule-1 s-1` | `6.4e-15; bounds=(6e-15, 7e-15)` | `fixed` | OH + CH4 -> CH3 + H2O rate coefficient at the Arrhenius reference temperature (IUPAC preferred k298). | Source: literature; Atkinson et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006; datasheet II.A2.3: k = 6.4e-15 cm3 molecule-1 s-1 at 298 K, Delta log k = +-0.08; k = 1.85e-12 exp(-1690/T) over 200-300 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): CH4 + OH, 1.85e-12 exp(-1690/T) |
| `atmosphere.chemistry.k_oh_ch4_activation_energy` | `J mol-1` | `14051.4; bounds=(13220.0, 14882.9)` | `fixed` | Activation energy Ea = (E/R) R of OH + CH4 in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; E/R = 1690 K (IUPAC II.A2.3, MCM v3.3.1); positive: H abstraction speeds up with warming. Bounds are the IUPAC reliability E/R +- 100 K. | Source: literature; Atkinson et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006; datasheet II.A2.3: k = 1.85e-12 exp(-1690/T), Delta(E/R) = +-100 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): CH4 + OH, 1.85e-12 exp(-1690/T) |
| `atmosphere.chemistry.k_oh_co_298` | `cm3 molecule-1 s-1` | `2.3e-13; bounds=(2e-13, 2.5e-13)` | `fixed` | OH + CO -> HO2 + CO2 rate at 298K | Source: auto_migrated_parameter_database; MCM: pressure-dependent, simplified here; McCabe et al. (2001): Kinetics of OH + CO |
| `atmosphere.chemistry.k_oh_co_activation_energy` | `J mol-1` | `0.0; bounds=(-inf, inf)` | `fixed` | Activation energy of OH + CO in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K. IUPAC II.A2.12 recommends a temperature-independent k = 1.44e-13 (1 + [N2]/4.2e19 molecule cm-3) over 200-300 K and 0-1 bar, so Ea = 0; the pressure term is folded into k_oh_co_298 at 1 atm. The Arrhenius form admits either sign of Ea, so the interval is unbounded. | Source: literature; Atkinson et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006; datasheet II.A2.12: k = 1.44e-13 (1 + [N2]/4.2e19) cm3 molecule-1 s-1, no temperature dependence over 200-300 K |
| `atmosphere.chemistry.k_oh_no2_298` | `cm3 molecule-1 s-1` | `1.15e-11; bounds=(1e-11, 1.2e-11)` | `fixed` | OH + NO2 (+M) -> HNO3 rate coefficient at 1 atm and T_ref = 298.15 K: the IUPAC I.A3.43 fall-off k = k0 kinf/(k0 + kinf) Fc^(1/(1 + (log10(k0/kinf)/N)^2)), N = 0.75 - 1.27 log10 Fc, evaluated at [N2] = p/(kB T) = 2.46e19 molecule cm-3. | Source: literature; Atkinson et al. (2004) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume I, Atmos. Chem. Phys. 4, 1461-1738, doi:10.5194/acp-4-1461-2004; datasheet I.A3.43: k0 = 3.3e-30 (T/300)^-3.0 [N2], kinf = 4.1e-11 cm3 molecule-1 s-1, Fc = 0.4; Mollner et al. (2010) Rate of gas phase association of hydroxyl radical and nitrogen dioxide, Science 330, 646-649, doi:10.1126/science.1193030: HONO2 channel 9.2e-12 (+-0.4e-12) cm3 molecule-1 s-1 in air at 760 Torr, 25 degC |
| `atmosphere.chemistry.k_oh_no2_activation_energy` | `J mol-1` | `0.0; bounds=(-inf, inf)` | `fixed` | Activation energy of OH + NO2 (+M) -> HNO3 in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K. Convention Ea = 0: the termolecular falloff of IUPAC I.A3.43 is not implemented and k_oh_no2_298 is held at its 1 atm, 298.15 K value. The Arrhenius form admits either sign of Ea, so the interval is unbounded. | Source: relic_convention; Atkinson et al. (2004) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume I, Atmos. Chem. Phys. 4, 1461-1738, doi:10.5194/acp-4-1461-2004; datasheet I.A3.43: k0 = 3.3e-30 (T/300)^-3.0 [N2], k_inf = 4.1e-11 cm3 molecule-1 s-1, Fc = 0.4 |
| `atmosphere.chemistry.k_ro2_no_298` | `cm3 molecule-1 s-1` | `9e-12; bounds=(4.5e-12, 1.8e-11)` | `fixed` | Organic peroxy radical + NO rate coefficient at the Arrhenius reference temperature (RO2 + NO -> RO + NO2, MCM v3.3.1 generic KRO2NO = 2.7e-12 exp(360/T) cm3 molecule-1 s-1, k298 = 9.0e-12 cm3 molecule-1 s-1); governs the high-NOx propagation branch of isoprene-derived peroxy radicals. | Source: literature; Jenkin, M.E., Young, J.C. and Rickard, A.R. (2015) The MCM v3.3.1 degradation scheme for isoprene, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015; generic peroxy radical rate coefficient KRO2NO = 2.7e-12 exp(360/T) cm3 molecule-1 s-1, k298 = 9.0e-12 cm3 molecule-1 s-1; Atkinson, R. et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II - gas phase reactions of organic species, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006 (IUPAC datasheet II.A2.10 / peroxy + NO kinetics) |
| `atmosphere.chemistry.k_ro2_no_activation_energy` | `J mol-1` | `-2993.2; bounds=(-4656.1, -1330.3)` | `fixed` | Activation energy Ea = (E/R) R for RO2 + NO -> RO + NO2 in k(T) = k_298 exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; E/R = -360 K from MCM v3.3.1 KRO2NO = 2.7e-12 exp(360/T) cm3 molecule-1 s-1. | Source: literature; Jenkin, M.E., Young, J.C. and Rickard, A.R. (2015) The MCM v3.3.1 degradation scheme for isoprene, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015; KRO2NO = 2.7e-12 exp(360/T) cm3 molecule-1 s-1 (E/R = -360 K); Atkinson, R. et al. (2006) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II - gas phase reactions of organic species, Atmos. Chem. Phys. 6, 3625-4055, doi:10.5194/acp-6-3625-2006 |
| `atmosphere.chemistry.k_ro2_termination_loss` | `s-1` | `0.02; bounds=(0.0, 1.0)` | `fixed` | Effective first-order non-NO loss rate coefficient k_loss,RO2 of isoprene hydroxyperoxy radicals (ISOPOO termination with HO2 and 1,6-H-shift unimolecular isomerization) in the branching fraction beta_NO = k_RO2_NO(T)[NO] / (k_RO2_NO(T)[NO] + k_loss,RO2). | Source: literature; Peeters, J., Muller, J.-F., Stavrakou, T. and Nguyen, V.S. (2014) Hydroxyl radical recycling in isoprene oxidation driven by hydrogen bonding and hydrogen tunneling: The upgraded LIM1 mechanism, J. Phys. Chem. A 118, 8625-8643, doi:10.1021/jp5033146; Seinfeld, J.H. and Pandis, S.N. (2016) Atmospheric Chemistry and Physics, 3rd ed., Wiley, Chapter 6 (RO2 + NO vs RO2 + HO2 / isomerization branching) |
| `atmosphere.chemistry.mcm_no2_photolysis_l` | `s-1` | `0.01165; bounds=(0.01165, 0.01165)` | `fixed` | MCM clear-sky photolysis parameter l of J4 (NO2 -> NO + O(3P)) in J = l cos(chi)^m exp(-n sec(chi)); the ratio rho(chi) = J1/J4 = (l1/l4) cos(chi)^(m1-m4) exp(-(n1-n4) sec(chi)) converts the NO2 photolysis frequency into J(O1D) = J_NO2 rho(chi) (rho = 4.24e-3 for overhead sun). | Source: literature; Saunders, S.M., Jenkin, M.E., Derwent, R.G. and Pilling, M.J. (2003) Protocol for the development of the Master Chemical Mechanism, MCM v3 (Part A), Atmos. Chem. Phys. 3, 161-180, doi:10.5194/acp-3-161-2003: J = l cos(chi)^m exp(-n sec(chi)); J1 O3 -> O(1D) + O2: l = 6.073e-5 s-1, m = 1.743, n = 0.474; J4 NO2 -> NO + O(3P): l = 1.165e-2 s-1, m = 0.244, n = 0.267 (clear sky); MCM v3.3.1 photolysis rates, https://mcm.york.ac.uk/MCM/rates/photolysis (same J1 and J4 parameters) |
| `atmosphere.chemistry.mcm_no2_photolysis_m` | `dimensionless` | `0.244; bounds=(0.244, 0.244)` | `fixed` | MCM clear-sky photolysis parameter m of J4 (NO2 -> NO + O(3P)) in J = l cos(chi)^m exp(-n sec(chi)); the ratio rho(chi) = J1/J4 = (l1/l4) cos(chi)^(m1-m4) exp(-(n1-n4) sec(chi)) converts the NO2 photolysis frequency into J(O1D) = J_NO2 rho(chi) (rho = 4.24e-3 for overhead sun). | Source: literature; Saunders, S.M., Jenkin, M.E., Derwent, R.G. and Pilling, M.J. (2003) Protocol for the development of the Master Chemical Mechanism, MCM v3 (Part A), Atmos. Chem. Phys. 3, 161-180, doi:10.5194/acp-3-161-2003: J = l cos(chi)^m exp(-n sec(chi)); J1 O3 -> O(1D) + O2: l = 6.073e-5 s-1, m = 1.743, n = 0.474; J4 NO2 -> NO + O(3P): l = 1.165e-2 s-1, m = 0.244, n = 0.267 (clear sky); MCM v3.3.1 photolysis rates, https://mcm.york.ac.uk/MCM/rates/photolysis (same J1 and J4 parameters) |
| `atmosphere.chemistry.mcm_no2_photolysis_n` | `dimensionless` | `0.267; bounds=(0.267, 0.267)` | `fixed` | MCM clear-sky photolysis parameter n of J4 (NO2 -> NO + O(3P)) in J = l cos(chi)^m exp(-n sec(chi)); the ratio rho(chi) = J1/J4 = (l1/l4) cos(chi)^(m1-m4) exp(-(n1-n4) sec(chi)) converts the NO2 photolysis frequency into J(O1D) = J_NO2 rho(chi) (rho = 4.24e-3 for overhead sun). | Source: literature; Saunders, S.M., Jenkin, M.E., Derwent, R.G. and Pilling, M.J. (2003) Protocol for the development of the Master Chemical Mechanism, MCM v3 (Part A), Atmos. Chem. Phys. 3, 161-180, doi:10.5194/acp-3-161-2003: J = l cos(chi)^m exp(-n sec(chi)); J1 O3 -> O(1D) + O2: l = 6.073e-5 s-1, m = 1.743, n = 0.474; J4 NO2 -> NO + O(3P): l = 1.165e-2 s-1, m = 0.244, n = 0.267 (clear sky); MCM v3.3.1 photolysis rates, https://mcm.york.ac.uk/MCM/rates/photolysis (same J1 and J4 parameters) |
| `atmosphere.chemistry.mcm_o1d_photolysis_l` | `s-1` | `6.073e-05; bounds=(6.073e-05, 6.073e-05)` | `fixed` | MCM clear-sky photolysis parameter l of J1 (O3 -> O(1D) + O2) in J = l cos(chi)^m exp(-n sec(chi)); the ratio rho(chi) = J1/J4 = (l1/l4) cos(chi)^(m1-m4) exp(-(n1-n4) sec(chi)) converts the NO2 photolysis frequency into J(O1D) = J_NO2 rho(chi) (rho = 4.24e-3 for overhead sun). | Source: literature; Saunders, S.M., Jenkin, M.E., Derwent, R.G. and Pilling, M.J. (2003) Protocol for the development of the Master Chemical Mechanism, MCM v3 (Part A), Atmos. Chem. Phys. 3, 161-180, doi:10.5194/acp-3-161-2003: J = l cos(chi)^m exp(-n sec(chi)); J1 O3 -> O(1D) + O2: l = 6.073e-5 s-1, m = 1.743, n = 0.474; J4 NO2 -> NO + O(3P): l = 1.165e-2 s-1, m = 0.244, n = 0.267 (clear sky); MCM v3.3.1 photolysis rates, https://mcm.york.ac.uk/MCM/rates/photolysis (same J1 and J4 parameters) |
| `atmosphere.chemistry.mcm_o1d_photolysis_m` | `dimensionless` | `1.743; bounds=(1.743, 1.743)` | `fixed` | MCM clear-sky photolysis parameter m of J1 (O3 -> O(1D) + O2) in J = l cos(chi)^m exp(-n sec(chi)); the ratio rho(chi) = J1/J4 = (l1/l4) cos(chi)^(m1-m4) exp(-(n1-n4) sec(chi)) converts the NO2 photolysis frequency into J(O1D) = J_NO2 rho(chi) (rho = 4.24e-3 for overhead sun). | Source: literature; Saunders, S.M., Jenkin, M.E., Derwent, R.G. and Pilling, M.J. (2003) Protocol for the development of the Master Chemical Mechanism, MCM v3 (Part A), Atmos. Chem. Phys. 3, 161-180, doi:10.5194/acp-3-161-2003: J = l cos(chi)^m exp(-n sec(chi)); J1 O3 -> O(1D) + O2: l = 6.073e-5 s-1, m = 1.743, n = 0.474; J4 NO2 -> NO + O(3P): l = 1.165e-2 s-1, m = 0.244, n = 0.267 (clear sky); MCM v3.3.1 photolysis rates, https://mcm.york.ac.uk/MCM/rates/photolysis (same J1 and J4 parameters) |
| `atmosphere.chemistry.mcm_o1d_photolysis_n` | `dimensionless` | `0.474; bounds=(0.474, 0.474)` | `fixed` | MCM clear-sky photolysis parameter n of J1 (O3 -> O(1D) + O2) in J = l cos(chi)^m exp(-n sec(chi)); the ratio rho(chi) = J1/J4 = (l1/l4) cos(chi)^(m1-m4) exp(-(n1-n4) sec(chi)) converts the NO2 photolysis frequency into J(O1D) = J_NO2 rho(chi) (rho = 4.24e-3 for overhead sun). | Source: literature; Saunders, S.M., Jenkin, M.E., Derwent, R.G. and Pilling, M.J. (2003) Protocol for the development of the Master Chemical Mechanism, MCM v3 (Part A), Atmos. Chem. Phys. 3, 161-180, doi:10.5194/acp-3-161-2003: J = l cos(chi)^m exp(-n sec(chi)); J1 O3 -> O(1D) + O2: l = 6.073e-5 s-1, m = 1.743, n = 0.474; J4 NO2 -> NO + O(3P): l = 1.165e-2 s-1, m = 0.244, n = 0.267 (clear sky); MCM v3.3.1 photolysis rates, https://mcm.york.ac.uk/MCM/rates/photolysis (same J1 and J4 parameters) |
| `atmosphere.chemistry.minimum_concentration` | `mol mol-1` | `1e-12; bounds=(1e-15, 1e-06)` | `fixed` | Positive concentration metric scale used only by the level-3 Hodge audit; not a runtime species floor | Source: auto_migrated_parameter_database; Seinfeld & Pandis (2016) Atmospheric Chemistry and Physics, 3rd ed., Wiley |
| `atmosphere.chemistry.o1d_water_oh_yield` | `dimensionless` | `2.0; bounds=(2.0, 2.0)` | `fixed` | OH molecules formed per O(1D) + H2O reaction: the stoichiometry of O(1D) + H2O -> OH + OH in P_OH = y f J(O1D) [O3]. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Table 7.1: O(1D) + H2O -> OH + OH, source of data Sander, S.P. et al. (2006) Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation Number 15, JPL Publication 06-2 |
| `atmosphere.chemistry.photolysis.matsumi_amplitudes` | `dimensionless` | `[0.8036, 8.9061, 0.1192]; bounds=None` | `fixed` | Amplitudes A1, A2, A3 of the three terms of the Matsumi et al. (2002) O(1D) quantum-yield expression. | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.matsumi_band_centers` | `um` | `[0.304225, 0.314957, 0.310737]; bounds=None` | `fixed` | Centres x1, x2, x3 (304.225, 314.957, 310.737 nm) of the three terms of the Matsumi et al. (2002) expression, in micrometres. | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.matsumi_band_widths` | `um` | `[0.005576, 0.006601, 0.002187]; bounds=None` | `fixed` | Widths w1, w2, w3 (5.576, 6.601, 2.187 nm) of the three terms of the Matsumi et al. (2002) expression, in micrometres. | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.matsumi_energy_gap` | `cm-1` | `825.518; bounds=(825.518, 825.518)` | `fixed` | Energy nu2 (825.518 cm-1) of q = exp(-nu2/(R T)) in the Matsumi et al. (2002) expression; with nu2 in cm-1, R = k_B/(h c) = 0.695 cm-1 K-1 (derived from the SSOT constants k_B, h and c). | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.matsumi_offset` | `dimensionless` | `0.08; bounds=(0.08, 0.08)` | `fixed` | Constant C of the Matsumi et al. (2002) expression (spin-forbidden O(1D) channel). | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.matsumi_reference_temperature` | `K` | `300.0; bounds=(300.0, 300.0)` | `fixed` | Normalization temperature of the (T/300)^2 and (T/300)^1.5 factors of the Matsumi et al. (2002) expression. | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.matsumi_temperature_max` | `K` | `300.0; bounds=(300.0, 300.0)` | `fixed` | Upper temperature of the evaluated range of the Matsumi et al. (2002) O(1D) quantum-yield expression; above it the expression is evaluated at this temperature (nearest evaluated temperature held). | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.matsumi_temperature_min` | `K` | `200.0; bounds=(200.0, 200.0)` | `fixed` | Lower temperature of the evaluated range of the Matsumi et al. (2002) O(1D) quantum-yield expression; below it the expression is evaluated at this temperature (nearest evaluated temperature held). | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.matsumi_wavelength_max` | `um` | `0.328; bounds=(0.328, 0.328)` | `fixed` | Upper wavelength limit (328 nm) of the Matsumi et al. (2002) O(1D) quantum-yield expression. | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.matsumi_wavelength_min` | `um` | `0.306; bounds=(0.306, 0.306)` | `fixed` | Lower wavelength limit (306 nm) of the Matsumi et al. (2002) O(1D) quantum-yield expression. | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.no2_cross_section` | `cm2` | `[1.064e-19, 1.2989999999999998e-19, 1.604e-19, 1.882e-19, 2.174e-19, 2.5419999999999996e-19, 2.879e-19, 3.188e-19, 3.587e-19, 4.0199999999999996e-19, 4.175e-19, 4.61599999999999...; bounds=None` | `fixed` | NO2 absorption cross section per molecule at 298 K averaged over lambda +- 2.5 nm; the temperature dependence is minimal (Warneck and Williams 2012), so it is used at every air temperature. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.13a: 1e19 sigma (cm2), footnote 'the temperature dependence is minimal'; data of Jenouvrier et al. (1996), Merienne et al. (1995), Vandaele et al. (1998) |
| `atmosphere.chemistry.photolysis.no2_cross_section_wavelength` | `um` | `[0.295, 0.3, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37, 0.375, 0.38, 0.385, 0.39, 0.395, 0.4, 0.405, 0.41, 0.415, 0.42]; bounds=None` | `fixed` | Centres of the 5-nm averaging intervals (centre +- 2.5 nm) of the NO2 absorption cross section, 295-420 nm, in micrometres. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.13a (NO2 at 298 K, 205-650 nm) |
| `atmosphere.chemistry.photolysis.no2_quantum_yield_high_temperature` | `dimensionless` | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1...; bounds=None` | `fixed` | NO2 + hv -> NO + O(3P) quantum yield at the upper tabulated temperature (298 K) on no2_quantum_yield_wavelength: 1.00 at 300-398 nm, 0.95 at 399 nm falling to 0.04 at 417 nm. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.13b: quantum yields at 298 and 248 K from the critical assessment of Troe (2000), Z. Phys. Chem. 214, 573-581 |
| `atmosphere.chemistry.photolysis.no2_quantum_yield_low_temperature` | `dimensionless` | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1...; bounds=None` | `fixed` | NO2 + hv -> NO + O(3P) quantum yield at the lower tabulated temperature (248 K) on no2_quantum_yield_wavelength: 1.00 at 300-398 nm, 0.94 at 399 nm falling to 0.02 at 417 nm. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.13b: quantum yields at 298 and 248 K from the critical assessment of Troe (2000), Z. Phys. Chem. 214, 573-581 |
| `atmosphere.chemistry.photolysis.no2_quantum_yield_temperature_high` | `K` | `298.0; bounds=(298.0, 298.0)` | `fixed` | Temperature of no2_quantum_yield_high_temperature. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.13b column phi298 |
| `atmosphere.chemistry.photolysis.no2_quantum_yield_temperature_low` | `K` | `248.0; bounds=(248.0, 248.0)` | `fixed` | Temperature of no2_quantum_yield_low_temperature. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.13b column phi248 |
| `atmosphere.chemistry.photolysis.no2_quantum_yield_wavelength` | `um` | `[0.3, 0.301, 0.302, 0.303, 0.304, 0.305, 0.306, 0.307, 0.308, 0.309, 0.31, 0.311, 0.312, 0.313, 0.314, 0.315, 0.316, 0.317, 0.318, 0.319, 0.32, 0.321, 0.322, 0.323, 0.324, 0.325...; bounds=None` | `fixed` | Wavelengths (1-nm spacing, 300-417 nm) of the NO2 + hv -> NO + O(3P) quantum yields; also the integration grid of the NO2 photolysis temperature factor. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.13b (Troe 2000 evaluation) |
| `atmosphere.chemistry.photolysis.o1d_integration_wavelength` | `um` | `[0.29, 0.291, 0.292, 0.293, 0.294, 0.295, 0.296, 0.297, 0.298, 0.299, 0.3, 0.301, 0.302, 0.303, 0.304, 0.305, 0.306, 0.307, 0.308, 0.309, 0.31, 0.311, 0.312, 0.313, 0.314, 0.315...; bounds=None` | `fixed` | Integration grid (1-nm spacing, 290-354 nm) of the O3 -> O(1D) photolysis temperature factor: the range of the tabulated Hartley and Huggins cross sections above the surface ultraviolet cut-off. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Tables 6.11a-6.11c |
| `atmosphere.chemistry.photolysis.o1d_quantum_yield_long_wavelength` | `dimensionless` | `0.08; bounds=(0.0, 1.0)` | `fixed` | O(1D) quantum yield of O3 photolysis above matsumi_wavelength_max (spin-forbidden channel), independent of temperature. | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.o1d_quantum_yield_short_wavelength` | `dimensionless` | `0.9; bounds=(0.0, 1.0)` | `fixed` | O(1D) quantum yield of O3 photolysis below matsumi_wavelength_min, independent of temperature. | Source: literature; Matsumi, Y. et al. (2002) Quantum yields for production of O(1D) in the ultraviolet photolysis of ozone: Recommendation based on evaluation of laboratory data, J. Geophys. Res. 107(D3), 4024, doi:10.1029/2001JD000510: fitting expression for 306-328 nm and 200-300 K; recommended values 0.90 for 290-306 nm and 0.08 for 328-350 nm; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11c and text: Matsumi et al. (2002) formula and constants; phi(O1D) = 0.90 for 220-305 nm and 0.08 for 330-370 nm, independent of temperature |
| `atmosphere.chemistry.photolysis.o3_cross_section_high_temperature` | `cm2` | `[1.418e-18, 1.111e-18, 8.711e-19, 7.753e-19, 6.727e-19, 5.124e-19, 3.964e-19, 3.463e-19, 3.073e-19, 2.65e-19, 2.401e-19, 2.015e-19, 1.808e-19, 1.565e-19, 1.364e-19, 1.243e-19, 1...; bounds=None` | `fixed` | O3 absorption cross section per molecule at the upper tabulated temperature of each point (o3_cross_section_temperature_high). | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11a: O3 Hartley band at 298 and 226 K, Molina and Molina (1986) J. Geophys. Res. 91, 14501-14508; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11b: O3 Huggins bands at 295 and 218 K (312-340.2 nm, Malicet et al. 1995, J. Atmos. Chem. 21, 263-273) and 295 and 220 K (340.5-354 nm, Cacciani et al. 1989) |
| `atmosphere.chemistry.photolysis.o3_cross_section_low_temperature` | `cm2` | `[1.365e-18, 1.043e-18, 8.148e-19, 7.27e-19, 6.202e-19, 4.714e-19, 3.616e-19, 3.128e-19, 2.796e-19, 2.325e-19, 2.156e-19, 1.771e-19, 1.603e-19, 1.367e-19, 1.202e-19, 1.064e-19, 8...; bounds=None` | `fixed` | O3 absorption cross section per molecule at the lower tabulated temperature of each point (o3_cross_section_temperature_low). | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11a: O3 Hartley band at 298 and 226 K, Molina and Molina (1986) J. Geophys. Res. 91, 14501-14508; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11b: O3 Huggins bands at 295 and 218 K (312-340.2 nm, Malicet et al. 1995, J. Atmos. Chem. 21, 263-273) and 295 and 220 K (340.5-354 nm, Cacciani et al. 1989) |
| `atmosphere.chemistry.photolysis.o3_cross_section_temperature_high` | `K` | `[298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0, 295.0, 295.0, 295.0, 295.0, 295.0, 295.0, 295.0, 2...; bounds=None` | `fixed` | Upper tabulation temperature of each O3 cross section point: 298 K (Hartley), 295 K (Huggins, Chappuis). | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11a: O3 Hartley band at 298 and 226 K, Molina and Molina (1986) J. Geophys. Res. 91, 14501-14508; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11b: O3 Huggins bands at 295 and 218 K (312-340.2 nm, Malicet et al. 1995, J. Atmos. Chem. 21, 263-273) and 295 and 220 K (340.5-354 nm, Cacciani et al. 1989) |
| `atmosphere.chemistry.photolysis.o3_cross_section_temperature_low` | `K` | `[226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 226.0, 218.0, 218.0, 218.0, 218.0, 218.0, 218.0, 218.0, 2...; bounds=None` | `fixed` | Lower tabulation temperature of each O3 cross section point: 226 K (Hartley), 218 K (Huggins to 340.2 nm), 220 K (Huggins from 340.5 nm and Chappuis). | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11a: O3 Hartley band at 298 and 226 K, Molina and Molina (1986) J. Geophys. Res. 91, 14501-14508; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11b: O3 Huggins bands at 295 and 218 K (312-340.2 nm, Malicet et al. 1995, J. Atmos. Chem. 21, 263-273) and 295 and 220 K (340.5-354 nm, Cacciani et al. 1989) |
| `atmosphere.chemistry.photolysis.o3_cross_section_wavelength` | `um` | `[0.29, 0.292, 0.294, 0.295, 0.296, 0.298, 0.3, 0.301, 0.302, 0.303, 0.304, 0.305, 0.306, 0.307, 0.308, 0.309, 0.31, 0.311, 0.312, 0.3125, 0.313, 0.3135, 0.314, 0.3145, 0.315, 0....; bounds=None` | `fixed` | Wavelengths of the tabulated O3 absorption cross sections, 290-420 nm, in micrometres. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11a: O3 Hartley band at 298 and 226 K, Molina and Molina (1986) J. Geophys. Res. 91, 14501-14508; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.11b: O3 Huggins bands at 295 and 218 K (312-340.2 nm, Malicet et al. 1995, J. Atmos. Chem. 21, 263-273) and 295 and 220 K (340.5-354 nm, Cacciani et al. 1989) |
| `atmosphere.chemistry.photolysis.rayleigh_denominator_inverse_square` | `dimensionless` | `0.0027059889; bounds=(0.0027059889, 0.0027059889)` | `fixed` | Sea-level Rayleigh optical depth fit of Bodhaine et al. (1999) Eq. (30): denominator coefficient 0.0027059889 of lambda^-2. | Source: literature; Bodhaine, B.A., Wood, N.B., Dutton, E.G. and Slusser, J.R. (1999) On Rayleigh optical depth calculations, J. Atmos. Oceanic Technol. 16, 1854-1861, doi:10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2; Eq. (30): tau_R = 0.0021520 (1.0455996 - 341.29061 lambda^-2 - 0.90230850 lambda^2) / (1 + 0.0027059889 lambda^-2 - 85.968563 lambda^2), lambda in um, sea level 1013.25 hPa, 45 deg N, dry air with 360 ppm CO2 |
| `atmosphere.chemistry.photolysis.rayleigh_denominator_square` | `dimensionless` | `85.968563; bounds=(85.968563, 85.968563)` | `fixed` | Sea-level Rayleigh optical depth fit of Bodhaine et al. (1999) Eq. (30): denominator coefficient 85.968563 of lambda^2. | Source: literature; Bodhaine, B.A., Wood, N.B., Dutton, E.G. and Slusser, J.R. (1999) On Rayleigh optical depth calculations, J. Atmos. Oceanic Technol. 16, 1854-1861, doi:10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2; Eq. (30): tau_R = 0.0021520 (1.0455996 - 341.29061 lambda^-2 - 0.90230850 lambda^2) / (1 + 0.0027059889 lambda^-2 - 85.968563 lambda^2), lambda in um, sea level 1013.25 hPa, 45 deg N, dry air with 360 ppm CO2 |
| `atmosphere.chemistry.photolysis.rayleigh_numerator_constant` | `dimensionless` | `1.0455996; bounds=(1.0455996, 1.0455996)` | `fixed` | Sea-level Rayleigh optical depth fit of Bodhaine et al. (1999) Eq. (30): numerator constant 1.0455996. | Source: literature; Bodhaine, B.A., Wood, N.B., Dutton, E.G. and Slusser, J.R. (1999) On Rayleigh optical depth calculations, J. Atmos. Oceanic Technol. 16, 1854-1861, doi:10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2; Eq. (30): tau_R = 0.0021520 (1.0455996 - 341.29061 lambda^-2 - 0.90230850 lambda^2) / (1 + 0.0027059889 lambda^-2 - 85.968563 lambda^2), lambda in um, sea level 1013.25 hPa, 45 deg N, dry air with 360 ppm CO2 |
| `atmosphere.chemistry.photolysis.rayleigh_numerator_inverse_square` | `dimensionless` | `341.29061; bounds=(341.29061, 341.29061)` | `fixed` | Sea-level Rayleigh optical depth fit of Bodhaine et al. (1999) Eq. (30): numerator coefficient 341.29061 of lambda^-2. | Source: literature; Bodhaine, B.A., Wood, N.B., Dutton, E.G. and Slusser, J.R. (1999) On Rayleigh optical depth calculations, J. Atmos. Oceanic Technol. 16, 1854-1861, doi:10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2; Eq. (30): tau_R = 0.0021520 (1.0455996 - 341.29061 lambda^-2 - 0.90230850 lambda^2) / (1 + 0.0027059889 lambda^-2 - 85.968563 lambda^2), lambda in um, sea level 1013.25 hPa, 45 deg N, dry air with 360 ppm CO2 |
| `atmosphere.chemistry.photolysis.rayleigh_numerator_square` | `dimensionless` | `0.9023085; bounds=(0.9023085, 0.9023085)` | `fixed` | Sea-level Rayleigh optical depth fit of Bodhaine et al. (1999) Eq. (30): numerator coefficient 0.90230850 of lambda^2. | Source: literature; Bodhaine, B.A., Wood, N.B., Dutton, E.G. and Slusser, J.R. (1999) On Rayleigh optical depth calculations, J. Atmos. Oceanic Technol. 16, 1854-1861, doi:10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2; Eq. (30): tau_R = 0.0021520 (1.0455996 - 341.29061 lambda^-2 - 0.90230850 lambda^2) / (1 + 0.0027059889 lambda^-2 - 85.968563 lambda^2), lambda in um, sea level 1013.25 hPa, 45 deg N, dry air with 360 ppm CO2 |
| `atmosphere.chemistry.photolysis.rayleigh_scale` | `dimensionless` | `0.002152; bounds=(0.002152, 0.002152)` | `fixed` | Sea-level Rayleigh optical depth fit of Bodhaine et al. (1999) Eq. (30): leading factor 0.0021520 (sea level, 45 deg N). | Source: literature; Bodhaine, B.A., Wood, N.B., Dutton, E.G. and Slusser, J.R. (1999) On Rayleigh optical depth calculations, J. Atmos. Oceanic Technol. 16, 1854-1861, doi:10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2; Eq. (30): tau_R = 0.0021520 (1.0455996 - 341.29061 lambda^-2 - 0.90230850 lambda^2) / (1 + 0.0027059889 lambda^-2 - 85.968563 lambda^2), lambda in um, sea level 1013.25 hPa, 45 deg N, dry air with 360 ppm CO2 |
| `atmosphere.chemistry.photolysis.reference_ozone_column` | `molecule cm-2` | `8.0603e+18; bounds=(2.6868e+18, 1.3434e+19)` | `fixed` | Vertical ozone column of the reference slant path: 300 Dobson units (1 DU = 2.6868e16 molecule cm-2, a 0.01 mm layer of pure ozone at 273.15 K and 101325 Pa). | Source: literature; WMO definition of the Dobson unit: 0.01 mm of pure ozone at standard temperature and pressure (Loschmidt constant 2.6868e25 m-3, CODATA 2018); Seinfeld, J.H. and Pandis, S.N. (2016) Atmospheric Chemistry and Physics, 3rd ed., Wiley, ch. 5: global mean total ozone column about 300 DU |
| `atmosphere.chemistry.photolysis.reference_temperature` | `K` | `298.15; bounds=(248.0, 298.15)` | `fixed` | Reference air temperature T_ref of the photolysis temperature factor R(T) = S(T)/S(T_ref): the photolysis frequencies of the Trebs et al. (2009) J_NO2 relation and of the MCM J(O1D)/J(NO2) ratio are taken as T_ref values. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Tables 6.11c and 6.13b (298 K columns) |
| `atmosphere.chemistry.photolysis.solar_flux_wavelength` | `um` | `[0.2895, 0.2905, 0.2915, 0.2925, 0.2935, 0.2945, 0.2955, 0.2965, 0.2975, 0.2985, 0.2995, 0.3005, 0.3015, 0.3025, 0.3035, 0.3045, 0.3055, 0.3065, 0.3075, 0.3085, 0.3095, 0.3105, ...; bounds=None` | `fixed` | Centres of the 1-nm intervals of the extraterrestrial solar photon flux, 289.5-417.5 nm, written in micrometres (289.5 nm = 0.2895 um). | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.2 (180-400 nm, 1-nm intervals) and Table 6.4a (330-630 nm, 1-nm intervals) |
| `atmosphere.chemistry.photolysis.solar_photon_flux` | `photons cm-2 s-1 um-1` | `[7.1324e+16, 9.1035e+16, 8.8237e+16, 7.9558e+16, 8.1913e+16, 7.8086e+16, 8.5194e+16, 7.875e+16, 7.8657e+16, 7.2414e+16, 7.5552e+16, 6.5774e+16, 7.1579e+16, 7.7085e+16, 9.8165e+1...; bounds=None` | `fixed` | Extraterrestrial solar photon flux averaged over each 1-nm interval of solar_flux_wavelength (Table 6.2, Woods et al. 1996 UARS/ATLAS 2, for 289.5-409.5 nm; Table 6.4a, Neckel and Labs 1984, for 410.5-417.5 nm), per micrometre of wavelength. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.2: irradiances in 1e12 photon cm-2 s-1 nm-1, Woods et al. (1996) J. Geophys. Res. 101, 9541-9569; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences, Springer, Dordrecht, doi:10.1007/978-94-007-2275-0; Table 6.4a: irradiances in 1e12 photon cm-2 s-1 nm-1, Neckel and Labs (1984) Solar Phys. 90, 205-258 |
| `atmosphere.chemistry.photolysis_temperature_mode` | `dimensionless` | `'none'; bounds=None` | `fixed` | Air-temperature dependence of the photolysis frequencies J_NO2 and J(O1D). none (default): both are independent of air temperature; the NO2 quantum yield is 1.00 at 300-398 nm at 248 and 298 K and the NO2 cross-section temperature dependence is minimal (Warneck and Williams 2012 Tables 6.13a/b). spectral_quantum_yield: J_X(T) = J_X(T_ref) R_X(T), R_X(T) = sum_lambda sigma_X(lambda,T) phi_X(lambda,T) F(lambda) / sum_lambda sigma_X(lambda,T_ref) phi_X(lambda,T_ref) F(lambda), with the tables and the reference surface spectrum of atmosphere.chemistry.photolysis.*. | Source: literature; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Table 6.13a (NO2 cross sections, temperature dependence minimal) and Table 6.13b (NO2 quantum yields at 298 and 248 K, Troe 2000); Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Tables 6.11a-c (O3 cross sections at two temperatures; O(1D) quantum yields, Matsumi et al. 2002) |
| `atmosphere.chemistry.temperature_max` | `K` | `400.0; bounds=(300.0, 1000.0)` | `fixed` | Maximum temperature for chemistry validation checks | Source: auto_migrated_parameter_database; Seinfeld & Pandis (2016) |
| `atmosphere.chemistry.temperature_min` | `K` | `200.0; bounds=(200.0, 300.0)` | `fixed` | Minimum temperature for chemistry validation checks | Source: auto_migrated_parameter_database; Seinfeld & Pandis (2016) |
| `atmosphere.chemistry.uv_a_surface_albedo` | `dimensionless` | `0.03; bounds=(0.0, 0.2)` | `fixed` | Site UV-A surface albedo alpha of J_NO2 = (1 + alpha) (B1 G + B2 G^2) (Trebs et al. 2009 Eq. 2); 0.03 is typical of grassland and vegetation (2-3 percent in the UV-A). The relation is not to be used for alpha > 0.2 (snow, ice, sand), which is the upper bound. | Source: literature; Trebs, I., Bohn, B., Ammann, C., Rummel, U., Blumthaler, M., Konigstedt, R., Meixner, F.X., Fan, S. and Andreae, M.O. (2009) Relationship between the NO2 photolysis frequency and the solar global irradiance, Atmos. Meas. Tech. 2, 725-739, doi:10.5194/amt-2-725-2009; Sect. 4.6 and Table 1: UV-A surface albedo of vegetation about 2-3 percent, grassland 0.03 (Feister and Grewe 1995); the function should not be used for alpha > 0.2; Warneck, P. and Williams, J. (2012) The Atmospheric Chemist's Companion, Springer, doi:10.1007/978-94-007-2275-0; Table 6.7: ground-level photolysis frequencies for overhead sun and surface albedo 0.03 (Jacobson 2005) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | canonical UpdateItem ACCUMULATE increments for species and REPLACE observations for diagnostics |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\chemistry.md`.

### References

- Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics (FunctorMetadata, certificate)
- Finlayson-Pitts, B.J. & Pitts, J.N. (2000). Chemistry of Upper and Lower Atmosphere (FunctorMetadata, certificate)
- Atkinson, R. & Arey, J. (2003). Atmospheric degradation of volatile organic compounds (FunctorMetadata, certificate)
- Jenkin, M.E. & Clemitshaw, K.C. (2000). Ozone and secondary photochemical pollutants (FunctorMetadata, certificate)
- Warneck, P. & Williams, J. (2012). The Atmospheric Chemist's Companion, Springer (FunctorMetadata, certificate)
- Matsumi, Y. et al. (2002). J. Geophys. Res. 107(D3), 4024 (FunctorMetadata, certificate)
- Saunders, S.M. et al. (2003). Atmos. Chem. Phys. 3, 161-180 (FunctorMetadata, certificate)
- Bodhaine, B.A. et al. (1999). J. Atmos. Oceanic Technol. 16, 1854-1861 (FunctorMetadata, certificate)
- Jenkin, M.E. et al. (2015). Atmos. Chem. Phys. 15, 11433-11459 (FunctorMetadata)
- Peeters, J. et al. (2014). J. Phys. Chem. A 118, 8625-8643 (FunctorMetadata, certificate)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- Sander, S.P. et al. (2006). JPL Publication 06-2, Evaluation Number 15 (certificate)
- Trebs, I. et al. (2009). Atmos. Meas. Tech. 2, 725-739, doi:10.5194/amt-2-725-2009 (certificate)

(class-atmosphere-chemistry-chemistry-functor-closed-constant-volume)=
## ChemistryFunctor

Reduced-form atmospheric chemistry functor using box model approach.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.chemistry.ChemistryFunctor` |
| Source | `relic\scientific\atmosphere\chemistry.py:92` |
| Runtime name | `atmospheric_chemistry` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `field` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_5` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
ChemistryFunctor(name: str = 'atmospheric_chemistry', *, physical_boundary: Literal['open_mechanism', 'closed_constant_volume'], config: Mapping[str, Any] | None = None, **kwargs)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'atmospheric_chemistry'` | Functor identity used by registration, provenance, and dependency resolution. |
| `physical_boundary` | `typing.Literal['open_mechanism', 'closed_constant_volume']` | `required` | Constructor control for physical boundary; see the class runtime contract and ParameterManager table below. |
| `config` | <code>typing.Mapping[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

The runtime metadata below is introspected from the documented configuration `physical_boundary='closed_constant_volume'`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `box_no_amount` | state | `mol` | n_NO molar inventory, component 1 of the GENERIC state x |
| `box_no2_amount` | state | `mol` | n_NO2 molar inventory, component 2 of the GENERIC state x |
| `box_o3_amount` | state | `mol` | n_O3 molar inventory, component 3 of the GENERIC state x |
| `box_o2_amount` | state | `mol` | n_O2 molar inventory, component 4 of the GENERIC state x |
| `box_internal_energy` | state | `J` | U total internal energy including NASA7 formation terms, component 5 of x and the energy functional E |
| `box_volume` | forcing | `m3` | fixed box volume V entering p_i = n_i R T / V; never advanced |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `box_no_amount` | state | `mol` | decided at run time by the returned write mode | n_NO molar inventory, component 1 of the GENERIC state x |
| `box_no2_amount` | state | `mol` | decided at run time by the returned write mode | n_NO2 molar inventory, component 2 of the GENERIC state x |
| `box_o3_amount` | state | `mol` | decided at run time by the returned write mode | n_O3 molar inventory, component 3 of the GENERIC state x |
| `box_o2_amount` | state | `mol` | decided at run time by the returned write mode | n_O2 molar inventory, component 4 of the GENERIC state x |
| `box_internal_energy` | state | `J` | decided at run time by the returned write mode | U total internal energy including NASA7 formation terms, component 5 of x and the energy functional E |
| `diagnostics.chemistry.box_temperature` | diagnostic | `K` | diagnostics | canonical derived temperature observation |
| `diagnostics.chemistry.box_pressure` | diagnostic | `Pa` | diagnostics | canonical derived pressure observation |
| `diagnostics.chemistry.box_entropy` | diagnostic | `J K-1` | diagnostics | canonical entropy observation |
| `diagnostics.chemistry.box_entropy_production_rate` | diagnostic | `J K-1 s-1` | diagnostics | canonical entropy-production observation |
| `diagnostics.chemistry.box_reaction_affinity` | diagnostic | `J mol-1` | diagnostics | canonical affinity observation |
| `diagnostics.chemistry.box_reaction_rate` | diagnostic | `mol s-1` | diagnostics | canonical reaction-rate observation |
| `diagnostics.chemistry.box_reaction_power` | diagnostic | `W` | diagnostics | canonical affinity-rate power observation |
| `diagnostics.chemistry.box_rate_constant_no_o3` | diagnostic | `cm3 molecule-1 s-1` | diagnostics | canonical rate-coefficient observation |

### Governing equations and canonical operator

**Literature or process form**

$A = - \nu ^{\mathsf{T}} \mu$; $\mu_{i} = h_{i} ( T ) - T s_{i} ( T ) + R T \ln ( p_{i} / \mathrm{p0} )$; $\sigma = A r / T \geq 0$; $\dot{x} = L_{h} e_{E} + M_{h} e_{S}$ with $L_{h} = 0$ and $M_{h} = ( r T / A ) \nu \nu ^{\mathsf{T}} ( \mathrm{Grmela} \mathrm{and} \mathrm{Ottinger} 1997 ; \mathrm{Mielke} 2011 ; \mathrm{Kondepudi} \mathrm{and} \mathrm{Prigogine} 1998 )$

**Canonical RELIC structural form**

$\dot{x} = L_{h} e_{E} + M_{h} e_{S}$, $L_{h} = 0 ( \mathrm{zero}_{\mathrm{field}} )$, $M_{h} = ( r T / A ) \nu \nu ^{\mathsf{T}}$, absorbed reaction with $M_{h} e_{S} = N r \mathrm{checked} \mathrm{against} \mathrm{the}$ declared N r, $M_{h} e_{E} = 0$

| Entry | Meaning |
| --- | --- |
| `L_h` | zero Poisson operator declared through $\mathrm{poisson}_{\mathrm{operator}} = \mathrm{zero}_{\mathrm{field}}$; the Jacobi identity holds on the canonical zero-field basis verified by IPHSStructureSpec.validate |
| `M_h` | (r T / A) nu nu^T, symmetric positive semidefinite for $A > 0$ and $r \geq 0$, with a zero energy row and column so that $M_{h} e_{E} = 0$ |
| `G_h` | none, closed box without ports |
| `N` | $\nu = ( - 1 , + 1 , - 1 , + 1 , 0 ) \mathrm{reshaped}$ to 5 by 1; $\mathrm{reaction}_{\mathrm{representation}} = \mathrm{absorbed}$, so N r is not added a second time |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | $E = U$ and S(U, n, V) evaluated by standard_molar_properties, ideal_gas_mixture_properties, and temperature_from_internal_energy on the same autograd state graph |
| `effort_variable` | $e_{E} = \nabla ( E ) = ( 0 , 0 , 0 , 0 , 1 )$ and $e_{S} = \nabla ( S ) = ( - \mu / T , 1 / T ) \mathrm{obtained}$ by torch.autograd.grad inside canonical_drift |
| `state_advancer` | $\Delta x = \mathrm{d}t \dot{x}$ with $\dot{x} = L_{h} e_{E} + M_{h} e_{S} = \nu r$; the actual tensor candidate x + Delta x must retain strictly positive species inventories, valid $T ( U, n )$ and ideal-mixture properties, and $A > 0$ before emission |
| `explicit_stoichiometric_usage` | forward declares $\mathrm{stoichiometric}_{N} = \nu \mathrm{reshaped}$ to 5 by 1 with $\mathrm{reaction}_{\mathrm{representation}} = \mathrm{absorbed} \mathrm{in} \mathrm{IPHSStructureSpec}$, and passes the same N, reaction rate, $C_{N}$, $C_{O}$, and affinity to advance_reaction_network with $\mathrm{d}t = \mathrm{None}$ for the audits only. |
| `runtime_usage` | ChemistryFunctor.forward evaluates one GENERIC absorbed-reaction path through canonical_drift, verifies $M_{h} e_{S} = N r$ to reduction roundoff, validates the bound pre-step runtime point at $\mathrm{required}_{\mathrm{level}} = 5$, and audits $C_{N} \nu = 0$, $C_{O} \nu = 0$, and $A r \geq 0$; before publishing the audit or emitting five ACCUMULATE increments plus eight REPLACE diagnostics, the same pure thermodynamic-point owner checks the actual tensor Euler candidate for $n_{i} > 0$, valid $T ( U, n )$, and $A > 0$, without a second drift or state advance. Discrete entropy monotonicity, stiff-stability and accuracy are not certified. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

Executable `ports()` owner: `ChemistryFunctor`. The default instance returns no fixed PortSpec entries; its ports are configuration-conditional or observational.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `atmosphere.chemistry.k_no_o3_298` | `cm3 molecule-1 s-1` | `1.8e-14; bounds=(1.4e-14, 2e-14)` | `fixed` | NO + O3 -> NO2 + O2 rate coefficient at the Arrhenius reference temperature (IUPAC preferred k298). | Source: literature; Atkinson et al. (2004) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume I, Atmos. Chem. Phys. 4, 1461-1738, doi:10.5194/acp-4-1461-2004; datasheet I.A3.54: k = 1.8e-14 cm3 molecule-1 s-1 at 298 K, Delta log k = +-0.08; k = 1.4e-12 exp(-1310/T) over 195-308 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): NO + O3 = NO2, 1.4e-12 exp(-1310/T) |
| `atmosphere.chemistry.k_no_o3_activation_energy` | `J mol-1` | `10891.9; bounds=(9229.1, 12554.8)` | `fixed` | Activation energy Ea = (E/R) R of NO + O3 in f(T) = f(T_ref) exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; E/R = 1310 K (IUPAC I.A3.54, MCM v3.3.1); positive: the rate increases with temperature. Bounds are the IUPAC reliability E/R +- 200 K. | Source: literature; Atkinson et al. (2004) Evaluated kinetic and photochemical data for atmospheric chemistry: Volume I, Atmos. Chem. Phys. 4, 1461-1738, doi:10.5194/acp-4-1461-2004; datasheet I.A3.54: k = 1.4e-12 exp(-1310/T), Delta(E/R) = +-200 K; MCM v3.3.1 (Master Chemical Mechanism, http://mcm.york.ac.uk; Jenkin et al. 2015, Atmos. Chem. Phys. 15, 11433-11459, doi:10.5194/acp-15-11433-2015): NO + O3 = NO2, 1.4e-12 exp(-1310/T) |
| `atmosphere.chemistry.nasa7.a1` | `dimensionless` | `[[4.21859896, 3.26071234], [3.94403907, 4.88474429], [3.40738221, 12.3302914], [3.78245636, 3.66096083]]; bounds=(3.26071234, 12.3302914)` | `fixed` | NASA7 coefficient a1; rows follow species_order, columns are low/high temperature intervals. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.nasa7.a2` | `K-1` | `[[-0.00463988124, 0.00119101135], [-0.00158547444, 0.00217241639], [0.00205379063, -0.0119324783], [-0.00299673415, 0.000656365523]]; bounds=(-0.0119324783, 0.00217241639)` | `fixed` | NASA7 coefficient a2; rows follow species_order, columns are low/high temperature intervals. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.nasa7.a3` | `K-2` | `[[1.10443049e-05, -4.29122646e-07], [1.66578984e-05, -8.2807902e-07], [1.38486052e-05, 7.98741278e-06], [9.847302e-06, -1.41149485e-07]]; bounds=(-8.2807902e-07, 1.66578984e-05)` | `fixed` | NASA7 coefficient a3; rows follow species_order, columns are low/high temperature intervals. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.nasa7.a4` | `K-3` | `[[-9.34055507e-09, 6.94481463e-11], [-2.04754478e-08, 1.57477293e-10], [-2.23311542e-08, -1.77194552e-09], [-9.68129508e-09, 2.05797658e-11]]; bounds=(-2.23311542e-08, 1.57477293e-10)` | `fixed` | NASA7 coefficient a4; rows follow species_order, columns are low/high temperature intervals. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.nasa7.a5` | `K-4` | `[[2.80554874e-12, -4.03295681e-15], [7.83503265e-12, -1.05110549e-14], [9.76073226e-12, 1.26075824e-13], [3.24372836e-12, -1.29913248e-15]]; bounds=(-1.05110549e-14, 9.76073226e-12)` | `fixed` | NASA7 coefficient a5; rows follow species_order, columns are low/high temperature intervals. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.nasa7.a6` | `K` | `[[9845.09964, 9921.43132], [2896.59865, 2316.48462], [15864.4979, 12675.5831], [-1063.94356, -1215.97725]]; bounds=(-1215.97725, 15864.4979)` | `fixed` | NASA7 coefficient a6; rows follow species_order, columns are low/high temperature intervals. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.nasa7.a7` | `dimensionless` | `[[2.28061001, 6.36900518], [6.31196225, -0.117357075], [8.2824758, -40.8823374], [3.65767573, 3.41536184]]; bounds=(-40.8823374, 8.2824758)` | `fixed` | NASA7 coefficient a7; rows follow species_order, columns are low/high temperature intervals. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.nasa7.model` | `dimensionless` | `'NASA7'; bounds=('NASA7', 'NASA7')` | `fixed` | Thermodynamic polynomial family; NASA9 coefficients are not interchangeable. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.nasa7.species_order` | `dimensionless` | `['NO', 'NO2', 'O3', 'O2']; bounds=('NO', 'O2')` | `fixed` | Species order for the NASA7 coefficient row axis. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.nasa7.temperature_ranges` | `K` | `[200.0, 1000.0, 6000.0]; bounds=(200.0, 6000.0)` | `fixed` | Shared low/switch/high temperature limits of the frozen NASA7 fits. | Source: NASA7 gas fits, NASA TM-4513 dataset; McBride, Gordon and Reno (1993), NASA TM-4513, NASA7 thermodynamic polynomials. |
| `atmosphere.chemistry.temperature_max` | `K` | `400.0; bounds=(300.0, 1000.0)` | `fixed` | Maximum temperature for chemistry validation checks | Source: auto_migrated_parameter_database; Seinfeld & Pandis (2016) |
| `atmosphere.chemistry.temperature_min` | `K` | `200.0; bounds=(200.0, 300.0)` | `fixed` | Minimum temperature for chemistry validation checks | Source: auto_migrated_parameter_database; Seinfeld & Pandis (2016) |
| `computation.solvers.bracketed_bisection.atol` | `dimensionless` | `1e-10; bounds=(0.0, 0.0001)` | `fixed` | Absolute interval tolerance in the solved variable units | Source: computation.solvers.bracketed_bisection; Burden, R.L. and Faires, J.D. (2010). Numerical Analysis. |
| `computation.solvers.bracketed_bisection.max_iterations` | `dimensionless` | `80; bounds=(1.0, 512.0)` | `fixed` | Maximum bracket contractions for independent scalar roots | Source: computation.solvers.bracketed_bisection; Burden, R.L. and Faires, J.D. (2010). Numerical Analysis. |
| `computation.solvers.bracketed_bisection.rtol` | `dimensionless` | `1e-10; bounds=(0.0, 0.0001)` | `fixed` | Relative interval tolerance for bracketed scalar roots | Source: computation.solvers.bracketed_bisection; Burden, R.L. and Faires, J.D. (2010). Numerical Analysis. |
| `validation.iphs.runtime_point_tolerance` | `dimensionless` | `1e-10; bounds=(1e-16, 1e-06)` | `fixed` | Absolute tolerance shared by the skew, positive-semidefinite, degeneracy, entropy-production, and reaction-closure audits of one IPHS runtime point (dimensionless residual scale) | Source: RELIC parameter database; Grmela, M. & Ottinger, H.C. (1997). Phys. Rev. E 56, 6620-6632; van der Schaft, A. & Jeltsema, D. (2014). Port-Hamiltonian Systems Theory, Foundations and Trends in Systems and Control |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | canonical UpdateItem ACCUMULATE increments for the five state components and REPLACE observations for diagnostics |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\chemistry.md`.

### References

- Grmela, M. & Ottinger, H.C. (1997). Phys. Rev. E 56, 6620-6632 (FunctorMetadata, certificate)
- Ottinger, H.C. (2005). Beyond Equilibrium Thermodynamics, Wiley (FunctorMetadata, certificate)
- Mielke, A. (2011). Nonlinearity 24, 1329-1346 (FunctorMetadata, certificate)
- Kondepudi, D. & Prigogine, I. (1998). Modern Thermodynamics, Wiley (FunctorMetadata, certificate)
- McBride, B.J., Gordon, S. & Reno, M.A. (1993). NASA TM-4513 (FunctorMetadata, certificate)
- Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics (FunctorMetadata, certificate)
- Atkinson, R. et al. (2004). Atmos. Chem. Phys. 4, 1461-1738 (FunctorMetadata, certificate)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)

(class-relic-scientific-atmosphere-environmental-forcing-environmentalforcingfunctor)=
## EnvironmentalForcingFunctor

Derive atmospheric VPD from required external forcing inputs.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.environmental_forcing.EnvironmentalForcingFunctor` |
| Source | `relic\scientific\atmosphere\environmental_forcing.py:40` |
| Runtime name | `environmental_forcing` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `port` |
| Spatial scale | `grid_cell` |
| Variable kind | `forcing` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
EnvironmentalForcingFunctor(name: str = 'environmental_forcing', config: dict[str, Any] | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'environmental_forcing'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | required external air-temperature forcing [K] |
| `relative_humidity` | forcing | `1` | required external relative-humidity forcing [dimensionless] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `vapor_pressure_deficit` | forcing | `Pa` | decided at run time by the returned write mode | sole canonical output in Pa |

### Governing equations and canonical operator

**Literature or process form**

$$
\mathrm{VPD}_{\mathrm{Pa}} = \mathrm{kPa}_{\mathrm{to},\mathrm{Pa}} ( \max ( \mathrm{es0} \cdot \exp ( a \cdot T_{C} / ( T_{C} + b ) ) \cdot ( 1 - \mathrm{RH} ) , 0 ) ); T_{C} = T_{K} - \mathrm{T}_{0}
$$

**Canonical RELIC structural form**

$y_{h} = G_{h} u_{h}$ through the declared external forcing and observation port map

| Entry | Meaning |
| --- | --- |
| `G_h` | declared port map |
| `H_h` | positive port metric or conversion Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | external air-temperature and relative-humidity forcing port basis |
| `effort_variable` | validated external meteorological forcing values |
| `state_advancer` | $y_{h} = G_{h} u_{h}$ as a derived VPD observation with no local storage advance |
| `runtime_usage` | EnvironmentalForcingFunctor.forward evaluates one Magnus VPD path, converts kPa to Pa through UnitConverter, and returns only vapor_pressure_deficit. |

### Composition channels and executable PortSpec API

Certificate composition channels: `external_meteorological_forcing_port`, `vapor_pressure_deficit_observation_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `atmosphere.environmental_forcing.vpd.es_coefficient` | `kPa` | `0.6108; bounds=(0.5, 0.7)` | `fixed` | Coefficient for Magnus-type saturation vapor pressure (FAO-56) | Source: literature; Allen et al. (1998) FAO-56; Monteith & Unsworth (2013) |
| `atmosphere.environmental_forcing.vpd.a_coefficient` | `dimensionless` | `17.27; bounds=(15.0, 18.5)` | `fixed` | Slope coefficient a in Magnus-type saturation vapor pressure formula | Source: literature; Allen et al. (1998) FAO-56; Monteith & Unsworth (2013) |
| `atmosphere.environmental_forcing.vpd.b_coefficient` | `degC` | `237.3; bounds=(230.0, 240.0)` | `fixed` | Denominator coefficient b in Magnus-type saturation vapor pressure formula | Source: literature; Allen et al. (1998) FAO-56; Monteith & Unsworth (2013) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{True} )$ with unit Pa and source_functor provenance |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\atmosphere_environmental_forcing.md`.

### References

- Allen et al. (1998) FAO Irrigation and Drainage Paper 56 (FunctorMetadata, certificate)
- Monteith & Unsworth (2008) Principles of Environmental Physics (FunctorMetadata, certificate)
- Campbell & Norman (1998) Environmental Biophysics (FunctorMetadata, certificate)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)

(class-relic-scientific-atmosphere-radiation-energy-conservation-radiationenergyconservationvalidator)=
## RadiationEnergyConservationValidator

Audit the declared atmospheric radiation energy balance.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.radiation_energy_conservation.RadiationEnergyConservationValidator` |
| Source | `relic\scientific\atmosphere\radiation_energy_conservation.py:47` |
| Runtime name | `radiation_conservation_audit` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `observation` |
| Spatial scale | `point` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
RadiationEnergyConservationValidator(name: str = 'radiation_conservation_audit', config: Dict | None = None, parameter_manager: Any | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'radiation_conservation_audit'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `parameter_manager` | <code>typing.Any &#124; None</code> | `None` | Constructor control for parameter manager; see the class runtime contract and ParameterManager table below. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `shortwave_down` | flux | `W m-2` | shortwave_down mapped to canonical Phase 8 cochain basis |
| `shortwave_net` | flux | `W m-2` | shortwave_net mapped to canonical Phase 8 cochain basis |
| `longwave_down` | flux | `W m-2` | longwave_down mapped to canonical Phase 8 cochain basis |
| `longwave_up` | forcing | `W m-2` | total outgoing longwave observation, including emitted and reflected incident components |
| `longwave_net` | flux | `W m-2` | longwave_net mapped to canonical Phase 8 cochain basis |
| `net_radiation` | flux | `W m-2` | net_radiation optional input read from the state when present |
| `shortwave_reflected` | forcing | `W m-2` | shortwave_reflected optional input read from the state when present |
| `surface_albedo` | diagnostic | `dimensionless` | surface_albedo optional input read from the state when present |
| `surface_emissivity` | forcing | `1` | surface_emissivity optional input read from the state when present |
| `surface_temperature` | forcing | `K` | surface_temperature optional input read from the state when present |
| `air_temperature` | forcing | `K` | air_temperature optional input read from the state when present |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `diagnostics.energy_conservation_report` | output port | `non-quantity` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |

### Governing equations and canonical operator

**Literature or process form**

$$
E_{\mathrm{balance}} = \lvert E_{\mathrm{in}} - ( \mathrm{SW}_{\mathrm{net}} + \mathrm{LW}_{\mathrm{net}} + \mathrm{SW}_{\mathrm{reflected}} + \mathrm{LW}_{\mathrm{up}} ) \rvert / \max ( E_{\mathrm{in}} , \varepsilon )
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
| `runtime_usage` | RadiationEnergyConservationValidator.forward applies the canonical observation or diagnostic operator and emits validation diagnostics through the production output path. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `validation.conservation.radiation.canopy_conservation_tolerance` | `dimensionless` | `0.01; bounds=(0.001, 0.05)` | `fixed` | Relative tolerance for canopy radiation conservation checks | Source: radiation_validator_defaults; Bonan (2019). Climate Change and Terrestrial Ecosystem Modeling |
| `validation.conservation.radiation.enable_canopy_layer_check` | `dimensionless` | `True; bounds=(0.0, 100.0)` | `fixed` | Enable detailed canopy layer energy conservation checks | Source: radiation_validator_defaults; Bonan (2019). Climate Change and Terrestrial Ecosystem Modeling |
| `validation.conservation.radiation.max_albedo` | `dimensionless` | `1.0; bounds=(0.8, 1.0)` | `fixed` | Maximum physically plausible albedo | Source: radiation_validator_defaults; Campbell & Norman (1998). Environmental Biophysics |
| `validation.conservation.radiation.max_emissivity` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Maximum physically plausible surface emissivity | Source: radiation_validator_defaults; Monteith & Unsworth (2013). Principles of Environmental Physics |
| `validation.conservation.radiation.max_temperature` | `K` | `400.0; bounds=(320.0, 500.0)` | `fixed` | Maximum surface temperature for radiation validation range | Source: radiation_validator_defaults; Energy balance constraints for terrestrial surfaces |
| `validation.conservation.radiation.min_albedo` | `dimensionless` | `0.0; bounds=(0.0, 0.2)` | `fixed` | Minimum physically plausible albedo | Source: radiation_validator_defaults; Campbell & Norman (1998). Environmental Biophysics |
| `validation.conservation.radiation.min_emissivity` | `dimensionless` | `0.7; bounds=(0.5, 0.8)` | `fixed` | Minimum physically plausible surface emissivity | Source: radiation_validator_defaults; Monteith & Unsworth (2013). Principles of Environmental Physics |
| `validation.conservation.radiation.min_temperature` | `K` | `200.0; bounds=(180.0, 260.0)` | `fixed` | Minimum surface temperature for radiation validation range | Source: radiation_validator_defaults; Energy balance constraints for terrestrial surfaces |
| `validation.conservation.radiation.minor_tolerance` | `W m-2` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Absolute tolerance for minor radiation energy imbalances | Source: radiation_validator_defaults; Campbell & Norman (1998). Environmental Biophysics |
| `validation.conservation.radiation.moderate_tolerance` | `W m-2` | `1.0; bounds=(0.1, 10.0)` | `fixed` | Absolute tolerance for moderate radiation energy imbalances | Source: radiation_validator_defaults; Monteith & Unsworth (2013). Principles of Environmental Physics |
| `validation.conservation.radiation.relative_minor` | `dimensionless` | `0.001; bounds=(0.0001, 0.01)` | `fixed` | Relative tolerance for minor radiation energy imbalance (fraction of incoming energy) | Source: radiation_validator_defaults; Numerical stability requirements for energy balance models |
| `validation.conservation.radiation.relative_moderate` | `dimensionless` | `0.01; bounds=(0.001, 0.05)` | `fixed` | Relative tolerance for moderate radiation energy imbalance | Source: radiation_validator_defaults; Numerical stability requirements for energy balance models |
| `validation.conservation.radiation.relative_severe` | `dimensionless` | `0.05; bounds=(0.01, 0.2)` | `fixed` | Relative tolerance beyond which radiation imbalance is severe | Source: radiation_validator_defaults; Numerical stability requirements for energy balance models |
| `validation.conservation.radiation.severe_tolerance` | `W m-2` | `5.0; bounds=(1.0, 50.0)` | `fixed` | Absolute tolerance beyond which radiation imbalance is severe | Source: radiation_validator_defaults; Bonan (2019). Climate Change and Terrestrial Ecosystem Modeling |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\radiation.md`.

### References

- Campbell & Norman (1998) Environmental Biophysics (FunctorMetadata, certificate)
- Monteith & Unsworth (2013) Principles of Environmental Physics (FunctorMetadata, certificate)
- Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling (FunctorMetadata, certificate)
- Nicodemus (1965), doi:10.1364/AO.4.000767 (FunctorMetadata)
- Noah-MP Technical Note, Eqs. 3.6.330 and 3.6.416-417 (FunctorMetadata, certificate)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- Nicodemus (1965), Applied Optics 4, 767-775, doi:10.1364/AO.4.000767 (certificate)

(class-relic-scientific-atmosphere-radiation-radiationfield)=
## RadiationField

Physical Field Implementation for Atmospheric Radiation.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.radiation.RadiationField` |
| Source | `relic\scientific\atmosphere\radiation.py:59` |
| Runtime name | `radiation_field` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `field` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
RadiationField(device: torch.device | None = None, **kwargs: Any)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `kwargs` | `typing.Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `grid_coordinates` | forcing | `m` | grid_coordinates mapped to canonical Phase 8 cochain basis |
| `radiation_field` | state | `W m-2` | radiation_field mapped to canonical Phase 8 cochain basis |
| `source_coordinates` | forcing | `m` | source_coordinates mapped to canonical Phase 8 cochain basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `radiation_field` | state | `W m-2` | decided at run time by the returned write mode | radiation_field mapped to canonical Phase 8 cochain basis |
| `diagnostics.radiation_field.gradient` | output port | `W m-3` | diagnostics | explicit optional spatial-gradient diagnostic |
| `diagnostics.radiation_field.clipping_residual` | output port | `W m-2` | diagnostics | explicit bounded-projection residual |
| `diagnostics.radiation_field.query_clipping_residual` | output port | `W m-2` | diagnostics | explicit interpolation-query clipping residual |

### Governing equations and canonical operator

**Literature or process form**

$R ( x, y, t )$ field interpolation and evolution; $\nabla R = ( \frac{\partial R}{\partial x} , \frac{\partial R}{\partial y} )$

**Canonical RELIC structural form**

$\dot{R} = M_{h} e_{R} + G_{h} u_{R}$ with $M_{h} = \operatorname{diag} ( R_{\mathrm{max}} - R_{\mathrm{min}} ) \geq 0$ and boundary forcing routed through radiation_field_boundary_port

| Entry | Meaning |
| --- | --- |
| `J_h` | zero reversible interconnection for this reduced field container |
| `M_h` | positive semidefinite bounded radiation field metric from $R_{\mathrm{max}}$ - $R_{\mathrm{min}}$ |
| `G_h` | radiation_field_boundary_port boundary map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | radiation_field state supplied by the existing simulator and StateManager transaction |
| `state_advancer` | External StateManager transaction owns radiation_field evolution; forward validates, bounds, and interpolates that exact state without a second source-term advance. |
| `runtime_usage` | RadiationField.forward requires finite grid_coordinates and radiation_field cochains, rejects nonpositive dt, exposes bounded interpolation and clipping residuals, and does not claim an executable level-4 IPHS state advancer. |

### Composition channels and executable PortSpec API

Certificate composition channels: `radiation_field_boundary_port`

Executable `ports()` owner: `RadiationField`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `radiation_field_boundary_port` | `radiation_field [W m-2]` | `radiation_boundary_flux [W m-2]` | `W m-2` | into_system; radiation |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `atmosphere.radiation.field_bounds.generic_min_radiation` | `W m-2` | `-100.0; bounds=(-300.0, 0.0)` | `fixed` | Lower projection bound R_min of RadiationField; forward and get_value_at clamp the field to [R_min, R_max], route the clipped part to diagnostics.radiation_field.clipping_residual and use R_max - R_min as the Hodge metric | Source: model_configuration; Campbell & Norman (1998). Environmental Biophysics |
| `atmosphere.radiation.field_bounds.generic_max_radiation` | `W m-2` | `1200.0; bounds=(800.0, 1500.0)` | `fixed` | Upper projection bound R_max of RadiationField; forward and get_value_at clamp the field to [R_min, R_max], route the clipped part to diagnostics.radiation_field.clipping_residual and use R_max - R_min as the Hodge metric | Source: model_configuration; Campbell & Norman (1998). Environmental Biophysics |
| `atmosphere.radiation.field_interpolation.idw_power` | `dimensionless` | `1.0; bounds=(1.0, 4.0)` | `fixed` | Distance exponent for canonical inverse-distance interpolation | Source: literature; Shepard (1968). A two-dimensional interpolation function for irregularly-spaced data |
| `atmosphere.radiation.field_interpolation.coincident_tolerance` | `m` | `1e-09; bounds=(1e-12, 0.001)` | `fixed` | Distance tolerance for exact coincident radiation samples | Source: numerical_method; Shepard (1968). A two-dimensional interpolation function for irregularly-spaced data |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | radiation_field uses UpdateItem(REPLACE); diagnostics use explicit REPLACE updates |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\radiation.md`.

### References

- Brutsaert, W. (1975). On a derivable formula for long-wave radiation from clear skies. (FunctorMetadata, certificate)
- Campbell, G.S. and Norman, J.M. (1998). An Introduction to Environmental Biophysics. (FunctorMetadata, certificate)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)

(class-relic-scientific-atmosphere-radiation-radiationfunctor)=
## RadiationFunctor

Atmospheric radiation transfer functor for surface energy balance.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.radiation.RadiationFunctor` |
| Source | `relic\scientific\atmosphere\radiation.py:846` |
| Runtime name | `atmospheric_radiation` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `field` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
RadiationFunctor(name: str = 'atmospheric_radiation', config: Dict[str, Any] | None = None, **kwargs: Any)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'atmospheric_radiation'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `typing.Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `air_temperature` | forcing | `K` | air_temperature mapped to canonical Phase 8 cochain basis |
| `incoming_shortwave_potential` | forcing | `W m-2` | incoming_shortwave_potential mapped to canonical Phase 8 cochain basis |
| `cos_zenith` | forcing | `1` | cos_zenith mapped to the declared solar geometry input |
| `surface_temperature` | forcing | `K` | surface_temperature mapped to the surface thermal state |
| `relative_humidity` | forcing | `1` | relative_humidity mapped to the atmospheric moisture state |
| `surface_albedo` | diagnostic | `dimensionless` | surface_albedo mapped to the surface optical state |
| `surface_emissivity` | forcing | `1` | surface_emissivity mapped to the surface optical state |
| `cloud_fraction` | forcing | `1` | cloud_fraction mapped to the atmospheric cloud state |
| `leaf_area_index` | state | `m2 m-2` | conditionally required scalar canopy cochain when canopy transfer is enabled |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `shortwave_down` | flux | `W m-2` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `longwave_down` | flux | `W m-2` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `net_radiation` | flux | `W m-2` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `shortwave_net` | flux | `W m-2` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `longwave_net` | flux | `W m-2` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `atmospheric_emissivity` | diagnostic | `dimensionless` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `ppfd` | flux | `umol photons m-2 s-1` | decided at run time by the returned write mode | canonical photon-flux output derived unconditionally from shortwave_down |
| `diagnostics.radiation.par_energy` | output port | `W m-2` | diagnostics | unconditional read-only PAR energy diagnostic |
| `diagnostics.radiation.par_direct` | output port | `W m-2` | diagnostics | unconditional read-only direct PAR diagnostic |
| `diagnostics.radiation.par_diffuse` | output port | `W m-2` | diagnostics | unconditional read-only diffuse PAR diagnostic |
| `diagnostics.radiation.ppfd_direct` | output port | `umol photons m-2 s-1` | diagnostics | unconditional read-only direct photon-flux diagnostic |
| `diagnostics.radiation.ppfd_diffuse` | output port | `umol photons m-2 s-1` | diagnostics | unconditional read-only diffuse photon-flux diagnostic |
| `diagnostics.radiation.diffuse_fraction` | output port | `dimensionless` | diagnostics | unconditional read-only diffuse fraction diagnostic |
| `diagnostics.radiation.longwave_up` | output port | `W m-2` | diagnostics | unconditional read-only total emitted plus reflected upward longwave diagnostic |
| `diagnostics.radiation.longwave_emitted` | output port | `W m-2` | diagnostics | unconditional read-only surface-emitted longwave diagnostic |
| `diagnostics.radiation.longwave_reflected` | output port | `W m-2` | diagnostics | unconditional read-only reflected incident longwave diagnostic |
| `diagnostics.radiation.surface_albedo_used` | output port | `dimensionless` | diagnostics | unconditional read-only optical-state diagnostic |
| `diagnostics.radiation.surface_emissivity_used` | output port | `dimensionless` | diagnostics | unconditional read-only optical-state diagnostic |
| `diagnostics.radiation.cos_zenith_used` | output port | `dimensionless` | diagnostics | unconditional read-only solar-geometry diagnostic |
| `diagnostics.radiation.vapor_pressure_calculated` | output port | `Pa` | diagnostics | unconditional read-only actual-vapor-pressure diagnostic used by Brutsaert emissivity; it is not VPD |
| `diagnostics.radiation.shortwave_clipping_residual` | output port | `W m-2` | diagnostics | unconditional diagnostic residual with no state write |
| `diagnostics.radiation.emissivity_clipping_residual` | output port | `dimensionless` | diagnostics | unconditional diagnostic residual with no state write |
| `fpar` | diagnostic | `dimensionless` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `apar_layer` | flux | `W m-2` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.radiation.transmission_direct` | output port | `W m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.radiation.transmission_diffuse` | output port | `W m-2` | diagnostics | Output returned by forward(); the certificate gives no description. |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Input declaration difference: `leaf_area_index`.
Output declaration difference: `apar_layer`, `diagnostics.radiation.transmission_diffuse`, `diagnostics.radiation.transmission_direct`, `fpar`.
:::

### Governing equations and canonical operator

**Literature or process form**

$R_{\mathrm{net}} = \mathrm{SW}_{\mathrm{net}} + \mathrm{LW}_{\mathrm{net}}$; $\mathrm{PPFD} = \mathrm{SW}_{\mathrm{down}} \cdot f_{\mathrm{PAR}} \cdot c_{\mathrm{PPFD}}$; $k_{T} = \mathrm{SW}_{\mathrm{down}} / \mathrm{SW}_{\mathrm{ext},h}$; Erbs $f_{\mathrm{dif}} = 1 - 0.09 k_{T}$ for $k_{T} \leq 0.22$, fourth-order polynomial for $0.22 < k_{T} \leq 0.80$, and 0.165 above 0.80

**Canonical RELIC structural form**

Level-2 typed boundary wrapper over $R_{\mathrm{net}} = \mathrm{SW}_{\mathrm{net}} + \mathrm{LW}_{\mathrm{net}}$; no Hamiltonian storage, Onsager dissipation, or conjugate power port is asserted

| Entry | Meaning |
| --- | --- |
| `boundary_identity` | $R_{\mathrm{net}} = \mathrm{SW}_{\mathrm{net}} + \mathrm{LW}_{\mathrm{net}}$ is validated as an open energy-flux identity |
| `iphs_structure` | No H, $J_{h}$, $M_{h}$, $G_{h}$, entropy-production, or power-port operator is claimed at level 2 |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | typed open radiative-boundary flux cochain |
| `effort_variable` | not declared because no stored-energy functional or conjugate effort is available |
| `state_advancer` | The existing simulator and StateManager own state advancement; forward evaluates the declared radiative boundary flux without a second state-update path. |
| `runtime_usage` | RadiationFunctor.forward evaluates the production radiation balance and optional canopy operator through the level-2 wrapper only. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

Executable `ports()` owner: `RadiationFunctor`. The default instance returns no fixed PortSpec entries; its ports are configuration-conditional or observational.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `atmosphere.radiation.atmospheric_transmittance` | `dimensionless` | `0.75; bounds=(0.6, 0.85)` | `learnable` | Clear-sky atmospheric transmittance for shortwave radiation | Source: literature; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `atmosphere.radiation.cloud_attenuation_factor` | `dimensionless` | `0.8; bounds=(0.2, 1.0)` | `learnable` | Cloud attenuation strength in SW_down = SW_clear * (1 - cloud_fraction * factor) | Source: literature; Sellers (1985) Int. J. Remote Sensing 6, 1335-1372 |
| `atmosphere.radiation.brutsaert_coefficient` | `dimensionless` | `1.24; bounds=(1.0, 1.5)` | `learnable` | Brutsaert equation coefficient for longwave emissivity estimation | Source: literature; Brutsaert (1975) doi:10.1029/WR011i005p00742; Peng et al. (2025) Atmos. Meas. Tech. 18, 2877-2905 |
| `atmosphere.radiation.brutsaert_exponent` | `dimensionless` | `0.142857; bounds=(0.1, 0.2)` | `fixed` | Brutsaert equation exponent (1/7) for vapor pressure effect on emissivity; vapor pressure input uses hPa in the canonical form | Source: literature; Brutsaert (1975) doi:10.1029/WR011i005p00742; Peng et al. (2025) Atmos. Meas. Tech. 18, 2877-2905 |
| `atmosphere.radiation.emissivity_clear_min` | `dimensionless` | `0.4; bounds=(0.0, 1.0)` | `fixed` | Minimum clear-sky atmospheric emissivity used in Brutsaert formula | Source: literature; Brutsaert (1975) |
| `atmosphere.radiation.emissivity_clear_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Maximum clear-sky atmospheric emissivity used in Brutsaert formula | Source: literature; Brutsaert (1975) |
| `atmosphere.radiation.cloud_emissivity` | `dimensionless` | `0.98; bounds=(0.8, 1.0)` | `fixed` | Effective cloud emissivity for longwave enhancement | Source: literature; Campbell & Norman (1998). Environmental Biophysics |
| `atmosphere.radiation.canopy.model_type` | `dimensionless` | `'beer_lambert'; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Canonical canopy transfer selector; only beer_lambert is executable | Source: model_configuration; Monsi and Saeki (1953). Japanese Journal of Botany 14, 22-52 |
| `atmosphere.radiation.canopy.layers` | `dimensionless` | `20; bounds=(1.0, 100.0)` | `fixed` | Number of canopy layers used for radiative transfer discretization | Source: model_configuration; Bonan (2019). Climate Change and Terrestrial Ecosystem Modeling |
| `atmosphere.radiation.canopy.k_b` | `dimensionless` | `0.5; bounds=(0.2, 1.2)` | `learnable` | Direct-beam extinction coefficient for canopy radiation | Source: literature; Campbell & Norman (1998). Environmental Biophysics |
| `atmosphere.radiation.canopy.k_d` | `dimensionless` | `0.78; bounds=(0.3, 1.5)` | `learnable` | Diffuse radiation extinction coefficient for canopy layers | Source: literature; Campbell & Norman (1998). Environmental Biophysics |
| `atmosphere.radiation.canopy.erbs_low_threshold` | `dimensionless` | `0.22; bounds=(0.0, 1.0)` | `fixed` | Lower clearness-index threshold in the Erbs hourly diffuse-fraction correlation | Source: literature; Erbs et al. (1982). Solar Energy 28, 293-302. doi:10.1016/0038-092X(82)90302-4 |
| `atmosphere.radiation.canopy.erbs_high_threshold` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Upper clearness-index threshold in the Erbs hourly diffuse-fraction correlation | Source: literature; Erbs et al. (1982). Solar Energy 28, 293-302. doi:10.1016/0038-092X(82)90302-4 |
| `atmosphere.radiation.canopy.erbs_low_coefficients` | `dimensionless` | `[1.0, -0.09]; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Constant and linear coefficients for the low-clearness Erbs branch | Source: literature; Erbs et al. (1982). Solar Energy 28, 293-302. doi:10.1016/0038-092X(82)90302-4 |
| `atmosphere.radiation.canopy.erbs_mid_coefficients` | `dimensionless` | `[0.9511, -0.1604, 4.388, -16.638, 12.336]; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Polynomial coefficients ordered by increasing power for the middle-clearness Erbs branch | Source: literature; Erbs et al. (1982). Solar Energy 28, 293-302. doi:10.1016/0038-092X(82)90302-4 |
| `atmosphere.radiation.canopy.erbs_high_fraction` | `dimensionless` | `0.165; bounds=(0.0, 1.0)` | `fixed` | Diffuse fraction for the high-clearness Erbs branch | Source: literature; Erbs et al. (1982). Solar Energy 28, 293-302. doi:10.1016/0038-092X(82)90302-4 |
| `atmosphere.radiation.canopy.lai_profile` | `dimensionless` | `'uniform'; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Vertical distribution type for leaf area index | Source: model_configuration; Bonan (2019). Climate Change and Terrestrial Ecosystem Modeling |
| `atmosphere.radiation.canopy.beta_params` | `dimensionless` | `[2.0, 2.0]; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Beta distribution parameters [a, b] for LAI vertical profile | Source: literature; Bonan (2019). Climate Change and Terrestrial Ecosystem Modeling |
| `atmosphere.radiation.canopy.ppfd_conversion` | `umol J-1` | `4.6; bounds=(4.0, 5.0)` | `learnable` | Conversion factor between PAR energy and PPFD | Source: literature; McCree (1972). Agricultural Meteorology |
| `atmosphere.radiation.canopy.par_fraction_of_sw` | `dimensionless` | `0.45; bounds=(0.4, 0.5)` | `learnable` | Fraction of shortwave radiation represented as PAR | Source: literature; Jacovides et al. (2003). Agricultural and Forest Meteorology |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | every retained output is instantaneous UpdateItem mode REPLACE with explicit unit and per_timestep false; diagnostics are DERIVED |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\radiation.md`.

### References

- Allen, R.G. et al. (1998). FAO-56: Crop evapotranspiration guidelines (FunctorMetadata, certificate)
- Brutsaert, W. (1975). On a derivable formula for long-wave radiation from clear skies (FunctorMetadata, certificate)
- Campbell, G.S. & Norman, J.M. (1998). Environmental Biophysics (FunctorMetadata, certificate)
- Erbs, D.G. et al. (1982). Solar Energy 28, 293-302 (FunctorMetadata, certificate)
- Monteith, J.L. & Unsworth, M.H. (2013). Principles of Environmental Physics (FunctorMetadata, certificate)
- Nicodemus (1965), Applied Optics 4, 767-775, doi:10.1364/AO.4.000767 (FunctorMetadata, certificate)
- Noah-MP Technical Note, Eqs. 3.6.330 and 3.6.416-417 (FunctorMetadata, certificate)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- McCree, K.J. (1972). Test of current definitions of photosynthetically active radiation against leaf photosynthesis data (certificate)
- Jacovides, C.P. et al. (2003). Global photosynthetically active radiation and its relationship with global solar radiation (certificate)
- {'van der Schaft, A. and Jeltsema, D. (2014). Port-Hamiltonian systems theory': 'An introductory overview'} (certificate)

(class-relic-scientific-atmosphere-stochastic-rainfall-markedpoissonrainfallfunctor)=
## MarkedPoissonRainfallFunctor

Marked Poisson rainfall dN ~ Poisson(lambda dt), dH = sum z_i (PhysicalField contract).

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.stochastic_rainfall.MarkedPoissonRainfallFunctor` |
| Source | `relic\scientific\atmosphere\stochastic_rainfall.py:44` |
| Runtime name | `stochastic_rainfall` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `field` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MarkedPoissonRainfallFunctor(name: str = 'stochastic_rainfall', *, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'stochastic_rainfall'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Mapping[str, Any]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

The runtime metadata below is introspected from the documented configuration `seed=7`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `stochastic_rainfall_event_count` | state | `1` | prognostic counting state N [1]; input state and accumulated arrivals |
| `stochastic_rainfall_depth` | state | `mm` | prognostic accumulated depth H [mm]; input state and accumulated exponential marks |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `stochastic_rainfall_event_count` | state | `1` | decided at run time by the returned write mode | prognostic counting state N [1]; input state and accumulated arrivals |
| `stochastic_rainfall_depth` | state | `mm` | decided at run time by the returned write mode | prognostic accumulated depth H [mm]; input state and accumulated exponential marks |
| `diagnostics.stochastic_rainfall.event_count_increment` | diagnostic | `1` | diagnostics | arrivals k of the emitted step [1] |
| `diagnostics.stochastic_rainfall.depth_increment` | diagnostic | `mm` | diagnostics | sum of the exponential marks of the emitted step [mm] |
| `diagnostics.stochastic_rainfall.compensator_count` | diagnostic | `1` | diagnostics | predictable compensator lambda dt of the counting measure [1] |
| `diagnostics.stochastic_rainfall.compensator_depth` | diagnostic | `mm` | diagnostics | predictable compensator lambda alpha dt of the depth increment [mm] |

### Governing equations and canonical operator

**Literature or process form**

Marked Poisson rainfall with exponential depths (Rodriguez-Iturbe et al. 1999; Laio et al. 2001), compound Poisson moments (Karlin and Taylor 1981), direct SSA event clock (Gillespie 1977)

**Canonical RELIC structural form**

$\mathrm{d}x = h ( x ^ - , z ) N ( \mathrm{d}t , \mathrm{d}z )$ with $h = ( 1 , z )$ on (N, H), compensator lambda nu(dz) dt; no IPHS operator is claimed

| Entry | Meaning |
| --- | --- |
| `event_clock` | constant-hazard direct SSA (Gillespie 1977) over one fixed step in days on the Functor-owned generator; the per-cell arrival cap mu + 12 sqrt(mu) + 64 is a fail-closed guard |
| `mark_law` | exponential depth marks $z = \alpha E$, E ~ Exp(1), drawn in cell order from the same generator; alpha keeps its autograd graph |
| `compensator` | lambda dt and lambda alpha dt emitted as REPLACE diagnostics; the compensated representation must add them as drift |
| `time_conversion` | the simulator step in seconds is converted once to days through the SSOT time constant SECONDS_PER_DAY |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | prognostic count and depth states on the declared cell basis; no storage functional is claimed |
| `effort_variable` | none; arrivals follow a constant hazard and marks an exponential law, not a variational effort |
| `state_advancer` | $\Delta N = k$ from one GillespieSSA.sample call with $\mathrm{final}_{\mathrm{time}} = \mathrm{d}t \mathrm{in} \mathrm{days}$ and $Delta H = alpha sum_{i<=k} E_i$ from the Functor-owned generator |
| `runtime_usage` | MarkedPoissonRainfallFunctor.forward validates both states and the step, converts dt to days, samples the arrivals per cell with the direct SSA, draws the exponential marks in cell order, increments the substream draw count, and emits two ACCUMULATE increments plus four REPLACE diagnostics; a rejected step leaves the substream position unchanged. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `stochastic.weather.rainfall.event_rate` | `day-1` | `0.3; bounds=(0.01, 5.0)` | `fixed` | Poisson arrival rate lambda of rainfall events for the marked-Poisson rainfall process | Source: literature; Rodriguez-Iturbe, I., Porporato, A., Ridolfi, L., Isham, V. & Cox, D.R. (1999). Proc. R. Soc. A, 455, 3789-3805.; Laio, F., Porporato, A., Ridolfi, L. & Rodriguez-Iturbe, I. (2001). Adv. Water Resour., 24, 707-723. |
| `stochastic.weather.rainfall.mean_depth` | `mm` | `15.0; bounds=(0.1, 200.0)` | `learnable` | Mean alpha of the exponential rainfall depth mark per event for the marked-Poisson rainfall process | Source: literature; Rodriguez-Iturbe, I., Porporato, A., Ridolfi, L., Isham, V. & Cox, D.R. (1999). Proc. R. Soc. A, 455, 3789-3805.; Laio, F., Porporato, A., Ridolfi, L. & Rodriguez-Iturbe, I. (2001). Adv. Water Resour., 24, 707-723. |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem ACCUMULATE with per_timestep true for the count and depth and REPLACE observations for diagnostics |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\atmosphere_stochastic_rainfall.md`.

### References

- Rodriguez-Iturbe, I. et al. (1999). Proc. R. Soc. A 455, 3789-3805 (FunctorMetadata)
- Laio, F. et al. (2001). Adv. Water Resour. 24, 707-723 (FunctorMetadata)
- Gillespie, D.T. (1977). J. Phys. Chem. 81, 2340-2361 (FunctorMetadata, certificate)
- Karlin, S. & Taylor, H.M. (1981). A Second Course in Stochastic Processes (FunctorMetadata)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- Rodriguez-Iturbe, I., Porporato, A., Ridolfi, L., Isham, V. & Cox, D.R. (1999). Proc. R. Soc. A 455, 3789-3805 (certificate)
- Laio, F., Porporato, A., Ridolfi, L. & Rodriguez-Iturbe, I. (2001). Adv. Water Resour. 24, 707-723 (certificate)
- Karlin, S. & Taylor, H.M. (1981). A Second Course in Stochastic Processes, Academic Press (certificate)

(class-relic-scientific-atmosphere-stochastic-temperature-ornsteinuhlenbecktemperaturefunctor)=
## OrnsteinUhlenbeckTemperatureFunctor

Continuous-SDE anomaly dX = -alpha X dt + sigma dW (PhysicalField contract).

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.stochastic_temperature.OrnsteinUhlenbeckTemperatureFunctor` |
| Source | `relic\scientific\atmosphere\stochastic_temperature.py:36` |
| Runtime name | `stochastic_air_temperature` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `field` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
OrnsteinUhlenbeckTemperatureFunctor(name: str = 'stochastic_air_temperature', *, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'stochastic_air_temperature'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Mapping[str, Any]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

The runtime metadata below is introspected from the documented configuration `seed=7`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `stochastic_air_temperature_anomaly` | state | `K` | prognostic Ornstein-Uhlenbeck anomaly state X [K]; input state and accumulated increment |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `stochastic_air_temperature_anomaly` | state | `K` | decided at run time by the returned write mode | prognostic Ornstein-Uhlenbeck anomaly state X [K]; input state and accumulated increment |
| `diagnostics.stochastic_temperature.drift` | diagnostic | `K day-1` | diagnostics | drift a(X) = -alpha X at the step start [K day-1] |
| `diagnostics.stochastic_temperature.diffusion` | diagnostic | `K day-0.5` | diagnostics | diagonal diffusion coefficient sigma [K day-0.5] |
| `diagnostics.stochastic_temperature.wiener_increment` | diagnostic | `day-0.5` | diagnostics | Wiener increment dW = sqrt(dt) xi from the Functor-owned substream [day-0.5] |

### Governing equations and canonical operator

**Literature or process form**

$$
Ornstein-Uhlenbeck anomaly dX = -alpha X dt + sigma dW (Hasselmann 1976; Gardiner 2009) integrated by Euler-Maruyama X_{n+1} = X_n - alpha X_n dt + sigma sqrt(dt) xi (Kloeden and Platen 1992; Higham 2001)
$$

**Canonical RELIC structural form**

$\dot{x} = a ( x ) + b \frac{\mathrm{d}W}{\mathrm{d}t}$ with $a ( x ) = - \alpha x$ and diagonal $b = \sigma \mathrm{executed}$ by the canonical EulerMaruyamaSolver; no IPHS operator is claimed

| Entry | Meaning |
| --- | --- |
| `drift_operator` | $a ( x ) = - \alpha x$ evaluated by the Functor and validated by EulerMaruyamaSolver |
| `diffusion_operator` | diagonal $b = \sigma \mathrm{broadcast}$ to the state shape; correlated or general diffusion is [N] in SDEParameters |
| `time_conversion` | the simulator step in seconds is converted once to days through the SSOT time constant SECONDS_PER_DAY |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | prognostic anomaly state X on the declared cell basis; no storage functional is claimed |
| `effort_variable` | none; the drift $a ( X ) = - \alpha X$ is a linear relaxation, not a variational effort |
| `state_advancer` | $\Delta X = - \alpha X \mathrm{d}t + \sigma \mathrm{d}W_{n}$ from one EulerMaruyamaSolver.sample step with $\mathrm{num}_{\mathrm{steps}} = 1$ and the Functor-owned generator |
| `runtime_usage` | OrnsteinUhlenbeckTemperatureFunctor.forward validates the state and step, converts dt to days, evaluates one Euler-Maruyama step, increments the substream draw count, and emits one ACCUMULATE increment plus three REPLACE diagnostics; a rejected step leaves the substream position unchanged. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `stochastic.weather.temperature.reversion_rate` | `day-1` | `0.1; bounds=(0.01, 1.0)` | `learnable` | Mean reversion rate alpha for OU temperature process | Source: literature; Hasselmann (1976). Tellus, 28, 473-485. |
| `stochastic.weather.temperature.volatility` | `K day-0.5` | `2.0; bounds=(0.1, 10.0)` | `learnable` | Volatility sigma for OU temperature process | Source: literature; Hasselmann (1976). Tellus, 28, 473-485. |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem ACCUMULATE with per_timestep true for the anomaly and REPLACE observations for diagnostics |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\atmosphere_stochastic_temperature.md`.

### References

- Hasselmann, K. (1976). Tellus 28, 473-485 (FunctorMetadata)
- Gardiner, C. (2009). Stochastic Methods, 4th ed., Springer (FunctorMetadata, certificate)
- Kloeden, P.E. & Platen, E. (1992). Numerical Solution of SDEs, Springer (FunctorMetadata)
- Higham, D.J. (2001). SIAM Review 43, 525-546 (FunctorMetadata, certificate)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- Hasselmann, K. (1976). Stochastic climate models Part I. Tellus 28, 473-485 (certificate)
- Kloeden, P.E. & Platen, E. (1992). Numerical Solution of Stochastic Differential Equations, Springer (certificate)

(class-relic-scientific-atmosphere-transport-atmosphericadvectiondiffusionfunctor-atmosphericadvectiondiffusionfunctor)=
## AtmosphericAdvectionDiffusionFunctor

Atmospheric advection-diffusion transport functor.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.transport.AtmosphericAdvectionDiffusionFunctor.AtmosphericAdvectionDiffusionFunctor` |
| Source | `relic\scientific\atmosphere\transport\AtmosphericAdvectionDiffusionFunctor.py:73` |
| Runtime name | `atmospheric_advection_diffusion` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
AtmosphericAdvectionDiffusionFunctor(spatial_dims: int, grid_spacing: Tuple[float, ...] | None = None, *, stability_scheme: str, boundary_layer_height: float | None = None, device: torch.device | None = None, **kwargs) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `spatial_dims` | `int` | `required` | Constructor control for spatial dims; see the class runtime contract and ParameterManager table below. |
| `grid_spacing` | <code>typing.Tuple[float, ...] &#124; None</code> | `None` | Constructor control for grid spacing; see the class runtime contract and ParameterManager table below. |
| `stability_scheme` | `str` | `required` | Constructor control for stability scheme; see the class runtime contract and ParameterManager table below. |
| `boundary_layer_height` | <code>float &#124; None</code> | `None` | Constructor control for boundary layer height; see the class runtime contract and ParameterManager table below. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Configuration modes

The declared runtime interface depends on the constructor configuration; there is no default mode. Each documented mode was instantiated. The table lists the variables that only that mode declares; every other variable of the runtime contracts below is declared in all modes.

| Mode | Inputs declared only in this mode | Outputs declared only in this mode |
| --- | --- | --- |
| `spatial_dims=3, stability_scheme='monin-obukhov'` | none | none |
| `spatial_dims=3, stability_scheme='richardson'` | none | none |
| `spatial_dims=3, stability_scheme='constant'` | none | none |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | tracer_concentration mapped to canonical Phase 8 cochain basis |
| `wind_u` | forcing | `m s-1` | wind_u mapped to canonical Phase 8 cochain basis |
| `wind_v` | forcing | `m s-1` | wind_v mapped to canonical Phase 8 cochain basis |
| `wind_w` | forcing | `m s-1` | wind_w mapped to canonical Phase 8 cochain basis |
| `temperature` | forcing | `K` | temperature mapped to canonical Phase 8 cochain basis (monin-obukhov scheme only; no default, a missing value raises) |
| `sensible_heat_flux` | flux | `W m-2` | surface upward sensible heat flux H [W m-2] written by atmospheric_turbulence or supplied as forcing, read by the monin-obukhov scheme only; no default |
| `atmospheric_pressure` | forcing | `Pa` | atmospheric pressure p [Pa] of the moist-air density rho(T, p, RH*e_s(T)) in the Obukhov length, read by the monin-obukhov scheme only; no default |
| `relative_humidity` | forcing | `1` | relative humidity RH [1] of the vapour pressure e = RH*e_s(T) in the moist-air density, read by the monin-obukhov scheme only; no default |
| `source_term` | diagnostic | `kg m-3 s-1` | optional volumetric tracer source S [kg m-3 s-1] of the port atmosphere_tracer_source_port; absent means the closed no-source case S = 0, no ParameterManager default |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | decided at run time by the returned write mode | tracer_concentration mapped to canonical Phase 8 cochain basis |
| `diagnostics.atmospheric_advection_diffusion.cfl_number` | diagnostic | `1` | diagnostics | cell-centred Courant sum max(dt * sum_k \\|u_k\\| / dx_k) of the pre-step limit check, renamed from diagnostics.cfl_number, which is the finite-volume outward-face Courant sum of the generic transport Functors |
| `diagnostics.stability_number` | diagnostic | `1` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.advection_rate` | diagnostic | `kg m-3 s-1` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.diffusion_rate` | diagnostic | `kg m-3 s-1` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.eddy_diffusivity` | diagnostic | `m2 s-1` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.stability_parameter` | diagnostic | `1` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.conservation_error` | diagnostic | `dimensionless` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.boundary_flux_cochain` | output port | `kg m-2 s-1` | diagnostics | optional packed outward face flux ledger returned by the canonical operator when exterior faces exist |

### Governing equations and canonical operator

**Literature or process form**

$$
\frac{\mathrm{d}c}{\mathrm{d}t} + \nabla \cdot ( v c ) = \nabla \cdot ( K \nabla c ) + S
$$

**Canonical RELIC structural form**

$\dot{c} = - D_{h} q_{h} + G_{h} u_{h} + s_{h}$ with $q_{h} = v_{h} c_{h} - \star_{K} d_{h} c_{h}$

| Entry | Meaning |
| --- | --- |
| `D_h` | declared atmospheric tracer divergence map |
| `star_K` | ParameterManager-backed molecular plus eddy diffusivity Hodge map |
| `q_h` | declared advective plus diffusive tracer flux cochain |
| `G_h` | boundary and source port map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{c} = - D_{h} q_{h} + G_{h} u_{h} + s_{h}$ with declared tracer flux and boundary/source ports |
| `runtime_usage` | AtmosphericAdvectionDiffusionFunctor.forward applies the registered transport.advection_diffusion operator once to the caller's actual cells with the resolved homogeneous face law (zero-gradient neumann, periodic, or one static scalar dirichlet) in storage-order spacing, and returns the explicit integrated tracer increment plus CFL, stability, eddy diffusivity, conservation ledger, and packed outward face flux diagnostics. |

### Composition channels and executable PortSpec API

Certificate composition channels: `atmospheric_turbulence`, `atmosphere_tracer_boundary_export`, `atmosphere_tracer_source_port`

Executable `ports()` owner: `AtmosphericAdvectionDiffusionFunctor`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `atmosphere_tracer_boundary_export` | `tracer_concentration [kg m-3]` | `diagnostics.boundary_flux_cochain [kg m-2 s-1]` | `kg2 m-5 s-1` | out_of_system; matter |
| `atmosphere_tracer_source_port` | `tracer_concentration [kg m-3]` | `source_term [kg m-3 s-1]` | `kg2 m-6 s-1` | into_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `atmosphere.grid.dx` | `m` | `1000.0; bounds=(10.0, 100000.0)` | `fixed` | Grid spacing in the x-direction for atmospheric transport models | Source: model_configuration; Seinfeld & Pandis (2016) |
| `atmosphere.grid.dy` | `m` | `1000.0; bounds=(10.0, 100000.0)` | `fixed` | Grid spacing in the y-direction for atmospheric transport models | Source: model_configuration; Seinfeld & Pandis (2016) |
| `atmosphere.grid.dz` | `m` | `100.0; bounds=(1.0, 10000.0)` | `fixed` | Grid spacing in the vertical direction for atmospheric transport models | Source: model_configuration; Stull (1988) |
| `atmosphere.boundary_layer.boundary_layer_height` | `m` | `1000.0; bounds=(50.0, 3000.0)` | `fixed` | Representative planetary boundary-layer height used when diagnostics are unavailable | Source: boundary_layer_climatology; Stull (1988) Boundary Layer Meteorology |
| `atmosphere.turbulence.eddy_diffusivity_max` | `m2 s-1` | `50.0; bounds=(0.1, 100.0)` | `fixed` | Upper bound on eddy diffusivity used by atmospheric advection-diffusion functors | Source: boundary_layer_theory; Garratt (1992) The Atmospheric Boundary Layer |
| `atmosphere.turbulence.transport_neutral_diffusivity_fraction` | `dimensionless` | `0.1; bounds=(0.1, 0.1)` | `fixed` | Fixed neutral fraction in the existing atmospheric transport K closure; not a universal MOST coefficient. | Source: existing_implementation; Stull (1988), K-theory background; numerical value preserves the RELIC empirical closure. |
| `atmosphere.turbulence.transport_unstable_enhancement_coefficient` | `dimensionless` | `2.0; bounds=(2.0, 2.0)` | `fixed` | Fixed coefficient a in K=K0*(1+a*s/(1+s)) for s=-stability>0. | Source: existing_implementation; Stull (1988), K-theory background; numerical value preserves the RELIC empirical closure. |
| `atmosphere.turbulence.transport_stable_suppression_coefficient` | `dimensionless` | `10.0; bounds=(10.0, 10.0)` | `fixed` | Fixed coefficient b in K=K0/(1+b*stability) for stability>=0. | Source: existing_implementation; Stull (1988), K-theory background; numerical value preserves the RELIC empirical closure. |
| `atmosphere.turbulence.transport_stability_limit` | `dimensionless` | `10.0; bounds=(10.0, 10.0)` | `fixed` | Symmetric diagnostic stability limit retained by the existing atmospheric transport closure. | Source: existing_implementation; RELIC atmospheric advection-diffusion reduced stability closure. |
| `atmosphere.turbulence.minimum_friction_velocity` | `m s-1` | `0.01; bounds=(0.001, 0.1)` | `fixed` | Positive reduced friction-velocity scale used by atmospheric scalar transport's MOST diagnostic and structural Hodge audits; not a measured wind input. | Source: RELIC parameter database; Stull (1988) An Introduction to Boundary Layer Meteorology, Kluwer; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `atmosphere.stability.potential_temperature_gradient` | `K m-1` | `0.01; bounds=(-0.1, 0.1)` | `fixed` | Background potential temperature gradient used for Richardson number evaluation | Source: relic-transport; Businger, J.A. et al. (1971) Flux-profile relationships in the atmospheric surface layer; US EPA (1999) CMAQ Science Algorithms, Chapter 7 |
| `atmosphere.molecular_diffusion.molecular_diffusivity` | `m2 s-1` | `1.5e-05; bounds=(1e-06, 0.0001)` | `fixed` | Reference molecular diffusivity for scalar transport in air | Source: literature; Jacobson (2005) Fundamentals of Atmospheric Modeling |
| `transport.numerical.cfl_advection_max` | `dimensionless` | `1.0; bounds=(0.5, 1.0)` | `fixed` | Maximum allowable CFL number for advection-dominated schemes | Source: RELIC transport module; Courant et al. (1928) |
| `transport.numerical.cfl_diffusion_max` | `dimensionless` | `0.5; bounds=(0.25, 0.5)` | `fixed` | Maximum allowable CFL number for diffusion (Von Neumann condition) | Source: RELIC transport module; Von Neumann and Richtmyer (1950) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\atmosphere_advection_diffusion.md`.

### References

- Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology. (FunctorMetadata, certificate)
- Jacobson, M.Z. (2005). Fundamentals of Atmospheric Modeling. (FunctorMetadata, certificate)
- Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics. (FunctorMetadata, certificate)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Holton, J.R. & Hakim, G.J. (2012). An Introduction to Dynamic Meteorology, 5th ed., Eqs. 2.63-2.64 (certificate)
- Allen, R.G. et al. (1998). FAO Irrigation and Drainage Paper 56, Eqs. 11 and 19 (certificate)

(class-relic-scientific-atmosphere-transport-boundary-layer-emission-source-boundarylayeremissionsource)=
## BoundaryLayerEmissionSource

Hand the surface isoprene flux over to the well-mixed boundary layer.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.transport.boundary_layer_emission_source.BoundaryLayerEmissionSource` |
| Source | `relic\scientific\atmosphere\transport\boundary_layer_emission_source.py:52` |
| Runtime name | `boundary_layer_emission_source` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
BoundaryLayerEmissionSource(name: str = 'boundary_layer_emission_source', config: Dict[str, Any] | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'boundary_layer_emission_source'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `isoprene_emission` | flux | `mol m-2 s-1` | surface isoprene molar flux on a ground-area basis [mol m-2 s-1] |
| `pbl_height` | forcing | `m` | well-mixed planetary boundary-layer height [m] |
| `air_temperature` | forcing | `K` | air temperature of the ideal-gas air molar density [K] |
| `atmospheric_pressure` | forcing | `Pa` | pressure of the ideal-gas air molar density [Pa] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `isoprene_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | atmospheric isoprene state written as a timestep-integrated source increment [mol mol-1] |
| `diagnostics.boundary_layer_emission.relative_column_balance_residual` | diagnostic | `1` | diagnostics | column molar burden gained minus the amount crossing the lower boundary, relative to that amount [1] |

### Governing equations and canonical operator

**Literature or process form**

d(chi)/$\mathrm{d}t = F / ( h n )$, $n = P / ( R T )$

**Canonical RELIC structural form**

Level 1 compatibility records the executed source $\Delta \chi = F \mathrm{d}t R T / ( P h )$ with $F \geq 0$; no IPHS operator is claimed.

| Entry | Meaning |
| --- | --- |
| `source_law` | SpeciesMapper.flux_to_mixing_ratio evaluates $\Delta \chi = F \mathrm{d}t R T / ( P h )$ |
| `physical_constant` | relic.physical.constants.universal.gas_constant |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | not_applicable_algebraic_boundary_source_without_hamiltonian_potential |
| `effort_variable` | not_applicable_no_energy_or_entropy_effort |
| `state_advancer` | $\Delta \chi = F \mathrm{d}t R T / ( P h )$ with F, h, P and T held at their values in the step |
| `runtime_usage` | BoundaryLayerEmissionSource.forward evaluates one mixing-ratio increment and returns it as one ACCUMULATE UpdateItem, which StateManager adds once. |

### Composition channels and executable PortSpec API

Certificate composition channels: `isoprene_emission`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

This class has no numeric scientific parameter keys in its certificate; it acts as a structural observer or adapter over declared state and ports.

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | isoprene_mixing_ratio uses UpdateItem(ACCUMULATE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{True} )$; the diagnostic is a read-only tensor |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\atmosphere_boundary_layer_emission_source.md`.

### References

- Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology. Kluwer Academic (FunctorMetadata, certificate)
- Vila-Guerau de Arellano, J. et al. (2015). Atmospheric Boundary Layer: Integrating Air Chemistry and Land Interactions. Cambridge University Press (FunctorMetadata, certificate)
- Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics, 3rd ed. (FunctorMetadata, certificate)
- U.S. EPA (2019). CMAQ User's Guide, emissions conversion (FunctorMetadata, certificate)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)

(class-relic-scientific-atmosphere-transport-dry-deposition-aggregator-drydepositionaggregator)=
## DryDepositionAggregator

Apply the single declared resistance-law dry-deposition closure.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.transport.dry_deposition_aggregator.DryDepositionAggregator` |
| Source | `relic\scientific\atmosphere\transport\dry_deposition_aggregator.py:59` |
| Runtime name | `dry_deposition_aggregator` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
DryDepositionAggregator(name: str = 'dry_deposition_aggregator', config: Dict[str, Any] | None = None, deposition_config: relic.scientific.atmosphere.transport.dry_deposition_aggregator.DryDepositionConfig | None = None, *, surface_layer_source: Literal['atmospheric_turbulence', 'external_forcing'] = 'atmospheric_turbulence') -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'dry_deposition_aggregator'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `deposition_config` | <code>relic.scientific.atmosphere.transport.dry_deposition_aggregator.DryDepositionConfig &#124; None</code> | `None` | Constructor control for deposition config; see the class runtime contract and ParameterManager table below. |
| `surface_layer_source` | `typing.Literal['atmospheric_turbulence', 'external_forcing']` | `'atmospheric_turbulence'` | Constructor control for surface layer source; see the class runtime contract and ParameterManager table below. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `no_mixing_ratio` | state | `mol mol-1` | atmospheric NO state read as a mixing ratio and written as a timestep-integrated sink increment [mol mol-1] |
| `no2_mixing_ratio` | state | `mol mol-1` | atmospheric NO2 state read as a mixing ratio and written as a timestep-integrated sink increment [mol mol-1] |
| `o3_mixing_ratio` | state | `mol mol-1` | atmospheric O3 state read as a mixing ratio and written as a timestep-integrated sink increment [mol mol-1] |
| `so2_mixing_ratio` | state | `mol mol-1` | atmospheric SO2 state read as a mixing ratio and written as a timestep-integrated sink increment [mol mol-1] |
| `pbl_height` | forcing | `m` | pbl_height mapped to canonical Phase 8 cochain basis |
| `aerodynamic_resistance` | diagnostic | `s m-1` | aerodynamic resistance supplied by atmospheric_turbulence, or by external forcing under surface_layer_source = external_forcing [s m-1] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `no_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | atmospheric NO state read as a mixing ratio and written as a timestep-integrated sink increment [mol mol-1] |
| `no2_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | atmospheric NO2 state read as a mixing ratio and written as a timestep-integrated sink increment [mol mol-1] |
| `o3_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | atmospheric O3 state read as a mixing ratio and written as a timestep-integrated sink increment [mol mol-1] |
| `so2_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | atmospheric SO2 state read as a mixing ratio and written as a timestep-integrated sink increment [mol mol-1] |
| `diagnostics.dry_deposition.total_mixing_ratio_increment` | output port | `mol mol-1` | diagnostics | sum of species increments for monitoring [mol mol-1] |
| `diagnostics.dry_deposition.total_limiter_residual` | output port | `mol mol-1` | diagnostics | summed difference between unconstrained and submitted increments [mol mol-1] |
| `diagnostics.dry_deposition.normalized_mass_balance_residual` | output port | `mol mol-1` | diagnostics | per-species submitted increment minus unconstrained increment and limiter residual [mol mol-1] |
| `diagnostics.dry_deposition.enabled` | output port | `1` | diagnostics | explicit activation flag [1] |

### Governing equations and canonical operator

**Literature or process form**

$$
r_{\mathrm{eff},s} = r_{b,s} + r_{c,s}; v_{d,s} = 1 / ( r_{a} + r_{\mathrm{eff},s} ); \Delta C_{s} = \max ( - C_{s} , - v_{d,s} C_{s} \Delta t / h_{\mathrm{pbl}} )
$$

**Canonical RELIC structural form**

$$
\dot{x} = - D_{h} q_{h} + G_{h} u_{h}, q_{h} = \star_{\mathrm{dep}} C_{h}, y_{h} = G_{h} ^{\mathsf{T}} e_{h}
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | existing divergence/transport operator |
| `q_h` | declared flux cochain |
| `G_h` | boundary/source port map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\Delta C_{s} = \max ( - C_{s} , - C_{s} \Delta t / ( ( r_{a} + r_{\mathrm{eff},s} ) h_{\mathrm{pbl}} ) )$ with explicit boundary_export routing |
| `runtime_usage` | DryDepositionAggregator.forward evaluates one normalized mixing-ratio sink and returns timestep-integrated UpdateItems; no strict surface mass-flux proof is claimed. |

### Composition channels and executable PortSpec API

Certificate composition channels: `dry_deposition_surface_export`

Executable `ports()` owner: `DryDepositionAggregator`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `dry_deposition_surface_export` | `near_surface_species_mixing_ratio [mol mol-1]` | `dry_deposition_mixing_ratio_tendency [mol mol-1 s-1]` | `mol2 mol-2 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `atmosphere.dry_deposition.enabled` | `dimensionless` | `True; bounds=(0.0, 100.0)` | `fixed` | Enable dry deposition aggregator | Source: literature; Wesely (1989) |
| `atmosphere.dry_deposition.species_list` | `dimensionless` | `['no', 'no2', 'o3', 'so2']; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Ordered inorganic species handled by the canonical resistance closure | Source: configuration; Wesely (1989); Zhang et al. (2003) |
| `atmosphere.dry_deposition.species.no.effective_non_aerodynamic_resistance` | `s m-1` | `500.0; bounds=(1e-06, 50000.0)` | `fixed` | Effective NO resistance r_b plus r_c for the canonical series-resistance closure | Source: literature_parameterization; Wesely (1989) |
| `atmosphere.dry_deposition.species.no2.effective_non_aerodynamic_resistance` | `s m-1` | `200.0; bounds=(1e-06, 20000.0)` | `fixed` | Effective NO2 resistance r_b plus r_c for the canonical series-resistance closure | Source: literature_parameterization; Wesely (1989) |
| `atmosphere.dry_deposition.species.o3.effective_non_aerodynamic_resistance` | `s m-1` | `150.0; bounds=(1e-06, 15000.0)` | `fixed` | Effective O3 resistance r_b plus r_c for the canonical series-resistance closure | Source: literature_parameterization; Wesely (1989) |
| `atmosphere.dry_deposition.species.so2.effective_non_aerodynamic_resistance` | `s m-1` | `100.0; bounds=(1e-06, 10000.0)` | `fixed` | Effective SO2 resistance r_b plus r_c for the canonical series-resistance closure | Source: literature_parameterization; Wesely (1989) |
| `atmosphere.dry_deposition.reference_height` | `m` | `10.0; bounds=(1.0, 100.0)` | `fixed` | Reference height for deposition velocity normalization | Source: literature; Wesely (1989) |
| `atmosphere.dry_deposition.structural_reference_deposition_velocity` | `m s-1` | `0.01; bounds=(1e-06, 1.0)` | `fixed` | Positive reference velocity used only by the non-strict Phase 8 Hodge declaration; runtime velocity is state-dependent. | Source: literature_structural_reference; Wesely (1989); Zhang et al. (2003) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | atmospheric states use UpdateItem(ACCUMULATE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{True} )$; diagnostics are read-only tensors |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\atmosphere_dry_deposition_aggregator.md`.

### References

- Wesely, M.L. (1989). Atmos. Environ., 23, 1293-1304 (FunctorMetadata)
- Zhang, L., Brook, J.R., & Vet, R. (2003). Atmos. Chem. Phys., 3, 2067-2082 (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Wesely, M.L. (1989). Parameterization of surface resistances to gaseous dry deposition. Atmos. Environ., 23, 1293-1304 (certificate)
- Zhang, L., Brook, J.R., & Vet, R. (2003). A revised parameterization for gaseous dry deposition. Atmos. Chem. Phys., 3, 2067-2082 (certificate)
- Baldocchi, D.D., Hicks, B.B., & Camara, P. (1987). A canopy stomatal resistance model for gaseous deposition. Atmos. Environ., 21, 91-101 (certificate)
- Hicks, B.B. et al. (1987). A preliminary multiple resistance routine for deriving dry deposition velocities. Atmos. Environ., 21, 1039-1044 (certificate)

(class-relic-scientific-atmosphere-transport-eddy-diffusion-blm-eddydiffusionblmfunctor)=
## EddyDiffusionBLMFunctor

Boundary Layer Eddy Diffusion Transport Mechanism Functor

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.transport.eddy_diffusion_blm.EddyDiffusionBLMFunctor` |
| Source | `relic\scientific\atmosphere\transport\eddy_diffusion_blm.py:123` |
| Runtime name | `eddy_diffusion_blm` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
EddyDiffusionBLMFunctor(name: str = 'eddy_diffusion_blm', config: Dict[str, Any] | None = None, transport_config: relic.scientific.atmosphere.transport.eddy_diffusion_blm.EddyDiffusionConfig | None = None, device: torch.device | None = None, *, surface_layer_source: Literal['atmospheric_turbulence', 'external_forcing'] = 'atmospheric_turbulence') -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'eddy_diffusion_blm'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `transport_config` | <code>relic.scientific.atmosphere.transport.eddy_diffusion_blm.EddyDiffusionConfig &#124; None</code> | `None` | Constructor control for transport config; see the class runtime contract and ParameterManager table below. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `surface_layer_source` | `typing.Literal['atmospheric_turbulence', 'external_forcing']` | `'atmospheric_turbulence'` | Constructor control for surface layer source; see the class runtime contract and ParameterManager table below. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `no_mixing_ratio` | state | `mol mol-1` | no_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `no2_mixing_ratio` | state | `mol mol-1` | no2_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `o3_mixing_ratio` | state | `mol mol-1` | o3_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `isoprene_mixing_ratio` | state | `mol mol-1` | isoprene_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `pbl_height` | forcing | `m` | pbl_height mapped to canonical Phase 8 cochain basis |
| `friction_velocity` | diagnostic | `m s-1` | friction_velocity mapped to canonical turbulence forcing |
| `monin_obukhov_length` | diagnostic | `m` | monin_obukhov_length mapped to canonical turbulence forcing |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `no_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | no_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `no2_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | no2_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `o3_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | o3_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `isoprene_mixing_ratio` | state | `mol mol-1` | decided at run time by the returned write mode | isoprene_mixing_ratio mapped to canonical Phase 8 cochain basis |
| `diagnostics.eddy_diffusion.enabled` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.eddy_diffusion.eddy_diffusivity` | output port | `m2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.eddy_diffusion.pbl_height` | output port | `m` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.eddy_diffusion.friction_velocity` | output port | `m s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.eddy_diffusion.monin_obukhov_length` | output port | `m` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.eddy_diffusion.stability_number` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.eddy_diffusion.stability_parameter_z_over_L` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.eddy_diffusion.eddy_diffusivity_clamped` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.eddy_diffusion.stability_parameter_clipping_residual` | output port | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

$q=-K\,\partial C/\partial z$ with reduced one-box closure $\dot C=K h^{-2}(C_{bg}-C)$, $\zeta_{eff}=clip(\zeta_{raw},-\min(l_p,l_u),\min(l_p,l_s))$, and $K=\kappa u_* (h/2)/\phi_h(\zeta_{eff})$

**Canonical RELIC structural form**

$$
\dot{x} = - D_{h} q_{h} + G_{h} u_{h}, q_{h} = - \star_{K} D_{h} ^{\mathsf{T}} e_{h}, y_{h} = G_{h} ^{\mathsf{T}} e_{h}
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | existing divergence/transport operator |
| `q_h` | declared flux cochain |
| `G_h` | boundary/source port map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = - D_{h} q_{h} + G_{h} u_{h}$ with declared vertical turbulent flux $q_{h}$ and boundary exchange port |
| `runtime_usage` | EddyDiffusionBLMFunctor.forward evaluates only $\zeta_{\mathrm{eff}} = \operatorname{clip} ( \zeta_{\mathrm{raw}} , - \min ( l_{p}, l_{u} ) , \min ( l_{p}, l_{s} ) )$ and $\mathrm{DeltaC} = \mathrm{d}t \cdot K / h ^ 2 \cdot ( C_{\mathrm{background}} - C )$; no selectable diagnostic, dilution-only, fallback, or second process law remains. |

### Composition channels and executable PortSpec API

Certificate composition channels: `boundary_layer_exchange_port`

Executable `ports()` owner: `EddyDiffusionBLMFunctor`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `boundary_layer_exchange_port` | `boundary_layer_species_mixing_ratio [mol mol-1]` | `boundary_layer_turbulent_flux [mol mol-1 m s-1]` | `mol2 mol-2 m s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `atmosphere.transport.enabled` | `dimensionless` | `False; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Enable or disable boundary-layer eddy diffusion transport functor | Source: transport_configuration; Stull (1988) |
| `atmosphere.transport.target_species` | `dimensionless` | `['no', 'no2', 'o3', 'isoprene']; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Canonical ordered species schema transported by the eddy diffusion functor | Source: transport_configuration; Wesely (1989) |
| `atmosphere.transport.k_min` | `m2 s-1` | `0.01; bounds=(0.0001, 1.0)` | `fixed` | Minimum eddy diffusivity to maintain numerical stability | Source: transport_configuration; Garratt (1992) |
| `atmosphere.transport.k_max` | `m2 s-1` | `50.0; bounds=(1.0, 200.0)` | `fixed` | Maximum eddy diffusivity to prevent runaway diffusion in unstable conditions | Source: transport_configuration; Stull (1988) |
| `atmosphere.transport.z_over_L_limit` | `dimensionless` | `2.0; bounds=(0.5, 5.0)` | `fixed` | Stability limit for \\|z/L\\| used in MOST corrections | Source: transport_configuration; Stull (1988); Garratt (1992) |
| `atmosphere.transport.ustar_min` | `m s-1` | `0.01; bounds=(0.001, 0.5)` | `fixed` | Minimum friction velocity threshold for activating eddy diffusion | Source: transport_configuration; Businger et al. (1971) |
| `atmosphere.transport.ustar_max` | `m s-1` | `2.0; bounds=(0.1, 5.0)` | `fixed` | Maximum friction velocity used to bound eddy diffusivity | Source: transport_configuration; Stull (1988); Garratt (1992) |
| `atmosphere.transport.pbl_height_min` | `m` | `50.0; bounds=(10.0, 500.0)` | `fixed` | Minimum accepted PBL height for eddy diffusion transport. | Source: transport_configuration; Stull (1988) |
| `atmosphere.transport.pbl_height_max` | `m` | `3000.0; bounds=(500.0, 5000.0)` | `fixed` | Maximum accepted PBL height for eddy diffusion transport. | Source: transport_configuration; Stull (1988) |
| `atmosphere.transport.background_no_mixing_ratio` | `mol mol-1` | `1e-10; bounds=(0.0, 1e-06)` | `fixed` | Free-tropospheric background NO mixing ratio for transport relaxation. | Source: atmospheric_climatology; Seinfeld & Pandis (2016) |
| `atmosphere.transport.background_no2_mixing_ratio` | `mol mol-1` | `5e-10; bounds=(0.0, 1e-05)` | `fixed` | Free-tropospheric background NO2 mixing ratio for transport relaxation. | Source: atmospheric_climatology; Seinfeld & Pandis (2016) |
| `atmosphere.transport.background_o3_mixing_ratio` | `mol mol-1` | `4e-08; bounds=(1e-09, 1e-05)` | `fixed` | Free-tropospheric background O3 mixing ratio for transport relaxation. | Source: atmospheric_climatology; Seinfeld & Pandis (2016); Logan (1985) |
| `atmosphere.transport.background_isoprene_mixing_ratio` | `mol mol-1` | `1e-10; bounds=(0.0, 1e-06)` | `fixed` | Free-tropospheric background isoprene mixing ratio for transport relaxation. | Source: atmospheric_climatology; Guenther et al. (2012) |
| `transport.numerical.cfl_safety_factor` | `dimensionless` | `0.8; bounds=(0.1, 0.9)` | `fixed` | Safety factor applied to CFL stability limits | Source: RELIC transport module; Courant et al. (1928) |
| `transport.numerical.max_diffusion_stability` | `dimensionless` | `0.5; bounds=(0.1, 1.0)` | `fixed` | Maximum allowable stability number (CFL) for diffusion processes | Source: RELIC transport module; LeVeque (2007) Finite Difference Methods for Ordinary and Partial Differential Equations |
| `atmosphere.most.beta_stable` | `dimensionless` | `5.0; bounds=(4.0, 6.0)` | `fixed` | Shared stable-regime coefficient for Businger-Dyer momentum and scalar similarity functions | Source: literature; Businger et al. (1971); Dyer (1974) |
| `atmosphere.most.gamma_unstable` | `dimensionless` | `16.0; bounds=(12.0, 20.0)` | `fixed` | Shared unstable-regime coefficient for Businger-Dyer momentum and scalar similarity functions | Source: literature; Businger et al. (1971); Högström (1988) |
| `atmosphere.most.stable_correction_limit` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Configured positive zeta boundary for the RELIC Businger-Dyer operational domain; values outside require a different explicit similarity closure | Source: relic_configured_operational_envelope; Businger et al. (1971). Flux-profile relationships in the atmospheric surface layer; Foken (2006). 50 years of the Monin-Obukhov similarity theory |
| `atmosphere.most.unstable_correction_limit` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Configured absolute negative zeta boundary for the RELIC Businger-Dyer operational domain; values outside require a different explicit similarity closure | Source: relic_configured_operational_envelope; Businger et al. (1971). Flux-profile relationships in the atmospheric surface layer; Foken (2006). 50 years of the Monin-Obukhov similarity theory |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | UpdateItem(ACCUMULATE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{True} , \mathrm{unit} = ' \mathrm{mol} \mathrm{mol} - 1 ' )$ on canonical state keys |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\atmosphere_eddy_diffusion_blm.md`.

### References

- Businger, J.A. et al. (1971). Flux-profile relationships in surface layer. J. Atmos. Sci., 28, 181-189 (FunctorMetadata, certificate)
- Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology. Kluwer Academic (FunctorMetadata, certificate)
- Garratt, J.R. (1992). The Atmospheric Boundary Layer. Cambridge University Press (FunctorMetadata, certificate)
- Monin, A.S. & Obukhov, A.M. (1954). Basic laws of turbulent mixing. Trudy Geofiz. Inst. AN SSSR (FunctorMetadata, certificate)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)

(class-relic-scientific-atmosphere-turbulence-turbulencefunctor)=
## TurbulenceFunctor

Surface layer turbulence functor using Monin-Obukhov Similarity Theory.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.atmosphere.turbulence.TurbulenceFunctor` |
| Source | `relic\scientific\atmosphere\turbulence.py:103` |
| Runtime name | `atmospheric_turbulence` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
TurbulenceFunctor(name: str = 'atmospheric_turbulence', config: Dict[str, Any] | None = None, **kwargs)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'atmospheric_turbulence'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `wind_speed` | forcing | `m s-1` | explicit wind-speed forcing [m s-1] |
| `air_temperature` | forcing | `K` | explicit air-temperature forcing [K] |
| `surface_temperature` | forcing | `K` | explicit surface-temperature forcing [K] |
| `reference_height` | forcing | `m` | explicit observation height [m] |
| `roughness_length_momentum` | forcing | `m` | explicit momentum roughness length [m] |
| `displacement_height` | forcing | `m` | explicit zero-plane displacement height [m] |
| `atmospheric_pressure` | forcing | `Pa` | explicit atmospheric pressure forcing p [Pa] of the moist-air density; no default |
| `relative_humidity` | forcing | `1` | explicit relative humidity forcing RH [1] of the vapour pressure e = RH*e_s(Ta); no default |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `friction_velocity` | diagnostic | `m s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `aerodynamic_resistance` | diagnostic | `s m-1` | decided at run time by the returned write mode | canonical scalar heat and mass resistance rah = fh/(k*u*) |
| `diagnostics.turbulence.stability_parameter` | diagnostic | `dimensionless` | diagnostics | canonical namespaced z over L diagnostic |
| `monin_obukhov_length` | diagnostic | `m` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `sensible_heat_flux` | flux | `W m-2` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `momentum_flux` | flux | `kg m-1 s-2` | decided at run time by the returned write mode | surface stress port flow tau = rho_air u_star^2 with rho_air the moist-air density of air_temperature, atmospheric_pressure and relative_humidity |
| `diagnostics.turbulence.drag_coefficient` | diagnostic | `dimensionless` | diagnostics | canonical namespaced neutral drag diagnostic |

### Governing equations and canonical operator

**Literature or process form**

$$
f_m = \ln ( ( z - d ) / \mathrm{z0m} ) - \psi_m ( ( z - d ) / L ) + \psi_m ( \mathrm{z0m} / L ); f_h = \ln ( ( z - d ) / \mathrm{z0h} ) - \psi_h ( ( z - d ) / L ) + \psi_h ( \mathrm{z0h} / L ); u_* = k \cdot U / f_m; \theta_* = k \cdot ( \mathrm{Ts} - \mathrm{Ta} ) / f_h; H = \rho \cdot \mathrm{cp} \cdot u_* \theta_*; L = - \rho \cdot \mathrm{cp} \cdot T \cdot u_* ^ 3 / ( k \cdot g \cdot H ); r_{ah} = f_h / ( k \cdot u_* ); \tau = \rho \cdot u_* ^ 2; \rho = ( p - ( 1 - \varepsilon ) \cdot \mathrm{RH} \cdot e_{s} ( \mathrm{Ta} ) ) / ( R_{d} \cdot \mathrm{Ta} )
$$

**Canonical RELIC structural form**

$$
\dot{x} = - D_{h} q_{h} + G_{h} u_{h}, q_{h} = - \star_{\tau} D_{h} ^{\mathsf{T}} e_{h}, y_{h} = G_{h} ^{\mathsf{T}} e_{h}
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | existing divergence/transport operator |
| `q_h` | declared flux cochain |
| `G_h` | boundary/source port map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = - D_{h} q_{h} + G_{h} u_{h}$ with declared momentum flux $q_{h}$ and surface-layer exchange port |
| `runtime_usage` | TurbulenceFunctor.forward evaluates the single canonical MOST turbulence calculation. Friction velocity, stability, aerodynamic resistance, and momentum flux are exposed as ParameterManager-backed transport closures and diagnostics, not as a second state-advance branch. |

### Composition channels and executable PortSpec API

Certificate composition channels: `surface_layer_momentum_exchange_port`

Executable `ports()` owner: `TurbulenceFunctor`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `surface_layer_momentum_exchange_port` | `wind_speed [m s-1]` | `momentum_flux [kg m-1 s-2]` | `kg s-3` | out_of_system; momentum |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `atmosphere.turbulence.default_reference_height` | `m` | `10.0; bounds=(1.0, 100.0)` | `fixed` | Default measurement height above surface for meteorological observations | Source: standard_practice; Chave et al. (2014) Glob. Change Biol. 20, 3177-3190 |
| `atmosphere.turbulence.roughness_heat_ratio` | `dimensionless` | `0.1; bounds=(0.01, 1.0)` | `learnable` | Ratio of heat to momentum roughness length (z0h/z0m) | Source: literature; Brutsaert (1982) Evaporation into the Atmosphere, Springer |
| `atmosphere.turbulence.minimum_wind_speed` | `m s-1` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Minimum wind speed threshold for turbulence calculations to avoid division by zero | Source: numerical_stability; Monin & Obukhov (1954) Trudy Akad. Nauk SSSR Geofiz. Inst. 24, 163-187 |
| `atmosphere.turbulence.minimum_friction_velocity` | `m s-1` | `0.01; bounds=(0.001, 0.1)` | `fixed` | Positive reduced friction-velocity scale used by atmospheric scalar transport's MOST diagnostic and structural Hodge audits; not a measured wind input. | Source: RELIC parameter database; Stull (1988) An Introduction to Boundary Layer Meteorology, Kluwer; Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling, Cambridge Univ. Press |
| `atmosphere.turbulence.max_iterations` | `dimensionless` | `10; bounds=(3.0, 50.0)` | `fixed` | Maximum iterations for convergence in iterative turbulence algorithms | Source: numerical_method; Press et al. (2007) Numerical Recipes 3rd Ed., Cambridge Univ. Press |
| `atmosphere.turbulence.convergence_tolerance` | `dimensionless` | `0.01; bounds=(0.0001, 0.1)` | `fixed` | Convergence tolerance for iterative solutions in MOST calculations | Source: numerical_method; Press et al. (2007) Numerical Recipes 3rd Ed., Cambridge Univ. Press |
| `atmosphere.turbulence.neutral_obukhov_length` | `m` | `1000.0; bounds=(100.0, 10000.0)` | `fixed` | Finite magnitude used to represent the neutral Monin-Obukhov limit | Source: numerical_method; Stull (1988) An Introduction to Boundary Layer Meteorology; Garratt (1992) The Atmospheric Boundary Layer |
| `atmosphere.most.beta_stable` | `dimensionless` | `5.0; bounds=(4.0, 6.0)` | `fixed` | Shared stable-regime coefficient for Businger-Dyer momentum and scalar similarity functions | Source: literature; Businger et al. (1971); Dyer (1974) |
| `atmosphere.most.gamma_unstable` | `dimensionless` | `16.0; bounds=(12.0, 20.0)` | `fixed` | Shared unstable-regime coefficient for Businger-Dyer momentum and scalar similarity functions | Source: literature; Businger et al. (1971); Högström (1988) |
| `atmosphere.most.stable_correction_limit` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Configured positive zeta boundary for the RELIC Businger-Dyer operational domain; values outside require a different explicit similarity closure | Source: relic_configured_operational_envelope; Businger et al. (1971). Flux-profile relationships in the atmospheric surface layer; Foken (2006). 50 years of the Monin-Obukhov similarity theory |
| `atmosphere.most.unstable_correction_limit` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Configured absolute negative zeta boundary for the RELIC Businger-Dyer operational domain; values outside require a different explicit similarity closure | Source: relic_configured_operational_envelope; Businger et al. (1971). Flux-profile relationships in the atmospheric surface layer; Foken (2006). 50 years of the Monin-Obukhov similarity theory |
| `atmosphere.most.h_min_safeguard` | `W m-2` | `1e-06; bounds=(1e-06, 1e-06)` | `fixed` | Minimum absolute sensible heat flux used to avoid singular Monin-Obukhov length. | Source: numerical_stability_guidelines; Monin & Obukhov (1954); Stull (1988) |
| `atmosphere.most.monin_obukhov_length_clamp_min` | `m` | `-1000.0; bounds=(-5000.0, -10.0)` | `fixed` | Lower clamp bound for Monin-Obukhov length in MOST solvers. | Source: numerical_stability_guidelines; Stull (1988) |
| `atmosphere.most.monin_obukhov_length_clamp_max` | `m` | `1000.0; bounds=(10.0, 5000.0)` | `fixed` | Upper clamp bound for Monin-Obukhov length in MOST solvers. | Source: numerical_stability_guidelines; Stull (1988) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | all seven outputs use UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{True} )$ with explicit units and writer provenance |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Atmosphere theory](../theory/atmosphere.md).

Certificate theory owner: `docs\source\theory\turbulence.md`.

### References

- Monin, A.S. & Obukhov, A.M. (1954). Basic laws of turbulent mixing (FunctorMetadata, certificate)
- Businger, J.A. et al. (1971). Flux-profile relationships in the atmospheric surface layer (FunctorMetadata, certificate)
- Garratt, J.R. (1992). The Atmospheric Boundary Layer (FunctorMetadata, certificate)
- Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology (FunctorMetadata, certificate)
- Holton, J.R. & Hakim, G.J. (2012). An Introduction to Dynamic Meteorology, 5th ed., Eqs. 2.63-2.64 (FunctorMetadata, certificate)
- Allen, R.G. et al. (1998). FAO Irrigation and Drainage Paper 56, Eqs. 11 and 19 (FunctorMetadata, certificate)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
