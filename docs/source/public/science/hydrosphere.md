# Hydrosphere Scientific Classes

This reference covers all 14 certificate-backed scientific classes in `relic.scientific.hydrosphere`: 14 executable concrete Functors, 0 fail-closed unavailable boundaries, and 0 abstract contracts. Each entry is generated from the live class, its IPHS certificate, State Contract Model, and ParameterManager metadata. Fail-closed unavailable classes are not instantiated during documentation generation.

The tables distinguish constructor controls, runtime variables, and ParameterManager scientific parameters. Values shown as package defaults belong to the documented RELIC architecture version and must not be copied as hardcoded constants into scientific code.

Certificate wildcard parameter groups (`*`, `<pft>`) are expanded to their exact current YAML entries. Every other declared parameter is matched to the parameter database by its exact full key; a declared name is never mapped to a key that merely ends with the same name.

Exact ParameterManager rows report the live YAML value, bounds, gradient policy, description, and provenance. A row whose value reads `not declared in the parameter database` is intentionally not assigned an invented default: the class metadata or certificate declares the quantity, with the meaning and units it states, but the parameter database has no key of that exact name.

IPHS levels are read from the sole certificate owner. Levels 1 to 3 describe compatibility, wrapping, or declared structure; they are not strict IPHS/GENERIC claims.

The Channel column of each runtime output contract applies `relic.core.state.types.partition_outputs`, the only runtime output channel classifier, to the declared output name. A name that the classifier reserves for diagnostics, such as `diagnostics.*`, goes to the diagnostics channel and is never committed as state. For any other name the channel is decided at run time by the write mode of the returned value: `StateWriteMode.DERIVED` routes the output to the diagnostics channel, and any other write mode sends it to the StateManager commit (a value without a write mode takes the run's semantic-mode rules or the StateManager schema). The generator does not run `forward()`, so it does not predict that write mode. The Kind column is the State Contract Model kind of the variable and does not set its channel.

(class-relic-scientific-hydrosphere-evapotranspiration-penman-monteith-penmanmonteithet)=
## penman_monteith_et

Penman-Monteith Evapotranspiration Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.evapotranspiration.penman_monteith.PenmanMonteithET` |
| Source | `relic\scientific\hydrosphere\evapotranspiration\penman_monteith.py:175` |
| Runtime name | `penman_monteith_et` |
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
penman_monteith_et(name: str = 'penman_monteith_et', config: Dict | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'penman_monteith_et'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Configuration modes

The declared runtime interface depends on the constructor configuration; there is no default mode. Each documented mode was instantiated. The table lists the variables that only that mode declares; every other variable of the runtime contracts below is declared in all modes.

| Mode | Inputs declared only in this mode | Outputs declared only in this mode |
| --- | --- | --- |
| `canopy_resistance_source='state'` | `surface_resistance`, `canopy_conductance` | none |
| `canopy_resistance_source='fao56_reference_surface'` | none | none |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `net_radiation` | flux | `W m-2` | net radiation Rn [W m-2] |
| `soil_heat_flux` | forcing | `W m-2` | signed surface G, positive into soil; required only with soil_heat_flux_source state [W m-2] |
| `air_temperature` | forcing | `K` | air temperature T [K] |
| `wind_speed` | forcing | `m s-1` | wind speed u [m s-1] |
| `atmospheric_pressure` | forcing | `Pa` | surface atmospheric pressure P [Pa] |
| `relative_humidity` | forcing | `1` | relative humidity RH [dimensionless] |
| `vapor_pressure_deficit` | forcing | `Pa` | vapor pressure deficit VPD [Pa] |
| `surface_resistance` | forcing | `s m-1` | canopy (surface) resistance r_s, read only with canopy_resistance_source state [s m-1] |
| `canopy_conductance` | state | `m s-1` | canopy conductance g_c, r_s = 1 / g_c when surface_resistance is absent; read only with canopy_resistance_source state [m s-1] |
| `aerodynamic_resistance` | diagnostic | `s m-1` | aerodynamic resistance r_a [s m-1] |
| `precipitation_flux` | forcing | `kg m-2 s-1` | precipitation water flux P, read only with canopy interception enabled [kg m-2 s-1] |
| `canopy_storage` | state | `mm` | interception storage [mm] |
| `leaf_area_index` | state | `m2 m-2` | leaf area index LAI, read only with canopy interception enabled [m2 m-2] |
| `air_temperature_max` | forcing | `K` | daily maximum air temperature, read with temporal_support daily (FAO-56 Eq. 12 e_s) and daylight_gauss (diurnal course T_k) |
| `air_temperature_min` | forcing | `K` | daily minimum air temperature, read with temporal_support daily (FAO-56 Eq. 12 e_s) and daylight_gauss (diurnal course T_k) |
| `actual_vapor_pressure` | forcing | `Pa` | actual vapour pressure e_a, read only with temporal_support daylight_gauss |
| `canopy_conductance_daylight_samples` | diagnostic | `m s-1` | per-sample canopy conductance g_c,k, read only with temporal_support daylight_gauss |
| `shortwave_net` | flux | `W m-2` | daily net shortwave S_n redistributed over the daylight samples, read only with temporal_support daylight_gauss |
| `latitude` | forcing | `degrees_north` | latitude of the cell, read only with temporal_support daylight_gauss |
| `day_of_year` | forcing | `1` | day of year, read only with temporal_support daylight_gauss |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `evapotranspiration_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | water vapor mass flux q_v [kg m-2 s-1] |
| `latent_heat_flux` | flux | `W m-2` | decided at run time by the returned write mode | latent heat flux lambda q_v [W m-2] |
| `diagnostics.vapor_pressure_deficit` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.aerodynamic_conductance` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.canopy_conductance` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `canopy_storage` | state | `mm` | decided at run time by the returned write mode | interception storage [mm] |
| `diagnostics.evapotranspiration` | diagnostic | `dimensionless` | diagnostics | read-only aerodynamic, psychrometric, and residual diagnostics |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Input declaration difference: `actual_vapor_pressure`, `air_temperature_max`, `air_temperature_min`, `canopy_conductance_daylight_samples`, `day_of_year`, `latitude`, `shortwave_net`.
:::

### Governing equations and canonical operator

**Literature or process form**

$\mathrm{ET} = [ \Delta ( R_n - G ) + \rho_{a} \mathrm{cp} \mathrm{VPD} \mathrm{ga} ] / [ \lambda ( \Delta + \gamma ( 1 + \mathrm{ga} / \mathrm{gc} ) ) ]$, following Penman, Monteith, and FAO-56.

**Canonical RELIC structural form**

$q_{v} = \star_{\mathrm{ET}} X_{v}$ with $X_{v} = \mathrm{VPD}$ and $\star_{\mathrm{ET}} = \operatorname{diag} ( g_{\mathrm{eff}} / \lambda ) \geq 0$; latent heat port $L_{E} = \lambda q_{v} .$

| Entry | Meaning |
| --- | --- |
| `star_ET` | build_hodge_contract and build_hodge_terms HodgeStarSpec for vapor transport mobility |
| `psychrometrics` | Tetens saturation vapor pressure, saturation slope, and psychrometric constant |
| `aerodynamic_closure` | logarithmic wind profile or externally supplied aerodynamic resistance |
| `moist_air_density` | $\rho_{a} = ( p - ( 1 - \epsilon ) e_{a} ) / ( R_{d} T )$ from compute_moist_air_density (Holton and Hakim 2012 Eqs. 2.63-2.64; FAO-56 Annex 3) in the aerodynamic term of the actual and the reference ET |
| `soil_heat_flux` | Explicit source: state consumes finite signed surface G; fraction uses $G = c_{\mathrm{day}} R_n \mathrm{if} R_n > 0$, else $G = c_{\mathrm{night}} R_n ( \mathrm{FAO} - 56 \mathrm{Eqs} . 45 - 46 ) . \mathrm{Both} \mathrm{retain} \mathrm{soil}_{\mathrm{heat},\mathrm{flux},\mathrm{hourly},\mathrm{max},\mathrm{step}}$ and feed one PM law. |
| `canopy_resistance` | declared canopy_resistance_source; state uses surface_resistance or 1 / canopy_conductance, fao56_reference_surface uses the FAO-56 reference grass $r_{s}$; no implicit default |
| `canopy_partition` | optional Rutter/Gash-style interception storage partition |
| `projection` | nonnegative ET and canopy storage residual routing |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | surface vapor and available-energy state cochain $x_{v}$ |
| `effort_variable` | aerodynamic-surface resistance effort $X_{v}$ with positive ET mobility star_ET |
| `state_advancer` | $q_{v} = \star_{\mathrm{ET}} X_{v}$; $L_{E} = \lambda q_{v}$; $\dot{\mathrm{water}_{\mathrm{storage}}} = - q_{v}$ |
| `runtime_usage` | forward applies the canonical FAO-56 ET port relation and routes water and energy residual diagnostics. |

### Composition channels and executable PortSpec API

Certificate composition channels: `penman_monteith_vapor_flux`, `penman_monteith_latent_heat_flux`

Executable `ports()` owner: `PenmanMonteithET`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `penman_monteith_vapor_flux` | `vapor_pressure_deficit [Pa]` | `evapotranspiration_flux [kg m-2 s-1]` | `Pa kg m-2 s-1` | out_of_system; matter |
| `penman_monteith_latent_heat_flux` | `surface_energy_gradient [Pa]` | `latent_heat_flux [W m-2]` | `W m-2` | out_of_system; energy |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.evapotranspiration.soil_heat_flux_source` | `dimensionless` | `'fraction'; bounds=None` | `fixed` | Explicit ground-heat source for Priestley-Taylor and Penman-Monteith; fraction uses the process-specific documented radiation fraction, state requires signed surface soil_heat_flux without fallback | Source: explicit_surface_energy_input_contract; Priestley and Taylor (1972) Monthly Weather Review 100:81-92; Allen et al. (1998) FAO Irrigation and Drainage Paper 56, soil heat flux |
| `hydrosphere.evapotranspiration.roughness_length` | `m` | `2.0; bounds=(0.001, 10.0)` | `learnable` | Momentum roughness length for forest canopy aerodynamics | Source: literature_synthesis; FAO-56 Irrigation and Drainage Paper; Brutsaert (1982) Evaporation into the Atmosphere |
| `hydrosphere.evapotranspiration.displacement_height` | `m` | `13.4; bounds=(0.0, 50.0)` | `fixed` | Zero plane displacement height d of the FAO-56 logarithmic wind profile (FAO-56 relates d = 2/3 h for a canopy of height h; RELIC reads d directly) | Source: empirical_relationship; FAO-56 Irrigation and Drainage Paper; Thom (1975) Boundary Layer Meteorology 9:319-343 |
| `hydrosphere.evapotranspiration.minimum_conductance` | `m s-1` | `1e-06; bounds=(1e-09, 0.001)` | `fixed` | Minimum canopy conductance to prevent division by zero in calculations | Source: numerical_implementation; Technical parameter from model implementation |
| `hydrosphere.evapotranspiration.reference_height` | `m` | `30.0; bounds=(2.0, 200.0)` | `fixed` | Reference height for wind speed measurements | Source: measurement_standard; FAO-56 Irrigation and Drainage Paper; WMO Guide to Meteorological Instruments |
| `hydrosphere.evapotranspiration.enable_canopy_interception` | `dimensionless` | `False; bounds=(0.0, 1.0)` | `fixed` | Enable canopy interception modeling (Rutter/Gash-style interception) | Source: interception_modeling_literature; Rutter et al. (1971) J. Appl. Ecol. 8:667-689; Gash (1979) Q. J. R. Meteorol. Soc. 105:43-55 |
| `hydrosphere.evapotranspiration.interception_capacity_per_LAI` | `mm` | `0.2; bounds=(0.05, 2.0)` | `learnable` | Canopy water storage capacity per unit Leaf Area Index | Source: interception_modeling_literature; Rutter et al. (1971) J. Appl. Ecol. 8:667-689; Herwitz (1985) Water Resour. Res. 21:1465-1472 |
| `hydrosphere.evapotranspiration.canopy_evaporation_efficiency` | `dimensionless` | `1.0; bounds=(0.5, 1.5)` | `learnable` | Enhanced evaporation efficiency factor for wet canopy surfaces | Source: canopy_evaporation_theory; Shuttleworth & Calder (1979) J. Hydrol. 41:163-178; Stewart (1977) Water Resour. Res. 13:915-921 |
| `hydrosphere.evapotranspiration.hodge_reference_aerodynamic_conductance` | `m s-1` | `0.01; bounds=(0.001, 0.1)` | `fixed` | Reference aerodynamic conductance g_a of the static Penman-Monteith vapor Hodge contract (PenmanMonteithET.build_hodge_contract, star_ET = diag(g_eff / lambda) with g_eff = g_a g_c / (g_a + g_c)); never substituted for the runtime g_a, which is the aerodynamic_resistance state or the FAO-56 log wind profile | Source: numerical_implementation; Typical low-wind conductance values from boundary layer meteorology; Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56 |
| `hydrosphere.physical.soil_heat_flux_ratio_daytime` | `dimensionless` | `0.1; bounds=(0.04, 0.15)` | `learnable` | Ratio G/Rn of the hourly (or shorter) soil heat flux during daylight, Rn > 0 (FAO-56 Eq. 45; the ASCE-EWRI tall reference uses 0.04) | Source: fao_56_standard; Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56, Eqs. 45-46; ASCE-EWRI (2005). The ASCE Standardized Reference Evapotranspiration Equation, Table 1 |
| `hydrosphere.physical.soil_heat_flux_ratio_nighttime` | `dimensionless` | `0.5; bounds=(0.2, 0.6)` | `learnable` | Ratio G/Rn of the hourly (or shorter) soil heat flux during nighttime, Rn <= 0 (FAO-56 Eq. 46; the ASCE-EWRI tall reference uses 0.2) | Source: fao_56_standard; Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56, Eqs. 45-46; ASCE-EWRI (2005). The ASCE Standardized Reference Evapotranspiration Equation, Table 1 |
| `hydrosphere.physical.soil_heat_flux_hourly_max_step` | `s` | `3600.0; bounds=(3600.0, 3600.0)` | `fixed` | Longest step for which the FAO-56 hourly soil heat flux fractions G = 0.1 Rn (daylight) and G = 0.5 Rn (night) are defined; a longer step lies outside the hourly formulation | Source: fao_56_standard; Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56, Eqs. 45-46 (hourly or shorter periods) and Eq. 42 (day-long periods) |
| `hydrosphere.physical.reference_surface_resistance` | `s m-1` | `70.0; bounds=(50.0, 100.0)` | `fixed` | Surface resistance for reference grass (FAO-56 reference ET) | Source: fao_56_standard; Allen et al. (1998) FAO-56; Monteith (1981) Quart. J. Roy. Meteorol. Soc. 107:1-27 |
| `hydrosphere.physical.heat_to_momentum_roughness_ratio` | `dimensionless` | `0.1; bounds=(0.05, 0.2)` | `fixed` | Ratio of roughness length for heat to momentum (z0h/z0m) | Source: boundary_layer_theory; Brutsaert (1982) Evaporation into the Atmosphere; Garratt (1992) The Atmospheric Boundary Layer |
| `hydrosphere.physical.maximum_lai_factor` | `m2 m-2` | `6.0; bounds=(1.0, 10.0)` | `fixed` | Representative upper bound on LAI used for interception diagnostics and capacity checks | Source: canopy_interception_references; Gash (1979) J. Hydrol. 47:15-27; Wang et al. (2007) Agric. For. Meteorol. 143:122-138 |
| `hydrosphere.physical.moderate_lai_factor` | `m2 m-2` | `3.0; bounds=(1.0, 6.0)` | `fixed` | Representative LAI used when scaling interception capacity for mixed evergreen stands | Source: canopy_interception_references; Rutter et al. (1971) J. Appl. Ecol. 8:667-688; Gash (1979) Q. J. R. Meteorol. Soc. 105:43-55 |
| `hydrosphere.physical.atmospheric_pressure_min` | `Pa` | `44000.0; bounds=(30000.0, 110000.0)` | `fixed` | Minimum plausible surface atmospheric pressure used to guard hydrosphere flux calculations: US Standard Atmosphere 1976 pressure at 6,500 m, above the highest vascular plants (about 6,150 m, Himalaya). Lowered from 70,000 Pa on 2026-11-05 for vegetated plateaus (Tibet, Andes); the psychrometric constant gamma = c_p p / (epsilon L) holds at any pressure. | Source: atmospheric_climatology; US Standard Atmosphere 1976 |
| `hydrosphere.physical.atmospheric_pressure_max` | `Pa` | `110000.0; bounds=(80000.0, 120000.0)` | `fixed` | Maximum plausible surface atmospheric pressure used to guard hydrosphere flux calculations: above the Dead Sea shore (-430 m, about 106.5 kPa) under a strong anticyclone. Raised from 106,000 Pa on 2026-11-05. | Source: atmospheric_climatology; US Standard Atmosphere 1976 |
| `hydrosphere.numerical.temperature_bounds` | `K` | `[273.15, 323.15]; bounds=([183.15, 273.15], [323.15, 373.15])` | `fixed` | Admitted near-surface air temperature for hydrosphere calculations [min, max]. The default keeps 273.15 K as minimum because the default sub-freezing phase of the evapotranspiration laws is liquid. A run with sub_freezing_phase ice may set the minimum down to 183.15 K (range widened on 2026-11-05 from 223.15 K for global daily runs, terrestrial record low -89.2 degC): the Tetens form over ice deviates from Goff-Gratch by -1.6 percent at 233.15 K, -4.7 percent at 213.15 K and -7.2 percent at 203.15 K, where the saturation vapour pressure is below 0.3 Pa. | Source: meteorological_standards; WMO Guide to Meteorological Instruments; WMO Archive of Weather and Climate Extremes |
| `hydrosphere.numerical.net_radiation_bounds` | `W m-2` | `[-500.0, 1500.0]; bounds=([-2000.0, 0.0], [1000.0, 2500.0])` | `fixed` | Reasonable net radiation bounds for ET calculations [min, max] | Source: energy_balance_validation; Allen et al. (1998) FAO Irrigation and Drainage Paper 56; Monteith (1965) Symp. Soc. Exp. Biol. 19:205-224 |
| `hydrosphere.numerical.minimum_wind_speed_offset` | `m s-1` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Offset added to wind speed for validation to allow very low wind conditions | Source: numerical_implementation; Meteorological Modeling Best Practices |
| `hydrosphere.numerical.aerodynamic_conductance_bounds` | `m s-1` | `[0.001, 1.0]; bounds=([0.0001, 0.01], [0.1, 10.0])` | `fixed` | Physical bounds for aerodynamic conductance [min, max] | Source: boundary_layer_meteorology; Monteith (1965) Symp. Soc. Exp. Biol. 19:205-224; Allen et al. (1998) FAO-56 Irrigation Paper |
| `hydrosphere.numerical.evapotranspiration_flux_clamp_min` | `kg m-2 s-1` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Minimum evapotranspiration flux (non-negative constraint for physical validity) | Source: physical_constraint; Physical constraint - evapotranspiration cannot be negative |
| `hydrosphere.numerical.minimum_wind_speed` | `m s-1` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Minimum wind speed for aerodynamic calculations to prevent division by zero | Source: meteorological_modeling; Allen et al. (1998) FAO-56 Irrigation Paper; Monteith (1965) Symp. Soc. Exp. Biol. 19:205-224 |
| `hydrosphere.numerical.canopy_storage_bounds` | `mm` | `[0.01, 10.0]; bounds=([0.001, 0.1], [5.0, 20.0])` | `fixed` | Physical bounds for canopy water storage capacity [min, max] | Source: canopy_hydrology_literature; Rutter et al. (1971) J. Appl. Ecol. 8:667-689; Gash (1979) Q. J. R. Meteorol. Soc. 105:43-55 |
| `hydrosphere.numerical.mass_conservation_tolerance_mm` | `mm` | `1.0; bounds=(0.0, 10.0)` | `fixed` | Absolute mass balance tolerance for canopy interception checks in millimeters | Source: canopy_water_balance; Rutter et al. (1971) J. Appl. Ecol. 8:667-689; Gash (1979) Q. J. R. Meteorol. Soc. 105:43-55 |
| `hydrosphere.numerical.critical_mass_balance_error_threshold` | `mm` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Critical error threshold for mass balance violations in canopy processes | Source: conservation_law_theory; Numerical Methods for Conservation Laws |
| `hydrosphere.numerical.minimum_drainage_flux` | `kg m-2 s-1` | `1e-08; bounds=(1e-12, 1e-06)` | `fixed` | Minimum threshold for drainage flux calculations | Source: hydrological_modeling; Soil Water Dynamics Literature |
| `validation.numerical.clamp_min_zero` | `dimensionless` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Minimum clamp value for zero-bounded quantities | Source: numerical_methods; Numerical methods best practices |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | formal outputs use explicit UpdateItem(REPLACE) values through the existing StateManager path |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_penman_monteith.md`.

### References

- Monteith, J.L. (1965). Evaporation and environment. Symp. Soc. Exp. Biol. 19, 205-224 (FunctorMetadata)
- Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation paper 56 (FunctorMetadata)
- Penman, H.L. (1948). Natural evaporation from open water, bare soil and grass (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Penman, H.L. (1948). Natural evaporation from open water, bare soil and grass. Proceedings of the Royal Society A, 193, 120-145. (certificate)
- Monteith, J.L. (1965). Evaporation and environment. Symposia of the Society for Experimental Biology, 19, 205-234. (certificate)
- Allen, R.G., Pereira, L.S., Raes, D., and Smith, M. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56. (certificate)
- Rutter, A.J. et al. (1971). A predictive model of rainfall interception in forests. Agricultural Meteorology, 9, 367-384. (certificate)
- Gash, J.H.C. (1979). An analytical model of rainfall interception by forests. Quarterly Journal of the Royal Meteorological Society, 105, 43-55. (certificate)

(class-relic-scientific-hydrosphere-evapotranspiration-priestley-taylor-priestleytayloret)=
## priestley_taylor_et

Priestley-Taylor Evapotranspiration Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.evapotranspiration.priestley_taylor.PriestleyTaylorET` |
| Source | `relic\scientific\hydrosphere\evapotranspiration\priestley_taylor.py:91` |
| Runtime name | `priestley_taylor_et` |
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
priestley_taylor_et(name: str = 'priestley_taylor_et', config: Dict | None = None, device: torch.device | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'priestley_taylor_et'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `net_radiation` | flux | `W m-2` | net radiation Rn [W m-2] |
| `air_temperature` | forcing | `K` | air temperature T [K] |
| `atmospheric_pressure` | forcing | `Pa` | surface atmospheric pressure P [Pa] |
| `soil_heat_flux` | forcing | `W m-2` | signed soil-surface G [W m-2], required only for the explicit state source |
| `soil_moisture` | state | `m3 m-3` | volumetric soil water of the dynamic alpha, read only when dynamic_alpha_inputs declares it |
| `vapor_pressure_deficit` | forcing | `Pa` | vapour pressure deficit of the dynamic alpha, read only when dynamic_alpha_inputs declares it |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `evapotranspiration_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | energy-limited water vapor mass flux q_eq [kg m-2 s-1] |
| `latent_heat_flux` | flux | `W m-2` | decided at run time by the returned write mode | latent heat flux lambda q_eq [W m-2] |
| `diagnostics.evapotranspiration` | diagnostic | `dimensionless` | diagnostics | read-only psychrometric, alpha, energy, and residual diagnostics |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Input declaration difference: `soil_moisture`, `vapor_pressure_deficit`.
:::

### Governing equations and canonical operator

**Literature or process form**

$\mathrm{ET} = \alpha \cdot ( \Delta / ( \Delta + \gamma ) ) \cdot ( R_n - G ) / \lambda$, following Priestley and Taylor (1972).

**Canonical RELIC structural form**

$q_{\mathrm{eq}} = \star_{\mathrm{PT}} A$ with $A = R_n - G$ and $\star_{\mathrm{PT}} = \operatorname{diag} ( \alpha \cdot \Delta / ( ( \Delta + \gamma ) \cdot \lambda ) ) \geq 0$; latent heat port $L_{E} = \lambda q_{\mathrm{eq}} .$

| Entry | Meaning |
| --- | --- |
| `star_PT` | build_hodge_contract and build_hodge_terms HodgeStarSpec for energy-limited ET mobility |
| `psychrometrics` | Tetens saturation vapor pressure, saturation slope, and psychrometric constant; below 273.15 K the ice option evaluates slope and psychrometric constant over ice (Murray 1967 coefficients, latent heat of sublimation) |
| `alpha_closure` | fixed or dynamic Priestley-Taylor alpha coefficient |
| `energy_partition` | $G = \mathrm{soil}_{\mathrm{heat},\mathrm{fraction}} \cdot R_n$ for the fraction source or $G = \mathrm{state} \mathrm{soil}_{\mathrm{heat},\mathrm{flux}}$ for the state source; both use $A = R_n - G$ and one ET expression |
| `projection` | nonnegative ET residual routing |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | available-energy cochain $A_{h}$ |
| `effort_variable` | equilibrium vapor effort with Priestley-Taylor mobility star_PT |
| `state_advancer` | $q_{\mathrm{eq}} = \star_{\mathrm{PT}} A_{h}$; $L_{E} = \lambda q_{\mathrm{eq}}$; $\dot{\mathrm{water}_{\mathrm{storage}}} = - q_{\mathrm{eq}}$ |
| `runtime_usage` | forward applies the canonical energy-limited ET port relation and routes projection and energy residual diagnostics. |

### Composition channels and executable PortSpec API

Certificate composition channels: `priestley_taylor_equilibrium_flux`, `priestley_taylor_latent_heat_flux`

Executable `ports()` owner: `PriestleyTaylorET`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `priestley_taylor_equilibrium_flux` | `available_energy [W m-2]` | `evapotranspiration_flux [kg m-2 s-1]` | `W kg m-4 s-1` | out_of_system; matter |
| `priestley_taylor_latent_heat_flux` | `available_energy [W m-2]` | `latent_heat_flux [W m-2]` | `W m-2` | out_of_system; energy |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.soil_moisture.porosity` | `m3 m-3` | `0.45; bounds=(0.3, 0.8)` | `fixed` | Total soil porosity (saturated water content) | Source: soil_survey_data; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.residual_water_content` | `m3 m-3` | `0.045; bounds=(0.0, 0.2)` | `fixed` | Residual water content (minimum water content at very high suction) | Source: laboratory_measurements; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Schaap et al. (2001) J. Hydrol. 251:163-176 |
| `hydrosphere.physical.field_capacity_fraction` | `dimensionless` | `0.7; bounds=(0.5, 0.9)` | `learnable` | Field capacity as fraction of saturated water content (typical for loam) | Source: soil_physics_literature; Brady & Weil (2008) The Nature and Properties of Soils, 14th ed; Saxton & Rawls (2006) Soil Sci. Soc. Am. J. 70:1569-1578 |
| `hydrosphere.physical.wilting_point_fraction` | `dimensionless` | `1.5; bounds=(1.2, 2.0)` | `learnable` | Wilting point as multiplier of residual water content (typical for loam) | Source: soil_physics_literature; Brady & Weil (2008) The Nature and Properties of Soils, 14th ed; Saxton & Rawls (2006) Soil Sci. Soc. Am. J. 70:1569-1578 |
| `hydrosphere.physical.vpd_alpha_baseline` | `dimensionless` | `1.0; bounds=(0.6, 1.3)` | `learnable` | Baseline scaling factor for Priestley-Taylor alpha reduction as a function of vapor pressure deficit | Source: evapotranspiration_stress_response; Katerji & Rana (2011) Agric. For. Meteorol. 151:428-440 |
| `hydrosphere.physical.vpd_alpha_sensitivity` | `kPa-1` | `0.1; bounds=(0.02, 0.3)` | `learnable` | Sensitivity of Priestley-Taylor alpha to vapor pressure deficit above the threshold | Source: evapotranspiration_stress_response; Katerji & Rana (2011) Agric. For. Meteorol. 151:428-440; Fisher et al. (2008) Water Resour. Res. 44:W00H14 |
| `hydrosphere.physical.vpd_alpha_threshold` | `kPa` | `1.0; bounds=(0.3, 2.0)` | `learnable` | Vapor pressure deficit threshold beyond which Priestley-Taylor alpha is reduced | Source: evapotranspiration_stress_response; Gentine et al. (2016) Nature 529:400-403; Katerji & Rana (2011) Agric. For. Meteorol. 151:428-440 |
| `hydrosphere.physical.atmospheric_pressure_min` | `Pa` | `44000.0; bounds=(30000.0, 110000.0)` | `fixed` | Minimum plausible surface atmospheric pressure used to guard hydrosphere flux calculations: US Standard Atmosphere 1976 pressure at 6,500 m, above the highest vascular plants (about 6,150 m, Himalaya). Lowered from 70,000 Pa on 2026-11-05 for vegetated plateaus (Tibet, Andes); the psychrometric constant gamma = c_p p / (epsilon L) holds at any pressure. | Source: atmospheric_climatology; US Standard Atmosphere 1976 |
| `hydrosphere.physical.atmospheric_pressure_max` | `Pa` | `110000.0; bounds=(80000.0, 120000.0)` | `fixed` | Maximum plausible surface atmospheric pressure used to guard hydrosphere flux calculations: above the Dead Sea shore (-430 m, about 106.5 kPa) under a strong anticyclone. Raised from 106,000 Pa on 2026-11-05. | Source: atmospheric_climatology; US Standard Atmosphere 1976 |
| `hydrosphere.evapotranspiration.alpha_pt` | `dimensionless` | `1.26; bounds=(0.6, 1.6)` | `learnable` | Priestley-Taylor equilibrium evapotranspiration coefficient under well-watered conditions | Source: hydrometeorological_literature; Priestley & Taylor (1972) Mon. Weather Rev. 100:81-92; De Bruin (1983) J. Climate Appl. Meteor. 22:572-578 |
| `hydrosphere.evapotranspiration.soil_heat_fraction` | `dimensionless` | `0.1; bounds=(0.0, 0.3)` | `fixed` | Fraction of net radiation allocated to soil heat flux for daily Priestley-Taylor estimates | Source: evapotranspiration_guidelines; Allen et al. (1998) FAO Irrigation Paper 56; Brutsaert (1982) Evaporation into the Atmosphere |
| `hydrosphere.evapotranspiration.soil_heat_flux_source` | `dimensionless` | `'fraction'; bounds=None` | `fixed` | Explicit ground-heat source for Priestley-Taylor and Penman-Monteith; fraction uses the process-specific documented radiation fraction, state requires signed surface soil_heat_flux without fallback | Source: explicit_surface_energy_input_contract; Priestley and Taylor (1972) Monthly Weather Review 100:81-92; Allen et al. (1998) FAO Irrigation and Drainage Paper 56, soil heat flux |
| `hydrosphere.evapotranspiration.sub_freezing_phase` | `dimensionless` | `'liquid'; bounds=None` | `fixed` | Phase of the saturation vapor pressure and latent heat that Priestley-Taylor uses below 273.15 K when the configured temperature bounds admit such temperatures; liquid keeps the single Tetens form, ice uses the Murray (1967) ice coefficients and the latent heat of sublimation cell by cell | Source: sub_freezing_psychrometric_option; Murray (1967) Journal of Applied Meteorology 6:203-204, doi:10.1175/1520-0450(1967)006<0203:OTCOSV>2.0.CO;2; Priestley and Taylor (1972) Monthly Weather Review 100:81-92 |
| `hydrosphere.evapotranspiration.enable_dynamic_alpha` | `dimensionless` | `False; bounds=(0.0, 1.0)` | `fixed` | Enable soil-moisture and VPD-dependent adjustments to the Priestley-Taylor alpha coefficient | Source: evapotranspiration_stress_response; Katerji & Rana (2011) Agric. For. Meteorol. 151:428-440 |
| `hydrosphere.evapotranspiration.alpha_min` | `dimensionless` | `0.6; bounds=(0.4, 1.0)` | `learnable` | Lower bound for Priestley-Taylor alpha under strong atmospheric or soil moisture stress | Source: evapotranspiration_stress_response; McNaughton & Jarvis (1983) Adv. Ecol. Res. 13:1-67 |
| `hydrosphere.evapotranspiration.alpha_max` | `dimensionless` | `1.5; bounds=(1.1, 1.8)` | `learnable` | Upper bound for Priestley-Taylor alpha reflecting advective enhancement of evaporation | Source: hydrometeorological_literature; Priestley & Taylor (1972) Mon. Weather Rev. 100:81-92; Brutsaert (1982) Evaporation into the Atmosphere |
| `hydrosphere.evapotranspiration.stress_threshold` | `dimensionless` | `0.5; bounds=(0.1, 0.9)` | `learnable` | Relative soil moisture threshold at which Priestley-Taylor alpha begins to decline | Source: soil_moisture_response_analysis; Katerji & Rana (2011) Agric. For. Meteorol. 151:428-440 |
| `hydrosphere.evapotranspiration.hodge_reference_atmospheric_pressure` | `Pa` | `101325.0; bounds=(80000.0, 110000.0)` | `fixed` | Reference atmospheric pressure p_ref (sea-level standard) at which Priestley-Taylor assembles its static Hodge contract (psychrometric constant gamma(p_ref)); a structural reference state, never a forward input and never substituted for atmospheric_pressure | Source: international_standard; International Standard Atmosphere (ISO 2533); Allen, R.G. et al. (1998). FAO Irrigation and Drainage Paper 56, Eq. 8 |
| `hydrosphere.numerical.temperature_bounds` | `K` | `[273.15, 323.15]; bounds=([183.15, 273.15], [323.15, 373.15])` | `fixed` | Admitted near-surface air temperature for hydrosphere calculations [min, max]. The default keeps 273.15 K as minimum because the default sub-freezing phase of the evapotranspiration laws is liquid. A run with sub_freezing_phase ice may set the minimum down to 183.15 K (range widened on 2026-11-05 from 223.15 K for global daily runs, terrestrial record low -89.2 degC): the Tetens form over ice deviates from Goff-Gratch by -1.6 percent at 233.15 K, -4.7 percent at 213.15 K and -7.2 percent at 203.15 K, where the saturation vapour pressure is below 0.3 Pa. | Source: meteorological_standards; WMO Guide to Meteorological Instruments; WMO Archive of Weather and Climate Extremes |
| `hydrosphere.numerical.net_radiation_validation_offset` | `W m-2` | `1000.0; bounds=(500.0, 2000.0)` | `fixed` | Offset added to net radiation for validation to allow negative values | Source: energy_balance_modeling; Energy Balance Modeling Literature |
| `hydrosphere.numerical.evapotranspiration_flux_clamp_min` | `kg m-2 s-1` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Minimum evapotranspiration flux (non-negative constraint for physical validity) | Source: physical_constraint; Physical constraint - evapotranspiration cannot be negative |
| `hydrosphere.numerical.minimum_drainage_flux` | `kg m-2 s-1` | `1e-08; bounds=(1e-12, 1e-06)` | `fixed` | Minimum threshold for drainage flux calculations | Source: hydrological_modeling; Soil Water Dynamics Literature |
| `validation.numerical.clamp_min_zero` | `dimensionless` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Minimum clamp value for zero-bounded quantities | Source: numerical_methods; Numerical methods best practices |
| `validation.numerical.clamp_max_one` | `dimensionless` | `1.0; bounds=(1.0, 1.0)` | `fixed` | Maximum clamp value for unit-bounded quantities | Source: numerical_methods; Numerical methods best practices |
| `hydrosphere.evapotranspiration.hodge_reference_air_temperature` | `K` | `293.15; bounds=(233.15, 323.15)` | `fixed` | Reference air temperature T_ref (20 degC) at which Priestley-Taylor assembles its static Hodge contract (slope Delta(T_ref) of the saturation vapour pressure curve); a structural reference state, never a forward input and never substituted for air_temperature | Source: standard_conditions; Standard meteorological reference temperature; Allen, R.G. et al. (1998). FAO Irrigation and Drainage Paper 56, Eq. 13 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | outputs use wrap_update(UpdateMode.REPLACE) through existing StateManager path |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_priestley_taylor.md`.

### References

- Priestley, C.H.B. and Taylor, R.J. (1972). Monthly Weather Review, 100(2), 81-92 (FunctorMetadata)
- Penman, H.L. (1948). Proceedings of the Royal Society of London, 193, 120-145 (FunctorMetadata)
- de Bruin, H.A.R. (1983). Journal of Climate and Applied Meteorology, 22(4), 572-578 (FunctorMetadata)
- Murray, F.W. (1967). Journal of Applied Meteorology, 6(1), 203-204 (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Priestley, C.H.B. and Taylor, R.J. (1972). On the assessment of surface heat flux and evaporation using large-scale parameters. Monthly Weather Review, 100, 81-92. (certificate)
- Penman, H.L. (1948). Natural evaporation from open water, bare soil and grass. Proceedings of the Royal Society A, 193, 120-145. (certificate)
- de Bruin, H.A.R. (1983). A model for the Priestley-Taylor parameter alpha. Journal of Climate and Applied Meteorology, 22, 572-578. (certificate)
- McNaughton, K.G. and Jarvis, P.G. (1983). Predicting effects of vegetation changes on transpiration and evaporation. Advances in Ecological Research, 13, 1-47. (certificate)
- Murray, F.W. (1967). On the computation of saturation vapor pressure. Journal of Applied Meteorology, 6, 203-204. (certificate)

(class-relic-scientific-hydrosphere-groundwater-dynamics-groundwaterdynamics)=
## groundwater_dynamics

Groundwater Storage Dynamics

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.groundwater.dynamics.GroundwaterDynamics` |
| Source | `relic\scientific\hydrosphere\groundwater\dynamics.py:68` |
| Runtime name | `groundwater_dynamics` |
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
groundwater_dynamics(name: str = 'groundwater_dynamics', config: Dict | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'groundwater_dynamics'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `groundwater_storage` | state | `mm` | groundwater storage S_gw [mm] |
| `water_table_depth` | state | `m` | depth to water table z_wt [m] |
| `drainage_flux` | flux | `kg m-2 s-1` | recharge input from vadose-zone drainage [kg m-2 s-1] |
| `capillary_rise_flux` | flux | `kg m-2 s-1` | groundwater-to-soil capillary export [kg m-2 s-1] |
| `lateral_inflow` | flux | `kg m-2 s-1` | lateral groundwater input [kg m-2 s-1] |
| `groundwater_pumping` | flux | `kg m-2 s-1` | groundwater abstraction export [kg m-2 s-1] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `groundwater_storage` | state | `mm` | decided at run time by the returned write mode | groundwater storage S_gw [mm] |
| `water_table_depth` | state | `m` | decided at run time by the returned write mode | depth to water table z_wt [m] |
| `baseflow_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | baseflow matter output [kg m-2 s-1] |
| `diagnostics.groundwater` | output port | `not declared` | diagnostics | read-only groundwater balance and projection diagnostics |

### Governing equations and canonical operator

**Literature or process form**

Water-table storage follows integration of a depth-dependent porosity profile and baseflow follows storage-depth recession concepts used in distributed hydrologic models.

**Canonical RELIC structural form**

$$
\dot{S} = F_{\mathrm{drain}} + F_{\mathrm{lat}} - F_{\mathrm{pump}} - F_{\mathrm{cap}} - F_{\mathrm{base}}; S ( z ) = \int \mathrm{n0} \exp ( - z / p ) \mathrm{d}z; \star_{\mathrm{Sy}} = \operatorname{diag} ( \rho_{w} \cdot S_{y} ) \geq 0 .
$$

| Entry | Meaning |
| --- | --- |
| `G_gw` | groundwater recharge/baseflow port incidence |
| `G_h` | negative inverse storage Hodge maps signed boundary mass flux to positive-downward depth rate |
| `star_Sy` | build_hodge_contract and build_hodge_terms HodgeStarSpec |
| `porosity_profile` | _compute_porosity_at_depth and _integrate_storage exponential profile |
| `water_table_solver` | _solve_water_table_change analytical exponential integral inversion |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | groundwater storage and head cochain $x_{\mathrm{gw}}$ |
| `effort_variable` | state effort $e_{z} = \star_{\mathrm{Sy}} z_{h}$ and conjugate port effort $y_{h} = G_{h}^{\mathsf{T}} e_{z} = - z_{h}$ |
| `state_advancer` | $\dot{S} = F_{\mathrm{drain}} + F_{\mathrm{lat}} - F_{\mathrm{pump}} - F_{\mathrm{cap}} - F_{\mathrm{base}}$ |
| `runtime_usage` | forward evaluates baseflow from the accepted start-of-step state, applies all five signed boundary terms to one storage and water-table update, maps mass flux to head rate through $G_{h}$, and emits namespaced conservation and non-strict IPHS audit diagnostics. |

### Composition channels and executable PortSpec API

Certificate composition channels: `groundwater_recharge_flux`, `groundwater_lateral_inflow_flux`, `groundwater_pumping_flux`, `groundwater_capillary_rise_flux`, `groundwater_baseflow_flux`

Executable `ports()` owner: `GroundwaterDynamics`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `groundwater_recharge_flux` | `groundwater_head [m]` | `drainage_flux [kg m-2 s-1]` | `kg m-1 s-1` | into_system; matter |
| `groundwater_lateral_inflow_flux` | `groundwater_head [m]` | `lateral_inflow [kg m-2 s-1]` | `kg m-1 s-1` | into_system; matter |
| `groundwater_pumping_flux` | `groundwater_head [m]` | `groundwater_pumping [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |
| `groundwater_capillary_rise_flux` | `groundwater_head [m]` | `capillary_rise_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |
| `groundwater_baseflow_flux` | `groundwater_head [m]` | `baseflow_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.groundwater.surface_porosity` | `dimensionless` | `0.45; bounds=(0.3, 0.7)` | `fixed` | Porosity at soil surface for exponential decay profile n(z) = n0 * exp(-z/p) | Source: RHESSys compute_delta_water.c + literature; Beven, K.J. & Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69; Wigmosta, M.S. et al. (1994). Water Resources Research, 30(6), 1665-1679 |
| `hydrosphere.groundwater.porosity_decay_length` | `m` | `2.0; bounds=(0.5, 10.0)` | `fixed` | Exponential decay length scale p in porosity profile n(z) = n0 * exp(-z/p) | Source: RHESSys implementation + calibration; Beven, K.J. & Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69 |
| `hydrosphere.groundwater.max_water_table_depth` | `m` | `10.0; bounds=(2.0, 50.0)` | `fixed` | Maximum allowable water table depth below surface (positive downward) | Source: RHESSys configuration guidelines; Wigmosta, M.S. et al. (1994). Water Resources Research, 30(6), 1665-1679 |
| `hydrosphere.groundwater.bedrock_depth` | `m` | `5.0; bounds=(1.0, 20.0)` | `fixed` | Depth to impermeable bedrock below surface (positive downward) | Source: Site-specific geological data; Beven, K.J. & Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69 |
| `hydrosphere.groundwater.specific_yield` | `dimensionless` | `0.2; bounds=(0.05, 0.3)` | `fixed` | Specific yield for unconfined aquifer water table fluctuations | Source: USGS aquifer property database; Johnson, A.I. (1967). USGS Water-Supply Paper 1662-D |
| `hydrosphere.groundwater.minimum_water_table_depth` | `m` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Minimum allowable water table depth for numerical stability | Source: numerical_implementation; Numerical stability constraint for groundwater depth clamps |
| `hydrosphere.groundwater.minimum_depth_factor` | `dimensionless` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Minimum depth factor for baseflow scaling | Source: hydrological_modeling; Brutsaert and Nieber (1977); Tallaksen (1995) |
| `hydrosphere.groundwater.drainage_time_constant` | `s` | `86400.0; bounds=(3600.0, 2592000.0)` | `fixed` | Time constant tau for exponential baseflow recession Q = Q0 * exp(-t/tau) | Source: Recession analysis from hydrograph data; Beven, K.J. (2012). Rainfall-Runoff Modelling: The Primer. Wiley-Blackwell |
| `hydrosphere.groundwater.enable_lateral_flow` | `dimensionless` | `True; bounds=None` | `fixed` | Enable lateral groundwater inflow from adjacent grid cells or upslope areas | Source: Model configuration option; Boussinesq (1904) J. Math. Pures Appl. 10, 5-78 |
| `hydrosphere.groundwater.topographic_index` | `dimensionless` | `5.0; bounds=(1.0, 15.0)` | `fixed` | TOPMODEL topographic index ln(a/tan(beta)) for spatial baseflow distribution | Source: DEM-derived topographic analysis; Beven, K.J. & Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69 |
| `hydrosphere.numerical.minimum_timestep_tolerance` | `s` | `1e-12; bounds=(1e-15, 1e-06)` | `fixed` | Minimum timestep used to avoid division by zero in groundwater updates | Source: numerical_implementation; Numerical stability constraint for time integration |
| `hydrosphere.numerical.epsilon_small` | `dimensionless` | `1e-09; bounds=(1e-12, 1e-06)` | `fixed` | Small numerical epsilon for stability in groundwater and snow calculations | Source: numerical_implementation; Numerical stability constraint for division and thresholding |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | formal outputs use explicit UpdateItem(UpdateMode.REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} )$ through the existing StateManager path |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_groundwater_dynamics.md`.

### References

- RHESSys compute_delta_water.c (FunctorMetadata)
- Beven, K.J. & Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69 (FunctorMetadata)
- Wigmosta, M.S. et al. (1994). Water Resources Research, 30(6), 1665-1679 (FunctorMetadata)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- RHESSys compute_delta_water.c implementation. (certificate)
- Beven, K.J. and Kirkby, M.J. (1979). A physically based, variable contributing area model. Hydrological Sciences Bulletin, 24, 43-69. (certificate)
- Wigmosta, M.S. et al. (1994). A distributed hydrology-vegetation model for complex terrain. Water Resources Research, 30, 1665-1679. (certificate)
- Brutsaert, W. and Nieber, J.L. (1977). Regionalized drought flow hydrographs. Water Resources Research, 13, 637-643. (certificate)

(class-relic-scientific-hydrosphere-runoff-subsurface-subsurfacerunoff)=
## subsurface_runoff

Subsurface Runoff Generation Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.runoff.subsurface.SubsurfaceRunoff` |
| Source | `relic\scientific\hydrosphere\runoff\subsurface.py:98` |
| Runtime name | `subsurface_runoff` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `patch` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
subsurface_runoff(name: str = 'subsurface_runoff', config: Dict | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'subsurface_runoff'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | volumetric water content theta controlling interflow and TOPMODEL moisture scaling [m3 m-3] |
| `drainage_flux` | flux | `kg m-2 s-1` | required incoming drainage boundary flux from soil-water modules [kg m-2 s-1] |
| `groundwater_storage` | state | `mm` | groundwater storage S [mm] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `subsurface_runoff_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | lateral subsurface matter port flow q_h [kg m-2 s-1] |
| `groundwater_storage` | state | `mm` | decided at run time by the returned write mode | groundwater storage S [mm] |
| `diagnostics.subsurface_runoff.baseflow_flux` | output port | `kg m-2 s-1` | diagnostics | read-only baseflow component [kg m-2 s-1] |
| `diagnostics.subsurface_runoff.interflow_flux` | output port | `kg m-2 s-1` | diagnostics | read-only interflow component [kg m-2 s-1] |
| `diagnostics.subsurface_runoff.storage_fraction` | output port | `dimensionless` | diagnostics | read-only normalized storage [dimensionless] |
| `diagnostics.subsurface_runoff.water_table_depth` | output port | `m` | diagnostics | clipped water-table observation [m] |
| `diagnostics.subsurface_runoff.recession_time_constant_days` | output port | `d` | diagnostics | recession-time observation [d] |
| `diagnostics.subsurface_runoff.drainage_rate_mm_s` | output port | `mm s-1` | diagnostics | converted inlet depth rate [mm s-1] |
| `diagnostics.subsurface_runoff.total_runoff_rate_mm_s` | output port | `mm s-1` | diagnostics | actual outlet depth rate [mm s-1] |
| `diagnostics.subsurface_runoff.storage_capacity_overflow` | output port | `mm` | diagnostics | observed portion of subsurface_runoff_flux caused by the capacity projection [mm per time step] |
| `diagnostics.subsurface_runoff.outflow_limiter_residual` | output port | `mm` | diagnostics | rejected candidate outflow observation [mm per time step] |
| `diagnostics.subsurface_runoff.water_table_depth_clipping_residual` | output port | `m` | diagnostics | diagnostic depth clipping residual [m] |
| `diagnostics.subsurface_runoff.water_balance_residual` | output port | `mm` | diagnostics | open control-volume balance residual [mm per time step] |

### Governing equations and canonical operator

**Literature or process form**

Lateral subsurface discharge follows a storage-discharge recession law with an optional catchment-mean TOPMODEL closure $Q_{b} = 1000 \cdot \mathrm{T}_{0} \cdot \exp ( - \lambda ) \cdot \exp ( - S_{\mathrm{def}} / m ) \cdot f_{\theta} ( \mathrm{Beven} \mathrm{and} \mathrm{Kirkby} , 1979 )$ and a moisture-gated interflow drainage $Q_{i} = k_{i} \cdot c_{i} \cdot \max ( 0 , \theta - \theta_{t} ) ^ n \cdot S \mathrm{of} \mathrm{the} \mathrm{same} \mathrm{store} .$

**Canonical RELIC structural form**

$$
\dot{S} = I_{\mathrm{drain}} - D_{h} q_{\mathrm{total}}; q_{\mathrm{total}} = \star_{T} g_{h} + q_{\mathrm{interflow}} + q_{\mathrm{capacity}}; \star_{T} \geq 0 .
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | storage-to-port incidence represented by the existing StateManager update path |
| `star_T` | build_hodge_contract and build_hodge_terms HodgeStarSpec |
| `baseflow_closure` | _compute_exponential_baseflow, _compute_topmodel_baseflow, or _compute_linear_reservoir_baseflow |
| `interflow_closure` | _compute_interflow threshold function |
| `unit_conversion` | UnitConverter mass flux and depth-rate conversions |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | subsurface storage cochain $S_{h}$ |
| `effort_variable` | hydraulic gradient $g_{h}$ with transmissivity mobility star_T |
| `state_advancer` | $\mathrm{S}_{1} = \mathrm{S}_{0} + R \cdot \mathrm{d}t - Q_{\mathrm{total}} \cdot \mathrm{d}t$; $Q_{\mathrm{total}}$ includes the capacity-overflow component |
| `runtime_usage` | forward limits regular integrated outlet water by available storage, routes capacity overflow into the canonical outlet, and emits every clipping or limiter residual explicitly. |

### Composition channels and executable PortSpec API

Certificate composition channels: `subsurface_runoff_boundary_flux`

Executable `ports()` owner: `SubsurfaceRunoff`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `subsurface_runoff_boundary_flux` | `subsurface_hydraulic_head [m]` | `subsurface_runoff_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.runoff.baseflow_recession_constant` | `s-1` | `0.01; bounds=(0.001, 0.1)` | `learnable` | Baseflow recession constant for groundwater discharge | Source: hydrograph_analysis; Brutsaert & Nieber (1977) Water Resour. Res. 13:637-644; Tallaksen (1995) J. Hydrol. 165:349-370 |
| `hydrosphere.runoff.topographic_index_mean` | `dimensionless` | `8.0; bounds=(2.0, 20.0)` | `fixed` | Mean topographic index ln(a/tan(beta)) for the watershed, with the upslope contributing area per unit contour length a in metres, so exp(-lambda) carries m-1 and T0*exp(-lambda) is the TOPMODEL baseflow scale Q0 [m s-1] | Source: dem_analysis; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24:43-69; Quinn et al. (1991) Hydrol. Process. 5:3-19 |
| `hydrosphere.runoff.transmissivity` | `m2 s-1` | `0.1; bounds=(0.001, 10.0)` | `learnable` | Soil transmissivity for lateral subsurface flow | Source: calibration_studies; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24:43-69; O'Loughlin (1986) Water Resour. Res. 22:794-804 |
| `hydrosphere.runoff.threshold_moisture` | `m3 m-3` | `0.3; bounds=(0.1, 0.8)` | `learnable` | Threshold soil moisture for interflow generation (field capacity) | Source: soil_hydraulic_properties; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Rawls et al. (1982) Trans. ASAE 25:1316-1320 |
| `hydrosphere.runoff.return_flow_fraction` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `learnable` | Fraction of subsurface flow that returns as baseflow | Source: baseflow_separation; Nathan & McMahon (1990) Water Resour. Res. 26:2783-2797; Eckhardt (2005) J. Hydrol. 313:123-134 |
| `hydrosphere.runoff.interflow_scaling_factor` | `dimensionless` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Scaling factor converting return flow fraction to daily interflow coefficient | Source: hydrological_literature; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24:43-69; Moore & Clarke (1981) Hydrol. Sci. Bull. 26:373-390 |
| `hydrosphere.runoff.storage_capacity` | `mm` | `100.0; bounds=(10.0, 1000.0)` | `learnable` | Maximum groundwater storage capacity for subsurface runoff model | Source: watershed_analysis; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24:43-69; Sivapalan et al. (1987) Water Resour. Res. 23:1061-1077 |
| `hydrosphere.runoff.baseflow_method` | `dimensionless` | `'exponential'; bounds=None` | `fixed` | Baseflow calculation method (exponential, topmodel, or linear_reservoir) | Source: model_configuration; Brutsaert & Nieber (1977) Water Resour. Res. 13:637-644; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24:43-69 |
| `hydrosphere.runoff.topmodel_max_deficit` | `mm` | `200.0; bounds=(10.0, 1000.0)` | `fixed` | Maximum storage deficit used in simplified TOPMODEL baseflow | Source: topmodel_parameterization; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24:43-69; Sivapalan et al. (1987) Water Resour. Res. 23:1061-1077 |
| `hydrosphere.runoff.topmodel_m_scaling_factor` | `dimensionless` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Scaling factor mapping max storage deficit to TOPMODEL m parameter | Source: topmodel_parameterization; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24:43-69; Sivapalan et al. (1987) Water Resour. Res. 23:1061-1077 |
| `hydrosphere.runoff.interflow_exponent` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `fixed` | Exponent controlling interflow sensitivity to excess soil moisture | Source: hydrological_literature; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24:43-69; Moore & Clarke (1981) Hydrol. Sci. Bull. 26:373-390 |
| `hydrosphere.runoff.interflow_rate_scaling_factor` | `dimensionless` | `100.0; bounds=(1.0, 1000.0)` | `fixed` | Dimensionless multiplier of the moisture-gated interflow recession coefficient k_i*c_i*max(0,theta-theta_t)^n applied to the groundwater store | Source: numerical_implementation; RELIC threshold-interflow closure, docs/source/theory/hydrosphere_subsurface_runoff.md |
| `hydrosphere.runoff.topmodel_moisture_reference` | `m3 m-3` | `0.4; bounds=(0.05, 1.0)` | `fixed` | Reference soil moisture for TOPMODEL moisture scaling | Source: hydrological_parameterization; Beven & Kirkby (1979) Hydrol. Sci. Bull. 24:43-69 |
| `hydrosphere.runoff.topmodel_moisture_factor_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Minimum moisture scaling factor for TOPMODEL baseflow | Source: numerical_implementation; Numerical stability constraint for TOPMODEL moisture scaling |
| `hydrosphere.runoff.topmodel_moisture_factor_max` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Maximum moisture scaling factor for TOPMODEL baseflow | Source: numerical_implementation; Numerical stability constraint for TOPMODEL moisture scaling |
| `hydrosphere.runoff.water_table_depth_scale` | `m` | `2.0; bounds=(0.5, 10.0)` | `fixed` | Diagnostic scaling factor for groundwater table depth estimation | Source: diagnostic_approximation; Simplified diagnostic scaling for groundwater depth |
| `hydrosphere.runoff.water_table_depth_max` | `m` | `5.0; bounds=(1.0, 20.0)` | `fixed` | Maximum diagnostic groundwater table depth | Source: diagnostic_approximation; Simplified diagnostic scaling for groundwater depth |
| `hydrosphere.numerical.minimum_storage_outflow` | `mm` | `1e-08; bounds=(1e-12, 0.0001)` | `fixed` | Minimum storage outflow used to avoid division by zero in storage scaling | Source: numerical_implementation; Numerical stability constraint for storage outflow scaling |
| `hydrosphere.numerical.recession_coefficient_minimum` | `s-1` | `1e-06; bounds=(1e-12, 0.001)` | `fixed` | Minimum recession coefficient used to avoid division by zero | Source: numerical_implementation; Numerical stability constraint for recession time calculation |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | every output is an explicit UpdateItem(REPLACE) with unit and source_functor |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_subsurface_runoff.md`.

### References

- Beven, K.J. and Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69 (FunctorMetadata)
- Brutsaert, W. and Nieber, J.L. (1977). Water Resources Research, 13(3), 637-643 (FunctorMetadata)
- Tallaksen, L.M. (1995). Journal of Hydrology, 165(1-4), 349-370 (FunctorMetadata)
- Nathan, R.J. and McMahon, T.A. (1990). Water Resources Research, 26(7), 1465-1473 (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Beven, K.J. and Kirkby, M.J. (1979). A physically based, variable contributing area model of basin hydrology. Hydrological Sciences Bulletin, 24, 43-69. (certificate)
- Brutsaert, W. and Nieber, J.L. (1977). Regionalized drought flow hydrographs from a mature glaciated plateau. Water Resources Research, 13, 637-643. (certificate)
- Tallaksen, L.M. (1995). A review of baseflow recession analysis. Journal of Hydrology, 165, 349-370. (certificate)
- Nathan, R.J. and McMahon, T.A. (1990). Evaluation of automated techniques for base flow and recession analyses. Water Resources Research, 26, 1465-1473. (certificate)
- Darcy, H. (1856). Les Fontaines publiques de la ville de Dijon. Dalmont, Paris. (certificate)

(class-relic-scientific-hydrosphere-runoff-surface-ponding-hortonianpondingeventfunctor)=
## HortonianPondingEventFunctor

Infiltration-excess detector that requests the ponding topology transition.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.runoff.surface_ponding.HortonianPondingEventFunctor` |
| Source | `relic\scientific\hydrosphere\runoff\surface_ponding.py:107` |
| Runtime name | `ponding_event` |
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
HortonianPondingEventFunctor(name: str = 'ponding_event', *, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'ponding_event'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Mapping[str, Any]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `stochastic_rainfall_depth` | state | `mm` | cumulative rainfall depth H [mm] read from the rainfall provider |
| `surface_ponding_active` | state | `1` | connectivity flag a [1] (0 disconnected, 1 connected), changed only by the committed remap |
| `surface_ponded_depth` | state | `mm` | ponded depth h [mm] read for the cessation guard |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `surface_ponding_excess` | state | `mm` | decided at run time by the returned write mode | prognostic infiltration excess E of the step [mm], replaced every step |
| `diagnostics.ponding_event.excess_depth` | diagnostic | `mm` | diagnostics | E of the emitted step [mm] |
| `diagnostics.ponding_event.rainfall_increment` | diagnostic | `mm` | diagnostics | dH of the emitted step [mm] |
| `diagnostics.ponding_event.infiltrated_excess` | diagnostic | `mm` | diagnostics | E infiltrated through depression storage while disconnected [mm] |

### Governing equations and canonical operator

**Literature or process form**

Hortonian infiltration excess (Horton 1933) with a depression-storage onset threshold (Beven 2012) and an event-triggered hybrid jump (Goebel, Sanfelice and Teel 2012)

**Canonical RELIC structural form**

x^+ = P_h x^- + R_h eta_h on (h, E, a); C x^+ = C x^- (onset) or C x^+ = C x^- + Pi (cessation); no IPHS operator is claimed

| Entry | Meaning |
| --- | --- |
| `excess` | $E = \max ( 0 , \mathrm{d}H - I_{\mathrm{max}} \mathrm{d}t / 3600 )$ with $dH = H_n - H_{n-1} kept in module extra state (replay-critical)$ |
| `onset_request` | ExternalIntent(topology, ponding_onset) when $a = 0$ and $\max E \geq E_{\mathrm{on}}$ |
| `cessation_request` | ExternalIntent(topology, ponding_cessation) when $a = 1$, $E = 0$ and $\max h \leq h_{\mathrm{off}}$ |
| `commit` | relic.core.simulator.registration.apply_topology_transition commits retire/connect, the affine remap, the ledger audit, and one provider epoch after the accepted step |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | infiltration excess on the declared cell basis; no storage functional is claimed |
| `effort_variable` | none; the event is a threshold on the excess and the ponded depth |
| `state_advancer` | $E_{n} = \max ( 0 , \mathrm{d}H_{n} - I_{\mathrm{max}} \mathrm{d}t / 3600 ) \mathrm{emitted}$ as REPLACE; $\mathrm{the} \mathrm{topology} \mathrm{remap} ( h , E , a ) \rightarrow ( h + E , 0 , 1 )$ or (0, E, 0) is committed by the lifecycle owner |
| `runtime_usage` | HortonianPondingEventFunctor.forward validates the inputs, forms the increment from module extra state, emits the excess and three REPLACE diagnostics, and adds one ExternalIntent when a guard holds; the intent is validated before the step commits and executed after acceptance by retry_external_deliveries. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.runoff.ponding_infiltration_capacity` | `mm h-1` | `10.0; bounds=(1.0, 200.0)` | `learnable` | Steady infiltration capacity of the surface; rainfall depth arriving faster than this rate becomes Hortonian excess (loam order of magnitude) | Source: literature_typical_values; Horton, R.E. (1933). Trans. Am. Geophys. Union 14, 446-460; Rawls, W.J., Brakensiek, D.L. and Saxton, K.E. (1982). Trans. ASAE 25, 1316-1320 |
| `hydrosphere.runoff.ponding_onset_depth` | `mm` | `2.0; bounds=(0.1, 50.0)` | `learnable` | Depression-storage threshold; infiltration excess in one step at or above this depth connects the surface ponding store to the graph | Source: literature_typical_values; Beven, K.J. (2012). Rainfall-Runoff Modelling, The Primer, 2nd ed., Wiley, ch. 6; Goebel, R., Sanfelice, R.G. and Teel, A.R. (2012). Hybrid Dynamical Systems, Princeton |
| `hydrosphere.runoff.ponding_cessation_depth` | `mm` | `0.1; bounds=(0.001, 10.0)` | `learnable` | Ponded depth at or below which, with no new excess, the surface ponding store is retired and its residual infiltrates | Source: literature_typical_values; Beven, K.J. (2012). Rainfall-Runoff Modelling, The Primer, 2nd ed., Wiley, ch. 6 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem REPLACE for the excess and REPLACE observations for diagnostics; topology requests are ExternalIntent outputs |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_surface_ponding.md`.

### References

- Horton, R.E. (1933). Trans. Am. Geophys. Union 14, 446-460 (FunctorMetadata, certificate)
- Beven, K.J. (2012). Rainfall-Runoff Modelling: The Primer, 2nd ed., Wiley (FunctorMetadata)
- Goebel, R., Sanfelice, R.G. & Teel, A.R. (2012). Hybrid Dynamical Systems, Princeton (FunctorMetadata, certificate)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- Beven, K.J. (2012). Rainfall-Runoff Modelling, The Primer, 2nd ed., Wiley (certificate)

(class-relic-scientific-hydrosphere-runoff-surface-ponding-surfacepondingfunctor)=
## SurfacePondingFunctor

Ponded surface water draining as overland flow (linear reservoir).

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.runoff.surface_ponding.SurfacePondingFunctor` |
| Source | `relic\scientific\hydrosphere\runoff\surface_ponding.py:392` |
| Runtime name | `surface_ponding` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
SurfacePondingFunctor(name: str = 'surface_ponding', *, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'surface_ponding'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Mapping[str, Any]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

The runtime metadata below is introspected from the documented configuration `event_provider='ponding_event'`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `surface_ponded_depth` | state | `mm` | prognostic ponded depth h [mm]; input state and accumulated storage increment |
| `surface_ponding_excess` | state | `mm` | infiltration excess E of the step [mm] written by the ponding event detector |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `surface_ponded_depth` | state | `mm` | decided at run time by the returned write mode | prognostic ponded depth h [mm]; input state and accumulated storage increment |
| `diagnostics.surface_ponding.runoff_depth` | diagnostic | `mm` | diagnostics | overland-flow depth q drained in the step [mm] |
| `diagnostics.surface_ponding.water_balance_residual` | diagnostic | `mm` | diagnostics | Delta h - E + q of the emitted step [mm] |

### Governing equations and canonical operator

**Literature or process form**

Linear reservoir overland-flow recession (Dooge 1973; Beven 2012)

**Canonical RELIC structural form**

$$
h_{n+1} = (h_n + E_n) exp(-dt/tau), q_n = h_n + E_n - h_{n+1}; no IPHS operator is claimed
$$

| Entry | Meaning |
| --- | --- |
| `recession` | h' = (h + E) exp(-dt/tau), exact solution of $\frac{\mathrm{d}h}{\mathrm{d}t} = - h / \tau$ over the step after the excess enters |
| `runoff` | $q = h + E - h '$ is the drained overland-flow depth of the step (mass transfer to the channel routing port) |
| `budget` | $\mathrm{residual} = ( h ' - h ) - E + q \mathrm{vanishes}$ to rounding and is reported every step |
| `connectivity` | the store is registered by ponding_onset and retired by ponding_cessation through the registration owner; it never runs while disconnected |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | ponded depth on the declared cell basis; no storage functional is claimed |
| `effort_variable` | none; the recession is linear in the stored depth |
| `state_advancer` | $\Delta h = ( h + E ) \exp ( - \mathrm{d}t / \tau ) - h \mathrm{emitted}$ as one ACCUMULATE increment per step |
| `runtime_usage` | SurfacePondingFunctor.forward validates the inputs and the step, applies the exact recession, and emits the increment plus two REPLACE diagnostics; check_conservation_laws reports the budget residual to the local conservation hook. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.runoff.ponding_drainage_time` | `s` | `1800.0; bounds=(60.0, 86400.0)` | `learnable` | Linear-reservoir residence time of ponded water draining as overland flow (exact exponential recession) | Source: literature_typical_values; Dooge, J.C.I. (1973). Linear Theory of Hydrologic Systems, USDA Tech. Bull. 1468; Beven, K.J. (2012). Rainfall-Runoff Modelling, The Primer, 2nd ed., Wiley |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem ACCUMULATE with per_timestep true for the ponded depth and REPLACE observations for diagnostics |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_surface_ponding.md`.

### References

- Dooge, J.C.I. (1973). Linear Theory of Hydrologic Systems, USDA Tech. Bull. 1468 (FunctorMetadata, certificate)
- Beven, K.J. (2012). Rainfall-Runoff Modelling: The Primer, 2nd ed., Wiley (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Beven, K.J. (2012). Rainfall-Runoff Modelling, The Primer, 2nd ed., Wiley (certificate)
- Goebel, R., Sanfelice, R.G. & Teel, A.R. (2012). Hybrid Dynamical Systems, Princeton (certificate)

(class-relic-scientific-hydrosphere-runoff-surface-surfacerunoff)=
## surface_runoff

Surface Runoff Generation Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.runoff.surface.SurfaceRunoff` |
| Source | `relic\scientific\hydrosphere\runoff\surface.py:88` |
| Runtime name | `surface_runoff` |
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
surface_runoff(name: str = 'surface_runoff', config: Dict | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'surface_runoff'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `precipitation_flux` | forcing | `kg m-2 s-1` | boundary precipitation mass flux P, read only with surface_water_input_source precipitation_flux [kg m-2 s-1] |
| `rain_flux` | flux | `kg m-2 s-1` | rain reaching the ground from the snowpack partition, read only with surface_water_input_source rain_flux_plus_snow_melt_flux [kg m-2 s-1] |
| `snow_melt_flux` | flux | `kg m-2 s-1` | snowmelt release of the snowpack, read with rain_flux only under surface_water_input_source rain_flux_plus_snow_melt_flux [kg m-2 s-1] |
| `soil_moisture` | state | `m3 m-3` | scalar surface moisture or vertical antecedent profile whose uppermost layer is theta[0] [m3 m-3] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `surface_runoff_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | overland matter port flow u_surf [kg m-2 s-1] |
| `infiltration_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | precipitation minus surface runoff [kg m-2 s-1] |
| `diagnostics.surface_runoff.runoff_coefficient` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.surface_runoff.precipitation_rate_mm_h` | output port | `mm h-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.surface_runoff.runoff_rate_mm_h` | output port | `mm h-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.surface_runoff.scs_retention` | output port | `mm` | diagnostics | effective moisture-adjusted S used by the SCS equation [mm] |
| `diagnostics.surface_runoff.initial_abstraction` | output port | `mm` | diagnostics | effective Ia=lambda_a*S used by the SCS equation [mm] |
| `diagnostics.surface_runoff.runoff_nonnegative_residual` | output port | `kg m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.surface_runoff.precipitation_limit_residual` | output port | `kg m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.surface_runoff.water_balance_residual` | output port | `kg m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

Surface runoff is generated by infiltration-excess, saturation-excess, or SCS curve-number event retention and converted to overland mass flux.

**Canonical RELIC structural form**

$$
\dot{m} = G_{\mathrm{surf}} u_{\mathrm{surf}}; u_{\mathrm{surf}} = \min ( P , \max ( 0 , \rho_{w} \cdot Q ) ); \star_{\mathrm{surf}} = \operatorname{diag} ( \rho_{w} \cdot D_{\mathrm{surf}} ) \geq 0 .
$$

| Entry | Meaning |
| --- | --- |
| `G_surf` | surface_runoff_boundary_flux port incidence |
| `G_infiltration` | soil_infiltration_boundary carries canonical infiltration_flux to the selected soil-water process |
| `star_surf` | build_hodge_contract maps routing gradient $X_{\mathrm{surf}}$ [m-1] through $\rho_{w} \cdot D_{\mathrm{surf}}$ [kg m-1 s-1] to mass flux [kg m-2 s-1] |
| `runoff_closure` | _compute_scs_runoff, _compute_hortonian_runoff, _compute_saturation_runoff, or _compute_combined_runoff |
| `unit_conversion` | UnitConverter mass flux and depth-rate conversions |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | surface water source cochain $u_{\mathrm{surf}}$ |
| `effort_variable` | routing gradient $X_{\mathrm{surf}}$ [m-1] acted on by positive structural mobility star_surf |
| `state_advancer` | $\dot{m} = G_{\mathrm{surf}} u_{\mathrm{surf}}$; $u_{\mathrm{surf}} = \min ( P , \max ( 0 , \rho_{w} \cdot Q ) )$ |
| `runtime_usage` | forward applies the canonical surface runoff port flow and routes nonnegative projection residuals. |

### Composition channels and executable PortSpec API

Certificate composition channels: `surface_runoff_boundary_flux`, `soil_infiltration_boundary`

Executable `ports()` owner: `SurfaceRunoff`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `surface_runoff_boundary_flux` | `surface_routing_head [m]` | `surface_runoff_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |
| `soil_infiltration_boundary` | `surface_hydraulic_head [m]` | `infiltration_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.transport.surface_water_diffusivity` | `m2 s-1` | `1e-06; bounds=(1e-07, 1e-05)` | `fixed` | Effective diffusivity for surface water transport processes | Source: hydrological_modeling; Surface Water Hydrology Literature |
| `hydrosphere.runoff.curve_number` | `dimensionless` | `70.0; bounds=(30.0, 100.0)` | `learnable` | SCS Curve Number for runoff generation (forest land use, moderate infiltration) | Source: scs_tables; SCS (1972) National Engineering Handbook Section 4; Mockus (1972) SCS Technical Release 55 |
| `hydrosphere.runoff.initial_abstraction_ratio` | `dimensionless` | `0.2; bounds=(0.05, 0.3)` | `learnable` | Initial abstraction ratio (Ia/S) in SCS method | Source: scs_standard; SCS (1972) National Engineering Handbook Section 4; Hawkins et al. (2009) J. Hydrol. Eng. 14:226-235 |
| `hydrosphere.runoff.infiltration_capacity` | `mm h-1` | `10.0; bounds=(1.0, 100.0)` | `learnable` | Maximum infiltration capacity for Hortonian runoff generation | Source: infiltration_experiments; Horton (1933) Eos Trans. AGU 14:446-460; Green & Ampt (1911) J. Agric. Sci. 4:1-24 |
| `hydrosphere.runoff.saturated_area_threshold` | `m3 m-3` | `0.9; bounds=(0.5, 1.0)` | `learnable` | Soil moisture threshold for saturation excess runoff generation | Source: field_saturation_studies; Dunne & Black (1970) Water Resour. Res. 6:1296-1311; Hewlett & Hibbert (1967) Forest Hydrology, 275-290 |
| `hydrosphere.runoff.runoff_method` | `dimensionless` | `'combined'; bounds=None` | `fixed` | Runoff generation method (scs, horton, saturation, or combined) | Source: model_configuration; SCS (1972) National Engineering Handbook Section 4; Horton (1933) Eos Trans. AGU 14:446-460 |
| `hydrosphere.runoff.runoff_combination_method` | `dimensionless` | `'max'; bounds=None` | `fixed` | Combination rule for multi-mechanism runoff when runoff_method is combined (max selects dominant mechanism; sum accumulates mechanisms before clamping to precipitation) | Source: model_configuration; Dunne & Black (1970) Water Resour. Res. 6:1296-1311; Horton (1933) Eos Trans. AGU 14:446-460 |
| `hydrosphere.runoff.surface_water_input_source` | `dimensionless` | `'precipitation_flux'; bounds=None` | `fixed` | Declared surface water input of SurfaceRunoff. precipitation_flux reads the total precipitation (no snowpack Functor); rain_flux_plus_snow_melt_flux reads the rain reaching the ground plus the snowmelt of the snowpack partition and declares the snowpack Functor (SCM owner of both fluxes) as a required dependency. Exactly one representation is read; an instance config key of the same name overrides this value. | Source: model_configuration; E3SM ELM SnowHydrologyMod, column without snow layers (qflx_top_soil = qflx_rain_grnd + qflx_snomelt); Oleson et al. (2013) Technical Description of version 4.5 of the Community Land Model (CLM), NCAR Technical Note NCAR/TN-503+STR |
| `hydrosphere.runoff.antecedent_moisture_factor` | `dimensionless` | `1.0; bounds=(0.5, 2.0)` | `learnable` | Antecedent moisture condition factor for SCS method (AMC I: 0.5-0.8, AMC II: 1.0, AMC III: 1.2-2.0) | Source: scs_amc_tables; SCS (1972) National Engineering Handbook Section 4; Hjelmfelt (1991) J. Irrig. Drain. Eng. 117:202-219 |
| `hydrosphere.runoff.scs_storage_minimum` | `mm` | `0.1; bounds=(0.0, 10.0)` | `fixed` | Minimum SCS storage parameter S to avoid division issues | Source: numerical_implementation; SCS (1972) National Engineering Handbook Section 4 |
| `hydrosphere.runoff.moisture_adjustment_factor` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `learnable` | Moisture adjustment coefficient for SCS retention scaling | Source: hydrological_modeling; SCS (1972) National Engineering Handbook Section 4 |
| `hydrosphere.runoff.infiltration_moisture_reduction_factor` | `dimensionless` | `0.7; bounds=(0.0, 1.0)` | `learnable` | Soil moisture reduction coefficient for infiltration capacity | Source: hydrological_modeling; Horton (1933) infiltration excess runoff |
| `hydrosphere.physical.scs_numerator_constant` | `mm` | `25400.0; bounds=(25400.0, 25400.0)` | `fixed` | SCS Curve Number method numerator constant (1000 inches converted to mm) | Source: usda_nrcs_tr55; USDA-NRCS (1986) Urban Hydrology for Small Watersheds, TR-55; Ponce & Hawkins (1996) J. Hydrol. Eng. 1(1):11-19 |
| `hydrosphere.physical.scs_offset_constant` | `mm` | `254.0; bounds=(254.0, 254.0)` | `fixed` | SCS Curve Number method offset constant (10 inches converted to mm) | Source: usda_nrcs_tr55; USDA-NRCS (1986) Urban Hydrology for Small Watersheds, TR-55; Ponce & Hawkins (1996) J. Hydrol. Eng. 1(1):11-19 |
| `hydrosphere.numerical.minimum_flux_threshold` | `kg m-2 s-1` | `1e-08; bounds=(1e-12, 0.0001)` | `fixed` | Minimum flux threshold used in diagnostic ratio calculations | Source: numerical_implementation; Numerical stability constraint for flux diagnostics |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | formal flux outputs use UpdateItem(UpdateMode.REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} )$ through existing StateManager path |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_surface_runoff.md`.

### References

- Horton, R.E. (1933). Transactions, American Geophysical Union, 14(1), 446-460 (FunctorMetadata)
- Dunne, T. and Black, R.D. (1970). Water Resources Research, 6(5), 1296-1311 (FunctorMetadata)
- SCS (1972). National Engineering Handbook, Section 4: Hydrology (FunctorMetadata)
- USDA NRCS (2021). National Engineering Handbook, Part 650, Chapter 2 (FunctorMetadata)
- Chow, V.T. et al. (1988). Applied Hydrology (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Horton, R.E. (1933). The role of infiltration in the hydrologic cycle. Transactions, American Geophysical Union, 14, 446-460. (certificate)
- Dunne, T. and Black, R.D. (1970). Partial area contributions to storm runoff. Water Resources Research, 6, 1296-1311. (certificate)
- SCS (1972). National Engineering Handbook, Section 4: Hydrology. (certificate)
- USDA NRCS (2021). National Engineering Handbook, Part 650, Chapter 2: Estimating Runoff Volume and Peak Discharge. (certificate)
- Chow, V.T., Maidment, D.R., and Mays, L.W. (1988). Applied Hydrology. McGraw-Hill. (certificate)

(class-relic-scientific-hydrosphere-snow-albedo-snowalbedo)=
## snow_albedo

Snow Albedo Dynamics Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.snow.albedo.SnowAlbedo` |
| Source | `relic\scientific\hydrosphere\snow\albedo.py:78` |
| Runtime name | `snow_albedo` |
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
snow_albedo(name: str = 'snow_albedo', config: Dict | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'snow_albedo'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `snowfall` | flux | `kg m-2 s-1` | fresh snow reset driver [kg m-2 s-1] |
| `snow_surface_temperature` | state | `K` | BATS snow surface temperature driver [K] |
| `air_temperature` | forcing | `K` | age-based snow aging temperature driver [K] |
| `solar_zenith_angle` | forcing | `rad` | solar zenith angle theta_z [rad] |
| `snow_depth` | diagnostic | `m` | snow presence mask depth [m] |
| `shortwave_down` | flux | `W m-2` | optional shortwave port effort I_sw, surface downwelling shortwave flux [W m-2] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `surface_albedo` | diagnostic | `dimensionless` | decided at run time by the returned write mode | diagnostic snow surface albedo alpha_s [dimensionless] |
| `snow_surface_age` | state | `day` | decided at run time by the returned write mode | age-based scheme snow age state t_age [day] |
| `snow_nondimensional_age` | state | `dimensionless` | decided at run time by the returned write mode | BATS scheme nondimensional snow age state tau_s [dimensionless] |
| `diagnostics.snow_albedo` | diagnostic | `dimensionless` | diagnostics | read-only aging, snow presence, shortwave partition, port-power, and residual diagnostics |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Input declaration difference: `snow_nondimensional_age`.
:::

### Governing equations and canonical operator

**Literature or process form**

BATS snow albedo evolves snow surface age with temperature, melt/refreeze, and dust terms, then computes broadband albedo from visible and near-infrared components; age-based mode uses $\alpha = \alpha_{\mathrm{old}} + ( \alpha_{\mathrm{fresh}} - \alpha_{\mathrm{old}} ) \exp ( - \mathrm{age} / \tau ) .$

**Canonical RELIC structural form**

alpha_s is a typed diagnostic 0-cochain, absorbed shortwave fraction is $q_{\mathrm{rad}} = \star_{\alpha} I_{\mathrm{sw}}$ with $\star_{\alpha} = \operatorname{diag} ( 1 - \alpha_{s} ) \geq 0$, and port power is $P_{\mathrm{rad}} = I_{\mathrm{sw}} \cdot ( 1 - \alpha_{s} ) .$

| Entry | Meaning |
| --- | --- |
| `star_alpha` | build_hodge_contract and build_hodge_terms HodgeStarSpec for shortwave absorption fraction |
| `bats_age` | temperature, melt/refreeze, and dust aging law |
| `spectral_average` | visible and near-infrared BATS albedo combination |
| `age_decay` | exponential age-based albedo decay |
| `projection` | albedo bound clipping residual routing |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | snow optical state cochain $x_{\alpha}$ |
| `effort_variable` | shortwave radiation input $I_{\mathrm{sw}}$ with albedo metric star_alpha |
| `state_advancer` | $q_{\mathrm{rad}} = \star_{\alpha} I_{\mathrm{sw}}$; $\star_{\alpha} = \operatorname{diag} ( \max ( 1 - \alpha_{s} , \epsilon ) )$ |
| `runtime_usage` | forward applies the canonical snow-albedo observation law and routes clipping, no-snow, radiation port-power, and age-tendency diagnostics. |

### Composition channels and executable PortSpec API

Certificate composition channels: `snow_albedo_radiation_port`

Executable `ports()` owner: `SnowAlbedo`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `snow_albedo_radiation_port` | `shortwave_down [W m-2]` | `snow_absorbed_shortwave_fraction [dimensionless]` | `W m-2` | into_system; radiation |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.numerical.epsilon_small` | `dimensionless` | `1e-09; bounds=(1e-12, 1e-06)` | `fixed` | Small numerical epsilon for stability in groundwater and snow calculations | Source: numerical_implementation; Numerical stability constraint for division and thresholding |
| `hydrosphere.snow.albedo_scheme` | `dimensionless` | `'bats'; bounds=None` | `fixed` | Snow albedo parameterization scheme (bats or age_based) | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note; RHESSys snowpack_daily_F.c implementation |
| `hydrosphere.snow.fresh_snow_albedo` | `dimensionless` | `0.85; bounds=(0.8, 0.95)` | `fixed` | Albedo of fresh snow (visible spectrum) | Source: radiative_transfer_theory; Warren (1982) Rev. Geophys. Space Phys. 20:67-89; Wiscombe & Warren (1980) J. Atmos. Sci. 37:2712-2733 |
| `hydrosphere.snow.old_snow_albedo` | `dimensionless` | `0.5; bounds=(0.3, 0.7)` | `fixed` | Albedo of old snow (lower bound for age-based model) | Source: BATS_model; Warren (1982) Rev. Geophys. Space Phys. 20:67-89; Dickinson et al. (1993) BATS Version 1e |
| `hydrosphere.snow.bare_ground_albedo_factor` | `dimensionless` | `0.6; bounds=(0.1, 1.0)` | `fixed` | Multiplicative factor mapping old snow albedo to bare-ground fallback albedo | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note; Molotch and Bales (2006) JGR 111:D23105 |
| `hydrosphere.snow.snow_reset_threshold` | `kg m-2 s-1` | `0.005; bounds=(0.001, 0.02)` | `fixed` | Snowfall threshold for resetting snow age to zero | Source: RHESSys_model; RHESSys snowpack_daily_F.c implementation; Molotch & Bales (2006) JGR 111:D23105 |
| `hydrosphere.snow.bats_b_parameter` | `dimensionless` | `2.0; bounds=(1.0, 4.0)` | `fixed` | BATS solar zenith angle factor for albedo calculation | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note |
| `hydrosphere.snow.dust_soot_factor` | `dimensionless` | `0.3; bounds=(0.1, 1.0)` | `fixed` | BATS dust and soot contamination factor for snow aging | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note; Warren & Wiscombe (1980) J. Atmos. Sci. 37:2712-2733 |
| `hydrosphere.snow.albedo_decay_rate_cold` | `s-1` | `1.16e-06; bounds=(1e-07, 1e-05)` | `fixed` | Albedo decay rate for cold conditions (T < 0 degC), tau_cold = 10 days = 1/(1.16e-6 s-1) | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note; RHESSys snowpack_daily_F.c implementation |
| `hydrosphere.snow.albedo_decay_rate_warm` | `s-1` | `1.16e-05; bounds=(1e-06, 0.0001)` | `fixed` | Albedo decay rate for warm conditions (T > 0 degC), tau_warm = 1 day = 1/(1.16e-5 s-1) | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note; RHESSys snowpack_daily_F.c implementation |
| `hydrosphere.snow.temperature_threshold_warm` | `K` | `273.15; bounds=(270.0, 275.0)` | `fixed` | Temperature threshold for switching from cold to warm albedo decay (0 degC = 273.15 K) | Source: snow_physics_literature; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note; Anderson (1976) NOAA Tech. Rep. NWS 19 |
| `hydrosphere.snow.bats_grain_growth_temperature_factor` | `K` | `5000.0; bounds=(1000.0, 10000.0)` | `fixed` | BATS grain growth temperature factor b in r1 = exp(b (1/T_ref - 1/T)); the exponent carries no additional dimensionless multiplier (D55) | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note NCAR/TN-387+STR; Yang et al. (1997) J. Climate 10:353-373, BATS snow submodel validation |
| `hydrosphere.snow.age_based_temperature_sensitivity` | `K-1` | `0.01; bounds=(0.0, 1.0)` | `fixed` | Warm-snow aging sensitivity kappa of the age-based albedo scheme; the effective age is t_age*exp(kappa*(T - T_warm)) for T above the warm threshold | Source: relic_age_based_albedo_closure; Warren & Wiscombe (1980) J. Atmos. Sci. 37:2712-2733; RELIC age-based snow albedo closure, docs/source/theory/hydrosphere_snow_albedo.md |
| `hydrosphere.snow.bats_reference_temperature` | `K` | `273.16; bounds=(260.0, 280.0)` | `fixed` | BATS reference temperature for grain growth exponent | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note |
| `hydrosphere.snow.bats_temperature_epsilon` | `K` | `1e-06; bounds=(1e-08, 0.001)` | `fixed` | Small temperature offset to avoid division by zero in BATS r1 term | Source: numerical_implementation; Numerical stability constraint for BATS albedo |
| `hydrosphere.snow.bats_cos_zenith_threshold` | `dimensionless` | `0.5; bounds=(0.1, 0.9)` | `fixed` | Cosine zenith threshold for BATS fz adjustment | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note |
| `hydrosphere.snow.bats_albedo_vis_base` | `dimensionless` | `0.95; bounds=(0.8, 1.0)` | `fixed` | BATS visible-band fresh snow albedo coefficient | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note |
| `hydrosphere.snow.bats_albedo_vis_age_factor` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | BATS visible-band aging factor applied to fage | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note |
| `hydrosphere.snow.bats_albedo_nir_base` | `dimensionless` | `0.65; bounds=(0.4, 0.8)` | `fixed` | BATS near-infrared fresh snow albedo coefficient | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note |
| `hydrosphere.snow.bats_albedo_nir_age_factor` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | BATS near-infrared aging factor applied to fage | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note |
| `hydrosphere.snow.bats_albedo_zenith_factor` | `dimensionless` | `0.4; bounds=(0.0, 1.0)` | `fixed` | BATS solar zenith albedo adjustment factor for VIS and NIR | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note |
| `hydrosphere.snow.bats_albedo_average_weight` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Weight for combining VIS and NIR albedo components | Source: BATS_model; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note |
| `hydrosphere.snow.aging_time_scale` | `day` | `5.0; bounds=(1.0, 30.0)` | `fixed` | Characteristic time scale for age-based albedo decay model | Source: snow_modeling_studies; Anderson (1976) NOAA Tech. Rep. NWS 19; Verseghy (1991) J. Clim. 4:530-550 |
| `hydrosphere.snow.minimum_snow_depth` | `m` | `0.01; bounds=(0.001, 0.1)` | `fixed` | Minimum snow depth threshold for albedo calculation | Source: snow_physics_literature; Anderson (1976) NOAA Tech. Rep. NWS 19; RHESSys implementation |
| `hydrosphere.snow.albedo_clamp_min` | `dimensionless` | `0.1; bounds=(0.0, 0.3)` | `fixed` | Minimum allowed surface albedo value | Source: physical_constraints; Physical bounds for surface albedo |
| `hydrosphere.snow.albedo_clamp_max` | `dimensionless` | `0.95; bounds=(0.9, 1.0)` | `fixed` | Maximum allowed surface albedo value | Source: physical_constraints; Physical bounds for surface albedo |
| `hydrosphere.snow.hodge_reference_surface_albedo` | `dimensionless` | `0.5; bounds=(0.1, 0.9)` | `fixed` | Reference aged-snow albedo alpha_ref of the static absorption Hodge contract of SnowAlbedo, star_alpha = diag(max(1 - alpha_ref, epsilon)) (build_hodge_contract); a structural reference state, never a runtime albedo and never substituted for surface_albedo | Source: snow_physics_literature; Warren, S.G. (1982). Optical properties of snow. Reviews of Geophysics 20, 67-89 (aged snow albedo); van der Schaft, A.J. and Maschke, B.M. (1995). The Hamiltonian formulation of energy conserving physical systems with external ports. AEU 49, 362-371 |
| `hydrosphere.snow.temperature_clamp_min` | `K` | `223.15; bounds=(200.0, 250.0)` | `fixed` | Minimum temperature for BATS aging calculation (-50 degC) | Source: numerical_stability; Physical bounds for snow surface temperature |
| `hydrosphere.snow.temperature_clamp_max` | `K` | `283.15; bounds=(273.15, 290.0)` | `fixed` | Maximum temperature for BATS aging calculation (+10 degC) | Source: numerical_stability; Physical bounds for snow surface temperature |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | outputs use wrap_update(UpdateMode.REPLACE) through existing StateManager path |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_snow_albedo.md`.

### References

- Dickinson, R.E. et al. (1993). BATS Version 1e. NCAR Technical Note (FunctorMetadata)
- Molotch, N.P., Bales, R.C. (2006). SNTHERM validation. JGR 111, D23105 (FunctorMetadata)
- RHESSys snowpack_daily_F.c implementation (FunctorMetadata)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- Dickinson, R.E., Henderson-Sellers, A., Kennedy, P.J. (1993). Biosphere-Atmosphere Transfer Scheme (BATS) Version 1e as Coupled to the NCAR Community Climate Model. (certificate)
- Warren, S.G. (1982). Optical properties of snow. Reviews of Geophysics and Space Physics, 20, 67-89. (certificate)
- Wiscombe, W.J. and Warren, S.G. (1980). A model for the spectral albedo of snow. Journal of the Atmospheric Sciences, 37, 2712-2733. (certificate)
- Molotch, N.P. and Bales, R.C. (2006). SNTHERM model validation using laboratory experiments. Journal of Geophysical Research, 111, D23105. (certificate)

(class-relic-scientific-hydrosphere-snow-snowpack-snowpack)=
## snow_pack

SnowPack SWE Mass Balance Model.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.snow.snowpack.SnowPack` |
| Source | `relic\scientific\hydrosphere\snow\snowpack.py:67` |
| Runtime name | `snow_pack` |
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
snow_pack(name: str = 'snow_pack', config: Dict | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'snow_pack'` | Functor identifier. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Optional configuration overrides. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `precipitation_flux` | forcing | `kg m-2 s-1` | total precipitation mass flux [kg m-2 s-1] |
| `air_temperature` | forcing | `K` | air temperature controlling phase partition and melt [K] |
| `snow_water_equivalent` | state | `mm` | snow water equivalent storage SWE [mm] |
| `sublimation_flux` | flux | `kg m-2 s-1` | snow sublimation mass flux [kg m-2 s-1] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `snow_water_equivalent` | state | `mm` | decided at run time by the returned write mode | snow water equivalent storage SWE [mm] |
| `snow_depth` | diagnostic | `m` | decided at run time by the returned write mode | bulk-density snow depth estimate [m] |
| `snow_melt_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | meltwater release mass flux [kg m-2 s-1] |
| `rain_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | rainfall mass flux after phase partition [kg m-2 s-1] |
| `snowfall` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | snowfall mass flux after phase partition [kg m-2 s-1] |
| `diagnostics.snow_pack` | diagnostic | `dimensionless` | diagnostics | read-only mass partition, storage, and residual diagnostics |

### Governing equations and canonical operator

**Literature or process form**

$\frac{\mathrm{d}\mathrm{SWE}}{\mathrm{d}t} = P_{\mathrm{snow}} - M - E_{\mathrm{sub}}$ with $M = \mathrm{DDF} \max ( 0 , T_{\mathrm{air}} - T_{\mathrm{melt}} ) / \mathrm{seconds}_{\mathrm{per},\mathrm{day}}$, following Anderson (1976) and Hock (2003).

**Canonical RELIC structural form**

SWE is a typed storage 0-cochain and $\frac{\mathrm{d}\mathrm{SWE}}{\mathrm{d}t} = \star_{\mathrm{SWE}} ( P_{\mathrm{snow}} - M - E_{\mathrm{sub}} )$, with $\star_{\mathrm{SWE}} = \operatorname{diag} ( 1000 / \rho_{w} ) \geq 0 .$

| Entry | Meaning |
| --- | --- |
| `star_SWE` | build_hodge_contract and build_hodge_terms HodgeStarSpec for mass-flux to SWE-rate conversion |
| `precipitation_partition` | temperature threshold partition between snowfall and rainfall |
| `degree_day_melt` | $\mathrm{DDF} \max ( 0 , T_{\mathrm{air}} - T_{\mathrm{melt}} ) / \mathrm{seconds}_{\mathrm{per},\mathrm{day}}$ |
| `storage_projection` | nonnegative SWE projection and capacity residual routing |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | snow water equivalent state cochain SWE_h |
| `effort_variable` | accumulation and melt matter ports with SWE metric star_SWE |
| `state_advancer` | $\mathrm{dSWE}_{\mathrm{dt}} = \mathrm{accumulation} - \mathrm{melt} - \mathrm{sublimation}$ |
| `runtime_usage` | forward applies the canonical SWE field balance and routes melt, sublimation, and nonnegative projection residuals. |
| `time_integration` | sequential starting-inventory melt then sublimation caps, followed by snowfall addition and nonnegative SWE replacement; no second time multiplication |
| `sub_flow_accuracy` | FunctorMetadata.sub_flow_accuracy ORDER_1 conservatively for the complete update; fixed-input inactive-cap affine branches are exact to roundoff, while T11 demonstrates first-order replenishment error and a non-semigroup exported-flux partition at depletion |
| `splitting_boundary` | the existing S1 same-fast-group snowpack-to-bucket flux-coupling refusal remains; ORDER_1 is not global EXACT, second-order Strang or strict IPHS certification |

### Composition channels and executable PortSpec API

Certificate composition channels: `snowpack_accumulation_port`, `snowpack_melt_release_port`

Executable `ports()` owner: `SnowPack`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `snowpack_accumulation_port` | `snow_water_potential [m]` | `snowfall [kg m-2 s-1]` | `kg m-1 s-1` | into_system; matter |
| `snowpack_melt_release_port` | `snow_water_potential [m]` | `snow_melt_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.snow.degree_day_factor` | `mm K-1 day-1` | `5.0; bounds=(2.0, 10.0)` | `fixed` | Degree day melt factor for snow | Source: snow_physics_literature; Hock (2003) J. Hydrol. 282:104-115; Anderson (1976) NOAA Tech. Rep. NWS 19 |
| `hydrosphere.snow.temperature_threshold_warm` | `K` | `273.15; bounds=(270.0, 275.0)` | `fixed` | Temperature threshold for switching from cold to warm albedo decay (0 degC = 273.15 K) | Source: snow_physics_literature; Dickinson et al. (1993) BATS Version 1e NCAR Tech Note; Anderson (1976) NOAA Tech. Rep. NWS 19 |
| `hydrosphere.snow.rain_snow_temperature_threshold` | `K` | `273.15; bounds=(270.0, 276.0)` | `fixed` | Temperature threshold for rain snow partitioning | Source: snow_physics_literature; Feiccabrino et al. (2013) Hydrol. Earth Syst. Sci. 17:2689-2707; Anderson (1976) NOAA Tech. Rep. NWS 19 |
| `hydrosphere.snow.snow_density` | `kg m-3` | `250.0; bounds=(50.0, 500.0)` | `fixed` | Bulk snow density for SWE to depth conversion | Source: snow_physics_literature; Anderson (1976) NOAA Tech. Rep. NWS 19; Hock (2003) J. Hydrol. 282:104-115 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | outputs use wrap_update(UpdateMode.REPLACE) through existing StateManager path |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_snowpack.md`.

### References

- Anderson, E.A. (1976). NOAA Tech. Rep. NWS 19 (FunctorMetadata)
- Hock, R. (2003). J. Hydrol. 282:104-115 (FunctorMetadata)
- RHESSys snowpack_daily_F.c implementation (FunctorMetadata)
- Biome-BGC snowmelt.c implementation (FunctorMetadata)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- Anderson, E.A. (1976). A point energy and mass balance model of a snow cover. NOAA Technical Report NWS 19. (certificate)
- Hock, R. (2003). Temperature index melt modeling in mountain areas. Journal of Hydrology, 282, 104-115. (certificate)
- RHESSys snowpack_daily_F.c implementation. (certificate)
- Biome-BGC snowmelt.c implementation. (certificate)

(class-relic-scientific-hydrosphere-soil-moisture-bucket-bucketsoilmoisture)=
## bucket_soil_moisture

Bucket Soil Moisture Model

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.soil_moisture.bucket.BucketSoilMoisture` |
| Source | `relic\scientific\hydrosphere\soil_moisture\bucket.py:87` |
| Runtime name | `bucket_soil_moisture` |
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
bucket_soil_moisture(name: str = 'bucket_soil_moisture', config: Dict | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'bucket_soil_moisture'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | volumetric soil water content theta [m3 m-3] |
| `precipitation_flux` | forcing | `kg m-2 s-1` | incoming precipitation matter flux P, read only with water_input_source precipitation_flux [kg m-2 s-1] |
| `evapotranspiration_flux` | flux | `kg m-2 s-1` | outgoing evapotranspiration matter flux ET [kg m-2 s-1] |
| `rain_flux` | flux | `kg m-2 s-1` | rain reaching the ground from the snowpack partition, read only with water_input_source rain_flux_plus_snow_melt_flux [kg m-2 s-1] |
| `snow_melt_flux` | flux | `kg m-2 s-1` | snowmelt release of the snowpack, read with rain_flux only under water_input_source rain_flux_plus_snow_melt_flux [kg m-2 s-1] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | decided at run time by the returned write mode | volumetric soil water content theta [m3 m-3] |
| `surface_runoff_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | saturation-excess runoff matter flux Q [kg m-2 s-1] |
| `drainage_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | deep drainage matter flux D [kg m-2 s-1] |
| `diagnostics.soil_moisture_bucket` | output port | `not declared` | diagnostics | read-only residual and water-stress diagnostics |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Input declaration difference: `rain_flux`, `snow_melt_flux`.
:::

### Governing equations and canonical operator

**Literature or process form**

Single-layer bucket hydrology conserves water storage through P - ET - Q - D and uses empirical runoff and drainage closures bounded by wilting point and saturation.

**Canonical RELIC structural form**

$$
\dot{\theta} = ( P - \mathrm{ET} - Q - D ) / ( \rho_{w} \cdot Z ); q = \star_{K} g_{h}; \star_{K} = \operatorname{diag} ( \rho_{w} \cdot K_{\mathrm{sat}} ) \geq 0 .
$$

| Entry | Meaning |
| --- | --- |
| `B_h` | bucket storage balance incidence for P, ET, Q, and D |
| `star_K` | build_hodge_contract and build_hodge_terms HodgeStarSpec |
| `runoff_closure` | saturation-excess projection above theta_sat |
| `drainage_closure` | $K_{\mathrm{sat}} \cdot f_{\mathrm{drainage}} \cdot ( \theta / \theta_{\mathrm{sat}} ) ^ b$ limited by water above field capacity |
| `unit_conversion` | mass flux divided by rho_w and storage conversion $\theta \cdot Z \cdot \rho_{w}$ |
| `time_integration` | sequential held-input P-ET, saturation-excess runoff, explicit power-law drainage and bounded replacement of the integrated soil_moisture state; no second time multiplication |
| `sub_flow_accuracy` | FunctorMetadata.sub_flow_accuracy ORDER_1 on smooth inactive-limit branches; the unforced witness is $\dot{\theta} = - k \theta ^ b$ with $k = K_{\mathrm{sat}} f_{D} / ( Z \theta_{\mathrm{sat}} ^ b )$, advanced by explicit Euler; T11 validates the independent analytic solution and records the coarse drainage-cap plateau |
| `splitting_boundary` | the existing S1 same-fast-group snowpack-to-bucket flux-coupling refusal is unchanged; ORDER_1 is neither an EXACT declaration nor second-order Strang certification |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | bucket soil water storage cochain theta_h |
| `effort_variable` | infiltration, ET, runoff, and drainage matter ports with storage capacity rho_w Z |
| `state_advancer` | $\dot{\theta} = ( P - \mathrm{ET} - Q - D ) / ( \rho_{w} \cdot Z )$ |
| `runtime_usage` | forward applies the canonical bucket storage-port balance and routes overflow or clipping residuals. |

### Composition channels and executable PortSpec API

Certificate composition channels: `bucket_evapotranspiration_flux`, `bucket_surface_runoff_flux`, `bucket_drainage_flux`

Executable `ports()` owner: `BucketSoilMoisture`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `bucket_evapotranspiration_flux` | `bucket_water_potential [m]` | `evapotranspiration_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |
| `bucket_surface_runoff_flux` | `bucket_water_potential [m]` | `surface_runoff_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |
| `bucket_drainage_flux` | `bucket_water_potential [m]` | `drainage_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.soil_moisture.soil_depth` | `m` | `1.0; bounds=(0.1, 5.0)` | `learnable` | Effective soil layer depth for bucket model water balance | Source: bucket_model_parameterization; Manabe (1969) Mon. Wea. Rev. 97:739-774; Huang et al. (1996) J. Geophys. Res. 101:19079-19093 |
| `hydrosphere.soil_moisture.field_capacity` | `m3 m-3` | `0.3; bounds=(0.1, 0.7)` | `learnable` | Soil field capacity (maximum water content at gravitational drainage threshold) | Source: literature_synthesis; Manabe (1969) Mon. Wea. Rev. 97:739-774; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.wilting_point` | `m3 m-3` | `0.1; bounds=(0.05, 0.35)` | `learnable` | Soil wilting point (minimum plant-available water content) | Source: literature_synthesis; Manabe (1969) Mon. Wea. Rev. 97:739-774; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.porosity` | `m3 m-3` | `0.45; bounds=(0.3, 0.8)` | `fixed` | Total soil porosity (saturated water content) | Source: soil_survey_data; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.saturated_hydraulic_conductivity` | `m s-1` | `1e-05; bounds=(1e-08, 0.001)` | `learnable` | Saturated hydraulic conductivity for Richards equation | Source: field_measurements; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.drainage_factor` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Free-drainage factor f_d of the bucket gravity-drainage law, K_eff = f_d K_sat in D = K_eff (theta/theta_sat)^b above field capacity; a process-law parameter read every call (the value 1 applies K_sat unscaled) | Source: bucket_model_parameterization; Manabe (1969) Mon. Wea. Rev. 97:739-774 |
| `hydrosphere.soil_moisture.drainage_exponent` | `dimensionless` | `3.0; bounds=(1.0, 10.0)` | `learnable` | Drainage curve exponent for bucket model (power law parameter) | Source: bucket_model_parameterization; Manabe (1969) Mon. Wea. Rev. 97:739-774; Huang et al. (1996) J. Geophys. Res. 101:19079-19093 |
| `hydrosphere.soil_moisture.water_input_source` | `dimensionless` | `'precipitation_flux'; bounds=None` | `fixed` | Declared water input of the bucket soil store. precipitation_flux reads the total precipitation (no snowpack Functor upstream); rain_flux_plus_snow_melt_flux reads the rain reaching the ground plus the snowmelt of the snowpack partition and declares the snowpack Functor (SCM owner of both fluxes) as a lagged provider whose committed outflow the bucket consumes. Exactly one representation is read; an instance config key of the same name overrides this value. | Source: model_configuration; E3SM ELM SnowHydrologyMod, column without snow layers (qflx_top_soil = qflx_rain_grnd + qflx_snomelt); Oleson et al. (2013) Technical Description of version 4.5 of the Community Land Model (CLM), NCAR Technical Note NCAR/TN-503+STR |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | canonical state and flux outputs use explicit UpdateItem(REPLACE) through StateManager |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_bucket.md`.

### References

- Manabe, S. (1969). Monthly Weather Review, 97(11), 739-774 (FunctorMetadata)
- Huang, J. et al. (1996). Journal of Climate, 9(6), 1350-1362 (FunctorMetadata)
- Rodriguez-Iturbe, I. and Porporato, A. (2004). Ecohydrology of water-controlled ecosystems (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Manabe, S. (1969). Climate and the ocean circulation. Monthly Weather Review, 97, 739-774. (certificate)
- Huang, J., van den Dool, H.M., and Georgakakos, K.P. (1996). Analysis of model-calculated soil moisture. Journal of Climate, 9, 1350-1362. (certificate)
- Rodriguez-Iturbe, I. and Porporato, A. (2004). Ecohydrology of water-controlled ecosystems. Cambridge University Press. (certificate)
- Eagleson, P.S. (1978). Climate, soil, and vegetation: water balance dynamics. Water Resources Research, 14, 705-712. (certificate)

(class-relic-scientific-hydrosphere-soil-moisture-retention-soilwaterretention)=
## SoilWaterRetention

Soil water potential of the effective soil layer from its water content.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.soil_moisture.retention.SoilWaterRetention` |
| Source | `relic\scientific\hydrosphere\soil_moisture\retention.py:51` |
| Runtime name | `soil_water_retention` |
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
SoilWaterRetention(name: str = 'soil_water_retention', config: Dict[str, Any] | None = None, device: torch.device | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'soil_water_retention'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | required external or upstream StateManager port in m3 m-3 |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `soil_water_potential` | diagnostic | `MPa` | decided at run time by the returned write mode | canonical non-positive output from the retention observation map |

### Governing equations and canonical operator

**Literature or process form**

$$
\mathrm{Se} = ( \theta - \theta_{r} ) / ( \theta_{s} - \theta_{r} ); h = - ( \mathrm{Se} ^ ( - 1 / m ) - 1 ) ^ ( 1 / n ) / \alpha, m = 1 - 1 / n; \psi = \rho_{w} g h
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
| `potential_basis` | soil_moisture observation input in m3 m-3 |
| `effort_variable` | not applicable to this empirical level-1 observation |
| `state_advancer` | $y_{h} = O_{h} x_{h}$ through the declared observation operator |
| `runtime_usage` | SoilWaterRetention.forward evaluates the shared van Genuchten pressure head once and emits soil_water_potential without advancing water state. |

### Composition channels and executable PortSpec API

Certificate composition channels: `soil_moisture`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.soil_moisture.alpha_vg` | `m-1` | `7.5; bounds=(0.1, 15.0)` | `learnable` | Van Genuchten alpha parameter (inverse air entry pressure) | Source: pedotransfer_functions; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Carsel & Parrish (1988) Water Resour. Res. 24:755-769 (class means as tabulated in van Genuchten et al. 1991, EPA/600/2-91/065, Table 4) |
| `hydrosphere.soil_moisture.n_vg` | `dimensionless` | `1.89; bounds=(1.1, 10.0)` | `learnable` | Van Genuchten n parameter (pore size distribution index) | Source: pedotransfer_functions; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Carsel & Parrish (1988) Water Resour. Res. 24:755-769 (class means as tabulated in van Genuchten et al. 1991, EPA/600/2-91/065, Table 4) |
| `hydrosphere.soil_moisture.residual_water_content` | `m3 m-3` | `0.045; bounds=(0.0, 0.2)` | `fixed` | Residual water content (minimum water content at very high suction) | Source: laboratory_measurements; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Schaap et al. (2001) J. Hydrol. 251:163-176 |
| `hydrosphere.soil_moisture.porosity` | `m3 m-3` | `0.45; bounds=(0.3, 0.8)` | `fixed` | Total soil porosity (saturated water content) | Source: soil_survey_data; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Cosby et al. (1984) Water Resour. Res. 20:682-690 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | UpdateItem REPLACE with per_timestep false |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_retention.md`.

### References

- van Genuchten, M.T. (1980). Soil Science Society of America Journal, 44, 892-898 (FunctorMetadata, certificate)
- Hillel, D. (1998). Environmental Soil Physics. Academic Press (FunctorMetadata, certificate)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)

(class-relic-scientific-hydrosphere-soil-moisture-richards-richardssoilmoisture)=
## richards_soil_moisture

Richards Equation Soil Moisture Model Framework

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.soil_moisture.richards.RichardsSoilMoisture` |
| Source | `relic\scientific\hydrosphere\soil_moisture\richards.py:89` |
| Runtime name | `richards_soil_moisture` |
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
richards_soil_moisture(name: str = 'richards_soil_moisture', config: Dict | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'richards_soil_moisture'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | volumetric water content theta profile [m3 m-3] |
| `infiltration_flux` | flux | `kg m-2 s-1` | non-negative surface water input [kg m-2 s-1] |
| `evapotranspiration_flux` | flux | `kg m-2 s-1` | non-negative surface water export [kg m-2 s-1] |
| `capillary_rise_flux` | flux | `kg m-2 s-1` | optional non-negative groundwater-to-soil input selected by the capillary lower-boundary mode [kg m-2 s-1] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | decided at run time by the returned write mode | volumetric water content theta profile [m3 m-3] |
| `diagnostics.soil_moisture_richards.soil_moisture_change_rate` | output port | `m3 m-3 s-1` | diagnostics | full-profile water-content rate [m3 m-3 s-1] |
| `diagnostics.soil_moisture_richards.water_flux_profile` | output port | `kg m-2 s-1` | diagnostics | adaptive-substep time-mean vertical flux profile [kg m-2 s-1] |
| `soil_matric_potential` | diagnostic | `Pa` | decided at run time by the returned write mode | full-profile matric pressure rho_w g h [Pa] |
| `diagnostics.soil_moisture_richards.relative_moisture` | output port | `dimensionless` | diagnostics | field-capacity-normalized relative moisture observation [dimensionless] |
| `drainage_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | bottom boundary drainage flux [kg m-2 s-1] |
| `diagnostics.soil_moisture_richards` | output port | `not declared` | diagnostics | read-only solver and residual diagnostics |

### Governing equations and canonical operator

**Literature or process form**

Unsaturated vertical water flow follows continuity and Darcy flux $q = - K ( \theta ) ( \frac{\mathrm{d}h}{\mathrm{d}z} - 1 )$, with retention and conductivity from van Genuchten-Mualem laws.

**Canonical RELIC structural form**

$$
\dot{\theta} = - D_{z} q_{z} + s_{z}; q_{z} = \star_{K} ( d h - g_{z} ); \star_{K} = \operatorname{diag} ( \rho_{w} \cdot K_{s} ) \geq 0 .
$$

| Entry | Meaning |
| --- | --- |
| `D_z` | vertical discrete divergence from diagnostics.soil_moisture_richards.water_flux_profile to soil_moisture |
| `star_K` | build_hodge_contract and build_hodge_terms HodgeStarSpec |
| `retention_curve` | _compute_pressure_head van Genuchten relation |
| `matric_pressure_conversion` | $\mathrm{soil}_{\mathrm{matric},\mathrm{potential}} = \rho_{w} \cdot g \cdot \mathrm{pressure}_{\mathrm{head}}$ |
| `conductivity_law` | _compute_hydraulic_conductivity Mualem relation |
| `solver` | ParameterManager selects modified Picard or exact-Jacobian Newton, both evaluating the same canonical mixed-form residual |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | vertical soil moisture cochain theta_h |
| `effort_variable` | hydraulic head gradient d h - $g_{z}$ with conductivity mobility star_K |
| `state_advancer` | $\dot{\theta} = - D_{z} q_{z} + s_{z}$; $q_{z} = \star_{K} ( d h - g_{z} )$ |
| `runtime_usage` | forward applies the canonical mixed-form Richards residual, bounds the inner residual tolerance by the water ledger, and routes mass-balance and solver diagnostics. |

### Composition channels and executable PortSpec API

Certificate composition channels: `soil_infiltration_boundary`, `soil_evapotranspiration_boundary`, `soil_capillary_boundary`, `richards_drainage_flux`

Executable `ports()` owner: `RichardsSoilMoisture`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `soil_infiltration_boundary` | `surface_hydraulic_head [m]` | `infiltration_flux [kg m-2 s-1]` | `kg m-1 s-1` | into_system; matter |
| `soil_evapotranspiration_boundary` | `surface_hydraulic_head [m]` | `evapotranspiration_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |
| `soil_capillary_boundary` | `water_table_boundary_head [m]` | `capillary_rise_flux [kg m-2 s-1]` | `kg m-1 s-1` | into_system; matter |
| `richards_drainage_flux` | `richards_pressure_head [m]` | `drainage_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.soil_moisture.layer_count` | `dimensionless` | `10; bounds=(1.0, 200.0)` | `fixed` | Number of discrete soil layers used in Richards equation solver | Source: richards_solver_configuration; Celia et al. (1990) Water Resour. Res. 26:1483-1496; Šimůnek et al. (2008) Vadose Zone J. 7:587-600 |
| `hydrosphere.soil_moisture.soil_depth` | `m` | `1.0; bounds=(0.1, 5.0)` | `learnable` | Effective soil layer depth for bucket model water balance | Source: bucket_model_parameterization; Manabe (1969) Mon. Wea. Rev. 97:739-774; Huang et al. (1996) J. Geophys. Res. 101:19079-19093 |
| `hydrosphere.soil_moisture.residual_water_content` | `m3 m-3` | `0.045; bounds=(0.0, 0.2)` | `fixed` | Residual water content (minimum water content at very high suction) | Source: laboratory_measurements; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Schaap et al. (2001) J. Hydrol. 251:163-176 |
| `hydrosphere.soil_moisture.porosity` | `m3 m-3` | `0.45; bounds=(0.3, 0.8)` | `fixed` | Total soil porosity (saturated water content) | Source: soil_survey_data; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.alpha_vg` | `m-1` | `7.5; bounds=(0.1, 15.0)` | `learnable` | Van Genuchten alpha parameter (inverse air entry pressure) | Source: pedotransfer_functions; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Carsel & Parrish (1988) Water Resour. Res. 24:755-769 (class means as tabulated in van Genuchten et al. 1991, EPA/600/2-91/065, Table 4) |
| `hydrosphere.soil_moisture.n_vg` | `dimensionless` | `1.89; bounds=(1.1, 10.0)` | `learnable` | Van Genuchten n parameter (pore size distribution index) | Source: pedotransfer_functions; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Carsel & Parrish (1988) Water Resour. Res. 24:755-769 (class means as tabulated in van Genuchten et al. 1991, EPA/600/2-91/065, Table 4) |
| `hydrosphere.soil_moisture.saturated_hydraulic_conductivity` | `m s-1` | `1e-05; bounds=(1e-08, 0.001)` | `learnable` | Saturated hydraulic conductivity for Richards equation | Source: field_measurements; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.tortuosity_parameter` | `dimensionless` | `0.5; bounds=(-3.0, 3.0)` | `fixed` | Mualem tortuosity parameter for relative hydraulic conductivity | Source: theoretical_constraint; Mualem (1976) Water Resour. Res. 12:513-522; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898 |
| `hydrosphere.physical.field_capacity_fraction` | `dimensionless` | `0.7; bounds=(0.5, 0.9)` | `learnable` | Field capacity as fraction of saturated water content (typical for loam) | Source: soil_physics_literature; Brady & Weil (2008) The Nature and Properties of Soils, 14th ed; Saxton & Rawls (2006) Soil Sci. Soc. Am. J. 70:1569-1578 |
| `hydrosphere.physical.wilting_point_fraction` | `dimensionless` | `1.5; bounds=(1.2, 2.0)` | `learnable` | Wilting point as multiplier of residual water content (typical for loam) | Source: soil_physics_literature; Brady & Weil (2008) The Nature and Properties of Soils, 14th ed; Saxton & Rawls (2006) Soil Sci. Soc. Am. J. 70:1569-1578 |
| `hydrosphere.numerical.richards_max_picard_iterations` | `dimensionless` | `50; bounds=(1.0, 100.0)` | `fixed` | Maximum number of Picard iterations for Richards equation linearization | Source: computational_experience; Celia et al. (1990) Water Resour. Res. 26:1483-1496 |
| `hydrosphere.numerical.richards_max_newton_iterations` | `dimensionless` | `50; bounds=(1.0, 200.0)` | `fixed` | Maximum number of Newton-Raphson iterations for Richards equation | Source: nonlinear_solver_experience; Lehmann & Ackerer (1998) Adv. Water Resour. 21:97-107; Kelley (2003) Solving Nonlinear Equations with Newton's Method |
| `hydrosphere.numerical.richards_convergence_tolerance` | `m3 m-3` | `1e-06; bounds=(1e-12, 0.001)` | `fixed` | Primary convergence tolerance for Richards equation Picard iteration | Source: numerical_methods_optimization; Celia et al. (1990) Water Resour. Res. 26:1483-1496; Lehmann & Ackerer (1998) Adv. Water Resour. 21:97-107 |
| `hydrosphere.numerical.richards_newton_absolute_tolerance` | `m3 m-3` | `1e-08; bounds=(1e-16, 0.001)` | `fixed` | Absolute tolerance for Newton-Raphson method in Richards equation solver | Source: nonlinear_solver_theory; Lehmann & Ackerer (1998) Adv. Water Resour. 21:97-107; Kelley (2003) Solving Nonlinear Equations with Newton's Method |
| `hydrosphere.numerical.richards_newton_relative_tolerance` | `dimensionless` | `1e-06; bounds=(1e-12, 0.001)` | `fixed` | Relative tolerance for Newton-Raphson method in Richards equation solver | Source: nonlinear_solver_theory; Lehmann & Ackerer (1998) Adv. Water Resour. 21:97-107; Kelley (2003) Solving Nonlinear Equations with Newton's Method |
| `hydrosphere.soil_moisture.richards_enable_newton_raphson` | `dimensionless` | `False; bounds=None` | `fixed` | Enables Newton-Raphson corrections inside the Richards equation framework. | Source: Richards solver configuration; Celia et al. (1990) Water Resour. Res. 26:1483-1496; Simunek et al. (2008) Vadose Zone J. 7:587-600 |
| `hydrosphere.soil_moisture.richards_enable_mass_conservation_check` | `dimensionless` | `True; bounds=None` | `fixed` | Controls enforcement of layer-by-layer mass conservation diagnostics for Richards solver. | Source: Richards solver configuration; Richards (1931) Journal of Applied Physics 1:318-333; Celia et al. (1990) Water Resour. Res. 26:1483-1496 |
| `hydrosphere.numerical.mass_conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-12, 0.001)` | `fixed` | Tolerance for mass conservation validation in hydrosphere processes | Source: conservation_laws; Celia et al. (1990) Water Resour. Res. 26:1483-1496 |
| `hydrosphere.numerical.richards_cfl_coefficient` | `dimensionless` | `0.5; bounds=(0.1, 1.0)` | `fixed` | Courant-Friedrichs-Lewy (CFL) stability coefficient for Richards equation time stepping | Source: numerical_methods_literature; Courant et al. (1928) Math. Ann. 100:32-74; Celia et al. (1990) Water Resour. Res. 26:1483-1496 |
| `hydrosphere.numerical.saturation_epsilon` | `dimensionless` | `1e-06; bounds=(1e-12, 0.001)` | `fixed` | Small epsilon for effective saturation bounds to avoid numerical issues | Source: soil_physics_modeling; van Genuchten, M.T. (1980) Soil Sci. Soc. Am. J. 44:892-898 |
| `hydrosphere.numerical.minimum_pressure_term` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-06)` | `fixed` | Minimum value for pressure term in van Genuchten model to avoid numerical issues | Source: numerical_implementation; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898 |
| `hydrosphere.numerical.minimum_se_term` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-06)` | `fixed` | Minimum value for effective saturation term to avoid numerical issues | Source: numerical_implementation; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898 |
| `hydrosphere.numerical.minimum_term1` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-06)` | `fixed` | Minimum value for intermediate term1 in van Genuchten derivative calculation | Source: numerical_implementation; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898 |
| `hydrosphere.numerical.minimum_timestep` | `s` | `1e-06; bounds=(1e-12, 0.001)` | `fixed` | Minimum time step to avoid numerical issues in Richards equation | Source: numerical_methods_optimization; Celia et al. (1990) Water Resour. Res. 26:1483-1496 |
| `hydrosphere.numerical.harmonic_mean_factor` | `dimensionless` | `2.0; bounds=(1.5, 3.0)` | `fixed` | Factor for harmonic mean calculation of interface hydraulic conductivity | Source: finite_difference_methods; Celia et al. (1990) Water Resour. Res. 26:1483-1496; Haverkamp & Vauclin (1979) Water Resour. Res. 15:271-284 |
| `hydrosphere.numerical.tridiagonal_tolerance` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-06)` | `fixed` | Tolerance for tridiagonal matrix solver convergence | Source: numerical_recipes; Press et al. (2007) Numerical Recipes, 3rd Ed. |
| `hydrosphere.numerical.line_search_initial_alpha` | `dimensionless` | `1.0; bounds=(0.1, 2.0)` | `fixed` | Initial step size alpha for line search algorithm in Newton methods | Source: optimization_theory; Nocedal & Wright (2006) Numerical Optimization; Kelley (2003) Solving Nonlinear Equations with Newton's Method |
| `hydrosphere.numerical.line_search_max_backtracks` | `dimensionless` | `10; bounds=(1.0, 50.0)` | `fixed` | Maximum number of backtracking steps in line search for Newton methods | Source: optimization_theory; Nocedal & Wright (2006) Numerical Optimization; Kelley (2003) Solving Nonlinear Equations with Newton's Method |
| `hydrosphere.numerical.line_search_armijo_constant` | `dimensionless` | `0.0001; bounds=(1e-06, 0.1)` | `fixed` | Armijo constant for sufficient decrease condition in line search | Source: optimization_theory; Armijo (1966) Pacific J. Math. 16:1-3; Nocedal & Wright (2006) Numerical Optimization |
| `hydrosphere.numerical.line_search_alpha_reduction_factor` | `dimensionless` | `0.5; bounds=(0.1, 0.9)` | `fixed` | Reduction factor for alpha in backtracking line search | Source: optimization_theory; Nocedal & Wright (2006) Numerical Optimization; Armijo (1966) Pacific J. Math. 16:1-3 |
| `hydrosphere.numerical.line_search_min_alpha` | `dimensionless` | `0.0001; bounds=(1e-08, 0.01)` | `fixed` | Minimum step size alpha in line search algorithm | Source: optimization_theory; Nocedal & Wright (2006) Numerical Optimization |
| `hydrosphere.numerical.characteristic_pressure_head` | `m` | `1.0; bounds=(0.1, 10.0)` | `fixed` | Characteristic pressure head scale for Richards equation normalization | Source: numerical_methods_optimization; Celia et al. (1990) Water Resour. Res. 26:1483-1496 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | all retained state, tendency, flux, matric-pressure, and relative-moisture outputs use explicit UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} )$ with unit and source_functor through the existing StateManager path |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_richards.md`.

### References

- Richards, L.A. (1931). Journal of Applied Physics, 1(5), 318-333 (FunctorMetadata)
- Van Genuchten, M.T. (1980). Soil Science Society of America Journal, 44(5), 892-898 (FunctorMetadata)
- Mualem, Y. (1976). Water Resources Research, 12(3), 513-522 (FunctorMetadata)
- Feddes et al. (1978). Simulation of Field Water Use and Crop Yield (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Richards, L.A. (1931). Capillary conduction of liquids through porous mediums. Physics, 1, 318-333. (certificate)
- van Genuchten, M.T. (1980). A closed-form equation for predicting hydraulic conductivity. Soil Science Society of America Journal, 44, 892-898. (certificate)
- Mualem, Y. (1976). A new model for predicting hydraulic conductivity of unsaturated porous media. Water Resources Research, 12, 513-522. (certificate)
- Celia, M.A., Bouloutas, E.T., and Zarba, R.L. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. Water Resources Research, 26, 1483-1496. (certificate)
- Feddes, R.A., Kowalik, P.J., and Zaradny, H. (1978). Simulation of Field Water Use and Crop Yield. Pudoc, Wageningen. (certificate)

(class-relic-scientific-hydrosphere-transport-capillary-rise-capillaryrisetransport)=
## capillary_rise_transport

Capillary Rise Transport Mechanism

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.transport.capillary_rise.CapillaryRiseTransport` |
| Source | `relic\scientific\hydrosphere\transport\capillary_rise.py:63` |
| Runtime name | `capillary_rise_transport` |
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
capillary_rise_transport(name: str = 'capillary_rise_transport', config: Dict | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'capillary_rise_transport'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `water_table_depth` | state | `m` | boundary effort cochain Z_h [m] |
| `soil_moisture_profile` | state | `m3 m-3` | optional saturation limiter state cochain theta_h [m3 m-3] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `capillary_rise_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | port flow u_cap after density conversion [kg m-2 s-1] |
| `diagnostics.capillary_rise` | output port | `not declared` | diagnostics | read-only diagnostic payload |
| `diagnostics.capillary_rise.conductivity_clipping_residual` | output port | `kg m-2 s-1` | diagnostics | rho_w * max(B*a/Z^b - Ks, 0) |
| `diagnostics.capillary_rise.saturation_constraint_residual` | output port | `kg m-2 s-1` | diagnostics | diagnostic rho_w * (unconstrained q(z) profile - constrained q(z) profile); not a boundary-flow or system mass-balance residual |

### Governing equations and canonical operator

**Literature or process form**

Gardner-Eagleson steady capillary rise from a water table, derived as a steady upward flux approximation for Richards equation boundary transport.

**Canonical RELIC structural form**

$\dot{m} = G_{\mathrm{cap}} u_{\mathrm{cap}}$; $u_{\mathrm{cap}} = \star_{\mathrm{cap}} u_{\mathrm{vol}}$; $u_{\mathrm{vol}} = \max ( 0 , \min ( \mathrm{Ks} , B \cdot a / Z ^ b ) ) [ m s - 1 ]$; $\star_{\mathrm{cap}} = \operatorname{diag} ( \rho_{w} ) [ \mathrm{kg} m - 3 ] > 0 \mathrm{maps} u_{\mathrm{vol}}$ to the mass flux [kg m-2 s-1]; the Ks cap belongs to $u_{\mathrm{vol}}$, not to star_cap.

| Entry | Meaning |
| --- | --- |
| `G_cap` | capillary_groundwater_boundary_flux port incidence |
| `star_cap` | build_hodge_contract and build_hodge_terms HodgeStarSpec with matrix diag(rho_w), operator_units kg m-3, domain m s-1, codomain kg m-2 s-1 |
| `gardner_flux` | _compute_gardner_flux_with_residuals implements $B \cdot a / Z ^ b$ with cap and positivity projection |
| `saturation_limiter` | _apply_saturation_constraints |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | capillary source port cochain $u_{\mathrm{cap}}$ |
| `effort_variable` | water_table_boundary_head [m] on the port; star_cap is the volumetric-to-mass conversion Hodge and is not a head-to-flux mobility |
| `state_advancer` | structural port incidence only; no local storage state is advanced; $u_{\mathrm{cap}} = \rho_{w} \cdot \max ( 0 , \min ( \mathrm{Ks} , B \cdot a / Z ^ b ) )$ |
| `runtime_usage` | _compute_iphs_capillary_rise_update evaluates the canonical capillary port flow and routes clipping residuals. |

### Composition channels and executable PortSpec API

Certificate composition channels: `capillary_groundwater_boundary_flux`

Executable `ports()` owner: `CapillaryRiseTransport`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `capillary_groundwater_boundary_flux` | `water_table_boundary_head [m]` | `capillary_rise_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.transport.max_capillary_height` | `m` | `2.0; bounds=(0.5, 10.0)` | `learnable` | Maximum height of capillary fringe (water table depth limit for significant capillary rise) | Source: gardner_eagleson_theory; Gardner, W.R. (1958) Soil Science 85(4):228-232; Eagleson, P.S. (1978) Water Resour. Res. 14(5):722-730 |
| `hydrosphere.transport.brooks_corey_pore_size_index` | `dimensionless` | `0.5; bounds=(0.1, 2.0)` | `learnable` | Brooks-Corey pore size distribution index (lambda) for capillary rise calculations | Source: brooks_corey_parameterization; Brooks, R.H. & Corey, A.T. (1964) Colorado State University Hydrology Paper 3; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604 |
| `hydrosphere.transport.air_entry_pressure` | `m` | `-0.102; bounds=(-1.02, -0.01)` | `learnable` | Brooks-Corey air entry pressure head (water column) for capillary rise calculations | Source: brooks_corey_parameterization; Brooks, R.H. & Corey, A.T. (1964) Colorado State University Hydrology Paper 3; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604 |
| `hydrosphere.groundwater.surface_porosity` | `dimensionless` | `0.45; bounds=(0.3, 0.7)` | `fixed` | Porosity at soil surface for exponential decay profile n(z) = n0 * exp(-z/p) | Source: RHESSys compute_delta_water.c + literature; Beven, K.J. & Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69; Wigmosta, M.S. et al. (1994). Water Resources Research, 30(6), 1665-1679 |
| `hydrosphere.groundwater.porosity_decay_length` | `m` | `2.0; bounds=(0.5, 10.0)` | `fixed` | Exponential decay length scale p in porosity profile n(z) = n0 * exp(-z/p) | Source: RHESSys implementation + calibration; Beven, K.J. & Kirkby, M.J. (1979). Hydrological Sciences Bulletin, 24(1), 43-69 |
| `hydrosphere.transport.capillary_saturation_threshold_factor` | `dimensionless` | `0.9; bounds=(0.5, 1.0)` | `fixed` | Fraction of surface porosity defining saturation threshold for capillary rise | Source: capillary_rise_parameterization; Gardner (1958) Soil Sci. 85:228-232; Eagleson (1978) Water Resour. Res. 14(5):722-730 |
| `hydrosphere.transport.capillary_saturation_bandwidth_factor` | `dimensionless` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Saturation bandwidth factor for capillary rise reduction near saturation | Source: numerical_implementation; Numerical stabilization for capillary rise saturation constraint |
| `hydrosphere.transport.hydraulic_parameterization_method` | `dimensionless` | `'brooks_corey'; bounds=None` | `fixed` | Hydraulic parameterization method (brooks_corey or van_genuchten) | Source: model_configuration; Brooks-Corey (1964) vs Van Genuchten (1980) hydraulic functions |
| `hydrosphere.soil_moisture.saturated_hydraulic_conductivity` | `m s-1` | `1e-05; bounds=(1e-08, 0.001)` | `learnable` | Saturated hydraulic conductivity for Richards equation | Source: field_measurements; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.alpha_vg` | `m-1` | `7.5; bounds=(0.1, 15.0)` | `learnable` | Van Genuchten alpha parameter (inverse air entry pressure) | Source: pedotransfer_functions; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Carsel & Parrish (1988) Water Resour. Res. 24:755-769 (class means as tabulated in van Genuchten et al. 1991, EPA/600/2-91/065, Table 4) |
| `hydrosphere.soil_moisture.n_vg` | `dimensionless` | `1.89; bounds=(1.1, 10.0)` | `learnable` | Van Genuchten n parameter (pore size distribution index) | Source: pedotransfer_functions; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Carsel & Parrish (1988) Water Resour. Res. 24:755-769 (class means as tabulated in van Genuchten et al. 1991, EPA/600/2-91/065, Table 4) |
| `hydrosphere.transport.enable_capillary_depth_profile` | `dimensionless` | `True; bounds=None` | `fixed` | Enable depth-dependent capillary rise flux profile calculation | Source: model_configuration; Eagleson, P.S. (1978) Water Resour. Res. 14(5):722-730 |
| `hydrosphere.transport.capillary_profile_layer_count` | `dimensionless` | `5; bounds=(1.0, 50.0)` | `fixed` | Default number of soil layers for capillary rise flux profile when not provided by caller | Source: hydrosphere_flux_defaults; Eagleson (1978) Water Resour. Res. 14(5):722-730; RHESSys compute_capillary_rise.c |
| `hydrosphere.transport.capillary_profile_layer_thickness` | `m` | `0.2; bounds=(0.01, 1.0)` | `fixed` | Default layer thickness for capillary rise profile when not provided by caller | Source: hydrosphere_flux_defaults; Eagleson (1978) Water Resour. Res. 14(5):722-730; RHESSys compute_capillary_rise.c |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | capillary_rise_flux and flat residual observations use UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} )$; residuals are diagnostics and never water updates |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_capillary_rise.md`.

### References

- Gardner, W.R. (1958). Soil Science, 85(4), 228-232 (FunctorMetadata)
- Eagleson, P.S. (1978). Water Resources Research, 14(5), 722-730 (FunctorMetadata)
- RHESSys compute_capillary_rise.c (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Gardner, W.R. (1958). Some steady-state solutions of the unsaturated moisture flow equation. Soil Science, 85(4), 228-232. (certificate)
- Eagleson, P.S. (1978). Climate, soil, and vegetation: A reduced-order model of soil moisture movement in the liquid phase. Water Resources Research, 14(5), 722-730. (certificate)
- Richards, L.A. (1931). Capillary conduction of liquids through porous mediums. Journal of Applied Physics, 1, 318-333. (certificate)
- Hirani, A.N. (2003). Discrete Exterior Calculus. (certificate)

(class-relic-scientific-hydrosphere-transport-vertical-percolation-verticalpercolationtransport)=
## vertical_percolation_transport

Vertical Percolation Transport Mechanism Functor

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.hydrosphere.transport.vertical_percolation.VerticalPercolationTransport` |
| Source | `relic\scientific\hydrosphere\transport\vertical_percolation.py:94` |
| Runtime name | `vertical_percolation_transport` |
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
vertical_percolation_transport(name: str = 'vertical_percolation_transport', config: Dict[str, Any] | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'vertical_percolation_transport'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | required storage cochain theta_h [m3 m-3], read only |
| `soil_temperature` | forcing | `K` | required thermal modifier state [K], read only |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `drainage_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | downward open-boundary matter flux q_h [kg m-2 s-1] |
| `diagnostics.vertical_percolation.effective_saturation` | output port | `1` | diagnostics | dimensionless observation |
| `diagnostics.vertical_percolation.hydraulic_conductivity` | output port | `m s-1` | diagnostics | K(theta,T) [m s-1] |
| `diagnostics.vertical_percolation.relative_hydraulic_conductivity` | output port | `1` | diagnostics | K/Ks [dimensionless] |
| `diagnostics.vertical_percolation.field_capacity_excess` | output port | `m3 m-3` | diagnostics | max(theta - theta_fc, 0) [m3 m-3] |
| `diagnostics.vertical_percolation.water_balance_residual` | output port | `kg m-2 s-1` | diagnostics | computed minus exported boundary flux [kg m-2 s-1] |

### Governing equations and canonical operator

**Literature or process form**

Darcy-Buckingham transport with van Genuchten-Mualem hydraulic conductivity; the field-capacity time-constant law is an explicitly reduced-order boundary closure, not an exact Richards solver.

**Canonical RELIC structural form**

$q_{h} = \star_{K} g_{h}$; $\star_{K} = \operatorname{diag} ( \rho_{w} \cdot \mathrm{Ks} ) \geq 0$; the empirical storage-depth rate is exported through the drainage boundary port.

| Entry | Meaning |
| --- | --- |
| `G_drain` | vertical_drainage_boundary_flux port incidence |
| `star_K` | build_hodge_contract and build_hodge_terms HodgeStarSpec |
| `hydraulic_conductivity` | _calculate_hydraulic_conductivity |
| `drainage_flux` | _calculate_drainage_flux |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | read-only soil_moisture cochain theta_h |
| `effort_variable` | downward hydraulic gradient $g_{h}$ with positive mobility star_K |
| `state_advancer` | none in this Functor; $q_{h}$ is exported through the bottom-drainage boundary port |
| `runtime_usage` | forward evaluates the canonical reduced-order boundary flux without advancing shared soil-water storage. |

### Composition channels and executable PortSpec API

Certificate composition channels: `vertical_drainage_boundary_flux`

Executable `ports()` owner: `VerticalPercolationTransport`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `vertical_drainage_boundary_flux` | `vertical_drainage_head [m]` | `drainage_flux [kg m-2 s-1]` | `kg m-1 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `hydrosphere.transport.vertical_percolation.enable_temperature_effects` | `dimensionless` | `True; bounds=None` | `fixed` | Apply temperature dependence to drainage viscosity corrections | Source: model_configuration; Liu & Dane (1995) Soil Sci. Soc. Am. J. 59:19-25 |
| `hydrosphere.transport.vertical_percolation.hydraulic_model` | `dimensionless` | `'van_genuchten_mualem'; bounds=None` | `fixed` | Primary hydraulic conductivity formulation used by VerticalPercolationTransport. | Source: vertical_percolation_specification; van Genuchten, M.T. (1980) Soil Sci. Soc. Am. J. 44:892-898; Brooks, R.H. & Corey, A.T. (1964) Hydrology Papers 3 |
| `hydrosphere.soil_moisture.porosity` | `m3 m-3` | `0.45; bounds=(0.3, 0.8)` | `fixed` | Total soil porosity (saturated water content) | Source: soil_survey_data; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.physical.viscosity_temperature_factor` | `K-1` | `0.02; bounds=(0.01, 0.03)` | `fixed` | Linearized sensitivity of liquid water viscosity to temperature for unsaturated flow corrections | Source: water_viscosity_literature; Korson, L. et al. (1969) Journal of Physical Chemistry 73:34-39; Decharme, B. et al. (2011) Journal of Hydrometeorology 12:153-172 |
| `hydrosphere.soil_moisture.saturated_hydraulic_conductivity` | `m s-1` | `1e-05; bounds=(1e-08, 0.001)` | `learnable` | Saturated hydraulic conductivity for Richards equation | Source: field_measurements; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.residual_water_content` | `m3 m-3` | `0.045; bounds=(0.0, 0.2)` | `fixed` | Residual water content (minimum water content at very high suction) | Source: laboratory_measurements; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Schaap et al. (2001) J. Hydrol. 251:163-176 |
| `hydrosphere.soil_moisture.field_capacity` | `m3 m-3` | `0.3; bounds=(0.1, 0.7)` | `learnable` | Soil field capacity (maximum water content at gravitational drainage threshold) | Source: literature_synthesis; Manabe (1969) Mon. Wea. Rev. 97:739-774; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.wilting_point` | `m3 m-3` | `0.1; bounds=(0.05, 0.35)` | `learnable` | Soil wilting point (minimum plant-available water content) | Source: literature_synthesis; Manabe (1969) Mon. Wea. Rev. 97:739-774; Cosby et al. (1984) Water Resour. Res. 20:682-690 |
| `hydrosphere.soil_moisture.alpha_vg` | `m-1` | `7.5; bounds=(0.1, 15.0)` | `learnable` | Van Genuchten alpha parameter (inverse air entry pressure) | Source: pedotransfer_functions; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Carsel & Parrish (1988) Water Resour. Res. 24:755-769 (class means as tabulated in van Genuchten et al. 1991, EPA/600/2-91/065, Table 4) |
| `hydrosphere.soil_moisture.n_vg` | `dimensionless` | `1.89; bounds=(1.1, 10.0)` | `learnable` | Van Genuchten n parameter (pore size distribution index) | Source: pedotransfer_functions; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Carsel & Parrish (1988) Water Resour. Res. 24:755-769 (class means as tabulated in van Genuchten et al. 1991, EPA/600/2-91/065, Table 4) |
| `hydrosphere.soil_moisture.root_zone_depth` | `m` | `1.0; bounds=(0.1, 5.0)` | `learnable` | Effective rooting depth for water extraction | Source: vegetation_database; Canadell et al. (1996) Oecologia 108:583-595; Jackson et al. (1996) Oecologia 108:389-411 |
| `hydrosphere.transport.vg_l` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `learnable` | Mualem tortuosity parameter (l) for van Genuchten model | Source: mualem_1976; Mualem, Y. (1976) Water Resour. Res. 12:513-522 |
| `hydrosphere.transport.bc_lambda` | `dimensionless` | `0.2; bounds=(0.1, 1.0)` | `learnable` | Brooks-Corey pore size distribution index (lambda) | Source: brooks_corey_1964; Brooks & Corey (1964) Hydrol. Paper 3, Colorado State Univ. |
| `hydrosphere.transport.ch_b_parameter` | `dimensionless` | `5.39; bounds=(2.0, 12.0)` | `learnable` | Clapp-Hornberger b parameter for loam soil | Source: clapp_hornberger_1978; Clapp & Hornberger (1978) Water Resour. Res. 14:601-604 |
| `hydrosphere.transport.drainage_time_constant` | `s` | `3600.0; bounds=(600.0, 86400.0)` | `learnable` | Characteristic drainage time scale (1 hour default) | Source: drainage_modeling; FAO-56: Allen et al. (1998) Crop evapotranspiration |
| `hydrosphere.transport.soil_layer_thickness` | `m` | `1.0; bounds=(0.1, 5.0)` | `fixed` | Effective soil layer thickness for percolation calculations | Source: model_configuration; Soil water dynamics literature |
| `hydrosphere.numerical.temperature_bounds` | `K` | `[273.15, 323.15]; bounds=([183.15, 273.15], [323.15, 373.15])` | `fixed` | Admitted near-surface air temperature for hydrosphere calculations [min, max]. The default keeps 273.15 K as minimum because the default sub-freezing phase of the evapotranspiration laws is liquid. A run with sub_freezing_phase ice may set the minimum down to 183.15 K (range widened on 2026-11-05 from 223.15 K for global daily runs, terrestrial record low -89.2 degC): the Tetens form over ice deviates from Goff-Gratch by -1.6 percent at 233.15 K, -4.7 percent at 213.15 K and -7.2 percent at 203.15 K, where the saturation vapour pressure is below 0.3 Pa. | Source: meteorological_standards; WMO Guide to Meteorological Instruments; WMO Archive of Weather and Climate Extremes |
| `hydrosphere.numerical.saturation_epsilon` | `dimensionless` | `1e-06; bounds=(1e-12, 0.001)` | `fixed` | Small epsilon for effective saturation bounds to avoid numerical issues | Source: soil_physics_modeling; van Genuchten, M.T. (1980) Soil Sci. Soc. Am. J. 44:892-898 |
| `hydrosphere.numerical.effective_saturation_bounds` | `dimensionless` | `[1e-06, 1.0]; bounds=([1e-12, 0.001], [0.99, 1.0])` | `fixed` | Bounds for effective saturation in van Genuchten and other soil models [min, max] | Source: soil_physics_modeling; van Genuchten (1980) Soil Sci. Soc. Am. J. 44:892-898; Brooks & Corey (1964) Hydrol. Paper 3, Colorado State Univ. |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | drainage_flux and diagnostics use UpdateItem(REPLACE, $\mathrm{per}_{\mathrm{timestep}} = \mathrm{False} )$; soil moisture is never written |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Hydrosphere theory](../theory/hydrosphere.md).

Certificate theory owner: `docs\source\theory\hydrosphere_vertical_percolation.md`.

### References

- van Genuchten, M.T. (1980). Closed-form equation for hydraulic conductivity. Soil Sci. Soc. Am. J., 44, 892-898 (FunctorMetadata)
- Mualem, Y. (1976). New model for hydraulic conductivity of unsaturated media. Water Resources Research, 12, 513-522 (FunctorMetadata)
- Darcy, H. (1856). Les Fontaines publiques de la ville de Dijon. Dalmont, Paris (FunctorMetadata)
- Buckingham, E. (1907). Studies on movement of soil moisture. USDA Bureau of Soils Bulletin 38 (FunctorMetadata)
- FAO-56: Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56 (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Darcy, H. (1856). Les Fontaines publiques de la ville de Dijon. (certificate)
- Buckingham, E. (1907). Studies on the movement of soil moisture. USDA Bureau of Soils Bulletin 38. (certificate)
- Richards, L.A. (1931). Capillary conduction of liquids through porous mediums. Physics, 1, 318-333. (certificate)
- van Genuchten, M.T. (1980). A closed-form equation for predicting hydraulic conductivity. Soil Sci. Soc. Am. J., 44, 892-898. (certificate)
- Mualem, Y. (1976). A new model for predicting hydraulic conductivity. Water Resources Research, 12, 513-522. (certificate)
