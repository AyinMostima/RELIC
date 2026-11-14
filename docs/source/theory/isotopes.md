# Stable-Isotope Fractionation and Ledger Theory

## 1. Process Definition and Scope

The RELIC isotope subsystem separates pure ParameterManager-backed fractionation calculators (`[I]`) from prognostic reservoir orchestration (`[N]`):

- `IsotopeHeartbeatFunctor` (`relic/scientific/isotopes/functor.py`) and `IsotopeTrackerService` (`relic/scientific/isotopes/tracker_service.py`) are `[N]` (`runtime_unbound`, IPHS Level 0). Both constructor and `forward(state, time, dt, **kwargs)` fail closed with `IsotopeHeartbeatError` or `IsotopeTrackingError` prior to `StateManager` access, callback registration, or NetCDF file creation.
- `IsotopeFractionationCalculator` (`relic/scientific/isotopes/fractionation/base.py`) and its concrete subclasses `Carbon13FractionationCalculator` (`carbon_13.py`), `Nitrogen15FractionationCalculator` (`nitrogen_15.py`), and `Oxygen18FractionationCalculator` (`oxygen_18.py`) are `[I]` stateless evaluators providing $^{13}\mathrm{C}$, $^{15}\mathrm{N}$, and $^{18}\mathrm{O}$ fractionation factors, exact binary heavy-isotope flux partitioning, and read-only $\delta$-notation observation transforms.

Multi-nuclide $^{17}\mathrm{O}$ triple-oxygen closure, CAM photosynthetic carbon discrimination, and governed `StateManager` nuclide component pool updates remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 23 (Section 23.12).

## 2. Governing Equations

### 2.1 Isotope Ratios, Fractionation Factors, and Exact Binary Flux Partitioning

For heavy and light nuclide abundances $n_H$ and $n_L$, the isotope ratio $R$, kinetic isotope effect $\mathrm{KIE}$, fractionation factor $\alpha$, enrichment factor $\varepsilon$ (permil), and read-only $\delta$ observation operator relative to international reference ratio $R_{\mathrm{std}}$ are:

$$
R = \frac{n_H}{n_L}, \qquad
\alpha = \frac{R_{\mathrm{product}}}{R_{\mathrm{reactant}}} = \mathrm{KIE}^{-1}, \qquad
\varepsilon = (\alpha - 1)\times 1000, \qquad
\delta = \left(\frac{R}{R_{\mathrm{std}}} - 1\right)\times 1000
$$

Given oriented source-to-target total element flux $F_T \ge 0$, source ratio $R_s \ge 0$, and process fractionation factor $\alpha > 0$, the transferred product ratio is $R_f = \alpha R_s$. `IsotopeFractionationCalculator.calculate_isotope_flux` evaluates exact binary-component partitioning (Coplen, 2011):

$$
F_H =
\begin{cases}
F_T\,\dfrac{\alpha R_s}{1 + \alpha R_s}, & \text{basis} = \text{"amount\_of\_substance"} \\[1.0ex]
F_T\,\dfrac{A_H \alpha R_s}{A_L + A_H \alpha R_s}, & \text{basis} = \text{"element\_mass"}
\end{cases},
\qquad F_L = F_T - F_H
$$

where $A_L$ and $A_H$ are positive dimensionless relative atomic masses (`light_relative_atomic_mass`, `heavy_relative_atomic_mass`). The legacy trace-abundance approximation $F_H \approx \alpha R_s F_T$ holds only for $\alpha R_s \ll 1$ and is superseded by the exact rational closure above.

### 2.2 Carbon-13 Fractionation Laws (`Carbon13FractionationCalculator`)

1. **C3 and C4 photosynthetic discrimination** (Farquhar, 1983; Farquhar et al., 1989): for caller-supplied finite $c_i / c_a \in [0, 1]$,

$$
\varepsilon_{\mathrm{C3}} = a + (b - a)\frac{c_i}{c_a}, \qquad
\varepsilon_{\mathrm{C4}} = a + (b_4 + b_3 \phi - a)\frac{c_i}{c_a}, \qquad
\alpha_{\mathrm{photo}} = \left(1 + \frac{\varepsilon_{\mathrm{photo}}}{1000}\right)^{-1}
$$

where $a = 4.4$ permil is stomatal diffusion discrimination, $b = 27.0$ permil is C3 Rubisco discrimination, $b_4 = -5.7$ permil is net $\mathrm{CO_2}$--$\mathrm{HCO_3^-}$--PEPC discrimination at $298.15\ \mathrm{K}$, $b_3 = 30.0$ permil is bundle-sheath Rubisco discrimination, and $\phi = 0.2$ is bundle-sheath leakiness (Henderson et al., 1992; Kromdijk et al., 2014). Values of $c_i / c_a$ outside $[0, 1]$ or requests for `"CAM"` raise `ValueError`.

2. **Respiration, decomposition, and BVOC emission**: respiration (`leaf_respiration`, `root_respiration`, `autotrophic_respiration`, `heterotrophic_respiration`) uses direct $Q_{10}$ power scaling with optional substrate signature $\delta^{13}\mathrm{C}_{\mathrm{sub}}$, whereas decomposition and BVOC emission apply multiplicative substrate and environmental modifiers:

$$
\alpha_{\mathrm{resp}}(T) = \alpha_0 Q_{10}^{(T - T_{\mathrm{ref}})/10}\bigl(1 + s_{\mathrm{sub}}\,\delta^{13}\mathrm{C}_{\mathrm{sub}}\bigr)
$$

$$
\alpha_{\mathrm{decomp}}(T) = \alpha_{0,d} Q_{10,d}^{(T - T_{\mathrm{ref}})/10} f_{\mathrm{sub}} f_{\theta} f_{\mathrm{pH}} f_{\mathrm{O_2}}, \qquad
\alpha_{\mathrm{bvoc}}(T) = \alpha_{0,c} Q_{10,b}^{(T - T_{\mathrm{ref},b})/10} f_{\mathrm{PAR}} f_{\mathrm{sub},b}
$$

### 2.3 Nitrogen-15 Fractionation Laws (`Nitrogen15FractionationCalculator`)

Microbial $^{15}\mathrm{N}$ transformations scale the isotopic deviation $(\alpha_0 - 1)$ around unity (Mariotti et al., 1981; Casciotti et al., 2003; Granger et al., 2008; Hobbie and Ouimette, 2009). Mineralization and denitrification apply inverse-temperature $Q_{10}$ deviation scaling (`q10_temperature_dependence`), whereas nitrification applies Arrhenius activation scaling (`arrhenius_temperature_dependence`):

$$
\alpha_{\min/\mathrm{denit}}(T) = 1 + (\alpha_0 - 1) Q_{10}^{(T_{\mathrm{ref}} - T)/10}, \qquad
\alpha_{\mathrm{nit}}(T) = 1 + (\alpha_{\mathrm{nit},0} - 1)\exp\!\left[\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{\mathrm{ref}}}\right)\right]
$$

Note the structural distinction between nitrogen deviation scaling $1 + (\alpha_0 - 1)Q_{10}^{(T_{\mathrm{ref}} - T)/10}$ and carbon direct power scaling $\alpha_0 Q_{10}^{(T - T_{\mathrm{ref}})/10}$.

### 2.4 Oxygen-18 Fractionation Laws (`Oxygen18FractionationCalculator`)

1. **Liquid-vapor equilibrium regression** (Horita and Wesolowski, 1994):

$$
\varepsilon_{\mathrm{eq}}(T) = 1000\ln\alpha_{l\text{-}v}(T) = A + \frac{B}{T} + \frac{C}{T^2} + \frac{D}{T^3}
$$

yielding $\varepsilon_{\mathrm{eq}}(298.15\ \mathrm{K}) = 9.3034$ permil.

2. **Kinetic discrimination and Craig-Gordon-Peclet transpiration** (Craig and Gordon, 1965; Farquhar and Lloyd, 1993): with relative humidity $h \in [h_{\min}, h_{\max}]$, kinetic factor $\varepsilon_k = \varepsilon_{k,0}(T/T_{\mathrm{ref}})^{n_k} f_{\mathrm{bl}}(u) f_{\mathrm{st}}(g_s)$, Peclet leaf-lamina mixing factor $\xi_P$, and conversion scale $s$:

$$
\xi_P =
\begin{cases}
\dfrac{1 - e^{-P}}{P}, & P \ge P_{\mathrm{thr}} \\[1.0ex]
1 - \dfrac{P}{d_{\mathrm{Taylor}}}, & P < P_{\mathrm{thr}}
\end{cases},
\qquad
\varepsilon_{\mathrm{CG}} = \bigl[\varepsilon_{\mathrm{eq}}(T) + \varepsilon_k(1 - h)\bigr]\xi_P, \qquad
\alpha_{\mathrm{trans}} = \left(1 + \frac{s\,\varepsilon_{\mathrm{CG}}}{1000}\right)^{-1}
$$

Declared `OxygenProcessType` modes comprise `transpiration`, `soil_evaporation`, `canopy_evaporation`, `root_uptake`, `leaf_water_exchange`, and `equilibrium_vapor`; ambiguous `"evaporation"` aliases raise `ValueError`.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $F_T$ | Non-negative source-to-target total element flux | caller flux units | `total_flux` | `flux` |
| $R_s, R_f$ | Source and transferred heavy-to-light nuclide ratios | $\mathrm{1}$ | `source_ratio`, `transferred_ratio` | `state` / `diagnostic` |
| $T$ | Process thermodynamic temperature | $\mathrm{K}$ | `temperature` | `forcing` |
| $c_i / c_a$ | Intercellular-to-ambient $\mathrm{CO_2}$ mole-fraction ratio | $\mathrm{1}$ | `ci_ca_ratio` (or `ci_co2`, `ca_co2`) | `state` |
| $h, u, g_s$ | Relative humidity, wind speed, and stomatal conductance | $\mathrm{1}$, $\mathrm{m\,s^{-1}}$, $\mathrm{mol\,m^{-2}\,s^{-1}}$ | `relative_humidity`, `wind_speed`, `stomatal_conductance` | `forcing` / `state` |
| $A_L, A_H$ | Light and heavy nuclide relative atomic masses | $\mathrm{1}$ | `light_relative_atomic_mass`, `heavy_relative_atomic_mass` | `parameter` |
| $\alpha$ | Process isotope fractionation factor | $\mathrm{1}$ | `fractionation_factor` | `diagnostic` |
| $F_H$ | Partitioned heavy-isotope component flux | caller flux units | `calculate_isotope_flux(...)` return | `flux` |
| $\delta$ | Read-only isotope ratio signature in permil | $\text{permil}$ ($\mathrm{10^{-3}}$) | `delta_notation(...)` return | `observation` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $R_{\mathrm{PDB}}, R_{\mathrm{AIR}}, R_{\mathrm{VSMOW}}$ | `isotopes.standards.carbon_13.pdb_ratio`, `nitrogen_15.air_ratio`, `oxygen_18.vsmow_ratio` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` ($0.0112372$, $0.0036765$, $0.0020052$) | Craig (1957); Mariotti (1983); Baertschi (1976) |
| $T_{\min}, T_{\max}, T_{\mathrm{ref}}$ | `isotopes.temperature.min`, `max`, `reference` | $\mathrm{K}$ | `isotopes/isotope_parameters.yaml` ($200.0$, $400.0$, $298.15$) | Calculator thermodynamic domain |
| $\alpha_{\min}$ | `isotopes.fractionation.alpha_min` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` ($0.9$) | Lower fractionation stability bound |
| $a, b$ | `isotopes.fractionation.carbon.diffusion_fractionation`, `carboxylation_fractionation` | $\mathrm{1}$ (permil) | `isotopes/isotope_parameters.yaml` ($4.4$, $27.0$) | Farquhar et al. (1989) |
| $b_4, b_3, \phi$ | `isotopes.fractionation.carbon.c4_pep_net_fractionation`, `c4_rubisco_fractionation`, `c4_bundle_sheath_leakiness` | $\mathrm{1}$ (permil, permil, $\mathrm{1}$) | `isotopes/isotope_parameters.yaml` ($-5.7$, $30.0$, $0.2$) | Farquhar (1983); Henderson et al. (1992); Kromdijk et al. (2014) |
| $\alpha_{0,\mathrm{resp}}, Q_{10,\mathrm{resp}}, s_{\mathrm{sub}}$ | `isotopes.fractionation.carbon.respiration.{leaf,root,autotrophic,heterotrophic}.{alpha_0,q10}`, `substrate_sensitivity` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` | Tcherkez et al. (2011) |
| $\alpha_{0,\min}, Q_{10,\min}$ | `isotopes.fractionation.nitrogen.mineralization.alpha_0`, `Q10` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` ($0.995$, $2.2$) | Hobbie and Ouimette (2009); Templer et al. (2007) |
| $\alpha_{\mathrm{nit},0}, E_a$ | `isotopes.fractionation.nitrogen.nitrification.alpha_0`, `E_a` | $\mathrm{1}$, $\mathrm{J\,mol^{-1}}$ | `isotopes/isotope_parameters.yaml` ($0.985$, $50000.0$) | Mariotti et al. (1981); Casciotti et al. (2003) |
| $\alpha_{0,\mathrm{denit}}, Q_{10,\mathrm{denit}}$ | `isotopes.fractionation.nitrogen.denitrification.alpha_0`, `Q10` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` ($0.980$, $1.8$) | Granger et al. (2008); Ostrom et al. (2007) |
| $A, B, C, D$ | `isotopes.fractionation.oxygen.equilibrium.horita_a`, `horita_b`, `horita_c`, `horita_d` | $\mathrm{1}$, $\mathrm{K}$, $\mathrm{K^2}$, $\mathrm{K^3}$ | `isotopes/isotope_parameters.yaml` ($-7.685$, $6712.3$, $-1.6664\times 10^6$, $3.5041\times 10^8$) | Horita and Wesolowski (1994) |
| $\varepsilon_{k,0}, n_k, P, s$ | `isotopes.fractionation.oxygen.kinetic_fractionation`, `kinetic.temperature_exponent`, `peclet_number`, `discrimination_scale` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` ($32.0$, $1.75$, $2.0$, $0.5$) | Craig and Gordon (1965); Farquhar and Lloyd (1993) |
| $R$ | `relic.physical.constants.universal.gas_constant` | $\mathrm{J\,mol^{-1}\,K^{-1}}$ | `physical/constants.yaml` ($8.314462618$) | CODATA universal gas constant |

## 5. Numerical Methods and Discretization

The `[I]` fractionation calculators operate directly on differentiable `torch.Tensor` inputs, promoting floating-point dtypes across `total_flux`, `source_ratio`, and `temperature` while preserving device placement. Arrhenius exponents are clamped against `isotopes.validation.numerical.arrhenius_exponent_min` ($-10.0$) and `arrhenius_exponent_max` ($10.0$) only via explicit validation bounds, and Peclet division near $P \to 0$ switches at `isotopes.fractionation.oxygen.peclet.threshold` ($0.1$) to the smooth first-order Taylor expansion $1 - P/d_{\mathrm{Taylor}}$. Non-finite tensors, negative `total_flux`, negative `source_ratio`, non-positive $\alpha$, or temperatures outside `[temperature.min, temperature.max]` raise `FractionationError` or `ValueError` immediately without silent clipping or zero-flux substitution.

`IsotopeHeartbeatFunctor` and `IsotopeTrackerService` execute no time discretization and emit no `UpdateItem` entries because construction and `forward` fail closed before runtime stepping.

## 6. Conservation, Boundary Conditions, and Ledgers

For any future governed two-nuclide internal transfer over step $\Delta t > 0$, oriented stoichiometry $N_h = [-1,\; +1]^{\mathsf{T}}$ applied to component transfer vector $\Delta m = [F_T \Delta t,\; F_H \Delta t]^{\mathsf{T}}$ preserves both total binary element mass ($F_L + F_H = F_T$) and heavy-isotope mass across source and target pools, while atmospheric reservoirs act as open boundaries with ParameterManager-backed reference ratios. Read-only $\delta$ values are observation diagnostics and never replace prognostic nuclide masses. Binary $^{18}\mathrm{O}/^{16}\mathrm{O}$ partitioning closes the two-isotope subsystem but omits $^{17}\mathrm{O}$ mass. Under the current `[N]` runtime boundary, `IsotopeHeartbeatFunctor` owns no `StateManager` states and claims no active runtime conservation ledger.

## 7. Code Mapping

| Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `IsotopeHeartbeatFunctor` | `relic/scientific/isotopes/functor.py` | `ScaleCoupling` (`[N]` fail-closed boundary) | `ScaleCouplingFunctor` (`relic/core/functor/scale_coupling_bridge.py`) | Unbound (`[N]`, raises `IsotopeHeartbeatError`) |
| `IsotopeTrackerService` | `relic/scientific/isotopes/tracker_service.py` | Service boundary (`[N]` fail-closed) | None | Unbound (`[N]`, raises `IsotopeTrackingError`) |
| `IsotopeFractionationCalculator` | `relic/scientific/isotopes/fractionation/base.py` | Stateless fractionation base (`abc.ABC`) | None (pure library calculator) | Caller-invoked (`[I]`) |
| `Carbon13FractionationCalculator` | `relic/scientific/isotopes/fractionation/carbon_13.py` | $^{13}\mathrm{C}$ fractionation calculator | `IsotopeFractionationCalculator` | Caller-invoked (`[I]`) |
| `Nitrogen15FractionationCalculator` | `relic/scientific/isotopes/fractionation/nitrogen_15.py` | $^{15}\mathrm{N}$ fractionation calculator | `IsotopeFractionationCalculator` | Caller-invoked (`[I]`) |
| `Oxygen18FractionationCalculator` | `relic/scientific/isotopes/fractionation/oxygen_18.py` | $^{18}\mathrm{O}$ fractionation calculator | `IsotopeFractionationCalculator` | Caller-invoked (`[I]`) |

## 8. Implementation Status

- `IsotopeHeartbeatFunctor`: `[N]`, IPHS **Level 0** (`coupling`, `runtime_unbound`, `descriptive_only`, certificate `isotopes_functor_isotope_heartbeat_functor.yaml`). Construction and `forward` fail closed before `FunctorBase` metadata registration or `StateManager` interaction.
- `IsotopeTrackerService`: `[N]`, fail-closed service stub raising `IsotopeTrackingError`.
- `Carbon13FractionationCalculator`, `Nitrogen15FractionationCalculator`, `Oxygen18FractionationCalculator`, and `IsotopeFractionationCalculator.calculate_isotope_flux`: `[I]` pure `torch.Tensor` library calculators backed by `isotopes/isotope_parameters.yaml` and `physical/constants.yaml`.

Frozen certificate expressions (`isotopes_functor_isotope_heartbeat_functor.yaml`):

```text
Historical heartbeat and service paths coordinated a private diagnostic ledger but did not provide governed StateManager component updates.
[N] runtime_unbound; no executable scientific expression is exposed.
```

## 9. References

- Baertschi, P. (1976). Absolute $^{18}\mathrm{O}$ content of Standard Mean Ocean Water. *Earth and Planetary Science Letters*, 31(3), 341-344.
- Casciotti, K. L., Sigman, D. M., and Ward, B. B. (2003). Linking diversity and stable isotope fractionation in ammonia-oxidizing bacteria. *Geomicrobiology Journal*, 20(4), 335-353.
- Coplen, T. B. (2011). Guidelines and recommended terms for expression of stable-isotope-ratio and gas-ratio measurement results. *Pure and Applied Chemistry*, 83(10), 1873-1935.
- Craig, H. (1957). Isotopic standards for carbon and oxygen and correction factors for mass-spectrometric analysis of carbon dioxide. *Geochimica et Cosmochimica Acta*, 12(1-2), 133-149.
- Craig, H., and Gordon, L. I. (1965). Deuterium and oxygen 18 variations in the ocean and the marine atmosphere. In E. Tongiorgi (Ed.), *Stable Isotopes in Oceanographic Studies and Paleotemperatures* (pp. 9-130). Consiglio Nazionale delle Ricerche.
- Farquhar, G. D. (1983). On the nature of carbon isotope discrimination in C4 species. *Australian Journal of Plant Physiology*, 10(2), 205-226.
- Farquhar, G. D., Ehleringer, J. R., and Hubick, K. T. (1989). Carbon isotope discrimination and photosynthesis. *Annual Review of Plant Physiology and Plant Molecular Biology*, 40, 503-537.
- Farquhar, G. D., and Lloyd, J. (1993). Carbon and oxygen isotope effects in the exchange of carbon dioxide between terrestrial plants and the atmosphere. In J. R. Ehleringer, A. E. Hall, and G. D. Farquhar (Eds.), *Stable Isotopes and Plant Carbon-Water Relations* (pp. 47-70). Academic Press.
- Granger, J., Sigman, D. M., Lehmann, M. F., and Tortell, P. D. (2008). Nitrogen and oxygen isotope fractionation during dissimilatory nitrate reduction by denitrifying bacteria. *Limnology and Oceanography*, 53(6), 2533-2545.
- Henderson, S. A., von Caemmerer, S., and Farquhar, G. D. (1992). Short-term measurements of carbon isotope discrimination in several C4 species. *Australian Journal of Plant Physiology*, 19(3), 263-285.
- Hobbie, E. A., and Ouimette, A. P. (2009). Controls of nitrogen isotope patterns in soil profiles. *Biogeochemistry*, 95(2-3), 355-371.
- Horita, J., and Wesolowski, D. J. (1994). Liquid-vapor fractionation of oxygen and hydrogen isotopes of water from the freezing to the critical temperature. *Geochimica et Cosmochimica Acta*, 58(16), 3425-3437.
- Kromdijk, J., Ubierna, N., Cousins, A. B., and Griffiths, H. (2014). Bundle-sheath leakiness in C4 photosynthesis: a careful balancing act between $\mathrm{CO_2}$ concentration and assimilation. *Journal of Experimental Botany*, 65(13), 3443-3457.
- Mariotti, A., Germon, J. C., Hubert, P., et al. (1981). Experimental determination of nitrogen kinetic isotope fractionation: some principles; illustration for the denitrification and nitrification processes. *Plant and Soil*, 62(3), 413-430.
- Mariotti, A. (1983). Atmospheric nitrogen is a reliable standard for natural $^{15}\mathrm{N}$ abundance measurements. *Nature*, 303, 685-687.
- Ostrom, N. E., Pitt, A., Sutka, R., et al. (2007). Isotopologue effects during $\mathrm{N_2O}$ reduction in soils and in pure cultures of denitrifiers. *Journal of Geophysical Research: Biogeosciences*, 112(G2), G02005.
- Tcherkez, G., Mahé, A., and Hodges, M. (2011). $^{12}\mathrm{C}/^{13}\mathrm{C}$ fractionations in plant primary metabolism. *Trends in Plant Science*, 16(9), 499-506.
- Templer, P. H., Arthur, M. A., Lovett, G. M., and Weathers, K. C. (2007). Plant and soil natural abundance $\delta^{15}\mathrm{N}$: indicators of relative rates of nitrogen cycling in temperate forest ecosystems. *Oecologia*, 153(2), 399-406.
