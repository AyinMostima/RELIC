# Wetland Methane Process Chain

## 1. Process Definition and Scope

The RELIC pedosphere wetland methane chain comprises five certified Functors operating on a unified elemental carbon basis ($\mathrm{g\,C\,m^{-2}}$ for storage and $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ for fluxes):

1. `MethaneScaleCouplingFunctor` (`relic/scientific/pedosphere/methane/scales.py`, `[H]`): a phenomenological `ScaleCouplingFunctor` (`ExecutionPhase.COUPLING`) that maps wetland fraction, water-filled pore space, and a data-availability score into a bounded routing weight $w \in [0, 1]$.
2. `MethaneProductionFunctor` (`relic/scientific/pedosphere/methane/production.py`, `[H]`): a `BiologicalProcessFunctor` (`ExecutionPhase.MAIN_PROCESSES`) blending the iMAPLE heterotrophic-respiration methanogenesis formulation (Yue et al., 2024; Zhu et al., 2014) with a RELIC calibrated $R_h$-based redox-gated branch.
3. `MethaneOxidationFunctor` (`relic/scientific/pedosphere/methane/oxidation.py`, `[I]`): a `BiologicalProcessFunctor` (`ExecutionPhase.MAIN_PROCESSES`) implementing CLM4Me double Michaelis-Menten aerobic methanotrophy in saturated wetland soil (Riley et al., 2011; Oleson et al., 2013) with local step-availability limiting.
4. `MethaneTransportFunctor` (`relic/scientific/pedosphere/methane/transport.py`, `[H]`): a `TransportMechanismFunctor` (`ExecutionPhase.MAIN_PROCESSES`) evaluating raw requested methane transport across aqueous/gaseous diffusion, bubble ebullition, and plant aerenchyma pathways (Walter and Heimann, 2000; Zhuang et al., 2004; Chanton et al., 1992).
5. `MethaneBudgetCoupler` (`relic/scientific/pedosphere/methane/budget_coupler.py`, `[I]`/`[H]`): a `ScaleCouplingFunctor` (`ExecutionPhase.COUPLING`) that serves as the single authorized writer of `ch4_soil_storage`, enforcing non-negative storage through a shared transport-availability limiter (`[H]` operator-splitting priority) and emitting realized atmospheric $\mathrm{CH_4}\text{-}\mathrm{C}$ and $\mathrm{CH_4}$-mass fluxes (`[I]`).

DNDC substrate-kinetic acetate and $\mathrm{H_2}/\mathrm{CO_2}$ methanogenesis, unsaturated-zone matric-potential oxidation modifiers above the water table, and soil-texture-explicit iMAPLE Eq. 42 gas diffusivity are `[N]`.

## 2. Governing Equations

### 2.1 Phenomenological Pathway Gating (`MethaneScaleCouplingFunctor`, `[H]`)

Given wetland inundation fraction $f_{\mathrm{wet}} \in [0, 1]$, water-filled pore space $\mathrm{WFPS} \in [0, 1]$, and data-availability score $s_{\mathrm{data}} \in [0, 1]$, the unclipped score $s$ and calibrated-branch weight $w$ are:

$$
s = \alpha_{\mathrm{wet}} f_{\mathrm{wet}} + \alpha_{\mathrm{wfps}} \mathrm{WFPS} + \alpha_{\mathrm{data}} s_{\mathrm{data}}, \qquad w = \operatorname{clip}\!\left(s - \tau_{\mathrm{gate}},\, 0,\, 1\right)
$$

### 2.2 Methanogenesis (`MethaneProductionFunctor`, `[H]`)

The iMAPLE candidate rate $P_{\mathrm{iMAPLE}}$ (Yue et al., 2024, Eq. 34) and the RELIC calibrated redox-gated candidate rate $P_{\mathrm{cal}}$ share heterotrophic respiration $R_h \ge 0$ ($\mathrm{g\,C\,m^{-2}\,s^{-1}}$), $\mathrm{CH_4}\text{:}\mathrm{CO_2}$ release ratio $r$, and temperature peak response $f_T(T_s)$:

$$
P_{\mathrm{iMAPLE}} = R_h\, r\, f_T(T_s)\, f_{\mathrm{pH}}(\mathrm{pH})\, f_{\mathrm{wet}}, \qquad P_{\mathrm{cal}} = R_h\, r\, f_T(T_s)\, f_{\mathrm{Eh}}(\mathrm{Eh})\, A
$$

$$
P = \max\!\left((1 - w) P_{\mathrm{iMAPLE}} + w P_{\mathrm{cal}},\, 0\right)
$$

Temperature-dependent $Q_{10}(T_s)$ follows Yue et al. (2024, Eq. 39) with explicit clipping to $[Q_{10,\min}, Q_{10,\max}]$ ($Q_{10,\min} > 1$, Zhu et al., 2014, Table 2) and SSOT interval $\Delta T_{10} = 10\,\mathrm{K}$ (`relic.physical.constants.temperature.q10_interval`):

$$
Q_{10}(T_s) = \operatorname{clip}\!\left(r_b Q_b^{(T_s - T_{q,\mathrm{base}}) / \Delta T_{10}},\, Q_{10,\min},\, Q_{10,\max}\right)
$$

With span $\Delta T_{\mathrm{span}} = \max(T_{\max} - T_{\mathrm{opt}}, \varepsilon_T)$ and $L_T = \ln(Q_{10}(T_s)) \Delta T_{\mathrm{span}}$, RELIC evaluates the published Zhu et al. (2014, Eq. 5) exponent $x_t$ (avoiding the Fortran operator-precedence artifact $\ln^2(Q_{10}) + 2.05$ present in released iMAPLE v1 `ch4flux_wetland.f`):

$$
a_t = 1 + \frac{4 \Delta T_{10}}{L_T}, \qquad x_t = \frac{L_T^2 \left(1 + \sqrt{a_t}\right)^2}{\left(2 \Delta T_{10}\right)^2}, \qquad v_t = \frac{T_{\max} - T_s}{\Delta T_{\mathrm{span}}}
$$

$$
f_T(T_s) = \begin{cases} v_t^{x_t} \exp\!\left(x_t (1 - v_t)\right), & T_{\min} \le T_s \le T_{\max} \\ 0, & T_s < T_{\min}\ \text{or}\ T_s > T_{\max} \end{cases}
$$

The pH modifier (Zhu et al., 2014, Eq. 6) with $L_{\mathrm{span}} = \max(\mathrm{pH}_{\mathrm{opt}} - \mathrm{pH}_{\mathrm{low}}, \varepsilon_d)$ and $U_{\mathrm{span}} = \max(\mathrm{pH}_{\mathrm{high}} - \mathrm{pH}_{\mathrm{opt}}, \varepsilon_d)$, and the piecewise-linear redox gate $f_{\mathrm{Eh}}$ (Zhang et al., 2002; Zhu et al., 2014, Sect. 2.2.1; $\mathrm{Eh}_1 < \mathrm{Eh}_0$) are:

$$
f_{\mathrm{pH}}(\mathrm{pH}) = \begin{cases} \left(\dfrac{\mathrm{pH} - \mathrm{pH}_{\mathrm{low}}}{L_{\mathrm{span}}}\right) \left(\dfrac{\mathrm{pH}_{\mathrm{high}} - \mathrm{pH}}{U_{\mathrm{span}}}\right)^{U_{\mathrm{span}} / L_{\mathrm{span}}}, & \mathrm{pH}_{\mathrm{low}} \le \mathrm{pH} \le \mathrm{pH}_{\mathrm{high}} \\ 0, & \text{otherwise} \end{cases}
$$

$$
f_{\mathrm{Eh}}(\mathrm{Eh}) = \operatorname{clip}\!\left(\frac{\mathrm{Eh}_0 - \mathrm{Eh}}{\mathrm{Eh}_0 - \mathrm{Eh}_1},\, 0,\, 1\right)
$$

Because $f_T, f_{\mathrm{pH}}, f_{\mathrm{wet}}, f_{\mathrm{Eh}}, A \in [0, 1]$ and $r \le 1$, both branches satisfy $0 \le P \le r R_h \le R_h$.

### 2.3 Aerobic Methanotrophy (`MethaneOxidationFunctor`, `[I]`)

In saturated wetland soil ($F = 1$), aqueous methane concentration $C_{\mathrm{CH_4}}$ ($\mathrm{mol\,m^{-3}}$ water volume) is derived from areal storage $M \ge 0$ ($\mathrm{g\,C\,m^{-2}}$), mixing depth $z \ge \varepsilon_L$ ($\mathrm{m}$), porosity $\phi \in (0, 1]$, $\mathrm{WFPS} \in (0, 1]$, and carbon molar mass $M_C$ (`relic.physical.constants.molecular_weights.carbon`; Riley et al., 2011, Eq. 7; Oleson et al., 2013, Eqs. 19.1 and 19.11):

$$
C_{\mathrm{CH_4}} = \frac{M}{M_C\, z\, \phi\, \mathrm{WFPS}}, \qquad R_{\mathrm{ox}} = R_{o,\max} \frac{C_{\mathrm{CH_4}}}{K_{\mathrm{CH_4}} + C_{\mathrm{CH_4}}} \frac{C_{\mathrm{O_2}}}{K_{\mathrm{O_2}} + C_{\mathrm{O_2}}} Q_{10,\mathrm{ox}}^{(T_s - T_{\mathrm{ox,base}}) / \Delta T_{10}}
$$

Converting volumetric reaction demand $R_{\mathrm{ox}}$ ($\mathrm{mol\,m^{-3}\,s^{-1}}$ bulk soil) to areal carbon demand $O_{\mathrm{kin}} = R_{\mathrm{ox}} z M_C$ ($\mathrm{g\,C\,m^{-2}\,s^{-1}}$) and bounding by step availability $M / \Delta t$:

$$
O = \min\!\left(O_{\mathrm{kin}},\, \frac{M}{\Delta t}\right), \qquad R_{\mathrm{lim,ox}} = O_{\mathrm{kin}} - O \ge 0
$$

### 2.4 Three-Pathway Methane Transport Request (`MethaneTransportFunctor`, `[H]`)

With bulk concentration $C_{\mathrm{soil}} = M / z$ ($\mathrm{g\,C\,m^{-3}}$), molar concentration $C_{\mathrm{mol}} = C_{\mathrm{soil}} / M_C$ ($\mathrm{mol\,m^{-3}}$), and air-filled porosity $\theta_{\mathrm{air}} = \phi (1 - \mathrm{WFPS})$, effective diffusivity and the three pathway fluxes (Walter and Heimann, 2000; Zhuang et al., 2004; Chanton et al., 1992; Yue et al., 2024) are:

$$
D_{\mathrm{eff}} = \tau D_{\mathrm{air}} \frac{\theta_{\mathrm{air}}^{a_{\mathrm{MQ}}}}{\phi^{b_{\mathrm{MQ}}}} + D_{\mathrm{water}} \mathrm{WFPS}, \qquad F_{\mathrm{diff}} = -D_{\mathrm{eff}} \frac{C_{\mathrm{atm}} - C_{\mathrm{soil}}}{z}
$$

$$
F_{\mathrm{ebb}} = \frac{\max\!\left(C_{\mathrm{mol}} - C_{\mathrm{thr}},\, 0\right) M_C z}{\tau_{\mathrm{eb}}}, \qquad F_{\mathrm{plant}} = k_{\mathrm{plant}} \frac{C_{\mathrm{soil}}}{z} f_{\mathrm{aer}} \left(1 - f_{\mathrm{root,ox}}\right)
$$

$$
F_{\mathrm{req}} = F_{\mathrm{diff}} + F_{\mathrm{ebb}} + F_{\mathrm{plant}}, \qquad T_{\mathrm{trans}} = -F_{\mathrm{req}}
$$

`MethaneTransportFunctor` emits the unclipped request $F_{\mathrm{req}}$ and opposite tendency $T_{\mathrm{trans}}$, delegating storage limiting to `MethaneBudgetCoupler`.

### 2.5 Conserved Methane Budget Coupler (`MethaneBudgetCoupler`, `[I]`/`[H]`)

`MethaneBudgetCoupler` combines the three process tendencies $(T_{\mathrm{prod}}, T_{\mathrm{oxid}}, T_{\mathrm{trans}}) = (+P, -O, -F_{\mathrm{req}})$ and enforces $M + \Delta M \ge 0$ in one step (Yue et al., 2024, Eq. 33):

$$
\Delta M_{\mathrm{raw}} = \Delta t \left(T_{\mathrm{prod}} + T_{\mathrm{oxid}} + T_{\mathrm{trans}}\right), \qquad \Delta M = \max\!\left(\Delta M_{\mathrm{raw}},\, -M\right), \qquad R_{\mathrm{lim,tr}} = \frac{\Delta M - \Delta M_{\mathrm{raw}}}{\Delta t} \ge 0
$$

$$
F_{\mathrm{atm}} = F_{\mathrm{req}} - R_{\mathrm{lim,tr}}, \qquad F_{\mathrm{atm,CH_4}} = F_{\mathrm{atm}} \frac{M_{\mathrm{CH_4}}}{M_C}, \qquad \mathcal{R}_{\mathrm{CH_4}} = P - O - F_{\mathrm{atm}} - \frac{\Delta M}{\Delta t}
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $f_{\mathrm{wet}}$ | Wetland inundation fraction | $\mathrm{1}$ | `land_surface.wetland_fraction` | Scale & production input |
| $\mathrm{WFPS}$ | Water-filled pore-space fraction | $\mathrm{1}$ | `water_filled_pore_space` | Scale, oxidation, & transport input |
| $s_{\mathrm{data}}$ | Methane data-availability score | $\mathrm{1}$ | `methane_data_score` | Scale required input |
| $w, s$ | Calibrated-branch routing weight and raw score | $\mathrm{1}$ | `methane_gating_weight`, `methane_gating_score` | Scale outputs (`REPLACE`) |
| $T_s, \mathrm{pH}, \mathrm{Eh}$ | Soil temperature, pH, and redox potential | $\mathrm{K}$, $\mathrm{1}$, $\mathrm{mV}$ | `soil_temperature`, `soil_ph`, `redox_potential` | Production / oxidation inputs |
| $R_h, A$ | Heterotrophic respiration and microbial activity index | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$, $\mathrm{1}$ | `heterotrophic_respiration`, `microbial_activity_index` | Production required inputs |
| $P, T_{\mathrm{prod}}$ | Realized $\mathrm{CH_4}\text{-}\mathrm{C}$ production port and storage tendency | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `ch4_production_rate`, `ch4_storage_tendency_prod` | Production outputs (`REPLACE`) |
| $P_{\mathrm{iMAPLE}}, P_{\mathrm{cal}}$ | Candidate iMAPLE and calibrated production rates | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `diagnostics.methane_production.imaple_rate`, `diagnostics.methane_production.calibrated_rate` | Production diagnostics (`REPLACE`) |
| $M$ | Areal soil methane-carbon storage | $\mathrm{g\,C\,m^{-2}}$ | `ch4_soil_storage` | Oxidation/transport/coupler input; coupler `ACCUMULATE` output (`per_timestep=True`) |
| $C_{\mathrm{O_2}}, \phi$ | Aqueous $\mathrm{O_2}$ concentration and total porosity | $\mathrm{mol\,m^{-3}}$, $\mathrm{m^3\,m^{-3}}$ | `soil_o2_concentration`, `porosity` | Oxidation / transport inputs |
| $O, T_{\mathrm{oxid}}$ | Realized $\mathrm{CH_4}\text{-}\mathrm{C}$ oxidation rate and negative tendency | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `ch4_oxidation_rate`, `ch4_storage_tendency_oxid` | Oxidation outputs (`REPLACE`) |
| $O_{\mathrm{kin}}, R_{\mathrm{lim,ox}}$ | Unclipped CLM4Me kinetic demand and unmet demand residual | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `diagnostics.methane_oxidation.kinetic_rate`, `diagnostics.methane_oxidation.storage_limiter_residual` | Oxidation diagnostics (`REPLACE`) |
| $C_{\mathrm{atm}}$ | Atmospheric $\mathrm{CH_4}\text{-}\mathrm{C}$ boundary concentration | $\mathrm{g\,C\,m^{-3}}$ | `atmospheric_ch4_concentration` | Transport required input |
| $F_{\mathrm{req}}, T_{\mathrm{trans}}$ | Requested $\mathrm{CH_4}\text{-}\mathrm{C}$ transport rate and opposite tendency | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `ch4_transport_loss_rate`, `ch4_storage_tendency_trans` | Transport outputs (`REPLACE`) |
| $F_{\mathrm{diff}}, F_{\mathrm{ebb}}, F_{\mathrm{plant}}$ | Diffusion, ebullition, and plant transport pathway rates | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `diagnostics.methane_transport.diffusion_rate`, `diagnostics.methane_transport.ebullition_rate`, `diagnostics.methane_transport.plant_flux_rate` | Transport diagnostics (`REPLACE`) |
| $F_{\mathrm{atm}}$ | Realized atmospheric $\mathrm{CH_4}\text{-}\mathrm{C}$ emission flux | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `ch4_atmospheric_flux` | Budget coupler output (`REPLACE`) |
| $F_{\mathrm{atm,CH_4}}, \mathcal{R}_{\mathrm{CH_4}}, \dot{M}_{\mathrm{net}}, R_{\mathrm{lim,tr}}$ | $\mathrm{CH_4}$ mass flux, closure residual, net storage tendency, and denied transport residual | $\mathrm{g\,CH_4\,m^{-2}\,s^{-1}}$ / $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `diagnostics.methane_budget.atmospheric_ch4_mass_flux`, `diagnostics.methane_budget.mass_balance_residual`, `diagnostics.methane_budget.storage_tendency_net`, `diagnostics.methane_budget.transport_limiter_residual` | Budget coupler diagnostics (`REPLACE`) |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $\alpha_{\mathrm{wet}}, \alpha_{\mathrm{wfps}}, \alpha_{\mathrm{data}}, \tau_{\mathrm{gate}}$ | `pedosphere.methane.scales.gating.alpha_wetland`, `pedosphere.methane.scales.gating.alpha_wfps`, `pedosphere.methane.scales.gating.alpha_data_score`, `pedosphere.methane.scales.gating.threshold` | $\mathrm{1}$ | `pedosphere/methane.yaml` ($0.5$, $0.3$, $0.2$, $0.5$) | RELIC gating calibration |
| $r, r_b, Q_b, T_{q,\mathrm{base}}$ | `pedosphere.methane.production.release_ratio`, `pedosphere.methane.production.q10`, `pedosphere.methane.production.q10_temperature_sensitivity`, `pedosphere.methane.production.q10_base_temperature` | $\mathrm{1}$, $\mathrm{1}$, $\mathrm{1}$, $\mathrm{K}$ | `pedosphere/methane.yaml` ($0.2$, $3.0$, $1.33$, $298.15$) | Yue et al. (2024); Wania et al. (2010) |
| $Q_{10,\min}, Q_{10,\max}$ | `pedosphere.methane.production.q10_min`, `pedosphere.methane.production.q10_max` | $\mathrm{1}$ | `pedosphere/methane.yaml` ($1.7$, $16.0$) | Zhu et al. (2014) |
| $T_{\min}, T_{\mathrm{opt}}, T_{\max}$ | `pedosphere.methane.production.temperature_min`, `pedosphere.methane.production.temperature_optimum`, `pedosphere.methane.production.temperature_max` | $\mathrm{K}$ | `pedosphere/methane.yaml` ($273.15$, $298.15$, $318.15$) | Yue et al. (2024); Zhu et al. (2014) |
| $\mathrm{pH}_{\mathrm{low}}, \mathrm{pH}_{\mathrm{opt}}, \mathrm{pH}_{\mathrm{high}}$ | `pedosphere.methane.production.ph_lower_limit`, `pedosphere.methane.production.ph_optimum`, `pedosphere.methane.production.ph_upper_limit` | $\mathrm{1}$ | `pedosphere/methane.yaml` ($4.0$, $7.0$, $9.0$) | Zhu et al. (2014) |
| $\mathrm{Eh}_1, \mathrm{Eh}_0$ | `pedosphere.methane.production.calibrated.eh_full_activity`, `pedosphere.methane.production.calibrated.eh_zero_activity` | $\mathrm{mV}$ | `pedosphere/methane.yaml` ($-200.0$, $-100.0$) | Zhang et al. (2002); Zhu et al. (2014) |
| $R_{o,\max}, K_{\mathrm{CH_4}}, K_{\mathrm{O_2}}$ | `pedosphere.methane.oxidation.maximum_rate`, `pedosphere.methane.oxidation.half_saturation_ch4`, `pedosphere.methane.oxidation.half_saturation_o2` | $\mathrm{mol\,m^{-3}\,s^{-1}}$, $\mathrm{mol\,m^{-3}}$ | `pedosphere/methane.yaml` ($1.25 \times 10^{-5}$, $5 \times 10^{-3}$, $2 \times 10^{-2}$) | Riley et al. (2011); Oleson et al. (2013) |
| $Q_{10,\mathrm{ox}}, T_{\mathrm{ox,base}}$ | `pedosphere.methane.oxidation.q10`, `pedosphere.methane.oxidation.temperature_base` | $\mathrm{1}$, $\mathrm{K}$ | `pedosphere/methane.yaml` ($1.9$, $285.15$) | Riley et al. (2011); Oleson et al. (2013) |
| $D_{\mathrm{air}}, D_{\mathrm{water}}, \tau, z$ | `pedosphere.methane.transport.diffusivity_air`, `pedosphere.methane.transport.diffusivity_water`, `pedosphere.methane.transport.tortuosity_factor`, `pedosphere.methane.transport.layer_mixing_depth` | $\mathrm{m^2\,s^{-1}}$, $\mathrm{m^2\,s^{-1}}$, $\mathrm{1}$, $\mathrm{m}$ | `pedosphere/methane.yaml` ($1.8 \times 10^{-5}$, $2 \times 10^{-9}$, $0.66$, $0.05$) | Zhuang et al. (2004); Yue et al. (2024) |
| $C_{\mathrm{thr}}, \tau_{\mathrm{eb}}$ | `pedosphere.methane.transport.bubble_threshold_concentration`, `pedosphere.methane.transport.ebullition_timescale` | $\mathrm{mol\,m^{-3}}$, $\mathrm{s}$ | `pedosphere/methane.yaml` ($0.5$, $86400.0$) | Walter and Heimann (2000) |
| $k_{\mathrm{plant}}, f_{\mathrm{aer}}, f_{\mathrm{root,ox}}$ | `pedosphere.methane.transport.plant_conductivity`, `pedosphere.methane.transport.aerenchyma_fraction`, `pedosphere.methane.transport.root_oxidation_fraction` | $\mathrm{m^2\,s^{-1}}$, $\mathrm{1}$, $\mathrm{1}$ | `pedosphere/methane.yaml` ($10^{-7}$, $0.3$, $0.5$) | Chanton et al. (1992) |
| $a_{\mathrm{MQ}}, b_{\mathrm{MQ}}$ | `pedosphere.physical.millington_quirk_exponent_numerator`, `pedosphere.physical.millington_quirk_exponent_denominator` | $\mathrm{1}$ | `pedosphere/physical.yaml` ($3.333333$, $2.0$) | Millington and Quirk (1961) |
| $\varepsilon_{\mathrm{rel}}, \varepsilon_{\mathrm{abs}}$ | `pedosphere.methane.conservation.relative_tolerance`, `pedosphere.methane.conservation.absolute_tolerance` | $\mathrm{1}$, $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `pedosphere/methane.yaml` ($10^{-8}$, $10^{-7}$) | Budget closure audit |
| $\Delta t_{\min}, \varepsilon_L, \tau_{\min}, \varepsilon_T, \varepsilon_d$ | `pedosphere.numerical.minimum_timestep`, `pedosphere.numerical.epsilon_length`, `pedosphere.numerical.minimum_timescale`, `pedosphere.numerical.minimum_temperature_span`, `pedosphere.numerical.minimum_dimensionless` | $\mathrm{s}$, $\mathrm{m}$, $\mathrm{s}$, $\mathrm{K}$, $\mathrm{1}$ | `pedosphere/numerical.yaml` ($10^{-6}$) | Numerical safeguards |
| $M_C, M_{\mathrm{CH_4}}, \Delta T_{10}$ | `relic.physical.constants.molecular_weights.carbon`, `relic.physical.constants.molecular_weights.ch4`, `relic.physical.constants.temperature.q10_interval` | $\mathrm{g\,mol^{-1}}$, $\mathrm{g\,mol^{-1}}$, $\mathrm{K}$ | `physical/constants.yaml` ($12.011$, $16.0425$, $10.0$) | Physical constants SSOT |

## 5. Numerical Methods and Discretization

The methane chain executes in deterministic topological order across `ExecutionPhase.COUPLING` (`MethaneScaleCouplingFunctor`), `ExecutionPhase.MAIN_PROCESSES` (`MethaneProductionFunctor`, `MethaneOxidationFunctor`, `MethaneTransportFunctor`), and `ExecutionPhase.COUPLING` (`MethaneBudgetCoupler`, which declares explicit dependencies on production, oxidation, and transport).

- **Reaction-network tendency generation**: `MethaneProductionFunctor` and `MethaneOxidationFunctor` evaluate `advance_reaction_network` with stoichiometric vectors $N_{h,\mathrm{prod}} = (+1, +1)^{\mathsf{T}}$ (mapping $+P$ to the boundary source port and internal storage tendency) and $N_{h,\mathrm{oxid}} = (+1, -1)^{\mathsf{T}}$ (mapping $+O$ to oxidation export and $-O$ to internal storage tendency). Neither reaction Functor nor `MethaneTransportFunctor` mutates `ch4_soil_storage` directly.
- **Single-limiter operator-splitting policy**: `MethaneOxidationFunctor` first caps $O \le M / \Delta t$, so oxidation alone cannot exhaust more than $M$. `MethaneBudgetCoupler` then applies the shared step floor $\Delta M = \max(\Delta t (T_{\mathrm{prod}} + T_{\mathrm{oxid}} + T_{\mathrm{trans}}), -M)$ and verifies $R_{\mathrm{lim,tr}} \le \max(F_{\mathrm{req}}, 0) + \varepsilon_{\mathrm{abs}}$ (raising `ValueError` if violated), guaranteeing that any storage deficit beyond oxidation is absorbed exclusively by reducing outward atmospheric transport $F_{\mathrm{atm}} = F_{\mathrm{req}} - R_{\mathrm{lim,tr}}$ without creating artificial atmospheric methane uptake. Every step requires $\Delta t \ge \Delta t_{\min}$.

## 6. Conservation, Boundary Conditions, and Ledgers

- **Global carbon ledger (`carbon.total`)**: `ch4_production_rate` (`UpdateMode.REPLACE`, `element="C"`), `ch4_oxidation_rate` and `ch4_storage_tendency_oxid` (`UpdateMode.REPLACE`, `element="C"`), `ch4_atmospheric_flux` (`UpdateMode.REPLACE`, `element="C"`), and `ch4_soil_storage` (`UpdateMode.ACCUMULATE`, `per_timestep=True`, `element="C"`) carry explicit contract governance so production enters once, oxidation and realized atmospheric emission exit once, and `ch4_soil_storage` advances once per step. Internal port tendencies `ch4_storage_tendency_prod`, `ch4_transport_loss_rate`, and `ch4_storage_tendency_trans` omit `element` tags to prevent double counting.
- **Discrete budget closure and tolerance audit**: With consistent tendencies $(T_{\mathrm{prod}}, T_{\mathrm{oxid}}, T_{\mathrm{trans}}) = (P, -O, -F_{\mathrm{req}})$, the coupler residual $\mathcal{R}_{\mathrm{CH_4}} = P - O - F_{\mathrm{atm}} - \Delta M / \Delta t$ vanishes up to floating-point rounding. In `MethaneBudgetCoupler.forward`, the executable warning threshold uses the additive tolerance (whereas the class docstring writes a `max(abs_tol, ...)` expression):

$$
\lvert \mathcal{R}_{\mathrm{CH_4}} \rvert \le \varepsilon_{\mathrm{abs}} + \varepsilon_{\mathrm{rel}} \max\!\left(\lvert P \rvert,\, \lvert O \rvert,\, \lvert F_{\mathrm{atm}} \rvert\right)
$$

Exceeding this threshold logs a diagnostic warning without applying post-hoc pool rescaling.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `MethaneScaleCouplingFunctor` | `relic/scientific/pedosphere/methane/scales.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.COUPLING` |
| `MethaneProductionFunctor` | `relic/scientific/pedosphere/methane/production.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` |
| `MethaneOxidationFunctor` | `relic/scientific/pedosphere/methane/oxidation.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` |
| `MethaneTransportFunctor` | `relic/scientific/pedosphere/methane/transport.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |
| `MethaneBudgetCoupler` | `relic/scientific/pedosphere/methane/budget_coupler.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.COUPLING` |

## 8. Implementation Status

- **`MethaneScaleCouplingFunctor`**: `[H]` (IPHS Level 1 `observation`, calibratable RELIC phenomenological routing weight). Certificate: `pedosphere_methane_scales_methane_scale_coupling_functor.yaml`.

```text
w = clip(alpha_wetland * f_wet + alpha_wfps * WFPS + alpha_data * score_data - threshold, 0, 1)
y_h = O_h x_h through the declared observation operator
```

- **`MethaneProductionFunctor`**: `[H]` (IPHS Level 2 `reaction`; blends iMAPLE Eq. 34/39 `[I]` using published Zhu et al. 2014 Eq. 5 exponent $x_t$ with RELIC calibrated $R_h$-redox branch `[H]`; DNDC acetate/$\mathrm{H_2}$ substrate-kinetic methanogenesis is `[N]`). Certificate: `pedosphere_methane_production_methane_production_functor.yaml`.

```text
P = (1 - w) * Rh * r * f_T(Ts) * f_pH(pH) * f_wet + w * Rh * r * f_T(Ts) * f_Eh(Eh) * A
x_dot = N_h r_h with ParameterManager-backed process rates r_h and declared stoichiometric/output map N_h
```

- **`MethaneOxidationFunctor`**: `[I]` (IPHS Level 2 `reaction`, CLM4Me saturated wetland double Michaelis-Menten kinetics; unsaturated matric-potential factor $F = \exp(-P / P_c)$ is `[N]`). Certificate: `pedosphere_methane_oxidation_methane_oxidation_functor.yaml`.

```text
O = min(M/dt, R z M_C) with C_CH4 = M/(M_C z phi WFPS)
x_dot = N_h r_h with an open matter port, explicit limiter residual, and no independent H, e, J, M, port-power, or GENERIC claim
```

- **`MethaneTransportFunctor`**: `[H]` (IPHS Level 3 `transport`, calibrated three-pathway diffusion/ebullition/aerenchyma request; nominal Hodge contract uses $D_{\mathrm{air}} \tau$ when moisture state is absent, and texture-explicit iMAPLE Eq. 42 is `[N]`). Certificate: `pedosphere_methane_transport_methane_transport_functor.yaml`.

```text
F_transport = min(M/dt, F_diffusion + F_ebullition + F_plant)
theta_air=porosity*(1-WFPS); D_eff=tortuosity*D_air*theta_air^a/porosity^b+D_water*WFPS; F_request=F_diff+F_ebb+F_plant; T_trans=-F_request
```

- **`MethaneBudgetCoupler`**: `[I]` for discrete conservation ledger and `[H]` for oxidation-before-transport limiter priority (IPHS Level 2 `scale_coupling`). Certificate: `pedosphere_methane_budget_coupler_methane_budget_coupler.yaml`.

```text
Residual = P - O - F_transport - dM_raw/dt without a single shared availability limit
Residual = P - O - F_atm - dM/dt, with the denied transport rate routed to diagnostics.methane_budget.transport_limiter_residual
```

## 9. References

1. Cao, M., Dent, J. B., and Heal, O. W. (1995). Modeling methane emissions from rice paddies. *Global Biogeochemical Cycles*, 9(2), 183-195. doi:10.1029/94GB03231
2. Chanton, J. P., Whiting, G. J., Showers, W. J., and Crill, P. M. (1992). Methane flux from *Peltandra virginica*: Stable isotope tracing and chamber effects. *Global Biogeochemical Cycles*, 6(1), 15-31. doi:10.1029/91GB02969
3. Fumoto, T., Kobayashi, K., Li, C., Yagi, K., and Hasegawa, T. (2008). Revising a process-based biogeochemistry model (DNDC) to simulate methane emission from rice paddy fields under various residue management and fertilizer regimes. *Global Change Biology*, 14(2), 382-402. doi:10.1111/j.1365-2486.2007.01475.x
4. Millington, R. J., and Quirk, J. P. (1961). Permeability of porous solids. *Transactions of the Faraday Society*, 57, 1200-1207. doi:10.1039/TF9615701200
5. Oleson, K. W., Lawrence, D. M., Bonan, G. B., et al. (2013). *Technical Description of version 4.5 of the Community Land Model (CLM)*. NCAR Technical Note NCAR/TN-503+STR, Chapter 19.
6. Riley, W. J., Subin, Z. M., Lawrence, D. M., et al. (2011). Barriers to predicting changes in global terrestrial methane fluxes: Analyses using CLM4Me, a methane biogeochemistry model integrated in CESM. *Biogeosciences*, 8(7), 1925-1953. doi:10.5194/bg-8-1925-2011
7. van Bodegom, P. M., and Scholten, J. C. M. (2001). Microbial processes of $\mathrm{CH_4}$ production in a rice paddy soil: Model and experimental validation. *Geochimica et Cosmochimica Acta*, 65(13), 2055-2066. doi:10.1016/S0016-7037(01)00563-4
8. Walter, B. P., and Heimann, M. (2000). A process-based, climate-sensitive model to derive methane emissions from natural wetlands: Application to five wetland sites, sensitivity to model parameters, and climate. *Global Biogeochemical Cycles*, 14(3), 745-765. doi:10.1029/1999GB001204
9. Walter, B. P., Heimann, M., and Matthews, E. (2001). Modeling modern methane emissions from natural wetlands: 1. Model description and results. *Journal of Geophysical Research*, 106(D24), 34189-34206. doi:10.1029/2001JD900165
10. Wania, R., Ross, I., and Prentice, I. C. (2010). Implementation and evaluation of a new methane model within a dynamic global vegetation model: LPJ-WHyMe v1.3.1. *Geoscientific Model Development*, 3(2), 565-584. doi:10.5194/gmd-3-565-2010
11. Yue, X., et al. (2024). Development and evaluation of the interactive Model for Air Pollution and Land Ecosystems (iMAPLE) version 1.0. *Geoscientific Model Development*, 17, 4621-4642. doi:10.5194/gmd-17-4621-2024
12. Zhang, Y., Li, C., Trettin, C. C., Li, H., and Sun, G. (2002). An integrated model of soil, hydrology, and vegetation for carbon dynamics in wetland ecosystems. *Global Biogeochemical Cycles*, 16(4), 1061. doi:10.1029/2001GB001838
13. Zhu, Q., Liu, J., Peng, C., et al. (2014). Modelling methane emissions from natural wetlands by development and application of the TRIPLEX-GHG model. *Geoscientific Model Development*, 7(3), 981-999. doi:10.5194/gmd-7-981-2014
14. Zhuang, Q., Melillo, J. M., Kicklighter, D. W., et al. (2004). Methane fluxes between terrestrial ecosystems and the atmosphere at northern high latitudes during the past century: A retrospective analysis with a process-based biogeochemistry model. *Global Biogeochemical Cycles*, 18(3), GB3010. doi:10.1029/2004GB002239
