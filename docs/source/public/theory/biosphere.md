# Biosphere Theory

## Scope, process inventory, and certificate map

The RELIC biosphere domain (`relic.scientific.biosphere`) represents terrestrial
vegetation carbon assimilation, stomatal conductance, autotrophic respiration,
organ carbon allocation, structural biomass turnover, leaf phenology, abiotic
stress modifiers, and biogenic volatile organic compound (BVOC) synthesis and
emission (*Methods of Ecological Analysis*, Chapters 20--22). Living vegetation
is an open, sunlit, metabolically driven subsystem; accordingly, all 23 concrete
biosphere Functors inherit `BiologicalProcessFunctor` and carry `Level 1`
(read-only constitutive/observation map) or `Level 2` (open port/stoichiometric
balance) `IPHSEquivalenceCertificate` declarations. No closed-system `Level 4`
or `Level 5` thermodynamic equilibrium claim is made for living canopy kinetics.

| Subsystem | Functor | Cert. level | Status | Primary runtime role and state semantics |
| --- | --- | --- | --- | --- |
| Photosynthesis & stomata | {ref}`FarquharPhotosynthesis <class-relic-scientific-biosphere-photosynthesis-farquhar-farquharphotosynthesis>` | `Level 2` | `[I]` | FvCB $\mathrm{C_3}$ biochemical assimilation ($A_c, A_j, A_p$, gross and net $A$) (`REPLACE`) |
| Photosynthesis & stomata | {ref}`MedlynStomatalConductance <class-relic-scientific-biosphere-photosynthesis-stomatal-medlynstomatalconductance>` | `Level 1` | `[I]` | Conditional leaf-support optimal stomatal conductance $g_s$ (`REPLACE`) |
| Photosynthesis & stomata | {ref}`SimpleCanopyPhotosynthesis <class-relic-scientific-biosphere-surrogates-simple-photosynthesis-simplecanopyphotosynthesis>` | `Level 2` | `[I]` | Light-use-efficiency canopy GPP surrogate (`REPLACE`) |
| Autotrophic respiration | {ref}`MaintenanceRespiration <class-relic-scientific-biosphere-respiration-maintenance-respiration-maintenancerespiration>` | `Level 2` | `[I]` | Organ-resolved nitrogen/biomass $Q_{10}$ maintenance respiration $R_m$ (`REPLACE`) |
| Autotrophic respiration | {ref}`GrowthRespiration <class-relic-scientific-biosphere-respiration-growth-respiration-growthrespiration>` | `Level 2` | `[I]` | Biosynthetic construction respiration $R_g$ proportional to positive carbon gain (`REPLACE`) |
| Autotrophic respiration | {ref}`AutotrophicRespiration <class-relic-scientific-biosphere-respiration-autotrophic-respiration-autotrophicrespiration>` | `Level 2` | `[I]` | Coupled $R_a = R_m + R_g$, NPP, and carbon-use efficiency closure (`REPLACE`) |
| Allocation & growth | {ref}`BiomassAllocation <class-relic-scientific-biosphere-growth-biomass-allocation-biomassallocation>` | `Level 2` | `[I]` | Resource-responsive simplex partitioning of NPP across leaves, stems, and roots (`REPLACE`) |
| Allocation & growth | {ref}`StructuralGrowth <class-relic-scientific-biosphere-growth-structural-growth-structuralgrowth>` | `Level 2` | `[I]` | Organ carbon mass balance, senescence litter fluxes, and allometric LAI (`ACCUMULATE` / `REPLACE`) |
| Allocation & growth | {ref}`PhenologyGrowth <class-relic-scientific-biosphere-growth-phenology-growth-phenologygrowth>` | `Level 1` | `[I]` | Phenology-stage growth-modifier observation (`REPLACE`) |
| Phenology (`PhenologyBase`) | {ref}`StaticPhenology <class-relic-scientific-biosphere-phenology-models-staticphenology>` | `Level 2` | `[I]` | Prescribed time-invariant LAI and canopy cover baseline (`REPLACE`) |
| Phenology (`PhenologyBase`) | {ref}`GDDPhenology <class-relic-scientific-biosphere-phenology-models-gddphenology>` | `Level 2` | `[I]` | Growing-degree-day thermal budburst, expansion, and senescence (`ACCUMULATE` / `REPLACE`) |
| Phenology (`PhenologyBase`) | {ref}`GSIPhenology <class-relic-scientific-biosphere-phenology-models-gsiphenology>` | `Level 2` | `[I]` | Growing Season Index (temperature, photoperiod, VPD) canopy state (`REPLACE`) |
| Phenology (`PhenologyBase`) | {ref}`CropPhenology <class-relic-scientific-biosphere-phenology-models-cropphenology>` | `Level 2` | `[I]` | Stage-resolved agricultural thermal-time development and harvest maturity (`ACCUMULATE` / `REPLACE`) |
| Phenology (`PhenologyBase`) | {ref}`PhenologyFunctor <class-relic-scientific-biosphere-phenology-phenology-functor-phenologyfunctor>` | `Level 1` | `[I]` | Configured phenology observation and stage-index wrapper (`REPLACE`) |
| Abiotic stress | {ref}`WaterStress <class-relic-scientific-biosphere-stress-water-stress-waterstress>` | `Level 1` | `[I]` | Soil-moisture / matric-potential scalar $\beta_W \in [0, 1]$ (`REPLACE`) |
| Abiotic stress | {ref}`NutrientStress <class-relic-scientific-biosphere-stress-nutrient-stress-nutrientstress>` | `Level 1` | `[I]` | Liebig or multiplicative foliar N:P limitation scalar $\beta_N \in [0, 1]$ (`REPLACE`) |
| Abiotic stress | {ref}`ThermalStress <class-relic-scientific-biosphere-stress-thermal-stress-thermalstress>` | `Level 1` | `[I]` | Low- and high-temperature inhibition scalar $\beta_T \in [0, 1]$ (`REPLACE`) |
| Abiotic stress | {ref}`OzoneStress <class-relic-scientific-biosphere-stress-ozone-stress-ozonestress>` | `Level 2` | `[I]` | Cumulative stomatal ozone uptake ($\mathrm{POD}_y$) damage dynamics (`ACCUMULATE` / `REPLACE`) |
| BVOC synthesis & emission | {ref}`MEGANModel <class-relic-scientific-biosphere-bvoc-megan-model-meganmodel>` | `Level 2` | `[I]` | Canopy empirical BVOC emission factors and $\gamma$ activity multipliers (`REPLACE`) |
| BVOC synthesis & emission | {ref}`IsopreneSynthesis <class-relic-scientific-biosphere-bvoc-synthesis-isoprene-isoprenesynthesis>` | `Level 2` | `[I]` | Plastidic MEP-pathway DMADP/energy-limited isoprene synthesis (`REPLACE`) |
| BVOC synthesis & emission | {ref}`MonoterpeneSynthesis <class-relic-scientific-biosphere-bvoc-synthesis-monoterpene-monoterpenesynthesis>` | `Level 2` | `[I]` | Dual *de novo* plus bounded resin-pool ($P \in [0, P_{\max}]$) monoterpene emission (`ACCUMULATE` / `REPLACE`) |
| BVOC synthesis & emission | {ref}`SesquiterpeneSynthesis <class-relic-scientific-biosphere-bvoc-synthesis-sesquiterpene-sesquiterpenesynthesis>` | `Level 2` | `[I]` | Temperature- and stress-driven sesquiterpene synthesis and emission (`REPLACE`) |
| BVOC synthesis & emission | {ref}`SesquiterpeneStressModel <class-relic-scientific-biosphere-bvoc-synthesis-sesquiterpene-sesquiterpenestressmodel>` | `Level 1` | `[I]` | Oxidative, herbivory/mechanical, and thermal stress-induction factor (`REPLACE`) |

`relic.scientific.biosphere.phenology.base.PhenologyBase` is an abstract base
class shared by the concrete phenology implementations and carries no separate
certificate. General notation follows [Mathematical Foundations](foundations.md),
structural levels follow [DEC, IPHS, and GENERIC](dec_iphs_generic.md), and
integrator contracts follow [Numerical Methods](numerics.md).

## Photosynthesis and stomatal conductance

### Farquhar-von Caemmerer-Berry biochemical assimilation

`relic.scientific.biosphere.photosynthesis.farquhar.FarquharPhotosynthesis`
(`Level 2`) implements the $\mathrm{C_3}$ biochemical model of Farquhar, von
Caemmerer, and Berry (1980) with peaked Arrhenius temperature scaling of
$V_{c\max}$, $J_{\max}$, triose-phosphate utilization $T_p$, Michaelis constants
$K_c, K_o$, and $\mathrm{CO}_2$ compensation point $\Gamma^\ast$ (Bernacchi
et al., 2001; Medlyn et al., 2002; *Methods of Ecological Analysis*, Chapter 20).
At intercellular $\mathrm{CO}_2$ partial pressure or mole fraction $C_i$ and
oxygen level $O_i$, the Rubisco-limited ($W_c$), RuBP-regeneration-limited
($W_j$), and TPU-limited ($W_p$) gross carboxylation rates are

$$
W_c = V_{c\max}(T_l)\,\frac{C_i}{C_i + K_c(T_l)\left(1 + O_i / K_o(T_l)\right)},
\qquad
W_j = \frac{J(I_{\mathrm{APAR}}, T_l)}{4}\,\frac{C_i}{C_i + 2\Gamma^\ast(T_l)},
\qquad
W_p = 3\,T_p(T_l)\,\frac{C_i}{C_i - \Gamma^\ast(T_l)},
$$

where electron transport $J$ solves the non-rectangular hyperbola

$$
\theta_J J^2 - \left(\alpha_J I_{\mathrm{APAR}} + J_{\max}(T_l)\right)J + \alpha_J I_{\mathrm{APAR}} J_{\max}(T_l) = 0
$$

on its smaller positive root. Gross and net leaf or canopy assimilation
[$\mu$mol $\mathrm{CO}_2$ m$^{-2}$ s$^{-1}$] satisfy

$$
A_{\mathrm{gross}} = \left(1 - \frac{\Gamma^\ast(T_l)}{C_i}\right)\min_{\mathrm{smooth}}(W_c, W_j, W_p),
\qquad
A_{\mathrm{net}} = \beta_{\mathrm{stress}}\,A_{\mathrm{gross}} - R_d(T_l),
$$

where $R_d(T_l)$ is mitochondrial day respiration and $\min_{\mathrm{smooth}}$
denotes quadratic co-limitation (Collatz et al., 1991) or exact pointwise
minimum according to configuration.

### Optimal stomatal conductance (Medlyn closure)

`relic.scientific.biosphere.photosynthesis.stomatal.MedlynStomatalConductance`
(`Level 1`) evaluates the optimal water-use-efficiency stomatal conductance law
of Medlyn et al. (2011) as a conditional leaf-support observation under the
explicit boundary assumption `surface_conditions = "leaf_surface_equals_canopy_air"`.
Canopy-ground net assimilation is normalized onto leaf area using
`leaf_area_index` ($A_{\mathrm{leaf}}$), leaf-surface $\mathrm{CO}_2$ is $C_s$
[$\mu$mol mol$^{-1}$], and vapor-pressure deficit is bounded away from zero by
$D_{\mathrm{eff}} = \max(D_s, D_{\min})$ [kPa]. Conductance to water vapor
$g_s$ [mol m$^{-2}$ s$^{-1}$] is

$$
g_s =
\begin{cases}
g_0, & \mathrm{PPFD} \le \mathrm{PPFD}_{\mathrm{dark}} \;\text{ or }\; A_{\mathrm{leaf}} \le 0,\\
g_0 + 1.6\left(1 + \dfrac{g_1}{\sqrt{D_{\mathrm{eff}}}}\right)\dfrac{A_{\mathrm{leaf}}}{C_s}, & \mathrm{PPFD} > \mathrm{PPFD}_{\mathrm{dark}} \;\text{ and }\; A_{\mathrm{leaf}} > 0,
\end{cases}
$$

where $1.6$ is the molecular diffusivity ratio $D_{\mathrm{H_2O}}/D_{\mathrm{CO_2}}$,
$g_0 \ge 0$ is residual cuticular/minimum conductance, and $g_1$ [kPa$^{1/2}$]
is the marginal water-cost slope parameter. `MedlynStomatalConductance` emits
`stomatal_conductance` via `REPLACE` without solving an implicit leaf energy
balance or mutating carbon pools.

### Light-use-efficiency canopy surrogate

`relic.scientific.biosphere.surrogates.simple_photosynthesis.SimpleCanopyPhotosynthesis`
(`Level 2`) provides a diagnostic Monteith (1972) light-use-efficiency closure,

$$
\mathrm{GPP} = \varepsilon_{\max}\,\mathrm{APAR}\,f_T(T_a)\,f_{\mathrm{VPD}}(D)\,\beta_W,
$$

for coarse-scale or calibration workflows where biochemical $C_i$ iteration is
not selected. Exactly one photosynthesis owner (`FarquharPhotosynthesis` or
`SimpleCanopyPhotosynthesis`) may own `gpp` in a given simulator graph.

## Autotrophic respiration, allocation, and structural growth

### Maintenance, growth, and total autotrophic respiration

RELIC separates autotrophic respiration into modular `Level 2` owners (*Methods
of Ecological Analysis*, Chapter 21; Ryan, 1991; Amthor, 2000):

1. `relic.scientific.biosphere.respiration.maintenance_respiration.MaintenanceRespiration`
   evaluates organ-summed maintenance cost from live biomass $C_k$ or tissue
   nitrogen content $N_k$ with $Q_{10}$ or modified-Arrhenius temperature
   acclimation (Atkin and Tjoelker, 2003):

   $$R_m = \sum_{k \in \{\mathrm{leaf,stem,root}\}} r_{m,k}\,M_k\,f_{T,k}(T_k), \qquad f_{T,k}(T_k) = Q_{10,k}^{(T_k - T_{\mathrm{ref}})/10}.$$

2. `relic.scientific.biosphere.respiration.growth_respiration.GrowthRespiration`
   computes biosynthetic construction cost on positive net assimilatory surplus:

   $$R_g = f_{\mathrm{gr}}\,\max(\mathrm{GPP} - R_m,\; 0), \qquad 0 \le f_{\mathrm{gr}} < 1.$$

3. `relic.scientific.biosphere.respiration.autotrophic_respiration.AutotrophicRespiration`
   closes the canopy autotrophic carbon flux budget:

   $$R_a = R_m + R_g, \qquad \mathrm{NPP} = \mathrm{GPP} - R_a, \qquad \mathrm{CUE} = \frac{\mathrm{NPP}}{\max(\mathrm{GPP}, \varepsilon_{\mathrm{GPP}})}.$$

### Simplex carbon allocation and organ mass conservation

`relic.scientific.biosphere.growth.biomass_allocation.BiomassAllocation`
(`Level 2`) maps environmental light, water, and nutrient limitation signals to
organ allocation fractions $\mathbf{a} = (a_{\mathrm{leaf}}, a_{\mathrm{stem}}, a_{\mathrm{root}})^{\mathsf{T}}$
(Friedlingstein et al., 1999) satisfying exact simplex normalization:

$$
a_k \ge 0,
\qquad
\sum_{k \in \{\mathrm{leaf,stem,root}\}} a_k = 1,
\qquad
F_{\mathrm{alloc},k} = a_k\,\max(\mathrm{NPP}, 0).
$$

`relic.scientific.biosphere.growth.structural_growth.StructuralGrowth`
(`Level 2`) advances organ carbon pools $C_k$ [g C m$^{-2}$] with senescence
turnover rates $\tau_k^{-1}$:

$$
\frac{\mathrm{d}C_k}{\mathrm{d}t} = F_{\mathrm{alloc},k} - L_k,
\qquad
L_k = \frac{C_k}{\tau_k}\,f_{\mathrm{sen},k},
$$

emitting integrated pool increments $\Delta C_k$ via
`UpdateItem(ACCUMULATE, per_timestep=True)`, litterfall fluxes $L_k$ (`REPLACE`,
routed to pedosphere `LitterfallInterfaceMapper`), and prognostic leaf area
index $L_{\mathrm{AI}} = \mathrm{SLA}\cdot C_{\mathrm{leaf}}$ so the step carbon
ledger $\sum_k \Delta C_k + \sum_k L_k\Delta t - \sum_k F_{\mathrm{alloc},k}\Delta t = 0$
closes identically. `relic.scientific.biosphere.growth.phenology_growth.PhenologyGrowth`
(`Level 1`) supplies read-only stage-dependent growth multipliers without
mutating carbon stocks.

## Phenology and abiotic stress closures

### Leaf phenology models

Concrete phenology Functors inherit `relic.scientific.biosphere.phenology.base.PhenologyBase`:

- `StaticPhenology` (`Level 2`): emits prescribed stationary $L_{\mathrm{AI}}$
  and greenness fraction.
- `GDDPhenology` (`Level 2`): accumulates daily thermal forcing
  $\Delta \mathrm{GDD} = \max(T_a - T_{\mathrm{base}}, 0)\,\Delta t_{\mathrm{days}}$
  (`ACCUMULATE`, `per_timestep=True`) to drive logistic budburst, full-canopy
  plateau, and cold/photoperiod senescence.
- `GSIPhenology` (`Level 2`): evaluates the Jolly et al. (2005) Growing Season
  Index as the product of piecewise-linear minimum-temperature, photoperiod, and
  vapor-pressure-deficit indicators, $\mathrm{GSI} = i_{T_{\min}}\,i_{\mathrm{photo}}\,i_{\mathrm{VPD}} \in [0, 1]$.
- `CropPhenology` (`Level 2`): tracks vernalization- and photoperiod-weighted
  thermal units across emergence, vegetative, flowering, grain-filling, and
  maturity stages (`ACCUMULATE` thermal progress, `REPLACE` stage diagnostics).
- `PhenologyFunctor` (`Level 1`): projects active phenological stage indices and
  canopy-activity scalars without owning separate carbon storage.

### Water, nutrient, thermal, and ozone stress Functors

`WaterStress`, `NutrientStress`, and `ThermalStress` (`Level 1`) are pure
constitutive observers mapping soil water content or potential, foliar N and P
ratios, and leaf temperature extremes to bounded dimensionless scalars
$\beta_W, \beta_N, \beta_T \in [0, 1]$ (`REPLACE`).

`relic.scientific.biosphere.stress.ozone_stress.OzoneStress` (`Level 2`)
integrates phytotoxic ozone dose above flux threshold $y$ ($\mathrm{POD}_y$,
[mmol m$^{-2}$]; Emberson et al., 2000; Sitch et al., 2007):

$$
F_{\mathrm{O_3,stom}} = g_{s,\mathrm{O_3}}\,[\mathrm{O}_3]_{\mathrm{canopy}},
\qquad
\Delta \mathrm{POD}_y = \max\!\left(F_{\mathrm{O_3,stom}} - y,\; 0\right)\Delta t,
$$

committing $\Delta \mathrm{POD}_y$ via `UpdateItem(ACCUMULATE, per_timestep=True)`
and emitting the resulting photosynthetic damage factor
$\beta_{\mathrm{O_3}} = \operatorname{clip}(1 - a_{\mathrm{O_3}}\,\mathrm{POD}_y,\; \beta_{\mathrm{O_3,min}},\; 1)$
via `REPLACE`.

## Biogenic volatile organic compound (BVOC) synthesis and emission

### Canopy empirical MEGAN closure and continuous CO2 inhibition

`relic.scientific.biosphere.bvoc.megan_model.MEGANModel` (`Level 2`) evaluates
canopy BVOC emission rate $E_s$ for compound class $s$ following Guenther et al.
(2006, 2012) (*Methods of Ecological Analysis*, Chapter 22):

$$
E_s = \varepsilon_s\,\gamma_{\mathrm{CE}}(L_{\mathrm{AI}})\,\gamma_P(\mathrm{PPFD})\,\gamma_T(T_l)\,\gamma_{\mathrm{age}}\,\gamma_{\mathrm{SM}}(\theta)\,\gamma_{\mathrm{CO_2}}(C_a),
$$

where $\varepsilon_s$ is the standard-condition emission factor and
$\gamma_{\mathrm{CE}}, \gamma_P, \gamma_T, \gamma_{\mathrm{age}}, \gamma_{\mathrm{SM}}$
are the canopy-environment, light, temperature-optimum, leaf-age, and
soil-moisture activity factors. For isoprene $\mathrm{CO}_2$ inhibition, RELIC
evaluates the Wilkinson et al. (2009) / Guenther et al. (2012, Eq. 14) Hill
curve continuously from intercellular concentration $C_i = r_{C_i} C_a$
($r_{C_i} = 0.7$, so $C_i = 280$ ppm at $C_a = 400$ ppm):

$$
\gamma_{\mathrm{CO_2}}(C_a) = \mathrm{IS}_{\max} - \frac{\mathrm{IS}_{\max}\,C_i^h}{C_\ast^h + C_i^h},
$$

with $\mathrm{IS}_{\max} = 1.344$, $C_\ast = 585$ ppm, and $h = 1.4614$. At
ambient $C_a = 400$ ppm, this continuous parameterization evaluates to
`1.0024713532535012` in `float64` (`1.0024713277816772` in `float32`); RELIC
does not apply an ad-hoc discontinuity clamp forcing $\gamma_{\mathrm{CO_2}}(400\;\mathrm{ppm}) = 1$.

### Mechanistic plastidic isoprene, resin-pool monoterpene, and sesquiterpene synthesis

1. **Plastidic MEP isoprene synthesis (`IsopreneSynthesis`, `Level 2`)**:
   `relic.scientific.biosphere.bvoc.synthesis.isoprene.IsopreneSynthesis` models
   chloroplastic dimethylallyl diphosphate (DMADP) supply driven by
   photosynthetic electron transport $J$ and ATP/NADPH energetic availability
   (Niinemets et al., 1999; Morfopoulos et al., 2014), coupled with isoprene
   synthase Arrhenius kinetics. Without a permanent leaf storage organ, emission
   equals instantaneous enzymatic synthesis (`REPLACE`).

2. **Dual de-novo and bounded storage-pool monoterpene dynamics (`MonoterpeneSynthesis`, `Level 2`)**:
   `relic.scientific.biosphere.bvoc.synthesis.monoterpene.MonoterpeneSynthesis`
   separates light-dependent *de novo* synthesis $S_{\mathrm{denovo}}$ from a
   specialized resin-duct or glandular-trichome storage pool $P \in [0, P_{\max}]$
   (Schurgers et al., 2009; Grote et al., 2013). With storage-directed synthesis
   $S_{\mathrm{pool}}$ and temperature-driven volatilization release
   $E_{\mathrm{pool}} = k_{\mathrm{rel}}(T_l)\,P$, raw storage advancement
   $P_{\mathrm{raw}} = P + (S_{\mathrm{pool}} - E_{\mathrm{pool}})\Delta t$ is
   projected onto $[0, P_{\max}]$:

   $$P_{\mathrm{next}} = \operatorname{clip}(P_{\mathrm{raw}},\; 0,\; P_{\max}), \qquad r_{\mathrm{bound}} = \frac{P_{\mathrm{raw}} - P_{\mathrm{next}}}{\Delta t},$$

   committing $\Delta P = P_{\mathrm{next}} - P$ via
   `UpdateItem(ACCUMULATE, per_timestep=True)` and recording any capacity-clamp
   flux explicitly in `pool_boundary_residual_rate` ($r_{\mathrm{bound}}$) so the
   terpene carbon ledger closes identically:

   $$\frac{\Delta P}{\Delta t} + E_{\mathrm{pool}} + r_{\mathrm{bound}} - S_{\mathrm{pool}} = 0.$$

3. **Stress-induced sesquiterpene synthesis (`SesquiterpeneStressModel` `Level 1`, `SesquiterpeneSynthesis` `Level 2`)**:
   `relic.scientific.biosphere.bvoc.synthesis.sesquiterpene.SesquiterpeneStressModel`
   (`Level 1`) aggregates oxidative, thermal, and biotic/mechanical stress cues
   into a non-negative induction multiplier $\gamma_{\mathrm{stress}}$ (`REPLACE`),
   which `relic.scientific.biosphere.bvoc.synthesis.sesquiterpene.SesquiterpeneSynthesis`
   (`Level 2`) combines with $\beta$-exponential temperature volatility
   $\exp(\beta_{\mathrm{SQT}}(T_l - T_s))$ and light-modulated *de novo*
   farnesyl-diphosphate synthesis (Duhl et al., 2008).

In composed graphs, users select either canopy empirical `MEGANModel` or the
mechanistic compound synthesis owners for a given BVOC flux target so surface
emissions to atmospheric chemistry and observation operators have a single
writer.

## References

- Amthor, J. S. (2000). The McCree-de Wit-Penning de Vries-Thornley respiration paradigms: 30 years later. *Annals of Botany*, 86, 1-20.
- Atkin, O. K. and Tjoelker, M. G. (2003). Thermal acclimation and the dynamic response of plant respiration to temperature. *Trends in Plant Science*, 8, 343-351.
- Bernacchi, C. J., Singsaas, E. L., Pimentel, C., Portis, A. R., and Long, S. P. (2001). Improved temperature response functions for models of Rubisco-limited photosynthesis. *Plant, Cell & Environment*, 24, 253-259.
- Collatz, G. J., Ball, J. T., Grivet, C., and Berry, J. A. (1991). Physiological and environmental regulation of stomatal conductance, photosynthesis and transpiration: a model that includes a laminar boundary layer. *Agricultural and Forest Meteorology*, 54, 107-136.
- Duhl, T. R., Helmig, D., and Guenther, A. (2008). Sesquiterpene emissions from vegetation: a review. *Biogeosciences*, 5, 761-777. doi:10.5194/bg-5-761-2008.
- Emberson, L. D., Ashmore, M. R., Cambridge, H. M., Simpson, D., and Tuovinen, J.-P. (2000). Modelling stomatal ozone flux across Europe. *Environmental Pollution*, 109, 403-413.
- Farquhar, G. D., von Caemmerer, S., and Berry, J. A. (1980). A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. *Planta*, 149, 78-90.
- Friedlingstein, P., Joel, G., Field, C. B., and Fung, I. Y. (1999). Toward an allocation scheme for global terrestrial carbon models. *Global Change Biology*, 5, 755-770.
- Grote, R., Monson, R. K., and Niinemets, U. (2013). Leaf-level models of constitutive and stress-driven volatile organic compound emissions. In *Biology, Controls and Models of Tree Volatile Organic Compound Emissions*, Springer, 315-355.
- Guenther, A. et al. (2006). Estimates of global terrestrial isoprene emissions using MEGAN (Model of Emissions of Gases and Aerosols from Nature). *Atmospheric Chemistry and Physics*, 6, 3181-3210. doi:10.5194/acp-6-3181-2006.
- Guenther, A. B. et al. (2012). The Model of Emissions of Gases and Aerosols from Nature version 2.1 (MEGAN2.1): an extended and updated framework for modeling biogenic emissions. *Geoscientific Model Development*, 5, 1471-1492. doi:10.5194/gmd-5-1471-2012.
- Jolly, W. M., Nemani, R., and Running, S. W. (2005). A generalized, bioclimatic index to predict foliar phenology in response to climate. *Global Change Biology*, 11, 619-632.
- Medlyn, B. E. et al. (2002). Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data. *Plant, Cell & Environment*, 25, 1167-1179.
- Medlyn, B. E. et al. (2011). Reconciling the optimal and empirical approaches to modelling stomatal conductance. *Global Change Biology*, 17, 2134-2144. doi:10.1111/j.1365-2486.2010.02375.x.
- Monteith, J. L. (1972). Solar radiation and productivity in tropical ecosystems. *Journal of Applied Ecology*, 9, 747-766.
- Morfopoulos, C. et al. (2014). A model of plant isoprene emission based on available reducing power captures responses to atmospheric CO2. *New Phytologist*, 203, 125-139.
- Niinemets, U., Tenhunen, J. D., Harley, P. C., and Steinbrecher, R. (1999). A model of isoprene emission based on energetic requirements for isoprene synthesis and leaf photosynthetic properties for *Quercus coccifera*. *Plant, Cell & Environment*, 22, 1319-1335.
- Ryan, M. G. (1991). Effects of climate change on plant respiration. *Ecological Applications*, 1, 157-167.
- Schurgers, G., Arneth, A., Holzinger, R., and Goldstein, A. H. (2009). Process-based modelling of biogenic monoterpene emissions combining production and release from storage. *Atmospheric Chemistry and Physics*, 9, 3409-3423. doi:10.5194/acp-9-3409-2009.
- Sitch, S., Cox, P. M., Collins, W. J., and Huntingford, C. (2007). Indirect radiative forcing of climate change through ozone effects on the land-carbon sink. *Nature*, 448, 791-794.
- Wilkinson, M. J. et al. (2009). Leaf isoprene emission rate as a function of atmospheric CO2 concentration. *Global Change Biology*, 15, 1189-1200. doi:10.1111/j.1365-2486.2008.01803.x.
