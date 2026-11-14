# Pedosphere Theory

## Scope, process inventory, and certificate map

The RELIC pedosphere domain (`relic.scientific.pedosphere`) represents litter
and soil organic carbon turnover, cross-sphere litterfall and hydrological
chain maps, soil nitrogen mineralization, nitrification, denitrification, and
leaching, root nitrogen and phosphorus uptake, diagnostic C:N stoichiometry
auditing, wetland methane production, oxidation, three-pathway transport, and
single-owner budget realization, as well as porous-media water, gas, solute,
Freundlich index-1 equilibrium-sorption DAE, moving-domain ALE diffusion, and
mobile-immobile two-region mass transfer (*Methods of Ecological Analysis*,
Chapter 23).

Twenty-four concrete scientific Functors reside in `relic.scientific.pedosphere`,
each paired with an `IPHSEquivalenceCertificate` spanning `Level 1` (`3` classes),
`Level 2` (`13` classes), and `Level 3` (`8` classes):

| Subsystem | Functor | Canonical bridge | Cert. level | Status | Primary runtime role and state semantics |
| --- | --- | --- | --- | --- | --- |
| Soil organic carbon | {ref}`CarbonPoolsFunctor <class-relic-scientific-pedosphere-carbon-pools-carbon-pools-functor-carbonpoolsfunctor>` | `BiologicalProcessFunctor` | `Level 1` | `[I]` | Read-only 5-pool areal carbon stock summation (`REPLACE` observation) |
| Soil organic carbon | {ref}`PoolsMatrix <class-relic-scientific-pedosphere-carbon-pools-pools-matrix-poolsmatrix>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | Linear compartment-matrix $(T - I_d)K\mathbf{C}$ turnover and $R_{\mathrm{hetero}}$ (`ACCUMULATE` / `REPLACE`) |
| Soil organic carbon | {ref}`FirstOrderDecomposition <class-relic-scientific-pedosphere-decomposition-first-order-firstorderdecomposition>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | Pool-resolved $Q_{10}$/Arrhenius and moisture-scaled decay (`ACCUMULATE` / `REPLACE`) |
| Soil organic carbon | {ref}`MicrobialEnzymeDecomposition <class-relic-scientific-pedosphere-decomposition-microbial-enzyme-microbialenzymedecomposition>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | Reverse Michaelis-Menten microbial-biomass assimilation and turnover (`ACCUMULATE` / `REPLACE`) |
| Cross-sphere mappers | {ref}`LitterfallInterfaceMapper <class-relic-scientific-pedosphere-interface-mapping-litterfallinterfacemapper>` | `ScaleCouplingFunctor` | `Level 2` | `[I]` | Element-conserving plant litter and dung C/N projection onto soil input ports (`REPLACE`) |
| Cross-sphere mappers | {ref}`PedosphereInterfaceMapper <class-relic-scientific-pedosphere-interface-mapping-pedosphereinterfacemapper>` | `ScaleCouplingFunctor` | `Level 2` | `[I]` | Unit-preserving $I_3$ hydrological flux chain map (`REPLACE`) |
| Nitrogen & nutrients | {ref}`MineralizationProcess <class-relic-scientific-pedosphere-nutrient-cycling-mineralization-mineralizationprocess>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | Gross organic-N mineralization and CUE-driven $\mathrm{NH_4^+}$ immobilization (`ACCUMULATE` / `REPLACE`) |
| Nitrogen & nutrients | {ref}`NitrificationProcess <class-relic-scientific-pedosphere-nutrient-cycling-nitrification-nitrificationprocess>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | Aerobic Michaelis-Menten $\mathrm{NH_4^+} \rightarrow \mathrm{NO_3^-} + \mathrm{N_2O}$ oxidation (`ACCUMULATE` / `REPLACE`) |
| Nitrogen & nutrients | {ref}`DenitrificationProcess <class-relic-scientific-pedosphere-nutrient-cycling-denitrification-denitrificationprocess>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | Anaerobic WFPS-gated $\mathrm{NO_3^-} \rightarrow \mathrm{N_2O} + \mathrm{N_2}$ reduction (`ACCUMULATE` / `REPLACE`) |
| Nitrogen & nutrients | {ref}`NitrogenCycling <class-relic-scientific-pedosphere-nutrient-cycling-nitrogen-nitrogencycling>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | Diagnostic N-tendency aggregation plus external fertilizer/urine $\mathrm{NH_4^+}$ port (`ACCUMULATE` / `REPLACE`) |
| Nitrogen & nutrients | {ref}`NitrogenLeachingProcess <class-relic-scientific-pedosphere-nutrient-cycling-nitrogen-leaching-nitrogenleachingprocess>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Retarded drainage-driven $\mathrm{NO_3^-}$ export (`ACCUMULATE` / `REPLACE`) |
| Nitrogen & nutrients | {ref}`RootNutrientUptake <class-relic-scientific-pedosphere-nutrient-cycling-root-nutrient-uptake-rootnutrientuptake>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | Layered Michaelis-Menten $\mathrm{NO_3^-}$, $\mathrm{NH_4^+}$, and 4-way-bounded $\mathrm{P}$ root uptake (`ACCUMULATE` / `REPLACE`) |
| Nitrogen & nutrients | {ref}`StoichiometryConstraint <class-relic-scientific-pedosphere-nutrient-cycling-stoichiometry-stoichiometryconstraint>` | `ScaleCouplingFunctor` | `Level 1` | `[I]` | Read-only C:N flux-ratio and mass-balance violation audit (`REPLACE` diagnostics) |
| Wetland methane | {ref}`MethaneProductionFunctor <class-relic-scientific-pedosphere-methane-production-methaneproductionfunctor>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | iMAPLE inundation and redox-calibrated $\mathrm{CH_4}$ production tendency (`REPLACE`) |
| Wetland methane | {ref}`MethaneOxidationFunctor <class-relic-scientific-pedosphere-methane-oxidation-methaneoxidationfunctor>` | `BiologicalProcessFunctor` | `Level 2` | `[I]` | Dual-substrate ($\mathrm{CH_4}, \mathrm{O_2}$) Michaelis-Menten methanotrophy request (`REPLACE`) |
| Wetland methane | {ref}`MethaneTransportFunctor <class-relic-scientific-pedosphere-methane-transport-methanetransportfunctor>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Diffusion, ebullition, and aerenchyma plant-mediated $\mathrm{CH_4}$ flux request (`REPLACE`) |
| Wetland methane | {ref}`MethaneBudgetCoupler <class-relic-scientific-pedosphere-methane-budget-coupler-methanebudgetcoupler>` | `ScaleCouplingFunctor` | `Level 2` | `[I]` | Single storage-realization owner for `ch4_soil_storage` and `ch4_atmospheric_flux` (`ACCUMULATE` / `REPLACE`) |
| Wetland methane | {ref}`MethaneScaleCouplingFunctor <class-relic-scientific-pedosphere-methane-scales-methanescalecouplingfunctor>` | `ScaleCouplingFunctor` | `Level 1` | `[I]` | Bounded diagnostic wetland-pathway routing weight (`REPLACE`) |
| Porous-media transport | {ref}`SoilWaterTransportFunctor <class-relic-scientific-pedosphere-transport-soilwatertransportfunctor-soilwatertransportfunctor>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | 1D mixed-form Richards unsaturated soil-water solver (`REPLACE`) |
| Porous-media transport | {ref}`GasDiffusion1D <class-relic-scientific-pedosphere-transport-gas-diffusion-gasdiffusion1d>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | 1D Millington-Quirk $\mathrm{CO_2}$-C and $\mathrm{N_2O}$-N vadose gas diffusion (`ACCUMULATE` / `REPLACE`) |
| Porous-media transport | {ref}`SoluteAdvectionDiffusion1D <class-relic-scientific-pedosphere-transport-solute-advection-diffusion-soluteadvectiondiffusion1d>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | 1D retarded $\mathrm{NO_3^-}$-N and DOC-C advection-dispersion (`REPLACE`) |
| Porous-media transport | {ref}`EquilibriumSorptionFunctor <class-relic-scientific-pedosphere-transport-equilibrium-sorption-equilibriumsorptionfunctor>` | `PhysicalFieldFunctor` | `Level 2` | `[I]` | Freundlich index-1 descriptor DAE with implicit dissolved decay via `BracketedBisectionSolver` (`ACCUMULATE` / `REPLACE`) |
| Porous-media transport | {ref}`MovingDomainDiffusionFunctor <class-relic-scientific-pedosphere-transport-moving-domain-diffusion-movingdomaindiffusionfunctor>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | 1D ALE finite-volume diffusion on a uniformly stretching column with exact GCL (`ACCUMULATE` / `REPLACE`) |
| Porous-media transport | {ref}`TwoRegionMassTransferFunctor <class-relic-scientific-pedosphere-transport-two-region-mass-transfer-tworegionmasstransferfunctor>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Mobile-immobile exact matrix-exponential solute exchange (`SubFlowAccuracy.EXACT`, `ACCUMULATE` / `REPLACE`) |

General conservation ledgers follow [Mathematical Foundations](foundations.md),
discrete cochain operators follow [DEC, IPHS, and GENERIC](dec_iphs_generic.md),
and DAE, ALE, and splitting contracts follow [Numerical Methods](numerics.md).

## Soil organic carbon turnover and cross-sphere interface maps

### Five-pool observation and mutually exclusive decomposition closures

On the canonical five-pool areal carbon vector
$\mathbf{C} = (C_{\mathrm{met}}, C_{\mathrm{str}}, C_{\mathrm{fast}}, C_{\mathrm{slow}}, C_{\mathrm{passive}})^{\mathsf{T}}$
[g C m$^{-2}$], `CarbonPoolsFunctor` (`Level 1`) evaluates the per-cell total
$C_{\mathrm{total}}(c) = \sum_i C_i(c)$ across the pool axis without summing
across unequal-area grid cells or modifying stocks ($\Delta \mathbf{C} = \mathbf{0}$).

A simulation graph selects **at most one** prognostic decomposition owner for a
given soil carbon support (Parton et al., 1987; Coleman and Jenkinson, 1996;
Wieder et al., 2013):

1. **Compartment matrix (`PoolsMatrix`, `Level 2`)**: with diagonal turnover
   matrix $K$ [s$^{-1}$], column-sub-stochastic transfer matrix $T$ ($T_{ij} \ge 0$,
   $\sum_i T_{ij} \le 1$), environmental scalar $f_{\mathrm{env}}$, and litter
   input vector $\mathbf{I}$ [g C m$^{-2}$ s$^{-1}$],

   $$\mathbf{D} = f_{\mathrm{env}} K \mathbf{C}, \qquad \frac{\mathrm{d}\mathbf{C}}{\mathrm{d}t} = (T - I_d)\mathbf{D} + \mathbf{I}, \qquad R_{\mathrm{hetero}} = \mathbf{1}^{\mathsf{T}}(I_d - T)\mathbf{D},$$

   satisfying the exact open carbon ledger
   $\sum_i \Delta C_i + R_{\mathrm{hetero}}\Delta t - \sum_i I_i\Delta t = 0$.

2. **Pool-resolved first-order kinetics (`FirstOrderDecomposition`, `Level 2`)**:
   replaces scalar $f_{\mathrm{env}}K$ with pool-specific decay rates
   $D_i = k_i\,f_T(T)\,f_\theta(\theta)\,f_{q,i}\,C_i$ under $Q_{10}$ or
   Arrhenius thermal scaling and clay/lignin stabilization modifiers $f_{q,i}$,
   closing the same five-pool carbon and $R_{\mathrm{hetero}}$ ledger.

3. **Reverse Michaelis-Menten microbial-enzyme kinetics (`MicrobialEnzymeDecomposition`, `Level 2`)**:
   couples substrate pools $S_i$ [g C m$^{-2}$] to active microbial biomass
   $M_{\mathrm{mic}}$ [g C m$^{-2}$] with carbon-use efficiency $\mathrm{CUE} \in (0, 1)$:

   $$V_i = \frac{M_{\mathrm{mic}}\,V_{\max,i}(T, \theta)\,S_i}{K_{m,i}(T) + M_{\mathrm{mic}}}, \qquad \frac{\mathrm{d}M_{\mathrm{mic}}}{\mathrm{d}t} = \mathrm{CUE}\sum_i V_i - D_{\mathrm{mic}}, \qquad R_{\mathrm{microbial}} = (1 - \mathrm{CUE})\sum_i V_i + D_{\mathrm{mic}},$$

   where $D_{\mathrm{mic}} = (\tau_{\mathrm{turn}} + \tau_{\mathrm{maint}}f_T)M_{\mathrm{mic}}$
   so $\sum_i \Delta S_i + \Delta M_{\mathrm{mic}} + R_{\mathrm{microbial}}\Delta t = 0$
   prior to external litter addition.

### Litterfall and hydrological chain maps

`LitterfallInterfaceMapper` (`Level 2`) partitions plant litter and livestock
dung inputs onto metabolic and structural litter ports while preserving total C
and N mass:
$F_{\mathrm{met,C}} + F_{\mathrm{str,C}} = F_{\mathrm{plant,C}} + F_{\mathrm{dung,C}}$
and $F_{\mathrm{org,N}} = F_{\mathrm{plant,N}} + F_{\mathrm{dung,N}}$.
`PedosphereInterfaceMapper` (`Level 2`) projects hydrological boundary fluxes
(infiltration, evapotranspiration, drainage) onto pedosphere port names via the
identity chain map $\mathbf{y}_h = I_3\mathbf{x}_h$ (`REPLACE`) without
recomputing hydrological closures.

## Soil nitrogen, root P uptake, and C:N stoichiometry boundaries

### Mineralization, nitrification, denitrification, aggregation, and leaching

1. **Mineralization and immobilization (`MineralizationProcess`, `Level 2`)**:
   bounds gross organic-N mineralization and microbial $\mathrm{NH_4^+}$
   immobilization against available stocks over $\Delta t$:

   $$R_{\mathrm{gross}} = \min\!\left(k_{\min} N_{\mathrm{org}} f_{\mathrm{env}},\; \frac{N_{\mathrm{org}}}{\Delta t}\right), \qquad R_{\mathrm{immob}} = \min\!\left(\frac{\mathrm{CUE}\cdot \mathrm{DEC}}{\mathrm{CN}_{\mathrm{mic}}\,\eta_N},\; \frac{N_{\mathrm{NH_4}}}{\Delta t}\right),$$

   preserving $\Delta N_{\mathrm{org}} + \Delta N_{\mathrm{NH_4}} + \Delta N_{\mathrm{mic}} = 0$
   on internal transformations.

2. **Nitrification (`NitrificationProcess`, `Level 2`) and denitrification (`DenitrificationProcess`, `Level 2`)**:
   both consume `water_filled_pore_space` $W \in [0, 1]$ (Linn and Doran, 1984;
   Del Grosso et al., 2000) under documented RELIC piecewise-linear oxygen and
   anaerobic ramps ($W_{\mathrm{opt}} = W_{\mathrm{thr}} = 0.6$, minimum floor $0.1$):

   $$r_{\mathrm{nit}} = V_{\max,\mathrm{nit}}\,\frac{N_{\mathrm{NH_4}}}{K_{m,\mathrm{NH_4}} + N_{\mathrm{NH_4}}}\,f_T\,f_{\mathrm{pH}}\,f_{\mathrm{O_2}}(W), \qquad r_{\mathrm{denit}} = V_{\max,\mathrm{denit}}\,\frac{N_{\mathrm{NO_3}}}{K_{m,\mathrm{NO_3}} + N_{\mathrm{NO_3}}}\,f_{\mathrm{anaer}}(W)\,f_C\,f_T\,f_{\mathrm{pH}},$$

   closing local nitrogen mass ledgers
   $\Delta N_{\mathrm{NH_4}} + \Delta N_{\mathrm{NO_3}} + \Delta N_{\mathrm{N_2O,nit}} = 0$
   and $\Delta N_{\mathrm{NO_3}} + \Delta N_{\mathrm{N_2O,denit}} + \Delta N_{\mathrm{N_2}} = 0$.

3. **Diagnostic N aggregation (`NitrogenCycling`, `Level 2`) and retarded leaching (`NitrogenLeachingProcess`, `Level 3`)**:
   `NitrogenCycling` aggregates child process rates diagnostically without
   re-running child kinetics and commits only the external fertilizer-plus-urine
   ammonium increment $\Delta N_{\mathrm{NH_4,ext}} = 1000(F_{\mathrm{fert}} + F_{\mathrm{urine}})\Delta t$
   via `UpdateItem(ACCUMULATE, per_timestep=True)`. `NitrogenLeachingProcess`
   computes drainage-driven export from mobile retarded nitrate concentration
   $C_{\mathrm{NO_3,mob}} = \frac{N_{\mathrm{NO_3}}/(\theta z_s)}{1 + \rho_b K_d / \theta}$
   and water drainage $F_w$, bounded by pool availability and minimum depletion
   timescale so $\Delta N_{\mathrm{NO_3}} + F_{N,\mathrm{leach}}\Delta t = 0$.

### Root N and P uptake and C:N diagnostic scope

`relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake.RootNutrientUptake`
(`Level 2`) evaluates layered Michaelis-Menten root uptake
$U_i(z) = V_{\max,i}\frac{C_i(z)}{K_{m,i} + C_i(z)} f_T f_W W_{\mathrm{root}}(z)$
(Barber, 1995) for $\mathrm{NO_3^-}$, $\mathrm{NH_4^+}$, and dissolved
phosphorus. Molar plant nitrogen storage $S_N$ [mol N m$^{-2}$] and gram-basis
soil pools $m_{\mathrm{NO_3}}, m_{\mathrm{NH_4}}$ [g N m$^{-2}$] close via
$(\Delta m_{\mathrm{NO_3}} + \Delta m_{\mathrm{NH_4}})/M_N + \Delta S_N = 0$.
Realized molar phosphorus uptake rate $U_P$ [mol P m$^{-2}$ s$^{-1}$] is bounded
by a four-way minimum over kinetic uptake capacity $U_{P,\mathrm{kin}}$, plant
demand $D_P$, available labile soil phosphorus $P_{\mathrm{avail}}$ [g P m$^{-2}$],
and remaining plant P storage capacity $P_{\mathrm{cap,mol}}$ [mol P m$^{-2}$]:

$$
U_P = \min\!\left(U_{P,\mathrm{kin}},\; D_P,\; \frac{P_{\mathrm{avail}}}{M_P\,\Delta t},\; \frac{P_{\mathrm{cap,mol}}}{\Delta t}\right),
\qquad
\Delta P_{\mathrm{avail}} + \Delta P_{\mathrm{plant}} = 0.
$$

**Phosphorus and stoichiometry scope boundary**: `RootNutrientUptake` (together
with managed phosphate delivery from `FertilizationManagement` and foliar N:P
limitation in `NutrientStress`) constitutes the implemented `[I]` phosphorus
interface. Standalone multi-pool soil phosphorus weathering, occlusion,
mineralization, and P-leaching Functors remain `[N]`. Separately,
`relic.scientific.pedosphere.nutrient_cycling.stoichiometry.StoichiometryConstraint`
(`Level 1`, `ScaleCouplingFunctor`) is a read-only C:N flux-ratio and
mass-balance diagnostic observer ($\Delta \mathbf{x} = \mathbf{0}$); it neither
projects state onto a stoichiometric manifold nor audits phosphorus pools.

## Wetland methane request-and-coupler architecture

To prevent competing storage clamps across methanogenesis, methanotrophy, and
three-pathway gas escape, RELIC separates wetland $\mathrm{CH_4}$ into three
stateless tendency/request Functors, one prognostic budget owner, and one
diagnostic scale-routing observer:

1. **Production request (`MethaneProductionFunctor`, `Level 2`)**: blends the
   iMAPLE inundation branch (Yue et al., 2024, Eq. 34) and a redox-gated
   calibrated branch with gating weight $w \in [0, 1]$ and peaked temperature
   response $f_{ST}(T)$ (Zhu et al., 2014):

   $$P_{\mathrm{CH_4}} = (1 - w)\,R_h r f_{ST}(T) f_{\mathrm{pH}} f_{\mathrm{wet}} + w\,R_h r f_{ST}(T) f_{Eh}(Eh)\,A.$$

2. **Oxidation request (`MethaneOxidationFunctor`, `Level 2`)**: evaluates dual
   Michaelis-Menten kinetics in porewater $\mathrm{CH_4}$ and $\mathrm{O_2}$
   concentrations (Riley et al., 2011, Eq. 7; Oleson et al., 2013, Eq. 19.11),
   bounded by $M_{\mathrm{CH_4}}/\Delta t$.

3. **Three-pathway transport request (`MethaneTransportFunctor`, `Level 3`)**:
   sums aqueous-gaseous molecular diffusion $F_{\mathrm{diff}}$, threshold
   ebullition $F_{\mathrm{ebull}}$, and aerenchyma plant-mediated venting
   $F_{\mathrm{plant}}$ (Walter and Heimann, 2000) into
   $F_{\mathrm{req}} = F_{\mathrm{diff}} + F_{\mathrm{ebull}} + F_{\mathrm{plant}}$
   and $T_{\mathrm{trans}} = -F_{\mathrm{req}}$ (`REPLACE`).

4. **Single realization owner (`MethaneBudgetCoupler`, `Level 2`)**: integrates
   combined tendencies $\Delta M_{\mathrm{raw}} = (T_{\mathrm{prod}} + T_{\mathrm{oxid}} + T_{\mathrm{trans}})\Delta t$,
   enforces non-negativity once via $\Delta M = \max(\Delta M_{\mathrm{raw}}, -M_{\mathrm{CH_4}})$,
   commits $\Delta M$ to `ch4_soil_storage` (`ACCUMULATE`, `per_timestep=True`),
   and deducts any storage-limiter deficit $R_{\mathrm{lim}} = (\Delta M - \Delta M_{\mathrm{raw}})/\Delta t$
   from realized atmospheric emission $F_{\mathrm{atm}} = F_{\mathrm{req}} - R_{\mathrm{lim}}$
   so $\Delta M + (O - P_{\mathrm{CH_4}} + F_{\mathrm{atm}})\Delta t = 0$.
   `MethaneScaleCouplingFunctor` (`Level 1`) computes the bounded diagnostic
   routing weight $w_{\mathrm{DNDC}} = \operatorname{clip}(\alpha_{\mathrm{wet}}f_{\mathrm{wet}} + \alpha_{\mathrm{wfps}}W + \alpha_{\mathrm{data}}s_{\mathrm{data}} - \tau,\; 0,\; 1)$.

## Porous-media water, gas, solute, DAE, ALE, and memory transport

### Richards water, Millington-Quirk gas diffusion, and retarded solute transport

- `SoilWaterTransportFunctor` (`Level 3`) solves the 1D mixed Richards balance
  $\partial_t \theta = -\partial_z q + S$ with $q = -K(\theta)\partial_z(h + z)$
  in discrete cochain form $\dot{\boldsymbol{\theta}}_h = -D_h \mathbf{q}_h + G_h \mathbf{u}_h$
  ($\star_K = \star_K^{\mathsf{T}} \succeq 0$), emitting full next-state
  `soil_moisture` via `REPLACE`.
- `GasDiffusion1D` (`Level 3`) diffuses vadose $\mathrm{CO_2}$-C and
  $\mathrm{N_2O}$-N profiles under Millington and Quirk (1961) tortuosity above
  a no-flux lower boundary (`ACCUMULATE` concentration increments, `REPLACE`
  surface efflux).
- `SoluteAdvectionDiffusion1D` (`Level 3`) solves 1D retarded
  advection-dispersion $\partial_t\left((\theta + \rho_b K_d)C\right) = -\partial_z\left(q C - K_{\mathrm{disp}}\partial_z C\right)$
  for $\mathrm{NO_3^-}$-N and DOC-C, verifying
  $M^{n+1} - M^n - (F_{\mathrm{top}} - F_{\mathrm{bot}})\Delta t = 0$.

### Freundlich equilibrium-sorption index-1 descriptor DAE (`EquilibriumSorptionFunctor`, Level 2)

`relic.scientific.pedosphere.transport.equilibrium_sorption.EquilibriumSorptionFunctor`
(`Level 2`, `PhysicalFieldFunctor`) formulates nonlinear Freundlich equilibrium
sorption $S = K_f C^n$ ($n \in [0.3, 1.0]$) with first-order dissolved-phase
degradation rate $\lambda \ge 0$ (Jury and Horton, 2004; Mayes et al., 2012;
Hairer and Wanner, 1996) as a semi-explicit index-1 descriptor DAE
$E\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, $\mathbf{g}(\mathbf{x}) = \mathbf{0}$
on $\mathbf{x} = (y, C, S)^{\mathsf{T}}$ with $E = \operatorname{diag}(1, 0, 0)$:

$$
\frac{\mathrm{d}y}{\mathrm{d}t} = -\lambda\,\theta\,C,
\qquad
g_1(y, C, S) = \theta C + \rho_b S - y = 0,
\qquad
g_2(C, S) = S - K_f C^n = 0.
$$

Because $\partial_C(\theta C + \rho_b K_f C^n - y) = \theta + \rho_b K_f n C^{n-1} > 0$
for $\theta > 0$ and $C > 0$, the algebraic Jacobian is strictly positive
(differentiation index 1). At every step (including post-checkpoint restart
validation of the stored $(y, C, S)$ triple), implicit backward-Euler
elimination solves the strictly monotone scalar equation

$$
F(C') = \theta(1 + \lambda\,\Delta t)C' + \rho_b K_f (C')^n - y = 0,
\qquad
C' \in \left[0,\; \frac{y}{\theta}\right],
$$

via `relic.computation.solvers.nonlinear.BracketedBisectionSolver`, updates
differential total mass $y' = y - \lambda\,\theta\,C'\Delta t$ via
`UpdateItem(ACCUMULATE, per_timestep=True)`, and replaces algebraic states
$C'$ and $S' = K_f (C')^n$ (`REPLACE`), reporting the post-step manifold residual
$\lvert \theta C' + \rho_b S' - y' \rvert$.

### 1D ALE moving-domain diffusion and exact GCL (`MovingDomainDiffusionFunctor`, Level 3)

`relic.scientific.pedosphere.transport.moving_domain_diffusion.MovingDomainDiffusionFunctor`
(`Level 3`, `TransportMechanismFunctor`) solves Fickian solute diffusion on a
uniformly stretching or compressing 1D column $x(\xi, t) = \xi L(t)$,
$\xi \in [0, 1]$, driven by lower-boundary velocity $w_b$ [m s$^{-1}$] (Thomas
and Lombard, 1979; Donea et al., 2004). With $N$ uniform cells of thickness
$V_i(t) = \Delta x(t) = L(t)/N$ and face velocities $w_k = (k/N)w_b$
($k = 0, \dots, N$), the discrete Geometric Conservation Law (GCL) holds
identically:

$$
V_i(t + \Delta t) - V_i(t) = \Delta t\,(w_{i+1} - w_i) = \frac{w_b\,\Delta t}{N},
\qquad
L(t + \Delta t) = L(t) + w_b\,\Delta t.
$$

Subject to both diffusive Von Neumann and mesh-Courant ($\lvert w_b \rvert \Delta t / \Delta x \le \mathrm{CFL}_{\mathrm{mesh,max}}$)
bounds, cell solute mass $m_i = V_i c_i$ advances by zero-flux interior
diffusion plus upwind relative swept-face fluxes $s_k = c_{\mathrm{face},k} w_k$
(entraining exterior concentration $c_b$ when $w_b > 0$ and exporting bottom-cell
$c_{N-1}$ when $w_b < 0$):

$$
m_i(t + \Delta t) = m_i(t) + \Delta t\left(r_{\mathrm{diff},i}\,\Delta x(t) + s_{i+1} - s_i\right),
\qquad
c_i(t + \Delta t) = \frac{m_i(t + \Delta t)}{V_i(t + \Delta t)}.
$$

On the time-dependent Hodge metric $\star(t) = \operatorname{diag}(L(t)/N)$ with
quadratic storage $H = \frac{1}{2}\mathbf{c}^{\mathsf{T}}\star(t)\mathbf{c}$ and
$\dot{\star} = \operatorname{diag}(w_b / N)$, the moving-geometry power term
$-\frac{1}{2}\mathbf{c}^{\mathsf{T}}\dot{\star}\mathbf{c}$ is emitted explicitly
in `diagnostics.moving_domain.geometry_power` alongside `gcl_residual` and
`mass_balance_residual`.

### Mobile-immobile exact exponential memory exchange (`TwoRegionMassTransferFunctor`, Level 3)

`relic.scientific.pedosphere.transport.two_region_mass_transfer.TwoRegionMassTransferFunctor`
(`Level 3`, `TransportMechanismFunctor`) implements first-order mass transfer of
rate $\alpha > 0$ [s$^{-1}$] between mobile ($\theta_m, c_m$) and stagnant
immobile ($\theta_{\mathrm{im}}, c_{\mathrm{im}}$) pore-water regions (van
Genuchten and Wierenga, 1976; Haggerty and Gorelick, 1995). On the
$\theta$-weighted storage state $\mathbf{x} = (\theta_m c_m, \theta_{\mathrm{im}} c_{\mathrm{im}})^{\mathsf{T}}$
with quadratic free-energy proxy
$H(\mathbf{x}) = \frac{1}{2}(\theta_m c_m^2 + \theta_{\mathrm{im}} c_{\mathrm{im}}^2)$
and conjugate effort $\mathbf{e} = \frac{\partial H}{\partial \mathbf{x}} = (c_m, c_{\mathrm{im}})^{\mathsf{T}}$,
the dynamics form a purely dissipative system ($J_h = 0$):

$$
\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = -R_h\mathbf{e},
\qquad
R_h = \alpha
\begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix} \succeq 0,
\qquad
\frac{\mathrm{d}H}{\mathrm{d}t} = -\alpha(c_m - c_{\mathrm{im}})^2 \le 0.
$$

Eliminating $c_{\mathrm{im}}$ yields the passive exponential Volterra memory
kernel $K(t) = \alpha\omega\exp(-\omega t)$ with $\omega = \alpha / \theta_{\mathrm{im}}$.
Over each step $\Delta t$, `TwoRegionMassTransferFunctor` advances the linear
$2 \times 2$ exchange analytically via its exact matrix exponential
$c_m(t+\Delta t) - c_{\mathrm{im}}(t+\Delta t) = (c_m(t) - c_{\mathrm{im}}(t))\exp(-\kappa \Delta t)$
with $\kappa = \alpha(\theta_m^{-1} + \theta_{\mathrm{im}}^{-1})$, conserving
total pore solute $\theta_m \Delta c_m + \theta_{\mathrm{im}}\Delta c_{\mathrm{im}} = 0$
to machine precision and declaring `relic.core.functor.SubFlowAccuracy.EXACT` on
its state update.

## References

- Barber, S. A. (1995). *Soil Nutrient Bioavailability: A Mechanistic Approach* (2nd ed.). Wiley.
- Coleman, K. and Jenkinson, D. S. (1996). RothC-26.3 -- A model for the turnover of carbon in soil. In *Evaluation of Soil Organic Matter Models*, Springer, 237-246.
- Del Grosso, S. J. et al. (2000). General model for N2O and N2 gas emissions from soils due to denitrification. *Global Biogeochemical Cycles*, 14, 1045-1060. doi:10.1029/1999GB001225.
- Donea, J., Huerta, A., Ponthot, J.-Ph., and Rodriguez-Ferran, A. (2004). Arbitrary Lagrangian-Eulerian methods. In *Encyclopedia of Computational Mechanics*, Vol. 1, Ch. 14, Wiley.
- Haggerty, R. and Gorelick, S. M. (1995). Multiple-rate mass transfer for modeling diffusion and surface reactions in media with pore-scale heterogeneity. *Water Resources Research*, 31, 2383-2400.
- Hairer, E. and Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems* (2nd ed.). Springer.
- Jury, W. A. and Horton, R. (2004). *Soil Physics* (6th ed.). Wiley.
- Linn, D. M. and Doran, J. W. (1984). Effect of water-filled pore space on carbon dioxide and nitrous oxide production in tilled and nontilled soils. *Soil Science Society of America Journal*, 48, 1267-1272.
- Mayes, M. A., Heal, K. R., Brandt, C. C., Phillips, J. R., and Jardine, P. M. (2012). Relation between soil order and sorption of dissolved organic carbon in temperate subsoils. *Soil Science Society of America Journal*, 76, 199-206.
- Millington, R. J. and Quirk, J. P. (1961). Permeability of porous solids. *Transactions of the Faraday Society*, 57, 1200-1207.
- Oleson, K. W. et al. (2013). *Technical Description of version 4.5 of the Community Land Model (CLM)*. NCAR/TN-503+STR.
- Parton, W. J., Schimel, D. S., Cole, C. V., and Ojima, D. S. (1987). Analysis of factors controlling soil organic matter levels in Great Plains grasslands. *Soil Science Society of America Journal*, 51, 1173-1179.
- Riley, W. J. et al. (2011). Barriers to predicting changes in global terrestrial methane fluxes: analyses using CLM4Me, a methane biogeochemistry model integrated in CESM. *Biogeosciences*, 8, 1925-1953. doi:10.5194/bg-8-1925-2011.
- Simunek, J., van Genuchten, M. T., and Sejna, M. (2008). Development and applications of the HYDRUS and STANMOD software packages. *Vadose Zone Journal*, 7, 587-600. doi:10.2136/vzj2007.0077.
- Thomas, P. D. and Lombard, C. K. (1979). Geometric conservation law and its application to flow computations on moving grids. *AIAA Journal*, 17, 1030-1037.
- van Genuchten, M. T. and Wierenga, P. J. (1976). Mass transfer studies in sorbing porous media I. Analytical solutions. *Soil Science Society of America Journal*, 40, 473-480.
- Walter, B. P. and Heimann, M. (2000). A process-based, climate-sensitive model to derive methane emissions from natural wetlands. *Global Biogeochemical Cycles*, 14, 745-765.
- Wieder, W. R., Bonan, G. B., and Allison, S. D. (2013). Global soil carbon projections are improved by modelling microbial processes. *Nature Climate Change*, 3, 909-912.
- Yue, X. et al. (2024). Development and evaluation of the interactive Model for Air Pollution and Land Ecosystems (iMAPLE) version 1.0. *Geoscientific Model Development*, 17, 4621-4642. doi:10.5194/gmd-17-4621-2024.
- Zhu, Q., Liu, J., Peng, C., Chen, H., Fang, X., Jiang, H., Yang, G., Zhu, D., Wang, W., and Zhou, X. (2014). Modelling methane emissions from natural wetlands by development and application of the TRIPLEX-GHG model. *Geoscientific Model Development*, 7, 981-999.
