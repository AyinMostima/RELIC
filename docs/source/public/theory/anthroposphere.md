# Anthroposphere Theory

## Scope, process inventory, and fail-closed boundary status

The RELIC anthroposphere domain (`relic.scientific.anthroposphere`) governs
anthropogenic emission sector splitting and chemical-mechanism speciation,
continuous-time Markov land-use fraction transitions and area-weighted land-use
parameter observations, multi-resource management coordination, and agricultural
or disturbance interventions (irrigation, fertilization, harvest, livestock
grazing, prescribed/wildland fire, and tillage) (*Methods of Ecological
Analysis*, Chapter 24).

Fourteen classes in `relic.scientific.anthroposphere` carry declarative
`IPHSEquivalenceCertificate` files. RELIC enforces a strict distinction between
the **eleven implemented (`[I]`)** scientific Functors (`2` at `Level 1`, `3` at
`Level 2`, and `6` at `Level 3`) and the **three fail-closed (`[N]`, `Level 0`)**
boundary placeholders whose runtime interconnection contracts remain unbound:

| Subsystem | Functor | Canonical bridge | Cert. level | Status | Primary runtime role and state semantics |
| --- | --- | --- | --- | --- | --- |
| Fail-closed boundaries | {ref}`EmissionInventory <class-relic-scientific-anthroposphere-inventory-emissioninventory>` | `ScaleCouplingFunctor` | `Level 0` | `[N]` | Unbound inventory ingestion boundary; rejects at construction and `forward()` (`runtime_dependency_unbound`) |
| Fail-closed boundaries | {ref}`AtmosphereCoupling <class-relic-scientific-anthroposphere-emissions-coupling-to-atmosphere-atmospherecoupling>` | `TransportMechanismFunctor` | `Level 0` | `[N]` | Unbound vertical atmospheric injection boundary; rejects fail-closed (`runtime_dependency_unbound`) |
| Fail-closed boundaries | {ref}`AnthroposphereInterfaceMapping <class-relic-scientific-anthroposphere-interface-mapping-anthroposphereinterfacemapping>` | `ScaleCouplingFunctor` | `Level 0` | `[N]` | Unbound cross-sphere interface placeholder; rejects fail-closed (`runtime_dependency_unbound`) |
| Emissions & speciation | {ref}`SectorAllocation <class-relic-scientific-anthroposphere-emissions-sector-allocation-sectorallocation>` | `ScaleCouplingFunctor` | `Level 1` | `[I]` | Mass-conserving sector split of gridded inventory fluxes (`REPLACE`) |
| Emissions & speciation | {ref}`SpeciesMapper <class-relic-scientific-anthroposphere-emissions-species-mapping-speciesmapper>` | `ScaleCouplingFunctor` | `Level 3` | `[I]` | CB05 molar gas speciation vs mass-basis aerosol split with explicit unmapped ledger (`REPLACE`) |
| Land-use dynamics | {ref}`LandUseTransitions <class-relic-scientific-anthroposphere-land-use-transitions-landusetransitions>` | `ScaleCouplingFunctor` | `Level 3` | `[I]` | Continuous-time Markov matrix-exponential evolution of 9 land-use fractions (`REPLACE`) |
| Land-use dynamics | {ref}`LandUseParameterMapping <class-relic-scientific-anthroposphere-land-use-mapping-landuseparametermapping>` | `PhysicalFieldFunctor` | `Level 1` | `[I]` | Read-only area-fraction-weighted surface property observation (`REPLACE`) |
| Management & disturbance | {ref}`ManagementCoordinator <class-relic-scientific-anthroposphere-management-coordinator-managementcoordinator>` | `ScaleCouplingFunctor` | `Level 3` | `[I]` | Unit-normalized resource rationing and priority observation (`REPLACE`) |
| Management & disturbance | {ref}`IrrigationManagement <class-relic-scientific-anthroposphere-management-irrigation-irrigationmanagement>` | `TransportMechanismFunctor` | `Level 2` | `[I]` | Moisture-triggered root-zone irrigation, conveyance efficiency, and WEAP21 quota scaling (`REPLACE`) |
| Management & disturbance | {ref}`FertilizationManagement <class-relic-scientific-anthroposphere-management-fertilization-fertilizationmanagement>` | `TransportMechanismFunctor` | `Level 2` | `[I]` | Step-overlap-invariant elemental N and $\mathrm{P_2O_5}\rightarrow\mathrm{P}$ application ports (`REPLACE`) |
| Management & disturbance | {ref}`HarvestManagement <class-relic-scientific-anthroposphere-management-harvest-harvestmanagement>` | `BiologicalProcessFunctor` | `Level 3` | `[I]` | Aboveground harvest export, residue return, and root-mortality partitioning (`ACCUMULATE` / `REPLACE`) |
| Management & disturbance | {ref}`GrazingManagement <class-relic-scientific-anthroposphere-management-grazing-grazingmanagement>` | `BiologicalProcessFunctor` | `Level 3` | `[I]` | Linear body-mass livestock intake, dung/urine C and N partitioning, and trampling (`ACCUMULATE` / `REPLACE`) |
| Management & disturbance | {ref}`FireManagement <class-relic-scientific-anthroposphere-management-fire-management-firemanagement>` | `BiologicalProcessFunctor` | `Level 3` | `[I]` | Stoichiometric fuel combustion, trace-gas/aerosol emission factors, and ash ledger (`ACCUMULATE` / `REPLACE`) |
| Management & disturbance | {ref}`TillageManagement <class-relic-scientific-anthroposphere-management-tillage-tillagemanagement>` | `TransportMechanismFunctor` | `Level 2` | `[I]` | Soil disturbance decay, residue incorporation index, and empirical $\mathrm{CO_2}$ pulse (`REPLACE`) |

Because anthropogenic harvesting, irrigation withdrawal, fertilizer import, and
industrial emissions are externally driven open-system interventions, no
anthroposphere class claims `Level 4` or `Level 5` closed thermodynamic GENERIC
structure. General ledger conventions follow [Mathematical Foundations](foundations.md)
and structural classifications follow [DEC, IPHS, and GENERIC](dec_iphs_generic.md).

## Fail-closed Level-0 boundaries vs implemented emission operators

### Fail-closed boundary classes (`[N]`, Level 0)

`EmissionInventory`, `AtmosphereCoupling`, and `AnthroposphereInterfaceMapping`
are registered in the governance catalog at `Level 0` with empty runtime port
bindings (`implementation_status = "not_implemented_fail_closed"`). Instantiating
or invoking `forward()` on any of these three classes raises an explicit
`` `[N]` `` `runtime_dependency_unbound` error without fabricating proxy weights,
plume-rise heights, or silent zero fluxes. Preprocessed gridded emission fields
therefore enter active workflows as validated external forcings consumed directly
by `SectorAllocation` and `SpeciesMapper`.

### Conservative sector allocation (`SectorAllocation`, Level 1)

For species $s$ and grid support $i$ with total anthropogenic mass flux
$E_{s,i}$ [kg m$^{-2}$ s$^{-1}$], sector priors $\pi_{q,s} \ge 0$, and spatial
activity proxies $P_{q,i} \ge 0$ (Crippa et al., 2020; McDuffie et al., 2020),
`relic.scientific.anthroposphere.emissions.sector_allocation.SectorAllocation`
normalizes sector weights pointwise:

$$
W_{q,s,i} = \frac{\pi_{q,s}\,P_{q,i}}{\sum_r \pi_{r,s}\,P_{r,i}},
\qquad
E_{q,s,i} = W_{q,s,i}\,E_{s,i},
\qquad
R_{s,i} = E_{s,i} - \sum_q E_{q,s,i}.
$$

Zero or non-finite normalization denominators fail closed without defaulting
to uniform sector shares, and $R_{s,i}$ audits floating-point partition closure.

### Mechanism speciation: molar gas vs mass aerosol basis (`SpeciesMapper`, Level 3)

`relic.scientific.anthroposphere.emissions.species_mapping.SpeciesMapper`
projects sector-resolved inventory mass-flux vectors $\mathbf{u}_q$
[kg m$^{-2}$ s$^{-1}$] onto target chemical-mechanism species (such as CB05 gas
lumping plus primary aerosol classes; Seinfeld and Pandis, 2016) while keeping
gas-phase molar fluxes and aerosol mass fluxes on separate typed channels:

$$
\mathbf{F}_{\mathrm{gas,mol}} = \star_{\mathrm{MW}^{-1}}\,S_{\mathrm{gas}}\,\mathbf{u}_q
\quad [\mathrm{mol\,m^{-2}\,s^{-1}}],
\qquad
\mathbf{F}_{\mathrm{aer,mass}} = S_{\mathrm{aer}}\,\mathbf{u}_q
\quad [\mathrm{kg\,m^{-2}\,s^{-1}}],
$$

where $S_{\mathrm{gas}}$ and $S_{\mathrm{aer}}$ hold non-negative sector split
fractions and $\star_{\mathrm{MW}^{-1}}$ divides each gas channel by its
SSOT-backed molecular weight. Any inventory mass fraction not assigned to active
mechanism species is routed explicitly to $\mathbf{u}_{\mathrm{unmapped}}$ so
the mapping residual

$$
\mathbf{R}_{\mathrm{map}} = \mathbf{u}_q - S_{\mathrm{covered}}\,\mathbf{u}_q - \mathbf{u}_{\mathrm{unmapped}}
$$

vanishes within configured closure tolerance. Gas molar fluxes and aerosol mass
fluxes are never summed across incompatible units.

## Continuous-time Markov land-use transitions and parameter projection

`relic.scientific.anthroposphere.land_use.transitions.LandUseTransitions`
(`Level 3`) evolves the nine-class land-use area-fraction row vector
$\mathbf{p}(t) \in \Delta^8$ ($\mathbf{p}\ge \mathbf{0}$, $\mathbf{p}\mathbf{1} = 1$;
Hurtt et al., 2020) via a continuous-time Markov generator matrix $Q$ [yr$^{-1}$]
satisfying $Q_{ij} \ge 0$ for $i \neq j$ and zero row sums $Q\mathbf{1} = \mathbf{0}$:

$$
\mathbf{p}(t + \Delta t) = \mathbf{p}(t)\,\exp\!\left(Q\,\Delta t_{\mathrm{yr}}\right).
$$

Because $Q\mathbf{1} = \mathbf{0}$ and off-diagonal rates are non-negative, the
matrix exponential $\exp(Q\,\Delta t_{\mathrm{yr}})$ is row-stochastic for every
$\Delta t_{\mathrm{yr}} \ge 0$, preserving simplex non-negativity and exact
total-area conservation $\mathbf{p}(t + \Delta t)\mathbf{1} = \mathbf{p}(t)\mathbf{1} = 1$
in a single `REPLACE` step update. Optional socio-economic rate scaling
modulates $Q_{ij} = s'\,Q_{ij}^{\mathrm{base}}$ ($i \neq j$) while recomputing
$Q_{ii} = -\sum_{j \neq i} Q_{ij}$ so row-sum degeneracy is never broken.

`relic.scientific.anthroposphere.land_use.mapping.LandUseParameterMapping`
(`Level 1`, `PhysicalFieldFunctor`) computes read-only area-weighted surface
properties $\overline{\theta} = \sum_{i=1}^{9} p_i \theta_i$ (`REPLACE`) without
mutating vegetation, soil, or hydrological prognostic stores.

## Management coordination and agricultural water-nutrient ports

### Multi-resource management coordination (`ManagementCoordinator`, Level 3)

`relic.scientific.anthroposphere.management.coordinator.ManagementCoordinator`
observes natural state deficits and management flux requests to publish
normalized priority weights $p_i = q_i / \sum_j q_j$ ($q_i = w_i(1 + s_i)$,
$w_i \ge 0$) and dimensionless resource-availability scaling $\eta \in [0, 1]$
from unit-normalized demand ratios $c = \sum_r R_r / Q_r$. Water, labor, fuel,
and machinery quantities are normalized by their respective quota bases $Q_r$
before aggregation and are never added across unlike physical dimensions;
`ManagementCoordinator` never invokes child management Functors directly.

### Irrigation withdrawal, conveyance efficiency, and WEAP21 quota rationing (`IrrigationManagement`, Level 2)

`relic.scientific.anthroposphere.management.irrigation.IrrigationManagement`
(`Level 2`, `TransportMechanismFunctor`) evaluates demand-triggered agricultural
water withdrawal on irrigated cropland cells ($c_{\mathrm{lu},i} = 1$) when
volumetric soil moisture $\theta_i$ falls below
$\theta_{\mathrm{trig},i} = \theta_{\mathrm{wp},i} + f_{\mathrm{trig}}(\theta_{\mathrm{fc},i} - \theta_{\mathrm{wp},i})$
(Allen et al., 1998). With root-zone depth $Z_r$, liquid water density $\rho_w$
(`get_physical_constant("water_density")`), precipitation mass flux $F_{P,i}$,
system efficiency $\eta_i \in (0, 1]$ selected by `irrigation_type` (`1`: surface,
`2`: sprinkler, `3`: drip, `4`: subsurface), and maximum gross mass-flux rate
$F_{\max}$, pre-quota gross withdrawal mass $A_i$ [kg m$^{-2}$] is

$$
M_{\mathrm{net},i} = \max\!\left[\max(\theta_{\mathrm{fc},i} - \theta_i, 0)\,Z_r \rho_w - F_{P,i}\,\Delta t,\; 0\right],
\qquad
A_i = u_i\,\min\!\left(\frac{M_{\mathrm{net},i}}{\eta_i},\; F_{\max}\,\Delta t\right),
$$

where $u_i = \mathbf{1}_{\theta_i < \theta_{\mathrm{trig},i}}\,\mathbf{1}_{c_{\mathrm{lu},i} = 1}$.
Converting cell withdrawal to depth via $h_w = 1000 / \rho_w$ [mm m$^2$ kg$^{-1}$]
yields total requested domain quota depth $D = \sum_i A_i h_w$ [mm]. Against
scalar `irrigation_remaining_quota` $Q \ge 0$ [mm], single-priority WEAP21
equal-percentage rationing (Yates et al., 2005) scales all active cells
proportionally by $s \in [0, 1]$:

$$
s =
\begin{cases}
1, & D \le Q,\\
\dfrac{Q}{D}, & D > Q,
\end{cases}
\qquad
M_{\mathrm{gross},i} = s\,A_i,
\qquad
M_{\mathrm{deliv},i} = \eta_i\,M_{\mathrm{gross},i}.
$$

Step fluxes $F_{\mathrm{abs},i} = M_{\mathrm{gross},i}/\Delta t$
(`abstraction_flux`), $F_{\mathrm{irr},i} = M_{\mathrm{deliv},i}/\Delta t$
(`irrigation_flux`), and depleted quota $Q' = \max(Q - \sum_i M_{\mathrm{gross},i} h_w, 0)$
are emitted via `REPLACE`, enforcing the conveyance ledger
$\lvert F_{\mathrm{irr},i} - \eta_i F_{\mathrm{abs},i} \rvert \le \varepsilon_w$
without directly mutating `soil_moisture`.

### Step-overlap-invariant fertilization (`FertilizationManagement`, Level 2)

`relic.scientific.anthroposphere.management.fertilization.FertilizationManagement`
converts scheduled fertilizer product applications $A_e$ [kg m$^{-2}$] of
duration $\tau_e$ with elemental nitrogen fraction $f_{N,e}$ and labeled
phosphate fraction $f_{\mathrm{P_2O_5},e}$ into elemental mass inputs using
SSOT atomic masses $\chi_P = 2 M_P / (2 M_P + \frac{5}{2} M_{\mathrm{O_2}})$:

$$
\Delta M_N = \sum_e A_e f_{N,e}\,\frac{\omega_e}{\tau_e},
\qquad
\Delta M_P = \sum_e A_e f_{\mathrm{P_2O_5},e}\chi_P\,\frac{\omega_e}{\tau_e},
\qquad
F_{\mathrm{fert},N} = \frac{\Delta M_N}{\Delta t},
\qquad
F_{\mathrm{fert},P} = \frac{1000\,\Delta M_P}{\Delta t},
$$

where $\omega_e$ is the exact temporal intersection of $[t, t + \Delta t]$ with
event window $e$, guaranteeing cumulative mass invariance under arbitrary
time-step refinement.

## Managed biomass removal, livestock grazing, fire, and tillage

1. **Harvest export and residue partitioning (`HarvestManagement`, `Level 3`)**:
   partitions harvested aboveground biomass into exported economic yield
   $F_{\mathrm{harv}} = f_{\mathrm{rem}} B_{\mathrm{ag}} / \Delta t$, surface
   residue return $F_{\mathrm{res}} = (1 - f_{\mathrm{rem}}) B_{\mathrm{ag}} / \Delta t$,
   carbon export $F_{C,\mathrm{exp}} = f_C F_{\mathrm{harv}}$, and associated
   root mortality without losing biomass to unlogged clipping.

2. **Linear body-mass livestock grazing (`GrazingManagement`, `Level 3`)**:
   evaluates daily areal dry-matter intake demand $I_{\mathrm{area}}$
   [kg m$^{-2}$ day$^{-1}$] linearly in animal body weight $W$ [kg] with daily
   intake percentage $a$ [% body weight day$^{-1}$], forage digestibility/quality
   $Q$, selectivity $s_Q$, Hodgson (1990) asymptotic sward availability
   $1 - \exp(-k\cdot 10^4 B_{\mathrm{ag}})$, and stocking density $D$ [animals ha$^{-1}$]:

   $$I_{\mathrm{area}} = \frac{W\,a}{100}\,Q\,s_Q\left(1 - \exp\!\left(-10^4 k B_{\mathrm{ag}}\right)\right)\frac{D}{10^4}, \qquad F_{\mathrm{rem}} = \frac{\min\!\left(I_{\mathrm{area}},\; U_{\max} B_{\mathrm{ag}},\; B_{\mathrm{ag}}\,t_{\mathrm{day}}/\Delta t\right)}{t_{\mathrm{day}}}.$$

   Ingested forage carbon and nitrogen partition across livestock retention/export,
   dung return, and urine return via stoichiometric incidence $N_h \mathbf{r}_h$
   so $F_{C,\mathrm{in}} = F_{C,\mathrm{dung}} + F_{C,\mathrm{exp}}$ and
   $F_{N,\mathrm{in}} = F_{N,\mathrm{dung}} + F_{N,\mathrm{urine}} + F_{N,\mathrm{exp}}$
   close explicitly alongside moisture-sensitive trampling damage.

3. **Stoichiometric fire disturbance (`FireManagement`, `Level 3`)**:
   combusts fuel load $B_{\mathrm{burn}} = f_{\mathrm{burn}} B_{\mathrm{fuel}}$
   and maps burned dry matter through species emission factors $\mathrm{EF}_s$
   [g kg$^{-1}$] ($F_s = 10^{-3} B_{\mathrm{burn}}\,\mathrm{EF}_s / t_{\mathrm{day}}$),
   ash deposition, and unburned mortality residue within one closed `Level 3`
   reaction-port ledger.

4. **Tillage disturbance and empirical CO2 pulse (`TillageManagement`, `Level 2`)**:
   evaluates exponential post-tillage decay $\exp(-k_{\mathrm{dec}} t_{\mathrm{since}})$,
   moisture-modulated residue incorporation fraction $R_{\mathrm{inc}} \in [0, 1]$,
   and the phenomenological Reicosky (1997) soil $\mathrm{CO_2}$ efflux pulse
   (`REPLACE`) without directly mutating pedosphere carbon pools.

## References

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56.
- Arneth, A. et al. (2017). Historical carbon dioxide emissions caused by land-use changes are possibly larger than assumed. *Nature Geoscience*, 10, 79-84.
- Conant, R. T., Paustian, K., and Elliott, E. T. (2001). Grassland management and conversion into grassland: effects on soil carbon. *Ecological Applications*, 11, 343-355.
- Crippa, M. et al. (2020). High resolution temporal profiles in the Emissions Database for Global Atmospheric Research. *Earth System Science Data*, 12, 1939-1959.
- Hodgson, J. (1990). *Grazing Management: Science into Practice*. Longman Scientific & Technical.
- Hoesly, R. M. et al. (2018). Historical (1750-2014) anthropogenic emissions of reactive gases and aerosols from the Community Emissions Data System (CEDS). *Geoscientific Model Development*, 11, 369-408. doi:10.5194/gmd-11-369-2018.
- Hurtt, G. C. et al. (2020). Harmonization of global land-use change and management for the period 850-2100 (LUH2) for CMIP6. *Geoscientific Model Development*, 13, 5425-5464. doi:10.5194/gmd-13-5425-2020.
- McDuffie, E. E. et al. (2020). A global anthropogenic emission inventory of atmospheric pollutants from sector- and fuel-specific sources (1970-2017): an application of the Community Emissions Data System (CEDS). *Earth System Science Data*, 12, 3413-3442.
- Milchunas, D. G. and Lauenroth, W. K. (1993). Quantitative effects of grazing on vegetation and soils over a global range of environments. *Ecological Monographs*, 63, 327-366.
- Reicosky, D. C. (1997). Tillage-induced CO2 emission from soil. *Nutrient Cycling in Agroecosystems*, 49, 273-285.
- Seinfeld, J. H. and Pandis, S. N. (2016). *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change* (3rd ed.). Wiley.
- Yates, D., Sieber, J., Purkey, D., and Huber-Lee, A. (2005). WEAP21 -- A demand-, priority-, and preference-driven water planning model: Part 1: Model characteristics. *Water International*, 30, 487-500.
