# Atmosphere Theory

## Scope, process inventory, and certificate map

The RELIC atmosphere domain (`relic.scientific.atmosphere`) supplies radiative
energy partitioning, Monin-Obukhov surface-layer turbulence, vertical and
three-dimensional boundary-layer tracer transport, parallel dry-deposition
resistance networks, open photostationary and closed constant-volume gas-phase
chemistry, and exact-replay stochastic weather generators (*Methods of
Ecological Analysis*, Chapters 17, 18, and 27). In addition,
`relic.coupling.observation` provides read-only `ScaleCouplingFunctor`
observation operators that project surface fluxes and layered mixing ratios onto
satellite column retrievals and eddy-covariance tower supports.

Eleven concrete scientific Functors under `relic.scientific.atmosphere` carry
twelve declarative `IPHSEquivalenceCertificate` files (eleven primary class
certificates plus one secondary `Level 5` constructor-conditioned certificate on
`ChemistryFunctor`):

| Functor | Canonical bridge | Cert. level | Status | Primary runtime role and state semantics |
| --- | --- | --- | --- | --- |
| {ref}`RadiationFunctor <class-relic-scientific-atmosphere-radiation-radiationfunctor>` | `PhysicalFieldFunctor` | `Level 2` | `[I]` | Shortwave, longwave, PAR/PPFD, and Beer-Lambert canopy APAR (`REPLACE`) |
| {ref}`RadiationField <class-relic-scientific-atmosphere-radiation-radiationfield>` | `PhysicalFieldFunctor` | `Level 3` | `[I]` | Discrete cochain radiation field interpolation, finite-difference gradient, and clipping residual (`REPLACE`) |
| {ref}`RadiationEnergyConservationValidator <class-relic-scientific-atmosphere-radiation-energy-conservation-radiationenergyconservationvalidator>` | `PhysicalFieldFunctor` | `Level 1` | `[I]` | Read-only radiative energy-balance residual observation (`REPLACE` diagnostics) |
| {ref}`TurbulenceFunctor <class-relic-scientific-atmosphere-turbulence-turbulencefunctor>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Monin-Obukhov similarity ($u_\ast$, $L$, $r_a$, $K_m$, $K_h$) and bulk aerodynamic fluxes (`REPLACE`) |
| {ref}`EddyDiffusionBLMFunctor <class-relic-scientific-atmosphere-transport-eddy-diffusion-blm-eddydiffusionblmfunctor>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | 1D vertical $K$-profile boundary-layer tracer diffusion (`ACCUMULATE`, `per_timestep=True`) |
| {ref}`AtmosphericAdvectionDiffusionFunctor <class-relic-scientific-atmosphere-transport-atmosphericadvectiondiffusionfunctor-atmosphericadvectiondiffusionfunctor>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | 3D finite-volume upwind advection and $K$-theory eddy diffusion (`ACCUMULATE`, `per_timestep=True`) |
| {ref}`EnvironmentalForcingFunctor <class-relic-scientific-atmosphere-environmental-forcing-environmentalforcingfunctor>` | `PhysicalFieldFunctor` | `Level 2` | `[I]` | Validated boundary-forcing projection for meteorological and trace-gas inputs (`REPLACE`) |
| {ref}`DryDepositionAggregator <class-relic-scientific-atmosphere-transport-dry-deposition-aggregator-drydepositionaggregator>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Wesely three-resistance dry-deposition velocity $v_d$ and surface sink flux (`REPLACE`) |
| {ref}`BoundaryLayerEmissionSource <class-relic-scientific-atmosphere-transport-boundary-layer-emission-source-boundarylayeremissionsource>` | `TransportMechanismFunctor` | `Level 1` | `[I]` | Well-mixed boundary-layer source of the surface isoprene flux (`ACCUMULATE`, `per_timestep=True`) |
| {ref}`ChemistryFunctor <class-relic-scientific-atmosphere-chemistry-chemistryfunctor>` | `PhysicalFieldFunctor` | `Level 3` | `[I]` | Open tropospheric $\mathrm{NO}$-$\mathrm{NO}_2$-$\mathrm{O}_3$ photostationary, BVOC oxidation, and $\mathrm{HO}_x$ chemistry (`REPLACE`) |
| {ref}`ChemistryFunctor (closed constant-volume) <class-atmosphere-chemistry-chemistry-functor-closed-constant-volume>` | `PhysicalFieldFunctor` | `Level 5` | `[I]` | Closed rigid adiabatic $\mathrm{NO} + \mathrm{O}_3 \rightarrow \mathrm{NO}_2 + \mathrm{O}_2$ strict GENERIC runtime point (`ACCUMULATE` / `REPLACE`) |
| {ref}`OrnsteinUhlenbeckTemperatureFunctor <class-relic-scientific-atmosphere-stochastic-temperature-ornsteinuhlenbecktemperaturefunctor>` | `PhysicalFieldFunctor` | `Level 2` | `[I]` | Ornstein-Uhlenbeck air-temperature anomaly SDE via `EulerMaruyamaSolver` (`ACCUMULATE`, `per_timestep=True`) |
| {ref}`MarkedPoissonRainfallFunctor <class-relic-scientific-atmosphere-stochastic-rainfall-markedpoissonrainfallfunctor>` | `PhysicalFieldFunctor` | `Level 2` | `[I]` | Compound Poisson-exponential storm generator via `GillespieSSA` (`ACCUMULATE`, `per_timestep=True`) |

General notation and conservation ledgers follow [Mathematical Foundations](foundations.md),
cochain and metriplectic brackets follow [DEC, IPHS, and GENERIC](dec_iphs_generic.md),
and time-stepping stability and stochastic replay rules follow [Numerical Methods](numerics.md).

## Solar, thermal, and canopy radiation

### Surface radiative budget and spectral PAR partitioning

`relic.scientific.atmosphere.radiation.RadiationFunctor` (`Level 2`) evaluates
downward and net shortwave and longwave irradiances [W m$^{-2}$] from potential
insolation $S_{\mathrm{pot}}$, cosine solar zenith angle $\mu_0 = \cos\theta_z$,
cloud fraction $f_c \in [0, 1]$, air and surface temperatures $T_a, T_s$ [K],
relative humidity $\mathrm{RH} \in [0, 100]$, surface albedo $\alpha \in [0, 1]$,
and surface emissivity $\varepsilon_s \in (0, 1]$ (*Methods of Ecological
Analysis*, Chapter 17):

$$
S_\downarrow = S_{\mathrm{pot}}\,\tau_c(f_c),
\qquad
S_{\mathrm{net}} = (1 - \alpha)\,S_\downarrow,
$$

$$
L_\downarrow = \varepsilon_a(e_a, T_a, f_c)\,\sigma_{\mathrm{SB}} T_a^4,
\qquad
L_\uparrow = \varepsilon_s\,\sigma_{\mathrm{SB}} T_s^4 + (1 - \varepsilon_s)\,L_\downarrow,
$$

$$
L_{\mathrm{net}} = L_\downarrow - L_\uparrow
= \varepsilon_s\left(L_\downarrow - \sigma_{\mathrm{SB}} T_s^4\right),
\qquad
R_n = S_{\mathrm{net}} + L_{\mathrm{net}},
$$

where $\sigma_{\mathrm{SB}}$ is fetched from `relic.utils.physical_constants`
(`get_physical_constant("stefan_boltzmann")`) and clear-sky atmospheric
emissivity follows Brutsaert (1975), $\varepsilon_{a,\mathrm{clear}} = c_B\,(e_a / T_a)^{1/7}$,
modulated by cloud cover. Photosynthetically active radiation (PAR) and
photosynthetic photon flux density ($\mathrm{PPFD}$, [$\mu$mol m$^{-2}$ s$^{-1}$])
are split into direct-beam and diffuse components:

$$
I_{\mathrm{PAR}} = f_{\mathrm{PAR}}\,S_\downarrow,
\qquad
\mathrm{PPFD} = \kappa_{Q}\,I_{\mathrm{PAR}},
\qquad
\mathrm{PPFD}_{\mathrm{diff}} = f_{\mathrm{diff}}(\mu_0, f_c)\,\mathrm{PPFD},
\qquad
\mathrm{PPFD}_{\mathrm{dir}} = \mathrm{PPFD} - \mathrm{PPFD}_{\mathrm{diff}}.
$$

When `leaf_area_index` ($L_{\mathrm{AI}} \ge 0$) is present, canopy interception
follows Beer-Lambert extinction with direct and diffuse extinction coefficients
$k_b(\mu_0)$ and $k_d$:

$$
\tau_{\mathrm{dir}} = \exp\!\left(-k_b(\mu_0)\,L_{\mathrm{AI}}\right),
\qquad
\tau_{\mathrm{diff}} = \exp\!\left(-k_d\,L_{\mathrm{AI}}\right),
\qquad
f_{\mathrm{PAR,abs}} = 1 - \frac{I_{\mathrm{PAR,dir}}\tau_{\mathrm{dir}} + I_{\mathrm{PAR,diff}}\tau_{\mathrm{diff}}}{\max(I_{\mathrm{PAR}}, \varepsilon_I)}.
$$

Any positivity clamp applied to noisy shortwave or emissivity inputs is recorded
explicitly in `diagnostics.radiation.shortwave_clipping_residual` and
`diagnostics.radiation.emissivity_clipping_residual`.

### Cochain radiation fields and radiative balance validation

`relic.scientific.atmosphere.radiation.RadiationField` (`Level 3`) treats
gridded irradiance as a discrete scalar field on `grid_coordinates`, supporting
bilinear or trilinear query projection onto `source_coordinates` and discrete
gradient evaluation $g_h = d_h I_h$ while reporting domain-clamp residuals in
`diagnostics.radiation_field.clipping_residual`.

`relic.scientific.atmosphere.radiation_energy_conservation.RadiationEnergyConservationValidator`
(`Level 1`) is a pure diagnostic observer ($\Delta x_h = 0$) verifying the
pointwise radiative closure residual

$$
r_{\mathrm{rad}} = R_n - \left[(1 - \alpha)S_\downarrow + L_\downarrow - L_\uparrow\right]
$$

against the configured tolerance without overwriting any radiative flux channel.

## Surface-layer turbulence and boundary-layer tracer transport

### Monin-Obukhov similarity and aerodynamic resistance

`relic.scientific.atmosphere.turbulence.TurbulenceFunctor` (`Level 3`) closes
constant-flux surface-layer momentum and scalar exchange using Monin-Obukhov
similarity theory (Monin and Obukhov, 1954; Businger et al., 1971; Dyer, 1974).
For reference height $z_r$, displacement height $d_0$, aerodynamic and thermal
roughness lengths $z_{0m}, z_{0h}$, von Karman constant $\kappa$, and virtual
potential temperature $\theta_v$, the Obukhov length $L$ and dimensionless
stability parameter $\zeta = (z_r - d_0)/L$ satisfy

$$
L = -\frac{\rho_a c_p \theta_v u_\ast^3}{\kappa g H_v},
\qquad
u_\ast = \frac{\kappa U(z_r)}{\ln\!\left(\frac{z_r - d_0}{z_{0m}}\right) - \Psi_m(\zeta) + \Psi_m\!\left(\frac{z_{0m}}{L}\right)},
$$

$$
r_a = \frac{\ln\!\left(\frac{z_r - d_0}{z_{0h}}\right) - \Psi_h(\zeta) + \Psi_h\!\left(\frac{z_{0h}}{L}\right)}{\kappa u_\ast},
\qquad
K_m(z) = \frac{\kappa u_\ast (z - d_0)}{\phi_m(\zeta)},
\qquad
K_h(z) = \frac{\kappa u_\ast (z - d_0)}{\phi_h(\zeta)},
$$

where $\phi_m, \phi_h$ and their integrated stability corrections $\Psi_m, \Psi_h$
follow Businger-Dyer unstable ($\zeta < 0$) and linear stable ($\zeta \ge 0$)
branches. Sensible and latent heat fluxes $H = \rho_a c_p (\theta_s - \theta_a)/r_a$
and $\lambda E$ are emitted as instantaneous `REPLACE` boundary fields.

### Vertical K-profile and 3D advection-diffusion transport

`relic.scientific.atmosphere.transport.eddy_diffusion_blm.EddyDiffusionBLMFunctor`
(`Level 3`) advances 1D vertical boundary-layer tracer concentrations $c(z,t)$
across a boundary layer of height $h_{\mathrm{BL}}$ with profile diffusivity
(Troen and Mahrt, 1986; Holtslag and Boville, 1993):

$$
K_z(z) = \kappa\,w_s\,z\left(1 - \frac{z}{h_{\mathrm{BL}}}\right)^p,
\qquad
0 \le z \le h_{\mathrm{BL}},
$$

discretized in conservative finite-volume cochain form

$$
\frac{\mathrm{d}c_h}{\mathrm{d}t} = -D_h q_h + G_h u_h,
\qquad
q_h = -\star_{K_z} d_h c_h,
\qquad
\star_{K_z} = \star_{K_z}^{\mathsf{T}} \succeq 0,
$$

where surface emissions and dry-deposition sinks enter only through the boundary
port $G_h u_h$.

`relic.scientific.atmosphere.transport.AtmosphericAdvectionDiffusionFunctor.AtmosphericAdvectionDiffusionFunctor`
(`Level 3`) generalizes tracer transport to three-dimensional structured grids
with wind components $(u, v, w)$ and anisotropic eddy diffusivities
$\mathbf{K} = \operatorname{diag}(K_x, K_y, K_z)$:

$$
\frac{\partial c}{\partial t} + \nabla \cdot (\mathbf{v} c - \mathbf{K}\nabla c) = s_{\mathrm{ext}},
$$

using conservative upwind advective face fluxes and centered diffusive face
fluxes. Both transport Functors emit integrated per-step concentration
increments via `UpdateItem(ACCUMULATE, per_timestep=True)` and verify discrete
column or volume mass closure against boundary fluxes.

### Environmental forcing projection and Wesely dry deposition

`relic.scientific.atmosphere.environmental_forcing.EnvironmentalForcingFunctor`
(`Level 2`) validates physical bounds, unit consistency, and finiteness of
prescribed meteorological and trace-gas driver arrays before projecting them
into `StateManager` via `REPLACE`.

`relic.scientific.atmosphere.transport.dry_deposition_aggregator.DryDepositionAggregator`
(`Level 3`) computes species-resolved dry-deposition velocities $v_{d,s}$
[m s$^{-1}$] and downward surface removal fluxes $F_{\mathrm{dep},s} = v_{d,s} c_s(z_r)$
using the Wesely (1989) series-parallel resistance network (aerodynamic
resistance $r_a$, quasi-laminar sublayer resistance $r_{b,s} = \frac{2}{\kappa u_\ast}(\mathrm{Sc}_s / \mathrm{Pr})^{2/3}$,
and bulk canopy/surface resistance $r_{c,s}$ combining stomatal, cuticular,
mesophyll, and ground pathways in parallel):

$$
v_{d,s} = \frac{1}{r_a + r_{b,s} + r_{c,s}} + v_{\mathrm{grav},s},
\qquad
F_{\mathrm{dep},s} = v_{d,s}\,c_s(z_r) \ge 0.
$$

Because $r_a, r_{b,s}, r_{c,s} > 0$, conductance $v_{d,s}$ is strictly positive
and defines a passive linear boundary sink ($\star_{v_d} \succeq 0$) without
directly mutating atmospheric concentration storage; the selected atmospheric
transport owner consumes $F_{\mathrm{dep},s}$ once.

### Surface emission into the well-mixed boundary layer

`relic.scientific.atmosphere.transport.boundary_layer_emission_source.BoundaryLayerEmissionSource`
(`Level 1`) hands the surface isoprene molar flux $F$ [mol m$^{-2}$ s$^{-1}$,
ground area] over to the well-mixed boundary layer of height $h$ and air molar
density $n = P/(R\,T)$:

$$
\frac{d\chi}{dt} = \frac{F}{h\,n},
\qquad
\Delta\chi = \frac{F\,\Delta t\,R\,T}{P\,h}.
$$

The increment is evaluated by the existing
`SpeciesMapper.flux_to_mixing_ratio`, is submitted once through
`UpdateItem(ACCUMULATE, per_timestep=True)`, and closes the column molar ledger
$\Delta\chi\,h\,n = F\,\Delta t$ pointwise.

## Gas-phase chemistry: open tropospheric kinetics and closed Level-5 GENERIC point

### Open tropospheric photostationary and BVOC-HOx mode (Level 3)

In its default open-atmosphere configuration,
`relic.scientific.atmosphere.chemistry.ChemistryFunctor` (`Level 3`) evaluates
tropospheric $\mathrm{NO}$-$\mathrm{NO}_2$-$\mathrm{O}_3$ photostationary
kinetics coupled to biogenic VOC ($\mathrm{C_5H_8}$, monoterpene, and
sesquiterpene) oxidation by $\mathrm{OH}$, $\mathrm{O}_3$, and $\mathrm{NO}_3$
(*Methods of Ecological Analysis*, Chapter 18; Leighton, 1961; Atkinson et al.,
2006; Warneck and Williams, 2012). Key photolysis and thermal channels include:

$$
\mathrm{NO}_2 + h\nu \xrightarrow{J_{\mathrm{NO}_2}} \mathrm{NO} + \mathrm{O}(^3\mathrm{P}),
\qquad
\mathrm{O}(^3\mathrm{P}) + \mathrm{O}_2 + \mathrm{M} \longrightarrow \mathrm{O}_3 + \mathrm{M},
\qquad
\mathrm{NO} + \mathrm{O}_3 \xrightarrow{k_3(T)} \mathrm{NO}_2 + \mathrm{O}_2,
$$

with Arrhenius rate coefficient $k_3(T) = A_3 \exp(-E_3 / (R T))$ and spectral
quantum-yield temperature interpolation for $J_{\mathrm{NO}_2}$ and
$J_{\mathrm{O}(^1\mathrm{D})}$. In photostationary equilibrium without peroxy
radical perturbation, the Leighton ratio satisfies

$$
\Phi_{\mathrm{PSS}} = \frac{J_{\mathrm{NO}_2}[\mathrm{NO}_2]}{k_3(T)[\mathrm{NO}][\mathrm{O}_3]} = 1,
\qquad
[\mathrm{O}_3]_{\mathrm{PSS}} = \frac{J_{\mathrm{NO}_2}[\mathrm{NO}_2]}{k_3(T)[\mathrm{NO}]}.
$$

Because actinic photon driving ($J_{\mathrm{NO}_2} > 0$), external BVOC sources,
and open boundary conditions supply external free energy, the default mode is
certified at `Level 3` (stoichiometric reaction-operator factorization
$\dot{\mathbf{c}} = \mathbf{N}\mathbf{r}(\mathbf{c}, T, J)$ with verified $\mathrm{NO}_x$
and oxidant accounting, without claiming closed thermodynamic GENERIC degeneracy).

### Closed constant-volume adiabatic GENERIC runtime point (Level 5)

Under the secondary constructor-conditioned certificate
`constructor_selection = {physical_boundary: "closed_constant_volume"}`
({ref}`ChemistryFunctor closed constant-volume <class-atmosphere-chemistry-chemistry-functor-closed-constant-volume>`,
`Level 5`), `ChemistryFunctor` restricts execution to an isolated, rigid,
constant-volume ($V$ = `box_volume` [m$^3$]) adiabatic gas cell containing
dark thermal titration

$$
\mathrm{NO} + \mathrm{O}_3 \xrightarrow{k_3(T)} \mathrm{NO}_2 + \mathrm{O}_2
$$

with zero actinic flux ($J = 0$) and zero matter or heat ports. Let
$\mathbf{x} = (n_{\mathrm{NO}}, n_{\mathrm{NO}_2}, n_{\mathrm{O}_3}, n_{\mathrm{O}_2}, U)^{\mathsf{T}}$
denote the molar inventories $\mathbf{n}$ [mol] (`box_no_amount`,
`box_no2_amount`, `box_o3_amount`, `box_o2_amount`) and total NASA7 internal
energy $U$ [J] (`box_internal_energy`), with extended stoichiometric vector
$\tilde{\boldsymbol{\nu}} = (\boldsymbol{\nu}^{\mathsf{T}}, 0)^{\mathsf{T}} = (-1, +1, -1, +1, 0)^{\mathsf{T}}$.
On the strictly positive thermodynamic interior ($n_i > 0$, $T > 0$, partial
pressures $p_i = n_i R T / V$, NASA7 chemical potentials
$\mu_i = h_i(T) - T s_i(T) + R T \ln(p_i / p_0)$), the energy functional
$E(\mathbf{x}) = U$ and concave entropy $S(\mathbf{n}, U)$ yield conjugate
effort covectors

$$
\mathbf{e}_E = \frac{\partial E}{\partial \mathbf{x}} = (\mathbf{0}^{\mathsf{T}}, 1)^{\mathsf{T}},
\qquad
\mathbf{e}_S = \frac{\partial S}{\partial \mathbf{x}} = \left(-\frac{\boldsymbol{\mu}(\mathbf{n}, T)}{T}^{\mathsf{T}},\; \frac{1}{T}\right)^{\mathsf{T}},
$$

with positive reaction affinity $\mathcal{A} = -\boldsymbol{\nu}^{\mathsf{T}}\boldsymbol{\mu} > 0$
and forward molar rate $r > 0$ (Grmela and Ottinger, 1997; Mielke, 2011).
Advancement through `relic.computation.operators.iphs.advance_reaction_network`
satisfies the canonical GENERIC evolution ($L_h = 0$)

$$
\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t}
= L_h\mathbf{e}_E + M_h\mathbf{e}_S
= \tilde{\boldsymbol{\nu}}\,r,
\qquad
M_h(\mathbf{x}) = \frac{r\,T}{\mathcal{A}}\,\tilde{\boldsymbol{\nu}}\,\tilde{\boldsymbol{\nu}}^{\mathsf{T}} \succeq 0,
$$

with exact energy degeneracy $M_h(\mathbf{x})\mathbf{e}_E = \mathbf{0}$
($\Delta U = 0$), exact atomic nitrogen and oxygen conservation
($\Delta(n_{\mathrm{NO}} + n_{\mathrm{NO}_2}) = 0$,
$\Delta(n_{\mathrm{NO}} + 2n_{\mathrm{NO}_2} + 3n_{\mathrm{O}_3} + 2n_{\mathrm{O}_2}) = 0$),
and non-negative entropy production
$\sigma_S = \mathbf{e}_S^{\mathsf{T}} M_h(\mathbf{x})\mathbf{e}_S = \mathcal{A} r / T \ge 0$.
This `Level 5` certificate applies exclusively to the closed constant-volume
configuration and never transfers to photolytic or open-boundary runs.

## Stochastic atmospheric forcing and replay contracts

Both stochastic weather Functors inherit
`relic.core.functor.stochastic_substream.FunctorSubstreamMixin` (`[I]`), binding
an isolated PyTorch `Generator` substream whose byte state is persisted across
checkpoint and restart so trajectories replay bit-identically (*Methods of
Ecological Analysis*, Chapter 27). Step durations are converted to days via
$\Delta t_{\mathrm{days}} = \Delta t / 86400$.

### Ornstein-Uhlenbeck air-temperature anomaly

`relic.scientific.atmosphere.stochastic_temperature.OrnsteinUhlenbeckTemperatureFunctor`
(`Level 2`) models mean-reverting synoptic temperature anomalies $X_t$ [K] via
the Ito stochastic differential equation

$$
\mathrm{d}X_t = -\alpha X_t\,\mathrm{d}t_{\mathrm{days}} + \sigma\,\mathrm{d}W_{t_{\mathrm{days}}},
\qquad
\alpha > 0,\quad \sigma \ge 0,
$$

discretized over $\Delta t_{\mathrm{days}}$ by
`relic.stochastic.sde_solver.EulerMaruyamaSolver`:

$$
\Delta X_n = -\alpha X_n\,\Delta t_{\mathrm{days}} + \sigma\sqrt{\Delta t_{\mathrm{days}}}\,\xi_n,
\qquad
\xi_n \sim \mathcal{N}(0, 1).
$$

`stochastic_air_temperature_anomaly` is updated via
`UpdateItem(ACCUMULATE, per_timestep=True)`, while step drift, diffusion scale,
and Wiener increments are emitted as `REPLACE` diagnostics.

### Marked Poisson storm rainfall generator

`relic.scientific.atmosphere.stochastic_rainfall.MarkedPoissonRainfallFunctor`
(`Level 2`) generates intermittent storm arrivals of constant hazard rate
$\lambda > 0$ [day$^{-1}$] and independent exponential storm depths of mean
$\alpha > 0$ [mm] (Rodriguez-Iturbe et al., 1999). Exact inter-arrival sampling
over $\Delta t_{\mathrm{days}}$ is performed by
`relic.stochastic.gillespie.GillespieSSA` subject to a fail-closed per-cell
arrival cap $K_{\max} = \lceil \mu + 12\sqrt{\mu} + 64 \rceil$ ($\mu = \lambda \Delta t_{\mathrm{days}}$).
For $k = \Delta N_n$ arrivals with unit exponential variates $E_i \sim \operatorname{Exp}(1)$
(reparameterized as $z_i = \alpha E_i$ so pathwise $\partial \Delta H_n / \partial \alpha$
preserves PyTorch autograd), the cumulative event count and rainfall depth
increments are

$$
\Delta N_n = k,
\qquad
\Delta H_n = \alpha \sum_{i=1}^{k} E_i,
$$

with predictable Doob-Meyer compensators

$$
\widetilde{A}_{N,n} = \lambda\,\Delta t_{\mathrm{days}},
\qquad
\widetilde{A}_{H,n} = \lambda\,\alpha\,\Delta t_{\mathrm{days}},
$$

emitted in `diagnostics.stochastic_rainfall.compensator_count` and
`diagnostics.stochastic_rainfall.compensator_depth` so martingale innovation
residuals can be audited directly.

## Atmospheric observation operators

Read-only atmospheric observation Functors live in `relic.coupling.observation`
outside the 90 certified `relic.scientific.*` process classes. All five inherit
`ScaleCouplingFunctor` at `ExecutionPhase.POSTPROCESSING` and emit
`VariableKind.DIAGNOSTIC` outputs with `StateWriteMode.DERIVED` ($\Delta \mathbf{S} = 0$):

1. **Steady-state HCHO column yield (`HchoSteadyStateColumnOperator`)**:
   `relic.coupling.observation.hcho_steady_state_column.HchoSteadyStateColumnOperator`
   maps surface isoprene molar emission $E$ [mol m$^{-2}$ s$^{-1}$] (with
   stoichiometric carbon factor $n_C = 5$) and tropospheric $\mathrm{NO}_2$
   vertical column $\Omega_{\mathrm{NO}_2}$ [molecule cm$^{-2}$] to steady-state
   formaldehyde vertical column density $\Omega_{\mathrm{HCHO}}$ (Palmer et al.,
   2003; Marais et al., 2012):

   $$\Omega_{\mathrm{HCHO}} = S\!\left(\Omega_{\mathrm{NO}_2}\right) n_C E_{\mathrm{molec}} + B, \qquad S\!\left(\Omega_{\mathrm{NO}_2}\right) = \begin{cases} m\,\Omega_{\mathrm{NO}_2} + c, & \Omega_{\mathrm{NO}_2} \le \Omega_t,\\ S_{\mathrm{high}}, & \Omega_{\mathrm{NO}_2} > \Omega_t, \end{cases}$$

   where $E_{\mathrm{molec}}$ converts mol m$^{-2}$ s$^{-1}$ to molecule cm$^{-2}$ s$^{-1}$
   via `avogadro`, $\Omega_t = 1.0 \times 10^{15}$ molecule cm$^{-2}$, and $B$
   is background column density. Inputs below the empirical calibration bound
   $\Omega_{\mathrm{low}} = 2.0 \times 10^{14}$ molecule cm$^{-2}$ are evaluated
   unclipped along the linear branch while flagging `no2_below_calibrated_range = True`.

2. **Satellite averaging-kernel column projection (`SatelliteColumnKernelOperator`)**:
   `relic.coupling.observation.satellite_column_kernel.SatelliteColumnKernelOperator`
   constructs hybrid-sigma pressure centers $p_k = a_k + b_k p_s$ and arithmetic
   or logarithmic interior layer edges ($p_{e,k} = \frac{1}{2}(p_{k-1}+p_k)$ or
   $\sqrt{p_{k-1}p_k}$), computes hydrostatic dry-air and tracer partial columns

   $$n_k = \frac{p_{e,k} - p_{e,k+1}}{g\,M_{\mathrm{air}}}, \qquad x_{\mathrm{pc},k} = \chi_k\,n_k,$$

   conservatively remaps model partial columns onto retrieval pressure layers via
   1D pressure-interval overlap (`remap_partial_columns`, recording any
   out-of-range mass in `unmapped_column`), and applies the retrieval averaging
   kernel $A_l$ across the target vertical domain $T$ (Rodgers, 2000; Eskes and
   Boersma, 2003):

   $$N_{v,\mathrm{model}} = \sum_{l \in T} A_l\,x_{\mathrm{pc},l}^{\mathrm{retr}}.$$

3. **Tower footprint, profile, and post-observation quality operators**:
   `relic.coupling.observation.tower_flux.TowerFluxSupportOperator` projects
   gridded surface fluxes onto normalized eddy-covariance footprint weights;
   `relic.coupling.observation.tower_concentration.TowerConcentrationProfileOperator`
   interpolates vertical mixing-ratio profiles to tower intake heights; and
   `relic.coupling.observation.quality.PostObservationQualityOperator` evaluates
   retrieval validity masks and support-coverage flags without altering model
   prognostic state.

## References

- Atkinson, R. et al. (2006). Evaluated kinetic and photochemical data for atmospheric chemistry: Volume II -- gas phase reactions of organic species. *Atmospheric Chemistry and Physics*, 6, 3625-4055. doi:10.5194/acp-6-3625-2006.
- Brutsaert, W. (1975). On a derivable formula for long-wave radiation from clear skies. *Water Resources Research*, 11, 742-744. doi:10.1029/WR011i005p00742.
- Businger, J. A., Wyngaard, J. C., Izumi, Y., and Bradley, E. F. (1971). Flux-profile relationships in the atmospheric surface layer. *Journal of the Atmospheric Sciences*, 28, 181-189.
- Dyer, A. J. (1974). A review of flux-profile relationships. *Boundary-Layer Meteorology*, 7, 363-372.
- Eskes, H. J. and Boersma, K. F. (2003). Averaging kernels for DOAS total-column satellite retrievals. *Atmospheric Chemistry and Physics*, 3, 1285-1291. doi:10.5194/acp-3-1285-2003.
- Grmela, M. and Ottinger, H. C. (1997). Dynamics and thermodynamics of complex fluids. I. Development of a general formalism. *Physical Review E*, 56, 6620-6632.
- Holtslag, A. A. M. and Boville, B. A. (1993). Local versus nonlocal boundary-layer diffusion in a global climate model. *Journal of Climate*, 6, 1825-1842.
- Leighton, P. A. (1961). *Photochemistry of Air Pollution*. Academic Press.
- Marais, E. A. et al. (2012). Isoprene emissions in Africa inferred from OMI observations of formaldehyde columns. *Atmospheric Chemistry and Physics*, 12, 6219-6235. doi:10.5194/acp-12-6219-2012.
- Mielke, A. (2011). Formulation of thermoelastic dissipative material behavior using GENERIC. *Continuum Mechanics and Thermodynamics*, 23, 233-256.
- Monin, A. S. and Obukhov, A. M. (1954). Basic laws of turbulent mixing in the surface layer of the atmosphere. *Trudy Geofizicheskogo Instituta AN SSSR*, 24(151), 163-187.
- Palmer, P. I. et al. (2003). Mapping isoprene emissions over North America using formaldehyde column observations from space. *Journal of Geophysical Research*, 108(D6), 4180. doi:10.1029/2002JD002153.
- Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding: Theory and Practice*. World Scientific.
- Rodriguez-Iturbe, I., Porporato, A., Ridolfi, L., Isham, V., and Cox, D. R. (1999). Probabilistic modelling of water balance at a point: the role of climate, soil and vegetation. *Proceedings of the Royal Society of London A*, 455, 3789-3805.
- Troen, I. B. and Mahrt, L. (1986). A simple model of the atmospheric boundary layer; sensitivity to surface evaporation. *Boundary-Layer Meteorology*, 37, 129-148.
- Warneck, P. and Williams, J. (2012). *The Atmospheric Chemist's Companion: Numerical Data for Use in the Atmospheric Sciences*. Springer. doi:10.1007/978-94-007-2275-0.
- Wesely, M. L. (1989). Parameterization of surface resistances to gaseous dry deposition in regional-scale numerical models. *Atmospheric Environment*, 23, 1293-1304.
