# Hydrosphere Theory

## Scope, process inventory, and certificate map

The RELIC hydrosphere domain (`relic.scientific.hydrosphere`) governs land-surface
evapotranspiration, snowpack mass-energy balance and snow-aging albedo, van
Genuchten soil-water retention, single-bucket and vertical Richards vadose-zone
moisture dynamics, infiltration- and saturation-excess surface runoff, hybrid
event-triggered Hortonian surface ponding, subsurface lateral stormflow,
vadose-aquifer percolation and capillary exchange, and unconfined groundwater
storage (*Methods of Ecological Analysis*, Chapter 19).

Fourteen concrete scientific Functors reside in `relic.scientific.hydrosphere`,
each paired with an `IPHSEquivalenceCertificate` spanning `Level 1` through
`Level 3` (`2` at `Level 1`, `2` at `Level 2`, and `10` at `Level 3`). A related
vadose-zone solver,
{ref}`SoilWaterTransportFunctor <class-relic-scientific-pedosphere-transport-soilwatertransportfunctor-soilwatertransportfunctor>`
(`Level 3`), resides in `relic.scientific.pedosphere.transport` and is
contrasted below to enforce single-writer `soil_moisture` governance.

| Subsystem | Functor | Canonical bridge | Cert. level | Status | Primary runtime role and state semantics |
| --- | --- | --- | --- | --- | --- |
| Evapotranspiration | {ref}`PenmanMonteithET <class-relic-scientific-hydrosphere-evapotranspiration-penman-monteith-penmanmonteithet>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Aerodynamic-radiative-stomatal latent heat and ET partition (`REPLACE`) |
| Evapotranspiration | {ref}`PriestleyTaylorET <class-relic-scientific-hydrosphere-evapotranspiration-priestley-taylor-priestleytayloret>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Equilibrium-radiation $\alpha_{\mathrm{PT}}$ potential and stress-scaled ET (`REPLACE`) |
| Snowpack & albedo | {ref}`SnowPack <class-relic-scientific-hydrosphere-snow-snowpack-snowpack>` | `PhysicalFieldFunctor` | `Level 3` | `[I]` | Snow water equivalent (SWE), cold-content energy ledger, melt, and refreezing (`ACCUMULATE` / `REPLACE`) |
| Snowpack & albedo | {ref}`SnowAlbedo <class-relic-scientific-hydrosphere-snow-albedo-snowalbedo>` | `PhysicalFieldFunctor` | `Level 1` | `[I]` | Diagnostic snow-age and temperature-decay broadband albedo observation (`REPLACE`) |
| Soil water & retention | {ref}`SoilWaterRetention <class-relic-scientific-hydrosphere-soil-moisture-retention-soilwaterretention>` | `PhysicalFieldFunctor` | `Level 1` | `[I]` | Read-only van Genuchten $\theta \mapsto \psi(\theta)$ matric-potential map [MPa] (`REPLACE`) |
| Soil water & retention | {ref}`BucketSoilMoisture <class-relic-scientific-hydrosphere-soil-moisture-bucket-bucketsoilmoisture>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Root-zone single-reservoir water balance with saturation overflow and drainage (`ACCUMULATE` / `REPLACE`) |
| Soil water & retention | {ref}`RichardsSoilMoisture <class-relic-scientific-hydrosphere-soil-moisture-richards-richardssoilmoisture>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Multi-layer 1D mixed-form Richards unsaturated flow (`ACCUMULATE` / `REPLACE`) |
| Surface runoff & ponding | {ref}`SurfaceRunoff <class-relic-scientific-hydrosphere-runoff-surface-surfacerunoff>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | SCS curve-number, Hortonian, saturation-excess, and combined surface partitioning (`REPLACE`) |
| Surface runoff & ponding | {ref}`HortonianPondingEventFunctor <class-relic-scientific-hydrosphere-runoff-surface-ponding-hortonianpondingeventfunctor>` | `PhysicalFieldFunctor` | `Level 2` | `[I]` | Step rainfall-excess computation and hybrid ponding onset/cessation `ExternalIntent` (`REPLACE`) |
| Surface runoff & ponding | {ref}`SurfacePondingFunctor <class-relic-scientific-hydrosphere-runoff-surface-ponding-surfacepondingfunctor>` | `TransportMechanismFunctor` | `Level 2` | `[I]` | Exact linear-reservoir overland ponding depth $h$ and runoff recession (`ACCUMULATE` / `REPLACE`) |
| Subsurface & groundwater | {ref}`SubsurfaceRunoff <class-relic-scientific-hydrosphere-runoff-subsurface-subsurfacerunoff>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Perched/lateral interflow conductance drainage (`REPLACE`) |
| Subsurface & groundwater | {ref}`VerticalPercolationTransport <class-relic-scientific-hydrosphere-transport-vertical-percolation-verticalpercolationtransport>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Gravity-driven free-drainage percolation from root zone to aquifer (`ACCUMULATE` / `REPLACE`) |
| Subsurface & groundwater | {ref}`CapillaryRiseTransport <class-relic-scientific-hydrosphere-transport-capillary-rise-capillaryrisetransport>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Water-table-to-vadose suction-gradient capillary upflow (`ACCUMULATE` / `REPLACE`) |
| Subsurface & groundwater | {ref}`GroundwaterDynamics <class-relic-scientific-hydrosphere-groundwater-dynamics-groundwaterdynamics>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Unconfined aquifer storage, specific-yield water-table head, and baseflow (`ACCUMULATE` / `REPLACE`) |

General conservation rules follow [Mathematical Foundations](foundations.md),
discrete cochain operators follow [DEC, IPHS, and GENERIC](dec_iphs_generic.md),
and hybrid event semantics follow [Numerical Methods](numerics.md).

## Evapotranspiration closures

### Penman-Monteith combination equation

`relic.scientific.hydrosphere.evapotranspiration.penman_monteith.PenmanMonteithET`
(`Level 3`) combines surface energy balance with aerodynamic and surface-resistance
vapor transfer (Monteith, 1965; Allen et al., 1998). With available radiative
energy $A_e = R_n - G$ [W m$^{-2}$], saturation vapor-pressure slope
$\Delta = \frac{\mathrm{d}e_s}{\mathrm{d}T}(T_a)$ [Pa K$^{-1}$], psychrometric
constant $\gamma = \frac{c_p P}{\varepsilon_{\mathrm{mw}}\lambda_v}$, vapor
pressure deficit $D = \max(e_s(T_a) - e_a, 0)$ [Pa], aerodynamic resistance
$r_a$, and bulk surface resistance $r_s$, latent heat flux $\lambda_v E$
[W m$^{-2}$] and mass evapotranspiration flux $E_{\mathrm{ET}}$
[kg m$^{-2}$ s$^{-1}$] satisfy

$$
\lambda_v E = \frac{\Delta(R_n - G) + \rho_a c_p D / r_a}{\Delta + \gamma\left(1 + r_s / r_a\right)},
\qquad
E_{\mathrm{ET}} = \frac{\lambda_v E}{\lambda_v}.
$$

Canopy interception evaporation, plant transpiration, and bare-soil evaporation
are partitioned consistently with canopy cover fraction and root-zone moisture
availability without mutating soil water directly.

### Priestley-Taylor equilibrium-radiation closure

`relic.scientific.hydrosphere.evapotranspiration.priestley_taylor.PriestleyTaylorET`
(`Level 3`) evaluates equilibrium evaporation scaled by the Priestley and Taylor
(1972) advection coefficient $\alpha_{\mathrm{PT}}$ (typically $1.26$ for
well-watered surfaces) and soil-moisture stress scalar $\beta_W \in [0, 1]$:

$$
\lambda_v E_{\mathrm{pot}} = \alpha_{\mathrm{PT}}\,\frac{\Delta}{\Delta + \gamma}\,(R_n - G),
\qquad
E_{\mathrm{ET}} = \beta_W\,\frac{\max(\lambda_v E_{\mathrm{pot}}, 0)}{\lambda_v}.
$$

In any composed simulation graph, either `PenmanMonteithET` or
`PriestleyTaylorET` is selected as the sole writer of `evapotranspiration_flux`.

## Snowpack mass-energy balance and snow albedo

`relic.scientific.hydrosphere.snow.snowpack.SnowPack` (`Level 3`) maintains a
coupled mass-and-cold-content ledger for snow water equivalent $W_{\mathrm{snow}}$
[kg m$^{-2}$] and snow internal thermal state (Anderson, 1976; Tarboton and
Luce, 1996). Total precipitation $P$ is partitioned by wet-bulb or air
temperature thresholds into snowfall $P_{\mathrm{snow}}$ and liquid rain
$P_{\mathrm{rain}}$ ($P = P_{\mathrm{snow}} + P_{\mathrm{rain}}$). Given net
snowpack heat input $Q_{\mathrm{net}}$ [W m$^{-2}$] and latent heat of fusion
$L_f$ (`get_physical_constant("latent_heat_fusion")`), potential melt
$M_{\mathrm{pot}} = \max(Q_{\mathrm{net}}, 0)/L_f$ is bounded by available solid
storage and liquid-water holding capacity, satisfying the exact per-step mass
ledger

$$
\Delta W_{\mathrm{snow}} = \left(P_{\mathrm{snow}} + F_{\mathrm{refreeze}} - M_{\mathrm{snow}} - E_{\mathrm{sub}}\right)\Delta t,
$$

where $\Delta W_{\mathrm{snow}}$ is committed via
`UpdateItem(ACCUMULATE, per_timestep=True)` and `snow_melt_flux` ($M_{\mathrm{snow}}$)
and `rain_flux` ($P_{\mathrm{rain}}$) are emitted as `REPLACE` boundary fluxes.

`relic.scientific.hydrosphere.snow.albedo.SnowAlbedo` (`Level 1`) is a
read-only observation map computing broadband snow surface albedo
$\alpha_{\mathrm{snow}} \in [\alpha_{\min}, \alpha_{\max}]$ from snow age and
near-freezing metamorphism decay rates without modifying `W_{\mathrm{snow}}`.

## Soil-water retention, bucket storage, and Richards vadose flow

### van Genuchten retention observation (`SoilWaterRetention`, Level 1)

`relic.scientific.hydrosphere.soil_moisture.retention.SoilWaterRetention`
(`Level 1`) evaluates the equilibrium van Genuchten (1980) constitutive curve
mapping volumetric water content $\theta$ [m$^3$ m$^{-3}$] to soil matric
potential $\psi$ [MPa] without advancing water storage:

$$
S_e = \operatorname{clip}\!\left(\frac{\theta - \theta_r}{\theta_s - \theta_r},\; S_{e,\min},\; 1\right),
\qquad
h(S_e) = -\frac{1}{\alpha_{\mathrm{VG}}}\left(S_e^{-1/m} - 1\right)^{1/n},
\qquad
m = 1 - \frac{1}{n},
$$

$$
\psi = 10^{-6}\,\rho_w g\,h(S_e) \le 0\quad [\mathrm{MPa}].
$$

`soil_water_potential` is emitted via `REPLACE` for downstream plant-water-stress
and stomatal consumers.

### Root-zone bucket closure and 1D Richards unsaturated flow

`relic.scientific.hydrosphere.soil_moisture.bucket.BucketSoilMoisture` (`Level 3`)
evolves single-layer root-zone storage $S_w = \rho_w Z_r \theta$ [kg m$^{-2}$]
under infiltration $I$, evapotranspiration extraction $E_{\mathrm{ET}}$, gravity
drainage $Q_{\mathrm{drain}}(\theta)$, and saturation capacity $S_{w,\max} = \rho_w Z_r \theta_s$
(Manabe, 1969):

$$
S_{w,\mathrm{raw}} = S_w + \left(I - E_{\mathrm{ET}} - Q_{\mathrm{drain}}\right)\Delta t,
\qquad
Q_{\mathrm{sat,ex}} = \frac{\max(S_{w,\mathrm{raw}} - S_{w,\max},\; 0)}{\Delta t},
$$

so accepted storage increment $\Delta S_w$, drainage, and saturation-excess
overflow close to machine precision:
$\Delta S_w - (I - E_{\mathrm{ET}} - Q_{\mathrm{drain}} - Q_{\mathrm{sat,ex}})\Delta t = 0$.

`relic.scientific.hydrosphere.soil_moisture.richards.RichardsSoilMoisture`
(`Level 3`) solves the multi-layer vertical mixed-form Richards equation
(Richards, 1931; Celia et al., 1990) with Mualem-van Genuchten unsaturated
hydraulic conductivity $K(S_e) = K_{\mathrm{sat}} S_e^{\ell}\left[1 - (1 - S_e^{1/m})^m\right]^2$:

$$
\frac{\partial \theta}{\partial t} = -\frac{\partial q_z}{\partial z} - s_{\mathrm{root}}(z),
\qquad
q_z = -K(\theta)\left(\frac{\partial h(\theta)}{\partial z} + 1\right),
$$

in discrete cochain form $\dot{\boldsymbol{\theta}}_h = -D_h \mathbf{q}_h + G_h \mathbf{u}_h$
with symmetric positive-semidefinite face-conductance operator
$\star_K = \star_K^{\mathsf{T}} \succeq 0$.

**Single-writer ownership rule**: `BucketSoilMoisture`, `RichardsSoilMoisture`,
and pedosphere
`relic.scientific.pedosphere.transport.SoilWaterTransportFunctor.SoilWaterTransportFunctor`
(`Level 3`, which emits full next-state `soil_moisture` via `REPLACE`) are
mutually exclusive owners of volumetric soil moisture. A valid simulator graph
registers at most one soil-moisture state writer and routes cross-sphere fluxes
through `PedosphereInterfaceMapper`.

## Surface runoff and hybrid event-driven Hortonian ponding

### Surface runoff partitioning and liquid-input modes

`relic.scientific.hydrosphere.runoff.surface.SurfaceRunoff` (`Level 3`)
partitions liquid surface water input rate $P_{\mathrm{in}}$ [kg m$^{-2}$ s$^{-1}$]
into overland runoff $Q_{\mathrm{surf}}$ and soil infiltration $I$ satisfying
exact flux conservation:

$$
P_{\mathrm{in}} = Q_{\mathrm{surf}} + I,
\qquad
Q_{\mathrm{surf}} \ge 0,\quad I \ge 0.
$$

To prevent snowpack double counting when coupled with `SnowPack`, `SurfaceRunoff`
requires an explicit `surface_water_input_source` selection:

- `"precipitation_flux"`: snow-free mode reading total `precipitation_flux`
  ($P_{\mathrm{in}} = P$);
- `"rain_flux_plus_snow_melt_flux"`: snow-coupled mode reading liquid throughfall
  plus snowmelt ($P_{\mathrm{in}} = P_{\mathrm{rain}} + M_{\mathrm{snow}}$).

Supported constitutive partitioning closures include USDA-SCS Curve Number
event abstraction, Hortonian infiltration-capacity excess
$Q_{\mathrm{surf}} = \max(P_{\mathrm{in}} - f_c(\theta), 0)$, Dunne
saturation-excess runoff over variable contributing area $f_{\mathrm{sat}}(\theta)$,
and combined infiltration- plus saturation-excess routing.

### Hybrid Hortonian ponding event detection and overland recession

When transient storm-scale ponding topology transitions are resolved explicitly
(for example alongside `MarkedPoissonRainfallFunctor`), RELIC pairs two `Level 2`
Functors in `relic.scientific.hydrosphere.runoff.surface_ponding` (*Methods of
Ecological Analysis*, Chapters 19 and 27):

1. **Guard evaluation and topology intent (`HortonianPondingEventFunctor`, `Level 2`)**:
   Reads cumulative `stochastic_rainfall_depth` $H_n$ [mm] and its own
   replay-critical module extra state $H_{n-1}$ (persisted through PyTorch
   `get_extra_state` / `set_extra_state` for deterministic checkpoint/restart)
   to form the step rainfall depth increment $\Delta H_n = H_n - H_{n-1}$ and
   Hortonian infiltration-excess depth

   $$E_n = \max\!\left(0,\; \Delta H_n - I_{\max}\,\frac{\Delta t}{3600}\right)\quad [\mathrm{mm}],$$

   where $I_{\max}$ [mm hr$^{-1}$] is maximum infiltration capacity. Given active
   ponding indicator $a \in \{0, 1\}$ and current ponded depth $h_n$ [mm], hysteretic
   threshold guards ($E_{\mathrm{on}} > 0$, $h_{\mathrm{off}} \ge 0$) emit typed
   `relic.core.state.types.ExternalIntent` topology requests (`"ponding_onset"`
   when $a = 0$ and $\max E_n \ge E_{\mathrm{on}}$; `"ponding_cessation"` when
   $a = 1$, $\max E_n = 0$, and $\max h_n \le h_{\mathrm{off}}$), which the
   simulator commits between steps via `Simulator.apply_topology_transition`.

2. **Exact linear-reservoir ponding evolution (`SurfacePondingFunctor`, `Level 2`)**:
   Advances surface ponded water depth $h_n$ [mm] over recession timescale
   $\tau > 0$ [s] via the exact analytical exponential sub-flow:

   $$h_{n+1} = (h_n + E_n)\exp\!\left(-\frac{\Delta t}{\tau}\right), \qquad q_n = (h_n + E_n) - h_{n+1},$$

   emitting $\Delta h_n = h_{n+1} - h_n$ via
   `UpdateItem(ACCUMULATE, per_timestep=True)` and step runoff depth $q_n$ / rate
   $q_n / \Delta t$ via `REPLACE`. By construction, the step ponding mass-balance
   residual $(h_{n+1} - h_n) - E_n + q_n = 0$ vanishes to floating-point roundoff
   for every $\Delta t > 0$.

## Subsurface stormflow, vadose-aquifer exchange, and groundwater

1. **Subsurface lateral stormflow (`SubsurfaceRunoff`, `Level 3`)**:
   `relic.scientific.hydrosphere.runoff.subsurface.SubsurfaceRunoff` computes
   topographic-slope lateral interflow above low-permeability horizons as a
   monotone function of drainable water above field capacity,
   $Q_{\mathrm{sub}} = k_{\mathrm{sub}}\,\tan\beta_{\mathrm{topo}}\,\max(\theta - \theta_{\mathrm{fc}}, 0)^b$
   (Beven and Kirkby, 1979).

2. **Vertical percolation (`VerticalPercolationTransport`, `Level 3`) and capillary rise (`CapillaryRiseTransport`, `Level 3`)**:
   `VerticalPercolationTransport` routes gravity-driven free drainage
   $Q_{\mathrm{perc}} = K(\theta)$ downward from the bottom vadose layer into
   groundwater recharge (`ACCUMULATE` storage transfer, conserving total
   vadose-plus-aquifer water mass). Conversely, `CapillaryRiseTransport` routes
   upward matric-suction flux from a shallow water table $z_{\mathrm{wt}}$ into
   an unsaturated root zone when $\psi(\theta_{\mathrm{bot}})$ lies below hydrostatic
   equilibrium, bounded strictly by available aquifer storage and receiving pore
   capacity.

3. **Unconfined aquifer dynamics (`GroundwaterDynamics`, `Level 3`)**:
   `relic.scientific.hydrosphere.groundwater.dynamics.GroundwaterDynamics`
   advances unconfined aquifer water storage $S_{\mathrm{gw}}$ [kg m$^{-2}$] and
   water-table depth $z_{\mathrm{wt}}$ with specific yield $S_y$ and linear or
   nonlinear baseflow recession $Q_{\mathrm{base}}(S_{\mathrm{gw}})$ (Niu et al.,
   2007):

   $$\frac{\mathrm{d}S_{\mathrm{gw}}}{\mathrm{d}t} = R_{\mathrm{perc}} - C_{\mathrm{cap}} - Q_{\mathrm{base}}(S_{\mathrm{gw}}) - Q_{\mathrm{pump}}, \qquad \Delta z_{\mathrm{wt}} = -\frac{\Delta S_{\mathrm{gw}}}{\rho_w S_y},$$

   closing the coupled subsurface hydrological ledger
   $\Delta S_{\mathrm{gw}} - (R_{\mathrm{perc}} - C_{\mathrm{cap}} - Q_{\mathrm{base}} - Q_{\mathrm{pump}})\Delta t = 0$.

## References

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements*. FAO Irrigation and Drainage Paper 56.
- Anderson, E. A. (1976). *A Point Energy and Mass Balance Model of a Snow Cover*. NOAA Technical Report NWS 19.
- Beven, K. J. and Kirkby, M. J. (1979). A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 24, 43-69.
- Celia, M. A., Bouloutas, E. T., and Zarba, R. L. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. *Water Resources Research*, 26, 1483-1496.
- Manabe, S. (1969). Climate and the ocean circulation 1. The atmospheric circulation and the hydrology of the Earth's surface. *Monthly Weather Review*, 97, 739-774.
- Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205-234.
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. *Water Resources Research*, 12, 513-522.
- Niu, G.-Y., Yang, Z.-L., Dickinson, R. E., Gulden, L. E., and Su, H. (2007). Development of a simple groundwater model for use in climate models and evaluation with Gravity Recovery and Climate Experiment data. *Journal of Geophysical Research*, 112, D07103. doi:10.1029/2006JD007522.
- Priestley, C. H. B. and Taylor, R. J. (1972). On the assessment of surface heat flux and evaporation using large-scale parameters. *Monthly Weather Review*, 100, 81-92.
- Richards, L. A. (1931). Capillary conduction of liquids through porous mediums. *Physics*, 1, 318-333.
- Tarboton, D. G. and Luce, C. H. (1996). *Utah Energy Balance Snow Accumulation and Melt Model (UEB)*. Utah Water Research Laboratory.
- van Genuchten, M. T. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. *Soil Science Society of America Journal*, 44, 892-898.
