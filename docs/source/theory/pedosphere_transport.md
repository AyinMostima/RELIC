# Pedosphere Transport and Interface Mapping

## 1. Process Definition and Scope

This document defines the four certified pedosphere transport and cross-domain interface Functors in RELIC:

1. `LitterfallInterfaceMapper` (`relic/scientific/pedosphere/interface_mapping.py`, `[I]`): a `ScaleCouplingFunctor` (`ExecutionPhase.COUPLING`) that maps phenology carbon and nitrogen litterfall fluxes, litter lignin-to-nitrogen quality, and an optional indivisible grazing-dung pair onto metabolic litter carbon, structural litter carbon, and organic nitrogen boundary fluxes in $\mathrm{g\,element\,m^{-2}\,s^{-1}}$.
2. `PedosphereInterfaceMapper` (`relic/scientific/pedosphere/interface_mapping.py`, `[I]`): a `ScaleCouplingFunctor` (`ExecutionPhase.COUPLING`) whose formal `forward` operator executes the three-channel hydrological flux chain map $y_h = C_h x_h$ with $C_h = I_3$ in $\mathrm{kg\,m^{-2}\,s^{-1}}$, accompanied by auxiliary helper utilities for atmospheric deposition partitioning and linear retardation leaching estimates.
3. `SoluteAdvectionDiffusion1D` (`relic/scientific/pedosphere/transport/solute_advection_diffusion.py`, `[I]`): a `TransportMechanismFunctor` (`ExecutionPhase.MAIN_PROCESSES`) implementing conservative one-dimensional vertical advection, dispersion, and linear-equilibrium sorption for dissolved nitrate ($\mathrm{NO_3^-}\text{-}\mathrm{N}$) and dissolved organic carbon ($\mathrm{DOC}\text{-}\mathrm{C}$) without internal biological reaction source terms.
4. `GasDiffusion1D` (`relic/scientific/pedosphere/transport/gas_diffusion.py`, `[I]`): a `TransportMechanismFunctor` (`ExecutionPhase.MAIN_PROCESSES`) implementing one-dimensional vertical soil gas diffusion and optional soil-atmosphere exchange for $\mathrm{CO_2}\text{-}\mathrm{C}$ and $\mathrm{N_2O}\text{-}\mathrm{N}$ using Millington-Quirk, Penman, or empirical DNDC diffusivity closures.

Soil water movement (`SoilWaterTransportFunctor`) is owned separately by `soil_hydraulics.md`. Nonlinear Freundlich solute sorption inside `SoluteAdvectionDiffusion1D`, inline DOC biodegradation decay inside `SoluteAdvectionDiffusion1D`, and operator-split advection-diffusion-reaction steps inside `SoluteAdvectionDiffusion1D` are `[N]`.

## 2. Governing Equations

### 2.1 Litterfall Carbon and Nitrogen Chain Map

`LitterfallInterfaceMapper` converts biosphere phenology litterfall rates from $\mathrm{kg\,element\,m^{-2}\,s^{-1}}$ to $\mathrm{g\,element\,m^{-2}\,s^{-1}}$ through the central `UnitConverter` operator $\mathcal{U}_{\mathrm{kg,g}}$ (factor $1000$):

$$
F_{C,g} = \mathcal{U}_{\mathrm{kg,g}}\!\left(F_{\mathrm{leaf},C} + F_{\mathrm{root},C}\right), \qquad F_{\mathrm{dung},C,g} = \mathcal{U}_{\mathrm{kg,g}}\!\left(F_{\mathrm{dung},C}\right)
$$

Plant litter carbon is partitioned between metabolic and structural pools by a bounded exponential function of the lignin-to-nitrogen ratio $\mathrm{LN}$ (adapted from CENTURY-family litter quality partitioning; Parton et al., 1987):

$$
f_{\mathrm{met}} = \operatorname{clip}\!\left(\exp\!\left(-k_{\mathrm{part}} \max\!\left(\mathrm{LN} - \mathrm{LN}_0, c_{\min}\right)\right),\, f_{\min},\, f_{\max}\right)
$$

When `"grazing_management"` is declared in `get_dependencies()`, the indivisible dung flux pair $(F_{\mathrm{dung},C}, F_{\mathrm{dung},N})$ is required; when that dependency is absent, both dung fluxes vanish and supplying either key raises a `ValueError`. Manure carbon is split using a fixed two-pool projection fraction $f_{\mathrm{man}}$ (`[M]`, renormalized from the equal active decomposable and resistant fractions of RothC-26.3 farmyard manure; Coleman and Jenkinson, 1996):

$$
F_{\mathrm{met}} = f_{\mathrm{met}} F_{C,g} + f_{\mathrm{man}} F_{\mathrm{dung},C,g}, \qquad F_{\mathrm{str}} = \left(1 - f_{\mathrm{met}}\right) F_{C,g} + \left(1 - f_{\mathrm{man}}\right) F_{\mathrm{dung},C,g}
$$

$$
F_{\mathrm{org},N} = \mathcal{U}_{\mathrm{kg,g}}\!\left(F_{\mathrm{leaf},N} + F_{\mathrm{root},N} + F_{\mathrm{dung},N}\right)
$$

### 2.2 Hydrosphere-Pedosphere Flux Chain Map

`PedosphereInterfaceMapper.forward` maps the three hydrological boundary water fluxes $x_h = (q_{\mathrm{drain}}, q_{\mathrm{sub}}, q_{\mathrm{surf}})^{\mathsf{T}}$ to pedosphere water-loss ports $y_h = (q_{\mathrm{leach}}, q_{\mathrm{lat}}, q_{\mathrm{loss}})^{\mathsf{T}}$ through the identity chain map:

$$
y_h = C_h x_h, \qquad C_h = I_3
$$

All six variables carry units $\mathrm{kg\,m^{-2}\,s^{-1}}$ and satisfy $x_h \ge 0$. Auxiliary class methods provide wet/dry atmospheric N and P deposition speciation (`map_atmosphere_to_pedosphere`) and linear retardation factor evaluation $R = 1 + \rho_b K_d c_{\mathrm{L}} / \max(\theta, \varepsilon_c)$ (`linear_adsorption_retardation_term`), while `upscale` and `downscale` require explicit callable bridge operators and raise typed errors if omitted.

### 2.3 One-Dimensional Solute Advection-Dispersion with Linear Sorption

For each dissolved species $s \in \{\mathrm{NO_3}, \mathrm{DOC}\}$ with liquid-phase concentration $C_s(z,t)$ ($\mathrm{g\,element\,m^{-3}}$) and downward-positive vertical coordinate $z \in [0, Z_{\mathrm{soil}}]$, `SoluteAdvectionDiffusion1D` solves the mass conservation law (Bear, 1972; Simunek et al., 2008):

$$
\frac{\partial \left(B_s C_s\right)}{\partial t} = -\frac{\partial J_s}{\partial z}, \qquad B_s = \theta + \rho_b K_{d,s} c_{\mathrm{L}}
$$

where $B_s$ is the dimensionless bulk storage capacity per unit soil volume, $\theta$ is volumetric water content ($\mathrm{m^3\,m^{-3}}$), $\rho_b$ is dry bulk density ($\mathrm{kg\,m^{-3}}$), $K_{d,s}$ is the linear distribution coefficient ($\mathrm{L\,kg^{-1}}$, set to $0$ when `enable_sorption=False`), and $c_{\mathrm{L}} = 10^{-3}\,\mathrm{m^3\,L^{-1}}$ is read from `relic.physical.constants.conversions.liter_to_cubic_meter`. The downward solute mass flux $J_s$ ($\mathrm{g\,element\,m^{-2}\,s^{-1}}$) combines advection by downward Darcy water flux $q \ge 0$ ($\mathrm{m\,s^{-1}}$) and hydrodynamic dispersion-diffusion:

$$
J_s = q C_s - K_s \frac{\partial C_s}{\partial z}, \qquad K_s = \alpha_{L,s} \lvert q \rvert + D_{w,s}(T) \frac{\theta^{a_{\mathrm{MQ}}}}{\phi^{b_{\mathrm{MQ}}}}, \qquad D_{w,s}(T) = D_{w,\mathrm{ref},s} \left(\frac{T}{T_{\mathrm{ref}}}\right)^{\eta_T}
$$

Here $\alpha_{L,s}$ is longitudinal dispersivity ($\mathrm{m}$), $\phi$ is total porosity ($\mathrm{m^3\,m^{-3}}$), $a_{\mathrm{MQ}}$ and $b_{\mathrm{MQ}}$ are the Millington and Quirk (1961) bulk-flux exponents ($10/3$ and $2$), $D_{w,\mathrm{ref},s}$ is molecular diffusivity in free water at $T_{\mathrm{ref}}$, and $\eta_T$ is the aqueous diffusivity temperature exponent.

### 2.4 One-Dimensional Soil Gas Diffusion

For each trace gas $g \in \{\mathrm{CO_2}, \mathrm{N_2O}\}$ with bulk soil concentration $C_g(z,t)$ ($\mathrm{g\,element\,m^{-3}}$) and downward coordinate $z$, `GasDiffusion1D` evaluates Fickian interior diffusive flux $J_g$ and upward soil-to-atmosphere surface emission $F_{\mathrm{out},g}$:

$$
J_g = -D_{\mathrm{eff},g} \frac{\partial C_g}{\partial z}, \qquad F_{\mathrm{out},g} = -J_g(0)
$$

With air-filled porosity $\theta_a = \max(\phi - \theta, c_{\min})$ and temperature-corrected free-air diffusivity

$$
D_{0,g}(T) = D_{0,g}(T_{\mathrm{ref}}) \left(\frac{T}{T_{\mathrm{ref}}}\right)^{p_T}
$$

(reducing to $D_{0,g}(T_{\mathrm{ref}})$ when `temperature_correction` is disabled), three effective diffusivity closures are supported through `pedosphere.transport.diffusion_model`:

1. **Millington-Quirk (`millington_quirk`, `[I]`)** (Millington and Quirk, 1961):

$$
D_{\mathrm{eff},g}^{\mathrm{MQ}} = D_{0,g}(T) \frac{\theta_a^{a_{\mathrm{MQ}}}}{\max(\phi, \varepsilon)^{b_{\mathrm{MQ}}}}
$$

2. **Penman (`penman`, `[I]`)** (Penman, 1940):

$$
D_{\mathrm{eff},g}^{\mathrm{Pen}} = D_{0,g}(T)\, k_P\, \theta_a
$$

3. **DNDC Empirical $\mathrm{N_2O}$ Closure (`dndc`, `[H]`)** (Li et al., 1992; Song et al., 2021, Table S1 Eq. 20): for $\mathrm{N_2O}$, with clamped water-filled pore space $W_s = \operatorname{clip}(\theta / \max(\phi, \varepsilon), W_{\min}, W_{\max})$ and Celsius temperature $T_C = T - T_{\mathrm{0C}}$, the dimensionless hourly transfer factor is

$$
D_{\mathrm{N_2O}} = \max\!\left((1 - W_s)\left[a_0 + a_c f_{\mathrm{clay}} + \left(a_p - a_{cp} f_{\mathrm{clay}}\right) b_T^{T_C / T_{\mathrm{div}}} \phi\right],\, c_{\min}\right)
$$

which maps to the layer diffusivity and open top-boundary flux on uniform grid spacing $\Delta z$ via $t_h = 3600\,\mathrm{s}$ (`seconds_per_hour`):

$$
D_{\mathrm{eff,N_2O}}^{\mathrm{DNDC}} = \frac{D_{\mathrm{N_2O}} \Delta z^2}{t_h}, \qquad F_{\mathrm{out,N_2O}}^{\mathrm{DNDC}} = \frac{\left(C_{\mathrm{N_2O},0} - C_{\mathrm{atm,N_2O}}\right) \Delta z\, D_{\mathrm{N_2O}}}{t_h}
$$

Under `diffusion_model="dndc"`, $\mathrm{CO_2}$ retains the Millington-Quirk diffusivity closure.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $F_{\mathrm{leaf},C}$ | Leaf litterfall carbon flux | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `leaf_litterfall_carbon` | `LitterfallInterfaceMapper` required input |
| $F_{\mathrm{root},C}$ | Fine-root litterfall carbon flux | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `froot_litterfall_carbon` | `LitterfallInterfaceMapper` required input |
| $F_{\mathrm{leaf},N}$ | Leaf litterfall nitrogen flux | $\mathrm{kg\,N\,m^{-2}\,s^{-1}}$ | `leaf_litterfall_nitrogen` | `LitterfallInterfaceMapper` required input |
| $F_{\mathrm{root},N}$ | Fine-root litterfall nitrogen flux | $\mathrm{kg\,N\,m^{-2}\,s^{-1}}$ | `froot_litterfall_nitrogen` | `LitterfallInterfaceMapper` required input |
| $\mathrm{LN}$ | Litter lignin-to-nitrogen ratio | $\mathrm{1}$ | `lignin_n_ratio` | `LitterfallInterfaceMapper` required input |
| $F_{\mathrm{dung},C}, F_{\mathrm{dung},N}$ | Grazing dung C and N fluxes | $\mathrm{kg\,element\,m^{-2}\,s^{-1}}$ | `dung_c_flux`, `dung_n_flux` | `LitterfallInterfaceMapper` dependency-gated input pair |
| $F_{\mathrm{met}}, F_{\mathrm{str}}$ | Metabolic and structural litter C inputs | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `metabolic_litter_input`, `structural_litter_input` | `LitterfallInterfaceMapper` output (`REPLACE`) |
| $F_{\mathrm{org},N}$ | Pedosphere organic nitrogen input | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `organic_nitrogen_input` | `LitterfallInterfaceMapper` output (`REPLACE`) |
| $q_{\mathrm{drain}}, q_{\mathrm{sub}}, q_{\mathrm{surf}}$ | Hydrological drainage, subsurface, and surface runoff fluxes | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `drainage_flux`, `subsurface_runoff_flux`, `surface_runoff_flux` | `PedosphereInterfaceMapper` required inputs |
| $q_{\mathrm{leach}}, q_{\mathrm{lat}}, q_{\mathrm{loss}}$ | Mapped pedosphere leaching, lateral, and surface water fluxes | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `leaching_flux`, `lateral_flow`, `surface_loss` | `PedosphereInterfaceMapper` outputs (`REPLACE`) |
| $\theta$ | Volumetric soil water content profile (solute) or moisture (gas) | $\mathrm{m^3\,m^{-3}}$ | `soil_water_content`, `soil_moisture` | Transport required input |
| $q$ | Downward Darcy water flux profile ($q \ge 0$) | $\mathrm{m\,s^{-1}}$ | `darcy_flux` | `SoluteAdvectionDiffusion1D` required input |
| $C_{\mathrm{NO_3}}, C_{\mathrm{DOC}}$ | Layer liquid $\mathrm{NO_3}\text{-}\mathrm{N}$ and $\mathrm{DOC}\text{-}\mathrm{C}$ concentrations | $\mathrm{g\,element\,m^{-3}}$ | `no3_concentration`, `doc_concentration` | `SoluteAdvectionDiffusion1D` input / output (`REPLACE`, `per_timestep=True`) |
| $C_{\mathrm{surf,NO_3}}, C_{\mathrm{surf,DOC}}$ | Infiltrating top-boundary solute concentrations | $\mathrm{g\,element\,m^{-3}}$ | `surface_no3_concentration`, `surface_doc_concentration` | `SoluteAdvectionDiffusion1D` required scalar inputs |
| $\rho_b$ | Soil dry bulk density profile | $\mathrm{kg\,m^{-3}}$ | `bulk_density` | `SoluteAdvectionDiffusion1D` required input |
| $\phi$ | Soil total porosity profile | $\mathrm{m^3\,m^{-3}}$ | `porosity` | Transport required input |
| $T$ | Soil temperature profile | $\mathrm{K}$ | `soil_temperature` | Transport required input |
| $F_{\mathrm{leach},N}, F_{\mathrm{leach},C}$ | Bottom-boundary $\mathrm{NO_3}\text{-}\mathrm{N}$ and $\mathrm{DOC}\text{-}\mathrm{C}$ leaching fluxes | $\mathrm{g\,element\,m^{-2}\,s^{-1}}$ | `nitrogen_leaching_flux`, `carbon_leaching_flux` | `SoluteAdvectionDiffusion1D` output (`REPLACE`) |
| $F_{\mathrm{in},N}, F_{\mathrm{in},C}$ | Top-boundary Cauchy solute inflow rates | $\mathrm{g\,element\,m^{-2}\,s^{-1}}$ | `diagnostics.solute_transport.nitrogen_surface_input_flux`, `diagnostics.solute_transport.carbon_surface_input_flux` | `SoluteAdvectionDiffusion1D` diagnostic (`REPLACE`) |
| $\mathcal{R}_{\mathrm{sol},N}, \mathcal{R}_{\mathrm{sol},C}$ | Step-integrated solute mass-balance residuals | $\mathrm{g\,element\,m^{-2}}$ | `diagnostics.solute_transport.nitrogen_mass_balance_residual`, `diagnostics.solute_transport.carbon_mass_balance_residual` | `SoluteAdvectionDiffusion1D` diagnostic (`REPLACE`, `per_timestep=True`) |
| $f_{\mathrm{clay}}$ | Soil clay fraction profile | $\mathrm{1}$ | `clay_content` | `GasDiffusion1D` required input |
| $C_{\mathrm{CO_2}}, C_{\mathrm{N_2O}}$ | Layer soil $\mathrm{CO_2}\text{-}\mathrm{C}$ and $\mathrm{N_2O}\text{-}\mathrm{N}$ concentrations | $\mathrm{g\,element\,m^{-3}}$ | `soil_co2_concentration`, `n2o_concentration` | `GasDiffusion1D` input / increment output (`ACCUMULATE`, `per_timestep=True`) |
| $F_{\mathrm{out,CO_2}}, F_{\mathrm{out,N_2O}}$ | Upward soil-to-atmosphere gas fluxes | $\mathrm{g\,element\,m^{-2}\,s^{-1}}$ | `co2_surface_flux`, `n2o_surface_flux` | `GasDiffusion1D` output (`REPLACE`) |
| $J_{\mathrm{prof,CO_2}}, J_{\mathrm{prof,N_2O}}$ | Interior face diffusive gas flux profiles | $\mathrm{g\,element\,m^{-2}\,s^{-1}}$ | `co2_flux_profile`, `n2o_flux_profile` | `GasDiffusion1D` output (`REPLACE`) |
| $H_h, \sigma_h, \dot{H}_{\mathrm{audit}}, r_{\mathrm{port}}$ | Structural quadratic storage, interface dissipation, discrete storage rate, and port-power residual | audit units | `diagnostics.gas_diffusion.energy_storage`, `diagnostics.gas_diffusion.dissipation_rate`, `diagnostics.gas_diffusion.dH_dt_audit`, `diagnostics.gas_diffusion.port_power_residual` | `GasDiffusion1D` structural diagnostics (`REPLACE`) |
| $R_{\mathrm{clip,CO_2}}, R_{\mathrm{clip,N_2O}}$ | Layer non-negativity clipping residuals | $\mathrm{g\,element\,m^{-3}}$ | `diagnostics.gas_diffusion.co2_clipping_residual`, `diagnostics.gas_diffusion.n2o_clipping_residual` | `GasDiffusion1D` diagnostic (`REPLACE`) |
| $\Delta t_{\mathrm{gas}}$ | Accepted gas-diffusion step duration | $\mathrm{s}$ | `diagnostics.gas_diffusion.step_seconds` | `GasDiffusion1D` diagnostic (`REPLACE`) |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $\mathrm{LN}_0, k_{\mathrm{part}}$ | `pedosphere.decomposition.lignin_n_threshold`, `pedosphere.decomposition.lignin_partitioning_coefficient` | $\mathrm{1}$ | `pedosphere/decomposition.yaml` ($0.45$, $3.0$) | Parton et al. (1987) |
| $f_{\min}, f_{\max}, f_{\mathrm{man}}$ | `pedosphere.decomposition.min_metabolic_fraction`, `pedosphere.decomposition.max_metabolic_fraction`, `pedosphere.decomposition.manure_metabolic_fraction` | $\mathrm{1}$ | `pedosphere/decomposition.yaml` ($0.1$, $0.9$, $0.5$) | Parton et al. (1987); Coleman and Jenkinson (1996) |
| $f_{\mathrm{NO_y}}, f_{\mathrm{NH_x}}$ | `pedosphere.nutrient_cycling.n_noy_fraction`, `pedosphere.nutrient_cycling.n_nhx_fraction` | $\mathrm{1}$ | `pedosphere/nutrient_cycling.yaml` ($0.6$, $0.4$) | Interface deposition speciation |
| $f_{\mathrm{PO_4}}, f_{\mathrm{dust}}$ | `pedosphere.phosphorus.p_po4_fraction`, `pedosphere.phosphorus.p_dust_fraction` | $\mathrm{1}$ | `pedosphere/phosphorus.yaml` ($0.3$, $0.7$) | Wang et al. (2010) |
| $N, Z_{\mathrm{soil}}$ | `pedosphere.grid.n_layers`, `pedosphere.grid.soil_depth` | $\mathrm{1}$, $\mathrm{m}$ | `pedosphere/grid.yaml` ($10$, $1.5$) | Vertical grid geometry |
| $r, \mathrm{CFL}, \Delta t_{\max}$ | `pedosphere.transport.layer_thickness_ratio`, `pedosphere.transport.cfl_number`, `pedosphere.transport.time_step_maximum` | $\mathrm{1}$, $\mathrm{1}$, $\mathrm{s}$ | `pedosphere/transport.yaml` ($1.5$, $0.5$, $1800.0$) | Simunek et al. (2008) |
| $D_{w,\mathrm{def}}, \eta_T$ | `pedosphere.transport.default_molecular_diffusivity_water`, `pedosphere.transport.molecular_diffusivity_temperature_exponent` | $\mathrm{m^2\,s^{-1}}$, $\mathrm{1}$ | `pedosphere/transport.yaml` ($10^{-9}$, $0.5$) | Bear (1972) |
| $D_{w,\mathrm{NO_3}}, D_{w,\mathrm{DOC}}$ | `pedosphere.transport.nitrate_diffusivity_water`, `pedosphere.transport.doc_diffusivity_water` | $\mathrm{m^2\,s^{-1}}$ | `pedosphere/transport.yaml` ($1.9 \times 10^{-9}$, $6.0 \times 10^{-10}$) | Bear (1972); Jury and Horton (2004) |
| $\alpha_{L,\mathrm{NO_3}}, \alpha_{L,\mathrm{DOC}}$ | `pedosphere.transport.nitrate_longitudinal_dispersivity`, `pedosphere.transport.doc_longitudinal_dispersivity` | $\mathrm{m}$ | `pedosphere/transport.yaml` ($0.01$, $0.01$) | Bear (1972) |
| $K_{d,\mathrm{NO_3}}, K_{d,\mathrm{DOC}}$ | `pedosphere.transport.nitrate_distribution_coefficient`, `pedosphere.transport.doc_distribution_coefficient` | $\mathrm{L\,kg^{-1}}$ | `pedosphere/transport.yaml` ($0.1$, $2.0$) | Simunek et al. (2008) |
| $a_{\mathrm{MQ}}, b_{\mathrm{MQ}}, k_P$ | `pedosphere.physical.millington_quirk_exponent_numerator`, `pedosphere.physical.millington_quirk_exponent_denominator`, `pedosphere.physical.penman_coefficient` | $\mathrm{1}$ | `pedosphere/physical.yaml` ($3.333333$, $2.0$, $0.66$) | Millington and Quirk (1961); Penman (1940) |
| $D_{0,\mathrm{CO_2}}, D_{0,\mathrm{N_2O}}, p_T$ | `pedosphere.physical.co2_diffusivity_air_stp`, `pedosphere.physical.n2o_diffusivity_air_stp`, `pedosphere.physical.gas_diffusivity_temperature_exponent` | $\mathrm{m^2\,s^{-1}}$, $\mathrm{m^2\,s^{-1}}$, $\mathrm{1}$ | `pedosphere/physical.yaml` ($1.47 \times 10^{-5}$, $1.32 \times 10^{-5}$, $1.75$) | Jury and Horton (2004); Moldrup et al. (2000) |
| $C_{\mathrm{atm,CO_2}}, C_{\mathrm{atm,N_2O}}$ | `pedosphere.physical.atmospheric_co2_concentration`, `pedosphere.physical.atmospheric_n2o_concentration` | $\mathrm{g\,element\,m^{-3}}$ | `pedosphere/physical.yaml` ($0.214348$, $4.19948 \times 10^{-4}$) | Elemental ideal-gas boundary |
| $\text{model}, f_{T,\mathrm{corr}}, f_{\mathrm{atm}}, f_{\mathrm{ovr}}$ | `pedosphere.transport.diffusion_model`, `pedosphere.transport.temperature_correction`, `pedosphere.transport.atmospheric_exchange`, `pedosphere.transport.allow_constructor_overrides` | string / flag | `pedosphere/transport.yaml` (`millington_quirk`, $1.0$, $1.0$, $0.0$) | Gas transport switches |
| $a_c, a_p, a_0, a_{cp}, b_T, T_{\mathrm{div}}$ | `pedosphere.transport.n2o_dndc_clay_effect`, `pedosphere.transport.n2o_dndc_porosity_effect`, `pedosphere.transport.n2o_dndc_base_coefficient`, `pedosphere.transport.n2o_dndc_clay_porosity_coefficient`, `pedosphere.transport.n2o_dndc_temperature_base`, `pedosphere.transport.n2o_dndc_temperature_divisor` | $\mathrm{1}$, $\mathrm{K}$ | `pedosphere/transport.yaml` ($0.124$, $0.013$, $0.018$, $0.016$, $2.0$, $20.0$) | Li et al. (1992); Song et al. (2021) |
| $c_{\min}, \varepsilon, \varepsilon_L, \varepsilon_c, \varepsilon_m$ | `pedosphere.numerical.clamp_min`, `pedosphere.numerical.epsilon`, `pedosphere.numerical.epsilon_length`, `pedosphere.numerical.epsilon_concentration`, `pedosphere.numerical.mass_balance_tolerance` | numerical | `pedosphere/numerical.yaml` ($0.0$, $10^{-10}$, $10^{-6}$, $10^{-10}$, $10^{-6}$) | Numerical safeguards |
| $\Delta t_{\min,\mathrm{gas}}, z_0, W_{\min}, W_{\max}$ | `pedosphere.numerical.min_time_step`, `pedosphere.numerical.zero_initial`, `pedosphere.numerical.wfps_minimum`, `pedosphere.numerical.wfps_maximum` | $\mathrm{s}$, $\mathrm{1}$ | `pedosphere/numerical.yaml` ($60.0$, $0.0$, $0.0$, $1.0$) | Gas numerical bounds |
| $c_{\mathrm{L}}, T_{\mathrm{ref}}, T_{\mathrm{0C}}, t_h$ | `relic.physical.constants.conversions.liter_to_cubic_meter`, `relic.physical.constants.temperature.reference_temperature`, `relic.physical.constants.temperature.celsius_to_kelvin`, `relic.physical.constants.time.seconds_per_hour` | $\mathrm{m^3\,L^{-1}}$, $\mathrm{K}$, $\mathrm{K}$, $\mathrm{s}$ | `physical/constants.yaml` ($10^{-3}$, $298.15$, $273.15$, $3600.0$) | Physical constants SSOT |

## 5. Numerical Methods and Discretization

### 5.1 Geometric Grid and Unsplit Sub-Stepped Finite Volume in `SoluteAdvectionDiffusion1D`

`SoluteAdvectionDiffusion1D._setup_vertical_grid` constructs an $N$-layer vertical mesh spanning $Z_{\mathrm{soil}}$. When the thickness ratio $r$ (`layer_thickness_ratio`) equals $1$, layer thicknesses are uniform ($\Delta z_i = Z_{\mathrm{soil}} / N$); when $r$ differs from $1$, layer thicknesses follow a geometric progression normalized to $Z_{\mathrm{soil}}$:

$$
\Delta z_i = \Delta z_0 r^i \quad (i = 0, \dots, N - 1), \qquad \sum_{i=0}^{N-1} \Delta z_i = Z_{\mathrm{soil}}, \qquad z_i = \sum_{k=0}^{i-1} \Delta z_k + \frac{1}{2}\Delta z_i
$$

Unlike operator-split advection-diffusion-reaction textbook schemes, `SoluteAdvectionDiffusion1D.forward` advances the unsplit conservative finite-volume operator over $N_{\mathrm{sub}} = \lceil \Delta t / \Delta t_{\mathrm{stable}} \rceil$ explicit sub-steps of length $\delta t = \Delta t / N_{\mathrm{sub}}$. At interior faces $i + 1/2$ ($i = 0, \dots, N - 2$), advective flux uses first-order downward upwinding ($q \ge 0$) with arithmetic-mean face Darcy velocity $q_{i+1/2} = \frac{1}{2}(q_i + q_{i+1})$ and harmonic-mean bulk dispersion $K_{i+1/2}^{\mathrm{harm}}$:

$$
J_{i+1/2} = q_{i+1/2} C_i - K_{i+1/2}^{\mathrm{harm}} \frac{C_{i+1} - C_i}{z_{i+1} - z_i}, \qquad K_{i+1/2}^{\mathrm{harm}} = \frac{2 K_i K_{i+1}}{K_i + K_{i+1}}
$$

(vanishing whenever $K_i + K_{i+1} \le 0$). With top Cauchy inflow $J_{-1/2} = q_0 C_{\mathrm{surf}}$ and bottom free-drainage outflow $J_{N-1/2} = q_{N-1} C_{N-1}$, the capacity-weighted CFL sub-step bound and explicit update are:

$$
\Delta t_{\mathrm{stable}} = \min\!\left(\mathrm{CFL} \cdot \min_{0 \le i < N} \frac{B_i \Delta z_i}{a_{\mathrm{out},i}},\, \Delta t_{\max},\, \Delta t\right), \qquad B_i C_i^{(m+1)} = B_i C_i^{(m)} - \frac{\delta t}{\Delta z_i}\left(J_{i+1/2}^{(m)} - J_{i-1/2}^{(m)}\right)
$$

where $a_{\mathrm{out},i}$ sums the outgoing advective and dispersive face conductances of cell $i$. Any negative post-substep concentration raises a `ValueError` without post-hoc clamping or proportional rescaling.

### 5.2 Uniform Grid and Stencil Structure in `GasDiffusion1D`

`GasDiffusion1D` uses a uniform $N$-layer grid with $\Delta z = \max(Z_{\mathrm{soil}} / N, \varepsilon_L)$. In `_calculate_diffusive_flux`, cell-center concentration gradients $g_i = (\partial_z C)_i$ are computed via `OperatorFactory.create("gradient.central_difference_1d")` and averaged across interior faces alongside harmonic-mean effective diffusivities:

$$
J_{i+1/2} = -D_{\mathrm{eff},i+1/2}^{\mathrm{harm}} \cdot \frac{1}{2}\left(g_i + g_{i+1}\right), \qquad D_{\mathrm{eff},i+1/2}^{\mathrm{harm}} = \frac{2 D_{\mathrm{eff},i} D_{\mathrm{eff},i+1}}{\max(D_{\mathrm{eff},i} + D_{\mathrm{eff},i+1}, \varepsilon)}
$$

When `atmospheric_exchange` is enabled, the top face flux is $J_{-1/2} = -F_{\mathrm{out}}$ with $F_{\mathrm{out}} = D_{\mathrm{eff},0}(C_0 - C_{\mathrm{atm}}) / (0.5 \Delta z)$ for Fickian closures (or $F_{\mathrm{out,N_2O}}^{\mathrm{DNDC}}$ for $\mathrm{N_2O}$ under `dndc`); when `atmospheric_exchange` is disabled, $J_{-1/2} = 0$ and $F_{\mathrm{out}} = 0$. The bottom face is closed ($J_{N-1/2} = 0$). A single explicit step forms the raw increment and applies a non-negativity floor with explicit residual routing:

$$
\Delta C_{\mathrm{raw},i} = -\frac{\Delta t}{\Delta z}\left(J_{i+1/2} - J_{i-1/2}\right), \qquad \Delta C_i = \max\!\left(\Delta C_{\mathrm{raw},i}, -C_i\right), \qquad R_{\mathrm{clip},i} = \Delta C_i - \Delta C_{\mathrm{raw},i} \ge 0
$$

Because `_interface_dissipation` evaluates compact two-point face gradients $(C_{i+1} - C_i) / \Delta z$ (yielding $\sigma_h$ in $\mathrm{g^2\,m^{-5}\,s^{-1}}$) whereas `_calculate_diffusive_flux` uses wide averaged central differences and `dH_dt_audit` evaluates unweighted $\frac{1}{2}\sum_i (C_{\mathrm{new},i}^2 - C_i^2)/\Delta t$ ($\mathrm{g^2\,m^{-6}\,s^{-1}}$), the discrete structural power balance exhibits a nonzero stencil and metric residual reported explicitly in `diagnostics.gas_diffusion.port_power_residual`.

## 6. Conservation, Boundary Conditions, and Ledgers

- **`LitterfallInterfaceMapper`**: Pointwise boundary conversion closes carbon and nitrogen exactly:

$$
F_{\mathrm{met}} + F_{\mathrm{str}} - 1000\left(F_{\mathrm{leaf},C} + F_{\mathrm{root},C} + F_{\mathrm{dung},C}\right) = 0, \qquad F_{\mathrm{org},N} - 1000\left(F_{\mathrm{leaf},N} + F_{\mathrm{root},N} + F_{\mathrm{dung},N}\right) = 0
$$

- **`PedosphereInterfaceMapper`**: Pointwise water-flux identity preserves each hydrological channel ($q_{\mathrm{leach}} = q_{\mathrm{drain}}$, $q_{\mathrm{lat}} = q_{\mathrm{sub}}$, $q_{\mathrm{loss}} = q_{\mathrm{surf}}$).
- **`SoluteAdvectionDiffusion1D`**: Over the full step $\Delta t = \sum_{m=1}^{N_{\mathrm{sub}}} \delta t$, column storage $M_s = \sum_{i=0}^{N-1} B_{s,i} C_{s,i} \Delta z_i$ ($\mathrm{g\,element\,m^{-2}}$) satisfies the telescopic discrete balance:

$$
\mathcal{R}_{\mathrm{sol},s} = \left(M_s^{n+1} - M_s^n\right) - \left(\bar{F}_{\mathrm{in},s} - \bar{F}_{\mathrm{leach},s}\right)\Delta t = 0
$$

up to floating-point rounding, where $\bar{F}_{\mathrm{in},s}$ and $\bar{F}_{\mathrm{leach},s}$ are sub-step mean top Cauchy inflow and bottom free-drainage fluxes. Upward Darcy velocity ($q < 0$) raises `ValueError` so mass cannot exit through an undeclared surface outflow port. The concentration `UpdateItem`s carry `UpdateMode.REPLACE` (`per_timestep=True`) without an `element` tag because areal mass weighting requires $B_{s,i} \Delta z_i$.
- **`GasDiffusion1D`**: With closed bottom boundary $J_{N-1/2} = 0$ and routed clipping residual $R_{\mathrm{clip},g,i}$, each gas species satisfies the exact discrete column identity:

$$
\sum_{i=0}^{N-1} \Delta C_{g,i} \Delta z + F_{\mathrm{out},g} \Delta t - \sum_{i=0}^{N-1} R_{\mathrm{clip},g,i} \Delta z = 0
$$

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `LitterfallInterfaceMapper` | `relic/scientific/pedosphere/interface_mapping.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.COUPLING` |
| `PedosphereInterfaceMapper` | `relic/scientific/pedosphere/interface_mapping.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.COUPLING` |
| `SoluteAdvectionDiffusion1D` | `relic/scientific/pedosphere/transport/solute_advection_diffusion.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |
| `GasDiffusion1D` | `relic/scientific/pedosphere/transport/gas_diffusion.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

## 8. Implementation Status

- **`LitterfallInterfaceMapper`**: `[I]` (IPHS Level 2 `scale_coupling`, with `[M]` two-pool projection $f_{\mathrm{man}} = 0.5$ for optional grazing dung carbon). Certificate: `pedosphere_interface_mapping_litterfall_interface_mapper.yaml`.

```text
No executable cross-domain state or port mapping connected the phenology boundary to pedosphere inputs.
A single ScaleCoupling forward maps four required plant element ports, one required quality input, and a dependency-selected indivisible pair of grazing dung element ports to three required g element m-2 s-1 target ports.
```

- **`PedosphereInterfaceMapper`**: `[I]` (IPHS Level 2 `scale_coupling`). Certificate: `pedosphere_interface_mapping_pedosphere_interface_mapper.yaml`.

```text
dynamic alias mapping emitted state-relative raw tensor increments
y_h = C_h x_h, C_h = I_3, with three required kg m-2 s-1 input and output ports
```

- **`SoluteAdvectionDiffusion1D`**: `[I]` (IPHS Level 3 `transport`; quadratic concentration storage functional is structural audit metadata, without a thermodynamic chemical-potential Hamiltonian claim). Inline biological decay, Freundlich nonlinear sorption, and operator splitting inside this Functor are `[N]`. Certificate: `pedosphere_transport_solute_advection_diffusion_solute_advection_diffusion1_d.yaml`.

```text
$B_iC_i^{n+1}=B_iC_i^n-\Delta t(J_{i+1/2}-J_{i-1/2})/\Delta z_i$
B_h c_dot_h = -D_h J_h + G_h u_h with declared storage, solute flux, and top/bottom matter ports
```

- **`GasDiffusion1D`**: `[I]` for Millington-Quirk and Penman closures and `[H]` for the empirical DNDC $\mathrm{N_2O}$ closure (IPHS Level 3 `transport`; interior flux uses averaged cell-center gradients whereas structural dissipation uses two-point differences, with the mismatch routed to `diagnostics.gas_diffusion.port_power_residual`). Certificate: `pedosphere_transport_gas_diffusion_gas_diffusion1_d.yaml`.

```text
J = -Deff partialC/partialz; Deff = f(thetaa, phi, T)
c_dot = -D_h q_h + G_h u_h, q_h = -star_K D_h^T e_h, e_h = c_h, dH/dt = -sigma_h + e_h.T G_h u_h + residual
```

## 9. References

1. Bear, J. (1972). *Dynamics of Fluids in Porous Media*. American Elsevier Publishing Company.
2. Coleman, K., and Jenkinson, D. S. (1996). RothC-26.3: A model for the turnover of carbon in soil. In *Evaluation of Soil Organic Matter Models*, NATO ASI Series I, Vol. 38, 237-246.
3. Jury, W. A., and Horton, R. (2004). *Soil Physics* (6th ed.). John Wiley and Sons.
4. Kusa, K., Sawamoto, T., Hu, R., and Hatano, R. (2008). Comparison of the closed-chamber and gas concentration gradient methods for measurement of $\mathrm{CO_2}$ and $\mathrm{N_2O}$ fluxes in two upland field soils. *Soil Science and Plant Nutrition*, 54(5), 777-785. doi:10.1111/j.1747-0765.2008.00292.x
5. Li, C., Frolking, S., and Frolking, T. A. (1992). A model of nitrous oxide evolution from soil driven by rainfall events: 1. Model structure and sensitivity. *Journal of Geophysical Research*, 97(D9), 9759-9776. doi:10.1029/92JD00509
6. Millington, R. J., and Quirk, J. P. (1961). Permeability of porous solids. *Transactions of the Faraday Society*, 57, 1200-1207. doi:10.1039/TF9615701200
7. Moldrup, P., Olesen, T., Komatsu, T., Schjonning, P., and Rolston, D. E. (2000). Tortuosity, diffusivity, and permeability in the soil liquid and gaseous phases. *Soil Science Society of America Journal*, 64(4), 1276-1286.
8. Parton, W. J., Schimel, D. S., Cole, C. V., and Ojima, D. S. (1987). Analysis of factors controlling soil organic matter levels in Great Plains grasslands. *Soil Science Society of America Journal*, 51(5), 1173-1179.
9. Penman, H. L. (1940). Gas and vapour movements in soil: I. The diffusion of vapours through porous solids. *Journal of Agricultural Science*, 30(3), 437-462.
10. Simunek, J., van Genuchten, M. Th., and Sejna, M. (2008). Development and applications of the HYDRUS and STANMOD software packages and related codes. *Vadose Zone Journal*, 7(2), 587-600. doi:10.2136/vzj2007.0077
11. Song, X., Wang,M., Li, C., et al. (2021). Process equations of the DNDC nitrous oxide diffusion parameterization (Table S1, Eq. 20).
