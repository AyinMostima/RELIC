# Soil Organic Matter Decomposition and Pool Observation

## 1. Process Definition and Scope

Soil organic carbon turnover and pool observation in the pedosphere domain are governed by four Functors:

1. `CarbonPoolsFunctor` (`relic/scientific/pedosphere/carbon_pools/carbon_pools_functor.py`, `[I]`): state-neutral postprocessing observer projecting the five canonical soil carbon stocks (`metabolic_litter_c`, `structural_litter_c`, `fast_som_c`, `slow_som_c`, `passive_som_c`) into a stacked snapshot tensor and grid-cell areal total without mutating prognostic state.
2. `PoolsMatrix` (`relic/scientific/pedosphere/carbon_pools/pools_matrix.py`, `[I]`): linear five-pool compartment-matrix reaction closure driven by an explicit `environmental_modifier` input and five pool-specific boundary carbon input fluxes.
3. `FirstOrderDecomposition` (`relic/scientific/pedosphere/decomposition/first_order.py`, `[H]`): five-pool first-order decay and inter-pool transfer closure incorporating temperature, volumetric soil moisture, structural-litter lignin inhibition, clay SOM protection, and clay enhancement of humification transfers to `passive_som_c`.
4. `MicrobialEnzymeDecomposition` (`relic/scientific/pedosphere/decomposition/microbial_enzyme.py`, `[I]`): microbial-explicit four-pool reverse Michaelis-Menten closure over `metabolic_litter_c`, `structural_litter_c`, `available_som_c`, and `microbial_biomass_c`.

`PoolsMatrix`, `FirstOrderDecomposition`, and `MicrobialEnzymeDecomposition` represent mutually exclusive decomposition laws; a compiled simulator graph selects at most one reaction closure. Fresh plant and grazing-manure litter partitioning belongs upstream to `LitterfallInterfaceMapper` (6 ordinary `ParameterManager` keys including `pedosphere.decomposition.manure_metabolic_fraction`; combined with `FirstOrderDecomposition`, the chain spans 37 unique `ParameterManager` keys and 3 physical constants). Explicit necromass pool formation from microbial mortality and canonical `carbon_decomposition_rate` export from `MicrobialEnzymeDecomposition` remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 23 (Sections 23.1 and 23.2).

## 2. Governing Equations

### 2.1 Canonical Five-Pool Observation (`CarbonPoolsFunctor`)

For grid cell $c$ and canonical pool vector $x_h(c) = [C_{\mathrm{met}}, C_{\mathrm{str}}, C_{\mathrm{fast}}, C_{\mathrm{slow}}, C_{\mathrm{pass}}]^{\mathsf{T}}$ ($\mathrm{g\,C\,m^{-2}}$) with identity observation operator $O_h = I_5$:

$$
z_h(c) = O_h x_h(c), \qquad
C_{\mathrm{total}}(c) = \sum_{i=1}^{5} z_{h,i}(c), \qquad
\Delta x_h = 0
$$

Summation runs strictly over the pool axis (`dim=0`); cross-cell regional aggregation requiring cell areas is excluded.

### 2.2 Compartment-Matrix Closure (`PoolsMatrix`)

Given pool vector $C \in \mathbb{R}_{\ge 0}^{5}$ ($\mathrm{g\,C\,m^{-2}}$), diagonal turnover matrix $K = \operatorname{diag}(k_1,\dots,k_5)$ ($\mathrm{s^{-1}}$), inter-pool transfer matrix $T = (T_{ij})$ ($T_{ij} \ge 0$ from donor $j$ to receiver $i$, $\sum_i T_{ij} \le 1$), scalar or grid `environmental_modifier` $f_{\mathrm{env}} \ge 0$, and five-pool boundary input vector $I_{\mathrm{ext}}$ ($\mathrm{g\,C\,m^{-2}\,s^{-1}}$):

$$
D = f_{\mathrm{env}} K C, \qquad
\frac{dC}{dt} = (T - I_5) D + I_{\mathrm{ext}}, \qquad
R_{\mathrm{decomp}} = \mathbf{1}_{5}^{\mathsf{T}} D, \qquad
R_{\mathrm{hetero}} = \mathbf{1}_{5}^{\mathsf{T}}(I_5 - T) D
$$

In `advance_reaction_network`, the stoichiometric map is $N_h = [T - I_5,\; I_5] \in \mathbb{R}^{5 \times 10}$ acting on $r_h = [D^{\mathsf{T}},\; I_{\mathrm{ext}}^{\mathsf{T}}]^{\mathsf{T}}$.

### 2.3 First-Order Litter-Quality and Clay-Modified Closure (`FirstOrderDecomposition`)

Annual reference decay coefficients $k_{i,\mathrm{yr}}$ ($\mathrm{yr^{-1}}$) convert once to $k_{i,\mathrm{ref}} = k_{i,\mathrm{yr}} / s_{\mathrm{yr}}$ ($\mathrm{s^{-1}}$). For soil temperature $T \in [T_{\mathrm{bio,\min}}, T_{\mathrm{bio,\max}}]$ ($\mathrm{K}$) with reference $T_{\mathrm{ref}} = 298.15\,\mathrm{K}$, scale multiplier $p_{25}$ (`param_25`), and volumetric moisture $\theta \in [0, c_{\max}]$ ($\mathrm{m^3\,m^{-3}}$):

$$
f_T^{\mathrm{raw}}(T) =
\begin{cases}
p_{25}\,Q_{10}^{(T - T_{\mathrm{ref}})/\Delta T_{10}}, & \text{type} = \text{"q10"} \\
p_{25}\,\exp\!\left[\dfrac{E_a}{R_{\mathrm{gas}}}\left(\dfrac{1}{T_{\mathrm{ref}}} - \dfrac{1}{T}\right)\right], & \text{type} = \text{"arrhenius"}
\end{cases}, \qquad
f_T(T) = \operatorname{clip}\bigl(f_T^{\mathrm{raw}}(T),\; f_{T,\min},\; f_{T,\max}\bigr)
$$

$$
f_\theta(\theta) =
\begin{cases}
\left[\operatorname{clip}\!\left(\dfrac{\theta - \theta_{\mathrm{wp}}}{\max(\theta_{\mathrm{fc}} - \theta_{\mathrm{wp}},\, c_{\min})},\; c_{\min},\; c_{\max}\right)\right]^{p_\theta}, & \theta \le \theta_{\mathrm{fc}} \\
u_1, & \theta > \theta_{\mathrm{fc}}
\end{cases}, \qquad
f_{\mathrm{env}}(T, \theta) = f_T(T)\,f_\theta(\theta)
$$

Structural litter experiences lignin inhibition $g_{\mathrm{str}}(L) = \exp(-c_{\mathrm{lig}} L)$ for validated $0 \le L \le L_{\max}$ (`lignin_n_ratio`), metabolic litter has $g_{\mathrm{met}} = 1$, and the three SOM pools experience clay physical protection $g_{\mathrm{clay}}(f_{\mathrm{clay}}) = (1 + c_{\mathrm{prot}} f_{\mathrm{clay}})^{-1}$ for $0 \le f_{\mathrm{clay}} \le c_{\max}$:

$$
D_i = k_{i,\mathrm{ref}}\,f_{\mathrm{env}}(T, \theta)\,g_i\,C_i, \qquad
g_i =
\begin{cases}
1, & i = \mathrm{met} \\
\exp(-c_{\mathrm{lig}} L), & i = \mathrm{str} \\
(1 + c_{\mathrm{prot}} f_{\mathrm{clay}})^{-1}, & i \in \{\mathrm{fast},\, \mathrm{slow},\, \mathrm{pass}\}
\end{cases}
$$

Passive-humification transfer fractions scale with clay content via $a_{\mathrm{clay}} = \min(1 + c_{\mathrm{enh}} f_{\mathrm{clay}},\; a_{\mathrm{clay,\max}})$, giving $T_{\mathrm{pass,fast}}^{\mathrm{eff}} = a_{\mathrm{clay}}\,T_{\mathrm{pass,fast}}$ and $T_{\mathrm{pass,slow}}^{\mathrm{eff}} = a_{\mathrm{clay}}\,T_{\mathrm{pass,slow}}$ (re-validated to satisfy $\sum_j T_{ji}^{\mathrm{eff}} \le 1$):

$$
\frac{dC_i}{dt} = I_i - D_i + \sum_{j} T_{ij}^{\mathrm{eff}} D_j, \qquad
R_{\mathrm{decomp}} = \sum_{i=1}^{5} D_i, \qquad
R_{\mathrm{hetero}} = I_{\mathrm{met}} + I_{\mathrm{str}} - \sum_{i=1}^{5} \frac{dC_i}{dt}
$$

### 2.4 Microbial-Enzyme Reverse Michaelis-Menten Closure (`MicrobialEnzymeDecomposition`)

Over substrates $S_i$ ($i \in \{\mathrm{met},\, \mathrm{str},\, \mathrm{som}\}$ corresponding to `metabolic_litter_c`, `structural_litter_c`, `available_som_c`) and microbial biomass $B$ (`microbial_biomass_c`, bounded in $[B_{\min}, B_{\max}]$), Arrhenius enzyme kinetics use $T_{\mathrm{ref}} = 298.15\,\mathrm{K}$ while maintenance respiration $Q_{10,\mathrm{maint}}$ (`temp_factor`) uses $T_0 = 273.15\,\mathrm{K}$ (`celsius_to_kelvin`, $0\,^\circ\mathrm{C}$):

$$
V_{\max,i}(T) = \operatorname{clip}\!\left(V_{\max,i,\mathrm{ref}}\exp\!\left[\frac{E_{a,V}}{R_{\mathrm{gas}}}\left(\frac{1}{T_{\mathrm{ref}}} - \frac{1}{T}\right)\right],\; \alpha_{V,\min} V_{\max,i,\mathrm{ref}},\; \alpha_{V,\max} V_{\max,i,\mathrm{ref}}\right)
$$

$$
K_{m,i}(T) = \operatorname{clip}\!\left(K_{m,i,\mathrm{ref}}\exp\!\left[\frac{E_{a,K}}{R_{\mathrm{gas}}}\left(\frac{1}{T_{\mathrm{ref}}} - \frac{1}{T}\right)\right],\; \alpha_{K,\min} K_{m,i,\mathrm{ref}},\; \alpha_{K,\max} K_{m,i,\mathrm{ref}}\right)
$$

$$
f_{\theta,\mathrm{enz}}(\theta) = \operatorname{clip}\!\left(
\begin{cases}
\bigl(\theta / \max(\theta_{\mathrm{opt}}, \epsilon_c)\bigr)^{1/n_\theta}, & \theta \le \theta_{\mathrm{opt}} \\
\bigl((1 - \theta) / \max(1 - \theta_{\mathrm{opt}}, \epsilon_c)\bigr)^{1/n_\theta}, & \theta > \theta_{\mathrm{opt}}
\end{cases},\; f_{\theta,\mathrm{enz,\min}},\; c_{\max}\right)
$$

With $V_{\max,i}^{\mathrm{eff}}(T,\theta) = V_{\max,i}(T)\,f_{\theta,\mathrm{enz}}(\theta)$, reverse Michaelis-Menten uptake $V_i$, temperature-dependent maintenance factor $f_{T,\mathrm{maint}}(T)$, turnover $D_{\mathrm{mic}}$, and $5 \times 4$ stoichiometric matrix $N_h$ on $(S_{\mathrm{met}}, S_{\mathrm{str}}, S_{\mathrm{som}}, B, R_{\mathrm{hetero}})$:

$$
V_i = \frac{B\,V_{\max,i}^{\mathrm{eff}}(T,\theta)\,S_i}{K_{m,i}(T) + B}, \qquad
f_{T,\mathrm{maint}}(T) = \operatorname{clip}\!\left(Q_{10,\mathrm{maint}}^{(T - T_0)/\Delta T_{10}},\; f_{T,\mathrm{enz,\min}},\; f_{T,\mathrm{enz,\max}}\right), \qquad
D_{\mathrm{mic}} = \bigl(\tau_{\mathrm{turn}} + \tau_{\mathrm{maint}} f_{T,\mathrm{maint}}(T)\bigr) B
$$

$$
N_h =
\begin{bmatrix}
-1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
\mathrm{CUE} & \mathrm{CUE} & \mathrm{CUE} & -1 \\
1 - \mathrm{CUE} & 1 - \mathrm{CUE} & 1 - \mathrm{CUE} & 1
\end{bmatrix}, \qquad
r_h = \begin{bmatrix} V_{\mathrm{met}} & V_{\mathrm{str}} & V_{\mathrm{som}} & D_{\mathrm{mic}} \end{bmatrix}^{\mathsf{T}}, \qquad
\frac{dx_h}{dt} = N_h r_h
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $C_{\mathrm{met}}, C_{\mathrm{str}}, C_{\mathrm{fast}}, C_{\mathrm{slow}}, C_{\mathrm{pass}}$ | Five canonical litter and SOM carbon stocks | $\mathrm{g\,C\,m^{-2}}$ | `metabolic_litter_c`, `structural_litter_c`, `fast_som_c`, `slow_som_c`, `passive_som_c` | `state` |
| $S_{\mathrm{som}}, B$ | Enzyme-accessible SOM stock and microbial biomass carbon (`MicrobialEnzymeDecomposition`) | $\mathrm{g\,C\,m^{-2}}$ | `available_som_c`, `microbial_biomass_c` | `state` |
| $T, \theta$ | Soil temperature and volumetric soil moisture | $\mathrm{K}$, $\mathrm{m^3\,m^{-3}}$ | `soil_temperature`, `soil_moisture` | `forcing` |
| $L, f_{\mathrm{clay}}$ | Structural-litter lignin-to-nitrogen ratio and soil clay fraction (`FirstOrderDecomposition`) | $\mathrm{1}$ | `lignin_n_ratio`, `clay_content` | `forcing` |
| $f_{\mathrm{env}}$ | External environmental rate multiplier (`PoolsMatrix`) | $\mathrm{1}$ | `environmental_modifier` | `forcing` |
| $I_{\mathrm{met}}, I_{\mathrm{str}}$ | Boundary litter carbon input rates (`FirstOrderDecomposition`, `PoolsMatrix`) | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `metabolic_litter_input`, `structural_litter_input` | `flux` |
| $I_{\mathrm{fast}}, I_{\mathrm{slow}}, I_{\mathrm{pass}}$ | Boundary SOM carbon input rates (`PoolsMatrix`) | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `fast_som_input`, `slow_som_input`, `passive_som_input` | `flux` |
| $R_{\mathrm{decomp}}$ | Gross five-pool carbon decomposition rate (`FirstOrderDecomposition`, `PoolsMatrix`) | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `carbon_decomposition_rate` | `flux` |
| $R_{\mathrm{hetero}}$ | Outward heterotrophic respiration carbon flux to atmosphere | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `heterotrophic_respiration` | `flux` |
| $z_h, C_{\mathrm{total}}$ | Stacked five-pool snapshot and cell-wise total carbon stock (`CarbonPoolsFunctor`) | $\mathrm{g\,C\,m^{-2}}$ | `diagnostics.carbon_pools_snapshot`, `diagnostics.total_carbon_stock` | `diagnostic` |
| $D_i, T_{ij}^{\mathrm{eff}} D_j, f_T, f_\theta, f_{\mathrm{env}}, \Delta t$ | Pool decay fluxes, selected transfers, modifiers, and step length (`FirstOrderDecomposition`, 12 keys) | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$, $\mathrm{1}$, $\mathrm{s}$ | `diagnostics.first_order_decomposition.*` | `diagnostic` |
| $R_{\mathrm{step}}, T_{\mathrm{step}}, \Delta C_{\mathrm{net}}, C_{\mathrm{tot}}, r_C$ | Step-integrated respiration, transfers, post-step stocks, and residuals (`PoolsMatrix`, 9 keys) | $\mathrm{g\,C\,m^{-2}}$, $\mathrm{1}$ | `diagnostics.pools_matrix.*` | `diagnostic` |
| $G_{\mathrm{mic}}, D_{\mathrm{mic}}, V_i, V_{\max,i}^{\mathrm{eff}}, K_{m,i}, a_i, r_C$ | Microbial growth/death/respiration, substrate kinetics, CUE, residual, and step length (`MicrobialEnzymeDecomposition`, 24 keys) | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$, $\mathrm{s^{-1}}$, $\mathrm{g\,C\,m^{-2}}$, $\mathrm{1}$, $\mathrm{s}$ | `diagnostics.microbial_enzyme.*` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\mathrm{names}, f_{\mathrm{env,ref}}$ | `pedosphere.decomposition.pool_names`, `pedosphere.numerical.default_environmental_modifier` | `list`, $\mathrm{1}$ | `pedosphere/decomposition.yaml`, `pedosphere/numerical.yaml` | Parton et al. (1987); Sierra et al. (2012) |
| $k_{1..5}$ (`PoolsMatrix`) | `pedosphere.carbon_pools.turnover_rate_{metabolic,structural,fast_som,slow_som,passive_som}` | $\mathrm{s^{-1}}$ | `pedosphere/carbon_pools_parameters.yaml` | Sierra et al. (2012) |
| $k_{1..5,\mathrm{yr}}$ (`FirstOrder`) | `pedosphere.decomposition.{k_metabolic,k_structural,k_fast_som,k_slow_som,k_passive_som}` | $\mathrm{yr^{-1}}$ | `pedosphere/decomposition.yaml` | Parton et al. (1987); Coleman and Jenkinson (1996) |
| $T_{ij}$ | `pedosphere.decomposition.{t_met_fast,t_met_slow,t_str_fast,t_str_slow,t_fast_slow,t_fast_passive,t_slow_fast,t_slow_passive}` | $\mathrm{1}$ | `pedosphere/decomposition.yaml` | Parton et al. (1987); Coleman and Jenkinson (1996) |
| $m_T, p_{25}, Q_{10}, E_a$ | `pedosphere.decomposition.{temp_response_type,param_25}`, `pedosphere.physical.{q10_decomposition,activation_energy_decomposition}` | `str`, $\mathrm{1}$, $\mathrm{1}$, $\mathrm{J\,mol^{-1}}$ | `pedosphere/decomposition.yaml`, `pedosphere/physical.yaml` | Lloyd and Taylor (1994); Davidson and Janssens (2006) |
| $\theta_{\mathrm{fc}}, \theta_{\mathrm{wp}}, p_\theta$ | `pedosphere.decomposition.{field_capacity,wilting_point,moisture_response_power}` | $\mathrm{m^3\,m^{-3}}$, $\mathrm{m^3\,m^{-3}}$, $\mathrm{1}$ | `pedosphere/decomposition.yaml` | Parton et al. (1987) |
| $c_{\mathrm{lig}}, L_{\max}, c_{\mathrm{prot}}, c_{\mathrm{enh}}, a_{\mathrm{clay,\max}}$ | `pedosphere.decomposition.{lignin_inhibition_coefficient,max_lignin_n_ratio,clay_protection_coefficient,clay_enhancement_coefficient,max_clay_factor}` | $\mathrm{1}$ | `pedosphere/decomposition.yaml` | Parton et al. (1987); Coleman and Jenkinson (1996) |
| $T_{\mathrm{bio,\min}}, T_{\mathrm{bio,\max}}, f_{T,\min}, f_{T,\max}, u_1$ | `pedosphere.physical.{temperature_minimum_biological,temperature_maximum_biological}`, `pedosphere.numerical.{temperature_response_min,temperature_response_max,unity_multiplier}` | $\mathrm{K}$, $\mathrm{1}$ | `pedosphere/physical.yaml`, `pedosphere/numerical.yaml` | Biological domain and modifier bounds |
| $\mathrm{tol}_m, \epsilon, C_{\max}, c_{\min}, c_{\max}$ | `pedosphere.numerical.{mass_balance_tolerance,epsilon,max_pool_size,clamp_min,clamp_max}` | $\mathrm{1}$, $\mathrm{1}$, $\mathrm{g\,C\,m^{-2}}$, $\mathrm{1}$, $\mathrm{1}$ | `pedosphere/numerical.yaml` | Numerical tolerance and domain bounds |
| $V_{\max,i,\mathrm{ref}}, K_{m,i,\mathrm{ref}}$ | `pedosphere.decomposition.microbial_enzyme.kinetics.{metabolic,structural,som}.{vmax_ref,km_ref}` | $\mathrm{s^{-1}}$, $\mathrm{g\,C\,m^{-2}}$ | `pedosphere/decomposition.yaml` | Wieder et al. (2013); Manzoni and Porporato (2009) |
| $E_{a,V}, E_{a,K}, \alpha_{V,\min/\max}, \alpha_{K,\min/\max}$ | `pedosphere.decomposition.microbial_enzyme.{ea_vmax,ea_km,vmax_clamp_min_factor,vmax_clamp_max_factor,km_clamp_min_factor,km_clamp_max_factor}` | $\mathrm{J\,mol^{-1}}$, $\mathrm{1}$ | `pedosphere/decomposition.yaml` | Davidson et al. (2006); Wieder et al. (2013) |
| $\theta_{\mathrm{opt}}, n_\theta, Q_{10,\mathrm{maint}}$ | `pedosphere.decomposition.microbial_enzyme.{moisture_optimum,moisture_power}`, `pedosphere.decomposition.temp_factor` | $\mathrm{m^3\,m^{-3}}$, $\mathrm{1}$, $\mathrm{1}$ | `pedosphere/decomposition.yaml` | Wieder et al. (2013); Lloyd and Taylor (1994) |
| $\mathrm{CUE}, \tau_{\mathrm{turn}}, \tau_{\mathrm{maint}}, B_{\min}, B_{\max}$ | `pedosphere.decomposition.microbial_enzyme.microbial.{carbon_use_efficiency,turnover_rate,maintenance_rate,min_biomass,max_biomass}` | $\mathrm{1}$, $\mathrm{s^{-1}}$, $\mathrm{s^{-1}}$, $\mathrm{g\,C\,m^{-2}}$, $\mathrm{g\,C\,m^{-2}}$ | `pedosphere/decomposition.yaml` | Wieder et al. (2013) |
| $\epsilon_c, f_{\theta,\mathrm{enz,\min}}, f_{T,\mathrm{enz,\min/\max}}$ | `pedosphere.numerical.{epsilon_concentration,moisture_response_clamp_min,temp_response_clamp_min,temp_response_clamp_max}` | $\mathrm{1}$ | `pedosphere/numerical.yaml` | Enzyme diagnostic floor and modifier bounds |
| $s_{\mathrm{yr}}, T_{\mathrm{ref}}, T_0, \Delta T_{10}, R_{\mathrm{gas}}$ | `relic.physical.constants.time.seconds_per_year`, `relic.physical.constants.temperature.{reference_temperature,celsius_to_kelvin,q10_interval}`, `relic.physical.constants.universal.gas_constant` | $\mathrm{s}$, $\mathrm{K}$, $\mathrm{K}$, $\mathrm{K}$, $\mathrm{J\,mol^{-1}\,K^{-1}}$ | `physical/constants.yaml` | CODATA 2018 / SI constants |

## 5. Numerical Methods and Discretization

All three reaction closures advance over one positive scalar step $\Delta t$ ($\mathrm{s}$) via a single call to `advance_reaction_network` on differentiable `torch.Tensor` states:

- Prognostic pool updates are emitted as `UpdateItem(mode=UpdateMode.ACCUMULATE, per_timestep=True)` in $\mathrm{g\,C\,m^{-2}}$ (`PoolsMatrix` emits exact floating-point representable increments $\Delta C = (C + \Delta C_{\mathrm{raw}}) - C$).
- Process rates (`carbon_decomposition_rate`, `heterotrophic_respiration`) and all `diagnostics.*` keys use `UpdateItem(mode=UpdateMode.REPLACE, per_timestep=False)` (`CarbonPoolsFunctor` returns diagnostic tensors directly with zero state writes).
- Proposed post-step pool negativity, substrate overdraw ($V_i \Delta t > S_i$), microbial biomass excursions outside $[B_{\min}, B_{\max}]$, out-of-range drivers, or transfer column sums exceeding $1 + c_{\min}$ raise `ValueError` without silent clipping.

## 6. Conservation, Boundary Conditions, and Ledgers

- **`CarbonPoolsFunctor`**: pure observation ($\Delta x_h = 0$).
- **`FirstOrderDecomposition` and `PoolsMatrix`**: open-system carbon ledgers (`open_system_route="port"`) balancing internal pool increments against boundary litter/SOM inputs and outward atmospheric `heterotrophic_respiration`:

$$
\sum_{i=1}^{5} \Delta C_i + R_{\mathrm{hetero}}\,\Delta t - \sum_{i} I_i\,\Delta t = 0
$$

  `carbon_decomposition_rate` is an internal gross transformation observation consumed downstream by `MineralizationProcess` and is excluded from boundary carbon summation.
- **`MicrobialEnzymeDecomposition`**: closed substrate-to-port reaction incidence satisfying $C_h N_h = [0, 0, 0, 0]$ for $C_h = [1, 1, 1, 1, 1]$, yielding $\Delta S_{\mathrm{met}} + \Delta S_{\mathrm{str}} + \Delta S_{\mathrm{som}} + \Delta B + R_{\mathrm{hetero}}\,\Delta t = 0$.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `CarbonPoolsFunctor` | `relic/scientific/pedosphere/carbon_pools/carbon_pools_functor.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.POSTPROCESSING` (`TemporalScale.SECOND`, `SpatialScale.GRID_CELL`) |
| `PoolsMatrix` | `relic/scientific/pedosphere/carbon_pools/pools_matrix.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`TemporalScale.SECOND`, `SpatialScale.GRID_CELL`) |
| `FirstOrderDecomposition` | `relic/scientific/pedosphere/decomposition/first_order.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`TemporalScale.SECOND`, `SpatialScale.GRID_CELL`) |
| `MicrobialEnzymeDecomposition` | `relic/scientific/pedosphere/decomposition/microbial_enzyme.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`TemporalScale.SECOND`, `SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `CarbonPoolsFunctor`: `[I]`, IPHS **Level 1** (`observation`, certificate `pedosphere_carbon_pools_carbon_pools_functor_carbon_pools_functor.yaml`).
- `PoolsMatrix`: `[I]`, IPHS **Level 2** (`reaction`, certificate `pedosphere_carbon_pools_pools_matrix_pools_matrix.yaml`).
- `FirstOrderDecomposition`: `[H]`, IPHS **Level 2** (`reaction`, certificate `pedosphere_decomposition_first_order_first_order_decomposition.yaml`; wraps assembled net tendencies and respiration export via $N_h = I_6$ without edge-resolved chemical affinities).
- `MicrobialEnzymeDecomposition`: `[I]`, IPHS **Level 2** (`reaction`, certificate `pedosphere_decomposition_microbial_enzyme_microbial_enzyme_decomposition.yaml`; conservative $5 \times 4$ incidence matrix $C_h N_h = 0$ without Hamiltonian free-energy potentials).

Frozen certificate expressions:

```text
C_total = sum_i C_i
z_h = O_h x_h; delta_x = 0
dC/dt = I + T*K*C - diag(K)*C with mass balance
x_dot = N_h r_h with ParameterManager-backed process rates r_h and declared stoichiometric/output map N_h
dC/dt = -k(T,theta) * C with k(T,theta) = k_ref * f_T(T) * f_theta(theta)
V = (MIC * Vmax(T,theta) * [S]) / (Km(T) + MIC)
```

## 9. References

- Burke, I. C., Kaye, J. P., Bird, S. P., Hall, S. A., McCulley, R. L., and Sommerville, G. L. (2003). Evaluating and testing models of terrestrial biogeochemistry: The role of temperature in controlling decomposition. In *Models in Ecosystem Science* (pp. 225-253). Princeton University Press.
- Coleman, K., and Jenkinson, D. S. (1996). RothC-26.3: A model for the turnover of carbon in soil. In *Evaluation of Soil Organic Matter Models* (NATO ASI Series, Vol. 38, pp. 237-246). Springer. DOI: `10.1007/978-3-642-61094-3_17`.
- Davidson, E. A., and Janssens, I. A. (2006). Temperature sensitivity of soil carbon decomposition and feedbacks to climate change. *Nature*, 440, 165-173. DOI: `10.1038/nature04514`.
- Jenkinson, D. S., and Rayner, J. H. (1977). The turnover of soil organic matter in some of the Rothamsted classical experiments. *Soil Science*, 123(5), 298-305.
- Lloyd, J., and Taylor, J. A. (1994). On the temperature dependence of soil respiration. *Functional Ecology*, 8(3), 315-323. DOI: `10.2307/2389824`.
- Manzoni, S., and Porporato, A. (2009). Soil carbon and nitrogen mineralization: Theory and models across scales. *Soil Biology and Biochemistry*, 41(7), 1355-1379. DOI: `10.1016/j.soilbio.2009.02.031`.
- Melillo, J. M., Aber, J. D., and Muratore, J. F. (1982). Nitrogen and lignin control of hardwood leaf litter decomposition dynamics. *Ecology*, 63(3), 621-626.
- Parton, W. J., Schimel, D. S., Cole, C. V., and Ojima, D. S. (1987). Analysis of factors controlling soil organic matter levels in Great Plains grasslands. *Soil Science Society of America Journal*, 51(5), 1173-1179. DOI: `10.2136/sssaj1987.03615995005100050015x`.
- Sierra, C. A., Muller, M., and Trumbore, S. E. (2012). Models of soil organic matter decomposition: the SoilR package, version 1.0. *Geoscientific Model Development*, 5, 1045-1060. DOI: `10.5194/gmd-5-1045-2012`.
- Todd-Brown, K. E. O., Randerson, J. T., Post, W. M., et al. (2013). Causes of variation in soil carbon simulations from CMIP5 Earth system models and comparison with observations. *Biogeosciences*, 10, 1717-1736. DOI: `10.5194/bg-10-1717-2013`.
- Wieder, W. R., Bonan, G. B., and Allison, S. D. (2013). Global soil carbon projections are improved by modelling microbial processes. *Nature Climate Change*, 3, 909-912. DOI: `10.1038/nclimate1951`.
