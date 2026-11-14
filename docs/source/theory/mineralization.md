# Soil Nitrogen Mineralization and Immobilization

## 1. Process Definition and Scope

`MineralizationProcess` (`relic/scientific/pedosphere/nutrient_cycling/mineralization.py`, `[I]`) models Mineralization-Immobilization Turnover (MIT) across three soil nitrogen stocks (`organic_nitrogen`, `nh4_nitrogen`, `microbial_nitrogen`, $\mathrm{g\,N\,m^{-2}}$) coupled to the upstream gross carbon decomposition flux (`carbon_decomposition_rate`, $\mathrm{g\,C\,m^{-2}\,s^{-1}}$) and an open organic-material nitrogen boundary port (`organic_nitrogen_input`, $\mathrm{g\,N\,m^{-2}\,s^{-1}}$).

`FirstOrderDecomposition` and `PoolsMatrix` are the mutually exclusive upstream writers of `carbon_decomposition_rate`; `LitterfallInterfaceMapper` supplies `organic_nitrogen_input` (retranslocation-adjusted plant litter N plus dependency-selected fecal organic N). Direct nitrate immobilization, explicit carbon pool mutation within `MineralizationProcess`, and `carbon_decomposition_rate` coupling from `MicrobialEnzymeDecomposition` remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 23 (Section 23.3).

## 2. Governing Equations

### 2.1 Potential Transformation Rates and Environmental Modifiers

With annual mineralization constant $k_{\min,\mathrm{yr}}$ ($\mathrm{yr^{-1}}$) converted once via $k_{\min} = k_{\min,\mathrm{yr}} / s_{\mathrm{yr}}$ ($\mathrm{s^{-1}}$), upstream gross carbon decomposition $R_C$ (`carbon_decomposition_rate`), microbial carbon-use efficiency $\mathrm{CUE}$, microbial C:N mass ratio $\mathrm{CN}_{\mathrm{mic}}$, mineral-N assimilation efficiency $\eta_N$, and microbial death constant $k_{\mathrm{death}}$ ($\mathrm{s^{-1}}$):

$$
R_{\mathrm{gross}}^{*} = k_{\min}\,N_{\mathrm{org}}\,f_T(T)\,f_\theta(\theta)\,f_{\mathrm{pH}}(\mathrm{pH}), \qquad
C_{\mathrm{uptake}} = \mathrm{CUE}\,R_C, \qquad
R_{\mathrm{demand}} = \frac{C_{\mathrm{uptake}}}{\mathrm{CN}_{\mathrm{mic}}\,\eta_N}, \qquad
R_{\mathrm{death}}^{*} = k_{\mathrm{death}}\,N_{\mathrm{mic}}
$$

The separable environmental modifiers for $T > 0$ ($\mathrm{K}$), $\theta \in [0, 1]$ ($\mathrm{m^3\,m^{-3}}$), and $\mathrm{pH} \in [0, 14]$ are bounded prior to multiplication:

$$
f_T^{\mathrm{raw}}(T) =
\begin{cases}
Q_{10}^{(T - T_{\mathrm{ref}})/\Delta T_{10}}, & \text{type} = \text{"q10"} \\
\exp\!\left[\dfrac{E_a}{R_{\mathrm{gas}}}\left(\dfrac{1}{T_{\mathrm{ref}}} - \dfrac{1}{T}\right)\right], & \text{type} = \text{"arrhenius"}
\end{cases}, \qquad
f_T(T) = \operatorname{clip}\bigl(f_T^{\mathrm{raw}}(T),\; f_{T,\min},\; f_{T,\max}\bigr)
$$

$$
f_\theta(\theta) = \operatorname{clip}\!\left(\exp\!\left[-\left(\frac{\theta - \theta_{\mathrm{opt}}}{w_\theta}\right)^2\right],\; f_{\theta,\min},\; f_{\theta,\max}\right), \qquad
f_{\mathrm{pH}}(\mathrm{pH}) = \operatorname{clip}\!\left(\exp\!\left[-\left(\frac{\mathrm{pH} - \mathrm{pH}_{\mathrm{opt}}}{w_{\mathrm{pH}}}\right)^2\right],\; f_{\mathrm{pH,\min}},\; f_{\mathrm{pH,\max}}\right)
$$

Here `pedosphere.mineralization.moisture_response_power` acts as the Gaussian moisture width $w_\theta$, and the pH Gaussian exponent carries no $1/2$ prefactor.

### 2.2 Donor-Bounded Rates, Stoichiometric Matrix, and Boundary Routing

Over positive scalar step $\Delta t$ ($\mathrm{s}$), each internal channel is independently bounded by its beginning-of-step donor stock:

$$
R_{\mathrm{gross}} = \min\!\left(R_{\mathrm{gross}}^{*},\; \frac{N_{\mathrm{org}}}{\Delta t}\right), \qquad
R_{\mathrm{immob}} = \min\!\left(R_{\mathrm{demand}},\; \frac{N_{\mathrm{NH_4}}}{\Delta t}\right), \qquad
R_{\mathrm{death}} = \min\!\left(R_{\mathrm{death}}^{*},\; \frac{N_{\mathrm{mic}}}{\Delta t}\right), \qquad
R_{\mathrm{net}} = R_{\mathrm{gross}} - R_{\mathrm{immob}}
$$

On species vector $x_h = [N_{\mathrm{org}},\; N_{\mathrm{NH_4}},\; N_{\mathrm{mic}}]^{\mathsf{T}}$, reaction vector $r_h = [R_{\mathrm{gross}},\; R_{\mathrm{immob}},\; R_{\mathrm{death}}]^{\mathsf{T}}$, and organic-material boundary input $u_{\mathrm{litter}} = F_{\mathrm{org},N}$ (`organic_nitrogen_input`):

$$
N_h =
\begin{bmatrix}
-1 & 0 & 1 \\
1 & -1 & 0 \\
0 & 1 & -1
\end{bmatrix}, \qquad
G_h = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}^{\mathsf{T}}, \qquad
\frac{dx_h}{dt} = N_h r_h + G_h u_{\mathrm{litter}}, \qquad
\Delta x_h = \bigl(N_h r_h + G_h u_{\mathrm{litter}}\bigr)\,\Delta t
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $N_{\mathrm{org}}, N_{\mathrm{NH_4}}, N_{\mathrm{mic}}$ | Soil organic, ammonium, and microbial nitrogen stocks (required inputs and per-step increment outputs) | $\mathrm{g\,N\,m^{-2}}$ | `organic_nitrogen`, `nh4_nitrogen`, `microbial_nitrogen` | `state` |
| $T, \theta, \mathrm{pH}$ | Soil temperature, volumetric soil moisture, and soil pH | $\mathrm{K}$, $\mathrm{m^3\,m^{-3}}$, $\mathrm{1}$ | `soil_temperature`, `soil_moisture`, `soil_ph` | `forcing` |
| $R_C$ | Upstream gross carbon decomposition rate | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `carbon_decomposition_rate` | `flux` |
| $F_{\mathrm{org},N}$ | Plant-litter plus selected fecal organic nitrogen boundary input rate | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `organic_nitrogen_input` | `flux` |
| $R_{\mathrm{net}}, R_{\mathrm{gross}}, R_{\mathrm{immob}}, R_{\mathrm{death}}$ | Net mineralization, gross mineralization, immobilization, and microbial death rates | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `mineralization_flux`, `gross_mineralization_flux`, `immobilization_flux`, `microbial_death_flux` | `flux` |
| $F_{\mathrm{org},N}\Delta t$ | Step-integrated organic-material nitrogen boundary input | $\mathrm{g\,N\,m^{-2}}$ | `diagnostics.mineralization.organic_nitrogen_boundary_input` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $k_{\min,\mathrm{yr}}, k_{\mathrm{death}}$ | `pedosphere.mineralization.k_mineralization`, `pedosphere.nutrient_cycling.k_microbial_death` | $\mathrm{yr^{-1}}$, $\mathrm{s^{-1}}$ | `pedosphere/mineralization.yaml`, `pedosphere/nutrient_cycling.yaml` | Stanford and Smith (1972); Schimel and Weintraub (2003) |
| $m_T, Q_{10}, E_a$ | `pedosphere.mineralization.{temp_response_type,q10_mineralization,activation_energy}` | `str`, $\mathrm{1}$, $\mathrm{J\,mol^{-1}}$ | `pedosphere/mineralization.yaml` | Kirschbaum (1995); Manzoni and Porporato (2009) |
| $\theta_{\mathrm{opt}}, w_\theta$ | `pedosphere.mineralization.{moisture_optimum,moisture_response_power}` | $\mathrm{m^3\,m^{-3}}$, $\mathrm{1}$ | `pedosphere/mineralization.yaml` | Moyano et al. (2013); Manzoni and Porporato (2009) |
| $\mathrm{CUE}, \mathrm{CN}_{\mathrm{mic}}, \eta_N$ | `pedosphere.mineralization.{carbon_use_efficiency,cn_ratio_microbial,n_assimilation_efficiency}` | $\mathrm{1}$ | `pedosphere/mineralization.yaml` | Manzoni and Porporato (2009); Grant et al. (1993) |
| $\mathrm{pH}_{\mathrm{opt}}, w_{\mathrm{pH}}$ | `pedosphere.physical.{ph_optimum_decomposition,ph_response_width}` | $\mathrm{1}$ | `pedosphere/physical.yaml` | Curtin et al. (1998) |
| $f_{T,\min/\max}, f_{\theta,\min/\max}, f_{\mathrm{pH},\min/\max}$ | `pedosphere.numerical.{temperature_response_min,temperature_response_max,moisture_response_min,moisture_response_max,ph_response_min,ph_response_max}` | $\mathrm{1}$ | `pedosphere/numerical.yaml` | Dimensionless environmental modifier bounds |
| $T_{\mathrm{ref}}, s_{\mathrm{yr}}, R_{\mathrm{gas}}, \Delta T_{10}$ | `relic.physical.constants.temperature.{reference_temperature,q10_interval}`, `relic.physical.constants.time.seconds_per_year`, `relic.physical.constants.universal.gas_constant` | $\mathrm{K}$, $\mathrm{K}$, $\mathrm{s}$, $\mathrm{J\,mol^{-1}\,K^{-1}}$ | `physical/constants.yaml` | CODATA 2018 / SI constants |

## 5. Numerical Methods and Discretization

`MineralizationProcess.forward` validates all eight required tensor inputs (rejecting missing, non-finite, negative stocks/fluxes, $T \le 0$, $\theta \notin [0, 1]$, or $\mathrm{pH} \notin [0, 14]$), applies beginning-of-step donor caps to form $r_h$, advances the closed internal reaction network once via `advance_reaction_network`, and adds the open boundary step increment $F_{\mathrm{org},N}\Delta t$ to `organic_nitrogen`.

- Pool increments (`organic_nitrogen`, `nh4_nitrogen`, `microbial_nitrogen`) use `UpdateItem(mode=UpdateMode.ACCUMULATE, per_timestep=True)` in $\mathrm{g\,N\,m^{-2}}$ (no duplicate `delta_*` keys are emitted).
- Rate outputs (`mineralization_flux`, `gross_mineralization_flux`, `immobilization_flux`, `microbial_death_flux`) and `diagnostics.mineralization.organic_nitrogen_boundary_input` use `UpdateItem(mode=UpdateMode.REPLACE, per_timestep=False)`.

## 6. Conservation, Boundary Conditions, and Ledgers

With $C_h = [1, 1, 1]$, the internal stoichiometric matrix satisfies exact column-wise closure $C_h N_h = [0, 0, 0]$ because microbial mortality returns nitrogen internally to `organic_nitrogen`. Including the single external port $G_h = [1, 0, 0]^{\mathsf{T}}$, `MineralizationProcess.check_conservation_laws()` verifies the open-system nitrogen ledger:

$$
\Delta N_{\mathrm{org}} + \Delta N_{\mathrm{NH_4}} + \Delta N_{\mathrm{mic}} - F_{\mathrm{org},N}\,\Delta t = 0
$$

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `MineralizationProcess` | `relic/scientific/pedosphere/nutrient_cycling/mineralization.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`TemporalScale.SECOND`, `SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `MineralizationProcess`: `[I]`, IPHS **Level 2** (`reaction`, certificate `pedosphere_nutrient_cycling_mineralization_mineralization_process.yaml`). Donor-bounded three-pool MIT reaction network with $C_h N_h = 0$ internal conservation and single-port $G_h u_{\mathrm{litter}}$ organic-nitrogen routing.

Frozen certificate expressions:

```text
The retired runtime required an unowned organic_carbon input and evaluated a second DEC/CN_substrate nitrogen-release term.
x_dot = N_h r_h + G_h u_litter; C_h N_h = 0 and sum(Delta N) = u_litter dt
```

## 9. References

- Curtin, D., Campbell, C. A., and Jalil, A. (1998). Effects of soil pH on carbon dioxide evolution, nitrogen mineralization, and soluble organic matter. *Soil Biology and Biochemistry*, 30(1), 57-64.
- Grant, R. F., Juma, N. G., and McGill, W. B. (1993). Simulation of carbon and nitrogen transformations in soil: Mineralization. *Soil Biology and Biochemistry*, 25(10), 1317-1329.
- Jansson, S. L., and Persson, J. (1982). Mineralization and immobilization of soil nitrogen. In *Nitrogen in Agricultural Soils* (Agronomy Monograph 22, pp. 229-252). ASA-CSSA-SSSA.
- Kirschbaum, M. U. F. (1995). The temperature dependence of soil organic matter decomposition, and the effect of global warming on soil organic C storage. *Soil Biology and Biochemistry*, 27(6), 753-760.
- Lawrence, D. M., Fisher, R. A., Koven, C. D., et al. (2019). The Community Land Model Version 5: Description of new features, benchmarking, and impact of forcing uncertainty. *Journal of Advances in Modeling Earth Systems*, 11(12), 4245-4287.
- Manzoni, S., and Porporato, A. (2009). Soil carbon and nitrogen mineralization: Theory and models across scales. *Soil Biology and Biochemistry*, 41(7), 1355-1379. DOI: `10.1016/j.soilbio.2009.02.031`.
- Moyano, F. E., Manzoni, S., and Chenu, C. (2013). Responses of soil heterotrophic respiration to moisture availability: An exploration of processes and models. *Soil Biology and Biochemistry*, 59, 72-85.
- Parton, W. J., Schimel, D. S., Cole, C. V., and Ojima, D. S. (1987). Analysis of factors controlling soil organic matter levels in Great Plains grasslands. *Soil Science Society of America Journal*, 51(5), 1173-1179.
- Schimel, J. P., and Weintraub, M. N. (2003). The implications of exoenzyme activity on microbial carbon and nitrogen limitation in soil: a theoretical model. *Soil Biology and Biochemistry*, 35(4), 549-563.
- Stanford, G., and Smith, S. J. (1972). Nitrogen mineralization potentials of soils. *Soil Science Society of America Proceedings*, 36(3), 465-472.
