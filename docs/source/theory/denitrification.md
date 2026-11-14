# Soil Denitrification

## 1. Process Definition and Scope

`DenitrificationProcess` (`relic/scientific/pedosphere/nutrient_cycling/denitrification.py`, `[I]`) computes anaerobic microbial reduction of soil nitrate (`no3_nitrogen`, $\mathrm{g\,N\,m^{-2}}$) into nitrous oxide (`n2o_flux`, $\mathrm{g\,N\,m^{-2}\,s^{-1}}$) and dinitrogen (`n2_production`, $\mathrm{g\,N\,m^{-2}\,s^{-1}}$) using dual Michaelis-Menten substrate limitation (`no3_nitrogen` and `labile_carbon`) modulated by temperature, water-filled pore space (`water_filled_pore_space`), and soil pH (`soil_ph`).

All five state and environmental inputs are required runtime tensors without default substitution. `nitrification` is the sole named scheduler dependency; `labile_carbon` is consumed through the state interface from whichever carbon decomposition closure is active. Explicit intermediate nitrite/nitric-oxide pools (`NO2-`, `NO`), electron-balance coupling to `labile_carbon` depletion, and DAYCENT arctangent WFPS curves remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 23 (Section 23.4).

## 2. Governing Equations

### 2.1 Potential Kinetic Rate and Environmental Modifiers

For nitrate stock $[\mathrm{NO_3^-}]$ ($\mathrm{g\,N\,m^{-2}}$), labile carbon stock $C_{\mathrm{lab}}$ ($\mathrm{g\,C\,m^{-2}}$), soil temperature $T$ ($\mathrm{K}$), water-filled pore space $\mathrm{WFPS} \in [0, 1]$, and soil $\mathrm{pH}$:

$$
R_{\mathrm{denit}} = V_{\max}\,\frac{[\mathrm{NO_3^-}]}{K_{m,\mathrm{NO_3}} + [\mathrm{NO_3^-}]}\,f_T(T)\,f_{\mathrm{anaer}}(\mathrm{WFPS})\,f_C(C_{\mathrm{lab}})\,f_{\mathrm{pH}}(\mathrm{pH})
$$

$$
f_T(T) = \operatorname{clip}\!\left(Q_{10}^{(T - T_{\mathrm{ref}})/\Delta T_{10}},\; f_{T,\min},\; f_{T,\max}\right), \qquad
f_C(C_{\mathrm{lab}}) = \frac{C_{\mathrm{lab}}}{K_{m,C} + C_{\mathrm{lab}}}
$$

$$
f_{\mathrm{anaer}}(\mathrm{WFPS}) = \max\!\left(\operatorname{clip}\!\left(\frac{\mathrm{WFPS} - \mathrm{WFPS}_{\mathrm{thr}}}{w_{\mathrm{anaer}}},\; 0,\; 1\right),\; f_{\mathrm{anaer,\min}}\right)
$$

$$
f_{\mathrm{pH}}(\mathrm{pH}) = \operatorname{clip}\!\left(\exp\!\left[-\left(\frac{\mathrm{pH} - \mathrm{pH}_{\mathrm{opt}}}{w_{\mathrm{pH}}}\right)^2\right],\; f_{\mathrm{pH,\min}},\; 1\right)
$$

### 2.2 Gaseous Product Partitioning and Donor-Bounded Stoichiometric Network

Nitrate availability and low pH increase the $\mathrm{N_2O}$ product fraction $f_{\mathrm{N_2O}}$ through a separate dimensionless modifier $g_{\mathrm{pH}}$:

$$
g_{\mathrm{pH}}(\mathrm{pH}) = \operatorname{clip}\!\left(\frac{\mathrm{pH}_{\mathrm{ref}} - \mathrm{pH}}{s_{\mathrm{pH}}},\; g_{\mathrm{pH,\min}},\; g_{\mathrm{pH,\max}}\right), \qquad
f_{\mathrm{N_2O}} = \operatorname{clip}\!\left(f_{\mathrm{base}} + a_{\mathrm{NO_3}}\frac{[\mathrm{NO_3^-}]}{[\mathrm{NO_3^-}] + N_{\mathrm{thr}}}\,g_{\mathrm{pH}}(\mathrm{pH}),\; f_{\mathrm{base}},\; f_{\max}\right)
$$

Before assembling the reaction-rate vector over positive scalar step $\Delta t$ ($\mathrm{s}$), the realized rate $R_{\mathrm{exec}}$ enforces beginning-of-step nitrate donor availability:

$$
R_{\mathrm{exec}} = \max\!\left(\min\!\left(R_{\mathrm{denit}},\; \frac{[\mathrm{NO_3^-}]}{\Delta t}\right),\; 0\right), \qquad
r_{\mathrm{N_2O}} = R_{\mathrm{exec}}\,f_{\mathrm{N_2O}}, \qquad
r_{\mathrm{N_2}} = R_{\mathrm{exec}}\,(1 - f_{\mathrm{N_2O}})
$$

With species basis $x_h = [\mathrm{NO_3^-},\; \mathrm{N_2O},\; \mathrm{N_2}]^{\mathsf{T}}$ and reaction vector $r_h = [r_{\mathrm{N_2O}},\; r_{\mathrm{N_2}}]^{\mathsf{T}}$:

$$
N_h =
\begin{bmatrix}
-1 & -1 \\
1 & 0 \\
0 & 1
\end{bmatrix}, \qquad
\frac{dx_h}{dt} = N_h r_h, \qquad
\Delta x_h = N_h r_h\,\Delta t
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $[\mathrm{NO_3^-}]$ | Soil nitrate nitrogen stock (input donor and per-step increment output) | $\mathrm{g\,N\,m^{-2}}$ | `no3_nitrogen` | `state` |
| $C_{\mathrm{lab}}$ | Labile carbon substrate stock | $\mathrm{g\,C\,m^{-2}}$ | `labile_carbon` | `state` |
| $\mathrm{WFPS}, T, \mathrm{pH}$ | Water-filled pore space fraction, soil temperature, and soil pH | $\mathrm{1}$, $\mathrm{K}$, $\mathrm{1}$ | `water_filled_pore_space`, `soil_temperature`, `soil_ph` | `forcing` |
| $R_{\mathrm{exec}}$ | Realized nitrate reduction rate | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `denitrification_flux` | `flux` |
| $r_{\mathrm{N_2O}}$ | Shared additive $\mathrm{N_2O}$ emission rate contribution | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `n2o_flux` | `flux` |
| $r_{\mathrm{N_2}}$ | Dinitrogen gas production rate | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `n2_production` | `flux` |
| $f_{\mathrm{anaer}}, f_C, f_{\mathrm{N_2O}}, f_T, f_{\mathrm{pH}}$ | Anaerobic, carbon, $\mathrm{N_2O}$-fraction, temperature, and pH modifiers | $\mathrm{1}$ | `diagnostics.anaerobic_factor`, `diagnostics.carbon_factor`, `diagnostics.n2o_fraction`, `diagnostics.temperature_response`, `diagnostics.ph_factor` | `diagnostic` |
| $\Delta t$ | Executed step duration for public nitrogen ledger integration | $\mathrm{s}$ | `diagnostics.denitrification.step_seconds` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $V_{\max}, K_{m,\mathrm{NO_3}}, K_{m,C}$ | `pedosphere.denitrification.{vmax_denitrification,km_no3,carbon_half_saturation}` | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$, $\mathrm{g\,N\,m^{-2}}$, $\mathrm{g\,C\,m^{-2}}$ | `pedosphere/denitrification.yaml` | Parton et al. (1996); Li et al. (1992) |
| $\mathrm{WFPS}_{\mathrm{thr}}, w_{\mathrm{anaer}}, f_{\mathrm{anaer,\min}}$ | `pedosphere.physical.wfps_denitrification_threshold`, `pedosphere.denitrification.{anaerobic_transition_width,minimum_anaerobic_factor}` | $\mathrm{1}$ | `pedosphere/physical.yaml`, `pedosphere/denitrification.yaml` | Linn and Doran (1984); Del Grosso et al. (2000) |
| $Q_{10}, T_{\mathrm{ref}}, f_{T,\min}, f_{T,\max}$ | `pedosphere.physical.{q10_denitrification,reference_temperature}`, `pedosphere.denitrification.{min_rate_threshold,max_rate_factor}` | $\mathrm{1}$, $\mathrm{K}$, $\mathrm{1}$, $\mathrm{1}$ | `pedosphere/physical.yaml`, `pedosphere/denitrification.yaml` | Tiedje (1988) |
| $\mathrm{pH}_{\mathrm{opt}}, w_{\mathrm{pH}}, f_{\mathrm{pH,\min}}$ | `pedosphere.physical.{ph_optimum_denitrification,ph_response_width}`, `pedosphere.denitrification.ph_minimum_factor` | $\mathrm{1}$ | `pedosphere/physical.yaml`, `pedosphere/denitrification.yaml` | Knowles (1982) |
| $f_{\mathrm{base}}, f_{\max}, N_{\mathrm{thr}}, a_{\mathrm{NO_3}}$ | `pedosphere.denitrification.{base_n2o_fraction,max_n2o_fraction,n2o_no3_threshold,n2o_no3_enhancement}` | $\mathrm{1}$, $\mathrm{1}$, $\mathrm{g\,N\,m^{-2}}$, $\mathrm{1}$ | `pedosphere/denitrification.yaml` | Firestone and Davidson (1989); Parton et al. (1996) |
| $\mathrm{pH}_{\mathrm{ref}}, s_{\mathrm{pH}}, g_{\mathrm{pH,\min}}, g_{\mathrm{pH,\max}}$ | `pedosphere.denitrification.{n2o_ph_reference,n2o_ph_sensitivity,n2o_ph_effect_min,n2o_ph_effect_max}` | $\mathrm{1}$ | `pedosphere/denitrification.yaml` | Firestone and Davidson (1989) |
| $\mathrm{tol}_{\mathrm{cons}}, s_{\mathrm{strict}}$ | `validation.conservation.{conservation_tolerance,strict_conservation}` | $\mathrm{1}$, `bool` | `validation/conservation.yaml` | Process-level nitrogen ledger tolerance |
| $\Delta T_{10}$ | `relic.physical.constants.temperature.q10_interval` | $\mathrm{K}$ | `physical/constants.yaml` | $Q_{10}$ reference interval ($10\,\mathrm{K}$) |

## 5. Numerical Methods and Discretization

One governed `forward` invocation performs a single explicit Euler step $\Delta x_h = N_h r_h(x_h(t))\,\Delta t$ with environmental drivers held at step-start values, declaring `SubFlowAccuracy.ORDER_1` (Hairer, Norsett, and Wanner, 1993; Hairer, Lubich, and Wanner, 2006). First-order local truncation holds wherever the donor bound is inactive ($R_{\mathrm{denit}}\,\Delta t < [\mathrm{NO_3^-}]$). Non-finite or out-of-interval `water_filled_pore_space` ($\notin [0, 1]$) and non-positive or non-scalar $\Delta t$ raise `ValueError`.

State and flux outputs use distinct `UpdateItem` modes:

- `no3_nitrogen`: additive per-step stock increment (`UpdateMode.ACCUMULATE`, `per_timestep=True`, $\mathrm{g\,N\,m^{-2}}$).
- `n2o_flux`: shared additive gas-emission rate (`UpdateMode.ACCUMULATE`, `per_timestep=False`, $\mathrm{g\,N\,m^{-2}\,s^{-1}}$), summed with `NitrificationProcess` under `relic/contracts/variables/pedosphere/n2o_flux.yaml`.
- `denitrification_flux`, `n2_production`, and the six `diagnostics.*` outputs: `UpdateMode.REPLACE` (`per_timestep=False`).

## 6. Conservation, Boundary Conditions, and Ledgers

With conservation vector $C_h = [1, 1, 1]$, the stoichiometric operator satisfies $C_h N_h = [0, 0]$. `advance_reaction_network` enforces closed elemental nitrogen balance internally, and `DenitrificationProcess.check_conservation_laws()` audits the public proposal using `diagnostics.denitrification.step_seconds`:

$$
r_N = \Delta[\mathrm{NO_3^-}] + \bigl(r_{\mathrm{N_2O}} + r_{\mathrm{N_2}}\bigr)\,\Delta t = 0
$$

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `DenitrificationProcess` | `relic/scientific/pedosphere/nutrient_cycling/denitrification.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`TemporalScale.SECOND`, `SpatialScale.GRID_CELL`, `SubFlowAccuracy.ORDER_1`) |

## 8. Implementation Status

- `DenitrificationProcess`: `[I]`, IPHS **Level 2** (`reaction`, certificate `pedosphere_nutrient_cycling_denitrification_denitrification_process.yaml`). Donor-bounded two-channel nitrate reduction through `advance_reaction_network` with exact $C_h N_h = 0$ nitrogen closure.

Frozen certificate expressions:

```text
dNO3 = -R_den dt; dN2O = R_den f_N2O dt; dN2 = R_den(1 - f_N2O)dt.
x_dot = N_h r_h; N_h = [[-1,-1],[1,0],[0,1]]; r_h = [r_N2O, r_N2]^T.
```

## 9. References

- Bateman, E. J., and Baggs, E. M. (2005). Contributions of nitrification and denitrification to $\mathrm{N_2O}$ emissions from soils at different water-filled pore space. *Biology and Fertility of Soils*, 41, 379-388. DOI: `10.1007/s00374-005-0858-3`.
- Del Grosso, S. J., Parton, W. J., Mosier, A. R., et al. (2000). General model for $\mathrm{N_2O}$ and $\mathrm{N_2}$ gas emissions from soils due to denitrification. *Global Biogeochemical Cycles*, 14(4), 1045-1060. DOI: `10.1029/1999GB001225`.
- Firestone, M. K., and Davidson, E. A. (1989). Microbiological basis of $\mathrm{NO}$ and $\mathrm{N_2O}$ production and consumption in soil. In *Exchange of Trace Gases between Terrestrial Ecosystems and the Atmosphere* (pp. 7-21). Wiley.
- Hairer, E., Lubich, C., and Wanner, G. (2006). *Geometric Numerical Integration* (2nd ed.). Springer.
- Hairer, E., Norsett, S. P., and Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems* (2nd ed.). Springer.
- Knowles, R. (1982). Denitrification. *Microbiological Reviews*, 46(1), 43-70.
- Li, C., Frolking, S., and Frolking, T. A. (1992). A model of nitrous oxide evolution from soil driven by rainfall events: 1. Model structure and sensitivity. *Journal of Geophysical Research*, 97(D9), 9759-9776.
- Linn, D. M., and Doran, J. W. (1984). Effect of water-filled pore space on carbon dioxide and nitrous oxide production in tilled and nontilled soils. *Soil Science Society of America Journal*, 48(6), 1267-1272. DOI: `10.2136/sssaj1984.03615995004800060013x`.
- Parton, W. J., Mosier, A. R., Ojima, D. S., et al. (1996). Generalized model for $\mathrm{N_2}$ and $\mathrm{N_2O}$ production from nitrification and denitrification. *Global Biogeochemical Cycles*, 10(3), 401-412.
- Tiedje, J. M. (1988). Ecology of denitrification and dissimilatory nitrate reduction to ammonium. In *Biology of Anaerobic Microorganisms* (pp. 179-244). Wiley.
