# Soil Nitrification

## 1. Process Definition and Scope

`NitrificationProcess` (`relic/scientific/pedosphere/nutrient_cycling/nitrification.py`, `[I]`) models aerobic microbial oxidation of soil ammonium (`nh4_nitrogen`, $\mathrm{g\,N\,m^{-2}}$) into soil nitrate (`no3_nitrogen`, $\mathrm{g\,N\,m^{-2}}$) and a gaseous nitrous-oxide byproduct (`n2o_flux`, $\mathrm{g\,N\,m^{-2}\,s^{-1}}$) governed by Michaelis-Menten substrate kinetics and environmental modifiers for soil temperature, water-filled pore-space oxygen limitation (`water_filled_pore_space`), and soil pH (`soil_ph`).

All four inputs are mandatory runtime tensors; `mineralization` is the declared upstream scheduler dependency. Two-step nitrite (`NO2-`) intermediate kinetics, explicit dissolved-oxygen reaction coupling, and DAYCENT non-linear moisture modifiers remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 23 (Section 23.4).

## 2. Governing Equations

### 2.1 Potential Ammonium Oxidation Rate and Environmental Modifiers

For ammonium stock $[\mathrm{NH_4^+}]$ ($\mathrm{g\,N\,m^{-2}}$), soil temperature $T$ ($\mathrm{K}$), water-filled pore space $\mathrm{WFPS} \in [0, 1]$, and soil $\mathrm{pH}$:

$$
R_{\mathrm{nit}} = V_{\max}\,\frac{[\mathrm{NH_4^+}]}{K_{m,\mathrm{NH_4}} + [\mathrm{NH_4^+}]}\,f_T(T)\,f_{\mathrm{O_2}}(\mathrm{WFPS})\,f_{\mathrm{pH}}(\mathrm{pH})
$$

Temperature response combines a $Q_{10}$ curve (scaled by `unity_multiplier` $u_1 = 1$) with a piecewise extreme-temperature activity factor $f_{\mathrm{ext}}(T)$ outside $[T_{\min}, T_{\max}]$:

$$
f_{\mathrm{ext}}(T) =
\begin{cases}
u_1, & T_{\min} \le T \le T_{\max} \\
f_{\mathrm{ext,param}}, & T < T_{\min} \text{ or } T > T_{\max}
\end{cases}, \qquad
f_T(T) = \operatorname{clip}\!\left(u_1\,Q_{10}^{(T - T_{\mathrm{ref}})/\Delta T_{10}}\,f_{\mathrm{ext}}(T),\; f_{T,\min},\; f_{T,\max}\right)
$$

Aeration limitation $f_{\mathrm{O_2}}(\mathrm{WFPS})$ and the Gaussian pH response $f_{\mathrm{pH}}(\mathrm{pH})$ (carrying the standard $1/2$ Gaussian exponent factor) are:

$$
f_{\mathrm{O_2}}(\mathrm{WFPS}) =
\begin{cases}
1, & \mathrm{WFPS} \le \mathrm{WFPS}_{\mathrm{opt}} \\
\max\!\left(\dfrac{1 - \mathrm{WFPS}}{1 - \mathrm{WFPS}_{\mathrm{opt}}},\; f_{\mathrm{O_2,\min}}\right), & \mathrm{WFPS} > \mathrm{WFPS}_{\mathrm{opt}}
\end{cases}
$$

$$
f_{\mathrm{pH}}(\mathrm{pH}) = \operatorname{clip}\!\left(\exp\!\left[-\frac{1}{2}\left(\frac{\mathrm{pH} - \mathrm{pH}_{\mathrm{opt}}}{w_{\mathrm{pH}}}\right)^2\right],\; f_{\mathrm{pH,\min}},\; 1\right)
$$

### 2.2 Donor-Bounded Execution, Unmet-Demand Residual, and Stoichiometric Network

Over positive scalar step $\Delta t$ ($\mathrm{s}$), realized ammonium oxidation $R_{\mathrm{exec}}$ cannot exceed beginning-of-step donor availability, and any unmet kinetic demand is recorded as $R_{\mathrm{lim}}$ (`diagnostics.nitrification.substrate_limiter_residual`):

$$
R_{\mathrm{exec}} = \max\!\left(\min\!\left(R_{\mathrm{nit}},\; \frac{[\mathrm{NH_4^+}]}{\Delta t}\right),\; 0\right), \qquad
R_{\mathrm{lim}} = R_{\mathrm{nit}} - R_{\mathrm{exec}} \ge 0
$$

With constant byproduct fraction $f_{\mathrm{N_2O}}$ (`n2o_factor`), species vector $x_h = [\mathrm{NH_4^+},\; \mathrm{NO_3^-},\; \mathrm{N_2O}]^{\mathsf{T}}$, and reaction vector $r_h = [r_{\mathrm{NO_3}},\; r_{\mathrm{N_2O}}]^{\mathsf{T}}$:

$$
r_{\mathrm{NO_3}} = R_{\mathrm{exec}}\,(1 - f_{\mathrm{N_2O}}), \qquad
r_{\mathrm{N_2O}} = R_{\mathrm{exec}}\,f_{\mathrm{N_2O}}
$$

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
| $[\mathrm{NH_4^+}]$ | Soil ammonium nitrogen stock (input donor and per-step increment output) | $\mathrm{g\,N\,m^{-2}}$ | `nh4_nitrogen` | `state` |
| $[\mathrm{NO_3^-}]$ | Soil nitrate nitrogen per-step increment output | $\mathrm{g\,N\,m^{-2}}$ | `no3_nitrogen` | `state` |
| $T, \mathrm{WFPS}, \mathrm{pH}$ | Soil temperature, water-filled pore space fraction, and soil pH | $\mathrm{K}$, $\mathrm{1}$, $\mathrm{1}$ | `soil_temperature`, `water_filled_pore_space`, `soil_ph` | `forcing` |
| $R_{\mathrm{exec}}$ | Realized ammonium oxidation rate | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `nitrification_flux` | `flux` |
| $r_{\mathrm{N_2O}}$ | Shared additive $\mathrm{N_2O}$ byproduct emission rate contribution | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `n2o_flux` | `flux` |
| $f_T, f_{\mathrm{O_2}}, f_{\mathrm{pH}}$ | Temperature, WFPS oxygen, and pH response modifiers | $\mathrm{1}$ | `diagnostics.temperature_response`, `diagnostics.oxygen_factor`, `diagnostics.ph_factor` | `diagnostic` |
| $\Delta t$ | Executed step duration for public nitrogen ledger integration | $\mathrm{s}$ | `diagnostics.nitrification.step_seconds` | `diagnostic` |
| $R_{\mathrm{lim}}$ | Unmet kinetic ammonium oxidation demand when donor limit binds | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `diagnostics.nitrification.substrate_limiter_residual` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $V_{\max}, K_{m,\mathrm{NH_4}}, f_{\mathrm{N_2O}}$ | `pedosphere.nitrification.{vmax_nitrification,km_nh4,n2o_factor}` | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$, $\mathrm{g\,N\,m^{-2}}$, $\mathrm{1}$ | `pedosphere/nitrification.yaml` | Prosser (1989); Parton et al. (1996) |
| $\mathrm{WFPS}_{\mathrm{opt}}, f_{\mathrm{O_2,\min}}$ | `pedosphere.physical.wfps_nitrification_optimum`, `pedosphere.nitrification.minimum_oxygen_factor` | $\mathrm{1}$ | `pedosphere/physical.yaml`, `pedosphere/nitrification.yaml` | Linn and Doran (1984); Bateman and Baggs (2005) |
| $Q_{10}, T_{\mathrm{ref}}, T_{\min}, T_{\max}, f_{\mathrm{ext,param}}$ | `pedosphere.physical.{q10_nitrification,reference_temperature}`, `pedosphere.nitrification.{temperature_minimum,temperature_maximum,extreme_temperature_factor}` | $\mathrm{1}$, $\mathrm{K}$, $\mathrm{K}$, $\mathrm{K}$, $\mathrm{1}$ | `pedosphere/physical.yaml`, `pedosphere/nitrification.yaml` | Stark and Firestone (1995) |
| $f_{T,\min}, f_{T,\max}$ | `pedosphere.nitrification.{min_rate_threshold,max_rate_factor}` | $\mathrm{1}$ | `pedosphere/nitrification.yaml` | Dimensionless temperature-factor bounds |
| $\mathrm{pH}_{\mathrm{opt}}, w_{\mathrm{pH}}, f_{\mathrm{pH,\min}}$ | `pedosphere.physical.{ph_optimum_nitrification,ph_response_width}`, `pedosphere.nitrification.ph_minimum_factor` | $\mathrm{1}$ | `pedosphere/physical.yaml`, `pedosphere/nitrification.yaml` | Prosser (1989) |
| $u_1, \epsilon$ | `pedosphere.numerical.{unity_multiplier,epsilon}` | $\mathrm{1}$ | `pedosphere/numerical.yaml` | Rate normalization and statistical floor |
| $\mathrm{tol}_{\mathrm{cons}}, s_{\mathrm{strict}}$ | `validation.conservation.{conservation_tolerance,strict_conservation}` | $\mathrm{1}$, `bool` | `validation/conservation.yaml` | Process-level nitrogen ledger tolerance |
| $\Delta T_{10}$ | `relic.physical.constants.temperature.q10_interval` | $\mathrm{K}$ | `physical/constants.yaml` | $Q_{10}$ reference interval ($10\,\mathrm{K}$) |

## 5. Numerical Methods and Discretization

One governed `forward` invocation executes a single explicit Euler step $\Delta x_h = N_h r_h(x_h(t))\,\Delta t$ with environmental drivers held at step-start values, declaring `SubFlowAccuracy.ORDER_1` (Hairer, Norsett, and Wanner, 1993; Hairer, Lubich, and Wanner, 2006). First-order accuracy holds wherever $R_{\mathrm{nit}}\,\Delta t < [\mathrm{NH_4^+}]$. Entry validation rejects negative or non-finite `nh4_nitrogen`, `water_filled_pore_space` outside $[0, 1]$, and non-positive or non-scalar $\Delta t$ without silent clipping. Note that `reference_temperature` is converted to a Python `float` when calling `q10_temperature_response`.

Output `UpdateItem` semantics:

- `nh4_nitrogen` and `no3_nitrogen`: additive per-step stock increments (`UpdateMode.ACCUMULATE`, `per_timestep=True`, $\mathrm{g\,N\,m^{-2}}$).
- `n2o_flux`: shared additive gas-emission rate (`UpdateMode.ACCUMULATE`, `per_timestep=False`, $\mathrm{g\,N\,m^{-2}\,s^{-1}}$), accumulated alongside `DenitrificationProcess` under `relic/contracts/variables/pedosphere/n2o_flux.yaml`.
- `nitrification_flux` and all five `diagnostics.*` outputs: `UpdateMode.REPLACE` (`per_timestep=False`).

## 6. Conservation, Boundary Conditions, and Ledgers

With conservation vector $C_h = [1, 1, 1]$, the stoichiometric matrix satisfies $C_h N_h = [0, 0]$. `advance_reaction_network` verifies internal reaction closure, and `NitrificationProcess.check_conservation_laws()` audits the returned proposal using `diagnostics.nitrification.step_seconds`:

$$
r_N = \Delta[\mathrm{NH_4^+}] + \Delta[\mathrm{NO_3^-}] + r_{\mathrm{N_2O}}\,\Delta t = 0
$$

The diagnostic unmet-demand residual $R_{\mathrm{lim}}$ represents unrealized kinetic demand and enters no mass ledger.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `NitrificationProcess` | `relic/scientific/pedosphere/nutrient_cycling/nitrification.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`TemporalScale.SECOND`, `SpatialScale.GRID_CELL`, `SubFlowAccuracy.ORDER_1`) |

## 8. Implementation Status

- `NitrificationProcess`: `[I]`, IPHS **Level 2** (`reaction`, certificate `pedosphere_nutrient_cycling_nitrification_nitrification_process.yaml`). Donor-bounded two-branch ammonium oxidation through `advance_reaction_network` with exact $C_h N_h = 0$ nitrogen closure.

Frozen certificate expressions:

```text
dNH4 = -R_nit dt; dNO3 = R_nit(1 - f_N2O)dt; dN2O = R_nit f_N2O dt.
x_dot = N_h r_h; N_h = [[-1,-1],[1,0],[0,1]]; r_h = [r_NO3, r_N2O]^T.
```

## 9. References

- Bateman, E. J., and Baggs, E. M. (2005). Contributions of nitrification and denitrification to $\mathrm{N_2O}$ emissions from soils at different water-filled pore space. *Biology and Fertility of Soils*, 41, 379-388. DOI: `10.1007/s00374-005-0858-3`.
- Hairer, E., Lubich, C., and Wanner, G. (2006). *Geometric Numerical Integration* (2nd ed.). Springer.
- Hairer, E., Norsett, S. P., and Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems* (2nd ed.). Springer.
- Li, C., Frolking, S., and Frolking, T. A. (1992). A model of nitrous oxide evolution from soil driven by rainfall events: 1. Model structure and sensitivity. *Journal of Geophysical Research*, 97(D9), 9759-9776.
- Linn, D. M., and Doran, J. W. (1984). Effect of water-filled pore space on carbon dioxide and nitrous oxide production in tilled and nontilled soils. *Soil Science Society of America Journal*, 48(6), 1267-1272. DOI: `10.2136/sssaj1984.03615995004800060013x`.
- Oleson, K. W., Lawrence, D. M., Bonan, G. B., et al. (2013). *Technical Description of Version 4.5 of the Community Land Model (CLM)* (NCAR/TN-503+STR). NCAR.
- Parton, W. J., Mosier, A. R., Ojima, D. S., et al. (1996). Generalized model for $\mathrm{N_2}$ and $\mathrm{N_2O}$ production from nitrification and denitrification. *Global Biogeochemical Cycles*, 10(3), 401-412.
- Prosser, J. I. (1989). Autotrophic nitrification in bacteria. *Advances in Microbial Physiology*, 30, 125-181.
- Stark, J. M., and Firestone, M. K. (1995). Mechanisms for soil moisture effects on activity of nitrifying bacteria. *Applied and Environmental Microbiology*, 61(1), 218-221.
