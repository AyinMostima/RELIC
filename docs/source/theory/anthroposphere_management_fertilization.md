# Agricultural Fertilization Boundary Transfer

## 1. Process Definition and Scope

`FertilizationManagement` (`relic/scientific/anthroposphere/management/fertilization.py`) maps recurring annual mineral fertilizer schedules (`FertilizerEvent`) into elemental nitrogen and phosphorus matter-boundary inflows entering the pedosphere (`fertilizer_n_flux`, `fertilizer_p_flux`, and paired `available_phosphorus` accumulation).

The Functor implements `TransportMechanism` (`TRANSPORT_TYPE="managed_application"`) without internal soil hydrolysis, nitrification, denitrification, volatilization, leaching, or empirical application-loss factors; downstream soil reactions belong to their pedosphere owners. Four mineral `FertilizerType` products (`UREA`, `AMMONIUM_NITRATE`, `AMMONIUM_SULFATE`, `SUPERPHOSPHATE`) are `[I]`, whereas `POTASH`, `NPK_COMPOUND`, `ORGANIC`, and `SLOW_RELEASE` fail closed with `ValueError` (`[N]`) pending canonical elemental pool routes. `FertilizationManagement` evaluates its configured `fertilizer_events` schedule directly and does not consume `fertilizer_product_flux` emitted by `ManagementCoordinator`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 24 (Section 24.5.2).

## 2. Governing Equations

### 2.1 Stoichiometric Elemental Mass Conversion

For scheduled event $e$ applying commercial product mass $A_e > 0$ ($\mathrm{kg\,product\,m^{-2}}$) with elemental nitrogen mass fraction $f_{\mathrm{N},e} \in [0, 1]$ and phosphate-oxide label mass fraction $f_{\mathrm{P_2O_5},e} \in [0, 1]$, elemental phosphorus fraction $f_{\mathrm{P},e} = f_{\mathrm{P_2O_5},e}\,\chi_{\mathrm{P}}$ is derived from standard atomic and molecular molar weights $M_{\mathrm{P}}$ and $M_{\mathrm{O_2}}$:

$$
\chi_{\mathrm{P}} = \frac{2 M_{\mathrm{P}}}{2 M_{\mathrm{P}} + \frac{5}{2} M_{\mathrm{O_2}}}, \qquad
M_{\mathrm{N},e} = A_e\,f_{\mathrm{N},e}, \qquad
M_{\mathrm{P},e} = A_e\,f_{\mathrm{P_2O_5},e}\,\chi_{\mathrm{P}}
$$

### 2.2 Exact Calendar-Window Overlap and Boundary Fluxes

Each event $e$ spans recurring annual interval $[a_{e,y}, a_{e,y} + \tau_e)$ with $a_{e,y} = y\,Y(C)\,s_{\mathrm{day}} + (d_e - 1)s_{\mathrm{day}}$ and declared duration $\tau_e \in (0, s_{\mathrm{day}}]$. Over simulation step $[t, t + \Delta t)$, total window overlap $\omega_e(t, \Delta t)$ is

$$
\omega_e(t, \Delta t) = \sum_{y} \max\!\bigl(0,\; \min(t + \Delta t,\; a_{e,y} + \tau_e) - \max(t,\; a_{e,y})\bigr)
$$

(reducing to intra-day phase overlap when explicit `day_of_year` equals $d_e$, and $0$ on non-matching explicit days). Summing active fractions $\omega_e / \tau_e$ across events with mass-unit conversion $k_{\mathrm{g}} = 1000\,\mathrm{g\,kg^{-1}}$ gives step elemental increments and mean boundary rates:

$$
\Delta M_{\mathrm{N}} = \sum_{e} M_{\mathrm{N},e}\frac{\omega_e}{\tau_e}, \qquad
\Delta M_{\mathrm{P}} = k_{\mathrm{g}}\sum_{e} M_{\mathrm{P},e}\frac{\omega_e}{\tau_e}
$$

$$
F_{\mathrm{fert,N}} = \frac{\Delta M_{\mathrm{N}}}{\Delta t}, \qquad
F_{\mathrm{fert,P}} = \frac{\Delta M_{\mathrm{P}}}{\Delta t}, \qquad
\Delta P_{\mathrm{available}} = \Delta M_{\mathrm{P}} = F_{\mathrm{fert,P}}\,\Delta t
$$

Subdividing $[t, t + \Delta t)$ preserves $\sum \omega_e = \tau_e$ across any step-size sequence, preventing double application or sub-daily under-integration.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $d$ | Optional explicit simulator day-of-year coordinate | $\mathrm{d}$ | `day_of_year` | `forcing` |
| $F_{\mathrm{fert,N}}$ | Elemental nitrogen boundary mass flux to pedosphere | $\mathrm{kg\,N\,m^{-2}\,s^{-1}}$ | `fertilizer_n_flux` | `flux` |
| $F_{\mathrm{fert,P}}$ | Elemental phosphorus boundary mass flux to pedosphere | $\mathrm{g\,P\,m^{-2}\,s^{-1}}$ | `fertilizer_p_flux` | `flux` |
| $\Delta P_{\mathrm{available}}$ | Integrated labile soil phosphorus step increment | $\mathrm{g\,P\,m^{-2}}$ | `available_phosphorus` | `state` |
| $r_{\mathrm{N}}, r_{\mathrm{P}}$ | Local nitrogen and phosphorus step-integration residuals | $\mathrm{kg\,N\,m^{-2}}$, $\mathrm{g\,P\,m^{-2}}$ | `diagnostics.fertilization.nitrogen_boundary_residual`, `diagnostics.fertilization.phosphorus_boundary_residual` | `diagnostic` |
| $c_{\mathrm{iphs}}$ | Canonical IPHS boundary-port runtime indicator | $\mathrm{1}$ | `diagnostics.fertilization.canonical_iphs_runtime` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $A_{\mathrm{urea}}, A_{\mathrm{an}}, A_{\mathrm{ssp}}$ | `anthroposphere.management.fertilization.spring_urea_product_mass`, `anthroposphere.management.fertilization.side_dress_ammonium_nitrate_product_mass`, `anthroposphere.management.fertilization.spring_superphosphate_product_mass` | $\mathrm{kg\,m^{-2}}$ | `anthroposphere/management.yaml` | FAO (2000) |
| $d_{\mathrm{spr}}, d_{\mathrm{sd}}, d_{\mathrm{P}}$ | `anthroposphere.management.fertilization.spring_day`, `anthroposphere.management.fertilization.side_dress_day`, `anthroposphere.management.fertilization.phosphorus_day` | $\mathrm{d}$ | `anthroposphere/management.yaml` | Default annual schedule days |
| $f_{\mathrm{N}}$ | `anthroposphere.management.fertilization.urea.n_content`, `anthroposphere.management.fertilization.ammonium_nitrate.n_content`, `anthroposphere.management.fertilization.ammonium_sulfate.n_content` | $\mathrm{1}$ | `anthroposphere/management.yaml` | FAO (2000) mineral N grades |
| $f_{\mathrm{P_2O_5}}$ | `anthroposphere.management.fertilization.superphosphate.p2o5_mass_fraction` | $\mathrm{1}$ | `anthroposphere/management.yaml` | FAO (2000) superphosphate grade |
| $M_{\mathrm{P}}, M_{\mathrm{O_2}}$ | `relic.physical.constants.molecular_weights.phosphorus`, `relic.physical.constants.molecular_weights.o2` | $\mathrm{g\,mol^{-1}}$ | `physical/constants.yaml` | IUPAC standard atomic weights |
| $s_{\mathrm{day}}, k_{\mathrm{g}}$ | `relic.physical.constants.time.seconds_per_day`, `relic.physical.constants.conversions.kg_to_g` | $\mathrm{s\,d^{-1}}$, $\mathrm{g\,kg^{-1}}$ | `physical/constants.yaml` | BIPM (2019) SI constants |

## 5. Numerical Methods and Discretization

`FertilizationManagement.forward()` validates calendar bounds through `CalendarHandler` and emits `fertilizer_n_flux` and `fertilizer_p_flux` via `UpdateItem(UpdateMode.REPLACE, per_timestep=False)` alongside `available_phosphorus` via `UpdateItem(UpdateMode.ACCUMULATE, per_timestep=True, element="P")`, plus three plain diagnostic tensors. Inactive timesteps return explicit zero-valued `UpdateItem` updates with identical units and modes.

## 6. Conservation, Boundary Conditions, and Ledgers

`FertilizationManagement` exposes typed open matter input ports (`fertilization_nitrogen_input_port`, `fertilization_phosphorus_input_port`) with empty internal skew-symmetric and dissipative operators ($J = 0$, $M = 0$). Local boundary closure satisfies $r_{\mathrm{N}} = \Delta M_{\mathrm{N}} - F_{\mathrm{fert,N}}\Delta t = 0$ and $r_{\mathrm{P}} = \Delta M_{\mathrm{P}} - F_{\mathrm{fert,P}}\Delta t = 0$, verified by `check_conservation_laws()`. Downstream mineral nitrogen accumulation is owned by pedosphere nitrogen cycling, while `fertilizer_p_flux` and `available_phosphorus` enter the `phosphorus.total` open-system ledger together.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `FertilizationManagement` | `relic/scientific/anthroposphere/management/fertilization.py` | `TransportMechanism` (`TRANSPORT_TYPE="managed_application"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `FertilizationManagement`: `[I]`, IPHS **Level 2** (`port`, certificate `anthroposphere_management_fertilization_fertilization_management.yaml`). Partition-invariant calendar-window mineral N and P boundary transfer; `POTASH`, `NPK_COMPOUND`, `ORGANIC`, and `SLOW_RELEASE` product types remain `[N]`.

Frozen certificate expressions:

```text
Retired process-law branches advanced fertilizer pools and atmospheric losses inside the management module.
Delta M_q = sum_e A_e f_q,e omega_e / tau_e; F_q = Delta M_q / dt; Delta P_available = F_P dt.
```

## 9. References

- BIPM (2019). *The International System of Units (SI)* (9th ed.). Bureau International des Poids et Mesures.
- FAO (2000). *Fertilizers and Their Use* (4th ed.). Food and Agriculture Organization of the United Nations.
- Lawrence, D. M., Fisher, R. A., Koven, C. D., et al. (2019). The Community Land Model version 5: description of new features, benchmarking, and impact of forcing uncertainty. *Journal of Advances in Modeling Earth Systems*, 11(12), 4245-4287.
