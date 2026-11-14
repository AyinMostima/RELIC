# [H] Multi-Isotope Pool Consistency and Rayleigh Fractionation

## 1. Process Definition and Scope

This document records the historical (`[H]`) specification for cross-pool multi-nuclide mass consistency, stoichiometric ratio bounds, closed-system Rayleigh distillation, and read-only $\delta$-notation diagnostics. The former private pool-tracking runtime and its private pool consistency manager under `relic/scientific/isotopes/` were retired (`[X]`) because they duplicated `StateManager` state ownership outside `ValidationRegistry` and had no governed production caller.

All equations below are retained strictly as `SPEC` candidate equations for future `StateManager`-governed nuclide component pools; they are neither `[I]` runtime behavior nor active conservation proof. Related stateless fractionation calculators are documented in `isotopes.md`, and theoretical context appears in *Methods of Ecological Analysis*, Chapter 23 (Section 23.12).

## 2. Governing Equations

### 2.1 Cross-Pool Element Mass-Balance Residual (`SPEC`)

For total element mass $M_{\mathrm{old}} = \sum_p (m_{L,p}^{\mathrm{old}} + m_{H,p}^{\mathrm{old}})$ prior to an internal transfer step and $M_{\mathrm{new}} = \sum_p (m_{L,p}^{\mathrm{new}} + m_{H,p}^{\mathrm{new}})$ after the step, the regularized relative mass imbalance $\epsilon_{\mathrm{mass}}$ and acceptance constraint against tolerance $\tau_{\mathrm{cons}}$ are:

$$
\epsilon_{\mathrm{mass}} = \frac{\lvert M_{\mathrm{new}} - M_{\mathrm{old}}\rvert}{\lvert M_{\mathrm{old}}\rvert + \varepsilon}, \qquad
\epsilon_{\mathrm{mass}} \le \tau_{\mathrm{cons}}
$$

### 2.2 Stoichiometric Ratio-Range Check (`SPEC`)

For paired elemental pool masses $A$ and $B$ forming stoichiometric ratio $R_{\mathrm{elem}} = A / B$ with expected physiological interval $[R_{\mathrm{expected,min}},\; R_{\mathrm{expected,max}}]$ and relative slack factor $\tau_{\mathrm{stoich}}$, the admissible interval is:

$$
\mathrm{tol} = \bigl(R_{\mathrm{expected,max}} - R_{\mathrm{expected,min}}\bigr)\,\tau_{\mathrm{stoich}}, \qquad
R_{\mathrm{expected,min}} - \mathrm{tol} \le R_{\mathrm{elem}} \le R_{\mathrm{expected,max}} + \mathrm{tol}
$$

### 2.3 Closed-System Rayleigh Distillation and Delta Observation (`SPEC`)

In a closed homogeneous reactant pool whose remaining unreacted fraction is $f = M(t) / M(0) \in (0, 1]$ under constant instantaneous fractionation factor $\alpha$, the heavy-to-light nuclide ratio $R(t)$ evolves via Rayleigh (1896) and Mariotti et al. (1981):

$$
\frac{R(t)}{R(0)} = f^{\alpha - 1}, \qquad
\ln\!\left(\frac{1000 + \delta(t)}{1000 + \delta(0)}\right) = (\alpha - 1)\ln f, \qquad
\delta(t) \approx \delta(0) + \varepsilon\ln f
$$

where $\varepsilon = (\alpha - 1)\times 1000$ and the read-only $\delta$-notation observation operator relative to international standard ratio $R_{\mathrm{standard}}$ (Coplen, 2011; Werner and Brand, 2001) is:

$$
\delta = \left(\frac{R_{\mathrm{sample}}}{R_{\mathrm{standard}}} - 1.0\right)\times 1000.0
$$

Audit-equivalent ASCII specification markers:

```text
epsilon_mass = abs(M_new - M_old) / (abs(M_old) + epsilon)
delta = (R_sample / R_standard - 1.0) * 1000.0
```

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $M_{\mathrm{old}}, M_{\mathrm{new}}$ | Pre-step and post-step summed binary element pool masses | $\mathrm{kg\,m^{-2}}$ or $\mathrm{mol\,m^{-2}}$ | `element_masses[element]` (`SPEC`) | `state` |
| $\epsilon_{\mathrm{mass}}$ | Relative cross-pool element mass-balance residual | $\mathrm{1}$ | `epsilon_mass` (`SPEC`) | `diagnostic` |
| $R_{\mathrm{elem}}$ | Cross-element stoichiometric ratio $A / B$ | $\mathrm{1}$ | `stoichiometric_ratio` (`SPEC`) | `diagnostic` |
| $f$ | Remaining reactant fraction in closed Rayleigh distillation | $\mathrm{1}$ | `remaining_fraction` (`SPEC`) | `diagnostic` |
| $R_{\mathrm{sample}}$ | Pool heavy-to-light nuclide ratio $n_H / n_L$ | $\mathrm{1}$ | `isotope_ratios[species]` (`SPEC`) | `diagnostic` |
| $\delta$ | Relative isotope ratio difference from standard in permil | $\text{permil}$ ($\mathrm{10^{-3}}$) | `delta_values[species]` (`SPEC`) | `observation` |

## 4. Parameter Table

| Symbol | ParameterManager Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\tau_{\mathrm{cons}}$ | `isotopes.validation.numerical.conservation_tolerance` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` ($10^{-6}$) | Cross-pool mass-balance tolerance |
| $\tau_{\mathrm{stoich}}$ | `isotopes.validation.numerical.stoichiometry_tolerance` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` ($0.2$) | Stoichiometric interval slack factor |
| $\varepsilon$ | `isotopes.validation.numerical.epsilon` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` ($10^{-12}$) | Denominator regularization constant |
| $R_{\mathrm{standard}}$ | `isotopes.standards.carbon_13.pdb_ratio`, `nitrogen_15.air_ratio`, `oxygen_18.vsmow_ratio` | $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` ($0.0112372$, $0.0036765$, $0.0020052$) | Coplen et al. (2006); Mariotti (1983); Baertschi (1976) |
| $m_{\min}, R_{\max}$ | `isotopes.pool.min_total`, `isotopes.pool.max_ratio` | pool units, $\mathrm{1}$ | `isotopes/isotope_parameters.yaml` | Pool positivity and ratio sanity bounds |

## 5. Numerical Methods and Discretization

No numerical solver or time integrator executes for multi-isotope pool tracking in the live codebase. Parameters under `isotopes.multi.*` (`integration_timestep`, `convergence_tolerance`, `max_iterations`, `numerical_solver`) remain inert database entries from the retired prototype. Any future implementation must integrate prognostic nuclide component tensors at the parent biogeochemical Functor step $\Delta t$ through standard `UpdateItem` transactions without private sub-ledgers or post-hoc mass rescaling.

## 6. Conservation, Boundary Conditions, and Ledgers

Closed internal transfers in a future governed implementation must satisfy exact binary sum conservation $\Delta m_{L,\mathrm{src}} + \Delta m_{L,\mathrm{dst}} = 0$ and $\Delta m_{H,\mathrm{src}} + \Delta m_{H,\mathrm{dst}} = 0$ prior to commit, audited through `ValidationRegistry` and `StateManager` transactions. Open boundary fluxes (atmospheric $\mathrm{CO_2}$, $\mathrm{N_2}$, precipitation, and water vapor) must enter through explicit boundary ports with ParameterManager-backed isotopic signatures. No private `validation_history` list or empty-manager pass-through exists in production code.

## 7. Code Mapping

| Component | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| Retired private multi-isotope pool runtime (`[X]` deleted) | None (retired from `relic/scientific/isotopes/`) | None (`[H]` historical specification) | None | Removed (`[X]`) |
| `IsotopeHeartbeatFunctor` (`[N]` fail-closed) | `relic/scientific/isotopes/functor.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | Unbound (`[N]`, raises `IsotopeHeartbeatError`) |
| `IsotopeFractionationCalculator` (`[I]` stateless) | `relic/scientific/isotopes/fractionation/base.py` | Stateless fractionation base (`abc.ABC`) | None | Caller-invoked (`[I]`) |

## 8. Implementation Status

- Retired private multi-isotope pool tracker and pool consistency manager: `[X]` deleted; no IPHS certificate exists for this historical (`[H]`) specification.
- `IsotopeHeartbeatFunctor`: `[N]`, IPHS **Level 0** (`runtime_unbound`, certificate `isotopes_functor_isotope_heartbeat_functor.yaml`); submits no nuclide `UpdateItem` entries.
- Generic `isotope_pool_mass` and `fractionation_factor` `StateManager` variables: `[N]` unregistered; any future coupled nuclide ledger must bind explicit nuclide component states (`12C`, `13C`, `14N`, `15N`, `16O`, `18O`) directly in `StateManager`.

## 9. References

- Baertschi, P. (1976). Absolute $^{18}\mathrm{O}$ content of Standard Mean Ocean Water. *Earth and Planetary Science Letters*, 31(3), 341-344.
- Coplen, T. B., Brand, W. A., Gehre, M., et al. (2006). New guidelines for $\delta^{13}\mathrm{C}$ measurements. *Analytical Chemistry*, 78(7), 2439-2441.
- Coplen, T. B. (2011). Guidelines and recommended terms for expression of stable-isotope-ratio and gas-ratio measurement results. *Pure and Applied Chemistry*, 83(10), 1873-1935.
- Mariotti, A., Germon, J. C., Hubert, P., et al. (1981). Experimental determination of nitrogen kinetic isotope fractionation: some principles; illustration for the denitrification and nitrification processes. *Plant and Soil*, 62(3), 413-430.
- Mariotti, A. (1983). Atmospheric nitrogen is a reliable standard for natural $^{15}\mathrm{N}$ abundance measurements. *Nature*, 303, 685-687.
- Rayleigh, J. W. S. (1896). Theoretical considerations respecting the separation of gases by diffusion and similar processes. *Philosophical Magazine*, 42(258), 493-498.
- Werner, R. A., and Brand, W. A. (2001). Referencing strategies and techniques in stable isotope ratio analysis. *Rapid Communications in Mass Spectrometry*, 15(7), 501-519.
