# Stoichiometry Constraint

## 1. Process Definition and Scope

`StoichiometryConstraint` (`relic/scientific/pedosphere/nutrient_cycling/stoichiometry.py`) is a read-only post-processing observation operator that evaluates carbon-to-nitrogen (C:N) flux-ratio violations, relaxed diagnostic flux corrections, and elemental mass-balance errors across six soil organic and microbial pools (`metabolic_litter`, `structural_litter`, `fast_som`, `slow_som`, `passive_som`, and `microbial`).

The Functor writes only `diagnostics.*` keys ingested via `StateManager.ingest_module_diagnostics`; it commits no prognostic carbon or nitrogen state update and does not evaluate phosphorus stoichiometry (`[N]`). Upstream carbon fluxes (`diagnostics.carbon_fluxes`) come from whichever single decomposition owner (`first_order_decomposition` or `microbial_enzyme_decomposition`) is selected at graph compile time. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 23.

## 2. Governing Equations

### 2.1 Division-Free C:N Violation Indicator

For each pool $p$ whose step increments $\Delta C_p$ (`delta_<p>_c` in `diagnostics.carbon_fluxes`, $\mathrm{g\,C\,m^{-2}}$) and $\Delta N_p$ (`delta_<p>_n` in `diagnostics.nitrogen_fluxes`, $\mathrm{g\,N\,m^{-2}}$) are both present, with target C:N ratio $\mathrm{CN}_p$ and relative flexibility $\phi$ (Manzoni and Porporato, 2009; Cleveland and Liptzin, 2007):

$$
v_p =
\begin{cases}
1, & \lvert \Delta C_p - \mathrm{CN}_p\,\Delta N_p \rvert > \phi\,\mathrm{CN}_p\,\lvert \Delta N_p \rvert \\
0, & \text{otherwise}
\end{cases}
$$

When $\Delta N_p \ne 0$, this tests $\lvert \Delta C_p/\Delta N_p - \mathrm{CN}_p \rvert / \mathrm{CN}_p > \phi$ without division; when $\Delta N_p = 0$, $v_p = 1$ if and only if $\Delta C_p \ne 0$.

### 2.2 Relaxed Diagnostic Flux Adjustment

Where $v_p = 1$, relaxation strength $\kappa \in [0, 1]$ shifts the excess element toward target ratio $\mathrm{CN}_p$ (with nitrogen target $N_{\mathrm{target}} = \Delta C_p / \mathrm{CN}_p$):

$$
\Delta N_p' = \Delta N_p + \kappa\!\left(\frac{\Delta C_p}{\mathrm{CN}_p} - \Delta N_p\right) \quad \text{if } \lvert \Delta C_p \rvert > \lvert \Delta N_p \rvert\,\mathrm{CN}_p \;\text{(carbon excess)}
$$

$$
\Delta C_p' = \Delta C_p + \kappa\bigl(\Delta N_p\,\mathrm{CN}_p - \Delta C_p\bigr) \quad \text{if } \lvert \Delta C_p \rvert \le \lvert \Delta N_p \rvert\,\mathrm{CN}_p \;\text{(nitrogen excess)}
$$

The complementary element of the pair, and both elements when $v_p = 0$, remain unchanged.

### 2.3 Elemental Mass-Balance Residuals

When both boundary rate ports ($X_{\mathrm{in}}, X_{\mathrm{out}}$ in $\mathrm{g\,element\,m^{-2}\,s^{-1}}$) are supplied for element $X \in \{C, N\}$ over step $\Delta t$:

$$
e_{X,\mathrm{mass}} = \left\lvert \sum_p \Delta X_p' - (X_{\mathrm{in}} - X_{\mathrm{out}})\Delta t \right\rvert, \qquad
S_X = \left\lvert \sum_p \Delta X_p' \right\rvert + \lvert X_{\mathrm{in}}\Delta t \rvert + \lvert X_{\mathrm{out}}\Delta t \rvert
$$

$$
\rho_X = \begin{cases} \dfrac{e_{X,\mathrm{mass}}}{S_X}, & S_X > 0 \\ 0, & S_X = 0 \end{cases}, \qquad
e_{\mathrm{total}} = \rho_C + \rho_N
$$

where $\rho_X \in [0, 1]$ by the triangle inequality and $e_{\mathrm{total}}$ is emitted when both elements supply complete boundary pairs.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\Delta C_p$ | Per-pool carbon step changes (`delta_<p>_c` dictionary) | $\mathrm{g\,C\,m^{-2}}$ | `diagnostics.carbon_fluxes` | `diagnostic` |
| $\Delta N_p$ | Per-pool nitrogen step changes (`delta_<p>_n` dictionary) | $\mathrm{g\,N\,m^{-2}}$ | `diagnostics.nitrogen_fluxes` | `diagnostic` |
| $C_{\mathrm{in}}, C_{\mathrm{out}}$ | Optional aggregate carbon boundary input and output rates | $\mathrm{g\,C\,m^{-2}\,s^{-1}}$ | `diagnostics.carbon_inputs`, `diagnostics.carbon_outputs` | `diagnostic` |
| $N_{\mathrm{in}}, N_{\mathrm{out}}$ | Optional aggregate nitrogen boundary input and output rates | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `diagnostics.nitrogen_inputs`, `diagnostics.nitrogen_outputs` | `diagnostic` |
| $e_{C,\mathrm{mass}}$ | Corrected carbon mass-balance residual | $\mathrm{g\,C\,m^{-2}}$ | `diagnostics.stoichiometry_mass_balance_error_carbon` | `diagnostic` |
| $e_{N,\mathrm{mass}}$ | Corrected nitrogen mass-balance residual | $\mathrm{g\,N\,m^{-2}}$ | `diagnostics.stoichiometry_mass_balance_error_nitrogen` | `diagnostic` |
| $e_{\mathrm{total}}$ | Combined normalized carbon and nitrogen residual ($\rho_C + \rho_N$) | $\mathrm{1}$ | `diagnostics.stoichiometry_mass_balance_error_total` | `diagnostic` |
| $N_{\mathrm{viol}}$ | Cumulative count of `forward` calls on the instance with any $v_p = 1$ | $\mathrm{1}$ | `diagnostics.stoichiometry_constraint_violations_count` | `diagnostic` |
| $\Delta C_p', \Delta N_p', v_p$ | Dynamic per-pool corrected increments and violation indicator | $\mathrm{g\,m^{-2}}$, $\mathrm{1}$ | `diagnostics.stoichiometry_corrected_delta_<p>_c`, `diagnostics.stoichiometry_corrected_delta_<p>_n`, `diagnostics.stoichiometry_<p>_violation` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $\mathrm{CN}_{\mathrm{mic}}$ | `pedosphere.stoichiometry.microbial_biomass_cn` | $\mathrm{1}$ | `pedosphere/stoichiometry.yaml` | Cleveland and Liptzin (2007) |
| $\mathrm{CN}_{\mathrm{met}}, \mathrm{CN}_{\mathrm{str}}$ | `pedosphere.stoichiometry.metabolic_litter_cn`, `pedosphere.stoichiometry.structural_litter_cn` | $\mathrm{1}$ | `pedosphere/stoichiometry.yaml` | Manzoni and Porporato (2009) |
| $\mathrm{CN}_{\mathrm{fast}}, \mathrm{CN}_{\mathrm{slow}}, \mathrm{CN}_{\mathrm{pass}}$ | `pedosphere.stoichiometry.fast_som_cn`, `pedosphere.stoichiometry.slow_som_cn`, `pedosphere.stoichiometry.passive_som_cn` | $\mathrm{1}$ | `pedosphere/stoichiometry.yaml` | Manzoni and Porporato (2009) |
| $\phi$ | `pedosphere.numerical.stoichiometric_flexibility` | $\mathrm{1}$ | `pedosphere/numerical.yaml` | Manzoni and Porporato (2009) |
| $\kappa$ | `pedosphere.numerical.stoichiometric_correction_strength` | $\mathrm{1}$ | `pedosphere/numerical.yaml` | RELIC numerical diagnostic contract |

## 5. Numerical Methods and Discretization

Both violation testing and normalized mass-balance scaling avoid additive $\varepsilon$ floors: $v_p$ uses cross-multiplied absolute magnitudes, and $S_X = 0$ occurs only when every pool change and boundary term vanishes identically (producing $\rho_X = 0$). Pool key `'microbial'` maps directly to `pedosphere.stoichiometry.microbial_biomass_cn`. Instance counter `self._violations_count` increments once per `forward` call in which at least one evaluated pool satisfies $v_p = 1$.

## 6. Conservation, Boundary Conditions, and Ledgers

`StoichiometryConstraint` declares scheduler dependencies on upstream nitrogen processes so it executes in `ExecutionPhase.POST_PROCESSING` after reaction and transport commits. Because it emits only `diagnostics.*` entries, it does not mutate `StateManager` mass stocks; prognostic C:N conservation remains enforced inside the reaction networks of decomposition and mineralization Functors.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `StoichiometryConstraint` | `relic/scientific/pedosphere/nutrient_cycling/stoichiometry.py` | `ScaleCoupling` | `ScaleCouplingFunctor` | `ExecutionPhase.POST_PROCESSING` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `StoichiometryConstraint`: `[I]`, IPHS **Level 1** (`observation`, certificate `pedosphere_nutrient_cycling_stoichiometry_stoichiometry_constraint.yaml`). Pure diagnostic observation $y_h = O_h x_h$ of C:N consistency and elemental budget closure.
- Phosphorus C:N:P stoichiometric post-processing and prognostic pool projection: `[N]`.

Frozen audit and certificate expressions:

```text
N_target = C_flux / CN_target
e_C_mass = abs((C_inputs - C_outputs) - sum(Delta C_pools))
C/N balance with mass conservation
y_h = O_h x_h through the declared observation operator
```

## 9. References

- Cleveland, C. C., and Liptzin, D. (2007). C:N:P stoichiometry in soil: Is there a "Redfield ratio" for the microbial biomass? *Biogeochemistry*, 85, 235-252.
- Manzoni, S., and Porporato, A. (2009). Soil carbon and nitrogen mineralization: Theory and models across scales. *Soil Biology and Biochemistry*, 41(7), 1355-1379.
