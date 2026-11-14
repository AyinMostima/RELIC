# Soil Nitrogen Cycling Composition

## 1. Process Definition and Scope

`NitrogenCycling` (`relic/scientific/pedosphere/nutrient_cycling/nitrogen.py`) is an algebraic composition observer and external mineral-nitrogen port router (`[H]`). It does not re-execute or duplicate the reaction and transport laws owned by mineralization, nitrification, denitrification, nitrate leaching, or root nutrient uptake. Those child Functors run earlier in the `ExecutionPhase.MAIN_PROCESSES` dependency graph and submit their own pool increments directly to `StateManager`.

`NitrogenCycling` receives the already computed child proposals through the `subprocess_outputs` argument of `forward(state, time, dt, subprocess_outputs=...)` solely to project a read-only diagnostic mineral-nitrogen balance (`diagnostics.nitrogen_cycling.*`). Its sole prognostic state contribution is the integration of external mineral-nitrogen boundary inputs (`fertilizer_n_flux` and, when `grazing_management` is declared in the dependency graph, `urine_n_flux`) into `nh4_nitrogen` (`[H]`). No phosphorus pools, child process rates, or vertical layer profiles are read or emitted by this coordinator.

## 2. Governing Equations

Let $\Delta N_{\mathrm{NH_4},\mathrm{obs}}$ and $\Delta N_{\mathrm{NO_3},\mathrm{obs}}$ ($\mathrm{g\,N\,m^{-2}}$) denote the summed per-step mineral-nitrogen pool increments extracted from the enabled child mappings in `subprocess_outputs`. Over the current call interval $\Delta t > 0$ ($\mathrm{s}$), the coordinator forms the diagnostic rate vector

$$
r_h = \begin{bmatrix} \dfrac{\Delta N_{\mathrm{NH_4},\mathrm{obs}}}{\Delta t} \\[8pt] \dfrac{\Delta N_{\mathrm{NO_3},\mathrm{obs}}}{\Delta t} \end{bmatrix}, \qquad N_h = I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

and evaluates the read-only IPHS reaction-network projection (`open_system_route="diagnostic_residual"`):

$$
\frac{d d_h}{dt} = N_h r_h, \qquad \Delta d_h = \Delta t\,N_h r_h = \begin{bmatrix} \Delta N_{\mathrm{NH_4},\mathrm{obs}} \\ \Delta N_{\mathrm{NO_3},\mathrm{obs}} \end{bmatrix}.
$$

Because $\Delta t$ cancels identically in $\Delta t\,N_h (\Delta_{\mathrm{obs}} / \Delta t)$, the diagnostic outputs equal the observed child increments without interval rescaling:

$$
\Delta N_{\mathrm{NH_4},\mathrm{exp}} = \Delta N_{\mathrm{NH_4},\mathrm{min}} + \Delta N_{\mathrm{NH_4},\mathrm{nit}} + \Delta N_{\mathrm{NH_4},\mathrm{up}},
$$

$$
\Delta N_{\mathrm{NO_3},\mathrm{exp}} = \Delta N_{\mathrm{NO_3},\mathrm{nit}} + \Delta N_{\mathrm{NO_3},\mathrm{denit}} + \Delta N_{\mathrm{NO_3},\mathrm{leach}} + \Delta N_{\mathrm{NO_3},\mathrm{up}},
$$

$$
\Delta N_{\mathrm{min},\mathrm{exp}} = \Delta N_{\mathrm{NH_4},\mathrm{exp}} + \Delta N_{\mathrm{NO_3},\mathrm{exp}},
$$

where each signed term $\Delta N_{p,k}$ is the per-step `UpdateItem` value (`g N m-2`) produced by child process $k$ on pool $p$.

Independently of the read-only child audit, external fertilizer and grazing-urine nitrogen mass fluxes $F_{\mathrm{fert}} \ge 0$ and $F_{\mathrm{urine}} \ge 0$ ($\mathrm{kg\,N\,m^{-2}\,s^{-1}}$) are converted to areal gram increments via the physical-constants SSOT factor $c_{\mathrm{kg,g}} = 1000.0\,\mathrm{g\,kg^{-1}}$:

$$
\Delta N_{\mathrm{NH_4},\mathrm{fert}} = c_{\mathrm{kg,g}}\,F_{\mathrm{fert}}\,\Delta t, \qquad \Delta N_{\mathrm{NH_4},\mathrm{urine}} = c_{\mathrm{kg,g}}\,F_{\mathrm{urine}}\,\Delta t,
$$

$$
\Delta N_{\mathrm{NH_4},\mathrm{ext}} = \Delta N_{\mathrm{NH_4},\mathrm{fert}} + \Delta N_{\mathrm{NH_4},\mathrm{urine}}.
$$

The urine term is a reduced-order open boundary aggregation (`[H]`) that treats deposited urinary nitrogen as an immediate ammonium boundary addition; urea storage, enzymatic urease hydrolysis kinetics, ammonia volatilization, and patch-scale transport are not represented (`[N]`).

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| :--- | :--- | :--- | :--- | :--- |
| $F_{\mathrm{fert}}$ | External fertilizer nitrogen application rate | $\mathrm{kg\,N\,m^{-2}\,s^{-1}}$ | `fertilizer_n_flux` | `optional input` |
| $F_{\mathrm{urine}}$ | External grazing urine nitrogen deposition rate (valid iff `grazing_management` is a declared dependency) | $\mathrm{kg\,N\,m^{-2}\,s^{-1}}$ | `urine_n_flux` | `optional input` |
| $\Delta N_{\mathrm{NH_4},\mathrm{ext}}$ | External mineral-nitrogen increment routed to soil ammonium (emitted only when $F_{\mathrm{fert}}$ or $F_{\mathrm{urine}}$ is supplied) | $\mathrm{g\,N\,m^{-2}}$ | `nh4_nitrogen` | `output` (`state increment`) |
| $\Delta N_{\mathrm{NH_4},\mathrm{exp}}$ | Observed sum of enabled child increments on `nh4_nitrogen` | $\mathrm{g\,N\,m^{-2}}$ | `diagnostics.nitrogen_cycling.expected_delta_nh4_nitrogen` | `diagnostic` |
| $\Delta N_{\mathrm{NO_3},\mathrm{exp}}$ | Observed sum of enabled child increments on `no3_nitrogen` | $\mathrm{g\,N\,m^{-2}}$ | `diagnostics.nitrogen_cycling.expected_delta_no3_nitrogen` | `diagnostic` |
| $\Delta N_{\mathrm{min},\mathrm{exp}}$ | Sum $\Delta N_{\mathrm{NH_4},\mathrm{exp}} + \Delta N_{\mathrm{NO_3},\mathrm{exp}}$ | $\mathrm{g\,N\,m^{-2}}$ | `diagnostics.nitrogen_cycling.expected_delta_mineral_nitrogen` | `diagnostic` |
| $\Delta N_{\mathrm{NH_4},\mathrm{fert}}$ | Step-integrated fertilizer ammonium addition | $\mathrm{g\,N\,m^{-2}}$ | `diagnostics.nitrogen_cycling.external_fertilizer_n_delta` | `diagnostic` |
| $\Delta N_{\mathrm{NH_4},\mathrm{urine}}$ | Step-integrated grazing-urine ammonium addition | $\mathrm{g\,N\,m^{-2}}$ | `diagnostics.nitrogen_cycling.external_urine_n_delta` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
| :--- | :--- | :--- | :--- | :--- |
| $c_{\mathrm{kg,g}}$ | `relic.physical.constants.conversions.kg_to_g` | $\mathrm{g\,kg^{-1}}$ | `physical/constants.yaml` ($1000.0$) | BIPM SI Brochure; Galloway et al. (2004) |

`NitrogenCycling` introduces zero calibrated empirical parameters; all reaction and transport rate parameters belong exclusively to the five child Functors.

## 5. Numerical Methods and Discretization

1. **Dependency Injection and Input Validation**: At initialization, `NitrogenCycling` configures up to five enabled subprocess dependency names (`mineralization`, `nitrification`, `denitrification`, `nitrogen_leaching`, `root_nutrient_uptake`) via boolean constructor flags (all `True` by default) and merges them with any instance-level `config["dependencies"]` entries (such as `grazing_management`). Both `fertilizer_n_flux` and `urine_n_flux` are listed in `optional_input_variables`, so `validate_inputs(state)` imposes no unconditional soil-pool read requirements on `state`. Inside `forward`, every enabled subprocess key must appear as a dictionary in `subprocess_outputs`; missing entries or non-canonical legacy keys (`delta_nh4_nitrogen`, `delta_no3_nitrogen`) raise `KeyError`, and non-finite floating-point tensors raise `ValueError`.
2. **Dependency-Gated Urine Port**: `self._grazing_urine_selected` is `True` if and only if `"grazing_management"` is present in `get_dependencies()`. When selected, missing `urine_n_flux` raises `KeyError`; when unselected, providing `urine_n_flux` raises `ValueError`. Present external flux tensors must be non-empty, floating-point, finite, and non-negative.
3. **Splitting-Mode Interval Semantics**: `advance_reaction_network` evaluates $\Delta d_h = \Delta t\,I_2 (\Delta_{\mathrm{obs}} / \Delta t)$ with `open_system_route="diagnostic_residual"` and mutates no state. Under Lie splitting over $[t, t + h]$, each provider's most recent execution spans $[t, t + h]$ and $\Delta d_h$ reflects the full-step child increments. Under Strang splitting, when providers in a fast group have executed over the first half-step $[t, t + h/2]$, `subprocess_outputs` carries those half-step increments without rescaling; `StateManager` owns the authoritative whole-step committed ledger across all sub-steps.
4. **Conditional State Proposal**: All five `diagnostics.nitrogen_cycling.*` entries are returned on every call as raw `torch.Tensor` diagnostics. The governed state update `nh4_nitrogen` (`UpdateItem` with `mode=UpdateMode.ACCUMULATE`, `per_timestep=True`, `basis="per_step"`, `unit="g N m-2"`, `element="N"`) is emitted if and only if at least one external port (`fertilizer_n_flux` or `urine_n_flux`) is present in `state`.

## 6. Conservation, Boundary Conditions, and Ledgers

- **Single-Writer Separation**: Child mineral-nitrogen increments are committed once by their owning Functors (`mineralization`, `nitrification`, `root_nutrient_uptake` on `nh4_nitrogen`; `nitrification`, `denitrification`, `nitrogen_leaching`, `root_nutrient_uptake` on `no3_nitrogen`; additive `nitrification` and `denitrification` rates on `n2o_flux`). `NitrogenCycling` never re-emits child deltas.
- **External Mineral-N Port Ledger**: `check_conservation_laws(input_state, output_updates)` audits the coordinator's sole state-writing responsibility:

$$
r_{\mathrm{ext}} = \max \left\lvert \Delta N_{\mathrm{NH_4},\mathrm{actual}} - \left(\Delta N_{\mathrm{NH_4},\mathrm{fert}} + \Delta N_{\mathrm{NH_4},\mathrm{urine}}\right) \right\rvert,
$$

where $\Delta N_{\mathrm{NH_4},\mathrm{actual}}$ is `output_updates["nh4_nitrogen"].value` when `nh4_nitrogen` is emitted (verified for `UpdateMode.ACCUMULATE`, `per_timestep=True`, `unit="g N m-2"`, `element="N"`, and matching `source_functor`) and $0$ when neither external port is active. Exact closure gives $r_{\mathrm{ext}} = 0$ for `conservation_laws=["nitrogen"]`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| :--- | :--- | :--- | :--- | :--- |
| `NitrogenCycling` | `relic/scientific/pedosphere/nutrient_cycling/nitrogen.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

## 8. Implementation Status

- `NitrogenCycling` dependency-ledger diagnostic projection via `advance_reaction_network(..., open_system_route="diagnostic_residual")` and external fertilizer/grazing-urine ammonium port routing: `[H]` (certified IPHS Level 2 hybrid composition wrapper; certificate `pedosphere_nutrient_cycling_nitrogen_nitrogen_cycling.yaml`).
- Mechanistic soil urea state, urease hydrolysis kinetics, and ammonia volatilization from urine patches: `[N]`.

Certificate expressions (`pedosphere_nutrient_cycling_nitrogen_nitrogen_cycling.yaml`):

```text
legacy_expression: dNH4/dt = mineralization - nitrification - uptake_NH4; dNO3/dt = nitrification - denitrification - leaching - uptake_NO3
iphs_expression: diagnostic d_dot = N_h r_h with open_system_route=diagnostic_residual; fertilizer and dependency-selected urine inputs are routed once through the external NH4 port
```

## 9. References

1. Parton, W. J., Schimel, D. S., Cole, C. V., and Ojima, D. S. (1987). Analysis of factors controlling soil organic matter levels in Great Plains grasslands. *Soil Science Society of America Journal*, 51(5), 1173-1179.
2. Li, C., Frolking, S., and Frolking, T. A. (1992). A model of nitrous oxide evolution from soil driven by rainfall events: 1. Model structure and sensitivity. *Journal of Geophysical Research: Atmospheres*, 97(D9), 9759-9776.
3. Galloway, J. N., et al. (2004). Nitrogen cycles: past, present, and future. *Biogeochemistry*, 70(2), 153-226.
4. Davidson, E. A., Samanta, S., Caramori, S. S., and Savage, K. (2012). The Dual Arrhenius and Michaelis-Menten kinetics model for decomposition of soil organic matter at hourly to seasonal time scales. *Global Biogeochemical Cycles*, 26(1), GB1022.
5. Feinberg, M. (2019). *Foundations of Chemical Reaction Network Theory*. Springer.
