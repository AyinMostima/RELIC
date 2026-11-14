# Observation State Feedback: Governed Scalar Kalman Analysis

## 1. Process Definition and Scope

`ObservationStateFeedbackFunctor` (`relic/coupling/observation/state_feedback.py`) implements the single governed state-feedback path from a direct observation into a prognostic RELIC state variable. Whereas the forward observation operators documented in [observation_operators.md](observation_operators.md) emit strictly read-only `diagnostics.observation.*` outputs, `ObservationStateFeedbackFunctor` closes the analysis loop by computing a scalar optimal-interpolation / Kalman update with the identity observation operator (`IdentityOperator` from `relic/optimization/assimilation/base.py`), gating the normalized innovation against the configured error variances, and returning a governed `UpdateMode.REPLACE` (`per_timestep=True`) proposal committed solely by `StateManager`. Theoretical foundations for linear innovation filtering and bounded observation-feedback ledgers appear in *Methods of Ecological Analysis*, Chapters 11 and 26.

The Functor fulfills the **`ScaleCoupling`** abstraction through `ScaleCouplingFunctor` (`ProcessCategory.SCALE_COUPLING`, `ExecutionPhase.COUPLING`, `SpatialScale.GRID_CELL`, `VariableKind.STATE`, `source_scale="observation_support"`, `target_scale="model_state_support"`, `coupling_type="stochastic"`). Because observation and background state share identical spatial support, `upscale` and `downscale` act as identity maps. To write a prognostic variable, `ObservationStateFeedbackFunctor` must be explicitly authorized as a tendency writer in that variable's State Contract Management (SCM) specification (for example, `relic/contracts/variables/hydrosphere/soil_moisture.yaml` authorizes it for `soil_moisture`).

## 2. Governing Equations

### 2.1 Identity Observation Operator and Scalar Kalman Gain

For a directly observed state tensor $x$ (background forecast) and co-located observation tensor $y$ evaluated through $H = I$ (`IdentityOperator([state_variable])`), the innovation $d$ and scalar Kalman gain $K$ are (Kalman, 1960; Kalnay, 2003, Eqs. 5.3.11-5.3.13):

$$
d = y - x, \qquad K = \frac{B}{B + R},
$$

where $B > 0$ is the scalar background error variance and $R > 0$ is the scalar observation error variance, ensuring $0 < K < 1$. By sign convention, $d > 0$ indicates that the observation exceeds the model background so the analysis shifts upward toward $y$ by fraction $K$.

### 2.2 Normalized Innovation Gate and Analysis Update

Before emitting any state update, elementwise normalized innovation magnitudes are evaluated against the dimensionless gate threshold $\nu_{\max} > 0$:

$$
\nu = \frac{\lvert d \rvert}{\sqrt{B + R}} \le \nu_{\max}.
$$

When $\nu \le \nu_{\max}$ holds across every element of the support, the posterior analysis state $x^a$, analysis increment $\Delta x^a$, and posterior analysis error variance $P^a$ are:

$$
x^a = x + K d, \qquad \Delta x^a = x^a - x = K d, \qquad P^a = (1 - K) B = \frac{B R}{B + R}.
$$

For example, with $B = 0.01$, $R = 0.04$, $x = 0.30\,\mathrm{m^3\,m^{-3}}$, and $y = 0.35\,\mathrm{m^3\,m^{-3}}$, the equations yield $K = 0.2$, $d = 0.05\,\mathrm{m^3\,m^{-3}}$, $\nu \approx 0.2236 \le 4.0$, $x^a = 0.31\,\mathrm{m^3\,m^{-3}}$, $\Delta x^a = 0.01\,\mathrm{m^3\,m^{-3}}$, and $P^a = 0.008\,\mathrm{m^6\,m^{-6}}$. Because `analysis()` uses pure PyTorch tensor arithmetic without detached branches, exact reverse-mode parameter sensitivities (such as $\partial x^a / \partial R = -B d / (B + R)^2$) propagate through `x^a`.

## 3. Variable Table

| Symbol | Meaning | Units | Code / State Key | Role |
| --- | --- | --- | --- | --- |
| $x$ | Model background state tensor | `[state]` (e.g., $\mathrm{m^3\,m^{-3}}$) | `<config["state_variable"]>` | Prognostic input & governed `REPLACE` output (`per_timestep=True`) |
| $y$ | Direct observation forcing tensor | `[state]` | `<config["observation_variable"]>` | External forcing input |
| $d$ | Innovation $y - x$ | `[state]` | `diagnostics.observation.state_feedback.innovation` | Detached `REPLACE` diagnostic |
| $\nu$ | Normalized innovation $\lvert d \rvert / \sqrt{B + R}$ | $1$ | `diagnostics.observation.state_feedback.normalized_innovation` | Detached `REPLACE` diagnostic |
| $K$ | Scalar Kalman gain $B / (B + R)$ | $1$ | `diagnostics.observation.state_feedback.kalman_gain` | Detached `REPLACE` diagnostic |
| $\Delta x^a$ | Analysis increment $x^a - x = K d$ | `[state]` | `diagnostics.observation.state_feedback.analysis_increment` | Detached `REPLACE` diagnostic |
| $P^a$ | Posterior analysis error variance $(1 - K) B$ | `[state]^2` | `diagnostics.observation.state_feedback.analysis_error_variance` | Detached `REPLACE` diagnostic |

## 4. Parameter Table

All error-model parameters reside in `relic/parameters/database/relic/coupling/observation.yaml` and are retrieved at runtime via `self.get_parameter_value(..., required=True)`.

| Symbol | `ParameterManager` Key | Schema / Physical Units | Default (Bounds) | Reference |
| --- | --- | --- | --- | --- |
| $B$ | `coupling.observation.state_feedback.background_error_variance` | `dimensionless` (`[state]^2`) | `0.01` ($[10^{-12}, 10^6]$) | Kalman (1960); Kalnay (2003) |
| $R$ | `coupling.observation.state_feedback.observation_error_variance` | `dimensionless` (`[state]^2`) | `0.04` ($[10^{-12}, 10^6]$) | Kalman (1960); Kalnay (2003) |
| $\nu_{\max}$ | `coupling.observation.state_feedback.max_normalized_innovation` | `dimensionless` ($1$) | `4.0` ($[0.5, 100.0]$) | Kalnay (2003), Ch. 5 quality control |

Constructor `config` keys:
- `state_variable` (required non-empty string): target prognostic state key.
- `observation_variable` (required non-empty string, distinct from `state_variable`): state key holding the external observation tensor.
- `units` (optional string): explicit physical unit string attached to the returned `UpdateItem.unit` for `StateManager` unit verification.

## 5. Numerical Methods and Discretization

`ObservationStateFeedbackFunctor.forward` executes statelessly at each coupled step:

1. **Input and support validation**: `_require_tensor` verifies that both `state_variable` and `observation_variable` exist in `state`, are `torch.Tensor` instances, and contain strictly finite entries. Shape compatibility requires `y.shape == x.shape` or `y.numel() == x.numel()` (reshaping $y$ onto `x.shape` and casting to `x.dtype` and `x.device`); mismatched element counts raise `ValueError`.
2. **Variance positivity**: $B$ and $R$ are converted to tensors on `x.dtype`, and non-positive values ($B \le 0$ or $R \le 0$) raise `ValueError`.
3. **Whole-support innovation gate**: If any element of $\nu$ is non-finite or satisfies $\nu > \nu_{\max}$, `forward` raises `ValueError` before constructing any `UpdateItem`. No partial spatial masking or elementwise clamping of rejected innovations occurs.
4. **Update packaging**: The analysis tensor $x^a$ is wrapped with `self.wrap_update(result["analysis"], mode=UpdateMode.REPLACE, per_timestep=True)` (preserving the PyTorch autograd graph), while the five `diagnostics.observation.state_feedback.*` items are wrapped with `.detach()` under `UpdateMode.REPLACE`. Because the Functor stores no internal temporal buffers, checkpoint/restart reproducibility follows directly from the committed `StateManager` state.

## 6. Conservation, Boundary Conditions, and Ledgers

Data assimilation injects external observational information into the model state and is not an internally closed mass- or energy-conserving transformation (`FunctorMetadata.conservation_laws = []`). Consequently:

- The non-zero storage jump $\Delta x^a = x^a - x = K d$ is explicitly disclosed on `diagnostics.observation.state_feedback.analysis_increment` and attributed to `ObservationStateFeedbackFunctor` in `StateManager` provenance records.
- Physical bounds enforcement is never performed inside `analysis()`. When the target variable declares physical domain bounds $[l_v, u_v]$ in its SCM contract, `StateManager.apply_updates` applies any required bounds clamp at commit time and records the signed discrepancy $x^a_{\mathrm{req}} - \operatorname{clamp}(x^a_{\mathrm{req}}, l_v, u_v)$ in `StateManager.get_diagnostics()["clip_loss"]` (Invariant `I7`).

## 7. Code Mapping

| Class / Resource | File Path | Architectural Role |
| --- | --- | --- |
| `ObservationStateFeedbackFunctor` | `relic/coupling/observation/state_feedback.py` | Governed `ScaleCouplingFunctor` (`ExecutionPhase.COUPLING`) |
| `IdentityOperator` | `relic/optimization/assimilation/base.py` | Direct state-to-observation identity operator $H = I$ |
| `soil_moisture.yaml` | `relic/contracts/variables/hydrosphere/soil_moisture.yaml` | SCM variable contract listing `ObservationStateFeedbackFunctor` as tendency writer |
| `observation.yaml` | `relic/parameters/database/relic/coupling/observation.yaml` | Parameter database owner for `B`, `R`, and $\nu_{\max}$ |

## 8. Implementation Status

- **Implemented (`[I]`)**: Stateless scalar Kalman / optimal-interpolation analysis on co-located tensor supports, `IdentityOperator` evaluation, whole-field normalized-innovation gating, governed `UpdateMode.REPLACE` (`per_timestep=True`) commit integration with `StateManager` provenance, five detached feedback diagnostics, and reverse-mode autograd differentiability with respect to $x$, $y$, $B$, and $R$.
- **Not implemented (`[N]`)**: Spatially correlated background/observation error covariance matrices ($\mathbf{B}, \mathbf{R} \in \mathbb{R}^{n \times n}$), flow-dependent ensemble covariance updates (EnKF) inside the coupling loop, nonlinear or cross-support forward operators $H(x) \neq x$ inside `ObservationStateFeedbackFunctor`, and multi-step 3D-Var / 4D-Var variational trajectory optimization within this Functor.
- **IPHS Equivalence Certificates**: None (`0` certificates reference `observation_state_feedback.md`).

## 9. References

1. Evensen, G. (2003). The Ensemble Kalman Filter: Theoretical formulation and practical implementation. *Ocean Dynamics*, 53(4), 343-367.
2. Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35-45.
3. Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability* (Chapter 5, Eqs. 5.3.11-5.3.13). Cambridge University Press.
