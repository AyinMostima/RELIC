# Ornstein-Uhlenbeck Stochastic Air-Temperature Anomaly

## 1. Process Definition and Scope

`OrnsteinUhlenbeckTemperatureFunctor` (`relic/scientific/atmosphere/stochastic_temperature.py`) advances a zero-mean prognostic air-temperature anomaly field `stochastic_air_temperature_anomaly` ($X$, $\mathrm{K}$) governed by a linear Ito Ornstein-Uhlenbeck stochastic differential equation (`stochastic_category = "continuous_sde"`, `noise_classification = "external_forcing"`) under `PhysicalFieldFunctor` (`ProcessCategory.PHYSICAL_FIELD`, `ExecutionPhase.MAIN_PROCESSES`, `SpatialScale.GRID_CELL`, `TemporalScale.HOUR`). Operating as a zero-dimensional point model per grid cell (`SPATIAL_DIMS = None`), `forward` delegates each explicit step to `relic.stochastic.sde_solver.EulerMaruyamaSolver` (`solver_name = "euler_maruyama"`, `diffusion_structure = "diagonal"`) driven by a Functor-owned Wiener substream (`FunctorSubstreamMixin`).

Non-diagonal or spatially correlated diffusion matrices, Milstein or implicit stochastic integrators, non-zero or seasonally varying baseline temperatures inside the Functor, fluctuation-dissipation thermal closures, coordinate queries (`get_value_at` raises `NotImplementedError`), and cross-device substream migration after `draw_count > 0` remain `[N]`. Standalone exact-transition sampling of absolute air temperature around a non-zero dry-day mean appears in `stochastic_weather_generator.md`, and theoretical foundations appear in *Methods of Ecological Analysis*, Chapter 27 (Sections 27.2 and 27.5).

## 2. Governing Equations

### 2.1 Continuum Ito Ornstein-Uhlenbeck Anomaly Equation

Following Hasselmann (1976) and Gardiner (2009), synoptic atmospheric weather fluctuations force a slow air-temperature anomaly $X_t$ ($\mathrm{K}$) relaxing linearly toward $0\,\mathrm{K}$ at mean reversion rate $\alpha > 0$ ($\mathrm{day^{-1}}$) under additive white-noise volatility $\sigma > 0$ ($\mathrm{K\,day^{-1/2}}$):

$$
dX_t = a(X_t)\,dt + b(X_t)\,dW_t = -\alpha\,X_t\,dt + \sigma\,dW_t
$$

where $W_t$ is a standard Wiener process on the day clock ($\mathbb{E}[dW_t] = 0$, $\mathrm{Var}[dW_t] = dt$). Because the diffusion coefficient $b(X_t) = \sigma$ is state-independent, the Ito and Stratonovich interpretations coincide. Conditioning on initial anomaly $X_0$, the exact continuum Gaussian solution has moments:

$$
\mathbb{E}[X_t \mid X_0] = X_0\,e^{-\alpha t}, \qquad
\mathrm{Var}[X_t \mid X_0] = \frac{\sigma^2}{2\alpha}\bigl(1 - e^{-2\alpha t}\bigr), \qquad
\mathrm{Var}_{\infty}^{\mathrm{cont}}[X] = \frac{\sigma^2}{2\alpha}
$$

### 2.2 Discrete Euler-Maruyama AR(1) Recursion and Exact Discrete Moments

Converting the simulator step $\Delta t > 0$ ($\mathrm{s}$) to days via $\Delta t_d = \Delta t / S_d$ ($S_d = 86400\,\mathrm{s\,day^{-1}}$, `SECONDS_PER_DAY`) and defining the explicit autoregressive multiplier $\phi = 1 - \alpha\,\Delta t_d$, one Euler-Maruyama step advances $X_n$ according to (Kloeden and Platen, 1992; Higham, 2001):

$$
X_{n+1} = X_n - \alpha\,X_n\,\Delta t_d + \sigma\,\Delta W_n = \phi\,X_n + \sigma\,\Delta W_n, \qquad
\Delta W_n \sim \mathcal{N}(0, \Delta t_d)
$$

Under the strict explicit contraction admissibility constraint $0 < \alpha\,\Delta t_d < 1$ ($\phi \in (0, 1)$), the discrete trajectory forms a stationary first-order autoregressive Gaussian Markov chain whose exact step-$n$ conditional moments (`OrnsteinUhlenbeckTemperatureFunctor.discrete_moments`) and asymptotic stationary variance are:

$$
\mathbb{E}[X_n \mid X_0] = \phi^n\,X_0, \qquad
\mathrm{Var}[X_n \mid X_0] = \sigma^2\,\Delta t_d\,\frac{1 - \phi^{2n}}{1 - \phi^2}
$$

$$
\mathrm{Var}_{\infty}^{\mathrm{EM}}[X] = \frac{\sigma^2\,\Delta t_d}{1 - (1 - \alpha\,\Delta t_d)^2} = \frac{\sigma^2}{2\alpha\bigl(1 - \tfrac{1}{2}\alpha\,\Delta t_d\bigr)}
$$

Explicit Euler-Maruyama discretization inflates the stationary variance above the continuum value $\sigma^2/(2\alpha)$ by the factor $\bigl(1 - \tfrac{1}{2}\alpha\,\Delta t_d\bigr)^{-1} = 1 + \tfrac{1}{2}\alpha\,\Delta t_d + O\bigl((\alpha\,\Delta t_d)^2\bigr)$.

Because `wiener_increment` is detached before evaluation of the linear drift and diagonal diffusion inside `EulerMaruyamaSolver.sample`, pathwise automatic differentiation through the emitted increment $\Delta X_n = -\alpha\,X_n\,\Delta t_d + \sigma\,\Delta W_n$ yields exact sensitivities:

$$
\frac{\partial (\Delta X_n)}{\partial X_n} = -\alpha\,\Delta t_d, \qquad
\frac{\partial (\Delta X_n)}{\partial \alpha} = -X_n\,\Delta t_d, \qquad
\frac{\partial (\Delta X_n)}{\partial \sigma} = \Delta W_n
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $X$ | Prognostic zero-mean air-temperature anomaly (`UpdateMode.ACCUMULATE`, `per_timestep=True`) | $\mathrm{K}$ | `stochastic_air_temperature_anomaly` | `state` |
| $a(X_n)$ | Step-start linear relaxation drift $-\alpha\,X_n$ (`UpdateMode.REPLACE`, `per_timestep=True`) | $\mathrm{K\,day^{-1}}$ | `diagnostics.stochastic_temperature.drift` | `diagnostic` |
| $b(X_n)$ | Broadcast diagonal diffusion volatility $\sigma$ (`UpdateMode.REPLACE`, `per_timestep=True`) | $\mathrm{K\,day^{-1/2}}$ (`K day-0.5`) | `diagnostics.stochastic_temperature.diffusion` | `diagnostic` |
| $\Delta W_n$ | Realized Wiener increment $\sqrt{\Delta t_d}\,\xi_n$ (`UpdateMode.REPLACE`, `per_timestep=True`) | $\mathrm{day}^{1/2}$ (declared code/certificate unit string: `day-0.5`) | `diagnostics.stochastic_temperature.wiener_increment` | `diagnostic` |

**Wiener increment unit convention**: Mathematically, $\Delta W_n = \sqrt{\Delta t_d}\,\xi_n$ has dimension $\mathrm{time}^{1/2}$ ($\mathrm{day}^{1/2}$), so that multiplying $\sigma$ ($\mathrm{K\,day^{-1/2}}$) by $\Delta W_n$ ($\mathrm{day}^{1/2}$) yields the temperature increment in $\mathrm{K}$. In live runtime code (`stochastic_temperature.py`), the `StateContractManager` diagnostic contract, and the IPHS certificate `unit_map`, `diagnostics.stochastic_temperature.wiener_increment` is registered with the literal unit string `"day-0.5"` while storing the numerical tensor $\sqrt{\Delta t_d}\,\xi_n$.

## 4. Parameter Table

Parameters reside in `relic/parameters/database/relic/stochastic/weather_parameters.yaml` under `parameter_namespace = "stochastic.weather.temperature"`. `NamespaceDefaultMixin` evaluates explicit constructor overrides ahead of `ParameterManager` (`parameter_lookup_source_order = ("config", "manager")`) and rejects undeclared override names.

| Symbol | Parameter Key / Short Name | Default | Bounds | Units | Dtype / Grad | Meaning |
|---|---|---|---|---|---|---|
| $\alpha$ | `stochastic.weather.temperature.reversion_rate` (`reversion_rate`) | $0.1$ | $[0.01, 1.0]$ | $\mathrm{day^{-1}}$ | `float32`, `requires_grad=True` | Linear Ornstein-Uhlenbeck mean reversion rate |
| $\sigma$ | `stochastic.weather.temperature.volatility` (`volatility`) | $2.0$ | $[0.1, 10.0]$ | $\mathrm{K\,day^{-1/2}}$ (`K day-0.5`) | `float32`, `requires_grad=True` | Additive white-noise temperature volatility |
| $s_{\mathrm{rng}}$ | `config["seed"]` | required | $[0, 2^{63}-1]$ | $1$ | `int` | Mandatory integer seed initializing the Functor-owned Wiener substream |
| $S_d$ | `SECONDS_PER_DAY` (`relic.utils.time_constants`) | $86400.0$ | fixed SSOT | $\mathrm{s\,day^{-1}}$ | `float` | Seconds per standard day converting simulator `dt` to $\Delta t_d$ |

## 5. Numerical Methods and Discretization

### 5.1 Single-Step Diagonal Euler-Maruyama Execution and Admissibility Gate

On each `forward(state, time, dt)` call, `OrnsteinUhlenbeckTemperatureFunctor` computes $\Delta t_d = \Delta t / S_d$ and verifies $0 < \alpha\,\Delta t_d < 1$ prior to drawing from the random substream. Steps outside $(0, 1)$ (whether non-damped $\alpha\,\Delta t_d \le 0$, oscillatory $1 \le \alpha\,\Delta t_d < 2$, or unstable $\alpha\,\Delta t_d \ge 2$) raise `ValueError` without step clipping.

Once validated, `forward` draws `wiener_increment` on the Functor-owned generator and invokes `EulerMaruyamaSolver(SDEParameters(dt=dt_days, method="euler_maruyama", diffusion_structure="diagonal"), drift_func=drift, diffusion_func=diffusion)` with `num_samples=1`, `num_steps=1`, `initial_time=time_days`, and `wiener_increments=wiener_increment.detach().reshape(1, 1, *anomaly.shape)`. Because the diffusion term $\sigma$ is additive and constant ($\partial b / \partial X = 0$), the Milstein correction vanishes identically and Euler-Maruyama attains strong convergence order $1.0$ and weak convergence order $1.0$ (Kloeden and Platen, 1992).

### 5.2 Brownian-Tree Path Refinement Across Rejected Adaptive Steps and Substream Replay

`FunctorSubstreamMixin` (`relic/core/functor/stochastic_substream.py`) equips `OrnsteinUhlenbeckTemperatureFunctor` (`EXTRA_STATE_FORMAT = "relic.stochastic_temperature.wiener_substream"`, `EXTRA_STATE_VERSION = 1`) with an owned `torch.Generator` and a Brownian-tree pending ledger `wiener_pending` (Gaines and Lyons, 1997; Karlin and Taylor, 1981):

1. **Fresh interval**: When no pending increment starts at $t_d$, `draw_wiener_increment(t_d, h, like=anomaly)` draws $\xi \sim \mathcal{N}(0, I)$ on `self.generator` and returns $\Delta W(h) = \sqrt{h}\,\xi$.
2. **Rejected trial pinning (`record_rejected_trial_path`)**: When an outer adaptive controller rejects a trial step $[t_d, t_d + H)$, it first restores `extra_state` (`seed`, `draw_count`, `generator_state`, `wiener_pending`) to the pre-trial snapshot and then calls `record_rejected_trial_path()`, which replays the trial draw $\Delta W(H)$ from that exact generator state and appends `{time: t_d, width: H, increment: Delta W(H)}` to `wiener_pending`.
3. **Brownian-bridge split and recursive merge**: A subsequent retry on $[t_d, t_d + h)$ with $h < H$ splits the pinned increment $\Delta W(H)$ via the exact conditional Brownian bridge:

$$
\Delta W(h) = \frac{h}{H}\,\Delta W(H) + \sqrt{\frac{h(H - h)}{H}}\,\xi, \qquad
\Delta W(H - h) = \Delta W(H) - \Delta W(h), \quad \xi \sim \mathcal{N}(0, I)
$$

storing $\Delta W(H - h)$ as pending at $t_d + h$ so that $\Delta W(h) + \Delta W(H - h) = \Delta W(H)$ holds pathwise. If $h > H$, `_draw_on_path` consumes $\Delta W(H)$ and adds a recursive draw over $[t_d + H, t_d + h)$. Any pending interval overlapping $[t_d, t_d + h)$ without matching $t_d$ within relative tolerance $10^{-9}$ raises `RuntimeError`.

### 5.3 Comparison with Standalone `OrnsteinUhlenbeckTemperature`

| Feature | `OrnsteinUhlenbeckTemperatureFunctor` (`stochastic_temperature.py`) | `OrnsteinUhlenbeckTemperature` (`weather_generator.py`) |
|---|---|---|
| Execution contract | Governed `PhysicalFieldFunctor` step node in `StateManager` / `GraphIR` | Standalone `StochasticProcess` multi-step trajectory sampler |
| Prognostic variable | Zero-mean temperature anomaly $X$ ($\mathrm{K}$) | Absolute air temperature $T$ ($\mathrm{K}$) around dry-day mean $\mu$ |
| Discretization | Explicit diagonal Euler-Maruyama ($0 < \alpha\,\Delta t_d < 1$ enforced) | Exact Gillespie (1996) Gaussian transition kernel ($\forall\,\alpha\,\Delta t > 0$) |
| RNG & adaptive path | Private `torch.Generator` + Brownian-tree `wiener_pending` in `nn.Module` extra state | Scoped `torch.random.fork_rng` context during `sample()` |

## 6. Conservation, Boundary Conditions, and Ledgers

`OrnsteinUhlenbeckTemperatureFunctor` represents an open external climate-anomaly forcing closure (Hasselmann, 1976) and declares `conservation_laws = ["none"]`. No thermal energy budget, Hamiltonian functional, internal entropy production, or fluctuation-dissipation constraint is claimed.

Pre-draw validation is fail-closed: non-empty `kwargs` (`TypeError`), missing `stochastic_air_temperature_anomaly` (`ValueError`), non-tensor (`TypeError`), non-floating-point or non-finite state entries (`ValueError`), non-positive or non-finite `dt` (`ValueError`), non-scalar parameters (`ValueError`), $\alpha\,\Delta t_d \notin (0, 1)$ (`ValueError`), and mismatched `generator` versus state tensor device types (`ValueError`) all abort prior to advancing `self.generator` or incrementing `self.draw_count`.

## 7. Code Mapping

| Layer | Canonical Owner | Responsibility |
|---|---|---|
| Scientific Functor | `relic/scientific/atmosphere/stochastic_temperature.py` (`OrnsteinUhlenbeckTemperatureFunctor`) | Point-model `PhysicalFieldFunctor` advancing $X$ and three SDE diagnostics |
| Substream & Brownian tree | `relic/core/functor/stochastic_substream.py` (`FunctorSubstreamMixin`) | Seed validation, private `torch.Generator`, Brownian-bridge `wiener_pending`, `state_dict` extra state |
| SDE step solver | `relic/stochastic/sde_solver.py` (`EulerMaruyamaSolver`, `SDEParameters`) | Validates drift/diffusion shapes and executes one diagonal Euler-Maruyama step |
| Workflow & GraphIR binding | `relic/stochastic/workflows.py` (`build_bound_continuous_sde_graph`, `execute_bound_continuous_sde`) | Compiles single-node `GraphIR` validated by `stochastic_sdae_check` (`declares_stochastic_terms: true`) for CLI `stochastic sde` and `POST /stochastic/sde` |
| Parameter database | `relic/parameters/database/relic/stochastic/weather_parameters.yaml` | Defines `stochastic.weather.temperature.reversion_rate` and `stochastic.weather.temperature.volatility` |
| IPHS certificate | `docs/source/theory/iphs_certificates/atmosphere_stochastic_temperature_ornstein_uhlenbeck_temperature_functor.yaml` | Level-2 `field` (`continuous_sde`) equivalence, port, SDAE term, and reproducibility contract |

## 8. Implementation Status

| Capability | Status | Notes |
|---|---|---|
| Explicit diagonal Euler-Maruyama OU anomaly step with $0 < \alpha\,\Delta t_d < 1$ gate | `[I]` | Executed via `EulerMaruyamaSolver` with `num_steps=1` |
| Exact discrete AR(1) conditional moments (`discrete_moments`) and pathwise autograd | `[I]` | Differentiable with respect to $X_n$, `reversion_rate`, and `volatility` |
| Replayable `nn.Module` extra state and Brownian-tree rejected-step refinement | `[I]` | `EXTRA_STATE_FORMAT = "relic.stochastic_temperature.wiener_substream"`, `draw_wiener_increment`, `record_rejected_trial_path` |
| Governed `GraphIR` / `stochastic_sdae_check` workflow execution | `[I]` | Bound through `build_bound_continuous_sde_graph` / `execute_bound_continuous_sde` |
| Correlated / general non-diagonal diffusion matrices or higher-order SDE schemes | `[N]` | `SDEParameters` restricts `diffusion_structure` to `"diagonal"` and `method` to `"euler_maruyama"` |
| Spatial coordinate queries (`get_value_at`) or post-draw cross-device substream migration | `[N]` | Point model (`SPATIAL_DIMS = None`); `to_device` across device types requires `draw_count == 0` |

`OrnsteinUhlenbeckTemperatureFunctor` holds an IPHS Level-2 certificate (`canonical_role: field`, `iphs_level: 2`, `equivalence_type: exact`) in `atmosphere_stochastic_temperature_ornstein_uhlenbeck_temperature_functor.yaml`. Its frozen certificate expression literals are:

```text
dX = -alpha X dt + sigma dW
X_{n+1} = X_n - alpha X_n dt + sigma dW_n, dW_n ~ N(0, dt), Functor-owned Wiener substream
```

## 9. References

- Gaines, J. G., and Lyons, T. J. (1997). Variable step size control in the numerical solution of stochastic differential equations. *SIAM Journal on Applied Mathematics*, 57(5), 1455-1484.
- Gardiner, C. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences* (4th ed.). Springer.
- Hasselmann, K. (1976). Stochastic climate models. Part I. Theory. *Tellus*, 28(6), 473-485.
- Higham, D. J. (2001). An algorithmic introduction to numerical simulation of stochastic differential equations. *SIAM Review*, 43(3), 525-546.
- Karlin, S., and Taylor, H. M. (1981). *A Second Course in Stochastic Processes*. Academic Press.
- Kloeden, P. E., and Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
- *Methods of Ecological Analysis*, Chapter 27 (Sections 27.2 and 27.5): Ornstein-Uhlenbeck Climate Forcing and Brownian-Tree Substream Consistency.
