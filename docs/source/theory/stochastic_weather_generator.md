# Standalone Stochastic Weather Generator

## 1. Process Definition and Scope

`relic/stochastic/weather_generator.py` implements three standalone multi-step meteorological trajectory samplers subclassing `relic.stochastic.category.StochasticProcess` (outside the `FunctorBase` / `StateManager` step-node hierarchy):

1. `OrnsteinUhlenbeckTemperature`: exact discrete-time Gaussian transition sampler, log-likelihood evaluator, and stationary moment calculator for air temperature $T$ ($\mathrm{K}$) reverting toward dry-day mean $\mu$ ($\mathrm{K}$).
2. `MarkovGammaPrecipitation`: daily two-state first-order Markov wet/dry occurrence chain coupled to two-parameter $\mathrm{Gamma}(k, \theta)$ wet-day water depth ($\mathrm{mm\,day^{-1}}$), emitted in CF-1.8 SI precipitation mass-flux units $q$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$).
3. `WeatherGenerator`: coupled daily generator ($\Delta t = 1.0\,\mathrm{day}$ strictly enforced) combining `OrnsteinUhlenbeckTemperature` and `MarkovGammaPrecipitation` with a Richardson (1981) WGEN wet-day additive temperature shift $\delta_{\mathrm{wet}}$ ($\mathrm{K}$) and Tetens saturation vapor pressure $e_s(T)$ ($\mathrm{Pa}$).

Solar radiation synthesis, relative humidity and vapor-pressure deficit generation, seasonal Fourier parameter harmonics, multi-site spatial weather correlation, and coupled joint `log_probability` evaluation when $\delta_{\mathrm{wet}} \ne 0$ remain `[N]`. Relative humidity is intentionally omitted because WGEN (Richardson, 1981) generates only precipitation, temperature, and solar radiation, whereas the WXGEN humidity model (Sharpley and Williams, 1990) requires site dew-point climatology and produces unclipped wet-day values exceeding $1$. Governed step-wise Simulator Functors for temperature anomaly and rainfall events are documented in `atmosphere_stochastic_temperature.md` and `atmosphere_stochastic_rainfall.md`, and theoretical derivations appear in *Methods of Ecological Analysis*, Chapter 27 (Section 27.2).

## 2. Governing Equations

### 2.1 Exact Discrete Ornstein-Uhlenbeck Temperature Transition (`OrnsteinUhlenbeckTemperature`)

For continuum air temperature $dT_t = \alpha(\mu - T_t)\,dt + \sigma\,dW_t$ with dry-day equilibrium mean $\mu$ ($\mathrm{K}$), reversion rate $\alpha > 0$ ($\mathrm{day^{-1}}$), and volatility $\sigma > 0$ ($\mathrm{K\,day^{-1/2}}$), `OrnsteinUhlenbeckTemperature.sample` integrates the exact Gaussian transition kernel over step $\Delta t > 0$ ($\mathrm{day}$) without Euler-Maruyama discretization drift (Gillespie, 1996; Gardiner, 2009):

$$
T_{n+1} = T_n\,e^{-\alpha\Delta t} + \mu\bigl(1 - e^{-\alpha\Delta t}\bigr) + \sigma_{\mathrm{eff}}\,\xi_n, \qquad
\sigma_{\mathrm{eff}} = \sigma\sqrt{\frac{1 - e^{-2\alpha\Delta t}}{2\alpha}}, \qquad
\xi_n \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, 1)
$$

Deterministic trend relaxation from initial state $T_0$ at step index $n \in \{0, \dots, N_{\mathrm{steps}}-1\}$ is recorded alongside the stochastic path as $T_{n}^{\mathrm{trend}} = \mu + (T_0 - \mu)e^{-\alpha n\Delta t}$. The stationary marginal distribution, lag-$s$ autocorrelation, and integral autocorrelation time returned by `moments()` are:

$$
T_\infty \sim \mathcal{N}\!\left(\mu,\; \frac{\sigma^2}{2\alpha}\right), \qquad
\rho(s) = e^{-\alpha\lvert s\rvert}, \qquad
\tau_c = \frac{1}{\alpha}
$$

`OrnsteinUhlenbeckTemperature.log_probability(observations, conditioning_state)` evaluates the exact conditional Gaussian log-density $\log \mathcal{N}\bigl(T_n;\, T_{n-1}e^{-\alpha\Delta t} + \mu(1 - e^{-\alpha\Delta t}),\, \sigma_{\mathrm{eff}}^2\bigr)$ when `conditioning_state` is supplied, and the stationary marginal log-density $\log \mathcal{N}\bigl(T_n;\, \mu,\, \sigma^2/(2\alpha)\bigr)$ when `conditioning_state` is `None`.

### 2.2 Two-State Markov Occurrence and Gamma Intensity (`MarkovGammaPrecipitation`)

Daily wet/dry occurrence $S_n \in \{0, 1\}$ ($0 = \text{dry}$, $1 = \text{wet}$) evolves as a homogeneous first-order Markov chain with transition probabilities $p_{01} = \mathbb{P}(S_n = 1 \mid S_{n-1} = 0)$ (`p_wet_dry`) and $p_{11} = \mathbb{P}(S_n = 1 \mid S_{n-1} = 1)$ (`p_wet_wet`) (Richardson, 1981; Katz, 1996; Wilks, 2006, 2011):

$$
\mathbf{P} = \begin{pmatrix} 1 - p_{01} & p_{01} \\ 1 - p_{11} & p_{11} \end{pmatrix}, \qquad
\pi_{\mathrm{wet}} = \frac{p_{01}}{p_{01} + (1 - p_{11})} = \frac{p_{01}}{1 + p_{01} - p_{11}} \quad (1 + p_{01} - p_{11} > 0)
$$

When $p_{01} = 0$ and $p_{11} = 1$ ($1 + p_{01} - p_{11} = 0$), both dry and wet states are absorbing and no unique stationary distribution exists; `_p_wet_steady` is stored as `None` and any call to `moments()` or unconditional `log_probability()` raises `ValueError`.

On wet days ($S_n = 1$), daily water depth $P_n$ ($\mathrm{mm\,day^{-1}}$) follows a two-parameter Gamma distribution with shape $k > 0$ (`gamma_shape`) and scale $\theta > 0$ ($\mathrm{mm\,day^{-1}}$, `gamma_scale`, parameterized in PyTorch as `Gamma(concentration=k, rate=1/theta)`), and converts to SI mass flux $q_n$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) through $S_d = 86400\,\mathrm{s\,day^{-1}}$ (`SECONDS_PER_DAY`):

$$
f_\Gamma(P; k, \theta) = \frac{P^{k-1}\,e^{-P/\theta}}{\Gamma(k)\,\theta^k} \quad (P > 0), \qquad
q_n = \frac{S_n\,P_n}{S_d}
$$

Using the law of total expectation and total variance over $S \sim \mathrm{Bernoulli}(\pi_{\mathrm{wet}})$ with $\mathbb{E}[P \mid S=1] = k\theta$ and $\mathrm{Var}[P \mid S=1] = k\theta^2$, the unconditional stationary flux moments returned by `MarkovGammaPrecipitation.moments()` are:

$$
\mathbb{E}[q] = \frac{\pi_{\mathrm{wet}}\,k\,\theta}{S_d}, \qquad
\mathrm{Var}[q] = \frac{\pi_{\mathrm{wet}}\bigl(k\theta^2 + (1 - \pi_{\mathrm{wet}})\,k^2\theta^2\bigr)}{S_d^2}
$$

In `MarkovGammaPrecipitation.log_probability(observations, conditioning_state)`, observations $q_n$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) are partitioned into wet and dry regimes at the declared numerical trace threshold $q_{\mathrm{thr}} \ge 0$ (`wet_day_threshold`) and evaluated with $P_n = q_n S_d$ and $p_{\mathrm{wet}} \in \{p_{01}, p_{11}, \pi_{\mathrm{wet}}\}$:

$$
\log p(q_n \mid S_{n-1}) = \begin{cases}
\log p_{\mathrm{wet}} + \log f_\Gamma(q_n S_d; k, \theta), & q_n > q_{\mathrm{thr}}, \\
\log(1 - p_{\mathrm{wet}}), & q_n \le q_{\mathrm{thr}}
\end{cases}
$$

### 2.3 Wet-Day Temperature Shift and Tetens Saturation Vapor Pressure (`WeatherGenerator`)

`WeatherGenerator` generates independent realizations $T_n^{\mathrm{OU}}$ ($\Delta t = 1\,\mathrm{day}$) and $(S_n, q_n)$, then applies the same-day WGEN wet-day additive offset $\delta_{\mathrm{wet}}$ (`wet_day_temperature_offset`, $\mathrm{K}$, Richardson, 1981):

$$
T_n = T_n^{\mathrm{OU}} + \delta_{\mathrm{wet}}\,S_n, \qquad
\Delta T_n = \bigl(T_n^{\mathrm{OU}} - \mu\bigr) + \delta_{\mathrm{wet}}\,S_n
$$

At stationarity, when $\delta_{\mathrm{wet}} \ne 0$, $T_n$ is a two-component Gaussian mixture over independent $T_\infty^{\mathrm{OU}}$ and $S_\infty \sim \mathrm{Bernoulli}(\pi_{\mathrm{wet}})$ with exact mean and variance (`WeatherGenerator.moments()`):

$$
\mathbb{E}[T] = \mu + \delta_{\mathrm{wet}}\,\pi_{\mathrm{wet}}, \qquad
\mathrm{Var}[T] = \frac{\sigma^2}{2\alpha} + \delta_{\mathrm{wet}}^2\,\pi_{\mathrm{wet}}(1 - \pi_{\mathrm{wet}})
$$

Because the mixture autocorrelation is a weighted sum of OU and Markov geometric decays, `temperature_autocorr_time` is omitted from `WeatherGenerator.moments()` whenever $\delta_{\mathrm{wet}} \ne 0$. Similarly, `WeatherGenerator.log_probability` sums `temp_gen.log_probability` and `precip_gen.log_probability` when $\delta_{\mathrm{wet}} = 0$, and raises `NotImplementedError` when $\delta_{\mathrm{wet}} \ne 0$.

Derived saturation vapor pressure $e_s(T_n)$ ($\mathrm{Pa}$) delegates to the canonical psychrometric helper `compute_saturation_vapor_pressure` (`relic/scientific/hydrosphere/utils/psychrometrics.py`; Tetens, 1930; Allen et al., 1998, Eq. 11):

$$
e_s(T) = e_0\,\exp\!\left(\frac{a\,(T - T_0)}{(T - T_0) + b}\right)
$$

with physical constants read exclusively from `physical/constants.yaml` via `get_physical_constant`.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $T_0$ | Initial air temperature tensor supplied to `sample(input_state, ...)` | $\mathrm{K}$ | `input_state` | `input` |
| $T_n$ | Sampled air temperature trajectory (wet-day shifted in `WeatherGenerator`) | $\mathrm{K}$ | `air_temperature` | `output` |
| $\Delta T_n$ | Temperature departure $T_n - \mu$ from dry-day mean $\mu$ | $\mathrm{K}$ | `temperature_anomaly` | `output` |
| $T_n^{\mathrm{trend}}$ | Deterministic mean-reversion curve $\mu + (T_0 - \mu)e^{-\alpha n\Delta t}$ | $\mathrm{K}$ | `temperature_trend` | `output` |
| $\Delta t$ | Broadcast temperature recursion step metadata | $\mathrm{day}$ | `time_step` | `output` |
| $\alpha_{\mathrm{out}}$ | Broadcast reversion-rate metadata tensor | $\mathrm{day^{-1}}$ | `reversion_rate` | `output` |
| $\sigma_{\mathrm{out}}$ | Broadcast volatility metadata tensor | $\mathrm{K\,day^{-1/2}}$ (`K day-0.5`) | `volatility` | `output` |
| $q_n$ | Daily precipitation water mass flux $S_n P_n / S_d$ | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `precipitation_flux` | `output` |
| $S_n$ | Binary wet/dry state indicator ($1 = \text{wet}$, $0 = \text{dry}$) | $1$ | `precipitation_state` | `output` |
| $L_{\mathrm{wet},n}$ | Consecutive wet-spell duration at step $n$ | $\mathrm{day}$ | `wet_spell_length` | `output` |
| $L_{\mathrm{dry},n}$ | Consecutive dry-spell duration at step $n$ | $\mathrm{day}$ | `dry_spell_length` | `output` |
| $\mathbf{P}$ | Broadcast $2 \times 2$ Markov transition matrix `(num_samples, num_steps, 2, 2)` | $1$ | `transition_probs` | `output` |
| $e_s(T_n)$ | Liquid-water Tetens saturation vapor pressure (`WeatherGenerator` only) | $\mathrm{Pa}$ | `saturation_vapor_pressure` | `output` |

## 4. Parameter Table

Dataclasses `TemperatureParameters` and `PrecipitationParameters` carry no hardcoded Python defaults; `from_parameter_manager()` retrieves every parameter from `relic/parameters/database/relic/stochastic/weather_parameters.yaml`, while physical constants resolve from `relic/parameters/database/relic/physical/constants.yaml`.

| Symbol | Parameter / Constant Key | Default | Bounds | Units | Dtype / Grad | Meaning |
|---|---|---|---|---|---|---|
| $\mu$ | `stochastic.weather.temperature.mean_temp` | $288.15$ | $[250.0, 320.0]$ | $\mathrm{K}$ | `float32`, `requires_grad=True` | Long-term (dry-day) equilibrium air temperature |
| $\alpha$ | `stochastic.weather.temperature.reversion_rate` | $0.1$ | $[0.01, 1.0]$ | $\mathrm{day^{-1}}$ | `float32`, `requires_grad=True` | Ornstein-Uhlenbeck mean reversion rate |
| $\sigma$ | `stochastic.weather.temperature.volatility` | $2.0$ | $[0.1, 10.0]$ | $\mathrm{K\,day^{-1/2}}$ (`K day-0.5`) | `float32`, `requires_grad=True` | Ornstein-Uhlenbeck temperature volatility |
| $\Delta t$ | `stochastic.weather.temperature.dt` | $1.0$ | $[0.001, 1.0]$ | $\mathrm{day}$ | `float32`, `requires_grad=False` | Temperature step (`1.0` required by `WeatherGenerator`) |
| $\delta_{\mathrm{wet}}$ | `stochastic.weather.temperature.wet_day_temperature_offset` | $0.0$ | $[-15.0, 15.0]$ | $\mathrm{K}$ | `float32`, `requires_grad=False` | Additive temperature shift applied on wet days ($S_n = 1$) |
| $p_{11}$ | `stochastic.weather.precipitation.p_wet_wet` | $0.7$ | $[0.0, 1.0]$ | $1$ | `float32`, `requires_grad=True` | Conditional wet-to-wet transition probability $\mathbb{P}(S_n=1 \mid S_{n-1}=1)$ |
| $p_{01}$ | `stochastic.weather.precipitation.p_wet_dry` | $0.2$ | $[0.0, 1.0]$ | $1$ | `float32`, `requires_grad=True` | Conditional dry-to-wet transition probability $\mathbb{P}(S_n=1 \mid S_{n-1}=0)$ |
| $k$ | `stochastic.weather.precipitation.gamma_shape` | $2.0$ | $[0.1, 10.0]$ | $1$ | `float32`, `requires_grad=True` | Gamma shape parameter for wet-day precipitation depth |
| $\theta$ | `stochastic.weather.precipitation.gamma_scale` | $5.0$ | $[0.1, 50.0]$ | $\mathrm{mm\,day^{-1}}$ | `float32`, `requires_grad=True` | Gamma scale parameter for wet-day precipitation depth |
| $S_{-1}$ | `stochastic.weather.precipitation.initial_state` | `false` | $\{0, 1\}$ | $1$ | `bool`, `requires_grad=False` | Pre-simulation wet/dry indicator (`False` = dry, `True` = wet) |
| $q_{\mathrm{thr}}$ | `stochastic.weather.precipitation.wet_day_threshold` | $1.0 \times 10^{-8}$ | $[0.0, 1.0 \times 10^{-4}]$ | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `float64`, `requires_grad=False` | Numerical trace threshold ($\approx 8.64 \times 10^{-4}\,\mathrm{mm\,day^{-1}}$) in `log_probability` |
| $S_d$ | `SECONDS_PER_DAY` (`relic.utils.time_constants`) | $86400.0$ | fixed SSOT | $\mathrm{s\,day^{-1}}$ | `float` | Seconds per standard day |
| $T_0$ | `celsius_to_kelvin` (`physical/constants.yaml`) | $273.15$ | fixed SSOT | $\mathrm{K}$ | `float` | Freezing-point Kelvin offset for Tetens vapor pressure |
| $e_0, a, b$ | `relic.physical.constants.psychrometrics.tetens_*` | $610.78,\, 17.27,\, 237.3$ | fixed SSOT | $\mathrm{Pa},\, 1,\, ^\circ\mathrm{C}$ | `float` | Tetens base pressure $e_0$, coefficient $a$, and offset $b$ |

## 5. Numerical Methods and Discretization

### 5.1 Trajectory Indexing, Spell Counters, and Forked RNG Stream Splitting

1. **Exact OU trajectory indexing (`OrnsteinUhlenbeckTemperature.sample`)**: For each ensemble member `sample_idx` in `0 .. num_samples - 1`, step `0` stores the cloned `input_state` ($T_0$) directly into `temperatures[sample_idx, 0]`, `anomalies[sample_idx, 0] = T_0 - mu`, and `trends[sample_idx, 0] = T_0`, then draws `torch.randn_like(current_temp) * var_coeff` to update `current_temp` for step `1`. Because the update executes at the bottom of every loop iteration `0 .. num_steps - 1`, step `num_steps - 1` draws one trailing standard-normal innovation whose updated `current_temp` is not stored.
2. **Markov-Gamma spell-counter convention (`MarkovGammaPrecipitation.sample`)**: Output tensors allocate on the device resolved by `relic.core.device_policy.resolve_device` (`explicit_device=input_state.device`, `default="auto"`), while Bernoulli transitions and `Gamma.sample()` draws execute inside `_seeded_draws(seed, torch.device("cpu"))`. Prior to step `0`, `current_state = int(initial_state)` initializes `(wet_spell, dry_spell)` to `(0, 1)` when dry or `(1, 0)` when wet, and step `0` increments the active spell counter before recording `wet_spell_length` and `dry_spell_length` (yielding `2` on step `0` because `initial_state` represents the preceding day and `current_state` is also used for step `0` before the first Bernoulli transition at the end of step `0`).
3. **Global RNG isolation and `SeedSequence` child spawning**: `_seeded_draws(seed, device)` wraps `torch.random.fork_rng` (seeding `torch.default_generator` and, on CUDA, the target device generator via `torch.cuda.manual_seed(seed)`) so caller global RNG streams remain untouched. In `WeatherGenerator.sample`, a provided integer `seed` spawns two statistically independent child seeds through `numpy.random.SeedSequence(seed).spawn(2)` (`temp_seed`, `precip_seed`), avoiding ensemble-member stream overlap produced by fixed additive seed offsets.

### 5.2 Architectural Relationship to Governed Atmospheric Stochastic Functors

| Aspect | Standalone Generators (`relic/stochastic/weather_generator.py`) | Governed Functors (`relic/scientific/atmosphere/stochastic_*.py`) |
|---|---|---|
| Base class & lifecycle | `StochasticProcess` (`sample`, `log_probability`, `moments`, `describe`) | `FunctorSubstreamMixin` + `PhysicalFieldFunctor` (`forward` returning `UpdateItem`s) |
| Time advancement | Generates full `(num_samples, num_steps, ...)` forcing arrays in one call | Advances `StateManager` cochains by one simulator step `dt` per `forward` call |
| Temperature formulation | Exact Gillespie (1996) Gaussian transition for $T$ around $\mu$, optional $\delta_{\mathrm{wet}} S_n$ | Explicit diagonal `EulerMaruyamaSolver` for zero-mean anomaly $X$ ($0 < \alpha\,\Delta t_d < 1$) |
| Precipitation formulation | Discrete daily Markov wet/dry chain + $\mathrm{Gamma}(k, \theta)$ flux ($\mathrm{kg\,m^{-2}\,s^{-1}}$) | Continuous-time `GillespieSSA` Poisson counts + exponential storm depths ($\mathrm{mm}$) |
| Checkpoint & GraphIR | Offline / ensemble forcing utility; no `GraphIR` node or `nn.Module` extra state | `state_dict` extra-state replay, Brownian-tree / jump ledgers, `stochastic_sdae_check` |

## 6. Conservation, Boundary Conditions, and Ledgers

The standalone weather generator synthesizes external atmospheric boundary forcing series ($T$, $q$, $e_s$) and does not own an internal mass or energy conservation ledger. Physical water-mass and surface-energy balances apply downstream when generated `precipitation_flux` and `air_temperature` force pedosphere and hydrosphere Functors.

All three classes enforce fail-closed parameter and call-signature guards:

- `_reject_options(kwargs)` raises `TypeError` on any unrecognized keyword option passed to `sample()`.
- `OrnsteinUhlenbeckTemperature.__init__` rejects `reversion_rate <= 0`, `volatility <= 0`, or `dt <= 0` with `ValueError`.
- `PrecipitationParameters.from_parameter_manager` rejects non-boolean `initial_state` with `TypeError`; `MarkovGammaPrecipitation.__init__` rejects $p_{11} \notin [0, 1]$, $p_{01} \notin [0, 1]$, $k \le 0$, $\theta \le 0$, or $q_{\mathrm{thr}} < 0$ with `ValueError`, and `_require_stationary_wet_probability` raises `ValueError` for the two-absorbing-state chain $(p_{01}=0, p_{11}=1)$.
- `WeatherGenerator.__init__` raises `ValueError` whenever `temp_params.dt` differs from $1.0\,\mathrm{day}$, and `WeatherGenerator.log_probability` raises `NotImplementedError` when `wet_day_temperature_offset` is non-zero or `ValueError` when the trailing axis of `observations` / `conditioning_state` does not have length $2$.

## 7. Code Mapping

| Component | Canonical Path / Symbol | Role |
|---|---|---|
| Temperature sampler | `relic/stochastic/weather_generator.py` (`TemperatureParameters`, `OrnsteinUhlenbeckTemperature`) | Exact discrete OU air-temperature trajectory sampler, Gaussian log-density, and moments |
| Precipitation sampler | `relic/stochastic/weather_generator.py` (`PrecipitationParameters`, `MarkovGammaPrecipitation`) | Daily two-state Markov occurrence and Gamma wet-day SI flux sampler, log-density, and moments |
| Coupled generator | `relic/stochastic/weather_generator.py` (`WeatherGenerator`) | Daily coupled sampler with `SeedSequence` stream splitting, wet-day temperature shift, and Tetens $e_s$ |
| Stochastic interface | `relic/stochastic/category.py` (`StochasticProcess`, `ProcessMetadata`) | Abstract base contract for standalone stochastic samplers |
| Psychrometric owner | `relic/scientific/hydrosphere/utils/psychrometrics.py` (`compute_saturation_vapor_pressure`) | Canonical liquid-water Tetens saturation vapor pressure calculation |
| Device & parameter SSOT | `relic/core/device_policy.py`, `relic/parameters/database/relic/stochastic/weather_parameters.yaml`, `relic/parameters/database/relic/physical/constants.yaml` | Device resolution (`resolve_device`), weather parameter definitions, and Tetens / time constants |

Because `OrnsteinUhlenbeckTemperature`, `MarkovGammaPrecipitation`, and `WeatherGenerator` are standalone `StochasticProcess` utilities outside `FunctorBase`, they carry no IPHS certificate under `docs/source/theory/iphs_certificates/`.

## 8. Implementation Status

| Capability | Status | Notes |
|---|---|---|
| Exact discrete Gillespie (1996) OU temperature sampling, `log_probability`, and `moments` | `[I]` | Exact for all $\Delta t > 0$; forked RNG preserves global PyTorch generator state |
| Daily two-state Markov-Gamma precipitation sampling in CF-1.8 SI flux units ($\mathrm{kg\,m^{-2}\,s^{-1}}$) | `[I]` | Rejects stationary queries on degenerate absorbing chains $(p_{01}=0, p_{11}=1)$ |
| Coupled daily `WeatherGenerator` with WGEN wet-day temperature shift and mixture moments | `[I]` | Enforces `dt == 1.0` day and spawns independent child seeds via `np.random.SeedSequence` |
| Canonical Tetens saturation vapor pressure $e_s(T)$ from SSOT psychrometric constants | `[I]` | Delegates to `compute_saturation_vapor_pressure` |
| Joint `WeatherGenerator.log_probability` for non-zero `wet_day_temperature_offset` | `[N]` | Raises `NotImplementedError` whenever $\delta_{\mathrm{wet}} \ne 0$ |
| Relative humidity, vapor-pressure deficit, or solar radiation generation | `[N]` | Excluded by design; WGEN omits RH and unclipped WXGEN wet-day triangular draws exceed unity |
| Seasonal parameter harmonics or multi-site spatial weather correlation | `[N]` | Stationary scalar parameters per configured instance |

## 9. References

- Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements* (FAO Irrigation and Drainage Paper 56). FAO.
- Gardiner, C. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences* (4th ed.). Springer.
- Gillespie, D. T. (1996). Exact numerical simulation of the Ornstein-Uhlenbeck process and its integral. *Physical Review E*, 54(2), 2084-2091.
- Hasselmann, K. (1976). Stochastic climate models. Part I. Theory. *Tellus*, 28(6), 473-485.
- Katz, R. W. (1996). Use of conditional stochastic models to generate climate change scenarios. *Climatic Change*, 32(3), 237-255.
- Kilsby, C. G., Jones, P. D., Burton, A., Ford, A. C., Fowler, H. J., Harpham, C., James, P., Smith, A., and Wilby, R. L. (2007). A daily weather generator for use in climate change studies. *Environmental Modelling & Software*, 22(12), 1705-1719.
- Richardson, C. W. (1981). Stochastic simulation of daily precipitation, temperature, and solar radiation. *Water Resources Research*, 17(1), 182-190.
- Richardson, C. W., and Wright, D. A. (1984). *WGEN: A Model for Generating Daily Weather Variables* (ARS-8). U.S. Department of Agriculture, Agricultural Research Service.
- Srikanthan, R., and McMahon, T. A. (2001). Stochastic generation of annual, monthly and daily climate data: A review. *Hydrology and Earth System Sciences*, 5(4), 653-670.
- Tetens, O. (1930). Uber einige meteorologische Begriffe. *Zeitschrift fur Geophysik*, 6, 297-309.
- Wilks, D. S. (2006). *Statistical Methods in the Atmospheric Sciences* (2nd ed.). Academic Press.
- Wilks, D. S. (2011). *Statistical Methods in the Atmospheric Sciences* (3rd ed.). Academic Press.
- *Methods of Ecological Analysis*, Chapter 27 (Section 27.2): Stochastic Weather Generators and Exact Ornstein-Uhlenbeck Transitions.
