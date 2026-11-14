# Marked-Poisson Stochastic Rainfall

## 1. Process Definition and Scope

`MarkedPoissonRainfallFunctor` (`relic/scientific/atmosphere/stochastic_rainfall.py`) advances continuous-time marked-Poisson rainfall arrivals (`stochastic_category = "marked_jump"`, `noise_classification = "external_forcing"`) under `PhysicalFieldFunctor` (`ProcessCategory.PHYSICAL_FIELD`, `ExecutionPhase.MAIN_PROCESSES`, `SpatialScale.GRID_CELL`, `TemporalScale.HOUR`). Each cell evolves two prognostic point-model states (`SPATIAL_DIMS = None`): cumulative event count `stochastic_rainfall_event_count` ($N$, dimensionless `"1"`) and cumulative rainfall water depth `stochastic_rainfall_depth` ($H$, $\mathrm{mm}$). Arrivals within each simulator step are sampled through `relic.stochastic.gillespie.GillespieSSA` (`solver_name = "direct_ssa"`) on a single zero-order channel, and exponential storm depths (`mark_law = "exponential"`) are drawn from the same Functor-owned `torch.Generator` (`FunctorSubstreamMixin`).

State-dependent arrival hazards $\lambda(x, t)$, multi-species reaction networks, negative or removal depth marks, seasonal arrival-rate harmonics inside the Functor, fluctuation-dissipation closures, spatial coordinate queries (`get_value_at` raises `NotImplementedError`), and cross-device substream migration after `draw_count > 0` remain `[N]`. Standalone daily two-state Markov-Gamma precipitation sampling is documented in `stochastic_weather_generator.md`, and the mathematical foundations of ecohydrological marked-jump forcing and stochastic path consistency appear in *Methods of Ecological Analysis*, Chapter 27 (Sections 27.3 and 27.5).

## 2. Governing Equations

### 2.1 Homogeneous Poisson Counting Measure and Exponential Depth Marks

Storm arrivals follow a homogeneous Poisson process $N_t$ on $t \ge 0$ (measured in days) with constant arrival rate $\lambda > 0$ ($\mathrm{day^{-1}}$). Associated with $N_t$ is a Poisson random measure $\mathcal{N}(dt, dz)$ on $\mathbb{R}_+ \times (0, \infty)$ whose predictable compensator is $\nu(dt, dz) = \lambda\,p_Z(z)\,dz\,dt$, where individual storm depths $z_i \ge 0$ ($\mathrm{mm}$) are independent and identically distributed exponential marks with mean $\alpha > 0$ ($\mathrm{mm}$) and density $p_Z(z) = \alpha^{-1}\exp(-z/\alpha)$ (Rodriguez-Iturbe et al., 1999; Laio et al., 2001):

$$
dN_t = \int_0^\infty \mathcal{N}(dt, dz), \qquad
dH_t = \int_0^\infty z\,\mathcal{N}(dt, dz) = \sum_{i=1}^{dN_t} z_i
$$

Over a simulator time step $\Delta t > 0$ ($\mathrm{s}$), converted once to days via $\Delta t_d = \Delta t / S_d$ with $S_d = 86400\,\mathrm{s\,day^{-1}}$ (`SECONDS_PER_DAY`), the discrete increments and their predictable compensators with respect to the filtration $\mathcal{F}_{t_n}$ are:

$$
\Delta N_n \sim \mathrm{Poisson}(\lambda\,\Delta t_d), \qquad
\Delta H_n = \alpha \sum_{i=1}^{\Delta N_n} E_i, \quad E_i \stackrel{\mathrm{i.i.d.}}{\sim} \mathrm{Exp}(1)
$$

$$
\nu_{N,n} = \mathbb{E}\bigl[\Delta N_n \mid \mathcal{F}_{t_n}\bigr] = \lambda\,\Delta t_d, \qquad
\nu_{H,n} = \mathbb{E}\bigl[\Delta H_n \mid \mathcal{F}_{t_n}\bigr] = \lambda\,\alpha\,\Delta t_d
$$

In martingale-compensated notation $\tilde{\mathcal{N}}(dt, dz) = \mathcal{N}(dt, dz) - \nu(dt, dz)$, expressing the jump integral against $\tilde{\mathcal{N}}$ requires adding $(\lambda\,\Delta t_d, \lambda\,\alpha\,\Delta t_d)$ back as deterministic drift. `MarkedPoissonRainfallFunctor` emits the uncompensated increments $(\Delta N_n, \Delta H_n)$ directly onto the prognostic states and publishes $(\nu_{N,n}, \nu_{H,n})$ as `REPLACE` diagnostics, preserving the exact compound-Poisson mean without double-counting the compensator.

### 2.2 Exact Compound-Poisson Discrete Moments

For exponential marks with $\mathbb{E}[z] = \alpha$ and $\mathbb{E}[z^2] = 2\alpha^2$, Wald's identities for compound-Poisson processes (Karlin and Taylor, 1981) yield the exact step-$n$ mean and variance evaluated by `MarkedPoissonRainfallFunctor.discrete_moments`:

$$
\mathbb{E}[N_n] = N_0 + n\,\lambda\,\Delta t_d, \qquad
\mathrm{Var}[N_n] = n\,\lambda\,\Delta t_d
$$

$$
\mathbb{E}[H_n] = H_0 + n\,\lambda\,\alpha\,\Delta t_d, \qquad
\mathrm{Var}[H_n] = n\,\lambda\,\mathbb{E}[z^2]\,\Delta t_d = 2\,n\,\lambda\,\alpha^2\,\Delta t_d
$$

Because $\Delta H_n = \alpha \sum_{i=1}^{\Delta N_n} E_i$ scales fixed unit-exponential variates $E_i$ by the differentiable parameter tensor $\alpha$, the pathwise sensitivity with respect to mean storm depth is exact on every realized trajectory:

$$
\frac{\partial (\Delta H_n)}{\partial \alpha} = \sum_{i=1}^{\Delta N_n} E_i = \frac{\Delta H_n}{\alpha}
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $N$ | Cumulative rainfall event count (`UpdateMode.ACCUMULATE`, `per_timestep=True`) | $1$ | `stochastic_rainfall_event_count` | `state` |
| $H$ | Cumulative rainfall depth (`UpdateMode.ACCUMULATE`, `per_timestep=True`) | $\mathrm{mm}$ | `stochastic_rainfall_depth` | `state` |
| $\Delta N_n$ | Step arrival count observation (`UpdateMode.REPLACE`, `per_timestep=True`) | $1$ | `diagnostics.stochastic_rainfall.event_count_increment` | `diagnostic` |
| $\Delta H_n$ | Step rainfall depth increment observation (`UpdateMode.REPLACE`, `per_timestep=True`) | $\mathrm{mm}$ | `diagnostics.stochastic_rainfall.depth_increment` | `diagnostic` |
| $\nu_{N,n}$ | Predictable count compensator $\lambda\,\Delta t_d$ (`UpdateMode.REPLACE`, `per_timestep=True`) | $1$ | `diagnostics.stochastic_rainfall.compensator_count` | `diagnostic` |
| $\nu_{H,n}$ | Predictable depth compensator $\lambda\,\alpha\,\Delta t_d$ (`UpdateMode.REPLACE`, `per_timestep=True`) | $\mathrm{mm}$ | `diagnostics.stochastic_rainfall.compensator_depth` | `diagnostic` |

## 4. Parameter Table

Parameters live in `relic/parameters/database/relic/stochastic/weather_parameters.yaml` under `parameter_namespace = "stochastic.weather.rainfall"`. `NamespaceDefaultMixin` resolves explicit constructor overrides (`config={"seed": ..., "parameters": {...}}`) ahead of `ParameterManager` (`parameter_lookup_source_order = ("config", "manager")`), rejecting unknown override keys.

| Symbol | Parameter Key / Short Name | Default | Bounds | Units | Dtype / Grad | Meaning |
|---|---|---|---|---|---|---|
| $\lambda$ | `stochastic.weather.rainfall.event_rate` (`event_rate`) | $0.3$ | $[0.01, 5.0]$ | $\mathrm{day^{-1}}$ | `float64`, `requires_grad=False` | Homogeneous Poisson storm arrival rate |
| $\alpha$ | `stochastic.weather.rainfall.mean_depth` (`mean_depth`) | $15.0$ | $[0.1, 200.0]$ | $\mathrm{mm}$ | `float64`, `requires_grad=True` | Mean storm water depth of the exponential mark law |
| $s_{\mathrm{rng}}$ | `config["seed"]` | required | $[0, 2^{63}-1]$ | $1$ | `int` | Mandatory integer seed initializing the Functor-owned `torch.Generator` |
| $S_d$ | `SECONDS_PER_DAY` (`relic.utils.time_constants`) | $86400.0$ | fixed SSOT | $\mathrm{s\,day^{-1}}$ | `float` | Seconds per standard day converting simulator `dt` to $\Delta t_d$ |

## 5. Numerical Methods and Discretization

### 5.1 Direct Gillespie SSA Arrival Clock and Fail-Closed Tail Guard

`_arrival_process` constructs and caches a single-species, single-reaction `ReactionSystem` (`EVENT_SPECIES = "rain_event"`, `ARRIVAL_REACTION = "arrival"`, zero reactants, product stoichiometry $+1$, hazard $a(x) = \lambda$) owned by `GillespieSSA` (Gillespie, 1977) and rebuilds it only if $\lambda$ changes. Because `GillespieSSA.sample` preallocates trajectory buffers of length `max_reactions`, `MarkedPoissonRainfallFunctor.arrival_cap` computes a Chernoff-safe upper bound for expected per-cell arrivals $\mu = \lambda\,w_d$ over window width $w_d$ ($\mathrm{day}$):

$$
N_{\mathrm{cap}}(\mu) = \bigl\lceil \mu + s_{\mathrm{cap}}\sqrt{\mu} + b_{\mathrm{cap}} \bigr\rceil, \qquad
s_{\mathrm{cap}} = 12.0, \quad b_{\mathrm{cap}} = 64, \quad N_{\mathrm{cap}}(\mu) \le N_{\max} = 100000
$$

where `ARRIVAL_CAP_SIGMAS = 12.0`, `ARRIVAL_CAP_BASE = 64`, and `MAX_EVENTS_PER_STEP = 100000`. Two fail-closed guards prevent silent Poisson tail truncation:

1. If $N_{\mathrm{cap}}(\mu) > 100000$, `arrival_cap` raises `ValueError` prior to any random draw, requiring a smaller time step `dt`.
2. After `ssa.sample(..., final_time=window_days, max_reactions=cap, generator=self.generator)` returns `arrivals = sample["trajectory_lengths"] - 1`, any cell satisfying $\text{arrivals}_c \ge N_{\mathrm{cap}}(\mu)$ raises `RuntimeError` prior to state emission.

When total arrivals $K = \sum_{c=0}^{n_{\mathrm{cells}}-1} k_c > 0$, `_sample_jump_events` extracts arrival times $0 < \tau_{c,1} < \dots < \tau_{c,k_c} < w_d$ in cell-major order, draws $K$ unit exponential marks $E_m \sim \mathrm{Exp}(1)$ in one call (`unit_marks.exponential_(1.0, generator=self.generator)`), accumulates cell mark sums via `index_add_`, and multiplies by `mean_depth`.

### 5.2 Marked-Jump Path Ledger Across Adaptive Step Rejections and Substream Replay

`MarkedPoissonRainfallFunctor` inherits `FunctorSubstreamMixin` (`relic/core/functor/stochastic_substream.py`) and extends `get_extra_state` / `set_extra_state` (`EXTRA_STATE_FORMAT = "relic.stochastic_rainfall.event_substream"`, `EXTRA_STATE_VERSION = 1`) with a pending event ledger `jump_pending` (`None` or `{time, width, n_cells, cell_index, event_time, unit_mark}`).

- **Accepted-step rollback and durable restart**: `self.draw_count` increments only after all validation, SSA sampling, and mark generation succeed. Simulator rollback or checkpoint reload restores `(seed, draw_count, generator_state, wiener_pending, jump_pending)` without touching global Python, NumPy, or PyTorch RNG streams.
- **Path-consistent adaptive step refinement (`record_rejected_trial_path` and `_events_on_path`)**: When an outer adaptive controller rejects a trial step $[t_d, t_d + H_d)$ after rewinding `extra_state` to the pre-trial snapshot, calling `record_rejected_trial_path()` replays the exact trial arrivals and unit marks from the rewound generator and pins them in `_jump_pending`. On a subsequent retry of width $h_d < H_d$ starting at $t_d$ (matched within relative tolerance $10^{-9}$), `_events_on_path` retains exactly those pinned events with $\tau < h_d$ and leaves the remaining events (shifted by $-h_d$) pending at $t_d + h_d$. If $h_d > H_d$, the pinned events on $[t_d, t_d + H_d)$ are concatenated with fresh arrivals drawn on $[t_d + H_d, t_d + h_d)$. An overlapping request that does not share the pending interval start raises `RuntimeError`.

### 5.3 Comparison with Standalone `MarkovGammaPrecipitation`

| Feature | `MarkedPoissonRainfallFunctor` (`stochastic_rainfall.py`) | `MarkovGammaPrecipitation` (`weather_generator.py`) |
|---|---|---|
| Execution contract | Governed `PhysicalFieldFunctor` step node in `StateManager` / `GraphIR` | Standalone `StochasticProcess` multi-step trajectory sampler |
| Occurrence clock | Continuous-time homogeneous Poisson arrivals via `GillespieSSA` over $\Delta t_d$ | Discrete daily two-state first-order Markov wet/dry chain |
| Depth / flux law | Per-event exponential depth marks ($z_i \sim \mathrm{Exp}(\alpha)$) accumulated in $\mathrm{mm}$ | Wet-day aggregate depth $\mathrm{Gamma}(k, \theta)$ converted to $\mathrm{kg\,m^{-2}\,s^{-1}}$ |
| RNG ownership | Private `torch.Generator` + `jump_pending` serialized in `nn.Module` extra state | Scoped `torch.random.fork_rng` on CPU during `sample()` |

## 6. Conservation, Boundary Conditions, and Ledgers

`MarkedPoissonRainfallFunctor` models external atmospheric forcing entering an open domain (`conservation_laws = ["none"]`, certificate `ledger_behavior: open`, `open_system_route: port`). No closed water-mass invariant, internal Hamiltonian storage functional, entropy-production rate, or fluctuation-dissipation relation is claimed; downstream hydrological balance begins only when rainfall increments couple into soil or canopy water reservoirs.

Every call to `forward(state, time, dt)` enforces fail-closed domain checks before advancing `self.generator` or emitting `UpdateItem` records:

1. No extra keyword arguments are permitted (`TypeError` on non-empty `kwargs`).
2. Both `stochastic_rainfall_event_count` and `stochastic_rainfall_depth` must be present, floating-point `torch.Tensor` instances of identical shape containing only finite entries.
3. `stochastic_rainfall_event_count` must satisfy `torch.equal(count, torch.round(count))` and `count >= 0` elementwise.
4. `dt` must be finite and strictly positive, scalar parameters $\lambda$ and $\alpha$ must be strictly positive, and `count.device.type` must match `self.generator.device.type`.

## 7. Code Mapping

| Layer | Canonical Owner | Responsibility |
|---|---|---|
| Scientific Functor | `relic/scientific/atmosphere/stochastic_rainfall.py` (`MarkedPoissonRainfallFunctor`) | Point-model `PhysicalFieldFunctor` advancing `(N, H)` and four compensator/increment diagnostics |
| Substream & extra state | `relic/core/functor/stochastic_substream.py` (`FunctorSubstreamMixin`) + `_jump_pending` hooks | Seed validation, private `torch.Generator`, `state_dict` extra-state serialization, device guard |
| Jump clock solver | `relic/stochastic/gillespie.py` (`GillespieSSA`, `ReactionSystem`, `Reaction`) | Direct-method constant-hazard Poisson arrival times within $[0, \Delta t_d)$ |
| Workflow & GraphIR binding | `relic/stochastic/workflows.py` (`build_bound_marked_jump_graph`, `execute_bound_marked_jump`) | Compiles single-node `GraphIR` validated by `stochastic_sdae_check` (`requires_jump_reset: true`) for CLI `stochastic gillespie` and `POST /stochastic/gillespie` |
| Parameter database | `relic/parameters/database/relic/stochastic/weather_parameters.yaml` | Defines `stochastic.weather.rainfall.event_rate` and `stochastic.weather.rainfall.mean_depth` |
| IPHS certificate | `docs/source/theory/iphs_certificates/atmosphere_stochastic_rainfall_marked_poisson_rainfall_functor.yaml` | Level-2 `field` (`marked_jump`) equivalence, port, jump-reset, and reproducibility contract |

## 8. Implementation Status

| Capability | Status | Notes |
|---|---|---|
| Homogeneous Poisson arrivals via `GillespieSSA` direct method with fail-closed tail cap | `[I]` | `arrival_cap` ($\lceil\mu + 12\sqrt{\mu} + 64\rceil \le 100000$) rejects truncation |
| Exponential storm depth marks with pathwise autograd sensitivity in `mean_depth` | `[I]` | Unit marks $E_i \sim \mathrm{Exp}(1)$ scaled by tensor `mean_depth` |
| Predictable count and depth compensators as `REPLACE` diagnostics | `[I]` | Emits $\lambda\,\Delta t_d$ and $\lambda\,\alpha\,\Delta t_d$ per step |
| Replayable `nn.Module` extra state and rejected-trial jump-path restriction | `[I]` | `EXTRA_STATE_FORMAT = "relic.stochastic_rainfall.event_substream"`, `_events_on_path`, `record_rejected_trial_path` |
| Governed `GraphIR` / `stochastic_sdae_check` workflow execution | `[I]` | Bound through `build_bound_marked_jump_graph` / `execute_bound_marked_jump` |
| Inhomogeneous / state-dependent hazards $\lambda(x, t)$ or alternative mark distributions | `[N]` | Constant scalar `event_rate` and `mark_law = "exponential"` only |
| Spatial coordinate queries (`get_value_at`) or post-draw cross-device substream migration | `[N]` | Point model (`SPATIAL_DIMS = None`); `to_device` across device types requires `draw_count == 0` |

`MarkedPoissonRainfallFunctor` holds an IPHS Level-2 certificate (`canonical_role: field`, `iphs_level: 2`, `equivalence_type: exact`) in `atmosphere_stochastic_rainfall_marked_poisson_rainfall_functor.yaml`. Its frozen certificate expression literals are:

```text
dN = N(dt), N ~ Poisson(lambda dt); dH = sum_{i<=dN} z_i
Delta N = k, k ~ Poisson(lambda dt) via direct SSA; Delta H = alpha sum_{i<=k} E_i, E_i ~ Exp(1); Functor-owned event substream
```

## 9. References

- Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. *Journal of Physical Chemistry*, 81(25), 2340-2361.
- Karlin, S., and Taylor, H. M. (1981). *A Second Course in Stochastic Processes*. Academic Press.
- Laio, F., Porporato, A., Ridolfi, L., and Rodriguez-Iturbe, I. (2001). Plants in water-controlled ecosystems: active role in hydrologic processes and response to water stress — II. Probabilistic soil moisture dynamics. *Advances in Water Resources*, 24(7), 707-723.
- Rodriguez-Iturbe, I., Porporato, A., Ridolfi, L., Isham, V., and Cox, D. R. (1999). Probabilistic modelling of water balance at a point: the role of climate, soil and vegetation. *Proceedings of the Royal Society of London. Series A*, 455(1990), 3789-3805.
- *Methods of Ecological Analysis*, Chapter 27 (Sections 27.3 and 27.5): Marked-Poisson Ecohydrological Forcing and Reproducible Stochastic Substreams.
