# Pedosphere Scientific Classes

This reference covers all 24 certificate-backed scientific classes in `relic.scientific.pedosphere`: 24 executable concrete Functors, 0 fail-closed unavailable boundaries, and 0 abstract contracts. Each entry is generated from the live class, its IPHS certificate, State Contract Model, and ParameterManager metadata. Fail-closed unavailable classes are not instantiated during documentation generation.

The tables distinguish constructor controls, runtime variables, and ParameterManager scientific parameters. Values shown as package defaults belong to the documented RELIC architecture version and must not be copied as hardcoded constants into scientific code.

Certificate wildcard parameter groups (`*`, `<pft>`) are expanded to their exact current YAML entries. Every other declared parameter is matched to the parameter database by its exact full key; a declared name is never mapped to a key that merely ends with the same name.

Exact ParameterManager rows report the live YAML value, bounds, gradient policy, description, and provenance. A row whose value reads `not declared in the parameter database` is intentionally not assigned an invented default: the class metadata or certificate declares the quantity, with the meaning and units it states, but the parameter database has no key of that exact name.

IPHS levels are read from the sole certificate owner. Levels 1 to 3 describe compatibility, wrapping, or declared structure; they are not strict IPHS/GENERIC claims.

The Channel column of each runtime output contract applies `relic.core.state.types.partition_outputs`, the only runtime output channel classifier, to the declared output name. A name that the classifier reserves for diagnostics, such as `diagnostics.*`, goes to the diagnostics channel and is never committed as state. For any other name the channel is decided at run time by the write mode of the returned value: `StateWriteMode.DERIVED` routes the output to the diagnostics channel, and any other write mode sends it to the StateManager commit (a value without a write mode takes the run's semantic-mode rules or the StateManager schema). The generator does not run `forward()`, so it does not predict that write mode. The Kind column is the State Contract Model kind of the variable and does not set its channel.

(class-relic-scientific-pedosphere-carbon-pools-carbon-pools-functor-carbonpoolsfunctor)=
## CarbonPoolsFunctor

Pedosphere carbon pool observer over the complete runtime state basis.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.carbon_pools.carbon_pools_functor.CarbonPoolsFunctor` |
| Source | `relic\scientific\pedosphere\carbon_pools\carbon_pools_functor.py:26` |
| Runtime name | `carbon_pools_observer` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `observation` |
| Spatial scale | `grid_cell` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
CarbonPoolsFunctor(name: str = 'carbon_pools_observer', config: Dict[str, Any] | None = None, device: torch.device | str | int | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'carbon_pools_observer'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; str &#124; int &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `metabolic_litter_c` | state | `g C m-2` | metabolic_litter_c mapped to canonical Phase 8 cochain basis |
| `structural_litter_c` | state | `g C m-2` | structural_litter_c mapped to canonical Phase 8 cochain basis |
| `fast_som_c` | state | `g C m-2` | fast_som_c mapped to canonical Phase 8 cochain basis |
| `slow_som_c` | state | `g C m-2` | slow_som_c mapped to canonical Phase 8 cochain basis |
| `passive_som_c` | state | `g C m-2` | passive_som_c mapped to canonical Phase 8 cochain basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `diagnostics.carbon_pools_snapshot` | output port | `g C m-2` | diagnostics | O_h projection of the ordered canonical pool basis |
| `diagnostics.total_carbon_stock` | output port | `g C m-2` | diagnostics | per-cell sum of the observed canonical pool vector over the pool axis |

### Governing equations and canonical operator

**Literature or process form**

$$
C_{\mathrm{total}} = \sum_{i} C_{i}
$$

**Canonical RELIC structural form**

$z_{h} = O_{h} x_{h}$ and $C_{\mathrm{total}} = 1 ^{\mathsf{T}} z_{h}$ over the pool axis (one stock per grid cell) with no state advancement

| Entry | Meaning |
| --- | --- |
| `O_h` | parameter-free stacking projection over the declared canonical pool order |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared canonical pedosphere carbon-pool state basis |
| `effort_variable` | not applicable to a state-neutral observation projection |
| `state_advancer` | none; $z_{h} = O_{h} x_{h}$ is diagnostic only |
| `runtime_usage` | CarbonPoolsFunctor.forward stacks canonical pool values and reports their per-cell sum over the pool axis without emitting state updates. |

### Composition channels and executable PortSpec API

Certificate composition channels: `canonical carbon-pool observation port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

This class has no numeric scientific parameter keys in its certificate; it acts as a structural observer or adapter over declared state and ports.

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | diagnostics only; simulator initial-state configuration and the selected decomposition model retain state ownership |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\decomposition.md`.

### References

- Parton, W.J. et al. (1987). Soil carbon pools (CENTURY model). (FunctorMetadata, certificate)
- Coleman, K. & Jenkinson, D.S. (1996). RothC soil carbon model. (FunctorMetadata, certificate)

(class-relic-scientific-pedosphere-carbon-pools-pools-matrix-poolsmatrix)=
## PoolsMatrix

Carbon pool transfer matrix functor for soil biogeochemistry.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.carbon_pools.pools_matrix.PoolsMatrix` |
| Source | `relic\scientific\pedosphere\carbon_pools\pools_matrix.py:71` |
| Runtime name | `pools_matrix` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
PoolsMatrix(name: str = 'pools_matrix', config: Dict[str, Any] | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'pools_matrix'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `metabolic_litter_c` | state | `g C m-2` | metabolic_litter_c mapped to canonical Phase 8 cochain basis |
| `structural_litter_c` | state | `g C m-2` | structural_litter_c mapped to canonical Phase 8 cochain basis |
| `fast_som_c` | state | `g C m-2` | fast_som_c mapped to canonical Phase 8 cochain basis |
| `slow_som_c` | state | `g C m-2` | slow_som_c mapped to canonical Phase 8 cochain basis |
| `passive_som_c` | state | `g C m-2` | passive_som_c mapped to canonical Phase 8 cochain basis |
| `environmental_modifier` | forcing | `1` | required finite non-negative external rate modifier |
| `metabolic_litter_input` | flux | `g C m-2 s-1` | required open-system metabolic litter boundary flux |
| `structural_litter_input` | flux | `g C m-2 s-1` | required open-system structural litter boundary flux |
| `fast_som_input` | forcing | `g C m-2 s-1` | required open-system fast SOM boundary flux |
| `slow_som_input` | forcing | `g C m-2 s-1` | required open-system slow SOM boundary flux |
| `passive_som_input` | forcing | `g C m-2 s-1` | required open-system passive SOM boundary flux |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `metabolic_litter_c` | state | `g C m-2` | decided at run time by the returned write mode | metabolic_litter_c mapped to canonical Phase 8 cochain basis |
| `structural_litter_c` | state | `g C m-2` | decided at run time by the returned write mode | structural_litter_c mapped to canonical Phase 8 cochain basis |
| `fast_som_c` | state | `g C m-2` | decided at run time by the returned write mode | fast_som_c mapped to canonical Phase 8 cochain basis |
| `slow_som_c` | state | `g C m-2` | decided at run time by the returned write mode | slow_som_c mapped to canonical Phase 8 cochain basis |
| `passive_som_c` | state | `g C m-2` | decided at run time by the returned write mode | passive_som_c mapped to canonical Phase 8 cochain basis |
| `carbon_decomposition_rate` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | canonical gross five-pool decay rate |
| `heterotrophic_respiration` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | canonical atmosphere-facing carbon export rate |
| `diagnostics.pools_matrix.total_respiration_step` | output port | `g C m-2` | diagnostics | integrated respiration observation |
| `diagnostics.pools_matrix.total_transfer_step` | output port | `g C m-2` | diagnostics | integrated internal transfer observation |
| `diagnostics.pools_matrix.net_carbon_change_step` | output port | `g C m-2` | diagnostics | integrated net pool change observation |
| `diagnostics.pools_matrix.total_carbon_stock` | output port | `g C m-2` | diagnostics | post-step total stock observation |
| `diagnostics.pools_matrix.total_litter_carbon_stock` | output port | `g C m-2` | diagnostics | post-step litter stock observation |
| `diagnostics.pools_matrix.total_som_carbon_stock` | output port | `g C m-2` | diagnostics | post-step SOM stock observation |
| `diagnostics.pools_matrix.carbon_balance_residual` | output port | `g C m-2` | diagnostics | open carbon-ledger residual |
| `diagnostics.pools_matrix.carbon_balance_relative_residual` | output port | `dimensionless` | diagnostics | normalized open carbon-ledger residual |
| `diagnostics.pools_matrix.carbon_balance_ok` | output port | `dimensionless` | diagnostics | carbon-ledger tolerance observation |

### Governing equations and canonical operator

**Literature or process form**

$\frac{\mathrm{d}C}{\mathrm{d}t} = I + T \cdot K \cdot C - \operatorname{diag} ( K ) \cdot C$ with mass balance

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | explicit five-pool compartment incidence map plus the open heterotrophic export; this level-2 wrapper is not strict stoichiometric evidence |
| `r_h` | ParameterManager-backed gross decay, transfer, boundary-input, and respiration rate vector evaluated by the production forward path |
| `boundary_route` | five declared pool input ports supply carbon and heterotrophic_respiration exports carbon |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | empirical external environmental modifier; no Hamiltonian effort is claimed |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with the compartment incidence matrix and rate vector evaluated by advance_reaction_network |
| `runtime_usage` | PoolsMatrix.forward evaluates the five net pool tendencies and open respiration export exactly once through advance_reaction_network, then uses the StateManager-representable increment (C + Delta_C_raw) - C for both UpdateItem values and the carbon ledger. |
| `explicit_stoichiometric_usage` | The compartment incidence map is explicit level-2 runtime structure and is not evidence for Hamiltonian or GENERIC closure. |

### Composition channels and executable PortSpec API

Certificate composition channels: `metabolic_litter_input`, `structural_litter_input`, `fast_som_input`, `slow_som_input`, `passive_som_input`, `heterotrophic_respiration`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.decomposition.pool_names` | `dimensionless` | `['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']; bounds=None` | `fixed` | Names of carbon pools in the system | Source: CENTURY model; CENTURY model pool structure |
| `pedosphere.numerical.mass_balance_tolerance` | `dimensionless` | `1e-06; bounds=(1e-08, 0.01)` | `fixed` | Relative tolerance for mass balance checks | Source: numerical methods; Numerical analysis standards |
| `pedosphere.numerical.epsilon` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-08)` | `fixed` | Small value added to denominators to prevent division by zero | Source: numerical stability; Numerical stability |
| `pedosphere.numerical.max_pool_size` | `g C m-2` | `10000.0; bounds=(100.0, 100000.0)` | `fixed` | Maximum carbon pool size (sanity check) | Source: soil carbon measurements; Soil carbon measurements |
| `pedosphere.decomposition.t_met_fast` | `dimensionless` | `0.45; bounds=(0.3, 0.6)` | `learnable` | Transfer coefficient from metabolic litter to fast SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_met_slow` | `dimensionless` | `0.45; bounds=(0.3, 0.6)` | `learnable` | Transfer coefficient from metabolic litter to slow SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_str_fast` | `dimensionless` | `0.4; bounds=(0.25, 0.55)` | `learnable` | Transfer coefficient from structural litter to fast SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_str_slow` | `dimensionless` | `0.4; bounds=(0.25, 0.55)` | `learnable` | Transfer coefficient from structural litter to slow SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_fast_slow` | `dimensionless` | `0.42; bounds=(0.3, 0.6)` | `learnable` | Transfer coefficient from fast SOM to slow SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_fast_passive` | `dimensionless` | `0.03; bounds=(0.01, 0.1)` | `learnable` | Transfer coefficient from fast SOM to passive SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_slow_fast` | `dimensionless` | `0.45; bounds=(0.3, 0.6)` | `learnable` | Transfer coefficient from slow SOM to fast SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_slow_passive` | `dimensionless` | `0.03; bounds=(0.01, 0.08)` | `learnable` | Transfer coefficient from slow SOM to passive SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.carbon_pools.turnover_rate_metabolic` | `s-1` | `3.17e-07; bounds=(1e-08, 1e-05)` | `learnable` | Base turnover rate for metabolic litter (corresponds to ~10 year^-1 or 36.5 day turnover time) | Source: Converted from CENTURY yearly rate to per-second; DOI:10.2136/sssaj1987.03615995005100050015x |
| `pedosphere.carbon_pools.turnover_rate_structural` | `s-1` | `1.55e-07; bounds=(1e-08, 1e-06)` | `learnable` | Base turnover rate for structural litter (corresponds to ~4.9 year^-1 or 74.5 day turnover time) | Source: Converted from CENTURY yearly rate to per-second; DOI:10.2136/sssaj1987.03615995005100050015x |
| `pedosphere.carbon_pools.turnover_rate_fast_som` | `s-1` | `1.9e-07; bounds=(1e-08, 1e-06)` | `learnable` | Base turnover rate for fast SOM (corresponds to ~6 year^-1 or 2 month turnover time) | Source: Converted from CENTURY yearly rate to per-second; DOI:10.2136/sssaj1987.03615995005100050015x |
| `pedosphere.carbon_pools.turnover_rate_slow_som` | `s-1` | `6.34e-09; bounds=(1e-10, 1e-07)` | `learnable` | Base turnover rate for slow SOM (corresponds to ~0.2 year^-1 or 5 year turnover time) | Source: Converted from CENTURY yearly rate to per-second; DOI:10.2136/sssaj1987.03615995005100050015x |
| `pedosphere.carbon_pools.turnover_rate_passive_som` | `s-1` | `1.43e-10; bounds=(1e-12, 1e-08)` | `learnable` | Base turnover rate for passive SOM (corresponds to ~0.0045 year^-1 or 222 year turnover time) | Source: Converted from CENTURY yearly rate, consistent with radiocarbon measurements; DOI:10.1029/1999GB001232 |
| `pedosphere.numerical.default_environmental_modifier` | `dimensionless` | `1.0; bounds=(0.0, 2.0)` | `fixed` | Default environmental response modifier when no environmental limitation is present | Source: Environmental response function normalization; Standard response function normalization |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | explicit ACCUMULATE per-step pool increments and REPLACE rate or diagnostic outputs |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\decomposition.md`.

### References

- Sierra, C.A. et al. (2012). SoilR decomposition framework (FunctorMetadata, certificate)
- Coleman, K. & Jenkinson, D.S. (1996). RothC transfer matrix (FunctorMetadata, certificate)
- Parton, W.J. et al. (1987). CENTURY pool structure (FunctorMetadata, certificate)
- Luo, Y. & Weng, E. (2011). Dynamic disequilibrium (FunctorMetadata, certificate)

(class-relic-scientific-pedosphere-decomposition-first-order-firstorderdecomposition)=
## FirstOrderDecomposition

First-order kinetic decomposition functor for soil carbon pools.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.decomposition.first_order.FirstOrderDecomposition` |
| Source | `relic\scientific\pedosphere\decomposition\first_order.py:85` |
| Runtime name | `first_order_decomposition` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
FirstOrderDecomposition(name: str = 'first_order_decomposition', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'first_order_decomposition'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `metabolic_litter_c` | state | `g C m-2` | required canonical pool state; missing or non-finite values fail before calculation |
| `structural_litter_c` | state | `g C m-2` | required canonical pool state; missing or non-finite values fail before calculation |
| `fast_som_c` | state | `g C m-2` | required canonical pool state; missing or non-finite values fail before calculation |
| `slow_som_c` | state | `g C m-2` | required canonical pool state; missing or non-finite values fail before calculation |
| `passive_som_c` | state | `g C m-2` | required canonical pool state; missing or non-finite values fail before calculation |
| `soil_temperature` | forcing | `K` | required finite driver within ParameterManager biological bounds |
| `soil_moisture` | state | `m3 m-3` | required finite non-negative volumetric driver within the declared fractional bound |
| `metabolic_litter_input` | flux | `g C m-2 s-1` | required finite non-negative metabolic litter carbon boundary flux |
| `structural_litter_input` | flux | `g C m-2 s-1` | required finite non-negative structural litter carbon boundary flux |
| `lignin_n_ratio` | forcing | `1` | required finite non-negative litter-quality driver used only for decomposition inhibition |
| `clay_content` | forcing | `1` | required finite soil-texture driver within the declared fractional bound |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `metabolic_litter_c` | state | `g C m-2` | decided at run time by the returned write mode | required canonical pool state; missing or non-finite values fail before calculation |
| `structural_litter_c` | state | `g C m-2` | decided at run time by the returned write mode | required canonical pool state; missing or non-finite values fail before calculation |
| `fast_som_c` | state | `g C m-2` | decided at run time by the returned write mode | required canonical pool state; missing or non-finite values fail before calculation |
| `slow_som_c` | state | `g C m-2` | decided at run time by the returned write mode | required canonical pool state; missing or non-finite values fail before calculation |
| `passive_som_c` | state | `g C m-2` | decided at run time by the returned write mode | required canonical pool state; missing or non-finite values fail before calculation |
| `carbon_decomposition_rate` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | canonical gross internal transformation rate consumed by Mineralization and excluded from the elemental boundary ledger |
| `heterotrophic_respiration` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | canonical atmospheric carbon export rate routed through the carbon.total boundary ledger |
| `diagnostics.first_order_decomposition.metabolic_litter_decomposition_flux` | output port | `g C m-2 s-1` | diagnostics | metabolic litter gross decay observation |
| `diagnostics.first_order_decomposition.structural_litter_decomposition_flux` | output port | `g C m-2 s-1` | diagnostics | structural litter gross decay observation |
| `diagnostics.first_order_decomposition.fast_som_decomposition_flux` | output port | `g C m-2 s-1` | diagnostics | fast SOM gross decay observation |
| `diagnostics.first_order_decomposition.slow_som_decomposition_flux` | output port | `g C m-2 s-1` | diagnostics | slow SOM gross decay observation |
| `diagnostics.first_order_decomposition.passive_som_decomposition_flux` | output port | `g C m-2 s-1` | diagnostics | passive SOM gross decay observation |
| `diagnostics.first_order_decomposition.metabolic_to_fast_transfer_flux` | output port | `g C m-2 s-1` | diagnostics | metabolic-to-fast transfer observation |
| `diagnostics.first_order_decomposition.structural_to_slow_transfer_flux` | output port | `g C m-2 s-1` | diagnostics | structural-to-slow transfer observation |
| `diagnostics.first_order_decomposition.fast_to_passive_transfer_flux` | output port | `g C m-2 s-1` | diagnostics | fast-to-passive transfer observation |
| `diagnostics.first_order_decomposition.temperature_modifier` | output port | `dimensionless` | diagnostics | temperature response observation |
| `diagnostics.first_order_decomposition.moisture_modifier` | output port | `dimensionless` | diagnostics | moisture response observation |
| `diagnostics.first_order_decomposition.environmental_modifier` | output port | `dimensionless` | diagnostics | combined response observation |
| `diagnostics.first_order_decomposition.step_seconds` | output port | `s` | diagnostics | declared step length of the returned per-step proposal, read by the public ledger |

### Governing equations and canonical operator

**Literature or process form**

$\frac{\mathrm{d}C}{\mathrm{d}t} = - k ( T, \theta ) \cdot C$ with $k ( T, \theta ) = k_{\mathrm{ref}} \cdot f_{T} ( T ) \cdot f_{\theta} ( \theta )$

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | six-channel identity wrapper for five net pool tendencies and the heterotrophic export; this is level-2 wrapping, not strict stoichiometric evidence |
| `r_h` | ParameterManager-backed vector of five net pool tendencies and the heterotrophic respiration rate |
| `boundary_route` | metabolic_litter_input plus structural_litter_input is the declared carbon source and heterotrophic_respiration is the declared atmospheric export |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | empirical temperature, moisture, litter-quality, and texture drivers; no Hamiltonian effort is claimed |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with a six-channel identity wrapper evaluated by advance_reaction_network |
| `runtime_usage` | FirstOrderDecomposition.forward evaluates five net pool tendencies and heterotrophic respiration once through advance_reaction_network before emitting canonical outputs. |
| `explicit_stoichiometric_usage` | The identity $N_{h}$ wrapper is explicit level-2 runtime structure and is not evidence for strict stoichiometry or free-energy closure. |

### Composition channels and executable PortSpec API

Certificate composition channels: `metabolic_litter_input`, `structural_litter_input`, `carbon_decomposition_rate`, `heterotrophic_respiration`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.decomposition.param_25` | `dimensionless` | `1.0; bounds=(0.5, 2.0)` | `fixed` | Parameter normalization value at 25C reference temperature | Source: Temperature normalization protocol; Temperature normalization protocol |
| `pedosphere.decomposition.clay_enhancement_coefficient` | `dimensionless` | `2.5; bounds=(1.0, 5.0)` | `learnable` | Coefficient for clay enhancement of passive SOM formation, controls clay protection effect | Source: Clay protection parameterization from CENTURY model; Hassink (1997) doi:10.1016/S0038-0717(96)00217-0; Six et al. (2002) Soil Biology and Biochemistry |
| `pedosphere.decomposition.max_clay_factor` | `dimensionless` | `3.0; bounds=(1.5, 5.0)` | `fixed` | Maximum clay enhancement factor capping passive SOM formation rate | Source: CENTURY model maximum clay effect constraint; CENTURY model clay effect parameterization; Parton et al. (1987) Biogeochemistry 5:109-131 |
| `pedosphere.decomposition.clay_protection_coefficient` | `dimensionless` | `2.0; bounds=(0.5, 5.0)` | `fixed` | Coefficient governing clay-induced reduction of SOM decomposition rates. | Source: CENTURY/RothC clay protection formulation; Parton et al. (1987) Soil carbon dynamics; Coleman & Jenkinson (1996) RothC documentation |
| `pedosphere.decomposition.k_metabolic` | `yr-1` | `10.0; bounds=(5.0, 20.0)` | `learnable` | Metabolic litter decomposition rate constant | Source: RothC model; Coleman & Jenkinson (1996) RothC model |
| `pedosphere.decomposition.k_structural` | `yr-1` | `4.9; bounds=(2.0, 10.0)` | `learnable` | Structural litter decomposition rate constant | Source: CENTURY model; Parton et al. (1987) CENTURY model |
| `pedosphere.decomposition.k_fast_som` | `yr-1` | `6.0; bounds=(3.0, 12.0)` | `learnable` | Fast SOM decomposition rate constant | Source: SoilR framework; Sierra et al. (2012) SoilR framework |
| `pedosphere.decomposition.k_slow_som` | `yr-1` | `0.2; bounds=(0.1, 0.5)` | `learnable` | Slow SOM decomposition rate constant | Source: RothC model; Coleman & Jenkinson (1996) |
| `pedosphere.decomposition.k_passive_som` | `yr-1` | `0.0045; bounds=(0.001, 0.01)` | `learnable` | Passive SOM decomposition rate constant | Source: CENTURY model; Century model documentation |
| `pedosphere.decomposition.t_met_fast` | `dimensionless` | `0.45; bounds=(0.3, 0.6)` | `learnable` | Transfer coefficient from metabolic litter to fast SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_met_slow` | `dimensionless` | `0.45; bounds=(0.3, 0.6)` | `learnable` | Transfer coefficient from metabolic litter to slow SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_str_fast` | `dimensionless` | `0.4; bounds=(0.25, 0.55)` | `learnable` | Transfer coefficient from structural litter to fast SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_str_slow` | `dimensionless` | `0.4; bounds=(0.25, 0.55)` | `learnable` | Transfer coefficient from structural litter to slow SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_fast_slow` | `dimensionless` | `0.42; bounds=(0.3, 0.6)` | `learnable` | Transfer coefficient from fast SOM to slow SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_fast_passive` | `dimensionless` | `0.03; bounds=(0.01, 0.1)` | `learnable` | Transfer coefficient from fast SOM to passive SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_slow_fast` | `dimensionless` | `0.45; bounds=(0.3, 0.6)` | `learnable` | Transfer coefficient from slow SOM to fast SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.t_slow_passive` | `dimensionless` | `0.03; bounds=(0.01, 0.08)` | `learnable` | Transfer coefficient from slow SOM to passive SOM | Source: CENTURY model; CENTURY model transfer matrix |
| `pedosphere.decomposition.temp_response_type` | `dimensionless` | `'q10'; bounds=None` | `fixed` | Temperature response function type (q10 or arrhenius) | Source: Model configuration; Model choice |
| `pedosphere.physical.q10_decomposition` | `dimensionless` | `2.0; bounds=(1.5, 3.0)` | `learnable` | Temperature sensitivity coefficient for soil organic matter decomposition, doubling rate per 10K increase; also the temperature factor of the tillage CO2 pulse (oxidation of the exposed organic matter) | Source: Soil respiration literature; Lloyd & Taylor (1994) doi:10.2307/2389824; Kirschbaum (1995) doi:10.1016/0038-0717(94)00242-S |
| `pedosphere.physical.activation_energy_decomposition` | `J mol-1` | `65000.0; bounds=(50000.0, 80000.0)` | `learnable` | Arrhenius activation energy for soil organic matter decomposition | Source: Soil respiration modeling; Lloyd & Taylor (1994) doi:10.1111/j.1365-3040.1994.tb02021.x; Davidson & Janssens (2006) doi:10.1038/nature04514 |
| `pedosphere.decomposition.moisture_response_power` | `dimensionless` | `1.0; bounds=(0.5, 2.0)` | `learnable` | Power for moisture response function | Source: CENTURY model; CENTURY model |
| `pedosphere.decomposition.field_capacity` | `m3 m-3` | `0.25; bounds=(0.15, 0.35)` | `fixed` | Soil field capacity | Source: Standard soil hydraulics; Standard soil hydraulics |
| `pedosphere.decomposition.wilting_point` | `m3 m-3` | `0.1; bounds=(0.05, 0.2)` | `fixed` | Soil wilting point | Source: Standard soil hydraulics; Standard soil hydraulics |
| `pedosphere.numerical.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Minimum clamp value for numerical stability in rates and fluxes | Source: standard numerical practice; Standard numerical practice |
| `pedosphere.numerical.clamp_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Maximum clamp value for response functions and fractions | Source: standard numerical practice; Standard numerical practice |
| `pedosphere.physical.temperature_minimum_biological` | `K` | `233.15; bounds=(233.15, 273.15)` | `fixed` | Minimum temperature for biological activity in soils (-40 Celsius) | Source: Extreme soil ecology; Panikov et al. (2006) doi:10.1016/j.soilbio.2005.07.011; Mikan et al. (2002) doi:10.1016/S0038-0717(02)00084-7 |
| `pedosphere.physical.temperature_maximum_biological` | `K` | `333.15; bounds=(313.15, 343.15)` | `fixed` | Maximum temperature for biological activity in soils (60 Celsius) | Source: High temperature soil ecology; Pietikainen et al. (2005) doi:10.1016/j.soilbio.2004.11.005; Allison & Treseder (2008) doi:10.1016/j.soilbio.2007.12.020 |
| `pedosphere.numerical.temperature_response_min` | `dimensionless` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Minimum temperature response multiplier | Source: realistic temperature effects; Realistic temperature effects |
| `pedosphere.numerical.temperature_response_max` | `dimensionless` | `10.0; bounds=(5.0, 50.0)` | `fixed` | Maximum temperature response multiplier | Source: realistic temperature effects; Realistic temperature effects |
| `pedosphere.numerical.unity_multiplier` | `dimensionless` | `1.0; bounds=(1.0, 1.0)` | `fixed` | Unity multiplier for normalization operations (always equals 1.0) | Source: Mathematical constant; Standard mathematical constant |
| `pedosphere.decomposition.lignin_inhibition_coefficient` | `dimensionless` | `3.0; bounds=(1.0, 5.0)` | `learnable` | Coefficient for exponential lignin inhibition of structural litter decomposition: f = exp(-coeff * L/N); the metabolic litter pool is not lignin inhibited (Parton et al. 1987, Eq. 2-4) | Source: CENTURY model lignin inhibition; Parton et al. (1987) CENTURY model doi:10.2136/sssaj1987.03615995005100010015x; Melillo et al. (1989) Global change and litter lignin doi:10.1038/338030a0 |
| `pedosphere.decomposition.max_lignin_n_ratio` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Maximum lignin to nitrogen ratio for clamping lignin inhibition calculations to prevent numerical overflow | Source: CENTURY model numerical protocol; CENTURY model numerical stability |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | explicit UpdateItem ACCUMULATE per-step pool increments and REPLACE rate or diagnostic outputs |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\decomposition.md`.

### References

- Coleman, K. & Jenkinson, D.S. (1996). RothC-26.3 model (FunctorMetadata, certificate)
- Parton, W.J. et al. (1987). CENTURY model decomposition (FunctorMetadata, certificate)
- Sierra, C.A. et al. (2012). SoilR decomposition framework (FunctorMetadata, certificate)
- Lloyd, J. & Taylor, J.A. (1994). Temperature dependence (FunctorMetadata, certificate)

(class-relic-scientific-pedosphere-decomposition-microbial-enzyme-microbialenzymedecomposition)=
## MicrobialEnzymeDecomposition

Microbial-enzyme decomposition functor using Michaelis-Menten kinetics.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.decomposition.microbial_enzyme.MicrobialEnzymeDecomposition` |
| Source | `relic\scientific\pedosphere\decomposition\microbial_enzyme.py:75` |
| Runtime name | `microbial_enzyme_decomposition` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MicrobialEnzymeDecomposition(name: str = 'microbial_enzyme_decomposition', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'microbial_enzyme_decomposition'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `metabolic_litter_c` | state | `g C m-2` | metabolic_litter_c mapped to canonical Phase 8 cochain basis |
| `structural_litter_c` | state | `g C m-2` | structural_litter_c mapped to canonical Phase 8 cochain basis |
| `microbial_biomass_c` | state | `g C m-2` | microbial_biomass_c mapped to canonical Phase 8 cochain basis |
| `soil_temperature` | forcing | `K` | soil_temperature mapped to canonical Phase 8 cochain basis |
| `soil_moisture` | state | `m3 m-3` | soil_moisture mapped to canonical Phase 8 cochain basis |
| `available_som_c` | state | `g C m-2` | available_som_c mapped to canonical Phase 8 cochain basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `metabolic_litter_c` | state | `g C m-2` | decided at run time by the returned write mode | metabolic_litter_c mapped to canonical Phase 8 cochain basis |
| `structural_litter_c` | state | `g C m-2` | decided at run time by the returned write mode | structural_litter_c mapped to canonical Phase 8 cochain basis |
| `available_som_c` | state | `g C m-2` | decided at run time by the returned write mode | available_som_c mapped to canonical Phase 8 cochain basis |
| `microbial_biomass_c` | state | `g C m-2` | decided at run time by the returned write mode | microbial_biomass_c mapped to canonical Phase 8 cochain basis |
| `heterotrophic_respiration` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | canonical atmosphere-facing REPLACE carbon flux |
| `diagnostics.microbial_enzyme.microbial_growth_rate` | output port | `g C m-2 s-1` | diagnostics | microbial assimilation observation |
| `diagnostics.microbial_enzyme.microbial_death_rate` | output port | `g C m-2 s-1` | diagnostics | microbial turnover observation |
| `diagnostics.microbial_enzyme.microbial_respiration` | output port | `g C m-2 s-1` | diagnostics | observation only; not a state or flux writer |
| `diagnostics.microbial_enzyme.growth_respiration` | output port | `g C m-2 s-1` | diagnostics | substrate carbon not assimilated by CUE |
| `diagnostics.microbial_enzyme.maintenance_respiration` | output port | `g C m-2 s-1` | diagnostics | turnover carbon routed to respiration |
| `diagnostics.microbial_enzyme.metabolic_consumption` | output port | `g C m-2 s-1` | diagnostics | metabolic substrate uptake rate |
| `diagnostics.microbial_enzyme.structural_consumption` | output port | `g C m-2 s-1` | diagnostics | structural substrate uptake rate |
| `diagnostics.microbial_enzyme.som_consumption` | output port | `g C m-2 s-1` | diagnostics | available SOM uptake rate |
| `diagnostics.microbial_enzyme.metabolic_litter.vmax` | output port | `s-1` | diagnostics | metabolic effective Vmax observation |
| `diagnostics.microbial_enzyme.metabolic_litter.km` | output port | `g C m-2` | diagnostics | metabolic effective Km observation |
| `diagnostics.microbial_enzyme.metabolic_litter.activity` | output port | `s-1` | diagnostics | metabolic uptake per substrate stock |
| `diagnostics.microbial_enzyme.structural_litter.vmax` | output port | `s-1` | diagnostics | structural effective Vmax observation |
| `diagnostics.microbial_enzyme.structural_litter.km` | output port | `g C m-2` | diagnostics | structural effective Km observation |
| `diagnostics.microbial_enzyme.structural_litter.activity` | output port | `s-1` | diagnostics | structural uptake per substrate stock |
| `diagnostics.microbial_enzyme.som.vmax` | output port | `s-1` | diagnostics | available SOM effective Vmax observation |
| `diagnostics.microbial_enzyme.som.km` | output port | `g C m-2` | diagnostics | available SOM effective Km observation |
| `diagnostics.microbial_enzyme.som.activity` | output port | `s-1` | diagnostics | available SOM uptake per substrate stock |
| `diagnostics.microbial_enzyme.moisture_response` | output port | `dimensionless` | diagnostics | moisture response observation |
| `diagnostics.microbial_enzyme.temperature_celsius` | output port | `degC` | diagnostics | temperature observation in degC |
| `diagnostics.microbial_enzyme.microbial_biomass_content` | output port | `g C m-2` | diagnostics | post-step microbial biomass observation |
| `diagnostics.microbial_enzyme.total_substrate_consumed` | output port | `g C m-2 s-1` | diagnostics | total substrate uptake rate |
| `diagnostics.microbial_enzyme.carbon_use_efficiency_realized` | output port | `dimensionless` | diagnostics | realized assimilation fraction |
| `diagnostics.microbial_enzyme.carbon_balance_residual` | output port | `g C m-2` | diagnostics | closed per-step carbon-ledger residual |
| `diagnostics.microbial_enzyme.step_seconds` | output port | `s` | diagnostics | declared step length of the returned per-step proposal, read by the public ledger |

### Governing equations and canonical operator

**Literature or process form**

$$
V = \frac{\mathrm{MIC}\,V_{\max}(T,\theta)\,[S]}{K_m(T)+\mathrm{MIC}}
$$

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | explicit five-row four-column carbon incidence map for three uptake channels and microbial turnover |
| `r_h` | ParameterManager-backed metabolic, structural, available-SOM, and microbial-turnover rate vector |
| `C_h` | carbon ledger $[ 1 , 1 , 1 , 1 , 1 ]$ validates $C_{h} N_{h} = 0$ at runtime |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | three substrate stocks, microbial biomass, and the atmosphere-facing respiration port |
| `effort_variable` | empirical temperature, moisture, and substrate controls; no Hamiltonian effort is claimed |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with the explicit five-row four-column carbon incidence map |
| `runtime_usage` | MicrobialEnzymeDecomposition.forward evaluates three reverse Michaelis-Menten uptake rates and one microbial-turnover rate exactly once through advance_reaction_network. |
| `explicit_stoichiometric_usage` | $C_{h} = [ 1 , 1 , 1 , 1 , 1 ] \mathrm{validates} C_{h} N_{h} = 0$ before the StateManager-ready increments and respiration flux are emitted. |

### Composition channels and executable PortSpec API

Certificate composition channels: `heterotrophic_respiration`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.decomposition.microbial_enzyme.ea_vmax` | `J mol-1` | `47000.0; bounds=(30000.0, 80000.0)` | `fixed` | Activation energy controlling temperature sensitivity of maximum enzyme activity (Vmax). | Source: MIMICS parameterization; Wieder et al. (2013) doi:10.5194/gmd-6-2749-2013; Sulman et al. (2014) doi:10.1038/nclimate2436 |
| `pedosphere.decomposition.microbial_enzyme.ea_km` | `J mol-1` | `30000.0; bounds=(15000.0, 50000.0)` | `fixed` | Activation energy for half-saturation constant (Km) temperature response in microbial enzyme kinetics. | Source: Trait-based microbial models; Allison et al. (2010) Ecology 91:714-725; Schimel & Weintraub (2003) Soil Biol. Biochem. 35:549-563 |
| `pedosphere.decomposition.microbial_enzyme.moisture_optimum` | `m3 m-3` | `0.55; bounds=(0.35, 0.75)` | `fixed` | Optimal volumetric soil moisture for microbial enzyme activity under aerobic conditions. | Source: Microbial decomposition literature; Manzoni et al. (2012) Soil water controls on microbial processes; Wieder et al. (2013) MIMICS calibration dataset |
| `pedosphere.decomposition.microbial_enzyme.moisture_power` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `fixed` | Shape parameter for microbial moisture response function. | Source: Meta-analysis of soil respiration responses; Manzoni & Porporato (2009) doi:10.1111/j.1365-2486.2008.01716.x |
| `pedosphere.decomposition.microbial_enzyme.vmax_clamp_min_factor` | `dimensionless` | `0.1; bounds=(0.01, 1.0)` | `fixed` | Lower multiplier bound for temperature-adjusted Vmax in microbial enzyme kinetics. | Source: numerical stability policy; Davidson et al. (2014) Global Change Biology |
| `pedosphere.decomposition.microbial_enzyme.vmax_clamp_max_factor` | `dimensionless` | `10.0; bounds=(1.0, 50.0)` | `fixed` | Upper multiplier bound for temperature-adjusted Vmax in microbial enzyme kinetics. | Source: numerical stability policy; Davidson et al. (2014) Global Change Biology |
| `pedosphere.decomposition.microbial_enzyme.km_clamp_min_factor` | `dimensionless` | `0.5; bounds=(0.01, 1.0)` | `fixed` | Lower multiplier bound for temperature-adjusted Km in microbial enzyme kinetics. | Source: numerical stability policy; Davidson et al. (2014) Global Change Biology |
| `pedosphere.decomposition.microbial_enzyme.km_clamp_max_factor` | `dimensionless` | `5.0; bounds=(1.0, 50.0)` | `fixed` | Upper multiplier bound for temperature-adjusted Km in microbial enzyme kinetics. | Source: numerical stability policy; Davidson et al. (2014) Global Change Biology |
| `pedosphere.decomposition.microbial_enzyme.kinetics.metabolic.vmax_ref` | `s-1` | `5e-06; bounds=(1e-06, 1.5e-05)` | `learnable` | Reference maximum enzyme activity for metabolic litter substrates. | Source: MIMICS calibration; Wieder et al. (2013) MIMICS parameter table |
| `pedosphere.decomposition.microbial_enzyme.kinetics.metabolic.km_ref` | `g C m-2` | `0.01; bounds=(0.002, 0.05)` | `learnable` | Reference half-saturation constant for metabolic litter decomposition. | Source: Trait-based microbial kinetics; Allison et al. (2010) trait-based microbial kinetics |
| `pedosphere.decomposition.microbial_enzyme.kinetics.structural.vmax_ref` | `s-1` | `3e-06; bounds=(5e-07, 1e-05)` | `learnable` | Reference maximum enzyme activity for structural litter substrates. | Source: MIMICS calibration; Wieder et al. (2013) MIMICS parameter table |
| `pedosphere.decomposition.microbial_enzyme.kinetics.structural.km_ref` | `g C m-2` | `0.05; bounds=(0.01, 0.2)` | `learnable` | Reference half-saturation constant for structural litter decomposition. | Source: Trait-based microbial kinetics; Allison et al. (2010) |
| `pedosphere.decomposition.microbial_enzyme.kinetics.som.vmax_ref` | `s-1` | `2e-06; bounds=(5e-07, 8e-06)` | `learnable` | Reference maximum enzyme activity for stabilized SOM substrates. | Source: MIMICS calibration; Wieder et al. (2013) MIMICS parameter table |
| `pedosphere.decomposition.microbial_enzyme.kinetics.som.km_ref` | `g C m-2` | `0.02; bounds=(0.005, 0.08)` | `learnable` | Reference half-saturation constant for stabilized SOM decomposition. | Source: Trait-based microbial kinetics; Allison et al. (2010) |
| `pedosphere.decomposition.microbial_enzyme.microbial.carbon_use_efficiency` | `dimensionless` | `0.5; bounds=(0.2, 0.6)` | `learnable` | Fraction of consumed carbon allocated to microbial growth. | Source: Microbial growth efficiency synthesis; Sinsabaugh et al. (2013) doi:10.1016/j.soilbio.2013.04.010 |
| `pedosphere.decomposition.microbial_enzyme.microbial.turnover_rate` | `s-1` | `1e-07; bounds=(1e-08, 5e-07)` | `learnable` | Background microbial turnover rate constant. | Source: Microbial turnover calibration; Allison et al. (2010) microbial turnover analysis |
| `pedosphere.decomposition.microbial_enzyme.microbial.max_biomass` | `g C m-2` | `1.0; bounds=(0.2, 5.0)` | `fixed` | Upper bound for microbial biomass to maintain numerical stability. | Source: Numerical stability guidelines; Sulman et al. (2014) microbial trait bounds |
| `pedosphere.decomposition.microbial_enzyme.microbial.min_biomass` | `g C m-2` | `0.0001; bounds=(1e-06, 0.001)` | `fixed` | Lower bound for microbial biomass to avoid extinction due to rounding. | Source: Numerical stability guidelines; Numerical stabilization practices |
| `pedosphere.decomposition.microbial_enzyme.microbial.maintenance_rate` | `s-1` | `5e-08; bounds=(1e-08, 2e-07)` | `learnable` | Maintenance respiration rate constant for microbial biomass. | Source: Microbial maintenance respiration; Sinsabaugh et al. (2016) doi:10.1111/1365-2435.12621 |
| `pedosphere.decomposition.temp_factor` | `dimensionless` | `2.5; bounds=(1.5, 4.0)` | `learnable` | Q10 temperature factor for microbial enzyme decomposition processes | Source: Lloyd & Taylor (1994); Lloyd & Taylor (1994) temperature response |
| `pedosphere.numerical.epsilon_concentration` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-08)` | `fixed` | Epsilon for concentration calculations to prevent division by zero | Source: numerical stability for concentration; Numerical stability for concentration |
| `pedosphere.numerical.moisture_response_clamp_min` | `dimensionless` | `0.1; bounds=(0.0, 0.3)` | `fixed` | Minimum moisture response factor for environmental limitation | Source: response function limits; Response function limits |
| `pedosphere.numerical.clamp_max` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Maximum clamp value for response functions and fractions | Source: standard numerical practice; Standard numerical practice |
| `pedosphere.numerical.temp_response_clamp_min` | `dimensionless` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Minimum temperature response factor | Source: response function limits; Response function limits |
| `pedosphere.numerical.temp_response_clamp_max` | `dimensionless` | `15.0; bounds=(5.0, 50.0)` | `fixed` | Maximum temperature response factor for extreme high temperature conditions | Source: response function limits for microbial processes; Response function limits for microbial processes |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | explicit ACCUMULATE per-step pool increments and REPLACE flux or diagnostic outputs |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\decomposition.md`.

### References

- Allison, S.D. et al. (2010). Microbial abundance and composition influence (FunctorMetadata, certificate)
- Sulman, B.N. et al. (2014). Microbe-driven turnover in elevated CO2 (FunctorMetadata, certificate)

(class-relic-scientific-pedosphere-interface-mapping-litterfallinterfacemapper)=
## LitterfallInterfaceMapper

Map plant litterfall boundary fluxes into canonical pedosphere ports.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.interface_mapping.LitterfallInterfaceMapper` |
| Source | `relic\scientific\pedosphere\interface_mapping.py:595` |
| Runtime name | `litterfall_interface_mapper` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
LitterfallInterfaceMapper(name: str = 'litterfall_interface_mapper', config: Dict[str, Any] | None = None, parameter_manager: Any | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'litterfall_interface_mapper'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `parameter_manager` | <code>typing.Any &#124; None</code> | `None` | Constructor control for parameter manager; see the class runtime contract and ParameterManager table below. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `leaf_litterfall_carbon` | flux | `kg C m-2 s-1` | required phenology leaf carbon boundary flux in kg C m-2 s-1 |
| `froot_litterfall_carbon` | flux | `kg C m-2 s-1` | required phenology fine-root carbon boundary flux in kg C m-2 s-1 |
| `leaf_litterfall_nitrogen` | flux | `kg N m-2 s-1` | required phenology leaf nitrogen boundary flux in kg N m-2 s-1 |
| `froot_litterfall_nitrogen` | flux | `kg N m-2 s-1` | required phenology fine-root nitrogen boundary flux in kg N m-2 s-1 |
| `lignin_n_ratio` | forcing | `1` | required dimensionless litter-quality control |
| `dung_c_flux` | flux | `kg C m-2 s-1` | required grazing carbon return flux in kg C m-2 s-1 when grazing_management is selected |
| `dung_n_flux` | flux | `kg N m-2 s-1` | required fecal organic nitrogen return flux in kg N m-2 s-1 when grazing_management is selected |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `metabolic_litter_input` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | receives plant metabolic carbon plus the manure metabolic fraction |
| `structural_litter_input` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | receives complementary plant and manure carbon fractions |
| `organic_nitrogen_input` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | receives all converted plant litter and fecal organic nitrogen without retranslocation double counting |

### Governing equations and canonical operator

**Literature or process form**

Fresh non-woody plant litter uses a CENTURY-style quality partition; paired dung C/N uses a ParameterManager-backed conservative two-active-pool projection of the RothC FYM 49/49 active split.

**Canonical RELIC structural form**

$y_{h} = C_{h} x_{h}$ with element-preserving unit projection and $F_{\mathrm{met}}$ + $F_{\mathrm{str}} = F_{\mathrm{plant},C,g} + F_{\mathrm{dung},C,g}$, $F_{\mathrm{organic},N} = F_{\mathrm{plant},N,g} + F_{\mathrm{dung},N,g}$

| Entry | Meaning |
| --- | --- |
| `C_h` | element-preserving sum, central kg-to-g unit projection, and conservative carbon partition |
| `H_h` | no thermodynamic metric is asserted for this level 2 boundary adapter |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | four required phenology element boundary fluxes, one dimensionless litter-quality control, and a dependency-selected grazing dung C/N pair |
| `effort_variable` | no thermodynamic effort is asserted for this level 2 port adapter |
| `state_advancer` | no stock is advanced; the runtime publishes three instantaneous target-side boundary ports |
| `runtime_usage` | LitterfallInterfaceMapper.forward requires finite non-negative tensors, makes the dung pair mandatory only when grazing_management is declared, rejects undeclared dung ports, delegates kg-to-g conversion to UnitConverter, and returns exactly three explicit REPLACE UpdateItems. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phenology_litterfall_boundary`, `grazing_dung_boundary`, `pedosphere_litter_carbon_input`, `pedosphere_litter_nitrogen_input`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.decomposition.lignin_n_threshold` | `dimensionless` | `0.45; bounds=(0.3, 0.7)` | `learnable` | Lignin/N ratio threshold for litter partitioning | Source: CENTURY model; CENTURY model |
| `pedosphere.decomposition.lignin_partitioning_coefficient` | `dimensionless` | `3.0; bounds=(1.0, 5.0)` | `learnable` | Exponential coefficient for partitioning fresh litter between metabolic and structural pools based on lignin/N ratio | Source: CENTURY model litter partitioning algorithm; Parton et al. (1987) CENTURY model doi:10.2136/sssaj1987.03615995005100010015x; Metherell et al. (1993) CENTURY documentation |
| `pedosphere.decomposition.min_metabolic_fraction` | `dimensionless` | `0.1; bounds=(0.05, 0.3)` | `fixed` | Minimum fraction of fresh litter that enters metabolic pool regardless of litter quality (prevents zero metabolic input) | Source: CENTURY model partitioning constraints; CENTURY model litter partitioning algorithm |
| `pedosphere.decomposition.max_metabolic_fraction` | `dimensionless` | `0.9; bounds=(0.7, 0.95)` | `fixed` | Maximum fraction of fresh litter that can enter metabolic pool even with high quality litter (ensures some structural input) | Source: CENTURY model partitioning constraints; CENTURY model litter partitioning algorithm |
| `pedosphere.decomposition.manure_metabolic_fraction` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Conservative two-litter-pool projection fraction assigning fresh manure carbon to the metabolic litter input | Source: RothC FYM active-pool split normalized for the RELIC two-litter-pool boundary projection; Coleman and Jenkinson (1996) RothC-26.3 model guide; FYM carbon is split 49 percent DPM, 49 percent RPM, and 2 percent HUM |
| `pedosphere.numerical.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Minimum clamp value for numerical stability in rates and fluxes | Source: standard numerical practice; Standard numerical practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | all three instantaneous target ports use UpdateItem REPLACE with $\mathrm{per}_{\mathrm{timestep}} = \mathrm{false}$ |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\pedosphere_transport.md`.

### References

- Parton et al. (1987) Soil Science Society of America Journal 51, 1173-1179 (FunctorMetadata, certificate)
- Metherell et al. (1993) CENTURY Soil Organic Matter Model documentation (FunctorMetadata, certificate)
- Thornton et al. (2002) Agricultural and Forest Meteorology 113, 185-222 (FunctorMetadata, certificate)
- Coleman and Jenkinson (1996) RothC-26.3 model guide (FunctorMetadata, certificate)
- Pielke, R.A. (2002). Mesoscale meteorological modeling. (FunctorMetadata)
- Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming. (FunctorMetadata)

(class-relic-scientific-pedosphere-interface-mapping-pedosphereinterfacemapper)=
## PedosphereInterfaceMapper

Maps between pedosphere internal names and StateManager/external module names.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.interface_mapping.PedosphereInterfaceMapper` |
| Source | `relic\scientific\pedosphere\interface_mapping.py:39` |
| Runtime name | `pedosphere_interface_mapper` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
PedosphereInterfaceMapper(name: str = 'pedosphere_interface_mapper', **kwargs)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'pedosphere_interface_mapper'` | Functor identity used by registration, provenance, and dependency resolution. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `drainage_flux` | flux | `kg m-2 s-1` | drainage_flux mapped to canonical Phase 8 cochain basis |
| `subsurface_runoff_flux` | flux | `kg m-2 s-1` | subsurface_runoff_flux mapped to canonical Phase 8 cochain basis |
| `surface_runoff_flux` | flux | `kg m-2 s-1` | surface_runoff_flux mapped to canonical Phase 8 cochain basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `leaching_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `lateral_flow` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `surface_loss` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |

### Governing equations and canonical operator

**Literature or process form**

$y_{h} = C_{h} x_{h}$ with $C_{h} = I_{\mathrm{3}}$ for unit-preserving hydrological flux ports

**Canonical RELIC structural form**

$y_{h} = C_{h} x_{h}$ through the declared chain map between scales

| Entry | Meaning |
| --- | --- |
| `C_h` | declared chain map |
| `H_h` | positive coupling metric or normalization map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | three hydrosphere water-flux ports ordered as drainage, subsurface runoff, and surface runoff |
| `effort_variable` | no thermodynamic effort is asserted for this level 2 unit-preserving port adapter |
| `state_advancer` | $y_{h} = C_{h} x_{h}$ with $C_{h} = I_{\mathrm{3}}$; no state integration occurs |
| `runtime_usage` | PedosphereInterfaceMapper.forward validates all three required ports and returns exactly three REPLACE UpdateItems without dynamic keys, defaults, or state-relative increments. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.nutrient_cycling.n_noy_fraction` | `dimensionless` | `0.6; bounds=(0.3, 0.8)` | `fixed` | Fraction of total inorganic nitrogen in oxidized forms (NOy - NO3-, NO2-) | Source: Literature review and DNDC model; Davidson et al. (2000). Nitrogen gas partitioning in soils; DNDC model nitrogen speciation |
| `pedosphere.nutrient_cycling.n_nhx_fraction` | `dimensionless` | `0.4; bounds=(0.2, 0.7)` | `fixed` | Fraction of total inorganic nitrogen in reduced forms (NHx - NH4+, NH3) | Source: Literature review and DNDC model; Davidson et al. (2000). Nitrogen gas partitioning in soils; DNDC model nitrogen speciation |
| `pedosphere.phosphorus.p_po4_fraction` | `dimensionless` | `0.3; bounds=(0.1, 0.6)` | `fixed` | Fraction of total inorganic phosphorus as orthophosphate (PO4 3-) in soil solution. | Source: Literature review and soil P fractionation studies; Wang et al. (2010). Soil phosphorus fractionation and bioavailability; DOI: 10.1016/j.geoderma.2009.12.003 |
| `pedosphere.phosphorus.p_dust_fraction` | `dimensionless` | `0.7; bounds=(0.4, 0.9)` | `fixed` | Fraction of total inorganic phosphorus bound to mineral/dust particles (sorbed plus occluded). | Source: Hedley P fractionation protocol; Wang et al. (2010). Soil phosphorus fractionation and bioavailability; Hedley fractionation scheme |
| `pedosphere.transport.nitrate_distribution_coefficient` | `L kg-1` | `0.1; bounds=(0.0, 1.0)` | `learnable` | Distribution coefficient (Kd) for nitrate sorption on soil particles, very mobile anion with minimal sorption | Source: Solute transport literature; Jury & Horton (2004) Soil Physics 6th edition; van Genuchten & Wierenga (1986) doi:10.2136/sssaj1986.03615995005000050013x |
| `pedosphere.transport.doc_distribution_coefficient` | `L kg-1` | `2.0; bounds=(0.5, 10.0)` | `learnable` | Distribution coefficient (Kd) for dissolved organic carbon sorption, moderate sorption to soil minerals | Source: Organic carbon transport studies; Kalbitz et al. (2000) doi:10.1016/S0016-7061(00)00027-2; Kaiser & Guggenberger (2000) doi:10.1046/j.1365-2389.2000.00289.x |
| `pedosphere.numerical.epsilon_concentration` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-08)` | `fixed` | Epsilon for concentration calculations to prevent division by zero | Source: numerical stability for concentration; Numerical stability for concentration |
| `pedosphere.numerical.mass_balance_tolerance` | `dimensionless` | `1e-06; bounds=(1e-08, 0.01)` | `fixed` | Relative tolerance for mass balance checks | Source: numerical methods; Numerical analysis standards |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | all three instantaneous water-flux outputs use UpdateItem REPLACE with $\mathrm{per}_{\mathrm{timestep}} = \mathrm{false}$ |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\pedosphere_transport.md`.

### References

- Clark et al. (2011) doi:10.5194/gmd-4-677-2011 (FunctorMetadata, certificate)
- Clark et al. (2011) doi:10.5194/gmd-4-701-2011 (FunctorMetadata, certificate)
- Pielke, R.A. (2002). Mesoscale meteorological modeling. (FunctorMetadata)
- Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming. (FunctorMetadata)

(class-relic-scientific-pedosphere-methane-budget-coupler-methanebudgetcoupler)=
## MethaneBudgetCoupler

Aggregate methane production, oxidation, and transport into a conserved budget.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.methane.budget_coupler.MethaneBudgetCoupler` |
| Source | `relic\scientific\pedosphere\methane\budget_coupler.py:26` |
| Runtime name | `pedosphere_methane_budget` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MethaneBudgetCoupler(name: str = 'pedosphere_methane_budget', device: torch.device | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'pedosphere_methane_budget'` | Functor identity used by registration, provenance, and dependency resolution. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `ch4_production_rate` | flux | `g C m-2 s-1` | non-negative methane-carbon production rate P [g C m-2 s-1] |
| `ch4_oxidation_rate` | flux | `g C m-2 s-1` | non-negative methane-carbon oxidation export O [g C m-2 s-1] |
| `ch4_transport_loss_rate` | flux | `g C m-2 s-1` | signed requested atmospheric transport loss F_transport [g C m-2 s-1] |
| `ch4_storage_tendency_prod` | flux | `g C m-2 s-1` | production contribution to soil methane storage [g C m-2 s-1] |
| `ch4_storage_tendency_oxid` | flux | `g C m-2 s-1` | oxidation contribution to soil methane storage [g C m-2 s-1] |
| `ch4_storage_tendency_trans` | flux | `g C m-2 s-1` | requested transport contribution to soil methane storage [g C m-2 s-1] |
| `ch4_soil_storage` | state | `g C m-2` | non-negative soil methane-carbon storage M [g C m-2] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `ch4_atmospheric_flux` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | realized signed atmospheric methane-carbon boundary flux [g C m-2 s-1] |
| `ch4_soil_storage` | state | `g C m-2` | decided at run time by the returned write mode | non-negative soil methane-carbon storage M [g C m-2] |
| `diagnostics.methane_budget.atmospheric_ch4_mass_flux` | output port | `g CH4 m-2 s-1` | diagnostics | realized atmospheric methane mass flux [g CH4 m-2 s-1] |
| `diagnostics.methane_budget.mass_balance_residual` | output port | `g C m-2 s-1` | diagnostics | signed local methane-carbon budget residual [g C m-2 s-1] |
| `diagnostics.methane_budget.storage_tendency_net` | output port | `g C m-2 s-1` | diagnostics | realized storage tendency dM/dt [g C m-2 s-1] |
| `diagnostics.methane_budget.transport_limiter_residual` | output port | `g C m-2 s-1` | diagnostics | requested transport denied by shared storage availability [g C m-2 s-1] |

### Governing equations and canonical operator

**Literature or process form**

Yue et al. (2024) Eq. 33 gives $\frac{\mathrm{d}M}{\mathrm{d}t} = P - O - E - D - T$, with atmospheric emission equal to the sum of top-layer transport mechanisms.

**Canonical RELIC structural form**

$$
\mathrm{d}M_{\mathrm{raw}} = \mathrm{d}t \cdot ( T_{\mathrm{prod}} + T_{\mathrm{oxid}} + T_{\mathrm{trans}} ); \mathrm{d}M = \max ( \mathrm{d}M_{\mathrm{raw}} , - M ); R_{\mathrm{limit}} = ( \mathrm{d}M - \mathrm{d}M_{\mathrm{raw}} ) / \mathrm{d}t; F_{\mathrm{atm}} = F_{\mathrm{transport}} - R_{\mathrm{limit}}
$$

| Entry | Meaning |
| --- | --- |
| `port_aggregation` | one discrete aggregation of the three declared storage tendencies |
| `storage_limiter` | $\mathrm{d}M = \max ( \mathrm{d}t \cdot ( T_{\mathrm{prod}} + T_{\mathrm{oxid}} + T_{\mathrm{trans}} ) , - M )$ |
| `boundary_route` | $F_{\mathrm{atm}} = F_{\mathrm{transport}} - R_{\mathrm{limit}}$ |
| `strict_structure_boundary` | No H, effort, J, M, port-power, or GENERIC operator is claimed. |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | extensive soil methane-carbon storage and signed atmospheric boundary flux |
| `effort_variable` | No thermodynamic effort is asserted; the module consumes process rates and tendencies from declared scheduler dependencies. |
| `state_advancer` | one bounded explicit storage increment dM and one realized atmospheric boundary flux $F_{\mathrm{atm}}$ |
| `runtime_usage` | MethaneBudgetCoupler.forward aggregates the declared tendencies once and returns the storage delta through UpdateItem ACCUMULATE with per_timestep true. |

### Composition channels and executable PortSpec API

Certificate composition channels: `methane_atmospheric_boundary_port`, `methane_soil_storage_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.methane.conservation.relative_tolerance` | `dimensionless` | `1e-08; bounds=(1e-10, 1e-06)` | `fixed` | Relative tolerance for methane mass balance conservation | Source: model; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 |
| `pedosphere.methane.conservation.absolute_tolerance` | `g C m-2 s-1` | `1e-07; bounds=(1e-09, 1e-05)` | `fixed` | Absolute tolerance for methane carbon mass balance in runtime flux units | Source: model; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 |
| `pedosphere.numerical.minimum_timestep` | `s` | `1e-06; bounds=(1e-09, 1.0)` | `fixed` | Minimum timestep used in pedosphere denominators | Source: numerical stability; Numerical stability |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | explicit REPLACE boundary and diagnostic rates plus one ACCUMULATE per_timestep true storage increment |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\wetland_methane.md`.

### References

- Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 (FunctorMetadata, certificate)
- Walter et al. (2001) doi:10.1029/2001JD900165 (FunctorMetadata, certificate)
- Pielke, R.A. (2002). Mesoscale meteorological modeling. (FunctorMetadata)
- Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming. (FunctorMetadata)

(class-relic-scientific-pedosphere-methane-oxidation-methaneoxidationfunctor)=
## MethaneOxidationFunctor

Consume soil methane through methanotrophic oxidation.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.methane.oxidation.MethaneOxidationFunctor` |
| Source | `relic\scientific\pedosphere\methane\oxidation.py:24` |
| Runtime name | `pedosphere_methane_oxidation` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MethaneOxidationFunctor(name: str = 'pedosphere_methane_oxidation', device: torch.device | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'pedosphere_methane_oxidation'` | Functor identity used by registration, provenance, and dependency resolution. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `ch4_soil_storage` | state | `g C m-2` | required methane-carbon storage input M [g C m-2] |
| `soil_temperature` | forcing | `K` | Required or optional value consumed by forward(). |
| `soil_o2_concentration` | forcing | `mol m-3` | O2 concentration per water volume C_O2 [mol m-3] |
| `porosity` | forcing | `m3 m-3` | soil porosity phi [-] defining the water volume of the saturated layer |
| `water_filled_pore_space` | forcing | `1` | water-filled pore-space fraction WFPS [-] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `ch4_oxidation_rate` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | positive oxidation export mapped by r_h [g C m-2 s-1] |
| `ch4_storage_tendency_oxid` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | equal negative storage tendency mapped by -r_h [g C m-2 s-1] |
| `diagnostics.methane_oxidation.kinetic_rate` | output port | `g C m-2 s-1` | diagnostics | CLM4Me oxidation demand before the storage limit [g C m-2 s-1] |
| `diagnostics.methane_oxidation.storage_limiter_residual` | output port | `g C m-2 s-1` | diagnostics | unmet oxidation demand removed by the storage limit [g C m-2 s-1] |

### Governing equations and canonical operator

**Literature or process form**

$R = R_{o,\mathrm{max}} C_{\mathrm{CH4}} / ( K_{\mathrm{CH4}} + C_{\mathrm{CH4}} ) C_{\mathrm{O2}} / ( K_{\mathrm{O2}} + C_{\mathrm{O2}} ) \mathrm{Q}_{10} ^ ( ( \mathrm{Ts} - T_{\mathrm{base}} ) / 10 ) F$ following Riley et al. (2011) Eq. 7 and Table 1 and Oleson et al. (2013) CLM4.5 Technical Note Eq. 19.11 and Table 19.1 ($T_{\mathrm{base}} = 12 C )$, with $\mathrm{Q}_{10} = 1.9$ from Riley et al. (2011) Sect. 2.7, concentrations per water volume (Oleson et al. (2013) Eq. 19.1), $F = 1 \mathrm{in} \mathrm{the} \mathrm{saturated} \mathrm{layer}$, and demand scaling to the available storage (Riley et al. (2011) Sect. 2.1.7).

**Canonical RELIC structural form**

$\dot{x} = N_{h} r_{h}$ with $N_{h} = [ [ 1 ] , [ - 1 ] ]$ and $r_{h}$ equal to the storage-limited methane oxidation rate

| Entry | Meaning |
| --- | --- |
| `N_h` | explicit $\mathrm{stoichiometric}_{N} = [ [ 1 ] , [ - 1 ] ]$ mapping $r_{h}$ to the oxidation export and negative storage tendency |
| `r_h` | ParameterManager-backed nonnegative methane oxidation rate evaluated by the canonical forward path |
| `strict_structure_boundary` | No independent H, effort, J, M, port-power, or GENERIC operator is claimed |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | methane oxidation export and its equal negative soil-storage tendency on the CH4-C basis |
| `effort_variable` | No thermodynamic effort is asserted; temperature, methane and oxygen concentrations are kinetic rate drivers. |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with $N_{h} = [ [ 1 ] , [ - 1 ] ]$ and a single storage-limited oxidation rate $r_{h}$ |
| `runtime_usage` | MethaneOxidationFunctor.forward calls advance_reaction_network with declared stoichiometric_N and rates before emitting the four canonical outputs. |
| `explicit_stoichiometric_usage` | MethaneOxidationFunctor.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `methane_oxidation_open_matter_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.methane.oxidation.maximum_rate` | `mol m-3 s-1` | `1.25e-05; bounds=(1.25e-06, 0.000125)` | `learnable` | Maximum methane oxidation rate R_o,max per soil volume for wetlands (baseline; the bounds are the Riley et al. (2011) sensitivity range) | Source: literature; Riley et al. (2011) doi:10.5194/bg-8-1925-2011; Oleson et al. (2013) CLM4.5 Technical Note NCAR/TN-503+STR |
| `pedosphere.methane.oxidation.half_saturation_ch4` | `mol m-3` | `0.005; bounds=(0.0005, 0.05)` | `learnable` | CH4 half-saturation concentration K_CH4 per water volume for methane oxidation in wetlands (the bounds are the Riley et al. (2011) sensitivity range) | Source: literature; Riley et al. (2011) doi:10.5194/bg-8-1925-2011; Oleson et al. (2013) CLM4.5 Technical Note NCAR/TN-503+STR |
| `pedosphere.methane.oxidation.half_saturation_o2` | `mol m-3` | `0.02; bounds=(0.002, 0.2)` | `fixed` | O2 half-saturation concentration K_O2 per water volume for methane oxidation (the bounds are the Riley et al. (2011) sensitivity range) | Source: literature; Riley et al. (2011) doi:10.5194/bg-8-1925-2011; Oleson et al. (2013) CLM4.5 Technical Note NCAR/TN-503+STR |
| `pedosphere.methane.oxidation.q10` | `dimensionless` | `1.9; bounds=(1.5, 3.0)` | `fixed` | Q10 temperature coefficient for methane oxidation (baseline; the bounds span the Riley et al. (2011) sensitivity values 1.5 to 3) | Source: literature; Riley et al. (2011) doi:10.5194/bg-8-1925-2011 |
| `pedosphere.methane.oxidation.temperature_base` | `K` | `285.15; bounds=(285.15, 285.15)` | `fixed` | Base temperature of the methane oxidation Q10 factor (12 C) | Source: literature; Oleson et al. (2013) CLM4.5 Technical Note NCAR/TN-503+STR; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 |
| `pedosphere.methane.transport.layer_mixing_depth` | `m` | `0.05; bounds=(0.01, 0.2)` | `fixed` | Calibratable representative mixing depth for methane transport calculations | Source: literature; Walter and Heimann (2000) doi:10.1029/1999GB001204; Zhuang et al. (2004) doi:10.1029/2004GB002239 |
| `pedosphere.numerical.minimum_timestep` | `s` | `1e-06; bounds=(1e-09, 1.0)` | `fixed` | Minimum timestep used in pedosphere denominators | Source: numerical stability; Numerical stability |
| `pedosphere.numerical.epsilon_length` | `m` | `1e-06; bounds=(1e-09, 0.01)` | `fixed` | Minimum length scale to avoid zero layer thickness | Source: numerical stability; Numerical stability |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | explicit UpdateItem REPLACE rate ports with per_timestep false, canonical units, and source_functor provenance |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\wetland_methane.md`.

### References

- Riley et al. (2011) doi:10.5194/bg-8-1925-2011 (FunctorMetadata, certificate)
- Oleson et al. (2013) CLM4.5 Technical Note NCAR/TN-503+STR (FunctorMetadata, certificate)
- Walter and Heimann (2000) doi:10.1029/1999GB001204 (FunctorMetadata, certificate)

(class-relic-scientific-pedosphere-methane-production-methaneproductionfunctor)=
## MethaneProductionFunctor

Compute wetland methane production from an iMAPLE branch and a calibrated branch.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.methane.production.MethaneProductionFunctor` |
| Source | `relic\scientific\pedosphere\methane\production.py:23` |
| Runtime name | `pedosphere_methane_production` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MethaneProductionFunctor(name: str = 'pedosphere_methane_production', device: torch.device | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'pedosphere_methane_production'` | Functor identity used by registration, provenance, and dependency resolution. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_temperature` | forcing | `K` | soil_temperature mapped to canonical Phase 8 cochain basis |
| `soil_ph` | forcing | `1` | soil_ph mapped to canonical Phase 8 cochain basis |
| `heterotrophic_respiration` | flux | `g C m-2 s-1` | heterotrophic_respiration mapped to canonical Phase 8 cochain basis |
| `redox_potential` | forcing | `mV` | redox_potential mapped to canonical Phase 8 cochain basis |
| `land_surface.wetland_fraction` | forcing | `1` | land_surface.wetland_fraction mapped to canonical Phase 8 cochain basis |
| `microbial_activity_index` | forcing | `1` | microbial_activity_index in [0, 1] mapped to the calibrated branch activity factor |
| `methane_gating_weight` | diagnostic | `dimensionless` | methane_gating_weight mapped to the required scale-coupling control port |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `ch4_production_rate` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `ch4_storage_tendency_prod` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.methane_production.imaple_rate` | output port | `g C m-2 s-1` | diagnostics | canonical namespaced diagnostic for the iMAPLE branch rate |
| `diagnostics.methane_production.calibrated_rate` | output port | `g C m-2 s-1` | diagnostics | canonical namespaced diagnostic for the calibrated Rh-based branch rate |

### Governing equations and canonical operator

**Literature or process form**

$P_{\mathrm{iMAPLE}} = \mathrm{Rh} r f_{\mathrm{ST}} f_{\mathrm{pH}} f_{\mathrm{wet}} ( \mathrm{Yue} \mathrm{et} \mathrm{al} . ( 2024 ) \mathrm{Eq} . 34 )$ with $f_{\mathrm{ST}}$ from Yue et al. (2024) Eqs. 35-38 (Zhu et al. (2014) Eqs. 3-5 with the temperature-dependent Q10 of Yue et al. (2024) Eq. 39, $\mathrm{Q}_{10} = \operatorname{clip} ( r_{b} Q_{b} ^ ( ( \mathrm{Ts} - T_{\mathrm{base}} ) / 10 ) , 1.7 , 16 ) , r_{b} = 3.0 , Q_{b} = 1.33 , T_{\mathrm{base}} = 25 C ; \mathrm{the} \operatorname{clip} \mathrm{is} \mathrm{the} \mathrm{observed} \mathrm{range} \mathrm{of} \mathrm{Zhu} \mathrm{et} \mathrm{al} . ( 2014 ) \mathrm{applied} \mathrm{in} \mathrm{the} \mathrm{released} \mathrm{iMAPLE} \mathrm{code} )$, $f_{\mathrm{pH}}$ from Zhu et al. (2014) Eq. 6, and $f_{\mathrm{wet}}$ the continuous inundation fraction; no closure factor multiplies Eq. 34. The calibrated branch Rh r $f_{T}$ $f_{\mathrm{Eh}}$ A (the same Zhu et al. (2014) peak $f_{T}$, and the DNDC95 microbial activity index A in [0, 1]) and the blend weight are RELIC conventions with DNDC-type temperature and redox controls, not the DNDC substrate-kinetic model (Fumoto et al. 2008; van Bodegom and Scholten 2001), which is not implemented; its redox factor $f_{\mathrm{Eh}}$ is the piecewise-linear form of Zhu et al. (2014) Sect. 2.2.1 after Zhang et al. (2002).

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | MethaneProductionFunctor.forward calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical outputs. |
| `explicit_stoichiometric_usage` | MethaneProductionFunctor.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.methane.production.release_ratio` | `dimensionless` | `0.2; bounds=(0.1, 0.5)` | `learnable` | Ratio of methane to carbon dioxide released during heterotrophic respiration | Source: literature; Wania et al. (2010) doi:10.5194/gmd-3-565-2010; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014 |
| `pedosphere.methane.production.q10` | `dimensionless` | `3.0; bounds=(2.0, 5.0)` | `learnable` | Q10 of methane production at the base temperature T_base, r_b of Q10(Ts) = clip(r_b Q_b^((Ts - T_base)/10), Q10_min, Q10_max) (Yue et al. 2024 Eq. 39); the base Q10 3.0 of Zhu et al. (2014). | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014; Zhang et al. (2002) doi:10.1029/2001GB001838 |
| `pedosphere.methane.production.q10_temperature_sensitivity` | `dimensionless` | `1.33; bounds=(1.0, inf)` | `fixed` | Q_b of Yue et al. (2024) Eq. 39, the factor by which the methane production Q10 rises per 10 K above T_base. Q_b >= 1 keeps Q10 non-decreasing with temperature; no upper limit is fixed (inf). | Source: literature; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024; Paudel et al. (2016) doi:10.1088/1748-9326/11/3/034020 |
| `pedosphere.methane.production.q10_base_temperature` | `K` | `298.15; bounds=(0.0, inf)` | `fixed` | T_base of Yue et al. (2024) Eq. 39, the soil temperature at which the methane production Q10 equals r_b (25 C). An absolute temperature in K cannot be negative; the lower bound is 0 K. | Source: literature; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024; iMAPLE v1 source, LSM/src/model/iMAPLE/ch4flux_wetland.f (Yue et al. 2024 code release, doi:10.6084/m9.figshare.23593578.v1), function Q10fn_pro and subroutine CH4prod |
| `pedosphere.methane.production.q10_min` | `dimensionless` | `1.7; bounds=(1.0, inf)` | `fixed` | Lower clip Q10_min of the temperature-dependent methane production Q10; it must exceed 1 (ln(Q10) > 0 in the Zhu et al. 2014 exponent), which the functor checks at construction. | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014; Dunfield et al. (1993) Soil Biol. Biochem. 25, 321-326, doi:10.1016/0038-0717(93)90130-4 (production Q10 5.3-16) |
| `pedosphere.methane.production.q10_max` | `dimensionless` | `16.0; bounds=(1.0, inf)` | `fixed` | Upper clip Q10_max of the temperature-dependent methane production Q10 (must exceed q10_min, checked at construction). | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014; Dunfield et al. (1993) Soil Biol. Biochem. 25, 321-326, doi:10.1016/0038-0717(93)90130-4 (production Q10 5.3-16) |
| `pedosphere.methane.production.temperature_min` | `K` | `273.15; bounds=(263.15, 283.15)` | `fixed` | Minimum soil temperature for active methanogenesis | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014 |
| `pedosphere.methane.production.temperature_optimum` | `K` | `298.15; bounds=(293.15, 313.15)` | `learnable` | Optimal soil temperature for methane production | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014 |
| `pedosphere.methane.production.temperature_max` | `K` | `318.15; bounds=(313.15, 328.15)` | `fixed` | Upper soil temperature threshold for methane production | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014 |
| `pedosphere.methane.production.ph_optimum` | `dimensionless` | `7.0; bounds=(6.0, 8.0)` | `learnable` | Optimal soil pH for methanogenesis | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014 |
| `pedosphere.methane.production.ph_lower_limit` | `dimensionless` | `4.0; bounds=(4.0, 4.0)` | `fixed` | Lower soil pH limit for methanogenesis | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014, Eq. 6 |
| `pedosphere.methane.production.ph_upper_limit` | `dimensionless` | `9.0; bounds=(9.0, 9.0)` | `fixed` | Upper soil pH limit for methanogenesis | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014, Eq. 6 |
| `pedosphere.methane.production.calibrated.eh_full_activity` | `mV` | `-200.0; bounds=(-200.0, -200.0)` | `fixed` | Redox potential at and below which the calibrated methane production redox factor is 1 | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014; Zhang et al. (2002) doi:10.1029/2001GB001838 |
| `pedosphere.methane.production.calibrated.eh_zero_activity` | `mV` | `-100.0; bounds=(-100.0, -100.0)` | `fixed` | Redox potential at and above which the calibrated methane production redox factor is 0 | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014; Zhang et al. (2002) doi:10.1029/2001GB001838 |
| `pedosphere.numerical.minimum_temperature_span` | `K` | `1e-06; bounds=(1e-08, 1.0)` | `fixed` | Minimum temperature span used in response denominators | Source: numerical stability; Numerical stability |
| `pedosphere.numerical.minimum_dimensionless` | `dimensionless` | `1e-06; bounds=(1e-09, 1.0)` | `fixed` | Minimum dimensionless value used in denominators | Source: numerical stability; Numerical stability |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | explicit UpdateItem REPLACE rate ports with per_timestep false, canonical units, and source_functor provenance |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\wetland_methane.md`.

### References

- Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 (FunctorMetadata, certificate)
- Zhu et al. (2014) doi:10.5194/gmd-7-981-2014 (FunctorMetadata, certificate)
- Wania et al. (2010) doi:10.5194/gmd-3-565-2010 (FunctorMetadata, certificate)
- Zhang et al. (2002) doi:10.1029/2001GB001838 (FunctorMetadata, certificate)

(class-relic-scientific-pedosphere-methane-scales-methanescalecouplingfunctor)=
## MethaneScaleCouplingFunctor

Compute the hybrid weight blending the iMAPLE and calibrated methane branches.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.methane.scales.MethaneScaleCouplingFunctor` |
| Source | `relic\scientific\pedosphere\methane\scales.py:22` |
| Runtime name | `pedosphere_methane_scale` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `observation` |
| Spatial scale | `grid_cell` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MethaneScaleCouplingFunctor(name: str = 'pedosphere_methane_scale', device: torch.device | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'pedosphere_methane_scale'` | Functor identity used by registration, provenance, and dependency resolution. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `land_surface.wetland_fraction` | forcing | `1` | land_surface.wetland_fraction mapped to canonical Phase 8 cochain basis |
| `water_filled_pore_space` | forcing | `1` | water_filled_pore_space mapped to canonical Phase 8 cochain basis |
| `methane_data_score` | forcing | `1` | methane_data_score mapped to canonical Phase 8 cochain basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `methane_gating_weight` | diagnostic | `dimensionless` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `methane_gating_score` | diagnostic | `dimensionless` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |

### Governing equations and canonical operator

**Literature or process form**

No cited source defines the implemented weighted threshold equation; Yue et al. (2024) supports only the iMAPLE endpoint, and the calibrated Rh-based endpoint is a RELIC convention, not the DNDC substrate model.

**Canonical RELIC structural form**

$y_{h} = O_{h} x_{h}$ through the declared observation operator

| Entry | Meaning |
| --- | --- |
| `O_h` | ParameterManager-backed phenomenological observation map; no strict structural operator is claimed |
| `H_h` | not claimed for this level 1 diagnostic |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $y_{h} = O_{h} x_{h}$ through the declared observation operator |
| `runtime_usage` | MethaneScaleCouplingFunctor.forward applies the canonical observation or diagnostic operator and emits validation diagnostics through the production output path. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.methane.scales.gating.alpha_wetland` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `learnable` | Weight of wetland fraction in hybrid mode gating | Source: model; Hybrid methane scheme design |
| `pedosphere.methane.scales.gating.alpha_wfps` | `dimensionless` | `0.3; bounds=(0.0, 1.0)` | `learnable` | Weight of water-filled pore space in hybrid mode gating | Source: model; Hybrid methane scheme design |
| `pedosphere.methane.scales.gating.alpha_data_score` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Weight of data availability score in hybrid gating | Source: model; Hybrid methane scheme design |
| `pedosphere.methane.scales.gating.threshold` | `dimensionless` | `0.5; bounds=(0.3, 0.7)` | `fixed` | Threshold for gating between the iMAPLE and calibrated methane branches | Source: model; Hybrid methane scheme design |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\wetland_methane.md`.

### References

- Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 (FunctorMetadata, certificate)
- Pielke, R.A. (2002). Mesoscale meteorological modeling. (FunctorMetadata)
- Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming. (FunctorMetadata)

(class-relic-scientific-pedosphere-methane-transport-methanetransportfunctor)=
## MethaneTransportFunctor

Partition soil methane losses into diffusion, ebullition, and plant transport.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.methane.transport.MethaneTransportFunctor` |
| Source | `relic\scientific\pedosphere\methane\transport.py:25` |
| Runtime name | `pedosphere_methane_transport` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MethaneTransportFunctor(name: str = 'pedosphere_methane_transport', device: torch.device | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'pedosphere_methane_transport'` | Functor identity used by registration, provenance, and dependency resolution. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `ch4_soil_storage` | state | `g C m-2` | extensive methane-carbon storage used to calculate soil concentration |
| `water_filled_pore_space` | forcing | `1` | canonical water-filled pore-space fraction in [0,1] |
| `porosity` | forcing | `m3 m-3` | canonical total pore fraction in (0,1] |
| `atmospheric_ch4_concentration` | forcing | `g C m-3` | canonical atmospheric methane-carbon concentration boundary value |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `ch4_transport_loss_rate` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | canonical signed internal request retained |
| `ch4_storage_tendency_trans` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | canonical equal-opposite internal tendency retained |
| `diagnostics.methane_transport.diffusion_rate` | output port | `g C m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.methane_transport.ebullition_rate` | output port | `g C m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.methane_transport.plant_flux_rate` | output port | `g C m-2 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

Walter and Heimann (2000) separate diffusion, ebullition, and plant pathways; Yue et al. (2024), Eq. 42, includes texture-weighted gas and water diffusivity terms

**Canonical RELIC structural form**

$F_{\mathrm{request}} = F_{\mathrm{diffusion}} + F_{\mathrm{ebullition}} + F_{\mathrm{plant}}$; $T_{\mathrm{trans}} = - F_{\mathrm{request}}$; one aggregate internal interconnection is routed to MethaneBudgetCoupler

| Entry | Meaning |
| --- | --- |
| `D_h` | declared methane transport divergence map for structural audit |
| `star_K` | ParameterManager-backed non-negative effective diffusivity Hodge map |
| `q_h` | declared methane flux cochain |
| `G_h` | one aggregate methane_transport_request interconnection to MethaneBudgetCoupler |
| `runtime_request` | $F_{\mathrm{request}} = F_{\mathrm{diffusion}} + F_{\mathrm{ebullition}} + F_{\mathrm{plant}}$ |
| `shared_limiter_owner` | MethaneBudgetCoupler owns storage availability and the realized atmospheric boundary |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared methane storage, gradient, and flux cochain basis for structural audit only |
| `effort_variable` | soil-atmosphere methane concentration gradient |
| `state_advancer` | $T_{\mathrm{trans}} = - ( F_{\mathrm{diffusion}} + F_{\mathrm{ebullition}} + F_{\mathrm{plant}} )$ |
| `runtime_usage` | forward evaluates the calibrated hybrid once, returns one signed request and equal-opposite tendency, and delegates availability limiting to MethaneBudgetCoupler. |

### Composition channels and executable PortSpec API

Certificate composition channels: `methane_transport_request`

Executable `ports()` owner: `MethaneTransportFunctor`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `methane_transport_request` | `ch4_soil_concentration [g C m-3]` | `ch4_transport_loss_rate [g C m-2 s-1]` | `g2 C2 m-5 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.methane.transport.diffusivity_air` | `m2 s-1` | `1.8e-05; bounds=(1e-05, 3e-05)` | `fixed` | Methane molecular diffusivity in air at 25 Celsius | Source: literature; Zhuang et al. (2004) doi:10.1029/2004GB002239; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024, Eq. 42 |
| `pedosphere.methane.transport.diffusivity_water` | `m2 s-1` | `2e-09; bounds=(1e-09, 5e-09)` | `fixed` | Methane molecular diffusivity in water | Source: literature; Zhuang et al. (2004) doi:10.1029/2004GB002239; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024, Eq. 42 |
| `pedosphere.methane.transport.tortuosity_factor` | `dimensionless` | `0.66; bounds=(0.5, 0.8)` | `fixed` | Soil gas tortuosity factor for effective methane diffusion | Source: literature; Millington and Quirk (1961) Trans Faraday Soc 57:1200-1207 |
| `pedosphere.methane.transport.bubble_threshold_concentration` | `mol m-3` | `0.5; bounds=(0.1, 1.0)` | `learnable` | Calibratable dissolved methane concentration threshold for ebullition release | Source: literature; Walter and Heimann (2000) doi:10.1029/1999GB001204; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 |
| `pedosphere.methane.transport.ebullition_timescale` | `s` | `86400.0; bounds=(3600.0, 172800.0)` | `fixed` | Calibratable characteristic timescale for ebullition venting of excess methane | Source: literature; Walter and Heimann (2000) doi:10.1029/1999GB001204; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 |
| `pedosphere.methane.transport.plant_conductivity` | `m2 s-1` | `1e-07; bounds=(1e-08, 1e-06)` | `learnable` | Plant aerenchyma conductivity for methane transport | Source: literature; Zhang et al. (2002) doi:10.1029/2001GB001838; Chanton et al. (1992) doi:10.1029/91GB02969 |
| `pedosphere.methane.transport.aerenchyma_fraction` | `dimensionless` | `0.3; bounds=(0.1, 0.6)` | `fixed` | Fraction of root volume with aerenchyma tissue | Source: literature; Zhu et al. (2014) doi:10.5194/gmd-7-981-2014 |
| `pedosphere.methane.transport.root_oxidation_fraction` | `dimensionless` | `0.5; bounds=(0.2, 0.9)` | `fixed` | Fraction of methane oxidized within the root zone along plant pathway | Source: literature; Chanton et al. (1992) doi:10.1029/91GB02969; Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 |
| `pedosphere.methane.transport.layer_mixing_depth` | `m` | `0.05; bounds=(0.01, 0.2)` | `fixed` | Calibratable representative mixing depth for methane transport calculations | Source: literature; Walter and Heimann (2000) doi:10.1029/1999GB001204; Zhuang et al. (2004) doi:10.1029/2004GB002239 |
| `pedosphere.physical.millington_quirk_exponent_numerator` | `dimensionless` | `3.333333; bounds=(3.0, 3.5)` | `fixed` | Numerator exponent (10/3) controlling tortuosity in the Millington-Quirk gas diffusivity model. | Source: RELIC parameter database; Millington & Quirk (1961) Trans. Faraday Soc. 57:1200-1207; Moldrup et al. (2000) Soil Sci. Soc. Am. J. 64:94-100 |
| `pedosphere.physical.millington_quirk_exponent_denominator` | `dimensionless` | `2.0; bounds=(1.5, 2.5)` | `fixed` | Denominator exponent applied to total porosity in the Millington-Quirk tortuosity expression. | Source: RELIC parameter database; Millington & Quirk (1961) Trans. Faraday Soc. 57:1200-1207; Penman (1940) J. Agric. Sci. 30:437-462 |
| `pedosphere.numerical.minimum_timestep` | `s` | `1e-06; bounds=(1e-09, 1.0)` | `fixed` | Minimum timestep used in pedosphere denominators | Source: numerical stability; Numerical stability |
| `pedosphere.numerical.epsilon_length` | `m` | `1e-06; bounds=(1e-09, 0.01)` | `fixed` | Minimum length scale to avoid zero layer thickness | Source: numerical stability; Numerical stability |
| `pedosphere.numerical.minimum_timescale` | `s` | `1e-06; bounds=(1e-09, 1.0)` | `fixed` | Minimum timescale used in pedosphere denominators | Source: numerical stability; Numerical stability |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | explicit $\mathrm{REPLACE} \mathrm{per}_{\mathrm{timestep}} = \mathrm{False} \mathrm{internal} \mathrm{rate} \mathrm{ports}$ through StateManager; MethaneBudgetCoupler owns realized boundary and storage writes |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\wetland_methane.md`.

### References

- Walter and Heimann (2000) doi:10.1029/1999GB001204 (FunctorMetadata, certificate)
- Zhuang et al. (2004) doi:10.1029/2004GB002239 (FunctorMetadata, certificate)
- Chanton et al. (1992) doi:10.1029/91GB02969 (FunctorMetadata, certificate)
- Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 (FunctorMetadata, certificate)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)

(class-relic-scientific-pedosphere-nutrient-cycling-denitrification-denitrificationprocess)=
## denitrification_process

Denitrification process functor with anaerobic controls.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.nutrient_cycling.denitrification.DenitrificationProcess` |
| Source | `relic\scientific\pedosphere\nutrient_cycling\denitrification.py:55` |
| Runtime name | `denitrification` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
denitrification_process(name: str = 'denitrification', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'denitrification'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `no3_nitrogen` | state | `g N m-2` | state cochain x_h[0] |
| `labile_carbon` | forcing | `g C m-2` | Required or optional value consumed by forward(). |
| `water_filled_pore_space` | forcing | `1` | Required or optional value consumed by forward(). |
| `soil_temperature` | forcing | `K` | Required or optional value consumed by forward(). |
| `soil_ph` | forcing | `1` | Required or optional value consumed by forward(). |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `no3_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | state cochain x_h[0] |
| `denitrification_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | total NO3 reduction rate |
| `n2o_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | open gas-output component x_h[1] |
| `n2_production` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | open gas-output component x_h[2] |
| `diagnostics.anaerobic_factor` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.carbon_factor` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.n2o_fraction` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.temperature_response` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.ph_factor` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.denitrification.step_seconds` | output port | `s` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

Anaerobic nitrate reduction with Michaelis-Menten substrate limitation, carbon limitation, Q10 temperature response, a water-filled pore space anaerobic factor (DAYCENT basis, Del Grosso et al. 2000), pH response, and N2O:N2 product partitioning.

**Canonical RELIC structural form**

$$
\dot{x} = N_{h} r_{h} ( x_{h} , \theta ); x_{h} = [ \mathrm{NO}_{3} , \mathrm{N2O} , \mathrm{N}_{2} ] ^{\mathsf{T}} .
$$

| Entry | Meaning |
| --- | --- |
| `N_h` | explicit stoichiometric matrix in advance_reaction_network |
| `r_h` | denitrification rate split into N2O and N2 reaction channels |
| `substrate_limiter` | $\min ( R_{\mathrm{den}} , \mathrm{NO}_{3} / \mathrm{d}t )$ applied before $N_{h}$ $r_{h}$ |
| `time_integration` | one explicit Euler step $x ( t + \mathrm{d}t ) = x ( t ) + N_{h} r_{h} ( x ( t ) ) \mathrm{d}t \mathrm{per} \mathrm{governed} \mathrm{call}$ with the soil drivers held at their call-start values; simulator sub-steps repeat the step |
| `sub_flow_accuracy` | FunctorMetadata.sub_flow_accuracy ORDER_1 (Hairer, Norsett and Wanner 1993, Sect. II.3) while the donor limit is inactive, $R_{\mathrm{den}} \mathrm{d}t < \mathrm{NO}_{3}$, i.e. Vmax $f_{T} f_{\mathrm{anoxic}} f_{C} f_{\mathrm{pH}} \mathrm{d}t < K_m_{\mathrm{NO3}} + \mathrm{NO}_{3}$; binding on the two coarse grids of CR-A-04 T11 exhausts the numerical pool and yields equal final donor error, while smaller inactive-limit steps recover first order (a finite-step characterization, not loss of asymptotic convergence); the clipped N2O fraction only partitions the released nitrogen |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | nitrogen species cochain $x_{h} = [ \mathrm{NO}_{3} , \mathrm{N2O} , \mathrm{N}_{2} ] ^{\mathsf{T}}$ |
| `effort_variable` | this required certificate slot records reaction-rate tensor $r_{h}$ from ParameterManager-backed anaerobic kinetics; $r_{h}$ is a rate vector and no thermodynamic effort is asserted |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit $N_{h}$ and $r_{h}$ evaluated by advance_reaction_network |
| `runtime_usage` | DenitrificationProcess.forward calls advance_reaction_network before emitting NO3, N2O, and N2 outputs. |
| `explicit_stoichiometric_usage` | DenitrificationProcess.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `anaerobic_respiration`, `gas_emissions`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.denitrification.vmax_denitrification` | `g N m-2 s-1` | `1.5e-06; bounds=(1e-07, 1e-05)` | `learnable` | Maximum denitrification rate | Source: Denitrification kinetics studies; Parton et al. (1996); Li et al. (1992) |
| `pedosphere.denitrification.km_no3` | `g N m-2` | `0.002; bounds=(0.0005, 0.01)` | `learnable` | Half-saturation constant for NO3 | Source: Michaelis-Menten kinetics; Michaelis-Menten kinetics for denitrification |
| `pedosphere.denitrification.carbon_half_saturation` | `g C m-2` | `0.05; bounds=(0.01, 0.2)` | `learnable` | Half-saturation constant for labile carbon | Source: Carbon limitation of denitrification; Carbon limitation of denitrification |
| `pedosphere.physical.wfps_denitrification_threshold` | `dimensionless` | `0.6; bounds=(0.5, 0.8)` | `fixed` | Water-filled pore space WFPS_thr above which the denitrification anaerobic factor rises, f_anoxic = max(clip((WFPS - WFPS_thr)/w, 0, 1), f_min) with w = pedosphere.denitrification.anaerobic_transition_width and f_min = pedosphere.denitrification.minimum_anaerobic_factor. Linn and Doran (1984) found aerobic microbial activity maximal at 60 percent WFPS and declining above it through reduced aeration; DAYCENT (Del Grosso et al. 2000) drives denitrification with WFPS. Bounds: nitrification is the main N2O source up to 60 percent and denitrification produced all N2O at 70 percent WFPS (Bateman and Baggs 2005); 0.8 keeps the ramp inside the unit interval. | Source: literature; Linn, D.M. and Doran, J.W. (1984) Effect of water-filled pore space on carbon dioxide and nitrous oxide production in tilled and nontilled soils, Soil Sci. Soc. Am. J. 48, 1267-1272, doi:10.2136/sssaj1984.03615995004800060013x; Del Grosso, S.J., Parton, W.J., Mosier, A.R., Ojima, D.S., Kulmala, A.E. and Phongpan, S. (2000) General model for N2O and N2 gas emissions from soils due to dentrification, Global Biogeochem. Cycles 14, 1045-1060, doi:10.1029/1999GB001225 |
| `pedosphere.denitrification.anaerobic_transition_width` | `dimensionless` | `0.3; bounds=(0.1, 0.5)` | `learnable` | Width w, in water-filled pore space, of the linear rise of the denitrification anaerobic factor from WFPS_thr to WFPS_thr + w (0.6 to 0.9 with the defaults), f_anoxic = max(clip((WFPS - WFPS_thr)/w, 0, 1), f_min). | Source: relic_convention; RELIC convention value (width of the linear WFPS ramp; no literature value) |
| `pedosphere.denitrification.minimum_anaerobic_factor` | `dimensionless` | `0.1; bounds=(0.01, 0.3)` | `learnable` | Floor f_min of the denitrification anaerobic factor at low water-filled pore space (anaerobic microsites), f_anoxic = max(clip((WFPS - WFPS_thr)/w, 0, 1), f_min). | Source: relic_convention; RELIC convention value (anaerobic factor floor; no literature value) |
| `pedosphere.denitrification.ph_minimum_factor` | `dimensionless` | `0.2; bounds=(0.1, 0.5)` | `learnable` | Minimum pH response factor | Source: pH tolerance range; pH tolerance range |
| `pedosphere.denitrification.min_rate_threshold` | `dimensionless` | `1e-12; bounds=(0.0, inf)` | `fixed` | Lower bound of the denitrification temperature factor f_T = clip(Q10^((T - T_ref)/10), lower, upper) with Q10 = q10_denitrification and T_ref = reference_temperature. With the default Q10 = 2.2 and T_ref = 293.15 K the raw factor reaches 1e-12 only 350 K below T_ref, so the bound never binds at a physical soil temperature; it keeps f_T strictly positive. A floor of the temperature response factor cannot be negative; its upper limit is max_rate_factor, a relation between parameters that bounds cannot express, so the upper bound is open (inf). | Source: relic_convention; RELIC convention value (temperature factor lower bound; no literature value) |
| `pedosphere.denitrification.max_rate_factor` | `dimensionless` | `15.0; bounds=(1.0, 100.0)` | `fixed` | Maximum rate amplification factor | Source: Stability control; Stability control |
| `pedosphere.denitrification.base_n2o_fraction` | `dimensionless` | `0.1; bounds=(0.05, 0.2)` | `learnable` | Base N2O fraction (minimum) | Source: Firestone & Davidson (1989); Firestone & Davidson (1989) |
| `pedosphere.denitrification.max_n2o_fraction` | `dimensionless` | `0.7; bounds=(0.5, 0.9)` | `learnable` | Maximum N2O fraction | Source: High N2O production conditions; High N2O production conditions |
| `pedosphere.denitrification.n2o_no3_threshold` | `g N m-2` | `0.001; bounds=(0.0001, 0.01)` | `learnable` | NO3 threshold for N2O enhancement calculation | Source: Low NO3 conditions; Low NO3 conditions |
| `pedosphere.denitrification.n2o_no3_enhancement` | `dimensionless` | `0.6; bounds=(0.3, 0.8)` | `learnable` | NO3 enhancement factor for N2O production | Source: NO3 effect on N2O yield; NO3 effect on N2O yield |
| `pedosphere.denitrification.n2o_ph_reference` | `dimensionless` | `8.0; bounds=(7.5, 8.5)` | `learnable` | Reference pH for N2O production calculation | Source: Optimal pH for N2O reduction; Optimal pH for N2O reduction |
| `pedosphere.denitrification.n2o_ph_sensitivity` | `dimensionless` | `2.0; bounds=(1.0, 4.0)` | `learnable` | pH sensitivity coefficient for N2O production | Source: pH control of N2O reduction; pH control of N2O reduction |
| `pedosphere.denitrification.n2o_ph_effect_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `learnable` | Minimum pH effect factor for N2O fraction calculation | Source: pH control of N2O reduction; Firestone & Davidson (1989) |
| `pedosphere.denitrification.n2o_ph_effect_max` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `learnable` | Maximum pH effect factor for N2O fraction calculation | Source: pH control of N2O reduction; Firestone & Davidson (1989) |
| `pedosphere.physical.q10_denitrification` | `dimensionless` | `2.2; bounds=(2.0, 2.8)` | `learnable` | Temperature sensitivity coefficient for denitrification process | Source: Denitrification literature; Tiedje (1988) Biology of Anaerobic Microorganisms; Stanford et al. (1975) doi:10.2136/sssaj1975.03615995003900050024x |
| `pedosphere.physical.reference_temperature` | `K` | `293.15; bounds=(273.15, 323.15)` | `fixed` | Reference temperature for process normalization (default 293.15 K) | Source: ISO laboratory standards; Standard laboratory conditions; ISO 291:2008 plastics standard atmospheres for conditioning and testing |
| `pedosphere.physical.ph_optimum_denitrification` | `dimensionless` | `7.0; bounds=(6.5, 7.5)` | `learnable` | Optimal pH for denitrification process by facultative anaerobes | Source: Denitrification literature; Knowles (1982) Microbiological Reviews; Simek & Cooper (2002) doi:10.1016/S0038-0717(01)00183-X |
| `pedosphere.physical.ph_response_width` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `learnable` | Width parameter for Gaussian pH response function | Source: DNDC and CENTURY models; Parton et al. (1996) Global Biogeochemical Cycles; Li et al. (2000) doi:10.1016/S0304-3800(00)00248-9 |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |
| `validation.conservation.strict_conservation` | `dimensionless` | `False; bounds=None` | `fixed` | Whether to raise exception on conservation violation (true) or just log warning (false) | Source: validation_framework_design; Conservation law validation best practices |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | RELICSimulator applies emitted deltas and replace-mode flux outputs |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\denitrification.md`.

### References

- Parton, W.J. et al. (1996). N2 and N2O production (FunctorMetadata)
- Li, C. et al. (1992). DNDC denitrification model (FunctorMetadata)
- Parton, W.J. et al. (1996). Generalized model for N2 and N2O production. (certificate)
- Li, C. et al. (1992). DNDC denitrification model. (certificate)
- Tiedje, J.M. (1988). Ecology of denitrification and dissimilatory nitrate reduction to ammonium. (certificate)
- Feinberg, M. (2019). Foundations of Chemical Reaction Network Theory. (certificate)

(class-relic-scientific-pedosphere-nutrient-cycling-mineralization-mineralizationprocess)=
## MineralizationProcess

Nitrogen mineralization/immobilization process functor.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.nutrient_cycling.mineralization.MineralizationProcess` |
| Source | `relic\scientific\pedosphere\nutrient_cycling\mineralization.py:81` |
| Runtime name | `mineralization` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MineralizationProcess(name: str = 'mineralization', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'mineralization'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `organic_nitrogen` | state | `g N m-2` | organic_nitrogen mapped to canonical Phase 8 cochain basis |
| `nh4_nitrogen` | state | `g N m-2` | nh4_nitrogen mapped to canonical Phase 8 cochain basis |
| `microbial_nitrogen` | state | `g N m-2` | microbial_nitrogen mapped to canonical Phase 8 cochain basis |
| `soil_temperature` | forcing | `K` | soil_temperature mapped to canonical Phase 8 cochain basis |
| `soil_moisture` | state | `m3 m-3` | soil_moisture mapped to canonical Phase 8 cochain basis |
| `soil_ph` | forcing | `1` | soil_ph mapped to canonical Phase 8 cochain basis |
| `carbon_decomposition_rate` | flux | `g C m-2 s-1` | carbon_decomposition_rate mapped to canonical Phase 8 cochain basis |
| `organic_nitrogen_input` | flux | `g N m-2 s-1` | mapper-owned plant-litter plus dependency-selected fecal organic nitrogen boundary port mapped through G_h |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `organic_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | organic_nitrogen mapped to canonical Phase 8 cochain basis |
| `nh4_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | nh4_nitrogen mapped to canonical Phase 8 cochain basis |
| `microbial_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | microbial_nitrogen mapped to canonical Phase 8 cochain basis |
| `mineralization_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | canonical net mineralization rate |
| `gross_mineralization_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | canonical gross mineralization rate |
| `immobilization_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | canonical immobilization rate |
| `microbial_death_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | canonical microbial nitrogen turnover rate |
| `diagnostics.mineralization.organic_nitrogen_boundary_input` | output port | `g N m-2` | diagnostics | exact integrated organic-material nitrogen boundary amount |

### Governing equations and canonical operator

**Literature or process form**

$$
R_{\mathrm{net}} = \min ( k_{\mathrm{min}} N_{\mathrm{org}} f_{\mathrm{env}} , N_{\mathrm{org}} / \mathrm{d}t ) - \min ( \mathrm{CUE} \mathrm{DEC} / ( \mathrm{CN}_{\mathrm{mic}} \eta_{N} ) , N_{\mathrm{NH4}} / \mathrm{d}t )
$$

**Canonical RELIC structural form**

$\dot{x} = N_{h} r_{h} + G_{h} u_{\mathrm{organic}}$ with closed internal reactions and one explicit organic-material nitrogen boundary port

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |
| `G_h` | organic-material nitrogen boundary incidence vector $[ 1 , 0 , 0 ] ^{\mathsf{T}}$ |
| `u_organic` | organic_nitrogen_input supplied by LitterfallInterfaceMapper |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | Empirical environmental and stoichiometric drivers; no strict thermodynamic effort is claimed at level 2. |
| `state_advancer` | $\dot{x} = N_{h} r_{h} + G_{h} u_{\mathrm{organic}}$; advance_reaction_network owns the closed reaction term and the explicit port supplies the boundary term |
| `runtime_usage` | MineralizationProcess.forward validates $C_{h} N_{h} = 0$ through advance_reaction_network, then adds organic_nitrogen_input dt once to organic_nitrogen. |
| `explicit_stoichiometric_usage` | MineralizationProcess.forward materializes stoichiometric_N and rates for $N_{h}$ $r_{h}$ and keeps $G_{h}$ $u_{\mathrm{litter}}$ as the declared open port. |

### Composition channels and executable PortSpec API

Certificate composition channels: `carbon_decomposition_rate`, `organic_nitrogen_input`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.mineralization.temp_response_type` | `dimensionless` | `'arrhenius'; bounds=(-10000000000.0, 10000000000.0)` | `fixed` | Temperature response formulation used for gross mineralization (`arrhenius` or `q10`). | Source: model_configuration; Manzoni & Porporato (2009) Soil Biol. Biochem. 41:1355-1379 |
| `pedosphere.mineralization.q10_mineralization` | `dimensionless` | `2.4; bounds=(1.5, 3.5)` | `learnable` | Q10 temperature coefficient for mineralization rate (typically higher than decomposition due to microbial enzyme activation) | Source: Manzoni & Porporato 2009 meta-analysis; DOI:10.1016/j.soilbio.2008.10.012 |
| `pedosphere.mineralization.activation_energy` | `J mol-1` | `54000.0; bounds=(40000.0, 70000.0)` | `fixed` | Activation energy for Arrhenius temperature response of mineralization enzymes | Source: CLM-CN model (Thornton et al. 2007); DOI:10.1029/2003GB002199 |
| `pedosphere.mineralization.moisture_optimum` | `m3 m-3` | `0.6; bounds=(0.4, 0.8)` | `fixed` | Optimal volumetric soil moisture for mineralization (balance between diffusion and aeration) | Source: CENTURY model; DOI:10.2136/sssaj1988.03615995005200030009x |
| `pedosphere.mineralization.moisture_response_power` | `dimensionless` | `1.5; bounds=(0.5, 3.0)` | `fixed` | Power coefficient for moisture response curve shape | Source: Empirical calibration; Empirical moisture response |
| `pedosphere.mineralization.k_mineralization` | `yr-1` | `1.5; bounds=(0.1, 10.0)` | `learnable` | Base mineralization rate constant for gross nitrogen mineralization from organic matter | Source: CENTURY model parameterization; DOI:10.2136/sssaj1988.03615995005200030009x |
| `pedosphere.mineralization.cn_ratio_microbial` | `dimensionless` | `8.0; bounds=(6.0, 12.0)` | `fixed` | Microbial biomass C:N ratio (determines nitrogen demand during decomposition) | Source: Microbial ecology literature; DOI:10.1007/BF00336886 |
| `pedosphere.mineralization.n_assimilation_efficiency` | `dimensionless` | `0.6; bounds=(0.3, 1.0)` | `fixed` | Fraction of mineralized nitrogen assimilated into microbial biomass before release. | Source: Substrate-decomposer stoichiometry calibration; Manzoni & Porporato (2009) Soil Biol. Biochem. 41:1355-1379 |
| `pedosphere.mineralization.carbon_use_efficiency` | `dimensionless` | `0.5; bounds=(0.2, 0.7)` | `learnable` | Microbial carbon use efficiency (fraction of C assimilated into biomass vs respired) | Source: Manzoni et al. 2008 meta-analysis; DOI:10.1111/j.1365-2486.2007.01451.x |
| `pedosphere.physical.ph_optimum_decomposition` | `dimensionless` | `6.5; bounds=(6.0, 7.5)` | `learnable` | Optimal pH for organic matter decomposition by heterotrophic microbes | Source: Decomposition ecology; Swift et al. (1979) Decomposition in Terrestrial Ecosystems; Rousk et al. (2010) doi:10.1038/ismej.2010.58 |
| `pedosphere.physical.ph_response_width` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `learnable` | Width parameter for Gaussian pH response function | Source: DNDC and CENTURY models; Parton et al. (1996) Global Biogeochemical Cycles; Li et al. (2000) doi:10.1016/S0304-3800(00)00248-9 |
| `pedosphere.numerical.temperature_response_min` | `dimensionless` | `0.1; bounds=(0.01, 0.5)` | `fixed` | Minimum temperature response multiplier | Source: realistic temperature effects; Realistic temperature effects |
| `pedosphere.numerical.temperature_response_max` | `dimensionless` | `10.0; bounds=(5.0, 50.0)` | `fixed` | Maximum temperature response multiplier | Source: realistic temperature effects; Realistic temperature effects |
| `pedosphere.numerical.moisture_response_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Minimum moisture response multiplier | Source: moisture limitation; Moisture limitation |
| `pedosphere.numerical.moisture_response_max` | `dimensionless` | `1.0; bounds=(0.9, 1.1)` | `fixed` | Maximum moisture response multiplier | Source: moisture limitation; Moisture limitation |
| `pedosphere.numerical.ph_response_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Minimum pH response multiplier used for clamp limits in mineralization. | Source: pH response clamp defaults; Curtin et al. (1998) Soil Biology and Biochemistry; Brady and Weil (2017) The Nature and Properties of Soils |
| `pedosphere.numerical.ph_response_max` | `dimensionless` | `1.0; bounds=(0.9, 1.1)` | `fixed` | Maximum pH response multiplier used for clamp limits in mineralization. | Source: pH response clamp defaults; Curtin et al. (1998) Soil Biology and Biochemistry; Brady and Weil (2017) The Nature and Properties of Soils |
| `pedosphere.nutrient_cycling.k_microbial_death` | `s-1` | `1.388888888888889e-07; bounds=(5.787037037037037e-08, 8.101851851851852e-06)` | `learnable` | First-order microbial biomass death and nitrogen turnover rate. The canonical value is the Schimel-Weintraub linked C-N baseline of 0.012 d-1 converted once to s-1; dead microbial N returns to the organic N pool. | Source: Schimel and Weintraub (2003) linked carbon-nitrogen model; Manzoni & Porporato (2009). Soil Biology & Biochemistry, 41(7), 1355-1379; Schimel & Weintraub (2003). Soil Biology & Biochemistry, 35(4), 549-563, Table 1, Kt = 0.012 d-1 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\mineralization.md`.

### References

- Manzoni, S. & Porporato, A. (2009). Substrate-decomposer stoichiometry (FunctorMetadata, certificate)
- Grant, R.F. et al. (1993). Soil C and N transformations (FunctorMetadata, certificate)
- Stanford, G. & Smith, S.J. (1972). N mineralization potentials (FunctorMetadata, certificate)
- Jansson, S.L. & Persson, J. (1982). Mineralization and immobilization (FunctorMetadata, certificate)
- Parton et al. (1987). SSSA Journal 51, 1173-1179 (FunctorMetadata, certificate)
- Lawrence et al. (2019). JAMES 11, 4245-4287 (FunctorMetadata, certificate)

(class-relic-scientific-pedosphere-nutrient-cycling-nitrification-nitrificationprocess)=
## nitrification_process

Nitrification process functor using Michaelis-Menten kinetics.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.nutrient_cycling.nitrification.NitrificationProcess` |
| Source | `relic\scientific\pedosphere\nutrient_cycling\nitrification.py:61` |
| Runtime name | `nitrification` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
nitrification_process(name: str = 'nitrification', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'nitrification'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `nh4_nitrogen` | state | `g N m-2` | state cochain x_h[0] |
| `soil_temperature` | forcing | `K` | Required or optional value consumed by forward(). |
| `water_filled_pore_space` | forcing | `1` | Required or optional value consumed by forward(). |
| `soil_ph` | forcing | `1` | Required or optional value consumed by forward(). |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `nh4_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | state cochain x_h[0] |
| `no3_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | state cochain x_h[1] |
| `nitrification_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | total NH4 oxidation rate |
| `n2o_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | open gas-output component x_h[2] |
| `diagnostics.temperature_response` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.oxygen_factor` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.ph_factor` | output port | `not declared` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.nitrification.step_seconds` | output port | `s` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.nitrification.substrate_limiter_residual` | output port | `g N m-2 s-1` | diagnostics | unmet nitrification demand R_nit - min(R_nit, NH4 / dt); a demand, not nitrogen |

### Governing equations and canonical operator

**Literature or process form**

Michaelis-Menten ammonia oxidation with Q10, water-filled pore space oxygen (Linn and Doran 1984), and pH modifiers; N2O is a fixed fraction of nitrified nitrogen.

**Canonical RELIC structural form**

$$
\dot{x} = N_{h} r_{h} ( x_{h} , \theta ); x_{h} = [ \mathrm{NH}_{4} , \mathrm{NO}_{3} , \mathrm{N2O} ] ^{\mathsf{T}} .
$$

| Entry | Meaning |
| --- | --- |
| `N_h` | explicit stoichiometric matrix in advance_reaction_network |
| `r_h` | Michaelis-Menten nitrification rate split into NO3 and N2O reaction channels |
| `substrate_limiter` | $\min ( R_{\mathrm{nit}} , \mathrm{NH}_{4} / \mathrm{d}t )$ applied before $N_{h}$ $r_{h}$ |
| `time_integration` | one explicit Euler step $x ( t + \mathrm{d}t ) = x ( t ) + N_{h} r_{h} ( x ( t ) ) \mathrm{d}t \mathrm{per} \mathrm{governed} \mathrm{call}$ with the soil drivers held at their call-start values; simulator sub-steps repeat the step |
| `sub_flow_accuracy` | FunctorMetadata.sub_flow_accuracy ORDER_1 (Hairer, Norsett and Wanner 1993, Sect. II.3) while the donor limit is inactive, $R_{\mathrm{nit}} \mathrm{d}t < \mathrm{NH}_{4}$, i.e. Vmax $f_{T} f_{\mathrm{O2}} f_{\mathrm{pH}} \mathrm{d}t < K_m + \mathrm{NH}_{4}$; binding on the two coarse grids of CR-A-04 T11 exhausts the numerical pool and yields equal final donor error, while smaller inactive-limit steps recover first order (a finite-step characterization, not a loss of asymptotic convergence) |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | nitrogen species cochain $x_{h} = [ \mathrm{NH}_{4} , \mathrm{NO}_{3} , \mathrm{N2O} ] ^{\mathsf{T}}$ |
| `effort_variable` | this required certificate slot records reaction-rate tensor $r_{h}$ from ParameterManager-backed kinetics; $r_{h}$ is a rate vector and no thermodynamic effort is asserted |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit $N_{h}$ and $r_{h}$ evaluated by advance_reaction_network |
| `runtime_usage` | NitrificationProcess.forward calls advance_reaction_network before emitting NH4, NO3, and N2O outputs. |
| `explicit_stoichiometric_usage` | NitrificationProcess.forward materializes stoichiometric_N and rates, then lowers $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `soil_chemistry`, `gas_emissions`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.nitrification.vmax_nitrification` | `g N m-2 s-1` | `2e-06; bounds=(1e-07, 1e-05)` | `learnable` | Maximum nitrification rate | Source: Nitrification kinetics studies; Parton et al. (1996); Stark & Firestone (1995) |
| `pedosphere.nitrification.km_nh4` | `g N m-2` | `0.005; bounds=(0.001, 0.02)` | `learnable` | Areal NH4 half-saturation constant Km of the RELIC nitrification rate R = Vmax NH4/(Km + NH4) f_T f_O2 f_pH. | Source: relic_convention; RELIC convention value (areal NH4 half-saturation; no literature value) |
| `pedosphere.physical.wfps_nitrification_optimum` | `dimensionless` | `0.6; bounds=(0.35, 0.7)` | `learnable` | Water-filled pore space WFPS_opt at which aerobic microbial activity, and with it nitrification, peaks; above it oxygen limits nitrification, f_O2 = max((1 - WFPS)/(1 - WFPS_opt), pedosphere.nitrification.minimum_oxygen_factor), and below it f_O2 = 1. Linn and Doran (1984) found the maximum at 60 percent WFPS with a decline above it through reduced aeration. The bounds are the Bateman and Baggs (2005) range: nitrification is the main N2O source at 35-60 percent WFPS and denitrification produced all N2O at 70 percent. | Source: literature; Linn, D.M. and Doran, J.W. (1984) Effect of water-filled pore space on carbon dioxide and nitrous oxide production in tilled and nontilled soils, Soil Sci. Soc. Am. J. 48, 1267-1272, doi:10.2136/sssaj1984.03615995004800060013x; Bateman, E.J. and Baggs, E.M. (2005) Contributions of nitrification and denitrification to N2O emissions from soils at different water-filled pore space, Biol. Fertil. Soils 41, 379-388, doi:10.1007/s00374-005-0858-3 |
| `pedosphere.nitrification.minimum_oxygen_factor` | `dimensionless` | `0.1; bounds=(0.01, 0.3)` | `learnable` | Floor of the dimensionless nitrification oxygen response factor, f_O2 = max((1 - WFPS)/(1 - WFPS_opt), f_O2_min) above WFPS_opt = pedosphere.physical.wfps_nitrification_optimum; with the default 0.1 and 0.6 it binds only for WFPS > 0.96. | Source: relic_convention; RELIC convention value (oxygen response factor floor; no literature value) |
| `pedosphere.nitrification.extreme_temperature_factor` | `dimensionless` | `0.001; bounds=(0.0001, 0.01)` | `learnable` | Multiplier of the nitrification temperature factor f_T outside [temperature_minimum, temperature_maximum]. | Source: relic_convention; RELIC convention value (temperature factor multiplier outside the limits; no literature value) |
| `pedosphere.nitrification.temperature_minimum` | `K` | `273.15; bounds=(0.0, inf)` | `fixed` | Lower soil temperature limit of the nitrification temperature factor: below it f_T is multiplied by extreme_temperature_factor. An absolute (thermodynamic) temperature in K cannot be negative; the lower bound is 0 K. | Source: relic_convention; RELIC convention value (temperature factor limit at the freezing point; no literature value) |
| `pedosphere.nitrification.temperature_maximum` | `K` | `323.15; bounds=(0.0, inf)` | `fixed` | Upper soil temperature limit of the nitrification temperature factor (50 degC): above it f_T is multiplied by extreme_temperature_factor. An absolute (thermodynamic) temperature in K cannot be negative; the lower bound is 0 K. | Source: relic_convention; RELIC convention value (temperature factor upper limit; no literature value) |
| `pedosphere.nitrification.ph_minimum_factor` | `dimensionless` | `0.1; bounds=(0.01, 0.3)` | `learnable` | Floor of the nitrification pH response factor f_pH = clip(exp(-0.5 ((pH - pH_opt)/w_pH)^2), ph_minimum_factor, 1). | Source: relic_convention; RELIC convention value (pH response factor floor; no literature value) |
| `pedosphere.nitrification.min_rate_threshold` | `dimensionless` | `0.001; bounds=(0.0, inf)` | `fixed` | Lower bound of the nitrification temperature factor f_T = clip(Q10^((T - T_ref)/10) * f_extreme, lower, upper). With the default parameters it binds only below temperature_minimum (273.15 K), where it holds f_T at 0.001. A floor of the temperature response factor cannot be negative; its upper limit is max_rate_factor, a relation between parameters that bounds cannot express, so the upper bound is open (inf). | Source: relic_convention; RELIC convention value (temperature factor lower bound; no literature value) |
| `pedosphere.nitrification.max_rate_factor` | `dimensionless` | `10.0; bounds=(1.0, 100.0)` | `fixed` | Upper bound of the nitrification temperature factor f_T. With the default parameters it never binds (f_T <= 8 at 323.15 K, and 0.001 times the Q10 term above it), so it is a protective bound; it binds inside the range only when Q10^((T_max - T_ref)/10) > 10, for example Q10 = 2.5 above about 318.3 K. | Source: relic_convention; RELIC convention value (temperature factor upper bound; no literature value) |
| `pedosphere.nitrification.n2o_factor` | `dimensionless` | `0.02; bounds=(0.005, 0.05)` | `learnable` | Fraction of nitrified N converted to N2O | Source: IPCC emission factors; Bouwman et al. (2002); IPCC emission factors |
| `pedosphere.physical.q10_nitrification` | `dimensionless` | `2.0; bounds=(1.8, 2.5)` | `learnable` | Temperature sensitivity coefficient for nitrification process | Source: Nitrogen cycling literature; Stark & Firestone (1995) doi:10.1016/0038-0717(94)00192-6; Avrahami & Conrad (2003) doi:10.1128/AEM.69.3.1593-1600.2003 |
| `pedosphere.physical.reference_temperature` | `K` | `293.15; bounds=(273.15, 323.15)` | `fixed` | Reference temperature for process normalization (default 293.15 K) | Source: ISO laboratory standards; Standard laboratory conditions; ISO 291:2008 plastics standard atmospheres for conditioning and testing |
| `pedosphere.physical.ph_optimum_nitrification` | `dimensionless` | `7.5; bounds=(7.0, 8.5)` | `learnable` | Optimal pH for nitrification process by ammonia-oxidizing bacteria | Source: Nitrifier physiology literature; Prosser (1989) Advances in Microbial Ecology; Nicol et al. (2008) doi:10.1111/j.1462-2920.2008.01786.x |
| `pedosphere.physical.ph_response_width` | `dimensionless` | `2.0; bounds=(1.0, 3.0)` | `learnable` | Width parameter for Gaussian pH response function | Source: DNDC and CENTURY models; Parton et al. (1996) Global Biogeochemical Cycles; Li et al. (2000) doi:10.1016/S0304-3800(00)00248-9 |
| `pedosphere.numerical.unity_multiplier` | `dimensionless` | `1.0; bounds=(1.0, 1.0)` | `fixed` | Unity multiplier for normalization operations (always equals 1.0) | Source: Mathematical constant; Standard mathematical constant |
| `pedosphere.numerical.epsilon` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-08)` | `fixed` | Small value added to denominators to prevent division by zero | Source: numerical stability; Numerical stability |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |
| `validation.conservation.strict_conservation` | `dimensionless` | `False; bounds=None` | `fixed` | Whether to raise exception on conservation violation (true) or just log warning (false) | Source: validation_framework_design; Conservation law validation best practices |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | RELICSimulator applies emitted deltas and replace-mode flux outputs |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\nitrification.md`.

### References

- Parton, W.J. et al. (1996). N2 and N2O production model (FunctorMetadata)
- Li, C. et al. (1992). DNDC nitrification formulations (FunctorMetadata)
- Parton, W.J. et al. (1996). Generalized model for N2 and N2O production. (certificate)
- Li, C. et al. (1992). DNDC nitrification formulations. (certificate)
- Prosser, J.I. (1989). Autotrophic nitrification in bacteria. (certificate)
- Feinberg, M. (2019). Foundations of Chemical Reaction Network Theory. (certificate)

(class-relic-scientific-pedosphere-nutrient-cycling-nitrogen-leaching-nitrogenleachingprocess)=
## NitrogenLeachingProcess

Nitrogen leaching process functor with hydrological coupling.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching.NitrogenLeachingProcess` |
| Source | `relic\scientific\pedosphere\nutrient_cycling\nitrogen_leaching.py:71` |
| Runtime name | `nitrogen_leaching` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
NitrogenLeachingProcess(name: str = 'nitrogen_leaching', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'nitrogen_leaching'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `no3_nitrogen` | state | `g N m-2` | no3_nitrogen mapped to canonical Phase 8 cochain basis |
| `soil_moisture` | state | `m3 m-3` | soil_moisture mapped to canonical Phase 8 cochain basis |
| `drainage_flux` | flux | `kg m-2 s-1` | drainage_flux mapped to canonical Phase 8 cochain basis |
| `soil_depth` | forcing | `m` | soil_depth forcing z [m] of the soil water volume theta z |
| `bulk_density` | forcing | `kg m-3` | bulk_density forcing rho_b [kg m-3] of the retardation R = 1 + rho_b K_d / theta |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `nitrogen_leaching_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | retained canonical outward nitrate export rate |
| `no3_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | no3_nitrogen mapped to canonical Phase 8 cochain basis |
| `diagnostics.nitrogen_leaching.nitrate_concentration` | output port | `g N m-3` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.nitrogen_leaching.mobile_fraction` | output port | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.nitrogen_leaching.soil_water_volume` | output port | `m` | diagnostics | soil water volume per unit ground area, theta z_soil, a volume per area (m3 m-2) reported as the equivalent depth in m |
| `diagnostics.nitrogen_leaching.step_seconds` | output port | `s` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

$$
F_{N} = ( F_{\mathrm{water}} / \rho_{w} ) \cdot ( N_{\mathrm{NO3}} / ( \theta z_{\mathrm{soil}} ) ) / ( 1 + \rho_{b} K_{d} / \theta )
$$

**Canonical RELIC structural form**

$$
\dot{x} = - D_{h} q_{h} + G_{h} u_{h}, q_{h} = \min ( \star_{Q} C_{\mathrm{NO3}} , N_{\mathrm{NO3}} / \tau_{\mathrm{leach}} , f_{\mathrm{max}} N_{\mathrm{NO3}} / \mathrm{d}t ), y_{h} = G_{h} ^{\mathsf{T}} e_{h}
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | existing divergence/transport operator |
| `q_h` | declared flux cochain |
| `G_h` | boundary/source port map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = - D_{h} q_{h} + G_{h} u_{h}$ with declared flux $q_{h}$ and open drainage port |
| `runtime_usage` | NitrogenLeachingProcess.forward evaluates the single canonical nitrate drainage transport calculation. Empirical drainage and adsorption formulas are used only as ParameterManager-backed mobility and concentration coefficients. |

### Composition channels and executable PortSpec API

Certificate composition channels: `nitrate_drainage_export`

Executable `ports()` owner: `NitrogenLeachingProcess`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `nitrate_drainage_export` | `nitrate_concentration [g N m-3]` | `nitrogen_leaching_flux [g N m-2 s-1]` | `g2 N2 m-5 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.nutrient_cycling.leaching_time_constant` | `s` | `3600.0; bounds=(600.0, 86400.0)` | `fixed` | Time constant for maximum nitrogen leaching rate. Used to prevent unrealistic depletion of N pools in single timestep. Typical value: 1 hour (3600 s). | Source: Numerical stability constraint; RHESSys compute_N_leached: Conservative limiting |
| `pedosphere.nutrient_cycling.min_no3_pool` | `g N m-2` | `1e-08; bounds=(0.0, inf)` | `fixed` | Minimum NO3 pool size for numerical stability. A pool floor cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). | Source: Numerical stability; Numerical stability |
| `pedosphere.transport.nitrate_distribution_coefficient` | `L kg-1` | `0.1; bounds=(0.0, 1.0)` | `learnable` | Distribution coefficient (Kd) for nitrate sorption on soil particles, very mobile anion with minimal sorption | Source: Solute transport literature; Jury & Horton (2004) Soil Physics 6th edition; van Genuchten & Wierenga (1986) doi:10.2136/sssaj1986.03615995005000050013x |
| `pedosphere.nutrient_cycling.hodge_reference_soil_depth` | `m` | `1.0; bounds=(0.1, 3.0)` | `fixed` | Reference soil depth of the static nitrate-leaching Hodge contract (NitrogenLeachingProcess.build_hodge_contract metric key); a structural reference, never the runtime depth, which is the soil_depth forcing input | Source: structural_reference; Cameron and Haynes (1986), nitrate movement in soil |
| `pedosphere.numerical.max_flux_fraction` | `dimensionless` | `0.5; bounds=(0.1, 0.9)` | `fixed` | Maximum fraction of any soil pool allowed to flux within a single timestep to maintain stability | Source: RELIC numerical stability design; CENTURY/DAYCENT soil carbon numerical stability guidelines |
| `validation.conservation.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-10, 0.001)` | `fixed` | General relative tolerance for conservation law violations (dimensionless fraction) | Source: numerical_stability_requirement; Numerical Methods for Conservation Laws (LeVeque, 1992) |
| `validation.conservation.strict_conservation` | `dimensionless` | `False; bounds=None` | `fixed` | Whether to raise exception on conservation violation (true) or just log warning (false) | Source: validation_framework_design; Conservation law validation best practices |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\nitrogen_leaching.md`.

### References

- Li et al. (2006), doi:10.1016/j.ecolmodel.2006.02.007 (FunctorMetadata, certificate)
- van Genuchten and Wierenga (1986), soil solute transport (FunctorMetadata, certificate)
- Biome-BGC nleaching.c: MOBILEN_PROPORTION formulation (FunctorMetadata, certificate)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)

(class-relic-scientific-pedosphere-nutrient-cycling-nitrogen-nitrogencycling)=
## NitrogenCycling

Observe the composed mineral-N ledger without becoming a second writer.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.nutrient_cycling.nitrogen.NitrogenCycling` |
| Source | `relic\scientific\pedosphere\nutrient_cycling\nitrogen.py:31` |
| Runtime name | `NitrogenCycling` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
NitrogenCycling(config: Dict | None = None, device: torch.device | None = None, enable_mineralization: bool = True, enable_nitrification: bool = True, enable_denitrification: bool = True, enable_leaching: bool = True, enable_root_uptake: bool = True)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `enable_mineralization` | `bool` | `True` | Enables or disables the mineralization pathway. |
| `enable_nitrification` | `bool` | `True` | Enables or disables the nitrification pathway. |
| `enable_denitrification` | `bool` | `True` | Enables or disables the denitrification pathway. |
| `enable_leaching` | `bool` | `True` | Enables or disables the leaching pathway. |
| `enable_root_uptake` | `bool` | `True` | Enables or disables the root uptake pathway. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `fertilizer_n_flux` | flux | `kg N m-2 s-1` | fertilizer_n_flux mapped to the declared external nitrogen port |
| `urine_n_flux` | flux | `kg N m-2 s-1` | dependency-selected grazing urine N mapped to the declared external nitrogen port |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `nh4_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | only the aggregate external-port increment is emitted by NitrogenCycling |
| `diagnostics.nitrogen_cycling.expected_delta_nh4_nitrogen` | output port | `g N m-2` | diagnostics | read-only dependency projection |
| `diagnostics.nitrogen_cycling.expected_delta_no3_nitrogen` | output port | `g N m-2` | diagnostics | read-only dependency projection |
| `diagnostics.nitrogen_cycling.expected_delta_mineral_nitrogen` | output port | `g N m-2` | diagnostics | sum of the two diagnostic mineral-pool increments |
| `diagnostics.nitrogen_cycling.external_fertilizer_n_delta` | output port | `g N m-2` | diagnostics | explicit open-port contribution |
| `diagnostics.nitrogen_cycling.external_urine_n_delta` | output port | `g N m-2` | diagnostics | explicit non-strict grazing urine open-port contribution |

### Governing equations and canonical operator

**Literature or process form**

$$
\frac{\mathrm{d}\mathrm{NH}_{4}}{\mathrm{d}t} = \mathrm{mineralization} - \mathrm{nitrification} - \mathrm{uptake}_{\mathrm{NH4}}; \frac{\mathrm{d}\mathrm{NO}_{3}}{\mathrm{d}t} = \mathrm{nitrification} - \mathrm{denitrification} - \mathrm{leaching} - \mathrm{uptake}_{\mathrm{NO3}}
$$

**Canonical RELIC structural form**

diagnostic $\dot{d} = N_{h} r_{h}$ for dependency increments; prognostic $\Delta_{\mathrm{NH4},\mathrm{ext}} = c_{\mathrm{kg},\mathrm{to},g} ( F_{\mathrm{fert}} + F_{\mathrm{urine}} ) \mathrm{d}t$

| Entry | Meaning |
| --- | --- |
| `N_h` | declared stoichiometric or output incidence map for this reaction process |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | ammonium and nitrate dependency increments in g N m-2 |
| `effort_variable` | already evaluated child-process increments supplied by the dependency graph |
| `state_advancer` | $\dot{d} = N_{h} r_{h}$ is evaluated only as a diagnostic projection; the aggregate fertilizer and selected-urine boundary supplies the sole StateManager increment |
| `runtime_usage` | NitrogenCycling.forward calls advance_reaction_network with explicit $N_{h}$ and $r_{h}$ under diagnostic_residual routing, never re-emits child state or flux outputs, and emits only the aggregate external NH4 increment. |
| `explicit_stoichiometric_usage` | NitrogenCycling.forward materializes the two-row mineral-N diagnostic incidence map and evaluates $N_{h}$ $r_{h}$ through advance_reaction_network. |

### Composition channels and executable PortSpec API

Certificate composition channels: `fertilizer_n_flux -> nh4_nitrogen`, `urine_n_flux -> nh4_nitrogen when grazing_management is selected`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Physical constants (SSOT)

These values are resolved from the physical-constants database. They are not domain parameters and must not be duplicated in scientific YAML or Python code.

| Constant key | Units | Value and fixed bounds | Gradient | Meaning | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `relic.physical.constants.conversions.kg_to_g` | `g kg-1` | `1000.0; bounds=(1000.0, 1000.0)` | `fixed` | Conversion multiplier from kilograms to grams | Source: SI_definition; BIPM SI Brochure 9th Edition (2019) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, $\mathrm{subprocess}_{\mathrm{outputs}} = \mathrm{None} , \cdot \cdot \mathrm{kwargs} )$ |
| `state_write_mode` | explicit UpdateItem ACCUMULATE per_timestep through the existing StateManager SCM ledger |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\nitrogen_cycling.md`.

### References

- Parton, W.J. et al. (1987). Soil Sci. Soc. Am. J., 51(5), 1173-1179. (FunctorMetadata, certificate)
- Davidson, E.A. et al. (2012). Global Biogeochem. Cycles, 26(2). (FunctorMetadata, certificate)
- Li, C. et al. (1992). J. Geophys. Res., 97(D9), 9759-9776. (FunctorMetadata, certificate)

(class-relic-scientific-pedosphere-nutrient-cycling-root-nutrient-uptake-rootnutrientuptake)=
## RootNutrientUptake

Root Nutrient Uptake Process

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake.RootNutrientUptake` |
| Source | `relic\scientific\pedosphere\nutrient_cycling\root_nutrient_uptake.py:42` |
| Runtime name | `root_nutrient_uptake` |
| Scientific abstraction | `BiologicalProcess` |
| Canonical bridge | `BiologicalProcessFunctor` |
| Structural role | `reaction` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
RootNutrientUptake(name: str = 'root_nutrient_uptake', config: Dict | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'root_nutrient_uptake'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `no3_nitrogen` | state | `g N m-2` | no3_nitrogen mapped to canonical Phase 8 cochain basis |
| `nh4_nitrogen` | state | `g N m-2` | nh4_nitrogen mapped to canonical Phase 8 cochain basis |
| `available_phosphorus` | state | `g P m-2` | available_phosphorus mapped to canonical Phase 8 cochain basis |
| `soil_water_content` | forcing | `m3 m-3` | soil_water_content mapped to canonical Phase 8 cochain basis |
| `soil_temperature` | forcing | `K` | soil_temperature mapped to canonical Phase 8 cochain basis |
| `soil_depth` | forcing | `m` | soil_depth mapped to canonical Phase 8 cochain basis |
| `plant_n_storage` | state | `mol N m-2` | plant_n_storage mapped to canonical Phase 8 cochain basis |
| `plant_p_storage` | state | `g P m-2` | plant_p_storage mapped to canonical Phase 8 cochain basis |
| `root_density` | forcing | `kg m-3` | root_density mapped to the normalized layer root-weight basis |
| `plant_n_demand` | forcing | `mol N m-2 s-1` | plant_n_demand mapped to the N uptake-rate limiter |
| `plant_p_demand` | forcing | `mol P m-2 s-1` | plant_p_demand mapped to the P uptake-rate limiter |
| `field_capacity` | forcing | `m3 m-3` | field_capacity mapped to the water-response upper threshold |
| `wilting_point` | forcing | `m3 m-3` | wilting_point mapped to the water-response lower threshold |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `no3_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | no3_nitrogen mapped to canonical Phase 8 cochain basis |
| `nh4_nitrogen` | state | `g N m-2` | decided at run time by the returned write mode | nh4_nitrogen mapped to canonical Phase 8 cochain basis |
| `available_phosphorus` | state | `g P m-2` | decided at run time by the returned write mode | available_phosphorus mapped to canonical Phase 8 cochain basis |
| `plant_n_storage` | state | `mol N m-2` | decided at run time by the returned write mode | plant_n_storage mapped to canonical Phase 8 cochain basis |
| `plant_p_storage` | state | `g P m-2` | decided at run time by the returned write mode | plant_p_storage mapped to canonical Phase 8 cochain basis |
| `nitrogen_uptake_rate` | flux | `mol N m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `phosphorus_uptake_rate` | flux | `g P m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `no3_uptake_by_layer` | diagnostic | `mol N m-2 s-1` | decided at run time by the returned write mode | canonical layer-resolved uptake diagnostic |
| `nh4_uptake_by_layer` | diagnostic | `mol N m-2 s-1` | decided at run time by the returned write mode | canonical layer-resolved uptake diagnostic |
| `po4_uptake_by_layer` | diagnostic | `g P m-2 s-1` | decided at run time by the returned write mode | canonical layer-resolved uptake diagnostic |

### Governing equations and canonical operator

**Literature or process form**

$$
U = V_{\max} \cdot C / ( K_m + C ) \cdot f_{T} \cdot f_{W} \cdot W_{\mathrm{root}}
$$

**Canonical RELIC structural form**

$$
\dot{x}=N_h r_h
$$

with ParameterManager-backed process rates $r_h$ and declared stoichiometric/output map $N_h$

| Entry | Meaning |
| --- | --- |
| `N_h` | three signed transfer columns map nitrate and ammonium to plant N and soil P to plant P |
| `r_h` | ParameterManager-backed process-rate vector evaluated by the production forward path |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $\dot{x} = N_{h} r_{h}$ with explicit stoichiometric_N and rates evaluated by advance_reaction_network |
| `runtime_usage` | RootNutrientUptake.forward calls advance_reaction_network with declared stoichiometric_N and rates before emitting canonical outputs. |
| `explicit_stoichiometric_usage` | RootNutrientUptake.forward materializes nitrate, ammonium, and phosphate transfer columns and validates equivalent-mole N plus mass-basis P ledgers before lowering $\dot{x} = N_{h} r_{h}$ through advance_reaction_network. |
| `layered_donor_increments` | donor pools declared with a layer axis receive per-layer increments $\Delta_{\mathrm{pool}} ( z ) = - M U ( z ) \mathrm{d}t$ whose layer sum equals the areal $N_{h}$ $r_{h}$ dt transfer; plant_n_storage and plant_p_storage receive the areal increment, so StateManager shapes match the declared pool shapes |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.nutrient_cycling.vmax_n` | `mol m-2 s-1` | `1e-06; bounds=(0.0, 0.0001)` | `learnable` | Maximum nitrogen uptake rate per unit root surface area (Michaelis-Menten Vmax for N) | Source: Root uptake kinetics literature; Barber, S.A. (1995). Soil Nutrient Bioavailability; Clarkson, D.T. (1985). Factors affecting mineral nutrient acquisition by plants |
| `pedosphere.nutrient_cycling.vmax_p` | `mol m-2 s-1` | `1e-07; bounds=(0.0, 1e-05)` | `learnable` | Maximum phosphorus uptake rate per unit root surface area (Michaelis-Menten Vmax for P) | Source: Root uptake kinetics literature; Barber, S.A. (1995). Soil Nutrient Bioavailability; Jungk, A. (2001). Root hairs and the acquisition of plant nutrients from soil |
| `pedosphere.nutrient_cycling.km_n` | `mol m-3` | `0.1; bounds=(0.0, 10.0)` | `learnable` | Michaelis constant for nitrogen uptake (half-saturation concentration) | Source: Root uptake kinetics experiments; Barber, S.A. (1995). Soil Nutrient Bioavailability, Chapter 4; Silberbush, M. & Barber, S.A. (1983). Plant and Soil, 74, 201-212 |
| `pedosphere.nutrient_cycling.km_p` | `mol m-3` | `0.01; bounds=(0.0, 1.0)` | `learnable` | Michaelis constant for phosphorus uptake (half-saturation concentration) | Source: Root uptake kinetics experiments; Barber, S.A. (1995). Soil Nutrient Bioavailability; Jungk, A. (2001). Journal of Plant Nutrition and Soil Science |
| `pedosphere.nutrient_cycling.max_storage_n` | `mol m-2` | `10.0; bounds=(0.0, 100.0)` | `fixed` | Maximum plant nitrogen storage capacity | Source: Plant nutrient storage literature; Chapin, F.S. et al. (1990). Annual Review of Ecology and Systematics, 21, 423-447 |
| `pedosphere.nutrient_cycling.max_storage_p` | `mol m-2` | `1.0; bounds=(0.0, 10.0)` | `fixed` | Maximum plant phosphorus storage capacity | Source: Plant nutrient storage literature; Chapin, F.S. et al. (1990). Annual Review of Ecology and Systematics |
| `pedosphere.nutrient_cycling.root_decay_coeff` | `m-1` | `2.0; bounds=(0.1, 10.0)` | `fixed` | Exponential decay coefficient for root density with soil depth | Source: Global root distribution meta-analysis; Jackson, R.B. et al. (1996). Oecologia, 108, 389-411; Schenk, H.J. & Jackson, R.B. (2002). Ecology Letters, 5, 235-241 |
| `pedosphere.nutrient_cycling.q10_value` | `dimensionless` | `2.0; bounds=(1.0, 5.0)` | `fixed` | Q10 temperature coefficient for root nutrient uptake processes | Source: Temperature response literature; Lloyd, J. & Taylor, J.A. (1994). Functional Ecology, 8, 315-323; Atkin, O.K. & Tjoelker, M.G. (2003). Functional Ecology, 17, 747-754 |
| `pedosphere.nutrient_cycling.reference_temperature` | `K` | `298.15; bounds=(273.15, 323.15)` | `fixed` | Reference temperature for Q10 response function (25 Celsius = 298.15 K) | Source: Standard biochemical reference temperature; Lloyd & Taylor (1994). Standard reference temperature |
| `pedosphere.nutrient_cycling.demand_regulation_enabled` | `dimensionless` | `True; bounds=None` | `fixed` | Enable plant demand regulation of nutrient uptake (limits uptake to plant requirement) | Source: Model configuration defaults; Model configuration option |
| `pedosphere.nutrient_cycling.water_stress_enabled` | `dimensionless` | `True; bounds=None` | `fixed` | Enable water stress effect on root nutrient uptake | Source: Model configuration defaults; Model configuration option |
| `pedosphere.numerical.temperature_clamp_min` | `K` | `250.0; bounds=(200.0, 273.15)` | `fixed` | Minimum realistic soil temperature for calculations (hard lower bound) | Source: physical constraints for soil processes; Physical constraints for soil processes |
| `pedosphere.numerical.temperature_clamp_max` | `K` | `350.0; bounds=(300.0, 400.0)` | `fixed` | Maximum realistic soil temperature for calculations (hard upper bound) | Source: physical constraints for soil processes; Physical constraints for soil processes |
| `validation.conservation.strict_conservation` | `dimensionless` | `False; bounds=None` | `fixed` | Whether to raise exception on conservation violation (true) or just log warning (false) | Source: validation_framework_design; Conservation law validation best practices |
| `validation.numerical.response.ph_optimum` | `dimensionless` | `7.0; bounds=(0.0, 14.0)` | `fixed` | Default optimal pH for generic biological responses. The pH of dilute aqueous soil solutions lies on the 0-14 scale. This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.ph_width` | `dimensionless` | `1.5; bounds=(0.0, inf)` | `fixed` | Default width for pH response curve. A response width in pH units cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil chemistry practice |
| `validation.numerical.response.moisture_half_saturation` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Default half-saturation point for moisture response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_min` | `dimensionless` | `0.1; bounds=(0.0, 1.0)` | `fixed` | Default minimum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_linear_max` | `dimensionless` | `0.8; bounds=(0.0, 1.0)` | `fixed` | Default maximum moisture for linear response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_optimum` | `dimensionless` | `0.6; bounds=(0.0, 1.0)` | `fixed` | Default optimum moisture for bell-shaped response. A relative soil moisture lies in [0, 1]. This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |
| `validation.numerical.response.moisture_width` | `dimensionless` | `0.2; bounds=(0.0, inf)` | `fixed` | Default width for bell-shaped moisture response. A response width cannot be negative; no physical upper limit is fixed, so the upper bound is open (inf). This runtime metadata binding is optional. | Source: response_defaults; General soil moisture response practice |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | BiologicalProcess |
| `bridge_class` | BiologicalProcessFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\root_nutrient_uptake.md`.

### References

- Barber, S.A. (1995). Soil Nutrient Bioavailability: A Mechanistic Approach (FunctorMetadata, certificate)
- Tinker, P.B. & Nye, P.H. (2000). Solute Movement in the Rhizosphere (FunctorMetadata, certificate)
- Chapin, F.S. et al. (1993). Evolution of suites of traits (FunctorMetadata, certificate)
- Theory doc: docs/source/theory/root_nutrient_uptake.md (FunctorMetadata, certificate)
- Griffiths, M. & York, L.M. (2020). Targeting Root Ion Uptake Kinetics to Increase Plant Productivity and Nutrient Use Efficiency (certificate)
- von Bloh, W. et al. (2018). Implementing the nitrogen cycle into LPJmL 5.0 (certificate)
- NIST Chemistry WebBook SRD 69; CIAAW Standard Atomic Weights (certificate)

(class-relic-scientific-pedosphere-nutrient-cycling-stoichiometry-stoichiometryconstraint)=
## StoichiometryConstraint

Stoichiometric constraint functor for C-N coupling.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.nutrient_cycling.stoichiometry.StoichiometryConstraint` |
| Source | `relic\scientific\pedosphere\nutrient_cycling\stoichiometry.py:44` |
| Runtime name | `stoichiometry_constraint` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `observation` |
| Spatial scale | `grid_cell` |
| Variable kind | `diagnostic` |
| IPHS level | `level_1` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
StoichiometryConstraint(name: str = 'stoichiometry_constraint', config: Dict | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'stoichiometry_constraint'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `diagnostics.carbon_fluxes` | input port | `not declared` | diagnostics.carbon_fluxes mapped to canonical Phase 8 cochain basis |
| `diagnostics.nitrogen_fluxes` | input port | `not declared` | diagnostics.nitrogen_fluxes mapped to canonical Phase 8 cochain basis |
| `diagnostics.carbon_inputs` | input port | `not declared` | diagnostics.carbon_inputs mapped to canonical Phase 8 cochain basis |
| `diagnostics.nitrogen_inputs` | input port | `not declared` | diagnostics.nitrogen_inputs mapped to canonical Phase 8 cochain basis |
| `diagnostics.carbon_outputs` | input port | `not declared` | diagnostics.carbon_outputs mapped to canonical Phase 8 cochain basis |
| `diagnostics.nitrogen_outputs` | input port | `not declared` | diagnostics.nitrogen_outputs mapped to canonical Phase 8 cochain basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `diagnostics.stoichiometry_mass_balance_error_carbon` | output port | `g C m-2` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.stoichiometry_mass_balance_error_nitrogen` | output port | `g N m-2` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.stoichiometry_mass_balance_error_total` | output port | `dimensionless` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |
| `diagnostics.stoichiometry_constraint_violations_count` | output port | `dimensionless` | diagnostics | same interface variable preserved between legacy and IPHS runtime contracts |

### Governing equations and canonical operator

**Literature or process form**

C/N balance with mass conservation

**Canonical RELIC structural form**

$y_{h} = O_{h} x_{h}$ through the declared observation operator

| Entry | Meaning |
| --- | --- |
| `O_h` | declared observation map |
| `H_h` | diagnostic metric or identity Hodge map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | declared RELIC state, port, or observation cochain basis |
| `effort_variable` | ParameterManager-backed thermodynamic effort, driver, or diagnostic input used by the module |
| `state_advancer` | $y_{h} = O_{h} x_{h}$ through the declared observation operator |
| `runtime_usage` | StoichiometryConstraint.forward applies the canonical observation or diagnostic operator and emits validation diagnostics through the production output path. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.stoichiometry.microbial_biomass_cn` | `dimensionless` | `8.0; bounds=(6.0, 12.0)` | `fixed` | C:N ratio of microbial biomass (relatively constant across ecosystems) | Source: Cleveland & Liptzin 2007 global mean; DOI:10.1111/j.1365-2486.2007.01386.x |
| `pedosphere.stoichiometry.metabolic_litter_cn` | `dimensionless` | `15.0; bounds=(10.0, 25.0)` | `fixed` | C:N ratio of metabolic litter (low lignin, high quality plant material like leaves and fine roots) | Source: CENTURY model default values; DOI:10.2136/sssaj1988.03615995005200030009x |
| `pedosphere.stoichiometry.structural_litter_cn` | `dimensionless` | `40.0; bounds=(25.0, 60.0)` | `fixed` | C:N ratio of structural litter (high lignin, low quality woody material) | Source: CENTURY model default values; DOI:10.2136/sssaj1988.03615995005200030009x |
| `pedosphere.stoichiometry.fast_som_cn` | `dimensionless` | `12.0; bounds=(8.0, 20.0)` | `fixed` | C:N ratio of fast (active) soil organic matter pool | Source: CENTURY model default values; DOI:10.2136/sssaj1988.03615995005200030009x |
| `pedosphere.stoichiometry.slow_som_cn` | `dimensionless` | `20.0; bounds=(12.0, 30.0)` | `learnable` | C:N ratio of slow (protected) soil organic matter pool - CORRECTED to match CENTURY model standard (Parton et al. 1987) | Source: CENTURY model parameterization (corrected from 15.0 to 20.0 in Phase 6.4); Parton et al. (1987) CENTURY model doi:10.2136/sssaj1987.03615995005100010015x; DOI:10.2136/sssaj1988.03615995005200030009x |
| `pedosphere.stoichiometry.passive_som_cn` | `dimensionless` | `25.0; bounds=(15.0, 35.0)` | `learnable` | C:N ratio of passive (recalcitrant) soil organic matter pool - CRITICAL FIX: Corrected to CENTURY model standard | Source: CENTURY model parameterization (corrected from 12.0 to 25.0 in Phase 6.4); Parton et al. (1987) CENTURY model doi:10.2136/sssaj1987.03615995005100010015x; Trumbore (2000) Radiocarbon evidence doi:10.1016/S0921-8181(99)00073-1 |
| `pedosphere.numerical.stoichiometric_flexibility` | `dimensionless` | `0.2; bounds=(0.0, 1.0)` | `fixed` | Allowed relative deviation from target C:N ratio before correction is applied | Source: stoichiometric constraint modeling; Stoichiometric constraint modeling |
| `pedosphere.numerical.stoichiometric_correction_strength` | `dimensionless` | `0.5; bounds=(0.0, 1.0)` | `fixed` | Strength of stoichiometric correction (0=no correction, 1=full correction to target) | Source: stoichiometric constraint modeling; Stoichiometric constraint modeling |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | existing RELIC simulator and StateManager update path |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\stoichiometry_constraint.md`.

### References

- Manzoni, S. & Porporato, A. (2009). Substrate-decomposer stoichiometry (FunctorMetadata, certificate)
- Cleveland, C.C. & Liptzin, D. (2007). C:N:P stoichiometry in soil (FunctorMetadata, certificate)
- Theory doc: docs/source/theory/stoichiometry_constraint.md (FunctorMetadata, certificate)
- Pielke, R.A. (2002). Mesoscale meteorological modeling. (FunctorMetadata)
- Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming. (FunctorMetadata)

(class-relic-scientific-pedosphere-transport-equilibrium-sorption-equilibriumsorptionfunctor)=
## EquilibriumSorptionFunctor

Index-1 equilibrium sorption DAE with implicit dissolved decay (PhysicalField contract).

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.transport.equilibrium_sorption.EquilibriumSorptionFunctor` |
| Source | `relic\scientific\pedosphere\transport\equilibrium_sorption.py:41` |
| Runtime name | `equilibrium_sorption` |
| Scientific abstraction | `PhysicalField` |
| Canonical bridge | `PhysicalFieldFunctor` |
| Structural role | `field` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_2` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
EquilibriumSorptionFunctor(name: str = 'equilibrium_sorption', *, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'equilibrium_sorption'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Mapping[str, Any]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `equilibrium_total_solute_mass` | state | `kg m-3` | differential state y = theta C + rho_b S [kg m-3]; input state and accumulated increment |
| `equilibrium_dissolved_concentration` | state | `kg m-3` | algebraic state C [kg m-3]; input state and full-state replacement each step |
| `equilibrium_sorbed_concentration` | state | `kg kg-1` | algebraic state S = K_f C^n [kg kg-1]; input state and full-state replacement each step |
| `soil_moisture` | state | `m3 m-3` | volumetric liquid soil water content theta [m3 m-3], the prognostic soil_moisture state, read and never written |
| `bulk_density` | forcing | `kg m-3` | dry bulk density rho_b [kg m-3] read as forcing (bulk_density contract) |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `equilibrium_total_solute_mass` | state | `kg m-3` | decided at run time by the returned write mode | differential state y = theta C + rho_b S [kg m-3]; input state and accumulated increment |
| `equilibrium_dissolved_concentration` | state | `kg m-3` | decided at run time by the returned write mode | algebraic state C [kg m-3]; input state and full-state replacement each step |
| `equilibrium_sorbed_concentration` | state | `kg kg-1` | decided at run time by the returned write mode | algebraic state S = K_f C^n [kg kg-1]; input state and full-state replacement each step |
| `diagnostics.equilibrium_sorption.constraint_residual` | diagnostic | `kg m-3` | diagnostics | abs(theta C' + rho_b S' - y') after the step [kg m-3] |
| `diagnostics.equilibrium_sorption.degradation_flux` | diagnostic | `kg m-3 s-1` | diagnostics | lambda theta C' [kg m-3 s-1] |
| `diagnostics.equilibrium_sorption.solver_iterations` | diagnostic | `1` | diagnostics | bisection iterations of the step [1] |

### Governing equations and canonical operator

**Literature or process form**

Freundlich equilibrium sorption with retardation (Jury and Horton 2004; Mayes et al. 2012) and first-order dissolved-phase degradation; semi-explicit index-1 DAE with algebraic elimination (Hairer and Wanner 1996)

**Canonical RELIC structural form**

$E \dot{x} = f ( x )$, $g ( x ) = 0$ with $x = ( y , C , S )$, $E = \operatorname{diag} ( 1 , 0 , 0 )$, $g = ( \theta C + \rho_{b} S - y , S - K_{f} C ^ n )$; $g_{z}$ invertible (index 1)

| Entry | Meaning |
| --- | --- |
| `constraint` | $g ( y , C ) = \theta C + \rho_{b} K_{f} C ^ n - y$ with $g_{C} = \theta + \rho_{b} K_{f} n C ^ ( n - 1 ) > 0 ( \mathrm{index} 1 )$ |
| `elimination` | scalar monotone equation $F ( C ' ) = \theta C ' + \rho_{b} K_{f} C ' ^ n + \mathrm{d}t \lambda \theta C ' - y \mathrm{solved}$ on the exact bracket [0, y/theta] by the existing BracketedBisectionSolver |
| `differential_update` | implicit Euler y' = y - dt lambda theta C' |
| `restart_projection` | the stored (y, C, S) triple is re-verified for consistency at every step, so a restored checkpoint is re-projected by the same check before advancing |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | differential total mass y on the cell basis with the algebraic pair (C, S) on the constraint manifold; no storage functional is claimed |
| `effort_variable` | none; the partition and isotherm are algebraic constraints, not variational efforts |
| `state_advancer` | y' = y - dt lambda theta C' with C' from the bracketed solve of the implicit constraint and S' = $K_{f}$ C'^n |
| `runtime_usage` | EquilibriumSorptionFunctor.forward validates states and water content, rejects inconsistent triples, solves the scalar constraint with BracketedBisectionSolver, raises on non-convergence, emits one ACCUMULATE increment, two REPLACE algebraic states, and three REPLACE diagnostics, and reports the total-mass ledger residual through check_conservation_laws. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.transport.equilibrium_sorption_freundlich_coefficient` | `m3 kg-1` | `0.002; bounds=(1e-06, 1.0)` | `learnable` | Freundlich coefficient K_f of the equilibrium isotherm S = K_f C^n with C in kg m-3 and S in kg kg-1 (generic sorbing solute) | Source: Sorption isotherm literature; Jury, W.A. & Horton, R. (2004). Soil Physics, 6th ed., Wiley, ch. 7.; Mayes, M.A., Heal, K.R., Brandt, C.C., Phillips, J.R. & Jardine, P.M. (2012). Soil Sci. Soc. Am. J., 76, 199-206. |
| `pedosphere.transport.equilibrium_sorption_freundlich_exponent` | `dimensionless` | `0.8; bounds=(0.3, 1.0)` | `fixed` | Freundlich exponent n of the equilibrium isotherm S = K_f C^n (1 is linear) | Source: Sorption isotherm literature; Jury, W.A. & Horton, R. (2004). Soil Physics, 6th ed., Wiley, ch. 7. |
| `pedosphere.transport.equilibrium_sorption_dissolved_decay_rate` | `s-1` | `1e-06; bounds=(0.0, 0.001)` | `learnable` | First-order degradation rate lambda of the dissolved phase of the sorbing solute | Source: Solute transport literature; Jury, W.A. & Horton, R. (2004). Soil Physics, 6th ed., Wiley, ch. 7. |
| `pedosphere.transport.equilibrium_sorption_consistency_tolerance` | `dimensionless` | `1e-09; bounds=(1e-14, 0.0001)` | `fixed` | Relative tolerance of the consistent-initialization and constraint-residual checks of the equilibrium sorption DAE | Source: DAE literature; Hairer, E. & Wanner, G. (1996). Solving Ordinary Differential Equations II, Springer, ch. VI (index-1 DAEs). |
| `computation.solvers.bracketed_bisection.atol` | `dimensionless` | `1e-10; bounds=(0.0, 0.0001)` | `fixed` | Absolute interval tolerance in the solved variable units | Source: computation.solvers.bracketed_bisection; Burden, R.L. and Faires, J.D. (2010). Numerical Analysis. |
| `computation.solvers.bracketed_bisection.rtol` | `dimensionless` | `1e-10; bounds=(0.0, 0.0001)` | `fixed` | Relative interval tolerance for bracketed scalar roots | Source: computation.solvers.bracketed_bisection; Burden, R.L. and Faires, J.D. (2010). Numerical Analysis. |
| `computation.solvers.bracketed_bisection.max_iterations` | `dimensionless` | `80; bounds=(1.0, 512.0)` | `fixed` | Maximum bracket contractions for independent scalar roots | Source: computation.solvers.bracketed_bisection; Burden, R.L. and Faires, J.D. (2010). Numerical Analysis. |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem ACCUMULATE with per_timestep true for the differential state and REPLACE for the algebraic states and diagnostics |
| `abstraction_contract` | PhysicalField |
| `bridge_class` | PhysicalFieldFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\pedosphere_equilibrium_sorption.md`.

### References

- Jury, W.A. & Horton, R. (2004). Soil Physics, 6th ed., Wiley, ch. 7 (FunctorMetadata, certificate)
- Mayes, M.A. et al. (2012). Soil Sci. Soc. Am. J. 76, 199-206 (FunctorMetadata)
- Hairer, E. & Wanner, G. (1996). Solving Ordinary Differential Equations II, ch. VI (FunctorMetadata)
- Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. (FunctorMetadata)
- Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy. (FunctorMetadata)
- Mayes, M.A., Heal, K.R., Brandt, C.C., Phillips, J.R. & Jardine, P.M. (2012). Relation between soil order and sorption of dissolved organic carbon in temperate subsoils. Soil Sci. Soc. Am. J. 76, 199-206 (certificate)
- Hairer, E. & Wanner, G. (1996). Solving Ordinary Differential Equations II, Springer, ch. VI (certificate)
- Burden, R.L. & Faires, J.D. (2010). Numerical Analysis (bracketed bisection) (certificate)

(class-relic-scientific-pedosphere-transport-gas-diffusion-gasdiffusion1d)=
## GasDiffusion1D

One-dimensional vertical gas diffusion in soil.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.transport.gas_diffusion.GasDiffusion1D` |
| Source | `relic\scientific\pedosphere\transport\gas_diffusion.py:117` |
| Runtime name | `GasDiffusion1D` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
GasDiffusion1D(n_layers: int | None = None, soil_depth: float | None = None, diffusivity_model: str | None = None, enable_temperature_correction: bool | None = None, enable_atmospheric_exchange: bool | None = None, device: torch.device | None = None)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `n_layers` | <code>int &#124; None</code> | `None` | Constructor control for n layers; see the class runtime contract and ParameterManager table below. |
| `soil_depth` | <code>float &#124; None</code> | `None` | Constructor control for soil depth; see the class runtime contract and ParameterManager table below. |
| `diffusivity_model` | <code>str &#124; None</code> | `None` | Constructor control for diffusivity model; see the class runtime contract and ParameterManager table below. |
| `enable_temperature_correction` | <code>bool &#124; None</code> | `None` | Enables or disables the temperature correction pathway. |
| `enable_atmospheric_exchange` | <code>bool &#124; None</code> | `None` | Enables or disables the atmospheric exchange pathway. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | soil_moisture mapped to canonical Phase 8 cochain basis |
| `porosity` | forcing | `m3 m-3` | porosity mapped to canonical Phase 8 cochain basis |
| `soil_temperature` | forcing | `K` | canonical soil temperature in K mapped to the Phase 8 cochain basis |
| `clay_content` | forcing | `1` | clay_content mapped to canonical Phase 8 cochain basis |
| `soil_co2_concentration` | state | `g C m-3` | canonical pedosphere CO2 state key mapped to the Phase 8 cochain basis; generic CO2 aliases are rejected |
| `n2o_concentration` | state | `g N m-3` | n2o_concentration mapped to canonical Phase 8 cochain basis |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `co2_surface_flux` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `n2o_surface_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `co2_flux_profile` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `n2o_flux_profile` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | same interface variable preserved between legacy and IPHS runtime contracts |
| `soil_co2_concentration` | state | `g C m-3` | decided at run time by the returned write mode | canonical pedosphere CO2 state key mapped to the Phase 8 cochain basis; generic CO2 aliases are rejected |
| `n2o_concentration` | state | `g N m-3` | decided at run time by the returned write mode | n2o_concentration mapped to canonical Phase 8 cochain basis |
| `diagnostics.gas_diffusion.energy_storage` | output port | `g2 m-6` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.gas_diffusion.dissipation_rate` | output port | `g2 m-5 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.gas_diffusion.dH_dt_audit` | output port | `g2 m-6 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.gas_diffusion.port_power_residual` | output port | `g2 m-5 s-1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.gas_diffusion.co2_clipping_residual` | output port | `g C m-3` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.gas_diffusion.n2o_clipping_residual` | output port | `g N m-3` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.gas_diffusion.step_seconds` | output port | `s` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

$$
J_z = - \mathrm{Deff} \frac{\partial C}{\partial z}; F_{\mathrm{surface},\mathrm{out}} = - J_z ( 0 )
$$

**Canonical RELIC structural form**

$$
\dot{c} = - D_{h} q_{h} + G_{h} u_{h}, q_{h} = - \star_{K} D_{h} ^{\mathsf{T}} e_{h}, e_{h} = \delta H_{h} / \delta c_{h}
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | existing gradient and flux-divergence operator used by forward() |
| `star_K` | HodgeStarSpec assembled from ParameterManager-backed CO2 and N2O effective diffusivities |
| `q_h` | soil gas diffusive flux cochain computed by Fick law |
| `G_h` | soil-atmosphere CO2 and N2O matter-port map |
| `H_h` | quadratic concentration storage $H_{h} = 0.5 \cdot c_{h}^{\mathsf{T}} c_{h}$ |
| `sigma_h` | non-negative diffusion dissipation $X_{h}^{\mathsf{T}} \star_{K} X_{h}$ |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | CO2 and N2O layer concentrations are declared as primal 0-cochains on pedosphere_soil_gas_diffusion |
| `effort_variable` | $e_{h} = \delta H_{h} / \delta c_{h} = c_{h}$ for stacked CO2 and N2O concentration cochains |
| `state_advancer` | $\delta C_{\mathrm{raw},i} = - ( J_{i} + 1 / 2 - J_{i} - 1 / 2 ) \frac{\mathrm{d}t}{\mathrm{d}z}$ with $F_{\mathrm{surface},\mathrm{out}} = - J_{\mathrm{top}}$ and $J_{\mathrm{bottom}} = 0$; $\delta C_{\mathrm{accepted},i} = \max ( - C_{i} , \delta C_{\mathrm{raw},i} )$ |
| `runtime_usage` | GasDiffusion1D.forward computes ParameterManager-backed effective diffusivity once, advances each exact-grid concentration through one conservative finite-volume divergence, and returns explicit UpdateItems plus signed species-specific clipping routes. |

### Composition channels and executable PortSpec API

Certificate composition channels: `phase8_runtime_port`

Executable `ports()` owner: `GasDiffusion1D`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `soil_atmosphere_co2_flux` | `soil_co2_surface_concentration [g C m-3]` | `co2_surface_flux [g C m-2 s-1]` | `g2 C2 m-5 s-1` | out_of_system; matter |
| `soil_atmosphere_n2o_flux` | `soil_n2o_surface_concentration [g N m-3]` | `n2o_surface_flux [g N m-2 s-1]` | `g2 N2 m-5 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.grid.n_layers` | `dimensionless` | `10; bounds=(1.0, 100.0)` | `fixed` | Default number of soil layers for 1D pedosphere transport and gas diffusion | Source: model_configuration; Campbell & Norman (1998) Environmental Biophysics |
| `pedosphere.grid.soil_depth` | `m` | `1.5; bounds=(0.1, 10.0)` | `fixed` | Total soil column depth used for pedosphere grids | Source: soil_physics_literature; Saxton & Rawls (2006) Soil Sci. Soc. Am. J. 70:1569-1578 |
| `pedosphere.transport.diffusion_model` | `dimensionless` | `'millington_quirk'; bounds=None` | `fixed` | Gas diffusion model selection for soil air-filled pore space. Options: millington_quirk (default), penman, dndc | Source: Gas diffusion model selection; Millington & Quirk (1961) doi:10.1097/00010694-196111000-00003; Penman (1940) Journal of Agricultural Science |
| `pedosphere.transport.temperature_correction` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Enable temperature-dependent gas diffusivity correction (1.0=enabled, 0.0=disabled) | Source: Gas diffusion temperature correction protocol; Fuller et al. (1966) Temperature dependence doi:10.1021/ie50677a007; Massman (1998) doi:10.1016/S0168-1923(98)00105-0 |
| `pedosphere.transport.atmospheric_exchange` | `dimensionless` | `1.0; bounds=(0.0, 1.0)` | `fixed` | Enable atmospheric gas exchange at soil surface (1.0=enabled, 0.0=disabled, zero-gradient BC) | Source: Soil-atmosphere coupling configuration; Boundary condition protocol for soil-atmosphere gas exchange |
| `pedosphere.transport.allow_constructor_overrides` | `dimensionless` | `0.0; bounds=(0.0, 1.0)` | `fixed` | Allow constructor overrides for pedosphere transport functors (1.0 enabled, 0.0 disabled). | Source: Transport configuration governance; Parameter governance policy for transport configuration overrides |
| `pedosphere.transport.n2o_dndc_clay_effect` | `dimensionless` | `0.124; bounds=(0.05, 0.25)` | `learnable` | Dimensionless clay coefficient in the DNDC N2O diffusion factor | Source: DNDC N2O diffusion formulation; Li et al. (1992) DNDC model structure doi:10.1029/92JD00509; Li et al. (2000) DNDC parameterization doi:10.1016/S0304-3800(00)00248-9 |
| `pedosphere.transport.n2o_dndc_porosity_effect` | `dimensionless` | `0.013; bounds=(0.005, 0.025)` | `learnable` | Dimensionless DNDC porosity coefficient in the N2O diffusion factor | Source: DNDC N2O diffusion formulation; Li et al. (1992) DNDC model structure doi:10.1029/92JD00509; Li et al. (2000) DNDC parameterization doi:10.1016/S0304-3800(00)00248-9 |
| `pedosphere.transport.n2o_dndc_base_coefficient` | `dimensionless` | `0.018; bounds=(0.01, 0.05)` | `learnable` | Dimensionless base coefficient in the DNDC N2O diffusion factor | Source: DNDC N2O diffusion formulation; Li et al. (1992) DNDC model structure doi:10.1029/92JD00509; Li et al. (2000) DNDC parameterization doi:10.1016/S0304-3800(00)00248-9 |
| `pedosphere.transport.n2o_dndc_clay_porosity_coefficient` | `dimensionless` | `0.016; bounds=(0.01, 0.03)` | `learnable` | DNDC model clay-porosity interaction coefficient for temperature-dependent N2O diffusion | Source: DNDC N2O diffusion formulation; Li et al. (1992) DNDC model structure doi:10.1029/92JD00509; Li et al. (2000) DNDC parameterization doi:10.1016/S0304-3800(00)00248-9 |
| `pedosphere.transport.n2o_dndc_temperature_base` | `dimensionless` | `2.0; bounds=(1.5, 3.0)` | `learnable` | Dimensionless base for the DNDC N2O exponential temperature factor | Source: DNDC N2O diffusion formulation; Li et al. (1992) DNDC model structure doi:10.1029/92JD00509; Li et al. (2000) DNDC parameterization doi:10.1016/S0304-3800(00)00248-9 |
| `pedosphere.transport.n2o_dndc_temperature_divisor` | `K` | `20.0; bounds=(10.0, 30.0)` | `learnable` | Temperature interval divisor in the DNDC N2O exponential factor | Source: DNDC N2O diffusion formulation; Li et al. (1992) DNDC model structure doi:10.1029/92JD00509; Li et al. (2000) DNDC parameterization doi:10.1016/S0304-3800(00)00248-9 |
| `pedosphere.physical.co2_diffusivity_air_stp` | `m2 s-1` | `1.47e-05; bounds=(1.45e-05, 1.5e-05)` | `fixed` | Molecular diffusivity of CO2 in free air at standard temperature and pressure | Source: RELIC parameter database; Massman (1998) doi:10.1016/S0168-1923(98)00105-0; Marrero & Mason (1972) doi:10.1063/1.3253094 |
| `pedosphere.physical.n2o_diffusivity_air_stp` | `m2 s-1` | `1.32e-05; bounds=(1.1e-05, 1.5e-05)` | `fixed` | Molecular diffusion coefficient of N2O in air at standard temperature and pressure. | Source: RELIC parameter database; Massman (1998) Agricultural and Forest Meteorology 90:57-75; Li et al. (1992) Journal of Geophysical Research 97:9759-9776 |
| `pedosphere.physical.gas_diffusivity_temperature_exponent` | `dimensionless` | `1.75; bounds=(1.5, 2.0)` | `fixed` | Temperature exponent n in D0(T)=D0(T0)*(T/T0)^n used for soil gas diffusivity scaling. | Source: RELIC parameter database; Millington & Quirk (1961) Trans. Faraday Soc. 57:1200-1207; Davidson et al. (2012) Global Change Biology 18:371-384 |
| `pedosphere.physical.atmospheric_co2_concentration` | `g C m-3` | `0.21434846649719208; bounds=(0.18755490818504308, 0.2411420248093411)` | `fixed` | Elemental carbon mass concentration equivalent to 400 ppm CO2 at standard pressure and temperature. | Source: atmospheric_monitoring_records; NOAA ESRL Global Monitoring Laboratory (2024); WMO Greenhouse Gas Bulletin (2024) |
| `pedosphere.physical.atmospheric_n2o_concentration` | `g N m-3` | `0.0004199481034035439; bounds=(0.00039995057467004185, 0.00043744594104535824)` | `fixed` | Elemental nitrogen mass concentration equivalent to 0.336 ppm N2O at standard pressure and temperature. | Source: atmospheric_monitoring_records; NOAA ESRL Global Monitoring Laboratory (2024); IPCC AR6 WGI Chapter 5 |
| `pedosphere.physical.millington_quirk_exponent_numerator` | `dimensionless` | `3.333333; bounds=(3.0, 3.5)` | `fixed` | Numerator exponent (10/3) controlling tortuosity in the Millington-Quirk gas diffusivity model. | Source: RELIC parameter database; Millington & Quirk (1961) Trans. Faraday Soc. 57:1200-1207; Moldrup et al. (2000) Soil Sci. Soc. Am. J. 64:94-100 |
| `pedosphere.physical.millington_quirk_exponent_denominator` | `dimensionless` | `2.0; bounds=(1.5, 2.5)` | `fixed` | Denominator exponent applied to total porosity in the Millington-Quirk tortuosity expression. | Source: RELIC parameter database; Millington & Quirk (1961) Trans. Faraday Soc. 57:1200-1207; Penman (1940) J. Agric. Sci. 30:437-462 |
| `pedosphere.physical.penman_coefficient` | `dimensionless` | `0.66; bounds=(0.5, 0.8)` | `fixed` | Empirical coefficient used in the Penman gas diffusivity formulation (Deff = D0 * coeff * theta_a). | Source: RELIC parameter database; Penman, H.L. (1940) J. Agric. Sci. 30:437-462; Moldrup et al. (2000) Soil Sci. Soc. Am. J. 64:94-100 |
| `pedosphere.numerical.clamp_min` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Minimum clamp value for numerical stability in rates and fluxes | Source: standard numerical practice; Standard numerical practice |
| `pedosphere.numerical.epsilon` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-08)` | `fixed` | Small value added to denominators to prevent division by zero | Source: numerical stability; Numerical stability |
| `pedosphere.numerical.epsilon_length` | `m` | `1e-06; bounds=(1e-09, 0.01)` | `fixed` | Minimum length scale to avoid zero layer thickness | Source: numerical stability; Numerical stability |
| `pedosphere.numerical.min_time_step` | `s` | `60.0; bounds=(1.0, 3600.0)` | `fixed` | Minimum allowed time step | Source: computational efficiency; Computational efficiency |
| `pedosphere.numerical.zero_initial` | `dimensionless` | `0.0; bounds=(0.0, 0.0)` | `fixed` | Zero initialization value for numerical stability and default operations | Source: numerical initialization protocol; Numerical initialization protocol |
| `pedosphere.numerical.wfps_minimum` | `dimensionless` | `0.0; bounds=(0.0, 0.1)` | `fixed` | Minimum water-filled pore space fraction for numerical stability in anaerobic process calculations | Source: Numerical stability requirements for water-filled pore space; WFPS-based denitrification models; Davidson et al. (2000) doi:10.1046/j.1365-2486.2000.00327.x |
| `pedosphere.numerical.wfps_maximum` | `dimensionless` | `1.0; bounds=(0.9, 1.0)` | `fixed` | Maximum water-filled pore space fraction, represents complete saturation | Source: Physical constraint for water-filled pore space; Physical limit for WFPS calculation; Linn & Doran (1984) doi:10.2136/sssaj1984.03615995004800060013x |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | concentration states use ACCUMULATE per_timestep updates; instantaneous fluxes and diagnostics use REPLACE |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\pedosphere_transport.md`.

### References

- Millington, R.J. & Quirk, J.P. (1961). Trans. Faraday Soc. 57, 1200-1207 (FunctorMetadata)
- Penman, H.L. (1940). J. Agric. Sci. 30, 437-462 (FunctorMetadata, certificate)
- Moldrup, P. et al. (2000). Soil Sci. Soc. Am. J. 64, 94-100 (FunctorMetadata, certificate)
- Li, C. et al. (1992). J. Geophys. Res. 97, 9759-9776 (FunctorMetadata)
- Song, H. et al. (2021). GMD discussion supplement, Table S1, Eq. 20 (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Millington, R.J. & Quirk, J.P. (1961). Trans. Faraday Soc. 57, 1200-1207. doi:10.1039/TF9615701200 (certificate)
- Li, C. et al. (1992). J. Geophys. Res. 97, 9759-9776. doi:10.1029/92JD00509 (certificate)
- Kusa, K. et al. (2008). Soil Sci. Plant Nutr. 54, 777-785. doi:10.1111/j.1747-0765.2008.00292.x (certificate)
- Song, H. et al. (2021). GMD discussion supplement, Table S1, Eq. 20. doi:10.5194/gmd-2021-23-supplement (certificate)

(class-relic-scientific-pedosphere-transport-moving-domain-diffusion-movingdomaindiffusionfunctor)=
## MovingDomainDiffusionFunctor

ALE finite-volume diffusion on a uniformly stretched column (TransportMechanism contract).

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.transport.moving_domain_diffusion.MovingDomainDiffusionFunctor` |
| Source | `relic\scientific\pedosphere\transport\moving_domain_diffusion.py:41` |
| Runtime name | `moving_domain_diffusion` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
MovingDomainDiffusionFunctor(name: str = 'moving_domain_diffusion', *, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'moving_domain_diffusion'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Mapping[str, Any]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `moving_domain_solute_concentration` | state | `kg m-3` | cell concentration cochain c on the stretched column [kg m-3]; input state and accumulated increment |
| `moving_domain_length` | state | `m` | column length L, the geometry state of the mapping x = xi L [m]; input state and accumulated increment w_b dt |
| `moving_domain_boundary_velocity` | forcing | `m s-1` | prescribed lower-boundary velocity w_b [m s-1] (forcing) |
| `moving_boundary_solute_concentration` | forcing | `kg m-3` | concentration of fluid entrained by a growing boundary c_b [kg m-3] (forcing) |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `moving_domain_solute_concentration` | state | `kg m-3` | decided at run time by the returned write mode | cell concentration cochain c on the stretched column [kg m-3]; input state and accumulated increment |
| `moving_domain_length` | state | `m` | decided at run time by the returned write mode | column length L, the geometry state of the mapping x = xi L [m]; input state and accumulated increment w_b dt |
| `diagnostics.moving_domain.boundary_mass_flux` | diagnostic | `kg m-2 s-1` | diagnostics | swept mass flux s_N = c_face,N w_b through the moving face [kg m-2 s-1] |
| `diagnostics.moving_domain.mass_balance_residual` | diagnostic | `kg m-2` | diagnostics | sum_i m_i(t+dt) - sum_i m_i(t) - dt s_N [kg m-2] |
| `diagnostics.moving_domain.gcl_residual` | diagnostic | `m` | diagnostics | max_i \\|V_i(t+dt) - V_i(t) - dt (w_{i+1} - w_i)\\| [m] |
| `diagnostics.moving_domain.geometry_power` | diagnostic | `kg2 m-5 s-1` | diagnostics | -0.5 c^T dot(star) c [kg2 m-5 s-1] |

### Governing equations and canonical operator

**Literature or process form**

$$
ALE conservation form with relative flux F - c w (Donea et al. 2004) and the discrete geometric conservation law V_i(t+dt) = V_i(t) + dt (w_{i+1/2} - w_{i-1/2}) (Thomas and Lombard 1979); Fickian diffusion (Crank 1975)
$$

**Canonical RELIC structural form**

$\dot{x} = - R_{h} e + f_{\mathrm{geo}} + s_{\mathrm{boundary}}$ with $e = c$, $R_{h} = d_{h} ^{\mathsf{T}} \operatorname{diag} ( D ) d_{h} / \mathrm{d}x ^ 2$, e^$T f_{\mathrm{geo}} = - 0.5 c ^{\mathsf{T}} \cdot ( \mathrm{star} ) c$, $\cdot ( \mathrm{star} ) = \operatorname{diag} ( w_{b} / N )$

| Entry | Meaning |
| --- | --- |
| `diffusive_rate` | canonical _compute_diffusion_rate_and_boundary_flux_1d with zero-flux closed faces on the current metric $\mathrm{d}x = L / N$ |
| `relative_flux` | $s_{k} = c_{\mathrm{face},k} w_{k}$ with the face concentration upwind of the face motion ($c_{b}$ outside a growing boundary; bottom-cell value for a shrinking boundary) |
| `geometric_conservation` | $V_i(t+dt) = L(t+dt)/N = V_i(t) + dt (w_{i+1} - w_i) exactly$ for the uniform stretching mapping |
| `hodge_time_derivative` | $\cdot ( \mathrm{star} ) = \operatorname{diag} ( w_{b} / N )$ on the concentration cochain; the geometry power is emitted as a diagnostic |
| `mapping` | uniform stretching $x = \mathrm{xi} L ( t )$; refinement, non-uniform mapping, and remap are [N] |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | concentration cochain c with metric $\mathrm{star} = \operatorname{diag} ( L / N )$; storage $H = 0.5 c ^{\mathsf{T}} \mathrm{star} c$ |
| `effort_variable` | $e = c$ |
| `state_advancer` | $m_i(t+dt) = m_i(t) + dt [rate_i dx + s_{i+1} - s_i]$, $c_{i} ( t + \mathrm{d}t ) = m_{i} ( t + \mathrm{d}t ) / V_{i} ( t + \mathrm{d}t )$, $L ( t + \mathrm{d}t ) = L + w_{b} \mathrm{d}t$ |
| `runtime_usage` | MovingDomainDiffusionFunctor.forward validates states, forcings, and the step, enforces the diffusion and mesh Courant limits, evaluates the diffusive rate with the canonical owner, adds the relative fluxes, emits two ACCUMULATE increments and four REPLACE diagnostics, and reports the column ledger residual through check_conservation_laws. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.transport.moving_domain_diffusivity` | `m2 s-1` | `1e-09; bounds=(1e-12, 0.0001)` | `learnable` | Effective solute diffusivity D of the moving-domain (ALE) column diffusion process | Source: Solute transport literature; Crank, J. (1975). The Mathematics of Diffusion. Oxford University Press.; Jury, W.A. & Horton, R. (2004). Soil Physics, 6th ed., Wiley. |
| `pedosphere.transport.moving_domain_mesh_courant_max` | `dimensionless` | `0.9; bounds=(0.1, 0.99)` | `fixed` | Maximum mesh Courant number \\|w_b\\| dt / dx admitted by the explicit ALE moving-domain update | Source: ALE literature; Thomas, P.D. & Lombard, C.K. (1979). AIAA J., 17, 1030-1037.; Donea, J., Huerta, A., Ponthot, J.-Ph. & Rodriguez-Ferran, A. (2004). Arbitrary Lagrangian-Eulerian Methods, Encyclopedia of Computational Mechanics, ch. 14. |
| `transport.numerical.cfl_diffusion_max` | `dimensionless` | `0.5; bounds=(0.25, 0.5)` | `fixed` | Maximum allowable CFL number for diffusion (Von Neumann condition) | Source: RELIC transport module; Von Neumann and Richtmyer (1950) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem ACCUMULATE with per_timestep true for concentration and length and REPLACE observations for diagnostics |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\pedosphere_moving_domain_diffusion.md`.

### References

- Thomas, P.D. & Lombard, C.K. (1979). AIAA J. 17, 1030-1037 (FunctorMetadata)
- Donea, J. et al. (2004). ALE methods, Encyclopedia of Computational Mechanics (FunctorMetadata)
- Crank, J. (1975). The Mathematics of Diffusion (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Thomas, P.D. & Lombard, C.K. (1979). Geometric conservation law and its application to flow computations on moving grids. AIAA J. 17, 1030-1037 (certificate)
- Donea, J., Huerta, A., Ponthot, J.-Ph. & Rodriguez-Ferran, A. (2004). Arbitrary Lagrangian-Eulerian methods. Encyclopedia of Computational Mechanics, ch. 14 (certificate)
- Crank, J. (1975). The Mathematics of Diffusion. Oxford University Press (certificate)
- Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow. Hemisphere (certificate)

(class-relic-scientific-pedosphere-transport-soilwatertransportfunctor-soilwatertransportfunctor)=
## soil_water_transport

Soil water transport functor for Richards equation and boundary fluxes.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.transport.SoilWaterTransportFunctor.SoilWaterTransportFunctor` |
| Source | `relic\scientific\pedosphere\transport\SoilWaterTransportFunctor.py:48` |
| Runtime name | `soil_water_transport` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
soil_water_transport(config: dict | None = None, device: torch.device | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `config` | <code>dict &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | required cellwise volumetric-water-content profile and absolute StateManager full state theta_(n+1) |
| `pressure_head` | forcing | `m` | effort potential h on primal 0-cochains |
| `infiltration_flux` | flux | `kg m-2 s-1` | required surface matter-port inflow [kg m-2 s-1] |
| `evapotranspiration_flux` | flux | `kg m-2 s-1` | required surface matter-port outflow [kg m-2 s-1] |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `soil_moisture` | state | `m3 m-3` | decided at run time by the returned write mode | required cellwise volumetric-water-content profile and absolute StateManager full state theta_(n+1) |
| `diagnostics.soil_water_transport.clipping_residual` | output port | `m3 m-3` | diagnostics | theta_raw - clamp(theta_raw, wilting_point, porosity) |
| `diagnostics.soil_water_transport.water_balance_residual` | output port | `m` | diagnostics | Delta W - q_surface dt + q_bottom dt |
| `diagnostics.soil_water_transport.darcy_flux` | output port | `m s-1` | diagnostics | downward-positive interface Darcy flux profile |
| `diagnostics.soil_water_transport.richards_iterations` | output port | `dimensionless` | diagnostics | successful Picard iteration count |

### Governing equations and canonical operator

**Literature or process form**

Mixed-form Richards water balance with Darcy flux $q = - K ( \theta ) \nabla ( h + z )$ and storage tendency $\frac{\mathrm{d}\theta}{\mathrm{d}t} = - \operatorname{div} ( q ) + S .$

**Canonical RELIC structural form**

$$
\dot{\theta} = - D_{h} q_{h} + G_{h} u_{h}; q_{h} = - \star_{K} d ( h + z ); u_{h} = ( \mathrm{infiltration}_{\mathrm{flux}} - \mathrm{evapotranspiration}_{\mathrm{flux}} ) / \mathrm{water}_{\mathrm{density}}; \star_{K} = \star_{K}^{\mathsf{T}} \geq 0 .
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | existing one-dimensional finite-difference divergence used by the shared RichardsNumericalSolver modified-Picard correction |
| `star_K` | build_hodge_contract and build_hodge_terms HodgeStarSpec with saturated_conductivity |
| `source_port` | $( \mathrm{infiltration}_{\mathrm{flux}} - \mathrm{evapotranspiration}_{\mathrm{flux}} ) / \mathrm{water}_{\mathrm{density}}$ |
| `boundary_operator` | one-dimensional Richards boundary handling with $q_{\mathrm{bottom}} = K ( \theta_{\mathrm{bottom}} ) g_{z}$ for free drainage and explicit surface/drainage PortSpecIPHS declarations |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | cellwise volumetric-water-content profile theta_h; column storage is derived with dz |
| `effort_variable` | hydraulic-head gradient $X_{h} = d ( h + z )$ with conductivity mobility star_K |
| `state_advancer` | $\dot{\theta} = - D_{h} q_{h} + G_{h} u_{h}$; $q_{h} = - \star_{K} X_{h}$ |
| `runtime_usage` | _compute_iphs_richards_state_update applies the shared canonical mixed-residual correction, accepts only configured residual or verified machine-precision convergence, and routes mass residual diagnostics. |

### Composition channels and executable PortSpec API

Certificate composition channels: `surface_water_flux_boundary`, `bottom_drainage_boundary`

Executable `ports()` owner: `SoilWaterTransportFunctor`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `surface_water_flux_boundary` | `surface_hydraulic_head [m]` | `surface_water_flux [m s-1]` | `m2 s-1` | into_system; matter |
| `bottom_drainage_boundary` | `bottom_hydraulic_head [m]` | `bottom_water_flux [m s-1]` | `m2 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.grid.dz` | `m` | `0.05; bounds=(0.01, 0.2)` | `fixed` | Default vertical layer thickness used for 1D Richards columns near the surface | Source: pedosphere_transport_defaults; Hillel (1998) Environmental Soil Physics; Maxwell & Miller (2005) Advances in Water Resources 28:273-284 |
| `pedosphere.hydraulic.saturated_conductivity` | `m s-1` | `1e-05; bounds=(1e-07, 0.0001)` | `fixed` | Saturated hydraulic conductivity (Ks) for representative loam soil used by Richards solver | Source: pedosphere_hydraulics_refactor; Clapp & Hornberger (1978) Water Resources Research 14:601-604; Saxton & Rawls (2006) Soil Science Society of America Journal 70:1569-1578 |
| `pedosphere.moisture.wilting_point` | `m3 m-3` | `0.15; bounds=(0.05, 0.25)` | `fixed` | Permanent wilting point (theta_wp) for the same soil texture class | Source: pedosphere_hydraulics_refactor; Clapp & Hornberger (1978) Water Resources Research 14:601-604; Saxton & Rawls (2006) Soil Science Society of America Journal 70:1569-1578 |
| `pedosphere.soil.porosity` | `m3 m-3` | `0.45; bounds=(0.3, 0.55)` | `fixed` | Total soil porosity for the active rooting layer, consistent with loam pedotransfer sets | Source: pedosphere_hydraulics_refactor; Clapp & Hornberger (1978) Water Resources Research 14:601-604; Schaap & van Genuchten (2006) Vadose Zone Journal 5:1079-1085 |
| `pedosphere.van_genuchten.alpha_vg` | `m-1` | `3.6; bounds=(1.0, 10.0)` | `fixed` | van Genuchten alpha parameter (1/air-entry suction) for loam soils | Source: pedosphere_hydraulics_refactor; van Genuchten (1980) Soil Science Society of America Journal 44:892-898; Carsel & Parrish (1988) Water Resources Research 24:755-769 (class means as tabulated in van Genuchten et al. 1991, EPA/600/2-91/065, Table 4) |
| `pedosphere.van_genuchten.n_vg` | `dimensionless` | `1.56; bounds=(1.2, 2.2)` | `fixed` | van Genuchten n parameter controlling pore size distribution | Source: pedosphere_hydraulics_refactor; van Genuchten (1980) Soil Science Society of America Journal 44:892-898 |
| `pedosphere.transport.gravity_head_gradient` | `dimensionless` | `1.0; bounds=(0.9, 1.1)` | `fixed` | Vertical head gradient term in Darcy-Buckingham flux (dz/dz) for 1D soil water flow. | Source: Pedosphere transport numerical configuration; Darcy (1856); Richards (1931) doi:10.1063/1.1745010 |
| `transport.numerical.min_timestep` | `s` | `1e-06; bounds=(1e-09, 0.001)` | `fixed` | Minimum allowable timestep to prevent underflow and numerical issues | Source: RELIC transport module; Numerical stability requirement |
| `transport.numerical.richards_tolerance` | `dimensionless` | `1e-06; bounds=(1e-08, 0.0001)` | `fixed` | Relative residual tolerance for the implicit Richards solver used in pedosphere transport | Source: pedosphere_transport_defaults; Celia et al. (1990) Water Resources Research 26:1483-1496; Maxwell & Miller (2005) Advances in Water Resources 28:273-284 |
| `transport.numerical.max_richards_iterations` | `dimensionless` | `50; bounds=(10.0, 200.0)` | `fixed` | Maximum Newton iterations allowed per Richards solve before falling back to smaller timesteps | Source: pedosphere_transport_defaults; Celia et al. (1990) Water Resources Research 26:1483-1496 |
| `pedosphere.numerical.cfl_number` | `dimensionless` | `0.5; bounds=(0.1, 0.9)` | `fixed` | CFL number for stability | Source: numerical methods; Numerical methods |
| `pedosphere.numerical.epsilon_length` | `m` | `1e-06; bounds=(1e-09, 0.01)` | `fixed` | Minimum length scale to avoid zero layer thickness | Source: numerical stability; Numerical stability |
| `pedosphere.numerical.epsilon_diffusivity` | `dimensionless` | `1e-12; bounds=(1e-15, 1e-10)` | `fixed` | Epsilon for diffusivity calculations (smaller for precise flux computation) | Source: numerical stability for diffusion; Numerical stability for diffusion |
| `pedosphere.numerical.water_capacity_epsilon` | `dimensionless` | `1e-06; bounds=(1e-10, 0.0001)` | `fixed` | Epsilon for available water capacity calculations | Source: numerical stability for water capacity; Numerical stability for water capacity |
| `pedosphere.numerical.harmonic_mean_factor` | `dimensionless` | `2.0; bounds=(1.0, 2.0)` | `fixed` | Harmonic mean scaling for interface conductivity in Richards discretization | Source: pedosphere_transport_defaults; Celia et al. (1990) Water Resources Research 26:1483-1496; Hillel (1998) Environmental Soil Physics |
| `pedosphere.numerical.linear_solver_tolerance` | `dimensionless` | `1e-08; bounds=(1e-12, 1e-06)` | `fixed` | Linear solver tolerance | Source: linear algebra; Linear algebra |
| `pedosphere.numerical.under_relaxation_factor` | `dimensionless` | `1.0; bounds=(0.1, 1.0)` | `fixed` | Under-relaxation factor for stability | Source: numerical stability; Numerical stability |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | soil_moisture uses REPLACE with $\mathrm{per}_{\mathrm{timestep}} = \mathrm{false}$ and contains the absolute theta_(n+1) full state |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\soil_hydraulics.md`.

### References

- Darcy, H. (1856). Les fontaines publiques de la ville de Dijon. (FunctorMetadata, certificate)
- Richards, L.A. (1931). Capillary conduction of liquids through porous mediums. (FunctorMetadata)
- Celia, M.A. et al. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. (FunctorMetadata)
- van Genuchten, M.Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. (FunctorMetadata)
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. (FunctorMetadata)
- Hirani, A.N. (2003). Discrete Exterior Calculus. (FunctorMetadata, certificate)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Richards, L.A. (1931). Capillary conduction of liquids through porous mediums. Physics, 1, 318-333. (certificate)
- Celia, M.A., Bouloutas, E.T., and Zarba, R.L. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. Water Resources Research, 26, 1483-1496. (certificate)
- van Genuchten, M.Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. Soil Science Society of America Journal, 44, 892-898. (certificate)
- Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. Water Resources Research, 12, 513-522. (certificate)

(class-relic-scientific-pedosphere-transport-solute-advection-diffusion-soluteadvectiondiffusion1d)=
## SoluteAdvectionDiffusion1D

Finite-volume transport closure for aqueous NO3 and DOC.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.transport.solute_advection_diffusion.SoluteAdvectionDiffusion1D` |
| Source | `relic\scientific\pedosphere\transport\solute_advection_diffusion.py:59` |
| Runtime name | `SoluteAdvectionDiffusion1D` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `flux` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
SoluteAdvectionDiffusion1D(n_layers: Optional[int] = None, soil_depth: Optional[float] = None, layer_thickness_ratio: Optional[float] = None, cfl_number: Optional[float] = None, max_timestep: Optional[float] = None, enable_sorption: bool = True)
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `n_layers` | `Optional[int]` | `None` | Constructor control for n layers; see the class runtime contract and ParameterManager table below. |
| `soil_depth` | `Optional[float]` | `None` | Constructor control for soil depth; see the class runtime contract and ParameterManager table below. |
| `layer_thickness_ratio` | `Optional[float]` | `None` | Constructor control for layer thickness ratio; see the class runtime contract and ParameterManager table below. |
| `cfl_number` | `Optional[float]` | `None` | Constructor control for cfl number; see the class runtime contract and ParameterManager table below. |
| `max_timestep` | `Optional[float]` | `None` | Constructor control for max timestep; see the class runtime contract and ParameterManager table below. |
| `enable_sorption` | `bool` | `True` | Enables or disables the sorption pathway. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `soil_water_content` | forcing | `m3 m-3` | soil_water_content mapped to canonical Phase 8 cochain basis |
| `darcy_flux` | forcing | `m s-1` | darcy_flux mapped to canonical Phase 8 cochain basis, positive downward |
| `no3_concentration` | state | `g N m-3` | no3_concentration mapped to canonical Phase 8 cochain basis |
| `doc_concentration` | state | `g C m-3` | doc_concentration mapped to canonical Phase 8 cochain basis |
| `bulk_density` | forcing | `kg m-3` | bulk_density mapped to canonical Phase 8 cochain basis |
| `porosity` | forcing | `m3 m-3` | porosity mapped to canonical Phase 8 cochain basis |
| `soil_temperature` | forcing | `K` | soil_temperature controls ParameterManager-backed molecular diffusivity scaling |
| `surface_no3_concentration` | forcing | `g N m-3` | surface_no3_concentration is the open top NO3 Cauchy boundary input |
| `surface_doc_concentration` | forcing | `g C m-3` | surface_doc_concentration is the open top DOC Cauchy boundary input |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `nitrogen_leaching_flux` | flux | `g N m-2 s-1` | decided at run time by the returned write mode | canonical bottom NO3 export |
| `carbon_leaching_flux` | flux | `g C m-2 s-1` | decided at run time by the returned write mode | canonical bottom DOC export |
| `no3_concentration` | state | `g N m-3` | decided at run time by the returned write mode | no3_concentration mapped to canonical Phase 8 cochain basis |
| `doc_concentration` | state | `g C m-3` | decided at run time by the returned write mode | doc_concentration mapped to canonical Phase 8 cochain basis |
| `diagnostics.solute_transport.nitrogen_surface_input_flux` | output port | `g N m-2 s-1` | diagnostics | explicit top NO3 input observation |
| `diagnostics.solute_transport.carbon_surface_input_flux` | output port | `g C m-2 s-1` | diagnostics | explicit top DOC input observation |
| `diagnostics.solute_transport.nitrogen_mass_balance_residual` | output port | `g N m-2` | diagnostics | explicit NO3 finite-volume ledger residual |
| `diagnostics.solute_transport.carbon_mass_balance_residual` | output port | `g C m-2` | diagnostics | explicit DOC finite-volume ledger residual |

### Governing equations and canonical operator

**Literature or process form**

$\frac{\partial(B C)}{\partial t} = -\frac{\partial J}{\partial z}$, $B=\theta+\rho_bK_d$, $K=\alpha_L\lvert q\rvert+D_w(T)\theta^{10/3}/\phi^2$

**Canonical RELIC structural form**

$B_{h} \dot{c}_{h} = - D_{h} J_{h} + G_{h} u_{h}$ with declared storage, solute flux, and top/bottom matter ports

| Entry | Meaning |
| --- | --- |
| `D_h` | finite-volume incidence/divergence map on the nonuniform vertical grid |
| `star_D` | level-3 structural PSD Hodge initialized by the ParameterManager-backed default molecular diffusivity; the full runtime K coefficient is evaluated directly by forward() |
| `q_h` | $J=qC_{upwind}-K\,\frac{\mathrm{d}C}{\mathrm{d}z}$ evaluated once at every finite-volume face |
| `G_h` | explicit top Cauchy input and bottom free-drainage export matter-port map |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | NO3 and DOC layer concentrations are declared as primal 0-cochains on pedosphere_solute_transport |
| `effort_variable` | The level 3 structural audit uses concentration cochains and a positive storage capacity B; it does not claim a thermodynamic effort or strict port-power pairing |
| `state_advancer` | $B_iC_i^{n+1}=B_iC_i^n-\Delta t(J_{i+1/2}-J_{i-1/2})/\Delta z_i$ with explicit top and bottom ports |
| `runtime_usage` | SoluteAdvectionDiffusion1D.forward evaluates one nonuniform-grid finite-volume transport law with capacity-weighted CFL substeps; it contains no reaction branch, second operator path, or concentration fallback. |

### Composition channels and executable PortSpec API

Certificate composition channels: `solute_surface_inflow`, `solute_bottom_leaching`

Executable `ports()` owner: `SoluteAdvectionDiffusion1D`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `solute_surface_inflow` | `surface_solute_concentration [g m-3]` | `solute_surface_input_flux [g m-2 s-1]` | `g2 m-5 s-1` | into_system; matter |
| `solute_bottom_leaching` | `bottom_solute_concentration [g m-3]` | `bottom_leaching_flux [g m-2 s-1]` | `g2 m-5 s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.grid.n_layers` | `dimensionless` | `10; bounds=(1.0, 100.0)` | `fixed` | Default number of soil layers for 1D pedosphere transport and gas diffusion | Source: model_configuration; Campbell & Norman (1998) Environmental Biophysics |
| `pedosphere.grid.soil_depth` | `m` | `1.5; bounds=(0.1, 10.0)` | `fixed` | Total soil column depth used for pedosphere grids | Source: soil_physics_literature; Saxton & Rawls (2006) Soil Sci. Soc. Am. J. 70:1569-1578 |
| `pedosphere.transport.cfl_number` | `dimensionless` | `0.5; bounds=(0.1, 0.9)` | `fixed` | Courant-Friedrichs-Lewy (CFL) number for numerical stability in advection-diffusion solver | Source: Numerical analysis literature; Courant et al. (1928) Mathematische Annalen; LeVeque (2002) Finite Volume Methods for Hyperbolic Problems |
| `pedosphere.transport.layer_thickness_ratio` | `dimensionless` | `1.5; bounds=(1.0, 3.0)` | `fixed` | Geometric progression ratio for layer thickness, 1.0 is uniform, greater than 1.0 gives finer resolution near surface | Source: Numerical grid design; Grid refinement protocol for soil transport |
| `pedosphere.transport.time_step_maximum` | `s` | `1800.0; bounds=(600.0, 3600.0)` | `fixed` | Maximum time step for adaptive timestepping in transport solver (30 minutes) | Source: Numerical solver configuration; Adaptive timestepping protocol for transport processes |
| `pedosphere.transport.default_molecular_diffusivity_water` | `m2 s-1` | `1e-09; bounds=(1e-10, 1e-08)` | `fixed` | Default molecular diffusivity used for solute transport initialization | Source: Solute transport literature; Bear (1972) Dynamics of Fluids in Porous Media; Jury & Horton (2004) Soil Physics |
| `pedosphere.transport.molecular_diffusivity_temperature_exponent` | `dimensionless` | `0.5; bounds=(0.0, 50.0)` | `fixed` | Temperature exponent n in D(T) = D0 * (T/T_ref)^n for molecular diffusion in water (Stokes-Einstein scaling). | Source: pedosphere solute diffusion scaling; Einstein (1905) Annalen der Physik 17:549-560; Bear (1972) Dynamics of Fluids in Porous Media |
| `pedosphere.transport.nitrate_diffusivity_water` | `m2 s-1` | `1.9e-09; bounds=(1.5e-09, 2.2e-09)` | `fixed` | Molecular diffusivity of nitrate ion in water at 25 Celsius | Source: Chemical transport data; Li & Gregory (1974) doi:10.1016/0016-7037(74)90022-2; Appelo & Postma (2005) Geochemistry, Groundwater and Pollution |
| `pedosphere.transport.doc_diffusivity_water` | `m2 s-1` | `6e-10; bounds=(3e-10, 1e-09)` | `fixed` | Molecular diffusivity of dissolved organic carbon in water at 25 Celsius, lower than small ions due to larger molecular size | Source: Organic solute diffusion data; Appelo & Postma (2005) Geochemistry, Groundwater and Pollution; Qualls & Haines (1991) doi:10.2136/sssaj1991.03615995005500060005x |
| `pedosphere.transport.nitrate_longitudinal_dispersivity` | `m` | `0.01; bounds=(0.001, 0.1)` | `learnable` | Longitudinal dispersivity for nitrate in soil, scale-dependent parameter | Source: Field-scale dispersion studies; Gelhar et al. (1992) doi:10.1029/92WR00607; Xu & Eckstein (1995) doi:10.1029/95WR00912 |
| `pedosphere.transport.doc_longitudinal_dispersivity` | `m` | `0.01; bounds=(0.001, 0.1)` | `learnable` | Longitudinal dispersivity for dissolved organic carbon in soil | Source: Field-scale dispersion studies; Gelhar et al. (1992) doi:10.1029/92WR00607; Kalbitz et al. (2000) doi:10.1016/S0016-7061(00)00027-2 |
| `pedosphere.transport.nitrate_distribution_coefficient` | `L kg-1` | `0.1; bounds=(0.0, 1.0)` | `learnable` | Distribution coefficient (Kd) for nitrate sorption on soil particles, very mobile anion with minimal sorption | Source: Solute transport literature; Jury & Horton (2004) Soil Physics 6th edition; van Genuchten & Wierenga (1986) doi:10.2136/sssaj1986.03615995005000050013x |
| `pedosphere.transport.doc_distribution_coefficient` | `L kg-1` | `2.0; bounds=(0.5, 10.0)` | `learnable` | Distribution coefficient (Kd) for dissolved organic carbon sorption, moderate sorption to soil minerals | Source: Organic carbon transport studies; Kalbitz et al. (2000) doi:10.1016/S0016-7061(00)00027-2; Kaiser & Guggenberger (2000) doi:10.1046/j.1365-2389.2000.00289.x |
| `pedosphere.physical.millington_quirk_exponent_numerator` | `dimensionless` | `3.333333; bounds=(3.0, 3.5)` | `fixed` | Numerator exponent (10/3) controlling tortuosity in the Millington-Quirk gas diffusivity model. | Source: RELIC parameter database; Millington & Quirk (1961) Trans. Faraday Soc. 57:1200-1207; Moldrup et al. (2000) Soil Sci. Soc. Am. J. 64:94-100 |
| `pedosphere.physical.millington_quirk_exponent_denominator` | `dimensionless` | `2.0; bounds=(1.5, 2.5)` | `fixed` | Denominator exponent applied to total porosity in the Millington-Quirk tortuosity expression. | Source: RELIC parameter database; Millington & Quirk (1961) Trans. Faraday Soc. 57:1200-1207; Penman (1940) J. Agric. Sci. 30:437-462 |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | all retained outputs are explicit REPLACE UpdateItems; concentration states and integrated residuals are per-step, while boundary fluxes are instantaneous |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\pedosphere_transport.md`.

### References

- Bear, J. (1972). Dynamics of Fluids in Porous Media (FunctorMetadata, certificate)
- Simunek, J. et al. (2008). Vadose Zone Journal 7:587-600 (FunctorMetadata)
- HYDRUS-1D Version 4.16 Technical Manual (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Millington, R.J. and Quirk, J.P. (1961), doi:10.1039/TF9615701200 (certificate)
- Simunek, J. et al. (2008), doi:10.2136/vzj2007.0077 (certificate)
- HYDRUS-1D version 4.16 technical manual (certificate)
- U.S. EPA IWEM 3.1 Technical Background Document, linear equilibrium sorption storage (certificate)

(class-relic-scientific-pedosphere-transport-two-region-mass-transfer-tworegionmasstransferfunctor)=
## TwoRegionMassTransferFunctor

Mobile-immobile first-order exchange (TransportMechanism contract, memory semantics).

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.pedosphere.transport.two_region_mass_transfer.TwoRegionMassTransferFunctor` |
| Source | `relic\scientific\pedosphere\transport\two_region_mass_transfer.py:39` |
| Runtime name | `two_region_mass_transfer` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
TwoRegionMassTransferFunctor(name: str = 'two_region_mass_transfer', *, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `name` | `str` | `'two_region_mass_transfer'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Mapping[str, Any]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `mobile_solute_concentration` | state | `kg m-3` | effort state c_m of the mobile region [kg m-3]; input state and accumulated increment |
| `immobile_solute_concentration` | state | `kg m-3` | hidden memory realization c_im of the immobile region [kg m-3]; input state and accumulated increment |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `mobile_solute_concentration` | state | `kg m-3` | decided at run time by the returned write mode | effort state c_m of the mobile region [kg m-3]; input state and accumulated increment |
| `immobile_solute_concentration` | state | `kg m-3` | decided at run time by the returned write mode | hidden memory realization c_im of the immobile region [kg m-3]; input state and accumulated increment |
| `diagnostics.two_region_exchange.mass_transfer_rate` | diagnostic | `kg m-3 s-1` | diagnostics | alpha (c_m - c_im) at the step start [kg m-3 s-1] |
| `diagnostics.two_region_exchange.storage_dissipation` | diagnostic | `kg2 m-6 s-1` | diagnostics | -alpha (c_m - c_im)^2 at the step start [kg2 m-6 s-1] |
| `diagnostics.two_region_exchange.memory_residual` | diagnostic | `kg m-3` | diagnostics | theta_m Delta c_m + theta_im Delta c_im of the emitted step [kg m-3] |

### Governing equations and canonical operator

**Literature or process form**

Two-region mobile-immobile first-order mass transfer (van Genuchten and Wierenga 1976); multirate mass transfer memory function with exponential kernel (Haggerty and Gorelick 1995)

**Canonical RELIC structural form**

$$
\dot{x} = - R e, R = \alpha [ [ 1 , - 1 ] , [ - 1 , 1 ] ], e = \frac{\mathrm{d}H}{\mathrm{d}x} = ( c_{m} , c_{\mathrm{im}} ), H = 0.5 ( \theta_{m} c_{m} ^ 2 + \theta_{\mathrm{im}} c_{\mathrm{im}} ^ 2 )
$$

| Entry | Meaning |
| --- | --- |
| `R_h` | alpha [[1, -1], [-1, 1]] symmetric positive semidefinite dissipation on the effort pair ($c_{m}$, $c_{\mathrm{im}}$) |
| `J_h` | zero; the exchange is purely dissipative |
| `memory_kernel` | $K ( t ) = \alpha \mathrm{omega} \exp ( - \mathrm{omega} t )$ with $\mathrm{omega} = \alpha / \theta_{\mathrm{im}} \mathrm{obtained}$ by eliminating $c_{\mathrm{im}}$; realization d eta/$\mathrm{d}t = - \mathrm{omega} \eta + \mathrm{omega} c_{m}$ with $\eta = c_{\mathrm{im}}$ |
| `time_integration` | exact matrix-exponential update of the linear exchange with coefficients constant during one call; governed T11 verifies the dt-halving ladder against an independent 2-by-2 matrix exponential and the semigroup under unequal partitions |
| `sub_flow_accuracy` | FunctorMetadata.sub_flow_accuracy EXACT for the concentration state update up to floating-point rounding; the step-start diagnostic rates are not time-integrated by this declaration; one fast member reports exact, while an interacting composition remains limited by its other sub-flows and composition order |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | theta-weighted storage $x = ( \theta_{m} c_{m} , \theta_{\mathrm{im}} c_{\mathrm{im}} )$ with $H = 0.5 ( \theta_{m} c_{m} ^ 2 + \theta_{\mathrm{im}} c_{\mathrm{im}} ^ 2 )$ |
| `effort_variable` | $e = \frac{\mathrm{d}H}{\mathrm{d}x} = ( c_{m} , c_{\mathrm{im}} )$ |
| `state_advancer` | $\dot{x} = - R e \mathrm{realized} \mathrm{exactly} \mathrm{per} \mathrm{step}$ by $d ( t + \mathrm{d}t ) = d ( t ) \exp ( - \kappa \mathrm{d}t )$ with the theta-weighted total held fixed |
| `runtime_usage` | TwoRegionMassTransferFunctor.forward validates both states and the step, evaluates the exact exchange, emits two ACCUMULATE increments and three REPLACE diagnostics, and reports the theta-weighted ledger residual through check_conservation_laws. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `pedosphere.transport.two_region_mass_transfer_coefficient` | `s-1` | `2e-05; bounds=(1e-08, 0.01)` | `learnable` | First-order mass transfer coefficient alpha between mobile and immobile soil water regions (two-region model) | Source: Solute transport literature; van Genuchten, M.Th. & Wierenga, P.J. (1976). Soil Sci. Soc. Am. J., 40, 473-480.; Haggerty, R. & Gorelick, S.M. (1995). Water Resour. Res., 31, 2383-2400. |
| `pedosphere.transport.two_region_mobile_water_content` | `m3 m-3` | `0.25; bounds=(0.01, 0.6)` | `fixed` | Mobile (flowing) volumetric water content theta_m of the two-region model | Source: Solute transport literature; van Genuchten, M.Th. & Wierenga, P.J. (1976). Soil Sci. Soc. Am. J., 40, 473-480. |
| `pedosphere.transport.two_region_immobile_water_content` | `m3 m-3` | `0.1; bounds=(0.001, 0.5)` | `fixed` | Immobile (stagnant) volumetric water content theta_im of the two-region model | Source: Solute transport literature; van Genuchten, M.Th. & Wierenga, P.J. (1976). Soil Sci. Soc. Am. J., 40, 473-480. |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem ACCUMULATE with per_timestep true for both concentrations and REPLACE observations for diagnostics |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Pedosphere theory](../theory/pedosphere.md).

Certificate theory owner: `docs\source\theory\pedosphere_two_region_mass_transfer.md`.

### References

- van Genuchten, M.Th. & Wierenga, P.J. (1976). Soil Sci. Soc. Am. J. 40, 473-480 (FunctorMetadata)
- Haggerty, R. & Gorelick, S.M. (1995). Water Resour. Res. 31, 2383-2400 (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- van Genuchten, M.Th. & Wierenga, P.J. (1976). Mass transfer studies in sorbing porous media I. Soil Sci. Soc. Am. J. 40, 473-480 (certificate)
- Haggerty, R. & Gorelick, S.M. (1995). Multiple-rate mass transfer for modeling diffusion and surface reactions in media with pore-scale heterogeneity. Water Resour. Res. 31, 2383-2400 (certificate)
- RELIC Mathematical Foundations Revised v2, memory/Volterra semantics (exponential-kernel passive realization theorem) (certificate)
