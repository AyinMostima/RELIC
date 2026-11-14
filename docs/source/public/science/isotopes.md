# Stable-Isotope Scientific Classes

This reference covers all 1 certificate-backed scientific classes in `relic.scientific.isotopes`: 0 executable concrete Functors, 1 fail-closed unavailable boundary, and 0 abstract contracts. Each entry is generated from the live class, its IPHS certificate, State Contract Model, and ParameterManager metadata. Fail-closed unavailable classes are not instantiated during documentation generation.

The tables distinguish constructor controls, runtime variables, and ParameterManager scientific parameters. Values shown as package defaults belong to the documented RELIC architecture version and must not be copied as hardcoded constants into scientific code.

Certificate wildcard parameter groups (`*`, `<pft>`) are expanded to their exact current YAML entries. Every other declared parameter is matched to the parameter database by its exact full key; a declared name is never mapped to a key that merely ends with the same name.

Exact ParameterManager rows report the live YAML value, bounds, gradient policy, description, and provenance. A row whose value reads `not declared in the parameter database` is intentionally not assigned an invented default: the class metadata or certificate declares the quantity, with the meaning and units it states, but the parameter database has no key of that exact name.

IPHS levels are read from the sole certificate owner. Levels 1 to 3 describe compatibility, wrapping, or declared structure; they are not strict IPHS/GENERIC claims.

The Channel column of each runtime output contract applies `relic.core.state.types.partition_outputs`, the only runtime output channel classifier, to the declared output name. A name that the classifier reserves for diagnostics, such as `diagnostics.*`, goes to the diagnostics channel and is never committed as state. For any other name the channel is decided at run time by the write mode of the returned value: `StateWriteMode.DERIVED` routes the output to the diagnostics channel, and any other write mode sends it to the StateManager commit (a value without a write mode takes the run's semantic-mode rules or the StateManager schema). The generator does not run `forward()`, so it does not predict that write mode. The Kind column is the State Contract Model kind of the variable and does not set its channel.

(class-relic-scientific-isotopes-functor-isotopeheartbeatfunctor)=
## IsotopeHeartbeatFunctor

ScaleCoupling type whose current executable contract is [N].

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.isotopes.functor.IsotopeHeartbeatFunctor` |
| Source | `relic\scientific\isotopes\functor.py:21` |
| Runtime name | `not_applicable_runtime_unbound` |
| Scientific abstraction | `ScaleCoupling` |
| Canonical bridge | `ScaleCouplingFunctor` |
| Structural role | `scale_coupling` |
| Spatial scale | `runtime_unbound` |
| Variable kind | `runtime_unbound` |
| IPHS level | `level_0` |
| Implementation status | `not_implemented_fail_closed` |
| Class status | `fail-closed unavailable boundary` |

:::{warning}
This class is `[N] runtime_unbound`. Construction and `forward(...)` reject before state access or side effects. Empty runtime input, output, and port declarations below are the documented fail-closed boundary, not missing documentation.
:::

### Constructor and argument meanings

```python
IsotopeHeartbeatFunctor(*args: Any, **kwargs: Any) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `args` | `Any` | `required` | Additional positional arguments accepted by the inherited constructor. |
| `kwargs` | `Any` | `required` | Additional canonical bridge or FunctorBase options. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |

### Governing equations and canonical operator

**Literature or process form**

Stable-isotope fractionation laws require an explicitly owned component ledger, complete transfer routes, units, and restart semantics before they can advance model state.

**Canonical RELIC structural form**

No IPHS formulation is claimed while the governed StateManager isotope component ledger and complete routing contract are unavailable.

| Entry | Meaning |
| --- | --- |
| `runtime_status` | no executable isotope observation, reaction, transport, or scale-coupling operator is registered |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | not_applicable_runtime_unbound |
| `effort_variable` | not_applicable_runtime_unbound |
| `state_advancer` | none_runtime_unbound |
| `runtime_usage` | no UpdateItem, private diagnostic ledger, or state mutation is produced |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

This class does not expose a class-level executable `ports()` schema. Certificate channels therefore describe composition evidence, not an instantiated PortSpec API.
No executable dependency or composition path is declared while this capability remains fail-closed.

### Scientific parameter and numeric binding contract

This fail-closed class does not resolve or execute scientific parameters.

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `constructor` | rejects before config, parameter, module, service, writer, or output effects |
| `forward` | independently rejects unsafe constructor bypass before state access |
| `abstraction_contract` | ScaleCoupling |
| `bridge_class` | ScaleCouplingFunctor |

Detailed derivation: [Isotopes theory](../theory/isotopes_transport.md).

Certificate theory owner: `docs\source\theory\isotopes.md`.

### References

- RELIC Simulator Architecture Documentation. (certificate)
- Stable-isotope mass-balance and fractionation literature summarized in the isotope theory document. (certificate)
