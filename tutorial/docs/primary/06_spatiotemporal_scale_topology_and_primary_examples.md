# Spatiotemporal Scale Topology and Primary Examples

This note connects `docs\source\theory\spatiotemporal_scale_topology.md`
to the primary notebook suite.

## Main Concepts

| Concept | Meaning | Best Notebook Anchor |
|---------|---------|----------------------|
| `SpatialScale` | Functor-level spatial label used by the label-level scale check; not a geometry | `06_spatiotemporal_scale_topology_walkthrough.ipynb` |
| `TemporalScale` | Functor-level time-scale label; one input (after explicit `execution_group`, the Functor attribute and `characteristic_time_days`) to fast/medium/slow classification; execution frequency comes from `medium_interval_seconds`, `slow_interval_seconds` or calendar months | `06_spatiotemporal_scale_topology_walkthrough.ipynb` and `12_operator_splitting_adaptive_timestep.ipynb` |
| `VariableKind` | Functor-level role label: state, flux, diagnostic, parameter, forcing (unknown = undeclared); not intensive/extensive | `06_spatiotemporal_scale_topology_walkthrough.ipynb` |
| operator splitting | how the Functors of the fast group are composed (Lie or Strang); medium and slow groups follow sequentially when triggered | `12_operator_splitting_adaptive_timestep.ipynb` |
| cross-scale transfer | how information is moved between scales | `23_multi_scale_coupling.ipynb` |
| conservation under multi-rate execution | how split workflows are still checked for consistency | `11_dec_conservation_audit.ipynb`, `29_cross_domain_validation.ipynb` |

## Recommended Reading Order for Scale Topics

1. `06_spatiotemporal_scale_topology_walkthrough.ipynb`
2. `12_operator_splitting_adaptive_timestep.ipynb`
3. `23_multi_scale_coupling.ipynb`
4. `29_cross_domain_validation.ipynb`

## Code Anchors

| Topic | Main Code Area |
|-------|----------------|
| scale and variable enums | `relic\core\functor\functor_types.py` |
| time-group assignment | `relic\core\simulator\dependency_ops.py`, `relic\core\simulator\registration.py` |
| rate-group triggers and interval commit | `relic\core\simulator\time_ops.py` |
| execution scheduling | `relic\core\simulator\execution_loop.py` |
| scale transfer operators | `relic\coupling\scales\` |

## What Readers Should Verify in Notebook Outputs

- fast and slow groups are visibly different
- accepted time steps are sensible for the process being shown
- split solutions converge at the order allowed by the sub-integrators (exact sub-flows: Lie 1, Strang 2; explicit-Euler sub-steps: 1 for both)
- scale transfer results match hand-computed oracles: e.g. 2 and 4 kg h-1 over [0, 3600) s and [3600, 5400) s give 4 kg; an accumulation [10 kg, missing] over the same intervals stays 10 kg with valid coverage 2/3 (no scale-up); a 4/8 kg overlap remap with coverage 1/0.5 gives 8 kg and leaves 4 kg unmapped
