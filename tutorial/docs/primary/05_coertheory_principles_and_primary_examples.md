# Coertheory Principles and Primary Examples

This note links the main ideas in `docs\source\coertheory.md` to the
release tutorial notebooks.

## Core Principles and Where They Appear

| Principle | Meaning in RELIC | Notebook Anchors |
|-----------|------------------|------------------|
| Graph-structured execution | processes and state relations are represented through explicit dependencies | `01`, `04`, `08`, `10`, `11`, `17` |
| Metadata and contracts | Functors declare inputs, outputs, and semantics instead of hiding them | `02`, `06`, `18`, `21` |
| Transactional state updates | process increments are assembled through the RELIC runtime chain | `01`, `03`, `10`, `17` |
| Conservation auditing | storage change and flux budgets are checked rather than assumed | `03`, `10`, `11`, `19`, `29` |
| Stability control | diagnostics and time-step control are part of the execution logic | `07`, `12`, `17` |
| Differentiability | the primary chain stays compatible with PyTorch-based calibration and assimilation | `09`, `20`, `25`, `27` |
| Zero hardcoding for parameters | parameters are drawn from the RELIC parameter system instead of notebook-local constants | `03`, `04`, `05`, `20` |

## How to Use This Bridge

If a reader encounters a concept in `coertheory.md` and wants the closest
notebook example:

- start with `02_real_components_and_functors.ipynb` for metadata and contracts
- use `03_real_coupling_and_validation.ipynb` for budgets and validation
- use `11_dec_conservation_audit.ipynb` for structure and conservation
- use `12_operator_splitting_adaptive_timestep.ipynb` for splitting order and the adaptive step-size heuristic (no stability or CFL guarantee)
- use `20_calibration_sensitivity_uncertainty.ipynb` and
  `27_data_assimilation.ipynb` for differentiable workflows

## What to Look For in Outputs

- explicit dependency structure
- declared variables and metadata
- residual or drift diagnostics
- time-step adaptation behavior
- parameter-system driven configuration
