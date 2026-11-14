# RELIC Feature Map and Reading Roadmap

## What RELIC Emphasizes

RELIC is a PyTorch-based biogeochemical modeling framework built around:

- composable Functors
- simulator-managed execution
- centralized parameter and contract management
- real coupling, validation, and data I/O chains

The primary notebook suite is organized to expose those ideas in a usable
order rather than in raw source-code order.

## Feature Map

| Capability Area | Main Package | First Notebook |
|-----------------|--------------|----------------|
| Core lifecycle | `relic\core` | `01_minimal_relic_run.ipynb` |
| Real scientific Functors | `relic\scientific` | `02_real_components_and_functors.ipynb` |
| Coupling and validation | `relic\coupling`, `relic\validation` | `03_real_coupling_and_validation.ipynb` |
| Configuration and CLI | `relic\interfaces`, `relic\parameters` | `04_run_layered_and_cli_walkthrough.ipynb` |
| Hydro and soil processes | `relic\scientific\hydrosphere`, `relic\scientific\pedosphere` | `05_real_hydrosphere_and_pedosphere.ipynb` |
| Scale topology | `relic\core`, `relic\coupling\scales` | `06_spatiotemporal_scale_topology_walkthrough.ipynb` |
| Monitoring and utilities | `relic\monitoring`, `relic\utils`, `relic\extensions` | `07_extensions_monitoring_utils_public_api_walkthrough.ipynb` |
| Orchestration and data | `relic\orchestration`, `relic\data` | `08_orchestration_data_public_api_walkthrough.ipynb` |
| Optimization and stochastic tools | `relic\optimization`, `relic\stochastic`, `relic\computation` | `09_optimization_stochastic_computation_public_api_walkthrough.ipynb` |
| Applied workflows | multiple packages | `16_data_preprocessing.ipynb` onward |

## Reading Roadmap

### Start Here

`00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12`

This is the main onboarding path. It is the best route for new users.

### Then Choose One of Two Paths

- Theory path: `13 -> 14 -> 15`
- Applied path: `16 -> 17 -> 18 -> 19 -> 20 -> 21 -> 22 -> 23 -> 24 -> 25 -> 26 -> 27 -> 28 -> 29`

## Good Entry Points by Goal

| Goal | Start With |
|------|------------|
| I need the smallest possible runnable example | `01_minimal_relic_run.ipynb` |
| I want to see real RELIC components immediately | `02_real_components_and_functors.ipynb` |
| I care about coupling and validation | `03_real_coupling_and_validation.ipynb` |
| I want a realistic end-to-end workflow | `17_real_case_simulation.ipynb` |
| I need calibration, uncertainty, or assimilation | `20_calibration_sensitivity_uncertainty.ipynb` and `27_data_assimilation.ipynb` |
| I need extension guidance | `21_developer_guide.ipynb` |

## Do Not Start With

- `tutorial\case` if you have not run the foundation notebooks
- deep theory notebooks before the foundation block
- advanced notebooks as isolated API samples without the architecture context
