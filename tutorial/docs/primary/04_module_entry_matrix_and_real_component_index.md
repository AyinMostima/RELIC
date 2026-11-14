# Module Entry Matrix and Real Component Index

## Main Entry Points

This is a compact lookup table for the primary RELIC packages that appear in
the notebook suite.

| Package | Main Entry Point | Typical Component | First Notebook |
|---------|------------------|-------------------|----------------|
| `relic\core` | `relic\core\__init__.py` | `FunctorBase`, `RELICSimulator` | `01_minimal_relic_run.ipynb` |
| `relic\parameters` | `relic\parameters\manager.py` | `ParameterManager` | `03_real_coupling_and_validation.ipynb` |
| `relic\contracts` | `relic\contracts\` | contract registry and specs | `02_real_components_and_functors.ipynb` |
| `relic\scientific` | `relic\scientific\` | real Functors by domain | `02_real_components_and_functors.ipynb` |
| `relic\coupling` | `relic\coupling\` | scale and domain coupling APIs | `03_real_coupling_and_validation.ipynb` |
| `relic\data` | `relic\data\` | forcing prep, quality checks, I/O | `08_orchestration_data_public_api_walkthrough.ipynb` |
| `relic\validation` | `relic\validation\` | health checks, conservation validation | `03_real_coupling_and_validation.ipynb` |
| `relic\orchestration` | `relic\orchestration\` | graph IR and compilation | `08_orchestration_data_public_api_walkthrough.ipynb` |
| `relic\optimization` | `relic\optimization\` | calibration and assimilation | `09_optimization_stochastic_computation_public_api_walkthrough.ipynb` |
| `relic\stochastic` | `relic\stochastic\` | ensemble and stochastic tools | `09_optimization_stochastic_computation_public_api_walkthrough.ipynb` |
| `relic\interfaces` | `relic\interfaces\` | CLI, web, REST interfaces | `04_run_layered_and_cli_walkthrough.ipynb` |
| `relic\utils` | `relic\utils\` | constants and utility helpers | `07_extensions_monitoring_utils_public_api_walkthrough.ipynb` |

## Domain Examples

| Domain | Example Notebook | What You See |
|--------|------------------|--------------|
| Atmosphere and biosphere | `02_real_components_and_functors.ipynb` | real scientific module inspection |
| Hydrosphere and pedosphere | `05_real_hydrosphere_and_pedosphere.ipynb` | water and soil process outputs |
| Multi-domain composition | `10_multi_functor_composition.ipynb` | coupled execution and conservation audit |
| Full application chain | `17_real_case_simulation.ipynb` | site-level workflow using RELIC runtime components |

## Shortest Paths from Notebook to Code

| Need | Notebook | Code Anchor |
|------|----------|-------------|
| learn the lifecycle | `01_minimal_relic_run.ipynb` | `relic\core\simulator\main.py` |
| inspect metadata and contracts | `02_real_components_and_functors.ipynb` | `relic\core\functor\functor_base.py` |
| inspect validation logic | `03_real_coupling_and_validation.ipynb` | `relic\validation\` |
| inspect configuration-driven runs | `04_run_layered_and_cli_walkthrough.ipynb` | `relic\interfaces\cli\` |
| inspect data pipelines | `24_data_io_pipeline.ipynb` | `relic\data\` |
| inspect developer extension patterns | `21_developer_guide.ipynb` | `relic\extensions\` |
