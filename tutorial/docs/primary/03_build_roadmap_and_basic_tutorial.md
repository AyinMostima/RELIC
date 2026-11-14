# Build Roadmap and Basic Tutorial

## Minimal Learning Workflow

If a user only needs one stable path through the primary tutorials, use this:

1. `00_installation_and_environment.ipynb`
2. `01_minimal_relic_run.ipynb`
3. `02_real_components_and_functors.ipynb`
4. `03_real_coupling_and_validation.ipynb`
5. `04_run_layered_and_cli_walkthrough.ipynb`
6. `05_real_hydrosphere_and_pedosphere.ipynb`
7. `17_real_case_simulation.ipynb`
8. `19_post_processing_evaluation.ipynb`

That path is short enough for onboarding and rich enough to show a real
RELIC workflow from setup to evaluation.

## Goal-to-Notebook Map

| If you want to... | Start here | Then continue with |
|-------------------|-----------|--------------------|
| verify the environment | `00_installation_and_environment.ipynb` | `01_minimal_relic_run.ipynb` |
| understand the lifecycle | `01_minimal_relic_run.ipynb` | `02_real_components_and_functors.ipynb` |
| inspect real modules | `02_real_components_and_functors.ipynb` | `18_scientific_module_gallery.ipynb` |
| study coupling and validation | `03_real_coupling_and_validation.ipynb` | `29_cross_domain_validation.ipynb` |
| run from configuration | `04_run_layered_and_cli_walkthrough.ipynb` | `17_real_case_simulation.ipynb` |
| focus on hydro and soil science | `05_real_hydrosphere_and_pedosphere.ipynb` | `17_real_case_simulation.ipynb` |
| focus on optimization or UQ | `20_calibration_sensitivity_uncertainty.ipynb` | `27_data_assimilation.ipynb` |
| focus on data pipelines | `16_data_preprocessing.ipynb` | `24_data_io_pipeline.ipynb` |
| focus on visualization | `19_post_processing_evaluation.ipynb` | `22_visualization_cookbook.ipynb` |
| extend the framework | `21_developer_guide.ipynb` | `23_multi_scale_coupling.ipynb` |

## Release-Friendly Grouping

### Onboarding

- `00_installation_and_environment.ipynb`
- `01_minimal_relic_run.ipynb`
- `02_real_components_and_functors.ipynb`

### Core Workflow

- `03_real_coupling_and_validation.ipynb`
- `04_run_layered_and_cli_walkthrough.ipynb`
- `05_real_hydrosphere_and_pedosphere.ipynb`
- `17_real_case_simulation.ipynb`
- `19_post_processing_evaluation.ipynb`

### Architecture and Theory

- `06_spatiotemporal_scale_topology_walkthrough.ipynb`
- `10_multi_functor_composition.ipynb`
- `11_dec_conservation_audit.ipynb`
- `12_operator_splitting_adaptive_timestep.ipynb`
- `13_structuralist_foundations.ipynb`
- `14_dec_topology_auditing.ipynb`
- `15_category_theory_scales.ipynb`

### Advanced Practice

- `16_data_preprocessing.ipynb` through `29_cross_domain_validation.ipynb`

## Verification Expectation

For distribution, the recommended state is:

- every notebook has been executed successfully
- outputs are stored inside each notebook
- figures and text outputs are visible without rerunning
- the execution summary JSON is kept in `reports`
- every selected summary reports `status: "ok"` and `error_outputs: 0`
- numerical or scientific examples pass their stated value, unit, residual,
  conservation, convergence, or restart oracle
- the notebook output diff contains no unrelated source or data changes

Use `tools\notebook_primary_audit.py` for a read-only structure/output audit.
Use `tools\execute_primary_notebooks.py --pattern <name>` when only one
notebook needs refresh; the execution tool rewrites output cells in place.
