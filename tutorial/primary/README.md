# RELIC Primary Tutorials

## Overview

`tutorial\primary` is the main notebook suite for RELIC. It starts with
environment verification and a minimal run, then expands into real
components, coupling, theory, evaluation, optimization, data pipelines,
surrogate modeling, data assimilation, and web interfaces.

**Total: 30 notebooks (`00` through `29`).**

The learning workflow follows the operational pattern used by mature Earth
system projects: verify the environment, run a minimal case, inspect inputs
and expected outputs, diagnose failures, and only then move to scientific and
benchmark cases. The organization is informed by the official
[WRF Users Guide](https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/index.html)
and [CMAQ tutorials](https://github.com/USEPA/CMAQ/blob/main/DOCS/Users_Guide/Tutorials/README.md);
RELIC retains its own architecture and scientific contracts.

## Verified Execution Snapshot

The 2026-09-15 full-suite run executed all 30 notebooks in place with the
RELIC interpreter kernel (`RELIC_PROJECT_ROOT` set for the notebooks that
locate the repository through it), followed by the 25 numbered scripts in
`tutorial\theory` (helper scripts prefixed with `_` are maintenance tools and
are not executed as tutorials):

| Evidence | Result |
|---|---:|
| Notebook status | 30/30 `ok` |
| Executed code cells | 259 |
| Error outputs | 0 |
| Empty code outputs | 0 |
| Aggregate notebook runtime | 741.329 s |
| Theory scripts | 25/25 exit 0, sources unchanged by execution |
| Execution report | `output_audit_tmp\relic-repair-20260912\h03-tutorial-execution.json` |

Notebook 12 now admits a committed provider-view reader token before probing
the adaptive error norm (the step-level reader contract); its reported norms
are unchanged. The 2026-08-31 batch (618.107 s, report SHA-256
`AE44FA36AB1DC1B2C7369378697EA1489D696C850A1D928B775C3CECF79DFCD3`) remains
the previous reference.

Representative result checks include a surrogate holdout R2 of 0.9862 with
zero reload drift, an Euler-Maruyama deterministic-oracle error of
`1.747e-07` with exact seeded replay, EnKF/3D-Var RMSE improvements from 1.794
to 0.900/0.870, and cross-domain water/carbon ledger residuals below their
declared tolerances. Notebook 29 also reports the current `DEGRADED` health
subsystems and one not-implemented coupling capability; an executable tutorial
does not convert those system boundaries into implemented features.

## Suggested Learning Order

### Foundation

`00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12`

### Theory

`13 -> 14 -> 15`

### Applied and Advanced

`16 -> 17 -> 18 -> 19 -> 20 -> 21 -> 22 -> 23 -> 24 -> 25 -> 26 -> 27 -> 28 -> 29`

## Minimal Distribution Path

If a reader only needs one concise path, use:

1. `00_installation_and_environment.ipynb`
2. `01_minimal_relic_run.ipynb`
3. `02_real_components_and_functors.ipynb`
4. `03_real_coupling_and_validation.ipynb`
5. `04_run_layered_and_cli_walkthrough.ipynb`
6. `05_real_hydrosphere_and_pedosphere.ipynb`
7. `17_real_case_simulation.ipynb`
8. `19_post_processing_evaluation.ipynb`

## How to Run All Notebooks

Use the project root as the working directory.

```powershell
cd C:\path\to\RELIC
conda activate RELIC
pytest test/tutorial/ -q
jupyter nbconvert --to notebook --execute tutorial\primary\*.ipynb
```

The batch runner clears and rewrites output cells and execution counts in every
selected notebook. Inspect `git status --short tutorial\primary` first. Use a
dedicated working copy, or select one notebook while learning:

```powershell
jupyter nbconvert --to notebook --execute tutorial\primary\01_minimal_relic_run.ipynb
```

That command executes every notebook under `tutorial\primary`, saves the
fresh outputs back into each `.ipynb`, and writes an execution summary to
`reports\notebook_primary_execution_final.json`.

## What to Expect After Execution

- code cell outputs are stored inside each notebook
- figures are embedded in notebook outputs
- failures, if any, are reported in the JSON execution summary
- the suite is ready to distribute without requiring recipients to rerun it first

## Acceptance Checklist

A notebook refresh is complete only when:

1. every selected result has `status: "ok"`;
2. every selected result has `error_outputs: 0`;
3. the notebook's own numerical, unit, shape, conservation, or API assertions
   pass;
4. generated figures and files match the notebook's stated learning objective;
5. the execution summary and notebook outputs come from the same run; and
6. the Git diff contains only the intended output refresh.

An executed cell with no exception is not sufficient evidence for a scientific
claim. Conservation examples require their stated analytic or ledger oracle;
calibration examples require objective and parameter diagnostics; restart
examples require a continuous-versus-restart comparison.

## From Notebooks to a Research Case

After the minimal distribution path, use `tutorial\case\README.md` for the
Tapajos BVOC case. That case adds forcing provenance, parameter YAML, a full
simulation window, monitoring, checkpoints, evaluation, and publication
figures. Its archived outputs are examples tied to their recorded run identity;
they do not replace a fresh run on a changed source tree.

## Important References

- `docs\source\architecture_overview.md`
- `docs\source\coertheory.md`
- `docs\source\theory\spatiotemporal_scale_topology.md`
- `tutorial\docs\primary\00_primary_index.md`
- `tutorial\case\README.md`
