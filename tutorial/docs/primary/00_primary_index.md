# RELIC Primary Tutorial Package

## What This Folder Is

`tutorial\docs\primary` is the lightweight guide for the
`tutorial\primary` notebook suite. It is meant for release and distribution,
not for internal audit work. The goal is simple:

- explain what the primary tutorials contain
- show where to start
- point users back to the architecture SSOT
- give a stable run order for the notebooks

This folder does **not** replace `docs\source\architecture_overview.md`,
`docs\source\coertheory.md`, or the full case-study materials.

## Basic Information

| Item | Value |
|------|-------|
| Notebook directory | `tutorial\primary` |
| Guide directory | `tutorial\docs\primary` |
| Notebook count | 30 notebooks (`00` to `29`) |
| Audience | New users, reviewers, and developers who need a guided entry point |
| Runtime model | Real RELIC modules, real simulator chain, notebook outputs saved in-place |
| Main architecture reference | `docs\source\architecture_overview.md` |
| Theory references | `docs\source\coertheory.md`, `docs\source\theory\spatiotemporal_scale_topology.md` |

## Current Distribution Evidence

The 2026-08-31 distribution snapshot was executed as one 30-notebook batch:
30/30 notebooks passed, all 259 code cells have saved outputs, and both error
and empty-output counts are zero. The authoritative run summary is
`reports\notebook_primary_execution_final.json` (SHA-256
`AE44FA36AB1DC1B2C7369378697EA1489D696C850A1D928B775C3CECF79DFCD3`).
Treat this as a source-bound release snapshot: rerun the suite after changing
code, parameters, or notebook sources.

As in the official WRF and CMAQ user guides, the recommended route separates
environment checks, minimal execution, scientific cases, expected outputs,
diagnostics, and troubleshooting. This is an organizational reference only;
RELIC's Functor, parameter, unit, and conservation contracts remain the
authoritative implementation rules.

## Notebook Structure

### Foundation: `00`-`12`

Use this block first. It covers environment setup, minimal runs, real
components, coupling, CLI/configuration, hydrosphere-pedosphere workflows,
scale topology, utilities, orchestration, optimization APIs, composition,
DEC auditing, and adaptive time stepping.

### Theory: `13`-`15`

Use this block after the foundation notebooks. It explains the conceptual
background behind RELIC's structuralist stance, DEC-based topology auditing,
and category-theoretic scale language.

### Applied and Advanced: `16`-`29`

Use this block after the foundation notebooks. It covers data preparation,
real-case simulation, gallery-style module inspection, evaluation,
calibration, developer extension patterns, visualization, multi-scale
coupling, data I/O, surrogate modeling, stochastic simulation, data
assimilation, web interfaces, and cross-domain validation.

## Recommended Reading and Run Order

1. Read this file.
2. Read `01_task_context_scope_and_deliverables.md`.
3. Read `02_relic_feature_map_and_reading_roadmap.md`.
4. Read `tutorial\primary\README.md`.
5. Run notebooks `00` through `12` in order.
6. Then choose either the theory track (`13`-`15`) or the applied track (`16`-`29`).

## Quick Start

```powershell
cd C:\path\to\RELIC
conda activate RELIC
pytest test/tutorial/ -q
jupyter nbconvert --to notebook --execute tutorial\primary\*.ipynb
```

That command executes every notebook under `tutorial\primary`, stores the
fresh outputs back into each `.ipynb`, and writes a JSON execution summary
to `reports\notebook_primary_execution_final.json`.

The execution command rewrites notebook outputs. Check the worktree first, or
use `--pattern "01_minimal_relic_run.ipynb"` to refresh one notebook. Accept a
refresh only when the selected summaries are `ok`, `error_outputs` is zero,
the notebook's numerical checks pass, and the diff contains only intended
output changes.

## Release Notes for This Tutorial Layer

- The tutorial suite is organized around **real RELIC APIs**, not toy
  stand-ins.
- The notebooks are intended to be distributed with their outputs already
  saved.
- The final executable artifacts are the notebooks in `tutorial\primary`;
  this folder only explains how to approach them.

## Related Files in This Folder

| File | Role |
|------|------|
| `01_task_context_scope_and_deliverables.md` | Scope, audience, and expected deliverables |
| `02_relic_feature_map_and_reading_roadmap.md` | High-level map from RELIC modules to notebooks |
| `03_build_roadmap_and_basic_tutorial.md` | Short, task-oriented learning workflow |
| `04_module_entry_matrix_and_real_component_index.md` | Main package entry points and where they appear |
| `05_coertheory_principles_and_primary_examples.md` | Core theory-to-notebook bridge |
| `06_spatiotemporal_scale_topology_and_primary_examples.md` | Scale theory-to-notebook bridge |
