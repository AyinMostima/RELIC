RELIC Workflows
===============

Recomposable Ecosystem Library for Integrated Coupling (RELIC) organizes its
runtime workflows around explicit state-space evolution, typed process
composition, and parameter-governed calibration. Following standard dynamical
systems and ecological modeling practice (state-variable formulation and
mass-balance step operators, coupled multi-compartment and cross-scale graphs,
and parameter estimation, sensitivity analysis, uncertainty propagation, and
stochastic forcing), read these pages in sequence for a first simulation or
enter at the track matching your task.

Learning tracks
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Track
     - Recommended pages
     - Repository companions
   * - First successful run
     - Installation, configuration, then state/Functor simulation
     - Notebooks ``00`` through ``04`` under ``tutorial\primary``
   * - Scientific composition
     - State/Functor simulation, graph validation, then coupling
     - Notebooks ``05``, ``10`` through ``15``, ``18``, ``23``, and ``29``
   * - Data-to-result workflow
     - Configuration, persistence/Data I/O, then calibration and UQ
     - Notebooks ``16``, ``17``, ``19``, ``20``, ``22``, and ``24``
   * - Runtime interfaces
     - Devices/distributed execution, CLI/extensions, and persistence
     - Notebooks ``07`` through ``09`` and ``21`` through ``28``
   * - Worked research case
     - All core workflow pages, then operations
     - ``tutorial\case\README.md`` and ``tutorial\docs\case``

How to use a tutorial
---------------------

For each task, follow this order:

1. Read the prerequisites and canonical-owner table before copying code.
2. Run the smallest example and compare its output with the documented value.
3. Inspect state, units, diagnostics, provenance, and any generated files; do
   not use process exit alone as the acceptance criterion.
4. Exercise the relevant negative condition so that a missing input, invalid
   unit, conflicting writer, or failed assertion is known to stop correctly.
5. Move to the next page only after the current example and its checks pass.

Notebook execution is intentionally separate from the public-site build. The
batch runner below clears and rewrites output cells in the selected notebooks,
so inspect the worktree first and use it only on notebooks you intend to
refresh:

.. code-block:: powershell

   cd F:\path\to\RELIC_PROJECT
   conda activate RELIC
   pytest test/tutorial/ -q
   jupyter nbconvert --to notebook --execute tutorial\primary\01_minimal_relic_run.ipynb

After the selected notebook is green, run the distribution suite from the
same unchanged source tree:

.. code-block:: powershell

   jupyter nbconvert --to notebook --execute tutorial\primary\*.ipynb

An executed notebook is acceptable when its summary status is ``ok``, it has
no error outputs, no unexplained empty code outputs, its documented numerical
checks pass, and the resulting diff contains only the intended notebook output
refresh. Confirm that the report count equals 30 before distributing the
suite. Use
``tutorial\primary\README.md`` for the full 30-notebook order.

.. toctree::
   :maxdepth: 2

   installation_and_imports
   configuration_and_parameters
   state_functors_and_simulation
   graph_validation_and_execution
   persistence_outputs_and_data_io
   coupling_workflows
   calibration_sensitivity_and_uq
   devices_and_distributed
   cli_and_extensions
