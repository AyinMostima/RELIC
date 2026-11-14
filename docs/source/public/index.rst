RELIC Documentation
===================

This site documents the public architecture and scientific contract for the
Recomposable Ecosystem Library for Integrated Coupling (RELIC) |release|. It
covers mathematical foundations, all 90 certified concrete scientific
``Functor`` classes (backed by 92 IPHS certificates), runtime workflows, graph
orchestration, extension rules, and the core Python API across the 16
subpackages of ``relic\`` (594 subpackage ``.py`` files, 595 Python files
including ``relic\__init__.py``, 316,129 lines of Python, seven architecture
layers plus an orthogonal control plane, and 4,643 default runtime parameters
loaded from 138 YAML files).

The version shown in the page title and footer is read from the package version
in ``pyproject.toml``. Every section is governed by that versioned architecture
contract and its live public owners.

Start with a learning route
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Goal
     - Start here
     - Evidence that the step is complete
   * - Verify an installation
     - :doc:`operations` and :doc:`workflows/installation_and_imports`
     - Package version prints, documented module entry points return help, and
       imports resolve from the intended environment.
   * - Run the first state transition
     - :doc:`workflows/state_functors_and_simulation`
     - The complete minimal example prints the documented state, update key,
       and step count exactly.
   * - Assemble scientific processes
     - :doc:`theory/index`, :doc:`science/index`, and
       :doc:`workflows/configuration_and_parameters`
     - Each selected process has a valid owner, inputs, units, parameters,
       applicability limits, and an evidence-appropriate certificate status.
   * - Validate and execute GraphIR
     - :doc:`workflows/graph_validation_and_execution`
     - Validation has no blocking errors, required assertions pass, and the
       compiled plan resolves to the canonical simulator lifecycle.
   * - Prepare a reproducible run
     - :doc:`workflows/persistence_outputs_and_data_io` and
       :doc:`workflows/devices_and_distributed`
     - Inputs, configuration, device policy, outputs, checkpoint receipts,
       hashes, and provenance are recorded for the same run.
   * - Follow a worked research example
     - Repository notebooks under ``tutorial\primary`` and the Tapajos case
       under ``tutorial\case``
     - The execution summary reports no notebook errors, and the case output
       passes its documented completeness, unit, and scientific checks.

Executable tutorial layers
--------------------------

RELIC provides three complementary tutorial layers. They are connected, but
they serve different purposes:

1. The workflow pages on this site provide small copy-and-run examples with
   signatures, expected output, exceptions, and current limitations.
2. ``tutorial\primary`` contains 30 notebooks, ordered from environment setup
   through real components, coupling, conservation, calibration, data I/O,
   surrogate models, ensembles, assimilation, and cross-domain validation.
3. ``tutorial\case`` is the full Tapajos BVOC case, with forcing provenance,
   configuration, simulation, monitoring, checkpoints, evaluation, and
   figures. It is a worked research case, not a universal benchmark dataset.

Use the same evidence pattern throughout: preflight the environment; declare
inputs and configuration; run one bounded case; inspect logs and outputs;
compare numerical results with an explicit oracle or reference; record
provenance; and state what the run does not prove. A successful process exit by
itself is not a scientific validation.

Documentation design references
-------------------------------

The guide organization follows established earth-system documentation
practice: system overview, preparation, configuration, execution, outputs,
troubleshooting, science options, and benchmark-style tutorials. The
`WRF Users' Guide <https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/index.html>`_
and the
`CMAQ User's Guide <https://github.com/USEPA/CMAQ/blob/main/DOCS/Users_Guide/README.md>`_
are organizational references only. RELIC's APIs, equations, evidence levels,
and runtime claims remain governed by this repository's current owners and
tests.

Language navigation
-------------------

Use the fixed ``Chinese`` or ``English`` control on any page to open the same
section in the other language. The publication landing page also provides
direct English and Chinese entry points.

.. toctree::
   :maxdepth: 2
   :caption: RELIC Guide

   architecture
   theory/index
   science/index
   workflows/index
   orchestration
   operations
   extension_contracts
   api/index
