RELIC Documentation
=====================

RELIC is a composable, differentiable framework for biogeochemical
simulation. This public documentation is organized around stable architecture,
orchestration and governance, scientific foundations, and process theory.

.. important::

   ``architecture_overview`` is the single source of truth for the
   ``relic\`` package architecture, canonical owners, and extension contracts.
   Internal migration plans and generated snapshots under ``analysis_results\``
   do not replace current code, tests, or commit-linked execution evidence.
   ``docs\build\`` is generated output and must not be edited directly.

.. toctree::
   :maxdepth: 2
   :caption: Architecture and Governance

   architecture_overview
   graph_orchestration
   dependency_governance

.. toctree::
   :maxdepth: 2
   :caption: Scientific Foundations

   coertheory

.. toctree::
   :maxdepth: 2
   :caption: Process Theory Library
   :glob:

   theory/*
