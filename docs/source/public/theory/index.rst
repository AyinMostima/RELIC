Scientific Theory and Governing Equations
==========================================

The Recomposable Ecosystem Library for Integrated Coupling (RELIC) formulates
terrestrial and interfacial Earth-system simulation as a composable graph of
typed PyTorch ``FunctorBase`` operators acting on governed state reservoirs.
Universal mass, energy, and entropy balance laws are separated from discrete
cochain and port-Hamiltonian operators, numerical time integrators, and
sphere-specific constitutive closures (*Methods of Ecological Analysis*,
Chapters 1--30).

Equations across these pages define the live ``forward()`` and ``StateManager``
contracts of the installed RELIC package. Literature constitutive forms are
distinguished from canonical runtime operators: structural and thermodynamic
equivalence claims (`Level 0` through `Level 5`) are owned exclusively by each
process's declarative ``IPHSEquivalenceCertificate``.

.. toctree::
   :maxdepth: 2
   :caption: Mathematical foundations

   foundations
   dec_iphs_generic
   numerics

.. toctree::
   :maxdepth: 2
   :caption: Earth-system process theory

   atmosphere
   biosphere
   hydrosphere
   pedosphere
   anthroposphere
   isotopes_transport
