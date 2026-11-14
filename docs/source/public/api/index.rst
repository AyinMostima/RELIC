Core API Reference
==================

This reference documents the canonical architecture classes that define
Recomposable Ecosystem Library for Integrated Coupling (RELIC) composition,
state advancement, parameter ownership, graph orchestration, validation, and
scientific structure. It does not enumerate every concrete scientific Functor
or internal helper.

Each class entry identifies the canonical import path, constructor parameters,
key methods, lifecycle role, and failure semantics.

Coverage contract
-----------------

``core_runtime`` covers every public class exported by ``relic.core`` and the
canonical runtime and configuration classes needed to use them. The remaining
pages cover the adjacent owners used by that runtime: ``parameters_contracts``
(``4,643`` default runtime parameters across ``138`` YAML files, physical
constants, and variable contracts), ``orchestration_validation`` (``GraphIR``,
``PortSpec``, six concrete adapter classes across seven selector keys,
validation, compilation, and ``GraphExecutor``), and
``scientific_structure_extensions`` (DEC cochains, ``LinearChainMap``,
``HodgeStarSpec``, ``IPHSStructureSpec``, ``IPHSEquivalenceCertificate``, Python
extensions, and C ABI coupling plugins). All ``90`` certified concrete
scientific ``Functor`` classes (``BiologicalProcessFunctor``: ``37``,
``TransportMechanismFunctor``: ``27``, ``PhysicalFieldFunctor``: ``14``,
``ScaleCouplingFunctor``: ``12``, backed by ``92`` IPHS YAML certificates) are
documented separately in the :doc:`scientific class reference
<../science/index>`; private helpers remain outside the public reference.

.. toctree::
   :maxdepth: 2
   :caption: Core Classes

   core_runtime
   parameters_contracts
   orchestration_validation
   scientific_structure_extensions
