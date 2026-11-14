"""Fail-closed boundary for atmosphere emission-source coupling.

The conditional vertical-allocation identity is scientifically meaningful only
when every required surface source has a provenance-bound provider in the same
compiled Graph. RELIC currently has no executable writer for the six canonical
anthropogenic surface-rate inputs. A selected BVOC writer cannot substitute for
that missing joint dependency.

References:
    - CMAQ User Guide, emissions source terms and vertical allocation.
    - Seinfeld and Pandis (2016), Atmospheric Chemistry and Physics.
    - Emmons et al. (2010), MOZART-4 model description and evaluation.

Required runtime contract:
    A governed provider must resolve all six anthropogenic surface fluxes and
    any exactly selected BVOC flux before Functor registration. The provider,
    this node, its outputs, and StateManager commit must share one compiled
    dependency graph. That contract is currently unavailable.

Domain and sign constraints:
    Missing providers are structural runtime blockers. Preloaded tensors, zero
    fields, cached values, optional dependency downgrades, and name-only dummy
    providers cannot establish scientific provenance or writer identity.
"""

from __future__ import annotations

from typing import Any

from relic.core.abstractions import FluxDict, StateDict
from relic.core.functor import TransportMechanismFunctor


_RUNTIME_BLOCKER = (
    "[N] runtime_dependency_unbound: AtmosphereCoupling requires the canonical "
    "emission_inventory dependency and six anthropogenic surface-flux providers, "
    "but no executable Graph writer is currently bound"
)


class AtmosphereCoupling(TransportMechanismFunctor):
    """Canonical coupling node whose governed scientific runtime is unbound."""

    TRANSPORT_TYPE = "gas_exchange"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject before Functor, parameter, metadata, cache, or state effects."""

        _ = args, kwargs
        raise RuntimeError(_RUNTIME_BLOCKER)

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """Reject even when unsafe construction bypasses ``__init__``."""

        # References: CMAQ emissions-source contracts; Seinfeld and Pandis (2016).
        # Mathematical expression: runtime_unbound implies no x_next or source field is evaluated.
        # Parameter meanings: F_surface are provenance-bound species fluxes and G_atm their port map.
        # Units: prospective surface fluxes [kg m-2 s-1]; time and dt [s]; no output is emitted.
        # Domain/constraints: all six providers and any selected BVOC writer must share one compiled Graph.
        # Sign convention: no atmosphere-inward source sign or zero-field fallback is assigned while unbound.
        _ = self, state, time, dt, kwargs
        raise RuntimeError(_RUNTIME_BLOCKER)


__all__ = ["AtmosphereCoupling"]
