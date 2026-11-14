"""Fail-closed boundary for anthroposphere CF metadata mapping.

The SCM ``ContractRegistry`` is the sole owner of variable names, units, CF
standard names, state ownership, and writer permissions. The removed generic
mapping path duplicated that owner and accepted ParameterManager strings as CF
evidence without proving official-table membership or physical semantics.

References:
    - CF Standard Name Table v94, 9 June 2026.
    - CF Conventions, standard names, units, coordinates, and cell methods.
    - RELIC architecture overview, ContractRegistry state-variable contracts.

Required mathematical and semantic contract:
    m in V_CF(v94), U_actual equivalent to U_canonical(m), and
    Q_SCM = Q_CF(m) on the same basis, support, coordinates, and cell methods.

Parameter meanings and units:
    m is an official CF standard-name entry or alias [text identifier].
    V_CF(v94) is the official controlled vocabulary [set].
    U_actual and U_canonical are physical units for the same quantity.
    Q_SCM and Q_CF are the SCM and CF quantity semantics, respectively.

Domain and constraints:
    Exact membership and compatible units are necessary but not sufficient.
    Chemical species, mass or mole basis, spatial support, coordinates,
    providers, compiled Graph inputs, and the metadata consumer must all be
    explicit. RELIC currently has no governed path satisfying that joint
    contract for this generic node, so construction and execution fail before
    metadata, state, parameter, cache, registry, or diagnostic effects.
"""

from __future__ import annotations

from typing import Any

from relic.core.abstractions import FluxDict, StateDict
from relic.core.functor import ScaleCouplingFunctor


_RUNTIME_BLOCKER = (
    "[N] runtime_metadata_domain_unbound: anthroposphere CF metadata requires "
    "an exact SCM variable, official standard-name membership, physical-basis "
    "and coordinate binding, and a governed metadata application path"
)


class AnthroposphereInterfaceMapping(ScaleCouplingFunctor):
    """Canonical interface node whose generic metadata runtime is unbound."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject before Functor, metadata, parameter, registry, or state effects."""

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

        # References: CF Conventions and CF Standard Name Table v94.
        # Mathematical expression: an unbound SCM-to-CF map defines no metadata transform m(v).
        # Parameter meanings: m is a CF name, U its units, Q its semantics, and v the SCM variable.
        # Units: time and dt [s]; no data or metadata unit transform is emitted while unbound.
        # Domain/constraints: official-name, unit, basis, support, coordinate, and consumer bindings are absent.
        # Sign convention: no physical flux direction or metadata rewrite is assigned while unbound.
        _ = self, state, time, dt, kwargs
        raise RuntimeError(_RUNTIME_BLOCKER)


__all__ = ["AnthroposphereInterfaceMapping"]
