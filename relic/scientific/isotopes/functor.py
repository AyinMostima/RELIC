"""Fail-closed ScaleCoupling boundary for stable-isotope runtime.

The former daily heartbeat, private service coordination, performance history,
and NetCDF writes were removed. A future isotope Functor must return governed
StateManager component UpdateItems at the parent-process time scale. Until that
contract exists, construction and execution both fail closed.
"""

from __future__ import annotations

from typing import Any

from ...core.abstractions import FluxDict, StateDict
from ...core.functor import ScaleCouplingFunctor


class IsotopeHeartbeatError(Exception):
    """Raised when the runtime-unbound isotope Functor is requested."""


class IsotopeHeartbeatFunctor(ScaleCouplingFunctor):
    """ScaleCoupling type whose current executable contract is [N]."""

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "runtime_unbound"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject before config, parameter, module, service, or output effects."""
        _ = args, kwargs
        raise IsotopeHeartbeatError(
            "[N] isotope runtime_unbound: governed component updates are absent"
        )

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """Reject even if an unsafe constructor bypass produced an instance."""
        # References: RELIC isotope component-ledger architecture contract.
        # Mathematical expression: runtime_unbound implies no isotope x_next is evaluated.
        # Parameter meanings: x is the governed component ledger and J_transfer its complete routes.
        # Units: prospective stocks/fluxes retain isotope-specific units; time and dt [s]; no output is emitted.
        # Domain/constraints: component UpdateItems, transfer routes, parent time owner, and restart semantics are absent.
        # Sign convention: no isotope transfer direction, heartbeat update, or zero diagnostic is assigned.
        _ = state, time, dt, kwargs
        raise IsotopeHeartbeatError(
            "[N] isotope runtime_unbound: governed component updates are absent"
        )


__all__ = ["IsotopeHeartbeatFunctor", "IsotopeHeartbeatError"]
