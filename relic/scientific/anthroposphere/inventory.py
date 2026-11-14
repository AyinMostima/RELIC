"""Fail-closed boundary for anthropogenic emission inventory allocation.

The available ParameterManager records are HTAPv3.2/EDGAR 2020 global annual
sector-species totals. RELIC currently has no executable owner for the target
domain fraction or for a verified global source-to-target grid binding.

References:
    - Guizzardi et al. (2025), Earth System Science Data 17, 5915-5950.
    - Crippa et al. (2024), Earth System Science Data 16, 2811-2830.
    - CF Conventions, calendar and cell-area contracts.

Required mathematical expression:
    F[s,k,i,t] = E_global[s,k] * beta[s,k,D] * T[s,k,t] * P[s,k,i|D]
                 / (Y[C] * seconds_per_day * A[i]).

Parameter meanings and units:
    E_global is a global annual mass [kg inventory-year-1].
    beta is the sector/species fraction assigned to target domain D [1].
    T is a calendar-normalized temporal factor [1].
    P is a non-negative within-domain allocation with sum_i(P)=1 [1].
    Y is the simulator-calendar year length [day inventory-year-1].
    A is target cell area [m2], and F is a surface source rate [kg m-2 s-1].

Domain and sign constraints:
    beta must be executable, source-bound, and in [0, 1]. P must be bound to
    the same target domain and source inventory. Because neither binding exists,
    constructing or executing this node fails before parameter, calendar, state,
    cache, checkpoint, or output effects. No zero-flux fallback is permitted.
"""

from __future__ import annotations

from typing import Any

from relic.core.abstractions import FluxDict, StateDict
from relic.core.functor import ScaleCouplingFunctor


_RUNTIME_BLOCKER = (
    "[N] runtime_domain_unbound: global HTAPv3.2/EDGAR annual totals "
    "lack an executable sector/species target-domain fraction or verified "
    "global target-grid binding"
)


class EmissionInventory(ScaleCouplingFunctor):
    """Canonical inventory node whose current scientific runtime is unbound."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject before Functor, parameter, calendar, proxy, or ledger effects."""

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

        # References: Guizzardi et al. (2025); Crippa et al. (2024).
        # Mathematical expression: missing beta and P bindings imply no F[s,k,i,t] evaluation.
        # Parameter meanings: beta is target-domain share, P grid allocation, and F surface source.
        # Units: prospective F [kg m-2 s-1]; time and dt [s]; no output is emitted while unbound.
        # Domain/constraints: beta and P are absent, so every call fails before any side effect.
        # Sign convention: no emission orientation or zero-flux fallback is assigned while unbound.
        _ = self, state, time, dt, kwargs
        raise RuntimeError(_RUNTIME_BLOCKER)


__all__ = ["EmissionInventory"]
