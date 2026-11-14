"""Shared state types for StateManager."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple
import time

from torch import Tensor


class StateVariableType(str, Enum):
    """Storage role of a StateManager variable.

    Only storage roles belong here. The spatial quantity basis
    (extensive/intensive/categorical) is owned by ``SpatialQuantityBasis``
    (relic.coupling.scales.types) and the prognostic/diagnostic distinction by
    the SCM contract kind (``StateVariableSpec.kind``). CONCENTRATION is read by
    the conservation filter; the other roles carry no runtime behaviour.
    """
    SCALAR = "scalar"
    GRID = "grid"
    CONCENTRATION = "concentration"
    PARAMETER = "parameter"


class UpdateMode(str, Enum):
    """Update application mode for state variables."""
    ACCUMULATE = "accumulate"
    REPLACE = "replace"
    FLUX = "flux"


class StateWriteMode(str, Enum):
    """State write semantics for updates."""
    MUTABLE = "mutable"
    IMMUTABLE = "immutable"
    DERIVED = "derived"


@dataclass
class UpdateItem:
    """Wrapper for state updates with explicit application mode and metadata.

    Notes
    -----
    ``basis`` is retained for backward compatibility. New code should prefer
    ``per_timestep``:
    - per_timestep=True: value already integrated over the current dt
    - per_timestep=False: value is rate-like and may require dt scaling
    """
    value: Any
    mode: UpdateMode = UpdateMode.ACCUMULATE
    write_mode: Optional[StateWriteMode] = None
    unit: Optional[str] = None
    basis: str = "per_step"  # Deprecated; use per_timestep.
    per_timestep: bool = False
    element: Optional[str] = None
    source_functor: Optional[str] = None


@dataclass(frozen=True)
class ExternalIntent:
    """A pure request to an existing I/O owner, never a state variable.

    The descriptor contains selections/options only. Accepted scientific data
    is captured by the simulator before delivery; live tensors, callables,
    writers and simulator references cannot be embedded here. Delivery keys,
    retry and compensation belong to the accepting lifecycle and I/O owner.
    """

    operation: str
    resource: str
    request_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("operation", "resource", "request_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ExternalIntent {name} must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise TypeError("ExternalIntent payload must be a mapping")
        active: set[int] = set()

        def freeze(value: Any) -> Any:
            if value is None or type(value) in (str, bool, int):
                return value
            if type(value) is float:
                if not math.isfinite(value):
                    raise ValueError("ExternalIntent descriptors require finite numbers")
                return value
            if isinstance(value, (Mapping, list, tuple)):
                identity = id(value)
                if identity in active:
                    raise ValueError("ExternalIntent payload must not be cyclic")
                active.add(identity)
                try:
                    if isinstance(value, Mapping):
                        if any(not isinstance(key, str) for key in value):
                            raise TypeError("ExternalIntent descriptor keys must be strings")
                        return MappingProxyType({key: freeze(item) for key, item in value.items()})
                    return tuple(freeze(item) for item in value)
                finally:
                    active.remove(identity)
            raise TypeError(
                f"ExternalIntent payload accepts data descriptors only, got {type(value).__name__}"
            )

        object.__setattr__(self, "payload", freeze(self.payload))


def partition_outputs(
    updates: Mapping[str, Any],
    *,
    declared_outputs: Collection[str] = (),
    write_modes: Optional[Mapping[str, StateWriteMode]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, ExternalIntent]]:
    """Project the existing output mapping into state, diagnostics and intents.

    This is the sole channel classifier. Reserved diagnostic names and DERIVED
    are explicit syntax, not guesses from values or absent output declarations.
    Callers may supply write modes already resolved by StateManager; this
    function does not resolve schemas, coerce state payloads, access managers,
    copy tensors or do I/O. State candidates retain the existing UpdateItem,
    tuple and mapping boundary forms for their canonical downstream owner.
    """
    if not isinstance(updates, Mapping):
        raise TypeError("Functor outputs must be a mapping")
    state: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    intents: Dict[str, ExternalIntent] = {}
    declared = set(declared_outputs)
    active: set[int] = set()

    def contains_intent(value: Any) -> bool:
        if isinstance(value, ExternalIntent):
            return True
        if isinstance(value, UpdateItem):
            return contains_intent(value.value)
        if isinstance(value, (Mapping, list, tuple)):
            identity = id(value)
            if identity in active:
                raise ValueError("Cyclic output payload is not supported")
            active.add(identity)
            try:
                items = value.values() if isinstance(value, Mapping) else value
                return any(contains_intent(item) for item in items)
            finally:
                active.remove(identity)
        return False

    for name, update in updates.items():
        if not isinstance(name, str) or not name:
            raise TypeError("Output names must be non-empty strings")
        mode = update.write_mode if isinstance(update, UpdateItem) else None
        if write_modes is not None:
            mode = write_modes.get(name, mode)
        is_diagnostic = (
            name == "diagnostics" or name.startswith(("diagnostics.", "__"))
            or mode == StateWriteMode.DERIVED
        )
        if isinstance(update, ExternalIntent):
            if is_diagnostic or name in declared:
                raise ValueError("An external intent cannot also declare a scientific output")
            intents[name] = update
        elif contains_intent(update):
            raise TypeError("ExternalIntent must be a top-level output value")
        elif is_diagnostic:
            diagnostics[name] = update
        else:
            state[name] = update
    return state, diagnostics, intents


@dataclass
class StateVariableSpec:
    """Specification of a state variable for schema validation."""
    name: str
    units: str
    bounds: Optional[Tuple[float, float]] = None
    conserved: bool = False
    spatial_dims: int = 0
    required: bool = True
    default: Optional[Tensor] = None
    write_mode: StateWriteMode = StateWriteMode.MUTABLE
    # SCM fields (optional)
    cf_standard_name: Optional[str] = None
    kind: str = "state"
    owner: Optional[str] = None
    writers: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    incrementable: bool = True
    accumulates: Optional[str] = None
    flux_in: List[str] = field(default_factory=list)
    flux_out: List[str] = field(default_factory=list)
    numerics: Optional[Dict[str, float]] = None
    # Compatibility fields used by some tests
    description: str = ""
    default_value: Optional[float] = None
    variable_type: StateVariableType = StateVariableType.SCALAR


@dataclass
class ConservationSummary:
    """Summary object for conservation checks and diagnostics collection."""
    total_violations: int = field(kw_only=True)
    warnings: List[str] = field(default_factory=list)
    custom_checks: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    # Per-element reports
    carbon_report: Optional["ElementConservationReport"] = None
    nitrogen_report: Optional["ElementConservationReport"] = None
    phosphorus_report: Optional["ElementConservationReport"] = None
    balance_error: Dict[str, float] = field(default_factory=dict)
    # Dynamic tolerance tracking
    dt: Optional[float] = None
    dynamic_tolerance_used: bool = False
    # The tolerance applied to this summary (dt * abs_tol with a step, abs_tol
    # without one), recorded by the producer.
    tolerance_value: float = field(kw_only=True)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        key_lower = key.lower()
        return key_lower in {"carbon", "nitrogen", "phosphorus", "water"}

    def __getitem__(self, key: str) -> Dict[str, float]:
        key_lower = key.lower()
        report = None
        if key_lower == "carbon":
            report = self.carbon_report
        elif key_lower == "nitrogen":
            report = self.nitrogen_report
        elif key_lower == "phosphorus":
            report = self.phosphorus_report
        if report is not None:
            return {
                "balance_error": report.balance_error,
                "relative_change": report.relative_error,
                "relative_error": report.relative_error,
            }
        if key_lower == "water":
            balance = self.balance_error.get("W", 0.0)
            return {
                "balance_error": balance,
                "relative_change": 0.0,
                "relative_error": 0.0,
            }
        raise KeyError(key)


@dataclass
class ElementConservationReport:
    """Detailed report for a single conserved element."""
    balance_error: float
    relative_error: float


__all__ = [
    "StateVariableType",
    "UpdateMode",
    "StateWriteMode",
    "UpdateItem",
    "ExternalIntent",
    "partition_outputs",
    "StateVariableSpec",
    "ConservationSummary",
    "ElementConservationReport",
]
