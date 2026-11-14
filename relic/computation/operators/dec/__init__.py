"""DEC structural views and adapters for RELIC computation operators."""

from .forms import (
    BoundaryRole,
    ChainSpaceSpec,
    CochainSpaceSpec,
    DiagnosticSpec,
    Extensivity,
    FormDegree,
    ObservationOperatorSpec,
    Primality,
    TypedChain,
    TypedCochain,
    cochain_space_from_ssot_entry,
)
from .complex import DECComplexBundle
from .hodge import HodgeStarSpec
from .adapters import typed_cochain_delta_to_update_item
from .chain_map import LinearChainMap, close_within

__all__ = [
    "BoundaryRole",
    "ChainSpaceSpec",
    "CochainSpaceSpec",
    "DiagnosticSpec",
    "Extensivity",
    "FormDegree",
    "ObservationOperatorSpec",
    "Primality",
    "TypedChain",
    "TypedCochain",
    "cochain_space_from_ssot_entry",
    "DECComplexBundle",
    "HodgeStarSpec",
    "typed_cochain_delta_to_update_item",
    "LinearChainMap",
    "close_within",
]
