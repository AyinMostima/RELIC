"""Functor subpackage with base class, bridges, and shared types."""

from .functor_base import (
    FunctorBase,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    FunctorState,
)
from ..abstractions import EnvironmentDict, FluxDict, StateDict
from .functor_bridges import (
    BiologicalProcessFunctor,
    TransportMechanismFunctor,
    PhysicalFieldFunctor,
    ScaleCouplingFunctor,
)
from .functor_exceptions import (
    FunctorException,
    ConfigurationError,
    DependencyError,
    ExecutionError,
    ConservationError,
)
from .functor_types import (
    ConfigDict,
    ParameterDict,
    MetadataDict,
    ValidationErrors,
    DependencyList,
    DependencyDeclaration,
    ConflictResolutionPolicy,
    SpatialScale,
    SubFlowAccuracy,
    TemporalScale,
    VariableKind,
)
from .functor_bridge_utils import BridgeParameterMixin, _ensure_tensor
from .iphs_protocol import IPHSDECFunctor
from .stochastic_substream import FunctorSubstreamMixin

__all__ = [
    "FunctorBase",
    "FunctorMetadata",
    "ProcessCategory",
    "ExecutionPhase",
    "FunctorState",
    "SpatialScale",
    "TemporalScale",
    "VariableKind",
    "SubFlowAccuracy",
    "ConflictResolutionPolicy",
    "StateDict",
    "EnvironmentDict",
    "FluxDict",
    "BiologicalProcessFunctor",
    "TransportMechanismFunctor",
    "PhysicalFieldFunctor",
    "ScaleCouplingFunctor",
    "FunctorException",
    "ConfigurationError",
    "DependencyError",
    "ExecutionError",
    "ConservationError",
    "ConfigDict",
    "ParameterDict",
    "MetadataDict",
    "ValidationErrors",
    "DependencyList",
    "DependencyDeclaration",
    "BridgeParameterMixin",
    "_ensure_tensor",
    "IPHSDECFunctor",
    "FunctorSubstreamMixin",
]
