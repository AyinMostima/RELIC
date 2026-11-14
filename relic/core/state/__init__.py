"""State subpackage for StateManager and related helpers."""

from .manager import (
    StateManager,
    feature_manager,
    variable_name_mapper,
)
from .types import (
    StateVariableType,
    UpdateMode,
    StateWriteMode,
    UpdateItem,
    ExternalIntent,
    partition_outputs,
    StateVariableSpec,
    ConservationSummary,
    ElementConservationReport,
)
from .conservation import check_conservation

__all__ = [
    "StateManager",
    "feature_manager",
    "variable_name_mapper",
    "StateVariableType",
    "UpdateMode",
    "StateWriteMode",
    "UpdateItem",
    "ExternalIntent",
    "partition_outputs",
    "StateVariableSpec",
    "ConservationSummary",
    "ElementConservationReport",
    "check_conservation",
]
