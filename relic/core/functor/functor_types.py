"""Core functor type aliases and metadata containers.

This module holds shared enums, dataclasses, and type aliases to keep
functor_base.py focused on execution logic.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from torch import Tensor

from relic.utils.time_constants import (
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
)

from .iphs_metadata import ensure_iphs_equivalent_expression

ConfigDict = Dict[str, Any]
ParameterDict = Dict[str, Union[float, int, str, bool, Tensor]]
MetadataDict = Dict[str, Any]
ValidationErrors = List[str]
DependencyList = List[str]
DependencyDeclaration = Union[DependencyList, Dict[str, List[str]]]

class ProcessCategory(Enum):
    """
    Scientific process categories following the four-pillar architecture.

    Maps functors to abstract base classes for validation and type checking.
    """
    PHYSICAL_FIELD = "physical_field"
    BIOLOGICAL_PROCESS = "biological_process"
    TRANSPORT_MECHANISM = "transport_mechanism"
    SCALE_COUPLING = "scale_coupling"

    def to_namespace_prefix(self) -> str:
        """Map category to canonical parameter namespace prefix."""
        mapping = {
            "scale_coupling": "coupling.scales",
            "biological_process": "biosphere",
            "physical_field": "physical",
            "transport_mechanism": "transport",
        }
        return mapping.get(self.value, self.value)


class ExecutionPhase(Enum):
    """
    Execution phases for functor scheduling during time stepping.
    """
    INITIALIZATION = "initialization"
    PREPROCESSING = "preprocessing"
    MAIN_PROCESSES = "main_processes"
    COUPLING = "coupling"
    POSTPROCESSING = "postprocessing"
    VALIDATION = "validation"


class FunctorState(Enum):
    """
    Functor execution state for debugging and monitoring.
    """
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


class SpatialScale(Enum):
    """
    Spatial scale classification for functor output variables.

    Used by the simulator's scale consistency checker to validate
    that coupled functors operate at compatible spatial resolutions.
    """
    POINT = "point"           # 0D / site-level
    PATCH = "patch"           # Sub-grid tile
    GRID_CELL = "grid_cell"   # 2D grid cell
    HILLSLOPE = "hillslope"   # Hillslope unit (between patch and basin)
    BASIN = "basin"           # Catchment / basin aggregate
    LANDSCAPE = "landscape"   # Multi-cell aggregate (alias-level for basin)
    GRID_2D = "grid_2d"       # Structured 2D grid
    GRID_3D = "grid_3d"       # Structured 3D grid
    REGIONAL = "regional"     # Regional domain
    GLOBAL = "global"         # Full globe
    UNKNOWN = "unknown"       # Not declared

    @classmethod
    def from_value(cls, value: Union["SpatialScale", str, None]) -> "SpatialScale":
        """Build SpatialScale from a member, a member value, or None.

        ``None`` and the value ``"unknown"`` give UNKNOWN, the "not declared"
        sentinel: it is the FunctorMetadata default, ``FunctorMetadata.to_dict``
        writes it as ``"unknown"``, and only this undeclared case leaves scale
        detection to the heuristics (scale_ops Tier 2). A string matches a
        member value case-insensitively after stripping surrounding
        whitespace; there are no aliases. Any other string, the empty string
        included, names no scale and raises ValueError listing the member
        values; a value that is neither a member, a string, nor None raises
        TypeError.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        if not isinstance(value, str):
            raise TypeError(
                "SpatialScale.from_value expects a SpatialScale member, a string, "
                f"or None; got {type(value).__name__} {value!r}."
            )
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unrecognized spatial scale {value!r}; expected one of "
            f"{[member.value for member in cls]} (case-insensitive), a SpatialScale "
            "member, or None."
        )


class TemporalScale(Enum):
    """
    Temporal scale classification for functor execution frequency.

    Used by determine_execution_group() to automatically assign functors to
    fast/medium/slow execution groups when no explicit group is set. Fixed SI
    scales use the time-constant SSOT. Calendar- or schedule-dependent scales
    require their owner to supply the realized period duration.
    """

    SECOND = "second"                 # Sub-second to second processes
    MINUTE = "minute"                 # Minute-scale processes
    HOUR = "hour"                     # Hourly processes
    SUB_DAILY = "sub_daily"           # Sub-daily (few hours)
    DAY = "day"                       # Daily processes
    WEEK = "week"                     # Weekly processes
    MONTH = "month"                   # Monthly processes
    SEASON = "season"                 # Seasonal processes
    YEAR = "year"                     # Annual processes
    STATIC = "static"                 # Parameters that do not change
    STEADY_STATE = "steady_state"     # Equilibrium / diagnostic
    UNKNOWN = "unknown"               # Not declared

    @classmethod
    def from_value(cls, value: Union["TemporalScale", str, None]) -> "TemporalScale":
        """Build TemporalScale from a member, a member value, or None.

        ``None`` and the value ``"unknown"`` give UNKNOWN, the "not declared"
        sentinel: it is the FunctorMetadata default and
        ``FunctorMetadata.to_dict`` writes it as ``"unknown"``. A string
        matches a member value case-insensitively after stripping surrounding
        whitespace; there are no aliases (an execution-group name such as
        "fast" is not a scale). Any other string, the empty string included,
        names no scale and raises ValueError listing the member values; a
        value that is neither a member, a string, nor None raises TypeError.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        if not isinstance(value, str):
            raise TypeError(
                "TemporalScale.from_value expects a TemporalScale member, a string, "
                f"or None; got {type(value).__name__} {value!r}."
            )
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unrecognized temporal scale {value!r}; expected one of "
            f"{[member.value for member in cls]} (case-insensitive), a TemporalScale "
            "member, or None."
        )

    def to_characteristic_time_days(self, *, period_days: float | None = None) -> float:
        """Resolve the characteristic execution period in days.

        References: CF Conventions calendar and time-coordinate contracts.
        Mathematical expression: tau_d = tau_s / S_d for fixed SI scales.
        Parameter meanings: tau_s is the scale duration and S_d is seconds per day;
        period_days is the realized duration supplied by a calendar or scheduler.
        Units: input period_days and returned tau_d are in days.
        Domain/constraints: dynamic periods must be finite and strictly positive.
        Sign convention: elapsed durations are positive; static states use zero.
        """
        if self is TemporalScale.SECOND:
            return 1 / float(SECONDS_PER_DAY)
        if self is TemporalScale.MINUTE:
            return float(SECONDS_PER_MINUTE) / float(SECONDS_PER_DAY)
        if self is TemporalScale.HOUR:
            return float(SECONDS_PER_HOUR) / float(SECONDS_PER_DAY)
        if self is TemporalScale.DAY:
            return float(SECONDS_PER_DAY) / float(SECONDS_PER_DAY)
        if self in (TemporalScale.STATIC, TemporalScale.STEADY_STATE):
            return 0.0
        if self is TemporalScale.UNKNOWN:
            raise ValueError("TemporalScale.UNKNOWN has no characteristic period")
        if period_days is None:
            raise ValueError(
                f"period_days is required for calendar- or schedule-dependent scale '{self.value}'"
            )
        if isinstance(period_days, bool):
            raise TypeError("period_days must be a finite real duration in days")
        resolved_period_days = float(period_days)
        if not math.isfinite(resolved_period_days) or resolved_period_days <= 0.0:
            raise ValueError("period_days must be finite and strictly positive")
        return resolved_period_days


class VariableKind(Enum):
    """
    Classification of variables produced/consumed by functors.

    Enables the simulator to distinguish state increments from
    diagnostic outputs and forcing inputs.
    """
    STATE = "state"             # Prognostic state variable
    FLUX = "flux"               # Rate / tendency
    DIAGNOSTIC = "diagnostic"   # Non-prognostic output
    PARAMETER = "parameter"     # Static input parameter
    FORCING = "forcing"         # External driver
    UNKNOWN = "unknown"         # Not declared


class SubFlowAccuracy(Enum):
    """
    Local accuracy of a Functor's own sub-flow for one ``(time, dt)`` call.

    The simulator binds the order of its operator splitting to these
    declarations (``execution_loop.splitting_order_report``).

    References: Hairer, Lubich and Wanner (2006), Geometric Numerical
    Integration, Sect. II.4-II.5 (order of compositions of methods).
    Mathematical expression: Phi_num(t, dt) x - Phi_exact(t, dt) x = O(dt^{p+1})
    for a sub-flow of order p; EXACT means the call returns the exact sub-flow
    over [t, t + dt] (p = infinity).
    Parameter meanings: p is the local order of the sub-integrator used by the
    Functor inside one call (with its own sub-stepping).
    Units: dimensionless order.
    Domain/constraints: UNDECLARED (the default) makes any splitting-order
    claim unverified; values are declared by the Functor author, not inferred.
    Sign convention: not applicable.
    """

    EXACT = "exact"
    ORDER_2 = "order_2"
    ORDER_1 = "order_1"
    UNDECLARED = "undeclared"

    @property
    def order(self) -> Optional[float]:
        """Local order p (infinity for EXACT, None for UNDECLARED)."""
        if self is SubFlowAccuracy.EXACT:
            return math.inf
        if self is SubFlowAccuracy.ORDER_2:
            return 2
        if self is SubFlowAccuracy.ORDER_1:
            return 1
        return None


class ConflictResolutionPolicy(Enum):
    """
    Output overlap policy for Functor metadata contracts.

    INHERIT follows simulator.strict_dependencies.
    WARN always warns on overlap.
    ERROR always raises on overlap.
    EXCLUSIVE is a strict single-writer contract (same enforcement as ERROR).
    """

    INHERIT = "inherit"
    WARN = "warn"
    ERROR = "error"
    EXCLUSIVE = "exclusive"


@dataclass
class FunctorMetadata:
    """
    Comprehensive metadata container for scientific process documentation.
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    category: ProcessCategory = ProcessCategory.BIOLOGICAL_PROCESS
    execution_phase: ExecutionPhase = ExecutionPhase.MAIN_PROCESSES
    mathematical_expression: str = ""
    references: List[str] = field(default_factory=list)
    parameters: Dict[str, str] = field(default_factory=dict)
    input_variables: List[str] = field(default_factory=list)
    output_variables: List[str] = field(default_factory=list)
    optional_output_variables: List[str] = field(default_factory=list)
    conservation_laws: List[str] = field(default_factory=list)
    spatial_dimensions: int = 0
    temporal_scale: TemporalScale = TemporalScale.UNKNOWN
    substep_count: int = 1
    sub_flow_accuracy: SubFlowAccuracy = SubFlowAccuracy.UNDECLARED
    coupling_interfaces: List[str] = field(default_factory=list)
    form_contracts: tuple[object, ...] = field(default_factory=tuple)
    hodge_contracts: tuple[object, ...] = field(default_factory=tuple)
    iphs_terms: tuple[str, ...] = field(default_factory=tuple)
    port_power_contracts: tuple[object, ...] = field(default_factory=tuple)
    stoichiometric_contracts: tuple[object, ...] = field(default_factory=tuple)
    structural_role: tuple[str, ...] = field(default_factory=tuple)
    spatial_scale: SpatialScale = SpatialScale.UNKNOWN
    variable_kind: VariableKind = VariableKind.UNKNOWN
    async_safe: bool = False
    stream_hint: Optional[str] = None
    conflict_resolution_policy: ConflictResolutionPolicy = (
        ConflictResolutionPolicy.INHERIT
    )

    def __contains__(self, item: str) -> bool:
        """Allow membership checks against metadata fields."""
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        """Return field value with default when missing."""
        return getattr(self, item, default)

    def __post_init__(self) -> None:
        """Validate metadata consistency after initialization."""
        try:
            self.spatial_scale = SpatialScale.from_value(self.spatial_scale)
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"Invalid spatial_scale for functor '{self.name}': {exc}"
            ) from exc
        try:
            self.temporal_scale = TemporalScale.from_value(self.temporal_scale)
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"Invalid temporal_scale for functor '{self.name}': {exc}"
            ) from exc
        try:
            self.sub_flow_accuracy = SubFlowAccuracy(self.sub_flow_accuracy)
        except ValueError as exc:
            raise ValueError(
                f"Invalid sub_flow_accuracy {self.sub_flow_accuracy!r} for functor "
                f"'{self.name}'; expected one of "
                f"{[member.value for member in SubFlowAccuracy]}"
            ) from exc
        if isinstance(self.optional_output_variables, str):
            raise TypeError("optional_output_variables must be a sequence of names")
        self.optional_output_variables = list(self.optional_output_variables)
        undeclared_optional_outputs = sorted(
            set(self.optional_output_variables) - set(self.output_variables)
        )
        if undeclared_optional_outputs:
            raise ValueError(
                "optional_output_variables must be declared in output_variables: "
                f"{undeclared_optional_outputs}"
            )
        if len(self.optional_output_variables) != len(set(self.optional_output_variables)):
            raise ValueError("optional_output_variables must not contain duplicates")
        self.mathematical_expression = ensure_iphs_equivalent_expression(
            self.mathematical_expression,
            category=self.category,
            variable_kind=self.variable_kind,
            parameters=self.parameters,
            conservation_laws=self.conservation_laws,
            input_variables=self.input_variables,
            output_variables=self.output_variables,
            functor_name=self.name,
        )
        if not self.name:
            raise ValueError("Functor name cannot be empty")
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid functor name format: {self.name}")
        if not self.description:
            warnings.warn(f"No description provided for functor {self.name}")
        if not self.references:
            warnings.warn(f"No scientific references provided for functor {self.name}")
        if not self.input_variables and not self.output_variables:
            warnings.warn(
                f"Functor '{self.name}' has empty input_variables and output_variables. "
                f"Consider declaring them for overlap detection and provenance tracking."
            )
        if self.spatial_dimensions not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid spatial dimensions: {self.spatial_dimensions}")
        if isinstance(self.substep_count, bool) or not isinstance(self.substep_count, int):
            raise TypeError(
                f"substep_count must be an integer for functor '{self.name}', "
                f"got {self.substep_count!r}"
            )
        if self.substep_count < 1:
            raise ValueError(
                f"substep_count must be >= 1 for functor '{self.name}', "
                f"got {self.substep_count}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category.value,
            "execution_phase": self.execution_phase.value,
            "mathematical_expression": self.mathematical_expression,
            "references": self.references,
            "parameters": self.parameters,
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
            "optional_output_variables": self.optional_output_variables,
            "conservation_laws": self.conservation_laws,
            "spatial_dimensions": self.spatial_dimensions,
            "temporal_scale": self.temporal_scale.value,
            "substep_count": self.substep_count,
            "sub_flow_accuracy": self.sub_flow_accuracy.value,
            "coupling_interfaces": self.coupling_interfaces,
            "form_contracts": self.form_contracts,
            "hodge_contracts": self.hodge_contracts,
            "iphs_terms": self.iphs_terms,
            "port_power_contracts": self.port_power_contracts,
            "stoichiometric_contracts": self.stoichiometric_contracts,
            "structural_role": self.structural_role,
            "spatial_scale": self.spatial_scale.value,
            "variable_kind": self.variable_kind.value,
            "async_safe": self.async_safe,
            "stream_hint": self.stream_hint,
            "conflict_resolution_policy": self.conflict_resolution_policy.value,
        }


__all__ = [
    "ConfigDict",
    "ParameterDict",
    "MetadataDict",
    "ValidationErrors",
    "DependencyList",
    "DependencyDeclaration",
    "ProcessCategory",
    "ExecutionPhase",
    "FunctorState",
    "SpatialScale",
    "TemporalScale",
    "VariableKind",
    "SubFlowAccuracy",
    "ConflictResolutionPolicy",
    "FunctorMetadata",
    "ensure_iphs_equivalent_expression",
]
