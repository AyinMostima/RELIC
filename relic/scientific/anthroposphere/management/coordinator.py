"""
Management Coordinator Module

This module implements a unified coordinator for anthroposphere management activities,
handling event scheduling, priority resolution, resource allocation, and integration
with the event scheduler system.

The coordinator serves as the central hub for:
- Scheduling and coordinating management events
- Resolving conflicts between competing activities
- Ensuring resource constraints and quotas
- Maintaining consistency across management practices
- Integrating with the broader ecosystem simulation

References:
    - Integrated agricultural systems management approaches
    - Conflict resolution in multi-objective optimization
    - Resource allocation algorithms in agricultural systems
    - Decision support systems for farm management

Mathematical Expression:
    Priority-based scheduling:
    P_eff = P_base * W_resource * W_timing * W_conflict
    
    where:
    - P_eff is effective priority [1]
    - P_base is base event priority [1]
    - W_resource is resource availability weight [1]
    - W_timing is timing constraint weight [1]
    - W_conflict is conflict resolution weight [1]
"""

import copy
import math

import torch
from typing import Dict, List, Mapping, Optional, Tuple, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import logging
from collections import defaultdict

from ....core.functor import ScaleCouplingFunctor, TemporalScale
from ....core.functor import FunctorMetadata, ProcessCategory, ExecutionPhase, SpatialScale, VariableKind
from ....core.functor.functor_bridge_utils import NamespaceDefaultMixin
from ....core.state import UpdateItem, UpdateMode
from ....computation.operators.dec.forms import CochainSpaceSpec
from ....computation.operators.dec.hodge import HodgeStarSpec
from ....computation.operators.iphs import PortSpecIPHS
from ....coupling.scales.temporal.calendar import CalendarHandler
from ....utils.physical_constants import get_physical_constant
from .event_scheduler import (
    ConflictResolution,
    EventConflict,
    EventPriority,
    EventScheduler,
    EventType,
    ManagementEvent,
    ResourceQuota,
    resolve_day_of_year,
)

logger = logging.getLogger(__name__)


class ManagementStrategy(Enum):
    """Management strategy approaches"""
    CONVENTIONAL = "conventional"      # Traditional intensive management
    CONSERVATION = "conservation"      # Conservation-focused management
    ORGANIC = "organic"               # Organic farming practices
    PRECISION = "precision"           # Precision agriculture
    INTEGRATED = "integrated"         # Integrated pest/nutrient management
    SUSTAINABLE = "sustainable"      # Sustainable intensification
    REGENERATIVE = "regenerative"    # Regenerative agriculture
    PRODUCTIVITY_FOCUSED = "productivity_focused"  # Yield-oriented coordination
    SUSTAINABILITY_FOCUSED = "sustainability_focused"  # Resource/impact balancing
    BALANCED = "balanced"            # Balanced productivity and sustainability
    CONSERVATION_FOCUSED = "conservation_focused"  # Conservation-first strategy


class CoordinationMode(Enum):
    """Coordination modes for management activities"""
    STRICT_PRIORITY = "strict_priority"    # Strict priority-based scheduling
    ADAPTIVE = "adaptive"                  # Adaptive scheduling based on conditions
    OPTIMAL = "optimal"                    # Optimization-based coordination
    COOPERATIVE = "cooperative"           # Cooperative multi-objective approach
    REACTIVE = "reactive"                 # Reactive coordination (event-driven)


@dataclass
class ManagementObjective:
    """
    Management objective specification for multi-objective optimization
    
    Attributes:
        name: Objective name
        weight: Importance weight in multi-objective optimization
        measurement_function: Function to evaluate objective (without it the
            event heuristic of _heuristic_objective_score scores the objective)
    """
    name: str
    weight: float = 1.0
    measurement_function: Optional[
        Callable[[List[ManagementEvent], Dict[str, torch.Tensor]], float]
    ] = None


@dataclass
class CoordinationResult:
    """
    Result of management coordination
    
    Attributes:
        scheduled_events: List of scheduled management events
        conflicts_resolved: Number of conflicts resolved
        resource_allocation: Final resource allocation
        objective_scores: Achievement scores for management objectives
        coordination_status: Status of coordination process
        warnings: List of warnings generated during coordination
    """
    scheduled_events: List[ManagementEvent]
    conflicts_resolved: int
    resource_allocation: Dict[str, float]
    objective_scores: Dict[str, float]
    coordination_status: str
    warnings: List[str] = field(default_factory=list)


class ManagementCoordinator(NamespaceDefaultMixin, ScaleCouplingFunctor):
    """
    Unified Management Coordinator
    
    Coordinates all anthroposphere management activities including irrigation,
    tillage, harvest, grazing, and fire management with priority-based conflict
    resolution and resource allocation.
    
    The coordinator integrates with the EventScheduler to provide:
    - Centralized management event coordination
    - Resource quota enforcement
    - Conflict resolution between activities
    - Multi-objective optimization
    - Adaptive scheduling based on environmental conditions
    
    Parameters:
        management_strategy: Overall management strategy approach
        coordination_mode: Mode of coordination between activities
        management_objectives: List of management objectives to optimize
        resource_quotas: Available resource quotas
        priority_weights: Weights for different priority factors
    """

    REQUIRED_DEPENDENCIES: List[str] = []
    OPTIONAL_DEPENDENCIES: List[str] = []
    LAGGED_DEPENDENCIES: List[str] = [
        "irrigation_management",
        "harvest_management",
        "grazing_management",
        "fire_management",
    ]
    REQUIRED_STATE_INPUTS: Tuple[str, ...] = (
        "soil_moisture",
        "soil_nitrogen",
        "soil_organic_carbon",
        "crop_maturity",
        "aboveground_biomass",
        "air_temperature",
        "irrigation_flux",
        "harvest_flux",
        "grazing_flux",
        "co2_emission",
        "field_capacity",
        "wilting_point",
        "evapotranspiration_flux",
        "precipitation_flux",
    )
    OPTIONAL_STATE_INPUTS: Tuple[str, ...] = (
        "day_of_year",
    )
    OUTPUT_UNITS: Dict[str, str] = {
        "coordination_status": "1",
        "total_irrigation_flux": "kg m-2 s-1",
        "total_harvest_flux": "kg DM m-2 s-1",
        "total_grazing_flux": "kg DM m-2 s-1",
        "total_fire_co2_flux": "kg CO2 m-2 s-1",
        "irrigation_demand_flux": "kg m-2 s-1",
        "fertilizer_product_flux": "kg product m-2 s-1",
        "tillage_intensity": "1",
        "diagnostics.management_coordinator.active": "1",
        "diagnostics.management_coordinator.events_executed": "1",
        "diagnostics.management_coordinator.irrigation_events_count": "1",
        "diagnostics.management_coordinator.tillage_events_count": "1",
        "diagnostics.management_coordinator.harvest_events_count": "1",
        "diagnostics.management_coordinator.grazing_events_count": "1",
        "diagnostics.management_coordinator.fire_events_count": "1",
        "diagnostics.management_coordinator.priority_weights_sum": "1",
        "diagnostics.management_coordinator.interaction_effects_count": "1",
        "diagnostics.management_coordinator.conflict_resolution_applied": "1",
        "diagnostics.coordinator_mass_balance_error": "kg DM m-2 s-1",
        "diagnostics.management_coordinator.canonical_iphs_runtime": "1",
    }
    
    parameter_lookup_namespace = "anthroposphere.management.coordinator"
    parameter_lookup_source_order = ("config", "manager")
    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    _RUNTIME_STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        name: Union[str, Dict[str, Any]] = "management_coordinator",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize management coordinator
        
        Args:
            name: Coordinator name
            config: Configuration dictionary
        """
        if isinstance(name, dict):
            incoming_config = name
            base_name = incoming_config.get("name", "management_coordinator")
            name = base_name if base_name.endswith("coordinator") else f"{base_name}_coordinator"
            if config is not None:
                merged_config = {**incoming_config, **config}
            else:
                merged_config = incoming_config
            config = merged_config
        # Set up metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.1.0",
            description="Management coordinator that schedules and coordinates anthroposphere activities through RELICSimulator",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "q_i = w_i * (1 + s_i); p_i = q_i / sum_j(q_j); "
                "c = sum_r(R_r / Q_r); eta = clip(N / c, 0, 1); "
                "0 <= U_r <= Q_r for each resource r; "
                "F_irr,net = max(E - P, 0) with P_e = P, a simplification that "
                "underestimates the net irrigation requirement (FAO P_e <= P)"
            ),
            references=[
                "Allen et al. (1998), FAO Irrigation and Drainage Paper 56",
                "Lal (2004), Science 304, 1623-1627",
                "Matson et al. (1997), Science 277, 504-509",
                "Robertson and Swinton (2005), Frontiers in Ecology and the Environment 3, 38-46",
                "van der Schaft and Maschke (1995), AEU 49, 362-371",
            ],
            parameters={
                "anthroposphere.management.coordinator.management_strategy": (
                    "Configured management policy selector [dimensionless]"
                ),
                "anthroposphere.management.coordinator.coordination_mode": (
                    "Mode of coordination between activities [dimensionless]"
                ),
                "anthroposphere.management.coordinator.conflict_resolution_strategy": (
                    "Strategy for resolving management conflicts [dimensionless]"
                ),
                "anthroposphere.management.coordinator.coordination_interval_days": (
                    "Policy evaluation interval [day]"
                ),
                "anthroposphere.management.coordinator.planning_horizon_days": (
                    "Planning horizon for management scheduling decisions [day]"
                ),
                "anthroposphere.management.coordinator.statistics_history_limit": (
                    "Number of recent coordination results kept by the statistics [dimensionless]"
                ),
                "anthroposphere.management.coordinator.decision_threshold": (
                    "Threshold for applying coordination decisions [dimensionless]"
                ),
                "anthroposphere.management.coordinator.high_priority_threshold": (
                    "Need score above which an event is high priority [dimensionless]"
                ),
                "anthroposphere.management.coordinator.candidate_need_threshold": (
                    "Minimum need score of a candidate event [dimensionless]"
                ),
                "anthroposphere.management.coordinator.coordination_efficiency": (
                    "Efficiency scalar for coordination decisions [dimensionless]"
                ),
                "anthroposphere.management.coordinator.interaction_factor": (
                    "Interaction multiplier between management practices [dimensionless]"
                ),
                "anthroposphere.management.coordinator.interaction_multiplier_min": (
                    "Lower bound of the management interaction multiplier [dimensionless]"
                ),
                "anthroposphere.management.coordinator.interaction_multiplier_max": (
                    "Upper bound of the management interaction multiplier [dimensionless]"
                ),
                "anthroposphere.management.coordinator.maturity_factor_min": (
                    "Lower bound of the crop-maturity multiplier [dimensionless]"
                ),
                "anthroposphere.management.coordinator.maturity_factor_max": (
                    "Upper bound of the crop-maturity multiplier [dimensionless]"
                ),
                "anthroposphere.management.coordinator.harvest_biomass_threshold": (
                    "Aboveground biomass threshold for harvest readiness [kg m-2]"
                ),
                "anthroposphere.management.coordinator.score_limits.benefit_max": (
                    "Upper bound of the normalized productivity and soil-health benefit scores "
                    "[dimensionless]"
                ),
                "anthroposphere.management.coordinator.event_defaults.tillage_intensity_class": (
                    "Tillage intensity class of a coordinated tillage event [dimensionless]"
                ),
                "anthroposphere.management.coordinator.priority_weights.productivity": (
                    "Base weight for productivity objectives [dimensionless]"
                ),
                "anthroposphere.management.coordinator.priority_weights.environmental": (
                    "Base weight for environmental impact objectives [dimensionless]"
                ),
                "anthroposphere.management.coordinator.priority_weights.water": (
                    "Base weight for water management objectives [dimensionless]"
                ),
                "anthroposphere.management.coordinator.priority_weights.nutrients": (
                    "Base weight for nutrient objectives [dimensionless]"
                ),
                "anthroposphere.management.coordinator.priority_weights.soil_health": (
                    "Base weight for soil health objectives [dimensionless]"
                ),
                "anthroposphere.management.coordinator.objectives.*": (
                    "Scoring weight of each management objective [dimensionless]"
                ),
                "anthroposphere.management.coordinator.objective_scoring.*": (
                    "Objective score coefficients and neutral score [dimensionless]"
                ),
                "anthroposphere.management.coordinator.strategies.*": (
                    "Nutrient and objective weight multipliers per management strategy "
                    "[dimensionless]"
                ),
                "anthroposphere.management.coordinator.reference.*": (
                    "Positive reference states that normalize the need scores [units per key]"
                ),
                "anthroposphere.management.coordinator.resource_quotas.*": (
                    "Scheduling quota per resource (fuel, labor, machinery, nitrogen, phosphorus, "
                    "water) [units per key]"
                ),
                "anthroposphere.management.coordinator.resource_requirements.*": (
                    "Resource requirement per event of each activity, in the quota units [units "
                    "per key]"
                ),
                "anthroposphere.management.coordinator.need_scores.*": (
                    "Need-score coefficients per activity [units per key]"
                ),
                "anthroposphere.management.coordinator.event_delay_days.*": (
                    "Scheduling delay per activity [day]"
                ),
                "anthroposphere.management.coordinator.event_duration_hours.*": (
                    "Event duration per activity [h]"
                ),
                "anthroposphere.management.coordinator.season_boundaries.*": (
                    "Day of year at which each season starts [day]"
                ),
                "anthroposphere.management.coordinator.status.*": (
                    "Coordination-status weights and neutral score [dimensionless]"
                ),
                "anthroposphere.management.clamp_min": (
                    "Lower bound of the management response factors [dimensionless]"
                ),
                "anthroposphere.management.clamp_max": (
                    "Upper bound of the management response factors [dimensionless]"
                ),
                "anthroposphere.management.fertilization.spring_urea_product_mass": (
                    "Urea product mass of the coordinated spring fertilization [kg product m-2]"
                ),
                "anthroposphere.management.irrigation.default_application_depth": (
                    "Irrigation depth per coordinated event [mm]"
                ),
                "anthroposphere.management.tillage.low_intensity_factor": (
                    "Intensity multiplier of low-intensity tillage [dimensionless]"
                ),
                "anthroposphere.management.fire.default_fire_type": (
                    "Fire type of a coordinated fire event [dimensionless]"
                ),
                "anthroposphere.management.fire.default_fuel_load_fraction": (
                    "Fraction of aboveground biomass treated as fuel [dimensionless]"
                ),
                "anthroposphere.management.fire.fuel_consumption_fraction": (
                    "Fraction of the available fuel consumed [dimensionless]"
                ),
                "anthroposphere.management.grazing.base_efficiency": (
                    "Baseline efficiency scalar of grazing management [dimensionless]"
                ),
                "anthroposphere.management.grazing.daily_intake_rate": (
                    "Daily dry-matter intake per body weight [kg DM kg-1 body weight day-1]"
                ),
                "anthroposphere.management.grazing.manure_return_fraction": (
                    "Fraction of consumed nutrients returned as manure [dimensionless]"
                ),
                "anthroposphere.management.grazing.default_schedule.spring.body_weight": (
                    "Animal body weight of the spring grazing event [kg]"
                ),
                "anthroposphere.management.grazing.default_schedule.spring.stocking_rate": (
                    "Stocking rate of the spring grazing event [animals ha-1]"
                ),
                "anthroposphere.management.harvest.default_harvest_index": (
                    "Harvest index of a coordinated harvest [dimensionless]"
                ),
                "validation.conservation.conservation_tolerance": (
                    "General relative tolerance of the conservation checks [dimensionless]"
                ),
            },
            input_variables=list(
                self.REQUIRED_STATE_INPUTS + self.OPTIONAL_STATE_INPUTS
            ),
            output_variables=list(self.OUTPUT_UNITS),
            conservation_laws=["mass", "resource_quotas"],
            spatial_dimensions=2,
            temporal_scale=TemporalScale.DAY,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
            coupling_interfaces=["biosphere", "pedosphere", "hydrosphere", "atmosphere"]
        )
        
        # Scale coupling: grid-cell management state is aggregated to the domain
        # coordination decisions (upscale: mean over the cells) and the decisions
        # are broadcast back to the cells (downscale).
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
            source_scale="grid_cell",
            target_scale="management_domain",
            coupling_type="spatial",
        )
        configured_calendar = (config or {}).get("calendar_type")
        self._configured_calendar_type = (
            None
            if configured_calendar is None
            else str(configured_calendar)
        )
        self.runtime_path = "canonical_iphs_management_port_coordinator"
        self.optional_input_variables = set(self.OPTIONAL_STATE_INPUTS)
        self._last_management_mass_residual = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )

        self.seconds_per_day = float(get_physical_constant("seconds_per_day"))
        self.seconds_per_hour = float(get_physical_constant("seconds_per_hour"))

        # Initialize event scheduler
        conflict_strategy_param = self.get_parameter_value(
            "conflict_resolution_strategy",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",), required=True)
        conflict_strategy_name = (
            self._scalar_value(conflict_strategy_param)
            if isinstance(conflict_strategy_param, torch.Tensor)
            else conflict_strategy_param
        )
        try:
            conflict_strategy = ConflictResolution[str(conflict_strategy_name).upper()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported management conflict resolution strategy: {conflict_strategy_name}"
            ) from exc
        self.scheduler = EventScheduler(conflict_strategy=conflict_strategy, enable_logging=True)
        
        # Management coordination state (no direct functor instances)
        # All management functors will be managed by RELICSimulator
        management_strategy_name = self.get_parameter_value(
            "management_strategy",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
            required=True,
        )
        coordination_mode_name = self.get_parameter_value(
            "coordination_mode",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
            required=True,
        )
        try:
            self.management_strategy = ManagementStrategy(str(management_strategy_name))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported management strategy: {management_strategy_name}"
            ) from exc
        try:
            self.coordination_mode = CoordinationMode(str(coordination_mode_name))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported management coordination mode: {coordination_mode_name}"
            ) from exc
        self.strategy_nutrient_multiplier = self.require_scalar(
            "strategies.baseline.nutrient_multiplier",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        # Management objectives
        self.management_objectives: List[ManagementObjective] = []
        self._initialize_default_objectives()
        self.set_management_strategy(self.management_strategy)
        
        # Coordination state
        coordination_interval_days = self.require_scalar(
            "coordination_interval_days",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        if coordination_interval_days <= 0.0:
            raise ValueError("Management coordination interval must be positive")
        self.coordination_interval = coordination_interval_days * self.seconds_per_day
        self.current_season: Optional[str] = None
        self.last_coordination_time = -self.coordination_interval

        # Priority weights for different factors (ParameterManager-driven)
        self.priority_weights = self._load_priority_weights()

        # Event tracking
        self.scheduled_events: List[ManagementEvent] = []
        self.executed_events: List[ManagementEvent] = []
        self.coordination_history: List[CoordinationResult] = []
        
        # Initialize resource quotas
        self._initialize_resource_quotas()
        
        # Register event handlers
        self._register_event_handlers()
        self.management_state: Dict[str, Any] = {}

    @staticmethod
    def _finite_runtime_float(value: Any, field_name: str) -> float:
        """Return a finite scalar for a replay-critical runtime field."""

        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be a real scalar, not bool")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{field_name} must be a real scalar") from exc
        if not math.isfinite(scalar):
            raise ValueError(f"{field_name} must be finite")
        return scalar

    @classmethod
    def _nonnegative_runtime_float(cls, value: Any, field_name: str) -> float:
        """Return a finite scalar in the non-negative physical domain."""

        scalar = cls._finite_runtime_float(value, field_name)
        if scalar < 0.0:
            raise ValueError(f"{field_name} must be non-negative")
        return scalar

    @classmethod
    def _positive_runtime_float(cls, value: Any, field_name: str) -> float:
        """Return a finite scalar in the positive physical domain."""

        scalar = cls._finite_runtime_float(value, field_name)
        if scalar <= 0.0:
            raise ValueError(f"{field_name} must be positive")
        return scalar

    @staticmethod
    def _nonnegative_runtime_int(value: Any, field_name: str) -> int:
        """Return a non-negative integer without coercing bool or fractions."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer")
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative")
        return value

    @classmethod
    def _serialize_runtime_event(cls, event: ManagementEvent) -> Dict[str, Any]:
        """Encode one event without creating a second checkpoint owner.

        References:
          - PyTorch Module.get_extra_state contract for state_dict integration.
          - RELIC simulator checkpoint transaction contract.
        Mathematical expression:
          Z_event = (t, duration, priority, executed, resource allocation).
        Parameter meanings:
          t and duration are simulation seconds, priority orders the event heap,
          and resource allocation contains quota-unit amounts.
        """

        if not isinstance(event.event_id, str) or not event.event_id:
            raise ValueError("Management event_id must be a non-empty string")
        if event.handler is not None:
            raise RuntimeError(
                "[N] Direct callable event handlers are not checkpoint-safe; "
                "register the handler by EventType instead"
            )
        requirements = {
            str(name): cls._nonnegative_runtime_float(
                amount,
                f"event {event.event_id} resource {name}",
            )
            for name, amount in event.resource_requirements.items()
        }
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "scheduled_time": cls._nonnegative_runtime_float(
                event.scheduled_time,
                f"event {event.event_id} scheduled_time",
            ),
            "priority": int(event.priority),
            "duration": cls._positive_runtime_float(
                event.duration,
                f"event {event.event_id} duration",
            ),
            "parameters": copy.deepcopy(dict(event.parameters)),
            "resource_requirements": requirements,
            "spatial_extent": copy.deepcopy(event.spatial_extent),
            "metadata": copy.deepcopy(dict(event.metadata)),
            "executed": bool(event.executed),
            "execution_result": copy.deepcopy(event.execution_result),
        }

    @classmethod
    def _deserialize_runtime_event(
        cls,
        payload: Mapping[str, Any],
    ) -> ManagementEvent:
        """Rebuild one event record while preserving the existing scheduler owner."""

        if not isinstance(payload, Mapping):
            raise TypeError("Management event runtime record must be a mapping")
        event_id = payload.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Management event_id must be a non-empty string")
        parameters = payload.get("parameters", {})
        requirements = payload.get("resource_requirements", {})
        metadata = payload.get("metadata", {})
        if not isinstance(parameters, Mapping):
            raise TypeError(f"Event {event_id} parameters must be a mapping")
        if not isinstance(requirements, Mapping):
            raise TypeError(f"Event {event_id} resource requirements must be a mapping")
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Event {event_id} metadata must be a mapping")
        return ManagementEvent(
            event_id=event_id,
            event_type=EventType(str(payload.get("event_type"))),
            scheduled_time=cls._nonnegative_runtime_float(
                payload.get("scheduled_time"),
                f"event {event_id} scheduled_time",
            ),
            priority=EventPriority(int(payload.get("priority"))),
            duration=cls._positive_runtime_float(
                payload.get("duration"),
                f"event {event_id} duration",
            ),
            parameters=copy.deepcopy(dict(parameters)),
            resource_requirements={
                str(name): cls._nonnegative_runtime_float(
                    amount,
                    f"event {event_id} resource {name}",
                )
                for name, amount in requirements.items()
            },
            spatial_extent=copy.deepcopy(payload.get("spatial_extent")),
            handler=None,
            metadata=copy.deepcopy(dict(metadata)),
            executed=bool(payload.get("executed", False)),
            execution_result=copy.deepcopy(payload.get("execution_result")),
        )

    def _runtime_event_table(self) -> Dict[str, ManagementEvent]:
        """Collect replay events once so restored containers retain object identity."""

        events: Dict[str, ManagementEvent] = {}

        def register(event: ManagementEvent) -> None:
            existing = events.get(event.event_id)
            if existing is not None and existing is not event:
                raise ValueError(
                    f"Duplicate management event_id maps to distinct objects: {event.event_id}"
                )
            events[event.event_id] = event

        for event in self.scheduled_events:
            register(event)
        for event in self.executed_events:
            register(event)
        for event in self.scheduler.events:
            register(event)
        for event in self.scheduler.execution_log:
            register(event)
        for conflict in self.scheduler.conflict_log:
            register(conflict.event1)
            register(conflict.event2)
        for result in self.coordination_history:
            for event in result.scheduled_events:
                register(event)
        return events

    def get_extra_state(self) -> Dict[str, Any]:
        """Expose replay-critical coordinator state through the existing state_dict.

        References:
          - PyTorch Module.get_extra_state contract.
          - RELIC accepted-step transaction and simulator checkpoint contracts.
        Mathematical expression:
          Z_hidden(t+dt) = F(Z_hidden(t), events, quotas); rejection restores
          Z_hidden(t) exactly before deterministic replay.
        Parameter meanings:
          Z_hidden contains the event heap, execution/conflict logs, quota usage,
          season, coordination clock, history, and the last mass residual.
        """

        event_table = self._runtime_event_table()
        residual = self._last_management_mass_residual.detach().clone()
        if not bool(torch.isfinite(residual).all().item()):
            raise ValueError("Coordinator mass residual must be finite")
        return {
            "schema_version": self._RUNTIME_STATE_SCHEMA_VERSION,
            "current_season": self.current_season,
            "last_coordination_time": self._finite_runtime_float(
                self.last_coordination_time,
                "last_coordination_time",
            ),
            "last_management_mass_residual": residual,
            "event_records": {
                event_id: self._serialize_runtime_event(event)
                for event_id, event in event_table.items()
            },
            "scheduled_event_ids": [event.event_id for event in self.scheduled_events],
            "executed_event_ids": [event.event_id for event in self.executed_events],
            "scheduler_event_ids": [event.event_id for event in self.scheduler.events],
            "scheduler_execution_log_ids": [
                event.event_id for event in self.scheduler.execution_log
            ],
            "scheduler_conflicts": [
                {
                    "event1_id": conflict.event1.event_id,
                    "event2_id": conflict.event2.event_id,
                    "conflict_type": str(conflict.conflict_type),
                    "resolution": conflict.resolution.value,
                    "resolved_at": self._finite_runtime_float(
                        conflict.resolved_at,
                        "conflict resolved_at",
                    ),
                    "outcome": str(conflict.outcome),
                }
                for conflict in self.scheduler.conflict_log
            ],
            "resource_quota_usage": {
                name: self._finite_runtime_float(
                    quota.current_used,
                    f"resource quota {name} current_used",
                )
                for name, quota in self.scheduler.resource_quotas.items()
            },
            "coordination_history": [
                {
                    "scheduled_event_ids": [
                        event.event_id for event in result.scheduled_events
                    ],
                    "conflicts_resolved": self._nonnegative_runtime_int(
                        result.conflicts_resolved,
                        "coordination conflicts_resolved",
                    ),
                    "resource_allocation": {
                        str(name): self._nonnegative_runtime_float(
                            value,
                            f"coordination resource allocation {name}",
                        )
                        for name, value in result.resource_allocation.items()
                    },
                    "objective_scores": {
                        str(name): self._finite_runtime_float(
                            value,
                            f"coordination objective score {name}",
                        )
                        for name, value in result.objective_scores.items()
                    },
                    "coordination_status": str(result.coordination_status),
                    "warnings": [str(item) for item in result.warnings],
                }
                for result in self.coordination_history
            ],
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        """Restore replay-critical coordinator state from state_dict payload."""

        if not isinstance(state, Mapping):
            raise TypeError("Coordinator extra state must be a mapping")
        if state.get("schema_version") != self._RUNTIME_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported coordinator runtime state schema_version")
        records = state.get("event_records")
        if not isinstance(records, Mapping):
            raise TypeError("Coordinator event_records must be a mapping")
        events: Dict[str, ManagementEvent] = {}
        for event_id, payload in records.items():
            if not isinstance(event_id, str):
                raise TypeError("Coordinator event record keys must be strings")
            event = self._deserialize_runtime_event(payload)
            if event.event_id != event_id:
                raise ValueError("Coordinator event record key does not match event_id")
            events[event_id] = event

        def resolve_event_ids(field_name: str) -> List[ManagementEvent]:
            event_ids = state.get(field_name)
            if not isinstance(event_ids, list):
                raise TypeError(f"Coordinator {field_name} must be a list")
            resolved: List[ManagementEvent] = []
            for event_id in event_ids:
                if not isinstance(event_id, str) or event_id not in events:
                    raise ValueError(
                        f"Coordinator {field_name} references unknown event_id: {event_id}"
                    )
                resolved.append(events[event_id])
            return resolved

        season = state.get("current_season")
        allowed_seasons = {None, "spring", "summer", "autumn", "winter"}
        if season not in allowed_seasons:
            raise ValueError(
                "Coordinator current_season must be spring, summer, autumn, winter, or None"
            )
        residual = state.get("last_management_mass_residual")
        if not isinstance(residual, torch.Tensor):
            raise TypeError("Coordinator mass residual must be a torch.Tensor")
        if residual.numel() == 0 or not torch.is_floating_point(residual):
            raise ValueError("Coordinator mass residual must be a non-empty real tensor")
        if not bool(torch.isfinite(residual).all().item()):
            raise ValueError("Coordinator mass residual must be finite")

        quota_usage = state.get("resource_quota_usage")
        if not isinstance(quota_usage, Mapping):
            raise TypeError("Coordinator resource_quota_usage must be a mapping")
        expected_quotas = set(self.scheduler.resource_quotas)
        if set(quota_usage) != expected_quotas:
            raise ValueError("Coordinator resource quota names do not match runtime configuration")
        restored_quota_usage: Dict[str, float] = {}
        for name, quota in self.scheduler.resource_quotas.items():
            current_used = self._nonnegative_runtime_float(
                quota_usage[name],
                f"resource quota {name} current_used",
            )
            if current_used > quota.total_available:
                raise ValueError(
                    f"Resource quota {name} current_used exceeds total_available"
                )
            restored_quota_usage[name] = current_used

        history_payload = state.get("coordination_history")
        if not isinstance(history_payload, list):
            raise TypeError("Coordinator coordination_history must be a list")
        restored_history: List[CoordinationResult] = []
        for item in history_payload:
            if not isinstance(item, Mapping):
                raise TypeError("Coordinator history records must be mappings")
            history_ids = item.get("scheduled_event_ids")
            if not isinstance(history_ids, list):
                raise TypeError("Coordinator history event IDs must be a list")
            history_events: List[ManagementEvent] = []
            for event_id in history_ids:
                if not isinstance(event_id, str) or event_id not in events:
                    raise ValueError(
                        f"Coordinator history references unknown event_id: {event_id}"
                    )
                history_events.append(events[event_id])
            allocations = item.get("resource_allocation", {})
            scores = item.get("objective_scores", {})
            warnings_payload = item.get("warnings", [])
            if not isinstance(allocations, Mapping) or not isinstance(scores, Mapping):
                raise TypeError("Coordinator history allocations and scores must be mappings")
            if not isinstance(warnings_payload, list):
                raise TypeError("Coordinator history warnings must be a list")
            conflicts_resolved = self._nonnegative_runtime_int(
                item.get("conflicts_resolved", 0),
                "coordination conflicts_resolved",
            )
            restored_allocations = {
                str(name): self._nonnegative_runtime_float(
                    value,
                    f"coordination resource allocation {name}",
                )
                for name, value in allocations.items()
            }
            restored_scores = {
                str(name): self._finite_runtime_float(
                    value,
                    f"coordination objective score {name}",
                )
                for name, value in scores.items()
            }
            restored_history.append(
                CoordinationResult(
                    scheduled_events=history_events,
                    conflicts_resolved=conflicts_resolved,
                    resource_allocation=restored_allocations,
                    objective_scores=restored_scores,
                    coordination_status=str(item.get("coordination_status", "")),
                    warnings=[str(value) for value in warnings_payload],
                )
            )

        conflicts_payload = state.get("scheduler_conflicts")
        if not isinstance(conflicts_payload, list):
            raise TypeError("Coordinator scheduler_conflicts must be a list")
        restored_conflicts: List[EventConflict] = []
        for item in conflicts_payload:
            if not isinstance(item, Mapping):
                raise TypeError("Coordinator conflict records must be mappings")
            event1_id = item.get("event1_id")
            event2_id = item.get("event2_id")
            if event1_id not in events or event2_id not in events:
                raise ValueError("Coordinator conflict references an unknown event_id")
            restored_conflicts.append(
                EventConflict(
                    event1=events[str(event1_id)],
                    event2=events[str(event2_id)],
                    conflict_type=str(item.get("conflict_type", "")),
                    resolution=ConflictResolution(str(item.get("resolution"))),
                    resolved_at=self._finite_runtime_float(
                        item.get("resolved_at"),
                        "conflict resolved_at",
                    ),
                    outcome=str(item.get("outcome", "")),
                )
            )

        restored_scheduler_events = resolve_event_ids("scheduler_event_ids")
        for child_index in range(1, len(restored_scheduler_events)):
            parent_index = (child_index - 1) // 2
            if restored_scheduler_events[child_index] < restored_scheduler_events[parent_index]:
                raise ValueError("Coordinator scheduler_event_ids do not encode a valid heap")
        restored_scheduled_events = resolve_event_ids("scheduled_event_ids")
        restored_executed_events = resolve_event_ids("executed_event_ids")
        restored_execution_log = resolve_event_ids("scheduler_execution_log_ids")

        self.current_season = season
        self.last_coordination_time = self._finite_runtime_float(
            state.get("last_coordination_time"),
            "last_coordination_time",
        )
        self._last_management_mass_residual = residual.detach().clone().to(self.device)
        self.scheduled_events = restored_scheduled_events
        self.executed_events = restored_executed_events
        self.scheduler.events = restored_scheduler_events
        self.scheduler.execution_log = restored_execution_log
        self.scheduler.conflict_log = restored_conflicts
        self.coordination_history = restored_history
        for name, quota in self.scheduler.resource_quotas.items():
            quota.current_used = restored_quota_usage[name]

    def _structural_n_dofs(self, n_dofs: Optional[int] = None) -> int:
        """Resolve structural audit size for management objectives."""

        value = 5 if n_dofs is None else int(n_dofs)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare cochain spaces for the management coordination port map.

        References:
          - Robertson and Swinton (2005), agricultural productivity and integrity.
          - Matson et al. (1997), agricultural intensification interactions.
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        Mathematical expression:
          u_m = Pi_h(x, theta), y_m = G_m u_m, and allocation_r <= quota_r.
        Parameter meanings:
          u_m is the vector of dimensionless management controls, Pi_h is the
          priority projection from state and parameters, and y_m contains
          declared management flux or modifier outputs.
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="anthroposphere_management_coordinator",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="1",
                extensivity="intensive",
                physical_domain="anthroposphere",
                semantic_name="management_priority_state",
                orientation_convention="positive_management_need",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_management_coordinator",
                degree=1,
                primality="primal",
                n_dofs=n_dofs,
                units="1",
                extensivity="intensive",
                physical_domain="anthroposphere",
                semantic_name="management_control_projection",
                orientation_convention="positive_management_action",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_management_coordinator",
                degree=1,
                primality="dual",
                n_dofs=n_dofs,
                units="1",
                extensivity="integral",
                physical_domain="anthroposphere",
                semantic_name="management_resource_allocation",
                orientation_convention="positive_resource_use",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager namespaces used by runtime and structure."""

        return (
            "anthroposphere.management.coordinator.*",
            "anthroposphere.management.coordinator.priority_weights.*",
            "anthroposphere.management.coordinator.resource_quotas.*",
            "anthroposphere.management.*",
            "validation.conservation.conservation_tolerance",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive coordination priority Hodge contract.

        References:
          - Robertson and Swinton (2005), multi-objective agricultural management.
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        Mathematical expression:
          H_m = diag(w_i / sum_j w_j), w_i >= 0.
        Parameter meanings:
          w_i are ParameterManager-backed priority weights for management
          objectives and H_m is the positive priority metric [1].
        """

        n_dofs = self._structural_n_dofs(n_dofs)
        priority_weights = getattr(self, "priority_weights", {})
        expected_names = (
            "nutrients",
            "water",
            "soil_health",
            "productivity",
            "environmental",
        )
        if n_dofs != len(expected_names) or set(priority_weights) != set(expected_names):
            raise ValueError(
                "Management coordination priority weights must exactly match "
                f"{expected_names} for {len(expected_names)} structural degrees of freedom"
            )
        weight_values = [priority_weights[name] for name in expected_names]
        weights = torch.as_tensor(
            weight_values[:n_dofs],
            device=getattr(self, "device", torch.device("cpu")),
            dtype=torch.float32,
        )
        if torch.any(weights < 0):
            raise ValueError("Management coordination weights must be non-negative.")
        total = torch.sum(weights)
        if total <= 0:
            raise ValueError("Management coordination weights require positive support.")
        matrix = torch.diag(weights / total)
        domain = CochainSpaceSpec(
            complex_id="anthroposphere_management_coordinator",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="1",
            extensivity="intensive",
            physical_domain="anthroposphere",
            semantic_name="management_priority_state",
            orientation_convention="positive_management_need",
            boundary_role="interior",
        )
        codomain = CochainSpaceSpec(
            complex_id="anthroposphere_management_coordinator",
            degree=0,
            primality="dual",
            n_dofs=n_dofs,
            units="1",
            extensivity="intensive",
            physical_domain="anthroposphere",
            semantic_name="management_control_projection",
            orientation_convention="positive_management_action",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_priority_hodge",
            domain=domain,
            codomain=codomain,
            matrix=matrix,
            parameter_keys=("anthroposphere.management.coordinator.priority_weights.*",),
            metric_keys=("validation.conservation.conservation_tolerance",),
            material_law="management_priority_projection",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build Hodge terms for IPHS structural assembly."""

        _ = complex_bundle, resolved_parameters
        return (self.build_hodge_contract(strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no independent stored energy for coordination decisions."""

        _ = state
        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the non-negative priority allocation audit term."""

        _ = state
        return {
            "management_coordination_entropy": "sigma = z.T H_m z >= 0",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the declared management port projection metadata."""

        _ = state, complex_bundle
        return {
            "state_vars": (
                "nutrients",
                "water",
                "soil_health",
                "productivity",
                "environmental",
            ),
            "chain_map": "u_m = Pi_h(x, theta)",
            "quota_constrained": True,
            "ports": self.ports(),
        }

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive priority metric for audits."""

        _ = state, complex_bundle
        hodge = self.build_hodge_contract(strict=True)
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"management_priority_state[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the management coordination control port."""

        effort_space = CochainSpaceSpec(
            complex_id="anthroposphere_management_coordinator",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="1",
            extensivity="intensive",
            physical_domain="anthroposphere",
            semantic_name="management_priority_effort",
            orientation_convention="positive_management_need",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="anthroposphere_management_coordinator",
            degree=1,
            primality="dual",
            n_dofs=1,
            units="1",
            extensivity="integral",
            physical_domain="anthroposphere",
            semantic_name="management_resource_allocation",
            orientation_convention="positive_resource_use",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="management_coordination_control_port",
                effort_var="management_priority_effort",
                flow_var="resource_allocation",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="1",
                positive_power_convention="into_system",
                port_kind="management",
            ),
        )

    def compute(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
    ) -> Mapping[str, Any]:
        """Run the canonical calculation through the existing Functor lifecycle."""

        return self.forward(dict(state), time, dt)

    def validate_state(self, state: Dict[str, torch.Tensor]) -> List[str]:
        """Return canonical input-contract issues without state substitution."""
        try:
            self._validate_canonical_inputs(state)
        except (KeyError, TypeError, ValueError) as exc:
            return [str(exc)]
        return []

    def _validate_canonical_inputs(
        self,
        state: Dict[str, torch.Tensor],
        calendar_type: Optional[str] = None,
    ) -> None:
        """Validate dynamic decision inputs before any scheduling calculation."""
        if not isinstance(state, dict):
            raise TypeError("ManagementCoordinator state must be a dictionary")

        required = {
            name: self._require_state_tensor(state, name)
            for name in self.REQUIRED_STATE_INPUTS
        }
        reference_shape = required["soil_moisture"].shape
        for name, tensor in required.items():
            if tensor.shape != reference_shape:
                raise ValueError(
                    f"ManagementCoordinator input '{name}' has shape {tensor.shape}; "
                    f"expected {reference_shape}"
                )

        bounded_unit_interval = (
            "soil_moisture",
            "crop_maturity",
            "field_capacity",
            "wilting_point",
        )
        for name in bounded_unit_interval:
            tensor = required[name]
            if torch.any((tensor < 0.0) | (tensor > 1.0)):
                raise ValueError(
                    f"ManagementCoordinator input '{name}' must be within [0, 1]"
                )
        for name in ("soil_nitrogen", "soil_organic_carbon", "aboveground_biomass"):
            if torch.any(required[name] < 0.0):
                raise ValueError(
                    f"ManagementCoordinator input '{name}' must be non-negative"
                )
        if torch.any(required["air_temperature"] <= 0.0):
            raise ValueError(
                "ManagementCoordinator input 'air_temperature' must be positive in K"
            )

        for name in ("irrigation_flux", "harvest_flux", "grazing_flux", "co2_emission"):
            if name not in state:
                continue
            tensor = self._require_state_tensor(state, name)
            if tensor.shape != reference_shape:
                raise ValueError(
                    f"ManagementCoordinator input '{name}' has shape {tensor.shape}; "
                    f"expected {reference_shape}"
                )
            if torch.any(tensor < 0.0):
                raise ValueError(
                    f"ManagementCoordinator input '{name}' must be non-negative"
                )

        if "day_of_year" in state:
            day = self._require_state_tensor(state, "day_of_year")
            if self._configured_calendar_type is None and calendar_type is None:
                self._validate_unbound_simulator_day(day)
                return
            resolved_calendar = CalendarHandler.resolve_simulator_calendar(
                configured_calendar=self._configured_calendar_type,
                runtime_calendar=calendar_type,
                owner_name=self.__class__.__name__,
            )
            days_per_year = CalendarHandler.days_per_simulator_year(
                resolved_calendar
            )
            try:
                CalendarHandler.validate_simulator_day(day, resolved_calendar)
            except ValueError as exc:
                raise ValueError(
                    "ManagementCoordinator input 'day_of_year' must be a finite "
                    "integer numeric scalar within "
                    f"[1, {days_per_year}]"
                ) from exc

    @staticmethod
    def _validate_unbound_simulator_day(day: torch.Tensor) -> None:
        """Validate the calendar-independent domain ``d in N, d >= 1``.

        ``FunctorBase.__call__`` validates state before forwarding simulator
        integration keywords.  At that lifecycle boundary the calendar year
        length ``Y(C)`` is intentionally unavailable, so this preflight checks
        only the calendar-independent natural-day envelope.  ``forward`` then
        resolves the simulator-owned calendar and enforces ``1 <= d <= Y(C)``
        through ``CalendarHandler`` before any scheduling calculation.
        """

        if day.numel() != 1 or day.dtype == torch.bool or torch.is_complex(day):
            raise ValueError(
                "ManagementCoordinator input 'day_of_year' must be a finite "
                "integer numeric scalar greater than or equal to 1"
            )
        scalar_day = day.reshape(())
        if torch.any(scalar_day < 1) or torch.any(
            scalar_day != torch.floor(scalar_day)
        ):
            raise ValueError(
                "ManagementCoordinator input 'day_of_year' must be a finite "
                "integer numeric scalar greater than or equal to 1"
            )

    @staticmethod
    def _require_state_tensor(
        state: Dict[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        """Return one required finite state tensor without aliases or defaults."""
        if name not in state:
            raise KeyError(f"ManagementCoordinator requires state input '{name}'")
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"ManagementCoordinator input '{name}' must be a torch.Tensor"
            )
        if value.numel() == 0 or not torch.all(torch.isfinite(value)):
            raise ValueError(
                f"ManagementCoordinator input '{name}' must be non-empty and finite"
            )
        return value

    def _infer_device(self, state: Dict[str, torch.Tensor]) -> torch.device:
        """Return the device of the state tensors the coordination reads.

        The coordination runs on the device of its declared state inputs; a state
        without any tensor has no inputs to place, and raises.
        """
        for value in state.values():
            if isinstance(value, torch.Tensor):
                return value.device
        raise ValueError(
            "ManagementCoordinator state holds no tensor to place the computation on"
        )

    @staticmethod
    def _scalar_value(value: Any) -> float:
        """Convert one finite scalar-like value without a silent default."""
        if isinstance(value, torch.Tensor):
            detached = value.detach()
            if detached.numel() == 0:
                raise ValueError("Cannot convert an empty tensor to a scalar")
            scalar = float(detached.reshape(-1)[0].tolist())
        else:
            scalar = float(value)
        if not torch.isfinite(torch.as_tensor(scalar)):
            raise ValueError("Scalar value must be finite")
        return scalar

    def _get_coord_weight(self, name: str) -> float:
        """Fetch coordinator priority weight from ParameterManager."""
        return self.require_scalar(
            name,
            namespace="anthroposphere.management.coordinator.priority_weights",
            source_order=("manager",),
        )

    def _load_priority_weights(self) -> Dict[str, float]:
        """
        Load and normalize coordinator priority weights.

        References:
            - Multi-objective scheduling; weights sum to 1 for scoring normalization.
            - Parameters stored under anthroposphere.management.coordinator.priority_weights.*
        """
        weights = {
            "nutrients": self._get_coord_weight("nutrients"),
            "water": self._get_coord_weight("water"),
            "soil_health": self._get_coord_weight("soil_health"),
            "productivity": self._get_coord_weight("productivity"),
            "environmental": self._get_coord_weight("environmental"),
        }
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("Management coordination priority weights must sum to a positive value")
        return {key: value / total for key, value in weights.items()}

    def calculate_priority_weights(
        self,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate normalized priority weights for management goals.

        References:
            - Lal (2004) soil carbon-water-nutrient interactions.
            - Allen et al. (1998) irrigation scheduling (ET-based).
        Mathematical Expression:
            w_i = w_i0 * (1 + S_i) / sum_j w_j0 * (1 + S_j)
            where S_i is stress factor derived from state variables.
        Parameters:
            state : Dict[str, Tensor]
                Declared state inputs: soil_nitrogen, soil_moisture,
                soil_organic_carbon, crop_maturity and air_temperature.
        Returns:
            Dict[str, Tensor]: Priority weights summing to ~1 on the current device.
        """
        device = self._infer_device(state)
        target_dtype = torch.get_default_dtype()

        base_weights = {
            "nutrients": torch.as_tensor(
                self.require_scalar(
                    "priority_weights.nutrients",
                    namespace="anthroposphere.management.coordinator",
                    source_order=("manager",),
                ),
                device=device,
                dtype=target_dtype,
            ),
            "water": torch.as_tensor(
                self.require_scalar(
                    "priority_weights.water",
                    namespace="anthroposphere.management.coordinator",
                    source_order=("manager",),
                ),
                device=device,
                dtype=target_dtype,
            ),
            "soil_health": torch.as_tensor(
                self.require_scalar(
                    "priority_weights.soil_health",
                    namespace="anthroposphere.management.coordinator",
                    source_order=("manager",),
                ),
                device=device,
                dtype=target_dtype,
            ),
            "productivity": torch.as_tensor(
                self.require_scalar(
                    "priority_weights.productivity",
                    namespace="anthroposphere.management.coordinator",
                    source_order=("manager",),
                ),
                device=device,
                dtype=target_dtype,
            ),
            "environmental": torch.as_tensor(
                self.require_scalar(
                    "priority_weights.environmental",
                    namespace="anthroposphere.management.coordinator",
                    source_order=("manager",),
                ),
                device=device,
                dtype=target_dtype,
            ),
        }

        soil_n = self._require_state_tensor(state, "soil_nitrogen").to(
            device=device, dtype=target_dtype
        )
        soil_m = self._require_state_tensor(state, "soil_moisture").to(
            device=device, dtype=target_dtype
        )
        soil_c = self._require_state_tensor(state, "soil_organic_carbon").to(
            device=device, dtype=target_dtype
        )
        maturity = self._require_state_tensor(state, "crop_maturity").to(
            device=device, dtype=target_dtype
        )
        air_temp_tensor = self._require_state_tensor(
            state, "air_temperature"
        ).to(device=device, dtype=target_dtype)

        def _positive_reference(name: str) -> torch.Tensor:
            value = torch.as_tensor(
                self.get_parameter_value(
                    f"reference.{name}",
                    namespace="anthroposphere.management.coordinator",
                    source_order=("manager",),
                    required=True,
                ),
                device=device,
                dtype=target_dtype,
            )
            if value.numel() != 1 or not torch.all(torch.isfinite(value)) or torch.any(value <= 0.0):
                raise ValueError(
                    f"ManagementCoordinator reference parameter '{name}' must be one finite positive scalar"
                )
            return value.reshape(())

        n_ref = _positive_reference("soil_nitrogen_optimal")
        m_ref = _positive_reference("soil_moisture_optimal")
        c_ref = _positive_reference("soil_carbon_reference")
        p_ref = _positive_reference("productivity_reference")
        t_ref = _positive_reference("environment_temperature_optimal")
        benefit_max = self.require_scalar(
            "score_limits.benefit_max",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )

        nutrient_stress = torch.clamp(1.0 - soil_n / n_ref, min=0.0, max=1.0)
        water_stress = torch.clamp(1.0 - soil_m / m_ref, min=0.0, max=1.0)
        soil_health_score = torch.clamp(soil_c / c_ref, min=0.0, max=benefit_max)
        productivity_score = torch.clamp(maturity / p_ref, min=0.0, max=benefit_max)
        temp_diff = torch.abs(air_temp_tensor - t_ref)
        environmental_stress = torch.clamp(temp_diff / t_ref, min=0.0, max=1.0)

        weighted = {
            "nutrients": base_weights["nutrients"] * (1.0 + nutrient_stress.mean()),
            "water": base_weights["water"] * (1.0 + water_stress.mean()),
            "soil_health": base_weights["soil_health"] * (1.0 + soil_health_score.mean()),
            "productivity": base_weights["productivity"] * (1.0 + productivity_score.mean()),
            "environmental": base_weights["environmental"] * (1.0 + environmental_stress.mean()),
        }

        total = sum(weighted.values())
        if not torch.all(torch.isfinite(total)) or torch.any(total <= 0.0):
            raise ValueError("ManagementCoordinator weighted priorities must sum to a positive finite value")
        priorities = {k: torch.clamp(v / total, min=0.0) for k, v in weighted.items()}
        return priorities

    def calculate_interaction_effects(
        self,
        state: Dict[str, torch.Tensor],
        priority_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate interaction multipliers between management practices.

        References:
            - Matson et al. (1997) interactions of fertilization and tillage.
        Mathematical Expression:
            I_ab = k_int * (priority_a + priority_b)
        Parameter meanings:
            state : Dict[str, Tensor]
                State variables used to compute priority weights.
            priority_weights : Optional[Dict[str, Tensor]]
                Precomputed priority weights (dimensionless).
        """
        device = self._infer_device(state)
        target_dtype = torch.get_default_dtype()
        priorities = priority_weights or self.calculate_priority_weights(state)
        interaction_factor = torch.as_tensor(
            self.get_parameter_value(
                "interaction_factor",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",), required=True),
            device=device,
            dtype=target_dtype,
        )
        interaction_min = self.require_scalar(
            "interaction_multiplier_min",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        interaction_max = self.require_scalar(
            "interaction_multiplier_max",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        if interaction_min > interaction_max:
            raise ValueError("Management interaction multiplier bounds are reversed")

        def _pair(a: str, b: str) -> torch.Tensor:
            return torch.clamp(
                interaction_factor * (priorities[a] + priorities[b]),
                min=interaction_min,
                max=interaction_max,
            )

        return {
            "fertilizer_irrigation": _pair("nutrients", "water"),
            "tillage_fertilizer": _pair("soil_health", "nutrients"),
            "irrigation_tillage": _pair("water", "soil_health"),
            "fertilizer_tillage": _pair("nutrients", "soil_health"),
        }

    def optimize_management_practices(
        self,
        state: Dict[str, torch.Tensor],
        priority_weights: Optional[Dict[str, torch.Tensor]] = None,
        interaction_effects: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize management decisions based on priorities and interactions.

        References:
            - Doorenbos and Pruitt (1977), FAO Irrigation and Drainage Paper 24.
            - Allen et al. (1998), FAO Irrigation and Drainage Paper 56, Eq. 85.
            - Integrated management optimization literature (policy heuristic).
        Mathematical Expression:
            decision_i = efficiency * priority_i * interaction_i * base_rate_i
                for the fertilizer product flux and the tillage intensity;
            irrigation_demand_flux = max(E - P, 0), the net irrigation requirement.
        Parameter meanings:
            state : Dict[str, Tensor]
                State variables used for spatial templates, maturity scaling and
                priority scoring, and the evapotranspiration and precipitation
                fluxes E and P.
            priority_weights : Optional[Dict[str, Tensor]]
                Precomputed priority weights (dimensionless).
            interaction_effects : Optional[Dict[str, Tensor]]
                Interaction multipliers between management practices.
        """
        device = self._infer_device(state)
        target_dtype = torch.get_default_dtype()
        priorities = priority_weights or self.calculate_priority_weights(state)
        interactions = interaction_effects or self.calculate_interaction_effects(
            state, priority_weights=priorities
        )
        efficiency = torch.as_tensor(
            self.get_parameter_value(
                "coordination_efficiency",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",), required=True),
            device=device,
            dtype=target_dtype,
        )
        decision_threshold = torch.as_tensor(
            self.get_parameter_value(
                "decision_threshold",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",), required=True),
            device=device,
            dtype=target_dtype,
        )

        def _template_tensor() -> torch.Tensor:
            # Prefer a spatial grid template ([B, H, W]) for control outputs.
            soil_moisture = self._require_state_tensor(state, "soil_moisture")
            return torch.zeros_like(
                soil_moisture,
                device=device,
                dtype=target_dtype,
            )

        template = _template_tensor().to(device=device, dtype=target_dtype)

        spring_product_mass = torch.as_tensor(
            self.get_parameter_value(
                "fertilization.spring_urea_product_mass",
                namespace="anthroposphere.management",
                source_order=("manager",), required=True),
            device=device,
            dtype=target_dtype,
        )
        # References: FAO (2000), Fertilizers and Their Use, 4th edition.
        # Mathematical expression: F_product = M_product / tau_application.
        # Parameters: M_product [kg product m-2], tau_application [s day-1],
        # and F_product [kg product m-2 s-1]. This is an open management port.
        fert_base = spring_product_mass / torch.as_tensor(
            self.seconds_per_day,
            device=device,
            dtype=target_dtype,
        )
        tillage_base = torch.as_tensor(
            self.get_parameter_value(
                "tillage.low_intensity_factor",
                namespace="anthroposphere.management",
                source_order=("manager",), required=True),
            device=device,
            dtype=target_dtype,
        )

        maturity_ref = torch.as_tensor(
            self.get_parameter_value(
                "reference.productivity_reference",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",), required=True),
            device=device,
            dtype=target_dtype,
        )
        maturity_state = self._require_state_tensor(state, "crop_maturity")
        if maturity_ref.numel() != 1 or not torch.all(torch.isfinite(maturity_ref)) or torch.any(maturity_ref <= 0.0):
            raise ValueError(
                "ManagementCoordinator reference parameter 'productivity_reference' "
                "must be one finite positive scalar"
            )
        maturity_min = self.require_scalar(
            "maturity_factor_min",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        maturity_max = self.require_scalar(
            "maturity_factor_max",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        if maturity_min > maturity_max:
            raise ValueError("Management maturity factor bounds are reversed")
        maturity_factor = torch.clamp(
            maturity_state.to(device=device, dtype=target_dtype)
            / maturity_ref.reshape(()),
            min=maturity_min,
            max=maturity_max,
        ).mean()

        fertilizer_value = torch.clamp(
            efficiency
            * priorities["nutrients"]
            * interactions["fertilizer_irrigation"]
            * fert_base
            * maturity_factor
            * torch.as_tensor(self.strategy_nutrient_multiplier, device=device, dtype=target_dtype),
            min=0.0,
        )
        # References: Doorenbos and Pruitt (1977), FAO Irrigation and Drainage Paper 24
        #   (net irrigation requirement); Allen et al. (1998), FAO Irrigation and
        #   Drainage Paper 56, Eq. 85 (root-zone depletion by crop evapotranspiration).
        # Mathematical expression: F_net = max(E - P, 0).
        # Parameter meanings: E is the surface evapotranspiration flux, the actual crop
        #   evapotranspiration that depletes the root zone; P is the precipitation
        #   flux, taken as fully effective (P_e = P) as in the IrrigationManagement
        #   root-zone balance; F_net is the net irrigation requirement rate. P_e = P
        #   is a simplification: FAO effective-precipitation methods (USDA-SCS;
        #   FAO/AGLW) give P_e <= P, so F_net underestimates the FAO requirement.
        # Units: E, P, F_net [kg m-2 s-1].
        # Domain/constraints: E and P have the soil_moisture cell shape and P >= 0,
        #   both checked here; E is signed. The gross application F_net / eta_a needs
        #   the application efficiency of the irrigation system and is computed only
        #   by IrrigationManagement; no coordination score scales the requirement.
        # Sign convention: E is positive upward (evaporation) and negative for dew; P
        #   is positive downward; F_net >= 0 is the water the root zone needs.
        evapotranspiration = self._require_state_tensor(
            state, "evapotranspiration_flux"
        ).to(device=device, dtype=target_dtype)
        precipitation = self._require_state_tensor(
            state, "precipitation_flux"
        ).to(device=device, dtype=target_dtype)
        for name, tensor in (
            ("evapotranspiration_flux", evapotranspiration),
            ("precipitation_flux", precipitation),
        ):
            if tensor.shape != template.shape:
                raise ValueError(
                    f"ManagementCoordinator input '{name}' has shape "
                    f"{tuple(tensor.shape)}; expected {tuple(template.shape)}"
                )
        if bool(torch.any(precipitation < 0.0).item()):
            raise ValueError(
                "ManagementCoordinator input 'precipitation_flux' must be non-negative"
            )
        irrigation_value = torch.clamp(evapotranspiration - precipitation, min=0.0)
        clamp_min = torch.as_tensor(
            self.get_parameter_value(
                "clamp_min",
                namespace="anthroposphere.management",
                source_order=("manager",),
                required=True,
            ),
            device=device,
            dtype=target_dtype,
        )
        clamp_max = torch.as_tensor(
            self.get_parameter_value(
                "clamp_max",
                namespace="anthroposphere.management",
                source_order=("manager",),
                required=True,
            ),
            device=device,
            dtype=target_dtype,
        )
        tillage_value = torch.clamp(
            efficiency
            * priorities["soil_health"]
            * interactions["tillage_fertilizer"]
            * tillage_base,
            min=clamp_min,
            max=clamp_max,
        )

        decisions = {
            "fertilizer_product_flux": torch.ones_like(template) * fertilizer_value,
            "irrigation_demand_flux": irrigation_value,
            "tillage_intensity": torch.ones_like(template) * tillage_value,
        }

        constraint_satisfaction = torch.clamp(efficiency - decision_threshold, min=0.0, max=1.0)
        return {
            "optimized_decisions": decisions,
            "constraint_satisfaction": constraint_satisfaction,
        }

    def compute_coordination_decisions(
        self,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Compute coordination decisions for the current declared state.

        Returns:
            Dict[str, Any]: management_decisions, priority_weights, interaction_effects
        """
        priorities = self.calculate_priority_weights(state)
        interactions = self.calculate_interaction_effects(state, priority_weights=priorities)
        optimization = self.optimize_management_practices(
            state,
            priority_weights=priorities,
            interaction_effects=interactions,
        )
        return {
            "management_decisions": optimization["optimized_decisions"],
            "priority_weights": priorities,
            "interaction_effects": interactions,
            "constraint_satisfaction": optimization["constraint_satisfaction"],
        }
    
    def _initialize_default_objectives(self) -> None:
        """Initialize default management objectives"""
        def _objective_value(objective: str, field: str) -> float:
            return self.require_scalar(
                f"objectives.{objective}.{field}",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )

        self.management_objectives = [
            ManagementObjective(
                name="water_use_efficiency",
                weight=_objective_value("water_use_efficiency", "weight"),
            ),
            ManagementObjective(
                name="nutrient_use_efficiency",
                weight=_objective_value("nutrient_use_efficiency", "weight"),
            ),
            ManagementObjective(
                name="soil_carbon_sequestration",
                weight=_objective_value("soil_carbon_sequestration", "weight"),
            ),
            ManagementObjective(
                name="economic_return",
                weight=_objective_value("economic_return", "weight"),
            ),
            ManagementObjective(
                name="biodiversity_index",
                weight=_objective_value("biodiversity_index", "weight"),
            ),
        ]
    
    def _initialize_resource_quotas(self) -> None:
        """Initialize resource quotas for management activities"""
        manager = self.get_parameter_manager()
        if manager is None:
            raise RuntimeError(
                "ManagementCoordinator requires an attached ParameterManager for quota metadata."
            )

        def _quota_value(name: str, namespace: str) -> float:
            return self.require_scalar(
                name,
                namespace=namespace,
                source_order=("manager",),
            )

        def _quota_unit(full_key: str) -> str:
            spec = manager.get_parameter(full_key)
            if spec is None or not getattr(spec, "units", None):
                raise KeyError(f"Missing units for parameter '{full_key}'")
            return str(spec.units)

        water_key = "anthroposphere.management.coordinator.resource_quotas.water"
        quotas = [
            ResourceQuota(
                "water",
                _quota_value("resource_quotas.water", "anthroposphere.management.coordinator"),
                unit=_quota_unit(water_key),
                reset_period="yearly",
            ),
            ResourceQuota(
                "nitrogen",
                _quota_value("resource_quotas.nitrogen", "anthroposphere.management.coordinator"),
                unit=_quota_unit("anthroposphere.management.coordinator.resource_quotas.nitrogen"),
                reset_period="yearly",
            ),
            ResourceQuota(
                "phosphorus",
                _quota_value("resource_quotas.phosphorus", "anthroposphere.management.coordinator"),
                unit=_quota_unit("anthroposphere.management.coordinator.resource_quotas.phosphorus"),
                reset_period="yearly",
            ),
            ResourceQuota(
                "labor",
                _quota_value("resource_quotas.labor", "anthroposphere.management.coordinator"),
                unit=_quota_unit("anthroposphere.management.coordinator.resource_quotas.labor"),
                reset_period="monthly",
            ),
            ResourceQuota(
                "machinery",
                _quota_value("resource_quotas.machinery", "anthroposphere.management.coordinator"),
                unit=_quota_unit("anthroposphere.management.coordinator.resource_quotas.machinery"),
                reset_period="monthly",
            ),
            ResourceQuota(
                "fuel",
                _quota_value("resource_quotas.fuel", "anthroposphere.management.coordinator"),
                unit=_quota_unit("anthroposphere.management.coordinator.resource_quotas.fuel"),
                reset_period="yearly",
            ),
        ]

        for quota in quotas:
            self.scheduler.register_resource_quota(quota)

    def _register_event_handlers(self) -> None:
        """Register event handlers for different management activities"""
        self.scheduler.register_event_handler(EventType.IRRIGATION, self._handle_irrigation_event)
        self.scheduler.register_event_handler(EventType.TILLAGE, self._handle_tillage_event)
        self.scheduler.register_event_handler(EventType.HARVEST, self._handle_harvest_event)
        self.scheduler.register_event_handler(EventType.GRAZING, self._handle_grazing_event)
        self.scheduler.register_event_handler(EventType.FIRE, self._handle_fire_event)
    
    def get_dependencies(self) -> Dict[str, List[str]]:
        """
        Return declared dependencies for scheduling and integration.

        Dependencies are Functor identifiers, never state-variable names. The
        canonical management processes are lagged (no same-step ordering edge)
        because this coordinator is meant to observe their last accepted fluxes
        while issuing controls for the next step. The lag holds only when the
        coordinator runs before them in its group order; otherwise it reads
        the values they produced earlier in the same group call.

        Instance-declared dependencies (graph config, set_dependencies,
        add_dependency) are merged through FunctorBase: a new name becomes a
        required edge, a name already declared lagged stays lagged, and the
        canonical lagged declarations are never replaced.
        """
        return self._merge_declared_dependencies(
            {
                "required": list(self.REQUIRED_DEPENDENCIES),
                "optional": list(self.OPTIONAL_DEPENDENCIES),
                "lagged": list(self.LAGGED_DEPENDENCIES),
            }
        )
    
    def set_management_strategy(self, strategy: ManagementStrategy) -> None:
        """Set overall management strategy"""
        if not isinstance(strategy, ManagementStrategy):
            raise TypeError("Management strategy must be a ManagementStrategy value")
        self.management_strategy = strategy
        self._reset_strategy_baseline()
        strategy_handlers: Dict[ManagementStrategy, Callable[[], None]] = {
            ManagementStrategy.CONSERVATION: self._apply_conservation_strategy,
            ManagementStrategy.ORGANIC: self._apply_organic_strategy,
            ManagementStrategy.PRECISION: self._apply_precision_strategy,
            ManagementStrategy.PRODUCTIVITY_FOCUSED: self._apply_productivity_strategy,
            ManagementStrategy.SUSTAINABILITY_FOCUSED: self._apply_conservation_strategy,
            ManagementStrategy.CONSERVATION_FOCUSED: self._apply_conservation_strategy,
            ManagementStrategy.SUSTAINABLE: self._apply_sustainable_strategy,
            ManagementStrategy.BALANCED: self._apply_balanced_strategy,
        }
        handler = strategy_handlers.get(strategy)
        if handler is not None:
            handler()

    def _reset_strategy_baseline(self) -> None:
        """Restore ParameterManager objective weights before applying one policy."""
        for objective in self.management_objectives:
            objective.weight = self.require_scalar(
                f"objectives.{objective.name}.weight",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
        self.strategy_nutrient_multiplier = self.require_scalar(
            "strategies.baseline.nutrient_multiplier",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )

    def _apply_organic_strategy(self) -> None:
        for obj in self.management_objectives:
            if obj.name == "soil_carbon_sequestration":
                obj.weight = self.require_scalar(
                    "strategies.organic.soil_carbon_weight",
                    namespace="anthroposphere.management.coordinator",
                    source_order=("manager",),
                )
            elif obj.name == "biodiversity_index":
                obj.weight = self.require_scalar(
                    "strategies.organic.biodiversity_weight",
                    namespace="anthroposphere.management.coordinator",
                    source_order=("manager",),
                )
        self.strategy_nutrient_multiplier = self.require_scalar(
            "strategies.organic.nutrient_multiplier",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )

    def _apply_precision_strategy(self) -> None:
        weight_multiplier = self.require_scalar(
            "strategies.precision.efficiency_weight_multiplier",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        for obj in self.management_objectives:
            if "efficiency" in obj.name:
                obj.weight *= weight_multiplier
        self.strategy_nutrient_multiplier = self.require_scalar(
            "strategies.precision.nutrient_multiplier",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )

    def _apply_productivity_strategy(self) -> None:
        weight_multiplier = self.require_scalar(
            "strategies.productivity.objective_weight_multiplier",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        for obj in self.management_objectives:
            if obj.name in ("economic_return", "water_use_efficiency"):
                obj.weight = min(1.0, obj.weight * weight_multiplier)
        self.strategy_nutrient_multiplier = self.require_scalar(
            "strategies.productivity.nutrient_multiplier",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )

    def _apply_conservation_strategy(self) -> None:
        weight_multiplier = self.require_scalar(
            "strategies.conservation.objective_weight_multiplier",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        for obj in self.management_objectives:
            if obj.name in ("soil_carbon_sequestration", "biodiversity_index"):
                obj.weight = min(1.0, obj.weight * weight_multiplier)
        self.strategy_nutrient_multiplier = self.require_scalar(
            "strategies.conservation.nutrient_multiplier",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )

    def _apply_sustainable_strategy(self) -> None:
        total_weight = sum(obj.weight for obj in self.management_objectives)
        if total_weight <= 0.0:
            raise ValueError("Management objective weights must sum to a positive value")
        for obj in self.management_objectives:
            obj.weight = obj.weight / total_weight

    def _apply_balanced_strategy(self) -> None:
        total_weight = sum(obj.weight for obj in self.management_objectives)
        if total_weight <= 0.0:
            raise ValueError("Management objective weights must sum to a positive value")
        for obj in self.management_objectives:
            obj.weight = obj.weight / total_weight
    
    def set_coordination_mode(self, mode: CoordinationMode) -> None:
        """Set coordination mode"""
        self.coordination_mode = mode
        
        if mode == CoordinationMode.STRICT_PRIORITY:
            self.scheduler.conflict_strategy = ConflictResolution.PRIORITY_BASED
        elif mode == CoordinationMode.COOPERATIVE:
            self.scheduler.conflict_strategy = ConflictResolution.MERGE
        elif mode == CoordinationMode.ADAPTIVE:
            self.scheduler.conflict_strategy = ConflictResolution.SEQUENTIAL

    def _event_duration_seconds(self, event_type: EventType) -> float:
        """Return one ParameterManager-backed event duration in seconds."""
        parameter_names = {
            EventType.IRRIGATION: "event_duration_hours.irrigation",
            EventType.TILLAGE: "event_duration_hours.tillage",
            EventType.HARVEST: "event_duration_hours.harvest",
            EventType.GRAZING: "event_duration_hours.grazing",
            EventType.FIRE: "event_duration_hours.fire_management",
        }
        if event_type not in parameter_names:
            raise ValueError(
                f"ManagementCoordinator has no declared duration policy for '{event_type.value}'"
            )
        duration_hours = self.require_scalar(
            parameter_names[event_type],
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        if duration_hours <= 0.0:
            raise ValueError("Management event duration must be positive")
        return duration_hours * self.seconds_per_hour
    
    def schedule_management_event(
        self,
        event_type: EventType,
        scheduled_time: float,
        parameters: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        resource_requirements: Optional[Dict[str, float]] = None,
        spatial_extent: Optional[Any] = None
    ) -> bool:
        """
        Schedule a management event through the coordinator
        
        Args:
            event_type: Type of management event
            scheduled_time: When to execute the event
            parameters: Event-specific parameters
            priority: Event priority level
            resource_requirements: Required resources
            spatial_extent: Spatial coverage of event
            
        Returns:
            bool: True if event scheduled successfully
        """
        # Generate unique event ID
        event_id = f"{event_type.value}_{int(scheduled_time)}_{len(self.scheduled_events)}"
        
        duration = self._event_duration_seconds(event_type)
        
        # Create event
        event = ManagementEvent(
            event_id=event_id,
            event_type=event_type,
            scheduled_time=scheduled_time,
            priority=priority,
            duration=duration,
            parameters=parameters,
            resource_requirements=resource_requirements or {},
            spatial_extent=spatial_extent
        )
        
        # Schedule through scheduler
        success = self.scheduler.schedule_event(event)
        
        if success:
            self.scheduled_events.append(event)
            logger.info(f"Scheduled {event_type.value} event at time {scheduled_time}")
        else:
            logger.warning(f"Failed to schedule {event_type.value} event at time {scheduled_time}")
        
        return bool(success)
    
    def coordinate_management_activities(
        self,
        current_time: float,
        state: Dict[str, torch.Tensor],
        planning_horizon: Optional[float] = None,
        *,
        calendar_type: Optional[str] = None,
    ) -> CoordinationResult:
        """
        Coordinate management activities for the planning horizon

        Args:
            current_time: Current simulation time
            state: Current system state (the declared inputs)
            planning_horizon: Planning horizon in days. If None, use ParameterManager value.
            calendar_type: Simulator calendar passed as the forward keyword; resolved
                against the configured calendar.
            
        Returns:
            CoordinationResult: Result of coordination process
        """
        warnings_list = []
        if planning_horizon is None:
            planning_horizon = self.require_scalar(
                "coordinator.planning_horizon_days",
                source_order=("manager",),
            )
        
        resolved_calendar = CalendarHandler.resolve_simulator_calendar(
            configured_calendar=self._configured_calendar_type,
            runtime_calendar=calendar_type,
            owner_name=self.__class__.__name__,
        )

        # Update the seasonal context from the simulator calendar
        self._update_environmental_context(
            state,
            current_time,
            resolved_calendar,
        )

        # Assess current conditions and needs
        management_needs = self._assess_management_needs(state, current_time)
        
        # Generate candidate management events
        candidate_events = self._generate_candidate_events(
            current_time, planning_horizon, management_needs, state
        )
        
        # Optimize event scheduling based on objectives
        if self.coordination_mode == CoordinationMode.OPTIMAL:
            optimized_events = self._optimize_event_schedule(candidate_events, state)
        else:
            optimized_events = candidate_events
        
        # Schedule optimized events
        scheduled_events = []
        conflicts_resolved = 0
        
        for event in optimized_events:
            success = self.scheduler.schedule_event(event)
            if success:
                scheduled_events.append(event)
            else:
                conflicts_resolved += 1
                warnings_list.append(f"Failed to schedule {event.event_type.value} event")
        
        # Calculate resource allocation
        resource_allocation = self._calculate_resource_allocation(scheduled_events)
        
        # Evaluate objective achievement
        objective_scores = self._evaluate_objectives(scheduled_events, state)
        
        # Create result
        result = CoordinationResult(
            scheduled_events=scheduled_events,
            conflicts_resolved=conflicts_resolved,
            resource_allocation=resource_allocation,
            objective_scores=objective_scores,
            coordination_status="completed",
            warnings=warnings_list
        )
        
        self.coordination_history.append(result)
        
        return result
    
    def forward(
        # References: Allen et al. (1998), FAO-56; Lal (2004); Matson et al. (1997);
        #   Robertson and Swinton (2005); van der Schaft and Maschke (1995); CF-1.8 calendars.
        # Mathematical expression: P_eff = P_base W_resource W_timing W_conflict;
        #   q_i = w_i (1 + s_i); p_i = q_i / sum_j q_j; c = sum_r R_r / Q_r;
        #   eta = clip(N / c, 0, 1); 0 <= U_r <= Q_r; d = floor(t / seconds_per_day) mod Y(C) + 1.
        # Parameter meanings: P_eff effective priority; w_i, s_i weight and stress of activity i;
        #   R_r requested and Q_r available amount of resource r; U_r allocated use; N the
        #   resource budget; C the explicit simulator calendar with Y(C) days.
        # Units: priorities, weights, p_i, eta [1]; R_r, Q_r, U_r in the quota unit declared
        #   per resource (_quota_unit); t, dt [s]; d [day].
        # Domain/constraints: finite non-negative requests; allocations never exceed quotas;
        #   the calendar must be declared; the coordinator schedules and allocates and does not
        #   advance biogeochemical state itself.
        # Sign convention: U_r is positive consumption of resource r; priorities are non-negative.
        self,
        state: Dict[str, torch.Tensor],
        time: float,
        dt: float,
        calendar_type: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Execute coordination and management activities.

        References:
            Integrated agricultural systems management; CF-1.8 calendar
            coordinates; revised foundations, management boundary contract.
        Mathematical expression:
            P_eff = P_base W_resource W_timing W_conflict, with event day
            d = floor(t / seconds_per_day) mod Y(C) + 1.
        Parameter meanings:
            P_eff is effective priority [1], t and dt are simulation time and
            step [s], C is the explicit simulator calendar, and Y(C) is its
            exact year length [day year-1].
        """
        calendar_type = CalendarHandler.resolve_simulator_calendar(
            configured_calendar=self._configured_calendar_type,
            runtime_calendar=calendar_type,
            owner_name=self.__class__.__name__,
        )
        self._validate_canonical_inputs(state, calendar_type=calendar_type)
        dt_tensor = torch.as_tensor(dt, dtype=torch.float64)
        if dt_tensor.numel() != 1 or not torch.all(torch.isfinite(dt_tensor)):
            raise ValueError("ManagementCoordinator dt must be one finite scalar")
        if torch.any(dt_tensor <= 0.0):
            raise ValueError("ManagementCoordinator dt must be positive")

        working_state = self._prepare_forward_state(state)
        self._coordinate_if_due(
            time,
            working_state,
            calendar_type,
        )

        # Execute scheduled events
        executed_events = self.scheduler.execute_until(time, working_state)
        self.executed_events.extend(executed_events)

        # Aggregate management effects
        management_effects = self._aggregate_management_effects(executed_events, working_state, dt)
        coordination_decisions = self.compute_coordination_decisions(working_state)
        self._merge_coordination_decisions(management_effects, coordination_decisions)

        # Update coordination status
        coordination_status = self._get_coordination_status()
        
        # Prepare outputs
        public_effect_keys = (
            "total_irrigation_flux",
            "total_harvest_flux",
            "total_grazing_flux",
            "total_fire_co2_flux",
            "irrigation_demand_flux",
            "fertilizer_product_flux",
            "tillage_intensity",
        )
        public_effects = {
            key: management_effects[key]
            for key in public_effect_keys
        }
        reference = next(iter(public_effects.values()))
        outputs: Dict[str, torch.Tensor] = {
            "coordination_status": reference.new_tensor(coordination_status),
            **public_effects,
            "diagnostics.management_coordinator.active": reference.new_tensor(
                len(executed_events) > 0
            ),
            "diagnostics.management_coordinator.events_executed": reference.new_tensor(
                len(executed_events)
            ),
            "diagnostics.management_coordinator.irrigation_events_count": management_effects[
                "irrigation_events_count"
            ],
            "diagnostics.management_coordinator.tillage_events_count": management_effects[
                "tillage_events_count"
            ],
            "diagnostics.management_coordinator.harvest_events_count": management_effects[
                "harvest_events_count"
            ],
            "diagnostics.management_coordinator.grazing_events_count": management_effects[
                "grazing_events_count"
            ],
            "diagnostics.management_coordinator.fire_events_count": management_effects[
                "fire_events_count"
            ],
            "diagnostics.management_coordinator.priority_weights_sum": management_effects[
                "priority_weights_sum"
            ],
            "diagnostics.management_coordinator.interaction_effects_count": management_effects[
                "interaction_effects_count"
            ],
            "diagnostics.management_coordinator.conflict_resolution_applied": management_effects[
                "conflict_resolution_applied"
            ],
        }
        mass_balance_error = self._check_conservation(
            state=working_state,
            effects=management_effects,
            dt=dt,
        )
        outputs["diagnostics.coordinator_mass_balance_error"] = mass_balance_error
        self._last_management_mass_residual = mass_balance_error.detach()
        outputs = self._attach_canonical_runtime_diagnostic(outputs)
        return self._wrap_outputs(outputs)

    def _wrap_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, UpdateItem]:
        """Attach explicit replacement semantics to every retained output."""

        if set(outputs) != set(self.OUTPUT_UNITS):
            missing = sorted(set(self.OUTPUT_UNITS) - set(outputs))
            extra = sorted(set(outputs) - set(self.OUTPUT_UNITS))
            raise RuntimeError(
                f"ManagementCoordinator output contract mismatch: missing={missing}, extra={extra}"
            )
        return {
            name: UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit=self.OUTPUT_UNITS[name],
                source_functor=self.name,
            )
            for name, value in outputs.items()
        }

    def _attach_canonical_runtime_diagnostic(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Attach the Phase 8 canonical runtime diagnostic flag."""

        device = getattr(self, "device", torch.device("cpu"))
        dtype = torch.float32
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                dtype = value.dtype if torch.is_floating_point(value) else torch.float32
                break
        outputs["diagnostics.management_coordinator.canonical_iphs_runtime"] = torch.ones(
            (), device=device, dtype=dtype
        )
        return outputs

    def _prepare_forward_state(
        self,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Copy the accepted canonical state without environment side channels."""
        return dict(state)

    def _coordinate_if_due(
        self,
        time: float,
        working_state: Dict[str, torch.Tensor],
        calendar_type: str,
    ) -> None:
        if time - self.last_coordination_time < self.coordination_interval:
            return
        _ = self.coordinate_management_activities(
            time,
            working_state,
            calendar_type=calendar_type,
        )
        self.last_coordination_time = time

    def _merge_coordination_decisions(
        self,
        management_effects: Dict[str, torch.Tensor],
        coordination_decisions: Dict[str, Any],
    ) -> None:
        for key, value in coordination_decisions["management_decisions"].items():
            management_effects[key] = value
        priority_stack = torch.stack([tensor for tensor in coordination_decisions["priority_weights"].values()])
        management_effects["priority_weights_sum"] = priority_stack.sum()
        management_effects["interaction_effects_count"] = torch.tensor(
            len(coordination_decisions["interaction_effects"]),
            dtype=torch.float32,
        )
        management_effects["conflict_resolution_applied"] = coordination_decisions["constraint_satisfaction"]

    def _check_conservation(
        self,
        *,
        state: Dict[str, torch.Tensor],
        effects: Dict[str, torch.Tensor],
        dt: float,
    ) -> torch.Tensor:
        """
        Check management mass-balance constraints for aggregated coordinator fluxes.

        References:
            - Allen et al. (1998) FAO-56
            - Lal (2005) Environment International 31, 575-584
        Mathematical expression:
            epsilon = max(0, (F_harvest + F_grazing) - B_ag / dt - tol) + sum(max(0, -F_i))
        Parameter meanings:
            F_harvest: total harvest removal flux [kg m-2 s-1]
            F_grazing: total grazing removal flux [kg m-2 s-1]
            B_ag: available aboveground biomass [kg m-2]
            dt: timestep [s]
            F_i: managed mass-related fluxes expected to be non-negative [kg m-2 s-1]
            tol: conservation tolerance [kg m-2 s-1]
        """
        reference = effects["total_irrigation_flux"]
        if not isinstance(reference, torch.Tensor):
            raise TypeError("Management total_irrigation_flux must be a torch.Tensor")
        if not torch.is_floating_point(reference):
            reference = reference.to(dtype=torch.float32)

        tol_val = self.get_parameter_value(
            "conservation_tolerance",
            namespace="validation.conservation",
            source_order=("manager",),
            required=True,
            validate=True,
        )
        if isinstance(tol_val, torch.Tensor):
            tol_val = self._scalar_value(tol_val)
        tolerance = torch.as_tensor(
            float(tol_val),
            dtype=reference.dtype,
            device=reference.device,
        )

        def _as_reference(value: Any) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                tensor = value.to(device=reference.device, dtype=reference.dtype)
            else:
                tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
            if tensor.shape == reference.shape:
                return tensor
            if tensor.dim() == 0:
                return torch.ones_like(reference) * tensor.reshape(())
            return torch.ones_like(reference) * tensor.detach().mean()

        harvest_flux = _as_reference(effects["total_harvest_flux"])
        grazing_flux = _as_reference(effects["total_grazing_flux"])
        irrigation_flux = _as_reference(effects["total_irrigation_flux"])
        dt_tensor = torch.as_tensor(
            dt,
            dtype=reference.dtype,
            device=reference.device,
        )
        aboveground = _as_reference(
            self._require_state_tensor(state, "aboveground_biomass")
        )
        max_removal_flux = torch.clamp(aboveground, min=0.0) / dt_tensor

        biomass_removal_flux = torch.clamp(harvest_flux, min=0.0) + torch.clamp(grazing_flux, min=0.0)
        biomass_excess_error = torch.relu(biomass_removal_flux - max_removal_flux - tolerance)

        negative_biomass_flux_error = (
            torch.relu(-harvest_flux) + torch.relu(-grazing_flux)
        )
        _ = irrigation_flux
        return biomass_excess_error + negative_biomass_flux_error

    def check_conservation_laws(
        self,
        input_state: Dict[str, torch.Tensor],
        output_updates: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Audit open management ports and scheduler resource quotas separately."""
        _ = input_state

        def _unwrap_tensor(value: Any) -> torch.Tensor:
            payload = value.value if hasattr(value, "value") else value
            return torch.as_tensor(payload, device=self.device, dtype=torch.float32)

        # References:
        #   - Robertson and Swinton (2005), managed agroecosystem exchanges.
        #   - Matson et al. (1997), external agricultural resource inputs.
        # Mathematical expressions:
        #   epsilon_mass = max(|R_mass|, max_p relu(-F_p)).
        #   epsilon_quota = max_r(relu(U_r - Q_r) + relu(-U_r)).
        # Parameter meanings:
        #   R_mass: biomass-removal residual diagnosed for the current step
        #     [kg m-2 s-1].
        #   F_p: signed flow on an external management port, positive in its
        #     declared input or export direction [port-specific flux units].
        #   U_r: allocated amount for resource r [quota-specific units].
        #   Q_r: available amount for resource r [quota-specific units].
        if "diagnostics.coordinator_mass_balance_error" not in output_updates:
            raise KeyError("Missing coordinator mass-balance diagnostic")
        residual_value = output_updates["diagnostics.coordinator_mass_balance_error"]
        mass_violation = float(_unwrap_tensor(residual_value).detach().abs().max().item())

        open_mass_ports = ("total_harvest_flux", "total_grazing_flux")
        for port_name in open_mass_ports:
            if port_name not in output_updates:
                continue
            port_value = _unwrap_tensor(output_updates[port_name])
            negative_flow = float(torch.relu(-port_value).detach().max().item())
            mass_violation = max(mass_violation, negative_flow)

        quota_violation = 0.0
        for quota in self.scheduler.resource_quotas.values():
            overuse = max(0.0, float(quota.current_used - quota.total_available))
            negative_use = max(0.0, float(-quota.current_used))
            quota_violation = max(quota_violation, overuse + negative_use)

        return {
            "mass": mass_violation,
            "resource_quotas": quota_violation,
        }
    
    def _update_environmental_context(
        self,
        state: Dict[str, torch.Tensor],
        current_time: float,
        calendar_type: str,
    ) -> None:
        """Update the seasonal context from the simulator calendar.

        References:
            CF Conventions 1.8, Section 4.4; revised foundations, calendar
            and management-window contract.
        Mathematical expression:
            d = floor(t / seconds_per_day) mod Y(C) + 1, followed by the
            declared seasonal partition of d.
        Parameter meanings:
            t is current_time [s], C is calendar_type, Y(C) is the exact
            simulator year length [day year-1], and d is day_of_year [day].
        """
        # Determine current season from canonical state or simulation time.
        day_tensor: Optional[torch.Tensor] = None
        if "day_of_year" in state:
            declared_day = self._require_state_tensor(state, "day_of_year")
            day_tensor = torch.as_tensor(
                resolve_day_of_year(
                    current_time,
                    declared_day,
                    calendar_type=calendar_type,
                ),
                device=declared_day.device,
                dtype=torch.float32,
            )

        if day_tensor is None:
            day_tensor = torch.as_tensor(
                resolve_day_of_year(
                    current_time,
                    calendar_type=calendar_type,
                ),
                device=getattr(self, "device", None),
                dtype=torch.float32,
            )

        if day_tensor is not None and not torch.is_floating_point(day_tensor):
            day_tensor = day_tensor.to(dtype=torch.float32)

        day_value = self._scalar_value(day_tensor.mean())
        spring_start = self.require_scalar(
            "season_boundaries.spring_start_day",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        summer_start = self.require_scalar(
            "season_boundaries.summer_start_day",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        autumn_start = self.require_scalar(
            "season_boundaries.autumn_start_day",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        winter_start = self.require_scalar(
            "season_boundaries.winter_start_day",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        days_per_year = CalendarHandler.days_per_simulator_year(calendar_type)
        if not (
            1.0
            <= spring_start
            < summer_start
            < autumn_start
            < winter_start
            <= float(days_per_year)
        ):
            raise ValueError(
                "Management season boundaries must be strictly ordered within "
                f"[1, {days_per_year}]"
            )

        if spring_start <= day_value < summer_start:
            self.current_season = "spring"
        elif summer_start <= day_value < autumn_start:
            self.current_season = "summer"
        elif autumn_start <= day_value < winter_start:
            self.current_season = "autumn"
        else:
            self.current_season = "winter"
    
    def _assess_management_needs(
        self,
        state: Dict[str, torch.Tensor],
        current_time: float
    ) -> Dict[str, float]:
        """
        Assess management needs based on current conditions
        
        This method evaluates the current system state to determine the urgency
        and necessity of different management activities. The assessment is based
        on scientific principles of agronomy and ecosystem management.
        
        Mathematical Expressions:
            Soil moisture deficit calculation:
            D_moisture = (FC - SM) / (FC - WP)
            
            Biomass readiness calculation:
            R_biomass = min(AGB / AGB_threshold, 1.0)
            
            where:
            - D_moisture is moisture deficit [1]
            - FC is field capacity [m3/m3]
            - SM is current soil moisture [m3/m3]
            - WP is wilting point [m3/m3]
            - R_biomass is biomass readiness [1]
            - AGB is aboveground biomass [kg/m2]
            - AGB_threshold is harvest threshold [kg/m2]
            
        Args:
            state: Current system state containing soil moisture, biomass, etc.
            current_time: Current simulation time for seasonal assessment
            
        Returns:
            Dictionary of management need scores [0-1] for each activity type
        """
        needs = {}

        # References: Allen et al. (1998), FAO Irrigation and Drainage Paper 56,
        #   Chapter 8 (root-zone depletion between field capacity and wilting point).
        # Mathematical expression: D = clip((theta_fc - theta) / (theta_fc - theta_wp),
        #   c_min, c_max); need_irrigation = mean(D).
        # Parameter meanings: theta is the soil moisture; theta_fc and theta_wp are the
        #   site's field capacity and wilting point (state inputs, not parameters); D
        #   is the relative depletion; c_min and c_max are the management response
        #   limits.
        # Units: theta, theta_fc, theta_wp [m3 m-3]; D, c_min, c_max [1].
        # Domain/constraints: theta_fc > theta_wp in every cell, checked here; a
        #   missing limit raises in _require_state_tensor.
        # Sign convention: D = 0 at field capacity and grows toward 1 as the root
        #   zone dries to the wilting point.
        soil_moisture = self._require_state_tensor(state, "soil_moisture")
        field_capacity = self._require_state_tensor(state, "field_capacity").to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        wilting_point = self._require_state_tensor(state, "wilting_point").to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        if bool(torch.any(field_capacity <= wilting_point).item()):
            raise ValueError(
                "ManagementCoordinator input 'field_capacity' must exceed 'wilting_point'"
            )
        clamp_min = self.require_scalar(
            "clamp_min",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        clamp_max = self.require_scalar(
            "clamp_max",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        moisture_deficit = (field_capacity - soil_moisture) / (field_capacity - wilting_point)
        needs["irrigation"] = self._scalar_value(
            torch.clamp(moisture_deficit, min=clamp_min, max=clamp_max).mean()
        )

        # Harvest need based on crop maturity and seasonal timing
        # Scientific basis: Crops should be harvested when they reach physiological
        # maturity and seasonal conditions are optimal
        # Reference: FAO, 2018, Crop Water Productivity
        aboveground_biomass = self._require_state_tensor(
            state, "aboveground_biomass"
        )
        biomass_threshold = self.require_scalar(
            "coordinator.harvest_biomass_threshold",
            source_order=("manager",),
        )
        # Typical agricultural crops reach harvest maturity at 1.5-3.0 kg/m2
        # Reference: Hay, R. K. M., 1995. Harvest Index: A Review  # kg/m^2 threshold for harvest readiness
        biomass_readiness = torch.clamp(aboveground_biomass / biomass_threshold, min=clamp_min, max=clamp_max)
        
        # Higher need in harvest season (autumn)
        season_factor_name = (
            "need_scores.harvest.autumn_factor"
            if self.current_season == "autumn"
            else "need_scores.harvest.other_factor"
        )
        season_factor = self.require_scalar(
            season_factor_name,
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        needs["harvest"] = self._scalar_value(biomass_readiness.mean()) * season_factor
        
        # Tillage need based on season and management strategy
        if self.current_season in ["spring", "autumn"]:
            tillage_score_name = (
                "need_scores.tillage.conventional_peak"
                if self.management_strategy == ManagementStrategy.CONVENTIONAL
                else "need_scores.tillage.alternative_peak"
            )
        else:
            tillage_score_name = "need_scores.tillage.off_season"
        needs["tillage"] = self.require_scalar(
            tillage_score_name,
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        
        # Grazing need (seasonal and biomass dependent)
        if self.current_season in ["spring", "summer"]:
            grazing_biomass_reference = self.require_scalar(
                "need_scores.grazing.biomass_reference",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            if grazing_biomass_reference <= 0.0:
                raise ValueError("Grazing biomass reference must be positive")
            grazing_biomass_factor = torch.clamp(
                aboveground_biomass / grazing_biomass_reference,
                min=clamp_min,
                max=clamp_max,
            )
            needs["grazing"] = self._scalar_value(grazing_biomass_factor.mean())
        else:
            needs["grazing"] = self.require_scalar(
                "need_scores.grazing.off_season",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
        
        # Fire management need (low but present for prescribed burns)
        fire_factor_name = (
            "need_scores.fire.peak_factor"
            if self.current_season in ["spring", "autumn"]
            else "need_scores.fire.off_season_factor"
        )
        fire_season_factor = self.require_scalar(
            fire_factor_name,
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        fire_base_need = self.require_scalar(
            "need_scores.fire.base_need",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        needs["fire_management"] = fire_season_factor * fire_base_need
        
        return needs
    
    def _generate_candidate_events(
        self,
        current_time: float,
        planning_horizon: float,
        management_needs: Dict[str, float],
        state: Dict[str, torch.Tensor]
    ) -> List[ManagementEvent]:
        """Generate candidate management events for the planning horizon"""
        candidates = []
        horizon_seconds = planning_horizon * self.seconds_per_day
        need_threshold = self.require_scalar(
            "candidate_need_threshold",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        high_priority_threshold = self.require_scalar(
            "high_priority_threshold",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )
        activity_event_types = {
            "irrigation": EventType.IRRIGATION,
            "harvest": EventType.HARVEST,
            "tillage": EventType.TILLAGE,
            "grazing": EventType.GRAZING,
            "fire_management": EventType.FIRE,
        }
        activity_delays = {
            activity: self.require_scalar(
                f"event_delay_days.{activity}",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            for activity in activity_event_types
        }
        
        # Generate events based on needs and timing
        for activity, need_score in management_needs.items():
            if activity not in activity_event_types:
                raise KeyError(f"Unsupported management activity '{activity}'")
            if not torch.isfinite(torch.as_tensor(need_score)) or not 0.0 <= need_score <= 1.0:
                raise ValueError(
                    f"Management need score for '{activity}' must be finite and within [0, 1]"
                )
            if need_score > need_threshold:
                
                # Determine optimal timing within horizon
                if activity == "irrigation":
                    priority = (
                        EventPriority.HIGH
                        if need_score > high_priority_threshold
                        else EventPriority.NORMAL
                    )
                    
                elif activity == "harvest":
                    priority = EventPriority.CRITICAL
                    
                elif activity == "tillage":
                    priority = EventPriority.NORMAL
                    
                elif activity == "grazing":
                    priority = EventPriority.NORMAL
                    
                else:
                    priority = EventPriority.LOW

                event_time = current_time + activity_delays[activity] * self.seconds_per_day
                
                # Only schedule if within planning horizon
                if event_time <= current_time + horizon_seconds:
                    # Create candidate event
                    event_id = f"candidate_{activity}_{int(event_time)}"
                    event_type = activity_event_types[activity]
                    
                    candidate = ManagementEvent(
                        event_id=event_id,
                        event_type=event_type,
                        scheduled_time=event_time,
                        priority=priority,
                        duration=self._event_duration_seconds(event_type),
                        parameters={"need_score": need_score},
                        resource_requirements=self._estimate_resource_requirements(activity, need_score)
                    )
                    
                    candidates.append(candidate)
        
        return candidates
    
    def _optimize_event_schedule(
        self,
        candidate_events: List[ManagementEvent],
        state: Dict[str, torch.Tensor]
    ) -> List[ManagementEvent]:
        """Optimize event schedule using multi-objective optimization"""
        if not candidate_events:
            return []
        
        # Simple greedy optimization based on priority and resource efficiency
        # In a full implementation, this could use more sophisticated algorithms
        
        # Sort candidates by effective priority
        def effective_priority(event: ManagementEvent) -> float:
            base_priority = event.priority.value
            resource_efficiency = self._calculate_resource_efficiency(event)
            return float(base_priority + (1.0 - resource_efficiency))
        
        optimized = sorted(candidate_events, key=effective_priority)
        
        # Remove events that would exceed resource quotas
        filtered = []
        resource_usage: Dict[str, float] = defaultdict(float)
        
        for event in optimized:
            can_add = True
            for resource, amount in event.resource_requirements.items():
                if resource in self.scheduler.resource_quotas:
                    quota = self.scheduler.resource_quotas[resource]
                    if resource_usage[resource] + amount > quota.remaining:
                        can_add = False
                        break
            
            if can_add:
                filtered.append(event)
                for resource, amount in event.resource_requirements.items():
                    resource_usage[resource] += amount
        
        return filtered
    
    def _estimate_resource_requirements(self, activity: str, need_score: float) -> Dict[str, float]:
        """
        Estimate resource requirements for management activity.

        Requirements are based on activity type and scaled by need_score.
        Uses structured base values per activity type with economic and
        ecological cost categories.

        References
        ----------
        - FAO (2017). Guidelines for sustainable agricultural management.
        - Smith et al. (2016). Resource use efficiency in agricultural systems.

        Mathematical Expression
        -----------------------
        R_i = R_base_i * scale_factor
        where:
        - R_i is resource requirement for resource type i [units vary]
        - R_base_i is base requirement for the activity-resource pair
        - scale_factor = need_score for demand-responsive resources (e.g. water),
          1.0 for fixed resources (e.g. labor, machinery)

        Parameters
        ----------
        activity : str
            Management activity type (irrigation, tillage, harvest, grazing,
            fire_management)
        need_score : float
            Need-based scaling factor [0, 1] for demand-responsive resources

        Returns
        -------
        Dict[str, float]
            Resource type to amount mapping
        """
        # References: configurable farm-operation budgets.
        # Mathematical expression: R_r = R_base,r * s for demand-scaled
        # resources and R_r = R_base,r otherwise.
        # Parameter meanings: R_base,r and the registered quota Q_r use the
        # same administrative unit; s is the dimensionless need score.
        resource_scaling = {
            "irrigation": {
                "water": True,
                "labor": False,
                "fuel": False,
            },
            "tillage": {
                "fuel": False,
                "labor": False,
                "machinery": False,
            },
            "harvest": {
                "fuel": False,
                "labor": False,
                "machinery": False,
            },
            "grazing": {
                "labor": False,
            },
            "fire_management": {
                "labor": False,
                "water": True,
            },
        }

        table = resource_scaling.get(activity)
        if table is None:
            raise KeyError(f"Unsupported management activity '{activity}'")
        if not torch.isfinite(torch.as_tensor(need_score)) or not 0.0 <= need_score <= 1.0:
            raise ValueError("Management resource need_score must be finite and within [0, 1]")
        manager = self.get_parameter_manager()
        if manager is None:
            raise RuntimeError("ManagementCoordinator requires ParameterManager resource metadata")

        result = {}
        for resource, is_scaled in table.items():
            relative_key = f"resource_requirements.{activity}.{resource}"
            full_key = f"anthroposphere.management.coordinator.{relative_key}"
            base_amount = self.require_scalar(
                relative_key,
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            spec = manager.get_parameter(full_key)
            quota = self.scheduler.resource_quotas.get(resource)
            if spec is None or quota is None:
                raise KeyError(f"Missing resource contract for '{full_key}'")
            if str(spec.units) != str(quota.unit):
                raise ValueError(
                    f"Resource requirement '{full_key}' unit '{spec.units}' does not "
                    f"match quota unit '{quota.unit}'"
                )
            if is_scaled:
                result[resource] = base_amount * need_score
            else:
                result[resource] = base_amount
        return result
    
    def _calculate_resource_efficiency(self, event: ManagementEvent) -> float:
        """Calculate a dimensionless quota-normalized scheduling score.

        References:
            - Robertson and Swinton (2005), multi-objective farm management.
        Mathematical expression:
            c = sum_r R_r / Q_r; eta = min(max(N / c, 0), 1).
        Parameter meanings:
            R_r is the event requirement in the declared quota unit for resource r,
            Q_r is the positive available quota in the same unit, N is the
            dimensionless management need score, and eta is a scheduling proxy.

        Notes:
            This is a configurable decision heuristic, not a universal physical law.
        """
        if not event.resource_requirements:
            return 1.0

        if "need_score" not in event.parameters:
            raise KeyError(
                f"Management event '{event.event_id}' requires parameter 'need_score'"
            )
        need_score = float(event.parameters["need_score"])
        if not torch.isfinite(torch.as_tensor(need_score)) or not 0.0 <= need_score <= 1.0:
            raise ValueError("Management event need_score must be finite and within [0, 1]")

        normalized_cost = 0.0
        for resource, raw_amount in event.resource_requirements.items():
            quota = self.scheduler.resource_quotas.get(resource)
            if quota is None:
                raise KeyError(
                    f"Management event '{event.event_id}' references unknown resource '{resource}'"
                )
            amount = float(raw_amount)
            if not torch.isfinite(torch.as_tensor(amount)) or amount < 0.0:
                raise ValueError(
                    f"Management resource requirement '{resource}' must be finite and non-negative"
                )
            if quota.total_available <= 0.0:
                raise ValueError(
                    f"Management resource quota '{resource}' must be positive"
                )
            normalized_cost += amount / float(quota.total_available)

        if normalized_cost == 0.0:
            return 1.0
        efficiency = need_score / normalized_cost
        return min(max(efficiency, 0.0), 1.0)
    
    def _calculate_resource_allocation(self, events: List[ManagementEvent]) -> Dict[str, float]:
        """Calculate final resource allocation for scheduled events"""
        allocation: Dict[str, float] = defaultdict(float)
        
        for event in events:
            for resource, amount in event.resource_requirements.items():
                allocation[resource] += amount
        
        return dict(allocation)
    
    def _evaluate_objectives(
        self,
        events: List[ManagementEvent],
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate achievement of management objectives"""
        scores = {}
        
        for objective in self.management_objectives:
            if objective.measurement_function:
                score = objective.measurement_function(events, state)
            else:
                # Simple heuristic scoring
                score = self._heuristic_objective_score(objective, events, state)
            
            scores[objective.name] = score
        
        return scores
    
    def _heuristic_objective_score(
        self,
        objective: ManagementObjective,
        events: List[ManagementEvent],
        state: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate heuristic objective achievement score"""
        if objective.name == "water_use_efficiency":
            irrigation_events = [e for e in events if e.event_type == EventType.IRRIGATION]
            event_penalty = self.require_scalar(
                "objective_scoring.water_use_efficiency.event_penalty",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            return max(0.0, min(1.0, 1.0 - len(irrigation_events) * event_penalty))
        
        elif objective.name == "nutrient_use_efficiency":
            # Heuristic: fewer fertilization events relative to harvest events
            # indicates better nutrient cycling / uptake efficiency.
            # Reference: Cassman et al. (2002), Ecological intensification
            # Math: NUE_score = 1.0 - n_fert / max(n_harvest, 1) * 0.3
            #   Capped to [0, 1]. More fertilization per harvest = lower score.
            fertilization_events = [
                e for e in events
                if e.event_type == EventType.FERTILIZATION
            ]
            harvest_events = [
                e for e in events
                if e.event_type == EventType.HARVEST
            ]
            n_fert = len(fertilization_events)
            n_harvest = max(len(harvest_events), 1)
            ratio_penalty = self.require_scalar(
                "objective_scoring.nutrient_use_efficiency.event_ratio_penalty",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            return max(0.0, min(1.0, 1.0 - n_fert / n_harvest * ratio_penalty))
        
        elif objective.name == "soil_carbon_sequestration":
            # Positive for reduced tillage, negative for intensive management
            tillage_events = [e for e in events if e.event_type == EventType.TILLAGE]
            tillage_penalty = self.require_scalar(
                "objective_scoring.soil_carbon_sequestration.tillage_penalty",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            return max(0.0, min(1.0, 1.0 - len(tillage_events) * tillage_penalty))
        
        elif objective.name == "economic_return":
            # Simple proxy based on harvest events
            harvest_events = [e for e in events if e.event_type == EventType.HARVEST]
            harvest_reward = self.require_scalar(
                "objective_scoring.economic_return.harvest_reward",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            return max(0.0, min(1.0, len(harvest_events) * harvest_reward))
        
        elif objective.name == "biodiversity_index":
            # Negative impact from intensive management
            total_events = len(events)
            event_penalty = self.require_scalar(
                "objective_scoring.biodiversity_index.event_penalty",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            return max(0.0, min(1.0, 1.0 - total_events * event_penalty))
        
        return self.require_scalar(
            "objective_scoring.neutral_score",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
        )

    def _build_reference_tensor(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Use canonical soil moisture as the unique spatial template."""
        return self._require_state_tensor(state, "soil_moisture").to(
            dtype=torch.get_default_dtype()
        )

    def _to_spatial_tensor(
        self,
        value: Any,
        ref_tensor: torch.Tensor,
        default_dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=ref_tensor.device, dtype=default_dtype)
        else:
            tensor = torch.as_tensor(value, device=ref_tensor.device, dtype=default_dtype)
        if tensor.dim() == 0:
            return torch.ones_like(ref_tensor, device=ref_tensor.device, dtype=default_dtype) * tensor.reshape(())
        if tensor.shape != ref_tensor.shape:
            raise ValueError(
                f"ManagementCoordinator input shape {tensor.shape} does not match "
                f"canonical spatial shape {ref_tensor.shape}"
            )
        return tensor

    def _seed_management_effects(
        self,
        state: Dict[str, torch.Tensor],
        ref_tensor: torch.Tensor,
        default_dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        return {
            "irrigation_flux": self._to_spatial_tensor(
                state["irrigation_flux"],
                ref_tensor,
                default_dtype,
            ),
            "harvest_flux": self._to_spatial_tensor(
                state["harvest_flux"],
                ref_tensor,
                default_dtype,
            ),
            "grazing_flux": self._to_spatial_tensor(
                state["grazing_flux"],
                ref_tensor,
                default_dtype,
            ),
            "fire_emissions": self._to_spatial_tensor(
                state["co2_emission"],
                ref_tensor,
                default_dtype,
            ),
        }
    
    def _aggregate_management_effects(
        self,
        executed_events: List[ManagementEvent],
        state: Dict[str, torch.Tensor],
        dt: float
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate effects from executed management events
        
        This method combines the outputs from various management activities
        into a unified set of effects on the ecosystem. The aggregation respects
        conservation laws and ensures proper mass balance.
        
        Mathematical Expressions:
            Mass balance for irrigation:
            dSM = sum(Irrigation_flux * dt) / Soil_depth
            
            Biomass removal balance:
            dAGB = sum(Harvest_flux + Grazing_flux) * dt
            
            where:
            - dSM is change in soil moisture [m3/m3]
            - dAGB is change in aboveground biomass [kg/m2]
            - dt is time step [s]
            
        Args:
            executed_events: List of management events that were executed
            state: Current system state
            dt: Time step in seconds
            
        Returns:
            Dictionary of aggregated effects on state variables
        """
        _ = dt
        default_dtype = torch.get_default_dtype()
        ref_tensor = self._build_reference_tensor(state)
        effects = self._seed_management_effects(state, ref_tensor, default_dtype)
        effects["total_irrigation_flux"] = effects["irrigation_flux"]
        effects["total_harvest_flux"] = effects["harvest_flux"]
        effects["total_grazing_flux"] = effects["grazing_flux"]
        effects["total_fire_co2_flux"] = effects["fire_emissions"]

        # Event counts for tracking
        irrigation_events = len([e for e in executed_events if e.event_type == EventType.IRRIGATION])
        tillage_events = len([e for e in executed_events if e.event_type == EventType.TILLAGE])
        harvest_events = len([e for e in executed_events if e.event_type == EventType.HARVEST])
        grazing_events = len([e for e in executed_events if e.event_type == EventType.GRAZING])
        fire_events = len([e for e in executed_events if e.event_type == EventType.FIRE])

        effects["irrigation_events_count"] = torch.tensor(irrigation_events, dtype=torch.float32, device=ref_tensor.device)
        effects["tillage_events_count"] = torch.tensor(tillage_events, dtype=torch.float32, device=ref_tensor.device)
        effects["harvest_events_count"] = torch.tensor(harvest_events, dtype=torch.float32, device=ref_tensor.device)
        effects["grazing_events_count"] = torch.tensor(grazing_events, dtype=torch.float32, device=ref_tensor.device)
        effects["fire_events_count"] = torch.tensor(fire_events, dtype=torch.float32, device=ref_tensor.device)
        return effects
    
    def _get_coordination_status(self) -> float:
        """Get numerical coordination status"""
        # 1.0 = excellent coordination, 0.0 = poor coordination
        if not self.coordination_history:
            return self.require_scalar(
                "status.neutral_score",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
        
        latest = self.coordination_history[-1]
        
        # Base score from successful scheduling
        if latest.scheduled_events:
            success_rate = len(latest.scheduled_events) / (
                len(latest.scheduled_events) + latest.conflicts_resolved
            )
        else:
            success_rate = 0.0
        
        # Adjust for objective achievement
        if latest.objective_scores:
            avg_objective_score = sum(latest.objective_scores.values()) / len(latest.objective_scores)
            success_weight = self.require_scalar(
                "status.success_weight",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            objective_weight = self.require_scalar(
                "status.objective_weight",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
            weight_error = abs(success_weight + objective_weight - 1.0)
            if weight_error > float(torch.finfo(torch.float32).eps):
                raise ValueError("Management coordination status weights must sum to 1")
            combined_score = success_weight * success_rate + objective_weight * avg_objective_score
        else:
            combined_score = success_rate
        
        return combined_score
    
    # Event handler methods
    def _handle_irrigation_event(self, state: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an irrigation control intent for simulator scheduling.
        
        References:
            - Allen et al. (1998) FAO 56 irrigation scheduling.
        Mathematical Expression:
            u_irrigation = (requested_depth, requested_duration)
        Parameter meanings:
            water_amount : float
                Requested irrigation depth [mm].
            dt : float
                Requested event duration [s].

        Notes:
            The coordinator does not evaluate the IrrigationManagement process
            equation. The simulator-owned irrigation Functor remains the only
            writer of ``irrigation_flux``.
        """
        _ = state
        default_water_amount = self.require_scalar(
            "irrigation.default_application_depth",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        water_amount = float(parameters.get("water_amount", default_water_amount))
        duration = float(
            parameters.get("dt", self._event_duration_seconds(EventType.IRRIGATION))
        )
        if not torch.isfinite(torch.tensor(water_amount)) or water_amount < 0.0:
            raise ValueError("Irrigation request depth must be finite and non-negative.")
        if not torch.isfinite(torch.tensor(duration)) or duration <= 0.0:
            raise ValueError("Irrigation request duration must be finite and positive.")
        return {
            "status": "completed",
            "requested_water_depth": water_amount,
            "requested_duration": duration,
        }
    
    def _handle_tillage_event(self, state: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and expose a tillage control intent for simulator scheduling.

        References:
            - Lal (2004), Science 304, 1623-1627.
        Mathematical Expression:
            u_tillage = class(intensity_request)
        Parameter meanings:
            intensity : str
                Tillage intensity category (no_till, reduced, moderate, intensive).
        """
        default_intensity = self.get_parameter_value(
            "event_defaults.tillage_intensity_class",
            namespace="anthroposphere.management.coordinator",
            source_order=("manager",),
            required=True,
        )
        intensity = str(parameters.get("intensity", default_intensity))
        valid_intensity_classes = {"no_till", "reduced", "moderate", "intensive"}
        if intensity not in valid_intensity_classes:
            raise ValueError(f"Unsupported tillage intensity class: {intensity}")
        return {
            "status": "completed",
            "tillage_intensity_class": intensity,
        }
    
    def _handle_harvest_event(self, state: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a harvest control intent for simulator scheduling.

        References:
            - Harvest index concept in crop physiology (Plant physiology and development, 7th ed.).
        Mathematical Expression:
            u_harvest = (requested_harvest_index[, expected_yield])
        Parameter meanings:
            harvest_index : float
                Requested fraction of aboveground biomass [1].
            expected_yield : float, optional
                Event-supplied expected yield [t ha-1], echoed only when present.

        Notes:
            HarvestManagement is the only process owner of ``harvest_flux``.
        """
        _ = state
        default_harvest_index = self.require_scalar(
            "harvest.default_harvest_index",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        harvest_index = float(parameters.get("harvest_index", default_harvest_index))
        if not torch.isfinite(torch.tensor(harvest_index)) or not 0.0 <= harvest_index <= 1.0:
            raise ValueError("Harvest index must be finite and within [0, 1].")
        request: Dict[str, Any] = {
            "status": "completed",
            "requested_harvest_index": harvest_index,
        }
        # The coordinator has no yield model: an expected yield is reported only
        # when the event supplies it.
        if "expected_yield" in parameters:
            expected_yield = float(parameters["expected_yield"])
            if not torch.isfinite(torch.tensor(expected_yield)) or expected_yield < 0.0:
                raise ValueError("Expected yield must be finite and non-negative.")
            request["expected_yield"] = expected_yield
        return request
    
    def _handle_grazing_event(self, state: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a grazing control intent for simulator scheduling.

        References:
            - DNDC95 grazing routines (referenceinstruct/DNDC95/Dndc_farming.cpp).
        Mathematical Expression:
            u_grazing = (stocking_rate, grazing_efficiency,
                         daily_intake_rate, livestock_weight)
        Parameter meanings:
            stocking_rate : float
                Livestock units per hectare (LU ha-1).
            grazing_efficiency : float
                Fraction of intake realized as biomass removal (dimensionless).
            intake_rate : float
                Daily intake per unit body weight (kg DM kg-1 day-1).
            livestock_weight : float
                Reference body weight per livestock unit (kg).

        Notes:
            GrazingManagement is the only process owner of ``grazing_flux``
            and the associated biomass and excreta ledger.
        """
        _ = state
        default_stocking_rate = self.require_scalar(
            "grazing.default_schedule.spring.stocking_rate",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        default_grazing_efficiency = self.require_scalar(
            "grazing.base_efficiency",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        default_intake_rate = self.require_scalar(
            "grazing.daily_intake_rate",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        default_livestock_weight = self.require_scalar(
            "grazing.default_schedule.spring.body_weight",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        default_manure_return = self.require_scalar(
            "grazing.manure_return_fraction",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )

        request = {
            "requested_stocking_rate": float(
                parameters.get("stocking_rate", default_stocking_rate)
            ),
            "requested_grazing_efficiency": float(
                parameters.get("grazing_efficiency", default_grazing_efficiency)
            ),
            "requested_daily_intake_rate": float(
                parameters.get("daily_intake_rate", default_intake_rate)
            ),
            "requested_livestock_weight": float(
                parameters.get("livestock_weight", default_livestock_weight)
            ),
            "requested_manure_return_fraction": float(
                parameters.get("manure_return_fraction", default_manure_return)
            ),
        }
        for name, value in request.items():
            if not torch.isfinite(torch.tensor(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in (
            "requested_grazing_efficiency",
            "requested_manure_return_fraction",
        ):
            if request[name] > 1.0:
                raise ValueError(f"{name} must be within [0, 1].")
        return {
            "status": "completed",
            **request,
        }
    
    def _handle_fire_event(self, state: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a fire control intent for simulator scheduling.

        References:
            - Andreae and Merlet (2001) biomass burning emission factors.
        Mathematical Expression:
            u_fire = (fire_type, fuel_consumption, fuel_load_fraction)
        Parameter meanings:
            fuel_consumption : float
                Requested fraction of fuel load consumed [1].
            fuel_load_fraction : float
                Requested fraction of aboveground biomass treated as fuel [1].

        Notes:
            FireManagement is the only process owner of combustion and
            species-specific emission equations.
        """
        _ = state
        default_fire_type = self.get_parameter_value(
            "fire.default_fire_type",
            namespace="anthroposphere.management",
            source_order=("manager",), required=True)
        if isinstance(default_fire_type, torch.Tensor):
            default_fire_type = default_fire_type.detach().reshape(-1)[0].tolist()
        fire_type = parameters.get('fire_type', default_fire_type)
        default_fuel_consumption = self.require_scalar(
            "fire.fuel_consumption_fraction",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        fuel_consumption = parameters.get('fuel_consumption', default_fuel_consumption)  # fraction
        default_fuel_load_fraction = self.require_scalar(
            "fire.default_fuel_load_fraction",
            namespace="anthroposphere.management",
            source_order=("manager",),
        )
        fuel_consumption_value = float(fuel_consumption)
        fuel_load_fraction_value = float(
            parameters.get("fuel_load_fraction", default_fuel_load_fraction)
        )
        for name, value in (
            ("fuel_consumption", fuel_consumption_value),
            ("fuel_load_fraction", fuel_load_fraction_value),
        ):
            if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1].")
        return {
            "status": "completed",
            "requested_fire_type": fire_type,
            "requested_fuel_consumption": fuel_consumption_value,
            "requested_fuel_load_fraction": fuel_load_fraction_value,
        }
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordination statistics"""
        history_limit = int(
            self.require_scalar(
                "statistics_history_limit",
                namespace="anthroposphere.management.coordinator",
                source_order=("manager",),
            )
        )
        if history_limit < 1:
            raise ValueError("Management statistics history limit must be positive")
        stats = {
            'total_events_scheduled': len(self.scheduled_events),
            'total_events_executed': len(self.executed_events),
            'current_season': self.current_season,
            'management_strategy': self.management_strategy.value,
            'coordination_mode': self.coordination_mode.value,
            'scheduler_stats': self.scheduler.get_statistics(),
            'recent_coordination_results': self.coordination_history[-history_limit:] if self.coordination_history else [],
            'objective_weights': {obj.name: obj.weight for obj in self.management_objectives}
        }
        
        # Calculate success rates
        if self.scheduled_events:
            execution_rate = len(self.executed_events) / len(self.scheduled_events)
            stats['execution_success_rate'] = execution_rate
        
        return stats
    
    def reset_annual_state(self) -> None:
        """Reset annual coordination state"""
        self.scheduler.reset_quotas("yearly")
        self.coordination_history.clear()
        logger.info("Reset annual coordination state")
    
    def reset_monthly_state(self) -> None:
        """Reset monthly coordination state"""
        self.scheduler.reset_quotas("monthly")
        logger.info("Reset monthly coordination state")

    def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
        """
        Upscale fine-resolution coordination outputs to coarse scale.

        References: Conservation of mass during aggregation.
        Mathematical Expression: X_coarse = mean(X_fine)
        Parameters:
            fine_data : Tensor
                Fine-resolution field to aggregate.
        Returns:
            Tensor: Coarse aggregated tensor.
        """
        if fine_data.dim() == 0:
            return fine_data
        return fine_data.mean(dim=0, keepdim=False)

    def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
        """
        Downscale coarse coordination signals to fine scale.

        References: Nearest-neighbor replication to preserve magnitude.
        Mathematical Expression: X_fine = repeat(X_coarse)
        Parameters:
            coarse_data : Tensor
                Coarse-resolution field to distribute.
        Returns:
            Tensor: Fine-resolution tensor with broadcasted values.
        """
        if coarse_data.dim() == 0:
            return coarse_data
        return coarse_data.expand_as(coarse_data)


