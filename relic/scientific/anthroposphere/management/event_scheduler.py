"""
Unified Event Scheduler for Anthropogenic Management Activities

This module provides a centralized scheduling system for all management
activities (fertilization, irrigation, tillage, harvest, etc.) with
conflict resolution, priority handling, and resource quota management.

The scheduler ensures:
1. Temporal coordination of management events
2. Priority-based conflict resolution
3. Resource quota enforcement
4. Idempotent operation updates
5. Event logging and tracking
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import heapq
import warnings

from relic.coupling.scales.temporal.calendar import CalendarHandler
# Parameter management imports
from relic.utils.time_constants import (
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
)


def resolve_day_of_year(
    time: Any,
    explicit_day: Optional[Any] = None,
    *,
    calendar_type: str,
) -> int:
    """Resolve the management calendar day from state or simulation time.

    References:
      - CF-1.8 calendar coordinates.
    Mathematical expression:
      day = floor(t / seconds_per_day) mod days_per_year(C) + 1.
    Parameter meanings:
      time is simulation time [s]; explicit_day is a declared calendar cochain;
      C is the explicit simulator calendar.
    """

    if explicit_day is not None:
        return CalendarHandler.validate_simulator_day(
            explicit_day,
            calendar_type,
        )

    return CalendarHandler.day_of_year_from_elapsed_seconds(
        time,
        calendar_type,
    )


class EventPriority(IntEnum):
    """Priority levels for management events (lower number = higher priority)."""
    EMERGENCY = 0      # Emergency interventions (e.g., pest control, drought response)
    CRITICAL = 1       # Critical operations (e.g., harvest at optimal time)
    HIGH = 2          # High priority (e.g., fertilization at key growth stage)
    NORMAL = 3        # Normal operations (e.g., routine irrigation)
    LOW = 4           # Low priority (e.g., optional amendments)
    DEFERRED = 5      # Can be postponed if needed


class EventType(Enum):
    """Types of management events."""
    FERTILIZATION = "fertilization"
    IRRIGATION = "irrigation"
    TILLAGE = "tillage"
    HARVEST = "harvest"
    PLANTING = "planting"
    PRUNING = "pruning"
    PESTICIDE = "pesticide"
    FIRE = "fire_management"
    GRAZING = "grazing"
    CUSTOM = "custom"


class ConflictResolution(Enum):
    """Strategies for resolving event conflicts."""
    PRIORITY_BASED = "priority"      # Higher priority wins
    FIRST_COME = "first_come"        # First scheduled wins
    LAST_COME = "last_come"          # Last scheduled wins
    MERGE = "merge"                   # Attempt to merge consistent events
    SEQUENTIAL = "sequential"         # Execute sequentially if possible
    REJECT = "reject"                 # Reject conflicting event


@dataclass
class ResourceQuota:
    """Resource quota specification."""
    resource_type: str                # e.g., "water", "nitrogen", "labor"
    total_available: float            # Total available amount
    current_used: float = 0.0         # Currently allocated amount
    unit: str = ""                    # Unit of measurement
    reset_period: Optional[str] = None  # When quota resets (e.g., "daily", "weekly")
    
    @property
    def remaining(self) -> float:
        """Calculate remaining quota."""
        return max(0.0, self.total_available - self.current_used)
    
    def can_allocate(self, amount: float) -> bool:
        """Check if amount can be allocated."""
        return self.remaining >= amount
    
    def allocate(self, amount: float) -> bool:
        """
        Attempt to allocate resource amount.
        
        Returns
        -------
        bool
            True if allocation successful, False if insufficient quota
        """
        if self.can_allocate(amount):
            self.current_used += amount
            return True
        return False
    
    def release(self, amount: float) -> None:
        """Release allocated resources."""
        self.current_used = max(0.0, self.current_used - amount)
    
    def reset(self) -> None:
        """Reset quota to full availability."""
        self.current_used = 0.0


@dataclass
class ManagementEvent:
    """Specification for a management event."""
    event_id: str                     # Unique identifier
    event_type: EventType             # Type of event
    scheduled_time: float             # Scheduled execution time (simulation time)
    priority: EventPriority           # Event priority
    duration: float                   # Expected duration
    parameters: Dict[str, Any]        # Event-specific parameters
    resource_requirements: Dict[str, float] = field(default_factory=dict)  # Resources needed
    spatial_extent: Optional[Any] = None  # Spatial coverage (e.g., field IDs, coordinates)
    handler: Optional[Callable] = None    # Function to execute event
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    executed: bool = False            # Execution status
    execution_result: Optional[Any] = None  # Result of execution
    
    def __lt__(self, other: "ManagementEvent") -> bool:
        """Compare events by time, priority, and stable unique identifier."""
        return (
            self.scheduled_time,
            self.priority,
            self.event_id,
        ) < (
            other.scheduled_time,
            other.priority,
            other.event_id,
        )


@dataclass
class EventConflict:
    """Record of an event conflict."""
    event1: ManagementEvent
    event2: ManagementEvent
    conflict_type: str               # Type of conflict
    resolution: ConflictResolution    # Applied resolution strategy
    resolved_at: float               # Time of resolution
    outcome: str                     # Description of outcome


class EventScheduler:
    """
    Unified event scheduler for management activities.
    
    This scheduler manages all anthropogenic interventions, ensuring
    proper sequencing, resource allocation, and conflict resolution.
    
    Attributes
    ----------
    events : List[ManagementEvent]
        Priority queue of scheduled events
    resource_quotas : Dict[str, ResourceQuota]
        Available resource quotas
    conflict_strategy : ConflictResolution
        Default conflict resolution strategy
    execution_log : List[ManagementEvent]
        History of executed events
    conflict_log : List[EventConflict]
        History of resolved conflicts
    """
    
    def __init__(self,
                 conflict_strategy: ConflictResolution = ConflictResolution.PRIORITY_BASED,
                 enable_logging: bool = True):
        """
        Initialize the event scheduler.
        
        Parameters
        ----------
        conflict_strategy : ConflictResolution
            Default strategy for conflict resolution
        enable_logging : bool
            Whether to maintain execution and conflict logs
        """
        self.events: List[ManagementEvent] = []
        self.resource_quotas: Dict[str, ResourceQuota] = {}
        self.conflict_strategy = conflict_strategy
        self.enable_logging = enable_logging
        
        # Logs
        self.execution_log: List[ManagementEvent] = []
        self.conflict_log: List[EventConflict] = []
        
        # Event handlers registry
        self.event_handlers: Dict[EventType, Callable] = {}
        
        # Temporal constraints
        self.minimum_intervals: Dict[Tuple[EventType, EventType], float] = {}
        self.blackout_periods: List[
            Tuple[float, float, Optional[EventType]]
        ] = []
        
        # Initialize as min-heap for priority queue
        heapq.heapify(self.events)
    
    def register_resource_quota(self, quota: ResourceQuota) -> None:
        """
        Register a resource quota.
        
        Parameters
        ----------
        quota : ResourceQuota
            Resource quota to register
        """
        self.resource_quotas[quota.resource_type] = quota
    
    def register_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Register a handler function for an event type.
        
        Parameters
        ----------
        event_type : EventType
            Type of event
        handler : Callable
            Function to handle the event
        """
        self.event_handlers[event_type] = handler
    
    def set_minimum_interval(self, 
                            event_type1: EventType, 
                            event_type2: EventType,
                            min_interval: float) -> None:
        """
        Set minimum time interval between two event types.
        
        Parameters
        ----------
        event_type1 : EventType
            First event type
        event_type2 : EventType
            Second event type
        min_interval : float
            Minimum time interval required between events
        """
        key = (event_type1, event_type2)
        self.minimum_intervals[key] = min_interval
        # Also set reverse
        if event_type1 != event_type2:
            self.minimum_intervals[(event_type2, event_type1)] = min_interval
    
    def add_blackout_period(self,
                           start_time: float,
                           end_time: float,
                           event_type: Optional[EventType] = None) -> None:
        """
        Add a blackout period when certain events cannot occur.
        
        Parameters
        ----------
        start_time : float
            Start of blackout period
        end_time : float
            End of blackout period
        event_type : EventType, optional
            Specific event type to block (None = all events)
        """
        self.blackout_periods.append((start_time, end_time, event_type))
    
    def schedule_event(self, event: ManagementEvent) -> bool:
        """
        Schedule a management event.
        
        Parameters
        ----------
        event : ManagementEvent
            Event to schedule
            
        Returns
        -------
        bool
            True if event scheduled successfully, False otherwise
        """
        known_events = [*self.events, *self.execution_log]
        if any(existing.event_id == event.event_id for existing in known_events):
            warnings.warn(f"Duplicate event_id rejected: {event.event_id}")
            return False

        # Check blackout periods
        if self._is_in_blackout(event):
            warnings.warn(f"Event {event.event_id} falls in blackout period")
            return False
        
        # Check for conflicts
        conflicts = self._find_conflicts(event)
        
        if conflicts:
            # Resolve conflicts
            for conflict_event in conflicts:
                resolution = self._resolve_conflict(event, conflict_event)
                
                if resolution == ConflictResolution.REJECT:
                    warnings.warn(f"Event {event.event_id} rejected due to conflict")
                    return False
                elif resolution == ConflictResolution.MERGE:
                    # Merge events
                    self._merge_events(event, conflict_event)
                    return True
                elif resolution == ConflictResolution.SEQUENTIAL:
                    # Adjust timing to make sequential
                    event.scheduled_time = conflict_event.scheduled_time + conflict_event.duration
        
        # Check resource availability
        if not self._check_resource_availability(event):
            warnings.warn(f"Insufficient resources for event {event.event_id}")
            return False
        
        # Check minimum intervals
        if not self._check_minimum_intervals(event):
            warnings.warn(f"Event {event.event_id} violates minimum interval constraints")
            return False
        
        # Add to queue
        heapq.heappush(self.events, event)
        
        return True
    
    def execute_until(self, current_time: float, state: Dict[str, Any]) -> List[ManagementEvent]:
        """
        Execute all events scheduled up to current time.
        
        Parameters
        ----------
        current_time : float
            Current simulation time
        state : Dict[str, Any]
            Current system state
            
        Returns
        -------
        List[ManagementEvent]
            List of executed events
        """
        executed = []
        
        while self.events and self.events[0].scheduled_time <= current_time:
            event = heapq.heappop(self.events)
            
            if event.executed:
                continue  # Skip already executed (idempotent)
            
            # Allocate resources
            if self._allocate_resources(event):
                # Execute event
                try:
                    if event.handler:
                        result = event.handler(state, event.parameters)
                    elif event.event_type in self.event_handlers:
                        result = self.event_handlers[event.event_type](state, event.parameters)
                    else:
                        result = self._default_handler(event, state)
                    
                    event.execution_result = result
                    event.executed = True
                    executed.append(event)
                    
                    if self.enable_logging:
                        self.execution_log.append(event)
                    
                except Exception:  # noqa: BLE001 - simulator transaction owns rollback
                    self._release_resources(event)
                    heapq.heappush(self.events, event)
                    raise
            else:
                # Reschedule if resources not available
                event.scheduled_time = current_time + SECONDS_PER_HOUR  # Retry in 1 hour
                heapq.heappush(self.events, event)
                warnings.warn(f"Rescheduling event {event.event_id} due to resource constraints")
        
        return executed
    
    def cancel_event(self, event_id: str) -> bool:
        """
        Cancel a scheduled event.
        
        Parameters
        ----------
        event_id : str
            ID of event to cancel
            
        Returns
        -------
        bool
            True if event cancelled, False if not found
        """
        for i, event in enumerate(self.events):
            if event.event_id == event_id:
                self.events.pop(i)
                heapq.heapify(self.events)  # Restore heap property
                return True
        return False
    
    def reschedule_event(self, event_id: str, new_time: float) -> bool:
        """
        Reschedule an event to a new time.
        
        Parameters
        ----------
        event_id : str
            ID of event to reschedule
        new_time : float
            New scheduled time
            
        Returns
        -------
        bool
            True if rescheduled, False if not found
        """
        for event in self.events:
            if event.event_id == event_id:
                event.scheduled_time = new_time
                heapq.heapify(self.events)  # Restore heap property
                return True
        return False
    
    def get_upcoming_events(self, 
                           time_window: float,
                           current_time: float,
                           event_type: Optional[EventType] = None) -> List[ManagementEvent]:
        """
        Get events scheduled within time window.
        
        Parameters
        ----------
        time_window : float
            Look-ahead time window
        current_time : float
            Current simulation time
        event_type : EventType, optional
            Filter by event type
            
        Returns
        -------
        List[ManagementEvent]
            List of upcoming events
        """
        upcoming = []
        end_time = current_time + time_window
        
        for event in self.events:
            if event.scheduled_time > end_time:
                continue  # Beyond window
            if event.scheduled_time >= current_time:
                if event_type is None or event.event_type == event_type:
                    upcoming.append(event)
        
        return sorted(
            upcoming,
            key=lambda event: (
                event.scheduled_time,
                event.priority,
                event.event_id,
            ),
        )
    
    def reset_quotas(self, period: str) -> None:
        """
        Reset resource quotas for given period.
        
        Parameters
        ----------
        period : str
            Reset period (e.g., "daily", "weekly")
        """
        for quota in self.resource_quotas.values():
            if quota.reset_period == period:
                quota.reset()
    
    def _is_in_blackout(self, event: ManagementEvent) -> bool:
        """Check if event falls in blackout period."""
        for start, end, blocked_type in self.blackout_periods:
            if start <= event.scheduled_time <= end:
                if blocked_type is None or blocked_type == event.event_type:
                    return True
        return False
    
    def _find_conflicts(self, new_event: ManagementEvent) -> List[ManagementEvent]:
        """Find events that conflict with new event."""
        conflicts = []
        
        for event in self.events:
            # Check temporal overlap
            if self._events_overlap(new_event, event):
                # Check spatial overlap if applicable
                if self._spatial_conflict(new_event, event):
                    conflicts.append(event)
            
            # Check resource conflicts
            if self._resource_conflict(new_event, event):
                conflicts.append(event)
        
        return conflicts
    
    def _events_overlap(self, event1: ManagementEvent, event2: ManagementEvent) -> bool:
        """Check if two events overlap in time."""
        start1 = event1.scheduled_time
        end1 = start1 + event1.duration
        start2 = event2.scheduled_time
        end2 = start2 + event2.duration
        
        return not (end1 <= start2 or end2 <= start1)
    
    def _spatial_conflict(self, event1: ManagementEvent, event2: ManagementEvent) -> bool:
        """Check if two events have spatial conflict."""
        if event1.spatial_extent is None or event2.spatial_extent is None:
            return False  # No spatial constraints
        
        # Simple overlap check (can be extended for complex geometries)
        # Baseline overlap check; full implementation depends on
        # spatial representation (e.g., field IDs, polygons, etc.)
        return event1.spatial_extent == event2.spatial_extent
    
    def _resource_conflict(self, event1: ManagementEvent, event2: ManagementEvent) -> bool:
        """Check if two events have resource conflicts."""
        for resource in event1.resource_requirements:
            if resource in event2.resource_requirements:
                total_needed = (event1.resource_requirements[resource] + 
                              event2.resource_requirements[resource])
                
                if resource in self.resource_quotas:
                    if total_needed > self.resource_quotas[resource].total_available:
                        return True
        
        return False
    
    def _resolve_conflict(self, 
                         new_event: ManagementEvent, 
                         existing_event: ManagementEvent) -> ConflictResolution:
        """Resolve conflict between two events."""
        strategy = self.conflict_strategy
        
        if strategy == ConflictResolution.PRIORITY_BASED:
            if new_event.priority < existing_event.priority:
                # New event has higher priority (lower number)
                self.cancel_event(existing_event.event_id)
                return ConflictResolution.PRIORITY_BASED
            else:
                return ConflictResolution.REJECT
        
        elif strategy == ConflictResolution.FIRST_COME:
            return ConflictResolution.REJECT  # Existing event wins
        
        elif strategy == ConflictResolution.LAST_COME:
            self.cancel_event(existing_event.event_id)
            return ConflictResolution.LAST_COME
        
        elif strategy == ConflictResolution.MERGE:
            if self._can_merge(new_event, existing_event):
                return ConflictResolution.MERGE
            return ConflictResolution.REJECT
        
        elif strategy == ConflictResolution.SEQUENTIAL:
            if self._can_sequence(new_event, existing_event):
                return ConflictResolution.SEQUENTIAL
            return ConflictResolution.REJECT
        
        return ConflictResolution.REJECT
    
    def _can_merge(self, event1: ManagementEvent, event2: ManagementEvent) -> bool:
        """Check if two events can be merged."""
        # Same type and close in time
        return (event1.event_type == event2.event_type and 
                abs(event1.scheduled_time - event2.scheduled_time) < SECONDS_PER_HOUR)  # Within 1 hour
    
    def _merge_events(self, event1: ManagementEvent, event2: ManagementEvent) -> None:
        """Merge two consistent events."""
        # Combine resource requirements
        for resource, amount in event1.resource_requirements.items():
            if resource in event2.resource_requirements:
                event2.resource_requirements[resource] += amount
            else:
                event2.resource_requirements[resource] = amount
        
        # Merge parameters (event2 takes precedence)
        event2.parameters = {**event1.parameters, **event2.parameters}
        
        # Use higher priority
        event2.priority = min(event1.priority, event2.priority)
        
        # Log the merge
        if self.enable_logging:
            conflict = EventConflict(
                event1, event2, "resource_overlap", 
                ConflictResolution.MERGE,
                event2.scheduled_time,
                f"Merged {event1.event_id} into {event2.event_id}"
            )
            self.conflict_log.append(conflict)
    
    def _can_sequence(self, event1: ManagementEvent, event2: ManagementEvent) -> bool:
        """Check if events can be sequenced."""
        # Check if there's enough time to sequence
        min_interval = self.minimum_intervals.get(
            (event1.event_type, event2.event_type), 0
        )
        
        return (event1.duration + min_interval) < SECONDS_PER_DAY  # Less than a day total
    
    def _check_minimum_intervals(self, new_event: ManagementEvent) -> bool:
        """Check if event respects minimum interval constraints."""
        for event in self.events:
            if event.executed:
                continue
                
            key = (new_event.event_type, event.event_type)
            if key in self.minimum_intervals:
                min_interval = self.minimum_intervals[key]
                time_diff = abs(new_event.scheduled_time - event.scheduled_time)
                
                if time_diff < min_interval:
                    return False
        
        return True
    
    def _check_resource_availability(self, event: ManagementEvent) -> bool:
        """Check if resources are available for event."""
        for resource, amount in event.resource_requirements.items():
            if resource in self.resource_quotas:
                if not self.resource_quotas[resource].can_allocate(amount):
                    return False
        
        return True
    
    def _allocate_resources(self, event: ManagementEvent) -> bool:
        """Allocate resources for event execution."""
        # First check all resources are available
        if not self._check_resource_availability(event):
            return False
        
        # Allocate all resources
        allocated = []
        for resource, amount in event.resource_requirements.items():
            if resource in self.resource_quotas:
                if self.resource_quotas[resource].allocate(amount):
                    allocated.append((resource, amount))
                else:
                    # Rollback on failure
                    for alloc_resource, alloc_amount in allocated:
                        self.resource_quotas[alloc_resource].release(alloc_amount)
                    return False
        
        return True
    
    def _release_resources(self, event: ManagementEvent) -> None:
        """Release resources after event completion."""
        for resource, amount in event.resource_requirements.items():
            if resource in self.resource_quotas:
                self.resource_quotas[resource].release(amount)
    
    def _default_handler(self, event: ManagementEvent, state: Dict[str, Any]) -> Dict[str, Any]:
        """Default event handler."""
        # Basic implementation - just log the event
        return {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'executed_at': event.scheduled_time,
            'parameters': event.parameters,
            'status': 'completed'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            'scheduled_events': len(self.events),
            'executed_events': len(self.execution_log),
            'conflicts_resolved': len(self.conflict_log),
            'resource_utilization': {}
        }
        
        # Resource utilization
        for resource_type, quota in self.resource_quotas.items():
            stats['resource_utilization'][resource_type] = {
                'used': quota.current_used,
                'total': quota.total_available,
                'utilization_percent': (quota.current_used / quota.total_available * 100 
                                       if quota.total_available > 0 else 0)
            }
        
        # Event type distribution
        type_counts = {}
        for event in self.events:
            type_counts[event.event_type.value] = type_counts.get(event.event_type.value, 0) + 1
        stats['event_type_distribution'] = type_counts
        
        return stats


# Export main classes
__all__ = [
    'EventScheduler',
    'ManagementEvent',
    'ResourceQuota',
    'EventType',
    'EventPriority',
    'ConflictResolution',
    'EventConflict',
    'resolve_day_of_year',
]

