"""
Anthropogenic Management Module

This module provides comprehensive management of anthropogenic activities including
agricultural practices, land management, and resource utilization in the RELIC
modeling framework.

Core Management Components:
    - FertilizationManagement: Nutrient application and fertilizer management
    - IrrigationManagement: Water resource allocation and irrigation scheduling
    - TillageManagement: Soil tillage operations and disturbance management
    - HarvestManagement: Crop harvesting and residue management
    - GrazingManagement: Livestock grazing and pasture management
    - FireManagement: Prescribed burning and fire suppression activities

Coordination Components:
    - ManagementCoordinator: Integrated management strategy coordination
    - EventScheduler: Temporal scheduling and conflict resolution

All management classes inherit from appropriate RELIC base classes (BiologicalProcessFunctor,
ScaleCouplingFunctor) and integrate with the unified parameter management system.
"""

# Core management classes
from .fertilization import (
    FertilizationManagement,
    FertilizerType,
    ApplicationTiming,
    ApplicationMethod,
    FertilizerEvent,
)

from .irrigation import (
    IrrigationManagement,
)

from .tillage import (
    TillageManagement,
    TillageDepth,
    TillageIntensity,
    TillageEvent,
)

from .harvest import (
    HarvestManagement,
    CropType,
    HarvestEvent,
)

from .grazing import (
    GrazingManagement,
    LivestockType,
    GrazingSystem,
    GrazingIntensity,
    GrazingEvent,
)

from .fire_management import (
    FireManagement,
    FireType,
    FireIntensity,
    VegetationType,
    FireEvent,
)

# Coordination and scheduling
from .coordinator import (
    ManagementCoordinator,
    ManagementStrategy,
    CoordinationMode,
    ManagementObjective,
    CoordinationResult,
)

from .event_scheduler import (
    EventScheduler,
    EventPriority,
    EventType,
    ConflictResolution,
    ManagementEvent,
    ResourceQuota,
    EventConflict,
)

# Export public interface
__all__ = [
    # Core management classes
    "FertilizationManagement",
    "IrrigationManagement",
    "TillageManagement",
    "HarvestManagement",
    "GrazingManagement",
    "FireManagement",
    
    # Coordination and scheduling
    "ManagementCoordinator",
    "EventScheduler",
    
    # Management events and data structures
    "ManagementEvent",
    "FertilizerEvent",
    "HarvestEvent",
    "GrazingEvent",
    "FireEvent",
    "TillageEvent",
    
    # Management enums
    "FertilizerType",
    "ApplicationTiming",
    "ApplicationMethod",
    "TillageDepth",
    "TillageIntensity",
    "CropType",
    "LivestockType",
    "GrazingSystem",
    "GrazingIntensity",
    "FireType",
    "FireIntensity",
    "VegetationType",
    
    # Coordination enums and data structures
    "ManagementStrategy",
    "CoordinationMode",
    "EventPriority",
    "EventType",
    "ConflictResolution",
    "ManagementObjective",
    "CoordinationResult",
    "ResourceQuota",
    "EventConflict",
]
