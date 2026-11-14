"""
Anthroposphere Module

This module provides comprehensive modeling of anthropogenic activities and their impacts
on the Earth system, including land use management, emissions, and resource utilization.

Key Components:
    - Management: Agricultural and land management practices
    - Emissions: Anthropogenic emission inventories and atmospheric coupling
    - Land Use: Land use transitions and parameter mapping
    - Interface: CF convention mappings and variable standardization

The anthroposphere module integrates human activities with natural processes in the
RELIC modeling framework, enabling assessment of human-Earth system interactions.
"""

# Management imports
from .management import (
    # Core management classes
    FertilizationManagement,
    IrrigationManagement,
    TillageManagement,
    HarvestManagement,
    GrazingManagement,
    FireManagement,
    
    # Coordination and scheduling
    ManagementCoordinator,
    EventScheduler,
    
    # Management data structures
    ManagementEvent,
    FertilizerEvent,
    HarvestEvent,
    GrazingEvent,
    FireEvent,
    TillageEvent,
    
    # Management enums
    ManagementStrategy,
    CoordinationMode,
    EventPriority,
    EventType,
    ConflictResolution,
    
    # Management enums and data classes
    FertilizerType,
    ApplicationTiming,
    ApplicationMethod,
    TillageDepth,
    TillageIntensity,
    CropType,
    LivestockType,
    GrazingSystem,
    GrazingIntensity,
    FireType,
    FireIntensity,
    VegetationType,
    
    # Management data classes
    ManagementObjective,
    CoordinationResult,
    ResourceQuota,
    EventConflict,
)

# Emissions imports
from .emissions import (
    # Main emission classes
    AtmosphereCoupling,
    SpeciesMapper,

    # Chemistry mechanism taxonomy
    ChemicalMechanism,

    # Fail-closed inventory runtime boundary
    EmissionInventory,
    # SectorAllocation,
    # TemporalProfile,
    # SpatialAllocation,
)

# Land use imports
from .land_use import (
    # Land use dynamics
    LandUseTransitions,
    LandUseType,
    TransitionType,
    TransitionEvent,
    
    # Parameter mapping
    LandUseParameterMapping,
    VegetationParameters,
    SoilParameters,
    HydrologicalParameters,
)

# Interface and utility imports
from .interface_mapping import AnthroposphereInterfaceMapping

# Parameter management imports
# Export public interface
__all__ = [
    # Management module
    "FertilizationManagement",
    "IrrigationManagement", 
    "TillageManagement",
    "HarvestManagement",
    "GrazingManagement",
    "FireManagement",
    "ManagementCoordinator",
    "EventScheduler",
    "ManagementEvent",
    "FertilizerEvent",
    "HarvestEvent", 
    "GrazingEvent",
    "FireEvent",
    "TillageEvent",
    "ManagementStrategy",
    "CoordinationMode",
    "EventPriority",
    "EventType",
    "ConflictResolution",
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
    "ManagementObjective",
    "CoordinationResult",
    "ResourceQuota",
    "EventConflict",
    
    # Emissions module
    "AtmosphereCoupling",
    "SpeciesMapper",
    "ChemicalMechanism",
    "EmissionInventory",
    
    # Land use module
    "LandUseTransitions",
    "LandUseType",
    "TransitionType",
    "TransitionEvent",
    "LandUseParameterMapping",
    "VegetationParameters",
    "SoilParameters",
    "HydrologicalParameters",
    
    # Interface and utilities
    "AnthroposphereInterfaceMapping",
]
