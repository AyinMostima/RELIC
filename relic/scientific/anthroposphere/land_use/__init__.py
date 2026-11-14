"""
Land Use Module - Anthroposphere Component

This module provides land use transition dynamics and parameter mapping
for the RELIC ecosystem model.
"""

from .transitions import (
    LandUseTransitions,
    LandUseType,
    TransitionType,
    TransitionEvent
)

from .mapping import (LandUseParameterMapping,
    VegetationParameters,
    SoilParameters,
    HydrologicalParameters
)

# Parameter management imports
__all__ = [
    # Transitions
    "LandUseTransitions",
    "LandUseType",
    "TransitionType",
    "TransitionEvent",
    # Mapping
    "LandUseParameterMapping",
    "VegetationParameters",
    "SoilParameters",
    "HydrologicalParameters"
]
