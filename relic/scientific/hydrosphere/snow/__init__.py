"""
Snow Physics Module for RELIC

This module provides snow-related process models for the hydrosphere
including snow albedo dynamics, snowmelt processes, and snow-atmosphere
energy exchanges.

Available Models:
- SnowAlbedo: Dynamic snow surface albedo based on BATS and age-based schemes
- SnowPack: SWE mass balance with degree day melt and precipitation partitioning
"""

from .albedo import SnowAlbedo
from .snowpack import SnowPack

# Parameter management imports
__all__ = [
    "SnowAlbedo",
    "SnowPack",
]
