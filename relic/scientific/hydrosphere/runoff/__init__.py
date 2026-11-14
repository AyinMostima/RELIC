"""
Runoff Module for RELIC

This module provides runoff generation models for the hydrosphere
component of the RELIC framework. It implements surface and
subsurface runoff mechanisms with unified interfaces.

Available Models:
- SurfaceRunoff: Surface runoff using SCS, Hortonian, and saturation excess methods
- SubsurfaceRunoff: Subsurface runoff using TOPMODEL, exponential recession, and interflow

All models follow the FunctorBase interface and provide
mass-conservative water routing calculations.
"""

from .surface import SurfaceRunoff
from .subsurface import SubsurfaceRunoff
from .surface_ponding import HortonianPondingEventFunctor, SurfacePondingFunctor

# Parameter management imports
__all__ = [
    "SurfaceRunoff",
    "SubsurfaceRunoff",
    "HortonianPondingEventFunctor",
    "SurfacePondingFunctor",
]
