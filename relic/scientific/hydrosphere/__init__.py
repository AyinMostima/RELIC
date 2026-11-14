"""
Hydrosphere Module for RELIC

This module provides comprehensive hydrological process models
for the RELIC framework, including water cycle components
such as evapotranspiration, soil moisture dynamics, and runoff generation.

Available Submodules:
- evapotranspiration: Penman-Monteith and Priestley-Taylor ET models
- soil_moisture: Bucket and Richards equation soil water models  
- runoff: Surface and subsurface runoff generation models
- transport: Capillary rise and vertical percolation transport mechanisms
- groundwater: Groundwater dynamics and storage tracking models
- snow: Snow albedo dynamics and snow physics models

All models follow the FunctorBase interface and integrate with
the unified response function library for consistent behavior.
"""

# Import main classes from submodules
from .evapotranspiration import PenmanMonteithET, PriestleyTaylorET
from .soil_moisture import BucketSoilMoisture, RichardsSoilMoisture
from .runoff import SurfaceRunoff, SubsurfaceRunoff
from .transport import CapillaryRiseTransport, VerticalPercolationTransport
from .groundwater import GroundwaterDynamics
from .snow import SnowAlbedo, SnowPack

__all__ = [
    # Evapotranspiration models
    "PenmanMonteithET",
    "PriestleyTaylorET",
    # Soil moisture models
    "BucketSoilMoisture", 
    "RichardsSoilMoisture",
    # Runoff models
    "SurfaceRunoff",
    "SubsurfaceRunoff",
    # Transport mechanisms
    "CapillaryRiseTransport",
    "VerticalPercolationTransport",
    # Groundwater dynamics
    "GroundwaterDynamics",
    # Snow physics
    "SnowAlbedo",
    "SnowPack",
]
