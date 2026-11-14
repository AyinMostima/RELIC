"""
Soil Moisture Module for RELIC

This module provides soil moisture models for the hydrosphere
component of the RELIC framework. It implements multiple
soil water dynamics formulations with unified interfaces.

Available Models:
- BucketSoilMoisture: Simple bucket model for water balance
- RichardsSoilMoisture: Richards equation framework (simplified implementation)
- SoilWaterRetention: Soil water potential of the effective soil layer (diagnostic)

All models follow the FunctorBase interface and provide
boundary condition setting capabilities.
"""

from .bucket import BucketSoilMoisture
from .richards import RichardsSoilMoisture
from .retention import SoilWaterRetention

# Parameter management imports
__all__ = [
    "BucketSoilMoisture",
    "RichardsSoilMoisture",
    "SoilWaterRetention"
]
