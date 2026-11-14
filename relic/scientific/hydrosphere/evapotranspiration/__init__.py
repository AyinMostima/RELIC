"""
Evapotranspiration Module for RELIC

This module provides evapotranspiration models for the hydrosphere
component of the RELIC framework. It implements multiple ET 
formulations with unified interfaces and CF-1.8 consistent units.

Available Models:
- PenmanMonteithET: Full Penman-Monteith equation with surface resistance
- PriestleyTaylorET: Simplified equilibrium ET model

All models follow the FunctorBase interface and integrate with the
unified response function library.
"""

from .penman_monteith import PenmanMonteithET
from .priestley_taylor import PriestleyTaylorET


__all__ = [
    "PenmanMonteithET",
    "PriestleyTaylorET"
]

