"""
Groundwater Dynamics Module for RELIC

This module provides groundwater storage dynamics, water table depth evolution,
and coupling interfaces for subsurface runoff and capillary rise processes.
"""

from .dynamics import GroundwaterDynamics

__all__ = ["GroundwaterDynamics"]
