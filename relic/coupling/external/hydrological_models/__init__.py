"""
Hydrological Model Adapters for External Model Coupling

This package provides adapters for hydrological model outputs including:
- VIC: Variable Infiltration Capacity Model
- SWAT: Soil and Water Assessment Tool
- Other hydrological model formats as needed

All adapters provide unified access to hydrological datasets following
CF conventions and RELIC internal standards.
"""

from .vic_adapter import VICAdapter, VICConfig
from .swat_adapter import SWATAdapter, SWATConfig

__all__ = [
    "VICAdapter",
    "VICConfig",
    "SWATAdapter",
    "SWATConfig"
]
