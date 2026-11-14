"""
Atmospheric Model Adapters for External Model Coupling

This package provides adapters for atmospheric model outputs including:
- WRF: Weather Research and Forecasting Model
- CESM: Community Earth System Model atmospheric component
- Other atmospheric model formats as needed

All adapters provide unified access to atmospheric datasets following
CF conventions and RELIC internal standards.
"""

from .wrf_adapter import WRFAdapter, WRFConfig
from .cesm_adapter import CESMAdapter, CESMConfig

__all__ = [
    "WRFAdapter",
    "WRFConfig",
    "CESMAdapter", 
    "CESMConfig"
]
