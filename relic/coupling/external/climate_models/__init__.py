"""
Climate Model Adapters for External Model Coupling

This package provides adapters for climate model outputs including:
- CMIP6: Coupled Model Intercomparison Project Phase 6
- Other climate model formats as needed

All adapters provide unified access to climate datasets following
CF conventions and RELIC internal standards.
"""

from .cmip6_adapter import CMIP6Adapter, CMIP6Config, CMIP6FileInfo

__all__ = [
    "CMIP6Adapter",
    "CMIP6Config", 
    "CMIP6FileInfo"
]
