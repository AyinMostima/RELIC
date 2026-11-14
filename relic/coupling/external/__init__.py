"""
External Model Coupling Interface

This package provides unified interfaces for coupling RELIC with external 
models from different domains:

- Climate Models: CMIP6 data archives
- Atmospheric Models: WRF, CESM atmospheric components  
- Hydrological Models: VIC, SWAT watershed models

All adapters follow consistent patterns for data discovery, loading,
transformation, and validation while respecting model-specific formats
and conventions.

Key Features:
- Unified adapter interface via ExternalAdapter base class
- CF-1.8 compliant data handling
- PyTorch tensor integration with automatic differentiation
- Comprehensive validation and quality control
- Flexible spatial and temporal subsetting
"""

# Import all adapter modules
from . import common
from . import climate_models
from . import atmospheric_models
from . import hydrological_models

# Import key classes for convenience
from .common import (
    ExternalAdapter,
    DataSource,
    TimeWindow,
    SpatialWindow,
    ValidationReport
)

from .climate_models import (
    CMIP6Adapter,
    CMIP6Config
)

from .atmospheric_models import (
    WRFAdapter,
    WRFConfig,
    CESMAdapter,
    CESMConfig
)

from .hydrological_models import (VICAdapter,
    VICConfig,
    SWATAdapter,
    SWATConfig
)

__all__ = [
    # Submodules
    "common",
    "climate_models", 
    "atmospheric_models",
    "hydrological_models",
    
    # Base classes and enums
    "ExternalAdapter",
    "DataSource",
    "TimeWindow",
    "SpatialWindow",
    "ValidationReport",
    
    # Climate model adapters
    "CMIP6Adapter",
    "CMIP6Config",
    
    # Atmospheric model adapters
    "WRFAdapter",
    "WRFConfig",
    "CESMAdapter", 
    "CESMConfig",
    
    # Hydrological model adapters
    "VICAdapter",
    "VICConfig",
    "SWATAdapter",
    "SWATConfig"
]
