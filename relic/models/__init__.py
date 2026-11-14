"""
Hierarchical Scale Model Definitions

Provides functor and variable mappings for multi-scale ecosystem modeling.
Defines four hierarchical scales: individual, patch, landscape, regional.

Scale Hierarchy:
    Individual (cm-m) -> Patch (10m-1km) -> Landscape (1-100km) -> Regional (100-1000km+)

Usage:
    from . import individual, patch, landscape, regional
    
    # Get functors for specific scale
    individual_processes = individual.LAYER_FUNCTORS
    
    # Get variables exposed at patch scale  
    patch_vars = patch.EXPOSED_VARIABLES

References:
    - Wu, J. (1999). Hierarchy and scaling: extrapolating information along a scaling ladder
    - Peters, D.P. et al. (2007). Living in an increasingly connected world
    - Urban, D.L. et al. (1987). Landscape ecology: A hierarchical perspective
"""

from . import individual
from . import patch
from . import hillslope
from . import landscape
from . import basin
from . import regional

# Import surrogate models
from .surrogates import (
    SurrogateFunctor, 
    ONNXSurrogate,
    SurrogateError,
    ModelLoadError, 
    InferenceError, 
    ConservationViolationError
)
from ..core.constants import get_scale_hierarchy

from .pytorch_surrogate import PyTorchSurrogate, SurrogateDataset

# Scale hierarchy definition (parameter-system driven)
SCALE_HIERARCHY = get_scale_hierarchy()


def _build_transition_maps(hierarchy):
    up = {}
    down = {}
    for idx, scale in enumerate(hierarchy):
        up[scale] = hierarchy[idx + 1] if idx + 1 < len(hierarchy) else None
        down[scale] = hierarchy[idx - 1] if idx - 1 >= 0 else None
    return up, down


UPSCALING_MAP, DOWNSCALING_MAP = _build_transition_maps(SCALE_HIERARCHY)

# Module registry (filtered by available hierarchy)
_ALL_SCALE_MODULES = {
    "individual": individual,
    "patch": patch,
    "hillslope": hillslope,
    "basin": basin,
    "landscape": landscape,
    "regional": regional
}
SCALE_MODULES = {k: v for k, v in _ALL_SCALE_MODULES.items() if k in SCALE_HIERARCHY}

def get_scale_functors(scale_name: str):
    """Get layer functors for specified scale."""
    if scale_name not in SCALE_MODULES:
        raise ValueError(f"Unknown scale: {scale_name}")
    return SCALE_MODULES[scale_name].LAYER_FUNCTORS

def get_scale_variables(scale_name: str):
    """Get exposed variables for specified scale.""" 
    if scale_name not in SCALE_MODULES:
        raise ValueError(f"Unknown scale: {scale_name}")
    return SCALE_MODULES[scale_name].EXPOSED_VARIABLES

def get_upscaling_target(scale_name: str):
    """Get target scale for upscaling."""
    return UPSCALING_MAP.get(scale_name)

def get_downscaling_source(scale_name: str):
    """Get source scale for downscaling."""
    return DOWNSCALING_MAP.get(scale_name)

__all__ = [
    # Scale hierarchy models
    "individual", "patch", "hillslope", "landscape", "basin", "regional",
    "SCALE_HIERARCHY", "UPSCALING_MAP", "DOWNSCALING_MAP", 
    "get_scale_functors", "get_scale_variables",
    "get_upscaling_target", "get_downscaling_source",
    
    # Surrogate models
    "SurrogateFunctor", "ONNXSurrogate",
    "SurrogateError", "ModelLoadError", "InferenceError", "ConservationViolationError",
    "PyTorchSurrogate", "SurrogateDataset",
]
