"""
Spatial Scale Coupling Module

Provides spatial regridding functionality for the RELIC framework.
"""

from .grid_utils import (
    GridDescriptor,
    GridUtils,
    compute_cell_areas,
    spherical_area,
    cartesian_area
)

from .regrid_bilinear import BilinearRegridder
from .regrid_conservative import ConservativeRegridder
from .regrid_nearest import NearestNeighborRegridder
from .factory import SpatialRegridderFactory, AutoRegridder

__all__ = [
    # Grid utilities
    'GridDescriptor',
    'GridUtils',
    'compute_cell_areas',
    'spherical_area',
    'cartesian_area',
    
    # Regridders
    'BilinearRegridder',
    'ConservativeRegridder',
    'NearestNeighborRegridder',
    'AutoRegridder',
    'SpatialRegridderFactory',
]
