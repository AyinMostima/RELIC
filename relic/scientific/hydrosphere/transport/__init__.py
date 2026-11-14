"""
Hydrosphere Transport Processes

This module implements transport mechanisms specific to water systems,
including vertical percolation, surface flow, subsurface flow, and solute transport in water.

Transport Mechanisms:
    - VerticalPercolationTransport: Gravitational drainage using Darcy-Buckingham equation

Mathematical Foundations:
    Darcy-Buckingham equation: q = -D(theta)nablatheta + K(theta)nablaH
    van Genuchten-Mualem functions: K(Se) = Ks * Se^l * [1-(1-Se^(1/m))^m]^2
    Mass conservation: partialtheta/partialt + nabla*q = S

References:
    - Darcy, H. (1856). Les Fontaines publiques de la ville de Dijon
    - van Genuchten, M.T. (1980). Closed-form equation for hydraulic conductivity. Soil Sci. Soc. Am. J.
    - Bear, J. (1972). Dynamics of Fluids in Porous Media. American Elsevier Publishing Company
    - Freeze, R.A. & Cherry, J.A. (1979). Groundwater. Prentice-Hall
"""


# Import hydrosphere-specific transport implementations
# Primary vertical percolation transport
from .vertical_percolation import (
    VerticalPercolationTransport,
    HydraulicModel
)

# Capillary rise transport mechanism
from .capillary_rise import CapillaryRiseTransport

__all__ = [
    # Primary hydrosphere transport mechanisms
    'VerticalPercolationTransport',
    'HydraulicModel',
    'CapillaryRiseTransport'
]
