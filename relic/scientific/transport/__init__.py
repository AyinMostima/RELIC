"""
Transport Processes Module for RELIC

This module provides comprehensive transport mechanisms for scalar quantities
including molecular diffusion, advective transport, and combined advection-diffusion
processes. All transport mechanisms support 1D, 2D, and 3D spatial discretizations
with unified interfaces and conservation enforcement.

Key Components:
    - operators: Fundamental differential operators and grid utilities
    - boundary_conditions: Unified boundary condition implementations
    - diffusion: Fickian diffusion with explicit/semi-implicit schemes
    - advection: Linear advection with upwind and higher-order schemes
    - advection_diffusion: Combined transport with operator splitting

Mathematical Framework:
    General transport equation: partialc/partialt + nabla*(vc) = nabla*(Knablac) + S
    
    Where:
    - c: concentration [kg m-3]
    - v: velocity field [m s-1]
    - K: diffusivity tensor [m^2 s-1]
    - S: source/sink term [kg m-3 s-1]

References:
    - Bird, R.B. et al. (2007). Transport Phenomena. Wiley
    - Crank, J. (1975). The Mathematics of Diffusion. Oxford
    - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs
    - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of Time-Dependent 
      Advection-Diffusion-Reaction Equations
"""

from relic.computation.operators.functional import (
    compute_gradient_1d, compute_gradient_2d, compute_gradient_3d,
    compute_divergence_2d, compute_divergence_3d,
    compute_conservative_divergence_2d, compute_conservative_divergence_3d,
    compute_laplacian_1d, compute_laplacian_2d, compute_laplacian_3d,
    compute_grid_metrics, compute_cfl_number, compute_diffusion_stability,
    check_conservation, apply_non_negative_constraint
)

from .boundary_conditions import (
    BoundaryType, BoundaryCondition, BoundaryConditionManager,
    create_dirichlet_walls, create_periodic_boundaries, create_open_boundaries
)
from .advection import LinearAdvection, create_uniform_advection
from .diffusion import FickianDiffusion
from .advection_diffusion import (
    AdvectionDiffusion,
    create_wind_transport,
    create_groundwater_transport,
)

# Export all transport classes and utilities
__all__ = [
    # Operators and utilities
    'compute_gradient_1d', 'compute_gradient_2d', 'compute_gradient_3d',
    'compute_divergence_2d', 'compute_divergence_3d',
    'compute_conservative_divergence_2d', 'compute_conservative_divergence_3d',
    'compute_laplacian_1d', 'compute_laplacian_2d', 'compute_laplacian_3d',
    'compute_grid_metrics', 'compute_cfl_number', 'compute_diffusion_stability',
    'check_conservation', 'apply_non_negative_constraint',
    
    # Boundary conditions
    'BoundaryType', 'BoundaryCondition', 'BoundaryConditionManager',
    'create_dirichlet_walls', 'create_periodic_boundaries', 'create_open_boundaries',
    
    # Transport classes
    'LinearAdvection', 'FickianDiffusion', 'AdvectionDiffusion',
    'create_uniform_advection',
    'create_wind_transport', 'create_groundwater_transport'
]

# Version info
__version__ = "1.1.0"  # Updated for reorganization
__author__ = "Yuchong Long"

# No consistency aliases or lazy attribute accessors.

