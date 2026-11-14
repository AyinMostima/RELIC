"""
Monin-Obukhov Similarity Theory (MOST) Standard Library

This package provides a comprehensive implementation of MOST functions
for atmospheric boundary layer calculations, eliminating code duplication
across the RELIC framework.

Modules:
    - similarity_functions: Stability functions (psi_m, psi_h, phi_m, phi_h)
    - stability_parameters: Core parameters (u*, L, zeta)
    - surface_fluxes: Flux calculations and aerodynamic resistance
    - mixing_length: Mixing length formulations

References:
    - Monin & Obukhov (1954)
    - Businger et al. (1971)
    - Garratt (1992)
    - Stull (1988)

Parameter Access Convention:
    - This package is a non-Functor utility library, so it may use
      ``get_global_manager().get(...)`` for scalar coefficients.
    - Scientific Functor subclasses must use ``self.get_parameter_value()``
      (or ``self.get_parameter_value(required=True)``) to preserve override resolution.
"""

# Import all functions from submodules for convenient access
from .similarity_functions import (
    psi_m,
    psi_h,
    phi_m,
    phi_h
)

from .stability_parameters import (
    friction_velocity,
    monin_obukhov_length,
    zeta,
    richardson_number,
    zeta_from_richardson,
    temperature_scale,
    iterative_most_solution,
    VON_KARMAN_CONSTANT,
    GRAVITY_ACCELERATION,
    SPECIFIC_HEAT_AIR
)

from .surface_fluxes import (
    aerodynamic_resistance,
    aerodynamic_resistance_momentum,
    eddy_diffusivity,
    momentum_flux,
    sensible_heat_flux,
    latent_heat_flux,
    scalar_flux,
    bulk_transfer_coefficients,
    roughness_sublayer_correction
)

from .mixing_length import (prandtl_mixing_length,
    blackadar_mixing_length,
    stability_corrected_mixing_length,
    k_theory_K,
    tke_mixing_length,
    canopy_mixing_length,
    local_mixing_length
)

# Define what gets imported with "from most import *"
__all__ = [
    # Similarity functions
    'psi_m',
    'psi_h',
    'phi_m',
    'phi_h',
    
    # Stability parameters
    'friction_velocity',
    'monin_obukhov_length',
    'zeta',
    'richardson_number',
    'zeta_from_richardson',
    'temperature_scale',
    'iterative_most_solution',
    
    # Surface fluxes
    'aerodynamic_resistance',
    'aerodynamic_resistance_momentum',
    'eddy_diffusivity',
    'momentum_flux',
    'sensible_heat_flux',
    'latent_heat_flux',
    'scalar_flux',
    'bulk_transfer_coefficients',
    'roughness_sublayer_correction',
    
    # Mixing length
    'prandtl_mixing_length',
    'blackadar_mixing_length',
    'stability_corrected_mixing_length',
    'k_theory_K',
    'tke_mixing_length',
    'canopy_mixing_length',
    'local_mixing_length',
    
    # Constants
    'VON_KARMAN_CONSTANT',
    'GRAVITY_ACCELERATION',
    'SPECIFIC_HEAT_AIR'
]

# Version information
__version__ = '1.0.0'
__author__ = 'Yuchong Long'

