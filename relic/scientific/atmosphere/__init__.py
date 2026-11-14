"""
Atmospheric Processes Module for RELIC

This module provides comprehensive atmospheric process functors for ecosystem modeling,
implementing radiation transfer, turbulence dynamics, and atmospheric chemistry.
All functors follow the unified RELIC architecture with consistent interfaces,
CF-1.8 compliant variable naming, and seamless coupling capabilities.

Main Components:
    RadiationFunctor: Shortwave/longwave radiation balance with cloud effects
    TurbulenceFunctor: MOST-based surface layer turbulence and aerodynamic resistance  
    ChemistryFunctor: Simplified atmospheric chemistry with NOx-O3 cycle and BVOC interactions

Key Features:
    - Unified FunctorBase inheritance for consistent interfaces
    - Temperature/light response functions from unified response library
    - CF-1.8 compliant variable names and units
    - Comprehensive scientific documentation with mathematical expressions
    - Energy and mass conservation validation
    - Coupling interfaces for ecosystem integration

Usage Example:
    ```python
    from relic.scientific.atmosphere import RadiationFunctor, TurbulenceFunctor
    
    # Initialize functors
    radiation = RadiationFunctor()
    turbulence = TurbulenceFunctor()
    
    # Compute radiation and turbulence in sequence
    rad_fluxes = radiation(state, time, dt)
    turb_params = turbulence(state, time, dt)
    
    # Use outputs for evapotranspiration coupling
    net_radiation = rad_fluxes['net_radiation']
    aerodynamic_resistance = turb_params['aerodynamic_resistance']
    ```

Scientific Validation:
    - Energy conservation: SW_in = SW_absorbed + SW_reflected
    - MOST validation: u* scaling with wind speed and stability
    - Chemical conservation: mass balance in reaction networks
    - Physical bounds: 0 <= albedo <= 1, u* >= 0, reasonable stability ranges

References:
    - Allen, R.G. et al. (1998). FAO-56: Crop evapotranspiration guidelines
    - Garratt, J.R. (1992). The Atmospheric Boundary Layer
    - Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics
    - Monin, A.S. & Obukhov, A.M. (1954). Basic laws of turbulent mixing
"""

from .radiation import RadiationFunctor
from .turbulence import TurbulenceFunctor  
from .chemistry import ChemistryFunctor
from .environmental_forcing import EnvironmentalForcingFunctor
from .co2_unit_conversion import mixing_ratio_umol_mol_to_mass_conc_kg_m3

# Module metadata
__version__ = "1.0.0"
__author__ = "Yuchong Long"
__email__ = "relic@example.org"

# Export main functor classes
__all__ = [
    'RadiationFunctor',
    'TurbulenceFunctor', 
    'ChemistryFunctor',
    'EnvironmentalForcingFunctor',
    'mixing_ratio_umol_mol_to_mass_conc_kg_m3',
]

# Module-level constants
ATMOSPHERE_MODULE_INFO = {
    'description': 'Atmospheric processes for ecosystem modeling',
    'version': __version__,
    'components': ['radiation', 'turbulence', 'chemistry'],
    'dependencies': ['core', 'utils'],
    'coupling_domains': ['biosphere', 'hydrosphere', 'pedosphere'],
    'scientific_domains': ['micrometeorology', 'atmospheric_physics', 'atmospheric_chemistry'],
    'standards_compliance': ['CF-1.8', 'FAIR_data_principles', 'PyTorch_integration']
}
