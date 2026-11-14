"""
Atmospheric Transport Processes

This module implements transport mechanisms specific to the atmosphere,
including turbulent diffusion, advection by wind fields, and vertical mixing.

References
----------
.. [1] Stull, R. B. (1988). An Introduction to Boundary Layer Meteorology.
       Kluwer Academic Publishers.
.. [2] Garratt, J. R. (1992). The Atmospheric Boundary Layer. 
       Cambridge University Press.
"""

# Import atmosphere-specific implementations
from .AtmosphericAdvectionDiffusionFunctor import AtmosphericAdvectionDiffusionFunctor
AtmosphericAdvectionDiffusion = AtmosphericAdvectionDiffusionFunctor

from .eddy_diffusion_blm import EddyDiffusionBLMFunctor as EddyDiffusionBLM

# Import canonical transport classes
from ...transport.advection import LinearAdvection
from ...transport.diffusion import FickianDiffusion
from ...transport.advection_diffusion import AdvectionDiffusion

# Re-export with atmosphere-specific names
AtmosphericAdvection = LinearAdvection
TurbulentDiffusion = FickianDiffusion
AtmosphericTransport = AdvectionDiffusion

__all__ = [
    'AtmosphericAdvectionDiffusionFunctor',
    'AtmosphericAdvectionDiffusion',
    'EddyDiffusionBLM',
    'AtmosphericAdvection',
    'TurbulentDiffusion',
    'AtmosphericTransport'
]
