"""
Pedosphere Transport Processes

This module implements transport mechanisms specific to soil systems,
including solute transport, gas diffusion, and preferential flow.

References
----------
.. [1] Jury, W. A., & Horton, R. (2004). Soil Physics (6th ed.).
       John Wiley & Sons. ISBN: 978-0-471-05965-3
.. [2] Hillel, D. (2003). Introduction to Environmental Soil Physics.
       Academic Press. ISBN: 978-0-12-348655-4
.. [3] Bear, J. (1972). Dynamics of Fluids in Porous Media. 
       American Elsevier Publishing Company.
.. [4] Simunek, J., van Genuchten, M.T., & Sejna, M. (2008). 
       Development and applications of the HYDRUS and STANMOD software packages
       and related codes. Vadose Zone Journal, 7(2), 587-600.
.. [5] Jarvis, N.J. (2007). A review of non-equilibrium water flow and solute
       transport in soil macropores: principles, controlling factors and consequences
       for water quality. European Journal of Soil Science, 58(3), 523-546.

Mathematical Framework
----------------------
Solute Transport (Advection-Dispersion Equation):
    partialC/partialt + rhopartialS/partialt = nabla*(DnablaC) - nabla*(qC) + phi
    
    where:
    - C = solute concentration in liquid phase [M L?^3]
    - S = sorbed concentration [M M?^1]
    - rho = bulk density [M L?^3]
    - D = dispersion coefficient tensor [L^2 T?^1]
    - q = Darcy flux [L T?^1]
    - phi = source/sink term [M L?^3 T?^1]

Gas Diffusion (Fick's Law in Porous Media):
    Jg = -Dg,s * nablaC
    
    where:
    - Jg = gas flux [M L?^2 T?^1]
    - Dg,s = gas diffusion coefficient in soil [L^2 T?^1]
    - Dg,s = Dg,0 * tau * thetaa
    - Dg,0 = diffusion coefficient in free air [L^2 T?^1]
    - tau = tortuosity factor (Millington-Quirk: thetaa^(10/3)/thetat^2)
    - thetaa = air-filled porosity [L^3 L?^3]

Preferential Flow (Dual-Porosity Model):
    partialthetam*Cm/partialt = nabla*(DmnablaCm) - nabla*(qm*Cm) - Gammaw
    partialthetaim*Cim/partialt = Gammaw
    
    where:
    - subscripts m and im denote mobile and immobile regions
    - Gammaw = exchange rate between regions [M L-3 T-1]
    - Gammaw = alpha(Cm - Cim)
    - alpha = first-order mass transfer coefficient [T-1]

Retardation Factor:
    R = 1 + (rho*Kd)/theta
    
    where:
    - R = retardation factor [dimensionless]
    - Kd = distribution coefficient [L^3 M-1]
    - theta = volumetric water content [L^3 L-3]

Parameters Units
----------------
- Hydraulic conductivity: m s-1
- Diffusion coefficients: m^2 s-1
- Dispersivity: m
- Porosity: m^3 m-3
- Bulk density: kg m-3
- Sorption coefficients: m^3 kg-1
"""


# Import new pedosphere-specific implementations
from .SoilWaterTransportFunctor import SoilWaterTransportFunctor

# Import pedosphere-specific transport functors
from .solute_advection_diffusion import SoluteAdvectionDiffusion1D
from .gas_diffusion import GasDiffusion1D, GasSpecies, DiffusivityModel

__all__ = [
    # New implementations (recommended)
    'SoilWaterTransportFunctor',
    'SoluteAdvectionDiffusion1D',
    'GasDiffusion1D',
    'GasSpecies',
    'DiffusivityModel',
]
