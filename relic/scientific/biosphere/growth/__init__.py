"""
Plant Growth Processes for RELIC.

This module provides production growth components for biomass allocation,
structural development, and phenological control.

Main Components
---------------
- BiomassAllocation: Canonical NPP allocation to leaf, stem, and root ports
- StructuralGrowth: Observation-only LAI tendency from the leaf-growth port
- PhenologyGrowth: Selected-owner growth-activity diagnostic

Mathematical Framework
----------------------
1. Carbon allocation: dB_i/dt = f_i * NPP, with sum_i(f_i) = 1
2. LAI observation: r_net = SLA * growth_rate_leaf - k_leaf * LAI
3. Activity diagnostic: growth_active = phenological_factor > threshold

Units
-----
- Biomass (B): kg C m-2
- Growth rates: kg C m-2 s^-1
- Allocation fractions (f): dimensionless [0-1]
- LAI tendency: m2 m-2 s^-1

References
----------
- Cannell, M.G.R. & Dewar, R.C. (1994). Carbon allocation in trees. 
  Annals of Botany 73, 17-30
- Franklin, O., et al. (2012). Modeling carbon allocation in trees. 
  Tree Physiology 32, 648-666
- Poorter, H., et al. (2012). Biomass allocation to leaves, stems and roots. 
  New Phytologist 193, 30-50

Example Usage
-------------
    from relic.scientific.biosphere.growth import BiomassAllocation
    
    growth = BiomassAllocation()

    # Required state inputs include:
    # - net_primary_productivity [kg C m-2 s-1]
    # Enabled controls require air_temperature, water_stress_factor,
    # leaf_area_index, and phenology_phase.

    # state = {...}  # user-provided state dictionary
    # result = growth.forward(state, time=0.0, dt=3600.0)
"""

# Lazy imports keep package import side effects small.
from typing import Any

from .biomass_allocation import BiomassAllocation

__all__ = [
    'BiomassAllocation',
]

def __getattr__(name: str) -> Any:
    """Load public growth classes on first access."""
    if name == "StructuralGrowth":
        from .structural_growth import StructuralGrowth

        return StructuralGrowth
    if name == "PhenologyGrowth":
        from .phenology_growth import PhenologyGrowth

        return PhenologyGrowth
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Note: Growth classes are lazily loaded to keep import side effects small.
