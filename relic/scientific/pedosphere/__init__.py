"""
Pedosphere processes for soil biogeochemistry.

This package provides comprehensive soil biogeochemical cycle implementations
including decomposition, carbon pools, and nutrient cycling.
"""

from . import decomposition
from . import carbon_pools
from . import nutrient_cycling
from . import methane

from .decomposition import FirstOrderDecomposition, MicrobialEnzymeDecomposition
from .carbon_pools import CarbonPoolsFunctor, PoolsMatrix
from .nutrient_cycling import MineralizationProcess, NitrificationProcess, DenitrificationProcess, StoichiometryConstraint, NitrogenLeachingProcess
from .methane import (
    MethaneScaleCouplingFunctor,
    MethaneProductionFunctor,
    MethaneOxidationFunctor,
    MethaneTransportFunctor,
    MethaneBudgetCoupler,
    register_methane_functors,
)

# Parameter management imports
__all__ = [
    'decomposition', 'carbon_pools', 'nutrient_cycling', 'methane',
    'FirstOrderDecomposition', 'MicrobialEnzymeDecomposition',
    'CarbonPoolsFunctor', 'PoolsMatrix',
    'MineralizationProcess', 'NitrificationProcess', 'DenitrificationProcess', 'StoichiometryConstraint', 'NitrogenLeachingProcess',
    'MethaneScaleCouplingFunctor', 'MethaneProductionFunctor', 'MethaneOxidationFunctor',
    'MethaneTransportFunctor', 'MethaneBudgetCoupler', 'register_methane_functors'
]
