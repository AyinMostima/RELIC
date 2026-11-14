"""
Anthropogenic Emissions Module

This module handles anthropogenic emissions including inventory management,
sector allocation, chemical speciation, and coupling to atmospheric chemistry.

Primary Components:
    - AtmosphereCoupling: Fail-closed dependency-unbound runtime boundary
    - SpeciesMapper: Chemical mechanism mapping and unit conversions
    - SectorAllocation: Canonical sector/species emission allocation
    - EmissionInventory: Fail-closed domain-unbound runtime boundary

"""

# Import main implementations
from .coupling_to_atmosphere import AtmosphereCoupling
from .sector_allocation import SectorAllocation
from .species_mapping import SpeciesMapper
from .atmosphere_interface_contract import ChemicalMechanism

# Parameter management imports
# Import the fail-closed canonical inventory runtime boundary.
from ..inventory import EmissionInventory

# Export public interface
__all__ = [
    # Public implementations and fail-closed runtime boundaries
    "AtmosphereCoupling",
    "SpeciesMapper",
    "ChemicalMechanism",
    "SectorAllocation",

    # Canonical inventory runtime boundary
    "EmissionInventory",
]
