"""
Plant Functional Type (PFT) module for RELIC.

This module provides the PFT framework for modeling different vegetation types.
"""

from .pft_registry import PFTRegistry
from .pft_types import PFTDefinition, PFTCategory, PhotosynthesisPathway

__all__ = [
    'PFTRegistry', 
    'PFTDefinition',
    'PFTCategory',
    'PhotosynthesisPathway'
]
