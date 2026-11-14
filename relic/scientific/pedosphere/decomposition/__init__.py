"""
Decomposition processes for soil carbon dynamics.

This package provides decomposition models including first-order kinetics
and microbial-enzyme explicit approaches.
"""

from .first_order import FirstOrderDecomposition
from .microbial_enzyme import MicrobialEnzymeDecomposition

__all__ = ['FirstOrderDecomposition', 'MicrobialEnzymeDecomposition']
