"""
Carbon pools for soil carbon dynamics.

This package provides carbon pool transfer matrix implementation
with mass balance conservation.
"""

from .pools_matrix import PoolsMatrix
from .carbon_pools_functor import CarbonPoolsFunctor

__all__ = ['PoolsMatrix', 'CarbonPoolsFunctor']
