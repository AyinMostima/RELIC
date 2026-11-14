"""
Hierarchical Scale Coupling Module

Provides hierarchical aggregation and disaggregation capabilities for
multi-scale ecosystem modeling with conservation enforcement.
"""

from .hierarchy_index import HierarchyIndex
from .aggregator import HierarchicalAggregator

__all__ = ['HierarchyIndex', 'HierarchicalAggregator']
