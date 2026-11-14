"""
Scale Coupling Module

Provides temporal and spatial scale coupling functionality for the RELIC framework.
"""

# Import temporal modules
from .scale_manager import (
    ScaleManager,
    build_quickstart_scale_config,
)
from .spatial import (
    AutoRegridder,
    BilinearRegridder,
    ConservativeRegridder,
    GridDescriptor,
    GridUtils,
    NearestNeighborRegridder,
    SpatialRegridderFactory,
)
from .temporal import (
    AggregationConfig,
    AggregationMethod,
    AggregationResult,
    # Calendar handling
    CalendarHandler,
    ConservationLevel,
    DisaggregationConfig,
    DisaggregationMethod,
    DisaggregationResult,
    # Quality control
    QualityFlags,
    ScalingDirection,
    # Aggregation
    TemporalAggregator,
    TemporalConfig,
    # Disaggregation
    TemporalDisaggregator,
    TemporalScaling,
    # Base classes
    TemporalScalingBase,
    TemporalSemantics,
)
from .topology_builder import TopologyBuilder
from .types import (
    ScalingAudit,
    ScalingStrategy,
    SpatialQuantityBasis,
    convert_spatial_basis,
    resolve_spatial_basis,
)

__all__ = [
    # Temporal
    'TemporalScalingBase',
    'TemporalScaling',
    'TemporalConfig',
    'TemporalSemantics',
    'ScalingDirection',
    'ConservationLevel',
    'TemporalAggregator',
    'AggregationMethod',
    'AggregationConfig',
    'AggregationResult',
    'TemporalDisaggregator',
    'DisaggregationMethod',
    'DisaggregationConfig',
    'DisaggregationResult',
    'QualityFlags',
    'CalendarHandler',
    # Spatial
    'GridDescriptor',
    'GridUtils',
    'ConservativeRegridder',
    'BilinearRegridder',
    'NearestNeighborRegridder',
    'SpatialRegridderFactory',
    'AutoRegridder',
    # Core manager entry
    'ScaleManager',
    'build_quickstart_scale_config',
    # Shared types
    'ScalingStrategy',
    'ScalingAudit',
    'SpatialQuantityBasis',
    'resolve_spatial_basis',
    'convert_spatial_basis',
    'TopologyBuilder',
]
