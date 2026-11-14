"""
Common Utilities for Phenology Module

This package provides shared utilities for phenological modeling including
strategy patterns, temporal state containers,
and response curve functions. These components reduce code duplication and
improve maintainability across different phenology models.

Components:
    - strategies: Growth/senescence strategy patterns
    - temporal_state: Tensor-based time-series containers
    - response_curves: Mathematical response functions
"""

from .strategies import (
    PhenologyStrategy,
    GrowthFactorStrategy,
    SenescenceStrategy, 
    TransitionStrategy,
    StaticGrowthStrategy,
    GDDGrowthStrategy,
    TimeBasedSenescenceStrategy,
    CompositeStrategy,
    create_temperate_deciduous_strategy
)

from .temporal_state import (
    RollingWindowState,
    TemporalAccumulator,
    MultiVariableRollingWindow,
    create_rolling_buffer
)

from .response_curves import (
    sigmoid_response,
    tanh_response,
    double_sigmoid_response,
    linear_threshold_response,
    exponential_response,
    step_response
)

# Parameter management imports
__all__ = [
    # Strategy patterns
    'PhenologyStrategy',
    'GrowthFactorStrategy',
    'SenescenceStrategy', 
    'TransitionStrategy',
    'StaticGrowthStrategy',
    'GDDGrowthStrategy',
    'TimeBasedSenescenceStrategy',
    'CompositeStrategy',
    'create_temperate_deciduous_strategy',
    
    # Temporal containers
    'RollingWindowState',
    'TemporalAccumulator', 
    'MultiVariableRollingWindow',
    'create_rolling_buffer',
    
    # Response curves
    'sigmoid_response',
    'tanh_response',
    'double_sigmoid_response',
    'linear_threshold_response',
    'exponential_response',
    'step_response',
]

__version__ = "1.0.0"
