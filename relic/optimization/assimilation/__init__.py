"""
Data Assimilation Module for RELIC Optimization Framework

This module provides comprehensive data assimilation capabilities for 
biogeochemical model state estimation. It includes both ensemble and 
variational methods with flexible observation operator support.

Available Methods:
- Ensemble Kalman Filter (EnKF): Monte Carlo approach with ensemble forecasts
- 3D Variational (3D-Var): Optimization-based single-time analysis  
- 4D Variational (4D-Var): Framework for temporal optimization (skeleton)

Key Components:
- Base abstractions: Assimilator, ObservationOperator, AssimilationResult
- Observation handling: Observation, ObservationSet with quality control
- Method implementations: EnsembleKalmanFilter, ThreeDVar, FourDVar
- Configuration: AssimilationConfig, EnKFConfig, VariationalConfig

Scientific Applications:
- GPP/NEE flux tower data assimilation
- Remote sensing LAI/fAPAR integration
- Soil moisture and temperature profile updates
- Multi-variable ecosystem state estimation

References:
    - Evensen, G. (2003). The Ensemble Kalman Filter
    - Kalnay, E. (2003). Atmospheric Modeling, Data Assimilation and Predictability
    - Talagrand, O. & Courtier, P. (1987). Variational assimilation methods
"""

# Base classes and interfaces
from .base import (
    Assimilator,
    ObservationOperator, 
    IdentityOperator,
    Observation,
    ObservationSet,
    AssimilationResult,
    AssimilationConfig,
    AssimilationMethod
)

# Ensemble Kalman Filter
from .enkf import (
    EnsembleKalmanFilter,
    EnKFConfig
)

# Variational methods
from .variational import (
    ThreeDVar,
    FourDVar, 
    VariationalConfig
)

from .gradient_fusion import DualTrackGradientFusion

# Re-export all public components
__all__ = [
    # Base abstractions
    'Assimilator',
    'ObservationOperator',
    'IdentityOperator',
    'Observation', 
    'ObservationSet',
    'AssimilationResult',
    'AssimilationConfig',
    'AssimilationMethod',
    # Ensemble methods
    'EnsembleKalmanFilter',
    'EnKFConfig',
    # Variational methods
    'ThreeDVar',
    'FourDVar',
    'VariationalConfig',
    'DualTrackGradientFusion',
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "Yuchong Long"
