"""
Phenology Module for RELIC Framework

This module provides comprehensive phenological modeling capabilities for
vegetation seasonal dynamics. It includes multiple phenology approaches
from simple static date-based models to sophisticated multi-factor
environmental models.

Architecture:
    - PhenologyBase: Abstract base class for all phenology models  
    - Concrete implementations: Static, GDD, GSI, and Crop models
    - Parameter management: Type-safe parameter containers
    - Utility functions: Mathematical and computational tools
    
Models Available:
    - StaticPhenology: Fixed calendar date phenology
    - GDDPhenology: Growing degree day accumulation model
    - GSIPhenology: Growing season index model (Jolly et al. 2005)
    - CropPhenology: Crop-specific development model
    
Integration:
    - Fully consistent with RELIC core architecture
    - Inherits from BiologicalProcess and FunctorBase
    - Supports automatic differentiation through PyTorch
    - Integrates with carbon/nitrogen cycling and growth models

References:
    - Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict 
      foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.
    - White, M.A., et al. (1997). A continental phenology model for monitoring 
      vegetation responses to interannual climatic variability.
    - McMaster, G.S. & Wilhelm, W.W. (1997). Growing degree-days: one equation, 
      two interpretations. Agricultural and Forest Meteorology, 87(4), 291-300.
"""

# Import base classes and core components
from .base import (
    PhenologyBase,
    PhenologyPhase,
    PhenologyOutput,
    PhenologyState,
    BiomassFluxes
)

# Import parameter system
from .parameters import (
    PhenologyParameters,
    PhenologyParameterSets,
    PhenologyType,
    CropType
)

# Import utility functions
from .utils import (
    compute_gdd,
    compute_gsi,
    compute_gsi_components,
    compute_daylength,
    compute_photoperiod_factor,
    julian_day_to_date,
    date_to_julian_day,
    rolling_average,
    thermal_time_accumulation,
    phenological_threshold_response,
    compute_chilling_hours,
    validate_environmental_data
)

# Import concrete model implementations
from .models import (StaticPhenology,
    GDDPhenology,
    GSIPhenology,
    CropPhenology
)

# Import main executable functor
from .phenology_functor import PhenologyFunctor

# Export all public components
__all__ = [
    # Base classes and data structures
    'PhenologyBase',
    'PhenologyPhase', 
    'PhenologyOutput',
    'PhenologyState',
    'BiomassFluxes',
    
    # Parameter system
    'PhenologyParameters',
    'PhenologyParameterSets',
    'PhenologyType',
    'CropType',
    
    # Utility functions
    'compute_gdd',
    'compute_gsi',
    'compute_gsi_components',
    'compute_daylength',
    'compute_photoperiod_factor',
    'julian_day_to_date',
    'date_to_julian_day',
    'rolling_average',
    'thermal_time_accumulation',
    'phenological_threshold_response',
    'compute_chilling_hours',
    'validate_environmental_data',
    
    # Concrete model implementations
    'StaticPhenology',
    'GDDPhenology',
    'GSIPhenology', 
    'CropPhenology',
    
    # Main executable functor
    'PhenologyFunctor'
]

# Version information
__version__ = "1.0.0"
__author__ = "Yuchong Long"

# Module metadata
__doc_url__ = "https://relic.readthedocs.io/en/latest/modules/phenology.html"
__scientific_references__ = [
    "Jolly, W.M., et al. (2005). Global Change Biology, 11(4), 619-632",
    "White, M.A., et al. (1997). Global Biogeochemical Cycles, 11(2), 217-234",
    "McMaster, G.S. & Wilhelm, W.W. (1997). Agricultural and Forest Meteorology, 87(4), 291-300",
    "Richardson, A.D., et al. (2013). Agricultural and Forest Meteorology, 169, 156-173"
]

