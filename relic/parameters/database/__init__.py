"""
Parameter Database Management for RELIC

This package provides the two-tier parameter database system for RELIC:

- relic/: Native RELIC parameters with full scientific documentation
- calibrated/: Calibration results and optimized parameters with provenance

Each tier maintains complete metadata including units, bounds, references,
and provenance information following FAIR data principles.

Architecture:
    - Hierarchical loading with precedence rules
    - Version tracking and schema validation
    - Automatic format detection (YAML/JSON/TOML)
    - Metadata preservation and enhancement
    - Integration with parameter converters

Database Structure:
    database/
     relic/
        biosphere/
           photosynthesis.yaml
           respiration.yaml
           bvoc.yaml
        hydrosphere/
           evapotranspiration.yaml
           soil_moisture.yaml
        pedosphere/
            decomposition.yaml
            nutrient_cycling.yaml
     calibrated/
         site_specific/
         regional/
         global/

References:
    - CF-1.8 Convention for standard parameter naming
    - FAIR Data Principles for scientific data management
    - PyTorch Parameter system documentation
    - YAML 1.2 specification for file formats
"""

from pathlib import Path
from typing import Dict, Any, Optional

# Version information
__version__ = "1.0.0"
__author__ = "RELIC Parameter Database Team"

# Database tier definitions
DATABASE_TIERS = {
    'relic': {
        'name': 'Native Parameters',
        'description': 'Native RELIC parameters with full scientific validation',
        'precedence': 1,  # Lowest precedence
        'schema_validation': True,
        'required_metadata': ['description', 'units', 'bounds']
    },
    'calibrated': {
        'name': 'Calibrated Parameters',
        'description': 'Calibration results and optimized parameters',
        'precedence': 2,  # Highest precedence
        'schema_validation': True,
        'required_metadata': ['description', 'calibration_method', 'performance_metrics']
    }
}

# Standard parameter namespaces
PARAMETER_NAMESPACES = {
    'biosphere': {
        'photosynthesis': 'Leaf-level photosynthesis parameters',
        'respiration': 'Plant respiration parameters',
        'bvoc': 'Biogenic volatile organic compound emissions',
        'growth': 'Plant growth and allocation parameters',
        'phenology': 'Phenological development parameters'
    },
    'hydrosphere': {
        'evapotranspiration': 'Evapotranspiration model parameters',
        'soil_moisture': 'Soil water balance parameters', 
        'runoff': 'Surface and subsurface runoff parameters'
    },
    'pedosphere': {
        'decomposition': 'Soil organic matter decomposition',
        'nutrient_cycling': 'Nitrogen and phosphorus cycling',
        'carbon_pools': 'Soil carbon pool parameters'
    },
    'atmosphere': {
        'radiation': 'Radiation transfer parameters',
        'turbulence': 'Atmospheric turbulence parameters',
        'chemistry': 'Atmospheric chemistry parameters'
    },
    'anthroposphere': {
        'land_use': 'Land use change parameters',
        'management': 'Agricultural management parameters',
        'emissions': 'Anthropogenic emission parameters'
    },
    'physical': {
        'constants': 'Physical and chemical constants',
        'material_properties': 'Material property parameters'
    }
}

# Schema validation settings
SCHEMA_VERSIONS = {
    '1.0.0': {
        'required_fields': ['key', 'value', 'units'],
        'optional_fields': [
            'description',
            'bounds',
            'dtype',
            'category',
            'references',
            'tags',
            'dimension',
            'requires_grad',
            'provenance',
        ],
        'validation_level': 'standard'
    }
}

def get_database_root() -> Path:
    """
    Get the root directory for parameter database.
    
    Returns:
        Path: Database root directory path
    """
    return Path(__file__).parent


def validate_database_structure() -> Dict[str, bool]:
    """
    Validate that database directory structure exists.
    
    Returns:
        Dict[str, bool]: Validation results for each tier
    """
    database_root = get_database_root()
    validation_results = {}
    
    for tier_name in DATABASE_TIERS.keys():
        tier_path = database_root / tier_name
        validation_results[tier_name] = tier_path.exists() and tier_path.is_dir()
    
    return validation_results


def create_database_structure() -> None:
    """
    Create the complete database directory structure.
    
    Creates all necessary directories for the three-tier parameter
    database with proper namespace organization.
    """
    database_root = get_database_root()
    
    # Create main tier directories
    for tier_name in DATABASE_TIERS.keys():
        tier_path = database_root / tier_name
        tier_path.mkdir(parents=True, exist_ok=True)
        
        # Create namespace subdirectories for relic tier
        if tier_name == 'relic':
            for namespace in PARAMETER_NAMESPACES.keys():
                namespace_path = tier_path / namespace
                namespace_path.mkdir(exist_ok=True)
        
        # Create calibration subdirectories
        elif tier_name == 'calibrated':
            calib_types = ['site_specific', 'regional', 'global']
            for calib_type in calib_types:
                calib_path = tier_path / calib_type
                calib_path.mkdir(exist_ok=True)


def get_tier_info(tier: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a database tier.
    
    Parameters:
        tier : str
            Database tier name
            
    Returns:
        Optional[Dict[str, Any]]: Tier information or None if not found
    """
    return DATABASE_TIERS.get(tier)


def list_namespaces(tier: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    List available parameter namespaces.
    
    Parameters:
        tier : Optional[str]
            Specific tier to check, all tiers if None
            
    Returns:
        Dict[str, Dict[str, str]]: Namespace information
    """
    if tier == 'relic':
        return PARAMETER_NAMESPACES
    else:
        return PARAMETER_NAMESPACES


# Export main components
__all__ = [
    'DATABASE_TIERS',
    'PARAMETER_NAMESPACES', 
    'SCHEMA_VERSIONS',
    'get_database_root',
    'validate_database_structure',
    'create_database_structure',
    'get_tier_info',
    'list_namespaces'
]
