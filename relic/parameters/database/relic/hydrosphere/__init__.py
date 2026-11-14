"""
Hydrosphere Parameter Database Module

This module provides access to all hydrosphere-related parameters including
evapotranspiration, soil moisture, runoff, groundwater, snow, transport,
physical constants, and numerical controls following the
RELIC unified parameter management system.

The parameters are organized into functional groups:
- evapotranspiration.yaml: Penman-Monteith and Priestley-Taylor parameters
- soil_moisture.yaml: Bucket model and Richards equation parameters
- runoff.yaml: Surface and subsurface flow parameters
- groundwater.yaml: Groundwater storage and dynamics parameters
- snow.yaml: Snow albedo and snow physics parameters
- transport.yaml: Capillary rise and vertical percolation transport parameters
- physical.yaml: Physical constants and reference values
- numerical.yaml: Numerical tolerances and stability controls

All parameters follow CF-1.8 naming conventions and include:
- Physical bounds validation
- Unit consistency checking
- Literature references (DOI/citations)
- Calibration status and gradient requirements
- Scientific provenance tracking

Usage Example:
    from relic.parameters import get_global_manager
    
    manager = get_global_manager()
    
    # Get evapotranspiration parameters
    roughness_length = manager.get('hydrosphere.evapotranspiration.roughness_length')
    
    # Get soil moisture parameters
    field_capacity = manager.get('hydrosphere.soil_moisture.field_capacity')
    
    # Get runoff parameters
    curve_number = manager.get('hydrosphere.runoff.curve_number')

References:
    - Allen, R.G. et al. (1998). Crop evapotranspiration - Guidelines for computing 
      crop water requirements. FAO Irrigation paper 56
    - Manabe, S. (1969). Climate and the ocean circulation. Mon. Wea. Rev., 97, 739-774
    - van Genuchten, M.Th. (1980). A closed-form equation for predicting hydraulic 
      conductivity. Soil Sci. Soc. Am. J., 44, 892-898
    - Beven, K. & Kirkby, M.J. (1979). A physically based variable contributing area 
      model. Hydrol. Sci. Bull., 24, 43-69
"""
from typing import Any, Dict, List

# Parameter file inventory
PARAMETER_FILES = {
    'evapotranspiration': 'evapotranspiration.yaml',
    'soil_moisture': 'soil_moisture.yaml',
    'runoff': 'runoff.yaml',
    'groundwater': 'groundwater.yaml',
    'snow': 'snow.yaml',
    'transport': 'transport.yaml',
    'numerical': 'numerical.yaml',
    'physical': 'physical.yaml',
}

# Parameter namespace mapping
PARAMETER_NAMESPACES = {
    'evapotranspiration': 'hydrosphere.evapotranspiration',
    'soil_moisture': 'hydrosphere.soil_moisture',
    'runoff': 'hydrosphere.runoff',
    'groundwater': 'hydrosphere.groundwater',
    'snow': 'hydrosphere.snow',
    'transport': 'hydrosphere.transport',
    'numerical': 'hydrosphere.numerical',
    'physical': 'hydrosphere.physical',
}

# Key parameter groups for quick access
KEY_PARAMETERS = {
    'evapotranspiration_core': [
        'hydrosphere.evapotranspiration.roughness_length',
        'hydrosphere.evapotranspiration.displacement_height',
        'hydrosphere.evapotranspiration.reference_height',
        'hydrosphere.evapotranspiration.soil_heat_fraction',
        'hydrosphere.evapotranspiration.alpha_pt',
        'hydrosphere.evapotranspiration.enable_canopy_interception',
        'hydrosphere.evapotranspiration.interception_capacity_per_LAI',
        'hydrosphere.evapotranspiration.canopy_evaporation_efficiency',
    ],
    'soil_hydraulics': [
        'hydrosphere.soil_moisture.field_capacity',
        'hydrosphere.soil_moisture.wilting_point',
        'hydrosphere.soil_moisture.porosity',
        'hydrosphere.soil_moisture.saturated_hydraulic_conductivity',
        'hydrosphere.soil_moisture.residual_water_content',
    ],
    'van_genuchten': [
        'hydrosphere.soil_moisture.alpha_vg',
        'hydrosphere.soil_moisture.n_vg',
    ],
    'runoff_generation': [
        'hydrosphere.runoff.curve_number',
        'hydrosphere.runoff.initial_abstraction_ratio',
        'hydrosphere.runoff.infiltration_capacity',
    ],
    'groundwater_core': [
        'hydrosphere.groundwater.surface_porosity',
        'hydrosphere.groundwater.porosity_decay_length',
    ],
    'snow_core': [
        'hydrosphere.snow.fresh_snow_albedo',
        'hydrosphere.snow.old_snow_albedo',
        'hydrosphere.snow.albedo_decay_rate_cold',
        'hydrosphere.snow.albedo_decay_rate_warm',
        'hydrosphere.snow.temperature_threshold_warm',
        'hydrosphere.snow.degree_day_factor',
        'hydrosphere.snow.rain_snow_temperature_threshold',
        'hydrosphere.snow.snow_density',
    ],
    'transport_core': [
        'hydrosphere.transport.vertical_percolation.enable_temperature_effects',
        'hydrosphere.transport.vertical_percolation.hydraulic_model',
        'hydrosphere.transport.vg_l',
        'hydrosphere.transport.bc_lambda',
        'hydrosphere.transport.ch_b_parameter',
        'hydrosphere.transport.drainage_time_constant',
        'hydrosphere.transport.soil_layer_thickness',
    ],
    'numerical_core': [
        'hydrosphere.numerical.mass_conservation_tolerance_mm',
        'hydrosphere.numerical.critical_mass_balance_error_threshold',
        'hydrosphere.numerical.minimum_flux_threshold',
        'hydrosphere.numerical.minimum_storage_outflow',
    ],
}

# Physical constants used across hydrosphere modules
PHYSICAL_CONSTANTS = [
    'relic.physical.constants.water.density_liquid',
    'relic.physical.constants.air.density_stp',
    'relic.physical.constants.universal.gas_constant',
    'relic.physical.constants.earth.gravity_acceleration',
    'relic.physical.constants.turbulence.von_karman_constant',
]

# Calibratable parameters (requires_grad=True)
CALIBRATABLE_PARAMETERS = [
    # Evapotranspiration
    'hydrosphere.evapotranspiration.roughness_length',
    'hydrosphere.evapotranspiration.soil_heat_fraction',
    'hydrosphere.evapotranspiration.alpha_pt',
    'hydrosphere.evapotranspiration.interception_capacity_per_LAI',
    'hydrosphere.evapotranspiration.canopy_evaporation_efficiency',
    # Soil moisture
    'hydrosphere.soil_moisture.field_capacity',
    'hydrosphere.soil_moisture.wilting_point',
    'hydrosphere.soil_moisture.root_zone_depth',
    'hydrosphere.soil_moisture.alpha_vg',
    'hydrosphere.soil_moisture.n_vg',
    'hydrosphere.soil_moisture.saturated_hydraulic_conductivity',
    # Runoff
    'hydrosphere.runoff.curve_number',
    'hydrosphere.runoff.initial_abstraction_ratio',
    'hydrosphere.runoff.transmissivity',
    'hydrosphere.runoff.baseflow_recession_constant',
    'hydrosphere.runoff.return_flow_fraction',
    # Transport
    'hydrosphere.transport.drainage_time_constant',
]

def get_hydrosphere_parameters() -> Dict[str, List[str]]:
    """
    Get all hydrosphere parameter keys organized by category.
    
    Returns:
        dict: Dictionary of parameter categories and their keys
    """
    return KEY_PARAMETERS

def get_calibratable_parameters() -> List[str]:
    """
    Get list of all calibratable hydrosphere parameters.
    
    Returns:
        list: Parameter keys that can be calibrated (requires_grad=True)
    """
    return CALIBRATABLE_PARAMETERS

def get_hydrosphere_constant_keys() -> List[str]:
    """
    Get list of physical constants used in hydrosphere calculations.
    
    Returns:
        list: Parameter keys for physical constants
    """
    return PHYSICAL_CONSTANTS

def validate_parameter_consistency(manager: Any) -> List[str]:
    """
    Validate consistency relationships between hydrosphere parameters.
    
    Args:
        manager: UnifiedParameterManager instance
        
    Returns:
        list: Validation error messages if any
    """
    errors = []
    
    # Soil moisture relationship validation
    try:
        porosity = manager.get('hydrosphere.soil_moisture.porosity')
        field_capacity = manager.get('hydrosphere.soil_moisture.field_capacity') 
        wilting_point = manager.get('hydrosphere.soil_moisture.wilting_point')
        residual_content = manager.get('hydrosphere.soil_moisture.residual_water_content')
        
        if porosity <= field_capacity:
            errors.append("Porosity must be greater than field capacity")
        if field_capacity <= wilting_point:
            errors.append("Field capacity must be greater than wilting point")
        if wilting_point <= residual_content:
            errors.append("Wilting point must be greater than residual water content")
            
    except KeyError as e:
        errors.append(f"Missing parameter for soil moisture validation: {e}")
    
    # Canopy structure: the FAO-56 profile needs the displacement height below the
    # reference height (no canopy height is a parameter any more).
    try:
        reference_height = manager.get('hydrosphere.evapotranspiration.reference_height')
        displacement_height = manager.get('hydrosphere.evapotranspiration.displacement_height')
        roughness_length = manager.get('hydrosphere.evapotranspiration.roughness_length')

        if reference_height - displacement_height <= roughness_length:
            errors.append("Reference height minus displacement must exceed roughness length")

    except KeyError as e:
        errors.append(f"Missing canopy parameter: {e}")
    
    return errors

__all__ = [
    'PARAMETER_FILES',
    'PARAMETER_NAMESPACES', 
    'KEY_PARAMETERS',
    'PHYSICAL_CONSTANTS',
    'CALIBRATABLE_PARAMETERS',
    'get_hydrosphere_parameters',
    'get_calibratable_parameters',
    'get_hydrosphere_constant_keys',
    'validate_parameter_consistency'
]
