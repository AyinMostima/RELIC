"""
Hillslope Scale Model Configuration

Defines functors and variables exposed at the hillslope scale.
Represents slope units with hydrological connectivity (100m - 10km scale).

Scale Characteristics:
    - Spatial extent: Hydrological slope units (100m - 10km)
    - Key processes: Lateral water flow, nutrient transport, erosion
    - Primary variables: Streamflow, baseflow, sediment transport

Mathematical Expression:
    Hillslope scale processes:
    Q_streamflow = sum(patch_runoff * area_fraction)  [Streamflow aggregation]
    Q_baseflow = K_sat * h_avg * gradient             [Baseflow routing]
    Sediment = f(runoff, slope, vegetation_cover)     [Erosion processes]

References:
    - Band, L.E. et al. (2001). RHESSys: Regional Hydro-Ecologic Simulation System
    - Tague, C.L. & Band, L.E. (2004). RHESSys: Integrating spatially distributed
    - McDonnell, J.J. (2003). Where does water go when it rains?
"""

from typing import List, Dict, Any

# Import patch scale components
from .patch import LAYER_FUNCTORS as PATCH_FUNCTORS
from .patch import EXPOSED_VARIABLES as PATCH_VARIABLES

# Additional functors active at hillslope scale
HILLSLOPE_FUNCTORS: List[str] = [
    # Hydrological routing
    "subsurface_flow.lateral_routing",
    "surface_runoff.overland_flow",
    "streamflow.routing_kinematic",
    
    # Nutrient transport
    "nitrogen_transport.hillslope_routing",
    "carbon_transport.dissolved_organic",
    
    # Erosion and sediment
    "soil_erosion.rusle_modified",
    "sediment_transport.hillslope_routing"
]

# Combined functors (patch + hillslope specific)
LAYER_FUNCTORS: List[str] = PATCH_FUNCTORS + HILLSLOPE_FUNCTORS

# Variables exposed at hillslope scale
EXPOSED_VARIABLES: Dict[str, Dict[str, Any]] = {
    **PATCH_VARIABLES,  # Inherit all patch variables
    
    # Hydrological variables
    "streamflow": {
        "units": "m3 s-1",
        "standard_name": "water_volume_transport_in_river_channel",
        "variable_type": "extensive",
        "spatial_basis": "cell_total",
        "description": "Hillslope streamflow",
        "aggregation_method": "sum"
    },
    
    "baseflow": {
        "units": "m3 s-1", 
        "standard_name": None,
        "variable_type": "extensive",
        "spatial_basis": "cell_total",
        "description": "Subsurface baseflow (volume flow); CF v94 subsurface_runoff_flux is a mass flux density (kg m-2 s-1), so no standard name",
        "aggregation_method": "sum"
    },
    
    "lateral_flow": {
        "units": "kg m-2 s-1",
        "standard_name": "subsurface_runoff_flux", 
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Lateral subsurface flow",
        "aggregation_method": "area_weighted_mean"
    },
    
    # Nutrient transport
    "dissolved_n": {
        "units": "kg N m-2 s-1",
        "standard_name": None,
        "variable_type": "extensive", 
        "spatial_basis": "area_density",
        "description": "Dissolved nitrogen transport (mass of nitrogen); no CF v94 standard name for lateral dissolved-nitrogen transport",
        "aggregation_method": "area_weighted_mean"
    },
    
    "dissolved_c": {
        "units": "kg C m-2 s-1",
        "standard_name": None,
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Dissolved organic carbon transport (mass of carbon); no CF v94 standard name for lateral DOC transport",
        "aggregation_method": "area_weighted_mean"
    },
    
    # Erosion variables
    "soil_erosion_rate": {
        "units": "kg m-2 s-1",
        "standard_name": None,
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Soil erosion rate; no CF v94 standard name for soil erosion",
        "aggregation_method": "area_weighted_mean"
    },
    
    "sediment_load": {
        "units": "kg m-2 s-1", 
        "standard_name": None,
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Sediment transport load; no CF v94 standard name for a land sediment load",
        "aggregation_method": "area_weighted_mean"
    }
}

# Scale-specific constants
CHARACTERISTIC_LENGTH = 1000.0  # m, typical hillslope length
TYPICAL_SLOPE = 0.1             # dimensionless, typical slope gradient
