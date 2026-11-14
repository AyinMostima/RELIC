"""
Landscape Scale Model Configuration

Defines functors and variables exposed at the landscape scale.
Represents heterogeneous mosaics of patches (1km - 100km scale).

Scale Characteristics:
    - Spatial extent: Watershed, regional ecosystem (1km - 100km)
    - Key processes: Lateral transport, land-atmosphere exchange, disturbances
    - Primary variables: Regional fluxes, spatial heterogeneity patterns

Mathematical Expression:
    Landscape scale processes:
    F_regional = Sigma_patches (F_patch * A_patch) / A_total   [Flux aggregation]
    Lateral_transport = nabla * (u  * C)                       [Advective transport]
    Heterogeneity = sigma(F_patches) / mu(F_patches)           [Spatial variability]

References:
    - Turner, M.G. et al. (2001). Landscape ecology in theory and practice
    - Peters, D.P. et al. (2004). Cross-scale interactions and landscape-scale patterns
    - Wu, J. & Li, H. (2006). Concepts of scale and scaling in landscape ecology
"""

from typing import List, Dict, Any

# Import patch scale components
from .patch import LAYER_FUNCTORS as PATCH_FUNCTORS
from .patch import EXPOSED_VARIABLES as PATCH_VARIABLES

# Additional functors active at landscape scale
LANDSCAPE_FUNCTORS: List[str] = [
    # Lateral transport processes
    "lateral_transport.groundwater", 
    "lateral_transport.surface_water",
    "lateral_transport.carbon_leaching",
    
    # Disturbance processes
    "disturbance.fire_spread", 
    "disturbance.pest_outbreak",
    "disturbance.windthrow",
    
    # Land-atmosphere coupling
    "atmospheric_coupling.boundary_layer",
    "atmospheric_coupling.mesoscale_circulation",
    
    # Spatial heterogeneity processes
    "spatial_aggregation.flux_footprint",
    "spatial_aggregation.heterogeneity_effects"
]

# Combined functors (patch + landscape specific)
LAYER_FUNCTORS: List[str] = PATCH_FUNCTORS + LANDSCAPE_FUNCTORS

# Variables exposed by landscape scale
LANDSCAPE_VARIABLES: Dict[str, Dict[str, Any]] = {
    "regional_nee": {
        "standard_name": None,
        "units": "kg C m-2 s-1",
        "variable_type": "extensive", 
        "spatial_basis": "area_density",
        "description": "Regional net ecosystem exchange NEE = Reco - GPP, landscape average (mass of carbon, positive from ecosystem to atmosphere); CF v94 names only the downward net land flux, not NEE, so no standard name",
        "cell_methods": "time: mean area: mean landscape: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "regional_evapotranspiration": {
        "standard_name": "water_evapotranspiration_flux",
        "units": "kg m-2 s-1",
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Regional evapotranspiration flux",
        "cell_methods": "time: mean area: mean landscape: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "spatial_heterogeneity_nee": {
        "standard_name": None,
        "units": "1",
        "variable_type": "intensive",
        "description": "Spatial coefficient of variation of NEE; no CF v94 standard name for this statistic",
        "cell_methods": "landscape: standard_deviation landscape: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "lateral_carbon_flux": {
        "standard_name": None,
        "units": "kg C m-2 s-1", 
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Lateral carbon transport (leaching + groundwater, mass of carbon); no CF v94 standard name for this sum",
        "cell_methods": "time: mean area: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "fire_burned_fraction": {
        "standard_name": "burned_area_fraction",
        "units": "1",
        "variable_type": "intensive",
        "description": "Fraction of landscape burned by fire",
        "cell_methods": "time: sum landscape: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "disturbance_intensity": {
        "standard_name": None,
        "units": "1",
        "variable_type": "intensive",
        "description": "Combined disturbance intensity index; no CF v94 standard name for this index",
        "cell_methods": "time: maximum landscape: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "boundary_layer_height": {
        "standard_name": "atmosphere_boundary_layer_thickness",
        "units": "m",
        "variable_type": "intensive",
        "description": "Atmospheric boundary layer height",
        "cell_methods": "time: mean landscape: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "surface_roughness": {
        "standard_name": "surface_roughness_length",
        "units": "m", 
        "variable_type": "intensive",
        "description": "Effective landscape surface roughness",
        "cell_methods": "area: mean",
        "aggregation_method": "effective_roughness_length"
    },
    
    "flux_footprint_area": {
        "standard_name": None,
        "units": "m2",
        "variable_type": "intensive",
        "description": "Representative flux measurement footprint; no CF v94 standard name for a flux footprint",
        "cell_methods": "time: mean",
        "aggregation_method": "area_weighted_mean"
    }
}

# Combined variables (patch + landscape specific)  
EXPOSED_VARIABLES: Dict[str, Dict[str, Any]] = {**PATCH_VARIABLES, **LANDSCAPE_VARIABLES}

# Coordinate variables for landscape scale
COORDINATE_VARIABLES: Dict[str, Dict[str, Any]] = {
    "landscape_id": {
        "standard_name": "landscape_unit_identifier",
        "units": "1",
        "description": "Unique identifier for landscape units"
    },
    
    "land_cover_type": {
        "standard_name": "land_cover_lccs",
        "units": "1",
        "description": "Land cover classification (LCCS standard)"
    },
    
    "elevation_class": {
        "standard_name": "surface_altitude_class", 
        "units": "1",
        "description": "Elevation class for topographic effects"
    },
    
    "disturbance_history": {
        "standard_name": "ecosystem_disturbance_history_index",
        "units": "1",
        "description": "Historical disturbance regime classification"
    }
}

# Scale coupling information
SCALE_COUPLING: Dict[str, Any] = {
    "downscaling_source": "patch",
    "upscaling_target": "regional",
    "aggregation_methods": {
        "regional_nee": "area_weighted_mean",
        "regional_evapotranspiration": "area_weighted_mean",
        "lateral_carbon_flux": "sum", 
        "fire_burned_fraction": "area_weighted_mean",
        "boundary_layer_height": "volume_weighted_mean",
        "surface_roughness": "area_weighted_mean"
    },
    "conservation_variables": [
        "regional_nee", "regional_evapotranspiration", 
        "lateral_carbon_flux"
    ],
    "heterogeneity_metrics": [
        "spatial_heterogeneity_nee", "disturbance_intensity"
    ]
}