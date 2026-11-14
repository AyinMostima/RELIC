"""
Patch Scale Model Configuration

Defines functors and variables exposed at the patch/stand scale.
Represents small homogeneous vegetation units (10m - 1km scale).

Scale Characteristics:
    - Spatial extent: Forest stand, agricultural field (10m - 1km)
    - Key processes: Canopy processes, water balance, BVOC emissions
    - Primary variables: NEE, evapotranspiration, canopy temperature

Mathematical Expression:
    Patch scale processes:
    NEE = GPP - R_eco                    [Net ecosystem exchange]
    ET = E_canopy + E_soil + T           [Total evapotranspiration]
    BVOC = f(T, LAI, light) * emission_factors [Biogenic volatile emissions]

References:
    - Baldocchi, D. et al. (2001). FLUXNET: A new tool to study carbon and water cycling
    - Guenther, A. et al. (2012). The Model of Emissions of Gases and Aerosols from Nature
    - Bonan, G.B. (2019). Climate Change and Terrestrial Ecosystem Modeling
"""

from typing import List, Dict, Any

# Import individual scale components
from .individual import LAYER_FUNCTORS as INDIVIDUAL_FUNCTORS
from .individual import EXPOSED_VARIABLES as INDIVIDUAL_VARIABLES

# Additional functors active at patch scale
PATCH_FUNCTORS: List[str] = [
    # Water cycle processes
    "evapotranspiration.penman_monteith",
    "soil_evaporation.basic",
    "canopy_interception.rutter",
    
    # Carbon cycle processes  
    "soil_respiration.temperature_dependent",
    "canopy_photosynthesis.scaling",
    
    # BVOC emissions
    "bvoc_emission.megan_basic",
    "bvoc_emission.isoprene_guenther",
    
    # Energy balance
    "canopy_temperature.multi_layer",
    "surface_energy_balance.basic"
]

# Combined functors (individual + patch specific)
LAYER_FUNCTORS: List[str] = INDIVIDUAL_FUNCTORS + PATCH_FUNCTORS

# Variables exposed by patch scale
PATCH_VARIABLES: Dict[str, Dict[str, Any]] = {
    "nee": {
        "standard_name": None,
        "units": "kg C m-2 s-1", 
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Net ecosystem exchange NEE = Reco - GPP (mass of carbon, positive from ecosystem to atmosphere); CF v94 names only the downward net land flux, not NEE, so no standard name",
        "cell_methods": "time: mean area: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "evapotranspiration": {
        "standard_name": "water_evapotranspiration_flux",
        "units": "kg m-2 s-1",
        "variable_type": "extensive", 
        "spatial_basis": "area_density",
        "description": "Total evapotranspiration (canopy + soil)",
        "cell_methods": "time: mean area: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "soil_respiration": {
        "standard_name": "surface_upward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_respiration_in_soil",
        "units": "kg C m-2 s-1",
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Heterotrophic + root (autotrophic) soil respiration (mass of carbon, upward positive)",
        "cell_methods": "time: mean area: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "canopy_temperature": {
        "standard_name": "canopy_temperature", 
        "units": "K",
        "variable_type": "intensive",
        "description": "Bulk canopy temperature",
        "cell_methods": "time: mean area: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "isoprene_emission": {
        "standard_name": None,
        "units": "mol m-2 s-1",
        "variable_type": "extensive", 
        "spatial_basis": "area_density",
        "description": "Isoprene emission flux (molar); CF v94 names isoprene emission only as a mass flux (kg m-2 s-1), so no standard name",
        "cell_methods": "time: mean area: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "monoterpenes_emission": {
        "standard_name": None,
        "units": "mol m-2 s-1", 
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Monoterpenes emission flux (molar); CF v94 names monoterpene emission only as a mass flux (kg m-2 s-1), so no standard name",
        "cell_methods": "time: mean area: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "sensible_heat": {
        "standard_name": "surface_upward_sensible_heat_flux",
        "units": "W m-2",
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Sensible heat flux from surface",
        "cell_methods": "time: mean area: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "latent_heat": {
        "standard_name": "surface_upward_latent_heat_flux", 
        "units": "W m-2",
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Latent heat flux from evapotranspiration",
        "cell_methods": "time: mean area: mean",
        "aggregation_method": "area_weighted_mean"
    }
}

# Combined variables (individual + patch specific)
EXPOSED_VARIABLES: Dict[str, Dict[str, Any]] = {**INDIVIDUAL_VARIABLES, **PATCH_VARIABLES}

# Coordinate variables for patch scale
COORDINATE_VARIABLES: Dict[str, Dict[str, Any]] = {
    "patch_id": {
        "standard_name": "vegetation_patch_identifier",
        "units": "1",
        "description": "Unique identifier for vegetation patches"
    },
    
    "pft": {
        "standard_name": "plant_functional_type",
        "units": "1", 
        "description": "Plant functional type classification"
    },
    
    "canopy_layer": {
        "standard_name": "canopy_vertical_layer",
        "units": "1",
        "description": "Vertical canopy layer (understory, midstory, canopy)"
    }
}

# Scale coupling information
SCALE_COUPLING: Dict[str, Any] = {
    "downscaling_source": "individual",
    "upscaling_target": "landscape", 
    "aggregation_methods": {
        "nee": "area_weighted_mean",
        "evapotranspiration": "area_weighted_mean", 
        "isoprene_emission": "sum",
        "monoterpenes_emission": "sum",
        "canopy_temperature": "area_weighted_mean",
        "sensible_heat": "area_weighted_mean",
        "latent_heat": "area_weighted_mean"
    },
    "conservation_variables": [
        "nee", "evapotranspiration", "soil_respiration",
        "isoprene_emission", "monoterpenes_emission"
    ]
}