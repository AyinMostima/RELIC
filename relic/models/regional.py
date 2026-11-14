"""
Regional Scale Model Configuration

Defines functors and variables exposed at the regional scale.
Represents large-scale ecosystem patterns (100km - 1000km scale).

Scale Characteristics:
    - Spatial extent: Biomes, continental regions (100km - 1000km+)
    - Key processes: Climate-ecosystem feedbacks, biogeochemical cycles, large-scale transport
    - Primary variables: Biome-scale budgets, climate coupling, emergent patterns

Mathematical Expression:
    Regional scale processes:
    Budget_regional = integralintegral F(x,y) dx dy                    [Spatial integration]
    Climate_feedback = partialT/partial(surface_fluxes)             [Land-atmosphere coupling] 
    Emergent_pattern = f(climate, vegetation, disturbance) [Pattern formation]

References:
    - Chapin III, F.S. et al. (2011). Principles of Terrestrial Ecosystem Ecology
    - Prentice, I.C. et al. (2007). Dynamic global vegetation modeling
    - Bonan, G.B. (2008). Forests and climate change: forcings, feedbacks, and the climate benefits
"""

from typing import List, Dict, Any

# Import landscape scale components  
from .landscape import LAYER_FUNCTORS as LANDSCAPE_FUNCTORS
from .landscape import EXPOSED_VARIABLES as LANDSCAPE_VARIABLES

# Additional functors active at regional scale
REGIONAL_FUNCTORS: List[str] = [
    # Climate-ecosystem feedbacks
    "climate_coupling.albedo_feedback",
    "climate_coupling.evapotranspiration_feedback", 
    "climate_coupling.roughness_feedback",
    
    # Biogeochemical cycles
    "nitrogen_cycle.regional_budget",
    "phosphorus_cycle.regional_budget", 
    "carbon_cycle.regional_budget",
    
    # Large-scale transport
    "atmospheric_transport.regional_advection",
    "hydrological_transport.river_network",
    
    # Vegetation dynamics
    "vegetation_dynamics.biome_transition",
    "vegetation_dynamics.species_migration",
    
    # External forcings
    "external_forcing.climate_variability",
    "external_forcing.co2_fertilization",
    "external_forcing.nitrogen_deposition"
]

# Combined functors (landscape + regional specific)
LAYER_FUNCTORS: List[str] = LANDSCAPE_FUNCTORS + REGIONAL_FUNCTORS

# Variables exposed by regional scale
REGIONAL_VARIABLES: Dict[str, Dict[str, Any]] = {
    "carbon_balance": {
        "standard_name": "surface_carbon_dioxide_mole_flux",
        "units": "mol m-2 s-1",
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Regional carbon balance (sources - sinks)",
        "cell_methods": "time: mean area: mean region: sum",
        "aggregation_method": "area_weighted_mean"
    },
    
    "water_balance": {
        "standard_name": "water_flux_into_sea_water",
        "units": "kg m-2 s-1", 
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Regional water balance (P - ET - R)",
        "cell_methods": "time: mean area: mean region: sum",
        "aggregation_method": "area_weighted_mean"
    },
    
    "nitrogen_balance": {
        "standard_name": None,
        "units": "mol m-2 s-1",
        "variable_type": "extensive", 
        "spatial_basis": "area_density",
        "description": "Regional nitrogen balance; no CF v94 standard name for a regional nitrogen balance",
        "cell_methods": "time: mean area: mean region: sum",
        "aggregation_method": "area_weighted_mean"
    },
    
    "albedo_regional": {
        "standard_name": "surface_albedo",
        "units": "1",
        "variable_type": "intensive",
        "description": "Regional effective surface albedo",
        "cell_methods": "time: mean area: area_weighted_mean region: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "surface_temperature_regional": {
        "standard_name": "surface_temperature", 
        "units": "K",
        "variable_type": "intensive",
        "description": "Regional mean surface temperature",
        "cell_methods": "time: mean area: mean region: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "vegetation_carbon": {
        "standard_name": "vegetation_mass_content_of_carbon",
        "units": "kg C m-2",
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Total vegetation carbon per unit area (mass of carbon)",
        "cell_methods": "time: mean area: mean region: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "soil_carbon": {
        "standard_name": "soil_mass_content_of_carbon",
        "units": "kg C m-2", 
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Total soil carbon per unit area (mass of carbon)",
        "cell_methods": "time: mean area: mean region: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "biome_transition_rate": {
        "standard_name": None,
        "units": "s-1",
        "variable_type": "intensive",
        "description": "Rate of biome transition/vegetation change; no CF v94 standard name for a biome transition rate",
        "cell_methods": "time: mean region: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "climate_sensitivity": {
        "standard_name": None,
        "units": "K", 
        "variable_type": "intensive",
        "description": "Regional climate sensitivity to ecosystem change; no CF v94 standard name for this sensitivity",
        "cell_methods": "time: mean region: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "npp_regional": {
        "standard_name": "net_primary_productivity_of_biomass_expressed_as_carbon",
        "units": "kg C m-2 s-1",
        "variable_type": "extensive", 
        "spatial_basis": "area_density",
        "description": "Regional net primary productivity (mass of carbon)",
        "cell_methods": "time: mean area: mean region: mean",
        "aggregation_method": "area_weighted_mean"
    }
}

# Combined variables (landscape + regional specific)
EXPOSED_VARIABLES: Dict[str, Dict[str, Any]] = {**LANDSCAPE_VARIABLES, **REGIONAL_VARIABLES}

# Coordinate variables for regional scale
COORDINATE_VARIABLES: Dict[str, Dict[str, Any]] = {
    "region_id": {
        "standard_name": "region_identifier", 
        "units": "1",
        "description": "Unique identifier for regional units"
    },
    
    "biome_type": {
        "standard_name": "biome_type",
        "units": "1",
        "description": "Biome classification (e.g., tropical forest, grassland)"
    },
    
    "climate_zone": {
        "standard_name": "climate_zone_classification",
        "units": "1", 
        "description": "Koeppen-Geiger climate classification"
    },
    
    "continent": {
        "standard_name": "continent_name",
        "units": "1",
        "description": "Continental region name"
    }
}

# Scale coupling information
SCALE_COUPLING: Dict[str, Any] = {
    "downscaling_source": "landscape",
    "upscaling_target": None,  # Regional is the top scale
    "aggregation_methods": {
        "carbon_balance": "area_weighted_mean",
        "water_balance": "area_weighted_mean", 
        "nitrogen_balance": "area_weighted_mean",
        "albedo_regional": "area_weighted_mean",
        "surface_temperature_regional": "area_weighted_mean",
        "vegetation_carbon": "area_weighted_mean",
        "soil_carbon": "area_weighted_mean", 
        "npp_regional": "area_weighted_mean"
    },
    "conservation_variables": [
        "carbon_balance", "water_balance", "nitrogen_balance",
        "vegetation_carbon", "soil_carbon", "npp_regional"
    ],
    "climate_feedbacks": [
        "albedo_regional", "surface_temperature_regional",
        "climate_sensitivity"
    ]
}