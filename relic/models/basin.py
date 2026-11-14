"""
Basin Scale Model Configuration

Defines functors and variables exposed at the basin/watershed scale.
Represents complete watersheds with integrated water and biogeochemical cycles.

Scale Characteristics:
    - Spatial extent: Small to medium watersheds (1km - 100km) 
    - Key processes: Basin water balance, biogeochemical export, climate integration
    - Primary variables: Basin discharge, nutrient export, water yield

Mathematical Expression:
    Basin scale processes:
    Q_basin = sum(hillslope_discharge)               [Basin discharge]  
    Water_yield = Q_basin / Precipitation            [Water yield efficiency]
    N_export = sum(hillslope_N_flux * area)         [Nitrogen export]
    C_export = sum(hillslope_C_flux * area)         [Carbon export]

References:
    - Band, L.E. et al. (2001). RHESSys: Regional Hydro-Ecologic Simulation System
    - Tague, C.L. & Band, L.E. (2004). RHESSys watershed model
    - Sivapalan, M. (2003). Process complexity at hillslope scale
"""

from typing import List, Dict, Any

# Import landscape scale components  
from .landscape import LAYER_FUNCTORS as LANDSCAPE_FUNCTORS
from .landscape import EXPOSED_VARIABLES as LANDSCAPE_VARIABLES

# Additional functors active at basin scale
BASIN_FUNCTORS: List[str] = [
    # Basin-scale water balance
    "water_balance.basin_integrated",
    "discharge.basin_outlet", 
    "water_yield.efficiency_calculation",
    
    # Biogeochemical export
    "nitrogen_export.basin_integrated",
    "carbon_export.basin_integrated", 
    "phosphorus_export.basin_integrated",
    
    # Climate integration
    "climate_response.basin_sensitivity",
    "water_stress.basin_indicator"
]

# Combined functors (landscape + basin specific)
LAYER_FUNCTORS: List[str] = LANDSCAPE_FUNCTORS + BASIN_FUNCTORS

# Variables exposed at basin scale
EXPOSED_VARIABLES: Dict[str, Dict[str, Any]] = {
    **LANDSCAPE_VARIABLES,  # Inherit all landscape variables
    
    # Basin hydrological variables
    "basin_discharge": {
        "units": "m3 s-1",
        "standard_name": "water_volume_transport_in_river_channel", 
        "variable_type": "extensive",
        "spatial_basis": "cell_total",
        "description": "Basin outlet discharge",
        "aggregation_method": "sum"
    },
    
    "water_yield": {
        "units": "1",
        "standard_name": None,
        "variable_type": "intensive", 
        "description": "Basin water yield (Q/P); no CF v94 standard name for a water-yield ratio",
        "aggregation_method": "area_weighted_mean"
    },
    
    "runoff_ratio": {
        "units": "1", 
        "standard_name": None,
        "variable_type": "intensive",
        "description": "Basin runoff ratio; no CF v94 standard name for a runoff ratio",
        "aggregation_method": "area_weighted_mean" 
    },
    
    # Basin biogeochemical export
    "nitrogen_export": {
        "units": "kg N s-1",
        "standard_name": "nitrogen_mass_transport_in_river_channel",
        "variable_type": "extensive",
        "spatial_basis": "cell_total",
        "description": "Basin nitrogen export flux (total nitrogen mass transported in the river channel at the outlet)",
        "aggregation_method": "sum"
    },
    
    "carbon_export": {
        "units": "kg C s-1",
        "standard_name": "carbon_mass_transport_in_river_channel",
        "variable_type": "extensive",
        "spatial_basis": "cell_total",
        "description": "Basin carbon export flux (total carbon mass transported in the river channel at the outlet)",
        "aggregation_method": "sum"
    },
    
    "phosphorus_export": {
        "units": "kg P s-1",
        "standard_name": "phosphorus_mass_transport_in_river_channel",
        "variable_type": "extensive", 
        "spatial_basis": "cell_total",
        "description": "Basin phosphorus export flux (total phosphorus mass transported in the river channel at the outlet)",
        "aggregation_method": "sum"
    },
    
    # Basin-scale indicators
    "basin_water_storage": {
        "units": "m",
        "standard_name": None,
        "variable_type": "intensive",
        "description": "Basin-averaged water storage (liquid water equivalent depth); no CF v94 standard name for total basin storage",
        "aggregation_method": "area_weighted_mean"
    },
    
    "basin_et_efficiency": {
        "units": "1",
        "standard_name": None,
        "variable_type": "intensive",
        "description": "Basin evapotranspiration efficiency; no CF v94 standard name for this ratio",
        "aggregation_method": "area_weighted_mean"
    },
    
    "basin_temperature": {
        "units": "K",
        "standard_name": "air_temperature",
        "variable_type": "intensive",
        "description": "Basin-averaged air temperature",
        "aggregation_method": "area_weighted_mean"
    }
}

# Scale-specific constants
CHARACTERISTIC_LENGTH = 10000.0   # m, typical basin length scale
MIN_BASIN_AREA = 1.0e6           # m2, minimum area for basin scale (1 km2)
