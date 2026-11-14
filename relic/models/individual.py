"""
Individual Scale Model Configuration

Defines functors and variables exposed at the individual organism/leaf scale.
This is the finest spatial resolution in the hierarchical model structure.

Scale Characteristics:
    - Spatial extent: Single plant/leaf (~cm to m scale)
    - Key processes: Photosynthesis, transpiration, leaf respiration
    - Primary variables: GPP, LAI, leaf temperature, stomatal conductance

Mathematical Expression:
    Individual scale processes:
    GPP = A_n(I, T, C_i, g_s)  [Farquhar-von Caemmerer-Berry model]
    ET = g_s * VPD / P         [Penman-Monteith transpiration]
    R_leaf = R_d(T)            [Dark respiration]

References:
    - Farquhar, G.D. et al. (1980). A biochemical model of photosynthetic CO2 assimilation
    - Ball, J.T. et al. (1987). A model predicting stomatal conductance
    - Medlyn, B.E. et al. (2011). Reconciling the optimal and empirical approaches
"""

from typing import List, Dict, Any

# Functors active at individual scale
LAYER_FUNCTORS: List[str] = [
    # Core leaf-level processes
    "photosynthesis.basic",
    "leaf_respiration.basic", 
    "stomatal_conductance.medlyn",
    
    # Energy balance
    "leaf_temperature.energy_balance",
    
    # Phenology (individual leaf/plant level)
    "phenology.leaf_development",
]

# Variables exposed by individual scale. Leaf-level GPP is the carbon-mass flux under
# its canonical RELIC name gross_primary_productivity (SSOT contract unit kg C m-2 s-1,
# CF v94 gross_primary_productivity_of_biomass_expressed_as_carbon); the short name gpp
# belongs to the SSOT as the molar flux in umol m-2 s-1, and the two convert through
# UnitConverter carbon_flux (umol m-2 s-1 <-> kg C m-2 s-1 with the SSOT carbon molar mass).
EXPOSED_VARIABLES: Dict[str, Dict[str, Any]] = {
    "gross_primary_productivity": {
        "standard_name": "gross_primary_productivity_of_biomass_expressed_as_carbon",
        "units": "kg C m-2 s-1",
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Gross primary productivity at leaf level (mass of carbon)",
        "cell_methods": "time: mean (individual: sum)",
        "aggregation_method": "area_weighted_mean"
    },
    
    "leaf_respiration": {
        "standard_name": "surface_upward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_plant_respiration_in_leaves",
        "units": "kg C m-2 s-1",
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Leaf dark respiration rate (mass of carbon, upward positive)",
        "cell_methods": "time: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "transpiration": {
        "standard_name": "transpiration_flux",
        "units": "kg m-2 s-1", 
        "variable_type": "extensive",
        "spatial_basis": "area_density",
        "description": "Individual plant transpiration flux",
        "cell_methods": "time: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "lai": {
        "standard_name": "leaf_area_index",
        "units": "1",
        "variable_type": "intensive",
        "description": "Leaf area index per individual",
        "cell_methods": "time: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "leaf_temperature": {
        "standard_name": "surface_temperature",
        "units": "K",
        "variable_type": "intensive", 
        "description": "Individual leaf surface temperature",
        "cell_methods": "time: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    # CF v94 (relic/data/standards/cf_standard_name_table.xml) has no standard name
    # for stomatal conductance; its only stomatal entry,
    # stomatal_resistance_to_transpiration_assuming_fully_open (s m-1), is the
    # inverse of the fully open maximum, not the actual conductance. The former
    # surface_downward_mole_flux_of_carbon_dioxide is a CO2 flux, so no standard
    # name is declared. Units and meaning follow the SSOT entry stomatal_conductance
    # (mol m-2 s-1, conductance to water vapour, as read by ozone_stress).
    "stomatal_conductance": {
        "standard_name": None,
        "units": "mol m-2 s-1",
        "variable_type": "intensive",
        "description": (
            "Leaf-level stomatal conductance to water vapour (SSOT stomatal_conductance); "
            "no CF v94 standard name exists for stomatal conductance"
        ),
        "cell_methods": "time: mean",
        "aggregation_method": "area_weighted_mean"
    },
    
    "phenology_stage": {
        "standard_name": None,
        "units": "1",
        "variable_type": "categorical",
        "description": "Phenological development stage; no CF v94 standard name for a phenological stage",
        "cell_methods": "time: point",
        "aggregation_method": "mode"
    }
}

# Coordinate variables for individual scale
COORDINATE_VARIABLES: Dict[str, Dict[str, Any]] = {
    "individual_id": {
        "standard_name": "individual_plant_identifier",
        "units": "1", 
        "description": "Unique identifier for individual plants"
    },
    
    "leaf_layer": {
        "standard_name": "canopy_layer_index",
        "units": "1",
        "description": "Vertical layer within individual canopy"
    }
}

# Scale coupling information
SCALE_COUPLING: Dict[str, Any] = {
    "upscaling_target": "patch",
    "aggregation_methods": {
        "gross_primary_productivity": "sum",  # Total GPP across individuals
        "lai": "mean", # Average LAI
        "leaf_temperature": "area_weighted_mean",
        "transpiration": "sum"
    },
    "conservation_variables": ["gross_primary_productivity", "leaf_respiration", "transpiration"]
}