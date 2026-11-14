"""Hydrosphere Interface Mapping - CF-1.8 Standard Names

This module defines the canonical variable catalog for hydrosphere processes
using CF-1.8 standard names. It enforces consistent naming, units metadata,
and interface validation across hydrosphere components.

References
----------
- CF Conventions Version 1.8: http://cfconventions.org/cf-conventions/v1.8.0/cf-conventions.html
- CF Standard Names Table: https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
- FAO-56: Crop evapotranspiration - Guidelines for computing crop water requirements (1998)
- WMO Guide to Meteorological Instruments and Methods of Observation (2018)

Architecture Integration
------------------------
This mapping integrates with:
- relic.validation.cross_domain_validator.py for interface validation
- Hydrosphere functor metadata for bounded CF variable fields
- Parameter system for unit consistency checks
- Documentation system for reference generation
"""

from typing import (
    Any,
    Dict,
    List,
    Optional
)
from dataclasses import dataclass, field
from enum import Enum


# Parameter management imports
class VariableCategory(Enum):
    """Categories for hydrosphere variables following process grouping."""
    FORCING = "forcing"              # External drivers (precipitation, radiation)
    STATE = "state"                  # System state variables (soil moisture, temperature)
    FLUX = "flux"                    # Process fluxes (evaporation, runoff, drainage)
    DIAGNOSTIC = "diagnostic"        # Derived quantities (water stress, saturation)
    PARAMETER = "parameter"          # Model parameters (hydraulic properties)


@dataclass
class VariableMapping:
    """
    Variable metadata definition with bounded CF fields.

    Attributes
    ----------
    standard_name : str
        Canonical variable name (CF-1.8 standard when available)
    units : str
        SI units following CF conventions
    description : str
        Physical meaning and context
    category : VariableCategory
        Process category classification
    aliases : List[str]
        Alternative names for documentation or UI display
    cf_standard_name : Optional[str]
        Official CF standard name if exists
    """
    standard_name: str
    units: str
    description: str
    category: VariableCategory
    aliases: List[str] = field(default_factory=list)
    cf_standard_name: Optional[str] = None
    
    def __post_init__(self) -> None:
        if self.aliases is None:
            self.aliases = []


# Core Hydrosphere Variable Mappings
# =================================

# Forcing Variables (External Drivers)
FORCING_MAPPINGS = [
    # The one entry of the surface precipitation mass flux (liquid plus solid, CF
    # precipitation_flux); the bare name 'precipitation' is refused by get_standard_name, and
    # the liquid part is rain_flux (CF rainfall_flux).
    VariableMapping(
        standard_name="precipitation_flux",
        units="kg m-2 s-1",
        description="Precipitation flux at surface (liquid plus solid)",
        category=VariableCategory.FORCING,
        aliases=["precipitation_rate"],
        cf_standard_name="precipitation_flux"
    ),
    VariableMapping( 
        standard_name="net_radiation",
        units="W m-2",
        description="Net radiative flux at surface",
        category=VariableCategory.FLUX,
        aliases=["surface_net_downward_radiative_flux", "rnet", "radiation_balance"],
        cf_standard_name="surface_net_downward_radiative_flux"
    ),
    VariableMapping(
        standard_name="shortwave_down",
        units="W m-2",
        description="Surface downwelling shortwave radiative flux (global radiation at the lower boundary of the atmosphere; the incident shortwave of the snow albedo port); this variable holds the CF identity surface_downwelling_shortwave_flux_in_air",
        category=VariableCategory.FLUX,
        aliases=["shortwave_radiation"],
        cf_standard_name="surface_downwelling_shortwave_flux_in_air",
    ),
    VariableMapping(
        standard_name="soil_heat_flux",
        units="W m-2",
        description="Signed soil-surface heat flux supplied as forcing; positive into soil, with any plate-to-surface storage correction declared by the source",
        category=VariableCategory.FORCING,
        aliases=[],
        cf_standard_name="downward_heat_flux_in_soil",
    ),
    VariableMapping(
        standard_name="air_temperature", 
        units="K",
        description="Air temperature at reference height",
        category=VariableCategory.FORCING,
        aliases=["temp_air", "ta"],
        cf_standard_name="air_temperature"
    ),
    VariableMapping(
        standard_name="relative_humidity",
        units="1",
        description="Relative humidity (0-1)",
        category=VariableCategory.FORCING,
        aliases=["humidity", "rh", "rel_hum"],
        cf_standard_name="relative_humidity"
    ),
    VariableMapping(
        standard_name="wind_speed",
        units="m s-1", 
        description="Wind speed at reference height",
        category=VariableCategory.FORCING,
        aliases=["windspeed", "u_wind", "wind_velocity"],
        cf_standard_name="wind_speed"
    ),
    VariableMapping(
        standard_name="leaf_area_index",
        units="m2 m-2",
        description="Leaf area per unit ground area (read by canopy interception)",
        category=VariableCategory.STATE,
        aliases=["lai"],
        cf_standard_name="leaf_area_index"
    ),
    VariableMapping(
        standard_name="atmospheric_pressure", 
        units="Pa",
        description="Surface atmospheric pressure",
        category=VariableCategory.FORCING,
        aliases=["pressure", "atm_pressure", "surface_pressure", "air_pressure", "surface_air_pressure"],
        cf_standard_name="surface_air_pressure"
    ),
    VariableMapping(
        standard_name="solar_zenith_angle",
        units="rad",
        description="Solar zenith angle at surface (CF v94 solar_zenith_angle, canonical degree)",
        category=VariableCategory.FORCING,
        aliases=["zenith_angle", "solar_zenith"],
        cf_standard_name="solar_zenith_angle"
    ),
]

# State Variables (System State)
STATE_MAPPINGS = [
    VariableMapping(
        standard_name="soil_moisture",
        units="m3 m-3",
        description=(
            "Volumetric liquid soil water content (internal name)."
            " No CF name: CF v94 volume_fraction_of_water_in_soil is an alias of volume_fraction_of_condensed_water_in_soil (liquid water and ice); RELIC carries liquid water only."
        ),
        category=VariableCategory.STATE,
        aliases=["theta"],
        cf_standard_name=None
    ),
    VariableMapping(
        standard_name="soil_water_storage",
        units="kg m-2",
        description="Soil water storage per unit ground area",
        category=VariableCategory.STATE,
        aliases=["soil_water_mass", "total_soil_water_mass"]
    ),
    VariableMapping(
        standard_name="soil_temperature",
        units="K",
        description="Soil temperature",
        category=VariableCategory.FORCING,
        aliases=["temp_soil", "tsoil"],
        cf_standard_name="soil_temperature"
    ),
    VariableMapping(
        standard_name="snow_surface_temperature",
        units="K",
        description="Snow surface temperature",
        category=VariableCategory.STATE,
        aliases=["snow_temp", "snow_skin_temperature"]
    ),
    VariableMapping(
        standard_name="snow_water_equivalent",
        units="mm",
        description="Snow water equivalent depth (CF v94 lwe_thickness_of_surface_snow_amount, canonical m; stored in mm)",
        category=VariableCategory.STATE,
        aliases=["swe", "snow_water_depth"],
        cf_standard_name="lwe_thickness_of_surface_snow_amount"
    ),
    VariableMapping(
        standard_name="soil_moisture_profile",
        units="m3 m-3",
        description="Soil moisture profile by layer",
        category=VariableCategory.STATE,
        aliases=["theta_profile", "soil_water_profile"]
    ),
    VariableMapping( 
        standard_name="groundwater_storage",
        units="mm",
        description="Groundwater storage depth equivalent",
        category=VariableCategory.STATE,
        aliases=["storage", "water_storage", "gw_storage"]
    ),
    # New state variables from GroundwaterDynamics module
    VariableMapping(
        standard_name="water_table_depth",
        units="m",
        description="Depth to water table below ground surface",
        category=VariableCategory.STATE,
        aliases=["wt_depth", "water_table"],
        cf_standard_name="water_table_depth"
    ),
    # New state variables from Canopy Interception Enhancement (P2)
    VariableMapping(
        standard_name="canopy_storage",
        units="mm",
        description="Water storage on canopy surfaces from interception",
        category=VariableCategory.STATE,
        aliases=["canopy_intercepted_water_content", "intercepted_water", "canopy_water_content"]
    )
]

# Flux Variables (Process Rates)  
FLUX_MAPPINGS = [
    VariableMapping(
        standard_name="evapotranspiration_flux", 
        units="kg m-2 s-1",
        description="Combined evaporation and transpiration flux (upward positive, dew and condensation negative); this variable holds the CF identity water_evapotranspiration_flux (contract evapotranspiration_flux, written by the ET models or an external adapter)",
        category=VariableCategory.FLUX,
        aliases=["et", "evap_transp", "latent_heat_flux_equivalent"],
        cf_standard_name="water_evapotranspiration_flux"
    ),
    VariableMapping(
        standard_name="snow_melt_flux",
        units="kg m-2 s-1",
        description="Snow melt flux from snowpack to liquid water",
        category=VariableCategory.FLUX,
        aliases=["snowmelt_flux", "snow_melt", "melt_flux"],
        cf_standard_name="surface_snow_melt_flux"
    ),
    VariableMapping(
        standard_name="rain_flux",
        units="kg m-2 s-1",
        description="Liquid precipitation flux after partitioning",
        category=VariableCategory.FLUX,
        aliases=["rainfall_flux", "rainfall_rate", "liquid_precipitation_flux"],
        cf_standard_name="rainfall_flux"
    ),
    VariableMapping(
        standard_name="snowfall",
        units="kg m-2 s-1",
        description="Snowfall mass flux after the snow_pack rain/snow phase partition (CF v94 snowfall_flux)",
        category=VariableCategory.FLUX,
        aliases=["snowfall_rate", "snow_precipitation"],
        cf_standard_name="snowfall_flux"
    ),
    VariableMapping(
        standard_name="sublimation_flux",
        units="kg m-2 s-1",
        description="Snow sublimation flux to atmosphere",
        category=VariableCategory.FLUX,
        aliases=["snow_sublimation_flux", "sublimation_rate"],
        cf_standard_name="surface_snow_sublimation_flux"
    ),
    VariableMapping(
        standard_name="surface_runoff_flux",
        units="kg m-2 s-1", 
        description="Surface runoff flux",
        category=VariableCategory.FLUX,
        aliases=["surface_runoff", "runoff_surface", "overland_flow", "surface_flow"],
        cf_standard_name="surface_runoff_flux"
    ),
    VariableMapping(
        standard_name="subsurface_runoff_flux",
        units="kg m-2 s-1",
        description="Lateral subsurface runoff flux (baseflow/interflow)", 
        category=VariableCategory.FLUX,
        aliases=["interflow", "subsurface_flow", "lateral_subsurface_flow"],
        cf_standard_name="subsurface_runoff_flux"  
    ),
    # Hortonian ponding topology transition (G09-TOPO): the event detector
    # reads the cumulative stochastic rainfall depth and the connectivity
    # flag; the ponding store holds the ponded depth and the per-step
    # infiltration-excess input (units follow the SCM contracts).
    VariableMapping(
        standard_name="stochastic_rainfall_depth",
        units="mm",
        description="Cumulative marked-Poisson rainfall depth",
        category=VariableCategory.STATE,
        aliases=["cumulative_rainfall_depth"],
        cf_standard_name=None,
    ),
    VariableMapping(
        standard_name="surface_ponded_depth",
        units="mm",
        description="Ponded water depth of the connected surface store",
        category=VariableCategory.STATE,
        aliases=["ponded_depth", "surface_water_depth"],
        cf_standard_name=None,
    ),
    VariableMapping(
        standard_name="surface_ponding_excess",
        units="mm",
        description="Infiltration-excess depth routed to the ponding store in one step",
        category=VariableCategory.STATE,
        aliases=["ponding_excess_depth", "infiltration_excess_depth"],
        cf_standard_name=None,
    ),
    VariableMapping(
        standard_name="surface_ponding_active",
        units="1",
        description="Ponding connectivity flag (0 disconnected, 1 connected)",
        category=VariableCategory.STATE,
        aliases=["ponding_active", "ponding_connectivity_flag"],
        cf_standard_name=None,
    ),
    VariableMapping(
        standard_name="infiltration_flux",
        units="kg m-2 s-1",
        description=(
            "Infiltration flux into soil. No CF name: CF v94 has no infiltration entry."
        ),
        category=VariableCategory.FLUX,
        aliases=[],
        cf_standard_name=None
    ),
    VariableMapping( 
        standard_name="drainage_flux",
        units="kg m-2 s-1",
        description="Deep drainage flux (bottom boundary)",
        category=VariableCategory.FLUX,
        aliases=["deep_drainage", "percolation", "bottom_boundary_flux"]
    ),
    VariableMapping(
        standard_name="soil_percolation_rate",
        units="mm s-1",
        description="Soil percolation rate (rate, not per-step depth)",
        category=VariableCategory.FLUX,
        aliases=[],
    ),
    VariableMapping(
        standard_name="lateral_inflow",
        units="kg m-2 s-1",
        description="Lateral groundwater mass influx",
        category=VariableCategory.FLUX,
        aliases=[]
    ),
    VariableMapping(
        standard_name="groundwater_pumping",
        units="kg m-2 s-1",
        description="Managed groundwater abstraction mass flux",
        category=VariableCategory.FLUX,
        aliases=[]
    ),
    # New flux variables from CapillaryRiseTransport module
    VariableMapping(
        standard_name="capillary_rise_flux",
        units="kg m-2 s-1",
        description="Upward water flux from groundwater via capillary rise",
        category=VariableCategory.FLUX,
        aliases=[]
    ),
    VariableMapping(
        standard_name="capillary_rise_flux_profile", 
        units="kg m-2 s-1",
        description="Depth-resolved capillary rise flux profile",
        category=VariableCategory.FLUX,
        aliases=["capillary_profile", "layer_capillary_flux"]
    ),
    VariableMapping(
        standard_name="total_capillary_water_flux",
        units="kg m-2 s-1", 
        description="Total integrated capillary water input",
        category=VariableCategory.FLUX,
        aliases=["total_capillary", "integrated_capillary_flux"]
    ),
    # New flux variables from GroundwaterDynamics module
    # English comments only
    VariableMapping(
        standard_name="baseflow_flux",
        units="kg m-2 s-1",
        description="Baseflow mass flux from groundwater to stream network",
        category=VariableCategory.FLUX,
        aliases=[]
    ),
    VariableMapping(
        standard_name="groundwater_storage_change_rate",
        units="mm s-1",
        description="Rate of change in groundwater storage", 
        category=VariableCategory.FLUX,
        aliases=["gw_storage_rate", "dstorage_dt"]
    ),
    VariableMapping(
        standard_name="water_table_depth_change_rate", 
        units="m s-1",
        description="Rate of change in water table depth",
        category=VariableCategory.FLUX,
        aliases=["wt_change_rate", "water_table_rate", "dwt_dt", "water_table_change_rate"]
    ),
    VariableMapping(
        standard_name="net_groundwater_flux",
        units="mm s-1",
        description="Net groundwater flux (drainage + lateral - pumping)",
        category=VariableCategory.FLUX,
        aliases=["net_gw_flux", "groundwater_balance", "gw_net_flux"]
    ),
    # New flux variables from Canopy Interception Enhancement (P2)
    VariableMapping(
        standard_name="evaporation_from_interception",
        units="kg m-2 s-1",
        description="Evaporation flux from intercepted water on canopy surfaces",
        category=VariableCategory.FLUX,
        aliases=["interception_evaporation", "canopy_evaporation", "wet_canopy_evaporation"]
    ),
    VariableMapping( 
        standard_name="transpiration_flux",
        units="kg m-2 s-1",
        description="Plant transpiration flux (separated from interception evaporation)",
        category=VariableCategory.FLUX,
        aliases=["plant_transpiration", "dry_canopy_transpiration", "stomatal_flux"]
    ),
    VariableMapping(
        standard_name="precipitation_throughfall_flux", 
        units="kg m-2 s-1",
        description="Precipitation flux reaching soil after canopy interception",
        category=VariableCategory.FLUX,
        aliases=["throughfall", "net_precipitation", "soil_precipitation_input"]
    )
]

# Diagnostic Variables (Derived Quantities)
DIAGNOSTIC_MAPPINGS = [
    VariableMapping(
        standard_name="snow_depth",
        units="m",
        description="Thickness of the snowpack on the ground diagnosed by snow_pack from the snow water equivalent and the bulk snow density (CF v94 surface_snow_thickness)",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["snow_height", "snowpack_depth"],
        cf_standard_name="surface_snow_thickness"
    ),
    VariableMapping(
        standard_name="diagnostics.evapotranspiration",
        units="dimensionless",
        description="Evapotranspiration diagnostics payload",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["et_diagnostics", "diagnostics.et"]
    ),
    VariableMapping(
        standard_name="vegetation_water_stress_factor", 
        units="1",
        description="Water stress factor for vegetation (0=stress, 1=no stress)",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["stress", "water_limitation", "beta_factor"]
    ),
    VariableMapping(
        standard_name="soil_moisture_saturation_deficit",
        units="m3 m-3", 
        description="Deficit from soil saturation",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["deficit", "moisture_deficit"]
    ),
    VariableMapping(
        standard_name="water_stress_factor",
        units="1",
        description="Soil water stress factor",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["beta", "stress_factor"]
    ),
    VariableMapping(
        standard_name="plant_available_water_content",
        units="m3 m-3",
        description="Plant-available water content above wilting point",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["awc", "available_water", "plant_water"]
    ),
    VariableMapping(
        standard_name="vapor_pressure_deficit",
        units="Pa",
        description="Atmospheric vapor pressure deficit",
        category=VariableCategory.FORCING,
        aliases=["humidity_deficit", "saturation_deficit_air"],
        cf_standard_name="water_vapor_saturation_deficit_in_air"
    ),
    VariableMapping(
        standard_name="surface_resistance",
        units="s m-1",
        description="Surface resistance to water vapor transfer",
        category=VariableCategory.FORCING,
        aliases=["rs", "canopy_resistance", "stomatal_resistance", "surface_resistance_to_water_vapor"]
    ),
    # Parameter variables from hydrosphere modules
    VariableMapping(
        standard_name="aerodynamic_resistance",
        units="s m-1",
        description="Aerodynamic resistance to vapor transfer",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="degree_day_factor",
        units="mm K-1 day-1",
        description="Degree day melt factor for snow",
        category=VariableCategory.PARAMETER,
        aliases=["ddf", "snow_degree_day_factor"]
    ),
    VariableMapping(
        standard_name="temperature_threshold_warm",
        units="K",
        description="Melt temperature threshold for snow",
        category=VariableCategory.PARAMETER,
        aliases=["snow_melt_threshold", "melt_temperature_threshold"]
    ),
    VariableMapping(
        standard_name="rain_snow_temperature_threshold",
        units="K",
        description="Rain snow partitioning temperature threshold",
        category=VariableCategory.PARAMETER,
        aliases=["rain_snow_threshold", "snow_rain_threshold"]
    ),
    VariableMapping(
        standard_name="snow_density",
        units="kg m-3",
        description="Bulk snow density for SWE to depth conversion",
        category=VariableCategory.PARAMETER,
        aliases=["rho_snow", "bulk_snow_density"]
    ),
    VariableMapping(
        standard_name="alpha_pt",
        units="dimensionless",
        description="Priestley-Taylor coefficient",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="curve_number",
        units="dimensionless",
        description="SCS curve number for runoff estimation",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="curve_number_dynamic",
        units="dimensionless",
        description="Dynamic curve number override from state",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="enable_dynamic_curve_number",
        units="dimensionless",
        description="Enable dynamic curve number override flag",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="scs_storage_minimum",
        units="mm",
        description="Minimum SCS storage parameter",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="moisture_adjustment_factor",
        units="dimensionless",
        description="Moisture adjustment factor for SCS retention scaling",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="infiltration_moisture_reduction_factor",
        units="dimensionless",
        description="Soil moisture reduction factor for infiltration capacity",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="topmodel_moisture_reference",
        units="m3 m-3",
        description="Reference soil moisture for TOPMODEL scaling",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="topmodel_moisture_factor_min",
        units="dimensionless",
        description="Minimum moisture scaling factor for TOPMODEL baseflow",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="topmodel_moisture_factor_max",
        units="dimensionless",
        description="Maximum moisture scaling factor for TOPMODEL baseflow",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="water_table_depth_scale",
        units="m",
        description="Water table depth scaling factor",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="water_table_depth_max",
        units="m",
        description="Maximum diagnostic water table depth",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="capillary_saturation_threshold_factor",
        units="dimensionless",
        description="Saturation threshold factor for capillary rise reduction",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="capillary_saturation_bandwidth_factor",
        units="dimensionless",
        description="Saturation bandwidth factor for capillary rise reduction",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="stomatal_conductance",
        units="mol m-2 s-1",
        description=(
            "Leaf-level stomatal conductance to water vapour in the unit of its readers "
            "ozone_stress and oxygen_18 (coupling.scales.defaults.variables); no relic Functor "
            "writes it, so it is supplied externally"
        ),
        category=VariableCategory.FORCING,
        aliases=["gs", "stomatal_gs"]
    ),
    VariableMapping(
        standard_name="canopy_conductance",
        units="m s-1",
        description="Canopy conductance to water vapor",
        category=VariableCategory.STATE,
        aliases=["gc", "canopy_gs"]
    ),
    VariableMapping(
        standard_name="saturated_area_threshold",
        units="m3 m-3",
        description="Soil moisture threshold for saturation excess runoff",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="recession_coefficient",
        units="s-1",
        description="Baseflow recession coefficient",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="topographic_index",
        units="dimensionless",
        description="Mean topographic index",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="transmissivity",
        units="m2 s-1",
        description="Soil transmissivity",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="threshold_moisture",
        units="m3 m-3",
        description="Moisture threshold for interflow generation",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="soil_depth",
        units="m",
        description="Effective soil layer depth",
        category=VariableCategory.FORCING
    ),
    VariableMapping(
        standard_name="field_capacity",
        units="m3 m-3",
        description="Field capacity water content",
        category=VariableCategory.FORCING
    ),
    VariableMapping(
        standard_name="wilting_point",
        units="m3 m-3",
        description="Permanent wilting point",
        category=VariableCategory.FORCING
    ),
    VariableMapping(
        standard_name="porosity",
        units="m3 m-3",
        description="Soil porosity (saturation water content)",
        category=VariableCategory.FORCING
    ),
    VariableMapping(
        standard_name="pressure_head",
        units="m",
        description="Soil water pressure head",
        category=VariableCategory.FORCING
    ),
    VariableMapping(
        standard_name="soil_layers",
        units="dimensionless",
        description="Number of soil layers",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="theta_residual",
        units="m3 m-3",
        description="Residual water content",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="theta_saturated",
        units="m3 m-3",
        description="Saturated water content",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="air_entry_pressure",
        units="m",
        description="Brooks-Corey air entry pressure head",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="pore_size_index",
        units="dimensionless",
        description="Brooks-Corey pore size index",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="surface_porosity",
        units="dimensionless",
        description="Surface porosity",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="porosity_decay_length",
        units="m",
        description="Porosity decay length scale",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="max_capillary_height",
        units="m",
        description="Maximum capillary rise height",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="max_drainage_flux",
        units="kg m-2 s-1",
        description="Maximum drainage flux limit",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="residual_water_content",
        units="m3 m-3",
        description="Residual water content for percolation model",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="saturated_hydraulic_conductivity",
        units="m s-1",
        description="Saturated hydraulic conductivity",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="saturated_water_content",
        units="m3 m-3",
        description="Saturated water content",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="soil_layer_thickness",
        units="m",
        description="Soil layer thickness",
        category=VariableCategory.PARAMETER
    ),
    VariableMapping(
        standard_name="vg_alpha",
        units="m-1",
        description="van Genuchten alpha parameter",
        category=VariableCategory.PARAMETER,
        aliases=["alpha_vg"]
    ),
    VariableMapping(
        standard_name="vg_n",
        units="dimensionless",
        description="van Genuchten n parameter",
        category=VariableCategory.PARAMETER,
        aliases=["n_vg"]
    ),
    # New diagnostic variables from CapillaryRiseTransport module
    VariableMapping(
        standard_name="effective_capillary_fringe_height",
        units="m",
        description="Effective height of capillary fringe above water table",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["capillary_height", "fringe_height", "capillary_zone_height"]
    ),
    VariableMapping(
        standard_name="capillary_water_availability_factor",
        units="1",
        description="Water availability factor for capillary processes (0-1)",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["capillary_factor", "capillary_availability_index", "capillary_potential"]
    ),
    # New diagnostic variables from GroundwaterDynamics module
    VariableMapping(
        standard_name="groundwater_storage_deficit",
        units="mm",
        description="Deficit in groundwater storage capacity",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["gw_deficit", "groundwater_deficit", "storage_capacity_deficit"]
    ),
    VariableMapping(
        standard_name="groundwater_saturation_fraction",
        units="1",
        description="Fraction of groundwater storage capacity that is filled (0-1)",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["sat_fraction", "gw_saturation", "groundwater_fullness"]
    ),
    VariableMapping(
        standard_name="effective_soil_porosity",
        units="1",
        description="Depth-averaged effective porosity for groundwater storage",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["eff_porosity", "average_porosity", "bulk_porosity"]
    ),
    # New diagnostic variables from Canopy Interception Enhancement (P2)
    VariableMapping(
        standard_name="canopy_water_storage_change_rate",
        units="mm h-1",
        description="Rate of change in canopy intercepted water storage",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["canopy_storage_rate", "interception_rate"]
    ),
    VariableMapping(
        standard_name="groundwater_storage_increment",
        units="mm",
        description="Incremental change in groundwater storage over the step",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["gw_storage_increment", "delta_groundwater_storage"]
    ),
    VariableMapping(
        standard_name="gardner_a_coefficient",
        units="m^(b+1) s-1",
        description="Gardner A coefficient for capillary rise formulation",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["gardner_a", "capillary_a_coefficient"]
    ),
    VariableMapping(
        standard_name="gardner_b_coefficient",
        units="dimensionless",
        description="Gardner b exponent for capillary rise formulation",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["gardner_b", "capillary_b_exponent"]
    ),
    VariableMapping(
        standard_name="gardner_B_coefficient",
        units="dimensionless",
        description="Gardner B coefficient for capillary rise formulation",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["gardner_B", "capillary_B_coefficient"]
    ),
    VariableMapping(
        standard_name="soil_matric_potential",
        units="Pa",
        description="Layer-resolved soil matric potential converted to pressure",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["matric_potential_pa"]
    ),
    VariableMapping(
        standard_name="soil_water_potential",
        units="MPa",
        description="Soil water potential of the soil-water store (matric potential, non-positive)",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["soil_water_store_potential"]
    ),
    VariableMapping(
        standard_name="diagnostics.snow_albedo",
        units="dimensionless",
        description="Snow albedo diagnostics payload",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["snow_albedo_diagnostics"]
    ),
    VariableMapping(
        standard_name="diagnostics.snow_pack",
        units="dimensionless",
        description="Snow pack diagnostics payload",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["snow_pack_diagnostics"]
    ),
    # New diagnostic variables from Snow Albedo Module (P2)
    VariableMapping(
        standard_name="surface_albedo",
        units="1",
        description="Surface albedo including snow and vegetation effects",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["albedo", "reflectance", "surface_reflectance"],
        cf_standard_name="surface_albedo"
    ),
    VariableMapping(
        standard_name="snow_surface_age",
        units="day",
        description="Snow surface age in days for the age-based albedo scheme",
        category=VariableCategory.STATE,
        aliases=["snow_age", "surface_age", "aging_time"]
    ),
    VariableMapping(
        standard_name="snow_nondimensional_age",
        units="1",
        description="BATS nondimensional snow age tau_s with F_age = tau_s/(1+tau_s)",
        category=VariableCategory.STATE,
        aliases=["bats_snow_age", "nondimensional_snow_age"]
    ),
    VariableMapping(
        standard_name="snow_aging_rate",
        units="1",
        description="Snow surface aging rate (per day for age-based scheme)",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["albedo_aging_rate"]
    ),
    VariableMapping(
        standard_name="snow_albedo_reset_flag",
        units="1",
        description="Boolean flag indicating fresh snow albedo reset event",
        category=VariableCategory.DIAGNOSTIC,
        aliases=["snow_reset", "fresh_snow_flag", "albedo_reset"]
    )
]

PHASE8_RUNTIME_MAPPINGS = [
    VariableMapping(
        standard_name="latent_heat_flux",
        units="W m-2",
        description="Latent heat flux associated with evapotranspiration (upward positive, dew and condensation negative); this variable holds the CF identity surface_upward_latent_heat_flux",
        category=VariableCategory.FLUX,
        cf_standard_name="surface_upward_latent_heat_flux",
    ),
    VariableMapping(
        standard_name="percolation_drainage_flux",
        units="kg m-2 s-1",
        description="Drainage flux from vertical percolation",
        category=VariableCategory.FLUX,
    ),
    VariableMapping(
        standard_name="diagnostics.aerodynamic_conductance",
        units="m s-1",
        description="Aerodynamic conductance diagnostic",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.canopy_conductance",
        units="m s-1",
        description="Canopy conductance diagnostic",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.vapor_pressure_deficit",
        units="Pa",
        description="Vapor pressure deficit diagnostic",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.groundwater",
        units="dimensionless",
        description="Groundwater diagnostics payload",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.surface_runoff",
        units="dimensionless",
        description="Surface runoff diagnostics payload",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.subsurface_runoff",
        units="dimensionless",
        description="Subsurface runoff diagnostics payload",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.soil_moisture_bucket",
        units="dimensionless",
        description="Bucket soil moisture diagnostics payload",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.soil_moisture_richards",
        units="dimensionless",
        description="Richards soil moisture diagnostics payload",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.capillary_rise",
        units="dimensionless",
        description="Capillary rise diagnostics payload",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.snow_albedo.albedo_clipping_residual",
        units="dimensionless",
        description="Snow albedo clipping residual diagnostic",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.snow_albedo.snow_absorbed_shortwave_fraction",
        units="dimensionless",
        description="Snow absorbed shortwave fraction diagnostic",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.snow_pack.melt_capacity_residual_mm_s",
        units="mm s-1",
        description="Snow melt capacity residual diagnostic",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.snow_pack.sublimation_capacity_residual_mm_s",
        units="mm s-1",
        description="Snow sublimation capacity residual diagnostic",
        category=VariableCategory.DIAGNOSTIC,
    ),
    VariableMapping(
        standard_name="diagnostics.snow_pack.swe_nonnegative_projection_residual_mm",
        units="mm",
        description="Snow water equivalent nonnegative projection residual",
        category=VariableCategory.DIAGNOSTIC,
    ),
]

for _diagnostic_residual_name, _diagnostic_residual_units in {
    "diagnostics.evapotranspiration.available_energy_residual": "W m-2",
    "diagnostics.evapotranspiration.canopy_storage_residual": "mm",
    "diagnostics.evapotranspiration.negative_et_projection_residual": (
        "kg m-2 s-1"
    ),
    "diagnostics.groundwater.storage_balance_residual": "mm",
    "diagnostics.groundwater.water_table_bounds_residual": "m",
    "diagnostics.groundwater.groundwater_storage_negative_residual": "mm",
    "diagnostics.subsurface_runoff.water_table_depth_clipping_residual": "m",
    "diagnostics.surface_runoff.precipitation_limit_residual": "kg m-2 s-1",
    "diagnostics.surface_runoff.runoff_nonnegative_residual": "kg m-2 s-1",
    "diagnostics.soil_moisture_bucket.clipped_mass_residual": "kg m-2 s-1",
    "diagnostics.soil_moisture_bucket.drainage_limiter_residual": "m s-1",
    "diagnostics.soil_moisture_richards.mass_conservation_error": "dimensionless",
    "diagnostics.soil_moisture_richards.moisture_clipping_residual": "m3 m-3",
    "diagnostics.capillary_rise.conductivity_clipping_residual": "kg m-2 s-1",
    "diagnostics.capillary_rise.saturation_constraint_residual": "kg m-2 s-1",
}.items():
    PHASE8_RUNTIME_MAPPINGS.append(
        VariableMapping(
            standard_name=_diagnostic_residual_name,
            units=_diagnostic_residual_units,
            description=f"Phase 8 runtime diagnostic residual: {_diagnostic_residual_name}",
            category=VariableCategory.DIAGNOSTIC,
        )
    )

# Combined mapping registry
ALL_MAPPINGS = (
    FORCING_MAPPINGS
    + STATE_MAPPINGS
    + FLUX_MAPPINGS
    + DIAGNOSTIC_MAPPINGS
    + PHASE8_RUNTIME_MAPPINGS
)


class HydrosphereInterfaceMapper:
    """
    Central interface mapper for hydrosphere variable names.

    Provides standard-name validation, metadata lookup, and consistency checks.
    """

    def __init__(self) -> None:
        """Initialize mapping registry."""
        self._variable_info: Dict[str, VariableMapping] = {}
        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build internal mapping registry from definitions."""
        for mapping in ALL_MAPPINGS:
            self._variable_info[mapping.standard_name] = mapping

    def get_standard_name(self, name: str) -> str:
        """
        Return the canonical standard name for a variable.

        Parameters
        ----------
        name : str
            Variable name to validate.

        Returns
        -------
        str
            Canonical standard name.

        Raises
        ------
        ValueError
            If the name is not recognized.
        """
        deprecated_aliases = {
            "precipitation": "precipitation_flux",
        }
        if name in deprecated_aliases:
            replacement = deprecated_aliases[name]
            raise ValueError(
                f"Deprecated hydrosphere variable alias '{name}'. "
                f"Use canonical name '{replacement}'."
            )
        if name in self._variable_info:
            return name
        raise ValueError(
            f"Unknown variable name '{name}'. Available names: {self.get_all_names()}"
        )

    def get_variable_info(self, name: str) -> VariableMapping:
        """Get complete variable information."""
        standard_name = self.get_standard_name(name)
        return self._variable_info[standard_name]

    def get_units(self, name: str) -> str:
        """Get SI units for a variable."""
        return self.get_variable_info(name).units

    def get_category(self, name: str) -> VariableCategory:
        """Get process category for a variable."""
        return self.get_variable_info(name).category

    def is_flux_variable(self, name: str) -> bool:
        """Check if a variable represents a flux."""
        return self.get_category(name) == VariableCategory.FLUX

    def validate_interface_consistency(
        self, input_vars: List[str], output_vars: List[str]
    ) -> List[str]:
        """
        Validate interface variable consistency.

        Parameters
        ----------
        input_vars : List[str]
            Input variable names.
        output_vars : List[str]
            Output variable names.

        Returns
        -------
        List[str]
            Validation warnings or errors.
        """
        issues: List[str] = []
        all_vars = input_vars + output_vars
        for var in all_vars:
            try:
                self.get_standard_name(var)
            except ValueError as exc:
                issues.append(f"Unrecognized variable: {exc}")

        water_flux_vars = [
            v
            for v in all_vars
            if self.is_flux_variable(v) and "flux" in self.get_standard_name(v)
        ]
        expected_flux_unit = "kg m-2 s-1"
        for var in water_flux_vars:
            unit = self.get_units(var)
            if unit == expected_flux_unit:
                continue
            if unit == "mm s-1":
                continue
            issues.append(
                f"Water flux variable '{var}' has inconsistent units: {unit} != {expected_flux_unit}"
            )
        return issues

    def get_all_names(self) -> Dict[str, List[str]]:
        """Get available standard names."""
        return {"standard": list(self._variable_info.keys())}

    def create_interface_dict(
        self, variables: List[str], use_standard_names: bool = True
    ) -> Dict[str, str]:
        """Create standardized interface dictionary."""
        _ = use_standard_names
        return {var: self.get_standard_name(var) for var in variables}


# Global mapper instance
hydrosphere_mapper = HydrosphereInterfaceMapper()


class HydrosphereInterfaceMixin:
    """
    Hydrosphere-specific state and update normalization hooks.

    These hooks operate on canonical standard names only.
    """

    def preprocess_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state without alias expansion."""
        return state if isinstance(state, dict) else state

    def postprocess_updates(
        self, original_state: Dict[str, Any], updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return updates without alias remapping."""
        return updates if isinstance(updates, dict) else updates


# Convenience functions for common operations
def standardize_variable_name(name: str) -> str:
    """Convert any variable name to CF-1.8 standard format."""
    return hydrosphere_mapper.get_standard_name(name)

def get_variable_units(name: str) -> str:
    """Get SI units for any variable name.""" 
    return hydrosphere_mapper.get_units(name)

def validate_hydrosphere_interface(input_vars: List[str], 
                                 output_vars: List[str]) -> List[str]:
    """Validate hydrosphere module interface consistency."""
    return hydrosphere_mapper.validate_interface_consistency(input_vars, output_vars)


# Export main components
__all__ = [
    "HydrosphereInterfaceMapper", 
    "HydrosphereInterfaceMixin",
    "VariableMapping", 
    "VariableCategory",
    "hydrosphere_mapper",
    "standardize_variable_name",
    "get_variable_units", 
    "validate_hydrosphere_interface",
    "FORCING_MAPPINGS",
    "STATE_MAPPINGS", 
    "FLUX_MAPPINGS",
    "DIAGNOSTIC_MAPPINGS",
    "ALL_MAPPINGS"
]
