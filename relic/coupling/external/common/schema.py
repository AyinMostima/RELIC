"""
RELIC Internal Schema and Variable Mappings

This module defines the standardized internal schema for RELIC, including:
- Standard variable names following CF-1.8 conventions
- Required units and dimensionality
- Mappings from external model variables to internal schema
- Data quality requirements and validation rules

The schema ensures consistency across all external model adapters and 
provides a unified interface for scientific modules within RELIC.

Mathematical Framework:
Each variable v has the mapping:
v_internal = M(v_external) where M includes:
- Name mapping: external_name -> cf_standard_name
- Unit conversion: external_unit -> SI/standard_unit  
- Dimension ordering: external_dims -> (time, lat, lon, [level])
- Quality flags: data_quality_masks and fill_values

References:
- CF Standard Names: http://cfconventions.org/standard-names.html
- CMIP6 Variable Definitions: https://pcmdi.llnl.gov/CMIP6/Guide/dataUsers.html
- UDUNITS-2: https://www.unidata.ucar.edu/software/udunits/
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datasource import DataSource

from relic.parameters import get_global_manager
from relic.coupling.scales.types import SpatialQuantityBasis, resolve_spatial_basis
from ....parameters.namespaces import ParameterNamespace, resolve_namespace


class VariableCategory(Enum):
    """Categories for grouping related variables."""
    ATMOSPHERIC_STATE = "atmospheric_state"           # T, P, humidity
    ATMOSPHERIC_FLUX = "atmospheric_flux"            # radiation, precipitation  
    SURFACE_STATE = "surface_state"                  # soil temp, moisture
    SURFACE_FLUX = "surface_flux"                    # sensible, latent heat
    VEGETATION_STATE = "vegetation_state"            # LAI, biomass, phenology
    VEGETATION_FLUX = "vegetation_flux"              # photosynthesis, respiration
    SOIL_STATE = "soil_state"                       # soil moisture, temperature
    SOIL_FLUX = "soil_flux"                         # decomposition, N cycling
    HYDROLOGIC_STATE = "hydrologic_state"           # water table, streamflow
    HYDROLOGIC_FLUX = "hydrologic_flux"             # ET, runoff, drainage


@dataclass
class VariableDefinition:
    """
    Complete definition of a standardized internal variable.
    
    Includes bounded CF variable metadata, units, valid ranges, and quality
    control parameters needed for consistent data handling.
    """
    # Core CF metadata
    standard_name: str                                # CF standard name
    long_name: str                                   # Descriptive name
    units: str                                       # Standard units (SI preferred)
    spatial_basis: SpatialQuantityBasis              # Explicit spatial representation (no default)
    category: VariableCategory                       # Declared category (no default)
    
    # Data specifications  
    dtype: str = "float32"                           # Data type
    dimensions: Tuple[str, ...] = ("time", "lat", "lon")  # Dimension order
    fill_value: Optional[Union[int, float]] = None    # Missing data value
    
    # Quality control
    valid_min: Optional[float] = None                # Physical minimum value
    valid_max: Optional[float] = None                # Physical maximum value  
    
    # CF cell_methods that distinguish variables sharing a standard_name (e.g. daily extrema)
    cell_methods: Optional[str] = None

    # Metadata
    comment: str = ""                                # Additional description
    references: str = ""                             # Scientific references

    @property
    def positive_definite(self) -> bool:
        """True when the catalog valid range excludes negative values (valid_min >= 0)."""
        # References: CF Conventions 1.8 section 2.5.1 (valid_min/valid_max define the valid
        #   range of a variable); coupling.external.variables.variable_definitions.
        # Mathematical expression: positive_definite = (valid_min >= 0).
        # Parameter meanings: valid_min is the smallest physically valid value in the catalog.
        # Units: valid_min carries the units of the variable; the result is a boolean.
        # Domain/constraints: valid_min must be declared; a missing value raises.
        # Sign convention: a non-negative valid_min forbids negative values of the field.
        if self.valid_min is None:
            raise ValueError(
                f"Variable {self.standard_name!r} declares no valid_min; its positivity is undefined."
            )
        return float(self.valid_min) >= 0.0


class InternalSchema:
    """
    Defines RELIC's internal variable schema and mapping utilities.
    
    Maintains the canonical definitions for all variables used within
    RELIC scientific modules, ensuring scoped metadata consistency
    throughout the modeling system.
    """
    
    def __init__(
        self,
        parameter_manager=None,
        parameter_manager_resolver=None,
    ):
        """Initialize schema with standard variable definitions."""
        self._parameter_manager = parameter_manager
        self._parameter_manager_resolver = parameter_manager_resolver
        self._variables: Optional[Dict[str, VariableDefinition]] = None
        self._mappings = self._define_external_mappings()
        self._validate_mapped_variables()

    def _get_parameter_manager(self):
        """
        Lazily resolve the ParameterManager to avoid import-time side effects.
        """
        if self._parameter_manager is None and self._parameter_manager_resolver is not None:
            self._parameter_manager = self._parameter_manager_resolver()
        if self._parameter_manager is None:
            self._parameter_manager = get_global_manager()
        return self._parameter_manager

    def _ensure_variables_loaded(self) -> None:
        """Load variable definitions once, using the parameter manager."""
        if self._variables is not None:
            return
        loaded = self._load_variable_definitions_from_params()
        if loaded is None:
            raise RuntimeError(
                "Failed to load external variable definitions from parameter system. "
                "Ensure 'coupling.external.variables.variable_definitions' exists."
            )
        self._variables = loaded

    # Keys every catalog entry declares; a missing key raises (no entry is skipped and no value,
    # in particular no category, is assumed).
    _REQUIRED_CATALOG_KEYS = ("standard_name", "long_name", "units", "category")

    def _load_variable_definitions_from_params(self) -> Optional[Dict[str, VariableDefinition]]:
        """Load the external variable catalog; every entry must declare its required fields."""
        pm = self._get_parameter_manager()
        # A missing or invalid catalog raises from the parameter manager with its own error.
        raw_defs = pm.get(
            "variable_definitions",
            namespace=resolve_namespace(ParameterNamespace.COUPLING_EXTERNAL_VARIABLES),
            expected_type=dict,
            required=True,
        )
        if not raw_defs:
            raise KeyError(
                "Empty 'variable_definitions' under namespace "
                "'coupling.external.variables' in parameter database."
            )
        variables: Dict[str, VariableDefinition] = {}
        for name, cfg in raw_defs.items():
            if name.startswith("__"):
                # Reserved catalog metadata (for example __meta__), not a variable.
                continue
            if not isinstance(cfg, dict):
                raise TypeError(
                    f"Catalog entry {name!r} of coupling.external.variables.variable_definitions must "
                    f"be a mapping, got {type(cfg).__name__}."
                )
            missing = [key for key in self._REQUIRED_CATALOG_KEYS if key not in cfg]
            if missing:
                raise KeyError(
                    f"Catalog entry {name!r} of coupling.external.variables.variable_definitions "
                    f"declares no {missing}."
                )
            if cfg["category"] not in VariableCategory.__members__:
                raise ValueError(
                    f"Catalog entry {name!r} declares category {cfg['category']!r}; expected one of "
                    f"{sorted(VariableCategory.__members__)}."
                )
            category = VariableCategory[cfg["category"]]
            variables[name] = VariableDefinition(
                standard_name=cfg["standard_name"],
                long_name=cfg["long_name"],
                units=cfg["units"],
                spatial_basis=resolve_spatial_basis(
                    name,
                    {
                        key: value
                        for key, value in (
                            ("spatial_basis", cfg.get("spatial_basis")),
                            ("variable_type", cfg.get("dimension")),
                        )
                        if value is not None
                    },
                ),
                valid_min=cfg.get("valid_min"),
                cell_methods=cfg.get("cell_methods"),
                valid_max=cfg.get("valid_max"),
                category=category,
                comment=cfg.get("description", ""),
                references=", ".join(cfg.get("references", [])),
            )
        return variables
    
    
    def _collect_mapped_variables(self) -> set:
        """Collect the set of internal variable names referenced by external mappings."""
        mapped = set()
        for mapping in self._mappings.values():
            mapped.update(mapping.values())
        return mapped

    def _validate_mapped_variables(self) -> None:
        """
        Validate that all mapped internal variables have parameter definitions.

        Raises:
            KeyError: If any mapped variable is missing from the parameter database.
        """
        self._ensure_variables_loaded()
        mapped_variables = self._collect_mapped_variables()
        missing_definitions = sorted(
            var_name for var_name in mapped_variables if var_name not in self._variables
        )
        if missing_definitions:
            raise KeyError(
                "Missing variable definitions for mapped external variables. "
                "Add entries under 'coupling.external.variables.variable_definitions' "
                f"in the parameter database for: {missing_definitions}"
            )
    
    def _define_external_mappings(self) -> Dict["DataSource", Dict[str, str]]:
        """
        Define mappings from external model variables to internal schema.
        
        Returns:
            Nested dictionary: {data_source: {external_name: internal_name}}
        """
        from .datasource import DataSource

        mappings: Dict[DataSource, Dict[str, str]] = {}

        # WRF variable mappings
        mappings[DataSource.WRF] = {
            # Temperature variables
            "T2": "air_temperature",                    # 2m temperature [K]
            "TSK": "surface_temperature",               # Surface skin temperature [K]
            "TSLB": "soil_temperature",                 # Soil temperature [K]
            
            # Moisture variables  
            "Q2": "specific_humidity",                  # 2 m water vapour mixing ratio w [kg/kg] -> q = w/(1+w)
            "RH2": "relative_humidity",                 # 2m relative humidity [%] -> [1]
            "SMOIS": "mass_content_of_water_in_soil_layer",  # Volumetric soil moisture per layer [m3/m3]
            
            # Pressure: the contract name (air_pressure is its alias, which StateManager refuses)
            "PSFC": "atmospheric_pressure",             # Surface pressure [Pa] (surface_air_pressure)
            
            # Wind
            "U10": "eastward_wind",                     # 10 m U-wind [m/s], grid-relative -> rotated
            "V10": "northward_wind",                    # 10 m V-wind [m/s], grid-relative -> rotated
            
            # Radiation
            # Radiation: the contract variables that hold the CF identities
            # surface_downwelling_{shortwave,longwave}_flux_in_air
            "SWDOWN": "shortwave_down",                 # SW down [W/m2]
            "LWDOWN": "longwave_down",                  # LW down [W/m2]
            "GLW": "longwave_down",                     # Alternative LW down
            
            # Precipitation - WRF has cumulative, need rate conversion
            "RAINC": "convective_precipitation_flux",   # Cumulative convective precip [mm]
            "RAINNC": "large_scale_precipitation_flux", # Cumulative grid-scale precip [mm]
            "PRCP_ACC": "precipitation_flux",           # Total accumulated [mm]
            
            # Surface fluxes
            "HFX": "sensible_heat_flux",                # Sensible heat flux [W/m2] (CF surface_upward_sensible_heat_flux)
            "LH": "latent_heat_flux",                   # Latent heat flux [W/m2] (CF surface_upward_latent_heat_flux)
        }
        
        # CESM/CIME/CLM variable mappings
        mappings[DataSource.CESM] = {
            # Atmospheric state
            "TSA": "air_temperature",                   # 2m air temperature [K]
            "TREFHT": "air_temperature",                # Alternative 2m temp [K]
            # PBOT/QBOT/UBOT/VBOT (and TBOT) are lowest-model-level fields, not near-surface
            # values (CAM history: "Lowest model level ..."; QBOT is a mixing ratio); unmapped.
            # CAM near-surface fields with RELIC consumers (atmospheric chemistry and
            # Penman-Monteith read atmospheric_pressure, the contract name of surface_air_pressure;
            # turbulence, Penman-Monteith and isotope fractionation read wind_speed). QREFHT
            # (reference-height specific humidity) has no RELIC consumer and is not mapped.
            "PS": "atmospheric_pressure",               # Surface pressure [Pa] (surface_air_pressure)
            "U10": "wind_speed",                        # 10 m wind speed [m/s]
            
            # Radiation
            "FSDS": "shortwave_down",                   # SW down [W/m2]
            "FLDS": "longwave_down",                    # LW down [W/m2]
            
            # Precipitation (CAM cam_diagnostics): PRECT total and PRECL large-scale
            # (stable) precipitation rates are liquid plus ice; the snow rates are
            # PRECSL (large-scale) and PRECSC (convective), and CAM writes no PRECS.
            # Surface rain is PRECT - PRECSC - PRECSL and snowfall PRECSC + PRECSL,
            # combinations of several fields, so no CESM field maps to rain_flux or
            # snowfall.
            "PRECT": "precipitation_flux",              # Total precipitation [m/s] -> [kg m-2 s-1]
            "PRECL": "large_scale_precipitation_flux",  # Large-scale precipitation, liq + ice [m/s]
            
            # Surface
            "TS": "surface_temperature",                # Surface temperature [K]
            "H2OSOI": "mass_content_of_water_in_soil_layer",  # Volumetric soil water per layer [mm3/mm3]
            "TSOI": "soil_temperature",                 # Soil temperature [K]
            
            # Vegetation
            "TLAI": "leaf_area_index",                  # Total LAI [m2/m2]
            
            # Fluxes  
            "FSH": "sensible_heat_flux",                # Sensible heat [W/m2] (CF surface_upward_sensible_heat_flux)
            "EFLX_LH_TOT": "latent_heat_flux",          # Latent heat [W/m2] (CF surface_upward_latent_heat_flux)
        }
        
        # CMIP6 variable mappings (following CMIP6 data request)
        mappings[DataSource.CMIP6] = {
            # Near-surface atmospheric state
            "tas": "air_temperature",                   # Near-surface air temp [K]
            "ps": "atmospheric_pressure",               # Surface pressure [Pa] (surface_air_pressure)
            "huss": "specific_humidity",                # Near-surface specific humidity [kg/kg]
            "hurs": "relative_humidity",                # Near-surface relative humidity [%] -> [1]
            
            # Extremes: air_temperature with cell_methods time: maximum / minimum (the
            # canonical ContractRegistry names; daily_*_air_temperature are aliases)
            "tasmax": "air_temperature_max",            # Daily max temp [K]
            "tasmin": "air_temperature_min",            # Daily min temp [K]
            
            # Wind
            "uas": "eastward_wind",                     # Eastward near-surface wind [m/s]
            "vas": "northward_wind",                    # Northward near-surface wind [m/s]
            "sfcWind": "wind_speed",                    # Near-surface wind speed [m/s]
            
            # Radiation
            "rsds": "shortwave_down",                   # SW down [W/m2]
            "rlds": "longwave_down",                    # LW down [W/m2]
            
            # Precipitation
            "pr": "precipitation_flux",                 # Precipitation [kg m-2 s-1]
            "prc": "convective_precipitation_flux",     # Convective precipitation [kg m-2 s-1]
            "prsn": "snowfall",                         # Snowfall flux, solid phase [kg m-2 s-1]
            
            # Surface
            "ts": "surface_temperature",                # Surface temperature [K]
            
            # Fluxes
            "hfss": "sensible_heat_flux",               # Sensible heat [W/m2] (CF surface_upward_sensible_heat_flux)
            "hfls": "latent_heat_flux",                 # Latent heat [W/m2] (CF surface_upward_latent_heat_flux)
        }
        
        # VIC hydrology model mappings
        mappings[DataSource.VIC] = {
            # Meteorological forcings (VIC inputs)
            "PRCP": "precipitation_flux",               # Precipitation per record [mm] -> [kg m-2 s-1]
            "TMAX": "air_temperature_max",              # Max daily temperature [C] -> [K]
            "TMIN": "air_temperature_min",              # Min daily temperature [C] -> [K]
            "WIND": "wind_speed",                       # Wind speed [m/s]
            
            # VIC outputs
            "OUT_EVAP": "evapotranspiration_flux",      # Total evapotranspiration per record [mm] (CF water_evapotranspiration_flux)
            "OUT_RUNOFF": "surface_runoff_flux",        # Surface runoff per record [mm]
            "OUT_BASEFLOW": "subsurface_runoff_flux",   # Baseflow per record [mm]
            "OUT_SWE": "surface_snow_amount",           # Snow water equivalent [mm]
            "OUT_SOIL_MOIST": "mass_content_of_water_in_soil_layer",  # Soil water per layer [mm]
            "OUT_SOIL_TEMP": "soil_temperature",        # Soil temperature [C] -> [K]
        }
        
        # SWAT watershed model mappings: SWAT 2012 output.hru column names (unit suffix removed).
        # TMP_MX/TMP_MN/TMP_AV and SOLAR are not mapped: they are daily values for a daily
        # print and period means of daily values for a monthly or annual print.
        mappings[DataSource.SWAT] = {
            "PRECIP": "precipitation_flux",             # Precipitation per record [mm]
            "SURQ_GEN": "surface_runoff_flux",          # Surface runoff generated per record [mm]
            "LATQGEN": "subsurface_runoff_flux",        # Lateral flow generated per record [mm]
            "ET": "evapotranspiration_flux",            # Actual evapotranspiration per record [mm] (CF water_evapotranspiration_flux)
            # SW_INIT/SW_END are not mapped: profile water above the wilting point, not the
            # soil water content (SWAT 2009 theoretical documentation, soil water section).
            "SOL_TMP": "soil_temperature",              # Soil temperature [degC] -> [K]
        }

        return mappings
    
    @staticmethod
    def canonical_sources() -> Dict["DataSource", Dict[str, str]]:
        """
        Declared canonical external variable per internal variable with several sources.

        Only an external field that measures the internal quantity itself qualifies (for
        example CMIP6 tas for air_temperature, not the daily extremes tasmax/tasmin).

        Returns:
            {data_source: {internal_name: canonical_external_name}}
        """
        from .datasource import DataSource

        return {
            # GLW is the standard wrfout downward longwave flux at the ground; LWDOWN is the
            # same quantity in forcing-style outputs.
            DataSource.WRF: {"longwave_down": "GLW"},
            # TREFHT (CAM reference height) and TSA (CLM, land only) are both the 2 m air
            # temperature; TREFHT is the grid-cell value.
            DataSource.CESM: {"air_temperature": "TREFHT"},
        }

    @staticmethod
    def source_transforms() -> Dict["DataSource", Dict[str, Dict[str, Any]]]:
        """
        Declared source-field transforms applied before unit conversion.

        ``accumulated``: the field is an accumulation since simulation start; the read path
        de-accumulates it over the output interval. ``layered``: the field is a volumetric
        content per layer; the read path converts it with the layer thickness the output carries
        (``thickness_variable``) or, when the output lacks it, the thickness declared in the
        adapter configuration field named by ``configured_thickness`` (None: no such field).
        ``water_depth``: the field is a liquid-equivalent water depth (per layer when
        ``layer_dimension`` is set); the read path converts it to mass per unit area.
        ``water_depth_flux``: the field is a water depth per unit time, or a water depth per
        record; the read path converts it to a mass flux, a per-record depth over the record
        interval of the file's CF time bounds. ``mixing_ratio``: the field is a water vapour mixing
        ratio w (mass of vapour per mass of dry air, in ``mixing_ratio_units``); the read path
        converts it to the specific humidity q = w / (1 + w). ``earth_relative_wind``: the field is
        one grid-relative wind component; the read path rotates the pair (``u_field``,
        ``v_field``) with the file's ``cos_field``/``sin_field`` to the earth-relative
        ``component`` (eastward or northward).

        Returns:
            {data_source: {external_name: declaration}}
        """
        from .datasource import DataSource

        def _earth_relative_wind(component: str) -> Dict[str, Any]:
            return {
                "kind": "earth_relative_wind",
                "component": component,
                "u_field": "U10",
                "v_field": "V10",
                "cos_field": "COSALPHA",
                "sin_field": "SINALPHA",
            }

        def _accumulated(counter: Optional[str]) -> Dict[str, Any]:
            return {
                "kind": "accumulated",
                "accumulation_units": "mm",
                "time_dimension": "Time",
                "time_coordinate": "time",
                "bucket_counter": counter,
                "bucket_size_attribute": "BUCKET_MM" if counter is not None else None,
            }

        return {
            DataSource.WRF: {
                "RAINNC": _accumulated("I_RAINNC"),
                "RAINC": _accumulated("I_RAINC"),
                "PRCP_ACC": _accumulated(None),
                # WRF surface layer (module_sf_sfclay): Q2 is the diagnostic 2 m water vapour
                # mixing ratio, not a specific humidity.
                "Q2": {"kind": "mixing_ratio", "mixing_ratio_units": "kg kg-1"},
                # U10/V10 are grid-relative components (UCAR MMM WRF support); eastward_wind and
                # northward_wind are earth-relative: rotated with the file's COSALPHA/SINALPHA.
                "U10": _earth_relative_wind("eastward"),
                "V10": _earth_relative_wind("northward"),
                # wrfout carries the Noah/Noah-MP layer thicknesses DZS(Time, soil_layers_stag).
                "SMOIS": {
                    "kind": "layered",
                    "content_units": "m3 m-3",
                    "layer_dimensions": ("soil_layers_stag",),
                    "thickness_variable": "DZS",
                    "thickness_units": "m",
                    "thickness_layer_dimension": "soil_layers_stag",
                    "configured_thickness": None,
                },
            },
            DataSource.CESM: {
                # CLM history: H2OSOI on levsoi (CLM5) or levgrnd (CLM4.5, ELM); the time-constant
                # DZSOI is on levgrnd, whose top layers are the levsoi soil layers.
                "H2OSOI": {
                    "kind": "layered",
                    "content_units": "mm3/mm3",
                    "layer_dimensions": ("levsoi", "levgrnd"),
                    "thickness_variable": "DZSOI",
                    "thickness_units": "m",
                    "thickness_layer_dimension": "levgrnd",
                    "configured_thickness": "soil_layer_thickness",
                },
            },
            DataSource.VIC: {
                # VIC 5 image driver: soil moisture content per layer as water depth [mm].
                "OUT_SOIL_MOIST": {"kind": "water_depth", "layer_dimension": "nlayer"},
                # Snow water equivalent [mm] (includes vegetation-intercepted snow in VIC).
                "OUT_SWE": {"kind": "water_depth", "layer_dimension": None},
                # Water depths summed over each output record [mm].
                "PRCP": {"kind": "water_depth_flux"},
                "OUT_EVAP": {"kind": "water_depth_flux"},
                "OUT_RUNOFF": {"kind": "water_depth_flux"},
                "OUT_BASEFLOW": {"kind": "water_depth_flux"},
            },
            DataSource.SWAT: {
                # SWAT output.hru records carry water depths per record [mm H2O].
                "PRECIP": {"kind": "water_depth_flux"},
                "SURQ_GEN": {"kind": "water_depth_flux"},
                "LATQGEN": {"kind": "water_depth_flux"},
                "ET": {"kind": "water_depth_flux"},
            },
        }

    def get_source_transform(self, source: "DataSource", external_name: str) -> Optional[Dict[str, Any]]:
        """Declared transform of one external field, or None for a direct field."""
        declaration = self.source_transforms().get(source, {}).get(external_name)
        return dict(declaration) if declaration is not None else None

    def resolve_external_names(
        self,
        source: "DataSource",
        internal_variables: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Invert the external -> internal mapping of one source for the requested variables.

        Parameters:
            source: External data source type
            internal_variables: Internal names to resolve; None resolves every internal name
                the source maps to

        Returns:
            {internal_name: external_name}

        Raises:
            KeyError: If the source or a requested internal name has no mapping
            ValueError: If several external names map to one internal name and no canonical
                source is declared in canonical_sources()
        """
        mapping = self.get_source_mapping(source)
        externals_by_internal: Dict[str, List[str]] = {}
        for external, internal in mapping.items():
            externals_by_internal.setdefault(internal, []).append(external)
        requested = (
            sorted(externals_by_internal) if internal_variables is None else list(internal_variables)
        )
        canonical = self.canonical_sources().get(source, {})
        resolved: Dict[str, str] = {}
        for internal in requested:
            externals = externals_by_internal.get(internal)
            if externals is None:
                raise KeyError(f"Data source {source} provides no mapping for {internal!r}.")
            if len(externals) == 1:
                resolved[internal] = externals[0]
            elif internal in canonical:
                resolved[internal] = canonical[internal]
            else:
                raise ValueError(
                    f"Data source {source} maps {sorted(externals)} to {internal!r}; declare the "
                    "canonical external variable in InternalSchema.canonical_sources()."
                )
        return resolved

    def get_source_mapping(self, source: "DataSource") -> Dict[str, str]:
        """
        Return the full external -> internal variable name mapping of one data source.

        Parameters:
            source: External data source type

        Returns:
            Copy of the mapping {external_name: internal_name}

        Raises:
            KeyError: If the source has no declared mapping
        """
        if source not in self._mappings:
            raise KeyError(f"Data source {source} has no variable mapping in InternalSchema.")
        return dict(self._mappings[source])

    def get_variable(self, name: str) -> Optional[VariableDefinition]:
        """
        Get variable definition by internal name.
        
        Parameters:
            name: Internal variable name
            
        Returns:
            Variable definition or None if not found
        """
        self._ensure_variables_loaded()
        return self._variables.get(name)
    
    def list_variables(self, 
                      category: Optional[VariableCategory] = None) -> List[str]:
        """
        List all available internal variables.
        
        Parameters:
            category: Optional filter by variable category
            
        Returns:
            List of internal variable names
        """
        self._ensure_variables_loaded()
        if category is None:
            return list(self._variables.keys())
        else:
            return [name for name, defn in self._variables.items() 
                   if defn.category == category]
    
    def create_mapping(self, 
                      source: DataSource,
                      target_variables: List[str]) -> Dict[str, str]:
        """
        Create variable name mapping for external data source.
        
        Parameters:
            source: External data source type
            target_variables: Desired internal variable names
            
        Returns:
            Mapping from external names to internal names of every requested variable

        Raises:
            KeyError: If the source is not supported, or naming every requested variable the
                source does not provide (no partial mapping is returned)
        """
        provided = set(self.get_source_mapping(source).values())
        unavailable = [target for target in target_variables if target not in provided]
        if unavailable:
            raise KeyError(
                f"Variables not available from {source.value}: {unavailable}; the "
                f"{source.value} mapping provides {sorted(provided)}."
            )
        resolved = self.resolve_external_names(source, list(target_variables))
        return {external: internal for internal, external in resolved.items()}
    
    def get_cf_metadata(self, internal_name: str) -> Dict[str, Any]:
        """
        Get scoped CF variable metadata for an internal variable.
        
        Parameters:
            internal_name: Internal variable name
            
        Returns:
            Dictionary of CF metadata attributes
        """
        self._ensure_variables_loaded()
        var_def = self.get_variable(internal_name)
        if not var_def:
            raise KeyError(f"Unknown variable: {internal_name}")
        
        return {
            'standard_name': var_def.standard_name,
            'long_name': var_def.long_name, 
            'units': var_def.units,
            '_FillValue': var_def.fill_value,
            'valid_min': var_def.valid_min,
            'valid_max': var_def.valid_max,
            'comment': var_def.comment,
            'references': var_def.references
        }



