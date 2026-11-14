"""
Plant Functional Type (PFT) Data Structures for RELIC

This module defines the PFT data structures and type annotations based on
CLM4, CASA-CNP, and other ecosystem model conventions. Provides unified
type definitions for parameter management across biosphere processes.

References
----------
- Guenther, A.B. et al. (2012). The Model of Emissions of Gases and Aerosols 
  from Nature version 2.1 (MEGAN2.1)
- Oleson, K.W. et al. (2013). Technical description of version 4.5 of the 
  Community Land Model (CLM)
- Wieder, W.R. et al. (2015). Regridded Harmonized World Soil Database v1.2

Mathematical Framework
-----------------------
PFT parameter aggregation: P_effective = sum(fi * Pi) where fi is PFT fraction.
Parameter scaling uses the canonical reference temperature key
``physical.constants.standard_temperature`` (Kelvin) from the ParameterManager.
Phenology scaling: P(DOY) = P_max * f_pheno(DOY, parameters).

Units and Conventions
----------------------
- Photosynthetic parameters: umol m-2 s-1 at the CF reference temperature (Kelvin).
- Allocation fractions: dimensionless [0-1], must sum to 1.0.
- Temperature parameters: K
- Time parameters: days
- BVOC emission factors: ug g-1 h-1 at standard conditions
"""

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

# Import base class and parameter manager
from relic.parameters import UnifiedParameterManager
from relic.parameters import set_value as set_parameter_value
from relic.scientific.biosphere.pft.pft_parameter_base import PFTParameterBase

_SHARED_PROCESS_PARAMETER_BINDINGS = {
    ("photosynthesis", "rd_q10"): "biosphere.photosynthesis.rd_q10",
    (
        "respiration",
        "q10_maint_resp",
    ): "biosphere.respiration.maintenance.q10_respiration",
    (
        "respiration",
        "n_effect_maint_resp",
    ): "biosphere.respiration.growth.np_response_coeff",
    (
        "respiration",
        "base_resp_rate",
    ): "biosphere.respiration.maintenance.min_respiration_rate",
}
_RESPIRATION_PARAMETER_PATH_MAP = {
    "maint_resp_leaf": "respiration.maintenance.rm_leaf_25",
    "maint_resp_stem": "respiration.maintenance.rm_stem_25",
    "maint_resp_root": "respiration.maintenance.rm_root_25",
    "q10_maint_resp": "respiration.maintenance.q10_respiration",
    "q10_growth_resp": "respiration.growth.q10_growth",
    "n_effect_maint_resp": "respiration.growth.np_response_coeff",
    "base_resp_rate": "respiration.maintenance.min_respiration_rate",
    "growth_respiration_fraction": "respiration.growth.ygrow_base",
    "maintenance_respiration_coefficient": "respiration.maintenance.rm_leaf_25",
}

PFT_TYPE_METADATA = {
    "mathematical_expression": "P_effective = sum(fi * Pi); P(DOY) = P_max * f_pheno(DOY)",
    "references": [
        "Oleson, K.W. et al. (2013). Technical description of version 4.5 of the Community Land Model (CLM).",
        "Guenther, A.B. et al. (2012). MEGAN2.1 technical documentation.",
    ],
    "conservation_laws": [],
    "parameters": {},
}

class PhotosynthesisPathway(Enum):
    """Photosynthesis pathway type."""
    C3 = "C3"
    C4 = "C4"
    CAM = "CAM"


class PFTCategory(Enum):
    """Plant functional type broad categories."""
    NEEDLELEAF_EVERGREEN_TEMPERATE = "needleleaf_evergreen_temperate"
    NEEDLELEAF_EVERGREEN_BOREAL = "needleleaf_evergreen_boreal"
    NEEDLELEAF_DECIDUOUS_BOREAL = "needleleaf_deciduous_boreal"
    BROADLEAF_EVERGREEN_TROPICAL = "broadleaf_evergreen_tropical"
    BROADLEAF_EVERGREEN_TEMPERATE = "broadleaf_evergreen_temperate"
    BROADLEAF_DECIDUOUS_TROPICAL = "broadleaf_deciduous_tropical"
    BROADLEAF_DECIDUOUS_TEMPERATE = "broadleaf_deciduous_temperate"
    BROADLEAF_DECIDUOUS_BOREAL = "broadleaf_deciduous_boreal"
    SHRUB_EVERGREEN_TEMPERATE = "shrub_evergreen_temperate"
    SHRUB_DECIDUOUS_TEMPERATE = "shrub_deciduous_temperate"
    C3_GRASS_ARCTIC = "c3_grass_arctic"
    C3_GRASS_TEMPERATE = "c3_grass_temperate"  
    C4_GRASS_TEMPERATE = "c4_grass_temperate"
    CROP_TEMPERATE = "crop_temperate"
    CROP_TROPICAL = "crop_tropical"


@dataclass
class PhotosynthesisParameters(PFTParameterBase):
    """Photosynthesis-related PFT parameters."""
    
    # Core biochemical parameters at the ParameterManager-backed reference temperature.
    vcmax25: float = None  # Maximum carboxylation rate at 25 degC [umol CO2 m-2 s-1]
    jmax25: float = None   # Maximum electron transport rate at 25 degC [umol e- m-2 s-1]
    rd25: float = None     # Dark respiration rate at 25 degC [umol CO2 m-2 s-1]
    
    # Pathway and kinetics
    pathway: PhotosynthesisPathway = None
    
    # Temperature response parameters
    vcmax_ea: float = None    # Activation energy for Vcmax [J mol-1]
    vcmax_hd: float = None   # Deactivation energy for Vcmax [J mol-1] 
    vcmax_se: float = None    # Entropy term for Vcmax [J mol-1 K-1]
    
    jmax_ea: float = None     # Activation energy for Jmax [J mol-1]
    jmax_hd: float = None    # Deactivation energy for Jmax [J mol-1]
    jmax_se: float = None     # Entropy term for Jmax [J mol-1 K-1]
    
    rd_ea: float = None       # Activation energy for Rd [J mol-1]
    rd_q10: float = None      # Q10 temperature coefficient for Rd
    
    # Light response parameters
    alpha: float = None       # Quantum yield of electron transport [mol e- mol-1 photon]
    theta: float = None       # Curvature parameter for light response [dimensionless]
    
    # Stomatal conductance parameters (Ball-Berry model)
    g0: float = None          # Minimum stomatal conductance [mol m-2 s-1]
    g1: float = None          # Stomatal slope parameter [dimensionless]
    
    # Ozone sensitivity parameters
    o3_sensitivity: float = None   # Ozone sensitivity factor [dimensionless, 0-2]
    o3_threshold: float = None    # Ozone damage threshold [ppb]
    
    def __post_init__(self, parameter_manager: UnifiedParameterManager):
        """Initialize parameters from ParameterManager and validate."""
        super().__post_init__(parameter_manager)
        if not self._pft_namespace:
            missing_required = [
                name for name in ("vcmax25", "jmax25", "rd25")
                if getattr(self, name) is None
            ]
            if missing_required:
                raise ValueError("PFT namespace required for parameter retrieval")
            if self.pathway is None:
                default_key = "biosphere.pft.default.photosynthesis.pathway"
                default_pathway = self._parameter_manager.get(
                    default_key,
                )
                if default_pathway is None:
                    raise ValueError("PFT namespace required for parameter retrieval")
                if isinstance(default_pathway, str):
                    self.pathway = PhotosynthesisPathway(default_pathway)
                else:
                    self.pathway = default_pathway
            self._validate_parameters()
            return
        self._fetch_from_parameter_manager()
        self._validate_parameters()
    
    def _fetch_from_parameter_manager(self):
        """Fetch missing parameters from ParameterManager."""
        namespace = self._pft_namespace if self._pft_namespace else "biosphere.pft"
        
        # Define parameter metadata (without hardcoded defaults)
        param_metadata = {
            'vcmax25': {
                'units': 'umol CO2 m-2 s-1',
                'description': 'Maximum carboxylation rate at 25degC',
                'references': ['doi:10.1071/PP9900137']
            },
            'jmax25': {
                'units': 'umol e- m-2 s-1',
                'description': 'Maximum electron transport rate at 25degC',
                'references': ['doi:10.1071/PP9900137']
            },
            'rd25': {
                'units': 'umol CO2 m-2 s-1',
                'description': 'Dark respiration rate at 25degC',
                'references': ['doi:10.1071/PP9900137']
            },
            'pathway': {
                'units': 'dimensionless',
                'description': 'Photosynthetic pathway type',
                'references': ['Sage, R.F. (2004) The evolution of C4 photosynthesis']
            },
            'vcmax_ea': {
                'units': 'J mol-1',
                'description': 'Activation energy for Vcmax temperature response',
                'references': ['Bernacchi et al. (2001) Plant Cell Environ 24:253']
            },
            'vcmax_hd': {
                'units': 'J mol-1',
                'description': 'Deactivation energy for Vcmax at high temperature',
                'references': ['Bernacchi et al. (2001) Plant Cell Environ 24:253']
            },
            'vcmax_se': {
                'units': 'J mol-1 K-1',
                'description': 'Entropy term for Vcmax temperature response',
                'references': ['Bernacchi et al. (2001) Plant Cell Environ 24:253']
            },
            'jmax_ea': {
                'units': 'J mol-1',
                'description': 'Activation energy for Jmax temperature response',
                'references': ['Bernacchi et al. (2001) Plant Cell Environ 24:253']
            },
            'jmax_hd': {
                'units': 'J mol-1',
                'description': 'Deactivation energy for Jmax at high temperature',
                'references': ['Bernacchi et al. (2001) Plant Cell Environ 24:253']
            },
            'jmax_se': {
                'units': 'J mol-1 K-1',
                'description': 'Entropy term for Jmax temperature response',
                'references': ['Bernacchi et al. (2001) Plant Cell Environ 24:253']
            },
            'rd_ea': {
                'units': 'J mol-1',
                'description': 'Activation energy for dark respiration',
                'references': ['Atkin et al. (2005) New Phytol 165:351']
            },
            'rd_q10': {
                'units': 'dimensionless',
                'description': 'Q10 temperature coefficient for dark respiration',
                'references': ['Atkin et al. (2005) New Phytol 165:351']
            },
            'alpha': {
                'units': 'mol e- mol-1 photon',
                'description': 'Quantum yield of electron transport',
                'references': ['Evans (1987) Aust J Plant Physiol 14:159']
            },
            'theta': {
                'units': 'dimensionless',
                'description': 'Curvature parameter for light response',
                'references': ['Farquhar & Wong (1984) Funct Plant Biol 11:191']
            },
            'g0': {
                'units': 'mol m-2 s-1',
                'description': 'Minimum stomatal conductance (Ball-Berry model)',
                'references': ['Ball et al. (1987) in Progress in Photosynthesis']
            },
            'g1': {
                'units': 'dimensionless',
                'description': 'Stomatal slope parameter (Ball-Berry model)',
                'references': ['Ball et al. (1987) in Progress in Photosynthesis']
            },
            'o3_sensitivity': {
                'units': 'dimensionless',
                'description': 'Ozone sensitivity factor (0=resistant, >1=sensitive)',
                'references': ['Pleijel et al. (2007) Atmos Environ 41:8407']
            },
            'o3_threshold': {
                'units': 'ppb',
                'description': 'Ozone damage threshold concentration',
                'references': ['Pleijel et al. (2007) Atmos Environ 41:8407']
            }
        }
        
        # Fetch parameters using base class method
        mandatory_params = {"vcmax25", "jmax25", "rd25", "pathway"}
        
        for attr_name, metadata in param_metadata.items():
            if getattr(self, attr_name) is None:
                param_key = f"{namespace}.photosynthesis.{attr_name}"
                required_flag = attr_name in mandatory_params
                value = self.get_parameter_from_manager(
                    param_key,
                    required=False,
                    register_if_missing=False,
                    metadata=metadata
                )
                shared_key = _SHARED_PROCESS_PARAMETER_BINDINGS.get(
                    ("photosynthesis", attr_name)
                )
                if value is None and shared_key is not None:
                    value = self._parameter_manager.get(
                        shared_key,
                        required=False,
                    )

                if value is None and required_flag:
                    raise ValueError(f"Missing required photosynthesis parameter {param_key}")
                
                # Handle pathway enum conversion
                if attr_name == 'pathway' and value is not None:
                    if isinstance(value, str):
                        value = PhotosynthesisPathway(value)
                
                setattr(self, attr_name, value)
    
    def _validate_parameters(self):
        """Validate photosynthesis parameters."""
        if self.vcmax25 is not None and self.vcmax25 <= 0:
            raise ValueError("vcmax25 must be positive")
        if self.jmax25 is not None and self.jmax25 <= 0:
            raise ValueError("jmax25 must be positive") 
        if self.rd25 is not None and self.rd25 < 0:
            raise ValueError("rd25 must be non-negative")
        if self.jmax25 is not None and self.vcmax25 is not None and self.vcmax25 > 0:
            ratio = self.jmax25 / self.vcmax25
            if not 1.3 <= ratio <= 2.5:
                import warnings
                warnings.warn(f"Jmax25/Vcmax25 ratio {ratio:.2f} outside typical range [1.3, 2.5]")


@dataclass
class PhenologyParameters(PFTParameterBase):
    """Phenology-related PFT parameters."""
    
    # Growing degree day parameters
    gdd_min: float = None         # Minimum temperature for GDD accumulation [K]
    gdd_leaf_on: float = None     # GDD threshold for leaf onset [degree-days]
    gdd_leaf_off: float = None    # GDD threshold for leaf senescence [degree-days]
    
    # Critical temperatures
    temp_cold_kill: float = None  # Temperature causing mortality [K]
    temp_growth_min: float = None # Minimum temperature for growth [K]
    temp_growth_opt: float = None # Optimal temperature for growth [K]
    temp_growth_max: float = None # Maximum temperature for growth [K]
    
    # Photoperiod parameters
    photoperiod_critical: float = None    # Critical day length [hours]
    photoperiod_sensitivity: float = None # Photoperiod sensitivity [dimensionless]
    
    # Leaf area and longevity
    specific_leaf_area: float = None  # Specific leaf area [m2 kg-1]
    leaf_longevity: float = None      # Leaf longevity [days]
    
    # Growth form parameters
    max_height: float = None          # Maximum vegetation height [m]
    wood_density: float = None        # Wood density [g cm-3]
    
    def __post_init__(self, parameter_manager: UnifiedParameterManager):
        """Initialize parameters from ParameterManager and validate."""
        super().__post_init__(parameter_manager)
        if self._pft_namespace:
            self._fetch_from_parameter_manager()
            self._validate_parameters()
    
    def _fetch_from_parameter_manager(self):
        """Fetch missing parameters from ParameterManager."""
        namespace = self._pft_namespace if self._pft_namespace else "biosphere.pft"
        
        # Define parameter metadata
        param_metadata = {
            'gdd_min': {
                'units': 'K',
                'description': 'Minimum temperature for growing degree day accumulation',
                'references': ['McMaster & Wilhelm (1997) Agric Forest Meteorol 87:291']
            },
            'gdd_leaf_on': {
                'units': 'degree-days',
                'description': 'Growing degree days threshold for leaf onset',
                'references': ['Richardson et al. (2006) Global Change Biol 12:1611']
            },
            'gdd_leaf_off': {
                'units': 'degree-days',
                'description': 'Growing degree days threshold for leaf senescence',
                'references': ['Richardson et al. (2006) Global Change Biol 12:1611']
            },
            'temp_cold_kill': {
                'units': 'K',
                'description': 'Temperature causing plant mortality',
                'references': ['Sakai & Larcher (1987) Frost Survival of Plants']
            },
            'temp_growth_min': {
                'units': 'K',
                'description': 'Minimum temperature for growth activity',
                'references': ['Parent & Tardif (2002) Tree Physiol 22:1245']
            },
            'temp_growth_opt': {
                'units': 'K',
                'description': 'Optimal temperature for maximum growth',
                'references': ['Parent & Tardif (2002) Tree Physiol 22:1245']
            },
            'temp_growth_max': {
                'units': 'K',
                'description': 'Maximum temperature for growth activity',
                'references': ['Parent & Tardif (2002) Tree Physiol 22:1245']
            },
            'photoperiod_critical': {
                'units': 'hours',
                'description': 'Critical day length for photoperiod response',
                'references': ['Thomas & Vince-Prue (1996) Photoperiodism in Plants']
            },
            'photoperiod_sensitivity': {
                'units': 'dimensionless',
                'description': 'Sensitivity to photoperiod changes',
                'references': ['Thomas & Vince-Prue (1996) Photoperiodism in Plants']
            },
            'specific_leaf_area': {
                'units': 'm2 kg-1',
                'description': 'Specific leaf area (leaf area per unit leaf mass)',
                'references': ['Wright et al. (2004) Nature 428:821']
            },
            'leaf_longevity': {
                'units': 'days',
                'description': 'Average leaf lifespan',
                'references': ['Wright et al. (2004) Nature 428:821']
            },
            'max_height': {
                'units': 'm',
                'description': 'Maximum vegetation height',
                'references': ['Falster & Westoby (2003) J Ecol 91:644']
            },
            'wood_density': {
                'units': 'g cm-3',
                'description': 'Wood density',
                'references': ['Chave et al. (2006) Oecologia 145:87']
            }
        }
        
        # Fetch parameters
        for attr_name, metadata in param_metadata.items():
            if getattr(self, attr_name) is None:
                param_key = f"{namespace}.phenology.{attr_name}"
                value = self.get_parameter_from_manager(
                    param_key,
                    required=False,  # Make optional to allow test usage
                    register_if_missing=False,
                    metadata=metadata
                )
                setattr(self, attr_name, value)

    def _validate_parameters(self):
        """Validate phenology parameters."""
        if self.gdd_leaf_off is not None and self.gdd_leaf_on is not None:
            if self.gdd_leaf_off <= self.gdd_leaf_on:
                raise ValueError("gdd_leaf_off must be greater than gdd_leaf_on")
        if self.temp_growth_max is not None and self.temp_growth_min is not None:
            if self.temp_growth_max <= self.temp_growth_min:
                raise ValueError("temp_growth_max must be greater than temp_growth_min")


@dataclass
class AllocationParameters(PFTParameterBase):
    """Carbon allocation PFT parameters."""
    
    # Tissue allocation fractions (must sum to 1.0)
    f_leaf: float = None   # Leaf allocation fraction [dimensionless]
    f_stem: float = None   # Stem allocation fraction [dimensionless] 
    f_root: float = None   # Root allocation fraction [dimensionless]
    
    # Storage allocation
    f_storage: float = None  # Storage organ fraction [dimensionless]
    
    # Allometric relationships
    allometry_leaf_stem: float = None   # Leaf:stem allometric exponent
    allometry_root_shoot: float = None  # Root:shoot allometric exponent
    
    # Growth respiration coefficients
    growth_resp_leaf: float = None  # Growth respiration coefficient for leaves [g C g-1 C]
    growth_resp_stem: float = None  # Growth respiration coefficient for stems [g C g-1 C]
    growth_resp_root: float = None  # Growth respiration coefficient for roots [g C g-1 C]
    
    # Tissue construction costs
    construction_cost_leaf: float = None  # Construction cost for leaves [g glucose g-1 biomass]
    construction_cost_stem: float = None  # Construction cost for stems [g glucose g-1 biomass]
    construction_cost_root: float = None  # Construction cost for roots [g glucose g-1 biomass]

    @property
    def leaf_allocation_fraction(self) -> Optional[float]:
        return self.f_leaf

    @property
    def stem_allocation_fraction(self) -> Optional[float]:
        return self.f_stem

    @property
    def root_allocation_fraction(self) -> Optional[float]:
        return self.f_root

    def __post_init__(self, parameter_manager: UnifiedParameterManager):
        """Initialize parameters from ParameterManager and validate."""
        super().__post_init__(parameter_manager)
        if self._pft_namespace:
            self._fetch_from_parameter_manager()
            self._validate_parameters()
    
    def _fetch_from_parameter_manager(self):
        """Fetch missing parameters from ParameterManager."""
        namespace = self._pft_namespace if self._pft_namespace else "biosphere.pft"
        
        # Define parameter metadata
        param_metadata = {
            'f_leaf': {
                'units': 'dimensionless',
                'description': 'Fraction of carbon allocated to leaves',
                'references': ['Litton et al. (2007) Forests 4:913']
            },
            'f_stem': {
                'units': 'dimensionless',
                'description': 'Fraction of carbon allocated to stems',
                'references': ['Litton et al. (2007) Forests 4:913']
            },
            'f_root': {
                'units': 'dimensionless',
                'description': 'Fraction of carbon allocated to roots',
                'references': ['Litton et al. (2007) Forests 4:913']
            },
            'f_storage': {
                'units': 'dimensionless',
                'description': 'Fraction of carbon allocated to storage',
                'references': ['Sala et al. (2012) Tree Physiol 32:648']
            },
            'allometry_leaf_stem': {
                'units': 'dimensionless',
                'description': 'Allometric exponent for leaf:stem scaling',
                'references': ['Enquist & Niklas (2002) Nature 419:556']
            },
            'allometry_root_shoot': {
                'units': 'dimensionless',
                'description': 'Allometric exponent for root:shoot scaling',
                'references': ['Enquist & Niklas (2002) Nature 419:556']
            },
            'growth_resp_leaf': {
                'units': 'g C g-1 C',
                'description': 'Growth respiration coefficient for leaves',
                'references': ['Penning de Vries et al. (1974) Ann Bot 38:809']
            },
            'growth_resp_stem': {
                'units': 'g C g-1 C',
                'description': 'Growth respiration coefficient for stems',
                'references': ['Penning de Vries et al. (1974) Ann Bot 38:809']
            },
            'growth_resp_root': {
                'units': 'g C g-1 C',
                'description': 'Growth respiration coefficient for roots',
                'references': ['Penning de Vries et al. (1974) Ann Bot 38:809']
            },
            'construction_cost_leaf': {
                'units': 'g glucose g-1 biomass',
                'description': 'Construction cost for leaf tissue',
                'references': ['Penning de Vries et al. (1974) Ann Bot 38:809']
            },
            'construction_cost_stem': {
                'units': 'g glucose g-1 biomass',
                'description': 'Construction cost for stem tissue',
                'references': ['Penning de Vries et al. (1974) Ann Bot 38:809']
            },
            'construction_cost_root': {
                'units': 'g glucose g-1 biomass',
                'description': 'Construction cost for root tissue',
                'references': ['Penning de Vries et al. (1974) Ann Bot 38:809']
            }
        }
        
        # Fetch parameters
        for attr_name, metadata in param_metadata.items():
            if getattr(self, attr_name) is None:
                param_key = f"{namespace}.allocation.{attr_name}"
                value = self.get_parameter_from_manager(
                    param_key,
                    required=False,  # Make optional to allow test usage
                    register_if_missing=False,
                    metadata=metadata
                )
                setattr(self, attr_name, value)

    def _validate_parameters(self):
        """Validate allocation parameters."""
        if all(x is not None for x in [self.f_leaf, self.f_stem, self.f_root]):
            total_allocation = self.f_leaf + self.f_stem + self.f_root
            if abs(total_allocation - 1.0) > 0.01:
                raise ValueError(f"Allocation fractions must sum to 1.0, got {total_allocation:.3f}")


@dataclass
class RespirationParameters(PFTParameterBase):
    """Respiration PFT parameters."""
    
    # Maintenance respiration coefficients (at 25degC)
    maint_resp_leaf: float = None  # Leaf maintenance respiration [g C g-1 C day-1]
    maint_resp_stem: float = None  # Stem maintenance respiration [g C g-1 C day-1]
    maint_resp_root: float = None  # Root maintenance respiration [g C g-1 C day-1]
    
    # Temperature sensitivity (Q10 values)
    q10_maint_resp: float = None   # Q10 for maintenance respiration
    q10_growth_resp: float = None  # Q10 for growth respiration
    
    # Nitrogen effect on respiration
    n_effect_maint_resp: float = None  # Nitrogen effect on maintenance respiration
    
    # Base respiration rates (alternative parameterization)
    base_resp_rate: float = None  # Base respiration rate [umol CO2 g-1 s-1]
    growth_respiration_fraction: float = None  # Growth respiration fraction (dimensionless)
    maintenance_respiration_coefficient: float = None  # Representative maintenance respiration coefficient

    def __post_init__(self, parameter_manager: UnifiedParameterManager):
        """Initialize parameters from ParameterManager and validate."""
        super().__post_init__(parameter_manager)
        if self._pft_namespace:
            self._fetch_from_parameter_manager()
            self._validate_parameters()
    
    def _fetch_from_parameter_manager(self):
        """Fetch missing parameters from ParameterManager."""
        namespace = self._pft_namespace if self._pft_namespace else "biosphere.pft"
        
        # Define parameter metadata
        param_metadata = {
            'maint_resp_leaf': {
                'units': 'g C g-1 C day-1',
                'description': 'Leaf maintenance respiration coefficient at 25degC',
                'references': ['Atkin & Tjoelker (2003) New Phytol 157:405']
            },
            'maint_resp_stem': {
                'units': 'g C g-1 C day-1',
                'description': 'Stem maintenance respiration coefficient at 25degC',
                'references': ['Atkin & Tjoelker (2003) New Phytol 157:405']
            },
            'maint_resp_root': {
                'units': 'g C g-1 C day-1',
                'description': 'Root maintenance respiration coefficient at 25degC',
                'references': ['Atkin & Tjoelker (2003) New Phytol 157:405']
            },
            'q10_maint_resp': {
                'units': 'dimensionless',
                'description': 'Q10 temperature coefficient for maintenance respiration',
                'references': ['Tjoelker et al. (2001) Global Change Biol 7:223']
            },
            'q10_growth_resp': {
                'units': 'dimensionless',
                'description': 'Q10 temperature coefficient for growth respiration',
                'references': ['Tjoelker et al. (2001) Global Change Biol 7:223']
            },
            'n_effect_maint_resp': {
                'units': 'dimensionless',
                'description': 'Nitrogen effect scaling factor on maintenance respiration',
                'references': ['Reich et al. (2008) Nature 451:49']
            },
            'base_resp_rate': {
                'units': 'umol CO2 g-1 s-1',
                'description': 'Base respiration rate coefficient',
                'references': ['Atkin & Tjoelker (2003) New Phytol 157:405']
            },
            'growth_respiration_fraction': {
                'units': 'dimensionless',
                'description': 'Fraction of carbon allocated to growth respiration',
                'references': ['Penning de Vries et al. (1974) Simulation of photosynthetic systems']
            },
            'maintenance_respiration_coefficient': {
                'units': 'g C g-1 C day-1',
                'description': 'Representative maintenance respiration coefficient',
                'references': ['Atkin & Tjoelker (2003) New Phytol 157:405']
            }
        }
        
        # Fetch parameters
        for attr_name, metadata in param_metadata.items():
            if getattr(self, attr_name) is None:
                parameter_path = _RESPIRATION_PARAMETER_PATH_MAP.get(
                    attr_name,
                    f"respiration.{attr_name}",
                )
                param_key = f"{namespace}.{parameter_path}"
                value = self.get_parameter_from_manager(
                    param_key,
                    required=False,  # Make optional to allow test usage
                    register_if_missing=False,
                    metadata=metadata
                )
                shared_key = _SHARED_PROCESS_PARAMETER_BINDINGS.get(
                    ("respiration", attr_name)
                )
                if value is None and shared_key is not None:
                    value = self._parameter_manager.get(
                        shared_key,
                        required=False,
                    )
                setattr(self, attr_name, value)

    def _validate_parameters(self):
        """Validate respiration parameters."""
        for attr_name in ['maint_resp_leaf', 'maint_resp_stem', 'maint_resp_root']:
            value = getattr(self, attr_name)
            if value is not None and value < 0:
                raise ValueError(f"{attr_name} must be non-negative, got {value}")


@dataclass
class BVOCParameters(PFTParameterBase):
    """Biogenic VOC emission PFT parameters."""
    
    # Emission factors (ug g-1 h-1 at standard conditions: 30degC, 1000 umol m-2 s-1)
    ef_isoprene: float = None        # Isoprene emission factor
    ef_monoterpene: float = None     # Monoterpene emission factor  
    ef_sesquiterpene: float = None   # Sesquiterpene emission factor
    ef_other_voc: float = None       # Other VOCs emission factor
    
    # Temperature and light response parameters
    isoprene_beta: float = None      # Isoprene temperature response parameter [K-1]
    monoterpene_beta: float = None   # Monoterpene temperature response parameter [K-1]
    
    # Light dependency factors
    isoprene_alpha: float = None        # Isoprene light response parameter
    isoprene_cl: float = None           # Isoprene light response constant
    light_dep_monoterpene: float = None # Light dependency for monoterpene emissions [dimensionless]
    temp_dep_monoterpene: float = None  # Temperature dependency for monoterpene emissions [K-1]
    
    # Leaf age effect parameters
    leaf_age_factor: float = None       # Leaf age effect on emissions [dimensionless]
    
    # Stress effect parameters
    drought_stress_effect: float = None # Drought stress effect on emissions

    @property
    def isoprene_emission_factor(self) -> Optional[float]:
        return self.ef_isoprene

    @property
    def monoterpene_emission_factor(self) -> Optional[float]:
        return self.ef_monoterpene
    
    def __post_init__(self, parameter_manager: UnifiedParameterManager):
        """Initialize parameters from ParameterManager and validate."""
        super().__post_init__(parameter_manager)
        if self._pft_namespace:
            self._fetch_from_parameter_manager()
            self._validate_parameters()
    
    def _fetch_from_parameter_manager(self):
        """Fetch missing parameters from ParameterManager."""
        namespace = self._pft_namespace if self._pft_namespace else "biosphere.pft"
        
        # Define parameter metadata
        param_metadata = {
            'ef_isoprene': {
                'units': 'ug g-1 h-1',
                'description': 'Isoprene emission factor at standard conditions',
                'references': ['Guenther et al. (2012) Geosci Model Dev 5:1471']
            },
            'ef_monoterpene': {
                'units': 'ug g-1 h-1',
                'description': 'Monoterpene emission factor at standard conditions',
                'references': ['Guenther et al. (2012) Geosci Model Dev 5:1471']
            },
            'ef_sesquiterpene': {
                'units': 'ug g-1 h-1',
                'description': 'Sesquiterpene emission factor at standard conditions',
                'references': ['Guenther et al. (2012) Geosci Model Dev 5:1471']
            },
            'ef_other_voc': {
                'units': 'ug g-1 h-1',
                'description': 'Other VOC emission factor at standard conditions',
                'references': ['Guenther et al. (2012) Geosci Model Dev 5:1471']
            },
            'isoprene_beta': {
                'units': 'K-1',
                'description': 'Isoprene temperature response parameter',
                'references': ['Guenther et al. (1993) J Geophys Res 98:12609']
            },
            'monoterpene_beta': {
                'units': 'K-1',
                'description': 'Monoterpene temperature response parameter',
                'references': ['Guenther et al. (1993) J Geophys Res 98:12609']
            },
            'isoprene_alpha': {
                'units': 'dimensionless',
                'description': 'Isoprene light response parameter',
                'references': ['Guenther et al. (1999) Atmos Environ 33:2067']
            },
            'isoprene_cl': {
                'units': 'dimensionless',
                'description': 'Isoprene light response constant',
                'references': ['Guenther et al. (1999) Atmos Environ 33:2067']
            },
            'leaf_age_factor': {
                'units': 'dimensionless',
                'description': 'Leaf age effect factor on BVOC emissions',
                'references': ['Guenther et al. (2006) Atmos Chem Phys 6:3181']
            },
            'drought_stress_effect': {
                'units': 'dimensionless',
                'description': 'Drought stress effect factor on BVOC emissions',
                'references': ['Guenther et al. (2006) Atmos Chem Phys 6:3181']
            },
            'light_dep_monoterpene': {
                'units': 'dimensionless',
                'description': 'Light dependency factor for monoterpene emissions',
                'references': ['Guenther et al. (1993) J Geophys Res 98:12609']
            },
            'temp_dep_monoterpene': {
                'units': 'K-1',
                'description': 'Temperature dependency factor for monoterpene emissions',
                'references': ['Guenther et al. (1993) J Geophys Res 98:12609']
            }
        }
        
        # Fetch parameters
        for attr_name, metadata in param_metadata.items():
            if getattr(self, attr_name) is None:
                param_key = f"{namespace}.bvoc.{attr_name}"
                value = self.get_parameter_from_manager(
                    param_key,
                    required=False,  # Make optional to allow test usage
                    register_if_missing=False,
                    metadata=metadata
                )
                setattr(self, attr_name, value)

    def _validate_parameters(self):
        """Validate BVOC parameters."""
        emission_factors = [self.ef_isoprene, self.ef_monoterpene, 
                           self.ef_sesquiterpene, self.ef_other_voc]
        emission_factors = [ef for ef in emission_factors if ef is not None]
        if any(ef < 0 for ef in emission_factors):
            raise ValueError("All emission factors must be non-negative")


@dataclass
class StressParameters(PFTParameterBase):
    """Stress response PFT parameters."""
    
    # Water stress parameters
    water_stress_threshold: float = None  # Water stress threshold [MPa]
    water_stress_slope: float = None      # Water stress response slope
    osmotic_adjustment: float = None      # Osmotic adjustment capacity [MPa]
    
    # Temperature stress parameters  
    temp_stress_min: float = None         # Minimum temperature for activity [K]
    temp_stress_max: float = None         # Maximum temperature for activity [K]
    heat_shock_threshold: float = None    # Heat shock threshold [K]
    frost_tolerance: float = None         # Frost tolerance threshold [K]
    
    # Nutrient stress parameters
    nitrogen_stress_half_sat: float = None    # N limitation half-saturation [g N m-2]
    phosphorus_stress_half_sat: float = None  # P limitation half-saturation [g P m-2]
    
    # Ozone stress parameters (linked to PhotosynthesisParameters)
    ozone_uptake_rate: float = None       # Ozone uptake rate constant [dimensionless]
    ozone_detox_capacity: float = None    # Ozone detoxification capacity [dimensionless]
    
    # Light stress parameters
    light_stress_threshold: float = None  # Light stress threshold [umol m-2 s-1]
    temp_stress_threshold: float = None   # Temperature stress threshold [K]

    @property
    def temp_min(self) -> Optional[float]:
        return self.temp_stress_min

    @property
    def temp_max(self) -> Optional[float]:
        return self.temp_stress_max
    
    def __post_init__(self, parameter_manager: UnifiedParameterManager):
        """Initialize parameters from ParameterManager and validate."""
        super().__post_init__(parameter_manager)
        if self._pft_namespace:
            self._fetch_from_parameter_manager()
            self._validate_parameters()
    
    def _fetch_from_parameter_manager(self):
        """Fetch missing parameters from ParameterManager."""
        namespace = self._pft_namespace if self._pft_namespace else "biosphere.pft"
        
        # Define parameter metadata
        param_metadata = {
            'water_stress_threshold': {
                'units': 'MPa',
                'description': 'Water potential threshold for stress onset',
                'references': ['Hsiao (1973) Annu Rev Plant Physiol 24:519']
            },
            'water_stress_slope': {
                'units': 'dimensionless',
                'description': 'Water stress response slope parameter',
                'references': ['Hsiao (1973) Annu Rev Plant Physiol 24:519']
            },
            'osmotic_adjustment': {
                'units': 'MPa',
                'description': 'Osmotic adjustment capacity under water stress',
                'references': ['Turner & Jones (1980) Plant Cell Environ 3:133']
            },
            'temp_stress_min': {
                'units': 'K',
                'description': 'Minimum temperature for physiological activity',
                'references': ['Larcher (2003) Physiological Plant Ecology']
            },
            'temp_stress_max': {
                'units': 'K',
                'description': 'Maximum temperature for physiological activity',
                'references': ['Larcher (2003) Physiological Plant Ecology']
            },
            'heat_shock_threshold': {
                'units': 'K',
                'description': 'Temperature threshold for heat shock response',
                'references': ['Vierling (1991) Annu Rev Plant Physiol Plant Mol Biol 42:579']
            },
            'frost_tolerance': {
                'units': 'K',
                'description': 'Frost tolerance threshold temperature',
                'references': ['Sakai & Larcher (1987) Frost Survival of Plants']
            },
            'nitrogen_stress_half_sat': {
                'units': 'g N m-2',
                'description': 'Half-saturation constant for nitrogen limitation',
                'references': ['Aber et al. (1989) Bioscience 39:378']
            },
            'phosphorus_stress_half_sat': {
                'units': 'g P m-2',
                'description': 'Half-saturation constant for phosphorus limitation',
                'references': ['Vitousek et al. (2010) Science 328:1408']
            },
            'ozone_uptake_rate': {
                'units': 'dimensionless',
                'description': 'Ozone uptake rate constant (relative to standard)',
                'references': ['Reich (1987) New Phytol 105:395']
            },
            'ozone_detox_capacity': {
                'units': 'dimensionless',
                'description': 'Ozone detoxification capacity (relative to standard)',
                'references': ['Reich (1987) New Phytol 105:395']
            },
            'light_stress_threshold': {
                'units': 'umol m-2 s-1',
                'description': 'Light threshold for stress onset',
                'references': ['Givnish (1988) in Plant Form and Function']
            },
            'temp_stress_threshold': {
                'units': 'K',
                'description': 'Temperature threshold for heat stress onset',
                'references': ['Larcher (2003) Physiological Plant Ecology']
            }
        }
        
        # Fetch parameters
        for attr_name, metadata in param_metadata.items():
            if getattr(self, attr_name) is None:
                param_key = f"{namespace}.stress.{attr_name}"
                value = self.get_parameter_from_manager(
                    param_key,
                    required=False,  # Make optional to allow test usage
                    register_if_missing=False,
                    metadata=metadata
                )
                setattr(self, attr_name, value)

    def _validate_parameters(self):
        """Validate stress parameters."""
        if self.temp_stress_max is not None and self.temp_stress_min is not None:
            if self.temp_stress_max <= self.temp_stress_min:
                raise ValueError("temp_stress_max must be greater than temp_stress_min")


@dataclass
class PFTDefinition:
    """
    Complete Plant Functional Type definition.
    
    This is the main PFT class that aggregates all parameter categories
    and provides methods for parameter access and validation.
    """

    parameter_manager: InitVar[UnifiedParameterManager] = field(kw_only=True)
    
    # Basic metadata
    name: str = ""                     # PFT name identifier
    category: PFTCategory = None       # PFT broad category
    description: str = ""               # Human-readable description
    
    # Parameter groups - will be initialized in __post_init__
    photosynthesis: PhotosynthesisParameters = field(default=None)
    phenology: PhenologyParameters = field(default=None) 
    allocation: AllocationParameters = field(default=None)
    respiration: RespirationParameters = field(default=None)
    bvoc: BVOCParameters = field(default=None)
    stress: StressParameters = field(default=None)
    
    # Validation and version metadata
    version: str = "1.0.0"              # Parameter set version
    source: str = "default"             # Parameter source/reference
    validation_status: str = "unchecked"  # Validation status
    
    # Whether to use ParameterManager for loading
    _use_param_manager: bool = field(default=True, repr=False, compare=False)
    # PFT namespace for parameter retrieval (used in tests)
    _pft_namespace: Optional[str] = field(default=None, repr=False, compare=False)
    
    def __post_init__(self, parameter_manager: UnifiedParameterManager):
        """Post-initialization validation and parameter loading."""
        if parameter_manager is None:
            raise TypeError("parameter_manager is required")
        self._parameter_manager = parameter_manager
        # Handle _pft_namespace parameter passed directly (for tests)
        if self._pft_namespace and not self.name:
            # Extract name from namespace like 'biosphere.pft.broadleaf_deciduous_temperate'
            parts = self._pft_namespace.split('.')
            if len(parts) >= 3 and parts[0] == 'biosphere' and parts[1] == 'pft':
                self.name = '.'.join(parts[2:])
                # Set category based on name if not set
                if not hasattr(self, 'category') or self.category is None:
                    try:
                        self.category = PFTCategory(self.name)
                    except ValueError:
                        # Default to a generic category if name doesn't match
                        self.category = PFTCategory.BROADLEAF_DECIDUOUS_TEMPERATE
        
        if not self.name:
            raise ValueError("PFT name cannot be empty")

        if not self._use_param_manager:
            return

        # Initialize parameter groups with proper namespace
        pft_namespace = self._pft_namespace or f"biosphere.pft.{self.name}"
        
        # Parameter group classes
        param_classes = {
            'photosynthesis': PhotosynthesisParameters,
            'phenology': PhenologyParameters,
            'allocation': AllocationParameters,
            'respiration': RespirationParameters,
            'bvoc': BVOCParameters,
            'stress': StressParameters
        }
        
        # Initialize parameter groups if they haven't been set
        for param_group_name, param_class in param_classes.items():
            current_value = getattr(self, param_group_name)
            if current_value is None:
                # Create new parameter group with namespace
                if self._use_param_manager:
                    param_group = param_class(
                        _pft_namespace=pft_namespace,
                        parameter_manager=self._parameter_manager,
                    )
                else:
                    # For tests or manual creation without parameter manager
                    param_group = param_class(
                        parameter_manager=self._parameter_manager
                    )
                setattr(self, param_group_name, param_group)
            else:
                # Update existing parameter group with namespace if needed
                if self._use_param_manager and hasattr(current_value, '_pft_namespace'):
                    current_value._pft_namespace = pft_namespace
                    # Trigger parameter loading if needed
                    if hasattr(current_value, '_fetch_from_parameter_manager'):
                        current_value._fetch_from_parameter_manager()
        
        # Cross-parameter consistency checks
        self._validate_consistency()
    
    def _validate_consistency(self):
        """Validate consistency between parameter groups."""
        
        # Check Jmax/Vcmax ratio consistency with pathway
        if (self.photosynthesis.jmax25 is not None and 
            self.photosynthesis.vcmax25 is not None and 
            self.photosynthesis.vcmax25 > 0):
            jmax_vcmax_ratio = self.photosynthesis.jmax25 / self.photosynthesis.vcmax25
            
            if self.photosynthesis.pathway == PhotosynthesisPathway.C3:
                if not 1.2 <= jmax_vcmax_ratio <= 2.5:
                    import warnings
                    warnings.warn(f"C3 Jmax/Vcmax ratio {jmax_vcmax_ratio:.2f} outside typical range [1.2, 2.5]")
            elif self.photosynthesis.pathway == PhotosynthesisPathway.C4:
                if not 0.8 <= jmax_vcmax_ratio <= 1.5:
                    import warnings
                    warnings.warn(f"C4 Jmax/Vcmax ratio {jmax_vcmax_ratio:.2f} outside typical range [0.8, 1.5]")
        
        # Check temperature consistency
        if (self.stress.temp_stress_min is not None and 
            self.phenology.temp_cold_kill is not None):
            if self.stress.temp_stress_min < self.phenology.temp_cold_kill:
                import warnings
                warnings.warn("Temperature stress minimum below cold kill temperature")
    
    def get_parameter(self, param_path: str) -> Union[float, str, Any]:
        """
        Get parameter value by dot notation path.
        
        Parameters
        ----------
        param_path : str
            Parameter path like 'photosynthesis.vcmax25' or 'allocation.f_leaf'
            
        Returns
        -------
        Union[float, str, Any]
            Parameter value
        """
        parts = param_path.split('.')
        obj = self
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise KeyError(f"Parameter path '{param_path}' not found")
                
        return obj
    
    def set_parameter(self, param_path: str, value: Any) -> None:
        """
        Set parameter value by dot notation path.
        
        Parameters  
        ----------
        param_path : str
            Parameter path like 'photosynthesis.vcmax25'
        value : Any
            New parameter value
        """
        parts = param_path.split('.')
        obj = self
        
        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise KeyError(f"Parameter path '{param_path}' not found")
        
        # Set the final attribute
        final_attr = parts[-1]
        if hasattr(obj, final_attr):
            # First update the value in ParameterManager if we're using it
            if self._use_param_manager:
                # Construct the full parameter key
                # For nested parameters like photosynthesis.vcmax25
                full_key = f"biosphere.pft.{self.name}.{param_path}"
                
                # Update in parameter manager; a rejected value raises before the local
                # value changes, so the manager and the PFT never disagree.
                set_parameter_value(full_key, value, validate=True, notify=False)
            
            # Then set the local value
            setattr(obj, final_attr, value)
            
            # Only validate, don't re-fetch from parameter manager
            if hasattr(obj, '_validate_parameters'):
                obj._validate_parameters()
        else:
            raise KeyError(f"Parameter path '{param_path}' not found")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PFT definition to dictionary for serialization.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of PFT
        """
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        parameter_manager: UnifiedParameterManager,
    ) -> 'PFTDefinition':
        """
        Create PFT definition from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with PFT data
            
        Returns
        -------
        PFTDefinition
            PFT definition instance
        """
        # Handle enum conversion
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = PFTCategory(data['category'])
        
        # Handle pathway conversion in photosynthesis parameters
        if 'photosynthesis' in data and 'pathway' in data['photosynthesis']:
            if isinstance(data['photosynthesis']['pathway'], str):
                data['photosynthesis']['pathway'] = PhotosynthesisPathway(data['photosynthesis']['pathway'])
        
        # Extract PFT name for namespace construction
        pft_name = data.get('name', '')
        pft_namespace = f"biosphere.pft.{pft_name}" if pft_name else None
        
        # Create parameter objects with namespace
        param_classes = {
            'photosynthesis': PhotosynthesisParameters,
            'phenology': PhenologyParameters,
            'allocation': AllocationParameters, 
            'respiration': RespirationParameters,
            'bvoc': BVOCParameters,
            'stress': StressParameters
        }
        
        for param_group, param_class in param_classes.items():
            if param_group in data and isinstance(data[param_group], dict):
                # Add namespace to the parameter group
                if pft_namespace:
                    data[param_group]['_pft_namespace'] = pft_namespace
                data[param_group] = param_class(
                    **data[param_group],
                    parameter_manager=parameter_manager,
                )
        
        return cls(**data, parameter_manager=parameter_manager)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of key PFT parameters for quick reference.
        
        Returns
        -------
        Dict[str, Any]
            Summary dictionary with key parameters
        """
        return {
            'name': self.name,
            'category': self.category.value if self.category else None,
            'pathway': self.photosynthesis.pathway.value if self.photosynthesis.pathway else None,
            'vcmax25': self.photosynthesis.vcmax25,
            'jmax25': self.photosynthesis.jmax25,
            'leaf_longevity': self.phenology.leaf_longevity,
            'allocation_fractions': {
                'leaf': self.allocation.f_leaf,
                'stem': self.allocation.f_stem, 
                'root': self.allocation.f_root
            },
            'isoprene_emission': self.bvoc.ef_isoprene,
            'ozone_sensitivity': self.photosynthesis.o3_sensitivity
        }


# Type aliases for convenience
PFTParameterDict = Dict[str, Union[float, int, str]]
PFTCollection = Dict[str, PFTDefinition]

# Export all classes and enums
__all__ = [
    'PhotosynthesisPathway',
    'PFTCategory', 
    'PhotosynthesisParameters',
    'PhenologyParameters',
    'AllocationParameters',
    'RespirationParameters',
    'BVOCParameters',
    'StressParameters',
    'PFTDefinition',
    'PFTParameterDict',
    'PFTCollection'
]
