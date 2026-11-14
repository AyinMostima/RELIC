"""
Runtime-Disabled Candidate Configuration Models for 13C, 15N, and 18O

The schemas in this module describe candidate isotope pools and processes.
They do not enable isotope runtime execution. The production default keeps all
three species disabled until canonical parent selection, nuclide-mass
conversion, StateManager component ownership, and oxygen-17 closure exist.

Architecture:
    - IsotopeConfig: Runtime-disabled candidate root for all isotope species
    - IsotopeSpeciesConfig: Species-specific configuration (13C, 15N, 18O)
    - CarbonProcessConfig: Carbon cycle processes
    - OxygenProcessConfig: Oxygen/water cycle processes
    - CrossIsotopeConstraints: Inter-isotope consistency checks

Mathematical Framework:
    - R = n_heavy / n_light is a heavy-to-light amount ratio.
    - delta = 1000 * (R_sample / R_standard - 1) permil.
    - Oxygen runtime requires explicit 16O, 17O, and 18O closure.

References:
    - Farquhar et al. (1989). Carbon isotope discrimination and photosynthesis
    - Craig & Gordon (1965). Deuterium and oxygen-18 variations in the ocean
    - Mariotti et al. (1981). Nitrogen isotope fractionation
"""

from typing import ClassVar, Dict, Any, FrozenSet, Literal, Optional, Set, Union
from pydantic import Field, ValidationInfo, field_validator, model_validator
from enum import Enum
import warnings

# Import base models from the existing configuration system
from relic.parameters import get_global_manager
from .models import (
    IsotopeConfigurationModel, FractionationModelType, FractionationParameters,
    ProcessConfig, PoolConfig, OutputVariableConfig,
    OutputConfig, Nitrogen15Config, _default_output_config
)


def _require_param(key: str, *, units: str, expected_type: type = float) -> Any:
    """Fetch a required parameter from the unified parameter access path."""
    value = get_global_manager().get(
        key,
        units=units,
        required=True,
        expected_type=None,
    )
    if expected_type in (float, int):
        try:
            return expected_type(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Isotope parameter '{key}' must be convertible to {expected_type.__name__}."
            ) from exc
    if expected_type is str:
        return str(value)
    return value


def _get_bounds(key: str) -> Optional[tuple]:
    """Return optional bounds metadata for warning-only checks."""
    _ = key
    # Bounds validation is already enforced in _require_param(..., validate=True).
    return None


def _read_standards() -> Dict["IsotopeStandard", float]:
    """Read isotope standard ratios from the canonical ParameterManager."""
    return {
        IsotopeStandard.PDB: _require_param(
            "isotopes.standards.carbon_13.pdb_ratio", units="dimensionless", expected_type=float
        ),
        IsotopeStandard.AIR_N2: _require_param(
            "isotopes.standards.nitrogen_15.air_ratio", units="dimensionless", expected_type=float
        ),
        IsotopeStandard.VSMOW: _require_param(
            "isotopes.standards.oxygen_18.vsmow_ratio", units="dimensionless", expected_type=float
        ),
    }


def get_isotope_standards() -> Dict["IsotopeStandard", float]:
    """Return isotope standard ratios loaded from the parameter system."""
    return dict(_read_standards())


class IsotopeSpecies(str, Enum):
    """Supported isotope species."""
    CARBON_13 = "13C"
    NITROGEN_15 = "15N" 
    OXYGEN_18 = "18O"


class CarbonProcessType(str, Enum):
    """Carbon cycle processes for 13C tracking."""
    PHOTOSYNTHESIS = "photosynthesis"
    LEAF_RESPIRATION = "leaf_respiration"
    ROOT_RESPIRATION = "root_respiration"
    AUTOTROPHIC_RESPIRATION = "autotrophic_respiration"
    HETEROTROPHIC_RESPIRATION = "heterotrophic_respiration"
    DECOMPOSITION = "decomposition"
    BVOC_EMISSION = "bvoc_emission"


class OxygenProcessType(str, Enum):
    """Oxygen/water cycle processes for 18O tracking."""
    TRANSPIRATION = "transpiration"
    SOIL_EVAPORATION = "soil_evaporation"
    CANOPY_EVAPORATION = "canopy_evaporation"
    WATER_UPTAKE = "water_uptake"
    LEAF_WATER_EXCHANGE = "leaf_water_exchange"


class PhotosynthesisType(str, Enum):
    """Photosynthesis pathway types."""
    C3 = "C3"
    C4 = "C4"
    CAM = "CAM"


class IsotopeStandard(str, Enum):
    """Isotope reference standards."""
    PDB = "PDB"  # 13C standard
    AIR_N2 = "AIR_N2"  # 15N standard
    VSMOW = "VSMOW"  # 18O standard


# Standard isotope ratios are loaded from ParameterManager via get_isotope_standards().


class CarbonFractionationParameters(IsotopeConfigurationModel):
    """
    Carbon isotope fractionation parameters for photosynthesis.
    
    Based on Farquhar et al. (1989) model for C3 photosynthesis:
    delta13C_plant = delta13C_air - a - (b-a)(ci/ca)
    
    Where:
    - a: diffusion fractionation (4.4?)
    - b: carboxylation fractionation (27? for Rubisco)
    - ci/ca: intercellular to atmospheric CO2 ratio
    
    Parameters:
        photosynthesis_type: C3, C4, or CAM pathway
        diffusion_fractionation: CO2 diffusion fractionation factor (a)
        carboxylation_fractionation: Rubisco fractionation factor (b)
        (ci/ca is a physiological state and an input of the calculation, not a parameter)
    """
    
    photosynthesis_type: PhotosynthesisType = Field(
        ...,
        description="Photosynthetic pathway type"
    )
    
    # Farquhar model parameters for C3 plants
    diffusion_fractionation: float = Field(
        ...,
        description="CO2 diffusion fractionation factor (a) [per mil]"
    )
    
    carboxylation_fractionation: float = Field(
        ...,
        description="Rubisco carboxylation fractionation (b) [per mil]"
    )
    
    # C4 law parameters (Farquhar et al. 1989, Eq. 10)
    c4_bundle_sheath_leakiness: float = Field(
        ...,
        description="C4 bundle-sheath leakiness phi [1]"
    )

    c4_pep_net_fractionation: float = Field(
        ...,
        description=(
            "C4 net discrimination b4 of the fixation of gaseous CO2 by PEP carboxylase "
            "[per mil] (-5.7 at 25 degC)"
        ),
    )

    c4_rubisco_fractionation: float = Field(
        ...,
        description="Rubisco discrimination b3 of the C4 bundle sheath [per mil]"
    )
    
    @classmethod
    def from_manager(cls, manager: Any) -> "CarbonFractionationParameters":
        """The one construction of the configured carbon fractionation parameter set.

        Every field is read from its isotopes.fractionation.carbon key of the given
        ParameterManager (the calculator's injected manager, or the global manager of
        the configuration layer); a missing key raises."""
        prefix = "isotopes.fractionation.carbon"

        def read(name: str) -> Any:
            return manager.get(f"{prefix}.{name}", units="dimensionless", required=True)

        return cls(
            photosynthesis_type=PhotosynthesisType(str(read("photosynthesis_type"))),
            diffusion_fractionation=float(read("diffusion_fractionation")),
            carboxylation_fractionation=float(read("carboxylation_fractionation")),
            c4_bundle_sheath_leakiness=float(read("c4_bundle_sheath_leakiness")),
            c4_pep_net_fractionation=float(read("c4_pep_net_fractionation")),
            c4_rubisco_fractionation=float(read("c4_rubisco_fractionation")),
        )
    
    @model_validator(mode='after')
    def validate_photosynthesis_parameters(self):
        """Validate parameters based on photosynthesis type."""
        if self.photosynthesis_type == PhotosynthesisType.C3:
            # C3 plants use Farquhar model parameters
            bounds = _get_bounds("isotopes.fractionation.carbon.carboxylation_fractionation")
            if bounds and (self.carboxylation_fractionation < bounds[0] or self.carboxylation_fractionation > bounds[1]):
                warnings.warn("C3 carboxylation fractionation outside parameter bounds")
        
        elif self.photosynthesis_type == PhotosynthesisType.C4:
            # C4 plants have different fractionation
            bounds = _get_bounds("isotopes.fractionation.carbon.c4_pep_net_fractionation")
            if bounds and (
                self.c4_pep_net_fractionation < bounds[0]
                or self.c4_pep_net_fractionation > bounds[1]
            ):
                warnings.warn("C4 PEP fractionation outside parameter bounds")
        
        return self


class OxygenFractionationParameters(IsotopeConfigurationModel):
    """
    Oxygen isotope fractionation parameters for water processes.
    
    Based on Craig-Gordon model for leaf water enrichment:
    delta18O_leaf = delta18O_source + epsilon* + epsilonk * (ea - ei)/ea
    
    Where:
    - epsilon*: equilibrium fractionation factor
    - epsilonk: kinetic fractionation factor
    - ea: ambient vapor pressure
    - ei: intercellular vapor pressure
    
    Parameters:
        kinetic_fractionation: Kinetic fractionation during diffusion
        humidity_sensitivity: Sensitivity to relative humidity
    """
    
    # Craig-Gordon model parameters
    kinetic_fractionation: float = Field(
        default_factory=lambda: _require_param(
            "isotopes.fractionation.oxygen.kinetic_fractionation",
            units="dimensionless",
            expected_type=float,
        ),
        description="Kinetic fractionation during diffusion [per mil]"
    )
    
    reference_temperature: float = Field(
        default_factory=lambda: _require_param(
            "isotopes.temperature.reference",
            units="K",
            expected_type=float,
        ),
        description="Reference temperature for fractionation [K]"
    )
    
    # Environmental sensitivity
    humidity_sensitivity: float = Field(
        default_factory=lambda: _require_param(
            "isotopes.fractionation.oxygen.humidity_sensitivity",
            units="dimensionless",
            expected_type=float,
        ),
        description="Sensitivity to relative humidity changes"
    )
    
    boundary_layer_conductance: Optional[float] = Field(
        default_factory=lambda: _require_param(
            "isotopes.fractionation.oxygen.boundary_layer_conductance",
            units="m s-1",
            expected_type=float,
        ),
        description="Boundary layer conductance [m/s]"
    )
    
    # Peclet effect for leaf water enrichment
    peclet_number: float = Field(
        default_factory=lambda: _require_param(
            "isotopes.fractionation.oxygen.peclet_number",
            units="dimensionless",
            expected_type=float,
        ),
        description="Peclet number for leaf water advection"
    )
    
class CarbonProcessConfig(ProcessConfig):
    """
    Carbon-specific process configuration extending base ProcessConfig.
    
    Adds carbon cycle specific parameters for 13C fractionation
    in photosynthesis, respiration, and decomposition processes.
    """
    
    carbon_fractionation: Optional[CarbonFractionationParameters] = Field(
        None,
        description="Carbon-specific fractionation parameters"
    )
    
    process_type: CarbonProcessType = Field(
        ...,
        description="Type of carbon cycle process"
    )
    
    # Substrate and product pools
    substrate_pool: Optional[str] = Field(
        None,
        description="Carbon substrate pool (e.g., glucose, cellulose)"
    )
    
    product_pool: Optional[str] = Field(
        None,
        description="Carbon product pool (e.g., CO2, biomass)"
    )
    
    # Environmental controls
    light_dependence: bool = Field(
        False,
        description="Whether process depends on light availability"
    )
    
    moisture_dependence: bool = Field(
        False,
        description="Whether process depends on soil moisture"
    )
    
    @model_validator(mode='after')
    def validate_carbon_process(self):
        """Validate carbon process configuration."""
        # Photosynthesis must have carbon fractionation parameters
        if (self.process_type == CarbonProcessType.PHOTOSYNTHESIS and 
            self.carbon_fractionation is None):
            raise ValueError("Photosynthesis process requires carbon_fractionation parameters")
        
        # Light-dependent processes
        light_dependent_processes = {
            CarbonProcessType.PHOTOSYNTHESIS,
            CarbonProcessType.BVOC_EMISSION
        }
        
        if self.process_type in light_dependent_processes:
            object.__setattr__(self, "light_dependence", True)
        
        return self


class OxygenProcessConfig(ProcessConfig):
    """
    Oxygen-specific process configuration extending base ProcessConfig.
    
    Adds water cycle specific parameters for 18O fractionation
    in transpiration, evaporation, and water exchange processes.
    """
    
    oxygen_fractionation: Optional[OxygenFractionationParameters] = Field(
        None,
        description="Oxygen-specific fractionation parameters"
    )
    
    process_type: OxygenProcessType = Field(
        ...,
        description="Type of water/oxygen cycle process"
    )
    
    # Water source and sink
    water_source: Optional[str] = Field(
        None,
        description="Water source pool (e.g., soil_water, leaf_water)"
    )
    
    water_sink: Optional[str] = Field(
        None,
        description="Water sink pool (e.g., atmosphere, groundwater)"
    )
    
    # Environmental controls
    radiation_dependence: bool = Field(
        False,
        description="Whether process depends on radiation"
    )
    
    vapor_pressure_dependence: bool = Field(
        True,
        description="Whether process depends on vapor pressure deficit"
    )
    
    wind_dependence: bool = Field(
        False,
        description="Whether process depends on wind speed"
    )
    
    @model_validator(mode='after')
    def validate_oxygen_process(self):
        """Validate oxygen process configuration."""
        # Transpiration and evaporation need fractionation parameters
        if (self.process_type in {OxygenProcessType.TRANSPIRATION, 
                                  OxygenProcessType.SOIL_EVAPORATION,
                                  OxygenProcessType.CANOPY_EVAPORATION} and
            self.oxygen_fractionation is None):
            raise ValueError(f"{self.process_type} process requires oxygen_fractionation parameters")
        
        # Evaporation processes depend on radiation and wind
        evap_processes = {
            OxygenProcessType.TRANSPIRATION,
            OxygenProcessType.SOIL_EVAPORATION,
            OxygenProcessType.CANOPY_EVAPORATION
        }
        
        if self.process_type in evap_processes:
            object.__setattr__(self, "radiation_dependence", True)
            object.__setattr__(self, "wind_dependence", True)
        
        return self


class IsotopeSpeciesConfig(IsotopeConfigurationModel):
    """
    Configuration for a specific isotope species (13C, 15N, or 18O).
    
    Defines the complete setup for tracking one isotope species
    including its pools, processes, and species-specific parameters.
    
    Parameters:
        species: Isotope species identifier
        standard: Reference standard for delta calculations
        standard_heavy_to_light_ratio: Standard heavy-to-light amount ratio
        ratio_convention: Definition of the configured isotope ratio
        component_amount_basis: Basis required for component state amounts
        runtime_blocker: Missing contract that prevents runtime enablement
        enabled: Whether to track this isotope species
        pools: Species-specific pool configurations
        processes: Species-specific process configurations
    """

    model_config = {
        **IsotopeConfigurationModel.model_config,
        "revalidate_instances": "always",
    }
    _immutable_contract_fields: ClassVar[FrozenSet[str]] = frozenset({
        "species",
        "standard",
        "standard_heavy_to_light_ratio",
        "ratio_convention",
        "component_amount_basis",
        "runtime_blocker",
    })
    
    # Species identification
    species: IsotopeSpecies = Field(
        ...,
        description="Isotope species identifier"
    )
    
    standard: IsotopeStandard = Field(
        ...,
        description="Reference standard for delta calculations"
    )
    
    standard_heavy_to_light_ratio: float = Field(
        ...,
        gt=0.0,
        description="Reference heavy-to-light isotope amount ratio"
    )

    ratio_convention: Literal["heavy_to_light_amount_ratio"] = Field(
        "heavy_to_light_amount_ratio",
        description="R = n_heavy / n_light on an amount-of-substance basis"
    )

    component_amount_basis: Literal[
        "amount_of_substance",
        "element_mass",
    ] = Field(
        ...,
        description="Physical basis of the candidate component amounts"
    )

    runtime_blocker: str = Field(
        ...,
        min_length=1,
        description="Fail-closed reason for the unimplemented runtime binding"
    )

    enabled: bool = Field(
        False,
        description="Whether the fully governed runtime binding is enabled"
    )
    
    # Species-specific pools
    pools: Dict[str, PoolConfig] = Field(
        default_factory=dict,
        description="Pools specific to this isotope species"
    )
    
    # Species-specific processes
    carbon_processes: Dict[str, CarbonProcessConfig] = Field(
        default_factory=dict,
        description="Carbon cycle processes (for 13C only)"
    )
    
    nitrogen_processes: Dict[str, ProcessConfig] = Field(
        default_factory=dict,
        description="Nitrogen cycle processes (for 15N only)"
    )
    
    oxygen_processes: Dict[str, OxygenProcessConfig] = Field(
        default_factory=dict,
        description="Oxygen/water cycle processes (for 18O only)"
    )
    
    # Delta notation calculations
    delta_calculation: bool = Field(
        False,
        description="Candidate delta intent; disabled until runtime closure"
    )
    
    # Species-specific output variables
    output_variables: Dict[str, OutputVariableConfig] = Field(
        default_factory=dict,
        description="Species-specific output variable configurations"
    )

    @field_validator("enabled", mode="before")
    @classmethod
    def reject_runtime_enablement(
        cls,
        value: bool,
        info: ValidationInfo,
    ) -> bool:
        """Reject runtime enablement before assignment mutates the object."""
        if value is not False:
            blocker = info.data.get("runtime_blocker", "runtime_unbound")
            raise ValueError(f"[N] runtime_unbound: {blocker}")
        return False

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep scientific identity fields immutable after construction."""
        if name in self._immutable_contract_fields and name in self.__dict__:
            raise ValueError(f"Isotope candidate field '{name}' is immutable")
        super().__setattr__(name, value)
    
    @model_validator(mode='after')
    def validate_species_consistency(self):
        """Validate species configuration consistency."""
        # Check standard matches species
        expected_standards = {
            IsotopeSpecies.CARBON_13: IsotopeStandard.PDB,
            IsotopeSpecies.NITROGEN_15: IsotopeStandard.AIR_N2,
            IsotopeSpecies.OXYGEN_18: IsotopeStandard.VSMOW
        }
        
        if self.species in expected_standards:
            if self.standard != expected_standards[self.species]:
                raise ValueError(
                    f"Unusual standard {self.standard} for {self.species}, "
                    f"expected {expected_standards[self.species]}"
                )
        
        # Check the standard ratio against ParameterManager.
        standards = get_isotope_standards()
        expected_ratios = {
            IsotopeSpecies.CARBON_13: standards[IsotopeStandard.PDB],
            IsotopeSpecies.NITROGEN_15: standards[IsotopeStandard.AIR_N2],
            IsotopeSpecies.OXYGEN_18: standards[IsotopeStandard.VSMOW],
        }
        tolerance = _require_param(
            "isotopes.validation.abundance_tolerance_fraction",
            units="dimensionless",
            expected_type=float,
        )
        if self.species in expected_ratios:
            expected = expected_ratios[self.species]
            if (
                abs(self.standard_heavy_to_light_ratio - expected)
                > expected * tolerance
            ):
                raise ValueError(
                    "Standard heavy-to-light ratio "
                    f"{self.standard_heavy_to_light_ratio} for {self.species} "
                    f"differs from expected {expected}"
                )
        
        # Check processes are appropriate for species
        if self.species != IsotopeSpecies.CARBON_13 and self.carbon_processes:
            raise ValueError(f"Carbon processes specified for non-carbon species {self.species}")
        
        if self.species != IsotopeSpecies.NITROGEN_15 and self.nitrogen_processes:
            raise ValueError(f"Nitrogen processes specified for non-nitrogen species {self.species}")
        
        if self.species != IsotopeSpecies.OXYGEN_18 and self.oxygen_processes:
            raise ValueError(f"Oxygen processes specified for non-oxygen species {self.species}")

        return self

    def model_copy(
        self,
        *,
        update: Optional[Dict[str, Any]] = None,
        deep: bool = False,
    ) -> "IsotopeSpeciesConfig":
        """Return a fully revalidated copy; updates cannot bypass blockers."""
        _ = deep
        payload = self.model_dump(mode="python", round_trip=True)
        if update:
            payload.update(update)
        return type(self).model_validate(payload)

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: Optional[Dict[str, Any]] = None,
        deep: bool = False,
    ) -> "IsotopeSpeciesConfig":
        """Route the compatibility copy API through strict revalidation."""
        if include is not None or exclude is not None:
            raise ValueError("Strict isotope copy does not allow include/exclude")
        return self.model_copy(update=update, deep=deep)
    
    def get_all_processes(self) -> Dict[str, Union[CarbonProcessConfig, ProcessConfig, OxygenProcessConfig]]:
        """Get all processes for this species combined."""
        all_processes = {}
        all_processes.update(self.carbon_processes)
        all_processes.update(self.nitrogen_processes)
        all_processes.update(self.oxygen_processes)
        return all_processes


class CrossIsotopeConstraints(IsotopeConfigurationModel):
    """
    Cross-isotope constraints and consistency checks.
    
    Defines relationships and constraints between different isotope
    species that must be maintained for physical consistency.
    
    Parameters:
        mass_balance_check: Whether to check mass balance across isotopes
        thermodynamic_consistency: Check thermodynamic constraints
        stoichiometric_constraints: Enforce stoichiometric relationships
    """
    
    # Mass balance constraints
    mass_balance_check: bool = Field(
        False,
        description="Candidate mass-balance intent; runtime disabled"
    )
    
    mass_balance_tolerance: float = Field(
        default_factory=lambda: _require_param(
            "isotopes.validation.numerical.conservation_tolerance",
            units="dimensionless",
            expected_type=float,
        ),
        description="Tolerance for mass balance violations"
    )
    
    # Thermodynamic consistency
    thermodynamic_consistency: bool = Field(
        False,
        description="Candidate thermodynamic intent; runtime disabled"
    )
    
    # Stoichiometric relationships
    stoichiometric_constraints: bool = Field(
        False,
        description="Candidate stoichiometric intent; runtime disabled"
    )
    
    # Cross-species correlations
    c13_n15_correlation: bool = Field(
        False,
        description="Enable C-N isotope correlation constraints"
    )
    
    o18_h2_correlation: bool = Field(
        False,
        description="Enable O-H isotope correlation (if deuterium tracked)"
    )
    
    # Constraint violation handling
    raise_on_violation: bool = Field(
        False,
        description="Raise exception on constraint violation"
    )
    
    log_violations: bool = Field(
        True,
        description="Log constraint violations"
    )


class IsotopeConfig(IsotopeConfigurationModel):
    """
    Main configuration for multi-isotope tracking system.
    
    Root configuration that defines the complete setup for tracking
    multiple isotope species (13C, 15N, 18O) with their interactions
    and constraints.
    
    Parameters:
        tracked_species: Set of isotope species to track
        species_configs: Individual configurations for each species
        cross_constraints: Cross-isotope constraints
        unified_output: Unified output configuration
        integration_timestep: Integration timestep for isotope calculations
    """

    model_config = {
        **IsotopeConfigurationModel.model_config,
        "revalidate_instances": "always",
    }
    _immutable_root_fields: ClassVar[FrozenSet[str]] = frozenset({
        "species_configs",
    })
    
    # Metadata
    config_version: str = Field(
        "2.0.0",
        description="Multi-isotope configuration schema version"
    )
    
    module_name: str = Field(
        "isotope_candidate_config",
        description="Module identifier"
    )
    
    description: str = Field(
        "Runtime-disabled candidate configuration for 13C, 15N, and 18O",
        description="Module description"
    )
    
    # Core settings
    tracked_species: FrozenSet[IsotopeSpecies] = Field(
        default_factory=frozenset,
        description="Immutable set; empty until a runtime binding is implemented"
    )
    
    enabled: bool = Field(
        False,
        description="Root runtime enablement; currently fail-closed"
    )
    
    # Individual species configurations
    species_configs: Dict[IsotopeSpecies, IsotopeSpeciesConfig] = Field(
        default_factory=dict,
        description="Configuration for each isotope species"
    )
    
    # Cross-isotope constraints
    cross_constraints: CrossIsotopeConstraints = Field(
        default_factory=CrossIsotopeConstraints,
        description="Cross-isotope constraints and consistency checks"
    )
    
    # Unified output configuration
    unified_output: OutputConfig = Field(
        default_factory=_default_output_config,
        description="Unified output configuration for all species"
    )
    
    # Integration settings
    integration_timestep: float = Field(
        default_factory=lambda: _require_param(
            "isotopes.multi.integration_timestep",
            units="s",
            expected_type=float,
        ),
        description="Integration timestep for isotope calculations [s]"
    )
    
    adaptive_timestep: bool = Field(
        False,
        description="Use adaptive timestep based on process rates"
    )
    
    # Numerical settings
    numerical_solver: str = Field(
        default_factory=lambda: _require_param(
            "isotopes.multi.numerical_solver",
            units="dimensionless",
            expected_type=str,
        ),
        description="Numerical solver for isotope integration"
    )
    
    convergence_tolerance: float = Field(
        default_factory=lambda: _require_param(
            "isotopes.multi.convergence_tolerance",
            units="dimensionless",
            expected_type=float,
        ),
        description="Convergence tolerance for iterative solvers"
    )
    
    max_iterations: int = Field(
        default_factory=lambda: _require_param(
            "isotopes.multi.max_iterations",
            units="dimensionless",
            expected_type=int,
        ),
        description="Maximum iterations for iterative solvers"
    )

    @field_validator("tracked_species", mode="before")
    @classmethod
    def reject_tracked_species(
        cls,
        value: Any,
    ) -> FrozenSet[IsotopeSpecies]:
        """Reject non-empty tracking declarations before object mutation."""
        requested = frozenset(value or ())
        if requested:
            raise ValueError("[N] isotope tracked species are runtime disabled")
        return frozenset()

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep the species mapping immutable after construction."""
        if name in self._immutable_root_fields and name in self.__dict__:
            raise ValueError(f"Isotope root field '{name}' is immutable")
        super().__setattr__(name, value)

    @field_validator("enabled", mode="before")
    @classmethod
    def reject_root_enablement(cls, value: bool) -> bool:
        """Reject root runtime enablement before object mutation."""
        if value is not False:
            raise ValueError("[N] isotope root runtime_unbound")
        return False
    
    @model_validator(mode='after')
    def validate_multi_isotope_configuration(self):
        """Validate multi-isotope configuration consistency."""
        # Check that all tracked species have configurations
        for species in self.tracked_species:
            if species not in self.species_configs:
                raise ValueError(f"Missing configuration for tracked species: {species}")
        
        # Check for unused species configurations
        configured_species = set(self.species_configs.keys())
        unused_species = configured_species - self.tracked_species
        unused_enabled_species = {
            species
            for species in unused_species
            if self.species_configs[species].enabled
        }
        if unused_enabled_species:
            warnings.warn(
                f"Unused enabled species configurations: {unused_enabled_species}"
            )
        
        # Validate species configurations
        for species, config in self.species_configs.items():
            if config.species != species:
                raise ValueError(
                    f"Species mismatch: key {species} != config.species {config.species}"
                )
            
            if species in self.tracked_species and not config.enabled:
                raise ValueError(
                    f"[N] tracked species {species.value} is runtime disabled: "
                    f"{config.runtime_blocker}"
                )
        
        return self

    def model_copy(
        self,
        *,
        update: Optional[Dict[str, Any]] = None,
        deep: bool = False,
    ) -> "IsotopeConfig":
        """Return a fully revalidated copy; updates cannot bypass blockers."""
        _ = deep
        payload = self.model_dump(mode="python", round_trip=True)
        if update:
            payload.update(update)
        return type(self).model_validate(payload)

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: Optional[Dict[str, Any]] = None,
        deep: bool = False,
    ) -> "IsotopeConfig":
        """Route the compatibility copy API through strict revalidation."""
        if include is not None or exclude is not None:
            raise ValueError("Strict isotope copy does not allow include/exclude")
        return self.model_copy(update=update, deep=deep)
    
    def get_enabled_species(self) -> Set[IsotopeSpecies]:
        """Get set of enabled isotope species."""
        if not self.enabled:
            return set()
        return {
            species for species in self.tracked_species
            if species in self.species_configs and self.species_configs[species].enabled
        }
    
    def get_all_pools(self) -> Dict[str, PoolConfig]:
        """Get all pools from all species."""
        all_pools = {}
        for species_config in self.species_configs.values():
            if species_config.enabled:
                for pool_name, pool_config in species_config.pools.items():
                    # Prefix with species to avoid conflicts
                    prefixed_name = f"{species_config.species.value}_{pool_name}"
                    all_pools[prefixed_name] = pool_config
        return all_pools
    
    def get_all_processes(self) -> Dict[str, Union[CarbonProcessConfig, ProcessConfig, OxygenProcessConfig]]:
        """Get all processes from all species."""
        all_processes = {}
        for species_config in self.species_configs.values():
            if species_config.enabled:
                species_processes = species_config.get_all_processes()
                for process_name, process_config in species_processes.items():
                    # Prefix with species to avoid conflicts
                    prefixed_name = f"{species_config.species.value}_{process_name}"
                    all_processes[prefixed_name] = process_config
        return all_processes
    
    def create_default_carbon_config(self) -> IsotopeSpeciesConfig:
        """Create default 13C configuration."""
        standards = get_isotope_standards()
        temp_ref = _require_param("isotopes.temperature.reference", units="K", expected_type=float)
        temp_min = _require_param("isotopes.temperature.min", units="K", expected_type=float)
        temp_max = _require_param("isotopes.temperature.max", units="K", expected_type=float)
        photosynth_alpha = _require_param(
            "isotopes.fractionation.carbon.photosynthesis.alpha_0",
            units="dimensionless",
            expected_type=float,
        )
        leaf_alpha = _require_param(
            "isotopes.fractionation.carbon.respiration.leaf.alpha_0",
            units="dimensionless",
            expected_type=float,
        )
        leaf_q10 = _require_param(
            "isotopes.fractionation.carbon.respiration.leaf.q10",
            units="dimensionless",
            expected_type=float,
        )
        root_alpha = _require_param(
            "isotopes.fractionation.carbon.respiration.root.alpha_0",
            units="dimensionless",
            expected_type=float,
        )
        root_q10 = _require_param(
            "isotopes.fractionation.carbon.respiration.root.q10",
            units="dimensionless",
            expected_type=float,
        )
        decomp_alpha = _require_param(
            "isotopes.fractionation.carbon.decomposition.alpha_0",
            units="dimensionless",
            expected_type=float,
        )
        decomp_q10 = _require_param(
            "isotopes.fractionation.carbon.decomposition.q10",
            units="dimensionless",
            expected_type=float,
        )
        return IsotopeSpeciesConfig(
            species=IsotopeSpecies.CARBON_13,
            standard=IsotopeStandard.PDB,
            standard_heavy_to_light_ratio=standards[IsotopeStandard.PDB],
            component_amount_basis="amount_of_substance",
            enabled=False,
            runtime_blocker=(
                "canonical_parent_selection_unbound"
                "(atmospheric_co2,soil_organic_carbon)+"
                "open_boundary_ratio_unbound+vpdb_standard_unbound+"
                "initial_ratio_semantics_unbound+"
                "parent_mass_to_amount_conversion_unbound+"
                "state_manager_component_ownership_unbound+"
                "executable_output_mapping_unbound"
            ),
            pools={
                "atmospheric_co2": PoolConfig(
                    pool_name="atmospheric_co2",
                    state_variable="atmospheric_co2",
                    initial_abundance=standards[IsotopeStandard.PDB],
                    units="mol m-2",
                    description="Atmospheric CO2 carbon pool"
                ),
                "leaf_biomass": PoolConfig(
                    pool_name="leaf_biomass",
                    # Carbon-basis parent: leaf_biomass_c (state, kg C m-2, written by
                    # biomass_allocation); a carbon isotope ratio needs a carbon-basis parent.
                    state_variable="leaf_biomass_c",
                    initial_abundance=standards[IsotopeStandard.PDB],
                    # kg C m-2 -> mol C m-2 is the declared, still unbound mass-to-amount
                    # conversion at the boundary (runtime blocker).
                    units="kg C m-2",
                    description="Leaf biomass carbon pool"
                ),
                "root_biomass": PoolConfig(
                    pool_name="root_biomass",
                    # Carbon-basis parent: root_biomass_c (state, kg C m-2, written by
                    # biomass_allocation); the dry-matter root_biomass [kg DM m-2] cannot
                    # carry a carbon isotope ratio without a carbon fraction.
                    state_variable="root_biomass_c",
                    initial_abundance=standards[IsotopeStandard.PDB], 
                    # kg C m-2 -> mol C m-2 is the declared, still unbound mass-to-amount
                    # conversion at the boundary (runtime blocker).
                    units="kg C m-2",
                    description="Root biomass carbon pool"
                ),
                "soil_organic_carbon": PoolConfig(
                    pool_name="soil_organic_carbon",
                    state_variable="soil_organic_carbon",
                    initial_abundance=standards[IsotopeStandard.PDB],
                    # Parent contract unit; kg C m-2 -> mol C m-2 is the declared, still
                    # unbound mass-to-amount conversion at the boundary (runtime blocker).
                    units="kg C m-2",
                    description="Soil organic carbon pool"
                )
            },
            carbon_processes={
                "photosynthesis": CarbonProcessConfig(
                    process_name="photosynthesis",
                    enabled=False,
                    process_type=CarbonProcessType.PHOTOSYNTHESIS,
                    fractionation=FractionationParameters(
                        model_type=FractionationModelType.CONSTANT,
                        alpha_0=photosynth_alpha,
                        T_ref=temp_ref,
                        T_min=temp_min,
                        T_max=temp_max,
                    ),
                    carbon_fractionation=CarbonFractionationParameters.from_manager(
                        get_global_manager()
                    ),
                    source_pool="atmospheric_co2",
                    target_pool="leaf_biomass",
                    flux_variable="photosynthesis_flux",
                    references=["Farquhar et al. (1989)", "O'Leary (1988)"]
                ),
                "leaf_respiration": CarbonProcessConfig(
                    process_name="leaf_respiration",
                    enabled=False,
                    process_type=CarbonProcessType.LEAF_RESPIRATION,
                    fractionation=FractionationParameters(
                        model_type=FractionationModelType.Q10,
                        alpha_0=leaf_alpha,
                        Q10=leaf_q10,
                        T_ref=temp_ref,
                        T_min=temp_min,
                        T_max=temp_max,
                    ),
                    source_pool="leaf_biomass",
                    target_pool="atmospheric_co2", 
                    flux_variable="leaf_respiration_flux",
                    references=["Tcherkez et al. (2003)"]
                ),
                "root_respiration": CarbonProcessConfig(
                    process_name="root_respiration",
                    enabled=False,
                    process_type=CarbonProcessType.ROOT_RESPIRATION,
                    fractionation=FractionationParameters(
                        model_type=FractionationModelType.Q10,
                        alpha_0=root_alpha,
                        Q10=root_q10,
                        T_ref=temp_ref,
                        T_min=temp_min,
                        T_max=temp_max,
                    ),
                    source_pool="root_biomass",
                    target_pool="atmospheric_co2",
                    flux_variable="root_respiration_flux",
                    references=["Bowling et al. (2008)"]
                ),
                "decomposition": CarbonProcessConfig(
                    process_name="decomposition", 
                    enabled=False,
                    process_type=CarbonProcessType.DECOMPOSITION,
                    fractionation=FractionationParameters(
                        model_type=FractionationModelType.Q10,
                        alpha_0=decomp_alpha,
                        Q10=decomp_q10,
                        T_ref=temp_ref,
                        T_min=temp_min,
                        T_max=temp_max,
                    ),
                    source_pool="soil_organic_carbon",
                    target_pool="atmospheric_co2",
                    flux_variable="decomposition_flux",
                    references=["Aubert et al. (2005)"]
                )
            }
        )

    def create_default_nitrogen_config(self) -> IsotopeSpeciesConfig:
        """Create the runtime-disabled 15N candidate configuration."""
        nitrogen_config = Nitrogen15Config()
        standards = get_isotope_standards()
        pool_names = {
            "orgN": "organic_nitrogen",
            "NH4": "nh4_nitrogen",
            "NO3": "no3_nitrogen",
        }
        pools = {
            canonical_name: type(nitrogen_config.pools[old_name]).model_validate(
                {
                    **nitrogen_config.pools[old_name].model_dump(
                        mode="python",
                        round_trip=True,
                    ),
                    "pool_name": canonical_name,
                    "state_variable": canonical_name,
                    "units": "g N m-2",
                }
            )
            for old_name, canonical_name in pool_names.items()
        }
        processes = {
            process_name: type(process).model_validate(
                {
                    **process.model_dump(mode="python", round_trip=True),
                    "enabled": False,
                    "source_pool": pool_names.get(
                        process.source_pool,
                        process.source_pool,
                    ),
                    "target_pool": pool_names.get(
                        process.target_pool,
                        process.target_pool,
                    ),
                }
            )
            for process_name, process in nitrogen_config.processes.items()
        }
        return IsotopeSpeciesConfig(
            species=IsotopeSpecies.NITROGEN_15,
            standard=IsotopeStandard.AIR_N2,
            standard_heavy_to_light_ratio=standards[IsotopeStandard.AIR_N2],
            component_amount_basis="element_mass",
            enabled=False,
            runtime_blocker=(
                "nuclide_mass_conversion_unbound+"
                "initial_ratio_semantics_unbound+"
                "denitrification_open_route_unbound+"
                "state_manager_component_ownership_unbound+"
                "executable_output_mapping_unbound"
            ),
            pools=pools,
            nitrogen_processes=processes,
            output_variables={},
        )
    
    def create_default_oxygen_config(self) -> IsotopeSpeciesConfig:
        """Create default 18O configuration."""
        standards = get_isotope_standards()
        temp_ref = _require_param("isotopes.temperature.reference", units="K", expected_type=float)
        temp_min = _require_param("isotopes.temperature.min", units="K", expected_type=float)
        temp_max = _require_param("isotopes.temperature.max", units="K", expected_type=float)
        transpiration_alpha = _require_param(
            "isotopes.fractionation.oxygen.transpiration.alpha_0",
            units="dimensionless",
            expected_type=float,
        )
        soil_evap_alpha = _require_param(
            "isotopes.fractionation.oxygen.soil_evaporation.alpha_0",
            units="dimensionless",
            expected_type=float,
        )
        return IsotopeSpeciesConfig(
            species=IsotopeSpecies.OXYGEN_18,
            standard=IsotopeStandard.VSMOW,
            standard_heavy_to_light_ratio=standards[IsotopeStandard.VSMOW],
            component_amount_basis="amount_of_substance",
            enabled=False,
            runtime_blocker=(
                "missing_17O_closure+canonical_water_parent_unbound+"
                "open_boundary_ratio_unbound+"
                "initial_ratio_semantics_unbound+"
                "compound_mass_to_oxygen_amount_conversion_unbound+"
                "state_manager_component_ownership_unbound+"
                "executable_output_mapping_unbound"
            ),
            pools={
                "soil_water": PoolConfig(
                    pool_name="soil_water",
                    state_variable="soil_water",
                    initial_abundance=standards[IsotopeStandard.VSMOW],
                    units="mol m-2",
                    description="Soil water oxygen pool"
                ),
                "leaf_water": PoolConfig(
                    pool_name="leaf_water", 
                    state_variable="leaf_water",
                    initial_abundance=standards[IsotopeStandard.VSMOW],
                    units="mol m-2",
                    description="Leaf water oxygen pool"
                ),
                "atmospheric_water": PoolConfig(
                    pool_name="atmospheric_water",
                    state_variable="atmospheric_water",
                    initial_abundance=standards[IsotopeStandard.VSMOW],
                    units="mol m-2",
                    description="Atmospheric water vapor oxygen pool"
                )
            },
            oxygen_processes={
                "transpiration": OxygenProcessConfig(
                    process_name="transpiration",
                    enabled=False,
                    process_type=OxygenProcessType.TRANSPIRATION,
                    fractionation=FractionationParameters(
                        model_type=FractionationModelType.CONSTANT,
                        alpha_0=transpiration_alpha,
                        T_ref=temp_ref,
                        T_min=temp_min,
                        T_max=temp_max,
                    ),
                    oxygen_fractionation=OxygenFractionationParameters(),
                    source_pool="leaf_water",
                    target_pool="atmospheric_water",
                    flux_variable="transpiration_flux",
                    references=["Craig & Gordon (1965)", "Flanagan et al. (1991)"]
                ),
                "soil_evaporation": OxygenProcessConfig(
                    process_name="soil_evaporation",
                    enabled=False,
                    process_type=OxygenProcessType.SOIL_EVAPORATION, 
                    fractionation=FractionationParameters(
                        model_type=FractionationModelType.CONSTANT,
                        alpha_0=soil_evap_alpha,
                        T_ref=temp_ref,
                        T_min=temp_min,
                        T_max=temp_max,
                    ),
                    oxygen_fractionation=OxygenFractionationParameters(),
                    source_pool="soil_water",
                    target_pool="atmospheric_water",
                    flux_variable="soil_evaporation_flux",
                    references=["Barnes & Allison (1988)"]
                )
            }
        )


def get_default_isotope_config() -> IsotopeConfig:
    """Build inspectable candidate specs with every runtime path disabled."""
    seed = IsotopeConfig(tracked_species=set(), species_configs={})
    species_configs = {
        IsotopeSpecies.CARBON_13: seed.create_default_carbon_config(),
        IsotopeSpecies.NITROGEN_15: seed.create_default_nitrogen_config(),
        IsotopeSpecies.OXYGEN_18: seed.create_default_oxygen_config(),
    }
    unified_output = seed.unified_output.model_copy(deep=True)
    unified_output.variables = {}
    return IsotopeConfig(
        tracked_species=set(),
        species_configs=species_configs,
        unified_output=unified_output,
    )


# Export multi-isotope configuration models
__all__ = [
    'IsotopeSpecies',
    'CarbonProcessType', 
    'OxygenProcessType',
    'PhotosynthesisType',
    'IsotopeStandard',
    'get_isotope_standards',
    'CarbonFractionationParameters',
    'OxygenFractionationParameters', 
    'CarbonProcessConfig',
    'OxygenProcessConfig',
    'IsotopeSpeciesConfig',
    'CrossIsotopeConstraints',
    'IsotopeConfig',
    'get_default_isotope_config',
]


