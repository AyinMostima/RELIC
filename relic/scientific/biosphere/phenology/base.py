'\nPhenology Base Class for RELIC Framework\n\nThis module provides the unified abstract base class for all phenological \nprocesses in the RELIC ecosystem. It integrates with the core framework\narchitecture, inheriting from both BiologicalProcess and FunctorBase to \nprovide a complete interface for seasonal vegetation dynamics.\n\nThe design follows category theory principles where phenology functors\nrepresent morphisms that transform vegetation states through time while\npreserving ecological and physical constraints. This enables flexible,\nmodular phenological modeling with automatic differentiation support.\n\nArchitecture:\n    - Inherits from BiologicalProcess for biological process interface\n    - Inherits from FunctorBase for unified functor framework\n    - Provides abstract methods for different phenology models\n    - Implements compute_rate method for biological process integration  \n    - Handles biomass turnover and carbon/nitrogen cycling\n    - Enforces conservation laws and physical constraints\n\nMathematical Foundation:\n    Phenology operator: P: VegetationState * Environment * Time -> PhenologyState\n    State transition: f_pheno(t+dt) = P(vegetation_state, environment, t, dt, theta)\n    \n    Key equations:\n    - GSI (Jolly et al. 2005): GSI = f_temp * f_vpd * f_daylen * f_swp\n    - GDD: Sigma max(0, (T_max + T_min)/2 - T_base)\n    - LAI dynamics: dLAI/dt = f_allocation(pheno) - f_senescence(pheno) \n    - Biomass turnover: dC_turnover/dt = C_pool * turnover_rate * f_pheno(t)\n\nReferences:\n    - Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict \n      foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.\n    - White, M.A., et al. (1997). A continental phenology model for monitoring \n      vegetation responses to interannual climatic variability. \n      Global Biogeochemical Cycles, 11(2), 217-234.\n    - Richardson, A.D., et al. (2013). Climate change, phenology, and phenological \n      control of vegetation feedbacks to the climate system. \n      Agricultural and Forest Meteorology, 169, 156-173.\n    - Keenan, T.F. & Richardson, A.D. (2015). The timing of autumn senescence \n      is affected by the timing of spring phenology. Ecology, 96(5), 1212-1221.\n'
import logging
import math
from abc import abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import torch
from torch import Tensor
from relic.core.functor import BiologicalProcessFunctor, StateDict, FluxDict
from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    ConfigDict,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.computation.operators.iphs import advance_reaction_network
from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import ConversionContext, unit_converter
from .parameters import (
    PHENOLOGY_BASE_PARAMETER_DESCRIPTIONS,
    PHENOLOGY_FIELD_PARAMETER_DESCRIPTIONS,
    PHENOLOGY_PARAMETER_FIELD_KEYS,
    PHENOLOGY_PARAMETER_FIELD_UNITS,
    PhenologyParameters,
    phenology_field_names,
)
from .utils import validate_environmental_data
logger = logging.getLogger(__name__)
PhenologyState = Dict[str, Tensor]
BiomassFluxes = Dict[str, Tensor]

class PhenologyPhase(Enum):
    """
    Phenological phases for vegetation development.
    
    Standardized phenological stages following international conventions
    for agricultural and natural vegetation monitoring systems.
    """
    DORMANCY = 'dormancy'
    CHILLING = 'chilling'
    VERNALIZATION = 'vernalization'
    BUD_BREAK = 'bud_break'
    LEAF_EXPANSION = 'leaf_expansion'
    FULL_LEAF = 'full_leaf'
    FLOWERING = 'flowering'
    SENESCENCE = 'senescence'
    ABSCISSION = 'abscission'

@dataclass
class PhenologyOutput:
    """
    Structured output from phenological calculations.
    
    Provides a standardized container for all phenology-related outputs
    including phenological state, biomass changes, and diagnostic variables.
    
    Attributes:
        phenology_phase : PhenologyPhase
            Current phenological phase
        growing_season_index : Optional[Tensor]
            GSI value (0-1) if applicable
        thermal_time : Optional[Tensor] 
            Accumulated thermal time (GDD or PHU)
        leaf_area_index : Tensor
            Current leaf area index [m^2 m-2]
        growth_factor : Tensor
            Growth activity factor (0-1)
        senescence_rate : Tensor
            Senescence rate factor (0-1)
        turnover_rates : Dict[str, Tensor]
            Biomass turnover rates by pool
        allocation_fractions : Dict[str, Tensor]
            Carbon allocation fractions by pool
        diagnostic_variables : Dict[str, Tensor]
            Additional diagnostic outputs
    """
    phenology_phase: PhenologyPhase
    growing_season_index: Optional[Tensor] = None
    thermal_time: Optional[Tensor] = None
    leaf_area_index: Tensor = None
    growth_factor: Tensor = None
    senescence_rate: Tensor = None
    turnover_rates: Dict[str, Tensor] = None
    allocation_fractions: Dict[str, Tensor] = None
    diagnostic_variables: Dict[str, Tensor] = None
    additional_outputs: Dict[str, Tensor] = None

    def __post_init__(self):
        if self.turnover_rates is None:
            self.turnover_rates = {}
        if self.allocation_fractions is None:
            self.allocation_fractions = {}
        if self.diagnostic_variables is None:
            self.diagnostic_variables = {}
        if self.additional_outputs is None:
            self.additional_outputs = {}

class PhenologyBase(BiologicalProcessFunctor):
    """
    Abstract Base Class for Phenological Processes
    
    Provides the unified interface for all phenological models in the RELIC
    framework. Integrates seasonal vegetation dynamics with carbon and nitrogen
    cycling through standardized methods for phenological state calculation,
    biomass turnover, and ecosystem coupling.
    
    This class serves as the foundation for specific phenology models including:
    - Static date-based phenology
    - Growing degree day (GDD) models  
    - Growing season index (GSI) models
    - Crop-specific phenology models
    
    The implementation ensures consistency with the four-pillar architecture
    and provides hooks for integration with photosynthesis, respiration,
    growth, and biogeochemical cycling processes.

    Dependencies: phenology declares no static Functor dependency (the bridge
    default returns only instance-declared names). Its inputs are declared
    state ports; downstream allocation, stress, litter, and coupling processes
    must not be declared as reverse dependencies here.

    Key Features:
        - Unified interface for multiple phenology approaches
        - Integration with RELIC core architecture
        - Conservation law enforcement for mass balance
        - Automatic differentiation support through PyTorch
        - Comprehensive parameter validation and management
        - Scientific metadata and literature referencing
        
    Mathematical Expression:
        Phenological state transition:
        S_pheno(t+dt) = P(S_pheno(t), E(t), theta, dt)
        
        Where:
        - S_pheno: phenological state vector
        - E: environmental driving variables
        - theta: parameter vector
        - P: phenological process operator
        
        Biomass coupling:
        dC/dt = f_growth(S_pheno) - f_turnover(S_pheno) - f_senescence(S_pheno)
        
    References:
        - Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict 
          foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.
        - White, M.A., et al. (1997). A continental phenology model for monitoring 
          vegetation responses to interannual climatic variability.
        - Keenan, T.F. & Richardson, A.D. (2015). The timing of autumn senescence.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    PHENOLOGY_TYPE: Optional[str] = None

    COMMON_INPUT_VARIABLES = (
        "day_of_year",
        "air_temperature",
        "leaf_carbon",
        "froot_carbon",
        "leaf_nitrogen",
        "froot_nitrogen",
    )
    POOL_INPUT_VARIABLES = (
        "leaf_carbon",
        "froot_carbon",
        "leaf_nitrogen",
        "froot_nitrogen",
    )
    COMMON_OUTPUT_VARIABLES = (
        "leaf_area_index",
        "phenological_factor",
        "phenology_senescence_factor",
        "phenology_phase",
        "leaf_carbon",
        "froot_carbon",
        "leaf_nitrogen",
        "froot_nitrogen",
        "plant_nitrogen_pool",
        "leaf_litterfall_carbon",
        "froot_litterfall_carbon",
        "leaf_litterfall_nitrogen",
        "froot_litterfall_nitrogen",
        "nitrogen_retranslocation",
        "diagnostics.phenology.carbon_turnover_balance_residual",
        "diagnostics.phenology.nitrogen_turnover_balance_residual",
    )
    STATE_INCREMENT_OUTPUT_VARIABLES = (
        "leaf_carbon",
        "froot_carbon",
        "leaf_nitrogen",
        "froot_nitrogen",
        "plant_nitrogen_pool",
    )
    BOUNDARY_RATE_OUTPUT_VARIABLES = (
        "leaf_litterfall_carbon",
        "froot_litterfall_carbon",
        "leaf_litterfall_nitrogen",
        "froot_litterfall_nitrogen",
        "nitrogen_retranslocation",
    )
    OUTPUT_UNITS = {
        "leaf_area_index": "m2 m-2",
        "phenological_factor": "dimensionless",
        "phenology_senescence_factor": "dimensionless",
        "phenology_phase": "dimensionless",
        "leaf_carbon": "kg C m-2",
        "froot_carbon": "kg C m-2",
        "leaf_nitrogen": "kg N m-2",
        "froot_nitrogen": "kg N m-2",
        "plant_nitrogen_pool": "kg N m-2",
        "leaf_litterfall_carbon": "kg C m-2 s-1",
        "froot_litterfall_carbon": "kg C m-2 s-1",
        "leaf_litterfall_nitrogen": "kg N m-2 s-1",
        "froot_litterfall_nitrogen": "kg N m-2 s-1",
        "nitrogen_retranslocation": "kg N m-2 s-1",
        "diagnostics.phenology.carbon_turnover_balance_residual": "kg C m-2",
        "diagnostics.phenology.nitrogen_turnover_balance_residual": "kg N m-2",
    }
    OUTPUT_ELEMENTS = {
        "leaf_carbon": "C",
        "froot_carbon": "C",
        "leaf_litterfall_carbon": "C",
        "froot_litterfall_carbon": "C",
        "leaf_nitrogen": "N",
        "froot_nitrogen": "N",
        "plant_nitrogen_pool": "N",
        "leaf_litterfall_nitrogen": "N",
        "froot_litterfall_nitrogen": "N",
    }
    PARAMETER_FIELD_KEYS = PHENOLOGY_PARAMETER_FIELD_KEYS
    PARAMETER_FIELD_UNITS = PHENOLOGY_PARAMETER_FIELD_UNITS

    @staticmethod
    def _tensor_scalar(value: Tensor) -> float:
        """Convert scalar tensor to Python float without using scalar extraction sync calls."""
        detached = value.detach().reshape(-1)
        values = detached.tolist()
        if not values:
            raise ValueError("Expected non-empty scalar tensor.")
        return float(values[0])

    @staticmethod
    def _to_scalar(value: Union[Tensor, float, int]) -> float:
        """
        Convert tensors or numeric inputs to plain Python floats for parameter construction.
        """
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(f'Phenology parameter expected scalar tensor, received shape {tuple(value.shape)}')
            return PhenologyBase._tensor_scalar(value)
        return float(value)

    def __init__(self, name: str, config: Optional[ConfigDict]=None, phenology_parameters: Optional[PhenologyParameters]=None, metadata: Optional[FunctorMetadata]=None, device: Optional[torch.device]=None, **kwargs) -> None:
        """
        Initialize phenology functor with parameters and configuration.
        
        Parameters:
            name : str
                Unique identifier for this phenology model
            config : ConfigDict
                Configuration dictionary with model parameters
            phenology_parameters : Optional[PhenologyParameters]
                Typed phenology parameter container
            metadata : Optional[FunctorMetadata] 
                Scientific metadata for documentation
            device : Optional[torch.device]
                Computation device (CPU/GPU)
            **kwargs
                Additional configuration parameters
                
        Raises:
            ValueError: If required parameters are missing
            ConfigurationError: If configuration is invalid
        """
        if config is None:
            config = {}
        if device is None:
            config_device = config.get('device') if isinstance(config, dict) else None
            if config_device is not None:
                if isinstance(config_device, torch.device):
                    device = config_device
                elif isinstance(config_device, str):
                    device = torch.device(config_device)
        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                version='1.0.0',
                description=f'Phenological process model: {name}',
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                execution_phase=ExecutionPhase.MAIN_PROCESSES,
                mathematical_expression=self._get_mathematical_expression(),
                references=self._get_references(),
                parameters={},
                input_variables=self._get_input_variables(),
                output_variables=self._get_output_variables(),
                conservation_laws=['mass', 'carbon', 'nitrogen'],
                spatial_dimensions=0,
                temporal_scale=TemporalScale.DAY,
                coupling_interfaces=['photosynthesis', 'respiration', 'growth', 'soil'],
                spatial_scale=SpatialScale.GRID_CELL,
                variable_kind=VariableKind.STATE,
            )
        normalized_inputs = list(dict.fromkeys(
            [*metadata.input_variables, *self.COMMON_INPUT_VARIABLES]
        ))
        metadata.input_variables = normalized_inputs
        metadata.output_variables = list(dict.fromkeys(
            [*metadata.output_variables, *self.COMMON_OUTPUT_VARIABLES]
        ))
        metadata.conservation_laws = ["carbon", "nitrogen"]
        # Declared full keys: the base always reads PHENOLOGY_BASE_PARAMETER_DESCRIPTIONS and,
        # when no typed container is given, the container field keys of the configured model
        # only (phenology_field_names). A concrete class fixes its model (PHENOLOGY_TYPE);
        # the generic base declares the phenology_type key here and the fields of the model
        # it reads there once the key is read below. The model entries follow.
        declared = dict(PHENOLOGY_BASE_PARAMETER_DESCRIPTIONS)
        if phenology_parameters is None:
            if self.PHENOLOGY_TYPE is None:
                type_key = self.PARAMETER_FIELD_KEYS["phenology_type"]
                declared[type_key] = PHENOLOGY_FIELD_PARAMETER_DESCRIPTIONS[type_key]
            else:
                declared.update(self._field_parameter_descriptions(self.PHENOLOGY_TYPE))
        metadata.parameters = {**declared, **dict(metadata.parameters)}
        BiologicalProcessFunctor.__init__(self, name=name, config=config, metadata=metadata, device=device, **kwargs)
        to_scalar = self._to_scalar
        if phenology_parameters is None:
            phenology_type = self.PHENOLOGY_TYPE
            if phenology_type is None:
                phenology_type = self.get_parameter_value(
                    self.PARAMETER_FIELD_KEYS["phenology_type"],
                    required=True,
                )
                self.metadata.parameters.update(
                    self._field_parameter_descriptions(phenology_type)
                )
            parameter_values = {"phenology_type": phenology_type}
            for field_name in phenology_field_names(phenology_type):
                if field_name == "phenology_type":
                    continue
                parameter_key = self.PARAMETER_FIELD_KEYS[field_name]
                parameter_value = self.get_parameter_value(
                    parameter_key,
                    required=True,
                )
                if field_name in self.PARAMETER_FIELD_UNITS:
                    # References: SI Brochure, 9th edition, temperature scales.
                    # Mathematical expression: T_C = T_K - Delta_CK.
                    # Parameter meanings: Delta_CK is read by UnitConverter from
                    # the physical-constant SSOT.
                    # Units: T_K [K], Delta_CK [K], T_C [degC].
                    # Open route: affine conversion only; no energy is exchanged.
                    parameter_value = unit_converter.convert(
                        parameter_value,
                        "generic",
                        "K",
                        self.PARAMETER_FIELD_UNITS[field_name],
                        ConversionContext(),
                    )
                if isinstance(parameter_value, Tensor):
                    parameter_value = to_scalar(parameter_value)
                parameter_values[field_name] = parameter_value
            phenology_parameters = PhenologyParameters(**parameter_values)
        if (
            self.PHENOLOGY_TYPE is not None
            and phenology_parameters.phenology_type != self.PHENOLOGY_TYPE
        ):
            raise ValueError(
                f"{self.__class__.__name__} requires phenology_type "
                f"'{self.PHENOLOGY_TYPE}', received "
                f"'{phenology_parameters.phenology_type}'"
            )
        self.phenology_params = phenology_parameters
        self.conservation_tolerance = to_scalar(
            self.phenology_params.numerical_tolerance
        )
        self.enforce_physical_constraints = True
        self.nitrogen_retrans_efficiency = to_scalar(self.get_parameter_value('turnover.n_retrans_efficiency', namespace='biosphere.phenology'))
        logger.info(f"Initialized {self.__class__.__name__} '{name}' on device {self.device}")

    @classmethod
    def _field_parameter_descriptions(cls, phenology_type: str) -> Dict[str, str]:
        """Declared meaning of the container field keys the given model reads."""
        return {
            cls.PARAMETER_FIELD_KEYS[field_name]: PHENOLOGY_FIELD_PARAMETER_DESCRIPTIONS[
                cls.PARAMETER_FIELD_KEYS[field_name]
            ]
            for field_name in phenology_field_names(phenology_type)
            if field_name != "phenology_type"
        }

    def get_parameter_value(self, name: str, **kwargs):
        """
        Enforce ParameterManager-first lookups with validation and no config defaults.
        """
        if 'source_order' not in kwargs:
            kwargs['source_order'] = ('manager',)
        validate = kwargs.pop('validate', True)
        required = kwargs.pop('required', kwargs.get('default') is None)
        return super().get_parameter_value(name, validate=validate, required=required, **kwargs)

    def _get_mathematical_expression(self) -> str:
        """Get LaTeX mathematical expression for this phenology model."""
        return '\n        Phenological state dynamics:\n        \\frac{dS_{pheno}}{dt} = P(E(t), S_{veg}(t), \\theta)\n        \n        Where:\n        - S_{pheno}: phenological state vector\n        - E(t): environmental drivers (T, VPD, photoperiod, moisture)\n        - S_{veg}: vegetation state (LAI, biomass pools)\n        - \\theta: parameter vector\n        \n        GSI formulation (Jolly et al. 2005):\n        GSI = f_{temp}(T_{min}) \\times f_{vpd}(VPD) \\times f_{day}(DL) \\times f_{water}(SWP)\n        \n        GDD accumulation:\n        GDD = \\sum_{i=1}^{n} \\max(0, \\frac{T_{max,i} + T_{min,i}}{2} - T_{base})\n        \n        LAI dynamics:\n        \\frac{dLAI}{dt} = f_{growth}(S_{pheno}) - f_{senescence}(S_{pheno})\n        '

    def _get_references(self) -> List[str]:
        """Get literature references for phenology models."""
        return ['Jolly, W.M., et al. (2005). A generalized, bioclimatic index to predict foliar phenology in response to climate. Global Change Biology, 11(4), 619-632.', 'White, M.A., et al. (1997). A continental phenology model for monitoring vegetation responses to interannual climatic variability. Global Biogeochemical Cycles, 11(2), 217-234.', 'Richardson, A.D., et al. (2013). Climate change, phenology, and phenological control of vegetation feedbacks to the climate system. Agricultural and Forest Meteorology, 169, 156-173.', 'Keenan, T.F. & Richardson, A.D. (2015). The timing of autumn senescence is affected by the timing of spring phenology. Ecology, 96(5), 1212-1221.', 'Piao, S., et al. (2019). Plant phenology and global climate change: Current progresses and challenges. Global Change Biology, 25(6), 1922-1940.']

    def _get_input_variables(self) -> List[str]:
        """Get required input state variables."""
        return list(self.COMMON_INPUT_VARIABLES)

    def _get_output_variables(self) -> List[str]:
        """Get output state variables modified by phenology."""
        return list(self.COMMON_OUTPUT_VARIABLES)

    @abstractmethod
    def compute_phenological_state(self, state: StateDict, *, dt: float) -> PhenologyOutput:
        """
        Compute phenological state from the declared scientific inputs.

        This is the core abstract method that must be implemented by
        all concrete phenology models. It calculates the current
        phenological state and associated variables.

        Parameters:
            state : StateDict
                The metadata-declared scientific inputs of the model: day_of_year
                [day], air_temperature [K] and the model-specific drivers and
                StateManager-owned temporal states (for example
                air_temperature_min/max [K], vapor_pressure_deficit [Pa],
                daylength [s]). No other mapping carries scientific inputs.
            dt : float
                Step length of the simulator call [s]; the one time control.

        Returns:
            PhenologyOutput: Complete phenological state information

        Raises:
            RuntimeError: Raised when subclasses do not override this method
            ValueError: If required inputs are missing or invalid

        Mathematical Expression:
            Model-specific implementation of:
            S_pheno = f(DOY, T, E, theta)

            Where f depends on the phenology model type:
            - Static: f(DOY) - calendar date based
            - GDD: f(GDD_accumulated, theta_GDD)
            - GSI: f(GSI(T_min, VPD, DL, SWP), theta_GSI)
            - Crop: f(PHU, development_stage, theta_crop)
        """
        raise RuntimeError('compute_phenological_state must be implemented by phenology subclasses')

    def _assert_non_negative_value(self, value: Any, error_prefix: str) -> None:
        """Assert a scalar or tensor value is non-negative."""
        if isinstance(value, torch.Tensor):
            if torch.any(value < 0):
                min_value = self._to_scalar_float(value.detach().min())
                raise ValueError(f"{error_prefix}: min={min_value}")
            return
        if value < 0:
            raise ValueError(f"{error_prefix}: value={value}")

    def _assert_value_in_range(
        self,
        value: Any,
        *,
        lower: float,
        upper: float,
        error_prefix: str,
    ) -> None:
        """Assert a scalar or tensor value lies in [lower, upper]."""
        if isinstance(value, torch.Tensor):
            if torch.any(value < lower) or torch.any(value > upper):
                min_value = self._to_scalar_float(value.detach().min())
                max_value = self._to_scalar_float(value.detach().max())
                raise ValueError(
                    f"{error_prefix}: min={min_value}, max={max_value}"
                )
            return
        if value < lower or value > upper:
            raise ValueError(f"{error_prefix}: value={value}")

    def _to_scalar_float(self, value: Any) -> float:
        """Convert scalar or tensor statistic to float."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return self._tensor_scalar(value)
            return self._tensor_scalar(value.detach().mean())
        return float(value)

    def _validate_allocation_fractions(self, allocation_fractions: Dict[str, Any]) -> None:
        """Validate allocation fraction sum and per-organ non-negativity."""
        total = sum(allocation_fractions.values())
        total_scalar = self._to_scalar_float(total)
        if abs(total_scalar - 1.0) > self.conservation_tolerance:
            raise ValueError(f'Allocation fractions do not sum to 1: sum={total_scalar}')
        for organ, fraction in allocation_fractions.items():
            if isinstance(fraction, torch.Tensor):
                if torch.any(fraction < 0):
                    min_fraction = self._to_scalar_float(fraction.detach().min())
                    raise ValueError(
                        f'Negative allocation fraction for {organ}: min={min_fraction}'
                    )
            elif fraction < 0:
                raise ValueError(f'Negative allocation fraction for {organ}: {fraction}')

    def _validate_output_bounds(self, output: PhenologyOutput) -> None:
        """
        Validate phenological output bounds and constraints.

        This method validates physical plausibility constraints, not elemental
        conservation laws.

        Parameters:
            output : PhenologyOutput
                The phenological output to validate

        Raises:
            ValueError: If bounds or constraints are violated
        """
        if output.leaf_area_index is not None:
            lai = output.leaf_area_index
            self._assert_non_negative_value(lai, "Negative LAI detected")
            max_lai_threshold_param = self.get_parameter_value(
                'max_lai_threshold',
                namespace='biosphere.phenology.constants',
            )
            max_lai_threshold = torch.as_tensor(
                max_lai_threshold_param,
                device=lai.device,
                dtype=lai.dtype,
            )
            if torch.any(lai > max_lai_threshold):
                max_lai = self._to_scalar_float(lai.detach().max())
                raise ValueError(f'LAI exceeds maximum threshold: max={max_lai}')

        if output.growth_factor is not None:
            self._assert_value_in_range(
                output.growth_factor,
                lower=0.0,
                upper=1.0 + self.conservation_tolerance,
                error_prefix='Growth factor out of bounds [0,1]',
            )

        if output.senescence_rate is not None:
            self._assert_value_in_range(
                output.senescence_rate,
                lower=0.0,
                upper=1.0 + self.conservation_tolerance,
                error_prefix='Senescence rate out of bounds [0,1]',
            )

        if output.allocation_fractions:
            self._validate_allocation_fractions(output.allocation_fractions)

        if output.thermal_time is not None:
            self._assert_non_negative_value(output.thermal_time, 'Negative thermal time')

    def _compute_rate_core(self, state: StateDict, dt: float) -> FluxDict:
        """
        Compute biological process rates - implements BiologicalProcess interface.

        This method bridges the BiologicalProcess abstraction with the phenology-
        specific computations. The phenology law receives only the
        metadata-declared scientific inputs (the vegetation pools enter the
        turnover step); the step length is the simulator dt.

        Parameters:
            state : StateDict
                Current ecosystem state holding the declared inputs
            dt : float
                Step length [s]

        Returns:
            FluxDict: Process rates and state updates

        Raises:
            KeyError: If a declared input is missing from state
            ValueError: If required inputs fail validation
            RuntimeError: If phenological calculation fails

        Mathematical Expression:
            Process rates: R = P_rate(S_pheno, S_eco, E, theta)

            Biomass turnover:
            dC_leaf/dt = -C_leaf * r_turnover * f_pheno(t)
            dC_root/dt = -C_root * r_turnover * f_pheno(t)

            Nitrogen retranslocation:
            dN_retrans/dt = N_leaf * r_retrans * f_senescence(t)
        """
        pool_inputs = set(self.POOL_INPUT_VARIABLES)
        inputs = {
            key: state[key]
            for key in self.metadata.input_variables
            if key not in pool_inputs
        }
        validation_result = validate_environmental_data(
            inputs,
            required_vars=["day_of_year", "air_temperature"],
            check_ranges=True,
        )
        if not validation_result["valid"]:
            missing_vars = validation_result.get("missing_vars", [])
            range_warnings = validation_result.get("range_warnings", [])
            raise ValueError(
                "Environmental data validation failed: "
                f"{missing_vars + range_warnings}"
            )

        pheno_output = self.compute_phenological_state(inputs, dt=dt)
        self._validate_output_bounds(pheno_output)
        outputs: FluxDict = self._compute_biomass_turnover(
            state=state,
            pheno_output=pheno_output,
            dt=dt,
        )
        outputs["leaf_area_index"] = pheno_output.leaf_area_index
        outputs["phenological_factor"] = pheno_output.growth_factor
        outputs["phenology_senescence_factor"] = pheno_output.senescence_rate
        phase_mapping = {
            PhenologyPhase.DORMANCY: 0,
            PhenologyPhase.CHILLING: 0,
            PhenologyPhase.VERNALIZATION: 0,
            PhenologyPhase.BUD_BREAK: 1,
            PhenologyPhase.LEAF_EXPANSION: 1,
            PhenologyPhase.FULL_LEAF: 2,
            PhenologyPhase.FLOWERING: 2,
            PhenologyPhase.SENESCENCE: 3,
            PhenologyPhase.ABSCISSION: 3,
        }
        phase_value = phase_mapping[pheno_output.phenology_phase]
        outputs["phenology_phase"] = torch.as_tensor(
            float(phase_value),
            device=pheno_output.leaf_area_index.device,
            dtype=pheno_output.leaf_area_index.dtype,
        )
        declared_outputs = tuple(self.metadata.output_variables)
        for key, value in pheno_output.additional_outputs.items():
            if key in declared_outputs:
                outputs[key] = value
        for key, value in pheno_output.diagnostic_variables.items():
            if key in declared_outputs:
                outputs[key] = value

        missing_outputs = sorted(set(declared_outputs) - set(outputs))
        if missing_outputs:
            raise ValueError(
                f"{self.name} did not produce declared outputs: {missing_outputs}"
            )
        # StateManager owns scientific state and restart persistence. The
        # abstract process must not cache forward results on the Functor.
        return {key: outputs[key] for key in declared_outputs}

    def forward(self, state: StateDict, time: float, dt: float) -> FluxDict:
        """
        Forward pass for functor execution - implements FunctorBase interface.
        
        Parameters:
            state : StateDict
                Current system state
            time : float
                Current simulation time [s]
            dt : float
                Time step [s]
        Returns:
            FluxDict: State updates and process rates
        """
        # References: Jolly et al. (2005); White et al. (1997); Richardson et al. (2013);
        #   Keenan and Richardson (2015).
        # Mathematical expression: dS_pheno/dt = P(E(t), S_veg(t), theta);
        #   GSI = f_temp(Tmin) f_vpd(VPD) f_day(DL) f_water(SWP);
        #   GDD = sum_i max(0, (Tmax_i + Tmin_i)/2 - T_base);
        #   dLAI/dt = f_growth(S_pheno) - f_senescence(S_pheno).
        # Parameter meanings: S_pheno phenological state, E environmental drivers, S_veg the
        #   vegetation state, theta the ParameterManager phenology parameters; the concrete
        #   subclass law is evaluated by _compute_rate_core.
        # Units: outputs carry OUTPUT_UNITS (indices and factors [1], LAI [m2 m-2], carbon and
        #   nitrogen stocks [kg m-2], litterfall rates [kg m-2 s-1]); T [K]; VPD [Pa].
        # Domain/constraints: dt is a positive finite scalar; stock outputs are per-step
        #   ACCUMULATE increments, boundary rates are REPLACE rates, other outputs REPLACE
        #   observations; the C and N ledgers are audited by check_conservation_laws.
        # Sign convention: growth increases LAI and stocks; senescence and litterfall decrease
        #   them, with litterfall positive out of the plant.
        if not math.isfinite(float(time)):
            raise ValueError("time must be finite")
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        raw_outputs = self._compute_rate_core(state, float(dt))

        state_increment_keys = set(self.STATE_INCREMENT_OUTPUT_VARIABLES)
        rate_keys = set(self.BOUNDARY_RATE_OUTPUT_VARIABLES)
        updates: FluxDict = {}
        for key, value in raw_outputs.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"Output '{key}' must be a torch.Tensor")
            if key not in self.OUTPUT_UNITS:
                raise ValueError(f"No canonical unit declared for output '{key}'")
            if key in state_increment_keys:
                mode = UpdateMode.ACCUMULATE
                per_timestep = True
            else:
                mode = UpdateMode.REPLACE
                per_timestep = key not in rate_keys
            updates[key] = UpdateItem(
                value=value,
                mode=mode,
                unit=self.OUTPUT_UNITS[key],
                per_timestep=per_timestep,
                element=self.OUTPUT_ELEMENTS.get(key),
                source_functor=self.name,
            )
        return updates

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Return the explicit phenology carbon and nitrogen ledger residuals.

        References:
            StateManager ledger-based conservation accounting and the turnover
            equations documented in ``docs/source/theory/phenology.md``.
        Mathematical expression:
            r_C = delta C_leaf + delta C_root + dt * (F_leaf,C + F_root,C)
            r_N = delta N_leaf + delta N_root + delta N_plant
                  + dt * (F_leaf,N + F_root,N)
        Parameter meanings:
            The residual diagnostics have stock units (kg C m-2 and
            kg N m-2); zero denotes closure across retained pools and explicit
            litter boundary exports.
        """
        del input_state
        diagnostic_keys = {
            "carbon": "diagnostics.phenology.carbon_turnover_balance_residual",
            "nitrogen": "diagnostics.phenology.nitrogen_turnover_balance_residual",
        }
        violations: Dict[str, float] = {}
        for element, key in diagnostic_keys.items():
            if key not in output_updates:
                raise ValueError(f"Missing required conservation diagnostic: {key}")
            update = output_updates[key]
            if not isinstance(update, UpdateItem):
                raise TypeError(f"Conservation diagnostic '{key}' must be an UpdateItem")
            value = update.value
            if not isinstance(value, Tensor):
                raise TypeError(f"Conservation diagnostic '{key}' must contain a tensor")
            violations[element] = self._to_scalar_float(value.detach().abs().max())
        return violations

    @staticmethod
    def _require_pool_tensor(state: StateDict, key: str) -> Tensor:
        """Return one required finite, non-negative vegetation pool tensor."""
        if key not in state:
            raise ValueError(f"Missing required phenology state input: {key}")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"Phenology state input '{key}' must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"Phenology state input '{key}' must be floating point")
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"Phenology state input '{key}' must be finite")
        if torch.any(value < 0.0):
            raise ValueError(f"Phenology state input '{key}' must be non-negative")
        return value

    def _compute_biomass_turnover(
        self,
        state: StateDict,
        pheno_output: PhenologyOutput,
        dt: float,
    ) -> BiomassFluxes:
        """
        Compute biomass turnover fluxes based on phenological state.
        
        Parameters:
            state : StateDict
                Current ecosystem state
            pheno_output : PhenologyOutput
                Phenological state information
            dt : float
                Integration interval [s]
                
        Returns:
            BiomassFluxes: Biomass turnover and cycling fluxes
        """
        # References:
        # - Thornton et al. (2002), Biome-BGC daily phenology turnover.
        # - CLM/E3SM CNOffsetLitterfall, leaf N litter and retranslocation split.
        # Mathematical expressions:
        # - F_leaf_C = C_leaf * k_leaf * f_sen; F_root_C = C_root * k_root.
        # - F_leaf_N = N_leaf * k_leaf * f_sen.
        # - F_retrans_N = eta_N * F_leaf_N; F_litter_N = (1 - eta_N) * F_leaf_N.
        # - delta_x = N_h * r_h * dt and C_h * N_h = 0 on the extended
        #   vegetation-pool plus litter-boundary ledger.
        # Parameter meanings and units:
        # - k_leaf, k_root: ParameterManager turnover coefficients [s-1].
        # - f_sen: subclass senescence response factor [1].
        # - eta_N: leaf nitrogen retranslocation efficiency [1].
        # - C and N pools: vegetation stocks [kg C m-2] and [kg N m-2].
        # Open-system route:
        # - Litter C and N rows are explicit boundary_export ports; retranslocated
        #   N is accumulated into plant_nitrogen_pool.
        leaf_carbon = self._require_pool_tensor(state, "leaf_carbon")
        prototype = leaf_carbon
        pool_values = [
            leaf_carbon,
            self._require_pool_tensor(state, "froot_carbon").to(
                device=prototype.device,
                dtype=prototype.dtype,
            ),
            self._require_pool_tensor(state, "leaf_nitrogen").to(
                device=prototype.device,
                dtype=prototype.dtype,
            ),
            self._require_pool_tensor(state, "froot_nitrogen").to(
                device=prototype.device,
                dtype=prototype.dtype,
            ),
        ]
        if not isinstance(pheno_output.senescence_rate, Tensor):
            raise TypeError("Phenology senescence factor must be a torch.Tensor")
        senescence_factor = pheno_output.senescence_rate.to(
            device=prototype.device,
            dtype=prototype.dtype,
        )
        if not torch.all(torch.isfinite(senescence_factor)):
            raise ValueError("Phenology senescence factor must be finite")
        leaf_carbon, froot_carbon, leaf_nitrogen, froot_nitrogen, senescence_factor = (
            torch.broadcast_tensors(*pool_values, senescence_factor)
        )

        leaf_turnover_coefficient = torch.as_tensor(
            self.phenology_params.leaf_turnover_rate,
            device=prototype.device,
            dtype=prototype.dtype,
        )
        froot_turnover_coefficient = torch.as_tensor(
            self.phenology_params.froot_turnover_rate,
            device=prototype.device,
            dtype=prototype.dtype,
        )
        retranslocation_efficiency = torch.as_tensor(
            self.nitrogen_retrans_efficiency,
            device=prototype.device,
            dtype=prototype.dtype,
        )
        leaf_turnover_flux = (
            leaf_carbon * leaf_turnover_coefficient * senescence_factor
        )
        froot_turnover_flux = froot_carbon * froot_turnover_coefficient
        leaf_nitrogen_turnover = (
            leaf_nitrogen * leaf_turnover_coefficient * senescence_factor
        )
        nitrogen_retranslocation = (
            leaf_nitrogen_turnover * retranslocation_efficiency
        )
        leaf_litterfall_nitrogen = (
            leaf_nitrogen_turnover - nitrogen_retranslocation
        )
        froot_litterfall_nitrogen = (
            froot_nitrogen * froot_turnover_coefficient
        )

        species = (
            "leaf_carbon",
            "froot_carbon",
            "leaf_nitrogen",
            "froot_nitrogen",
            "plant_nitrogen_pool",
            "leaf_litterfall_carbon",
            "froot_litterfall_carbon",
            "leaf_litterfall_nitrogen",
            "froot_litterfall_nitrogen",
        )
        reactions = (
            "leaf_carbon_turnover",
            "froot_carbon_turnover",
            "leaf_nitrogen_retranslocation",
            "leaf_nitrogen_litterfall",
            "froot_nitrogen_litterfall",
        )
        stoichiometric_n = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=prototype.device,
            dtype=prototype.dtype,
        )
        conservation_matrix = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ],
            device=prototype.device,
            dtype=prototype.dtype,
        )
        network = advance_reaction_network(
            name=f"{self.name}.phenology_cn_turnover",
            species=species,
            reactions=reactions,
            stoichiometric_N=stoichiometric_n,
            rates=(
                leaf_turnover_flux,
                froot_turnover_flux,
                nitrogen_retranslocation,
                leaf_litterfall_nitrogen,
                froot_litterfall_nitrogen,
            ),
            dt=dt,
            conservation_matrix=conservation_matrix,
            conservation_tolerance=self.conservation_tolerance,
            strict_conservation=True,
            require_conservation_matrix=True,
            open_system_route="boundary_export",
        )
        if network.state_delta is None:
            raise RuntimeError("Phenology turnover network did not produce state deltas")
        state_delta = network.state_delta
        delta_by_species = {
            key: state_delta[index]
            for index, key in enumerate(species)
        }
        carbon_residual = (
            delta_by_species["leaf_carbon"]
            + delta_by_species["froot_carbon"]
            + dt
            * (
                network.species_rate("leaf_litterfall_carbon")
                + network.species_rate("froot_litterfall_carbon")
            )
        )
        nitrogen_residual = (
            delta_by_species["leaf_nitrogen"]
            + delta_by_species["froot_nitrogen"]
            + delta_by_species["plant_nitrogen_pool"]
            + dt
            * (
                network.species_rate("leaf_litterfall_nitrogen")
                + network.species_rate("froot_litterfall_nitrogen")
            )
        )
        tolerance = torch.as_tensor(
            self.conservation_tolerance,
            device=prototype.device,
            dtype=prototype.dtype,
        )
        if torch.any(torch.abs(carbon_residual) > tolerance):
            raise ValueError("Phenology carbon turnover ledger did not close")
        if torch.any(torch.abs(nitrogen_residual) > tolerance):
            raise ValueError("Phenology nitrogen turnover ledger did not close")

        return {
            "leaf_carbon": delta_by_species["leaf_carbon"],
            "froot_carbon": delta_by_species["froot_carbon"],
            "leaf_nitrogen": delta_by_species["leaf_nitrogen"],
            "froot_nitrogen": delta_by_species["froot_nitrogen"],
            "plant_nitrogen_pool": delta_by_species["plant_nitrogen_pool"],
            "leaf_litterfall_carbon": network.species_rate(
                "leaf_litterfall_carbon"
            ),
            "froot_litterfall_carbon": network.species_rate(
                "froot_litterfall_carbon"
            ),
            "leaf_litterfall_nitrogen": network.species_rate(
                "leaf_litterfall_nitrogen"
            ),
            "froot_litterfall_nitrogen": network.species_rate(
                "froot_litterfall_nitrogen"
            ),
            "nitrogen_retranslocation": nitrogen_retranslocation,
            "diagnostics.phenology.carbon_turnover_balance_residual": carbon_residual,
            "diagnostics.phenology.nitrogen_turnover_balance_residual": nitrogen_residual,
        }

    def validate_phenology_parameters(self) -> List[str]:
        """
        Validate phenology parameter consistency and ranges.
        
        Returns:
            List[str]: Validation error messages
        """
        errors = []
        try:
            self.phenology_params.validate_for_phenology_type()
        except ValueError as e:
            errors.append(str(e))
        if self.phenology_params.phenology_type == 'gdd':
            if self.phenology_params.gdd_threshold_start >= self.phenology_params.gdd_threshold_end:
                errors.append('GDD start threshold must be less than end threshold')
        if self.phenology_params.phenology_type == 'gsi':
            if self.phenology_params.gsi_tmin_optimal <= self.phenology_params.gsi_tmin_threshold:
                errors.append('GSI optimal temperature must be greater than threshold')
        if self.phenology_params.max_lai <= self.phenology_params.min_lai:
            errors.append('Maximum LAI must be greater than minimum LAI')
        return errors
