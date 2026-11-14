"""
Water-Carbon Coupling Module

This module implements the coupling between water status and carbon processes,
including photosynthesis, autotrophic respiration, and soil heterotrophic 
respiration and decomposition. It quantifies how soil moisture, vapor pressure 
deficit, and plant water stress affect carbon cycle processes.

The coupling process involves:
1. Water stress effects on photosynthetic capacity (Vcmax, Jmax scaling)
2. Soil moisture regulation of heterotrophic respiration and decomposition  
3. VPD and soil drought impacts on plant carbon allocation
4. Optimal moisture content functions for soil biogeochemical processes

Mathematical Framework:
- Photosynthetic water stress: f_photo(theta, psi) = logistic or linear scaling
- Respiration moisture response: f_resp(theta) = parabolic with optimal theta_opt
- Q10 temperature response: R = R_ref * Q10^((T-Tref)/10)
- Combined: R = R_ref * Q10^((T-Tref)/10) * f_moisture(theta)

Units and Variables (CF-1.8 compliant):
- theta: soil moisture content [m3 m-3]
- psi: soil matric potential [Pa]
- VPD: vapor pressure deficit [Pa]
- f_*: dimensionless stress/response factors [1]
- GPP: gross primary productivity [kg C m-2 s-1]
- R_h: heterotrophic respiration [kg C m-2 s-1]

Conservation Laws:
- Carbon mass balance: GPP - R_a - R_h - BVOC - losses = DeltaC_storage
- Water-carbon consistency: water stress affects all C processes proportionally
- Non-negativity: all stress factors ? [0, 1]

References:
- Medlyn, B.E. et al. (2002). Temperature response of parameters of a 
  biochemically based model of photosynthesis. Plant, Cell & Environment
- Moyano, F.E. et al. (2013). Responses of soil heterotrophic respiration to 
  moisture availability. Soil Biology and Biochemistry
- Sierra, C.A. et al. (2015). Sensitivity of decomposition rates to moisture 
  and temperature in soil organic matter models. Journal of Advances in 
  Modeling Earth Systems
- Davidson, E.A. et al. (2012). The Dual Arrhenius and Michaelis-Menten (DAMM) 
  model for soil CO2 efflux. Global Change Biology
"""

import logging
import weakref

# Import parameter manager for default values

from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.functor import ScaleCouplingFunctor
from relic.core.abstractions import StateDict, FluxDict, DiagnosticsDict
from relic.core.variable_mapper import variable_name_mapper
from relic.data.standards.cf_conventions import CFConventions
from relic.utils.unified_response_functions import (
    temperature_response_q10,
    validate_temperature_range
)
from relic.utils.physical_constants import get_physical_constant
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace
from relic.coupling import WC_NAMESPACE
from relic.coupling.internal.common import CouplingParameterMixin

WATER_CARBON_NAMESPACE = WC_NAMESPACE
VALIDATION_NUMERICAL_NAMESPACE = ParameterNamespace.VALIDATION_NUMERICAL
DEFAULT_SOIL_DEPENDENCY = "soil_vegetation"

# Configure logger
logger = logging.getLogger(__name__)

MATHEMATICAL_EXPRESSION_WC = (
    "f_soil = soil-moisture response of theta (logistic, linear or piecewise); "
    "f_vpd = 1 / (1 + D / D_50); f_gpp = clamp(f_soil f_vpd) (Jarvis product); "
    "f_Ra = clamp(sqrt(f_gpp)); f_T = clamp(Q10^((T - T_ref)/10)); "
    "f_Rh = clamp(f_theta_resp min(f_T, c_T)); k_dec = f_theta_resp f_T; "
    "f_soil_carbon = k_dec; f_overall = clamp(min(f_soil, f_vpd)) (3-PG lower modifier)"
)

REFERENCES_WC = [
    "Landsberg, J.J. & Waring, R.H. (1997). A generalised model of forest productivity using "
    "simplified concepts of radiation-use efficiency, carbon balance and partitioning. "
    "For. Ecol. Manage. 95, 209-228",
    "Jarvis, P.G. (1976). The interpretation of the variations in leaf water potential and "
    "stomatal conductance found in canopies in the field. Phil. Trans. R. Soc. Lond. B 273, 593-610",
    "Lohammar, T., Larsson, S., Linder, S. & Falk, S.O. (1980). FAST - simulation models of "
    "gaseous exchange in Scots pine. Ecological Bulletins 32, 505-523",
    "Leuning, R. (1995). A critical appraisal of a combined stomatal-photosynthesis model for C3 "
    "plants. Plant, Cell & Environment 18, 339-357",
    "Medlyn, B.E. et al. (2002). Temperature response of parameters of a photosynthesis model",
    "Moyano, F.E. et al. (2013). Responses of soil heterotrophic respiration to moisture availability",
    "Sierra, C.A. et al. (2015). Sensitivity of decomposition rates to moisture and temperature",
    "Davidson, E.A. et al. (2012). Dual Arrhenius and Michaelis-Menten model for soil CO2 efflux",
]


@dataclass
class CarbonWaterStressParams:
    """
    Parameters for water stress response functions.
    
    Attributes:
        stress_function_type : str
            Type of stress function ("logistic", "linear", "piecewise")
        wilting_point_volumetric : float
            Permanent wilting point [m3 m-3] (volumetric)
        field_capacity : float
            Field capacity [m3 m-3] or [Pa]
        optimal_moisture : float
            Optimal soil moisture for processes [m3 m-3]
        stress_onset_threshold : float
            Threshold for onset of water stress [m3 m-3] or [Pa]
    """
    stress_function_type: str
    wilting_point_volumetric: float
    field_capacity: float
    optimal_moisture: float
    stress_onset_threshold: float


@dataclass
class RespirationMoistureParams:
    """
    Parameters for soil respiration moisture response.
    
    Attributes:
        optimal_moisture_fraction : float
            Optimal moisture as fraction of saturation [1]
        moisture_shape_parameter : float
            Shape parameter for moisture response curve
        anaerobic_threshold : float
            Moisture threshold for anaerobic conditions [m3 m-3]
        drought_threshold : float
            Moisture threshold for drought stress [m3 m-3]
    """
    optimal_moisture_fraction: float
    moisture_shape_parameter: float
    anaerobic_threshold: float
    drought_threshold: float


@dataclass
class WaterCarbonParameterCache:
    """Cached water-carbon coupling parameters for fast reuse."""

    stress_function_type: str
    field_capacity: Tensor
    wilting_point_volumetric: Tensor
    optimal_moisture: Tensor
    stress_onset_threshold: Tensor
    logistic_steepness: Tensor
    piecewise_partial_factor: Tensor
    optimal_moisture_fraction: Tensor
    moisture_shape_parameter: Tensor
    anaerobic_threshold: Tensor
    drought_threshold: Tensor
    minimum_stress_factor: Tensor
    maximum_moisture_factor: Tensor
    minimum_moisture_factor: Tensor
    q10_soil: Tensor
    q10_reference_normalized: Tensor
    temperature_response_cap: Tensor
    temperature_response_min_bound: Tensor
    temperature_response_max_bound: Tensor
    high_temperature_warning_threshold: Tensor
    vpd_half_saturation: Tensor
    reference_temperature: Tensor
    clamp_min_zero: Tensor
    clamp_max_one: Tensor


class WaterCarbonCoupling(CouplingParameterMixin, ScaleCouplingFunctor):
    """
    Water-Carbon Coupling Functor
    
    Implements the coupling between water availability and carbon processes
    following established plant and soil physiological models. This functor
    quantifies how drought stress, optimal moisture conditions, and water
    limitations affect ecosystem carbon cycling.
    
    Key Features:
    - Multiple water stress functions for photosynthesis (logistic, linear)
    - Soil moisture optimization curves for respiration and decomposition
    - VPD effects on plant carbon allocation and stress responses
    - Integration with Q10 temperature responses for soil processes
    - Consistent units and conservation checking
    
    Input Variables:
    - soil_moisture: Soil moisture content [m3 m-3]
    - vapor_pressure_deficit: Atmospheric VPD [Pa] (required)
    - soil_temperature: Soil temperature [K]
    
    Output Variables:
    - gpp_water_stress_factor: GPP reduction factor [1]
    - autotrophic_respiration_modifier: R_a modifier [1] 
    - heterotrophic_respiration_modifier: R_h modifier [1]
    - decomposition_rate_modifier: Decomposition modifier [1]
    - overall_water_stress: Lower of the soil-moisture and VPD factors [1] (3-PG)
    - soil_carbon_turnover_factor: Soil C cycling modifier [1]
    - diagnostics.water_carbon.temperature_response_factor: capped Q10 response min(f_T, c_T) [1]
    - diagnostics.water_carbon.temperature_response_factor_uncapped: uncapped Q10 response f_T [1]
    - diagnostics.water_carbon.moisture_response_factor: moisture response f_theta [1]
    
    Feedback semantics:
    - gpp_water_stress_factor is produced for downstream photosynthesis functors and is applied on the next timestep
      to avoid circular dependencies in the current coupling group. Initial conditions assume a neutral factor of 1.0.
    
    Mathematical Expressions:
    Water Stress Functions:
        Logistic: f(theta) = 1 / (1 + exp(-k*(theta - theta_50)))
        Linear: f(theta) = (theta - theta_min) / (theta_opt - theta_min) for theta < theta_opt
        Piecewise: Multi-segment linear function
    
    Soil Moisture Response for Respiration:
        Parabolic: f(theta) = 4 * (theta/theta_sat) * (1 - theta/theta_sat) * (theta_opt/theta_sat)^-1
        Where theta_opt is optimal moisture, theta_sat is saturation
    
    Temperature-Moisture Interaction:
        R = R_ref * Q10^((T-Tref)/10) * f_moisture(theta)
        Where Q10 varies with moisture availability
    
    VPD Effects:
        f_VPD(D) = 1 / (1 + D/D_50) for exponential decline
        Applied to photosynthesis and stomatal conductance

    Parameters
    ----------
    name : str
        Functor name.
    config : Dict
        Configuration for water-carbon coupling.
    metadata : Optional[FunctorMetadata]
        Scientific metadata (mathematical_expression, references, units).

    Returns
    -------
    Dict[str, Tensor]
        Water stress modifiers [dimensionless] and adjusted fluxes [kg C m-2 s-1].

    References
    ----------
    Rodriguez-Iturbe, I. & Porporato, A. (2004) ecohydrology of water-controlled ecosystems.
    Xu, L. & Baldocchi, D. (2004) seasonal variation of carbon exchange under drought.
    Bonan, G.B. (2019) terrestrial ecosystem modeling (water stress formulations).
    """
    DEPENDENCY_OVERRIDE_KEYS = (
        "upstream_soil_vegetation_functor",
        "upstream_soil_vegetation",
        "upstream_dependency",
        "depends_on",
    )
    
    def __init__(self, name: str, config: Dict, metadata: Optional[FunctorMetadata] = None,
                 parameter_manager: Optional[Any] = None):
        """
        Initialize water-carbon coupling functor.
        
        Parameters:
            name : str
                Functor name
            config : Dict
                Configuration containing water stress parameters
            metadata : Optional[FunctorMetadata]
                Scientific metadata
        """
        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                version="1.0.0",
                description=(
                    "Couples water availability with carbon processes including photosynthesis and respiration. "
                    "gpp_water_stress_factor is produced for downstream photosynthesis modules and is intended to be "
                    "consumed on the next timestep to avoid circular dependencies within the current step."
                ),
                category=ProcessCategory.SCALE_COUPLING,
                execution_phase=ExecutionPhase.COUPLING,
                mathematical_expression=MATHEMATICAL_EXPRESSION_WC,
                references=REFERENCES_WC,
                parameters={
                    "stress_function_type": "Functional form of soil water stress response (-)",
                    "wilting_point_volumetric": "Permanent wilting point (m3 m-3)",
                    "field_capacity_volumetric": "Field capacity moisture threshold (m3 m-3)",
                    "optimal_moisture": "Optimal soil moisture for photosynthesis (m3 m-3)",
                    "stress_onset_threshold": "Moisture threshold where drought stress begins (m3 m-3)",
                    "logistic_steepness": "Steepness of logistic stress curve (-)",
                    "piecewise_partial_factor": "Intermediate factor for piecewise stress response (-)",
                    "optimal_moisture_fraction": "Optimal respiration moisture as fraction of field capacity (-)",
                    "moisture_shape_parameter": "Shape coefficient for respiration response (-)",
                    "anaerobic_threshold": "Fraction of saturation triggering anaerobic limits (-)",
                    "drought_threshold": "Fraction of field capacity triggering drought limits (-)",
                    "minimum_stress_factor": "Lower guard for stress factors (-)",
                    "maximum_moisture_factor": "Upper guard for stress factors (-)",
                    "minimum_moisture_factor": "Lower guard for respiration multipliers (-)",
                    "q10_soil": "Q10 temperature coefficient for soil processes (-)",
                    "reference_temperature_celsius": "Reference temperature for Q10 scaling (degC)",
                    "temperature_response_min_bound": "Lower clamp for temperature response (-)",
                    "temperature_response_max_bound": "Upper clamp for temperature response (-)",
                    "high_temperature_warning_threshold": "Diagnostic threshold for extreme temperature response (-)",
                    "vpd_half_saturation": (
                        "D_0 of the hyperbolic VPD multiplier 1 / (1 + D / D_0) (Lohammar et al. "
                        "1980; Leuning 1995): the deficit at which the multiplier is 1/2 (Pa)"
                    ),
                },
                input_variables=[
                    "soil_moisture", "vapor_pressure_deficit",
                    "soil_temperature",
                ],
                output_variables=[
                    "gpp_water_stress_factor", "autotrophic_respiration_modifier",
                    "heterotrophic_respiration_modifier", "decomposition_rate_modifier",
                    "overall_water_stress", "soil_carbon_turnover_factor",
                    # Response factors read by the local consistency check.
                    "diagnostics.water_carbon.temperature_response_factor",
                    "diagnostics.water_carbon.temperature_response_factor_uncapped",
                    "diagnostics.water_carbon.moisture_response_factor",
                ],
                conservation_laws=[],
                spatial_dimensions=0,
                temporal_scale=TemporalScale.SECOND,
                coupling_interfaces=["photosynthesis", "respiration", "decomposition", "soil_moisture"],
                spatial_scale=SpatialScale.GRID_CELL,
                variable_kind=VariableKind.DIAGNOSTIC,
            )
        
        # Point-wise cross-domain coupling in one grid cell: the grid-cell soil water state and
        # vapour pressure deficit give grid-cell carbon-process modifiers (no change of scale).
        super().__init__(
            name,
            config,
            metadata,
            parameter_manager=parameter_manager,
            source_scale="grid_cell",
            target_scale="grid_cell",
            coupling_type="spatial",
        )
        # Resolved predecessors (instance declaration, else the override key,
        # else the default soil-vegetation owner) become the instance
        # declaration; the bridge default merges later add_dependency calls.
        self._initialize_coupling_dependencies(
            current_dependencies=self._dependencies,
            config=config,
            default_target=DEFAULT_SOIL_DEPENDENCY,
            override_keys=self.DEPENDENCY_OVERRIDE_KEYS,
        )
        self._wc_parameter_cache: Optional[WaterCarbonParameterCache] = None
        self._parameter_change_callback: Optional[Callable[[str, Any, Any], None]] = None
        self._t_freeze = float(get_physical_constant("celsius_to_kelvin"))
        self._wc_parameter_cache = self._load_parameter_cache()
        self._register_parameter_change_callback()

    def _wc_param(
        self,
        key: str,
        units: str,
        *,
        as_tensor: bool = True,
        required: bool = True,
        expected_type: Optional[type] = None,
    ) -> Any:
        """Shorthand for coupling.scales.water_carbon parameter retrieval."""
        return self._get_coupling_parameter(
            key,
            namespace=WATER_CARBON_NAMESPACE,
            units=units,
            required=required,
            as_tensor=as_tensor,
            expected_type=expected_type,
        )

    def _validation_param(self, key: str, units: str) -> Tensor:
        """Fetch validation/numerical guardrails with device/dtype alignment."""
        value = self.get_parameter_value(
            key,
            namespace=VALIDATION_NUMERICAL_NAMESPACE,
            units=units,
            validate=True,
            required=True,
        )
        return self._ensure_tensor(value, dtype=torch.float32)

    def _tensor_to_float(self, value: Union[Tensor, float, int]) -> float:
        """Convert tensors (or scalars) to floats for dataclasses."""
        return self._to_scalar(value)

    def _load_parameter_cache(self) -> WaterCarbonParameterCache:
        """Load and cache all frequently accessed parameters."""
        reference_temp_c = self._wc_param(
            "reference_temperature_celsius",
            "degC",
            as_tensor=False,
            required=True,
        )
        reference_temp_k = float(reference_temp_c) + self._t_freeze
        return WaterCarbonParameterCache(
            stress_function_type=str(
                self._wc_param(
                    "stress_function_type",
                    "dimensionless",
                    as_tensor=False,
                    expected_type=str,
                )
            ),
            field_capacity=self._ensure_tensor(
                self._wc_param("field_capacity_volumetric", "m3 m-3"), dtype=torch.float32
            ),
            wilting_point_volumetric=self._ensure_tensor(
                self._wc_param("wilting_point_volumetric", "m3 m-3"), dtype=torch.float32
            ),
            optimal_moisture=self._ensure_tensor(
                self._wc_param("optimal_moisture", "m3 m-3"), dtype=torch.float32
            ),
            stress_onset_threshold=self._ensure_tensor(
                self._wc_param("stress_onset_threshold", "m3 m-3"), dtype=torch.float32
            ),
            logistic_steepness=self._ensure_tensor(
                self._wc_param("logistic_steepness", "dimensionless"),
                dtype=torch.float32,
            ),
            piecewise_partial_factor=self._ensure_tensor(
                self._wc_param("piecewise_partial_factor", "dimensionless"),
                dtype=torch.float32,
            ),
            optimal_moisture_fraction=self._ensure_tensor(
                self._wc_param("optimal_moisture_fraction", "dimensionless"),
                dtype=torch.float32,
            ),
            moisture_shape_parameter=self._ensure_tensor(
                self._wc_param("moisture_shape_parameter", "dimensionless"),
                dtype=torch.float32,
            ),
            anaerobic_threshold=self._ensure_tensor(
                self._wc_param("anaerobic_threshold", "dimensionless"),
                dtype=torch.float32,
            ),
            drought_threshold=self._ensure_tensor(
                self._wc_param("drought_threshold", "dimensionless"),
                dtype=torch.float32,
            ),
            minimum_stress_factor=self._ensure_tensor(
                self._wc_param("minimum_stress_factor", "dimensionless"),
                dtype=torch.float32,
            ),
            maximum_moisture_factor=self._ensure_tensor(
                self._wc_param("maximum_moisture_factor", "dimensionless"),
                dtype=torch.float32,
            ),
            minimum_moisture_factor=self._ensure_tensor(
                self._wc_param("minimum_moisture_factor", "dimensionless"),
                dtype=torch.float32,
            ),
            q10_soil=self._ensure_tensor(
                self._wc_param("q10_soil", "dimensionless"), dtype=torch.float32
            ),
            q10_reference_normalized=self._ensure_tensor(
                self._wc_param("q10_reference_normalized", "dimensionless"),
                dtype=torch.float32,
            ),
            temperature_response_cap=self._ensure_tensor(
                self._wc_param("temperature_response_cap", "dimensionless"),
                dtype=torch.float32,
            ),
            temperature_response_min_bound=self._ensure_tensor(
                self._wc_param("temperature_response_min_bound", "dimensionless"),
                dtype=torch.float32,
            ),
            temperature_response_max_bound=self._ensure_tensor(
                self._wc_param("temperature_response_max_bound", "dimensionless"),
                dtype=torch.float32,
            ),
            high_temperature_warning_threshold=self._ensure_tensor(
                self._wc_param("high_temperature_warning_threshold", "dimensionless"),
                dtype=torch.float32,
            ),
            vpd_half_saturation=self._ensure_tensor(
                self._wc_param("vpd_half_saturation", "Pa"), dtype=torch.float32
            ),
            reference_temperature=self._ensure_tensor(
                reference_temp_k, dtype=torch.float32
            ),
            clamp_min_zero=self._validation_param("clamp_min_zero", "dimensionless"),
            clamp_max_one=self._validation_param("clamp_max_one", "dimensionless"),
        )

    def _get_cache(self) -> WaterCarbonParameterCache:
        """Return cached parameters, loading on first access."""
        if self._wc_parameter_cache is None:
            self._wc_parameter_cache = self._load_parameter_cache()
        return self._wc_parameter_cache

    def _refresh_cached_parameters(self) -> None:
        """Refresh cached parameters when the ParameterManager changes."""
        self._wc_parameter_cache = self._load_parameter_cache()

    def _is_relevant_parameter(self, key: str) -> bool:
        """Determine whether a parameter change should refresh the cache."""
        prefixes = (
            f"{resolve_namespace(WATER_CARBON_NAMESPACE)}.",
            f"{resolve_namespace(VALIDATION_NUMERICAL_NAMESPACE)}.",
        )
        return any(key.startswith(prefix) for prefix in prefixes)

    def _register_parameter_change_callback(self) -> None:
        """Register callback to rebuild cache after ParameterManager updates."""
        manager = self.get_parameter_manager()
        if manager is None:
            return

        # A weak reference: the manager's callback list must not keep the coupling alive,
        # otherwise __del__ (which unregisters the callback) could never run.
        owner = weakref.ref(self)

        def _callback(key: str, old_value: Any, new_value: Any) -> None:
            coupling = owner()
            if coupling is not None and coupling._is_relevant_parameter(key):
                coupling._refresh_cached_parameters()

        manager.add_change_callback(_callback)
        self._parameter_change_callback = _callback

    def __del__(self) -> None:
        # Teardown: unregister the callback this instance registered. An instance whose
        # construction stopped earlier registered none and has nothing to undo;
        # remove_change_callback does not raise for an unregistered callback.
        callback = self.__dict__.get("_parameter_change_callback")
        manager = self.__dict__.get("_parameter_manager")
        if callback is not None and manager is not None:
            manager.remove_change_callback(callback)

    @property
    def stress_function_type(self) -> str:
        return self._get_cache().stress_function_type

    @property
    def field_capacity(self) -> Tensor:
        return self._get_cache().field_capacity

    @property
    def wilting_point_volumetric(self) -> Tensor:
        return self._get_cache().wilting_point_volumetric

    @property
    def optimal_moisture(self) -> Tensor:
        return self._get_cache().optimal_moisture

    @property
    def stress_onset_threshold(self) -> Tensor:
        return self._get_cache().stress_onset_threshold

    @property
    def logistic_steepness(self) -> Tensor:
        return self._get_cache().logistic_steepness

    @property
    def piecewise_partial_factor(self) -> Tensor:
        return self._get_cache().piecewise_partial_factor

    @property
    def optimal_moisture_fraction(self) -> Tensor:
        return self._get_cache().optimal_moisture_fraction

    @property
    def moisture_shape_parameter(self) -> Tensor:
        return self._get_cache().moisture_shape_parameter

    @property
    def anaerobic_threshold(self) -> Tensor:
        return self._get_cache().anaerobic_threshold

    @property
    def drought_threshold(self) -> Tensor:
        return self._get_cache().drought_threshold

    @property
    def minimum_stress_factor(self) -> Tensor:
        return self._get_cache().minimum_stress_factor

    @property
    def maximum_moisture_factor(self) -> Tensor:
        return self._get_cache().maximum_moisture_factor

    @property
    def minimum_moisture_factor(self) -> Tensor:
        return self._get_cache().minimum_moisture_factor

    @property
    def q10_soil(self) -> Tensor:
        return self._get_cache().q10_soil

    @property
    def q10_reference_normalized(self) -> Tensor:
        return self._get_cache().q10_reference_normalized

    @property
    def temperature_response_cap(self) -> Tensor:
        return self._get_cache().temperature_response_cap

    @property
    def temperature_response_min_bound(self) -> Tensor:
        return self._get_cache().temperature_response_min_bound

    @property
    def temperature_response_max_bound(self) -> Tensor:
        return self._get_cache().temperature_response_max_bound

    @property
    def high_temperature_warning_threshold(self) -> Tensor:
        return self._get_cache().high_temperature_warning_threshold

    @property
    def vpd_half_saturation(self) -> Tensor:
        return self._get_cache().vpd_half_saturation

    @property
    def reference_temperature(self) -> Tensor:
        return self._get_cache().reference_temperature

    @property
    def clamp_min_zero(self) -> Tensor:
        return self._get_cache().clamp_min_zero

    @property
    def clamp_max_one(self) -> Tensor:
        return self._get_cache().clamp_max_one

    def _get_stress_params(self) -> CarbonWaterStressParams:
        """Build fresh stress parameter dataclass to reflect current ParameterManager values."""
        return CarbonWaterStressParams(
            stress_function_type=self.stress_function_type,
            wilting_point_volumetric=self._tensor_to_float(self.wilting_point_volumetric),
            field_capacity=self._tensor_to_float(self.field_capacity),
            optimal_moisture=self._tensor_to_float(self.optimal_moisture),
            stress_onset_threshold=self._tensor_to_float(self.stress_onset_threshold),
        )

    def _get_respiration_params(self) -> RespirationMoistureParams:
        """Build respiration moisture parameters on demand."""
        return RespirationMoistureParams(
            optimal_moisture_fraction=self._tensor_to_float(self.optimal_moisture_fraction),
            moisture_shape_parameter=self._tensor_to_float(self.moisture_shape_parameter),
            anaerobic_threshold=self._tensor_to_float(self.anaerobic_threshold),
            drought_threshold=self._tensor_to_float(self.drought_threshold),
        )

    def compute_soil_moisture_stress_factor(self, soil_moisture: Tensor) -> Tensor:
        """
        Compute the soil-moisture stress factor f_soil.
        
        Parameters:
            soil_moisture : Tensor
                Soil moisture content [m3 m-3]
                
        Returns:
            Tensor: Soil-moisture stress factor f_soil [1] (0 = full stress, 1 = no stress)
            
        Mathematical Expressions:
            Logistic: f(theta) = 1 / (1 + exp(-k*(theta - theta_50)))
            Linear: f(theta) = max(0, min(1, (theta - theta_wilt)/(theta_opt - theta_wilt)))
            Piecewise: Multi-segment linear with thresholds
        """
        stress_params = self._get_stress_params()
        theta = torch.clamp(soil_moisture, min=self.clamp_min_zero, max=self.clamp_max_one)
        
        theta_wilt = self.wilting_point_volumetric
        theta_opt = self.optimal_moisture
        theta_onset = self.stress_onset_threshold
        
        if stress_params.stress_function_type == "logistic":
            # Logistic function centered at optimal moisture
            f_soil = torch.sigmoid(self.logistic_steepness * (theta - theta_onset))
            
        elif stress_params.stress_function_type == "linear":
            # Linear function from wilting point to optimal
            f_soil = torch.clamp(
                (theta - theta_wilt) / (theta_opt - theta_wilt),
                min=self.minimum_stress_factor,
                max=self.maximum_moisture_factor,
            )
            
        elif stress_params.stress_function_type == "piecewise":
            # Piecewise linear function with multiple segments
            f_soil = torch.zeros_like(theta)
            
            # No stress above optimal moisture
            f_soil = torch.where(theta >= theta_opt, torch.ones_like(theta), f_soil)
            
            # Linear increase from onset to optimal
            mask1 = (theta >= theta_onset) & (theta < theta_opt)
            f_soil = torch.where(mask1, (theta - theta_onset) / (theta_opt - theta_onset), f_soil)
            
            # Linear decrease from wilting to onset
            mask2 = (theta >= theta_wilt) & (theta < theta_onset)
            f_soil = torch.where(
                mask2,
                self.piecewise_partial_factor
                * (theta - theta_wilt)
                / (theta_onset - theta_wilt),
                f_soil,
            )
            
            # Zero below wilting point
            f_soil = torch.where(theta < theta_wilt, torch.zeros_like(theta), f_soil)
            
        else:
            raise ValueError(f"Unknown stress function type: {stress_params.stress_function_type}")
        
        return f_soil

    def compute_vpd_stress_factor(self, vpd: Tensor) -> Tensor:
        """Compute the VPD stress factor f_vpd [1] of a required, non-negative D [Pa]."""
        # References: Lohammar et al. (1980), Ecological Bulletins 32, 505-523: the
        #   hyperbolic multiplier (1 + D_s/D_o)^-1 of multiple-constraint stomatal models
        #   (review: Oren et al. 1999, Plant Cell Environ. 22, 1515-1526); the same humidity
        #   factor 1 / (1 + D_s/D_0) enters the gs model of Leuning (1995), Plant Cell
        #   Environ. 18, 339-357. Not the Medlyn et al. (2011) g1 / sqrt(D) form.
        # Mathematical expression: f_vpd = 1 / (1 + D / D_50).
        # Parameter meanings: D the vapor pressure deficit; D_50 vpd_half_saturation, the
        #   empirical D_0 of Lohammar and Leuning (fitted by least squares), the deficit
        #   at which the multiplier is 1/2.
        # Units: D and D_50 [Pa]; f_vpd [1].
        # Domain/constraints: D is a required input, finite and D >= 0 (the contract
        #   lower bound, e_s >= e_a); a negative or non-finite D raises and is not clamped.
        # Sign convention: f_vpd is a limitation factor, 1 meaning no stress.
        if not bool(torch.all(torch.isfinite(vpd))) or bool(torch.any(vpd < 0)):
            raise ValueError(
                "vapor_pressure_deficit must be finite and non-negative (D = e_s - e_a >= 0)"
            )
        return 1.0 / (1.0 + vpd / self.vpd_half_saturation)

    def compute_photosynthetic_water_stress(
        self,
        soil_stress_factor: Tensor,
        vpd_stress_factor: Tensor,
    ) -> Tensor:
        """Photosynthetic water stress f_gpp = clamp(f_soil f_vpd) [1].

        References: Jarvis (1976), Phil. Trans. R. Soc. Lond. B 273, 593-610
          (multiplicative limitation factors of stomatal conductance).
        Mathematical expression: f_gpp = clamp(f_soil f_vpd, f_min, f_max).
        Parameter meanings: f_soil and f_vpd the soil-moisture and VPD factors;
          f_min minimum_stress_factor, f_max maximum_moisture_factor.
        Units: all factors [1].
        Domain/constraints: f_soil and f_vpd in [0, 1]; the product is clamped to
          [f_min, f_max].
        Sign convention: factors are limitations, 1 meaning no stress.
        """
        return torch.clamp(
            soil_stress_factor * vpd_stress_factor,
            min=self.minimum_stress_factor,
            max=self.maximum_moisture_factor,
        )
    
    def compute_soil_respiration_moisture_response(self,
                                                 soil_moisture: Tensor,
                                                 soil_temperature: Tensor) -> Tensor:
        """
        Compute soil moisture response factor for respiration and decomposition.
        
        Parameters:
            soil_moisture : Tensor
                Soil moisture content [m3 m-3] 
            soil_temperature : Tensor
                Soil temperature [K]
                
        Returns:
            Tensor: Moisture response factor [1]
            
        Mathematical Expression:
            Parabolic form: f(theta) = a * theta * (1 - theta/theta_sat) where a normalizes to peak at theta_opt
            Combined: f(theta) = f_drought(theta) * f_anaerobic(theta) 
            
            Following CASA/CENTURY approach:
            f(theta) = (theta/theta_opt)^p * (1 - theta/theta_sat)^q
            where p, q are shape parameters
        """
        theta = torch.clamp(
            soil_moisture,
            min=self.minimum_moisture_factor,
            max=self.clamp_max_one,
        )
        theta_sat = self.field_capacity
        theta_opt = self.optimal_moisture_fraction * theta_sat
        theta_drought = self.drought_threshold
        theta_anaer = self.anaerobic_threshold * theta_sat
        
        # Parabolic moisture response with optimal point
        # Following Sierra et al. (2015) formulation
        shape_param = self.moisture_shape_parameter
        
        # Drought limitation (increases from 0 to 1)
        f_drought = torch.where(
            theta < theta_drought,
            torch.zeros_like(theta),
            torch.clamp(theta / theta_opt, max=self.clamp_max_one),
        )
        
        # Optimal range response  
        f_optimal = torch.where((theta >= theta_drought) & (theta <= theta_opt),
                               torch.ones_like(theta),
                               torch.zeros_like(theta))
        
        # Anaerobic limitation (decreases from 1 to 0)
        f_anaerobic = torch.where(
            theta > theta_opt,
            torch.clamp(
                1.0 - (theta - theta_opt) / (theta_anaer - theta_opt),
                min=self.minimum_stress_factor,
            ),
            torch.ones_like(theta),
        )
        
        # Combined response
        f_moisture = f_drought * f_optimal + torch.where(theta > theta_opt, f_anaerobic, torch.zeros_like(theta))
        
        # Apply shape parameter to modulate response curve
        f_moisture = torch.pow(f_moisture, 1.0 / shape_param)
        
        return torch.clamp(
            f_moisture,
            min=self.minimum_moisture_factor,
            max=self.maximum_moisture_factor,
        )
    
    def compute_temperature_response(self,
                                   temperature: Tensor,
                                   q10: Optional[Tensor] = None) -> Tensor:
        """
        Compute Q10 temperature response factor.
        
        Parameters:
            temperature : Tensor
                Soil or air temperature [K]
            q10 : Optional[Tensor]
                Q10 parameter [1], defaults to configured value
                
        Returns:
            Tensor: Temperature response factor [1]
            
        Mathematical Expression:
            f_T = Q10^((T - T_ref)/10)
            where T_ref = 10degC = 283.15 K
        """
        # Validate temperature range
        validate_temperature_range(temperature)
        
        if q10 is None:
            q10 = self.q10_soil.to(temperature.device)

        # Use unified response function
        f_temp = temperature_response_q10(
            param_25=self.q10_reference_normalized.to(temperature.device),  # Normalized reference value at 25 degC
            temperature=temperature,
            q10=q10,
            reference_temp=self.reference_temperature.to(temperature.device)
        )
        
        # Apply reasonable bounds to prevent numerical instability
        f_temp = torch.clamp(
            f_temp,
            min=self.temperature_response_min_bound,
            max=self.temperature_response_max_bound,
        )
        
        # Log warnings for extreme values
        if torch.any(f_temp > self.high_temperature_warning_threshold):
            logger.warning(f"High temperature response factor detected: max = {f_temp.max().detach().item():.2f}")
        
        return f_temp
    
    def compute_decomposition_rate_modifier(self,
                                          soil_moisture: Tensor,
                                          soil_temperature: Tensor) -> Tensor:
        """
        Compute combined temperature-moisture modifier for decomposition.
        
        Parameters:
            soil_moisture : Tensor
                Soil moisture [m3 m-3]
            soil_temperature : Tensor
                Soil temperature [K]
                
        Returns:
            Tensor: Combined rate modifier [1]
            
        Mathematical Expression:
            k = k_ref * f_T(T) * f_theta(theta)
            where k_ref is reference rate, f_T is temperature response, 
            f_theta is moisture response
        """
        f_temp = self.compute_temperature_response(soil_temperature)
        f_moisture = self.compute_soil_respiration_moisture_response(soil_moisture, soil_temperature)
        
        # Multiplicative combination
        combined_modifier = f_temp * f_moisture
        
        return combined_modifier

    def _combine_water_stress(
        self,
        soil_stress_factor: Tensor,
        vpd_stress: Tensor,
    ) -> Tensor:
        """
        Combine the soil-moisture and VPD factors: the lower one limits (3-PG).

        References:
            - Landsberg, J.J. and Waring, R.H. (1997). For. Ecol. Manage. 95, 209-228:
              the soil-water and vapour pressure deficit modifiers are not multiplicative
              relative to each other, and 3-PG uses the lower value of f_D and f_theta
              (stomata already closed by D make soil water non-limiting, and vice versa).
            - Jarvis (1976): the multiplicative form, kept for the photosynthetic factor
              f_gpp = f_soil f_vpd, not for this combination.
        Mathematical expression:
            f_overall = clamp(min(f_soil, f_vpd), f_min, f_max)
        Parameter meanings:
            f_soil the soil-moisture factor (f_theta), f_vpd the VPD factor;
            f_min minimum_stress_factor, f_max clamp_max_one (the guards of every
            emitted stress factor).
        Units:
            all factors and bounds are dimensionless [1].
        Domain/constraints:
            f_soil and f_vpd in [0, 1]; no weights are involved.
        Sign convention:
            stress factors are limitations, 1 meaning no stress.
        """
        combined = torch.minimum(soil_stress_factor, vpd_stress)

        return torch.clamp(
            combined,
            min=self.minimum_stress_factor,
            max=self.clamp_max_one,
        )
    
    def forward(self, 
               state: StateDict, 
               time: float, 
               dt: float,
               **kwargs) -> FluxDict:
        """
        Forward pass for water-carbon coupling.
        
        Computes water stress factors for photosynthesis, respiration, and 
        decomposition based on soil moisture, soil potential, and atmospheric
        conditions. Applies these factors to modify carbon process rates.
        
        Parameters:
            state : StateDict
                Current system state containing:
                - soil_moisture [m3 m-3]
                - soil_temperature [K]
                - vapor_pressure_deficit [Pa] (required, >= 0)
            time : float
                Current simulation time [s]
            dt : float
                Time step [s]
            **kwargs
                Additional environment parameters
                
        Returns:
            FluxDict: Water stress factors and modifiers
                - gpp_water_stress_factor [1]
                - autotrophic_respiration_modifier [1]
                - heterotrophic_respiration_modifier [1] 
                - decomposition_rate_modifier [1]
                - overall_water_stress [1]
                - soil_carbon_turnover_factor [1]
                - diagnostics.water_carbon.temperature_response_factor, diagnostics.water_carbon.temperature_response_factor_uncapped,
                  diagnostics.water_carbon.moisture_response_factor [1] (diagnostics)
        """
        # References: Moyano et al. (2013); Sierra et al. (2015); Davidson et al. (2012);
        #   Medlyn et al. (2002); Jarvis (1976) (f_gpp product); Lohammar et al. (1980) and
        #   Leuning (1995) (f_vpd); Landsberg and Waring (1997) (f_overall); module docstring.
        # Mathematical expression: f_gpp = f_soil(theta) / (1 + D/D_50); f_T =
        #   Q10^((T - T_ref)/10) clamped to the response bounds; f_theta = piecewise moisture
        #   response^(1/p); f_Ra = sqrt(f_gpp); f_Rh = clamp(f_theta min(f_T, c_T)) with the
        #   temperature_response_cap c_T; k_dec = f_T f_theta (uncapped f_T), the
        #   dimensionless Q10 multiplier of the decomposition owner's rate;
        #   f_all = w_theta f_gpp + w_vpd f_vpd.
        # Parameter meanings: theta soil moisture (profile mean if given); T soil
        #   temperature; D vapor pressure deficit; D_50 half-saturation VPD; Q10 soil
        #   coefficient; T_ref reference; p shape; w_* stress weights.
        # Units: theta [m3 m-3]; T, T_ref [K]; D, D_50 [Pa]; f_*, k_dec [1].
        # Domain/constraints: soil moisture and temperature (value or profile) and the
        #   vapor pressure deficit required; T in [233.15, 333.15] K; modifiers clamped to YAML guards in
        #   [0, 1], f_T to temperature_response_* bounds; bare tensors; no state advance.
        # Sign convention: not applicable: outputs are non-negative multiplicative
        #   modifiers; no flux, source, or sink is produced.
        # Use centralized variable name mapping for consistent key resolution
        state_std = variable_name_mapper.create_mapped_view(state, 'standard')
        
        # Extract required state variables with standardized names
        # Handle both single value and profile inputs
        if 'soil_moisture' in state_std:
            soil_moisture = state_std['soil_moisture']  # m3 m-3
        elif 'soil_moisture_profile' in state_std:
            # Take mean if profile is provided
            soil_moisture = torch.mean(state_std['soil_moisture_profile'])
        else:
            raise ValueError("Required soil moisture not found. Expected 'soil_moisture' or 'soil_moisture_profile'")
        
        # Handle temperature with profile input
        if 'soil_temperature' in state_std:
            soil_temp = state_std['soil_temperature']  # K
        elif 'soil_temperature_profile' in state_std:
            soil_temp = torch.mean(state_std['soil_temperature_profile'])
        else:
            raise ValueError("Required soil temperature not found")
        
        # The vapor pressure deficit enters f_gpp and the combined stress; it is a
        # required input (no f_vpd = 1 stand-in, which would mean D = 0).
        if 'vapor_pressure_deficit' not in state_std:
            raise ValueError(
                "WaterCarbonCoupling requires the input vapor_pressure_deficit [Pa]"
            )
        vpd = state_std['vapor_pressure_deficit']  # Pa
        
        # Ensure tensors have proper dimensions
        if soil_moisture.dim() == 0:
            soil_moisture = soil_moisture.unsqueeze(0)
        if soil_temp.dim() == 0:
            soil_temp = soil_temp.unsqueeze(0)
        
        # Soil-moisture and VPD factors, each evaluated once; the photosynthetic factor
        # is their clamped Jarvis product f_gpp = clamp(f_soil f_vpd).
        soil_stress_factor = self.compute_soil_moisture_stress_factor(soil_moisture)
        vpd_stress = self.compute_vpd_stress_factor(vpd)
        gpp_stress_factor = self.compute_photosynthetic_water_stress(
            soil_stress_factor, vpd_stress
        )
        
        # Compute soil respiration moisture response
        resp_moisture_response = self.compute_soil_respiration_moisture_response(
            soil_moisture=soil_moisture,
            soil_temperature=soil_temp
        )
        
        # Compute temperature response for soil processes
        # Two temperature usages are distinct on purpose: the uncapped Q10 factor scales
        # the decomposition rate (k_dec = f_theta f_T), while the bounded heterotrophic
        # modifier uses min(f_T, temperature_response_cap) so it stays a limitation factor.
        temp_response_uncapped = self.compute_temperature_response(soil_temp)
        temp_response = torch.clamp(
            temp_response_uncapped,
            max=torch.as_tensor(
                self.temperature_response_cap,
                device=temp_response_uncapped.device,
                dtype=temp_response_uncapped.dtype,
            ),
        )
        
        # Compute combined decomposition rate modifier
        decomp_modifier = self.compute_decomposition_rate_modifier(
            soil_moisture=soil_moisture,
            soil_temperature=soil_temp
        )
        
        # Autotrophic respiration modifier (similar to photosynthesis but less sensitive)
        # Apply square root to make it less sensitive than photosynthesis
        # Clamp the result to ensure it stays within [0, 1]
        ra_modifier = torch.clamp(
            torch.sqrt(
                torch.clamp(
                    gpp_stress_factor,
                    min=self.minimum_stress_factor,
                    max=self.maximum_moisture_factor,
                )
            ),
            min=self.minimum_stress_factor,
            max=self.clamp_max_one,
        )
        
        # Heterotrophic respiration modifier (soil moisture and temperature)
        # Clamp to ensure within bounds [0, 1] for consistency
        rh_modifier = torch.clamp(
            resp_moisture_response * temp_response,
            min=self.minimum_moisture_factor,
            max=self.maximum_moisture_factor,
        )
        
        # References: Landsberg and Waring (1997), For. Ecol. Manage. 95, 209-228 (3-PG):
        #   the soil-water and VPD modifiers are not multiplicative relative to each
        #   other; the lower one limits (stomata closed by D make soil water
        #   non-limiting, and vice versa). The Jarvis (1976) product is f_gpp above.
        # Mathematical expression: f_overall = clamp(min(f_soil, f_vpd)), f_vpd =
        #   1 / (1 + D / D_50).
        # Parameter meanings: f_soil the soil-moisture factor (f_theta), f_vpd the VPD
        #   factor, D_50 vpd_half_saturation.
        # Units: D and D_50 [Pa]; factors [1].
        # Domain/constraints: f_soil, f_vpd in [0, 1]; the result is clamped to
        #   [minimum_stress_factor, clamp_max_one]; no weights are involved.
        # Sign convention: factors are limitations, 1 meaning no stress.
        overall_stress = self._combine_water_stress(
            soil_stress_factor=soil_stress_factor,
            vpd_stress=vpd_stress,
        )
        
        # Soil carbon turnover factor (combination of decomposition processes)
        soil_carbon_factor = decomp_modifier
        
        
        stress_params = self._get_stress_params()
        # Generate bounded variable metadata for water-carbon coupling.
        CFConventions()
        _ = {
            'cell_methods': f'time: instantaneous (interval: {dt} seconds)',
            'history': f'Computed by {self.metadata.name} v{self.metadata.version} at time {time}s',
            'coupling_type': 'water_carbon_interaction',
            'stress_functions': f'{stress_params.stress_function_type}_water_stress'
        }
        
        outputs = {
            'gpp_water_stress_factor': gpp_stress_factor,
            'autotrophic_respiration_modifier': ra_modifier,
            'heterotrophic_respiration_modifier': rh_modifier,
            'decomposition_rate_modifier': decomp_modifier,
            'overall_water_stress': overall_stress,
            'soil_carbon_turnover_factor': soil_carbon_factor,
            'diagnostics.water_carbon.temperature_response_factor': temp_response,
            'diagnostics.water_carbon.temperature_response_factor_uncapped': temp_response_uncapped,
            'diagnostics.water_carbon.moisture_response_factor': resp_moisture_response,
        }

        # Definitional consistency of the emitted modifiers (no mass is advanced here).
        conservation_diags = self.check_conservation_laws(state_std, outputs)
        if conservation_diags:
            logger.debug("Modifier consistency diagnostics: %s", conservation_diags)

        return outputs

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Verify the definitional consistency of the emitted modifiers.

        References:
            - Moyano et al. (2013); Davidson et al. (2012): multiplicative moisture and
              temperature modifiers of decomposition and respiration.
        Mathematical expression:
            r_mod = max|f_Rh - clamp(f_theta * min(f_T, c_T), [f_min, f_max])|
                  + max|k_dec - f_theta * f_T| + max|f_soil - k_dec|;
            n_bound = count of emitted stress factors outside [0, 1].
        Parameter meanings:
            f_theta moisture response, f_T uncapped Q10 response, c_T the
            temperature_response_cap, f_Rh heterotrophic modifier, k_dec the
            decomposition rate modifier, f_soil the soil carbon turnover factor;
            f_theta, f_T and min(f_T, c_T) are the declared diagnostics
            diagnostics.water_carbon.{moisture_response_factor,
            temperature_response_factor_uncapped, temperature_response_factor}.
        Units:
            all modifiers and r_mod are dimensionless [1]; n_bound is a count.
        Domain/constraints:
            the coupling advances no mass, so no element ledger applies; residuals
            are computed only from outputs present in output_updates; missing
            outputs leave the corresponding residual out of the result.
        Sign convention:
            not applicable: residuals are absolute magnitudes of definitional
            identities, not directed fluxes.
        """
        diagnostics: Dict[str, float] = {}

        def _tensor(name: str) -> Optional[Tensor]:
            value = output_updates.get(name)
            if value is None:
                return None
            if isinstance(value, Tensor):
                return value
            return torch.as_tensor(value, device=self.device, dtype=torch.float32)

        f_theta = _tensor("diagnostics.water_carbon.moisture_response_factor")
        f_t_uncapped = _tensor("diagnostics.water_carbon.temperature_response_factor_uncapped")
        f_t_capped = _tensor("diagnostics.water_carbon.temperature_response_factor")
        f_rh = _tensor("heterotrophic_respiration_modifier")
        k_dec = _tensor("decomposition_rate_modifier")
        f_soil = _tensor("soil_carbon_turnover_factor")

        residual: Optional[Tensor] = None
        if f_theta is not None and f_t_capped is not None and f_rh is not None:
            expected_rh = torch.clamp(
                f_theta * f_t_capped,
                min=self.minimum_moisture_factor.to(device=f_theta.device, dtype=f_theta.dtype),
                max=self.maximum_moisture_factor.to(device=f_theta.device, dtype=f_theta.dtype),
            )
            residual = torch.max(torch.abs(f_rh - expected_rh))
        if f_theta is not None and f_t_uncapped is not None and k_dec is not None:
            term = torch.max(torch.abs(k_dec - f_theta * f_t_uncapped))
            residual = term if residual is None else residual + term
        if k_dec is not None and f_soil is not None:
            term = torch.max(torch.abs(f_soil - k_dec))
            residual = term if residual is None else residual + term
        if residual is not None:
            diagnostics["modifier_consistency"] = float(residual.detach().cpu().item())

        violations = 0.0
        for name in (
            "gpp_water_stress_factor",
            "overall_water_stress",
            "autotrophic_respiration_modifier",
            "heterotrophic_respiration_modifier",
            "diagnostics.water_carbon.moisture_response_factor",
        ):
            factor = _tensor(name)
            if factor is not None:
                violations += float(torch.sum((factor < 0) | (factor > 1)).item())
        diagnostics["bound_violations"] = violations
        return diagnostics
    
    def diagnose_conservation(self, 
                            input_state: StateDict,
                            output_fluxes: FluxDict) -> 'DiagnosticsDict':
        """
        Diagnose conservation and consistency checks.
        
        Parameters:
            input_state : StateDict
                Input state variables
            output_fluxes : FluxDict
                Computed output fluxes
                
        Returns:
            DiagnosticsDict: Conservation diagnostics as tensors
        """
        diagnostics = {}

        reference_tensor = next(iter(output_fluxes.values())) if output_fluxes else torch.zeros(1)
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        conservation = self.check_conservation_laws(input_state, output_fluxes)
        # The residual exists only when its outputs were emitted; an absent residual is
        # not reported as a zero (a zero would read as a satisfied identity).
        if "modifier_consistency" in conservation:
            diagnostics['modifier_consistency_residual'] = torch.tensor(
                conservation["modifier_consistency"], device=device, dtype=dtype
            )

        # Keep existing quality checks
        for key in ['gpp_water_stress_factor', 'overall_water_stress', 
                   'autotrophic_respiration_modifier', 'heterotrophic_respiration_modifier']:
            if key in output_fluxes:
                factor = output_fluxes[key]
                violations = torch.sum((factor < 0) | (factor > 1)).to(dtype=dtype)
                diagnostics[f'{key}_bound_violations'] = violations
        
        decomp_modifier = output_fluxes.get('decomposition_rate_modifier')
        if decomp_modifier is not None:
            negative_decomp_violations = torch.sum(decomp_modifier < 0).to(dtype=dtype)
            diagnostics['negative_decomposition_violations'] = negative_decomp_violations
        
        moisture_response = output_fluxes.get('diagnostics.water_carbon.moisture_response_factor')
        if moisture_response is not None:
            moisture_violations = torch.sum((moisture_response < 0) | (moisture_response > 1)).to(dtype=dtype)
            diagnostics['moisture_response_violations'] = moisture_violations
        
        return diagnostics

__all__ = [
    'WaterCarbonCoupling',
    'CarbonWaterStressParams',
    'RespirationMoistureParams',
]
