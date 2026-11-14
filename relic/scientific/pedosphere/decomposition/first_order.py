"""
First-Order Decomposition Model for RELIC Soil Carbon

This module implements first-order kinetic decomposition of soil organic matter
following the approaches established in RothC, CENTURY, and CASA models. The
implementation provides multiple soil carbon pools with temperature and moisture
controls on decomposition rates.

Scientific Basis:
-----------------
First-order decomposition follows the basic principle that decomposition rate
is proportional to the size of the carbon pool. This approach is widely used
in soil carbon models and provides a good balance between simplicity and
biological realism.

References:
- Coleman, K. & Jenkinson, D.S. (1996). RothC-26.3 - A Model for the turnover
  of carbon in soil. In: Evaluation of Soil Organic Matter Models. NATO ASI Series I 38.
- Parton, W.J. et al. (1987). Analysis of factors controlling soil organic matter
  levels in Great Plains grasslands. Soil Science Society of America Journal, 51, 1173-1179.
- Sierra, C.A. et al. (2012). Models of soil organic matter decomposition: the SoilR package.
  Geoscientific Model Development, 5, 1045-1060.
- Lloyd, J. & Taylor, J.A. (1994). On the temperature dependence of soil respiration.
  Functional Ecology, 8, 315-323.

Mathematical Expression:
-----------------------
Basic first-order equation:
dC/dt = -k(T,theta) * C

Where:
- C: carbon pool size [g C m^-2]
- k(T,theta): environmentally-modified decomposition rate [s^-1]
- T: temperature [K]
- theta: soil moisture [m^3 m^-3]

Environmental modification:
k(T,theta) = k_ref * f_T(T) * f_theta(theta)

Transfer between pools:
dC_i/dt = Sigma(T_ji * k_j * C_j) - k_i * C_i + I_i

Where:
- T_ji: transfer coefficient from pool j to pool i
- I_i: external inputs to pool i
"""

from typing import Any, Dict, Optional
import logging
import torch

# Import core framework
from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.abstractions import StateDict, FluxDict

# Import unified response functions
from relic.utils.unified_response_functions import (
    q10_temperature_response,
    arrhenius_temperature_response,
    validate_temperature_range,
    validate_positive_values,
)
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.computation.operators.iphs import advance_reaction_network
from relic.utils.physical_constants import get_physical_constant
from relic.scientific.pedosphere.api_contract import (
    PedosphereContractMixin,
    FluxDirection,
    FluxSpecification,
    FluxType,
    StandardPedosphereOutputs,
)

logger = logging.getLogger(__name__)


class FirstOrderDecomposition(PedosphereContractMixin, BiologicalProcessFunctor):
    """
    First-order kinetic decomposition functor for soil carbon pools.

    Implements environmentally-controlled first-order decomposition with
    multiple carbon pools, transfer matrices, and quality/texture controls.

    References:
    - Coleman & Jenkinson (1996) RothC model structure
    - Sierra et al. (2012) General decomposition framework
    - Lloyd & Taylor (1994) Temperature response

    Mathematical Expression:
    -----------------------
    For each pool i:
    dC_i/dt = Sigma_j(T_ij * k_j * f_env * C_j) - k_i * f_env * C_i + I_i

    Where:
    - C_i: carbon in pool i [g C m^-2]
    - k_i: intrinsic decomposition rate of pool i [s^-1]
    - T_ij: transfer coefficient from pool j to i [dimensionless]
    - f_env: environmental modifier f_T(T) * f_theta(theta) [dimensionless]
    - I_i: external inputs to pool i [g C m^-2 s^-1]

    Respiration flux:
    R_hetero = Sigma_i(k_i * f_env * C_i * (1 - Sigma_j(T_ji)))

    Units:
    - Carbon pools: g C m^-2
    - Decomposition rates: s^-1
    - Respiration flux: g C m^-2 s^-1
    - Temperature: K
    - Soil moisture: m^3 m^-3
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    POOL_NAMES = (
        "metabolic_litter",
        "structural_litter",
        "fast_som",
        "slow_som",
        "passive_som",
    )

    PARAMETER_DESCRIPTIONS = {
        "pedosphere.decomposition.param_25": "Q10 response normalization at the reference temperature [dimensionless]",
        "pedosphere.decomposition.clay_enhancement_coefficient": "Clay enhancement coefficient for passive transfers [dimensionless]",
        "pedosphere.decomposition.max_clay_factor": "Upper bound on the clay transfer multiplier [dimensionless]",
        "pedosphere.decomposition.clay_protection_coefficient": "Clay protection coefficient for SOM decay [dimensionless]",
        "pedosphere.decomposition.k_metabolic": "Metabolic litter intrinsic decay rate [yr-1]",
        "pedosphere.decomposition.k_structural": "Structural litter intrinsic decay rate [yr-1]",
        "pedosphere.decomposition.k_fast_som": "Fast SOM intrinsic decay rate [yr-1]",
        "pedosphere.decomposition.k_slow_som": "Slow SOM intrinsic decay rate [yr-1]",
        "pedosphere.decomposition.k_passive_som": "Passive SOM intrinsic decay rate [yr-1]",
        "pedosphere.decomposition.t_met_fast": "Metabolic litter to fast SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_met_slow": "Metabolic litter to slow SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_str_fast": "Structural litter to fast SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_str_slow": "Structural litter to slow SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_fast_slow": "Fast SOM to slow SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_fast_passive": "Fast SOM to passive SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_slow_fast": "Slow SOM to fast SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.t_slow_passive": "Slow SOM to passive SOM transfer fraction [dimensionless]",
        "pedosphere.decomposition.temp_response_type": "Temperature response selector [dimensionless string]",
        "pedosphere.physical.q10_decomposition": "Shared decomposition Q10 coefficient [dimensionless]",
        "pedosphere.physical.activation_energy_decomposition": "Decomposition activation energy [J mol-1]",
        "pedosphere.decomposition.moisture_response_power": "Moisture response exponent [dimensionless]",
        "pedosphere.decomposition.field_capacity": "Field-capacity water content [m3 m-3]",
        "pedosphere.decomposition.wilting_point": "Wilting-point water content [m3 m-3]",
        "pedosphere.numerical.clamp_min": "Minimum dimensionless response clamp",
        "pedosphere.numerical.clamp_max": "Maximum dimensionless response clamp",
        "pedosphere.physical.temperature_minimum_biological": "Minimum biological temperature [K]",
        "pedosphere.physical.temperature_maximum_biological": "Maximum biological temperature [K]",
        "pedosphere.numerical.temperature_response_min": "Minimum temperature response multiplier [dimensionless]",
        "pedosphere.numerical.temperature_response_max": "Maximum temperature response multiplier [dimensionless]",
        "pedosphere.numerical.unity_multiplier": "Unity response multiplier [dimensionless]",
        "pedosphere.decomposition.lignin_inhibition_coefficient": "Lignin inhibition coefficient [dimensionless]",
        "pedosphere.decomposition.max_lignin_n_ratio": "Maximum lignin to nitrogen ratio [dimensionless]",
    }

    BASE_NAMESPACE = "pedosphere.decomposition"
    PHYSICAL_NAMESPACE = "pedosphere.physical"
    NUMERICAL_NAMESPACE = "pedosphere.numerical"

    def __init__(self, name: str = "first_order_decomposition", config: Optional[Dict] = None):
        """
        Initialize first-order decomposition functor.

        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance.
        config : Dict, optional
            Configuration dictionary containing parameters and settings.
            When omitted, defaults are loaded from the parameter system.
        """
        if config is None:
            config = {}

        # Define comprehensive metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="First-order kinetic decomposition of soil organic matter with multiple pools",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="dC/dt = -k(T,theta) * C with k(T,theta) = k_ref * f_T(T) * f_theta(theta)",
            references=[
                "Coleman, K. & Jenkinson, D.S. (1996). RothC-26.3 model",
                "Parton, W.J. et al. (1987). CENTURY model decomposition",
                "Sierra, C.A. et al. (2012). SoilR decomposition framework",
                "Lloyd, J. & Taylor, J.A. (1994). Temperature dependence"
            ],
            parameters=dict(self.PARAMETER_DESCRIPTIONS),
            input_variables=[
                "metabolic_litter_c", "structural_litter_c", "fast_som_c",      
                "slow_som_c", "passive_som_c", "soil_temperature",
                "soil_moisture", "metabolic_litter_input",
                "structural_litter_input", "lignin_n_ratio", "clay_content"
            ],
            output_variables=[
                "metabolic_litter_c",
                "structural_litter_c",
                "fast_som_c",
                "slow_som_c",
                "passive_som_c",
                "carbon_decomposition_rate",
                "heterotrophic_respiration",
                "diagnostics.first_order_decomposition.metabolic_litter_decomposition_flux",
                "diagnostics.first_order_decomposition.structural_litter_decomposition_flux",
                "diagnostics.first_order_decomposition.fast_som_decomposition_flux",
                "diagnostics.first_order_decomposition.slow_som_decomposition_flux",
                "diagnostics.first_order_decomposition.passive_som_decomposition_flux",
                "diagnostics.first_order_decomposition.metabolic_to_fast_transfer_flux",
                "diagnostics.first_order_decomposition.structural_to_slow_transfer_flux",
                "diagnostics.first_order_decomposition.fast_to_passive_transfer_flux",
                "diagnostics.first_order_decomposition.temperature_modifier",
                "diagnostics.first_order_decomposition.moisture_modifier",
                "diagnostics.first_order_decomposition.environmental_modifier",
                "diagnostics.first_order_decomposition.step_seconds",
            ],
            conservation_laws=["carbon", "mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=[
                "soil_temperature",
                "soil_moisture",
                "metabolic_litter_input",
                "structural_litter_input",
                "carbon_decomposition_rate",
                "heterotrophic_respiration",
            ],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )

        # Initialize BiologicalProcessFunctor with proper parameters
        super().__init__(name=name, config=config, metadata=metadata, device=None)
        # This process uses its own declared response functions. Remove generic
        # bridge helper parameters that are not consumed by this runtime.
        self.metadata.parameters = dict(self.PARAMETER_DESCRIPTIONS)

        self.logger = logger
        self.optional_input_variables = []

        # Physical constants use the dedicated SSOT accessor and are not
        # ordinary decomposition ParameterManager dependencies.
        self.seconds_per_year = float(get_physical_constant("seconds_per_year"))
        self.reference_temperature = float(get_physical_constant("reference_temperature"))
        self.gas_constant = float(get_physical_constant("gas_constant"))
        self.param_25 = self.require_scalar(
            "param_25",
            namespace=self.BASE_NAMESPACE,
            units="dimensionless",
        )

        self.clay_enhancement_coefficient = self.require_scalar(
            "clay_enhancement_coefficient",
            namespace=self.BASE_NAMESPACE,
            units="dimensionless",
        )
        self.max_clay_factor = self.require_scalar(
            "max_clay_factor",
            namespace=self.BASE_NAMESPACE,
            units="dimensionless",
        )
        self.clay_protection_coefficient = self.require_scalar(
            "clay_protection_coefficient",
            namespace=self.BASE_NAMESPACE,
            units="dimensionless",
        )

        # Load the pool-specific rates used by the active carbon equation.
        self.pool_params = self._initialize_pool_parameters()

        # Load transfer matrix parameters
        self.transfer_params = self._initialize_transfer_parameters()

        # Temperature response configuration
        # Reference: Q10 vs Arrhenius response functions
        # Mathematical expression: temperature scaling method selector
        # Parameters: response function type [dimensionless string]
        self.temp_response_type = self.get_parameter_value(
            "temp_response_type",
            namespace=self.BASE_NAMESPACE,
            units="dimensionless",
            required=True,
        )
        # Reference: Lloyd & Taylor (1994) soil respiration
        # Mathematical expression: Q10 = f(T+10C) / f(T)
        # Parameters: temperature sensitivity coefficient [dimensionless]
        self.q10 = self.require_scalar(
            "q10_decomposition",
            namespace=self.PHYSICAL_NAMESPACE,
            units="dimensionless",
        )
        # Reference: Arrhenius kinetics for decomposition
        # Mathematical expression: k = A * exp(-Ea/RT)
        # Parameters: activation energy [J mol-1]
        self.activation_energy = self.require_scalar(
            "activation_energy_decomposition",
            namespace=self.PHYSICAL_NAMESPACE,
            units="J mol-1",
        )

        # Moisture response configuration
        # Reference: Soil water potential effects on decomposition
        # Mathematical expression: f_moisture = ((theta - wp)/(fc - wp))^n
        # Parameters: moisture response power exponent [dimensionless]
        self.moisture_power = self.require_scalar(
            "moisture_response_power",
            namespace=self.BASE_NAMESPACE,
            units="dimensionless",
        )
        # Reference: Soil hydraulic properties
        # Mathematical expression: field capacity moisture content
        # Parameters: volumetric water content at field capacity [m3 m-3]
        self.field_capacity = self.require_scalar(
            "field_capacity",
            namespace=self.BASE_NAMESPACE,
            units="m3 m-3",
        )
        # Reference: Permanent wilting point
        # Mathematical expression: wilting point moisture content
        # Parameters: volumetric water content at wilting point [m3 m-3]
        self.wilting_point = self.require_scalar(
            "wilting_point",
            namespace=self.BASE_NAMESPACE,
            units="m3 m-3",
        )

        # Numerical stability parameters
        # Reference: Numerical bounds for response functions
        # Mathematical expression: minimum clipping value
        # Parameters: minimum allowed value for clamping [dimensionless]
        self.clamp_min = self.require_scalar(
            "clamp_min",
            namespace=self.NUMERICAL_NAMESPACE,
            units="dimensionless",
        )
        # Reference: Numerical bounds for response functions
        # Mathematical expression: maximum clipping value
        # Parameters: maximum allowed value for clamping [dimensionless]
        self.clamp_max = self.require_scalar(
            "clamp_max",
            namespace=self.NUMERICAL_NAMESPACE,
            units="dimensionless",
        )

        # Temperature response bounds
        self.temp_minimum = self.require_scalar(
            "temperature_minimum_biological",
            namespace="pedosphere.physical",
            units="K",
        )
        self.temp_maximum = self.require_scalar(
            "temperature_maximum_biological",
            namespace="pedosphere.physical",
            units="K",
        )
        self.temperature_response_min = self.require_scalar(
            "temperature_response_min",
            namespace=self.NUMERICAL_NAMESPACE,
            units="dimensionless",
        )
        self.temperature_response_max = self.require_scalar(
            "temperature_response_max",
            namespace=self.NUMERICAL_NAMESPACE,
            units="dimensionless",
        )
        self.unity_multiplier = self.require_scalar(
            "unity_multiplier",
            namespace=self.NUMERICAL_NAMESPACE,
            units="dimensionless",
        )

        # Lignin inhibition parameters for structural litter decomposition.
        self.lignin_inhibition_coefficient = self.require_scalar(
            "lignin_inhibition_coefficient",
            namespace=self.BASE_NAMESPACE,
            units="dimensionless",
        )
        self.max_lignin_n_ratio = self.require_scalar(
            "max_lignin_n_ratio",
            namespace=self.BASE_NAMESPACE,
            units="dimensionless",
        )
        self._validate_transfer_parameters()

    def _get_state_tensor(
        self,
        state: StateDict,
        key: str,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return one required StateManager tensor without a fallback path."""
        if key not in state:
            raise KeyError(f"Missing required state input: {key}")
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be torch.Tensor")
        resolved_dtype = dtype
        if resolved_dtype is None:
            resolved_dtype = value.dtype
        return self.to_tensor(value, dtype=resolved_dtype).clone()

    def _get_pool_carbon(self, state: StateDict, pool_name: str) -> torch.Tensor:
        """Return one required finite non-negative carbon pool tensor."""
        key = f"{pool_name}_c"
        pool_carbon = self._get_state_tensor(state, key)
        if not torch.isfinite(pool_carbon).all():
            raise ValueError(f"{key} must be a finite non-negative carbon pool")
        if torch.any(pool_carbon < 0):
            raise ValueError(f"{key} must be a non-negative carbon pool")
        return pool_carbon

    def _initialize_pool_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize pool-specific decomposition parameters.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Pool parameters with rates, transfer coefficients, etc.
        """
        base = self.BASE_NAMESPACE
        return {
            'metabolic_litter': {
                'k_ref': self.require_scalar(
                    'k_metabolic',
                    namespace=base,
                    units='yr-1',
                ) / self.seconds_per_year,
                'clay_protection': False
            },
            'structural_litter': {
                'k_ref': self.require_scalar(
                    'k_structural',
                    namespace=base,
                    units='yr-1',
                ) / self.seconds_per_year,
                'clay_protection': False
            },
            'fast_som': {
                'k_ref': self.require_scalar(
                    'k_fast_som',
                    namespace=base,
                    units='yr-1',
                ) / self.seconds_per_year,
                'clay_protection': True
            },
            'slow_som': {
                'k_ref': self.require_scalar(
                    'k_slow_som',
                    namespace=base,
                    units='yr-1',
                ) / self.seconds_per_year,
                'clay_protection': True
            },
            'passive_som': {
                'k_ref': self.require_scalar(
                    'k_passive_som',
                    namespace=base,
                    units='yr-1',
                ) / self.seconds_per_year,
                'clay_protection': True
            }
        }

    def _initialize_transfer_parameters(self) -> Dict[str, float]:
        """Load carbon transfer coefficients between pools."""
        base = self.BASE_NAMESPACE
        transfers = {
            'metabolic_to_fast': self.require_scalar(
                't_met_fast',
                namespace=base,
                units='dimensionless',
            ),
            'metabolic_to_slow': self.require_scalar(
                't_met_slow',
                namespace=base,
                units='dimensionless',
            ),
            'structural_to_fast': self.require_scalar(
                't_str_fast',
                namespace=base,
                units='dimensionless',
            ),
            'structural_to_slow': self.require_scalar(
                't_str_slow',
                namespace=base,
                units='dimensionless',
            ),
            'fast_to_slow': self.require_scalar(
                't_fast_slow',
                namespace=base,
                units='dimensionless',
            ),
            'fast_to_passive': self.require_scalar(
                't_fast_passive',
                namespace=base,
                units='dimensionless',
            ),
            'slow_to_fast': self.require_scalar(
                't_slow_fast',
                namespace=base,
                units='dimensionless',
            ),
            'slow_to_passive': self.require_scalar(
                't_slow_passive',
                namespace=base,
                units='dimensionless',
            ),
        }

        return transfers

    def _effective_transfer_parameters(
        self,
        clay_content: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return transfer fractions driven by the canonical clay state input."""
        # References: Coleman and Jenkinson (1996); Sierra et al. (2012).
        # Mathematical expression: a_passive(theta_c) = a_passive,0 *
        # min(1 + beta_c * theta_c, a_c,max).
        # Parameter meanings: theta_c is clay_content [1], beta_c is the clay
        # enhancement coefficient [1], and a_c,max bounds the multiplier [1].
        # Units: all inputs and returned transfer fractions are dimensionless.
        # Open-system route: the non-transferred fraction remains the explicit
        # heterotrophic_respiration boundary export.
        clay_factor = torch.minimum(
            self.to_tensor(
                1.0,
                dtype=clay_content.dtype,
            )
            + self.to_tensor(
                self.clay_enhancement_coefficient,
                dtype=clay_content.dtype,
            )
            * clay_content,
            self.to_tensor(
                self.max_clay_factor,
                dtype=clay_content.dtype,
            ),
        )
        effective = {
            key: self.to_tensor(value, dtype=clay_content.dtype)
            for key, value in self.transfer_params.items()
        }
        effective["fast_to_passive"] = (
            effective["fast_to_passive"] * clay_factor
        )
        effective["slow_to_passive"] = (
            effective["slow_to_passive"] * clay_factor
        )
        self._validate_transfer_parameters(effective)
        return effective

    def _validate_transfer_parameters(
        self,
        transfer_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Validate nonnegative compartment transfers and their source sums."""
        # References: Sierra et al. (2012); CASA-CNP model documentation.
        # Mathematical expression: 0 <= T_ji and sum_j(T_ji) <= 1 for each
        # source pool i; the remainder 1 - sum_j(T_ji) is respired as CO2.
        # Parameter meanings: T_ji is the fraction of decomposed carbon from
        # source pool i transferred to receiving pool j [dimensionless].
        # Units: all transfer coefficients and their sums are dimensionless.
        # Open-system route: the non-transferred fraction exits through
        # heterotrophic_respiration; no negative residual is discarded.
        source_transfers = {
            "metabolic_litter": (
                "metabolic_to_fast",
                "metabolic_to_slow",
            ),
            "structural_litter": (
                "structural_to_fast",
                "structural_to_slow",
            ),
            "fast_som": (
                "fast_to_slow",
                "fast_to_passive",
            ),
            "slow_som": (
                "slow_to_fast",
                "slow_to_passive",
            ),
        }
        parameters = (
            self.transfer_params
            if transfer_parameters is None
            else transfer_parameters
        )
        tolerance = float(self.clamp_min)
        for source_pool, transfer_names in source_transfers.items():
            values = [
                self.to_tensor(parameters[name])
                for name in transfer_names
            ]
            if not all(torch.isfinite(value).all() for value in values):
                raise ValueError(
                    f"{source_pool} transfer coefficients must be finite"
                )
            if any(
                torch.any(value < 0.0) or torch.any(value > 1.0)
                for value in values
            ):
                raise ValueError(
                    f"{source_pool} transfer coefficients must be in [0, 1]"
                )
            transfer_sum = sum(values)
            if torch.any(transfer_sum > 1.0 + tolerance):
                raise ValueError(
                    f"{source_pool} transfer coefficient sum must be <= 1"
                )

    def _compute_rate_core(self, state: StateDict) -> FluxDict:
        """
        Compute first-order decomposition rates given current state and environment.
        """
        pools = {
            pool: self._get_pool_carbon(state, pool)
            for pool in self.POOL_NAMES
        }

        soil_temp = self._get_state_tensor(state, 'soil_temperature')
        soil_moisture = self._get_state_tensor(
            state,
            'soil_moisture',
            dtype=soil_temp.dtype,
        )
        lignin_n_ratio = self._get_state_tensor(
            state,
            'lignin_n_ratio',
            dtype=soil_temp.dtype,
        )
        clay_content = self._get_state_tensor(
            state,
            'clay_content',
            dtype=soil_temp.dtype,
        )

        if not torch.isfinite(lignin_n_ratio).all():
            raise ValueError("lignin_n_ratio must be finite")
        if torch.any(lignin_n_ratio < 0) or torch.any(
            lignin_n_ratio > self.max_lignin_n_ratio
        ):
            raise ValueError(
                "lignin_n_ratio must be within the ParameterManager bounds"
            )
        if not torch.isfinite(clay_content).all():
            raise ValueError("clay_content must be finite")
        if torch.any(clay_content < 0) or torch.any(
            clay_content > self.clamp_max
        ):
            raise ValueError("clay_content must be in [0, 1]")

        temp_response = self._calculate_temperature_response(soil_temp)
        moisture_response = self._calculate_moisture_response(soil_moisture)
        env_modifier = temp_response * moisture_response

        decomp_rates: Dict[str, torch.Tensor] = {}

        lignin_factor = self._calculate_lignin_effects(lignin_n_ratio)
        clay_protection = self.to_tensor(self.clay_protection_coefficient, dtype=soil_temp.dtype)
        clay_modifier = 1.0 / (1.0 + clay_protection * clay_content)

        # References: Parton et al. (1987) SSSAJ 51, 1173-1179, Eq. [2]-[4];
        #   Melillo et al. (1982) Ecology 63, 621-626.
        # Mathematical expression: r_i = k_i f_T f_theta g_i C_i with
        #   g_structural = exp(-c_L L), g_metabolic = 1,
        #   g_SOM = 1 / (1 + c_clay f_clay) for clay-protected pools.
        # Parameter meanings: k_i intrinsic decay rate, f_T and f_theta the
        #   temperature and moisture modifiers, c_L the lignin inhibition
        #   coefficient, L the litter lignin driver, c_clay the clay protection
        #   coefficient, f_clay the clay fraction, C_i the pool carbon.
        # Units: r_i g C m-2 s-1, k_i s-1, C_i g C m-2, modifiers dimensionless.
        # Domain/constraints: only the structural litter pool is lignin
        #   inhibited (the metabolic pool is the lignin-free fraction by
        #   construction of the litter partition); 0 < g_i <= 1.
        # Sign convention: r_i >= 0 is the gross carbon leaving pool i.
        for pool_name, pool_carbon in pools.items():
            k_ref = self.pool_params[pool_name]['k_ref']
            k_tensor = self.to_tensor(k_ref, dtype=pool_carbon.dtype)
            if self.pool_params[pool_name].get('clay_protection', False):
                rate = k_tensor * env_modifier * self.to_tensor(clay_modifier, dtype=pool_carbon.dtype) * pool_carbon
            elif pool_name == 'structural_litter':
                rate = k_tensor * env_modifier * lignin_factor * pool_carbon
            else:
                rate = k_tensor * env_modifier * pool_carbon
            decomp_rates[f'{pool_name}_decomp_rate'] = rate

        # References: Coleman & Jenkinson (1996); Parton et al. (1987)
        # Mathematical expression: f_env = f_T * f_theta
        # Parameter meanings: f_T temperature response [-], f_theta moisture response [-]
        decomp_rates["temperature_modifier"] = temp_response
        decomp_rates["moisture_modifier"] = moisture_response
        decomp_rates["environmental_modifier"] = env_modifier
        return decomp_rates

    def required_parameters(self) -> tuple[str, ...]:
        """Return the exact ParameterManager dependency contract."""
        return tuple(self.PARAMETER_DESCRIPTIONS)

    def _calculate_temperature_response(self, temperature: torch.Tensor) -> torch.Tensor:
        """Calculate temperature response modifier."""
        temperature = self.to_tensor(
            temperature,
            dtype=temperature.dtype if isinstance(temperature, torch.Tensor) else None,
        )
        if not torch.isfinite(temperature).all():
            raise ValueError("soil_temperature must be finite")
        try:
            validate_temperature_range(
                temperature,
                min_temp=self.temp_minimum,
                max_temp=self.temp_maximum,
            )
        except ValueError as exc:
            raise ValueError(f"soil_temperature {exc}") from exc

        if self.temp_response_type.lower() == 'q10':
            param_25 = self.to_tensor(self.param_25, dtype=temperature.dtype)
            reference_temp = self.to_tensor(
                self.reference_temperature,
                dtype=temperature.dtype,
            )
            response = q10_temperature_response(
                param_25=param_25,
                temperature=temperature,
                q10=self.to_tensor(self.q10, dtype=temperature.dtype),
                reference_temp=reference_temp
            )
        elif self.temp_response_type.lower() == 'arrhenius':
            param_25 = self.to_tensor(self.param_25, dtype=temperature.dtype)
            reference_temp = self.to_tensor(
                self.reference_temperature,
                dtype=temperature.dtype,
            )
            gas_constant = self.to_tensor(self.gas_constant, dtype=temperature.dtype)
            response = arrhenius_temperature_response(
                param_25=param_25,
                temperature=temperature,
                activation_energy=self.to_tensor(self.activation_energy, dtype=temperature.dtype),
                gas_constant=gas_constant,
                reference_temp=reference_temp
            )
        else:
            raise ValueError(f"Unsupported temperature response type: {self.temp_response_type}")

        response = torch.clamp(
            response,
            min=self.temperature_response_min,
            max=self.temperature_response_max
        )
        return response

    def _calculate_moisture_response(self, soil_moisture: torch.Tensor) -> torch.Tensor:
        """Calculate moisture response modifier."""
        soil_moisture = self.to_tensor(
            soil_moisture,
            dtype=(
                soil_moisture.dtype
                if isinstance(soil_moisture, torch.Tensor)
                else None
            ),
        )
        if not torch.isfinite(soil_moisture).all():
            raise ValueError("soil_moisture must be finite")
        validate_positive_values(soil_moisture, "soil_moisture")
        if torch.any(soil_moisture > self.clamp_max):
            raise ValueError("soil_moisture must be in [0, 1]")

        fc = self.to_tensor(self.field_capacity, dtype=soil_moisture.dtype)
        wp = self.to_tensor(self.wilting_point, dtype=soil_moisture.dtype)
        moisture_factor = (soil_moisture - wp) / torch.clamp(fc - wp, min=self.to_tensor(self.clamp_min, dtype=soil_moisture.dtype))
        response = torch.pow(torch.clamp(moisture_factor, min=self.clamp_min, max=self.clamp_max), self.moisture_power)

        unity = self.to_tensor(self.unity_multiplier, dtype=soil_moisture.dtype)
        response = torch.where(soil_moisture > fc, unity, response)

        return response

    def _calculate_lignin_effects(self, lignin_n_ratio: torch.Tensor) -> torch.Tensor:
        """Calculate lignin inhibition effects on decomposition."""
        lignin_n_ratio = self.to_tensor(
            lignin_n_ratio,
            dtype=(
                lignin_n_ratio.dtype
                if isinstance(lignin_n_ratio, torch.Tensor)
                else None
            ),
        )
        return torch.exp(
            -self.lignin_inhibition_coefficient * lignin_n_ratio
        )

    def forward(self,
        # References: Coleman and Jenkinson (1996); Parton et al. (1987); Sierra et al. (2012);
        #   Lloyd and Taylor (1994).
        # Mathematical expression: R_i = k_i f_T(T) f_theta(theta) C_i;
        #   Delta C_i = (sum_j T_ij R_j - R_i) dt; R_hetero = sum_i R_i (1 - sum_j T_ij).
        # Parameter meanings: k_i reference decay rate of pool i; f_T, f_theta temperature and
        #   moisture modifiers; T_ij transfer fraction from pool j to pool i; R_hetero the
        #   heterotrophic respiration.
        # Units: C_i, Delta C_i [g C m-2]; R_i, R_hetero [g C m-2 s-1]; k_i [s-1];
        #   f_T, f_theta, T_ij [1]; dt [s].
        # Domain/constraints: finite non-negative pools; transfer fractions of a donor sum to at
        #   most one; explicit first-order step over dt.
        # Sign convention: R_i leaves pool i; transfers are positive into the receiving pool;
        #   respiration is a positive loss to the atmosphere.
                state: StateDict,
                time: float,
                dt: torch.Tensor | float,
                **kwargs) -> FluxDict:
        """
        Execute first-order decomposition for one time step.

        Parameters
        ----------
        state : StateDict
            Current system state including carbon pools and environment
        time : float
            Current simulation time [s]
        dt : float
            Time step size [s]
        **kwargs
            Additional environmental variables and forcing data

        Returns
        -------
        FluxDict
            Carbon pool updates and respiration fluxes

        Mathematical Expression
        -----------------------
        For each pool i:
        DeltaC_i = (Sigma_j(T_ij * R_j) - R_i) * dt
        R_hetero = Sigma_i(R_i * (1 - Sigma_j(T_ij)))

        Where R_i = k_i * f_env * C_i
        """
        # References: Coleman and Jenkinson 1996 (RothC); Parton et al. 1987 (CENTURY); Sierra et al. 2012
        # Equation: dC_i/dt = inputs_i - k_i(T,theta) * C_i + sum_j(T_ji * k_j(T,theta) * C_j)
        # Parameters: k_ref [s-1], f_T and f_theta [1], transfer fractions
        # T_ij [1], and mapped metabolic/structural inputs [g C m-2 s-1].
        dt_tensor = self.to_tensor(dt)
        if dt_tensor.numel() != 1 or not torch.isfinite(dt_tensor).all():
            raise ValueError("dt must be one finite positive scalar")
        if float(dt_tensor.detach().item()) <= 0.0:
            raise ValueError("dt must be one finite positive scalar")
        # Extract current carbon pools (g C m^-2)
        pools = {
            pool: self._get_pool_carbon(state, pool)
            for pool in self.POOL_NAMES
        }

        metabolic_input = self._get_state_tensor(
            state,
            'metabolic_litter_input',
            dtype=pools['metabolic_litter'].dtype,
        )
        structural_input = self._get_state_tensor(
            state,
            'structural_litter_input',
            dtype=pools['structural_litter'].dtype,
        )
        for key, value in (
            ('metabolic_litter_input', metabolic_input),
            ('structural_litter_input', structural_input),
        ):
            if not torch.isfinite(value).all() or torch.any(value < 0):
                raise ValueError(f"{key} must be finite and non-negative")

        # Evaluate the decomposition law once for the current StateManager state.
        decomp_rate_dict = self._compute_rate_core(state)
        clay_content = self._get_state_tensor(
            state,
            "clay_content",
            dtype=pools["fast_som"].dtype,
        )
        effective_transfer_parameters = self._effective_transfer_parameters(
            clay_content
        )

        # Extract decomposition fluxes from compute_rate output
        decomp_fluxes = {
            'metabolic_litter': decomp_rate_dict['metabolic_litter_decomp_rate'],
            'structural_litter': decomp_rate_dict['structural_litter_decomp_rate'],
            'fast_som': decomp_rate_dict['fast_som_decomp_rate'],
            'slow_som': decomp_rate_dict['slow_som_decomp_rate'],
            'passive_som': decomp_rate_dict['passive_som_decomp_rate']
        }

        # References: Coleman & Jenkinson (1996); Parton et al. (1987)
        # Mathematical expression: f_env = f_T * f_theta
        # Parameter meanings: f_T temperature response [-], f_theta moisture response [-]
        temp_response = decomp_rate_dict["temperature_modifier"]
        moisture_response = decomp_rate_dict["moisture_modifier"]
        env_modifier = decomp_rate_dict["environmental_modifier"]

        # Calculate transfers between pools
        transfers = {}

        # Metabolic litter transfers
        met_flux = decomp_fluxes['metabolic_litter']
        transfers['metabolic_to_fast'] = met_flux * effective_transfer_parameters['metabolic_to_fast']
        transfers['metabolic_to_slow'] = met_flux * effective_transfer_parameters['metabolic_to_slow']

        # Structural litter transfers
        str_flux = decomp_fluxes['structural_litter']
        transfers['structural_to_fast'] = str_flux * effective_transfer_parameters['structural_to_fast']
        transfers['structural_to_slow'] = str_flux * effective_transfer_parameters['structural_to_slow']

        # Fast SOM transfers
        fast_flux = decomp_fluxes['fast_som']
        transfers['fast_to_slow'] = fast_flux * effective_transfer_parameters['fast_to_slow']
        transfers['fast_to_passive'] = fast_flux * effective_transfer_parameters['fast_to_passive']

        # Slow SOM transfers
        slow_flux = decomp_fluxes['slow_som']
        transfers['slow_to_fast'] = slow_flux * effective_transfer_parameters['slow_to_fast']
        transfers['slow_to_passive'] = slow_flux * effective_transfer_parameters['slow_to_passive']

        # Calculate net changes in each pool (g C m^-2 s^-1)
        delta_metabolic = (
            metabolic_input
            - decomp_fluxes['metabolic_litter']
        )

        delta_structural = (
            structural_input
            - decomp_fluxes['structural_litter']
        )

        delta_fast = (
            transfers['metabolic_to_fast'] + transfers['structural_to_fast'] + transfers['slow_to_fast']
            - decomp_fluxes['fast_som']
        )

        delta_slow = (
            transfers['metabolic_to_slow'] + transfers['structural_to_slow'] + transfers['fast_to_slow']
            - decomp_fluxes['slow_som']
        )

        delta_passive = (
            transfers['fast_to_passive'] + transfers['slow_to_passive']
            - decomp_fluxes['passive_som']
        )

        # References: Parton et al. (1987); Sierra et al. (2012).
        # Mathematical expression: R_hetero = I_met + I_str - sum_i(dC_i/dt),
        # equivalently sum_i(D_i) - sum_ji(T_ji D_i).
        # Parameter meanings: I_met and I_str are boundary inputs and dC_i/dt
        # are the already assembled pool tendencies, all in g C m-2 s-1.
        # Units: R_hetero is g C m-2 s-1.
        # Open-system route: the exact non-transferred remainder is exported to
        # the atmosphere; using the assembled tendencies preserves the discrete
        # stock ledger under the current tensor dtype.
        total_pool_tendency = torch.stack(
            [
                delta_metabolic,
                delta_structural,
                delta_fast,
                delta_slow,
                delta_passive,
            ]
        ).sum(dim=0)
        hetero_respiration = (
            metabolic_input + structural_input - total_pool_tendency
        )

        if not torch.isfinite(hetero_respiration).all():
            raise ValueError("heterotrophic_respiration must be finite")
        if torch.any(hetero_respiration < 0):
            raise ValueError(
                "heterotrophic_respiration became negative; check transfer fractions"
            )

        # References: Coleman & Jenkinson (1996); Parton et al. (1987); Sierra et al. (2012).
        # Mathematical expression: x_dot = N_h r_h and delta_x = x_dot * dt.
        # Parameter meanings: N_h maps the five net pool tendencies and the
        # heterotrophic carbon export; r_h contains ParameterManager-backed rates.
        # Units: pool tendencies and respiration use g C m-2 s-1; dt uses s.
        # Open-system route: litter enters through the boundary input port and
        # heterotrophic respiration leaves through the atmosphere-facing port.
        decomposition_network = advance_reaction_network(
            name="first_order_decomposition_pool_network",
            species=(
                "metabolic_litter_c",
                "structural_litter_c",
                "fast_som_c",
                "slow_som_c",
                "passive_som_c",
                "heterotrophic_respiration",
            ),
            reactions=(
                "metabolic_pool_tendency",
                "structural_pool_tendency",
                "fast_som_tendency",
                "slow_som_tendency",
                "passive_som_tendency",
                "heterotrophic_respiration",
            ),
            stoichiometric_N=torch.eye(
                6,
                device=hetero_respiration.device,
                dtype=hetero_respiration.dtype,
            ),
            rates=(
                delta_metabolic,
                delta_structural,
                delta_fast,
                delta_slow,
                delta_passive,
                hetero_respiration,
            ),
            dt=dt,
            **self._iphs_reaction_conservation_kwargs(open_system_route="port"),
        )
        delta_metabolic_step = decomposition_network.state_delta[
            decomposition_network.species.index("metabolic_litter_c")
        ]
        delta_structural_step = decomposition_network.state_delta[
            decomposition_network.species.index("structural_litter_c")
        ]
        delta_fast_step = decomposition_network.state_delta[
            decomposition_network.species.index("fast_som_c")
        ]
        delta_slow_step = decomposition_network.state_delta[
            decomposition_network.species.index("slow_som_c")
        ]
        delta_passive_step = decomposition_network.state_delta[
            decomposition_network.species.index("passive_som_c")
        ]
        hetero_respiration = decomposition_network.species_rate("heterotrophic_respiration")

        # References: Sierra et al. (2012); Manzoni and Porporato (2009).
        # Mathematical expression: C_i^(n+1) = C_i^n + Delta C_i >= 0.
        # Parameter meanings: C_i is the current carbon pool [g C m-2] and
        # Delta C_i is the once-integrated reaction increment [g C m-2].
        # Units: every proposed pool state uses g C m-2.
        # Open-system route: litter input and respiration export remain explicit;
        # invalid steps are rejected without clipping or internal subcycling.
        proposed_pool_states = {
            "metabolic_litter_c": pools["metabolic_litter"] + delta_metabolic_step,
            "structural_litter_c": pools["structural_litter"] + delta_structural_step,
            "fast_som_c": pools["fast_som"] + delta_fast_step,
            "slow_som_c": pools["slow_som"] + delta_slow_step,
            "passive_som_c": pools["passive_som"] + delta_passive_step,
        }
        invalid_pool_states = [
            key
            for key, value in proposed_pool_states.items()
            if not torch.isfinite(value).all() or torch.any(value < 0)
        ]
        if invalid_pool_states:
            invalid_names = ", ".join(invalid_pool_states)
            raise ValueError(
                "FirstOrderDecomposition proposed a negative or non-finite "
                f"carbon pool state for {invalid_names}; configure existing "
                "simulator-managed substeps through FunctorMetadata.substep_count"
            )

        # Prepare updates dictionary for conservation check
        updates = {
            'metabolic_litter_c': delta_metabolic_step,
            'structural_litter_c': delta_structural_step,
            'fast_som_c': delta_fast_step,
            'slow_som_c': delta_slow_step,
            'passive_som_c': delta_passive_step,
            'heterotrophic_respiration': hetero_respiration * dt,
            # External litter input is a source term for this open system.
            # Use negative sign so local conservation closure follows:
            # sum(pool_deltas) + respiration_export - litter_input = 0.
            'litter_input_flux': -(metabolic_input + structural_input) * dt,
        }

        # References: Parton et al. (1987); Coleman and Jenkinson (1996).
        # Mathematical expression: sum_i(Delta C_i) + R_hetero*dt - I_litter*dt = 0.
        # Parameter meanings: Delta C_i are pool increments [g C m-2],
        # R_hetero is the atmospheric carbon export [g C m-2 s-1],
        # I_litter is the boundary input [g C m-2 s-1], and dt is time [s].
        # Units: every term in the ledger is g C m-2.
        # Open-system route: litter enters through a boundary port and respiration
        # leaves through the atmosphere-facing carbon port.
        self._enforce_forward_conservation(state, updates)

        total_decomposition_rate = (
            decomp_fluxes['metabolic_litter']
            + decomp_fluxes['structural_litter']
            + decomp_fluxes['fast_som']
            + decomp_fluxes['slow_som']
            + decomp_fluxes['passive_som']
        )

        # Return explicit state updates, canonical fluxes, and diagnostics.
        return {
            "metabolic_litter_c": UpdateItem(
                value=delta_metabolic_step,
                mode=UpdateMode.ACCUMULATE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=True,
                element="C",
                source_functor=self.name,
            ),
            "structural_litter_c": UpdateItem(
                value=delta_structural_step,
                mode=UpdateMode.ACCUMULATE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=True,
                element="C",
                source_functor=self.name,
            ),
            "fast_som_c": UpdateItem(
                value=delta_fast_step,
                mode=UpdateMode.ACCUMULATE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=True,
                element="C",
                source_functor=self.name,
            ),
            "slow_som_c": UpdateItem(
                value=delta_slow_step,
                mode=UpdateMode.ACCUMULATE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=True,
                element="C",
                source_functor=self.name,
            ),
            "passive_som_c": UpdateItem(
                value=delta_passive_step,
                mode=UpdateMode.ACCUMULATE,
                unit="g C m-2",
                basis="per_step",
                per_timestep=True,
                element="C",
                source_functor=self.name,
            ),
            "carbon_decomposition_rate": UpdateItem(
                value=total_decomposition_rate,
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "heterotrophic_respiration": UpdateItem(
                value=hetero_respiration,
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                element="C",
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.metabolic_litter_decomposition_flux": UpdateItem(
                value=decomp_fluxes["metabolic_litter"],
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.structural_litter_decomposition_flux": UpdateItem(
                value=decomp_fluxes["structural_litter"],
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.fast_som_decomposition_flux": UpdateItem(
                value=decomp_fluxes["fast_som"],
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.slow_som_decomposition_flux": UpdateItem(
                value=decomp_fluxes["slow_som"],
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.passive_som_decomposition_flux": UpdateItem(
                value=decomp_fluxes["passive_som"],
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.metabolic_to_fast_transfer_flux": UpdateItem(
                value=transfers["metabolic_to_fast"],
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.structural_to_slow_transfer_flux": UpdateItem(
                value=transfers["structural_to_slow"],
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.fast_to_passive_transfer_flux": UpdateItem(
                value=transfers["fast_to_passive"],
                mode=UpdateMode.REPLACE,
                unit="g C m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.temperature_modifier": UpdateItem(
                value=temp_response,
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.moisture_modifier": UpdateItem(
                value=moisture_response,
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.first_order_decomposition.environmental_modifier": UpdateItem(
                value=env_modifier,
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
            # Declared step length so the public ledger can integrate the rates.
            "diagnostics.first_order_decomposition.step_seconds": UpdateItem(
                value=dt_tensor.detach().to(dtype=delta_metabolic_step.dtype).reshape(()),
                mode=UpdateMode.REPLACE,
                unit="s",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            ),
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
        dt: Optional[torch.Tensor | float] = None,
    ) -> Dict[str, float]:
        """Return the carbon ledger residual computed from supplied updates."""
        required_outputs = {
            "metabolic_litter_c",
            "structural_litter_c",
            "fast_som_c",
            "slow_som_c",
            "passive_som_c",
            "heterotrophic_respiration",
        }
        if not required_outputs.issubset(output_updates):
            return super().check_conservation_laws(input_state, output_updates)

        def _payload(key: str) -> torch.Tensor:
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            return self.to_tensor(
                value,
                dtype=value.dtype if isinstance(value, torch.Tensor) else None,
            )

        pool_delta = torch.stack(
            [
                _payload("metabolic_litter_c"),
                _payload("structural_litter_c"),
                _payload("fast_som_c"),
                _payload("slow_som_c"),
                _payload("passive_som_c"),
            ]
        ).sum(dim=0)

        if "litter_input_flux" in output_updates:
            residual = (
                pool_delta
                + _payload("heterotrophic_respiration")
                + _payload("litter_input_flux")
            )
        else:
            if dt is not None:
                dt_value = self.to_tensor(dt)
            else:
                step_key = "diagnostics.first_order_decomposition.step_seconds"
                if step_key not in output_updates:
                    raise ValueError("dt is required for the public carbon-rate ledger")
                step = output_updates[step_key]
                dt_value = self.to_tensor(step.value if isinstance(step, UpdateItem) else step)
            if dt_value.numel() != 1 or not torch.isfinite(dt_value).all():
                raise ValueError("dt is required for the public carbon-rate ledger")
            if float(dt_value.detach().item()) <= 0.0:
                raise ValueError("dt is required for the public carbon-rate ledger")
            litter_input_fluxes = {
                key: self._get_state_tensor(
                    input_state,
                    key,
                    dtype=pool_delta.dtype,
                )
                for key in (
                    "metabolic_litter_input",
                    "structural_litter_input",
                )
            }
            for key, value in litter_input_fluxes.items():
                if not torch.isfinite(value).all() or torch.any(value < 0):
                    raise ValueError(f"{key} must be finite and non-negative")
            litter_input = sum(litter_input_fluxes.values())
            dt_tensor = dt_value.to(
                device=pool_delta.device,
                dtype=pool_delta.dtype,
            )
            residual = (
                pool_delta
                + _payload("heterotrophic_respiration") * dt_tensor
                - litter_input * dt_tensor
            )

        magnitude = float(torch.max(torch.abs(residual)).item())
        return {
            law: magnitude
            for law in self.metadata.conservation_laws
        }

    def get_output_specifications(self) -> Dict[str, FluxSpecification]:
        return {
            "heterotrophic_respiration": StandardPedosphereOutputs.HETEROTROPHIC_RESPIRATION,
            "carbon_decomposition_rate": FluxSpecification(
                name="carbon_decomposition_rate",
                units="g C m-2 s-1",
                direction=FluxDirection.INTERNAL,
                flux_type=FluxType.TRANSFORMATION,
                description="Gross carbon decomposition rate across all active pools",
                is_rate=True,
            ),
        }

    def get_pool_turnover_times(self) -> Dict[str, float]:
        """
        Calculate mean residence times for each carbon pool.

        Returns
        -------
        Dict[str, float]
            Mean residence times in years

        Mathematical Expression
        -----------------------
        tau = 1/k_ref / seconds_per_year [years]
        where k_ref is in s^-1
        """
        # Reference: Time unit conversion
        # Mathematical expression: seconds per Julian year
        # Parameters: conversion factor [s yr-1]
        turnover_times = {}

        for pool_name, params in self.pool_params.items():
            k_ref = params['k_ref']
            turnover_time = 1.0 / k_ref / self.seconds_per_year
            turnover_times[f"{pool_name}_turnover_years"] = turnover_time

        return turnover_times

# Export main class
__all__ = ['FirstOrderDecomposition']

