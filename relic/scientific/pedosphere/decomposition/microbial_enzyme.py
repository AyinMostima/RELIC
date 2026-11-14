"""
Microbial-Enzyme Decomposition Model for RELIC Soil Carbon

This module implements microbial-explicit decomposition using Michaelis-Menten 
enzyme kinetics. It provides an alternative to first-order kinetics by explicitly
representing microbial biomass and enzyme activity as drivers of decomposition.

Scientific Basis:
-----------------
The model follows the microbial-explicit framework established in MIMICS, CORPSE,
and other trait-based soil carbon models. It uses reverse Michaelis-Menten kinetics
where microbes are the enzyme and substrate availability limits decomposition rates.

References:
- Wieder, W.R. et al. (2013). The MIMICS microbial-mineral carbon stabilization model.
  Geoscientific Model Development, 6, 1479-1480.
- Sulman, B.N. et al. (2014). Microbe-driven turnover offsets mineral-mediated storage 
  of soil carbon under elevated CO2. Nature Climate Change, 4, 1099-1102.
- Allison, S.D. et al. (2010). Microbial abundance and composition influence litter
  decomposition response to environmental change. Ecology, 91, 714-725.
- Schimel, J.P. & Weintraub, M.N. (2003). The implications of exoenzyme activity 
  on microbial carbon and nitrogen limitation in soil. Soil Biology & Biochemistry, 35, 549-563.

Mathematical Expression:
-----------------------
Reverse Michaelis-Menten equation:
V = (MIC * Vmax(T,theta) * [S]) / (Km(T) + MIC)

Where:
- V: decomposition rate [g C m^-2 s^-1]
- MIC: microbial biomass [g C m^-2]  
- Vmax(T,theta): maximum specific activity [s^-1]
- [S]: substrate concentration [g C m^-2]
- Km(T): half-saturation constant [g C m^-2]

Temperature dependencies:
Vmax(T) = Vmax_ref * exp(Ea_vmax / (R * T_ref)) * exp(-Ea_vmax / (R * T))
Km(T) = Km_ref * exp(Ea_km / (R * T_ref)) * exp(-Ea_km / (R * T))
"""

from typing import Dict, Optional, Tuple, Union
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
    arrhenius_temperature_response,
    validate_temperature_range,
    validate_positive_values,
)
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.computation.operators.iphs import advance_reaction_network
from relic.scientific.pedosphere.api_contract import (
    FluxSpecification,
    PedosphereContractMixin,
    StandardPedosphereOutputs,
)
from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)


class MicrobialEnzymeDecomposition(PedosphereContractMixin, BiologicalProcessFunctor):
    """
    Microbial-enzyme decomposition functor using Michaelis-Menten kinetics.
    
    Implements microbial-explicit decomposition where enzyme activity and
    microbial biomass control substrate processing rates.
    
    References:
    - Wieder et al. (2013) MIMICS framework
    - Allison et al. (2010) Microbial trait-based decomposition
    - Schimel & Weintraub (2003) Enzyme activity implications
    
    Mathematical Expression:
    -----------------------
    Reverse Michaelis-Menten kinetics:
    V = (MIC * Vmax(T,theta) * [S]) / (Km(T) + MIC)
    
    Microbial growth:
    dMIC/dt = CUE * V - tau_mic * MIC
    
    Where:
    - V: substrate consumption rate [g C m^-2 s^-1]
    - MIC: microbial biomass [g C m^-2]
    - Vmax: maximum specific activity [s^-1]
    - [S]: substrate concentration [g C m^-2]
    - Km: half-saturation constant [g C m^-2]
    - CUE: carbon use efficiency [dimensionless]
    - tau_mic: microbial turnover rate [s^-1]
    
    Temperature responses:
    Vmax(T) = Vmax_ref * exp(Ea_v/R * (1/T_ref - 1/T))
    Km(T) = Km_ref * exp(Ea_k/R * (1/T_ref - 1/T))
    
    Units:
    - Substrate pools: g C m^-2
    - Microbial biomass: g C m^-2
    - Decomposition rates: g C m^-2 s^-1
    - Vmax: s^-1
    - Km: g C m^-2
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    PARAMETER_DESCRIPTIONS = {
        "pedosphere.decomposition.microbial_enzyme.ea_vmax": "Vmax activation energy [J mol-1]",
        "pedosphere.decomposition.microbial_enzyme.ea_km": "Km activation energy [J mol-1]",
        "pedosphere.decomposition.microbial_enzyme.moisture_optimum": "Optimal volumetric water content [m3 m-3]",
        "pedosphere.decomposition.microbial_enzyme.moisture_power": "Moisture response shape exponent [dimensionless]",
        "pedosphere.decomposition.microbial_enzyme.vmax_clamp_min_factor": "Minimum Vmax response multiplier [dimensionless]",
        "pedosphere.decomposition.microbial_enzyme.vmax_clamp_max_factor": "Maximum Vmax response multiplier [dimensionless]",
        "pedosphere.decomposition.microbial_enzyme.km_clamp_min_factor": "Minimum Km response multiplier [dimensionless]",
        "pedosphere.decomposition.microbial_enzyme.km_clamp_max_factor": "Maximum Km response multiplier [dimensionless]",
        "pedosphere.decomposition.microbial_enzyme.kinetics.metabolic.vmax_ref": "Metabolic litter reference Vmax [s-1]",
        "pedosphere.decomposition.microbial_enzyme.kinetics.metabolic.km_ref": "Metabolic litter reference Km [g C m-2]",
        "pedosphere.decomposition.microbial_enzyme.kinetics.structural.vmax_ref": "Structural litter reference Vmax [s-1]",
        "pedosphere.decomposition.microbial_enzyme.kinetics.structural.km_ref": "Structural litter reference Km [g C m-2]",
        "pedosphere.decomposition.microbial_enzyme.kinetics.som.vmax_ref": "Available SOM reference Vmax [s-1]",
        "pedosphere.decomposition.microbial_enzyme.kinetics.som.km_ref": "Available SOM reference Km [g C m-2]",
        "pedosphere.decomposition.microbial_enzyme.microbial.carbon_use_efficiency": "Microbial carbon use efficiency [dimensionless]",
        "pedosphere.decomposition.microbial_enzyme.microbial.turnover_rate": "Microbial turnover rate [s-1]",
        "pedosphere.decomposition.microbial_enzyme.microbial.max_biomass": "Maximum supported microbial biomass [g C m-2]",
        "pedosphere.decomposition.microbial_enzyme.microbial.min_biomass": "Minimum supported microbial biomass [g C m-2]",
        "pedosphere.decomposition.microbial_enzyme.microbial.maintenance_rate": "Microbial maintenance rate [s-1]",
        "pedosphere.decomposition.temp_factor": "Microbial maintenance Q10 base [dimensionless]",
        "pedosphere.numerical.epsilon_concentration": "Positive denominator tolerance [g C m-2]",
        "pedosphere.numerical.moisture_response_clamp_min": "Minimum moisture response [dimensionless]",
        "pedosphere.numerical.clamp_max": "Maximum moisture response [dimensionless]",
        "pedosphere.numerical.temp_response_clamp_min": "Minimum maintenance temperature response [dimensionless]",
        "pedosphere.numerical.temp_response_clamp_max": "Maximum maintenance temperature response [dimensionless]",
    }
    
    def __init__(self, name: str = "microbial_enzyme_decomposition", config: Optional[Dict] = None):
        """
        Initialize microbial-enzyme decomposition functor.
        
        Parameters:
        -----------
        name : str
            Unique identifier for this functor instance
        config : Dict
            Configuration dictionary containing parameters and settings
        """
        if config is None:
            config = {}

        # Define comprehensive metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Microbial-explicit decomposition using Michaelis-Menten enzyme kinetics",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="V = (MIC * Vmax(T,theta) * [S]) / (Km(T) + MIC)",
            references=[
                "Allison, S.D. et al. (2010). Microbial abundance and composition influence",
                "Sulman, B.N. et al. (2014). Microbe-driven turnover in elevated CO2"
            ],
            parameters=dict(self.PARAMETER_DESCRIPTIONS),
            input_variables=[
                "metabolic_litter_c", "structural_litter_c", "microbial_biomass_c",
                "soil_temperature", "soil_moisture", "available_som_c"
            ],
            output_variables=[
                "metabolic_litter_c",
                "structural_litter_c",
                "available_som_c",
                "microbial_biomass_c",
                "heterotrophic_respiration",
                "diagnostics.microbial_enzyme.microbial_growth_rate",
                "diagnostics.microbial_enzyme.microbial_death_rate",
                "diagnostics.microbial_enzyme.microbial_respiration",
                "diagnostics.microbial_enzyme.growth_respiration",
                "diagnostics.microbial_enzyme.maintenance_respiration",
                "diagnostics.microbial_enzyme.metabolic_consumption",
                "diagnostics.microbial_enzyme.structural_consumption",
                "diagnostics.microbial_enzyme.som_consumption",
                "diagnostics.microbial_enzyme.metabolic_litter.vmax",
                "diagnostics.microbial_enzyme.metabolic_litter.km",
                "diagnostics.microbial_enzyme.metabolic_litter.activity",
                "diagnostics.microbial_enzyme.structural_litter.vmax",
                "diagnostics.microbial_enzyme.structural_litter.km",
                "diagnostics.microbial_enzyme.structural_litter.activity",
                "diagnostics.microbial_enzyme.som.vmax",
                "diagnostics.microbial_enzyme.som.km",
                "diagnostics.microbial_enzyme.som.activity",
                "diagnostics.microbial_enzyme.moisture_response",
                "diagnostics.microbial_enzyme.temperature_celsius",
                "diagnostics.microbial_enzyme.microbial_biomass_content",
                "diagnostics.microbial_enzyme.total_substrate_consumed",
                "diagnostics.microbial_enzyme.carbon_use_efficiency_realized",
                "diagnostics.microbial_enzyme.carbon_balance_residual",
                "diagnostics.microbial_enzyme.step_seconds",
            ],
            conservation_laws=["carbon", "mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["soil_temperature", "soil_moisture", "substrate_availability"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        
        # Initialize BiologicalProcessFunctor with proper parameters
        super().__init__(name=name, config=config, metadata=metadata, device=None)
        
        # Add FunctorBase consistency attributes
        self.name = name
        self._config_raw = dict(config) if config else {}
        self.metadata = metadata
        self.logger = logger
        self._execution_count = 0
        self.optional_input_variables = []
        
        # Namespaces for parameter organization
        self._base_namespace = "pedosphere.decomposition.microbial_enzyme"
        self._kinetics_namespace = f"{self._base_namespace}.kinetics"
        self._microbial_namespace = f"{self._base_namespace}.microbial"
        self._numerical_namespace = "pedosphere.numerical"
        self._decomposition_namespace = "pedosphere.decomposition"

        # Load kinetic and microbial parameters from ParameterManager
        self.kinetic_params = self._initialize_kinetic_parameters()
        self.microbial_params = self._initialize_microbial_parameters()

        # Temperature response configuration (activation energies in J mol^-1)
        self.ea_vmax = self._get_scalar(self._base_namespace, "ea_vmax")
        self.ea_km = self._get_scalar(self._base_namespace, "ea_km")

        # Moisture response configuration
        self.moisture_optimum = self._get_scalar(self._base_namespace, "moisture_optimum")
        self.moisture_power = self._get_scalar(self._base_namespace, "moisture_power")
        self.vmax_clamp_min_factor = self._get_scalar(self._base_namespace, "vmax_clamp_min_factor")
        self.vmax_clamp_max_factor = self._get_scalar(self._base_namespace, "vmax_clamp_max_factor")
        self.km_clamp_min_factor = self._get_scalar(self._base_namespace, "km_clamp_min_factor")
        self.km_clamp_max_factor = self._get_scalar(self._base_namespace, "km_clamp_max_factor")

        # Numerical safeguards reused across calculations
        self.epsilon_concentration = self._get_scalar(self._numerical_namespace, "epsilon_concentration")
        self.moisture_clamp_min = self._get_scalar(self._numerical_namespace, "moisture_response_clamp_min")
        self.moisture_clamp_max = self._get_scalar(self._numerical_namespace, "clamp_max")
        self.temp_response_clamp_min = self._get_scalar(self._numerical_namespace, "temp_response_clamp_min")
        self.temp_response_clamp_max = self._get_scalar(self._numerical_namespace, "temp_response_clamp_max")
        self.celsius_offset = float(get_physical_constant("celsius_to_kelvin"))
        self.q10_temperature_interval = float(
            get_physical_constant("relic.physical.constants.temperature.q10_interval")
        )
        self.reference_temperature = float(get_physical_constant("reference_temperature"))
        self.gas_constant = float(get_physical_constant("gas_constant"))
        self.temp_factor_base = self._get_scalar(self._decomposition_namespace, "temp_factor")

    @staticmethod
    def _scalar_from_value(value: Union[float, torch.Tensor]) -> float:
        """Convert numeric values to Python float without tensor.item/cpu APIs."""
        if isinstance(value, torch.Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    def _get_scalar(self,
                    namespace: str,
                    name: str) -> float:
        """Retrieve a scalar parameter with full validation."""
        value = self.get_parameter_value(
            name=name,
            namespace=namespace,
            source_order=("manager",),
        )
        if value is None:
            full_key = f"{namespace}.{name}" if namespace else name
            raise KeyError(f"Missing required parameter '{full_key}'")
        if isinstance(value, torch.Tensor):
            return self._scalar_from_value(value)
        return self._scalar_from_value(value)

    def _to_tensor(self,
                   value: Union[float, torch.Tensor],
                   dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:
        """Convert a value to a tensor on the functor device."""
        if isinstance(value, torch.Tensor):
            tensor = value
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            if self.device is not None and tensor.device != self.device:
                tensor = tensor.to(self.device)
            return tensor
        return torch.tensor(value, dtype=dtype, device=self.device)

    def _require_state_tensor(
        self,
        state: StateDict,
        key: str,
        *,
        dtype: Optional[torch.dtype] = None,
        non_negative: bool = False,
    ) -> torch.Tensor:
        """Return one required finite tensor without aliases or defaults."""
        if key not in state:
            raise KeyError(f"Missing required state variable '{key}'")
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        tensor = value.to(device=self.device)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key} must contain only finite values")
        if non_negative and torch.any(tensor < 0.0):
            raise ValueError(f"{key} must contain only non-negative values")
        return tensor

    def _accumulate_carbon(self, value: torch.Tensor) -> UpdateItem:
        """Build one per-step carbon stock increment."""
        return UpdateItem(
            value=value,
            mode=UpdateMode.ACCUMULATE,
            unit="g C m-2",
            basis="per_step",
            per_timestep=True,
            element="C",
            source_functor=self.name,
        )

    def _replace(
        self,
        value: torch.Tensor,
        *,
        unit: str,
        basis: str,
        element: Optional[str] = None,
    ) -> UpdateItem:
        """Build one explicit instantaneous or rate observation."""
        return UpdateItem(
            value=value,
            mode=UpdateMode.REPLACE,
            unit=unit,
            basis=basis,
            per_timestep=False,
            element=element,
            source_functor=self.name,
        )

    def _validate_dt(self, dt: torch.Tensor | float, *, dtype: torch.dtype) -> torch.Tensor:
        """Return a finite positive scalar time step on the functor device."""
        dt_tensor = self._to_tensor(dt, dtype=dtype)
        if dt_tensor.numel() != 1 or not torch.isfinite(dt_tensor).all():
            raise ValueError("dt must be one finite positive scalar")
        if float(dt_tensor.detach().reshape(-1)[0].tolist()) <= 0.0:
            raise ValueError("dt must be one finite positive scalar")
        return dt_tensor

    def _initialize_kinetic_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize substrate-specific kinetic parameters.
        
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Kinetic parameters for each substrate type
        """
        # Default parameters based on MIMICS and literature values
        # Vmax values in s^-1, Km values in g C m^-2
        
        metabolic_ns = f"{self._kinetics_namespace}.metabolic"
        structural_ns = f"{self._kinetics_namespace}.structural"
        som_ns = f"{self._kinetics_namespace}.som"

        return {
            'metabolic_litter': {
                'vmax_ref': self._get_scalar(metabolic_ns, 'vmax_ref'),
                'km_ref': self._get_scalar(metabolic_ns, 'km_ref'),
                'enzyme_type': 'cellulase',
                'substrate_quality': 'high'
            },
            'structural_litter': {
                'vmax_ref': self._get_scalar(structural_ns, 'vmax_ref'),
                'km_ref': self._get_scalar(structural_ns, 'km_ref'),
                'enzyme_type': 'ligninase',
                'substrate_quality': 'low'
            },
            'som': {
                'vmax_ref': self._get_scalar(som_ns, 'vmax_ref'),
                'km_ref': self._get_scalar(som_ns, 'km_ref'),
                'enzyme_type': 'mixed',
                'substrate_quality': 'medium'
            }
        }
        
    def _initialize_microbial_parameters(self) -> Dict[str, float]:
        """
        Initialize microbial community parameters.
        
        Returns:
        --------
        Dict[str, float]
            Microbial parameters
        """
        namespace = self._microbial_namespace
        return {
            'carbon_use_efficiency': self._get_scalar(namespace, 'carbon_use_efficiency'),
            'turnover_rate': self._get_scalar(namespace, 'turnover_rate'),
            'max_biomass': self._get_scalar(namespace, 'max_biomass'),
            'min_biomass': self._get_scalar(namespace, 'min_biomass'),
            'maintenance_rate': self._get_scalar(namespace, 'maintenance_rate'),
        }
        
    def get_output_specifications(self) -> Dict[str, FluxSpecification]:
        """Return the canonical atmosphere-facing carbon flux contract."""
        return {
            "heterotrophic_respiration": (
                StandardPedosphereOutputs.HETEROTROPHIC_RESPIRATION
            )
        }
    
    def _calculate_vmax_temperature_response(self,
                                           substrate_type: str,
                                           temperature: torch.Tensor) -> torch.Tensor:
        """
        Calculate temperature response for maximum enzyme activity.
        
        Parameters:
        -----------
        substrate_type : str
            Type of substrate (metabolic_litter, structural_litter, som)
        temperature : torch.Tensor
            Soil temperature [K]
            
        Returns:
        --------
        torch.Tensor
            Temperature-modified Vmax [s^-1]
            
        Mathematical Expression:
        -----------------------
        Vmax(T) = Vmax_ref * exp(Ea_vmax/R * (1/T_ref - 1/T))
        """
        validate_temperature_range(temperature)
        
        vmax_ref = self._to_tensor(self.kinetic_params[substrate_type]['vmax_ref'], dtype=temperature.dtype)
        activation_energy = self._to_tensor(self.ea_vmax, dtype=temperature.dtype)

        vmax_temp = arrhenius_temperature_response(
            param_25=vmax_ref,
            temperature=temperature,
            activation_energy=activation_energy,
            gas_constant=self._to_tensor(self.gas_constant, dtype=temperature.dtype),
            reference_temp=self._to_tensor(self.reference_temperature, dtype=temperature.dtype),
        )

        # References:
        #   - Davidson et al. (2014) DAMM calibration guidance for bounded Vmax response
        # Mathematical expression:
        #   Vmax_clamped = clamp(Vmax(T), fmin_vmax * Vmax_ref, fmax_vmax * Vmax_ref)
        # Parameter meanings:
        #   vmax_clamp_min_factor: lower bound multiplier for Vmax(T) [dimensionless]
        #   vmax_clamp_max_factor: upper bound multiplier for Vmax(T) [dimensionless]
        vmax_min_factor = self._to_tensor(self.vmax_clamp_min_factor, dtype=temperature.dtype)
        vmax_max_factor = self._to_tensor(self.vmax_clamp_max_factor, dtype=temperature.dtype)
        vmax_temp = torch.clamp(vmax_temp, min=vmax_ref * vmax_min_factor, max=vmax_ref * vmax_max_factor)
        
        return vmax_temp
        
    def _calculate_km_temperature_response(self,
                                         substrate_type: str,
                                         temperature: torch.Tensor) -> torch.Tensor:
        """
        Calculate temperature response for half-saturation constant.
        
        Parameters:
        -----------
        substrate_type : str
            Type of substrate (metabolic_litter, structural_litter, som)
        temperature : torch.Tensor
            Soil temperature [K]
            
        Returns:
        --------
        torch.Tensor
            Temperature-modified Km [g C m^-2]
            
        Mathematical Expression:
        -----------------------
        Km(T) = Km_ref * exp(Ea_km/R * (1/T_ref - 1/T))
        """
        validate_temperature_range(temperature)
        
        km_ref = self._to_tensor(self.kinetic_params[substrate_type]['km_ref'], dtype=temperature.dtype)
        activation_energy = self._to_tensor(self.ea_km, dtype=temperature.dtype)

        km_temp = arrhenius_temperature_response(
            param_25=km_ref,
            temperature=temperature,
            activation_energy=activation_energy,
            gas_constant=self._to_tensor(self.gas_constant, dtype=temperature.dtype),
            reference_temp=self._to_tensor(self.reference_temperature, dtype=temperature.dtype),
        )

        # References:
        #   - Davidson et al. (2014) DAMM calibration guidance for bounded Km response
        # Mathematical expression:
        #   Km_clamped = clamp(Km(T), fmin_km * Km_ref, fmax_km * Km_ref)
        # Parameter meanings:
        #   km_clamp_min_factor: lower bound multiplier for Km(T) [dimensionless]
        #   km_clamp_max_factor: upper bound multiplier for Km(T) [dimensionless]
        km_min_factor = self._to_tensor(self.km_clamp_min_factor, dtype=temperature.dtype)
        km_max_factor = self._to_tensor(self.km_clamp_max_factor, dtype=temperature.dtype)
        km_temp = torch.clamp(km_temp, min=km_ref * km_min_factor, max=km_ref * km_max_factor)
        
        return km_temp
    
    def _calculate_moisture_response(self, soil_moisture: torch.Tensor) -> torch.Tensor:
        """
        Calculate moisture response modifier for enzyme activity.
        
        Parameters:
        -----------
        soil_moisture : torch.Tensor
            Volumetric soil moisture [m^3 m^-3]
            
        Returns:
        --------
        torch.Tensor
            Moisture response multiplier [dimensionless]
            
        Mathematical Expression:
        -----------------------
        Optimum response curve:
        f_theta = (theta / theta_opt)^(1/n) for theta <= theta_opt
        f_theta = ((1-theta) / (1-theta_opt))^(1/n) for theta > theta_opt
        """
        validate_positive_values(soil_moisture, "soil_moisture")
        
        theta_opt = self._to_tensor(self.moisture_optimum, dtype=soil_moisture.dtype)
        power = 1.0 / self.moisture_power
        one = self._to_tensor(1.0, dtype=soil_moisture.dtype)
        denom_below = torch.clamp(theta_opt, min=self._to_tensor(self.epsilon_concentration, dtype=soil_moisture.dtype))
        denom_above = torch.clamp(one - theta_opt, min=self._to_tensor(self.epsilon_concentration, dtype=soil_moisture.dtype))

        moisture_factor = torch.where(
            soil_moisture <= theta_opt,
            # Below optimum: gradual increase
            torch.pow(soil_moisture / denom_below, power),
            # Above optimum: gradual decrease due to anaerobic conditions
            torch.pow((one - soil_moisture) / denom_above, power)
        )
        
        # Ensure non-negative response
        moisture_factor = torch.clamp(
            moisture_factor,
            min=self._to_tensor(self.moisture_clamp_min, dtype=soil_moisture.dtype),
            max=self._to_tensor(self.moisture_clamp_max, dtype=soil_moisture.dtype)
        )
        
        return moisture_factor
    
    def _calculate_michaelis_menten_rate(self,
                                   substrate_concentration: torch.Tensor,
                                   microbial_biomass: torch.Tensor,
                                   vmax: torch.Tensor,
                                   km: torch.Tensor) -> torch.Tensor:
        """
        Calculate enzyme-mediated consumption rate using reverse Michaelis-Menten kinetics.
        
        Parameters:
        -----------
        substrate_concentration : torch.Tensor
            Available substrate [g C m^-2]
        microbial_biomass : torch.Tensor
            Microbial biomass [g C m^-2]
        vmax : torch.Tensor
            Maximum specific enzyme activity [s^-1]
        km : torch.Tensor
            Half-saturation constant [g C m^-2]
            
        Returns:
        --------
        torch.Tensor
            Substrate consumption rate [g C m^-2 s^-1]
            
        Mathematical Expression:
        -----------------------
        V = (MIC * Vmax * [S]) / (Km + MIC)
        
        This is the "reverse" MM equation where microbes are limiting
        """
        # Reverse Michaelis-Menten equation
        # Microbes are the "enzyme", substrate availability modifies rate
        consumption_rate = (microbial_biomass * vmax * substrate_concentration) / (km + microbial_biomass)
        return consumption_rate
    
    def _calculate_microbial_dynamics(self,
                                    consumption_rates: Dict[str, torch.Tensor],
                                    microbial_biomass: torch.Tensor,
                                    temperature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate microbial growth, death, and net change.
        
        Parameters:
        -----------
        consumption_rates : Dict[str, torch.Tensor]
            Substrate consumption rates by type [g C m^-2 s^-1]
        microbial_biomass : torch.Tensor
            Current microbial biomass [g C m^-2]
        temperature : torch.Tensor
            Soil temperature [K]
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Growth rate, death rate, net change [g C m^-2 s^-1]
            
        Mathematical Expression:
        -----------------------
        Growth: G = CUE * Sigma(consumption_i)
        Death: D = (tau + tau_maint * f_T) * MIC
        Net change: dMIC/dt = G - D
        """
        # Calculate total carbon consumption
        total_consumption = sum(consumption_rates.values())
        
        # Microbial growth from carbon assimilation
        cue = self._to_tensor(self.microbial_params['carbon_use_efficiency'], dtype=total_consumption.dtype)
        growth_rate = cue * total_consumption
        
        # Temperature-dependent turnover and maintenance
        turnover_base = self._to_tensor(self.microbial_params['turnover_rate'], dtype=microbial_biomass.dtype)
        maintenance_base = self._to_tensor(self.microbial_params['maintenance_rate'], dtype=microbial_biomass.dtype)
        
        # References: Lloyd & Taylor (1994); Davidson & Janssens (2006).
        # Expression: f_T = Q10 ** ((T - T_ref) / T_interval).
        # Parameters: Q10 (temp_factor_base), T_ref (0 C via celsius_offset), T_interval (q10_interval).
        temp_celsius = temperature - self._to_tensor(self.celsius_offset, dtype=temperature.dtype)
        temp_base = self._to_tensor(self.temp_factor_base, dtype=temperature.dtype)
        q10_interval = self._to_tensor(self.q10_temperature_interval, dtype=temperature.dtype)
        exponent = temp_celsius / q10_interval
        temp_factor = torch.pow(temp_base, exponent)
        temp_factor = torch.clamp(
            temp_factor,
            min=self._to_tensor(self.temp_response_clamp_min, dtype=temp_factor.dtype),
            max=self._to_tensor(self.temp_response_clamp_max, dtype=temp_factor.dtype)
        )
        
        # Combined death rate
        death_rate = (turnover_base + maintenance_base * temp_factor) * microbial_biomass
        
        # Net microbial change
        net_change = growth_rate - death_rate
        
        return growth_rate, death_rate, net_change
    
    def forward(
        self,
        state: StateDict,
        time: float,
        dt: torch.Tensor | float,
    ) -> FluxDict:
        """
        Execute microbial-enzyme decomposition for one time step.
        
        Parameters:
        -----------
        state : StateDict
            Current system state including substrate and microbial pools
        time : float
            Current simulation time [s]
        dt : float
            Time step size [s]
        Returns:
        --------
        FluxDict
            Substrate consumption, microbial dynamics, and enzyme activities
            
        Mathematical Expression:
        -----------------------
        For each substrate i:
        V_i = (MIC * Vmax_i(T,theta) * [S_i]) / (Km_i(T) + MIC)
        
        Microbial dynamics:
        dMIC/dt = CUE * Sigma(V_i) - tau(T) * MIC
        
        Respiration:
        R = (1 - CUE) * Sigma(V_i) + tau(T) * MIC
        """
        # References: Wieder et al. (2013); Allison et al. (2010);
        # Schimel and Weintraub (2003); Manzoni and Porporato (2009).
        # Mathematical expression: V_i = MIC * Vmax_i(T,theta) * S_i / (Km_i(T) + MIC),
        # dMIC/dt = CUE * sum(V_i) - D, and R = (1-CUE) * sum(V_i) + D.
        # Parameter meanings: S_i and MIC are carbon stocks [g C m-2]; Vmax [s-1];
        # Km [g C m-2]; CUE [1]; D and R [g C m-2 s-1]; dt [s].
        # Open route: heterotrophic_respiration is the atmospheric carbon port.
        metabolic_litter = self._require_state_tensor(
            state,
            "metabolic_litter_c",
            non_negative=True,
        )
        dtype = metabolic_litter.dtype
        structural_litter = self._require_state_tensor(
            state,
            "structural_litter_c",
            dtype=dtype,
            non_negative=True,
        )
        available_som = self._require_state_tensor(
            state,
            "available_som_c",
            dtype=dtype,
            non_negative=True,
        )
        substrates = {
            'metabolic_litter': metabolic_litter,
            'structural_litter': structural_litter,
            'som': available_som,
        }

        microbial_biomass = self._require_state_tensor(
            state,
            "microbial_biomass_c",
            dtype=dtype,
            non_negative=True,
        )
        soil_temp = self._require_state_tensor(state, "soil_temperature", dtype=dtype)
        soil_moisture = self._require_state_tensor(
            state,
            "soil_moisture",
            dtype=dtype,
            non_negative=True,
        )
        if torch.any(soil_moisture > 1.0):
            raise ValueError("soil_moisture must be in the closed interval [0, 1]")
        validate_temperature_range(soil_temp)
        dt_tensor = self._validate_dt(dt, dtype=dtype)

        min_biomass = self._to_tensor(self.microbial_params['min_biomass'], dtype=dtype)
        max_biomass = self._to_tensor(self.microbial_params['max_biomass'], dtype=dtype)
        if torch.any(microbial_biomass < min_biomass) or torch.any(microbial_biomass > max_biomass):
            raise ValueError("microbial_biomass_c is outside the declared biomass bounds")
        
        # Calculate environmental modifiers
        moisture_response = self._calculate_moisture_response(soil_moisture)
        epsilon_conc_tensor = self._to_tensor(self.epsilon_concentration, dtype=soil_temp.dtype)
        
        # Calculate substrate-specific consumption rates
        consumption_rates = {}
        enzyme_activities = {}
        
        for substrate_type, substrate_pool in substrates.items():
            # Temperature-dependent kinetic parameters
            vmax_temp = self._calculate_vmax_temperature_response(substrate_type, soil_temp)
            km_temp = self._calculate_km_temperature_response(substrate_type, soil_temp)
            
            # Apply moisture modifier to Vmax
            vmax_effective = vmax_temp * moisture_response
            
            # Calculate Michaelis-Menten consumption rate
            consumption_rate = self._calculate_michaelis_menten_rate(
                substrate_pool, microbial_biomass, vmax_effective, km_temp
            )
            
            consumption_rates[substrate_type] = consumption_rate
            
            # Store enzyme activity for diagnostics
            enzyme_activities[f'{substrate_type}_vmax'] = vmax_effective
            enzyme_activities[f'{substrate_type}_km'] = km_temp
            enzyme_activities[f'{substrate_type}_activity'] = consumption_rate / torch.clamp(substrate_pool, min=epsilon_conc_tensor)

            if torch.any(consumption_rate * dt_tensor > substrate_pool):
                raise ValueError(
                    f"dt would overdraw the '{substrate_type}' substrate pool"
                )

        # Calculate microbial dynamics
        growth_rate, death_rate, net_microbial_change = self._calculate_microbial_dynamics(
            consumption_rates, microbial_biomass, soil_temp
        )

        new_microbial_biomass = microbial_biomass + net_microbial_change * dt_tensor
        if torch.any(new_microbial_biomass < min_biomass) or torch.any(
            new_microbial_biomass > max_biomass
        ):
            raise ValueError("dt would move microbial_biomass_c outside declared bounds")
        
        # Calculate respiration components
        # 1. Growth respiration (1 - CUE) of consumption
        total_consumption = sum(consumption_rates.values())
        cue_tensor = self._to_tensor(self.microbial_params['carbon_use_efficiency'], dtype=total_consumption.dtype)
        growth_respiration = (1.0 - cue_tensor) * total_consumption
        
        # 2. Maintenance respiration (microbial turnover)
        maintenance_respiration = death_rate
        
        # Total heterotrophic respiration
        total_respiration = growth_respiration + maintenance_respiration

        # References: Wieder et al. (2013); Schimel and Weintraub (2003);
        # Feinberg (2019).
        # Formula: x_dot = N_h r_h with C_h N_h = 0 and C_h = [1,1,1,1,1].
        # Parameters: each uptake column removes one substrate unit, routes CUE
        # to microbial biomass, and routes 1-CUE to respiration; mortality moves
        # microbial carbon to the respiration port. All rates are g C m-2 s-1.
        zero = torch.zeros_like(cue_tensor)
        one = torch.ones_like(cue_tensor)
        stoichiometric_n = torch.stack(
            (
                torch.stack((-one, zero, zero, zero)),
                torch.stack((zero, -one, zero, zero)),
                torch.stack((zero, zero, -one, zero)),
                torch.stack((cue_tensor, cue_tensor, cue_tensor, -one)),
                torch.stack((one - cue_tensor, one - cue_tensor, one - cue_tensor, one)),
            ),
            dim=0,
        )
        carbon_ledger = torch.ones(
            (1, 5),
            device=total_respiration.device,
            dtype=total_respiration.dtype,
        )
        enzyme_network = advance_reaction_network(
            name="microbial_enzyme_decomposition_network",
            species=(
                "metabolic_litter_c",
                "structural_litter_c",
                "available_som_c",
                "microbial_biomass_c",
                "heterotrophic_respiration",
            ),
            reactions=(
                "metabolic_substrate_consumption",
                "structural_substrate_consumption",
                "som_substrate_consumption",
                "microbial_turnover",
            ),
            stoichiometric_N=stoichiometric_n,
            rates=(
                consumption_rates["metabolic_litter"],
                consumption_rates["structural_litter"],
                consumption_rates["som"],
                death_rate,
            ),
            dt=dt_tensor,
            **self._iphs_reaction_conservation_kwargs(
                conservation_matrix=carbon_ledger,
                open_system_route="port",
                default_strict=True,
            ),
        )
        total_respiration = enzyme_network.species_rate("heterotrophic_respiration")
        
        # Calculate substrate pool changes
        substrate_changes = {
            "metabolic_litter": enzyme_network.state_delta[
                enzyme_network.species.index("metabolic_litter_c")
            ],
            "structural_litter": enzyme_network.state_delta[
                enzyme_network.species.index("structural_litter_c")
            ],
            "som": enzyme_network.state_delta[
                enzyme_network.species.index("available_som_c")
            ],
        }
        delta_microbial_biomass = enzyme_network.state_delta[
            enzyme_network.species.index("microbial_biomass_c")
        ]
        new_microbial_biomass = microbial_biomass + delta_microbial_biomass

        respiration_step = total_respiration * dt_tensor
        updates = {
            'metabolic_litter_c': substrate_changes['metabolic_litter'],
            'structural_litter_c': substrate_changes['structural_litter'],
            'available_som_c': substrate_changes['som'],
            'microbial_biomass_c': delta_microbial_biomass,
            'heterotrophic_respiration': total_respiration,
            'diagnostics.microbial_enzyme.step_seconds': dt_tensor.detach().reshape(()),
        }
        self._enforce_forward_conservation(state, updates)

        carbon_balance_residual = (
            substrate_changes['metabolic_litter']
            + substrate_changes['structural_litter']
            + substrate_changes['som']
            + delta_microbial_biomass
            + respiration_step
        )
        realized_cue = torch.where(
            total_consumption > 0.0,
            growth_rate / total_consumption,
            torch.zeros_like(total_consumption),
        )

        return {
            'metabolic_litter_c': self._accumulate_carbon(
                substrate_changes['metabolic_litter']
            ),
            'structural_litter_c': self._accumulate_carbon(
                substrate_changes['structural_litter']
            ),
            'available_som_c': self._accumulate_carbon(substrate_changes['som']),
            'microbial_biomass_c': self._accumulate_carbon(delta_microbial_biomass),
            'heterotrophic_respiration': self._replace(
                total_respiration,
                unit="g C m-2 s-1",
                basis="per_second",
                element="C",
            ),
            'diagnostics.microbial_enzyme.microbial_growth_rate': self._replace(
                growth_rate, unit="g C m-2 s-1", basis="per_second"
            ),
            'diagnostics.microbial_enzyme.microbial_death_rate': self._replace(
                death_rate, unit="g C m-2 s-1", basis="per_second"
            ),
            'diagnostics.microbial_enzyme.microbial_respiration': self._replace(
                total_respiration, unit="g C m-2 s-1", basis="per_second"
            ),
            'diagnostics.microbial_enzyme.growth_respiration': self._replace(
                growth_respiration, unit="g C m-2 s-1", basis="per_second"
            ),
            'diagnostics.microbial_enzyme.maintenance_respiration': self._replace(
                maintenance_respiration, unit="g C m-2 s-1", basis="per_second"
            ),
            'diagnostics.microbial_enzyme.metabolic_consumption': self._replace(
                consumption_rates['metabolic_litter'],
                unit="g C m-2 s-1",
                basis="per_second",
            ),
            'diagnostics.microbial_enzyme.structural_consumption': self._replace(
                consumption_rates['structural_litter'],
                unit="g C m-2 s-1",
                basis="per_second",
            ),
            'diagnostics.microbial_enzyme.som_consumption': self._replace(
                consumption_rates['som'], unit="g C m-2 s-1", basis="per_second"
            ),
            'diagnostics.microbial_enzyme.metabolic_litter.vmax': self._replace(
                enzyme_activities['metabolic_litter_vmax'],
                unit="s-1",
                basis="instantaneous",
            ),
            'diagnostics.microbial_enzyme.metabolic_litter.km': self._replace(
                enzyme_activities['metabolic_litter_km'],
                unit="g C m-2",
                basis="instantaneous",
            ),
            'diagnostics.microbial_enzyme.metabolic_litter.activity': self._replace(
                enzyme_activities['metabolic_litter_activity'],
                unit="s-1",
                basis="instantaneous",
            ),
            'diagnostics.microbial_enzyme.structural_litter.vmax': self._replace(
                enzyme_activities['structural_litter_vmax'],
                unit="s-1",
                basis="instantaneous",
            ),
            'diagnostics.microbial_enzyme.structural_litter.km': self._replace(
                enzyme_activities['structural_litter_km'],
                unit="g C m-2",
                basis="instantaneous",
            ),
            'diagnostics.microbial_enzyme.structural_litter.activity': self._replace(
                enzyme_activities['structural_litter_activity'],
                unit="s-1",
                basis="instantaneous",
            ),
            'diagnostics.microbial_enzyme.som.vmax': self._replace(
                enzyme_activities['som_vmax'], unit="s-1", basis="instantaneous"
            ),
            'diagnostics.microbial_enzyme.som.km': self._replace(
                enzyme_activities['som_km'], unit="g C m-2", basis="instantaneous"
            ),
            'diagnostics.microbial_enzyme.som.activity': self._replace(
                enzyme_activities['som_activity'], unit="s-1", basis="instantaneous"
            ),
            'diagnostics.microbial_enzyme.moisture_response': self._replace(
                moisture_response, unit="dimensionless", basis="instantaneous"
            ),
            'diagnostics.microbial_enzyme.temperature_celsius': self._replace(
                soil_temp
                - self._to_tensor(self.celsius_offset, dtype=soil_temp.dtype),
                unit="degC",
                basis="instantaneous",
            ),
            'diagnostics.microbial_enzyme.microbial_biomass_content': self._replace(
                new_microbial_biomass,
                unit="g C m-2",
                basis="instantaneous",
            ),
            'diagnostics.microbial_enzyme.total_substrate_consumed': self._replace(
                total_consumption, unit="g C m-2 s-1", basis="per_second"
            ),
            'diagnostics.microbial_enzyme.carbon_use_efficiency_realized': self._replace(
                realized_cue, unit="dimensionless", basis="instantaneous"
            ),
            'diagnostics.microbial_enzyme.carbon_balance_residual': self._replace(
                carbon_balance_residual,
                unit="g C m-2",
                basis="per_step",
            ),
            # Declared step length so the public ledger can integrate the rates.
            'diagnostics.microbial_enzyme.step_seconds': self._replace(
                dt_tensor.detach().reshape(()),
                unit="s",
                basis="instantaneous",
            ),
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
        dt: Optional[torch.Tensor | float] = None,
    ) -> Dict[str, float]:
        """Return the canonical carbon-ledger residual for public outputs.

        References: Wieder et al. (2013); Schimel and Weintraub (2003).
        Mathematical expression: sum(delta C_pool) + R_microbial * dt = 0.
        Parameter meanings: delta C_pool is a per-step stock increment
        [g C m-2], R_microbial is the boundary respiration rate
        [g C m-2 s-1], and dt is the time step [s].
        """
        required_outputs = {
            "metabolic_litter_c",
            "structural_litter_c",
            "available_som_c",
            "microbial_biomass_c",
            "heterotrophic_respiration",
        }
        if not required_outputs.issubset(output_updates):
            return super().check_conservation_laws(input_state, output_updates)

        def _payload(key: str) -> torch.Tensor:
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            return self._to_tensor(value, dtype=None)

        pool_delta = torch.stack(
            [
                _payload("metabolic_litter_c"),
                _payload("structural_litter_c"),
                _payload("available_som_c"),
                _payload("microbial_biomass_c"),
            ]
        ).sum(dim=0)
        if dt is not None:
            dt_value = self._to_tensor(dt, dtype=pool_delta.dtype)
        else:
            step_key = "diagnostics.microbial_enzyme.step_seconds"
            if step_key not in output_updates:
                raise ValueError("dt is required for the public carbon-rate ledger")
            dt_value = _payload(step_key)
        if dt_value.numel() != 1 or not torch.isfinite(dt_value).all():
            raise ValueError("dt is required for the public carbon-rate ledger")
        if float(dt_value.detach().reshape(-1)[0].tolist()) <= 0.0:
            raise ValueError("dt is required for the public carbon-rate ledger")
        residual = pool_delta + _payload("heterotrophic_respiration") * dt_value.to(
            device=pool_delta.device,
            dtype=pool_delta.dtype,
        )
        magnitude = float(torch.max(torch.abs(residual)).detach().tolist())
        return {
            law: magnitude
            for law in self.metadata.conservation_laws
        }

# Export main class
__all__ = ['MicrobialEnzymeDecomposition']

