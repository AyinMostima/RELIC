"""
Nitrogen Mineralization Process Module for RELIC

This module implements nitrogen mineralization and immobilization processes
following the substrate-decomposer stoichiometry framework. It provides 
coupled carbon-nitrogen dynamics with microbial mediation.

Scientific Basis:
-----------------
The model follows the theoretical framework established by Manzoni & Porporato 
(2009) for substrate-decomposer stoichiometry, integrating C and N cycling 
through microbial biomass dynamics with stoichiometric constraints.

References:
- Manzoni, S. & Porporato, A. (2009). Soil carbon and nitrogen mineralization: 
  Theory and models across scales. Soil Biology & Biochemistry, 41, 1355-1379.
- Grant, R.F. et al. (1993). Simulation of carbon and nitrogen transformations 
  in soil: Mineralization. Soil Biology & Biochemistry, 25, 1317-1329.
- Stanford, G. & Smith, S.J. (1972). Nitrogen mineralization potentials of soils.
  Soil Science Society of America Journal, 36, 465-472.
- Jansson, S.L. & Persson, J. (1982). Mineralization and immobilization of soil
  nitrogen. Ecological Bulletins, 33, 229-252.

Mathematical Expression:
-----------------------
Net mineralization/immobilization:
NET_MIN = GROSS_MIN - IMMOBILIZATION

Gross mineralization:
GROSS_MIN = k_min * N_org * f_T(T) * f_theta(theta) * f_pH(pH)

Immobilization demand:
IMM_DEMAND = C_uptake / ((C/N)_B * eta_N)

Where:
- NET_MIN: net N mineralization rate [g N m^-2 s^-1]
- k_min: mineralization rate constant [s^-1]
- N_org: organic nitrogen pool [g N m^-2]
- DEC: carbon decomposition rate [g C m^-2 s^-1]
- (C/N)_B: microbial C:N ratio [dimensionless]
- eta_N: effective mineral-N assimilation efficiency [dimensionless]
"""

from typing import Any, Dict, Optional
import logging
import torch
from torch import Tensor

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
from relic.core.state import UpdateItem, UpdateMode

# Import unified response functions
from relic.utils.unified_response_functions import (
    q10_temperature_response,
    arrhenius_temperature_response,
)
from relic.core.functor import BiologicalProcessFunctor
from relic.computation.operators.iphs import advance_reaction_network
from relic.utils.physical_constants import get_physical_constant
from relic.scientific.pedosphere.api_contract import (
    PedosphereContractMixin,
    FluxDirection,
    FluxSpecification,
    FluxType,
    StandardPedosphereOutputs,
    validate_functor_output,
)

logger = logging.getLogger(__name__)


class MineralizationProcess(PedosphereContractMixin, BiologicalProcessFunctor):
    """
    Nitrogen mineralization/immobilization process functor.
    
    Implements coupled C-N dynamics with microbial stoichiometric constraints
    following the substrate-decomposer framework.
    
    References:
    - Manzoni & Porporato (2009) substrate-decomposer stoichiometry
    - Grant et al. (1993) soil C and N transformations
    - Stanford & Smith (1972) mineralization potentials
    
    Mathematical Expression:
    -----------------------
    Substrate-decomposer balance:
    
    Gross mineralization:
    R_gross_min = k_min * N_org * f_env(T, theta, pH)
    
    Microbial N demand:
    N_demand = C_uptake / (CN_mic * eta_N)
    
    Net mineralization:
    R_net_min = R_gross_min - R_immob
    
    Immobilization:
    R_immob = min(N_demand, N_mineral_available / dt)
    
    Where:
    - R_gross_min: gross mineralization rate [g N m^-2 s^-1]
    - k_min: temperature-sensitive rate constant [s^-1]
    - N_org: organic nitrogen content [g N m^-2]
    - f_env: environmental response function [dimensionless]
    - N_demand: microbial nitrogen demand [g N m^-2 s^-1]
    - C_uptake: carbon uptake by microbes [g C m^-2 s^-1]
    - CN_mic: microbial C:N ratio [dimensionless]
    - eta: nitrogen assimilation efficiency [dimensionless]
    - R_net_min: net mineralization rate [g N m^-2 s^-1]
    - R_immob: immobilization rate [g N m^-2 s^-1]
    
    Units:
    - Nitrogen pools: g N m^-2
    - Mineralization rates: g N m^-2 s^-1
    - Temperature: K
    - Soil moisture: m^3 m^-3
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    PARAMETER_DESCRIPTIONS = {
        "pedosphere.mineralization.temp_response_type": "Temperature response selector [string]",
        "pedosphere.mineralization.q10_mineralization": "Q10 response coefficient [dimensionless]",
        "pedosphere.mineralization.activation_energy": "Arrhenius activation energy [J mol-1]",
        "pedosphere.mineralization.moisture_optimum": "Optimal volumetric soil moisture [m3 m-3]",
        "pedosphere.mineralization.moisture_response_power": "Gaussian moisture-response width [dimensionless]",
        "pedosphere.mineralization.k_mineralization": "Base mineralization rate [yr-1]",
        "pedosphere.mineralization.cn_ratio_microbial": "Microbial carbon-to-nitrogen ratio [dimensionless]",
        "pedosphere.mineralization.n_assimilation_efficiency": "Nitrogen assimilation efficiency [dimensionless]",
        "pedosphere.mineralization.carbon_use_efficiency": "Microbial carbon use efficiency [dimensionless]",
        "pedosphere.physical.ph_optimum_decomposition": "Optimal pH for decomposition [dimensionless]",
        "pedosphere.physical.ph_response_width": "Gaussian pH-response width [dimensionless]",
        "pedosphere.numerical.temperature_response_min": "Minimum temperature response [dimensionless]",
        "pedosphere.numerical.temperature_response_max": "Maximum temperature response [dimensionless]",
        "pedosphere.numerical.moisture_response_min": "Minimum moisture response [dimensionless]",
        "pedosphere.numerical.moisture_response_max": "Maximum moisture response [dimensionless]",
        "pedosphere.numerical.ph_response_min": "Minimum pH response [dimensionless]",
        "pedosphere.numerical.ph_response_max": "Maximum pH response [dimensionless]",
        "pedosphere.nutrient_cycling.k_microbial_death": "Microbial nitrogen death rate [s-1]",
    }
    
    def __init__(self, name: str = "mineralization", config: Optional[Dict] = None):
        """
        Initialize nitrogen mineralization process functor.
        
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
            description="Nitrogen mineralization/immobilization with microbial stoichiometry",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "x_dot = N_h r_h + G_h u_litter; "
                "NET_MIN = GROSS_MIN - IMMOB"
            ),
            references=[
                "Manzoni, S. & Porporato, A. (2009). Substrate-decomposer stoichiometry",
                "Grant, R.F. et al. (1993). Soil C and N transformations",
                "Stanford, G. & Smith, S.J. (1972). N mineralization potentials",
                "Jansson, S.L. & Persson, J. (1982). Mineralization and immobilization",
                "Parton et al. (1987). SSSA Journal 51, 1173-1179",
                "Lawrence et al. (2019). JAMES 11, 4245-4287",
            ],
            parameters=dict(self.PARAMETER_DESCRIPTIONS),
            input_variables=[
                "organic_nitrogen",
                "nh4_nitrogen",
                "microbial_nitrogen",
                "soil_temperature",
                "soil_moisture",
                "soil_ph",
                "carbon_decomposition_rate",
                "organic_nitrogen_input",
            ],
            output_variables=[
                "organic_nitrogen",
                "nh4_nitrogen",
                "microbial_nitrogen",
                "mineralization_flux",
                "gross_mineralization_flux",
                "immobilization_flux",
                "microbial_death_flux",
                "diagnostics.mineralization.organic_nitrogen_boundary_input",
            ],
            conservation_laws=["nitrogen", "mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=[
                "carbon_decomposition",
                "litterfall_interface_mapper",
                "microbial_dynamics",
                "soil_chemistry",
            ],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        
        # Initialize base functor with prepared metadata
        super().__init__(name=name, config=config, metadata=metadata, device=None)
        self.metadata.parameters = dict(self.PARAMETER_DESCRIPTIONS)
        # Ensure raw config equality for unit tests
        self._config_raw = dict(config) if config else {}
        self.logger = logger
        self.optional_input_variables = []
        
        # Temperature response configuration
        self.temp_response_type = self.get_parameter_value(
            'temp_response_type',
            namespace='pedosphere.mineralization',
            required=True,
        )
        self.q10 = self.get_parameter_value(
            'q10_mineralization',
            namespace='pedosphere.mineralization',
            required=True,
        )
        self.k_mineralization = self.get_parameter_value(
            'k_mineralization',
            namespace='pedosphere.mineralization',
            required=True,
        )
        self.c_use_efficiency = self.get_parameter_value(
            'carbon_use_efficiency',
            namespace='pedosphere.mineralization',
            required=True,
        )

        # Environmental response parameters
        self.ph_optimum = self.get_parameter_value(
            'ph_optimum_decomposition',
            namespace='pedosphere.physical',
            required=True,
        )
        self.ph_range = self.get_parameter_value(
            'ph_response_width',
            namespace='pedosphere.physical',
            required=True,
        )

        # References: RELIC architecture_overview.md 6.1.2 L3 access path
        # Mathematical expression: fixed PM values follow module dtype and device.
        # Parameter meanings: response bounds constrain dimensionless modifiers.
        fixed_parameters = {
            'activation_energy': ('activation_energy', 'pedosphere.mineralization'),
            'moisture_optimum': ('moisture_optimum', 'pedosphere.mineralization'),
            'moisture_response_power': (
                'moisture_response_power',
                'pedosphere.mineralization',
            ),
            'temperature_response_min': (
                'temperature_response_min',
                'pedosphere.numerical',
            ),
            'temperature_response_max': (
                'temperature_response_max',
                'pedosphere.numerical',
            ),
            'moisture_response_min': ('moisture_response_min', 'pedosphere.numerical'),
            'moisture_response_max': ('moisture_response_max', 'pedosphere.numerical'),
            'ph_response_min': ('ph_response_min', 'pedosphere.numerical'),
            'ph_response_max': ('ph_response_max', 'pedosphere.numerical'),
            'cn_ratio_microbial': ('cn_ratio_microbial', 'pedosphere.mineralization'),
            'n_assimilation_efficiency': (
                'n_assimilation_efficiency',
                'pedosphere.mineralization',
            ),
        }
        for attribute_name, (parameter_name, namespace) in fixed_parameters.items():
            value = self.get_parameter_value(
                parameter_name,
                namespace=namespace,
                required=True,
            )
            buffer = self._ensure_tensor(value).detach().clone()
            self.register_buffer(attribute_name, buffer)

        self.k_microbial_death = self.get_parameter_value(
            'k_microbial_death',
            namespace='pedosphere.nutrient_cycling',
            required=True,
        )
        self.reference_temperature = float(get_physical_constant("t_ref"))
        self.seconds_per_year = float(get_physical_constant("seconds_per_year"))

    def _calculate_temperature_response(self, soil_temp: Tensor) -> Tensor:
        """
        Calculate temperature response for mineralization rates.

        Returns
        -------
        Tensor
            Temperature response factor [dimensionless]
        """
        # References: Kirschbaum (1995); Manzoni and Porporato (2009).
        # Mathematical expression: f_T=clip(Q10^((T-T_ref)/dT10),f_min,f_max), or
        #   clip(exp(Ea/R*(1/T_ref-1/T)),f_min,f_max) for the declared Arrhenius selection.
        # Parameter meanings: Q10 or Ea belongs to the configured response family;
        #   R and dT10 come from the existing shared response/physical-constant owners.
        # Units: T,T_ref,dT10 [K]; Ea [J mol-1]; R [J mol-1 K-1]; f_T [1].
        # Domain/constraints: Positive absolute temperatures; selection is configuration,
        #   not a fallback after a failed formula. Shared response functions own evaluation.
        # Sign convention: The response scales a nonnegative potential transformation rate.
        temp_tensor = self._ensure_tensor(soil_temp)
        ref_temp = self._ensure_tensor(self.reference_temperature).to(
            dtype=temp_tensor.dtype,
            device=temp_tensor.device,
        )
        one_tensor = torch.ones_like(temp_tensor)
        if self.temp_response_type == 'arrhenius':
            activation = self.activation_energy.to(
                dtype=temp_tensor.dtype,
                device=temp_tensor.device,
            )
            response = arrhenius_temperature_response(one_tensor, temp_tensor, activation, ref_temp)
        else:
            q10_value = self.q10.to(
                dtype=temp_tensor.dtype,
                device=temp_tensor.device,
            )
            response = q10_temperature_response(one_tensor, temp_tensor, q10_value, ref_temp)
        # References: RELIC numerical stability guidelines
        # Mathematical expression: f_T_clamped = clamp(f_T, f_T_min, f_T_max)
        # Parameters: temperature_response_min [1], temperature_response_max [1]
        min_tensor = self._ensure_tensor(self.temperature_response_min).to(
            dtype=temp_tensor.dtype,
            device=temp_tensor.device,
        )
        max_tensor = self._ensure_tensor(self.temperature_response_max).to(
            dtype=temp_tensor.dtype,
            device=temp_tensor.device,
        )
        return torch.clamp(response, min=min_tensor, max=max_tensor)

    def _calculate_moisture_response(self, soil_moisture: Tensor) -> Tensor:
        """
        Calculate moisture response for mineralization rates.

        Returns
        -------
        Tensor
            Moisture response factor [dimensionless]
        """
        # References: Moyano et al. (2013); Manzoni and Porporato (2009).
        # Mathematical expression: f_theta=clip(exp(-((theta-theta_opt)/w_theta)^2),f_min,f_max).
        # Parameter meanings: moisture_response_power is the existing Gaussian width
        #   despite its name; moisture_optimum is the response optimum.
        # Units: theta,theta_opt,w_theta [m3 m-3]; response and its bounds [1].
        # Domain/constraints: Positive width and moisture in [0,1]; the Gaussian
        #   response is a reduced environmental modifier, not a water-balance process.
        # Sign convention: Moving away from optimum suppresses mineralization magnitude.
        moisture_tensor = self._ensure_tensor(soil_moisture)
        optimum = self.moisture_optimum.to(
            dtype=moisture_tensor.dtype,
            device=moisture_tensor.device,
        )
        width = self.moisture_response_power.to(
            dtype=moisture_tensor.dtype,
            device=moisture_tensor.device,
        )
        response = torch.exp(-((moisture_tensor - optimum) / width) ** 2)
        # References: RELIC numerical stability guidelines
        # Mathematical expression: f_theta_clamped = clamp(f_theta, theta_min, theta_max)
        # Parameters: moisture_response_min [1], moisture_response_max [1]
        min_tensor = self._ensure_tensor(self.moisture_response_min).to(
            dtype=moisture_tensor.dtype,
            device=moisture_tensor.device,
        )
        max_tensor = self._ensure_tensor(self.moisture_response_max).to(
            dtype=moisture_tensor.dtype,
            device=moisture_tensor.device,
        )
        return torch.clamp(response, min=min_tensor, max=max_tensor)

    def _calculate_ph_response(self, soil_ph: Tensor) -> Tensor:
        """
        Calculate pH response for mineralization rates.

        Returns
        -------
        Tensor
            pH response factor [dimensionless]
        """
        # References: Curtin et al. (1998); Brady and Weil (2017).
        # Mathematical expression: f_pH=clip(exp(-((pH-pH_opt)/w_pH)^2),f_min,f_max).
        # Parameter meanings: ph_optimum is the optimum; ph_range holds the existing
        #   response width; bounds are the declared ph_response_min/max.
        # Units: pH coordinates, width, response and bounds [1].
        # Domain/constraints: Positive response width and physical pH domain;
        #   this empirical modifier does not solve chemical acid-base equilibria.
        # Sign convention: Deviations from optimum reduce the rate multiplier.
        ph_tensor = self._ensure_tensor(soil_ph)
        optimum = self.ph_optimum.to(dtype=ph_tensor.dtype, device=ph_tensor.device)
        width = self.ph_range.to(dtype=ph_tensor.dtype, device=ph_tensor.device)
        response = torch.exp(-((ph_tensor - optimum) / width) ** 2)
        # References: RELIC numerical stability guidelines
        # Mathematical expression: f_pH_clamped = clamp(f_pH, pH_min, pH_max)
        # Parameters: ph_response_min [1], ph_response_max [1]
        min_tensor = self._ensure_tensor(self.ph_response_min).to(
            dtype=ph_tensor.dtype,
            device=ph_tensor.device,
        )
        max_tensor = self._ensure_tensor(self.ph_response_max).to(
            dtype=ph_tensor.dtype,
            device=ph_tensor.device,
        )
        return torch.clamp(response, min=min_tensor, max=max_tensor)

    def _calculate_environmental_response(
        self,
        soil_temp: Tensor,
        soil_moisture: Tensor,
        soil_ph: Tensor,
    ) -> Tensor:
        """
        Calculate combined environmental response multiplier.

        Returns
        -------
        Tensor
            Combined response factor [dimensionless]
        """
        # References: Manzoni and Porporato (2009); Grant et al. (1993).
        # Mathematical expression: f_env=f_T(T)*f_theta(theta)*f_pH(pH).
        # Parameter meanings: Each factor is evaluated by its existing dedicated helper
        #   with the same selected ParameterManager response coefficients.
        # Units: T [K]; theta [m3 m-3]; pH and all factors [1].
        # Domain/constraints: Retains the separable multiplicative closure; no new
        #   empirical interaction term is implied by the environmental composition.
        # Sign convention: Nonnegative controls multiply gross-mineralization magnitude.
        temp_response = self._calculate_temperature_response(soil_temp)
        moisture_response = self._calculate_moisture_response(soil_moisture)
        ph_response = self._calculate_ph_response(soil_ph)
        return temp_response * moisture_response * ph_response

    def _calculate_gross_mineralization(self, organic_n: Tensor, env_response: Tensor) -> Tensor:
        """
        Calculate gross mineralization rate.

        Returns
        -------
        Tensor
            Gross mineralization rate [g N m-2 s-1]
        """
        # References: Stanford and Smith (1972); Manzoni and Porporato (2009).
        # Mathematical expression: R_gross=(k_min/seconds_per_year)*N_org*f_env.
        # Parameter meanings: k_min is the YAML annual mineralization coefficient;
        #   N_org is donor organic nitrogen; seconds_per_year is the existing conversion.
        # Units: k_min [yr-1]; seconds_per_year [s yr-1]; N_org [g N m-2];
        #   R_gross [g N m-2 s-1]; f_env [1].
        # Domain/constraints: Nonnegative pool/coefficient and valid environment factor;
        #   this is a potential rate before the forward donor limiter.
        # Sign convention: Positive R_gross transfers organic N to NH4.
        k_min_tensor = (self.k_mineralization / self.seconds_per_year).to(
            dtype=organic_n.dtype,
            device=organic_n.device,
        )
        return k_min_tensor * organic_n * env_response

    def _calculate_microbial_nitrogen_demand(
        self,
        carbon_uptake: Tensor,
    ) -> Tensor:
        """
        Calculate microbial nitrogen demand.

        Returns
        -------
        Tensor
            Nitrogen demand rate [g N m-2 s-1]
        """
        # References: Manzoni and Porporato (2009), Section 3.3 MIT pathway.
        # Mathematical expression: N_demand=C_uptake/(CN_microbe*eta_N).
        # Parameter meanings: C_uptake is CUE times upstream carbon decomposition;
        #   CN_microbe is microbial C:N mass ratio and eta_N is assimilation efficiency.
        # Units: C_uptake [g C m-2 s-1]; CN_microbe [g C g-1 N]; eta_N [1];
        #   N_demand [g N m-2 s-1].
        # Domain/constraints: Positive CN_microbe/eta_N and nonnegative upstream C;
        #   the initial NH4 donor constraint is applied once in forward.
        # Sign convention: Positive demand requests transfer of NH4 to microbial N.
        cn_tensor = self.cn_ratio_microbial.to(
            dtype=carbon_uptake.dtype,
            device=carbon_uptake.device,
        )
        eta_tensor = self.n_assimilation_efficiency.to(
            dtype=carbon_uptake.dtype,
            device=carbon_uptake.device,
        )
        microbial_n_requirement = carbon_uptake / cn_tensor
        return microbial_n_requirement / eta_tensor

    def _compute_rate_core(self, state: StateDict) -> FluxDict:
        """
        Compute nitrogen mineralization/immobilization rates given the current state.
        
        This method implements the BiologicalProcess interface for coupled C-N dynamics
        with microbial stoichiometric constraints.
        
        Parameters:
            state : StateDict
                Canonical StateManager inputs including nitrogen pools,
                soil drivers, and carbon coupling ports (the one input channel).
                
        Returns:
            rates : FluxDict
                Instantaneous N mineralization/immobilization rates
                Keys: "gross_mineralization_rate", "potential_immobilization_rate",
                      "microbial_death_rate", and current N pools
                Values: N transformation rates [g N m^-2 s^-1]
                
        Mathematical Expression:
            Substrate-decomposer balance:
            
            Gross mineralization:
            R_gross_min = k_min * N_org * f_env(T, theta, pH)
            
            Microbial N demand:  
            N_demand = C_uptake / (CN_mic * eta_N)
            
            Net mineralization:
            R_net_min = R_gross_min - R_immob
        """
        # References: Manzoni and Porporato (2009); Grant et al. (1993).
        # Mathematical expression: R_gross=_calculate_gross_mineralization(N_org,f_env);
        #   C_uptake=CUE*C_decomp; R_immob_potential=C_uptake/(CN_microbe*eta_N);
        #   R_death=k_death*N_microbe.
        # Parameter meanings: C_decomp is the supplied carbon port, CUE the microbial
        #   carbon-use efficiency, and k_death the microbial-N return coefficient.
        # Units: carbon port [g C m-2 s-1]; N pools [g N m-2]; k_death [s-1];
        #   returned transformation rates [g N m-2 s-1].
        # Domain/constraints: State holds canonical required inputs; core returns rates
        #   only, while donor limits and dt multiplication belong to forward.
        # Sign convention: Gross mineralization, immobilization and death are positive
        #   channel magnitudes with directions assigned by the existing stoichiometry.
        organic_n = state['organic_nitrogen']
        nh4_n = state['nh4_nitrogen']
        microbial_n = state['microbial_nitrogen']
        soil_temp = state['soil_temperature']
        soil_moisture = state['soil_moisture']
        soil_ph = state['soil_ph']
        carbon_decomposition = state['carbon_decomposition_rate']

        # ======================================================================
        # CORE RATE CALCULATIONS
        # ======================================================================
        # References: Manzoni and Porporato (2009); Grant et al. (1993).
        # Equations: R_gross_min = k_min * N_org * f_env(T, theta, pH);
        #            C_uptake = CUE * C_decomp;
        #            N_demand = C_uptake / (CN_mic * eta_N);
        #            R_immob,potential = N_demand;
        #            R_net = R_gross_min - R_immob.
        # Parameters: k_min (mineralization rate), N_org (organic N pool),
        #             f_env (temperature, moisture, pH modifiers),
        #             CUE (carbon-use efficiency), CN_mic (microbial C:N),
        #             eta_N (N assimilation efficiency), NH4_available (NH4 pool).
        # Reference: Manzoni & Porporato (2009) SBB 41:1355-1379
        # Mathematical expressions as per module docstring (line 24-42, 78-111)

        # 1. Environmental response function f_env(T, theta, pH)
        env_response = self._calculate_environmental_response(
            soil_temp,
            soil_moisture,
            soil_ph,
        )

        # 2. Gross mineralization: R_gross_min = k_min * N_org * f_env
        gross_mineralization = self._calculate_gross_mineralization(organic_n, env_response)

        # 3. Microbial nitrogen demand from the upstream carbon flux.
        cue = self.c_use_efficiency.to(
            dtype=carbon_decomposition.dtype,
            device=carbon_decomposition.device,
        )
        carbon_uptake = carbon_decomposition * cue
        nitrogen_demand = self._calculate_microbial_nitrogen_demand(carbon_uptake)

        # 4. Potential immobilization is microbial N demand. The current-step
        # NH4 donor constraint is applied once in forward, where dt is known.
        potential_immobilization = nitrogen_demand
        
        # Get microbial death rate parameter (new architecture)
        # Calculate microbial death nitrogen RATE (returns to organic pool)
        # Reference: Manzoni & Porporato (2009) SBB 41:1355-1379, Eq. 13
        # dN_microbial/dt = immobilization - microbial_death
        # Units: [s^-1] * [g N m-2] = [g N m-2 s^-1]
        death_rate = self.k_microbial_death.to(
            dtype=microbial_n.dtype,
            device=microbial_n.device,
        )
        microbial_death_rate = death_rate * microbial_n
        
        # Return ONLY rates (not deltas) - deltas will be computed in forward()
        return {
            # Primary process rates [g N m-2 s^-1]
            'gross_mineralization_rate': gross_mineralization,
            'potential_immobilization_rate': potential_immobilization,
            'microbial_death_rate': microbial_death_rate,

            # Current pool sizes (for diagnostics)
            'current_organic_n': organic_n,
            'current_nh4_n': nh4_n,
            'current_microbial_n': microbial_n
        }
    
    def forward(self, state: StateDict, time: float, dt: torch.Tensor | float, **kwargs) -> FluxDict:
        """
        Forward pass adapter that evaluates rates and formats RELICSimulator outputs.

        This method wraps _compute_rate_core with dt application, numerical stability checks,
        and conservation validation.

        Parameters:
        -----------
        state : StateDict
            Current system state with nitrogen pools
        time : float
            Current simulation time [s]
        dt : float
            Time step [s]
        **kwargs : Dict
            Not read for inputs: every declared input, carbon_decomposition_rate
            included, is read from the state (the one input channel).

        Returns:
        --------
        FluxDict
            Pool changes (deltas) and diagnostic information
        """
        # References: Manzoni and Porporato (2009); Stanford and Smith (1972);
        #   Lawrence et al. (2019) and Parton et al. (1987), organic-N input routing.
        # Mathematical expression: A_g=min(R_g*dt,N_org); A_i=min(R_i*dt,N_NH4);
        #   A_d=min(R_d*dt,N_microbe); DeltaN=[-A_g+A_d+F_org*dt,A_g-A_i,A_i-A_d].
        # Parameter meanings: A_g,A_i,A_d are donor-limited internal step transfers;
        #   F_org is the dependency-selected organic nitrogen input boundary.
        # Units: A and DeltaN [g N m-2]; R and F_org [g N m-2 s-1]; dt [s].
        # Domain/constraints: Finite required inputs, positive scalar dt, physical
        #   driver domains; initial pools limit each channel and internal columns sum to zero.
        # Sign convention: Organic input is positive once; reaction donor and recipient
        #   increments have opposite signs. ACCUMULATE pool amounts are already per_step.
        dt_tensor = torch.as_tensor(dt)
        if dt_tensor.numel() != 1 or not torch.isfinite(dt_tensor).all():
            raise ValueError("dt must be one finite positive scalar")
        if bool((dt_tensor <= 0).detach().item()):
            raise ValueError("dt must be one finite positive scalar")

        def _required_input(key: str) -> Tensor:
            # The state is the one input channel; no keyword supplies a declared input.
            value = state[key] if key in state else None
            if value is None:
                raise KeyError(f"{key} is required for nitrogen mineralization")
            if not isinstance(value, Tensor):
                raise TypeError(f"{key} must be a torch.Tensor")
            tensor = value.to(self.device)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{key} must be finite")
            return tensor

        validated_state = {
            key: _required_input(key)
            for key in self.metadata.input_variables
        }
        non_negative_inputs = (
            "organic_nitrogen",
            "nh4_nitrogen",
            "microbial_nitrogen",
            "soil_moisture",
            "soil_ph",
            "carbon_decomposition_rate",
            "organic_nitrogen_input",
        )
        for key in non_negative_inputs:
            if torch.any(validated_state[key] < 0):
                raise ValueError(f"{key} must be non-negative")
        if torch.any(validated_state["soil_temperature"] <= 0):
            raise ValueError("soil_temperature must be positive in K")
        if torch.any(validated_state["soil_moisture"] > 1):
            raise ValueError("soil_moisture must be within [0, 1]")
        if torch.any(validated_state["soil_ph"] > 14):
            raise ValueError("soil_ph must be within [0, 14]")

        reference = validated_state["organic_nitrogen"]
        dt_tensor = dt_tensor.to(
            device=reference.device,
            dtype=reference.dtype,
        ).reshape(())
        litter_nitrogen_input = validated_state["organic_nitrogen_input"].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        try:
            litter_nitrogen_input = torch.broadcast_to(
                litter_nitrogen_input,
                reference.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "organic_nitrogen_input must be scalar or broadcastable to "
                "organic_nitrogen"
            ) from exc

        # Evaluate instantaneous rates through the single production rate path.
        rates = self._compute_rate_core(validated_state)

        # Extract key rates
        gross_mineralization = rates['gross_mineralization_rate']
        potential_immobilization = rates['potential_immobilization_rate']
        microbial_death_rate = rates['microbial_death_rate']

        # Extract current pool sizes for numerical stability checks
        organic_n = rates['current_organic_n']
        nh4_n = rates['current_nh4_n']
        microbial_n = rates['current_microbial_n']

        # ======================================================================
        # NUMERICAL STABILITY: Clamp consumption to available substrate
        # ======================================================================
        # Reference: Phase 7.2.3 stability fixes from nitrification.py
        # Prevents mass creation when pools are depleted

        # 1. Gross mineralization limited by available organic N
        gross_min_theoretical = gross_mineralization * dt_tensor
        gross_min_actual = torch.clamp(gross_min_theoretical, max=organic_n)
        actual_gross_min_rate = gross_min_actual / dt_tensor

        # References: Manzoni and Porporato (2009), Sections 3.3-3.4.
        # Mathematical expression: Delta_N_immob = min(R_demand*dt, N_NH4).
        # Parameter meanings: R_demand is potential microbial N demand
        # [g N m-2 s-1], dt is the current step [s], and N_NH4 is the
        # initial ammonium donor pool [g N m-2].
        # Open-system route: none; NH4 is transferred to microbial N.
        immob_theoretical = potential_immobilization * dt_tensor
        immob_actual = torch.clamp(immob_theoretical, max=nh4_n)
        actual_immob_rate = immob_actual / dt_tensor

        # 3. Microbial death limited by available microbial N
        death_theoretical = microbial_death_rate * dt_tensor
        death_actual = torch.clamp(death_theoretical, max=microbial_n)
        actual_death_rate = death_actual / dt_tensor

        # ======================================================================
        # POOL CHANGES (deltas) with CONSERVATION
        # ======================================================================
        # Nitrogen flow:
        #   Organic N --(gross_min)--> NH4+
        #   NH4+ --(immob)--> Microbial N
        #   Microbial N --(death)--> Organic N
        # Reference: Manzoni & Porporato (2009) SBB 41:1355-1379

        # References: Manzoni & Porporato (2009); Stanford & Smith (1972); Feinberg (2019).
        # Mathematical expression: x_dot = N_h r_h for Organic N -> NH4,
        # NH4 -> Microbial N, and Microbial N -> Organic N channels.
        # Parameter meanings: N_h maps conserved nitrogen species; r_h contains
        # ParameterManager-backed gross mineralization, immobilization, and death rates.
        mineralization_network = advance_reaction_network(
            name="nitrogen_mineralization_network",
            species=("organic_nitrogen", "nh4_nitrogen", "microbial_nitrogen"),
            reactions=(
                "organic_to_nh4_gross_mineralization",
                "nh4_to_microbial_immobilization",
                "microbial_to_organic_death",
            ),
            stoichiometric_N=torch.tensor(
                [
                    [-1.0, 0.0, 1.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, -1.0],
                ],
                device=actual_gross_min_rate.device,
                dtype=actual_gross_min_rate.dtype,
            ),
            rates=(
                actual_gross_min_rate,
                actual_immob_rate,
                actual_death_rate,
            ),
            dt=dt_tensor,
            **self._iphs_reaction_conservation_kwargs(
                conservation_matrix=torch.ones(
                    (1, 3),
                    device=actual_gross_min_rate.device,
                    dtype=actual_gross_min_rate.dtype,
                ),
            ),
        )
        # Convert to actual deltas for conservation check (N_h r_h * dt)
        delta_organic_n_reaction = mineralization_network.state_delta[
            mineralization_network.species.index("organic_nitrogen")
        ]
        delta_nh4_total = mineralization_network.state_delta[
            mineralization_network.species.index("nh4_nitrogen")
        ]
        delta_microbial_total = mineralization_network.state_delta[
            mineralization_network.species.index("microbial_nitrogen")
        ]

        # ======================================================================
        # CONSERVATION CHECK
        # ======================================================================
        reaction_updates = {
            'organic_nitrogen': delta_organic_n_reaction,
            'nh4_nitrogen': delta_nh4_total,
            'microbial_nitrogen': delta_microbial_total
        }
        self._enforce_forward_conservation(state, reaction_updates)

        # References:
        # - Lawrence et al. (2019), CLM5 vegetation turnover and CN pools.
        # - Parton et al. (1987), CENTURY litter and soil organic matter routing.
        # Mathematical expression:
        # - x_dot = N_h r_h + G_h u_litter.
        # - Delta_N_org = (N_h r_h)_org * dt + F_litter,N * dt.
        # Parameter meanings:
        # - N_h is the closed three-pool reaction stoichiometry [1].
        # - r_h contains internal transformation rates [g N m-2 s-1].
        # - F_organic,N is organic_nitrogen_input [g N m-2 s-1] and can
        #   combine plant litter N with dependency-selected fecal organic N.
        # Units and route:
        # - The litter boundary enters organic N once; internal reactions remain closed.
        litter_nitrogen_step = litter_nitrogen_input * dt_tensor
        delta_organic_n_total = (
            delta_organic_n_reaction + litter_nitrogen_step
        )

        # ======================================================================
        # RETURN FORMAT
        # ======================================================================
        # References: Manzoni and Porporato (2009); Grant et al. (1993).
        # Mathematical expression: delta_N = (N_h r_h + G_h u_litter) dt and
        # R_net = R_gross - R_immobilization.
        # Parameter meanings: N_h is the closed three-pool nitrogen
        # stoichiometric matrix; r_h contains gross mineralization,
        # immobilization, and microbial death rates.
        # Units: pool increments use g N m-2; rates use g N m-2 s-1.
        # Open-system route: organic_nitrogen_input is the explicit
        # organic-material N source; urine N remains on a separate mineral route.
        output = {
            # Pool changes (per step) [g N m-2]
            'organic_nitrogen': UpdateItem(
                value=delta_organic_n_total,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),
            'nh4_nitrogen': UpdateItem(
                value=delta_nh4_total,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),
            'microbial_nitrogen': UpdateItem(
                value=delta_microbial_total,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            ),

            # Primary rates [g N m-2 s^-1]
            'mineralization_flux': UpdateItem(
                value=actual_gross_min_rate - actual_immob_rate,
                mode=UpdateMode.REPLACE,
                unit="g N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            'gross_mineralization_flux': UpdateItem(
                value=actual_gross_min_rate,
                mode=UpdateMode.REPLACE,
                unit="g N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            'immobilization_flux': UpdateItem(
                value=actual_immob_rate,
                mode=UpdateMode.REPLACE,
                unit="g N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            'microbial_death_flux': UpdateItem(
                value=actual_death_rate,
                mode=UpdateMode.REPLACE,
                unit="g N m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            ),
            'diagnostics.mineralization.organic_nitrogen_boundary_input': UpdateItem(
                value=litter_nitrogen_step,
                mode=UpdateMode.REPLACE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=False,
                source_functor=self.name,
            ),
        }

        validation_errors = validate_functor_output(output, self.get_output_specifications())
        if validation_errors:
            raise ValueError(
                f"Output validation failed for {self.name}: "
                + "; ".join(validation_errors)
            )
        return output

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Check the open litter-input plus internal three-pool nitrogen ledger."""
        # References: Manzoni and Porporato (2009); Parton et al. (1987).
        # Mathematical expression: residual=max(abs(DeltaN_org+DeltaN_NH4+DeltaN_microbe
        #   -organic_nitrogen_boundary_input)).
        # Parameter meanings: Three pool increments include reaction transfers;
        #   boundary_input is the explicit organic-N step amount returned by forward.
        # Units: all terms and residual [g N m-2].
        # Domain/constraints: Public UpdateItems are unwrapped; missing pool terms
        #   return no result. Absent boundary diagnostic is treated as zero by current code.
        # Sign convention: Positive external input is subtracted from summed pool gain.
        pool_names = (
            "organic_nitrogen",
            "nh4_nitrogen",
            "microbial_nitrogen",
        )
        if not all(name in output_updates for name in pool_names):
            return {}
        values = []
        for name in pool_names:
            value = output_updates[name]
            if isinstance(value, UpdateItem):
                value = value.value
            values.append(torch.as_tensor(value))
        boundary = output_updates.get(
            "diagnostics.mineralization.organic_nitrogen_boundary_input"
        )
        if isinstance(boundary, UpdateItem):
            boundary = boundary.value
        boundary_step = (
            torch.zeros_like(values[0])
            if boundary is None
            else torch.as_tensor(
                boundary,
                device=values[0].device,
                dtype=values[0].dtype,
            )
        )
        residual = values[0] + values[1] + values[2] - boundary_step
        magnitude = float(residual.detach().abs().max().item())
        return {law: magnitude for law in self.metadata.conservation_laws}

    def get_output_specifications(self) -> Dict[str, Any]:
        return {
            "mineralization_flux": StandardPedosphereOutputs.MINERALIZATION,
            "gross_mineralization_flux": FluxSpecification(
                name="gross_mineralization_flux",
                units="g N m-2 s-1",
                direction=FluxDirection.INTERNAL,
                flux_type=FluxType.TRANSFORMATION,
                description="Gross nitrogen mineralization rate",
                is_rate=True,
            ),
            "immobilization_flux": FluxSpecification(
                name="immobilization_flux",
                units="g N m-2 s-1",
                direction=FluxDirection.INTERNAL,
                flux_type=FluxType.TRANSFORMATION,
                description="Nitrogen immobilization rate",
                is_rate=True,
            ),
            "microbial_death_flux": FluxSpecification(
                name="microbial_death_flux",
                units="g N m-2 s-1",
                direction=FluxDirection.INTERNAL,
                flux_type=FluxType.TRANSFORMATION,
                description="Microbial nitrogen turnover rate",
                is_rate=True,
            ),
        }


__all__ = ['MineralizationProcess']
